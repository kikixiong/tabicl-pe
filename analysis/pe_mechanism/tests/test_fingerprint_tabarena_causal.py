from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import tarfile
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from pe_mechanism.fingerprint_causal import (
    FINGERPRINT_INTERVENTIONS,
    FingerprintForwardPreHook,
    _prediction_evidence,
    aggregate_causal_results,
    capture_causal_prediction,
    cyclic_derangement_indices,
    fingerprint_checkpoint_contract,
    load_and_validate_captures,
    make_fingerprint_causal_system,
)
from pe_mechanism.fingerprint_causal_verification import (
    _safe_extract_results,
    verify_fingerprint_causal_run,
)
from pe_mechanism.tabarena_evaluation import (
    _FIXED_CLASSIFIER_OPTIONS,
    _archive_directory,
)


FULLSIZE = {
    "embed_dim": 128,
    "col_num_blocks": 3,
    "col_nhead": 8,
    "col_num_inds": 128,
    "row_num_blocks": 3,
    "row_nhead": 8,
    "icl_num_blocks": 12,
    "icl_nhead": 8,
}
PACKAGE_ROOT = Path(__file__).parents[1]
WRAPPER = PACKAGE_ROOT / "scripts" / "slurm_fingerprint_tabarena_causal.sh"
RUNNER = PACKAGE_ROOT / "scripts" / "run_fingerprint_tabarena_causal.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_fingerprint_tabarena_causal", RUNNER
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER_MODULE = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER_MODULE)
VERIFIER = PACKAGE_ROOT / "scripts" / "verify_fingerprint_tabarena_causal.py"
VERIFIER_SPEC = importlib.util.spec_from_file_location(
    "verify_fingerprint_tabarena_causal", VERIFIER
)
assert VERIFIER_SPEC is not None and VERIFIER_SPEC.loader is not None
VERIFIER_MODULE = importlib.util.module_from_spec(VERIFIER_SPEC)
VERIFIER_SPEC.loader.exec_module(VERIFIER_MODULE)


class _HookTarget(torch.nn.Module):
    row_fingerprint = True
    row_identity_mode = "none"

    def _num_row_identity_tokens(self, num_input_features: int) -> int:
        return num_input_features

    def forward(self, X: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        return {"X": X, **kwargs}


def _checkpoint(path: Path, *, step: int = 50_000) -> None:
    torch.save(
        {
            "curr_step": step,
            "config": {
                **FULLSIZE,
                "row_identity_mode": "none",
                "row_fingerprint": True,
                "row_fingerprint_dim": 16,
                "col_feature_group": "same",
                "col_target_aware": True,
            },
            "state_dict": {"weight": torch.zeros(2, 3)},
            "prior_stream": {
                "cursor": step,
                "experiment_seed": 42,
                "ddp_rank": 0,
                "world_size": 1,
            },
        },
        path,
    )


def test_cyclic_derangement_is_fixed_and_rng_free() -> None:
    before = torch.random.get_rng_state().clone()
    indices, metadata = cyclic_derangement_indices(4)

    assert indices == (1, 2, 3, 0)
    assert metadata["fixed_point_count"] == 0
    assert metadata["effective"] is True
    assert metadata["degenerate_reason"] is None
    assert torch.equal(torch.random.get_rng_state(), before)


def test_single_token_permutation_is_explicitly_degenerate() -> None:
    indices, metadata = cyclic_derangement_indices(1)

    assert indices == (0,)
    assert metadata["fixed_point_count"] == 1
    assert metadata["effective"] is False
    assert metadata["degenerate_reason"] == "single_feature_token_has_no_derangement"


@pytest.mark.parametrize("intervention", FINGERPRINT_INTERVENTIONS)
def test_forward_hook_injects_each_condition_and_is_removable(
    intervention: str,
) -> None:
    model = _HookTarget()
    hook = FingerprintForwardPreHook(intervention)
    handle = model.register_forward_pre_hook(hook, with_kwargs=True)
    X = torch.zeros(2, 3, 4)

    observed = model(X=X)
    assert observed["fingerprint_intervention"] == intervention
    if intervention == "permuted":
        assert observed["fingerprint_permutation"].tolist() == [
            [1, 2, 3, 0],
            [1, 2, 3, 0],
        ]
    else:
        assert "fingerprint_permutation" not in observed
    assert hook.metadata()["forward_call_count"] == 1

    handle.remove()
    assert model(X=X) == {"X": X}


class _BaseSystem:
    def __init__(self, *, classifier_options: dict[str, Any], **kwargs: Any) -> None:
        del kwargs
        self.classifier_options = classifier_options
        self.model = None

    def _fit_system(self, X: Any, y: Any, **kwargs: Any) -> "_BaseSystem":
        del X, y, kwargs
        self.model = SimpleNamespace(
            model_=_HookTarget(),
            model_kv_cache_=None,
        )
        return self

    def cleanup(self) -> None:
        self.model = None


def _base_factory(_: type) -> type:
    return _BaseSystem


def test_causal_system_rejects_kv_cache_and_cleans_hook() -> None:
    system_cls = make_fingerprint_causal_system(object, _base_factory)
    with pytest.raises(ValueError, match="kv_cache=False"):
        system_cls(intervention="zero", classifier_options={"kv_cache": True})

    system = system_cls(
        intervention="collapsed", classifier_options={"kv_cache": False}
    )
    system._fit_system(None, None)
    raw_model = system.model.model_
    assert len(raw_model._forward_pre_hooks) == 1
    assert raw_model(X=torch.zeros(1, 2, 3))["fingerprint_intervention"] == "collapsed"
    extra = raw_model.register_forward_pre_hook(
        lambda _module, args, kwargs: (args, kwargs), with_kwargs=True
    )
    with pytest.raises(RuntimeError, match="removed or duplicated"):
        system.causal_metadata()
    extra.remove()
    assert system.causal_metadata()["intervention"] == "collapsed"
    system.cleanup()
    assert len(raw_model._forward_pre_hooks) == 0
    assert "fingerprint_intervention" not in raw_model(X=torch.zeros(1, 2, 3))


def test_fingerprint_checkpoint_contract_accepts_only_step50k_treatment(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "fingerprint.ckpt"
    _checkpoint(checkpoint)
    contract = fingerprint_checkpoint_contract(checkpoint)
    assert contract["curr_step"] == 50_000
    assert contract["model_state_elements"] == 6

    wrong_step = tmp_path / "wrong-step.ckpt"
    _checkpoint(wrong_step, step=49_999)
    with pytest.raises(ValueError, match="not at step 50000"):
        fingerprint_checkpoint_contract(wrong_step)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["config"]["row_fingerprint"] = False
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="treatment is invalid"):
        fingerprint_checkpoint_contract(checkpoint)


def _capture_grid(root: Path) -> tuple[str, ...]:
    roster = ("toy",)
    train_X = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]}, index=[10, 11])
    train_y = pd.Series([0, 1], index=train_X.index, name="target")
    test_X = pd.DataFrame({"a": [0.2, 0.8], "b": [0.8, 0.2]}, index=[20, 21])
    test_y = pd.Series([0, 1], index=test_X.index, name="target")
    for intervention in FINGERPRINT_INTERVENTIONS:
        probabilities = pd.DataFrame(
            [[0.75, 0.25], [0.1, 0.9]],
            index=test_X.index,
            columns=[0, 1],
        )
        permutation = None
        if intervention == "permuted":
            _, permutation = cyclic_derangement_indices(2)
        capture_causal_prediction(
            probabilities,
            train_features=train_X,
            train_targets=train_y,
            test_features=test_X,
            test_targets=test_y,
            dataset="toy",
            intervention=intervention,
            causal_metadata={
                "intervention": intervention,
                "forward_call_count": 1,
                "feature_token_count": 2,
                "permutation": permutation,
            },
            capture_root=root,
        )
    return roster


def _synthetic_causal_run(tmp_path: Path) -> dict[str, Path | str]:
    run = tmp_path / "run"
    run.mkdir()
    capture_root = run / "private_predictions"
    roster = _capture_grid(capture_root)
    captures = load_and_validate_captures(capture_root, roster=roster)

    checkpoint = tmp_path / "fingerprint.ckpt"
    _checkpoint(checkpoint)
    checkpoint_sha = RUNNER_MODULE._sha256(checkpoint)
    analysis_sha = "a" * 40
    model_sha = "b" * 40
    tabarena_sha = "c" * 40
    roster_path = tmp_path / "roster.json"
    roster_path.write_text(
        json.dumps(
            {
                "count": 1,
                "names": ["toy"],
                "source_commit": tabarena_sha,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    roster_sha = RUNNER_MODULE._sha256(roster_path)

    errors = {
        "correct": 0.2,
        "zero": 0.3,
        "permuted": 0.4,
        "collapsed": 0.1,
    }
    normalized = []
    results_root = tmp_path / "results"
    for index, intervention in enumerate(FINGERPRINT_INTERVENTIONS):
        framework = (
            "TabICL_Fullsize_Fingerprint_Step50000_Causal_"
            + intervention.capitalize()
            + "_c1_default"
        )
        raw = {
            "experiment_metadata": {},
            "framework": framework,
            "memory_usage": None,
            "metric": "roc_auc",
            "metric_error": errors[intervention],
            "problem_type": "binary",
            "simulation_artifacts": None,
            "task_metadata": {
                "tid": 7,
                "name": "toy",
                "fold": 0,
                "repeat": 0,
                "sample": 0,
                "split_idx": 0,
            },
            "time_infer_s": 0.2 + index,
            "time_train_s": 0.1 + index,
        }
        destination = results_root / intervention / "results.pkl"
        destination.parent.mkdir(parents=True)
        with destination.open("wb") as handle:
            pickle.dump(raw, handle)
        normalized.append(
            {
                "arm": intervention,
                "framework": framework,
                "dataset": "toy",
                "task_id": 7,
                "fold": 0,
                "repeat": 0,
                "sample": 0,
                "split_idx": 0,
                "problem_type": "binary",
                "metric": "roc_auc",
                "metric_error": errors[intervention],
                "time_train_s": 0.1 + index,
                "time_infer_s": 0.2 + index,
            }
        )
    _archive_directory(results_root, run / "results.tar.gz")
    rows = {(row["arm"], row["dataset"]): row for row in normalized}
    comparisons = aggregate_causal_results(
        rows,
        roster=roster,
        n_resamples=10_000,
        seed=42,
    )
    comparison_sha = hashlib.sha256(
        json.dumps(
            comparisons,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    summary = {
        "schema_version": 1,
        "study": "fullsize-fingerprint-step50000-tabarena-causal",
        "formal_eligible": False,
        "leaderboard_replication": False,
        "seed": 42,
        "comparison_step": 50_000,
        "task_subset": "lite",
        "task_count": 1,
        "result_count": 4,
        "condition_order": list(FINGERPRINT_INTERVENTIONS),
        **comparisons,
        "permutation_audit": {
            "algorithm": "cyclic_shift_left_one_v1",
            "h_greater_than_one_has_no_fixed_points": True,
            "single_token_degenerate_dataset_count": 0,
            "single_token_degenerate_datasets": [],
        },
        "datasets": [
            {
                "dataset": "toy",
                "task_id": 7,
                "problem_type": "binary",
                "metric": "roc_auc",
                "feature_token_count": 2,
                "conditions": {
                    intervention: {
                        "metric_error": errors[intervention],
                        "time_train_s": 0.1 + index,
                        "time_infer_s": 0.2 + index,
                    }
                    for index, intervention in enumerate(FINGERPRINT_INTERVENTIONS)
                },
            }
        ],
        "checkpoint": {
            "sha256": checkpoint_sha,
            "size_bytes": checkpoint.stat().st_size,
            "contract": fingerprint_checkpoint_contract(checkpoint),
        },
        "code_provenance": {
            "analysis_sha": analysis_sha,
            "model_sha": model_sha,
            "tabarena_sha": tabarena_sha,
            "roster_file_sha256": roster_sha,
        },
        "inference_budget": {
            "n_estimators": 1,
            "augmentation": "none",
            "classifier_options": dict(_FIXED_CLASSIFIER_OPTIONS),
        },
        "prediction_capture": {
            "private": True,
            "dtype": "float32",
            "four_way_alignment_fields": [
                "shape",
                "encoded_target",
                "row",
                "test_target",
                "class",
                "train_content",
                "test_content",
            ],
        },
    }
    runtime = {
        "schema_version": 1,
        "python": "3.10.0",
        "numpy": "2.0.0",
        "pandas": "2.0.0",
        "scikit_learn": "1.6.0",
        "openml": "0.15.0",
        "autogluon_core": "1.4.0",
        "torch": "2.5.0",
        "tabarena": "0.1.0",
        "tabicl": "2.0.0",
        "benchmark_code_sha": tabarena_sha,
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_device_name": "NVIDIA A10",
        "cuda_device_capability": [8, 6],
        "cuda_runtime": "12.1",
        "cudnn": 8900,
        "nvidia_driver": "535.0",
        "duration_seconds": 1.0,
        "result_count": 4,
    }
    captures_manifest = RUNNER_MODULE._capture_manifest(
        captures,
        root=capture_root,
        interventions=FINGERPRINT_INTERVENTIONS,
        roster=roster,
    )
    for name, payload in (
        ("summary.json", summary),
        ("runtime.json", runtime),
        ("captures_manifest.json", captures_manifest),
    ):
        (run / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    artifacts = {
        name: {
            "sha256": RUNNER_MODULE._sha256(run / name),
            "size_bytes": (run / name).stat().st_size,
        }
        for name in (
            "summary.json",
            "runtime.json",
            "results.tar.gz",
            "captures_manifest.json",
        )
    }
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "study": "fullsize-fingerprint-step50000-tabarena-causal",
                "formal_eligible": False,
                "contains_private_predictions": True,
                "artifacts": artifacts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return {
        "run": run,
        "results": results_root,
        "checkpoint": checkpoint,
        "checkpoint_sha": checkpoint_sha,
        "roster": roster_path,
        "roster_sha": roster_sha,
        "scratch": scratch,
        "analysis_sha": analysis_sha,
        "model_sha": model_sha,
        "tabarena_sha": tabarena_sha,
        "comparison_sha": comparison_sha,
    }


def test_private_float32_capture_grid_aligns_and_detects_target_mismatch(
    tmp_path: Path,
) -> None:
    roster = _capture_grid(tmp_path)
    captures = load_and_validate_captures(tmp_path, roster=roster)
    assert captures["correct"]["toy"]["prediction_evidence"][
        "probability_dtype"
    ] == "<f4"

    digest = captures["zero"]["toy"]["dataset_sha256"]
    directory = tmp_path / "zero" / digest
    prediction = directory / "predictions.npz"
    with np.load(prediction, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["test_target_fingerprints"] = arrays["test_target_fingerprints"].copy()
    arrays["test_target_fingerprints"][0, 0] ^= np.uint8(1)
    with prediction.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    record_path = directory / "capture.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["prediction_evidence"] = _prediction_evidence(prediction)
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="test_target_sha256"):
        load_and_validate_captures(tmp_path, roster=roster)


def test_capture_validation_recomputes_permutation_semantics(tmp_path: Path) -> None:
    roster = _capture_grid(tmp_path)
    digest = hashlib.sha256(b"toy").hexdigest()
    record_path = tmp_path / "permuted" / digest / "capture.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["causal_metadata"]["permutation"]["algorithm"] = "forged"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="permutation evidence"):
        load_and_validate_captures(tmp_path, roster=roster)


def test_postpublication_verifier_rehashes_and_recomputes_complete_run(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_causal_run(tmp_path)

    report = verify_fingerprint_causal_run(
        fixture["run"],
        checkpoint=fixture["checkpoint"],
        roster_path=fixture["roster"],
        scratch_root=fixture["scratch"],
        expected_checkpoint_sha256=fixture["checkpoint_sha"],
        expected_analysis_sha=fixture["analysis_sha"],
        expected_model_sha=fixture["model_sha"],
        expected_tabarena_sha=fixture["tabarena_sha"],
        expected_roster_sha256=fixture["roster_sha"],
        expected_task_count=1,
    )

    assert report["status"] == "verified"
    assert report["task_count"] == 1
    assert report["result_count"] == 4
    assert report["capture_record_count"] == 4
    assert report["capture_file_count"] == 8
    assert report["bootstrap_resamples"] == 10_000
    assert report["recomputed_comparisons_sha256"] == fixture["comparison_sha"]
    assert not list(fixture["scratch"].iterdir())

    capture = next((fixture["run"] / "private_predictions").rglob("capture.json"))
    capture.write_bytes(capture.read_bytes() + b" ")
    with pytest.raises(ValueError, match="expected SHA-256"):
        verify_fingerprint_causal_run(
            fixture["run"],
            checkpoint=fixture["checkpoint"],
            roster_path=fixture["roster"],
            scratch_root=fixture["scratch"],
            expected_checkpoint_sha256=fixture["checkpoint_sha"],
            expected_analysis_sha=fixture["analysis_sha"],
            expected_model_sha=fixture["model_sha"],
            expected_tabarena_sha=fixture["tabarena_sha"],
            expected_roster_sha256=fixture["roster_sha"],
            expected_task_count=1,
        )


def test_postpublication_verifier_rejects_unsafe_results_archive(tmp_path: Path) -> None:
    payload = b"unsafe"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("tabarena-results/../escape/results.pkl")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="unsafe member"):
        _safe_extract_results(buffer.getvalue(), tmp_path)


def test_postpublication_verifier_rejects_overlaps_and_repacked_metric(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_causal_run(tmp_path)
    run = Path(fixture["run"])
    common = {
        "checkpoint": fixture["checkpoint"],
        "roster_path": fixture["roster"],
        "expected_checkpoint_sha256": fixture["checkpoint_sha"],
        "expected_analysis_sha": fixture["analysis_sha"],
        "expected_model_sha": fixture["model_sha"],
        "expected_tabarena_sha": fixture["tabarena_sha"],
        "expected_roster_sha256": fixture["roster_sha"],
        "expected_task_count": 1,
    }
    with pytest.raises(ValueError, match="scratch_root"):
        verify_fingerprint_causal_run(
            run,
            scratch_root=run / "private_predictions",
            **common,
        )
    with pytest.raises(ValueError, match="must not overlap"):
        VERIFIER_MODULE._receipt_path(
            str(run / "verification.json"),
            run_dir=run,
        )
    with pytest.raises(ValueError, match="verifier repository"):
        VERIFIER_MODULE._receipt_path(
            str(PACKAGE_ROOT / "verification-receipt-never-create.json"),
            run_dir=run,
        )

    result_path = Path(fixture["results"]) / "correct" / "results.pkl"
    with result_path.open("rb") as handle:
        result = pickle.load(handle)
    result["metric_error"] = 0.9
    with result_path.open("wb") as handle:
        pickle.dump(result, handle)
    archive_path = run / "results.tar.gz"
    _archive_directory(Path(fixture["results"]), archive_path)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["results.tar.gz"] = {
        "sha256": RUNNER_MODULE._sha256(archive_path),
        "size_bytes": archive_path.stat().st_size,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="differs from the causal summary"):
        verify_fingerprint_causal_run(
            run,
            scratch_root=fixture["scratch"],
            **common,
        )


def test_verifier_git_provenance_requires_clean_detached_exact_head(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
        )

    git("init", "-q")
    tracked = repository / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    git("add", "tracked.txt")
    git(
        "-c",
        "user.name=Verifier Test",
        "-c",
        "user.email=verifier@example.invalid",
        "commit",
        "-q",
        "-m",
        "fixture",
    )
    head = git("rev-parse", "HEAD").stdout.strip()
    with pytest.raises(ValueError, match="detached HEAD"):
        VERIFIER_MODULE._git_head(repository, head)
    git("checkout", "--detach", "-q", head)

    assert VERIFIER_MODULE._git_head(repository, head) == head
    with pytest.raises(ValueError, match="HEAD mismatch"):
        VERIFIER_MODULE._git_head(repository, "0" * 40)
    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be clean"):
        VERIFIER_MODULE._git_head(repository, head)


def test_verifier_cli_writes_private_code_bound_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    receipt_parent = tmp_path / "receipts"
    receipt_parent.mkdir()
    receipt = receipt_parent / "verification.json"
    verifier_sha = "d" * 40

    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setattr(
        VERIFIER_MODULE,
        "_git_head",
        lambda _root, expected: expected,
    )
    monkeypatch.setattr(
        VERIFIER_MODULE,
        "verify_fingerprint_causal_run",
        lambda *_args, **_kwargs: {
            "status": "verified",
            "task_count": 1,
            "result_count": 4,
        },
    )
    monkeypatch.setattr(
        VERIFIER_MODULE.sys,
        "argv",
        [
            str(VERIFIER),
            "--run-dir",
            str(run),
            "--fingerprint-checkpoint",
            str(tmp_path / "checkpoint.ckpt"),
            "--receipt",
            str(receipt),
            "--expected-verifier-sha",
            verifier_sha,
        ],
    )

    assert VERIFIER_MODULE.main() == 0
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["verifier_provenance"]["git_sha"] == verifier_sha
    for name in (
        "script_sha256",
        "verification_module_sha256",
        "capture_module_sha256",
    ):
        assert len(payload["verifier_provenance"][name]) == 64
    assert payload["verifier_runtime"]["python"]
    assert payload["verifier_runtime"]["numpy"] == np.__version__
    assert receipt.stat().st_mode & 0o777 == 0o600


def test_causal_aggregation_keeps_metrics_separate_and_delta_direction() -> None:
    roster = ("a", "b", "c")
    metrics = {"a": "log_loss", "b": "log_loss", "c": "roc_auc"}
    errors = {
        "correct": (0.2, 0.3, 0.1),
        "zero": (0.4, 0.1, 0.2),
        "permuted": (0.5, 0.4, 0.2),
        "collapsed": (0.2, 0.3, 0.1),
    }
    rows = {
        (intervention, dataset): {
            "metric": metrics[dataset],
            "metric_error": errors[intervention][index],
        }
        for intervention in FINGERPRINT_INTERVENTIONS
        for index, dataset in enumerate(roster)
    }

    summary = aggregate_causal_results(rows, roster=roster, n_resamples=100)
    assert set(summary["metric_groups"]) == {"log_loss", "roc_auc"}
    assert "mean_raw_metric_error_delta" not in summary[
        "overall_scale_free_correct_vs_interventions"
    ][0]
    zero = summary["overall_scale_free_correct_vs_interventions"][0]
    assert (zero["left_wins"], zero["right_wins"], zero["ties"]) == (2, 1, 0)
    permuted_log_loss = summary["metric_groups"]["log_loss"][
        "raw_metric_error_delta_correct_vs_interventions"
    ][1]
    assert permuted_log_loss["mean_raw_metric_error_delta"] == pytest.approx(0.2)
    assert permuted_log_loss["positive_means_correct_is_better"] is True


def test_slurm_wrapper_resources_tmpdir_and_invocation(tmp_path: Path) -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    for directive in (
        "#SBATCH --partition=normal",
        "#SBATCH --qos=short",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=16",
        "#SBATCH --mem=64G",
        "#SBATCH --time=03:00:00",
    ):
        assert directive in source
    assert 'scratch_parent="${TMPDIR:?' in source
    assert "scratch_parent=/" not in source

    syntax = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    capture = tmp_path / "invocation.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "[[ -d \"$TMPDIR\" ]]\n"
        "printf 'TMPDIR=%s\\n' \"$TMPDIR\" > \"$WRAPPER_CAPTURE\"\n"
        "printf 'ARG=%s\\n' \"$@\" >> \"$WRAPPER_CAPTURE\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "TMPDIR": str(scratch),
        "SLURM_JOB_ID": "12345",
        "WRAPPER_CAPTURE": str(capture),
        "PE_ANALYSIS_ROOT": "/analysis",
        "PE_MODEL_ROOT": "/model",
        "PE_TABARENA_ROOT": "/tabarena",
        "PE_OPENML_CACHE": "/cache",
        "PE_FINGERPRINT_CHECKPOINT": "/checkpoint.ckpt",
        "PE_FINGERPRINT_SHA256": "a" * 64,
        "PE_OUTPUT_DIR": "/output",
        "PE_PYTHON": str(fake_python),
        "PE_EXPECTED_ANALYSIS_SHA": "b" * 40,
        "PE_EXPECTED_MODEL_SHA": "c" * 40,
        "PE_EXPECTED_TABARENA_SHA": "d" * 40,
        "PE_DATASET": "blood-transfusion-service-center",
    }
    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    lines = capture.read_text(encoding="utf-8").splitlines()
    job_tmp = scratch / "fingerprint-causal-12345"
    assert lines[0] == f"TMPDIR={job_tmp}"
    assert "ARG=--dataset" in lines
    assert "ARG=blood-transfusion-service-center" in lines
    assert not job_tmp.exists()


def test_failed_run_retains_partial_evidence_and_leaves_output_fresh(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    staging = tmp_path / ".run.tmp-example"
    staging.mkdir()
    partial = staging / "private_predictions" / "partial.npz"
    partial.parent.mkdir()
    partial.write_bytes(b"partial evidence")
    output = tmp_path / "run"

    RUNNER_MODULE._retain_failed_run(staging, output, RuntimeError("gpu oom"))

    assert partial.read_bytes() == b"partial evidence"
    assert not output.exists()
    failure = json.loads((staging / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["error_type"] == "RuntimeError"
    assert failure["retained_staging_dir"] == str(staging)
    assert str(staging) in capsys.readouterr().err
