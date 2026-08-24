from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
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
