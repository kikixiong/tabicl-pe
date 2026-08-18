from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from pe_mechanism.provenance import (
    GitEvidence,
    RunContract,
    VerifiedConfiguration,
    VerifiedNamedInput,
    VerifiedRunContext,
    VerifiedRunInputs,
    verify_file,
)
from pe_mechanism.tabarena_evaluation import (
    EvaluationSpec,
    _load_cached_results,
    _make_system_model,
    _normalize_results,
    _paired_comparison,
    _parse_config,
    _validate_checkpoint_pair,
    _validate_roster,
)


PACKAGE_ROOT = Path(__file__).parents[1]


def test_system_model_uses_same_supported_columns_for_fit_and_predict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pandas as pd

    seen: list[list[str]] = []

    class FakeClassifier:
        classes_ = np.array([0, 1])

        def __init__(self, **kwargs) -> None:
            del kwargs

        def fit(self, X, y) -> None:
            del y
            seen.append(list(X.columns))

        def predict(self, X):
            seen.append(list(X.columns))
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            seen.append(list(X.columns))
            return np.tile([0.75, 0.25], (len(X), 1))

    class ExternalSystemModel:
        def __init__(self, **kwargs) -> None:
            del kwargs

    tabicl = ModuleType("tabicl")
    monkeypatch.setitem(sys.modules, "tabicl", tabicl)
    monkeypatch.setattr(tabicl, "TabICLClassifier", FakeClassifier, raising=False)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    system = _make_system_model(ExternalSystemModel)(
        checkpoint=str(checkpoint),
        arm="rope",
        device="cuda",
        n_estimators=1,
        seed=42,
        classifier_options={},
    )
    train = pd.DataFrame(
        {
            "numeric": [1.0, 2.0],
            "category": ["a", "b"],
            "missing_in_train": [np.nan, np.nan],
            "masked_at_test": [3.0, 4.0],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    test = train.copy()
    test["missing_in_train"] = [5.0, 6.0]
    test["masked_at_test"] = np.nan
    system._fit_system(
        train,
        np.array([0, 1]),
        target_name="target",
        problem_type="binary",
        eval_metric=None,
        validation_metadata=None,
        num_cpus=1,
        num_gpus=1,
        memory_limit=None,
        time_limit=None,
        random_state=42,
    )
    system._predict(test)
    probabilities = system._predict_proba(test)

    expected = ["numeric", "category", "masked_at_test"]
    assert seen == [expected, expected, expected]
    assert probabilities.shape == (2, 2)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_example_freezes_the_complete_tabarena_contract() -> None:
    payload = json.loads(
        (PACKAGE_ROOT / "examples" / "tabarena-evaluate.example.json").read_text(
            encoding="utf-8"
        )
    )
    spec = _parse_config(payload)
    assert spec.comparison_step == 250_000
    assert spec.expected_task_count == 38
    assert spec.expected_result_count == 114
    assert spec.classifier_options["use_fa3"] is False
    assert spec.classifier_options["norm_methods"] == ["none", "power"]

    payload["study"]["surprise"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        _parse_config(payload)


def test_public_roster_is_exact_and_bound_to_benchmark_commit() -> None:
    path = PACKAGE_ROOT / "manifests" / "tabarena-v0.1-classification-roster.json"
    roster = _validate_roster(
        verify_file(path),
        benchmark_sha="c987d91556a14d4c9b3383c35d1b0ec68ff81883",
        expected_count=38,
    )
    assert len(roster) == 38
    assert roster == tuple(sorted(roster))
    assert "APSFailure" in roster


def _result(
    *, framework: str, dataset: str, task_id: int, metric: str, error: float
) -> dict:
    return {
        "experiment_metadata": {},
        "framework": framework,
        "memory_usage": {},
        "metric": metric,
        "metric_error": np.float64(error),
        "problem_type": "binary" if metric == "roc_auc" else "multiclass",
        "simulation_artifacts": None,
        "task_metadata": {
            "tid": task_id,
            "name": dataset,
            "fold": 0,
            "repeat": 0,
            "sample": 0,
            "split_idx": 0,
        },
        "time_infer_s": 0.2,
        "time_train_s": 0.1,
    }


def _small_result_matrix() -> tuple[list[dict], dict[str, str], tuple[str, ...], dict]:
    frameworks = {"rope-f": "rope", "none-f": "none", "released-f": "released"}
    roster = ("binary-task", "multiclass-task")
    expected = {
        "binary-task": {"task_id": 10, "problem_type": "binary", "metric": "roc_auc"},
        "multiclass-task": {
            "task_id": 11,
            "problem_type": "multiclass",
            "metric": "log_loss",
        },
    }
    errors = {
        "rope": (0.10, 0.20),
        "none": (0.12, 0.19),
        "released": (0.09, 0.18),
    }
    inverse = {arm: framework for framework, arm in frameworks.items()}
    results = []
    for arm in ("rope", "none", "released"):
        for offset, dataset in enumerate(roster):
            results.append(
                _result(
                    framework=inverse[arm],
                    dataset=dataset,
                    task_id=10 + offset,
                    metric="roc_auc" if offset == 0 else "log_loss",
                    error=errors[arm][offset],
                )
            )
    return results, frameworks, roster, expected


def test_result_matrix_requires_exact_official_metadata_and_finite_values() -> None:
    results, frameworks, roster, expected = _small_result_matrix()
    normalized = _normalize_results(
        results,
        framework_to_arm=frameworks,
        roster=roster,
        expected_count=6,
        expected_tasks=expected,
    )
    assert len(normalized) == 6

    broken = [dict(item) for item in results]
    broken[0] = {**broken[0], "metric_error": float("nan")}
    with pytest.raises(ValueError, match="finite"):
        _normalize_results(
            broken,
            framework_to_arm=frameworks,
            roster=roster,
            expected_count=6,
            expected_tasks=expected,
        )

    broken = [dict(item) for item in results]
    broken[0] = {**broken[0], "metric": "log_loss"}
    with pytest.raises(ValueError, match="differs across arms|official"):
        _normalize_results(
            broken,
            framework_to_arm=frameworks,
            roster=roster,
            expected_count=6,
            expected_tasks=expected,
        )


def test_paired_comparison_uses_lower_metric_error_as_better() -> None:
    results, frameworks, roster, expected = _small_result_matrix()
    normalized = _normalize_results(
        results,
        framework_to_arm=frameworks,
        roster=roster,
        expected_count=6,
        expected_tasks=expected,
    )
    rows = {(item["arm"], item["dataset"]): item for item in normalized}
    comparison = _paired_comparison(
        rows,
        roster=roster,
        left="rope",
        right="none",
        seed=42,
        n_resamples=100,
    )
    assert comparison["left_wins"] == 1
    assert comparison["right_wins"] == 1
    assert comparison["mean_metric_error_improvement"] == pytest.approx(0.005)


def test_fresh_cache_loader_accepts_plain_and_gzip_and_rejects_extras(
    tmp_path: Path,
) -> None:
    first = tmp_path / "data" / "a" / "1" / "0_0" / "results.pkl"
    second = tmp_path / "data" / "b" / "2" / "0_0" / "results.pkl"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    with first.open("wb") as handle:
        pickle.dump({"framework": "a"}, handle)
    with gzip.open(second, "wb") as handle:
        pickle.dump({"framework": "b"}, handle)
    assert len(_load_cached_results(tmp_path)) == 2

    (tmp_path / "unexpected.txt").write_text("no", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        _load_cached_results(tmp_path)


def _git_evidence(root: Path, sha: str) -> GitEvidence:
    return GitEvidence(
        head_sha=sha,
        evidence_level="strict",
        legacy_reasons=(),
        root=root,
        status_sha256="0" * 64,
    )


def test_checkpoint_pair_gate_checks_bytes_config_state_and_checksum_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = 250_000
    source_sha = "2" * 40
    common_state = {"weight": torch.zeros(2, 2)}
    none_path = tmp_path / "none-seed42-stage1-step-250000.ckpt"
    rope_path = tmp_path / "rope-seed42-stage1-step-250000.ckpt"
    released_path = tmp_path / "released.ckpt"
    base_config = {"embed_dim": 8, "row_nhead": 2}
    torch.save(
        {
            "curr_step": step,
            "config": {**base_config, "row_identity_mode": "none"},
            "state_dict": common_state,
        },
        none_path,
    )
    torch.save(
        {
            "curr_step": step,
            "config": {**base_config, "row_identity_mode": "rope"},
            "state_dict": {
                **common_state,
                "row_interactor.tf_row.rope.freqs": torch.zeros(2),
            },
        },
        rope_path,
    )
    torch.save(
        {
            "config": base_config,
            "state_dict": {"row_interactor.tf_row.rope.freqs": torch.zeros(2)},
        },
        released_path,
    )
    monkeypatch.setattr(
        "pe_mechanism.tabarena_evaluation._EXPECTED_RELEASED_SHA256",
        _digest(released_path),
    )
    monkeypatch.setattr(
        "pe_mechanism.tabarena_evaluation._EXPECTED_RELEASED_SIZE_BYTES",
        released_path.stat().st_size,
    )
    arms = {
        "none": {
            "bytes": none_path.stat().st_size,
            "curr_step": step,
            "row_identity_mode": "none",
            "sha256": _digest(none_path),
            "snapshot_checkpoint": none_path.name,
            "state_dict_tensor_count": 1,
        },
        "rope": {
            "bytes": rope_path.stat().st_size,
            "curr_step": step,
            "row_identity_mode": "rope",
            "sha256": _digest(rope_path),
            "snapshot_checkpoint": rope_path.name,
            "state_dict_tensor_count": 2,
        },
    }
    pair_path = tmp_path / "SNAPSHOT_MANIFEST.json"
    pair_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "exploratory_same_step_pilot_checkpoint_pair",
                "formal_eligible": False,
                "comparison_step": step,
                "seed": 42,
                "pilot_source_commit": source_sha,
                "arms": arms,
            }
        ),
        encoding="utf-8",
    )
    sums_path = tmp_path / "SHA256SUMS"
    sums_path.write_text(
        "\n".join(
            [
                f"{arms['none']['sha256']}  {none_path.name}",
                f"{arms['rope']['sha256']}  {rope_path.name}",
                f"{_digest(pair_path)}  {pair_path.name}",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    config_path = tmp_path / "config.json"
    roster_path = tmp_path / "roster.json"
    config_path.write_text("{}", encoding="utf-8")
    roster_path.write_text("{}", encoding="utf-8")
    git = _git_evidence(tmp_path, source_sha)
    inputs = VerifiedRunInputs(
        configuration=VerifiedConfiguration({}, verify_file(config_path)),
        checkpoint=verify_file(rope_path),
        dataset_manifest=verify_file(roster_path),
        training_code=git,
        model_code=git,
        analysis_code=git,
        additional_inputs=tuple(
            sorted(
                (
                    VerifiedNamedInput("none_checkpoint", verify_file(none_path)),
                    VerifiedNamedInput("pair_checksums", verify_file(sums_path)),
                    VerifiedNamedInput("pair_manifest", verify_file(pair_path)),
                    VerifiedNamedInput(
                        "released_checkpoint", verify_file(released_path)
                    ),
                ),
                key=lambda item: item.role,
            )
        ),
        contract=RunContract(
            command="tabarena-evaluate",
            model_family="tabicl-v2",
            model_revision="step-250000-plus-released",
            condition="tabarena-rope-none-released",
            sites=("tabarena-v0.1-classification",),
            seed=42,
        ),
        evidence_level="strict",
        legacy_reasons=(),
    )
    context = VerifiedRunContext(
        inputs=inputs,
        model_family="tabicl-v2",
        model_revision="step-250000-plus-released",
        condition="tabarena-rope-none-released",
        sites=("tabarena-v0.1-classification",),
    )
    spec = EvaluationSpec(
        comparison_step=step,
        seed=42,
        device="cuda",
        bootstrap_resamples=100,
        classifier_options={},
        pair_manifest_path=pair_path,
        pair_manifest_sha256=_digest(pair_path),
        pair_checksums_path=sums_path,
        pair_checksums_sha256=_digest(sums_path),
        none_path=none_path,
        none_sha256=_digest(none_path),
        released_path=released_path,
        released_sha256=_digest(released_path),
        tabarena_code_root=tmp_path,
        tabarena_code_sha="1" * 40,
        openml_cache_root=tmp_path,
        expected_task_count=38,
        expected_result_count=114,
    )
    digests = _validate_checkpoint_pair(context, spec)
    assert digests["rope"]["sha256"] == _digest(rope_path)
    assert digests["none"]["sha256"] == _digest(none_path)

    sums_path.write_text("tampered\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="changed"):
        _validate_checkpoint_pair(context, spec)
