from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_fingerprint_talent_exploratory.py"
SPEC = importlib.util.spec_from_file_location("fingerprint_talent", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_a10_protocol_processes_one_ensemble_member_at_a_time():
    assert MODULE.ESTIMATOR_OPTIONS["n_estimators"] == 2
    assert MODULE.ESTIMATOR_OPTIONS["batch_size"] == 1


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_discovery_roster_excludes_only_native_class_failures():
    assignments = [
        {"name": name, "split": "discovery"}
        for name in sorted(MODULE.EXCLUDED_NON_NATIVE_CLASS_COUNTS)
    ]
    assignments.extend(
        {"name": f"dataset-{index:03d}", "split": "discovery"}
        for index in range(MODULE.EXPECTED_DISCOVERY_COUNT)
    )
    assignments.append({"name": "protected", "split": "held_out"})
    roster = MODULE._discovery_roster({"assignments": assignments})
    assert len(roster) == 109
    assert "protected" not in roster
    assert not set(roster) & MODULE.EXCLUDED_NON_NATIVE_CLASS_COUNTS


def test_exact_sign_test_counts_direction_and_ties():
    result = MODULE._exact_sign_p(
        np.asarray([0.9, 0.8, 0.7, 0.6]),
        np.asarray([0.8, 0.7, 0.7, 0.9]),
        lower_is_better=False,
    )
    assert result["left_wins"] == 2
    assert result["right_wins"] == 1
    assert result["ties"] == 1
    assert result["two_sided_exact_sign_test_p"] == 1.0


def test_aggregate_primary_direction_is_fingerprint_over_rope():
    records = [
        {
            "arms": {
                "rope": {"accuracy": 0.70, "log_loss": 0.60},
                "fingerprint": {"accuracy": 0.80, "log_loss": 0.50},
                "released": {"accuracy": 0.90, "log_loss": 0.40},
            }
        },
        {
            "arms": {
                "rope": {"accuracy": 0.60, "log_loss": 0.70},
                "fingerprint": {"accuracy": 0.65, "log_loss": 0.65},
                "released": {"accuracy": 0.80, "log_loss": 0.50},
            }
        },
    ]
    result = MODULE._aggregate(records)
    primary = result["primary_matched_comparison"]

    assert result["dataset_count"] == 2
    assert result["macro_mean"]["rope"]["accuracy"] == pytest.approx(0.65)
    assert primary["left"] == "fingerprint"
    assert primary["right"] == "rope"
    assert primary["accuracy"]["left_wins"] == 2
    assert primary["accuracy"]["right_wins"] == 0
    assert primary["accuracy"]["mean_improvement_left_over_right"] == pytest.approx(
        0.075
    )
    assert primary["log_loss"]["left_wins"] == 2
    assert primary["log_loss"]["right_wins"] == 0
    assert primary["log_loss"]["mean_improvement_left_over_right"] == pytest.approx(
        0.075
    )
    assert primary["accuracy"]["inferential_role"] == "primary"
    assert primary["log_loss"]["inferential_role"] == "supportive_unadjusted"
    assert all(
        item["inferential_role"] == "descriptive_only_not_treatment_matched"
        for item in result["external_released_reference"].values()
    )


class _FakeRowTransformer:
    def __init__(self, rope: object | None) -> None:
        self.rope = rope


class _FakeRow:
    def __init__(self, identity_mode: str, *, fingerprint: bool) -> None:
        self.identity_mode = identity_mode
        self.tf_row = _FakeRowTransformer(None if fingerprint else object())
        self.fingerprint_q_gates = object() if fingerprint else None
        self.fingerprint_k_gates = object() if fingerprint else None
        self.fingerprint_q_projections = object() if fingerprint else None
        self.fingerprint_k_projections = object() if fingerprint else None


class _FakeModel:
    def __init__(self, identity_mode: str, *, fingerprint: bool) -> None:
        self.row_identity_mode = identity_mode
        self.row_fingerprint = fingerprint
        self.row_fingerprint_dim = 16
        self.row_interactor = _FakeRow(identity_mode, fingerprint=fingerprint)


class _FakeDriver:
    def __init__(self, arm: str) -> None:
        fingerprint = arm == "fingerprint"
        identity = "none" if fingerprint else "rope"
        self.model_sha = "1" * 40
        self.checkpoint_sha = "2" * 64
        self.source_evidence_level = "strict"
        self.estimator = SimpleNamespace(
            model_=_FakeModel(identity, fingerprint=fingerprint)
        )


@pytest.mark.parametrize("arm", ("rope", "fingerprint", "released"))
def test_treatment_gate_accepts_fully_aligned_strict_models(arm: str):
    observed = MODULE._assert_treatment(
        _FakeDriver(arm),
        arm,
        expected_model_sha="1" * 40,
        expected_checkpoint_sha="2" * 64,
    )
    assert observed["row_identity_mode"] == ("none" if arm == "fingerprint" else "rope")


@pytest.mark.parametrize(
    ("fault", "match"),
    (
        ("model_sha", "model SHA changed"),
        ("checkpoint_sha", "checkpoint SHA changed"),
        ("source_evidence", "lost strict source evidence"),
        ("model_identity", "inference treatment is misaligned"),
        ("row_identity", "inference treatment is misaligned"),
        ("row_fingerprint", "inference treatment is misaligned"),
        ("row_rope", "inference treatment is misaligned"),
        ("fingerprint_component", "inference treatment is misaligned"),
        ("fingerprint_dim", "dimension is not 16"),
    ),
)
def test_treatment_gate_fails_closed_on_provenance_or_model_faults(
    fault: str, match: str
):
    driver = _FakeDriver("fingerprint")
    model = driver.estimator.model_
    if fault == "model_sha":
        driver.model_sha = "3" * 40
    elif fault == "checkpoint_sha":
        driver.checkpoint_sha = "4" * 64
    elif fault == "source_evidence":
        driver.source_evidence_level = "best_effort"
    elif fault == "model_identity":
        model.row_identity_mode = "rope"
    elif fault == "row_identity":
        model.row_interactor.identity_mode = "rope"
    elif fault == "row_fingerprint":
        model.row_fingerprint = False
    elif fault == "row_rope":
        model.row_interactor.tf_row.rope = object()
    elif fault == "fingerprint_component":
        model.row_interactor.fingerprint_k_projections = None
    elif fault == "fingerprint_dim":
        model.row_fingerprint_dim = 8
    else:  # pragma: no cover - protects the table above
        raise AssertionError(fault)

    with pytest.raises(RuntimeError, match=match):
        MODULE._assert_treatment(
            driver,
            "fingerprint",
            expected_model_sha="1" * 40,
            expected_checkpoint_sha="2" * 64,
        )


def _lineage_payloads() -> dict[str, dict[str, Any]]:
    model_sha = "a" * 40
    shared_launch = {
        "schema_version": 1,
        "study": MODULE.LINEAGE_STUDY,
        "formal_evidence": False,
        "fresh_from_scratch": True,
        "max_steps": 5000,
        "seed": 42,
        "source_commit": model_sha,
        "batch_size": 64,
        "batch_size_per_gp": 8,
        "micro_batch_size": 8,
        "n_jobs": 48,
        "gpu": "NVIDIA H100 80GB HBM3",
        "architecture": {
            key: MODULE.LINEAGE_ARCHITECTURE[key]
            for key in (
                "embed_dim",
                "col_num_blocks",
                "row_num_blocks",
                "icl_num_blocks",
            )
        },
        "python": "3.11.6",
        "torch": "2.11.0",
        "cuda_build": "12.8",
    }
    payloads: dict[str, dict[str, Any]] = {
        "submission": {
            "schema_version": 1,
            "study": MODULE.LINEAGE_STUDY,
            "formal_evidence": False,
            "fresh_from_scratch": True,
            "max_steps": 5000,
            "seed": 42,
            "source_commit": model_sha,
            "jobs": {"rope": "101", "fingerprint": "102"},
        }
    }
    for arm, job_id in (("rope", "101"), ("fingerprint", "102")):
        payloads[f"{arm}_launch"] = {
            **copy.deepcopy(shared_launch),
            "arm": arm,
            "slurm_job_id": job_id,
        }
        payloads[f"{arm}_completion"] = {
            "schema_version": 1,
            "study": MODULE.LINEAGE_STUDY,
            "arm": arm,
            "formal_evidence": False,
            "curr_step": 5000,
            "source_commit": model_sha,
            "architecture": copy.deepcopy(MODULE.LINEAGE_ARCHITECTURE),
            "checkpoint": {
                "sha256": ("b" if arm == "rope" else "c") * 64,
                "size_bytes": 0,
                "state_elements": 27_000_000,
            },
            "optimizer_prefix": {"sha256": "d" * 64},
            "treatment": {
                "row_identity_mode": "rope" if arm == "rope" else "none",
                "row_fingerprint": arm == "fingerprint",
                "row_fingerprint_dim": 16,
            },
        }
    return payloads


def _write_lineage_fixture(
    root: Path, payloads: dict[str, dict[str, Any]]
) -> tuple[dict[str, Path], dict[str, Path], dict[str, str]]:
    checkpoints = {
        "rope": root / "rope.ckpt",
        "fingerprint": root / "fingerprint.ckpt",
    }
    for path in checkpoints.values():
        path.write_bytes(b"")
    receipts = {}
    for name, payload in payloads.items():
        path = root / f"{name}.json"
        _write_json(path, payload)
        receipts[name] = path
    digests = {"rope": "b" * 64, "fingerprint": "c" * 64}
    return receipts, checkpoints, digests


def test_lineage_contract_accepts_one_exact_matched_cohort(tmp_path: Path):
    receipts, checkpoints, digests = _write_lineage_fixture(
        tmp_path, _lineage_payloads()
    )
    contract = MODULE._lineage_contract(
        receipts=receipts,
        checkpoints=checkpoints,
        checkpoint_digests=digests,
        comparison_step=5000,
        model_sha="a" * 40,
    )

    assert contract["source_commit"] == "a" * 40
    assert contract["max_steps"] == 5000
    assert contract["shared_launch"]["batch_size"] == 64
    assert contract["arms"]["rope"]["treatment"]["row_identity_mode"] == "rope"
    assert contract["arms"]["fingerprint"]["treatment"]["row_fingerprint"] is True
    assert set(contract["receipt_sha256"]) == {
        "submission",
        "rope_launch",
        "rope_completion",
        "fingerprint_launch",
        "fingerprint_completion",
    }


@pytest.mark.parametrize(
    ("fault", "match"),
    (
        ("submission_job", "does not belong to the submitted cohort"),
        ("source_commit", "contract mismatch"),
        ("launch_architecture", "launch architecture is invalid"),
        ("completion_treatment", "completion treatment is invalid"),
        ("shared_runtime", "matched arms differ in launch field python"),
        ("optimizer_prefix", "matched arms differ in optimizer prefix"),
        ("checkpoint_digest", "completion checkpoint contract mismatch"),
    ),
)
def test_lineage_contract_rejects_mixed_or_mutated_cohorts(
    tmp_path: Path, fault: str, match: str
):
    payloads = _lineage_payloads()
    if fault == "submission_job":
        payloads["submission"]["jobs"]["fingerprint"] = "999"
    elif fault == "source_commit":
        payloads["rope_launch"]["source_commit"] = "e" * 40
    elif fault == "launch_architecture":
        payloads["rope_launch"]["architecture"]["embed_dim"] = 64
    elif fault == "completion_treatment":
        payloads["fingerprint_completion"]["treatment"]["row_fingerprint"] = False
    elif fault == "shared_runtime":
        payloads["fingerprint_launch"]["python"] = "3.12.0"
    elif fault == "optimizer_prefix":
        payloads["fingerprint_completion"]["optimizer_prefix"] = {"sha256": "e" * 64}
    elif fault == "checkpoint_digest":
        payloads["rope_completion"]["checkpoint"]["sha256"] = "e" * 64
    else:  # pragma: no cover - protects the table above
        raise AssertionError(fault)
    receipts, checkpoints, digests = _write_lineage_fixture(tmp_path, payloads)

    with pytest.raises(ValueError, match=match):
        MODULE._lineage_contract(
            receipts=receipts,
            checkpoints=checkpoints,
            checkpoint_digests=digests,
            comparison_step=5000,
            model_sha="a" * 40,
        )


def _cached_dataset_fixture(
    root: Path, *, run_contract_sha256: str = "f" * 64
) -> tuple[Path, Path]:
    name = "toy"
    talent_root = root / "talent"
    dataset_root = talent_root / name
    dataset_root.mkdir(parents=True)
    input_path = dataset_root / "info.json"
    input_path.write_bytes(b'{"task_type":"classification"}\n')

    destination = root / "cached"
    destination.mkdir()
    target = np.asarray([0, 1], dtype=np.int64)
    arm_probabilities = {
        "rope": np.asarray([[0.8, 0.2], [0.4, 0.6]], dtype=np.float32),
        "fingerprint": np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32),
        "released": np.asarray([[0.7, 0.3], [0.1, 0.9]], dtype=np.float32),
    }
    np.savez_compressed(
        destination / "predictions.npz", target=target, **arm_probabilities
    )
    arms = {}
    for arm, probabilities in arm_probabilities.items():
        selected = probabilities[np.arange(target.size), target]
        arms[arm] = {
            "accuracy": float(np.mean(np.argmax(probabilities, axis=1) == target)),
            "log_loss": float(
                -np.log(np.clip(selected, np.finfo(np.float32).tiny, 1.0)).mean()
            ),
            "probabilities_sha256": MODULE._array_sha256(probabilities),
        }
    result = {
        "schema_version": 1,
        "run_contract_sha256": run_contract_sha256,
        "dataset": name,
        "ordinal": 0,
        "fit_split": "train",
        "evaluation_split": "val",
        "row_sampling": "none_full_split_original_order",
        "n_evaluation": 2,
        "n_classes": 2,
        "input_sha256": {"info.json": MODULE._sha256(input_path)},
        "arms": arms,
    }
    _write_json(destination / "result.json", result)
    _write_json(
        destination / "manifest.json",
        {
            "result.json": MODULE._sha256(destination / "result.json"),
            "predictions.npz": MODULE._sha256(destination / "predictions.npz"),
        },
    )
    return destination, talent_root


def _validate_cached(destination: Path, talent_root: Path):
    return MODULE._validated_cached_record(
        destination=destination,
        name="toy",
        ordinal=0,
        run_contract_sha256="f" * 64,
        talent_root=talent_root,
    )


def test_cached_record_accepts_exact_self_consistent_artifacts(tmp_path: Path):
    destination, talent_root = _cached_dataset_fixture(tmp_path)
    record = _validate_cached(destination, talent_root)
    assert record["dataset"] == "toy"
    assert set(record["arms"]) == set(MODULE.ARMS)


def test_cached_record_rejects_non_exact_manifest_keys(tmp_path: Path):
    destination, talent_root = _cached_dataset_fixture(tmp_path)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["untracked.bin"] = "0" * 64
    _write_json(manifest_path, manifest)

    with pytest.raises(RuntimeError, match="manifest is not exact"):
        _validate_cached(destination, talent_root)


def test_cached_record_rejects_corrupted_artifact_bytes(tmp_path: Path):
    destination, talent_root = _cached_dataset_fixture(tmp_path)
    with (destination / "predictions.npz").open("ab") as handle:
        handle.write(b"corruption")

    with pytest.raises(RuntimeError, match="digest mismatch"):
        _validate_cached(destination, talent_root)


def test_cached_record_rejects_contract_transplant_with_rehashed_manifest(
    tmp_path: Path,
):
    destination, talent_root = _cached_dataset_fixture(tmp_path)
    result_path = destination / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["run_contract_sha256"] = "e" * 64
    _write_json(result_path, result)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result.json"] = MODULE._sha256(result_path)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="contract mismatch"):
        _validate_cached(destination, talent_root)


def test_cached_record_rejects_changed_talent_input_even_when_cache_is_untouched(
    tmp_path: Path,
):
    destination, talent_root = _cached_dataset_fixture(tmp_path)
    (talent_root / "toy" / "info.json").write_bytes(b"changed\n")

    with pytest.raises(RuntimeError, match="input bytes changed"):
        _validate_cached(destination, talent_root)


def _artifact_tree(root: Path, roster: tuple[str, ...]) -> Path:
    work = root / "work"
    work.mkdir()
    for filename in (
        "attempts.json",
        "environment-contract.json",
        "run-contract.json",
        "runtime.json",
        "summary.json",
    ):
        (work / filename).write_text(f"{filename}\n", encoding="utf-8")
    results = work / "datasets"
    results.mkdir()
    for ordinal, name in enumerate(roster):
        dataset = results / f"{ordinal:04d}"
        dataset.mkdir()
        for filename in MODULE.DATASET_ARTIFACTS:
            (dataset / filename).write_text(f"{name}/{filename}\n", encoding="utf-8")
    return work


def test_artifact_manifest_recursively_commits_every_expected_file(tmp_path: Path):
    roster = ("alpha", "beta")
    work = _artifact_tree(tmp_path, roster)
    manifest = MODULE._artifact_manifest(work, roster)

    artifacts = manifest["artifacts"]
    expected = {
        "attempts.json",
        "environment-contract.json",
        "run-contract.json",
        "runtime.json",
        "summary.json",
        *(
            f"datasets/{ordinal:04d}/{name}"
            for ordinal in range(2)
            for name in MODULE.DATASET_ARTIFACTS
        ),
    }
    assert set(artifacts) == expected
    for logical, facts in artifacts.items():
        path = work / logical
        assert facts == {
            "sha256": MODULE._sha256(path),
            "size_bytes": path.stat().st_size,
        }


@pytest.mark.parametrize("location", ("top", "dataset"))
def test_artifact_manifest_rejects_extra_files(tmp_path: Path, location: str):
    work = _artifact_tree(tmp_path, ("alpha",))
    if location == "top":
        (work / "debug.log").write_text("unexpected\n", encoding="utf-8")
        match = "unexpected top-level artifacts"
    else:
        (work / "datasets" / "0000" / "debug.log").write_text(
            "unexpected\n", encoding="utf-8"
        )
        match = "unexpected artifact roster"

    with pytest.raises(RuntimeError, match=match):
        MODULE._artifact_manifest(work, ("alpha",))


def test_run_lock_rejects_concurrent_owner_and_is_reusable(tmp_path: Path):
    lock_path = tmp_path / "evaluation.lock"
    with MODULE._RunLock(lock_path):
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
        assert owner["pid"] == os.getpid()
        with pytest.raises(RuntimeError, match="another evaluator owns"):
            with MODULE._RunLock(lock_path):
                pass

    with MODULE._RunLock(lock_path):
        assert json.loads(lock_path.read_text(encoding="utf-8"))["pid"] == os.getpid()


class _FakeTensor:
    def __init__(self, elements: int = 1) -> None:
        self.elements = elements

    def numel(self) -> int:
        return self.elements


def _checkpoint_payloads() -> dict[str, dict[str, Any]]:
    shared_config = {
        **MODULE.LINEAGE_ARCHITECTURE,
        "row_fingerprint_dim": 16,
        "prior": "matched",
    }
    prior_stream = {
        "cursor": 5000,
        "experiment_seed": 42,
        "ddp_rank": 0,
        "world_size": 1,
    }
    identity = {
        "schema_version": 1,
        "identity_rng_seed": 42,
        "sampler_version": None,
        "world_size": 1,
    }
    return {
        "rope": {
            "curr_step": 5000,
            "config": {
                **shared_config,
                "row_identity_mode": "rope",
                "row_fingerprint": False,
            },
            "state_dict": {
                "row_interactor.tf_row.rope.freqs": _FakeTensor(16),
                "shared.weight": _FakeTensor(8),
            },
            "identity_treatment": {**identity, "row_identity_mode": "rope"},
            "prior_stream": copy.deepcopy(prior_stream),
        },
        "fingerprint": {
            "curr_step": 5000,
            "config": {
                **shared_config,
                "row_identity_mode": "none",
                "row_fingerprint": True,
            },
            "state_dict": {
                "row_interactor.fingerprint_q_gates": _FakeTensor(3),
                "row_interactor.fingerprint_k_gates": _FakeTensor(3),
                "row_interactor.fingerprint_q_projections.0": _FakeTensor(4),
                "row_interactor.fingerprint_k_projections.0": _FakeTensor(4),
                "shared.weight": _FakeTensor(8),
            },
            "identity_treatment": {**identity, "row_identity_mode": "none"},
            "prior_stream": copy.deepcopy(prior_stream),
        },
    }


def _patch_checkpoint_load(monkeypatch: pytest.MonkeyPatch, payloads: dict[str, Any]):
    import torch

    def fake_load(path: Path, **_: object):
        return copy.deepcopy(payloads[Path(path).stem])

    monkeypatch.setattr(torch, "load", fake_load)


def test_checkpoint_contract_accepts_small_matched_fake_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payloads = _checkpoint_payloads()
    _patch_checkpoint_load(monkeypatch, payloads)
    contract = MODULE._checkpoint_contract(
        tmp_path / "rope.ckpt",
        tmp_path / "fingerprint.ckpt",
        comparison_step=5000,
    )

    assert contract["rope"]["curr_step"] == 5000
    assert contract["rope"]["model_state_elements"] == 24
    assert contract["fingerprint"]["model_state_elements"] == 22


@pytest.mark.parametrize(
    ("fault", "match"),
    (
        ("wrong_step", "wrong step"),
        ("identity_metadata", "identity treatment metadata is invalid"),
        ("prior_stream", "prior streams are not exactly matched"),
        ("architecture", "not full-size"),
        ("config_drift", "configs differ outside the treatment"),
        ("rope_state", "unexpectedly contains Fingerprint state"),
        ("fingerprint_state", "state does not encode the treatment"),
    ),
)
def test_checkpoint_contract_rejects_misaligned_or_mutated_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    match: str,
):
    payloads = _checkpoint_payloads()
    if fault == "wrong_step":
        payloads["rope"]["curr_step"] = 4999
    elif fault == "identity_metadata":
        payloads["fingerprint"]["identity_treatment"]["world_size"] = 2
    elif fault == "prior_stream":
        payloads["fingerprint"]["prior_stream"]["extra"] = "different"
    elif fault == "architecture":
        payloads["rope"]["config"]["embed_dim"] = 64
    elif fault == "config_drift":
        payloads["fingerprint"]["config"]["prior"] = "different"
    elif fault == "rope_state":
        payloads["rope"]["state_dict"]["row_interactor.fingerprint_q_gates"] = (
            _FakeTensor()
        )
    elif fault == "fingerprint_state":
        del payloads["fingerprint"]["state_dict"][
            "row_interactor.fingerprint_k_projections.0"
        ]
    else:  # pragma: no cover - protects the table above
        raise AssertionError(fault)
    _patch_checkpoint_load(monkeypatch, payloads)

    with pytest.raises(ValueError, match=match):
        MODULE._checkpoint_contract(
            tmp_path / "rope.ckpt",
            tmp_path / "fingerprint.ckpt",
            comparison_step=5000,
        )
