from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pe_mechanism.talent_full_suite import (
    ARM_DATASET_ARTIFACTS,
    DATASET_ARTIFACTS,
    FULLSIZE_ARCHITECTURE,
    PLAN_KIND,
    RUN_CONFIG_STUDY,
    array_sha256,
    atomic_json,
    build_shard_plan_payload,
    canary_dataset_records,
    directory_manifest,
    frozen_discovery_roster,
    load_private_run_config,
    require_disjoint_output,
    self_hashed_document,
    sha256_file,
    validate_checkpoint_pairs,
    validate_shard_plan,
)


ROOT = Path(__file__).parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = _script("run_talent_paired_full_suite.py")
AGGREGATOR = _script("aggregate_talent_paired_full_suite.py")
SUBMITTER = _script("submit_talent_paired_full_suite.py")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _model_config(kind: str) -> dict[str, object]:
    return {
        **FULLSIZE_ARCHITECTURE,
        "row_identity_mode": "rope" if kind == "rope" else "none",
        "row_fingerprint": False,
        "row_fingerprint_dim": 16,
    }


def _legacy_config(
    tmp_path: Path, *, step: int = 250_000, future: bool = False
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    arms = []
    receipt_arms = {}
    receipt_specs = []
    for kind in ("rope", "none"):
        state = {"base.weight": torch.ones(2, 2)}
        if kind == "rope":
            state["row_interactor.tf_row.rope.freqs"] = torch.ones(2)
        checkpoint = tmp_path / f"{kind}-{step}.ckpt"
        torch.save(
            {
                "config": _model_config(kind),
                "state_dict": state,
                "optimizer_state": {},
                "scheduler_state": {},
                "curr_step": step,
            },
            checkpoint,
        )
        digest = sha256_file(checkpoint)
        arms.append(
            {
                "arm_id": f"{kind}-{step}",
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": digest,
                "treatment": {"kind": kind},
            }
        )
        if future:
            body = {
                "schema_version": 1,
                "kind": "tabicl-legacy-pilot-checkpoint-snapshot",
                "classification": "exploratory-pilot-only",
                "continuation_id": "unit-test-continuation-v1",
                "created_at_utc": "2026-08-17T00:00:00+00:00",
                "mode": kind,
                "stage": "stage1",
                "step": step,
                "continuation_source_commit": "2" * 40,
                "source_provenance_status": (
                    "operational-history-only-not-checkpoint-bound"
                ),
                "source_checkpoint": str(checkpoint),
                "snapshot_filename": checkpoint.name,
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": digest,
                "limitations": [
                    "checkpoint_has_no_source_sha",
                    "checkpoint_has_no_rng_or_dataloader_state",
                    "checkpoint_has_no_grad_scaler_state",
                    "checkpoint_has_no_parent_lineage",
                ],
            }
            encoded = json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            receipt = {
                **body,
                "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
            }
            receipt_path = tmp_path / f"receipt-{kind}-{step}.json"
            _write_json(receipt_path, receipt)
            receipt_specs.append(
                {"path": str(receipt_path), "sha256": sha256_file(receipt_path)}
            )
        else:
            receipt_arms[kind] = {
                "bytes": checkpoint.stat().st_size,
                "curr_step": step,
                "row_identity_mode": kind,
                "sha256": digest,
                "snapshot_checkpoint": checkpoint.name,
                "source_checkpoint": str(checkpoint),
                "source_job_id": "123",
                "state_dict_tensor_count": len(state),
            }
    source = "2" * 40
    if not future:
        receipt = {
            "schema_version": 1,
            "kind": "exploratory_same_step_pilot_checkpoint_pair",
            "formal_eligible": False,
            "comparison_step": step,
            "seed": 42,
            "pilot_source_commit": source,
            "captured_at_utc": "2026-08-17T00:00:00Z",
            "notes": "exploratory",
            "arms": receipt_arms,
        }
        receipt_path = tmp_path / f"receipt-{step}.json"
        _write_json(receipt_path, receipt)
        receipt_specs.append(
            {"path": str(receipt_path), "sha256": sha256_file(receipt_path)}
        )
    config = {
        "schema_version": 1,
        "study": RUN_CONFIG_STUDY,
        "seed": 42,
        "pairs": [
            {
                "pair_id": f"legacy-{step}",
                "comparison_step": step,
                "training_source_commit": source,
                "match_level": "same_step_legacy",
                "arms": arms,
                "lineage_receipts": receipt_specs,
            }
        ],
    }
    config_path = tmp_path / f"config-{step}.json"
    _write_json(config_path, config)
    return config_path


def _records() -> list[dict[str, object]]:
    names = ["small", "madeline", "large", "largest"]
    rows = [10, 20, 30, 40]
    records = []
    for ordinal, (name, n_train) in enumerate(zip(names, rows, strict=True)):
        info = f"{ordinal + 1:064x}"
        records.append(
            {
                "ordinal": ordinal,
                "name": name,
                "n_train": n_train,
                "n_validation": 5,
                "n_test": 5,
                "n_features": 50 + ordinal * 30,
                "n_classes": 2,
                "info_sha256": info,
                "input_sha256": {"info.json": info},
            }
        )
    return records


def test_frozen_roster_is_exact_native_discovery_109():
    roster = frozen_discovery_roster(ROOT.parents[1])
    assert len(roster) == 109
    assert len(set(roster)) == 109
    assert "texture" not in roster


def test_plan_is_deterministic_lpt_and_tamper_fails():
    payload = build_shard_plan_payload(_records(), shard_count=2)
    payload["analysis_sha"] = "a" * 40
    document = self_hashed_document(PLAN_KIND, payload)
    assert validate_shard_plan(document, expected_roster=[r["name"] for r in _records()])
    tampered = json.loads(json.dumps(payload))
    tampered["shards"].reverse()
    with pytest.raises(ValueError, match="shard IDs|deterministic"):
        validate_shard_plan(
            self_hashed_document(PLAN_KIND, tampered),
            expected_roster=[r["name"] for r in _records()],
        )


def test_canary_is_two_largest_plus_madeline():
    plan = build_shard_plan_payload(_records(), shard_count=2)
    assert {record["name"] for record in canary_dataset_records(plan)} == {
        "large",
        "largest",
        "madeline",
    }


def test_legacy_snapshot_and_future_continuation_snapshot_validate(tmp_path: Path):
    for config_path in (
        _legacy_config(tmp_path / "legacy"),
        _legacy_config(tmp_path / "future", step=500_000, future=True),
    ):
        config = load_private_run_config(config_path)
        contract = validate_checkpoint_pairs(config)
        assert contract["pairs"][0]["prior_stream_sha256"] is None
        assert config["pairs"][0]["lineage_contract"]["formal_eligible"] is False


def test_future_snapshot_pair_rejects_different_continuation_ids(tmp_path: Path):
    config_path = _legacy_config(tmp_path, step=500_000, future=True)
    config_document = json.loads(config_path.read_text(encoding="utf-8"))
    receipt_spec = config_document["pairs"][0]["lineage_receipts"][1]
    receipt_path = Path(receipt_spec["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["continuation_id"] = "different-continuation-v1"
    body = {key: value for key, value in receipt.items() if key != "manifest_sha256"}
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    receipt["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    _write_json(receipt_path, receipt)
    receipt_spec["sha256"] = sha256_file(receipt_path)
    _write_json(config_path, config_document)
    with pytest.raises(ValueError, match="pair lineage differs"):
        load_private_run_config(config_path)


def test_seed_is_fixed_to_42(tmp_path: Path):
    config_path = _legacy_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["seed"] = 43
    _write_json(config_path, payload)
    with pytest.raises(ValueError, match="header"):
        load_private_run_config(config_path)


def test_partial_fingerprint_residue_is_rejected(tmp_path: Path):
    config_path = _legacy_config(tmp_path)
    config = load_private_run_config(config_path)
    none = config["pairs"][0]["arms"][1]["checkpoint"]
    payload = torch.load(none, map_location="cpu", weights_only=True)
    payload["state_dict"]["row_interactor.fingerprint_q_gates"] = torch.ones(3)
    torch.save(payload, none)
    config["pairs"][0]["arms"][1]["checkpoint_sha256"] = sha256_file(none)
    with pytest.raises(ValueError, match="No-RoPE state"):
        validate_checkpoint_pairs(config)


def test_output_must_be_disjoint(tmp_path: Path):
    protected = tmp_path / "protected"
    protected.mkdir()
    outside = tmp_path / "outside"
    require_disjoint_output(outside, [protected], name="output")
    with pytest.raises(ValueError, match="overlaps"):
        require_disjoint_output(protected / "result", [protected], name="output")


def test_resume_cleanup_rejects_symlink_dataset_root(tmp_path: Path):
    work = tmp_path / "work"
    target = tmp_path / "target"
    work.mkdir()
    target.mkdir()
    (work / "datasets").symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="dataset root"):
        RUNNER._clean_transients(work, work / "datasets", expected_ordinals={0})


def test_second_arm_oom_restarts_pair_through_disk_and_publishes_per_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    info = "a" * 64
    dataset = SimpleNamespace(
        info_sha256=info,
        input_sha256={"info.json": info},
        n_numeric_features=2,
        n_categorical_features=0,
        train=SimpleNamespace(X=np.ones((4, 2)), y=np.array([0, 1, 0, 1])),
        val=SimpleNamespace(X=np.ones((2, 2)), y=np.array([0, 1])),
        test=SimpleNamespace(X=np.ones((2, 2)), y=np.array([0, 1])),
    )
    import pe_mechanism.official_tabicl as official

    monkeypatch.setattr(official, "load_raw_talent_splits", lambda *a, **k: dataset)
    monkeypatch.setattr(
        RUNNER, "_available_bytes", lambda path: 25 * 1024**3
    )
    calls: list[tuple[str, str]] = []

    def fit(path, *args, estimator_options, **kwargs):
        arm_id = Path(path).stem
        mode = estimator_options["offload_mode"]
        calls.append((arm_id, mode))
        if mode == "disk":
            disk_root = Path(estimator_options["disk_offload_dir"])
            assert disk_root.is_dir()
            assert disk_root.name == arm_id
            assert disk_root.is_relative_to(tmp_path / "scratch")
        else:
            assert "disk_offload_dir" not in estimator_options
        if arm_id == "none" and mode in {"auto", "cpu"}:
            raise torch.cuda.OutOfMemoryError("synthetic OOM")
        return SimpleNamespace(
            checkpoint_sha=f"{1 if arm_id == 'rope' else 2:064x}",
            model_sha="b" * 40,
            estimator=SimpleNamespace(),
        )

    monkeypatch.setattr(official, "fit_official_tabicl_driver", fit)
    monkeypatch.setattr(official, "_expected_forward_schedule", lambda estimator: [])
    monkeypatch.setattr(
        RUNNER,
        "_assert_treatment",
        lambda driver, treatment, **kwargs: {
            "kind": treatment["kind"],
            "row_identity_mode": treatment["kind"],
            "row_fingerprint": False,
            "row_fingerprint_dim": 16,
            "row_rope_installed": treatment["kind"] == "rope",
            "parameter_dtype": "float32",
        },
    )
    monkeypatch.setattr(
        RUNNER,
        "_predict_in_chunks",
        lambda *args, **kwargs: (
            np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float32),
            np.asarray([0, 1], dtype=np.int64),
            ["builtins.int:0", "builtins.int:1"],
            {
                "chunk_count": 1,
                "maximum_rows_per_call": RUNNER.PREDICTION_CHUNK_ROWS,
                "source_evidence_level": "strict",
                "all_chunks_exact_baseline_verified": True,
            },
        ),
    )
    config = {
        "arm_order": ("rope", "none"),
        "pairs": [
            {
                "training_source_commit": "b" * 40,
                "arms": [
                    {
                        "arm_id": "rope",
                        "checkpoint_sha256": f"{1:064x}",
                        "treatment": {"kind": "rope"},
                    },
                    {
                        "arm_id": "none",
                        "checkpoint_sha256": f"{2:064x}",
                        "treatment": {"kind": "none"},
                    },
                ],
            }
        ],
    }
    record = {
        "ordinal": 0,
        "name": "synthetic",
        "n_train": 4,
        "n_validation": 2,
        "n_test": 2,
        "n_features": 2,
        "n_classes": 2,
        "info_sha256": info,
        "input_sha256": {"info.json": info},
    }
    output = tmp_path / "datasets" / "0000"
    output.parent.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    task = RUNNER._evaluate_dataset(
        dataset_record=record,
        talent_root=tmp_path,
        config=config,
        staged={"rope": tmp_path / "rope.ckpt", "none": tmp_path / "none.ckpt"},
        model_root=tmp_path,
        model_sha="b" * 40,
        run_contract_sha256="c" * 64,
        destination=output,
        scratch_root=scratch,
    )
    assert calls == [
        ("rope", "auto"),
        ("none", "auto"),
        ("rope", "cpu"),
        ("none", "cpu"),
        ("rope", "disk"),
        ("none", "disk"),
    ]
    assert task["oom_fallback"]["selected_level"] == "disk"
    assert not list(scratch.iterdir())
    assert {path.name for path in output.iterdir()} == DATASET_ARTIFACTS
    for arm_id in config["arm_order"]:
        assert {path.name for path in (output / "arms" / arm_id).iterdir()} == (
            ARM_DATASET_ARTIFACTS
        )
    validated = AGGREGATOR._validate_dataset(
        output,
        expected=record,
        summary_result=task,
        config=config,
        run_contract_sha256="c" * 64,
    )
    assert validated["arms"]["rope"]["accuracy"] == pytest.approx(1.0)
    wrong_split = {**task, "evaluation_split": "test"}
    atomic_json(output / "task.json", wrong_split)
    atomic_json(
        output / "manifest.json", directory_manifest(output, kind=RUNNER.DATASET_KIND)
    )
    with pytest.raises(ValueError, match="evaluation_split"):
        AGGREGATOR._validate_dataset(
            output,
            expected=record,
            summary_result=wrong_split,
            config=config,
            run_contract_sha256="c" * 64,
        )


def test_pair_statistics_include_fixed_column_ood_and_deterministic_ci():
    datasets = []
    for ordinal, features in enumerate((50, 80, 120, 150)):
        datasets.append(
            {
                "ordinal": ordinal,
                "dataset": f"d{ordinal}",
                "n_features": features,
                "arms": {
                    "left": {"accuracy": 0.8, "log_loss": 0.4},
                    "right": {"accuracy": 0.7, "log_loss": 0.5},
                },
            }
        )
    pair = {
        "pair_id": "pair",
        "comparison_step": 1,
        "match_level": "same_step_legacy",
        "training_source_commit": "a" * 40,
        "arms": [{"arm_id": "left"}, {"arm_id": "right"}],
    }
    first = AGGREGATOR._pair_statistics(pair, datasets)
    second = AGGREGATOR._pair_statistics(pair, datasets)
    assert first == second
    assert first["subgroups"]["column_count_ood_gt_100"]["dataset_count"] == 2
    assert first["subgroups"]["column_count_id_le_100"]["dataset_count"] == 2
    assert first["subgroups"]["all_datasets"]["accuracy"]["inferential_role"] == (
        "primary"
    )


def test_arm_prediction_rejects_nonfinite_checkpoint_and_runtime_mismatch(
    tmp_path: Path,
):
    arm_root = tmp_path / "arm"
    arm_root.mkdir()
    target = np.asarray([0, 1], dtype=np.int64)
    probability = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float32)
    np.savez_compressed(
        arm_root / "predictions.npz", target=target, probabilities=probability
    )
    treatment = {
        "kind": "rope",
        "row_identity_mode": "rope",
        "row_fingerprint": False,
        "row_fingerprint_dim": 16,
        "row_rope_installed": True,
        "parameter_dtype": "float32",
    }
    result = {
        "n_evaluation": 2,
        "n_classes": 2,
        "target_sha256": array_sha256(target),
        "probabilities_sha256": array_sha256(probability),
        "checkpoint_sha256": "a" * 64,
        "model_runtime_sha": "b" * 40,
        "offload_mode": "auto",
        "treatment": treatment,
        "accuracy": 1.0,
        "log_loss": float(
            -np.log(np.asarray([0.8, 0.7], dtype=np.float32)).mean()
        ),
    }
    arm = {
        "arm_id": "rope",
        "checkpoint_sha256": "a" * 64,
        "treatment": {"kind": "rope"},
    }
    assert AGGREGATOR._load_arm_prediction(
        arm_root,
        result=result,
        arm=arm,
        training_source_commit="b" * 40,
        selected_offload_mode="auto",
    )["accuracy"] == 1.0
    for field, value in (
        ("checkpoint_sha256", "c" * 64),
        ("model_runtime_sha", "d" * 40),
    ):
        bad = {**result, field: value}
        with pytest.raises(ValueError, match="probability contract"):
            AGGREGATOR._load_arm_prediction(
                arm_root,
                result=bad,
                arm=arm,
                training_source_commit="b" * 40,
                selected_offload_mode="auto",
            )
    nonfinite = probability.copy()
    nonfinite[0, 0] = np.nan
    np.savez_compressed(
        arm_root / "predictions.npz", target=target, probabilities=nonfinite
    )
    with pytest.raises(ValueError, match="probability contract"):
        AGGREGATOR._load_arm_prediction(
            arm_root,
            result={**result, "probabilities_sha256": array_sha256(nonfinite)},
            arm=arm,
            training_source_commit="b" * 40,
            selected_offload_mode="auto",
        )


def test_joint_aggregate_rejects_missing_and_duplicate_dataset_union(tmp_path: Path):
    plan = build_shard_plan_payload(_records(), shard_count=2)
    plan["analysis_sha"] = "a" * 40
    config = {
        "config_sha256": "b" * 64,
        "portable": {},
        "portable_sha256": "c" * 64,
        "checkpoint_contract": {},
        "arm_order": ("left", "right"),
        "pairs": [
            {
                "pair_id": "pair",
                "comparison_step": 1,
                "match_level": "same_step_legacy",
                "training_source_commit": "d" * 40,
                "arms": [{"arm_id": "left"}, {"arm_id": "right"}],
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}")
    document = {"sha256": "e" * 64}
    missing = [{"ordinal": index} for index in range(len(plan["datasets"]) - 1)]
    with pytest.raises(ValueError, match="exact frozen roster"):
        AGGREGATOR._combine(
            configs=[config],
            by_config={config["config_sha256"]: missing},
            evidence={config["config_sha256"]: []},
            canary_evidence={},
            receipt_evidence={},
            plan=plan,
            plan_path=plan_path,
            plan_file_sha256="f" * 64,
            plan_document=document,
            analysis_sha="a" * 40,
        )
    duplicate = [{"ordinal": index} for index in range(len(plan["datasets"]))]
    duplicate[-1] = {"ordinal": 0}
    with pytest.raises(ValueError, match="exact frozen roster|duplicate"):
        AGGREGATOR._combine(
            configs=[config],
            by_config={config["config_sha256"]: duplicate},
            evidence={config["config_sha256"]: []},
            canary_evidence={},
            receipt_evidence={},
            plan=plan,
            plan_path=plan_path,
            plan_file_sha256="f" * 64,
            plan_document=document,
            analysis_sha="a" * 40,
        )


def test_submit_plan_is_canary_then_array_then_cpu_aggregate(tmp_path: Path):
    plan = SUBMITTER._command_plan(
        sbatch="sbatch",
        analysis_root=tmp_path,
        model_root=tmp_path / "model",
        talent_root=tmp_path / "talent",
        run_config=tmp_path / "config.json",
        shard_plan=tmp_path / "plan.json",
        output_root=tmp_path / "output",
        scratch_root=tmp_path / "scratch",
        python=tmp_path / "python",
        analysis_sha="a" * 40,
        model_sha="b" * 40,
        run_config_sha256="c" * 64,
        shard_plan_sha256="d" * 64,
    )
    assert "--hold" in plan["canary"]
    assert "--array=0-7%2" in plan["array"]
    assert "--dependency=afterok:CANARY_JOB_ID" in plan["array"]
    assert "--dependency=afterok:ARRAY_JOB_ID" in plan["aggregate"]


def test_capacity_22gib_is_warning_and_20gib_is_hard_reserve():
    warning = SUBMITTER._capacity_contract(
        output_available=21 * 1024**3,
        capacity_available=23 * 1024**3,
        scratch_available=22 * 1024**3,
    )
    assert warning["warning_triggered"] is True
    assert warning["pre_submit_status_by_filesystem"]["output"] == (
        "warning_below_22_gib"
    )
    SUBMITTER._require_hard_capacity(warning, context="test")

    hard_failure = SUBMITTER._capacity_contract(
        output_available=19 * 1024**3,
        capacity_available=23 * 1024**3,
        scratch_available=23 * 1024**3,
    )
    with pytest.raises(RuntimeError, match="20 GiB"):
        SUBMITTER._require_hard_capacity(hard_failure, context="test")


def test_slurm_wrapper_is_h100_medium_and_monitored():
    wrapper = (ROOT / "scripts/slurm_talent_paired_full_suite.sh").read_text()
    assert "#SBATCH --partition=h100" in wrapper
    assert "#SBATCH --qos=medium" in wrapper
    assert "#SBATCH --mem=128G" in wrapper
    assert "sleep 30" in wrapper
    assert "PE_TALENT_GPU_CSV_ROOT" in wrapper
    assert "disk_offload_dir" not in wrapper
    assert "SLURM_JOB_ID" not in wrapper.split("gpu_csv=", 1)[1]


def test_campaign_receipts_are_strict_bound_and_public_safe(tmp_path: Path):
    config = load_private_run_config(_legacy_config(tmp_path / "config"))
    checkpoint_contract = validate_checkpoint_pairs(config)
    submission_payload = {
        "schema_version": 1,
        "formal_eligible": False,
        "study": "tabicl-talent-paired-checkpoints-exploratory-v1",
        "pair_contract": config["portable"],
        "pair_contract_sha256": config["portable_sha256"],
        "checkpoint_contract": checkpoint_contract,
        "analysis_sha": "a" * 40,
        "model_runtime_sha": "2" * 40,
        "shard_plan_file_sha256": "b" * 64,
        "shard_plan_document_sha256": "c" * 64,
        "canary_dataset_ordinals": [1, 2, 3],
        "shard_count": 8,
        "capacity": {
            "output_pre_submit_available_bytes": 21 * 1024**3,
            "capacity_root_pre_submit_available_bytes": 23 * 1024**3,
            "scratch_root_pre_submit_available_bytes": 23 * 1024**3,
            "warning_threshold_bytes": 22 * 1024**3,
            "hard_reserve_bytes": 20 * 1024**3,
            "warning_triggered": True,
            "pre_submit_status_by_filesystem": {
                "output": "warning_below_22_gib",
                "capacity": "ok_at_or_above_22_gib",
                "scratch": "ok_at_or_above_22_gib",
            },
        },
        "job_roles": {
            "canary": {"submission_state": "held"},
            "full_array": {"dependency": "afterok:canary"},
            "aggregate": {"dependency": "afterok:full_array"},
        },
        "command_plan_sha256_by_role": {
            "canary": "d" * 64,
            "array": "e" * 64,
            "aggregate": "f" * 64,
        },
        "operation_journal_sha256_at_submission": "1" * 64,
        "created_at_unix_seconds": 100.0,
    }
    submission = self_hashed_document(SUBMITTER.SUBMISSION_KIND, submission_payload)
    submission_path = tmp_path / "submission.json"
    atomic_json(submission_path, submission)
    release = self_hashed_document(
        SUBMITTER.RELEASE_KIND,
        {
            "schema_version": 1,
            "submission_document_sha256": submission["sha256"],
            "submission_file_sha256": sha256_file(submission_path),
            "released_job_role": "canary",
            "operation_journal_sha256_after_release": "2" * 64,
            "released_at_unix_seconds": 101.0,
        },
    )
    release_path = tmp_path / "release.json"
    atomic_json(release_path, release)
    evidence = AGGREGATOR._validate_campaign_receipts(
        submission_path=submission_path,
        release_path=release_path,
        config=config,
        checkpoint_contract=checkpoint_contract,
        analysis_sha="a" * 40,
        plan_file_sha256="b" * 64,
        plan_document_sha256="c" * 64,
        canary_ordinals=[1, 2, 3],
    )
    AGGREGATOR._assert_public_safe(evidence)
    assert set(evidence) == {
        "submission_receipt_file_sha256",
        "submission_receipt_document_sha256",
        "release_receipt_file_sha256",
        "release_receipt_document_sha256",
    }
    bad_release = self_hashed_document(
        SUBMITTER.RELEASE_KIND,
        {**release["payload"], "released_job_role": "123456"},
    )
    atomic_json(release_path, bad_release)
    with pytest.raises(ValueError, match="release receipt"):
        AGGREGATOR._validate_campaign_receipts(
            submission_path=submission_path,
            release_path=release_path,
            config=config,
            checkpoint_contract=checkpoint_contract,
            analysis_sha="a" * 40,
            plan_file_sha256="b" * 64,
            plan_document_sha256="c" * 64,
            canary_ordinals=[1, 2, 3],
        )


def test_gpu_csv_and_cross_job_common_contract_gates(tmp_path: Path):
    config_sha = "a" * 64
    gpu_name = "NVIDIA H100 80GB HBM3"
    header = (
        "unix_seconds,run_config_sha256,analysis_sha,model_sha,shard_id,"
        "execution_scope,index,name,utilization_gpu_percent,memory_used_mib,"
        "memory_total_mib,power_draw_watts\n"
    )
    rows = [
        f"100,{config_sha},{'b' * 40},{'c' * 40},000,full_planned_shard,0,"
        f"{gpu_name},85,1000,80000,300\n",
        f"130,{config_sha},{'b' * 40},{'c' * 40},000,full_planned_shard,0,"
        f"{gpu_name},95,2000,80000,350\n",
    ]
    monitor_path = tmp_path / "monitor.csv"
    monitor_path.write_text(header + "".join(rows), encoding="utf-8")
    common = {
        "common_run_contract_sha256": "d" * 64,
        "environment_contract_sha256": "e" * 64,
        "environment_contract": {"safe": True},
    }
    jobs = [dict(common), dict(common)]
    AGGREGATOR._require_consistent_job_group("pair", jobs)
    jobs[1]["environment_contract_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="environment_contract_sha256"):
        AGGREGATOR._require_consistent_job_group("pair", jobs)
    key, result = AGGREGATOR._validate_gpu_csv(
        monitor_path,
        expected_jobs={
            (config_sha, "000"): {
                "model_runtime_sha": "c" * 40,
                "execution_scope": "full_planned_shard",
                "gpu_name": gpu_name,
                "monitor_window": {
                    "started_at_unix_seconds": 100.0,
                    "completed_at_unix_seconds": 130.0,
                },
            }
        },
        analysis_sha="b" * 40,
    )
    assert key == (config_sha, "000")
    assert result["active_window_mean_percent"] == pytest.approx(90.0)
    monitor_path.write_text(
        (header + "".join(rows)).replace(",95,", ",10,"), encoding="utf-8"
    )
    _, low_utilization = AGGREGATOR._validate_gpu_csv(
        monitor_path,
        expected_jobs={
            (config_sha, "000"): {
                "model_runtime_sha": "c" * 40,
                "execution_scope": "full_planned_shard",
                "gpu_name": gpu_name,
                "monitor_window": {
                    "started_at_unix_seconds": 100.0,
                    "completed_at_unix_seconds": 130.0,
                },
            }
        },
        analysis_sha="b" * 40,
    )
    assert low_utilization["active_window_mean_percent"] == pytest.approx(47.5)
    assert low_utilization["below_80_percent_observation"] is True


def test_durable_journal_and_verified_per_job_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    journal = SUBMITTER._DurableJournal(tmp_path / "journal.json")

    def run(command, **kwargs):
        del kwargs
        if command[0] == "sbatch":
            return SimpleNamespace(returncode=0, stdout="123;cluster\n", stderr="")
        if command[0] == "scancel":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[0] == "sacct":
            return SimpleNamespace(
                returncode=0, stdout="123|CANCELLED|\n", stderr=""
            )
        raise AssertionError(command)

    monkeypatch.setattr(SUBMITTER.subprocess, "run", run)
    monkeypatch.setattr(SUBMITTER.time, "sleep", lambda seconds: None)
    job_id = SUBMITTER._submit_job(
        ["sbatch", "wrapper.sh"], role="canary", journal=journal
    )
    assert job_id == "123"
    rollback = SUBMITTER._rollback_jobs(
        {"canary": job_id}, journal=journal, failure_type="RuntimeError"
    )
    assert rollback["payload"]["rollback_verified"] is True
    assert "123" not in json.dumps(rollback, sort_keys=True)
    journal_document = json.loads((tmp_path / "journal.json").read_text())
    assert journal_document["kind"] == SUBMITTER.OPERATION_JOURNAL_KIND
    assert any(
        event.get("job_id") == "123"
        for event in journal_document["payload"]["events"]
    )


def test_rollback_never_claims_success_for_terminal_non_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    journal = SUBMITTER._DurableJournal(tmp_path / "journal.json")

    def run(command, **kwargs):
        del kwargs
        if command[0] in {"scancel", "squeue"}:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[0] == "sacct":
            return SimpleNamespace(returncode=0, stdout="123|COMPLETED|\n", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(SUBMITTER.subprocess, "run", run)
    monkeypatch.setattr(SUBMITTER.time, "sleep", lambda seconds: None)
    rollback = SUBMITTER._rollback_jobs(
        {"canary": "123"}, journal=journal, failure_type="RuntimeError"
    )
    assert rollback["payload"]["rollback_verified"] is False
    assert rollback["payload"]["rollback_status"] == "unverified_or_incomplete"
