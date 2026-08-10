from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import tarfile
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from pe_mechanism import cli
from pe_mechanism.manifest import ArtifactDigest, FileDigest, new_manifest
import pe_mechanism.tabarena_formal_evaluation as intake
import pe_mechanism.tabarena_formal_runner as runner


PACKAGE_ROOT = Path(__file__).parents[1]


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _formal_input_config(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    value = json.loads(
        (PACKAGE_ROOT / "examples" / "tabarena-formal-inputs.example.json").read_text(
            encoding="utf-8"
        )
    )
    training = tmp_path / "training"
    artifacts = tmp_path / "artifacts"
    value["training_code_root"] = str(training)
    value["artifact_root"] = str(artifacts)
    value["transaction_ledger_path"] = str(artifacts / "transaction-ledger.json")
    for arm in intake.FORMAL_ARMS:
        for stage, step in intake.FORMAL_STAGES:
            root = artifacts / "arms" / arm / stage
            entry = value["arms"][arm][stage]
            entry["checkpoint_path"] = str(root / f"step-{step}.ckpt")
            entry["finalized_manifest_path"] = str(root / "finalized-checkpoint.json")
    path = tmp_path / "formal-inputs.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, value


def _runner_config(tmp_path: Path) -> dict[str, Any]:
    formal_path, formal = _formal_input_config(tmp_path)
    campaign = tmp_path / "campaign"
    artifacts = Path(formal["artifact_root"])
    fixed = deepcopy(runner._FIXED_CLASSIFIER_OPTIONS)
    environment = json.loads(
        (
            PACKAGE_ROOT / "examples" / "tabarena-formal-environment.example.json"
        ).read_text(encoding="utf-8")
    )
    environment_path = tmp_path / "evaluation-environment.json"
    environment_path.write_text(json.dumps(environment), encoding="utf-8")
    return {
        "schema_version": 1,
        "study": {
            "seed": 42,
            "temporary_identity_seed": 42,
            "stage": "stage3",
            "terminal_step": 10_000,
            "task_subset": "lite",
            "problem_types": ["binary", "multiclass"],
            "device": "cuda",
            "n_estimators": 1,
            "augmentation": "none",
            "ensemble_size": 1,
            "cache_mode": "ignore",
            "debug_mode": True,
            "bootstrap_resamples": 10_000,
            "familywise_alpha": 0.05,
            "classifier_options": fixed,
        },
        "formal_inputs": {
            "config_path": str(formal_path),
            "expected_config_sha256": _sha(formal_path.read_bytes()),
        },
        "readiness": {
            "campaign_path": str(campaign / "campaign.json"),
            "expected_campaign_file_sha256": "a" * 64,
            "expected_campaign_manifest_sha256": "b" * 64,
            "acceptance_registry": str(campaign / "acceptances"),
            "terminal_attestation_path": str(
                artifacts / "terminal-scheduler-logs.json"
            ),
            "expected_terminal_file_sha256": "c" * 64,
            "expected_terminal_manifest_sha256": "d" * 64,
            "submission_receipt_path": str(artifacts / "submission-receipt.json"),
            "expected_submission_receipt_file_sha256": "e" * 64,
            "expected_submission_receipt_manifest_sha256": "f" * 64,
            "protocol_metadata_allowance_bytes": 1 << 20,
        },
        "benchmark": {
            "tabarena_code_root": str(tmp_path / "tabarena"),
            "expected_tabarena_code_sha": "2" * 40,
            "openml_cache_root": str(tmp_path / "openml"),
            "environment_manifest_path": str(environment_path),
            "expected_environment_manifest_sha256": _sha(environment_path.read_bytes()),
            "suite_version": "v0.1",
            "expected_task_count": 38,
            "expected_result_count": 114,
        },
        "provenance": {
            "model_family": "tabicl-v2",
            "model_revision": "formal-stage3-step-10000",
            "condition": "tabarena-formal-rope-temporary-none",
            "sites": ["tabarena-v0.1-classification"],
            "checkpoint_path": formal["arms"]["rope"]["stage3"]["checkpoint_path"],
            "dataset_manifest_path": str(tmp_path / "roster.json"),
            "training_code_root": formal["training_code_root"],
            "model_code_root": formal["training_code_root"],
            "analysis_code_root": str(tmp_path / "analysis"),
            "expected_checkpoint_sha256": formal["arms"]["rope"]["stage3"][
                "expected_checkpoint_sha256"
            ],
            "expected_dataset_manifest_sha256": (runner._EXPECTED_ROSTER_FILE_SHA256),
            "expected_training_code_sha": formal["expected_training_code_sha"],
            "expected_model_code_sha": formal["expected_training_code_sha"],
            "expected_analysis_code_sha": "3" * 40,
            "allow_exploratory_legacy": False,
        },
    }


def test_runner_contract_is_stage3_38x3_and_cache_free(tmp_path: Path) -> None:
    value = _runner_config(tmp_path)
    spec = runner.parse_runner_config(value)
    assert spec.formal_inputs.stage == "stage3"
    assert spec.expected_result_count == 114
    assert spec.classifier_options["kv_cache"] is False
    assert spec.temporary_identity_seed == 42
    assert spec.bootstrap_resamples == 10_000
    assert spec.environment_contract["cuda_device_name"] == "NVIDIA A10"

    broken = deepcopy(value)
    broken["study"]["stage"] = "stage2"
    broken["study"]["terminal_step"] = 40_000
    with pytest.raises(ValueError, match="Stage 3 only"):
        runner.parse_runner_config(broken)

    broken = deepcopy(value)
    broken["study"]["bootstrap_resamples"] = 100
    with pytest.raises(ValueError, match="10000 resamples"):
        runner.parse_runner_config(broken)

    broken = deepcopy(value)
    broken["study"]["temporary_identity_seed"] = 43
    with pytest.raises(ValueError, match="temporary_identity_seed"):
        runner.parse_runner_config(broken)

    broken = deepcopy(value)
    broken["study"]["classifier_options"]["kv_cache"] = True
    with pytest.raises(ValueError, match="classifier_options|kv_cache"):
        runner.parse_runner_config(broken)


def test_public_formal_runner_example_exposes_the_complete_contract() -> None:
    value = json.loads(
        (PACKAGE_ROOT / "examples" / "tabarena-formal-evaluate.example.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(value) == {
        "schema_version",
        "study",
        "formal_inputs",
        "readiness",
        "benchmark",
        "provenance",
    }
    assert value["study"]["temporary_identity_seed"] == value["study"]["seed"]
    assert value["study"]["classifier_options"]["kv_cache"] is False
    assert value["benchmark"]["expected_result_count"] == 114
    assert value["benchmark"]["environment_manifest_path"].startswith("/absolute/")
    assert value["readiness"]["campaign_path"].startswith("/absolute/")


def test_temporary_reset_changes_only_module_local_generator() -> None:
    torch.manual_seed(1234)
    global_before = torch.random.get_rng_state().clone()
    local = torch.Generator(device="cpu")
    local.manual_seed(999)
    estimator = SimpleNamespace(
        model_=SimpleNamespace(
            row_identity_mode="temporary",
            row_interactor=SimpleNamespace(
                identity_mode="temporary", _identity_generator=local
            ),
        )
    )
    runner._reset_temporary_identity_rng(estimator, arm="temporary", seed=42)
    assert torch.equal(torch.random.get_rng_state(), global_before)
    expected = torch.Generator(device="cpu")
    expected.manual_seed(42)
    assert torch.equal(local.get_state(), expected.get_state())

    unchanged = local.get_state().clone()
    runner._reset_temporary_identity_rng(estimator, arm="rope", seed=17)
    assert torch.equal(local.get_state(), unchanged)


def test_formal_network_gate_fails_closed_and_restores_socket() -> None:
    original = socket.socket
    with runner._deny_network_access():
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(RuntimeError, match="network access is disabled"):
                connection.connect(("127.0.0.1", 9))
        finally:
            connection.close()
    assert socket.socket is original


def test_runtime_environment_must_match_every_frozen_field(tmp_path: Path) -> None:
    spec = runner.parse_runner_config(_runner_config(tmp_path))
    observed = dict(spec.environment_contract)
    runner._assert_runtime_environment(
        expected=spec.environment_contract,
        observed=observed,
    )
    observed["torch"] = "unexpected"
    with pytest.raises(RuntimeError, match="torch"):
        runner._assert_runtime_environment(
            expected=spec.environment_contract,
            observed=observed,
        )


def test_external_runtime_root_cannot_overlap_protected_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protected = tmp_path / "protected"
    runtime = protected / "runtime"
    runtime.mkdir(parents=True)
    monkeypatch.setenv("PE_RUNTIME_ROOT", str(runtime))
    with pytest.raises(ValueError, match="outside the source tree"):
        runner._external_runtime_root((protected,))


def test_temporary_table_seed_is_lifecycle_stable_and_table_specific() -> None:
    X_a = pd.DataFrame({"x": [1, 2, 3], "z": [4.0, 5.0, 6.0]})
    X_b = pd.DataFrame({"x": [1, 2, 9], "z": [4.0, 5.0, 6.0]})
    y = pd.Series([0, 1, 0], name="target")
    global_before = torch.random.get_rng_state().clone()
    fingerprint_a = runner._table_content_fingerprint(X_a, y)
    fingerprint_repeat = runner._table_content_fingerprint(X_a.copy(), y.copy())
    fingerprint_b = runner._table_content_fingerprint(X_b, y)
    seed_a = runner._temporary_table_seed(
        formal_seed=42, table_fingerprint_sha256=fingerprint_a
    )
    seed_repeat = runner._temporary_table_seed(
        formal_seed=42, table_fingerprint_sha256=fingerprint_repeat
    )
    seed_b = runner._temporary_table_seed(
        formal_seed=42, table_fingerprint_sha256=fingerprint_b
    )
    assert fingerprint_a == fingerprint_repeat
    assert fingerprint_a != fingerprint_b
    assert seed_a == seed_repeat
    assert seed_a != seed_b
    local = torch.Generator(device="cpu")
    local.manual_seed(seed_a)
    first = torch.randperm(64, generator=local)
    local.manual_seed(seed_a)
    assert torch.equal(first, torch.randperm(64, generator=local))
    local.manual_seed(seed_b)
    assert not torch.equal(first, torch.randperm(64, generator=local))
    assert torch.equal(torch.random.get_rng_state(), global_before)


def test_formal_prediction_capture_is_float32_atomic_and_self_verifying(
    tmp_path: Path,
) -> None:
    root = tmp_path / "predictions"
    probabilities = pd.DataFrame(
        [[0.25, 0.75], [0.6, 0.4]],
        index=pd.Index([101, 205]),
        columns=pd.Index(["negative", "positive"]),
    )
    metadata = runner._capture_formal_prediction(
        probabilities,
        test_targets=pd.Series([1, 0], index=probabilities.index, name="target"),
        prediction_root=root,
        arm="rope",
        dataset="safe/filename/is-hashed",
        task_id=7,
        fold=0,
        repeat=0,
        sample=0,
        split_idx=0,
    )
    assert "/" not in metadata["file_name"]
    assert "\\" not in metadata["file_name"]
    assert metadata["dtype"] == "float32"
    assert metadata["shape"] == [2, 2]
    runner._verify_prediction_npz(root / metadata["file_name"], metadata)
    with np.load(root / metadata["file_name"], allow_pickle=False) as payload:
        assert payload["probabilities"].dtype == np.float32
        assert np.isfinite(payload["probabilities"]).all()
        assert payload["row_fingerprints"].shape == (2, 32)
        assert payload["test_target_fingerprints"].shape == (2, 32)

    bad = probabilities.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        runner._capture_formal_prediction(
            bad,
            test_targets=pd.Series([1, 0], index=bad.index, name="target"),
            prediction_root=root,
            arm="none",
            dataset="bad",
            task_id=8,
            fold=0,
            repeat=0,
            sample=0,
            split_idx=0,
        )
    assert len(list(root.iterdir())) == 1


def test_formal_prediction_runner_keeps_arrays_out_of_result_cache(
    tmp_path: Path,
) -> None:
    class FakeRunner:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.task = SimpleNamespace(task_id=11)
            self.task_name = "dataset-a"
            self.fold = self.repeat = self.sample = self.task_split_idx = 0

        def post_evaluate(self, out: dict[str, Any]) -> dict[str, Any]:
            out["method_metadata"] = {"arm": "rope"}
            return out

        def convert_to_output(self, out: dict[str, Any]) -> dict[str, Any]:
            out.pop("predictions")
            out.pop("probabilities")
            return out

        def _load_y_test(self) -> pd.Series:
            return pd.Series([1, 0], index=[4, 9], name="target")

    capture_cls = runner._make_formal_prediction_runner(FakeRunner)
    instance = capture_cls(formal_prediction_root=str(tmp_path / "private"))
    output = instance.post_evaluate(
        {
            "predictions": pd.Series([1, 0]),
            "probabilities": pd.DataFrame(
                [[0.2, 0.8], [0.7, 0.3]], columns=[0, 1], index=[4, 9]
            ),
        }
    )
    output = instance.convert_to_output(output)
    assert "predictions" not in output
    assert "probabilities" not in output
    assert set(output) == {"method_metadata", "formal_prediction_metadata"}
    assert output["formal_prediction_metadata"]["row_count"] == 2


def test_checkpoint_scratch_copy_is_streamed_and_rehashed(tmp_path: Path) -> None:
    source = tmp_path / "source.ckpt"
    source.write_bytes(b"a" * (runner._COPY_CHUNK_BYTES + 17))
    expected = _sha(source.read_bytes())
    copied = runner._stream_copy_checkpoint(
        source, tmp_path / "scratch" / "rope.ckpt", expected_sha256=expected
    )
    assert copied["sha256"] == expected
    assert copied["size_bytes"] == source.stat().st_size
    assert Path(copied["path"]).read_bytes() == source.read_bytes()

    destination = tmp_path / "scratch" / "bad.ckpt"
    with pytest.raises(ValueError, match="digest or size"):
        runner._stream_copy_checkpoint(source, destination, expected_sha256="0" * 64)
    assert not destination.exists()


def test_run_context_keeps_campaign_terminal_and_receipt_live_until_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = runner.parse_runner_config(_runner_config(tmp_path))
    observed: dict[str, Any] = {}
    sentinel = SimpleNamespace()

    def verify(*args: Any, **kwargs: Any) -> SimpleNamespace:
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(runner, "verify_configured_run_inputs", verify)
    assert runner._context(SimpleNamespace(), spec) is sentinel
    paths = observed["additional_input_paths"]
    hashes = observed["expected_additional_sha256"]
    assert paths["formal_campaign"] == spec.readiness.campaign_path
    assert paths["formal_terminal_attestation"] == (
        spec.readiness.terminal_attestation_path
    )
    assert paths["formal_submission_receipt"] == (
        spec.readiness.submission_receipt_path
    )
    assert paths["evaluation_environment"] == spec.environment_manifest_path
    assert hashes["formal_campaign"] == spec.readiness.campaign_file_sha256
    assert hashes["formal_terminal_attestation"] == (
        spec.readiness.terminal_file_sha256
    )
    assert hashes["formal_submission_receipt"] == (
        spec.readiness.submission_receipt_file_sha256
    )
    assert hashes["evaluation_environment"] == spec.environment_manifest_sha256


def _row(arm: str, dataset: str, error: float) -> dict[str, Any]:
    return {
        "arm": arm,
        "framework": f"framework-{arm}",
        "dataset": dataset,
        "task_id": 1 if dataset == "a" else 2,
        "fold": 0,
        "repeat": 0,
        "sample": 0,
        "split_idx": 0,
        "problem_type": "binary",
        "metric": "roc_auc",
        "metric_error": error,
        "time_train_s": 0.1,
        "time_infer_s": 0.2,
    }


def test_single_seed_summary_has_all_pairs_holm_and_no_formal_claim(
    tmp_path: Path,
) -> None:
    config = _runner_config(tmp_path)
    spec = runner.parse_runner_config(config)
    normalized = [
        _row(arm, dataset, error)
        for dataset, errors in (
            ("a", {"rope": 0.3, "temporary": 0.2, "none": 0.1}),
            ("b", {"rope": 0.1, "temporary": 0.3, "none": 0.2}),
        )
        for arm, error in errors.items()
    ]
    code = SimpleNamespace(head_sha="1" * 40)
    context = SimpleNamespace(
        inputs=SimpleNamespace(
            training_code=code,
            model_code=code,
            analysis_code=SimpleNamespace(head_sha="3" * 40),
            dataset_manifest=SimpleNamespace(
                digest=SimpleNamespace(sha256=runner._EXPECTED_ROSTER_FILE_SHA256)
            ),
        )
    )
    public, private = runner._aggregate(
        normalized,
        roster=("a", "b"),
        spec=spec,
        framework_to_arm={f"framework-{arm}": arm for arm in intake.FORMAL_ARMS},
        lineage={"kind": "lineage"},
        readiness={"benchmark_execution_ready": True},
        context=context,
        benchmark_sha="2" * 40,
        evaluation_protocol={
            "schema_version": 1,
            "kind": "formal_tabarena_evaluation_protocol",
            "sha256": "9" * 64,
        },
    )
    assert public["formal_claim_ready"] is False
    assert len(public["overall_scale_free_pairwise"]) == 3
    assert len(public["metric_groups"]["roc_auc"]["paired_comparisons"]) == 3
    assert all(
        item["multiplicity_method"] == "holm"
        for item in public["overall_scale_free_pairwise"]
        + public["metric_groups"]["roc_auc"]["paired_comparisons"]
    )
    assert all(
        isinstance(item["familywise_reject"], bool)
        for item in public["overall_scale_free_pairwise"]
        + public["metric_groups"]["roc_auc"]["paired_comparisons"]
    )
    assert public["raw_predictions_saved"] is False
    assert public["campaign_acceptance_ready"] is False
    assert public["primary_inference_family"] == "overall_scale_free_pairwise"
    assert all(
        group["inferential_role"] == "secondary_exploratory"
        for group in public["metric_groups"].values()
    )
    assert len(private["results"]) == 6


def test_single_seed_summary_exposes_exact_t_sanity_receipt_inputs(
    tmp_path: Path,
) -> None:
    spec = runner.parse_runner_config(_runner_config(tmp_path))
    normalized = [
        _row(arm, dataset, error)
        for dataset, errors in (
            ("a", {"rope": 0.3, "temporary": 0.2, "none": 0.1}),
            ("b", {"rope": 0.1, "temporary": 0.3, "none": 0.2}),
        )
        for arm, error in errors.items()
    ]
    code = SimpleNamespace(head_sha="1" * 40)
    context = SimpleNamespace(
        inputs=SimpleNamespace(
            training_code=code,
            model_code=code,
            analysis_code=SimpleNamespace(head_sha="3" * 40),
            dataset_manifest=SimpleNamespace(
                digest=SimpleNamespace(sha256=runner._EXPECTED_ROSTER_FILE_SHA256)
            ),
        )
    )
    arm_receipts = {
        arm: hashlib.sha256(arm.encode()).hexdigest() for arm in intake.FORMAL_ARMS
    }
    prediction_manifest = {
        "kind": "formal_tabarena_prediction_manifest",
        "prediction_count": 6,
        "matched_dataset_sha256": "8" * 64,
        "arm_logical_sha256": arm_receipts,
        "overall_logical_sha256": "7" * 64,
        "arms": {arm: {"evaluated_examples": 24} for arm in intake.FORMAL_ARMS},
    }
    public, _private = runner._aggregate(
        normalized,
        roster=("a", "b"),
        spec=spec,
        framework_to_arm={f"framework-{arm}": arm for arm in intake.FORMAL_ARMS},
        lineage={"kind": "lineage"},
        readiness={"benchmark_execution_ready": True},
        context=context,
        benchmark_sha="2" * 40,
        evaluation_protocol={
            "schema_version": 1,
            "kind": "formal_tabarena_evaluation_protocol",
            "sha256": "9" * 64,
        },
        prediction_manifest=prediction_manifest,
    )
    receipt = public["minimum_evaluation_sanity_inputs"]
    assert receipt["matched_dataset_sha256"] == "8" * 64
    assert set(receipt["results_by_arm"]) == set(intake.FORMAL_ARMS)
    assert all(
        receipt["results_by_arm"][arm]["prediction_manifest_sha256"]
        == arm_receipts[arm]
        and receipt["results_by_arm"][arm]["evaluated_datasets"] == 2
        and receipt["results_by_arm"][arm]["evaluated_examples"] == 24
        and set(receipt["results_by_arm"][arm]["metrics"]) == {"mean_rank"}
        for arm in intake.FORMAL_ARMS
    )
    assert public["campaign_acceptance_ready"] is False


def test_method_metadata_is_exact_and_prediction_manifest_is_38x3(
    tmp_path: Path,
) -> None:
    spec = runner.parse_runner_config(_runner_config(tmp_path))
    fingerprint = "a" * 64
    table_seed = runner._temporary_table_seed(
        formal_seed=spec.seed, table_fingerprint_sha256=fingerprint
    )
    temporary = runner._expected_method_metadata(
        arm="temporary",
        seed=spec.seed,
        checkpoint_sha256=spec.formal_inputs.arms["temporary"][
            "stage3"
        ].checkpoint_sha256,
        training_code_sha=spec.formal_inputs.training_code_sha,
        classifier_options=spec.classifier_options,
        table_fingerprint_sha256=fingerprint,
        test_feature_fingerprint_sha256="b" * 64,
        temporary_table_identity_seed=table_seed,
    )
    assert (
        runner._validate_method_metadata(temporary, arm="temporary", spec=spec)
        == temporary
    )
    tampered = {**temporary, "kv_cache": True}
    with pytest.raises(ValueError, match="exact evaluation contract"):
        runner._validate_method_metadata(tampered, arm="temporary", spec=spec)

    prediction_root = tmp_path / "prediction-matrix"
    roster = tuple(f"dataset-{index:02d}" for index in range(38))
    evidence: dict[tuple[str, str], dict[str, Any]] = {}
    for arm in intake.FORMAL_ARMS:
        for task_id, dataset in enumerate(roster):
            metadata = runner._capture_formal_prediction(
                pd.DataFrame(
                    [[0.2, 0.8], [0.7, 0.3]],
                    columns=[0, 1],
                    index=[task_id * 2, task_id * 2 + 1],
                ),
                test_targets=pd.Series(
                    [1, 0],
                    index=[task_id * 2, task_id * 2 + 1],
                    name="target",
                ),
                prediction_root=prediction_root,
                arm=arm,
                dataset=dataset,
                task_id=task_id,
                fold=0,
                repeat=0,
                sample=0,
                split_idx=0,
            )
            evidence[(arm, dataset)] = {
                "method_metadata": {
                    "table_fingerprint_sha256": hashlib.sha256(
                        f"train:{dataset}".encode()
                    ).hexdigest(),
                    "test_feature_fingerprint_sha256": hashlib.sha256(
                        f"test:{dataset}".encode()
                    ).hexdigest(),
                },
                "prediction": metadata,
            }
    manifest = runner._prediction_manifest(
        evidence,
        prediction_root=prediction_root,
        roster=roster,
        spec=spec,
    )
    reversed_manifest = runner._prediction_manifest(
        dict(reversed(list(evidence.items()))),
        prediction_root=prediction_root,
        roster=roster,
        spec=spec,
    )
    assert manifest["prediction_count"] == 114
    assert set(manifest["arm_logical_sha256"]) == set(intake.FORMAL_ARMS)
    assert len(manifest["matched_dataset_sha256"]) == 64
    assert set(manifest["matched_dataset_by_name_sha256"]) == set(roster)
    assert all(
        manifest["arms"][arm]["prediction_count"] == 38 for arm in intake.FORMAL_ARMS
    )
    assert manifest["arm_logical_sha256"] == reversed_manifest["arm_logical_sha256"]
    assert (
        manifest["overall_logical_sha256"]
        == reversed_manifest["overall_logical_sha256"]
    )
    mismatched = deepcopy(evidence)
    mismatched[("none", roster[0])]["method_metadata"][
        "test_feature_fingerprint_sha256"
    ] = "f" * 64
    with pytest.raises(ValueError, match="differs across arms"):
        runner._prediction_manifest(
            mismatched,
            prediction_root=prediction_root,
            roster=roster,
            spec=spec,
        )
    mismatched_target = deepcopy(evidence)
    mismatched_target[("none", roster[0])]["prediction"]["test_target_sha256"] = (
        "e" * 64
    )
    with pytest.raises(ValueError, match="differs across arms"):
        runner._prediction_manifest(
            mismatched_target,
            prediction_root=prediction_root,
            roster=roster,
            spec=spec,
        )
    archive = tmp_path / "predictions.tar.gz"
    runner._archive_prediction_directory(prediction_root, archive)
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
    assert len(members) == 114
    assert all(
        member.isfile()
        and member.name.startswith("predictions/prediction-")
        and ".." not in member.name
        for member in members
    )


def test_cli_manifest_and_a10_wrapper_register_only_the_new_command() -> None:
    assert cli._COMMAND_MODULES["tabarena-formal-evaluate"] == (
        "tabarena_formal_runner"
    )
    manifest = new_manifest(
        command="tabarena-formal-evaluate",
        model_family="tabicl-v2",
        model_revision="formal-stage3-step-10000",
        training_code_sha="1" * 40,
        model_code_sha="1" * 40,
        analysis_code_sha="3" * 40,
        configuration=FileDigest("4" * 64, 1),
        checkpoint=FileDigest("5" * 64, 2),
        dataset_manifest=FileDigest("6" * 64, 3),
        condition="tabarena-formal-rope-temporary-none",
        sites=("tabarena-v0.1-classification",),
        seed=42,
        artifacts=(ArtifactDigest("public-finding.json", "7" * 64, 4),),
        created_at_utc="2026-08-10T00:00:00Z",
    )
    assert manifest.command == "tabarena-formal-evaluate"
    wrapper = (
        PACKAGE_ROOT / "scripts" / "slurm_tabarena_formal_evaluate.sh"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=normal" in wrapper
    assert "#SBATCH --time=03:00:00" in wrapper
    assert '"NVIDIA A10"' in wrapper
    assert "tabarena-formal-evaluate" in wrapper
    assert "requires NVIDIA A10" in wrapper
    assert "PE_RUNTIME_ROOT must not exist" in wrapper
