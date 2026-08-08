from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import stat
import sys

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_formal_identity_production_job.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("formal_production_runtime", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _manifest(kind: str, payload: dict) -> dict:
    body = {"schema_version": 1, "kind": kind, "payload": payload}
    return {**body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def _write(path: Path, value: dict) -> None:
    path.write_bytes(_canonical(value) + b"\n")


def _runtime_fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    observed_time_limit: str = "14-00:00:00",
):
    exact = tmp_path / "exact"
    scripts = exact / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(
        Path(__file__).parents[1] / "scripts/verify_git_repository.py",
        scripts / "verify_git_repository.py",
    )
    artifact = tmp_path / "artifacts" / "study-seed42"
    logs = artifact / "scheduler-logs"
    completions = artifact / "runtime-completions"
    logs.mkdir(parents=True)
    completions.mkdir()
    stdout_path = logs / "rope-stage1.out"
    stderr_path = logs / "rope-stage1.err"
    stdout_path.write_bytes(b"bootstrap\n")
    stderr_path.write_bytes(b"")
    completion_path = completions / "rope-stage1.json"
    stage_dir = artifact / "arms" / "rope" / "stage1"
    stage_dir.mkdir(parents=True)
    checkpoint_path = stage_dir / "step-500000.ckpt"
    checkpoint_path.write_bytes(b"checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    digests = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in (
            "source",
            "environment",
            "prior",
            "architecture",
            "optimizer",
            "scientific",
            "cohort",
            "arm",
            "provenance",
            "seed",
            "treatment",
        )
    }
    ledger_entry = {
        "arm": "rope",
        "stage": "stage1",
        "terminal_step": 500_000,
        "upstream_identity": "study-seed42:rope:stage1",
        "artifact_identity": "study-seed42.rope.stage1.final",
        "checkpoint_relpath": "arms/rope/stage1/step-500000.ckpt",
        "finalized_manifest_relpath": "arms/rope/stage1/finalized-checkpoint.json",
        "np_seed": 42,
        "torch_seed": 42,
        "identity_rng_seed": 42,
        "world_size": 1,
        "cuda_device_count": 1,
        "max_checkpoint_bytes": 1000,
        "source_sha256": digests["source"],
        "environment_sha256": digests["environment"],
        "prior_sha256": digests["prior"],
        "architecture_sha256": digests["architecture"],
        "optimizer_sha256": digests["optimizer"],
        "scientific_sha256": digests["scientific"],
        "cohort_protocol_sha256": digests["cohort"],
        "arm_protocol_sha256": digests["arm"],
    }
    h100_gate = {
        "attestation_sha256": "7" * 64,
        "checkpoint_ceiling_bytes": 1000,
        "gpu_model": "NVIDIA H100 80GB HBM3",
        "driver_version": "570.00",
        "nvidia_smi_sha256": "6" * 64,
    }
    campaign_binding = {
        "campaign_id": "study",
        "campaign_manifest_sha256": "8" * 64,
        "training_commit_sha": "a" * 40,
        "training_tree_sha": "b" * 40,
        "source_manifest_sha256": digests["source"],
        "environment_sha256": digests["environment"],
        "h100_attestation_sha256": h100_gate["attestation_sha256"],
        "nvidia_smi_sha256": "6" * 64,
        "checkpoint_ceiling_bytes": 1000,
        "static_protocol_sha256_by_stage": {
            "stage1": "1" * 64,
            "stage2": "2" * 64,
            "stage3": "3" * 64,
        },
        "time_limit_by_stage": {
            "stage1": "14-00:00:00",
            "stage2": "3-00:00:00",
            "stage3": "1-00:00:00",
        },
        "predecessor_acceptance_sha256_by_seed": {},
    }
    ledger = _manifest(
        "transaction_ledger",
        {
            "study_id": "study-seed42",
            "protocol_metadata_allowance_bytes": 3_000_000,
            "entries": [ledger_entry],
            "h100_gate": h100_gate,
            "campaign_binding": campaign_binding,
            "runtime_tools": {"nvidia_smi_sha256": "6" * 64},
        },
    )
    ledger_path = artifact / "transaction-ledger.json"
    _write(ledger_path, ledger)
    job_name = "tabicl-0123456789abcdef0123456789abcdef-rope-s1"
    scontrol = tmp_path / "scontrol"
    scontrol.write_text(
        "#!/bin/bash\n"
        f"printf '%s\\n' 'JobId=1001 JobName={job_name} Partition=h100 "
        f"QOS=long TimeLimit={observed_time_limit} "
        "NumNodes=1 NumCPUs=64 CPUs/Task=64 MinMemoryNode=128G "
        "TresPerNode=gres:gpu:1'\n"
    )
    scontrol.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    scontrol_sha256 = hashlib.sha256(scontrol.read_bytes()).hexdigest()
    helper_spec = importlib.util.spec_from_file_location(
        "formal_runtime_repository_helper", scripts / "verify_git_repository.py"
    )
    assert helper_spec is not None and helper_spec.loader is not None
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    git_sha256 = "d" * 64
    repository_binding = helper.expected_repository_binding(
        expected_commit_sha="a" * 40,
        git_sha256=git_sha256,
    )
    receipt = _manifest(
        "held_submission_receipt",
        {
            "study_id": "study-seed42",
            "seed": 42,
            "transaction_id": "0123456789abcdef0123456789abcdef",
            "transaction_ledger_sha256": ledger["sha256"],
            "source_commit_sha": "a" * 40,
            "source_tree_sha": "b" * 40,
            "repository_binding": repository_binding,
            "h100_gate": h100_gate,
            "campaign_binding": campaign_binding,
            "runtime_tools": {"nvidia_smi_sha256": "6" * 64},
            "jobs_held_at_publication": True,
            "run_log_ceiling_bytes": 10_000,
            "manifest_ceiling_bytes": 10_000,
            "protocol_metadata_allowance_bytes": 3_000_000,
            "runtime_completion_ceiling_bytes": 65_536,
            "terminal_log_attestation_path": str(
                artifact / "terminal-scheduler-logs.json"
            ),
            "transaction_commit_path": str(
                artifact / "transaction-committed.json"
            ),
            "terminal_log_attestation_ceiling_bytes": 131_072,
            "scheduler": {
                "partition": "h100",
                "qos": "long",
                "cpus_per_task": 64,
                "memory_mb": 131_072,
                "gpus_per_job": 1,
                "time_limit_by_stage": {
                    "stage1": "14-00:00:00",
                    "stage2": "3-00:00:00",
                    "stage3": "1-00:00:00",
                },
                "sacct_path": str(scontrol),
                "sacct_sha256": scontrol_sha256,
                "scontrol_sha256": scontrol_sha256,
            },
            "job_ids": ["1001"],
            "jobs": [
                {
                    "arm": "rope",
                    "stage": "stage1",
                    "terminal_step": 500_000,
                    "time_limit": "14-00:00:00",
                    "job_id": "1001",
                    "cluster": None,
                    "job_name": job_name,
                    "parent_job_id": None,
                    "sbatch_argv_sha256": "f" * 64,
                    "scheduler_stdout": str(stdout_path),
                    "scheduler_stderr": str(stderr_path),
                    "completion_path": str(completion_path),
                }
            ],
        },
    )
    receipt_path = artifact / "submission-receipt.json"
    _write(receipt_path, receipt)
    commit = _manifest(
        "formal_submission_commit",
        {
            "study_id": "study-seed42",
            "transaction_id": "0123456789abcdef0123456789abcdef",
            "submission_receipt_sha256": receipt["sha256"],
            "transaction_ledger_sha256": ledger["sha256"],
            "protocol_metadata_allowance_bytes": 3_000_000,
            "h100_gate": h100_gate,
            "campaign_binding": campaign_binding,
            "runtime_tools": {"nvidia_smi_sha256": "6" * 64},
            "job_ids": ["1001"],
        },
    )
    _write(artifact / "transaction-committed.json", commit)
    finalized = _manifest(
        "finalized_checkpoint",
        {
            "study_id": "study-seed42",
            "arm": "rope",
            "stage": "stage1",
            "terminal_step": 500_000,
            "upstream_identity": ledger_entry["upstream_identity"],
            "artifact_identity": ledger_entry["artifact_identity"],
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_size": checkpoint_path.stat().st_size,
            "provenance_sha256": digests["provenance"],
            "source_sha256": digests["source"],
            "environment_sha256": digests["environment"],
            "prior_sha256": digests["prior"],
            "architecture_sha256": digests["architecture"],
            "optimizer_sha256": digests["optimizer"],
            "seed_sha256": digests["seed"],
            "treatment_sha256": digests["treatment"],
            "scientific_sha256": digests["scientific"],
            "cohort_protocol_sha256": digests["cohort"],
            "arm_protocol_sha256": digests["arm"],
            "cuda_device_count": 1,
            "max_checkpoint_bytes": 1000,
        },
    )
    _write(stage_dir / "finalized-checkpoint.json", finalized)
    environment = {
        "FORMAL_PROTOCOL_METADATA_ALLOWANCE_BYTES": "3000000",
        "FORMAL_SUBMISSION_RECEIPT": str(receipt_path),
        "FORMAL_TRANSACTION_LEDGER": str(ledger_path),
        "FORMAL_TRANSACTION_LEDGER_SHA256": ledger["sha256"],
        "FORMAL_STUDY_ID": "study-seed42",
        "FORMAL_SEED": "42",
        "FORMAL_TRANSACTION_ID": "0123456789abcdef0123456789abcdef",
        "FORMAL_SOURCE_COMMIT_SHA": "a" * 40,
        "FORMAL_SOURCE_TREE_SHA": "b" * 40,
        "FORMAL_SOURCE_SHA256": digests["source"],
        "FORMAL_ENVIRONMENT_SHA256": digests["environment"],
        "FORMAL_H100_ATTESTATION_SHA256": h100_gate["attestation_sha256"],
        "FORMAL_NVIDIA_SMI_SHA256": "6" * 64,
        "FORMAL_CAMPAIGN_BINDING_SHA256": hashlib.sha256(
            _canonical(campaign_binding)
        ).hexdigest(),
        "FORMAL_GIT_SHA256": git_sha256,
        "CANDIDATE_REPOSITORY": repository_binding["repository_url"],
        "CANDIDATE_REPOSITORY_REF": repository_binding["repository_ref"],
        "FORMAL_REPOSITORY_IDENTITY_SHA256": repository_binding[
            "repository_identity_sha256"
        ],
        "FORMAL_REPOSITORY_QUERY_SHA256": repository_binding["query_sha256"],
        "FORMAL_EXPECTED_JOB_NAME": job_name,
        "FORMAL_SCHEDULER_STDOUT": str(stdout_path),
        "FORMAL_SCHEDULER_STDERR": str(stderr_path),
        "FORMAL_TIME_LIMIT": "14-00:00:00",
        "FORMAL_COMPLETION_EVIDENCE": str(completion_path),
        "FORMAL_RUNTIME_COMPLETION_CEILING_BYTES": "65536",
        "FORMAL_SCONTROL": str(scontrol),
        "FORMAL_SCONTROL_SHA256": scontrol_sha256,
        "FORMAL_RUN_LOG_CEILING_BYTES": "10000",
        "FORMAL_MANIFEST_CEILING_BYTES": "10000",
        "CHECKPOINT_CEILING_BYTES": "1000",
        "FORMAL_ARTIFACT_ROOT": str(artifact),
        "FORMAL_CHECKPOINT_DIR": str(stage_dir),
        "TABICL_EXACT_ROOT": str(exact),
        "SLURM_JOB_ID": "1001",
        "SLURM_JOB_NAME": job_name,
        "SLURM_JOB_PARTITION": "h100",
        "SLURM_CPUS_PER_TASK": "64",
        "SLURM_NNODES": "1",
        "CUDA_VISIBLE_DEVICES": "7",
        "FORMAL_VISIBLE_GPU_NAME": "NVIDIA H100 80GB HBM3",
        "FORMAL_VISIBLE_GPU_UUID": "GPU-fixture",
        "FORMAL_VISIBLE_GPU_DRIVER_VERSION": "570.00",
        "FORMAL_EXPECTED_GPU_MODEL": "NVIDIA H100 80GB HBM3",
        "FORMAL_EXPECTED_DRIVER_VERSION": "570.00",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    return exact, scripts, completion_path, receipt


def test_runtime_binds_receipt_ledger_resources_and_publishes_completion(
    tmp_path, monkeypatch
):
    module = _load()
    exact, scripts, completion_path, receipt = _runtime_fixture(tmp_path, monkeypatch)
    stage = scripts / "formal_train_v2_clf_identity_stage1.sh"
    stage.write_text("#!/bin/bash\nexit 0\n")
    stage.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    monkeypatch.setattr(module, "_verify_formal_environment", lambda _root: None)

    assert module.main(
        ["--exact-root", str(exact), "--mode", "rope", "--stage", "1"]
    ) == 0
    completion = json.loads(completion_path.read_text())
    assert completion["kind"] == "formal_job_completion"
    assert completion["payload"]["submission_receipt_sha256"] == receipt["sha256"]
    assert completion["payload"]["job_id"] == "1001"
    scheduler = completion["payload"]["scheduler"]
    assert {key: scheduler[key] for key in scheduler if key != "query_sha256"} == {
        "partition": "h100",
        "qos": "long",
        "cpus_per_task": 64,
        "memory_mb": 131_072,
        "gpus_per_job": 1,
        "time_limit": "14-00:00:00",
    }
    assert len(scheduler["query_sha256"]) == 64
    assert completion["payload"]["completed"] is True
    assert completion["payload"]["finalized_artifact"]["checkpoint_sha256"] == hashlib.sha256(
        b"checkpoint"
    ).hexdigest()
    assert completion["payload"]["scheduler_logs_terminal_verified"] is False
    assert completion["payload"]["protocol_metadata_allowance_bytes"] == 3_000_000
    assert completion["sha256"] == hashlib.sha256(
        _canonical(
            {
                "schema_version": 1,
                "kind": "formal_job_completion",
                "payload": completion["payload"],
            }
        )
    ).hexdigest()


def test_post_stage_environment_drift_prevents_completion(tmp_path, monkeypatch):
    module = _load()
    exact, scripts, completion_path, _receipt = _runtime_fixture(
        tmp_path, monkeypatch
    )
    stage = scripts / "formal_train_v2_clf_identity_stage1.sh"
    stage.write_text("#!/bin/bash\nexit 0\n")
    stage.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

    def reject(_root):
        raise ValueError("post-stage formal environment verification failed")

    monkeypatch.setattr(module, "_verify_formal_environment", reject)
    with pytest.raises(ValueError, match="post-stage formal environment"):
        module.main(
            ["--exact-root", str(exact), "--mode", "rope", "--stage", "1"]
        )
    assert not completion_path.exists()


def test_main_uses_the_startup_validated_ledger_snapshot(tmp_path, monkeypatch):
    module = _load()
    exact, _scripts, _completion_path, _receipt = _runtime_fixture(
        tmp_path, monkeypatch
    )
    validated = module._validate_runtime(
        mode="rope", stage_index="1", exact_root=exact
    )
    ledger_entries = validated[4]
    ledger_path = Path(os.environ["FORMAL_TRANSACTION_LEDGER"])
    ledger_path.write_text("replaced after startup\n")
    assert ledger_entries[0]["arm"] == "rope"
    assert "_read_manifest(" not in inspect.getsource(module.main)


@pytest.mark.parametrize("runtime_ceiling", ["999", "1001"])
def test_runtime_rejects_checkpoint_ceiling_drift_from_ledger_before_training(
    tmp_path, monkeypatch, runtime_ceiling
):
    module = _load()
    exact, _scripts, completion_path, _receipt = _runtime_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setenv("CHECKPOINT_CEILING_BYTES", runtime_ceiling)

    with pytest.raises(ValueError, match="checkpoint ceiling.*immutable ledger"):
        module._validate_runtime(mode="rope", stage_index="1", exact_root=exact)

    assert not completion_path.exists()


def test_runtime_completion_publication_is_bounded_and_write_once(tmp_path):
    module = _load()
    output = tmp_path / "completion.json"
    with pytest.raises(ValueError, match="byte ceiling"):
        module._publish_no_replace(output, {"payload": "too large"}, max_bytes=1)
    assert not output.exists()


@pytest.mark.parametrize(
    "value",
    (
        "gres:gpu:1",
        "gres/gpu:1",
        "cpu=64,gres:gpu:h100:1,mem=128G",
        "cpu=64,gres/gpu:h100:1,mem=128G",
    ),
)
def test_scheduler_tres_accepts_exactly_one_real_or_legacy_gpu(value):
    module = _load()
    assert module._is_exactly_one_gpu_tres(value) is True


@pytest.mark.parametrize(
    "value",
    (
        "gres:gpu:2",
        "gres/gpu:0",
        "gres:gpu:01",
        "gres:gpu:1,gres:gpu:h100:1",
        "gres/gpu:h100:1,gres/gpu:a100:2",
        "gres:gpu",
        "gpu:1",
        "gres:gpu:h100:1(S:0)",
    ),
)
def test_scheduler_tres_rejects_noncanonical_or_multiple_gpu_claims(value):
    module = _load()
    assert module._is_exactly_one_gpu_tres(value) is False


def test_stable_reader_rejects_symlink_and_path_replacement(
    tmp_path, monkeypatch
):
    module = _load()
    target = tmp_path / "manifest.json"
    target.write_bytes(b"{}\n")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(ValueError, match="bounded regular|following links"):
        module._read_manifest(alias, max_bytes=100, kind="fixture")

    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b"[]\n")
    real_read = module.os.read
    swapped = False

    def replace_after_read(fd, count):
        nonlocal swapped
        raw = real_read(fd, count)
        if raw and not swapped:
            swapped = True
            os.replace(replacement, target)
        return raw

    monkeypatch.setattr(module.os, "read", replace_after_read)
    with pytest.raises(ValueError, match="changed during verification"):
        module._read_manifest(target, max_bytes=100, kind="fixture")


@pytest.mark.parametrize(
    "name,value",
    [
        ("SLURM_JOB_PARTITION", "a10"),
        ("SLURM_CPUS_PER_TASK", "63"),
        ("SLURM_NNODES", "2"),
        ("SLURM_JOB_NAME", "foreign-job"),
        ("CANDIDATE_REPOSITORY_REF", "refs/heads/other"),
        ("FORMAL_REPOSITORY_QUERY_SHA256", "f" * 64),
        ("FORMAL_MANIFEST_CEILING_BYTES", "9999"),
        ("FORMAL_PROTOCOL_METADATA_ALLOWANCE_BYTES", "3000001"),
        ("FORMAL_TIME_LIMIT", "13-23:59:59"),
    ],
)
def test_runtime_fails_closed_on_scheduler_or_identity_drift(
    tmp_path, monkeypatch, name, value
):
    module = _load()
    exact, _scripts, completion_path, _receipt = _runtime_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        module._validate_runtime(mode="rope", stage_index="1", exact_root=exact)
    assert not completion_path.exists()


def test_runtime_rejects_protocol_metadata_allowance_above_ceiling(
    tmp_path, monkeypatch
):
    module = _load()
    exact, _scripts, completion_path, _receipt = _runtime_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setenv(
        "FORMAL_PROTOCOL_METADATA_ALLOWANCE_BYTES",
        str(module.FORMAL_METADATA_CEILING_BYTES + 1),
    )

    with pytest.raises(ValueError, match="outside its allowed range"):
        module._validate_runtime(mode="rope", stage_index="1", exact_root=exact)

    assert not completion_path.exists()


def test_runtime_fails_closed_on_scontrol_time_limit_drift(
    tmp_path, monkeypatch
):
    module = _load()
    exact, _scripts, completion_path, _receipt = _runtime_fixture(
        tmp_path,
        monkeypatch,
        observed_time_limit="13-23:59:59",
    )
    with pytest.raises(ValueError, match="Slurm resources"):
        module._validate_runtime(mode="rope", stage_index="1", exact_root=exact)
    assert not completion_path.exists()


def test_runtime_rejects_stale_receipt_without_transaction_commit(
    tmp_path, monkeypatch
):
    module = _load()
    exact, _scripts, completion_path, receipt = _runtime_fixture(
        tmp_path, monkeypatch
    )
    commit_path = Path(receipt["payload"]["transaction_commit_path"])
    commit_path.unlink()
    wait = module._wait_for_commit_manifest
    monkeypatch.setattr(
        module,
        "_wait_for_commit_manifest",
        lambda path, *, max_bytes: wait(path, max_bytes=max_bytes, attempts=1),
    )
    with pytest.raises(ValueError, match="commit marker"):
        module._validate_runtime(mode="rope", stage_index="1", exact_root=exact)
    assert not completion_path.exists()


def test_success_exit_cannot_publish_completion_for_tampered_checkpoint(
    tmp_path, monkeypatch
):
    module = _load()
    exact, scripts, completion_path, _receipt = _runtime_fixture(tmp_path, monkeypatch)
    stage = scripts / "formal_train_v2_clf_identity_stage1.sh"
    stage.write_text("#!/bin/bash\nexit 0\n")
    stage.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    monkeypatch.setattr(module, "_verify_formal_environment", lambda _root: None)
    checkpoint = (
        tmp_path
        / "artifacts"
        / "study-seed42"
        / "arms"
        / "rope"
        / "stage1"
        / "step-500000.ckpt"
    )
    checkpoint.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="digest or size mismatch"):
        module.main(
            ["--exact-root", str(exact), "--mode", "rope", "--stage", "1"]
        )
    assert not completion_path.exists()
