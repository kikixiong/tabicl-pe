from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_formal_overlay.py"
ARMS = ("rope", "temporary", "none")
STAGES = (("stage1", 500_000), ("stage2", 40_000), ("stage3", 10_000))


def _load():
    spec = importlib.util.spec_from_file_location("formal_overlay", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _manifest(kind: str, payload: dict) -> dict:
    body = {"schema_version": 1, "kind": kind, "payload": payload}
    return {**body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def _identity_treatment(mode: str) -> dict:
    payload = {
        "schema_version": 1,
        "row_identity_mode": mode,
        "identity_rng_seed": 42,
        "seed_policy": "sha256-domain-separated-base-seed-and-rank-v1",
        "sampler_version": (
            "tabicl-temporary-identity/randperm-cpu-v1"
            if mode == "temporary"
            else None
        ),
        "world_size": 1,
    }
    return {
        **payload,
        "manifest_sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
    }


def _overlay(tmp_path: Path) -> dict:
    digest = {
        "source": "1" * 64,
        "environment": "2" * 64,
        "prior": "3" * 64,
        "architecture": "4" * 64,
        "optimizer": "5" * 64,
        "scientific": "6" * 64,
    }
    seed = _manifest(
        "seed",
        {
            "np_seed": 42,
            "torch_seed": 42,
            "identity_rng_seed": 42,
            "world_size": 1,
        },
    )
    stages = []
    for stage, budget in STAGES:
        cohort = _manifest(
            "cohort_protocol",
            {
                "stage": stage,
                "terminal_step": budget,
                "source_sha256": digest["source"],
                "environment_sha256": digest["environment"],
                "architecture_sha256": digest["architecture"],
                "prior_sha256": digest["prior"],
                "optimizer_sha256": digest["optimizer"],
                "seed_sha256": seed["sha256"],
                "scientific_config_sha256": digest["scientific"],
            },
        )
        arm_protocol = {}
        for arm in ARMS:
            treatment = _manifest("treatment", _identity_treatment(arm))
            arm_protocol[arm] = _manifest(
                "arm_protocol",
                {
                    "cohort_protocol_sha256": cohort["sha256"],
                    "mode": arm,
                    "treatment_sha256": treatment["sha256"],
                },
            )["sha256"]
        stages.append(
            {
                "stage": stage,
                "terminal_step": budget,
                "prior_sha256": digest["prior"],
                "architecture_sha256": digest["architecture"],
                "optimizer_sha256": digest["optimizer"],
                "scientific_sha256": digest["scientific"],
                "cohort_protocol_sha256": cohort["sha256"],
                "arm_protocol_sha256": arm_protocol,
            }
        )
    parent = tmp_path / "artifacts"
    parent.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    python = Path(sys.executable).resolve()
    git = Path("/usr/bin/git")
    return {
        "schema_version": 1,
        "run_policy": "fresh",
        "study_id": "study-a",
        "artifact_root": str(parent / "study-a"),
        "source": {
            "commit_sha": "a" * 40,
            "tree_sha": "b" * 40,
            "manifest_path": str(external / "source-manifest.json"),
            "manifest_sha256": digest["source"],
            "environment_sha256": digest["environment"],
            "candidate_repository": str(tmp_path / "candidate.git"),
        },
        "smoke": {
            "attestation_path": str(external / "h100-smoke.json"),
            "expected_sha256": "7" * 64,
            "expected_gpu_model": "NVIDIA H100 80GB HBM3",
        },
        "capacity": {
            "checkpoint_ceiling_bytes": 2_000_000_000,
            "durable_log_allowance_bytes": 1_000_000_000,
            "run_log_ceiling_bytes": 20_000_000,
            "attestation_ceiling_bytes": 2_000_000,
            "manifest_ceiling_bytes": 2_000_000,
            "protocol_metadata_allowance_bytes": 100_000_000,
        },
        "runtime": {
            "python": str(python),
            "git": str(git),
            "nvidia_smi": "/usr/bin/true",
            "job_work_root": str(tmp_path / "job-work"),
        },
        "scheduler": {
            "partition": "h100",
            "qos": "long",
            "cpus_per_task": 32,
            "memory_mb": 131_072,
            "gpus_per_job": 1,
        },
        "stages": stages,
    }


def test_valid_overlay_builds_exact_nine_entry_ledger_and_fixed_job_plan(tmp_path):
    module = _load()
    plan = module.validate_overlay(_overlay(tmp_path), exact_root=tmp_path / "exact")

    assert plan["run_policy"] == "fresh"
    assert len(plan["jobs"]) == len(plan["ledger"]["payload"]["entries"]) == 9
    assert {(job["arm"], job["stage"]) for job in plan["jobs"]} == {
        (arm, stage) for arm in ARMS for stage, _ in STAGES
    }
    assert [job["terminal_step"] for job in plan["jobs"]] == [
        500_000,
        40_000,
        10_000,
    ] * 3
    assert all(job["gpus"] == 1 for job in plan["jobs"])
    entries = plan["ledger"]["payload"]["entries"]
    assert all(
        set(entry)
        == {
            "arm",
            "stage",
            "terminal_step",
            "upstream_identity",
            "artifact_identity",
            "checkpoint_relpath",
            "finalized_manifest_relpath",
            "np_seed",
            "torch_seed",
            "identity_rng_seed",
            "world_size",
            "cuda_device_count",
            "max_checkpoint_bytes",
            "source_sha256",
            "environment_sha256",
            "prior_sha256",
            "architecture_sha256",
            "optimizer_sha256",
            "scientific_sha256",
            "cohort_protocol_sha256",
            "arm_protocol_sha256",
        }
        for entry in entries
    )
    assert all(
        "checkpoint_sha256" not in entry and "parent_checkpoint_sha256" not in entry
        for entry in entries
    )
    assert len({entry["checkpoint_relpath"] for entry in entries}) == 9
    assert len({entry["finalized_manifest_relpath"] for entry in entries}) == 9
    assert plan["ledger"]["sha256"] == hashlib.sha256(
        _canonical(
            {
                "schema_version": 1,
                "kind": "transaction_ledger",
                "payload": plan["ledger"]["payload"],
            }
        )
    ).hexdigest()


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value.update(run_policy="resume"), "fresh"),
        (lambda value: value.update(schema_version=True), "schema_version"),
        (
            lambda value: value["scheduler"].update(gpus_per_job=2),
            "exactly one H100",
        ),
        (
            lambda value: value["scheduler"].update(gpus_per_job=True),
            "exactly one H100",
        ),
        (
            lambda value: value["scheduler"].update(partition="a10"),
            "H100 partition",
        ),
        (
            lambda value: value["stages"][0].update(terminal_step=499_999),
            "budget",
        ),
        (
            lambda value: value["stages"].append(copy.deepcopy(value["stages"][0])),
            "stages",
        ),
        (
            lambda value: value["stages"][0]["arm_protocol_sha256"].pop("none"),
            "arms",
        ),
        (
            lambda value: value["stages"][1].update(
                cohort_protocol_sha256="f" * 64
            ),
            "cohort_protocol",
        ),
        (
            lambda value: value["stages"][2]["arm_protocol_sha256"].update(
                temporary="e" * 64
            ),
            "arm_protocol",
        ),
        (
            lambda value: value["source"].update(environment_sha256="bad"),
            "environment",
        ),
        (
            lambda value: value["smoke"].update(expected_sha256="bad"),
            "smoke",
        ),
        (
            lambda value: value["runtime"].update(nvidia_smi="nvidia-smi"),
            "nvidia_smi",
        ),
        (
            lambda value: value["capacity"].update(
                protocol_metadata_allowance_bytes=1
            ),
            "protocol metadata",
        ),
    ],
)
def test_overlay_rejects_noncanonical_or_undeclared_drift(tmp_path, mutate, match):
    module = _load()
    overlay = _overlay(tmp_path)
    mutate(overlay)
    with pytest.raises(ValueError, match=match):
        module.validate_overlay(overlay, exact_root=tmp_path / "exact")


def test_job_exports_are_an_explicit_allowlist_with_exact_parent_chain(tmp_path):
    module = _load()
    plan = module.validate_overlay(_overlay(tmp_path), exact_root=tmp_path / "exact")
    jobs = {(job["arm"], job["stage"]): job for job in plan["jobs"]}

    for job in jobs.values():
        exports = job["exports"]
        assert exports["RUN_POLICY"] == "fresh"
        assert exports["NUM_GPUS"] == "1"
        assert exports["PYTHONNOUSERSITE"] == "1"
        assert exports["PYTHONPATH"] == str(tmp_path / "exact" / "src")
        assert exports["GIT"] == "/usr/bin/git"
        assert exports["NVIDIA_SMI"] == "/usr/bin/true"
        assert "PYTHONHOME" not in exports
        assert "ALL" not in exports
        assert exports["FORMAL_TRANSACTION_LEDGER_SHA256"] == plan["ledger"][
            "sha256"
        ]
        assert exports["FORMAL_ATTESTATION_CEILING_BYTES"] == "2000000"
        assert exports["FORMAL_MANIFEST_CEILING_BYTES"] == "2000000"
        assert exports["FORMAL_PROTOCOL_METADATA_ALLOWANCE_BYTES"] == "100000000"
        assert "FORMAL_STAGE_REMAINING_CHECKPOINTS" not in exports
        assert "FORMAL_MONITOR_ROOT" not in exports
        assert "FORMAL_GPU_MONITOR_CEILING_BYTES" not in exports
    assert "FORMAL_PARENT_CHECKPOINT" not in jobs[("rope", "stage1")]["exports"]
    assert jobs[("rope", "stage2")]["parent_job"] == ("rope", "stage1")
    assert jobs[("rope", "stage3")]["parent_job"] == ("rope", "stage2")
    stage2 = jobs[("rope", "stage2")]["exports"]
    parent_entry = next(
        entry
        for entry in plan["ledger"]["payload"]["entries"]
        if entry["arm"] == "rope" and entry["stage"] == "stage1"
    )
    assert stage2["FORMAL_PARENT_CHECKPOINT"] == str(
        Path(plan["artifact_root"]) / parent_entry["checkpoint_relpath"]
    )
    assert stage2["FORMAL_PARENT_FINALIZED_MANIFEST"] == str(
        Path(plan["artifact_root"]) / parent_entry["finalized_manifest_relpath"]
    )


@pytest.mark.parametrize("fault", ["write", "fsync", "rename", "dir_fsync"])
def test_ledger_publication_fault_never_leaves_a_reported_success(tmp_path, fault):
    module = _load()
    plan = module.validate_overlay(_overlay(tmp_path), exact_root=tmp_path / "exact")
    root = Path(plan["artifact_root"])
    root.mkdir()

    with pytest.raises(OSError, match="injected"):
        module.publish_ledger(plan, fault=fault)
    if (root / "transaction-ledger.json").exists():
        published = json.loads((root / "transaction-ledger.json").read_text())
        assert published == plan["ledger"]


def test_ledger_publication_is_write_once_canonical_and_read_only(tmp_path):
    module = _load()
    plan = module.validate_overlay(_overlay(tmp_path), exact_root=tmp_path / "exact")
    root = Path(plan["artifact_root"])
    root.mkdir()
    path = module.publish_ledger(plan)

    assert path == root / "transaction-ledger.json"
    assert path.read_bytes() == _canonical(plan["ledger"]) + b"\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        module.publish_ledger(plan)


def test_rollback_incomplete_record_is_durable_canonical_and_truthful(tmp_path):
    module = _load()
    plan = module.validate_overlay(_overlay(tmp_path), exact_root=tmp_path / "exact")
    root = Path(plan["artifact_root"])
    root.mkdir()
    path = module.publish_rollback_incomplete(
        plan,
        accepted_job_ids=("101", "102", "103"),
        cancelled_job_ids=("103",),
        remaining_job_ids=("102", "101"),
        reason="release_failed",
        ledger_published=True,
    )
    record = json.loads(path.read_text())
    assert record["kind"] == "rollback_incomplete"
    assert record["payload"]["accepted_job_ids"] == ["101", "102", "103"]
    assert record["payload"]["cancelled_job_ids"] == ["103"]
    assert record["payload"]["remaining_job_ids"] == ["102", "101"]
    assert record["payload"]["ledger_published"] is True
    assert stat.S_IMODE(path.stat().st_mode) == 0o444


def test_scheduler_job_ids_are_bounded_decimal_values():
    module = _load()
    assert module._job_ids(("1", "18446744073709551615"), "jobs") == [
        "1",
        "18446744073709551615",
    ]
    with pytest.raises(ValueError, match="invalid job ID"):
        module._job_ids(("1" * 21,), "jobs")


def test_nofollow_directory_open_rejects_a_symlinked_parent(tmp_path):
    module = _load()
    physical = tmp_path / "physical"
    physical.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(physical, target_is_directory=True)

    with pytest.raises(OSError):
        module._open_directory_nofollow(alias, where="test root")
