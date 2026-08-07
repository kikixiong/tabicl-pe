from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import stat
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_formal_overlay.py"
ARMS = ("rope", "temporary", "none")
STAGES = (("stage1", 500_000), ("stage2", 40_000), ("stage3", 10_000))
TIME_LIMIT_BY_STAGE = {
    "stage1": "14-00:00:00",
    "stage2": "3-00:00:00",
    "stage3": "1-00:00:00",
}


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


def _identity_treatment(mode: str, seed: int) -> dict:
    payload = {
        "schema_version": 1,
        "row_identity_mode": mode,
        "identity_rng_seed": seed,
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


def _overlay(tmp_path: Path, seed_value: int = 42) -> dict:
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
            "np_seed": seed_value,
            "torch_seed": seed_value,
            "identity_rng_seed": seed_value,
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
            treatment = _manifest(
                "treatment", _identity_treatment(arm, seed_value)
            )
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
    parent.mkdir(parents=True, exist_ok=True)
    external = tmp_path / "external"
    external.mkdir(parents=True, exist_ok=True)
    python = Path(sys.executable).resolve()
    git = Path("/usr/bin/git")
    scheduler_executable = Path("/usr/bin/true")
    scheduler_sha256 = hashlib.sha256(scheduler_executable.read_bytes()).hexdigest()
    study_id = f"study-a-seed{seed_value}"
    return {
        "schema_version": 1,
        "run_policy": "fresh",
        "seed": seed_value,
        "study_id": study_id,
        "artifact_root": str(parent / study_id),
        "source": {
            "commit_sha": "a" * 40,
            "tree_sha": "b" * 40,
            "manifest_path": str(external / "source-manifest.json"),
            "manifest_sha256": digest["source"],
            "environment_sha256": digest["environment"],
            "candidate_repository": "https://github.com/kikixiong/tabicl-pe.git",
            "candidate_ref": "refs/heads/codex/position-identity-v1",
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
            "git_sha256": hashlib.sha256(git.read_bytes()).hexdigest(),
            "nvidia_smi": "/usr/bin/true",
            "job_work_root": str(tmp_path / "job-work"),
        },
        "scheduler": {
            "partition": "h100",
            "qos": "long",
            "cpus_per_task": 64,
            "memory_mb": 131_072,
            "gpus_per_job": 1,
            "time_limit_by_stage": dict(TIME_LIMIT_BY_STAGE),
            "commands": {
                name: {
                    "path": str(scheduler_executable),
                    "sha256": scheduler_sha256,
                }
                for name in ("sbatch", "scontrol", "scancel", "squeue", "sacct")
            },
        },
        "stages": stages,
    }


def _maximal_terminal_fixture(module, plan):
    maximal_ids = [str(10_000_000_000_000_000_000 + index) for index in range(9)]
    ids_by_key = {}
    submitted = []
    for job, job_id in zip(plan["jobs"], maximal_ids):
        parent_key = job["parent_job"]
        parent_id = ids_by_key[parent_key] if parent_key is not None else None
        ids_by_key[(job["arm"], job["stage"])] = job_id
        submitted.append(
            {
                "job": job,
                "job_id": job_id,
                "cluster": "c" * 64,
                "job_name": f"tabicl-{'f' * 32}-{job['arm']}-s{job['stage_index']}",
                "parent_job_id": parent_id,
            }
        )
    receipt = module._submission_receipt(plan, submitted)
    attestation = module._maximal_terminal_attestation(
        plan, receipt=receipt, submitted=submitted
    )
    return attestation, len(module.canonical_json_bytes(attestation)) + 1


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
        500_000,
        500_000,
        40_000,
        40_000,
        40_000,
        10_000,
        10_000,
        10_000,
    ]
    assert [(job["arm"], job["stage"]) for job in plan["jobs"]] == [
        (arm, stage) for stage, _ in STAGES for arm in ARMS
    ]
    assert all(job["gpus"] == 1 for job in plan["jobs"])
    assert {
        job["stage"]: job["time_limit"] for job in plan["jobs"]
    } == TIME_LIMIT_BY_STAGE
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
            "partition",
        ),
        (
            lambda value: value["scheduler"].update(partition="H100"),
            "partition",
        ),
        (
            lambda value: value["scheduler"].update(qos="short"),
            "qos",
        ),
        (
            lambda value: value["scheduler"].update(cpus_per_task=63),
            "cpus_per_task",
        ),
        (
            lambda value: value["scheduler"].update(memory_mb=131_071),
            "memory_mb",
        ),
        (
            lambda value: value["scheduler"]["time_limit_by_stage"].update(
                stage1="336:00:00"
            ),
            "canonical Slurm duration",
        ),
        (
            lambda value: value["scheduler"]["time_limit_by_stage"].pop("stage3"),
            "time_limit_by_stage",
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
            lambda value: value["source"].update(candidate_repository="--poison"),
            "canonical public GitHub URL",
        ),
        (
            lambda value: value["source"].update(candidate_repository="/tmp/local.git"),
            "canonical public GitHub URL",
        ),
        (
            lambda value: value["source"].update(candidate_ref="refs/heads/other"),
            "canonical training branch",
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


@pytest.mark.parametrize("seed_value", [42, 43, 44])
def test_formal_seed_is_bound_everywhere_without_changing_job_shape(
    tmp_path, seed_value
):
    module = _load()
    plan = module.validate_overlay(
        _overlay(tmp_path, seed_value), exact_root=tmp_path / "exact"
    )

    assert plan["seed"] == seed_value
    assert plan["study_id"].endswith(f"-seed{seed_value}")
    assert Path(plan["artifact_root"]).name == plan["study_id"]
    assert all(
        entry["np_seed"]
        == entry["torch_seed"]
        == entry["identity_rng_seed"]
        == seed_value
        for entry in plan["ledger"]["payload"]["entries"]
    )
    assert all(
        job["exports"]["FORMAL_SEED"] == str(seed_value)
        for job in plan["jobs"]
    )
    assert len(
        {
            job["exports"]["FORMAL_COHORT_PROTOCOL_SHA256"]
            for job in plan["jobs"]
        }
    ) == 3


@pytest.mark.parametrize("seed_value", [41, 45, True, "42", None])
def test_overlay_rejects_seed_outside_frozen_set(tmp_path, seed_value):
    module = _load()
    overlay = _overlay(tmp_path)
    overlay["seed"] = seed_value
    with pytest.raises(ValueError, match="42, 43, or 44"):
        module.validate_overlay(overlay, exact_root=tmp_path / "exact")


def test_overlay_rejects_seed_protocol_or_namespace_mismatch(tmp_path):
    module = _load()

    wrong_protocol = _overlay(tmp_path, 42)
    wrong_protocol["seed"] = 43
    wrong_protocol["study_id"] = "study-a-seed43"
    wrong_protocol["artifact_root"] = str(
        Path(wrong_protocol["artifact_root"]).parent / "study-a-seed43"
    )
    with pytest.raises(ValueError, match="cohort_protocol"):
        module.validate_overlay(wrong_protocol, exact_root=tmp_path / "exact")

    wrong_study = _overlay(tmp_path, 43)
    wrong_study["study_id"] = "study-a-seed42"
    wrong_study["artifact_root"] = str(
        Path(wrong_study["artifact_root"]).parent / "study-a-seed42"
    )
    with pytest.raises(ValueError, match="selected formal seed namespace"):
        module.validate_overlay(wrong_study, exact_root=tmp_path / "exact")

    wrong_root = _overlay(tmp_path, 44)
    wrong_root["artifact_root"] = str(
        Path(wrong_root["artifact_root"]).parent / "different-seed44"
    )
    with pytest.raises(ValueError, match="basename"):
        module.validate_overlay(wrong_root, exact_root=tmp_path / "exact")


def test_seed_changes_both_cohort_and_identity_treatment_protocols(tmp_path):
    module = _load()
    treatments = {
        seed_value: module._identity_treatment("temporary", seed=seed_value)
        for seed_value in (42, 43, 44)
    }
    assert {
        treatment["identity_rng_seed"] for treatment in treatments.values()
    } == {42, 43, 44}
    assert len(
        {treatment["manifest_sha256"] for treatment in treatments.values()}
    ) == 3

    plans = {
        seed_value: module.validate_overlay(
            _overlay(tmp_path / str(seed_value), seed_value),
            exact_root=tmp_path / "exact",
        )
        for seed_value in (42, 43, 44)
    }
    for stage, _ in STAGES:
        cohorts = {
            next(
                entry["cohort_protocol_sha256"]
                for entry in plan["ledger"]["payload"]["entries"]
                if entry["stage"] == stage
            )
            for plan in plans.values()
        }
        assert len(cohorts) == 3
        for arm in ARMS:
            arm_protocols = {
                next(
                    entry["arm_protocol_sha256"]
                    for entry in plan["ledger"]["payload"]["entries"]
                    if entry["stage"] == stage and entry["arm"] == arm
                )
                for plan in plans.values()
            }
            assert len(arm_protocols) == 3


def test_overlay_rejects_identity_treatment_bound_to_a_different_seed(tmp_path):
    module = _load()
    overlay = _overlay(tmp_path, 43)
    stage = overlay["stages"][0]
    wrong_treatment = _manifest("treatment", _identity_treatment("temporary", 42))
    stage["arm_protocol_sha256"]["temporary"] = _manifest(
        "arm_protocol",
        {
            "cohort_protocol_sha256": stage["cohort_protocol_sha256"],
            "mode": "temporary",
            "treatment_sha256": wrong_treatment["sha256"],
        },
    )["sha256"]
    with pytest.raises(ValueError, match="arm_protocol"):
        module.validate_overlay(overlay, exact_root=tmp_path / "exact")


def test_overlay_requires_explicit_seed_key(tmp_path):
    module = _load()
    overlay = _overlay(tmp_path)
    overlay.pop("seed")
    with pytest.raises(ValueError, match="keys mismatch"):
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
        assert exports["FORMAL_RUNTIME_COMPLETION_CEILING_BYTES"] == "65536"
        assert exports["FORMAL_SUBMISSION_RECEIPT"] == str(
            Path(plan["artifact_root"]) / "submission-receipt.json"
        )
        assert Path(exports["FORMAL_SCHEDULER_STDOUT"]).parent.name == "scheduler-logs"
        assert Path(exports["FORMAL_SCHEDULER_STDERR"]).parent.name == "scheduler-logs"
        assert exports["FORMAL_TIME_LIMIT"] == TIME_LIMIT_BY_STAGE[job["stage"]]
        assert Path(exports["FORMAL_COMPLETION_EVIDENCE"]).parent.name == "runtime-completions"
        assert "FORMAL_STAGE_REMAINING_CHECKPOINTS" not in exports
        assert "FORMAL_MONITOR_ROOT" not in exports
        assert "FORMAL_GPU_MONITOR_CEILING_BYTES" not in exports
    for arm in ARMS:
        assert "FORMAL_PARENT_CHECKPOINT" not in jobs[(arm, "stage1")]["exports"]
        assert jobs[(arm, "stage2")]["parent_job"] == (arm, "stage1")
        assert jobs[(arm, "stage3")]["parent_job"] == (arm, "stage2")
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


def test_protocol_metadata_budget_reserves_nine_completions_terminal_logs_and_full_names(
    tmp_path,
):
    module = _load()
    plan = module.validate_overlay(_overlay(tmp_path), exact_root=tmp_path / "exact")
    maximal_ids = [str(10_000_000_000_000_000_000 + index) for index in range(9)]
    ids_by_key = {}
    submitted = []
    for job, job_id in zip(plan["jobs"], maximal_ids):
        parent_key = job["parent_job"]
        parent_id = ids_by_key[parent_key] if parent_key is not None else None
        ids_by_key[(job["arm"], job["stage"])] = job_id
        submitted.append(
            {
                "job": job,
                "job_id": job_id,
                "cluster": "cluster-max",
                "job_name": f"tabicl-{'f' * 32}-{job['arm']}-s{job['stage_index']}",
                "parent_job_id": parent_id,
            }
        )
    receipt = module._submission_receipt(plan, submitted)
    commit = module._submission_commit(
        plan, receipt=receipt, submitted=submitted
    )
    recovery = max(
        (
            module._rollback_record(
                plan,
                accepted=maximal_ids,
                cancelled=(),
                remaining=tuple(reversed(maximal_ids)),
                reason=reason,
                ledger_published=True,
                unresolved_job_names=(f"tabicl-{'f' * 32}-rope-s1",),
            )
            for reason in (
                "sbatch_failed",
                "ledger_publication_failed",
                "receipt_publication_failed",
                "release_failed",
            )
        ),
        key=lambda value: len(module.canonical_json_bytes(value)),
    )
    controller_only = module.TRANSACTION_JOURNAL_CEILING_BYTES + sum(
        len(module.canonical_json_bytes(value)) + 1
        for value in (plan["ledger"], receipt, commit, recovery)
    )
    plan["capacity"]["protocol_metadata_allowance_bytes"] = controller_only
    with pytest.raises(ValueError, match="protocol metadata"):
        module._validate_protocol_metadata_budget(plan)
    plan["capacity"]["protocol_metadata_allowance_bytes"] = (
        controller_only
        + 9 * module.RUNTIME_COMPLETION_CEILING_BYTES
        + module.TERMINAL_LOG_ATTESTATION_CEILING_BYTES
    )
    module._validate_protocol_metadata_budget(plan)
    assert all("f" * 32 in item["job_name"] for item in submitted)


def test_terminal_attestation_preflight_synthesizes_full_maximum_and_exact_boundary(
    tmp_path, monkeypatch
):
    module = _load()
    plan = module.validate_overlay(_overlay(tmp_path), exact_root=tmp_path / "exact")
    attestation, required = _maximal_terminal_fixture(module, plan)
    payload = attestation["payload"]

    assert payload["path_format"] == "artifact_root_relative_posix_v1"
    assert len(payload["jobs"]) == 9
    assert len(payload["terminal_queries"][0]["job_ids"]) == 9
    assert all(
        not Path(path).is_absolute()
        for job in payload["jobs"]
        for path in (
            job["completion"]["path"],
            *(log["path"] for log in job["scheduler_logs"]),
            job["finalized_artifact"]["binding"]["checkpoint_path"],
            job["finalized_artifact"]["binding"]["finalized_manifest_path"],
        )
    )
    assert required <= module.TERMINAL_LOG_ATTESTATION_CEILING_BYTES

    monkeypatch.setattr(module, "TERMINAL_LOG_ATTESTATION_CEILING_BYTES", required)
    module._validate_protocol_metadata_budget(plan)
    monkeypatch.setattr(
        module, "TERMINAL_LOG_ATTESTATION_CEILING_BYTES", required - 1
    )
    with pytest.raises(ValueError, match="terminal scheduler-log attestation maximum"):
        module._validate_protocol_metadata_budget(plan)


def test_terminal_attestation_preflight_is_independent_of_artifact_parent_length(
    tmp_path,
):
    module = _load()
    short_plan = module.validate_overlay(
        _overlay(tmp_path / "short"), exact_root=tmp_path / "exact"
    )
    long_parent = tmp_path / "long"
    for index in range(12):
        long_parent /= f"parent-{index:02d}-" + "x" * 220
    long_plan = module.validate_overlay(
        _overlay(long_parent), exact_root=tmp_path / "exact"
    )

    _short_attestation, short_size = _maximal_terminal_fixture(module, short_plan)
    long_attestation, long_size = _maximal_terminal_fixture(module, long_plan)
    assert long_size == short_size
    assert str(long_parent) not in module.canonical_json_bytes(long_attestation).decode()


def test_nofollow_directory_open_rejects_a_symlinked_parent(tmp_path):
    module = _load()
    physical = tmp_path / "physical"
    physical.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(physical, target_is_directory=True)

    with pytest.raises(OSError):
        module._open_directory_nofollow(alias, where="test root")


def test_scheduler_commands_are_absolute_digest_bound_regular_executables(tmp_path):
    module = _load()
    plan = module.validate_overlay(_overlay(tmp_path), exact_root=tmp_path / "exact")
    commands, scheduler_env = module._scheduler_commands(plan)
    assert set(commands) == {"sbatch", "scontrol", "scancel", "squeue", "sacct"}
    assert set(commands.values()) == {"/usr/bin/true"}
    assert scheduler_env["PATH"] == "/usr/bin:/bin"

    tampered = copy.deepcopy(plan)
    tampered["scheduler"]["commands"]["sbatch"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module._scheduler_commands(tampered)

    alias = tmp_path / "scheduler-alias"
    alias.symlink_to("/usr/bin/true")
    symlinked = copy.deepcopy(plan)
    symlinked["scheduler"]["commands"]["sbatch"] = {
        "path": str(alias),
        "sha256": hashlib.sha256(Path("/usr/bin/true").read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="no-follow executable"):
        module._scheduler_commands(symlinked)


def test_job_work_root_must_be_disjoint_from_exact_and_artifact_roots(tmp_path):
    module = _load()
    exact = tmp_path / "exact"
    inside_exact = _overlay(tmp_path / "inside-exact")
    inside_exact["runtime"]["job_work_root"] = str(exact / "work")
    with pytest.raises(ValueError, match="disjoint from exact_root"):
        module.validate_overlay(inside_exact, exact_root=exact)

    inside_artifacts = _overlay(tmp_path / "inside-artifacts")
    inside_artifacts["runtime"]["job_work_root"] = str(
        Path(inside_artifacts["artifact_root"]) / "work"
    )
    with pytest.raises(ValueError, match="disjoint from artifact_root"):
        module.validate_overlay(inside_artifacts, exact_root=exact)


def test_imported_submit_default_never_rewrites_host_signal_handlers(tmp_path):
    module = _load()
    before = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    with pytest.raises(Exception):
        module.submit_overlay(
            tmp_path / "missing.json",
            exact_root=tmp_path,
        )
    assert {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
    } == before
