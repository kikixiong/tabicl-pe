#!/usr/bin/env python3
"""Plan, or explicitly submit, one fail-closed TALENT checkpoint pair campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from pe_mechanism.talent_full_suite import (
    OPERATION_JOURNAL_KIND,
    PLAN_KIND,
    RELEASE_KIND,
    ROLLBACK_KIND,
    SUBMISSION_KIND,
    absolute_path,
    atomic_json,
    canonical_json_sha256,
    canary_dataset_records,
    frozen_discovery_roster,
    load_json_object_with_sha256,
    load_private_run_config,
    require_disjoint_output,
    self_hashed_document,
    sha256_file,
    validate_checkpoint_pairs,
    validate_shard_plan,
    verify_clean_detached_git,
)


CAPACITY_WARNING_BYTES = 22 * 1024**3
CAPACITY_HARD_RESERVE_BYTES = 20 * 1024**3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--talent-root", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--shard-plan", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--capacity-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--expected-analysis-sha", required=True)
    parser.add_argument("--expected-model-sha", required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def _available_bytes(path: Path) -> int:
    values = os.statvfs(path)
    return int(values.f_bavail * values.f_frsize)


def _capacity_contract(
    *, output_available: int, capacity_available: int, scratch_available: int
) -> dict[str, Any]:
    available = {
        "output": output_available,
        "capacity": capacity_available,
        "scratch": scratch_available,
    }
    statuses = {
        name: (
            "warning_below_22_gib"
            if value < CAPACITY_WARNING_BYTES
            else "ok_at_or_above_22_gib"
        )
        for name, value in available.items()
    }
    return {
        "output_pre_submit_available_bytes": output_available,
        "capacity_root_pre_submit_available_bytes": capacity_available,
        "scratch_root_pre_submit_available_bytes": scratch_available,
        "warning_threshold_bytes": CAPACITY_WARNING_BYTES,
        "hard_reserve_bytes": CAPACITY_HARD_RESERVE_BYTES,
        "warning_triggered": any(
            value < CAPACITY_WARNING_BYTES for value in available.values()
        ),
        "pre_submit_status_by_filesystem": statuses,
    }


def _require_hard_capacity(contract: Mapping[str, Any], *, context: str) -> None:
    if min(
        contract["output_pre_submit_available_bytes"],
        contract["capacity_root_pre_submit_available_bytes"],
        contract["scratch_root_pre_submit_available_bytes"],
    ) < CAPACITY_HARD_RESERVE_BYTES:
        raise RuntimeError(f"TALENT campaign fell below the 20 GiB {context} reserve")


def _export_argument(values: Mapping[str, str]) -> str:
    for name, value in values.items():
        if any(character in value for character in (",", "\n", "\r")):
            raise ValueError(f"Slurm export value is unsafe: {name}")
    return "ALL," + ",".join(f"{name}={value}" for name, value in values.items())


def _command_plan(
    *,
    sbatch: str,
    analysis_root: Path,
    model_root: Path,
    talent_root: Path,
    run_config: Path,
    shard_plan: Path,
    output_root: Path,
    scratch_root: Path,
    python: Path,
    analysis_sha: str,
    model_sha: str,
    run_config_sha256: str,
    shard_plan_sha256: str,
    canary_dependency: str = "CANARY_JOB_ID",
    array_dependency: str = "ARRAY_JOB_ID",
) -> dict[str, list[str]]:
    gpu_wrapper = analysis_root / "analysis/pe_mechanism/scripts/slurm_talent_paired_full_suite.sh"
    aggregate_wrapper = (
        analysis_root
        / "analysis/pe_mechanism/scripts/slurm_talent_paired_aggregate.sh"
    )
    shared = {
        "PE_TALENT_ANALYSIS_ROOT": str(analysis_root),
        "PE_TALENT_MODEL_ROOT": str(model_root),
        "PE_TALENT_DATA_ROOT": str(talent_root),
        "PE_TALENT_RUN_CONFIG": str(run_config),
        "PE_TALENT_SHARD_PLAN": str(shard_plan),
        "PE_TALENT_PYTHON": str(python),
        "PE_TALENT_EXPECTED_ANALYSIS_SHA": analysis_sha,
        "PE_TALENT_EXPECTED_MODEL_SHA": model_sha,
        "PE_TALENT_EXPECTED_RUN_CONFIG_SHA256": run_config_sha256,
        "PE_TALENT_EXPECTED_SHARD_PLAN_SHA256": shard_plan_sha256,
        "PE_TALENT_SCRATCH_ROOT": str(scratch_root),
        "PE_TALENT_GPU_CSV_ROOT": str(output_root / "monitor"),
    }
    canary_env = {
        **shared,
        "PE_TALENT_OUTPUT_ROOT": str(output_root / "canary"),
        "PE_TALENT_CANARY": "1",
    }
    full_env = {
        **shared,
        "PE_TALENT_OUTPUT_ROOT": str(output_root / "shards"),
        "PE_TALENT_CANARY": "0",
    }
    aggregate_env = {
        "PE_TALENT_ANALYSIS_ROOT": str(analysis_root),
        "PE_TALENT_RUN_CONFIG": str(run_config),
        "PE_TALENT_SHARD_PLAN": str(shard_plan),
        "PE_TALENT_OUTPUT_ROOT": str(output_root),
        "PE_TALENT_PYTHON": str(python),
        "PE_TALENT_EXPECTED_ANALYSIS_SHA": analysis_sha,
        "PE_TALENT_EXPECTED_RUN_CONFIG_SHA256": run_config_sha256,
        "PE_TALENT_EXPECTED_SHARD_PLAN_SHA256": shard_plan_sha256,
    }
    return {
        "canary": [
            sbatch,
            "--parsable",
            "--hold",
            "--output",
            str(output_root / "logs/canary.out"),
            "--error",
            str(output_root / "logs/canary.err"),
            "--export",
            _export_argument(canary_env),
            str(gpu_wrapper),
        ],
        "array": [
            sbatch,
            "--parsable",
            "--array=0-7%2",
            f"--dependency=afterok:{canary_dependency}",
            "--output",
            str(output_root / "logs/full-%a.out"),
            "--error",
            str(output_root / "logs/full-%a.err"),
            "--export",
            _export_argument(full_env),
            str(gpu_wrapper),
        ],
        "aggregate": [
            sbatch,
            "--parsable",
            f"--dependency=afterok:{array_dependency}",
            "--output",
            str(output_root / "logs/aggregate.out"),
            "--error",
            str(output_root / "logs/aggregate.err"),
            "--export",
            _export_argument(aggregate_env),
            str(aggregate_wrapper),
        ],
    }


class _DurableJournal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.campaign_token = secrets.token_hex(16)
        self.events: list[dict[str, Any]] = []
        self.sha256 = ""
        self._write()

    def _write(self) -> None:
        document = self_hashed_document(
            OPERATION_JOURNAL_KIND,
            {
                "schema_version": 1,
                "campaign_token": self.campaign_token,
                "events": self.events,
            },
        )
        atomic_json(self.path, document)
        self.sha256 = document["sha256"]

    def append(self, event: Mapping[str, Any]) -> dict[str, Any]:
        record = {
            "sequence": len(self.events) + 1,
            "at_unix_seconds": time.time(),
            **dict(event),
        }
        self.events.append(record)
        self._write()
        return record


def _parse_job_id(stdout: str) -> str:
    token = stdout.strip().split(";", 1)[0]
    if not token.isdigit() or int(token) <= 0:
        raise RuntimeError("sbatch returned an invalid job ID")
    return token


def _submit_job(
    command: Sequence[str], *, role: str, journal: _DurableJournal
) -> str:
    command_sha256 = canonical_json_sha256(list(command))
    journal.append(
        {
            "event": "submit_started",
            "job_role": role,
            "command_sha256": command_sha256,
        }
    )
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        journal.append(
            {
                "event": "submit_failed",
                "job_role": role,
                "command_sha256": command_sha256,
                "returncode": completed.returncode,
            }
        )
        raise subprocess.CalledProcessError(completed.returncode, list(command))
    try:
        job_id = _parse_job_id(completed.stdout)
    except RuntimeError:
        journal.append(
            {
                "event": "submit_returned_invalid_id",
                "job_role": role,
                "command_sha256": command_sha256,
            }
        )
        raise
    journal.append(
        {
            "event": "submit_succeeded",
            "job_role": role,
            "command_sha256": command_sha256,
            "job_id": job_id,
        }
    )
    return job_id


def _accounting_state(job_id: str) -> tuple[list[str], str | None]:
    queue = subprocess.run(
        ["squeue", "--noheader", "--jobs", job_id, "--format=%T"],
        check=False,
        capture_output=True,
        text=True,
    )
    if queue.returncode != 0:
        return ["QUERY_FAILED"], None
    active_states = [line.strip().upper() for line in queue.stdout.splitlines() if line.strip()]
    accounting = subprocess.run(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            "--jobs",
            job_id,
            "--format=JobIDRaw,State",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if accounting.returncode != 0:
        return active_states, None
    state = None
    for line in accounting.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 2 and fields[0].strip() == job_id:
            state = fields[1].strip().upper().split()[0].split("+")[0]
            break
    return active_states, state


def _cancel_and_verify(
    *, job_id: str, role: str, journal: _DurableJournal
) -> dict[str, Any]:
    journal.append({"event": "cancel_started", "job_role": role, "job_id": job_id})
    cancellation = subprocess.run(
        ["scancel", job_id], check=False, capture_output=True, text=True
    )
    journal.append(
        {
            "event": "cancel_returned",
            "job_role": role,
            "job_id": job_id,
            "returncode": cancellation.returncode,
        }
    )
    active_states: list[str] = []
    accounting_state: str | None = None
    for _ in range(10):
        active_states, accounting_state = _accounting_state(job_id)
        if not active_states and accounting_state is not None:
            break
        time.sleep(1)
    verified = (
        cancellation.returncode == 0
        and not active_states
        and accounting_state == "CANCELLED"
    )
    journal.append(
        {
            "event": "cancel_verified" if verified else "cancel_unverified",
            "job_role": role,
            "job_id": job_id,
            "active_states": active_states,
            "accounting_state": accounting_state,
        }
    )
    return {
        "cancel_returncode": cancellation.returncode,
        "active_states": active_states,
        "accounting_state": accounting_state,
        "verified_cancelled": verified,
    }


def _rollback_jobs(
    jobs: Mapping[str, str], *, journal: _DurableJournal, failure_type: str
) -> dict[str, Any]:
    outcomes = {
        role: _cancel_and_verify(job_id=job_id, role=role, journal=journal)
        for role, job_id in reversed(list(jobs.items()))
    }
    verified = bool(outcomes) and all(
        outcome["verified_cancelled"] for outcome in outcomes.values()
    )
    return self_hashed_document(
        ROLLBACK_KIND,
        {
            "schema_version": 1,
            "failure_type": failure_type,
            "rollback_status": (
                "verified_cancelled" if verified else "unverified_or_incomplete"
            ),
            "rollback_verified": verified,
            "job_outcomes_by_role": outcomes,
            "operation_journal_sha256": journal.sha256,
        },
    )


def _verify_inputs_unchanged(
    *,
    analysis_root: Path,
    model_root: Path,
    analysis_sha: str,
    model_sha: str,
    config: Mapping[str, Any],
    plan_path: Path,
    plan_file_sha: str,
) -> None:
    verify_clean_detached_git(analysis_root, expected_sha=analysis_sha)
    verify_clean_detached_git(model_root, expected_sha=model_sha)
    if sha256_file(config["path"]) != config["config_sha256"]:
        raise RuntimeError("run config changed after validation")
    if sha256_file(plan_path) != plan_file_sha:
        raise RuntimeError("shard plan changed after validation")
    for pair in config["pairs"]:
        for arm in pair["arms"]:
            if sha256_file(arm["checkpoint"]) != arm["checkpoint_sha256"]:
                raise RuntimeError("checkpoint changed after validation")
        for receipt in pair["lineage_receipts"]:
            if sha256_file(receipt["path"]) != receipt["sha256"]:
                raise RuntimeError("lineage receipt changed after validation")


def _run(args: argparse.Namespace) -> int:
    analysis_root = absolute_path(
        args.analysis_root, name="analysis_root", directory=True
    )
    model_root = absolute_path(args.model_root, name="model_root", directory=True)
    talent_root = absolute_path(args.talent_root, name="talent_root", directory=True)
    scratch_root = absolute_path(
        args.scratch_root, name="scratch_root", directory=True
    )
    capacity_root = absolute_path(
        args.capacity_root, name="capacity_root", directory=True
    )
    python = absolute_path(args.python, name="python")
    if not os.access(python, os.X_OK):
        raise ValueError("python must be executable")
    analysis_sha = verify_clean_detached_git(
        analysis_root, expected_sha=args.expected_analysis_sha
    )
    model_sha = verify_clean_detached_git(model_root, expected_sha=args.expected_model_sha)
    run_config_path = absolute_path(args.run_config, name="run_config")
    config = load_private_run_config(run_config_path)
    if len(config["pairs"]) != 1:
        raise ValueError("one TALENT submission accepts exactly one checkpoint pair")
    if config["pairs"][0]["training_source_commit"] != model_sha:
        raise ValueError("pair training source differs from the model runtime")
    checkpoint_contract = validate_checkpoint_pairs(config)
    plan_path = absolute_path(args.shard_plan, name="shard_plan")
    plan_document, plan_file_sha = load_json_object_with_sha256(
        plan_path, name="TALENT shard plan"
    )
    plan = validate_shard_plan(
        plan_document, expected_roster=frozen_discovery_roster(analysis_root)
    )
    if plan_document.get("kind") != PLAN_KIND or plan.get("analysis_sha") != analysis_sha:
        raise ValueError("TALENT shard plan is from another analysis commit")
    if plan["shard_count"] != 8:
        raise ValueError("TALENT production submission requires exactly eight shards")
    canary = canary_dataset_records(plan)
    output_value = Path(args.output_root)
    if not output_value.is_absolute():
        raise ValueError("output_root must be absolute")
    output_root = output_value.parent.resolve(strict=True) / output_value.name
    if output_root.exists() or output_root.is_symlink():
        raise ValueError("fresh TALENT output_root must be absent")
    protected = [
        analysis_root,
        model_root,
        talent_root,
        scratch_root,
        run_config_path,
        plan_path,
        python,
        *(arm["checkpoint"] for arm in config["pairs"][0]["arms"]),
        *(receipt["path"] for receipt in config["pairs"][0]["lineage_receipts"]),
    ]
    require_disjoint_output(output_root, protected, name="TALENT campaign output")
    output_available = _available_bytes(output_root.parent)
    capacity_available = _available_bytes(capacity_root)
    scratch_available = _available_bytes(scratch_root)
    capacity_contract = _capacity_contract(
        output_available=output_available,
        capacity_available=capacity_available,
        scratch_available=scratch_available,
    )
    _require_hard_capacity(capacity_contract, context="pre-submit")
    if capacity_contract["warning_triggered"]:
        warning_filesystems = [
            name
            for name, status in capacity_contract[
                "pre_submit_status_by_filesystem"
            ].items()
            if status == "warning_below_22_gib"
        ]
        print(
            json.dumps(
                {
                    "warning": "capacity_below_22_gib",
                    "filesystem_roles": warning_filesystems,
                    "hard_reserve_gib": 20,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
    sbatch = shutil.which("sbatch") or "sbatch"
    dry_plan = _command_plan(
        sbatch=sbatch,
        analysis_root=analysis_root,
        model_root=model_root,
        talent_root=talent_root,
        run_config=run_config_path,
        shard_plan=plan_path,
        output_root=output_root,
        scratch_root=scratch_root,
        python=python,
        analysis_sha=analysis_sha,
        model_sha=model_sha,
        run_config_sha256=config["config_sha256"],
        shard_plan_sha256=plan_file_sha,
    )
    command_hashes = {
        role: canonical_json_sha256(command) for role, command in dry_plan.items()
    }
    preview = {
        "mode": "execute" if args.execute else "dry_run",
        "pair_id": config["pairs"][0]["pair_id"],
        "canary_datasets": [record["name"] for record in canary],
        "shard_count": 8,
        "capacity": capacity_contract,
        "command_plan_sha256_by_role": command_hashes,
    }
    if not args.execute:
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0
    for command in ("sbatch", "scancel", "scontrol", "squeue", "sacct"):
        if shutil.which(command) is None:
            raise RuntimeError(f"required Slurm command is unavailable: {command}")
    output_root.mkdir()
    for name in ("canary", "shards", "monitor", "logs"):
        (output_root / name).mkdir()
    private_operations = output_root / ".private-operations"
    private_operations.mkdir(mode=0o700)
    if min(
        _available_bytes(output_root),
        _available_bytes(capacity_root),
        _available_bytes(scratch_root),
    ) < CAPACITY_HARD_RESERVE_BYTES:
        raise RuntimeError("TALENT campaign fell below the 20 GiB post-create reserve")
    journal = _DurableJournal(private_operations / "job-journal.json")
    created_jobs: dict[str, str] = {}
    try:
        canary_job = _submit_job(
            dry_plan["canary"], role="canary", journal=journal
        )
        created_jobs["canary"] = canary_job
        live_plan = _command_plan(
            sbatch=shutil.which("sbatch") or "sbatch",
            analysis_root=analysis_root,
            model_root=model_root,
            talent_root=talent_root,
            run_config=run_config_path,
            shard_plan=plan_path,
            output_root=output_root,
            scratch_root=scratch_root,
            python=python,
            analysis_sha=analysis_sha,
            model_sha=model_sha,
            run_config_sha256=config["config_sha256"],
            shard_plan_sha256=plan_file_sha,
            canary_dependency=canary_job,
        )
        array_job = _submit_job(
            live_plan["array"], role="full_array", journal=journal
        )
        created_jobs["full_array"] = array_job
        live_plan = _command_plan(
            sbatch=shutil.which("sbatch") or "sbatch",
            analysis_root=analysis_root,
            model_root=model_root,
            talent_root=talent_root,
            run_config=run_config_path,
            shard_plan=plan_path,
            output_root=output_root,
            scratch_root=scratch_root,
            python=python,
            analysis_sha=analysis_sha,
            model_sha=model_sha,
            run_config_sha256=config["config_sha256"],
            shard_plan_sha256=plan_file_sha,
            canary_dependency=canary_job,
            array_dependency=array_job,
        )
        aggregate_job = _submit_job(
            live_plan["aggregate"], role="aggregate", journal=journal
        )
        created_jobs["aggregate"] = aggregate_job
        _verify_inputs_unchanged(
            analysis_root=analysis_root,
            model_root=model_root,
            analysis_sha=analysis_sha,
            model_sha=model_sha,
            config=config,
            plan_path=plan_path,
            plan_file_sha=plan_file_sha,
        )
        if min(
            _available_bytes(output_root),
            _available_bytes(capacity_root),
            _available_bytes(scratch_root),
        ) < CAPACITY_HARD_RESERVE_BYTES:
            raise RuntimeError("TALENT campaign lost its reserve before release")
        receipt_payload: dict[str, Any] = {
            "schema_version": 1,
            "formal_eligible": False,
            "study": "tabicl-talent-paired-checkpoints-exploratory-v1",
            "pair_contract": config["portable"],
            "pair_contract_sha256": config["portable_sha256"],
            "checkpoint_contract": checkpoint_contract,
            "analysis_sha": analysis_sha,
            "model_runtime_sha": model_sha,
            "shard_plan_file_sha256": plan_file_sha,
            "shard_plan_document_sha256": plan_document["sha256"],
            "canary_dataset_ordinals": [record["ordinal"] for record in canary],
            "shard_count": 8,
            "capacity": capacity_contract,
            "job_roles": {
                "canary": {"submission_state": "held"},
                "full_array": {"dependency": "afterok:canary"},
                "aggregate": {"dependency": "afterok:full_array"},
            },
            "command_plan_sha256_by_role": {
                role: canonical_json_sha256(command)
                for role, command in live_plan.items()
            },
            "operation_journal_sha256_at_submission": journal.sha256,
            "created_at_unix_seconds": time.time(),
        }
        receipt = self_hashed_document(SUBMISSION_KIND, receipt_payload)
        submission_path = output_root / "submission-receipt.json"
        atomic_json(submission_path, receipt)
        journal.append(
            {"event": "release_started", "job_role": "canary", "job_id": canary_job}
        )
        released = subprocess.run(
            ["scontrol", "release", canary_job],
            check=False,
            capture_output=True,
            text=True,
        )
        journal.append(
            {
                "event": "release_returned",
                "job_role": "canary",
                "job_id": canary_job,
                "returncode": released.returncode,
            }
        )
        if released.returncode != 0:
            raise subprocess.CalledProcessError(
                released.returncode, ["scontrol", "release", canary_job]
            )
        release = self_hashed_document(
            RELEASE_KIND,
            {
                "schema_version": 1,
                "submission_document_sha256": receipt["sha256"],
                "submission_file_sha256": sha256_file(submission_path),
                "released_job_role": "canary",
                "operation_journal_sha256_after_release": journal.sha256,
                "released_at_unix_seconds": time.time(),
            },
        )
        atomic_json(output_root / "release-receipt.json", release)
    except BaseException as error:
        if created_jobs:
            try:
                rollback = _rollback_jobs(
                    created_jobs,
                    journal=journal,
                    failure_type=type(error).__name__,
                )
            except BaseException as rollback_error:
                rollback = self_hashed_document(
                    ROLLBACK_KIND,
                    {
                        "schema_version": 1,
                        "failure_type": type(error).__name__,
                        "rollback_status": "unverified_or_incomplete",
                        "rollback_verified": False,
                        "rollback_error_type": type(rollback_error).__name__,
                        "operation_journal_sha256": journal.sha256,
                    },
                )
            atomic_json(output_root / "rollback-sidecar.json", rollback)
            if rollback["payload"].get("rollback_verified") is not True:
                error.add_note("Slurm rollback was not fully verified; inspect sidecar")
        raise
    print(
        json.dumps(
            {
                "status": "submitted_and_released",
                "job_roles": ["canary", "full_array", "aggregate"],
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    return _run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
