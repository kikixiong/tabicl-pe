#!/usr/bin/env python3
"""Dry-run-first submission of one canary -> 8 shards -> aggregate campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from pe_mechanism.full_suite_pair import (  # noqa: E402
    load_pair_manifest,
    load_roster,
    load_shard_plan,
)


_WARN_FREE_GIB = 22
_BLOCK_FREE_GIB = 20
_TERMINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
)


def _reject_symlink_components(path: Path, *, label: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{label} must not traverse symlinks: {current}")


def _absolute(value: str, *, label: str, file: bool = False) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    _reject_symlink_components(path, label=label)
    if file and (path.is_symlink() or not path.is_file()):
        raise ValueError(f"{label} must be a real file")
    return path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _overlap(left: Path, right: Path) -> bool:
    return _is_within(left, right) or _is_within(right, left)


def _git_checkout(root: Path, *, expected: str, label: str) -> str:
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise ValueError(f"{label} checkout must be clean")
    symbolic = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "-q", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if symbolic.returncode == 0:
        raise ValueError(f"{label} checkout must be detached")
    if symbolic.returncode != 1:
        raise RuntimeError(f"could not determine whether {label} is detached")
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected:
        raise ValueError(f"{label} checkout SHA mismatch: {head} != {expected}")
    return head


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_export(values: Mapping[str, str]) -> str:
    for name, value in values.items():
        if not name or any(character in value for character in (",", "\n", "\r")):
            raise ValueError(f"unsafe Slurm export value: {name}")
    return "ALL," + ",".join(f"{name}={value}" for name, value in values.items())


def _sbatch_command(
    *,
    wrapper: Path,
    exports: Mapping[str, str],
    output: Path,
    qos: str,
    walltime: str,
    dependency: str | None = None,
    array: str | None = None,
    hold: bool = False,
) -> list[str]:
    command = [
        "sbatch",
        "--parsable",
        f"--qos={qos}",
        f"--time={walltime}",
        f"--output={output}",
        f"--error={output.with_suffix('.err')}",
        f"--export={_safe_export(exports)}",
    ]
    if hold:
        command.append("--hold")
    if dependency is not None:
        command.append(f"--dependency=afterok:{dependency}")
    if array is not None:
        command.append(f"--array={array}")
    command.append(str(wrapper))
    return command


def _submit(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"sbatch returned an invalid job id: {completed.stdout!r}")
    return job_id


def _job_state(job_id: str) -> dict[str, Any]:
    queued = subprocess.run(
        ["squeue", "--noheader", "--jobs", job_id, "--format=%T"],
        check=False,
        capture_output=True,
        text=True,
    )
    queue_states = [line.strip().upper() for line in queued.stdout.splitlines() if line.strip()]
    if queue_states:
        state = queue_states[0].split("+", 1)[0]
        return {
            "source": "squeue",
            "state": state,
            "terminal": state in _TERMINAL_STATES,
            "query_returncode": queued.returncode,
        }
    accounting = subprocess.run(
        ["sacct", "--noheader", "--parsable2", "--jobs", job_id, "--format=JobIDRaw,State"],
        check=False,
        capture_output=True,
        text=True,
    )
    states: list[str] = []
    for line in accounting.stdout.splitlines():
        fields = line.split("|")
        raw_state = fields[1].strip().upper() if len(fields) >= 2 else ""
        if fields and fields[0] == job_id and raw_state:
            states.append(raw_state.split("+", 1)[0].split()[0])
    state = states[-1] if states else "UNKNOWN"
    return {
        "source": "sacct",
        "state": state,
        "terminal": state in _TERMINAL_STATES,
        "query_returncode": accounting.returncode,
    }


def _cancel_and_verify(job_ids: list[str], *, attempts: int = 20) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for job_id in job_ids:
        cancelled = subprocess.run(
            ["scancel", job_id], check=False, capture_output=True, text=True
        )
        records[job_id] = {
            "scancel_returncode": cancelled.returncode,
            "scancel_stderr": cancelled.stderr.strip(),
        }
    pending = set(job_ids)
    for attempt in range(attempts):
        for job_id in tuple(pending):
            observation = _job_state(job_id)
            records[job_id]["last_observation"] = observation
            records[job_id]["poll_count"] = attempt + 1
            if observation["terminal"]:
                pending.remove(job_id)
        if not pending:
            break
        if attempt + 1 < attempts:
            time.sleep(0.5)
    return {
        "all_jobs_terminal": not pending,
        "terminal_job_ids": sorted(set(job_ids) - pending, key=int),
        "unverified_job_ids": sorted(pending, key=int),
        "jobs": records,
    }


def _publish_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("submission receipt must be fresh")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("beyond-h100", "tabarena-a10"))
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--tabarena-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument(
        "--capacity-root",
        required=True,
        help="real directory on the primary shared filesystem subject to 20/22 GiB gates",
    )
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--roster", required=True)
    parser.add_argument("--shard-plan", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--receipt")
    parser.add_argument("--python", required=True)
    parser.add_argument("--expected-analysis-sha", required=True)
    parser.add_argument("--expected-tabarena-sha", required=True)
    parser.add_argument(
        "--submit",
        action="store_true",
        help="perform the three sbatch calls; default is a read-only dry run",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    analysis_root = _absolute(args.analysis_root, label="analysis_root").resolve(strict=True)
    model_root = _absolute(args.model_root, label="model_root").resolve(strict=True)
    tabarena_root = _absolute(args.tabarena_root, label="tabarena_root").resolve(strict=True)
    cache_root = _absolute(args.cache_root, label="cache_root").resolve(strict=True)
    capacity_root = _absolute(
        args.capacity_root, label="capacity_root"
    ).resolve(strict=True)
    python = _absolute(args.python, label="python", file=True).resolve(strict=True)
    pair_path = _absolute(args.pair_manifest, label="pair_manifest", file=True)
    roster_path = _absolute(args.roster, label="roster", file=True)
    plan_path = _absolute(args.shard_plan, label="shard_plan", file=True)
    run_root = _absolute(args.run_root, label="run_root")
    if run_root.exists() or run_root.is_symlink():
        raise FileExistsError("run_root must be absent for a fresh campaign")
    if not run_root.parent.is_dir() or run_root.parent.is_symlink():
        raise ValueError("run_root parent must be a real existing directory")
    _reject_symlink_components(run_root.parent, label="run_root parent")
    receipt = (
        _absolute(args.receipt, label="receipt")
        if args.receipt
        else run_root.parent / f"{run_root.name}.submission-receipt.json"
    )
    if receipt.exists() or receipt.is_symlink():
        raise FileExistsError("submission receipt must be absent")
    release_receipt = receipt.with_name(f"{receipt.name}.release.json")
    if release_receipt.exists() or release_receipt.is_symlink():
        raise FileExistsError("submission release receipt must be absent")
    sources = (analysis_root, model_root, tabarena_root)
    if any(_overlap(left, right) for i, left in enumerate(sources) for right in sources[i + 1 :]):
        raise ValueError("analysis, model, and TabArena checkouts must be disjoint")
    if _overlap(run_root, cache_root):
        raise ValueError("run_root and cache_root must be disjoint")
    for output in (run_root, receipt, cache_root):
        for source in (analysis_root, model_root, tabarena_root):
            if _overlap(output, source):
                raise ValueError(
                    "campaign outputs/cache and source checkouts must be bidirectionally disjoint"
                )

    pair = load_pair_manifest(pair_path, verify_checkpoints=True)
    roster = load_roster(roster_path)
    plan = load_shard_plan(plan_path, roster=roster)
    expected_suite = "beyondarena" if args.mode == "beyond-h100" else "tabarena-v0.1"
    if roster.suite_id != expected_suite:
        raise ValueError("submission mode and roster suite disagree")
    if args.expected_tabarena_sha != roster.source_commit:
        raise ValueError("expected TabArena SHA differs from roster source")
    analysis_sha = _git_checkout(
        analysis_root, expected=args.expected_analysis_sha, label="analysis"
    )
    model_sha = _git_checkout(
        model_root, expected=pair.model_source_sha, label="model"
    )
    tabarena_sha = _git_checkout(
        tabarena_root, expected=roster.source_commit, label="TabArena"
    )
    disk_checks: dict[str, dict[str, Any]] = {}
    for label, root in (("capacity", capacity_root), ("run", run_root.parent)):
        free_bytes = shutil.disk_usage(root).free
        free_gib = free_bytes / (1024**3)
        disk_checks[label] = {
            "free_bytes": free_bytes,
            "below_warning_threshold": free_gib < _WARN_FREE_GIB,
        }
        if free_gib < _BLOCK_FREE_GIB:
            raise RuntimeError(
                f"{label} filesystem free disk is below the 20 GiB submission block threshold"
            )

    scripts = analysis_root / "analysis" / "pe_mechanism" / "scripts"
    beyond_canary = scripts / "slurm_pair_full_suite_canary.sh"
    beyond_full = scripts / "slurm_pair_full_suite_shard.sh"
    tabarena_worker = scripts / "slurm_pair_tabarena_a10.sh"
    aggregate_wrapper = scripts / "slurm_pair_full_suite_aggregate.sh"
    for wrapper in (beyond_canary, beyond_full, tabarena_worker, aggregate_wrapper):
        if wrapper.is_symlink() or not wrapper.is_file():
            raise FileNotFoundError(f"missing submission wrapper: {wrapper}")
    input_digests = {
        "pair_manifest_sha256": pair.sha256,
        "roster_sha256": roster.sha256,
        "shard_plan_sha256": plan.sha256,
        "wrappers_sha256": {
            wrapper.name: _sha256(wrapper)
            for wrapper in (beyond_canary, beyond_full, tabarena_worker, aggregate_wrapper)
        },
    }
    log_root = run_root.parent / f".{run_root.name}-slurm"
    common = {
        "PE_PAIR_ANALYSIS_ROOT": str(analysis_root),
        "PE_PAIR_MODEL_ROOT": str(model_root),
        "PE_PAIR_TABARENA_ROOT": str(tabarena_root),
        "PE_PAIR_CACHE_ROOT": str(cache_root),
        "PE_PAIR_MANIFEST": str(pair_path),
        "PE_PAIR_ROSTER": str(roster_path),
        "PE_PAIR_SHARD_PLAN": str(plan_path),
        "PE_PAIR_RUN_ROOT": str(run_root),
        "PE_PAIR_PYTHON": str(python),
        "PE_PAIR_EXPECTED_ANALYSIS_SHA": args.expected_analysis_sha,
        "PE_PAIR_EXPECTED_TABARENA_SHA": args.expected_tabarena_sha,
        "PE_PAIR_EXPECTED_MANIFEST_SHA256": pair.sha256,
        "PE_PAIR_EXPECTED_ROSTER_SHA256": roster.sha256,
        "PE_PAIR_EXPECTED_SHARD_PLAN_SHA256": plan.sha256,
    }
    if args.mode == "beyond-h100":
        canary_wrapper = beyond_canary
        canary_exports = dict(common)
        canary_array = None
        full_wrapper = beyond_full
        full_exports = dict(common)
    else:
        canary_wrapper = tabarena_worker
        canary_exports = {**common, "PE_PAIR_PHASE": "canary"}
        canary_array = "0-0"
        full_wrapper = tabarena_worker
        full_exports = {**common, "PE_PAIR_PHASE": "full"}

    canary_command = _sbatch_command(
        wrapper=canary_wrapper,
        exports=canary_exports,
        output=log_root / "canary-%A_%a.out",
        qos="medium" if args.mode == "beyond-h100" else "short",
        walltime="24:00:00" if args.mode == "beyond-h100" else "03:00:00",
        array=canary_array,
        hold=True,
    )
    plan_summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "two_arm_full_suite_submission_receipt",
        "status": "dry_run" if not args.submit else "submitting",
        "mode": args.mode,
        "pair_id": pair.pair_id,
        "input_digests": input_digests,
        "source_commits": {
            "analysis": analysis_sha,
            "model": model_sha,
            "tabarena": tabarena_sha,
        },
        "run_root": str(run_root),
        "disk_checks": disk_checks,
        "disk_block_threshold_gib": _BLOCK_FREE_GIB,
        "disk_warning_threshold_gib": _WARN_FREE_GIB,
        "capacity_gate_required": True,
        "canary_command": canary_command,
        "full_policy": {
            "array": "0-7%2",
            "dependency": "afterok:CANARY_JOB_ID",
            "qos": "medium",
            "walltime": "24:00:00",
        },
        "aggregate_policy": {
            "dependency": "afterok:FULL_ARRAY_JOB_ID",
            "qos": "short",
            "walltime": "01:00:00",
        },
        "cross_pair_submission": False,
    }
    if not args.submit:
        print(json.dumps(plan_summary, indent=2, sort_keys=True))
        return 0

    log_root.mkdir(parents=False)
    created_jobs: list[str] = []
    rollback_performed: dict[str, Any] | None = None
    try:
        canary_id = _submit(canary_command)
        created_jobs.append(canary_id)
        full_command = _sbatch_command(
            wrapper=full_wrapper,
            exports=full_exports,
            output=log_root / "full-%A_%a.out",
            qos="medium",
            walltime="24:00:00",
            dependency=canary_id,
            array="0-7%2",
        )
        full_id = _submit(full_command)
        created_jobs.append(full_id)
        aggregate_exports = {
            key: common[key]
            for key in (
                "PE_PAIR_ANALYSIS_ROOT",
                "PE_PAIR_MANIFEST",
                "PE_PAIR_ROSTER",
                "PE_PAIR_SHARD_PLAN",
                "PE_PAIR_RUN_ROOT",
                "PE_PAIR_PYTHON",
                "PE_PAIR_EXPECTED_ANALYSIS_SHA",
                "PE_PAIR_EXPECTED_MANIFEST_SHA256",
                "PE_PAIR_EXPECTED_ROSTER_SHA256",
                "PE_PAIR_EXPECTED_SHARD_PLAN_SHA256",
            )
        }
        aggregate_command = _sbatch_command(
            wrapper=aggregate_wrapper,
            exports=aggregate_exports,
            output=log_root / "aggregate-%j.out",
            qos="short",
            walltime="01:00:00",
            dependency=full_id,
        )
        aggregate_id = _submit(aggregate_command)
        created_jobs.append(aggregate_id)
        plan_summary.update(
            {
                "status": "submitted",
                "jobs": {
                    "canary": canary_id,
                    "full_array": full_id,
                    "aggregate": aggregate_id,
                },
            }
        )
        plan_summary["status"] = "held_chain_recorded_before_release"
        _publish_receipt(receipt, plan_summary)
        try:
            subprocess.run(
                ["scontrol", "release", canary_id],
                check=True,
                capture_output=True,
                text=True,
            )
        except BaseException as error:
            rollback = _cancel_and_verify(created_jobs)
            rollback_performed = rollback
            _publish_receipt(
                release_receipt,
                {
                    "schema_version": 1,
                    "kind": "two_arm_full_suite_release_status",
                    "status": (
                        "release_failed_jobs_terminal_verified"
                        if rollback["all_jobs_terminal"]
                        else "release_failed_rollback_unverified"
                    ),
                    "jobs": plan_summary["jobs"],
                    "canary_job_id": canary_id,
                    "error_type": type(error).__name__,
                    "rollback": rollback,
                },
            )
            raise
        _publish_receipt(
            release_receipt,
            {
                "schema_version": 1,
                "kind": "two_arm_full_suite_release_status",
                "status": "canary_released",
                "jobs": plan_summary["jobs"],
                "canary_job_id": canary_id,
            },
        )
    except BaseException:
        rollback = rollback_performed
        if created_jobs and rollback is None:
            rollback = _cancel_and_verify(created_jobs)
        if not receipt.exists() and not receipt.is_symlink():
            failure = {
                **plan_summary,
                "status": (
                    "submission_failed_jobs_terminal_verified"
                    if rollback is not None and rollback["all_jobs_terminal"]
                    else "submission_failed_rollback_unverified"
                ),
                "created_jobs": created_jobs,
                "rollback": rollback,
            }
            _publish_receipt(receipt, failure)
        raise
    print(json.dumps({"receipt": str(receipt), "jobs": created_jobs}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
