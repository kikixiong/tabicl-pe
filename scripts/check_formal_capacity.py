#!/usr/bin/env python3
"""Fail-closed byte-exact capacity gate for a formal identity study.

The submitter calls this once for the empty cohort.  Stage wrappers first audit
the complete study tree and subtract the byte-exact amount already consumed
from that same whole-study peak budget.  All arithmetic is integer arithmetic;
in particular, no GiB value is converted to ``float``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Callable, Iterable


GIB = 1 << 30
ABSOLUTE_FLOOR_BYTES = 20 * GIB
EMPTY_STUDY_CHECKPOINT_SLOTS = 15
FORMAL_JOB_COUNT = 9


def _nonnegative_integer(raw: str) -> int:
    try:
        value = int(raw, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a base-10 integer") from error
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def available_bytes(
    path: str | os.PathLike[str],
    *,
    statvfs: Callable[[str | os.PathLike[str]], os.statvfs_result] = os.statvfs,
) -> int:
    """Return user-available filesystem bytes without floating-point rounding."""
    values = statvfs(path)
    result = values.f_bavail * values.f_frsize
    if result < 0:
        raise ValueError("statvfs returned negative available bytes")
    return result


def empty_study_remaining_bytes(
    checkpoint_ceiling_bytes: int, durable_log_allowance_bytes: int
) -> int:
    """Peak remaining allocation for a fresh three-arm, three-stage cohort.

    The 15 checkpoint slots are nine retained stage finals plus, for each arm,
    one retained temporary and one simultaneous atomic/new checkpoint.
    """
    _validate_nonnegative("checkpoint_ceiling_bytes", checkpoint_ceiling_bytes)
    _validate_nonnegative(
        "durable_log_allowance_bytes", durable_log_allowance_bytes
    )
    if checkpoint_ceiling_bytes == 0:
        raise ValueError("checkpoint_ceiling_bytes must be positive")
    return (
        EMPTY_STUDY_CHECKPOINT_SLOTS * checkpoint_ceiling_bytes
        + durable_log_allowance_bytes
    )


def remaining_after_audit(
    checkpoint_ceiling_bytes: int,
    durable_log_allowance_bytes: int,
    *,
    checkpoint_count: int,
    checkpoint_bytes: int,
    durable_bytes: int,
) -> tuple[int, int, int]:
    """Return ``(initial, consumed, remaining)`` for one shared peak budget.

    The initial 15 checkpoint slots contain nine eventual finals and six
    temp/atomic peak slots.  Subtracting audited bytes preserves those slots
    without assuming lockstep arms.  Callers must bracket statvfs with two
    identical artifact snapshots before using this result.
    """
    initial = empty_study_remaining_bytes(
        checkpoint_ceiling_bytes, durable_log_allowance_bytes
    )
    _validate_nonnegative("checkpoint_bytes", checkpoint_bytes)
    _validate_nonnegative("durable_bytes", durable_bytes)
    _validate_nonnegative("checkpoint_count", checkpoint_count)
    if checkpoint_count > EMPTY_STUDY_CHECKPOINT_SLOTS:
        raise ValueError("artifact tree has more than fifteen checkpoint slots")
    if checkpoint_bytes > EMPTY_STUDY_CHECKPOINT_SLOTS * checkpoint_ceiling_bytes:
        raise ValueError("artifact tree checkpoint bytes exceed study peak budget")
    if durable_bytes > durable_log_allowance_bytes:
        raise ValueError("artifact tree durable bytes exceed study allowance")
    consumed = checkpoint_bytes + durable_bytes
    if consumed > initial:
        raise ValueError("artifact tree consumed bytes exceed study peak budget")
    return initial, consumed, initial - consumed


def required_bytes(remaining_bytes: int) -> int:
    _validate_nonnegative("remaining_bytes", remaining_bytes)
    return ABSOLUTE_FLOOR_BYTES + (5 * remaining_bytes + 3) // 4


def evaluate_capacity(available: int, remaining: int) -> dict[str, object]:
    _validate_nonnegative("available_bytes", available)
    _validate_nonnegative("remaining_bytes", remaining)
    required = required_bytes(remaining)
    reasons: list[str] = []
    if available < ABSOLUTE_FLOOR_BYTES:
        reasons.append("absolute_floor")
    if available < required:
        reasons.append("remaining_capacity")
    return {
        "schema_version": 1,
        "allowed": not reasons,
        "available_bytes": available,
        "remaining_bytes": remaining,
        "required_bytes": required,
        "reasons": reasons,
    }


def _validate_nonnegative(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _regular_file_size(path: Path) -> int:
    if path.is_symlink():
        raise ValueError(f"artifact path must not be a symlink: {path}")
    try:
        info = path.stat()
    except OSError as error:
        raise ValueError(f"cannot stat artifact: {path}") from error
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"artifact must be a regular file: {path}")
    return info.st_size


def audit_artifact_ceilings(
    *,
    checkpoint_paths: Iterable[Path],
    log_paths: Iterable[Path],
    monitor_paths: Iterable[Path],
    checkpoint_ceiling_bytes: int,
    durable_log_allowance_bytes: int,
) -> dict[str, object]:
    """Enforce the measured checkpoint and aggregate durable-output ceilings."""
    _validate_nonnegative("checkpoint_ceiling_bytes", checkpoint_ceiling_bytes)
    _validate_nonnegative(
        "durable_log_allowance_bytes", durable_log_allowance_bytes
    )
    if checkpoint_ceiling_bytes == 0:
        raise ValueError("checkpoint ceiling must be positive")

    checkpoint_sizes = [_regular_file_size(Path(path)) for path in checkpoint_paths]
    if any(size > checkpoint_ceiling_bytes for size in checkpoint_sizes):
        raise ValueError("checkpoint artifact exceeds checkpoint ceiling")
    log_sizes = [_regular_file_size(Path(path)) for path in log_paths]
    monitor_sizes = [_regular_file_size(Path(path)) for path in monitor_paths]
    durable_bytes = sum(log_sizes) + sum(monitor_sizes)
    if durable_bytes > durable_log_allowance_bytes:
        raise ValueError("durable log/monitor artifacts exceed durable-log ceiling")
    return {
        "schema_version": 1,
        "allowed": True,
        "checkpoint_count": len(checkpoint_sizes),
        "checkpoint_bytes": sum(checkpoint_sizes),
        "checkpoint_ceiling_bytes": checkpoint_ceiling_bytes,
        "durable_log_bytes": durable_bytes,
        "durable_log_allowance_bytes": durable_log_allowance_bytes,
    }


def validate_durable_budget_partition(
    *,
    durable_log_allowance_bytes: int,
    run_log_ceiling_bytes: int,
    attestation_ceiling_bytes: int,
    manifest_ceiling_bytes: int,
    protocol_metadata_allowance_bytes: int,
) -> dict[str, int]:
    for name, value in (
        ("durable_log_allowance_bytes", durable_log_allowance_bytes),
        ("run_log_ceiling_bytes", run_log_ceiling_bytes),
        ("attestation_ceiling_bytes", attestation_ceiling_bytes),
        ("manifest_ceiling_bytes", manifest_ceiling_bytes),
        ("protocol_metadata_allowance_bytes", protocol_metadata_allowance_bytes),
    ):
        _validate_nonnegative(name, value)
    assigned = (
        2 * FORMAL_JOB_COUNT * run_log_ceiling_bytes
        + 2 * FORMAL_JOB_COUNT * attestation_ceiling_bytes
        + FORMAL_JOB_COUNT * manifest_ceiling_bytes
        + protocol_metadata_allowance_bytes
    )
    if assigned > durable_log_allowance_bytes:
        raise ValueError("formal durable-output sub-budgets exceed aggregate allowance")
    return {
        "assigned_bytes": assigned,
        "durable_log_allowance_bytes": durable_log_allowance_bytes,
    }


def audit_artifact_tree(
    root: Path,
    *,
    checkpoint_ceiling_bytes: int,
    durable_log_allowance_bytes: int,
    run_log_ceiling_bytes: int,
    attestation_ceiling_bytes: int,
    manifest_ceiling_bytes: int,
) -> dict[str, object]:
    """Count every durable non-checkpoint byte under the study exactly once."""
    checkpoint_bytes = 0
    checkpoint_count = 0
    durable_bytes = 0
    files = 0
    snapshot: list[dict[str, object]] = []
    for directory, names, filenames in os.walk(root, followlinks=False):
        names.sort()
        filenames.sort()
        directory_path = Path(directory)
        for name in names:
            if (directory_path / name).is_symlink():
                raise ValueError("artifact tree contains a symlink directory")
        for name in filenames:
            path = directory_path / name
            size = _regular_file_size(path)
            info = path.stat()
            if size != info.st_size:
                raise ValueError("artifact changed while tree was audited")
            snapshot.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": size,
                    "mtime_ns": info.st_mtime_ns,
                    "ctime_ns": info.st_ctime_ns,
                    "device": info.st_dev,
                    "inode": info.st_ino,
                }
            )
            files += 1
            if name.endswith(".ckpt") or (name.startswith(".step-") and name.endswith(".tmp")):
                if size > checkpoint_ceiling_bytes:
                    raise ValueError("checkpoint artifact exceeds checkpoint ceiling")
                checkpoint_bytes += size
                checkpoint_count += 1
                continue
            if name in {"train.log", "validation-report.json"} and size > run_log_ceiling_bytes:
                raise ValueError("run log/report exceeds per-output ceiling")
            if name.startswith("source-attestation") and size > attestation_ceiling_bytes:
                raise ValueError("source attestation exceeds per-output ceiling")
            if name == "finalized-checkpoint.json" and size > manifest_ceiling_bytes:
                raise ValueError("finalized manifest exceeds per-output ceiling")
            durable_bytes += size
    if durable_bytes > durable_log_allowance_bytes:
        raise ValueError("artifact tree durable outputs exceed aggregate allowance")
    if checkpoint_count > EMPTY_STUDY_CHECKPOINT_SLOTS:
        raise ValueError("artifact tree has more than fifteen checkpoint slots")
    if checkpoint_bytes > EMPTY_STUDY_CHECKPOINT_SLOTS * checkpoint_ceiling_bytes:
        raise ValueError("artifact tree checkpoint bytes exceed study peak budget")
    if checkpoint_bytes + durable_bytes > empty_study_remaining_bytes(
        checkpoint_ceiling_bytes, durable_log_allowance_bytes
    ):
        raise ValueError("artifact tree consumed bytes exceed study peak budget")
    return {
        "artifact_files": files,
        "checkpoint_count": checkpoint_count,
        "checkpoint_bytes": checkpoint_bytes,
        "durable_bytes": durable_bytes,
        "snapshot_sha256": hashlib.sha256(
            json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-ceiling-bytes", required=True, type=_nonnegative_integer
    )
    parser.add_argument(
        "--durable-log-allowance-bytes", required=True, type=_nonnegative_integer
    )
    remaining_group = parser.add_mutually_exclusive_group()
    remaining_group.add_argument(
        "--remaining-checkpoints",
        type=_nonnegative_integer,
        help="explicit future checkpoint slots; fresh cohort defaults to fifteen",
    )
    remaining_group.add_argument("--remaining-from-audit", action="store_true")
    parser.add_argument("--checkpoint-path", action="append", type=Path, default=[])
    parser.add_argument("--log-path", action="append", type=Path, default=[])
    parser.add_argument("--monitor-path", action="append", type=Path, default=[])
    parser.add_argument("--audit-tree", action="store_true")
    parser.add_argument("--run-log-ceiling-bytes", type=_nonnegative_integer)
    parser.add_argument("--attestation-ceiling-bytes", type=_nonnegative_integer)
    parser.add_argument("--manifest-ceiling-bytes", type=_nonnegative_integer)
    parser.add_argument(
        "--protocol-metadata-allowance-bytes", type=_nonnegative_integer
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    available_bytes_fn: Callable[[str | os.PathLike[str]], int] = available_bytes,
) -> int:
    args = _parser().parse_args(argv)
    root = args.artifact_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("artifact root must be a directory")
    if args.checkpoint_ceiling_bytes == 0:
        raise ValueError("checkpoint ceiling must be positive")

    partition_values = (
        args.run_log_ceiling_bytes,
        args.attestation_ceiling_bytes,
        args.manifest_ceiling_bytes,
        args.protocol_metadata_allowance_bytes,
    )
    if any(value is not None for value in partition_values):
        if not all(value is not None for value in partition_values):
            raise ValueError("every durable-output partition ceiling is required together")
        validate_durable_budget_partition(
            durable_log_allowance_bytes=args.durable_log_allowance_bytes,
            run_log_ceiling_bytes=args.run_log_ceiling_bytes,
            attestation_ceiling_bytes=args.attestation_ceiling_bytes,
            manifest_ceiling_bytes=args.manifest_ceiling_bytes,
            protocol_metadata_allowance_bytes=args.protocol_metadata_allowance_bytes,
        )
    tree_report: dict[str, object] | None = None
    available: int | None = None
    if args.audit_tree:
        if not all(value is not None for value in partition_values):
            raise ValueError("artifact-tree audit requires durable-output partition ceilings")

        def audit_once() -> dict[str, object]:
            return audit_artifact_tree(
                root,
                checkpoint_ceiling_bytes=args.checkpoint_ceiling_bytes,
                durable_log_allowance_bytes=args.durable_log_allowance_bytes,
                run_log_ceiling_bytes=args.run_log_ceiling_bytes,
                attestation_ceiling_bytes=args.attestation_ceiling_bytes,
                manifest_ceiling_bytes=args.manifest_ceiling_bytes,
            )

        if args.remaining_from_audit:
            for _attempt in range(3):
                before = audit_once()
                observed_available = available_bytes_fn(root)
                after = audit_once()
                if before["snapshot_sha256"] == after["snapshot_sha256"]:
                    tree_report = after
                    available = observed_available
                    break
            else:
                raise ValueError("artifact tree changed around statvfs capacity snapshot")
        else:
            tree_report = audit_once()

    if args.checkpoint_path or args.log_path or args.monitor_path:
        audit_artifact_ceilings(
            checkpoint_paths=args.checkpoint_path,
            log_paths=args.log_path,
            monitor_paths=args.monitor_path,
            checkpoint_ceiling_bytes=args.checkpoint_ceiling_bytes,
            durable_log_allowance_bytes=args.durable_log_allowance_bytes,
        )
    if args.remaining_from_audit:
        if tree_report is None:
            raise ValueError("remaining-from-audit requires artifact-tree audit")
        initial, consumed, remaining = remaining_after_audit(
            args.checkpoint_ceiling_bytes,
            args.durable_log_allowance_bytes,
            checkpoint_count=tree_report["checkpoint_count"],
            checkpoint_bytes=tree_report["checkpoint_bytes"],
            durable_bytes=tree_report["durable_bytes"],
        )
    else:
        slots = (
            EMPTY_STUDY_CHECKPOINT_SLOTS
            if args.remaining_checkpoints is None
            else args.remaining_checkpoints
        )
        remaining = slots * args.checkpoint_ceiling_bytes + args.durable_log_allowance_bytes
    if available is None:
        available = available_bytes_fn(root)
    report = evaluate_capacity(available, remaining)
    if args.remaining_from_audit:
        report.update(
            {
                "initial_study_bytes": initial,
                "consumed_bytes": consumed,
                "consumed_checkpoint_bytes": tree_report["checkpoint_bytes"],
                "consumed_checkpoint_count": tree_report["checkpoint_count"],
                "consumed_durable_bytes": tree_report["durable_bytes"],
                "remaining_source": "audited_study_bytes",
            }
        )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["allowed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"formal capacity check failed: {error}", file=sys.stderr)
        raise SystemExit(2)
