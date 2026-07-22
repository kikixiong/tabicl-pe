#!/usr/bin/env python3
"""Bounded, read-only anomaly monitor for formal identity studies."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import statistics
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Protocol, Sequence
import zipfile


MIN_GPU_SAMPLES = 10
MIN_GPU_SAMPLE_GAP_SECONDS = 0.5
MAX_GPU_SAMPLE_GAP_SECONDS = 1.5
MIN_GPU_MEAN_PERCENT = 80.0
GIB = 1 << 30
DISK_WARNING_BYTES = 22 * GIB
DISK_SUBMIT_BLOCK_BYTES = 20 * GIB
DEFAULT_COMMAND_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_TAIL_BYTES = 1 << 20
MAX_VALIDATION_REPORT_BYTES = 1 << 20
MAX_MONITOR_MANIFEST_BYTES = 1 << 20
MAX_MONITOR_STATE_BYTES = 4 << 20
MAX_ALLOWED_TAIL_BYTES = 16 << 20
DEFAULT_MAX_EVENT_LEDGER_BYTES = 1 << 20
DEFAULT_DURATION_SECONDS = 30 * 60.0
DEFAULT_POLL_INTERVAL_SECONDS = 30.0
FAILED_SCHEDULER_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }
)
EXPLICIT_STEP_PATTERN = re.compile(
    r"\bcurr[_ -]?step\s*[:=]\s*(\d+)\b|\bstep\s*=\s*(\d+)\b",
    re.IGNORECASE,
)
TQDM_STEP_PATTERN = re.compile(
    r"(?:^|[\r\n])\s*Step:\s*\d+(?:\.\d+)?%\|[^\r\n]*\|\s*(\d+)\s*/\s*(\d+)\b"
)
LOG_SIGNATURES = (
    ("log_oom", re.compile(r"\b(?:cuda\s+)?out\s+of\s+memory\b|\bcuda\s+oom\b", re.IGNORECASE)),
    ("log_nonfinite", re.compile(r"\b(?:nan|inf|infinity|non[-_ ]?finite)\b", re.IGNORECASE)),
    ("log_traceback", re.compile(r"\btraceback\b", re.IGNORECASE)),
    ("log_error", re.compile(r"\b(?:[a-z_]*error|exception)\b", re.IGNORECASE)),
)
FORMAL_ARMS = frozenset({"rope", "temporary", "none"})
FORMAL_STAGES = frozenset({"stage1", "stage2", "stage3"})
SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


@dataclass(frozen=True)
class GpuRecord:
    """One explicit active-window marker or unfiltered GPU sample."""

    kind: str
    monotonic_seconds: float
    gpu_uuid: str | None = None
    utilization_percent: float | None = None
    expected_gpu_uuids: tuple[str, ...] | None = None
    expected_gpu_count: int | None = None


@dataclass(frozen=True)
class MonitorIssue:
    code: str
    details: dict[str, Any]


@dataclass(frozen=True)
class CheckpointFacts:
    checkpoint_sha256: str
    checkpoint_size: int


@dataclass(frozen=True)
class CheckpointInspection:
    facts: CheckpointFacts | None
    issues: tuple[MonitorIssue, ...]


@dataclass(frozen=True)
class GpuWindowEvaluation:
    complete: bool
    sample_counts: dict[str, int]
    means: dict[str, float]
    issues: tuple[MonitorIssue, ...]


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class Clock(Protocol):
    def time(self) -> float: ...

    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def time(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    arm: str
    stage: str
    log_path: Path
    checkpoint_path: Path
    validation_report_path: Path
    gpu_samples_path: Path | None
    utilization_required: bool
    stall_seconds: float
    checkpoint_stale_seconds: float
    max_checkpoint_bytes: int
    expected_gpu_uuids: tuple[str, ...] = ()
    expected_gpu_count: int = 0


@dataclass(frozen=True)
class MonitorSpec:
    disk_path: Path
    jobs: tuple[JobSpec, ...]
    max_event_ledger_bytes: int = DEFAULT_MAX_EVENT_LEDGER_BYTES


@dataclass(frozen=True)
class MonitorEvent:
    category: str
    code: str
    severity: str
    observed_at: float
    job_id: str | None
    arm: str | None
    stage: str | None
    details: dict[str, Any]


@dataclass(frozen=True)
class PollResult:
    events: tuple[MonitorEvent, ...]
    state: dict[str, Any]


CommandRunner = Callable[..., CommandResult]
CheckpointChecker = Callable[
    [Path, Sequence[Any], int], CheckpointInspection
]


def run_read_only_command(
    argv: Sequence[str], *, timeout_seconds: float
) -> CommandResult:
    if not argv or argv[0] not in {"squeue", "sacct"}:
        raise ValueError("monitor command is not read-only scheduler introspection")
    completed = subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _parse_scheduler_rows(output: str) -> dict[str, str]:
    states: dict[str, str] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) < 2 or not fields[0] or not fields[1]:
            continue
        states[fields[0]] = fields[1].split()[0].rstrip("+").upper()
    return states


def _bounded_tail(path: Path, max_bytes: int) -> tuple[list[Any], str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("observed log is not a regular file")
        offset = max(0, metadata.st_size - max_bytes)
        os.lseek(fd, offset, os.SEEK_SET)
        chunks: list[bytes] = []
        remaining = min(max_bytes, metadata.st_size)
        while remaining:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        identity = [str(path), metadata.st_mtime_ns, metadata.st_size]
        return identity, b"".join(chunks).decode("utf-8", errors="replace")
    finally:
        os.close(fd)


def _regular_file_identity(path: Path) -> list[Any]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("observed checkpoint is not a regular file")
        return [str(path), metadata.st_mtime_ns, metadata.st_size]
    finally:
        os.close(fd)


def _read_bounded_regular_file(path: Path, max_bytes: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("validation report is not a regular file")
        if metadata.st_size > max_bytes:
            raise ValueError("validation report exceeds byte ceiling")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        if (
            metadata.st_dev != after.st_dev
            or metadata.st_ino != after.st_ino
            or metadata.st_mtime_ns != after.st_mtime_ns
            or metadata.st_size != after.st_size
        ):
            raise ValueError("validation report changed while being read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _strict_json_loads(payload: bytes | str) -> Any:
    return json.loads(
        payload,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_unique_json_object,
    )


def _last_observed_step(text: str) -> int | None:
    """Return the last explicit or tqdm-completed training step in a log tail."""

    candidates: list[tuple[int, int]] = []
    for match in EXPLICIT_STEP_PATTERN.finditer(text):
        raw = match.group(1) or match.group(2)
        candidates.append((match.end(), int(raw)))
    for match in TQDM_STEP_PATTERN.finditer(text):
        completed, total = (int(value) for value in match.groups())
        if completed <= total:
            candidates.append((match.end(), completed))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _require_exact_keys(
    value: dict[str, Any], expected: set[str], *, where: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{where} keys mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_absolute_path(value: Any, *, where: str) -> Path:
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError(f"{where} must be an absolute path string")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{where} must be absolute")
    return path


def _positive_number(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{where} must be a positive finite number")
    return result


def load_monitor_spec(path: Path) -> MonitorSpec:
    """Load the exact, public monitor-manifest schema."""

    if not path.is_absolute():
        raise ValueError("monitor manifest path must be absolute")
    payload = _strict_json_loads(
        _read_bounded_regular_file(path, MAX_MONITOR_MANIFEST_BYTES)
    )
    if not isinstance(payload, dict):
        raise ValueError("monitor manifest root must be an object")
    _require_exact_keys(
        payload,
        {"schema_version", "disk_path", "max_event_ledger_bytes", "jobs"},
        where="monitor manifest",
    )
    if payload["schema_version"] != 1 or not isinstance(payload["jobs"], list):
        raise ValueError("monitor manifest schema is invalid")
    if not payload["jobs"]:
        raise ValueError("monitor manifest must contain at least one job")
    max_event_ledger_bytes = payload["max_event_ledger_bytes"]
    if (
        isinstance(max_event_ledger_bytes, bool)
        or not isinstance(max_event_ledger_bytes, int)
        or max_event_ledger_bytes < 1
    ):
        raise ValueError("max_event_ledger_bytes must be a positive integer")

    job_keys = {
        "job_id",
        "arm",
        "stage",
        "log_path",
        "checkpoint_path",
        "validation_report_path",
        "gpu_samples_path",
        "utilization_required",
        "stall_seconds",
        "checkpoint_stale_seconds",
        "max_checkpoint_bytes",
        "expected_gpu_uuids",
        "expected_gpu_count",
    }
    jobs: list[JobSpec] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(payload["jobs"]):
        if not isinstance(item, dict):
            raise ValueError(f"jobs[{index}] must be an object")
        _require_exact_keys(item, job_keys, where=f"jobs[{index}]")
        job_id = item["job_id"]
        if not isinstance(job_id, str) or SAFE_JOB_ID.fullmatch(job_id) is None:
            raise ValueError(f"jobs[{index}].job_id is invalid")
        if job_id in seen_ids:
            raise ValueError("monitor job IDs must be unique")
        seen_ids.add(job_id)
        if item["arm"] not in FORMAL_ARMS or item["stage"] not in FORMAL_STAGES:
            raise ValueError(f"jobs[{index}] arm/stage is invalid")
        if not isinstance(item["utilization_required"], bool):
            raise ValueError(f"jobs[{index}].utilization_required must be boolean")
        gpu_path = item["gpu_samples_path"]
        if gpu_path is not None:
            gpu_path = _require_absolute_path(
                gpu_path, where=f"jobs[{index}].gpu_samples_path"
            )
        gpu_uuids = item["expected_gpu_uuids"]
        gpu_count = item["expected_gpu_count"]
        if (
            not isinstance(gpu_uuids, list)
            or any(not isinstance(value, str) or not value for value in gpu_uuids)
            or len(set(gpu_uuids)) != len(gpu_uuids)
            or isinstance(gpu_count, bool)
            or not isinstance(gpu_count, int)
            or gpu_count < 0
        ):
            raise ValueError(f"jobs[{index}] GPU expectation is invalid")
        if item["utilization_required"]:
            if gpu_path is None or gpu_count < 1 or len(gpu_uuids) not in {
                0,
                gpu_count,
            }:
                raise ValueError(
                    f"jobs[{index}] utilization gate requires GPU count and optional exact UUIDs"
                )
        elif gpu_count != 0 or gpu_uuids:
            raise ValueError(
                f"jobs[{index}] functional-only job must not claim GPU utilization"
            )
        max_checkpoint_bytes = item["max_checkpoint_bytes"]
        if (
            isinstance(max_checkpoint_bytes, bool)
            or not isinstance(max_checkpoint_bytes, int)
            or max_checkpoint_bytes < 1
        ):
            raise ValueError(f"jobs[{index}].max_checkpoint_bytes is invalid")
        jobs.append(
            JobSpec(
                job_id=job_id,
                arm=item["arm"],
                stage=item["stage"],
                log_path=_require_absolute_path(
                    item["log_path"], where=f"jobs[{index}].log_path"
                ),
                checkpoint_path=_require_absolute_path(
                    item["checkpoint_path"],
                    where=f"jobs[{index}].checkpoint_path",
                ),
                validation_report_path=_require_absolute_path(
                    item["validation_report_path"],
                    where=f"jobs[{index}].validation_report_path",
                ),
                gpu_samples_path=gpu_path,
                utilization_required=item["utilization_required"],
                stall_seconds=_positive_number(
                    item["stall_seconds"], where=f"jobs[{index}].stall_seconds"
                ),
                checkpoint_stale_seconds=_positive_number(
                    item["checkpoint_stale_seconds"],
                    where=f"jobs[{index}].checkpoint_stale_seconds",
                ),
                max_checkpoint_bytes=max_checkpoint_bytes,
                expected_gpu_uuids=tuple(gpu_uuids),
                expected_gpu_count=gpu_count,
            )
        )
    return MonitorSpec(
        disk_path=_require_absolute_path(payload["disk_path"], where="disk_path"),
        jobs=tuple(jobs),
        max_event_ledger_bytes=max_event_ledger_bytes,
    )


def parse_gpu_records(text: str) -> tuple[list[GpuRecord], int]:
    """Parse bounded JSONL or simple CSV active-window records."""

    records: list[GpuRecord] = []
    invalid = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            if line.startswith("{"):
                value = _strict_json_loads(line)
                if not isinstance(value, dict):
                    raise ValueError("GPU record must be an object")
                kind = value.get("kind", value.get("event"))
                timestamp = value.get(
                    "monotonic_seconds", value.get("timestamp_seconds")
                )
                gpu_uuid = value.get("gpu_uuid")
                utilization = value.get(
                    "utilization_percent", value.get("utilization")
                )
                marker_gpu_uuids = value.get("expected_gpu_uuids")
                marker_gpu_count = value.get("expected_gpu_count")
            else:
                fields = [field.strip() for field in line.split(",")]
                if fields[0].lower() in {"kind", "timestamp", "monotonic_seconds"}:
                    continue
                if fields[0] in {"training_start", "training_end"} and len(fields) == 2:
                    kind, timestamp = fields
                    gpu_uuid = utilization = None
                    marker_gpu_uuids = marker_gpu_count = None
                elif len(fields) == 2 and fields[1] in {
                    "training_start",
                    "training_end",
                }:
                    timestamp, kind = fields
                    gpu_uuid = utilization = None
                    marker_gpu_uuids = marker_gpu_count = None
                elif fields[0] == "sample" and len(fields) == 4:
                    kind, timestamp, gpu_uuid, utilization = fields
                    marker_gpu_uuids = marker_gpu_count = None
                elif len(fields) == 3:
                    timestamp, gpu_uuid, utilization = fields
                    kind = "sample"
                    marker_gpu_uuids = marker_gpu_count = None
                else:
                    raise ValueError("unsupported GPU CSV record")
            if kind not in {"training_start", "training_end", "sample"}:
                raise ValueError("GPU record kind is invalid")
            if isinstance(timestamp, bool):
                raise ValueError("GPU timestamp is invalid")
            timestamp_value = float(timestamp)
            if kind == "sample":
                if not isinstance(gpu_uuid, str) or not gpu_uuid:
                    raise ValueError("GPU UUID is invalid")
                if isinstance(utilization, bool):
                    raise ValueError("GPU utilization is invalid")
                utilization_value = float(utilization)
            else:
                gpu_uuid = None
                utilization_value = None
            if kind == "training_start" and marker_gpu_uuids is not None:
                if (
                    not isinstance(marker_gpu_uuids, list)
                    or any(
                        not isinstance(value, str) or not value
                        for value in marker_gpu_uuids
                    )
                    or len(set(marker_gpu_uuids)) != len(marker_gpu_uuids)
                    or isinstance(marker_gpu_count, bool)
                    or not isinstance(marker_gpu_count, int)
                ):
                    raise ValueError("GPU start attestation is invalid")
                bound_gpu_uuids = tuple(marker_gpu_uuids)
                bound_gpu_count = marker_gpu_count
            else:
                bound_gpu_uuids = None
                bound_gpu_count = None
            records.append(
                GpuRecord(
                    kind,
                    timestamp_value,
                    gpu_uuid,
                    utilization_value,
                    bound_gpu_uuids,
                    bound_gpu_count,
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            invalid += 1
    return records, invalid


def inspect_changed_checkpoint(
    checkpoint_path: Path,
    identity: Sequence[Any],
    max_checkpoint_bytes: int,
) -> CheckpointInspection:
    """Perform bounded ZIP/hash inspection for one new checkpoint identity."""

    issues: list[MonitorIssue] = []
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(checkpoint_path, flags)
    except OSError as error:
        return CheckpointInspection(
            None,
            (
                MonitorIssue(
                    "checkpoint_read_failed", {"reason": type(error).__name__}
                ),
            ),
        )
    try:
        before = os.fstat(fd)
        actual_identity = [str(checkpoint_path), before.st_mtime_ns, before.st_size]
        if list(identity) != actual_identity or not stat.S_ISREG(before.st_mode):
            return CheckpointInspection(
                None, (MonitorIssue("checkpoint_changed_during_observation", {}),)
            )
        if (
            isinstance(max_checkpoint_bytes, bool)
            or not isinstance(max_checkpoint_bytes, int)
            or max_checkpoint_bytes < 1
        ):
            return CheckpointInspection(
                None, (MonitorIssue("checkpoint_byte_ceiling_invalid", {}),)
            )
        if before.st_size > max_checkpoint_bytes:
            return CheckpointInspection(
                None,
                (
                    MonitorIssue(
                        "checkpoint_too_large",
                        {
                            "checkpoint_size": before.st_size,
                            "max_checkpoint_bytes": max_checkpoint_bytes,
                        },
                    ),
                ),
            )
        with os.fdopen(fd, "rb", closefd=False) as handle:
            try:
                with zipfile.ZipFile(handle) as archive:
                    if archive.testzip() is not None:
                        issues.append(MonitorIssue("checkpoint_zip_invalid", {}))
            except (OSError, ValueError, zipfile.BadZipFile):
                issues.append(MonitorIssue("checkpoint_zip_invalid", {}))
            handle.seek(0)
            digest = hashlib.sha256()
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
        after = os.fstat(fd)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_size != after.st_size
        ):
            issues.append(MonitorIssue("checkpoint_changed_during_observation", {}))
    finally:
        os.close(fd)

    if any(issue.code == "checkpoint_changed_during_observation" for issue in issues):
        return CheckpointInspection(None, tuple(issues))
    return CheckpointInspection(
        CheckpointFacts(digest.hexdigest(), before.st_size), tuple(issues)
    )


def validate_checkpoint_report(
    validation_report_path: Path,
    facts: CheckpointFacts,
) -> tuple[MonitorIssue, ...]:
    """Bind a changed validator report to cached checkpoint facts."""

    try:
        raw_report = _read_bounded_regular_file(
            validation_report_path, MAX_VALIDATION_REPORT_BYTES
        )
    except FileNotFoundError:
        return (MonitorIssue("checkpoint_validation_missing", {}),)
    except (OSError, ValueError) as error:
        return (
            MonitorIssue(
                "checkpoint_validation_invalid", {"reason": type(error).__name__}
            ),
        )
    try:
        report = _strict_json_loads(raw_report)
        if not isinstance(report, dict):
            raise ValueError("validation report root must be an object")
        if set(report) == {"schema_version", "kind", "payload", "sha256"}:
            if report["schema_version"] != 1 or report["kind"] not in {
                "finalized_checkpoint",
                "strict_checkpoint_validation",
            }:
                raise ValueError("validation manifest identity is invalid")
            body = {
                "schema_version": report["schema_version"],
                "kind": report["kind"],
                "payload": report["payload"],
            }
            if report["sha256"] != hashlib.sha256(
                _canonical_json_bytes(body)
            ).hexdigest():
                raise ValueError("validation manifest digest mismatch")
            if not isinstance(report["payload"], dict):
                raise ValueError("validation manifest payload must be an object")
            report = report["payload"]
        expected_digest = report["checkpoint_sha256"]
        expected_size = report["checkpoint_size"]
        if (
            not isinstance(expected_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
        ):
            raise ValueError("validation report checkpoint fields are invalid")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return (MonitorIssue("checkpoint_validation_invalid", {}),)
    if (
        expected_digest != facts.checkpoint_sha256
        or expected_size != facts.checkpoint_size
    ):
        return (MonitorIssue("checkpoint_validation_mismatch", {}),)
    return ()


def check_changed_checkpoint(
    checkpoint_path: Path,
    validation_report_path: Path,
    identity: Sequence[Any],
    max_checkpoint_bytes: int,
) -> tuple[MonitorIssue, ...]:
    """Compatibility helper combining inspection and current report binding."""

    inspection = inspect_changed_checkpoint(
        checkpoint_path, identity, max_checkpoint_bytes
    )
    if inspection.facts is None:
        return inspection.issues
    return inspection.issues + validate_checkpoint_report(
        validation_report_path, inspection.facts
    )


class FormalIdentityMonitor:
    """One-poll observation engine with all external effects injected."""

    def __init__(
        self,
        spec: MonitorSpec,
        *,
        command_runner: CommandRunner = run_read_only_command,
        clock: Clock | None = None,
        statvfs: Callable[[os.PathLike[str] | str], Any] = os.statvfs,
        command_timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
        max_tail_bytes: int = DEFAULT_MAX_TAIL_BYTES,
        checkpoint_checker: CheckpointChecker = inspect_changed_checkpoint,
    ) -> None:
        self.spec = spec
        self.command_runner = command_runner
        self.clock = clock or SystemClock()
        self.statvfs = statvfs
        self.command_timeout_seconds = command_timeout_seconds
        if max_tail_bytes < 1:
            raise ValueError("max_tail_bytes must be positive")
        self.max_tail_bytes = max_tail_bytes
        self.checkpoint_checker = checkpoint_checker

    @staticmethod
    def _event_for_job(
        job: JobSpec,
        *,
        category: str,
        code: str,
        severity: str,
        observed_at: float,
        details: dict[str, Any],
    ) -> MonitorEvent:
        return MonitorEvent(
            category,
            code,
            severity,
            observed_at,
            job.job_id,
            job.arm,
            job.stage,
            details,
        )

    def _observe_log(
        self,
        job: JobSpec,
        job_state: dict[str, Any],
        *,
        scheduler_state: str | None,
        now: float,
    ) -> list[MonitorEvent]:
        events: list[MonitorEvent] = []
        try:
            identity, tail = _bounded_tail(job.log_path, self.max_tail_bytes)
        except (OSError, ValueError) as error:
            fingerprint = f"{type(error).__name__}:{error}"
            if job_state.get("log_read_failure") != fingerprint:
                events.append(
                    self._event_for_job(
                        job,
                        category="anomaly",
                        code="log_read_failed",
                        severity="error",
                        observed_at=now,
                        details={"reason": type(error).__name__},
                    )
                )
            job_state["log_read_failure"] = fingerprint
            return events
        job_state.pop("log_read_failure", None)

        changed = identity != job_state.get("log_identity")
        if changed:
            job_state["log_identity"] = identity
            seen = set(job_state.get("seen_log_signatures", []))
            for line in tail.splitlines():
                for code, pattern in LOG_SIGNATURES:
                    if pattern.search(line) is None:
                        continue
                    fingerprint = hashlib.sha256(
                        f"{code}\0{line}".encode("utf-8", errors="replace")
                    ).hexdigest()
                    if fingerprint not in seen:
                        events.append(
                            self._event_for_job(
                                job,
                                category="anomaly",
                                code=code,
                                severity="error",
                                observed_at=now,
                                details={"line_sha256": hashlib.sha256(line.encode()).hexdigest()},
                            )
                        )
                        seen.add(fingerprint)
                    break
            job_state["seen_log_signatures"] = sorted(seen)[-256:]

            observed_step = _last_observed_step(tail)
            if observed_step is not None:
                previous_step = job_state.get("last_step")
                if previous_step is not None and observed_step < previous_step:
                    regression = f"{previous_step}:{observed_step}"
                    if job_state.get("step_regression_reported") != regression:
                        events.append(
                            self._event_for_job(
                                job,
                                category="anomaly",
                                code="step_regressed",
                                severity="error",
                                observed_at=now,
                                details={"previous_step": previous_step, "observed_step": observed_step},
                            )
                        )
                    job_state["step_regression_reported"] = regression
                elif previous_step is None or observed_step > previous_step:
                    job_state["last_step"] = observed_step
                    job_state["last_progress_at"] = now
                    job_state.pop("step_stall_reported", None)
                    job_state.pop("step_regression_reported", None)

        if scheduler_state == "RUNNING":
            job_state.setdefault("last_progress_at", now)
            last_progress = float(job_state["last_progress_at"])
            stalled_step = job_state.get("last_step")
            stall_key = str(stalled_step) if stalled_step is not None else "no-step"
            if now - last_progress >= job.stall_seconds:
                if job_state.get("step_stall_reported") != stall_key:
                    events.append(
                        self._event_for_job(
                            job,
                            category="anomaly",
                            code="step_stalled",
                            severity="error",
                            observed_at=now,
                            details={
                                "last_step": stalled_step,
                                "stalled_seconds": now - last_progress,
                            },
                        )
                    )
                job_state["step_stall_reported"] = stall_key
        return events

    def _scheduler_states(self) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
        job_ids = ",".join(job.job_id for job in self.spec.jobs)
        commands = (
            ("squeue", "--noheader", "--jobs", job_ids, "--format=%i|%T"),
            (
                "sacct",
                "--noheader",
                "--parsable2",
                "--jobs",
                job_ids,
                "--format=JobIDRaw,State,ExitCode",
            ),
        )
        outputs: dict[str, CommandResult] = {}
        failures: dict[str, dict[str, Any]] = {}
        for argv in commands:
            try:
                result = self.command_runner(
                    argv, timeout_seconds=self.command_timeout_seconds
                )
            except Exception as error:
                failures[argv[0]] = {
                    "command": argv[0],
                    "reason": type(error).__name__,
                }
                continue
            if result.returncode != 0:
                failures[argv[0]] = {
                    "command": argv[0],
                    "reason": "nonzero_exit",
                    "returncode": result.returncode,
                }
                continue
            outputs[argv[0]] = result

        states = (
            _parse_scheduler_rows(outputs["squeue"].stdout)
            if "squeue" in outputs
            else {}
        )
        if "sacct" in outputs:
            for job_id, state in _parse_scheduler_rows(outputs["sacct"].stdout).items():
                states.setdefault(job_id, state)
        return states, failures

    def _observe_validation_report(
        self,
        job: JobSpec,
        job_state: dict[str, Any],
        *,
        facts: CheckpointFacts,
        now: float,
    ) -> list[MonitorEvent]:
        events: list[MonitorEvent] = []
        try:
            report_identity = _regular_file_identity(job.validation_report_path)
        except FileNotFoundError:
            job_state.pop("validation_report_identity", None)
            if not job_state.get("validation_report_missing", False):
                events.append(
                    self._event_for_job(
                        job,
                        category="checkpoint_issue",
                        code="checkpoint_validation_missing",
                        severity="error",
                        observed_at=now,
                        details={},
                    )
                )
            job_state["validation_report_missing"] = True
            return events
        except (OSError, ValueError) as error:
            reason = type(error).__name__
            if job_state.get("validation_report_read_failure") != reason:
                events.append(
                    self._event_for_job(
                        job,
                        category="checkpoint_issue",
                        code="checkpoint_validation_invalid",
                        severity="error",
                        observed_at=now,
                        details={"reason": reason},
                    )
                )
            job_state["validation_report_read_failure"] = reason
            return events

        job_state.pop("validation_report_read_failure", None)
        job_state.pop("validation_report_missing", None)
        if report_identity != job_state.get("validation_report_identity"):
            job_state["validation_report_identity"] = report_identity
            for issue in validate_checkpoint_report(
                job.validation_report_path, facts
            ):
                events.append(
                    self._event_for_job(
                        job,
                        category="checkpoint_issue",
                        code=issue.code,
                        severity="error",
                        observed_at=now,
                        details=issue.details,
                    )
                )
        return events

    def _observe_checkpoint(
        self,
        job: JobSpec,
        job_state: dict[str, Any],
        *,
        scheduler_state: str | None,
        now: float,
    ) -> list[MonitorEvent]:
        events: list[MonitorEvent] = []
        try:
            identity = _regular_file_identity(job.checkpoint_path)
        except FileNotFoundError:
            code = None
            if "checkpoint_identity" in job_state or job_state.get(
                "checkpoint_was_seen", False
            ):
                code = "checkpoint_disappeared"
            elif scheduler_state == "COMPLETED":
                code = "checkpoint_missing"
            if code is not None and job_state.get("checkpoint_missing_reported") != code:
                events.append(
                    self._event_for_job(
                        job,
                        category="checkpoint_issue",
                        code=code,
                        severity="error",
                        observed_at=now,
                        details={},
                    )
                )
                job_state["checkpoint_missing_reported"] = code
            job_state.pop("checkpoint_identity", None)
            job_state.pop("checkpoint_inspected_identity", None)
            job_state.pop("checkpoint_facts", None)
            job_state.pop("validation_report_identity", None)
            return events
        except (OSError, ValueError) as error:
            reason = type(error).__name__
            if job_state.get("checkpoint_read_failure") != reason:
                events.append(
                    self._event_for_job(
                        job,
                        category="checkpoint_issue",
                        code="checkpoint_read_failed",
                        severity="error",
                        observed_at=now,
                        details={"reason": reason},
                    )
                )
            job_state["checkpoint_read_failure"] = reason
            return events

        job_state.pop("checkpoint_missing_reported", None)
        job_state.pop("checkpoint_read_failure", None)
        job_state["checkpoint_was_seen"] = True
        if identity != job_state.get("checkpoint_inspected_identity"):
            job_state["checkpoint_identity"] = identity
            job_state["checkpoint_inspected_identity"] = identity
            job_state["checkpoint_last_change_at"] = now
            job_state.pop("checkpoint_stall_reported", None)
            job_state.pop("checkpoint_facts", None)
            job_state.pop("validation_report_identity", None)
            inspection = self.checkpoint_checker(
                job.checkpoint_path,
                identity,
                job.max_checkpoint_bytes,
            )
            for issue in inspection.issues:
                events.append(
                    self._event_for_job(
                        job,
                        category="checkpoint_issue",
                        code=issue.code,
                        severity="error",
                        observed_at=now,
                        details=issue.details,
                    )
                )
            if inspection.facts is not None:
                job_state["checkpoint_facts"] = {
                    "checkpoint_sha256": inspection.facts.checkpoint_sha256,
                    "checkpoint_size": inspection.facts.checkpoint_size,
                }

        facts_payload = job_state.get("checkpoint_facts")
        if isinstance(facts_payload, dict):
            digest = facts_payload.get("checkpoint_sha256")
            size = facts_payload.get("checkpoint_size")
            if (
                isinstance(digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                and isinstance(size, int)
                and not isinstance(size, bool)
                and size >= 0
            ):
                events.extend(
                    self._observe_validation_report(
                        job,
                        job_state,
                        facts=CheckpointFacts(digest, size),
                        now=now,
                    )
                )

        if scheduler_state == "RUNNING":
            changed_at = float(job_state.setdefault("checkpoint_last_change_at", now))
            if now - changed_at >= job.checkpoint_stale_seconds:
                identity_key = json.dumps(identity, separators=(",", ":"))
                if job_state.get("checkpoint_stall_reported") != identity_key:
                    events.append(
                        self._event_for_job(
                            job,
                            category="checkpoint_issue",
                            code="checkpoint_stalled",
                            severity="warning",
                            observed_at=now,
                            details={"stalled_seconds": now - changed_at},
                        )
                    )
                job_state["checkpoint_stall_reported"] = identity_key
        return events

    def _observe_gpu_window(
        self,
        job: JobSpec,
        job_state: dict[str, Any],
        *,
        scheduler_state: str | None,
        now: float,
    ) -> list[MonitorEvent]:
        if not job.utilization_required:
            return []
        events: list[MonitorEvent] = []
        if job.gpu_samples_path is None:
            reason = "unconfigured"
            identity = None
            text = ""
        else:
            try:
                identity, text = _bounded_tail(
                    job.gpu_samples_path, self.max_tail_bytes
                )
                reason = None
            except FileNotFoundError:
                identity = None
                text = ""
                reason = "missing"
            except (OSError, ValueError) as error:
                identity = None
                text = ""
                reason = type(error).__name__
        if reason is not None:
            if scheduler_state in {"RUNNING", "COMPLETED"} | FAILED_SCHEDULER_STATES:
                if job_state.get("gpu_read_failure") != reason:
                    events.append(
                        self._event_for_job(
                            job,
                            category="anomaly",
                            code="gpu_samples_missing"
                            if reason in {"missing", "unconfigured"}
                            else "gpu_samples_read_failed",
                            severity="error",
                            observed_at=now,
                            details={"reason": reason},
                        )
                    )
                job_state["gpu_read_failure"] = reason
            return events
        job_state.pop("gpu_read_failure", None)

        if identity != job_state.get("gpu_identity"):
            records, invalid_lines = parse_gpu_records(text)
            evaluation = evaluate_gpu_window(
                records,
                utilization_required=True,
                expected_gpu_uuids=job.expected_gpu_uuids,
                expected_gpu_count=job.expected_gpu_count,
            )
            job_state["gpu_identity"] = identity
            job_state["gpu_window_complete"] = evaluation.complete
            if invalid_lines:
                events.append(
                    self._event_for_job(
                        job,
                        category="anomaly",
                        code="gpu_sample_invalid",
                        severity="error",
                        observed_at=now,
                        details={"invalid_records": invalid_lines},
                    )
                )
            for issue in evaluation.issues:
                events.append(
                    self._event_for_job(
                        job,
                        category="anomaly",
                        code=issue.code,
                        severity="error",
                        observed_at=now,
                        details=issue.details,
                    )
                )
            if evaluation.complete:
                job_state.pop("gpu_incomplete_reported", None)

        terminal = scheduler_state == "COMPLETED" or scheduler_state in FAILED_SCHEDULER_STATES
        if terminal and not job_state.get("gpu_window_complete", False):
            if not job_state.get("gpu_incomplete_reported", False):
                events.append(
                    self._event_for_job(
                        job,
                        category="anomaly",
                        code="gpu_window_incomplete",
                        severity="error",
                        observed_at=now,
                        details={},
                    )
                )
            job_state["gpu_incomplete_reported"] = True
        return events

    def poll(self, state: dict[str, Any]) -> PollResult:
        new_state = copy.deepcopy(state) if state else {"schema_version": 1, "jobs": {}}
        job_states = new_state.setdefault("jobs", {})
        events: list[MonitorEvent] = []
        now = self.clock.time()
        scheduler_states, scheduler_failures = self._scheduler_states()
        previous_query_failures = new_state.get("scheduler_query_failures", {})
        for command in ("squeue", "sacct"):
            failure = scheduler_failures.get(command)
            if failure is not None and previous_query_failures.get(command) != failure:
                events.append(
                    MonitorEvent(
                        "anomaly",
                        "scheduler_query_failed",
                        "error",
                        now,
                        None,
                        None,
                        None,
                        failure,
                    )
                )
        new_state["scheduler_query_failures"] = scheduler_failures
        for job in self.spec.jobs:
            job_state = job_states.setdefault(job.job_id, {})
            current = scheduler_states.get(job.job_id)
            previous = job_state.get("scheduler_state")
            if current is None and not scheduler_failures:
                if not job_state.get("scheduler_missing_reported", False):
                    events.append(
                        self._event_for_job(
                            job,
                            category="anomaly",
                            code="scheduler_state_missing",
                            severity="error",
                            observed_at=now,
                            details={},
                        )
                    )
                job_state["scheduler_missing_reported"] = True
            if current is not None:
                job_state.pop("scheduler_missing_reported", None)
                if previous is not None and previous != current:
                    events.append(
                        MonitorEvent(
                            "stage_transition",
                            "scheduler_state_changed",
                            "info",
                            now,
                            job.job_id,
                            job.arm,
                            job.stage,
                            {"from": previous, "to": current},
                        )
                    )
                job_state["scheduler_state"] = current
                if current in FAILED_SCHEDULER_STATES:
                    if job_state.get("scheduler_failure_reported") != current:
                        events.append(
                            MonitorEvent(
                                "anomaly",
                                "scheduler_failure",
                                "error",
                                now,
                                job.job_id,
                                job.arm,
                                job.stage,
                                {"state": current},
                            )
                        )
                    job_state["scheduler_failure_reported"] = current
                else:
                    job_state.pop("scheduler_failure_reported", None)
            events.extend(
                self._observe_log(
                    job,
                    job_state,
                    scheduler_state=current,
                    now=now,
                )
            )
            events.extend(
                self._observe_checkpoint(
                    job,
                    job_state,
                    scheduler_state=current,
                    now=now,
                )
            )
            events.extend(
                self._observe_gpu_window(
                    job,
                    job_state,
                    scheduler_state=current,
                    now=now,
                )
            )

        try:
            stats = self.statvfs(self.spec.disk_path)
            available_bytes = int(stats.f_bavail) * int(stats.f_frsize)
        except Exception as error:
            failure = {"reason": type(error).__name__}
            if new_state.get("disk_query_failure") != failure:
                events.append(
                    MonitorEvent(
                        "anomaly",
                        "disk_query_failed",
                        "error",
                        now,
                        None,
                        None,
                        None,
                        failure,
                    )
                )
            new_state["disk_query_failure"] = failure
            return PollResult(tuple(events), new_state)
        new_state.pop("disk_query_failure", None)
        if available_bytes < DISK_SUBMIT_BLOCK_BYTES:
            disk_band = "submit_block"
        elif available_bytes < DISK_WARNING_BYTES:
            disk_band = "warning"
        else:
            disk_band = "normal"
        previous_disk_band = new_state.get("disk_band")
        if disk_band != "normal" and previous_disk_band != disk_band:
            events.append(
                MonitorEvent(
                    "anomaly",
                    "disk_space_low",
                    disk_band,
                    now,
                    None,
                    None,
                    None,
                    {
                        "available_bytes": available_bytes,
                        "warning_below_bytes": DISK_WARNING_BYTES,
                        "submit_block_below_bytes": DISK_SUBMIT_BLOCK_BYTES,
                    },
                )
            )
        new_state["disk_band"] = disk_band
        return PollResult(tuple(events), new_state)


def evaluate_gpu_window(
    records: Sequence[GpuRecord],
    *,
    utilization_required: bool,
    expected_gpu_uuids: Sequence[str] = (),
    expected_gpu_count: int = 0,
) -> GpuWindowEvaluation:
    """Evaluate a complete, explicitly marked GPU-active interval."""

    if not utilization_required:
        return GpuWindowEvaluation(False, {}, {}, ())

    start_records = [record for record in records if record.kind == "training_start"]
    starts = [record.monotonic_seconds for record in start_records]
    ends = [record.monotonic_seconds for record in records if record.kind == "training_end"]
    if len(starts) != 1 or not ends:
        return GpuWindowEvaluation(False, {}, {}, ())
    if len(ends) != 1 or not math.isfinite(starts[0]) or not math.isfinite(ends[0]) or ends[0] <= starts[0]:
        return GpuWindowEvaluation(
            True,
            {},
            {},
            (MonitorIssue("gpu_window_markers_invalid", {}),),
        )

    start, end = starts[0], ends[0]
    samples: dict[str, list[tuple[float, float]]] = defaultdict(list)
    invalid_sample = False
    for record in records:
        if record.kind != "sample":
            continue
        if not (start <= record.monotonic_seconds <= end):
            continue
        if (
            not record.gpu_uuid
            or record.utilization_percent is None
            or not math.isfinite(record.monotonic_seconds)
            or not math.isfinite(record.utilization_percent)
            or not 0.0 <= record.utilization_percent <= 100.0
        ):
            invalid_sample = True
            continue
        samples[record.gpu_uuid].append(
            (record.monotonic_seconds, record.utilization_percent)
        )

    issues: list[MonitorIssue] = []
    expected_uuids = tuple(expected_gpu_uuids)
    start_attestation = start_records[0]
    bound_uuids = start_attestation.expected_gpu_uuids
    bound_count = start_attestation.expected_gpu_count
    if not expected_uuids and bound_uuids is not None:
        expected_uuids = bound_uuids
    if bound_count is not None and (
        bound_count != expected_gpu_count
        or bound_uuids is None
        or len(bound_uuids) != bound_count
    ):
        issues.append(MonitorIssue("gpu_expectation_mismatch", {}))
    if (
        tuple(expected_gpu_uuids)
        and bound_uuids is not None
        and set(bound_uuids) != set(expected_gpu_uuids)
    ):
        issues.append(MonitorIssue("gpu_expectation_mismatch", {}))
    if (
        expected_gpu_count < 1
        or len(expected_uuids) != expected_gpu_count
        or len(set(expected_uuids)) != len(expected_uuids)
        or any(not gpu_uuid for gpu_uuid in expected_uuids)
    ):
        issues.append(MonitorIssue("gpu_expectation_invalid", {}))
    if invalid_sample or not samples:
        issues.append(MonitorIssue("gpu_sample_invalid", {}))

    actual_uuids = set(samples)
    if set(expected_uuids) != actual_uuids or len(actual_uuids) != expected_gpu_count:
        issues.append(
            MonitorIssue(
                "gpu_set_mismatch",
                {
                    "expected_gpu_uuids": sorted(expected_uuids),
                    "observed_gpu_uuids": sorted(actual_uuids),
                    "expected_gpu_count": expected_gpu_count,
                    "observed_gpu_count": len(actual_uuids),
                },
            )
        )

    counts: dict[str, int] = {}
    means: dict[str, float] = {}
    for gpu_uuid, values in sorted(samples.items()):
        ordered = values
        counts[gpu_uuid] = len(ordered)
        means[gpu_uuid] = statistics.fmean(value for _, value in ordered)
        if len(ordered) < MIN_GPU_SAMPLES:
            issues.append(
                MonitorIssue(
                    "gpu_samples_insufficient",
                    {
                        "gpu_uuid": gpu_uuid,
                        "actual": len(ordered),
                        "required": MIN_GPU_SAMPLES,
                    },
                )
            )
        gaps = [
            current[0] - previous[0]
            for previous, current in zip(ordered, ordered[1:])
        ]
        if any(
            gap < MIN_GPU_SAMPLE_GAP_SECONDS
            or gap > MAX_GPU_SAMPLE_GAP_SECONDS
            for gap in gaps
        ):
            issues.append(
                MonitorIssue(
                    "gpu_sample_cadence_invalid",
                    {"gpu_uuid": gpu_uuid},
                )
            )
        edge_gaps = (ordered[0][0] - start, end - ordered[-1][0])
        if any(gap > MAX_GPU_SAMPLE_GAP_SECONDS for gap in edge_gaps):
            issues.append(
                MonitorIssue(
                    "gpu_sample_cadence_invalid",
                    {"gpu_uuid": gpu_uuid},
                )
            )
        if means[gpu_uuid] < MIN_GPU_MEAN_PERCENT:
            issues.append(
                MonitorIssue(
                    "gpu_utilization_low",
                    {
                        "gpu_uuid": gpu_uuid,
                        "mean_percent": means[gpu_uuid],
                        "required_percent": MIN_GPU_MEAN_PERCENT,
                    },
                )
            )

    return GpuWindowEvaluation(True, counts, means, tuple(issues))


def _open_exact_directory(path: Path) -> int:
    if not path.is_absolute():
        raise ValueError("monitor writable directories must be absolute")
    lexical = Path(os.path.abspath(path))
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError("monitor writable directory does not exist") from error
    if resolved != lexical:
        raise ValueError("monitor writable directory must not traverse symlinks")
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lexical, flags)
    if not stat.S_ISDIR(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("monitor writable path is not a directory")
    return fd


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("monitor write made no progress")
        view = view[written:]


class MonitorStore:
    """The monitor's only writable surface: explicit state and event directories."""

    def __init__(
        self,
        state_dir: Path,
        event_ledger_dir: Path,
        *,
        max_event_ledger_bytes: int = DEFAULT_MAX_EVENT_LEDGER_BYTES,
    ) -> None:
        state_fd = _open_exact_directory(state_dir)
        event_fd = _open_exact_directory(event_ledger_dir)
        os.close(state_fd)
        os.close(event_fd)
        if state_dir == event_ledger_dir:
            raise ValueError("state and event-ledger directories must be distinct")
        if (
            isinstance(max_event_ledger_bytes, bool)
            or not isinstance(max_event_ledger_bytes, int)
            or max_event_ledger_bytes < 1
        ):
            raise ValueError("max_event_ledger_bytes must be a positive integer")
        self.state_dir = state_dir
        self.event_ledger_dir = event_ledger_dir
        self.max_event_ledger_bytes = max_event_ledger_bytes
        self._state_bytes: bytes | None = None

    def load_state(self) -> dict[str, Any]:
        directory_fd = _open_exact_directory(self.state_dir)
        try:
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                fd = os.open("state.json", flags, dir_fd=directory_fd)
            except FileNotFoundError:
                self._state_bytes = None
                return {}
            try:
                metadata = os.fstat(fd)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_size > MAX_MONITOR_STATE_BYTES
                ):
                    raise ValueError("monitor state file is invalid")
                chunks: list[bytes] = []
                remaining = metadata.st_size
                while remaining:
                    chunk = os.read(fd, remaining)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                raw = b"".join(chunks)
            finally:
                os.close(fd)
        finally:
            os.close(directory_fd)
        state = _strict_json_loads(raw)
        if not isinstance(state, dict):
            raise ValueError("monitor state root must be an object")
        if state and (
            state.get("schema_version") != 1 or not isinstance(state.get("jobs"), dict)
        ):
            raise ValueError("monitor state schema is invalid")
        self._state_bytes = _canonical_json_bytes(state)
        return state

    def save_state(self, state: dict[str, Any]) -> None:
        payload = _canonical_json_bytes(state)
        if payload == self._state_bytes:
            return
        if len(payload) > MAX_MONITOR_STATE_BYTES:
            raise ValueError("monitor state exceeds byte ceiling")
        directory_fd = _open_exact_directory(self.state_dir)
        temporary_name = f".state.{secrets.token_hex(12)}.tmp"
        temporary_created = False
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
            temporary_created = True
            try:
                _write_all(fd, payload)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(
                temporary_name,
                "state.json",
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            temporary_created = False
            os.fsync(directory_fd)
            self._state_bytes = payload
        finally:
            if temporary_created:
                try:
                    os.unlink(temporary_name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
            os.close(directory_fd)

    def append_events(self, payloads: Sequence[bytes]) -> None:
        if not payloads:
            return
        directory_fd = _open_exact_directory(self.event_ledger_dir)
        try:
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open("events.jsonl", flags, 0o600, dir_fd=directory_fd)
            try:
                metadata = os.fstat(fd)
                addition_bytes = sum(len(payload) + 1 for payload in payloads)
                if metadata.st_size + addition_bytes > self.max_event_ledger_bytes:
                    raise ValueError("event ledger exceeds byte ceiling")
                for payload in payloads:
                    _write_all(fd, payload + b"\n")
                os.fsync(fd)
            finally:
                os.close(fd)
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


@contextmanager
def nonblocking_monitor_lock(state_dir: Path):
    """Acquire the fixed monitor lock without ever waiting for another run."""

    directory_fd = _open_exact_directory(state_dir)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    lock_fd = os.open("monitor.lock", flags, 0o600, dir_fd=directory_fd)
    os.close(directory_fd)
    acquired = False
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            yield False
            return
        yield True
    finally:
        if acquired:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def event_json_bytes(event: MonitorEvent) -> bytes:
    body = {
        "schema_version": 1,
        "category": event.category,
        "code": event.code,
        "severity": event.severity,
        "observed_at": event.observed_at,
        "job_id": event.job_id,
        "arm": event.arm,
        "stage": event.stage,
        "details": event.details,
    }
    return _canonical_json_bytes(
        {**body, "event_id": hashlib.sha256(_canonical_json_bytes(body)).hexdigest()}
    )


def run_bounded_monitor(
    monitor: FormalIdentityMonitor,
    store: MonitorStore,
    *,
    duration_seconds: float,
    poll_interval_seconds: float,
    once: bool,
    output: io.TextIOBase,
) -> None:
    duration = _positive_number(duration_seconds, where="duration_seconds")
    interval = _positive_number(
        poll_interval_seconds, where="poll_interval_seconds"
    )
    state = store.load_state()
    deadline = monitor.clock.monotonic() + duration
    first_poll = True
    while first_poll or monitor.clock.monotonic() < deadline:
        first_poll = False
        result = monitor.poll(state)
        payloads = [event_json_bytes(event) for event in result.events]
        store.append_events(payloads)
        store.save_state(result.state)
        for payload in payloads:
            output.write(payload.decode("utf-8") + "\n")
        if payloads:
            output.flush()
        state = result.state
        if once:
            break
        remaining = deadline - monitor.clock.monotonic()
        if remaining <= 0:
            break
        monitor.clock.sleep(min(interval, remaining))


def _arg_positive_float(value: str) -> float:
    try:
        return _positive_number(float(value), where="argument")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _arg_positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("argument must be a positive integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("argument must be a positive integer")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only, event-only monitor for a formal identity study."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--event-ledger-dir", required=True, type=Path)
    parser.add_argument(
        "--duration-seconds", type=_arg_positive_float, default=DEFAULT_DURATION_SECONDS
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=_arg_positive_float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--command-timeout-seconds",
        type=_arg_positive_float,
        default=DEFAULT_COMMAND_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--max-tail-bytes", type=_arg_positive_int, default=DEFAULT_MAX_TAIL_BYTES
    )
    parser.add_argument("--once", action="store_true")
    return parser


def _path_is_within(path: Path, directory: Path) -> bool:
    canonical_path = Path(os.path.realpath(os.path.abspath(path)))
    canonical_directory = Path(os.path.realpath(os.path.abspath(directory)))
    try:
        canonical_path.relative_to(canonical_directory)
    except ValueError:
        return False
    return True


def _validate_write_scope(
    spec: MonitorSpec, manifest_path: Path, state_dir: Path, event_dir: Path
) -> None:
    writable = (state_dir, event_dir)
    observed = [manifest_path]
    for job in spec.jobs:
        observed.extend(
            [job.log_path, job.checkpoint_path, job.validation_report_path]
        )
        if job.gpu_samples_path is not None:
            observed.append(job.gpu_samples_path)
    if any(_path_is_within(path, directory) for path in observed for directory in writable):
        raise ValueError("monitored artifacts must be outside monitor writable directories")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.manifest.is_absolute():
        raise ValueError("--manifest must be absolute")
    if args.max_tail_bytes > MAX_ALLOWED_TAIL_BYTES:
        raise ValueError("--max-tail-bytes exceeds the fixed monitor ceiling")
    spec = load_monitor_spec(args.manifest)
    store = MonitorStore(
        args.state_dir,
        args.event_ledger_dir,
        max_event_ledger_bytes=spec.max_event_ledger_bytes,
    )
    _validate_write_scope(
        spec, args.manifest, args.state_dir, args.event_ledger_dir
    )
    with nonblocking_monitor_lock(args.state_dir) as acquired:
        if not acquired:
            return 0
        monitor = FormalIdentityMonitor(
            spec,
            command_timeout_seconds=args.command_timeout_seconds,
            max_tail_bytes=args.max_tail_bytes,
        )
        run_bounded_monitor(
            monitor,
            store,
            duration_seconds=args.duration_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            once=args.once,
            output=sys.stdout,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"monitor failed: {error}", file=sys.stderr)
        raise SystemExit(2)
