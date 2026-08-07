#!/usr/bin/env python3
"""Bounded, allocation-scoped GPU telemetry for formal production jobs.

The production contract is deliberately different from the one-second H100
smoke window.  It records one canonical JSONL stream at a fixed 60-second
cadence, never queries GPUs outside the one CUDA-visible allocation token, and
reserves enough space for an explicit ``end`` or ``abort`` record.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Mapping


SCHEMA_VERSION = 1
FORMAT = "formal_production_gpu_jsonl_v1"
CADENCE_SECONDS = 60
CADENCE_DRIFT_SECONDS = 15
MINIMUM_MEAN_SAMPLES = 10
LOW_UTILIZATION_THRESHOLD_PERCENT = 80.0
MAX_GPU_ID_CHARS = 128
MAX_MONOTONIC_NS = (1 << 63) - 1
QUERY_TIMEOUT_SECONDS = 15
SLURM_DURATION = re.compile(
    r"^(?:(?P<days>[1-9][0-9]{0,3})-)?"
    r"(?P<hours>[0-9]{2}):(?P<minutes>[0-5][0-9]):(?P<seconds>[0-5][0-9])$"
)
GPU_TOKEN = re.compile(
    r"^(?:[0-9]+|GPU-[A-Za-z0-9._:-]+|MIG-[A-Za-z0-9._:/-]+)$"
)
GPU_UUID = re.compile(
    r"^(?:GPU-[A-Za-z0-9._:-]+|MIG-[A-Za-z0-9._:/-]+)$"
)
ABORT_REASONS = frozenset(
    {
        "byte_ceiling_exceeded",
        "cadence_drift",
        "control_invalid",
        "gpu_query_failed",
        "gpu_uuid_drift",
        "sample_ceiling_exceeded",
        "signal_termination",
        "supervisor_abort",
        "training_failed",
    }
)
EXTERNAL_ABORT_REASONS = frozenset(
    {"signal_termination", "supervisor_abort", "training_failed"}
)


class TelemetryFailure(RuntimeError):
    """A fail-closed telemetry error with a bounded canonical reason."""

    def __init__(self, reason: str, message: str):
        if reason not in ABORT_REASONS:
            raise ValueError("telemetry failure reason is not canonical")
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class TelemetryContract:
    schema_version: int
    format: str
    cadence_seconds: int
    cadence_drift_seconds: int
    minimum_mean_samples: int
    low_utilization_threshold_percent: float
    time_limit: str
    time_limit_seconds: int
    sample_ceiling: int
    byte_ceiling: int


@dataclass(frozen=True)
class TelemetrySummary:
    terminal_kind: str | None
    abort_reason: str | None
    sample_count: int
    utilization_mean_percent: float | None
    last_sample_monotonic_ns: int | None


@dataclass(frozen=True)
class TelemetryEvidence:
    size_bytes: int
    sha256: str
    summary: TelemetrySummary


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite telemetry number")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("telemetry object key is invalid")
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    raise ValueError("telemetry value has an unsupported type")


def canonical_line(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _normalize(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def parse_slurm_duration(value: str) -> int:
    if not isinstance(value, str):
        raise ValueError("TimeLimit is not a canonical Slurm duration")
    match = SLURM_DURATION.fullmatch(value)
    if match is None or int(match.group("hours")) > 23:
        raise ValueError("TimeLimit is not a canonical Slurm duration")
    seconds = (
        (int(match.group("days") or "0") * 24 + int(match.group("hours")))
        * 3600
        + int(match.group("minutes")) * 60
        + int(match.group("seconds"))
    )
    if seconds <= 0:
        raise ValueError("TimeLimit must be positive")
    return seconds


def _max_record_lengths(
    *,
    time_limit: str,
    time_limit_seconds: int,
    sample_ceiling: int,
) -> tuple[int, int, int]:
    max_uuid = "GPU-" + "x" * (MAX_GPU_ID_CHARS - 4)
    start = canonical_line(
        {
            "cadence_drift_seconds": CADENCE_DRIFT_SECONDS,
            "cadence_seconds": CADENCE_SECONDS,
            "expected_gpu_uuid": max_uuid,
            "format": FORMAT,
            "kind": "start",
            "monotonic_ns": MAX_MONOTONIC_NS,
            "sample_ceiling": sample_ceiling,
            "schema_version": SCHEMA_VERSION,
            "time_limit": time_limit,
            "time_limit_seconds": time_limit_seconds,
        }
    )
    sample = canonical_line(
        {
            "gpu_uuid": max_uuid,
            "kind": "sample",
            "monotonic_ns": MAX_MONOTONIC_NS,
            "scheduled_monotonic_ns": MAX_MONOTONIC_NS,
            "sequence": sample_ceiling - 1,
            "utilization_percent": 100,
        }
    )
    terminal_values = [
        {
            "kind": "end",
            "monotonic_ns": MAX_MONOTONIC_NS,
            "sample_count": sample_ceiling,
        }
    ]
    terminal_values.extend(
        {
            "kind": "abort",
            "monotonic_ns": MAX_MONOTONIC_NS,
            "reason": reason,
            "sample_count": sample_ceiling,
        }
        for reason in ABORT_REASONS
    )
    return (
        len(start),
        len(sample),
        max(len(canonical_line(value)) for value in terminal_values),
    )


def telemetry_contract(time_limit: str) -> TelemetryContract:
    seconds = parse_slurm_duration(time_limit)
    # An immediate sample at t=0 followed by one sample at every full cadence
    # boundary is the largest possible stream before the frozen allocation end.
    sample_ceiling = seconds // CADENCE_SECONDS + 1
    start_bytes, sample_bytes, terminal_bytes = _max_record_lengths(
        time_limit=time_limit,
        time_limit_seconds=seconds,
        sample_ceiling=sample_ceiling,
    )
    return TelemetryContract(
        schema_version=SCHEMA_VERSION,
        format=FORMAT,
        cadence_seconds=CADENCE_SECONDS,
        cadence_drift_seconds=CADENCE_DRIFT_SECONDS,
        minimum_mean_samples=MINIMUM_MEAN_SAMPLES,
        low_utilization_threshold_percent=LOW_UTILIZATION_THRESHOLD_PERCENT,
        time_limit=time_limit,
        time_limit_seconds=seconds,
        sample_ceiling=sample_ceiling,
        byte_ceiling=start_bytes + sample_ceiling * sample_bytes + terminal_bytes,
    )


def telemetry_contract_sha256(contract: TelemetryContract) -> str:
    if contract != telemetry_contract(contract.time_limit):
        raise ValueError("telemetry contract is not canonical")
    return hashlib.sha256(
        json.dumps(
            asdict(contract),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _validate_gpu_token(value: str, *, uuid: bool = False) -> str:
    pattern = GPU_UUID if uuid else GPU_TOKEN
    if (
        not isinstance(value, str)
        or len(value) > MAX_GPU_ID_CHARS
        or pattern.fullmatch(value) is None
    ):
        raise ValueError("allocated GPU token/UUID is invalid")
    # The broad suffix alphabet intentionally accommodates both current and
    # legacy NVIDIA UUID spellings, but it must not accept two concatenated
    # top-level tokens.
    if value.startswith("GPU-") and (
        "GPU-" in value[4:] or "MIG-" in value[4:]
    ):
        raise ValueError("allocated GPU token/UUID is invalid")
    if value.startswith("MIG-") and "MIG-" in value[4:]:
        raise ValueError("allocated GPU token/UUID is invalid")
    return value


def _stat_signature(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


class _ScopedNvidiaSmi:
    def __init__(
        self,
        executable: Path,
        expected_sha256: str,
        token: str,
        expected_uuid: str,
    ):
        if not executable.is_absolute():
            raise ValueError("nvidia-smi must be an absolute executable")
        if (
            not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        ):
            raise ValueError("nvidia-smi SHA-256 is invalid")
        token = _validate_gpu_token(token)
        expected_uuid = _validate_gpu_token(expected_uuid, uuid=True)
        try:
            before = executable.lstat()
        except OSError as error:
            raise ValueError("nvidia-smi is unavailable") from error
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_mode & 0o111 == 0
            or executable.is_symlink()
        ):
            raise ValueError("nvidia-smi must be a physical executable")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self._fd = os.open(executable, flags)
        opened = os.fstat(self._fd)
        if (
            _stat_signature(opened) != _stat_signature(before)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_mode & 0o111 == 0
        ):
            os.close(self._fd)
            raise ValueError("nvidia-smi changed while it was opened")
        if opened.st_size > 128 * 1024 * 1024:
            os.close(self._fd)
            raise ValueError("nvidia-smi exceeds its executable byte ceiling")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(self._fd, min(1024 * 1024, opened.st_size - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > opened.st_size:
                os.close(self._fd)
                raise ValueError("nvidia-smi changed while it was hashed")
            digest.update(chunk)
        after = os.fstat(self._fd)
        try:
            path_after = executable.lstat()
        except OSError as error:
            os.close(self._fd)
            raise ValueError("nvidia-smi disappeared while it was hashed") from error
        if (
            total != opened.st_size
            or _stat_signature(after) != _stat_signature(opened)
            or _stat_signature(path_after) != _stat_signature(after)
            or digest.hexdigest() != expected_sha256
        ):
            os.close(self._fd)
            raise ValueError("nvidia-smi stable SHA-256 verification failed")
        os.lseek(self._fd, 0, os.SEEK_SET)
        self._token = token
        self._expected_uuid = expected_uuid

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def query(self, fields: str) -> tuple[str, ...]:
        if self._fd < 0 or fields not in {"uuid", "uuid,utilization.gpu"}:
            raise TelemetryFailure("gpu_query_failed", "invalid GPU query")
        try:
            completed = subprocess.run(
                [
                    f"/proc/self/fd/{self._fd}",
                    f"--id={self._token}",
                    f"--query-gpu={fields}",
                    "--format=csv,noheader,nounits",
                ],
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=QUERY_TIMEOUT_SECONDS,
                pass_fds=(self._fd,),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise TelemetryFailure(
                "gpu_query_failed", "allocation-scoped nvidia-smi query failed"
            ) from error
        if (
            completed.returncode != 0
            or completed.stderr
            or len(completed.stdout.encode("utf-8")) > 512
        ):
            raise TelemetryFailure(
                "gpu_query_failed", "allocation-scoped nvidia-smi query failed"
            )
        lines = completed.stdout.splitlines()
        if len(lines) != 1 or not lines[0] or lines[0] != lines[0].strip():
            raise TelemetryFailure(
                "gpu_query_failed", "nvidia-smi did not return exactly one row"
            )
        values = tuple(part.strip() for part in lines[0].split(","))
        expected_fields = 1 if fields == "uuid" else 2
        if len(values) != expected_fields or any(not value for value in values):
            raise TelemetryFailure("gpu_query_failed", "nvidia-smi row is malformed")
        try:
            observed_uuid = _validate_gpu_token(values[0], uuid=True)
        except ValueError as error:
            raise TelemetryFailure(
                "gpu_query_failed", "nvidia-smi UUID is malformed"
            ) from error
        if observed_uuid != self._expected_uuid:
            raise TelemetryFailure(
                "gpu_uuid_drift", "allocated GPU UUID changed during telemetry"
            )
        if fields == "uuid":
            return (observed_uuid,)
        if re.fullmatch(r"(?:0|[1-9][0-9]{0,2})", values[1]) is None:
            raise TelemetryFailure(
                "gpu_query_failed", "GPU utilization is not an integer percentage"
            )
        utilization = int(values[1])
        if utilization > 100:
            raise TelemetryFailure(
                "gpu_query_failed", "GPU utilization is outside [0,100]"
            )
        return observed_uuid, str(utilization)


class _BoundedWriter:
    def __init__(self, path: Path, contract: TelemetryContract):
        if not path.is_absolute():
            raise ValueError("telemetry output path must be absolute")
        parent = path.parent
        metadata = parent.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or parent.is_symlink():
            raise ValueError("telemetry parent must be a physical directory")
        if path.exists() or path.is_symlink():
            raise ValueError("telemetry output must be fresh")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
        )
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self._fd = os.open(path, flags, 0o400)
        opened = os.fstat(self._fd)
        if not stat.S_ISREG(opened.st_mode):
            os.close(self._fd)
            raise ValueError("telemetry output is not a regular file")
        self.contract = contract
        (
            self.start_ceiling,
            self.sample_ceiling,
            self.terminal_ceiling,
        ) = _max_record_lengths(
            time_limit=contract.time_limit,
            time_limit_seconds=contract.time_limit_seconds,
            sample_ceiling=contract.sample_ceiling,
        )
        self.bytes_written = 0
        self.terminal_written = False

    def close(self) -> None:
        if self._fd >= 0:
            os.fsync(self._fd)
            os.close(self._fd)
            self._fd = -1

    def write(self, value: Mapping[str, Any]) -> None:
        kind = value.get("kind")
        if kind == "start":
            record_ceiling = self.start_ceiling
        elif kind == "sample":
            record_ceiling = self.sample_ceiling
        elif kind in {"end", "abort"}:
            record_ceiling = self.terminal_ceiling
        else:
            raise TelemetryFailure("control_invalid", "telemetry record kind is invalid")
        raw = canonical_line(value)
        reserve = 0 if kind in {"end", "abort"} else self.terminal_ceiling
        if (
            len(raw) > record_ceiling
            or self.bytes_written + len(raw) + reserve > self.contract.byte_ceiling
        ):
            raise TelemetryFailure(
                "byte_ceiling_exceeded", "telemetry byte ceiling would be exceeded"
            )
        offset = 0
        while offset < len(raw):
            written = os.write(self._fd, raw[offset:])
            if written <= 0:
                raise OSError("short telemetry write")
            offset += written
        self.bytes_written += len(raw)
        os.fsync(self._fd)
        if kind in {"end", "abort"}:
            self.terminal_written = True


def _path_state(path: Path, *, empty: bool) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise TelemetryFailure("control_invalid", "telemetry control is invalid")
    if empty and metadata.st_size != 0:
        raise TelemetryFailure("control_invalid", "telemetry control is not empty")
    return True


def _read_abort_reason(path: Path) -> str | None:
    if not _path_state(path, empty=False):
        return None
    before = path.lstat()
    if before.st_size > 64:
        raise TelemetryFailure("control_invalid", "abort control is oversized")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        raw = os.read(fd, 65)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (
        len(raw) > 64
        or _stat_signature(opened) != _stat_signature(before)
        or _stat_signature(after) != _stat_signature(opened)
        or raw.decode("ascii", errors="strict") not in {
            f"{reason}\n" for reason in EXTERNAL_ABORT_REASONS
        }
    ):
        raise TelemetryFailure("control_invalid", "abort control is not canonical")
    return raw.decode("ascii").strip()


def _create_ready(path: Path) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ValueError("telemetry ready path must be absolute and fresh")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
    )
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o400)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _terminal_record(
    kind: str, monotonic_ns: int, sample_count: int, reason: str | None = None
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "kind": kind,
        "monotonic_ns": monotonic_ns,
        "sample_count": sample_count,
    }
    if reason is not None:
        value["reason"] = reason
    return value


def record_telemetry(
    *,
    output_path: Path,
    ready_path: Path,
    end_path: Path,
    abort_path: Path,
    nvidia_smi: Path,
    expected_nvidia_smi_sha256: str,
    visible_gpu_token: str,
    expected_gpu_uuid: str,
    contract: TelemetryContract,
    expected_sample_ceiling: int,
    expected_byte_ceiling: int,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> TelemetrySummary:
    if contract != telemetry_contract(contract.time_limit):
        raise ValueError("telemetry contract is not canonical")
    if (
        expected_sample_ceiling != contract.sample_ceiling
        or expected_byte_ceiling != contract.byte_ceiling
    ):
        raise ValueError("exported telemetry ceilings differ from frozen TimeLimit")
    if any(
        not path.is_absolute()
        for path in (ready_path, end_path, abort_path, output_path)
    ):
        raise ValueError("telemetry paths must be absolute")
    if any(path.exists() or path.is_symlink() for path in (ready_path, end_path, abort_path)):
        raise ValueError("telemetry controls must be fresh")
    expected_gpu_uuid = _validate_gpu_token(expected_gpu_uuid, uuid=True)
    scoped = _ScopedNvidiaSmi(
        nvidia_smi,
        expected_nvidia_smi_sha256,
        visible_gpu_token,
        expected_gpu_uuid,
    )
    writer: _BoundedWriter | None = None
    samples: list[int] = []
    last_sample_ns: int | None = None
    start_ns: int | None = None
    try:
        scoped.query("uuid")
        start_ns = monotonic_ns()
        if not isinstance(start_ns, int) or not 0 <= start_ns <= MAX_MONOTONIC_NS:
            raise TelemetryFailure("cadence_drift", "monotonic clock is invalid")
        writer = _BoundedWriter(output_path, contract)
        writer.write(
            {
                "cadence_drift_seconds": contract.cadence_drift_seconds,
                "cadence_seconds": contract.cadence_seconds,
                "expected_gpu_uuid": expected_gpu_uuid,
                "format": contract.format,
                "kind": "start",
                "monotonic_ns": start_ns,
                "sample_ceiling": contract.sample_ceiling,
                "schema_version": contract.schema_version,
                "time_limit": contract.time_limit,
                "time_limit_seconds": contract.time_limit_seconds,
            }
        )
        try:
            _create_ready(ready_path)
        except (OSError, ValueError) as error:
            raise TelemetryFailure(
                "control_invalid", "telemetry ready control could not be published"
            ) from error
        while True:
            end_requested = _path_state(end_path, empty=True)
            abort_reason = _read_abort_reason(abort_path)
            if end_requested and abort_reason is not None:
                raise TelemetryFailure(
                    "control_invalid", "end and abort controls both exist"
                )
            now = monotonic_ns()
            if not isinstance(now, int) or not 0 <= now <= MAX_MONOTONIC_NS:
                raise TelemetryFailure("cadence_drift", "monotonic clock is invalid")
            if last_sample_ns is not None and now < last_sample_ns:
                raise TelemetryFailure("cadence_drift", "monotonic clock moved backward")
            if end_requested:
                writer.write(_terminal_record("end", now, len(samples)))
                return TelemetrySummary(
                    "end",
                    None,
                    len(samples),
                    (sum(samples) / len(samples)) if samples else None,
                    last_sample_ns,
                )
            if abort_reason is not None:
                writer.write(
                    _terminal_record("abort", now, len(samples), abort_reason)
                )
                return TelemetrySummary(
                    "abort",
                    abort_reason,
                    len(samples),
                    (sum(samples) / len(samples)) if samples else None,
                    last_sample_ns,
                )
            if len(samples) >= contract.sample_ceiling:
                raise TelemetryFailure(
                    "sample_ceiling_exceeded",
                    "telemetry reached its TimeLimit-derived sample ceiling",
                )
            scheduled_ns = (
                start_ns
                + len(samples) * contract.cadence_seconds * 1_000_000_000
            )
            if now < scheduled_ns:
                sleep_fn(min(0.25, (scheduled_ns - now) / 1_000_000_000))
                continue
            if (
                now - scheduled_ns
                > contract.cadence_drift_seconds * 1_000_000_000
            ):
                raise TelemetryFailure(
                    "cadence_drift", "telemetry sampling deadline drifted"
                )
            observed_uuid, utilization_raw = scoped.query("uuid,utilization.gpu")
            sampled_ns = monotonic_ns()
            if (
                not isinstance(sampled_ns, int)
                or sampled_ns < scheduled_ns
                or sampled_ns > MAX_MONOTONIC_NS
                or sampled_ns - scheduled_ns
                > contract.cadence_drift_seconds * 1_000_000_000
            ):
                raise TelemetryFailure(
                    "cadence_drift", "nvidia-smi query exceeded cadence drift"
                )
            utilization = int(utilization_raw)
            writer.write(
                {
                    "gpu_uuid": observed_uuid,
                    "kind": "sample",
                    "monotonic_ns": sampled_ns,
                    "scheduled_monotonic_ns": scheduled_ns,
                    "sequence": len(samples),
                    "utilization_percent": utilization,
                }
            )
            samples.append(utilization)
            last_sample_ns = sampled_ns
    except TelemetryFailure as error:
        if (
            writer is not None
            and start_ns is not None
            and not writer.terminal_written
        ):
            try:
                now = monotonic_ns()
                if not isinstance(now, int) or not 0 <= now <= MAX_MONOTONIC_NS:
                    now = start_ns
                now = max(now, start_ns, last_sample_ns or start_ns)
                writer.write(
                    _terminal_record("abort", now, len(samples), error.reason)
                )
            except (OSError, TelemetryFailure):
                pass
        raise
    finally:
        if writer is not None:
            writer.close()
        scoped.close()


def _json_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise ValueError("telemetry record contains duplicate keys")
        value[key] = item
    return value


def parse_telemetry_bytes(
    raw: bytes,
    *,
    contract: TelemetryContract,
    expected_gpu_uuid: str,
    require_terminal: bool,
) -> TelemetrySummary:
    expected_gpu_uuid = _validate_gpu_token(expected_gpu_uuid, uuid=True)
    if len(raw) > contract.byte_ceiling:
        raise ValueError("telemetry exceeds its TimeLimit-derived byte ceiling")
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("telemetry is empty or not newline terminated")
    records: list[Mapping[str, Any]] = []
    for line in raw.splitlines(keepends=True):
        try:
            value = json.loads(
                line,
                object_pairs_hook=_json_pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"invalid number {token}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("telemetry JSONL is invalid") from error
        if not isinstance(value, Mapping) or canonical_line(value) != line:
            raise ValueError("telemetry JSONL is not canonical")
        records.append(value)
    start_keys = {
        "cadence_drift_seconds",
        "cadence_seconds",
        "expected_gpu_uuid",
        "format",
        "kind",
        "monotonic_ns",
        "sample_ceiling",
        "schema_version",
        "time_limit",
        "time_limit_seconds",
    }
    start = records[0]
    if set(start) != start_keys or any(
        (
            start["kind"] != "start",
            start["schema_version"] != SCHEMA_VERSION,
            start["format"] != FORMAT,
            start["cadence_seconds"] != CADENCE_SECONDS,
            start["cadence_drift_seconds"] != CADENCE_DRIFT_SECONDS,
            start["time_limit"] != contract.time_limit,
            start["time_limit_seconds"] != contract.time_limit_seconds,
            start["sample_ceiling"] != contract.sample_ceiling,
            start["expected_gpu_uuid"] != expected_gpu_uuid,
            isinstance(start["monotonic_ns"], bool),
            not isinstance(start["monotonic_ns"], int),
            not 0 <= start["monotonic_ns"] <= MAX_MONOTONIC_NS,
        )
    ):
        raise ValueError("telemetry start record differs from its contract")
    start_ns = int(start["monotonic_ns"])
    samples: list[int] = []
    last_sample_ns: int | None = None
    terminal_kind: str | None = None
    abort_reason: str | None = None
    for index, record in enumerate(records[1:]):
        kind = record.get("kind")
        if terminal_kind is not None:
            raise ValueError("telemetry contains records after its terminal record")
        if kind == "sample":
            if set(record) != {
                "gpu_uuid",
                "kind",
                "monotonic_ns",
                "scheduled_monotonic_ns",
                "sequence",
                "utilization_percent",
            }:
                raise ValueError("telemetry sample schema mismatch")
            sequence = record["sequence"]
            utilization = record["utilization_percent"]
            scheduled_ns = record["scheduled_monotonic_ns"]
            sampled_ns = record["monotonic_ns"]
            expected_scheduled = (
                start_ns + len(samples) * CADENCE_SECONDS * 1_000_000_000
            )
            if (
                record["gpu_uuid"] != expected_gpu_uuid
                or isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence != len(samples)
                or isinstance(utilization, bool)
                or not isinstance(utilization, int)
                or not 0 <= utilization <= 100
                or isinstance(scheduled_ns, bool)
                or not isinstance(scheduled_ns, int)
                or scheduled_ns != expected_scheduled
                or isinstance(sampled_ns, bool)
                or not isinstance(sampled_ns, int)
                or not scheduled_ns <= sampled_ns <= MAX_MONOTONIC_NS
                or sampled_ns - scheduled_ns
                > CADENCE_DRIFT_SECONDS * 1_000_000_000
                or (last_sample_ns is not None and sampled_ns <= last_sample_ns)
            ):
                raise ValueError("telemetry sample violates cadence or GPU contract")
            samples.append(utilization)
            last_sample_ns = sampled_ns
            if len(samples) > contract.sample_ceiling:
                raise ValueError("telemetry sample count exceeds its ceiling")
            continue
        if kind not in {"end", "abort"}:
            raise ValueError("telemetry record kind is invalid")
        keys = {"kind", "monotonic_ns", "sample_count"}
        if kind == "abort":
            keys.add("reason")
        if set(record) != keys:
            raise ValueError("telemetry terminal schema mismatch")
        terminal_ns = record["monotonic_ns"]
        if (
            isinstance(terminal_ns, bool)
            or not isinstance(terminal_ns, int)
            or not start_ns <= terminal_ns <= MAX_MONOTONIC_NS
            or (last_sample_ns is not None and terminal_ns < last_sample_ns)
            or isinstance(record["sample_count"], bool)
            or record["sample_count"] != len(samples)
        ):
            raise ValueError("telemetry terminal record is invalid")
        terminal_kind = kind
        if kind == "abort":
            if record["reason"] not in ABORT_REASONS:
                raise ValueError("telemetry abort reason is invalid")
            abort_reason = record["reason"]
        if index != len(records) - 2:
            raise ValueError("telemetry terminal record is not final")
    if require_terminal and terminal_kind is None:
        raise ValueError("telemetry terminal record is missing")
    return TelemetrySummary(
        terminal_kind=terminal_kind,
        abort_reason=abort_reason,
        sample_count=len(samples),
        utilization_mean_percent=(sum(samples) / len(samples)) if samples else None,
        last_sample_monotonic_ns=last_sample_ns,
    )


def read_telemetry_file(
    path: Path,
    *,
    contract: TelemetryContract,
    expected_gpu_uuid: str,
    require_terminal: bool,
) -> TelemetryEvidence:
    """Stably read, hash, and parse one bounded non-symlink telemetry file."""

    if not path.is_absolute():
        raise ValueError("telemetry path must be absolute")
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError("telemetry file is unavailable") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or path.is_symlink()
        or before.st_size > contract.byte_ceiling
    ):
        raise ValueError("telemetry file is not a bounded physical file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise ValueError("telemetry file could not be opened without links") from error
    try:
        opened = os.fstat(fd)
        if _stat_signature(opened) != _stat_signature(before):
            raise ValueError("telemetry file changed while it was opened")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, contract.byte_ceiling - total + 1))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > contract.byte_ceiling:
                raise ValueError("telemetry file exceeded its byte ceiling")
        after = os.fstat(fd)
        try:
            path_after = path.lstat()
        except OSError as error:
            raise ValueError("telemetry path disappeared during verification") from error
        if (
            total != after.st_size
            or _stat_signature(after) != _stat_signature(opened)
            or _stat_signature(path_after) != _stat_signature(after)
        ):
            raise ValueError("telemetry file changed during verification")
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    summary = parse_telemetry_bytes(
        raw,
        contract=contract,
        expected_gpu_uuid=expected_gpu_uuid,
        require_terminal=require_terminal,
    )
    return TelemetryEvidence(
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        summary=summary,
    )


def _positive_decimal(value: str, where: str) -> int:
    if re.fullmatch(r"[0-9]+", value) is None:
        raise argparse.ArgumentTypeError(f"{where} must be decimal")
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{where} must be positive")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    contract_parser = subparsers.add_parser("contract")
    contract_parser.add_argument("--time-limit", required=True)
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--output", required=True, type=Path)
    record_parser.add_argument("--ready", required=True, type=Path)
    record_parser.add_argument("--end", required=True, type=Path)
    record_parser.add_argument("--abort", required=True, type=Path)
    record_parser.add_argument("--nvidia-smi", required=True, type=Path)
    record_parser.add_argument("--expected-nvidia-smi-sha256", required=True)
    record_parser.add_argument("--visible-gpu-token", required=True)
    record_parser.add_argument("--expected-gpu-uuid", required=True)
    record_parser.add_argument("--time-limit", required=True)
    record_parser.add_argument(
        "--expected-sample-ceiling",
        required=True,
        type=lambda value: _positive_decimal(value, "sample ceiling"),
    )
    record_parser.add_argument(
        "--expected-byte-ceiling",
        required=True,
        type=lambda value: _positive_decimal(value, "byte ceiling"),
    )
    args = parser.parse_args(argv)
    if args.command == "contract":
        print(
            json.dumps(
                asdict(telemetry_contract(args.time_limit)),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    contract = telemetry_contract(args.time_limit)
    summary = record_telemetry(
        output_path=args.output,
        ready_path=args.ready,
        end_path=args.end,
        abort_path=args.abort,
        nvidia_smi=args.nvidia_smi,
        expected_nvidia_smi_sha256=args.expected_nvidia_smi_sha256,
        visible_gpu_token=args.visible_gpu_token,
        expected_gpu_uuid=args.expected_gpu_uuid,
        contract=contract,
        expected_sample_ceiling=args.expected_sample_ceiling,
        expected_byte_ceiling=args.expected_byte_ceiling,
    )
    return 0 if summary.terminal_kind == "end" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TelemetryFailure) as error:
        print(f"formal production GPU telemetry failed: {error}", file=sys.stderr)
        raise SystemExit(1)
