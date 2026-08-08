#!/usr/bin/env python3
"""Enforce the shared, unfiltered formal GPU active-window contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import selectors
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

QUERY_TIMEOUT_SECONDS = 15
QUERY_MAX_BYTES = 512
STREAM_POLL_SECONDS = 0.05


def _monitor_module():
    path = Path(__file__).with_name("monitor_formal_identity.py")
    spec = importlib.util.spec_from_file_location("_formal_monitor_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load formal monitor contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summarize(
    path: Path,
    *,
    utilization_required: bool,
    expected_gpu_uuids: tuple[str, ...],
    expected_gpu_count: int,
) -> dict[str, object]:
    if not utilization_required:
        return {
            "schema_version": 1,
            "utilization_required": False,
            "utilization_claimed": False,
        }
    if path.is_symlink() or not path.is_file():
        raise ValueError("GPU sample path must be a regular non-symlink file")
    monitor = _monitor_module()
    records, invalid = monitor.parse_gpu_records(path.read_text(encoding="utf-8"))
    if invalid:
        raise ValueError(f"GPU sample file has {invalid} invalid rows")
    result = monitor.evaluate_gpu_window(
        records,
        utilization_required=True,
        expected_gpu_uuids=expected_gpu_uuids,
        expected_gpu_count=expected_gpu_count,
    )
    if not result.complete or result.issues:
        codes = sorted(issue.code for issue in result.issues)
        raise ValueError(f"GPU utilization window failed: {codes}")
    return {
        "schema_version": 1,
        "utilization_required": True,
        "utilization_claimed": True,
        "sample_counts": result.sample_counts,
        "means": result.means,
    }


def _encoded_record(value: dict[str, object]) -> tuple[str, str]:
    kind = value["kind"]
    timestamp = value["monotonic_seconds"]
    if kind == "sample":
        csv_record = (
            f"sample,{timestamp:.9f},{value['gpu_uuid']},"
            f"{value['utilization_percent']}\n"
        )
    else:
        csv_record = f"{kind},{timestamp:.9f}\n"
    jsonl_record = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    return csv_record, jsonl_record


def _write_record(
    csv_handle,
    jsonl_handle,
    value: dict[str, object],
    *,
    max_bytes: int,
) -> None:
    csv_record, jsonl_record = _encoded_record(value)
    projected = (
        csv_handle.tell()
        + jsonl_handle.tell()
        + len(csv_record.encode("utf-8"))
        + len(jsonl_record.encode("utf-8"))
    )
    if projected > max_bytes:
        raise ValueError("GPU monitor artifacts exceed configured ceiling")
    csv_handle.write(csv_record)
    jsonl_handle.write(jsonl_record)
    csv_handle.flush()
    jsonl_handle.flush()


class _GpuSampleStream(Protocol):
    def read_sample(self, timeout_seconds: float) -> tuple[str, ...] | None: ...

    def close(self) -> None: ...


class _PersistentNvidiaSmiSampler:
    """Read allocation-scoped samples from long-lived nvidia-smi loops."""

    def __init__(
        self,
        *,
        executable: str,
        tokens: tuple[str, ...],
        nvidia_fd: int | None,
        interval_seconds: float,
        popen_fn: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        wall_monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if interval_seconds != 1.0 or not tokens:
            raise ValueError("persistent nvidia-smi requires exact 1s cadence")
        self._selector = selectors.DefaultSelector()
        self._processes: list[subprocess.Popen[bytes]] = []
        self._stdout_buffers = [bytearray() for _token in tokens]
        self._pending_rows: list[bytes | None] = [None for _token in tokens]
        self._wall_monotonic_fn = wall_monotonic_fn
        self._last_complete = wall_monotonic_fn()
        self._closed = False
        try:
            for index, token in enumerate(tokens):
                arguments = [
                    executable,
                    f"--id={token}",
                    "--query-gpu=uuid,utilization.gpu",
                    "--format=csv,noheader,nounits",
                    "--loop-ms=1000",
                ]
                query_kwargs = {
                    "stdin": subprocess.DEVNULL,
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.PIPE,
                    "text": False,
                    "bufsize": 0,
                }
                if nvidia_fd is not None:
                    query_kwargs["pass_fds"] = (nvidia_fd,)
                process = popen_fn(arguments, **query_kwargs)
                self._processes.append(process)
                if process.stdout is None or process.stderr is None:
                    raise RuntimeError("persistent nvidia-smi pipes are unavailable")
                for channel, stream in (
                    ("stdout", process.stdout),
                    ("stderr", process.stderr),
                ):
                    os.set_blocking(stream.fileno(), False)
                    self._selector.register(
                        stream, selectors.EVENT_READ, (index, channel)
                    )
        except BaseException:
            try:
                self.close()
            except OSError:
                pass
            raise

    def _check_processes(self) -> None:
        if any(process.poll() is not None for process in self._processes):
            raise RuntimeError("persistent nvidia-smi exited before monitor shutdown")

    def _check_stalled(self) -> None:
        if self._wall_monotonic_fn() - self._last_complete >= QUERY_TIMEOUT_SECONDS:
            raise RuntimeError("persistent nvidia-smi sample timed out")

    def _consume(self, timeout_seconds: float) -> bool:
        events = self._selector.select(timeout_seconds)
        if not events:
            self._check_processes()
            self._check_stalled()
            return False
        for key, _mask in events:
            index, channel = key.data
            try:
                chunk = os.read(key.fd, QUERY_MAX_BYTES + 1)
            except BlockingIOError:
                continue
            if not chunk:
                raise RuntimeError("persistent nvidia-smi stream closed unexpectedly")
            if channel == "stderr":
                raise RuntimeError("persistent nvidia-smi wrote to stderr")
            buffer = self._stdout_buffers[index]
            buffer.extend(chunk)
            if len(buffer) > QUERY_MAX_BYTES:
                raise RuntimeError("persistent nvidia-smi row exceeds its byte ceiling")
            newline = buffer.find(b"\n")
            if newline < 0:
                continue
            if self._pending_rows[index] is not None:
                raise RuntimeError("persistent nvidia-smi emitted duplicate GPU rows")
            self._pending_rows[index] = bytes(buffer[:newline])
            del buffer[: newline + 1]
            if b"\n" in buffer:
                raise RuntimeError("persistent nvidia-smi emitted extra GPU rows")
        self._check_processes()
        return all(row is not None for row in self._pending_rows)

    def read_sample(self, timeout_seconds: float) -> tuple[str, ...] | None:
        if self._closed or timeout_seconds < 0 or not math.isfinite(timeout_seconds):
            raise RuntimeError("persistent nvidia-smi read is invalid")
        deadline = self._wall_monotonic_fn() + timeout_seconds
        while True:
            if all(row is not None for row in self._pending_rows):
                raw_rows = tuple(row for row in self._pending_rows if row is not None)
                self._pending_rows = [None for _row in self._pending_rows]
                try:
                    rows = tuple(row.decode("ascii") for row in raw_rows)
                except UnicodeDecodeError as error:
                    raise RuntimeError(
                        "persistent nvidia-smi row is not ASCII"
                    ) from error
                self._last_complete = self._wall_monotonic_fn()
                return rows
            remaining = max(0.0, deadline - self._wall_monotonic_fn())
            complete = self._consume(remaining)
            if complete:
                continue
            if self._wall_monotonic_fn() >= deadline:
                return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for process in self._processes:
            if process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
        for process in self._processes:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        self._selector.close()
        for process in self._processes:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()


def _parse_sample_rows(
    rows: tuple[str, ...], expected_gpu_uuids: tuple[str, ...]
) -> tuple[tuple[str, float], ...]:
    if len(rows) != len(expected_gpu_uuids):
        raise RuntimeError("nvidia-smi sample GPU count changed")
    parsed: list[tuple[str, float]] = []
    for row, expected_uuid in zip(rows, expected_gpu_uuids):
        if not row or row != row.strip():
            raise ValueError("invalid nvidia-smi sample")
        fields = [item.strip() for item in row.split(",")]
        if len(fields) != 2 or not fields[0]:
            raise ValueError("invalid nvidia-smi sample")
        try:
            utilization = float(fields[1])
        except ValueError as error:
            raise ValueError("invalid nvidia-smi sample") from error
        if not math.isfinite(utilization) or not 0.0 <= utilization <= 100.0:
            raise ValueError("invalid nvidia-smi sample")
        if fields[0] != expected_uuid:
            raise RuntimeError("visible GPU UUID set changed inside active window")
        parsed.append((fields[0], utilization))
    return tuple(parsed)


def record_gpu_window(
    csv_path: Path,
    jsonl_path: Path,
    *,
    ready_path: Path,
    stop_path: Path,
    active_start_path: Path,
    active_end_path: Path,
    max_bytes: int,
    expected_gpu_count: int,
    interval_seconds: float = 1.0,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sampler_factory: Callable[..., _GpuSampleStream] = _PersistentNvidiaSmiSampler,
) -> None:
    """Record every sample, and only samples, in the harness-signaled window."""
    if max_bytes < 1 or interval_seconds != 1.0 or expected_gpu_count not in {1, 2}:
        raise ValueError("formal GPU recorder requires a positive ceiling and 1s cadence")
    executable = os.environ.get("NVIDIA_SMI")
    raw_fd = os.environ.get("FORMAL_NVIDIA_SMI_FD")
    if (
        not executable
        or not os.path.isabs(executable)
        or not os.path.isfile(executable)
        or not os.access(executable, os.X_OK)
    ):
        raise ValueError("NVIDIA_SMI must be an absolute executable")
    nvidia_fd = None
    if raw_fd is not None:
        if not raw_fd.isdecimal() or executable != f"/proc/self/fd/{raw_fd}":
            raise ValueError("NVIDIA_SMI descriptor binding is invalid")
        nvidia_fd = int(raw_fd)
    raw_tokens = os.environ.get("FORMAL_VISIBLE_GPU_TOKENS", "")
    tokens = tuple(raw_tokens.split(",")) if raw_tokens else ()
    token_pattern = re.compile(r"^(?:[0-9]+|GPU-[A-Za-z0-9._-]+|MIG-[A-Za-z0-9._-]+)$")
    if (
        len(tokens) != expected_gpu_count
        or len(set(tokens)) != expected_gpu_count
        or any(token_pattern.fullmatch(token) is None for token in tokens)
    ):
        raise ValueError("formal GPU recorder requires exact CUDA-visible GPU tokens")
    raw_uuids = os.environ.get("FORMAL_VISIBLE_GPU_UUIDS", "")
    expected_gpu_uuids = tuple(raw_uuids.split(",")) if raw_uuids else ()
    uuid_pattern = re.compile(r"^(?:GPU|MIG)-[A-Za-z0-9._:/-]+$")
    if (
        len(expected_gpu_uuids) != expected_gpu_count
        or len(set(expected_gpu_uuids)) != expected_gpu_count
        or any(uuid_pattern.fullmatch(value) is None for value in expected_gpu_uuids)
    ):
        raise ValueError("formal GPU recorder requires exact visible GPU UUIDs")
    if any(
        path.exists() or path.is_symlink()
        for path in (
            csv_path,
            jsonl_path,
            ready_path,
            active_start_path,
            active_end_path,
        )
    ):
        raise ValueError("formal GPU recorder outputs must be fresh")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    sampler: _GpuSampleStream | None = None
    try:
        with csv_path.open("x", encoding="utf-8") as csv_handle, jsonl_path.open(
            "x", encoding="utf-8"
        ) as jsonl_handle:
            sampler = sampler_factory(
                executable=executable,
                tokens=tokens,
                nvidia_fd=nvidia_fd,
                interval_seconds=interval_seconds,
            )
            initial_rows = sampler.read_sample(QUERY_TIMEOUT_SECONDS)
            if initial_rows is None:
                raise RuntimeError("persistent nvidia-smi sample timed out")
            _parse_sample_rows(initial_rows, expected_gpu_uuids)
            ready_path.touch(exist_ok=False)
            while not active_start_path.exists():
                if stop_path.exists():
                    for handle in (csv_handle, jsonl_handle):
                        os.fsync(handle.fileno())
                    return
                discarded = sampler.read_sample(STREAM_POLL_SECONDS)
                if discarded is not None:
                    _parse_sample_rows(discarded, expected_gpu_uuids)
            start = _read_signal_timestamp(active_start_path)
            while True:
                discarded = sampler.read_sample(0.0)
                if discarded is None:
                    break
                _parse_sample_rows(discarded, expected_gpu_uuids)
            _write_record(
                csv_handle,
                jsonl_handle,
                {
                    "kind": "training_start",
                    "monotonic_seconds": start,
                    "expected_gpu_uuids": list(expected_gpu_uuids),
                    "expected_gpu_count": expected_gpu_count,
                },
                max_bytes=max_bytes,
            )
            while not active_end_path.exists():
                if stop_path.exists():
                    raise RuntimeError("compute ended before active training window closed")
                rows = sampler.read_sample(STREAM_POLL_SECONDS)
                if rows is None:
                    continue
                timestamp = monotonic_fn()
                if active_end_path.exists():
                    observed_end = _read_signal_timestamp(active_end_path)
                    if timestamp > observed_end:
                        break
                for gpu_uuid, utilization in _parse_sample_rows(
                    rows, expected_gpu_uuids
                ):
                    _write_record(
                        csv_handle,
                        jsonl_handle,
                        {
                            "kind": "sample",
                            "monotonic_seconds": timestamp,
                            "gpu_uuid": gpu_uuid,
                            "utilization_percent": utilization,
                        },
                        max_bytes=max_bytes,
                    )
            end = _read_signal_timestamp(active_end_path)
            if end <= start:
                raise ValueError("active training signal timestamps are invalid")
            _write_record(
                csv_handle,
                jsonl_handle,
                {
                    "kind": "training_end",
                    "monotonic_seconds": end,
                    "expected_gpu_uuids": list(expected_gpu_uuids),
                    "expected_gpu_count": expected_gpu_count,
                },
                max_bytes=max_bytes,
            )
            for handle in (csv_handle, jsonl_handle):
                os.fsync(handle.fileno())
    finally:
        if sampler is not None:
            sampler.close()


def _read_signal_timestamp(path: Path) -> float:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128:
        raise ValueError("active training signal is invalid")
    raw = path.read_text(encoding="ascii")
    try:
        value = float(raw.strip())
    except ValueError as error:
        raise ValueError("active training signal timestamp is invalid") from error
    if not math.isfinite(value):
        raise ValueError("active training signal timestamp is non-finite")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", type=Path)
    parser.add_argument("--utilization-required", action="store_true")
    parser.add_argument("--expected-gpu-uuid", action="append", default=[])
    parser.add_argument("--expected-gpus", type=int, default=0)
    parser.add_argument("--record-jsonl", type=Path)
    parser.add_argument("--ready-path", type=Path)
    parser.add_argument("--stop-path", type=Path)
    parser.add_argument("--active-start-path", type=Path)
    parser.add_argument("--active-end-path", type=Path)
    parser.add_argument("--max-bytes", type=int)
    parser.add_argument("--record-expected-gpus", type=int)
    args = parser.parse_args(argv)
    if args.record_jsonl is not None:
        if (
            args.csv_path is None
            or args.ready_path is None
            or args.stop_path is None
            or args.active_start_path is None
            or args.active_end_path is None
            or args.max_bytes is None
            or args.record_expected_gpus is None
        ):
            raise ValueError("record mode requires CSV, JSONL, ready, stop, and max bytes")
        record_gpu_window(
            args.csv_path,
            args.record_jsonl,
            ready_path=args.ready_path,
            stop_path=args.stop_path,
            active_start_path=args.active_start_path,
            active_end_path=args.active_end_path,
            max_bytes=args.max_bytes,
            expected_gpu_count=args.record_expected_gpus,
        )
        return 0
    if args.csv_path is None:
        raise ValueError("summary mode requires a CSV path")
    report = summarize(
        args.csv_path,
        utilization_required=args.utilization_required,
        expected_gpu_uuids=tuple(args.expected_gpu_uuid),
        expected_gpu_count=args.expected_gpus,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"formal GPU utilization gate failed: {error}", file=sys.stderr)
        raise SystemExit(1)
