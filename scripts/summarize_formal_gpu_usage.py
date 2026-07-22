#!/usr/bin/env python3
"""Enforce the shared, unfiltered formal GPU active-window contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable


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
    sleep_fn: Callable[[float], None] = time.sleep,
    query_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Record every sample, and only samples, in the harness-signaled window."""
    if max_bytes < 1 or interval_seconds != 1.0 or expected_gpu_count not in {1, 2}:
        raise ValueError("formal GPU recorder requires a positive ceiling and 1s cadence")
    executable = os.environ.get("NVIDIA_SMI")
    if (
        not executable
        or not os.path.isabs(executable)
        or not os.path.isfile(executable)
        or not os.access(executable, os.X_OK)
    ):
        raise ValueError("NVIDIA_SMI must be an absolute executable")
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
    with csv_path.open("x", encoding="utf-8") as csv_handle, jsonl_path.open(
        "x", encoding="utf-8"
    ) as jsonl_handle:
        initial = query_fn(
            [
                executable,
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        expected_gpu_uuids = tuple(
            line.strip() for line in initial.stdout.splitlines() if line.strip()
        )
        if (
            initial.returncode != 0
            or len(expected_gpu_uuids) != expected_gpu_count
            or len(set(expected_gpu_uuids)) != expected_gpu_count
        ):
            raise RuntimeError("visible GPU UUID set/count does not match case contract")
        ready_path.touch(exist_ok=False)
        while not active_start_path.exists():
            if stop_path.exists():
                for handle in (csv_handle, jsonl_handle):
                    os.fsync(handle.fileno())
                return
            sleep_fn(0.05)
        start = _read_signal_timestamp(active_start_path)
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
            loop_start = monotonic_fn()
            result = query_fn(
                [
                    executable,
                    "--query-gpu=uuid,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError("nvidia-smi sample failed")
            timestamp = monotonic_fn()
            observed_gpu_uuids: list[str] = []
            for raw in result.stdout.splitlines():
                fields = [item.strip() for item in raw.split(",")]
                if len(fields) != 2 or not fields[0]:
                    raise ValueError("invalid nvidia-smi sample")
                utilization = float(fields[1])
                observed_gpu_uuids.append(fields[0])
                _write_record(
                    csv_handle,
                    jsonl_handle,
                    {
                        "kind": "sample",
                        "monotonic_seconds": timestamp,
                        "gpu_uuid": fields[0],
                        "utilization_percent": utilization,
                    },
                    max_bytes=max_bytes,
                )
            if tuple(observed_gpu_uuids) != expected_gpu_uuids:
                raise RuntimeError("visible GPU UUID set changed inside active window")
            delay = interval_seconds - (monotonic_fn() - loop_start)
            if delay > 0:
                sleep_fn(delay)
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
