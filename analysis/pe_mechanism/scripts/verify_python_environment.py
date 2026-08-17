#!/usr/bin/env python3
"""Fail closed unless this process matches a frozen venv entry contract."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path, PurePosixPath
import select
import stat
import subprocess
import sys
import time


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from pe_mechanism.python_environment import (  # noqa: E402
    verify_python_environment_contract_file,
)


GPU_SAMPLE_INTERVAL_SECONDS = 30
GPU_QUERY_TIMEOUT_SECONDS = 15
NVIDIA_SMI = "/usr/bin/nvidia-smi"


def _open_exclusive_nofollow(path_text: str) -> int:
    path = PurePosixPath(path_text)
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise ValueError("monitor CSV must be a normalized absolute file path")
    parts = path.parts[1:]
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("monitor CSV must not contain empty, dot, or parent components")

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    directory_fd = os.open("/", directory_flags)
    try:
        for component in parts[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(
            parts[-1],
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
    finally:
        os.close(directory_fd)


def _write_all(file_descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(file_descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write to GPU monitor CSV")
        offset += written


def _sample_gpus(csv_fd: int) -> None:
    completed = subprocess.run(
        [
            NVIDIA_SMI,
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=GPU_QUERY_TIMEOUT_SECONDS,
    )
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    rows = "".join(
        f"{timestamp},{line}\n"
        for line in completed.stdout.splitlines()
        if line.strip()
    )
    if not rows:
        raise RuntimeError("nvidia-smi returned no visible GPU rows")
    _write_all(csv_fd, rows.encode("utf-8"))


def _run_gpu_monitor(output_csv: Path) -> int:
    nvidia_smi_stat = os.lstat(NVIDIA_SMI)
    if stat.S_ISLNK(nvidia_smi_stat.st_mode) or not stat.S_ISREG(
        nvidia_smi_stat.st_mode
    ):
        raise RuntimeError("nvidia-smi must be a real regular file")
    if not os.access(NVIDIA_SMI, os.X_OK):
        raise RuntimeError("nvidia-smi must be executable")
    csv_fd = _open_exclusive_nofollow(str(output_csv))
    try:
        _write_all(
            csv_fd,
            b"timestamp,index,name,utilization_gpu_pct,memory_used_mib,memory_total_mib\n",
        )
        _sample_gpus(csv_fd)
        print("READY", flush=True)
        next_sample = time.monotonic() + GPU_SAMPLE_INTERVAL_SECONDS
        while True:
            timeout = max(0.0, next_sample - time.monotonic())
            readable, _, _ = select.select((sys.stdin,), (), (), timeout)
            if readable:
                control = sys.stdin.readline()
                if control == "":
                    raise RuntimeError("GPU monitor stdin closed before STOP")
                if control != "STOP\n":
                    raise RuntimeError("GPU monitor accepts only the exact STOP line")
                _sample_gpus(csv_fd)
                print("COMPLETE", flush=True)
                return 0
            _sample_gpus(csv_fd)
            next_sample = time.monotonic() + GPU_SAMPLE_INTERVAL_SECONDS
    finally:
        os.close(csv_fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--expected-file-sha256", required=True)
    parser.add_argument("--expected-document-sha256", required=True)
    parser.add_argument(
        "--gpu-monitor",
        type=Path,
        help="run the in-process READY/STOP/COMPLETE GPU monitor into this CSV",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="optional Python arguments to run with the currently bound process image",
    )
    args = parser.parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if args.gpu_monitor is not None and command:
        parser.error("--gpu-monitor cannot be combined with a Python command")
    verification = {
        "entry": args.entry,
        "contract_path": args.contract,
        "expected_file_sha256": args.expected_file_sha256,
        "expected_document_sha256": args.expected_document_sha256,
    }
    verify_python_environment_contract_file(**verification)
    if args.gpu_monitor is not None:
        return _run_gpu_monitor(args.gpu_monitor)
    if not command:
        return 0
    completed = subprocess.run(
        [str(args.entry), *command],
        executable="/proc/self/exe",
        check=False,
    )
    # This same, still-bound verifier process performs the postcondition check.
    # A failure here overrides a successful workload so Slurm dependencies stay held.
    verify_python_environment_contract_file(**verification)
    if completed.returncode < 0:
        return 128 - completed.returncode
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
