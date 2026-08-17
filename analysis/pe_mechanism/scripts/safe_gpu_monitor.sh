#!/usr/bin/env bash

set -euo pipefail

(( $# == 2 )) || {
  printf 'usage: %s PYTHON ABSOLUTE_OUTPUT_CSV\n' "$0" >&2
  exit 2
}

python_bin=$1
output_csv=$2
[[ "$python_bin" == /* && -x "$python_bin" && ! -L "$python_bin" ]] || {
  printf 'error: monitor Python must be an absolute real executable\n' >&2
  exit 2
}
[[ "$output_csv" == /* ]] || {
  printf 'error: monitor CSV path must be absolute\n' >&2
  exit 2
}

exec "$python_bin" -I -B - "$output_csv" <<'PY'
from __future__ import annotations

from datetime import datetime
import os
from pathlib import PurePosixPath
import subprocess
import sys
import time


def open_exclusive_nofollow(path_text: str) -> int:
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


def write_all(file_descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(file_descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write to GPU monitor CSV")
        offset += written


csv_fd = open_exclusive_nofollow(sys.argv[1])
try:
    write_all(
        csv_fd,
        b"timestamp,index,name,utilization_gpu_pct,memory_used_mib,memory_total_mib\n",
    )
    announced_ready = False
    while True:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        rows = "".join(
            f"{timestamp},{line}\n"
            for line in completed.stdout.splitlines()
            if line.strip()
        )
        if not rows:
            raise RuntimeError("nvidia-smi returned no visible GPU rows")
        write_all(csv_fd, rows.encode("utf-8"))
        if not announced_ready:
            print("READY", flush=True)
            announced_ready = True
        time.sleep(30)
finally:
    os.close(csv_fd)
PY
