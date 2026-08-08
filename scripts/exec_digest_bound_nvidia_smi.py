#!/usr/bin/env python3
"""Re-exec a compute wrapper with one verified nvidia-smi inode held open."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


MAX_EXECUTABLE_BYTES = 128 * 1024 * 1024
QUERY_MAX_BYTES = 16 * 1024
QUERY_TIMEOUT_SECONDS = 15
HEX64 = re.compile(r"^[0-9a-f]{64}$")
GPU_TOKEN = re.compile(
    r"^(?:[0-9]+|GPU-[A-Za-z0-9._:-]+|MIG-[A-Za-z0-9._:/-]+)$"
)
QUERY_FIELDS = frozenset(
    {
        "name,uuid,driver_version",
        "uuid,name,driver_version",
        "uuid",
        "uuid,utilization.gpu",
    }
)


def _signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _verify_open_fd(
    fd: int,
    expected_sha256: str,
    *,
    expected_signature: tuple[int, ...] | None = None,
) -> int:
    try:
        opened = os.fstat(fd)
        if (
            (expected_signature is not None and _signature(opened) != expected_signature)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_mode & 0o111 == 0
            or opened.st_size > MAX_EXECUTABLE_BYTES
        ):
            raise ValueError("nvidia-smi is not a stable bounded executable")
        digest = hashlib.sha256()
        remaining = opened.st_size
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                raise ValueError("nvidia-smi was truncated while hashing")
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        if _signature(after) != _signature(opened) or digest.hexdigest() != expected_sha256:
            raise ValueError("nvidia-smi stable SHA-256 verification failed")
        os.lseek(fd, 0, os.SEEK_SET)
        os.set_inheritable(fd, True)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _open_verified(path: Path, expected_sha256: str) -> int:
    if (
        not path.is_absolute()
        or os.fspath(path) != os.path.abspath(os.fspath(path))
        or HEX64.fullmatch(expected_sha256) is None
    ):
        raise ValueError("nvidia-smi path or digest is invalid")
    directory_flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    directory_fd = os.open(os.path.sep, directory_flags)
    try:
        for component in path.parts[1:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        before = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        flags = (
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        fd = os.open(path.name, flags, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    return _verify_open_fd(fd, expected_sha256, expected_signature=_signature(before))


def _open_retained(owner_pid: int, retained_fd: int, expected_sha256: str) -> int:
    if (
        isinstance(owner_pid, bool)
        or owner_pid < 1
        or isinstance(retained_fd, bool)
        or retained_fd < 0
        or HEX64.fullmatch(expected_sha256) is None
    ):
        raise ValueError("retained nvidia-smi descriptor source is invalid")
    descriptor_path = Path(f"/proc/{owner_pid}/fd/{retained_fd}")
    before = os.stat(descriptor_path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(descriptor_path, flags)
    return _verify_open_fd(fd, expected_sha256, expected_signature=_signature(before))


def _bounded_query(fd: int, token: str, fields: str) -> int:
    if GPU_TOKEN.fullmatch(token) is None or fields not in QUERY_FIELDS:
        raise ValueError("allocation-scoped nvidia-smi query is invalid")
    try:
        completed = subprocess.run(
            [
                f"/proc/self/fd/{fd}",
                f"--id={token}",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=QUERY_TIMEOUT_SECONDS,
            pass_fds=(fd,),
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError("allocation-scoped nvidia-smi query timed out") from error
    if (
        completed.returncode != 0
        or completed.stderr
        or len(completed.stdout) > QUERY_MAX_BYTES
    ):
        raise ValueError("allocation-scoped nvidia-smi query failed")
    sys.stdout.buffer.write(completed.stdout)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--nvidia-smi", type=Path)
    source.add_argument("--retained-fd-owner-pid", type=int)
    parser.add_argument("--retained-fd", type=int)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--query-token")
    parser.add_argument("--query-fields")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command.pop(0)
    query_mode = args.query_token is not None or args.query_fields is not None
    if query_mode:
        if command or args.query_token is None or args.query_fields is None:
            raise ValueError("digest-bound query arguments are incomplete")
    elif not command or not os.path.isabs(command[0]):
        raise ValueError("digest-bound wrapper command must be absolute")
    if args.nvidia_smi is not None:
        if args.retained_fd is not None:
            raise ValueError("retained descriptor requires its owner PID")
        fd = _open_verified(args.nvidia_smi, args.expected_sha256)
    else:
        if args.retained_fd is None:
            raise ValueError("retained descriptor number is required")
        fd = _open_retained(
            args.retained_fd_owner_pid,
            args.retained_fd,
            args.expected_sha256,
        )
    if query_mode:
        try:
            return _bounded_query(fd, args.query_token, args.query_fields)
        finally:
            os.close(fd)
    environment = dict(os.environ)
    environment["FORMAL_NVIDIA_SMI_FD"] = str(fd)
    environment["FORMAL_NVIDIA_SMI_FD_OWNER_PID"] = str(os.getpid())
    environment["NVIDIA_SMI"] = f"/proc/self/fd/{fd}"
    os.execve(command[0], command, environment)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"digest-bound nvidia-smi launch failed: {error}", file=sys.stderr)
        raise SystemExit(2)
