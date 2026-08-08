#!/usr/bin/env python3
"""Re-exec an H100 wrapper while retaining one verified Git executable inode."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import stat
import sys


MAX_EXECUTABLE_BYTES = 128 * 1024 * 1024
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_verified(path: Path, expected_sha256: str) -> int:
    if (
        not path.is_absolute()
        or os.fspath(path) != os.path.abspath(os.fspath(path))
        or HEX64.fullmatch(expected_sha256) is None
    ):
        raise ValueError("Git path or digest is invalid")
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
    try:
        opened = os.fstat(fd)
        if (
            _signature(opened) != _signature(before)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_mode & 0o111 == 0
            or opened.st_size > MAX_EXECUTABLE_BYTES
        ):
            raise ValueError("Git is not a stable bounded executable")
        digest = hashlib.sha256()
        remaining = opened.st_size
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                raise ValueError("Git was truncated while hashing")
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        if _signature(after) != _signature(opened) or digest.hexdigest() != expected_sha256:
            raise ValueError("Git stable SHA-256 verification failed")
        os.lseek(fd, 0, os.SEEK_SET)
        os.set_inheritable(fd, True)
        return fd
    except BaseException:
        os.close(fd)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--git", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command.pop(0)
    if not command or not os.path.isabs(command[0]):
        raise ValueError("digest-bound Git wrapper command must be absolute")
    fd = _open_verified(args.git, args.expected_sha256)
    environment = dict(os.environ)
    environment["FORMAL_GIT_FD"] = str(fd)
    environment["FORMAL_GIT_FD_OWNER_PID"] = str(os.getpid())
    environment["FORMAL_GIT_COMMAND"] = f"/proc/self/fd/{fd}"
    os.execve(command[0], command, environment)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"digest-bound Git launch failed: {error}", file=sys.stderr)
        raise SystemExit(2)
