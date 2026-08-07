#!/usr/bin/env python3
"""Prove that the exact candidate commit is advertised by the public GitHub ref."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
import sys
import time
from typing import Any, Mapping


REPOSITORY_URL = "https://github.com/kikixiong/tabicl-pe.git"
REPOSITORY_REF = "refs/heads/codex/position-identity-v1"
QUERY_TIMEOUT_SECONDS = 30.0
QUERY_STREAM_CEILING_BYTES = 65_536
EXECUTABLE_CEILING_BYTES = 128 * 1024 * 1024
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
QUERY_CWD = "/"
HERMETIC_GIT_ENV = {
    "PATH": "/usr/bin:/bin",
    "LC_ALL": "C",
    "LANG": "C",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_COUNT": "0",
    "GIT_CEILING_DIRECTORIES": "/",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_PROTOCOL_FROM_USER": "0",
    "GIT_ALLOW_PROTOCOL": "https",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def repository_identity_sha256(commit_sha: str) -> str:
    if not isinstance(commit_sha, str) or HEX40.fullmatch(commit_sha) is None:
        raise ValueError("repository commit must be a lowercase Git object ID")
    return _sha256(
        {
            "schema_version": 1,
            "repository_url": REPOSITORY_URL,
            "repository_ref": REPOSITORY_REF,
            "commit_sha": commit_sha,
        }
    )


def expected_repository_binding(
    *, expected_commit_sha: str, git_sha256: str
) -> dict[str, Any]:
    if not isinstance(expected_commit_sha, str) or HEX40.fullmatch(expected_commit_sha) is None:
        raise ValueError("expected repository commit is malformed")
    if not isinstance(git_sha256, str) or HEX64.fullmatch(git_sha256) is None:
        raise ValueError("Git executable SHA-256 is malformed")
    stdout = f"{expected_commit_sha}\t{REPOSITORY_REF}\n".encode("ascii")
    stderr = b""
    query_body = {
        "schema_version": 1,
        "argv": ["ls-remote", "--refs", REPOSITORY_URL, REPOSITORY_REF],
        "cwd": QUERY_CWD,
        "environment": dict(HERMETIC_GIT_ENV),
        "git_sha256": git_sha256,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
    }
    return {
        "schema_version": 1,
        "repository_url": REPOSITORY_URL,
        "repository_ref": REPOSITORY_REF,
        "commit_sha": expected_commit_sha,
        "repository_identity_sha256": repository_identity_sha256(
            expected_commit_sha
        ),
        "git_sha256": git_sha256,
        "query_sha256": _sha256(query_body),
        "query_stdout_sha256": query_body["stdout_sha256"],
    }


def _absolute_normalized(path: Path, *, where: str) -> Path:
    if (
        not path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or os.fspath(path) != os.path.abspath(os.fspath(path))
    ):
        raise ValueError(f"{where} must be a normalized absolute path")
    return path


def _directory_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("no-follow executable traversal is unavailable")
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_directory_nofollow(path: Path, *, where: str) -> int:
    path = _absolute_normalized(path, where=where)
    flags = _directory_flags()
    fd = os.open(os.path.sep, flags)
    try:
        for component in path.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _trusted_git_fd(path: Path, expected_sha256: str) -> int:
    path = _absolute_normalized(path, where="Git executable")
    if not isinstance(expected_sha256, str) or HEX64.fullmatch(expected_sha256) is None:
        raise ValueError("Git executable SHA-256 is malformed")
    parent_fd = _open_directory_nofollow(path.parent, where="Git executable parent")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path.name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError("Git executable is not a no-follow regular file") from error
    finally:
        os.close(parent_fd)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_mode & 0o111 == 0
            or before.st_size > EXECUTABLE_CEILING_BYTES
        ):
            raise ValueError("Git executable is not a bounded executable file")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                raise ValueError("Git executable was truncated while hashing")
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        signature = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in signature):
            raise ValueError("Git executable changed while being attested")
        if digest.hexdigest() != expected_sha256:
            raise ValueError("Git executable digest mismatch")
        os.lseek(fd, 0, os.SEEK_SET)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _kill_and_wait(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _bounded_query(git_fd: int) -> tuple[bytes, bytes]:
    argv = [
        f"/proc/self/fd/{git_fd}",
        "ls-remote",
        "--refs",
        REPOSITORY_URL,
        REPOSITORY_REF,
    ]
    cwd = Path(QUERY_CWD)
    if (
        cwd.resolve(strict=True) != cwd
        or not cwd.is_dir()
        or (cwd / ".git").exists()
        or (cwd / ".git").is_symlink()
    ):
        raise ValueError("GitHub query cwd must be a physical non-repository directory")
    process = subprocess.Popen(
        argv,
        env=dict(HERMETIC_GIT_ENV),
        cwd=QUERY_CWD,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        pass_fds=(git_fd,),
    )
    assert process.stdout is not None and process.stderr is not None
    stdout_fd = process.stdout.fileno()
    stderr_fd = process.stderr.fileno()
    selector = selectors.DefaultSelector()
    streams = {stdout_fd: bytearray(), stderr_fd: bytearray()}
    selector.register(process.stdout, selectors.EVENT_READ)
    selector.register(process.stderr, selectors.EVENT_READ)
    deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_and_wait(process)
                raise ValueError("GitHub repository query timed out")
            events = selector.select(remaining)
            if not events:
                _kill_and_wait(process)
                raise ValueError("GitHub repository query timed out")
            for key, _mask in events:
                chunk = os.read(key.fd, 16_384)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                target = streams[key.fd]
                target.extend(chunk)
                if len(target) > QUERY_STREAM_CEILING_BYTES:
                    _kill_and_wait(process)
                    raise ValueError("GitHub repository query exceeded its byte ceiling")
        try:
            returncode = process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as error:
            _kill_and_wait(process)
            raise ValueError("GitHub repository query timed out") from error
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
    stdout = bytes(streams[stdout_fd])
    stderr = bytes(streams[stderr_fd])
    if returncode != 0 or stderr:
        raise ValueError("GitHub repository query failed")
    return stdout, stderr


def query_exact_repository(
    *, git: Path, git_sha256: str, expected_commit_sha: str
) -> dict[str, Any]:
    if not isinstance(expected_commit_sha, str) or HEX40.fullmatch(expected_commit_sha) is None:
        raise ValueError("expected repository commit is malformed")
    git_fd = _trusted_git_fd(Path(git), git_sha256)
    try:
        stdout, stderr = _bounded_query(git_fd)
    finally:
        os.close(git_fd)
    expected = expected_repository_binding(
        expected_commit_sha=expected_commit_sha, git_sha256=git_sha256
    )
    expected_stdout = f"{expected_commit_sha}\t{REPOSITORY_REF}\n".encode("ascii")
    if stdout != expected_stdout:
        raise ValueError("GitHub ref does not advertise the exact candidate commit")
    if hashlib.sha256(stdout).hexdigest() != expected["query_stdout_sha256"]:
        raise ValueError("GitHub repository query output digest mismatch")
    return expected


def validate_repository_binding(
    value: Mapping[str, Any], *, expected_commit_sha: str, expected_git_sha256: str
) -> dict[str, Any]:
    keys = {
        "schema_version",
        "repository_url",
        "repository_ref",
        "commit_sha",
        "repository_identity_sha256",
        "git_sha256",
        "query_sha256",
        "query_stdout_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("repository binding schema mismatch")
    expected = expected_repository_binding(
        expected_commit_sha=expected_commit_sha,
        git_sha256=expected_git_sha256,
    )
    if dict(value) != expected:
        raise ValueError("repository binding mismatch")
    return dict(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--git", required=True, type=Path)
    parser.add_argument("--git-sha256", required=True)
    parser.add_argument("--expected-commit-sha", required=True)
    parser.add_argument("--expected-identity-sha256")
    parser.add_argument("--expected-query-sha256")
    args = parser.parse_args(argv)
    result = query_exact_repository(
        git=args.git,
        git_sha256=args.git_sha256,
        expected_commit_sha=args.expected_commit_sha,
    )
    for key, expected in (
        ("repository_identity_sha256", args.expected_identity_sha256),
        ("query_sha256", args.expected_query_sha256),
    ):
        if expected is not None and result[key] != expected:
            raise ValueError(f"repository {key} differs from the frozen export")
    sys.stdout.buffer.write(_canonical(result) + b"\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"repository verification failed: {error}", file=sys.stderr)
        raise SystemExit(2)
