#!/usr/bin/env python3
"""Reject non-finite metrics and OOM/error signatures in durable training logs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import select
import signal
import stat
import subprocess
import sys
import time


ERROR_PATTERNS = (
    ("non-finite value", re.compile(r"(?i)(?<![A-Za-z0-9_])(?:nan|[+-]?inf(?:inity)?|non[- ]finite)(?![A-Za-z0-9_])")),
    ("out of memory", re.compile(r"(?i)(?:out of memory|\bOOM\b)")),
    (
        "storage exhausted",
        re.compile(
            r"(?i)(?:\bENOSPC\b|no space left on device|\[Errno\s+28\])"
        ),
    ),
    ("traceback", re.compile(r"(?m)^Traceback \(most recent call last\):")),
)


def _scan_text(text: str) -> list[str]:
    return [name for name, pattern in ERROR_PATTERNS if pattern.search(text)]


def _read_bounded_regular(path: Path, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot open log safely: {path}") from error
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"log must be a regular file: {path}")
        if info.st_size > max_bytes:
            raise ValueError("durable log exceeds configured ceiling")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > max_bytes:
            raise ValueError("durable log exceeds configured ceiling")
        after = os.fstat(fd)
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("log changed while it was being scanned")
        return raw
    finally:
        os.close(fd)


def scan_log(path: Path, *, max_bytes: int) -> dict[str, object]:
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    raw = _read_bounded_regular(path, max_bytes)
    text = raw.decode("utf-8", errors="replace")
    findings = _scan_text(text)
    if findings:
        raise ValueError("rejected log signatures: " + ", ".join(findings))
    return {"schema_version": 1, "ok": True, "bytes": len(raw)}


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    for name in ("O_DIRECTORY", "O_CLOEXEC", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(
    process: subprocess.Popen[bytes], *, grace_seconds: float = 1.0
) -> None:
    """TERM, then KILL the isolated command group, including descendants."""

    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_seconds
    while _process_group_exists(pgid) and time.monotonic() < deadline:
        try:
            process.wait(timeout=min(0.05, max(0.0, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            pass
        time.sleep(0.01)
    if _process_group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def capture_durable_log(
    output: Path,
    *,
    max_bytes: int,
    command: list[str],
    live_output: Path | None = None,
) -> tuple[dict[str, object], int]:
    """Capture bounded live output, then atomically publish the durable log.

    The incomplete byte stream is exposed through ``live_output`` (by default
    ``OUTPUT.live``).  ``output`` itself remains absent until the command has
    stopped, the stream has been flushed and fsynced, and a no-replace hard-link
    publication succeeds.  The live name and final name therefore refer to the
    exact same inode during publication; no copy or mutable rename window can
    make the durable result differ from what was monitored.
    """
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    if not command:
        raise ValueError("capture requires a command")
    parent = output.parent.resolve(strict=True)
    if output.parent.is_symlink() or output.exists() or output.is_symlink():
        raise ValueError("durable log output must be a fresh non-symlink path")
    if live_output is None:
        live_output = output.with_name(output.name + ".live")
    if live_output.parent.resolve(strict=True) != parent:
        raise ValueError("live log must be a sibling of the durable output")
    if (
        live_output.parent.is_symlink()
        or live_output.exists()
        or live_output.is_symlink()
    ):
        raise ValueError("live log output must be a fresh non-symlink path")

    live_created = False
    process: subprocess.Popen[bytes] | None = None
    process_group_cleaned = False
    overflow = False
    try:
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        fd = os.open(live_output, flags, 0o600)
        live_created = True
        with os.fdopen(fd, "w+b", buffering=0) as handle:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            assert process.stdout is not None
            total = 0
            while True:
                pipe_fd = process.stdout.fileno()
                readable, _, _ = select.select([pipe_fd], [], [], 0.1)
                if not readable:
                    # A descendant can inherit stdout after the direct child
                    # exits.  A blocking read would then wait forever and
                    # leave that descendant outside the cleanup path.
                    if process.poll() is not None and not process_group_cleaned:
                        _terminate_process_group(process)
                        process_group_cleaned = True
                    continue
                # Reading the pipe FD directly returns the currently available
                # bytes instead of waiting for a full BufferedReader request.
                chunk = os.read(pipe_fd, 1 << 16)
                if not chunk:
                    break
                allowed = max_bytes - total
                if len(chunk) > allowed:
                    if allowed > 0:
                        handle.write(chunk[:allowed])
                        total += allowed
                    overflow = True
                    break
                handle.write(chunk)
                total += len(chunk)
                if process.poll() is not None and not process_group_cleaned:
                    _terminate_process_group(process)
                    process_group_cleaned = True
            if overflow:
                _terminate_process_group(process)
                process_group_cleaned = True
            status = process.wait()
            if not process_group_cleaned:
                _terminate_process_group(process)
                process_group_cleaned = True
            process.stdout.close()
            handle.flush()
            os.fsync(handle.fileno())

        # Hard-link publication is atomic and fails if OUTPUT appeared after
        # the initial freshness check; unlike mv it cannot overwrite a rival.
        # It also proves that the monitored live bytes and published bytes are
        # the same inode.
        os.link(live_output, output, follow_symlinks=False)
        live_output.unlink()
        live_created = False
        _fsync_directory(parent)

        if overflow:
            raise ValueError("durable log exceeded configured ceiling while capturing")
        report = scan_log(output, max_bytes=max_bytes)
        return report, status
    finally:
        if process is not None and not process_group_cleaned:
            _terminate_process_group(process)
        if live_created:
            try:
                live_output.unlink()
            except FileNotFoundError:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", type=Path)
    parser.add_argument("--max-bytes", type=int, required=True)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.capture is not None:
        if args.log is not None:
            raise ValueError("capture mode does not accept a positional log")
        command = args.command
        if command[:1] == ["--"]:
            command = command[1:]
        report, status = capture_durable_log(
            args.capture, max_bytes=args.max_bytes, command=command
        )
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return status
    if args.log is None or args.command:
        raise ValueError("scan mode requires exactly one log path")
    report = scan_log(args.log, max_bytes=args.max_bytes)
    print(f"log accepted: {report['bytes']} bytes")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"formal log rejected: {error}", file=sys.stderr)
        raise SystemExit(1)
