#!/usr/bin/env python3
"""Fail closed unless writable work and durable artifacts use different filesystems.

The helper intentionally uses only the Python standard library.  Every path
component is opened with ``O_NOFOLLOW`` and the comparison is made from the
resulting live directory descriptors, not from path-based ``stat`` results.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


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
        raise ValueError("no-follow physical-directory traversal is unavailable")
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def open_physical_directory(path: Path, *, where: str) -> int:
    """Return a descriptor after opening every absolute path component no-follow."""

    path = _absolute_normalized(Path(path), where=where)
    flags = _directory_flags()
    fd = os.open(os.path.sep, flags)
    try:
        for component in path.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        metadata = os.fstat(fd)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{where} must be a physical directory")
        return fd
    except BaseException:
        os.close(fd)
        raise


def require_distinct_filesystems(
    work_root: Path,
    artifact_root: Path,
    *,
    work_label: str = "work root",
    artifact_label: str = "artifact root",
) -> dict[str, Any]:
    """Verify two physical directories reside on different ``st_dev`` values."""

    work_root = _absolute_normalized(Path(work_root), where=work_label)
    artifact_root = _absolute_normalized(Path(artifact_root), where=artifact_label)
    work_fd = open_physical_directory(work_root, where=work_label)
    try:
        artifact_fd = open_physical_directory(artifact_root, where=artifact_label)
        try:
            work_metadata = os.fstat(work_fd)
            artifact_metadata = os.fstat(artifact_fd)
            if work_metadata.st_dev == artifact_metadata.st_dev:
                raise ValueError(
                    f"{work_label} must use a different filesystem from {artifact_label}"
                )
            return {
                "schema_version": 1,
                "work_root": os.fspath(work_root),
                "work_device": int(work_metadata.st_dev),
                "artifact_root": os.fspath(artifact_root),
                "artifact_device": int(artifact_metadata.st_dev),
            }
        finally:
            os.close(artifact_fd)
    finally:
        os.close(work_fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--work-label", default="work root")
    parser.add_argument("--artifact-label", default="artifact root")
    args = parser.parse_args(argv)
    result = require_distinct_filesystems(
        args.work_root,
        args.artifact_root,
        work_label=args.work_label,
        artifact_label=args.artifact_label,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"filesystem isolation failed: {error}", file=sys.stderr)
        raise SystemExit(2)
