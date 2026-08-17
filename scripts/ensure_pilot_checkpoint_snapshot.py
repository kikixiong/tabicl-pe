#!/usr/bin/env python3
"""Create or verify an immutable copy and SHA manifest for a pilot checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SHA40 = re.compile(r"^[0-9a-f]{40}$")
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")


def load_validator():
    path = Path(__file__).with_name("validate_pilot_checkpoint.py")
    spec = importlib.util.spec_from_file_location("pilot_checkpoint_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pilot checkpoint validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def copy_no_replace(source: Path, destination: Path) -> None:
    """Copy through an fsynced private inode, then link without replacement."""

    if destination.exists() or destination.is_symlink():
        return
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    source_metadata = source.lstat()
    if not stat.S_ISREG(source_metadata.st_mode):
        raise ValueError("snapshot source must be a regular file")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    output = os.open(temporary, flags, 0o400)
    try:
        with (
            source.open("rb") as input_handle,
            os.fdopen(output, "wb", closefd=False) as output_handle,
        ):
            for chunk in iter(lambda: input_handle.read(4 * 1024 * 1024), b""):
                output_handle.write(chunk)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.link(temporary, destination, follow_symlinks=False)
        fsync_directory(destination.parent)
    finally:
        os.close(output)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_no_replace(raw: bytes, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        return
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o400)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
        os.link(temporary, destination, follow_symlinks=False)
        fsync_directory(destination.parent)
    finally:
        os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def validate_manifest(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"existing snapshot manifest is invalid: {error}") from error
    if not isinstance(value, dict) or "manifest_sha256" not in value:
        raise ValueError("existing snapshot manifest has the wrong schema")
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if value["manifest_sha256"] != hashlib.sha256(canonical_json(body)).hexdigest():
        raise ValueError("existing snapshot manifest self-hash is invalid")
    for key in (
        "schema_version",
        "kind",
        "classification",
        "continuation_id",
        "mode",
        "stage",
        "step",
        "continuation_source_commit",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "snapshot_filename",
    ):
        if value.get(key) != expected.get(key):
            raise ValueError(f"existing snapshot manifest differs at {key}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--continuation-id", required=True)
    parser.add_argument("--continuation-source-commit", required=True)
    parser.add_argument("--mode", required=True, choices=("rope", "none"))
    parser.add_argument("--stage", required=True, choices=("stage1", "stage2"))
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--expected-sha256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if SAFE_ID.fullmatch(args.continuation_id) is None:
            raise ValueError("continuation ID is unsafe")
        if SHA40.fullmatch(args.continuation_source_commit) is None:
            raise ValueError("continuation source commit must be a full Git SHA")
        snapshot_dir = args.snapshot_dir.absolute()
        if snapshot_dir.exists() and snapshot_dir.is_symlink():
            raise ValueError("snapshot directory must not be a symlink")
        snapshot_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

        validator = load_validator()
        validation = validator.validate_checkpoint(
            args.checkpoint,
            expected_mode=args.mode,
            expected_step=args.step,
            expected_sha256=args.expected_sha256,
        )
        source = args.checkpoint.absolute()
        snapshot = snapshot_dir / source.name
        copy_no_replace(source, snapshot)
        snapshot_metadata = snapshot.lstat()
        if not stat.S_ISREG(snapshot_metadata.st_mode):
            raise ValueError("snapshot checkpoint is not a regular file")
        snapshot_sha256 = sha256_file(snapshot)
        if (
            snapshot_metadata.st_size != validation["size_bytes"]
            or snapshot_sha256 != validation["sha256"]
        ):
            raise ValueError("snapshot copy does not match its validated source")

        body = {
            "schema_version": 1,
            "kind": "tabicl-legacy-pilot-checkpoint-snapshot",
            "classification": "exploratory-pilot-only",
            "continuation_id": args.continuation_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "stage": args.stage,
            "step": args.step,
            "continuation_source_commit": args.continuation_source_commit,
            "source_provenance_status": "operational-history-only-not-checkpoint-bound",
            "source_checkpoint": str(source),
            "snapshot_filename": snapshot.name,
            "checkpoint_size_bytes": snapshot_metadata.st_size,
            "checkpoint_sha256": snapshot_sha256,
            "limitations": validation["limitations"],
        }
        manifest = dict(body)
        manifest["manifest_sha256"] = hashlib.sha256(canonical_json(body)).hexdigest()
        manifest_path = snapshot_dir / "snapshot-manifest.json"
        if manifest_path.exists() or manifest_path.is_symlink():
            manifest = validate_manifest(manifest_path, manifest)
        else:
            write_no_replace(canonical_json(manifest) + b"\n", manifest_path)
            manifest = validate_manifest(manifest_path, manifest)
        print(canonical_json(manifest).decode("utf-8"))
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"pilot checkpoint snapshot failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
