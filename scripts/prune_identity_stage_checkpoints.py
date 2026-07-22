#!/usr/bin/env python3
"""After strict final validation, retain only a stage's terminal checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys


_STEP = re.compile(r"^step-([0-9]+)\.ckpt$")


def _pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_finalized(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("finalized manifest must be a regular non-symlink file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid finalized manifest JSON") from error
    if set(value) != {"schema_version", "kind", "payload", "sha256"}:
        raise ValueError("finalized manifest keys mismatch")
    if value["schema_version"] != 1 or value["kind"] != "finalized_checkpoint":
        raise ValueError("finalized manifest schema/kind mismatch")
    body = {key: value[key] for key in ("schema_version", "kind", "payload")}
    if hashlib.sha256(_canonical(body)).hexdigest() != value["sha256"]:
        raise ValueError("finalized manifest self-hash mismatch")
    return value


def prune_validated_stage(
    checkpoint_dir: Path,
    *,
    terminal_step: int,
    finalized_manifest: Path,
    checkpoint_ceiling_bytes: int,
) -> dict[str, object]:
    if terminal_step < 1 or checkpoint_ceiling_bytes < 1:
        raise ValueError("terminal step and checkpoint ceiling must be positive")
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ValueError("checkpoint directory must be a regular non-symlink directory")
    manifest = _load_finalized(finalized_manifest)
    payload = manifest["payload"]
    required = {"checkpoint_sha256", "checkpoint_size", "terminal_step"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("finalized manifest is missing checkpoint binding fields")
    if payload["terminal_step"] != terminal_step:
        raise ValueError("finalized manifest terminal step mismatch")

    terminal = checkpoint_dir / f"step-{terminal_step}.ckpt"
    if terminal.is_symlink() or not terminal.is_file():
        raise ValueError("validated terminal checkpoint is missing or unsafe")
    size = terminal.stat().st_size
    if size > checkpoint_ceiling_bytes:
        raise ValueError("terminal checkpoint exceeds checkpoint ceiling")
    if payload["checkpoint_size"] != size or payload["checkpoint_sha256"] != _sha256(terminal):
        raise ValueError("terminal checkpoint does not match finalized manifest")

    checkpoints: list[tuple[int, Path]] = []
    for child in checkpoint_dir.iterdir():
        match = _STEP.fullmatch(child.name)
        if match:
            if child.is_symlink() or not child.is_file():
                raise ValueError(f"unsafe checkpoint entry: {child.name}")
            if child.stat().st_size > checkpoint_ceiling_bytes:
                raise ValueError(f"checkpoint exceeds ceiling: {child.name}")
            checkpoints.append((int(match.group(1)), child))
        elif child.name.endswith(".tmp") and child.name.startswith(".step-"):
            raise ValueError("unfinished atomic checkpoint exists")
    if len(checkpoints) > 2:
        raise ValueError("checkpoint peak exceeded two simultaneous files")
    temporary = [item for item in checkpoints if item[0] != terminal_step]
    if len(temporary) > 1:
        raise ValueError("more than one retained temporary checkpoint")

    removed: list[str] = []
    for _step, path in temporary:
        path.unlink()
        removed.append(path.name)
    directory_fd = os.open(checkpoint_dir, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    remaining = sorted(path.name for _step, path in checkpoints if path.exists())
    if remaining != [terminal.name]:
        raise RuntimeError("post-validation prune did not leave final-only state")
    return {
        "schema_version": 1,
        "terminal_checkpoint": terminal.name,
        "removed": sorted(removed),
        "remaining": remaining,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--terminal-step", required=True, type=int)
    parser.add_argument("--finalized-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-ceiling-bytes", required=True, type=int)
    args = parser.parse_args(argv)
    report = prune_validated_stage(
        args.checkpoint_dir,
        terminal_step=args.terminal_step,
        finalized_manifest=args.finalized_manifest,
        checkpoint_ceiling_bytes=args.checkpoint_ceiling_bytes,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"formal checkpoint prune failed: {error}", file=sys.stderr)
        raise SystemExit(1)
