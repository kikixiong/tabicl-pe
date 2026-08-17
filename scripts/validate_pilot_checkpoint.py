#!/usr/bin/env python3
"""Strict validator for the legacy RoPE/No-PE pilot checkpoints.

This validator intentionally understands only the small checkpoint schema used
by the historical pilot.  Passing it does not upgrade a pilot checkpoint to
formal evidence: the legacy schema has no source, RNG, DataLoader, scaler, or
parent-lineage provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

MAX_CHECKPOINT_BYTES = 300 * 1024 * 1024
REQUIRED_KEYS = {
    "config",
    "curr_step",
    "optimizer_state",
    "scheduler_state",
    "state_dict",
}
STEP_NAME = re.compile(r"^step-([1-9][0-9]*)\.ckpt$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_tensors(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from nested_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from nested_tensors(item)


def all_finite(value: Any) -> bool:
    return all(
        bool(torch.isfinite(tensor).all().item())
        for tensor in nested_tensors(value)
        if tensor.is_floating_point() or tensor.is_complex()
    )


def checkpoint_steps(directory: Path) -> list[tuple[int, Path]]:
    values: list[tuple[int, Path]] = []
    if not directory.is_dir():
        return values
    for child in directory.iterdir():
        match = STEP_NAME.fullmatch(child.name)
        if match is not None:
            values.append((int(match.group(1)), child))
    return sorted(values)


def validate_checkpoint(
    checkpoint_path: Path,
    *,
    expected_mode: str,
    expected_step: int,
    expected_sha256: str | None = None,
    require_latest_in_dir: bool = False,
) -> dict[str, Any]:
    if expected_mode not in {"rope", "none"}:
        raise ValueError("legacy pilot validator accepts only rope or none")
    if expected_step < 1:
        raise ValueError("expected step must be positive")
    if expected_sha256 is not None and SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("expected SHA-256 is malformed")

    path = checkpoint_path.absolute()
    metadata_before = path.lstat()
    if not stat.S_ISREG(metadata_before.st_mode):
        raise ValueError("checkpoint must be a regular non-symlink file")
    if metadata_before.st_size < 1 or metadata_before.st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint size is outside the pilot safety bounds")
    match = STEP_NAME.fullmatch(path.name)
    if match is None or int(match.group(1)) != expected_step:
        raise ValueError("checkpoint filename does not match expected step")

    if require_latest_in_dir:
        steps = checkpoint_steps(path.parent)
        if not steps or steps[-1][1].absolute() != path:
            latest = steps[-1][1].name if steps else "<none>"
            raise ValueError(
                f"checkpoint is not the latest step file in its directory (latest={latest})"
            )

    try:
        with zipfile.ZipFile(path) as archive:
            corrupt_member = archive.testzip()
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"checkpoint ZIP container is invalid: {error}") from error
    if corrupt_member is not None:
        raise ValueError(f"checkpoint ZIP CRC failed for member {corrupt_member}")

    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {expected_sha256}, observed {digest}"
        )

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"checkpoint could not be loaded safely: {error}") from error
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint root must be a mapping")
    missing = REQUIRED_KEYS.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is missing required keys: {sorted(missing)}")
    if checkpoint["curr_step"] != expected_step:
        raise ValueError("checkpoint curr_step does not match expected step")

    config = checkpoint["config"]
    if not isinstance(config, dict) or config.get("row_identity_mode") != expected_mode:
        raise ValueError("checkpoint row_identity_mode does not match expected mode")
    if not isinstance(checkpoint["state_dict"], dict) or not checkpoint["state_dict"]:
        raise ValueError("checkpoint model state is empty or malformed")
    if not isinstance(checkpoint["optimizer_state"], dict):
        raise TypeError("checkpoint optimizer state is malformed")
    if not isinstance(checkpoint["scheduler_state"], dict):
        raise TypeError("checkpoint scheduler state is malformed")
    if not all_finite(checkpoint["state_dict"]):
        raise ValueError("checkpoint model state contains non-finite tensors")
    if not all_finite(checkpoint["optimizer_state"]):
        raise ValueError("checkpoint optimizer state contains non-finite tensors")

    scheduler = checkpoint["scheduler_state"]
    if scheduler.get("last_epoch") != expected_step:
        raise ValueError("checkpoint scheduler last_epoch does not match curr_step")
    if scheduler.get("_step_count") != expected_step + 1:
        raise ValueError("checkpoint scheduler step count does not match curr_step")
    for learning_rate in scheduler.get("_last_lr", []):
        if not isinstance(learning_rate, (int, float)) or not math.isfinite(
            learning_rate
        ):
            raise ValueError("checkpoint scheduler contains a non-finite learning rate")

    metadata_after = path.lstat()
    signature = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if signature(metadata_before) != signature(metadata_after):
        raise ValueError("checkpoint changed while it was being validated")

    return {
        "schema_version": 1,
        "kind": "legacy-pilot-checkpoint-validation",
        "classification": "exploratory-pilot-only",
        "path": str(path),
        "mode": expected_mode,
        "step": expected_step,
        "size_bytes": metadata_after.st_size,
        "sha256": digest,
        "zip_crc_valid": True,
        "model_and_optimizer_finite": True,
        "scheduler_step_aligned": True,
        "limitations": [
            "checkpoint_has_no_source_sha",
            "checkpoint_has_no_rng_or_dataloader_state",
            "checkpoint_has_no_grad_scaler_state",
            "checkpoint_has_no_parent_lineage",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("rope", "none"))
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--require-latest-in-dir", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = validate_checkpoint(
            args.checkpoint,
            expected_mode=args.mode,
            expected_step=args.step,
            expected_sha256=args.expected_sha256,
            require_latest_in_dir=args.require_latest_in_dir,
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"pilot checkpoint validation failed: {error}", file=sys.stderr)
        return 1
    print(canonical_json(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
