from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
VALIDATOR = ROOT / "scripts" / "validate_pilot_checkpoint.py"
SNAPSHOTTER = ROOT / "scripts" / "ensure_pilot_checkpoint_snapshot.py"


def write_checkpoint(path: Path, *, mode: str, step: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "config": {"row_identity_mode": mode},
        "state_dict": {"weight": torch.tensor([1.0, 2.0])},
        "optimizer_state": {"state": {0: {"momentum": torch.tensor([0.5])}}},
        "scheduler_state": {
            "last_epoch": step,
            "_step_count": step + 1,
            "_last_lr": [1e-4],
        },
        "curr_step": step,
    }
    torch.save(checkpoint, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validator_accepts_exact_latest_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stage1" / "step-479000.ckpt"
    digest = write_checkpoint(checkpoint, mode="rope", step=479000)
    result = subprocess.run(
        [
            sys.executable,
            VALIDATOR,
            "--checkpoint",
            checkpoint,
            "--mode",
            "rope",
            "--step",
            "479000",
            "--expected-sha256",
            digest,
            "--require-latest-in-dir",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["classification"] == "exploratory-pilot-only"
    assert value["sha256"] == digest
    assert value["scheduler_step_aligned"] is True


def test_validator_rejects_wrong_mode_and_non_latest_checkpoint(tmp_path: Path) -> None:
    older = tmp_path / "stage1" / "step-478000.ckpt"
    newer = tmp_path / "stage1" / "step-479000.ckpt"
    write_checkpoint(older, mode="rope", step=478000)
    write_checkpoint(newer, mode="rope", step=479000)

    wrong_mode = subprocess.run(
        [
            sys.executable,
            VALIDATOR,
            "--checkpoint",
            newer,
            "--mode",
            "none",
            "--step",
            "479000",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert wrong_mode.returncode == 1
    assert "row_identity_mode" in wrong_mode.stderr

    not_latest = subprocess.run(
        [
            sys.executable,
            VALIDATOR,
            "--checkpoint",
            older,
            "--mode",
            "rope",
            "--step",
            "478000",
            "--require-latest-in-dir",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert not_latest.returncode == 1
    assert "not the latest" in not_latest.stderr


def test_snapshot_copy_and_manifest_are_idempotent_and_self_hashed(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "source" / "step-500000.ckpt"
    digest = write_checkpoint(checkpoint, mode="none", step=500000)
    snapshot_dir = tmp_path / "snapshots" / "none" / "stage1-step500000"
    command = [
        sys.executable,
        SNAPSHOTTER,
        "--checkpoint",
        checkpoint,
        "--snapshot-dir",
        snapshot_dir,
        "--continuation-id",
        "unit-test-v1",
        "--continuation-source-commit",
        "1" * 40,
        "--mode",
        "none",
        "--stage",
        "stage1",
        "--step",
        "500000",
        "--expected-sha256",
        digest,
    ]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert json.loads(first.stdout) == json.loads(second.stdout)

    snapshot = snapshot_dir / checkpoint.name
    manifest_path = snapshot_dir / "snapshot-manifest.json"
    value = json.loads(manifest_path.read_text())
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    raw = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    assert value["manifest_sha256"] == hashlib.sha256(raw).hexdigest()
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == digest
    assert (
        value["source_provenance_status"]
        == "operational-history-only-not-checkpoint-bound"
    )
