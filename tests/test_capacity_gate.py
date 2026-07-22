from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tabicl.train._checkpoint_io import atomic_torch_save
from tabicl.train._run import Trainer


SCRIPT = Path(__file__).parents[1] / "scripts" / "check_formal_capacity.py"
LOG_SCRIPT = Path(__file__).parents[1] / "scripts" / "reject_nonfinite_log.py"
PRUNE_SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "prune_identity_stage_checkpoints.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("check_formal_capacity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_empty_three_arm_study_accounts_for_all_peaks_and_logs():
    capacity = _module()
    checkpoint_ceiling = (1 << 53) + 17
    log_allowance = 123_457

    remaining = capacity.empty_study_remaining_bytes(
        checkpoint_ceiling, log_allowance
    )

    assert remaining == 15 * checkpoint_ceiling + log_allowance
    assert capacity.required_bytes(remaining) == (
        20 * (1 << 30) + (5 * remaining + 3) // 4
    )


def test_audited_remaining_handles_staggered_cross_arm_progress():
    capacity = _module()
    checkpoint_ceiling = 101
    allowance = 17
    initial, consumed, remaining = capacity.remaining_after_audit(
        checkpoint_ceiling,
        allowance,
        checkpoint_count=3,
        # One arm may already have a final while two arms retain differently
        # sized live temporaries.  Byte subtraction neither assumes lockstep
        # progress nor double-counts those existing artifacts.
        checkpoint_bytes=101 + 73 + 29,
        durable_bytes=11,
    )
    assert initial == 15 * checkpoint_ceiling + allowance
    assert consumed == 214
    assert remaining == initial - consumed


def test_statvfs_available_bytes_uses_integer_bavail_times_frsize(tmp_path):
    capacity = _module()
    fake = SimpleNamespace(f_bavail=(1 << 53) + 3, f_frsize=4097)

    assert capacity.available_bytes(tmp_path, statvfs=lambda _path: fake) == (
        ((1 << 53) + 3) * 4097
    )


@pytest.mark.parametrize("delta,allowed", [(0, True), (-1, False)])
def test_capacity_boundary_is_exact(delta, allowed):
    capacity = _module()
    remaining = (1 << 53) + 117
    required = capacity.required_bytes(remaining)

    report = capacity.evaluate_capacity(required + delta, remaining)

    assert report["available_bytes"] == required + delta
    assert report["remaining_bytes"] == remaining
    assert report["required_bytes"] == required
    assert report["allowed"] is allowed


def test_absolute_twenty_gibibyte_floor_blocks_even_with_no_remaining_bytes():
    capacity = _module()

    report = capacity.evaluate_capacity(20 * (1 << 30) - 1, 0)

    assert report["allowed"] is False
    assert "absolute_floor" in report["reasons"]


def test_artifact_ceiling_audit_counts_checkpoint_log_and_monitor_bytes(tmp_path):
    capacity = _module()
    checkpoint = tmp_path / "stage1" / "step-1.ckpt"
    log = tmp_path / "logs" / "run.log"
    monitor = tmp_path / "gpu" / "run.csv"
    checkpoint.parent.mkdir()
    log.parent.mkdir()
    monitor.parent.mkdir()
    checkpoint.write_bytes(b"c" * 11)
    log.write_bytes(b"l" * 7)
    monitor.write_bytes(b"m" * 5)

    report = capacity.audit_artifact_ceilings(
        checkpoint_paths=[checkpoint],
        log_paths=[log],
        monitor_paths=[monitor],
        checkpoint_ceiling_bytes=11,
        durable_log_allowance_bytes=12,
    )

    assert report["checkpoint_bytes"] == 11
    assert report["durable_log_bytes"] == 12
    assert report["allowed"] is True


@pytest.mark.parametrize("kind", ["checkpoint", "durable_logs"])
def test_artifact_ceiling_overflow_fails_closed(tmp_path, kind):
    capacity = _module()
    checkpoint = tmp_path / "step-1.ckpt"
    log = tmp_path / "run.log"
    checkpoint.write_bytes(b"c" * (12 if kind == "checkpoint" else 11))
    log.write_bytes(b"l" * (13 if kind == "durable_logs" else 12))

    with pytest.raises(ValueError, match="ceiling"):
        capacity.audit_artifact_ceilings(
            checkpoint_paths=[checkpoint],
            log_paths=[log],
            monitor_paths=[],
            checkpoint_ceiling_bytes=11,
            durable_log_allowance_bytes=12,
        )


def test_main_outputs_canonical_report_and_refuses_one_byte_short(tmp_path, capsys):
    capacity = _module()
    checkpoint_ceiling = 101
    log_allowance = 17
    remaining = 15 * checkpoint_ceiling + log_allowance
    required = 20 * (1 << 30) + (5 * remaining + 3) // 4
    result = capacity.main(
        [
            "--artifact-root",
            str(tmp_path),
            "--checkpoint-ceiling-bytes",
            str(checkpoint_ceiling),
            "--durable-log-allowance-bytes",
            str(log_allowance),
        ],
        available_bytes_fn=lambda _path: required - 1,
    )

    assert result == 1
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "allowed": False,
        "available_bytes": required - 1,
        "remaining_bytes": remaining,
        "required_bytes": required,
        "reasons": ["remaining_capacity"],
        "schema_version": 1,
    }


@pytest.mark.parametrize("delta,allowed", [(0, True), (-1, False)])
def test_audited_staggered_capacity_exact_boundary(tmp_path, capsys, delta, allowed):
    capacity = _module()
    checkpoint_ceiling = 101
    allowance = 1_000
    (tmp_path / "step-1.ckpt").write_bytes(b"c" * 73)
    (tmp_path / "protocol-ledger.json").write_bytes(b"p" * 29)
    initial = 15 * checkpoint_ceiling + allowance
    consumed = 73 + 29
    remaining = initial - consumed
    required = capacity.required_bytes(remaining)

    result = capacity.main(
        [
            "--artifact-root",
            str(tmp_path),
            "--checkpoint-ceiling-bytes",
            str(checkpoint_ceiling),
            "--durable-log-allowance-bytes",
            str(allowance),
            "--audit-tree",
            "--remaining-from-audit",
            "--run-log-ceiling-bytes",
            "10",
            "--attestation-ceiling-bytes",
            "10",
            "--manifest-ceiling-bytes",
            "10",
            "--protocol-metadata-allowance-bytes",
            "550",
        ],
        available_bytes_fn=lambda _path: required + delta,
    )

    assert result == (0 if allowed else 1)
    report = json.loads(capsys.readouterr().out)
    assert report["allowed"] is allowed
    assert report["initial_study_bytes"] == initial
    assert report["consumed_checkpoint_bytes"] == 73
    assert report["consumed_checkpoint_count"] == 1
    assert report["consumed_durable_bytes"] == 29
    assert report["consumed_bytes"] == consumed
    assert report["remaining_bytes"] == remaining
    assert report["remaining_source"] == "audited_study_bytes"


def test_capacity_retries_when_tree_mutates_between_audit_and_statvfs(
    tmp_path, capsys
):
    capacity = _module()
    artifact = tmp_path / "protocol-ledger.json"
    artifact.write_bytes(b"x" * 10)
    calls = 0

    def mutate_once(_path):
        nonlocal calls
        calls += 1
        if calls == 1:
            artifact.write_bytes(b"y" * 5)
        return 1 << 60

    result = capacity.main(
        [
            "--artifact-root",
            str(tmp_path),
            "--checkpoint-ceiling-bytes",
            "10",
            "--durable-log-allowance-bytes",
            "1000",
            "--audit-tree",
            "--remaining-from-audit",
            "--run-log-ceiling-bytes",
            "10",
            "--attestation-ceiling-bytes",
            "10",
            "--manifest-ceiling-bytes",
            "10",
            "--protocol-metadata-allowance-bytes",
            "550",
        ],
        available_bytes_fn=mutate_once,
    )
    report = json.loads(capsys.readouterr().out)
    assert result == 0
    assert calls == 2
    assert report["consumed_durable_bytes"] == 5


def test_audited_remaining_never_becomes_negative():
    capacity = _module()
    with pytest.raises(ValueError, match="checkpoint bytes exceed"):
        capacity.remaining_after_audit(
            1,
            0,
            checkpoint_count=15,
            checkpoint_bytes=16,
            durable_bytes=0,
        )


@pytest.mark.parametrize(
    "checkpoint_count,checkpoint_bytes,durable_bytes,match",
    [
        (16, 15, 0, "more than fifteen"),
        (15, 151, 0, "checkpoint bytes exceed"),
        (15, 150, 101, "durable bytes exceed"),
    ],
)
def test_audited_remaining_rejects_consumed_budget_overflows(
    checkpoint_count, checkpoint_bytes, durable_bytes, match
):
    capacity = _module()
    with pytest.raises(ValueError, match=match):
        capacity.remaining_after_audit(
            10,
            100,
            checkpoint_count=checkpoint_count,
            checkpoint_bytes=checkpoint_bytes,
            durable_bytes=durable_bytes,
        )


def test_negative_or_boolean_like_integer_input_is_rejected(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--artifact-root",
            str(tmp_path),
            "--checkpoint-ceiling-bytes",
            "-1",
            "--durable-log-allowance-bytes",
            "0",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2


def _checkpoint_manager(tmp_path: Path, *, formal: bool = True) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(
        checkpoint_dir=str(tmp_path),
        max_checkpoints=1,
        save_perm_every=100,
        formal_training=formal,
    )
    return trainer


def test_one_temporary_retained_and_atomic_write_never_exceeds_two_files(
    tmp_path, monkeypatch
):
    trainer = _checkpoint_manager(tmp_path)
    atomic_torch_save({"step": 1}, tmp_path / "step-1.ckpt")
    observed_counts: list[int] = []
    real_save = torch.save

    def observing_save(*args, **kwargs):
        observed_counts.append(len(list(tmp_path.iterdir())))
        return real_save(*args, **kwargs)

    monkeypatch.setattr(torch, "save", observing_save)
    atomic_torch_save({"step": 2}, tmp_path / "step-2.ckpt")
    observed_counts.append(len(list(tmp_path.iterdir())))
    trainer.manage_checkpoint()

    assert max(observed_counts) == 2
    assert [path.name for path in tmp_path.iterdir()] == ["step-2.ckpt"]


def test_checkpoint_prune_unlink_failure_is_fatal(tmp_path, monkeypatch):
    trainer = _checkpoint_manager(tmp_path)
    (tmp_path / "step-1.ckpt").write_bytes(b"old")
    (tmp_path / "step-2.ckpt").write_bytes(b"new")

    def fail_remove(_path):
        raise PermissionError("permission denied")

    monkeypatch.setattr(os, "remove", fail_remove)
    with pytest.raises(PermissionError, match="permission denied"):
        trainer.manage_checkpoint()


def test_nonformal_checkpoint_prune_preserves_legacy_warning_behavior(
    tmp_path, monkeypatch, capsys
):
    trainer = _checkpoint_manager(tmp_path, formal=False)
    (tmp_path / "step-1.ckpt").write_bytes(b"old")
    (tmp_path / "step-2.ckpt").write_bytes(b"new")

    monkeypatch.setattr(
        os, "remove", lambda _path: (_ for _ in ()).throw(OSError("legacy boom"))
    )
    trainer.manage_checkpoint()

    assert "Error removing checkpoint" in capsys.readouterr().out


def test_durable_capture_is_bounded_and_rejects_nonfinite_or_oom(tmp_path):
    logs = _script_module("reject_nonfinite_log", LOG_SCRIPT)
    output = tmp_path / "run.log"
    with pytest.raises(ValueError, match="out of memory"):
        logs.capture_durable_log(
            output,
            max_bytes=100,
            command=[sys.executable, "-c", "print('CUDA out of memory')"],
        )
    assert output.stat().st_size <= 100

    overflow = tmp_path / "overflow.log"
    with pytest.raises(ValueError, match="exceeded"):
        logs.capture_durable_log(
            overflow,
            max_bytes=11,
            command=[sys.executable, "-c", "print('x' * 10000)"],
        )
    assert overflow.stat().st_size == 11


def test_durable_capture_never_replaces_existing_output(tmp_path):
    logs = _script_module("reject_nonfinite_log_race", LOG_SCRIPT)
    output = tmp_path / "run.log"
    output.write_text("authority")
    with pytest.raises(ValueError, match="fresh"):
        logs.capture_durable_log(
            output,
            max_bytes=100,
            command=[sys.executable, "-c", "print('replacement')"],
        )
    assert output.read_text() == "authority"


def _finalized_manifest(checkpoint: Path, terminal_step: int) -> dict:
    payload = {
        "terminal_step": terminal_step,
        "checkpoint_size": checkpoint.stat().st_size,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    body = {"schema_version": 1, "kind": "finalized_checkpoint", "payload": payload}
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return {**body, "sha256": hashlib.sha256(encoded).hexdigest()}


def test_post_validation_prune_leaves_only_terminal_checkpoint(tmp_path):
    prune = _script_module("prune_identity_stage_checkpoints", PRUNE_SCRIPT)
    checkpoint_dir = tmp_path / "stage1"
    checkpoint_dir.mkdir()
    temporary = checkpoint_dir / "step-1.ckpt"
    terminal = checkpoint_dir / "step-500000.ckpt"
    temporary.write_bytes(b"temporary")
    terminal.write_bytes(b"terminal")
    finalized = tmp_path / "stage1.finalized.json"
    finalized.write_text(json.dumps(_finalized_manifest(terminal, 500000)))

    report = prune.prune_validated_stage(
        checkpoint_dir,
        terminal_step=500000,
        finalized_manifest=finalized,
        checkpoint_ceiling_bytes=100,
    )

    assert report["remaining"] == ["step-500000.ckpt"]
    assert sorted(path.name for path in checkpoint_dir.iterdir()) == [
        "step-500000.ckpt"
    ]


def test_post_validation_prune_refuses_three_checkpoint_peak(tmp_path):
    prune = _script_module("prune_identity_stage_checkpoints_peak", PRUNE_SCRIPT)
    checkpoint_dir = tmp_path / "stage1"
    checkpoint_dir.mkdir()
    for step in (1, 2, 500000):
        (checkpoint_dir / f"step-{step}.ckpt").write_bytes(str(step).encode())
    terminal = checkpoint_dir / "step-500000.ckpt"
    finalized = tmp_path / "stage1.finalized.json"
    finalized.write_text(json.dumps(_finalized_manifest(terminal, 500000)))

    with pytest.raises(ValueError, match="two simultaneous"):
        prune.prune_validated_stage(
            checkpoint_dir,
            terminal_step=500000,
            finalized_manifest=finalized,
            checkpoint_ceiling_bytes=100,
        )


def test_durable_subbudgets_sum_across_all_nine_jobs_once():
    capacity = _module()
    report = capacity.validate_durable_budget_partition(
        durable_log_allowance_bytes=18 * 100 + 18 * 10 + 9 * 20 + 40,
        run_log_ceiling_bytes=100,
        attestation_ceiling_bytes=10,
        manifest_ceiling_bytes=20,
        protocol_metadata_allowance_bytes=40,
    )
    assert report["assigned_bytes"] == report["durable_log_allowance_bytes"]

    with pytest.raises(ValueError, match="aggregate"):
        capacity.validate_durable_budget_partition(
            durable_log_allowance_bytes=report["assigned_bytes"] - 1,
            run_log_ceiling_bytes=100,
            attestation_ceiling_bytes=10,
            manifest_ceiling_bytes=20,
            protocol_metadata_allowance_bytes=40,
        )


def test_artifact_tree_audit_counts_every_noncheckpoint_byte(tmp_path):
    capacity = _module()
    (tmp_path / "step-1.ckpt").write_bytes(b"c" * 10)
    (tmp_path / "train.log").write_bytes(b"l" * 5)
    (tmp_path / "source-attestation-trainer.json").write_bytes(b"a" * 3)
    (tmp_path / "finalized-checkpoint.json").write_bytes(b"m" * 4)
    (tmp_path / "protocol-ledger.json").write_bytes(b"p" * 2)

    report = capacity.audit_artifact_tree(
        tmp_path,
        checkpoint_ceiling_bytes=10,
        durable_log_allowance_bytes=14,
        run_log_ceiling_bytes=5,
        attestation_ceiling_bytes=3,
        manifest_ceiling_bytes=4,
    )
    assert report["checkpoint_bytes"] == 10
    assert report["durable_bytes"] == 14

    with pytest.raises(ValueError, match="aggregate"):
        capacity.audit_artifact_tree(
            tmp_path,
            checkpoint_ceiling_bytes=10,
            durable_log_allowance_bytes=13,
            run_log_ceiling_bytes=5,
            attestation_ceiling_bytes=3,
            manifest_ceiling_bytes=4,
        )
