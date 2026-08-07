from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from tabicl.train._checkpoint_io import atomic_torch_save
from tabicl.train._run import Trainer


SCRIPT = Path(__file__).parents[1] / "scripts" / "check_formal_capacity.py"
LOG_SCRIPT = Path(__file__).parents[1] / "scripts" / "reject_nonfinite_log.py"
DURABLE_LOG_WRAPPER = Path(__file__).parents[1] / "scripts" / "run_with_durable_log.sh"
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
    unit = 4097

    remaining = capacity.empty_study_remaining_bytes(
        checkpoint_ceiling, log_allowance, allocation_unit=unit
    )
    components = capacity.physical_budget_components(
        checkpoint_ceiling, log_allowance, allocation_unit=unit
    )

    assert remaining == components["total_physical_budget_bytes"]
    assert components["checkpoint_physical_budget_bytes"] == 15 * (
        (checkpoint_ceiling + unit - 1) // unit * unit
    )
    assert components["directory_entry_slots"] == 114
    assert components["directory_physical_budget_bytes"] == 131 * unit
    assert capacity.required_bytes(remaining) == (
        20 * (1 << 30) + (5 * remaining + 3) // 4
    )


def test_audited_remaining_handles_staggered_cross_arm_progress():
    capacity = _module()
    checkpoint_ceiling = 101
    allowance = 17
    unit = 64
    initial, consumed, remaining = capacity.remaining_after_audit(
        checkpoint_ceiling,
        allowance,
        checkpoint_count=3,
        # One arm may already have a final while two arms retain differently
        # sized live temporaries.  Byte subtraction neither assumes lockstep
        # progress nor double-counts those existing artifacts.
        checkpoint_bytes=320,
        durable_bytes=64,
        durable_file_count=1,
        directory_count=5,
        directory_bytes=200,
        allocation_unit=unit,
    )
    assert initial == capacity.empty_study_remaining_bytes(
        checkpoint_ceiling, allowance, allocation_unit=unit
    )
    assert consumed == 584
    assert remaining == initial - consumed


def test_statvfs_available_bytes_uses_integer_bavail_times_frsize(tmp_path):
    capacity = _module()
    fake = SimpleNamespace(f_bavail=(1 << 53) + 3, f_frsize=4097)

    assert capacity.available_bytes(tmp_path, statvfs=lambda _path: fake) == (
        ((1 << 53) + 3) * 4097
    )


def test_preflight_rounds_tiny_slots_and_counts_every_directory_slot():
    capacity = _module()
    unit = 4096
    report = capacity.physical_budget_components(
        1, 1, allocation_unit=unit
    )

    assert report["checkpoint_physical_budget_bytes"] == 15 * unit
    assert report["durable_physical_budget_bytes"] == unit
    assert report["directory_entry_slots"] == 114
    assert report["directory_physical_budget_bytes"] == 131 * unit
    assert report["total_physical_budget_bytes"] == 147 * unit
    assert report["durable_file_slots"] == 78
    assert report["durable_entry_slots"] == 82


def test_aggregate_rounding_uses_all_bounded_durable_file_slots_exactly():
    capacity = _module()
    unit = 4096
    # 78 one-byte files use 78 fragments; the 79th aggregate byte must share
    # an already active slot and does not invent an unbounded 79th file.
    assert capacity._aggregate_physical_budget(79, 78, unit) == 78 * unit


@pytest.mark.parametrize("delta,allowed", [(0, True), (-1, False)])
def test_physical_preflight_boundary_is_exact(tmp_path, capsys, delta, allowed):
    capacity = _module()
    unit = 4096
    remaining = capacity.empty_study_remaining_bytes(
        1, 1, allocation_unit=unit
    )
    required = capacity.required_bytes(remaining)
    result = capacity.main(
        [
            "--artifact-root",
            str(tmp_path),
            "--checkpoint-ceiling-bytes",
            "1",
            "--durable-log-allowance-bytes",
            "1",
        ],
        available_bytes_fn=lambda _path: required + delta,
        allocation_unit_bytes_fn=lambda _path: unit,
    )
    report = json.loads(capsys.readouterr().out)
    assert result == (0 if allowed else 1)
    assert report["allowed"] is allowed
    assert report["remaining_bytes"] == remaining
    assert report["required_bytes"] == required


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
    unit = 4096
    remaining = capacity.empty_study_remaining_bytes(
        checkpoint_ceiling, log_allowance, allocation_unit=unit
    )
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
        allocation_unit_bytes_fn=lambda _path: unit,
    )

    assert result == 1
    report = json.loads(capsys.readouterr().out)
    assert report["allowed"] is False
    assert report["available_bytes"] == required - 1
    assert report["remaining_bytes"] == remaining
    assert report["required_bytes"] == required
    assert report["reasons"] == ["remaining_capacity"]
    assert report["schema_version"] == 1
    assert report["allocation_unit_bytes"] == unit
    assert report["durable_file_slots"] == 78
    assert report["durable_entry_slots"] == 82
    assert report["directory_slots"] == 17
    assert report["budget_basis"] == "physical_allocation_bytes"


@pytest.mark.parametrize("delta,allowed", [(0, True), (-1, False)])
def test_audited_staggered_capacity_exact_boundary(tmp_path, capsys, delta, allowed):
    capacity = _module()
    checkpoint_ceiling = 101
    allowance = 1_000
    (tmp_path / "step-1.ckpt").write_bytes(b"c" * 73)
    (tmp_path / "protocol-ledger.json").write_bytes(b"p" * 29)
    unit = 4096
    audited = capacity.audit_artifact_tree(
        tmp_path,
        checkpoint_ceiling_bytes=checkpoint_ceiling,
        durable_log_allowance_bytes=allowance,
        run_log_ceiling_bytes=10,
        attestation_ceiling_bytes=10,
        manifest_ceiling_bytes=10,
        allocation_unit=unit,
    )
    initial, consumed, remaining = capacity.remaining_after_audit(
        checkpoint_ceiling,
        allowance,
        checkpoint_count=audited["checkpoint_count"],
        checkpoint_bytes=audited["checkpoint_bytes"],
        durable_bytes=audited["durable_bytes"],
        durable_file_count=audited["durable_file_count"],
        directory_count=audited["directory_count"],
        directory_bytes=audited["directory_bytes"],
        allocation_unit=unit,
    )
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
            "370",
        ],
        available_bytes_fn=lambda _path: required + delta,
        allocation_unit_bytes_fn=lambda _path: unit,
    )

    assert result == (0 if allowed else 1)
    report = json.loads(capsys.readouterr().out)
    assert report["allowed"] is allowed
    assert report["initial_study_bytes"] == initial
    assert report["consumed_checkpoint_bytes"] == audited["checkpoint_bytes"]
    assert report["consumed_checkpoint_logical_bytes"] == 73
    assert report["consumed_checkpoint_count"] == 1
    assert report["consumed_durable_bytes"] == audited["durable_bytes"]
    assert report["consumed_durable_logical_bytes"] == 29
    assert report["consumed_directory_bytes"] == audited["directory_bytes"]
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
            "370",
        ],
        available_bytes_fn=mutate_once,
        allocation_unit_bytes_fn=lambda _path: 4096,
    )
    report = json.loads(capsys.readouterr().out)
    assert result == 0
    assert calls == 2
    assert report["consumed_durable_logical_bytes"] == 5
    assert report["consumed_durable_bytes"] == capacity._physical_bytes(
        artifact.stat()
    )


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


def test_durable_log_wrapper_keeps_capture_command_out_of_scan_position(tmp_path):
    output = tmp_path / "run.log"
    result = subprocess.run(
        [
            DURABLE_LOG_WRAPPER,
            output,
            "1024",
            sys.executable,
            "-c",
            "import json,sys; print(json.dumps(sys.argv[1:]))",
            "--capture",
            "child-output",
            "--max-bytes",
            "7",
        ],
        env={"PATH": "/usr/bin:/bin", "PYTHON": sys.executable},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    expected = '["--capture", "child-output", "--max-bytes", "7"]\n'
    assert json.loads(result.stdout) == {
        "bytes": len(expected),
        "ok": True,
        "schema_version": 1,
    }
    assert output.read_text() == expected

    scanned = subprocess.run(
        [sys.executable, LOG_SCRIPT, output, "--max-bytes", "1024"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert scanned.returncode == 0, scanned.stderr
    assert scanned.stdout == f"log accepted: {output.stat().st_size} bytes\n"


def test_durable_log_wrapper_preserves_child_failure_and_publishes_log(tmp_path):
    output = tmp_path / "failed.log"
    result = subprocess.run(
        [
            DURABLE_LOG_WRAPPER,
            output,
            "1024",
            sys.executable,
            "-c",
            "print('child stopped'); raise SystemExit(7)",
        ],
        env={"PATH": "/usr/bin:/bin", "PYTHON": sys.executable},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 7, result.stderr
    assert json.loads(result.stdout)["ok"] is True
    assert output.read_text() == "child stopped\n"
    assert not output.with_name(output.name + ".live").exists()


@pytest.mark.parametrize("tail", [[], ["--"]])
def test_capture_cli_rejects_missing_separator_or_empty_command(tmp_path, tail):
    output = tmp_path / f"invalid-{len(tail)}.log"
    result = subprocess.run(
        [
            sys.executable,
            LOG_SCRIPT,
            "--max-bytes",
            "1024",
            "--capture",
            output,
            *tail,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    assert not output.exists()


def test_durable_capture_overflow_kills_stubborn_process_group_descendants(
    tmp_path,
):
    logs = _script_module("reject_nonfinite_log_process_group", LOG_SCRIPT)
    output = tmp_path / "overflow-group.log"
    grandchild_pid_path = tmp_path / "grandchild.pid"
    child_script = """
import pathlib
import signal
import subprocess
import sys
import time

grandchild = subprocess.Popen([
    sys.executable,
    "-c",
    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
])
pathlib.Path(sys.argv[1]).write_text(str(grandchild.pid))
sys.stdout.write("x" * 65536)
sys.stdout.flush()
time.sleep(60)
"""

    started = time.monotonic()
    with pytest.raises(ValueError, match="exceeded"):
        logs.capture_durable_log(
            output,
            max_bytes=16,
            command=[
                sys.executable,
                "-u",
                "-c",
                child_script,
                str(grandchild_pid_path),
            ],
        )
    assert time.monotonic() - started < 10.0
    grandchild_pid = int(grandchild_pid_path.read_text())

    def process_is_live(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        proc_stat = Path(f"/proc/{pid}/stat")
        try:
            # Zombies no longer execute or retain the captured pipe; their
            # parent/init reaps them asynchronously.
            return proc_stat.read_text().split()[2] != "Z"
        except FileNotFoundError:
            return False

    deadline = time.monotonic() + 5.0
    while process_is_live(grandchild_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not process_is_live(grandchild_pid)
    assert output.stat().st_size == 16
    assert not (tmp_path / "overflow-group.log.live").exists()


def test_durable_capture_publication_exception_leaves_no_process_group(
    tmp_path, monkeypatch
):
    logs = _script_module("reject_nonfinite_log_exception_group", LOG_SCRIPT)
    output = tmp_path / "exception-group.log"
    grandchild_pid_path = tmp_path / "exception-grandchild.pid"
    child_script = """
import pathlib
import subprocess
import sys

grandchild = subprocess.Popen([
    sys.executable,
    "-c",
    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
])
pathlib.Path(sys.argv[1]).write_text(str(grandchild.pid))
"""

    monkeypatch.setattr(
        logs.os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("forced publication failure")
        ),
    )
    with pytest.raises(OSError, match="forced publication failure"):
        logs.capture_durable_log(
            output,
            max_bytes=16,
            command=[
                sys.executable,
                "-u",
                "-c",
                child_script,
                str(grandchild_pid_path),
            ],
        )
    grandchild_pid = int(grandchild_pid_path.read_text())
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{grandchild_pid}/stat").read_text().split()[2]
        except (FileNotFoundError, ProcessLookupError):
            break
        if state == "Z":
            break
        time.sleep(0.02)
    else:
        pytest.fail("stubborn grandchild survived capture exception cleanup")
    assert not output.exists()
    assert not (tmp_path / "exception-group.log.live").exists()


def test_durable_capture_exposes_bounded_live_log_then_publishes_same_inode(
    tmp_path,
):
    logs = _script_module("reject_nonfinite_log_live", LOG_SCRIPT)
    output = tmp_path / "train.log"
    live = tmp_path / "train.log.live"
    release = tmp_path / "release"
    result = {}

    def capture():
        try:
            result["value"] = logs.capture_durable_log(
                output,
                max_bytes=512,
                command=[
                    sys.executable,
                    "-u",
                    "-c",
                    (
                        "import pathlib,sys,time; "
                        "gate=pathlib.Path(sys.argv[1]); "
                        "print('step=1 loss=0.5', flush=True); "
                        "\nwhile not gate.exists(): time.sleep(0.01)\n"
                        "print('step=2 loss=0.4', flush=True)"
                    ),
                    str(release),
                ],
            )
        except BaseException as error:  # surfaced in the assertion thread
            result["error"] = error

    thread = threading.Thread(target=capture)
    thread.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if live.exists() and b"step=1" in live.read_bytes():
            break
        time.sleep(0.01)
    else:
        pytest.fail("bounded live log did not become readable")

    live_stat = live.stat()
    assert live_stat.st_size <= 512
    assert not output.exists()
    release.touch()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert "error" not in result
    assert result["value"][1] == 0
    assert not live.exists()
    assert output.stat().st_ino == live_stat.st_ino
    assert output.stat().st_size <= 512
    assert output.read_text().splitlines() == [
        "step=1 loss=0.5",
        "step=2 loss=0.4",
    ]


@pytest.mark.parametrize(
    "message",
    ["ENOSPC", "No space left on device", "OSError: [Errno 28] failed"],
)
def test_durable_capture_rejects_storage_exhaustion_signatures(tmp_path, message):
    logs = _script_module(f"reject_enospc_{hash(message)}", LOG_SCRIPT)
    output = tmp_path / "run.log"

    with pytest.raises(ValueError, match="storage exhausted"):
        logs.capture_durable_log(
            output,
            max_bytes=1_024,
            command=[sys.executable, "-c", f"print({message!r})"],
        )

    assert output.exists()
    assert not (tmp_path / "run.log.live").exists()


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
        durable_log_allowance_bytes=36 * 100 + 18 * 10 + 9 * 20 + 40,
        run_log_ceiling_bytes=100,
        attestation_ceiling_bytes=10,
        manifest_ceiling_bytes=20,
        protocol_metadata_allowance_bytes=40,
    )
    assert report["assigned_bytes"] == report["durable_log_allowance_bytes"]
    assert report["assigned_bytes"] == 4 * 9 * 100 + 2 * 9 * 10 + 9 * 20 + 40

    # The historical two-output/job calculation omitted the externally owned
    # scheduler stdout/stderr slots.  The standalone stage capacity gate must
    # reject that under-budget even when every other partition is exact.
    legacy_two_output_budget = 2 * 9 * 100 + 2 * 9 * 10 + 9 * 20 + 40
    with pytest.raises(ValueError, match="aggregate"):
        capacity.validate_durable_budget_partition(
            durable_log_allowance_bytes=legacy_two_output_budget,
            run_log_ceiling_bytes=100,
            attestation_ceiling_bytes=10,
            manifest_ceiling_bytes=20,
            protocol_metadata_allowance_bytes=40,
        )

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
    assert report["checkpoint_logical_bytes"] == 10
    assert report["durable_logical_bytes"] == 14
    assert report["checkpoint_bytes"] == capacity._physical_bytes(
        (tmp_path / "step-1.ckpt").stat()
    )
    assert report["durable_bytes"] == sum(
        capacity._physical_bytes(path.stat())
        for path in tmp_path.iterdir()
        if path.is_file() and path.name != "step-1.ckpt"
    )

    with pytest.raises(ValueError, match="aggregate"):
        capacity.audit_artifact_tree(
            tmp_path,
            checkpoint_ceiling_bytes=10,
            durable_log_allowance_bytes=13,
            run_log_ceiling_bytes=5,
            attestation_ceiling_bytes=3,
            manifest_ceiling_bytes=4,
        )


def test_artifact_tree_counts_empty_directory_physical_allocation(tmp_path):
    capacity = _module()
    empty = tmp_path / "empty-stage"
    empty.mkdir()
    report = capacity.audit_artifact_tree(
        tmp_path,
        checkpoint_ceiling_bytes=1,
        durable_log_allowance_bytes=1,
        run_log_ceiling_bytes=1,
        attestation_ceiling_bytes=1,
        manifest_ceiling_bytes=1,
    )
    expected = capacity._physical_bytes(tmp_path.stat()) + capacity._physical_bytes(
        empty.stat()
    )
    assert report["directory_count"] == 2
    assert report["directory_observed_physical_bytes"] == expected
    assert report["directory_entry_count"] == 2
    assert report["directory_reserved_consumed_bytes"] == 4 * 4096
    assert report["directory_bytes"] == max(expected, 4 * 4096)
    assert report["durable_bytes"] == 0


@pytest.mark.parametrize("delta,allowed", [(0, True), (-1, False)])
def test_audited_parent_dirent_growth_boundary_is_exact(
    tmp_path, capsys, delta, allowed
):
    capacity = _module()
    unit = 4096
    audited = capacity.audit_artifact_tree(
        tmp_path,
        checkpoint_ceiling_bytes=1,
        durable_log_allowance_bytes=1_000,
        run_log_ceiling_bytes=10,
        attestation_ceiling_bytes=10,
        manifest_ceiling_bytes=10,
        allocation_unit=unit,
    )
    initial, consumed, remaining = capacity.remaining_after_audit(
        1,
        1_000,
        checkpoint_count=0,
        checkpoint_bytes=0,
        durable_bytes=0,
        durable_file_count=0,
        directory_count=audited["directory_count"],
        directory_bytes=audited["directory_bytes"],
        allocation_unit=unit,
    )
    assert audited["directory_entry_count"] == 1
    assert audited["directory_reserved_consumed_bytes"] == 2 * unit
    assert consumed == audited["directory_bytes"]
    assert initial - consumed == remaining
    required = capacity.required_bytes(remaining)
    result = capacity.main(
        [
            "--artifact-root",
            str(tmp_path),
            "--checkpoint-ceiling-bytes",
            "1",
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
            "370",
        ],
        available_bytes_fn=lambda _path: required + delta,
        allocation_unit_bytes_fn=lambda _path: unit,
    )
    report = json.loads(capsys.readouterr().out)
    assert result == (0 if allowed else 1)
    assert report["allowed"] is allowed
    assert report["remaining_bytes"] == remaining


def test_artifact_tree_rejects_more_than_bounded_directory_slots(tmp_path):
    capacity = _module()
    for index in range(capacity.FORMAL_DIRECTORY_SLOTS):
        (tmp_path / f"directory-{index:02d}").mkdir()
    with pytest.raises(ValueError, match="too many directory slots"):
        capacity.audit_artifact_tree(
            tmp_path,
            checkpoint_ceiling_bytes=1,
            durable_log_allowance_bytes=1,
            run_log_ceiling_bytes=1,
            attestation_ceiling_bytes=1,
            manifest_ceiling_bytes=1,
        )


def test_artifact_tree_deduplicates_atomic_hardlink_physical_bytes(tmp_path):
    capacity = _module()
    originals = []
    for index in range(capacity.DURABLE_FILE_SLOTS):
        path = tmp_path / f"metadata-{index:02d}.json"
        path.write_bytes(b"x")
        originals.append(path)
    (tmp_path / ".metadata-atomic.tmp").hardlink_to(originals[0])
    report = capacity.audit_artifact_tree(
        tmp_path,
        checkpoint_ceiling_bytes=1,
        durable_log_allowance_bytes=capacity.DURABLE_FILE_SLOTS,
        run_log_ceiling_bytes=1,
        attestation_ceiling_bytes=1,
        manifest_ceiling_bytes=1,
    )
    assert report["durable_file_count"] == capacity.DURABLE_FILE_SLOTS
    assert report["durable_entry_count"] == capacity.DURABLE_FILE_SLOTS + 1
    assert report["durable_logical_bytes"] == capacity.DURABLE_FILE_SLOTS
    assert report["durable_bytes"] == sum(
        capacity._physical_bytes(path.stat()) for path in originals
    )


def test_directory_audit_counts_cross_fragment_dirent_growth(tmp_path):
    capacity = _module()
    originals = []
    for index in range(capacity.DURABLE_FILE_SLOTS):
        path = tmp_path / ("metadata-" + "x" * 180 + f"-{index:02d}")
        path.write_bytes(b"x")
        originals.append(path)
    for index in range(capacity.ATOMIC_DURABLE_ENTRY_SLOTS):
        (tmp_path / f".atomic-long-name-{index}.tmp").hardlink_to(
            originals[index]
        )
    report = capacity.audit_artifact_tree(
        tmp_path,
        checkpoint_ceiling_bytes=1,
        durable_log_allowance_bytes=capacity.DURABLE_FILE_SLOTS,
        run_log_ceiling_bytes=1,
        attestation_ceiling_bytes=1,
        manifest_ceiling_bytes=1,
    )
    assert report["durable_entry_count"] == capacity.DURABLE_ENTRY_SLOTS
    assert report["directory_reserved_consumed_bytes"] == (
        report["directory_count"] + report["directory_entry_count"]
    ) * report["allocation_unit_bytes"]
    assert report["directory_reserved_consumed_bytes"] > report[
        "allocation_unit_bytes"
    ]
    assert report["directory_bytes"] == max(
        report["directory_observed_physical_bytes"],
        report["directory_reserved_consumed_bytes"],
    )


def test_artifact_tree_rejects_more_than_atomic_entry_peak(tmp_path):
    capacity = _module()
    originals = []
    for index in range(capacity.DURABLE_FILE_SLOTS):
        path = tmp_path / f"metadata-{index:02d}.json"
        path.write_bytes(b"x")
        originals.append(path)
    for index in range(capacity.ATOMIC_DURABLE_ENTRY_SLOTS + 1):
        (tmp_path / f".atomic-{index}.tmp").hardlink_to(originals[index])
    with pytest.raises(ValueError, match="too many durable file entries"):
        capacity.audit_artifact_tree(
            tmp_path,
            checkpoint_ceiling_bytes=1,
            durable_log_allowance_bytes=capacity.DURABLE_FILE_SLOTS,
            run_log_ceiling_bytes=1,
            attestation_ceiling_bytes=1,
            manifest_ceiling_bytes=1,
        )


def test_artifact_tree_rejects_more_than_durable_inode_slots(tmp_path):
    capacity = _module()
    for index in range(capacity.DURABLE_FILE_SLOTS + 1):
        (tmp_path / f"unique-{index:02d}.json").write_bytes(b"x")
    with pytest.raises(ValueError, match="too many durable file slots"):
        capacity.audit_artifact_tree(
            tmp_path,
            checkpoint_ceiling_bytes=1,
            durable_log_allowance_bytes=capacity.DURABLE_FILE_SLOTS + 1,
            run_log_ceiling_bytes=1,
            attestation_ceiling_bytes=1,
            manifest_ceiling_bytes=1,
        )
