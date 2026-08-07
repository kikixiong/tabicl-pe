from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from dataclasses import replace
from datetime import datetime, timedelta
import io
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import zipfile

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "monitor_formal_identity.py"


def test_monitor_is_a_standalone_public_script() -> None:
    assert SCRIPT.is_file()


@pytest.fixture(scope="module")
def monitor_module():
    spec = importlib.util.spec_from_file_location("formal_anomaly_monitor", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _complete_gpu_window(module, values_by_gpu: dict[str, list[float]], *, gap: float = 1.0):
    records = [module.GpuRecord("training_start", 99.5)]
    for gpu_uuid, values in values_by_gpu.items():
        records.extend(
            module.GpuRecord("sample", 100.0 + index * gap, gpu_uuid, value)
            for index, value in enumerate(values)
        )
    records.append(module.GpuRecord("training_end", 110.0))
    return records


def test_gpu_gate_uses_every_one_second_sample_and_accepts_exact_threshold(
    monitor_module,
) -> None:
    records = _complete_gpu_window(
        monitor_module,
        {
            "GPU-a": [0.0] + [90.0] * 9,
            "GPU-b": [80.0] * 10,
        },
    )

    result = monitor_module.evaluate_gpu_window(
        records,
        utilization_required=True,
        expected_gpu_uuids=("GPU-a", "GPU-b"),
        expected_gpu_count=2,
    )

    assert result.complete is True
    assert result.issues == ()
    assert result.sample_counts == {"GPU-a": 10, "GPU-b": 10}
    assert result.means == {"GPU-a": 81.0, "GPU-b": 80.0}


@pytest.mark.parametrize(
    ("records", "expected_code"),
    [
        (lambda module: _complete_gpu_window(module, {"GPU-a": [90.0] * 9}), "gpu_samples_insufficient"),
        (
            lambda module: _complete_gpu_window(
                module, {"GPU-a": [90.0] * 10}, gap=0.49
            ),
            "gpu_sample_cadence_invalid",
        ),
        (
            lambda module: _complete_gpu_window(
                module, {"GPU-a": [90.0] * 10}, gap=1.51
            ),
            "gpu_sample_cadence_invalid",
        ),
        (
            lambda module: _complete_gpu_window(module, {"GPU-a": [79.9] * 10}),
            "gpu_utilization_low",
        ),
    ],
)
def test_gpu_gate_fails_closed_on_window_contract_violations(
    monitor_module, records, expected_code
) -> None:
    result = monitor_module.evaluate_gpu_window(
        records(monitor_module),
        utilization_required=True,
        expected_gpu_uuids=("GPU-a",),
        expected_gpu_count=1,
    )

    assert expected_code in {issue.code for issue in result.issues}


def test_gpu_gate_is_not_evaluated_for_functional_only_cases(monitor_module) -> None:
    result = monitor_module.evaluate_gpu_window([], utilization_required=False)

    assert result.complete is False
    assert result.issues == ()


def test_gpu_gate_rejects_a_missing_allocated_gpu(monitor_module) -> None:
    records = _complete_gpu_window(monitor_module, {"GPU-a": [90.0] * 10})

    result = monitor_module.evaluate_gpu_window(
        records,
        utilization_required=True,
        expected_gpu_uuids=("GPU-a", "GPU-b"),
        expected_gpu_count=2,
    )

    assert "gpu_set_mismatch" in {issue.code for issue in result.issues}


def test_gpu_gate_can_bind_allocated_uuids_from_start_attestation(
    monitor_module,
) -> None:
    records = _complete_gpu_window(monitor_module, {"GPU-a": [90.0] * 10})
    records[0] = monitor_module.GpuRecord(
        "training_start",
        99.5,
        expected_gpu_uuids=("GPU-a", "GPU-b"),
        expected_gpu_count=2,
    )

    result = monitor_module.evaluate_gpu_window(
        records,
        utilization_required=True,
        expected_gpu_uuids=(),
        expected_gpu_count=2,
    )

    assert "gpu_set_mismatch" in {issue.code for issue in result.issues}


def test_gpu_gate_rejects_nonmonotonic_sample_order(monitor_module) -> None:
    records = _complete_gpu_window(monitor_module, {"GPU-a": [90.0] * 10})
    records[2], records[3] = records[3], records[2]

    result = monitor_module.evaluate_gpu_window(
        records,
        utilization_required=True,
        expected_gpu_uuids=("GPU-a",),
        expected_gpu_count=1,
    )

    assert "gpu_sample_cadence_invalid" in {issue.code for issue in result.issues}


@pytest.mark.parametrize(("start", "end"), [(0.0, 110.0), (99.5, 1_000.0)])
def test_gpu_gate_rejects_unobserved_active_window_edges(
    monitor_module, start, end
) -> None:
    records = _complete_gpu_window(monitor_module, {"GPU-a": [90.0] * 10})
    records[0] = monitor_module.GpuRecord("training_start", start)
    records[-1] = monitor_module.GpuRecord("training_end", end)

    result = monitor_module.evaluate_gpu_window(
        records,
        utilization_required=True,
        expected_gpu_uuids=("GPU-a",),
        expected_gpu_count=1,
    )

    assert "gpu_sample_cadence_invalid" in {issue.code for issue in result.issues}


@pytest.mark.parametrize(
    "raw",
    [
        '{"kind":"sample","kind":"training_start","monotonic_seconds":1}',
        '{"kind":"sample","monotonic_seconds":NaN,"gpu_uuid":"GPU-a",'
        '"utilization_percent":80}',
    ],
)
def test_gpu_jsonl_parser_rejects_duplicate_keys_and_nonfinite_constants(
    monitor_module, raw
) -> None:
    records, invalid = monitor_module.parse_gpu_records(raw + "\n")

    assert records == []
    assert invalid == 1


class FakeClock:
    def __init__(self, value: float = 1_000.0) -> None:
        self.value = value

    def time(self) -> float:
        return self.value

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class FakeScheduler:
    def __init__(self, module, states: dict[str, str]) -> None:
        self.module = module
        self.states = states
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv, *, timeout_seconds):
        del timeout_seconds
        command = tuple(str(item) for item in argv)
        self.calls.append(command)
        if Path(command[0]).name == "squeue":
            output = "\n".join(
                f"{job_id}|{state}"
                for job_id, state in self.states.items()
                if state in {"PENDING", "RUNNING"}
            )
            return self.module.CommandResult(0, output, "")
        if Path(command[0]).name == "sacct":
            output = "\n".join(
                f"{job_id}|{state}|0:0"
                for job_id, state in self.states.items()
            )
            return self.module.CommandResult(0, output, "")
        raise AssertionError(f"unexpected command: {command!r}")


class FailingScheduler(FakeScheduler):
    def __init__(self, module, states: dict[str, str], failing_command: str) -> None:
        super().__init__(module, states)
        self.failing_command = failing_command

    def __call__(self, argv, *, timeout_seconds):
        if Path(argv[0]).name == self.failing_command:
            self.calls.append(tuple(str(item) for item in argv))
            raise TimeoutError("bounded scheduler query timed out")
        return super().__call__(argv, timeout_seconds=timeout_seconds)


def _healthy_spec(module, tmp_path: Path, *, utilization_required: bool = False):
    log_path = tmp_path / "train.log"
    log_path.write_text("step=1 loss=0.5\n")
    gpu_path = tmp_path / "gpu.jsonl"
    if utilization_required:
        gpu_path.write_text("")
    job = module.JobSpec(
        job_id="job-a",
        arm="rope",
        stage="stage1",
        log_path=log_path,
        checkpoint_path=tmp_path / "step-500000.ckpt",
        validation_report_path=tmp_path / "validation.json",
        gpu_samples_path=gpu_path if utilization_required else None,
        utilization_required=utilization_required,
        stall_seconds=60.0,
        checkpoint_stale_seconds=120.0,
        max_checkpoint_bytes=1 << 20,
        expected_gpu_uuids=("GPU-a",) if utilization_required else (),
        expected_gpu_count=1 if utilization_required else 0,
    )
    return module.MonitorSpec(disk_path=tmp_path, jobs=(job,))


def _statvfs_with_available(available_bytes: int):
    return SimpleNamespace(f_bavail=available_bytes, f_frsize=1)


def test_unchanged_healthy_polls_are_silent_and_scheduler_queries_are_read_only(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    clock = FakeClock()
    scheduler = FakeScheduler(monitor_module, {"job-a": "PENDING"})
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=clock,
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    first = monitor.poll({})
    second = monitor.poll(first.state)

    assert first.events == ()
    assert second.events == ()
    assert {Path(call[0]).name for call in scheduler.calls} == {"squeue", "sacct"}
    assert len(scheduler.calls) == 4


def test_scheduler_state_change_emits_one_stage_transition_then_is_silent(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    clock = FakeClock()
    scheduler = FakeScheduler(monitor_module, {"job-a": "PENDING"})
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=clock,
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )
    initial = monitor.poll({})
    scheduler.states["job-a"] = "RUNNING"

    changed = monitor.poll(initial.state)
    unchanged = monitor.poll(changed.state)

    assert [(event.category, event.code) for event in changed.events] == [
        ("stage_transition", "scheduler_state_changed")
    ]
    assert changed.events[0].details == {"from": "PENDING", "to": "RUNNING"}
    assert unchanged.events == ()


def test_disk_thresholds_emit_once_and_below_twenty_is_only_an_event(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    scheduler = FakeScheduler(monitor_module, {"job-a": "PENDING"})
    available = [22 * (1 << 30), 22 * (1 << 30) - 1, 20 * (1 << 30) - 1]
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(available[0]),
    )
    state = monitor.poll({}).state
    assert monitor.poll(state).events == ()

    monitor.statvfs = lambda _path: _statvfs_with_available(available[1])
    warning = monitor.poll(state)
    assert [(event.code, event.severity) for event in warning.events] == [
        ("disk_space_low", "warning")
    ]
    assert monitor.poll(warning.state).events == ()

    monitor.statvfs = lambda _path: _statvfs_with_available(available[2])
    blocked = monitor.poll(warning.state)
    assert [(event.code, event.severity) for event in blocked.events] == [
        ("disk_space_low", "submit_block")
    ]
    assert monitor.poll(blocked.state).events == ()


def test_failed_scheduler_state_is_an_anomaly_even_on_first_observation(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    scheduler = FakeScheduler(monitor_module, {"job-a": "OUT_OF_MEMORY"})
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    failed = monitor.poll({})
    repeated = monitor.poll(failed.state)

    assert [(event.category, event.code) for event in failed.events] == [
        ("anomaly", "scheduler_failure")
    ]
    assert failed.events[0].details["state"] == "OUT_OF_MEMORY"
    assert repeated.events == ()


def test_running_step_stall_emits_once_until_progress_resumes(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    spec.jobs[0].log_path.write_text("step=10 loss=0.5\n")
    clock = FakeClock()
    scheduler = FakeScheduler(monitor_module, {"job-a": "RUNNING"})
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=clock,
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    initial = monitor.poll({})
    clock.value += 59.9
    assert monitor.poll(initial.state).events == ()
    clock.value += 0.1
    stalled = monitor.poll(initial.state)
    assert [(event.category, event.code) for event in stalled.events] == [
        ("anomaly", "step_stalled")
    ]
    assert monitor.poll(stalled.state).events == ()

    # Change the byte length as well as the content. Some shared filesystems
    # coalesce back-to-back same-size writes into one observable mtime tick.
    spec.jobs[0].log_path.write_text("step=11 loss=0.40\n")
    progressed = monitor.poll(stalled.state)
    assert progressed.events == ()
    clock.value += 60.0
    stalled_again = monitor.poll(progressed.state)
    assert [event.code for event in stalled_again.events] == ["step_stalled"]


def test_tqdm_step_progress_uses_completed_numerator_not_percent(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    spec.jobs[0].log_path.write_text(
        "Step:  42%|████▏     | 210000/500000 [2:00:00<2:50:00, 29.41it/s]\n"
    )
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=FakeScheduler(monitor_module, {"job-a": "RUNNING"}),
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    observed = monitor.poll({})

    assert observed.events == ()
    assert observed.state["jobs"]["job-a"]["last_step"] == 210_000


def test_explicit_pilot_step_offset_reports_global_progress(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    spec.jobs[0].log_path.write_text(
        "Step:  7%|▋         | 33945/494000 [18:59:12<1:00:00, 1.90s/it]\r"
    )
    spec = replace(spec, jobs=(replace(spec.jobs[0], step_offset=6_000),))
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=FakeScheduler(monitor_module, {"job-a": "RUNNING"}),
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    observed = monitor.poll({})

    assert observed.events == ()
    assert observed.state["jobs"]["job-a"]["last_step"] == 39_945


def test_step_regression_and_error_signatures_are_deduplicated(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    spec.jobs[0].log_path.write_text("step=10 loss=0.5\n")
    scheduler = FakeScheduler(monitor_module, {"job-a": "RUNNING"})
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )
    initial = monitor.poll({})
    spec.jobs[0].log_path.write_text(
        "step=9\n"
        "loss became non-finite\n"
        "CUDA out of memory\n"
        "OSError: [Errno 28] No space left on device\n"
        "Traceback (most recent call last):\n"
        "RuntimeError: failed\n"
    )

    anomalous = monitor.poll(initial.state)
    repeated = monitor.poll(anomalous.state)

    assert {event.code for event in anomalous.events} == {
        "step_regressed",
        "log_nonfinite",
        "log_oom",
        "log_enospc",
        "log_traceback",
        "log_error",
    }
    assert repeated.events == ()


def test_log_observation_is_bounded_to_tail_and_never_mutates_artifact(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    original = b"CUDA out of memory\n" + b"x" * 2048 + b"\nstep=12 loss=0.4\n"
    spec.jobs[0].log_path.write_bytes(original)
    scheduler = FakeScheduler(monitor_module, {"job-a": "RUNNING"})
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
        max_tail_bytes=128,
    )

    result = monitor.poll({})

    assert all(event.code != "log_oom" for event in result.events)
    assert spec.jobs[0].log_path.read_bytes() == original


def test_monitor_follows_bounded_live_log_through_atomic_final_publication(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    final_log = spec.jobs[0].log_path
    final_log.unlink()
    live_log = tmp_path / "train.log.live"
    live_log.write_text("step=1 loss=0.5\n")
    live_inode = live_log.stat().st_ino
    spec = replace(
        spec, jobs=(replace(spec.jobs[0], live_log_path=live_log),)
    )
    scheduler = FakeScheduler(monitor_module, {"job-a": "RUNNING"})
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    active = monitor.poll({})
    assert active.events == ()
    assert active.state["jobs"]["job-a"]["last_step"] == 1

    os.link(live_log, final_log, follow_symlinks=False)
    live_log.unlink()
    assert final_log.stat().st_ino == live_inode
    scheduler.states["job-a"] = "COMPLETED"
    published = monitor.poll(active.state)

    assert [event.code for event in published.events] == [
        "scheduler_state_changed",
        "checkpoint_missing",
    ]
    assert "durable_log_missing" not in {
        event.code for event in published.events
    }


def test_scheduler_query_timeout_is_an_event_and_is_deduplicated(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    scheduler = FailingScheduler(
        monitor_module, {"job-a": "PENDING"}, failing_command="squeue"
    )
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    failed = monitor.poll({})
    repeated = monitor.poll(failed.state)

    assert [event.code for event in failed.events] == ["scheduler_query_failed"]
    assert failed.events[0].details == {"command": "squeue", "reason": "TimeoutError"}
    assert repeated.events == ()


def test_scheduler_success_without_expected_job_is_an_anomaly(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    scheduler = FakeScheduler(monitor_module, {})
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    missing = monitor.poll({})
    repeated = monitor.poll(missing.state)

    assert [event.code for event in missing.events] == ["scheduler_state_missing"]
    assert repeated.events == ()


def test_statvfs_failure_is_an_event_and_is_deduplicated(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    scheduler = FakeScheduler(monitor_module, {"job-a": "PENDING"})

    def failed_statvfs(_path):
        raise OSError("unavailable")

    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=FakeClock(),
        statvfs=failed_statvfs,
    )

    failed = monitor.poll({})
    repeated = monitor.poll(failed.state)

    assert [event.code for event in failed.events] == ["disk_query_failed"]
    assert repeated.events == ()


def _write_valid_checkpoint_and_report(spec) -> None:
    checkpoint = spec.jobs[0].checkpoint_path
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("checkpoint/data.pkl", b"checkpoint-payload")
    payload = checkpoint.read_bytes()
    spec.jobs[0].validation_report_path.write_text(
        json.dumps(
            {
                "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
                "checkpoint_size": len(payload),
            }
        )
    )


def _directory_spec(module, tmp_path: Path, *, validation_required: bool):
    spec = _healthy_spec(module, tmp_path)
    stage_dir = tmp_path / "stage1"
    stage_dir.mkdir()
    report_path = stage_dir / "validation-report.json"
    return replace(
        spec,
        jobs=(
            replace(
                spec.jobs[0],
                checkpoint_path=None,
                checkpoint_dir=stage_dir,
                validation_report_path=(
                    report_path if validation_required else None
                ),
                validation_report_required=validation_required,
                gpu_sample_format="none",
            ),
        ),
    )


def _write_zip_checkpoint(path: Path, payload: bytes = b"checkpoint-payload") -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("checkpoint/data.pkl", payload)


def _write_report(path: Path, checkpoint: Path) -> None:
    payload = checkpoint.read_bytes()
    path.write_text(
        json.dumps(
            {
                "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
                "checkpoint_size": len(payload),
            }
        )
    )


def _wrapped_manifest(kind: str, payload: dict) -> dict:
    body = {"schema_version": 1, "kind": kind, "payload": payload}
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return {**body, "sha256": hashlib.sha256(encoded).hexdigest()}


def _formal_directory_spec(module, tmp_path: Path):
    spec = _directory_spec(module, tmp_path, validation_required=True)
    job = spec.jobs[0]
    assert job.checkpoint_dir is not None
    digest = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in (
            "source",
            "environment",
            "prior",
            "architecture",
            "optimizer",
            "seed",
            "treatment",
            "scientific",
            "cohort",
            "arm",
            "provenance",
        )
    }
    report_path = job.checkpoint_dir / "finalized-checkpoint.json"
    ledger_entry = {
        "arm": "rope",
        "stage": "stage1",
        "terminal_step": 500_000,
        "upstream_identity": "study-a:rope:stage1",
        "artifact_identity": "rope-stage1-final",
        "checkpoint_relpath": "stage1/step-500000.ckpt",
        "finalized_manifest_relpath": "stage1/finalized-checkpoint.json",
        "np_seed": 42,
        "torch_seed": 42,
        "identity_rng_seed": 42,
        "world_size": 1,
        "cuda_device_count": 1,
        "max_checkpoint_bytes": job.max_checkpoint_bytes,
        "source_sha256": digest["source"],
        "environment_sha256": digest["environment"],
        "prior_sha256": digest["prior"],
        "architecture_sha256": digest["architecture"],
        "optimizer_sha256": digest["optimizer"],
        "scientific_sha256": digest["scientific"],
        "cohort_protocol_sha256": digest["cohort"],
        "arm_protocol_sha256": digest["arm"],
    }
    finalized_payload = {
        "study_id": "study-a",
        "arm": "rope",
        "stage": "stage1",
        "terminal_step": 500_000,
        "upstream_identity": "study-a:rope:stage1",
        "artifact_identity": "rope-stage1-final",
        "provenance_sha256": digest["provenance"],
        "source_sha256": digest["source"],
        "environment_sha256": digest["environment"],
        "prior_sha256": digest["prior"],
        "architecture_sha256": digest["architecture"],
        "optimizer_sha256": digest["optimizer"],
        "seed_sha256": digest["seed"],
        "treatment_sha256": digest["treatment"],
        "scientific_sha256": digest["scientific"],
        "cohort_protocol_sha256": digest["cohort"],
        "arm_protocol_sha256": digest["arm"],
        "cuda_device_count": 1,
        "max_checkpoint_bytes": job.max_checkpoint_bytes,
    }
    ledger = _wrapped_manifest(
        "transaction_ledger",
        {"study_id": "study-a", "entries": [ledger_entry]},
    )
    ledger_path = tmp_path / "transaction-ledger.json"
    ledger_path.write_text(json.dumps(ledger))
    formal = module.FormalValidationSpec(
        transaction_ledger_path=ledger_path,
        transaction_ledger_sha256=ledger["sha256"],
        artifact_root=tmp_path,
        ledger_entry=ledger_entry,
        finalized_payload=finalized_payload,
    )
    return replace(
        spec,
        jobs=(
            replace(
                job,
                validation_report_path=report_path,
                formal_validation=formal,
            ),
        ),
    )


def _write_formal_report(spec, checkpoint: Path, **updates) -> None:
    formal = spec.jobs[0].formal_validation
    assert formal is not None
    payload = dict(formal.finalized_payload)
    payload.update(
        {
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "checkpoint_size": checkpoint.stat().st_size,
        }
    )
    payload.update(updates)
    spec.jobs[0].validation_report_path.write_text(
        json.dumps(_wrapped_manifest("finalized_checkpoint", payload))
    )


def test_checkpoint_directory_selects_numeric_latest_and_detects_same_tick_replace(
    monitor_module, tmp_path
) -> None:
    spec = _directory_spec(
        monitor_module, tmp_path, validation_required=False
    )
    stage_dir = spec.jobs[0].checkpoint_dir
    assert stage_dir is not None
    _write_zip_checkpoint(stage_dir / "step-2.ckpt", b"older")
    latest = stage_dir / "step-10.ckpt"
    _write_zip_checkpoint(latest, b"latest")
    _write_zip_checkpoint(stage_dir / "step-010.ckpt", b"noncanonical")
    _write_zip_checkpoint(stage_dir / "final.ckpt", b"not-a-step")
    calls = []

    def checker(checkpoint_path, identity, max_checkpoint_bytes):
        calls.append((checkpoint_path, tuple(identity)))
        return monitor_module.inspect_changed_checkpoint(
            checkpoint_path, identity, max_checkpoint_bytes
        )

    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=FakeScheduler(monitor_module, {"job-a": "RUNNING"}),
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
        checkpoint_checker=checker,
    )

    first = monitor.poll({})
    unchanged = monitor.poll(first.state)
    assert first.events == ()
    assert unchanged.events == ()
    assert [path.name for path, _identity in calls] == ["step-10.ckpt"]
    assert first.state["jobs"]["job-a"]["checkpoint_step"] == 10

    before = latest.stat()
    replacement = stage_dir / ".replacement.ckpt"
    replacement.write_bytes(latest.read_bytes())
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    replacement.replace(latest)
    after = latest.stat()
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_ino != before.st_ino

    replaced = monitor.poll(unchanged.state)
    assert replaced.events == ()
    assert len(calls) == 2

    _write_zip_checkpoint(stage_dir / "step-11.ckpt", b"new-latest")
    advanced = monitor.poll(replaced.state)
    assert advanced.events == ()
    assert calls[-1][0].name == "step-11.ckpt"
    assert advanced.state["jobs"]["job-a"]["checkpoint_step"] == 11


def test_pilot_checkpoint_directory_explicitly_allows_no_validation_report(
    monitor_module, tmp_path
) -> None:
    spec = _directory_spec(
        monitor_module, tmp_path, validation_required=False
    )
    checkpoint = spec.jobs[0].checkpoint_dir / "step-500000.ckpt"
    _write_zip_checkpoint(checkpoint)
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=FakeScheduler(monitor_module, {"job-a": "COMPLETED"}),
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    observed = monitor.poll({})

    assert observed.events == ()
    assert "validation_report_identity" not in observed.state["jobs"]["job-a"]


def test_formal_checkpoint_directory_requires_terminal_validation_report(
    monitor_module, tmp_path
) -> None:
    spec = _formal_directory_spec(monitor_module, tmp_path)
    checkpoint = spec.jobs[0].checkpoint_dir / "step-500000.ckpt"
    _write_zip_checkpoint(checkpoint)
    scheduler = FakeScheduler(monitor_module, {"job-a": "COMPLETED"})
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    missing = monitor.poll({})
    assert [event.code for event in missing.events] == [
        "checkpoint_validation_missing"
    ]

    _write_formal_report(spec, checkpoint)
    supplied = monitor.poll(missing.state)
    assert supplied.events == ()


@pytest.mark.parametrize(
    ("updates", "kind"),
    [
        ({"source_sha256": "f" * 64}, "finalized_checkpoint"),
        ({"arm": "none"}, "finalized_checkpoint"),
        ({"stage": "stage2"}, "finalized_checkpoint"),
        ({"terminal_step": 40_000}, "finalized_checkpoint"),
        ({}, "strict_checkpoint_validation"),
    ],
)
def test_formal_report_rejects_self_consistent_identity_or_schema_drift(
    monitor_module, tmp_path, updates, kind
) -> None:
    spec = _formal_directory_spec(monitor_module, tmp_path)
    checkpoint = spec.jobs[0].checkpoint_dir / "step-500000.ckpt"
    _write_zip_checkpoint(checkpoint)
    formal = spec.jobs[0].formal_validation
    payload = dict(formal.finalized_payload)
    payload.update(
        {
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "checkpoint_size": checkpoint.stat().st_size,
        }
    )
    payload.update(updates)
    spec.jobs[0].validation_report_path.write_text(
        json.dumps(_wrapped_manifest(kind, payload))
    )
    facts = monitor_module.CheckpointFacts(
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(), checkpoint.stat().st_size
    )

    issues = monitor_module.validate_checkpoint_report(
        spec.jobs[0].validation_report_path,
        facts,
        job=spec.jobs[0],
        checkpoint_path=checkpoint,
    )

    assert [issue.code for issue in issues] == ["checkpoint_validation_invalid"]


def test_formal_report_rejects_minimal_digest_size_manifest(
    monitor_module, tmp_path
) -> None:
    spec = _formal_directory_spec(monitor_module, tmp_path)
    checkpoint = spec.jobs[0].checkpoint_dir / "step-500000.ckpt"
    _write_zip_checkpoint(checkpoint)
    payload = {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_size": checkpoint.stat().st_size,
    }
    spec.jobs[0].validation_report_path.write_text(
        json.dumps(_wrapped_manifest("finalized_checkpoint", payload))
    )
    facts = monitor_module.CheckpointFacts(
        payload["checkpoint_sha256"], payload["checkpoint_size"]
    )

    issues = monitor_module.validate_checkpoint_report(
        spec.jobs[0].validation_report_path,
        facts,
        job=spec.jobs[0],
        checkpoint_path=checkpoint,
    )

    assert [issue.code for issue in issues] == ["checkpoint_validation_invalid"]


@pytest.mark.parametrize("mutation", ["ledger_digest", "duplicate_entry", "artifact_path"])
def test_formal_report_rejects_ledger_or_artifact_binding_drift(
    monitor_module, tmp_path, mutation
) -> None:
    spec = _formal_directory_spec(monitor_module, tmp_path)
    checkpoint = spec.jobs[0].checkpoint_dir / "step-500000.ckpt"
    _write_zip_checkpoint(checkpoint)
    _write_formal_report(spec, checkpoint)
    formal = spec.jobs[0].formal_validation
    assert formal is not None
    if mutation == "ledger_digest":
        ledger = json.loads(formal.transaction_ledger_path.read_text())
        ledger["payload"]["entries"][0]["source_sha256"] = "e" * 64
        formal.transaction_ledger_path.write_text(
            json.dumps(_wrapped_manifest("transaction_ledger", ledger["payload"]))
        )
    elif mutation == "duplicate_entry":
        ledger = json.loads(formal.transaction_ledger_path.read_text())
        ledger["payload"]["entries"].append(dict(formal.ledger_entry))
        rewritten = _wrapped_manifest("transaction_ledger", ledger["payload"])
        formal.transaction_ledger_path.write_text(json.dumps(rewritten))
        spec = replace(
            spec,
            jobs=(
                replace(
                    spec.jobs[0],
                    formal_validation=replace(
                        formal, transaction_ledger_sha256=rewritten["sha256"]
                    ),
                ),
            ),
        )
    else:
        wrong_entry = dict(formal.ledger_entry)
        wrong_entry["checkpoint_relpath"] = "stage1/other.ckpt"
        ledger = _wrapped_manifest(
            "transaction_ledger",
            {"study_id": "study-a", "entries": [wrong_entry]},
        )
        formal.transaction_ledger_path.write_text(json.dumps(ledger))
        spec = replace(
            spec,
            jobs=(
                replace(
                    spec.jobs[0],
                    formal_validation=replace(
                        formal,
                        transaction_ledger_sha256=ledger["sha256"],
                        ledger_entry=wrong_entry,
                    ),
                ),
            ),
        )
    facts = monitor_module.CheckpointFacts(
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(), checkpoint.stat().st_size
    )

    issues = monitor_module.validate_checkpoint_report(
        spec.jobs[0].validation_report_path,
        facts,
        job=spec.jobs[0],
        checkpoint_path=checkpoint,
    )

    assert [issue.code for issue in issues] == ["checkpoint_validation_invalid"]


def test_formal_monitor_detects_ledger_change_without_report_or_checkpoint_change(
    monitor_module, tmp_path
) -> None:
    spec = _formal_directory_spec(monitor_module, tmp_path)
    checkpoint = spec.jobs[0].checkpoint_dir / "step-500000.ckpt"
    _write_zip_checkpoint(checkpoint)
    _write_formal_report(spec, checkpoint)
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=FakeScheduler(monitor_module, {"job-a": "RUNNING"}),
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )
    healthy = monitor.poll({})
    assert healthy.events == ()
    spec.jobs[0].formal_validation.transaction_ledger_path.unlink()

    changed = monitor.poll(healthy.state)

    assert [event.code for event in changed.events] == [
        "checkpoint_validation_invalid"
    ]
    assert changed.events[0].details["reason"].startswith("TransactionLedger")


@pytest.mark.parametrize("symlink_kind", ["directory", "latest_checkpoint"])
def test_checkpoint_directory_audit_rejects_symlinks(
    monitor_module, tmp_path, symlink_kind
) -> None:
    spec = _directory_spec(
        monitor_module, tmp_path, validation_required=False
    )
    stage_dir = spec.jobs[0].checkpoint_dir
    outside = tmp_path / "outside.ckpt"
    _write_zip_checkpoint(outside)
    if symlink_kind == "latest_checkpoint":
        (stage_dir / "step-10.ckpt").symlink_to(outside)
    else:
        real_dir = tmp_path / "real-stage"
        stage_dir.rename(real_dir)
        stage_dir.symlink_to(real_dir, target_is_directory=True)
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=FakeScheduler(monitor_module, {"job-a": "RUNNING"}),
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    result = monitor.poll({})

    assert [event.code for event in result.events] == ["checkpoint_read_failed"]


def test_checkpoint_directory_inspection_fails_closed_on_selection_open_race(
    monitor_module, tmp_path
) -> None:
    spec = _directory_spec(
        monitor_module, tmp_path, validation_required=False
    )
    checkpoint = spec.jobs[0].checkpoint_dir / "step-10.ckpt"
    _write_zip_checkpoint(checkpoint, b"original")

    def racing_checker(checkpoint_path, identity, max_checkpoint_bytes):
        replacement = checkpoint_path.with_name(".racing.ckpt")
        _write_zip_checkpoint(replacement, b"replacement")
        replacement.replace(checkpoint_path)
        return monitor_module.inspect_changed_checkpoint(
            checkpoint_path, identity, max_checkpoint_bytes
        )

    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=FakeScheduler(monitor_module, {"job-a": "RUNNING"}),
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
        checkpoint_checker=racing_checker,
    )

    result = monitor.poll({})

    assert [event.code for event in result.events] == [
        "checkpoint_changed_during_observation"
    ]


def test_checkpoint_directory_enforces_byte_ceiling_before_zip_or_hash(
    monitor_module, tmp_path
) -> None:
    spec = _directory_spec(
        monitor_module, tmp_path, validation_required=False
    )
    checkpoint = spec.jobs[0].checkpoint_dir / "step-10.ckpt"
    _write_zip_checkpoint(checkpoint)
    spec = replace(
        spec,
        jobs=(
            replace(
                spec.jobs[0], max_checkpoint_bytes=checkpoint.stat().st_size - 1
            ),
        ),
    )
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=FakeScheduler(monitor_module, {"job-a": "RUNNING"}),
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    result = monitor.poll({})

    assert [event.code for event in result.events] == ["checkpoint_too_large"]


def test_checkpoint_integrity_and_strict_report_run_only_when_identity_changes(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    _write_valid_checkpoint_and_report(spec)
    scheduler = FakeScheduler(monitor_module, {"job-a": "RUNNING"})
    calls = []

    def checker(checkpoint_path, identity, max_checkpoint_bytes):
        calls.append(
            (
                checkpoint_path,
                tuple(identity),
                max_checkpoint_bytes,
            )
        )
        return monitor_module.inspect_changed_checkpoint(
            checkpoint_path, identity, max_checkpoint_bytes
        )

    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
        checkpoint_checker=checker,
    )

    first = monitor.poll({})
    second = monitor.poll(first.state)
    stat_before = spec.jobs[0].checkpoint_path.stat()
    replacement = spec.jobs[0].checkpoint_path.with_suffix(".replacement")
    replacement.write_bytes(spec.jobs[0].checkpoint_path.read_bytes())
    os.utime(
        replacement,
        ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns),
    )
    replacement.replace(spec.jobs[0].checkpoint_path)
    stat_after = spec.jobs[0].checkpoint_path.stat()
    assert stat_after.st_size == stat_before.st_size
    assert stat_after.st_mtime_ns == stat_before.st_mtime_ns
    assert stat_after.st_ino != stat_before.st_ino
    changed = monitor.poll(second.state)

    assert first.events == ()
    assert second.events == ()
    assert changed.events == ()
    assert len(calls) == 2
    assert calls[0][1][0] == "file-v2"
    assert calls[0][1][-2:] == (stat_before.st_mtime_ns, stat_before.st_size)
    assert calls[0][2] == 1 << 20


def test_new_monitor_process_revalidates_checkpoint_despite_persisted_identity(
    monitor_module, tmp_path
) -> None:
    spec = replace(
        _healthy_spec(monitor_module, tmp_path), manifest_sha256="a" * 64
    )
    _write_valid_checkpoint_and_report(spec)
    scheduler = FakeScheduler(monitor_module, {"job-a": "RUNNING"})
    inspections = []

    def inspector(checkpoint_path, identity, max_checkpoint_bytes):
        inspections.append(tuple(identity))
        return monitor_module.inspect_changed_checkpoint(
            checkpoint_path, identity, max_checkpoint_bytes
        )

    first_monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
        checkpoint_checker=inspector,
    )
    first = first_monitor.poll({})
    assert first.events == ()
    assert len(inspections) == 1

    restarted_monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
        checkpoint_checker=inspector,
    )
    restarted = restarted_monitor.poll(first.state)

    assert restarted.events == ()
    assert len(inspections) == 2


def test_new_monitor_process_never_trusts_fabricated_cached_checkpoint_facts(
    monitor_module, tmp_path
) -> None:
    spec = replace(
        _healthy_spec(monitor_module, tmp_path), manifest_sha256="b" * 64
    )
    _write_valid_checkpoint_and_report(spec)
    identity = monitor_module._regular_file_identity(spec.jobs[0].checkpoint_path)
    state = {
        "schema_version": 1,
        "monitor_manifest_sha256": spec.manifest_sha256,
        "jobs": {
            "job-a": {
                "scheduler_state": "RUNNING",
                "checkpoint_inspected_identity": identity,
                "checkpoint_facts": {
                    "checkpoint_sha256": "0" * 64,
                    "checkpoint_size": 1,
                },
            }
        },
    }
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=FakeScheduler(monitor_module, {"job-a": "RUNNING"}),
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    observed = monitor.poll(state)

    facts = observed.state["jobs"]["job-a"]["checkpoint_facts"]
    assert facts["checkpoint_sha256"] == hashlib.sha256(
        spec.jobs[0].checkpoint_path.read_bytes()
    ).hexdigest()
    assert facts["checkpoint_size"] == spec.jobs[0].checkpoint_path.stat().st_size
    assert observed.events == ()


def test_new_monitor_process_clears_cached_facts_when_revalidation_cannot_open(
    monitor_module, tmp_path
) -> None:
    spec = replace(
        _healthy_spec(monitor_module, tmp_path), manifest_sha256="e" * 64
    )
    _write_valid_checkpoint_and_report(spec)
    checkpoint = spec.jobs[0].checkpoint_path
    outside = tmp_path / "outside.ckpt"
    checkpoint.replace(outside)
    checkpoint.symlink_to(outside)
    state = {
        "schema_version": 1,
        "monitor_manifest_sha256": spec.manifest_sha256,
        "jobs": {
            "job-a": {
                "scheduler_state": "RUNNING",
                "checkpoint_facts": {
                    "checkpoint_sha256": "0" * 64,
                    "checkpoint_size": 1,
                },
            }
        },
    }
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=FakeScheduler(monitor_module, {"job-a": "RUNNING"}),
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    observed = monitor.poll(state)

    assert [event.code for event in observed.events] == ["checkpoint_read_failed"]
    assert "checkpoint_facts" not in observed.state["jobs"]["job-a"]


def test_monitor_manifest_digest_change_discards_all_persisted_job_facts(
    monitor_module, tmp_path
) -> None:
    spec = replace(
        _healthy_spec(monitor_module, tmp_path), manifest_sha256="c" * 64
    )
    _write_valid_checkpoint_and_report(spec)
    state = {
        "schema_version": 1,
        "monitor_manifest_sha256": "d" * 64,
        "jobs": {
            "job-a": {
                "scheduler_state": "PENDING",
                "attacker_fact": "must-not-survive",
                "last_step": 999_999,
            }
        },
    }
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=FakeScheduler(monitor_module, {"job-a": "PENDING"}),
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    observed = monitor.poll(state)

    job_state = observed.state["jobs"]["job-a"]
    assert "attacker_fact" not in job_state
    assert job_state["last_step"] == 1
    assert observed.state["monitor_manifest_sha256"] == "c" * 64
    assert observed.events == ()


@pytest.mark.parametrize(
    ("corruption", "expected_code"),
    [
        ("zip", "checkpoint_zip_invalid"),
        ("report", "checkpoint_validation_mismatch"),
        ("missing_report", "checkpoint_validation_missing"),
    ],
)
def test_checkpoint_problems_are_events(
    monitor_module, tmp_path, corruption, expected_code
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    _write_valid_checkpoint_and_report(spec)
    if corruption == "zip":
        spec.jobs[0].checkpoint_path.write_bytes(b"not a checkpoint zip")
    elif corruption == "report":
        report = json.loads(spec.jobs[0].validation_report_path.read_text())
        report["checkpoint_sha256"] = "0" * 64
        spec.jobs[0].validation_report_path.write_text(json.dumps(report))
    else:
        spec.jobs[0].validation_report_path.unlink()
    scheduler = FakeScheduler(monitor_module, {"job-a": "RUNNING"})
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    observed = monitor.poll({})
    unchanged = monitor.poll(observed.state)

    assert expected_code in {event.code for event in observed.events}
    assert all(event.category == "checkpoint_issue" for event in observed.events)
    assert unchanged.events == ()


def test_checkpoint_byte_ceiling_is_enforced_before_archive_validation(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    _write_valid_checkpoint_and_report(spec)
    size = spec.jobs[0].checkpoint_path.stat().st_size
    spec = replace(
        spec,
        jobs=(replace(spec.jobs[0], max_checkpoint_bytes=size - 1),),
    )
    scheduler = FakeScheduler(monitor_module, {"job-a": "RUNNING"})
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    result = monitor.poll({})

    assert [event.code for event in result.events] == ["checkpoint_too_large"]


def test_failed_checkpoint_inspection_is_not_repeated_without_identity_change(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    _write_valid_checkpoint_and_report(spec)
    size = spec.jobs[0].checkpoint_path.stat().st_size
    spec = replace(
        spec,
        jobs=(replace(spec.jobs[0], max_checkpoint_bytes=size - 1),),
    )
    inspections = []

    def inspector(checkpoint_path, identity, max_checkpoint_bytes):
        inspections.append(tuple(identity))
        return monitor_module.inspect_changed_checkpoint(
            checkpoint_path, identity, max_checkpoint_bytes
        )

    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=FakeScheduler(monitor_module, {"job-a": "RUNNING"}),
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
        checkpoint_checker=inspector,
    )

    first = monitor.poll({})
    repeated = monitor.poll(first.state)

    assert [event.code for event in first.events] == ["checkpoint_too_large"]
    assert repeated.events == ()
    assert len(inspections) == 1


@pytest.mark.parametrize(
    ("late_report_kind", "expected_codes"),
    [
        ("good", []),
        ("mismatch", ["checkpoint_validation_mismatch"]),
        ("malformed", ["checkpoint_validation_invalid"]),
    ],
)
def test_late_validation_report_is_checked_without_reinspecting_checkpoint(
    monitor_module, tmp_path, late_report_kind, expected_codes
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    _write_valid_checkpoint_and_report(spec)
    good_report = spec.jobs[0].validation_report_path.read_text()
    spec.jobs[0].validation_report_path.unlink()
    scheduler = FakeScheduler(monitor_module, {"job-a": "RUNNING"})
    inspections = []

    def inspector(checkpoint_path, identity, max_checkpoint_bytes):
        inspections.append(tuple(identity))
        return monitor_module.inspect_changed_checkpoint(
            checkpoint_path, identity, max_checkpoint_bytes
        )

    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
        checkpoint_checker=inspector,
    )
    checkpoint_first = monitor.poll({})
    assert [event.code for event in checkpoint_first.events] == [
        "checkpoint_validation_missing"
    ]

    if late_report_kind == "good":
        late_report = good_report
    elif late_report_kind == "mismatch":
        report = json.loads(good_report)
        report["checkpoint_sha256"] = "0" * 64
        late_report = json.dumps(report)
    else:
        late_report = "{malformed"
    spec.jobs[0].validation_report_path.write_text(late_report)

    late = monitor.poll(checkpoint_first.state)
    unchanged = monitor.poll(late.state)

    assert [event.code for event in late.events] == expected_codes
    assert unchanged.events == ()
    assert len(inspections) == 1


def test_validation_report_disappearance_alerts_without_checkpoint_reinspection(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    _write_valid_checkpoint_and_report(spec)
    scheduler = FakeScheduler(monitor_module, {"job-a": "RUNNING"})
    inspections = []

    def inspector(checkpoint_path, identity, max_checkpoint_bytes):
        inspections.append(tuple(identity))
        return monitor_module.inspect_changed_checkpoint(
            checkpoint_path, identity, max_checkpoint_bytes
        )

    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
        checkpoint_checker=inspector,
    )
    healthy = monitor.poll({})
    spec.jobs[0].validation_report_path.unlink()

    missing = monitor.poll(healthy.state)
    unchanged = monitor.poll(missing.state)

    assert [event.code for event in missing.events] == [
        "checkpoint_validation_missing"
    ]
    assert unchanged.events == ()
    assert len(inspections) == 1


@pytest.mark.parametrize(
    "raw_report",
    [
        '{"checkpoint_sha256":"%s","checkpoint_sha256":"%s","checkpoint_size":1}',
        '{"checkpoint_sha256":"%s","checkpoint_size":NaN}',
    ],
)
def test_validation_report_rejects_duplicate_keys_and_nonfinite_json(
    monitor_module, tmp_path, raw_report
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    _write_valid_checkpoint_and_report(spec)
    digest = hashlib.sha256(spec.jobs[0].checkpoint_path.read_bytes()).hexdigest()
    substitutions = (digest, digest) if raw_report.count("%s") == 2 else digest
    spec.jobs[0].validation_report_path.write_text(raw_report % substitutions)
    scheduler = FakeScheduler(monitor_module, {"job-a": "RUNNING"})
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    result = monitor.poll({})

    assert "checkpoint_validation_invalid" in {event.code for event in result.events}


def test_completed_job_with_no_checkpoint_and_stale_running_checkpoint_alert_once(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    spec = replace(spec, jobs=(replace(spec.jobs[0], stall_seconds=1_000.0),))
    scheduler = FakeScheduler(monitor_module, {"job-a": "COMPLETED"})
    clock = FakeClock()
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=clock,
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )
    missing = monitor.poll({})
    assert [event.code for event in missing.events] == ["checkpoint_missing"]
    assert monitor.poll(missing.state).events == ()

    _write_valid_checkpoint_and_report(spec)
    scheduler.states["job-a"] = "RUNNING"
    created = monitor.poll(missing.state)
    assert [event.code for event in created.events] == ["scheduler_state_changed"]
    clock.value += 120.0
    stale = monitor.poll(created.state)
    assert [event.code for event in stale.events] == ["checkpoint_stalled"]
    assert monitor.poll(stale.state).events == ()


def _write_gpu_jsonl(path: Path, module, values: list[float], *, include_end=True) -> None:
    records = [
        {
            "kind": "training_start",
            "monotonic_seconds": 99.5,
        },
        *[
            {
                "kind": "sample",
                "monotonic_seconds": 100.0 + index,
                "gpu_uuid": "GPU-a",
                "utilization_percent": value,
            }
            for index, value in enumerate(values)
        ],
    ]
    if include_end:
        records.append({"kind": "training_end", "monotonic_seconds": 110.0})
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def _pilot_gpu_csv_text(values: list[float]) -> str:
    start = datetime(2026, 7, 22, 5, 53, 29, 871000)
    return "".join(
        f"{(start + timedelta(seconds=30 * index)).strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]}, "
        f"0, NVIDIA H100 80GB HBM3, {value}, 63801, 81559, 474.28, 700.00\n"
        for index, value in enumerate(values)
    )


def test_pilot_gpu_csv_uses_explicit_fourth_column_and_thirty_second_cadence(
    monitor_module, tmp_path
) -> None:
    text = _pilot_gpu_csv_text([0.0, 0.0, 0.0] + [84.0] * 40)
    evaluation, invalid = monitor_module.evaluate_pilot_gpu_csv(
        text, expected_gpu_count=1
    )

    assert invalid == 0
    assert evaluation.complete is True
    assert evaluation.issues == ()
    assert evaluation.sample_counts == {"GPU-index-0": 28}
    assert evaluation.means == {"GPU-index-0": 84.0}

    spec = _healthy_spec(monitor_module, tmp_path, utilization_required=True)
    spec.jobs[0].gpu_samples_path.write_text(text)
    spec = replace(
        spec,
        jobs=(
            replace(
                spec.jobs[0],
                expected_gpu_uuids=(),
                gpu_sample_format="pilot_nvidia_smi_csv_v1",
            ),
        ),
    )
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=FakeScheduler(monitor_module, {"job-a": "RUNNING"}),
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    observed = monitor.poll({})

    assert observed.events == ()
    assert observed.state["jobs"]["job-a"]["gpu_means"] == {
        "GPU-index-0": 84.0
    }


def test_pilot_gpu_csv_cannot_be_silently_parsed_as_formal_active_window(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path, utilization_required=True)
    spec.jobs[0].gpu_samples_path.write_text(
        _pilot_gpu_csv_text([90.0] * 40)
    )
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=FakeScheduler(monitor_module, {"job-a": "RUNNING"}),
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    observed = monitor.poll({})

    assert [event.code for event in observed.events] == ["gpu_sample_invalid"]


def test_monitor_enforces_gpu_gate_only_for_utilization_marked_jobs(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path, utilization_required=True)
    _write_gpu_jsonl(spec.jobs[0].gpu_samples_path, monitor_module, [79.0] * 10)
    scheduler = FakeScheduler(monitor_module, {"job-a": "RUNNING"})
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    low = monitor.poll({})
    assert [event.code for event in low.events] == ["gpu_utilization_low"]
    assert monitor.poll(low.state).events == ()

    functional_spec = replace(
        spec,
        jobs=(
            replace(
                spec.jobs[0],
                utilization_required=False,
                expected_gpu_uuids=(),
                expected_gpu_count=0,
            ),
        ),
    )
    functional_monitor = monitor_module.FormalIdentityMonitor(
        functional_spec,
        command_runner=scheduler,
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )
    functional = functional_monitor.poll({})
    assert all(not event.code.startswith("gpu_") for event in functional.events)


def test_incomplete_gpu_window_is_silent_while_active_but_fails_when_complete(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path, utilization_required=True)
    _write_valid_checkpoint_and_report(spec)
    _write_gpu_jsonl(
        spec.jobs[0].gpu_samples_path,
        monitor_module,
        [90.0] * 10,
        include_end=False,
    )
    scheduler = FakeScheduler(monitor_module, {"job-a": "RUNNING"})
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    active = monitor.poll({})
    assert active.events == ()
    scheduler.states["job-a"] = "COMPLETED"
    complete = monitor.poll(active.state)
    assert [event.code for event in complete.events] == [
        "scheduler_state_changed",
        "gpu_window_incomplete",
    ]
    assert monitor.poll(complete.state).events == ()


def _with_scheduler_paths(spec, tmp_path: Path):
    bin_dir = tmp_path / "scheduler-bin"
    bin_dir.mkdir(exist_ok=True)
    paths = []
    for name in ("squeue", "sacct"):
        path = bin_dir / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o700)
        paths.append(path)
    return replace(
        spec,
        squeue_path=paths[0],
        sacct_path=paths[1],
        squeue_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        sacct_sha256=hashlib.sha256(paths[1].read_bytes()).hexdigest(),
    )


def _write_monitor_manifest(path: Path, spec, *, schema_version: int = 1) -> None:
    jobs = []
    for job in spec.jobs:
        payload = {
                "job_id": job.job_id,
                "arm": job.arm,
                "stage": job.stage,
                "log_path": str(job.log_path),
                "validation_report_path": (
                    str(job.validation_report_path)
                    if job.validation_report_path is not None
                    else None
                ),
                "gpu_samples_path": (
                    str(job.gpu_samples_path)
                    if job.gpu_samples_path is not None
                    else None
                ),
                "utilization_required": job.utilization_required,
                "stall_seconds": job.stall_seconds,
                "checkpoint_stale_seconds": job.checkpoint_stale_seconds,
                "max_checkpoint_bytes": job.max_checkpoint_bytes,
                "expected_gpu_uuids": list(job.expected_gpu_uuids),
                "expected_gpu_count": job.expected_gpu_count,
        }
        if schema_version == 1:
            payload["checkpoint_path"] = str(job.checkpoint_path)
        else:
            formal_validation = None
            if job.formal_validation is not None:
                formal_validation = {
                    "transaction_ledger_path": str(
                        job.formal_validation.transaction_ledger_path
                    ),
                    "transaction_ledger_sha256": (
                        job.formal_validation.transaction_ledger_sha256
                    ),
                    "artifact_root": str(job.formal_validation.artifact_root),
                    "ledger_entry": job.formal_validation.ledger_entry,
                    "finalized_payload": job.formal_validation.finalized_payload,
                }
            payload.update(
                {
                    "checkpoint_dir": str(job.checkpoint_dir),
                    "live_log_path": (
                        str(job.live_log_path)
                        if job.live_log_path is not None
                        else None
                    ),
                    "validation_report_required": job.validation_report_required,
                    "gpu_sample_format": job.gpu_sample_format,
                    "step_offset": job.step_offset,
                    "formal_validation": formal_validation,
                }
            )
        jobs.append(payload)
    root_payload = {
                "schema_version": schema_version,
                "disk_path": str(spec.disk_path),
                "max_event_ledger_bytes": spec.max_event_ledger_bytes,
                "jobs": jobs,
    }
    if schema_version == 2:
        root_payload.update(
            {
                "squeue_path": str(spec.squeue_path),
                "squeue_sha256": spec.squeue_sha256,
                "sacct_path": str(spec.sacct_path),
                "sacct_sha256": spec.sacct_sha256,
            }
        )
    path.write_text(json.dumps(root_payload))


def test_manifest_loader_is_strict_and_cli_defaults_to_thirty_minutes(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    manifest = tmp_path / "monitor.json"
    _write_monitor_manifest(manifest, spec)

    loaded = monitor_module.load_monitor_spec(manifest)
    args = monitor_module.build_parser().parse_args(
        [
            "--manifest",
            str(manifest),
            "--state-dir",
            str(tmp_path / "state"),
            "--event-ledger-dir",
            str(tmp_path / "events"),
            "--once",
        ]
    )

    assert replace(
        loaded, manifest_sha256=None, manifest_schema_version=2
    ) == spec
    assert args.duration_seconds == 1_800.0
    assert args.once is True

    payload = json.loads(manifest.read_text())
    payload["jobs"][0]["log_path"] = "relative.log"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="absolute"):
        monitor_module.load_monitor_spec(manifest)


def test_v2_manifest_explicitly_separates_pilot_and_formal_validation(
    monitor_module, tmp_path
) -> None:
    pilot = _directory_spec(
        monitor_module, tmp_path, validation_required=False
    )
    pilot = _with_scheduler_paths(pilot, tmp_path)
    manifest = tmp_path / "pilot-monitor.json"
    _write_monitor_manifest(manifest, pilot, schema_version=2)

    loaded = monitor_module.load_monitor_spec(manifest)
    assert replace(loaded, manifest_sha256=None) == pilot
    assert loaded.manifest_sha256 is not None

    payload = json.loads(manifest.read_text())
    payload["jobs"][0]["validation_report_required"] = True
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="validation_report_path"):
        monitor_module.load_monitor_spec(manifest)

    payload["jobs"][0]["validation_report_required"] = False
    payload["jobs"][0]["validation_report_path"] = str(
        tmp_path / "silently-ignored.json"
    )
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="must be null"):
        monitor_module.load_monitor_spec(manifest)


def test_v2_formal_manifest_loads_complete_ledger_and_finalized_expectations(
    monitor_module, tmp_path
) -> None:
    formal = _with_scheduler_paths(
        _formal_directory_spec(monitor_module, tmp_path), tmp_path
    )
    manifest = tmp_path / "formal-monitor.json"
    _write_monitor_manifest(manifest, formal, schema_version=2)

    loaded = monitor_module.load_monitor_spec(manifest)

    assert replace(loaded, manifest_sha256=None) == formal
    assert loaded.jobs[0].formal_validation is not None
    assert set(loaded.jobs[0].formal_validation.ledger_entry) == (
        monitor_module.LEDGER_ENTRY_KEYS
    )
    assert set(loaded.jobs[0].formal_validation.finalized_payload) == (
        monitor_module.FINALIZED_EXPECTED_PAYLOAD_KEYS
    )


def test_v2_manifest_requires_explicit_pilot_gpu_csv_format(
    monitor_module, tmp_path
) -> None:
    pilot = _directory_spec(
        monitor_module, tmp_path, validation_required=False
    )
    pilot = _with_scheduler_paths(pilot, tmp_path)
    gpu_path = tmp_path / "pilot-gpu.csv"
    gpu_path.write_text(_pilot_gpu_csv_text([90.0] * 40))
    pilot = replace(
        pilot,
        jobs=(
            replace(
                pilot.jobs[0],
                gpu_samples_path=gpu_path,
                utilization_required=True,
                expected_gpu_count=1,
                gpu_sample_format="pilot_nvidia_smi_csv_v1",
            ),
        ),
    )
    manifest = tmp_path / "pilot-gpu-monitor.json"
    _write_monitor_manifest(manifest, pilot, schema_version=2)

    loaded = monitor_module.load_monitor_spec(manifest)
    assert replace(loaded, manifest_sha256=None) == pilot
    assert loaded.jobs[0].gpu_sample_format == "pilot_nvidia_smi_csv_v1"

    payload = json.loads(manifest.read_text())
    del payload["jobs"][0]["gpu_sample_format"]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="keys mismatch"):
        monitor_module.load_monitor_spec(manifest)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("relative", "absolute"),
        ("digest", "SHA-256 mismatch"),
        ("non_executable", "executable regular file"),
        ("symlink", "no-follow executable"),
    ],
)
def test_v2_manifest_binds_scheduler_executable_path_and_sha256(
    monitor_module, tmp_path, mutation, match
) -> None:
    pilot = _with_scheduler_paths(
        _directory_spec(monitor_module, tmp_path, validation_required=False),
        tmp_path,
    )
    manifest = tmp_path / "scheduler-bound-monitor.json"
    _write_monitor_manifest(manifest, pilot, schema_version=2)
    payload = json.loads(manifest.read_text())
    if mutation == "relative":
        payload["squeue_path"] = "squeue"
    elif mutation == "digest":
        payload["squeue_sha256"] = "0" * 64
    elif mutation == "non_executable":
        pilot.squeue_path.chmod(0o600)
    else:
        real = pilot.squeue_path.with_name("real-squeue")
        pilot.squeue_path.rename(real)
        pilot.squeue_path.symlink_to(real)
    manifest.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=match):
        monitor_module.load_monitor_spec(manifest)


def test_scheduler_binary_is_rehashed_before_every_poll(
    monitor_module, tmp_path
) -> None:
    pilot = _with_scheduler_paths(
        _directory_spec(monitor_module, tmp_path, validation_required=False),
        tmp_path,
    )
    manifest = tmp_path / "scheduler-bound-monitor.json"
    _write_monitor_manifest(manifest, pilot, schema_version=2)
    loaded = monitor_module.load_monitor_spec(manifest)
    scheduler = FakeScheduler(monitor_module, {"job-a": "PENDING"})
    monitor = monitor_module.FormalIdentityMonitor(
        loaded,
        command_runner=scheduler,
        clock=FakeClock(),
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )

    first = monitor.poll({})
    assert first.events == ()
    loaded.squeue_path.write_text("#!/bin/sh\nexit 9\n")
    loaded.squeue_path.chmod(0o700)
    tampered = monitor.poll(first.state)

    assert "scheduler_query_failed" in {event.code for event in tampered.events}
    assert [Path(call[0]).name for call in scheduler.calls].count("squeue") == 1
    assert [Path(call[0]).name for call in scheduler.calls].count("sacct") == 2


def test_read_only_scheduler_runner_rejects_ambient_path_command(
    monitor_module,
) -> None:
    with pytest.raises(ValueError, match="read-only scheduler introspection"):
        monitor_module.run_read_only_command(
            ["squeue", "--noheader"], timeout_seconds=1.0
        )


def test_bounded_runner_writes_only_state_and_event_directories(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    artifact_before = spec.jobs[0].log_path.read_bytes()
    state_dir = tmp_path / "state"
    event_dir = tmp_path / "events"
    state_dir.mkdir()
    event_dir.mkdir()
    scheduler = FakeScheduler(monitor_module, {"job-a": "PENDING"})
    clock = FakeClock()
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=clock,
        statvfs=lambda _path: _statvfs_with_available(22 * (1 << 30)),
    )
    store = monitor_module.MonitorStore(state_dir, event_dir)
    output = io.StringIO()

    monitor_module.run_bounded_monitor(
        monitor,
        store,
        duration_seconds=1_800.0,
        poll_interval_seconds=30.0,
        once=True,
        output=output,
    )

    assert output.getvalue() == ""
    assert {path.name for path in state_dir.iterdir()} == {"state.json"}
    assert list(event_dir.iterdir()) == []
    assert spec.jobs[0].log_path.read_bytes() == artifact_before


def test_bounded_runner_emits_jsonl_and_stops_at_deadline(
    monitor_module, tmp_path
) -> None:
    spec = _healthy_spec(monitor_module, tmp_path)
    state_dir = tmp_path / "state"
    event_dir = tmp_path / "events"
    state_dir.mkdir()
    event_dir.mkdir()
    scheduler = FakeScheduler(monitor_module, {"job-a": "PENDING"})
    clock = FakeClock()
    monitor = monitor_module.FormalIdentityMonitor(
        spec,
        command_runner=scheduler,
        clock=clock,
        statvfs=lambda _path: _statvfs_with_available(20 * (1 << 30) - 1),
    )
    store = monitor_module.MonitorStore(state_dir, event_dir)
    output = io.StringIO()

    monitor_module.run_bounded_monitor(
        monitor,
        store,
        duration_seconds=90.0,
        poll_interval_seconds=30.0,
        once=False,
        output=output,
    )

    lines = output.getvalue().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["severity"] == "submit_block"
    assert (event_dir / "events.jsonl").read_text().splitlines() == lines
    assert len(scheduler.calls) == 6
    assert clock.monotonic() == 1_090.0


def test_monitor_lock_is_nonblocking(monitor_module, tmp_path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    with monitor_module.nonblocking_monitor_lock(state_dir) as acquired:
        assert acquired is True
        with monitor_module.nonblocking_monitor_lock(state_dir) as second:
            assert second is False


def test_event_ledger_ceiling_accepts_equality_and_rejects_one_byte_over(
    monitor_module, tmp_path
) -> None:
    state_dir = tmp_path / "state"
    event_dir = tmp_path / "events"
    state_dir.mkdir()
    event_dir.mkdir()
    payload = b'{"event":"one"}'
    store = monitor_module.MonitorStore(
        state_dir,
        event_dir,
        max_event_ledger_bytes=len(payload) + 1,
    )

    store.append_events([payload])
    with pytest.raises(ValueError, match="event ledger exceeds byte ceiling"):
        store.append_events([b"x"])


def test_event_ledger_uses_independent_single_link_lock_and_file(
    monitor_module, tmp_path
) -> None:
    state_dir = tmp_path / "state"
    event_dir = tmp_path / "events"
    state_dir.mkdir()
    event_dir.mkdir()
    store = monitor_module.MonitorStore(state_dir, event_dir)

    store.append_events([b'{"event":"one"}'])

    assert {path.name for path in event_dir.iterdir()} == {
        "events.lock",
        "events.jsonl",
    }
    assert (event_dir / "events.lock").stat().st_nlink == 1
    assert (event_dir / "events.jsonl").stat().st_nlink == 1


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("events.lock", "event ledger lock is invalid"),
        ("events.jsonl", "event ledger file is invalid"),
    ],
)
def test_event_ledger_rejects_hardlink_escape(
    monitor_module, tmp_path, name, match
) -> None:
    state_dir = tmp_path / "state"
    event_dir = tmp_path / "events"
    state_dir.mkdir()
    event_dir.mkdir()
    store = monitor_module.MonitorStore(state_dir, event_dir)
    store.append_events([b"first"])
    os.link(event_dir / name, tmp_path / f"outside-{name}")

    with pytest.raises(ValueError, match=match):
        store.append_events([b"second"])


def test_distinct_monitor_stores_serialize_event_appends_under_one_lock(
    monitor_module, tmp_path
) -> None:
    event_dir = tmp_path / "events"
    state_a = tmp_path / "state-a"
    state_b = tmp_path / "state-b"
    for path in (event_dir, state_a, state_b):
        path.mkdir()
    stores = [
        monitor_module.MonitorStore(state_a, event_dir),
        monitor_module.MonitorStore(state_b, event_dir),
    ]
    barrier = threading.Barrier(3)
    errors = []

    def append(store, payload):
        try:
            barrier.wait()
            store.append_events([payload])
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=append, args=(stores[0], b"store-a")),
        threading.Thread(target=append, args=(stores[1], b"store-b")),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5.0)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert sorted((event_dir / "events.jsonl").read_bytes().splitlines()) == [
        b"store-a",
        b"store-b",
    ]


def test_concurrent_event_appends_cannot_race_past_shared_ceiling(
    monitor_module, tmp_path
) -> None:
    event_dir = tmp_path / "events"
    state_a = tmp_path / "state-a"
    state_b = tmp_path / "state-b"
    for path in (event_dir, state_a, state_b):
        path.mkdir()
    stores = [
        monitor_module.MonitorStore(
            state_a, event_dir, max_event_ledger_bytes=4
        ),
        monitor_module.MonitorStore(
            state_b, event_dir, max_event_ledger_bytes=4
        ),
    ]
    barrier = threading.Barrier(3)
    outcomes = []

    def append(store):
        barrier.wait()
        try:
            store.append_events([b"abc"])
        except ValueError as error:
            outcomes.append(type(error).__name__)
        else:
            outcomes.append("ok")

    threads = [threading.Thread(target=append, args=(store,)) for store in stores]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5.0)

    assert sorted(outcomes) == ["ValueError", "ok"]
    assert (event_dir / "events.jsonl").read_bytes() == b"abc\n"


def test_write_scope_rejects_parent_symlink_alias_into_state(
    monitor_module, tmp_path
) -> None:
    state_dir = tmp_path / "state"
    event_dir = tmp_path / "events"
    state_dir.mkdir()
    event_dir.mkdir()
    alias = tmp_path / "state-alias"
    alias.symlink_to(state_dir, target_is_directory=True)
    spec = _healthy_spec(monitor_module, tmp_path)
    spec = replace(
        spec,
        jobs=(replace(spec.jobs[0], log_path=alias / "observed.log"),),
    )

    with pytest.raises(ValueError, match="outside monitor writable"):
        monitor_module._validate_write_scope(
            spec, tmp_path / "monitor.json", state_dir, event_dir
        )


def test_monitor_topology_rejects_three_device_false_positive(
    monitor_module,
) -> None:
    with pytest.raises(
        ValueError, match="log_path must use the monitored disk_path filesystem"
    ):
        monitor_module._require_monitor_device_topology(
            disk_device=11,
            artifact_devices=(("jobs[job-a].log_path", 22),),
            writable_devices=(
                ("monitor state directory", 33),
                ("monitor event-ledger directory", 33),
            ),
        )


def test_pending_artifacts_bind_nearest_same_device_physical_ancestor(
    monitor_module, tmp_path
) -> None:
    artifact_root = tmp_path / "artifacts"
    nearest = artifact_root / "reserved"
    nearest.mkdir(parents=True)
    spec = _healthy_spec(monitor_module, artifact_root)
    pending = nearest / "not-created" / "stage1"
    spec = replace(
        spec,
        jobs=(
            replace(
                spec.jobs[0],
                log_path=pending / "train.log",
                checkpoint_path=pending / "step-500000.ckpt",
                validation_report_path=pending / "validation.json",
            ),
        ),
    )

    binding = monitor_module._validate_monitored_artifact_filesystem(spec)

    assert binding.disk_device == artifact_root.stat().st_dev
    assert {item.label for item in binding.artifacts} == {
        "jobs[job-a].log_path",
        "jobs[job-a].checkpoint_path",
        "jobs[job-a].validation_report_path",
    }
    assert all(not item.exists for item in binding.artifacts)
    assert all(item.nearest_existing_path == nearest for item in binding.artifacts)
    assert all(item.device == binding.disk_device for item in binding.artifacts)


def test_monitored_artifact_must_be_inside_fixed_disk_namespace(
    monitor_module, tmp_path
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    spec = _healthy_spec(monitor_module, artifact_root)
    outside = tmp_path / "outside.log"
    outside.write_text("step=1\n")
    spec = replace(
        spec,
        jobs=(replace(spec.jobs[0], log_path=outside),),
    )

    with pytest.raises(ValueError, match="fixed monitored disk_path namespace"):
        monitor_module._validate_monitored_artifact_filesystem(spec)


def test_monitored_artifact_binding_rejects_parent_symlink(
    monitor_module, tmp_path
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (artifact_root / "redirect").symlink_to(outside, target_is_directory=True)
    spec = _healthy_spec(monitor_module, artifact_root)
    spec = replace(
        spec,
        jobs=(
            replace(
                spec.jobs[0],
                log_path=artifact_root / "redirect" / "train.log",
            ),
        ),
    )

    with pytest.raises(ValueError, match="no-follow physical traversal failed"):
        monitor_module._validate_monitored_artifact_filesystem(spec)


def test_monitored_artifact_binding_fails_closed_when_opened_name_vanishes(
    monitor_module, tmp_path, monkeypatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    spec = _healthy_spec(monitor_module, artifact_root)
    real_stat = monitor_module.os.stat

    def vanishing_stat(path, *args, **kwargs):
        if path == "train.log" and kwargs.get("dir_fd") is not None:
            raise FileNotFoundError(path)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(monitor_module.os, "stat", vanishing_stat)

    with pytest.raises(ValueError, match="changed during physical binding"):
        monitor_module._validate_monitored_artifact_filesystem(spec)
