from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from dataclasses import replace
import io
from pathlib import Path
import sys
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
        if command[0] == "squeue":
            output = "\n".join(
                f"{job_id}|{state}"
                for job_id, state in self.states.items()
                if state in {"PENDING", "RUNNING"}
            )
            return self.module.CommandResult(0, output, "")
        if command[0] == "sacct":
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
        if argv[0] == self.failing_command:
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
    assert {call[0] for call in scheduler.calls} == {"squeue", "sacct"}
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
        "Traceback (most recent call last):\n"
        "RuntimeError: failed\n"
    )

    anomalous = monitor.poll(initial.state)
    repeated = monitor.poll(anomalous.state)

    assert {event.code for event in anomalous.events} == {
        "step_regressed",
        "log_nonfinite",
        "log_oom",
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
    # Make the metadata identity change deterministic even on filesystems with
    # coarse or coalesced wall-clock timestamp updates.
    os.utime(
        spec.jobs[0].checkpoint_path,
        ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns + 1_000_000_000),
    )
    changed = monitor.poll(second.state)

    assert first.events == ()
    assert second.events == ()
    assert changed.events == ()
    assert len(calls) == 2
    assert calls[0][1][1:] == (stat_before.st_mtime_ns, stat_before.st_size)
    assert calls[0][2] == 1 << 20


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


def _write_monitor_manifest(path: Path, spec) -> None:
    jobs = []
    for job in spec.jobs:
        jobs.append(
            {
                "job_id": job.job_id,
                "arm": job.arm,
                "stage": job.stage,
                "log_path": str(job.log_path),
                "checkpoint_path": str(job.checkpoint_path),
                "validation_report_path": str(job.validation_report_path),
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
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "disk_path": str(spec.disk_path),
                "max_event_ledger_bytes": spec.max_event_ledger_bytes,
                "jobs": jobs,
            }
        )
    )


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

    assert loaded == spec
    assert args.duration_seconds == 1_800.0
    assert args.once is True

    payload = json.loads(manifest.read_text())
    payload["jobs"][0]["log_path"] = "relative.log"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="absolute"):
        monitor_module.load_monitor_spec(manifest)


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
