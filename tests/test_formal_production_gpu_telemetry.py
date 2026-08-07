from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def telemetry_module():
    path = ROOT / "scripts" / "formal_production_gpu_telemetry.py"
    spec = importlib.util.spec_from_file_location(
        "_formal_production_gpu_telemetry_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeClock:
    def __init__(self, now_ns: int = 1_000_000_000):
        self.now_ns = now_ns

    def monotonic_ns(self) -> int:
        return self.now_ns

    def sleep(self, seconds: float) -> None:
        self.now_ns += int(seconds * 1_000_000_000)


def _fake_nvidia_smi(
    tmp_path: Path,
    *,
    uuid_rows: tuple[str, ...] = ("GPU-fixture",),
    sample_rows: tuple[str, ...] = ("GPU-fixture, 90", "GPU-fixture, 80"),
    end_path: Path | None = None,
    end_after_sample: int | None = 2,
) -> tuple[Path, Path]:
    executable = tmp_path / "nvidia-smi"
    log = tmp_path / "nvidia-args.jsonl"
    counter = tmp_path / "nvidia-counter"
    script = f"""#!/usr/bin/python3
import json
from pathlib import Path
import sys

log = Path({str(log)!r})
counter_path = Path({str(counter)!r})
with log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(sys.argv[1:], separators=(",", ":")) + "\\n")
query = next((arg for arg in sys.argv[1:] if arg.startswith("--query-gpu=")), "")
if query == "--query-gpu=uuid":
    rows = {uuid_rows!r}
    print("\\n".join(rows))
    raise SystemExit(0)
if query != "--query-gpu=uuid,utilization.gpu":
    print("unexpected query", file=sys.stderr)
    raise SystemExit(9)
count = int(counter_path.read_text()) if counter_path.exists() else 0
rows = {sample_rows!r}
row = rows[min(count, len(rows) - 1)]
counter_path.write_text(str(count + 1))
print(row)
end_path = {str(end_path) if end_path is not None else ""!r}
end_after = {end_after_sample!r}
if end_path and end_after is not None and count + 1 >= end_after:
    Path(end_path).touch(exist_ok=True)
"""
    executable.write_text(script, encoding="utf-8")
    executable.chmod(0o755)
    return executable, log


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "output_path": tmp_path / "gpu.jsonl",
        "ready_path": tmp_path / "ready",
        "end_path": tmp_path / "end",
        "abort_path": tmp_path / "abort",
    }


def _record(
    module,
    tmp_path: Path,
    *,
    nvidia_smi: Path,
    clock: FakeClock,
    time_limit: str = "00:02:00",
):
    paths = _paths(tmp_path)
    contract = module.telemetry_contract(time_limit)
    summary = module.record_telemetry(
        **paths,
        nvidia_smi=nvidia_smi,
        expected_nvidia_smi_sha256=hashlib.sha256(
            nvidia_smi.read_bytes()
        ).hexdigest(),
        visible_gpu_token="0",
        expected_gpu_uuid="GPU-fixture",
        contract=contract,
        expected_sample_ceiling=contract.sample_ceiling,
        expected_byte_ceiling=contract.byte_ceiling,
        monotonic_ns=clock.monotonic_ns,
        sleep_fn=clock.sleep,
    )
    return paths, contract, summary


def test_contract_is_exactly_derived_from_canonical_time_limit(telemetry_module):
    one_minute = telemetry_module.telemetry_contract("00:01:00")
    one_hour = telemetry_module.telemetry_contract("01:00:00")
    assert one_minute.sample_ceiling == 2
    assert one_hour.sample_ceiling == 61
    assert one_hour.byte_ceiling > one_minute.byte_ceiling
    assert one_hour.cadence_seconds == 60
    assert one_hour.minimum_mean_samples == 10
    assert one_hour.low_utilization_threshold_percent == 80.0
    assert one_hour == telemetry_module.telemetry_contract(one_hour.time_limit)
    assert len(telemetry_module.telemetry_contract_sha256(one_hour)) == 64


@pytest.mark.parametrize(
    "value",
    [
        "0:01:00",
        "24:00:00",
        "00:60:00",
        "00:00:00",
        "01:00",
        "0-01:00:00",
        " 01:00:00",
    ],
)
def test_contract_rejects_noncanonical_or_zero_time_limits(
    telemetry_module, value: str
):
    with pytest.raises(ValueError, match="TimeLimit"):
        telemetry_module.telemetry_contract(value)


@pytest.mark.parametrize(
    ("value", "uuid"),
    [
        ("0", False),
        ("17", False),
        ("GPU-01234567-abcd", False),
        ("MIG-GPU-01234567/1/2", False),
        ("GPU-01234567-abcd", True),
        ("MIG-GPU-01234567/1/2", True),
    ],
)
def test_allocated_token_forms_are_accepted(
    telemetry_module, value: str, uuid: bool
):
    assert telemetry_module._validate_gpu_token(value, uuid=uuid) == value


@pytest.mark.parametrize(
    "value",
    [
        "0GPU-a",
        "GPU-aGPU-b",
        "GPU-aMIG-b",
        "MIG-aMIG-b",
        "GPU-a,GPU-b",
        "0,1",
    ],
)
def test_concatenated_or_multi_gpu_tokens_are_rejected(
    telemetry_module, value: str
):
    with pytest.raises(ValueError, match="token/UUID"):
        telemetry_module._validate_gpu_token(value)


def test_recorder_writes_canonical_scoped_stream_and_terminal_end(
    telemetry_module, tmp_path: Path
):
    paths = _paths(tmp_path)
    nvidia_smi, log = _fake_nvidia_smi(
        tmp_path, end_path=paths["end_path"], end_after_sample=2
    )
    clock = FakeClock()
    paths, contract, summary = _record(
        telemetry_module, tmp_path, nvidia_smi=nvidia_smi, clock=clock
    )
    raw = paths["output_path"].read_bytes()
    parsed = telemetry_module.parse_telemetry_bytes(
        raw,
        contract=contract,
        expected_gpu_uuid="GPU-fixture",
        require_terminal=True,
    )
    assert summary == parsed
    assert parsed.terminal_kind == "end"
    assert parsed.sample_count == 2
    assert parsed.utilization_mean_percent == 85.0
    assert len(raw) <= contract.byte_ceiling
    records = [json.loads(line) for line in raw.splitlines()]
    assert [record["kind"] for record in records] == [
        "start",
        "sample",
        "sample",
        "end",
    ]
    invocations = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(invocations) == 3
    assert all("--id=0" in invocation for invocation in invocations)
    assert all(
        "--format=csv,noheader,nounits" in invocation for invocation in invocations
    )
    assert invocations[0] == [
        "--id=0",
        "--query-gpu=uuid",
        "--format=csv,noheader,nounits",
    ]
    assert all("--query-gpu=name,uuid" not in invocation for invocation in invocations)


def test_uuid_drift_records_abort_and_fails_closed(telemetry_module, tmp_path: Path):
    nvidia_smi, _log = _fake_nvidia_smi(
        tmp_path,
        sample_rows=("GPU-other, 90",),
        end_after_sample=None,
    )
    paths = _paths(tmp_path)
    contract = telemetry_module.telemetry_contract("00:02:00")
    with pytest.raises(telemetry_module.TelemetryFailure) as raised:
        telemetry_module.record_telemetry(
            **paths,
            nvidia_smi=nvidia_smi,
            expected_nvidia_smi_sha256=hashlib.sha256(
                nvidia_smi.read_bytes()
            ).hexdigest(),
            visible_gpu_token="GPU-fixture",
            expected_gpu_uuid="GPU-fixture",
            contract=contract,
            expected_sample_ceiling=contract.sample_ceiling,
            expected_byte_ceiling=contract.byte_ceiling,
            monotonic_ns=FakeClock().monotonic_ns,
            sleep_fn=lambda _seconds: None,
        )
    assert raised.value.reason == "gpu_uuid_drift"
    parsed = telemetry_module.parse_telemetry_bytes(
        paths["output_path"].read_bytes(),
        contract=contract,
        expected_gpu_uuid="GPU-fixture",
        require_terminal=True,
    )
    assert parsed.terminal_kind == "abort"
    assert parsed.abort_reason == "gpu_uuid_drift"


def test_multiple_nvidia_rows_are_rejected_and_abort_is_reserved(
    telemetry_module, tmp_path: Path
):
    nvidia_smi, _log = _fake_nvidia_smi(
        tmp_path,
        sample_rows=("GPU-fixture, 90\nGPU-unallocated, 99",),
        end_after_sample=None,
    )
    paths = _paths(tmp_path)
    contract = telemetry_module.telemetry_contract("00:01:00")
    with pytest.raises(telemetry_module.TelemetryFailure) as raised:
        telemetry_module.record_telemetry(
            **paths,
            nvidia_smi=nvidia_smi,
            expected_nvidia_smi_sha256=hashlib.sha256(
                nvidia_smi.read_bytes()
            ).hexdigest(),
            visible_gpu_token="0",
            expected_gpu_uuid="GPU-fixture",
            contract=contract,
            expected_sample_ceiling=contract.sample_ceiling,
            expected_byte_ceiling=contract.byte_ceiling,
            monotonic_ns=FakeClock().monotonic_ns,
            sleep_fn=lambda _seconds: None,
        )
    assert raised.value.reason == "gpu_query_failed"
    raw = paths["output_path"].read_bytes()
    assert len(raw) <= contract.byte_ceiling
    assert json.loads(raw.splitlines()[-1])["reason"] == "gpu_query_failed"


def test_cadence_drift_records_abort_before_querying_sample(
    telemetry_module, tmp_path: Path
):
    nvidia_smi, log = _fake_nvidia_smi(tmp_path, end_after_sample=None)
    paths = _paths(tmp_path)
    contract = telemetry_module.telemetry_contract("00:01:00")
    values = iter(
        [
            1_000_000_000,
            17_000_000_001,
            17_000_000_001,
        ]
    )
    with pytest.raises(telemetry_module.TelemetryFailure) as raised:
        telemetry_module.record_telemetry(
            **paths,
            nvidia_smi=nvidia_smi,
            expected_nvidia_smi_sha256=hashlib.sha256(
                nvidia_smi.read_bytes()
            ).hexdigest(),
            visible_gpu_token="0",
            expected_gpu_uuid="GPU-fixture",
            contract=contract,
            expected_sample_ceiling=contract.sample_ceiling,
            expected_byte_ceiling=contract.byte_ceiling,
            monotonic_ns=lambda: next(values),
            sleep_fn=lambda _seconds: None,
        )
    assert raised.value.reason == "cadence_drift"
    assert len(log.read_text().splitlines()) == 1
    assert json.loads(paths["output_path"].read_bytes().splitlines()[-1])[
        "reason"
    ] == "cadence_drift"


def test_time_limit_sample_ceiling_aborts_instead_of_growing(
    telemetry_module, tmp_path: Path
):
    nvidia_smi, _log = _fake_nvidia_smi(
        tmp_path,
        sample_rows=("GPU-fixture, 90",),
        end_after_sample=None,
    )
    paths = _paths(tmp_path)
    contract = telemetry_module.telemetry_contract("00:01:00")
    clock = FakeClock()
    with pytest.raises(telemetry_module.TelemetryFailure) as raised:
        telemetry_module.record_telemetry(
            **paths,
            nvidia_smi=nvidia_smi,
            expected_nvidia_smi_sha256=hashlib.sha256(
                nvidia_smi.read_bytes()
            ).hexdigest(),
            visible_gpu_token="0",
            expected_gpu_uuid="GPU-fixture",
            contract=contract,
            expected_sample_ceiling=contract.sample_ceiling,
            expected_byte_ceiling=contract.byte_ceiling,
            monotonic_ns=clock.monotonic_ns,
            sleep_fn=clock.sleep,
        )
    assert raised.value.reason == "sample_ceiling_exceeded"
    parsed = telemetry_module.parse_telemetry_bytes(
        paths["output_path"].read_bytes(),
        contract=contract,
        expected_gpu_uuid="GPU-fixture",
        require_terminal=True,
    )
    assert parsed.sample_count == contract.sample_ceiling
    assert parsed.abort_reason == "sample_ceiling_exceeded"


def test_ceiling_exports_must_match_frozen_time_limit(
    telemetry_module, tmp_path: Path
):
    nvidia_smi, _log = _fake_nvidia_smi(tmp_path)
    paths = _paths(tmp_path)
    contract = telemetry_module.telemetry_contract("00:01:00")
    with pytest.raises(ValueError, match="ceilings"):
        telemetry_module.record_telemetry(
            **paths,
            nvidia_smi=nvidia_smi,
            expected_nvidia_smi_sha256=hashlib.sha256(
                nvidia_smi.read_bytes()
            ).hexdigest(),
            visible_gpu_token="0",
            expected_gpu_uuid="GPU-fixture",
            contract=contract,
            expected_sample_ceiling=contract.sample_ceiling,
            expected_byte_ceiling=contract.byte_ceiling + 1,
        )
    assert not paths["output_path"].exists()


def test_nvidia_smi_digest_must_match_before_any_query(
    telemetry_module, tmp_path: Path
):
    nvidia_smi, log = _fake_nvidia_smi(tmp_path)
    paths = _paths(tmp_path)
    contract = telemetry_module.telemetry_contract("00:01:00")
    with pytest.raises(ValueError, match="SHA-256"):
        telemetry_module.record_telemetry(
            **paths,
            nvidia_smi=nvidia_smi,
            expected_nvidia_smi_sha256="0" * 64,
            visible_gpu_token="0",
            expected_gpu_uuid="GPU-fixture",
            contract=contract,
            expected_sample_ceiling=contract.sample_ceiling,
            expected_byte_ceiling=contract.byte_ceiling,
        )
    assert not paths["output_path"].exists()
    assert not log.exists()


def test_external_abort_reason_is_canonical_and_terminal(
    telemetry_module, tmp_path: Path
):
    paths = _paths(tmp_path)
    nvidia_smi, _log = _fake_nvidia_smi(
        tmp_path,
        end_path=None,
        end_after_sample=None,
    )
    paths["abort_path"].write_text("training_failed\n", encoding="ascii")
    contract = telemetry_module.telemetry_contract("00:01:00")
    # Abort controls must be fresh before the recorder starts.  Simulate the
    # supervisor writing it immediately after the recorder publishes ready.
    paths["abort_path"].unlink()

    class AbortClock(FakeClock):
        def monotonic_ns(self) -> int:
            value = super().monotonic_ns()
            if paths["ready_path"].exists() and not paths["abort_path"].exists():
                paths["abort_path"].write_text("training_failed\n", encoding="ascii")
            return value

    summary = telemetry_module.record_telemetry(
        **paths,
        nvidia_smi=nvidia_smi,
        expected_nvidia_smi_sha256=hashlib.sha256(
            nvidia_smi.read_bytes()
        ).hexdigest(),
        visible_gpu_token="0",
        expected_gpu_uuid="GPU-fixture",
        contract=contract,
        expected_sample_ceiling=contract.sample_ceiling,
        expected_byte_ceiling=contract.byte_ceiling,
        monotonic_ns=AbortClock().monotonic_ns,
        sleep_fn=lambda _seconds: None,
    )
    assert summary.terminal_kind == "abort"
    assert summary.abort_reason == "training_failed"


def test_parser_rejects_noncanonical_and_missing_terminal(
    telemetry_module, tmp_path: Path
):
    paths = _paths(tmp_path)
    nvidia_smi, _log = _fake_nvidia_smi(
        tmp_path, end_path=paths["end_path"], end_after_sample=1
    )
    paths, contract, _summary = _record(
        telemetry_module, tmp_path, nvidia_smi=nvidia_smi, clock=FakeClock()
    )
    raw = paths["output_path"].read_bytes()
    partial = b"\n".join(raw.splitlines()[:-1]) + b"\n"
    live = telemetry_module.parse_telemetry_bytes(
        partial,
        contract=contract,
        expected_gpu_uuid="GPU-fixture",
        require_terminal=False,
    )
    assert live.terminal_kind is None
    with pytest.raises(ValueError, match="terminal"):
        telemetry_module.parse_telemetry_bytes(
            partial,
            contract=contract,
            expected_gpu_uuid="GPU-fixture",
            require_terminal=True,
        )
    first, *rest = raw.splitlines()
    noncanonical = first.replace(b'{"cadence', b'{ "cadence', 1)
    with pytest.raises(ValueError, match="canonical"):
        telemetry_module.parse_telemetry_bytes(
            b"\n".join([noncanonical, *rest]) + b"\n",
            contract=contract,
            expected_gpu_uuid="GPU-fixture",
            require_terminal=True,
        )


def test_stable_file_reader_binds_size_digest_and_rejects_symlink(
    telemetry_module, tmp_path: Path
):
    paths = _paths(tmp_path)
    nvidia_smi, _log = _fake_nvidia_smi(
        tmp_path, end_path=paths["end_path"], end_after_sample=1
    )
    paths, contract, summary = _record(
        telemetry_module, tmp_path, nvidia_smi=nvidia_smi, clock=FakeClock()
    )
    evidence = telemetry_module.read_telemetry_file(
        paths["output_path"].resolve(),
        contract=contract,
        expected_gpu_uuid="GPU-fixture",
        require_terminal=True,
    )
    raw = paths["output_path"].read_bytes()
    assert evidence.summary == summary
    assert evidence.size_bytes == len(raw)
    assert evidence.sha256 == hashlib.sha256(raw).hexdigest()
    link = tmp_path / "gpu-link.jsonl"
    link.symlink_to(paths["output_path"])
    with pytest.raises(ValueError, match="physical"):
        telemetry_module.read_telemetry_file(
            link,
            contract=contract,
            expected_gpu_uuid="GPU-fixture",
            require_terminal=True,
        )
