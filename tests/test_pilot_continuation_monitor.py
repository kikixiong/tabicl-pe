from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
MONITOR_PATH = ROOT / "scripts" / "monitor_pilot_continuation.py"
RECEIPT_WRITER = ROOT / "scripts" / "write_pilot_continuation_receipt.py"


def load_monitor():
    spec = importlib.util.spec_from_file_location("pilot_monitor", MONITOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_monitor_detects_fatal_log_signatures(tmp_path: Path) -> None:
    monitor = load_monitor()
    healthy = tmp_path / "healthy.err"
    failed = tmp_path / "failed.err"
    healthy.write_text("training step completed\n")
    failed.write_text("Traceback: CUDA out of memory\n")
    assert monitor.tail_has_anomaly(healthy) is None
    assert monitor.tail_has_anomaly(failed) == "Traceback"


def test_monitor_gpu_summary_uses_active_window(tmp_path: Path) -> None:
    monitor = load_monitor()
    path = tmp_path / "gpu.csv"
    values = [0] * 5 + [90] * 40 + [0] * 8
    path.write_text(
        "\n".join(
            f"2026/08/17 12:00:{index:02d}, 0, NVIDIA H100, {value}, 1000, 81559, 500, 700"
            for index, value in enumerate(values)
        )
    )
    assert monitor.gpu_summary(path) == {0: 90.0}


def test_receipt_writer_and_monitor_validate_self_hash(tmp_path: Path) -> None:
    monitor = load_monitor()
    receipt = tmp_path / "receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            RECEIPT_WRITER,
            "--output",
            receipt,
            "--continuation-id",
            "unit-test-v1",
            "--source-commit",
            "2" * 40,
            "--repository-root",
            tmp_path,
            "--rope-stage1-job",
            "101",
            "--rope-stage2-job",
            "102",
            "--none-stage2-job",
            "103",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert monitor.load_receipt(receipt)["jobs"]["rope-stage2-after-500k"] == "102"

    value = json.loads(receipt.read_text())
    value["source_commit"] = "3" * 40
    receipt.chmod(0o600)
    receipt.write_text(json.dumps(value))
    try:
        monitor.load_receipt(receipt)
    except ValueError as error:
        assert "self-hash" in str(error)
    else:
        raise AssertionError("tampered receipt was accepted")
