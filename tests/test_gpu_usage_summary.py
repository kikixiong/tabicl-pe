from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_gpu_usage.py"


def write_samples(path: Path, utilization_by_gpu: dict[int, float]) -> None:
    rows = []
    for sample in range(40):
        for gpu, utilization in utilization_by_gpu.items():
            rows.append(
                f"2026/07/18 12:00:{sample:02d}, {gpu}, NVIDIA H100, {utilization}, 1000, 81559, 500, 700"
            )
    path.write_text("\n".join(rows))


def test_gpu_usage_gate_passes_when_every_gpu_is_busy(tmp_path):
    csv_path = tmp_path / "busy.csv"
    write_samples(csv_path, {0: 90, 1: 91, 2: 92, 3: 93})
    result = subprocess.run([sys.executable, SCRIPT, csv_path], capture_output=True, text=True)
    assert result.returncode == 0
    assert "gate passed" in result.stdout


def test_gpu_usage_gate_fails_for_one_underutilized_gpu(tmp_path):
    csv_path = tmp_path / "idle.csv"
    write_samples(csv_path, {0: 90, 1: 91, 2: 50, 3: 93})
    result = subprocess.run([sys.executable, SCRIPT, csv_path], capture_output=True, text=True)
    assert result.returncode == 1
    assert "GPU 2" in result.stdout
    assert "gate failed" in result.stdout


def test_gpu_usage_gate_can_trim_idle_samples_after_training(tmp_path):
    csv_path = tmp_path / "training-window.csv"
    samples = [0] * 8 + [90] * 40 + [0] * 20
    csv_path.write_text(
        "\n".join(
            f"2026/07/18 12:00:{sample:02d}, 0, NVIDIA H100, {utilization}, 1000, 81559, 500, 700"
            for sample, utilization in enumerate(samples)
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            csv_path,
            "--expected-gpus",
            "1",
            "--start-after-active",
            "--end-after-active",
            "--warmup-samples",
            "2",
            "--min-samples",
            "24",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "samples=38 mean=90.0%" in result.stdout
    assert "gate passed" in result.stdout
