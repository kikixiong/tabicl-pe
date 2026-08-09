from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


PACKAGE_ROOT = Path(__file__).parents[1]
WRAPPER = PACKAGE_ROOT / "scripts" / "slurm_tabpfn_localize.sh"


def _valid_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    tabpfn_root = tmp_path / "tabpfn-source"
    (tabpfn_root / "src" / "tabpfn").mkdir(parents=True)
    subprocess.run(
        ["git", "init", "--quiet", str(tabpfn_root)],
        check=True,
        capture_output=True,
    )

    private_root = tmp_path / "private"
    runtime_root = private_root / "runtime"
    runtime_root.mkdir(parents=True)
    config = private_root / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    capture = private_root / "capture.txt"
    python = private_root / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "{\n"
        "  printf 'HOME=%s\\n' \"${HOME-}\"\n"
        "  printf 'HF_HOME=%s\\n' \"${HF_HOME-}\"\n"
        "  printf 'XDG_CACHE_HOME=%s\\n' \"${XDG_CACHE_HOME-}\"\n"
        "  printf 'TMPDIR=%s\\n' \"${TMPDIR-}\"\n"
        "  printf 'HF_HUB_OFFLINE=%s\\n' \"${HF_HUB_OFFLINE-}\"\n"
        "  printf 'PYTHONNOUSERSITE=%s\\n' \"${PYTHONNOUSERSITE-}\"\n"
        "  printf 'PYTHONDONTWRITEBYTECODE=%s\\n' \"${PYTHONDONTWRITEBYTECODE-}\"\n"
        "  printf 'PYTHONHASHSEED=%s\\n' \"${PYTHONHASHSEED-}\"\n"
        "  printf 'PYTHONPATH=%s\\n' \"${PYTHONPATH-}\"\n"
        "  printf 'TABPFN_TOKEN=%s\\n' \"${TABPFN_TOKEN-unset}\"\n"
        "  printf 'HF_TOKEN=%s\\n' \"${HF_TOKEN-unset}\"\n"
        "  printf 'HUGGING_FACE_HUB_TOKEN=%s\\n' "
        '"${HUGGING_FACE_HUB_TOKEN-unset}"\n'
        "  printf 'ARG=%s\\n' \"$@\"\n"
        '} > "$WRAPPER_CAPTURE"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "WRAPPER_CAPTURE": str(capture),
        "PE_ANALYSIS_ROOT": str(PACKAGE_ROOT),
        "PE_TABPFN_ROOT": str(tabpfn_root),
        "PE_CONFIG": str(config),
        "PE_OUTPUT_DIR": str(private_root / "run"),
        "PE_RUNTIME_ROOT": str(runtime_root),
        "PE_PYTHON": str(python),
        "TABPFN_TOKEN": "must-not-reach-inference",
        "HF_TOKEN": "must-not-reach-inference",
        "HUGGING_FACE_HUB_TOKEN": "must-not-reach-inference",
    }
    return environment, capture, runtime_root


def test_tabpfn_wrapper_fixes_a10_resources_and_offline_invocation(
    tmp_path: Path,
) -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    for directive in (
        "#SBATCH --partition=normal",
        "#SBATCH --qos=short",
        "#SBATCH --time=03:00:00",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=16",
        "#SBATCH --mem=64G",
    ):
        assert directive in text
    assert "#sbatch --partition=h100" not in text.lower()
    assert "export HF_HUB_OFFLINE=1" in text
    assert "unset TABPFN_TOKEN HF_TOKEN HUGGING_FACE_HUB_TOKEN" in text

    syntax = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr

    environment, capture, runtime_root = _valid_environment(tmp_path)
    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    assert capture.read_text(encoding="utf-8").splitlines() == [
        f"HOME={runtime_root}/home",
        f"HF_HOME={runtime_root}/cache/huggingface",
        f"XDG_CACHE_HOME={runtime_root}/cache/xdg",
        f"TMPDIR={runtime_root}/tmp",
        "HF_HUB_OFFLINE=1",
        "PYTHONNOUSERSITE=1",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONHASHSEED=0",
        (
            f"PYTHONPATH={environment['PE_ANALYSIS_ROOT']}/src:"
            f"{environment['PE_TABPFN_ROOT']}/src"
        ),
        "TABPFN_TOKEN=unset",
        "HF_TOKEN=unset",
        "HUGGING_FACE_HUB_TOKEN=unset",
        "ARG=-B",
        "ARG=-m",
        "ARG=pe_mechanism",
        "ARG=tabpfn-localize",
        "ARG=--config",
        f"ARG={environment['PE_CONFIG']}",
        "ARG=--output-dir",
        f"ARG={environment['PE_OUTPUT_DIR']}",
    ]


@pytest.mark.parametrize(
    "name",
    (
        "PE_ANALYSIS_ROOT",
        "PE_TABPFN_ROOT",
        "PE_CONFIG",
        "PE_OUTPUT_DIR",
        "PE_RUNTIME_ROOT",
        "PE_PYTHON",
    ),
)
def test_tabpfn_wrapper_rejects_every_relative_input(
    tmp_path: Path,
    name: str,
) -> None:
    environment, capture, _ = _valid_environment(tmp_path)
    environment[name] = "relative/path"
    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert f"{name} must be a non-empty absolute path" in completed.stderr
    assert not capture.exists()


def test_tabpfn_wrapper_rejects_a_source_checkout_working_directory(
    tmp_path: Path,
) -> None:
    environment, capture, _ = _valid_environment(tmp_path)
    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        cwd=PACKAGE_ROOT,
    )
    assert completed.returncode == 2
    assert "Slurm working directory must be outside" in completed.stderr
    assert not capture.exists()


def test_tabpfn_wrapper_requires_a_fresh_output_directory(tmp_path: Path) -> None:
    environment, capture, _ = _valid_environment(tmp_path)
    Path(environment["PE_OUTPUT_DIR"]).mkdir()
    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert "PE_OUTPUT_DIR must not exist" in completed.stderr
    assert not capture.exists()
