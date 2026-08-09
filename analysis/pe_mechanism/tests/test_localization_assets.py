from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess

import pytest

from pe_mechanism.localization import _validate_configuration
from pe_mechanism.provenance import VerifiedConfiguration, load_verified_json_config


PACKAGE_ROOT = Path(__file__).parents[1]
EXAMPLE = PACKAGE_ROOT / "examples" / "localize-matched-step.example.json"
WRAPPER = PACKAGE_ROOT / "scripts" / "slurm_fixed_weight_localization.sh"
COLLECT_WRAPPER = PACKAGE_ROOT / "scripts" / "slurm_official_collect.sh"


def _valid_wrapper_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    analysis_root = tmp_path / "analysis"
    model_root = tmp_path / "model"
    (analysis_root / "src" / "pe_mechanism").mkdir(parents=True)
    (model_root / "src" / "tabicl").mkdir(parents=True)
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    capture = tmp_path / "capture.txt"
    python = tmp_path / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "{\n"
        "  printf 'PYTHONNOUSERSITE=%s\\n' \"${PYTHONNOUSERSITE-}\"\n"
        "  printf 'PYTHONDONTWRITEBYTECODE=%s\\n' \"${PYTHONDONTWRITEBYTECODE-}\"\n"
        "  printf 'PYTHONHASHSEED=%s\\n' \"${PYTHONHASHSEED-}\"\n"
        "  printf 'PYTHONPATH=%s\\n' \"${PYTHONPATH-}\"\n"
        "  printf 'ARG=%s\\n' \"$@\"\n"
        "} > \"$WRAPPER_CAPTURE\"\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "WRAPPER_CAPTURE": str(capture),
        "PE_ANALYSIS_ROOT": str(analysis_root),
        "PE_MODEL_ROOT": str(model_root),
        "PE_CONFIG": str(config),
        "PE_OUTPUT_DIR": str(tmp_path / "private" / "run"),
        "PE_PYTHON": str(python),
    }
    return environment, capture


def test_localization_example_matches_the_registered_config_shape(tmp_path: Path) -> None:
    source = load_verified_json_config(EXAMPLE)
    raw = copy.deepcopy(source.data)
    private_root = tmp_path / "private"
    dataset_id = raw["datasets"][0]["dataset_id"]
    dataset_root = tmp_path / dataset_id
    private_root.mkdir()
    dataset_root.mkdir()
    raw["private_study_root"] = str(private_root)
    raw["datasets"][0]["path"] = str(dataset_root)

    config, specs, conditions = _validate_configuration(
        VerifiedConfiguration(data=raw, file=source.file)
    )

    assert config["comparison_step"] == 250000
    assert [spec.dataset_id for spec in specs] == [dataset_id]
    assert [condition.name for condition in conditions] == [
        "full",
        "all_off",
        "q_only",
        "k_only",
        "phase_half",
        "keep_low_only",
        "keep_high_only",
        "block_0_off",
        "block_1_off",
        "block_2_off",
    ]
    assert sum(condition.is_full_policy for condition in conditions) == 1

    original = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    path_values = [
        original["private_study_root"],
        original["datasets"][0]["path"],
        original["secondary_checkpoint_path"],
        original["snapshot_manifest_path"],
        original["provenance"]["checkpoint_path"],
        original["provenance"]["dataset_manifest_path"],
        original["provenance"]["training_code_root"],
        original["provenance"]["model_code_root"],
        original["provenance"]["analysis_code_root"],
    ]
    assert all(Path(value).is_absolute() for value in path_values)


def test_slurm_wrapper_has_fixed_resources_and_exact_invocation(tmp_path: Path) -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    for directive in (
        "#SBATCH --partition=h100",
        "#SBATCH --qos=short",
        "#SBATCH --time=03:00:00",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=16",
        "#SBATCH --mem=64G",
    ):
        assert directive in text
    assert "nvidia-smi" not in text
    assert "git " not in text

    environment, capture = _valid_wrapper_environment(tmp_path)
    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "PYTHONNOUSERSITE=1",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONHASHSEED=0",
        (
            f"PYTHONPATH={environment['PE_ANALYSIS_ROOT']}/src:"
            f"{environment['PE_MODEL_ROOT']}/src"
        ),
        "ARG=-B",
        "ARG=-m",
        "ARG=pe_mechanism",
        "ARG=localize",
        "ARG=--config",
        f"ARG={environment['PE_CONFIG']}",
        "ARG=--output-dir",
        f"ARG={environment['PE_OUTPUT_DIR']}",
    ]


def test_official_collect_wrapper_freezes_hash_seed_and_exact_invocation(
    tmp_path: Path,
) -> None:
    text = COLLECT_WRAPPER.read_text(encoding="utf-8")
    for directive in (
        "#SBATCH --partition=h100",
        "#SBATCH --qos=short",
        "#SBATCH --time=01:00:00",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=16",
        "#SBATCH --mem=64G",
    ):
        assert directive in text
    assert "export PYTHONHASHSEED=0" in text
    assert "nvidia-smi" not in text
    assert "git " not in text

    environment, capture = _valid_wrapper_environment(tmp_path)
    completed = subprocess.run(
        ["bash", str(COLLECT_WRAPPER)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "PYTHONNOUSERSITE=1",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONHASHSEED=0",
        (
            f"PYTHONPATH={environment['PE_ANALYSIS_ROOT']}/src:"
            f"{environment['PE_MODEL_ROOT']}/src"
        ),
        "ARG=-B",
        "ARG=-m",
        "ARG=pe_mechanism",
        "ARG=official-collect",
        "ARG=--config",
        f"ARG={environment['PE_CONFIG']}",
        "ARG=--output-dir",
        f"ARG={environment['PE_OUTPUT_DIR']}",
    ]


@pytest.mark.parametrize(
    "name",
    (
        "PE_ANALYSIS_ROOT",
        "PE_MODEL_ROOT",
        "PE_CONFIG",
        "PE_OUTPUT_DIR",
        "PE_PYTHON",
    ),
)
def test_slurm_wrapper_rejects_every_relative_input(tmp_path: Path, name: str) -> None:
    environment, capture = _valid_wrapper_environment(tmp_path)
    environment[name] = "relative/path"
    completed = subprocess.run(
        ["bash", str(WRAPPER)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 2
    assert f"{name} must be a non-empty absolute path" in completed.stderr
    assert not capture.exists()
