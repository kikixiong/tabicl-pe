from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
import torch


PACKAGE_ROOT = Path(__file__).parents[1]
SCRIPT = PACKAGE_ROOT / "scripts" / "run_matched_beyondarena_exploratory.py"
WRAPPER = PACKAGE_ROOT / "scripts" / "slurm_matched_beyondarena_exploratory.sh"
SPEC = importlib.util.spec_from_file_location(
    "run_matched_beyondarena_exploratory", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pair(tmp_path: Path) -> Path:
    common_config = {
        "embed_dim": 128,
        "row_nhead": 8,
        "row_num_blocks": 3,
        "col_num_blocks": 3,
        "icl_num_blocks": 12,
    }
    arms = {}
    for arm in ("none", "rope"):
        state = {"weight": torch.zeros(2, 3)}
        if arm == "rope":
            state["row_interactor.tf_row.rope.freqs"] = torch.zeros(8)
        checkpoint = tmp_path / f"{arm}-seed42-stage1-step-250000.ckpt"
        torch.save(
            {
                "curr_step": MODULE.EXPECTED_COMPARISON_STEP,
                "config": {**common_config, "row_identity_mode": arm},
                "state_dict": state,
            },
            checkpoint,
        )
        arms[arm] = {
            "bytes": checkpoint.stat().st_size,
            "curr_step": MODULE.EXPECTED_COMPARISON_STEP,
            "row_identity_mode": arm,
            "sha256": _sha256(checkpoint),
            "snapshot_checkpoint": checkpoint.name,
            "state_dict_tensor_count": len(state),
        }
    manifest = tmp_path / "SNAPSHOT_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "exploratory_same_step_pilot_checkpoint_pair",
                "formal_eligible": False,
                "comparison_step": MODULE.EXPECTED_COMPARISON_STEP,
                "seed": MODULE.EXPECTED_SEED,
                "pilot_source_commit": MODULE.EXPECTED_MODEL_SHA,
                "arms": arms,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    lines = [
        f"{arms[arm]['sha256']}  {arms[arm]['snapshot_checkpoint']}"
        for arm in ("none", "rope")
    ]
    lines.append(f"{_sha256(manifest)}  {manifest.name}")
    (tmp_path / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")
    return manifest


def test_frozen_pair_loader_checks_bytes_modes_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _pair(tmp_path)
    monkeypatch.setattr(MODULE, "EXPECTED_MANIFEST_SHA256", _sha256(manifest))
    checkpoints, digests, contract = MODULE._load_matched_pair(manifest)
    assert set(checkpoints) == {"rope", "none"}
    assert digests["none"] == _sha256(checkpoints["none"])
    assert contract["comparison_step"] == 250_000
    assert contract["seed"] == 42
    assert contract["pilot_source_commit"] == MODULE.EXPECTED_MODEL_SHA


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed", 43, "frozen pilot contract"),
        ("pilot_source_commit", "f" * 40, "frozen pilot contract"),
        ("comparison_step", 249_999, "frozen pilot contract"),
    ],
)
def test_pair_loader_rejects_wrong_manifest_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    manifest = _pair(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload[field] = value
    manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(MODULE, "EXPECTED_MANIFEST_SHA256", _sha256(manifest))
    with pytest.raises(ValueError, match=message):
        MODULE._load_matched_pair(manifest)


def test_pair_loader_rejects_checkpoint_digest_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _pair(tmp_path)
    monkeypatch.setattr(MODULE, "EXPECTED_MANIFEST_SHA256", _sha256(manifest))
    checkpoint = tmp_path / "none-seed42-stage1-step-250000.ckpt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="bytes disagree"):
        MODULE._load_matched_pair(manifest)


def test_default_smoke_roster_covers_iid_grouped_and_temporal() -> None:
    assert MODULE._requested_datasets(()) == (
        "blood_transfusion",
        "parkinsons_biomedical_voice_measurements",
        "ghanas_indigenous_intel",
    )


def test_requested_datasets_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="unique"):
        MODULE._requested_datasets(("blood_transfusion", "blood_transfusion"))


def test_mean_ranks_by_regime_uses_dynamic_arm_order() -> None:
    calls = []

    def mean_ranks(rows, *, roster, arm_order):
        calls.append((tuple(roster), tuple(arm_order)))
        return {arm: 1.5 for arm in arm_order}

    result = MODULE._mean_ranks_by_regime(
        {},
        roster=("iid-id", "grouped-id"),
        expected={
            "iid-id": {"split_regime": "iid"},
            "grouped-id": {"split_regime": "grouped"},
        },
        arm_order=("rope", "none"),
        mean_ranks=mean_ranks,
    )
    assert set(result) == {"iid", "grouped"}
    assert all(arms == ("rope", "none") for _, arms in calls)


def test_wrapper_is_bounded_a10_smoke() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    for directive in (
        "#SBATCH --partition=normal",
        "#SBATCH --qos=short",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=16",
        "#SBATCH --mem=64G",
        "#SBATCH --time=03:00:00",
    ):
        assert directive in text
    completed = subprocess.run(
        ["bash", "-n", str(WRAPPER)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_runner_builds_explicit_dataset_grid_before_run() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "jobs = arena.build_jobs(" in text
    assert "dataset_names=list(requested)" in text
    assert text.index("jobs = arena.build_jobs(") < text.index("results = arena.run_jobs(")
