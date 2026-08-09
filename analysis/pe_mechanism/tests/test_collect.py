from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pe_mechanism.provenance as provenance_module
import pytest
from pe_mechanism.collect import ActivationReservoir, collect_activation_files, run


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.invalid")
    (repository / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-q", "-m", "initial")
    monkeypatch.setattr(provenance_module, "__file__", str(repository / "tracked.txt"))
    checkpoint = tmp_path / "raw.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    dataset = tmp_path / "dataset.json"
    dataset.write_text('{"datasets":["alpha"]}\n', encoding="utf-8")
    return {
        "model_family": "tabicl-v2",
        "model_revision": "step-180000",
        "condition": "none",
        "sites": ["blocks.0"],
        "checkpoint_path": str(checkpoint),
        "dataset_manifest_path": str(dataset),
        "training_code_root": str(repository),
        "model_code_root": str(repository),
        "analysis_code_root": str(repository),
    }


@pytest.fixture(autouse=True)
def ample_private_disk(monkeypatch):
    from pe_mechanism import collect

    monkeypatch.setattr(
        collect.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": 100 * 1024**3})(),
    )


def test_reservoir_is_bounded_and_deterministic() -> None:
    values = np.arange(200, dtype=np.float32).reshape(50, 4)
    first = ActivationReservoir(7, seed=5)
    second = ActivationReservoir(7, seed=5)
    for reservoir in (first, second):
        reservoir.add(values[:25], dataset_index=0)
        reservoir.add(values[25:], dataset_index=1)
    first_values, first_datasets = first.arrays()
    second_values, second_datasets = second.arrays()
    np.testing.assert_array_equal(first_values, second_values)
    np.testing.assert_array_equal(first_datasets, second_datasets)
    assert first.seen == 50
    assert first.retained == 7


def test_collect_writes_path_free_bounded_index(tmp_path: Path) -> None:
    source_a = tmp_path / "a.npy"
    source_b = tmp_path / "b.npy"
    np.save(source_a, np.ones((2, 3, 4), dtype=np.float32))
    np.save(source_b, np.full((2, 3, 4), 2.0, dtype=np.float32))
    output = tmp_path / "output"
    index = collect_activation_files(
        {
            "schema_version": 1,
            "seed": 9,
            "max_vectors_per_dataset_site": 5,
            "records": [
                {
                    "site": "row.blocks[0]",
                    "axis_names": ["table", "row", "embedding"],
                    "dataset_id": "alpha",
                    "path": str(source_a),
                    "feature_group_map": [[0, 1, 2]],
                },
                {
                    "site": "row.blocks[0]",
                    "axis_names": ["table", "row", "embedding"],
                    "dataset_id": "beta",
                    "path": str(source_b),
                    "feature_group_map": [[0, 1, 2]],
                },
            ],
        },
        output,
    )
    dataset_entries = index["sites"][0]["datasets"]
    assert [entry["dataset_id"] for entry in dataset_entries] == ["alpha", "beta"]
    assert [entry["seen_vectors"] for entry in dataset_entries] == [6, 6]
    assert [entry["retained_vectors"] for entry in dataset_entries] == [5, 5]
    raw = (output / "activation-index.json").read_text(encoding="utf-8")
    assert str(source_a) not in raw
    shard = np.load(output / dataset_entries[0]["file"], allow_pickle=False)
    assert shard["activations"].shape == (5, 4)
    assert "dataset_index" not in shard.files


def test_reservoir_rejects_overflow_in_published_dtype() -> None:
    reservoir = ActivationReservoir(2, seed=1, dtype="float16")
    with pytest.raises(ValueError, match="cast to float16"):
        reservoir.add(np.asarray([[1e20, 0.0]], dtype=np.float32), dataset_index=0)


def test_missing_provenance_leaves_no_collect_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.npy"
    np.save(source, np.ones((2, 3), dtype=np.float32))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "private_study_root": str(tmp_path),
                "records": [
                    {
                        "site": "blocks.0",
                        "axis_names": ["row", "embedding"],
                        "dataset_id": "alpha",
                        "path": str(source),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="provenance"):
        run(argparse.Namespace(config=config, output_dir=output))
    assert not output.exists()


def test_collect_run_hashes_actual_activation_and_atomically_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.npy"
    np.save(source, np.ones((2, 3), dtype=np.float32))
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                    "schema_version": 1,
                    "private_study_root": str(tmp_path),
                    "seed": 4,
                    "max_vectors_per_dataset_site": 3,
                "records": [
                    {
                        "site": "blocks.0",
                        "axis_names": ["row", "embedding"],
                        "dataset_id": "alpha",
                        "path": str(source),
                        "expected_sha256": expected,
                    }
                ],
                "provenance": _provenance(tmp_path, monkeypatch),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "published"
    assert run(argparse.Namespace(config=config, output_dir=output)) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["inputs"] == [
        {
            "role": "activation.000000",
            "sha256": expected,
            "size_bytes": source.stat().st_size,
        }
    ]
    names = {path.name for path in output.iterdir()}
    assert {"activation-index.json", "manifest.json"} <= names
    assert len([name for name in names if name.startswith("activation-") and name.endswith(".npz")]) == 1
    assert not list(tmp_path.glob(".published.*.staging"))


@pytest.mark.parametrize(
    "override",
    [
        {"min_free_bytes": 0},
        {"max_private_bytes": 31 * 1024**3},
        {"max_private_bytes": -1},
    ],
)
def test_collect_budget_configuration_can_only_tighten_hard_limits(
    tmp_path: Path, override: dict[str, int]
) -> None:
    source = tmp_path / "source.npy"
    np.save(source, np.ones((2, 3), dtype=np.float32))
    config = {
        "schema_version": 1,
        "records": [
            {
                "site": "blocks.0",
                "axis_names": ["row", "embedding"],
                "dataset_id": "alpha",
                "path": str(source),
            }
        ],
        **override,
    }
    with pytest.raises(ValueError):
        collect_activation_files(config, tmp_path / "output")
    assert not (tmp_path / "output").exists()
