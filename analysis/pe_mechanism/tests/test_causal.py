from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pe_mechanism.provenance as provenance_module
import pytest
import torch
from pe_mechanism.causal import (
    activation_frequency,
    intervene_latents,
    matched_random_control_features,
    no_op_reconstruction,
    paired_effect_records,
    run,
    summarize_effect_records,
)
from pe_mechanism.manifest import ArtifactDigest, FileDigest, new_manifest
from pe_mechanism.representation import (
    save_representation_checkpoint,
    train_autoencoder,
)


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
    raw_checkpoint = tmp_path / "raw-model.ckpt"
    raw_checkpoint.write_bytes(b"raw model checkpoint")
    dataset_manifest = tmp_path / "dataset-manifest.json"
    dataset_manifest.write_text(
        '{"datasets":["toy-dataset","a","b"]}\n', encoding="utf-8"
    )
    return {
        "model_family": "tabicl_v2",
        "model_revision": "exploratory_step_1",
        "condition": "none",
        "sites": ["row_block_0"],
        "checkpoint_path": str(raw_checkpoint),
        "dataset_manifest_path": str(dataset_manifest),
        "training_code_root": str(repository),
        "model_code_root": str(repository),
        "analysis_code_root": str(repository),
    }


def _digest(path: Path) -> FileDigest:
    raw = path.read_bytes()
    return FileDigest(hashlib.sha256(raw).hexdigest(), len(raw))


def _representation_run(
    tmp_path: Path,
    training_result: object,
    provenance: dict[str, object],
    *,
    command: str = "train-repr",
) -> Path:
    run_dir = tmp_path / "representation-run"
    run_dir.mkdir()
    model_path = run_dir / "model.pt"
    save_representation_checkpoint(model_path, training_result)
    model_digest = _digest(model_path)
    parent_config = tmp_path / "representation-config.json"
    parent_config.write_text('{"schema_version":1}\n', encoding="utf-8")
    repository = Path(str(provenance["analysis_code_root"]))
    source_sha = _git(repository, "rev-parse", "HEAD")
    manifest = new_manifest(
        command=command,
        model_family=str(provenance["model_family"]),
        model_revision=str(provenance["model_revision"]),
        training_code_sha=source_sha,
        model_code_sha=source_sha,
        analysis_code_sha=source_sha,
        configuration=_digest(parent_config),
        checkpoint=_digest(Path(str(provenance["checkpoint_path"]))),
        dataset_manifest=_digest(Path(str(provenance["dataset_manifest_path"]))),
        condition=str(provenance["condition"]),
        sites=tuple(provenance["sites"]),
        seed=7,
        artifacts=(
            ArtifactDigest(
                name="model.pt",
                sha256=model_digest.sha256,
                size_bytes=model_digest.size_bytes,
            ),
        ),
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), sort_keys=True), encoding="utf-8"
    )
    return run_dir


def test_noop_baseline_and_paired_interventions_do_not_mutate_source() -> None:
    latents = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    original = latents.clone()
    paired = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    no_op = intervene_latents(latents, [1], mode="no-op")
    baseline = intervene_latents(
        latents, [0, 2], mode="baseline", baseline=torch.tensor([0.5, 1.5, 2.5])
    )
    replaced = intervene_latents(latents, [1], mode="paired", paired_latents=paired)

    assert torch.equal(no_op, latents)
    assert no_op.data_ptr() != latents.data_ptr()
    assert torch.equal(baseline[:, [0, 2]], torch.tensor([[0.5, 2.5], [0.5, 2.5]]))
    assert torch.equal(replaced[:, 1], paired[:, 1])
    assert torch.equal(latents, original)


def test_frequency_and_norm_matched_control_is_deterministic_and_excludes_targets() -> None:
    frequencies = torch.tensor([0.10, 0.11, 0.80, 0.81, 0.40, 0.41])
    norms = torch.tensor([1.00, 1.02, 4.00, 3.95, 2.00, 2.04])

    first = matched_random_control_features(
        [0, 2], frequencies, norms, seed=8, candidate_pool_size=1
    )
    second = matched_random_control_features(
        [0, 2], frequencies, norms, seed=8, candidate_pool_size=1
    )

    assert first == second == [1, 3]
    assert not ({0, 2} & set(first))


def test_activation_frequency_counts_nonzero_rows() -> None:
    latents = torch.tensor([[0.0, 1.0], [2.0, 0.0], [3.0, 4.0]])
    assert torch.allclose(activation_frequency(latents), torch.tensor([2 / 3, 2 / 3]))


def test_paired_effect_records_and_summary_preserve_pairing() -> None:
    records = paired_effect_records(
        [1.0, 2.0],
        [1.5, 1.0],
        condition="feature_baseline",
        sample_ids=["a", "b"],
        target_features=[3],
        outcome_name="error",
        metadata={"split": "validation"},
    )

    assert [record["sample_id"] for record in records] == ["a", "b"]
    assert [record["effect"] for record in records] == pytest.approx([0.5, -1.0])
    assert all(record["target_features"] == [3] for record in records)
    assert all(record["metadata"] == {"split": "validation"} for record in records)
    summary = summarize_effect_records(records)
    assert summary["feature_baseline"]["count"] == 2
    assert summary["feature_baseline"]["mean_effect"] == pytest.approx(-0.25)


def test_no_op_reconstruction_matches_direct_model_forward() -> None:
    result = train_autoencoder(
        torch.randn(20, 4, generator=torch.Generator().manual_seed(2)),
        model_config={"type": "dense", "latent_dim": 3},
        epochs=1,
        batch_size=5,
    )
    inputs = torch.randn(6, 4)
    expected_reconstruction, expected_latents = result.model(inputs)
    reconstruction, latents = no_op_reconstruction(result.model, inputs)

    assert torch.equal(latents, expected_latents)
    assert torch.equal(reconstruction, expected_reconstruction)


def test_causal_run_writes_noop_target_pair_and_matched_control_records(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = torch.randn(18, 4, generator=torch.Generator().manual_seed(11))
    training_result = train_autoencoder(
        values,
        validation_activations=values.clone(),
        model_config={"type": "pca", "latent_dim": 4},
        epochs=1,
        batch_size=6,
        seed=7,
    )
    provenance = _provenance(tmp_path, monkeypatch)
    representation_run = _representation_run(tmp_path, training_result, provenance)
    np.save(tmp_path / "activations.npy", values.numpy())
    np.save(tmp_path / "paired.npy", values.flip(0).numpy())
    config = {
        "representation_run_dir": str(representation_run),
        "activations": "activations.npy",
        "paired_activations": "paired.npy",
        "dataset_id": "toy-dataset",
        "paired_dataset_id": "toy-dataset",
        "sample_ids": list(range(len(values))),
        "paired_sample_ids": list(range(len(values))),
        "features": [0],
        "baseline": "mean",
        "random_control": {"candidate_pool_size": 2},
        "seed": 13,
        "provenance": provenance,
    }
    config_path = tmp_path / "causal.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = tmp_path / "causal-output"

    assert run(SimpleNamespace(config=config_path, output_dir=output_dir)) == 0
    records = [json.loads(line) for line in (output_dir / "effects.jsonl").read_text().splitlines()]
    conditions = {record["condition"] for record in records}

    assert conditions == {
        "no_op",
        "feature_baseline",
        "paired_replacement",
        "matched_random_control",
    }
    assert len(records) == 4 * len(values)
    assert all(record["effect"] == 0.0 for record in records if record["condition"] == "no_op")
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["command"] == "reconstruction-sensitivity"
    activation_binding = hashlib.sha256(
        json.dumps(
            {
                "dataset_id": "toy-dataset",
                "sample_ids": list(range(len(values))),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert {item["role"] for item in manifest["inputs"]} == {
        f"activations.paired.{activation_binding}",
        f"activations.primary.{activation_binding}",
        "representation.model",
        "representation.parent_manifest",
    }
    assert not list(output_dir.glob(".*.tmp"))


def test_invalid_features_and_nonfinite_effects_fail_closed() -> None:
    latents = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="unique"):
        intervene_latents(latents, [1, 1], mode="baseline")
    with pytest.raises(IndexError):
        intervene_latents(latents, [3], mode="baseline")
    with pytest.raises(ValueError, match="finite"):
        paired_effect_records([0.0], [float("nan")], condition="bad")


def test_reconstruction_run_rejects_empty_features_and_unpaired_identity(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = torch.randn(8, 2, generator=torch.Generator().manual_seed(1))
    result = train_autoencoder(
        values,
        validation_activations=values,
        model_config={"type": "pca", "latent_dim": 2},
        epochs=1,
        batch_size=4,
    )
    provenance = _provenance(tmp_path, monkeypatch)
    representation_run = _representation_run(tmp_path, result, provenance)
    np.save(tmp_path / "values.npy", values.numpy())
    base = {
        "representation_run_dir": str(representation_run),
        "activations": "values.npy",
        "dataset_id": "a",
        "sample_ids": list(range(len(values))),
        "provenance": provenance,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="at least one"):
        run(SimpleNamespace(config=config_path, output_dir=tmp_path / "empty"))

    base.update(
        {
            "features": [0],
            "paired_activations": "values.npy",
            "paired_dataset_id": "b",
            "paired_sample_ids": list(range(len(values))),
        }
    )
    config_path.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="same dataset"):
        run(SimpleNamespace(config=config_path, output_dir=tmp_path / "mismatch"))


def test_reconstruction_rejects_unverified_representation_parent(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = torch.randn(6, 2, generator=torch.Generator().manual_seed(3))
    result = train_autoencoder(
        values,
        validation_activations=values,
        model_config={"type": "pca", "latent_dim": 2},
        epochs=1,
        batch_size=3,
    )
    provenance = _provenance(tmp_path, monkeypatch)
    representation_run = _representation_run(
        tmp_path, result, provenance, command="collect"
    )
    np.save(tmp_path / "values.npy", values.numpy())
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "representation_run_dir": str(representation_run),
                "activations": "values.npy",
                "dataset_id": "a",
                "sample_ids": list(range(len(values))),
                "features": [0],
                "provenance": provenance,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="command must be train-repr"):
        run(SimpleNamespace(config=config_path, output_dir=tmp_path / "rejected"))
