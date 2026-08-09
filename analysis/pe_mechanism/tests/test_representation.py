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
from pe_mechanism.manifest import (
    ArtifactDigest,
    FileDigest,
    InputDigest,
    new_manifest,
)
from pe_mechanism.representation import (
    DenseAutoencoder,
    MeanRMSNormalizer,
    PCARepresentation,
    TopKSparseAutoencoder,
    balance_condition_activations,
    load_representation_checkpoint,
    reconstruction_metrics,
    representation_qualification,
    run,
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
    (repository / "tracked.txt").write_text("current\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-q", "-m", "current")
    monkeypatch.setattr(provenance_module, "__file__", str(repository / "tracked.txt"))
    raw_checkpoint = tmp_path / "raw-model.ckpt"
    raw_checkpoint.write_bytes(b"raw model checkpoint")
    dataset_manifest = tmp_path / "dataset-manifest.json"
    dataset_manifest.write_text(
        json.dumps(
            {
                "assignments": [
                    {"name": "train-a", "split": "discovery"},
                    {"name": "valid-a", "split": "validation"},
                ]
            }
        ),
        encoding="utf-8",
    )
    return {
        "model_family": "tabicl-v2",
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


def _artifact_digest(path: Path) -> ArtifactDigest:
    digest = _digest(path)
    return ArtifactDigest(path.name, digest.sha256, digest.size_bytes)


def _strict_collect_run(
    tmp_path: Path,
    provenance: dict[str, object],
    *,
    condition: str,
    train_values: np.ndarray,
    validation_values: np.ndarray,
) -> tuple[Path, dict[str, str]]:
    run_dir = tmp_path / f"collect-{condition}"
    run_dir.mkdir()
    artifact_names = {
        "train-a": f"activation-{condition}-train.npz",
        "valid-a": f"activation-{condition}-valid.npz",
    }
    arrays = {"train-a": train_values, "valid-a": validation_values}
    for dataset_id, values in arrays.items():
        with (run_dir / artifact_names[dataset_id]).open("wb") as handle:
            np.savez_compressed(handle, activations=values)
    site = str(provenance["sites"][0])  # type: ignore[index]
    index = {
        "schema_version": 1,
        "kind": "bounded_activation_index",
        "seed": 42,
        "datasets": ["train-a", "valid-a"],
        "sites": [
            {
                "site": site,
                "axis_names": ["row", "embedding"],
                "feature_group_maps": {},
                "datasets": [
                    {
                        "dataset_id": dataset_id,
                        "file": artifact_names[dataset_id],
                        "feature_dim": int(arrays[dataset_id].shape[-1]),
                        "dtype": str(arrays[dataset_id].dtype),
                        "seen_vectors": int(arrays[dataset_id].shape[0]),
                        "retained_vectors": int(arrays[dataset_id].shape[0]),
                    }
                    for dataset_id in ("train-a", "valid-a")
                ],
            }
        ],
    }
    (run_dir / "activation-index.json").write_text(
        json.dumps(index, sort_keys=True), encoding="utf-8"
    )
    source_inputs: list[InputDigest] = []
    for index_number, (dataset_id, values) in enumerate(arrays.items()):
        source = tmp_path / f"source-{condition}-{dataset_id}.npy"
        np.save(source, values)
        source_digest = _digest(source)
        source_inputs.append(
            InputDigest(
                role=f"activation.{index_number:06d}",
                sha256=source_digest.sha256,
                size_bytes=source_digest.size_bytes,
            )
        )
    collect_config = tmp_path / f"collect-{condition}.json"
    collect_config.write_text(
        json.dumps({"condition": condition, "seed": 42}), encoding="utf-8"
    )
    repository = Path(str(provenance["analysis_code_root"]))
    head = _git(repository, "rev-parse", "HEAD")
    artifact_digests = tuple(
        sorted(
            (_artifact_digest(path) for path in run_dir.iterdir()),
            key=lambda item: item.name,
        )
    )
    manifest = new_manifest(
        command="collect",
        model_family=str(provenance["model_family"]),
        model_revision=str(provenance["model_revision"]),
        training_code_sha=head,
        model_code_sha=head,
        analysis_code_sha=_git(repository, "rev-parse", "HEAD^"),
        configuration=_digest(collect_config),
        checkpoint=_digest(Path(str(provenance["checkpoint_path"]))),
        dataset_manifest=_digest(Path(str(provenance["dataset_manifest_path"]))),
        inputs=tuple(source_inputs),
        condition=condition,
        sites=(site,),
        seed=42,
        artifacts=artifact_digests,
        evidence_level="strict",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), sort_keys=True), encoding="utf-8"
    )
    return run_dir, artifact_names


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _strict_official_collect_run(
    tmp_path: Path,
    provenance: dict[str, object],
    *,
    condition: str,
    split: str,
    dataset_id: str,
    values: np.ndarray,
    checkpoint: Path,
    coordinate_variant: int = 0,
    raw_feature_count: int = 2,
    activation_token_count: int = 2,
    coordinate_token: int = 0,
) -> tuple[Path, dict[str, str]]:
    assignment_split = "discovery" if split == "training" else "validation"
    run_dir = tmp_path / f"official-collect-{condition}-{split}"
    run_dir.mkdir()
    artifact_name = f"activation-{condition}-{split}.npz"
    rows = int(values.shape[0])
    call_index = np.zeros(rows, dtype=np.int32)
    coordinates = np.column_stack(
        (
            np.zeros(rows, dtype=np.int64),
            np.arange(rows, dtype=np.int64),
            np.full(rows, coordinate_token, dtype=np.int64),
        )
    )
    with (run_dir / artifact_name).open("wb") as handle:
        np.savez_compressed(
            handle,
            activations=values,
            call_index=call_index,
            axis_coordinates=coordinates,
        )
    repository = Path(str(provenance["analysis_code_root"]))
    head = _git(repository, "rev-parse", "HEAD")
    logical_names = [
        "N_test.npy",
        "N_train.npy",
        "N_val.npy",
        "info.json",
        "y_test.npy",
        "y_train.npy",
        "y_val.npy",
    ]
    logical_hashes = {
        name: hashlib.sha256(f"{dataset_id}:{name}".encode()).hexdigest()
        for name in logical_names
    }
    inputs = [
        {
            "logical_name": name,
            "role": f"talent.0000.{name.removesuffix('.npy').replace('.', '_')}",
            "sha256": logical_hashes[name],
            "size_bytes": len(f"{dataset_id}:{name}".encode()),
        }
        for name in logical_names
    ]
    parent_inputs = tuple(
        InputDigest(
            role=item["role"],
            sha256=item["sha256"],
            size_bytes=item["size_bytes"],
        )
        for item in inputs
    )
    site = str(provenance["sites"][0])  # type: ignore[index]
    local_group_map = [
        [(index + coordinate_variant) % raw_feature_count]
        for index in range(raw_feature_count)
    ]
    cls_token_count = activation_token_count - len(local_group_map)
    if cls_token_count < 0:
        raise ValueError("test activation token count is smaller than its group map")
    forward_calls = [
        {
            "call_index": 0,
            "norm_method": "none",
            "norm_view_indices": [0],
            "ensemble_indices": [0],
            "feature_shuffles": [list(range(raw_feature_count))],
            "class_shuffles": [[0, 1]],
            "raw_input_shape": [1, rows, raw_feature_count],
            "train_size": 1,
            "preprocessing_view_id": "official-tabicl/none/raw-call-0000",
            "post_filter_feature_group_maps": [local_group_map],
            "feature_coordinate_system": (
                "official-encoded-after-constant-filter-before-view-shuffle"
            ),
        }
    ]
    trace_sha = hashlib.sha256(f"trace:{condition}:{split}".encode()).hexdigest()
    dataset_entry = {
        "dataset_id": dataset_id,
        "task_type": "classification",
        "assignment_split": assignment_split,
        "evaluation_split": "val",
        "fit_context": "train",
        "n_numeric_features": 1,
        "n_categorical_features": 0,
        "info_sha256": logical_hashes["info.json"],
        "inputs": inputs,
        "input_bundle_sha256": _canonical_sha256(logical_hashes),
        "preprocessing_trace_sha256": trace_sha,
        "sample_roster_sha256": hashlib.sha256(
            f"roster:{dataset_id}:val".encode()
        ).hexdigest(),
        "probabilities_sha256": hashlib.sha256(
            f"probabilities:{condition}:{dataset_id}".encode()
        ).hexdigest(),
        "metrics": {"accuracy": 0.5, "log_loss": 0.7, "n_samples": rows},
        "exact_baseline_verified": True,
        "feature_group_mode": "same",
        "local_feature_group_map": local_group_map,
        "official_forward_calls": forward_calls,
    }
    axis_names = ["table", "row", "feature_group_or_cls", "embedding"]
    site_entry = {
        "dataset_id": dataset_id,
        "file": artifact_name,
        "axis_names": axis_names,
        "activation_shapes": [
            [1, rows, activation_token_count, int(values.shape[1])]
        ],
        "coordinate_axis_names": axis_names[:-1],
        "vector_axis_name": "embedding",
        "feature_group_token_offset": cls_token_count,
        "cls_token_count": cls_token_count,
        "feature_dim": int(values.shape[1]),
        "dtype": str(values.dtype),
        "seen_vectors": rows * activation_token_count,
        "retained_vectors": rows,
        "preprocessing_trace_sha256": trace_sha,
    }
    inference_contract = {
        "protocol": "official-tabicl-sklearn-raw-cache-free-v1",
        "model_code_sha": head,
        "fit_context": "train",
        "estimator_options": {"random_state": 42},
        "locked_options": {
            "allow_auto_download": False,
            "feature_group": "same",
            "kv_cache": False,
            "support_many_classes": False,
        },
    }
    index = {
        "schema_version": 1,
        "kind": "official_tabicl_bounded_activation_index",
        "model_family": provenance["model_family"],
        "model_revision": provenance["model_revision"],
        "condition": condition,
        "checkpoint_sha256": _digest(checkpoint).sha256,
        "model_code_sha": head,
        "seed": 42,
        "assignment_split": assignment_split,
        "evaluation_split": "val",
        "fit_context": "train",
        "inference_contract_sha256": _canonical_sha256(inference_contract),
        "max_classes": 10,
        "datasets": [dataset_entry],
        "sites": [{"site": site, "axis_names": axis_names, "datasets": [site_entry]}],
    }
    (run_dir / "activation-index.json").write_text(
        json.dumps(index, sort_keys=True), encoding="utf-8"
    )
    collect_config = tmp_path / f"official-collect-{condition}-{split}.json"
    collect_config.write_text(
        json.dumps({"condition": condition, "split": split, "seed": 42}),
        encoding="utf-8",
    )
    artifact_digests = tuple(
        sorted(
            (_artifact_digest(path) for path in run_dir.iterdir()),
            key=lambda item: item.name,
        )
    )
    manifest = new_manifest(
        command="collect",
        model_family=str(provenance["model_family"]),
        model_revision=str(provenance["model_revision"]),
        training_code_sha=head,
        model_code_sha=head,
        analysis_code_sha=_git(repository, "rev-parse", "HEAD^"),
        configuration=_digest(collect_config),
        checkpoint=_digest(checkpoint),
        dataset_manifest=_digest(Path(str(provenance["dataset_manifest_path"]))),
        inputs=parent_inputs,
        condition=condition,
        sites=(site,),
        seed=42,
        artifacts=artifact_digests,
        evidence_level="strict",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), sort_keys=True), encoding="utf-8"
    )
    return run_dir, {dataset_id: artifact_name}


def _strict_lineage_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    raw_feature_count: int = 2,
    activation_token_count: int = 2,
    coordinate_token: int = 0,
) -> tuple[dict[str, object], dict[str, tuple[Path, dict[str, str]]]]:
    provenance = _provenance(tmp_path, monkeypatch)
    activations = np.random.default_rng(4).normal(size=(32, 4)).astype(np.float32)
    rope_checkpoint = tmp_path / "raw-model-rope.ckpt"
    rope_checkpoint.write_bytes(b"distinct rope model checkpoint")
    checkpoints = {
        "none": Path(str(provenance["checkpoint_path"])),
        "rope": rope_checkpoint,
    }
    runs = {
        "none": _strict_official_collect_run(
            tmp_path,
            provenance,
            condition="none",
            split="training",
            dataset_id="train-a",
            values=activations,
            checkpoint=checkpoints["none"],
            raw_feature_count=raw_feature_count,
            activation_token_count=activation_token_count,
            coordinate_token=coordinate_token,
        ),
        "none-validation": _strict_official_collect_run(
            tmp_path,
            provenance,
            condition="none",
            split="validation",
            dataset_id="valid-a",
            values=activations[:12].copy() + 10.0,
            checkpoint=checkpoints["none"],
            raw_feature_count=raw_feature_count,
            activation_token_count=activation_token_count,
            coordinate_token=coordinate_token,
        ),
        "rope": _strict_official_collect_run(
            tmp_path,
            provenance,
            condition="rope",
            split="training",
            dataset_id="train-a",
            values=activations + 0.1,
            checkpoint=checkpoints["rope"],
            raw_feature_count=raw_feature_count,
            activation_token_count=activation_token_count,
            coordinate_token=coordinate_token,
        ),
        "rope-validation": _strict_official_collect_run(
            tmp_path,
            provenance,
            condition="rope",
            split="validation",
            dataset_id="valid-a",
            values=activations[:12].copy() + 10.1,
            checkpoint=checkpoints["rope"],
            raw_feature_count=raw_feature_count,
            activation_token_count=activation_token_count,
            coordinate_token=coordinate_token,
        ),
    }

    def reference(condition: str, dataset_id: str, split: str) -> dict[str, str]:
        key = condition if split == "training" else f"{condition}-validation"
        run_dir, artifacts = runs[key]
        return {
            "run_dir": str(run_dir),
            "artifact": artifacts[dataset_id],
            "key": "activations",
        }

    config: dict[str, object] = {
        "reference_condition": "none",
        "training_by_condition": {
            condition: {"train-a": reference(condition, "train-a", "training")}
            for condition in ("none", "rope")
        },
        "validation_by_condition": {
            condition: {"valid-a": reference(condition, "valid-a", "validation")}
            for condition in ("none", "rope")
        },
        "model": {"type": "topk", "expansion_factor": 8, "top_k": 16},
        "training": {"epochs": 1, "batch_size": 8, "seed": 42},
        "provenance": provenance,
    }
    return config, runs


def _rewrite_manifest(run_dir: Path, **updates: object) -> None:
    path = run_dir / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _mutate_index_and_reseal(run_dir: Path, mutation) -> None:
    index_path = run_dir / "activation-index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    mutation(index)
    index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
    digest = _digest(index_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = next(
        item for item in manifest["artifacts"] if item["name"] == "activation-index.json"
    )
    artifact["sha256"] = digest.sha256
    artifact["size_bytes"] = digest.size_bytes
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def _mutate_activation_and_reseal(
    run_dir: Path, artifact_name: str, mutation
) -> None:
    artifact_path = run_dir / artifact_name
    with np.load(artifact_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    mutation(arrays)
    with artifact_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    digest = _digest(artifact_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = next(
        item for item in manifest["artifacts"] if item["name"] == artifact_name
    )
    artifact["sha256"] = digest.sha256
    artifact["size_bytes"] = digest.size_bytes
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def test_mean_rms_normalizer_centers_scales_and_round_trips() -> None:
    values = torch.tensor(
        [[1.0, 2.0, 7.0], [3.0, 4.0, 7.0], [5.0, 6.0, 7.0]]
    )
    normalizer = MeanRMSNormalizer.fit(values)
    normalized = normalizer.normalize(values)

    assert torch.allclose(normalized.mean(0), torch.zeros(3), atol=1e-6)
    assert torch.allclose(normalized[:, :2].square().mean(0), torch.ones(2), atol=1e-6)
    assert torch.equal(normalized[:, 2], torch.zeros(3))
    assert torch.allclose(normalizer.denormalize(normalized), values)


@pytest.mark.parametrize("top_k", [16, 32, 64])
def test_topk_sparse_autoencoder_is_8d_and_never_exceeds_k(top_k: int) -> None:
    model = TopKSparseAutoencoder(8, expansion_factor=8, top_k=top_k)
    reconstruction, latents = model(torch.randn(5, 8))

    assert reconstruction.shape == (5, 8)
    assert latents.shape == (5, 64)
    assert torch.all((latents != 0).sum(dim=1) <= top_k)


def test_topk_default_remains_valid_for_tiny_unit_test_dimensions() -> None:
    model = TopKSparseAutoencoder(2)
    assert model.latent_dim == 16
    assert model.top_k == 16


def test_reconstruction_metrics_report_quality_and_feature_use() -> None:
    values = torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, -1.0]])
    latents = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0]])
    metrics = reconstruction_metrics(values, values.clone(), latents)

    assert metrics["explained_variance"] == pytest.approx(1.0)
    assert metrics["normalized_mse"] == pytest.approx(0.0)
    assert metrics["dead_features"] == 1
    assert metrics["active_count"] == pytest.approx(4 / 3)


def test_training_is_deterministic_for_a_fixed_seed() -> None:
    generator = torch.Generator().manual_seed(9)
    values = torch.randn(48, 6, generator=generator)
    kwargs = {
        "model_config": {"type": "dense", "latent_dim": 4},
        "epochs": 3,
        "batch_size": 12,
        "learning_rate": 2e-3,
        "seed": 17,
    }

    first = train_autoencoder(values, **kwargs)
    second = train_autoencoder(values, **kwargs)

    assert first.history == second.history
    assert first.metrics == second.metrics
    for key, value in first.model.state_dict().items():
        assert torch.equal(value, second.model.state_dict()[key])


def test_validation_metrics_use_a_held_out_activation_set() -> None:
    training = torch.randn(32, 4, generator=torch.Generator().manual_seed(1))
    validation = torch.randn(11, 4, generator=torch.Generator().manual_seed(2)) + 2.0
    result = train_autoencoder(
        training,
        validation_activations=validation,
        model_config={"type": "dense", "latent_dim": 2},
        epochs=1,
        batch_size=8,
        seed=3,
    )
    assert result.metric_split == "validation"
    assert result.metrics != result.training_metrics
    qualification = representation_qualification(result)
    assert qualification["held_out_validation"] is True
    assert qualification["native_score_gate"] == "pending"


def test_qualification_uses_the_worst_heldout_condition() -> None:
    values = torch.randn(24, 4, generator=torch.Generator().manual_seed(2))
    result = train_autoencoder(
        values,
        validation_activations=values[:8].clone(),
        model_config={"type": "pca", "latent_dim": 3},
        seed=42,
    )
    result.metrics["explained_variance"] = 0.99
    result.validation_metrics_by_condition = {
        "none": {"explained_variance": 0.99},
        "rope": {"explained_variance": 0.94},
    }

    qualification = representation_qualification(result)

    assert qualification["activation_fidelity_passed"] is False
    assert qualification["worst_condition_explained_variance"] == pytest.approx(0.94)
    assert set(qualification["validation_by_condition"]) == {"none", "rope"}


@pytest.mark.parametrize(
    "condition_metrics",
    [
        {},
        {"none": "forged"},
        {"none": {"explained_variance": 0.99}},
    ],
)
def test_loaded_qualification_fails_closed_on_missing_or_forged_condition_metrics(
    condition_metrics: object,
) -> None:
    metadata = {
        "metric_split": "validation",
        "metrics": {"explained_variance": 0.99},
        "validation_metrics_by_condition": condition_metrics,
        "metadata": {
            "source_lineage": {
                "condition_checkpoints_sha256": {
                    "none": "a" * 64,
                    "rope": "b" * 64,
                }
            }
        },
    }

    with pytest.raises(ValueError):
        representation_qualification(metadata)


def test_pca_is_deterministic_heldout_and_checkpoint_round_trips(tmp_path) -> None:
    basis = torch.tensor(
        [[1.0, 0.5, -0.25, 2.0, 0.0], [0.25, -1.0, 1.5, 0.0, 0.75]]
    )
    offset = torch.tensor([3.0, -2.0, 0.5, 4.0, -1.0])
    training_coefficients = torch.randn(
        64, 2, generator=torch.Generator().manual_seed(10)
    )
    validation_coefficients = torch.randn(
        21, 2, generator=torch.Generator().manual_seed(11)
    )
    training = training_coefficients @ basis + offset
    validation = validation_coefficients @ basis + offset
    kwargs = {
        "validation_activations": validation,
        "model_config": {"type": "pca", "latent_dim": 2},
        "seed": 19,
    }

    first = train_autoencoder(training, **kwargs)
    second = train_autoencoder(training, **kwargs)

    assert isinstance(first.model, PCARepresentation)
    assert first.history == []
    assert first.metric_split == "validation"
    assert first.metrics["explained_variance"] == pytest.approx(1.0, abs=1e-6)
    assert representation_qualification(first)["activation_fidelity_passed"] is True
    for key, value in first.model.state_dict().items():
        assert torch.equal(value, second.model.state_dict()[key])

    normalized_validation = first.normalizer.normalize(validation)
    reconstruction, latents = first.model(normalized_validation)
    assert torch.allclose(
        reconstruction,
        first.model.decode(first.model.encode(normalized_validation)),
    )
    assert torch.allclose(reconstruction, normalized_validation, atol=2e-6)
    assert latents.shape == (validation.shape[0], 2)

    checkpoint = tmp_path / "pca.pt"
    save_representation_checkpoint(checkpoint, first, metadata={"baseline": "pca"})
    loaded, loaded_normalizer, metadata = load_representation_checkpoint(checkpoint)
    loaded_inputs = loaded_normalizer.normalize(validation)
    loaded_reconstruction, loaded_latents = loaded(loaded_inputs)

    assert isinstance(loaded, PCARepresentation)
    assert torch.equal(loaded_reconstruction, reconstruction)
    assert torch.equal(loaded_latents, latents)
    assert metadata["model"]["solver"] == "full_svd"
    assert metadata["metadata"] == {"baseline": "pca"}


def test_pca_fails_closed_before_fit_and_for_unregistered_solver() -> None:
    from pe_mechanism.representation import build_autoencoder

    model = PCARepresentation(4, 2)
    with pytest.raises(RuntimeError, match="fitted"):
        model.encode(torch.ones(3, 4))
    with pytest.raises(ValueError, match="full_svd"):
        build_autoencoder(
            4,
            {"type": "pca", "latent_dim": 2, "solver": "randomized"},
        )


def _activation_block(base: float, rows: int) -> np.ndarray:
    return (
        np.arange(rows * 3, dtype=np.float32).reshape(rows, 3) + np.float32(base)
    )


def test_condition_balancing_has_equal_counts_and_is_order_invariant() -> None:
    training = {
        "rope": {
            "train-b": _activation_block(1_100, 5),
            "train-a": _activation_block(1_000, 6),
        },
        "none": {
            "train-b": _activation_block(100, 5),
            "train-a": _activation_block(0, 6),
        },
    }
    validation = {
        "rope": {
            "valid-b": _activation_block(11_100, 5),
            "valid-a": _activation_block(11_000, 4),
        },
        "none": {
            "valid-b": _activation_block(10_100, 5),
            "valid-a": _activation_block(10_000, 4),
        },
    }
    first = balance_condition_activations(
        training,
        validation,
        seed=71,
        training_rows_per_condition=7,
        validation_rows_per_condition=5,
    )
    reordered_training = {
        condition: dict(reversed(tuple(training[condition].items())))
        for condition in reversed(tuple(training))
    }
    reordered_validation = {
        condition: dict(reversed(tuple(validation[condition].items())))
        for condition in reversed(tuple(validation))
    }
    second = balance_condition_activations(
        reordered_training,
        reordered_validation,
        seed=71,
        training_rows_per_condition=7,
        validation_rows_per_condition=5,
    )

    assert first.condition_names == ("none", "rope")
    assert first.training_dataset_roster == ("train-a", "train-b")
    assert first.validation_dataset_roster == ("valid-a", "valid-b")
    assert torch.equal(torch.bincount(first.training_condition_indices), torch.tensor([7, 7]))
    assert torch.equal(torch.bincount(first.validation_condition_indices), torch.tensor([5, 5]))
    assert torch.equal(first.training, second.training)
    assert torch.equal(first.validation, second.validation)
    assert torch.equal(first.training_condition_indices, second.training_condition_indices)
    assert torch.equal(first.validation_condition_indices, second.validation_condition_indices)
    assert first.training_rows_by_dataset == (("train-a", 4), ("train-b", 3))
    assert first.validation_rows_by_dataset == (("valid-a", 3), ("valid-b", 2))
    assert first.training_selection_sha256 == second.training_selection_sha256
    assert first.validation_selection_sha256 == second.validation_selection_sha256
    # The same within-dataset rows are reused for every condition.  These toy
    # arrays differ only by a fixed condition offset, so any independently
    # sampled rows would break this exact relation.
    assert torch.equal(first.training[7:] - first.training[:7], torch.full((7, 3), 1_000.0))
    assert torch.equal(
        first.validation[5:] - first.validation[:5], torch.full((5, 3), 1_000.0)
    )
    assert float(first.training.max()) < 10_000
    assert float(first.validation.min()) >= 10_000


def test_default_condition_balancing_equalizes_dataset_contribution() -> None:
    training = {
        condition: {
            "train-a": _activation_block(offset, 7),
            "train-b": _activation_block(offset + 100, 3),
        }
        for condition, offset in (("none", 0), ("rope", 1_000))
    }
    validation = {
        condition: {
            "valid-a": _activation_block(offset + 10_000, 6),
            "valid-b": _activation_block(offset + 11_000, 2),
        }
        for condition, offset in (("none", 0), ("rope", 1_000))
    }

    balanced = balance_condition_activations(training, validation, seed=9)

    assert balanced.training_rows_by_dataset == (("train-a", 3), ("train-b", 3))
    assert balanced.validation_rows_by_dataset == (("valid-a", 2), ("valid-b", 2))
    assert balanced.training_rows_per_condition == 6
    assert balanced.validation_rows_per_condition == 4


def test_condition_balancing_rejects_condition_and_dataset_roster_mismatches() -> None:
    block = _activation_block(0, 4)
    training = {
        "none": {"train": block.copy()},
        "rope": {"train": block.copy()},
    }
    with pytest.raises(ValueError, match="condition rosters"):
        balance_condition_activations(
            training,
            {"none": {"valid": block.copy()}},
        )

    validation = {
        "none": {"valid-a": block.copy()},
        "rope": {"valid-b": block.copy()},
    }
    with pytest.raises(ValueError, match="dataset roster"):
        balance_condition_activations(training, validation)

    overlapping_validation = {
        "none": {"train": block.copy()},
        "rope": {"train": block.copy()},
    }
    with pytest.raises(ValueError, match="disjoint"):
        balance_condition_activations(training, overlapping_validation)


def test_pooled_condition_shorthand_cannot_bypass_split_guards() -> None:
    none_training = _activation_block(0, 6)
    rope_training = _activation_block(100, 6)
    pooled_training = {"none": none_training, "rope": rope_training}
    pooled_validation = {
        "none": _activation_block(1_000, 4),
        "rope": _activation_block(1_100, 4),
    }
    with pytest.raises(ValueError, match="explicit"):
        balance_condition_activations(pooled_training, pooled_validation)

    balanced = balance_condition_activations(
        pooled_training,
        pooled_validation,
        training_dataset_roster=("train",),
        validation_dataset_roster=("valid",),
    )
    assert balanced.training_rows_per_condition == 6
    assert balanced.validation_rows_per_condition == 4

    with pytest.raises(ValueError, match="share source memory"):
        balance_condition_activations(
            pooled_training,
            {"none": none_training, "rope": rope_training},
            training_dataset_roster=("train",),
            validation_dataset_roster=("valid",),
        )


def test_cli_model_builder_rejects_unregistered_sparse_k() -> None:
    from pe_mechanism.representation import build_autoencoder

    with pytest.raises(ValueError, match="top_k"):
        build_autoencoder(8, {"type": "topk", "top_k": 7})


def test_checkpoint_round_trip_is_complete_and_atomic(tmp_path) -> None:
    values = torch.randn(24, 5, generator=torch.Generator().manual_seed(3))
    result = train_autoencoder(
        values,
        model_config={"type": "topk", "expansion_factor": 8, "top_k": 16},
        epochs=1,
        batch_size=8,
    )
    checkpoint = tmp_path / "model.pt"
    save_representation_checkpoint(checkpoint, result, metadata={"seed": 42})
    model, normalizer, metadata = load_representation_checkpoint(checkpoint)

    assert isinstance(model, TopKSparseAutoencoder)
    assert torch.equal(normalizer.mean, result.normalizer.mean)
    assert metadata["metadata"] == {"seed": 42}
    assert "activation_fidelity_passed" in metadata["qualification"]
    assert not list(tmp_path.glob("*.tmp"))


def test_run_reads_json_and_publishes_only_to_validated_output(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _strict_lineage_config(tmp_path, monkeypatch)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = tmp_path / "external-output"

    assert run(SimpleNamespace(config=config_path, output_dir=output_dir)) == 0
    assert {path.name for path in output_dir.iterdir()} == {
        "history.json",
        "manifest.json",
        "metrics.json",
        "model.pt",
    }
    assert json.loads((output_dir / "manifest.json").read_text())["command"] == "train-repr"
    inputs = json.loads((output_dir / "manifest.json").read_text())["inputs"]
    assert len(inputs) == 12
    roles = [item["role"] for item in inputs]
    # Four completed official collect runs (two conditions by two assignment
    # splits) each bind their manifest and path-free index, while all four
    # activation artifacts remain individually bound.
    assert sum(role.startswith("source.collect_manifest.") for role in roles) == 4
    assert sum(role.startswith("source.collect_index.") for role in roles) == 4
    assert sum(role.startswith("source.activation.") for role in roles) == 4
    metrics = json.loads((output_dir / "metrics.json").read_text())
    lineage = metrics["source_lineage"]
    assert lineage["reference_condition"] == "none"
    assert lineage["condition_checkpoints_sha256"]["none"] != (
        lineage["condition_checkpoints_sha256"]["rope"]
    )
    assert lineage["inference_contract_sha256"] is not None
    assert set(lineage["alignment_sha256"]) == {"training", "validation"}
    assert not list(output_dir.glob(".*.tmp"))

    with pytest.raises(FileExistsError):
        run(SimpleNamespace(config=config_path, output_dir=output_dir))


def test_run_rejects_relative_output_before_writing(tmp_path, monkeypatch) -> None:
    config, _ = _strict_lineage_config(tmp_path, monkeypatch)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="absolute"):
        run(SimpleNamespace(config=config_path, output_dir="relative-output"))
    assert not (tmp_path / "relative-output").exists()


def test_run_requires_heldout_validation_activations(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    np.save(tmp_path / "activations.npy", np.ones((8, 2), dtype=np.float32))
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "training_by_condition": {"none": {"train-a": "activations.npy"}},
                "model": {"type": "dense", "latent_dim": 1},
                "training": {"epochs": 1},
                "provenance": _provenance(tmp_path, monkeypatch),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="validation_by_condition"):
        run(SimpleNamespace(config=config_path, output_dir=output))
    assert not output.exists()


def test_run_rejects_bare_activation_paths_for_strict_cli(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    activation = tmp_path / "activation.npy"
    np.save(activation, np.ones((8, 2), dtype=np.float32))
    config = {
        "training_by_condition": {"none": {"train-a": str(activation)}},
        "validation_by_condition": {"none": {"valid-a": str(activation)}},
        "reference_condition": "none",
        "model": {"type": "dense", "latent_dim": 1},
        "training": {"epochs": 1, "seed": 42},
        "provenance": _provenance(tmp_path, monkeypatch),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="completed collect run"):
        run(SimpleNamespace(config=config_path, output_dir=tmp_path / "output"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"command": "train-repr"}, "come from a collect run"),
        (
            {"evidence_level": "exploratory_legacy", "legacy_reasons": ["forged"]},
            "strict collect evidence",
        ),
    ],
)
def test_run_rejects_forged_collect_qualification(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, object],
    message: str,
) -> None:
    config, runs = _strict_lineage_config(tmp_path, monkeypatch)
    _rewrite_manifest(runs["none"][0], **mutation)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        run(SimpleNamespace(config=config_path, output_dir=tmp_path / "output"))


def test_run_rejects_forged_collect_raw_source_lineage(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, runs = _strict_lineage_config(tmp_path, monkeypatch)
    run_dir = runs["none"][0]
    manifest_path = run_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["inputs"][0]["role"] = "activation.000000.forged"
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="TALENT inputs differ"):
        run(SimpleNamespace(config=config_path, output_dir=tmp_path / "output"))


def test_run_rejects_collect_artifact_replacement(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, runs = _strict_lineage_config(tmp_path, monkeypatch)
    run_dir, artifacts = runs["none"]
    with (run_dir / artifacts["train-a"]).open("wb") as handle:
        np.savez_compressed(handle, activations=np.zeros((32, 4), dtype=np.float32))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="hash or size mismatch"):
        run(SimpleNamespace(config=config_path, output_dir=tmp_path / "output"))


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("condition", "condition does not match"),
        ("site", "sites do not match"),
        ("dataset", "artifact dataset mismatch"),
    ],
)
def test_run_rejects_collect_condition_site_and_dataset_mismatches(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
    message: str,
) -> None:
    config, runs = _strict_lineage_config(tmp_path, monkeypatch)
    run_dir, _ = runs["none"]
    if mismatch == "condition":
        _rewrite_manifest(run_dir, condition="temporary")
    elif mismatch == "site":
        _rewrite_manifest(run_dir, sites=["other_site"])
    else:
        training = config["training_by_condition"]
        assert isinstance(training, dict)
        none = training["none"]
        assert isinstance(none, dict)
        none["wrong-dataset"] = none.pop("train-a")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        run(SimpleNamespace(config=config_path, output_dir=tmp_path / "output"))


def test_run_rejects_collect_analysis_from_unrelated_history(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, runs = _strict_lineage_config(tmp_path, monkeypatch)
    _rewrite_manifest(runs["none"][0], analysis_code_sha="f" * 40)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="analysis ancestry"):
        run(SimpleNamespace(config=config_path, output_dir=tmp_path / "output"))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("extra", "fields mismatch"),
        ("condition", "index condition"),
        ("site", "index sites"),
        ("dataset", "unknown dataset"),
        ("checkpoint", "index checkpoint"),
        ("model_code", "index model code"),
    ],
)
def test_run_rejects_tampered_official_index_contract_fields(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    config, runs = _strict_lineage_config(tmp_path, monkeypatch)
    run_dir = runs["none"][0]

    def mutate(index: dict[str, object]) -> None:
        if case == "extra":
            index["unexpected"] = True
        elif case == "condition":
            index["condition"] = "temporary"
        elif case == "site":
            index["sites"][0]["site"] = "other_site"  # type: ignore[index]
        elif case == "dataset":
            index["datasets"][0]["dataset_id"] = "other-dataset"  # type: ignore[index]
        elif case == "checkpoint":
            index["checkpoint_sha256"] = "a" * 64
        else:
            index["model_code_sha"] = "f" * 40

    _mutate_index_and_reseal(run_dir, mutate)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        run(SimpleNamespace(config=config_path, output_dir=tmp_path / "output"))


def test_run_rejects_cross_condition_shuffle_metadata_mismatch(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, runs = _strict_lineage_config(tmp_path, monkeypatch)
    run_dir = runs["rope"][0]

    def mutate(index: dict[str, object]) -> None:
        call = index["datasets"][0]["official_forward_calls"][0]  # type: ignore[index]
        call["feature_shuffles"] = [[1, 0]]

    _mutate_index_and_reseal(run_dir, mutate)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="coordinates differ across conditions"):
        run(SimpleNamespace(config=config_path, output_dir=tmp_path / "output"))


def test_cls_coordinates_are_bounded_by_activation_shape_not_raw_features(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, runs = _strict_lineage_config(
        tmp_path,
        monkeypatch,
        raw_feature_count=3,
        activation_token_count=7,
        coordinate_token=6,
    )
    index = json.loads(
        (runs["none"][0] / "activation-index.json").read_text(encoding="utf-8")
    )
    site_entry = index["sites"][0]["datasets"][0]
    assert site_entry["activation_shapes"][0][2] == 7
    assert site_entry["feature_group_token_offset"] == 4
    assert site_entry["cls_token_count"] == 4
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert run(
        SimpleNamespace(config=config_path, output_dir=tmp_path / "output")
    ) == 0


def test_cls_coordinate_beyond_activation_shape_is_rejected(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, runs = _strict_lineage_config(
        tmp_path,
        monkeypatch,
        raw_feature_count=3,
        activation_token_count=7,
        coordinate_token=6,
    )
    run_dir, artifacts = runs["rope"]

    def overflow(arrays: dict[str, np.ndarray]) -> None:
        arrays["axis_coordinates"][:, 2] = 7

    _mutate_activation_and_reseal(
        run_dir, artifacts["train-a"], overflow
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="activation shape bound"):
        run(SimpleNamespace(config=config_path, output_dir=tmp_path / "output"))


def test_run_rejects_cross_parent_inference_contract_mismatch(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, runs = _strict_lineage_config(tmp_path, monkeypatch)
    run_dir = runs["rope"][0]
    _mutate_index_and_reseal(
        run_dir,
        lambda index: index.update({"inference_contract_sha256": "b" * 64}),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="inference contracts must match"):
        run(SimpleNamespace(config=config_path, output_dir=tmp_path / "output"))


def test_run_rejects_two_checkpoints_within_one_condition(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, runs = _strict_lineage_config(tmp_path, monkeypatch)
    run_dir = runs["rope-validation"][0]
    alternate = tmp_path / "alternate-rope.ckpt"
    alternate.write_bytes(b"another rope checkpoint")
    alternate_digest = _digest(alternate)
    _mutate_index_and_reseal(
        run_dir,
        lambda index: index.update(
            {"checkpoint_sha256": alternate_digest.sha256}
        ),
    )
    _rewrite_manifest(
        run_dir,
        checkpoint={
            "sha256": alternate_digest.sha256,
            "size_bytes": alternate_digest.size_bytes,
        },
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="within one condition"):
        run(SimpleNamespace(config=config_path, output_dir=tmp_path / "output"))


def test_run_rejects_outer_checkpoint_not_owned_by_reference_condition(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, runs = _strict_lineage_config(tmp_path, monkeypatch)
    rope_manifest = json.loads(
        (runs["rope"][0] / "manifest.json").read_text(encoding="utf-8")
    )
    wrong_checkpoint = tmp_path / "wrong-reference.ckpt"
    wrong_checkpoint.write_bytes(b"distinct rope model checkpoint")
    assert _digest(wrong_checkpoint).sha256 == rope_manifest["checkpoint"]["sha256"]
    provenance = config["provenance"]
    assert isinstance(provenance, dict)
    provenance["checkpoint_path"] = str(wrong_checkpoint)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="reference-condition checkpoint"):
        run(SimpleNamespace(config=config_path, output_dir=tmp_path / "output"))


def test_single_condition_generic_collect_remains_supported(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance = _provenance(tmp_path, monkeypatch)
    values = np.random.default_rng(8).normal(size=(16, 4)).astype(np.float32)
    run_dir, artifacts = _strict_collect_run(
        tmp_path,
        provenance,
        condition="none",
        train_values=values,
        validation_values=values[:8].copy() + 2,
    )
    config = {
        "reference_condition": "none",
        "training_by_condition": {
            "none": {
                "train-a": {
                    "run_dir": str(run_dir),
                    "artifact": artifacts["train-a"],
                    "key": "activations",
                }
            }
        },
        "validation_by_condition": {
            "none": {
                "valid-a": {
                    "run_dir": str(run_dir),
                    "artifact": artifacts["valid-a"],
                    "key": "activations",
                }
            }
        },
        "model": {"type": "dense", "latent_dim": 2},
        "training": {"epochs": 1, "batch_size": 8, "seed": 42},
        "provenance": provenance,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "generic-output"

    assert run(SimpleNamespace(config=config_path, output_dir=output)) == 0
    lineage = json.loads((output / "metrics.json").read_text())["source_lineage"]
    assert lineage["source_kind"] == "bounded_activation_index"
    assert lineage["inference_contract_sha256"] is None


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("root_unknown", "config fields mismatch"),
        ("topk_expansion", "expansion_factor=8"),
        ("topk_unknown", "topk evidence model fields mismatch"),
        ("seed", "seed must be one of"),
        ("training_unknown", "representation training fields mismatch"),
        ("sampling_unknown", "representation sampling fields mismatch"),
        ("dense_unknown", "dense baseline model fields mismatch"),
        ("pca_solver", "full_svd"),
    ],
)
def test_cli_representation_config_fails_closed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    config, _ = _strict_lineage_config(tmp_path, monkeypatch)
    if case == "root_unknown":
        config["unregistered"] = True
    elif case == "topk_expansion":
        config["model"]["expansion_factor"] = 4  # type: ignore[index]
    elif case == "topk_unknown":
        config["model"]["latent_dim"] = 32  # type: ignore[index]
    elif case == "seed":
        config["training"]["seed"] = 41  # type: ignore[index]
    elif case == "training_unknown":
        config["training"]["scheduler"] = "none"  # type: ignore[index]
    elif case == "sampling_unknown":
        config["sampling"] = {"unregistered": 1}
    elif case == "dense_unknown":
        config["model"] = {"type": "dense", "latent_dim": 2, "dropout": 0.1}
    else:
        config["model"] = {"type": "pca", "latent_dim": 2, "solver": "randomized"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        run(SimpleNamespace(config=config_path, output_dir=tmp_path / "output"))


def test_dense_autoencoder_exposes_encode_decode_contract() -> None:
    model = DenseAutoencoder(4, 2, activation="linear")
    inputs = torch.randn(3, 4)
    reconstruction, latents = model(inputs)
    assert torch.equal(latents, model.encode(inputs))
    assert torch.equal(reconstruction, model.decode(latents))
