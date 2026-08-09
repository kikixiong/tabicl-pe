from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pe_mechanism.provenance as provenance_module
import pytest
from pe_mechanism.adapters.base import ActivationRecord
from pe_mechanism.official_collect import run_official_collection
from pe_mechanism.official_tabicl import (
    OfficialForwardCapture,
    OfficialForwardMetadata,
    OfficialInferenceMetrics,
    OfficialInferenceResult,
)
from pe_mechanism.representation import run as run_representation


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _write_talent_dataset(
    root: Path,
    *,
    name: str = "toy",
    n_classes: int = 2,
) -> Path:
    dataset = root / name
    dataset.mkdir(parents=True)
    (dataset / "info.json").write_text(
        json.dumps({"name": name, "task_type": "classification"}),
        encoding="utf-8",
    )
    train_rows = max(6, n_classes)
    labels = np.arange(train_rows, dtype=np.int64) % n_classes
    split_labels = {
        "train": labels,
        "val": np.arange(4, dtype=np.int64) % n_classes,
        "test": np.arange(5, dtype=np.int64) % n_classes,
    }
    for split, y in split_labels.items():
        X = np.arange(y.shape[0] * 3, dtype=np.float64).reshape(-1, 3)
        X += {"train": 0.0, "val": 100.0, "test": 200.0}[split]
        np.save(dataset / f"N_{split}.npy", X)
        np.save(dataset / f"y_{split}.npy", y)
    return dataset


def _strict_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    checkpoint: Path,
    dataset_manifest: Path,
    condition: str = "none",
) -> dict[str, object]:
    repository = tmp_path / "verified-source"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.invalid")
    tracked = repository / "tracked.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", "tracked.py")
    _git(repository, "commit", "-q", "-m", "initial")
    monkeypatch.setattr(provenance_module, "__file__", str(tracked))
    head = _git(repository, "rev-parse", "HEAD")
    return {
        "model_family": "tabicl-v2",
        "model_revision": "step-210000",
        "condition": condition,
        "sites": ["row_interactor.blocks.0"],
        "checkpoint_path": str(checkpoint),
        "dataset_manifest_path": str(dataset_manifest),
        "training_code_root": str(repository),
        "model_code_root": str(repository),
        "analysis_code_root": str(repository),
        "expected_training_code_sha": head,
        "expected_model_code_sha": head,
        "expected_analysis_code_sha": head,
    }


class _FakeOfficialDriver:
    def __init__(
        self,
        *,
        model_sha: str,
        checkpoint_sha: str,
        mode: str = "none",
        feature_group: object = "same",
        cache: object = None,
        fail_prediction: bool = False,
        extra_cls_tokens: int = 0,
    ) -> None:
        raw = SimpleNamespace(
            _cache=cache,
            training=False,
            row_identity_mode=mode,
            row_interactor=SimpleNamespace(identity_mode=mode),
            col_embedder=SimpleNamespace(
                feature_group=feature_group,
                feature_group_size=2,
            ),
        )
        self.estimator = SimpleNamespace(
            model_=raw,
            kv_cache=False,
            model_kv_cache_=None,
            support_many_classes=False,
            n_classes_=2,
        )
        self.model_sha = model_sha
        self.checkpoint_sha = checkpoint_sha
        self.fit_context = "talent-train"
        self.source_evidence_level = "strict"
        self.fail_prediction = fail_prediction
        self.extra_cls_tokens = extra_cls_tokens
        self.last_prediction_rows: int | None = None

    def predict_proba(self, X, *, y, sites, require_exact_baseline):
        if self.fail_prediction:
            raise RuntimeError("synthetic prediction failure")
        assert require_exact_baseline is True
        assert tuple(sites) == ("row_interactor.blocks.0",)
        rows = len(X)
        self.last_prediction_rows = rows
        labels = np.asarray(y, dtype=np.int64)
        probabilities = np.full((rows, 2), 0.15, dtype=np.float64)
        probabilities[np.arange(rows), labels] = 0.85
        token_count = 3 + self.extra_cls_tokens
        activation = np.arange(
            2 * (rows + 6) * token_count * 4, dtype=np.float32
        ).reshape(
            2, rows + 6, token_count, 4
        )
        group_map = ((1, 2), (2, 0), (0, 1))
        metadata = OfficialForwardMetadata(
            call_index=0,
            norm_method="none",
            norm_view_indices=(0, 1),
            ensemble_indices=(0, 1),
            feature_shuffles=((0, 1, 2), (2, 1, 0)),
            class_shuffles=((0, 1), (1, 0)),
            raw_input_shape=(2, rows + 6, 3),
            train_size=6,
            preprocessing_view_id="official-tabicl/none/raw-call-0000",
            post_filter_feature_group_maps=(group_map, group_map[::-1]),
        )
        record = ActivationRecord(
            tensor=activation,
            site="row_interactor.blocks.0",
            axis_names=("table", "row", "feature_group_or_cls", "embedding"),
            shape=activation.shape,
            model_sha=self.model_sha,
            checkpoint_sha=self.checkpoint_sha,
            preprocessing_view_id=metadata.preprocessing_view_id,
            feature_group_map=group_map,
        )
        return OfficialInferenceResult(
            probabilities=probabilities,
            baseline_probabilities=probabilities.copy(),
            classes=np.asarray([0, 1]),
            forward_calls=(
                OfficialForwardCapture(
                    metadata=metadata,
                    activations={"row_interactor.blocks.0": record},
                ),
            ),
            metrics=OfficialInferenceMetrics(
                accuracy=1.0,
                log_loss=float(-np.log(0.85)),
                n_samples=rows,
            ),
            exact_baseline_verified=True,
            source_evidence_level="strict",
        )


class _FakeFactory:
    def __init__(
        self,
        *,
        mode: str = "none",
        feature_group: object = "same",
        cache: object = None,
        fail_prediction: bool = False,
        mutate_after_load: Path | None = None,
        extra_cls_tokens: int = 0,
    ) -> None:
        self.mode = mode
        self.feature_group = feature_group
        self.cache = cache
        self.fail_prediction = fail_prediction
        self.mutate_after_load = mutate_after_load
        self.extra_cls_tokens = extra_cls_tokens
        self.fit_contexts: list[str] = []
        self.training_rows: list[int] = []
        self.estimator_options: list[dict[str, object]] = []
        self.drivers: list[_FakeOfficialDriver] = []

    def __call__(
        self,
        dataset,
        checkpoint,
        *,
        context_split,
        device,
        model_sha,
        estimator_options,
        expected_source_root,
    ):
        del device
        assert expected_source_root.is_dir()
        self.fit_contexts.append(context_split)
        self.training_rows.append(len(dataset.train.X))
        self.estimator_options.append(dict(estimator_options or {}))
        if self.mutate_after_load is not None:
            self.mutate_after_load.write_bytes(b"changed after provenance verification")
        driver = _FakeOfficialDriver(
            model_sha=model_sha,
            checkpoint_sha=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            mode=self.mode,
            feature_group=self.feature_group,
            cache=self.cache,
            fail_prediction=self.fail_prediction,
            extra_cls_tokens=self.extra_cls_tokens,
        )
        self.drivers.append(driver)
        return driver


def _write_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    evaluation_split: str = "val",
    max_classes: int = 10,
    condition: str = "none",
    n_classes: int = 2,
) -> tuple[Path, Path, Path, Path]:
    dataset = _write_talent_dataset(
        tmp_path / "talent", name="toy", n_classes=n_classes
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"fixed checkpoint")
    dataset_manifest = tmp_path / "datasets.json"
    dataset_manifest.write_text(
        json.dumps({"assignments": [{"name": "toy", "split": "discovery"}]}),
        encoding="utf-8",
    )
    private_root = tmp_path / "private-study"
    private_root.mkdir()
    config = {
        "schema_version": 1,
        "private_study_root": str(private_root),
        "datasets": [{"dataset_id": "toy", "path": str(dataset)}],
        "roster_split": "discovery",
        "evaluation_split": evaluation_split,
        "seed": 17,
        "max_classes": max_classes,
        "max_vectors_per_dataset_site": 7,
        "dtype": "float16",
        "provenance": _strict_provenance(
            tmp_path,
            monkeypatch,
            checkpoint=checkpoint,
            dataset_manifest=dataset_manifest,
            condition=condition,
        ),
    }
    config_path = tmp_path / "official-collect.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, private_root, dataset, checkpoint


@pytest.fixture(autouse=True)
def ample_private_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    from pe_mechanism import collect

    monkeypatch.setattr(
        collect.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": 100 * 1024**3})(),
    )


def test_official_collection_is_train_only_bounded_deterministic_and_path_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, private_root, dataset, checkpoint = _write_config(tmp_path, monkeypatch)
    factory = _FakeFactory()
    first_output = private_root / "run-one"
    first = run_official_collection(config, first_output, driver_factory=factory)
    second_output = private_root / "run-two"
    second = run_official_collection(config, second_output, driver_factory=factory)

    assert factory.fit_contexts == ["train", "train"]
    assert factory.training_rows == [6, 6]
    assert factory.estimator_options == [
        {"random_state": 17},
        {"random_state": 17},
    ]
    assert [driver.last_prediction_rows for driver in factory.drivers] == [4, 4]
    assert first == second
    assert first["evaluation_split"] == "val"
    assert first["fit_context"] == "train"
    assert first["condition"] == "none"
    assert len(first["inference_contract_sha256"]) == 64
    dataset_entry = first["datasets"][0]
    assert dataset_entry["metrics"]["accuracy"] == 1.0
    assert dataset_entry["exact_baseline_verified"] is True
    assert dataset_entry["feature_group_mode"] == "same"
    assert len(dataset_entry["inputs"]) == 7
    assert all("path" not in item for item in dataset_entry["inputs"])
    assert dataset_entry["official_forward_calls"][0]["norm_method"] == "none"

    shard_entry = first["sites"][0]["datasets"][0]
    assert shard_entry["seen_vectors"] == 2 * 10 * 3
    assert shard_entry["retained_vectors"] == 7
    assert shard_entry["coordinate_axis_names"] == [
        "table",
        "row",
        "feature_group_or_cls",
    ]
    first_shard = np.load(first_output / shard_entry["file"], allow_pickle=False)
    second_shard = np.load(second_output / shard_entry["file"], allow_pickle=False)
    assert set(first_shard.files) == {
        "activations",
        "call_index",
        "axis_coordinates",
    }
    for name in first_shard.files:
        np.testing.assert_array_equal(first_shard[name], second_shard[name])
    assert first_shard["activations"].shape == (7, 4)
    assert first_shard["axis_coordinates"].shape == (7, 3)

    serialized = (first_output / "activation-index.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert str(dataset) not in serialized
    assert str(checkpoint) not in serialized
    manifest = json.loads((first_output / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["inputs"]) == 7
    assert all(item["role"].startswith("talent.0000.") for item in manifest["inputs"])
    assert all("/" not in item["role"] for item in manifest["inputs"])


def test_completed_official_collect_runs_feed_strict_representation_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    talent_root = tmp_path / "talent"
    train_dataset = _write_talent_dataset(talent_root, name="train-a")
    validation_dataset = _write_talent_dataset(talent_root, name="valid-a")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"fixed checkpoint")
    dataset_manifest = tmp_path / "datasets.json"
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
    provenance = _strict_provenance(
        tmp_path,
        monkeypatch,
        checkpoint=checkpoint,
        dataset_manifest=dataset_manifest,
        condition="none",
    )
    private_root = tmp_path / "private-study"
    private_root.mkdir()
    sources: dict[str, tuple[Path, str]] = {}
    for split, dataset in (
        ("discovery", train_dataset),
        ("validation", validation_dataset),
    ):
        config = {
            "schema_version": 1,
            "private_study_root": str(private_root),
            "datasets": [{"dataset_id": dataset.name, "path": str(dataset)}],
            "roster_split": split,
            "evaluation_split": "val",
            "seed": 42,
            "max_classes": 10,
            "max_vectors_per_dataset_site": 7,
            "dtype": "float16",
            "provenance": provenance,
        }
        config_path = tmp_path / f"official-{split}.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        output = private_root / f"official-{split}"
        index = run_official_collection(
            config_path, output, driver_factory=_FakeFactory()
        )
        sources[split] = (output, index["sites"][0]["datasets"][0]["file"])

    representation_config = {
        "reference_condition": "none",
        "training_by_condition": {
            "none": {
                "train-a": {
                    "run_dir": str(sources["discovery"][0]),
                    "artifact": sources["discovery"][1],
                    "key": "activations",
                }
            }
        },
        "validation_by_condition": {
            "none": {
                "valid-a": {
                    "run_dir": str(sources["validation"][0]),
                    "artifact": sources["validation"][1],
                    "key": "activations",
                }
            }
        },
        "model": {"type": "dense", "latent_dim": 2},
        "training": {"epochs": 1, "batch_size": 4, "seed": 42},
        "provenance": provenance,
    }
    representation_config_path = tmp_path / "representation.json"
    representation_config_path.write_text(
        json.dumps(representation_config), encoding="utf-8"
    )
    representation_output = tmp_path / "representation-output"

    assert run_representation(
        SimpleNamespace(
            config=representation_config_path,
            output_dir=representation_output,
        )
    ) == 0
    metrics = json.loads(
        (representation_output / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["source_lineage"]["source_kind"] == (
        "official_tabicl_bounded_activation_index"
    )


def test_mixed_case_site_roster_uses_the_canonical_dataset_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    talent_root = tmp_path / "talent"
    alpha = _write_talent_dataset(talent_root, name="Alpha")
    beta = _write_talent_dataset(talent_root, name="beta")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"fixed checkpoint")
    dataset_manifest = tmp_path / "datasets.json"
    dataset_manifest.write_text(
        json.dumps(
            {
                "assignments": [
                    {"name": "Alpha", "split": "discovery"},
                    {"name": "beta", "split": "discovery"},
                ]
            }
        ),
        encoding="utf-8",
    )
    private_root = tmp_path / "private-study"
    private_root.mkdir()
    config = {
        "schema_version": 1,
        "private_study_root": str(private_root),
        "datasets": [
            {"dataset_id": "beta", "path": str(beta)},
            {"dataset_id": "Alpha", "path": str(alpha)},
        ],
        "roster_split": "discovery",
        "evaluation_split": "val",
        "seed": 42,
        "max_vectors_per_dataset_site": 7,
        "dtype": "float16",
        "provenance": _strict_provenance(
            tmp_path,
            monkeypatch,
            checkpoint=checkpoint,
            dataset_manifest=dataset_manifest,
            condition="none",
        ),
    }
    config_path = tmp_path / "mixed-case.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    index = run_official_collection(
        config_path,
        private_root / "mixed-case",
        driver_factory=_FakeFactory(),
    )

    canonical = ["Alpha", "beta"]
    assert [item["dataset_id"] for item in index["datasets"]] == canonical
    for site in index["sites"]:
        assert [item["dataset_id"] for item in site["datasets"]] == canonical


def test_official_collection_records_cls_extended_activation_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, private_root, _, _ = _write_config(tmp_path, monkeypatch)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["max_vectors_per_dataset_site"] = 140
    config.write_text(json.dumps(payload), encoding="utf-8")
    output = private_root / "cls-extended"

    index = run_official_collection(
        config,
        output,
        driver_factory=_FakeFactory(extra_cls_tokens=4),
    )

    entry = index["sites"][0]["datasets"][0]
    assert index["datasets"][0]["official_forward_calls"][0][
        "raw_input_shape"
    ] == [2, 10, 3]
    assert entry["activation_shapes"] == [[2, 10, 7, 4]]
    assert entry["feature_group_token_offset"] == 4
    assert entry["cls_token_count"] == 4
    shard = np.load(output / entry["file"], allow_pickle=False)
    assert np.any(shard["axis_coordinates"][:, 2] >= 3)


@pytest.mark.parametrize("evaluation_split", ["train", "train+val", "validation"])
def test_evaluation_split_rejects_fit_rows_before_creating_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evaluation_split: str,
) -> None:
    config, private_root, _dataset, _checkpoint = _write_config(
        tmp_path, monkeypatch, evaluation_split=evaluation_split
    )
    output = private_root / "invalid"
    with pytest.raises(ValueError, match="evaluation_split"):
        run_official_collection(config, output, driver_factory=_FakeFactory())
    assert not output.exists()


def test_registered_class_limit_cannot_be_relaxed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, private_root, _dataset, _checkpoint = _write_config(
        tmp_path, monkeypatch, max_classes=11
    )
    with pytest.raises(ValueError, match="cannot exceed 10"):
        run_official_collection(
            config, private_root / "invalid", driver_factory=_FakeFactory()
        )


@pytest.mark.parametrize("random_state", [None, 18, True, "17"])
def test_official_random_state_must_equal_collection_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    random_state: object,
) -> None:
    config, private_root, _dataset, _checkpoint = _write_config(tmp_path, monkeypatch)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["estimator_options"] = {"random_state": random_state}
    config.write_text(json.dumps(payload), encoding="utf-8")
    output = private_root / "invalid-random-state"
    with pytest.raises(ValueError, match="random_state"):
        run_official_collection(config, output, driver_factory=_FakeFactory())
    assert not output.exists()


def test_inference_contract_changes_with_registered_estimator_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, private_root, _dataset, _checkpoint = _write_config(tmp_path, monkeypatch)
    baseline = run_official_collection(
        config, private_root / "baseline-contract", driver_factory=_FakeFactory()
    )
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["estimator_options"] = {"n_estimators": 4, "random_state": 17}
    config.write_text(json.dumps(payload), encoding="utf-8")
    changed = run_official_collection(
        config, private_root / "changed-contract", driver_factory=_FakeFactory()
    )
    assert baseline["inference_contract_sha256"] != changed["inference_contract_sha256"]


def test_dataset_above_class_limit_fails_before_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, private_root, _dataset, _checkpoint = _write_config(
        tmp_path, monkeypatch, n_classes=11
    )
    factory = _FakeFactory()
    with pytest.raises(ValueError, match="above max_classes=10"):
        run_official_collection(
            config, private_root / "invalid", driver_factory=factory
        )
    assert factory.fit_contexts == []


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (_FakeFactory(feature_group="separate"), "feature_group='same'"),
        (_FakeFactory(cache=object()), "raw-model caches"),
        (_FakeFactory(mode="rope"), "identity mode"),
    ],
)
def test_driver_safety_drift_fails_closed_and_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory: _FakeFactory,
    message: str,
) -> None:
    config, private_root, _dataset, _checkpoint = _write_config(tmp_path, monkeypatch)
    output = private_root / "invalid"
    with pytest.raises((ValueError, RuntimeError), match=message):
        run_official_collection(config, output, driver_factory=factory)
    assert not output.exists()


def test_prediction_failure_rolls_back_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, private_root, _dataset, _checkpoint = _write_config(tmp_path, monkeypatch)
    output = private_root / "failed-run"
    with pytest.raises(RuntimeError, match="synthetic prediction failure"):
        run_official_collection(
            config,
            output,
            driver_factory=_FakeFactory(fail_prediction=True),
        )
    assert not output.exists()
    assert not list(private_root.glob(".failed-run.*.staging"))


def test_consumed_talent_file_change_is_detected_before_atomic_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, private_root, dataset, _checkpoint = _write_config(tmp_path, monkeypatch)
    output = private_root / "changed-input"
    with pytest.raises((RuntimeError, ValueError), match="changed|SHA-256"):
        run_official_collection(
            config,
            output,
            driver_factory=_FakeFactory(mutate_after_load=dataset / "N_train.npy"),
        )
    assert not output.exists()


def test_dataset_manifest_split_is_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, private_root, _dataset, _checkpoint = _write_config(tmp_path, monkeypatch)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["roster_split"] = "held_out"
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="split mismatch"):
        run_official_collection(
            config, private_root / "wrong-roster", driver_factory=_FakeFactory()
        )


def test_test_split_is_allowed_but_never_added_to_fit_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, private_root, _dataset, _checkpoint = _write_config(
        tmp_path, monkeypatch, evaluation_split="test"
    )
    factory = _FakeFactory()
    index = run_official_collection(
        config, private_root / "test-evaluation", driver_factory=factory
    )
    assert factory.fit_contexts == ["train"]
    assert factory.training_rows == [6]
    assert factory.drivers[0].last_prediction_rows == 5
    assert index["datasets"][0]["metrics"]["n_samples"] == 5
