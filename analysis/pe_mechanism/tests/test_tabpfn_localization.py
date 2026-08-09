from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pe_mechanism.provenance as provenance_module
import pytest

from pe_mechanism.official_tabpfn import (
    OfficialTabPFNV26Result,
    TabPFNV26ConditionResult,
)
from pe_mechanism.provenance import VerifiedConfiguration, load_verified_json_config
from pe_mechanism.tabpfn_localization import (
    _array_sha256,
    _json_sha256,
    _validate_configuration,
    run_tabpfn_localization,
)


EXAMPLES = Path(__file__).parents[1] / "examples"


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _write_talent_dataset(root: Path, *, labels: tuple[str, str]) -> Path:
    dataset = root / "path-labels"
    dataset.mkdir(parents=True)
    (dataset / "info.json").write_text(
        json.dumps({"name": dataset.name, "task_type": "classification"}),
        encoding="utf-8",
    )
    split_labels = {
        "train": np.asarray(
            [labels[0], labels[1], labels[0], labels[1], labels[0], labels[1]],
            dtype=object,
        ),
        "val": np.asarray([labels[0], labels[1], labels[0]], dtype=object),
        "test": np.asarray([labels[1], labels[0]], dtype=object),
    }
    for offset, (split, y) in enumerate(split_labels.items()):
        X = np.arange(y.shape[0] * 2, dtype=np.float64).reshape(-1, 2)
        np.save(dataset / f"N_{split}.npy", X + offset * 100.0)
        np.save(dataset / f"y_{split}.npy", y)
    return dataset


def _strict_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    checkpoint: Path,
    dataset_manifest: Path,
) -> dict[str, object]:
    repository = tmp_path / "verified-source"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "TabPFN Tests")
    _git(repository, "config", "user.email", "tests@example.invalid")
    tracked = repository / "tracked.py"
    tracked.write_text("VALUE = 'strict'\n", encoding="utf-8")
    _git(repository, "add", "tracked.py")
    _git(repository, "commit", "-q", "-m", "strict source")
    head = _git(repository, "rev-parse", "HEAD")
    monkeypatch.setattr(provenance_module, "__file__", str(tracked))
    return {
        "model_family": "tabpfn-v2.6",
        "model_revision": "v7.1.1",
        "condition": "position-components",
        "sites": ["feature_positional_embedding"],
        "checkpoint_path": str(checkpoint),
        "dataset_manifest_path": str(dataset_manifest),
        "training_code_root": str(repository),
        "model_code_root": str(repository),
        "analysis_code_root": str(repository),
        "expected_checkpoint_sha256": _sha256(checkpoint),
        "expected_dataset_manifest_sha256": _sha256(dataset_manifest),
        "expected_training_code_sha": head,
        "expected_model_code_sha": head,
        "expected_analysis_code_sha": head,
        "allow_exploratory_legacy": False,
    }


class _FakeTabPFNDriver:
    def __init__(
        self,
        dataset,
        *,
        classes: np.ndarray,
        checkpoint_sha256: str,
        model_sha: str,
        seed: int,
        mutate_after_evaluate: Path | None = None,
        invalidate_full_gate: bool = False,
    ) -> None:
        self.dataset = dataset
        self.model_code_sha = model_sha
        self.loaded_checkpoint_sha256 = checkpoint_sha256
        self.seed = seed
        self.estimator = SimpleNamespace(
            n_estimators=1,
            random_state=seed,
            classes_=classes.copy(),
        )
        self._classes = classes.copy()
        self._mutate_after_evaluate = mutate_after_evaluate
        self._invalidate_full_gate = invalidate_full_gate
        self.evaluation_splits: list[str] = []

    def evaluate(self, split="test"):
        self.evaluation_splits.append(split)
        assert split == "val"
        labels = np.asarray(self.dataset.val.y)
        lookup = {value: index for index, value in enumerate(self._classes.tolist())}
        encoded = np.asarray([lookup[value] for value in labels], dtype=np.int64)

        def probabilities(confidence: float) -> np.ndarray:
            values = np.full(
                (encoded.shape[0], self._classes.shape[0]),
                (1.0 - confidence) / (self._classes.shape[0] - 1),
                dtype=np.float32,
            )
            values[np.arange(encoded.shape[0]), encoded] = confidence
            return values

        arrays = {
            "full": probabilities(0.90),
            "weight": probabilities(0.75),
            "bias": probabilities(0.60),
            "none": probabilities(0.35),
        }
        conditions = {
            name: _condition(name, values, encoded)
            for name, values in arrays.items()
        }
        result = OfficialTabPFNV26Result(
            dataset_name=self.dataset.name,
            split="val",
            classes=self._classes.copy(),
            native_probabilities=arrays["full"].copy(),
            conditions=conditions,
            model_code_sha=self.model_code_sha,
            loaded_checkpoint_sha256=self.loaded_checkpoint_sha256,
            seed=self.seed,
            exact_full_native_verified=not self._invalidate_full_gate,
            policies_restored_verified=True,
        )
        if self._mutate_after_evaluate is not None:
            self._mutate_after_evaluate.write_bytes(b"changed after evaluation")
        return result


def _condition(
    component: str, probabilities: np.ndarray, labels: np.ndarray
) -> TabPFNV26ConditionResult:
    selected = probabilities[np.arange(labels.shape[0]), labels].astype(np.float64)
    return TabPFNV26ConditionResult(
        component=component,
        probabilities=probabilities,
        accuracy=float(np.mean(np.argmax(probabilities, axis=1) == labels)),
        log_loss=float(-np.log(selected).mean()),
        n_samples=int(labels.shape[0]),
    )


class _FakeFactory:
    def __init__(
        self,
        *,
        classes: np.ndarray,
        mutate_after_evaluate: Path | None = None,
        invalidate_full_gate: bool = False,
    ) -> None:
        self.classes = classes
        self.mutate_after_evaluate = mutate_after_evaluate
        self.invalidate_full_gate = invalidate_full_gate
        self.calls: list[dict[str, object]] = []
        self.drivers: list[_FakeTabPFNDriver] = []

    def __call__(
        self,
        dataset,
        checkpoint,
        *,
        checkpoint_sha256,
        model_sha,
        expected_source_root,
        seed,
        device,
    ):
        assert Path(checkpoint).is_file()
        assert Path(expected_source_root).is_dir()
        self.calls.append(
            {
                "dataset": dataset,
                "checkpoint": Path(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "model_sha": model_sha,
                "expected_source_root": Path(expected_source_root),
                "seed": seed,
                "device": device,
                "training_rows": len(dataset.train.X),
                "validation_rows": len(dataset.val.X),
            }
        )
        driver = _FakeTabPFNDriver(
            dataset,
            classes=self.classes,
            checkpoint_sha256=checkpoint_sha256,
            model_sha=model_sha,
            seed=seed,
            mutate_after_evaluate=self.mutate_after_evaluate,
            invalidate_full_gate=self.invalidate_full_gate,
        )
        self.drivers.append(driver)
        return driver


def _write_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    missing_input_hash: str | None = None,
    roster_split: str = "discovery",
    evaluation_split: str = "val",
) -> SimpleNamespace:
    private_root = tmp_path / "private-study"
    private_root.mkdir()
    secret_label = str(tmp_path / "never-publish-this-class-label")
    labels = (secret_label, "ordinary-class")
    dataset = _write_talent_dataset(tmp_path / "talent", labels=labels)
    expected_inputs = {
        path.name: _sha256(path)
        for path in sorted(dataset.iterdir())
        if path.is_file()
    }
    if missing_input_hash is not None:
        expected_inputs.pop(missing_input_hash)
    checkpoint = tmp_path / "tabpfn-v2.6.ckpt"
    checkpoint.write_bytes(b"content-bound fake released checkpoint")
    dataset_manifest = tmp_path / "dataset-roster.json"
    dataset_manifest.write_text(
        json.dumps(
            {"assignments": [{"name": dataset.name, "split": "discovery"}]}
        ),
        encoding="utf-8",
    )
    provenance = _strict_provenance(
        tmp_path,
        monkeypatch,
        checkpoint=checkpoint,
        dataset_manifest=dataset_manifest,
    )
    config = {
        "schema_version": 1,
        "private_study_root": str(private_root),
        "datasets": [
            {
                "dataset_id": dataset.name,
                "path": str(dataset),
                "expected_input_sha256": expected_inputs,
            }
        ],
        "roster_split": roster_split,
        "evaluation_split": evaluation_split,
        "trusted_pickle": True,
        "seed": 17,
        "max_classes": 10,
        "device": "cpu",
        "provenance": provenance,
    }
    config_path = tmp_path / "tabpfn-localization.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return SimpleNamespace(
        config_path=config_path,
        private_root=private_root,
        dataset=dataset,
        checkpoint=checkpoint,
        dataset_manifest=dataset_manifest,
        labels=labels,
        provenance=provenance,
    )


@pytest.fixture(autouse=True)
def ample_private_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    from pe_mechanism import collect

    monkeypatch.setattr(
        collect.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": 100 * 1024**3})(),
    )


def test_placeholder_example_is_complete_anonymized_and_valid(tmp_path: Path) -> None:
    example = load_verified_json_config(EXAMPLES / "tabpfn-localize.example.json")
    original = copy.deepcopy(dict(example.data))
    private_root = tmp_path / "private-study"
    private_root.mkdir()
    dataset = tmp_path / "talent" / "placeholder-dataset"
    dataset.mkdir(parents=True)

    configured = copy.deepcopy(original)
    configured["private_study_root"] = str(private_root)
    configured["datasets"][0]["path"] = str(dataset)
    config, specs = _validate_configuration(
        VerifiedConfiguration(data=configured, file=example.file)
    )

    assert config["roster_split"] == "discovery"
    assert config["evaluation_split"] == "val"
    assert [spec.dataset_id for spec in specs] == ["placeholder-dataset"]
    assert set(specs[0].expected_input_sha256) == {
        "info.json",
        "N_train.npy",
        "N_val.npy",
        "N_test.npy",
        "y_train.npy",
        "y_val.npy",
        "y_test.npy",
    }
    assert all(
        len(value) == 64 for value in specs[0].expected_input_sha256.values()
    )

    provenance = original["provenance"]
    path_fields = (
        original["private_study_root"],
        original["datasets"][0]["path"],
        provenance["checkpoint_path"],
        provenance["dataset_manifest_path"],
        provenance["training_code_root"],
        provenance["model_code_root"],
        provenance["analysis_code_root"],
    )
    assert all(value.startswith("/absolute/") for value in path_fields)
    serialized = json.dumps(original, sort_keys=True).lower()
    assert not any(
        forbidden in serialized
        for forbidden in ("/mnt/", "/users/", "jiaxio", "slurm", "@")
    )


def test_fixed_weight_runner_is_train_only_atomic_paired_and_path_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _write_config(tmp_path, monkeypatch)
    classes = np.asarray([bundle.labels[1], bundle.labels[0]], dtype=object)
    factory = _FakeFactory(classes=classes)
    output = bundle.private_root / "complete-run"

    summary = run_tabpfn_localization(
        bundle.config_path,
        output,
        driver_factory=factory,
    )

    assert len(factory.calls) == 1
    call = factory.calls[0]
    assert call["checkpoint"] == bundle.checkpoint.resolve()
    assert call["checkpoint_sha256"] == _sha256(bundle.checkpoint)
    assert call["model_sha"] == bundle.provenance["expected_model_code_sha"]
    assert call["seed"] == 17
    assert call["device"] == "cpu"
    assert call["training_rows"] == 6
    assert call["validation_rows"] == 3
    assert factory.drivers[0].evaluation_splits == ["val"]

    assert output.is_dir()
    assert os.stat(output).st_mode & 0o077 == 0
    assert sorted(path.name for path in output.iterdir()) == [
        "manifest.json",
        "predictions.npz",
        "results.json",
        "summary.json",
    ]
    for path in output.iterdir():
        assert os.stat(path).st_mode & 0o077 == 0

    predictions = np.load(output / "predictions.npz", allow_pickle=False)
    assert predictions.files == [
        "d0000_labels",
        "d0000_full",
        "d0000_weight",
        "d0000_bias",
        "d0000_none",
    ]
    assert predictions["d0000_labels"].dtype == np.int64
    assert predictions["d0000_labels"].tolist() == [1, 0, 1]
    for component in ("full", "weight", "bias", "none"):
        assert predictions[f"d0000_{component}"].dtype == np.float32
        assert predictions[f"d0000_{component}"].shape == (3, 2)

    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    dataset = results["datasets"][0]
    assert results["assignment_split"] == "discovery"
    assert results["evaluation_split"] == "val"
    assert results["fit_context"] == "train"
    assert results["official_estimator"] == {
        "download_if_not_exists": False,
        "fit_mode": "fit_preprocessors",
        "memory_saving_mode": False,
        "n_estimators": 1,
        "random_state": 17,
    }
    assert [item["value_type"] for item in dataset["class_order"]] == [
        "str",
        "str",
    ]
    assert all(set(item) == {"index", "value_sha256", "value_type"} for item in dataset["class_order"])
    assert dataset["labels_sha256"] == _array_sha256(
        predictions["d0000_labels"]
    )
    for component in ("full", "weight", "bias", "none"):
        assert dataset["conditions"][component]["probability_sha256"] == (
            _array_sha256(predictions[f"d0000_{component}"])
        )
    assert dataset["exact_full_native_verified"] is True
    assert dataset["policies_restored_verified"] is True
    assert len(dataset["inputs"]) == 7
    assert all("path" not in item for item in dataset["inputs"])
    assert results["runtime_attestation_sha256"] == _json_sha256(
        results["runtime_attestation"]
    )

    assert summary["baseline_condition"] == "full"
    assert summary["dataset_count"] == 1
    assert tuple(summary["conditions"]) == ("full", "weight", "bias", "none")
    assert summary["conditions"]["full"]["accuracy_effect_vs_full"][
        "mean_effect"
    ] == 0.0
    assert summary["conditions"]["none"]["accuracy_effect_vs_full"][
        "mean_effect"
    ] < 0.0

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["command"] == "tabpfn-localize"
    assert manifest["model_family"] == "tabpfn-v2.6"
    assert manifest["checkpoint"]["sha256"] == _sha256(bundle.checkpoint)
    assert manifest["dataset_manifest"]["sha256"] == _sha256(
        bundle.dataset_manifest
    )
    assert manifest["evidence_level"] == "strict"
    assert len(manifest["inputs"]) == 7
    assert all(item["role"].startswith("talent.0000.") for item in manifest["inputs"])

    serialized = b"\n".join(
        (output / name).read_bytes()
        for name in ("results.json", "summary.json", "manifest.json")
    )
    assert str(tmp_path).encode() not in serialized
    assert bundle.labels[0].encode() not in serialized
    assert str(bundle.dataset).encode() not in serialized
    assert str(bundle.checkpoint).encode() not in serialized


def test_requires_every_talent_file_hash_before_driver_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _write_config(
        tmp_path,
        monkeypatch,
        missing_input_hash="N_val.npy",
    )
    factory = _FakeFactory(classes=np.asarray(bundle.labels, dtype=object))
    output = bundle.private_root / "must-not-exist"
    with pytest.raises(ValueError, match="every TALENT input role"):
        run_tabpfn_localization(
            bundle.config_path,
            output,
            driver_factory=factory,
        )
    assert factory.calls == []
    assert not output.exists()


@pytest.mark.parametrize(
    ("roster_split", "evaluation_split", "message"),
    [
        ("held_out", "val", "discovery datasets"),
        ("discovery", "test", "evaluates val only"),
    ],
)
def test_rejects_non_discovery_or_test_evaluation_before_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    roster_split: str,
    evaluation_split: str,
    message: str,
) -> None:
    bundle = _write_config(
        tmp_path,
        monkeypatch,
        roster_split=roster_split,
        evaluation_split=evaluation_split,
    )
    factory = _FakeFactory(classes=np.asarray(bundle.labels, dtype=object))
    with pytest.raises(ValueError, match=message):
        run_tabpfn_localization(
            bundle.config_path,
            bundle.private_root / "must-not-exist",
            driver_factory=factory,
        )
    assert factory.calls == []


def test_input_mutation_aborts_atomic_commit_and_removes_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _write_config(tmp_path, monkeypatch)
    factory = _FakeFactory(
        classes=np.asarray([bundle.labels[1], bundle.labels[0]], dtype=object),
        mutate_after_evaluate=bundle.checkpoint,
    )
    output = bundle.private_root / "failed-run"
    with pytest.raises((RuntimeError, ValueError), match="changed|SHA-256"):
        run_tabpfn_localization(
            bundle.config_path,
            output,
            driver_factory=factory,
        )
    assert not output.exists()
    assert not list(bundle.private_root.glob(".failed-run.*.staging"))


def test_failed_full_native_gate_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _write_config(tmp_path, monkeypatch)
    factory = _FakeFactory(
        classes=np.asarray([bundle.labels[1], bundle.labels[0]], dtype=object),
        invalidate_full_gate=True,
    )
    output = bundle.private_root / "invalid-full"
    with pytest.raises(RuntimeError, match="not exact to native"):
        run_tabpfn_localization(
            bundle.config_path,
            output,
            driver_factory=factory,
        )
    assert not output.exists()
