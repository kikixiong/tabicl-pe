from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pe_mechanism.talent import (
    UNKNOWN_CATEGORY_SENTINEL,
    assess_talent_eligibility,
    discover_talent_datasets,
    load_talent_dataset,
)


def _write_dataset(
    root: Path,
    name: str = "toy-classification",
    *,
    task_type: str = "binclass",
    val_numeric: float = 100.0,
    val_category: object = "blue",
    val_label: object = "class-a",
    empty_val: bool = False,
) -> Path:
    dataset = root / "benchmark_data" / "data" / name
    dataset.mkdir(parents=True)
    info = {
        "name": name,
        "task_type": task_type,
        "n_num_features": 2,
        "n_cat_features": 1,
        "n_classes": 2,
        "train_size": 3,
        "val_size": 0 if empty_val else 2,
        "test_size": 2,
    }
    (dataset / "info.json").write_text(json.dumps(info), encoding="utf-8")

    numeric = {
        "train": np.asarray([[0.0, np.nan], [2.0, 10.0], [4.0, 14.0]]),
        "val": np.empty((0, 2))
        if empty_val
        else np.asarray([[val_numeric, np.nan], [8.0, 12.0]]),
        "test": np.asarray([[1.0, 11.0], [3.0, 13.0]]),
    }
    categorical = {
        "train": np.asarray([["red"], [None], ["red"]], dtype=object),
        "val": np.empty((0, 1), dtype=object)
        if empty_val
        else np.asarray([[val_category], ["red"]], dtype=object),
        "test": np.asarray([[None], ["red"]], dtype=object),
    }
    targets = {
        "train": np.asarray(["class-a", "class-b", "class-a"], dtype=object),
        "val": np.empty(0, dtype=object)
        if empty_val
        else np.asarray([val_label, "class-b"], dtype=object),
        "test": np.asarray(["class-a", "class-b"], dtype=object),
    }
    for split in ("train", "val", "test"):
        np.save(dataset / f"N_{split}.npy", numeric[split])
        np.save(dataset / f"C_{split}.npy", categorical[split])
        np.save(dataset / f"y_{split}.npy", targets[split])
    return dataset


def test_discovery_requires_absolute_root_and_is_sorted(tmp_path: Path) -> None:
    _write_dataset(tmp_path, "zeta")
    _write_dataset(tmp_path, "Alpha")

    references = discover_talent_datasets(tmp_path.resolve())
    assert [reference.name for reference in references] == ["Alpha", "zeta"]
    assert all(reference.directory.is_absolute() for reference in references)
    with pytest.raises(ValueError, match="absolute"):
        discover_talent_datasets(Path("relative-talent-root"))


def test_discovery_uses_directory_key_when_info_display_names_repeat(tmp_path: Path) -> None:
    first = _write_dataset(tmp_path, "shared")
    second = _write_dataset(tmp_path, "shared_reg", task_type="regression")
    for dataset in (first, second):
        info_path = dataset / "info.json"
        info = json.loads(info_path.read_text())
        info["name"] = "shared-display-name"
        info_path.write_text(json.dumps(info), encoding="utf-8")

    references = discover_talent_datasets(tmp_path.resolve())
    assert [reference.name for reference in references] == ["shared", "shared_reg"]


def test_preprocessing_uses_only_training_statistics_and_fixed_unknown_sentinel(
    tmp_path: Path,
) -> None:
    first_path = _write_dataset(tmp_path / "first", val_numeric=100.0, val_category="blue")
    second_path = _write_dataset(
        tmp_path / "second", val_numeric=1_000_000.0, val_category="never-seen"
    )

    first = load_talent_dataset(first_path)
    second = load_talent_dataset(second_path)

    # Validation changes cannot alter train-fitted means, scales, imputation, or vocabularies.
    assert np.array_equal(first.train.X, second.train.X)
    assert np.allclose(first.train.X[:, :2].mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(first.train.X[:, :2].mean(axis=0), 0.0, atol=1e-7)
    assert first.val.X[0, 1] == pytest.approx(0.0)  # train median and mean are both 12
    assert first.val.X[0, 2] == UNKNOWN_CATEGORY_SENTINEL
    assert second.val.X[0, 2] == UNKNOWN_CATEGORY_SENTINEL
    assert first.test.X[0, 2] == 0.0  # missing category receives train-fitted mode "red"
    assert first.preprocessing_sha256 != second.preprocessing_sha256


def test_unknown_validation_label_is_flagged_and_encoded_without_vocab_leakage(
    tmp_path: Path,
) -> None:
    dataset_path = _write_dataset(tmp_path, val_label="new-class")
    report = assess_talent_eligibility(dataset_path)

    assert not report.eligible
    assert "unknown_val_labels" in report.reasons
    with pytest.raises(ValueError, match="unknown_val_labels"):
        load_talent_dataset(dataset_path)


def test_eligibility_reports_model_limits_and_structural_reasons(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path / "limited")
    eligible = assess_talent_eligibility(dataset_path, max_features=3, max_classes=2)
    limited = assess_talent_eligibility(dataset_path, max_features=2, max_classes=1)

    assert eligible.eligible
    assert eligible.n_features == 3
    assert eligible.n_classes == 2
    assert set(limited.reasons) == {"too_many_features", "too_many_classes"}

    regression_path = _write_dataset(tmp_path / "regression", task_type="regression")
    regression = assess_talent_eligibility(regression_path)
    assert not regression.eligible
    assert "unsupported_task_type" in regression.reasons

    empty_path = _write_dataset(tmp_path / "empty", empty_val=True)
    empty = assess_talent_eligibility(empty_path)
    assert "empty_val_split" in empty.reasons
    with pytest.raises(ValueError, match="ineligible"):
        load_talent_dataset(empty_path)


def test_eligibility_checks_every_split_shape_before_loading(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path)
    np.save(dataset_path / "N_val.npy", np.ones((2, 1), dtype=np.float64))
    report = assess_talent_eligibility(dataset_path)
    assert not report.eligible
    assert "n_val_width_mismatch" in report.reasons
    with pytest.raises(ValueError, match="n_val_width_mismatch"):
        load_talent_dataset(dataset_path)


def test_model_limits_and_task_type_are_hard_loading_gates(tmp_path: Path) -> None:
    classification = _write_dataset(tmp_path / "classification")
    with pytest.raises(ValueError, match="too_many_features"):
        load_talent_dataset(classification, max_features=2)
    regression = _write_dataset(tmp_path / "regression", task_type="regression")
    with pytest.raises(ValueError, match="unsupported_task_type"):
        load_talent_dataset(regression)


def test_object_typed_numeric_storage_is_ineligible_not_an_unhandled_load(
    tmp_path: Path,
) -> None:
    dataset_path = _write_dataset(tmp_path)
    for split in ("train", "val", "test"):
        values = np.load(dataset_path / f"N_{split}.npy").astype(object)
        np.save(dataset_path / f"N_{split}.npy", values)
    report = assess_talent_eligibility(dataset_path)
    assert not report.eligible
    assert "unsafe_object_numeric_features" in report.reasons
    with pytest.raises(ValueError, match="unsafe_object_numeric_features"):
        load_talent_dataset(dataset_path)


def test_view_hash_and_public_metadata_are_root_independent_and_path_free(tmp_path: Path) -> None:
    first_root = tmp_path / "private-location-one"
    second_root = tmp_path / "private-location-two"
    first_path = _write_dataset(first_root)
    second_path = _write_dataset(second_root)

    first_reference = discover_talent_datasets(first_root.resolve())[0]
    second_reference = discover_talent_datasets(second_root.resolve())[0]
    first = load_talent_dataset(first_path)
    second = load_talent_dataset(second_path)

    assert first.preprocessing_sha256 == second.preprocessing_sha256
    assert first.preprocessing_view_id == second.preprocessing_view_id
    assert first_reference.portable_metadata() == second_reference.portable_metadata()
    assert first.portable_metadata() == second.portable_metadata()
    serialized = json.dumps(first.portable_metadata(), sort_keys=True)
    assert str(first_root) not in serialized
    assert str(second_root) not in serialized
    assert "directory" not in serialized


def test_loading_is_deterministic(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path)
    first = load_talent_dataset(dataset_path)
    second = load_talent_dataset(dataset_path)

    assert first.preprocessing_view_id == second.preprocessing_view_id
    for split_name in ("train", "val", "test"):
        first_split = getattr(first, split_name)
        second_split = getattr(second, split_name)
        assert np.array_equal(first_split.X, second_split.X)
        assert np.array_equal(first_split.y, second_split.y)
