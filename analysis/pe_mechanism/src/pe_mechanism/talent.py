"""Leakage-safe loading of pre-split TALENT NumPy datasets.

The caller supplies the TALENT root at runtime.  Paths are retained only in the
private dataset reference and never enter portable metadata or view hashes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


UNKNOWN_CATEGORY_SENTINEL = -1.0
UNKNOWN_LABEL_SENTINEL = -1
_CLASSIFICATION_TASKS = frozenset(
    {
        "binclass",
        "binary",
        "binary_classification",
        "classification",
        "multiclass",
        "multiclass_classification",
    }
)
_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class TalentDatasetReference:
    """A discovered dataset; ``directory`` is private runtime state."""

    name: str
    task_type: str
    info_sha256: str
    directory: Path = field(repr=False, compare=False)

    def portable_metadata(self) -> dict[str, str]:
        return {
            "dataset_name": self.name,
            "task_type": self.task_type,
            "info_sha256": self.info_sha256,
        }


@dataclass(frozen=True)
class TalentEligibility:
    eligible: bool
    reasons: tuple[str, ...]
    task_type: str
    n_features: int
    n_classes: int
    split_sizes: tuple[int, int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "task_type": self.task_type,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "split_sizes": dict(zip(_SPLITS, self.split_sizes, strict=True)),
        }


@dataclass(frozen=True)
class TalentSplit:
    X: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class TalentDataset:
    name: str
    task_type: str
    train: TalentSplit
    val: TalentSplit
    test: TalentSplit
    eligibility: TalentEligibility
    preprocessing_view_id: str
    preprocessing_sha256: str
    n_numeric_features: int
    n_categorical_features: int
    info_sha256: str

    def portable_metadata(self) -> dict[str, Any]:
        """Return provenance that cannot reveal the user-supplied TALENT root."""

        return {
            "dataset_name": self.name,
            "task_type": self.task_type,
            "info_sha256": self.info_sha256,
            "preprocessing_view_id": self.preprocessing_view_id,
            "preprocessing_sha256": self.preprocessing_sha256,
            "n_numeric_features": self.n_numeric_features,
            "n_categorical_features": self.n_categorical_features,
            "n_features": self.eligibility.n_features,
            "n_classes": self.eligibility.n_classes,
            "split_sizes": dict(zip(_SPLITS, self.eligibility.split_sizes, strict=True)),
            "unknown_category_sentinel": UNKNOWN_CATEGORY_SENTINEL,
        }


def discover_talent_datasets(root: str | Path) -> tuple[TalentDatasetReference, ...]:
    """Discover every dataset below an absolute caller-provided TALENT root."""

    raw_root = Path(root).expanduser()
    if not raw_root.is_absolute():
        raise ValueError("TALENT root must be an absolute path")
    resolved_root = raw_root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError("TALENT root must be a directory")

    references: list[TalentDatasetReference] = []
    seen_names: dict[str, Path] = {}
    for info_path in sorted(resolved_root.rglob("info.json")):
        info = _read_info(info_path)
        # TALENT contains classification/regression pairs whose info.json files
        # intentionally reuse a display name (for example ``foo`` and
        # ``foo_reg``).  The dataset directory is the canonical roster key.
        dataset_name = info_path.parent.name.strip()
        if not dataset_name:
            raise ValueError(f"dataset at {info_path.parent.name!r} has an empty name")
        if dataset_name in seen_names:
            raise ValueError(f"duplicate TALENT dataset name: {dataset_name!r}")
        seen_names[dataset_name] = info_path.parent
        info_digest = _json_digest(info)
        references.append(
            TalentDatasetReference(
                name=dataset_name,
                task_type=_task_type(info),
                info_sha256=info_digest,
                directory=info_path.parent.resolve(strict=True),
            )
        )
    return tuple(sorted(references, key=lambda item: (item.name.casefold(), item.name)))


def assess_talent_eligibility(
    reference: TalentDatasetReference | str | Path,
    *,
    max_features: int | None = None,
    max_classes: int | None = None,
) -> TalentEligibility:
    """Inspect task, split sizes, features, and labels without fitting preprocessing."""

    _validate_limit("max_features", max_features)
    _validate_limit("max_classes", max_classes)
    dataset_dir, info = _reference_parts(reference)
    task_type = _task_type(info)
    reasons: list[str] = []
    if task_type not in _CLASSIFICATION_TASKS:
        reasons.append("unsupported_task_type")

    targets: dict[str, np.ndarray] = {}
    missing_files = False
    for split in _SPLITS:
        target_path = dataset_dir / f"y_{split}.npy"
        if not target_path.is_file():
            missing_files = True
            targets[split] = np.empty(0, dtype=object)
        else:
            targets[split] = _target_vector(np.load(target_path, allow_pickle=True))
    if missing_files:
        reasons.append("missing_split_files")
    split_sizes = tuple(int(targets[split].shape[0]) for split in _SPLITS)
    for split, size in zip(_SPLITS, split_sizes, strict=True):
        if size == 0:
            reasons.append(f"empty_{split}_split")

    widths: list[int] = []
    for prefix in ("N", "C"):
        paths = [dataset_dir / f"{prefix}_{split}.npy" for split in _SPLITS]
        present = [path.is_file() for path in paths]
        if any(present) and not all(present):
            reasons.append(f"incomplete_{prefix.lower()}_feature_splits")
        if all(present):
            try:
                arrays = {
                    split: _feature_matrix(
                        np.load(path, allow_pickle=prefix == "C")
                    )
                    for split, path in zip(_SPLITS, paths, strict=True)
                }
            except ValueError as error:
                if prefix == "N" and "Object arrays cannot be loaded" in str(error):
                    reasons.append("unsafe_object_numeric_features")
                    widths.append(0)
                    continue
                raise
            train_width = arrays["train"].shape[1]
            widths.append(train_width)
            for split in _SPLITS:
                if arrays[split].shape[0] != split_sizes[_SPLITS.index(split)]:
                    reasons.append(f"{prefix.lower()}_{split}_row_mismatch")
                if arrays[split].shape[1] != train_width:
                    reasons.append(f"{prefix.lower()}_{split}_width_mismatch")
        else:
            widths.append(0)
    n_features = sum(widths)
    if n_features == 0:
        reasons.append("no_features")
    if max_features is not None and n_features > max_features:
        reasons.append("too_many_features")

    train_labels = {_value_token(value) for value in targets["train"] if not _is_missing(value)}
    n_classes = len(train_labels)
    if task_type in _CLASSIFICATION_TASKS and n_classes < 2:
        reasons.append("fewer_than_two_train_classes")
    if max_classes is not None and n_classes > max_classes:
        reasons.append("too_many_classes")
    for split in ("val", "test"):
        split_labels = {_value_token(value) for value in targets[split] if not _is_missing(value)}
        if split_labels - train_labels:
            reasons.append(f"unknown_{split}_labels")

    return TalentEligibility(
        eligible=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        task_type=task_type,
        n_features=n_features,
        n_classes=n_classes,
        split_sizes=split_sizes,
    )


def load_talent_dataset(
    reference: TalentDatasetReference | str | Path,
    *,
    max_features: int | None = None,
    max_classes: int | None = None,
) -> TalentDataset:
    """Load and train-fit preprocess a TALENT dataset without split leakage."""

    dataset_dir, info = _reference_parts(reference)
    name = dataset_dir.name.strip()
    task_type = _task_type(info)
    info_sha256 = _json_digest(info)
    eligibility = assess_talent_eligibility(
        dataset_dir, max_features=max_features, max_classes=max_classes
    )
    if not eligibility.eligible:
        raise ValueError(
            "cannot preprocess ineligible TALENT dataset: "
            f"{list(eligibility.reasons)}"
        )

    raw_y = {
        split: _target_vector(np.load(dataset_dir / f"y_{split}.npy", allow_pickle=True))
        for split in _SPLITS
    }
    row_counts = {split: len(raw_y[split]) for split in _SPLITS}
    numeric = _load_feature_splits(dataset_dir, "N", row_counts, allow_pickle=False)
    categorical = _load_feature_splits(dataset_dir, "C", row_counts, allow_pickle=True)

    numeric_processed, numeric_state = _preprocess_numeric(numeric)
    categorical_processed, categorical_state = _preprocess_categorical(categorical)
    y_processed, label_state = _preprocess_labels(raw_y)

    X: dict[str, np.ndarray] = {}
    for split in _SPLITS:
        parts = (numeric_processed[split], categorical_processed[split])
        X[split] = np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)
        if not np.isfinite(X[split]).all():
            raise ValueError(f"preprocessed {split} features are not finite")

    preprocessing_state = {
        "schema_version": 1,
        "algorithm": "talent_train_only_mean_rms_and_ordinal_v1",
        "dataset_name": name,
        "task_type": task_type,
        "numeric": numeric_state,
        "categorical": categorical_state,
        "labels": label_state,
        "unknown_category_sentinel": UNKNOWN_CATEGORY_SENTINEL,
        "unknown_label_sentinel": UNKNOWN_LABEL_SENTINEL,
    }
    preprocessing_sha256 = _preprocessing_digest(preprocessing_state, X, y_processed)
    view_id = f"talent-v1-{preprocessing_sha256[:16]}"
    return TalentDataset(
        name=name,
        task_type=task_type,
        train=TalentSplit(X["train"], y_processed["train"]),
        val=TalentSplit(X["val"], y_processed["val"]),
        test=TalentSplit(X["test"], y_processed["test"]),
        eligibility=eligibility,
        preprocessing_view_id=view_id,
        preprocessing_sha256=preprocessing_sha256,
        n_numeric_features=numeric_processed["train"].shape[1],
        n_categorical_features=categorical_processed["train"].shape[1],
        info_sha256=info_sha256,
    )


def _preprocess_numeric(
    splits: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train = np.asarray(splits["train"], dtype=np.float64)
    width = train.shape[1]
    if width == 0:
        empty = {split: np.empty((values.shape[0], 0), dtype=np.float32) for split, values in splits.items()}
        return empty, {"imputation": [], "mean": [], "rms": []}

    imputation = np.empty(width, dtype=np.float64)
    for column in range(width):
        observed = train[np.isfinite(train[:, column]), column]
        imputation[column] = float(np.median(observed)) if observed.size else 0.0
    train_filled = _fill_numeric(train, imputation)
    mean = train_filled.mean(axis=0)
    rms = np.sqrt(np.mean(np.square(train_filled - mean), axis=0))
    rms = np.where(rms > 1e-12, rms, 1.0)

    processed = {
        split: np.asarray((_fill_numeric(values, imputation) - mean) / rms, dtype=np.float32)
        for split, values in splits.items()
    }
    state = {
        "imputation": imputation.tolist(),
        "mean": mean.tolist(),
        "rms": rms.tolist(),
    }
    return processed, state


def _preprocess_categorical(
    splits: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train = np.asarray(splits["train"], dtype=object)
    width = train.shape[1]
    processed = {
        split: np.empty((values.shape[0], width), dtype=np.float32)
        for split, values in splits.items()
    }
    fill_tokens: list[str] = []
    vocabularies: list[list[str]] = []
    for column in range(width):
        observed = [_value_token(value) for value in train[:, column] if not _is_missing(value)]
        if observed:
            counts = Counter(observed)
            fill_token = min(counts, key=lambda token: (-counts[token], token))
        else:
            fill_token = "missing:"
        train_tokens = [
            fill_token if _is_missing(value) else _value_token(value) for value in train[:, column]
        ]
        vocabulary = sorted(set(train_tokens))
        mapping = {token: index for index, token in enumerate(vocabulary)}
        for split, values in splits.items():
            tokens = [
                fill_token if _is_missing(value) else _value_token(value)
                for value in np.asarray(values, dtype=object)[:, column]
            ]
            processed[split][:, column] = np.asarray(
                [mapping.get(token, UNKNOWN_CATEGORY_SENTINEL) for token in tokens],
                dtype=np.float32,
            )
        fill_tokens.append(fill_token)
        vocabularies.append(vocabulary)
    return processed, {"fill_tokens": fill_tokens, "vocabularies": vocabularies}


def _preprocess_labels(
    splits: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train_tokens = [_value_token(value) for value in splits["train"] if not _is_missing(value)]
    vocabulary = sorted(set(train_tokens))
    mapping = {token: index for index, token in enumerate(vocabulary)}
    processed: dict[str, np.ndarray] = {}
    for split, values in splits.items():
        tokens = [None if _is_missing(value) else _value_token(value) for value in values]
        processed[split] = np.asarray(
            [mapping.get(token, UNKNOWN_LABEL_SENTINEL) for token in tokens], dtype=np.int64
        )
    return processed, {"vocabulary": vocabulary}


def _load_feature_splits(
    dataset_dir: Path,
    prefix: str,
    row_counts: Mapping[str, int],
    *,
    allow_pickle: bool,
) -> dict[str, np.ndarray]:
    paths = {split: dataset_dir / f"{prefix}_{split}.npy" for split in _SPLITS}
    if not any(path.exists() for path in paths.values()):
        dtype = object if allow_pickle else np.float64
        return {
            split: np.empty((row_counts[split], 0), dtype=dtype) for split in _SPLITS
        }
    if not all(path.is_file() for path in paths.values()):
        raise ValueError(f"{prefix} feature files must exist for train, val, and test")
    arrays = {
        split: _feature_matrix(np.load(path, allow_pickle=allow_pickle))
        for split, path in paths.items()
    }
    width = arrays["train"].shape[1]
    for split, values in arrays.items():
        if values.shape != (row_counts[split], width):
            raise ValueError(f"{prefix}_{split} shape does not match targets/train width")
    return arrays


def _fill_numeric(values: np.ndarray, imputation: np.ndarray) -> np.ndarray:
    numeric = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(numeric), numeric, imputation)


def _feature_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("feature arrays must be two-dimensional")
    return array


def _feature_width(values: np.ndarray) -> int:
    return int(_feature_matrix(values).shape[1])


def _target_vector(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError("target arrays must be one-dimensional")
    return array


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return True
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8", errors="replace")
    return isinstance(value, str) and value.strip().casefold() in {"", "nan", "none", "null"}


def _value_token(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return "string:" + bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, str):
        return "string:" + value
    if isinstance(value, (bool, np.bool_)):
        return "bool:" + ("true" if bool(value) else "false")
    if isinstance(value, (int, np.integer)):
        return f"int:{int(value)}"
    if isinstance(value, (float, np.floating)):
        return "float:" + float(value).hex()
    return f"{type(value).__name__}:{value!s}"


def _preprocessing_digest(
    state: Mapping[str, Any], X: Mapping[str, np.ndarray], y: Mapping[str, np.ndarray]
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for split in _SPLITS:
        _update_array_digest(digest, f"X_{split}", X[split])
        _update_array_digest(digest, f"y_{split}", y[split])
    return digest.hexdigest()


def _update_array_digest(digest: Any, name: str, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(name.encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))


def _reference_parts(
    reference: TalentDatasetReference | str | Path,
) -> tuple[Path, dict[str, Any]]:
    if isinstance(reference, TalentDatasetReference):
        dataset_dir = reference.directory.resolve(strict=True)
    else:
        raw_path = Path(reference).expanduser()
        if not raw_path.is_absolute():
            raise ValueError("TALENT dataset directory must be an absolute path")
        dataset_dir = raw_path.resolve(strict=True)
    if not dataset_dir.is_dir():
        raise ValueError("TALENT dataset path must be a directory")
    return dataset_dir, _read_info(dataset_dir / "info.json")


def _read_info(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        info = json.load(handle)
    if not isinstance(info, dict):
        raise ValueError("TALENT info.json must contain a JSON object")
    return info


def _task_type(info: Mapping[str, Any]) -> str:
    return str(info.get("task_type", "unknown")).strip().lower().replace("-", "_")


def _json_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_limit(name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or int(value) <= 0):
        raise ValueError(f"{name} must be a positive integer or None")


# Convenient aliases for callers that use shorter workflow names.
discover_datasets = discover_talent_datasets
check_eligibility = assess_talent_eligibility
evaluate_eligibility = assess_talent_eligibility
load_dataset = load_talent_dataset


__all__ = [
    "TalentDataset",
    "TalentDatasetReference",
    "TalentEligibility",
    "TalentSplit",
    "UNKNOWN_CATEGORY_SENTINEL",
    "UNKNOWN_LABEL_SENTINEL",
    "assess_talent_eligibility",
    "check_eligibility",
    "discover_datasets",
    "discover_talent_datasets",
    "evaluate_eligibility",
    "load_dataset",
    "load_talent_dataset",
]
