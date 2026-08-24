"""CPU-only stratification of a completed RoPE/NoPE TabArena pair.

The analysis deliberately reopens every frozen r0f0 task and verifies the
canonical table-content digests embedded in both arms' prediction archives.
Only scale-free win statistics are pooled across metrics; raw error differences
remain metric-specific throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .statistics import paired_bootstrap_ci
from .tabarena_formal_runner import (
    _feature_table_fingerprint,
    _table_content_fingerprint,
)


EXPECTED_TASK_COUNT = 38
EXPECTED_RESULT_COUNT = 76
CORRELATION_THRESHOLD = 0.95
BOOTSTRAP_SEED = 20260817
BOOTSTRAP_RESAMPLES = 20_000
FEATURE_COUNT_BIN_COUNT = 4


@dataclass(frozen=True)
class ModelFacingColumns:
    """Supported non-empty columns selected by the evaluation adapter."""

    feature_columns: tuple[Any, ...]
    categorical_columns: tuple[Any, ...]
    numeric_columns: tuple[Any, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _real_file(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an existing absolute real file")
    return path


def _real_directory(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be an existing absolute real directory")
    return path


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    _real_file(path, label=label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be an integer")
    integer = int(value)
    if float(value) != integer or integer < 1:
        raise ValueError(f"{label} must be a positive integer")
    return integer


def load_validated_aggregate(
    path: Path,
    *,
    expected_sha256: str,
    expected_task_count: int = EXPECTED_TASK_COUNT,
    expected_result_count: int = EXPECTED_RESULT_COUNT,
) -> dict[str, Any]:
    """Load a completed aggregate only when its frozen digest and shape match."""

    path = _real_file(path, label="aggregate")
    expected_digest = _sha256(expected_sha256, label="expected aggregate SHA256")
    if _sha256_file(path) != expected_digest:
        raise ValueError("aggregate SHA256 differs from the exact expected digest")
    payload = _load_json_object(path, label="aggregate")
    if payload.get("complete") is not True:
        raise ValueError("aggregate is not complete")
    if payload.get("arm_order") != ["rope", "none"]:
        raise ValueError("aggregate arm order must be exactly rope then none")
    if payload.get("task_count") != expected_task_count:
        raise ValueError("aggregate task_count differs from the expected count")
    if payload.get("result_count") != expected_result_count:
        raise ValueError("aggregate result_count differs from the expected count")

    datasets = payload.get("datasets")
    artifacts = payload.get("task_artifacts")
    if not isinstance(datasets, list) or len(datasets) != expected_task_count:
        raise ValueError("aggregate datasets do not exactly cover the expected tasks")
    if not isinstance(artifacts, list) or len(artifacts) != expected_task_count:
        raise ValueError("aggregate task artifacts do not exactly cover the tasks")

    names: set[str] = set()
    task_ids: set[int] = set()
    observed_results = 0
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise ValueError(f"aggregate dataset {index} is not an object")
        task = dataset.get("task")
        results = dataset.get("results")
        if not isinstance(task, dict) or not isinstance(results, dict):
            raise ValueError(f"aggregate dataset {index} lacks task/results objects")
        name = task.get("dataset_name")
        task_id = task.get("task_id")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("aggregate dataset names must be non-empty and unique")
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id < 1
            or task_id in task_ids
        ):
            raise ValueError("aggregate task IDs must be positive and unique")
        names.add(name)
        task_ids.add(task_id)
        if (
            task.get("fold"),
            task.get("repeat"),
            task.get("split_index"),
        ) != (0, 0, 0):
            raise ValueError(f"aggregate task is not exact r0f0: {name}")
        metric = task.get("metric")
        problem_type = task.get("problem_type")
        expected_metric = "roc_auc" if problem_type == "binary" else "log_loss"
        if problem_type not in {"binary", "multiclass"} or metric != expected_metric:
            raise ValueError(f"aggregate classification metric is invalid: {name}")
        if set(results) != {"rope", "none"}:
            raise ValueError(f"aggregate task does not have exactly two arms: {name}")
        for arm in ("rope", "none"):
            result = results[arm]
            if not isinstance(result, dict):
                raise ValueError(f"aggregate {arm} result is invalid: {name}")
            _finite_number(result.get("metric_error"), label=f"{name} {arm} error")
            observed_results += 1
    if observed_results != expected_result_count:
        raise ValueError("aggregate arm result coverage differs from result_count")

    artifact_directories: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("aggregate task artifact is not an object")
        directory = artifact.get("directory")
        if (
            not isinstance(directory, str)
            or not directory
            or Path(directory).name != directory
            or directory in artifact_directories
        ):
            raise ValueError("aggregate task artifact directory is unsafe or duplicated")
        artifact_directories.add(directory)
        _sha256(artifact.get("manifest_sha256"), label="task manifest SHA256")
        _positive_integer(
            artifact.get("manifest_size_bytes"), label="task manifest size"
        )
    return payload


def select_model_facing_columns(frame: Any) -> ModelFacingColumns:
    """Apply the exact dtype/non-empty selector used by TabArena evaluation."""

    import pandas as pd
    from sklearn.compose import make_column_selector

    if not isinstance(frame, pd.DataFrame) or len(frame) < 1:
        raise TypeError("model-facing feature selection requires a non-empty DataFrame")
    if not frame.columns.is_unique:
        raise ValueError("model-facing feature columns must be unique")
    categorical_raw = make_column_selector(
        dtype_include=["string", "object", "category", "boolean"]
    )(frame)
    numeric_raw = make_column_selector(dtype_include="number")(frame)
    supported = set(categorical_raw) | set(numeric_raw)
    feature_columns = tuple(
        column
        for column in frame.columns
        if column in supported and bool(frame[column].notna().any())
    )
    selected = set(feature_columns)
    categorical_columns = tuple(
        column for column in categorical_raw if column in selected
    )
    numeric_columns = tuple(column for column in numeric_raw if column in selected)
    return ModelFacingColumns(
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )


def _encoded_training_matrix(
    frame: Any,
    *,
    numeric_only: bool,
) -> tuple[np.ndarray, ModelFacingColumns]:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler

    columns = select_model_facing_columns(frame)
    active = columns.numeric_columns if numeric_only else columns.feature_columns
    if not active:
        return np.empty((len(frame), 0), dtype=np.float64), columns

    vectors: dict[Any, np.ndarray] = {}
    if columns.numeric_columns:
        numeric = frame.loc[:, list(columns.numeric_columns)].to_numpy(dtype=np.float64)
        numeric = SimpleImputer(strategy="mean").fit_transform(numeric)
        for index, column in enumerate(columns.numeric_columns):
            vectors[column] = numeric[:, index]
    if not numeric_only and columns.categorical_columns:
        # This intentionally mirrors the legacy preprocessing: missing categorical
        # values become the literal string token before deterministic ordinal coding.
        categorical = frame.loc[:, list(columns.categorical_columns)].apply(
            lambda series: series.map(str)
        )
        encoded = OrdinalEncoder().fit_transform(categorical)
        for index, column in enumerate(columns.categorical_columns):
            vectors[column] = encoded[:, index]
    matrix = np.column_stack([vectors[column] for column in active]).astype(
        np.float64, copy=False
    )
    if not np.isfinite(matrix).all():
        raise ValueError("encoded training features contain non-finite values")
    matrix = StandardScaler().fit_transform(matrix)
    return np.ascontiguousarray(matrix, dtype=np.float64), columns


def duplicate_graph_summary(
    correlation: np.ndarray,
    *,
    threshold: float = CORRELATION_THRESHOLD,
) -> dict[str, Any]:
    """Summarize the strict |Pearson| > threshold undirected column graph."""

    matrix = np.asarray(correlation, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("correlation must be a square matrix")
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T):
        raise ValueError("correlation must be finite and symmetric")
    if not 0.0 < threshold < 1.0:
        raise ValueError("correlation threshold must lie between zero and one")
    feature_count = matrix.shape[0]
    possible_pairs = feature_count * (feature_count - 1) // 2
    adjacency = np.abs(matrix) > threshold
    np.fill_diagonal(adjacency, False)
    upper = np.triu(adjacency, k=1)
    pair_count = int(upper.sum())
    incident = np.flatnonzero(adjacency.any(axis=0))

    parents = list(range(feature_count))

    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left, right in np.argwhere(upper):
        union(int(left), int(right))
    component_sizes: dict[int, int] = {}
    for column in range(feature_count):
        root = find(column)
        component_sizes[root] = component_sizes.get(root, 0) + 1

    return {
        "feature_count": feature_count,
        "near_duplicate_pair_count": pair_count,
        "near_duplicate_pair_density": (
            float(pair_count / possible_pairs) if possible_pairs else 0.0
        ),
        "incident_column_count": int(incident.size),
        "duplicate_column_fraction": (
            float(incident.size / feature_count) if feature_count else 0.0
        ),
        "maximum_connected_component_size": (
            max(component_sizes.values()) if component_sizes else 0
        ),
    }


def _duplicate_summary(
    frame: Any,
    *,
    numeric_only: bool,
    threshold: float,
) -> dict[str, Any]:
    matrix, columns = _encoded_training_matrix(frame, numeric_only=numeric_only)
    feature_count = matrix.shape[1]
    if feature_count < 2 or matrix.shape[0] < 2:
        correlation = np.eye(feature_count, dtype=np.float64)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = np.corrcoef(matrix, rowvar=False)
        correlation = np.asarray(correlation, dtype=np.float64)
        if correlation.ndim == 0:
            correlation = correlation.reshape(1, 1)
        correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(correlation, 1.0)
    summary = duplicate_graph_summary(correlation, threshold=threshold)
    expected_count = (
        len(columns.numeric_columns) if numeric_only else len(columns.feature_columns)
    )
    if summary["feature_count"] != expected_count:
        raise AssertionError("encoded feature count differs from selected columns")
    return summary


def all_supported_duplicate_summary(
    frame: Any,
    *,
    threshold: float = CORRELATION_THRESHOLD,
) -> dict[str, Any]:
    return _duplicate_summary(frame, numeric_only=False, threshold=threshold)


def numeric_only_duplicate_summary(
    frame: Any,
    *,
    threshold: float = CORRELATION_THRESHOLD,
) -> dict[str, Any]:
    return _duplicate_summary(frame, numeric_only=True, threshold=threshold)


def _row_number(row: Mapping[str, Any], key: str) -> float:
    return _finite_number(row.get(key), label=f"{key} for {row.get('dataset_name')}")


def _balanced_sizes(count: int, bin_count: int) -> list[int]:
    if count < 1 or bin_count < 1:
        raise ValueError("balanced bins require positive counts")
    bin_count = min(count, bin_count)
    base, remainder = divmod(count, bin_count)
    return [base + (index < remainder) for index in range(bin_count)]


def assign_equal_count_strata(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    bin_count: int,
    label_prefix: str,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Assign deterministic equal-count rank bins, recording boundary tie splits."""

    if not rows:
        raise ValueError("equal-count strata require at least one row")
    names = [row.get("dataset_name") for row in rows]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("stratification rows require non-empty dataset names")
    if len(set(names)) != len(names):
        raise ValueError("stratification dataset names must be unique")
    ordered = sorted(
        rows,
        key=lambda row: (_row_number(row, value_key), str(row["dataset_name"])),
    )
    assignments: dict[str, str] = {}
    summaries: list[dict[str, Any]] = []
    offset = 0
    sizes = _balanced_sizes(len(ordered), bin_count)
    for index, size in enumerate(sizes):
        members = ordered[offset : offset + size]
        label = f"{label_prefix}{index + 1}"
        values = [_row_number(member, value_key) for member in members]
        for member in members:
            assignments[str(member["dataset_name"])] = label
        next_value = (
            _row_number(ordered[offset + size], value_key)
            if offset + size < len(ordered)
            else None
        )
        summaries.append(
            {
                "label": label,
                "dataset_count": size,
                "minimum": min(values),
                "maximum": max(values),
                "boundary_tie_split_after": (
                    next_value is not None and max(values) == next_value
                ),
            }
        )
        offset += size
    return assignments, summaries


def assign_density_strata(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Keep zero-density tasks together and split positive density in half."""

    if not rows:
        raise ValueError("density strata require at least one row")
    zeros = [row for row in rows if _row_number(row, value_key) == 0.0]
    positives = [row for row in rows if _row_number(row, value_key) > 0.0]
    if len(zeros) + len(positives) != len(rows):
        raise ValueError("density values must be non-negative")
    assignments: dict[str, str] = {}
    summaries: list[dict[str, Any]] = []
    if zeros:
        for row in zeros:
            name = row.get("dataset_name")
            if not isinstance(name, str) or not name or name in assignments:
                raise ValueError("density rows require unique dataset names")
            assignments[name] = "zero"
        summaries.append(
            {
                "label": "zero",
                "dataset_count": len(zeros),
                "minimum": 0.0,
                "maximum": 0.0,
                "boundary_tie_split_after": False,
            }
        )
    if positives:
        positive_bins = 1 if len(positives) == 1 else 2
        positive_assignments, positive_summaries = assign_equal_count_strata(
            positives,
            value_key=value_key,
            bin_count=positive_bins,
            label_prefix="positive_",
        )
        replacement = (
            ["positive"]
            if positive_bins == 1
            else ["positive_low", "positive_high"]
        )
        for index, summary in enumerate(positive_summaries):
            old_label = summary["label"]
            label = replacement[index]
            summary["label"] = label
            for name, assigned in positive_assignments.items():
                if assigned == old_label:
                    if name in assignments:
                        raise ValueError("density dataset names must be unique")
                    assignments[name] = label
            summaries.append(summary)
    if len(assignments) != len(rows):
        raise ValueError("density dataset names must be unique")
    return assignments, summaries


def _scale_free_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("comparison requires at least one dataset")
    rope_wins = none_wins = ties = 0
    for row in rows:
        rope = _row_number(row, "rope_metric_error")
        none = _row_number(row, "none_metric_error")
        if rope < none:
            rope_wins += 1
        elif none < rope:
            none_wins += 1
        else:
            ties += 1
    non_ties = rope_wins + none_wins
    if non_ties:
        smaller = min(rope_wins, none_wins)
        tail = sum(math.comb(non_ties, value) for value in range(smaller + 1))
        sign_p = min(1.0, 2.0 * tail / (2**non_ties))
    else:
        sign_p = 1.0
    count = len(rows)
    return {
        "dataset_count": count,
        "rope_wins": rope_wins,
        "none_wins": none_wins,
        "ties": ties,
        "mean_rank_lower_is_better": {
            "rope": (rope_wins + 2 * none_wins + 1.5 * ties) / count,
            "none": (none_wins + 2 * rope_wins + 1.5 * ties) / count,
        },
        "two_sided_exact_sign_test_p": float(sign_p),
        "rank_win_sign_statistics_are_scale_free": True,
    }


def _raw_delta_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    differences = np.asarray(
        [
            _row_number(row, "rope_metric_error")
            - _row_number(row, "none_metric_error")
            for row in rows
        ],
        dtype=np.float64,
    )
    low, high = paired_bootstrap_ci(
        differences,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    return {
        "metric": metric,
        "dataset_count": len(rows),
        "mean": float(differences.mean()),
        "median": float(np.median(differences)),
        "minimum": float(differences.min()),
        "maximum": float(differences.max()),
        "direction": "rope_metric_error_minus_none_metric_error",
        "negative_means_rope_is_better": True,
        "paired_bootstrap_95ci": [low, high],
        "paired_bootstrap_seed": bootstrap_seed,
        "paired_bootstrap_resamples": bootstrap_resamples,
    }


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("summary requires at least one dataset")
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    metrics: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        metric = row.get("metric")
        if not isinstance(metric, str) or not metric:
            raise ValueError("summary rows require a metric")
        metrics.setdefault(metric, []).append(row)
    raw_by_metric = {
        metric: _raw_delta_summary(
            metric_rows,
            metric=metric,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        )
        for metric, metric_rows in sorted(metrics.items())
    }
    result: dict[str, Any] = {
        "scale_free": _scale_free_summary(rows),
        "raw_metric_error_difference_by_metric": raw_by_metric,
    }
    if len(raw_by_metric) == 1:
        result["raw_metric_error_difference"] = next(iter(raw_by_metric.values()))
    else:
        result["raw_metric_error_difference"] = None
        result["raw_difference_omission_reason"] = (
            "mixed metric scales; raw deltas are reported only within metric"
        )
    return result


def continuous_spearman_descriptions(
    rows: Sequence[Mapping[str, Any]],
    *,
    covariates: Sequence[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Describe continuous associations within metric without pooling scales."""

    from scipy.stats import spearmanr

    metrics = sorted({str(row.get("metric")) for row in rows})
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for metric in metrics:
        metric_rows = [row for row in rows if row.get("metric") == metric]
        result[metric] = {}
        delta = np.asarray(
            [
                _row_number(row, "rope_metric_error")
                - _row_number(row, "none_metric_error")
                for row in metric_rows
            ],
            dtype=np.float64,
        )
        for covariate in covariates:
            values = np.asarray(
                [_row_number(row, covariate) for row in metric_rows],
                dtype=np.float64,
            )
            description: dict[str, Any] = {"dataset_count": len(metric_rows)}
            if len(metric_rows) < 2:
                description.update(
                    rho=None,
                    two_sided_p_value=None,
                    undefined_reason="fewer than two datasets",
                )
            elif np.all(values == values[0]):
                description.update(
                    rho=None,
                    two_sided_p_value=None,
                    undefined_reason="constant covariate",
                )
            elif np.all(delta == delta[0]):
                description.update(
                    rho=None,
                    two_sided_p_value=None,
                    undefined_reason="constant raw metric-error difference",
                )
            else:
                statistic = spearmanr(values, delta)
                rho, p_value = float(statistic.statistic), float(statistic.pvalue)
                if not math.isfinite(rho) or not math.isfinite(p_value):
                    raise ValueError("Spearman calculation returned non-finite values")
                description.update(
                    rho=rho,
                    two_sided_p_value=p_value,
                    undefined_reason=None,
                )
            result[metric][covariate] = description
    return result


def _verify_artifact(path: Path, record: Any, *, label: str) -> None:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} record is invalid")
    _real_file(path, label=label)
    expected_size = _positive_integer(record.get("size_bytes"), label=f"{label} size")
    expected_sha = _sha256(record.get("sha256"), label=f"{label} SHA256")
    if path.stat().st_size != expected_size or _sha256_file(path) != expected_sha:
        raise ValueError(f"{label} does not match its manifest binding")


def _task_artifacts(
    run_root: Path,
    aggregate: Mapping[str, Any],
    *,
    expected_tabarena_sha: str,
) -> dict[str, dict[str, Any]]:
    tasks_root = _real_directory(run_root / "tasks", label="tasks root")
    datasets = {
        str(dataset["task"]["dataset_name"]): dataset
        for dataset in aggregate["datasets"]
    }
    found: dict[str, dict[str, Any]] = {}
    for item in aggregate["task_artifacts"]:
        directory = tasks_root / str(item["directory"])
        _real_directory(directory, label="task artifact directory")
        manifest_path = directory / "manifest.json"
        _verify_artifact(
            manifest_path,
            {
                "sha256": item["manifest_sha256"],
                "size_bytes": item["manifest_size_bytes"],
            },
            label="task manifest",
        )
        manifest = _load_json_object(manifest_path, label="task manifest")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("task manifest lacks artifacts")
        task_path = directory / "task.json"
        _verify_artifact(task_path, artifacts.get("task.json"), label="task result")
        task_payload = _load_json_object(task_path, label="task result")
        task = task_payload.get("task")
        if not isinstance(task, dict):
            raise ValueError("task result lacks task metadata")
        name = task.get("dataset_name")
        if not isinstance(name, str) or name not in datasets or name in found:
            raise ValueError("task artifacts do not map one-to-one to aggregate datasets")
        expected_task = datasets[name]["task"]
        for key in (
            "dataset_name",
            "task_id",
            "metric",
            "problem_type",
            "fold",
            "repeat",
            "split_index",
            "split_regime",
        ):
            if task.get(key) != expected_task.get(key):
                raise ValueError(f"task artifact metadata differs for {name}: {key}")
        provenance = task_payload.get("code_provenance")
        if (
            not isinstance(provenance, dict)
            or provenance.get("tabarena_sha") != expected_tabarena_sha
        ):
            raise ValueError(f"task artifact TabArena provenance differs for {name}")
        results = task_payload.get("results")
        if not isinstance(results, dict):
            raise ValueError(f"task artifact lacks results for {name}")
        for arm in ("rope", "none"):
            if _finite_number(
                results.get(arm, {}).get("metric_error"),
                label=f"task artifact {name} {arm} error",
            ) != _finite_number(
                datasets[name]["results"][arm]["metric_error"],
                label=f"aggregate {name} {arm} error",
            ):
                raise ValueError(f"task artifact metric differs for {name} {arm}")

        arm_records = artifacts.get("arms")
        if not isinstance(arm_records, dict) or set(arm_records) != {"rope", "none"}:
            raise ValueError(f"task manifest arm coverage differs for {name}")
        npz_paths: dict[str, Path] = {}
        for arm in ("rope", "none"):
            arm_manifest_path = directory / arm / "manifest.json"
            _verify_artifact(
                arm_manifest_path,
                arm_records[arm],
                label=f"{name} {arm} manifest",
            )
            arm_manifest = _load_json_object(
                arm_manifest_path, label=f"{name} {arm} manifest"
            )
            if arm_manifest.get("arm") != arm:
                raise ValueError(f"arm manifest identity differs for {name} {arm}")
            arm_artifacts = arm_manifest.get("artifacts")
            if not isinstance(arm_artifacts, dict):
                raise ValueError(f"arm manifest artifacts are invalid for {name} {arm}")
            predictions = directory / arm / "predictions.npz"
            _verify_artifact(
                predictions,
                arm_artifacts.get("predictions.npz"),
                label=f"{name} {arm} predictions",
            )
            npz_paths[arm] = predictions
        found[name] = {"directory": directory, "npz_paths": npz_paths}
    if set(found) != set(datasets):
        raise ValueError("verified task artifacts do not exactly cover aggregate datasets")
    return found


def _npz_content_hashes(path: Path, *, label: str) -> tuple[str, str]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            values = []
            for key in ("train_content_sha256", "test_content_sha256"):
                if key not in archive:
                    raise ValueError(f"{label} lacks {key}")
                value = np.asarray(archive[key])
                if value.dtype != np.uint8 or value.shape != (32,):
                    raise ValueError(f"{label} {key} has invalid encoding")
                values.append(np.ascontiguousarray(value).tobytes().hex())
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(label):
            raise
        raise ValueError(f"{label} is not a valid prediction archive") from error
    return values[0], values[1]


def _verify_git_checkout(root: Path, *, expected_sha: str) -> str:
    root = _real_directory(root, label="TabArena checkout")
    expected_sha = expected_sha.lower()
    if len(expected_sha) != 40 or any(c not in "0123456789abcdef" for c in expected_sha):
        raise ValueError("expected TabArena SHA must be a full lowercase Git SHA")
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = completed.stdout.splitlines()
    if len(lines) != 2 or Path(lines[0]).resolve() != root.resolve():
        raise ValueError("TabArena checkout is not the requested Git root")
    if lines[1] != expected_sha:
        raise ValueError("TabArena checkout HEAD differs from the expected SHA")
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    )
    if dirty.stdout.strip():
        raise ValueError("TabArena checkout has tracked modifications")
    return expected_sha


def _open_tabarena(
    checkout_root: Path,
    *,
    openml_cache_root: Path,
) -> tuple[Any, Any]:
    source_root = checkout_root / "packages" / "tabarena" / "src"
    _real_directory(source_root, label="TabArena source root")
    _real_directory(openml_cache_root, label="OpenML cache root")
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from tabarena.benchmark.task.openml.task_wrapper import OpenMLTaskWrapper
    from tabarena.caching import CacheConfig
    from tabarena.contexts import TabArenaContext
    import tabarena

    imported = Path(tabarena.__file__).resolve()
    if not imported.is_relative_to(source_root.resolve()):
        raise ValueError("imported TabArena package is outside the pinned checkout")
    cache = CacheConfig(
        openml=openml_cache_root,
        apply_on_run=True,
        scope_openml=True,
    )
    arena = TabArenaContext(methods=[], backend="native", cache_config=cache)
    return arena, OpenMLTaskWrapper


def _copy_duplicate_fields(
    row: dict[str, Any],
    summary: Mapping[str, Any],
    *,
    prefix: str,
) -> None:
    for key, value in summary.items():
        if key == "feature_count" and prefix == "all_supported":
            row["model_feature_count"] = value
        elif key == "feature_count":
            row[f"{prefix}_feature_count"] = value
        else:
            row[f"{prefix}_{key}"] = value


def _summaries_for_strata(
    rows: Sequence[Mapping[str, Any]],
    *,
    assignments: Mapping[str, str],
    bins: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for bin_definition in bins:
        label = str(bin_definition["label"])
        members = [
            row
            for row in rows
            if assignments[str(row["dataset_name"])] == label
        ]
        item = dict(bin_definition)
        item["comparison"] = summarize_rows(members)
        output.append(item)
    return output


def analyze_completed_run(
    *,
    run_root: Path,
    expected_aggregate_sha256: str,
    tabarena_root: Path,
    expected_tabarena_sha: str,
    openml_cache_root: Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Verify and stratify the frozen 500k pair entirely on CPU."""

    notify = progress or (lambda _message: None)
    run_root = _real_directory(run_root, label="run root")
    aggregate_path = run_root / "aggregate.json"
    aggregate = load_validated_aggregate(
        aggregate_path,
        expected_sha256=expected_aggregate_sha256,
    )
    aggregate_sha = _sha256_file(aggregate_path)
    tabarena_sha = _verify_git_checkout(
        tabarena_root, expected_sha=expected_tabarena_sha
    )
    if aggregate.get("code_provenance", {}).get("tabarena_sha") != tabarena_sha:
        raise ValueError("aggregate TabArena provenance differs from pinned checkout")
    artifacts = _task_artifacts(
        run_root,
        aggregate,
        expected_tabarena_sha=tabarena_sha,
    )
    notify("validated aggregate and all bound task/prediction artifacts")
    arena, wrapper_class = _open_tabarena(
        tabarena_root,
        openml_cache_root=openml_cache_root,
    )

    rows: list[dict[str, Any]] = []
    total = len(aggregate["datasets"])
    for index, dataset in enumerate(aggregate["datasets"], start=1):
        task = dataset["task"]
        name = str(task["dataset_name"])
        arm_hashes = {
            arm: _npz_content_hashes(
                artifacts[name]["npz_paths"][arm],
                label=f"{name} {arm} predictions",
            )
            for arm in ("rope", "none")
        }
        if arm_hashes["rope"] != arm_hashes["none"]:
            raise ValueError(f"RoPE/NoPE content hashes differ for {name}")
        expected_train_hash, expected_test_hash = arm_hashes["rope"]

        wrapper = wrapper_class.from_task_id(int(task["task_id"]))
        X_train, y_train, X_test, y_test = wrapper.get_train_test_split(
            fold=0,
            repeat=0,
            sample=0,
        )
        observed_train_hash = _table_content_fingerprint(X_train, y_train)
        observed_test_hash = _feature_table_fingerprint(X_test, domain="test")
        if observed_train_hash != expected_train_hash:
            raise ValueError(f"reloaded r0f0 training content differs for {name}")
        if observed_test_hash != expected_test_hash:
            raise ValueError(f"reloaded r0f0 test content differs for {name}")

        primary = all_supported_duplicate_summary(
            X_train, threshold=CORRELATION_THRESHOLD
        )
        numeric = numeric_only_duplicate_summary(
            X_train, threshold=CORRELATION_THRESHOLD
        )
        row: dict[str, Any] = {
            "dataset_name": name,
            "benchmark_dataset_id": task.get("benchmark_dataset_id"),
            "task_id": int(task["task_id"]),
            "metric": str(task["metric"]),
            "problem_type": str(task["problem_type"]),
            "train_row_count": len(X_train),
            "test_row_count": len(X_test),
            "train_content_sha256": observed_train_hash,
            "test_content_sha256": observed_test_hash,
            "rope_metric_error": _finite_number(
                dataset["results"]["rope"]["metric_error"],
                label=f"{name} RoPE metric error",
            ),
            "none_metric_error": _finite_number(
                dataset["results"]["none"]["metric_error"],
                label=f"{name} NoPE metric error",
            ),
        }
        row["rope_minus_none_metric_error"] = (
            row["rope_metric_error"] - row["none_metric_error"]
        )
        row["winner"] = (
            "rope"
            if row["rope_metric_error"] < row["none_metric_error"]
            else "none"
            if row["none_metric_error"] < row["rope_metric_error"]
            else "tie"
        )
        _copy_duplicate_fields(row, primary, prefix="all_supported")
        _copy_duplicate_fields(row, numeric, prefix="numeric_only")
        rows.append(row)
        notify(f"[{index:02d}/{total:02d}] verified and analyzed {name}")
        del wrapper, X_train, y_train, X_test, y_test
        gc.collect()

    feature_assignments, feature_bins = assign_equal_count_strata(
        rows,
        value_key="model_feature_count",
        bin_count=FEATURE_COUNT_BIN_COUNT,
        label_prefix="Q",
    )
    primary_assignments, primary_bins = assign_density_strata(
        rows,
        value_key="all_supported_near_duplicate_pair_density",
    )
    numeric_assignments, numeric_bins = assign_density_strata(
        rows,
        value_key="numeric_only_near_duplicate_pair_density",
    )
    for row in rows:
        name = str(row["dataset_name"])
        row["feature_count_stratum"] = feature_assignments[name]
        row["all_supported_density_stratum"] = primary_assignments[name]
        row["numeric_only_density_stratum"] = numeric_assignments[name]

    covariates = (
        "model_feature_count",
        "all_supported_near_duplicate_pair_density",
        "all_supported_duplicate_column_fraction",
        "numeric_only_near_duplicate_pair_density",
        "numeric_only_duplicate_column_fraction",
    )
    implementation_sha = _sha256_file(Path(__file__).resolve())
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "tabarena_rope_none_cpu_stratification",
        "study_role": "descriptive_post_hoc_analysis_of_frozen_predictions",
        "source": {
            "pair_id": aggregate.get("pair_id"),
            "suite": aggregate.get("suite"),
            "subset": aggregate.get("subset"),
            "aggregate_sha256": aggregate_sha,
            "pair_manifest_sha256": aggregate.get("pair_manifest_sha256"),
            "roster_sha256": aggregate.get("roster_sha256"),
            "shard_plan_sha256": aggregate.get("shard_plan_sha256"),
            "code_provenance": aggregate.get("code_provenance"),
            "checkpoints": aggregate.get("checkpoints"),
            "formal_eligible": aggregate.get("formal_eligible"),
        },
        "implementation_sha256": implementation_sha,
        "validation": {
            "aggregate_complete": True,
            "expected_and_observed_task_count": EXPECTED_TASK_COUNT,
            "expected_and_observed_result_count": EXPECTED_RESULT_COUNT,
            "tabarena_checkout_sha": tabarena_sha,
            "split": "r0f0",
            "prediction_arms_checked": ["rope", "none"],
            "canonical_train_content_hash_verified_for_every_task_and_arm": True,
            "canonical_test_content_hash_verified_for_every_task_and_arm": True,
            "verified_task_count": len(rows),
        },
        "method": {
            "model_facing_selector": (
                "number/string/object/category/boolean columns with at least one "
                "non-missing value, preserving source order"
            ),
            "near_duplicate_split": "training split only",
            "near_duplicate_rule": "absolute Pearson correlation > 0.95",
            "near_duplicate_threshold": CORRELATION_THRESHOLD,
            "primary_encoding": (
                "all model-facing columns; mean-imputed numeric values and "
                "deterministic ordinal-encoded categorical/string/boolean values"
            ),
            "numeric_only_sensitivity": "same selector restricted to numeric columns",
            "pair_density_denominator": "model-facing column pairs n_features choose 2",
            "feature_count_strata": (
                "four deterministic equal-count rank bins sorted by feature count "
                "then dataset name; boundary tie splits are reported"
            ),
            "density_strata": (
                "zero density retained as a control stratum; positive-density tasks "
                "split into deterministic equal-count low/high strata"
            ),
            "cross_metric_statistics": (
                "wins, mean ranks, and exact sign tests only; raw error differences "
                "are never pooled across metrics"
            ),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "continuous_statistics": (
                "metric-specific two-sided Spearman descriptions; exploratory and "
                "unadjusted for multiplicity"
            ),
            "categorical_caveat": (
                "Pearson correlation after ordinal coding is label-order-sensitive; "
                "numeric-only results are the prespecified sensitivity analysis"
            ),
        },
        "overall": summarize_rows(rows),
        "feature_count_equal_count_strata": _summaries_for_strata(
            rows,
            assignments=feature_assignments,
            bins=feature_bins,
        ),
        "all_supported_density_strata": _summaries_for_strata(
            rows,
            assignments=primary_assignments,
            bins=primary_bins,
        ),
        "numeric_only_density_strata_sensitivity": _summaries_for_strata(
            rows,
            assignments=numeric_assignments,
            bins=numeric_bins,
        ),
        "continuous_spearman_by_metric": continuous_spearman_descriptions(
            rows,
            covariates=covariates,
        ),
        "datasets": sorted(rows, key=lambda row: str(row["dataset_name"])),
    }
    # Refuse NaN/Infinity now, before any output file can be mutated.
    json.dumps(payload, allow_nan=False)
    del arena
    return payload


def _fmt_number(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}g}"


def render_markdown(payload: Mapping[str, Any]) -> str:
    """Render a concise, path-free report from the portable JSON payload."""

    validation = payload["validation"]
    overall = payload["overall"]
    scale = overall["scale_free"]
    lines = [
        "# TabArena 500k RoPE/NoPE CPU stratification",
        "",
        "This is a descriptive post-hoc analysis of the frozen predictions. It does "
        "not replace the terminal Stage 2/3 paired evaluation.",
        "",
        "## Validation",
        "",
        f"- Aggregate SHA256: `{payload['source']['aggregate_sha256']}`",
        f"- Complete coverage: {validation['verified_task_count']} tasks / "
        f"{validation['expected_and_observed_result_count']} arm results",
        "- Every r0f0 task was reopened from the pinned TabArena cache; canonical "
        "train/test hashes matched both RoPE and NoPE prediction archives.",
        f"- Pinned TabArena commit: `{validation['tabarena_checkout_sha']}`",
        "",
        "## Overall scale-free comparison",
        "",
        f"RoPE {scale['rope_wins']} wins, NoPE {scale['none_wins']} wins, "
        f"{scale['ties']} ties; exact two-sided sign-test "
        f"p={_fmt_number(scale['two_sided_exact_sign_test_p'])}.",
        "",
        "Raw metric-error differences use `RoPE - NoPE` (negative favors RoPE) "
        "and are kept separate by metric:",
        "",
        "| Metric | n | Mean delta | Median delta | Bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, raw in overall["raw_metric_error_difference_by_metric"].items():
        low, high = raw["paired_bootstrap_95ci"]
        lines.append(
            f"| {metric} | {raw['dataset_count']} | {_fmt_number(raw['mean'])} | "
            f"{_fmt_number(raw['median'])} | [{_fmt_number(low)}, "
            f"{_fmt_number(high)}] |"
        )

    def add_strata(title: str, strata: Sequence[Mapping[str, Any]]) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Stratum | Range | n | RoPE wins | NoPE wins | Ties | Sign p |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in strata:
            item_scale = item["comparison"]["scale_free"]
            tie_note = "*" if item.get("boundary_tie_split_after") else ""
            lines.append(
                f"| {item['label']}{tie_note} | {_fmt_number(item['minimum'])}–"
                f"{_fmt_number(item['maximum'])} | {item['dataset_count']} | "
                f"{item_scale['rope_wins']} | {item_scale['none_wins']} | "
                f"{item_scale['ties']} | "
                f"{_fmt_number(item_scale['two_sided_exact_sign_test_p'])} |"
            )
        if any(item.get("boundary_tie_split_after") for item in strata):
            lines.extend(
                [
                    "",
                    "*Note:* Equal-count ranking split an identical boundary value; dataset "
                    "name provides the deterministic tie-break.",
                ]
            )

    add_strata(
        "Feature-count equal-count strata",
        payload["feature_count_equal_count_strata"],
    )
    add_strata(
        "Near-duplicate pair-density strata (all supported columns)",
        payload["all_supported_density_strata"],
    )
    add_strata(
        "Near-duplicate pair-density strata (numeric-only sensitivity)",
        payload["numeric_only_density_strata_sensitivity"],
    )

    lines.extend(
        [
            "",
            "## Continuous metric-specific descriptions",
            "",
            "Spearman rho relates each covariate to `RoPE - NoPE` error. Positive rho "
            "means larger covariate values shift the observed difference toward NoPE; "
            "these p-values are descriptive and unadjusted.",
            "",
            "| Metric | Covariate | n | rho | p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for metric, descriptions in payload["continuous_spearman_by_metric"].items():
        for covariate, description in descriptions.items():
            reason = description.get("undefined_reason")
            rho = reason if reason else _fmt_number(description["rho"])
            p_value = "NA" if reason else _fmt_number(description["two_sided_p_value"])
            lines.append(
                f"| {metric} | {covariate} | {description['dataset_count']} | "
                f"{rho} | {p_value} |"
            )

    lines.extend(
        [
            "",
            "## Dataset audit table",
            "",
            "| Dataset | Metric | Features | All-supported pair density | "
            "Numeric-only pair density | RoPE-NoPE error | Winner |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["datasets"]:
        lines.append(
            f"| {row['dataset_name']} | {row['metric']} | "
            f"{row['model_feature_count']} | "
            f"{_fmt_number(row['all_supported_near_duplicate_pair_density'])} | "
            f"{_fmt_number(row['numeric_only_near_duplicate_pair_density'])} | "
            f"{_fmt_number(row['rope_minus_none_metric_error'])} | "
            f"{row['winner']} |"
        )
    lines.extend(
        [
            "",
            "Categorical/string/boolean columns are ordinal-encoded in the primary "
            "legacy-compatible analysis, so their Pearson correlations depend on label "
            "order. Treat the numeric-only analysis as the sensitivity check.",
            "",
        ]
    )
    return "\n".join(lines)


def write_new_outputs(
    payload: Mapping[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """Write new private outputs without deleting or overwriting failed evidence."""

    for path, label in ((json_path, "JSON output"), (markdown_path, "Markdown output")):
        if not path.is_absolute():
            raise ValueError(f"{label} must be absolute")
        _real_directory(path.parent, label=f"{label} parent")
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"{label} already exists; refusing to overwrite")
    json_text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    markdown_text = render_markdown(payload)
    # The full analysis and both serializations succeed before the first mutation.
    json_path.write_text(json_text, encoding="utf-8")
    markdown_path.write_text(markdown_text, encoding="utf-8")
