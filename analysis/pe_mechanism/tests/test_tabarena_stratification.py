from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from pe_mechanism.tabarena_stratification import (
    all_supported_duplicate_summary,
    assign_density_strata,
    assign_equal_count_strata,
    continuous_spearman_descriptions,
    duplicate_graph_summary,
    load_validated_aggregate,
    numeric_only_duplicate_summary,
    select_model_facing_columns,
    summarize_rows,
)


def _analysis_row(
    name: str,
    *,
    metric: str,
    rope: float,
    none: float,
    feature_count: int,
    density: float,
) -> dict[str, object]:
    return {
        "dataset_name": name,
        "metric": metric,
        "rope_metric_error": rope,
        "none_metric_error": none,
        "model_feature_count": feature_count,
        "all_supported_near_duplicate_pair_density": density,
    }


def test_selector_matches_model_facing_supported_dtypes_and_order() -> None:
    frame = pd.DataFrame(
        {
            "number": [1.0, 2.0, 3.0],
            "text": pd.Series(["a", "b", "c"], dtype="string"),
            "category": pd.Series(["x", "y", "x"], dtype="category"),
            "flag": pd.Series([True, False, True], dtype="boolean"),
            "all_missing": [np.nan, np.nan, np.nan],
            "timestamp": pd.date_range("2025-01-01", periods=3),
        }
    )

    selected = select_model_facing_columns(frame)

    assert selected.feature_columns == ("number", "text", "category", "flag")
    assert selected.numeric_columns == ("number",)
    assert selected.categorical_columns == ("text", "category", "flag")


def test_duplicate_graph_uses_strict_threshold_and_connected_components() -> None:
    correlation = np.asarray(
        [
            [1.0, 0.96, -0.97, 0.95],
            [0.96, 1.0, 0.20, 0.10],
            [-0.97, 0.20, 1.0, 0.00],
            [0.95, 0.10, 0.00, 1.0],
        ]
    )

    summary = duplicate_graph_summary(correlation, threshold=0.95)

    assert summary["near_duplicate_pair_count"] == 2
    assert summary["near_duplicate_pair_density"] == pytest.approx(2 / 6)
    assert summary["incident_column_count"] == 3
    assert summary["duplicate_column_fraction"] == pytest.approx(3 / 4)
    assert summary["maximum_connected_component_size"] == 3


def test_primary_encoding_includes_categories_and_numeric_sensitivity_does_not() -> None:
    frame = pd.DataFrame(
        {
            "number_a": [0.0, 1.0, np.nan, 3.0, 4.0, 5.0],
            "number_b": [0.0, 1.0, np.nan, 3.0, 4.0, 5.0],
            "category_a": ["a", "b", "a", "c", None, "b"],
            "category_b": ["a", "b", "a", "c", None, "b"],
        }
    )

    primary = all_supported_duplicate_summary(frame, threshold=0.95)
    numeric = numeric_only_duplicate_summary(frame, threshold=0.95)

    assert primary["feature_count"] == 4
    assert primary["near_duplicate_pair_count"] == 2
    assert primary["incident_column_count"] == 4
    assert numeric["feature_count"] == 2
    assert numeric["near_duplicate_pair_count"] == 1
    assert numeric["incident_column_count"] == 2


def test_equal_count_strata_are_balanced_and_deterministic() -> None:
    rows = [
        {
            "dataset_name": f"d{index:02d}",
            "model_feature_count": 10 - index,
        }
        for index in range(10)
    ]

    assignments_a, bins_a = assign_equal_count_strata(
        rows,
        value_key="model_feature_count",
        bin_count=4,
        label_prefix="Q",
    )
    assignments_b, bins_b = assign_equal_count_strata(
        list(reversed(rows)),
        value_key="model_feature_count",
        bin_count=4,
        label_prefix="Q",
    )

    assert assignments_a == assignments_b
    assert bins_a == bins_b
    assert [item["dataset_count"] for item in bins_a] == [3, 3, 2, 2]
    assert [item["label"] for item in bins_a] == ["Q1", "Q2", "Q3", "Q4"]
    assert max(
        row["model_feature_count"]
        for row in rows
        if assignments_a[row["dataset_name"]] == "Q1"
    ) <= min(
        row["model_feature_count"]
        for row in rows
        if assignments_a[row["dataset_name"]] == "Q2"
    )


def test_density_strata_preserve_zero_as_its_own_control_group() -> None:
    rows = [
        {"dataset_name": "z1", "density": 0.0},
        {"dataset_name": "z2", "density": 0.0},
        {"dataset_name": "p3", "density": 0.3},
        {"dataset_name": "p1", "density": 0.1},
        {"dataset_name": "p4", "density": 0.4},
        {"dataset_name": "p2", "density": 0.2},
    ]

    assignments, bins = assign_density_strata(rows, value_key="density")

    assert assignments["z1"] == assignments["z2"] == "zero"
    assert assignments["p1"] == assignments["p2"] == "positive_low"
    assert assignments["p3"] == assignments["p4"] == "positive_high"
    assert [item["label"] for item in bins] == [
        "zero",
        "positive_low",
        "positive_high",
    ]


def test_scale_free_summary_keeps_raw_deltas_separate_by_metric() -> None:
    rows = [
        _analysis_row(
            "auc-rope", metric="roc_auc", rope=0.10, none=0.20,
            feature_count=1, density=0.0,
        ),
        _analysis_row(
            "auc-none", metric="roc_auc", rope=0.30, none=0.20,
            feature_count=2, density=0.1,
        ),
        _analysis_row(
            "loss-rope", metric="log_loss", rope=0.50, none=0.60,
            feature_count=3, density=0.2,
        ),
        _analysis_row(
            "loss-none", metric="log_loss", rope=0.70, none=0.60,
            feature_count=4, density=0.3,
        ),
    ]

    result = summarize_rows(rows, bootstrap_resamples=100, bootstrap_seed=7)

    assert result["scale_free"]["rope_wins"] == 2
    assert result["scale_free"]["none_wins"] == 2
    assert result["scale_free"]["ties"] == 0
    assert result["scale_free"]["two_sided_exact_sign_test_p"] == 1.0
    assert result["raw_metric_error_difference"] is None
    assert result["raw_difference_omission_reason"].startswith("mixed metric")
    assert set(result["raw_metric_error_difference_by_metric"]) == {
        "log_loss",
        "roc_auc",
    }
    assert result["raw_metric_error_difference_by_metric"]["roc_auc"][
        "mean"
    ] == pytest.approx(0.0)


def test_spearman_descriptions_are_metric_specific_and_json_safe() -> None:
    rows = [
        _analysis_row(
            "a1", metric="roc_auc", rope=0.1, none=0.4,
            feature_count=1, density=0.1,
        ),
        _analysis_row(
            "a2", metric="roc_auc", rope=0.3, none=0.4,
            feature_count=2, density=0.2,
        ),
        _analysis_row(
            "a3", metric="roc_auc", rope=0.5, none=0.4,
            feature_count=3, density=0.3,
        ),
        _analysis_row(
            "l1", metric="log_loss", rope=0.2, none=0.3,
            feature_count=5, density=0.0,
        ),
        _analysis_row(
            "l2", metric="log_loss", rope=0.4, none=0.3,
            feature_count=5, density=0.1,
        ),
    ]

    result = continuous_spearman_descriptions(
        rows,
        covariates=("model_feature_count",),
    )

    assert result["roc_auc"]["model_feature_count"]["rho"] == pytest.approx(1.0)
    assert result["log_loss"]["model_feature_count"] == {
        "dataset_count": 2,
        "rho": None,
        "two_sided_p_value": None,
        "undefined_reason": "constant covariate",
    }
    json.dumps(result, allow_nan=False)


def test_aggregate_validation_fails_closed_on_completeness_and_counts(tmp_path) -> None:
    datasets = []
    artifacts = []
    for index, name in enumerate(("one", "two")):
        datasets.append(
            {
                "task": {
                    "dataset_name": name,
                    "task_id": index + 1,
                    "metric": "roc_auc",
                    "problem_type": "binary",
                    "fold": 0,
                    "repeat": 0,
                    "split_index": 0,
                    "split_regime": "iid",
                },
                "results": {
                    "rope": {"metric_error": 0.1},
                    "none": {"metric_error": 0.2},
                },
            }
        )
        artifacts.append(
            {
                "directory": f"000{index}-{name}",
                "manifest_sha256": "0" * 64,
                "manifest_size_bytes": 1,
            }
        )
    payload = {
        "schema_version": 1,
        "pair_id": "synthetic",
        "complete": True,
        "task_count": 2,
        "result_count": 4,
        "arm_order": ["rope", "none"],
        "datasets": datasets,
        "task_artifacts": artifacts,
    }
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(aggregate.read_bytes()).hexdigest()

    loaded = load_validated_aggregate(
        aggregate,
        expected_sha256=digest,
        expected_task_count=2,
        expected_result_count=4,
    )
    assert loaded["pair_id"] == "synthetic"

    payload["complete"] = False
    aggregate.write_text(json.dumps(payload), encoding="utf-8")
    incomplete_digest = hashlib.sha256(aggregate.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="complete"):
        load_validated_aggregate(
            aggregate,
            expected_sha256=incomplete_digest,
            expected_task_count=2,
            expected_result_count=4,
        )
