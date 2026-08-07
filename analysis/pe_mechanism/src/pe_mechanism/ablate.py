"""Paired aggregation for internal method-ablation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import math
import re

from .identifiers import require_portable_identifier
from .provenance import (
    RunTransaction,
    assert_dataset_roster,
    load_verified_json_config,
    manifest_from_verified_inputs,
    verify_configured_run_inputs,
)
from .statistics import summarize_paired, summary_dict


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _load_json_bytes(raw: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must contain one UTF-8 JSON object") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _metric_map_from_payload(
    payload: Mapping[str, Any], metric: str, *, name: str
) -> dict[str, tuple[float, str]]:
    result: dict[str, tuple[float, str]] = {}
    for dataset, values in payload.items():
        if not isinstance(dataset, str) or not isinstance(values, dict) or metric not in values:
            raise ValueError(f"invalid metric record in {name}")
        portable_dataset = require_portable_identifier(dataset, name="dataset identifier")
        metric_value = float(values[metric])
        if not math.isfinite(metric_value):
            raise ValueError(f"metric {metric!r} for dataset {portable_dataset!r} must be finite")
        roster_sha256 = values.get("prediction_roster_sha256")
        if not isinstance(roster_sha256, str) or not _SHA256_PATTERN.fullmatch(
            roster_sha256
        ):
            raise ValueError(
                f"dataset {portable_dataset!r} requires a lowercase prediction_roster_sha256"
            )
        result[portable_dataset] = (metric_value, roster_sha256)
    if not result:
        raise ValueError(f"{name} must contain at least one dataset metric record")
    return result


def _metric_map(path: Path, metric: str) -> dict[str, tuple[float, str]]:
    return _metric_map_from_payload(_load_json_object(path), metric, name="metric input")


def _summarize_metric_maps(
    *,
    metric: str,
    higher_is_better: bool,
    baseline: Mapping[str, tuple[float, str]],
    conditions: list[tuple[str, Mapping[str, tuple[float, str]]]],
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    datasets = sorted(baseline)
    for offset, (name, observed) in enumerate(conditions):
        if observed.keys() != baseline.keys():
            raise ValueError(f"condition {name!r} dataset roster differs from baseline")
        mismatched = [
            dataset
            for dataset in datasets
            if observed[dataset][1] != baseline[dataset][1]
        ]
        if mismatched:
            raise ValueError(
                f"condition {name!r} prediction roster differs from baseline for "
                f"datasets: {mismatched}"
            )
        summary = summarize_paired(
            [baseline[dataset][0] for dataset in datasets],
            [observed[dataset][0] for dataset in datasets],
            higher_is_better=higher_is_better,
            n_resamples=bootstrap_resamples,
            seed=seed + offset,
        )
        summaries.append({"name": name, **summary_dict(summary)})
    pairing_payload = {
        dataset: baseline[dataset][1] for dataset in datasets
    }
    pairing_sha256 = hashlib.sha256(
        json.dumps(pairing_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "kind": "paired_method_ablation",
        "metric": metric,
        "higher_is_better": higher_is_better,
        "dataset_count": len(baseline),
        "prediction_pairing_sha256": pairing_sha256,
        "conditions": summaries,
    }


def summarize_conditions(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != 1:
        raise ValueError("ablation config schema_version must be 1")
    metric = require_portable_identifier(config.get("metric", "accuracy"), name="metric")
    higher_is_better = config.get("higher_is_better", True)
    if not isinstance(higher_is_better, bool):
        raise ValueError("higher_is_better must be a JSON boolean")
    baseline_path = Path(str(config.get("baseline_metrics", ""))).expanduser()
    if not baseline_path.is_absolute() or not baseline_path.is_file():
        raise ValueError("baseline_metrics must be an existing absolute path")
    baseline = _metric_map(baseline_path, metric)
    conditions = config.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("conditions must be a non-empty list")
    resolved_conditions: list[tuple[str, Mapping[str, tuple[float, str]]]] = []
    for condition in conditions:
        if not isinstance(condition, dict):
            raise ValueError("each condition must be an object")
        name = require_portable_identifier(condition.get("name", ""), name="condition name")
        path = Path(str(condition.get("metrics", ""))).expanduser()
        if not path.is_absolute() or not path.is_file():
            raise ValueError("condition name and absolute metrics path are required")
        observed = _metric_map(path, metric)
        resolved_conditions.append((name, observed))
    return _summarize_metric_maps(
        metric=metric,
        higher_is_better=higher_is_better,
        baseline=baseline,
        conditions=resolved_conditions,
        bootstrap_resamples=int(config.get("bootstrap_resamples", 10_000)),
        seed=int(config.get("seed", 42)),
    )


def _publish_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> int:
    configuration = load_verified_json_config(Path(args.config))
    config = dict(configuration.data)
    baseline_path, baseline_expected = _input_spec(
        config.get("baseline_metrics"), name="baseline_metrics"
    )
    conditions = config.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("conditions must be a non-empty list")
    additional_paths: dict[str, Path] = {"metrics.baseline": baseline_path}
    expected_hashes: dict[str, str] = {}
    if baseline_expected is not None:
        expected_hashes["metrics.baseline"] = baseline_expected
    verified_conditions: list[dict[str, Any]] = []
    for index, condition in enumerate(conditions):
        if not isinstance(condition, dict):
            raise ValueError("each condition must be an object")
        role = f"metrics.condition.{index:06d}"
        path, expected = _input_spec(condition.get("metrics"), name="condition metrics")
        additional_paths[role] = path
        if expected is not None:
            expected_hashes[role] = expected
        verified_conditions.append(dict(condition))

    seed = int(config.get("seed", 42))
    context = verify_configured_run_inputs(
        configuration,
        command="ablate",
        seed=seed,
        additional_input_paths=additional_paths,
        expected_additional_sha256=expected_hashes,
    )
    metric = require_portable_identifier(config.get("metric", "accuracy"), name="metric")
    higher_is_better = config.get("higher_is_better", True)
    if not isinstance(higher_is_better, bool):
        raise ValueError("higher_is_better must be a JSON boolean")
    baseline = _metric_map_from_payload(
        _load_json_bytes(
            context.additional_file("metrics.baseline").read_bytes(),
            name="verified baseline metrics",
        ),
        metric,
        name="verified baseline metrics",
    )
    assert_dataset_roster(
        context.inputs.dataset_manifest, tuple(sorted(baseline))
    )
    resolved_conditions: list[tuple[str, Mapping[str, tuple[float, str]]]] = []
    for index, condition in enumerate(verified_conditions):
        name = require_portable_identifier(
            condition.get("name", ""), name="condition name"
        )
        role = f"metrics.condition.{index:06d}"
        observed = _metric_map_from_payload(
            _load_json_bytes(
                context.additional_file(role).read_bytes(),
                name=f"verified condition metrics {index}",
            ),
            metric,
            name=f"verified condition metrics {index}",
        )
        resolved_conditions.append((name, observed))
    summary = _summarize_metric_maps(
        metric=metric,
        higher_is_better=higher_is_better,
        baseline=baseline,
        conditions=resolved_conditions,
        bootstrap_resamples=int(config.get("bootstrap_resamples", 10_000)),
        seed=seed,
    )
    roots = (
        context.inputs.training_code.root,
        context.inputs.model_code.root,
        context.inputs.analysis_code.root,
    )
    with RunTransaction(Path(args.output_dir), source_roots=roots) as transaction:
        _publish_json(transaction.staging_dir / "ablation-summary.json", summary)
        artifacts = transaction.artifact_digests(("ablation-summary.json",))
        manifest = manifest_from_verified_inputs(
            context.inputs,
            artifacts=artifacts,
        )
        transaction.commit(manifest, verified_inputs=context.inputs)
    return 0


def _input_spec(value: Any, *, name: str) -> tuple[Path, str | None]:
    expected: str | None = None
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - {"path", "expected_sha256"})
        if unknown or "path" not in value:
            raise ValueError(f"{name} input fields mismatch: unknown={unknown}")
        raw_path = value["path"]
        raw_expected = value.get("expected_sha256")
        if raw_expected is not None and not isinstance(raw_expected, str):
            raise ValueError(f"{name} expected_sha256 must be a string")
        expected = raw_expected
    else:
        raw_path = value
    path = Path(str(raw_path or "")).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must use an absolute path")
    return path, expected
