"""Independent post-publication verification for the fingerprint causal run."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import Any, Mapping, Sequence

from .fingerprint_causal import (
    FINGERPRINT_INTERVENTIONS,
    aggregate_causal_results,
    fingerprint_checkpoint_contract,
    load_and_validate_captures,
)
from .provenance import VerifiedFile, verify_file
from .tabarena_evaluation import (
    _FIXED_CLASSIFIER_OPTIONS,
    _load_cached_results,
    _normalize_results,
)


STUDY = "fullsize-fingerprint-step50000-tabarena-causal"
COMPARISON_STEP = 50_000
EXPECTED_CHECKPOINT_SHA256 = (
    "9848a89bf8724c9bda0e16d797cf38b476399609f706c6989073eec64bc33a3b"
)
EXPECTED_RUN_ANALYSIS_SHA = "db29bb8582d516b94043ebfb2038c5ceaa3082a7"
EXPECTED_MODEL_SHA = "a2ae49a828e2f7f10c9e393b7b59ab47b388da72"
EXPECTED_TABARENA_SHA = "c987d91556a14d4c9b3383c35d1b0ec68ff81883"
EXPECTED_ROSTER_FILE_SHA256 = (
    "3c26133c8b986aba530624b5c7a5dee42e4b29938bc09afa04f76b6462c4adf1"
)
EXPECTED_TASK_COUNT = 38
BOOTSTRAP_RESAMPLES = 10_000
_FRAMEWORK_PREFIX = "TabICL_Fullsize_Fingerprint_Step50000_Causal_"
_ARTIFACT_NAMES = (
    "summary.json",
    "runtime.json",
    "results.tar.gz",
    "captures_manifest.json",
)
_TOP_LEVEL_NAMES = frozenset((*_ARTIFACT_NAMES, "manifest.json", "private_predictions"))
_SUMMARY_FIELDS = {
    "schema_version",
    "study",
    "formal_eligible",
    "leaderboard_replication",
    "seed",
    "comparison_step",
    "task_subset",
    "task_count",
    "result_count",
    "condition_order",
    "overall_scale_free_correct_vs_interventions",
    "metric_groups",
    "permutation_audit",
    "datasets",
    "checkpoint",
    "code_provenance",
    "inference_budget",
    "prediction_capture",
}
_RUNTIME_FIELDS = {
    "schema_version",
    "python",
    "numpy",
    "pandas",
    "scikit_learn",
    "openml",
    "autogluon_core",
    "torch",
    "tabarena",
    "tabicl",
    "benchmark_code_sha",
    "cuda_available",
    "cuda_device_count",
    "cuda_device_name",
    "cuda_device_capability",
    "cuda_runtime",
    "cudnn",
    "nvidia_driver",
    "duration_seconds",
    "result_count",
}


def _object(
    value: object,
    *,
    label: str,
    fields: set[str] | frozenset[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if fields is not None and set(value) != set(fields):
        missing = sorted(set(fields) - set(value))
        extra = sorted(set(value) - set(fields))
        raise ValueError(f"{label} fields mismatch: missing={missing}, extra={extra}")
    return value


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON object contains duplicate key: {key}")
        value[key] = item
    return value


def _load_json(path: Path, *, label: str) -> tuple[Mapping[str, Any], VerifiedFile]:
    verified = verify_file(path)
    try:
        value = json.loads(verified.read_bytes(), object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    verified.assert_unchanged()
    return _object(value, label=label), verified


def _finite_number(value: object, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise ValueError(f"{label} must be finite")
    return number


def _validate_run_tree(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("run directory must be an absolute real directory")
    root = path.resolve(strict=True)
    if not root.is_dir() or os.path.ismount(root):
        raise ValueError("run directory must be an absolute real directory")
    entries = list(root.iterdir())
    if {entry.name for entry in entries} != _TOP_LEVEL_NAMES:
        raise ValueError("causal run top-level entries differ from the frozen contract")
    for entry in root.rglob("*"):
        if entry.is_symlink() or not (entry.is_file() or entry.is_dir()):
            raise ValueError("causal run contains a symlink or special file")
    for name in (*_ARTIFACT_NAMES, "manifest.json"):
        if not (root / name).is_file():
            raise ValueError(f"causal run is missing a regular {name}")
    if not (root / "private_predictions").is_dir():
        raise ValueError("causal run is missing private_predictions")
    return root


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return whether two resolved paths contain one another."""
    return left == right or left in right.parents or right in left.parents


def _validate_top_manifest(root: Path) -> tuple[Mapping[str, Any], dict[str, VerifiedFile]]:
    manifest, manifest_file = _load_json(root / "manifest.json", label="run manifest")
    _object(
        manifest,
        label="run manifest",
        fields={
            "schema_version",
            "study",
            "formal_eligible",
            "contains_private_predictions",
            "artifacts",
        },
    )
    if (
        manifest["schema_version"] != 1
        or manifest["study"] != STUDY
        or manifest["formal_eligible"] is not False
        or manifest["contains_private_predictions"] is not True
    ):
        raise ValueError("run manifest violates the frozen causal contract")
    artifacts = _object(manifest["artifacts"], label="run manifest artifacts")
    if set(artifacts) != set(_ARTIFACT_NAMES):
        raise ValueError("run manifest artifact set is not exact")
    verified: dict[str, VerifiedFile] = {"manifest.json": manifest_file}
    for name in _ARTIFACT_NAMES:
        entry = _object(
            artifacts[name],
            label=f"run manifest artifact {name}",
            fields={"sha256", "size_bytes"},
        )
        file = verify_file(root / name, expected_sha256=entry["sha256"])
        if entry["size_bytes"] != file.digest.size_bytes:
            raise ValueError(f"run manifest size differs for {name}")
        verified[name] = file
    manifest_file.assert_unchanged()
    return manifest, verified


def _validate_roster(
    path: Path,
    *,
    expected_sha256: str,
    expected_tabarena_sha: str,
    expected_task_count: int,
) -> tuple[str, ...]:
    payload, verified = _load_json(path, label="TabArena roster")
    if verified.digest.sha256 != expected_sha256:
        raise ValueError("TabArena roster file digest mismatch")
    names = payload.get("names")
    if not isinstance(names, list) or not all(
        isinstance(name, str) and name for name in names
    ):
        raise ValueError("TabArena roster names are invalid")
    roster = tuple(names)
    if (
        payload.get("count") != expected_task_count
        or len(roster) != expected_task_count
        or len(set(roster)) != expected_task_count
        or payload.get("source_commit") != expected_tabarena_sha
    ):
        raise ValueError("TabArena roster violates the frozen causal contract")
    return roster


def _validate_capture_manifest(
    root: Path,
    payload: Mapping[str, Any],
    *,
    roster: tuple[str, ...],
) -> dict[str, dict[str, dict[str, Any]]]:
    _object(
        payload,
        label="capture manifest",
        fields={"schema_version", "private", "prediction_dtype", "file_count", "files"},
    )
    files = _object(payload["files"], label="capture manifest files")
    expected_paths = {
        f"private_predictions/{intervention}/"
        f"{hashlib.sha256(dataset.encode('utf-8')).hexdigest()}/{name}"
        for intervention in FINGERPRINT_INTERVENTIONS
        for dataset in roster
        for name in ("capture.json", "predictions.npz")
    }
    if (
        payload["schema_version"] != 1
        or payload["private"] is not True
        or payload["prediction_dtype"] != "float32"
        or payload["file_count"] != len(expected_paths)
        or set(files) != expected_paths
    ):
        raise ValueError("capture manifest violates the frozen causal grid")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in (root / "private_predictions").rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError("private prediction files differ from the capture manifest")
    for relative in sorted(expected_paths):
        entry = _object(
            files[relative],
            label=f"capture artifact {relative}",
            fields={"sha256", "size_bytes"},
        )
        file = verify_file(root / relative, expected_sha256=entry["sha256"])
        if entry["size_bytes"] != file.digest.size_bytes:
            raise ValueError(f"capture artifact size differs for {relative}")
    return load_and_validate_captures(root / "private_predictions", roster=roster)


def _safe_extract_results(archive_bytes: bytes, destination: Path) -> Path:
    seen: set[str] = set()
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or not relative.parts
                or relative.parts[0] != "tabarena-results"
                or any(part in {"", ".", ".."} for part in relative.parts)
                or not (member.isdir() or member.isfile())
                or relative.as_posix() in seen
            ):
                raise ValueError("results archive contains an unsafe member")
            seen.add(relative.as_posix())
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                if target.exists() and not target.is_dir():
                    raise ValueError("results archive contains a path collision")
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                raise ValueError("results archive contains a path collision")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("results archive member cannot be read")
            with source, target.open("xb") as handle:
                shutil.copyfileobj(source, handle)
            if target.stat().st_size != member.size:
                raise ValueError("results archive member size changed during extraction")
    results_root = destination / "tabarena-results"
    if not results_root.is_dir():
        raise ValueError("results archive lacks tabarena-results")
    return results_root


def _validate_summary_core(
    summary: Mapping[str, Any],
    *,
    roster: tuple[str, ...],
    captures: Mapping[str, Mapping[str, Mapping[str, Any]]],
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    expected_analysis_sha: str,
    expected_model_sha: str,
    expected_tabarena_sha: str,
    expected_roster_sha256: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    _object(summary, label="causal summary", fields=_SUMMARY_FIELDS)
    expected_result_count = len(roster) * len(FINGERPRINT_INTERVENTIONS)
    if (
        summary["schema_version"] != 1
        or summary["study"] != STUDY
        or summary["formal_eligible"] is not False
        or summary["leaderboard_replication"] is not False
        or summary["seed"] != 42
        or summary["comparison_step"] != COMPARISON_STEP
        or summary["task_subset"] != "lite"
        or summary["task_count"] != len(roster)
        or summary["result_count"] != expected_result_count
        or summary["condition_order"] != list(FINGERPRINT_INTERVENTIONS)
    ):
        raise ValueError("causal summary violates the frozen study contract")

    checkpoint_file = verify_file(checkpoint, expected_sha256=expected_checkpoint_sha256)
    checkpoint_summary = _object(
        summary["checkpoint"],
        label="causal summary checkpoint",
        fields={"sha256", "size_bytes", "contract"},
    )
    contract = fingerprint_checkpoint_contract(checkpoint, comparison_step=COMPARISON_STEP)
    checkpoint_file.assert_unchanged()
    if checkpoint_summary != {
        "sha256": checkpoint_file.digest.sha256,
        "size_bytes": checkpoint_file.digest.size_bytes,
        "contract": contract,
    }:
        raise ValueError("causal summary checkpoint evidence differs from exact bytes")

    if _object(summary["code_provenance"], label="code provenance") != {
        "analysis_sha": expected_analysis_sha,
        "model_sha": expected_model_sha,
        "tabarena_sha": expected_tabarena_sha,
        "roster_file_sha256": expected_roster_sha256,
    }:
        raise ValueError("causal summary code provenance mismatch")
    if _object(summary["inference_budget"], label="inference budget") != {
        "n_estimators": 1,
        "augmentation": "none",
        "classifier_options": dict(_FIXED_CLASSIFIER_OPTIONS),
    }:
        raise ValueError("causal summary inference budget mismatch")
    if _object(summary["prediction_capture"], label="prediction capture") != {
        "private": True,
        "dtype": "float32",
        "four_way_alignment_fields": [
            "shape",
            "encoded_target",
            "row",
            "test_target",
            "class",
            "train_content",
            "test_content",
        ],
    }:
        raise ValueError("causal summary prediction-capture contract mismatch")

    h1 = [
        dataset
        for dataset in roster
        if captures["permuted"][dataset]["causal_metadata"]["feature_token_count"] == 1
    ]
    if _object(summary["permutation_audit"], label="permutation audit") != {
        "algorithm": "cyclic_shift_left_one_v1",
        "h_greater_than_one_has_no_fixed_points": True,
        "single_token_degenerate_dataset_count": len(h1),
        "single_token_degenerate_datasets": h1,
    }:
        raise ValueError("causal summary permutation audit mismatch")

    datasets = summary["datasets"]
    if not isinstance(datasets, list) or [item.get("dataset") for item in datasets] != list(
        roster
    ):
        raise ValueError("causal summary dataset order differs from the frozen roster")
    by_dataset: dict[str, Mapping[str, Any]] = {}
    expected_tasks: dict[str, Mapping[str, Any]] = {}
    for item in datasets:
        row = _object(
            item,
            label="causal dataset summary",
            fields={
                "dataset",
                "task_id",
                "problem_type",
                "metric",
                "feature_token_count",
                "conditions",
            },
        )
        dataset = row["dataset"]
        if (
            isinstance(row["task_id"], bool)
            or not isinstance(row["task_id"], int)
            or row["problem_type"] not in {"binary", "multiclass"}
            or not isinstance(row["metric"], str)
            or not row["metric"]
            or row["feature_token_count"]
            != captures["correct"][dataset]["causal_metadata"]["feature_token_count"]
        ):
            raise ValueError(f"causal dataset metadata is invalid for {dataset}")
        conditions = _object(row["conditions"], label=f"conditions for {dataset}")
        if set(conditions) != set(FINGERPRINT_INTERVENTIONS):
            raise ValueError(f"causal condition grid is incomplete for {dataset}")
        for intervention, values in conditions.items():
            condition = _object(
                values,
                label=f"{dataset}/{intervention}",
                fields={"metric_error", "time_train_s", "time_infer_s"},
            )
            for name, value in condition.items():
                _finite_number(value, label=f"{dataset}/{intervention}/{name}")
        by_dataset[dataset] = row
        expected_tasks[dataset] = {
            "task_id": row["task_id"],
            "problem_type": row["problem_type"],
            "metric": row["metric"],
        }
    return by_dataset, expected_tasks


def _validate_runtime(
    runtime: Mapping[str, Any],
    *,
    expected_tabarena_sha: str,
    expected_result_count: int,
) -> None:
    _object(runtime, label="runtime summary", fields=_RUNTIME_FIELDS)
    version_fields = (
        "python",
        "numpy",
        "pandas",
        "scikit_learn",
        "openml",
        "autogluon_core",
        "torch",
        "tabarena",
        "tabicl",
        "cuda_runtime",
        "nvidia_driver",
    )
    if any(not isinstance(runtime[name], str) or not runtime[name] for name in version_fields):
        raise ValueError("runtime summary contains an invalid version field")
    if (
        runtime["schema_version"] != 1
        or runtime["benchmark_code_sha"] != expected_tabarena_sha
        or runtime["cuda_available"] is not True
        or runtime["cuda_device_count"] != 1
        or runtime["cuda_device_name"] != "NVIDIA A10"
        or runtime["cuda_device_capability"] != [8, 6]
        or isinstance(runtime["cudnn"], bool)
        or not isinstance(runtime["cudnn"], int)
        or runtime["result_count"] != expected_result_count
    ):
        raise ValueError("runtime summary is not the frozen single-A10 execution")
    _finite_number(runtime["duration_seconds"], label="duration_seconds", positive=True)


def verify_fingerprint_causal_run(
    run_dir: Path,
    *,
    checkpoint: Path,
    roster_path: Path,
    scratch_root: Path,
    expected_checkpoint_sha256: str = EXPECTED_CHECKPOINT_SHA256,
    expected_analysis_sha: str = EXPECTED_RUN_ANALYSIS_SHA,
    expected_model_sha: str = EXPECTED_MODEL_SHA,
    expected_tabarena_sha: str = EXPECTED_TABARENA_SHA,
    expected_roster_sha256: str = EXPECTED_ROSTER_FILE_SHA256,
    expected_task_count: int = EXPECTED_TASK_COUNT,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Re-hash, reload and independently recompute one published causal run."""
    root = _validate_run_tree(run_dir)
    if not scratch_root.is_absolute() or scratch_root.is_symlink():
        raise ValueError("scratch_root must be an absolute real directory")
    scratch = scratch_root.resolve(strict=True)
    if not scratch.is_dir() or _paths_overlap(root, scratch):
        raise ValueError("scratch_root must be an absolute real directory")
    roster = _validate_roster(
        roster_path,
        expected_sha256=expected_roster_sha256,
        expected_tabarena_sha=expected_tabarena_sha,
        expected_task_count=expected_task_count,
    )
    _, artifact_files = _validate_top_manifest(root)
    summary, summary_file = _load_json(root / "summary.json", label="causal summary")
    runtime, runtime_file = _load_json(root / "runtime.json", label="runtime summary")
    capture_manifest, capture_manifest_file = _load_json(
        root / "captures_manifest.json", label="capture manifest"
    )
    for name, file in (
        ("summary.json", summary_file),
        ("runtime.json", runtime_file),
        ("captures_manifest.json", capture_manifest_file),
    ):
        if file.digest != artifact_files[name].digest:
            raise ValueError(f"{name} changed after manifest verification")

    captures = _validate_capture_manifest(root, capture_manifest, roster=roster)
    by_dataset, expected_tasks = _validate_summary_core(
        summary,
        roster=roster,
        captures=captures,
        checkpoint=checkpoint,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_analysis_sha=expected_analysis_sha,
        expected_model_sha=expected_model_sha,
        expected_tabarena_sha=expected_tabarena_sha,
        expected_roster_sha256=expected_roster_sha256,
    )
    expected_result_count = len(roster) * len(FINGERPRINT_INTERVENTIONS)
    _validate_runtime(
        runtime,
        expected_tabarena_sha=expected_tabarena_sha,
        expected_result_count=expected_result_count,
    )

    archive_bytes = artifact_files["results.tar.gz"].read_bytes()
    with tempfile.TemporaryDirectory(prefix="fingerprint-causal-verify-", dir=scratch) as temp:
        results_root = _safe_extract_results(archive_bytes, Path(temp))
        normalized = _normalize_results(
            _load_cached_results(results_root),
            framework_to_arm={
                f"{_FRAMEWORK_PREFIX}{intervention.capitalize()}_c1_default": intervention
                for intervention in FINGERPRINT_INTERVENTIONS
            },
            roster=roster,
            expected_count=expected_result_count,
            expected_tasks=expected_tasks,
            arm_order=FINGERPRINT_INTERVENTIONS,
        )
    rows = {(row["arm"], row["dataset"]): row for row in normalized}
    for intervention in FINGERPRINT_INTERVENTIONS:
        for dataset in roster:
            result = rows[(intervention, dataset)]
            summary_row = by_dataset[dataset]
            if any(
                result[name] != summary_row[name]
                for name in ("task_id", "problem_type", "metric")
            ) or any(
                result[name] != summary_row["conditions"][intervention][name]
                for name in ("metric_error", "time_train_s", "time_infer_s")
            ):
                raise ValueError("result archive differs from the causal summary")
    comparisons = aggregate_causal_results(
        rows,
        roster=roster,
        n_resamples=bootstrap_resamples,
        seed=42,
    )
    if comparisons != {
        "overall_scale_free_correct_vs_interventions": summary[
            "overall_scale_free_correct_vs_interventions"
        ],
        "metric_groups": summary["metric_groups"],
    }:
        raise ValueError("recomputed causal comparisons differ from the summary")
    comparison_bytes = json.dumps(
        comparisons,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    for file in artifact_files.values():
        file.assert_unchanged()
    capture_count = len(roster) * len(FINGERPRINT_INTERVENTIONS)
    return {
        "schema_version": 1,
        "kind": "fingerprint_causal_postpublication_verification",
        "status": "verified",
        "study": STUDY,
        "run_dir": str(root),
        "task_count": len(roster),
        "result_count": len(normalized),
        "capture_record_count": capture_count,
        "capture_file_count": capture_count * 2,
        "condition_order": list(FINGERPRINT_INTERVENTIONS),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "run_analysis_sha": expected_analysis_sha,
        "run_manifest_sha256": artifact_files["manifest.json"].digest.sha256,
        "artifact_sha256": {
            name: artifact_files[name].digest.sha256 for name in _ARTIFACT_NAMES
        },
        "bootstrap_resamples": bootstrap_resamples,
        "recomputed_comparisons_sha256": hashlib.sha256(comparison_bytes).hexdigest(),
    }
