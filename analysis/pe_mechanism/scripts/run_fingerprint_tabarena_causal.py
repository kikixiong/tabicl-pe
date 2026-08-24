#!/usr/bin/env python3
"""Run four matched fingerprint interventions on the frozen TabArena-lite roster."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping


COMPARISON_STEP = 50_000
CHECKPOINT_SHA256 = "9848a89bf8724c9bda0e16d797cf38b476399609f706c6989073eec64bc33a3b"
MODEL_SHA = "a2ae49a828e2f7f10c9e393b7b59ab47b388da72"
TABARENA_SHA = "c987d91556a14d4c9b3383c35d1b0ec68ff81883"
ROSTER_SHA256 = "3c26133c8b986aba530624b5c7a5dee42e4b29938bc09afa04f76b6462c4adf1"


def _absolute(value: str, *, name: str, directory: bool) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    resolved = path.resolve(strict=True)
    if path.is_symlink() or (directory and not resolved.is_dir()) or (
        not directory and not resolved.is_file()
    ):
        raise ValueError(f"{name} has the wrong file type")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(root: Path, *, expected: str, label: str) -> str:
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise ValueError(f"{label} source checkout is not clean: {root}")
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected:
        raise ValueError(f"{label} source SHA mismatch: {head} != {expected}")
    return head


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--tabarena-root", required=True)
    parser.add_argument("--openml-cache", required=True)
    parser.add_argument("--fingerprint-checkpoint", required=True)
    parser.add_argument("--fingerprint-sha256", default=CHECKPOINT_SHA256)
    parser.add_argument("--expected-analysis-sha", required=True)
    parser.add_argument("--expected-model-sha", default=MODEL_SHA)
    parser.add_argument("--expected-tabarena-sha", default=TABARENA_SHA)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser


def _capture_manifest(
    captures: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    root: Path,
    interventions: tuple[str, ...],
    roster: tuple[str, ...],
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for intervention in interventions:
        for dataset in roster:
            record = captures[intervention][dataset]
            digest = record["dataset_sha256"]
            for name in ("capture.json", "predictions.npz"):
                path = root / intervention / digest / name
                relative = path.relative_to(root.parent).as_posix()
                files[relative] = {
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
    return {
        "schema_version": 1,
        "private": True,
        "prediction_dtype": "float32",
        "file_count": len(files),
        "files": files,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"JSON destination must be fresh: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _retain_failed_run(staging: Path, output: Path, error: BaseException) -> None:
    failure = {
        "schema_version": 1,
        "study": "fullsize-fingerprint-step50000-tabarena-causal",
        "status": "failed",
        "requested_output_dir": str(output),
        "retained_staging_dir": str(staging),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    try:
        _write_json(staging / "failure.json", failure)
    except BaseException as failure_error:
        print(
            f"could not write retained failure metadata: {failure_error}",
            file=sys.stderr,
        )
    print(f"retained failed causal run at {staging}", file=sys.stderr)


def main() -> int:
    args = _parser().parse_args()
    if args.bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    analysis_root = _absolute(args.analysis_root, name="analysis_root", directory=True)
    model_root = _absolute(args.model_root, name="model_root", directory=True)
    tabarena_root = _absolute(args.tabarena_root, name="tabarena_root", directory=True)
    openml_cache = _absolute(args.openml_cache, name="openml_cache", directory=True)
    checkpoint = _absolute(
        args.fingerprint_checkpoint,
        name="fingerprint_checkpoint",
        directory=False,
    )
    output = Path(args.output_dir)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ValueError("output_dir must be an absent absolute path")
    output_parent = output.parent.resolve(strict=True)
    output = output_parent / output.name
    sources = (analysis_root, model_root, tabarena_root)
    for source in sources:
        if _is_within(output, source) or _is_within(source, output):
            raise ValueError("output_dir and source checkouts must be disjoint")
    if _is_within(output, openml_cache) or _is_within(openml_cache, output):
        raise ValueError("output_dir and OpenML cache must be disjoint")

    expected_checkpoint_sha = args.fingerprint_sha256.lower()
    if expected_checkpoint_sha != CHECKPOINT_SHA256:
        raise ValueError("fingerprint digest differs from the frozen 50k checkpoint")
    if args.expected_model_sha != MODEL_SHA:
        raise ValueError("model SHA differs from the frozen fingerprint checkout")
    if args.expected_tabarena_sha != TABARENA_SHA:
        raise ValueError("TabArena SHA differs from the frozen evaluation checkout")
    observed_checkpoint_sha = _sha256(checkpoint)
    if observed_checkpoint_sha != expected_checkpoint_sha:
        raise ValueError("fingerprint checkpoint digest mismatch")
    analysis_sha = _git_head(
        analysis_root,
        expected=args.expected_analysis_sha,
        label="analysis",
    )
    model_sha = _git_head(
        model_root,
        expected=args.expected_model_sha,
        label="model",
    )
    tabarena_sha = _git_head(
        tabarena_root,
        expected=args.expected_tabarena_sha,
        label="TabArena",
    )

    package_root = analysis_root / "analysis" / "pe_mechanism" / "src"
    sys.path.insert(0, str(package_root))
    from pe_mechanism.fingerprint_causal import (
        FINGERPRINT_INTERVENTIONS,
        aggregate_causal_results,
        fingerprint_checkpoint_contract,
        load_and_validate_captures,
        make_causal_prediction_runner,
        make_fingerprint_causal_system,
    )
    from pe_mechanism.tabarena_evaluation import (
        _FIXED_CLASSIFIER_OPTIONS,
        _archive_directory,
        _import_runtime,
        _load_cached_results,
        _make_system_model,
        _normalize_results,
        _runtime_summary,
        _validate_context_tasks,
    )

    contract = fingerprint_checkpoint_contract(
        checkpoint,
        comparison_step=COMPARISON_STEP,
    )
    roster_path = (
        analysis_root
        / "analysis/pe_mechanism/manifests/tabarena-v0.1-classification-roster.json"
    )
    if _sha256(roster_path) != ROSTER_SHA256:
        raise ValueError("TabArena roster file digest mismatch")
    roster_payload = json.loads(roster_path.read_text(encoding="utf-8"))
    full_roster = tuple(roster_payload["names"])
    if len(full_roster) != 38 or len(set(full_roster)) != 38:
        raise ValueError("TabArena classification roster is not the frozen 38-task set")
    if args.dataset:
        if len(args.dataset) != len(set(args.dataset)):
            raise ValueError("requested datasets must be unique")
        unknown = sorted(set(args.dataset) - set(full_roster))
        if unknown:
            raise ValueError(f"unknown datasets: {unknown}")
        requested = set(args.dataset)
        roster = tuple(dataset for dataset in full_roster if dataset in requested)
    else:
        roster = full_roster

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output_parent))
    try:
        staging.chmod(0o700)
        capture_root = (staging / "private_predictions").resolve()
        scratch = staging / "scratch"
        results_root = staging / "results"
        runtime = _import_runtime(tabarena_root, model_root)
        external_runner = importlib.import_module(
            "tabarena.benchmark.experiment"
        ).OOFExperimentRunner
        prediction_runner = make_causal_prediction_runner(external_runner)
        model_cls = make_fingerprint_causal_system(
            runtime["ExternalSystemModel"], _make_system_model
        )
        framework_to_arm: dict[str, str] = {}
        experiments = []
        for intervention in FINGERPRINT_INTERVENTIONS:
            name = (
                "TabICL_Fullsize_Fingerprint_Step50000_Causal_"
                + intervention.capitalize()
            )
            generator = runtime["SystemConfigGenerator"](
                model_cls=model_cls,
                name=name,
                manual_configs=[
                    {
                        "checkpoint": str(checkpoint),
                        "arm": intervention,
                        "intervention": intervention,
                        "device": "cuda",
                        "n_estimators": 1,
                        "seed": 42,
                        "classifier_options": dict(_FIXED_CLASSIFIER_OPTIONS),
                    }
                ],
            )
            built = runtime["TabArenaV0pt1ExperimentBundle"](
                models=[(generator, 0)],
                system_experiments=True,
                model_artifacts_base_path=scratch / "model-artifacts",
            ).build_experiments()
            if len(built) != 1 or built[0].name in framework_to_arm:
                raise RuntimeError("TabArena did not build one unique causal experiment")
            built[0].experiment_cls = prediction_runner
            built[0].experiment_kwargs = {
                **built[0].experiment_kwargs,
                "causal_capture_root": str(capture_root),
                "causal_intervention": intervention,
            }
            framework_to_arm[built[0].name] = intervention
            experiments.extend(built)

        cache = runtime["CacheConfig"](
            openml=openml_cache,
            huggingface=scratch / "huggingface",
            data_foundry=scratch / "data-foundry",
            tabarena=scratch / "tabarena",
            results=results_root,
            apply_on_run=True,
            scope_openml=True,
        )
        arena = runtime["TabArenaContext"](
            methods=[], backend="native", cache_config=cache
        )
        expected_tasks = _validate_context_tasks(arena, roster)
        jobs = arena.build_jobs(
            experiments,
            subset=["lite"],
            dataset_names=list(roster),
            problem_types=["binary", "multiclass"],
        )
        expected_count = len(roster) * len(FINGERPRINT_INTERVENTIONS)
        expected_job_keys = {
            (framework, dataset, 0, 0)
            for framework in framework_to_arm
            for dataset in roster
        }
        observed_job_keys = {
            (job.experiment.name, job.task.dataset, job.task.fold, job.task.repeat)
            for job in jobs
        }
        if len(jobs) != expected_count or observed_job_keys != expected_job_keys:
            raise RuntimeError("TabArena job matrix differs from the frozen causal grid")

        started = time.monotonic()
        returned = arena.run_jobs(
            jobs,
            expname=results_root,
            register=False,
            cache_mode="ignore",
            debug_mode=True,
            raise_on_failure=True,
        )
        duration = time.monotonic() - started
        normalize_kwargs = {
            "framework_to_arm": framework_to_arm,
            "roster": roster,
            "expected_count": expected_count,
            "expected_tasks": expected_tasks,
            "arm_order": FINGERPRINT_INTERVENTIONS,
        }
        normalized = _normalize_results(returned, **normalize_kwargs)
        cached = _normalize_results(
            _load_cached_results(results_root), **normalize_kwargs
        )
        if normalized != cached:
            raise RuntimeError("returned and cached TabArena results differ")
        if _sha256(checkpoint) != expected_checkpoint_sha:
            raise RuntimeError("fingerprint checkpoint changed during evaluation")

        captures = load_and_validate_captures(capture_root, roster=roster)
        rows = {(row["arm"], row["dataset"]): row for row in normalized}
        comparisons = aggregate_causal_results(
            rows,
            roster=roster,
            n_resamples=args.bootstrap_resamples,
            seed=42,
        )
        h1_datasets = [
            dataset
            for dataset in roster
            if captures["permuted"][dataset]["causal_metadata"][
                "feature_token_count"
            ]
            == 1
        ]
        datasets = [
            {
                "dataset": dataset,
                "task_id": rows[("correct", dataset)]["task_id"],
                "problem_type": rows[("correct", dataset)]["problem_type"],
                "metric": rows[("correct", dataset)]["metric"],
                "feature_token_count": captures["correct"][dataset][
                    "causal_metadata"
                ]["feature_token_count"],
                "conditions": {
                    intervention: {
                        key: rows[(intervention, dataset)][key]
                        for key in ("metric_error", "time_train_s", "time_infer_s")
                    }
                    for intervention in FINGERPRINT_INTERVENTIONS
                },
            }
            for dataset in roster
        ]
        summary = {
            "schema_version": 1,
            "study": "fullsize-fingerprint-step50000-tabarena-causal",
            "formal_eligible": False,
            "leaderboard_replication": False,
            "seed": 42,
            "comparison_step": COMPARISON_STEP,
            "task_subset": "lite",
            "task_count": len(roster),
            "result_count": len(normalized),
            "condition_order": list(FINGERPRINT_INTERVENTIONS),
            **comparisons,
            "permutation_audit": {
                "algorithm": "cyclic_shift_left_one_v1",
                "h_greater_than_one_has_no_fixed_points": True,
                "single_token_degenerate_dataset_count": len(h1_datasets),
                "single_token_degenerate_datasets": h1_datasets,
            },
            "datasets": datasets,
            "checkpoint": {
                "sha256": observed_checkpoint_sha,
                "size_bytes": checkpoint.stat().st_size,
                "contract": contract,
            },
            "code_provenance": {
                "analysis_sha": analysis_sha,
                "model_sha": model_sha,
                "tabarena_sha": tabarena_sha,
                "roster_file_sha256": ROSTER_SHA256,
            },
            "inference_budget": {
                "n_estimators": 1,
                "augmentation": "none",
                "classifier_options": dict(_FIXED_CLASSIFIER_OPTIONS),
            },
            "prediction_capture": {
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
            },
        }
        runtime_summary = _runtime_summary(
            runtime=runtime,
            benchmark_sha=tabarena_sha,
            duration_seconds=duration,
            result_count=len(normalized),
        )
        captures_manifest = _capture_manifest(
            captures,
            root=capture_root,
            interventions=FINGERPRINT_INTERVENTIONS,
            roster=roster,
        )
        _archive_directory(results_root, staging / "results.tar.gz")
        shutil.rmtree(results_root)
        shutil.rmtree(scratch, ignore_errors=True)
        _write_json(staging / "summary.json", summary)
        _write_json(staging / "runtime.json", runtime_summary)
        _write_json(staging / "captures_manifest.json", captures_manifest)
        artifact_names = (
            "summary.json",
            "runtime.json",
            "results.tar.gz",
            "captures_manifest.json",
        )
        manifest = {
            "schema_version": 1,
            "study": summary["study"],
            "formal_eligible": False,
            "contains_private_predictions": True,
            "artifacts": {
                name: {
                    "sha256": _sha256(staging / name),
                    "size_bytes": (staging / name).stat().st_size,
                }
                for name in artifact_names
            },
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, output)
    except BaseException as error:
        _retain_failed_run(staging, output, error)
        raise
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "task_count": len(roster),
                "result_count": len(roster) * 4,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
