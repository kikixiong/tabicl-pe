#!/usr/bin/env python3
"""Run matched RoPE/Fingerprint checkpoints on a bounded BeyondArena slice.

This is an exploratory, classification-only comparison.  It deliberately builds
the exact job grid before materialization so a smoke run cannot accidentally
download the complete 142-dataset suite.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


ARMS = ("rope", "fingerprint", "released")
DEFAULT_DATASETS = (
    "blood_transfusion",
    "parkinsons_biomedical_voice_measurements",
    "ghanas_indigenous_intel",
)
RELEASED_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"


def _absolute(value: str, *, name: str, directory: bool = False) -> Path:
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


def _git_head(root: Path) -> str:
    import subprocess

    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise ValueError(f"source checkout is not clean: {root}")
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _checkpoint_contract(
    rope: Path,
    fingerprint: Path,
    *,
    comparison_step: int,
) -> dict[str, Any]:
    import torch

    expected_architecture = {
        "embed_dim": 128,
        "col_num_blocks": 3,
        "col_nhead": 8,
        "col_num_inds": 128,
        "row_num_blocks": 3,
        "row_nhead": 8,
        "icl_num_blocks": 12,
        "icl_nhead": 8,
    }
    if comparison_step <= 0:
        raise ValueError("comparison_step must be positive")
    payloads = {
        arm: torch.load(path, map_location="cpu", weights_only=True)
        for arm, path in (("rope", rope), ("fingerprint", fingerprint))
    }
    for arm, payload in payloads.items():
        if payload.get("curr_step") != comparison_step:
            raise ValueError(f"{arm} checkpoint is not at step {comparison_step}")
        if not isinstance(payload.get("config"), dict) or not isinstance(
            payload.get("state_dict"), dict
        ):
            raise ValueError(f"{arm} checkpoint lacks model config/state")
        stream = payload.get("prior_stream")
        if not isinstance(stream, dict) or (
            stream.get("cursor") != comparison_step
            or stream.get("experiment_seed") != 42
            or stream.get("ddp_rank") != 0
            or stream.get("world_size") != 1
        ):
            raise ValueError(f"{arm} checkpoint prior stream is invalid")
    if payloads["rope"]["prior_stream"] != payloads["fingerprint"]["prior_stream"]:
        raise ValueError("checkpoint prior streams are not exactly matched")
    configs = {arm: dict(payload["config"]) for arm, payload in payloads.items()}
    if (
        configs["rope"].get("row_identity_mode") != "rope"
        or configs["rope"].get("row_fingerprint") is not False
    ):
        raise ValueError("RoPE checkpoint treatment is invalid")
    if (
        configs["fingerprint"].get("row_identity_mode") != "none"
        or configs["fingerprint"].get("row_fingerprint") is not True
        or configs["fingerprint"].get("row_fingerprint_dim") != 16
    ):
        raise ValueError("Fingerprint checkpoint treatment is invalid")
    for arm, config in configs.items():
        observed = {key: config.get(key) for key in expected_architecture}
        if observed != expected_architecture:
            raise ValueError(f"{arm} checkpoint is not the expected full-size architecture")
    treatment = {"row_identity_mode", "row_fingerprint"}
    if {k: v for k, v in configs["rope"].items() if k not in treatment} != {
        k: v for k, v in configs["fingerprint"].items() if k not in treatment
    }:
        raise ValueError("checkpoint model configs differ outside the treatment")
    return {
        arm: {
            "curr_step": int(payloads[arm]["curr_step"]),
            "model_state_tensors": len(payloads[arm]["state_dict"]),
            "model_state_elements": sum(
                tensor.numel() for tensor in payloads[arm]["state_dict"].values()
            ),
            "model_config": configs[arm],
        }
        for arm in ("rope", "fingerprint")
    }


def _select_dataset_metadata(
    metadata: Any, requested: Sequence[str]
) -> tuple[tuple[str, ...], dict[str, dict[str, Any]]]:
    if not requested or any(not isinstance(name, str) or not name for name in requested):
        raise ValueError("at least one non-empty dataset name is required")
    if len(requested) != len(set(requested)):
        raise ValueError("dataset names must be unique")
    selected = metadata[metadata["dataset_name"].isin(requested)].copy()
    observed = set(selected["dataset_name"].astype(str))
    missing = sorted(set(requested) - observed)
    if missing:
        raise ValueError(f"BeyondArena lacks requested datasets: {missing}")
    if len(selected) != len(requested):
        raise ValueError("BeyondArena dataset metadata is not one-to-one")
    if not set(selected["problem_type"]).issubset({"binary", "multiclass"}):
        raise ValueError("BeyondArena smoke datasets must be classification tasks")
    if any(int(value) != 0 for value in selected["num_text_cols"]):
        raise ValueError("BeyondArena smoke datasets must not contain text columns")
    expected: dict[str, dict[str, Any]] = {}
    by_public_name: dict[str, str] = {}
    for row in selected.to_dict(orient="records"):
        dataset = str(row["dataset"])
        public_name = str(row["dataset_name"])
        by_public_name[public_name] = dataset
        expected[dataset] = {
            "dataset_name": public_name,
            "task_id": int(row["tid"]),
            "problem_type": str(row["problem_type"]),
            "metric": str(row["eval_metric"]),
            "split_regime": "iid" if row["task_type"] == "random" else str(row["task_type"]),
            "num_instances": int(row["num_instances"]),
            "num_features": int(row["n_features"]),
        }
    roster = tuple(by_public_name[name] for name in requested)
    if len(expected) != len(requested):
        raise ValueError("BeyondArena resolved duplicate dataset identifiers")
    return roster, expected


def _mean_ranks_by_regime(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    roster: Sequence[str],
    expected: Mapping[str, Mapping[str, Any]],
    mean_ranks: Any,
) -> dict[str, dict[str, float]]:
    regimes = sorted({str(expected[dataset]["split_regime"]) for dataset in roster})
    return {
        regime: mean_ranks(
            rows,
            roster=tuple(
                dataset
                for dataset in roster
                if expected[dataset]["split_regime"] == regime
            ),
            arm_order=ARMS,
        )
        for regime in regimes
    }


def _normalize_classification_target(y: Any) -> Any:
    """Return a one-dimensional scalar Series for sklearn-compatible targets.

    Data Foundry grouped tasks may retain a pandas categorical/extension dtype
    after the group-aware split.  Converting through Python scalars preserves
    the labels while avoiding sklearn's ``unknown`` target classification.
    """
    import numpy as np
    import pandas as pd
    from sklearn.utils.multiclass import type_of_target

    values = np.asarray(y)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError("BeyondArena classification target must be one-dimensional")
    index = getattr(y, "index", None)
    if index is not None and len(index) != len(values):
        raise ValueError("BeyondArena classification target index has the wrong length")
    normalized = pd.Series(
        values.tolist(),
        index=index,
        name=getattr(y, "name", None),
    )
    if type_of_target(normalized) not in {"binary", "multiclass"}:
        raise ValueError("BeyondArena classification target is not discrete")
    return normalized


def _make_beyond_system_model(external_system_model: type, base_factory: Any) -> type:
    base = base_factory(external_system_model)

    class BeyondFixedTabICLSystem(base):
        def _fit_system(self, X: Any, y: Any, **kwargs: Any) -> Any:
            return super()._fit_system(
                X,
                _normalize_classification_target(y),
                **kwargs,
            )

    return BeyondFixedTabICLSystem


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--tabarena-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--rope-checkpoint", required=True)
    parser.add_argument("--rope-sha256", required=True)
    parser.add_argument("--fingerprint-checkpoint", required=True)
    parser.add_argument("--fingerprint-sha256", required=True)
    parser.add_argument("--released-checkpoint", required=True)
    parser.add_argument("--comparison-step", required=True, type=int)
    parser.add_argument("--expected-model-sha", required=True)
    parser.add_argument("--expected-tabarena-sha", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", action="append", default=[])
    return parser


def main() -> int:
    args = _parser().parse_args()
    analysis_root = _absolute(args.analysis_root, name="analysis_root", directory=True)
    model_root = _absolute(args.model_root, name="model_root", directory=True)
    tabarena_root = _absolute(args.tabarena_root, name="tabarena_root", directory=True)
    cache_root = _absolute(args.cache_root, name="cache_root", directory=True)
    rope = _absolute(args.rope_checkpoint, name="rope_checkpoint")
    fingerprint = _absolute(args.fingerprint_checkpoint, name="fingerprint_checkpoint")
    released = _absolute(args.released_checkpoint, name="released_checkpoint")
    output = Path(args.output_dir)
    if not output.is_absolute() or output.exists():
        raise ValueError("output_dir must be an absent absolute path")

    checkpoints = {"rope": rope, "fingerprint": fingerprint, "released": released}
    expected_digests = {
        "rope": args.rope_sha256,
        "fingerprint": args.fingerprint_sha256,
        "released": RELEASED_SHA256,
    }
    observed_digests = {arm: _sha256(path) for arm, path in checkpoints.items()}
    if observed_digests != expected_digests:
        raise ValueError(f"checkpoint digest mismatch: {observed_digests}")
    contract = _checkpoint_contract(rope, fingerprint, comparison_step=args.comparison_step)

    analysis_sha = _git_head(analysis_root)
    model_sha = _git_head(model_root)
    tabarena_sha = _git_head(tabarena_root)
    if model_sha != args.expected_model_sha:
        raise ValueError(f"model source SHA mismatch: {model_sha} != {args.expected_model_sha}")
    if tabarena_sha != args.expected_tabarena_sha:
        raise ValueError(
            f"TabArena source SHA mismatch: {tabarena_sha} != {args.expected_tabarena_sha}"
        )

    sys.path.insert(0, str(analysis_root / "analysis/pe_mechanism/src"))
    from pe_mechanism.tabarena_evaluation import (
        _FIXED_CLASSIFIER_OPTIONS,
        _archive_directory,
        _import_runtime,
        _load_cached_results,
        _make_system_model,
        _mean_ranks,
        _normalize_results,
        _runtime_summary,
        _scale_free_comparison,
    )

    runtime = dict(_import_runtime(tabarena_root, model_root))
    runtime["BeyondArenaExperimentBundle"] = importlib.import_module(
        "tabarena.benchmark.experiment"
    ).BeyondArenaExperimentBundle
    runtime["BeyondArenaContext"] = importlib.import_module(
        "tabarena.contexts"
    ).BeyondArenaContext

    requested = tuple(args.dataset_name) if args.dataset_name else DEFAULT_DATASETS
    persistent_cache = runtime["CacheConfig"].from_root(cache_root)
    arena = runtime["BeyondArenaContext"](
        methods=[], backend="native", cache_config=persistent_cache
    )
    roster, expected_tasks = _select_dataset_metadata(arena.task_metadata, requested)

    model_cls = _make_beyond_system_model(
        runtime["ExternalSystemModel"], _make_system_model
    )
    names = {
        "rope": f"TabICL_Fullsize_RoPE_Step{args.comparison_step}",
        "fingerprint": f"TabICL_Fullsize_Fingerprint_Step{args.comparison_step}",
        "released": "TabICL_Released_Reference",
    }
    framework_to_arm: dict[str, str] = {}
    experiments = []
    for arm in ARMS:
        generator = runtime["SystemConfigGenerator"](
            model_cls=model_cls,
            name=names[arm],
            manual_configs=[
                {
                    "checkpoint": str(checkpoints[arm]),
                    "arm": arm,
                    "device": "cuda",
                    "n_estimators": 1,
                    "seed": 42,
                    "classifier_options": dict(_FIXED_CLASSIFIER_OPTIONS),
                }
            ],
        )
        built = runtime["BeyondArenaExperimentBundle"](
            models=[(generator, 0)], system_experiments=True
        ).build_experiments()
        if len(built) != 1 or built[0].name in framework_to_arm:
            raise RuntimeError("BeyondArena did not build one unique experiment per arm")
        framework_to_arm[built[0].name] = arm
        experiments.extend(built)

    jobs = arena.build_jobs(
        experiments,
        subset=["lite"],
        dataset_names=list(requested),
        problem_types=["binary", "multiclass"],
    )
    expected_count = len(ARMS) * len(roster)
    observed_job_keys = {
        (job.experiment.name, job.task.dataset, job.task.fold, job.task.repeat)
        for job in jobs
    }
    expected_job_keys = {
        (framework, dataset, 0, 0)
        for framework in framework_to_arm
        for dataset in roster
    }
    if (
        len(jobs) != expected_count
        or len(observed_job_keys) != len(jobs)
        or observed_job_keys != expected_job_keys
    ):
        raise RuntimeError("BeyondArena job matrix differs from the bounded lite grid")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        results_root = staging / "results"
        started = time.monotonic()
        results = arena.run_jobs(
            jobs,
            expname=results_root,
            register=False,
            cache_mode="ignore",
            debug_mode=True,
            raise_on_failure=True,
        )
        duration = time.monotonic() - started
        normalized = _normalize_results(
            results,
            framework_to_arm=framework_to_arm,
            roster=roster,
            expected_count=expected_count,
            expected_tasks=expected_tasks,
            arm_order=ARMS,
        )
        cached = _normalize_results(
            _load_cached_results(results_root),
            framework_to_arm=framework_to_arm,
            roster=roster,
            expected_count=expected_count,
            expected_tasks=expected_tasks,
            arm_order=ARMS,
        )
        if normalized != cached:
            raise RuntimeError("returned and cached BeyondArena results differ")
        rows = {(row["arm"], row["dataset"]): row for row in normalized}
        pairs = (("rope", "fingerprint"), ("rope", "released"), ("fingerprint", "released"))
        summary = {
            "schema_version": 1,
            "study": "fullsize-fingerprint-beyondarena-exploratory",
            "formal_eligible": False,
            "leaderboard_replication": False,
            "comparison_step": args.comparison_step,
            "seed": 42,
            "task_subset": "lite-explicit-roster",
            "task_count": len(roster),
            "result_count": len(normalized),
            "arm_order": list(ARMS),
            "overall_mean_rank_lower_is_better": _mean_ranks(
                rows, roster=roster, arm_order=ARMS
            ),
            "mean_rank_by_split_regime": _mean_ranks_by_regime(
                rows,
                roster=roster,
                expected=expected_tasks,
                mean_ranks=_mean_ranks,
            ),
            "overall_scale_free_comparisons": [
                _scale_free_comparison(rows, roster=roster, left=left, right=right)
                for left, right in pairs
            ],
            "datasets": [
                {
                    "dataset": expected_tasks[dataset]["dataset_name"],
                    "benchmark_dataset_id": dataset,
                    "task_id": expected_tasks[dataset]["task_id"],
                    "problem_type": expected_tasks[dataset]["problem_type"],
                    "metric": expected_tasks[dataset]["metric"],
                    "split_regime": expected_tasks[dataset]["split_regime"],
                    "num_instances": expected_tasks[dataset]["num_instances"],
                    "num_features": expected_tasks[dataset]["num_features"],
                    "arms": {
                        arm: {
                            key: rows[(arm, dataset)][key]
                            for key in ("metric_error", "time_train_s", "time_infer_s")
                        }
                        for arm in ARMS
                    },
                }
                for dataset in roster
            ],
            "checkpoint_digests": {
                arm: {
                    "sha256": observed_digests[arm],
                    "size_bytes": checkpoints[arm].stat().st_size,
                }
                for arm in ARMS
            },
            "checkpoint_contract": contract,
            "code_provenance": {
                "analysis_sha": analysis_sha,
                "model_sha": model_sha,
                "tabarena_sha": tabarena_sha,
            },
            "inference_budget": {
                "n_estimators": 1,
                "augmentation": "none",
                "classifier_options": dict(_FIXED_CLASSIFIER_OPTIONS),
            },
            "interpretation_limits": (
                "Bounded exploratory smoke only. It is neither the BeyondArena core protocol "
                "nor a leaderboard comparison, and it does not replace column-permutation tests."
            ),
        }
        runtime_summary = _runtime_summary(
            runtime=runtime,
            benchmark_sha=tabarena_sha,
            duration_seconds=duration,
            result_count=len(normalized),
        )
        runtime_summary["data_foundry"] = importlib.metadata.version("data-foundry")
        _archive_directory(results_root, staging / "results.tar.gz")
        shutil.rmtree(results_root)
        for name, payload in (("summary.json", summary), ("runtime.json", runtime_summary)):
            (staging / name).write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        manifest = {
            "schema_version": 1,
            "study": summary["study"],
            "formal_eligible": False,
            "artifacts": {
                name: {
                    "sha256": _sha256(staging / name),
                    "size_bytes": (staging / name).stat().st_size,
                }
                for name in ("summary.json", "runtime.json", "results.tar.gz")
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"output_dir": str(output), "task_count": len(roster)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
