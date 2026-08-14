#!/usr/bin/env python3
"""Run the fixed TabArena-lite roster for matched RoPE/Fingerprint checkpoints.

This is deliberately exploratory.  It reuses the frozen single-estimator
inference budget and official 38-dataset roster from ``tabarena-evaluate`` but
does not claim leaderboard or formal-campaign evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any


ARMS = ("rope", "fingerprint", "released")
RELEASED_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
ROSTER_SHA256 = "3c26133c8b986aba530624b5c7a5dee42e4b29938bc09afa04f76b6462c4adf1"


def _absolute(value: str, *, name: str, directory: bool = False) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    resolved = path.resolve(strict=True)
    if (
        path.is_symlink()
        or (directory and not resolved.is_dir())
        or (not directory and not resolved.is_file())
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
    model_scale: str,
) -> dict[str, Any]:
    import torch

    expected_architectures = {
        "compact": {
            "embed_dim": 64,
            "col_num_blocks": 1,
            "col_nhead": 4,
            "col_num_inds": 32,
            "row_num_blocks": 2,
            "row_nhead": 4,
            "icl_num_blocks": 3,
            "icl_nhead": 4,
        },
        "fullsize": {
            "embed_dim": 128,
            "col_num_blocks": 3,
            "col_nhead": 8,
            "col_num_inds": 128,
            "row_num_blocks": 3,
            "row_nhead": 8,
            "icl_num_blocks": 12,
            "icl_nhead": 8,
        },
    }
    if comparison_step <= 0 or model_scale not in expected_architectures:
        raise ValueError("invalid comparison step or model scale")
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
        prior_stream = payload.get("prior_stream")
        if not isinstance(prior_stream, dict) or (
            prior_stream.get("cursor") != comparison_step
            or prior_stream.get("experiment_seed") != 42
            or prior_stream.get("ddp_rank") != 0
            or prior_stream.get("world_size") != 1
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
    ):
        raise ValueError("Fingerprint checkpoint treatment is invalid")
    if configs["fingerprint"].get("row_fingerprint_dim") != 16:
        raise ValueError("Fingerprint checkpoint dimension is not 16")
    expected_architecture = expected_architectures[model_scale]
    for arm, config in configs.items():
        observed_architecture = {key: config.get(key) for key in expected_architecture}
        if observed_architecture != expected_architecture:
            raise ValueError(
                f"{arm} checkpoint is not the expected {model_scale} architecture"
            )
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
                value.numel() for value in payloads[arm]["state_dict"].values()
            ),
            "model_config": configs[arm],
        }
        for arm in ("rope", "fingerprint")
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--tabarena-root", required=True)
    parser.add_argument("--openml-cache", required=True)
    parser.add_argument("--rope-checkpoint", required=True)
    parser.add_argument("--rope-sha256", required=True)
    parser.add_argument("--fingerprint-checkpoint", required=True)
    parser.add_argument("--fingerprint-sha256", required=True)
    parser.add_argument("--released-checkpoint", required=True)
    parser.add_argument("--comparison-step", required=True, type=int)
    parser.add_argument("--model-scale", required=True, choices=("compact", "fullsize"))
    parser.add_argument("--expected-model-sha", required=True)
    parser.add_argument("--expected-tabarena-sha", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", action="append", default=[])
    return parser


def main() -> int:
    args = _parser().parse_args()
    analysis_root = _absolute(args.analysis_root, name="analysis_root", directory=True)
    model_root = _absolute(args.model_root, name="model_root", directory=True)
    tabarena_root = _absolute(args.tabarena_root, name="tabarena_root", directory=True)
    openml_cache = _absolute(args.openml_cache, name="openml_cache", directory=True)
    rope = _absolute(args.rope_checkpoint, name="rope_checkpoint")
    fingerprint = _absolute(args.fingerprint_checkpoint, name="fingerprint_checkpoint")
    released = _absolute(args.released_checkpoint, name="released_checkpoint")
    output = Path(args.output_dir)
    if not output.is_absolute() or output.exists():
        raise ValueError("output_dir must be an absent absolute path")

    expected = {
        "rope": args.rope_sha256,
        "fingerprint": args.fingerprint_sha256,
        "released": RELEASED_SHA256,
    }
    checkpoints = {"rope": rope, "fingerprint": fingerprint, "released": released}
    observed = {arm: _sha256(path) for arm, path in checkpoints.items()}
    if observed != expected:
        raise ValueError(f"checkpoint digest mismatch: {observed}")

    contract = _checkpoint_contract(
        rope,
        fingerprint,
        comparison_step=args.comparison_step,
        model_scale=args.model_scale,
    )
    analysis_sha = _git_head(analysis_root)
    model_sha = _git_head(model_root)
    if model_sha != args.expected_model_sha:
        raise ValueError(
            f"model source SHA mismatch: {model_sha} != {args.expected_model_sha}"
        )
    tabarena_sha = _git_head(tabarena_root)
    if tabarena_sha != args.expected_tabarena_sha:
        raise ValueError(
            "TabArena source SHA mismatch: "
            f"{tabarena_sha} != {args.expected_tabarena_sha}"
        )

    roster_path = (
        analysis_root
        / "analysis/pe_mechanism/manifests/tabarena-v0.1-classification-roster.json"
    )
    if _sha256(roster_path) != ROSTER_SHA256:
        raise ValueError("TabArena roster file digest mismatch")
    roster_payload = json.loads(roster_path.read_text(encoding="utf-8"))
    full_roster = tuple(roster_payload["names"])
    if args.dataset:
        unknown = sorted(set(args.dataset) - set(full_roster))
        if unknown:
            raise ValueError(f"unknown datasets: {unknown}")
        roster = tuple(name for name in full_roster if name in set(args.dataset))
    else:
        roster = full_roster

    package_root = analysis_root / "analysis/pe_mechanism/src"
    sys.path.insert(0, str(package_root))
    from pe_mechanism.tabarena_evaluation import (
        _FIXED_CLASSIFIER_OPTIONS,
        _archive_directory,
        _import_runtime,
        _load_cached_results,
        _make_system_model,
        _mean_ranks,
        _normalize_results,
        _paired_comparison,
        _runtime_summary,
        _scale_free_comparison,
        _validate_context_tasks,
    )

    runtime = _import_runtime(tabarena_root, model_root)
    model_cls = _make_system_model(runtime["ExternalSystemModel"])
    scale_label = "Compact" if args.model_scale == "compact" else "Fullsize"
    names = {
        "rope": f"TabICL_{scale_label}_RoPE_Step{args.comparison_step}",
        "fingerprint": (f"TabICL_{scale_label}_Fingerprint_Step{args.comparison_step}"),
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
        built = runtime["TabArenaV0pt1ExperimentBundle"](
            models=[(generator, 0)], system_experiments=True
        ).build_experiments()
        if len(built) != 1:
            raise RuntimeError("TabArena did not build one experiment per arm")
        framework_to_arm[built[0].name] = arm
        experiments.extend(built)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        results_root = staging / "results"
        scratch = staging / "scratch"
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
        expected_count = len(roster) * len(ARMS)
        if len(jobs) != expected_count:
            raise RuntimeError(
                f"TabArena built {len(jobs)} jobs, expected {expected_count}"
            )
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
            raise RuntimeError("returned and cached TabArena results differ")

        rows = {(row["arm"], row["dataset"]): row for row in normalized}
        pairs = (
            ("rope", "fingerprint"),
            ("rope", "released"),
            ("fingerprint", "released"),
        )
        metric_groups: dict[str, Any] = {}
        for metric in sorted({row["metric"] for row in normalized}):
            metric_roster = tuple(
                dataset
                for dataset in roster
                if rows[("rope", dataset)]["metric"] == metric
            )
            metric_groups[metric] = {
                "dataset_count": len(metric_roster),
                "macro_mean_metric_error": {
                    arm: sum(rows[(arm, d)]["metric_error"] for d in metric_roster)
                    / len(metric_roster)
                    for arm in ARMS
                },
                "paired_comparisons": [
                    _paired_comparison(
                        rows,
                        roster=metric_roster,
                        left=left,
                        right=right,
                        seed=42 + index * 10,
                        n_resamples=10_000,
                    )
                    for index, (left, right) in enumerate(pairs, start=1)
                ],
            }
        datasets = [
            {
                "dataset": dataset,
                "task_id": rows[("rope", dataset)]["task_id"],
                "problem_type": rows[("rope", dataset)]["problem_type"],
                "metric": rows[("rope", dataset)]["metric"],
                "arms": {
                    arm: {
                        key: rows[(arm, dataset)][key]
                        for key in ("metric_error", "time_train_s", "time_infer_s")
                    }
                    for arm in ARMS
                },
            }
            for dataset in roster
        ]
        summary = {
            "schema_version": 1,
            "study": (f"{args.model_scale}-fingerprint-tabarena-v0.1-exploratory"),
            "formal_eligible": False,
            "leaderboard_replication": False,
            "seed": 42,
            "comparison_step": args.comparison_step,
            "model_scale": args.model_scale,
            "task_subset": "lite",
            "task_count": len(roster),
            "result_count": len(normalized),
            "arm_order": list(ARMS),
            "overall_mean_rank_lower_is_better": _mean_ranks(
                rows, roster=roster, arm_order=ARMS
            ),
            "overall_scale_free_comparisons": [
                _scale_free_comparison(rows, roster=roster, left=left, right=right)
                for left, right in pairs
            ],
            "metric_groups": metric_groups,
            "datasets": datasets,
            "checkpoint_digests": {
                arm: {
                    "sha256": observed[arm],
                    "size_bytes": checkpoints[arm].stat().st_size,
                }
                for arm in ARMS
            },
            "checkpoint_contract": contract,
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
        }
        runtime_summary = _runtime_summary(
            runtime=runtime,
            benchmark_sha=tabarena_sha,
            duration_seconds=duration,
            result_count=len(normalized),
        )
        archive_path = staging / "results.tar.gz"
        _archive_directory(results_root, archive_path)
        shutil.rmtree(results_root)
        shutil.rmtree(scratch, ignore_errors=True)
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / "runtime.json").write_text(
            json.dumps(runtime_summary, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
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
    print(
        json.dumps(
            {"output_dir": str(output), "task_count": len(roster)}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
