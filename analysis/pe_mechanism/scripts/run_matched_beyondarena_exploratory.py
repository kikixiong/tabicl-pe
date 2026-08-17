#!/usr/bin/env python3
"""Run the frozen step-250k No-RoPE/RoPE pair on a bounded BeyondArena slice.

The run is exploratory only.  Dataset names are passed explicitly to
``build_jobs`` before any Data Foundry materialization, so this smoke cannot
silently expand to all 142 BeyondArena datasets.
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


ARMS_BASE = ("rope", "none")
DEFAULT_DATASETS = (
    "blood_transfusion",
    "parkinsons_biomedical_voice_measurements",
    "ghanas_indigenous_intel",
)
EXPECTED_COMPARISON_STEP = 250_000
EXPECTED_SEED = 42
EXPECTED_MODEL_SHA = "2d44e7540ffd4ec6216aa0058970b1de1034aa1d"
EXPECTED_MANIFEST_SHA256 = (
    "14a8c5069f1b6609ac5020caf9c544d98fb276b910028203912379e47b483447"
)
EXPECTED_RELEASED_SHA256 = (
    "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
)
ROPE_ONLY_STATE_KEYS = frozenset({"row_interactor.tf_row.rope.freqs"})


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
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(root: Path, *, expected: str, label: str) -> str:
    import subprocess

    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise ValueError(f"{label} source checkout is not clean: {root}")
    observed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != expected:
        raise ValueError(f"{label} source SHA mismatch: {observed} != {expected}")
    return observed


def _load_matched_pair(
    manifest_path: Path,
) -> tuple[dict[str, Path], dict[str, str], dict[str, Any]]:
    """Verify and load the one frozen watcher-published pilot pair."""
    import torch

    manifest_digest = _sha256(manifest_path)
    if manifest_digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError("matched checkpoint manifest digest is not the frozen pair")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("matched checkpoint manifest is not valid JSON") from error
    if not isinstance(manifest, Mapping) or (
        manifest.get("schema_version") != 1
        or manifest.get("kind")
        != "exploratory_same_step_pilot_checkpoint_pair"
        or manifest.get("formal_eligible") is not False
        or manifest.get("comparison_step") != EXPECTED_COMPARISON_STEP
        or manifest.get("seed") != EXPECTED_SEED
        or manifest.get("pilot_source_commit") != EXPECTED_MODEL_SHA
    ):
        raise ValueError("matched checkpoint manifest violates the frozen pilot contract")
    arms = manifest.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {"rope", "none"}:
        raise ValueError("matched checkpoint manifest must contain RoPE and No-RoPE")

    checksum_path = manifest_path.parent / "SHA256SUMS"
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise FileNotFoundError("matched checkpoint checksum marker is missing")
    checkpoints: dict[str, Path] = {}
    digests: dict[str, str] = {}
    configs: dict[str, dict[str, Any]] = {}
    signatures: dict[str, dict[str, tuple[tuple[int, ...], str]]] = {}
    expected_checksum_lines = {
        f"{manifest_digest}  {manifest_path.name}",
    }
    for arm in ARMS_BASE:
        entry = arms[arm]
        if not isinstance(entry, Mapping):
            raise ValueError(f"invalid matched checkpoint entry: {arm}")
        filename = entry.get("snapshot_checkpoint")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"invalid matched checkpoint filename: {arm}")
        checkpoint = manifest_path.parent / filename
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        digest = _sha256(checkpoint)
        if (
            entry.get("sha256") != digest
            or entry.get("bytes") != checkpoint.stat().st_size
            or entry.get("curr_step") != EXPECTED_COMPARISON_STEP
            or entry.get("row_identity_mode") != arm
        ):
            raise ValueError(f"matched checkpoint bytes disagree with manifest: {arm}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = payload.get("config") if isinstance(payload, Mapping) else None
        state = payload.get("state_dict") if isinstance(payload, Mapping) else None
        if (
            not isinstance(config, dict)
            or not isinstance(state, dict)
            or not state
            or payload.get("curr_step") != EXPECTED_COMPARISON_STEP
            or config.get("row_identity_mode") != arm
        ):
            raise ValueError(f"matched checkpoint payload violates contract: {arm}")
        signature: dict[str, tuple[tuple[int, ...], str]] = {}
        for name, tensor in state.items():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise ValueError(f"matched checkpoint state is not tensor-only: {arm}")
            signature[name] = (tuple(tensor.shape), str(tensor.dtype))
        if entry.get("state_dict_tensor_count") != len(signature):
            raise ValueError(f"matched checkpoint tensor count disagrees: {arm}")
        checkpoints[arm] = checkpoint
        digests[arm] = digest
        configs[arm] = dict(config)
        signatures[arm] = signature
        expected_checksum_lines.add(f"{digest}  {filename}")

    comparable = {arm: dict(configs[arm]) for arm in ARMS_BASE}
    for config in comparable.values():
        config.pop("row_identity_mode")
    if comparable["rope"] != comparable["none"]:
        raise ValueError("matched configs differ beyond row_identity_mode")
    none_state = signatures["none"]
    rope_state = signatures["rope"]
    if set(none_state) - set(rope_state) or set(rope_state) - set(
        none_state
    ) != ROPE_ONLY_STATE_KEYS:
        raise ValueError("matched states differ beyond the derived RoPE frequencies")
    if any(
        none_state[name] != rope_state[name]
        for name in set(none_state) & set(rope_state)
    ):
        raise ValueError("matched state tensor schemas differ")
    embed_dim = configs["rope"].get("embed_dim")
    row_nhead = configs["rope"].get("row_nhead")
    if (
        type(embed_dim) is not int
        or type(row_nhead) is not int
        or embed_dim <= 0
        or row_nhead <= 0
        or embed_dim % row_nhead != 0
        or (embed_dim // row_nhead) % 2
    ):
        raise ValueError("matched config cannot derive the RoPE frequency shape")
    expected_freq = ((embed_dim // row_nhead // 2,), "torch.float32")
    if rope_state["row_interactor.tf_row.rope.freqs"] != expected_freq:
        raise ValueError("RoPE frequency tensor disagrees with model architecture")

    observed_checksum_lines = set(
        checksum_path.read_text(encoding="ascii").splitlines()
    )
    if observed_checksum_lines != expected_checksum_lines:
        raise ValueError("matched checkpoint checksum marker disagrees with payload")
    contract = {
        "manifest_sha256": manifest_digest,
        "checksums_sha256": _sha256(checksum_path),
        "comparison_step": EXPECTED_COMPARISON_STEP,
        "seed": EXPECTED_SEED,
        "pilot_source_commit": EXPECTED_MODEL_SHA,
        "arms": {
            arm: {
                "sha256": digests[arm],
                "size_bytes": checkpoints[arm].stat().st_size,
                "state_dict_tensor_count": len(signatures[arm]),
                "row_identity_mode": configs[arm]["row_identity_mode"],
            }
            for arm in ARMS_BASE
        },
    }
    return checkpoints, digests, contract


def _requested_datasets(values: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(values) if values else DEFAULT_DATASETS
    if any(not name or name.startswith("-") for name in requested):
        raise ValueError("dataset names must be non-empty and must not begin with '-'")
    if len(requested) != len(set(requested)):
        raise ValueError("dataset names must be unique")
    return requested


def _mean_ranks_by_regime(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    roster: Sequence[str],
    expected: Mapping[str, Mapping[str, Any]],
    arm_order: Sequence[str],
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
            arm_order=arm_order,
        )
        for regime in regimes
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--tabarena-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--snapshot-manifest", required=True)
    parser.add_argument("--released-checkpoint")
    parser.add_argument("--expected-analysis-sha", required=True)
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
    snapshot_manifest = _absolute(args.snapshot_manifest, name="snapshot_manifest")
    output = Path(args.output_dir)
    if not output.is_absolute() or output.exists():
        raise ValueError("output_dir must be an absent absolute path")
    if args.expected_model_sha != EXPECTED_MODEL_SHA:
        raise ValueError("model SHA is not the frozen pilot source commit")

    checkpoints, checkpoint_digests, checkpoint_contract = _load_matched_pair(
        snapshot_manifest
    )
    arms = list(ARMS_BASE)
    if args.released_checkpoint is not None:
        released = _absolute(args.released_checkpoint, name="released_checkpoint")
        released_digest = _sha256(released)
        if released_digest != EXPECTED_RELEASED_SHA256:
            raise ValueError("released checkpoint digest is not the frozen reference")
        checkpoints["released"] = released
        checkpoint_digests["released"] = released_digest
        arms.append("released")
    arm_order = tuple(arms)

    analysis_sha = _git_head(
        analysis_root, expected=args.expected_analysis_sha, label="analysis"
    )
    model_sha = _git_head(model_root, expected=EXPECTED_MODEL_SHA, label="model")
    tabarena_sha = _git_head(
        tabarena_root, expected=args.expected_tabarena_sha, label="TabArena"
    )

    scripts_root = analysis_root / "analysis/pe_mechanism/scripts"
    package_root = analysis_root / "analysis/pe_mechanism/src"
    sys.path.insert(0, str(scripts_root))
    sys.path.insert(0, str(package_root))
    from run_fingerprint_beyondarena_exploratory import (
        _make_beyond_system_model,
        _select_dataset_metadata,
    )
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
    requested = _requested_datasets(args.dataset_name)
    arena = runtime["BeyondArenaContext"](
        methods=[],
        backend="native",
        cache_config=runtime["CacheConfig"].from_root(cache_root),
    )
    roster, expected_tasks = _select_dataset_metadata(arena.task_metadata, requested)

    model_cls = _make_beyond_system_model(
        runtime["ExternalSystemModel"], _make_system_model
    )
    framework_to_arm: dict[str, str] = {}
    experiments = []
    display = {
        "rope": "TabICL_RoPE_Step250000",
        "none": "TabICL_NoRoPE_Step250000",
        "released": "TabICL_Released_Reference",
    }
    for arm in arm_order:
        generator = runtime["SystemConfigGenerator"](
            model_cls=model_cls,
            name=display[arm],
            manual_configs=[
                {
                    "checkpoint": str(checkpoints[arm]),
                    "arm": arm,
                    "device": "cuda",
                    "n_estimators": 1,
                    "seed": EXPECTED_SEED,
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
    expected_count = len(arm_order) * len(roster)
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
        raise RuntimeError("BeyondArena job matrix differs from the explicit lite grid")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("output_dir appeared during preflight")
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
        normalize_kwargs = {
            "framework_to_arm": framework_to_arm,
            "roster": roster,
            "expected_count": expected_count,
            "expected_tasks": expected_tasks,
            "arm_order": arm_order,
        }
        normalized = _normalize_results(results, **normalize_kwargs)
        cached = _normalize_results(
            _load_cached_results(results_root), **normalize_kwargs
        )
        if normalized != cached:
            raise RuntimeError("returned and cached BeyondArena results differ")
        rows = {(row["arm"], row["dataset"]): row for row in normalized}
        pairs = tuple(
            (arm_order[left], arm_order[right])
            for left in range(len(arm_order))
            for right in range(left + 1, len(arm_order))
        )
        summary = {
            "schema_version": 1,
            "study": "matched-step250k-rope-none-beyondarena-exploratory",
            "formal_eligible": False,
            "leaderboard_replication": False,
            "comparison_step": EXPECTED_COMPARISON_STEP,
            "seed": EXPECTED_SEED,
            "task_subset": "lite-explicit-roster",
            "task_count": len(roster),
            "result_count": len(normalized),
            "arm_order": list(arm_order),
            "overall_mean_rank_lower_is_better": _mean_ranks(
                rows, roster=roster, arm_order=arm_order
            ),
            "mean_rank_by_split_regime": _mean_ranks_by_regime(
                rows,
                roster=roster,
                expected=expected_tasks,
                arm_order=arm_order,
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
                            for key in (
                                "metric_error",
                                "time_train_s",
                                "time_infer_s",
                            )
                        }
                        for arm in arm_order
                    },
                }
                for dataset in roster
            ],
            "checkpoint_provenance": {
                "matched_pair": checkpoint_contract,
                "match_scope": (
                    "The watcher attests the same step, seed label, source commit, "
                    "architecture/configuration, and state schema. These legacy "
                    "checkpoints do not contain prior/DataLoader/RNG stream state, "
                    "so an identical per-batch training trajectory is not attested."
                ),
                "arms": {
                    arm: {
                        "sha256": checkpoint_digests[arm],
                        "size_bytes": checkpoints[arm].stat().st_size,
                    }
                    for arm in arm_order
                },
            },
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
                "Legacy matched pilot checkpoints on a three-dataset lite smoke. "
                "This is neither the BeyondArena core protocol nor formal evidence."
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
    print(
        json.dumps(
            {"output_dir": str(output), "task_count": len(roster)}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
