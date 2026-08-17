#!/usr/bin/env python3
"""Verify and aggregate complete paired-checkpoint TALENT shard runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from pe_mechanism.talent_full_suite import (
    ARM_DATASET_ARTIFACTS,
    ARM_DATASET_KIND,
    AGGREGATE_KIND,
    DATASET_ARTIFACTS,
    DATASET_KIND,
    DISK_OFFLOAD_MIN_FREE_BYTES,
    DISK_OFFLOAD_SCRATCH_CONTRACT,
    SHARD_KIND,
    TALENT_ESTIMATOR_OPTIONS,
    OOM_FALLBACK_SEQUENCE,
    RELEASE_KIND,
    SUBMISSION_KIND,
    absolute_path,
    array_sha256,
    atomic_json,
    canary_dataset_records,
    directory_manifest,
    exact_sign_test,
    frozen_discovery_roster,
    fsync_directory,
    json_document_sha256,
    load_json_object,
    load_json_object_with_sha256,
    load_private_run_config,
    require_disjoint_output,
    sha256_file,
    validate_checkpoint_pairs,
    validate_directory_manifest,
    validate_observed_treatment,
    validate_self_hashed_document,
    validate_shard_plan,
    verify_clean_detached_git,
)
from pe_mechanism.statistics import paired_bootstrap_ci


BOOTSTRAP_RESAMPLES = 10_000
SIGN_FLIP_RESAMPLES = 10_000
COLUMN_COUNT_TRAINING_MAXIMUM = 100


class _RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.descriptor: int | None = None

    def __enter__(self) -> "_RunLock":
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self.descriptor = os.open(self.path, flags, 0o600)
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(self.descriptor)
            self.descriptor = None
            raise RuntimeError("another aggregator owns this output lock") from error
        return self

    def __exit__(self, *_: object) -> None:
        assert self.descriptor is not None
        fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        os.close(self.descriptor)
        self.descriptor = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--expected-analysis-sha", required=True)
    parser.add_argument("--shard-plan", required=True)
    parser.add_argument("--run-config", action="append", required=True)
    parser.add_argument("--canary-output", action="append", required=True)
    parser.add_argument("--shard-output", action="append", required=True)
    parser.add_argument("--submission-receipt", action="append", required=True)
    parser.add_argument("--release-receipt", action="append", required=True)
    parser.add_argument("--gpu-csv", action="append", required=True)
    parser.add_argument("--expected-run-config-sha256", required=False)
    parser.add_argument("--expected-shard-plan-sha256", required=False)
    parser.add_argument("--output-dir", required=True)
    return parser


def _load_configs(paths: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    by_digest: dict[str, Any] = {}
    pair_ids: set[str] = set()
    arm_ids: set[str] = set()
    for value in paths:
        config = load_private_run_config(Path(value))
        if len(config["pairs"]) != 1:
            raise ValueError("each aggregate run config must contain exactly one pair")
        if config["config_sha256"] in by_digest:
            raise ValueError("aggregate run configs must be content-distinct")
        pair = config["pairs"][0]
        if pair["pair_id"] in pair_ids:
            raise ValueError("pair IDs must be unique across aggregate run configs")
        pair_ids.add(pair["pair_id"])
        for arm in pair["arms"]:
            if arm["arm_id"] in arm_ids:
                raise ValueError("arm IDs must be unique across aggregate run configs")
            arm_ids.add(arm["arm_id"])
        config["checkpoint_contract"] = validate_checkpoint_pairs(config)
        configs.append(config)
        by_digest[config["config_sha256"]] = config
    return configs, by_digest


def _is_hex(value: object, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _validate_environment(document: Mapping[str, Any]) -> None:
    if set(document) != {
        "schema_version",
        "python",
        "python_executable_sha256",
        "packages",
        "numpy",
        "pandas",
        "sklearn",
        "torch",
        "torch_cuda_build",
        "cudnn",
        "nvidia_driver",
        "gpu",
        "imports",
    }:
        raise ValueError("environment contract keys are invalid")
    packages = document.get("packages")
    gpu = document.get("gpu")
    imports = document.get("imports")
    if (
        document.get("schema_version") != 1
        or not isinstance(document.get("python"), str)
        or not _is_hex(document.get("python_executable_sha256"), 64)
        or not isinstance(packages, Mapping)
        or set(packages) != {"numpy", "pandas", "scikit-learn", "torch"}
        or any(not isinstance(value, str) or not value for value in packages.values())
        or not isinstance(document.get("nvidia_driver"), str)
        or not document["nvidia_driver"]
        or not isinstance(gpu, Mapping)
        or set(gpu) != {
            "name",
            "total_memory_bytes",
            "compute_capability",
        }
        or "H100" not in str(gpu.get("name"))
        or not isinstance(gpu.get("total_memory_bytes"), int)
        or isinstance(gpu.get("total_memory_bytes"), bool)
        or gpu["total_memory_bytes"] <= 0
        or gpu.get("compute_capability") != [9, 0]
        or not isinstance(imports, Mapping)
        or set(imports) != {"pe_mechanism", "tabicl"}
    ):
        raise ValueError("environment contract is invalid")
    for name, source_tree in (("pe_mechanism", "analysis"), ("tabicl", "model")):
        record = imports[name]
        if (
            not isinstance(record, Mapping)
            or set(record) != {"source_tree", "relative_path"}
            or record.get("source_tree") != source_tree
            or not isinstance(record.get("relative_path"), str)
            or not record["relative_path"]
            or Path(record["relative_path"]).is_absolute()
            or ".." in Path(record["relative_path"]).parts
        ):
            raise ValueError("environment import provenance is invalid")


def _validate_campaign_receipts(
    *,
    submission_path: Path,
    release_path: Path,
    config: Mapping[str, Any],
    checkpoint_contract: Mapping[str, Any],
    analysis_sha: str,
    plan_file_sha256: str,
    plan_document_sha256: str,
    canary_ordinals: Sequence[int],
) -> dict[str, Any]:
    submission_document, submission_file_sha256 = load_json_object_with_sha256(
        submission_path, name="TALENT submission receipt"
    )
    submission = validate_self_hashed_document(
        submission_document, kind=SUBMISSION_KIND, name="TALENT submission receipt"
    )
    expected_submission_keys = {
        "schema_version",
        "formal_eligible",
        "study",
        "pair_contract",
        "pair_contract_sha256",
        "checkpoint_contract",
        "analysis_sha",
        "model_runtime_sha",
        "shard_plan_file_sha256",
        "shard_plan_document_sha256",
        "canary_dataset_ordinals",
        "shard_count",
        "capacity",
        "job_roles",
        "command_plan_sha256_by_role",
        "operation_journal_sha256_at_submission",
        "created_at_unix_seconds",
    }
    capacity = submission.get("capacity")
    jobs = submission.get("job_roles")
    command_hashes = submission.get("command_plan_sha256_by_role")
    if (
        set(submission) != expected_submission_keys
        or submission.get("schema_version") != 1
        or submission.get("formal_eligible") is not False
        or submission.get("study") != "tabicl-talent-paired-checkpoints-exploratory-v1"
        or submission.get("pair_contract") != config["portable"]
        or submission.get("pair_contract_sha256") != config["portable_sha256"]
        or submission.get("checkpoint_contract") != checkpoint_contract
        or submission.get("analysis_sha") != analysis_sha
        or submission.get("model_runtime_sha")
        != config["pairs"][0]["training_source_commit"]
        or submission.get("shard_plan_file_sha256") != plan_file_sha256
        or submission.get("shard_plan_document_sha256") != plan_document_sha256
        or submission.get("canary_dataset_ordinals") != list(canary_ordinals)
        or submission.get("shard_count") != 8
        or not isinstance(capacity, Mapping)
        or set(capacity)
        != {
            "output_pre_submit_available_bytes",
            "capacity_root_pre_submit_available_bytes",
            "scratch_root_pre_submit_available_bytes",
            "warning_threshold_bytes",
            "hard_reserve_bytes",
            "warning_triggered",
            "pre_submit_status_by_filesystem",
        }
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (
                capacity.get("output_pre_submit_available_bytes"),
                capacity.get("capacity_root_pre_submit_available_bytes"),
                capacity.get("scratch_root_pre_submit_available_bytes"),
                capacity.get("warning_threshold_bytes"),
                capacity.get("hard_reserve_bytes"),
            )
        )
        or capacity.get("warning_threshold_bytes") != 22 * 1024**3
        or capacity.get("hard_reserve_bytes") != 20 * 1024**3
        or min(
            capacity.get("output_pre_submit_available_bytes", 0),
            capacity.get("capacity_root_pre_submit_available_bytes", 0),
            capacity.get("scratch_root_pre_submit_available_bytes", 0),
        )
        < 20 * 1024**3
        or capacity.get("warning_triggered")
        is not any(
            value < 22 * 1024**3
            for value in (
                capacity.get("output_pre_submit_available_bytes", 0),
                capacity.get("capacity_root_pre_submit_available_bytes", 0),
                capacity.get("scratch_root_pre_submit_available_bytes", 0),
            )
        )
        or capacity.get("pre_submit_status_by_filesystem")
        != {
            name: (
                "warning_below_22_gib"
                if value < 22 * 1024**3
                else "ok_at_or_above_22_gib"
            )
            for name, value in {
                "output": capacity.get("output_pre_submit_available_bytes", 0),
                "capacity": capacity.get(
                    "capacity_root_pre_submit_available_bytes", 0
                ),
                "scratch": capacity.get(
                    "scratch_root_pre_submit_available_bytes", 0
                ),
            }.items()
        }
        or jobs
        != {
            "canary": {"submission_state": "held"},
            "full_array": {"dependency": "afterok:canary"},
            "aggregate": {"dependency": "afterok:full_array"},
        }
        or not isinstance(command_hashes, Mapping)
        or set(command_hashes) != {"canary", "array", "aggregate"}
        or any(not _is_hex(value, 64) for value in command_hashes.values())
        or not _is_hex(
            submission.get("operation_journal_sha256_at_submission"), 64
        )
        or not isinstance(submission.get("created_at_unix_seconds"), (int, float))
        or isinstance(submission.get("created_at_unix_seconds"), bool)
    ):
        raise ValueError("TALENT submission receipt contract is invalid")

    release_document, release_file_sha256 = load_json_object_with_sha256(
        release_path, name="TALENT release receipt"
    )
    release = validate_self_hashed_document(
        release_document, kind=RELEASE_KIND, name="TALENT release receipt"
    )
    if (
        set(release)
        != {
            "schema_version",
            "submission_document_sha256",
            "submission_file_sha256",
            "released_job_role",
            "operation_journal_sha256_after_release",
            "released_at_unix_seconds",
        }
        or release.get("schema_version") != 1
        or release.get("submission_document_sha256")
        != submission_document["sha256"]
        or release.get("submission_file_sha256") != submission_file_sha256
        or release.get("released_job_role") != "canary"
        or not _is_hex(
            release.get("operation_journal_sha256_after_release"), 64
        )
        or not isinstance(release.get("released_at_unix_seconds"), (int, float))
        or isinstance(release.get("released_at_unix_seconds"), bool)
        or release["released_at_unix_seconds"]
        < submission["created_at_unix_seconds"]
    ):
        raise ValueError("TALENT release receipt contract is invalid")
    return {
        "submission_receipt_file_sha256": submission_file_sha256,
        "submission_receipt_document_sha256": submission_document["sha256"],
        "release_receipt_file_sha256": release_file_sha256,
        "release_receipt_document_sha256": release_document["sha256"],
    }


def _validate_gpu_csv(
    path: Path,
    *,
    expected_jobs: Mapping[tuple[str, str], Mapping[str, Any]],
    analysis_sha: str,
) -> tuple[tuple[str, str], dict[str, Any]]:
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 10_000_000:
        raise ValueError("GPU monitor CSV is unsafe or outside size bounds")
    digest_before = sha256_file(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_fields = [
            "unix_seconds",
            "run_config_sha256",
            "analysis_sha",
            "model_sha",
            "shard_id",
            "execution_scope",
            "index",
            "name",
            "utilization_gpu_percent",
            "memory_used_mib",
            "memory_total_mib",
            "power_draw_watts",
        ]
        if reader.fieldnames != expected_fields:
            raise ValueError("GPU monitor CSV header is invalid")
        rows = list(reader)
    if sha256_file(path) != digest_before or len(rows) < 2:
        raise ValueError("GPU monitor CSV changed or has too few samples")
    first = rows[0]
    key = (first["run_config_sha256"], first["shard_id"])
    expected = expected_jobs.get(key)
    if expected is None:
        raise ValueError("GPU monitor does not belong to an expected job")
    timestamps: list[int] = []
    utilization: list[float] = []
    memory_total: float | None = None
    for row in rows:
        try:
            timestamp = int(row["unix_seconds"])
            gpu_index = int(row["index"])
            current_utilization = float(row["utilization_gpu_percent"])
            memory_used = float(row["memory_used_mib"])
            current_memory_total = float(row["memory_total_mib"])
            power = float(row["power_draw_watts"])
        except (TypeError, ValueError) as error:
            raise ValueError("GPU monitor contains non-numeric metrics") from error
        observed_gpu_name = row["name"].strip()
        if (
            row["run_config_sha256"] != key[0]
            or row["analysis_sha"] != analysis_sha
            or row["model_sha"] != expected["model_runtime_sha"]
            or row["shard_id"] != key[1]
            or row["execution_scope"] != expected["execution_scope"]
            or gpu_index != 0
            or observed_gpu_name != expected["gpu_name"]
            or "H100" not in observed_gpu_name
            or not all(
                math.isfinite(value)
                for value in (
                    current_utilization,
                    memory_used,
                    current_memory_total,
                    power,
                )
            )
            or not 0.0 <= current_utilization <= 100.0
            or not 0.0 <= memory_used <= current_memory_total
            or current_memory_total <= 0.0
            or power < 0.0
            or (memory_total is not None and current_memory_total != memory_total)
        ):
            raise ValueError("GPU monitor row contract is invalid")
        memory_total = current_memory_total
        timestamps.append(timestamp)
        utilization.append(current_utilization)
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("GPU monitor timestamps are not strictly increasing")
    window = expected["monitor_window"]
    if (
        timestamps[0] > window["started_at_unix_seconds"] + 30
        or timestamps[-1] < window["completed_at_unix_seconds"] - 35
    ):
        raise ValueError("GPU monitor does not cover the evaluator attempt")
    active = [value for value in utilization if value > 0.0]
    active_mean = float(np.mean(active)) if active else 0.0
    return key, {
        "csv_sha256": digest_before,
        "sample_count": len(rows),
        "gpu_name": first["name"].strip(),
        "active_sample_count": len(active),
        "active_window_mean_percent": active_mean,
        "below_80_percent_observation": active_mean < 80.0,
        "maximum_memory_used_mib": float(
            max(float(row["memory_used_mib"]) for row in rows)
        ),
    }


def _require_consistent_job_group(
    pair_id: str, jobs: Sequence[Mapping[str, Any]]
) -> None:
    if not jobs:
        raise ValueError(f"pair {pair_id} has no job evidence")
    for field in (
        "common_run_contract_sha256",
        "environment_contract_sha256",
        "environment_contract",
    ):
        reference = jobs[0][field]
        if any(job[field] != reference for job in jobs[1:]):
            raise ValueError(f"pair {pair_id} jobs differ at {field}")


def _assert_public_safe(value: Any, *, location: str = "aggregate") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} contains a non-string key")
            lowered = key.lower()
            if lowered in {"path", "job_id", "slurm_job_id", "output_dir"} or (
                lowered.endswith("_path") and lowered != "relative_path"
            ):
                raise ValueError(f"{location} contains private locator field {key}")
            _assert_public_safe(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_public_safe(item, location=f"{location}[{index}]")
    elif isinstance(value, str) and Path(value).is_absolute():
        raise ValueError(f"{location} contains an absolute path")


def _assert_equal(actual: object, expected: object, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} differs from the aggregate contract")


def _load_arm_prediction(
    arm_root: Path,
    *,
    result: Mapping[str, Any],
    arm: Mapping[str, Any],
    training_source_commit: str,
    selected_offload_mode: str,
) -> dict[str, float]:
    try:
        with np.load(arm_root / "predictions.npz", allow_pickle=False) as archive:
            if set(archive.files) != {"target", "probabilities"}:
                raise ValueError("prediction archive keys are invalid")
            target = np.asarray(archive["target"])
            probability = np.asarray(archive["probabilities"])
    except (OSError, ValueError) as error:
        raise ValueError("prediction archive is invalid") from error
    if (
        target.dtype != np.dtype("int64")
        or target.ndim != 1
        or target.shape[0] != result.get("n_evaluation")
        or array_sha256(target) != result.get("target_sha256")
    ):
        raise ValueError("stored target array is invalid")
    n_classes = result.get("n_classes")
    if (
        not isinstance(n_classes, int)
        or isinstance(n_classes, bool)
        or n_classes < 2
        or np.any(target < 0)
        or np.any(target >= n_classes)
    ):
        raise ValueError("stored target encoding is invalid")
    if (
        probability.dtype != np.dtype("float32")
        or probability.shape != (target.size, n_classes)
        or not np.isfinite(probability).all()
        or np.any(probability < 0.0)
        or not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=2e-5)
        or array_sha256(probability) != result.get("probabilities_sha256")
        or result.get("checkpoint_sha256") != arm["checkpoint_sha256"]
        or result.get("model_runtime_sha") != training_source_commit
        or result.get("offload_mode") != selected_offload_mode
    ):
        raise ValueError(f"stored probability contract is invalid for {arm['arm_id']}")
    validate_observed_treatment(
        result.get("treatment"), arm["treatment"], name=f"stored {arm['arm_id']}"
    )
    selected = probability[np.arange(target.size), target]
    accuracy = float(np.mean(np.argmax(probability, axis=1) == target))
    log_loss = float(
        -np.log(np.clip(selected, np.finfo(np.float32).tiny, 1.0)).mean()
    )
    for metric, expected in (("accuracy", accuracy), ("log_loss", log_loss)):
        observed = result.get(metric)
        if (
            not isinstance(observed, (int, float))
            or isinstance(observed, bool)
            or not math.isfinite(float(observed))
            or not math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(f"stored {metric} is invalid for {arm['arm_id']}")
    return {"accuracy": accuracy, "log_loss": log_loss}


def _validate_dataset(
    dataset_root: Path,
    *,
    expected: Mapping[str, Any],
    summary_result: Mapping[str, Any],
    config: Mapping[str, Any],
    run_contract_sha256: str,
) -> dict[str, Any]:
    if dataset_root.is_symlink() or not dataset_root.is_dir():
        raise ValueError("dataset output directory is unsafe")
    if {path.name for path in dataset_root.iterdir()} != DATASET_ARTIFACTS:
        raise ValueError("dataset output artifact roster is invalid")
    validate_directory_manifest(dataset_root, kind=DATASET_KIND)
    task = load_json_object(dataset_root / "task.json", name="dataset task")
    if dict(task) != dict(summary_result):
        raise ValueError("shard summary does not exactly bind the dataset task")
    if set(task) != {
        "schema_version",
        "run_contract_sha256",
        "ordinal",
        "dataset",
        "fit_split",
        "evaluation_split",
        "n_train",
        "n_evaluation",
        "n_features",
        "n_classes",
        "row_sampling",
        "row_roster_sha256",
        "class_tokens",
        "info_sha256",
        "input_sha256",
        "target_sha256",
        "oom_fallback",
        "arms",
    }:
        raise ValueError("dataset task keys are invalid")
    for key, value in (
        ("run_contract_sha256", run_contract_sha256),
        ("ordinal", expected["ordinal"]),
        ("dataset", expected["name"]),
        ("n_train", expected["n_train"]),
        ("n_evaluation", expected["n_validation"]),
        ("n_features", expected["n_features"]),
        ("n_classes", expected["n_classes"]),
        ("info_sha256", expected["info_sha256"]),
        ("input_sha256", expected["input_sha256"]),
        ("fit_split", "train"),
        ("evaluation_split", "val"),
        ("row_sampling", "none_full_split_original_order"),
    ):
        _assert_equal(task.get(key), value, name=f"dataset {expected['name']} {key}")
    class_tokens = task.get("class_tokens")
    row_roster = task.get("row_roster_sha256")
    if (
        not isinstance(class_tokens, list)
        or len(class_tokens) != expected["n_classes"]
        or len(set(class_tokens)) != len(class_tokens)
        or not isinstance(row_roster, Mapping)
        or set(row_roster) != {"train", "val"}
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in row_roster.values()
        )
    ):
        raise ValueError("dataset class/row roster contract is invalid")
    fallback = task.get("oom_fallback")
    if (
        not isinstance(fallback, Mapping)
        or set(fallback)
        != {
            "sequence",
            "attempted_levels",
            "selected_level",
            "retry_scope",
            "non_oom_retry",
            "disk_offload_scratch",
        }
        or fallback.get("sequence") != list(OOM_FALLBACK_SEQUENCE)
        or fallback.get("selected_level") not in OOM_FALLBACK_SEQUENCE
        or fallback.get("attempted_levels")
        != list(
            OOM_FALLBACK_SEQUENCE[
                : OOM_FALLBACK_SEQUENCE.index(fallback["selected_level"]) + 1
            ]
        )
        or fallback.get("retry_scope")
        != "discard_entire_dataset_pair_and_restart_both_arms"
        or fallback.get("non_oom_retry") is not False
    ):
        raise ValueError("dataset OOM fallback contract is invalid")
    scratch = fallback.get("disk_offload_scratch")
    attempted_levels = fallback["attempted_levels"]
    if (
        not isinstance(scratch, Mapping)
        or set(scratch)
        != set(DISK_OFFLOAD_SCRATCH_CONTRACT)
        | {"available_bytes_before_attempt"}
        or any(
            scratch.get(key) != value
            for key, value in DISK_OFFLOAD_SCRATCH_CONTRACT.items()
        )
        or not isinstance(scratch.get("available_bytes_before_attempt"), Mapping)
        or set(scratch["available_bytes_before_attempt"]) != set(attempted_levels)
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < DISK_OFFLOAD_MIN_FREE_BYTES
            for value in scratch["available_bytes_before_attempt"].values()
        )
    ):
        raise ValueError("dataset disk-offload scratch contract is invalid")
    task_arms = task.get("arms")
    if not isinstance(task_arms, Mapping) or set(task_arms) != set(
        config["arm_order"]
    ):
        raise ValueError("dataset task arm roster is invalid")
    arms_root = dataset_root / "arms"
    if arms_root.is_symlink() or not arms_root.is_dir() or {
        path.name for path in arms_root.iterdir()
    } != set(config["arm_order"]):
        raise ValueError("dataset arm directory union is invalid")
    arm_lookup = {
        arm["arm_id"]: arm for pair in config["pairs"] for arm in pair["arms"]
    }
    metrics: dict[str, dict[str, float]] = {}
    schedules = []
    for arm_id in config["arm_order"]:
        arm_root = arms_root / arm_id
        if arm_root.is_symlink() or not arm_root.is_dir() or {
            path.name for path in arm_root.iterdir()
        } != ARM_DATASET_ARTIFACTS:
            raise ValueError("dataset arm artifact roster is invalid")
        validate_directory_manifest(arm_root, kind=ARM_DATASET_KIND)
        if sha256_file(arm_root / "manifest.json") != task_arms[arm_id].get(
            "arm_manifest_sha256"
        ):
            raise ValueError("dataset arm manifest binding is invalid")
        arm_result = load_json_object(arm_root / "result.json", name="arm result")
        common = {key: value for key, value in task.items() if key not in {"arms", "oom_fallback"}}
        expected_arm_result = {
            **common,
            "arm_id": arm_id,
            **{
                key: value
                for key, value in task_arms[arm_id].items()
                if key != "arm_manifest_sha256"
            },
        }
        if dict(arm_result) != expected_arm_result:
            raise ValueError("arm result differs from dataset task")
        prediction_contract = arm_result.get("prediction_contract")
        if (
            not isinstance(prediction_contract, Mapping)
            or set(prediction_contract)
            != {
                "chunk_count",
                "maximum_rows_per_call",
                "source_evidence_level",
                "all_chunks_exact_baseline_verified",
            }
            or prediction_contract.get("source_evidence_level") != "strict"
            or prediction_contract.get("all_chunks_exact_baseline_verified") is not True
            or prediction_contract.get("maximum_rows_per_call") != 65_536
            or prediction_contract.get("chunk_count")
            != math.ceil(expected["n_validation"] / 65_536)
            or any(
                not isinstance(arm_result.get(key), (int, float))
                or isinstance(arm_result.get(key), bool)
                or not math.isfinite(float(arm_result[key]))
                or float(arm_result[key]) < 0.0
                for key in ("fit_seconds", "predict_seconds")
            )
        ):
            raise ValueError("arm prediction/timing contract is invalid")
        metrics[arm_id] = _load_arm_prediction(
            arm_root,
            result=arm_result,
            arm=arm_lookup[arm_id],
            training_source_commit=config["pairs"][0]["training_source_commit"],
            selected_offload_mode=fallback["selected_level"],
        )
        schedules.append(arm_result.get("official_forward_schedule"))
    if not schedules or any(schedule != schedules[0] for schedule in schedules[1:]):
        raise ValueError("paired arms used different stored forward schedules")
    return {
        "ordinal": expected["ordinal"],
        "dataset": expected["name"],
        "info_sha256": expected["info_sha256"],
        "input_sha256": expected["input_sha256"],
        "row_roster_sha256": dict(row_roster),
        "class_tokens": list(class_tokens),
        "target_sha256": task["target_sha256"],
        "n_train": expected["n_train"],
        "n_evaluation": expected["n_validation"],
        "n_features": expected["n_features"],
        "n_classes": expected["n_classes"],
        "arms": metrics,
        "oom_fallback": dict(fallback),
        "arm_manifest_sha256": {
            arm_id: task_arms[arm_id]["arm_manifest_sha256"]
            for arm_id in config["arm_order"]
        },
        "dataset_manifest_sha256": sha256_file(dataset_root / "manifest.json"),
    }


def _validate_shard(
    root: Path,
    *,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_path: Path,
    plan_file_sha256: str,
    plan_document: Mapping[str, Any],
    analysis_sha: str,
    canary: bool = False,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    validate_directory_manifest(root, kind=SHARD_KIND)
    summary = load_json_object(root / "summary.json", name="shard summary")
    contract = load_json_object(root / "run-contract.json", name="shard contract")
    environment = load_json_object(
        root / "environment-contract.json", name="environment contract"
    )
    _validate_environment(environment)
    checkpoint_contract = load_json_object(
        root / "checkpoint-contract.json", name="checkpoint contract"
    )
    expected_common = dict(
        (
        ("schema_version", 1),
        ("kind", SHARD_KIND),
        ("formal_eligible", False),
        ("evidence_scope", "exploratory_discovery_only"),
        ("analysis_sha", analysis_sha),
        (
            "model_runtime_sha",
            config["pairs"][0]["training_source_commit"],
        ),
        ("private_run_config_sha256", config["config_sha256"]),
        ("portable_pair_contract", config["portable"]),
        ("portable_pair_contract_sha256", config["portable_sha256"]),
        ("checkpoint_contract", config["checkpoint_contract"]),
        ("shard_plan_sha256", plan_file_sha256),
        ("shard_plan_document_sha256", plan_document["sha256"]),
        ("roster_sha256", plan["roster_sha256"]),
        ("fit_split", "train"),
        ("evaluation_split", "val"),
        ("test_split_policy", "hashed_for_integrity_only_not_used"),
        ("row_sampling", "none_full_split_original_order"),
        ("prediction_chunk_rows", 65_536),
        ("estimator_options", TALENT_ESTIMATOR_OPTIONS),
        (
            "precision_contract",
            {"model_parameters": "float32", "inference": "float32", "amp": False},
        ),
        (
            "oom_fallback_contract",
            {
                "sequence": list(OOM_FALLBACK_SEQUENCE),
                "batch_size_is_minimum": True,
                "retry_scope": "discard_entire_dataset_pair_and_restart_both_arms",
                "non_oom_retry": False,
            },
        ),
        ("disk_offload_scratch_contract", DISK_OFFLOAD_SCRATCH_CONTRACT),
        ("environment_contract_sha256", json_document_sha256(environment)),
        )
    )
    expected_contract_keys = set(expected_common) | {
        "common_run_contract_sha256",
        "job_contract",
        "job_contract_sha256",
    }
    if set(contract) != expected_contract_keys:
        raise ValueError("shard run contract keys are invalid")
    if set(summary) != expected_contract_keys | {
        "run_contract_sha256",
        "dataset_count",
        "prediction_rows",
        "datasets",
    }:
        raise ValueError("shard summary keys are invalid")
    for key, expected in expected_common.items():
        _assert_equal(contract.get(key), expected, name=f"shard contract {key}")
        _assert_equal(summary.get(key), expected, name=f"shard summary {key}")
    common_contract = {key: contract[key] for key in expected_common}
    if contract.get("common_run_contract_sha256") != json_document_sha256(
        common_contract
    ):
        raise ValueError("shard common run contract digest is invalid")
    if summary.get("common_run_contract_sha256") != contract[
        "common_run_contract_sha256"
    ]:
        raise ValueError("shard summary common contract digest is invalid")
    job_contract = contract.get("job_contract")
    if (
        not isinstance(job_contract, Mapping)
        or set(job_contract)
        != {"shard_id", "execution_scope", "dataset_ordinals", "dataset_names"}
        or contract.get("job_contract_sha256")
        != json_document_sha256(job_contract)
        or summary.get("job_contract") != job_contract
        or summary.get("job_contract_sha256")
        != contract["job_contract_sha256"]
    ):
        raise ValueError("shard job-specific contract is invalid")
    if checkpoint_contract != config["checkpoint_contract"]:
        raise ValueError("checkpoint contract file is invalid")
    run_contract_sha256 = json_document_sha256(contract)
    if summary.get("run_contract_sha256") != run_contract_sha256:
        raise ValueError("shard run contract digest is invalid")
    shard_id = job_contract["shard_id"]
    if canary:
        if (
            shard_id != "canary"
            or job_contract["execution_scope"]
            != "canary_largest_two_plus_madeline"
        ):
            raise ValueError("canary job-specific contract is invalid")
        records = canary_dataset_records(plan)
        ordinals = [record["ordinal"] for record in records]
    else:
        if job_contract["execution_scope"] != "full_planned_shard":
            raise ValueError("full shard execution scope is invalid")
        shard = next(
            (record for record in plan["shards"] if record["shard_id"] == shard_id),
            None,
        )
        if shard is None:
            raise ValueError("shard output ID is absent from the plan")
        ordinals = shard["dataset_ordinals"]
        records = [plan["datasets"][ordinal] for ordinal in ordinals]
    for key, expected in (
        ("dataset_ordinals", ordinals),
        ("dataset_names", [record["name"] for record in records]),
    ):
        _assert_equal(job_contract.get(key), expected, name=f"shard job {key}")
    summary_datasets = summary.get("datasets")
    if (
        not isinstance(summary_datasets, list)
        or len(summary_datasets) != len(records)
        or summary.get("dataset_count") != len(records)
    ):
        raise ValueError("shard dataset summary is invalid")
    results_root = root / "datasets"
    if results_root.is_symlink() or not results_root.is_dir():
        raise ValueError("shard dataset root is unsafe")
    expected_directories = {f"{ordinal:04d}" for ordinal in ordinals}
    if {path.name for path in results_root.iterdir()} != expected_directories:
        raise ValueError("shard dataset directory union is not exact")
    validated: list[dict[str, Any]] = []
    for record, result in zip(records, summary_datasets, strict=True):
        if not isinstance(result, Mapping):
            raise ValueError("shard dataset summary entry is invalid")
        validated.append(
            _validate_dataset(
                results_root / f"{record['ordinal']:04d}",
                expected=record,
                summary_result=result,
                config=config,
                run_contract_sha256=run_contract_sha256,
            )
        )
    if summary.get("prediction_rows") != sum(
        result["n_evaluation"] for result in validated
    ):
        raise ValueError("shard prediction row count is invalid")
    attempts_path = root / "attempts.json"
    if attempts_path.is_symlink() or not attempts_path.is_file():
        raise ValueError("shard attempt history is unsafe")
    try:
        attempts = json.loads(attempts_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("shard attempt history is invalid JSON") from error
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("shard attempt history is invalid")
    completed_attempt = attempts[-1]
    expected_role = "canary" if canary else f"shard-{shard_id}"
    if (
        not isinstance(completed_attempt, Mapping)
        or completed_attempt.get("status") != "completed"
        or completed_attempt.get("job_role") != expected_role
        or not isinstance(completed_attempt.get("started_at_unix_seconds"), (int, float))
        or isinstance(completed_attempt.get("started_at_unix_seconds"), bool)
        or not isinstance(
            completed_attempt.get("completed_at_unix_seconds"), (int, float)
        )
        or isinstance(completed_attempt.get("completed_at_unix_seconds"), bool)
        or completed_attempt["completed_at_unix_seconds"]
        < completed_attempt["started_at_unix_seconds"]
    ):
        raise ValueError("completed shard attempt contract is invalid")
    evidence = {
        "shard_id": shard_id,
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "run_contract_sha256": run_contract_sha256,
        "environment_contract_sha256": contract["environment_contract_sha256"],
        "common_run_contract_sha256": contract["common_run_contract_sha256"],
        "model_runtime_sha": contract["model_runtime_sha"],
        "execution_scope": job_contract["execution_scope"],
        "gpu_name": environment["gpu"]["name"],
        "environment_contract": dict(environment),
        "monitor_window": {
            "started_at_unix_seconds": completed_attempt[
                "started_at_unix_seconds"
            ],
            "completed_at_unix_seconds": completed_attempt[
                "completed_at_unix_seconds"
            ],
        },
    }
    return shard_id, validated, evidence


def _pair_statistics(
    pair: Mapping[str, Any], datasets: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    left, right = (arm["arm_id"] for arm in pair["arms"])

    def deterministic_seed(*tokens: str) -> int:
        import hashlib

        digest = hashlib.sha256("\0".join(tokens).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")

    def two_sided_sign_flip(values: np.ndarray, *, seed: int) -> float:
        observed = abs(float(values.mean()))
        if values.size <= 20:
            total = 1 << int(values.size)
            exceedances = 0
            bit_positions = np.arange(values.size, dtype=np.uint64)
            for start in range(0, total, 2048):
                masks = np.arange(
                    start, min(start + 2048, total), dtype=np.uint64
                )[:, None]
                bits = (masks >> bit_positions[None, :]) & np.uint64(1)
                signs = bits.astype(np.float64) * 2.0 - 1.0
                exceedances += int(
                    np.count_nonzero(np.abs((signs * values).mean(axis=1)) >= observed)
                )
            return float(exceedances / total)
        rng = np.random.default_rng(seed)
        exceedances = 0
        remaining = SIGN_FLIP_RESAMPLES
        while remaining:
            batch = min(remaining, 2048)
            signs = rng.integers(0, 2, size=(batch, values.size), dtype=np.int8)
            signs = signs.astype(np.float64) * 2.0 - 1.0
            exceedances += int(
                np.count_nonzero(np.abs((signs * values).mean(axis=1)) >= observed)
            )
            remaining -= batch
        return float((exceedances + 1) / (SIGN_FLIP_RESAMPLES + 1))

    def comparison(
        subset: Sequence[Mapping[str, Any]], *, metric: str, role: str, subgroup: str
    ) -> dict[str, Any]:
        if not subset:
            raise ValueError(f"paired statistics subgroup is empty: {subgroup}")
        if metric == "accuracy":
            values = np.asarray(
                [
                    dataset["arms"][left][metric]
                    - dataset["arms"][right][metric]
                    for dataset in subset
                ],
                dtype=np.float64,
            )
            definition = "left_accuracy_minus_right_accuracy"
        elif metric == "log_loss":
            values = np.asarray(
                [
                    dataset["arms"][right][metric]
                    - dataset["arms"][left][metric]
                    for dataset in subset
                ],
                dtype=np.float64,
            )
            definition = "right_log_loss_minus_left_log_loss"
        else:
            raise AssertionError(metric)
        if not np.isfinite(values).all():
            raise ValueError("paired metric differences must be finite")
        left_wins = int(np.count_nonzero(values > 0.0))
        right_wins = int(np.count_nonzero(values < 0.0))
        ties = int(values.size - left_wins - right_wins)
        bootstrap_seed = deterministic_seed(pair["pair_id"], subgroup, metric, "bootstrap")
        test_seed = deterministic_seed(pair["pair_id"], subgroup, metric, "sign-flip")
        low, high = paired_bootstrap_ci(
            values, n_resamples=BOOTSTRAP_RESAMPLES, seed=bootstrap_seed
        )
        return {
            "inferential_role": role,
            "difference_definition_positive_means_left_better": definition,
            "dataset_count": int(values.size),
            "paired_differences": [
                {
                    "ordinal": dataset["ordinal"],
                    "dataset": dataset["dataset"],
                    "n_features": dataset["n_features"],
                    "difference": float(value),
                }
                for dataset, value in zip(subset, values, strict=True)
            ],
            "mean_difference": float(values.mean()),
            "median_difference": float(np.median(values)),
            "paired_bootstrap_95ci": [low, high],
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": bootstrap_seed,
            "left_wins": left_wins,
            "right_wins": right_wins,
            "ties": ties,
            "two_sided_exact_sign_test_p": exact_sign_test(left_wins, right_wins),
            "two_sided_paired_sign_flip_p": two_sided_sign_flip(
                values, seed=test_seed
            ),
            "sign_flip_resamples": (
                "exact_enumeration" if values.size <= 20 else SIGN_FLIP_RESAMPLES
            ),
            "sign_flip_seed": None if values.size <= 20 else test_seed,
        }

    groups = {
        "all_datasets": list(datasets),
        "column_count_id_le_100": [
            dataset
            for dataset in datasets
            if dataset["n_features"] <= COLUMN_COUNT_TRAINING_MAXIMUM
        ],
        "column_count_ood_gt_100": [
            dataset
            for dataset in datasets
            if dataset["n_features"] > COLUMN_COUNT_TRAINING_MAXIMUM
        ],
    }

    def group_statistics(
        subgroup: str, subset: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        return {
            "dataset_count": len(subset),
            "column_count_threshold": COLUMN_COUNT_TRAINING_MAXIMUM,
            "column_count_rule": (
                "all"
                if subgroup == "all_datasets"
                else (
                    "n_features<=100"
                    if subgroup == "column_count_id_le_100"
                    else "n_features>100"
                )
            ),
            "accuracy": comparison(
                subset, metric="accuracy", role="primary", subgroup=subgroup
            ),
            "log_loss": comparison(
                subset,
                metric="log_loss",
                role="supportive_unadjusted",
                subgroup=subgroup,
            ),
        }

    return {
        "pair_id": pair["pair_id"],
        "left_arm_id": left,
        "right_arm_id": right,
        "comparison_step": pair["comparison_step"],
        "match_level": pair["match_level"],
        "training_source_commit": pair["training_source_commit"],
        "primary_metric": "accuracy",
        "supportive_metric": "log_loss",
        "column_count_training_maximum": COLUMN_COUNT_TRAINING_MAXIMUM,
        "subgroups": {
            name: group_statistics(name, subset) for name, subset in groups.items()
        },
    }


def _combine(
    *,
    configs: Sequence[Mapping[str, Any]],
    by_config: Mapping[str, Sequence[Mapping[str, Any]]],
    evidence: Mapping[str, Sequence[Mapping[str, Any]]],
    canary_evidence: Mapping[str, Mapping[str, Any]],
    receipt_evidence: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any],
    plan_path: Path,
    plan_file_sha256: str,
    plan_document: Mapping[str, Any],
    analysis_sha: str,
) -> dict[str, Any]:
    expected_ordinals = list(range(len(plan["datasets"])))
    joined: dict[int, dict[str, Any]] = {}
    config_records: list[dict[str, Any]] = []
    for config in configs:
        digest = config["config_sha256"]
        records = list(by_config[digest])
        if sorted(record["ordinal"] for record in records) != expected_ordinals:
            raise ValueError("one pair's dataset union is not the exact frozen roster")
        if len({record["ordinal"] for record in records}) != len(records):
            raise ValueError("one pair contains duplicate dataset outputs")
        records.sort(key=lambda record: record["ordinal"])
        pair = config["pairs"][0]
        for record in records:
            ordinal = record["ordinal"]
            if ordinal not in joined:
                joined[ordinal] = {
                    key: record[key]
                    for key in (
                        "ordinal",
                        "dataset",
                        "info_sha256",
                        "input_sha256",
                        "row_roster_sha256",
                        "class_tokens",
                        "target_sha256",
                        "n_train",
                        "n_evaluation",
                        "n_features",
                        "n_classes",
                    )
                }
                joined[ordinal]["arms"] = {}
                joined[ordinal]["dataset_manifest_sha256_by_pair"] = {}
                joined[ordinal]["arm_manifest_sha256_by_pair"] = {}
                joined[ordinal]["oom_fallback_by_pair"] = {}
                joined[ordinal]["column_count_regime"] = (
                    "id_le_100"
                    if record["n_features"] <= COLUMN_COUNT_TRAINING_MAXIMUM
                    else "ood_gt_100"
                )
            reference = joined[ordinal]
            for key in (
                "dataset",
                "info_sha256",
                "input_sha256",
                "row_roster_sha256",
                "class_tokens",
                "target_sha256",
                "n_train",
                "n_evaluation",
                "n_features",
                "n_classes",
            ):
                _assert_equal(
                    record[key],
                    reference[key],
                    name=f"cross-pair dataset {ordinal} {key}",
                )
            reference["arms"].update(record["arms"])
            reference["dataset_manifest_sha256_by_pair"][pair["pair_id"]] = record[
                "dataset_manifest_sha256"
            ]
            reference["arm_manifest_sha256_by_pair"][pair["pair_id"]] = record[
                "arm_manifest_sha256"
            ]
            reference["oom_fallback_by_pair"][pair["pair_id"]] = record[
                "oom_fallback"
            ]
        config_records.append(
            {
                "private_run_config_sha256": digest,
                "portable_pair_contract": config["portable"],
                "portable_pair_contract_sha256": config["portable_sha256"],
                "checkpoint_contract": config["checkpoint_contract"],
                "campaign_receipts": receipt_evidence[digest],
                "canary": {
                    key: value
                    for key, value in canary_evidence[digest].items()
                    if key
                    in {
                        "shard_id",
                        "manifest_sha256",
                        "run_contract_sha256",
                        "environment_contract_sha256",
                        "common_run_contract_sha256",
                        "gpu_monitor",
                    }
                },
                "shards": [
                    {
                        key: value
                        for key, value in item.items()
                        if key
                        in {
                            "shard_id",
                            "manifest_sha256",
                            "run_contract_sha256",
                            "environment_contract_sha256",
                            "common_run_contract_sha256",
                            "gpu_monitor",
                        }
                    }
                    for item in sorted(
                        evidence[digest], key=lambda item: item["shard_id"]
                    )
                ],
            }
        )
    datasets = [joined[ordinal] for ordinal in expected_ordinals]
    arms = [arm for config in configs for arm in config["arm_order"]]
    arm_statistics = {
        arm_id: {
            "macro_accuracy": float(
                np.mean([dataset["arms"][arm_id]["accuracy"] for dataset in datasets])
            ),
            "macro_log_loss": float(
                np.mean([dataset["arms"][arm_id]["log_loss"] for dataset in datasets])
            ),
        }
        for arm_id in arms
    }
    pair_statistics = [
        _pair_statistics(config["pairs"][0], datasets) for config in configs
    ]
    return {
        "schema_version": 1,
        "kind": AGGREGATE_KIND,
        "formal_eligible": False,
        "evidence_scope": "exploratory_discovery_only",
        "analysis_sha": analysis_sha,
        "shard_plan_sha256": plan_file_sha256,
        "shard_plan_document_sha256": plan_document["sha256"],
        "roster_kind": plan["roster_kind"],
        "roster_sha256": plan["roster_sha256"],
        "dataset_count": len(datasets),
        "arm_count": len(arms),
        "pair_count": len(configs),
        "fit_split": "train",
        "evaluation_split": "val",
        "test_split_policy": "hashed_for_integrity_only_not_used",
        "column_count_subgroup_contract": {
            "training_prior_max_features": COLUMN_COUNT_TRAINING_MAXIMUM,
            "id": "n_features<=100",
            "ood": "n_features>100",
            "threshold_source": "frozen_training_prior_all_stages",
        },
        "statistics_contract": {
            "accuracy": "primary",
            "log_loss": "supportive_unadjusted",
            "paired_bootstrap_confidence": 0.95,
            "paired_bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "two_sided_paired_test": "sign_flip_plus_exact_sign_test",
            "sign_flip_resamples_above_20_pairs": SIGN_FLIP_RESAMPLES,
        },
        "cross_pair_statistics_policy": "none_training_lineages_differ",
        "run_groups": config_records,
        "arm_statistics": arm_statistics,
        "pair_statistics": pair_statistics,
        "datasets": datasets,
    }


def _run(args: argparse.Namespace) -> int:
    analysis_root = absolute_path(
        args.analysis_root, name="analysis_root", directory=True
    )
    analysis_sha = verify_clean_detached_git(
        analysis_root, expected_sha=args.expected_analysis_sha
    )
    plan_path = absolute_path(args.shard_plan, name="shard_plan")
    plan_document, plan_file_sha256 = load_json_object_with_sha256(
        plan_path, name="TALENT shard plan"
    )
    plan = validate_shard_plan(
        plan_document, expected_roster=frozen_discovery_roster(analysis_root)
    )
    if plan.get("analysis_sha") != analysis_sha:
        raise ValueError("shard plan belongs to a different analysis commit")
    if (
        args.expected_shard_plan_sha256 is not None
        and plan_file_sha256 != args.expected_shard_plan_sha256
    ):
        raise ValueError("shard plan differs from the submission-bound digest")
    configs, config_by_digest = _load_configs(args.run_config)
    if args.expected_run_config_sha256 is not None and (
        len(configs) != 1
        or configs[0]["config_sha256"] != args.expected_run_config_sha256
    ):
        raise ValueError("run config differs from the submission-bound digest")
    shard_roots = [
        absolute_path(path, name="shard_output", directory=True)
        for path in args.shard_output
    ]
    expected_shard_count = len(configs) * plan["shard_count"]
    if len(shard_roots) != expected_shard_count:
        raise ValueError("aggregate requires every planned shard for every pair")
    canary_roots = [
        absolute_path(path, name="canary_output", directory=True)
        for path in args.canary_output
    ]
    if len(canary_roots) != len(configs):
        raise ValueError("aggregate requires one canary output for every pair")
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    observed_shards: set[tuple[str, str]] = set()
    for root in shard_roots:
        summary = load_json_object(root / "summary.json", name="shard summary")
        digest = summary.get("private_run_config_sha256")
        if not isinstance(digest, str) or digest not in config_by_digest:
            raise ValueError("shard output does not belong to a supplied run config")
        shard_id, records, shard_evidence = _validate_shard(
            root,
            config=config_by_digest[digest],
            plan=plan,
            plan_path=plan_path,
            plan_file_sha256=plan_file_sha256,
            plan_document=plan_document,
            analysis_sha=analysis_sha,
        )
        key = (digest, shard_id)
        if key in observed_shards:
            raise ValueError("aggregate contains a duplicate pair/shard output")
        observed_shards.add(key)
        by_config[digest].extend(records)
        evidence[digest].append(shard_evidence)
    expected_shards = {
        (config["config_sha256"], shard["shard_id"])
        for config in configs
        for shard in plan["shards"]
    }
    if observed_shards != expected_shards:
        raise ValueError("aggregate pair/shard union is not exact")

    canary_evidence: dict[str, dict[str, Any]] = {}
    for root in canary_roots:
        summary = load_json_object(root / "summary.json", name="canary summary")
        digest = summary.get("private_run_config_sha256")
        if not isinstance(digest, str) or digest not in config_by_digest:
            raise ValueError("canary output does not belong to a supplied run config")
        if digest in canary_evidence:
            raise ValueError("aggregate contains duplicate canary evidence")
        shard_id, _, current = _validate_shard(
            root,
            config=config_by_digest[digest],
            plan=plan,
            plan_path=plan_path,
            plan_file_sha256=plan_file_sha256,
            plan_document=plan_document,
            analysis_sha=analysis_sha,
            canary=True,
        )
        if shard_id != "canary":
            raise ValueError("canary output has the wrong job alias")
        canary_evidence[digest] = current
    if set(canary_evidence) != set(config_by_digest):
        raise ValueError("canary evidence union is not exact")

    for config in configs:
        digest = config["config_sha256"]
        jobs = [canary_evidence[digest], *evidence[digest]]
        _require_consistent_job_group(config["pairs"][0]["pair_id"], jobs)

    expected_gpu_jobs = {
        (digest, job["shard_id"]): job
        for digest in config_by_digest
        for job in [canary_evidence[digest], *evidence[digest]]
    }
    gpu_paths = [absolute_path(path, name="gpu_csv") for path in args.gpu_csv]
    if len(gpu_paths) != len(expected_gpu_jobs):
        raise ValueError("aggregate requires one GPU CSV for every canary/full job")
    observed_gpu_jobs: set[tuple[str, str]] = set()
    gpu_digest_by_path: dict[Path, str] = {}
    for path in gpu_paths:
        key, monitor = _validate_gpu_csv(
            path, expected_jobs=expected_gpu_jobs, analysis_sha=analysis_sha
        )
        if key in observed_gpu_jobs:
            raise ValueError("aggregate contains duplicate GPU monitor evidence")
        observed_gpu_jobs.add(key)
        expected_gpu_jobs[key]["gpu_monitor"] = monitor
        gpu_digest_by_path[path] = monitor["csv_sha256"]
    if observed_gpu_jobs != set(expected_gpu_jobs):
        raise ValueError("GPU monitor job union is not exact")

    submission_paths = [
        absolute_path(path, name="submission_receipt")
        for path in args.submission_receipt
    ]
    release_paths = [
        absolute_path(path, name="release_receipt") for path in args.release_receipt
    ]
    if len(submission_paths) != len(configs) or len(release_paths) != len(configs):
        raise ValueError("aggregate requires one receipt pair per run config")
    release_by_submission: dict[str, Path] = {}
    for path in release_paths:
        document = load_json_object(path, name="release receipt")
        payload = validate_self_hashed_document(
            document, kind=RELEASE_KIND, name="release receipt"
        )
        submission_digest = payload.get("submission_document_sha256")
        if not isinstance(submission_digest, str) or submission_digest in release_by_submission:
            raise ValueError("release receipt binding is duplicate or malformed")
        release_by_submission[submission_digest] = path
    config_by_portable = {
        config["portable_sha256"]: config for config in configs
    }
    receipt_evidence: dict[str, dict[str, Any]] = {}
    receipt_paths_by_config: dict[str, tuple[Path, Path]] = {}
    for submission_path in submission_paths:
        document = load_json_object(submission_path, name="submission receipt")
        payload = validate_self_hashed_document(
            document, kind=SUBMISSION_KIND, name="submission receipt"
        )
        config = config_by_portable.get(payload.get("pair_contract_sha256"))
        release_path = release_by_submission.get(document["sha256"])
        if config is None or release_path is None:
            raise ValueError("submission receipt does not bind a supplied campaign")
        digest = config["config_sha256"]
        if digest in receipt_evidence:
            raise ValueError("aggregate contains duplicate campaign receipts")
        receipt_evidence[digest] = _validate_campaign_receipts(
            submission_path=submission_path,
            release_path=release_path,
            config=config,
            checkpoint_contract=config["checkpoint_contract"],
            analysis_sha=analysis_sha,
            plan_file_sha256=plan_file_sha256,
            plan_document_sha256=plan_document["sha256"],
            canary_ordinals=[
                record["ordinal"] for record in canary_dataset_records(plan)
            ],
        )
        receipt_paths_by_config[digest] = (submission_path, release_path)
    if set(receipt_evidence) != set(config_by_digest):
        raise ValueError("campaign receipt union is not exact")
    aggregate = _combine(
        configs=configs,
        by_config=by_config,
        evidence=evidence,
        canary_evidence=canary_evidence,
        receipt_evidence=receipt_evidence,
        plan=plan,
        plan_path=plan_path,
        plan_file_sha256=plan_file_sha256,
        plan_document=plan_document,
        analysis_sha=analysis_sha,
    )
    _assert_public_safe(aggregate)
    verify_clean_detached_git(analysis_root, expected_sha=analysis_sha)
    if sha256_file(plan_path) != aggregate["shard_plan_sha256"]:
        raise RuntimeError("shard plan changed during aggregation")
    for config in configs:
        if sha256_file(config["path"]) != config["config_sha256"]:
            raise RuntimeError("private run config changed during aggregation")
        for pair in config["pairs"]:
            for arm in pair["arms"]:
                if sha256_file(arm["checkpoint"]) != arm["checkpoint_sha256"]:
                    raise RuntimeError("checkpoint changed during aggregation")
            for receipt in pair["lineage_receipts"]:
                if sha256_file(receipt["path"]) != receipt["sha256"]:
                    raise RuntimeError("lineage receipt changed during aggregation")
        submission_path, release_path = receipt_paths_by_config[
            config["config_sha256"]
        ]
        campaign = receipt_evidence[config["config_sha256"]]
        if (
            sha256_file(submission_path)
            != campaign["submission_receipt_file_sha256"]
            or sha256_file(release_path) != campaign["release_receipt_file_sha256"]
        ):
            raise RuntimeError("campaign receipt changed during aggregation")
    if any(sha256_file(path) != digest for path, digest in gpu_digest_by_path.items()):
        raise RuntimeError("GPU monitor CSV changed during aggregation")
    output = absolute_path(args.output_dir, name="output_dir", absent=True)
    require_disjoint_output(
        output,
        [
            analysis_root,
            plan_path,
            *(config["path"] for config in configs),
            *shard_roots,
            *canary_roots,
            *gpu_paths,
            *submission_paths,
            *release_paths,
            *(
                arm["checkpoint"]
                for config in configs
                for pair in config["pairs"]
                for arm in pair["arms"]
            ),
            *(
                receipt["path"]
                for config in configs
                for pair in config["pairs"]
                for receipt in pair["lineage_receipts"]
            ),
        ],
        name="TALENT aggregate output",
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        atomic_json(staging / "aggregate.json", aggregate)
        atomic_json(
            staging / "manifest.json", directory_manifest(staging, kind=AGGREGATE_KIND)
        )
        os.replace(staging, output)
        fsync_directory(output.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": "completed",
                "dataset_count": aggregate["dataset_count"],
                "arm_count": aggregate["arm_count"],
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = _parser().parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute():
        raise ValueError("output_dir must be absolute")
    output_parent = output.parent.resolve(strict=True)
    require_disjoint_output(
        output_parent / output.name,
        [
            Path(args.analysis_root),
            Path(args.shard_plan),
            *(Path(path) for path in args.run_config),
            *(Path(path) for path in args.canary_output),
            *(Path(path) for path in args.shard_output),
            *(Path(path) for path in args.submission_receipt),
            *(Path(path) for path in args.release_receipt),
            *(Path(path) for path in args.gpu_csv),
        ],
        name="TALENT aggregate output",
    )
    lock_path = output_parent / f".{output.name}.lock"
    with _RunLock(lock_path):
        return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
