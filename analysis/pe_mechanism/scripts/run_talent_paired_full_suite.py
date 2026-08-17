#!/usr/bin/env python3
"""Run one shard of paired checkpoints on the frozen TALENT discovery roster."""

from __future__ import annotations

import argparse
import atexit
from dataclasses import asdict, is_dataclass
import fcntl
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from pe_mechanism.talent_full_suite import (
    ARM_DATASET_ARTIFACTS,
    ARM_DATASET_KIND,
    DATASET_ARTIFACTS,
    DATASET_KIND,
    DISK_OFFLOAD_MIN_FREE_BYTES,
    DISK_OFFLOAD_SCRATCH_CONTRACT,
    SHARD_KIND,
    TALENT_ESTIMATOR_OPTIONS,
    OOM_FALLBACK_SEQUENCE,
    absolute_path,
    array_sha256,
    atomic_json,
    canary_dataset_records,
    directory_manifest,
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
    validate_shard_plan,
    validate_observed_treatment,
    verify_clean_detached_git,
)


ESTIMATOR_OPTIONS = TALENT_ESTIMATOR_OPTIONS
PREDICTION_CHUNK_ROWS = 65_536


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--talent-root", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--shard-plan", required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--expected-analysis-sha", required=True)
    parser.add_argument("--expected-model-sha", required=True)
    parser.add_argument("--expected-run-config-sha256", required=True)
    parser.add_argument("--expected-shard-plan-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scratch-root", required=True)
    return parser


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
            raise RuntimeError("another evaluator owns this output lock") from error
        return self

    def __exit__(self, *_: object) -> None:
        assert self.descriptor is not None
        fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        os.close(self.descriptor)
        self.descriptor = None


def _available_bytes(path: Path) -> int:
    values = os.statvfs(path)
    return int(values.f_bavail * values.f_frsize)


def _make_attempt_scratch(
    scratch_root: Path,
    *,
    ordinal: int,
    offload_mode: str,
    arm_ids: Sequence[str],
) -> tuple[Path, dict[str, Path], int]:
    available = _available_bytes(scratch_root)
    if available < DISK_OFFLOAD_MIN_FREE_BYTES:
        raise RuntimeError("TALENT offload scratch fell below the 20 GiB reserve")
    attempt = Path(
        tempfile.mkdtemp(
            prefix=f"dataset-{ordinal:04d}-{offload_mode}-",
            dir=scratch_root,
        )
    )
    arm_directories: dict[str, Path] = {}
    try:
        for arm_id in arm_ids:
            directory = attempt / arm_id
            directory.mkdir(mode=0o700)
            arm_directories[arm_id] = directory
        fsync_directory(attempt)
        return attempt, arm_directories, available
    except BaseException:
        _remove_attempt_scratch(attempt, scratch_root=scratch_root)
        raise


def _remove_attempt_scratch(path: Path, *, scratch_root: Path) -> None:
    if path.parent != scratch_root:
        raise RuntimeError("attempt scratch escaped the required job-local root")
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise RuntimeError("attempt scratch was replaced with an unsafe object")
    for child in path.rglob("*"):
        child_metadata = child.lstat()
        if child.is_symlink() or not (
            stat.S_ISDIR(child_metadata.st_mode)
            or stat.S_ISREG(child_metadata.st_mode)
        ):
            raise RuntimeError("attempt scratch contains an unsafe object")
    shutil.rmtree(path)
    fsync_directory(scratch_root)


def _runtime_environment(analysis_root: Path, model_root: Path) -> dict[str, Any]:
    import pandas
    import pe_mechanism
    import sklearn
    import tabicl
    import torch

    origins = {
        "pe_mechanism": Path(pe_mechanism.__file__).resolve(strict=True),
        "tabicl": Path(tabicl.__file__).resolve(strict=True),
    }
    if not origins["pe_mechanism"].is_relative_to(analysis_root):
        raise RuntimeError("pe_mechanism was imported outside the analysis tree")
    if not origins["tabicl"].is_relative_to(model_root):
        raise RuntimeError("tabicl was imported outside the model tree")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("TALENT evaluation requires exactly one visible CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    driver_lines = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if len(driver_lines) != 1 or not driver_lines[0].strip():
        raise RuntimeError("could not establish one NVIDIA driver version")
    packages = {
        distribution: importlib.metadata.version(distribution)
        for distribution in ("numpy", "pandas", "scikit-learn", "torch")
    }
    return {
        "schema_version": 1,
        "python": sys.version,
        "python_executable_sha256": sha256_file(
            Path(sys.executable).resolve(strict=True)
        ),
        "packages": packages,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_driver": driver_lines[0].strip(),
        "gpu": {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [properties.major, properties.minor],
        },
        "imports": {
            "pe_mechanism": {
                "source_tree": "analysis",
                "relative_path": origins["pe_mechanism"]
                .relative_to(analysis_root)
                .as_posix(),
            },
            "tabicl": {
                "source_tree": "model",
                "relative_path": origins["tabicl"].relative_to(model_root).as_posix(),
            },
        },
    }


def _stage_checkpoints(config: Mapping[str, Any]) -> tuple[Path, dict[str, Path]]:
    temporary_parent = os.environ.get("SLURM_TMPDIR")
    if not temporary_parent or not Path(temporary_parent).is_dir():
        raise ValueError("SLURM_TMPDIR must name existing job-local storage")
    directory = Path(
        tempfile.mkdtemp(prefix="talent-paired-checkpoints-", dir=temporary_parent)
    )
    atexit.register(shutil.rmtree, directory, True)
    staged: dict[str, Path] = {}
    try:
        for pair in config["pairs"]:
            for arm in pair["arms"]:
                destination = directory / f"{arm['arm_id']}.ckpt"
                shutil.copyfile(arm["checkpoint"], destination)
                with destination.open("rb") as handle:
                    os.fsync(handle.fileno())
                if sha256_file(destination) != arm["checkpoint_sha256"]:
                    raise RuntimeError(f"staged checkpoint changed: {arm['arm_id']}")
                destination.chmod(0o400)
                staged[arm["arm_id"]] = destination
        fsync_directory(directory)
        return directory, staged
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def _class_token(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"


def _class_tokens(values: Sequence[Any]) -> list[str]:
    return [_class_token(value) for value in values]


def _row_roster_sha256(dataset: str, split: str, labels: np.ndarray) -> str:
    import hashlib

    tokens = [_class_token(value) for value in np.asarray(labels).reshape(-1)]
    return hashlib.sha256(
        json.dumps(
            {"dataset": dataset, "split": split, "labels": tokens},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _forward_schedule(entries: Sequence[Any]) -> dict[str, Any]:
    normalized = []
    for entry in entries:
        if is_dataclass(entry):
            normalized.append(asdict(entry))
        elif isinstance(entry, Mapping):
            normalized.append(dict(entry))
        else:
            normalized.append(vars(entry))
    return {"entries": normalized, "sha256": json_document_sha256(normalized)}


def _assert_treatment(
    driver: Any,
    treatment: Mapping[str, Any],
    *,
    model_sha: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    import torch

    if driver.model_sha != model_sha or driver.checkpoint_sha != checkpoint_sha256:
        raise RuntimeError("official driver source/checkpoint identity changed")
    if driver.source_evidence_level != "strict":
        raise RuntimeError("official driver lost strict source evidence")
    model = driver.estimator.model_
    parameter_dtypes = {parameter.dtype for parameter in model.parameters()}
    if parameter_dtypes != {torch.float32}:
        raise RuntimeError("official driver model parameters are not strict FP32")
    row = model.row_interactor
    kind = treatment["kind"]
    has_rope = getattr(getattr(row, "tf_row", None), "rope", None) is not None
    fingerprint = bool(getattr(model, "row_fingerprint", False))
    fragments = (
        getattr(row, "fingerprint_q_gates", None),
        getattr(row, "fingerprint_k_gates", None),
        getattr(row, "fingerprint_q_projections", None),
        getattr(row, "fingerprint_k_projections", None),
    )
    has_fingerprint = all(value is not None for value in fragments)
    if kind == "rope":
        valid = (
            getattr(model, "row_identity_mode", None) == "rope"
            and getattr(row, "identity_mode", None) == "rope"
            and has_rope
            and not fingerprint
            and not has_fingerprint
        )
    elif kind == "none":
        valid = (
            getattr(model, "row_identity_mode", None) == "none"
            and getattr(row, "identity_mode", None) == "none"
            and not has_rope
            and not fingerprint
            and not has_fingerprint
        )
    elif kind == "fingerprint":
        valid = (
            getattr(model, "row_identity_mode", None) == "none"
            and getattr(row, "identity_mode", None) == "none"
            and not has_rope
            and fingerprint
            and has_fingerprint
            and getattr(model, "row_fingerprint_dim", None) == treatment["dimension"]
        )
    else:  # protected by configuration validation
        raise AssertionError(kind)
    if not valid:
        raise RuntimeError(f"loaded model does not implement treatment {kind}")
    return {
        "kind": kind,
        "row_identity_mode": getattr(model, "row_identity_mode", None),
        "row_fingerprint": fingerprint,
        "row_fingerprint_dim": getattr(model, "row_fingerprint_dim", None),
        "row_rope_installed": has_rope,
        "parameter_dtype": "float32",
    }


def _all_missing_columns(values: Any) -> np.ndarray:
    if hasattr(values, "isna"):
        return np.asarray(values.isna().all(axis=0), dtype=bool)
    array = np.asarray(values)
    return np.asarray(
        [all(value is None or (isinstance(value, float) and math.isnan(value)) for value in array[:, index])
         for index in range(array.shape[1])],
        dtype=bool,
    )


def _row_slice(values: Any, start: int, stop: int) -> Any:
    if hasattr(values, "iloc"):
        return values.iloc[start:stop].copy()
    return np.asarray(values)[start:stop].copy()


def _predict_in_chunks(
    driver: Any,
    *,
    X: Any,
    y: np.ndarray,
    treatment: Mapping[str, Any],
    model_sha: str,
    checkpoint_sha256: str,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    import torch

    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    expected_classes: list[str] | None = None
    full_missing = _all_missing_columns(X)
    chunk_count = 0
    for start in range(0, len(y), PREDICTION_CHUNK_ROWS):
        stop = min(len(y), start + PREDICTION_CHUNK_ROWS)
        chunk_X = _row_slice(X, start, stop)
        chunk_y = np.asarray(y[start:stop])
        if not np.array_equal(_all_missing_columns(chunk_X), full_missing):
            raise RuntimeError("prediction chunk changes all-missing feature mask")
        result = driver.predict_proba(chunk_X, y=chunk_y)
        _assert_treatment(
            driver,
            treatment,
            model_sha=model_sha,
            checkpoint_sha256=checkpoint_sha256,
        )
        if (
            not result.exact_baseline_verified
            or result.source_evidence_level != "strict"
            or result.forward_calls
        ):
            raise RuntimeError("official prediction did not remain direct and strict")
        current = np.asarray(result.probabilities, dtype=np.float32)
        baseline = np.asarray(result.baseline_probabilities, dtype=np.float32)
        if (
            current.ndim != 2
            or current.shape[0] != len(chunk_y)
            or not np.isfinite(current).all()
            or not np.array_equal(current, baseline)
        ):
            raise RuntimeError("official prediction probabilities are invalid")
        classes = _class_tokens(result.classes)
        if expected_classes is None:
            expected_classes = classes
        elif classes != expected_classes:
            raise RuntimeError("class encoding changed between chunks")
        target = np.asarray(driver.estimator.y_encoder_.transform(chunk_y), dtype=np.int64)
        if np.any(target < 0) or np.any(target >= current.shape[1]):
            raise RuntimeError("encoded validation targets are invalid")
        probabilities.append(current)
        targets.append(target)
        chunk_count += 1
        del result, baseline
        gc.collect()
        torch.cuda.empty_cache()
    if expected_classes is None or not probabilities:
        raise RuntimeError("validation split produced no predictions")
    return (
        np.concatenate(probabilities, axis=0),
        np.concatenate(targets, axis=0),
        expected_classes,
        {
            "chunk_count": chunk_count,
            "maximum_rows_per_call": PREDICTION_CHUNK_ROWS,
            "source_evidence_level": "strict",
            "all_chunks_exact_baseline_verified": True,
        },
    )


def _evaluate_dataset(
    *,
    dataset_record: Mapping[str, Any],
    talent_root: Path,
    config: Mapping[str, Any],
    staged: Mapping[str, Path],
    model_root: Path,
    model_sha: str,
    run_contract_sha256: str,
    destination: Path,
    scratch_root: Path,
) -> dict[str, Any]:
    import torch
    from pe_mechanism.official_tabicl import (
        _expected_forward_schedule,
        fit_official_tabicl_driver,
        load_raw_talent_splits,
    )

    name = dataset_record["name"]
    dataset = load_raw_talent_splits(talent_root / name, trusted_pickle=True)
    if (
        dataset.info_sha256 != dataset_record["info_sha256"]
        or dict(sorted(dataset.input_sha256.items()))
        != dataset_record["input_sha256"]
        or len(dataset.train.y) != dataset_record["n_train"]
        or len(dataset.val.y) != dataset_record["n_validation"]
        or len(dataset.test.y) != dataset_record["n_test"]
        or dataset.n_numeric_features + dataset.n_categorical_features
        != dataset_record["n_features"]
    ):
        raise RuntimeError(f"TALENT input differs from shard plan: {name}")
    arm_lookup = {
        arm["arm_id"]: arm for pair in config["pairs"] for arm in pair["arms"]
    }
    attempted_levels: list[str] = []
    scratch_available_bytes: dict[str, int] = {}
    selected_level: str | None = None
    last_oom: BaseException | None = None
    for offload_mode in OOM_FALLBACK_SEQUENCE:
        attempted_levels.append(offload_mode)
        attempt_scratch, arm_scratch, available = _make_attempt_scratch(
            scratch_root,
            ordinal=dataset_record["ordinal"],
            offload_mode=offload_mode,
            arm_ids=config["arm_order"],
        )
        scratch_available_bytes[offload_mode] = available
        arm_records: dict[str, Any] = {}
        probabilities: dict[str, np.ndarray] = {}
        encoded_target: np.ndarray | None = None
        expected_classes: list[str] | None = None
        expected_schedule: dict[str, Any] | None = None
        driver: Any | None = None
        try:
            for arm_id in config["arm_order"]:
                arm = arm_lookup[arm_id]
                fit_started = time.monotonic()
                estimator_options = {
                    **ESTIMATOR_OPTIONS,
                    "offload_mode": offload_mode,
                }
                if offload_mode == "disk":
                    estimator_options["disk_offload_dir"] = str(
                        arm_scratch[arm_id]
                    )
                driver = fit_official_tabicl_driver(
                    staged[arm_id],
                    dataset.train.X,
                    dataset.train.y,
                    device="cuda",
                    model_sha=model_sha,
                    estimator_options=estimator_options,
                    expected_source_root=model_root,
                    fit_context="talent-train",
                )
                fit_seconds = time.monotonic() - fit_started
                treatment = _assert_treatment(
                    driver,
                    arm["treatment"],
                    model_sha=model_sha,
                    checkpoint_sha256=arm["checkpoint_sha256"],
                )
                schedule = _forward_schedule(
                    _expected_forward_schedule(driver.estimator)
                )
                if expected_schedule is None:
                    expected_schedule = schedule
                elif schedule != expected_schedule:
                    raise RuntimeError("paired arms used different inference schedules")
                predict_started = time.monotonic()
                current, target, classes, prediction_contract = _predict_in_chunks(
                    driver,
                    X=dataset.val.X,
                    y=np.asarray(dataset.val.y),
                    treatment=arm["treatment"],
                    model_sha=model_sha,
                    checkpoint_sha256=arm["checkpoint_sha256"],
                )
                predict_seconds = time.monotonic() - predict_started
                if expected_classes is None:
                    expected_classes = classes
                    encoded_target = target
                elif classes != expected_classes or not np.array_equal(
                    target, encoded_target
                ):
                    raise RuntimeError(
                        "paired arms used different class/target encodings"
                    )
                selected = current[np.arange(target.size), target]
                accuracy = float(np.mean(np.argmax(current, axis=1) == target))
                log_loss = float(
                    -np.log(
                        np.clip(selected, np.finfo(np.float32).tiny, 1.0)
                    ).mean()
                )
                probabilities[arm_id] = current
                arm_records[arm_id] = {
                    "accuracy": accuracy,
                    "log_loss": log_loss,
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "checkpoint_sha256": driver.checkpoint_sha,
                    "model_runtime_sha": driver.model_sha,
                    "treatment": treatment,
                    "probabilities_sha256": array_sha256(current),
                    "official_forward_schedule": schedule,
                    "prediction_contract": prediction_contract,
                    "offload_mode": offload_mode,
                }
                driver = None
                gc.collect()
                torch.cuda.empty_cache()
            selected_level = offload_mode
            break
        except torch.cuda.OutOfMemoryError as error:
            last_oom = error
            arm_records.clear()
            probabilities.clear()
            encoded_target = None
            expected_classes = None
            driver = None
            gc.collect()
            torch.cuda.empty_cache()
            continue
        finally:
            driver = None
            gc.collect()
            torch.cuda.empty_cache()
            _remove_attempt_scratch(attempt_scratch, scratch_root=scratch_root)
    if selected_level is None:
        assert last_oom is not None
        raise RuntimeError("all fixed TALENT OOM fallback levels failed") from last_oom
    assert encoded_target is not None and expected_classes is not None
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{dataset_record['ordinal']:04d}.tmp-", dir=destination.parent
        )
    )
    try:
        common = {
            "schema_version": 1,
            "run_contract_sha256": run_contract_sha256,
            "ordinal": dataset_record["ordinal"],
            "dataset": name,
            "fit_split": "train",
            "evaluation_split": "val",
            "n_train": int(len(dataset.train.y)),
            "n_evaluation": int(len(dataset.val.y)),
            "n_features": dataset_record["n_features"],
            "n_classes": len(expected_classes),
            "row_sampling": "none_full_split_original_order",
            "row_roster_sha256": {
                "train": _row_roster_sha256(name, "train", dataset.train.y),
                "val": _row_roster_sha256(name, "val", dataset.val.y),
            },
            "class_tokens": expected_classes,
            "info_sha256": dataset.info_sha256,
            "input_sha256": dict(sorted(dataset.input_sha256.items())),
            "target_sha256": array_sha256(encoded_target),
        }
        arms_root = staging / "arms"
        arms_root.mkdir()
        task_arms: dict[str, Any] = {}
        for arm_id in config["arm_order"]:
            arm_root = arms_root / arm_id
            arm_root.mkdir()
            np.savez_compressed(
                arm_root / "predictions.npz",
                target=encoded_target,
                probabilities=probabilities[arm_id],
            )
            with (arm_root / "predictions.npz").open("rb") as handle:
                os.fsync(handle.fileno())
            arm_result = {**common, "arm_id": arm_id, **arm_records[arm_id]}
            atomic_json(arm_root / "result.json", arm_result)
            atomic_json(
                arm_root / "manifest.json",
                directory_manifest(arm_root, kind=ARM_DATASET_KIND),
            )
            task_arms[arm_id] = {
                **arm_records[arm_id],
                "arm_manifest_sha256": sha256_file(arm_root / "manifest.json"),
            }
        task = {
            **common,
            "oom_fallback": {
                "sequence": list(OOM_FALLBACK_SEQUENCE),
                "attempted_levels": attempted_levels,
                "selected_level": selected_level,
                "retry_scope": "discard_entire_dataset_pair_and_restart_both_arms",
                "non_oom_retry": False,
                "disk_offload_scratch": {
                    **DISK_OFFLOAD_SCRATCH_CONTRACT,
                    "available_bytes_before_attempt": scratch_available_bytes,
                },
            },
            "arms": task_arms,
        }
        atomic_json(staging / "task.json", task)
        atomic_json(
            staging / "manifest.json", directory_manifest(staging, kind=DATASET_KIND)
        )
        os.replace(staging, destination)
        fsync_directory(destination.parent)
        return task
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _cached_dataset(
    destination: Path,
    *,
    expected: Mapping[str, Any],
    run_contract_sha256: str,
    config: Mapping[str, Any],
    model_sha: str,
    talent_root: Path,
) -> dict[str, Any]:
    from pe_mechanism.official_tabicl import load_raw_talent_splits

    if destination.is_symlink() or not destination.is_dir():
        raise RuntimeError("cached dataset directory is unsafe")
    if {path.name for path in destination.iterdir()} != DATASET_ARTIFACTS:
        raise RuntimeError("cached dataset artifact roster is invalid")
    validate_directory_manifest(destination, kind=DATASET_KIND)
    task = load_json_object(destination / "task.json", name="dataset task")
    expected_task_keys = {
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
    }
    if (
        set(task) != expected_task_keys
        or task.get("schema_version") != 1
        or task.get("run_contract_sha256") != run_contract_sha256
        or task.get("dataset") != expected["name"]
        or task.get("ordinal") != expected["ordinal"]
        or task.get("input_sha256") != expected["input_sha256"]
        or task.get("info_sha256") != expected["info_sha256"]
        or task.get("n_train") != expected["n_train"]
        or task.get("n_evaluation") != expected["n_validation"]
        or task.get("n_features") != expected["n_features"]
        or task.get("n_classes") != expected["n_classes"]
        or task.get("fit_split") != "train"
        or task.get("evaluation_split") != "val"
        or task.get("row_sampling") != "none_full_split_original_order"
    ):
        raise RuntimeError("cached dataset contract differs from this run")
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
        or fallback.get("retry_scope")
        != "discard_entire_dataset_pair_and_restart_both_arms"
        or fallback.get("non_oom_retry") is not False
        or fallback.get("selected_level") not in OOM_FALLBACK_SEQUENCE
        or fallback.get("attempted_levels")
        != list(
            OOM_FALLBACK_SEQUENCE[
                : OOM_FALLBACK_SEQUENCE.index(fallback["selected_level"]) + 1
            ]
        )
    ):
        raise RuntimeError("cached OOM fallback contract is invalid")
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
        raise RuntimeError("cached disk-offload scratch contract is invalid")
    dataset_root = talent_root / expected["name"]
    if dataset_root.is_symlink() or not dataset_root.is_dir():
        raise RuntimeError("cached TALENT input directory is unsafe")
    dataset = load_raw_talent_splits(dataset_root, trusted_pickle=True)
    if (
        dataset.info_sha256 != expected["info_sha256"]
        or dict(sorted(dataset.input_sha256.items())) != expected["input_sha256"]
        or len(dataset.train.y) != expected["n_train"]
        or len(dataset.val.y) != expected["n_validation"]
        or len(dataset.test.y) != expected["n_test"]
        or dataset.n_numeric_features + dataset.n_categorical_features
        != expected["n_features"]
        or task.get("row_roster_sha256")
        != {
            "train": _row_roster_sha256(
                expected["name"], "train", dataset.train.y
            ),
            "val": _row_roster_sha256(expected["name"], "val", dataset.val.y),
        }
    ):
        raise RuntimeError("cached TALENT input bytes/rows changed")
    arm_lookup = {
        arm["arm_id"]: arm for pair in config["pairs"] for arm in pair["arms"]
    }
    arms = task.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(config["arm_order"]):
        raise RuntimeError("cached dataset arm roster is invalid")
    arms_root = destination / "arms"
    if arms_root.is_symlink() or not arms_root.is_dir() or {
        path.name for path in arms_root.iterdir()
    } != set(config["arm_order"]):
        raise RuntimeError("cached arm directory union is invalid")
    n_classes = task["n_classes"]
    schedules = []
    for arm_id in config["arm_order"]:
        arm_root = arms_root / arm_id
        if arm_root.is_symlink() or not arm_root.is_dir() or {
            path.name for path in arm_root.iterdir()
        } != ARM_DATASET_ARTIFACTS:
            raise RuntimeError("cached arm artifact roster is invalid")
        validate_directory_manifest(arm_root, kind=ARM_DATASET_KIND)
        if sha256_file(arm_root / "manifest.json") != arms[arm_id].get(
            "arm_manifest_sha256"
        ):
            raise RuntimeError("cached arm manifest binding is invalid")
        arm_result = load_json_object(arm_root / "result.json", name="arm result")
        expected_arm_result = {
            key: value for key, value in task.items() if key not in {"arms", "oom_fallback"}
        }
        expected_arm_result.update(
            {
                "arm_id": arm_id,
                **{
                    key: value
                    for key, value in arms[arm_id].items()
                    if key != "arm_manifest_sha256"
                },
            }
        )
        if dict(arm_result) != expected_arm_result:
            raise RuntimeError("cached arm result differs from dataset task")
        try:
            with np.load(arm_root / "predictions.npz", allow_pickle=False) as archive:
                if set(archive.files) != {"target", "probabilities"}:
                    raise RuntimeError("cached arm prediction keys are invalid")
                target = np.asarray(archive["target"])
                probability = np.asarray(archive["probabilities"])
        except (OSError, ValueError) as error:
            raise RuntimeError("cached arm prediction archive is invalid") from error
        arm = arm_lookup[arm_id]
        if (
            target.dtype != np.dtype("int64")
            or target.shape != (expected["n_validation"],)
            or array_sha256(target) != task.get("target_sha256")
            or np.any(target < 0)
            or np.any(target >= n_classes)
            or probability.dtype != np.dtype("float32")
            or probability.shape != (target.size, n_classes)
            or not np.isfinite(probability).all()
            or np.any(probability < 0.0)
            or not np.allclose(
                probability.sum(axis=1), 1.0, rtol=0.0, atol=2e-5
            )
            or array_sha256(probability)
            != arm_result.get("probabilities_sha256")
            or arm_result.get("checkpoint_sha256") != arm["checkpoint_sha256"]
            or arm_result.get("model_runtime_sha") != model_sha
            or arm_result.get("offload_mode") != fallback["selected_level"]
        ):
            raise RuntimeError("cached probability/treatment contract is invalid")
        validate_observed_treatment(
            arm_result.get("treatment"),
            arm["treatment"],
            name=f"cached {arm_id}",
        )
        selected = probability[np.arange(target.size), target]
        accuracy = float(np.mean(np.argmax(probability, axis=1) == target))
        log_loss = float(
            -np.log(np.clip(selected, np.finfo(np.float32).tiny, 1.0)).mean()
        )
        for key, value in (("accuracy", accuracy), ("log_loss", log_loss)):
            if not math.isclose(
                float(arm_result.get(key, math.nan)), value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise RuntimeError(f"cached {key} is invalid")
        if any(
            not isinstance(arm_result.get(key), (int, float))
            or isinstance(arm_result.get(key), bool)
            or not math.isfinite(float(arm_result[key]))
            or float(arm_result[key]) < 0.0
            for key in ("fit_seconds", "predict_seconds")
        ):
            raise RuntimeError("cached arm timing is invalid")
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
            or prediction_contract.get("maximum_rows_per_call")
            != PREDICTION_CHUNK_ROWS
            or prediction_contract.get("chunk_count")
            != math.ceil(target.size / PREDICTION_CHUNK_ROWS)
        ):
            raise RuntimeError("cached prediction contract is invalid")
        schedules.append(arm_result["official_forward_schedule"])
    if any(schedule != schedules[0] for schedule in schedules[1:]):
        raise RuntimeError("cached paired forward schedules differ")
    return dict(task)


def _clean_transients(
    work: Path, results_root: Path, *, expected_ordinals: set[int]
) -> list[str]:
    cleaned: list[str] = []
    allowed_files = {
        "run-contract.json",
        "environment-contract.json",
        "checkpoint-contract.json",
        "attempts.json",
        "summary.json",
        "runtime.json",
    }
    for path in work.iterdir():
        if path.name == "datasets":
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError("resumable dataset root is unsafe")
        elif path.name in allowed_files:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("resumable work artifact is unsafe")
        elif path.name == "manifest.json":
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("resumable manifest artifact is unsafe")
            path.unlink()
            cleaned.append(path.name)
        elif (
            path.name.startswith(".")
            and ".tmp-" in path.name
            and path.is_file()
            and not path.is_symlink()
        ):
            path.unlink()
            cleaned.append(path.name)
        else:
            raise RuntimeError(f"unknown resumable work artifact: {path.name}")
    for path in results_root.iterdir():
        expected_name = (
            path.name.isdigit()
            and len(path.name) == 4
            and int(path.name) in expected_ordinals
        )
        if expected_name:
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError("cached dataset path is unsafe")
        elif (
            path.name.startswith(".")
            and ".tmp-" in path.name
            and path.is_dir()
            and not path.is_symlink()
        ):
            shutil.rmtree(path)
            cleaned.append(f"datasets/{path.name}")
        else:
            raise RuntimeError(f"unknown resumable dataset artifact: {path.name}")
    return cleaned


def _assert_complete_work_tree(work: Path, *, expected_ordinals: set[int]) -> None:
    expected_top = {
        "run-contract.json",
        "environment-contract.json",
        "checkpoint-contract.json",
        "attempts.json",
        "summary.json",
        "runtime.json",
        "datasets",
    }
    if {path.name for path in work.iterdir()} != expected_top:
        raise RuntimeError("completed shard top-level artifact union is not exact")
    for path in work.iterdir():
        if path.is_symlink() or (path.name == "datasets") != path.is_dir():
            raise RuntimeError("completed shard artifact type is unsafe")
    results_root = work / "datasets"
    expected_names = {f"{ordinal:04d}" for ordinal in expected_ordinals}
    if {path.name for path in results_root.iterdir()} != expected_names:
        raise RuntimeError("completed shard dataset union is not exact")
    if any(path.is_symlink() or not path.is_dir() for path in results_root.iterdir()):
        raise RuntimeError("completed shard dataset directory is unsafe")


def _run(args: argparse.Namespace) -> int:
    analysis_root = absolute_path(
        args.analysis_root, name="analysis_root", directory=True
    )
    model_root = absolute_path(args.model_root, name="model_root", directory=True)
    talent_root = absolute_path(args.talent_root, name="talent_root", directory=True)
    scratch_root = absolute_path(
        args.scratch_root, name="scratch_root", directory=True
    )
    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if (
        not slurm_tmpdir
        or not Path(slurm_tmpdir).is_absolute()
        or Path(slurm_tmpdir).resolve(strict=True) != scratch_root
    ):
        raise ValueError("scratch_root must be the exact job-local SLURM_TMPDIR")
    if _available_bytes(scratch_root) < DISK_OFFLOAD_MIN_FREE_BYTES:
        raise RuntimeError("TALENT offload scratch requires a 20 GiB reserve")
    analysis_sha = verify_clean_detached_git(
        analysis_root, expected_sha=args.expected_analysis_sha
    )
    model_sha = verify_clean_detached_git(
        model_root, expected_sha=args.expected_model_sha
    )
    config = load_private_run_config(Path(args.run_config))
    if config["config_sha256"] != args.expected_run_config_sha256:
        raise ValueError("run config differs from the submission-bound digest")
    if len(config["pairs"]) != 1:
        raise ValueError("one shard run must contain exactly one checkpoint pair")
    if config["pairs"][0]["training_source_commit"] != model_sha:
        raise ValueError(
            "checkpoint pair training_source_commit differs from model runtime SHA"
        )
    checkpoint_contract = validate_checkpoint_pairs(config)
    plan_path = absolute_path(args.shard_plan, name="shard_plan")
    plan_document, plan_file_sha256 = load_json_object_with_sha256(
        plan_path, name="TALENT shard plan"
    )
    if plan_file_sha256 != args.expected_shard_plan_sha256:
        raise ValueError("shard plan differs from the submission-bound digest")
    roster = frozen_discovery_roster(analysis_root)
    plan = validate_shard_plan(plan_document, expected_roster=roster)
    if plan.get("analysis_sha") != analysis_sha:
        raise ValueError("shard plan belongs to a different analysis commit")
    if args.canary:
        if args.shard_id != "canary":
            raise ValueError("canary execution requires shard_id=canary")
        records = canary_dataset_records(plan)
        execution_scope = "canary_largest_two_plus_madeline"
    else:
        shard = next(
            (item for item in plan["shards"] if item["shard_id"] == args.shard_id),
            None,
        )
        if shard is None:
            raise ValueError("shard_id is absent from the exact plan")
        records = [
            plan["datasets"][ordinal] for ordinal in shard["dataset_ordinals"]
        ]
        execution_scope = "full_planned_shard"
    expected_ordinals = {record["ordinal"] for record in records}
    environment = _runtime_environment(analysis_root, model_root)
    common_contract = {
        "schema_version": 1,
        "kind": SHARD_KIND,
        "formal_eligible": False,
        "evidence_scope": "exploratory_discovery_only",
        "analysis_sha": analysis_sha,
        "model_runtime_sha": model_sha,
        "private_run_config_sha256": config["config_sha256"],
        "portable_pair_contract": config["portable"],
        "portable_pair_contract_sha256": config["portable_sha256"],
        "checkpoint_contract": checkpoint_contract,
        "shard_plan_sha256": plan_file_sha256,
        "shard_plan_document_sha256": plan_document["sha256"],
        "roster_sha256": plan["roster_sha256"],
        "fit_split": "train",
        "evaluation_split": "val",
        "test_split_policy": "hashed_for_integrity_only_not_used",
        "row_sampling": "none_full_split_original_order",
        "prediction_chunk_rows": PREDICTION_CHUNK_ROWS,
        "estimator_options": dict(ESTIMATOR_OPTIONS),
        "precision_contract": {
            "model_parameters": "float32",
            "inference": "float32",
            "amp": False,
        },
        "oom_fallback_contract": {
            "sequence": list(OOM_FALLBACK_SEQUENCE),
            "batch_size_is_minimum": True,
            "retry_scope": "discard_entire_dataset_pair_and_restart_both_arms",
            "non_oom_retry": False,
        },
        "disk_offload_scratch_contract": dict(DISK_OFFLOAD_SCRATCH_CONTRACT),
        "environment_contract_sha256": json_document_sha256(environment),
    }
    job_contract = {
        "shard_id": args.shard_id,
        "execution_scope": execution_scope,
        "dataset_ordinals": [record["ordinal"] for record in records],
        "dataset_names": [record["name"] for record in records],
    }
    contract = {
        **common_contract,
        "common_run_contract_sha256": json_document_sha256(common_contract),
        "job_contract": job_contract,
        "job_contract_sha256": json_document_sha256(job_contract),
    }
    run_contract_sha256 = json_document_sha256(contract)
    output_argument = Path(args.output_dir)
    if not output_argument.is_absolute():
        raise ValueError("output_dir must be absolute")
    output = output_argument.parent.resolve(strict=True) / output_argument.name
    protected = [
        analysis_root,
        model_root,
        talent_root,
        scratch_root,
        config["path"],
        plan_path,
        *(
            arm["checkpoint"]
            for pair in config["pairs"]
            for arm in pair["arms"]
        ),
        *(
            receipt["path"]
            for pair in config["pairs"]
            for receipt in pair["lineage_receipts"]
        ),
    ]
    require_disjoint_output(output, protected, name="TALENT shard output")
    require_disjoint_output(
        output.with_name(f".{output.name}.work"),
        protected,
        name="TALENT resumable work output",
    )
    if output.exists() or output.is_symlink():
        raise ValueError("output_dir must be absent")
    work = output.with_name(f".{output.name}.work")
    if work.exists():
        if work.is_symlink() or not work.is_dir():
            raise ValueError("resumable work directory is unsafe")
        for required_name in (
            "run-contract.json",
            "environment-contract.json",
            "checkpoint-contract.json",
        ):
            required_path = work / required_name
            if required_path.is_symlink() or not required_path.is_file():
                raise ValueError("resumable work lacks a safe immutable contract")
        if load_json_object(work / "run-contract.json", name="run contract") != contract:
            raise ValueError("resumable work contract changed")
        if load_json_object(
            work / "environment-contract.json", name="environment contract"
        ) != environment:
            raise ValueError("resumable environment contract changed")
        if load_json_object(
            work / "checkpoint-contract.json", name="checkpoint contract"
        ) != checkpoint_contract:
            raise ValueError("resumable checkpoint contract changed")
    else:
        work.mkdir()
        fsync_directory(work.parent)
        atomic_json(work / "run-contract.json", contract)
        atomic_json(work / "environment-contract.json", environment)
        atomic_json(work / "checkpoint-contract.json", checkpoint_contract)
    results_root = work / "datasets"
    if results_root.exists() or results_root.is_symlink():
        if results_root.is_symlink() or not results_root.is_dir():
            raise ValueError("resumable dataset root is unsafe")
    else:
        results_root.mkdir()
        fsync_directory(work)
    cleaned = _clean_transients(
        work, results_root, expected_ordinals=expected_ordinals
    )
    attempts_path = work / "attempts.json"
    attempts = (
        json.loads(attempts_path.read_text(encoding="utf-8"))
        if attempts_path.exists()
        else []
    )
    if not isinstance(attempts, list):
        raise RuntimeError("attempt history is invalid")
    attempt = {
        "attempt": len(attempts) + 1,
        "started_at_unix_seconds": time.time(),
        "job_role": "canary" if args.canary else f"shard-{args.shard_id}",
        "status": "started",
        "cleaned_transients": cleaned,
    }
    attempts.append(attempt)
    atomic_json(attempts_path, attempts)
    started = time.monotonic()
    stage_directory: Path | None = None
    try:
        stage_directory, staged = _stage_checkpoints(config)
        result_records = []
        for record in records:
            destination = results_root / f"{record['ordinal']:04d}"
            if destination.exists():
                result = _cached_dataset(
                    destination,
                    expected=record,
                    run_contract_sha256=run_contract_sha256,
                    config=config,
                    model_sha=model_sha,
                    talent_root=talent_root,
                )
            else:
                result = _evaluate_dataset(
                    dataset_record=record,
                    talent_root=talent_root,
                    config=config,
                    staged=staged,
                    model_root=model_root,
                    model_sha=model_sha,
                    run_contract_sha256=run_contract_sha256,
                    destination=destination,
                    scratch_root=scratch_root,
                )
            result_records.append(result)
            print(
                json.dumps(
                    {
                        "shard_id": args.shard_id,
                        "completed": len(result_records),
                        "total": len(records),
                        "dataset": record["name"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        verify_clean_detached_git(analysis_root, expected_sha=analysis_sha)
        verify_clean_detached_git(model_root, expected_sha=model_sha)
        if sha256_file(config["path"]) != config["config_sha256"]:
            raise RuntimeError("private run config changed during evaluation")
        if sha256_file(plan_path) != contract["shard_plan_sha256"]:
            raise RuntimeError("shard plan changed during evaluation")
        for pair in config["pairs"]:
            for arm in pair["arms"]:
                if sha256_file(arm["checkpoint"]) != arm["checkpoint_sha256"]:
                    raise RuntimeError("checkpoint changed during evaluation")
            for receipt in pair["lineage_receipts"]:
                if sha256_file(receipt["path"]) != receipt["sha256"]:
                    raise RuntimeError("lineage receipt changed during evaluation")
        if _runtime_environment(analysis_root, model_root) != environment:
            raise RuntimeError("runtime environment changed during evaluation")
        summary = {
            **contract,
            "run_contract_sha256": run_contract_sha256,
            "dataset_count": len(result_records),
            "prediction_rows": sum(record["n_evaluation"] for record in result_records),
            "datasets": result_records,
        }
        atomic_json(work / "summary.json", summary)
        duration = time.monotonic() - started
        attempt.update(
            {
                "status": "completed",
                "duration_seconds": duration,
                "completed_at_unix_seconds": time.time(),
                "completed_datasets": len(result_records),
            }
        )
        atomic_json(attempts_path, attempts)
        atomic_json(
            work / "runtime.json",
            {
                "schema_version": 1,
                "attempt_count": len(attempts),
                "duration_seconds_final_attempt": duration,
            },
        )
        _assert_complete_work_tree(work, expected_ordinals=expected_ordinals)
        atomic_json(work / "manifest.json", directory_manifest(work, kind=SHARD_KIND))
        os.replace(work, output)
        fsync_directory(output.parent)
    except BaseException as error:
        attempt.update(
            {
                "status": "failed",
                "duration_seconds": time.monotonic() - started,
                "failed_at_unix_seconds": time.time(),
                "error_type": type(error).__name__,
            }
        )
        if work.is_dir() and not work.is_symlink():
            atomic_json(attempts_path, attempts)
        raise
    finally:
        if stage_directory is not None:
            shutil.rmtree(stage_directory, ignore_errors=True)
    print(
        json.dumps(
            {
                "status": "completed",
                "job_role": "canary" if args.canary else f"shard-{args.shard_id}",
                "dataset_count": len(records),
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
            Path(args.model_root),
            Path(args.talent_root),
            Path(args.run_config),
            Path(args.shard_plan),
        ],
        name="TALENT shard output",
    )
    lock_path = output_parent / f".{output.name}.lock"
    with _RunLock(lock_path):
        return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
