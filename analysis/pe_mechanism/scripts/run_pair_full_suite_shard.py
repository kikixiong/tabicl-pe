#!/usr/bin/env python3
"""Run one resumable shard of a strict two-arm TabArena/BeyondArena lite study."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import importlib
import importlib.metadata
import json
import math
from numbers import Real
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping


_NATIVE_LITE_COLUMNS = frozenset(
    {
        "dataset_name",
        "tabarena_task_name",
        "task_id_str",
        "problem_type",
        "eval_metric",
        "num_instances",
        "num_features",
        "num_cols_after_preprocessing",
        "num_text_cols",
        "task_type",
        "repeat",
        "fold",
        "split_index",
        "num_instances_train",
        "num_instances_test",
    }
)


def _metadata_text(value: Any, field: str, *, dataset: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"native metadata {field} is invalid for {dataset}")
    return value


def _metadata_number(value: Any, field: str, *, dataset: str) -> Real:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"native metadata {field} is not numeric for {dataset}")
    if not math.isfinite(float(value)):
        raise ValueError(f"native metadata {field} is not finite for {dataset}")
    return value


def _metadata_integer(
    value: Any, field: str, *, dataset: str, minimum: int
) -> int:
    number = _metadata_number(value, field, dataset=dataset)
    integer = int(number)
    if number != integer or integer < minimum:
        raise ValueError(f"native metadata {field} is invalid for {dataset}")
    return integer


def _metadata_positive_number(value: Any, field: str, *, dataset: str) -> Real:
    number = _metadata_number(value, field, dataset=dataset)
    if number <= 0:
        raise ValueError(f"native metadata {field} must be positive for {dataset}")
    return number


def _metadata_missing(value: Any) -> bool:
    return value is None or (
        isinstance(value, Real) and math.isnan(float(value))
    )


def _native_lite_metadata(
    arena: Any, *, names: tuple[str, ...], suite_id: str
) -> dict[str, dict[str, Any]]:
    """Read exact r0f0 rows from the pinned native task-metadata API.

    ``arena.task_metadata`` is intentionally not accepted here: in pinned TabArena
    c987 it is a lossy legacy bridge whose train/test sizes are split means.
    """
    collection = getattr(arena, "task_metadata_collection", None)
    if collection is None:
        raise TypeError("benchmark context lacks native task_metadata_collection")
    try:
        lite = collection.subset_tasks(
            dataset_names=list(names), split_indices="lite"
        )
        frame = lite.to_dataframe()
    except AttributeError as error:
        raise TypeError("benchmark native task metadata API is unavailable") from error
    columns = set(getattr(frame, "columns", ()))
    missing_columns = sorted(_NATIVE_LITE_COLUMNS - columns)
    if missing_columns:
        raise ValueError(
            f"native lite metadata is missing required columns: {missing_columns}"
        )
    rows = frame.to_dict(orient="records")
    expected_names = set(names)
    observed_names = {row.get("dataset_name") for row in rows}
    if observed_names != expected_names:
        raise ValueError("native lite metadata does not exactly cover the frozen roster")
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = _metadata_text(row["dataset_name"], "dataset_name", dataset="<roster>")
        if name in by_name:
            raise ValueError(f"native lite metadata is not one-to-one for {name}")
        repeat = _metadata_integer(row["repeat"], "repeat", dataset=name, minimum=0)
        fold = _metadata_integer(row["fold"], "fold", dataset=name, minimum=0)
        split_index = _metadata_text(row["split_index"], "split_index", dataset=name)
        if (repeat, fold, split_index) != (0, 0, "r0f0"):
            raise ValueError(f"native lite metadata is not exact r0f0 for {name}")

        internal = _metadata_text(
            row["tabarena_task_name"], "tabarena_task_name", dataset=name
        )
        problem = _metadata_text(row["problem_type"], "problem_type", dataset=name)
        if problem not in {"binary", "multiclass"}:
            raise ValueError(f"roster task is not classification: {name}")
        metric = _metadata_text(row["eval_metric"], "eval_metric", dataset=name)
        expected_metric = "roc_auc" if problem == "binary" else "log_loss"
        if metric != expected_metric:
            raise ValueError(f"native metadata eval_metric is invalid for {name}")
        task_type = _metadata_text(row["task_type"], "task_type", dataset=name)
        if task_type not in {"random", "temporal", "grouped"}:
            raise ValueError(f"native metadata task_type is invalid for {name}")

        num_instances = _metadata_integer(
            row["num_instances"], "num_instances", dataset=name, minimum=1
        )
        train_raw = _metadata_positive_number(
            row["num_instances_train"], "num_instances_train", dataset=name
        )
        test_raw = _metadata_positive_number(
            row["num_instances_test"], "num_instances_test", dataset=name
        )
        if (
            train_raw > num_instances
            or test_raw > num_instances
            or train_raw + test_raw > num_instances
        ):
            raise ValueError(f"native lite split sizes exceed dataset size for {name}")

        if suite_id == "beyondarena":
            dimensions = _metadata_integer(
                row["num_cols_after_preprocessing"],
                "num_cols_after_preprocessing",
                dataset=name,
                minimum=1,
            )
            if (
                _metadata_integer(
                    row["num_text_cols"], "num_text_cols", dataset=name, minimum=0
                )
                != 0
            ):
                raise ValueError(f"BeyondArena task contains text features: {name}")
            train = _metadata_integer(
                train_raw, "num_instances_train", dataset=name, minimum=1
            )
            test = _metadata_integer(
                test_raw, "num_instances_test", dataset=name, minimum=1
            )
            split_regime = "iid" if task_type == "random" else task_type
        elif suite_id == "tabarena-v0.1":
            if task_type != "random":
                raise ValueError(f"TabArena-v0.1 task is not an IID split: {name}")
            if not _metadata_missing(row["num_cols_after_preprocessing"]):
                raise ValueError(
                    "TabArena-v0.1 unexpectedly supplies post-preprocessing dimensions "
                    f"for {name}"
                )
            dimensions = _metadata_integer(
                row["num_features"], "num_features", dataset=name, minimum=1
            )
            # The pinned v0.1 source stores nominal 2/3 and 1/3 sizes as floats;
            # retain them exactly here; the v2 shard-plan validator applies floor.
            train = train_raw
            test = test_raw
            split_regime = "iid"
        else:
            raise ValueError(f"unsupported benchmark suite: {suite_id}")

        by_name[name] = {
            "dataset_name": name,
            "benchmark_dataset_id": internal,
            "problem_type": problem,
            "metric": metric,
            "num_instances": num_instances,
            "num_cols_after_preprocessing": dimensions,
            "num_instances_train": train,
            "num_instances_test": test,
            "split_regime": split_regime,
        }

    try:
        tids = lite.dataset_to_tid()
    except AttributeError as error:
        raise TypeError("benchmark native dataset_to_tid API is unavailable") from error
    if not isinstance(tids, Mapping):
        raise TypeError("benchmark native dataset_to_tid result is not a mapping")
    internal_ids: set[str] = set()
    for name in names:
        record = by_name[name]
        internal = record["benchmark_dataset_id"]
        if internal in internal_ids:
            raise ValueError(f"native benchmark dataset id is duplicated: {internal}")
        internal_ids.add(internal)
        if internal not in tids:
            raise ValueError(f"native task id is missing for {name}")
        record["task_id"] = _metadata_integer(
            tids[internal], "task_id", dataset=name, minimum=1
        )
    return by_name


def _reject_symlink_components(path: Path, *, label: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{label} must not traverse symlinks: {current}")


def _ensure_real_directory(path: Path, *, label: str) -> Path:
    _reject_symlink_components(path, label=label)
    path.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path, label=label)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a real directory")
    return path


@contextmanager
def _disk_offload_attempt(
    scratch_root: Path, *, task_index: int, fallback: Mapping[str, Any]
):
    if fallback["offload_mode"] != "disk":
        yield None
        return
    parent = _ensure_real_directory(scratch_root, label="job scratch root").resolve(
        strict=True
    )
    attempt = Path(
        tempfile.mkdtemp(prefix=f"disk-offload-{task_index:04d}-", dir=parent)
    )
    attempt.chmod(0o700)
    _reject_symlink_components(attempt, label="disk offload attempt")
    resolved = attempt.resolve(strict=True)
    if attempt.is_symlink() or not attempt.is_dir() or resolved.parent != parent:
        raise RuntimeError("disk offload attempt escaped job-local scratch")
    try:
        yield resolved
    finally:
        _reject_symlink_components(attempt, label="disk offload cleanup target")
        if (
            attempt.is_symlink()
            or not attempt.is_dir()
            or attempt.resolve(strict=True) != resolved
        ):
            raise RuntimeError("refusing unsafe disk offload cleanup")
        shutil.rmtree(attempt)


def _absolute_directory(value: str, *, label: str, must_exist: bool = True) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    _reject_symlink_components(path, label=label)
    if must_exist:
        resolved = path.resolve(strict=True)
        if path.is_symlink() or not resolved.is_dir():
            raise ValueError(f"{label} must be a real directory")
        return resolved
    return path


def _git_head(root: Path, *, expected: str, label: str) -> str:
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise ValueError(f"{label} source checkout is not clean: {root}")
    symbolic = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "-q", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if symbolic.returncode == 0:
        raise ValueError(f"{label} source checkout must be detached")
    if symbolic.returncode != 1:
        raise RuntimeError(f"could not verify detached {label} checkout")
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_environment() -> dict[str, Any]:
    import numpy as np
    import sklearn
    import torch

    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    if len(gpu) != 1:
        raise ValueError("pair worker must see exactly one allocated GPU")
    package_versions: dict[str, str] = {}
    for distribution in ("tabicl", "tabarena"):
        try:
            package_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            package_versions[distribution] = "source-checkout"
    return {
        "schema_version": 1,
        "kind": "two_arm_pair_runtime_environment",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "packages": package_versions,
        "gpu_name_driver": gpu[0],
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "official_split_contract": {"fold": 0, "repeat": 0, "split_index": 0},
        "data_content_binding": (
            "each private prediction NPZ binds encoded targets plus train/test feature content"
        ),
    }


def _stage_pair_checkpoints(pair: Any, *, scratch_root: Path) -> tuple[Any, Path]:
    scratch_root = _ensure_real_directory(scratch_root, label="job scratch root")
    stage = Path(tempfile.mkdtemp(prefix="pair-checkpoints-", dir=scratch_root))
    _reject_symlink_components(stage, label="checkpoint stage")
    staged_arms = []
    try:
        for arm in pair.arms:
            destination = stage / f"{arm.arm_id}.ckpt"
            source_before = arm.checkpoint_path.stat()
            with arm.checkpoint_path.open("rb") as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output, length=8 << 20)
                output.flush()
                os.fsync(output.fileno())
            source_after = arm.checkpoint_path.stat()
            if (
                source_before.st_ino != source_after.st_ino
                or source_before.st_size != source_after.st_size
                or source_before.st_mtime_ns != source_after.st_mtime_ns
            ):
                raise RuntimeError("checkpoint changed while creating immutable job staging")
            if destination.stat().st_size != arm.checkpoint_size_bytes or _sha256(
                destination
            ) != arm.checkpoint_sha256:
                raise RuntimeError("job-local checkpoint staging digest mismatch")
            destination.chmod(0o400)
            staged_arms.append(replace(arm, checkpoint_path=destination))
        stage.chmod(0o500)
        return replace(pair, arms=tuple(staged_arms)), stage
    except BaseException:
        stage.chmod(0o700)
        shutil.rmtree(stage)
        raise


def _verify_bound_inputs(pair: Any, roster: Any, plan: Any) -> None:
    for path, expected, label in (
        (pair.path, pair.sha256, "pair manifest"),
        (roster.path, roster.sha256, "roster"),
        (plan.path, plan.sha256, "shard plan"),
    ):
        if _sha256(path) != expected:
            raise RuntimeError(f"{label} changed during evaluation")
    for arm in pair.arms:
        if (
            arm.checkpoint_path.stat().st_size != arm.checkpoint_size_bytes
            or _sha256(arm.checkpoint_path) != arm.checkpoint_sha256
        ):
            raise RuntimeError(f"checkpoint changed during evaluation: {arm.arm_id}")


def _capture_pair_prediction(
    probabilities: Any,
    *,
    train_features: Any,
    train_targets: Any,
    test_features: Any,
    test_targets: Any,
    output: Path,
) -> None:
    """Use the formal runner's canonical row/target/class evidence for one NPZ."""
    import numpy as np
    import pandas as pd
    from pe_mechanism.tabarena_formal_runner import (
        _canonical_scalar,
        _class_label_evidence,
        _feature_table_fingerprint,
        _row_fingerprints,
        _table_content_fingerprint,
        _test_target_fingerprints,
    )

    if not isinstance(probabilities, pd.DataFrame):
        raise TypeError("pair probabilities must be a pandas DataFrame")
    source = probabilities.to_numpy(copy=False)
    if source.ndim != 2 or not np.issubdtype(source.dtype, np.floating):
        raise ValueError("pair probabilities must be a floating matrix")
    values = np.ascontiguousarray(source, dtype=np.dtype("<f4"))
    if values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("pair probability matrix is empty")
    if not np.isfinite(values).all():
        raise ValueError("pair probabilities must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("pair probabilities must lie in [0,1]")
    if not np.allclose(
        values.sum(axis=1, dtype=np.float64), 1.0, rtol=1e-6, atol=1e-6
    ):
        raise ValueError("pair probability rows must sum to one")
    class_bytes, class_labels = _class_label_evidence(list(probabilities.columns))
    rows = _row_fingerprints(list(probabilities.index))
    targets = _test_target_fingerprints(
        test_targets,
        expected_index=probabilities.index,
    )
    if rows.shape[0] != values.shape[0] or len(class_labels) != values.shape[1]:
        raise RuntimeError("pair probability rows or classes are misaligned")
    canonical_columns = [
        json.dumps(
            _canonical_scalar(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for value in probabilities.columns
    ]
    class_to_index = {value: index for index, value in enumerate(canonical_columns)}
    encoded_targets = np.asarray(
        [
            class_to_index[
                json.dumps(
                    _canonical_scalar(value),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            ]
            for value in test_targets
        ],
        dtype=np.dtype("<i8"),
    )
    if encoded_targets.shape != (values.shape[0],):
        raise RuntimeError("encoded test targets are misaligned")
    train_digest = bytes.fromhex(
        _table_content_fingerprint(train_features, train_targets)
    )
    test_digest = bytes.fromhex(
        _feature_table_fingerprint(test_features, domain="test")
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError("pair prediction destination must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as handle:
            np.savez_compressed(
                handle,
                probabilities=values,
                encoded_targets=encoded_targets,
                row_fingerprints=rows,
                test_target_fingerprints=targets,
                class_labels_utf8=np.frombuffer(class_bytes, dtype=np.uint8),
                train_content_sha256=np.frombuffer(train_digest, dtype=np.uint8),
                test_content_sha256=np.frombuffer(test_digest, dtype=np.uint8),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _make_prediction_runner(external_runner: type) -> type:
    class PairPredictionRunner(external_runner):
        def __init__(self, *, pair_prediction_path: str, **kwargs: Any) -> None:
            path = Path(pair_prediction_path)
            if not path.is_absolute():
                raise ValueError("pair prediction path must be absolute")
            self.pair_prediction_path = path
            super().__init__(**kwargs)

        def post_evaluate(self, out: dict[str, Any]) -> dict[str, Any]:
            result = super().post_evaluate(out)
            probabilities = result.get("probabilities")
            if probabilities is None:
                raise RuntimeError("classification result lacks probabilities")
            train_features, train_targets, test_features, test_targets = (
                self._train_test_split()
            )
            _capture_pair_prediction(
                probabilities,
                train_features=train_features,
                train_targets=train_targets,
                test_features=test_features,
                test_targets=test_targets,
                output=self.pair_prediction_path,
            )
            return result

    PairPredictionRunner.__name__ = "PairPredictionRunner"
    PairPredictionRunner.__qualname__ = "PairPredictionRunner"
    return PairPredictionRunner


def _normalize_classification_target(y: Any) -> Any:
    import numpy as np
    import pandas as pd
    from sklearn.utils.multiclass import type_of_target

    values = np.asarray(y)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError("classification target must be one-dimensional")
    index = getattr(y, "index", None)
    normalized = pd.Series(values.tolist(), index=index, name=getattr(y, "name", None))
    if type_of_target(normalized) not in {"binary", "multiclass"}:
        raise ValueError("classification target is not discrete")
    return normalized


def _make_beyond_model(external_system_model: type, base_factory: Any) -> type:
    base = base_factory(external_system_model)

    class BeyondPairSystem(base):
        def _fit_system(self, X: Any, y: Any, **kwargs: Any) -> Any:
            return super()._fit_system(
                X,
                _normalize_classification_target(y),
                **kwargs,
            )

    BeyondPairSystem.__name__ = "BeyondPairSystem"
    BeyondPairSystem.__qualname__ = "BeyondPairSystem"
    return BeyondPairSystem


class PairTaskRuntime:
    def __init__(
        self,
        *,
        runtime: Mapping[str, Any],
        pair: Any,
        roster: Any,
        cache_root: Path,
        run_root: Path,
        scratch_root: Path,
        make_system_model: Any,
        normalize_results: Any,
        load_cached_results: Any,
    ) -> None:
        self.runtime = runtime
        self.pair = pair
        self.roster = roster
        self.run_root = run_root
        self.scratch_root = _ensure_real_directory(
            scratch_root, label="job scratch root"
        )
        self.make_system_model = make_system_model
        self.normalize_results = normalize_results
        self.load_cached_results = load_cached_results
        cache = runtime["CacheConfig"].from_root(cache_root)
        context_cls = (
            runtime["BeyondArenaContext"]
            if roster.suite_id == "beyondarena"
            else runtime["TabArenaContext"]
        )
        self.arena = context_cls(methods=[], backend="native", cache_config=cache)

    def _lite_metadata(self) -> dict[str, dict[str, Any]]:
        cached = getattr(self, "_lite_metadata_cache", None)
        if cached is None:
            cached = _native_lite_metadata(
                self.arena,
                names=tuple(self.roster.names),
                suite_id=self.roster.suite_id,
            )
            self._lite_metadata_cache = cached
        return cached

    def plan_records(self) -> dict[str, dict[str, Any]]:
        metadata = self._lite_metadata()
        return {
            name: {
                field: metadata[name][field]
                for field in (
                    "num_instances",
                    "num_cols_after_preprocessing",
                    "num_instances_train",
                    "num_instances_test",
                    "split_regime",
                )
            }
            for name in self.roster.names
        }

    def _task_metadata(self, name: str) -> tuple[str, dict[str, Any]]:
        metadata = self._lite_metadata()
        if name not in metadata:
            raise ValueError(f"task is outside the frozen roster: {name}")
        row = metadata[name]
        internal = row["benchmark_dataset_id"]
        expected = {
            "task_id": row["task_id"],
            "problem_type": row["problem_type"],
            "metric": row["metric"],
        }
        task = {
            "dataset_name": name,
            "benchmark_dataset_id": internal,
            "task_id": expected["task_id"],
            "problem_type": expected["problem_type"],
            "metric": expected["metric"],
            "split_regime": row["split_regime"],
            "fold": 0,
            "repeat": 0,
            "split_index": 0,
        }
        return internal, {"expected": expected, "task": task}

    def __call__(
        self, task: Any, staging: Path, fallback: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
        with _disk_offload_attempt(
            self.scratch_root, task_index=task.index, fallback=fallback
        ) as disk_offload_dir:
            return self._execute(
                task,
                staging,
                fallback,
                disk_offload_dir=disk_offload_dir,
            )

    def _execute(
        self,
        task: Any,
        staging: Path,
        fallback: Mapping[str, Any],
        *,
        disk_offload_dir: Path | None,
    ) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
        internal, evidence = self._task_metadata(task.name)
        model_cls = self.make_system_model(self.runtime["ExternalSystemModel"])
        if self.roster.suite_id == "beyondarena":
            model_cls = _make_beyond_model(
                self.runtime["ExternalSystemModel"], self.make_system_model
            )
            bundle_cls = self.runtime["BeyondArenaExperimentBundle"]
        else:
            bundle_cls = self.runtime["TabArenaV0pt1ExperimentBundle"]
        prediction_runner = _make_prediction_runner(self.runtime["OOFExperimentRunner"])
        experiments = []
        framework_to_arm: dict[str, str] = {}
        classifier_options = dict(self.pair.inference["classifier_options"])
        classifier_options["batch_size"] = fallback["batch_size"]
        classifier_options["offload_mode"] = fallback["offload_mode"]
        if fallback["offload_mode"] == "disk":
            if disk_offload_dir is None:
                raise RuntimeError("disk fallback lacks controlled job-local scratch")
            classifier_options["disk_offload_dir"] = str(disk_offload_dir)
        elif disk_offload_dir is not None:
            raise RuntimeError("non-disk fallback unexpectedly received disk scratch")
        for arm in self.pair.arms:
            generator = self.runtime["SystemConfigGenerator"](
                model_cls=model_cls,
                name=f"PairEval_{self.pair.sha256[:12]}_{arm.arm_id}",
                manual_configs=[
                    {
                        "checkpoint": str(arm.checkpoint_path),
                        "arm": arm.arm_id,
                        "device": self.pair.inference["device"],
                        "n_estimators": self.pair.inference["n_estimators"],
                        "seed": self.pair.inference["seed"],
                        "classifier_options": dict(classifier_options),
                    }
                ],
            )
            built = bundle_cls(models=[(generator, 0)], system_experiments=True).build_experiments()
            if len(built) != 1 or built[0].name in framework_to_arm:
                raise RuntimeError("benchmark did not build one unique experiment per arm")
            built[0].experiment_cls = prediction_runner
            built[0].experiment_kwargs = {
                **built[0].experiment_kwargs,
                "pair_prediction_path": str(
                    (staging / arm.arm_id / "predictions.npz").resolve()
                ),
            }
            framework_to_arm[built[0].name] = arm.arm_id
            experiments.extend(built)

        jobs = self.arena.build_jobs(
            experiments,
            subset=["lite"],
            dataset_names=[task.name],
            problem_types=["binary", "multiclass"],
        )
        expected_keys = {
            (framework, internal, 0, 0) for framework in framework_to_arm
        }
        observed_keys = {
            (job.experiment.name, job.task.dataset, job.task.fold, job.task.repeat)
            for job in jobs
        }
        if len(jobs) != 2 or observed_keys != expected_keys:
            raise RuntimeError("benchmark job matrix differs from one paired lite task")

        scratch_parent = self.run_root / ".runtime"
        _ensure_real_directory(scratch_parent, label="benchmark runtime directory")
        with tempfile.TemporaryDirectory(
            prefix=f"task-{task.index:04d}-", dir=scratch_parent
        ) as temporary:
            results_root = Path(temporary) / "results"
            try:
                returned = self.arena.run_jobs(
                    jobs,
                    expname=results_root,
                    register=False,
                    cache_mode="ignore",
                    debug_mode=True,
                    raise_on_failure=True,
                )
            except Exception as error:
                import torch
                from pe_mechanism.full_suite_pair import PairTaskOOM

                current: BaseException | None = error
                seen: set[int] = set()
                cuda_oom = False
                while current is not None and id(current) not in seen:
                    seen.add(id(current))
                    if isinstance(current, torch.cuda.OutOfMemoryError):
                        cuda_oom = True
                        break
                    current = current.__cause__ or current.__context__
                if not cuda_oom:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise PairTaskOOM(
                    f"paired task {task.name} exhausted CUDA memory at "
                    f"fallback level {fallback['level']}"
                ) from error
            normalize_kwargs = {
                "framework_to_arm": framework_to_arm,
                "roster": (internal,),
                "expected_count": 2,
                "expected_tasks": {internal: evidence["expected"]},
                "arm_order": self.pair.arm_order,
            }
            normalized = self.normalize_results(returned, **normalize_kwargs)
            cached = self.normalize_results(
                self.load_cached_results(results_root), **normalize_kwargs
            )
            if normalized != cached:
                raise RuntimeError("returned and cached benchmark results differ")
        rows = {row["arm"]: row for row in normalized}
        results = {
            arm: {
                key: rows[arm][key]
                for key in ("metric_error", "time_train_s", "time_infer_s")
            }
            for arm in self.pair.arm_order
        }
        return evidence["task"], results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--tabarena-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--roster", required=True)
    parser.add_argument("--shard-plan", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--expected-analysis-sha", required=True)
    parser.add_argument("--expected-tabarena-sha", required=True)
    parser.add_argument("--expected-python-contract-document-sha256", required=True)
    parser.add_argument("--expected-python-contract-file-sha256", required=True)
    parser.add_argument(
        "--expected-suite", required=True, choices=("beyondarena", "tabarena-v0.1")
    )
    parser.add_argument("--phase", required=True, choices=("canary", "full"))
    parser.add_argument("--shard-index", type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    python_environment_contract = {
        "document_sha256": args.expected_python_contract_document_sha256,
        "file_sha256": args.expected_python_contract_file_sha256,
    }
    if any(
        len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
        for value in python_environment_contract.values()
    ):
        raise ValueError("Python environment contract digests are malformed")
    analysis_root = _absolute_directory(args.analysis_root, label="analysis_root")
    model_root = _absolute_directory(args.model_root, label="model_root")
    tabarena_root = _absolute_directory(args.tabarena_root, label="tabarena_root")
    cache_root = _absolute_directory(args.cache_root, label="cache_root")
    run_root = _absolute_directory(args.run_root, label="run_root", must_exist=False)
    sources = (analysis_root, model_root, tabarena_root)
    for output in (run_root, cache_root):
        for source in sources:
            if _is_within(output, source) or _is_within(source, output):
                raise ValueError(
                    "run/cache roots and source checkouts must be bidirectionally disjoint"
                )
    if _is_within(run_root, cache_root) or _is_within(cache_root, run_root):
        raise ValueError("run_root and cache_root must be disjoint")

    package_root = analysis_root / "analysis" / "pe_mechanism" / "src"
    sys.path.insert(0, str(package_root))
    from pe_mechanism.full_suite_pair import (
        code_provenance,
        load_pair_manifest,
        load_roster,
        load_shard_plan,
        run_canary,
        run_shard,
        validate_plan_metadata,
    )
    from pe_mechanism.tabarena_evaluation import (
        _import_runtime,
        _load_cached_results,
        _make_system_model,
        _normalize_results,
    )

    pair = load_pair_manifest(Path(args.pair_manifest), verify_checkpoints=True)
    original_pair = pair
    roster = load_roster(Path(args.roster))
    if roster.suite_id != args.expected_suite:
        raise ValueError("roster suite differs from the scheduled GPU wrapper")
    plan = load_shard_plan(Path(args.shard_plan), roster=roster)
    if roster.source_commit != args.expected_tabarena_sha:
        raise ValueError("roster source commit differs from expected TabArena SHA")
    analysis_sha = _git_head(
        analysis_root, expected=args.expected_analysis_sha, label="analysis"
    )
    model_sha = _git_head(
        model_root, expected=pair.model_source_sha, label="model"
    )
    tabarena_sha = _git_head(
        tabarena_root, expected=args.expected_tabarena_sha, label="TabArena"
    )

    environment = _runtime_environment()
    scratch_root = _absolute_directory(
        os.environ.get("TMPDIR", str(run_root / ".slurm-runtime")),
        label="job scratch root",
        must_exist=False,
    )
    _ensure_real_directory(scratch_root, label="job scratch root")
    pair, checkpoint_stage = _stage_pair_checkpoints(
        original_pair, scratch_root=scratch_root
    )
    runtime = dict(_import_runtime(tabarena_root, model_root))
    experiment_module = importlib.import_module("tabarena.benchmark.experiment")
    context_module = importlib.import_module("tabarena.contexts")
    runner_module = importlib.import_module(
        "tabarena.benchmark.experiment.experiment_runner"
    )
    runtime.update(
        {
            "BeyondArenaExperimentBundle": experiment_module.BeyondArenaExperimentBundle,
            "BeyondArenaContext": context_module.BeyondArenaContext,
            "OOFExperimentRunner": runner_module.OOFExperimentRunner,
        }
    )
    provenance = code_provenance(
        analysis_sha=analysis_sha,
        model_sha=model_sha,
        tabarena_sha=tabarena_sha,
    )
    try:
        executor = PairTaskRuntime(
            runtime=runtime,
            pair=pair,
            roster=roster,
            cache_root=cache_root,
            run_root=run_root,
            scratch_root=scratch_root,
            make_system_model=_make_system_model,
            normalize_results=_normalize_results,
            load_cached_results=_load_cached_results,
        )
        validate_plan_metadata(plan, roster=roster, records=executor.plan_records())
        if args.phase == "canary":
            if args.shard_index is not None:
                raise ValueError("canary phase does not accept shard_index")
            summary = run_canary(
                run_root,
                pair=pair,
                roster=roster,
                plan=plan,
                provenance=provenance,
                executor=executor,
                runtime_environment=environment,
                python_environment_contract=python_environment_contract,
            )
        else:
            if args.shard_index is None:
                raise ValueError("full phase requires shard_index")
            summary = run_shard(
                run_root,
                pair=pair,
                roster=roster,
                plan=plan,
                provenance=provenance,
                shard_index=args.shard_index,
                executor=executor,
                runtime_environment=environment,
                python_environment_contract=python_environment_contract,
            )
        _verify_bound_inputs(original_pair, roster, plan)
        _verify_bound_inputs(pair, roster, plan)
        _git_head(analysis_root, expected=analysis_sha, label="analysis")
        _git_head(model_root, expected=model_sha, label="model")
        _git_head(tabarena_root, expected=tabarena_sha, label="TabArena")
    finally:
        _reject_symlink_components(checkpoint_stage, label="checkpoint stage cleanup")
        if checkpoint_stage.is_symlink() or not checkpoint_stage.is_dir():
            raise RuntimeError("checkpoint staging directory changed before cleanup")
        checkpoint_stage.chmod(0o700)
        for child in checkpoint_stage.iterdir():
            if child.is_symlink() or not child.is_file():
                raise RuntimeError("checkpoint staging contents changed before cleanup")
            child.chmod(0o600)
        shutil.rmtree(checkpoint_stage)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main())
