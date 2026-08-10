"""Execute one accepted Stage-3 formal TabICL cohort on TabArena v0.1.

This runner is separate from the frozen exploratory ``tabarena-evaluate``
workflow.  It consumes the strict RoPE/Temporary/No-PE intake contract, requires
independent campaign-readiness evidence, copies the three evaluated checkpoints
into transaction scratch with streaming re-hashing, and publishes one atomic
private run plus a path-free public finding.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import stat
import sys
import tarfile
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .manifest import validate_output_dir
from .provenance import (
    RunTransaction,
    VerifiedRunContext,
    load_verified_json_config,
    manifest_from_verified_inputs,
    verify_configured_run_inputs,
    verify_file,
    verify_git_tree,
)
from .statistics import adjust_holm
from .tabarena_evaluation import (
    _BENCHMARK_FIELDS,
    _CLASSIFIER_FIELDS,
    _EXPECTED_ROSTER_FILE_SHA256,
    _FIXED_CLASSIFIER_OPTIONS,
    _PROBLEM_TYPES,
    _RESULT_FIELDS,
    _archive_directory,
    _load_cached_results,
    _mean_ranks,
    _normalize_results,
    _paired_comparison,
    _publish_json,
    _require_module_under,
    _runtime_summary,
    _scale_free_comparison,
    _validate_context_tasks,
    _validate_roster,
)
from .tabarena_formal_evaluation import (
    FORMAL_ARMS,
    FormalTabArenaInputSpec,
    parse_formal_tabarena_config,
    validate_formal_tabarena_inputs,
)


_TOP_LEVEL_FIELDS = {
    "schema_version",
    "study",
    "formal_inputs",
    "readiness",
    "benchmark",
    "provenance",
}
_FORMAL_BENCHMARK_FIELDS = _BENCHMARK_FIELDS | {
    "environment_manifest_path",
    "expected_environment_manifest_sha256",
}
_STUDY_FIELDS = {
    "seed",
    "temporary_identity_seed",
    "stage",
    "terminal_step",
    "task_subset",
    "problem_types",
    "device",
    "n_estimators",
    "augmentation",
    "ensemble_size",
    "cache_mode",
    "debug_mode",
    "bootstrap_resamples",
    "familywise_alpha",
    "classifier_options",
}
_FORMAL_INPUT_FIELDS = {
    "config_path",
    "expected_config_sha256",
}
_PROVENANCE_FIELDS = {
    "model_family",
    "model_revision",
    "condition",
    "sites",
    "checkpoint_path",
    "dataset_manifest_path",
    "training_code_root",
    "model_code_root",
    "analysis_code_root",
    "expected_checkpoint_sha256",
    "expected_dataset_manifest_sha256",
    "expected_training_code_sha",
    "expected_model_code_sha",
    "expected_analysis_code_sha",
    "allow_exploratory_legacy",
}
_EXPECTED_STAGE = "stage3"
_EXPECTED_TERMINAL_STEP = 10_000
_EXPECTED_TASK_COUNT = 38
_EXPECTED_RESULT_COUNT = 114
_EXPECTED_BOOTSTRAP_RESAMPLES = 10_000
_EXPECTED_FAMILYWISE_ALPHA = 0.05
_EXPECTED_CUDA_DEVICE_COUNT = 1
_EXPECTED_CUDA_DEVICE_NAME = "NVIDIA A10"
_EXPECTED_CUDA_DEVICE_CAPABILITY = [8, 6]
_HEX = frozenset("0123456789abcdef")
_COPY_CHUNK_BYTES = 8 * 1024 * 1024
_FORMAL_RAW_RESULT_FIELDS = _RESULT_FIELDS | {
    "formal_prediction_metadata",
    "method_metadata",
}
_METHOD_METADATA_FIELDS = {
    "schema_version",
    "kind",
    "arm",
    "seed",
    "stage",
    "terminal_step",
    "checkpoint_sha256",
    "training_code_sha",
    "n_estimators",
    "kv_cache",
    "classifier_options_sha256",
    "prediction_capture",
    "table_fingerprint_sha256",
    "test_feature_fingerprint_sha256",
    "temporary_table_identity_seed",
}
_PREDICTION_METADATA_FIELDS = {
    "schema_version",
    "kind",
    "arm",
    "dataset",
    "task_id",
    "fold",
    "repeat",
    "sample",
    "split_idx",
    "file_name",
    "file_sha256",
    "file_size_bytes",
    "dtype",
    "shape",
    "row_count",
    "class_count",
    "class_labels",
    "class_labels_sha256",
    "row_fingerprint_sha256",
    "test_target_sha256",
    "probability_sha256",
    "logical_sha256",
}
_PREDICTION_NPZ_FIELDS = {
    "class_labels_utf8",
    "probabilities",
    "row_fingerprints",
    "test_target_fingerprints",
}
_PAIR_ORDER = (
    ("rope", "temporary"),
    ("rope", "none"),
    ("temporary", "none"),
)
_ENVIRONMENT_FIELDS = {
    "schema_version",
    "kind",
    "python_executable_sha256",
    "python",
    "numpy",
    "pandas",
    "scikit_learn",
    "openml",
    "autogluon_core",
    "torch",
    "cuda_runtime",
    "cudnn",
    "cuda_device_count",
    "cuda_device_name",
    "cuda_device_capability",
    "nvidia_driver",
}


class FormalRunnerSpec:
    """Parsed runner configuration plus its already verified intake spec."""

    def __init__(
        self,
        *,
        seed: int,
        temporary_identity_seed: int,
        device: str,
        bootstrap_resamples: int,
        familywise_alpha: float,
        classifier_options: Mapping[str, Any],
        formal_config_path: Path,
        formal_config_sha256: str,
        formal_inputs: FormalTabArenaInputSpec,
        readiness: Any,
        tabarena_code_root: Path,
        tabarena_code_sha: str,
        openml_cache_root: Path,
        environment_manifest_path: Path,
        environment_manifest_sha256: str,
        environment_contract: Mapping[str, Any],
        expected_task_count: int,
        expected_result_count: int,
    ) -> None:
        self.seed = seed
        self.temporary_identity_seed = temporary_identity_seed
        self.device = device
        self.bootstrap_resamples = bootstrap_resamples
        self.familywise_alpha = familywise_alpha
        self.classifier_options = dict(classifier_options)
        self.formal_config_path = formal_config_path
        self.formal_config_sha256 = formal_config_sha256
        self.formal_inputs = formal_inputs
        self.readiness = readiness
        self.tabarena_code_root = tabarena_code_root
        self.tabarena_code_sha = tabarena_code_sha
        self.openml_cache_root = openml_cache_root
        self.environment_manifest_path = environment_manifest_path
        self.environment_manifest_sha256 = environment_manifest_sha256
        self.environment_contract = dict(environment_contract)
        self.expected_task_count = expected_task_count
        self.expected_result_count = expected_result_count


def _exact_object(value: object, *, label: str, fields: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        raise ValueError(
            f"{label} fields mismatch: missing={missing}, unknown={unknown}"
        )
    return value


def _digest(value: object, *, label: str, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase {length}-character digest")
    return value


def _absolute_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an absolute path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return path


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _parse_formal_input_file(
    path: Path, *, expected_sha256: str
) -> FormalTabArenaInputSpec:
    file = verify_file(path, expected_sha256=expected_sha256)
    try:
        value = json.loads(file.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("formal input config is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("formal input config must contain one object")
    spec = parse_formal_tabarena_config(value)
    file.assert_unchanged()
    return spec


def _parse_environment_manifest(
    path: Path, *, expected_sha256: str
) -> Mapping[str, Any]:
    file = verify_file(path, expected_sha256=expected_sha256)
    try:
        raw = json.loads(file.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("evaluation environment manifest is not valid JSON") from error
    value = _exact_object(
        raw,
        label="evaluation environment manifest",
        fields=_ENVIRONMENT_FIELDS,
    )
    if (
        value["schema_version"] != 1
        or value["kind"] != "formal_tabarena_a10_environment"
    ):
        raise ValueError("evaluation environment manifest identity is invalid")
    _digest(
        value["python_executable_sha256"],
        label="evaluation Python executable SHA-256",
    )
    for name in (
        "python",
        "numpy",
        "pandas",
        "scikit_learn",
        "openml",
        "autogluon_core",
        "torch",
        "cuda_runtime",
        "nvidia_driver",
    ):
        item = value[name]
        if not isinstance(item, str) or not item or len(item) > 128:
            raise ValueError(f"evaluation environment {name} is invalid")
    if (
        isinstance(value["cudnn"], bool)
        or not isinstance(value["cudnn"], int)
        or value["cudnn"] < 1
        or value["cuda_device_count"] != _EXPECTED_CUDA_DEVICE_COUNT
        or value["cuda_device_name"] != _EXPECTED_CUDA_DEVICE_NAME
        or value["cuda_device_capability"] != _EXPECTED_CUDA_DEVICE_CAPABILITY
    ):
        raise ValueError("evaluation environment must describe exactly one NVIDIA A10")
    file.assert_unchanged()
    return dict(value)


def parse_runner_config(data: Mapping[str, Any]) -> FormalRunnerSpec:
    config = _exact_object(
        data, label="formal TabArena runner config", fields=_TOP_LEVEL_FIELDS
    )
    if config["schema_version"] != 1:
        raise ValueError("formal TabArena runner schema_version must be 1")
    study = _exact_object(config["study"], label="study", fields=_STUDY_FIELDS)
    formal = _exact_object(
        config["formal_inputs"], label="formal_inputs", fields=_FORMAL_INPUT_FIELDS
    )
    benchmark = _exact_object(
        config["benchmark"], label="benchmark", fields=_FORMAL_BENCHMARK_FIELDS
    )
    provenance = _exact_object(
        config["provenance"], label="provenance", fields=_PROVENANCE_FIELDS
    )
    seed = study["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in {42, 43, 44}:
        raise ValueError("seed must be exactly one of 42, 43, or 44")
    temporary_identity_seed = study["temporary_identity_seed"]
    if (
        isinstance(temporary_identity_seed, bool)
        or not isinstance(temporary_identity_seed, int)
        or temporary_identity_seed != seed
    ):
        raise ValueError(
            "temporary_identity_seed must equal the formal experiment seed"
        )
    if (
        study["stage"] != _EXPECTED_STAGE
        or study["terminal_step"] != _EXPECTED_TERMINAL_STEP
    ):
        raise ValueError("formal TabArena execution accepts Stage 3 only")
    if (
        study["task_subset"] != "lite"
        or study["problem_types"] != list(_PROBLEM_TYPES)
        or study["device"] != "cuda"
        or study["n_estimators"] != 1
        or study["augmentation"] != "none"
        or study["ensemble_size"] != 1
        or study["cache_mode"] != "ignore"
        or study["debug_mode"] is not True
    ):
        raise ValueError("study differs from the frozen formal inference budget")
    classifier_options = _exact_object(
        study["classifier_options"],
        label="classifier_options",
        fields=_CLASSIFIER_FIELDS,
    )
    if dict(classifier_options) != _FIXED_CLASSIFIER_OPTIONS:
        raise ValueError("classifier_options differ from the frozen inference contract")
    if classifier_options["kv_cache"] is not False:
        raise ValueError("formal TabArena execution requires kv_cache=false")
    bootstrap_resamples = _positive_int(
        study["bootstrap_resamples"], label="bootstrap_resamples"
    )
    alpha = study["familywise_alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("familywise_alpha must be numeric")
    familywise_alpha = float(alpha)
    if (
        bootstrap_resamples != _EXPECTED_BOOTSTRAP_RESAMPLES
        or not math.isfinite(familywise_alpha)
        or familywise_alpha != _EXPECTED_FAMILYWISE_ALPHA
    ):
        raise ValueError(
            "formal statistics require exactly 10000 resamples and alpha=0.05"
        )

    formal_path = _absolute_path(formal["config_path"], label="formal input config")
    formal_sha = _digest(
        formal["expected_config_sha256"], label="formal input config SHA-256"
    )
    formal_inputs = _parse_formal_input_file(formal_path, expected_sha256=formal_sha)
    if (
        formal_inputs.seed != seed
        or formal_inputs.stage != _EXPECTED_STAGE
        or formal_inputs.terminal_step != _EXPECTED_TERMINAL_STEP
    ):
        raise ValueError("runner and formal input seed/stage do not match")
    from .tabarena_formal_readiness import parse_formal_readiness_config

    readiness = parse_formal_readiness_config(config["readiness"], formal_inputs)

    if benchmark["suite_version"] != "v0.1":
        raise ValueError("suite_version must be v0.1")
    expected_tasks = _positive_int(
        benchmark["expected_task_count"], label="expected_task_count"
    )
    expected_results = _positive_int(
        benchmark["expected_result_count"], label="expected_result_count"
    )
    if (
        expected_tasks != _EXPECTED_TASK_COUNT
        or expected_results != _EXPECTED_RESULT_COUNT
        or expected_results != expected_tasks * len(FORMAL_ARMS)
    ):
        raise ValueError("formal TabArena matrix must be exactly 38 tasks by 3 arms")

    environment_path = _absolute_path(
        benchmark["environment_manifest_path"],
        label="environment_manifest_path",
    )
    environment_sha256 = _digest(
        benchmark["expected_environment_manifest_sha256"],
        label="expected_environment_manifest_sha256",
    )
    environment_contract = _parse_environment_manifest(
        environment_path,
        expected_sha256=environment_sha256,
    )

    expected_revision = "formal-stage3-step-10000"
    rope = formal_inputs.arms["rope"][_EXPECTED_STAGE]
    if (
        provenance["model_family"] != "tabicl-v2"
        or provenance["model_revision"] != expected_revision
        or provenance["condition"] != "tabarena-formal-rope-temporary-none"
        or provenance["sites"] != ["tabarena-v0.1-classification"]
        or provenance["allow_exploratory_legacy"] is not False
    ):
        raise ValueError("provenance labels do not match formal TabArena execution")
    if (
        Path(str(provenance["checkpoint_path"])) != rope.checkpoint_path
        or provenance["expected_checkpoint_sha256"] != rope.checkpoint_sha256
        or Path(str(provenance["training_code_root"]))
        != formal_inputs.training_code_root
        or Path(str(provenance["model_code_root"])) != formal_inputs.training_code_root
        or provenance["expected_training_code_sha"] != formal_inputs.training_code_sha
        or provenance["expected_model_code_sha"] != formal_inputs.training_code_sha
        or provenance["expected_dataset_manifest_sha256"]
        != _EXPECTED_ROSTER_FILE_SHA256
    ):
        raise ValueError("provenance does not bind the exact formal cohort and source")

    return FormalRunnerSpec(
        seed=seed,
        temporary_identity_seed=temporary_identity_seed,
        device=str(study["device"]),
        bootstrap_resamples=bootstrap_resamples,
        familywise_alpha=familywise_alpha,
        classifier_options=dict(classifier_options),
        formal_config_path=formal_path,
        formal_config_sha256=formal_sha,
        formal_inputs=formal_inputs,
        readiness=readiness,
        tabarena_code_root=_absolute_path(
            benchmark["tabarena_code_root"], label="tabarena_code_root"
        ),
        tabarena_code_sha=_digest(
            benchmark["expected_tabarena_code_sha"],
            label="expected_tabarena_code_sha",
            length=40,
        ),
        openml_cache_root=_absolute_path(
            benchmark["openml_cache_root"], label="openml_cache_root"
        ),
        environment_manifest_path=environment_path,
        environment_manifest_sha256=environment_sha256,
        environment_contract=environment_contract,
        expected_task_count=expected_tasks,
        expected_result_count=expected_results,
    )


def _prepare_exact_training_import(spec: FormalTabArenaInputSpec) -> None:
    if any(name == "tabicl" or name.startswith("tabicl.") for name in sys.modules):
        raise RuntimeError(
            "tabicl was imported before the exact formal training source"
        )
    source = (spec.training_code_root / "src").resolve(strict=True)
    sys.path.insert(0, str(source))
    importlib.invalidate_caches()


@contextlib.contextmanager
def _deny_network_access():
    """Fail closed on cache misses instead of downloading benchmark inputs."""

    original_socket = socket.socket
    original_create_connection = socket.create_connection

    class OfflineSocket(original_socket):
        def connect(self, *args: Any, **kwargs: Any) -> Any:
            if self.family in {socket.AF_INET, socket.AF_INET6}:
                raise RuntimeError("network access is disabled for formal evaluation")
            return super().connect(*args, **kwargs)

        def connect_ex(self, *args: Any, **kwargs: Any) -> int:
            if self.family in {socket.AF_INET, socket.AF_INET6}:
                raise RuntimeError("network access is disabled for formal evaluation")
            return super().connect_ex(*args, **kwargs)

    def deny_create_connection(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("network access is disabled for formal evaluation")

    socket.socket = OfflineSocket
    socket.create_connection = deny_create_connection
    try:
        yield
    finally:
        socket.socket = original_socket
        socket.create_connection = original_create_connection


def _readiness(
    spec: FormalTabArenaInputSpec,
    checkpoint_report: Mapping[str, Any],
    readiness_spec: Any,
) -> Mapping[str, Any]:
    from .tabarena_formal_readiness import validate_formal_tabarena_readiness

    report = validate_formal_tabarena_readiness(spec, checkpoint_report, readiness_spec)
    if not isinstance(report, Mapping):
        raise RuntimeError("formal readiness validator returned an invalid report")
    required_true = (
        "checkpoint_lineage_verified",
        "campaign_authorization_verified",
        "terminal_scheduler_evidence_verified",
        "benchmark_execution_ready",
    )
    if any(report.get(field) is not True for field in required_true):
        raise ValueError("formal campaign is not authorized and terminal-ready")
    if report.get("campaign_acceptance_verified") is not False:
        raise ValueError("current-seed acceptance must remain post-evaluation")
    return dict(report)


def _stream_copy_checkpoint(
    source: Path, destination: Path, *, expected_sha256: str
) -> dict[str, Any]:
    """Copy one no-follow checkpoint and hash source/destination in one pass."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError("checkpoint scratch destination must be fresh")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    target_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | os.O_NOFOLLOW
    )
    source_fd = os.open(source, source_flags)
    target_fd = -1
    digest = hashlib.sha256()
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("formal checkpoint source must be a regular file")
        target_fd = os.open(destination, target_flags, 0o600)
        while True:
            chunk = os.read(source_fd, _COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(target_fd, view)
                if written <= 0:
                    raise OSError("checkpoint scratch copy made no progress")
                view = view[written:]
        os.fsync(target_fd)
        after = os.fstat(source_fd)
        copied = os.fstat(target_fd)
    except BaseException:
        if target_fd >= 0:
            os.close(target_fd)
            target_fd = -1
        destination.unlink(missing_ok=True)
        raise
    finally:
        if target_fd >= 0:
            os.close(target_fd)
        os.close(source_fd)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    observed = digest.hexdigest()
    if identity_before != identity_after:
        destination.unlink(missing_ok=True)
        raise RuntimeError("formal checkpoint changed during scratch copy")
    if observed != expected_sha256 or copied.st_size != before.st_size:
        destination.unlink(missing_ok=True)
        raise ValueError("formal checkpoint scratch copy digest or size mismatch")
    verified_copy = verify_file(destination, expected_sha256=expected_sha256)
    return {
        "path": verified_copy.path,
        "sha256": verified_copy.digest.sha256,
        "size_bytes": verified_copy.digest.size_bytes,
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise TypeError("object arrays cannot be content-hashed safely")
    header = _canonical_json_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def _canonical_scalar(value: Any) -> dict[str, Any]:
    scalar = value.item() if isinstance(value, np.generic) else value
    if isinstance(scalar, bool):
        return {"type": "bool", "value": scalar}
    if isinstance(scalar, int):
        return {"type": "int", "value": str(scalar)}
    if isinstance(scalar, float):
        if not math.isfinite(scalar):
            raise ValueError("prediction labels and row identifiers must be finite")
        return {"type": "float", "value": scalar.hex()}
    if isinstance(scalar, str):
        return {"type": "str", "value": scalar}
    if isinstance(scalar, bytes):
        return {"type": "bytes", "value": scalar.hex()}
    raise TypeError(
        "prediction labels and row identifiers must be scalar bool/int/float/str/bytes"
    )


def _class_label_evidence(values: Sequence[Any]) -> tuple[bytes, list[dict[str, Any]]]:
    canonical = [_canonical_scalar(value) for value in values]
    tokens = [_canonical_json_bytes(value) for value in canonical]
    if len(tokens) < 2 or len(set(tokens)) != len(tokens):
        raise ValueError("formal prediction class labels must be unique")
    encoded = _canonical_json_bytes(canonical)
    public = [
        {
            "index": index,
            "value_type": value["type"],
            "value_sha256": hashlib.sha256(token).hexdigest(),
        }
        for index, (value, token) in enumerate(zip(canonical, tokens))
    ]
    return encoded, public


def _decode_class_label_evidence(encoded: bytes) -> list[dict[str, Any]]:
    try:
        values = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("prediction class labels are not canonical JSON") from error
    if not isinstance(values, list):
        raise ValueError("prediction class labels must be a list")
    canonical: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping) or set(value) != {"type", "value"}:
            raise ValueError("prediction class label schema is invalid")
        kind = value["type"]
        raw = value["value"]
        if kind == "bool":
            if not isinstance(raw, bool):
                raise ValueError("boolean class label is invalid")
            scalar: Any = raw
        elif kind == "int":
            if not isinstance(raw, str):
                raise ValueError("integer class label is invalid")
            try:
                scalar = int(raw)
            except ValueError as error:
                raise ValueError("integer class label is invalid") from error
            if str(scalar) != raw:
                raise ValueError("integer class label is not canonical")
        elif kind == "float":
            if not isinstance(raw, str):
                raise ValueError("float class label is invalid")
            try:
                scalar = float.fromhex(raw)
            except ValueError as error:
                raise ValueError("float class label is invalid") from error
        elif kind == "str":
            if not isinstance(raw, str):
                raise ValueError("string class label is invalid")
            scalar = raw
        elif kind == "bytes":
            if not isinstance(raw, str):
                raise ValueError("bytes class label is invalid")
            try:
                scalar = bytes.fromhex(raw)
            except ValueError as error:
                raise ValueError("bytes class label is invalid") from error
        else:
            raise ValueError("prediction class label type is invalid")
        canonical.append(_canonical_scalar(scalar))
    if _canonical_json_bytes(canonical) != encoded:
        raise ValueError("prediction class labels are not canonically encoded")
    _, public = _class_label_evidence(
        [
            value["value"]
            if value["type"] in {"bool", "str"}
            else (
                int(value["value"])
                if value["type"] == "int"
                else (
                    float.fromhex(value["value"])
                    if value["type"] == "float"
                    else bytes.fromhex(value["value"])
                )
            )
            for value in canonical
        ]
    )
    return public


def _row_fingerprints(index: Sequence[Any]) -> np.ndarray:
    rows = list(index)
    fingerprints = np.empty((len(rows), 32), dtype=np.uint8)
    for position, value in enumerate(rows):
        token = _canonical_json_bytes(_canonical_scalar(value))
        digest = hashlib.sha256(
            b"formal-tabarena-row-v1\0"
            + position.to_bytes(8, byteorder="big", signed=False)
            + len(token).to_bytes(8, byteorder="big", signed=False)
            + token
        ).digest()
        fingerprints[position] = np.frombuffer(digest, dtype=np.uint8)
    return np.ascontiguousarray(fingerprints)


def _test_target_fingerprints(targets: Any, *, expected_index: Any) -> np.ndarray:
    import pandas as pd

    if (
        not isinstance(targets, pd.Series)
        or len(targets) < 1
        or not targets.index.equals(expected_index)
    ):
        raise ValueError("formal test targets must align with restored prediction rows")
    fingerprints = np.empty((len(targets), 32), dtype=np.uint8)
    for position, (index, value) in enumerate(targets.items()):
        index_token = _canonical_json_bytes(_canonical_scalar(index))
        value_token = _canonical_json_bytes(_canonical_scalar(value))
        digest = hashlib.sha256(
            b"formal-tabarena-test-target-v1\0"
            + position.to_bytes(8, byteorder="big", signed=False)
            + len(index_token).to_bytes(8, byteorder="big", signed=False)
            + index_token
            + len(value_token).to_bytes(8, byteorder="big", signed=False)
            + value_token
        ).digest()
        fingerprints[position] = np.frombuffer(digest, dtype=np.uint8)
    return np.ascontiguousarray(fingerprints)


def _feature_table_fingerprint(X: Any, *, domain: str) -> str:
    import pandas as pd

    if not isinstance(X, pd.DataFrame) or len(X) < 1:
        raise TypeError("formal table fingerprint requires a non-empty DataFrame")
    if not isinstance(domain, str) or not domain:
        raise ValueError("formal table fingerprint domain must be non-empty")
    schema = {
        "columns": [_canonical_scalar(value) for value in X.columns],
        "feature_dtypes": [str(dtype) for dtype in X.dtypes],
        "row_count": len(X),
    }
    try:
        feature_hashes = pd.util.hash_pandas_object(
            X, index=True, categorize=True
        ).to_numpy(dtype=np.dtype("<u8"), copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("formal table contains unhashable feature values") from error
    digest = hashlib.sha256()
    for component in (
        f"formal-{domain}-feature-table-v1".encode("ascii"),
        _canonical_json_bytes(schema),
        np.ascontiguousarray(feature_hashes).tobytes(order="C"),
    ):
        digest.update(len(component).to_bytes(8, byteorder="big", signed=False))
        digest.update(component)
    return digest.hexdigest()


def _table_content_fingerprint(X: Any, y: Any) -> str:
    import pandas as pd

    if not isinstance(y, pd.Series) or len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("formal training features and labels are misaligned")
    feature_digest = _feature_table_fingerprint(X, domain="train")
    target_schema = {
        "target_name": _canonical_scalar(y.name) if y.name is not None else None,
        "target_dtype": str(y.dtype),
        "row_count": len(y),
    }
    try:
        target_hashes = pd.util.hash_pandas_object(
            y, index=True, categorize=True
        ).to_numpy(dtype=np.dtype("<u8"), copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("formal table contains unhashable target values") from error
    return hashlib.sha256(
        b"formal-training-table-v1\0"
        + bytes.fromhex(feature_digest)
        + _canonical_json_bytes(target_schema)
        + np.ascontiguousarray(target_hashes).tobytes(order="C")
    ).hexdigest()


def _temporary_table_seed(*, formal_seed: int, table_fingerprint_sha256: str) -> int:
    fingerprint = _digest(table_fingerprint_sha256, label="Temporary table fingerprint")
    encoded = (
        b"formal-temporary-table-seed-v1\0"
        + int(formal_seed).to_bytes(8, byteorder="big", signed=True)
        + bytes.fromhex(fingerprint)
    )
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def _prediction_file_name(*, arm: str, dataset: str) -> str:
    if arm not in FORMAL_ARMS or not isinstance(dataset, str) or not dataset:
        raise ValueError("formal prediction arm or dataset is invalid")
    dataset_digest = hashlib.sha256(dataset.encode("utf-8")).hexdigest()
    name = f"prediction-{arm}-{dataset_digest}.npz"
    if Path(name).name != name or "/" in name or "\\" in name or ".." in name:
        raise RuntimeError("formal prediction filename is not portable")
    return name


def _expected_method_metadata(
    *,
    arm: str,
    seed: int,
    checkpoint_sha256: str,
    training_code_sha: str,
    classifier_options: Mapping[str, Any],
    table_fingerprint_sha256: str,
    test_feature_fingerprint_sha256: str,
    temporary_table_identity_seed: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "formal_tabarena_method_metadata",
        "arm": arm,
        "seed": seed,
        "stage": _EXPECTED_STAGE,
        "terminal_step": _EXPECTED_TERMINAL_STEP,
        "checkpoint_sha256": checkpoint_sha256,
        "training_code_sha": training_code_sha,
        "n_estimators": 1,
        "kv_cache": False,
        "classifier_options_sha256": _json_sha256(dict(classifier_options)),
        "prediction_capture": "restored_test_probabilities_float32_v1",
        "table_fingerprint_sha256": table_fingerprint_sha256,
        "test_feature_fingerprint_sha256": test_feature_fingerprint_sha256,
        "temporary_table_identity_seed": temporary_table_identity_seed,
    }


def _prediction_logical_payload(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "domain": "formal_tabarena_prediction_v1",
        "arm": metadata["arm"],
        "dataset": metadata["dataset"],
        "task_id": metadata["task_id"],
        "fold": metadata["fold"],
        "repeat": metadata["repeat"],
        "sample": metadata["sample"],
        "split_idx": metadata["split_idx"],
        "dtype": metadata["dtype"],
        "shape": metadata["shape"],
        "row_count": metadata["row_count"],
        "class_count": metadata["class_count"],
        "class_labels": metadata["class_labels"],
        "class_labels_sha256": metadata["class_labels_sha256"],
        "row_fingerprint_sha256": metadata["row_fingerprint_sha256"],
        "test_target_sha256": metadata["test_target_sha256"],
        "probability_sha256": metadata["probability_sha256"],
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_prediction_npz(
    path: Path,
    *,
    probabilities: np.ndarray,
    row_fingerprints: np.ndarray,
    test_target_fingerprints: np.ndarray,
    class_labels_utf8: bytes,
) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("formal prediction destination must be fresh")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("formal prediction root must be a real directory")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as handle:
            np.savez_compressed(
                handle,
                probabilities=probabilities,
                row_fingerprints=row_fingerprints,
                test_target_fingerprints=test_target_fingerprints,
                class_labels_utf8=np.frombuffer(class_labels_utf8, dtype=np.uint8),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _capture_formal_prediction(
    probabilities: Any,
    *,
    test_targets: Any,
    prediction_root: Path,
    arm: str,
    dataset: str,
    task_id: int,
    fold: int,
    repeat: int,
    sample: int,
    split_idx: int,
) -> dict[str, Any]:
    import pandas as pd

    if not isinstance(probabilities, pd.DataFrame):
        raise TypeError("formal TabArena probabilities must be a pandas DataFrame")
    source = probabilities.to_numpy(copy=False)
    if source.ndim != 2 or not np.issubdtype(source.dtype, np.floating):
        raise ValueError("formal TabArena probabilities must be a floating matrix")
    values = np.ascontiguousarray(source, dtype=np.dtype("<f4"))
    if values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("formal TabArena probability matrix is empty")
    if not np.isfinite(values).all():
        raise ValueError("formal TabArena probabilities must be finite float32")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("formal TabArena probabilities must lie in [0,1]")
    if not np.allclose(values.sum(axis=1, dtype=np.float64), 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("formal TabArena probability rows must sum to one")
    class_bytes, class_labels = _class_label_evidence(list(probabilities.columns))
    rows = _row_fingerprints(list(probabilities.index))
    target_fingerprints = _test_target_fingerprints(
        test_targets,
        expected_index=probabilities.index,
    )
    if rows.shape[0] != values.shape[0] or len(class_labels) != values.shape[1]:
        raise RuntimeError("formal prediction rows or class labels are misaligned")
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "kind": "formal_tabarena_prediction",
        "arm": arm,
        "dataset": dataset,
        "task_id": int(task_id),
        "fold": int(fold),
        "repeat": int(repeat),
        "sample": int(sample),
        "split_idx": int(split_idx),
        "file_name": _prediction_file_name(arm=arm, dataset=dataset),
        "dtype": "float32",
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "row_count": int(values.shape[0]),
        "class_count": int(values.shape[1]),
        "class_labels": class_labels,
        "class_labels_sha256": hashlib.sha256(class_bytes).hexdigest(),
        "row_fingerprint_sha256": _array_sha256(rows),
        "test_target_sha256": _array_sha256(target_fingerprints),
        "probability_sha256": _array_sha256(values),
    }
    metadata["logical_sha256"] = _json_sha256(_prediction_logical_payload(metadata))
    output = prediction_root / metadata["file_name"]
    _publish_prediction_npz(
        output,
        probabilities=values,
        row_fingerprints=rows,
        test_target_fingerprints=target_fingerprints,
        class_labels_utf8=class_bytes,
    )
    verified = verify_file(output)
    metadata["file_sha256"] = verified.digest.sha256
    metadata["file_size_bytes"] = verified.digest.size_bytes
    return metadata


def _reset_temporary_identity_rng(estimator: Any, *, arm: str, seed: int) -> None:
    """Reset only the Temporary model's CPU generator, never global RNG state."""

    if arm != "temporary":
        return
    import torch

    raw = getattr(estimator, "model_", None)
    row = getattr(raw, "row_interactor", None)
    if (
        getattr(raw, "row_identity_mode", None) != "temporary"
        or getattr(row, "identity_mode", None) != "temporary"
    ):
        raise RuntimeError("Temporary checkpoint did not construct a Temporary model")
    generator = getattr(row, "_identity_generator", None)
    if generator is None or not hasattr(generator, "manual_seed"):
        raise RuntimeError("Temporary model lacks its module-local identity generator")
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else ()
    )
    generator.manual_seed(seed)
    if not torch.equal(torch.random.get_rng_state(), cpu_before):
        raise RuntimeError("Temporary module-local reset changed global CPU RNG")
    if torch.cuda.is_available():
        cuda_after = torch.cuda.get_rng_state_all()
        if len(cuda_after) != len(cuda_before) or any(
            not torch.equal(actual, expected)
            for actual, expected in zip(cuda_after, cuda_before)
        ):
            raise RuntimeError("Temporary module-local reset changed global CUDA RNG")


def _make_formal_system_model(external_system_model: type) -> type:
    class FormalTabICLSystem(external_system_model):
        def __init__(
            self,
            *,
            checkpoint: str,
            checkpoint_sha256: str,
            training_code_sha: str,
            arm: str,
            device: str,
            n_estimators: int,
            seed: int,
            temporary_identity_seed: int,
            classifier_options: Mapping[str, Any],
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            if arm not in FORMAL_ARMS:
                raise ValueError("formal TabICL arm is invalid")
            if classifier_options.get("kv_cache") is not False:
                raise ValueError("formal TabICL system requires kv_cache=false")
            self.checkpoint = checkpoint
            self.checkpoint_sha256 = _digest(
                checkpoint_sha256, label="formal method checkpoint SHA-256"
            )
            self.training_code_sha = _digest(
                training_code_sha, label="formal method training code SHA", length=40
            )
            self.arm = arm
            self.device = device
            self.n_estimators = n_estimators
            self.seed = seed
            self.temporary_identity_seed = temporary_identity_seed
            self.classifier_options = dict(classifier_options)
            self.model: Any | None = None
            self.table_fingerprint_sha256: str | None = None
            self.test_feature_fingerprint_sha256: str | None = None
            self.table_identity_seed: int | None = None

        def _fit_system(
            self,
            X: Any,
            y: Any,
            *,
            target_name: str,
            problem_type: str,
            eval_metric: Any,
            validation_metadata: Any,
            num_cpus: int | None,
            num_gpus: int | None,
            memory_limit: float | None,
            time_limit: float | None,
            random_state: int | None,
        ) -> "FormalTabICLSystem":
            del target_name, eval_metric, validation_metadata, num_gpus, memory_limit
            del time_limit, random_state
            if problem_type not in _PROBLEM_TYPES:
                raise ValueError("formal TabArena runner supports classification only")
            from tabicl import TabICLClassifier

            self.table_fingerprint_sha256 = _table_content_fingerprint(X, y)
            if self.arm == "temporary":
                self.table_identity_seed = _temporary_table_seed(
                    formal_seed=self.temporary_identity_seed,
                    table_fingerprint_sha256=self.table_fingerprint_sha256,
                )
            checkpoint = Path(self.checkpoint)
            if checkpoint.is_symlink() or not checkpoint.is_file():
                raise FileNotFoundError("transaction checkpoint scratch disappeared")
            self.model = TabICLClassifier(
                model_path=str(checkpoint),
                allow_auto_download=False,
                device=self.device,
                n_estimators=self.n_estimators,
                random_state=self.seed,
                n_jobs=max(1, int(num_cpus or 1)),
                **self.classifier_options,
            )
            self.model.fit(X, y)
            if getattr(self.model.model_, "row_identity_mode", None) != self.arm:
                raise RuntimeError(
                    "loaded checkpoint identity mode differs from its arm"
                )
            _reset_temporary_identity_rng(
                self.model,
                arm=self.arm,
                seed=(
                    self.table_identity_seed
                    if self.table_identity_seed is not None
                    else self.temporary_identity_seed
                ),
            )
            return self

        def post_fit(self, X: Any, y: Any, X_test: Any) -> None:
            fresh_training = _table_content_fingerprint(X, y)
            if fresh_training != self.table_fingerprint_sha256:
                raise RuntimeError(
                    "fresh TabArena reload differs from the fitted training table"
                )
            self.test_feature_fingerprint_sha256 = _feature_table_fingerprint(
                X_test, domain="test"
            )

        def _predict(self, X: Any) -> Any:
            if self.model is None:
                raise RuntimeError("formal TabICL system is not fitted")
            import pandas as pd

            _reset_temporary_identity_rng(
                self.model,
                arm=self.arm,
                seed=(
                    self.table_identity_seed
                    if self.table_identity_seed is not None
                    else self.temporary_identity_seed
                ),
            )
            return pd.Series(self.model.predict(X), index=X.index)

        def _predict_proba(self, X: Any) -> Any:
            if self.model is None:
                raise RuntimeError("formal TabICL system is not fitted")
            import pandas as pd

            _reset_temporary_identity_rng(
                self.model,
                arm=self.arm,
                seed=(
                    self.table_identity_seed
                    if self.table_identity_seed is not None
                    else self.temporary_identity_seed
                ),
            )
            values = self.model.predict_proba(X)
            return pd.DataFrame(values, index=X.index, columns=self.model.classes_)

        def get_metadata(self) -> dict[str, Any]:
            if (
                self.table_fingerprint_sha256 is None
                or self.test_feature_fingerprint_sha256 is None
            ):
                raise RuntimeError("formal table fingerprints are incomplete")
            return _expected_method_metadata(
                arm=self.arm,
                seed=self.seed,
                checkpoint_sha256=self.checkpoint_sha256,
                training_code_sha=self.training_code_sha,
                classifier_options=self.classifier_options,
                table_fingerprint_sha256=self.table_fingerprint_sha256,
                test_feature_fingerprint_sha256=(self.test_feature_fingerprint_sha256),
                temporary_table_identity_seed=self.table_identity_seed,
            )

        def cleanup(self) -> None:
            self.model = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:  # pragma: no cover
                pass

    FormalTabICLSystem.__name__ = "FormalTabICLSystem"
    FormalTabICLSystem.__qualname__ = "FormalTabICLSystem"
    return FormalTabICLSystem


def _make_formal_prediction_runner(external_runner: type) -> type:
    class FormalPredictionRunner(external_runner):
        def __init__(self, *, formal_prediction_root: str, **kwargs: Any) -> None:
            root = Path(formal_prediction_root)
            if not root.is_absolute():
                raise ValueError("formal prediction root must be absolute")
            self.formal_prediction_root = root
            super().__init__(**kwargs)

        def post_evaluate(self, out: dict[str, Any]) -> dict[str, Any]:
            result = super().post_evaluate(out)
            if "formal_prediction_metadata" in result:
                raise RuntimeError("formal prediction metadata already exists")
            method = result.get("method_metadata")
            if not isinstance(method, Mapping) or method.get("arm") not in FORMAL_ARMS:
                raise RuntimeError(
                    "formal method metadata is unavailable at prediction capture"
                )
            probabilities = result.get("probabilities")
            if probabilities is None:
                raise RuntimeError("formal classification result lacks probabilities")
            test_targets = self._load_y_test()
            result["formal_prediction_metadata"] = _capture_formal_prediction(
                probabilities,
                test_targets=test_targets,
                prediction_root=self.formal_prediction_root,
                arm=str(method["arm"]),
                dataset=self.task_name,
                task_id=int(self.task.task_id),
                fold=int(self.fold),
                repeat=int(self.repeat),
                sample=int(self.sample),
                split_idx=int(self.task_split_idx),
            )
            return result

    FormalPredictionRunner.__name__ = "FormalPredictionRunner"
    FormalPredictionRunner.__qualname__ = "FormalPredictionRunner"
    return FormalPredictionRunner


def _import_bound_runtime(tabarena_root: Path, model_root: Path) -> Mapping[str, Any]:
    """Import TabArena while accepting only the already-bound exact-T TabICL."""

    if any(name == "tabarena" or name.startswith("tabarena.") for name in sys.modules):
        raise RuntimeError("tabarena was imported before its verified source path")
    tabarena_src = (tabarena_root / "packages" / "tabarena" / "src").resolve(
        strict=True
    )
    model_src = (model_root / "src").resolve(strict=True)
    existing = sys.modules.get("tabicl")
    if existing is not None:
        _require_module_under(existing, model_src, name="tabicl")
    sys.path.insert(0, str(model_src))
    sys.path.insert(0, str(tabarena_src))
    importlib.invalidate_caches()
    tabarena = importlib.import_module("tabarena")
    tabicl = importlib.import_module("tabicl")
    _require_module_under(tabarena, tabarena_src, name="tabarena")
    _require_module_under(tabicl, model_src, name="tabicl")
    return {
        "tabarena": tabarena,
        "tabicl": tabicl,
        "ExternalSystemModel": importlib.import_module(
            "tabarena.benchmark.exec_models"
        ).ExternalSystemModel,
        "OOFExperimentRunner": importlib.import_module(
            "tabarena.benchmark.experiment.experiment_runner"
        ).OOFExperimentRunner,
        "TabArenaV0pt1ExperimentBundle": importlib.import_module(
            "tabarena.benchmark.experiment"
        ).TabArenaV0pt1ExperimentBundle,
        "CacheConfig": importlib.import_module("tabarena.caching").CacheConfig,
        "TabArenaContext": importlib.import_module("tabarena.contexts").TabArenaContext,
        "SystemConfigGenerator": importlib.import_module(
            "tabarena.utils.config_utils"
        ).SystemConfigGenerator,
    }


def _execute_tabarena(
    *,
    runtime: Mapping[str, Any],
    spec: FormalRunnerSpec,
    checkpoints: Mapping[str, Path],
    roster: tuple[str, ...],
    results_root: Path,
    scratch_root: Path,
    prediction_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, dict[str, Any]]]:
    model_cls = _make_formal_system_model(runtime["ExternalSystemModel"])
    prediction_runner_cls = _make_formal_prediction_runner(
        runtime["OOFExperimentRunner"]
    )
    framework_to_arm: dict[str, str] = {}
    experiments = []
    for arm in FORMAL_ARMS:
        generator = runtime["SystemConfigGenerator"](
            model_cls=model_cls,
            name=f"TabICL_Formal_{arm}_Seed{spec.seed}_Stage3",
            manual_configs=[
                {
                    "checkpoint": str(checkpoints[arm]),
                    "checkpoint_sha256": spec.formal_inputs.arms[arm][
                        _EXPECTED_STAGE
                    ].checkpoint_sha256,
                    "training_code_sha": spec.formal_inputs.training_code_sha,
                    "arm": arm,
                    "device": spec.device,
                    "n_estimators": 1,
                    "seed": spec.seed,
                    "temporary_identity_seed": spec.temporary_identity_seed,
                    "classifier_options": dict(spec.classifier_options),
                }
            ],
        )
        built = runtime["TabArenaV0pt1ExperimentBundle"](
            models=[(generator, 0)], system_experiments=True
        ).build_experiments()
        if len(built) != 1 or built[0].name in framework_to_arm:
            raise RuntimeError(
                "TabArena did not construct one unique framework per arm"
            )
        built[0].experiment_cls = prediction_runner_cls
        built[0].experiment_kwargs = {
            **built[0].experiment_kwargs,
            "formal_prediction_root": str(prediction_root),
        }
        framework_to_arm[built[0].name] = arm
        experiments.extend(built)
    cache = runtime["CacheConfig"](
        openml=spec.openml_cache_root,
        huggingface=scratch_root / "huggingface",
        data_foundry=scratch_root / "data-foundry",
        tabarena=scratch_root / "tabarena",
        results=results_root,
        apply_on_run=True,
        scope_openml=True,
    )
    arena = runtime["TabArenaContext"](methods=[], backend="native", cache_config=cache)
    expected_tasks = _validate_context_tasks(arena, roster)
    jobs = arena.build_jobs(
        experiments,
        subset=["lite"],
        dataset_names=list(roster),
        problem_types=list(_PROBLEM_TYPES),
    )
    if len(jobs) != _EXPECTED_RESULT_COUNT:
        raise RuntimeError("TabArena did not construct the exact formal 38x3 matrix")
    expected_keys = {
        (framework, dataset, 0, 0)
        for framework in framework_to_arm
        for dataset in roster
    }
    observed_keys = {
        (job.experiment.name, job.task.dataset, job.task.fold, job.task.repeat)
        for job in jobs
    }
    if observed_keys != expected_keys or len(observed_keys) != len(jobs):
        raise RuntimeError("TabArena jobs differ from the formal 38x3 lite grid")
    with _deny_network_access():
        results = arena.run_jobs(
            jobs,
            expname=results_root,
            register=False,
            cache_mode="ignore",
            debug_mode=True,
            raise_on_failure=True,
        )
    if not all(isinstance(item, dict) for item in results):
        raise RuntimeError("TabArena returned a non-dictionary result")
    return list(results), framework_to_arm, expected_tasks


def _validate_method_metadata(
    value: Any, *, arm: str, spec: FormalRunnerSpec
) -> dict[str, Any]:
    metadata = _exact_object(
        value, label="formal method_metadata", fields=_METHOD_METADATA_FIELDS
    )
    fingerprint = _digest(
        metadata["table_fingerprint_sha256"], label="training table fingerprint"
    )
    test_fingerprint = _digest(
        metadata["test_feature_fingerprint_sha256"],
        label="test feature fingerprint",
    )
    table_seed = metadata["temporary_table_identity_seed"]
    if arm == "temporary":
        if isinstance(table_seed, bool) or not isinstance(table_seed, int):
            raise ValueError("Temporary table identity seed must be an integer")
        if table_seed != _temporary_table_seed(
            formal_seed=spec.temporary_identity_seed,
            table_fingerprint_sha256=fingerprint,
        ):
            raise ValueError(
                "Temporary table identity seed does not match its fingerprint"
            )
    elif table_seed is not None:
        raise ValueError("RoPE/No-PE method metadata contains Temporary RNG state")
    stage3 = spec.formal_inputs.arms[arm][_EXPECTED_STAGE]
    expected = _expected_method_metadata(
        arm=arm,
        seed=spec.seed,
        checkpoint_sha256=stage3.checkpoint_sha256,
        training_code_sha=spec.formal_inputs.training_code_sha,
        classifier_options=spec.classifier_options,
        table_fingerprint_sha256=fingerprint,
        test_feature_fingerprint_sha256=test_fingerprint,
        temporary_table_identity_seed=table_seed,
    )
    if dict(metadata) != expected:
        raise ValueError(
            "formal method_metadata differs from the exact evaluation contract"
        )
    return expected


def _verify_prediction_npz(path: Path, metadata: Mapping[str, Any]) -> None:
    verified = verify_file(path, expected_sha256=str(metadata["file_sha256"]))
    if verified.digest.size_bytes != metadata["file_size_bytes"]:
        raise ValueError("formal prediction file size changed")
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != _PREDICTION_NPZ_FIELDS:
                raise ValueError("formal prediction NPZ fields differ from its schema")
            probabilities = np.asarray(payload["probabilities"])
            rows = np.asarray(payload["row_fingerprints"])
            targets = np.asarray(payload["test_target_fingerprints"])
            labels_array = np.asarray(payload["class_labels_utf8"])
    except (OSError, ValueError) as error:
        raise ValueError("formal prediction NPZ is unreadable") from error
    if probabilities.dtype != np.dtype("<f4"):
        raise ValueError("formal prediction probabilities are not float32")
    if list(probabilities.shape) != metadata["shape"]:
        raise ValueError("formal prediction probability shape changed")
    if not np.isfinite(probabilities).all():
        raise ValueError("formal prediction probabilities contain non-finite values")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("formal prediction probabilities are outside [0,1]")
    if not np.allclose(
        probabilities.sum(axis=1, dtype=np.float64), 1.0, rtol=1e-6, atol=1e-6
    ):
        raise ValueError("formal prediction probability rows do not sum to one")
    if rows.dtype != np.uint8 or rows.shape != (metadata["row_count"], 32):
        raise ValueError("formal prediction row fingerprints are invalid")
    if targets.dtype != np.uint8 or targets.shape != (metadata["row_count"], 32):
        raise ValueError("formal prediction test-target fingerprints are invalid")
    if labels_array.dtype != np.uint8 or labels_array.ndim != 1:
        raise ValueError("formal prediction class-label bytes are invalid")
    label_bytes = labels_array.tobytes(order="C")
    public_labels = _decode_class_label_evidence(label_bytes)
    if public_labels != metadata["class_labels"]:
        raise ValueError("formal prediction class labels changed")
    checks = {
        "class_labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "row_fingerprint_sha256": _array_sha256(rows),
        "test_target_sha256": _array_sha256(targets),
        "probability_sha256": _array_sha256(probabilities),
    }
    if any(metadata[name] != digest for name, digest in checks.items()):
        raise ValueError("formal prediction content hash changed")
    verified.assert_unchanged()


def _validate_prediction_metadata(
    value: Any,
    *,
    arm: str,
    dataset: str,
    task: Mapping[str, Any],
    prediction_root: Path,
) -> dict[str, Any]:
    metadata = _exact_object(
        value,
        label="formal prediction metadata",
        fields=_PREDICTION_METADATA_FIELDS,
    )
    expected_identity = {
        "schema_version": 1,
        "kind": "formal_tabarena_prediction",
        "arm": arm,
        "dataset": dataset,
        "task_id": int(task["tid"]),
        "fold": int(task["fold"]),
        "repeat": int(task["repeat"]),
        "sample": int(task["sample"]),
        "split_idx": int(task["split_idx"]),
        "file_name": _prediction_file_name(arm=arm, dataset=dataset),
        "dtype": "float32",
    }
    if any(
        metadata.get(name) != expected for name, expected in expected_identity.items()
    ):
        raise ValueError("formal prediction metadata identity differs from its result")
    shape = metadata["shape"]
    rows = metadata["row_count"]
    classes = metadata["class_count"]
    size = metadata["file_size_bytes"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in shape)
        or isinstance(rows, bool)
        or not isinstance(rows, int)
        or isinstance(classes, bool)
        or not isinstance(classes, int)
        or shape != [rows, classes]
        or rows < 1
        or classes < 2
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
    ):
        raise ValueError("formal prediction shape/count/size metadata is invalid")
    labels = metadata["class_labels"]
    if not isinstance(labels, list) or len(labels) != classes:
        raise ValueError("formal prediction class label metadata is invalid")
    for index, label in enumerate(labels):
        if (
            not isinstance(label, Mapping)
            or set(label) != {"index", "value_type", "value_sha256"}
            or label["index"] != index
            or label["value_type"] not in {"bool", "int", "float", "str", "bytes"}
        ):
            raise ValueError("formal prediction class label metadata is invalid")
        _digest(label["value_sha256"], label="class label digest")
    for name in (
        "file_sha256",
        "class_labels_sha256",
        "row_fingerprint_sha256",
        "test_target_sha256",
        "probability_sha256",
        "logical_sha256",
    ):
        _digest(metadata[name], label=name)
    if metadata["logical_sha256"] != _json_sha256(
        _prediction_logical_payload(metadata)
    ):
        raise ValueError("formal prediction logical digest changed")
    result = dict(metadata)
    _verify_prediction_npz(prediction_root / result["file_name"], result)
    return result


def _normalize_formal_results(
    results: Sequence[Mapping[str, Any]],
    *,
    framework_to_arm: Mapping[str, str],
    roster: tuple[str, ...],
    spec: FormalRunnerSpec,
    expected_tasks: Mapping[str, Mapping[str, Any]],
    prediction_root: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    stripped: list[dict[str, Any]] = []
    evidence: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in results:
        if not isinstance(raw, Mapping) or set(raw) != _FORMAL_RAW_RESULT_FIELDS:
            raise ValueError(
                "formal TabArena result fields differ from the exact schema"
            )
        framework = raw["framework"]
        if not isinstance(framework, str) or framework not in framework_to_arm:
            raise ValueError("formal TabArena result has an unknown framework")
        arm = framework_to_arm[framework]
        task = raw["task_metadata"]
        if not isinstance(task, Mapping) or not isinstance(task.get("name"), str):
            raise ValueError("formal TabArena task metadata is invalid")
        dataset = str(task["name"])
        method = _validate_method_metadata(raw["method_metadata"], arm=arm, spec=spec)
        prediction = _validate_prediction_metadata(
            raw["formal_prediction_metadata"],
            arm=arm,
            dataset=dataset,
            task=task,
            prediction_root=prediction_root,
        )
        key = (arm, dataset)
        if key in evidence:
            raise ValueError(
                "formal prediction evidence contains a duplicate arm/dataset"
            )
        evidence[key] = {"method_metadata": method, "prediction": prediction}
        stripped.append({name: raw[name] for name in _RESULT_FIELDS})
    normalized = _normalize_results(
        stripped,
        framework_to_arm=framework_to_arm,
        roster=roster,
        expected_count=spec.expected_result_count,
        expected_tasks=expected_tasks,
        arm_order=FORMAL_ARMS,
    )
    return normalized, evidence


def _prediction_manifest(
    evidence: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    prediction_root: Path,
    roster: tuple[str, ...],
    spec: FormalRunnerSpec,
) -> dict[str, Any]:
    expected_keys = {(arm, dataset) for arm in FORMAL_ARMS for dataset in roster}
    if set(evidence) != expected_keys:
        raise ValueError("formal prediction evidence is not the exact 38x3 matrix")
    expected_files = {
        str(evidence[key]["prediction"]["file_name"]) for key in expected_keys
    }
    entries = list(prediction_root.iterdir()) if prediction_root.is_dir() else []
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("formal prediction root contains a non-regular file")
    if {entry.name for entry in entries} != expected_files:
        raise ValueError("formal prediction root contains missing or extra files")
    matched_datasets: dict[str, str] = {}
    for dataset in roster:
        signatures = []
        for arm in FORMAL_ARMS:
            item = evidence[(arm, dataset)]
            method = item["method_metadata"]
            prediction = item["prediction"]
            signatures.append(
                {
                    "dataset": dataset,
                    "task_id": prediction["task_id"],
                    "fold": prediction["fold"],
                    "repeat": prediction["repeat"],
                    "sample": prediction["sample"],
                    "split_idx": prediction["split_idx"],
                    "training_table_sha256": method["table_fingerprint_sha256"],
                    "test_features_sha256": method["test_feature_fingerprint_sha256"],
                    "row_count": prediction["row_count"],
                    "row_fingerprint_sha256": prediction["row_fingerprint_sha256"],
                    "test_target_sha256": prediction["test_target_sha256"],
                    "class_count": prediction["class_count"],
                    "class_labels": prediction["class_labels"],
                    "class_labels_sha256": prediction["class_labels_sha256"],
                }
            )
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError(
                f"formal train/test/target/row/class evidence differs across arms for {dataset}"
            )
        matched_datasets[dataset] = _json_sha256(
            {
                "schema_version": 1,
                "domain": "formal_tabarena_matched_dataset_v1",
                **signatures[0],
            }
        )
    matched_dataset_sha256 = _json_sha256(
        {
            "schema_version": 1,
            "domain": "formal_tabarena_matched_dataset_matrix_v1",
            "datasets": [
                {"dataset": dataset, "sha256": matched_datasets[dataset]}
                for dataset in roster
            ],
        }
    )
    arms: dict[str, Any] = {}
    arm_receipts: dict[str, str] = {}
    for arm in FORMAL_ARMS:
        predictions = []
        logical = []
        for dataset in roster:
            item = dict(evidence[(arm, dataset)]["prediction"])
            _verify_prediction_npz(prediction_root / item["file_name"], item)
            item["archive_member"] = f"predictions/{item['file_name']}"
            predictions.append(item)
            logical.append(
                {
                    "dataset": dataset,
                    "matched_dataset_sha256": matched_datasets[dataset],
                    "logical_sha256": item["logical_sha256"],
                }
            )
        arm_digest = _json_sha256(
            {
                "schema_version": 1,
                "domain": "formal_tabarena_arm_predictions_v1",
                "arm": arm,
                "seed": spec.seed,
                "training_code_sha": spec.formal_inputs.training_code_sha,
                "checkpoint_sha256": spec.formal_inputs.arms[arm][
                    _EXPECTED_STAGE
                ].checkpoint_sha256,
                "predictions": logical,
            }
        )
        arm_receipts[arm] = arm_digest
        arms[arm] = {
            "checkpoint_sha256": spec.formal_inputs.arms[arm][
                _EXPECTED_STAGE
            ].checkpoint_sha256,
            "prediction_count": len(predictions),
            "evaluated_examples": int(sum(item["row_count"] for item in predictions)),
            "logical_sha256": arm_digest,
            "predictions": predictions,
        }
    manifest = {
        "schema_version": 1,
        "kind": "formal_tabarena_prediction_manifest",
        "format": "npz_float32_probabilities_v1",
        "seed": spec.seed,
        "stage": _EXPECTED_STAGE,
        "terminal_step": _EXPECTED_TERMINAL_STEP,
        "training_code_sha": spec.formal_inputs.training_code_sha,
        "task_count": len(roster),
        "prediction_count": len(evidence),
        "archive_artifact_name": "predictions.tar.gz",
        "matched_dataset_sha256": matched_dataset_sha256,
        "matched_dataset_by_name_sha256": matched_datasets,
        "arm_logical_sha256": arm_receipts,
        "overall_logical_sha256": _json_sha256(
            {
                "schema_version": 1,
                "domain": "formal_tabarena_all_predictions_v1",
                "seed": spec.seed,
                "arm_logical_sha256": arm_receipts,
            }
        ),
        "arms": arms,
    }
    _assert_path_free(manifest)
    return manifest


def _archive_prediction_directory(source: Path, destination: Path) -> None:
    entries = sorted(source.iterdir(), key=lambda path: path.name)
    if not entries or any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("formal prediction archive source is invalid")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("formal prediction archive destination must be fresh")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed:
                with tarfile.open(mode="w", fileobj=compressed) as archive:
                    for path in entries:
                        member = f"predictions/{path.name}"
                        info = archive.gettarinfo(str(path), arcname=member)
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mtime = 0
                        with path.open("rb") as handle:
                            archive.addfile(info, handle)
            raw.flush()
            os.fsync(raw.fileno())
        os.link(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _holm_family(
    comparisons: list[dict[str, Any]], *, p_field: str, alpha: float
) -> None:
    adjusted = adjust_holm([float(item[p_field]) for item in comparisons])
    for item, adjusted_p in zip(comparisons, adjusted):
        item["holm_adjusted_p"] = float(adjusted_p)
        item["holm_raw_p_field"] = p_field
        item["multiplicity_method"] = "holm"
        item["multiplicity_family_size"] = len(comparisons)
        item["familywise_alpha"] = alpha
        item["familywise_reject"] = bool(adjusted_p <= alpha)


def _aggregate(
    normalized: Sequence[Mapping[str, Any]],
    *,
    roster: tuple[str, ...],
    spec: FormalRunnerSpec,
    framework_to_arm: Mapping[str, str],
    lineage: Mapping[str, Any],
    readiness: Mapping[str, Any],
    context: VerifiedRunContext,
    benchmark_sha: str,
    evaluation_protocol: Mapping[str, Any],
    prediction_manifest: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = {(str(item["arm"]), str(item["dataset"])): item for item in normalized}
    overall = [
        _scale_free_comparison(rows, roster=roster, left=left, right=right)
        for left, right in _PAIR_ORDER
    ]
    _holm_family(
        overall,
        p_field="two_sided_exact_sign_test_p",
        alpha=spec.familywise_alpha,
    )
    mean_ranks = _mean_ranks(rows, roster=roster, arm_order=FORMAL_ARMS)
    metric_groups: dict[str, Any] = {}
    for metric_index, metric in enumerate(
        sorted({str(item["metric"]) for item in normalized}), start=1
    ):
        metric_roster = tuple(
            dataset for dataset in roster if rows[("rope", dataset)]["metric"] == metric
        )
        comparisons = [
            _paired_comparison(
                rows,
                roster=metric_roster,
                left=left,
                right=right,
                seed=spec.seed + 100 * metric_index + pair_index,
                n_resamples=spec.bootstrap_resamples,
            )
            for pair_index, (left, right) in enumerate(_PAIR_ORDER, start=1)
        ]
        _holm_family(
            comparisons,
            p_field="two_sided_exact_sign_test_p",
            alpha=spec.familywise_alpha,
        )
        metric_groups[metric] = {
            "inferential_role": "secondary_exploratory",
            "dataset_count": len(metric_roster),
            "macro_mean_metric_error": {
                arm: float(
                    np.mean(
                        [
                            rows[(arm, dataset)]["metric_error"]
                            for dataset in metric_roster
                        ]
                    )
                )
                for arm in FORMAL_ARMS
            },
            "paired_comparisons": comparisons,
        }
    framework_by_arm = {arm: name for name, arm in framework_to_arm.items()}
    sanity_inputs = (
        None
        if prediction_manifest is None
        else {
            "matched_dataset_sha256": prediction_manifest["matched_dataset_sha256"],
            "results_by_arm": {
                arm: {
                    "evaluated_datasets": len(roster),
                    "evaluated_examples": prediction_manifest["arms"][arm][
                        "evaluated_examples"
                    ],
                    "prediction_manifest_sha256": prediction_manifest[
                        "arm_logical_sha256"
                    ][arm],
                    "metrics": {"mean_rank": mean_ranks[arm]},
                }
                for arm in FORMAL_ARMS
            },
        }
    )
    public = {
        "schema_version": 1,
        "kind": "formal_tabarena_single_seed_finding",
        "formal_claim_ready": False,
        "formal_claim_blocker": "requires_all_three_prespecified_seeds",
        "leaderboard_replication": False,
        "seed": spec.seed,
        "stage": _EXPECTED_STAGE,
        "terminal_step": _EXPECTED_TERMINAL_STEP,
        "task_count": len(roster),
        "result_count": len(normalized),
        "arms": list(FORMAL_ARMS),
        "familywise_alpha": spec.familywise_alpha,
        "bootstrap_resamples": spec.bootstrap_resamples,
        "primary_inference_family": "overall_scale_free_pairwise",
        "temporary_identity_inference_policy": (
            "derive_from_training_seed_and_training_table_content_then_reuse_within_table"
        ),
        "temporary_identity_seed": spec.temporary_identity_seed,
        "kv_cache": False,
        "raw_predictions_saved": prediction_manifest is not None,
        "network_access_disabled": True,
        "prediction_receipt": (
            None
            if prediction_manifest is None
            else {
                "manifest_kind": prediction_manifest["kind"],
                "prediction_count": prediction_manifest["prediction_count"],
                "matched_dataset_sha256": prediction_manifest["matched_dataset_sha256"],
                "arm_logical_sha256": prediction_manifest["arm_logical_sha256"],
                "overall_logical_sha256": prediction_manifest["overall_logical_sha256"],
            }
        ),
        "campaign_acceptance_ready": False,
        "campaign_acceptance_blocker": (
            "per_sample_prediction_manifest_not_yet_emitted"
            if prediction_manifest is None
            else "formal_acceptance_registry_is_post_evaluation"
        ),
        "minimum_evaluation_sanity_inputs": sanity_inputs,
        "checkpoint_transaction_scratch_reverified": True,
        "framework_by_arm": framework_by_arm,
        "formal_lineage": lineage,
        "formal_readiness": readiness,
        "code_provenance": {
            "training_code_sha": context.inputs.training_code.head_sha,
            "model_code_sha": context.inputs.model_code.head_sha,
            "analysis_code_sha": context.inputs.analysis_code.head_sha,
            "benchmark_code_sha": benchmark_sha,
            "dataset_roster_sha256": context.inputs.dataset_manifest.digest.sha256,
            "environment_manifest_sha256": spec.environment_manifest_sha256,
        },
        "evaluation_protocol": dict(evaluation_protocol),
        "overall_mean_rank_lower_is_better": mean_ranks,
        "overall_scale_free_pairwise": overall,
        "metric_groups": metric_groups,
        "statistical_note": (
            "The three overall scale-free comparisons are the sole primary Holm "
            "family. Per-metric comparisons are separately adjusted secondary "
            "exploration. This single-seed record is not the final three-seed claim."
        ),
    }
    private = {
        "schema_version": 1,
        "kind": "formal_tabarena_private_metric_evidence",
        "formal_lineage": lineage,
        "formal_readiness": readiness,
        "results": list(normalized),
    }
    _assert_path_free(public)
    return public, private


def _assert_path_free(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower().endswith(("_path", "_root")):
                raise ValueError("public formal finding contains a path field")
            _assert_path_free(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_path_free(child)
    elif isinstance(value, str) and (value.startswith(("/", "~")) or "\\" in value):
        raise ValueError("public formal finding contains a private path")


def _context(configuration: Any, spec: FormalRunnerSpec) -> VerifiedRunContext:
    formal = spec.formal_inputs
    stage3 = {arm: formal.arms[arm][_EXPECTED_STAGE] for arm in FORMAL_ARMS}
    additional_paths = {
        "formal_input_config": spec.formal_config_path,
        "formal_campaign": spec.readiness.campaign_path,
        "formal_terminal_attestation": spec.readiness.terminal_attestation_path,
        "formal_submission_receipt": spec.readiness.submission_receipt_path,
        "evaluation_environment": spec.environment_manifest_path,
        "transaction_ledger": formal.transaction_ledger_path,
        "temporary_checkpoint": stage3["temporary"].checkpoint_path,
        "none_checkpoint": stage3["none"].checkpoint_path,
        **{
            f"{arm}_finalized_manifest": stage3[arm].finalized_manifest_path
            for arm in FORMAL_ARMS
        },
    }
    expected = {
        "formal_input_config": spec.formal_config_sha256,
        "formal_campaign": spec.readiness.campaign_file_sha256,
        "formal_terminal_attestation": spec.readiness.terminal_file_sha256,
        "formal_submission_receipt": spec.readiness.submission_receipt_file_sha256,
        "evaluation_environment": spec.environment_manifest_sha256,
        "transaction_ledger": formal.transaction_ledger_file_sha256,
        "temporary_checkpoint": stage3["temporary"].checkpoint_sha256,
        "none_checkpoint": stage3["none"].checkpoint_sha256,
        **{
            f"{arm}_finalized_manifest": stage3[arm].finalized_manifest_file_sha256
            for arm in FORMAL_ARMS
        },
    }
    return verify_configured_run_inputs(
        configuration,
        command="tabarena-formal-evaluate",
        seed=spec.seed,
        additional_input_paths=additional_paths,
        expected_additional_sha256=expected,
    )


def _runtime_environment_evidence(
    *, runtime: Mapping[str, Any], benchmark_sha: str
) -> dict[str, Any]:
    observed = _runtime_summary(
        runtime=runtime,
        benchmark_sha=benchmark_sha,
        duration_seconds=0.0,
        result_count=0,
    )
    executable = verify_file(Path(sys.executable).resolve(strict=True))
    evidence = {
        name: observed[name]
        for name in _ENVIRONMENT_FIELDS
        if name not in {"schema_version", "kind", "python_executable_sha256"}
    }
    return {
        "schema_version": 1,
        "kind": "formal_tabarena_a10_environment",
        "python_executable_sha256": executable.digest.sha256,
        **evidence,
    }


def _assert_runtime_environment(
    *, expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> None:
    if dict(observed) != dict(expected):
        mismatches = sorted(
            name
            for name in _ENVIRONMENT_FIELDS
            if observed.get(name) != expected.get(name)
        )
        raise RuntimeError(
            "runtime differs from the frozen A10 evaluation environment: "
            + ", ".join(mismatches)
        )


def _evaluation_protocol(
    *,
    spec: FormalRunnerSpec,
    context: VerifiedRunContext,
    benchmark_sha: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "kind": "formal_tabarena_evaluation_protocol",
        "formal_seed_policy": [42, 43, 44],
        "temporary_identity_seed_policy": (
            "training_seed_plus_training_table_content_fingerprint"
        ),
        "stage": _EXPECTED_STAGE,
        "terminal_step": _EXPECTED_TERMINAL_STEP,
        "suite_version": "v0.1",
        "task_subset": "lite",
        "problem_types": list(_PROBLEM_TYPES),
        "arms": list(FORMAL_ARMS),
        "task_count": spec.expected_task_count,
        "result_count": spec.expected_result_count,
        "device": "cuda",
        "cuda_device_name": _EXPECTED_CUDA_DEVICE_NAME,
        "cuda_device_capability": _EXPECTED_CUDA_DEVICE_CAPABILITY,
        "n_estimators": 1,
        "augmentation": "none",
        "ensemble_size": 1,
        "cache_mode": "ignore",
        "debug_mode": True,
        "network_access": "disabled_fail_closed",
        "bootstrap_resamples": _EXPECTED_BOOTSTRAP_RESAMPLES,
        "familywise_alpha": _EXPECTED_FAMILYWISE_ALPHA,
        "primary_multiplicity_family": "overall_scale_free_pairwise",
        "classifier_options": dict(spec.classifier_options),
        "training_code_sha": context.inputs.training_code.head_sha,
        "analysis_code_sha": context.inputs.analysis_code.head_sha,
        "benchmark_code_sha": benchmark_sha,
        "dataset_roster_sha256": context.inputs.dataset_manifest.digest.sha256,
        "environment_manifest_sha256": spec.environment_manifest_sha256,
    }
    return {**payload, "sha256": _json_sha256(payload)}


def _external_runtime_root(protected_roots: Sequence[Path]) -> Path:
    raw = os.environ.get("PE_RUNTIME_ROOT")
    if raw is None:
        raise RuntimeError("PE_RUNTIME_ROOT is required for formal evaluation")
    runtime = _absolute_path(raw, label="PE_RUNTIME_ROOT")
    if runtime.is_symlink() or not runtime.is_dir():
        raise ValueError("PE_RUNTIME_ROOT must be an existing real directory")
    for root in protected_roots:
        validate_output_dir(runtime, source_root=root)
    return runtime.resolve(strict=True)


def run_evaluation(config_path: Path, output_dir: Path) -> dict[str, Any]:
    configuration = load_verified_json_config(config_path)
    spec = parse_runner_config(configuration.data)
    _prepare_exact_training_import(spec.formal_inputs)
    lineage = validate_formal_tabarena_inputs(spec.formal_inputs)
    readiness = _readiness(spec.formal_inputs, lineage, spec.readiness)
    context = _context(configuration, spec)
    benchmark_git = verify_git_tree(
        spec.tabarena_code_root, expected_sha=spec.tabarena_code_sha
    )
    roster = _validate_roster(
        context.inputs.dataset_manifest,
        benchmark_sha=benchmark_git.head_sha,
        expected_count=spec.expected_task_count,
    )
    if spec.openml_cache_root.is_symlink() or not spec.openml_cache_root.is_dir():
        raise ValueError("openml_cache_root must be an existing real directory")
    import torch

    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != _EXPECTED_CUDA_DEVICE_COUNT
        or torch.cuda.get_device_name(0) != _EXPECTED_CUDA_DEVICE_NAME
        or list(torch.cuda.get_device_capability(0)) != _EXPECTED_CUDA_DEVICE_CAPABILITY
    ):
        raise RuntimeError(
            "formal TabArena execution requires exactly one visible NVIDIA A10"
        )
    protected_roots = (
        context.inputs.training_code.root,
        context.inputs.model_code.root,
        context.inputs.analysis_code.root,
        benchmark_git.root,
        spec.formal_inputs.artifact_root,
        spec.readiness.campaign_path.parent,
        spec.readiness.acceptance_registry,
        spec.openml_cache_root,
    )
    external_runtime_root = _external_runtime_root(protected_roots)
    roots = (*protected_roots, external_runtime_root)
    evaluation_protocol = _evaluation_protocol(
        spec=spec,
        context=context,
        benchmark_sha=benchmark_git.head_sha,
    )
    runtime = _import_bound_runtime(
        spec.tabarena_code_root, context.inputs.model_code.root
    )
    runtime_environment = _runtime_environment_evidence(
        runtime=runtime,
        benchmark_sha=benchmark_git.head_sha,
    )
    _assert_runtime_environment(
        expected=spec.environment_contract,
        observed=runtime_environment,
    )
    with RunTransaction(output_dir, source_roots=roots) as transaction:
        checkpoint_scratch = transaction.staging_dir / "checkpoint-scratch"
        checkpoints: dict[str, Path] = {}
        scratch_evidence: dict[str, Any] = {}
        for arm in FORMAL_ARMS:
            stage = spec.formal_inputs.arms[arm][_EXPECTED_STAGE]
            copied = _stream_copy_checkpoint(
                stage.checkpoint_path,
                checkpoint_scratch / f"{arm}.ckpt",
                expected_sha256=stage.checkpoint_sha256,
            )
            checkpoints[arm] = copied.pop("path")
            scratch_evidence[arm] = copied
        lineage = {**lineage, "checkpoint_transaction_scratch": scratch_evidence}
        results_root = transaction.staging_dir / "tabarena-results"
        runtime_root = transaction.staging_dir / "runtime-cache"
        prediction_root = transaction.staging_dir / "prediction-scratch"
        started = time.monotonic()
        returned, framework_to_arm, expected_tasks = _execute_tabarena(
            runtime=runtime,
            spec=spec,
            checkpoints=checkpoints,
            roster=roster,
            results_root=results_root,
            scratch_root=runtime_root,
            prediction_root=prediction_root,
        )
        duration = time.monotonic() - started
        normalized, prediction_evidence = _normalize_formal_results(
            returned,
            framework_to_arm=framework_to_arm,
            roster=roster,
            spec=spec,
            expected_tasks=expected_tasks,
            prediction_root=prediction_root,
        )
        cached, cached_prediction_evidence = _normalize_formal_results(
            _load_cached_results(results_root),
            framework_to_arm=framework_to_arm,
            roster=roster,
            spec=spec,
            expected_tasks=expected_tasks,
            prediction_root=prediction_root,
        )
        if cached != normalized:
            raise RuntimeError("returned results differ from the fresh TabArena cache")
        if cached_prediction_evidence != prediction_evidence:
            raise RuntimeError(
                "returned prediction/method metadata differ from the fresh TabArena cache"
            )
        prediction_manifest = _prediction_manifest(
            prediction_evidence,
            prediction_root=prediction_root,
            roster=roster,
            spec=spec,
        )
        public, private = _aggregate(
            normalized,
            roster=roster,
            spec=spec,
            framework_to_arm=framework_to_arm,
            lineage=lineage,
            readiness=readiness,
            context=context,
            benchmark_sha=benchmark_git.head_sha,
            evaluation_protocol=evaluation_protocol,
            prediction_manifest=prediction_manifest,
        )
        _publish_json(transaction.staging_dir / "public-finding.json", public)
        _publish_json(transaction.staging_dir / "private-results.json", private)
        _publish_json(transaction.staging_dir / "input-validation.json", lineage)
        _publish_json(
            transaction.staging_dir / "prediction-manifest.json",
            prediction_manifest,
        )
        _publish_json(transaction.staging_dir / "readiness-validation.json", readiness)
        _publish_json(
            transaction.staging_dir / "evaluation-protocol.json",
            evaluation_protocol,
        )
        runtime_report = _runtime_summary(
            runtime=runtime,
            benchmark_sha=benchmark_git.head_sha,
            duration_seconds=duration,
            result_count=len(normalized),
        )
        runtime_report.update(
            {
                "environment_contract_verified": True,
                "network_access_disabled": True,
                "environment_manifest_sha256": spec.environment_manifest_sha256,
                "evaluation_protocol_sha256": evaluation_protocol["sha256"],
                "python_executable_sha256": runtime_environment[
                    "python_executable_sha256"
                ],
            }
        )
        _publish_json(transaction.staging_dir / "runtime.json", runtime_report)
        _archive_directory(results_root, transaction.staging_dir / "results.tar.gz")
        _archive_prediction_directory(
            prediction_root, transaction.staging_dir / "predictions.tar.gz"
        )
        shutil.rmtree(results_root)
        shutil.rmtree(prediction_root)
        if runtime_root.exists():
            shutil.rmtree(runtime_root)
        shutil.rmtree(checkpoint_scratch)
        artifacts = transaction.artifact_digests(
            (
                "private-results.json",
                "evaluation-protocol.json",
                "prediction-manifest.json",
                "predictions.tar.gz",
                "public-finding.json",
                "input-validation.json",
                "readiness-validation.json",
                "results.tar.gz",
                "runtime.json",
            )
        )
        manifest = manifest_from_verified_inputs(context.inputs, artifacts=artifacts)
        benchmark_git.assert_unchanged()
        transaction.commit(manifest, verified_inputs=context.inputs)
    return public


def run(args: argparse.Namespace) -> int:
    run_evaluation(Path(args.config), Path(args.output_dir))
    return 0


__all__ = [
    "FormalRunnerSpec",
    "parse_runner_config",
    "run",
    "run_evaluation",
]
