"""Strict fixed-weight TabPFN v2.6 position localization on TALENT discovery data.

This module is intentionally independent of the command-line layer.  It fits
the official classifier on each TALENT training split, evaluates validation
rows only, and atomically publishes paired ``Wp+b``/``Wp``/``b``/zero results.
All filesystem paths remain confined to the verified private configuration;
published artifacts contain only portable roles and measured content hashes.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import re
from typing import Any

import numpy as np

from .collect import (
    DEFAULT_MAX_PRIVATE_BYTES,
    DEFAULT_MIN_FREE_BYTES,
    _check_budget,
    _publish_json,
    _publish_npz,
)
from .official_collect import (
    DEFAULT_MAX_CLASSES,
    _DatasetSpec,
    _assert_dataset_inputs,
    _dataset_input_bindings,
    _enforce_class_limit,
    _private_study_root,
    _validate_configuration as _validate_talent_configuration,
)
from .official_tabicl import load_raw_talent_splits
from .official_tabpfn import (
    OfficialTabPFNV26Driver,
    OfficialTabPFNV26Result,
    TabPFNV26ConditionResult,
    fit_official_tabpfn_v26_driver,
)
from .provenance import (
    RunTransaction,
    VerifiedConfiguration,
    VerifiedRunContext,
    assert_dataset_roster,
    load_verified_json_config,
    manifest_from_verified_inputs,
    verify_configured_run_inputs,
)
from .statistics import summarize_paired, summary_dict


_COMPONENTS = ("full", "weight", "bias", "none")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")
_ROOT_FIELDS = {
    "schema_version",
    "private_study_root",
    "datasets",
    "roster_split",
    "evaluation_split",
    "trusted_pickle",
    "seed",
    "max_classes",
    "device",
    "max_private_bytes",
    "min_free_bytes",
    "provenance",
}
_REQUIRED_ROOT_FIELDS = {
    "schema_version",
    "private_study_root",
    "datasets",
    "roster_split",
    "evaluation_split",
    "trusted_pickle",
    "seed",
    "device",
    "provenance",
}
_REQUIRED_PROVENANCE_ASSERTIONS = {
    "expected_checkpoint_sha256": _SHA256,
    "expected_dataset_manifest_sha256": _SHA256,
    "expected_training_code_sha": _GIT_SHA,
    "expected_model_code_sha": _GIT_SHA,
    "expected_analysis_code_sha": _GIT_SHA,
}


def run_tabpfn_localization(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    driver_factory: Callable[..., OfficialTabPFNV26Driver] = (
        fit_official_tabpfn_v26_driver
    ),
) -> Mapping[str, Any]:
    """Run the strict discovery/validation workflow and atomically publish it."""

    configuration = load_verified_json_config(config_path)
    config, specs = _validate_configuration(configuration)
    additional_paths, expected_hashes, role_maps = _dataset_input_bindings(specs)
    missing_hashes = sorted(set(additional_paths) - set(expected_hashes))
    if missing_hashes:
        raise ValueError(
            "TabPFN localization requires expected SHA-256 for every TALENT "
            f"input role: {missing_hashes}"
        )

    seed = int(config["seed"])
    context = verify_configured_run_inputs(
        configuration,
        command="tabpfn-localize",
        seed=seed,
        additional_input_paths=additional_paths,
        expected_additional_sha256=expected_hashes,
    )
    _validate_contract(context, config=config)
    dataset_ids = tuple(spec.dataset_id for spec in specs)
    assert_dataset_roster(
        context.inputs.dataset_manifest,
        dataset_ids,
        exact=True,
        required_split="discovery",
    )

    arrays: dict[str, np.ndarray] = {}
    datasets: list[dict[str, Any]] = []
    for spec in specs:
        dataset = load_raw_talent_splits(
            spec.directory,
            trusted_pickle=bool(config["trusted_pickle"]),
        )
        _assert_dataset_inputs(
            dataset,
            spec=spec,
            context=context,
            role_map=role_maps[spec.dataset_id],
        )
        n_classes = _enforce_class_limit(
            dataset,
            int(config.get("max_classes", DEFAULT_MAX_CLASSES)),
        )
        driver = driver_factory(
            dataset,
            context.inputs.checkpoint.path,
            checkpoint_sha256=context.inputs.checkpoint.digest.sha256,
            model_sha=context.inputs.model_code.head_sha,
            expected_source_root=context.inputs.model_code.root,
            seed=seed,
            device=str(config["device"]),
        )
        _validate_driver(
            driver,
            dataset=dataset,
            context=context,
            expected_n_classes=n_classes,
        )
        result = driver.evaluate("val")
        _validate_driver(
            driver,
            dataset=dataset,
            context=context,
            expected_n_classes=n_classes,
        )
        _require_same_class_order(
            np.asarray(driver.estimator.classes_),
            np.asarray(result.classes),
        )
        dataset_metadata, dataset_arrays = _dataset_result(
            result,
            dataset=dataset,
            spec=spec,
            context=context,
            input_roles=role_maps[spec.dataset_id],
        )
        collision = sorted(set(arrays) & set(dataset_arrays))
        if collision:
            raise RuntimeError(f"TabPFN prediction array key collision: {collision}")
        arrays.update(dataset_arrays)
        datasets.append(dataset_metadata)

    runtime = _runtime_attestation(device=str(config["device"]))
    runtime_sha256 = _json_sha256(runtime)
    source_binding = {
        "checkpoint_sha256": context.inputs.checkpoint.digest.sha256,
        "dataset_manifest_sha256": context.inputs.dataset_manifest.digest.sha256,
        "training_code_sha": context.inputs.training_code.head_sha,
        "model_code_sha": context.inputs.model_code.head_sha,
        "analysis_code_sha": context.inputs.analysis_code.head_sha,
        "configuration_sha256": context.inputs.configuration.file.digest.sha256,
    }
    condition_definitions = [
        {
            "condition": "full",
            "position_component": "Wp+b",
            "official_native_equivalent": True,
        },
        {
            "condition": "weight",
            "position_component": "Wp",
            "official_native_equivalent": False,
        },
        {
            "condition": "bias",
            "position_component": "b",
            "official_native_equivalent": False,
        },
        {
            "condition": "none",
            "position_component": "zero",
            "official_native_equivalent": False,
        },
    ]
    results = {
        "schema_version": 1,
        "kind": "official_tabpfn_v2.6_fixed_weight_position_localization",
        "evidence_scope": "exploratory-discovery-validation",
        "formal_eligible": False,
        "model_family": context.model_family,
        "model_revision": context.model_revision,
        "assignment_split": "discovery",
        "evaluation_split": "val",
        "fit_context": "train",
        "seed": seed,
        "device": str(config["device"]),
        "official_estimator": {
            "n_estimators": 1,
            "random_state": seed,
            "fit_mode": "fit_preprocessors",
            "memory_saving_mode": False,
            "download_if_not_exists": False,
        },
        "condition_definitions": condition_definitions,
        "source_binding": source_binding,
        "runtime_attestation": runtime,
        "runtime_attestation_sha256": runtime_sha256,
        "datasets": datasets,
    }
    summary = _paired_summary(datasets, seed=seed)
    summary["condition_definitions"] = condition_definitions
    summary["source_binding"] = source_binding
    summary["runtime_attestation"] = runtime
    summary["runtime_attestation_sha256"] = runtime_sha256
    _assert_path_free(results)
    _assert_path_free(summary)

    private_root = _private_study_root(config["private_study_root"])
    resolved_output = Path(output_dir).expanduser()
    if not resolved_output.is_absolute():
        raise ValueError("TabPFN localization output_dir must be absolute")
    resolved_output = resolved_output.resolve(strict=False)
    if resolved_output == private_root or not resolved_output.is_relative_to(private_root):
        raise ValueError("TabPFN localization output must be below private_study_root")
    projected_bytes = (
        sum(int(array.nbytes) for array in arrays.values())
        + len(json.dumps(results, allow_nan=False))
        + len(json.dumps(summary, allow_nan=False))
        + (8 << 20)
    )
    _check_budget(
        private_root,
        projected_bytes=projected_bytes,
        max_private_bytes=int(
            config.get("max_private_bytes", DEFAULT_MAX_PRIVATE_BYTES)
        ),
        min_free_bytes=int(config.get("min_free_bytes", DEFAULT_MIN_FREE_BYTES)),
    )

    source_roots = (
        context.inputs.training_code.root,
        context.inputs.model_code.root,
        context.inputs.analysis_code.root,
    )
    with RunTransaction(resolved_output, source_roots=source_roots) as transaction:
        _publish_npz(transaction.staging_dir / "predictions.npz", **arrays)
        _publish_json(transaction.staging_dir / "results.json", results)
        _publish_json(transaction.staging_dir / "summary.json", summary)
        artifacts = transaction.artifact_digests(
            ("predictions.npz", "results.json", "summary.json")
        )
        manifest = manifest_from_verified_inputs(
            context.inputs,
            artifacts=artifacts,
        )
        transaction.commit(manifest, verified_inputs=context.inputs)
    return summary


def run(args: argparse.Namespace) -> int:
    """CLI adapter for :mod:`pe_mechanism.cli`."""

    run_tabpfn_localization(args.config, args.output_dir)
    return 0


def _validate_configuration(
    configuration: VerifiedConfiguration,
) -> tuple[dict[str, Any], tuple[_DatasetSpec, ...]]:
    config = dict(configuration.data)
    unknown = sorted(set(config) - _ROOT_FIELDS)
    missing = sorted(_REQUIRED_ROOT_FIELDS - set(config))
    if unknown or missing:
        raise ValueError(
            "TabPFN localization configuration fields mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    projected = VerifiedConfiguration(data=config, file=configuration.file)
    _, specs = _validate_talent_configuration(projected)
    if config["schema_version"] != 1:
        raise ValueError("TabPFN localization schema_version must be 1")
    if config["roster_split"] != "discovery":
        raise ValueError("TabPFN localization is restricted to discovery datasets")
    if config["evaluation_split"] != "val":
        raise ValueError("TabPFN localization reserves TALENT test and evaluates val only")
    if not isinstance(config["trusted_pickle"], bool):
        raise TypeError("trusted_pickle must be a JSON boolean")
    seed = config["seed"]
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed >= 2**32
    ):
        raise ValueError("seed must be an unsigned 32-bit integer")
    if not isinstance(config["device"], str) or not config["device"].strip():
        raise ValueError("device must be a non-empty string")

    provenance = config["provenance"]
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be an object")
    for field, pattern in _REQUIRED_PROVENANCE_ASSERTIONS.items():
        value = provenance.get(field)
        if not isinstance(value, str) or pattern.fullmatch(value) is None:
            raise ValueError(f"provenance.{field} must be an explicit content hash")
    if provenance.get("allow_exploratory_legacy", False) is not False:
        raise ValueError("TabPFN localization forbids exploratory legacy provenance")
    return config, specs


def _validate_contract(
    context: VerifiedRunContext, *, config: Mapping[str, Any]
) -> None:
    if context.inputs.evidence_level != "strict":
        raise RuntimeError("TabPFN localization requires strict provenance")
    if context.model_family != "tabpfn-v2.6":
        raise ValueError("model_family must be 'tabpfn-v2.6'")
    if context.model_revision != "v7.1.1":
        raise ValueError("model_revision must be the audited 'v7.1.1' API")
    if context.condition != "position-components":
        raise ValueError("condition must be 'position-components'")
    if context.sites != ("feature_positional_embedding",):
        raise ValueError(
            "sites must contain only the TabPFN feature positional embedding"
        )
    provenance = config["provenance"]
    assert isinstance(provenance, Mapping)
    if provenance.get("allow_exploratory_legacy", False) is not False:
        raise ValueError("TabPFN localization requires clean Git trees")


def _validate_driver(
    driver: Any,
    *,
    dataset: Any,
    context: VerifiedRunContext,
    expected_n_classes: int,
) -> None:
    required = (
        "dataset",
        "estimator",
        "model_code_sha",
        "loaded_checkpoint_sha256",
        "seed",
        "evaluate",
    )
    missing = [name for name in required if not hasattr(driver, name)]
    if missing:
        raise TypeError(f"official TabPFN driver is missing attributes: {missing}")
    if driver.dataset is not dataset:
        raise RuntimeError("official TabPFN driver replaced the loaded TALENT dataset")
    if driver.model_code_sha != context.inputs.model_code.head_sha:
        raise RuntimeError("official TabPFN driver model source SHA changed")
    if driver.loaded_checkpoint_sha256 != context.inputs.checkpoint.digest.sha256:
        raise RuntimeError("official TabPFN driver checkpoint SHA changed")
    if driver.seed != context.inputs.contract.seed:
        raise RuntimeError("official TabPFN driver seed changed")
    estimator = driver.estimator
    if int(getattr(estimator, "n_estimators", -1)) != 1:
        raise RuntimeError("official TabPFN runner requires n_estimators=1")
    if getattr(estimator, "random_state", None) != context.inputs.contract.seed:
        raise RuntimeError("official TabPFN estimator random_state changed")
    classes = np.asarray(getattr(estimator, "classes_", ()))
    if classes.ndim != 1 or classes.shape[0] != expected_n_classes:
        raise RuntimeError("official TabPFN fitted class count changed")


def _dataset_result(
    result: OfficialTabPFNV26Result,
    *,
    dataset: Any,
    spec: _DatasetSpec,
    context: VerifiedRunContext,
    input_roles: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if result.dataset_name != dataset.name or result.split != "val":
        raise RuntimeError("official TabPFN result dataset/split changed")
    if result.model_code_sha != context.inputs.model_code.head_sha:
        raise RuntimeError("official TabPFN result model source SHA changed")
    if result.loaded_checkpoint_sha256 != context.inputs.checkpoint.digest.sha256:
        raise RuntimeError("official TabPFN result checkpoint SHA changed")
    if result.seed != context.inputs.contract.seed:
        raise RuntimeError("official TabPFN result seed changed")
    if not result.exact_full_native_verified:
        raise RuntimeError("TabPFN full Wp+b policy was not exact to native")
    if not result.policies_restored_verified:
        raise RuntimeError("TabPFN position policies were not fully restored")
    if tuple(result.conditions) != _COMPONENTS:
        raise RuntimeError("TabPFN position condition roster/order changed")

    classes = np.asarray(result.classes)
    labels, class_order = _encode_labels(np.asarray(dataset.val.y), classes)
    expected_shape = (labels.shape[0], classes.shape[0])
    native = _probability_array(result.native_probabilities, expected_shape=expected_shape)
    full = _probability_array(
        result.conditions["full"].probabilities,
        expected_shape=expected_shape,
    )
    if not np.array_equal(native, full):
        raise RuntimeError("stored full probabilities differ from official native")

    prefix = f"d{spec.ordinal:04d}"
    label_key = f"{prefix}_labels"
    arrays: dict[str, np.ndarray] = {label_key: labels}
    condition_payload: dict[str, Any] = {}
    probability_hashes: dict[str, str] = {}
    for component in _COMPONENTS:
        condition = result.conditions[component]
        if condition.component != component:
            raise RuntimeError("TabPFN condition metadata differs from its key")
        probabilities = _probability_array(
            condition.probabilities,
            expected_shape=expected_shape,
        )
        key = f"{prefix}_{component}"
        arrays[key] = probabilities
        metrics = _metrics_from_stored_probabilities(probabilities, labels)
        _require_driver_metrics(condition, metrics=metrics, expected_rows=labels.shape[0])
        probability_sha256 = _array_sha256(probabilities)
        probability_hashes[component] = probability_sha256
        condition_payload[component] = {
            **metrics,
            "probability_sha256": probability_sha256,
            "prediction_array_key": key,
        }

    labels_sha256 = _array_sha256(labels)
    class_order_sha256 = _json_sha256(class_order)
    pairing_sha256 = _json_sha256(
        {
            "labels_sha256": labels_sha256,
            "class_order_sha256": class_order_sha256,
            "probability_sha256": probability_hashes,
        }
    )
    inputs = [
        {
            "logical_name": name,
            "role": input_roles[name],
            "sha256": dataset.input_sha256[name],
            "size_bytes": context.additional_file(input_roles[name]).digest.size_bytes,
        }
        for name in sorted(input_roles)
    ]
    metadata = {
        "dataset_id": dataset.name,
        "task_type": dataset.task_type,
        "assignment_split": "discovery",
        "evaluation_split": "val",
        "fit_context": "train",
        "n_train_samples": int(np.asarray(dataset.train.y).shape[0]),
        "n_validation_samples": int(labels.shape[0]),
        "n_numeric_features": int(dataset.n_numeric_features),
        "n_categorical_features": int(dataset.n_categorical_features),
        "n_classes": int(classes.shape[0]),
        "inputs": inputs,
        "input_bundle_sha256": _json_sha256(
            {
                name: dataset.input_sha256[name]
                for name in sorted(dataset.input_sha256)
            }
        ),
        "label_encoding": "int64_zero_based_index_into_probability_columns",
        "label_array_key": label_key,
        "labels_sha256": labels_sha256,
        "class_order": class_order,
        "class_order_sha256": class_order_sha256,
        "pairing_sha256": pairing_sha256,
        "conditions": condition_payload,
        "exact_full_native_verified": True,
        "policies_restored_verified": True,
    }
    return metadata, arrays


def _encode_labels(
    labels: np.ndarray, classes: np.ndarray
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if classes.ndim != 1 or classes.shape[0] < 2:
        raise RuntimeError("official TabPFN class order must be a non-empty vector")
    class_tokens = [_canonical_label(value) for value in classes]
    if len(set(class_tokens)) != len(class_tokens):
        raise RuntimeError("official TabPFN class order contains duplicates")
    lookup = {token: index for index, token in enumerate(class_tokens)}
    vector = np.asarray(labels)
    if vector.ndim == 2 and vector.shape[1] == 1:
        vector = vector[:, 0]
    if vector.ndim != 1 or vector.shape[0] == 0:
        raise ValueError("TALENT validation labels must be a non-empty vector")
    try:
        encoded = np.asarray(
            [lookup[_canonical_label(value)] for value in vector],
            dtype=np.int64,
        )
    except KeyError as error:
        raise RuntimeError("validation labels differ from official class order") from error
    class_order = [
        {
            "index": index,
            "value_type": json.loads(token)["type"],
            "value_sha256": hashlib.sha256(token).hexdigest(),
        }
        for index, token in enumerate(class_tokens)
    ]
    return np.ascontiguousarray(encoded), class_order


def _require_same_class_order(expected: np.ndarray, observed: np.ndarray) -> None:
    if expected.ndim != 1 or observed.ndim != 1 or expected.shape != observed.shape:
        raise RuntimeError("official TabPFN class order shape changed")
    expected_tokens = tuple(_canonical_label(value) for value in expected)
    observed_tokens = tuple(_canonical_label(value) for value in observed)
    if expected_tokens != observed_tokens:
        raise RuntimeError("official TabPFN class order changed during evaluation")


def _canonical_label(value: Any) -> bytes:
    scalar = value.item() if isinstance(value, np.generic) else value
    if isinstance(scalar, bool):
        payload = {"type": "bool", "value": scalar}
    elif isinstance(scalar, int):
        payload = {"type": "int", "value": str(scalar)}
    elif isinstance(scalar, float):
        if not math.isfinite(scalar):
            raise ValueError("class labels must not be non-finite")
        payload = {"type": "float", "value": scalar.hex()}
    elif isinstance(scalar, str):
        payload = {"type": "str", "value": scalar}
    elif isinstance(scalar, bytes):
        payload = {"type": "bytes", "value": scalar.hex()}
    else:
        raise TypeError(
            "class labels must be scalar bool/int/float/str/bytes values"
        )
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _probability_array(values: Any, *, expected_shape: tuple[int, int]) -> np.ndarray:
    source = np.asarray(values)
    if source.shape != expected_shape or not np.issubdtype(source.dtype, np.floating):
        raise RuntimeError("official TabPFN probabilities have the wrong shape/dtype")
    probabilities = np.ascontiguousarray(source, dtype=np.float32)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("stored TabPFN probabilities contain non-finite values")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise RuntimeError("stored TabPFN probabilities are outside [0, 1]")
    if not np.allclose(
        probabilities.sum(axis=1, dtype=np.float64),
        1.0,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError("stored TabPFN probability rows do not sum to one")
    return probabilities


def _metrics_from_stored_probabilities(
    probabilities: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    selected = probabilities[np.arange(labels.shape[0]), labels].astype(np.float64)
    return {
        "accuracy": float(np.mean(np.argmax(probabilities, axis=1) == labels)),
        "log_loss": float(
            -np.log(np.clip(selected, np.finfo(np.float64).tiny, 1.0)).mean()
        ),
        "n_samples": int(labels.shape[0]),
    }


def _require_driver_metrics(
    condition: TabPFNV26ConditionResult,
    *,
    metrics: Mapping[str, Any],
    expected_rows: int,
) -> None:
    if condition.n_samples != expected_rows:
        raise RuntimeError("official TabPFN metric sample count changed")
    if not math.isclose(
        float(condition.accuracy),
        float(metrics["accuracy"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("official TabPFN accuracy differs from stored predictions")
    if not math.isclose(
        float(condition.log_loss),
        float(metrics["log_loss"]),
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        raise RuntimeError("official TabPFN log loss differs from stored predictions")


def _paired_summary(
    datasets: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    if not datasets:
        raise RuntimeError("TabPFN paired summary requires at least one dataset")
    full_accuracy = [item["conditions"]["full"]["accuracy"] for item in datasets]
    full_loss = [item["conditions"]["full"]["log_loss"] for item in datasets]
    conditions: dict[str, Any] = {}
    for offset, component in enumerate(_COMPONENTS):
        accuracy = [item["conditions"][component]["accuracy"] for item in datasets]
        loss = [item["conditions"][component]["log_loss"] for item in datasets]
        conditions[component] = {
            "accuracy_effect_vs_full": summary_dict(
                summarize_paired(
                    full_accuracy,
                    accuracy,
                    higher_is_better=True,
                    seed=seed + offset,
                )
            ),
            "log_loss_effect_vs_full": summary_dict(
                summarize_paired(
                    full_loss,
                    loss,
                    higher_is_better=False,
                    seed=seed + 10_000 + offset,
                )
            ),
        }
    dataset_ids = [str(item["dataset_id"]) for item in datasets]
    return {
        "schema_version": 1,
        "kind": "official_tabpfn_v2.6_paired_dataset_summary",
        "evidence_scope": "exploratory-discovery-validation",
        "formal_eligible": False,
        "baseline_condition": "full",
        "effect_direction": "positive_means_condition_better_than_full",
        "assignment_split": "discovery",
        "evaluation_split": "val",
        "fit_context": "train",
        "dataset_count": len(datasets),
        "paired_dataset_order_sha256": _json_sha256(dataset_ids),
        "conditions": conditions,
    }


def _runtime_attestation(*, device: str) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for distribution in (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
        "tabpfn",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = "not-installed"
    import torch

    attestation: dict[str, Any] = {
        "schema_version": 1,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "packages": packages,
        "requested_device": device,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_num_threads": int(torch.get_num_threads()),
        "probability_storage_dtype": "float32",
        "label_storage_dtype": "int64",
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        attestation["cuda_device_name"] = str(torch.cuda.get_device_name(index))
        attestation["cuda_device_capability"] = list(
            torch.cuda.get_device_capability(index)
        )
    return attestation


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype.hasobject:
        raise TypeError("object arrays cannot be safely content-hashed")
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + contiguous.tobytes(order="C")).hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_path_free(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_path_free(str(key))
            _assert_path_free(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_path_free(child)
    elif isinstance(value, str) and Path(value).is_absolute():
        raise RuntimeError("published TabPFN metadata must not contain absolute paths")


__all__ = ["run", "run_tabpfn_localization"]
