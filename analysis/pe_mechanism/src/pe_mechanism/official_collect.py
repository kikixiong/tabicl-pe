"""Leakage-safe activation collection through official TabICL inference.

The workflow in this module deliberately fits and calls the public sklearn
``TabICLClassifier`` rather than replaying its raw model.  Consequently,
mixed-type preprocessing, ensemble construction, class unshuffling, and
probability aggregation remain owned by the installed official implementation.

Runtime paths are accepted only through the private, hashed configuration.
Published artifacts contain logical dataset/file roles and measured content
hashes, never filesystem paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .adapters.base import ActivationRecord
from .adapters.tabicl import same_feature_group_map
from .collect import (
    DEFAULT_MAX_PRIVATE_BYTES,
    DEFAULT_MIN_FREE_BYTES,
    _check_budget,
    _publish_json,
    _publish_npz,
    _reservoir_seed,
    _site_filename,
)
from .identifiers import require_portable_identifier, require_public_label
from .official_tabicl import (
    OFFICIAL_INFERENCE_PROTOCOL,
    OfficialForwardMetadata,
    OfficialInferenceResult,
    OfficialTabICLDriver,
    RawTalentDataset,
    fit_official_talent_driver,
    load_raw_talent_splits,
    official_inference_contract_sha256,
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

DEFAULT_MAX_CLASSES = 10
DEFAULT_MAX_VECTORS_PER_DATASET_SITE = 100_000
_SPLITS = ("train", "val", "test")
_ARRAY_NAMES = tuple(
    f"{prefix}_{split}.npy" for prefix in ("N", "C", "y") for split in _SPLITS
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VIEW_ID = re.compile(
    r"^official-tabicl/([A-Za-z0-9][A-Za-z0-9_.-]{0,191})/raw-call-([0-9]{4,})$"
)
_ROOT_FIELDS = {
    "schema_version",
    "private_study_root",
    "datasets",
    "roster_split",
    "evaluation_split",
    "trusted_pickle",
    "seed",
    "max_classes",
    "max_vectors_per_dataset_site",
    "dtype",
    "device",
    "estimator_options",
    "max_private_bytes",
    "min_free_bytes",
    "provenance",
}


@dataclass(frozen=True)
class _DatasetSpec:
    dataset_id: str
    directory: Path
    expected_input_sha256: Mapping[str, str]
    ordinal: int


@dataclass(frozen=True)
class _CollectedDataset:
    metadata: Mapping[str, Any]
    site_entries: Mapping[str, Mapping[str, Any]]
    shard_arrays: Mapping[str, Mapping[str, np.ndarray]]


class _CoordinateReservoir:
    """Uniform priority reservoir with raw-call and axis coordinates.

    Sampling priorities depend only on ``seed`` and the order/shape of captured
    tensors, not their values.  Matched conditions with the same official call
    schedule therefore retain identical within-dataset coordinates.
    """

    def __init__(self, capacity: int, *, seed: int, dtype: str) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("reservoir capacity must be a positive integer")
        resolved_dtype = np.dtype(dtype)
        if resolved_dtype.kind != "f":
            raise ValueError("activation dtype must be floating point")
        self.capacity = capacity
        self.dtype = resolved_dtype
        self._rng = np.random.default_rng(seed)
        self._values: np.ndarray | None = None
        self._priorities: np.ndarray | None = None
        self._call_indices: np.ndarray | None = None
        self._coordinates: np.ndarray | None = None
        self.seen = 0

    @property
    def retained(self) -> int:
        return 0 if self._values is None else int(self._values.shape[0])

    @property
    def feature_dim(self) -> int | None:
        return None if self._values is None else int(self._values.shape[1])

    @property
    def coordinate_dim(self) -> int | None:
        return None if self._coordinates is None else int(self._coordinates.shape[1])

    def add(self, values: Any, *, call_index: int) -> None:
        if (
            isinstance(call_index, bool)
            or not isinstance(call_index, int)
            or call_index < 0
        ):
            raise ValueError("call_index must be a non-negative integer")
        array = _to_numpy(values)
        if array.ndim < 2:
            raise ValueError("captured activations must have an embedding axis")
        if any(dimension <= 0 for dimension in array.shape):
            raise ValueError("captured activation dimensions must be non-empty")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("captured activations must be numeric")
        if not np.isfinite(array).all():
            raise ValueError("captured activations must be finite")
        flattened = array.reshape(-1, array.shape[-1])
        if self.feature_dim is not None and flattened.shape[1] != self.feature_dim:
            raise ValueError("activation embedding dimension changed across raw calls")
        coordinate_dim = array.ndim - 1
        if self.coordinate_dim is not None and coordinate_dim != self.coordinate_dim:
            raise ValueError("activation rank changed across raw calls")

        new_priorities = self._rng.random(flattened.shape[0])
        old_count = self.retained
        if self._priorities is None:
            all_priorities = new_priorities
        else:
            all_priorities = np.concatenate((self._priorities, new_priorities))
        keep = min(self.capacity, all_priorities.shape[0])
        if keep == all_priorities.shape[0]:
            selected = np.arange(keep, dtype=np.int64)
        else:
            selected = np.argpartition(all_priorities, keep - 1)[:keep]

        old_mask = selected < old_count
        new_positions = selected[~old_mask] - old_count
        retained_values = np.empty((keep, flattened.shape[1]), dtype=self.dtype)
        retained_calls = np.empty(keep, dtype=np.int32)
        retained_coordinates = np.empty((keep, coordinate_dim), dtype=np.int64)
        if old_mask.any():
            assert self._values is not None
            assert self._call_indices is not None
            assert self._coordinates is not None
            old_positions = selected[old_mask]
            retained_values[old_mask] = self._values[old_positions]
            retained_calls[old_mask] = self._call_indices[old_positions]
            retained_coordinates[old_mask] = self._coordinates[old_positions]
        if new_positions.size:
            with np.errstate(over="ignore", invalid="ignore"):
                narrowed = flattened[new_positions].astype(self.dtype, copy=False)
            if not np.isfinite(narrowed).all():
                raise ValueError(
                    "captured activations overflowed or became non-finite in "
                    f"published dtype {self.dtype.name}"
                )
            retained_values[~old_mask] = narrowed
            retained_calls[~old_mask] = call_index
            unraveled = np.unravel_index(new_positions, array.shape[:-1])
            retained_coordinates[~old_mask] = np.column_stack(unraveled)

        self._values = retained_values
        self._call_indices = retained_calls
        self._coordinates = retained_coordinates
        self._priorities = all_priorities[selected]
        self.seen += int(flattened.shape[0])

    def arrays(self) -> Mapping[str, np.ndarray]:
        if self._values is None:
            raise ValueError("activation reservoir is empty")
        assert self._priorities is not None
        assert self._call_indices is not None
        assert self._coordinates is not None
        order = np.argsort(self._priorities, kind="stable")
        return {
            "activations": self._values[order],
            "call_index": self._call_indices[order],
            "axis_coordinates": self._coordinates[order],
        }


def run_official_collection(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    driver_factory: Callable[..., OfficialTabICLDriver] = fit_official_talent_driver,
) -> Mapping[str, Any]:
    """Collect bounded official TabICL activations and atomically publish them."""

    configuration = load_verified_json_config(config_path)
    config, specs = _validate_configuration(configuration)
    additional_paths, expected_hashes, role_maps = _dataset_input_bindings(specs)
    seed = _non_negative_integer(config.get("seed", 42), name="seed")
    context = verify_configured_run_inputs(
        configuration,
        command="collect",
        seed=seed,
        additional_input_paths=additional_paths,
        expected_additional_sha256=expected_hashes,
    )
    _validate_contract(context, config=config)
    roster_split = require_portable_identifier(
        config["roster_split"], name="roster_split"
    )
    dataset_ids = tuple(spec.dataset_id for spec in specs)
    assert_dataset_roster(
        context.inputs.dataset_manifest,
        dataset_ids,
        required_split=roster_split,
    )

    collected: list[_CollectedDataset] = []
    for spec in specs:
        raw_dataset = load_raw_talent_splits(
            spec.directory,
            trusted_pickle=bool(config.get("trusted_pickle", False)),
        )
        _assert_dataset_inputs(
            raw_dataset,
            spec=spec,
            context=context,
            role_map=role_maps[spec.dataset_id],
        )
        n_classes = _enforce_class_limit(
            raw_dataset, int(config.get("max_classes", DEFAULT_MAX_CLASSES))
        )
        driver = driver_factory(
            raw_dataset,
            context.inputs.checkpoint.path,
            context_split="train",
            device=str(config.get("device", "cpu")),
            model_sha=context.inputs.model_code.head_sha,
            estimator_options=_resolved_estimator_options(config),
            expected_source_root=context.inputs.model_code.root,
        )
        _validate_driver(driver, context=context, expected_n_classes=n_classes)
        collected.append(
            _collect_one_dataset(
                raw_dataset,
                driver,
                context=context,
                config=config,
                input_roles=role_maps[spec.dataset_id],
            )
        )
        _validate_driver(driver, context=context, expected_n_classes=n_classes)

    index, shard_arrays = _assemble_index(
        collected,
        context=context,
        config=config,
        roster_split=roster_split,
    )
    private_root = _private_study_root(config["private_study_root"])
    resolved_output = Path(output_dir).expanduser()
    if not resolved_output.is_absolute():
        raise ValueError("output_dir must be an absolute path")
    resolved_output = resolved_output.resolve(strict=False)
    if not resolved_output.is_relative_to(private_root):
        raise ValueError("official collection output must be inside private_study_root")
    _enforce_budget(private_root, config=config, shard_arrays=shard_arrays, index=index)

    roots = (
        context.inputs.training_code.root,
        context.inputs.model_code.root,
        context.inputs.analysis_code.root,
    )
    with RunTransaction(resolved_output, source_roots=roots) as transaction:
        _publish_json(transaction.staging_dir / "activation-index.json", index)
        for filename, arrays in sorted(shard_arrays.items()):
            _publish_npz(transaction.staging_dir / filename, **arrays)
        artifact_names = ("activation-index.json", *sorted(shard_arrays))
        artifacts = transaction.artifact_digests(artifact_names)
        manifest = manifest_from_verified_inputs(context.inputs, artifacts=artifacts)
        transaction.commit(manifest, verified_inputs=context.inputs)
    return index


def run(args: argparse.Namespace) -> int:
    run_official_collection(args.config, args.output_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect official TabICL activations on TALENT val/test splits"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


def _validate_configuration(
    configuration: VerifiedConfiguration,
) -> tuple[dict[str, Any], tuple[_DatasetSpec, ...]]:
    config = dict(configuration.data)
    unknown = sorted(set(config) - _ROOT_FIELDS)
    if unknown:
        raise ValueError(f"unknown official collect configuration fields: {unknown}")
    required = {
        "schema_version",
        "private_study_root",
        "datasets",
        "roster_split",
        "provenance",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing official collect configuration fields: {missing}")
    if config["schema_version"] != 1:
        raise ValueError("official collect schema_version must be 1")
    evaluation_split = config.get("evaluation_split", "val")
    if evaluation_split not in {"val", "test"}:
        raise ValueError(
            "evaluation_split must be 'val' or 'test'; train and train+val leak fit rows"
        )
    trusted_pickle = config.get("trusted_pickle", False)
    if not isinstance(trusted_pickle, bool):
        raise TypeError("trusted_pickle must be a JSON boolean")
    seed = _non_negative_integer(config.get("seed", 42), name="seed")
    if seed > np.iinfo(np.uint32).max:
        raise ValueError("seed must fit the official classifier's 32-bit random state")
    max_classes = _positive_integer(
        config.get("max_classes", DEFAULT_MAX_CLASSES), name="max_classes"
    )
    if max_classes > DEFAULT_MAX_CLASSES:
        raise ValueError(f"max_classes cannot exceed {DEFAULT_MAX_CLASSES}")
    _positive_integer(
        config.get(
            "max_vectors_per_dataset_site",
            DEFAULT_MAX_VECTORS_PER_DATASET_SITE,
        ),
        name="max_vectors_per_dataset_site",
    )
    dtype = np.dtype(str(config.get("dtype", "float16")))
    if dtype.kind != "f":
        raise ValueError("dtype must be floating point")
    if not isinstance(config.get("device", "cpu"), str):
        raise TypeError("device must be a string")
    estimator_options = config.get("estimator_options")
    if estimator_options is not None and not isinstance(estimator_options, Mapping):
        raise TypeError("estimator_options must be an object")
    if isinstance(estimator_options, Mapping) and "random_state" in estimator_options:
        random_state = estimator_options["random_state"]
        if (
            isinstance(random_state, bool)
            or not isinstance(random_state, int)
            or random_state != seed
        ):
            raise ValueError(
                "estimator_options.random_state must equal the collection seed"
            )
    max_private = _positive_integer(
        config.get("max_private_bytes", DEFAULT_MAX_PRIVATE_BYTES),
        name="max_private_bytes",
    )
    min_free = _positive_integer(
        config.get("min_free_bytes", DEFAULT_MIN_FREE_BYTES),
        name="min_free_bytes",
    )
    if max_private > DEFAULT_MAX_PRIVATE_BYTES:
        raise ValueError(f"max_private_bytes cannot exceed {DEFAULT_MAX_PRIVATE_BYTES}")
    if min_free < DEFAULT_MIN_FREE_BYTES:
        raise ValueError(f"min_free_bytes must be at least {DEFAULT_MIN_FREE_BYTES}")
    _private_study_root(config["private_study_root"])
    if not isinstance(config["roster_split"], str):
        raise TypeError("roster_split must be a string")
    require_portable_identifier(config["roster_split"], name="roster_split")

    raw_specs = config["datasets"]
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("datasets must be a non-empty list")
    preliminary: list[tuple[str, Path, Mapping[str, str]]] = []
    for raw in raw_specs:
        if not isinstance(raw, Mapping):
            raise TypeError("each dataset entry must be an object")
        unknown_dataset = sorted(
            set(raw) - {"dataset_id", "path", "expected_input_sha256"}
        )
        if unknown_dataset:
            raise ValueError(f"unknown dataset entry fields: {unknown_dataset}")
        if not {"dataset_id", "path"} <= set(raw):
            raise ValueError("each dataset entry requires dataset_id and path")
        if not isinstance(raw["dataset_id"], str):
            raise TypeError("dataset_id must be a string")
        if not isinstance(raw["path"], str):
            raise TypeError("TALENT dataset path must be a string")
        dataset_id = require_public_label(raw["dataset_id"], name="dataset_id")
        directory = Path(raw["path"]).expanduser()
        if not directory.is_absolute():
            raise ValueError("TALENT dataset paths must be absolute")
        directory = directory.resolve(strict=True)
        if not directory.is_dir():
            raise ValueError("TALENT dataset path must be a directory")
        if directory.name != dataset_id:
            raise ValueError("dataset_id must exactly match the TALENT directory name")
        raw_expected = raw.get("expected_input_sha256", {})
        if not isinstance(raw_expected, Mapping):
            raise TypeError("expected_input_sha256 must be an object")
        expected: dict[str, str] = {}
        for name, digest in raw_expected.items():
            if name not in {"info.json", *_ARRAY_NAMES}:
                raise ValueError(f"unknown TALENT logical input name: {name!r}")
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise ValueError("expected TALENT input hashes must be SHA-256 strings")
            expected[str(name)] = digest
        preliminary.append((dataset_id, directory, expected))
    ids = [item[0] for item in preliminary]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset IDs must be unique")
    directories = [item[1] for item in preliminary]
    if len(directories) != len(set(directories)):
        raise ValueError("TALENT dataset directories must be unique")
    ordered = sorted(preliminary, key=lambda item: (item[0].casefold(), item[0]))
    specs = tuple(
        _DatasetSpec(dataset_id, directory, expected, ordinal)
        for ordinal, (dataset_id, directory, expected) in enumerate(ordered)
    )
    return config, specs


def _dataset_input_bindings(
    specs: Sequence[_DatasetSpec],
) -> tuple[dict[str, Path], dict[str, str], dict[str, dict[str, str]]]:
    paths: dict[str, Path] = {}
    expected: dict[str, str] = {}
    role_maps: dict[str, dict[str, str]] = {}
    for spec in specs:
        names = ["info.json"] + [
            name for name in _ARRAY_NAMES if (spec.directory / name).is_file()
        ]
        if any(
            name not in names for name in ("y_train.npy", "y_val.npy", "y_test.npy")
        ):
            raise FileNotFoundError("TALENT dataset is missing a target split")
        unknown_expected = sorted(set(spec.expected_input_sha256) - set(names))
        if unknown_expected:
            raise ValueError(
                "expected_input_sha256 references files not consumed by the loader: "
                f"{unknown_expected}"
            )
        role_map: dict[str, str] = {}
        for name in names:
            token = name.removesuffix(".npy").replace(".", "_")
            role = f"talent.{spec.ordinal:04d}.{token}"
            paths[role] = spec.directory / name
            role_map[name] = role
            if name in spec.expected_input_sha256:
                expected[role] = spec.expected_input_sha256[name]
        role_maps[spec.dataset_id] = role_map
    return paths, expected, role_maps


def _validate_contract(
    context: VerifiedRunContext, *, config: Mapping[str, Any]
) -> None:
    if context.inputs.evidence_level != "strict":
        raise RuntimeError(
            "official activation collection requires strict Git evidence"
        )
    if context.model_family != "tabicl-v2":
        raise ValueError("official collection supports only model_family='tabicl-v2'")
    if context.condition not in {"rope", "temporary", "none"}:
        raise ValueError("condition must be exactly rope, temporary, or none")
    if not context.sites or len(set(context.sites)) != len(context.sites):
        raise ValueError("provenance sites must be a non-empty unique sequence")
    for site in context.sites:
        require_portable_identifier(site, name="activation site")
    provenance = config["provenance"]
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be an object")
    if provenance.get("allow_exploratory_legacy", False) is not False:
        raise ValueError("official collection forbids exploratory legacy provenance")


def _assert_dataset_inputs(
    dataset: RawTalentDataset,
    *,
    spec: _DatasetSpec,
    context: VerifiedRunContext,
    role_map: Mapping[str, str],
) -> None:
    if dataset.name != spec.dataset_id:
        raise RuntimeError("loaded TALENT dataset identity changed")
    if set(dataset.input_sha256) != set(role_map):
        raise RuntimeError("TALENT loader consumed an unexpected file roster")
    for logical_name, role in role_map.items():
        measured = context.additional_file(role).digest.sha256
        if dataset.input_sha256[logical_name] != measured:
            raise RuntimeError("TALENT bytes changed between provenance and loading")
    if dataset.info_sha256 != dataset.input_sha256["info.json"]:
        raise RuntimeError("TALENT info.json hash disagrees with the consumed inputs")


def _enforce_class_limit(dataset: RawTalentDataset, max_classes: int) -> int:
    count = len(
        {
            json.dumps(_label_token(value), sort_keys=True, separators=(",", ":"))
            for value in np.asarray(dataset.train.y).reshape(-1)
        }
    )
    if count < 2:
        raise ValueError("TALENT training split must contain at least two classes")
    if count > max_classes:
        raise ValueError(
            f"TALENT training split has {count} classes, above max_classes={max_classes}"
        )
    return count


def _validate_driver(
    driver: OfficialTabICLDriver,
    *,
    context: VerifiedRunContext,
    expected_n_classes: int,
) -> None:
    required = (
        "estimator",
        "model_sha",
        "checkpoint_sha",
        "fit_context",
        "source_evidence_level",
        "predict_proba",
    )
    missing = [name for name in required if not hasattr(driver, name)]
    if missing:
        raise TypeError(f"official driver is missing required attributes: {missing}")
    if driver.model_sha != context.inputs.model_code.head_sha:
        raise RuntimeError(
            "official driver model SHA differs from verified model source"
        )
    if driver.checkpoint_sha != context.inputs.checkpoint.digest.sha256:
        raise RuntimeError("official driver checkpoint SHA differs from verified input")
    if driver.fit_context != "talent-train":
        raise RuntimeError("official TALENT driver must fit only the train split")
    if driver.source_evidence_level != "strict":
        raise RuntimeError("official driver must retain strict source evidence")
    estimator = driver.estimator
    if getattr(estimator, "kv_cache", None) is not False:
        raise ValueError("official collection requires kv_cache=False")
    if getattr(estimator, "model_kv_cache_", None) is not None:
        raise ValueError("official collection refuses fitted classifier caches")
    if getattr(estimator, "support_many_classes", None) is not False:
        raise ValueError("official collection disables many-class recursion")
    raw = getattr(estimator, "model_", None)
    if raw is None:
        raise TypeError("official classifier has no fitted raw model")
    if getattr(raw, "_cache", None) is not None:
        raise ValueError("official collection refuses raw-model caches")
    if getattr(raw, "training", None) is not False:
        raise ValueError("official collection requires model.eval()")
    col = getattr(raw, "col_embedder", None)
    feature_group = getattr(col, "feature_group", None)
    if feature_group is not True and feature_group != "same":
        raise ValueError("official collection supports only feature_group='same'")
    row = getattr(raw, "row_interactor", None)
    raw_mode = getattr(raw, "row_identity_mode", None)
    row_mode = getattr(row, "identity_mode", raw_mode)
    if raw_mode != context.condition or row_mode != context.condition:
        raise RuntimeError(
            "checkpoint row identity mode differs from condition contract"
        )
    n_classes = int(getattr(estimator, "n_classes_", -1))
    if n_classes < 2 or n_classes > DEFAULT_MAX_CLASSES:
        raise ValueError(
            "fitted class count is outside the registered <=10-class scope"
        )
    if n_classes != expected_n_classes:
        raise RuntimeError("fitted class count differs from the TALENT train split")


def _collect_one_dataset(
    dataset: RawTalentDataset,
    driver: OfficialTabICLDriver,
    *,
    context: VerifiedRunContext,
    config: Mapping[str, Any],
    input_roles: Mapping[str, str],
) -> _CollectedDataset:
    evaluation_split = str(config.get("evaluation_split", "val"))
    split = getattr(dataset, evaluation_split)
    result = driver.predict_proba(
        split.X,
        y=split.y,
        sites=context.sites,
        require_exact_baseline=True,
    )
    _validate_result(
        result,
        dataset=dataset,
        expected_rows=int(np.asarray(split.y).shape[0]),
        expected_n_classes=int(driver.estimator.n_classes_),
        sites=context.sites,
    )
    calls = tuple(result.forward_calls)
    call_metadata = [_portable_forward_metadata(call.metadata) for call in calls]
    local_group_map: tuple[tuple[int, ...], ...] | None = None
    reservoirs = {
        site: _CoordinateReservoir(
            _positive_integer(
                config.get(
                    "max_vectors_per_dataset_site",
                    DEFAULT_MAX_VECTORS_PER_DATASET_SITE,
                ),
                name="max_vectors_per_dataset_site",
            ),
            seed=_reservoir_seed(int(config.get("seed", 42)), site, dataset.name),
            dtype=str(config.get("dtype", "float16")),
        )
        for site in context.sites
    }
    axis_names: dict[str, tuple[str, ...]] = {}
    activation_shapes: dict[str, list[list[int]]] = {
        site: [] for site in context.sites
    }
    token_semantics: dict[str, tuple[int | None, int | None]] = {}
    for expected_call_index, call in enumerate(calls):
        if call.metadata.call_index != expected_call_index:
            raise RuntimeError("official raw-call indices are not contiguous")
        for site in context.sites:
            record = call.activations[site]
            _validate_activation_record(
                record,
                metadata=call.metadata,
                driver=driver,
                site=site,
            )
            previous_axes = axis_names.setdefault(site, tuple(record.axis_names))
            if previous_axes != tuple(record.axis_names):
                raise RuntimeError("activation axis semantics changed across raw calls")
            activation_shapes[site].append([int(value) for value in record.shape])
            observed_semantics = _activation_token_semantics(record)
            previous_semantics = token_semantics.setdefault(site, observed_semantics)
            if previous_semantics != observed_semantics:
                raise RuntimeError(
                    "activation feature-group/CLS token semantics changed across raw calls"
                )
            if local_group_map is None:
                local_group_map = record.feature_group_map
            elif local_group_map != record.feature_group_map:
                raise RuntimeError(
                    "feature_group='same' mapping changed across raw calls"
                )
            reservoirs[site].add(record.tensor, call_index=expected_call_index)
    if local_group_map is None:
        raise RuntimeError("official capture did not expose a feature-group map")

    official_inputs = [
        {
            "logical_name": name,
            "role": input_roles[name],
            "sha256": dataset.input_sha256[name],
            "size_bytes": context.additional_file(input_roles[name]).digest.size_bytes,
        }
        for name in sorted(input_roles)
    ]
    trace_payload = {
        "schema_version": 1,
        "policy": "official-tabicl-train-only-evaluation-v1",
        "evaluation_split": evaluation_split,
        "model_sha": driver.model_sha,
        "checkpoint_sha256": driver.checkpoint_sha,
        "input_sha256": {
            name: dataset.input_sha256[name] for name in sorted(dataset.input_sha256)
        },
        "n_numeric_features": dataset.n_numeric_features,
        "n_categorical_features": dataset.n_categorical_features,
        "feature_group_mode": "same",
        "local_feature_group_map": [list(group) for group in local_group_map],
        "official_forward_calls": call_metadata,
    }
    preprocessing_trace_sha256 = _json_sha256(trace_payload)
    metrics = result.metrics
    assert metrics is not None
    probabilities = np.asarray(result.probabilities)
    dataset_metadata = {
        "dataset_id": dataset.name,
        "task_type": dataset.task_type,
        "assignment_split": config["roster_split"],
        "evaluation_split": evaluation_split,
        "fit_context": "train",
        "n_numeric_features": dataset.n_numeric_features,
        "n_categorical_features": dataset.n_categorical_features,
        "info_sha256": dataset.info_sha256,
        "inputs": official_inputs,
        "input_bundle_sha256": _json_sha256(
            {name: dataset.input_sha256[name] for name in sorted(dataset.input_sha256)}
        ),
        "preprocessing_trace_sha256": preprocessing_trace_sha256,
        "sample_roster_sha256": _sample_roster_sha256(
            dataset.name, evaluation_split, np.asarray(split.y)
        ),
        "probabilities_sha256": _array_sha256(probabilities),
        "metrics": {
            "accuracy": float(metrics.accuracy),
            "log_loss": float(metrics.log_loss),
            "n_samples": int(metrics.n_samples),
        },
        "exact_baseline_verified": True,
        "feature_group_mode": "same",
        "local_feature_group_map": [list(group) for group in local_group_map],
        "official_forward_calls": call_metadata,
    }
    site_entries: dict[str, Mapping[str, Any]] = {}
    shard_arrays: dict[str, Mapping[str, np.ndarray]] = {}
    for site in context.sites:
        reservoir = reservoirs[site]
        filename = _site_filename(site, dataset.name)
        arrays = reservoir.arrays()
        shard_arrays[filename] = arrays
        site_entries[site] = {
            "dataset_id": dataset.name,
            "file": filename,
            "axis_names": list(axis_names[site]),
            "activation_shapes": activation_shapes[site],
            "coordinate_axis_names": list(axis_names[site][:-1]),
            "vector_axis_name": axis_names[site][-1],
            "feature_group_token_offset": token_semantics[site][0],
            "cls_token_count": token_semantics[site][1],
            "feature_dim": reservoir.feature_dim,
            "dtype": reservoir.dtype.name,
            "seen_vectors": reservoir.seen,
            "retained_vectors": reservoir.retained,
            "preprocessing_trace_sha256": preprocessing_trace_sha256,
        }
    return _CollectedDataset(dataset_metadata, site_entries, shard_arrays)


def _activation_token_semantics(
    record: ActivationRecord,
) -> tuple[int | None, int | None]:
    axes = tuple(record.axis_names)
    shape = tuple(int(value) for value in record.shape)
    if "feature_group_or_cls" in axes:
        token_axis = axes.index("feature_group_or_cls")
        if record.feature_group_map is None:
            raise RuntimeError("feature-group activation omitted its exact group map")
        group_count = len(record.feature_group_map)
        cls_count = shape[token_axis] - group_count
        if cls_count < 0:
            raise RuntimeError(
                "feature-group activation has fewer tokens than feature groups"
            )
        return cls_count, cls_count
    if "cls" in axes:
        cls_axis = axes.index("cls")
        return None, shape[cls_axis]
    return None, None


def _validate_result(
    result: OfficialInferenceResult,
    *,
    dataset: RawTalentDataset,
    expected_rows: int,
    expected_n_classes: int,
    sites: Sequence[str],
) -> None:
    if not result.exact_baseline_verified:
        raise RuntimeError("official capture was not exact to its direct baseline")
    if result.source_evidence_level != "strict":
        raise RuntimeError("official inference lost strict source evidence")
    probabilities = np.asarray(result.probabilities)
    baseline = np.asarray(result.baseline_probabilities)
    if not np.array_equal(probabilities, baseline):
        raise RuntimeError("official capture probabilities differ from direct baseline")
    if probabilities.ndim != 2 or probabilities.shape[0] != expected_rows:
        raise RuntimeError("official probabilities have an unexpected shape")
    if probabilities.shape[1] != expected_n_classes:
        raise RuntimeError(
            "official probability class count differs from fitted classes"
        )
    classes = np.asarray(result.classes)
    if classes.ndim != 1 or classes.shape[0] != expected_n_classes:
        raise RuntimeError("official class labels differ from the fitted class count")
    if not np.isfinite(probabilities).all():
        raise RuntimeError("official probabilities are non-finite")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-7):
        raise RuntimeError("official probabilities do not sum to one")
    if result.metrics is None or result.metrics.n_samples != expected_rows:
        raise RuntimeError("official metrics are missing or have a wrong sample count")
    if not math.isfinite(result.metrics.accuracy) or not math.isfinite(
        result.metrics.log_loss
    ):
        raise RuntimeError("official metrics must be finite")
    if not result.forward_calls:
        raise RuntimeError("official activation capture returned no raw-model calls")
    for call in result.forward_calls:
        if set(call.activations) != set(sites):
            raise RuntimeError("official capture site roster differs from provenance")
    if dataset.task_type.strip().lower() not in {
        "binclass",
        "binary",
        "binary_classification",
        "classification",
        "multiclass",
        "multiclass_classification",
    }:
        raise ValueError("official collection supports only classification datasets")


def _validate_activation_record(
    record: ActivationRecord,
    *,
    metadata: OfficialForwardMetadata,
    driver: OfficialTabICLDriver,
    site: str,
) -> None:
    if record.site != site:
        raise RuntimeError("captured activation site differs from requested site")
    if (
        record.model_sha != driver.model_sha
        or record.checkpoint_sha != driver.checkpoint_sha
    ):
        raise RuntimeError(
            "captured activation provenance differs from the fitted driver"
        )
    if record.preprocessing_view_id != metadata.preprocessing_view_id:
        raise RuntimeError(
            "captured activation view differs from official call metadata"
        )
    if not record.axis_names or record.axis_names[-1] != "embedding":
        raise RuntimeError("captured activation must end in an embedding axis")
    for axis in record.axis_names:
        require_portable_identifier(axis, name="activation axis")
    array = _to_numpy(record.tensor)
    if tuple(array.shape) != tuple(record.shape):
        raise RuntimeError("captured activation shape metadata is stale")
    if array.ndim != len(record.axis_names):
        raise RuntimeError("captured activation rank differs from axis metadata")
    if record.feature_group_map is None:
        raise RuntimeError("feature_group='same' capture omitted its exact group map")
    col = driver.estimator.model_.col_embedder
    group_size = int(getattr(col, "feature_group_size", 0))
    expected_group_map = same_feature_group_map(metadata.raw_input_shape[2], group_size)
    if record.feature_group_map != expected_group_map:
        raise RuntimeError(
            "captured feature-group map differs from feature_group='same'"
        )


def _portable_forward_metadata(metadata: OfficialForwardMetadata) -> dict[str, Any]:
    norm_method = require_portable_identifier(
        metadata.norm_method, name="official normalization method"
    )
    match = _VIEW_ID.fullmatch(metadata.preprocessing_view_id)
    if match is None or match.group(1) != norm_method:
        raise RuntimeError("official preprocessing view ID has an unexpected form")
    if int(match.group(2)) != metadata.call_index:
        raise RuntimeError("official preprocessing view ID disagrees with call_index")
    if metadata.feature_coordinate_system != (
        "official-encoded-after-constant-filter-before-view-shuffle"
    ):
        raise RuntimeError("official feature coordinate system changed")
    if len(metadata.post_filter_feature_group_maps) != metadata.raw_input_shape[0]:
        raise RuntimeError("official per-table feature-group maps have a wrong count")
    return {
        "call_index": int(metadata.call_index),
        "norm_method": norm_method,
        "norm_view_indices": list(metadata.norm_view_indices),
        "ensemble_indices": list(metadata.ensemble_indices),
        "feature_shuffles": [list(value) for value in metadata.feature_shuffles],
        "class_shuffles": [list(value) for value in metadata.class_shuffles],
        "raw_input_shape": list(metadata.raw_input_shape),
        "train_size": int(metadata.train_size),
        "preprocessing_view_id": metadata.preprocessing_view_id,
        "post_filter_feature_group_maps": [
            [list(group) for group in table]
            for table in metadata.post_filter_feature_group_maps
        ],
        "feature_coordinate_system": metadata.feature_coordinate_system,
    }


def _assemble_index(
    collected: Sequence[_CollectedDataset],
    *,
    context: VerifiedRunContext,
    config: Mapping[str, Any],
    roster_split: str,
) -> tuple[dict[str, Any], dict[str, Mapping[str, np.ndarray]]]:
    datasets = [dict(item.metadata) for item in collected]
    site_axes: dict[str, list[str]] = {}
    site_datasets: dict[str, list[Mapping[str, Any]]] = {
        site: [] for site in context.sites
    }
    shards: dict[str, Mapping[str, np.ndarray]] = {}
    for item in collected:
        for site, entry in item.site_entries.items():
            axes = list(entry["axis_names"])
            previous = site_axes.setdefault(site, axes)
            if previous != axes:
                raise RuntimeError("activation axis semantics changed between datasets")
            site_datasets[site].append(entry)
        for filename, arrays in item.shard_arrays.items():
            if filename in shards:
                raise RuntimeError("activation shard filename collision")
            shards[filename] = arrays
    index = {
        "schema_version": 1,
        "kind": "official_tabicl_bounded_activation_index",
        "model_family": context.model_family,
        "model_revision": context.model_revision,
        "condition": context.condition,
        "checkpoint_sha256": context.inputs.checkpoint.digest.sha256,
        "model_code_sha": context.inputs.model_code.head_sha,
        "seed": int(config.get("seed", 42)),
        "assignment_split": roster_split,
        "evaluation_split": str(config.get("evaluation_split", "val")),
        "fit_context": "train",
        "inference_contract_sha256": _inference_contract_sha256(
            context=context,
            config=config,
        ),
        "max_classes": int(config.get("max_classes", DEFAULT_MAX_CLASSES)),
        "datasets": datasets,
        "sites": [
            {
                "site": site,
                "axis_names": site_axes[site],
                "datasets": sorted(
                    site_datasets[site],
                    key=lambda entry: (
                        str(entry["dataset_id"]).casefold(),
                        str(entry["dataset_id"]),
                    ),
                ),
            }
            for site in context.sites
        ],
    }
    return index, shards


def _enforce_budget(
    private_root: Path,
    *,
    config: Mapping[str, Any],
    shard_arrays: Mapping[str, Mapping[str, np.ndarray]],
    index: Mapping[str, Any],
) -> None:
    array_bytes = sum(
        int(array.nbytes)
        for arrays in shard_arrays.values()
        for array in arrays.values()
    )
    index_bytes = len(
        (json.dumps(index, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    )
    projected = array_bytes + index_bytes + len(shard_arrays) * (1 << 20) + (2 << 20)
    _check_budget(
        private_root,
        projected_bytes=projected,
        max_private_bytes=int(
            config.get("max_private_bytes", DEFAULT_MAX_PRIVATE_BYTES)
        ),
        min_free_bytes=int(config.get("min_free_bytes", DEFAULT_MIN_FREE_BYTES)),
    )


def _private_study_root(value: Any) -> Path:
    root = Path(str(value)).expanduser()
    if not root.is_absolute():
        raise ValueError("private_study_root must be absolute")
    if root.is_symlink():
        raise ValueError("private_study_root must be a real directory")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("private_study_root must be a real directory")
    return root


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "to"):
        value = value.to(device="cpu")
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + contiguous.tobytes(order="C")).hexdigest()


def _sample_roster_sha256(
    dataset_id: str, evaluation_split: str, labels: np.ndarray
) -> str:
    flattened = np.asarray(labels).reshape(-1)
    return _json_sha256(
        {
            "dataset_id": dataset_id,
            "evaluation_split": evaluation_split,
            "row_indices": list(range(int(flattened.shape[0]))),
            "labels": [_label_token(value) for value in flattened],
        }
    )


def _label_token(value: Any) -> Mapping[str, Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return {"type": "none", "value": None}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("classification labels must be finite")
        return {"type": "float", "value": value.hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    raise TypeError(f"unsupported classification label type: {type(value).__name__}")


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _non_negative_integer(value: Any, *, name: str) -> int:
    resolved = _integer(value, name=name)
    if resolved < 0:
        raise ValueError(f"{name} must be non-negative")
    return resolved


def _positive_integer(value: Any, *, name: str) -> int:
    resolved = _integer(value, name=name)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive")
    return resolved


def _resolved_estimator_options(config: Mapping[str, Any]) -> dict[str, Any]:
    options = dict(config.get("estimator_options") or {})
    seed = _non_negative_integer(config.get("seed", 42), name="seed")
    configured = options.get("random_state", seed)
    if configured != seed:
        raise ValueError("official classifier random_state must equal the collection seed")
    options["random_state"] = seed
    return options


def _inference_contract_sha256(
    *, context: VerifiedRunContext, config: Mapping[str, Any]
) -> str:
    return official_inference_contract_sha256(
        context.inputs.model_code.head_sha,
        _resolved_estimator_options(config),
    )


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MAX_CLASSES",
    "DEFAULT_MAX_VECTORS_PER_DATASET_SITE",
    "OFFICIAL_INFERENCE_PROTOCOL",
    "build_parser",
    "main",
    "run",
    "run_official_collection",
]
