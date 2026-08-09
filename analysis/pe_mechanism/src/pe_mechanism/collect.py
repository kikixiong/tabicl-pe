"""Bounded, deterministic publication of private activation shards."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .identifiers import require_portable_identifier
from .provenance import (
    RunTransaction,
    assert_dataset_roster,
    load_verified_json_config,
    manifest_from_verified_inputs,
    verify_configured_run_inputs,
)


DEFAULT_MAX_PRIVATE_BYTES = 30 * 1024**3
DEFAULT_MIN_FREE_BYTES = 20 * 1024**3
_PORTABLE_SITE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:\[\]-]{0,191}$")


class ActivationReservoir:
    """Uniform priority reservoir retaining at most ``capacity`` activation rows."""

    def __init__(self, capacity: int, *, seed: int, dtype: str = "float16") -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        resolved_dtype = np.dtype(dtype)
        if resolved_dtype.kind != "f":
            raise ValueError("activation dtype must be floating point")
        self.capacity = int(capacity)
        self.dtype = resolved_dtype
        self._rng = np.random.default_rng(int(seed))
        self._values: np.ndarray | None = None
        self._dataset_indices: np.ndarray | None = None
        self._priorities: np.ndarray | None = None
        self.seen = 0

    @property
    def feature_dim(self) -> int | None:
        return None if self._values is None else int(self._values.shape[1])

    @property
    def retained(self) -> int:
        return 0 if self._values is None else int(self._values.shape[0])

    def add(self, values: np.ndarray, *, dataset_index: int) -> None:
        array = np.asarray(values)
        if array.ndim < 2:
            raise ValueError("activation arrays must have an embedding axis")
        flattened = array.reshape(-1, array.shape[-1])
        if flattened.shape[0] == 0:
            return
        if not np.isfinite(flattened).all():
            raise ValueError("activation arrays must contain only finite values")
        if self.feature_dim is not None and flattened.shape[1] != self.feature_dim:
            raise ValueError(
                f"activation feature dimension changed from {self.feature_dim} to {flattened.shape[1]}"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            flattened = flattened.astype(self.dtype, copy=False)
        # A value can be finite in the source dtype but overflow when it is
        # narrowed (for example 1e20 -> float16 inf).  Check the bytes that will
        # actually be retained and published, not only the input array.
        if not np.isfinite(flattened).all():
            raise ValueError(
                f"activation arrays overflowed or became non-finite when cast to {self.dtype.name}"
            )
        priorities = self._rng.random(flattened.shape[0])
        dataset_indices = np.full(flattened.shape[0], int(dataset_index), dtype=np.int32)
        self.seen += flattened.shape[0]

        if self._values is not None:
            flattened = np.concatenate((self._values, flattened), axis=0)
            priorities = np.concatenate((self._priorities, priorities), axis=0)
            dataset_indices = np.concatenate((self._dataset_indices, dataset_indices), axis=0)
        if flattened.shape[0] > self.capacity:
            selected = np.argpartition(priorities, self.capacity - 1)[: self.capacity]
            flattened = flattened[selected]
            priorities = priorities[selected]
            dataset_indices = dataset_indices[selected]
        self._values = flattened
        self._priorities = priorities
        self._dataset_indices = dataset_indices

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self._values is None:
            raise ValueError("reservoir is empty")
        order = np.argsort(self._priorities, kind="stable")
        return self._values[order], self._dataset_indices[order]


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _publish_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_npz(path: Path, **arrays: np.ndarray) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _tree_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("activation output trees must not contain symlinks")
        if path.is_file():
            total += path.stat().st_size
    return total


def _check_budget(
    budget_root: Path,
    *,
    projected_bytes: int,
    max_private_bytes: int,
    min_free_bytes: int,
) -> None:
    ancestor = budget_root
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    if free - projected_bytes < min_free_bytes:
        raise RuntimeError("activation write would cross the free-space reserve")
    if _tree_bytes(budget_root) + projected_bytes > max_private_bytes:
        raise RuntimeError("activation write would cross the private study cap")


def _site_filename(site: str, dataset: str) -> str:
    digest = hashlib.sha256(f"{site}\0{dataset}".encode("utf-8")).hexdigest()[:16]
    return f"activation-{digest}.npz"


def _reservoir_seed(seed: int, site: str, dataset: str) -> int:
    raw = hashlib.sha256(f"{seed}\0{site}\0{dataset}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], byteorder="big", signed=False)


def collect_activation_files(
    config: Mapping[str, Any],
    output_dir: Path,
    *,
    arrays_by_record: Sequence[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Collect bounded samples from external ``.npy`` activation chunks.

    Model adapters may call :class:`ActivationReservoir` directly.  The file
    route is the stable CLI boundary and keeps private input paths out of the
    written index.
    """
    if config.get("schema_version") != 1:
        raise ValueError("collect config schema_version must be 1")
    records = config.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("collect config records must be a non-empty list")
    if "max_vectors_per_site" in config:
        raise ValueError(
            "max_vectors_per_site is ambiguous; use max_vectors_per_dataset_site"
        )
    capacity = int(config.get("max_vectors_per_dataset_site", 1_000_000))
    seed = int(config.get("seed", 42))
    dtype = str(config.get("dtype", "float16"))
    reservoirs: dict[tuple[str, str], ActivationReservoir] = {}
    site_metadata: dict[str, dict[str, Any]] = {}
    dataset_names: list[str] = []
    dataset_index: dict[str, int] = {}
    if arrays_by_record is not None and len(arrays_by_record) != len(records):
        raise ValueError("arrays_by_record must align exactly to activation records")

    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("each activation record must be an object")
        site = str(record.get("site", ""))
        if not _PORTABLE_SITE.fullmatch(site):
            raise ValueError(f"invalid portable site name: {site!r}")
        axes = record.get("axis_names")
        if not isinstance(axes, list) or not axes or not all(isinstance(axis, str) for axis in axes):
            raise ValueError("axis_names must be a non-empty string list")
        axes = [require_portable_identifier(axis, name="axis name") for axis in axes]
        source = Path(str(record.get("path", ""))).expanduser()
        if source.suffix.lower() != ".npy":
            raise ValueError("activation collection inputs must be .npy files")
        if arrays_by_record is None and (
            not source.is_absolute() or not source.is_file()
        ):
            raise ValueError("activation input paths must be existing absolute files")
        dataset = require_portable_identifier(record.get("dataset_id", ""), name="dataset_id")
        if dataset not in dataset_index:
            dataset_index[dataset] = len(dataset_names)
            dataset_names.append(dataset)
        array = (
            np.load(source, allow_pickle=False)
            if arrays_by_record is None
            else np.asarray(arrays_by_record[record_index])
        )
        if array.ndim != len(axes):
            raise ValueError(f"site {site!r} axis_names do not match activation rank")
        previous = site_metadata.setdefault(
            site,
            {
                "site": site,
                "axis_names": axes,
                "feature_group_maps": {},
            },
        )
        if previous["axis_names"] != axes:
            raise ValueError(f"site {site!r} axis semantics changed between records")
        feature_group_map = record.get("feature_group_map")
        if feature_group_map is not None:
            if not isinstance(feature_group_map, list) or not all(
                isinstance(group, list) and all(isinstance(index, int) for index in group)
                for group in feature_group_map
            ):
                raise ValueError("feature_group_map must be a list of integer lists")
            existing_map = previous["feature_group_maps"].get(dataset)
            if existing_map is not None and existing_map != feature_group_map:
                raise ValueError(
                    f"site {site!r} feature_group_map changed for dataset {dataset!r}"
                )
            previous["feature_group_maps"][dataset] = feature_group_map
        reservoir_key = (site, dataset)
        reservoir = reservoirs.setdefault(
            reservoir_key,
            ActivationReservoir(
                capacity,
                seed=_reservoir_seed(seed, site, dataset),
                dtype=dtype,
            ),
        )
        reservoir.add(array, dataset_index=dataset_index[dataset])

    projected = sum(
        reservoir.retained * (int(reservoir.feature_dim or 0) * reservoir.dtype.itemsize + 4)
        for reservoir in reservoirs.values()
    )
    # Include conservative container/index/manifest overhead.  Configuration may
    # tighten the study limits but can never relax the registered 30/20 GiB gates.
    projected += len(reservoirs) * (1 << 20) + (2 << 20)
    max_private_bytes = int(config.get("max_private_bytes", DEFAULT_MAX_PRIVATE_BYTES))
    min_free_bytes = int(config.get("min_free_bytes", DEFAULT_MIN_FREE_BYTES))
    if max_private_bytes <= 0 or max_private_bytes > DEFAULT_MAX_PRIVATE_BYTES:
        raise ValueError(
            f"max_private_bytes must be in (0, {DEFAULT_MAX_PRIVATE_BYTES}]"
        )
    if min_free_bytes < DEFAULT_MIN_FREE_BYTES:
        raise ValueError(
            f"min_free_bytes must be at least {DEFAULT_MIN_FREE_BYTES}"
        )
    raw_budget_root = config.get("private_study_root")
    budget_root = (
        output_dir.parent
        if raw_budget_root is None
        else Path(str(raw_budget_root)).expanduser()
    )
    if not budget_root.is_absolute():
        raise ValueError("private_study_root must be an absolute path")
    if budget_root.is_symlink():
        raise ValueError("private_study_root must be a real directory")
    budget_root = budget_root.resolve(strict=True)
    if not budget_root.is_dir():
        raise ValueError("private_study_root must be a real directory")
    resolved_output = output_dir.resolve(strict=False)
    if not resolved_output.is_relative_to(budget_root):
        raise ValueError("activation output must be inside private_study_root")
    _check_budget(
        budget_root,
        projected_bytes=projected,
        max_private_bytes=max_private_bytes,
        min_free_bytes=min_free_bytes,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    index_sites: list[dict[str, Any]] = []
    for site in sorted(site_metadata):
        metadata = site_metadata[site]
        dataset_entries: list[dict[str, Any]] = []
        for dataset in sorted(
            dataset for observed_site, dataset in reservoirs if observed_site == site
        ):
            reservoir = reservoirs[(site, dataset)]
            values, _ = reservoir.arrays()
            filename = _site_filename(site, dataset)
            _publish_npz(output_dir / filename, activations=values)
            dataset_entries.append(
                {
                    "dataset_id": dataset,
                    "file": filename,
                    "feature_dim": reservoir.feature_dim,
                    "dtype": reservoir.dtype.name,
                    "seen_vectors": reservoir.seen,
                    "retained_vectors": reservoir.retained,
                }
            )
        index_sites.append(
            {
                **metadata,
                "datasets": dataset_entries,
            }
        )
    index = {
        "schema_version": 1,
        "kind": "bounded_activation_index",
        "seed": seed,
        "datasets": sorted(dataset_names),
        "sites": index_sites,
    }
    _publish_json(output_dir / "activation-index.json", index)
    return index


def run(args: argparse.Namespace) -> int:
    configuration = load_verified_json_config(Path(args.config))
    config = dict(configuration.data)
    if "private_study_root" not in config:
        raise ValueError("collect config must define private_study_root")
    records = config.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("collect config records must be a non-empty list")
    sites: set[str] = set()
    dataset_ids: set[str] = set()
    additional_paths: dict[str, Path] = {}
    expected_hashes: dict[str, str] = {}
    verified_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("each activation record must be an object")
        site = str(record.get("site", ""))
        if not _PORTABLE_SITE.fullmatch(site):
            raise ValueError(f"invalid portable site name: {site!r}")
        sites.add(site)
        dataset_ids.add(
            require_portable_identifier(record.get("dataset_id", ""), name="dataset_id")
        )
        role = f"activation.{index:06d}"
        source = Path(str(record.get("path", ""))).expanduser()
        if not source.is_absolute():
            raise ValueError("activation input paths must be absolute")
        additional_paths[role] = source
        expected = record.get("expected_sha256")
        if expected is not None:
            if not isinstance(expected, str):
                raise ValueError("record expected_sha256 must be a string")
            expected_hashes[role] = expected
        verified_records.append(dict(record))

    seed = int(config.get("seed", 42))
    context = verify_configured_run_inputs(
        configuration,
        command="collect",
        seed=seed,
        additional_input_paths=additional_paths,
        expected_additional_sha256=expected_hashes,
    )
    if tuple(sorted(sites)) != tuple(sorted(context.sites)):
        raise ValueError("provenance sites must exactly match the activation record sites")
    assert_dataset_roster(
        context.inputs.dataset_manifest, tuple(sorted(dataset_ids))
    )
    for index, record in enumerate(verified_records):
        record["path"] = str(context.additional_file(f"activation.{index:06d}").path)
        record.pop("expected_sha256", None)
    verified_config = {**config, "records": verified_records}
    verified_arrays = [
        np.load(
            io.BytesIO(context.additional_file(f"activation.{index:06d}").read_bytes()),
            allow_pickle=False,
        )
        for index in range(len(verified_records))
    ]
    roots = (
        context.inputs.training_code.root,
        context.inputs.model_code.root,
        context.inputs.analysis_code.root,
    )
    with RunTransaction(Path(args.output_dir), source_roots=roots) as transaction:
        index = collect_activation_files(
            verified_config,
            transaction.staging_dir,
            arrays_by_record=verified_arrays,
        )
        artifact_names = ["activation-index.json"] + [
            str(dataset["file"])
            for site in index["sites"]
            for dataset in site["datasets"]
        ]
        artifacts = transaction.artifact_digests(artifact_names)
        manifest = manifest_from_verified_inputs(
            context.inputs,
            artifacts=artifacts,
        )
        transaction.commit(manifest, verified_inputs=context.inputs)
    return 0
