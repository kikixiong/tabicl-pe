"""Canonical provenance and fail-closed validation for formal pre-training.

The self-hashes in this module detect corruption.  They are not trust roots: a
formal validator must also receive expected digests from outside the mutable
checkpoint/archive.  Exact parent-checkpoint bytes are the authority for model
weights because a manifest embedded in the same file cannot authenticate them.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import re
import secrets
import stat
import subprocess
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
import numpy as np


MANIFEST_SCHEMA_VERSION = 1
PROVENANCE_SCHEMA_VERSION = 1
FORMAL_MODES = frozenset({"rope", "temporary", "none"})
FORMAL_STAGES = frozenset({"stage1", "stage2", "stage3"})
ARCHITECTURE_TREATMENT_FIELD = "row_identity_mode"
TREATMENT_CONFIG_FIELDS = frozenset({ARCHITECTURE_TREATMENT_FIELD})
# These are the only run-config fields removed from the scientific manifest.
# Runtime choices such as world size, precision, compilation, FA3, recompute,
# prior device and worker count intentionally remain scientific.
OPERATIONAL_CONFIG_FIELDS = frozenset(
    {"checkpoint_dir", "checkpoint_path", "wandb_dir", "wandb_name", "wandb_id"}
)
FORMAL_PROTOCOL_CONFIG_FIELDS = frozenset(
    {
        "formal_training",
        "formal_stage",
        "formal_source_manifest",
        "formal_source_sha256",
        "formal_source_commit_sha",
        "formal_source_tree_sha",
        "formal_environment_sha256",
        "formal_study_id",
        "formal_output_id",
        "formal_transaction_ledger",
        "formal_transaction_ledger_sha256",
        "formal_parent_finalized_manifest",
        "formal_parent_stage",
        "formal_parent_upstream_identity",
        "formal_parent_artifact_identity",
        "formal_artifact_root",
    }
)
REQUIRED_MANIFESTS = frozenset(
    {
        "source",
        "environment",
        "architecture",
        "prior",
        "optimizer",
        "stage",
        "seed",
        "parent",
        "treatment",
        "scientific_config",
        "operational_config",
        "cohort_protocol",
        "arm_protocol",
    }
)
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SOURCE_MODES = frozenset({"100644", "100755", "120000"})


def _normalize_json(value: Any, *, where: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where} must contain only finite floats")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{where} object keys must be strings")
            if "\x00" in key:
                raise ValueError(f"{where} object keys must not contain NUL")
            normalized[key] = _normalize_json(item, where=f"{where}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json(item, where=f"{where}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{where} contains unsupported type {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one accepted UTF-8 JSON representation for a value."""
    normalized = _normalize_json(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, where: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{where} keys mismatch; missing={missing}, extra={extra}")


def _require_digest(name: str, value: Any) -> str:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def make_manifest(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(kind, str) or not kind:
        raise ValueError("manifest kind must be a non-empty string")
    body = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": kind,
        "payload": _normalize_json(payload, where=f"{kind}.payload"),
    }
    return {**body, "sha256": canonical_sha256(body)}


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_kind: str | None = None,
    expected_sha256: str | None = None,
) -> str:
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be an object")
    _require_exact_keys(
        manifest,
        {"schema_version", "kind", "payload", "sha256"},
        where="manifest",
    )
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("manifest schema_version mismatch")
    if not isinstance(manifest["kind"], str) or not isinstance(
        manifest["payload"], Mapping
    ):
        raise ValueError("manifest kind/payload is invalid")
    if expected_kind is not None and manifest["kind"] != expected_kind:
        raise ValueError(
            f"manifest kind mismatch: expected {expected_kind!r}, got {manifest['kind']!r}"
        )
    body = {
        "schema_version": manifest["schema_version"],
        "kind": manifest["kind"],
        "payload": manifest["payload"],
    }
    actual = canonical_sha256(body)
    if manifest["sha256"] != actual:
        raise ValueError(f"{manifest['kind']} manifest sha256 mismatch")
    if expected_sha256 is not None and actual != _require_digest(
        f"expected {manifest['kind']} sha256", expected_sha256
    ):
        raise ValueError(
            f"{manifest['kind']} manifest does not match external expected sha256"
        )
    return actual


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _open_regular_nofollow(path: str | os.PathLike[str]) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(os.fspath(path), flags)
    except OSError as error:
        raise ValueError(
            f"refusing to open non-regular or symlink path: {path}"
        ) from error
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        os.close(fd)
        raise ValueError(f"path must be a regular file: {path}")
    return fd, before


def _same_file_snapshot(
    before: os.stat_result, after: os.stat_result, *, where: str
) -> None:
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise ValueError(f"{where} changed while it was being read")


def strict_json_load_nofollow(
    path: str | os.PathLike[str], *, max_bytes: int = 32 << 20
) -> tuple[Any, str]:
    """Hash and strictly decode one no-follow regular-file snapshot."""
    fd, before = _open_regular_nofollow(path)
    try:
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > max_bytes:
            raise ValueError(f"JSON file exceeds {max_bytes} bytes: {path}")
        after = os.fstat(fd)
        _same_file_snapshot(before, after, where=str(path))
    finally:
        os.close(fd)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid strict JSON in {path}: {error}") from error
    return value, hashlib.sha256(raw).hexdigest()


def _validate_relative_path(path: Any, *, where: str) -> str:
    if not isinstance(path, str) or not path or "\x00" in path or "\\" in path:
        raise ValueError(f"{where} path is invalid")
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{where} path must be normalized and relative")
    normalized = pure.as_posix()
    if normalized != path:
        raise ValueError(f"{where} path must be normalized")
    return normalized


def _source_bytes(root: Path, relative: str, mode: str) -> bytes:
    path = root.joinpath(*PurePosixPath(relative).parts)
    if mode == "120000":
        if not path.is_symlink():
            raise ValueError(
                f"tracked source mode mismatch for {relative}: expected symlink"
            )
        target = os.readlink(path)
        return os.fsencode(target)
    if path.is_symlink():
        raise ValueError(
            f"tracked source path unexpectedly became a symlink: {relative}"
        )
    fd, before = _open_regular_nofollow(path)
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        _same_file_snapshot(before, after, where=relative)
    finally:
        os.close(fd)
    executable = bool(before.st_mode & stat.S_IXUSR)
    if executable != (mode == "100755"):
        raise ValueError(f"tracked source executable mode mismatch for {relative}")
    return b"".join(chunks)


def build_source_manifest_from_files(
    root: str | os.PathLike[str],
    *,
    commit_sha: str,
    tree_sha: str,
    tracked: Mapping[str, str],
    code_roots: Sequence[str] = ("src/tabicl", "scripts"),
) -> dict[str, Any]:
    root_path = Path(root).resolve(strict=True)
    if _HEX_40.fullmatch(commit_sha) is None or _HEX_40.fullmatch(tree_sha) is None:
        raise ValueError(
            "source commit/tree must be lowercase 40-character Git object IDs"
        )
    entries = []
    seen: set[str] = set()
    for raw_path, mode in tracked.items():
        relative = _validate_relative_path(raw_path, where="tracked source")
        if relative in seen:
            raise ValueError(f"duplicate tracked source path: {relative}")
        seen.add(relative)
        if mode not in _SOURCE_MODES:
            raise ValueError(f"unsupported tracked source mode {mode!r}")
        raw = _source_bytes(root_path, relative, mode)
        entries.append(
            {
                "path": relative,
                "mode": mode,
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    roots = sorted(
        {_validate_relative_path(path, where="code root") for path in code_roots}
    )
    payload = {
        "commit_sha": commit_sha,
        "tree_sha": tree_sha,
        "code_roots": roots,
        "entries": sorted(entries, key=lambda item: item["path"]),
    }
    return make_manifest("source", payload)


def build_git_source_manifest(
    repo: str | os.PathLike[str],
    *,
    commit_sha: str,
    code_roots: Sequence[str] = ("src/tabicl", "scripts"),
) -> dict[str, Any]:
    """Build the expected source manifest from immutable Git objects."""
    repo_path = Path(repo)
    commit = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", f"{commit_sha}^{{commit}}"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", f"{commit}^{{tree}}"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    listing = subprocess.run(
        ["git", "-C", str(repo_path), "ls-tree", "-rz", "--full-tree", commit],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    entries = []
    for record in listing.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode_b, object_type, object_id = metadata.split(b" ", 2)
        if object_type != b"blob":
            continue
        path = raw_path.decode("utf-8", errors="strict")
        mode = mode_b.decode("ascii")
        if mode not in _SOURCE_MODES:
            raise ValueError(f"unsupported Git source mode {mode!r} for {path}")
        blob = subprocess.run(
            [
                "git",
                "-C",
                str(repo_path),
                "cat-file",
                "blob",
                object_id.decode("ascii"),
            ],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        entries.append(
            {
                "path": _validate_relative_path(path, where="Git source"),
                "mode": mode,
                "size": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        )
    return make_manifest(
        "source",
        {
            "commit_sha": commit,
            "tree_sha": tree,
            "code_roots": sorted(
                {
                    _validate_relative_path(path, where="code root")
                    for path in code_roots
                }
            ),
            "entries": sorted(entries, key=lambda item: item["path"]),
        },
    )


def validate_source_manifest(manifest: Mapping[str, Any]) -> None:
    validate_manifest(manifest, expected_kind="source")
    payload = manifest["payload"]
    _require_exact_keys(
        payload,
        {"commit_sha", "tree_sha", "code_roots", "entries"},
        where="source payload",
    )
    if _HEX_40.fullmatch(str(payload["commit_sha"])) is None:
        raise ValueError("source commit_sha is invalid")
    if _HEX_40.fullmatch(str(payload["tree_sha"])) is None:
        raise ValueError("source tree_sha is invalid")
    if not isinstance(payload["code_roots"], list) or payload["code_roots"] != sorted(
        set(payload["code_roots"])
    ):
        raise ValueError("source code_roots must be sorted and unique")
    for root in payload["code_roots"]:
        _validate_relative_path(root, where="source code root")
    if not isinstance(payload["entries"], list):
        raise ValueError("source entries must be a list")
    seen: set[str] = set()
    last = ""
    for entry in payload["entries"]:
        if not isinstance(entry, Mapping):
            raise ValueError("source entry must be an object")
        _require_exact_keys(
            entry, {"path", "mode", "size", "sha256"}, where="source entry"
        )
        path = _validate_relative_path(entry["path"], where="source entry")
        if path in seen:
            raise ValueError(f"duplicate source manifest path: {path}")
        if last and path < last:
            raise ValueError("source entries must be sorted by path")
        seen.add(path)
        last = path
        if entry["mode"] not in _SOURCE_MODES:
            raise ValueError(f"source entry mode is invalid for {path}")
        if (
            isinstance(entry["size"], bool)
            or not isinstance(entry["size"], int)
            or entry["size"] < 0
        ):
            raise ValueError(f"source entry size is invalid for {path}")
        _require_digest(f"source entry {path}", entry["sha256"])


def load_source_manifest(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    expected_commit_sha: str,
    expected_tree_sha: str,
) -> dict[str, Any]:
    value, _raw_sha = strict_json_load_nofollow(path)
    if not isinstance(value, dict):
        raise ValueError("source manifest file must contain an object")
    validate_source_manifest(value)
    validate_manifest(value, expected_kind="source", expected_sha256=expected_sha256)
    payload = value["payload"]
    if payload["commit_sha"] != expected_commit_sha:
        raise ValueError("source commit does not match external expected commit")
    if payload["tree_sha"] != expected_tree_sha:
        raise ValueError("source tree does not match external expected tree")
    return value


def runtime_environment_manifest() -> dict[str, Any]:
    """Describe reproducibility-relevant runtime versions without local paths."""
    cudnn_version = (
        torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    )
    return make_manifest(
        "environment",
        {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform_system": platform.system(),
            "platform_release": platform.release(),
            "platform_machine": platform.machine(),
            "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": cudnn_version,
            "visible_cuda_device_count": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
        },
    )


def state_dict_schema(state_dict: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("state_dict must be a non-empty mapping")
    entries = []
    for key in sorted(state_dict):
        value = state_dict[key]
        if not isinstance(key, str) or not key or not isinstance(value, Tensor):
            raise ValueError("state_dict must map non-empty string keys to tensors")
        entries.append(
            {
                "key": key,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "layout": str(value.layout),
            }
        )
    return entries


def architecture_manifest(
    model_config: Mapping[str, Any], state_dict: Mapping[str, Any]
) -> dict[str, Any]:
    if ARCHITECTURE_TREATMENT_FIELD not in model_config:
        raise ValueError("model config is missing row_identity_mode")
    architecture = {
        key: value
        for key, value in model_config.items()
        if key != ARCHITECTURE_TREATMENT_FIELD
    }
    return make_manifest(
        "architecture",
        {
            "model_config": architecture,
            "state_dict_schema": state_dict_schema(state_dict),
        },
    )


def _strict_json_value(raw: str, *, where: str) -> Any:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{where} is not valid JSON") from error
    if canonical_json_bytes(value).decode("utf-8") != raw:
        raise ValueError(f"{where} must use canonical JSON")
    return value


def prior_manifest(prior_stream: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "algorithm",
        "schema",
        "schema_sha256",
        "experiment_seed",
        "ddp_rank",
        "world_size",
        "cursor",
        "manifest_sha256",
    }
    _require_exact_keys(prior_stream, required, where="logical prior stream")
    schema = prior_stream["schema"]
    if not isinstance(schema, str):
        raise ValueError("logical prior schema must be a string")
    _strict_json_value(schema, where="logical prior schema")
    if (
        hashlib.sha256(schema.encode("utf-8")).hexdigest()
        != prior_stream["schema_sha256"]
    ):
        raise ValueError("logical prior schema_sha256 mismatch")
    if (
        prior_stream["schema_version"] != 1
        or prior_stream["algorithm"] != "sha256-schema-seed-rank-logical-step-v1"
    ):
        raise ValueError("formal prior must be the supported logical on-the-fly stream")
    for name in ("experiment_seed", "ddp_rank", "world_size", "cursor"):
        value = prior_stream[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"logical prior {name} must be an integer")
    if prior_stream["world_size"] < 1 or prior_stream["cursor"] < 0:
        raise ValueError("logical prior world_size/cursor is invalid")
    payload_without_hash = {
        key: value for key, value in prior_stream.items() if key != "manifest_sha256"
    }
    internal = hashlib.sha256(
        repr(
            {key: payload_without_hash[key] for key in sorted(payload_without_hash)}
        ).encode("utf-8")
    ).hexdigest()
    if internal != prior_stream["manifest_sha256"]:
        raise ValueError("logical prior stream manifest hash mismatch")
    return make_manifest(
        "prior",
        {
            "schema_version": prior_stream["schema_version"],
            "algorithm": prior_stream["algorithm"],
            "schema": schema,
            "schema_sha256": prior_stream["schema_sha256"],
            "experiment_seed": prior_stream["experiment_seed"],
            "world_size": prior_stream["world_size"],
        },
    )


def partition_run_config(
    run_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    normalized = _normalize_json(run_config, where="run config")
    scientific = {
        key: value
        for key, value in normalized.items()
        if key not in OPERATIONAL_CONFIG_FIELDS and key not in TREATMENT_CONFIG_FIELDS
    }
    operational = {
        key: value
        for key, value in normalized.items()
        if key in OPERATIONAL_CONFIG_FIELDS
    }
    treatment = {
        key: value
        for key, value in normalized.items()
        if key in TREATMENT_CONFIG_FIELDS
    }
    return scientific, operational, treatment


def _validate_operational_context(
    context: Mapping[str, Any], *, mode: str | None = None
) -> None:
    _require_exact_keys(
        context, {"study_id", "arm", "output_id"}, where="operational context"
    )
    for name in ("study_id", "output_id"):
        if (
            not isinstance(context[name], str)
            or _SAFE_ID.fullmatch(context[name]) is None
        ):
            raise ValueError(f"operational {name} is invalid")
    if context["arm"] not in FORMAL_MODES:
        raise ValueError("operational arm is invalid")
    if mode is not None and context["arm"] != mode:
        raise ValueError("operational arm does not match treatment mode")
    if context["arm"] not in context["output_id"]:
        raise ValueError("operational output_id must identify its arm")


def _provenance_body(manifests: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "manifest_sha256": {
            name: manifests[name]["sha256"] for name in sorted(manifests)
        },
    }


def build_checkpoint_provenance(
    *,
    source_manifest: Mapping[str, Any],
    environment: Mapping[str, Any],
    model_config: Mapping[str, Any],
    state_dict: Mapping[str, Any],
    prior_stream: Mapping[str, Any],
    optimizer_config: Mapping[str, Any],
    stage: str,
    terminal_step: int,
    np_seed: int,
    torch_seed: int,
    identity_seed: int,
    world_size: int,
    identity_treatment: Mapping[str, Any],
    run_config: Mapping[str, Any],
    operational_context: Mapping[str, Any],
    parent_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    validate_source_manifest(source_manifest)
    if stage not in FORMAL_STAGES:
        raise ValueError("formal stage must be stage1, stage2, or stage3")
    for name, value, minimum in (
        ("terminal_step", terminal_step, 1),
        ("np_seed", np_seed, 0),
        ("torch_seed", torch_seed, 0),
        ("identity_rng_seed", identity_seed, 0),
        ("world_size", world_size, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    mode = identity_treatment.get("row_identity_mode")
    if mode not in FORMAL_MODES:
        raise ValueError("identity treatment mode is invalid")
    from tabicl.train._identity_rng import validate_identity_treatment

    validate_identity_treatment(
        dict(identity_treatment), mode=mode, seed=identity_seed, world_size=world_size
    )
    scientific, operational, treatment_config = partition_run_config(run_config)
    if treatment_config != {"row_identity_mode": mode}:
        raise ValueError("run config treatment does not match identity treatment")
    _validate_operational_context(operational_context, mode=mode)
    validate_manifest(parent_manifest, expected_kind="parent")
    manifests = {
        "source": dict(source_manifest),
        "environment": make_manifest("environment", environment),
        "architecture": architecture_manifest(model_config, state_dict),
        "prior": prior_manifest(prior_stream),
        "optimizer": make_manifest("optimizer", optimizer_config),
        "stage": make_manifest(
            "stage", {"stage": stage, "terminal_step": terminal_step}
        ),
        "seed": make_manifest(
            "seed",
            {
                "np_seed": np_seed,
                "torch_seed": torch_seed,
                "identity_rng_seed": identity_seed,
                "world_size": world_size,
            },
        ),
        "parent": dict(parent_manifest),
        "treatment": make_manifest("treatment", dict(identity_treatment)),
        "scientific_config": make_manifest("scientific_config", scientific),
        "operational_config": make_manifest(
            "operational_config",
            {"fields": operational, "context": dict(operational_context)},
        ),
    }
    manifests["cohort_protocol"] = make_manifest(
        "cohort_protocol",
        {
            "stage": stage,
            "terminal_step": terminal_step,
            "source_sha256": manifests["source"]["sha256"],
            "environment_sha256": manifests["environment"]["sha256"],
            "architecture_sha256": manifests["architecture"]["sha256"],
            "prior_sha256": manifests["prior"]["sha256"],
            "optimizer_sha256": manifests["optimizer"]["sha256"],
            "seed_sha256": manifests["seed"]["sha256"],
            "scientific_config_sha256": manifests["scientific_config"]["sha256"],
        },
    )
    manifests["arm_protocol"] = make_manifest(
        "arm_protocol",
        {
            "cohort_protocol_sha256": manifests["cohort_protocol"]["sha256"],
            "mode": mode,
            "treatment_sha256": manifests["treatment"]["sha256"],
        },
    )
    body = _provenance_body(manifests)
    return {**body, "manifests": manifests, "bundle_sha256": canonical_sha256(body)}


def validate_provenance_bundle(bundle: Mapping[str, Any]) -> str:
    if not isinstance(bundle, Mapping):
        raise ValueError("provenance must be an object")
    _require_exact_keys(
        bundle,
        {"schema_version", "manifest_sha256", "manifests", "bundle_sha256"},
        where="provenance",
    )
    if bundle["schema_version"] != PROVENANCE_SCHEMA_VERSION:
        raise ValueError("provenance schema_version mismatch")
    manifests = bundle["manifests"]
    if not isinstance(manifests, Mapping) or set(manifests) != REQUIRED_MANIFESTS:
        raise ValueError("provenance manifests are incomplete or contain unknown names")
    for name, manifest in manifests.items():
        validate_manifest(manifest, expected_kind=name)
    stage_payload = manifests["stage"]["payload"]
    cohort_payload = manifests["cohort_protocol"]["payload"]
    expected_cohort_payload = {
        "stage": stage_payload.get("stage"),
        "terminal_step": stage_payload.get("terminal_step"),
        "source_sha256": manifests["source"]["sha256"],
        "environment_sha256": manifests["environment"]["sha256"],
        "architecture_sha256": manifests["architecture"]["sha256"],
        "prior_sha256": manifests["prior"]["sha256"],
        "optimizer_sha256": manifests["optimizer"]["sha256"],
        "seed_sha256": manifests["seed"]["sha256"],
        "scientific_config_sha256": manifests["scientific_config"]["sha256"],
    }
    if cohort_payload != expected_cohort_payload:
        raise ValueError("cohort protocol does not bind the exact shared manifests")
    treatment_payload = manifests["treatment"]["payload"]
    expected_arm_payload = {
        "cohort_protocol_sha256": manifests["cohort_protocol"]["sha256"],
        "mode": treatment_payload.get("row_identity_mode"),
        "treatment_sha256": manifests["treatment"]["sha256"],
    }
    if manifests["arm_protocol"]["payload"] != expected_arm_payload:
        raise ValueError("arm protocol is not derived only from cohort and treatment")
    expected_digests = {name: manifests[name]["sha256"] for name in sorted(manifests)}
    if bundle["manifest_sha256"] != expected_digests:
        raise ValueError("provenance manifest_sha256 index mismatch")
    body = {
        "schema_version": bundle["schema_version"],
        "manifest_sha256": expected_digests,
    }
    actual = canonical_sha256(body)
    if bundle["bundle_sha256"] != actual:
        raise ValueError("provenance bundle_sha256 mismatch")
    return actual


def validate_cohort_provenance(
    bundles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(bundles) != FORMAL_MODES:
        raise ValueError(
            "formal cohort must contain exactly rope, temporary, and none arms"
        )
    invariant_names = (
        "source",
        "environment",
        "architecture",
        "prior",
        "optimizer",
        "stage",
        "seed",
        "scientific_config",
        "cohort_protocol",
    )
    study_id = None
    invariant: dict[str, str] = {}
    treatment_hashes: dict[str, str] = {}
    arm_protocol_hashes: dict[str, str] = {}
    for mode in sorted(bundles):
        bundle = bundles[mode]
        validate_provenance_bundle(bundle)
        manifests = bundle["manifests"]
        context = manifests["operational_config"]["payload"]["context"]
        _validate_operational_context(context, mode=mode)
        if study_id is None:
            study_id = context["study_id"]
            invariant = {name: manifests[name]["sha256"] for name in invariant_names}
        elif context["study_id"] != study_id:
            raise ValueError("cohort study_id mismatch")
        for name in invariant_names:
            if manifests[name]["sha256"] != invariant[name]:
                raise ValueError(f"cohort {name} manifest mismatch")
        treatment = manifests["treatment"]["payload"]
        if treatment.get("row_identity_mode") != mode:
            raise ValueError("cohort treatment mode mismatch")
        treatment_hashes[mode] = manifests["treatment"]["sha256"]
        arm_protocol_hashes[mode] = manifests["arm_protocol"]["sha256"]
    if len(set(treatment_hashes.values())) != len(FORMAL_MODES):
        raise ValueError("cohort treatments must be distinct")
    if len(set(arm_protocol_hashes.values())) != len(FORMAL_MODES):
        raise ValueError("cohort arm protocols must be distinct")
    return {
        "study_id": study_id,
        "arms": sorted(bundles),
        "cohort_protocol_sha256": invariant["cohort_protocol"],
        "arm_protocol_sha256": arm_protocol_hashes,
        "invariant_sha256": invariant,
        "treatment_sha256": treatment_hashes,
    }


@dataclass(frozen=True)
class ParentTrust:
    checkpoint_path: Path
    finalized_manifest_path: Path
    transaction_ledger_path: Path
    transaction_ledger_sha256: str
    study_id: str
    arm: str
    parent_stage: str
    upstream_identity: str
    artifact_identity: str
    artifact_root: Path | None = None


@dataclass(frozen=True)
class FinalizationTrust:
    transaction_ledger_path: Path
    transaction_ledger_sha256: str
    artifact_root: Path
    study_id: str
    upstream_identity: str
    artifact_identity: str


@dataclass(frozen=True)
class CheckpointExpectations:
    mode: str
    np_seed: int
    torch_seed: int
    identity_seed: int
    stage: str
    terminal_step: int
    source_sha256: str
    environment_sha256: str
    prior_sha256: str
    architecture_sha256: str
    optimizer_sha256: str
    scientific_sha256: str
    cohort_protocol_sha256: str
    arm_protocol_sha256: str
    world_size: int
    cuda_device_count: int
    parent_trust: ParentTrust | None = None


_LEDGER_ENTRY_KEYS = {
    "arm",
    "stage",
    "terminal_step",
    "upstream_identity",
    "artifact_identity",
    "checkpoint_relpath",
    "finalized_manifest_relpath",
    "np_seed",
    "torch_seed",
    "identity_rng_seed",
    "world_size",
    "source_sha256",
    "environment_sha256",
    "prior_sha256",
    "architecture_sha256",
    "optimizer_sha256",
    "scientific_sha256",
    "cohort_protocol_sha256",
    "arm_protocol_sha256",
}
_LEDGER_DIGEST_FIELDS = (
    "source_sha256",
    "environment_sha256",
    "prior_sha256",
    "architecture_sha256",
    "optimizer_sha256",
    "scientific_sha256",
    "cohort_protocol_sha256",
    "arm_protocol_sha256",
)
_LEDGER_INTEGER_FIELDS = (
    ("terminal_step", 1),
    ("np_seed", 0),
    ("torch_seed", 0),
    ("identity_rng_seed", 0),
    ("world_size", 1),
)


def _validate_ledger_entry(entry: Any) -> Mapping[str, Any]:
    if not isinstance(entry, Mapping):
        raise ValueError("transaction ledger entry must be an object")
    _require_exact_keys(entry, _LEDGER_ENTRY_KEYS, where="transaction ledger entry")
    for digest_name in _LEDGER_DIGEST_FIELDS:
        _require_digest(f"transaction ledger {digest_name}", entry[digest_name])
    for integer_name, minimum in _LEDGER_INTEGER_FIELDS:
        integer = entry[integer_name]
        if (
            isinstance(integer, bool)
            or not isinstance(integer, int)
            or integer < minimum
        ):
            raise ValueError(f"transaction ledger {integer_name} is invalid")
    if entry["arm"] not in FORMAL_MODES or entry["stage"] not in FORMAL_STAGES:
        raise ValueError("transaction ledger arm/stage is invalid")
    for name in ("upstream_identity", "artifact_identity"):
        if not isinstance(entry[name], str) or _SAFE_ID.fullmatch(entry[name]) is None:
            raise ValueError(f"transaction ledger {name} is invalid")
    _validate_relative_path(entry["checkpoint_relpath"], where="ledger checkpoint")
    _validate_relative_path(
        entry["finalized_manifest_relpath"], where="ledger finalized manifest"
    )
    return entry


def checkpoint_sha256(path: str | os.PathLike[str]) -> str:
    fd, before = _open_regular_nofollow(path)
    digest = hashlib.sha256()
    try:
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
        _same_file_snapshot(before, os.fstat(fd), where=str(path))
    finally:
        os.close(fd)
    return digest.hexdigest()


def _load_checkpoint_and_hash(path: Path) -> tuple[dict[str, Any], str, int]:
    fd, before = _open_regular_nofollow(path)
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "rb", closefd=False) as handle:
            while True:
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
            handle.seek(0)
            checkpoint = torch.load(handle, map_location="cpu", weights_only=True)
        _same_file_snapshot(before, os.fstat(fd), where=str(path))
    finally:
        os.close(fd)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint root must be a dictionary")
    return checkpoint, digest.hexdigest(), before.st_size


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _validate_rng_bundle_offline(
    bundle: Any, *, world_size: int, cuda_device_count: int
) -> None:
    if not isinstance(bundle, Mapping):
        raise ValueError("RNG state must be an all-rank object")
    _require_exact_keys(
        bundle,
        {"schema_version", "world_size", "rank_states", "manifest_sha256"},
        where="RNG bundle",
    )
    if bundle["schema_version"] != 1 or bundle["world_size"] != world_size:
        raise ValueError("RNG bundle world_size/schema mismatch")
    if (
        isinstance(cuda_device_count, bool)
        or not isinstance(cuda_device_count, int)
        or cuda_device_count < 0
    ):
        raise ValueError("expected CUDA device count must be a non-negative integer")
    ranks = bundle["rank_states"]
    expected_ranks = {str(rank) for rank in range(world_size)}
    if not isinstance(ranks, Mapping) or set(ranks) != expected_ranks:
        raise ValueError("RNG bundle rank coverage mismatch")
    state_hashes = {}
    for outer_rank in sorted(ranks, key=int):
        state = ranks[outer_rank]
        if not isinstance(state, Mapping):
            raise ValueError("RNG rank state must be an object")
        required = {
            "schema_version",
            "rank",
            "python",
            "numpy",
            "torch_cpu",
            "torch_cuda",
            "cuda_device_count",
            "manifest_sha256",
        }
        _require_exact_keys(state, required, where="RNG rank state")
        if state["schema_version"] != 1 or str(state["rank"]) != outer_rank:
            raise ValueError("RNG outer/inner rank mismatch")
        numpy_state = state["numpy"]
        if not isinstance(numpy_state, Mapping):
            raise ValueError("RNG NumPy state is invalid")
        _require_exact_keys(
            numpy_state,
            {"bit_generator", "keys", "position", "has_gauss", "cached_gaussian"},
            where="RNG NumPy state",
        )
        if not isinstance(numpy_state["keys"], Tensor) or not isinstance(
            state["torch_cpu"], Tensor
        ):
            raise ValueError("RNG tensor state is invalid")
        if not isinstance(state["torch_cuda"], list) or not all(
            isinstance(value, Tensor) for value in state["torch_cuda"]
        ):
            raise ValueError("RNG CUDA state is invalid")
        if state["cuda_device_count"] != len(state["torch_cuda"]):
            raise ValueError("RNG CUDA state count mismatch")
        if state["cuda_device_count"] != cuda_device_count:
            raise ValueError(
                "RNG CUDA device count does not match external expectation"
            )
        try:
            random.Random().setstate(state["python"])
        except Exception as error:
            raise ValueError("RNG Python state is not restorable") from error
        numpy_keys = numpy_state["keys"]
        if (
            numpy_state["bit_generator"] != "MT19937"
            or numpy_keys.dtype != torch.int64
            or numpy_keys.ndim != 1
            or isinstance(numpy_state["position"], bool)
            or not isinstance(numpy_state["position"], int)
            or isinstance(numpy_state["has_gauss"], bool)
            or not isinstance(numpy_state["has_gauss"], int)
            or not isinstance(numpy_state["cached_gaussian"], float)
            or not math.isfinite(numpy_state["cached_gaussian"])
        ):
            raise ValueError("RNG NumPy state is not restorable")
        try:
            np.random.RandomState().set_state(
                (
                    numpy_state["bit_generator"],
                    numpy_keys.detach().cpu().numpy().astype(np.uint32, copy=True),
                    numpy_state["position"],
                    numpy_state["has_gauss"],
                    numpy_state["cached_gaussian"],
                )
            )
        except Exception as error:
            raise ValueError("RNG NumPy state is not restorable") from error
        torch_cpu = state["torch_cpu"]
        if torch_cpu.dtype != torch.uint8 or torch_cpu.ndim != 1:
            raise ValueError("RNG Torch CPU state is not restorable")
        try:
            torch.Generator(device="cpu").set_state(torch_cpu.detach().cpu())
        except Exception as error:
            raise ValueError("RNG Torch CPU state is not restorable") from error
        if any(
            value.dtype != torch.uint8 or value.ndim != 1 or value.numel() == 0
            for value in state["torch_cuda"]
        ):
            raise ValueError("RNG CUDA state is not restorable")
        rank_manifest = {
            "schema_version": state["schema_version"],
            "rank": state["rank"],
            "python": state["python"],
            "numpy": {
                "bit_generator": numpy_state["bit_generator"],
                "keys_sha256": _tensor_sha256(numpy_state["keys"]),
                "position": numpy_state["position"],
                "has_gauss": numpy_state["has_gauss"],
                "cached_gaussian": numpy_state["cached_gaussian"],
            },
            "torch_cpu_sha256": _tensor_sha256(state["torch_cpu"]),
            "torch_cuda_sha256": [
                _tensor_sha256(value) for value in state["torch_cuda"]
            ],
            "cuda_device_count": state["cuda_device_count"],
        }
        expected_hash = hashlib.sha256(
            json.dumps(rank_manifest, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        if state["manifest_sha256"] != expected_hash:
            raise ValueError("RNG rank manifest hash mismatch")
        state_hashes[outer_rank] = expected_hash
    top = {
        "schema_version": 1,
        "world_size": world_size,
        "rank_state_hashes": state_hashes,
    }
    expected_top = hashlib.sha256(
        json.dumps(top, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if bundle["manifest_sha256"] != expected_top:
        raise ValueError("RNG bundle manifest hash mismatch")


def _validate_identity_sampler_offline(
    bundle: Any, *, seed: int, world_size: int, torch_version: str
) -> None:
    from tabicl.train._identity_rng import (
        BUNDLE_SCHEMA_VERSION,
        DEVICE_POLICY,
        SAMPLER_VERSION,
        SAMPLING_ALGORITHM,
        STATE_SCHEMA_VERSION,
    )

    if not isinstance(bundle, Mapping):
        raise ValueError("identity_sampler must be an all-rank object")
    _require_exact_keys(
        bundle,
        {
            "schema_version",
            "sampler_version",
            "base_seed",
            "world_size",
            "rank_states",
            "manifest_sha256",
        },
        where="identity sampler bundle",
    )
    if (
        bundle["schema_version"] != BUNDLE_SCHEMA_VERSION
        or bundle["sampler_version"] != SAMPLER_VERSION
        or bundle["base_seed"] != seed
        or bundle["world_size"] != world_size
    ):
        raise ValueError("identity sampler bundle seed/world/schema mismatch")
    rank_states = bundle["rank_states"]
    expected_ranks = {str(rank) for rank in range(world_size)}
    if not isinstance(rank_states, Mapping) or set(rank_states) != expected_ranks:
        raise ValueError("identity sampler rank coverage mismatch")
    records = []
    draw_count = None
    for outer_rank in sorted(rank_states, key=int):
        state = rank_states[outer_rank]
        if not isinstance(state, Mapping):
            raise ValueError("identity sampler rank state is invalid")
        required = {
            "schema_version",
            "sampler_version",
            "algorithm",
            "device_policy",
            "torch_version",
            "base_seed",
            "rank",
            "world_size",
            "derived_seed",
            "draw_count",
            "generator_state",
            "generator_state_sha256",
        }
        _require_exact_keys(state, required, where="identity sampler rank state")
        rank = int(outer_rank)
        derived = int.from_bytes(
            hashlib.sha256(
                f"{SAMPLER_VERSION}\0base_seed={seed}\0rank={rank}".encode()
            ).digest()[:8],
            "big",
        ) & ((1 << 63) - 1)
        expected = {
            "schema_version": STATE_SCHEMA_VERSION,
            "sampler_version": SAMPLER_VERSION,
            "algorithm": SAMPLING_ALGORITHM,
            "device_policy": DEVICE_POLICY,
            "base_seed": seed,
            "rank": rank,
            "world_size": world_size,
            "derived_seed": derived,
        }
        for field, value in expected.items():
            if state[field] != value:
                raise ValueError(f"identity sampler {field} mismatch")
        generator_state = state["generator_state"]
        if (
            not isinstance(generator_state, Tensor)
            or generator_state.dtype != torch.uint8
            or generator_state.ndim != 1
        ):
            raise ValueError("identity sampler generator_state is invalid")
        if (
            _tensor_sha256(generator_state.to(dtype=torch.uint8))
            != state["generator_state_sha256"]
        ):
            raise ValueError("identity sampler generator_state hash mismatch")
        if state["torch_version"] != torch_version:
            raise ValueError("identity sampler torch_version mismatch")
        try:
            torch.Generator(device="cpu").set_state(generator_state.detach().cpu())
        except Exception as error:
            raise ValueError(
                "identity sampler generator state is not restorable"
            ) from error
        if (
            isinstance(state["draw_count"], bool)
            or not isinstance(state["draw_count"], int)
            or state["draw_count"] < 0
        ):
            raise ValueError("identity sampler draw_count is invalid")
        if draw_count is None:
            draw_count = state["draw_count"]
        elif draw_count != state["draw_count"]:
            raise ValueError("identity sampler draw_count differs across ranks")
        records.append(
            {key: value for key, value in state.items() if key != "generator_state"}
        )
    top = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "sampler_version": SAMPLER_VERSION,
        "base_seed": seed,
        "world_size": world_size,
        "rank_states": records,
    }
    expected_hash = hashlib.sha256(
        json.dumps(top, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if bundle["manifest_sha256"] != expected_hash:
        raise ValueError("identity sampler bundle manifest hash mismatch")


def _validate_finite(value: Any, *, path: str) -> None:
    if isinstance(value, Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"non-finite tensor in {path}")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number in {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_finite(child, path=f"{path}[{index}]")


def _validate_optimizer_state(state: Mapping[str, Any]) -> None:
    _require_exact_keys(state, {"state", "param_groups"}, where="optimizer_state")
    slot_state = state["state"]
    groups = state["param_groups"]
    if (
        not isinstance(slot_state, Mapping)
        or not isinstance(groups, list)
        or not groups
    ):
        raise ValueError("optimizer_state state/param_groups structure is invalid")
    parameter_ids: list[int] = []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping) or "params" not in group:
            raise ValueError(f"optimizer_state param_groups[{index}] is missing params")
        params = group["params"]
        if not isinstance(params, list):
            raise ValueError("optimizer_state group params must be a list")
        for parameter_id in params:
            if isinstance(parameter_id, bool) or not isinstance(parameter_id, int):
                raise ValueError("optimizer_state param IDs must be non-bool integers")
            if parameter_id < 0:
                raise ValueError("optimizer_state param IDs must be non-negative")
            parameter_ids.append(parameter_id)
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("optimizer_state contains duplicate param IDs")
    state_ids = set()
    for parameter_id, value in slot_state.items():
        if isinstance(parameter_id, bool) or not isinstance(parameter_id, int):
            raise ValueError("optimizer_state state IDs must be non-bool integers")
        if not isinstance(value, Mapping):
            raise ValueError("optimizer_state slot state must be an object")
        state_ids.add(parameter_id)
    foreign = state_ids - set(parameter_ids)
    if foreign:
        raise ValueError(
            f"optimizer_state contains foreign state IDs: {sorted(foreign)}"
        )
    missing = set(parameter_ids) - state_ids
    if missing:
        raise ValueError(
            f"optimizer_state is missing state for param IDs: {sorted(missing)}"
        )


def _validate_scheduler_state(state: Mapping[str, Any], *, terminal_step: int) -> None:
    last_epoch = state.get("last_epoch")
    if isinstance(last_epoch, bool) or not isinstance(last_epoch, int):
        raise ValueError("scheduler_state must contain integer last_epoch")
    if last_epoch != terminal_step:
        raise ValueError("scheduler_state last_epoch does not equal terminal step")
    step_count = state.get("_step_count")
    if step_count is not None:
        if isinstance(step_count, bool) or not isinstance(step_count, int):
            raise ValueError("scheduler_state step count must be an integer")
        if step_count != terminal_step + 1:
            raise ValueError(
                "scheduler_state step count does not equal terminal step plus one"
            )


def _validate_scaler_state(state: Mapping[str, Any], *, amp_enabled: bool) -> None:
    if not isinstance(amp_enabled, bool):
        raise ValueError("scientific config amp must be a boolean")
    if not amp_enabled:
        if state:
            raise ValueError("disabled AMP scaler_state must be empty")
        return
    required = {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"enabled AMP scaler_state is missing: {', '.join(missing)}")


def _read_manifest_file(
    path: Path, *, kind: str, expected_sha: str
) -> Mapping[str, Any]:
    value, _raw_sha = strict_json_load_nofollow(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"{kind} manifest file must contain an object")
    validate_manifest(value, expected_kind=kind, expected_sha256=expected_sha)
    return value


def _resolved_ledger_path(root: Path, relative: str, *, where: str) -> Path:
    relative = _validate_relative_path(relative, where=where)
    root = root.resolve(strict=True)
    path = root.joinpath(*PurePosixPath(relative).parts).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{where} escapes artifact root") from error
    return path


def _expected_parent_manifest(
    trust: ParentTrust,
    *,
    checkpoint_sha256: str,
    finalized_manifest_sha256: str,
    ledger_entry: Mapping[str, Any],
) -> dict[str, Any]:
    return make_manifest(
        "parent",
        {
            "parent": {
                "checkpoint_sha256": checkpoint_sha256,
                "finalized_manifest_sha256": finalized_manifest_sha256,
                "transaction_ledger_sha256": trust.transaction_ledger_sha256,
                "study_id": trust.study_id,
                "arm": trust.arm,
                "stage": trust.parent_stage,
                "terminal_step": ledger_entry["terminal_step"],
                "upstream_identity": trust.upstream_identity,
                "artifact_identity": trust.artifact_identity,
                "np_seed": ledger_entry["np_seed"],
                "torch_seed": ledger_entry["torch_seed"],
                "identity_rng_seed": ledger_entry["identity_rng_seed"],
                "world_size": ledger_entry["world_size"],
                "source_sha256": ledger_entry["source_sha256"],
                "environment_sha256": ledger_entry["environment_sha256"],
                "prior_sha256": ledger_entry["prior_sha256"],
                "architecture_sha256": ledger_entry["architecture_sha256"],
                "optimizer_sha256": ledger_entry["optimizer_sha256"],
                "scientific_sha256": ledger_entry["scientific_sha256"],
                "cohort_protocol_sha256": ledger_entry["cohort_protocol_sha256"],
                "arm_protocol_sha256": ledger_entry["arm_protocol_sha256"],
            }
        },
    )


def validate_parent_trust(trust: ParentTrust) -> dict[str, Any]:
    _require_digest("parent transaction_ledger_sha256", trust.transaction_ledger_sha256)
    if trust.arm not in FORMAL_MODES or trust.parent_stage not in FORMAL_STAGES:
        raise ValueError("parent arm/stage is invalid")
    for name in ("study_id", "upstream_identity", "artifact_identity"):
        if _SAFE_ID.fullmatch(getattr(trust, name)) is None:
            raise ValueError(f"parent {name} is invalid")
    ledger = _read_manifest_file(
        trust.transaction_ledger_path,
        kind="transaction_ledger",
        expected_sha=trust.transaction_ledger_sha256,
    )
    ledger_payload = ledger["payload"]
    _require_exact_keys(
        ledger_payload, {"study_id", "entries"}, where="transaction ledger"
    )
    if ledger_payload["study_id"] != trust.study_id or not isinstance(
        ledger_payload["entries"], list
    ):
        raise ValueError("transaction ledger study/entries mismatch")
    matches = []
    for entry in ledger_payload["entries"]:
        entry = _validate_ledger_entry(entry)
        if (
            entry["arm"] == trust.arm
            and entry["stage"] == trust.parent_stage
            and entry["upstream_identity"] == trust.upstream_identity
            and entry["artifact_identity"] == trust.artifact_identity
        ):
            matches.append(entry)
    if len(matches) != 1:
        raise ValueError(
            "transaction ledger must contain exactly one expected producer entry"
        )
    entry = matches[0]
    artifact_root = trust.artifact_root or trust.transaction_ledger_path.parent
    expected_checkpoint_path = _resolved_ledger_path(
        artifact_root, entry["checkpoint_relpath"], where="ledger checkpoint"
    )
    expected_final_path = _resolved_ledger_path(
        artifact_root,
        entry["finalized_manifest_relpath"],
        where="ledger finalized manifest",
    )
    if expected_checkpoint_path != trust.checkpoint_path.resolve(strict=True):
        raise ValueError("parent checkpoint path does not match immutable ledger")
    if expected_final_path != trust.finalized_manifest_path.resolve(strict=True):
        raise ValueError("finalized manifest path does not match immutable ledger")

    finalized_value, _raw_sha = strict_json_load_nofollow(trust.finalized_manifest_path)
    if not isinstance(finalized_value, Mapping):
        raise ValueError("finalized checkpoint manifest file must contain an object")
    finalized_sha = validate_manifest(
        finalized_value, expected_kind="finalized_checkpoint"
    )
    finalized = finalized_value
    final_payload = finalized["payload"]
    final_keys = {
        "study_id",
        "arm",
        "stage",
        "terminal_step",
        "upstream_identity",
        "artifact_identity",
        "checkpoint_sha256",
        "checkpoint_size",
        "provenance_sha256",
        "source_sha256",
        "environment_sha256",
        "prior_sha256",
        "architecture_sha256",
        "optimizer_sha256",
        "seed_sha256",
        "treatment_sha256",
        "scientific_sha256",
        "cohort_protocol_sha256",
        "arm_protocol_sha256",
    }
    _require_exact_keys(final_payload, final_keys, where="finalized checkpoint")
    expected_identity = {
        "study_id": trust.study_id,
        "arm": trust.arm,
        "stage": trust.parent_stage,
        "upstream_identity": trust.upstream_identity,
        "artifact_identity": trust.artifact_identity,
    }
    for field, expected in expected_identity.items():
        if final_payload[field] != expected:
            raise ValueError(f"finalized checkpoint {field} mismatch")
    if final_payload["terminal_step"] != entry["terminal_step"]:
        raise ValueError("finalized checkpoint terminal_step mismatch")

    parent_checkpoint, parent_sha, parent_size = _load_checkpoint_and_hash(
        trust.checkpoint_path
    )
    _require_digest(
        "finalized parent checkpoint_sha256", final_payload["checkpoint_sha256"]
    )
    if final_payload["checkpoint_sha256"] != parent_sha:
        raise ValueError(
            "parent checkpoint sha256 does not match finalized exact bytes"
        )
    if final_payload["checkpoint_size"] != parent_size:
        raise ValueError("finalized checkpoint byte size mismatch")
    provenance = parent_checkpoint.get("provenance")
    validate_provenance_bundle(provenance)
    manifests = provenance["manifests"]
    checks = {
        "provenance_sha256": provenance["bundle_sha256"],
        "source_sha256": manifests["source"]["sha256"],
        "environment_sha256": manifests["environment"]["sha256"],
        "prior_sha256": manifests["prior"]["sha256"],
        "architecture_sha256": manifests["architecture"]["sha256"],
        "optimizer_sha256": manifests["optimizer"]["sha256"],
        "seed_sha256": manifests["seed"]["sha256"],
        "treatment_sha256": manifests["treatment"]["sha256"],
        "scientific_sha256": manifests["scientific_config"]["sha256"],
        "cohort_protocol_sha256": manifests["cohort_protocol"]["sha256"],
        "arm_protocol_sha256": manifests["arm_protocol"]["sha256"],
    }
    for field, actual in checks.items():
        if final_payload[field] != actual:
            raise ValueError(f"finalized checkpoint {field} mismatch")
    if manifests["stage"]["payload"] != {
        "stage": trust.parent_stage,
        "terminal_step": final_payload["terminal_step"],
    }:
        raise ValueError("parent checkpoint stage mismatch")
    if parent_checkpoint.get("curr_step") != final_payload["terminal_step"]:
        raise ValueError("parent checkpoint terminal step mismatch")
    if (
        parent_checkpoint.get("identity_treatment", {}).get("row_identity_mode")
        != trust.arm
    ):
        raise ValueError("parent checkpoint mode mismatch")
    seed_payload = manifests["seed"]["payload"]
    ledger_invariants = {
        "np_seed": seed_payload.get("np_seed"),
        "torch_seed": seed_payload.get("torch_seed"),
        "identity_rng_seed": seed_payload.get("identity_rng_seed"),
        "world_size": seed_payload.get("world_size"),
        "source_sha256": manifests["source"]["sha256"],
        "environment_sha256": manifests["environment"]["sha256"],
        "prior_sha256": manifests["prior"]["sha256"],
        "architecture_sha256": manifests["architecture"]["sha256"],
        "optimizer_sha256": manifests["optimizer"]["sha256"],
        "scientific_sha256": manifests["scientific_config"]["sha256"],
        "cohort_protocol_sha256": manifests["cohort_protocol"]["sha256"],
        "arm_protocol_sha256": manifests["arm_protocol"]["sha256"],
    }
    for field, actual in ledger_invariants.items():
        if entry[field] != actual:
            raise ValueError(f"parent {field} does not match immutable ledger")
    return _expected_parent_manifest(
        trust,
        checkpoint_sha256=parent_sha,
        finalized_manifest_sha256=finalized_sha,
        ledger_entry=entry,
    )


def _validate_expected_manifest_hashes(
    manifests: Mapping[str, Mapping[str, Any]], expected: CheckpointExpectations
) -> None:
    expected_hashes = {
        "source": expected.source_sha256,
        "environment": expected.environment_sha256,
        "prior": expected.prior_sha256,
        "architecture": expected.architecture_sha256,
        "optimizer": expected.optimizer_sha256,
        "scientific_config": expected.scientific_sha256,
        "cohort_protocol": expected.cohort_protocol_sha256,
        "arm_protocol": expected.arm_protocol_sha256,
    }
    for name, digest in expected_hashes.items():
        if manifests[name]["sha256"] != _require_digest(f"expected {name}", digest):
            raise ValueError(f"{name} manifest does not match explicit expected sha256")


def validate_identity_checkpoint(
    checkpoint_path: str | os.PathLike[str], expected: CheckpointExpectations
) -> dict[str, Any]:
    """Validate one complete formal checkpoint without restoring runtime RNG."""
    path = Path(checkpoint_path)
    checkpoint, checkpoint_digest, checkpoint_size = _load_checkpoint_and_hash(path)
    required = {
        "config",
        "state_dict",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "curr_step",
        "rng_state",
        "prior_stream",
        "identity_treatment",
        "provenance",
    }
    allowed = required | {"identity_sampler"}
    missing = sorted(required - set(checkpoint))
    extra = sorted(set(checkpoint) - allowed)
    if missing or extra:
        raise ValueError(
            f"formal checkpoint keys mismatch; missing={missing}, extra={extra}"
        )
    if expected.mode not in FORMAL_MODES:
        raise ValueError("expected mode is invalid")
    if expected.stage not in FORMAL_STAGES:
        raise ValueError("expected stage is invalid")
    if isinstance(expected.terminal_step, bool) or expected.terminal_step < 1:
        raise ValueError("expected terminal_step is invalid")
    if path.name != f"step-{expected.terminal_step}.ckpt":
        raise ValueError("checkpoint filename must equal the explicit terminal step")
    if checkpoint["curr_step"] != expected.terminal_step:
        raise ValueError("checkpoint curr_step does not equal explicit terminal_step")
    for name in ("optimizer_state", "scheduler_state", "scaler_state"):
        if not isinstance(checkpoint[name], Mapping):
            raise ValueError(f"checkpoint {name} must be an object")
        _validate_finite(checkpoint[name], path=name)
    _validate_optimizer_state(checkpoint["optimizer_state"])
    _validate_scheduler_state(
        checkpoint["scheduler_state"], terminal_step=expected.terminal_step
    )
    _validate_finite(checkpoint["state_dict"], path="state_dict")

    provenance = checkpoint["provenance"]
    validate_provenance_bundle(provenance)
    manifests = provenance["manifests"]
    _validate_expected_manifest_hashes(manifests, expected)
    validate_source_manifest(manifests["source"])
    environment_payload = manifests["environment"]["payload"]
    if (
        environment_payload.get("visible_cuda_device_count")
        != expected.cuda_device_count
    ):
        raise ValueError("environment visible CUDA device count mismatch")
    scientific_payload = manifests["scientific_config"]["payload"]
    _validate_scaler_state(
        checkpoint["scaler_state"], amp_enabled=scientific_payload.get("amp")
    )

    stage_payload = manifests["stage"]["payload"]
    if stage_payload.get("stage") != expected.stage:
        raise ValueError("stage does not match explicit expected stage")
    if stage_payload.get("terminal_step") != expected.terminal_step:
        raise ValueError("terminal_step does not match explicit expected terminal_step")
    seed_payload = manifests["seed"]["payload"]
    expected_seed = {
        "np_seed": expected.np_seed,
        "torch_seed": expected.torch_seed,
        "identity_rng_seed": expected.identity_seed,
        "world_size": expected.world_size,
    }
    if seed_payload != expected_seed:
        differing = [
            key for key in expected_seed if seed_payload.get(key) != expected_seed[key]
        ]
        raise ValueError(f"seed manifest mismatch: {', '.join(differing)}")

    if not isinstance(checkpoint["config"], Mapping):
        raise ValueError("checkpoint config must be an object")
    if checkpoint["config"].get("row_identity_mode") != expected.mode:
        raise ValueError("checkpoint row_identity_mode mismatch")
    actual_architecture = architecture_manifest(
        checkpoint["config"], checkpoint["state_dict"]
    )
    if actual_architecture["sha256"] != manifests["architecture"]["sha256"]:
        raise ValueError("architecture/state_dict key-shape-dtype schema mismatch")

    actual_prior = prior_manifest(checkpoint["prior_stream"])
    if actual_prior["sha256"] != manifests["prior"]["sha256"]:
        raise ValueError("prior definition does not match provenance")
    prior_state = checkpoint["prior_stream"]
    if prior_state["cursor"] != expected.terminal_step:
        raise ValueError("prior cursor does not equal terminal step")
    if prior_state["experiment_seed"] != expected.np_seed:
        raise ValueError("prior experiment seed mismatch")
    if prior_state["world_size"] != expected.world_size or prior_state["ddp_rank"] != 0:
        raise ValueError("prior world_size/master rank mismatch")

    treatment = checkpoint["identity_treatment"]
    if not isinstance(treatment, Mapping):
        raise ValueError("checkpoint identity_treatment is invalid")
    from tabicl.train._identity_rng import validate_identity_treatment

    validate_identity_treatment(
        dict(treatment),
        mode=expected.mode,
        seed=expected.identity_seed,
        world_size=expected.world_size,
    )
    if manifests["treatment"]["payload"] != treatment:
        raise ValueError("treatment manifest does not match checkpoint treatment")
    operational = manifests["operational_config"]["payload"]
    _require_exact_keys(
        operational, {"fields", "context"}, where="operational manifest"
    )
    if not isinstance(operational["fields"], Mapping) or not set(
        operational["fields"]
    ).issubset(OPERATIONAL_CONFIG_FIELDS):
        raise ValueError("operational fields contain an undeclared config key")
    _validate_operational_context(operational["context"], mode=expected.mode)

    _validate_rng_bundle_offline(
        checkpoint["rng_state"],
        world_size=expected.world_size,
        cuda_device_count=expected.cuda_device_count,
    )
    if expected.mode == "temporary":
        if "identity_sampler" not in checkpoint:
            raise ValueError("temporary checkpoint requires identity_sampler")
        _validate_identity_sampler_offline(
            checkpoint["identity_sampler"],
            seed=expected.identity_seed,
            world_size=expected.world_size,
            torch_version=environment_payload.get("torch_version"),
        )
    elif "identity_sampler" in checkpoint:
        raise ValueError("rope/none checkpoint must not contain identity_sampler")

    parent = manifests["parent"]
    if expected.stage == "stage1":
        if parent["payload"] != {"parent": None} or expected.parent_trust is not None:
            raise ValueError("Stage 1 requires an explicit null parent")
    else:
        if expected.parent_trust is None:
            raise ValueError("Stage 2/3 requires explicit parent trust inputs")
        trusted_parent = validate_parent_trust(expected.parent_trust)
        if parent != trusted_parent:
            raise ValueError(
                "parent manifest does not match immutable ledger and exact parent bytes"
            )
        expected_parent_stage = "stage1" if expected.stage == "stage2" else "stage2"
        if expected.parent_trust.parent_stage != expected_parent_stage:
            raise ValueError(
                f"{expected.stage} parent stage must be {expected_parent_stage}"
            )
        if expected.parent_trust.arm != expected.mode:
            raise ValueError("parent arm does not match child treatment mode")
        parent_payload = trusted_parent["payload"]["parent"]
        current_context = operational["context"]
        if parent_payload["study_id"] != current_context["study_id"]:
            raise ValueError("parent study_id does not match child study")
        parent_child_invariants = {
            "np_seed": expected.np_seed,
            "torch_seed": expected.torch_seed,
            "identity_rng_seed": expected.identity_seed,
            "world_size": expected.world_size,
            "source_sha256": manifests["source"]["sha256"],
            "environment_sha256": manifests["environment"]["sha256"],
            "architecture_sha256": manifests["architecture"]["sha256"],
        }
        for field, child_value in parent_child_invariants.items():
            if parent_payload[field] != child_value:
                raise ValueError(f"parent {field} does not match child checkpoint")

    return {
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_size": checkpoint_size,
        "provenance_sha256": provenance["bundle_sha256"],
        "source_sha256": manifests["source"]["sha256"],
        "environment_sha256": manifests["environment"]["sha256"],
        "prior_sha256": manifests["prior"]["sha256"],
        "architecture_sha256": manifests["architecture"]["sha256"],
        "optimizer_sha256": manifests["optimizer"]["sha256"],
        "seed_sha256": manifests["seed"]["sha256"],
        "treatment_sha256": manifests["treatment"]["sha256"],
        "scientific_sha256": manifests["scientific_config"]["sha256"],
        "cohort_protocol_sha256": manifests["cohort_protocol"]["sha256"],
        "arm_protocol_sha256": manifests["arm_protocol"]["sha256"],
        "study_id": operational["context"]["study_id"],
        "output_id": operational["context"]["output_id"],
        "mode": expected.mode,
        "stage": expected.stage,
        "terminal_step": expected.terminal_step,
    }


def _load_finalization_ledger_entry(
    trust: FinalizationTrust,
    *,
    mode: str,
    stage: str,
) -> Mapping[str, Any]:
    _require_digest(
        "finalization transaction_ledger_sha256", trust.transaction_ledger_sha256
    )
    for name in ("study_id", "upstream_identity", "artifact_identity"):
        value = getattr(trust, name)
        if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
            raise ValueError(f"finalization {name} is invalid")
    ledger = _read_manifest_file(
        trust.transaction_ledger_path,
        kind="transaction_ledger",
        expected_sha=trust.transaction_ledger_sha256,
    )
    payload = ledger["payload"]
    _require_exact_keys(payload, {"study_id", "entries"}, where="transaction ledger")
    if payload["study_id"] != trust.study_id or not isinstance(
        payload["entries"], list
    ):
        raise ValueError("transaction ledger study/entries mismatch")
    matches = []
    for candidate in payload["entries"]:
        entry = _validate_ledger_entry(candidate)
        if (
            entry["arm"] == mode
            and entry["stage"] == stage
            and entry["upstream_identity"] == trust.upstream_identity
            and entry["artifact_identity"] == trust.artifact_identity
        ):
            matches.append(entry)
    if len(matches) != 1:
        raise ValueError(
            "transaction ledger must contain exactly one finalization entry"
        )
    return matches[0]


def _resolved_future_ledger_path(root: Path, relative: str, *, where: str) -> Path:
    relative = _validate_relative_path(relative, where=where)
    root = root.resolve(strict=True)
    lexical = root.joinpath(*PurePosixPath(relative).parts)
    parent = lexical.parent.resolve(strict=True)
    try:
        parent.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{where} escapes artifact root") from error
    return parent / lexical.name


def _prepare_finalized_manifest(
    checkpoint_path: str | os.PathLike[str],
    expected: CheckpointExpectations,
    *,
    finalized_manifest_path: str | os.PathLike[str],
    trust: FinalizationTrust,
) -> dict[str, Any]:
    report = validate_identity_checkpoint(checkpoint_path, expected)
    entry = _load_finalization_ledger_entry(
        trust, mode=expected.mode, stage=expected.stage
    )
    expected_checkpoint = _resolved_ledger_path(
        trust.artifact_root, entry["checkpoint_relpath"], where="ledger checkpoint"
    )
    actual_checkpoint = Path(checkpoint_path).resolve(strict=True)
    if actual_checkpoint != expected_checkpoint:
        raise ValueError("checkpoint path does not match immutable finalization ledger")
    expected_final = _resolved_future_ledger_path(
        trust.artifact_root,
        entry["finalized_manifest_relpath"],
        where="ledger finalized manifest",
    )
    actual_final = (
        Path(finalized_manifest_path).parent.resolve(strict=True)
        / Path(finalized_manifest_path).name
    )
    if actual_final != expected_final:
        raise ValueError("finalized manifest path does not match immutable ledger")
    expected_invariants = {
        "terminal_step": expected.terminal_step,
        "np_seed": expected.np_seed,
        "torch_seed": expected.torch_seed,
        "identity_rng_seed": expected.identity_seed,
        "world_size": expected.world_size,
        "source_sha256": report["source_sha256"],
        "environment_sha256": report["environment_sha256"],
        "prior_sha256": report["prior_sha256"],
        "architecture_sha256": report["architecture_sha256"],
        "optimizer_sha256": report["optimizer_sha256"],
        "scientific_sha256": report["scientific_sha256"],
        "cohort_protocol_sha256": report["cohort_protocol_sha256"],
        "arm_protocol_sha256": report["arm_protocol_sha256"],
    }
    for field, actual in expected_invariants.items():
        if entry[field] != actual:
            raise ValueError(f"finalization {field} does not match immutable ledger")
    if report["study_id"] != trust.study_id:
        raise ValueError("finalization study_id does not match checkpoint provenance")
    return make_manifest(
        "finalized_checkpoint",
        {
            "study_id": trust.study_id,
            "arm": expected.mode,
            "stage": expected.stage,
            "terminal_step": expected.terminal_step,
            "upstream_identity": trust.upstream_identity,
            "artifact_identity": trust.artifact_identity,
            "checkpoint_sha256": report["checkpoint_sha256"],
            "checkpoint_size": report["checkpoint_size"],
            "provenance_sha256": report["provenance_sha256"],
            "source_sha256": report["source_sha256"],
            "environment_sha256": report["environment_sha256"],
            "prior_sha256": report["prior_sha256"],
            "architecture_sha256": report["architecture_sha256"],
            "optimizer_sha256": report["optimizer_sha256"],
            "seed_sha256": report["seed_sha256"],
            "treatment_sha256": report["treatment_sha256"],
            "scientific_sha256": report["scientific_sha256"],
            "cohort_protocol_sha256": report["cohort_protocol_sha256"],
            "arm_protocol_sha256": report["arm_protocol_sha256"],
        },
    )


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("write made no progress")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise ValueError(f"finalization parent must be a directory: {path}")
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_json_no_replace(path: Path, value: Mapping[str, Any]) -> None:
    """Publish complete bytes with link(2), never exposing a partial final file."""
    parent = path.parent.resolve(strict=True)
    final_path = parent / path.name
    payload = canonical_json_bytes(value) + b"\n"
    temp_path = parent / (f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temp_path, flags, 0o600)
    fd_open = True
    published = False
    try:
        _write_all(fd, payload)
        os.fsync(fd)
        os.close(fd)
        fd_open = False
        try:
            os.link(temp_path, final_path, follow_symlinks=False)
        except FileExistsError as error:
            raise ValueError(
                f"finalized manifest already exists: {final_path}"
            ) from error
        published = True
        _fsync_directory(parent)
        os.unlink(temp_path)
        _fsync_directory(parent)
    except BaseException:
        if fd_open:
            os.close(fd)
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        # If publication already happened, the final name refers only to the
        # fully-written, file-fsynced inode.  Keep it and report failure; an
        # explicit recovery call can verify identical bytes and fsync the dir.
        if not published:
            try:
                _fsync_directory(parent)
            except OSError:
                pass
        raise


def finalize_identity_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    expected: CheckpointExpectations,
    *,
    finalized_manifest_path: str | os.PathLike[str],
    trust: FinalizationTrust,
) -> dict[str, Any]:
    manifest = _prepare_finalized_manifest(
        checkpoint_path,
        expected,
        finalized_manifest_path=finalized_manifest_path,
        trust=trust,
    )
    _publish_json_no_replace(Path(finalized_manifest_path), manifest)
    return manifest


def recover_finalized_checkpoint_manifest(
    checkpoint_path: str | os.PathLike[str],
    expected: CheckpointExpectations,
    *,
    finalized_manifest_path: str | os.PathLike[str],
    trust: FinalizationTrust,
) -> dict[str, Any]:
    """Recover only an identical complete final after a directory-fsync error."""
    manifest = _prepare_finalized_manifest(
        checkpoint_path,
        expected,
        finalized_manifest_path=finalized_manifest_path,
        trust=trust,
    )
    existing, _raw_sha = strict_json_load_nofollow(finalized_manifest_path)
    if existing != manifest:
        raise ValueError(
            "existing finalized manifest is not the exact expected manifest"
        )
    _fsync_directory(Path(finalized_manifest_path).parent.resolve(strict=True))
    return manifest
