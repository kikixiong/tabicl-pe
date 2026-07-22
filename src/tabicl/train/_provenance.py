"""Canonical provenance and fail-closed validation for formal pre-training.

The self-hashes in this module detect corruption.  They are not trust roots: a
formal validator must also receive expected digests from outside the mutable
checkpoint/archive.  Exact parent-checkpoint bytes are the authority for model
weights because a manifest embedded in the same file cannot authenticate them.

No-replace finalization is a publisher API invariant, not filesystem
immutability.  A malicious artifact owner who can unlink and recreate both a
checkpoint and its unsigned finalization record is outside this threat model;
that case requires an external signature, append-only receipt, or a different
publishing principal.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
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
_SOURCE_MODES = frozenset({"100644", "100755"})
_LINK_SUPPORTS_DIR_FD = os.link in os.supports_dir_fd
_UNLINK_SUPPORTS_DIR_FD = os.unlink in os.supports_dir_fd


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
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if not candidate.name or candidate.name in {".", ".."}:
        raise ValueError(f"path must name a regular file: {path}")
    parent_fd = _open_directory_components_nofollow(
        candidate.parent, where="regular file parent"
    )
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(candidate.name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError(
            f"refusing to open non-regular or symlink path: {path}"
        ) from error
    finally:
        os.close(parent_fd)
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        os.close(fd)
        raise ValueError(f"path must be a regular file: {path}")
    return fd, before


def _open_directory_components_nofollow(
    path: str | os.PathLike[str], *, where: str
) -> int:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    parts = candidate.parts
    if (
        not parts
        or parts[0] != os.path.sep
        or any(part in {"", ".", ".."} for part in parts[1:])
    ):
        raise ValueError(f"{where} must be a normalized absolute path")
    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or os.open not in os.supports_dir_fd
    ):
        raise ValueError("no-follow path-component traversal is unavailable")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd: int | None = None
    try:
        fd = os.open(os.path.sep, flags)
        for part in parts[1:]:
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise ValueError(f"{where} is not a directory")
        return fd
    except OSError as error:
        if fd is not None:
            os.close(fd)
        raise ValueError(
            f"{where} has a symlink or invalid path component: {candidate}"
        ) from error


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
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    file_flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        file_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW
    directory_fds: list[int] = []
    try:
        directory_fds.append(
            _open_directory_components_nofollow(root, where="source root")
        )
        parts = PurePosixPath(relative).parts
        for part in parts[:-1]:
            directory_fds.append(
                os.open(part, directory_flags, dir_fd=directory_fds[-1])
            )
        fd = os.open(parts[-1], file_flags, dir_fd=directory_fds[-1])
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            os.close(fd)
            raise ValueError(f"tracked source must be a regular file: {relative}")
    except OSError as error:
        raise ValueError(
            f"tracked source has a symlink or invalid path component: {relative}"
        ) from error
    finally:
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)
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
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = Path.cwd() / root_path
    root_fd = _open_directory_components_nofollow(root_path, where="source root")
    os.close(root_fd)
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
        if mode == "120000":
            raise ValueError(f"tracked source symlinks are forbidden: {relative}")
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
    """Build a manifest for the final extracted bytes of one Git checkout."""
    repo_path = Path(repo)
    if not repo_path.is_absolute():
        repo_path = Path.cwd() / repo_path
    repo_fd = _open_directory_components_nofollow(repo_path, where="Git source root")
    os.close(repo_fd)
    commit = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", f"{commit_sha}^{{commit}}"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD^{commit}"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repo_path),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    if head != commit or status:
        raise ValueError(
            "final extracted bytes require a clean exact checkout of the requested commit"
        )
    tree = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", f"{commit}^{{tree}}"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    listing = subprocess.run(
        ["git", "-C", str(repo_path), "ls-tree", "-rz", "-r", "--full-tree", commit],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    records = []
    for record in listing.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode_b, object_type, _object_id = metadata.split(b" ", 2)
        if object_type != b"blob":
            raise ValueError(
                "unsupported Git object "
                f"{object_type.decode('ascii', errors='replace')!r} for "
                f"{raw_path.decode('utf-8', errors='replace')!r}"
            )
        path = raw_path.decode("utf-8", errors="strict")
        mode = mode_b.decode("ascii")
        if mode not in _SOURCE_MODES:
            raise ValueError(f"unsupported Git source mode {mode!r} for {path}")
        relative = _validate_relative_path(path, where="Git source")
        records.append((relative, mode))
    entries = []
    for relative, mode in records:
        blob = _source_bytes(repo_path, relative, mode)
        entries.append(
            {
                "path": relative,
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
        if entry["mode"] == "120000":
            raise ValueError(f"source symlink entries are forbidden: {path}")
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


def _qualified_type(value: Any) -> str:
    cls = value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def build_optimizer_protocol(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    *,
    scheduler_algorithm: str,
    scheduler_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe only static optimizer machinery and ordered parameter schema."""
    named = {
        id(parameter): (name, parameter) for name, parameter in model.named_parameters()
    }
    seen: set[int] = set()
    groups = []
    for index, group in enumerate(optimizer.param_groups):
        parameters = []
        for parameter in group["params"]:
            identity = id(parameter)
            if identity not in named:
                raise ValueError("optimizer must cover only named model parameters")
            if identity in seen:
                raise ValueError("optimizer contains a duplicate model parameter")
            seen.add(identity)
            name, parameter = named[identity]
            parameters.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "requires_grad": bool(parameter.requires_grad),
                }
            )
        base_lr = group.get("initial_lr", optimizer.defaults.get("lr"))
        if not isinstance(base_lr, (int, float)) or isinstance(base_lr, bool):
            raise ValueError(f"optimizer group {index} has no numeric base LR")
        static = {
            key: value
            for key, value in group.items()
            if key not in {"params", "lr", "initial_lr"}
        }
        groups.append(
            {
                "parameters": parameters,
                "base_lr": float(base_lr),
                "static": _normalize_json(
                    static, where=f"optimizer group {index} static config"
                ),
            }
        )
    if seen != set(named):
        missing = sorted(
            name for identity, (name, _) in named.items() if identity not in seen
        )
        raise ValueError(f"optimizer is missing named parameters: {missing}")
    if not groups:
        raise ValueError("optimizer protocol requires at least one parameter group")
    enabled = bool(scaler.is_enabled())
    scaler_static = {
        "device": str(getattr(scaler, "_device", "cuda")),
        "init_scale": float(getattr(scaler, "_init_scale", 65536.0)),
        "growth_factor": float(getattr(scaler, "_growth_factor", 2.0)),
        "backoff_factor": float(getattr(scaler, "_backoff_factor", 0.5)),
        "growth_interval": int(getattr(scaler, "_growth_interval", 2000)),
    }
    return {
        "schema_version": 1,
        "optimizer": {"type": _qualified_type(optimizer), "groups": groups},
        "scheduler": {
            "type": _qualified_type(scheduler),
            "algorithm": scheduler_algorithm,
            "static": _normalize_json(
                scheduler_config, where="scheduler static config"
            ),
        },
        "scaler": {
            "type": _qualified_type(scaler),
            "enabled": enabled,
            "static": scaler_static,
        },
    }


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
    _validate_optimizer_protocol(optimizer_config)
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
    _validate_optimizer_protocol(manifests["optimizer"]["payload"])
    stage_payload = manifests["stage"]["payload"]
    _require_exact_keys(
        stage_payload, {"stage", "terminal_step"}, where="stage payload"
    )
    if stage_payload["stage"] not in FORMAL_STAGES:
        raise ValueError("stage payload stage is invalid")
    if (
        isinstance(stage_payload["terminal_step"], bool)
        or not isinstance(stage_payload["terminal_step"], int)
        or stage_payload["terminal_step"] < 1
    ):
        raise ValueError("stage payload terminal_step is invalid")
    seed_payload = manifests["seed"]["payload"]
    _require_exact_keys(
        seed_payload,
        {"np_seed", "torch_seed", "identity_rng_seed", "world_size"},
        where="seed payload",
    )
    for field, minimum in (
        ("np_seed", 0),
        ("torch_seed", 0),
        ("identity_rng_seed", 0),
        ("world_size", 1),
    ):
        value = seed_payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"seed payload {field} is invalid")
    treatment_payload = manifests["treatment"]["payload"]
    _require_exact_keys(
        treatment_payload,
        {
            "schema_version",
            "row_identity_mode",
            "identity_rng_seed",
            "seed_policy",
            "sampler_version",
            "world_size",
            "manifest_sha256",
        },
        where="treatment payload",
    )
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
    max_checkpoint_bytes: int
    parent_trust: ParentTrust | None = None


@dataclass(frozen=True)
class ValidatedParent:
    manifest: dict[str, Any]
    checkpoint: dict[str, Any]
    checkpoint_sha256: str
    checkpoint_size: int


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
    "cuda_device_count",
    "max_checkpoint_bytes",
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
    ("cuda_device_count", 0),
    ("max_checkpoint_bytes", 1),
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


def _load_checkpoint_and_hash(
    path: Path, *, max_checkpoint_bytes: int
) -> tuple[dict[str, Any], str, int]:
    if (
        isinstance(max_checkpoint_bytes, bool)
        or not isinstance(max_checkpoint_bytes, int)
        or max_checkpoint_bytes < 1
    ):
        raise ValueError("max_checkpoint_bytes must be a positive integer")
    fd, before = _open_regular_nofollow(path)
    if before.st_size <= 0:
        os.close(fd)
        raise ValueError("checkpoint must contain at least one byte")
    if before.st_size > max_checkpoint_bytes:
        os.close(fd)
        raise ValueError(
            "checkpoint exceeds external byte ceiling: "
            f"{before.st_size} > {max_checkpoint_bytes}"
        )
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


def _require_finite_number(
    value: Any,
    *,
    where: str,
    minimum: float | None = None,
    strict_minimum: bool = False,
    maximum: float | None = None,
    strict_maximum: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{where} is invalid")
    number = float(value)
    if minimum is not None and (
        number < minimum or (strict_minimum and number == minimum)
    ):
        raise ValueError(f"{where} is invalid")
    if maximum is not None and (
        number > maximum or (strict_maximum and number == maximum)
    ):
        raise ValueError(f"{where} is invalid")
    return number


def _validate_betas(value: Any, *, where: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{where} is invalid")
    for index, beta in enumerate(value):
        _require_finite_number(
            beta,
            where=f"{where}[{index}]",
            minimum=0.0,
            maximum=1.0,
            strict_maximum=True,
        )


def _validate_optimizer_group_static(
    optimizer_type: str, static: Mapping[str, Any], *, group_index: int
) -> None:
    if optimizer_type == "torch.optim.adamw.AdamW":
        required = {
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "maximize",
            "foreach",
            "capturable",
            "differentiable",
            "fused",
        }
        optional = {"decoupled_weight_decay"}
        missing = sorted(required - set(static))
        extra = sorted(set(static) - required - optional)
        if missing or extra:
            raise ValueError(
                f"AdamW static group {group_index} keys mismatch; "
                f"missing={missing}, extra={extra}"
            )
        _validate_betas(static["betas"], where="AdamW static betas")
        _require_finite_number(
            static["eps"], where="AdamW static eps", minimum=0.0, strict_minimum=True
        )
        _require_finite_number(
            static["weight_decay"],
            where="AdamW static weight_decay",
            minimum=0.0,
        )
        for field in ("amsgrad", "maximize", "capturable", "differentiable"):
            if not isinstance(static[field], bool):
                raise ValueError(f"AdamW static {field} is invalid")
        if (
            "decoupled_weight_decay" in static
            and static["decoupled_weight_decay"] is not True
        ):
            raise ValueError("AdamW static decoupled_weight_decay is invalid")
        for field in ("foreach", "fused"):
            if static[field] is not None and not isinstance(static[field], bool):
                raise ValueError(f"AdamW static {field} is invalid")
        return

    if optimizer_type == "tabicl.train._muon.Muon":
        required = {
            "use_muon",
            "weight_decay",
            "matched_adamw_rms",
            "momentum",
            "nesterov",
            "ns_steps",
            "adamw_betas",
            "adamw_eps",
            "use_cautious_wd",
        }
        _require_exact_keys(static, required, where=f"Muon static group {group_index}")
        if static["use_muon"] is not True:
            raise ValueError("Muon static use_muon must be true")
        _require_finite_number(
            static["weight_decay"],
            where="Muon static weight_decay",
            minimum=0.0,
        )
        _require_finite_number(
            static["matched_adamw_rms"],
            where="Muon static matched_adamw_rms",
            minimum=0.0,
            strict_minimum=True,
        )
        _require_finite_number(
            static["momentum"],
            where="Muon static momentum",
            minimum=0.0,
            maximum=1.0,
            strict_maximum=True,
        )
        if not isinstance(static["nesterov"], bool):
            raise ValueError("Muon static nesterov is invalid")
        if (
            isinstance(static["ns_steps"], bool)
            or not isinstance(static["ns_steps"], int)
            or static["ns_steps"] < 1
        ):
            raise ValueError("Muon static ns_steps is invalid")
        _validate_betas(static["adamw_betas"], where="Muon static adamw_betas")
        _require_finite_number(
            static["adamw_eps"],
            where="Muon static adamw_eps",
            minimum=0.0,
            strict_minimum=True,
        )
        if not isinstance(static["use_cautious_wd"], bool):
            raise ValueError("Muon static use_cautious_wd is invalid")
        return

    raise ValueError("optimizer protocol optimizer type is unsupported")


def _validate_optimizer_protocol(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("optimizer protocol must be an object")
    _require_exact_keys(
        payload,
        {"schema_version", "optimizer", "scheduler", "scaler"},
        where="optimizer protocol",
    )
    if payload["schema_version"] != 1:
        raise ValueError("optimizer protocol schema_version mismatch")
    optimizer = payload["optimizer"]
    scheduler = payload["scheduler"]
    scaler = payload["scaler"]
    for name, value, keys in (
        ("optimizer", optimizer, {"type", "groups"}),
        ("scheduler", scheduler, {"type", "algorithm", "static"}),
        ("scaler", scaler, {"type", "enabled", "static"}),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"optimizer protocol {name} must be an object")
        _require_exact_keys(value, keys, where=f"optimizer protocol {name}")
        if not isinstance(value["type"], str) or not value["type"]:
            raise ValueError(f"optimizer protocol {name} type is invalid")
    supported_optimizer_types = {
        "torch.optim.adamw.AdamW",
        "tabicl.train._muon.Muon",
    }
    if optimizer["type"] not in supported_optimizer_types:
        raise ValueError("optimizer protocol optimizer type is unsupported")
    if scheduler["type"] != "torch.optim.lr_scheduler.LambdaLR":
        raise ValueError("optimizer protocol scheduler type is unsupported")
    if scaler["type"] != "torch.amp.grad_scaler.GradScaler":
        raise ValueError("optimizer protocol scaler type is unsupported")
    if not isinstance(optimizer["groups"], list) or not optimizer["groups"]:
        raise ValueError("optimizer protocol groups must be a non-empty list")
    names: set[str] = set()
    for group_index, group in enumerate(optimizer["groups"]):
        if not isinstance(group, Mapping):
            raise ValueError("optimizer protocol group must be an object")
        _require_exact_keys(
            group,
            {"parameters", "base_lr", "static"},
            where=f"optimizer protocol group {group_index}",
        )
        if (
            isinstance(group["base_lr"], bool)
            or not isinstance(group["base_lr"], (int, float))
            or not math.isfinite(group["base_lr"])
            or group["base_lr"] < 0
        ):
            raise ValueError("optimizer protocol base_lr is invalid")
        if not isinstance(group["static"], Mapping):
            raise ValueError("optimizer protocol group static config is invalid")
        _validate_optimizer_group_static(
            optimizer["type"], group["static"], group_index=group_index
        )
        parameters = group["parameters"]
        if not isinstance(parameters, list) or not parameters:
            raise ValueError("optimizer protocol group parameters must be non-empty")
        for parameter in parameters:
            if not isinstance(parameter, Mapping):
                raise ValueError("optimizer parameter schema must be an object")
            _require_exact_keys(
                parameter,
                {"name", "shape", "dtype", "requires_grad"},
                where="optimizer parameter schema",
            )
            name = parameter["name"]
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("optimizer parameter names must be unique")
            names.add(name)
            shape = parameter["shape"]
            if (
                not isinstance(shape, list)
                or any(
                    isinstance(size, bool) or not isinstance(size, int) or size < 0
                    for size in shape
                )
                or not isinstance(parameter["dtype"], str)
                or not parameter["dtype"].startswith("torch.")
                or (
                    parameter["requires_grad"] is not True
                    and parameter["requires_grad"] is not False
                )
            ):
                raise ValueError("optimizer parameter schema is invalid")
    algorithms = {
        "constant": {"max_steps"},
        "linear_warmup": {"max_steps", "warmup_steps"},
        "cosine_warmup": {"max_steps", "warmup_steps"},
        "cosine_with_restarts": {
            "max_steps",
            "warmup_steps",
            "num_cycles",
            "amplitude_decay",
            "lr_end",
        },
        "polynomial_decay_warmup": {
            "max_steps",
            "warmup_steps",
            "lr_end",
            "power",
        },
    }
    algorithm = scheduler["algorithm"]
    if algorithm not in algorithms or not isinstance(scheduler["static"], Mapping):
        raise ValueError("optimizer protocol scheduler algorithm/config is invalid")
    _require_exact_keys(
        scheduler["static"],
        algorithms[algorithm],
        where="optimizer protocol scheduler static config",
    )
    for field, value in scheduler["static"].items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"scheduler static {field} is invalid")
    if (
        isinstance(scheduler["static"]["max_steps"], bool)
        or not isinstance(scheduler["static"]["max_steps"], int)
        or scheduler["static"]["max_steps"] < 1
    ):
        raise ValueError("scheduler static max_steps is invalid")
    static = scheduler["static"]
    max_steps = static["max_steps"]
    if algorithm != "constant":
        warmup_steps = _require_finite_number(
            static["warmup_steps"],
            where="scheduler static warmup_steps",
            minimum=0.0,
        )
        if warmup_steps > max_steps:
            raise ValueError("scheduler static warmup_steps is invalid")
    if algorithm == "cosine_with_restarts":
        if (
            isinstance(static["num_cycles"], bool)
            or not isinstance(static["num_cycles"], int)
            or static["num_cycles"] < 1
        ):
            raise ValueError("scheduler static num_cycles is invalid")
        _require_finite_number(
            static["amplitude_decay"],
            where="scheduler static amplitude_decay",
            minimum=0.0,
            strict_minimum=True,
            maximum=1.0,
        )
    if algorithm == "polynomial_decay_warmup":
        _require_finite_number(
            static["power"],
            where="scheduler static power",
            minimum=0.0,
            strict_minimum=True,
        )
    if algorithm in {"cosine_with_restarts", "polynomial_decay_warmup"}:
        lr_end = _require_finite_number(
            static["lr_end"], where="scheduler static lr_end", minimum=0.0
        )
        if any(lr_end > group["base_lr"] for group in optimizer["groups"]):
            raise ValueError("scheduler static lr_end exceeds an optimizer base_lr")
    if scaler["enabled"] is not True and scaler["enabled"] is not False:
        raise ValueError("optimizer protocol scaler enabled is invalid")
    if not isinstance(scaler["static"], Mapping):
        raise ValueError("optimizer protocol scaler static config is invalid")
    _require_exact_keys(
        scaler["static"],
        {
            "device",
            "init_scale",
            "growth_factor",
            "backoff_factor",
            "growth_interval",
        },
        where="optimizer protocol scaler static config",
    )
    if scaler["static"]["device"] not in {"cpu", "cuda"}:
        raise ValueError("scaler static device is invalid")
    _require_finite_number(
        scaler["static"]["init_scale"],
        where="scaler static init_scale",
        minimum=0.0,
        strict_minimum=True,
    )
    _require_finite_number(
        scaler["static"]["growth_factor"],
        where="scaler static growth_factor",
        minimum=1.0,
        strict_minimum=True,
    )
    _require_finite_number(
        scaler["static"]["backoff_factor"],
        where="scaler static backoff_factor",
        minimum=0.0,
        strict_minimum=True,
        maximum=1.0,
        strict_maximum=True,
    )
    growth_interval = scaler["static"]["growth_interval"]
    if (
        isinstance(growth_interval, bool)
        or not isinstance(growth_interval, int)
        or growth_interval < 1
    ):
        raise ValueError("scaler static growth_interval is invalid")
    return payload


def _validate_parameter_tensor(
    tensor: Any, parameter: Mapping[str, Any], *, where: str
) -> None:
    if not isinstance(tensor, Tensor):
        raise ValueError(f"{where} must be a tensor")
    if list(tensor.shape) != parameter["shape"]:
        raise ValueError(f"{where} shape does not match parameter schema")
    if str(tensor.dtype) != parameter["dtype"]:
        raise ValueError(f"{where} dtype does not match parameter schema")
    _validate_finite(tensor, path=where)


def _validate_optimizer_state(
    state: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    model_state: Mapping[str, Any],
    terminal_step: int,
) -> tuple[list[float], list[float]]:
    _require_exact_keys(state, {"state", "param_groups"}, where="optimizer_state")
    slot_state = state["state"]
    groups = state["param_groups"]
    if (
        not isinstance(slot_state, Mapping)
        or not isinstance(groups, list)
        or not groups
    ):
        raise ValueError("optimizer_state state/param_groups structure is invalid")
    expected_groups = protocol["optimizer"]["groups"]
    if len(groups) != len(expected_groups):
        raise ValueError("optimizer_state parameter group count mismatch")
    parameter_ids: list[int] = []
    current_lrs: list[float] = []
    base_lrs: list[float] = []
    id_to_parameter: dict[int, Mapping[str, Any]] = {}
    for index, (group, expected_group) in enumerate(zip(groups, expected_groups)):
        if not isinstance(group, Mapping) or "params" not in group:
            raise ValueError(f"optimizer_state param_groups[{index}] is missing params")
        params = group["params"]
        expected_parameters = expected_group["parameters"]
        if not isinstance(params, list) or len(params) != len(expected_parameters):
            raise ValueError("optimizer_state group params must be a list")
        actual_static = {
            key: value
            for key, value in group.items()
            if key not in {"params", "lr", "initial_lr"}
        }
        normalized_static = _normalize_json(
            actual_static, where=f"optimizer_state group {index} static config"
        )
        if normalized_static != expected_group["static"]:
            differing = sorted(set(normalized_static) | set(expected_group["static"]))
            raise ValueError(
                "optimizer_state static config mismatch: " + ", ".join(differing)
            )
        if group.get("initial_lr") != expected_group["base_lr"]:
            raise ValueError("optimizer_state initial_lr mismatch")
        lr = group.get("lr")
        if (
            isinstance(lr, bool)
            or not isinstance(lr, (int, float))
            or not math.isfinite(lr)
            or lr < 0
        ):
            raise ValueError("optimizer_state current lr is invalid")
        current_lrs.append(float(lr))
        base_lrs.append(float(expected_group["base_lr"]))
        for parameter_id, parameter in zip(params, expected_parameters):
            if isinstance(parameter_id, bool) or not isinstance(parameter_id, int):
                raise ValueError("optimizer_state param IDs must be non-bool integers")
            if parameter_id < 0:
                raise ValueError("optimizer_state param IDs must be non-negative")
            parameter_ids.append(parameter_id)
            id_to_parameter[parameter_id] = parameter
            name = parameter["name"]
            if name not in model_state:
                raise ValueError(f"optimizer parameter {name} is absent from model")
            _validate_parameter_tensor(
                model_state[name], parameter, where=f"model parameter {name}"
            )
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
    required_state_ids = {
        parameter_id
        for parameter_id, parameter in id_to_parameter.items()
        if parameter["requires_grad"]
    }
    missing = required_state_ids - state_ids
    if missing:
        raise ValueError(
            f"optimizer_state is missing state for param IDs: {sorted(missing)}"
        )
    optimizer_type = protocol["optimizer"]["type"]
    for parameter_id in sorted(state_ids):
        slot = slot_state[parameter_id]
        parameter = id_to_parameter[parameter_id]
        if optimizer_type.endswith(".AdamW"):
            group = next(item for item in groups if parameter_id in item["params"])
            required = {"step", "exp_avg", "exp_avg_sq"}
            if group.get("amsgrad"):
                required.add("max_exp_avg_sq")
            _require_exact_keys(slot, required, where="AdamW optimizer slot")
            step = slot["step"]
            if isinstance(step, Tensor):
                if step.numel() != 1:
                    raise ValueError("AdamW optimizer step must be scalar")
                if not step.dtype.is_floating_point:
                    raise ValueError("AdamW optimizer step tensor dtype is invalid")
                step = float(step.detach().cpu().item())
            if (
                isinstance(step, bool)
                or not isinstance(step, (int, float))
                or not math.isfinite(step)
                or step != terminal_step
            ):
                raise ValueError("AdamW optimizer step does not equal terminal step")
            for key in required - {"step"}:
                _validate_parameter_tensor(
                    slot[key], parameter, where=f"AdamW optimizer {key}"
                )
        elif optimizer_type.endswith("._muon.Muon"):
            _require_exact_keys(slot, {"momentum_buffer"}, where="Muon optimizer slot")
            _validate_parameter_tensor(
                slot["momentum_buffer"],
                parameter,
                where="Muon optimizer momentum_buffer",
            )
        else:
            raise ValueError(f"unsupported formal optimizer type: {optimizer_type}")
    return current_lrs, base_lrs


def _validate_scheduler_state(
    state: Mapping[str, Any],
    *,
    terminal_step: int,
    protocol: Mapping[str, Any],
    current_lrs: Sequence[float],
    base_lrs: Sequence[float],
) -> None:
    required = {
        "base_lrs",
        "last_epoch",
        "_step_count",
        "_last_lr",
        "lr_lambdas",
    }
    optional = {"_is_initial", "_get_lr_called_within_step", "verbose"}
    missing = sorted(required - set(state))
    extra = sorted(set(state) - required - optional)
    if missing or extra:
        raise ValueError(
            f"scheduler_state keys mismatch; missing={missing}, extra={extra}"
        )
    for field in optional & set(state):
        if state[field] is not False:
            raise ValueError(f"scheduler_state {field} must be false")
    last_epoch = state.get("last_epoch")
    if isinstance(last_epoch, bool) or not isinstance(last_epoch, int):
        raise ValueError("scheduler_state must contain integer last_epoch")
    if last_epoch != terminal_step:
        raise ValueError("scheduler_state last_epoch does not equal terminal step")
    step_count = state["_step_count"]
    if isinstance(step_count, bool) or not isinstance(step_count, int):
        raise ValueError("scheduler_state step count must be an integer")
    if step_count != terminal_step + 1:
        raise ValueError(
            "scheduler_state step count does not equal terminal step plus one"
        )
    if state["base_lrs"] != list(base_lrs):
        raise ValueError("scheduler_state base_lrs mismatch")
    if state["_last_lr"] != list(current_lrs):
        raise ValueError("scheduler_state current LR mismatch")
    algorithm = protocol["scheduler"]["algorithm"]
    expected_lambdas = (
        [None] * len(base_lrs) if algorithm == "constant" else [{} for _ in base_lrs]
    )
    if state["lr_lambdas"] != expected_lambdas:
        raise ValueError("scheduler_state lambda structure mismatch")
    static = protocol["scheduler"]["static"]
    if static["max_steps"] != terminal_step:
        raise ValueError("scheduler protocol max_steps mismatch")
    expected_lrs = []
    for base_lr in base_lrs:
        if algorithm == "constant":
            expected_lrs.append(base_lr)
        elif algorithm in {"linear_warmup", "cosine_warmup"}:
            expected_lrs.append(0.0)
        elif algorithm == "cosine_with_restarts":
            expected_lrs.append(float(static["lr_end"]))
        elif algorithm == "polynomial_decay_warmup":
            expected_lrs.append(float(static["lr_end"]))
    if any(
        not math.isclose(actual, wanted, rel_tol=1e-9, abs_tol=1e-12)
        for actual, wanted in zip(current_lrs, expected_lrs)
    ):
        raise ValueError("scheduler_state terminal LR is inconsistent with protocol")
    try:
        parameters = [torch.nn.Parameter(torch.zeros(())) for _ in base_lrs]
        dummy_optimizer = torch.optim.SGD(
            [
                {"params": [parameter], "lr": base_lr}
                for parameter, base_lr in zip(parameters, base_lrs)
            ]
        )
        dummy_scheduler = torch.optim.lr_scheduler.LambdaLR(
            dummy_optimizer, [lambda _step: 1.0 for _ in base_lrs]
        )
        dummy_scheduler.load_state_dict(copy.deepcopy(dict(state)))
    except Exception as error:
        raise ValueError("scheduler_state is not restorable") from error


def _validate_scaler_state(
    state: Mapping[str, Any], *, protocol: Mapping[str, Any]
) -> None:
    scaler_protocol = protocol["scaler"]
    if not scaler_protocol["enabled"]:
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
    _require_exact_keys(state, required, where="enabled AMP scaler_state")
    static = scaler_protocol["static"]
    for field in ("growth_factor", "backoff_factor", "growth_interval"):
        if state[field] != static[field]:
            raise ValueError(f"enabled AMP scaler_state {field} mismatch")
    growth_tracker = state["_growth_tracker"]
    if (
        isinstance(growth_tracker, bool)
        or not isinstance(growth_tracker, int)
        or growth_tracker < 0
        or growth_tracker >= state["growth_interval"]
    ):
        raise ValueError("enabled AMP scaler_state growth tracker is invalid")
    if (
        not isinstance(state["scale"], (int, float))
        or isinstance(state["scale"], bool)
        or not math.isfinite(state["scale"])
        or state["scale"] <= 0
        or not (state["growth_factor"] > 1)
        or not (0 < state["backoff_factor"] < 1)
        or not isinstance(state["growth_interval"], int)
        or state["growth_interval"] < 1
    ):
        raise ValueError("enabled AMP scaler_state is invalid")
    try:
        dummy = torch.GradScaler(
            "cpu",
            enabled=True,
            init_scale=static["init_scale"],
            growth_factor=static["growth_factor"],
            backoff_factor=static["backoff_factor"],
            growth_interval=static["growth_interval"],
        )
        dummy.load_state_dict(copy.deepcopy(dict(state)))
        dummy.scale(torch.ones((), requires_grad=True))
    except Exception as error:
        raise ValueError("enabled AMP scaler_state is not restorable") from error


def _read_manifest_file(
    path: Path, *, kind: str, expected_sha: str
) -> Mapping[str, Any]:
    value, _raw_sha = strict_json_load_nofollow(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"{kind} manifest file must contain an object")
    validate_manifest(value, expected_kind=kind, expected_sha256=expected_sha)
    return value


def _absolute_lexical_path(path: str | os.PathLike[str], *, where: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    parts = candidate.parts
    if (
        not parts
        or parts[0] != os.path.sep
        or any(part in {"", ".", ".."} for part in parts[1:])
    ):
        raise ValueError(f"{where} must be a normalized absolute path")
    return candidate


def _regular_lexical_path(path: str | os.PathLike[str], *, where: str) -> Path:
    candidate = _absolute_lexical_path(path, where=where)
    try:
        fd, _snapshot = _open_regular_nofollow(candidate)
    except ValueError as error:
        raise ValueError(f"{where} has a symlink or invalid path component") from error
    os.close(fd)
    return candidate


def _resolved_ledger_path(root: Path, relative: str, *, where: str) -> Path:
    relative = _validate_relative_path(relative, where=where)
    root = _absolute_lexical_path(root, where="artifact root")
    root_fd = _open_directory_components_nofollow(root, where="artifact root")
    os.close(root_fd)
    path = root.joinpath(*PurePosixPath(relative).parts)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{where} escapes artifact root") from error
    return _regular_lexical_path(path, where=where)


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
                "cuda_device_count": ledger_entry["cuda_device_count"],
                "max_checkpoint_bytes": ledger_entry["max_checkpoint_bytes"],
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


def validate_parent_trust(trust: ParentTrust) -> ValidatedParent:
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
    actual_checkpoint_path = _regular_lexical_path(
        trust.checkpoint_path, where="parent checkpoint path"
    )
    actual_final_path = _regular_lexical_path(
        trust.finalized_manifest_path, where="parent finalized manifest path"
    )
    if expected_checkpoint_path != actual_checkpoint_path:
        raise ValueError("parent checkpoint path does not match immutable ledger")
    if expected_final_path != actual_final_path:
        raise ValueError("finalized manifest path does not match immutable ledger")

    finalized_value, _raw_sha = strict_json_load_nofollow(trust.finalized_manifest_path)
    if not isinstance(finalized_value, Mapping):
        raise ValueError("finalized checkpoint manifest file must contain an object")
    finalized_sha = validate_manifest(
        finalized_value,
        expected_kind="finalized_checkpoint",
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
        "cuda_device_count",
        "max_checkpoint_bytes",
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
    for field in ("cuda_device_count", "max_checkpoint_bytes"):
        if final_payload[field] != entry[field]:
            raise ValueError(f"finalized checkpoint {field} mismatch")

    parent_checkpoint, parent_sha, parent_size = _load_checkpoint_and_hash(
        trust.checkpoint_path,
        max_checkpoint_bytes=entry["max_checkpoint_bytes"],
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
    parent_expected = CheckpointExpectations(
        mode=entry["arm"],
        np_seed=entry["np_seed"],
        torch_seed=entry["torch_seed"],
        identity_seed=entry["identity_rng_seed"],
        stage=entry["stage"],
        terminal_step=entry["terminal_step"],
        source_sha256=entry["source_sha256"],
        environment_sha256=entry["environment_sha256"],
        prior_sha256=entry["prior_sha256"],
        architecture_sha256=entry["architecture_sha256"],
        optimizer_sha256=entry["optimizer_sha256"],
        scientific_sha256=entry["scientific_sha256"],
        cohort_protocol_sha256=entry["cohort_protocol_sha256"],
        arm_protocol_sha256=entry["arm_protocol_sha256"],
        world_size=entry["world_size"],
        cuda_device_count=entry["cuda_device_count"],
        max_checkpoint_bytes=entry["max_checkpoint_bytes"],
    )
    _validate_loaded_identity_checkpoint(
        parent_checkpoint,
        checkpoint_digest=parent_sha,
        checkpoint_size=parent_size,
        path=Path(trust.checkpoint_path),
        expected=parent_expected,
        validate_exact_parent=False,
    )
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
        "cuda_device_count": manifests["environment"]["payload"].get(
            "visible_cuda_device_count"
        ),
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
    return ValidatedParent(
        manifest=_expected_parent_manifest(
            trust,
            checkpoint_sha256=parent_sha,
            finalized_manifest_sha256=finalized_sha,
            ledger_entry=entry,
        ),
        checkpoint=parent_checkpoint,
        checkpoint_sha256=parent_sha,
        checkpoint_size=parent_size,
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


_PARENT_RECORD_KEYS = {
    "checkpoint_sha256",
    "finalized_manifest_sha256",
    "transaction_ledger_sha256",
    "study_id",
    "arm",
    "stage",
    "terminal_step",
    "upstream_identity",
    "artifact_identity",
    "np_seed",
    "torch_seed",
    "identity_rng_seed",
    "world_size",
    "cuda_device_count",
    "max_checkpoint_bytes",
    "source_sha256",
    "environment_sha256",
    "prior_sha256",
    "architecture_sha256",
    "optimizer_sha256",
    "scientific_sha256",
    "cohort_protocol_sha256",
    "arm_protocol_sha256",
}


def _validate_parent_lineage_structure(
    parent: Mapping[str, Any],
    *,
    expected: CheckpointExpectations,
    manifests: Mapping[str, Mapping[str, Any]],
    study_id: str,
) -> None:
    payload = parent["payload"]
    if expected.stage == "stage1":
        if payload != {"parent": None}:
            raise ValueError("Stage 1 requires an explicit null parent")
        return
    _require_exact_keys(payload, {"parent"}, where="parent manifest payload")
    record = payload["parent"]
    if not isinstance(record, Mapping):
        raise ValueError("Stage 2/3 parent lineage must be an object")
    _require_exact_keys(record, _PARENT_RECORD_KEYS, where="parent lineage")
    for field in (
        "checkpoint_sha256",
        "finalized_manifest_sha256",
        "transaction_ledger_sha256",
        "source_sha256",
        "environment_sha256",
        "prior_sha256",
        "architecture_sha256",
        "optimizer_sha256",
        "scientific_sha256",
        "cohort_protocol_sha256",
        "arm_protocol_sha256",
    ):
        _require_digest(f"parent lineage {field}", record[field])
    for field, minimum in (
        ("terminal_step", 1),
        ("np_seed", 0),
        ("torch_seed", 0),
        ("identity_rng_seed", 0),
        ("world_size", 1),
        ("cuda_device_count", 0),
        ("max_checkpoint_bytes", 1),
    ):
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"parent lineage {field} is invalid")
    for field in ("study_id", "upstream_identity", "artifact_identity"):
        value = record[field]
        if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
            raise ValueError(f"parent lineage {field} is invalid")
    predecessor = "stage1" if expected.stage == "stage2" else "stage2"
    locked = {
        "study_id": study_id,
        "arm": expected.mode,
        "stage": predecessor,
        "np_seed": expected.np_seed,
        "torch_seed": expected.torch_seed,
        "identity_rng_seed": expected.identity_seed,
        "world_size": expected.world_size,
        "cuda_device_count": expected.cuda_device_count,
        "source_sha256": manifests["source"]["sha256"],
        "environment_sha256": manifests["environment"]["sha256"],
        "architecture_sha256": manifests["architecture"]["sha256"],
    }
    for field, value in locked.items():
        if record[field] != value:
            raise ValueError(f"parent {field} lineage does not match child")


def _validate_loaded_identity_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_digest: str,
    checkpoint_size: int,
    path: Path,
    expected: CheckpointExpectations,
    validate_exact_parent: bool,
) -> dict[str, Any]:
    """Validate an already loaded formal checkpoint without reopening its path."""
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
    _validate_finite(checkpoint["state_dict"], path="state_dict")

    provenance = checkpoint["provenance"]
    validate_provenance_bundle(provenance)
    manifests = provenance["manifests"]
    _validate_expected_manifest_hashes(manifests, expected)
    if not isinstance(checkpoint["config"], Mapping):
        raise ValueError("checkpoint config must be an object")
    if checkpoint["config"].get("row_identity_mode") != expected.mode:
        raise ValueError("checkpoint row_identity_mode mismatch")
    actual_architecture = architecture_manifest(
        checkpoint["config"], checkpoint["state_dict"]
    )
    if actual_architecture["sha256"] != manifests["architecture"]["sha256"]:
        raise ValueError("architecture/state_dict key-shape-dtype schema mismatch")
    optimizer_protocol = _validate_optimizer_protocol(manifests["optimizer"]["payload"])
    current_lrs, base_lrs = _validate_optimizer_state(
        checkpoint["optimizer_state"],
        protocol=optimizer_protocol,
        model_state=checkpoint["state_dict"],
        terminal_step=expected.terminal_step,
    )
    _validate_scheduler_state(
        checkpoint["scheduler_state"],
        terminal_step=expected.terminal_step,
        protocol=optimizer_protocol,
        current_lrs=current_lrs,
        base_lrs=base_lrs,
    )
    validate_source_manifest(manifests["source"])
    environment_payload = manifests["environment"]["payload"]
    if (
        environment_payload.get("visible_cuda_device_count")
        != expected.cuda_device_count
    ):
        raise ValueError("environment visible CUDA device count mismatch")
    _validate_scaler_state(checkpoint["scaler_state"], protocol=optimizer_protocol)

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
    _validate_parent_lineage_structure(
        parent,
        expected=expected,
        manifests=manifests,
        study_id=operational["context"]["study_id"],
    )
    if expected.stage == "stage1":
        if expected.parent_trust is not None:
            raise ValueError("Stage 1 requires an explicit null parent")
    elif validate_exact_parent:
        if expected.parent_trust is None:
            raise ValueError("Stage 2/3 requires explicit parent trust inputs")
        trusted_parent = validate_parent_trust(expected.parent_trust).manifest
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
    elif expected.parent_trust is not None:
        raise ValueError(
            "non-recursive parent validation must not receive parent trust"
        )

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
        "max_checkpoint_bytes": expected.max_checkpoint_bytes,
    }


def validate_identity_checkpoint(
    checkpoint_path: str | os.PathLike[str], expected: CheckpointExpectations
) -> dict[str, Any]:
    """Validate one complete formal checkpoint without restoring runtime RNG."""
    path = Path(checkpoint_path)
    checkpoint, checkpoint_digest, checkpoint_size = _load_checkpoint_and_hash(
        path, max_checkpoint_bytes=expected.max_checkpoint_bytes
    )
    return _validate_loaded_identity_checkpoint(
        checkpoint,
        checkpoint_digest=checkpoint_digest,
        checkpoint_size=checkpoint_size,
        path=path,
        expected=expected,
        validate_exact_parent=True,
    )


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


def _open_future_ledger_target(
    root: Path, relative: str, *, where: str
) -> tuple[Path, int, str]:
    """Resolve a future regular-file name beneath a held no-follow parent FD."""
    relative = _validate_relative_path(relative, where=where)
    root = _absolute_lexical_path(root, where="artifact root")
    parts = PurePosixPath(relative).parts
    lexical = root.joinpath(*parts)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    parent_fd = _open_directory_components_nofollow(root, where="artifact root")
    try:
        for component in parts[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        if not stat.S_ISDIR(os.fstat(parent_fd).st_mode):
            raise ValueError(f"{where} parent is not a directory")
        return lexical, parent_fd, parts[-1]
    except OSError as error:
        os.close(parent_fd)
        raise ValueError(
            f"{where} has a symlink or invalid parent path component"
        ) from error


def _prepare_finalized_manifest(
    checkpoint_path: str | os.PathLike[str],
    expected: CheckpointExpectations,
    *,
    finalized_manifest_path: str | os.PathLike[str],
    trust: FinalizationTrust,
) -> tuple[dict[str, Any], int, str, Path]:
    report = validate_identity_checkpoint(checkpoint_path, expected)
    entry = _load_finalization_ledger_entry(
        trust, mode=expected.mode, stage=expected.stage
    )
    expected_checkpoint = _resolved_ledger_path(
        trust.artifact_root, entry["checkpoint_relpath"], where="ledger checkpoint"
    )
    actual_checkpoint = _regular_lexical_path(
        checkpoint_path, where="finalization checkpoint path"
    )
    if actual_checkpoint != expected_checkpoint:
        raise ValueError("checkpoint path does not match immutable finalization ledger")
    expected_final, parent_fd, final_name = _open_future_ledger_target(
        trust.artifact_root,
        entry["finalized_manifest_relpath"],
        where="ledger finalized manifest",
    )
    try:
        actual_final = _absolute_lexical_path(
            finalized_manifest_path, where="finalized manifest path"
        )
        if actual_final != expected_final:
            raise ValueError("finalized manifest path does not match immutable ledger")
        expected_invariants = {
            "terminal_step": expected.terminal_step,
            "np_seed": expected.np_seed,
            "torch_seed": expected.torch_seed,
            "identity_rng_seed": expected.identity_seed,
            "world_size": expected.world_size,
            "cuda_device_count": expected.cuda_device_count,
            "max_checkpoint_bytes": expected.max_checkpoint_bytes,
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
                raise ValueError(
                    f"finalization {field} does not match immutable ledger"
                )
        if report["study_id"] != trust.study_id:
            raise ValueError(
                "finalization study_id does not match checkpoint provenance"
            )
        manifest = make_manifest(
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
                "cuda_device_count": expected.cuda_device_count,
                "max_checkpoint_bytes": expected.max_checkpoint_bytes,
            },
        )
        return manifest, parent_fd, final_name, expected_final
    except BaseException:
        os.close(parent_fd)
        raise


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("write made no progress")
        view = view[written:]


def _strict_json_load_at(
    parent_fd: int, name: str, *, where: str, max_bytes: int = 32 << 20
) -> tuple[Any, str]:
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError(f"{where} is a symlink or invalid regular file") from error
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        os.close(fd)
        raise ValueError(f"{where} must be a regular file")
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
            raise ValueError(f"{where} exceeds {max_bytes} bytes")
        _same_file_snapshot(before, os.fstat(fd), where=where)
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
        raise ValueError(f"invalid strict JSON in {where}: {error}") from error
    return value, hashlib.sha256(raw).hexdigest()


def _publish_json_no_replace_at(
    parent_fd: int,
    final_name: str,
    value: Mapping[str, Any],
    *,
    display_path: Path,
) -> None:
    """Publish complete bytes relative to one verified, held parent directory."""
    if not _LINK_SUPPORTS_DIR_FD or not _UNLINK_SUPPORTS_DIR_FD:
        raise ValueError("dirfd no-replace publication is unavailable")
    payload = canonical_json_bytes(value) + b"\n"
    temp_name = f".{final_name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd = os.open(temp_name, flags, 0o600, dir_fd=parent_fd)
    fd_open = True
    published = False
    try:
        _write_all(fd, payload)
        os.fsync(fd)
        os.close(fd)
        fd_open = False
        try:
            os.link(
                temp_name,
                final_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise ValueError(
                f"finalized manifest already exists: {display_path}"
            ) from error
        published = True
        os.fsync(parent_fd)
        os.unlink(temp_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except BaseException:
        if fd_open:
            os.close(fd)
        try:
            os.unlink(temp_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        # If publication already happened, the final name refers only to the
        # fully-written, file-fsynced inode.  Keep it and report failure; an
        # explicit recovery call can verify identical bytes and fsync the dir.
        if not published:
            try:
                os.fsync(parent_fd)
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
    manifest, parent_fd, final_name, display_path = _prepare_finalized_manifest(
        checkpoint_path,
        expected,
        finalized_manifest_path=finalized_manifest_path,
        trust=trust,
    )
    try:
        _publish_json_no_replace_at(
            parent_fd,
            final_name,
            manifest,
            display_path=display_path,
        )
    finally:
        os.close(parent_fd)
    return manifest


def recover_finalized_checkpoint_manifest(
    checkpoint_path: str | os.PathLike[str],
    expected: CheckpointExpectations,
    *,
    finalized_manifest_path: str | os.PathLike[str],
    trust: FinalizationTrust,
) -> dict[str, Any]:
    """Recover only an identical complete final after a directory-fsync error."""
    manifest, parent_fd, final_name, _display_path = _prepare_finalized_manifest(
        checkpoint_path,
        expected,
        finalized_manifest_path=finalized_manifest_path,
        trust=trust,
    )
    try:
        existing, raw_sha = _strict_json_load_at(
            parent_fd,
            final_name,
            where="existing finalized manifest",
        )
        expected_bytes = canonical_json_bytes(manifest) + b"\n"
        if (
            existing != manifest
            or raw_sha != hashlib.sha256(expected_bytes).hexdigest()
        ):
            raise ValueError(
                "existing finalized manifest is not the exact expected manifest"
            )
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return manifest
