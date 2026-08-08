#!/usr/bin/env python3
"""Immutable cross-seed campaign and predecessor-acceptance registry.

The formal training controller submits one fresh nine-job transaction per
seed.  This module supplies the campaign-level contract that is deliberately
outside those per-seed namespaces:

* one write-once campaign manifest freezes exact training source, environment,
  H100 evidence, checkpoint ceiling, and the static three-stage protocol;
* one write-once evaluation sanity receipt and one write-once acceptance
  record are published for each completed seed; and
* a later seed is authorized only when the acceptance directory contains the
  exact canonical predecessor prefix (none for 42, 42 for 43, and 42/43 for
  44).

Readers use component-wise no-follow traversal, bounded stable reads,
duplicate-key rejection, canonical JSON, and manifest self-hashes.  Publishers
use same-directory hard-link publication so an existing destination is never
replaced.  This is an accidental-drift/same-owner coordination contract, not a
defence against a malicious process running as the same Unix owner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import secrets
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
ARMS = ("rope", "temporary", "none")
STAGES = (("stage1", 500_000), ("stage2", 40_000), ("stage3", 10_000))
FORMAL_SEEDS = (42, 43, 44)
CANONICAL_REPOSITORY = "https://github.com/kikixiong/tabicl-pe.git"
CANONICAL_TRAINING_REF = "refs/heads/codex/position-identity-v1"
CAMPAIGN_MANIFEST_CEILING_BYTES = 131_072
ACCEPTANCE_CEILING_BYTES = 131_072
EVALUATION_RECEIPT_CEILING_BYTES = 131_072
TERMINAL_ATTESTATION_CEILING_BYTES = 131_072
FORMAL_METADATA_CEILING_BYTES = 128 << 20
H100_ATTESTATION_CEILING_BYTES = 8 << 20
H100_SUBMISSION_RECEIPT_CEILING_BYTES = 1 << 20
SOURCE_MANIFEST_CEILING_BYTES = 32 << 20
TRUSTED_EXECUTABLE_CEILING_BYTES = 128 << 20
ENVIRONMENT_COMPLETION_CEILING_BYTES = 1 << 20
ENVIRONMENT_MANIFEST_CEILING_BYTES = 1 << 20
ENVIRONMENT_INVENTORY_CEILING_BYTES = 128 << 20
LOCAL_GIT_CONFIG_OVERRIDES = (
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.filemode=true",
)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_METRIC_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_SLURM_DURATION = re.compile(
    r"^(?:(?P<days>[1-9][0-9]{0,3})-)?"
    r"(?P<hours>[0-9]{2}):(?P<minutes>[0-5][0-9]):(?P<seconds>[0-5][0-9])$"
)
_TRANSACTION_ID = re.compile(r"^[0-9a-f]{32}$")
_JOB_ID = re.compile(r"^[1-9][0-9]{0,19}$")
_SAFE_CLUSTER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_VISIBLE_DEVICE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_GPU_UUID = re.compile(r"^(?:GPU|MIG)-[A-Za-z0-9_.:-]{1,192}$")

_TRAINING_SOURCE_KEYS = {
    "candidate_repository",
    "candidate_ref",
    "commit_sha",
    "tree_sha",
    "source_manifest_sha256",
    "environment_sha256",
}
_STAGE_KEYS = {
    "stage",
    "terminal_step",
    "time_limit",
    "prior_sha256",
    "architecture_sha256",
    "optimizer_sha256",
    "scientific_sha256",
    "static_protocol_sha256",
}
_CAMPAIGN_BINDING_KEYS = {
    "campaign_id",
    "campaign_manifest_sha256",
    "training_commit_sha",
    "training_tree_sha",
    "source_manifest_sha256",
    "environment_sha256",
    "h100_attestation_sha256",
    "nvidia_smi_sha256",
    "checkpoint_ceiling_bytes",
    "static_protocol_sha256_by_stage",
    "time_limit_by_stage",
    "predecessor_acceptance_sha256_by_seed",
}
_CAMPAIGN_VALIDATION_KEYS = {
    "exact_root",
    "source_manifest_path",
    "h100_attestation_path",
    "h100_submission_receipt_path",
    "environment_completion_path",
    "git_path",
    "git_sha256",
    "h100_attestation_max_bytes",
    "h100_submission_receipt_max_bytes",
    "source_manifest_max_bytes",
    "environment_completion_max_bytes",
    "environment_manifest_max_bytes",
    "environment_inventory_max_bytes",
    "h100_submission_receipt_sha256",
    "h100_validation_report_sha256",
    "environment_transaction_sha256",
    "environment_transaction_completion_raw_sha256",
    "two_gpu_environment_sha256",
    "sacct_sha256",
    "repository_identity_sha256",
    "repository_query_sha256",
}
_TERMINAL_JOB_KEYS = {
    "arm",
    "stage",
    "job_id",
    "cluster",
    "job_name",
    "state",
    "exit_code",
    "derived_exit_code",
    "completion",
    "scheduler_logs",
    "finalized_artifact",
    "parent_lineage",
}
_TERMINAL_COMPLETION_KEYS = {
    "path",
    "manifest_sha256",
    "file_sha256",
    "stdout_observed_size",
    "stderr_observed_size",
    "seed",
    "scheduler",
    "cuda_visible_devices",
    "gpu_name",
    "gpu_uuid",
    "gpu_driver_version",
    "repository_binding",
    "manifest_ceiling_bytes",
    "protocol_metadata_allowance_bytes",
}
_TERMINAL_LOG_KEYS = {
    "stream",
    "path",
    "size",
    "sha256",
    "fatal_signature_scan_passed",
    "stable_reads",
}
_TERMINAL_FINALIZED_KEYS = {
    "binding",
    "finalized_manifest_file_sha256",
    "finalized_manifest_size",
    "independent_stable_validations",
    "initial_validation_sha256",
    "publication_validation_sha256",
}
_TERMINAL_QUERY_KEYS = {
    "cluster",
    "job_ids",
    "query_argv_sha256",
    "stdout_sha256",
    "stderr_sha256",
}
_TERMINAL_RAW_EVIDENCE_KEYS = {
    "artifact_root",
    "submission_receipt_path",
    "transaction_ledger_path",
    "protocol_metadata_allowance_bytes",
}
_CAMPAIGN_REQUIRED_SOURCE_PATHS = frozenset(
    {
        "scripts/finalize_formal_scheduler_logs.py",
        "scripts/formal_campaign_registry.py",
        "scripts/run_h100_identity_validation.py",
        "scripts/verify_formal_environment_transaction.py",
        "scripts/verify_git_repository.py",
        "scripts/verify_runtime_source.py",
    }
)


def _normalize(value: Any, *, where: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where} contains a non-finite number")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or "\x00" in key:
                raise ValueError(f"{where} has an invalid object key")
            result[key] = _normalize(item, where=f"{where}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _normalize(item, where=f"{where}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{where} contains unsupported type {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def make_manifest(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(kind, str) or not kind:
        raise ValueError("manifest kind is invalid")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "payload": _normalize(payload, where=f"{kind}.payload"),
    }
    return {**body, "sha256": canonical_sha256(body)}


def _exact(value: Any, keys: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{where} schema mismatch; missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )
    return value


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _git_oid(value: Any, where: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        raise ValueError(f"{where} must be a lowercase 40-character Git object ID")
    return value


def _safe_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{where} is invalid")
    return value


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _formal_seed(value: Any, where: str = "formal seed") -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in FORMAL_SEEDS
    ):
        raise ValueError(f"{where} must be exactly one of 42, 43, or 44")
    return value


def _slurm_duration(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a canonical Slurm duration")
    match = _SLURM_DURATION.fullmatch(value)
    if match is None or int(match.group("hours")) > 23:
        raise ValueError(f"{where} must be a canonical Slurm duration")
    seconds = (
        (int(match.group("days") or "0") * 24 + int(match.group("hours"))) * 3600
        + int(match.group("minutes")) * 60
        + int(match.group("seconds"))
    )
    if seconds <= 0:
        raise ValueError(f"{where} must be a positive Slurm duration")
    return value


def _absolute_path(value: Any, where: str) -> Path:
    if (
        not isinstance(value, (str, os.PathLike))
        or "\x00" in os.fspath(value)
        or "\n" in os.fspath(value)
        or "\r" in os.fspath(value)
    ):
        raise ValueError(f"{where} must be a one-line path")
    path = Path(value)
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{where} must be an absolute normalized path")
    return path


def _open_directory_nofollow(path: Path, *, where: str) -> int:
    path = _absolute_path(path, where)
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ValueError("no-follow directory traversal is unavailable")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    fd = os.open(os.path.sep, flags)
    try:
        for component in path.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except Exception:
        os.close(fd)
        raise


def _stat_signature(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"manifest contains duplicate key {key!r}")
        result[key] = value
    return result


def read_canonical_manifest(
    path: str | os.PathLike[str],
    *,
    max_bytes: int,
    expected_kind: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Read one canonical newline-terminated manifest through no-follow FDs."""

    file_path = _absolute_path(path, "manifest path")
    ceiling = _positive_int(max_bytes, "manifest byte ceiling")
    parent_fd = _open_directory_nofollow(file_path.parent, where="manifest parent")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(file_path.name, flags, dir_fd=parent_fd)
        try:
            before = os.fstat(fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size < 2
                or before.st_size > ceiling
            ):
                raise ValueError("manifest is not a bounded singly-linked regular file")
            chunks: list[bytes] = []
            remaining = ceiling + 1
            while remaining:
                chunk = os.read(fd, min(1 << 20, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(fd)
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)
    if _stat_signature(before) != _stat_signature(after):
        raise ValueError("manifest changed during its bounded read")
    if len(raw) != before.st_size or len(raw) > ceiling:
        raise ValueError("manifest exceeds its byte ceiling or was not read completely")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("manifest is not strict UTF-8 JSON") from error
    manifest = _exact(
        value,
        {"schema_version", "kind", "payload", "sha256"},
        "manifest envelope",
    )
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["kind"] != expected_kind
    ):
        raise ValueError("manifest schema or kind mismatch")
    body = {
        "schema_version": manifest["schema_version"],
        "kind": manifest["kind"],
        "payload": manifest["payload"],
    }
    actual_sha256 = canonical_sha256(body)
    if manifest["sha256"] != actual_sha256:
        raise ValueError("manifest self-hash mismatch")
    if expected_sha256 is not None and actual_sha256 != _digest(
        expected_sha256, "expected manifest digest"
    ):
        raise ValueError("manifest does not match its externally committed digest")
    if raw != canonical_json_bytes(manifest) + b"\n":
        raise ValueError("manifest is not canonical newline-terminated JSON")
    return dict(manifest)


def publish_manifest_no_replace(
    path: str | os.PathLike[str],
    manifest: Mapping[str, Any],
    *,
    max_bytes: int,
) -> None:
    """Durably publish a canonical manifest without replacing any destination."""

    file_path = _absolute_path(path, "publication path")
    ceiling = _positive_int(max_bytes, "publication byte ceiling")
    payload = canonical_json_bytes(manifest) + b"\n"
    if len(payload) > ceiling:
        raise ValueError("manifest exceeds its publication byte ceiling")
    parent_fd = _open_directory_nofollow(file_path.parent, where="publication parent")
    temporary = f".{file_path.name}.tmp-{secrets.token_hex(16)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd: int | None = None
    linked = False
    try:
        fd = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write while publishing manifest")
            view = view[written:]
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != len(payload):
            raise ValueError(
                "published temporary manifest is not a stable regular file"
            )
        os.link(
            temporary,
            file_path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        linked = True
        os.unlink(temporary, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        if fd is not None:
            os.close(fd)
        if not linked:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _validated_training_source(value: Any) -> dict[str, str]:
    source = _exact(value, _TRAINING_SOURCE_KEYS, "campaign training source")
    repository = source["candidate_repository"]
    reference = source["candidate_ref"]
    if repository != CANONICAL_REPOSITORY:
        raise ValueError("campaign training repository is not canonical")
    if reference != CANONICAL_TRAINING_REF:
        raise ValueError("campaign training ref is not canonical")
    return {
        "candidate_repository": repository,
        "candidate_ref": reference,
        "commit_sha": _git_oid(source["commit_sha"], "campaign training commit"),
        "tree_sha": _git_oid(source["tree_sha"], "campaign training tree"),
        "source_manifest_sha256": _digest(
            source["source_manifest_sha256"], "campaign source manifest"
        ),
        "environment_sha256": _digest(
            source["environment_sha256"], "campaign environment"
        ),
    }


def _static_protocol(
    *,
    source: Mapping[str, str],
    stage: str,
    terminal_step: int,
    time_limit: str,
    prior_sha256: str,
    architecture_sha256: str,
    optimizer_sha256: str,
    scientific_sha256: str,
) -> str:
    return make_manifest(
        "formal_campaign_stage_static_protocol",
        {
            "training_commit_sha": source["commit_sha"],
            "training_tree_sha": source["tree_sha"],
            "source_manifest_sha256": source["source_manifest_sha256"],
            "environment_sha256": source["environment_sha256"],
            "stage": stage,
            "terminal_step": terminal_step,
            "time_limit": time_limit,
            "prior_sha256": prior_sha256,
            "architecture_sha256": architecture_sha256,
            "optimizer_sha256": optimizer_sha256,
            "scientific_sha256": scientific_sha256,
        },
    )["sha256"]


def _validated_campaign_validation_evidence(
    value: Any,
) -> dict[str, Any]:
    evidence = _exact(
        value,
        _CAMPAIGN_VALIDATION_KEYS,
        "campaign validation evidence",
    )
    paths = {
        key: str(_absolute_path(evidence[key], f"campaign {key}"))
        for key in (
            "exact_root",
            "source_manifest_path",
            "h100_attestation_path",
            "h100_submission_receipt_path",
            "environment_completion_path",
            "git_path",
        )
    }
    if Path(paths["source_manifest_path"]) == Path(paths["h100_attestation_path"]):
        raise ValueError("campaign source and H100 evidence paths must differ")
    exact_root = Path(paths["exact_root"])
    for key in (
        "source_manifest_path",
        "h100_attestation_path",
        "h100_submission_receipt_path",
        "environment_completion_path",
    ):
        candidate = Path(paths[key])
        if candidate == exact_root or exact_root in candidate.parents:
            raise ValueError("campaign external evidence must be outside exact T")
    ceilings = {
        key: _positive_int(evidence[key], f"campaign {key}")
        for key in (
            "h100_attestation_max_bytes",
            "h100_submission_receipt_max_bytes",
            "source_manifest_max_bytes",
            "environment_completion_max_bytes",
            "environment_manifest_max_bytes",
            "environment_inventory_max_bytes",
        )
    }
    if (
        ceilings["h100_attestation_max_bytes"] > H100_ATTESTATION_CEILING_BYTES
        or ceilings["h100_submission_receipt_max_bytes"]
        > H100_SUBMISSION_RECEIPT_CEILING_BYTES
        or ceilings["source_manifest_max_bytes"] > SOURCE_MANIFEST_CEILING_BYTES
        or ceilings["environment_completion_max_bytes"]
        != ENVIRONMENT_COMPLETION_CEILING_BYTES
        or ceilings["environment_manifest_max_bytes"]
        != ENVIRONMENT_MANIFEST_CEILING_BYTES
        or ceilings["environment_inventory_max_bytes"]
        != ENVIRONMENT_INVENTORY_CEILING_BYTES
    ):
        raise ValueError("campaign evidence byte ceiling exceeds its fixed maximum")
    digests = {
        key: _digest(evidence[key], f"campaign {key}")
        for key in _CAMPAIGN_VALIDATION_KEYS
        if key.endswith("sha256")
    }
    if digests["git_sha256"] == "0" * 64:
        raise ValueError("campaign Git executable digest is invalid")
    return {**paths, **ceilings, **digests}


def _open_trusted_executable(path: Path, *, expected_sha256: str, where: str) -> int:
    executable = _absolute_path(path, where)
    expected = _digest(expected_sha256, f"expected {where}")
    parent_fd = _open_directory_nofollow(executable.parent, where=f"{where} parent")
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(executable.name, flags, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_mode & 0o111 == 0
            or before.st_size < 1
            or before.st_size > TRUSTED_EXECUTABLE_CEILING_BYTES
        ):
            raise ValueError(f"{where} is not a bounded executable regular file")
        digest = hashlib.sha256()
        while chunk := os.read(fd, 1 << 20):
            digest.update(chunk)
        after = os.fstat(fd)
        path_after = os.stat(executable, follow_symlinks=False)
        if (
            _stat_signature(before) != _stat_signature(after)
            or after.st_dev != path_after.st_dev
            or after.st_ino != path_after.st_ino
            or digest.hexdigest() != expected
        ):
            raise ValueError(f"{where} path identity or digest mismatch")
        os.lseek(fd, 0, os.SEEK_SET)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _git_from_fd(
    fd: int, root: Path, *arguments: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [
            f"/proc/self/fd/{fd}",
            *LOCAL_GIT_CONFIG_OVERRIDES,
            "-C",
            os.fspath(root),
            *arguments,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        pass_fds=(fd,),
        env={
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "LANG": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_COUNT": "0",
            "GIT_CEILING_DIRECTORIES": "/",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_PROTOCOL_FROM_USER": "0",
            "GIT_ALLOW_PROTOCOL": "https",
        },
        timeout=30,
    )
    if check and completed.returncode != 0:
        raise ValueError(
            f"exact-T Git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed


def _require_campaign_manifest_coverage(
    verifier: Any, manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    return verifier.require_manifest_coverage(
        manifest,
        required_paths=tuple(sorted(_CAMPAIGN_REQUIRED_SOURCE_PATHS)),
        required_code_roots=("scripts", "src/tabicl"),
    )


def _attest_exact_campaign_source(
    *,
    exact_root: Path,
    source_manifest_path: Path,
    source_manifest_max_bytes: int,
    source: Mapping[str, str],
    git_path: Path,
    git_sha256: str,
) -> Mapping[str, Any]:
    root = _absolute_path(exact_root, "campaign exact root")
    expected_controller = root / "scripts/formal_campaign_registry.py"
    if Path(__file__).resolve(strict=True) != expected_controller.resolve(strict=True):
        raise ValueError("campaign publisher is not running from exact T")
    root_fd = _open_directory_nofollow(root, where="campaign exact root")
    os.close(root_fd)
    git_fd = _open_trusted_executable(
        git_path, expected_sha256=git_sha256, where="campaign Git executable"
    )
    try:
        if (
            _git_from_fd(git_fd, root, "rev-parse", "HEAD^{commit}").stdout.strip()
            != source["commit_sha"]
        ):
            raise ValueError("campaign exact checkout commit mismatch")
        if (
            _git_from_fd(git_fd, root, "rev-parse", "HEAD^{tree}").stdout.strip()
            != source["tree_sha"]
        ):
            raise ValueError("campaign exact checkout tree mismatch")
        if _git_from_fd(
            git_fd,
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).stdout:
            raise ValueError("campaign exact checkout is dirty")
        symbolic = _git_from_fd(git_fd, root, "symbolic-ref", "-q", "HEAD", check=False)
        if symbolic.returncode == 0:
            raise ValueError("campaign exact checkout must be detached")
        if symbolic.returncode != 1:
            raise ValueError("campaign exact checkout detached state is unknown")
    finally:
        os.close(git_fd)

    source_manifest_path = _absolute_path(
        source_manifest_path, "campaign source manifest path"
    )
    if source_manifest_max_bytes > SOURCE_MANIFEST_CEILING_BYTES:
        raise ValueError("campaign source manifest byte ceiling is too large")
    verifier_path = root / "scripts/verify_runtime_source.py"
    spec = importlib.util.spec_from_file_location(
        "_formal_campaign_source_verifier", verifier_path
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load exact-T source verifier")
    verifier = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(verifier)
    finally:
        sys.dont_write_bytecode = previous
    if Path(verifier.__file__).resolve(strict=True) != verifier_path.resolve(
        strict=True
    ):
        raise ValueError("source verifier was loaded outside exact T")
    manifest = verifier.load_and_validate_source_manifest(
        source_manifest_path,
        source["source_manifest_sha256"],
        source["commit_sha"],
        source["tree_sha"],
    )
    raw_size = source_manifest_path.stat(follow_symlinks=False).st_size
    if raw_size > source_manifest_max_bytes:
        raise ValueError("campaign source manifest exceeds its byte ceiling")
    _require_campaign_manifest_coverage(verifier, manifest)
    verifier.verify_archive(root, manifest)
    return manifest


def _load_exact_h100_contract(exact_root: Path):
    path = exact_root / "scripts/run_h100_identity_validation.py"
    spec = importlib.util.spec_from_file_location(
        "_formal_campaign_h100_validation", path
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load exact-T H100 validation helper")
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(helper)
    finally:
        sys.dont_write_bytecode = previous
    if Path(helper.__file__).resolve(strict=True) != path.resolve(strict=True):
        raise ValueError("H100 validator was loaded outside exact T")
    return helper


def _load_exact_repository_contract(exact_root: Path):
    path = exact_root / "scripts/verify_git_repository.py"
    spec = importlib.util.spec_from_file_location(
        "_formal_campaign_repository_query", path
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load exact-T repository query verifier")
    verifier = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(verifier)
    finally:
        sys.dont_write_bytecode = previous
    if Path(verifier.__file__).resolve(strict=True) != path.resolve(strict=True):
        raise ValueError("repository query verifier was loaded outside exact T")
    return verifier


def _load_exact_environment_transaction_contract(exact_root: Path):
    path = exact_root / "scripts/verify_formal_environment_transaction.py"
    spec = importlib.util.spec_from_file_location(
        "_formal_campaign_environment_transaction", path
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load exact-T environment transaction verifier")
    verifier = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(verifier)
    finally:
        sys.dont_write_bytecode = previous
    if Path(verifier.__file__).resolve(strict=True) != path.resolve(strict=True):
        raise ValueError("environment transaction verifier was loaded outside exact T")
    return verifier


def _revalidate_raw_environment_transaction(
    *,
    exact_root: Path,
    completion_path: Path,
    completion_max_bytes: int,
    manifest_max_bytes: int,
    inventory_max_bytes: int,
    receipt: Mapping[str, Any],
    source: Mapping[str, str],
) -> Mapping[str, Any]:
    summary = receipt["environment_transaction"]
    verifier = _load_exact_environment_transaction_contract(exact_root)
    rebuilt = verifier._verify(
        completion_path=_absolute_path(
            completion_path, "campaign environment completion marker"
        ),
        expected_completion_sha256=summary["completion_raw_sha256"],
        expected_transaction_sha256=summary["transaction_sha256"],
        expected_one_gpu_environment_sha256=receipt["environment_sha256_by_world_size"][
            "1"
        ],
        expected_two_gpu_environment_sha256=receipt["environment_sha256_by_world_size"][
            "2"
        ],
        expected_inventory_sha256=summary["inventory_sha256"],
        expected_source_commit_sha=source["commit_sha"],
        expected_source_tree_sha=source["tree_sha"],
        expected_git_sha256=receipt["repository_binding"]["git_sha256"],
        completion_max_bytes=completion_max_bytes,
        manifest_max_bytes=manifest_max_bytes,
        inventory_max_bytes=inventory_max_bytes,
    )
    if rebuilt != summary:
        raise ValueError(
            "raw environment transaction differs from H100 submission receipt"
        )
    return rebuilt


def _cross_validate_h100_receipt(
    *,
    attestation: Mapping[str, Any],
    receipt: Mapping[str, Any],
    source: Mapping[str, str],
) -> None:
    payload = attestation["payload"]
    if (
        payload["submission_receipt_sha256"] != receipt["sha256"]
        or payload["repository_binding"] != receipt["repository_binding"]
        or payload["nvidia_smi_sha256"] != receipt["nvidia_smi_sha256"]
        or payload["environment_sha256"]
        != receipt["environment_sha256_by_world_size"]["1"]
        or payload["environment_sha256"] != source["environment_sha256"]
    ):
        raise ValueError("H100 attestation and submission receipt binding mismatch")
    cases = payload["cases"]
    if {item["case_id"] for item in cases} != set(receipt["artifact_identities"]):
        raise ValueError("H100 attestation and receipt case sets differ")
    observed_environments: dict[str, str] = {}
    for item in cases:
        case_id = item["case_id"]
        if item["artifact_identity_sha256"] != receipt["artifact_identities"][case_id]:
            raise ValueError("H100 artifact identity differs from submission receipt")
        runtime = item["runtime_evidence"]["payload"]
        scheduler = runtime["scheduler_binding"]
        expected = receipt["job_bindings"][case_id]
        observed = {
            "job_id": scheduler["job_id"],
            "cluster": scheduler["slurm_receipt_cluster"],
            "requested_resource": scheduler["requested_resource"],
            "requested_resource_sha256": scheduler["requested_resource_sha256"],
            "held_plan_sha256": scheduler["held_plan_sha256"],
            "scontrol_sha256": scheduler["scontrol_sha256"],
        }
        if observed != expected:
            raise ValueError("H100 runtime scheduler evidence differs from receipt")
        world_size = str(item["world_size"])
        environment_sha = runtime["environment"]["sha256"]
        previous_environment = observed_environments.setdefault(
            world_size, environment_sha
        )
        if previous_environment != environment_sha:
            raise ValueError("H100 runtime environment differs within one world size")
    if observed_environments != receipt["environment_sha256_by_world_size"]:
        raise ValueError("H100 runtime environments differ from submission receipt")


def _derive_campaign_validation_evidence(
    *,
    exact_root: Path,
    source_manifest_path: Path,
    h100_attestation_path: Path,
    h100_submission_receipt_path: Path,
    environment_completion_path: Path,
    git_path: Path,
    git_sha256: str,
    h100_attestation_max_bytes: int,
    h100_submission_receipt_max_bytes: int,
    source_manifest_max_bytes: int,
    environment_completion_max_bytes: int,
    environment_manifest_max_bytes: int,
    environment_inventory_max_bytes: int,
    source: Mapping[str, str],
    expected_h100_sha256: str,
    checkpoint_ceiling_bytes: int,
    expected_gpu_model: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    ceilings = {
        "H100 attestation": (
            _positive_int(h100_attestation_max_bytes, "H100 attestation byte ceiling"),
            H100_ATTESTATION_CEILING_BYTES,
        ),
        "H100 submission receipt": (
            _positive_int(
                h100_submission_receipt_max_bytes,
                "H100 submission receipt byte ceiling",
            ),
            H100_SUBMISSION_RECEIPT_CEILING_BYTES,
        ),
        "source manifest": (
            _positive_int(source_manifest_max_bytes, "source manifest byte ceiling"),
            SOURCE_MANIFEST_CEILING_BYTES,
        ),
        "environment completion": (
            _positive_int(
                environment_completion_max_bytes,
                "environment completion byte ceiling",
            ),
            ENVIRONMENT_COMPLETION_CEILING_BYTES,
        ),
        "environment manifest": (
            _positive_int(
                environment_manifest_max_bytes,
                "environment manifest byte ceiling",
            ),
            ENVIRONMENT_MANIFEST_CEILING_BYTES,
        ),
        "environment inventory": (
            _positive_int(
                environment_inventory_max_bytes,
                "environment inventory byte ceiling",
            ),
            ENVIRONMENT_INVENTORY_CEILING_BYTES,
        ),
    }
    if any(value > maximum for value, maximum in ceilings.values()) or any(
        ceilings[label][0] != ceilings[label][1]
        for label in (
            "environment completion",
            "environment manifest",
            "environment inventory",
        )
    ):
        raise ValueError("campaign evidence byte ceiling exceeds its fixed maximum")
    root = _absolute_path(exact_root, "campaign exact root")
    completion_path = _absolute_path(
        environment_completion_path,
        "campaign environment completion marker",
    )
    if completion_path == root or root in completion_path.parents:
        raise ValueError("campaign environment evidence must be outside exact T")
    _attest_exact_campaign_source(
        exact_root=root,
        source_manifest_path=source_manifest_path,
        source_manifest_max_bytes=source_manifest_max_bytes,
        source=source,
        git_path=git_path,
        git_sha256=git_sha256,
    )
    helper = _load_exact_h100_contract(root)
    receipt = helper.validate_gate_submission_receipt(
        _absolute_path(h100_submission_receipt_path, "H100 submission receipt path"),
        expected_commit_sha=source["commit_sha"],
        expected_tree_sha=source["tree_sha"],
        expected_source_manifest_sha256=source["source_manifest_sha256"],
        expected_checkpoint_ceiling_bytes=checkpoint_ceiling_bytes,
        max_bytes=h100_submission_receipt_max_bytes,
    )
    repository_verifier = _load_exact_repository_contract(root)
    observed_repository = repository_verifier.query_exact_repository(
        git=_absolute_path(git_path, "campaign Git executable"),
        git_sha256=_digest(git_sha256, "campaign Git executable"),
        expected_commit_sha=source["commit_sha"],
    )
    if observed_repository != receipt["repository_binding"]:
        raise ValueError("fresh repository query differs from H100 submission receipt")
    if receipt["repository_binding"]["git_sha256"] != _digest(
        git_sha256, "campaign Git executable"
    ):
        raise ValueError("campaign and H100 receipt use different Git executables")
    if receipt["environment_sha256_by_world_size"]["1"] != source["environment_sha256"]:
        raise ValueError("campaign training environment differs from H100 receipt")
    _revalidate_raw_environment_transaction(
        exact_root=root,
        completion_path=completion_path,
        completion_max_bytes=environment_completion_max_bytes,
        manifest_max_bytes=environment_manifest_max_bytes,
        inventory_max_bytes=environment_inventory_max_bytes,
        receipt=receipt,
        source=source,
    )
    attestation = read_canonical_manifest(
        h100_attestation_path,
        max_bytes=h100_attestation_max_bytes,
        expected_kind="h100_identity_smoke",
        expected_sha256=expected_h100_sha256,
    )
    report = helper.validate_smoke_attestation(
        attestation,
        expected_sha256=expected_h100_sha256,
        expected_commit_sha=source["commit_sha"],
        expected_tree_sha=source["tree_sha"],
        expected_environment_sha256=source["environment_sha256"],
        expected_source_manifest_sha256=source["source_manifest_sha256"],
        expected_gpu_model=expected_gpu_model,
        expected_checkpoint_ceiling_bytes=checkpoint_ceiling_bytes,
        expected_submission_receipt_sha256=receipt["sha256"],
        expected_sacct_sha256=receipt["sacct_sha256"],
        expected_git_sha256=receipt["repository_binding"]["git_sha256"],
        expected_repository_identity_sha256=receipt["repository_binding"][
            "repository_identity_sha256"
        ],
        expected_repository_query_sha256=receipt["repository_binding"]["query_sha256"],
        expected_nvidia_smi_sha256=receipt["nvidia_smi_sha256"],
    )
    _cross_validate_h100_receipt(
        attestation=attestation,
        receipt=receipt,
        source=source,
    )
    transaction = receipt["environment_transaction"]
    evidence = {
        "exact_root": str(root),
        "source_manifest_path": str(
            _absolute_path(source_manifest_path, "campaign source manifest path")
        ),
        "h100_attestation_path": str(
            _absolute_path(h100_attestation_path, "H100 attestation path")
        ),
        "h100_submission_receipt_path": str(
            _absolute_path(h100_submission_receipt_path, "H100 submission receipt path")
        ),
        "environment_completion_path": str(completion_path),
        "git_path": str(_absolute_path(git_path, "campaign Git executable")),
        "git_sha256": receipt["repository_binding"]["git_sha256"],
        "h100_attestation_max_bytes": h100_attestation_max_bytes,
        "h100_submission_receipt_max_bytes": h100_submission_receipt_max_bytes,
        "source_manifest_max_bytes": source_manifest_max_bytes,
        "environment_completion_max_bytes": environment_completion_max_bytes,
        "environment_manifest_max_bytes": environment_manifest_max_bytes,
        "environment_inventory_max_bytes": environment_inventory_max_bytes,
        "h100_submission_receipt_sha256": receipt["sha256"],
        "h100_validation_report_sha256": canonical_sha256(report),
        "environment_transaction_sha256": canonical_sha256(transaction),
        "environment_transaction_completion_raw_sha256": transaction[
            "completion_raw_sha256"
        ],
        "two_gpu_environment_sha256": receipt["environment_sha256_by_world_size"]["2"],
        "sacct_sha256": receipt["sacct_sha256"],
        "repository_identity_sha256": receipt["repository_binding"][
            "repository_identity_sha256"
        ],
        "repository_query_sha256": receipt["repository_binding"]["query_sha256"],
    }
    return evidence, attestation


def validate_campaign_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    envelope = _exact(
        manifest,
        {"schema_version", "kind", "payload", "sha256"},
        "campaign manifest",
    )
    if envelope["schema_version"] != 1 or envelope["kind"] != "formal_campaign":
        raise ValueError("campaign manifest schema or kind mismatch")
    body = {key: envelope[key] for key in ("schema_version", "kind", "payload")}
    if envelope["sha256"] != canonical_sha256(body):
        raise ValueError("campaign manifest self-hash mismatch")
    payload = _exact(
        envelope["payload"],
        {
            "campaign_id",
            "training_source",
            "h100_gate",
            "validation_evidence",
            "supported_seeds",
            "formal_arms",
            "stages",
        },
        "campaign payload",
    )
    campaign_id = _safe_id(payload["campaign_id"], "campaign_id")
    source = _validated_training_source(payload["training_source"])
    gate = _exact(
        payload["h100_gate"],
        {
            "attestation_sha256",
            "checkpoint_ceiling_bytes",
            "observed_checkpoint_max_bytes",
            "nvidia_smi_sha256",
            "gpu_model",
            "driver_version",
        },
        "campaign H100 gate",
    )
    h100_sha256 = _digest(gate["attestation_sha256"], "campaign H100 attestation")
    nvidia_smi_sha256 = _digest(
        gate["nvidia_smi_sha256"], "campaign nvidia-smi executable"
    )
    checkpoint_ceiling = _positive_int(
        gate["checkpoint_ceiling_bytes"], "campaign checkpoint ceiling"
    )
    observed_max = _positive_int(
        gate["observed_checkpoint_max_bytes"], "campaign observed checkpoint maximum"
    )
    if observed_max > checkpoint_ceiling:
        raise ValueError("campaign H100 checkpoint maximum exceeds its ceiling")
    if (
        not isinstance(gate["gpu_model"], str)
        or not gate["gpu_model"]
        or "\n" in gate["gpu_model"]
        or "\r" in gate["gpu_model"]
        or "H100" not in gate["gpu_model"]
    ):
        raise ValueError("campaign H100 GPU model is invalid")
    if (
        not isinstance(gate["driver_version"], str)
        or not gate["driver_version"]
        or len(gate["driver_version"]) > 128
        or any(
            character in gate["driver_version"] for character in ("\x00", "\n", "\r")
        )
    ):
        raise ValueError("campaign H100 driver version is invalid")
    validation_evidence = _validated_campaign_validation_evidence(
        payload["validation_evidence"]
    )
    if validation_evidence["h100_submission_receipt_sha256"] == h100_sha256:
        raise ValueError("campaign H100 attestation and receipt digests must differ")
    if payload["supported_seeds"] != list(FORMAL_SEEDS):
        raise ValueError("campaign supported seeds must be exactly [42,43,44]")
    if payload["formal_arms"] != list(ARMS):
        raise ValueError("campaign arms must be exactly rope, temporary, none")
    stages = payload["stages"]
    if not isinstance(stages, list) or len(stages) != len(STAGES):
        raise ValueError("campaign stages must contain the exact three-stage protocol")
    normalized_stages: list[dict[str, Any]] = []
    for raw, expected in zip(stages, STAGES):
        stage = _exact(raw, _STAGE_KEYS, "campaign stage")
        expected_stage, expected_step = expected
        if stage["stage"] != expected_stage or stage["terminal_step"] != expected_step:
            raise ValueError("campaign stage order or terminal budget is not canonical")
        time_limit = _slurm_duration(
            stage["time_limit"], f"{expected_stage} time limit"
        )
        prior = _digest(stage["prior_sha256"], f"{expected_stage} prior")
        architecture = _digest(
            stage["architecture_sha256"], f"{expected_stage} architecture"
        )
        optimizer = _digest(stage["optimizer_sha256"], f"{expected_stage} optimizer")
        scientific = _digest(
            stage["scientific_sha256"], f"{expected_stage} scientific config"
        )
        expected_static = _static_protocol(
            source=source,
            stage=expected_stage,
            terminal_step=expected_step,
            time_limit=time_limit,
            prior_sha256=prior,
            architecture_sha256=architecture,
            optimizer_sha256=optimizer,
            scientific_sha256=scientific,
        )
        if stage["static_protocol_sha256"] != expected_static:
            raise ValueError(f"{expected_stage} static protocol digest mismatch")
        normalized_stages.append(dict(stage))
    return {
        "campaign_id": campaign_id,
        "manifest_sha256": envelope["sha256"],
        "training_source": source,
        "h100_gate": {
            "attestation_sha256": h100_sha256,
            "checkpoint_ceiling_bytes": checkpoint_ceiling,
            "observed_checkpoint_max_bytes": observed_max,
            "nvidia_smi_sha256": nvidia_smi_sha256,
            "gpu_model": gate["gpu_model"],
            "driver_version": gate["driver_version"],
        },
        "validation_evidence": validation_evidence,
        "stages": normalized_stages,
    }


def build_campaign_manifest(
    *,
    campaign_id: str,
    training_source: Mapping[str, Any],
    h100_attestation: Mapping[str, Any],
    expected_h100_sha256: str,
    checkpoint_ceiling_bytes: int,
    expected_gpu_model: str,
    stages: Sequence[Mapping[str, Any]],
    exact_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Build an explicitly non-authorizable draft for tests/schema tooling.

    A draft never has kind ``formal_campaign`` and therefore cannot pass
    :func:`validate_campaign_manifest`, seed authorization, or acceptance.
    Production callers must use :func:`publish_campaign`, which consumes the
    real exact-T source, H100 receipt, and H100 attestation before publication.
    """

    source = _validated_training_source(training_source)
    h100_sha = _digest(expected_h100_sha256, "expected H100 attestation")
    ceiling = _positive_int(checkpoint_ceiling_bytes, "checkpoint ceiling")
    if exact_root is not None:
        raise ValueError("draft builder cannot validate or publish a formal campaign")
    envelope = _exact(
        h100_attestation,
        {"schema_version", "kind", "payload", "sha256"},
        "H100 attestation",
    )
    if (
        envelope["schema_version"] != 1
        or envelope["kind"] != "h100_identity_smoke"
        or envelope["sha256"] != h100_sha
        or envelope["sha256"]
        != canonical_sha256(
            {key: envelope[key] for key in ("schema_version", "kind", "payload")}
        )
    ):
        raise ValueError("H100 attestation envelope or digest mismatch")
    smoke = envelope["payload"]
    for field, expected in (
        ("commit_sha", source["commit_sha"]),
        ("tree_sha", source["tree_sha"]),
        ("source_manifest_sha256", source["source_manifest_sha256"]),
        ("environment_sha256", source["environment_sha256"]),
        ("checkpoint_ceiling_bytes", ceiling),
        ("gpu_model", expected_gpu_model),
    ):
        if not isinstance(smoke, Mapping) or smoke.get(field) != expected:
            raise ValueError(f"H100 attestation {field} differs from campaign")
    observed_max = _positive_int(
        smoke.get("observed_checkpoint_max_bytes"), "H100 observed checkpoint maximum"
    )
    nvidia_smi_sha256 = _digest(
        smoke.get("nvidia_smi_sha256"), "H100 nvidia-smi executable"
    )
    if observed_max > ceiling:
        raise ValueError("H100 observed checkpoint maximum exceeds campaign ceiling")
    driver_version = smoke.get("driver_version")
    if (
        not isinstance(driver_version, str)
        or not driver_version
        or len(driver_version) > 128
        or any(character in driver_version for character in ("\x00", "\n", "\r"))
    ):
        raise ValueError("H100 attestation driver version is invalid")
    if len(stages) != len(STAGES):
        raise ValueError("campaign must supply exactly three stage protocols")
    normalized_stages: list[dict[str, Any]] = []
    for supplied, (expected_stage, expected_step) in zip(stages, STAGES):
        raw = _exact(
            supplied,
            {
                "stage",
                "terminal_step",
                "time_limit",
                "prior_sha256",
                "architecture_sha256",
                "optimizer_sha256",
                "scientific_sha256",
            },
            "campaign stage input",
        )
        if raw["stage"] != expected_stage or raw["terminal_step"] != expected_step:
            raise ValueError("campaign stage input is not canonical")
        time_limit = _slurm_duration(raw["time_limit"], f"{expected_stage} time limit")
        digests = {
            name: _digest(raw[f"{name}_sha256"], f"{expected_stage} {name}")
            for name in ("prior", "architecture", "optimizer", "scientific")
        }
        normalized_stages.append(
            {
                **dict(raw),
                "static_protocol_sha256": _static_protocol(
                    source=source,
                    stage=expected_stage,
                    terminal_step=expected_step,
                    time_limit=time_limit,
                    prior_sha256=digests["prior"],
                    architecture_sha256=digests["architecture"],
                    optimizer_sha256=digests["optimizer"],
                    scientific_sha256=digests["scientific"],
                ),
            }
        )
    return make_manifest(
        "formal_campaign_draft",
        {
            "campaign_id": _safe_id(campaign_id, "campaign_id"),
            "training_source": source,
            "h100_gate": {
                "attestation_sha256": h100_sha,
                "checkpoint_ceiling_bytes": ceiling,
                "observed_checkpoint_max_bytes": observed_max,
                "nvidia_smi_sha256": nvidia_smi_sha256,
                "gpu_model": expected_gpu_model,
                "driver_version": driver_version,
            },
            "supported_seeds": list(FORMAL_SEEDS),
            "formal_arms": list(ARMS),
            "stages": normalized_stages,
        },
    )


def _formal_campaign_from_draft(
    draft: Mapping[str, Any], validation_evidence: Mapping[str, Any]
) -> dict[str, Any]:
    envelope = _exact(
        draft,
        {"schema_version", "kind", "payload", "sha256"},
        "campaign draft",
    )
    body = {key: envelope[key] for key in ("schema_version", "kind", "payload")}
    if (
        envelope["schema_version"] != 1
        or envelope["kind"] != "formal_campaign_draft"
        or envelope["sha256"] != canonical_sha256(body)
    ):
        raise ValueError("campaign draft envelope is invalid")
    payload = dict(envelope["payload"])
    if "validation_evidence" in payload:
        raise ValueError("campaign draft must not contain validation evidence")
    payload["validation_evidence"] = dict(validation_evidence)
    campaign = make_manifest("formal_campaign", payload)
    validate_campaign_manifest(campaign)
    return campaign


def _campaign_publication_layout(output: Path) -> None:
    output = _absolute_path(output, "campaign publication path")
    if output.name != "campaign.json":
        raise ValueError("formal campaign output must be campaign-root/campaign.json")
    root = output.parent
    root_fd = _open_directory_nofollow(root, where="campaign root")
    os.close(root_fd)
    names = _directory_names(root, where="campaign root")
    if names != {"acceptances", "evaluations"}:
        raise ValueError(
            "fresh campaign root must contain only acceptance/evaluation directories"
        )
    for name in ("acceptances", "evaluations"):
        child_fd = _open_directory_nofollow(root / name, where=f"campaign {name}")
        try:
            if os.listdir(child_fd):
                raise ValueError("fresh campaign directories must be empty")
        finally:
            os.close(child_fd)


def _revalidate_campaign_evidence(campaign: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_campaign_manifest(campaign)
    source = validated["training_source"]
    gate = validated["h100_gate"]
    evidence = validated["validation_evidence"]
    observed, attestation = _derive_campaign_validation_evidence(
        exact_root=Path(evidence["exact_root"]),
        source_manifest_path=Path(evidence["source_manifest_path"]),
        h100_attestation_path=Path(evidence["h100_attestation_path"]),
        h100_submission_receipt_path=Path(evidence["h100_submission_receipt_path"]),
        environment_completion_path=Path(evidence["environment_completion_path"]),
        git_path=Path(evidence["git_path"]),
        git_sha256=evidence["git_sha256"],
        h100_attestation_max_bytes=evidence["h100_attestation_max_bytes"],
        h100_submission_receipt_max_bytes=evidence["h100_submission_receipt_max_bytes"],
        source_manifest_max_bytes=evidence["source_manifest_max_bytes"],
        environment_completion_max_bytes=evidence["environment_completion_max_bytes"],
        environment_manifest_max_bytes=evidence["environment_manifest_max_bytes"],
        environment_inventory_max_bytes=evidence["environment_inventory_max_bytes"],
        source=source,
        expected_h100_sha256=gate["attestation_sha256"],
        checkpoint_ceiling_bytes=gate["checkpoint_ceiling_bytes"],
        expected_gpu_model=gate["gpu_model"],
    )
    if observed != evidence:
        raise ValueError("campaign validation evidence changed after publication")
    if attestation["payload"]["driver_version"] != gate["driver_version"]:
        raise ValueError("campaign H100 driver differs from external attestation")
    if attestation["payload"]["nvidia_smi_sha256"] != gate["nvidia_smi_sha256"]:
        raise ValueError("campaign nvidia-smi differs from external attestation")
    if (
        attestation["payload"]["observed_checkpoint_max_bytes"]
        != gate["observed_checkpoint_max_bytes"]
    ):
        raise ValueError(
            "campaign checkpoint observation differs from H100 attestation"
        )
    return validated


def publish_campaign(
    *,
    output_path: str | os.PathLike[str],
    campaign_id: str,
    training_source: Mapping[str, Any],
    exact_root: str | os.PathLike[str],
    source_manifest_path: str | os.PathLike[str],
    h100_attestation_path: str | os.PathLike[str],
    expected_h100_sha256: str,
    h100_submission_receipt_path: str | os.PathLike[str],
    environment_completion_path: str | os.PathLike[str],
    environment_completion_max_bytes: int,
    environment_manifest_max_bytes: int,
    environment_inventory_max_bytes: int,
    git_path: str | os.PathLike[str],
    git_sha256: str,
    checkpoint_ceiling_bytes: int,
    expected_gpu_model: str,
    stages: Sequence[Mapping[str, Any]],
    h100_attestation_max_bytes: int = H100_ATTESTATION_CEILING_BYTES,
    h100_submission_receipt_max_bytes: int = H100_SUBMISSION_RECEIPT_CEILING_BYTES,
    source_manifest_max_bytes: int = SOURCE_MANIFEST_CEILING_BYTES,
    campaign_max_bytes: int = CAMPAIGN_MANIFEST_CEILING_BYTES,
) -> dict[str, Any]:
    """Validate and publish the only authorizable formal campaign form."""

    output = _absolute_path(output_path, "campaign publication path")
    _campaign_publication_layout(output)
    source = _validated_training_source(training_source)
    h100_path = _absolute_path(h100_attestation_path, "H100 attestation path")
    h100 = read_canonical_manifest(
        h100_path,
        max_bytes=h100_attestation_max_bytes,
        expected_kind="h100_identity_smoke",
        expected_sha256=expected_h100_sha256,
    )
    draft = build_campaign_manifest(
        campaign_id=campaign_id,
        training_source=source,
        h100_attestation=h100,
        expected_h100_sha256=expected_h100_sha256,
        checkpoint_ceiling_bytes=checkpoint_ceiling_bytes,
        expected_gpu_model=expected_gpu_model,
        stages=stages,
    )
    evidence, validated_h100 = _derive_campaign_validation_evidence(
        exact_root=_absolute_path(exact_root, "campaign exact root"),
        source_manifest_path=_absolute_path(
            source_manifest_path, "campaign source manifest path"
        ),
        h100_attestation_path=h100_path,
        h100_submission_receipt_path=_absolute_path(
            h100_submission_receipt_path, "H100 submission receipt path"
        ),
        environment_completion_path=_absolute_path(
            environment_completion_path,
            "campaign environment completion marker",
        ),
        git_path=_absolute_path(git_path, "campaign Git executable"),
        git_sha256=git_sha256,
        h100_attestation_max_bytes=h100_attestation_max_bytes,
        h100_submission_receipt_max_bytes=h100_submission_receipt_max_bytes,
        source_manifest_max_bytes=source_manifest_max_bytes,
        environment_completion_max_bytes=environment_completion_max_bytes,
        environment_manifest_max_bytes=environment_manifest_max_bytes,
        environment_inventory_max_bytes=environment_inventory_max_bytes,
        source=source,
        expected_h100_sha256=expected_h100_sha256,
        checkpoint_ceiling_bytes=checkpoint_ceiling_bytes,
        expected_gpu_model=expected_gpu_model,
    )
    if validated_h100 != h100:
        raise ValueError("H100 attestation changed during campaign validation")
    campaign = _formal_campaign_from_draft(draft, evidence)
    publish_manifest_no_replace(output, campaign, max_bytes=campaign_max_bytes)
    published = read_canonical_manifest(
        output,
        max_bytes=campaign_max_bytes,
        expected_kind="formal_campaign",
        expected_sha256=campaign["sha256"],
    )
    _revalidate_campaign_evidence(published)
    if published != campaign:
        raise ValueError("published formal campaign changed during validation")
    return campaign


def campaign_binding(
    campaign: Mapping[str, Any],
    *,
    predecessor_acceptance_sha256_by_seed: Mapping[str, str],
) -> dict[str, Any]:
    validated = validate_campaign_manifest(campaign)
    expected_keys = {
        str(seed) for seed in FORMAL_SEEDS[: len(predecessor_acceptance_sha256_by_seed)]
    }
    if set(predecessor_acceptance_sha256_by_seed) != expected_keys:
        raise ValueError("campaign predecessor acceptance prefix is not canonical")
    predecessors = {
        key: _digest(value, f"seed {key} predecessor acceptance")
        for key, value in predecessor_acceptance_sha256_by_seed.items()
    }
    source = validated["training_source"]
    gate = validated["h100_gate"]
    stages = validated["stages"]
    binding = {
        "campaign_id": validated["campaign_id"],
        "campaign_manifest_sha256": validated["manifest_sha256"],
        "training_commit_sha": source["commit_sha"],
        "training_tree_sha": source["tree_sha"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "environment_sha256": source["environment_sha256"],
        "h100_attestation_sha256": gate["attestation_sha256"],
        "nvidia_smi_sha256": gate["nvidia_smi_sha256"],
        "checkpoint_ceiling_bytes": gate["checkpoint_ceiling_bytes"],
        "static_protocol_sha256_by_stage": {
            stage["stage"]: stage["static_protocol_sha256"] for stage in stages
        },
        "time_limit_by_stage": {
            stage["stage"]: stage["time_limit"] for stage in stages
        },
        "predecessor_acceptance_sha256_by_seed": predecessors,
    }
    if set(binding) != _CAMPAIGN_BINDING_KEYS:
        raise AssertionError("campaign binding schema drift")
    return binding


def validate_campaign_binding(
    value: Any,
    *,
    campaign: Mapping[str, Any],
    seed: int,
    predecessor_acceptance_sha256_by_seed: Mapping[str, str],
) -> dict[str, Any]:
    binding = _exact(value, _CAMPAIGN_BINDING_KEYS, "campaign binding")
    expected = campaign_binding(
        campaign,
        predecessor_acceptance_sha256_by_seed=predecessor_acceptance_sha256_by_seed,
    )
    selected_seed = _formal_seed(seed)
    if set(expected["predecessor_acceptance_sha256_by_seed"]) != {
        str(value) for value in FORMAL_SEEDS if value < selected_seed
    }:
        raise ValueError(
            "campaign binding predecessor set does not match selected seed"
        )
    if dict(binding) != expected:
        raise ValueError("campaign binding differs from immutable campaign state")
    return expected


def campaign_registry_capacity(
    *,
    fragment_size: int,
    campaign_manifest_ceiling_bytes: int = CAMPAIGN_MANIFEST_CEILING_BYTES,
    acceptance_ceiling_bytes: int = ACCEPTANCE_CEILING_BYTES,
    evaluation_receipt_ceiling_bytes: int = EVALUATION_RECEIPT_CEILING_BYTES,
) -> dict[str, int]:
    """Worst-case physical campaign-registry allocation.

    The canonical registry owns three directories, seven durable files (one
    campaign, three evaluation receipts, and three acceptance records), and
    reserves seven simultaneous atomic temporary names.  Referenced terminal
    attestations remain charged to their independent per-seed formal
    namespaces.
    """

    allocation = _positive_int(fragment_size, "filesystem fragment size")
    campaign_ceiling = _positive_int(
        campaign_manifest_ceiling_bytes, "campaign manifest ceiling"
    )
    acceptance_ceiling = _positive_int(acceptance_ceiling_bytes, "acceptance ceiling")
    evaluation_ceiling = _positive_int(
        evaluation_receipt_ceiling_bytes, "evaluation receipt ceiling"
    )

    def rounded(value: int) -> int:
        return ((value + allocation - 1) // allocation) * allocation

    file_bytes = (
        rounded(campaign_ceiling)
        + 3 * rounded(acceptance_ceiling)
        + 3 * rounded(evaluation_ceiling)
    )
    directory_objects = 3
    directory_entries = 1 + 2 + 7 + 7
    directory_bytes = (directory_objects + directory_entries) * allocation
    return {
        "fragment_size": allocation,
        "campaign_manifest_file_slots": 1,
        "acceptance_file_slots": 3,
        "evaluation_receipt_file_slots": 3,
        "durable_file_slots": 7,
        "directory_objects": directory_objects,
        "directory_entry_slots": directory_entries,
        "atomic_temporary_entry_slots": 7,
        "physical_file_bytes": file_bytes,
        "physical_directory_bytes": directory_bytes,
        "required_physical_bytes": file_bytes + directory_bytes,
    }


def _registry_names(seed: int) -> tuple[set[str], str]:
    selected = _formal_seed(seed)
    previous = {
        f"seed-{value}-accepted.json" for value in FORMAL_SEEDS if value < selected
    }
    return previous, f"seed-{selected}-accepted.json"


def _directory_names(path: Path, *, where: str) -> set[str]:
    fd = _open_directory_nofollow(path, where=where)
    try:
        names = set(os.listdir(fd))
        for name in names:
            if not isinstance(name, str) or name in {"", ".", ".."} or "/" in name:
                raise ValueError(f"{where} contains an invalid entry")
        return names
    finally:
        os.close(fd)


def _campaign_root_layout(
    campaign_path: Path, acceptance_registry: Path
) -> tuple[Path, Path]:
    root = campaign_path.parent
    if campaign_path != root / "campaign.json":
        raise ValueError("campaign manifest must be campaign-root/campaign.json")
    if acceptance_registry != root / "acceptances":
        raise ValueError("acceptance registry must be campaign-root/acceptances")
    evaluation_root = root / "evaluations"
    for path, where in (
        (root, "campaign root"),
        (acceptance_registry, "acceptance registry"),
        (evaluation_root, "campaign evaluation receipt directory"),
    ):
        fd = _open_directory_nofollow(path, where=where)
        os.close(fd)
    root_names = _directory_names(root, where="campaign root")
    if root_names != {"campaign.json", "acceptances", "evaluations"}:
        raise ValueError("campaign root has a missing or unknown entry")
    return root, evaluation_root


def _load_terminal_contract():
    path = (
        Path(__file__)
        .resolve(strict=True)
        .with_name("finalize_formal_scheduler_logs.py")
    )
    spec = importlib.util.spec_from_file_location(
        "_formal_campaign_terminal_contract", path
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load exact-T terminal finalizer contract")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    loaded = Path(module.__file__).resolve(strict=True)
    if loaded != path:
        raise ValueError("terminal finalizer contract was loaded outside exact T")
    return module


def _relative_artifact_path(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
        or "\\" in value
        or os.path.normpath(value) != value
    ):
        raise ValueError(f"{where} must be a normalized relative POSIX path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{where} must be a normalized relative POSIX path")
    return value


def _nonnegative_int(value: Any, where: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{where} exceeds its immutable ceiling")
    return value


def _validate_terminal_attestation(
    terminal: Mapping[str, Any],
    *,
    campaign: Mapping[str, Any],
    seed: int,
    expected_campaign_binding: Mapping[str, Any],
) -> dict[str, Any]:
    envelope = _exact(
        terminal,
        {"schema_version", "kind", "payload", "sha256"},
        "formal terminal attestation",
    )
    body = {key: envelope[key] for key in ("schema_version", "kind", "payload")}
    if (
        envelope["schema_version"] != 1
        or envelope["kind"] != "formal_terminal_scheduler_logs"
        or envelope["sha256"] != canonical_sha256(body)
    ):
        raise ValueError("formal terminal attestation envelope is invalid")
    contract = _load_terminal_contract()
    stage_names = tuple(stage for stage, _terminal_step in STAGES)
    expected_pairs = tuple((arm, stage) for stage in stage_names for arm in ARMS)
    if (
        tuple(contract.ARMS) != ARMS
        or tuple(contract.STAGES) != stage_names
        or tuple(contract.PAIR_ORDER) != expected_pairs
        or contract.TERMINAL_LOG_ATTESTATION_CEILING_BYTES
        != TERMINAL_ATTESTATION_CEILING_BYTES
    ):
        raise ValueError("exact-T terminal finalizer contract has drifted")

    campaign_value = validate_campaign_manifest(campaign)
    payload = envelope["payload"]
    required = {
        "study_id",
        "seed",
        "transaction_id",
        "submission_receipt_sha256",
        "transaction_ledger_sha256",
        "source_commit_sha",
        "source_tree_sha",
        "repository_binding",
        "campaign_binding",
        "runtime_tools",
        "manifest_ceiling_bytes",
        "protocol_metadata_allowance_bytes",
        "sacct_sha256",
        "path_format",
        "terminal_verified",
        "terminal_queries",
        "jobs",
    }
    payload = _exact(payload, required, "formal terminal payload")
    selected_seed = _formal_seed(seed)
    expected_study = f"{campaign_value['campaign_id']}-seed{selected_seed}"
    if not isinstance(payload["campaign_binding"], Mapping) or dict(
        payload["campaign_binding"]
    ) != dict(expected_campaign_binding):
        raise ValueError("formal terminal attestation campaign binding mismatch")
    validate_campaign_binding(
        payload["campaign_binding"],
        campaign=campaign,
        seed=selected_seed,
        predecessor_acceptance_sha256_by_seed=expected_campaign_binding[
            "predecessor_acceptance_sha256_by_seed"
        ],
    )
    if (
        payload["study_id"] != expected_study
        or payload["seed"] != selected_seed
        or not isinstance(payload["transaction_id"], str)
        or _TRANSACTION_ID.fullmatch(payload["transaction_id"]) is None
        or payload["source_commit_sha"]
        != campaign_value["training_source"]["commit_sha"]
        or payload["source_tree_sha"] != campaign_value["training_source"]["tree_sha"]
        or payload["terminal_verified"] is not True
        or payload["path_format"] != "artifact_root_relative_posix_v1"
    ):
        raise ValueError("formal terminal attestation campaign binding mismatch")
    receipt_sha256 = _digest(
        payload["submission_receipt_sha256"], "terminal submission receipt"
    )
    ledger_sha256 = _digest(
        payload["transaction_ledger_sha256"], "terminal transaction ledger"
    )
    _digest(payload["sacct_sha256"], "terminal sacct executable")
    manifest_ceiling = _positive_int(
        payload["manifest_ceiling_bytes"], "terminal manifest ceiling"
    )
    protocol_metadata_allowance = _positive_int(
        payload["protocol_metadata_allowance_bytes"],
        "terminal protocol metadata allowance",
    )
    if (
        manifest_ceiling > 1 << 40
        or protocol_metadata_allowance > FORMAL_METADATA_CEILING_BYTES
    ):
        raise ValueError("terminal evidence ceiling is unbounded")
    repository_binding = contract._validated_repository_binding(
        payload["repository_binding"],
        expected_commit_sha=campaign_value["training_source"]["commit_sha"],
    )
    runtime_tools = _exact(
        payload["runtime_tools"], {"nvidia_smi_sha256"}, "terminal runtime tools"
    )
    if (
        _digest(runtime_tools["nvidia_smi_sha256"], "terminal nvidia-smi executable")
        != campaign_value["h100_gate"]["nvidia_smi_sha256"]
    ):
        raise ValueError("terminal runtime tools differ from campaign H100 gate")

    jobs = payload["jobs"]
    if not isinstance(jobs, list) or len(jobs) != len(expected_pairs):
        raise ValueError("formal terminal attestation must contain exactly nine jobs")
    observed_job_ids: set[str] = set()
    normalized_jobs: dict[tuple[str, str], Mapping[str, Any]] = {}
    h100_gate = campaign_value["h100_gate"]
    expected_seed_sha256 = contract._expected_seed_sha256(selected_seed)
    time_limits = expected_campaign_binding["time_limit_by_stage"]
    transaction_id = payload["transaction_id"]
    for raw_job, expected_pair in zip(jobs, expected_pairs):
        job = _exact(raw_job, _TERMINAL_JOB_KEYS, "formal terminal job")
        pair = (job["arm"], job["stage"])
        if pair != expected_pair or pair in normalized_jobs:
            raise ValueError("formal terminal job matrix/order is not canonical")
        arm, stage = expected_pair
        stage_index = stage_names.index(stage) + 1
        job_id = job["job_id"]
        cluster = job["cluster"]
        if (
            not isinstance(job_id, str)
            or _JOB_ID.fullmatch(job_id) is None
            or job_id in observed_job_ids
            or cluster is not None
            and (
                not isinstance(cluster, str) or _SAFE_CLUSTER.fullmatch(cluster) is None
            )
            or job["job_name"] != f"tabicl-{transaction_id}-{arm}-s{stage_index}"
        ):
            raise ValueError("formal terminal job scheduler identity is invalid")
        observed_job_ids.add(job_id)
        if (
            job["state"] != "COMPLETED"
            or job["exit_code"] != "0:0"
            or job["derived_exit_code"] != "0:0"
        ):
            raise ValueError("formal terminal job is not independently successful")

        completion = _exact(
            job["completion"],
            _TERMINAL_COMPLETION_KEYS,
            "formal terminal job completion",
        )
        expected_slug = f"{arm}-{stage}"
        if completion["path"] != f"runtime-completions/{expected_slug}.json":
            raise ValueError("formal terminal completion path is not canonical")
        _relative_artifact_path(completion["path"], "terminal completion path")
        _digest(completion["manifest_sha256"], "terminal completion manifest")
        _digest(completion["file_sha256"], "terminal completion file")
        stdout_size = _nonnegative_int(
            completion["stdout_observed_size"], "terminal completion stdout size"
        )
        stderr_size = _nonnegative_int(
            completion["stderr_observed_size"], "terminal completion stderr size"
        )
        scheduler = _exact(
            completion["scheduler"],
            set(contract.COMPLETION_SCHEDULER_KEYS),
            "formal terminal completion scheduler",
        )
        if (
            scheduler["partition"] != "h100"
            or scheduler["qos"] != "long"
            or scheduler["cpus_per_task"] != 64
            or isinstance(scheduler["cpus_per_task"], bool)
            or scheduler["memory_mb"] != 131_072
            or isinstance(scheduler["memory_mb"], bool)
            or scheduler["gpus_per_job"] != 1
            or isinstance(scheduler["gpus_per_job"], bool)
            or scheduler["time_limit"] != time_limits[stage]
        ):
            raise ValueError("formal terminal completion scheduler binding mismatch")
        _digest(scheduler["query_sha256"], "terminal completion scheduler query")
        if (
            completion["seed"] != selected_seed
            or completion["repository_binding"] != repository_binding
            or completion["gpu_name"] != h100_gate["gpu_model"]
            or completion["gpu_driver_version"] != h100_gate["driver_version"]
            or not isinstance(completion["cuda_visible_devices"], str)
            or _VISIBLE_DEVICE.fullmatch(completion["cuda_visible_devices"]) is None
            or not isinstance(completion["gpu_uuid"], str)
            or _GPU_UUID.fullmatch(completion["gpu_uuid"]) is None
            or completion["manifest_ceiling_bytes"] != manifest_ceiling
            or completion["protocol_metadata_allowance_bytes"]
            != protocol_metadata_allowance
        ):
            raise ValueError("formal terminal job completion binding mismatch")

        scheduler_logs = job["scheduler_logs"]
        if not isinstance(scheduler_logs, list) or len(scheduler_logs) != 2:
            raise ValueError(
                "formal terminal scheduler logs must contain stdout/stderr"
            )
        for log, stream, suffix, minimum_size in zip(
            scheduler_logs,
            ("stdout", "stderr"),
            ("out", "err"),
            (stdout_size, stderr_size),
        ):
            log = _exact(log, _TERMINAL_LOG_KEYS, "formal terminal scheduler log")
            expected_path = f"scheduler-logs/{expected_slug}.{suffix}"
            if (
                log["stream"] != stream
                or log["path"] != expected_path
                or _nonnegative_int(log["size"], "terminal scheduler log size")
                < minimum_size
                or log["fatal_signature_scan_passed"] is not True
                or log["stable_reads"] != 2
                or isinstance(log["stable_reads"], bool)
            ):
                raise ValueError("formal terminal scheduler log binding mismatch")
            _relative_artifact_path(log["path"], "terminal scheduler log path")
            _digest(log["sha256"], "terminal scheduler log")

        finalized = _exact(
            job["finalized_artifact"],
            _TERMINAL_FINALIZED_KEYS,
            "formal terminal finalized artifact",
        )
        binding = _exact(
            finalized["binding"],
            set(contract.FINALIZED_ARTIFACT_KEYS),
            "formal terminal finalized artifact binding",
        )
        terminal_step = dict(STAGES)[stage]
        if (
            binding["finalized_manifest_path"]
            != f"arms/{arm}/{stage}/finalized-checkpoint.json"
            or binding["checkpoint_path"]
            != f"arms/{arm}/{stage}/step-{terminal_step}.ckpt"
        ):
            raise ValueError("formal terminal finalized artifact path is not canonical")
        _relative_artifact_path(
            binding["finalized_manifest_path"], "terminal finalized manifest path"
        )
        _relative_artifact_path(binding["checkpoint_path"], "terminal checkpoint path")
        for field in (
            "finalized_manifest_sha256",
            "checkpoint_sha256",
            "provenance_sha256",
            "seed_sha256",
            "treatment_sha256",
        ):
            _digest(binding[field], f"terminal finalized artifact {field}")
        checkpoint_size = _nonnegative_int(
            binding["checkpoint_size"],
            "terminal finalized checkpoint size",
            maximum=h100_gate["checkpoint_ceiling_bytes"],
        )
        if (
            checkpoint_size <= 0
            or binding["seed_sha256"] != expected_seed_sha256
            or binding["treatment_sha256"]
            != contract._expected_treatment_sha256(arm, selected_seed)
        ):
            raise ValueError("formal terminal finalized artifact binding mismatch")
        _digest(
            finalized["finalized_manifest_file_sha256"],
            "terminal finalized manifest file",
        )
        finalized_size = _positive_int(
            finalized["finalized_manifest_size"], "terminal finalized manifest size"
        )
        initial_sha256 = _digest(
            finalized["initial_validation_sha256"],
            "terminal initial artifact validation",
        )
        publication_sha256 = _digest(
            finalized["publication_validation_sha256"],
            "terminal publication artifact validation",
        )
        if (
            finalized_size > manifest_ceiling
            or finalized["independent_stable_validations"] != 2
            or isinstance(finalized["independent_stable_validations"], bool)
            or initial_sha256 != publication_sha256
        ):
            raise ValueError("formal terminal finalized artifact validation mismatch")

        normalized_jobs[pair] = job

    for arm, stage in expected_pairs:
        job = normalized_jobs[(arm, stage)]
        stage_index = stage_names.index(stage)
        if stage_index == 0:
            expected_parent = None
        else:
            parent_stage = stage_names[stage_index - 1]
            parent_job = normalized_jobs[(arm, parent_stage)]
            expected_parent = {
                **dict(parent_job["finalized_artifact"]["binding"]),
                "job_id": parent_job["job_id"],
                "stage": parent_stage,
            }
        if job["parent_lineage"] != expected_parent:
            raise ValueError("formal terminal parent lineage is not canonical")

    queries = payload["terminal_queries"]
    if not isinstance(queries, list):
        raise ValueError("formal terminal scheduler queries must be a list")
    jobs_by_cluster: dict[str | None, list[str]] = {}
    for job in jobs:
        jobs_by_cluster.setdefault(job["cluster"], []).append(job["job_id"])
    if len(queries) != len(jobs_by_cluster):
        raise ValueError("formal terminal scheduler query set is incomplete")
    for query, (cluster, job_ids) in zip(queries, jobs_by_cluster.items()):
        query = _exact(query, _TERMINAL_QUERY_KEYS, "formal terminal scheduler query")
        argv: list[str] = []
        if cluster is not None:
            argv.append(f"--clusters={cluster}")
        argv.extend(
            [
                "--noheader",
                "--allocations",
                f"--jobs={','.join(job_ids)}",
                "--format=JobIDRaw,JobName,State,ExitCode,DerivedExitCode",
                "--parsable2",
            ]
        )
        if (
            query["cluster"] != cluster
            or query["job_ids"] != job_ids
            or query["query_argv_sha256"] != canonical_sha256(argv)
        ):
            raise ValueError("formal terminal scheduler query binding mismatch")
        _digest(query["stdout_sha256"], "terminal scheduler query stdout")
        _digest(query["stderr_sha256"], "terminal scheduler query stderr")

    if not receipt_sha256 or not ledger_sha256:
        raise AssertionError("validated terminal digests unexpectedly empty")
    return dict(payload)


def validate_evaluation_sanity_receipt(
    receipt: Mapping[str, Any],
    *,
    campaign: Mapping[str, Any],
    seed: int,
    terminal_attestation_sha256: str,
    expected_campaign_binding: Mapping[str, Any],
) -> dict[str, Any]:
    envelope = _exact(
        receipt,
        {"schema_version", "kind", "payload", "sha256"},
        "evaluation sanity receipt",
    )
    body = {key: envelope[key] for key in ("schema_version", "kind", "payload")}
    if (
        envelope["schema_version"] != 1
        or envelope["kind"] != "minimum_evaluation_sanity"
        or envelope["sha256"] != canonical_sha256(body)
    ):
        raise ValueError("evaluation sanity receipt envelope is invalid")
    payload = _exact(
        envelope["payload"],
        {
            "campaign_id",
            "campaign_manifest_sha256",
            "campaign_binding",
            "seed",
            "study_id",
            "formal_terminal_attestation_sha256",
            "formal_submission_receipt_sha256",
            "training_source",
            "evaluation_source",
            "evaluation_protocol_sha256",
            "matched_dataset_sha256",
            "results_by_arm",
            "acceptance",
        },
        "evaluation sanity payload",
    )
    validated = validate_campaign_manifest(campaign)
    selected_seed = _formal_seed(seed)
    expected_study = f"{validated['campaign_id']}-seed{selected_seed}"
    if (
        payload["campaign_id"] != validated["campaign_id"]
        or payload["campaign_manifest_sha256"] != validated["manifest_sha256"]
        or payload["campaign_binding"] != expected_campaign_binding
        or payload["seed"] != selected_seed
        or payload["study_id"] != expected_study
        or payload["formal_terminal_attestation_sha256"]
        != _digest(terminal_attestation_sha256, "terminal attestation")
        or payload["training_source"] != validated["training_source"]
    ):
        raise ValueError("evaluation receipt training/campaign binding mismatch")
    _digest(payload["formal_submission_receipt_sha256"], "evaluation formal receipt")
    _digest(payload["evaluation_protocol_sha256"], "evaluation protocol")
    _digest(payload["matched_dataset_sha256"], "matched evaluation dataset")
    evaluation_source = _exact(
        payload["evaluation_source"],
        {
            "candidate_repository",
            "commit_sha",
            "tree_sha",
            "source_manifest_sha256",
            "descendant_of_training_commit_sha",
            "ancestry_verified",
            "ancestry_query_sha256",
        },
        "evaluation source",
    )
    if (
        evaluation_source["candidate_repository"] != CANONICAL_REPOSITORY
        or _git_oid(evaluation_source["commit_sha"], "evaluation commit")
        == validated["training_source"]["commit_sha"]
        or _git_oid(evaluation_source["tree_sha"], "evaluation tree")
        == validated["training_source"]["tree_sha"]
        or evaluation_source["descendant_of_training_commit_sha"]
        != validated["training_source"]["commit_sha"]
        or evaluation_source["ancestry_verified"] is not True
    ):
        raise ValueError("evaluation source is not a distinct attested descendant E")
    _digest(evaluation_source["source_manifest_sha256"], "evaluation source manifest")
    _digest(evaluation_source["ancestry_query_sha256"], "evaluation ancestry query")
    results = _exact(payload["results_by_arm"], set(ARMS), "evaluation arm results")
    reference_shape: tuple[int, int, tuple[str, ...]] | None = None
    for arm in ARMS:
        result = _exact(
            results[arm],
            {
                "evaluated_datasets",
                "evaluated_examples",
                "prediction_manifest_sha256",
                "metrics",
            },
            f"evaluation result {arm}",
        )
        datasets = _positive_int(
            result["evaluated_datasets"], f"{arm} evaluated datasets"
        )
        examples = _positive_int(
            result["evaluated_examples"], f"{arm} evaluated examples"
        )
        _digest(result["prediction_manifest_sha256"], f"{arm} prediction manifest")
        metrics = result["metrics"]
        if not isinstance(metrics, Mapping) or not metrics:
            raise ValueError(f"{arm} metrics must be a non-empty object")
        metric_names = tuple(sorted(metrics))
        for name in metric_names:
            value = metrics[name]
            if _METRIC_NAME.fullmatch(name) is None:
                raise ValueError(f"{arm} metric name is invalid")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{arm} metric {name} is non-finite or non-numeric")
        shape = (datasets, examples, metric_names)
        if reference_shape is None:
            reference_shape = shape
        elif shape != reference_shape:
            raise ValueError("minimum evaluation is not matched across all three arms")
    if payload["acceptance"] != {
        "all_three_arms_present": True,
        "all_values_finite": True,
        "minimum_sanity_passed": True,
    }:
        raise ValueError("minimum evaluation acceptance flags are not all true")
    return dict(payload)


def _validated_terminal_raw_evidence(
    value: Any,
    *,
    terminal_path: Path,
) -> dict[str, Any]:
    evidence = _exact(
        value,
        _TERMINAL_RAW_EVIDENCE_KEYS,
        "formal terminal raw evidence",
    )
    artifact_root = _absolute_path(evidence["artifact_root"], "formal artifact root")
    receipt = _absolute_path(
        evidence["submission_receipt_path"], "formal submission receipt path"
    )
    ledger = _absolute_path(
        evidence["transaction_ledger_path"], "formal transaction ledger path"
    )
    allowance = _positive_int(
        evidence["protocol_metadata_allowance_bytes"],
        "formal protocol metadata allowance",
    )
    if allowance > FORMAL_METADATA_CEILING_BYTES:
        raise ValueError("formal protocol metadata allowance exceeds its fixed maximum")
    if (
        terminal_path != artifact_root / "terminal-scheduler-logs.json"
        or receipt != artifact_root / "submission-receipt.json"
        or ledger != artifact_root / "transaction-ledger.json"
    ):
        raise ValueError(
            "formal terminal evidence paths are not one canonical namespace"
        )
    return {
        "artifact_root": str(artifact_root),
        "submission_receipt_path": str(receipt),
        "transaction_ledger_path": str(ledger),
        "protocol_metadata_allowance_bytes": allowance,
    }


def _validate_existing_terminal_evidence(
    *,
    terminal_path: Path,
    terminal: Mapping[str, Any],
    raw_evidence: Mapping[str, Any],
) -> Mapping[str, Any]:
    evidence = _validated_terminal_raw_evidence(
        raw_evidence, terminal_path=terminal_path
    )
    contract = _load_terminal_contract()
    rebuilt = contract.validate_existing_terminal(
        terminal_attestation=terminal_path,
        submission_receipt=Path(evidence["submission_receipt_path"]),
        transaction_ledger=Path(evidence["transaction_ledger_path"]),
        artifact_root=Path(evidence["artifact_root"]),
        max_metadata_bytes=evidence["protocol_metadata_allowance_bytes"],
        expected_terminal_sha256=terminal["sha256"],
    )
    if rebuilt != terminal:
        raise ValueError("terminal attestation differs from raw formal evidence")
    return rebuilt


def _acceptance_payload(
    *,
    campaign: Mapping[str, Any],
    seed: int,
    predecessor_sha256: str | None,
    predecessor_map: Mapping[str, str],
    terminal_path: Path,
    terminal: Mapping[str, Any],
    terminal_raw_evidence: Mapping[str, Any],
    evaluation_path: Path,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_campaign_manifest(campaign)
    binding = campaign_binding(
        campaign, predecessor_acceptance_sha256_by_seed=predecessor_map
    )
    normalized_raw_evidence = _validated_terminal_raw_evidence(
        terminal_raw_evidence, terminal_path=terminal_path
    )
    _validate_existing_terminal_evidence(
        terminal_path=terminal_path,
        terminal=terminal,
        raw_evidence=normalized_raw_evidence,
    )
    if (
        terminal["payload"].get("protocol_metadata_allowance_bytes")
        != normalized_raw_evidence["protocol_metadata_allowance_bytes"]
    ):
        raise ValueError(
            "terminal protocol metadata allowance differs from acceptance input"
        )
    terminal_payload = _validate_terminal_attestation(
        terminal,
        campaign=campaign,
        seed=seed,
        expected_campaign_binding=binding,
    )
    evaluation_payload = validate_evaluation_sanity_receipt(
        evaluation,
        campaign=campaign,
        seed=seed,
        terminal_attestation_sha256=terminal["sha256"],
        expected_campaign_binding=binding,
    )
    if (
        evaluation_payload["formal_submission_receipt_sha256"]
        != terminal_payload["submission_receipt_sha256"]
    ):
        raise ValueError("evaluation and terminal evidence bind different submissions")
    return {
        "campaign_id": validated["campaign_id"],
        "campaign_manifest_sha256": validated["manifest_sha256"],
        "campaign_binding": binding,
        "seed": _formal_seed(seed),
        "study_id": f"{validated['campaign_id']}-seed{seed}",
        "predecessor_acceptance_sha256": predecessor_sha256,
        "formal_terminal_attestation": {
            "path": str(terminal_path),
            "sha256": terminal["sha256"],
        },
        "formal_terminal_raw_evidence": normalized_raw_evidence,
        "minimum_evaluation_sanity": {
            "path": str(evaluation_path),
            "sha256": evaluation["sha256"],
        },
        "formal_submission_receipt_sha256": terminal_payload[
            "submission_receipt_sha256"
        ],
        "accepted": True,
    }


def _load_acceptance(
    path: Path,
    *,
    campaign: Mapping[str, Any],
    seed: int,
    predecessor_sha256: str | None,
    predecessor_map: Mapping[str, str],
    acceptance_max_bytes: int,
    terminal_max_bytes: int,
    evaluation_max_bytes: int,
) -> dict[str, Any]:
    acceptance = read_canonical_manifest(
        path,
        max_bytes=acceptance_max_bytes,
        expected_kind="formal_seed_acceptance",
    )
    payload = _exact(
        acceptance["payload"],
        {
            "campaign_id",
            "campaign_manifest_sha256",
            "campaign_binding",
            "seed",
            "study_id",
            "predecessor_acceptance_sha256",
            "formal_terminal_attestation",
            "formal_terminal_raw_evidence",
            "minimum_evaluation_sanity",
            "formal_submission_receipt_sha256",
            "accepted",
        },
        "formal seed acceptance payload",
    )
    terminal_binding = _exact(
        payload["formal_terminal_attestation"],
        {"path", "sha256"},
        "acceptance terminal binding",
    )
    evaluation_binding = _exact(
        payload["minimum_evaluation_sanity"],
        {"path", "sha256"},
        "acceptance evaluation binding",
    )
    terminal_path = _absolute_path(
        terminal_binding["path"], "terminal attestation path"
    )
    evaluation_path = _absolute_path(
        evaluation_binding["path"], "evaluation sanity receipt path"
    )
    terminal = read_canonical_manifest(
        terminal_path,
        max_bytes=terminal_max_bytes,
        expected_kind="formal_terminal_scheduler_logs",
        expected_sha256=_digest(terminal_binding["sha256"], "accepted terminal"),
    )
    evaluation = read_canonical_manifest(
        evaluation_path,
        max_bytes=evaluation_max_bytes,
        expected_kind="minimum_evaluation_sanity",
        expected_sha256=_digest(evaluation_binding["sha256"], "accepted evaluation"),
    )
    expected_payload = _acceptance_payload(
        campaign=campaign,
        seed=seed,
        predecessor_sha256=predecessor_sha256,
        predecessor_map=predecessor_map,
        terminal_path=terminal_path,
        terminal=terminal,
        terminal_raw_evidence=payload["formal_terminal_raw_evidence"],
        evaluation_path=evaluation_path,
        evaluation=evaluation,
    )
    if dict(payload) != expected_payload or payload["accepted"] is not True:
        raise ValueError("formal seed acceptance differs from canonical evidence")
    return acceptance


def authorize_seed_submission(
    *,
    campaign_path: str | os.PathLike[str],
    campaign_expected_sha256: str,
    acceptance_registry: str | os.PathLike[str],
    seed: int,
    campaign_max_bytes: int = CAMPAIGN_MANIFEST_CEILING_BYTES,
    acceptance_max_bytes: int = ACCEPTANCE_CEILING_BYTES,
    terminal_max_bytes: int = TERMINAL_ATTESTATION_CEILING_BYTES,
    evaluation_max_bytes: int = EVALUATION_RECEIPT_CEILING_BYTES,
) -> dict[str, Any]:
    """Validate the immutable campaign and exact predecessor prefix for a seed."""

    selected_seed = _formal_seed(seed)
    campaign_file = _absolute_path(campaign_path, "campaign manifest path")
    registry = _absolute_path(acceptance_registry, "acceptance registry path")
    _campaign_root_layout(campaign_file, registry)
    campaign = read_canonical_manifest(
        campaign_file,
        max_bytes=campaign_max_bytes,
        expected_kind="formal_campaign",
        expected_sha256=campaign_expected_sha256,
    )
    _revalidate_campaign_evidence(campaign)
    expected_names, _current_name = _registry_names(selected_seed)
    actual_names = _directory_names(registry, where="acceptance registry")
    if actual_names != expected_names:
        raise ValueError(
            "acceptance registry is not the exact predecessor prefix for this seed"
        )
    predecessor_map: dict[str, str] = {}
    previous_sha: str | None = None
    for previous_seed in (value for value in FORMAL_SEEDS if value < selected_seed):
        path = registry / f"seed-{previous_seed}-accepted.json"
        acceptance = _load_acceptance(
            path,
            campaign=campaign,
            seed=previous_seed,
            predecessor_sha256=previous_sha,
            predecessor_map=predecessor_map,
            acceptance_max_bytes=acceptance_max_bytes,
            terminal_max_bytes=terminal_max_bytes,
            evaluation_max_bytes=evaluation_max_bytes,
        )
        previous_sha = acceptance["sha256"]
        predecessor_map[str(previous_seed)] = previous_sha
    return {
        "campaign": campaign,
        "campaign_path": str(campaign_file),
        "acceptance_registry": str(registry),
        "seed": selected_seed,
        "campaign_binding": campaign_binding(
            campaign,
            predecessor_acceptance_sha256_by_seed=predecessor_map,
        ),
    }


def publish_seed_acceptance(
    *,
    campaign_path: str | os.PathLike[str],
    campaign_expected_sha256: str,
    acceptance_registry: str | os.PathLike[str],
    seed: int,
    formal_artifact_root: str | os.PathLike[str],
    terminal_attestation_path: str | os.PathLike[str],
    submission_receipt_path: str | os.PathLike[str],
    transaction_ledger_path: str | os.PathLike[str],
    terminal_metadata_max_bytes: int,
    evaluation_receipt_path: str | os.PathLike[str],
    campaign_max_bytes: int = CAMPAIGN_MANIFEST_CEILING_BYTES,
    acceptance_max_bytes: int = ACCEPTANCE_CEILING_BYTES,
    terminal_max_bytes: int = TERMINAL_ATTESTATION_CEILING_BYTES,
    evaluation_max_bytes: int = EVALUATION_RECEIPT_CEILING_BYTES,
) -> dict[str, Any]:
    selected_seed = _formal_seed(seed)
    campaign_file = _absolute_path(campaign_path, "campaign manifest path")
    registry = _absolute_path(acceptance_registry, "acceptance registry path")
    _root, evaluation_root = _campaign_root_layout(campaign_file, registry)
    campaign = read_canonical_manifest(
        campaign_file,
        max_bytes=campaign_max_bytes,
        expected_kind="formal_campaign",
        expected_sha256=campaign_expected_sha256,
    )
    _revalidate_campaign_evidence(campaign)
    expected_previous, current_name = _registry_names(selected_seed)
    actual_names = _directory_names(registry, where="acceptance registry")
    if actual_names != expected_previous:
        raise ValueError("cannot accept a seed out of order or overwrite an acceptance")
    predecessor_map: dict[str, str] = {}
    previous_sha: str | None = None
    for previous_seed in (value for value in FORMAL_SEEDS if value < selected_seed):
        previous = _load_acceptance(
            registry / f"seed-{previous_seed}-accepted.json",
            campaign=campaign,
            seed=previous_seed,
            predecessor_sha256=previous_sha,
            predecessor_map=predecessor_map,
            acceptance_max_bytes=acceptance_max_bytes,
            terminal_max_bytes=terminal_max_bytes,
            evaluation_max_bytes=evaluation_max_bytes,
        )
        previous_sha = previous["sha256"]
        predecessor_map[str(previous_seed)] = previous_sha
    terminal_path = _absolute_path(
        terminal_attestation_path, "formal terminal attestation path"
    )
    terminal_raw_evidence = _validated_terminal_raw_evidence(
        {
            "artifact_root": str(
                _absolute_path(formal_artifact_root, "formal artifact root")
            ),
            "submission_receipt_path": str(
                _absolute_path(
                    submission_receipt_path, "formal submission receipt path"
                )
            ),
            "transaction_ledger_path": str(
                _absolute_path(
                    transaction_ledger_path, "formal transaction ledger path"
                )
            ),
            "protocol_metadata_allowance_bytes": terminal_metadata_max_bytes,
        },
        terminal_path=terminal_path,
    )
    evaluation_path = _absolute_path(
        evaluation_receipt_path, "minimum evaluation receipt path"
    )
    if evaluation_path != evaluation_root / f"seed-{selected_seed}.json":
        raise ValueError(
            "evaluation receipt path is not canonical for the selected seed"
        )
    terminal = read_canonical_manifest(
        terminal_path,
        max_bytes=terminal_max_bytes,
        expected_kind="formal_terminal_scheduler_logs",
    )
    evaluation = read_canonical_manifest(
        evaluation_path,
        max_bytes=evaluation_max_bytes,
        expected_kind="minimum_evaluation_sanity",
    )
    payload = _acceptance_payload(
        campaign=campaign,
        seed=selected_seed,
        predecessor_sha256=previous_sha,
        predecessor_map=predecessor_map,
        terminal_path=terminal_path,
        terminal=terminal,
        terminal_raw_evidence=terminal_raw_evidence,
        evaluation_path=evaluation_path,
        evaluation=evaluation,
    )
    acceptance = make_manifest("formal_seed_acceptance", payload)
    output = registry / current_name
    publish_manifest_no_replace(output, acceptance, max_bytes=acceptance_max_bytes)
    accepted = _load_acceptance(
        output,
        campaign=campaign,
        seed=selected_seed,
        predecessor_sha256=previous_sha,
        predecessor_map=predecessor_map,
        acceptance_max_bytes=acceptance_max_bytes,
        terminal_max_bytes=terminal_max_bytes,
        evaluation_max_bytes=evaluation_max_bytes,
    )
    if accepted != acceptance:
        raise ValueError("published seed acceptance changed after publication")
    return acceptance


def _read_campaign_spec(path: Path, *, max_bytes: int) -> Mapping[str, Any]:
    """Read a self-hashed ``formal_campaign_spec`` and return its payload."""

    wrapper = read_canonical_manifest(
        path,
        max_bytes=max_bytes,
        expected_kind="formal_campaign_spec",
    )
    return _exact(
        wrapper["payload"],
        {
            "campaign_id",
            "training_source",
            "exact_root",
            "source_manifest_path",
            "source_manifest_max_bytes",
            "h100_attestation_path",
            "h100_attestation_sha256",
            "h100_attestation_max_bytes",
            "h100_submission_receipt_path",
            "h100_submission_receipt_max_bytes",
            "environment_completion_path",
            "environment_completion_max_bytes",
            "environment_manifest_max_bytes",
            "environment_inventory_max_bytes",
            "git_path",
            "git_sha256",
            "checkpoint_ceiling_bytes",
            "expected_gpu_model",
            "stages",
        },
        "formal campaign spec",
    )


def _main(argv: Sequence[str] | None = None) -> int:
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise ValueError("formal campaign registry requires Python -I -B")
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser("publish-campaign")
    publish.add_argument("--spec", required=True, type=Path)
    publish.add_argument("--spec-max-bytes", required=True, type=int)
    publish.add_argument("--output", required=True, type=Path)
    publish.add_argument(
        "--campaign-max-bytes",
        type=int,
        default=CAMPAIGN_MANIFEST_CEILING_BYTES,
    )

    validate = subparsers.add_parser("authorize-seed")
    validate.add_argument("--campaign", required=True, type=Path)
    validate.add_argument("--campaign-sha256", required=True)
    validate.add_argument("--acceptance-registry", required=True, type=Path)
    validate.add_argument("--seed", required=True, type=int)

    accept = subparsers.add_parser("accept-seed")
    accept.add_argument("--campaign", required=True, type=Path)
    accept.add_argument("--campaign-sha256", required=True)
    accept.add_argument("--acceptance-registry", required=True, type=Path)
    accept.add_argument("--seed", required=True, type=int)
    accept.add_argument("--formal-artifact-root", required=True, type=Path)
    accept.add_argument("--terminal-attestation", required=True, type=Path)
    accept.add_argument("--submission-receipt", required=True, type=Path)
    accept.add_argument("--transaction-ledger", required=True, type=Path)
    accept.add_argument("--terminal-metadata-max-bytes", required=True, type=int)
    accept.add_argument("--evaluation-receipt", required=True, type=Path)

    capacity = subparsers.add_parser("capacity")
    capacity.add_argument("--fragment-size", required=True, type=int)

    args = parser.parse_args(argv)
    if args.command == "publish-campaign":
        spec = _read_campaign_spec(args.spec, max_bytes=args.spec_max_bytes)
        result = publish_campaign(
            output_path=args.output,
            campaign_id=spec["campaign_id"],
            training_source=spec["training_source"],
            exact_root=spec["exact_root"],
            source_manifest_path=spec["source_manifest_path"],
            h100_attestation_path=spec["h100_attestation_path"],
            expected_h100_sha256=spec["h100_attestation_sha256"],
            h100_submission_receipt_path=spec["h100_submission_receipt_path"],
            environment_completion_path=spec["environment_completion_path"],
            environment_completion_max_bytes=spec["environment_completion_max_bytes"],
            environment_manifest_max_bytes=spec["environment_manifest_max_bytes"],
            environment_inventory_max_bytes=spec["environment_inventory_max_bytes"],
            git_path=spec["git_path"],
            git_sha256=spec["git_sha256"],
            checkpoint_ceiling_bytes=spec["checkpoint_ceiling_bytes"],
            expected_gpu_model=spec["expected_gpu_model"],
            stages=spec["stages"],
            h100_attestation_max_bytes=spec["h100_attestation_max_bytes"],
            h100_submission_receipt_max_bytes=spec["h100_submission_receipt_max_bytes"],
            source_manifest_max_bytes=spec["source_manifest_max_bytes"],
            campaign_max_bytes=args.campaign_max_bytes,
        )
        print(canonical_json_bytes(result).decode("utf-8"))
    elif args.command == "authorize-seed":
        result = authorize_seed_submission(
            campaign_path=args.campaign,
            campaign_expected_sha256=args.campaign_sha256,
            acceptance_registry=args.acceptance_registry,
            seed=args.seed,
        )
        print(canonical_json_bytes(result["campaign_binding"]).decode("utf-8"))
    elif args.command == "accept-seed":
        result = publish_seed_acceptance(
            campaign_path=args.campaign,
            campaign_expected_sha256=args.campaign_sha256,
            acceptance_registry=args.acceptance_registry,
            seed=args.seed,
            formal_artifact_root=args.formal_artifact_root,
            terminal_attestation_path=args.terminal_attestation,
            submission_receipt_path=args.submission_receipt,
            transaction_ledger_path=args.transaction_ledger,
            terminal_metadata_max_bytes=args.terminal_metadata_max_bytes,
            evaluation_receipt_path=args.evaluation_receipt,
        )
        print(canonical_json_bytes(result).decode("utf-8"))
    else:
        print(
            canonical_json_bytes(
                campaign_registry_capacity(fragment_size=args.fragment_size)
            ).decode("utf-8")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
