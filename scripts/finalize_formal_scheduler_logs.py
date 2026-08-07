#!/usr/bin/env python3
"""Publish write-once terminal evidence for formal Slurm spool logs.

This controller must run outside the batch allocation.  An in-job completion
record cannot prove that Slurm has closed its stdout/stderr handles, so this
module first requires an exact successful ``sacct`` allocation row for every
job and only then scans and hashes each physical spool file twice.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


ARMS = ("rope", "temporary", "none")
STAGES = ("stage1", "stage2", "stage3")
TERMINAL_STEPS = {"stage1": 500_000, "stage2": 40_000, "stage3": 10_000}
PAIR_ORDER = tuple((arm, stage) for stage in STAGES for arm in ARMS)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
JOB_ID = re.compile(r"^[1-9][0-9]{0,19}$")
SAFE_CLUSTER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
VISIBLE_DEVICE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
GPU_UUID = re.compile(r"^(?:GPU|MIG)-[A-Za-z0-9_.:-]{1,192}$")
SLURM_DURATION = re.compile(
    r"^(?:(?P<days>[1-9][0-9]{0,3})-)?"
    r"(?P<hours>[0-9]{2}):(?P<minutes>[0-5][0-9]):(?P<seconds>[0-5][0-9])$"
)
RUNTIME_COMPLETION_CEILING_BYTES = 65_536
TERMINAL_LOG_ATTESTATION_CEILING_BYTES = 131_072
COMMAND_CEILING_BYTES = 128 * 1024 * 1024
QUERY_CEILING_BYTES = 1_000_000
TREATMENT_SCHEMA_VERSION = 1
SEED_POLICY = "sha256-domain-separated-base-seed-and-rank-v1"
SAMPLER_VERSION = "tabicl-temporary-identity/randperm-cpu-v1"
FATAL_PATTERNS = (
    (
        "non-finite value",
        re.compile(
            r"(?i)(?<![A-Za-z0-9_])(?:nan|[+-]?inf(?:inity)?|non[- ]finite)(?![A-Za-z0-9_])"
        ),
    ),
    ("out of memory", re.compile(r"(?i)(?:out of memory|\bOOM\b)")),
    (
        "storage exhausted",
        re.compile(r"(?i)(?:\bENOSPC\b|no space left on device|\[Errno\s+28\])"),
    ),
    ("traceback", re.compile(r"(?m)^Traceback \(most recent call last\):")),
)
COMPLETION_PAYLOAD_KEYS = {
    "study_id",
    "transaction_id",
    "submission_receipt_sha256",
    "transaction_ledger_sha256",
    "source_commit_sha",
    "source_tree_sha",
    "repository_binding",
    "job_id",
    "job_name",
    "arm",
    "stage",
    "seed",
    "scheduler",
    "cuda_visible_devices",
    "gpu_name",
    "gpu_uuid",
    "scheduler_stdout_path",
    "scheduler_stderr_path",
    "scheduler_log_ceiling_bytes",
    "manifest_ceiling_bytes",
    "scheduler_stdout_observed_size_at_completion",
    "scheduler_stderr_observed_size_at_completion",
    "scheduler_logs_terminal_verified",
    "finalized_artifact",
    "parent_lineage",
    "stage_exit_code",
    "completed",
}
LEDGER_ENTRY_KEYS = {
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
LEDGER_DIGEST_FIELDS = (
    "source_sha256",
    "environment_sha256",
    "prior_sha256",
    "architecture_sha256",
    "optimizer_sha256",
    "scientific_sha256",
    "cohort_protocol_sha256",
    "arm_protocol_sha256",
)
FINALIZED_PAYLOAD_KEYS = {
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
FINALIZED_DIGEST_FIELDS = (
    "checkpoint_sha256",
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
)
FINALIZED_ARTIFACT_KEYS = {
    "finalized_manifest_path",
    "finalized_manifest_sha256",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_size",
    "provenance_sha256",
    "seed_sha256",
    "treatment_sha256",
}
COMPLETION_SCHEDULER_KEYS = {
    "partition",
    "qos",
    "cpus_per_task",
    "memory_mb",
    "gpus_per_job",
    "time_limit",
    "query_sha256",
}


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite manifest number")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("manifest object key is invalid")
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    raise ValueError("manifest value has an unsupported type")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _exact(value: Any, keys: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{where} schema mismatch")
    return value


def _slurm_duration(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where} is not a canonical Slurm duration")
    match = SLURM_DURATION.fullmatch(value)
    if match is None or int(match.group("hours")) > 23:
        raise ValueError(f"{where} is not a canonical Slurm duration")
    seconds = (
        (int(match.group("days") or "0") * 24 + int(match.group("hours")))
        * 3600
        + int(match.group("minutes")) * 60
        + int(match.group("seconds"))
    )
    if seconds <= 0:
        raise ValueError(f"{where} is not a positive Slurm duration")
    return value


def _validated_repository_binding(
    value: Any, *, expected_commit_sha: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("repository binding must be an object")
    git_sha256 = value.get("git_sha256")
    if not isinstance(git_sha256, str) or HEX64.fullmatch(git_sha256) is None:
        raise ValueError("repository binding Git digest is malformed")
    path = Path(__file__).with_name("verify_git_repository.py")
    spec = importlib.util.spec_from_file_location(
        "_formal_terminal_git_repository", path
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load exact-T repository binding helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_repository_binding(
        value,
        expected_commit_sha=expected_commit_sha,
        expected_git_sha256=git_sha256,
    )


def _signature(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_physical_directory(path: Path, *, where: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"{where} is unavailable") from error
    if not stat.S_ISDIR(metadata.st_mode) or path.resolve(strict=True) != path:
        raise ValueError(f"{where} must be a physical directory")


def _read_stable_regular(
    path: Path, *, max_bytes: int, where: str, executable: bool = False
) -> tuple[bytes, tuple[int, int, int, int, int, int]]:
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes < 0
    ):
        raise ValueError(f"{where} byte ceiling is invalid")
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(f"{where} is unavailable") from error
    if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
        raise ValueError(f"{where} is not a bounded physical regular file")
    if executable and before.st_mode & 0o111 == 0:
        raise ValueError(f"{where} is not executable")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{where} could not be opened without following links") from error
    try:
        opened = os.fstat(fd)
        if _signature(opened) != _signature(before):
            raise ValueError(f"{where} changed while it was opened")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(1 << 20, max_bytes - total + 1))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"{where} exceeded its byte ceiling")
        after = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        path_after = path.lstat()
    except OSError as error:
        raise ValueError(f"{where} disappeared during verification") from error
    if _signature(opened) != _signature(after) or _signature(after) != _signature(
        path_after
    ):
        raise ValueError(f"{where} changed during verification")
    raw = b"".join(chunks)
    if len(raw) != after.st_size:
        raise ValueError(f"{where} produced a short read")
    return raw, _signature(after)


def _decode_manifest(raw: bytes, *, kind: str) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{kind} contains duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{kind} is invalid JSON") from error
    envelope = _exact(value, {"schema_version", "kind", "payload", "sha256"}, kind)
    body = {key: envelope[key] for key in ("schema_version", "kind", "payload")}
    if (
        envelope["schema_version"] != 1
        or envelope["kind"] != kind
        or envelope["sha256"] != _sha256(body)
        or raw != _canonical(envelope) + b"\n"
    ):
        raise ValueError(f"{kind} envelope or canonical bytes mismatch")
    return envelope


def _read_manifest(path: Path, *, max_bytes: int, kind: str) -> Mapping[str, Any]:
    raw, _signature_value = _read_stable_regular(
        path, max_bytes=max_bytes, where=kind
    )
    return _decode_manifest(raw, kind=kind)


def _relative_parts(value: Any, *, where: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{where} relative path is invalid")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{where} relative path is unsafe")
    return path.parts


def _artifact_relative_path(root: Path, value: Any, *, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where} artifact path is invalid")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{where} artifact path must be absolute before encoding")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{where} artifact path escaped its namespace") from error
    parts = _relative_parts(relative.as_posix(), where=where)
    return "/".join(parts)


def _relative_artifact_binding(
    root: Path, binding: Mapping[str, Any]
) -> dict[str, Any]:
    value = dict(_exact(binding, FINALIZED_ARTIFACT_KEYS, "artifact binding"))
    value["finalized_manifest_path"] = _artifact_relative_path(
        root,
        value["finalized_manifest_path"],
        where="finalized manifest",
    )
    value["checkpoint_path"] = _artifact_relative_path(
        root,
        value["checkpoint_path"],
        where="checkpoint",
    )
    return value


def _read_relative_stable_regular(
    root: Path,
    relative: str,
    *,
    max_bytes: int,
    where: str,
) -> tuple[bytes, tuple[int, int, int, int, int, int]]:
    """Read a root-relative regular file without following any path component."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError(f"{where} byte ceiling is invalid")
    parts = _relative_parts(relative, where=where)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(root, directory_flags)
    except OSError as error:
        raise ValueError(f"{where} root could not be opened without following links") from error
    try:
        for component in parts[:-1]:
            try:
                child_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            except OSError as error:
                raise ValueError(
                    f"{where} directory component could not be opened without following links"
                ) from error
            os.close(directory_fd)
            directory_fd = child_fd
        name = parts[-1]
        try:
            before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"{where} is unavailable") from error
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise ValueError(f"{where} is not a bounded physical regular file")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        try:
            fd = os.open(name, flags, dir_fd=directory_fd)
        except OSError as error:
            raise ValueError(f"{where} could not be opened without following links") from error
        try:
            opened = os.fstat(fd)
            if _signature(opened) != _signature(before):
                raise ValueError(f"{where} changed while it was opened")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(fd, min(1 << 20, max_bytes - total + 1))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"{where} exceeded its byte ceiling")
            after = os.fstat(fd)
        finally:
            os.close(fd)
        try:
            path_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"{where} disappeared during verification") from error
        if _signature(opened) != _signature(after) or _signature(after) != _signature(
            path_after
        ):
            raise ValueError(f"{where} changed during verification")
        raw = b"".join(chunks)
        if len(raw) != after.st_size:
            raise ValueError(f"{where} produced a short read")
        return raw, _signature(after)
    finally:
        os.close(directory_fd)


def _hash_relative_bounded_regular(
    root: Path,
    relative: str,
    *,
    max_bytes: int,
    where: str,
) -> tuple[int, str, tuple[int, int, int, int, int, int]]:
    """Stream and hash a bounded root-relative file without following links."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError(f"{where} byte ceiling is invalid")
    parts = _relative_parts(relative, where=where)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(root, directory_flags)
    except OSError as error:
        raise ValueError(f"{where} root could not be opened without following links") from error
    try:
        for component in parts[:-1]:
            try:
                child_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            except OSError as error:
                raise ValueError(
                    f"{where} directory component could not be opened without following links"
                ) from error
            os.close(directory_fd)
            directory_fd = child_fd
        name = parts[-1]
        try:
            before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"{where} is unavailable") from error
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise ValueError(f"{where} is not a bounded physical regular file")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        try:
            fd = os.open(name, flags, dir_fd=directory_fd)
        except OSError as error:
            raise ValueError(f"{where} could not be opened without following links") from error
        try:
            opened = os.fstat(fd)
            if _signature(opened) != _signature(before):
                raise ValueError(f"{where} changed while it was opened")
            digest = hashlib.sha256()
            total = 0
            while True:
                chunk = os.read(fd, min(1 << 20, max_bytes - total + 1))
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"{where} exceeded its byte ceiling")
            after = os.fstat(fd)
        finally:
            os.close(fd)
        try:
            path_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"{where} disappeared during verification") from error
        if _signature(opened) != _signature(after) or _signature(after) != _signature(
            path_after
        ):
            raise ValueError(f"{where} changed during verification")
        if total != after.st_size:
            raise ValueError(f"{where} produced a short read")
        return total, digest.hexdigest(), _signature(after)
    finally:
        os.close(directory_fd)


def _manifest_sha256(kind: str, payload: Mapping[str, Any]) -> str:
    return _sha256({"schema_version": 1, "kind": kind, "payload": payload})


def _expected_seed_sha256(seed: int) -> str:
    return _manifest_sha256(
        "seed",
        {
            "np_seed": seed,
            "torch_seed": seed,
            "identity_rng_seed": seed,
            "world_size": 1,
        },
    )


def _expected_treatment_sha256(arm: str, seed: int) -> str:
    identity = {
        "schema_version": TREATMENT_SCHEMA_VERSION,
        "row_identity_mode": arm,
        "identity_rng_seed": seed,
        "seed_policy": SEED_POLICY,
        "sampler_version": SAMPLER_VERSION if arm == "temporary" else None,
        "world_size": 1,
    }
    return _manifest_sha256(
        "treatment", {**identity, "manifest_sha256": _sha256(identity)}
    )


def _validate_ledger(
    *, ledger: Mapping[str, Any], study_id: str, seed: int
) -> tuple[
    dict[tuple[str, str], Mapping[str, Any]],
    str,
    dict[str, str],
]:
    ledger_payload = _exact(ledger["payload"], {"study_id", "entries"}, "ledger payload")
    entries = ledger_payload["entries"]
    if (
        ledger_payload["study_id"] != study_id
        or not isinstance(entries, list)
        or len(entries) != len(PAIR_ORDER)
    ):
        raise ValueError("transaction ledger formal matrix mismatch")
    by_pair: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw_entry, pair in zip(entries, PAIR_ORDER):
        entry = _exact(raw_entry, LEDGER_ENTRY_KEYS, "transaction ledger entry")
        arm, stage = pair
        terminal_step = TERMINAL_STEPS[stage]
        expected_checkpoint = f"arms/{arm}/{stage}/step-{terminal_step}.ckpt"
        expected_finalized = f"arms/{arm}/{stage}/finalized-checkpoint.json"
        if (
            (entry["arm"], entry["stage"]) != pair
            or entry["terminal_step"] != terminal_step
            or entry["upstream_identity"] != f"{study_id}:{arm}:{stage}"
            or entry["artifact_identity"] != f"{study_id}.{arm}.{stage}.final"
            or entry["checkpoint_relpath"] != expected_checkpoint
            or entry["finalized_manifest_relpath"] != expected_finalized
            or entry["np_seed"] != seed
            or entry["torch_seed"] != seed
            or entry["identity_rng_seed"] != seed
            or isinstance(entry["world_size"], bool)
            or entry["world_size"] != 1
            or isinstance(entry["cuda_device_count"], bool)
            or entry["cuda_device_count"] != 1
            or isinstance(entry["max_checkpoint_bytes"], bool)
            or not isinstance(entry["max_checkpoint_bytes"], int)
            or not 1 <= entry["max_checkpoint_bytes"] <= 1 << 40
        ):
            raise ValueError("transaction ledger entry binding is not canonical")
        _relative_parts(entry["checkpoint_relpath"], where="ledger checkpoint")
        _relative_parts(
            entry["finalized_manifest_relpath"], where="ledger finalized manifest"
        )
        if any(
            not isinstance(entry[field], str)
            or HEX64.fullmatch(entry[field]) is None
            for field in LEDGER_DIGEST_FIELDS
        ):
            raise ValueError("transaction ledger digest is invalid")
        by_pair[pair] = entry

    if len({entry["source_sha256"] for entry in entries}) != 1:
        raise ValueError("transaction ledger source digest differs across the cohort")
    if len({entry["environment_sha256"] for entry in entries}) != 1:
        raise ValueError("transaction ledger environment digest differs across the cohort")
    if len({entry["max_checkpoint_bytes"] for entry in entries}) != 1:
        raise ValueError("transaction ledger checkpoint ceiling differs across the cohort")

    seed_sha256 = _expected_seed_sha256(seed)
    treatment_by_arm = {
        arm: _expected_treatment_sha256(arm, seed) for arm in ARMS
    }
    for stage in STAGES:
        stage_entries = [by_pair[(arm, stage)] for arm in ARMS]
        for field in (
            "source_sha256",
            "environment_sha256",
            "prior_sha256",
            "architecture_sha256",
            "optimizer_sha256",
            "scientific_sha256",
            "cohort_protocol_sha256",
        ):
            if len({entry[field] for entry in stage_entries}) != 1:
                raise ValueError(f"transaction ledger {field} differs within a stage")
        exemplar = stage_entries[0]
        expected_cohort = _manifest_sha256(
            "cohort_protocol",
            {
                "stage": stage,
                "terminal_step": TERMINAL_STEPS[stage],
                "source_sha256": exemplar["source_sha256"],
                "environment_sha256": exemplar["environment_sha256"],
                "architecture_sha256": exemplar["architecture_sha256"],
                "prior_sha256": exemplar["prior_sha256"],
                "optimizer_sha256": exemplar["optimizer_sha256"],
                "seed_sha256": seed_sha256,
                "scientific_config_sha256": exemplar["scientific_sha256"],
            },
        )
        if exemplar["cohort_protocol_sha256"] != expected_cohort:
            raise ValueError("transaction ledger cohort protocol digest is not derived")
        for arm in ARMS:
            expected_arm = _manifest_sha256(
                "arm_protocol",
                {
                    "cohort_protocol_sha256": expected_cohort,
                    "mode": arm,
                    "treatment_sha256": treatment_by_arm[arm],
                },
            )
            if by_pair[(arm, stage)]["arm_protocol_sha256"] != expected_arm:
                raise ValueError("transaction ledger arm protocol digest is not derived")
    return by_pair, seed_sha256, treatment_by_arm


def _validate_finalized_artifact(
    *,
    artifact_root: Path,
    study_id: str,
    entry: Mapping[str, Any],
    seed_sha256: str,
    treatment_sha256: str,
    manifest_ceiling: int,
) -> dict[str, Any]:
    finalized_raw, _finalized_signature = _read_relative_stable_regular(
        artifact_root,
        entry["finalized_manifest_relpath"],
        max_bytes=manifest_ceiling,
        where="finalized checkpoint manifest",
    )
    finalized = _decode_manifest(finalized_raw, kind="finalized_checkpoint")
    finalized_payload = _exact(
        finalized["payload"], FINALIZED_PAYLOAD_KEYS, "finalized checkpoint payload"
    )
    expected_bindings = {
        "study_id": study_id,
        "arm": entry["arm"],
        "stage": entry["stage"],
        "terminal_step": entry["terminal_step"],
        "upstream_identity": entry["upstream_identity"],
        "artifact_identity": entry["artifact_identity"],
        "source_sha256": entry["source_sha256"],
        "environment_sha256": entry["environment_sha256"],
        "prior_sha256": entry["prior_sha256"],
        "architecture_sha256": entry["architecture_sha256"],
        "optimizer_sha256": entry["optimizer_sha256"],
        "scientific_sha256": entry["scientific_sha256"],
        "cohort_protocol_sha256": entry["cohort_protocol_sha256"],
        "arm_protocol_sha256": entry["arm_protocol_sha256"],
        "seed_sha256": seed_sha256,
        "treatment_sha256": treatment_sha256,
        "cuda_device_count": 1,
        "max_checkpoint_bytes": entry["max_checkpoint_bytes"],
    }
    if any(finalized_payload[key] != value for key, value in expected_bindings.items()):
        raise ValueError("finalized checkpoint differs from immutable ledger bindings")
    if any(
        not isinstance(finalized_payload[field], str)
        or HEX64.fullmatch(finalized_payload[field]) is None
        for field in FINALIZED_DIGEST_FIELDS
    ):
        raise ValueError("finalized checkpoint digest is invalid")
    if (
        isinstance(finalized_payload["checkpoint_size"], bool)
        or not isinstance(finalized_payload["checkpoint_size"], int)
        or not 0 <= finalized_payload["checkpoint_size"] <= entry["max_checkpoint_bytes"]
        or isinstance(finalized_payload["cuda_device_count"], bool)
        or not isinstance(finalized_payload["cuda_device_count"], int)
        or isinstance(finalized_payload["max_checkpoint_bytes"], bool)
        or not isinstance(finalized_payload["max_checkpoint_bytes"], int)
    ):
        raise ValueError("finalized checkpoint size is invalid")
    checkpoint_size, checkpoint_sha256, _checkpoint_signature = (
        _hash_relative_bounded_regular(
            artifact_root,
            entry["checkpoint_relpath"],
            max_bytes=entry["max_checkpoint_bytes"],
            where="finalized checkpoint",
        )
    )
    if (
        finalized_payload["checkpoint_size"] != checkpoint_size
        or finalized_payload["checkpoint_sha256"] != checkpoint_sha256
    ):
        raise ValueError("finalized checkpoint digest or size mismatch")
    binding = {
        "finalized_manifest_path": str(
            artifact_root / entry["finalized_manifest_relpath"]
        ),
        "finalized_manifest_sha256": finalized["sha256"],
        "checkpoint_path": str(artifact_root / entry["checkpoint_relpath"]),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size": checkpoint_size,
        "provenance_sha256": finalized_payload["provenance_sha256"],
        "seed_sha256": finalized_payload["seed_sha256"],
        "treatment_sha256": finalized_payload["treatment_sha256"],
    }
    if set(binding) != FINALIZED_ARTIFACT_KEYS:
        raise AssertionError("internal finalized-artifact schema drift")
    return {
        "binding": binding,
        "finalized_manifest_file_sha256": hashlib.sha256(finalized_raw).hexdigest(),
        "finalized_manifest_size": len(finalized_raw),
    }


def _trusted_executable(path: Path, expected_sha256: str) -> int:
    if HEX64.fullmatch(expected_sha256 or "") is None:
        raise ValueError("trusted sacct digest is invalid")
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_size > COMMAND_CEILING_BYTES
        or before.st_mode & 0o111 == 0
    ):
        raise ValueError("trusted sacct executable is invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if _signature(opened) != _signature(before):
            raise ValueError("trusted sacct executable changed while opening")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
        path_after = path.lstat()
        if (
            _signature(opened) != _signature(after)
            or _signature(after) != _signature(path_after)
            or digest.hexdigest() != expected_sha256
        ):
            raise ValueError("trusted sacct executable digest or stability mismatch")
        os.lseek(fd, 0, os.SEEK_SET)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _query_terminal_jobs(
    *,
    sacct_path: Path,
    sacct_sha256: str,
    jobs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    by_cluster: dict[str | None, list[Mapping[str, Any]]] = {}
    for job in jobs:
        cluster = job["cluster"]
        if cluster is not None and (
            not isinstance(cluster, str) or SAFE_CLUSTER.fullmatch(cluster) is None
        ):
            raise ValueError("submission receipt cluster is invalid")
        by_cluster.setdefault(cluster, []).append(job)
    observations: dict[str, dict[str, str]] = {}
    query_evidence: list[dict[str, Any]] = []
    for cluster, cluster_jobs in by_cluster.items():
        job_ids = [job["job_id"] for job in cluster_jobs]
        command_fd = _trusted_executable(sacct_path, sacct_sha256)
        argv = [f"/proc/self/fd/{command_fd}"]
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
        try:
            completed = subprocess.run(
                argv,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                pass_fds=(command_fd,),
            )
        finally:
            os.close(command_fd)
        if (
            completed.returncode != 0
            or len(completed.stdout) > QUERY_CEILING_BYTES
            or len(completed.stderr) > QUERY_CEILING_BYTES
        ):
            raise ValueError("trusted sacct terminal query failed")
        expected_names = {job["job_id"]: job["job_name"] for job in cluster_jobs}
        seen: dict[str, dict[str, str]] = {}
        for raw_line in completed.stdout.decode("utf-8", errors="strict").splitlines():
            fields = raw_line.split("|")
            if fields and fields[-1] == "":
                fields.pop()
            if len(fields) != 5:
                raise ValueError("sacct returned a malformed allocation row")
            job_id, job_name, raw_state, exit_code, derived_exit_code = fields
            if job_id not in expected_names or job_name != expected_names[job_id]:
                raise ValueError("sacct returned an unknown allocation row")
            if job_id in seen:
                raise ValueError("sacct returned a duplicate allocation row")
            state = raw_state.split()[0].split("+")[0] if raw_state.split() else ""
            if (
                state != "COMPLETED"
                or exit_code != "0:0"
                or derived_exit_code != "0:0"
            ):
                raise ValueError("formal allocation is not successfully terminal")
            seen[job_id] = {
                "state": state,
                "exit_code": exit_code,
                "derived_exit_code": derived_exit_code,
            }
        if set(seen) != set(expected_names):
            raise ValueError("sacct omitted a formal allocation row")
        observations.update(seen)
        query_evidence.append(
            {
                "cluster": cluster,
                "job_ids": job_ids,
                "query_argv_sha256": hashlib.sha256(
                    _canonical(argv[1:])
                ).hexdigest(),
                "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
            }
        )
    return observations, query_evidence


def _fatal_findings(raw: bytes) -> list[str]:
    text = raw.decode("utf-8", errors="replace")
    return [name for name, pattern in FATAL_PATTERNS if pattern.search(text)]


def _stable_log_evidence(path: Path, *, max_bytes: int, stream: str) -> dict[str, Any]:
    first, first_signature = _read_stable_regular(
        path, max_bytes=max_bytes, where=f"scheduler {stream} log"
    )
    findings = _fatal_findings(first)
    if findings:
        raise ValueError("scheduler log contains fatal signatures: " + ", ".join(findings))
    digest = hashlib.sha256(first).hexdigest()
    second, second_signature = _read_stable_regular(
        path, max_bytes=max_bytes, where=f"scheduler {stream} log second read"
    )
    if (
        first_signature != second_signature
        or len(second) != len(first)
        or hashlib.sha256(second).hexdigest() != digest
        or _fatal_findings(second)
    ):
        raise ValueError("scheduler log changed after its terminal hash")
    return {
        "stream": stream,
        "path": str(path),
        "size": len(first),
        "sha256": digest,
        "fatal_signature_scan_passed": True,
        "stable_reads": 2,
    }


def _publish_no_replace(path: Path, value: Mapping[str, Any], *, max_bytes: int) -> None:
    raw = _canonical(value) + b"\n"
    if len(raw) > max_bytes:
        raise ValueError("terminal scheduler-log attestation exceeds its byte ceiling")
    if path.exists() or path.is_symlink():
        raise FileExistsError("terminal scheduler-log attestation is write-once")
    parent = path.parent
    metadata = parent.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("terminal attestation parent must be a physical directory")
    temporary = parent / f".{path.name}.{os.getpid()}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temporary, flags, 0o400)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(fd, raw[offset:])
            if written <= 0:
                raise OSError("short terminal attestation write")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(
            parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def finalize(
    *,
    submission_receipt: Path,
    transaction_ledger: Path,
    artifact_root: Path,
    max_metadata_bytes: int,
) -> Mapping[str, Any]:
    if (
        isinstance(max_metadata_bytes, bool)
        or not isinstance(max_metadata_bytes, int)
        or not TERMINAL_LOG_ATTESTATION_CEILING_BYTES
        <= max_metadata_bytes
        <= 1 << 40
    ):
        raise ValueError("formal metadata byte ceiling is invalid")
    if not artifact_root.is_absolute() or artifact_root.resolve(strict=True) != artifact_root:
        raise ValueError("artifact root must be an absolute physical directory")
    _require_physical_directory(artifact_root, where="artifact root")
    if (
        submission_receipt != artifact_root / "submission-receipt.json"
        or transaction_ledger != artifact_root / "transaction-ledger.json"
    ):
        raise ValueError("receipt or ledger path is outside the formal namespace")
    receipt = _read_manifest(
        submission_receipt,
        max_bytes=max_metadata_bytes,
        kind="held_submission_receipt",
    )
    ledger = _read_manifest(
        transaction_ledger,
        max_bytes=max_metadata_bytes,
        kind="transaction_ledger",
    )
    payload = _exact(
        receipt["payload"],
        {
            "study_id",
            "seed",
            "transaction_id",
            "transaction_ledger_sha256",
            "source_commit_sha",
            "source_tree_sha",
            "repository_binding",
            "jobs_held_at_publication",
            "run_log_ceiling_bytes",
            "manifest_ceiling_bytes",
            "runtime_completion_ceiling_bytes",
            "terminal_log_attestation_path",
            "transaction_commit_path",
            "terminal_log_attestation_ceiling_bytes",
            "scheduler",
            "job_ids",
            "jobs",
        },
        "submission receipt payload",
    )
    if (
        payload["jobs_held_at_publication"] is not True
        or not isinstance(payload["study_id"], str)
        or SAFE_ID.fullmatch(payload["study_id"]) is None
        or isinstance(payload["seed"], bool)
        or payload["seed"] not in {42, 43, 44}
        or not payload["study_id"].endswith(f"-seed{payload['seed']}")
        or artifact_root.name != payload["study_id"]
        or not isinstance(payload["transaction_id"], str)
        or re.fullmatch(r"[0-9a-f]{32}", payload["transaction_id"]) is None
        or payload["transaction_ledger_sha256"] != ledger["sha256"]
        or not isinstance(payload["source_commit_sha"], str)
        or HEX40.fullmatch(payload["source_commit_sha"]) is None
        or not isinstance(payload["source_tree_sha"], str)
        or HEX40.fullmatch(payload["source_tree_sha"]) is None
        or payload["runtime_completion_ceiling_bytes"]
        != RUNTIME_COMPLETION_CEILING_BYTES
        or payload["terminal_log_attestation_ceiling_bytes"]
        != TERMINAL_LOG_ATTESTATION_CEILING_BYTES
        or not isinstance(payload["terminal_log_attestation_path"], str)
        or isinstance(payload["run_log_ceiling_bytes"], bool)
        or not isinstance(payload["run_log_ceiling_bytes"], int)
        or not 1 <= payload["run_log_ceiling_bytes"] <= 1 << 40
        or isinstance(payload["manifest_ceiling_bytes"], bool)
        or not isinstance(payload["manifest_ceiling_bytes"], int)
        or not 1 <= payload["manifest_ceiling_bytes"] <= max_metadata_bytes
    ):
        raise ValueError("submission receipt terminal-evidence binding mismatch")
    repository_binding = _validated_repository_binding(
        payload["repository_binding"],
        expected_commit_sha=payload["source_commit_sha"],
    )
    output = Path(payload["terminal_log_attestation_path"])
    if output != artifact_root / "terminal-scheduler-logs.json":
        raise ValueError("terminal attestation path is outside the formal namespace")
    commit_path = Path(payload["transaction_commit_path"])
    if commit_path != artifact_root / "transaction-committed.json":
        raise ValueError("submission commit path is outside the formal namespace")
    rollback_path = artifact_root / (
        f"rollback-incomplete-{payload['transaction_id']}.json"
    )
    if rollback_path.exists() or rollback_path.is_symlink():
        raise ValueError("formal submission has a rollback recovery record")
    commit = _read_manifest(
        commit_path,
        max_bytes=max_metadata_bytes,
        kind="formal_submission_commit",
    )
    commit_payload = _exact(
        commit["payload"],
        {
            "study_id",
            "transaction_id",
            "submission_receipt_sha256",
            "transaction_ledger_sha256",
            "job_ids",
        },
        "formal submission commit payload",
    )
    if (
        commit_payload["study_id"] != payload["study_id"]
        or commit_payload["transaction_id"] != payload["transaction_id"]
        or commit_payload["submission_receipt_sha256"] != receipt["sha256"]
        or commit_payload["transaction_ledger_sha256"] != ledger["sha256"]
        or commit_payload["job_ids"] != payload["job_ids"]
    ):
        raise ValueError("formal submission commit binding mismatch")
    scheduler = _exact(
        payload["scheduler"],
        {
            "partition",
            "qos",
            "cpus_per_task",
            "memory_mb",
            "gpus_per_job",
            "time_limit_by_stage",
            "sacct_path",
            "sacct_sha256",
            "scontrol_sha256",
        },
        "receipt scheduler",
    )
    time_limit_by_stage = _exact(
        scheduler["time_limit_by_stage"],
        set(STAGES),
        "receipt time_limit_by_stage",
    )
    normalized_time_limits = {
        stage: _slurm_duration(
            time_limit_by_stage[stage], f"receipt {stage} time limit"
        )
        for stage in STAGES
    }
    if (
        {
            key: scheduler[key]
            for key in (
                "partition",
                "qos",
                "cpus_per_task",
                "memory_mb",
                "gpus_per_job",
            )
        }
        != {
            "partition": "h100",
            "qos": "long",
            "cpus_per_task": 64,
            "memory_mb": 131_072,
            "gpus_per_job": 1,
        }
        or any(
            isinstance(scheduler[key], bool)
            or not isinstance(scheduler[key], int)
            for key in ("cpus_per_task", "memory_mb", "gpus_per_job")
        )
        or not isinstance(scheduler["sacct_sha256"], str)
        or HEX64.fullmatch(scheduler["sacct_sha256"]) is None
        or not isinstance(scheduler["scontrol_sha256"], str)
        or HEX64.fullmatch(scheduler["scontrol_sha256"]) is None
    ):
        raise ValueError("receipt scheduler contract is not canonical")
    if not isinstance(scheduler["sacct_path"], str):
        raise ValueError("receipt sacct path must be a string")
    sacct_path = Path(scheduler["sacct_path"])
    if not sacct_path.is_absolute():
        raise ValueError("receipt sacct path must be absolute")
    jobs = payload["jobs"]
    if not isinstance(jobs, list) or len(jobs) != 9:
        raise ValueError("submission receipt must contain exactly nine formal jobs")
    job_ids = payload["job_ids"]
    if not isinstance(job_ids, list) or len(job_ids) != 9 or any(
        not isinstance(value, str) or JOB_ID.fullmatch(value) is None
        for value in job_ids
    ):
        raise ValueError("submission receipt job ID set is invalid")
    expected_job_ids = [
        job.get("job_id") for job in jobs if isinstance(job, Mapping)
    ]
    if job_ids != expected_job_ids or len(set(job_ids)) != 9:
        raise ValueError("submission receipt job ID set is invalid")
    expected_pairs = {(arm, stage) for stage in STAGES for arm in ARMS}
    observed_pairs: set[tuple[str, str]] = set()
    normalized_jobs: list[Mapping[str, Any]] = []
    for raw_job, expected_pair in zip(jobs, PAIR_ORDER):
        job = _exact(
            raw_job,
            {
                "arm",
                "stage",
                "terminal_step",
                "time_limit",
                "job_id",
                "cluster",
                "job_name",
                "parent_job_id",
                "scheduler_stdout",
                "scheduler_stderr",
                "completion_path",
            },
            "submission receipt job",
        )
        pair = (job["arm"], job["stage"])
        if pair != expected_pair or pair in observed_pairs or pair not in expected_pairs:
            raise ValueError("submission receipt formal matrix is invalid")
        observed_pairs.add(pair)
        if (
            not isinstance(job["job_id"], str)
            or JOB_ID.fullmatch(job["job_id"]) is None
            or job["time_limit"] != normalized_time_limits[job["stage"]]
            or not isinstance(job["job_name"], str)
            or not job["job_name"]
            or job["cluster"] is not None
            and (
                not isinstance(job["cluster"], str)
                or SAFE_CLUSTER.fullmatch(job["cluster"]) is None
            )
        ):
            raise ValueError("submission receipt job scheduler identity is invalid")
        for key in ("scheduler_stdout", "scheduler_stderr", "completion_path"):
            if not isinstance(job[key], str):
                raise ValueError("submission receipt artifact path is invalid")
            path = Path(job[key])
            if not path.is_absolute() or artifact_root not in path.parents:
                raise ValueError("submission receipt artifact path escaped its namespace")
        expected_slug = f"{job['arm']}-{job['stage']}"
        if (
            Path(job["scheduler_stdout"])
            != artifact_root / "scheduler-logs" / f"{expected_slug}.out"
            or Path(job["scheduler_stderr"])
            != artifact_root / "scheduler-logs" / f"{expected_slug}.err"
            or Path(job["completion_path"])
            != artifact_root / "runtime-completions" / f"{expected_slug}.json"
        ):
            raise ValueError("submission receipt artifact layout is not canonical")
        normalized_jobs.append(job)
    if observed_pairs != expected_pairs:
        raise ValueError("submission receipt formal matrix is incomplete")
    _require_physical_directory(
        artifact_root / "scheduler-logs", where="scheduler log directory"
    )
    _require_physical_directory(
        artifact_root / "runtime-completions", where="runtime completion directory"
    )
    expected_log_names = {
        Path(job[key]).name
        for job in normalized_jobs
        for key in ("scheduler_stdout", "scheduler_stderr")
    }
    expected_completion_names = {
        Path(job["completion_path"]).name for job in normalized_jobs
    }
    if {
        path.name for path in (artifact_root / "scheduler-logs").iterdir()
    } != expected_log_names:
        raise ValueError("scheduler log directory has a missing or unknown entry")
    if {
        path.name for path in (artifact_root / "runtime-completions").iterdir()
    } != expected_completion_names:
        raise ValueError("runtime completion directory has a missing or unknown entry")
    jobs_by_pair = {(job["arm"], job["stage"]): job for job in normalized_jobs}
    terminal_steps = TERMINAL_STEPS
    for arm in ARMS:
        for stage_index, stage in enumerate(terminal_steps, start=1):
            job = jobs_by_pair[(arm, stage)]
            expected_parent = (
                None
                if stage_index == 1
                else jobs_by_pair[(arm, STAGES[stage_index - 2])]["job_id"]
            )
            if (
                job["terminal_step"] != terminal_steps[stage]
                or job["parent_job_id"] != expected_parent
                or job["job_name"]
                != f"tabicl-{payload['transaction_id']}-{arm}-s{stage_index}"
            ):
                raise ValueError("submission receipt stage lineage is not canonical")
    ledger_by_pair, expected_seed_sha256, expected_treatment_sha256 = _validate_ledger(
        ledger=ledger,
        study_id=payload["study_id"],
        seed=payload["seed"],
    )

    initial_artifacts: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in PAIR_ORDER:
        initial_artifacts[pair] = _validate_finalized_artifact(
            artifact_root=artifact_root,
            study_id=payload["study_id"],
            entry=ledger_by_pair[pair],
            seed_sha256=expected_seed_sha256,
            treatment_sha256=expected_treatment_sha256[pair[0]],
            manifest_ceiling=payload["manifest_ceiling_bytes"],
        )

    completion_evidence: dict[str, dict[str, Any]] = {}
    completion_envelopes: dict[str, Mapping[str, Any]] = {}
    for job in normalized_jobs:
        completion_path = Path(job["completion_path"])
        completion = _read_manifest(
            completion_path,
            max_bytes=RUNTIME_COMPLETION_CEILING_BYTES,
            kind="formal_job_completion",
        )
        # _read_manifest validates that its single stable read is exactly the
        # canonical newline-terminated envelope. Derive the file digest from
        # those same bytes instead of performing a second, mixable path read.
        completion_raw = _canonical(completion) + b"\n"
        completion_payload = _exact(
            completion["payload"],
            COMPLETION_PAYLOAD_KEYS,
            "formal runtime completion payload",
        )
        observed_stdout_size = completion_payload.get(
            "scheduler_stdout_observed_size_at_completion"
        )
        observed_stderr_size = completion_payload.get(
            "scheduler_stderr_observed_size_at_completion"
        )
        completion_scheduler = _exact(
            completion_payload["scheduler"],
            COMPLETION_SCHEDULER_KEYS,
            "formal runtime completion scheduler",
        )
        expected_scheduler_resources = {
            key: scheduler[key]
            for key in (
                "partition",
                "qos",
                "cpus_per_task",
                "memory_mb",
                "gpus_per_job",
            )
        }
        pair = (job["arm"], job["stage"])
        expected_scheduler_resources["time_limit"] = normalized_time_limits[
            job["stage"]
        ]
        completed_artifact = _exact(
            completion_payload["finalized_artifact"],
            FINALIZED_ARTIFACT_KEYS,
            "formal runtime completion finalized artifact",
        )
        stage_index = STAGES.index(job["stage"])
        if stage_index == 0:
            expected_parent_lineage = None
        else:
            parent_stage = STAGES[stage_index - 1]
            parent_job = jobs_by_pair[(job["arm"], parent_stage)]
            expected_parent_lineage = {
                **initial_artifacts[(job["arm"], parent_stage)]["binding"],
                "job_id": parent_job["job_id"],
                "stage": parent_stage,
            }
        if (
            completion_payload.get("study_id") != payload["study_id"]
            or completion_payload.get("transaction_id") != payload["transaction_id"]
            or completion_payload.get("submission_receipt_sha256") != receipt["sha256"]
            or completion_payload.get("transaction_ledger_sha256") != ledger["sha256"]
            or completion_payload.get("source_commit_sha") != payload["source_commit_sha"]
            or completion_payload.get("source_tree_sha") != payload["source_tree_sha"]
            or completion_payload.get("repository_binding") != repository_binding
            or completion_payload.get("job_id") != job["job_id"]
            or completion_payload.get("job_name") != job["job_name"]
            or completion_payload.get("arm") != job["arm"]
            or completion_payload.get("stage") != job["stage"]
            or completion_payload.get("seed") != payload["seed"]
            or {
                key: completion_scheduler[key]
                for key in expected_scheduler_resources
            }
            != expected_scheduler_resources
            or any(
                isinstance(completion_scheduler[key], bool)
                or not isinstance(completion_scheduler[key], int)
                for key in ("cpus_per_task", "memory_mb", "gpus_per_job")
            )
            or not isinstance(completion_scheduler["query_sha256"], str)
            or HEX64.fullmatch(completion_scheduler["query_sha256"]) is None
            or not isinstance(completion_payload.get("cuda_visible_devices"), str)
            or VISIBLE_DEVICE.fullmatch(
                completion_payload["cuda_visible_devices"]
            )
            is None
            or not isinstance(completion_payload.get("gpu_name"), str)
            or not 1 <= len(completion_payload["gpu_name"]) <= 256
            or "\n" in completion_payload["gpu_name"]
            or "\r" in completion_payload["gpu_name"]
            or "H100" not in completion_payload["gpu_name"]
            or not isinstance(completion_payload.get("gpu_uuid"), str)
            or GPU_UUID.fullmatch(completion_payload["gpu_uuid"]) is None
            or completion_payload.get("scheduler_stdout_path") != job["scheduler_stdout"]
            or completion_payload.get("scheduler_stderr_path") != job["scheduler_stderr"]
            or completion_payload.get("scheduler_log_ceiling_bytes")
            != payload["run_log_ceiling_bytes"]
            or completion_payload.get("manifest_ceiling_bytes")
            != payload["manifest_ceiling_bytes"]
            or isinstance(observed_stdout_size, bool)
            or not isinstance(observed_stdout_size, int)
            or not 0 <= observed_stdout_size <= payload["run_log_ceiling_bytes"]
            or isinstance(observed_stderr_size, bool)
            or not isinstance(observed_stderr_size, int)
            or not 0 <= observed_stderr_size <= payload["run_log_ceiling_bytes"]
            or completion_payload.get("scheduler_logs_terminal_verified") is not False
            or isinstance(completion_payload.get("stage_exit_code"), bool)
            or not isinstance(completion_payload.get("stage_exit_code"), int)
            or completion_payload.get("stage_exit_code") != 0
            or completion_payload.get("completed") is not True
            or completed_artifact != initial_artifacts[pair]["binding"]
            or completion_payload.get("parent_lineage") != expected_parent_lineage
        ):
            raise ValueError("formal runtime completion binding mismatch")
        completion_envelopes[job["job_id"]] = completion
        completion_evidence[job["job_id"]] = {
            "path": _artifact_relative_path(
                artifact_root,
                str(completion_path),
                where="runtime completion",
            ),
            "manifest_sha256": completion["sha256"],
            "file_sha256": hashlib.sha256(completion_raw).hexdigest(),
            "stdout_observed_size": observed_stdout_size,
            "stderr_observed_size": observed_stderr_size,
            "seed": completion_payload["seed"],
            "scheduler": dict(completion_scheduler),
            "cuda_visible_devices": completion_payload["cuda_visible_devices"],
            "gpu_name": completion_payload["gpu_name"],
            "gpu_uuid": completion_payload["gpu_uuid"],
            "repository_binding": dict(repository_binding),
            "manifest_ceiling_bytes": payload["manifest_ceiling_bytes"],
        }

    terminal, terminal_queries = _query_terminal_jobs(
        sacct_path=sacct_path,
        sacct_sha256=scheduler["sacct_sha256"],
        jobs=normalized_jobs,
    )
    final_jobs: list[dict[str, Any]] = []
    for job in normalized_jobs:
        logs = [
            _stable_log_evidence(
                Path(job["scheduler_stdout"]),
                max_bytes=payload["run_log_ceiling_bytes"],
                stream="stdout",
            ),
            _stable_log_evidence(
                Path(job["scheduler_stderr"]),
                max_bytes=payload["run_log_ceiling_bytes"],
                stream="stderr",
            ),
        ]
        for log in logs:
            log["path"] = _artifact_relative_path(
                artifact_root,
                log["path"],
                where=f"scheduler {log['stream']} log",
            )
        completion_record = completion_evidence[job["job_id"]]
        if (
            logs[0]["size"] < completion_record["stdout_observed_size"]
            or logs[1]["size"] < completion_record["stderr_observed_size"]
        ):
            raise ValueError("terminal scheduler log was truncated after job completion")
        final_jobs.append(
            {
                "arm": job["arm"],
                "stage": job["stage"],
                "job_id": job["job_id"],
                "cluster": job["cluster"],
                "job_name": job["job_name"],
                "state": terminal[job["job_id"]]["state"],
                "exit_code": terminal[job["job_id"]]["exit_code"],
                "derived_exit_code": terminal[job["job_id"]][
                    "derived_exit_code"
                ],
                "completion": completion_record,
                "scheduler_logs": logs,
            }
        )

    # Re-open every durable binding after the terminal scheduler query and log
    # scan.  The finalized manifests and checkpoints are intentionally hashed
    # independently from the in-job completion envelopes, then hashed again
    # immediately before publication.  A completion can therefore neither
    # substitute its own claimed digest nor hide a late artifact replacement.
    receipt_again = _read_manifest(
        submission_receipt,
        max_bytes=max_metadata_bytes,
        kind="held_submission_receipt",
    )
    ledger_again = _read_manifest(
        transaction_ledger,
        max_bytes=max_metadata_bytes,
        kind="transaction_ledger",
    )
    if receipt_again != receipt or ledger_again != ledger:
        raise ValueError("receipt or transaction ledger changed before publication")

    final_artifacts: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in PAIR_ORDER:
        artifact = _validate_finalized_artifact(
            artifact_root=artifact_root,
            study_id=payload["study_id"],
            entry=ledger_by_pair[pair],
            seed_sha256=expected_seed_sha256,
            treatment_sha256=expected_treatment_sha256[pair[0]],
            manifest_ceiling=payload["manifest_ceiling_bytes"],
        )
        if artifact != initial_artifacts[pair]:
            raise ValueError("finalized artifact changed before publication")
        final_artifacts[pair] = artifact

    for job in normalized_jobs:
        completion_again = _read_manifest(
            Path(job["completion_path"]),
            max_bytes=RUNTIME_COMPLETION_CEILING_BYTES,
            kind="formal_job_completion",
        )
        if completion_again != completion_envelopes[job["job_id"]]:
            raise ValueError("runtime completion changed before publication")

    final_jobs_by_pair = {
        (item["arm"], item["stage"]): item for item in final_jobs
    }
    for pair in PAIR_ORDER:
        relative_artifact = {
            **final_artifacts[pair],
            "binding": _relative_artifact_binding(
                artifact_root, final_artifacts[pair]["binding"]
            ),
        }
        final_jobs_by_pair[pair]["finalized_artifact"] = {
            **relative_artifact,
            "independent_stable_validations": 2,
            "initial_validation_sha256": _sha256(initial_artifacts[pair]),
            "publication_validation_sha256": _sha256(final_artifacts[pair]),
        }
        stage_index = STAGES.index(pair[1])
        final_jobs_by_pair[pair]["parent_lineage"] = (
            None
            if stage_index == 0
            else {
                **_relative_artifact_binding(
                    artifact_root,
                    final_artifacts[(pair[0], STAGES[stage_index - 1])][
                        "binding"
                    ],
                ),
                "job_id": jobs_by_pair[(pair[0], STAGES[stage_index - 1])][
                    "job_id"
                ],
                "stage": STAGES[stage_index - 1],
            }
        )
    terminal_payload = {
        "study_id": payload["study_id"],
        "seed": payload["seed"],
        "transaction_id": payload["transaction_id"],
        "submission_receipt_sha256": receipt["sha256"],
        "transaction_ledger_sha256": ledger["sha256"],
        "source_commit_sha": payload["source_commit_sha"],
        "source_tree_sha": payload["source_tree_sha"],
        "repository_binding": dict(repository_binding),
        "manifest_ceiling_bytes": payload["manifest_ceiling_bytes"],
        "sacct_sha256": scheduler["sacct_sha256"],
        "path_format": "artifact_root_relative_posix_v1",
        "terminal_verified": True,
        "terminal_queries": terminal_queries,
        "jobs": final_jobs,
    }
    body = {
        "schema_version": 1,
        "kind": "formal_terminal_scheduler_logs",
        "payload": terminal_payload,
    }
    attestation = {**body, "sha256": _sha256(body)}
    _publish_no_replace(
        output,
        attestation,
        max_bytes=TERMINAL_LOG_ATTESTATION_CEILING_BYTES,
    )
    return attestation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-receipt", required=True, type=Path)
    parser.add_argument("--transaction-ledger", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--max-metadata-bytes", required=True, type=int)
    args = parser.parse_args(argv)
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise ValueError("terminal scheduler-log finalizer requires Python -I -B")
    value = finalize(
        submission_receipt=args.submission_receipt,
        transaction_ledger=args.transaction_ledger,
        artifact_root=args.artifact_root,
        max_metadata_bytes=args.max_metadata_bytes,
    )
    sys.stdout.buffer.write(_canonical(value) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
