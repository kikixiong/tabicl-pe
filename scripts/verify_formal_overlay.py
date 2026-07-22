#!/usr/bin/env python3
"""Validate and transact the fresh three-arm formal submission overlay.

The immutable transaction ledger intentionally uses exactly the schema consumed
by :mod:`tabicl.train._provenance`.  Scheduler job IDs live only in a separate
submission receipt or rollback-recovery record; future checkpoint hashes are
not knowable, and therefore are not present, at initial submission time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
ARMS = ("rope", "temporary", "none")
STAGES = (("stage1", 500_000), ("stage2", 40_000), ("stage3", 10_000))
SEED = 42
WORLD_SIZE = 1
CUDA_DEVICE_COUNT = 1
TREATMENT_SCHEMA_VERSION = 1
SEED_POLICY = "sha256-domain-separated-base-seed-and-rank-v1"
SAMPLER_VERSION = "tabicl-temporary-identity/randperm-cpu-v1"

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_JOB_ID = re.compile(r"^[1-9][0-9]{0,19}$")
_SBATCH_RESULT = re.compile(
    r"^([1-9][0-9]{0,19})(?:;[A-Za-z0-9][A-Za-z0-9._-]{0,63})?$"
)
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


def _exact_keys(value: Any, expected: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{where} keys mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
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


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _safe_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{where} is invalid")
    return value


def _safe_text(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(character in value for character in ("\x00", "\n", "\r", ","))
    ):
        raise ValueError(f"{where} is invalid")
    return value


def _absolute_path(value: Any, where: str) -> Path:
    raw = _safe_text(value, where)
    path = Path(raw)
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{where} must be an absolute normalized path")
    return path


def _relative_path(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError(f"{where} is invalid")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{where} must be normalized and relative")
    if pure.as_posix() != value:
        raise ValueError(f"{where} must be normalized")
    return value


def _open_directory_nofollow(path: Path, *, where: str) -> int:
    """Open every directory component without following symbolic links."""
    if (
        not path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
    ):
        raise ValueError(f"{where} must be a normalized no-follow absolute directory")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    fd = os.open(os.path.sep, flags)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except Exception:
        os.close(fd)
        raise


def _require_physical_directory(path: Path, *, where: str) -> None:
    try:
        fd = _open_directory_nofollow(path, where=where)
    except OSError as error:
        raise ValueError(f"{where} must exist without symbolic-link components") from error
    os.close(fd)


def _identity_treatment(mode: str) -> dict[str, Any]:
    body = {
        "schema_version": TREATMENT_SCHEMA_VERSION,
        "row_identity_mode": mode,
        "identity_rng_seed": SEED,
        "seed_policy": SEED_POLICY,
        "sampler_version": SAMPLER_VERSION if mode == "temporary" else None,
        "world_size": WORLD_SIZE,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _expected_protocols(
    *,
    stage: str,
    terminal_step: int,
    source_sha256: str,
    environment_sha256: str,
    prior_sha256: str,
    architecture_sha256: str,
    optimizer_sha256: str,
    scientific_sha256: str,
) -> tuple[str, dict[str, str]]:
    seed = make_manifest(
        "seed",
        {
            "np_seed": SEED,
            "torch_seed": SEED,
            "identity_rng_seed": SEED,
            "world_size": WORLD_SIZE,
        },
    )
    cohort = make_manifest(
        "cohort_protocol",
        {
            "stage": stage,
            "terminal_step": terminal_step,
            "source_sha256": source_sha256,
            "environment_sha256": environment_sha256,
            "architecture_sha256": architecture_sha256,
            "prior_sha256": prior_sha256,
            "optimizer_sha256": optimizer_sha256,
            "seed_sha256": seed["sha256"],
            "scientific_config_sha256": scientific_sha256,
        },
    )
    arms = {}
    for mode in ARMS:
        treatment = make_manifest("treatment", _identity_treatment(mode))
        arms[mode] = make_manifest(
            "arm_protocol",
            {
                "cohort_protocol_sha256": cohort["sha256"],
                "mode": mode,
                "treatment_sha256": treatment["sha256"],
            },
        )["sha256"]
    return cohort["sha256"], arms


def _entry_identity(study_id: str, arm: str, stage: str) -> tuple[str, str]:
    return (
        f"{study_id}:{arm}:{stage}",
        f"{study_id}.{arm}.{stage}.final",
    )


def _entry_paths(arm: str, stage: str, terminal_step: int) -> tuple[str, str]:
    prefix = f"arms/{arm}/{stage}"
    return (
        f"{prefix}/step-{terminal_step}.ckpt",
        f"{prefix}/finalized-checkpoint.json",
    )


def validate_overlay(
    overlay: Mapping[str, Any], *, exact_root: str | os.PathLike[str]
) -> dict[str, Any]:
    """Return a canonical nine-job plan after strict pure validation."""
    overlay = _exact_keys(
        overlay,
        {
            "schema_version",
            "run_policy",
            "study_id",
            "artifact_root",
            "source",
            "smoke",
            "capacity",
            "runtime",
            "scheduler",
            "stages",
        },
        "formal overlay",
    )
    if (
        isinstance(overlay["schema_version"], bool)
        or not isinstance(overlay["schema_version"], int)
        or overlay["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("formal overlay schema_version mismatch")
    if overlay["run_policy"] != "fresh":
        raise ValueError("formal overlay is fresh-only")
    study_id = _safe_id(overlay["study_id"], "study_id")
    artifact_root = _absolute_path(overlay["artifact_root"], "artifact_root")
    root = Path(exact_root)
    if not root.is_absolute():
        raise ValueError("exact_root must be absolute")

    source = _exact_keys(
        overlay["source"],
        {
            "commit_sha",
            "tree_sha",
            "manifest_path",
            "manifest_sha256",
            "environment_sha256",
            "candidate_repository",
        },
        "source",
    )
    commit_sha = _git_oid(source["commit_sha"], "source commit")
    tree_sha = _git_oid(source["tree_sha"], "source tree")
    source_manifest = _absolute_path(source["manifest_path"], "source manifest")
    source_sha256 = _digest(source["manifest_sha256"], "source manifest")
    environment_sha256 = _digest(
        source["environment_sha256"], "environment sha256"
    )
    candidate_repository = _safe_text(
        source["candidate_repository"], "candidate_repository"
    )

    smoke = _exact_keys(
        overlay["smoke"],
        {"attestation_path", "expected_sha256", "expected_gpu_model"},
        "smoke",
    )
    smoke_path = _absolute_path(smoke["attestation_path"], "smoke attestation")
    smoke_sha256 = _digest(smoke["expected_sha256"], "smoke expected sha256")
    smoke_gpu_model = _safe_text(smoke["expected_gpu_model"], "smoke GPU model")
    if "H100" not in smoke_gpu_model:
        raise ValueError("smoke GPU model must identify an H100")

    capacity = _exact_keys(
        overlay["capacity"],
        {
            "checkpoint_ceiling_bytes",
            "durable_log_allowance_bytes",
            "run_log_ceiling_bytes",
            "attestation_ceiling_bytes",
            "manifest_ceiling_bytes",
            "protocol_metadata_allowance_bytes",
        },
        "capacity",
    )
    checkpoint_ceiling = _positive_int(
        capacity["checkpoint_ceiling_bytes"], "checkpoint ceiling"
    )
    durable_log_allowance = _positive_int(
        capacity["durable_log_allowance_bytes"], "durable log allowance"
    )
    run_log_ceiling = _positive_int(
        capacity["run_log_ceiling_bytes"], "run log ceiling"
    )
    attestation_ceiling = _positive_int(
        capacity["attestation_ceiling_bytes"], "attestation ceiling"
    )
    manifest_ceiling = _positive_int(
        capacity["manifest_ceiling_bytes"], "manifest ceiling"
    )
    protocol_metadata_allowance = _positive_int(
        capacity["protocol_metadata_allowance_bytes"],
        "protocol metadata allowance",
    )
    assigned_durable = (
        2 * len(ARMS) * len(STAGES) * run_log_ceiling
        + 2 * len(ARMS) * len(STAGES) * attestation_ceiling
        + len(ARMS) * len(STAGES) * manifest_ceiling
        + protocol_metadata_allowance
    )
    if assigned_durable > durable_log_allowance:
        raise ValueError("formal durable-output sub-budgets exceed aggregate allowance")

    runtime = _exact_keys(
        overlay["runtime"],
        {"python", "git", "nvidia_smi", "job_work_root"},
        "runtime",
    )
    python = _absolute_path(runtime["python"], "runtime python")
    git = _absolute_path(runtime["git"], "runtime git")
    nvidia_smi = _absolute_path(runtime["nvidia_smi"], "runtime nvidia_smi")
    job_work_root = _absolute_path(runtime["job_work_root"], "job work root")

    scheduler = _exact_keys(
        overlay["scheduler"],
        {
            "partition",
            "qos",
            "cpus_per_task",
            "memory_mb",
            "gpus_per_job",
        },
        "scheduler",
    )
    partition = _safe_text(scheduler["partition"], "scheduler partition")
    if partition.casefold() != "h100":
        raise ValueError("formal jobs require the dedicated H100 partition")
    qos = _safe_text(scheduler["qos"], "scheduler qos")
    cpus_per_task = _positive_int(
        scheduler["cpus_per_task"], "scheduler cpus_per_task"
    )
    memory_mb = _positive_int(scheduler["memory_mb"], "scheduler memory_mb")
    if (
        isinstance(scheduler["gpus_per_job"], bool)
        or not isinstance(scheduler["gpus_per_job"], int)
        or scheduler["gpus_per_job"] != 1
    ):
        raise ValueError("formal production jobs require exactly one H100")

    raw_stages = overlay["stages"]
    if not isinstance(raw_stages, list) or len(raw_stages) != len(STAGES):
        raise ValueError("stages must contain exactly stage1, stage2, and stage3")
    stages_by_name: dict[str, Mapping[str, Any]] = {}
    for stage_value in raw_stages:
        stage_value = _exact_keys(
            stage_value,
            {
                "stage",
                "terminal_step",
                "prior_sha256",
                "architecture_sha256",
                "optimizer_sha256",
                "scientific_sha256",
                "cohort_protocol_sha256",
                "arm_protocol_sha256",
            },
            "stage protocol",
        )
        stage_name = stage_value["stage"]
        if stage_name in stages_by_name:
            raise ValueError("stages contain a duplicate stage")
        stages_by_name[stage_name] = stage_value
    if set(stages_by_name) != {stage for stage, _ in STAGES}:
        raise ValueError("stages must be the exact canonical stage set")

    entries: list[dict[str, Any]] = []
    stage_protocols: dict[str, dict[str, Any]] = {}
    for stage_name, budget in STAGES:
        stage_value = stages_by_name[stage_name]
        if stage_value["terminal_step"] != budget:
            raise ValueError(f"{stage_name} budget must be exactly {budget}")
        prior = _digest(stage_value["prior_sha256"], f"{stage_name} prior")
        architecture = _digest(
            stage_value["architecture_sha256"], f"{stage_name} architecture"
        )
        optimizer = _digest(
            stage_value["optimizer_sha256"], f"{stage_name} optimizer"
        )
        scientific = _digest(
            stage_value["scientific_sha256"], f"{stage_name} scientific"
        )
        supplied_cohort = _digest(
            stage_value["cohort_protocol_sha256"],
            f"{stage_name} cohort_protocol",
        )
        arm_protocol = _exact_keys(
            stage_value["arm_protocol_sha256"], set(ARMS), f"{stage_name} arms"
        )
        expected_cohort, expected_arms = _expected_protocols(
            stage=stage_name,
            terminal_step=budget,
            source_sha256=source_sha256,
            environment_sha256=environment_sha256,
            prior_sha256=prior,
            architecture_sha256=architecture,
            optimizer_sha256=optimizer,
            scientific_sha256=scientific,
        )
        if supplied_cohort != expected_cohort:
            raise ValueError(
                f"{stage_name} cohort_protocol_sha256 is not the canonical shared protocol"
            )
        supplied_arms = {
            arm: _digest(arm_protocol[arm], f"{stage_name} {arm} arm_protocol")
            for arm in ARMS
        }
        if supplied_arms != expected_arms:
            raise ValueError(
                f"{stage_name} arm_protocol_sha256 is not derived solely from "
                "the cohort and canonical identity treatment"
            )
        stage_protocols[stage_name] = {
            "terminal_step": budget,
            "prior_sha256": prior,
            "architecture_sha256": architecture,
            "optimizer_sha256": optimizer,
            "scientific_sha256": scientific,
            "cohort_protocol_sha256": supplied_cohort,
            "arm_protocol_sha256": supplied_arms,
        }

    for arm in ARMS:
        for stage_name, budget in STAGES:
            upstream, artifact = _entry_identity(study_id, arm, stage_name)
            checkpoint_relpath, final_relpath = _entry_paths(
                arm, stage_name, budget
            )
            protocol = stage_protocols[stage_name]
            entries.append(
                {
                    "arm": arm,
                    "stage": stage_name,
                    "terminal_step": budget,
                    "upstream_identity": _safe_id(upstream, "upstream_identity"),
                    "artifact_identity": _safe_id(artifact, "artifact_identity"),
                    "checkpoint_relpath": _relative_path(
                        checkpoint_relpath, "checkpoint_relpath"
                    ),
                    "finalized_manifest_relpath": _relative_path(
                        final_relpath, "finalized_manifest_relpath"
                    ),
                    "np_seed": SEED,
                    "torch_seed": SEED,
                    "identity_rng_seed": SEED,
                    "world_size": WORLD_SIZE,
                    "cuda_device_count": CUDA_DEVICE_COUNT,
                    "max_checkpoint_bytes": checkpoint_ceiling,
                    "source_sha256": source_sha256,
                    "environment_sha256": environment_sha256,
                    "prior_sha256": protocol["prior_sha256"],
                    "architecture_sha256": protocol["architecture_sha256"],
                    "optimizer_sha256": protocol["optimizer_sha256"],
                    "scientific_sha256": protocol["scientific_sha256"],
                    "cohort_protocol_sha256": protocol[
                        "cohort_protocol_sha256"
                    ],
                    "arm_protocol_sha256": protocol["arm_protocol_sha256"][arm],
                }
            )
    if any(set(entry) != _LEDGER_ENTRY_KEYS for entry in entries):
        raise AssertionError("internal ledger schema drift")
    checkpoint_paths = {entry["checkpoint_relpath"] for entry in entries}
    final_paths = {entry["finalized_manifest_relpath"] for entry in entries}
    if len(checkpoint_paths) != 9 or len(final_paths) != 9:
        raise ValueError("formal artifact paths must be unique")

    ledger = make_manifest(
        "transaction_ledger", {"study_id": study_id, "entries": entries}
    )
    ledger_path = artifact_root / "transaction-ledger.json"
    entries_by_key = {(entry["arm"], entry["stage"]): entry for entry in entries}
    jobs = []
    for arm in ARMS:
        predecessor: str | None = None
        for stage_index, (stage_name, budget) in enumerate(STAGES, start=1):
            entry = entries_by_key[(arm, stage_name)]
            checkpoint_dir = artifact_root / "arms" / arm / stage_name
            exports = {
                "RUN_POLICY": "fresh",
                "MODE": arm,
                "STAGE": str(stage_index),
                "NUM_GPUS": "1",
                "PYTHON": str(python),
                "GIT": str(git),
                "NVIDIA_SMI": str(nvidia_smi),
                "PYTHONPATH": str(root / "src"),
                "PYTHONNOUSERSITE": "1",
                "CANDIDATE_REPOSITORY": candidate_repository,
                "FORMAL_SOURCE_COMMIT_SHA": commit_sha,
                "FORMAL_SOURCE_TREE_SHA": tree_sha,
                "FORMAL_JOB_WORK_ROOT": str(job_work_root),
                "FORMAL_ARTIFACT_ROOT": str(artifact_root),
                "FORMAL_CHECKPOINT_DIR": str(checkpoint_dir),
                "FORMAL_SOURCE_MANIFEST": str(source_manifest),
                "FORMAL_SOURCE_SHA256": source_sha256,
                "FORMAL_ENVIRONMENT_SHA256": environment_sha256,
                "FORMAL_STUDY_ID": study_id,
                "FORMAL_OUTPUT_ID": f"{study_id}-{arm}-{stage_name}",
                "FORMAL_PRIOR_SHA256": entry["prior_sha256"],
                "FORMAL_ARCHITECTURE_SHA256": entry["architecture_sha256"],
                "FORMAL_OPTIMIZER_SHA256": entry["optimizer_sha256"],
                "FORMAL_SCIENTIFIC_SHA256": entry["scientific_sha256"],
                "FORMAL_COHORT_PROTOCOL_SHA256": entry[
                    "cohort_protocol_sha256"
                ],
                "FORMAL_ARM_PROTOCOL_SHA256": entry["arm_protocol_sha256"],
                "FORMAL_TRANSACTION_LEDGER": str(ledger_path),
                "FORMAL_TRANSACTION_LEDGER_SHA256": ledger["sha256"],
                "FORMAL_UPSTREAM_IDENTITY": entry["upstream_identity"],
                "FORMAL_ARTIFACT_IDENTITY": entry["artifact_identity"],
                "CHECKPOINT_CEILING_BYTES": str(checkpoint_ceiling),
                "DURABLE_LOG_ALLOWANCE_BYTES": str(durable_log_allowance),
                "FORMAL_RUN_LOG_CEILING_BYTES": str(run_log_ceiling),
                "FORMAL_ATTESTATION_CEILING_BYTES": str(attestation_ceiling),
                "FORMAL_MANIFEST_CEILING_BYTES": str(manifest_ceiling),
                "FORMAL_PROTOCOL_METADATA_ALLOWANCE_BYTES": str(
                    protocol_metadata_allowance
                ),
            }
            parent_job = None
            if predecessor is not None:
                parent = entries_by_key[(arm, predecessor)]
                exports.update(
                    {
                        "FORMAL_PARENT_CHECKPOINT": str(
                            artifact_root / parent["checkpoint_relpath"]
                        ),
                        "FORMAL_PARENT_FINALIZED_MANIFEST": str(
                            artifact_root / parent["finalized_manifest_relpath"]
                        ),
                        "FORMAL_PARENT_UPSTREAM_IDENTITY": parent[
                            "upstream_identity"
                        ],
                        "FORMAL_PARENT_ARTIFACT_IDENTITY": parent[
                            "artifact_identity"
                        ],
                    }
                )
                parent_job = (arm, predecessor)
            if any(
                not isinstance(value, str)
                or not value
                or any(c in value for c in ("\x00", "\n", "\r", ","))
                for value in exports.values()
            ):
                raise ValueError("job export values must be non-empty and comma-free")
            jobs.append(
                {
                    "arm": arm,
                    "stage": stage_name,
                    "stage_index": stage_index,
                    "terminal_step": budget,
                    "gpus": 1,
                    "parent_job": parent_job,
                    "exports": exports,
                }
            )
            predecessor = stage_name

    plan = {
        "schema_version": SCHEMA_VERSION,
        "run_policy": "fresh",
        "study_id": study_id,
        "artifact_root": str(artifact_root),
        "ledger_path": str(ledger_path),
        "rollback_path": str(artifact_root / "rollback-incomplete.json"),
        "submission_receipt_path": str(
            artifact_root / "submission-receipt.json"
        ),
        "exact_root": str(root),
        "source": {
            "commit_sha": commit_sha,
            "tree_sha": tree_sha,
            "manifest_path": str(source_manifest),
            "manifest_sha256": source_sha256,
            "environment_sha256": environment_sha256,
        },
        "smoke": {
            "attestation_path": str(smoke_path),
            "expected_sha256": smoke_sha256,
            "expected_gpu_model": smoke_gpu_model,
        },
        "capacity": {
            "checkpoint_ceiling_bytes": checkpoint_ceiling,
            "durable_log_allowance_bytes": durable_log_allowance,
            "run_log_ceiling_bytes": run_log_ceiling,
            "attestation_ceiling_bytes": attestation_ceiling,
            "manifest_ceiling_bytes": manifest_ceiling,
            "protocol_metadata_allowance_bytes": protocol_metadata_allowance,
        },
        "runtime": {
            "python": str(python),
            "git": str(git),
            "nvidia_smi": str(nvidia_smi),
            "job_work_root": str(job_work_root),
        },
        "scheduler": {
            "partition": partition,
            "qos": qos,
            "cpus_per_task": cpus_per_task,
            "memory_mb": memory_mb,
            "gpus_per_job": 1,
        },
        "ledger": ledger,
        "jobs": jobs,
    }
    _validate_protocol_metadata_budget(plan)
    return plan


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("short write")
        offset += written


def _publish_no_replace(
    path: Path, payload: bytes, *, fault: str | None = None
) -> Path:
    if fault not in {None, "write", "fsync", "rename", "dir_fsync"}:
        raise ValueError("unknown publication fault")
    parent = path.parent
    dir_fd = _open_directory_nofollow(parent, where="publication parent")
    temp_name = f".{path.name}.{secrets.token_hex(16)}.tmp"
    file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        file_flags |= os.O_CLOEXEC
    fd = -1
    published = False
    try:
        fd = os.open(temp_name, file_flags, 0o444, dir_fd=dir_fd)
        os.fchmod(fd, 0o444)
        if fault == "write":
            raise OSError("injected ledger write failure")
        _write_all(fd, payload)
        if fault == "fsync":
            raise OSError("injected ledger fsync failure")
        os.fsync(fd)
        os.close(fd)
        fd = -1
        if fault == "rename":
            raise OSError("injected ledger rename failure")
        os.link(
            temp_name,
            path.name,
            src_dir_fd=dir_fd,
            dst_dir_fd=dir_fd,
            follow_symlinks=False,
        )
        published = True
        if fault == "dir_fsync":
            raise OSError("injected ledger directory fsync failure")
        os.fsync(dir_fd)
        os.unlink(temp_name, dir_fd=dir_fd)
        os.fsync(dir_fd)
        return path
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temp_name, dir_fd=dir_fd)
        except FileNotFoundError:
            pass
        finally:
            # If publication succeeded and the directory fsync failed, the
            # caller still receives failure and must roll jobs back.  Never
            # remove the published, complete, write-once final.
            _ = published
            os.close(dir_fd)


def publish_ledger(plan: Mapping[str, Any], *, fault: str | None = None) -> Path:
    path = Path(plan["ledger_path"])
    return _publish_no_replace(
        path, canonical_json_bytes(plan["ledger"]) + b"\n", fault=fault
    )


def _job_ids(values: Sequence[str], where: str) -> list[str]:
    result = list(values)
    if any(not isinstance(value, str) or _JOB_ID.fullmatch(value) is None for value in result):
        raise ValueError(f"{where} contains an invalid job ID")
    if len(result) != len(set(result)):
        raise ValueError(f"{where} contains duplicate job IDs")
    return result


def _parse_sbatch_result(raw: str) -> str:
    """Normalize documented ``--parsable`` JOBID[;CLUSTER] output."""

    match = _SBATCH_RESULT.fullmatch(raw)
    if match is None:
        raise ValueError("sbatch returned an empty or malformed job ID")
    return match.group(1)


def _submission_receipt(
    plan: Mapping[str, Any], submitted: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    job_ids = [item["job_id"] for item in submitted]
    _job_ids(job_ids, "submission receipt job_ids")
    jobs = []
    for item in submitted:
        job = item["job"]
        jobs.append(
            {
                "arm": job["arm"],
                "stage": job["stage"],
                "terminal_step": job["terminal_step"],
                "job_id": item["job_id"],
                "parent_job_id": item["parent_job_id"],
            }
        )
    return make_manifest(
        "held_submission_receipt",
        {
            "study_id": plan["study_id"],
            "transaction_ledger_sha256": plan["ledger"]["sha256"],
            "source_commit_sha": plan["source"]["commit_sha"],
            "source_tree_sha": plan["source"]["tree_sha"],
            "jobs_held_at_publication": True,
            "job_ids": job_ids,
            "jobs": jobs,
        },
    )


def _rollback_record(
    plan: Mapping[str, Any],
    *,
    accepted: Sequence[str],
    cancelled: Sequence[str],
    remaining: Sequence[str],
    reason: str,
    ledger_published: bool,
) -> dict[str, Any]:
    return make_manifest(
        "rollback_incomplete",
        {
            "study_id": plan["study_id"],
            "reason": _safe_id(reason, "rollback reason"),
            "accepted_job_ids": list(accepted),
            "cancelled_job_ids": list(cancelled),
            "remaining_job_ids": list(remaining),
            "ledger_published": bool(ledger_published),
        },
    )


def _validate_protocol_metadata_budget(plan: Mapping[str, Any]) -> None:
    # Slurm's public job identifier is bounded to twenty decimal digits here;
    # construct the largest durable success and recovery records up front so
    # the overlay's L partition covers every controller-produced metadata byte.
    maximal_ids = [str(10_000_000_000_000_000_000 + index) for index in range(9)]
    ids_by_key: dict[tuple[str, str], str] = {}
    submitted = []
    for job, job_id in zip(plan["jobs"], maximal_ids):
        parent_key = job["parent_job"]
        parent_id = ids_by_key[parent_key] if parent_key is not None else None
        ids_by_key[(job["arm"], job["stage"])] = job_id
        submitted.append(
            {"job": job, "job_id": job_id, "parent_job_id": parent_id}
        )
    receipt = _submission_receipt(plan, submitted)
    recovery = max(
        (
            _rollback_record(
                plan,
                accepted=maximal_ids,
                cancelled=(),
                remaining=tuple(reversed(maximal_ids)),
                reason=reason,
                ledger_published=True,
            )
            for reason in (
                "sbatch_failed",
                "ledger_publication_failed",
                "receipt_publication_failed",
                "release_failed",
            )
        ),
        key=lambda value: len(canonical_json_bytes(value)),
    )
    required = sum(
        len(canonical_json_bytes(value)) + 1
        for value in (plan["ledger"], receipt, recovery)
    )
    allowance = plan["capacity"]["protocol_metadata_allowance_bytes"]
    if required > allowance:
        raise ValueError(
            "protocol metadata allowance is smaller than the maximum durable "
            f"controller metadata ({required} bytes)"
        )


def publish_rollback_incomplete(
    plan: Mapping[str, Any],
    *,
    accepted_job_ids: Sequence[str],
    cancelled_job_ids: Sequence[str],
    remaining_job_ids: Sequence[str],
    reason: str,
    ledger_published: bool,
) -> Path:
    accepted = _job_ids(accepted_job_ids, "accepted_job_ids")
    cancelled = _job_ids(cancelled_job_ids, "cancelled_job_ids")
    remaining = _job_ids(remaining_job_ids, "remaining_job_ids")
    if set(cancelled) | set(remaining) != set(accepted):
        raise ValueError("cancelled and remaining IDs must partition accepted IDs")
    if set(cancelled) & set(remaining):
        raise ValueError("cancelled and remaining IDs overlap")
    record = _rollback_record(
        plan,
        accepted=accepted,
        cancelled=cancelled,
        remaining=remaining,
        reason=reason,
        ledger_published=ledger_published,
    )
    return _publish_no_replace(
        Path(plan["rollback_path"]), canonical_json_bytes(record) + b"\n"
    )


def _publish_submission_receipt(
    plan: Mapping[str, Any], submitted: Sequence[Mapping[str, Any]]
) -> tuple[Path, dict[str, Any]]:
    receipt = _submission_receipt(plan, submitted)
    path = _publish_no_replace(
        Path(plan["submission_receipt_path"]),
        canonical_json_bytes(receipt) + b"\n",
    )
    return path, receipt


def _minimal_env(*, path: str, pythonpath: str | None = None) -> dict[str, str]:
    environment = {"PATH": path, "LC_ALL": "C", "LANG": "C"}
    if pythonpath is not None:
        environment.update(
            {"PYTHONPATH": pythonpath, "PYTHONNOUSERSITE": "1"}
        )
    return environment


def _run(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    where: str,
    allowed_returncodes: set[int] = {0},
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(argv),
        env=dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode not in allowed_returncodes:
        raise RuntimeError(f"{where} failed with status {completed.returncode}")
    return completed


def _attest_exact_checkout(plan: Mapping[str, Any]) -> None:
    root = Path(plan["exact_root"])
    physical_root = root.resolve(strict=True)
    if physical_root != root or not physical_root.is_dir():
        raise ValueError("exact-T root must be an absolute physical directory")
    _require_physical_directory(root, where="exact-T root")
    expected_helper = root / "scripts" / "verify_formal_overlay.py"
    if Path(__file__).resolve(strict=True) != expected_helper.resolve(strict=True):
        raise ValueError("formal overlay verifier is not executing from exact T")

    python = Path(plan["runtime"]["python"])
    if not python.is_absolute() or not os.access(python, os.X_OK):
        raise ValueError("formal runtime Python is not executable")
    if Path(sys.executable).resolve(strict=True) != python.resolve(strict=True):
        raise ValueError("formal controller Python differs from overlay runtime")
    git = Path(plan["runtime"]["git"])
    if not git.is_absolute() or not os.access(git, os.X_OK):
        raise ValueError("formal runtime Git is not executable")
    nvidia_smi = Path(plan["runtime"]["nvidia_smi"])
    if not nvidia_smi.is_absolute() or not os.access(nvidia_smi, os.X_OK):
        raise ValueError("formal runtime NVIDIA_SMI is not executable")
    _require_physical_directory(
        Path(plan["runtime"]["job_work_root"]), where="formal job work root"
    )

    path = os.environ.get("PATH", "")
    git_env = _minimal_env(path=path)
    head = _run(
        [str(git), "-C", str(root), "rev-parse", "HEAD^{commit}"],
        env=git_env,
        where="exact-T HEAD attestation",
    ).stdout.strip()
    tree = _run(
        [str(git), "-C", str(root), "rev-parse", "HEAD^{tree}"],
        env=git_env,
        where="exact-T tree attestation",
    ).stdout.strip()
    if head != plan["source"]["commit_sha"]:
        raise ValueError("exact-T checkout commit mismatch")
    if tree != plan["source"]["tree_sha"]:
        raise ValueError("exact-T checkout tree mismatch")
    status = _run(
        [
            str(git),
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        env=git_env,
        where="exact-T clean-tree attestation",
    ).stdout
    if status:
        raise ValueError("exact-T checkout is dirty")
    symbolic = _run(
        [str(git), "-C", str(root), "symbolic-ref", "-q", "HEAD"],
        env=git_env,
        where="exact-T detached-HEAD attestation",
        allowed_returncodes={0, 1},
    )
    if symbolic.returncode == 0:
        raise ValueError("exact-T checkout must be detached")

    source = plan["source"]
    verifier = root / "scripts" / "verify_runtime_source.py"
    result = _run(
        [
            str(python),
            "-I",
            "-B",
            str(verifier),
            "--archive-root",
            str(root),
            "--source-manifest",
            source["manifest_path"],
            "--expected-manifest-sha256",
            source["manifest_sha256"],
            "--expected-commit-sha",
            source["commit_sha"],
            "--expected-tree-sha",
            source["tree_sha"],
        ],
        env=_minimal_env(path=path, pythonpath=str(root / "src")),
        where="exact-T source/import attestation",
    )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("exact-T source verifier returned invalid JSON") from error
    expected_report = {
        "schema_version": 1,
        "commit_sha": source["commit_sha"],
        "tree_sha": source["tree_sha"],
        "source_manifest_sha256": source["manifest_sha256"],
        "import_relative_path": "src/tabicl/__init__.py",
    }
    if report != expected_report:
        raise ValueError("exact-T source verifier returned unexpected evidence")


def _validate_h100_smoke(plan: Mapping[str, Any]) -> None:
    root = Path(plan["exact_root"])
    python = plan["runtime"]["python"]
    source = plan["source"]
    smoke = plan["smoke"]
    _run(
        [
            python,
            "-I",
            "-B",
            str(root / "scripts" / "run_h100_identity_validation.py"),
            "--validate-attestation",
            smoke["attestation_path"],
            "--expected-attestation-sha256",
            smoke["expected_sha256"],
            "--expected-commit-sha",
            source["commit_sha"],
            "--expected-tree-sha",
            source["tree_sha"],
            "--expected-environment-sha256",
            source["environment_sha256"],
            "--expected-source-manifest-sha256",
            source["manifest_sha256"],
            "--expected-gpu-model",
            smoke["expected_gpu_model"],
            "--expected-checkpoint-ceiling-bytes",
            str(plan["capacity"]["checkpoint_ceiling_bytes"]),
            "--max-input-bytes",
            str(plan["capacity"]["attestation_ceiling_bytes"]),
        ],
        env=_minimal_env(
            path=os.environ.get("PATH", ""), pythonpath=str(root / "src")
        ),
        where="external digest-bound H100 smoke attestation",
    )


def _run_submit_capacity_gate(plan: Mapping[str, Any]) -> None:
    root = Path(plan["exact_root"])
    artifact_root = Path(plan["artifact_root"])
    parent = artifact_root.parent
    _require_physical_directory(parent, where="artifact namespace parent")
    if artifact_root.exists() or artifact_root.is_symlink():
        raise FileExistsError("fresh study namespace already exists")
    capacity = plan["capacity"]
    _run(
        [
            plan["runtime"]["python"],
            "-I",
            "-B",
            str(root / "scripts" / "check_formal_capacity.py"),
            "--artifact-root",
            str(parent),
            "--checkpoint-ceiling-bytes",
            str(capacity["checkpoint_ceiling_bytes"]),
            "--durable-log-allowance-bytes",
            str(capacity["durable_log_allowance_bytes"]),
            "--remaining-checkpoints",
            "15",
            "--run-log-ceiling-bytes",
            str(capacity["run_log_ceiling_bytes"]),
            "--attestation-ceiling-bytes",
            str(capacity["attestation_ceiling_bytes"]),
            "--manifest-ceiling-bytes",
            str(capacity["manifest_ceiling_bytes"]),
            "--protocol-metadata-allowance-bytes",
            str(capacity["protocol_metadata_allowance_bytes"]),
        ],
        env=_minimal_env(
            path=os.environ.get("PATH", ""), pythonpath=str(root / "src")
        ),
        where="fresh-cohort capacity gate",
    )


def _create_fresh_namespace(plan: Mapping[str, Any]) -> None:
    root = Path(plan["artifact_root"])
    parent = root.parent
    parent_fd = _open_directory_nofollow(
        parent, where="artifact namespace parent"
    )
    try:
        os.mkdir(root.name, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _scheduler_commands() -> tuple[str, str, str, dict[str, str]]:
    path = os.environ.get("PATH", "")
    if not path:
        raise ValueError("scheduler PATH must be explicit")
    path_parts = path.split(os.pathsep)
    for index, raw in enumerate(path_parts):
        component = Path(raw)
        if not raw or not component.is_absolute():
            raise ValueError("scheduler PATH components must be explicit absolute paths")
        _require_physical_directory(
            component, where=f"scheduler PATH component {index}"
        )
    commands = []
    for name in ("sbatch", "scontrol", "scancel"):
        candidate = shutil.which(name, path=path)
        if candidate is None or not os.path.isabs(candidate) or not os.access(candidate, os.X_OK):
            raise ValueError(f"required scheduler command is unavailable: {name}")
        commands.append(candidate)
    return commands[0], commands[1], commands[2], _minimal_env(path=path)


def _export_argument(exports: Mapping[str, str]) -> str:
    if "ALL" in exports or "PYTHONHOME" in exports:
        raise ValueError("formal job export allowlist is contaminated")
    items = []
    for key in sorted(exports):
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError("formal job export name is invalid")
        value = exports[key]
        if any(character in value for character in (",", "\x00", "\n", "\r")):
            raise ValueError("formal job export value is invalid")
        items.append(f"{key}={value}")
    return "--export=" + ",".join(items)


def _sbatch_argv(
    plan: Mapping[str, Any], job: Mapping[str, Any], parent_job_id: str | None
) -> list[str]:
    scheduler = plan["scheduler"]
    argv = [
        "--parsable",
        "--hold",
        f"--partition={scheduler['partition']}",
        f"--qos={scheduler['qos']}",
        "--nodes=1",
        "--gres=gpu:1",
        f"--cpus-per-task={scheduler['cpus_per_task']}",
        f"--mem={scheduler['memory_mb']}M",
        f"--job-name=tabicl-{job['arm']}-s{job['stage_index']}",
    ]
    if parent_job_id is not None:
        argv.extend(
            [
                f"--dependency=afterok:{parent_job_id}",
                "--kill-on-invalid-dep=yes",
            ]
        )
    argv.extend(
        [
            _export_argument(job["exports"]),
            str(Path(plan["exact_root"]) / "scripts/slurm_h100_identity_formal.sh"),
        ]
    )
    return argv


def _ledger_is_published(plan: Mapping[str, Any]) -> bool:
    path = Path(plan["ledger_path"])
    try:
        return path.read_bytes() == canonical_json_bytes(plan["ledger"]) + b"\n"
    except OSError:
        return False


def _rollback(
    *,
    scancel: str,
    scheduler_env: Mapping[str, str],
    accepted_job_ids: Sequence[str],
) -> tuple[list[str], list[str]]:
    cancelled: list[str] = []
    remaining: list[str] = []
    for job_id in reversed(accepted_job_ids):
        completed = subprocess.run(
            [scancel, job_id],
            env=dict(scheduler_env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            cancelled.append(job_id)
        else:
            remaining.append(job_id)
    return cancelled, remaining


def submit_overlay(
    overlay_path: Path,
    *,
    exact_root: Path,
    ledger_fault: str | None = None,
) -> dict[str, Any]:
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise ValueError("formal submit controller requires Python -I -B")
    plan = validate_overlay(_load_strict_json(overlay_path), exact_root=exact_root)
    artifact_root = Path(plan["artifact_root"])
    if artifact_root == exact_root or exact_root in artifact_root.parents:
        raise ValueError("formal artifact namespace must be outside exact T")
    _attest_exact_checkout(plan)
    _validate_h100_smoke(plan)
    _run_submit_capacity_gate(plan)
    sbatch, scontrol, scancel, scheduler_env = _scheduler_commands()
    _create_fresh_namespace(plan)

    accepted_ids: list[str] = []
    submitted: list[dict[str, Any]] = []
    job_ids_by_key: dict[tuple[str, str], str] = {}
    failure_phase = "sbatch_failed"
    try:
        for job in plan["jobs"]:
            parent_key = job["parent_job"]
            parent_id = job_ids_by_key[parent_key] if parent_key is not None else None
            completed = _run(
                [sbatch, *_sbatch_argv(plan, job, parent_id)],
                env=scheduler_env,
                where=f"held submission for {job['arm']} {job['stage']}",
            )
            job_id = _parse_sbatch_result(completed.stdout.strip())
            if job_id in accepted_ids:
                raise ValueError("sbatch returned a duplicate job ID")
            accepted_ids.append(job_id)
            job_ids_by_key[(job["arm"], job["stage"])] = job_id
            submitted.append(
                {"job": job, "job_id": job_id, "parent_job_id": parent_id}
            )

        failure_phase = "ledger_publication_failed"
        publish_ledger(plan, fault=ledger_fault)
        failure_phase = "receipt_publication_failed"
        _receipt_path, receipt = _publish_submission_receipt(plan, submitted)
        failure_phase = "release_failed"
        for job_id in accepted_ids:
            _run(
                [scontrol, "release", job_id],
                env=scheduler_env,
                where=f"release of held job {job_id}",
            )
        return receipt
    except Exception as error:
        cancelled, remaining = _rollback(
            scancel=scancel,
            scheduler_env=scheduler_env,
            accepted_job_ids=accepted_ids,
        )
        if remaining:
            try:
                recovery = publish_rollback_incomplete(
                    plan,
                    accepted_job_ids=accepted_ids,
                    cancelled_job_ids=cancelled,
                    remaining_job_ids=remaining,
                    reason=failure_phase,
                    ledger_published=_ledger_is_published(plan),
                )
            except Exception as recovery_error:
                raise RuntimeError(
                    "formal submission failed and rollback remains incomplete; "
                    f"remaining job IDs: {','.join(remaining)}; recovery record failed"
                ) from recovery_error
            raise RuntimeError(
                "formal submission failed and rollback remains incomplete; "
                f"recovery record: {recovery}"
            ) from error
        raise RuntimeError("formal submission failed; accepted jobs were rolled back") from error


def _load_strict_json(path: Path) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=pairs, parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(f"invalid number {raw}")))
    if not isinstance(value, Mapping):
        raise ValueError("overlay must contain a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-overlay", type=Path)
    group.add_argument("--submit-overlay", type=Path)
    parser.add_argument("--exact-root", required=True, type=Path)
    parser.add_argument(
        "--fault-ledger-stage", choices=("write", "fsync", "rename", "dir_fsync")
    )
    args = parser.parse_args(argv)
    if args.validate_overlay:
        if args.fault_ledger_stage:
            parser.error("--fault-ledger-stage is valid only with --submit-overlay")
        result = validate_overlay(
            _load_strict_json(args.validate_overlay), exact_root=args.exact_root
        )
    else:
        result = submit_overlay(
            args.submit_overlay,
            exact_root=args.exact_root,
            ledger_fault=args.fault_ledger_stage,
        )
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"formal overlay verification failed: {error}", file=sys.stderr)
        raise SystemExit(2)
