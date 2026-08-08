#!/usr/bin/env python3
"""Validate one released formal job, run its stage, and publish completion."""

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
import time
from typing import Any, Mapping


SCHEDULER_QUERY_TIMEOUT_SECONDS = 30
ARMS = ("rope", "temporary", "none")
STAGES = {"1": ("stage1", 500_000), "2": ("stage2", 40_000), "3": ("stage3", 10_000)}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
JOB_ID = re.compile(r"^[1-9][0-9]{0,19}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SLURM_DURATION = re.compile(
    r"^(?:(?P<days>[1-9][0-9]{0,3})-)?"
    r"(?P<hours>[0-9]{2}):(?P<minutes>[0-5][0-9]):(?P<seconds>[0-5][0-9])$"
)
RUNTIME_COMPLETION_CEILING_BYTES = 65_536
TERMINAL_LOG_ATTESTATION_CEILING_BYTES = 131_072
FORMAL_METADATA_CEILING_BYTES = 128 << 20
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
H100_GATE_KEYS = {
    "attestation_sha256",
    "checkpoint_ceiling_bytes",
    "gpu_model",
    "driver_version",
    "nvidia_smi_sha256",
}
CAMPAIGN_BINDING_KEYS = {
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


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value or any(character in value for character in ("\x00", "\n", "\r")):
        raise ValueError(f"{name} is required and must be one line")
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


def _load_repository_helper(exact_root: Path):
    path = exact_root / "scripts/verify_git_repository.py"
    spec = importlib.util.spec_from_file_location(
        "_formal_runtime_git_repository", path
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load exact-T Git repository helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bounded_int(name: str, *, minimum: int = 1, maximum: int = 1 << 40) -> int:
    raw = _required_env(name)
    if re.fullmatch(r"[0-9]+", raw) is None:
        raise ValueError(f"{name} must be decimal")
    value = int(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _stat_signature(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_stable_regular(
    path: Path,
    *,
    max_bytes: int,
    where: str,
    executable: bool = False,
    read_contents: bool = True,
) -> tuple[int, bytes, os.stat_result]:
    """Open one bounded file without following its final path component.

    The caller owns the returned descriptor.  Comparing the pre-open path,
    descriptor before/after I/O, and post-I/O path closes the lstat/read race
    while retaining the exact descriptor for execution when requested.
    """

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError(f"{where} byte ceiling is invalid")
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(f"{where} is unavailable") from error
    if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
        raise ValueError(f"{where} is not a bounded regular file")
    if executable and before.st_mode & 0o111 == 0:
        raise ValueError(f"{where} is not executable")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{where} could not be opened without following links") from error
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_size > max_bytes
            or _stat_signature(opened) != _stat_signature(before)
        ):
            raise ValueError(f"{where} changed while it was opened")
        if executable and opened.st_mode & 0o111 == 0:
            raise ValueError(f"{where} is not executable")
        chunks: list[bytes] = []
        total = 0
        if read_contents:
            while True:
                chunk = os.read(fd, min(1024 * 1024, max_bytes - total + 1))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"{where} exceeded its byte ceiling")
        after = os.fstat(fd)
        try:
            path_after = path.lstat()
        except OSError as error:
            raise ValueError(f"{where} path disappeared during verification") from error
        if (
            _stat_signature(after) != _stat_signature(opened)
            or _stat_signature(path_after) != _stat_signature(after)
        ):
            raise ValueError(f"{where} changed during verification")
        raw = b"".join(chunks)
        if read_contents and len(raw) != after.st_size:
            raise ValueError(f"{where} produced a short read")
        return fd, raw, after
    except BaseException:
        os.close(fd)
        raise


def _read_manifest(path: Path, *, max_bytes: int, kind: str) -> Mapping[str, Any]:
    fd, raw, _metadata = _open_stable_regular(
        path, max_bytes=max_bytes, where=kind
    )
    os.close(fd)
    if not raw.endswith(b"\n"):
        raise ValueError(f"{kind} changed or is not newline terminated")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{kind} contains duplicate keys")
            result[key] = value
        return result

    value = json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"invalid number {token}")
        ),
    )
    envelope = _exact(value, {"schema_version", "kind", "payload", "sha256"}, kind)
    if envelope["schema_version"] != 1 or envelope["kind"] != kind:
        raise ValueError(f"{kind} envelope mismatch")
    body = {key: envelope[key] for key in ("schema_version", "kind", "payload")}
    if envelope["sha256"] != _sha256(body) or raw != _canonical(envelope) + b"\n":
        raise ValueError(f"{kind} digest or canonical bytes mismatch")
    return envelope


def _wait_for_commit_manifest(
    path: Path, *, max_bytes: int, attempts: int = 300
) -> Mapping[str, Any]:
    """Wait at most thirty seconds for the controller's commit publication."""

    for attempt in range(attempts):
        if path.exists() or path.is_symlink():
            return _read_manifest(
                path, max_bytes=max_bytes, kind="formal_submission_commit"
            )
        if attempt + 1 < attempts:
            time.sleep(0.1)
    raise ValueError("formal submission commit marker is unavailable")


def _hash_bounded_regular(path: Path, maximum: int) -> tuple[int, str]:
    fd, raw, metadata = _open_stable_regular(
        path, max_bytes=maximum, where="artifact"
    )
    os.close(fd)
    return len(raw), hashlib.sha256(raw).hexdigest()


def _observe_bounded_regular_size(path: Path, maximum: int) -> int:
    fd, _raw, metadata = _open_stable_regular(
        path,
        max_bytes=maximum,
        where="scheduler spool log",
        read_contents=False,
    )
    os.close(fd)
    return metadata.st_size


def _trusted_executable(path: Path, expected_sha256: str) -> int:
    fd, raw, _metadata = _open_stable_regular(
        path,
        max_bytes=128 * 1024 * 1024,
        where="trusted scheduler query command",
        executable=True,
    )
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        os.close(fd)
        raise ValueError("trusted scheduler query command digest mismatch")
    return fd


def _is_exactly_one_gpu_tres(value: str) -> bool:
    items = value.split(",")
    gpu_items = [
        item
        for item in items
        if item.startswith("gres:gpu") or item.startswith("gres/gpu")
    ]
    return len(gpu_items) == 1 and re.fullmatch(
        r"gres(?::|/)gpu(?::[A-Za-z0-9_.-]+)?:1", gpu_items[0]
    ) is not None


def _scheduler_observation(job_id: str, expected_sha256: str) -> dict[str, Any]:
    command_fd = _trusted_executable(
        Path(_required_env("FORMAL_SCONTROL")), expected_sha256
    )
    try:
        completed = subprocess.run(
            [f"/proc/self/fd/{command_fd}", "show", "job", "-o", job_id],
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=SCHEDULER_QUERY_TIMEOUT_SECONDS,
            pass_fds=(command_fd,),
        )
    finally:
        os.close(command_fd)
    raw = completed.stdout.encode()
    if completed.returncode != 0 or not raw or len(raw) > 1_000_000:
        raise ValueError("trusted scheduler resource query failed")

    def field(name: str) -> str:
        match = re.search(rf"(?:^|[ ]){re.escape(name)}=([^\s]+)", completed.stdout)
        if match is None:
            raise ValueError(f"scheduler query omitted {name}")
        return match.group(1)

    memory = field("MinMemoryNode")
    if memory not in {"128G", "131072M"}:
        raise ValueError("scheduler query memory differs from the formal allocation")
    tres = field("TresPerNode")
    if not _is_exactly_one_gpu_tres(tres):
        raise ValueError("scheduler query GPU allocation differs from one GPU")
    observation = {
        "job_id": field("JobId"),
        "job_name": field("JobName"),
        "partition": field("Partition"),
        "qos": field("QOS"),
        "time_limit": _slurm_duration(field("TimeLimit"), "observed TimeLimit"),
        "nodes": int(field("NumNodes")),
        "cpus_per_task": int(field("CPUs/Task")),
        "memory_mb": 131_072,
        "gpus_per_job": 1,
        "query_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return observation


def _artifact_path(root: Path, relative: Any, *, where: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"{where} relative path is invalid")
    path = Path(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{where} relative path is unsafe")
    result = root.joinpath(*path.parts)
    if root not in result.parents:
        raise ValueError(f"{where} escaped artifact root")
    return result


def _validate_finalized_artifact(
    *,
    artifact_root: Path,
    study_id: str,
    entry_value: Mapping[str, Any],
    manifest_ceiling: int,
) -> dict[str, Any]:
    entry = _exact(entry_value, LEDGER_ENTRY_KEYS, "transaction ledger entry")
    finalized_path = _artifact_path(
        artifact_root,
        entry["finalized_manifest_relpath"],
        where="finalized manifest",
    )
    checkpoint_path = _artifact_path(
        artifact_root, entry["checkpoint_relpath"], where="checkpoint"
    )
    finalized = _read_manifest(
        finalized_path,
        max_bytes=manifest_ceiling,
        kind="finalized_checkpoint",
    )
    payload = _exact(
        finalized["payload"], FINALIZED_PAYLOAD_KEYS, "finalized checkpoint payload"
    )
    bindings = {
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
        "cuda_device_count": entry["cuda_device_count"],
        "max_checkpoint_bytes": entry["max_checkpoint_bytes"],
    }
    if any(payload[key] != value for key, value in bindings.items()):
        raise ValueError("finalized checkpoint differs from immutable ledger bindings")
    checkpoint_size, checkpoint_sha256 = _hash_bounded_regular(
        checkpoint_path, entry["max_checkpoint_bytes"]
    )
    if (
        payload["checkpoint_size"] != checkpoint_size
        or payload["checkpoint_sha256"] != checkpoint_sha256
    ):
        raise ValueError("finalized checkpoint digest or size mismatch")
    for digest_name in (
        "provenance_sha256",
        "seed_sha256",
        "treatment_sha256",
    ):
        if HEX64.fullmatch(payload[digest_name]) is None:
            raise ValueError(f"finalized checkpoint {digest_name} is invalid")
    return {
        "finalized_manifest_path": str(finalized_path),
        "finalized_manifest_sha256": finalized["sha256"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size": checkpoint_size,
        "provenance_sha256": payload["provenance_sha256"],
        "seed_sha256": payload["seed_sha256"],
        "treatment_sha256": payload["treatment_sha256"],
    }


def _publish_no_replace(
    path: Path, value: Mapping[str, Any], *, max_bytes: int
) -> None:
    raw = _canonical(value) + b"\n"
    if len(raw) > max_bytes:
        raise ValueError("runtime completion exceeds its byte ceiling")
    if path.exists() or path.is_symlink():
        raise FileExistsError("runtime completion is write-once")
    parent = path.parent
    metadata = parent.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("runtime completion parent must be a physical directory")
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
                raise OSError("short runtime completion write")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _validate_runtime(
    *, mode: str, stage_index: str, exact_root: Path
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    tuple[Mapping[str, Any], ...],
]:
    if mode not in ARMS or stage_index not in STAGES:
        raise ValueError("formal arm or stage is invalid")
    stage, terminal_step = STAGES[stage_index]
    metadata_ceiling = _bounded_int(
        "FORMAL_PROTOCOL_METADATA_ALLOWANCE_BYTES",
        maximum=FORMAL_METADATA_CEILING_BYTES,
    )
    receipt_path = Path(_required_env("FORMAL_SUBMISSION_RECEIPT"))
    ledger_path = Path(_required_env("FORMAL_TRANSACTION_LEDGER"))
    if not receipt_path.is_absolute() or not ledger_path.is_absolute():
        raise ValueError("formal receipt and ledger paths must be absolute")
    receipt = _read_manifest(
        receipt_path, max_bytes=metadata_ceiling, kind="held_submission_receipt"
    )
    ledger = _read_manifest(
        ledger_path, max_bytes=metadata_ceiling, kind="transaction_ledger"
    )
    expected_ledger = _required_env("FORMAL_TRANSACTION_LEDGER_SHA256")
    if HEX64.fullmatch(expected_ledger) is None or ledger["sha256"] != expected_ledger:
        raise ValueError("transaction ledger digest mismatch")
    receipt_payload = _exact(
        receipt["payload"],
        {
            "study_id",
            "seed",
            "transaction_id",
            "transaction_ledger_sha256",
            "source_commit_sha",
            "source_tree_sha",
            "repository_binding",
            "h100_gate",
            "campaign_binding",
            "runtime_tools",
            "jobs_held_at_publication",
            "run_log_ceiling_bytes",
            "manifest_ceiling_bytes",
            "protocol_metadata_allowance_bytes",
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
        receipt_payload["study_id"] != _required_env("FORMAL_STUDY_ID")
        or receipt_payload["seed"] != int(_required_env("FORMAL_SEED"))
        or receipt_payload["transaction_id"] != _required_env("FORMAL_TRANSACTION_ID")
        or receipt_payload["transaction_ledger_sha256"] != ledger["sha256"]
        or receipt_payload["source_commit_sha"]
        != _required_env("FORMAL_SOURCE_COMMIT_SHA")
        or receipt_payload["source_tree_sha"] != _required_env("FORMAL_SOURCE_TREE_SHA")
        or receipt_payload["protocol_metadata_allowance_bytes"] != metadata_ceiling
        or receipt_payload["jobs_held_at_publication"] is not True
    ):
        raise ValueError("submission receipt binding mismatch")
    artifact_root = Path(_required_env("FORMAL_ARTIFACT_ROOT"))
    commit_path = Path(receipt_payload["transaction_commit_path"])
    if (
        not artifact_root.is_absolute()
        or commit_path != artifact_root / "transaction-committed.json"
    ):
        raise ValueError("submission commit path is outside the formal namespace")
    rollback_path = artifact_root / (
        f"rollback-incomplete-{receipt_payload['transaction_id']}.json"
    )
    if rollback_path.exists() or rollback_path.is_symlink():
        raise ValueError("formal submission has a rollback recovery record")
    commit = _wait_for_commit_manifest(commit_path, max_bytes=metadata_ceiling)
    commit_payload = _exact(
        commit["payload"],
        {
            "study_id",
            "transaction_id",
            "submission_receipt_sha256",
            "transaction_ledger_sha256",
            "protocol_metadata_allowance_bytes",
            "h100_gate",
            "campaign_binding",
            "runtime_tools",
            "job_ids",
        },
        "formal submission commit payload",
    )
    if (
        commit_payload["study_id"] != receipt_payload["study_id"]
        or commit_payload["transaction_id"] != receipt_payload["transaction_id"]
        or commit_payload["submission_receipt_sha256"] != receipt["sha256"]
        or commit_payload["transaction_ledger_sha256"] != ledger["sha256"]
        or commit_payload["protocol_metadata_allowance_bytes"] != metadata_ceiling
        or commit_payload["h100_gate"] != receipt_payload["h100_gate"]
        or commit_payload["campaign_binding"]
        != receipt_payload["campaign_binding"]
        or commit_payload["runtime_tools"] != receipt_payload["runtime_tools"]
        or commit_payload["job_ids"] != receipt_payload["job_ids"]
    ):
        raise ValueError("formal submission commit binding mismatch")
    repository_binding = _load_repository_helper(
        exact_root
    ).validate_repository_binding(
        receipt_payload["repository_binding"],
        expected_commit_sha=_required_env("FORMAL_SOURCE_COMMIT_SHA"),
        expected_git_sha256=_required_env("FORMAL_GIT_SHA256"),
    )
    if (
        repository_binding["repository_url"]
        != _required_env("CANDIDATE_REPOSITORY")
        or repository_binding["repository_ref"]
        != _required_env("CANDIDATE_REPOSITORY_REF")
        or repository_binding["repository_identity_sha256"]
        != _required_env("FORMAL_REPOSITORY_IDENTITY_SHA256")
        or repository_binding["query_sha256"]
        != _required_env("FORMAL_REPOSITORY_QUERY_SHA256")
    ):
        raise ValueError("runtime repository exports differ from the receipt")
    scheduler = _exact(
        receipt_payload["scheduler"],
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
    expected_scheduler = {
        "partition": "h100",
        "qos": "long",
        "cpus_per_task": 64,
        "memory_mb": 131_072,
        "gpus_per_job": 1,
    }
    time_limit_by_stage = _exact(
        scheduler["time_limit_by_stage"],
        {stage_name for stage_name, _terminal_step in STAGES.values()},
        "receipt time_limit_by_stage",
    )
    normalized_time_limits = {
        stage_name: _slurm_duration(
            time_limit_by_stage[stage_name],
            f"receipt {stage_name} time limit",
        )
        for stage_name, _terminal_step in STAGES.values()
    }
    if (
        {key: scheduler[key] for key in expected_scheduler} != expected_scheduler
        or not isinstance(scheduler["sacct_path"], str)
        or not Path(scheduler["sacct_path"]).is_absolute()
        or HEX64.fullmatch(scheduler["sacct_sha256"]) is None
        or HEX64.fullmatch(scheduler["scontrol_sha256"]) is None
        or scheduler["scontrol_sha256"] != _required_env("FORMAL_SCONTROL_SHA256")
    ):
        raise ValueError("receipt scheduler resources are not canonical")
    manifest_ceiling = _bounded_int("FORMAL_MANIFEST_CEILING_BYTES")
    if (
        receipt_payload["run_log_ceiling_bytes"]
        != _bounded_int("FORMAL_RUN_LOG_CEILING_BYTES")
        or receipt_payload["manifest_ceiling_bytes"] != manifest_ceiling
        or receipt_payload["runtime_completion_ceiling_bytes"]
        != RUNTIME_COMPLETION_CEILING_BYTES
        or _bounded_int("FORMAL_RUNTIME_COMPLETION_CEILING_BYTES")
        != RUNTIME_COMPLETION_CEILING_BYTES
        or receipt_payload["terminal_log_attestation_ceiling_bytes"]
        != TERMINAL_LOG_ATTESTATION_CEILING_BYTES
        or not isinstance(receipt_payload["terminal_log_attestation_path"], str)
        or not Path(receipt_payload["terminal_log_attestation_path"]).is_absolute()
    ):
        raise ValueError("receipt durable evidence ceilings are not canonical")
    job_id = _required_env("SLURM_JOB_ID")
    if JOB_ID.fullmatch(job_id) is None:
        raise ValueError("SLURM_JOB_ID is invalid")
    job_name = _required_env("SLURM_JOB_NAME")
    if job_name != _required_env("FORMAL_EXPECTED_JOB_NAME"):
        raise ValueError("Slurm job name differs from the transaction identity")
    observation = _scheduler_observation(job_id, scheduler["scontrol_sha256"])
    observed_resources = {
        "partition": observation["partition"],
        "qos": observation["qos"],
        "cpus_per_task": observation["cpus_per_task"],
        "memory_mb": observation["memory_mb"],
        "gpus_per_job": observation["gpus_per_job"],
        "time_limit": observation["time_limit"],
    }
    expected_observed_resources = {
        **expected_scheduler,
        "time_limit": normalized_time_limits[stage],
    }
    if (
        observed_resources != expected_observed_resources
        or observation["job_id"] != job_id
        or observation["job_name"] != job_name
        or observation["nodes"] != 1
        or _required_env("SLURM_JOB_PARTITION") != "h100"
        or _bounded_int("SLURM_CPUS_PER_TASK") != 64
        or _required_env("SLURM_NNODES") != "1"
    ):
        raise ValueError("observed Slurm resources differ from the receipt")
    matching = [
        item
        for item in receipt_payload["jobs"]
        if isinstance(item, Mapping)
        and item.get("job_id") == job_id
        and item.get("arm") == mode
        and item.get("stage") == stage
    ]
    if len(matching) != 1:
        raise ValueError("released job is not uniquely present in the receipt")
    job = _exact(
        matching[0],
        {
            "arm",
            "stage",
            "terminal_step",
            "time_limit",
            "job_id",
            "cluster",
            "job_name",
            "parent_job_id",
            "sbatch_argv_sha256",
            "scheduler_stdout",
            "scheduler_stderr",
            "completion_path",
        },
        "receipt job",
    )
    if (
        job["terminal_step"] != terminal_step
        or job["time_limit"] != normalized_time_limits[stage]
        or job["time_limit"]
        != _slurm_duration(_required_env("FORMAL_TIME_LIMIT"), "FORMAL_TIME_LIMIT")
        or job["job_name"] != job_name
        or not isinstance(job["sbatch_argv_sha256"], str)
        or HEX64.fullmatch(job["sbatch_argv_sha256"]) is None
        or job["scheduler_stdout"] != _required_env("FORMAL_SCHEDULER_STDOUT")
        or job["scheduler_stderr"] != _required_env("FORMAL_SCHEDULER_STDERR")
        or job["completion_path"] != _required_env("FORMAL_COMPLETION_EVIDENCE")
    ):
        raise ValueError("receipt job path or identity mismatch")
    ledger_payload = _exact(
        ledger["payload"],
        {
            "study_id",
            "protocol_metadata_allowance_bytes",
            "entries",
            "h100_gate",
            "campaign_binding",
            "runtime_tools",
        },
        "ledger payload",
    )
    h100_gate = _exact(
        ledger_payload["h100_gate"], H100_GATE_KEYS, "ledger H100 gate"
    )
    campaign_binding = _exact(
        ledger_payload["campaign_binding"],
        CAMPAIGN_BINDING_KEYS,
        "ledger campaign binding",
    )
    receipt_h100_gate = _exact(
        receipt_payload["h100_gate"], H100_GATE_KEYS, "receipt H100 gate"
    )
    receipt_campaign_binding = _exact(
        receipt_payload["campaign_binding"],
        CAMPAIGN_BINDING_KEYS,
        "receipt campaign binding",
    )
    runtime_tools = _exact(
        ledger_payload["runtime_tools"],
        {"nvidia_smi_sha256"},
        "ledger runtime tools",
    )
    receipt_runtime_tools = _exact(
        receipt_payload["runtime_tools"],
        {"nvidia_smi_sha256"},
        "receipt runtime tools",
    )
    selected_seed = int(_required_env("FORMAL_SEED"))
    expected_predecessors = {
        str(candidate) for candidate in (42, 43, 44) if candidate < selected_seed
    }
    predecessor_map = _exact(
        campaign_binding["predecessor_acceptance_sha256_by_seed"],
        expected_predecessors,
        "campaign predecessor acceptances",
    )
    static_protocols = _exact(
        campaign_binding["static_protocol_sha256_by_stage"],
        {stage_name for stage_name, _terminal_step in STAGES.values()},
        "campaign static protocols",
    )
    campaign_times = _exact(
        campaign_binding["time_limit_by_stage"],
        {stage_name for stage_name, _terminal_step in STAGES.values()},
        "campaign time limits",
    )
    if (
        h100_gate != receipt_h100_gate
        or campaign_binding != receipt_campaign_binding
        or runtime_tools != receipt_runtime_tools
        or ledger_payload["protocol_metadata_allowance_bytes"] != metadata_ceiling
        or ledger_payload["protocol_metadata_allowance_bytes"]
        != receipt_payload["protocol_metadata_allowance_bytes"]
        or runtime_tools["nvidia_smi_sha256"]
        != _required_env("FORMAL_NVIDIA_SMI_SHA256")
        or h100_gate["nvidia_smi_sha256"]
        != runtime_tools["nvidia_smi_sha256"]
        or campaign_binding["nvidia_smi_sha256"]
        != runtime_tools["nvidia_smi_sha256"]
        or HEX64.fullmatch(runtime_tools["nvidia_smi_sha256"]) is None
        or not isinstance(h100_gate["attestation_sha256"], str)
        or HEX64.fullmatch(h100_gate["attestation_sha256"]) is None
        or h100_gate["attestation_sha256"]
        != _required_env("FORMAL_H100_ATTESTATION_SHA256")
        or h100_gate["gpu_model"] != _required_env("FORMAL_EXPECTED_GPU_MODEL")
        or h100_gate["driver_version"]
        != _required_env("FORMAL_EXPECTED_DRIVER_VERSION")
        or not isinstance(h100_gate["gpu_model"], str)
        or "H100" not in h100_gate["gpu_model"]
        or not isinstance(h100_gate["driver_version"], str)
        or not h100_gate["driver_version"]
        or any(
            character in h100_gate["driver_version"]
            for character in ("\x00", "\n", "\r", ",")
        )
        or not isinstance(campaign_binding["campaign_manifest_sha256"], str)
        or HEX64.fullmatch(campaign_binding["campaign_manifest_sha256"]) is None
        or not isinstance(campaign_binding["campaign_id"], str)
        or SAFE_ID.fullmatch(campaign_binding["campaign_id"]) is None
        or receipt_payload["study_id"]
        != f"{campaign_binding['campaign_id']}-seed{selected_seed}"
        or not isinstance(campaign_binding["training_commit_sha"], str)
        or HEX40.fullmatch(campaign_binding["training_commit_sha"]) is None
        or not isinstance(campaign_binding["training_tree_sha"], str)
        or HEX40.fullmatch(campaign_binding["training_tree_sha"]) is None
        or campaign_binding["training_commit_sha"]
        != _required_env("FORMAL_SOURCE_COMMIT_SHA")
        or campaign_binding["training_tree_sha"]
        != _required_env("FORMAL_SOURCE_TREE_SHA")
        or campaign_binding["source_manifest_sha256"]
        != _required_env("FORMAL_SOURCE_SHA256")
        or campaign_binding["environment_sha256"]
        != _required_env("FORMAL_ENVIRONMENT_SHA256")
        or campaign_binding["h100_attestation_sha256"]
        != h100_gate["attestation_sha256"]
        or campaign_binding["checkpoint_ceiling_bytes"]
        != h100_gate["checkpoint_ceiling_bytes"]
        or campaign_times != normalized_time_limits
        or _sha256(campaign_binding)
        != _required_env("FORMAL_CAMPAIGN_BINDING_SHA256")
        or any(
            not isinstance(value, str) or HEX64.fullmatch(value) is None
            for value in (*predecessor_map.values(), *static_protocols.values())
        )
    ):
        raise ValueError("immutable H100 gate or campaign binding mismatch")
    matching_entries = [
        item
        for item in ledger_payload["entries"]
        if isinstance(item, Mapping)
        and item.get("arm") == mode
        and item.get("stage") == stage
    ]
    if (
        ledger_payload["study_id"] != receipt_payload["study_id"]
        or len(matching_entries) != 1
        or matching_entries[0].get("np_seed") != receipt_payload["seed"]
        or matching_entries[0].get("torch_seed") != receipt_payload["seed"]
        or matching_entries[0].get("identity_rng_seed") != receipt_payload["seed"]
    ):
        raise ValueError("ledger job seed or identity mismatch")
    ledger_entry = _exact(
        matching_entries[0], LEDGER_ENTRY_KEYS, "transaction ledger entry"
    )
    checkpoint_ceiling = _bounded_int("CHECKPOINT_CEILING_BYTES")
    ledger_ceiling = ledger_entry["max_checkpoint_bytes"]
    if (
        isinstance(ledger_ceiling, bool)
        or not isinstance(ledger_ceiling, int)
        or ledger_ceiling < 1
        or ledger_ceiling != checkpoint_ceiling
        or h100_gate["checkpoint_ceiling_bytes"] != ledger_ceiling
        or campaign_binding["checkpoint_ceiling_bytes"] != ledger_ceiling
    ):
        raise ValueError(
            "runtime checkpoint ceiling differs from the immutable ledger"
        )
    if (
        _required_env("FORMAL_ENVIRONMENT_SHA256")
        != ledger_entry["environment_sha256"]
    ):
        raise ValueError("runtime environment digest differs from the immutable ledger")
    if Path(_required_env("TABICL_EXACT_ROOT")) != exact_root:
        raise ValueError("runtime exact root mismatch")
    if (
        _required_env("FORMAL_VISIBLE_GPU_NAME")
        != _required_env("FORMAL_EXPECTED_GPU_MODEL")
        or _required_env("FORMAL_VISIBLE_GPU_DRIVER_VERSION")
        != _required_env("FORMAL_EXPECTED_DRIVER_VERSION")
    ):
        raise ValueError("runtime GPU model/driver differs from the H100 gate")
    return (
        receipt,
        job,
        observed_resources | {"query_sha256": observation["query_sha256"]},
        ledger_entry,
        tuple(ledger_payload["entries"]),
    )


def _verify_formal_environment(exact_root: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(exact_root / "scripts" / "verify_formal_environment.py"),
            "--exact-root",
            str(exact_root),
            "--expected-sha256",
            _required_env("FORMAL_ENVIRONMENT_SHA256"),
            "--expected-gpus",
            "1",
        ],
        env=dict(os.environ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=300,
    )
    if completed.returncode != 0 or completed.stdout:
        raise ValueError("post-stage formal environment verification failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-root", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=ARMS)
    parser.add_argument("--stage", required=True, choices=tuple(STAGES))
    args = parser.parse_args(argv)
    exact_root = args.exact_root.resolve(strict=True)
    receipt, job, resources, ledger_entry, ledger_entries = _validate_runtime(
        mode=args.mode, stage_index=args.stage, exact_root=exact_root
    )
    manifest_ceiling = receipt["payload"]["manifest_ceiling_bytes"]
    stage_script = exact_root / "scripts" / f"formal_train_v2_clf_identity_stage{args.stage}.sh"
    if not stage_script.is_file() or not os.access(stage_script, os.X_OK):
        raise ValueError("exact-T formal stage wrapper is unavailable")
    completed = subprocess.run(
        [str(stage_script), args.mode],
        env=dict(os.environ),
        stdin=subprocess.DEVNULL,
        check=False,
    )
    _verify_formal_environment(exact_root)
    log_ceiling = _bounded_int("FORMAL_RUN_LOG_CEILING_BYTES")
    stdout_size = _observe_bounded_regular_size(
        Path(job["scheduler_stdout"]), log_ceiling
    )
    stderr_size = _observe_bounded_regular_size(
        Path(job["scheduler_stderr"]), log_ceiling
    )
    artifact_binding = None
    parent_lineage = None
    if completed.returncode == 0:
        artifact_root = Path(_required_env("FORMAL_ARTIFACT_ROOT"))
        artifact_binding = _validate_finalized_artifact(
            artifact_root=artifact_root,
            study_id=_required_env("FORMAL_STUDY_ID"),
            entry_value=ledger_entry,
            manifest_ceiling=manifest_ceiling,
        )
        if Path(artifact_binding["finalized_manifest_path"]).parent != Path(
            _required_env("FORMAL_CHECKPOINT_DIR")
        ):
            raise ValueError("finalized artifact is outside the stage checkpoint directory")
        parent_stage = {"stage1": None, "stage2": "stage1", "stage3": "stage2"}[
            STAGES[args.stage][0]
        ]
        if parent_stage is None:
            if job["parent_job_id"] is not None:
                raise ValueError("Stage 1 receipt unexpectedly names a parent job")
        else:
            if not isinstance(job["parent_job_id"], str):
                raise ValueError("descendant receipt omits its parent job ID")
            parent_jobs = [
                item
                for item in receipt["payload"]["jobs"]
                if isinstance(item, Mapping)
                and item.get("job_id") == job["parent_job_id"]
                and item.get("arm") == args.mode
                and item.get("stage") == parent_stage
            ]
            parent_entries = [
                item
                for item in ledger_entries
                if isinstance(item, Mapping)
                and item.get("arm") == args.mode
                and item.get("stage") == parent_stage
            ]
            if len(parent_jobs) != 1 or len(parent_entries) != 1:
                raise ValueError("descendant parent lineage is not unique")
            parent_lineage = _validate_finalized_artifact(
                artifact_root=artifact_root,
                study_id=_required_env("FORMAL_STUDY_ID"),
                entry_value=parent_entries[0],
                manifest_ceiling=manifest_ceiling,
            )
            if (
                parent_lineage["checkpoint_path"]
                != _required_env("FORMAL_PARENT_CHECKPOINT")
                or parent_lineage["finalized_manifest_path"]
                != _required_env("FORMAL_PARENT_FINALIZED_MANIFEST")
                or parent_entries[0]["upstream_identity"]
                != _required_env("FORMAL_PARENT_UPSTREAM_IDENTITY")
                or parent_entries[0]["artifact_identity"]
                != _required_env("FORMAL_PARENT_ARTIFACT_IDENTITY")
            ):
                raise ValueError("descendant parent lineage differs from frozen exports")
            parent_lineage = {
                **parent_lineage,
                "job_id": job["parent_job_id"],
                "stage": parent_stage,
            }
    payload = {
        "study_id": _required_env("FORMAL_STUDY_ID"),
        "transaction_id": _required_env("FORMAL_TRANSACTION_ID"),
        "submission_receipt_sha256": receipt["sha256"],
        "transaction_ledger_sha256": _required_env(
            "FORMAL_TRANSACTION_LEDGER_SHA256"
        ),
        "source_commit_sha": _required_env("FORMAL_SOURCE_COMMIT_SHA"),
        "source_tree_sha": _required_env("FORMAL_SOURCE_TREE_SHA"),
        "repository_binding": dict(receipt["payload"]["repository_binding"]),
        "job_id": _required_env("SLURM_JOB_ID"),
        "job_name": _required_env("SLURM_JOB_NAME"),
        "arm": args.mode,
        "stage": STAGES[args.stage][0],
        "seed": int(_required_env("FORMAL_SEED")),
        "scheduler": resources,
        "cuda_visible_devices": _required_env("CUDA_VISIBLE_DEVICES"),
        "gpu_name": _required_env("FORMAL_VISIBLE_GPU_NAME"),
        "gpu_uuid": _required_env("FORMAL_VISIBLE_GPU_UUID"),
        "gpu_driver_version": _required_env(
            "FORMAL_VISIBLE_GPU_DRIVER_VERSION"
        ),
        "scheduler_stdout_path": job["scheduler_stdout"],
        "scheduler_stderr_path": job["scheduler_stderr"],
        "scheduler_log_ceiling_bytes": log_ceiling,
        "manifest_ceiling_bytes": manifest_ceiling,
        "protocol_metadata_allowance_bytes": _bounded_int(
            "FORMAL_PROTOCOL_METADATA_ALLOWANCE_BYTES",
            maximum=FORMAL_METADATA_CEILING_BYTES,
        ),
        "scheduler_stdout_observed_size_at_completion": stdout_size,
        "scheduler_stderr_observed_size_at_completion": stderr_size,
        "scheduler_logs_terminal_verified": False,
        "finalized_artifact": artifact_binding,
        "parent_lineage": parent_lineage,
        "stage_exit_code": completed.returncode,
        "completed": completed.returncode == 0,
    }
    body = {"schema_version": 1, "kind": "formal_job_completion", "payload": payload}
    completion = {**body, "sha256": _sha256(body)}
    _publish_no_replace(
        Path(job["completion_path"]),
        completion,
        max_bytes=RUNTIME_COMPLETION_CEILING_BYTES,
    )
    return completed.returncode if completed.returncode != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
