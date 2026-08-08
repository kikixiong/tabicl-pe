#!/usr/bin/env python3
"""Canonical, dry-runnable H100 validation matrix and smoke attestation gate.

This file defines contracts locally but does not claim that any GPU gate has
passed.  A result becomes acceptable only when the complete payload matches an
externally supplied SHA-256 digest.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


NVIDIA_QUERY_TIMEOUT_SECONDS = 15
SCHEDULER_QUERY_TIMEOUT_SECONDS = 30
ARMS = ("rope", "temporary", "none")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_VALIDATION_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_RUNTIME_ENVIRONMENT_KEYS = {
    "python_version",
    "python_implementation",
    "python_executable_sha256",
    "python_cache_tag",
    "python_soabi",
    "platform_system",
    "platform_release",
    "platform_machine",
    "torch_version",
    "numpy_version",
    "cuda_runtime_version",
    "cudnn_version",
    "environment_fingerprint_schema_version",
    "installed_distributions_sha256",
    "formal_runtime_distributions",
    "unavailable_formal_runtime_distributions",
    "flash_attn3_available",
    "nccl_version",
    "visible_cuda_device_count",
}
_DISTRIBUTION_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DISTRIBUTION_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.!+_-]{0,255}$")
_FORMAL_RUNTIME_DISTRIBUTIONS = (
    "einops",
    "flash-attn-3",
    "huggingface-hub",
    "numpy",
    "psutil",
    "scikit-learn",
    "scipy",
    "threadpoolctl",
    "torch",
    "tqdm",
    "transformers",
    "wandb",
    "xgboost",
)
_FORMAL_RUNTIME_MODULES = {
    "einops": "einops",
    "flash-attn-3": "flash_attn_interface",
    "huggingface-hub": "huggingface_hub",
    "numpy": "numpy",
    "psutil": "psutil",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "threadpoolctl": "threadpoolctl",
    "torch": "torch",
    "tqdm": "tqdm",
    "transformers": "transformers",
    "wandb": "wandb",
    "xgboost": "xgboost",
}


@dataclass(frozen=True)
class ValidationCase:
    case_id: str
    category: str
    arm: str | None
    stage: str | None
    world_size: int
    smoke_kind: str
    min_seq_len: int | None
    max_seq_len: int | None
    observed_sequence_length: int | None
    log_seq_len: bool
    replay_small: bool
    recompute: bool
    utilization_required: bool
    assertion: str
    repeat_steps: int
    clean_detached_required: bool = True
    tabicl_attestation_count: int = 1
    python_no_user_site: str = "1"


def build_matrix() -> tuple[ValidationCase, ...]:
    cases: list[ValidationCase] = []
    for arm in ARMS:
        cases.append(
            ValidationCase(
                f"stage1_{arm}_one_step",
                "stage1_one_step",
                arm,
                "stage1",
                1,
                "functional_one_step",
                1024,
                1024,
                1024,
                False,
                False,
                False,
                False,
                "trainer_completed_one_step_at_observed_length",
                1,
            )
        )
    for arm in ARMS:
        cases.append(
            ValidationCase(
                f"stage2_{arm}_maxseq10240",
                "stage2_maxseq",
                arm,
                "stage2",
                1,
                "maxseq_utilization",
                10240,
                10240,
                10240,
                False,
                False,
                False,
                True,
                "trainer_observed_fixed_maxseq_and_gpu_window_passed",
                16,
            )
        )
    for arm in ARMS:
        cases.append(
            ValidationCase(
                f"stage3_{arm}_maxseq60000_recompute",
                "stage3_maxseq",
                arm,
                "stage3",
                1,
                "maxseq_utilization",
                60000,
                60000,
                60000,
                False,
                False,
                True,
                True,
                "trainer_observed_fixed_maxseq_and_recompute_and_gpu_window_passed",
                16,
            )
        )
    cases.extend(
        (
            ValidationCase(
                "temporary_cuda_rng_resume",
                "cuda_rng_resume",
                "temporary",
                None,
                1,
                "cuda_resume",
                None,
                None,
                None,
                False,
                False,
                False,
                False,
                "temporary_cuda_rng_isolation_and_resume_exact",
                1,
            ),
            ValidationCase(
                "nccl_2gpu",
                "nccl",
                None,
                None,
                2,
                "nccl_collective",
                None,
                None,
                None,
                False,
                False,
                False,
                False,
                "nccl_all_reduce_exact",
                1,
            ),
            ValidationCase(
                "prior_dataloader_resume",
                "prior_resume",
                None,
                None,
                1,
                "prior_resume",
                None,
                None,
                None,
                False,
                False,
                False,
                False,
                "full_prior_dataloader_uninterrupted_equals_resume",
                1,
            ),
        )
    )
    return tuple(cases)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def gate_requested_resource(case: ValidationCase) -> dict[str, Any]:
    if case.world_size == 1:
        cpus, wrapper = 32, "scripts/slurm_h100_identity_maxseq_smoke.sh"
    elif case.world_size == 2 and case.case_id == "nccl_2gpu":
        cpus, wrapper = 64, "scripts/slurm_h100_identity_nccl_smoke.sh"
    else:
        raise ValueError("invalid H100 gate world-size/resource composition")
    return {
        "partition": "h100",
        "qos": "short",
        "time_limit": "03:00:00",
        "nodes": 1,
        "gpus": case.world_size,
        "cpus_per_task": cpus,
        "memory_mb": 131072,
        "wrapper": wrapper,
    }


def gate_requested_resource_sha256(case: ValidationCase) -> str:
    return hashlib.sha256(_canonical(gate_requested_resource(case))).hexdigest()


def submission_artifact_identity(
    *,
    validation_id: str,
    case_id: str,
    world_size: int,
    commit_sha: str,
    tree_sha: str,
    source_manifest_sha256: str,
    expected_environment_sha256: str,
    requested_resource_sha256: str,
    checkpoint_ceiling_bytes: int,
) -> str:
    """Derive the external identity precommitted for one scheduled case."""

    cases = {case.case_id: case for case in build_matrix()}
    case = cases.get(case_id)
    if _SAFE_VALIDATION_ID.fullmatch(validation_id or "") is None:
        raise ValueError("validation ID is malformed")
    if case is None or case.world_size != world_size:
        raise ValueError("case/world-size identity input mismatch")
    if _HEX40.fullmatch(commit_sha or "") is None:
        raise ValueError("artifact identity commit is malformed")
    if _HEX40.fullmatch(tree_sha or "") is None:
        raise ValueError("artifact identity tree is malformed")
    _digest(source_manifest_sha256, "artifact identity source manifest")
    _digest(expected_environment_sha256, "artifact identity environment")
    _digest(requested_resource_sha256, "artifact identity requested resource")
    if (
        isinstance(checkpoint_ceiling_bytes, bool)
        or not isinstance(checkpoint_ceiling_bytes, int)
        or checkpoint_ceiling_bytes < 1
    ):
        raise ValueError("artifact identity checkpoint ceiling is invalid")
    value = {
        "schema_version": 1,
        "kind": "h100_gate_artifact_identity",
        "payload": {
            "validation_id": validation_id,
            "case_id": case_id,
            "world_size": world_size,
            "commit_sha": commit_sha,
            "tree_sha": tree_sha,
            "source_manifest_sha256": source_manifest_sha256,
            "expected_environment_sha256": expected_environment_sha256,
            "requested_resource_sha256": requested_resource_sha256,
            "checkpoint_ceiling_bytes": checkpoint_ceiling_bytes,
        },
    }
    return hashlib.sha256(_canonical(value)).hexdigest()


def make_smoke_attestation(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "kind": "h100_identity_smoke",
        "payload": dict(payload),
    }
    return {**body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def make_scheduler_log_manifest(
    files: Sequence[Mapping[str, Any]], *, ceiling_bytes: int
) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "kind": "h100_scheduler_log_manifest",
        "payload": {
            "ceiling_bytes": ceiling_bytes,
            "files": [dict(item) for item in files],
        },
    }
    return {**body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def validate_scheduler_log_manifest(value: Mapping[str, Any]) -> str:
    actual = _validate_manifest_envelope(
        value,
        expected_kind="h100_scheduler_log_manifest",
        where="scheduler log manifest",
    )
    payload = value["payload"]
    _exact_keys(payload, {"ceiling_bytes", "files"}, "scheduler log manifest payload")
    ceiling = payload["ceiling_bytes"]
    if isinstance(ceiling, bool) or not isinstance(ceiling, int) or ceiling < 1:
        raise ValueError("scheduler log manifest ceiling is invalid")
    expected_names = sorted(
        f"{case.case_id}.{suffix}"
        for case in build_matrix()
        for suffix in ("err", "out")
    )
    files = payload["files"]
    if not isinstance(files, list) or len(files) != len(expected_names):
        raise ValueError("scheduler log manifest file set is incomplete")
    observed_names: list[str] = []
    for item in files:
        _exact_keys(item, {"name", "sha256", "size"}, "scheduler log manifest file")
        name = item["name"]
        size = item["size"]
        if (
            not isinstance(name, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or size > ceiling
        ):
            raise ValueError("scheduler log manifest file metadata is invalid")
        _digest(item["sha256"], f"scheduler log {name} digest")
        observed_names.append(name)
    if observed_names != expected_names:
        raise ValueError("scheduler log manifest file set/order mismatch")
    return actual


def make_scheduler_terminal_manifest(
    observations: Sequence[Mapping[str, Any]],
    *,
    submission_receipt_sha256: str,
    sacct_sha256: str,
) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "kind": "h100_scheduler_terminal_manifest",
        "payload": {
            "submission_receipt_sha256": submission_receipt_sha256,
            "sacct_sha256": sacct_sha256,
            "observations": [dict(item) for item in observations],
        },
    }
    return {**body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def validate_scheduler_terminal_manifest(
    value: Mapping[str, Any],
    *,
    expected_submission_receipt_sha256: str | None = None,
    expected_sacct_sha256: str | None = None,
    expected_job_bindings: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    actual = _validate_manifest_envelope(
        value,
        expected_kind="h100_scheduler_terminal_manifest",
        where="scheduler terminal manifest",
    )
    payload = value["payload"]
    _exact_keys(
        payload,
        {"submission_receipt_sha256", "sacct_sha256", "observations"},
        "scheduler terminal manifest payload",
    )
    receipt_sha = _digest(
        payload["submission_receipt_sha256"], "terminal submission receipt"
    )
    sacct_sha = _digest(payload["sacct_sha256"], "terminal sacct executable")
    if (
        expected_submission_receipt_sha256 is not None
        and receipt_sha
        != _digest(
            expected_submission_receipt_sha256,
            "expected terminal submission receipt",
        )
    ):
        raise ValueError("scheduler terminal manifest receipt mismatch")
    if expected_sacct_sha256 is not None and sacct_sha != _digest(
        expected_sacct_sha256, "expected terminal sacct executable"
    ):
        raise ValueError("scheduler terminal manifest command mismatch")

    matrix = build_matrix()
    observations = payload["observations"]
    if not isinstance(observations, list) or len(observations) != len(matrix):
        raise ValueError("scheduler terminal observation set is incomplete")
    if expected_job_bindings is not None and set(expected_job_bindings) != {
        case.case_id for case in matrix
    }:
        raise ValueError("scheduler terminal job-binding set mismatch")
    job_ids: set[str] = set()
    for case, item in zip(matrix, observations):
        _exact_keys(
            item,
            {
                "case_id",
                "job_id",
                "receipt_cluster",
                "observed_cluster",
                "state",
                "exit_code",
                "derived_exit_code",
                "query_returncode",
                "query_argv_sha256",
                "query_stdout_sha256",
                "query_stderr_sha256",
            },
            "scheduler terminal observation",
        )
        job_id = item["job_id"]
        receipt_cluster = item["receipt_cluster"]
        observed_cluster = item["observed_cluster"]
        if (
            item["case_id"] != case.case_id
            or not isinstance(job_id, str)
            or re.fullmatch(r"[1-9][0-9]{0,19}", job_id) is None
            or job_id in job_ids
            or (
                receipt_cluster is not None
                and (
                    not isinstance(receipt_cluster, str)
                    or re.fullmatch(r"[A-Za-z0-9._-]+", receipt_cluster) is None
                )
            )
            or not isinstance(observed_cluster, str)
            or re.fullmatch(r"[A-Za-z0-9._-]+", observed_cluster) is None
            or (
                receipt_cluster is not None
                and observed_cluster != receipt_cluster
            )
            or item["state"] != "COMPLETED"
            or item["exit_code"] != "0:0"
            or item["derived_exit_code"] != "0:0"
            or item["query_returncode"] != 0
        ):
            raise ValueError("scheduler terminal observation is not exact COMPLETED/0:0")
        for field in (
            "query_argv_sha256",
            "query_stdout_sha256",
            "query_stderr_sha256",
        ):
            _digest(item[field], f"scheduler terminal {field}")
        expected_argv = [
            "sacct",
            *(
                [f"--clusters={receipt_cluster}"]
                if receipt_cluster is not None
                else []
            ),
            "--noheader",
            "--allocations",
            f"--jobs={job_id}",
            "--format=JobIDRaw,Cluster,State,ExitCode,DerivedExitCode",
            "--parsable2",
        ]
        expected_stdout = (
            f"{job_id}|{observed_cluster}|COMPLETED|0:0|0:0\n".encode()
        )
        if (
            item["query_argv_sha256"]
            != hashlib.sha256(_canonical(expected_argv)).hexdigest()
            or item["query_stdout_sha256"]
            != hashlib.sha256(expected_stdout).hexdigest()
            or item["query_stderr_sha256"] != hashlib.sha256(b"").hexdigest()
        ):
            raise ValueError("scheduler terminal query digests are not canonical")
        if expected_job_bindings is not None:
            binding = expected_job_bindings[case.case_id]
            if (
                binding.get("job_id") != job_id
                or binding.get("cluster") != receipt_cluster
            ):
                raise ValueError("scheduler terminal observation differs from receipt")
        job_ids.add(job_id)
    return actual


def make_source_attestation(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "kind": "exact_t_source_attestation",
        "payload": dict(payload),
    }
    return {**body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def validate_source_attestation(
    value: Mapping[str, Any],
    *,
    case_id: str,
    artifact_identity_sha256: str,
    commit_sha: str,
    tree_sha: str,
    source_manifest_sha256: str,
) -> str:
    _exact_keys(
        value,
        {"schema_version", "kind", "payload", "sha256"},
        "source attestation",
    )
    if value["schema_version"] != 1 or value["kind"] != "exact_t_source_attestation":
        raise ValueError("source attestation schema/kind mismatch")
    body = {key: value[key] for key in ("schema_version", "kind", "payload")}
    actual = hashlib.sha256(_canonical(body)).hexdigest()
    if value["sha256"] != actual:
        raise ValueError("source attestation self-hash mismatch")
    payload = value["payload"]
    _exact_keys(
        payload,
        {
            "commit_sha",
            "tree_sha",
            "source_manifest_sha256",
            "import_relative_path",
            "action",
            "case_id",
            "artifact_identity_sha256",
            "python_isolated",
            "bytecode_disabled",
            "python_no_user_site",
        },
        "source attestation payload",
    )
    expected = {
        "commit_sha": commit_sha,
        "tree_sha": tree_sha,
        "source_manifest_sha256": source_manifest_sha256,
        "import_relative_path": "src/tabicl/__init__.py",
        "action": "h100-validation",
        "case_id": case_id,
        "artifact_identity_sha256": artifact_identity_sha256,
        "python_isolated": True,
        "bytecode_disabled": True,
        "python_no_user_site": "1",
    }
    if payload != expected:
        raise ValueError("source attestation is not bound to exact case/source/artifact identity")
    return actual


def make_action_completion(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "kind": "exact_t_action_completion",
        "payload": dict(payload),
    }
    return {**body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def validate_action_completion(
    value: Mapping[str, Any],
    *,
    case_id: str,
    artifact_identity_sha256: str,
    source_attestation_sha256: str,
    runtime_evidence_sha256: str,
    commit_sha: str,
    tree_sha: str,
    source_manifest_sha256: str,
    scheduler_binding: Mapping[str, Any],
    expected_job_binding: Mapping[str, Any] | None = None,
) -> str:
    _exact_keys(
        value,
        {"schema_version", "kind", "payload", "sha256"},
        "action completion",
    )
    if value["schema_version"] != 1 or value["kind"] != "exact_t_action_completion":
        raise ValueError("action completion schema/kind mismatch")
    body = {key: value[key] for key in ("schema_version", "kind", "payload")}
    actual = hashlib.sha256(_canonical(body)).hexdigest()
    if value["sha256"] != actual:
        raise ValueError("action completion self-hash mismatch")
    expected = {
        "case_id": case_id,
        "action": "h100-validation",
        "artifact_identity_sha256": artifact_identity_sha256,
        "source_attestation_sha256": source_attestation_sha256,
        "commit_sha": commit_sha,
        "tree_sha": tree_sha,
        "source_manifest_sha256": source_manifest_sha256,
        "runtime_evidence_sha256": runtime_evidence_sha256,
        "slurm_job_id": scheduler_binding["job_id"],
        "requested_resource_sha256": scheduler_binding["requested_resource_sha256"],
        "scheduler_binding_sha256": scheduler_binding["sha256"],
        "final_tabicl_attested": True,
        "completed": True,
    }
    if value["payload"] != expected:
        raise ValueError("action completion is not bound to successful exact-T case")
    if expected_job_binding is not None and (
        value["payload"]["slurm_job_id"] != expected_job_binding["job_id"]
        or value["payload"]["requested_resource_sha256"]
        != expected_job_binding["requested_resource_sha256"]
    ):
        raise ValueError("action completion differs from submission receipt")
    return actual


def make_runtime_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "kind": "h100_runtime_evidence",
        "payload": dict(payload),
    }
    return {**body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def _query_visible_gpu_devices() -> list[dict[str, str]]:
    executable = os.environ.get("NVIDIA_SMI")
    raw_fd = os.environ.get("FORMAL_NVIDIA_SMI_FD")
    if (
        not executable
        or not os.path.isabs(executable)
        or not os.path.isfile(executable)
        or not os.access(executable, os.X_OK)
    ):
        raise RuntimeError("NVIDIA_SMI must be an absolute executable")
    if (
        raw_fd is None
        or not raw_fd.isdecimal()
        or executable != f"/proc/self/fd/{raw_fd}"
    ):
        raise RuntimeError("NVIDIA_SMI must use its verified open descriptor")
    nvidia_fd = int(raw_fd)
    _digest(
        os.environ.get("FORMAL_NVIDIA_SMI_SHA256"),
        "runtime nvidia-smi executable",
    )
    raw_tokens = os.environ.get("FORMAL_VISIBLE_GPU_TOKENS", "")
    tokens = raw_tokens.split(",") if raw_tokens else []
    if not tokens or any(
        re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9._-]+|MIG-[A-Za-z0-9._-]+)", token)
        is None
        for token in tokens
    ):
        raise RuntimeError("runtime evidence lacks valid CUDA-visible GPU tokens")
    devices: list[dict[str, str]] = []
    for token in tokens:
        try:
            result = subprocess.run(
                [
                    executable,
                    f"--id={token}",
                    "--query-gpu=uuid,name,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=NVIDIA_QUERY_TIMEOUT_SECONDS,
                pass_fds=(nvidia_fd,),
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "CUDA-visible nvidia-smi runtime evidence query timed out"
            ) from error
        rows = [line for line in result.stdout.splitlines() if line.strip()]
        if result.returncode != 0 or len(rows) != 1:
            raise RuntimeError("CUDA-visible nvidia-smi runtime evidence query failed")
        line = rows[0]
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or any(not field for field in fields):
            raise RuntimeError("nvidia-smi runtime evidence row is invalid")
        devices.append(
            {"uuid": fields[0], "name": fields[1], "driver_version": fields[2]}
        )
    return devices


def _memory_mb(raw: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)([MGT]?)", raw or "")
    if match is None:
        raise RuntimeError("SLURM_MEM_PER_NODE is malformed")
    value = int(match.group(1))
    suffix = match.group(2)
    factors = {"": 1, "M": 1, "G": 1024, "T": 1 << 20}
    result = value * factors[suffix]
    if int(result) != result:
        raise RuntimeError("SLURM_MEM_PER_NODE is not an integral MiB value")
    return int(result)


def _scheduler_contract() -> dict[str, Any]:
    return {
        "partition": "h100",
        "qos": "short",
        "time_limit": "03:00:00",
        "resources_by_world_size": {
            "1": {
                "gpus": 1,
                "cpus_per_task": 32,
                "memory_mb": 131072,
                "wrapper": "scripts/slurm_h100_identity_maxseq_smoke.sh",
            },
            "2": {
                "gpus": 2,
                "cpus_per_task": 64,
                "memory_mb": 131072,
                "wrapper": "scripts/slurm_h100_identity_nccl_smoke.sh",
            },
        },
    }


def _load_held_scheduler_contract(
    *,
    case: ValidationCase,
    job_id: str,
    commit_sha: str,
    tree_sha: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    """Load the write-once, pre-release scheduler commitment for this job."""

    cases_root_raw = os.environ.get("H100_VALIDATION_ROOT", "")
    ceiling_raw = os.environ.get("FORMAL_ATTESTATION_CEILING_BYTES", "")
    try:
        ceiling = int(ceiling_raw)
    except ValueError as error:
        raise RuntimeError("held-plan byte ceiling is malformed") from error
    cases_root = Path(cases_root_raw)
    if (
        not cases_root.is_absolute()
        or any(part in {"", ".", ".."} for part in cases_root.parts[1:])
        or ceiling < 1
    ):
        raise RuntimeError("held-plan location or byte ceiling is invalid")
    held_path = cases_root.parent / "held-plan.json"
    try:
        held = _strict_json(
            _read_regular_bytes(held_path, max_bytes=ceiling),
            where="H100 gate held plan at runtime",
        )
        _exact_keys(
            held,
            {"schema_version", "kind", "payload", "sha256"},
            "runtime held plan",
        )
        body = {key: held[key] for key in ("schema_version", "kind", "payload")}
        if (
            held["schema_version"] != 1
            or held["kind"] != "h100_gate_held_plan"
            or held["sha256"] != hashlib.sha256(_canonical(body)).hexdigest()
        ):
            raise ValueError("runtime held plan integrity mismatch")
        payload = held["payload"]
        if not isinstance(payload, Mapping):
            raise ValueError("runtime held-plan payload is malformed")
        if (
            payload.get("commit_sha") != commit_sha
            or payload.get("tree_sha") != tree_sha
            or payload.get("source_manifest_sha256") != source_manifest_sha256
            or payload.get("jobs_held_at_publication") is not True
            or payload.get("scheduler") != _scheduler_contract()
        ):
            raise ValueError("runtime held plan differs from exact source/scheduler")
        repository = _validated_repository_binding(
            payload.get("repository_binding"),
            expected_commit_sha=commit_sha,
        )
        expected_repository_exports = {
            "repository_url": os.environ.get("CANDIDATE_REPOSITORY"),
            "repository_ref": os.environ.get("CANDIDATE_REPOSITORY_REF"),
            "git_sha256": os.environ.get("FORMAL_GIT_SHA256"),
            "repository_identity_sha256": os.environ.get(
                "FORMAL_REPOSITORY_IDENTITY_SHA256"
            ),
            "query_sha256": os.environ.get("FORMAL_REPOSITORY_QUERY_SHA256"),
        }
        if any(
            repository[key] != value
            for key, value in expected_repository_exports.items()
        ):
            raise ValueError("runtime repository exports differ from held plan")
        command_digests = payload.get("scheduler_command_sha256")
        _exact_keys(
            command_digests,
            {"sbatch", "scontrol", "scancel", "squeue", "sacct"},
            "runtime held-plan scheduler digests",
        )
        for name, digest in command_digests.items():
            _digest(digest, f"runtime held-plan {name} executable")
        planned_cases = payload.get("cases")
        matrix = build_matrix()
        if not isinstance(planned_cases, list) or len(planned_cases) != len(matrix):
            raise ValueError("runtime held-plan case matrix is incomplete")
        current: Mapping[str, Any] | None = None
        seen_jobs: set[str] = set()
        for expected_case, item in zip(matrix, planned_cases):
            _exact_keys(
                item,
                {
                    "case_id",
                    "world_size",
                    "artifact_identity_sha256",
                    "requested_resource",
                    "requested_resource_sha256",
                    "job_id",
                    "cluster",
                },
                "runtime held-plan case",
            )
            planned_job_id = item["job_id"]
            planned_cluster = item["cluster"]
            if (
                item["case_id"] != expected_case.case_id
                or item["world_size"] != expected_case.world_size
                or item["requested_resource"] != gate_requested_resource(expected_case)
                or item["requested_resource_sha256"]
                != gate_requested_resource_sha256(expected_case)
                or not isinstance(planned_job_id, str)
                or re.fullmatch(r"[1-9][0-9]{0,19}", planned_job_id) is None
                or planned_job_id in seen_jobs
                or (
                    planned_cluster is not None
                    and (
                        not isinstance(planned_cluster, str)
                        or re.fullmatch(r"[A-Za-z0-9._-]+", planned_cluster) is None
                    )
                )
            ):
                raise ValueError("runtime held-plan case binding is malformed")
            _digest(
                item["artifact_identity_sha256"],
                f"runtime held-plan {expected_case.case_id} artifact identity",
            )
            seen_jobs.add(planned_job_id)
            if expected_case.case_id == case.case_id:
                current = item
        if current is None or current["job_id"] != job_id:
            raise ValueError("runtime job is not the held-plan allocation for this case")
        expected_identity = os.environ.get("VALIDATION_ARTIFACT_IDENTITY_SHA256")
        if current["artifact_identity_sha256"] != _digest(
            expected_identity, "runtime artifact identity"
        ):
            raise ValueError("runtime artifact identity differs from held plan")
        return {
            "held_plan_sha256": held["sha256"],
            "scontrol_sha256": command_digests["scontrol"],
            "receipt_cluster": current["cluster"],
        }
    except (OSError, ValueError) as error:
        raise RuntimeError("cannot bind runtime allocation to held plan") from error


def _scontrol_field(line: str, name: str) -> str:
    values = re.findall(rf"(?:^| ){re.escape(name)}=([^ ]+)", line)
    if len(values) != 1 or not values[0]:
        raise RuntimeError(f"scontrol allocation field is missing or duplicate: {name}")
    return values[0]


def _tres_value(raw: str, name: str, *, separator: str = "=") -> str:
    prefix = f"{name}{separator}"
    values = [item[len(prefix):] for item in raw.split(",") if item.startswith(prefix)]
    if len(values) != 1 or not values[0]:
        raise RuntimeError(f"scontrol TRES field is missing or duplicate: {name}")
    return values[0]


def _capture_scontrol_allocation(
    *,
    case: ValidationCase,
    job_id: str,
    receipt_cluster: str | None,
    expected_scontrol_sha256: str,
) -> dict[str, Any]:
    configured = os.environ.get("SCONTROL")
    scontrol_path = Path(configured) if configured else Path("/usr/bin/scontrol")
    fd, before = _open_trusted_scheduler_executable(
        scontrol_path,
        expected_scontrol_sha256,
        command_name="scontrol",
    )
    semantic_argv = [
        "scontrol",
        *([f"--clusters={receipt_cluster}"] if receipt_cluster is not None else []),
        "show",
        "job",
        "-o",
        job_id,
    ]
    try:
        completed = subprocess.run(
            [f"/proc/self/fd/{fd}", *semantic_argv[1:]],
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=SCHEDULER_QUERY_TIMEOUT_SECONDS,
            pass_fds=(fd,),
        )
        _revalidate_trusted_scheduler_executable(
            fd,
            scontrol_path,
            before,
            expected_scontrol_sha256,
            command_name="scontrol",
        )
    finally:
        os.close(fd)
    stdout = completed.stdout
    stderr = completed.stderr
    if len(stdout) > 1_000_000 or len(stderr) > 1_000_000:
        raise RuntimeError("scontrol allocation query exceeded its byte ceiling")
    try:
        lines = stdout.decode("utf-8", errors="strict").splitlines()
        stderr_text = stderr.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise RuntimeError("scontrol allocation query emitted non-UTF-8 output") from error
    if completed.returncode != 0 or stderr_text or len(lines) != 1:
        raise RuntimeError("scontrol allocation query was nonzero, noisy, missing, or duplicate")
    if stdout != (lines[0] + "\n").encode("utf-8"):
        raise RuntimeError("scontrol allocation query output is not canonical")
    line = lines[0]
    requested = gate_requested_resource(case)
    fields = {
        name: _scontrol_field(line, name)
        for name in (
            "JobId",
            "Partition",
            "QOS",
            "TimeLimit",
            "NumNodes",
            "NumCPUs",
            "CPUs/Task",
            "MinMemoryNode",
            "ReqTRES",
            "AllocTRES",
            "TresPerNode",
        )
    }
    try:
        numeric = {
            "nodes": int(fields["NumNodes"]),
            "cpus": int(fields["NumCPUs"]),
            "cpus_per_task": int(fields["CPUs/Task"]),
            "memory_mb": _memory_mb(fields["MinMemoryNode"]),
            "req_cpu": int(_tres_value(fields["ReqTRES"], "cpu")),
            "req_memory_mb": _memory_mb(_tres_value(fields["ReqTRES"], "mem")),
            "req_nodes": int(_tres_value(fields["ReqTRES"], "node")),
            "req_gpus": int(_tres_value(fields["ReqTRES"], "gres/gpu")),
            "alloc_cpu": int(_tres_value(fields["AllocTRES"], "cpu")),
            "alloc_memory_mb": _memory_mb(
                _tres_value(fields["AllocTRES"], "mem")
            ),
            "alloc_nodes": int(_tres_value(fields["AllocTRES"], "node")),
            "alloc_gpus": int(_tres_value(fields["AllocTRES"], "gres/gpu")),
            "per_node_gpus": int(
                _tres_value(fields["TresPerNode"], "gres/gpu", separator=":")
            ),
        }
    except ValueError as error:
        raise RuntimeError("scontrol allocation contains malformed numeric resources") from error
    if (
        fields["JobId"] != job_id
        or fields["Partition"] != requested["partition"]
        or fields["QOS"] != requested["qos"]
        or fields["TimeLimit"] != requested["time_limit"]
        or numeric["nodes"] != requested["nodes"]
        or numeric["cpus"] != requested["cpus_per_task"]
        or numeric["cpus_per_task"] != requested["cpus_per_task"]
        or numeric["memory_mb"] != requested["memory_mb"]
        or numeric["req_cpu"] != requested["cpus_per_task"]
        or numeric["alloc_cpu"] != requested["cpus_per_task"]
        or numeric["req_memory_mb"] != requested["memory_mb"]
        or numeric["alloc_memory_mb"] != requested["memory_mb"]
        or numeric["req_nodes"] != requested["nodes"]
        or numeric["alloc_nodes"] != requested["nodes"]
        or numeric["req_gpus"] != requested["gpus"]
        or numeric["alloc_gpus"] != requested["gpus"]
        or numeric["per_node_gpus"] != requested["gpus"]
    ):
        raise RuntimeError("actual scontrol allocation differs from exact H100 gate resources")
    return {
        "scontrol_query_returncode": completed.returncode,
        "scontrol_query_argv_sha256": hashlib.sha256(
            _canonical(semantic_argv)
        ).hexdigest(),
        "scontrol_query_stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "scontrol_query_stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "slurm_job_partition": fields["Partition"],
        "slurm_qos": fields["QOS"],
        "slurm_time_limit": fields["TimeLimit"],
        "slurm_num_nodes": numeric["nodes"],
        "slurm_num_cpus": numeric["cpus"],
        "slurm_cpus_per_task": numeric["cpus_per_task"],
        "slurm_memory_per_node_mb": numeric["memory_mb"],
        "slurm_req_tres": fields["ReqTRES"],
        "slurm_alloc_tres": fields["AllocTRES"],
        "slurm_tres_per_node": fields["TresPerNode"],
    }


def capture_scheduler_binding(
    case: ValidationCase,
    devices: Sequence[Mapping[str, str]],
    *,
    commit_sha: str,
    tree_sha: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    requested = gate_requested_resource(case)
    requested_sha = gate_requested_resource_sha256(case)
    exported_sha = os.environ.get("FORMAL_REQUESTED_RESOURCE_SHA256")
    if exported_sha != requested_sha:
        raise RuntimeError("requested-resource digest differs from exact matrix")
    exported = {
        "partition": os.environ.get("FORMAL_REQUESTED_PARTITION"),
        "qos": os.environ.get("FORMAL_REQUESTED_QOS"),
        "time_limit": os.environ.get("FORMAL_REQUESTED_TIME_LIMIT"),
        "nodes": int(os.environ.get("FORMAL_REQUESTED_NODES", "0")),
        "gpus": int(os.environ.get("FORMAL_REQUESTED_GPUS", "0")),
        "cpus_per_task": int(os.environ.get("FORMAL_REQUESTED_CPUS", "0")),
        "memory_mb": int(os.environ.get("FORMAL_REQUESTED_MEMORY_MB", "0")),
        "wrapper": requested["wrapper"],
    }
    if exported != requested:
        raise RuntimeError("exported requested resources differ from exact matrix")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if re.fullmatch(r"[1-9][0-9]{0,19}", job_id) is None:
        raise RuntimeError("SLURM_JOB_ID is malformed")
    held = _load_held_scheduler_contract(
        case=case,
        job_id=job_id,
        commit_sha=commit_sha,
        tree_sha=tree_sha,
        source_manifest_sha256=source_manifest_sha256,
    )
    actual = _capture_scontrol_allocation(
        case=case,
        job_id=job_id,
        receipt_cluster=held["receipt_cluster"],
        expected_scontrol_sha256=held["scontrol_sha256"],
    )
    tokens = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    visible_uuids = [device["uuid"] for device in devices]
    if (
        len(tokens) != case.world_size
        or os.environ.get("FORMAL_VISIBLE_GPU_TOKENS") != ",".join(tokens)
        or os.environ.get("FORMAL_VISIBLE_GPU_UUIDS") != ",".join(visible_uuids)
        or os.environ.get("SLURM_JOB_PARTITION") != requested["partition"]
        or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != requested["cpus_per_task"]
        or _memory_mb(os.environ.get("SLURM_MEM_PER_NODE", "")) != requested["memory_mb"]
    ):
        raise RuntimeError("observed Slurm allocation differs from requested resources")
    body = {
        "job_id": job_id,
        "requested_resource": requested,
        "requested_resource_sha256": requested_sha,
        "held_plan_sha256": held["held_plan_sha256"],
        "scontrol_sha256": held["scontrol_sha256"],
        "slurm_receipt_cluster": held["receipt_cluster"],
        **actual,
        "slurm_cluster_name": os.environ.get("SLURM_CLUSTER_NAME"),
        "cuda_visible_devices": tokens,
        "visible_gpu_uuids": visible_uuids,
    }
    return {**body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def capture_runtime_evidence(
    *,
    case_id: str,
    commit_sha: str,
    tree_sha: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    """Capture post-action environment/GPU facts inside exact-T Python."""
    from tabicl.train._provenance import runtime_environment_manifest

    cases = {case.case_id: case for case in build_matrix()}
    case = cases.get(case_id)
    if case is None:
        raise ValueError("unknown runtime-evidence case")
    environment = runtime_environment_manifest(require_formal_runtime=True)
    devices = _query_visible_gpu_devices()
    if environment["payload"]["visible_cuda_device_count"] != case.world_size:
        raise RuntimeError("PyTorch visible CUDA count does not match case world size")
    if len(devices) != case.world_size:
        raise RuntimeError("nvidia-smi visible GPU count does not match case world size")
    scheduler_binding = capture_scheduler_binding(
        case,
        devices,
        commit_sha=commit_sha,
        tree_sha=tree_sha,
        source_manifest_sha256=source_manifest_sha256,
    )
    return make_runtime_evidence(
        {
            "case_id": case_id,
            "commit_sha": commit_sha,
            "tree_sha": tree_sha,
            "source_manifest_sha256": source_manifest_sha256,
            "world_size": case.world_size,
            "environment": environment,
            "gpu_devices": devices,
            "nvidia_smi_sha256": _digest(
                os.environ.get("FORMAL_NVIDIA_SMI_SHA256"),
                "runtime nvidia-smi executable",
            ),
            "scheduler_binding": scheduler_binding,
        }
    )


def _validate_manifest_envelope(
    value: Mapping[str, Any], *, expected_kind: str, where: str
) -> str:
    _exact_keys(value, {"schema_version", "kind", "payload", "sha256"}, where)
    if value["schema_version"] != 1 or value["kind"] != expected_kind:
        raise ValueError(f"{where} schema/kind mismatch")
    body = {key: value[key] for key in ("schema_version", "kind", "payload")}
    actual = hashlib.sha256(_canonical(body)).hexdigest()
    if value["sha256"] != actual:
        raise ValueError(f"{where} self-hash mismatch")
    return actual


def validate_runtime_evidence(
    value: Mapping[str, Any],
    *,
    case: ValidationCase,
    commit_sha: str,
    tree_sha: str,
    source_manifest_sha256: str,
    expected_job_binding: Mapping[str, Any] | None = None,
    expected_nvidia_smi_sha256: str | None = None,
) -> str:
    actual = _validate_manifest_envelope(
        value, expected_kind="h100_runtime_evidence", where="runtime evidence"
    )
    payload = value["payload"]
    _exact_keys(
        payload,
        {
            "case_id",
            "commit_sha",
            "tree_sha",
            "source_manifest_sha256",
            "world_size",
            "environment",
            "gpu_devices",
            "nvidia_smi_sha256",
            "scheduler_binding",
        },
        "runtime evidence payload",
    )
    expected = {
        "case_id": case.case_id,
        "commit_sha": commit_sha,
        "tree_sha": tree_sha,
        "source_manifest_sha256": source_manifest_sha256,
        "world_size": case.world_size,
    }
    if any(payload[key] != expected[key] for key in expected):
        raise ValueError("runtime evidence is not bound to exact case/source")
    _digest(payload["nvidia_smi_sha256"], "runtime nvidia-smi executable")
    if expected_nvidia_smi_sha256 is not None and payload[
        "nvidia_smi_sha256"
    ] != _digest(expected_nvidia_smi_sha256, "expected runtime nvidia-smi"):
        raise ValueError("runtime nvidia-smi digest mismatch")
    environment = payload["environment"]
    _validate_manifest_envelope(
        environment, expected_kind="environment", where="runtime environment"
    )
    _exact_keys(
        environment["payload"], _RUNTIME_ENVIRONMENT_KEYS, "runtime environment payload"
    )
    environment_payload = environment["payload"]
    _digest(
        environment_payload["python_executable_sha256"],
        "runtime Python executable",
    )
    for field in ("python_cache_tag", "python_soabi"):
        if (
            not isinstance(environment_payload[field], str)
            or not environment_payload[field]
            or any(ord(character) < 33 for character in environment_payload[field])
        ):
            raise ValueError(f"runtime {field} is malformed")
    fingerprint_schema = environment_payload[
        "environment_fingerprint_schema_version"
    ]
    if (
        isinstance(fingerprint_schema, bool)
        or not isinstance(fingerprint_schema, int)
        or fingerprint_schema != 2
    ):
        raise ValueError("runtime environment fingerprint schema is unsupported")
    _digest(
        environment_payload["installed_distributions_sha256"],
        "runtime installed-distribution inventory",
    )
    if environment_payload["unavailable_formal_runtime_distributions"] != []:
        raise ValueError("runtime formal-distribution inventory is incomplete")
    if environment_payload["flash_attn3_available"] is not True:
        raise ValueError("runtime FlashAttention 3 backend is unavailable")
    nccl_version = environment_payload["nccl_version"]
    if (
        not isinstance(nccl_version, list)
        or not nccl_version
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in nccl_version
        )
    ):
        raise ValueError("runtime NCCL version is malformed")
    installed = environment_payload["formal_runtime_distributions"]
    if not isinstance(installed, list) or len(installed) != len(
        _FORMAL_RUNTIME_DISTRIBUTIONS
    ):
        raise ValueError("runtime formal-distribution inventory is incomplete")
    for expected_name, entry in zip(_FORMAL_RUNTIME_DISTRIBUTIONS, installed):
        _exact_keys(
            entry,
            {
                "name",
                "version",
                "metadata_sha256",
                "record_sha256",
                "wheel_sha256",
                "module",
                "module_version",
                "module_origin_relative_path",
                "module_origin_sha256",
                "record_verified_file_count",
                "record_verified_total_bytes",
                "record_verified_files_sha256",
                "record_pyc_mismatch_count",
            },
            "runtime formal distribution",
        )
        name, version = entry["name"], entry["version"]
        if name != expected_name or _DISTRIBUTION_NAME.fullmatch(name) is None:
            raise ValueError("runtime formal-distribution name is malformed")
        if (
            not isinstance(version, str)
            or _DISTRIBUTION_VERSION.fullmatch(version) is None
        ):
            raise ValueError("runtime formal-distribution version is malformed")
        if entry["module"] != _FORMAL_RUNTIME_MODULES[expected_name]:
            raise ValueError("runtime formal-distribution module is malformed")
        origin = entry["module_origin_relative_path"]
        if (
            not isinstance(origin, str)
            or not origin
            or PurePosixPath(origin).is_absolute()
            or any(part in {"", ".", ".."} for part in PurePosixPath(origin).parts)
            or PurePosixPath(origin).as_posix() != origin
        ):
            raise ValueError("runtime formal-distribution module origin is malformed")
        _digest(
            entry["module_origin_sha256"],
            "runtime formal-distribution module origin",
        )
        verified_count = entry["record_verified_file_count"]
        verified_bytes = entry["record_verified_total_bytes"]
        mismatch_count = entry["record_pyc_mismatch_count"]
        if (
            isinstance(verified_count, bool)
            or not isinstance(verified_count, int)
            or not 1 <= verified_count <= 100_000
        ):
            raise ValueError("runtime formal-distribution RECORD count is malformed")
        if (
            isinstance(verified_bytes, bool)
            or not isinstance(verified_bytes, int)
            or not 0 <= verified_bytes <= (64 << 30)
        ):
            raise ValueError("runtime formal-distribution RECORD bytes are malformed")
        if (
            isinstance(mismatch_count, bool)
            or not isinstance(mismatch_count, int)
            or not 0 <= mismatch_count <= verified_count
        ):
            raise ValueError(
                "runtime formal-distribution RECORD pyc mismatch count is malformed"
            )
        _digest(
            entry["record_verified_files_sha256"],
            "runtime formal-distribution verified RECORD files",
        )
        module_version = entry["module_version"]
        if module_version is not None and (
            not isinstance(module_version, str)
            or _DISTRIBUTION_VERSION.fullmatch(module_version) is None
        ):
            raise ValueError("runtime formal-distribution module version is malformed")
        for field in ("metadata_sha256", "record_sha256", "wheel_sha256"):
            _digest(entry[field], f"runtime formal-distribution {field}")
    for field in (
        "python_version",
        "python_implementation",
        "platform_system",
        "platform_release",
        "platform_machine",
        "torch_version",
        "numpy_version",
        "cuda_runtime_version",
    ):
        value = environment_payload[field]
        if (
            not isinstance(value, str)
            or not value
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError(f"runtime environment {field} is malformed")
    cudnn_version = environment_payload["cudnn_version"]
    if (
        isinstance(cudnn_version, bool)
        or not isinstance(cudnn_version, int)
        or cudnn_version < 1
    ):
        raise ValueError("runtime environment cuDNN version is malformed")
    visible_count = environment["payload"]["visible_cuda_device_count"]
    if (
        isinstance(visible_count, bool)
        or not isinstance(visible_count, int)
        or visible_count != case.world_size
    ):
        raise ValueError("runtime environment visible CUDA count mismatch")
    devices = payload["gpu_devices"]
    if not isinstance(devices, list) or len(devices) != case.world_size:
        raise ValueError("runtime GPU device count mismatch")
    seen: set[str] = set()
    for device in devices:
        _exact_keys(device, {"uuid", "name", "driver_version"}, "runtime GPU device")
        if any(not isinstance(device[key], str) or not device[key] for key in device):
            raise ValueError("runtime GPU device field is invalid")
        if device["uuid"] in seen or "H100" not in device["name"]:
            raise ValueError("runtime GPU UUID/model is invalid")
        seen.add(device["uuid"])
    scheduler_binding = payload["scheduler_binding"]
    _exact_keys(
        scheduler_binding,
        {
            "job_id",
            "requested_resource",
            "requested_resource_sha256",
            "held_plan_sha256",
            "scontrol_sha256",
            "scontrol_query_returncode",
            "scontrol_query_argv_sha256",
            "scontrol_query_stdout_sha256",
            "scontrol_query_stderr_sha256",
            "slurm_job_partition",
            "slurm_cluster_name",
            "slurm_receipt_cluster",
            "slurm_qos",
            "slurm_time_limit",
            "slurm_num_nodes",
            "slurm_num_cpus",
            "slurm_cpus_per_task",
            "slurm_memory_per_node_mb",
            "slurm_req_tres",
            "slurm_alloc_tres",
            "slurm_tres_per_node",
            "cuda_visible_devices",
            "visible_gpu_uuids",
            "sha256",
        },
        "runtime scheduler binding",
    )
    binding_body = {key: scheduler_binding[key] for key in scheduler_binding if key != "sha256"}
    requested = gate_requested_resource(case)
    requested_sha = gate_requested_resource_sha256(case)
    scheduler_job_id = scheduler_binding["job_id"]
    receipt_cluster = scheduler_binding["slurm_receipt_cluster"]
    expected_query_argv = [
        "scontrol",
        *([f"--clusters={receipt_cluster}"] if receipt_cluster is not None else []),
        "show",
        "job",
        "-o",
        scheduler_job_id,
    ]
    if receipt_cluster is not None and (
        not isinstance(receipt_cluster, str)
        or re.fullmatch(r"[A-Za-z0-9._-]+", receipt_cluster) is None
    ):
        raise ValueError("runtime scheduler receipt cluster is malformed")
    held_plan_sha = _digest(
        scheduler_binding["held_plan_sha256"], "runtime held-plan digest"
    )
    scontrol_sha = _digest(
        scheduler_binding["scontrol_sha256"], "runtime scontrol executable"
    )
    _digest(
        scheduler_binding["scontrol_query_stdout_sha256"],
        "runtime scontrol stdout",
    )
    try:
        req_tres = scheduler_binding["slurm_req_tres"]
        alloc_tres = scheduler_binding["slurm_alloc_tres"]
        per_node_tres = scheduler_binding["slurm_tres_per_node"]
        if any(
            not isinstance(item, str) or not item
            for item in (req_tres, alloc_tres, per_node_tres)
        ):
            raise ValueError("TRES values must be non-empty strings")
        observed_tres = {
            "req_cpu": int(_tres_value(req_tres, "cpu")),
            "req_memory_mb": _memory_mb(_tres_value(req_tres, "mem")),
            "req_nodes": int(_tres_value(req_tres, "node")),
            "req_gpus": int(_tres_value(req_tres, "gres/gpu")),
            "alloc_cpu": int(_tres_value(alloc_tres, "cpu")),
            "alloc_memory_mb": _memory_mb(_tres_value(alloc_tres, "mem")),
            "alloc_nodes": int(_tres_value(alloc_tres, "node")),
            "alloc_gpus": int(_tres_value(alloc_tres, "gres/gpu")),
            "per_node_gpus": int(
                _tres_value(per_node_tres, "gres/gpu", separator=":")
            ),
        }
    except (RuntimeError, ValueError) as error:
        raise ValueError("runtime scheduler TRES evidence is malformed") from error
    if (
        scheduler_binding["sha256"] != hashlib.sha256(_canonical(binding_body)).hexdigest()
        or not isinstance(scheduler_job_id, str)
        or re.fullmatch(r"[1-9][0-9]{0,19}", scheduler_job_id) is None
        or scheduler_binding["requested_resource"] != requested
        or scheduler_binding["requested_resource_sha256"] != requested_sha
        or isinstance(scheduler_binding["scontrol_query_returncode"], bool)
        or scheduler_binding["scontrol_query_returncode"] != 0
        or scheduler_binding["scontrol_query_argv_sha256"]
        != hashlib.sha256(_canonical(expected_query_argv)).hexdigest()
        or scheduler_binding["scontrol_query_stderr_sha256"]
        != hashlib.sha256(b"").hexdigest()
        or scheduler_binding["slurm_job_partition"] != requested["partition"]
        or not isinstance(scheduler_binding["slurm_cluster_name"], str)
        or not scheduler_binding["slurm_cluster_name"]
        or (
            receipt_cluster is not None
            and scheduler_binding["slurm_cluster_name"] != receipt_cluster
        )
        or scheduler_binding["slurm_qos"] != requested["qos"]
        or scheduler_binding["slurm_time_limit"] != requested["time_limit"]
        or any(
            isinstance(scheduler_binding[field], bool)
            or not isinstance(scheduler_binding[field], int)
            for field in (
                "slurm_num_nodes",
                "slurm_num_cpus",
                "slurm_cpus_per_task",
                "slurm_memory_per_node_mb",
            )
        )
        or scheduler_binding["slurm_num_nodes"] != requested["nodes"]
        or scheduler_binding["slurm_num_cpus"] != requested["cpus_per_task"]
        or scheduler_binding["slurm_cpus_per_task"] != requested["cpus_per_task"]
        or scheduler_binding["slurm_memory_per_node_mb"] != requested["memory_mb"]
        or observed_tres["req_cpu"] != requested["cpus_per_task"]
        or observed_tres["alloc_cpu"] != requested["cpus_per_task"]
        or observed_tres["req_memory_mb"] != requested["memory_mb"]
        or observed_tres["alloc_memory_mb"] != requested["memory_mb"]
        or observed_tres["req_nodes"] != requested["nodes"]
        or observed_tres["alloc_nodes"] != requested["nodes"]
        or observed_tres["req_gpus"] != requested["gpus"]
        or observed_tres["alloc_gpus"] != requested["gpus"]
        or observed_tres["per_node_gpus"] != requested["gpus"]
        or not isinstance(scheduler_binding["cuda_visible_devices"], list)
        or len(scheduler_binding["cuda_visible_devices"]) != case.world_size
        or scheduler_binding["visible_gpu_uuids"] != [device["uuid"] for device in devices]
    ):
        raise ValueError("runtime scheduler/resource binding mismatch")
    if expected_job_binding is not None and (
        scheduler_binding["job_id"] != expected_job_binding["job_id"]
        or scheduler_binding["requested_resource"]
        != expected_job_binding["requested_resource"]
        or scheduler_binding["requested_resource_sha256"]
        != expected_job_binding["requested_resource_sha256"]
        or held_plan_sha
        != expected_job_binding.get("held_plan_sha256")
        or scontrol_sha
        != expected_job_binding.get("scontrol_sha256")
        or scheduler_binding["slurm_receipt_cluster"]
        != expected_job_binding.get("cluster")
        or (
            expected_job_binding.get("cluster") is not None
            and scheduler_binding["slurm_cluster_name"] != expected_job_binding["cluster"]
        )
    ):
        raise ValueError("runtime scheduler binding differs from submission receipt")
    return actual


def _exact_keys(value: Any, expected: set[str], where: str) -> None:
    actual = set(value) if isinstance(value, Mapping) else set()
    if not isinstance(value, Mapping) or actual != expected:
        raise ValueError(
            f"{where} keys mismatch; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _canonical_file_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value) + b"\n").hexdigest()


def validate_case_result(
    value: Mapping[str, Any], *, case: ValidationCase
) -> tuple[str, float, float]:
    actual = _validate_manifest_envelope(
        value, expected_kind="h100_case_result", where="case result"
    )
    payload = value["payload"]
    _exact_keys(
        payload,
        {
            "case_id",
            "assertion",
            "observed_sequence_length",
            "recompute",
            "world_size",
            "monotonic_start_seconds",
            "monotonic_end_seconds",
            "checkpoint_relative_path",
            "checkpoint_sha256",
            "checkpoint_size",
            "checkpoint_ceiling_bytes",
            "slurm_job_id",
            "requested_resource_sha256",
        },
        "case result payload",
    )
    expected = {
        "case_id": case.case_id,
        "assertion": case.assertion,
        "observed_sequence_length": case.observed_sequence_length,
        "recompute": case.recompute,
        "world_size": case.world_size,
    }
    if any(payload[key] != expected[key] for key in expected):
        raise ValueError("case result does not match fixed matrix")
    if (
        re.fullmatch(r"[1-9][0-9]{0,19}", payload["slurm_job_id"] or "") is None
        or payload["requested_resource_sha256"]
        != gate_requested_resource_sha256(case)
    ):
        raise ValueError("case result scheduler/resource binding mismatch")
    ceiling = payload["checkpoint_ceiling_bytes"]
    if isinstance(ceiling, bool) or not isinstance(ceiling, int) or ceiling < 1:
        raise ValueError("case checkpoint ceiling is invalid")
    if case.stage is not None:
        expected_path = f"step-{case.repeat_steps}.ckpt"
        if (
            payload["checkpoint_relative_path"] != expected_path
            or _HEX64.fullmatch(payload["checkpoint_sha256"] or "") is None
            or isinstance(payload["checkpoint_size"], bool)
            or not isinstance(payload["checkpoint_size"], int)
            or payload["checkpoint_size"] < 1
            or payload["checkpoint_size"] > ceiling
        ):
            raise ValueError("Trainer case checkpoint evidence is invalid")
    elif any(
        payload[field] is not None
        for field in (
            "checkpoint_relative_path",
            "checkpoint_sha256",
            "checkpoint_size",
        )
    ):
        raise ValueError("special case must not claim a checkpoint")
    start, end = payload["monotonic_start_seconds"], payload["monotonic_end_seconds"]
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
        or not math.isfinite(float(start))
        or not math.isfinite(float(end))
        or end <= start
    ):
        raise ValueError("case result monotonic span is invalid")
    return actual, float(start), float(end)


def validate_artifact_manifest(
    value: Mapping[str, Any],
    *,
    case: ValidationCase,
    embedded_raw_sha256: Mapping[str, str],
) -> str:
    _exact_keys(value, {"schema_version", "case_id", "files"}, "artifact manifest")
    if value["schema_version"] != 1 or value["case_id"] != case.case_id:
        raise ValueError("artifact manifest case/schema mismatch")
    required = {
        "source-attestation.json",
        "action-completion.json",
        "runtime-evidence.json",
        "compute/case-result.json",
        "compute.log",
    }
    if case.utilization_required:
        required |= {"gpu.csv", "gpu.jsonl", "gpu-summary.json"}
    if case.stage is not None:
        required.add(f"compute/step-{case.repeat_steps}.ckpt")
    files = value["files"]
    if not isinstance(files, list):
        raise ValueError("artifact manifest files must be a list")
    names: list[str] = []
    by_name: dict[str, Mapping[str, Any]] = {}
    for entry in files:
        _exact_keys(entry, {"name", "sha256", "size"}, "artifact file entry")
        name = entry["name"]
        if not isinstance(name, str) or name not in required or name in by_name:
            raise ValueError("artifact manifest file name is invalid or duplicate")
        _digest(entry["sha256"], f"artifact file {name} digest")
        size = entry["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("artifact manifest file size is invalid")
        names.append(name)
        by_name[name] = entry
    if set(names) != required or names != sorted(names):
        raise ValueError("artifact manifest file set/order mismatch")
    for name, expected_digest in embedded_raw_sha256.items():
        if by_name[name]["sha256"] != expected_digest:
            raise ValueError(f"artifact manifest does not bind embedded {name}")
    return hashlib.sha256(_canonical(value)).hexdigest()


def validate_smoke_attestation(
    attestation: Mapping[str, Any],
    *,
    expected_sha256: str,
    expected_commit_sha: str,
    expected_tree_sha: str,
    expected_environment_sha256: str,
    expected_source_manifest_sha256: str,
    expected_gpu_model: str,
    expected_checkpoint_ceiling_bytes: int,
    expected_submission_receipt_sha256: str | None = None,
    expected_sacct_sha256: str | None = None,
    expected_git_sha256: str | None = None,
    expected_repository_identity_sha256: str | None = None,
    expected_repository_query_sha256: str | None = None,
    expected_nvidia_smi_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate integrity against trust inputs held outside this payload."""
    _exact_keys(
        attestation,
        {"schema_version", "kind", "payload", "sha256"},
        "smoke attestation",
    )
    if attestation["schema_version"] != 1 or attestation["kind"] != "h100_identity_smoke":
        raise ValueError("smoke attestation schema/kind mismatch")
    body = {key: attestation[key] for key in ("schema_version", "kind", "payload")}
    actual = hashlib.sha256(_canonical(body)).hexdigest()
    if attestation["sha256"] != actual:
        raise ValueError("smoke attestation self-hash mismatch")
    if actual != _digest(expected_sha256, "external expected digest"):
        raise ValueError("smoke attestation does not match external expected digest")

    payload = attestation["payload"]
    _exact_keys(
        payload,
        {
            "commit_sha",
            "tree_sha",
            "environment_sha256",
            "source_manifest_sha256",
            "repository_binding",
            "nvidia_smi_sha256",
            "gpu_model",
            "driver_version",
            "checkpoint_ceiling_bytes",
            "observed_checkpoint_max_bytes",
            "submission_receipt_sha256",
            "scheduler_terminal_manifest",
            "scheduler_log_manifest",
            "cases",
        },
        "smoke payload",
    )
    if _HEX40.fullmatch(expected_commit_sha or "") is None or payload["commit_sha"] != expected_commit_sha:
        raise ValueError("smoke commit mismatch")
    if _HEX40.fullmatch(expected_tree_sha or "") is None or payload["tree_sha"] != expected_tree_sha:
        raise ValueError("smoke tree mismatch")
    if payload["environment_sha256"] != _digest(
        expected_environment_sha256, "expected environment digest"
    ):
        raise ValueError("smoke environment mismatch")
    if payload["source_manifest_sha256"] != _digest(
        expected_source_manifest_sha256, "expected source manifest digest"
    ):
        raise ValueError("smoke source manifest mismatch")
    repository_binding = _validated_repository_binding(
        payload["repository_binding"],
        expected_commit_sha=expected_commit_sha,
    )
    nvidia_smi_sha256 = _digest(
        payload["nvidia_smi_sha256"], "smoke nvidia-smi executable"
    )
    if expected_nvidia_smi_sha256 is not None and nvidia_smi_sha256 != _digest(
        expected_nvidia_smi_sha256, "expected smoke nvidia-smi executable"
    ):
        raise ValueError("smoke nvidia-smi binding mismatch")
    for key, external, label in (
        ("git_sha256", expected_git_sha256, "Git executable"),
        (
            "repository_identity_sha256",
            expected_repository_identity_sha256,
            "repository identity",
        ),
        ("query_sha256", expected_repository_query_sha256, "repository query"),
    ):
        if external is not None and repository_binding[key] != _digest(
            external, f"expected {label} digest"
        ):
            raise ValueError(f"smoke {label} binding mismatch")
    if not isinstance(expected_gpu_model, str) or not expected_gpu_model or payload["gpu_model"] != expected_gpu_model:
        raise ValueError("smoke GPU model mismatch")
    if "H100" not in payload["gpu_model"]:
        raise ValueError("smoke GPU must be an H100")
    if not isinstance(payload["driver_version"], str) or not payload["driver_version"]:
        raise ValueError("smoke driver version is missing")
    if (
        isinstance(expected_checkpoint_ceiling_bytes, bool)
        or not isinstance(expected_checkpoint_ceiling_bytes, int)
        or expected_checkpoint_ceiling_bytes < 1
        or payload["checkpoint_ceiling_bytes"] != expected_checkpoint_ceiling_bytes
    ):
        raise ValueError("smoke checkpoint ceiling does not match external commitment")
    receipt_sha = _digest(
        payload["submission_receipt_sha256"], "smoke submission receipt"
    )
    if expected_submission_receipt_sha256 is not None and receipt_sha != _digest(
        expected_submission_receipt_sha256, "expected smoke submission receipt"
    ):
        raise ValueError("smoke submission receipt mismatch")
    terminal_manifest = payload["scheduler_terminal_manifest"]
    validate_scheduler_terminal_manifest(
        terminal_manifest,
        expected_submission_receipt_sha256=receipt_sha,
        expected_sacct_sha256=expected_sacct_sha256,
    )
    terminal_by_case = {
        item["case_id"]: item
        for item in terminal_manifest["payload"]["observations"]
    }
    validate_scheduler_log_manifest(payload["scheduler_log_manifest"])

    expected_cases = {case.case_id: case for case in build_matrix()}
    evidence = payload["cases"]
    if not isinstance(evidence, list) or len(evidence) != len(expected_cases):
        raise ValueError("smoke case matrix is incomplete")
    evidence_by_id: dict[str, Mapping[str, Any]] = {}
    exact_evidence_keys = {
        "case_id",
        "arm",
        "stage",
        "world_size",
        "observed_sequence_length",
        "smoke_kind",
        "recompute",
        "active_start_seconds",
        "active_end_seconds",
        "artifact_sha256",
        "artifact_manifest",
        "artifact_identity_sha256",
        "case_result",
        "case_result_sha256",
        "checkpoint_relative_path",
        "checkpoint_sha256",
        "checkpoint_size",
        "source_attestation",
        "source_attestation_sha256",
        "runtime_evidence",
        "runtime_evidence_sha256",
        "action_completion",
        "action_completion_sha256",
        "case_binding_sha256",
        "checkout_commit_sha",
        "checkout_tree_sha",
        "clean_detached",
        "tabicl_attestation_count",
        "python_no_user_site",
        "gpu_window_sha256",
        "gpu_uuids",
        "gpu_sample_counts",
        "gpu_means",
        "gpu_min_gap_seconds",
        "gpu_max_gap_seconds",
    }
    source_attestations: set[str] = set()
    completion_receipts: set[str] = set()
    runtime_evidences: set[str] = set()
    artifact_identities: set[str] = set()
    one_gpu_environments: dict[str, Mapping[str, Any]] = {}
    two_gpu_environments: dict[str, Mapping[str, Any]] = {}
    observed_gpu_models: set[str] = set()
    observed_driver_versions: set[str] = set()
    observed_checkpoint_sizes: list[int] = []
    for item in evidence:
        _exact_keys(item, exact_evidence_keys, "smoke case evidence")
        case_id = item["case_id"]
        if case_id in evidence_by_id:
            raise ValueError("duplicate smoke case evidence")
        evidence_by_id[case_id] = item
        case = expected_cases.get(case_id)
        if case is None:
            raise ValueError("unknown smoke case ID")
        expected_fields = {
            "arm": case.arm,
            "stage": case.stage,
            "world_size": case.world_size,
            "observed_sequence_length": case.observed_sequence_length,
            "smoke_kind": case.smoke_kind,
            "recompute": case.recompute,
            "checkout_commit_sha": expected_commit_sha,
            "checkout_tree_sha": expected_tree_sha,
            "clean_detached": True,
            "tabicl_attestation_count": 1,
            "python_no_user_site": "1",
        }
        for field, expected in expected_fields.items():
            if item[field] != expected:
                raise ValueError(f"smoke case {case_id} {field} mismatch")
        case_result_digest, start, end = validate_case_result(
            item["case_result"], case=case
        )
        if item["case_result_sha256"] != case_result_digest:
            raise ValueError("case result digest mismatch")
        if item["active_start_seconds"] != start or item["active_end_seconds"] != end:
            raise ValueError("smoke active span is not derived from case result")
        result_payload = item["case_result"]["payload"]
        for field in (
            "checkpoint_relative_path",
            "checkpoint_sha256",
            "checkpoint_size",
        ):
            if item[field] != result_payload[field]:
                raise ValueError("checkpoint evidence is not derived from case result")
        if result_payload["checkpoint_ceiling_bytes"] != payload["checkpoint_ceiling_bytes"]:
            raise ValueError("case checkpoint ceiling differs from smoke commitment")
        if case.stage is not None:
            observed_checkpoint_sizes.append(item["checkpoint_size"])
        artifact_digest = _digest(item["artifact_sha256"], "case artifact digest")
        artifact_identity = _digest(
            item["artifact_identity_sha256"], "case artifact identity digest"
        )
        source_digest = validate_source_attestation(
            item["source_attestation"],
            case_id=case_id,
            artifact_identity_sha256=artifact_identity,
            commit_sha=expected_commit_sha,
            tree_sha=expected_tree_sha,
            source_manifest_sha256=payload["source_manifest_sha256"],
        )
        if item["source_attestation_sha256"] != source_digest:
            raise ValueError("case source attestation digest mismatch")
        runtime_digest = validate_runtime_evidence(
            item["runtime_evidence"],
            case=case,
            commit_sha=expected_commit_sha,
            tree_sha=expected_tree_sha,
            source_manifest_sha256=payload["source_manifest_sha256"],
            expected_nvidia_smi_sha256=nvidia_smi_sha256,
        )
        scheduler_binding = item["runtime_evidence"]["payload"]["scheduler_binding"]
        terminal_observation = terminal_by_case.get(case_id)
        if (
            terminal_observation is None
            or terminal_observation["job_id"] != scheduler_binding["job_id"]
            or terminal_observation["observed_cluster"]
            != scheduler_binding["slurm_cluster_name"]
            or result_payload["slurm_job_id"] != scheduler_binding["job_id"]
            or result_payload["requested_resource_sha256"]
            != scheduler_binding["requested_resource_sha256"]
        ):
            raise ValueError("case result differs from runtime scheduler binding")
        if item["runtime_evidence_sha256"] != runtime_digest:
            raise ValueError("case runtime evidence digest mismatch")
        completion_digest = validate_action_completion(
            item["action_completion"],
            case_id=case_id,
            artifact_identity_sha256=artifact_identity,
            source_attestation_sha256=source_digest,
            runtime_evidence_sha256=runtime_digest,
            commit_sha=expected_commit_sha,
            tree_sha=expected_tree_sha,
            source_manifest_sha256=payload["source_manifest_sha256"],
            scheduler_binding=scheduler_binding,
        )
        if item["action_completion_sha256"] != completion_digest:
            raise ValueError("case action completion digest mismatch")
        embedded_raw = {
            "source-attestation.json": _canonical_file_sha256(
                item["source_attestation"]
            ),
            "runtime-evidence.json": _canonical_file_sha256(
                item["runtime_evidence"]
            ),
            "action-completion.json": _canonical_file_sha256(
                item["action_completion"]
            ),
            "compute/case-result.json": _canonical_file_sha256(item["case_result"]),
        }
        actual_artifact_digest = validate_artifact_manifest(
            item["artifact_manifest"], case=case, embedded_raw_sha256=embedded_raw
        )
        if artifact_digest != actual_artifact_digest:
            raise ValueError("case artifact manifest digest mismatch")
        if case.stage is not None:
            checkpoint_name = f"compute/{item['checkpoint_relative_path']}"
            checkpoint_entries = {
                entry["name"]: entry for entry in item["artifact_manifest"]["files"]
            }
            if checkpoint_entries[checkpoint_name] != {
                "name": checkpoint_name,
                "sha256": item["checkpoint_sha256"],
                "size": item["checkpoint_size"],
            }:
                raise ValueError("artifact manifest checkpoint facts mismatch")
        case_binding = {
            "case_id": case_id,
            "source_attestation_sha256": source_digest,
            "runtime_evidence_sha256": runtime_digest,
            "action_completion_sha256": completion_digest,
            "artifact_identity_sha256": artifact_identity,
            "artifact_sha256": artifact_digest,
        }
        if item["case_binding_sha256"] != hashlib.sha256(_canonical(case_binding)).hexdigest():
            raise ValueError("case evidence is not bound to source and actual artifact")
        if source_digest in source_attestations:
            raise ValueError("one case cannot reuse another case's attestation")
        if runtime_digest in runtime_evidences:
            raise ValueError("one case cannot reuse another case's runtime evidence")
        if completion_digest in completion_receipts:
            raise ValueError("one case cannot reuse another case's completion receipt")
        if artifact_identity in artifact_identities:
            raise ValueError("one case cannot reuse another case's artifact identity")
        source_attestations.add(source_digest)
        runtime_evidences.add(runtime_digest)
        completion_receipts.add(completion_digest)
        artifact_identities.add(artifact_identity)

        environment = item["runtime_evidence"]["payload"]["environment"]
        environment_digest = environment["sha256"]
        target = one_gpu_environments if case.world_size == 1 else two_gpu_environments
        target[environment_digest] = environment
        for device in item["runtime_evidence"]["payload"]["gpu_devices"]:
            observed_gpu_models.add(device["name"])
            observed_driver_versions.add(device["driver_version"])

        gpu_window = {
            "gpu_uuids": item["gpu_uuids"],
            "gpu_sample_counts": item["gpu_sample_counts"],
            "gpu_means": item["gpu_means"],
            "gpu_min_gap_seconds": item["gpu_min_gap_seconds"],
            "gpu_max_gap_seconds": item["gpu_max_gap_seconds"],
        }
        if case.utilization_required:
            if (
                not isinstance(item["gpu_uuids"], list)
                or len(item["gpu_uuids"]) != case.world_size
                or len(set(item["gpu_uuids"])) != case.world_size
                or set(item["gpu_sample_counts"]) != set(item["gpu_uuids"])
                or set(item["gpu_means"]) != set(item["gpu_uuids"])
                or any(
                    isinstance(count, bool) or not isinstance(count, int) or count < 10
                    for count in item["gpu_sample_counts"].values()
                )
                or any(
                    isinstance(mean, bool)
                    or not isinstance(mean, (int, float))
                    or not math.isfinite(float(mean))
                    or mean < 80.0
                    for mean in item["gpu_means"].values()
                )
                or not isinstance(item["gpu_min_gap_seconds"], (int, float))
                or not isinstance(item["gpu_max_gap_seconds"], (int, float))
                or item["gpu_min_gap_seconds"] < 0.5
                or item["gpu_max_gap_seconds"] > 1.5
            ):
                raise ValueError("utilization-required case has an invalid GPU window")
            if item["gpu_window_sha256"] != hashlib.sha256(_canonical(gpu_window)).hexdigest():
                raise ValueError("GPU active-window digest mismatch")
            runtime_uuids = {
                device["uuid"]
                for device in item["runtime_evidence"]["payload"]["gpu_devices"]
            }
            if set(item["gpu_uuids"]) != runtime_uuids:
                raise ValueError("GPU window UUIDs differ from runtime evidence")
        elif gpu_window != {
            "gpu_uuids": [],
            "gpu_sample_counts": {},
            "gpu_means": {},
            "gpu_min_gap_seconds": None,
            "gpu_max_gap_seconds": None,
        } or item["gpu_window_sha256"] is not None:
            raise ValueError("functional/special case must make no utilization claim")
    if set(evidence_by_id) != set(expected_cases):
        raise ValueError("smoke case ID set mismatch")
    if len(one_gpu_environments) != 1 or payload["environment_sha256"] not in one_gpu_environments:
        raise ValueError("one-GPU runtime environments do not agree with top-level digest")
    if len(two_gpu_environments) != 1:
        raise ValueError("two-GPU NCCL runtime environment is missing or inconsistent")
    one_environment = next(iter(one_gpu_environments.values()))
    two_environment = next(iter(two_gpu_environments.values()))
    if one_environment["sha256"] == two_environment["sha256"]:
        raise ValueError("one-GPU and two-GPU environment digests must differ")
    one_payload = dict(one_environment["payload"])
    two_payload = dict(two_environment["payload"])
    if one_payload.pop("visible_cuda_device_count") != 1:
        raise ValueError("one-GPU environment count mismatch")
    if two_payload.pop("visible_cuda_device_count") != 2 or one_payload != two_payload:
        raise ValueError("NCCL environment may differ only by visible CUDA count")
    if observed_gpu_models != {payload["gpu_model"]}:
        raise ValueError("top-level GPU model is not derived from runtime evidence")
    if observed_driver_versions != {payload["driver_version"]}:
        raise ValueError("top-level driver is not derived from runtime evidence")
    if (
        len(observed_checkpoint_sizes) != 9
        or payload["observed_checkpoint_max_bytes"] != max(observed_checkpoint_sizes)
        or payload["observed_checkpoint_max_bytes"] > payload["checkpoint_ceiling_bytes"]
    ):
        raise ValueError("top-level observed checkpoint maximum is invalid")
    return {
        "schema_version": 1,
        "valid": True,
        "sha256": actual,
        "case_count": len(evidence_by_id),
        "gpu_model": payload["gpu_model"],
        "driver_version": payload["driver_version"],
        "nvidia_smi_sha256": nvidia_smi_sha256,
    }


def _open_directory_nofollow(path: Path) -> int:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ValueError("artifact directory must be a normalized absolute path")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ValueError("no-follow directory traversal is unavailable")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(os.path.sep, flags)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except OSError as error:
        os.close(fd)
        raise ValueError("artifact directory contains a symlink or invalid component") from error


def _open_trusted_scheduler_executable(
    path: Path,
    expected_sha256: str,
    *,
    command_name: str = "sacct",
) -> tuple[int, os.stat_result]:
    if command_name not in {"sacct", "scontrol"}:
        raise ValueError("trusted scheduler command name is unsupported")
    expected_sha256 = _digest(
        expected_sha256, f"trusted {command_name} executable"
    )
    if (
        not path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or path.name != command_name
    ):
        raise ValueError(
            f"trusted {command_name} path must be a normalized absolute {command_name} path"
        )
    parent_fd = _open_directory_nofollow(path.parent)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path.name, flags, dir_fd=parent_fd)
    except OSError as error:
        os.close(parent_fd)
        raise ValueError(
            f"trusted {command_name} cannot be opened without following links"
        ) from error
    os.close(parent_fd)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_mode & 0o111 == 0
            or before.st_size > 128 * (1 << 20)
        ):
            raise ValueError(
                f"trusted {command_name} must be a bounded executable regular file"
            )
        digest = hashlib.sha256()
        while chunk := os.read(fd, 1 << 20):
            digest.update(chunk)
        after = os.fstat(fd)
        stable = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_ctime_ns",
            "st_mtime_ns",
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable):
            raise ValueError(f"trusted {command_name} changed while being hashed")
        path_after = os.stat(path, follow_symlinks=False)
        if (
            path_after.st_dev != after.st_dev
            or path_after.st_ino != after.st_ino
            or not stat.S_ISREG(path_after.st_mode)
            or digest.hexdigest() != expected_sha256
        ):
            raise ValueError(
                f"trusted {command_name} path identity or digest mismatch"
            )
        os.lseek(fd, 0, os.SEEK_SET)
        return fd, before
    except BaseException:
        os.close(fd)
        raise


def _revalidate_trusted_scheduler_executable(
    fd: int,
    path: Path,
    before: os.stat_result,
    expected_sha256: str,
    *,
    command_name: str = "sacct",
) -> None:
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(fd, 1 << 20):
        digest.update(chunk)
    after = os.fstat(fd)
    path_after = os.stat(path, follow_symlinks=False)
    stable = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_ctime_ns",
        "st_mtime_ns",
    )
    if (
        any(getattr(before, field) != getattr(after, field) for field in stable)
        or path_after.st_dev != after.st_dev
        or path_after.st_ino != after.st_ino
        or digest.hexdigest() != expected_sha256
    ):
        raise ValueError(
            f"trusted {command_name} changed during scheduler reconciliation"
        )


def capture_scheduler_terminal_manifest(
    *,
    sacct_path: Path,
    expected_sacct_sha256: str,
    submission_receipt_sha256: str,
    expected_job_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove all twelve allocations are terminal before scheduler-log hashing."""

    receipt_sha = _digest(
        submission_receipt_sha256, "terminal submission receipt"
    )
    sacct_sha = _digest(expected_sacct_sha256, "terminal sacct executable")
    expected_cases = build_matrix()
    if set(expected_job_bindings) != {case.case_id for case in expected_cases}:
        raise ValueError("terminal scheduler job-binding set mismatch")
    fd, before = _open_trusted_scheduler_executable(sacct_path, sacct_sha)
    observations: list[dict[str, Any]] = []
    try:
        for case in expected_cases:
            binding = expected_job_bindings[case.case_id]
            job_id = binding.get("job_id")
            receipt_cluster = binding.get("cluster")
            if (
                not isinstance(job_id, str)
                or re.fullmatch(r"[1-9][0-9]{0,19}", job_id) is None
                or (
                    receipt_cluster is not None
                    and (
                        not isinstance(receipt_cluster, str)
                        or re.fullmatch(r"[A-Za-z0-9._-]+", receipt_cluster) is None
                    )
                )
            ):
                raise ValueError("terminal scheduler receipt binding is malformed")
            semantic_argv = [
                "sacct",
                *(
                    [f"--clusters={receipt_cluster}"]
                    if receipt_cluster is not None
                    else []
                ),
                "--noheader",
                "--allocations",
                f"--jobs={job_id}",
                "--format=JobIDRaw,Cluster,State,ExitCode,DerivedExitCode",
                "--parsable2",
            ]
            executable_argv = [f"/proc/self/fd/{fd}", *semantic_argv[1:]]
            completed = subprocess.run(
                executable_argv,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=SCHEDULER_QUERY_TIMEOUT_SECONDS,
                pass_fds=(fd,),
            )
            stdout = completed.stdout
            stderr = completed.stderr
            if len(stdout) > 1_000_000 or len(stderr) > 1_000_000:
                raise ValueError("sacct terminal query exceeded its byte ceiling")
            try:
                lines = stdout.decode("utf-8", errors="strict").splitlines()
                stderr_text = stderr.decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                raise ValueError("sacct terminal query emitted non-UTF-8 output") from error
            if completed.returncode != 0 or stderr_text or len(lines) != 1:
                raise ValueError("sacct terminal query was nonzero, noisy, missing, or duplicate")
            if stdout != (lines[0] + "\n").encode("utf-8"):
                raise ValueError("sacct terminal query output is not canonical")
            fields = lines[0].split("|")
            if len(fields) != 5:
                raise ValueError("sacct terminal query row is malformed or banner-spoofed")
            (
                observed_job_id,
                observed_cluster,
                state,
                exit_code,
                derived_exit_code,
            ) = fields
            if (
                observed_job_id != job_id
                or re.fullmatch(r"[A-Za-z0-9._-]+", observed_cluster) is None
                or (
                    receipt_cluster is not None
                    and observed_cluster != receipt_cluster
                )
                or state != "COMPLETED"
                or exit_code != "0:0"
                or derived_exit_code != "0:0"
            ):
                raise ValueError(
                    "sacct allocation is not uniquely bound and exact COMPLETED/0:0"
                )
            observations.append(
                {
                    "case_id": case.case_id,
                    "job_id": job_id,
                    "receipt_cluster": receipt_cluster,
                    "observed_cluster": observed_cluster,
                    "state": state,
                    "exit_code": exit_code,
                    "derived_exit_code": derived_exit_code,
                    "query_returncode": completed.returncode,
                    "query_argv_sha256": hashlib.sha256(
                        _canonical(semantic_argv)
                    ).hexdigest(),
                    "query_stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                    "query_stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                }
            )
        _revalidate_trusted_scheduler_executable(
            fd, sacct_path, before, sacct_sha
        )
    finally:
        os.close(fd)
    manifest = make_scheduler_terminal_manifest(
        observations,
        submission_receipt_sha256=receipt_sha,
        sacct_sha256=sacct_sha,
    )
    validate_scheduler_terminal_manifest(
        manifest,
        expected_submission_receipt_sha256=receipt_sha,
        expected_sacct_sha256=sacct_sha,
        expected_job_bindings=expected_job_bindings,
    )
    return manifest


def _read_regular_bytes(path: Path, *, max_bytes: int) -> bytes:
    if max_bytes < 1:
        raise ValueError("artifact byte ceiling must be positive")
    candidate = path if path.is_absolute() else Path.cwd() / path
    candidate = Path(os.path.abspath(candidate))
    parent_fd = _open_directory_nofollow(candidate.parent)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(candidate.name, flags, dir_fd=parent_fd)
    except OSError as error:
        os.close(parent_fd)
        raise ValueError(f"cannot open required artifact safely: {path.name}") from error
    os.close(parent_fd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise ValueError(f"artifact is not regular or exceeds ceiling: {path.name}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        snapshot = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if len(raw) != before.st_size or any(
            getattr(before, field) != getattr(after, field) for field in snapshot
        ):
            raise ValueError(f"artifact changed while being read: {path.name}")
        return raw
    finally:
        os.close(fd)


def _hash_regular_file(path: Path, *, max_bytes: int) -> tuple[str, int]:
    if max_bytes < 1:
        raise ValueError("artifact byte ceiling must be positive")
    candidate = path if path.is_absolute() else Path.cwd() / path
    candidate = Path(os.path.abspath(candidate))
    parent_fd = _open_directory_nofollow(candidate.parent)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(candidate.name, flags, dir_fd=parent_fd)
    except OSError as error:
        os.close(parent_fd)
        raise ValueError(f"cannot open required artifact safely: {path.name}") from error
    os.close(parent_fd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise ValueError(f"artifact is not regular or exceeds ceiling: {path.name}")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(fd)
        snapshot = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if total != before.st_size or any(
            getattr(before, field) != getattr(after, field) for field in snapshot
        ):
            raise ValueError(f"artifact changed while being hashed: {path.name}")
        return digest.hexdigest(), total
    finally:
        os.close(fd)


def _strict_json(raw: bytes, *, where: str) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {where}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token in {where}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact {where}") from error
    if not isinstance(value, Mapping) or raw != _canonical(value) + b"\n":
        raise ValueError(f"JSON artifact is not canonical newline-terminated JSON: {where}")
    return value


def validate_gate_submission_receipt(
    path: Path,
    *,
    expected_commit_sha: str,
    expected_tree_sha: str,
    expected_source_manifest_sha256: str,
    expected_checkpoint_ceiling_bytes: int,
    max_bytes: int,
) -> dict[str, Any]:
    """Validate the released twelve-job transaction before case assembly."""

    value = _strict_json(
        _read_regular_bytes(path, max_bytes=max_bytes),
        where="H100 gate submission receipt",
    )
    _exact_keys(
        value,
        {"schema_version", "kind", "payload", "sha256"},
        "H100 gate submission receipt",
    )
    if value["schema_version"] != 1 or value["kind"] != "h100_gate_submission_receipt":
        raise ValueError("H100 gate submission receipt schema/kind mismatch")
    body = {key: value[key] for key in ("schema_version", "kind", "payload")}
    if value["sha256"] != hashlib.sha256(_canonical(body)).hexdigest():
        raise ValueError("H100 gate submission receipt self-hash mismatch")
    payload = value["payload"]
    _exact_keys(
        payload,
        {
            "transaction_id",
            "validation_id",
            "commit_sha",
            "tree_sha",
            "source_manifest_sha256",
            "repository_binding",
            "environment_sha256_by_world_size",
            "environment_transaction",
            "checkpoint_ceiling_bytes",
            "limits",
            "capacity",
            "scheduler",
            "scheduler_command_sha256",
            "nvidia_smi_sha256",
            "jobs_held_at_publication",
            "all_jobs_released",
            "held_plan_sha256",
            "transaction_journal_sha256",
            "transaction_journal_events",
            "cases",
        },
        "H100 gate submission receipt payload",
    )
    validation_id = payload["validation_id"]
    if _SAFE_VALIDATION_ID.fullmatch(validation_id or "") is None:
        raise ValueError("submission receipt validation ID is malformed")
    transaction_id = payload["transaction_id"]
    if not isinstance(transaction_id, str) or re.fullmatch(r"[0-9a-f]{32}", transaction_id) is None:
        raise ValueError("submission receipt transaction ID is malformed")
    if payload["commit_sha"] != expected_commit_sha:
        raise ValueError("submission receipt commit mismatch")
    if payload["tree_sha"] != expected_tree_sha:
        raise ValueError("submission receipt tree mismatch")
    if payload["source_manifest_sha256"] != _digest(
        expected_source_manifest_sha256, "expected receipt source manifest"
    ):
        raise ValueError("submission receipt source manifest mismatch")
    repository_binding = _validated_repository_binding(
        payload["repository_binding"],
        expected_commit_sha=expected_commit_sha,
    )
    nvidia_smi_sha256 = _digest(
        payload["nvidia_smi_sha256"], "submission receipt nvidia-smi"
    )
    if payload["checkpoint_ceiling_bytes"] != expected_checkpoint_ceiling_bytes:
        raise ValueError("submission receipt checkpoint ceiling mismatch")
    if payload["jobs_held_at_publication"] is not True or payload["all_jobs_released"] is not True:
        raise ValueError("submission receipt does not prove a released held transaction")
    held_plan_sha256 = _digest(payload["held_plan_sha256"], "held transaction plan")
    recovery_path = path.parent / "rollback-recovery.json"
    if recovery_path.exists() or recovery_path.is_symlink():
        raise ValueError("released submission receipt coexists with rollback recovery")
    held = _strict_json(
        _read_regular_bytes(path.parent / "held-plan.json", max_bytes=max_bytes),
        where="H100 gate held plan",
    )
    _exact_keys(
        held,
        {"schema_version", "kind", "payload", "sha256"},
        "H100 gate held plan",
    )
    held_body = {key: held[key] for key in ("schema_version", "kind", "payload")}
    if (
        held["schema_version"] != 1
        or held["kind"] != "h100_gate_held_plan"
        or held["sha256"] != hashlib.sha256(_canonical(held_body)).hexdigest()
        or held["sha256"] != held_plan_sha256
    ):
        raise ValueError("H100 gate held plan integrity mismatch")
    expected_held_payload = dict(payload)
    expected_held_payload.pop("all_jobs_released")
    expected_held_payload.pop("held_plan_sha256")
    expected_held_payload.pop("transaction_journal_sha256")
    expected_held_payload.pop("transaction_journal_events")
    if held["payload"] != expected_held_payload:
        raise ValueError("released receipt differs from its held transaction plan")

    journal_raw = _read_regular_bytes(
        path.parent / "transaction-journal.jsonl", max_bytes=max_bytes
    )
    if hashlib.sha256(journal_raw).hexdigest() != _digest(
        payload["transaction_journal_sha256"], "transaction journal"
    ):
        raise ValueError("transaction journal digest mismatch")
    journal_lines = journal_raw.splitlines(keepends=True)
    event_count = payload["transaction_journal_events"]
    if (
        isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count < 1
        or len(journal_lines) != event_count
    ):
        raise ValueError("transaction journal event count mismatch")
    previous = "0" * 64
    for sequence, raw_line in enumerate(journal_lines, start=1):
        record = _strict_json(raw_line, where=f"transaction journal event {sequence}")
        _exact_keys(
            record,
            {
                "schema_version",
                "transaction_id",
                "sequence",
                "event",
                "payload",
                "previous_sha256",
                "sha256",
            },
            "transaction journal event",
        )
        body = {key: record[key] for key in record if key != "sha256"}
        if (
            record["schema_version"] != 1
            or record["transaction_id"] != transaction_id
            or record["sequence"] != sequence
            or record["previous_sha256"] != previous
            or not isinstance(record["event"], str)
            or not record["event"]
            or record["sha256"] != hashlib.sha256(_canonical(body)).hexdigest()
        ):
            raise ValueError("transaction journal hash chain mismatch")
        previous = record["sha256"]

    environments = payload["environment_sha256_by_world_size"]
    _exact_keys(environments, {"1", "2"}, "receipt environment map")
    environments = {
        key: _digest(value, f"receipt {key}-GPU environment")
        for key, value in environments.items()
    }
    if environments["1"] == environments["2"]:
        raise ValueError("one- and two-GPU expected environment digests must differ")
    environment_transaction = _validated_environment_transaction_summary(
        payload["environment_transaction"],
        expected_commit_sha=expected_commit_sha,
        expected_tree_sha=expected_tree_sha,
        expected_git_sha256=repository_binding["git_sha256"],
        expected_environments=environments,
    )

    limits = payload["limits"]
    _exact_keys(
        limits,
        {
            "checkpoint_ceiling_bytes",
            "run_log_ceiling_bytes",
            "gpu_monitor_ceiling_bytes",
            "attestation_ceiling_bytes",
            "receipt_ceiling_bytes",
            "scheduler_log_ceiling_bytes",
        },
        "receipt gate limits",
    )
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1
        for item in limits.values()
    ) or limits["checkpoint_ceiling_bytes"] != expected_checkpoint_ceiling_bytes:
        raise ValueError("receipt gate limits are invalid")
    capacity = payload["capacity"]
    if not isinstance(capacity, Mapping):
        raise ValueError("receipt gate capacity is malformed")
    allocation_unit = capacity.get("allocation_unit_bytes")
    if (
        isinstance(allocation_unit, bool)
        or not isinstance(allocation_unit, int)
        or allocation_unit < 1
    ):
        raise ValueError("receipt gate allocation unit is invalid")
    slots = {
        "checkpoint": 9,
        "case_log": 12,
        "case_attestation": 36,
        "final_attestation": 1,
        "case_result": 12,
        "gpu_raw": 12,
        "gpu_summary": 6,
        "controller_receipt": 4,
        "scheduler_log": 24,
    }
    if sum(slots.values()) != 116:
        raise ValueError("receipt gate file-slot accounting drifted")
    ceilings = {
        "checkpoint": limits["checkpoint_ceiling_bytes"],
        "case_log": limits["run_log_ceiling_bytes"],
        "case_attestation": limits["attestation_ceiling_bytes"],
        "final_attestation": limits["attestation_ceiling_bytes"],
        "case_result": 1 << 20,
        "gpu_raw": limits["gpu_monitor_ceiling_bytes"],
        "gpu_summary": limits["run_log_ceiling_bytes"],
        "controller_receipt": limits["receipt_ceiling_bytes"],
        "scheduler_log": limits["scheduler_log_ceiling_bytes"],
    }

    def round_up(value: int) -> int:
        return ((value + allocation_unit - 1) // allocation_unit) * allocation_unit

    logical = {
        category: slots[category] * ceiling
        for category, ceiling in ceilings.items()
    }
    physical = {
        category: slots[category] * round_up(ceiling)
        for category, ceiling in ceilings.items()
    }
    total_file_physical = sum(physical.values())
    total_dirents = 1 + 27 + 116 + 13
    base_directory_physical = 28 * allocation_unit
    dirent_physical = total_dirents * allocation_unit
    total_directory_physical = base_directory_physical + dirent_physical
    reserve = 20 * (1 << 30)
    expected_capacity = {
        "allocation_unit_bytes": allocation_unit,
        "reserve_bytes": reserve,
        "reserve_physical_bytes": round_up(reserve),
        "file_slots_by_category": slots,
        "total_file_slots": 116,
        **{f"{category}_bytes": value for category, value in logical.items()},
        "worst_case_gate_bytes": sum(logical.values()),
        **{
            f"{category}_physical_bytes": value
            for category, value in physical.items()
        },
        "total_file_physical_bytes": total_file_physical,
        "directory_slot_count": 28,
        "base_directory_physical_bytes": base_directory_physical,
        "artifact_root_dirent_slots": 1,
        "child_directory_dirent_slots": 27,
        "file_dirent_slots": 116,
        "atomic_extra_dirent_slots": 13,
        "total_dirent_slots": total_dirents,
        "dirent_physical_bytes": dirent_physical,
        "total_directory_physical_bytes": total_directory_physical,
        "worst_case_gate_physical_bytes": (
            total_file_physical + total_directory_physical
        ),
    }
    expected_capacity["required_free_bytes"] = (
        expected_capacity["reserve_physical_bytes"]
        + expected_capacity["worst_case_gate_physical_bytes"]
    )
    _exact_keys(
        capacity,
        {*expected_capacity, "available_blocks", "available_bytes"},
        "receipt gate capacity",
    )
    if any(capacity.get(key) != item for key, item in expected_capacity.items()):
        raise ValueError("receipt gate physical capacity formula mismatch")
    available_blocks = capacity["available_blocks"]
    available_bytes = capacity["available_bytes"]
    if (
        isinstance(available_blocks, bool)
        or not isinstance(available_blocks, int)
        or available_blocks < 0
        or isinstance(available_bytes, bool)
        or not isinstance(available_bytes, int)
        or available_bytes != available_blocks * allocation_unit
        or available_bytes < capacity["required_free_bytes"]
    ):
        raise ValueError("receipt gate capacity snapshot was insufficient")

    expected_scheduler = _scheduler_contract()
    if payload["scheduler"] != expected_scheduler:
        raise ValueError("submission receipt scheduler/resource contract mismatch")
    scheduler_digests = payload["scheduler_command_sha256"]
    _exact_keys(
        scheduler_digests,
        {"sbatch", "scontrol", "scancel", "squeue", "sacct"},
        "submission receipt scheduler command digests",
    )
    for name, digest in scheduler_digests.items():
        _digest(digest, f"submission receipt scheduler {name}")

    cases = payload["cases"]
    matrix = build_matrix()
    if not isinstance(cases, list) or len(cases) != len(matrix):
        raise ValueError("submission receipt case matrix is incomplete")
    identities: dict[str, str] = {}
    job_bindings: dict[str, dict[str, Any]] = {}
    job_ids: set[str] = set()
    for expected_case, item in zip(matrix, cases):
        _exact_keys(
            item,
            {
                "case_id",
                "world_size",
                "artifact_identity_sha256",
                "requested_resource",
                "requested_resource_sha256",
                "job_id",
                "cluster",
            },
            "submission receipt case",
        )
        if item["case_id"] != expected_case.case_id or item["world_size"] != expected_case.world_size:
            raise ValueError("submission receipt case order/world-size mismatch")
        job_id = item["job_id"]
        if not isinstance(job_id, str) or re.fullmatch(r"[1-9][0-9]{0,19}", job_id) is None:
            raise ValueError("submission receipt job ID is malformed")
        if job_id in job_ids:
            raise ValueError("submission receipt contains a duplicate job ID")
        job_ids.add(job_id)
        requested_resource = gate_requested_resource(expected_case)
        requested_resource_sha = gate_requested_resource_sha256(expected_case)
        if (
            item["requested_resource"] != requested_resource
            or item["requested_resource_sha256"] != requested_resource_sha
        ):
            raise ValueError("submission receipt requested-resource mismatch")
        cluster = item["cluster"]
        if cluster is not None and (
            not isinstance(cluster, str)
            or re.fullmatch(r"[A-Za-z0-9._-]+", cluster) is None
        ):
            raise ValueError("submission receipt cluster suffix is malformed")
        expected_identity = submission_artifact_identity(
            validation_id=validation_id,
            case_id=expected_case.case_id,
            world_size=expected_case.world_size,
            commit_sha=expected_commit_sha,
            tree_sha=expected_tree_sha,
            source_manifest_sha256=expected_source_manifest_sha256,
            expected_environment_sha256=environments[str(expected_case.world_size)],
            requested_resource_sha256=requested_resource_sha,
            checkpoint_ceiling_bytes=expected_checkpoint_ceiling_bytes,
        )
        if item["artifact_identity_sha256"] != expected_identity:
            raise ValueError("submission receipt artifact identity mismatch")
        identities[expected_case.case_id] = expected_identity
        job_bindings[expected_case.case_id] = {
            "job_id": job_id,
            "cluster": cluster,
            "requested_resource": requested_resource,
            "requested_resource_sha256": requested_resource_sha,
            "held_plan_sha256": held_plan_sha256,
            "scontrol_sha256": scheduler_digests["scontrol"],
        }
    return {
        "sha256": value["sha256"],
        "validation_id": validation_id,
        "repository_binding": repository_binding,
        "artifact_identities": identities,
        "job_bindings": job_bindings,
        "environment_sha256_by_world_size": environments,
        "environment_transaction": environment_transaction,
        "scheduler_log_ceiling_bytes": limits["scheduler_log_ceiling_bytes"],
        "final_attestation_ceiling_bytes": limits["attestation_ceiling_bytes"],
        "sacct_sha256": scheduler_digests["sacct"],
        "nvidia_smi_sha256": nvidia_smi_sha256,
    }


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixed validation helper: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _validated_repository_binding(
    value: Mapping[str, Any] | Any,
    *,
    expected_commit_sha: str,
) -> dict[str, Any]:
    helper = _load_module(
        "_h100_exact_git_repository",
        Path(__file__).with_name("verify_git_repository.py"),
    )
    if not isinstance(value, Mapping):
        raise ValueError("repository binding is malformed")
    git_sha256 = value.get("git_sha256")
    return helper.validate_repository_binding(
        value,
        expected_commit_sha=expected_commit_sha,
        expected_git_sha256=git_sha256,
    )


def _validated_environment_transaction_summary(
    value: Mapping[str, Any] | Any,
    *,
    expected_commit_sha: str,
    expected_tree_sha: str,
    expected_git_sha256: str,
    expected_environments: Mapping[str, str],
) -> dict[str, Any]:
    """Validate the path-free exact-T environment publication proof."""

    _exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "completion_raw_sha256",
            "transaction_sha256",
            "source_commit_sha",
            "source_tree_sha",
            "git_sha256",
            "one_gpu_environment_sha256",
            "two_gpu_environment_sha256",
            "inventory_sha256",
            "installed_distributions_sha256",
            "capture_visible_cuda_device_count",
            "output_raw_sha256_by_role",
        },
        "environment transaction verification summary",
    )
    if (
        value["schema_version"] != 1
        or value["kind"] != "formal_environment_transaction_verification"
        or value["source_commit_sha"] != expected_commit_sha
        or value["source_tree_sha"] != expected_tree_sha
        or value["git_sha256"] != _digest(
            expected_git_sha256, "expected environment transaction Git"
        )
        or value["one_gpu_environment_sha256"] != expected_environments["1"]
        or value["two_gpu_environment_sha256"] != expected_environments["2"]
    ):
        raise ValueError("environment transaction verification binding mismatch")
    for key in (
        "completion_raw_sha256",
        "transaction_sha256",
        "one_gpu_environment_sha256",
        "two_gpu_environment_sha256",
        "inventory_sha256",
        "installed_distributions_sha256",
    ):
        _digest(value[key], f"environment transaction {key}")
    capture_count = value["capture_visible_cuda_device_count"]
    if (
        isinstance(capture_count, bool)
        or not isinstance(capture_count, int)
        or capture_count < 0
    ):
        raise ValueError("environment transaction capture GPU count is invalid")
    output_digests = value["output_raw_sha256_by_role"]
    _exact_keys(
        output_digests,
        {"one_gpu", "two_gpu", "inventory"},
        "environment transaction raw-output map",
    )
    for role, digest in output_digests.items():
        _digest(digest, f"environment transaction {role} raw output")
    return {
        **dict(value),
        "output_raw_sha256_by_role": dict(output_digests),
    }


def _artifact_file(name: str, raw: bytes) -> dict[str, Any]:
    return {"name": name, "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}


def _publish_no_replace(path: Path, value: Mapping[str, Any], *, max_bytes: int) -> None:
    raw = _canonical(value) + b"\n"
    if max_bytes < 1 or len(raw) > max_bytes:
        raise ValueError("smoke attestation exceeds configured byte ceiling")
    if path.exists() or path.is_symlink():
        raise ValueError("smoke attestation output is no-replace")
    parent_fd = _open_directory_nofollow(path.parent)
    os.close(parent_fd)
    temporary: Path | None = None
    try:
        fd, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(raw_path)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        temporary = None
        directory_fd = _open_directory_nofollow(path.parent)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _gpu_window_from_raw(
    csv_raw: bytes,
    jsonl_raw: bytes,
    summary_raw: bytes,
    *,
    expected_gpu_count: int,
    expected_start_seconds: float,
    expected_end_seconds: float,
) -> dict[str, Any]:
    monitor = _load_module(
        f"_h100_monitor_contract_{hashlib.sha256(jsonl_raw).hexdigest()}",
        Path(__file__).with_name("monitor_formal_identity.py"),
    )
    csv_records, csv_invalid = monitor.parse_gpu_records(csv_raw.decode("utf-8"))
    jsonl_records, jsonl_invalid = monitor.parse_gpu_records(jsonl_raw.decode("utf-8"))
    if csv_invalid or jsonl_invalid or len(csv_records) != len(jsonl_records):
        raise ValueError("GPU raw artifacts contain invalid or inconsistent records")
    for csv_record, jsonl_record in zip(csv_records, jsonl_records):
        if (
            csv_record.kind != jsonl_record.kind
            or csv_record.gpu_uuid != jsonl_record.gpu_uuid
            or csv_record.utilization_percent != jsonl_record.utilization_percent
            or not math.isclose(
                csv_record.monotonic_seconds,
                jsonl_record.monotonic_seconds,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
        ):
            raise ValueError("GPU CSV and JSONL records disagree")
    starts = [record for record in jsonl_records if record.kind == "training_start"]
    if len(starts) != 1 or starts[0].expected_gpu_uuids is None:
        raise ValueError("GPU JSONL lacks one UUID-bound start marker")
    ends = [record for record in jsonl_records if record.kind == "training_end"]
    if (
        len(ends) != 1
        or starts[0].monotonic_seconds != expected_start_seconds
        or ends[0].monotonic_seconds != expected_end_seconds
    ):
        raise ValueError("GPU active markers differ from harness training signals")
    expected_uuids = tuple(starts[0].expected_gpu_uuids)
    result = monitor.evaluate_gpu_window(
        jsonl_records,
        utilization_required=True,
        expected_gpu_uuids=expected_uuids,
        expected_gpu_count=expected_gpu_count,
    )
    if not result.complete or result.issues:
        codes = sorted(issue.code for issue in result.issues)
        raise ValueError(f"GPU raw active window failed: {codes}")
    by_uuid: dict[str, list[float]] = {gpu_uuid: [] for gpu_uuid in expected_uuids}
    for record in jsonl_records:
        if record.kind == "sample" and record.gpu_uuid in by_uuid:
            by_uuid[record.gpu_uuid].append(record.monotonic_seconds)
    gaps = [
        current - previous
        for timestamps in by_uuid.values()
        for previous, current in zip(timestamps, timestamps[1:])
    ]
    if not gaps:
        raise ValueError("GPU active window has no measurable cadence")
    expected_summary = {
        "schema_version": 1,
        "utilization_required": True,
        "utilization_claimed": True,
        "sample_counts": result.sample_counts,
        "means": result.means,
    }
    if _strict_json(summary_raw, where="gpu-summary.json") != expected_summary:
        raise ValueError("GPU summary is not derived from the raw active window")
    return {
        "gpu_uuids": list(expected_uuids),
        "gpu_sample_counts": result.sample_counts,
        "gpu_means": result.means,
        "gpu_min_gap_seconds": min(gaps),
        "gpu_max_gap_seconds": max(gaps),
    }


def _case_directory(root: Path, case_id: str) -> Path:
    path = root / case_id
    fd = _open_directory_nofollow(path)
    os.close(fd)
    return path


def _environment_without_visible_count(environment: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(environment["payload"])
    payload.pop("visible_cuda_device_count")
    return payload


def _capture_scheduler_log_manifest(
    scheduler_logs_root: Path,
    *,
    expected_names: Sequence[str],
    ceiling_bytes: int,
    reject_log: Any,
) -> dict[str, Any]:
    directory_fd = _open_directory_nofollow(scheduler_logs_root)
    os.close(directory_fd)
    if {path.name for path in scheduler_logs_root.iterdir()} != set(expected_names):
        raise ValueError("scheduler log directory has a missing or unknown file")
    entries: list[dict[str, Any]] = []
    for name in expected_names:
        raw = _read_regular_bytes(
            scheduler_logs_root / name, max_bytes=ceiling_bytes
        )
        findings = reject_log._scan_text(raw.decode("utf-8", errors="replace"))
        if findings:
            raise ValueError(f"scheduler log contains rejected signatures: {name}")
        entries.append(
            {
                "name": name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            }
        )
    manifest = make_scheduler_log_manifest(entries, ceiling_bytes=ceiling_bytes)
    validate_scheduler_log_manifest(manifest)
    return manifest


def assemble_smoke_attestation(
    cases_root: Path,
    *,
    output: Path,
    expected_commit_sha: str,
    expected_tree_sha: str,
    expected_source_manifest_sha256: str,
    expected_checkpoint_ceiling_bytes: int,
    expected_scheduler_log_ceiling_bytes: int,
    expected_final_attestation_ceiling_bytes: int,
    expected_submission_receipt_sha256: str,
    sacct_path: Path,
    expected_sacct_sha256: str,
    max_input_bytes: int,
    max_output_bytes: int,
    expected_artifact_identities: Mapping[str, str] | None = None,
    expected_environment_sha256_by_world_size: Mapping[str, str] | None = None,
    expected_job_bindings: Mapping[str, Mapping[str, Any]] | None = None,
    expected_repository_binding: Mapping[str, Any] | None = None,
    expected_nvidia_smi_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate all raw case artifacts and no-replace publish one attestation."""
    if _HEX40.fullmatch(expected_commit_sha or "") is None:
        raise ValueError("expected commit must be a lowercase Git SHA")
    if _HEX40.fullmatch(expected_tree_sha or "") is None:
        raise ValueError("expected tree must be a lowercase Git SHA")
    _digest(expected_source_manifest_sha256, "expected source manifest digest")
    if (
        isinstance(expected_checkpoint_ceiling_bytes, bool)
        or not isinstance(expected_checkpoint_ceiling_bytes, int)
        or expected_checkpoint_ceiling_bytes < 1
    ):
        raise ValueError("expected checkpoint ceiling must be positive")
    if (
        isinstance(expected_scheduler_log_ceiling_bytes, bool)
        or not isinstance(expected_scheduler_log_ceiling_bytes, int)
        or expected_scheduler_log_ceiling_bytes < 1
    ):
        raise ValueError("expected scheduler log ceiling must be positive")
    if (
        isinstance(expected_final_attestation_ceiling_bytes, bool)
        or not isinstance(expected_final_attestation_ceiling_bytes, int)
        or expected_final_attestation_ceiling_bytes < 1
        or max_output_bytes > expected_final_attestation_ceiling_bytes
    ):
        raise ValueError("final attestation ceiling exceeds the submission budget")
    receipt_sha = _digest(
        expected_submission_receipt_sha256, "expected submission receipt"
    )
    sacct_sha = _digest(expected_sacct_sha256, "expected sacct executable")
    if expected_repository_binding is None:
        raise ValueError("expected repository binding is required")
    repository_binding = _validated_repository_binding(
        expected_repository_binding,
        expected_commit_sha=expected_commit_sha,
    )
    if output.parent != cases_root.parent or output.name in {"", ".", ".."}:
        raise ValueError("final attestation must stay in the budgeted gate namespace")
    root_fd = _open_directory_nofollow(cases_root)
    os.close(root_fd)
    expected_cases = {case.case_id: case for case in build_matrix()}
    if expected_job_bindings is None:
        raise ValueError("terminal assembly requires receipt-bound scheduler jobs")
    terminal_manifest = capture_scheduler_terminal_manifest(
        sacct_path=sacct_path,
        expected_sacct_sha256=sacct_sha,
        submission_receipt_sha256=receipt_sha,
        expected_job_bindings=expected_job_bindings,
    )
    terminal_by_case = {
        item["case_id"]: item
        for item in terminal_manifest["payload"]["observations"]
    }
    scheduler_logs_root = cases_root.parent / "scheduler-logs"
    expected_scheduler_names = sorted(
        f"{case_id}.{suffix}"
        for case_id in expected_cases
        for suffix in ("err", "out")
    )
    reject_log = _load_module(
        "_h100_reject_nonfinite_log",
        Path(__file__).with_name("reject_nonfinite_log.py"),
    )
    scheduler_log_manifest = _capture_scheduler_log_manifest(
        scheduler_logs_root,
        expected_names=expected_scheduler_names,
        ceiling_bytes=expected_scheduler_log_ceiling_bytes,
        reject_log=reject_log,
    )
    if expected_artifact_identities is not None:
        if set(expected_artifact_identities) != set(expected_cases):
            raise ValueError("external artifact identity case set mismatch")
        for case_id, digest in expected_artifact_identities.items():
            _digest(digest, f"external artifact identity for {case_id}")
    if expected_environment_sha256_by_world_size is not None:
        _exact_keys(
            expected_environment_sha256_by_world_size,
            {"1", "2"},
            "external environment digest map",
        )
        for world_size, digest in expected_environment_sha256_by_world_size.items():
            _digest(digest, f"external {world_size}-GPU environment digest")
    if expected_job_bindings is not None and set(expected_job_bindings) != set(expected_cases):
        raise ValueError("external scheduler job-binding case set mismatch")
    actual_names = {path.name for path in cases_root.iterdir()}
    if actual_names != set(expected_cases):
        raise ValueError("case artifact root has a missing or unknown case directory")

    evidence: list[dict[str, Any]] = []
    for case in build_matrix():
        case_root = _case_directory(cases_root, case.case_id)
        compute_root = _case_directory(case_root, "compute")
        expected_case_entries = {
            "source-attestation.json",
            "runtime-evidence.json",
            "action-completion.json",
            "compute.log",
            "compute",
        }
        if case.utilization_required:
            expected_case_entries |= {"gpu.csv", "gpu.jsonl", "gpu-summary.json"}
        if {path.name for path in case_root.iterdir()} != expected_case_entries:
            raise ValueError("case directory file set is not exact")
        expected_compute_entries = {"case-result.json"}
        if case.stage is not None:
            expected_compute_entries.add(f"step-{case.repeat_steps}.ckpt")
        if {path.name for path in compute_root.iterdir()} != expected_compute_entries:
            raise ValueError("case compute directory file set is not exact")
        raw_by_name: dict[str, bytes] = {}
        large_file_entries: dict[str, dict[str, Any]] = {}

        def read(name: str, path: Path) -> bytes:
            raw = _read_regular_bytes(path, max_bytes=max_input_bytes)
            raw_by_name[name] = raw
            return raw

        source = _strict_json(
            read("source-attestation.json", case_root / "source-attestation.json"),
            where=f"{case.case_id}/source-attestation.json",
        )
        source_payload = source.get("payload")
        if not isinstance(source_payload, Mapping):
            raise ValueError("source attestation payload is missing")
        artifact_identity = _digest(
            source_payload.get("artifact_identity_sha256"), "artifact identity"
        )
        if (
            expected_artifact_identities is not None
            and artifact_identity != expected_artifact_identities[case.case_id]
        ):
            raise ValueError("case artifact identity differs from submission receipt")
        source_digest = validate_source_attestation(
            source,
            case_id=case.case_id,
            artifact_identity_sha256=artifact_identity,
            commit_sha=expected_commit_sha,
            tree_sha=expected_tree_sha,
            source_manifest_sha256=expected_source_manifest_sha256,
        )
        runtime = _strict_json(
            read("runtime-evidence.json", case_root / "runtime-evidence.json"),
            where=f"{case.case_id}/runtime-evidence.json",
        )
        runtime_digest = validate_runtime_evidence(
            runtime,
            case=case,
            commit_sha=expected_commit_sha,
            tree_sha=expected_tree_sha,
            source_manifest_sha256=expected_source_manifest_sha256,
            expected_job_binding=(
                expected_job_bindings[case.case_id]
                if expected_job_bindings is not None
                else None
            ),
            expected_nvidia_smi_sha256=expected_nvidia_smi_sha256,
        )
        completion = _strict_json(
            read("action-completion.json", case_root / "action-completion.json"),
            where=f"{case.case_id}/action-completion.json",
        )
        completion_digest = validate_action_completion(
            completion,
            case_id=case.case_id,
            artifact_identity_sha256=artifact_identity,
            source_attestation_sha256=source_digest,
            runtime_evidence_sha256=runtime_digest,
            commit_sha=expected_commit_sha,
            tree_sha=expected_tree_sha,
            source_manifest_sha256=expected_source_manifest_sha256,
            scheduler_binding=runtime["payload"]["scheduler_binding"],
            expected_job_binding=(
                expected_job_bindings[case.case_id]
                if expected_job_bindings is not None
                else None
            ),
        )
        case_result = _strict_json(
            read("compute/case-result.json", compute_root / "case-result.json"),
            where=f"{case.case_id}/compute/case-result.json",
        )
        case_result_digest, active_start, active_end = validate_case_result(
            case_result, case=case
        )
        runtime_scheduler = runtime["payload"]["scheduler_binding"]
        terminal_observation = terminal_by_case[case.case_id]
        if (
            case_result["payload"]["slurm_job_id"] != runtime_scheduler["job_id"]
            or case_result["payload"]["requested_resource_sha256"]
            != runtime_scheduler["requested_resource_sha256"]
            or runtime_scheduler["job_id"] != terminal_observation["job_id"]
            or runtime_scheduler["slurm_cluster_name"]
            != terminal_observation["observed_cluster"]
        ):
            raise ValueError("case result/runtime differs from terminal scheduler binding")
        result_payload = case_result["payload"]
        if result_payload["checkpoint_ceiling_bytes"] != expected_checkpoint_ceiling_bytes:
            raise ValueError("case checkpoint ceiling differs from external commitment")
        if case.stage is not None:
            checkpoint_name = f"compute/{result_payload['checkpoint_relative_path']}"
            checkpoint_sha256, checkpoint_size = _hash_regular_file(
                compute_root / result_payload["checkpoint_relative_path"],
                max_bytes=expected_checkpoint_ceiling_bytes,
            )
            large_file_entries[checkpoint_name] = {
                "name": checkpoint_name,
                "sha256": checkpoint_sha256,
                "size": checkpoint_size,
            }
            if (
                checkpoint_size != result_payload["checkpoint_size"]
                or checkpoint_sha256 != result_payload["checkpoint_sha256"]
                or checkpoint_size > expected_checkpoint_ceiling_bytes
            ):
                raise ValueError("Trainer checkpoint bytes differ from case result")
        compute_log = read("compute.log", case_root / "compute.log")
        findings = reject_log._scan_text(compute_log.decode("utf-8", errors="replace"))
        if findings:
            raise ValueError("case compute log contains rejected signatures")

        if case.utilization_required:
            csv_raw = read("gpu.csv", case_root / "gpu.csv")
            jsonl_raw = read("gpu.jsonl", case_root / "gpu.jsonl")
            summary_raw = read("gpu-summary.json", case_root / "gpu-summary.json")
            gpu_window = _gpu_window_from_raw(
                csv_raw,
                jsonl_raw,
                summary_raw,
                expected_gpu_count=case.world_size,
                expected_start_seconds=active_start,
                expected_end_seconds=active_end,
            )
            gpu_window_sha256: str | None = hashlib.sha256(
                _canonical(gpu_window)
            ).hexdigest()
        else:
            gpu_window = {
                "gpu_uuids": [],
                "gpu_sample_counts": {},
                "gpu_means": {},
                "gpu_min_gap_seconds": None,
                "gpu_max_gap_seconds": None,
            }
            gpu_window_sha256 = None

        artifact_manifest = {
            "schema_version": 1,
            "case_id": case.case_id,
            "files": [
                (
                    large_file_entries[name]
                    if name in large_file_entries
                    else _artifact_file(name, raw_by_name[name])
                )
                for name in sorted(set(raw_by_name) | set(large_file_entries))
            ],
        }
        artifact_sha256 = hashlib.sha256(_canonical(artifact_manifest)).hexdigest()
        case_binding = {
            "case_id": case.case_id,
            "source_attestation_sha256": source_digest,
            "runtime_evidence_sha256": runtime_digest,
            "action_completion_sha256": completion_digest,
            "artifact_identity_sha256": artifact_identity,
            "artifact_sha256": artifact_sha256,
        }
        evidence.append(
            {
                "case_id": case.case_id,
                "arm": case.arm,
                "stage": case.stage,
                "world_size": case.world_size,
                "observed_sequence_length": case.observed_sequence_length,
                "smoke_kind": case.smoke_kind,
                "recompute": case.recompute,
                "active_start_seconds": active_start,
                "active_end_seconds": active_end,
                "artifact_sha256": artifact_sha256,
                "artifact_manifest": artifact_manifest,
                "artifact_identity_sha256": artifact_identity,
                "case_result": case_result,
                "case_result_sha256": case_result_digest,
                "checkpoint_relative_path": result_payload[
                    "checkpoint_relative_path"
                ],
                "checkpoint_sha256": result_payload["checkpoint_sha256"],
                "checkpoint_size": result_payload["checkpoint_size"],
                "source_attestation": source,
                "source_attestation_sha256": source_digest,
                "runtime_evidence": runtime,
                "runtime_evidence_sha256": runtime_digest,
                "action_completion": completion,
                "action_completion_sha256": completion_digest,
                "case_binding_sha256": hashlib.sha256(
                    _canonical(case_binding)
                ).hexdigest(),
                "checkout_commit_sha": expected_commit_sha,
                "checkout_tree_sha": expected_tree_sha,
                "clean_detached": True,
                "tabicl_attestation_count": 1,
                "python_no_user_site": "1",
                "gpu_window_sha256": gpu_window_sha256,
                **gpu_window,
            }
        )

    one_gpu_environments = {
        item["runtime_evidence"]["payload"]["environment"]["sha256"]:
        item["runtime_evidence"]["payload"]["environment"]
        for item in evidence
        if item["world_size"] == 1
    }
    two_gpu_environments = {
        item["runtime_evidence"]["payload"]["environment"]["sha256"]:
        item["runtime_evidence"]["payload"]["environment"]
        for item in evidence
        if item["world_size"] == 2
    }
    if len(one_gpu_environments) != 1 or len(two_gpu_environments) != 1:
        raise ValueError("runtime environments do not agree within GPU-count cohorts")
    one_environment = next(iter(one_gpu_environments.values()))
    two_environment = next(iter(two_gpu_environments.values()))
    if (
        _environment_without_visible_count(one_environment)
        != _environment_without_visible_count(two_environment)
        or one_environment["payload"]["visible_cuda_device_count"] != 1
        or two_environment["payload"]["visible_cuda_device_count"] != 2
    ):
        raise ValueError("NCCL runtime environment may differ only by CUDA count")
    if expected_environment_sha256_by_world_size is not None and (
        one_environment["sha256"]
        != expected_environment_sha256_by_world_size["1"]
        or two_environment["sha256"]
        != expected_environment_sha256_by_world_size["2"]
    ):
        raise ValueError("runtime environments differ from submission receipt")
    models = {
        device["name"]
        for item in evidence
        for device in item["runtime_evidence"]["payload"]["gpu_devices"]
    }
    drivers = {
        device["driver_version"]
        for item in evidence
        for device in item["runtime_evidence"]["payload"]["gpu_devices"]
    }
    if len(models) != 1 or len(drivers) != 1:
        raise ValueError("runtime GPU model/driver evidence is inconsistent")
    final_scheduler_log_manifest = _capture_scheduler_log_manifest(
        scheduler_logs_root,
        expected_names=expected_scheduler_names,
        ceiling_bytes=expected_scheduler_log_ceiling_bytes,
        reject_log=reject_log,
    )
    if final_scheduler_log_manifest != scheduler_log_manifest:
        raise ValueError("scheduler logs changed after terminal reconciliation")
    attestation = make_smoke_attestation(
        {
            "commit_sha": expected_commit_sha,
            "tree_sha": expected_tree_sha,
            "environment_sha256": one_environment["sha256"],
            "source_manifest_sha256": expected_source_manifest_sha256,
            "repository_binding": repository_binding,
            "nvidia_smi_sha256": _digest(
                expected_nvidia_smi_sha256, "receipt nvidia-smi executable"
            ),
            "gpu_model": next(iter(models)),
            "driver_version": next(iter(drivers)),
            "checkpoint_ceiling_bytes": expected_checkpoint_ceiling_bytes,
            "observed_checkpoint_max_bytes": max(
                item["checkpoint_size"]
                for item in evidence
                if item["checkpoint_size"] is not None
            ),
            "submission_receipt_sha256": receipt_sha,
            "scheduler_terminal_manifest": terminal_manifest,
            "scheduler_log_manifest": scheduler_log_manifest,
            "cases": evidence,
        }
    )
    validate_smoke_attestation(
        attestation,
        expected_sha256=attestation["sha256"],
        expected_commit_sha=expected_commit_sha,
        expected_tree_sha=expected_tree_sha,
        expected_environment_sha256=one_environment["sha256"],
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_gpu_model=next(iter(models)),
        expected_checkpoint_ceiling_bytes=expected_checkpoint_ceiling_bytes,
        expected_submission_receipt_sha256=receipt_sha,
        expected_sacct_sha256=sacct_sha,
        expected_git_sha256=repository_binding["git_sha256"],
        expected_repository_identity_sha256=repository_binding[
            "repository_identity_sha256"
        ],
        expected_repository_query_sha256=repository_binding["query_sha256"],
        expected_nvidia_smi_sha256=expected_nvidia_smi_sha256,
    )
    _publish_no_replace(output, attestation, max_bytes=max_output_bytes)
    return attestation


def execution_argv(case: ValidationCase, root: Path) -> list[str]:
    return [str(root / "scripts/run_h100_identity_maxseq_smoke.sh"), case.case_id]


def _assert_temporary_cuda_rng_resume() -> None:
    import torch
    from tabicl.train._identity_rng import TrainerIdentityRNG

    if not torch.cuda.is_available():
        raise RuntimeError("Temporary CUDA RNG gate requires CUDA")
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = [value.clone() for value in torch.cuda.get_rng_state_all()]
    controller = TrainerIdentityRNG(
        identity_mode="temporary", base_seed=1729, rank=0, world_size=1
    )
    controller.sample_permutations(4, 11, device="cuda")
    state = controller.state_dict()
    expected = controller.sample_permutations(4, 11, device="cuda").cpu()
    resumed = TrainerIdentityRNG(
        identity_mode="temporary", base_seed=1729, rank=0, world_size=1
    )
    resumed.load_state_dict(state)
    actual = resumed.sample_permutations(4, 11, device="cuda").cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), cpu_before, rtol=0, atol=0)
    for current, before in zip(torch.cuda.get_rng_state_all(), cuda_before):
        torch.testing.assert_close(current, before, rtol=0, atol=0)


def _assert_nccl_two_gpu() -> None:
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("NCCL gate requires exactly two visible CUDA GPUs")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    if dist.get_backend() != "nccl" or dist.get_world_size() != 2:
        raise RuntimeError("NCCL gate requires a two-rank NCCL process group")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    value = torch.tensor([rank + 1.0], device=f"cuda:{rank}")
    dist.all_reduce(value)
    torch.cuda.synchronize()
    if value.item() != 3.0:
        raise RuntimeError("NCCL all-reduce result mismatch")
    dist.barrier()


def _assert_prior_dataloader_resume() -> None:
    import torch
    from tabicl.prior._dataset import PriorDataset
    from tabicl.prior._genload import make_prior_dataloader
    from tabicl.prior.graph_lib._config import PriorConfig

    def dataset(cursor: int = 0):
        result = PriorDataset(
            regression=False,
            prior_type="graph_scm",
            batch_size=1,
            batch_size_per_gp=1,
            min_features=2,
            max_features=2,
            max_classes=2,
            min_seq_len=32,
            max_seq_len=32,
            min_train_size=12,
            max_train_size=20,
            n_jobs=1,
            device="cpu",
            config=PriorConfig(
                min_n_nodes=2,
                max_n_nodes=3,
                fct_types="lin",
                multi_fct_types="concat",
                filter_unpredictable_graphs=False,
                filter_unpredictable_datasets=False,
            ),
        )
        result.configure_logical_stream(
            schema="h100-graph-scm-prior-resume-v1",
            experiment_seed=2026,
            ddp_rank=0,
            world_size=1,
            cursor=cursor,
        )
        return result

    def batch_bytes(batch: Sequence[Any]) -> bytes:
        encoded = bytearray()
        for tensor in batch:
            tensor = tensor.detach().cpu().contiguous()
            dtype = str(tensor.dtype).encode("ascii")
            shape = repr(tuple(tensor.shape)).encode("ascii")
            payload = tensor.numpy().tobytes()
            for field in (dtype, shape, payload):
                encoded.extend(len(field).to_bytes(8, "big"))
                encoded.extend(field)
        return bytes(encoded)

    uninterrupted_source = dataset()
    uninterrupted_loader = make_prior_dataloader(
        uninterrupted_source,
        num_workers=2,
        prefetch_factor=4,
        pin_memory=False,
    )
    initial_generator_state = uninterrupted_loader.generator.get_state().clone()
    uninterrupted = iter(uninterrupted_loader)
    expected = [batch_bytes(next(uninterrupted)) for _ in range(5)]
    if torch.equal(uninterrupted_loader.generator.get_state(), initial_generator_state):
        raise RuntimeError("uninterrupted DataLoader generator was not initialized")
    stream_state = uninterrupted_source.logical_stream_state_dict(cursor=2)
    if stream_state["cursor"] != 2 or "manifest_sha256" not in stream_state:
        raise RuntimeError("prior stream checkpoint manifest is incomplete")
    if any("generator" in key for key in stream_state):
        raise RuntimeError("DataLoader generator state must be reconstructed, not saved")
    resumed_source = dataset()
    resumed_source.load_logical_stream_state_dict(stream_state)
    if resumed_source.logical_stream_state_dict() != stream_state:
        raise RuntimeError("restored prior stream manifest differs from checkpoint")
    resumed_loader = make_prior_dataloader(
        resumed_source,
        num_workers=1,
        prefetch_factor=1,
        pin_memory=False,
    )
    torch.testing.assert_close(
        resumed_loader.generator.get_state(), initial_generator_state, rtol=0, atol=0
    )
    resumed = iter(resumed_loader)
    actual = [batch_bytes(next(resumed)) for _ in range(3)]
    if actual != expected[2:]:
        raise RuntimeError("graph_scm prior/DataLoader resumed bytes differ")
    for iterator in (uninterrupted, resumed):
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()


def execute_special(case_id: str) -> None:
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise RuntimeError("special validation requires Python -I -B")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("special validation requires PYTHONNOUSERSITE=1")
    if os.environ.get("TABICL_ATTESTED_CASE_ID") != case_id:
        raise RuntimeError("special validation case is not independently attested")
    _digest(os.environ.get("TABICL_ATTESTED_ARTIFACT_SHA256"), "attested artifact digest")
    if case_id == "temporary_cuda_rng_resume":
        _assert_temporary_cuda_rng_resume()
    elif case_id == "nccl_2gpu":
        _assert_nccl_two_gpu()
    elif case_id == "prior_dataloader_resume":
        _assert_prior_dataloader_resume()
    else:
        raise ValueError("unknown special validation case")


def _training_config(
    case: ValidationCase,
    output_dir: Path,
    checkpoint_ceiling_bytes: int,
):
    from tabicl.train._train_config import build_parser

    if (
        isinstance(checkpoint_ceiling_bytes, bool)
        or not isinstance(checkpoint_ceiling_bytes, int)
        or checkpoint_ceiling_bytes < 1
    ):
        raise ValueError("H100 checkpoint ceiling must be a positive integer")
    if case.category == "stage1_one_step":
        steps, batch_size, micro_batch, lr, gradient_clip = 1, 64, 4, "8e-4", "10"
        flash = "false"
        # Exercise the same real Stage-1 graph prior and CPU-worker topology as
        # the formal run.  A dummy prior here would only prove the model path,
        # and would miss the Temporary identity + GraphPrior/DataLoader
        # integration that this one-step gate is meant to protect.
        prior_argv = [
            "--prior_type", "graph_scm",
            "--prior_device", "cpu",
            "--n_jobs", "16",
            "--batch_size_per_gp", "4",
            "--min_features", "1",
            "--max_features", "100",
            "--max_classes", "10",
            "--max_seq_len", str(case.observed_sequence_length),
            "--log_seq_len", "false",
            "--replay_small", "false",
            "--min_train_size", "0.3",
            "--max_train_size", "0.9",
            "--seq_len_per_gp", "true",
            "--graph_noise", "false",
            "--filter_unpredictable_graphs", "true",
            "--filter_unpredictable_datasets", "true",
            "--allow_act_warping", "false",
            "--min_n_nodes", "2",
            "--max_n_nodes", "32",
            "--cauchy_dag_offset", "0",
        ]
    elif case.category == "stage2_maxseq":
        steps, batch_size, micro_batch, lr, gradient_clip = case.repeat_steps, 64, 1, "1e-4", "10"
        flash = "true"
        prior_argv = [
            "--prior_type", "dummy",
            "--prior_device", "cpu",
            "--n_jobs", "1",
            "--batch_size_per_gp", "1",
            "--min_features", "100",
            "--max_features", "100",
            "--max_classes", "10",
            "--min_seq_len", str(case.observed_sequence_length),
            "--max_seq_len", str(case.observed_sequence_length),
            "--log_seq_len", "false",
            "--replay_small", "false",
            "--min_train_size", "0.8",
            "--max_train_size", "0.8",
            "--seq_len_per_gp", "false",
        ]
    elif case.category == "stage3_maxseq":
        steps, batch_size, micro_batch, lr, gradient_clip = case.repeat_steps, 64, 1, "2e-5", "1"
        flash = "true"
        prior_argv = [
            "--prior_type", "dummy",
            "--prior_device", "cpu",
            "--n_jobs", "1",
            "--batch_size_per_gp", "1",
            "--min_features", "100",
            "--max_features", "100",
            "--max_classes", "10",
            "--min_seq_len", str(case.observed_sequence_length),
            "--max_seq_len", str(case.observed_sequence_length),
            "--log_seq_len", "false",
            "--replay_small", "false",
            "--min_train_size", "0.8",
            "--max_train_size", "0.8",
            "--seq_len_per_gp", "false",
        ]
    else:
        raise ValueError("case is not a Trainer validation case")
    sequence_length = case.observed_sequence_length
    assert sequence_length is not None and case.arm is not None
    argv = [
        "--formal_training", "false",
        "--wandb_log", "false",
        "--device", "cuda",
        "--dtype", "float32",
        "--amp", "true",
        "--np_seed", "42",
        "--torch_seed", "42",
        "--identity_rng_seed", "42",
        "--max_steps", str(steps),
        "--batch_size", str(batch_size),
        "--micro_batch_size", str(micro_batch),
        "--lr", lr,
        "--muon", "true",
        "--beta1", "0.9",
        "--weight_decay", "0.01",
        "--use_cautious_wd", "false",
        "--scheduler", "cosine_with_restarts",
        "--warmup_proportion", "0.01",
        "--cosine_num_cycles", "1",
        "--cosine_amplitude_decay", "1",
        "--cosine_lr_end", "1e-7",
        "--gradient_clipping", gradient_clip,
        *prior_argv,
        "--embed_dim", "128",
        "--col_num_blocks", "3",
        "--col_nhead", "8",
        "--col_num_inds", "128",
        "--col_affine", "false",
        "--col_feature_group", "same",
        "--col_feature_group_size", "3",
        "--col_target_aware", "true",
        "--col_ssmax", "true",
        "--row_num_blocks", "3",
        "--row_nhead", "8",
        "--row_num_cls", "4",
        "--row_rope_base", "100000",
        "--row_rope_interleaved", "false",
        "--row_identity_mode", case.arm,
        "--icl_num_blocks", "12",
        "--icl_nhead", "8",
        "--icl_ssmax", "true",
        "--ssmax_type", "qassmax-mlp-elementwise",
        "--ff_factor", "2",
        "--norm_first", "true",
        "--zero_init", "false",
        "--use_flash_attn3", flash,
        "--recompute", "true" if case.recompute else "false",
        "--checkpoint_dir", str(output_dir),
        "--max_checkpoint_bytes", str(checkpoint_ceiling_bytes),
        "--save_temp_every", str(steps + 1),
        "--save_perm_every", str(steps + 1),
        "--max_checkpoints", "1",
    ]
    return build_parser().parse_args(argv)


def _publish_case_result(path: Path, payload: dict[str, Any]) -> None:
    body = {"schema_version": 1, "kind": "h100_case_result", "payload": payload}
    value = {**body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    _publish_no_replace(path, value, max_bytes=1 << 20)


def _publish_active_signal(environment_name: str) -> float:
    raw_path = os.environ.get(environment_name)
    if not raw_path:
        raise RuntimeError(f"{environment_name} is required for utilization cases")
    path = Path(raw_path)
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise RuntimeError("active training signal must be a fresh absolute path")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    timestamp = time.monotonic()
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, f"{timestamp:.17g}\n".encode("ascii"))
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_fd = _open_directory_nofollow(path.parent)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return timestamp


def execute_validation_case(case_id: str, argv: list[str]) -> None:
    """Execute one fixed case after ``run_exact_tabicl`` source attestation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-ceiling-bytes", required=True, type=int)
    args = parser.parse_args(argv)
    cases = {case.case_id: case for case in build_matrix()}
    case = cases.get(case_id)
    if case is None:
        raise ValueError("unknown validation case")
    if args.checkpoint_ceiling_bytes < 1:
        raise ValueError("checkpoint ceiling must be positive")
    if case.case_id == "nccl_2gpu":
        if args.output_dir.is_symlink():
            raise ValueError("validation case output must not be a symlink")
        # Both ranks enter concurrently inside an outer fresh-only CASE_ROOT.
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise ValueError("validation case output must be a fresh path")
        args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    checkpoint_relative_path = None
    checkpoint_sha256 = None
    checkpoint_size = None
    if case.category in {"stage1_one_step", "stage2_maxseq", "stage3_maxseq"}:
        import torch
        from tabicl.train._run import Trainer

        trainer = Trainer(
            _training_config(
                case,
                args.output_dir,
                args.checkpoint_ceiling_bytes,
            )
        )
        batch = next(iter(trainer.dataloader))
        observed = {int(value) for value in batch[3].detach().cpu().tolist()}
        if observed != {case.observed_sequence_length}:
            raise RuntimeError(f"runtime sequence length mismatch: {sorted(observed)}")
        if trainer.model_config.get("recompute") is not case.recompute:
            raise RuntimeError("runtime recompute activation mismatch")
        if case.utilization_required:
            started = _publish_active_signal("FORMAL_GPU_ACTIVE_START_SIGNAL")
        trainer.train()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if case.utilization_required:
            ended = _publish_active_signal("FORMAL_GPU_ACTIVE_END_SIGNAL")
        checkpoint_relative_path = f"step-{case.repeat_steps}.ckpt"
        trainer.save_checkpoint(name=checkpoint_relative_path)
        checkpoint_path = args.output_dir / checkpoint_relative_path
        checkpoint_sha256, checkpoint_size = _hash_regular_file(
            checkpoint_path, max_bytes=args.checkpoint_ceiling_bytes
        )
    else:
        execute_special(case_id)
    if not case.utilization_required:
        ended = time.monotonic()
    if case.case_id == "nccl_2gpu":
        import torch.distributed as dist

        if dist.get_rank() != 0:
            return
    _publish_case_result(
        args.output_dir / "case-result.json",
        {
            "case_id": case_id,
            "assertion": case.assertion,
            "observed_sequence_length": case.observed_sequence_length,
            "recompute": case.recompute,
            "world_size": case.world_size,
            "monotonic_start_seconds": started,
            "monotonic_end_seconds": ended,
            "checkpoint_relative_path": checkpoint_relative_path,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_size": checkpoint_size,
            "checkpoint_ceiling_bytes": args.checkpoint_ceiling_bytes,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "requested_resource_sha256": os.environ.get(
                "FORMAL_REQUESTED_RESOURCE_SHA256"
            ),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--print-matrix", action="store_true")
    group.add_argument("--dry-run-case", choices=[case.case_id for case in build_matrix()])
    group.add_argument("--validate-attestation", type=Path)
    group.add_argument("--assemble-cases-root", type=Path)
    group.add_argument(
        "--execute-special",
        choices=[
            "temporary_cuda_rng_resume",
            "nccl_2gpu",
            "prior_dataloader_resume",
        ],
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--expected-attestation-sha256")
    parser.add_argument("--expected-commit-sha")
    parser.add_argument("--expected-tree-sha")
    parser.add_argument("--expected-environment-sha256")
    parser.add_argument("--expected-source-manifest-sha256")
    parser.add_argument("--expected-gpu-model")
    parser.add_argument("--expected-checkpoint-ceiling-bytes", type=int)
    parser.add_argument("--expected-submission-receipt-sha256")
    parser.add_argument("--expected-sacct-sha256")
    parser.add_argument("--expected-git-sha256")
    parser.add_argument("--expected-repository-identity-sha256")
    parser.add_argument("--expected-repository-query-sha256")
    parser.add_argument("--expected-nvidia-smi-sha256")
    parser.add_argument("--submission-receipt", type=Path)
    parser.add_argument("--submission-receipt-max-bytes", type=int)
    parser.add_argument("--sacct", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-input-bytes", type=int)
    parser.add_argument("--max-output-bytes", type=int)
    args = parser.parse_args(argv)
    if args.print_matrix:
        print(json.dumps([asdict(case) for case in build_matrix()], sort_keys=True, separators=(",", ":")))
        return 0
    cases = {case.case_id: case for case in build_matrix()}
    if args.dry_run_case:
        case = cases[args.dry_run_case]
        print(json.dumps({"case": asdict(case), "argv": execution_argv(case, args.root)}, sort_keys=True, separators=(",", ":")))
        return 0
    if args.validate_attestation:
        if args.max_input_bytes is None:
            raise ValueError("attestation validation requires an input byte ceiling")
        value = _strict_json(
            _read_regular_bytes(
                args.validate_attestation, max_bytes=args.max_input_bytes
            ),
            where="smoke attestation",
        )
        report = validate_smoke_attestation(
            value,
            expected_sha256=args.expected_attestation_sha256,
            expected_commit_sha=args.expected_commit_sha,
            expected_tree_sha=args.expected_tree_sha,
            expected_environment_sha256=args.expected_environment_sha256,
            expected_source_manifest_sha256=args.expected_source_manifest_sha256,
            expected_gpu_model=args.expected_gpu_model,
            expected_checkpoint_ceiling_bytes=args.expected_checkpoint_ceiling_bytes,
            expected_submission_receipt_sha256=args.expected_submission_receipt_sha256,
            expected_sacct_sha256=args.expected_sacct_sha256,
            expected_git_sha256=args.expected_git_sha256,
            expected_repository_identity_sha256=(
                args.expected_repository_identity_sha256
            ),
            expected_repository_query_sha256=args.expected_repository_query_sha256,
            expected_nvidia_smi_sha256=args.expected_nvidia_smi_sha256,
        )
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0
    if args.assemble_cases_root:
        if (
            args.output is None
            or args.max_input_bytes is None
            or args.max_output_bytes is None
            or args.submission_receipt is None
            or args.submission_receipt_max_bytes is None
            or args.sacct is None
        ):
            raise ValueError(
                "assembly requires the released submission receipt and all input/output byte ceilings"
            )
        receipt = validate_gate_submission_receipt(
            args.submission_receipt,
            expected_commit_sha=args.expected_commit_sha,
            expected_tree_sha=args.expected_tree_sha,
            expected_source_manifest_sha256=args.expected_source_manifest_sha256,
            expected_checkpoint_ceiling_bytes=args.expected_checkpoint_ceiling_bytes,
            max_bytes=args.submission_receipt_max_bytes,
        )
        value = assemble_smoke_attestation(
            args.assemble_cases_root,
            output=args.output,
            expected_commit_sha=args.expected_commit_sha,
            expected_tree_sha=args.expected_tree_sha,
            expected_source_manifest_sha256=args.expected_source_manifest_sha256,
            expected_checkpoint_ceiling_bytes=args.expected_checkpoint_ceiling_bytes,
            expected_scheduler_log_ceiling_bytes=receipt[
                "scheduler_log_ceiling_bytes"
            ],
            expected_final_attestation_ceiling_bytes=receipt[
                "final_attestation_ceiling_bytes"
            ],
            expected_submission_receipt_sha256=receipt["sha256"],
            sacct_path=args.sacct,
            expected_sacct_sha256=receipt["sacct_sha256"],
            max_input_bytes=args.max_input_bytes,
            max_output_bytes=args.max_output_bytes,
            expected_artifact_identities=receipt["artifact_identities"],
            expected_environment_sha256_by_world_size=receipt[
                "environment_sha256_by_world_size"
            ],
            expected_job_bindings=receipt["job_bindings"],
            expected_repository_binding=receipt["repository_binding"],
            expected_nvidia_smi_sha256=receipt["nvidia_smi_sha256"],
        )
        print(
            json.dumps(
                {
                    "sha256": value["sha256"],
                    "cases": 12,
                    "submission_receipt_sha256": receipt["sha256"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    execute_special(args.execute_special)
    print(json.dumps({"case_id": args.execute_special, "passed": True}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"H100 identity validation failed: {error}", file=sys.stderr)
        raise SystemExit(2)
