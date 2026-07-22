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
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


ARMS = ("rope", "temporary", "none")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_ENVIRONMENT_KEYS = {
    "python_version",
    "python_implementation",
    "platform_system",
    "platform_release",
    "platform_machine",
    "torch_version",
    "numpy_version",
    "cuda_runtime_version",
    "cudnn_version",
    "visible_cuda_device_count",
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


def make_smoke_attestation(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "kind": "h100_identity_smoke",
        "payload": dict(payload),
    }
    return {**body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}


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
        "final_tabicl_attested": True,
        "completed": True,
    }
    if value["payload"] != expected:
        raise ValueError("action completion is not bound to successful exact-T case")
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
    if (
        not executable
        or not os.path.isabs(executable)
        or not os.path.isfile(executable)
        or not os.access(executable, os.X_OK)
    ):
        raise RuntimeError("NVIDIA_SMI must be an absolute executable")
    result = subprocess.run(
        [
            executable,
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("nvidia-smi runtime evidence query failed")
    devices: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or any(not field for field in fields):
            raise RuntimeError("nvidia-smi runtime evidence row is invalid")
        devices.append(
            {"uuid": fields[0], "name": fields[1], "driver_version": fields[2]}
        )
    return devices


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
    environment = runtime_environment_manifest()
    devices = _query_visible_gpu_devices()
    if environment["payload"]["visible_cuda_device_count"] != case.world_size:
        raise RuntimeError("PyTorch visible CUDA count does not match case world size")
    if len(devices) != case.world_size:
        raise RuntimeError("nvidia-smi visible GPU count does not match case world size")
    return make_runtime_evidence(
        {
            "case_id": case_id,
            "commit_sha": commit_sha,
            "tree_sha": tree_sha,
            "source_manifest_sha256": source_manifest_sha256,
            "world_size": case.world_size,
            "environment": environment,
            "gpu_devices": devices,
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
    environment = payload["environment"]
    _validate_manifest_envelope(
        environment, expected_kind="environment", where="runtime environment"
    )
    _exact_keys(
        environment["payload"], _RUNTIME_ENVIRONMENT_KEYS, "runtime environment payload"
    )
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
            "gpu_model",
            "driver_version",
            "checkpoint_ceiling_bytes",
            "observed_checkpoint_max_bytes",
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
        )
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


def _read_regular_bytes(path: Path, *, max_bytes: int) -> bytes:
    if max_bytes < 1:
        raise ValueError("artifact byte ceiling must be positive")
    candidate = path if path.is_absolute() else Path.cwd() / path
    candidate = Path(os.path.abspath(candidate))
    parent_fd = _open_directory_nofollow(candidate.parent)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
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


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixed validation helper: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


def assemble_smoke_attestation(
    cases_root: Path,
    *,
    output: Path,
    expected_commit_sha: str,
    expected_tree_sha: str,
    expected_source_manifest_sha256: str,
    expected_checkpoint_ceiling_bytes: int,
    max_input_bytes: int,
    max_output_bytes: int,
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
    root_fd = _open_directory_nofollow(cases_root)
    os.close(root_fd)
    expected_cases = {case.case_id: case for case in build_matrix()}
    actual_names = {path.name for path in cases_root.iterdir()}
    if actual_names != set(expected_cases):
        raise ValueError("case artifact root has a missing or unknown case directory")

    reject_log = _load_module(
        "_h100_reject_nonfinite_log",
        Path(__file__).with_name("reject_nonfinite_log.py"),
    )
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
        )
        case_result = _strict_json(
            read("compute/case-result.json", compute_root / "case-result.json"),
            where=f"{case.case_id}/compute/case-result.json",
        )
        case_result_digest, active_start, active_end = validate_case_result(
            case_result, case=case
        )
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
    attestation = make_smoke_attestation(
        {
            "commit_sha": expected_commit_sha,
            "tree_sha": expected_tree_sha,
            "environment_sha256": one_environment["sha256"],
            "source_manifest_sha256": expected_source_manifest_sha256,
            "gpu_model": next(iter(models)),
            "driver_version": next(iter(drivers)),
            "checkpoint_ceiling_bytes": expected_checkpoint_ceiling_bytes,
            "observed_checkpoint_max_bytes": max(
                item["checkpoint_size"]
                for item in evidence
                if item["checkpoint_size"] is not None
            ),
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


def _training_config(case: ValidationCase, output_dir: Path):
    from tabicl.train._train_config import build_parser

    if case.category == "stage1_one_step":
        steps, batch_size, micro_batch, lr, gradient_clip = 1, 64, 4, "8e-4", "10"
        flash = "false"
    elif case.category == "stage2_maxseq":
        steps, batch_size, micro_batch, lr, gradient_clip = case.repeat_steps, 64, 1, "1e-4", "10"
        flash = "true"
    elif case.category == "stage3_maxseq":
        steps, batch_size, micro_batch, lr, gradient_clip = case.repeat_steps, 64, 1, "2e-5", "1"
        flash = "true"
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
        "--prior_type", "dummy",
        "--prior_device", "cpu",
        "--n_jobs", "1",
        "--batch_size_per_gp", "1",
        "--min_features", "100",
        "--max_features", "100",
        "--max_classes", "10",
        "--min_seq_len", str(sequence_length),
        "--max_seq_len", str(sequence_length),
        "--log_seq_len", "false",
        "--replay_small", "false",
        "--min_train_size", "0.8",
        "--max_train_size", "0.8",
        "--seq_len_per_gp", "false",
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

        trainer = Trainer(_training_config(case, args.output_dir))
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
        )
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0
    if args.assemble_cases_root:
        if args.output is None or args.max_input_bytes is None or args.max_output_bytes is None:
            raise ValueError("assembly requires output and input/output byte ceilings")
        value = assemble_smoke_attestation(
            args.assemble_cases_root,
            output=args.output,
            expected_commit_sha=args.expected_commit_sha,
            expected_tree_sha=args.expected_tree_sha,
            expected_source_manifest_sha256=args.expected_source_manifest_sha256,
            expected_checkpoint_ceiling_bytes=args.expected_checkpoint_ceiling_bytes,
            max_input_bytes=args.max_input_bytes,
            max_output_bytes=args.max_output_bytes,
        )
        print(json.dumps({"sha256": value["sha256"], "cases": 12}, sort_keys=True, separators=(",", ":")))
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
