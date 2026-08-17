"""Shared contracts for paired-checkpoint TALENT discovery evaluation.

The module intentionally separates private runtime paths from portable result
contracts.  The legacy step-250000 checkpoints are exploratory and discovery
only, so the only supported roster is the frozen 109-dataset discovery set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any


SPLIT_MANIFEST_RELATIVE = Path("analysis/pe_mechanism/manifests/talent-classification-split-v1.json")
SPLIT_MANIFEST_SHA256 = (
    "1d82bf8dd40528bb9937a20204f9b23d6c48696a5c1b245304387d3b47e1ea38"
)
ROSTER_KIND = "talent-discovery-native-109-v1"
ROSTER_COUNT = 109
EXCLUDED_NON_NATIVE_CLASS_COUNTS = frozenset(
    {"texture", "walking-activity", "kr-vs-k", "letter", "UJI_Pen_Characters"}
)
RUN_CONFIG_STUDY = "tabicl-talent-paired-checkpoints-exploratory-v1"
PLAN_KIND = "talent-discovery-native-shard-plan-v1"
SHARD_KIND = "talent-paired-checkpoint-shard-v1"
AGGREGATE_KIND = "talent-paired-checkpoint-aggregate-v1"
SUBMISSION_KIND = "talent-paired-checkpoint-submission-v1"
RELEASE_KIND = "talent-paired-checkpoint-release-v1"
OPERATION_JOURNAL_KIND = "talent-paired-checkpoint-operation-journal-v1"
ROLLBACK_KIND = "talent-paired-checkpoint-rollback-v1"
DATASET_KIND = "talent-paired-checkpoint-dataset-v1"
DATASET_ARTIFACTS = frozenset({"manifest.json", "task.json", "arms"})
ARM_DATASET_KIND = "talent-paired-checkpoint-dataset-arm-v1"
ARM_DATASET_ARTIFACTS = frozenset(
    {"manifest.json", "predictions.npz", "result.json"}
)
OOM_FALLBACK_SEQUENCE = ("auto", "cpu", "disk")
DISK_OFFLOAD_MIN_FREE_BYTES = 20 * 1024**3
DISK_OFFLOAD_SCRATCH_CONTRACT: Mapping[str, Any] = {
    "root_scope": "required_job_local_scratch",
    "directory_scope": "per_dataset_attempt_per_arm",
    "path_disclosure": "none",
    "cleanup": "no_follow_after_each_attempt",
    "minimum_free_bytes_before_attempt": DISK_OFFLOAD_MIN_FREE_BYTES,
}
MATCH_LEVELS = frozenset({"same_step_legacy", "exact_prior_stream"})
TREATMENT_KINDS = frozenset({"rope", "none", "fingerprint"})
IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
FULLSIZE_ARCHITECTURE = {
    "embed_dim": 128,
    "col_num_blocks": 3,
    "col_nhead": 8,
    "col_num_inds": 128,
    "row_num_blocks": 3,
    "row_nhead": 8,
    "row_num_cls": 4,
    "icl_num_blocks": 12,
    "icl_nhead": 8,
    "max_classes": 10,
}
TALENT_ESTIMATOR_OPTIONS: Mapping[str, Any] = {
    "n_estimators": 2,
    "norm_methods": ["none", "power"],
    "feat_shuffle_method": "latin",
    "class_shuffle_method": "shift",
    "outlier_threshold": 4.0,
    "softmax_temperature": 0.9,
    "average_logits": True,
    "batch_size": 1,
    "random_state": 42,
    "n_jobs": 16,
    "use_amp": False,
    "use_fa3": False,
    "offload_mode": "auto",
    "verbose": False,
}


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_document_sha256(payload: object) -> str:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: Any) -> str:
    """Hash an ndarray's exact dtype, shape, and C-order bytes."""

    import numpy as np

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def load_json_object(path: Path, *, name: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return payload


def load_json_object_with_sha256(
    path: Path, *, name: str
) -> tuple[Mapping[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def absolute_path(
    value: str | Path, *, name: str, directory: bool = False, absent: bool = False
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    if absent:
        parent = path.parent.resolve(strict=True)
        candidate = parent / path.name
        if candidate.exists() or candidate.is_symlink():
            raise ValueError(f"{name} must be absent")
        return candidate
    resolved = path.resolve(strict=True)
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    if directory and not resolved.is_dir():
        raise ValueError(f"{name} must be a directory")
    if not directory and not resolved.is_file():
        raise ValueError(f"{name} must be a regular file")
    return resolved


def require_disjoint_output(candidate: Path, protected: Sequence[Path], *, name: str) -> None:
    """Reject an output that contains, equals, or is contained by protected input."""

    if not candidate.is_absolute():
        raise ValueError(f"{name} must be absolute")
    candidate_parent = candidate.parent.resolve(strict=True)
    normalized_candidate = candidate_parent / candidate.name
    for value in protected:
        normalized_protected = value.resolve(strict=True)
        if (
            normalized_candidate == normalized_protected
            or normalized_candidate.is_relative_to(normalized_protected)
            or normalized_protected.is_relative_to(normalized_candidate)
        ):
            raise ValueError(f"{name} overlaps a protected input: {normalized_protected}")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def self_hashed_document(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    base = {"kind": kind, "payload": dict(payload), "schema_version": 1}
    return {**base, "sha256": canonical_json_sha256(base)}


def validate_self_hashed_document(
    document: Mapping[str, Any], *, kind: str, name: str
) -> Mapping[str, Any]:
    payload = document.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} payload is invalid")
    base = {"kind": kind, "payload": payload, "schema_version": 1}
    if (
        document.get("schema_version") != 1
        or document.get("kind") != kind
        or document.get("sha256") != canonical_json_sha256(base)
        or set(document) != {"schema_version", "kind", "payload", "sha256"}
    ):
        raise ValueError(f"{name} self-hash contract is invalid")
    return payload


def frozen_discovery_roster(analysis_root: Path) -> tuple[str, ...]:
    manifest_path = analysis_root / SPLIT_MANIFEST_RELATIVE
    manifest, manifest_sha256 = load_json_object_with_sha256(
        manifest_path, name="TALENT split manifest"
    )
    if manifest_sha256 != SPLIT_MANIFEST_SHA256:
        raise ValueError("frozen TALENT split manifest digest changed")
    assignments = manifest.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("TALENT assignments are invalid")
    names = tuple(
        item.get("name")
        for item in assignments
        if isinstance(item, Mapping)
        and item.get("split") == "discovery"
        and item.get("name") not in EXCLUDED_NON_NATIVE_CLASS_COUNTS
    )
    if (
        len(names) != ROSTER_COUNT
        or len(set(names)) != ROSTER_COUNT
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError("frozen TALENT discovery roster is invalid")
    return names


def roster_sha256(names: Sequence[str]) -> str:
    return canonical_json_sha256(
        {"algorithm": "ordered-json-list-v1", "kind": ROSTER_KIND, "names": list(names)}
    )


def _require_identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")
    return value


def _require_hex(value: object, *, name: str, length: int) -> str:
    pattern = HEX40 if length == 40 else HEX64
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} must be {length} lowercase hexadecimal characters")
    return value


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], *, name: str
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{name} keys are invalid")


def _treatment_contract(payload: object, *, name: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be an object")
    kind = payload.get("kind")
    if kind not in TREATMENT_KINDS:
        raise ValueError(f"{name} kind is invalid")
    expected_keys = {"kind", "dimension"} if kind == "fingerprint" else {"kind"}
    _require_exact_keys(payload, expected_keys, name=name)
    if kind == "fingerprint":
        dimension = payload.get("dimension")
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise ValueError(f"{name} dimension is invalid")
        return {"kind": kind, "dimension": dimension}
    return {"kind": kind}


def validate_observed_treatment(
    observed: object, expected: Mapping[str, Any], *, name: str
) -> dict[str, Any]:
    if not isinstance(observed, Mapping) or set(observed) != {
        "kind",
        "row_identity_mode",
        "row_fingerprint",
        "row_fingerprint_dim",
        "row_rope_installed",
        "parameter_dtype",
    }:
        raise ValueError(f"{name} runtime treatment contract is invalid")
    kind = expected["kind"]
    dimension = observed.get("row_fingerprint_dim")
    if (
        observed.get("kind") != kind
        or observed.get("row_identity_mode")
        != ("rope" if kind == "rope" else "none")
        or observed.get("row_fingerprint") is not (kind == "fingerprint")
        or observed.get("row_rope_installed") is not (kind == "rope")
        or observed.get("parameter_dtype") != "float32"
        or (
            kind == "fingerprint"
            and dimension != expected.get("dimension")
        )
        or (
            kind != "fingerprint"
            and dimension is not None
            and (
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension <= 0
            )
        )
    ):
        raise ValueError(f"{name} runtime treatment differs from expectation")
    return dict(observed)


def _validated_existing_self_hash(
    document: Mapping[str, Any], *, kind: str, name: str
) -> Mapping[str, Any]:
    payload = document.get("payload")
    base = {"kind": kind, "payload": payload, "schema_version": 1}
    if (
        not isinstance(payload, Mapping)
        or set(document) != {"schema_version", "kind", "payload", "sha256"}
        or document.get("schema_version") != 1
        or document.get("kind") != kind
        or document.get("sha256") != canonical_json_sha256(base)
    ):
        raise ValueError(f"{name} self-hash is invalid")
    return payload


def _validate_snapshot_lineage(
    *,
    documents: Sequence[Mapping[str, Any]],
    pair_id: str,
    source_commit: str,
    seed: int,
    comparison_step: int,
    match_level: str,
    arms: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(documents) != 1:
        raise ValueError(f"{pair_id} legacy snapshot requires exactly one receipt")
    document = documents[0]
    expected_keys = {
        "schema_version",
        "kind",
        "formal_eligible",
        "comparison_step",
        "seed",
        "pilot_source_commit",
        "captured_at_utc",
        "notes",
        "arms",
    }
    _require_exact_keys(document, expected_keys, name=f"{pair_id} snapshot receipt")
    if (
        document.get("schema_version") != 1
        or document.get("kind")
        != "exploratory_same_step_pilot_checkpoint_pair"
        or document.get("formal_eligible") is not False
        or document.get("comparison_step") != comparison_step
        or document.get("seed") != seed
        or document.get("pilot_source_commit") != source_commit
        or match_level != "same_step_legacy"
    ):
        raise ValueError(f"{pair_id} legacy snapshot header is invalid")
    if not isinstance(document.get("captured_at_utc"), str) or not isinstance(
        document.get("notes"), str
    ):
        raise ValueError(f"{pair_id} legacy snapshot metadata is invalid")
    treatment_lookup = {arm["treatment"]["kind"]: arm for arm in arms}
    if set(treatment_lookup) != {"rope", "none"}:
        raise ValueError(f"{pair_id} legacy snapshot must be RoPE versus No-RoPE")
    receipt_arms = document.get("arms")
    if not isinstance(receipt_arms, Mapping) or set(receipt_arms) != {
        "rope",
        "none",
    }:
        raise ValueError(f"{pair_id} legacy snapshot arm roster is invalid")
    arm_facts: dict[str, Any] = {}
    for treatment, arm in treatment_lookup.items():
        record = receipt_arms[treatment]
        if not isinstance(record, Mapping):
            raise ValueError(f"{pair_id} legacy snapshot arm is invalid")
        _require_exact_keys(
            record,
            {
                "bytes",
                "curr_step",
                "row_identity_mode",
                "sha256",
                "snapshot_checkpoint",
                "source_checkpoint",
                "source_job_id",
                "state_dict_tensor_count",
            },
            name=f"{pair_id} legacy snapshot {treatment}",
        )
        if (
            record.get("bytes") != arm["checkpoint"].stat().st_size
            or record.get("curr_step") != comparison_step
            or record.get("row_identity_mode") != treatment
            or record.get("sha256") != arm["checkpoint_sha256"]
            or record.get("snapshot_checkpoint") != arm["checkpoint"].name
            or not isinstance(record.get("source_checkpoint"), str)
            or not Path(record["source_checkpoint"]).is_absolute()
            or not isinstance(record.get("source_job_id"), str)
            or not record["source_job_id"]
            or not isinstance(record.get("state_dict_tensor_count"), int)
            or isinstance(record.get("state_dict_tensor_count"), bool)
            or record["state_dict_tensor_count"] <= 0
        ):
            raise ValueError(f"{pair_id} legacy snapshot {treatment} is invalid")
        arm_facts[arm["arm_id"]] = {
            "state_dict_tensor_count": record["state_dict_tensor_count"]
        }
    return {
        "kind": "exploratory_same_step_pilot_checkpoint_pair",
        "formal_eligible": False,
        "arms": arm_facts,
    }


def _validate_legacy_continuation_snapshot(
    *,
    documents: Sequence[Mapping[str, Any]],
    pair_id: str,
    source_commit: str,
    seed: int,
    comparison_step: int,
    match_level: str,
    arms: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(documents) != 2:
        raise ValueError(
            f"{pair_id} continuation snapshot requires exactly two arm receipts"
        )
    expected_keys = {
        "schema_version",
        "kind",
        "classification",
        "continuation_id",
        "created_at_utc",
        "mode",
        "stage",
        "step",
        "continuation_source_commit",
        "source_provenance_status",
        "source_checkpoint",
        "snapshot_filename",
        "checkpoint_size_bytes",
        "checkpoint_sha256",
        "limitations",
        "manifest_sha256",
    }
    expected_limitations = [
        "checkpoint_has_no_source_sha",
        "checkpoint_has_no_rng_or_dataloader_state",
        "checkpoint_has_no_grad_scaler_state",
        "checkpoint_has_no_parent_lineage",
    ]
    treatment_lookup = {arm["treatment"]["kind"]: arm for arm in arms}
    if set(treatment_lookup) != {"rope", "none"}:
        raise ValueError(f"{pair_id} continuation must be RoPE versus No-RoPE")
    documents_by_mode: dict[str, Mapping[str, Any]] = {}
    continuation_ids: set[str] = set()
    arm_facts: dict[str, Any] = {}
    for document in documents:
        _require_exact_keys(
            document, expected_keys, name=f"{pair_id} continuation snapshot"
        )
        body = {
            key: value for key, value in document.items() if key != "manifest_sha256"
        }
        encoded_body = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if document.get("manifest_sha256") != hashlib.sha256(encoded_body).hexdigest():
            raise ValueError(f"{pair_id} continuation snapshot self-hash is invalid")
        mode = document.get("mode")
        continuation_id = document.get("continuation_id")
        if (
            document.get("schema_version") != 1
            or document.get("kind")
            != "tabicl-legacy-pilot-checkpoint-snapshot"
            or document.get("classification") != "exploratory-pilot-only"
            or not isinstance(continuation_id, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,95}", continuation_id)
            is None
            or not isinstance(document.get("created_at_utc"), str)
            or not document["created_at_utc"]
            or mode not in treatment_lookup
            or mode in documents_by_mode
            or document.get("stage") != "stage1"
            or document.get("step") != comparison_step
            or comparison_step != 500_000
            or document.get("continuation_source_commit") != source_commit
            or document.get("source_provenance_status")
            != "operational-history-only-not-checkpoint-bound"
            or not isinstance(document.get("source_checkpoint"), str)
            or not Path(document["source_checkpoint"]).is_absolute()
            or document.get("limitations") != expected_limitations
            or match_level != "same_step_legacy"
            or seed != 42
        ):
            raise ValueError(f"{pair_id} continuation snapshot header is invalid")
        arm = treatment_lookup[mode]
        if (
            document.get("snapshot_filename") != arm["checkpoint"].name
            or document.get("checkpoint_size_bytes")
            != arm["checkpoint"].stat().st_size
            or document.get("checkpoint_sha256") != arm["checkpoint_sha256"]
        ):
            raise ValueError(f"{pair_id} continuation snapshot arm binding is invalid")
        documents_by_mode[mode] = document
        continuation_ids.add(continuation_id)
        arm_facts[arm["arm_id"]] = {
            "checkpoint_size_bytes": document["checkpoint_size_bytes"],
            "manifest_sha256": document["manifest_sha256"],
            "mode": mode,
        }
    if set(documents_by_mode) != {"rope", "none"} or len(continuation_ids) != 1:
        raise ValueError(f"{pair_id} continuation snapshot pair lineage differs")
    return {
        "kind": "tabicl-legacy-pilot-checkpoint-snapshot-pair",
        "formal_eligible": False,
        "evidence_level": "legacy_same_step_no_exact_prior_or_rng",
        "continuation_id": next(iter(continuation_ids)),
        "continuation_source_commit": source_commit,
        "stage": "stage1",
        "step": comparison_step,
        "arms": arm_facts,
    }


def _validate_continuation_lineage(
    *,
    documents: Sequence[Mapping[str, Any]],
    pair_id: str,
    source_commit: str,
    seed: int,
    comparison_step: int,
    match_level: str,
    arms: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(documents) != 3:
        raise ValueError(f"{pair_id} continuation requires three exact receipts")
    submission_documents = [
        document
        for document in documents
        if document.get("kind") == "fingerprint_fullsize_continuation_release_receipt"
    ]
    completion_documents = [
        document
        for document in documents
        if document.get("kind") == "fingerprint_fullsize_segment_completion"
    ]
    if len(submission_documents) != 1 or len(completion_documents) != 2:
        raise ValueError(f"{pair_id} continuation receipt roster is invalid")
    submission = _validated_existing_self_hash(
        submission_documents[0],
        kind="fingerprint_fullsize_continuation_release_receipt",
        name=f"{pair_id} continuation submission",
    )
    expected_submission_keys = {
        "all_jobs_released",
        "capacity",
        "environment_manifest_sha256",
        "environment_sha256",
        "formal_eligible",
        "held_plan_file_sha256",
        "held_plan_manifest_sha256",
        "held_public_ref_sha",
        "jobs",
        "origin_source_commit",
        "parent_manifests",
        "qos",
        "run_kind",
        "scheduler_horizon_steps",
        "schema_version",
        "seed",
        "source_commit",
        "study",
        "time_limit",
    }
    _require_exact_keys(
        submission, expected_submission_keys, name=f"{pair_id} continuation submission"
    )
    if (
        submission.get("schema_version") != 1
        or submission.get("study")
        != "tabiclv2-fullsize-rope-fingerprint-continuation-v1"
        or submission.get("formal_eligible") is not False
        or submission.get("all_jobs_released") is not True
        or submission.get("run_kind") != "full"
        or submission.get("seed") != seed
        or submission.get("source_commit") != source_commit
        or submission.get("scheduler_horizon_steps") != 500_000
        or match_level != "exact_prior_stream"
    ):
        raise ValueError(f"{pair_id} continuation submission header is invalid")
    for name in (
        "environment_manifest_sha256",
        "environment_sha256",
        "held_plan_file_sha256",
        "held_plan_manifest_sha256",
    ):
        _require_hex(submission.get(name), name=f"{pair_id} {name}", length=64)
    _require_hex(
        submission.get("held_public_ref_sha"),
        name=f"{pair_id} held_public_ref_sha",
        length=40,
    )
    _require_hex(
        submission.get("origin_source_commit"),
        name=f"{pair_id} origin_source_commit",
        length=40,
    )
    if submission["environment_manifest_sha256"] != submission["environment_sha256"]:
        raise ValueError(f"{pair_id} continuation environment hashes differ")
    treatment_lookup = {arm["treatment"]["kind"]: arm for arm in arms}
    if set(treatment_lookup) != {"rope", "fingerprint"}:
        raise ValueError(f"{pair_id} continuation must be RoPE versus Fingerprint")
    jobs = submission.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError(f"{pair_id} continuation job roster is invalid")
    job_segments: dict[str, list[tuple[int, int]]] = {"rope": [], "fingerprint": []}
    job_ids: set[int] = set()
    for job in jobs:
        if not isinstance(job, Mapping) or set(job) != {
            "arm",
            "from_step",
            "job_id",
            "to_step",
        }:
            raise ValueError(f"{pair_id} continuation job is invalid")
        arm = job.get("arm")
        start = job.get("from_step")
        stop = job.get("to_step")
        job_id = job.get("job_id")
        if (
            arm not in job_segments
            or not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(stop, int)
            or isinstance(stop, bool)
            or start < 0
            or stop <= start
            or stop > comparison_step
            or not isinstance(job_id, int)
            or isinstance(job_id, bool)
            or job_id <= 0
            or job_id in job_ids
        ):
            raise ValueError(f"{pair_id} continuation job is invalid")
        job_ids.add(job_id)
        job_segments[arm].append((start, stop))
    if job_segments["rope"] != job_segments["fingerprint"]:
        raise ValueError(f"{pair_id} continuation arm segment plans differ")
    ordered = sorted(job_segments["rope"])
    if (
        not ordered
        or ordered[-1][1] != comparison_step
        or any(left[1] != right[0] for left, right in zip(ordered, ordered[1:]))
    ):
        raise ValueError(f"{pair_id} continuation segment chain is invalid")
    parents = submission.get("parent_manifests")
    if not isinstance(parents, Mapping) or set(parents) != set(treatment_lookup):
        raise ValueError(f"{pair_id} continuation parent roster is invalid")
    for record in parents.values():
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ValueError(f"{pair_id} continuation parent record is invalid")
        if not isinstance(record.get("path"), str) or not Path(record["path"]).is_absolute():
            raise ValueError(f"{pair_id} continuation parent path is invalid")
        _require_hex(
            record.get("sha256"), name=f"{pair_id} parent sha256", length=64
        )

    completions: dict[str, Mapping[str, Any]] = {}
    for document in completion_documents:
        payload = _validated_existing_self_hash(
            document,
            kind="fingerprint_fullsize_segment_completion",
            name=f"{pair_id} continuation completion",
        )
        expected_completion_keys = {
            "arm",
            "checkpoint",
            "environment_sha256",
            "formal_eligible",
            "from_step",
            "parent_checkpoint_sha256",
            "parent_manifest_path",
            "parent_manifest_sha256",
            "runtime",
            "scheduler_horizon_steps",
            "seed",
            "source_commit",
            "study",
            "to_step",
            "warmup_steps",
        }
        _require_exact_keys(
            payload,
            expected_completion_keys,
            name=f"{pair_id} continuation completion",
        )
        treatment = payload.get("arm")
        if treatment in completions or treatment not in treatment_lookup:
            raise ValueError(f"{pair_id} continuation completion arm is invalid")
        if (
            payload.get("study")
            != "tabiclv2-fullsize-rope-fingerprint-continuation-v1"
            or payload.get("formal_eligible") is not False
            or payload.get("seed") != seed
            or payload.get("source_commit") != source_commit
            or payload.get("scheduler_horizon_steps") != 500_000
            or payload.get("to_step") != comparison_step
            or (payload.get("from_step"), payload.get("to_step"))
            not in job_segments[treatment]
            or payload.get("environment_sha256")
            != submission["environment_sha256"]
        ):
            raise ValueError(f"{pair_id} continuation completion header is invalid")
        checkpoint = payload.get("checkpoint")
        if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
            "curr_step",
            "optimizer_state_entries",
            "path",
            "prior_manifest_sha256",
            "prior_schema_sha256",
            "scheduler_last_epoch",
            "scheduler_last_lr",
            "sha256",
            "size_bytes",
            "state_elements",
            "state_tensors",
        }:
            raise ValueError(f"{pair_id} continuation checkpoint record is invalid")
        arm = treatment_lookup[treatment]
        record_path = checkpoint.get("path")
        if (
            not isinstance(record_path, str)
            or not Path(record_path).is_absolute()
            or Path(record_path).resolve(strict=True) != arm["checkpoint"]
            or checkpoint.get("curr_step") != comparison_step
            or checkpoint.get("scheduler_last_epoch") != comparison_step
            or checkpoint.get("sha256") != arm["checkpoint_sha256"]
            or checkpoint.get("size_bytes") != arm["checkpoint"].stat().st_size
        ):
            raise ValueError(f"{pair_id} continuation checkpoint binding is invalid")
        for name in (
            "optimizer_state_entries",
            "state_elements",
            "state_tensors",
        ):
            value = checkpoint.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{pair_id} continuation checkpoint {name} is invalid")
        for name in ("prior_manifest_sha256", "prior_schema_sha256"):
            _require_hex(
                checkpoint.get(name), name=f"{pair_id} checkpoint {name}", length=64
            )
        for name in ("parent_checkpoint_sha256", "parent_manifest_sha256"):
            _require_hex(payload.get(name), name=f"{pair_id} {name}", length=64)
        if not isinstance(payload.get("parent_manifest_path"), str) or not Path(
            payload["parent_manifest_path"]
        ).is_absolute():
            raise ValueError(f"{pair_id} parent manifest path is invalid")
        runtime = payload.get("runtime")
        if (
            not isinstance(runtime, Mapping)
            or set(runtime) != {"cuda", "gpu", "python", "torch"}
            or runtime.get("gpu") != "NVIDIA H100 80GB HBM3"
            or any(not isinstance(value, str) or not value for value in runtime.values())
        ):
            raise ValueError(f"{pair_id} continuation runtime is invalid")
        completions[treatment] = payload
    if set(completions) != set(treatment_lookup):
        raise ValueError(f"{pair_id} continuation completion roster is invalid")
    rope_checkpoint = completions["rope"]["checkpoint"]
    fingerprint_checkpoint = completions["fingerprint"]["checkpoint"]
    for name in ("prior_manifest_sha256", "prior_schema_sha256"):
        if rope_checkpoint[name] != fingerprint_checkpoint[name]:
            raise ValueError(f"{pair_id} continuation prior {name} differs")
    if completions["rope"]["runtime"] != completions["fingerprint"]["runtime"]:
        raise ValueError(f"{pair_id} continuation runtimes differ")
    return {
        "kind": "fingerprint_fullsize_continuation_v1",
        "formal_eligible": False,
        "environment_sha256": submission["environment_sha256"],
        "prior_manifest_sha256": rope_checkpoint["prior_manifest_sha256"],
        "prior_schema_sha256": rope_checkpoint["prior_schema_sha256"],
        "runtime": dict(completions["rope"]["runtime"]),
        "arms": {
            treatment_lookup[treatment]["arm_id"]: {
                "optimizer_state_entries": completions[treatment]["checkpoint"][
                    "optimizer_state_entries"
                ],
                "scheduler_last_lr": completions[treatment]["checkpoint"][
                    "scheduler_last_lr"
                ],
                "state_elements": completions[treatment]["checkpoint"][
                    "state_elements"
                ],
                "state_tensors": completions[treatment]["checkpoint"]["state_tensors"],
            }
            for treatment in sorted(treatment_lookup)
        },
    }


def _validate_lineage_documents(
    *,
    documents: Sequence[Mapping[str, Any]],
    pair_id: str,
    source_commit: str,
    seed: int,
    comparison_step: int,
    match_level: str,
    arms: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    kinds = {document.get("kind") for document in documents}
    if kinds == {"exploratory_same_step_pilot_checkpoint_pair"}:
        return _validate_snapshot_lineage(
            documents=documents,
            pair_id=pair_id,
            source_commit=source_commit,
            seed=seed,
            comparison_step=comparison_step,
            match_level=match_level,
            arms=arms,
        )
    if kinds == {"tabicl-legacy-pilot-checkpoint-snapshot"}:
        return _validate_legacy_continuation_snapshot(
            documents=documents,
            pair_id=pair_id,
            source_commit=source_commit,
            seed=seed,
            comparison_step=comparison_step,
            match_level=match_level,
            arms=arms,
        )
    if kinds <= {
        "fingerprint_fullsize_continuation_release_receipt",
        "fingerprint_fullsize_segment_completion",
    }:
        return _validate_continuation_lineage(
            documents=documents,
            pair_id=pair_id,
            source_commit=source_commit,
            seed=seed,
            comparison_step=comparison_step,
            match_level=match_level,
            arms=arms,
        )
    raise ValueError(f"{pair_id} lineage receipt schema is unsupported")


def load_private_run_config(path: Path) -> dict[str, Any]:
    config_path = absolute_path(path, name="run_config")
    document, config_sha256 = load_json_object_with_sha256(
        config_path, name="private run config"
    )
    _require_exact_keys(
        document, {"schema_version", "study", "seed", "pairs"}, name="run config"
    )
    seed = document.get("seed")
    if (
        document.get("schema_version") != 1
        or document.get("study") != RUN_CONFIG_STUDY
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed != 42
    ):
        raise ValueError("run config header is invalid")
    pairs = document.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("run config must contain at least one pair")

    runtime_pairs: list[dict[str, Any]] = []
    portable_pairs: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    arm_ids: set[str] = set()
    for pair_index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping):
            raise ValueError("pair entry must be an object")
        _require_exact_keys(
            pair,
            {
                "pair_id",
                "comparison_step",
                "training_source_commit",
                "match_level",
                "arms",
                "lineage_receipts",
            },
            name=f"pair {pair_index}",
        )
        pair_id = _require_identifier(pair.get("pair_id"), name="pair_id")
        if pair_id in pair_ids:
            raise ValueError("pair IDs must be unique")
        pair_ids.add(pair_id)
        comparison_step = pair.get("comparison_step")
        if (
            not isinstance(comparison_step, int)
            or isinstance(comparison_step, bool)
            or comparison_step <= 0
        ):
            raise ValueError(f"{pair_id} comparison_step is invalid")
        source_commit = _require_hex(
            pair.get("training_source_commit"),
            name=f"{pair_id} training_source_commit",
            length=40,
        )
        match_level = pair.get("match_level")
        if match_level not in MATCH_LEVELS:
            raise ValueError(f"{pair_id} match_level is invalid")
        arms = pair.get("arms")
        if not isinstance(arms, list) or len(arms) != 2:
            raise ValueError(f"{pair_id} must contain exactly two arms")
        runtime_arms: list[dict[str, Any]] = []
        portable_arms: list[dict[str, Any]] = []
        treatments: list[dict[str, Any]] = []
        for arm_index, arm in enumerate(arms):
            if not isinstance(arm, Mapping):
                raise ValueError(f"{pair_id} arm must be an object")
            _require_exact_keys(
                arm,
                {"arm_id", "checkpoint_path", "checkpoint_sha256", "treatment"},
                name=f"{pair_id} arm {arm_index}",
            )
            arm_id = _require_identifier(arm.get("arm_id"), name="arm_id")
            if arm_id in arm_ids:
                raise ValueError("arm IDs must be globally unique")
            arm_ids.add(arm_id)
            checkpoint = absolute_path(
                str(arm.get("checkpoint_path")), name=f"{arm_id} checkpoint"
            )
            expected_sha = _require_hex(
                arm.get("checkpoint_sha256"),
                name=f"{arm_id} checkpoint_sha256",
                length=64,
            )
            if sha256_file(checkpoint) != expected_sha:
                raise ValueError(f"{arm_id} checkpoint digest mismatch")
            treatment = _treatment_contract(
                arm.get("treatment"), name=f"{arm_id} treatment"
            )
            treatments.append(treatment)
            runtime_arms.append(
                {
                    "arm_id": arm_id,
                    "checkpoint": checkpoint,
                    "checkpoint_sha256": expected_sha,
                    "treatment": treatment,
                }
            )
            portable_arms.append(
                {
                    "arm_id": arm_id,
                    "checkpoint_sha256": expected_sha,
                    "checkpoint_size_bytes": checkpoint.stat().st_size,
                    "treatment": treatment,
                }
            )
        if treatments[0] == treatments[1]:
            raise ValueError(f"{pair_id} arms must encode different treatments")

        receipts = pair.get("lineage_receipts")
        if not isinstance(receipts, list) or not receipts:
            raise ValueError(f"{pair_id} lineage receipt roster is empty")
        runtime_receipts: list[dict[str, Any]] = []
        portable_receipts: list[dict[str, Any]] = []
        receipt_digests: set[str] = set()
        receipt_documents: list[Mapping[str, Any]] = []
        for receipt_index, receipt in enumerate(receipts):
            if not isinstance(receipt, Mapping):
                raise ValueError(f"{pair_id} lineage receipt must be an object")
            _require_exact_keys(
                receipt,
                {"path", "sha256"},
                name=f"{pair_id} lineage receipt {receipt_index}",
            )
            receipt_path = absolute_path(
                str(receipt.get("path")), name=f"{pair_id} lineage receipt"
            )
            receipt_sha = _require_hex(
                receipt.get("sha256"),
                name=f"{pair_id} lineage receipt sha256",
                length=64,
            )
            if receipt_sha in receipt_digests:
                raise ValueError(f"{pair_id} lineage receipts must be unique")
            receipt_digests.add(receipt_sha)
            receipt_document, observed_receipt_sha = load_json_object_with_sha256(
                receipt_path, name=f"{pair_id} lineage receipt"
            )
            if observed_receipt_sha != receipt_sha:
                raise ValueError(f"{pair_id} lineage receipt digest mismatch")
            receipt_documents.append(receipt_document)
            runtime_receipts.append({"path": receipt_path, "sha256": receipt_sha})
            portable_receipts.append({"sha256": receipt_sha})

        lineage_contract = _validate_lineage_documents(
            documents=receipt_documents,
            pair_id=pair_id,
            source_commit=source_commit,
            seed=seed,
            comparison_step=comparison_step,
            match_level=match_level,
            arms=runtime_arms,
        )
        runtime_pair = {
            "pair_id": pair_id,
            "comparison_step": comparison_step,
            "training_source_commit": source_commit,
            "match_level": match_level,
            "arms": runtime_arms,
            "lineage_receipts": runtime_receipts,
            "lineage_contract": lineage_contract,
        }
        portable_pair = {
            "pair_id": pair_id,
            "comparison_step": comparison_step,
            "training_source_commit": source_commit,
            "match_level": match_level,
            "arms": portable_arms,
            "lineage_receipts": portable_receipts,
            "lineage_contract": lineage_contract,
        }
        runtime_pairs.append(runtime_pair)
        portable_pairs.append(portable_pair)

    portable = {
        "schema_version": 1,
        "study": RUN_CONFIG_STUDY,
        "seed": seed,
        "pairs": portable_pairs,
    }
    return {
        "path": config_path,
        "config_sha256": config_sha256,
        "seed": seed,
        "pairs": runtime_pairs,
        "arm_order": tuple(
            arm["arm_id"] for pair in runtime_pairs for arm in pair["arms"]
        ),
        "portable": portable,
        "portable_sha256": json_document_sha256(portable),
    }


def validate_checkpoint_pairs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate full-size state, treatment, step, and within-pair matching."""

    import torch

    pair_contracts: list[dict[str, Any]] = []
    for pair in config["pairs"]:
        arm_facts: list[dict[str, Any]] = []
        normalized_configs: list[dict[str, Any]] = []
        prior_streams: list[object] = []
        for arm in pair["arms"]:
            payload = torch.load(
                arm["checkpoint"], map_location="cpu", weights_only=True
            )
            if not isinstance(payload, Mapping):
                raise ValueError(f"{arm['arm_id']} checkpoint payload is invalid")
            if payload.get("curr_step") != pair["comparison_step"]:
                raise ValueError(f"{arm['arm_id']} checkpoint step is invalid")
            model_config = payload.get("config")
            state = payload.get("state_dict")
            if not isinstance(model_config, Mapping) or not isinstance(state, Mapping):
                raise ValueError(f"{arm['arm_id']} lacks model config/state")
            architecture = {
                key: model_config.get(key) for key in FULLSIZE_ARCHITECTURE
            }
            if architecture != FULLSIZE_ARCHITECTURE:
                raise ValueError(f"{arm['arm_id']} is not the frozen full-size model")
            treatment = arm["treatment"]
            kind = treatment["kind"]
            config_mode = model_config.get("row_identity_mode")
            config_fingerprint = model_config.get("row_fingerprint", False)
            if kind == "rope" and not (
                config_mode == "rope" and config_fingerprint is False
            ):
                raise ValueError(f"{arm['arm_id']} RoPE config is invalid")
            if kind == "none" and not (
                config_mode == "none" and config_fingerprint is False
            ):
                raise ValueError(f"{arm['arm_id']} No-RoPE config is invalid")
            if kind == "fingerprint" and not (
                config_mode == "none"
                and config_fingerprint is True
                and model_config.get("row_fingerprint_dim") == treatment["dimension"]
            ):
                raise ValueError(f"{arm['arm_id']} Fingerprint config is invalid")

            state_keys = set(state)
            rope_key = "row_interactor.tf_row.rope.freqs"
            fingerprint_fragments = (
                "row_interactor.fingerprint_q_gates",
                "row_interactor.fingerprint_k_gates",
                "row_interactor.fingerprint_q_projections.",
                "row_interactor.fingerprint_k_projections.",
            )
            any_fingerprint = any(
                "row_interactor.fingerprint_" in key for key in state_keys
            )
            has_fingerprint = all(
                any(fragment in key for key in state_keys)
                for fragment in fingerprint_fragments
            )
            if kind == "rope" and (rope_key not in state_keys or any_fingerprint):
                raise ValueError(f"{arm['arm_id']} RoPE state is invalid")
            if kind == "none" and (rope_key in state_keys or any_fingerprint):
                raise ValueError(f"{arm['arm_id']} No-RoPE state is invalid")
            if kind == "fingerprint" and (rope_key in state_keys or not has_fingerprint):
                raise ValueError(f"{arm['arm_id']} Fingerprint state is invalid")
            if kind == "fingerprint":
                dimension = treatment["dimension"]
                expected_fingerprint_shapes = {
                    "row_interactor.fingerprint_q_gates": (3,),
                    "row_interactor.fingerprint_k_gates": (3,),
                    **{
                        f"row_interactor.fingerprint_{side}_projections.{block}.1.weight": (
                            dimension,
                            128,
                        )
                        for side in ("q", "k")
                        for block in range(3)
                    },
                    **{
                        f"row_interactor.fingerprint_{side}_projections.{block}.3.weight": (
                            128,
                            dimension,
                        )
                        for side in ("q", "k")
                        for block in range(3)
                    },
                }
                observed_fingerprint_keys = {
                    key
                    for key in state_keys
                    if "row_interactor.fingerprint_" in key
                }
                if observed_fingerprint_keys != set(expected_fingerprint_shapes) or any(
                    tuple(state[key].shape) != shape
                    for key, shape in expected_fingerprint_shapes.items()
                ):
                    raise ValueError(
                        f"{arm['arm_id']} Fingerprint state roster is invalid"
                    )
            for key, value in state.items():
                if not hasattr(value, "numel"):
                    raise ValueError(f"{arm['arm_id']} state value {key} is invalid")
                if (value.is_floating_point() or value.is_complex()) and not bool(
                    torch.isfinite(value).all()
                ):
                    raise ValueError(f"{arm['arm_id']} contains non-finite state")

            treatment_keys = {
                "row_identity_mode",
                "row_fingerprint",
                "row_fingerprint_dim",
            }
            normalized_configs.append(
                {
                    key: value
                    for key, value in model_config.items()
                    if key not in treatment_keys
                }
            )
            prior_stream = payload.get("prior_stream")
            identity_treatment = payload.get("identity_treatment")
            if pair["match_level"] == "same_step_legacy":
                if prior_stream is not None or identity_treatment is not None:
                    raise ValueError(
                        f"{arm['arm_id']} legacy checkpoint metadata is unexpected"
                    )
                declared = pair["lineage_contract"]["arms"][arm["arm_id"]]
                declared_count = declared.get("state_dict_tensor_count")
                if declared_count is not None and declared_count != len(state):
                    raise ValueError(
                        f"{arm['arm_id']} legacy lineage state count is invalid"
                    )
            else:
                if not isinstance(prior_stream, Mapping) or set(prior_stream) != {
                    "schema_version",
                    "algorithm",
                    "schema",
                    "schema_sha256",
                    "experiment_seed",
                    "ddp_rank",
                    "world_size",
                    "cursor",
                    "manifest_sha256",
                }:
                    raise ValueError(f"{arm['arm_id']} prior stream is invalid")
                schema = prior_stream.get("schema")
                try:
                    prior_schema = json.loads(schema) if isinstance(schema, str) else None
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{arm['arm_id']} prior schema is invalid"
                    ) from error
                if (
                    prior_stream.get("schema_version") != 1
                    or prior_stream.get("algorithm")
                    != "sha256-schema-seed-rank-logical-step-v1"
                    or not isinstance(prior_schema, Mapping)
                    or canonical_json_sha256(prior_schema)
                    != prior_stream.get("schema_sha256")
                    or prior_schema.get("schema_version") != 1
                    or prior_schema.get("batch_size") != 64
                    or prior_schema.get("batch_size_per_gp") != 8
                    or prior_schema.get("max_features") != 100
                    or prior_schema.get("max_classes") != 10
                    or prior_schema.get("max_seq_len") != 1024
                    or prior_schema.get("prior_type") != "graph_scm"
                    or prior_stream.get("experiment_seed") != config["seed"]
                    or prior_stream.get("ddp_rank") != 0
                    or prior_stream.get("world_size") != 1
                    or prior_stream.get("cursor") != pair["comparison_step"]
                    or prior_stream.get("manifest_sha256")
                    != pair["lineage_contract"]["prior_manifest_sha256"]
                    or prior_stream.get("schema_sha256")
                    != pair["lineage_contract"]["prior_schema_sha256"]
                ):
                    raise ValueError(f"{arm['arm_id']} prior stream contract is invalid")
                if not isinstance(identity_treatment, Mapping) or set(
                    identity_treatment
                ) != {
                    "schema_version",
                    "row_identity_mode",
                    "identity_rng_seed",
                    "seed_policy",
                    "sampler_version",
                    "world_size",
                    "manifest_sha256",
                }:
                    raise ValueError(
                        f"{arm['arm_id']} identity treatment metadata is invalid"
                    )
                if (
                    identity_treatment.get("schema_version") != 1
                    or identity_treatment.get("row_identity_mode") != config_mode
                    or identity_treatment.get("identity_rng_seed") != config["seed"]
                    or identity_treatment.get("seed_policy")
                    != "sha256-domain-separated-base-seed-and-rank-v1"
                    or identity_treatment.get("sampler_version") is not None
                    or identity_treatment.get("world_size") != 1
                    or not isinstance(identity_treatment.get("manifest_sha256"), str)
                    or HEX64.fullmatch(identity_treatment["manifest_sha256"]) is None
                ):
                    raise ValueError(
                        f"{arm['arm_id']} identity treatment contract is invalid"
                    )
                optimizer = payload.get("optimizer_state")
                scheduler = payload.get("scheduler_state")
                declared = pair["lineage_contract"]["arms"][arm["arm_id"]]
                if (
                    not isinstance(optimizer, Mapping)
                    or set(optimizer) != {"state", "param_groups"}
                    or not isinstance(optimizer.get("state"), Mapping)
                    or len(optimizer["state"]) != declared["optimizer_state_entries"]
                    or not isinstance(scheduler, Mapping)
                    or scheduler.get("last_epoch") != pair["comparison_step"]
                    or scheduler.get("_last_lr") != [declared["scheduler_last_lr"]]
                    or declared["state_tensors"] != len(state)
                    or declared["state_elements"]
                    != sum(int(value.numel()) for value in state.values())
                ):
                    raise ValueError(
                        f"{arm['arm_id']} optimizer/scheduler lineage is invalid"
                    )
            prior_streams.append(prior_stream)
            arm_facts.append(
                {
                    "arm_id": arm["arm_id"],
                    "checkpoint_sha256": arm["checkpoint_sha256"],
                    "checkpoint_size_bytes": arm["checkpoint"].stat().st_size,
                    "curr_step": payload["curr_step"],
                    "state_tensors": len(state),
                    "state_elements": sum(int(value.numel()) for value in state.values()),
                    "architecture": architecture,
                    "treatment": treatment,
                }
            )
            del payload, state
        if normalized_configs[0] != normalized_configs[1]:
            raise ValueError(f"{pair['pair_id']} configs differ outside treatment")
        if pair["match_level"] == "exact_prior_stream":
            if (
                not isinstance(prior_streams[0], Mapping)
                or prior_streams[0] != prior_streams[1]
            ):
                raise ValueError(f"{pair['pair_id']} prior streams are not exact")
            prior_stream_sha256 = canonical_json_sha256(prior_streams[0])
        else:
            prior_stream_sha256 = None
        pair_contracts.append(
            {
                "pair_id": pair["pair_id"],
                "comparison_step": pair["comparison_step"],
                "match_level": pair["match_level"],
                "training_source_commit": pair["training_source_commit"],
                "prior_stream_sha256": prior_stream_sha256,
                "arms": arm_facts,
            }
        )
    return {"schema_version": 1, "pairs": pair_contracts}


def build_shard_plan_payload(
    dataset_records: Sequence[Mapping[str, Any]], *, shard_count: int
) -> dict[str, Any]:
    if not isinstance(shard_count, int) or isinstance(shard_count, bool) or shard_count <= 0:
        raise ValueError("shard_count must be a positive integer")
    if shard_count > len(dataset_records):
        raise ValueError("shard_count cannot exceed dataset count")
    records: list[dict[str, Any]] = []
    names: list[str] = []
    expected_record_keys = {
        "ordinal",
        "name",
        "n_train",
        "n_validation",
        "n_test",
        "n_features",
        "n_classes",
        "info_sha256",
        "input_sha256",
    }
    for ordinal, record in enumerate(dataset_records):
        if not isinstance(record, Mapping):
            raise ValueError("dataset record must be an object")
        _require_exact_keys(record, expected_record_keys, name="dataset record")
        name = record.get("name")
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\0" in name
        ):
            raise ValueError("dataset name is invalid")
        if record.get("ordinal") != ordinal:
            raise ValueError("dataset ordinals must be canonical and contiguous")
        names.append(name)
        n_train = record.get("n_train")
        n_validation = record.get("n_validation")
        n_test = record.get("n_test")
        n_features = record.get("n_features")
        n_classes = record.get("n_classes")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (n_train, n_validation, n_test, n_features, n_classes)
        ):
            raise ValueError(f"dataset cost metadata is invalid for {name}")
        if n_features > 500 or not 2 <= n_classes <= 10:
            raise ValueError(f"dataset eligibility metadata is invalid for {name}")
        _require_hex(
            record.get("info_sha256"), name=f"{name} info_sha256", length=64
        )
        inputs = record.get("input_sha256")
        if (
            not isinstance(inputs, Mapping)
            or not inputs
            or "info.json" not in inputs
            or inputs.get("info.json") != record.get("info_sha256")
            or any(
                not isinstance(key, str)
                or not key
                or Path(key).name != key
                or not isinstance(value, str)
                or HEX64.fullmatch(value) is None
                for key, value in inputs.items()
            )
        ):
            raise ValueError(f"dataset input hashes are invalid for {name}")
        cost_units = int((n_train + n_validation) * (100 + n_features))
        current = dict(record)
        current["cost_units"] = cost_units
        records.append(current)
    if len(set(names)) != len(names):
        raise ValueError("dataset names must be unique")

    buckets = [
        {"shard_id": f"{index:03d}", "dataset_ordinals": [], "cost_units": 0}
        for index in range(shard_count)
    ]
    for record in sorted(records, key=lambda item: (-item["cost_units"], item["name"])):
        bucket = min(buckets, key=lambda item: (item["cost_units"], item["shard_id"]))
        bucket["dataset_ordinals"].append(record["ordinal"])
        bucket["cost_units"] += record["cost_units"]
    for bucket in buckets:
        bucket["dataset_ordinals"].sort()
    return {
        "schema_version": 1,
        "roster_kind": ROSTER_KIND,
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "roster_sha256": roster_sha256(names),
        "cost_formula": "(n_train+n_validation)*(100+n_features)",
        "shard_count": shard_count,
        "datasets": records,
        "shards": buckets,
    }


def validate_shard_plan(
    document: Mapping[str, Any], *, expected_roster: Sequence[str]
) -> Mapping[str, Any]:
    payload = validate_self_hashed_document(
        document, kind=PLAN_KIND, name="TALENT shard plan"
    )
    expected_names = list(expected_roster)
    if (
        payload.get("schema_version") != 1
        or payload.get("roster_kind") != ROSTER_KIND
        or payload.get("split_manifest_sha256") != SPLIT_MANIFEST_SHA256
        or payload.get("roster_sha256") != roster_sha256(expected_names)
        or payload.get("cost_formula")
        != "(n_train+n_validation)*(100+n_features)"
    ):
        raise ValueError("TALENT shard plan header is invalid")
    records = payload.get("datasets")
    shards = payload.get("shards")
    shard_count = payload.get("shard_count")
    if (
        not isinstance(records, list)
        or not isinstance(shards, list)
        or not isinstance(shard_count, int)
        or isinstance(shard_count, bool)
        or len(records) != len(expected_names)
        or len(shards) != shard_count
    ):
        raise ValueError("TALENT shard plan roster is invalid")
    if [record.get("name") for record in records if isinstance(record, Mapping)] != expected_names:
        raise ValueError("TALENT shard plan dataset ordering changed")
    if [record.get("ordinal") for record in records if isinstance(record, Mapping)] != list(
        range(len(expected_names))
    ):
        raise ValueError("TALENT shard plan ordinals are invalid")
    observed_ordinals: list[int] = []
    observed_ids: list[str] = []
    for index, shard in enumerate(shards):
        if not isinstance(shard, Mapping):
            raise ValueError("TALENT shard record is invalid")
        expected_id = f"{index:03d}"
        if shard.get("shard_id") != expected_id:
            raise ValueError("TALENT shard IDs are invalid")
        ordinals = shard.get("dataset_ordinals")
        if (
            not isinstance(ordinals, list)
            or not ordinals
            or ordinals != sorted(ordinals)
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                or value >= len(records)
                for value in ordinals
            )
        ):
            raise ValueError(f"TALENT shard {expected_id} ordinals are invalid")
        expected_cost = sum(records[value]["cost_units"] for value in ordinals)
        if shard.get("cost_units") != expected_cost:
            raise ValueError(f"TALENT shard {expected_id} cost is invalid")
        observed_ordinals.extend(ordinals)
        observed_ids.append(expected_id)
    if sorted(observed_ordinals) != list(range(len(expected_names))) or len(
        observed_ordinals
    ) != len(set(observed_ordinals)):
        raise ValueError("TALENT shard union is not exact")
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("TALENT shard IDs are not unique")
    rebuilt_records = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("TALENT shard plan dataset record is invalid")
        rebuilt_records.append(
            {key: value for key, value in record.items() if key != "cost_units"}
        )
    rebuilt = build_shard_plan_payload(rebuilt_records, shard_count=shard_count)
    for key in (
        "schema_version",
        "roster_kind",
        "split_manifest_sha256",
        "roster_sha256",
        "cost_formula",
        "shard_count",
        "datasets",
        "shards",
    ):
        if payload.get(key) != rebuilt[key]:
            raise ValueError("TALENT shard plan is not the deterministic LPT plan")
    if set(payload) != set(rebuilt) | {"analysis_sha"}:
        raise ValueError("TALENT shard plan payload keys are invalid")
    _require_hex(payload.get("analysis_sha"), name="plan analysis_sha", length=40)
    return payload


def canary_dataset_records(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = plan.get("datasets")
    if not isinstance(records, list):
        raise ValueError("canary plan dataset roster is invalid")
    largest = sorted(
        records, key=lambda record: (-record["cost_units"], record["name"])
    )[:2]
    madeline = next((record for record in records if record["name"] == "madeline"), None)
    if madeline is None:
        raise ValueError("frozen TALENT roster lacks the madeline canary")
    by_ordinal = {record["ordinal"]: record for record in [*largest, madeline]}
    return [by_ordinal[ordinal] for ordinal in sorted(by_ordinal)]


def directory_manifest(root: Path, *, kind: str) -> dict[str, Any]:
    artifacts: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path == root / "manifest.json":
            continue
        if path.is_symlink():
            raise ValueError(f"artifact is a symlink: {path.relative_to(root)}")
        if path.is_file():
            artifacts[path.relative_to(root).as_posix()] = sha256_file(path)
        elif not path.is_dir():
            raise ValueError(f"artifact is not regular: {path.relative_to(root)}")
    return {"schema_version": 1, "kind": kind, "artifacts": artifacts}


def validate_directory_manifest(root: Path, *, kind: str) -> Mapping[str, str]:
    manifest_path = root / "manifest.json"
    document = load_json_object(manifest_path, name=f"{kind} manifest")
    if (
        document.get("schema_version") != 1
        or document.get("kind") != kind
        or not isinstance(document.get("artifacts"), Mapping)
        or set(document) != {"schema_version", "kind", "artifacts"}
    ):
        raise ValueError(f"{kind} manifest contract is invalid")
    expected = directory_manifest(root, kind=kind)["artifacts"]
    if document["artifacts"] != expected:
        raise ValueError(f"{kind} manifest does not match artifact bytes")
    return document["artifacts"]


def exact_sign_test(left_wins: int, right_wins: int) -> float:
    n = left_wins + right_wins
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, index) for index in range(min(left_wins, right_wins) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def verify_clean_detached_git(root: Path, *, expected_sha: str) -> str:
    _require_hex(expected_sha, name="expected Git SHA", length=40)
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise ValueError(f"source checkout is not clean: {root}")
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected_sha:
        raise ValueError(f"source checkout differs from expected SHA: {root}")
    symbolic = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "-q", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if symbolic.returncode == 0:
        raise ValueError(f"source checkout must be detached: {root}")
    if symbolic.returncode != 1:
        raise RuntimeError(f"could not establish detached HEAD state: {root}")
    return head
