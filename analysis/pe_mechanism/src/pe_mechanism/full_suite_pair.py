"""Resumable, fail-closed infrastructure for two-arm benchmark comparisons.

The module is deliberately benchmark-runtime agnostic.  A shard runner supplies
one callback that evaluates both arms for one dataset; this module owns the
immutable run contract, deterministic sharding, atomic task publication,
resume validation, and complete aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_PROBLEM_TYPES = frozenset({"binary", "multiclass"})
_SPLIT_REGIMES = frozenset({"iid", "grouped", "temporal"})
_BOOTSTRAP_SEED = 20_260_817
_BOOTSTRAP_RESAMPLES = 20_000
_PINNED_ROSTERS = {
    "beyondarena": {
        "count": 89,
        "names_sha256": "7b440b2676184bd0fe7b2e604f4aa6ad22e76a54cf74b626a79e8b58bd172473",
        "source_commit": "c987d91556a14d4c9b3383c35d1b0ec68ff81883",
    },
    "tabarena-v0.1": {
        "count": 38,
        "names_sha256": "b1f71e48085451cbe2126cbc02799c3969ecf9e7965f6aa71757db63fd6003ee",
        "source_commit": "c987d91556a14d4c9b3383c35d1b0ec68ff81883",
    },
}
_OOM_FALLBACK_POLICY = (
    {"level": 0, "name": "batch8-auto", "batch_size": 8, "offload_mode": "auto"},
    {"level": 1, "name": "batch4-auto", "batch_size": 4, "offload_mode": "auto"},
    {"level": 2, "name": "batch4-cpu", "batch_size": 4, "offload_mode": "cpu"},
    {"level": 3, "name": "batch4-disk", "batch_size": 4, "offload_mode": "disk"},
)


class PairTaskOOM(RuntimeError):
    """The only exception that authorizes a fresh paired fallback attempt."""
_FIXED_CLASSIFIER_OPTIONS = {
    "norm_methods": ["none", "power"],
    "feat_shuffle_method": "latin",
    "class_shuffle_method": "shift",
    "outlier_threshold": 4.0,
    "softmax_temperature": 0.9,
    "average_logits": True,
    "support_many_classes": True,
    "batch_size": 8,
    "kv_cache": False,
    "use_amp": "auto",
    "use_fa3": False,
    "offload_mode": "auto",
    "verbose": False,
}


@dataclass(frozen=True)
class Arm:
    arm_id: str
    display_name: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_size_bytes: int


@dataclass(frozen=True)
class PairManifest:
    path: Path
    sha256: str
    pair_id: str
    formal_eligible: bool
    model_source_sha: str
    arms: tuple[Arm, Arm]
    inference: Mapping[str, Any]
    checkpoint_contract: Mapping[str, Any]
    legacy_snapshot_receipts: Mapping[str, Path] | None
    training_receipts: Mapping[str, Path] | None

    @property
    def arm_order(self) -> tuple[str, str]:
        return tuple(arm.arm_id for arm in self.arms)  # type: ignore[return-value]


@dataclass(frozen=True)
class Roster:
    path: Path
    sha256: str
    names_sha256: str
    suite_id: str
    source_commit: str
    names: tuple[str, ...]


@dataclass(frozen=True)
class ShardPlan:
    path: Path
    sha256: str
    source_commit: str
    roster_names_sha256: str
    assignments: tuple[tuple[str, ...], ...]
    estimated_cost_units: tuple[int, ...]
    assignment_sha256: str
    canary_names: tuple[str, ...]
    canary_requirements: Mapping[str, str]

    @property
    def shard_count(self) -> int:
        return len(self.assignments)


@dataclass(frozen=True)
class Task:
    index: int
    name: str


def _exact_object(
    value: object, *, label: str, fields: set[str]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        raise ValueError(
            f"{label} fields mismatch: missing={missing}, unknown={unknown}"
        )
    return value


def _positive_int(value: object, *, label: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _git_sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase Git object id")
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a safe lowercase identifier")
    return value


def _real_file(value: object, *, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an absolute path string")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    _reject_symlink_components(path, label=label)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a real regular file")
    return path.resolve(strict=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    try:
        rendered = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            ensure_ascii=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("payload is not finite canonical JSON") from error
    return (rendered + "\n").encode("utf-8")


def _compact_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _reject_symlink_components(path: Path, *, label: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"{label} must not traverse symlinks: {current}")


def _real_directory(path: Path, *, label: str, create: bool = False) -> Path:
    _reject_symlink_components(path, label=label)
    if create:
        path.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(path, label=label)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a real directory")
    return path


def _load_json_file(path: Path, *, label: str) -> tuple[Mapping[str, Any], str]:
    if not path.is_absolute():
        path = path.absolute()
    _reject_symlink_components(path, label=label)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a real regular file")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def load_pair_manifest(path: Path, *, verify_checkpoints: bool = True) -> PairManifest:
    """Load an exact two-arm manifest and optionally verify checkpoint bytes."""
    payload, manifest_sha = _load_json_file(path, label="pair manifest")
    base_fields = {
        "schema_version",
        "kind",
        "pair_id",
        "formal_eligible",
        "model_source_sha",
        "arms",
        "inference",
    }
    optional_fields = {
        name
        for name in ("legacy_snapshot_receipts", "training_receipts")
        if name in payload
    }
    payload = _exact_object(
        payload,
        label="pair manifest",
        fields=base_fields | optional_fields,
    )
    if payload["schema_version"] != 1 or payload["kind"] != "two_arm_checkpoint_pair":
        raise ValueError("unsupported pair manifest schema or kind")
    pair_id = _identifier(payload["pair_id"], label="pair_id")
    formal_eligible = payload["formal_eligible"]
    if formal_eligible is not False:
        raise ValueError("every full-suite pair campaign must set formal_eligible=false")
    model_sha = _git_sha(payload["model_source_sha"], label="model_source_sha")

    arm_values = payload["arms"]
    if not isinstance(arm_values, list) or len(arm_values) != 2:
        raise ValueError("pair manifest must contain exactly two arms")
    arms: list[Arm] = []
    for index, raw_arm in enumerate(arm_values):
        arm = _exact_object(
            raw_arm,
            label=f"arms[{index}]",
            fields={
                "arm_id",
                "display_name",
                "checkpoint_path",
                "checkpoint_sha256",
                "checkpoint_size_bytes",
            },
        )
        arm_id = _identifier(arm["arm_id"], label=f"arms[{index}].arm_id")
        display_name = arm["display_name"]
        if not isinstance(display_name, str) or not display_name.strip():
            raise ValueError(f"arms[{index}].display_name must be non-empty")
        checkpoint_sha = _digest(
            arm["checkpoint_sha256"], label=f"arms[{index}].checkpoint_sha256"
        )
        checkpoint_size = _positive_int(
            arm["checkpoint_size_bytes"],
            label=f"arms[{index}].checkpoint_size_bytes",
        )
        raw_path = arm["checkpoint_path"]
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise ValueError(f"arms[{index}].checkpoint_path must be absolute")
        checkpoint = Path(raw_path)
        if verify_checkpoints:
            checkpoint = _real_file(raw_path, label=f"arms[{index}].checkpoint_path")
            if checkpoint.stat().st_size != checkpoint_size:
                raise ValueError(f"checkpoint size mismatch for arm {arm_id}")
            if sha256_file(checkpoint) != checkpoint_sha:
                raise ValueError(f"checkpoint digest mismatch for arm {arm_id}")
        arms.append(
            Arm(
                arm_id=arm_id,
                display_name=display_name.strip(),
                checkpoint_path=checkpoint,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_size_bytes=checkpoint_size,
            )
        )
    if len({arm.arm_id for arm in arms}) != 2:
        raise ValueError("pair arm identifiers must be unique")
    if len({arm.display_name for arm in arms}) != 2:
        raise ValueError("pair arm display names must be unique")
    legacy_receipts: dict[str, Path] | None = None
    if "legacy_snapshot_receipts" in payload:
        raw_receipts = _exact_object(
            payload["legacy_snapshot_receipts"],
            label="legacy_snapshot_receipts",
            fields={arm.arm_id for arm in arms},
        )
        legacy_receipts = {
            arm.arm_id: _real_file(
                raw_receipts[arm.arm_id],
                label=f"legacy snapshot receipt for {arm.arm_id}",
            )
            for arm in arms
        }
    training_receipts: dict[str, Path] | None = None
    if "training_receipts" in payload:
        required_training_receipts = {
            "submission",
            "rope_completion",
            "fingerprint_completion",
        }
        raw_receipts = _exact_object(
            payload["training_receipts"],
            label="training_receipts",
            fields=required_training_receipts,
        )
        training_receipts = {
            name: _real_file(value, label=f"training receipt {name}")
            for name, value in raw_receipts.items()
        }

    inference = _exact_object(
        payload["inference"],
        label="inference",
        fields={"device", "seed", "n_estimators", "classifier_options"},
    )
    if inference["device"] != "cuda":
        raise ValueError("pair evaluation device must be cuda")
    if inference["seed"] != 42:
        raise ValueError("pair evaluation requires inference.seed=42")
    if inference["n_estimators"] != 1:
        raise ValueError("pair evaluation requires n_estimators=1")
    if inference["classifier_options"] != _FIXED_CLASSIFIER_OPTIONS:
        raise ValueError("classifier options differ from the fixed pair contract")
    normalized_inference = json.loads(_canonical_json(dict(inference)))
    pair = PairManifest(
        path=path.resolve(strict=True),
        sha256=manifest_sha,
        pair_id=pair_id,
        formal_eligible=formal_eligible,
        model_source_sha=model_sha,
        arms=(arms[0], arms[1]),
        inference=normalized_inference,
        checkpoint_contract={},
        legacy_snapshot_receipts=legacy_receipts,
        training_receipts=training_receipts,
    )
    if verify_checkpoints:
        contract = validate_checkpoint_pair(pair)
        pair = PairManifest(
            path=pair.path,
            sha256=pair.sha256,
            pair_id=pair.pair_id,
            formal_eligible=pair.formal_eligible,
            model_source_sha=pair.model_source_sha,
            arms=pair.arms,
            inference=pair.inference,
            checkpoint_contract=contract,
            legacy_snapshot_receipts=pair.legacy_snapshot_receipts,
            training_receipts=pair.training_receipts,
        )
    return pair


def validate_checkpoint_pair(pair: PairManifest) -> dict[str, Any]:
    """Verify treatment semantics, matched configs, steps, and prior stream lineage."""
    import torch

    def validate_legacy_receipt(
        arm: Arm, path: Path, *, expected_step: int
    ) -> dict[str, Any]:
        receipt, receipt_file_sha = _load_json_file(
            path, label=f"legacy snapshot receipt for {arm.arm_id}"
        )
        receipt = _exact_object(
            receipt,
            label=f"legacy snapshot receipt for {arm.arm_id}",
            fields={
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
            },
        )
        body = {key: value for key, value in receipt.items() if key != "manifest_sha256"}
        expected_manifest_sha = hashlib.sha256(_compact_json(body)).hexdigest()
        if receipt["manifest_sha256"] != expected_manifest_sha:
            raise ValueError(f"legacy snapshot receipt self-hash failed for {arm.arm_id}")
        filename = receipt["snapshot_filename"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("legacy snapshot filename must be one safe basename")
        snapshot = _real_file(
            str(path.parent / filename),
            label=f"legacy snapshot checkpoint for {arm.arm_id}",
        )
        limitations = receipt["limitations"]
        if not isinstance(limitations, list) or not all(
            isinstance(value, str) and value for value in limitations
        ):
            raise ValueError("legacy snapshot receipt limitations must be explicit")
        if (
            receipt["schema_version"] != 1
            or receipt["kind"] != "tabicl-legacy-pilot-checkpoint-snapshot"
            or receipt["classification"] != "exploratory-pilot-only"
            or receipt["mode"] != arm.arm_id
            or receipt["stage"] != "stage1"
            or receipt["step"] != expected_step
            or receipt["source_provenance_status"]
            != "operational-history-only-not-checkpoint-bound"
            or receipt["checkpoint_size_bytes"] != arm.checkpoint_size_bytes
            or receipt["checkpoint_sha256"] != arm.checkpoint_sha256
            or snapshot != arm.checkpoint_path.resolve(strict=True)
            or snapshot.stat().st_size != arm.checkpoint_size_bytes
            or sha256_file(snapshot) != arm.checkpoint_sha256
        ):
            raise ValueError(
                f"legacy snapshot receipt does not bind the {arm.arm_id} checkpoint"
            )
        continuation_id = receipt["continuation_id"]
        if not isinstance(continuation_id, str) or not continuation_id:
            raise ValueError("legacy continuation_id must be non-empty")
        source_commit = _git_sha(
            receipt["continuation_source_commit"],
            label="legacy continuation_source_commit",
        )
        if source_commit != pair.model_source_sha:
            raise ValueError(
                "legacy snapshot continuation source must equal pair model_source_sha"
            )
        return {
            "receipt_file_sha256": receipt_file_sha,
            "receipt_manifest_sha256": expected_manifest_sha,
            "continuation_id": continuation_id,
            "continuation_source_commit": source_commit,
            "limitations_sha256": hashlib.sha256(
                _compact_json({"limitations": limitations})
            ).hexdigest(),
        }

    def validate_self_hashed_document(
        path: Path, *, label: str, expected_kind: str
    ) -> tuple[Mapping[str, Any], str]:
        document, file_sha = _load_json_file(path, label=label)
        document = _exact_object(
            document,
            label=label,
            fields={"schema_version", "kind", "payload", "sha256"},
        )
        payload = document["payload"]
        if not isinstance(payload, Mapping):
            raise ValueError(f"{label} payload must be an object")
        core = {"kind": expected_kind, "payload": payload, "schema_version": 1}
        if (
            document["schema_version"] != 1
            or document["kind"] != expected_kind
            or document["sha256"]
            != hashlib.sha256(_compact_json(core)).hexdigest()
        ):
            raise ValueError(f"{label} schema, kind, or self-hash is invalid")
        return payload, file_sha

    def validate_training_receipts(
        *, streams: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Any]:
        if pair.training_receipts is None:
            raise ValueError(
                "step-50000 rope/fingerprint requires submission and completion receipts"
            )
        submission, submission_sha = validate_self_hashed_document(
            pair.training_receipts["submission"],
            label="training submission receipt",
            expected_kind="fingerprint_fullsize_continuation_release_receipt",
        )
        common_expected = {
            "formal_eligible": False,
            "seed": 42,
            "source_commit": pair.model_source_sha,
            "study": "tabiclv2-fullsize-rope-fingerprint-continuation-v1",
            "scheduler_horizon_steps": 500_000,
        }
        if any(submission.get(key) != value for key, value in common_expected.items()):
            raise ValueError("training submission receipt contract is invalid")
        jobs = submission.get("jobs")
        if not isinstance(jobs, list):
            raise ValueError("training submission receipt job roster is invalid")
        terminal_jobs = {
            item.get("arm")
            for item in jobs
            if isinstance(item, Mapping) and item.get("to_step") == 50_000
        }
        if terminal_jobs != {"rope", "fingerprint"}:
            raise ValueError("training submission receipt lacks both terminal jobs")

        evidence: dict[str, Any] = {
            "submission_file_sha256": submission_sha,
            "submission_document_sha256": hashlib.sha256(
                _compact_json(
                    {
                        "kind": "fingerprint_fullsize_continuation_release_receipt",
                        "payload": submission,
                        "schema_version": 1,
                    }
                )
            ).hexdigest(),
            "completions": {},
        }
        environment_sha = submission.get("environment_sha256")
        _digest(environment_sha, label="training submission environment_sha256")
        for arm in pair.arms:
            completion, completion_sha = validate_self_hashed_document(
                pair.training_receipts[f"{arm.arm_id}_completion"],
                label=f"{arm.arm_id} training completion receipt",
                expected_kind="fingerprint_fullsize_segment_completion",
            )
            expected = {
                **common_expected,
                "arm": arm.arm_id,
                "to_step": 50_000,
                "environment_sha256": environment_sha,
            }
            if any(completion.get(key) != value for key, value in expected.items()):
                raise ValueError(
                    f"{arm.arm_id} training completion receipt contract is invalid"
                )
            checkpoint = completion.get("checkpoint")
            if not isinstance(checkpoint, Mapping):
                raise ValueError("training completion checkpoint record is invalid")
            stream = streams[arm.arm_id]
            checkpoint_expected = {
                "curr_step": 50_000,
                "sha256": arm.checkpoint_sha256,
                "size_bytes": arm.checkpoint_size_bytes,
                "prior_manifest_sha256": stream["manifest_sha256"],
                "prior_schema_sha256": stream["schema_sha256"],
            }
            if any(checkpoint.get(key) != value for key, value in checkpoint_expected.items()):
                raise ValueError(
                    f"{arm.arm_id} completion receipt does not bind its checkpoint"
                )
            evidence["completions"][arm.arm_id] = {
                "file_sha256": completion_sha,
                "document_sha256": hashlib.sha256(
                    _compact_json(
                        {
                            "kind": "fingerprint_fullsize_segment_completion",
                            "payload": completion,
                            "schema_version": 1,
                        }
                    )
                ).hexdigest(),
            }
        return evidence

    def validate_prior_stream(stream: Mapping[str, Any], *, step: int) -> dict[str, Any]:
        expected_fields = {
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
        if set(stream) != expected_fields:
            raise ValueError("paired checkpoint prior_stream fields are invalid")
        schema = stream.get("schema")
        if not isinstance(schema, str):
            raise ValueError("paired checkpoint prior_stream schema is invalid")
        try:
            parsed_schema = json.loads(
                schema,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite prior schema constant: {token}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError("paired checkpoint prior_stream schema is invalid") from error
        canonical_schema = json.dumps(
            parsed_schema,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        if canonical_schema != schema:
            raise ValueError("paired checkpoint prior_stream schema is not canonical")
        if (
            stream.get("schema_version") != 1
            or stream.get("algorithm")
            != "sha256-schema-seed-rank-logical-step-v1"
            or hashlib.sha256(schema.encode("utf-8")).hexdigest()
            != stream.get("schema_sha256")
            or stream.get("experiment_seed") != 42
            or stream.get("ddp_rank") != 0
            or stream.get("world_size") != 1
            or stream.get("cursor") != step
        ):
            raise ValueError("paired checkpoint prior_stream lineage is invalid")
        body = {key: value for key, value in stream.items() if key != "manifest_sha256"}
        expected_manifest = hashlib.sha256(
            repr({key: body[key] for key in sorted(body)}).encode("utf-8")
        ).hexdigest()
        if stream.get("manifest_sha256") != expected_manifest:
            raise ValueError("paired checkpoint prior_stream self-hash is invalid")
        return dict(stream)

    payloads: dict[str, Mapping[str, Any]] = {}
    configs: dict[str, dict[str, Any]] = {}
    steps: dict[str, int] = {}
    for arm in pair.arms:
        try:
            payload = torch.load(
                arm.checkpoint_path, map_location="cpu", weights_only=True
            )
        except Exception as error:
            raise ValueError(f"checkpoint is not weights-only loadable: {arm.arm_id}") from error
        if not isinstance(payload, Mapping):
            raise ValueError(f"checkpoint payload is not an object: {arm.arm_id}")
        config = payload.get("config")
        state = payload.get("state_dict")
        step = payload.get("curr_step")
        if (
            not isinstance(config, dict)
            or not isinstance(state, dict)
            or not state
            or isinstance(step, bool)
            or not isinstance(step, int)
            or step < 1
        ):
            raise ValueError(f"checkpoint lacks config/state/step: {arm.arm_id}")
        if not all(
            isinstance(name, str) and isinstance(tensor, torch.Tensor)
            for name, tensor in state.items()
        ):
            raise ValueError(f"checkpoint state_dict is not tensor-only: {arm.arm_id}")
        if any(
            (tensor.is_floating_point() or tensor.is_complex())
            and not bool(torch.isfinite(tensor).all())
            for tensor in state.values()
        ):
            raise ValueError(f"checkpoint state_dict is non-finite: {arm.arm_id}")
        expected_architecture = {
            "embed_dim": 128,
            "col_num_blocks": 3,
            "col_nhead": 8,
            "row_num_blocks": 3,
            "row_nhead": 8,
            "icl_num_blocks": 12,
            "icl_nhead": 8,
        }
        if any(config.get(key) != value for key, value in expected_architecture.items()):
            raise ValueError(f"checkpoint is not the frozen full-size architecture: {arm.arm_id}")
        if any("fingerprint" in name.casefold() for name in state) and arm.arm_id != "fingerprint":
            raise ValueError(f"checkpoint contains fingerprint treatment residue: {arm.arm_id}")
        row_mode = config.get("row_identity_mode")
        fingerprint = config.get("row_fingerprint", False)
        if arm.arm_id == "rope" and not (
            row_mode == "rope" and fingerprint is False
        ):
            raise ValueError("RoPE arm treatment does not match its checkpoint config")
        if arm.arm_id == "none" and not (
            row_mode == "none" and fingerprint is False
        ):
            raise ValueError("No-RoPE arm treatment does not match its checkpoint config")
        if arm.arm_id == "fingerprint" and not (
            row_mode == "none"
            and fingerprint is True
            and config.get("row_fingerprint_dim") == 16
        ):
            raise ValueError("Fingerprint arm treatment does not match its checkpoint config")
        payloads[arm.arm_id] = payload
        configs[arm.arm_id] = dict(config)
        steps[arm.arm_id] = step
    if len(set(steps.values())) != 1:
        raise ValueError("paired checkpoints must have the same training step")
    comparable = {}
    for arm, config in configs.items():
        normalized = dict(config)
        normalized.pop("row_identity_mode", None)
        normalized.pop("row_fingerprint", None)
        comparable[arm] = normalized
    first, second = pair.arm_order
    if comparable[first] != comparable[second]:
        raise ValueError("paired checkpoint configs differ beyond the PE treatment")

    step = steps[first]
    exact_pairs = {
        (250_000, ("rope", "none")),
        (50_000, ("rope", "fingerprint")),
        (500_000, ("rope", "none")),
    }
    if (step, pair.arm_order) not in exact_pairs:
        raise ValueError(
            "pair must be exactly step-250000 rope/none, step-50000 rope/fingerprint, "
            "or step-500000 rope/none"
        )
    if step in {250_000, 500_000} and pair.legacy_snapshot_receipts is None:
        raise ValueError(f"step-{step} rope/none requires two snapshot receipts")
    if step != 50_000 and pair.training_receipts is not None:
        raise ValueError("training receipts are only valid for the step-50000 pair")
    if step == 50_000 and pair.legacy_snapshot_receipts is not None:
        raise ValueError("step-50000 pair must not claim legacy snapshot receipts")

    streams = {arm: payloads[arm].get("prior_stream") for arm in pair.arm_order}
    if all(stream is None for stream in streams.values()):
        if step in {250_000, 500_000} and pair.legacy_snapshot_receipts is not None:
            receipt_evidence = {
                arm.arm_id: validate_legacy_receipt(
                    arm,
                    pair.legacy_snapshot_receipts[arm.arm_id],
                    expected_step=step,
                )
                for arm in pair.arms
            }
            left_receipt = receipt_evidence[first]
            right_receipt = receipt_evidence[second]
            for field in ("continuation_id", "continuation_source_commit"):
                if left_receipt[field] != right_receipt[field]:
                    raise ValueError(
                        f"legacy snapshot receipts disagree on {field}"
                    )
            prior_contract = {
                "mode": "same_step_legacy",
                "formal_eligible": False,
                "source_provenance_status": (
                    "operational-history-only-not-exact-prior-stream"
                ),
                "receipts": receipt_evidence,
            }
        else:
            raise ValueError(
                "missing prior_stream is only accepted for receipt-bound step-250000/500000 pairs"
            )
    else:
        if not all(isinstance(stream, Mapping) for stream in streams.values()):
            raise ValueError("prior_stream must be present for both paired checkpoints")
        if dict(streams[first]) != dict(streams[second]):
            raise ValueError("paired checkpoints have different prior_stream state")
        stream = validate_prior_stream(streams[first], step=step)
        prior_contract = {"mode": "exact", "state": stream}
        if step == 50_000:
            prior_contract["training_receipts"] = validate_training_receipts(
                streams={arm: streams[arm] for arm in pair.arm_order}  # type: ignore[misc]
            )
        elif pair.legacy_snapshot_receipts is not None:
            prior_contract["snapshot_receipts"] = {
                arm.arm_id: validate_legacy_receipt(
                    arm,
                    pair.legacy_snapshot_receipts[arm.arm_id],
                    expected_step=step,
                )
                for arm in pair.arms
            }
    config_bytes = _canonical_json(comparable[first])
    return {
        "comparison_step": step,
        "prior_stream": prior_contract,
        "shared_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "treatments": {
            arm: {
                "row_identity_mode": configs[arm].get("row_identity_mode"),
                "row_fingerprint": configs[arm].get("row_fingerprint", False),
                "row_fingerprint_dim": configs[arm].get("row_fingerprint_dim"),
            }
            for arm in pair.arm_order
        },
    }


def load_roster(path: Path, *, enforce_pinned: bool = True) -> Roster:
    """Load either the pinned TabArena v0.1 or BeyondArena lite roster."""
    payload, roster_file_sha = _load_json_file(path, label="dataset roster")
    common = {
        "schema_version",
        "kind",
        "suite",
        "version",
        "task_type",
        "count",
        "names",
        "roster_hash_algorithm",
        "roster_sha256",
        "source_commit",
        "source_url",
    }
    suite = payload.get("suite")
    if suite == "BeyondArena":
        fields = common | {"subset", "problem_types", "text_features"}
    elif suite == "TabArena":
        fields = common
    else:
        raise ValueError("roster suite must be BeyondArena or TabArena")
    payload = _exact_object(payload, label="dataset roster", fields=fields)
    if payload["schema_version"] != 1 or payload["kind"] != "dataset_roster":
        raise ValueError("unsupported dataset roster schema or kind")
    if payload["task_type"] != "classification":
        raise ValueError("pair evaluation roster must be classification-only")
    if payload["roster_hash_algorithm"] != "sha256_newline_join_sorted_names":
        raise ValueError("unsupported roster hash algorithm")
    names = payload["names"]
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
        or names != sorted(names)
        or len(names) != len(set(names))
        or payload["count"] != len(names)
    ):
        raise ValueError("dataset roster names are empty, duplicated, or non-canonical")
    names_sha = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    if payload["roster_sha256"] != names_sha:
        raise ValueError("dataset roster name digest mismatch")
    source_commit = _git_sha(payload["source_commit"], label="roster source_commit")
    source_url = payload["source_url"]
    if not isinstance(source_url, str) or not source_url.startswith("https://github.com/"):
        raise ValueError("dataset roster source_url must be a GitHub HTTPS URL")
    if suite == "BeyondArena":
        if (
            payload["version"] != "pinned"
            or payload["subset"] != "lite"
            or payload["problem_types"] != ["binary", "multiclass"]
            or payload["text_features"] != "excluded"
        ):
            raise ValueError("BeyondArena roster is not the non-text lite contract")
        suite_id = "beyondarena"
    else:
        if payload["version"] != "v0.1":
            raise ValueError("TabArena roster must be v0.1")
        suite_id = "tabarena-v0.1"
    roster = Roster(
        path=path.resolve(strict=True),
        sha256=roster_file_sha,
        names_sha256=names_sha,
        suite_id=suite_id,
        source_commit=source_commit,
        names=tuple(names),
    )
    if enforce_pinned:
        pinned = _PINNED_ROSTERS[roster.suite_id]
        if (
            len(roster.names) != pinned["count"]
            or roster.names_sha256 != pinned["names_sha256"]
            or roster.source_commit != pinned["source_commit"]
        ):
            raise ValueError(
                f"{roster.suite_id} roster differs from its exact pinned digest/count/source"
            )
    return roster


def load_shard_plan(path: Path, *, roster: Roster) -> ShardPlan:
    payload, plan_sha = _load_json_file(path, label="shard plan")
    payload = _exact_object(
        payload,
        label="shard plan",
        fields={
            "schema_version",
            "kind",
            "suite",
            "version",
            "source_commit",
            "roster_sha256",
            "shard_count",
            "cost_model",
            "canary",
            "shards",
            "assignment_hash_algorithm",
            "assignment_sha256",
        },
    )
    expected_suite = "BeyondArena" if roster.suite_id == "beyondarena" else "TabArena"
    if (
        payload["schema_version"] != 1
        or payload["kind"] != "cost_balanced_pair_shard_plan"
        or payload["suite"] != expected_suite
        or payload["version"] != "v1"
        or payload["source_commit"] != roster.source_commit
        or payload["roster_sha256"] != roster.names_sha256
        or payload["shard_count"] != 8
    ):
        raise ValueError("shard plan does not match the pinned eight-shard roster")
    cost_model = _exact_object(
        payload["cost_model"],
        label="cost model",
        fields={"name", "formula", "assignment"},
    )
    if cost_model != {
        "name": "tabicl_pair_lpt_proxy_v1",
        "formula": (
            "num_instances*num_cols_after_preprocessing + "
            "num_instances_test*min(num_instances_train,1024)"
        ),
        "assignment": (
            "longest_processing_time_first_ties_by_dataset_then_lowest_shard"
        ),
    }:
        raise ValueError("shard plan cost model differs from the frozen LPT proxy")
    raw_shards = payload["shards"]
    if not isinstance(raw_shards, list) or len(raw_shards) != 8:
        raise ValueError("shard plan must contain exactly eight shards")
    assignments: list[tuple[str, ...]] = []
    totals: list[int] = []
    roster_position = {name: index for index, name in enumerate(roster.names)}
    for index, raw_shard in enumerate(raw_shards):
        shard = _exact_object(
            raw_shard,
            label=f"shards[{index}]",
            fields={"index", "estimated_cost_units", "names"},
        )
        names = shard["names"]
        if (
            shard["index"] != index
            or not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) and name in roster_position for name in names)
            or names != sorted(names, key=roster_position.__getitem__)
        ):
            raise ValueError(f"shard {index} is empty, unknown, or non-canonical")
        assignments.append(tuple(names))
        totals.append(
            _positive_int(
                shard["estimated_cost_units"],
                label=f"shards[{index}].estimated_cost_units",
            )
        )
    flattened = [name for shard in assignments for name in shard]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(roster.names):
        raise ValueError("eight shard assignments are not the exact roster union")
    if payload["assignment_hash_algorithm"] != (
        "sha256_newline_join_roster_order_name_tab_shard_tab_cost"
    ):
        raise ValueError("unsupported shard assignment hash algorithm")
    assignment_sha = _digest(
        payload["assignment_sha256"], label="assignment_sha256"
    )
    canary = _exact_object(
        payload["canary"], label="canary", fields={"names", "requirements"}
    )
    canary_names = canary["names"]
    if (
        not isinstance(canary_names, list)
        or not canary_names
        or len(canary_names) != len(set(canary_names))
        or not all(name in roster_position for name in canary_names)
    ):
        raise ValueError("canary names must be a unique roster subset")
    requirement_fields = {"max_rows", "max_dimensions"}
    if roster.suite_id == "beyondarena":
        requirement_fields |= {"grouped", "temporal", "max_lpt_workload"}
    requirements = _exact_object(
        canary["requirements"],
        label="canary requirements",
        fields=requirement_fields,
    )
    if any(value not in canary_names for value in requirements.values()):
        raise ValueError("every canary requirement must name a canary task")
    return ShardPlan(
        path=path.resolve(strict=True),
        sha256=plan_sha,
        source_commit=roster.source_commit,
        roster_names_sha256=roster.names_sha256,
        assignments=tuple(assignments),
        estimated_cost_units=tuple(totals),
        assignment_sha256=assignment_sha,
        canary_names=tuple(canary_names),
        canary_requirements=dict(requirements),
    )


def _metadata_int(record: Mapping[str, Any], field: str, *, dataset: str) -> int:
    value = record.get(field)
    if isinstance(value, bool):
        raise ValueError(f"metadata {field} is invalid for {dataset}")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"metadata {field} is invalid for {dataset}") from error
    if number < 1:
        raise ValueError(f"metadata {field} must be positive for {dataset}")
    return number


def validate_plan_metadata(
    plan: ShardPlan,
    *,
    roster: Roster,
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute the frozen proxy and prove the explicit plan is canonical LPT."""
    if set(records) != set(roster.names):
        raise ValueError("benchmark metadata does not exactly cover the frozen roster")
    costs: dict[str, int] = {}
    normalized: dict[str, dict[str, Any]] = {}
    for name in roster.names:
        record = records[name]
        num_instances = _metadata_int(record, "num_instances", dataset=name)
        dimensions = _metadata_int(
            record, "num_cols_after_preprocessing", dataset=name
        )
        train = _metadata_int(record, "num_instances_train", dataset=name)
        test = _metadata_int(record, "num_instances_test", dataset=name)
        regime = record.get("split_regime")
        if regime not in _SPLIT_REGIMES:
            raise ValueError(f"metadata split regime is invalid for {name}")
        costs[name] = num_instances * dimensions + test * min(train, 1024)
        normalized[name] = {
            "num_instances": num_instances,
            "num_cols_after_preprocessing": dimensions,
            "num_instances_train": train,
            "num_instances_test": test,
            "split_regime": regime,
        }
    assignments: list[list[str]] = [[] for _ in range(8)]
    totals = [0] * 8
    for name in sorted(roster.names, key=lambda item: (-costs[item], item)):
        index = min(range(8), key=lambda item: (totals[item], item))
        assignments[index].append(name)
        totals[index] += costs[name]
    roster_position = {name: index for index, name in enumerate(roster.names)}
    canonical = tuple(
        tuple(sorted(names, key=roster_position.__getitem__)) for names in assignments
    )
    if canonical != plan.assignments or tuple(totals) != plan.estimated_cost_units:
        raise ValueError("explicit shard plan is not the canonical cost-balanced LPT plan")
    name_to_shard = {
        name: index for index, names in enumerate(canonical) for name in names
    }
    assignment_lines = "\n".join(
        f"{name}\t{name_to_shard[name]}\t{costs[name]}" for name in roster.names
    )
    observed_assignment_sha = hashlib.sha256(
        assignment_lines.encode("utf-8")
    ).hexdigest()
    if observed_assignment_sha != plan.assignment_sha256:
        raise ValueError("shard assignment digest disagrees with live benchmark metadata")
    requirements = plan.canary_requirements
    if normalized[requirements["max_rows"]]["num_instances"] != max(
        record["num_instances"] for record in normalized.values()
    ):
        raise ValueError("canary does not include a maximum-row task")
    if normalized[requirements["max_dimensions"]][
        "num_cols_after_preprocessing"
    ] != max(record["num_cols_after_preprocessing"] for record in normalized.values()):
        raise ValueError("canary does not include a maximum-dimension task")
    if "grouped" in requirements and normalized[requirements["grouped"]][
        "split_regime"
    ] != "grouped":
        raise ValueError("canary grouped requirement is not grouped")
    if "temporal" in requirements and normalized[requirements["temporal"]][
        "split_regime"
    ] != "temporal":
        raise ValueError("canary temporal requirement is not temporal")
    if "max_lpt_workload" in requirements and costs[
        requirements["max_lpt_workload"]
    ] != max(costs.values()):
        raise ValueError("canary does not include the maximum-LPT workload")
    return {
        "assignment_sha256": observed_assignment_sha,
        "estimated_cost_units": totals,
        "canary_requirements": dict(requirements),
    }


def shard_tasks(
    roster: Roster, *, plan: ShardPlan, shard_index: int
) -> tuple[Task, ...]:
    shard_index = _positive_int(shard_index, label="shard_index", allow_zero=True)
    if shard_index >= plan.shard_count:
        raise ValueError("shard_index must be smaller than the frozen shard count")
    positions = {name: index for index, name in enumerate(roster.names)}
    return tuple(
        Task(index=positions[name], name=name) for name in plan.assignments[shard_index]
    )


def canary_tasks(roster: Roster, *, plan: ShardPlan) -> tuple[Task, ...]:
    positions = {name: index for index, name in enumerate(roster.names)}
    return tuple(Task(index=positions[name], name=name) for name in plan.canary_names)


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
    if not slug:
        raise ValueError("dataset name cannot form a safe artifact name")
    return slug[:120]


def task_dirname(task: Task) -> str:
    return f"{task.index:04d}-{_slug(task.name)}"


def _finite(value: object, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def code_provenance(
    *, analysis_sha: str, model_sha: str, tabarena_sha: str
) -> dict[str, str]:
    return {
        "analysis_sha": _git_sha(analysis_sha, label="analysis_sha"),
        "model_sha": _git_sha(model_sha, label="model_sha"),
        "tabarena_sha": _git_sha(tabarena_sha, label="tabarena_sha"),
    }


def _inference_sha(pair: PairManifest) -> str:
    return hashlib.sha256(_canonical_json(dict(pair.inference))).hexdigest()


def _run_contract(
    *,
    pair: PairManifest,
    roster: Roster,
    plan: ShardPlan,
    provenance: Mapping[str, str],
    runtime_environment: Mapping[str, Any] | None = None,
    python_environment_contract: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if provenance.get("model_sha") != pair.model_source_sha:
        raise ValueError("model checkout does not match the pair manifest")
    if provenance.get("tabarena_sha") != roster.source_commit:
        raise ValueError("benchmark checkout does not match the dataset roster")
    environment = (
        dict(runtime_environment)
        if runtime_environment is not None
        else {"evidence_status": "not_supplied_by_external_executor"}
    )
    python_binding = (
        dict(python_environment_contract)
        if python_environment_contract is not None
        else {"evidence_status": "not_supplied_by_external_executor"}
    )
    if python_binding != {"evidence_status": "not_supplied_by_external_executor"} and (
        set(python_binding)
        != {"document_sha256", "file_sha256"}
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in python_binding.values()
        )
    ):
        raise ValueError("Python environment contract binding is malformed")
    return {
        "schema_version": 1,
        "kind": "two_arm_full_suite_run",
        "pair_id": pair.pair_id,
        "pair_manifest_sha256": pair.sha256,
        "roster_sha256": roster.sha256,
        "suite": roster.suite_id,
        "subset": "lite",
        "task_count": len(roster.names),
        "shard_count": plan.shard_count,
        "shard_plan_sha256": plan.sha256,
        "assignment_sha256": plan.assignment_sha256,
        "canary_names": list(plan.canary_names),
        "arm_order": list(pair.arm_order),
        "checkpoint_contract": dict(pair.checkpoint_contract),
        "inference_contract_sha256": _inference_sha(pair),
        "oom_fallback_policy": [dict(value) for value in _OOM_FALLBACK_POLICY],
        "code_provenance": dict(provenance),
        "runtime_environment": environment,
        "runtime_environment_sha256": hashlib.sha256(
            _canonical_json(environment)
        ).hexdigest(),
        "python_environment_contract": python_binding,
    }


def _atomic_publish_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one immutable JSON file; an identical concurrent writer is benign."""
    encoded = _canonical_json(payload)
    _real_directory(path.parent, label="artifact parent", create=True)
    if path.is_symlink():
        raise ValueError(f"refusing symlink artifact: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise ValueError(f"existing artifact disagrees with run contract: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                raise ValueError(
                    f"concurrent artifact disagrees with run contract: {path}"
                )
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    payload, _ = _load_json_file(path, label=label)
    return payload


def initialize_run(
    run_root: Path,
    *,
    pair: PairManifest,
    roster: Roster,
    plan: ShardPlan,
    provenance: Mapping[str, str],
    runtime_environment: Mapping[str, Any] | None = None,
    python_environment_contract: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    _real_directory(run_root, label="run_root", create=True)
    _real_directory(run_root / "tasks", label="tasks control directory", create=True)
    _real_directory(run_root / "shards", label="shards control directory", create=True)
    _real_directory(
        run_root / ".staging", label="staging control directory", create=True
    )
    contract = _run_contract(
        pair=pair,
        roster=roster,
        plan=plan,
        provenance=provenance,
        runtime_environment=runtime_environment,
        python_environment_contract=python_environment_contract,
    )
    _atomic_publish_json(run_root / "run.json", contract)
    return contract


def build_task_payload(
    *,
    task: Task,
    task_metadata: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
    fallback: Mapping[str, Any],
    run_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "two_arm_lite_task_result",
        "pair_id": run_contract["pair_id"],
        "pair_manifest_sha256": run_contract["pair_manifest_sha256"],
        "roster_sha256": run_contract["roster_sha256"],
        "shard_plan_sha256": run_contract["shard_plan_sha256"],
        "assignment_sha256": run_contract["assignment_sha256"],
        "suite": run_contract["suite"],
        "subset": "lite",
        "task_index": task.index,
        "task": dict(task_metadata),
        "arm_order": list(run_contract["arm_order"]),
        "results": {arm: dict(value) for arm, value in results.items()},
        "fallback": dict(fallback),
        "checkpoint_contract": dict(run_contract["checkpoint_contract"]),
        "inference_contract_sha256": run_contract["inference_contract_sha256"],
        "runtime_environment_sha256": run_contract["runtime_environment_sha256"],
        "python_environment_contract": dict(
            run_contract["python_environment_contract"]
        ),
        "code_provenance": dict(run_contract["code_provenance"]),
    }


def validate_task_payload(
    payload: Mapping[str, Any],
    *,
    task: Task,
    run_contract: Mapping[str, Any],
) -> dict[str, Any]:
    value = _exact_object(
        payload,
        label="task result",
        fields={
            "schema_version",
            "kind",
            "pair_id",
            "pair_manifest_sha256",
            "roster_sha256",
            "shard_plan_sha256",
            "assignment_sha256",
            "suite",
            "subset",
            "task_index",
            "task",
            "arm_order",
            "results",
            "fallback",
            "checkpoint_contract",
            "inference_contract_sha256",
            "runtime_environment_sha256",
            "python_environment_contract",
            "code_provenance",
        },
    )
    for field in (
        "pair_id",
        "pair_manifest_sha256",
        "roster_sha256",
        "shard_plan_sha256",
        "assignment_sha256",
        "suite",
        "arm_order",
        "checkpoint_contract",
        "inference_contract_sha256",
        "runtime_environment_sha256",
        "python_environment_contract",
        "code_provenance",
    ):
        if value[field] != run_contract[field]:
            raise ValueError(f"task result {field} differs from run contract")
    if (
        value["schema_version"] != 1
        or value["kind"] != "two_arm_lite_task_result"
        or value["subset"] != "lite"
        or value["task_index"] != task.index
    ):
        raise ValueError("task result header differs from the lite task contract")
    if value["fallback"] not in run_contract["oom_fallback_policy"]:
        raise ValueError("task result fallback level is outside the frozen OOM policy")
    metadata = _exact_object(
        value["task"],
        label="task metadata",
        fields={
            "dataset_name",
            "benchmark_dataset_id",
            "task_id",
            "problem_type",
            "metric",
            "split_regime",
            "fold",
            "repeat",
            "split_index",
        },
    )
    if metadata["dataset_name"] != task.name:
        raise ValueError("task result dataset name differs from the roster")
    if not isinstance(metadata["benchmark_dataset_id"], str) or not metadata[
        "benchmark_dataset_id"
    ]:
        raise ValueError("benchmark dataset id must be non-empty")
    if not isinstance(metadata["task_id"], (str, int)) or isinstance(
        metadata["task_id"], bool
    ):
        raise ValueError("task id must be a string or integer")
    if metadata["problem_type"] not in _PROBLEM_TYPES:
        raise ValueError("task result is not classification")
    if not isinstance(metadata["metric"], str) or not metadata["metric"]:
        raise ValueError("task metric must be non-empty")
    if metadata["split_regime"] not in _SPLIT_REGIMES:
        raise ValueError("task split regime is invalid")
    if any(metadata[field] != 0 for field in ("fold", "repeat", "split_index")):
        raise ValueError("task result is not the lite split")

    arm_order = tuple(run_contract["arm_order"])
    raw_results = _exact_object(
        value["results"], label="arm results", fields=set(arm_order)
    )
    normalized_results: dict[str, dict[str, float]] = {}
    for arm in arm_order:
        result = _exact_object(
            raw_results[arm],
            label=f"result[{arm}]",
            fields={"metric_error", "time_train_s", "time_infer_s"},
        )
        metric_error = _finite(
            result["metric_error"], label=f"result[{arm}].metric_error", nonnegative=True
        )
        if metadata["metric"] == "roc_auc" and not 0.0 <= metric_error <= 1.0:
            raise ValueError("ROC-AUC error must be in [0, 1]")
        normalized_results[arm] = {
            "metric_error": metric_error,
            "time_train_s": _finite(
                result["time_train_s"],
                label=f"result[{arm}].time_train_s",
                nonnegative=True,
            ),
            "time_infer_s": _finite(
                result["time_infer_s"],
                label=f"result[{arm}].time_infer_s",
                nonnegative=True,
            ),
        }
    normalized = dict(value)
    normalized["task"] = dict(metadata)
    normalized["results"] = normalized_results
    return normalized


def _write_staging_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"staging artifact already exists: {path}")
    path.write_bytes(_canonical_json(payload))
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _prediction_metadata(path: Path, *, metric: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"prediction capture is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "probabilities",
                "encoded_targets",
                "row_fingerprints",
                "test_target_fingerprints",
                "class_labels_utf8",
                "train_content_sha256",
                "test_content_sha256",
            }:
                raise ValueError("prediction archive fields differ from the pair contract")
            probabilities = np.asarray(archive["probabilities"])
            encoded_targets = np.asarray(archive["encoded_targets"])
            rows = np.asarray(archive["row_fingerprints"])
            targets = np.asarray(archive["test_target_fingerprints"])
            class_bytes_array = np.asarray(archive["class_labels_utf8"])
            train_content = np.asarray(archive["train_content_sha256"])
            test_content = np.asarray(archive["test_content_sha256"])
    except (OSError, ValueError) as error:
        raise ValueError("prediction archive is unreadable or unsafe") from error
    if probabilities.dtype != np.dtype("<f4") or probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional little-endian float32 array")
    if probabilities.shape[0] < 1 or probabilities.shape[1] < 2:
        raise ValueError("prediction probability matrix is empty")
    if not np.isfinite(probabilities).all():
        raise ValueError("prediction probabilities must be finite")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("prediction probabilities must lie in [0, 1]")
    if not np.allclose(
        probabilities.sum(axis=1, dtype=np.float64), 1.0, rtol=1e-6, atol=1e-6
    ):
        raise ValueError("prediction probability rows must sum to one")
    expected_fingerprint_shape = (probabilities.shape[0], 32)
    if (
        rows.dtype != np.uint8
        or targets.dtype != np.uint8
        or rows.shape != expected_fingerprint_shape
        or targets.shape != expected_fingerprint_shape
    ):
        raise ValueError("row or target fingerprints are malformed")
    if class_bytes_array.dtype != np.uint8 or class_bytes_array.ndim != 1:
        raise ValueError("class label bytes are malformed")
    if (
        encoded_targets.dtype != np.dtype("<i8")
        or encoded_targets.shape != (probabilities.shape[0],)
        or np.any(encoded_targets < 0)
        or np.any(encoded_targets >= probabilities.shape[1])
    ):
        raise ValueError("encoded test targets are malformed")
    if (
        train_content.dtype != np.uint8
        or test_content.dtype != np.uint8
        or train_content.shape != (32,)
        or test_content.shape != (32,)
    ):
        raise ValueError("train/test content fingerprints are malformed")
    class_bytes = class_bytes_array.tobytes()
    try:
        class_labels = json.loads(class_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("class label evidence is not canonical JSON") from error
    if (
        not isinstance(class_labels, list)
        or len(class_labels) != probabilities.shape[1]
        or not all(isinstance(label, Mapping) for label in class_labels)
    ):
        raise ValueError("class label evidence does not match prediction columns")
    if metric == "roc_auc":
        from sklearn.metrics import roc_auc_score

        if probabilities.shape[1] == 2:
            score = roc_auc_score(encoded_targets, probabilities[:, 1])
        else:
            score = roc_auc_score(
                encoded_targets,
                probabilities,
                labels=np.arange(probabilities.shape[1]),
                multi_class="ovr",
            )
        independent_error = 1.0 - float(score)
    elif metric == "log_loss":
        from sklearn.metrics import log_loss

        independent_error = float(
            log_loss(
                encoded_targets,
                probabilities,
                labels=np.arange(probabilities.shape[1]),
            )
        )
    elif metric == "accuracy":
        independent_error = 1.0 - float(
            np.mean(np.argmax(probabilities, axis=1) == encoded_targets)
        )
    else:
        raise ValueError(f"unsupported independently recomputed metric: {metric}")
    if not math.isfinite(independent_error) or independent_error < 0.0:
        raise ValueError("independently recomputed metric error is invalid")
    return {
        "schema_version": 1,
        "kind": "two_arm_raw_prediction",
        "file_name": "predictions.npz",
        "file_sha256": sha256_file(path),
        "file_size_bytes": path.stat().st_size,
        "dtype": "float32",
        "shape": [int(value) for value in probabilities.shape],
        "row_count": int(probabilities.shape[0]),
        "class_count": int(probabilities.shape[1]),
        "class_labels": class_labels,
        "class_labels_sha256": hashlib.sha256(class_bytes).hexdigest(),
        "row_fingerprint_sha256": _array_sha256(rows),
        "test_target_sha256": _array_sha256(targets),
        "encoded_target_sha256": _array_sha256(encoded_targets),
        "train_content_sha256": train_content.tobytes().hex(),
        "test_content_sha256": test_content.tobytes().hex(),
        "probability_sha256": _array_sha256(probabilities),
        "independent_metric": metric,
        "independent_metric_error": independent_error,
    }


def _arm_result_payload(
    *,
    task: Task,
    task_metadata: Mapping[str, Any],
    arm: str,
    result: Mapping[str, Any],
    prediction: Mapping[str, Any],
    fallback: Mapping[str, Any],
    run_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "two_arm_lite_arm_result",
        "pair_id": run_contract["pair_id"],
        "pair_manifest_sha256": run_contract["pair_manifest_sha256"],
        "roster_sha256": run_contract["roster_sha256"],
        "shard_plan_sha256": run_contract["shard_plan_sha256"],
        "assignment_sha256": run_contract["assignment_sha256"],
        "suite": run_contract["suite"],
        "subset": "lite",
        "task_index": task.index,
        "task": dict(task_metadata),
        "arm": arm,
        "result": dict(result),
        "prediction": dict(prediction),
        "fallback": dict(fallback),
        "checkpoint_contract": dict(run_contract["checkpoint_contract"]),
        "inference_contract_sha256": run_contract["inference_contract_sha256"],
        "runtime_environment_sha256": run_contract["runtime_environment_sha256"],
        "python_environment_contract": dict(
            run_contract["python_environment_contract"]
        ),
        "code_provenance": dict(run_contract["code_provenance"]),
    }


def _validate_task_directory(
    directory: Path,
    *,
    task: Task,
    run_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"task artifact is not an atomic real directory: {directory}")
    arm_order = tuple(run_contract["arm_order"])
    expected_top = {"task.json", "manifest.json", *arm_order}
    if {path.name for path in directory.iterdir()} != expected_top:
        raise ValueError("task directory contents differ from the pair contract")
    if any(path.is_symlink() for path in directory.rglob("*")):
        raise ValueError("task directory must not contain symlinks")
    combined = validate_task_payload(
        _read_json(directory / "task.json", label="combined task result"),
        task=task,
        run_contract=run_contract,
    )
    prediction_evidence: dict[str, dict[str, Any]] = {}
    arm_manifest_digests: dict[str, dict[str, Any]] = {}
    for arm in arm_order:
        arm_root = directory / arm
        if not arm_root.is_dir() or {path.name for path in arm_root.iterdir()} != {
            "predictions.npz",
            "result.json",
            "manifest.json",
        }:
            raise ValueError(f"arm artifact set is incomplete for {arm}")
        prediction = _prediction_metadata(
            arm_root / "predictions.npz", metric=combined["task"]["metric"]
        )
        prediction_evidence[arm] = prediction
        result_payload = _exact_object(
            _read_json(arm_root / "result.json", label=f"result for {arm}"),
            label=f"result for {arm}",
            fields={
                "schema_version",
                "kind",
                "pair_id",
                "pair_manifest_sha256",
                "roster_sha256",
                "shard_plan_sha256",
                "assignment_sha256",
                "suite",
                "subset",
                "task_index",
                "task",
                "arm",
                "result",
                "prediction",
                "fallback",
                "checkpoint_contract",
                "inference_contract_sha256",
                "runtime_environment_sha256",
                "python_environment_contract",
                "code_provenance",
            },
        )
        expected_result = _arm_result_payload(
            task=task,
            task_metadata=combined["task"],
            arm=arm,
            result=combined["results"][arm],
            prediction=prediction,
            fallback=combined["fallback"],
            run_contract=run_contract,
        )
        if result_payload != expected_result:
            raise ValueError(f"arm result disagrees with task/prediction bytes for {arm}")
        if not math.isclose(
            prediction["independent_metric_error"],
            combined["results"][arm]["metric_error"],
            rel_tol=1e-5,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"reported metric for {arm} cannot be recomputed from raw NPZ"
            )
        arm_manifest = {
            "schema_version": 1,
            "kind": "two_arm_lite_arm_manifest",
            "arm": arm,
            "artifacts": {
                "predictions.npz": {
                    "sha256": prediction["file_sha256"],
                    "size_bytes": prediction["file_size_bytes"],
                },
                "result.json": {
                    "sha256": sha256_file(arm_root / "result.json"),
                    "size_bytes": (arm_root / "result.json").stat().st_size,
                },
            },
        }
        if _read_json(arm_root / "manifest.json", label=f"manifest for {arm}") != arm_manifest:
            raise ValueError(f"arm manifest does not attest current bytes for {arm}")
        arm_manifest_digests[arm] = {
            "sha256": sha256_file(arm_root / "manifest.json"),
            "size_bytes": (arm_root / "manifest.json").stat().st_size,
        }
    left, right = arm_order
    for field in (
        "shape",
        "class_labels_sha256",
        "row_fingerprint_sha256",
        "test_target_sha256",
        "encoded_target_sha256",
        "train_content_sha256",
        "test_content_sha256",
    ):
        if prediction_evidence[left][field] != prediction_evidence[right][field]:
            raise ValueError(f"paired raw predictions disagree on {field}")
    task_manifest = {
        "schema_version": 1,
        "kind": "two_arm_lite_task_manifest",
        "pair_manifest_sha256": run_contract["pair_manifest_sha256"],
        "roster_sha256": run_contract["roster_sha256"],
        "task_index": task.index,
        "artifacts": {
            "task.json": {
                "sha256": sha256_file(directory / "task.json"),
                "size_bytes": (directory / "task.json").stat().st_size,
            },
            "arms": arm_manifest_digests,
        },
    }
    if _read_json(directory / "manifest.json", label="task manifest") != task_manifest:
        raise ValueError("task manifest does not attest the complete paired artifact")
    return combined


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        elif path.is_dir():
            directories.append(path)
    for directory in reversed(directories):
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


TaskExecutor = Callable[
    [Task, Path, Mapping[str, Any]],
    tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]],
]


def _execute_and_publish_task(
    *,
    run_root: Path,
    task: Task,
    run_contract: Mapping[str, Any],
    executor: TaskExecutor,
) -> Path:
    tasks_root = _real_directory(
        run_root / "tasks", label="tasks control directory"
    )
    final = tasks_root / task_dirname(task)
    staging_root = _real_directory(
        run_root / ".staging", label="staging control directory"
    )
    policy = run_contract.get("oom_fallback_policy")
    if policy != [dict(item) for item in _OOM_FALLBACK_POLICY]:
        raise ValueError("run contract OOM fallback policy is not frozen")
    for fallback_index, raw_fallback in enumerate(policy):
        fallback = dict(raw_fallback)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{task_dirname(task)}.tmp-", dir=staging_root)
        )
        _real_directory(staging, label="task staging directory")
        try:
            for arm in run_contract["arm_order"]:
                (staging / arm).mkdir()
            try:
                metadata, results = executor(task, staging, fallback)
            except PairTaskOOM:
                if fallback_index + 1 == len(policy):
                    raise
                continue
            combined = build_task_payload(
                task=task,
                task_metadata=metadata,
                results=results,
                fallback=fallback,
                run_contract=run_contract,
            )
            combined = validate_task_payload(
                combined, task=task, run_contract=run_contract
            )
            _write_staging_json(staging / "task.json", combined)
            arm_manifest_digests: dict[str, dict[str, Any]] = {}
            for arm in run_contract["arm_order"]:
                arm_root = staging / arm
                prediction = _prediction_metadata(
                    arm_root / "predictions.npz",
                    metric=combined["task"]["metric"],
                )
                result_payload = _arm_result_payload(
                    task=task,
                    task_metadata=combined["task"],
                    arm=arm,
                    result=combined["results"][arm],
                    prediction=prediction,
                    fallback=combined["fallback"],
                    run_contract=run_contract,
                )
                _write_staging_json(arm_root / "result.json", result_payload)
                arm_manifest = {
                    "schema_version": 1,
                    "kind": "two_arm_lite_arm_manifest",
                    "arm": arm,
                    "artifacts": {
                        "predictions.npz": {
                            "sha256": prediction["file_sha256"],
                            "size_bytes": prediction["file_size_bytes"],
                        },
                        "result.json": {
                            "sha256": sha256_file(arm_root / "result.json"),
                            "size_bytes": (arm_root / "result.json").stat().st_size,
                        },
                    },
                }
                _write_staging_json(arm_root / "manifest.json", arm_manifest)
                arm_manifest_digests[arm] = {
                    "sha256": sha256_file(arm_root / "manifest.json"),
                    "size_bytes": (arm_root / "manifest.json").stat().st_size,
                }
            task_manifest = {
                "schema_version": 1,
                "kind": "two_arm_lite_task_manifest",
                "pair_manifest_sha256": run_contract["pair_manifest_sha256"],
                "roster_sha256": run_contract["roster_sha256"],
                "task_index": task.index,
                "artifacts": {
                    "task.json": {
                        "sha256": sha256_file(staging / "task.json"),
                        "size_bytes": (staging / "task.json").stat().st_size,
                    },
                    "arms": arm_manifest_digests,
                },
            }
            _write_staging_json(staging / "manifest.json", task_manifest)
            _validate_task_directory(staging, task=task, run_contract=run_contract)
            _fsync_tree(staging)
            try:
                os.rename(staging, final)
            except OSError:
                if not final.exists():
                    raise
                _validate_task_directory(final, task=task, run_contract=run_contract)
            descriptor = os.open(
                tasks_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return final
        finally:
            if staging.exists():
                if staging.is_symlink() or not staging.is_dir():
                    raise RuntimeError(
                        "task staging directory was replaced during cleanup"
                    )
                if staging.parent != staging_root:
                    raise RuntimeError(
                        "task staging cleanup escaped its control directory"
                    )
                shutil.rmtree(staging)
    raise AssertionError("frozen OOM fallback policy was unexpectedly empty")


def run_shard(
    run_root: Path,
    *,
    pair: PairManifest,
    roster: Roster,
    plan: ShardPlan,
    provenance: Mapping[str, str],
    shard_index: int,
    executor: TaskExecutor,
    runtime_environment: Mapping[str, Any] | None = None,
    python_environment_contract: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    contract = initialize_run(
        run_root,
        pair=pair,
        roster=roster,
        plan=plan,
        provenance=provenance,
        runtime_environment=runtime_environment,
        python_environment_contract=python_environment_contract,
    )
    canary_marker = run_root / "canary.json"
    expected_canary = _expected_canary_marker(
        run_root=run_root,
        pair=pair,
        roster=roster,
        plan=plan,
    )
    if _read_json(canary_marker, label="canary marker") != expected_canary:
        raise ValueError("full shards require a complete, current canary marker")
    plans = shard_tasks(roster, plan=plan, shard_index=shard_index)
    executed = 0
    resumed = 0
    artifacts: list[dict[str, Any]] = []
    for task in plans:
        path = run_root / "tasks" / task_dirname(task)
        if path.exists():
            _validate_task_directory(path, task=task, run_contract=contract)
            resumed += 1
        else:
            path = _execute_and_publish_task(
                run_root=run_root,
                task=task,
                run_contract=contract,
                executor=executor,
            )
            executed += 1
        artifacts.append(
            {
                "directory": path.name,
                "manifest_sha256": sha256_file(path / "manifest.json"),
                "manifest_size_bytes": (path / "manifest.json").stat().st_size,
            }
        )
    marker = {
        "schema_version": 1,
        "kind": "two_arm_full_suite_shard",
        "pair_manifest_sha256": pair.sha256,
        "roster_sha256": roster.sha256,
        "shard_plan_sha256": plan.sha256,
        "assignment_sha256": plan.assignment_sha256,
        "shard_index": shard_index,
        "shard_count": plan.shard_count,
        "task_count": len(plans),
        "tasks": artifacts,
    }
    marker_path = run_root / "shards" / (
        f"shard-{shard_index:05d}-of-{plan.shard_count:05d}.json"
    )
    _atomic_publish_json(marker_path, marker)
    return {
        "shard_index": shard_index,
        "shard_count": plan.shard_count,
        "task_count": len(plans),
        "executed": executed,
        "resumed": resumed,
        "marker": str(marker_path),
    }


def _expected_marker(
    *,
    run_root: Path,
    pair: PairManifest,
    roster: Roster,
    plan: ShardPlan,
    shard_index: int,
) -> dict[str, Any]:
    tasks = shard_tasks(roster, plan=plan, shard_index=shard_index)
    artifacts = []
    for task in tasks:
        path = run_root / "tasks" / task_dirname(task)
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"missing atomic task result: {path.name}")
        artifacts.append(
            {
                "directory": path.name,
                "manifest_sha256": sha256_file(path / "manifest.json"),
                "manifest_size_bytes": (path / "manifest.json").stat().st_size,
            }
        )
    return {
        "schema_version": 1,
        "kind": "two_arm_full_suite_shard",
        "pair_manifest_sha256": pair.sha256,
        "roster_sha256": roster.sha256,
        "shard_plan_sha256": plan.sha256,
        "assignment_sha256": plan.assignment_sha256,
        "shard_index": shard_index,
        "shard_count": plan.shard_count,
        "task_count": len(tasks),
        "tasks": artifacts,
    }


def _expected_canary_marker(
    *,
    run_root: Path,
    pair: PairManifest,
    roster: Roster,
    plan: ShardPlan,
) -> dict[str, Any]:
    artifacts = []
    for task in canary_tasks(roster, plan=plan):
        path = run_root / "tasks" / task_dirname(task)
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"missing atomic canary task result: {path.name}")
        artifacts.append(
            {
                "directory": path.name,
                "manifest_sha256": sha256_file(path / "manifest.json"),
                "manifest_size_bytes": (path / "manifest.json").stat().st_size,
            }
        )
    return {
        "schema_version": 1,
        "kind": "two_arm_full_suite_canary",
        "pair_manifest_sha256": pair.sha256,
        "roster_sha256": roster.sha256,
        "shard_plan_sha256": plan.sha256,
        "assignment_sha256": plan.assignment_sha256,
        "task_count": len(artifacts),
        "requirements": dict(plan.canary_requirements),
        "tasks": artifacts,
    }


def run_canary(
    run_root: Path,
    *,
    pair: PairManifest,
    roster: Roster,
    plan: ShardPlan,
    provenance: Mapping[str, str],
    executor: TaskExecutor,
    runtime_environment: Mapping[str, Any] | None = None,
    python_environment_contract: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    contract = initialize_run(
        run_root,
        pair=pair,
        roster=roster,
        plan=plan,
        provenance=provenance,
        runtime_environment=runtime_environment,
        python_environment_contract=python_environment_contract,
    )
    executed = 0
    resumed = 0
    for task in canary_tasks(roster, plan=plan):
        path = run_root / "tasks" / task_dirname(task)
        if path.exists():
            _validate_task_directory(path, task=task, run_contract=contract)
            resumed += 1
        else:
            _execute_and_publish_task(
                run_root=run_root,
                task=task,
                run_contract=contract,
                executor=executor,
            )
            executed += 1
    marker = _expected_canary_marker(
        run_root=run_root,
        pair=pair,
        roster=roster,
        plan=plan,
    )
    _atomic_publish_json(run_root / "canary.json", marker)
    return {
        "phase": "canary",
        "task_count": len(plan.canary_names),
        "executed": executed,
        "resumed": resumed,
        "marker": str(run_root / "canary.json"),
    }


def _comparison(
    task_payloads: Sequence[Mapping[str, Any]], arm_order: tuple[str, str]
) -> dict[str, Any]:
    if not task_payloads:
        raise ValueError("paired comparison requires at least one task")
    left, right = arm_order
    left_wins = right_wins = ties = 0
    paired_differences: list[float] = []
    metrics = {
        str(payload.get("task", {}).get("metric", "unspecified"))
        for payload in task_payloads
    }
    for payload in task_payloads:
        left_error = float(payload["results"][left]["metric_error"])
        right_error = float(payload["results"][right]["metric_error"])
        paired_differences.append(left_error - right_error)
        if left_error < right_error:
            left_wins += 1
        elif right_error < left_error:
            right_wins += 1
        else:
            ties += 1
    non_ties = left_wins + right_wins
    if non_ties:
        smaller = min(left_wins, right_wins)
        tail = sum(math.comb(non_ties, value) for value in range(smaller + 1))
        sign_p = min(1.0, 2.0 * tail / (2**non_ties))
    else:
        sign_p = 1.0
    total_rank = left_wins + right_wins + ties
    comparison: dict[str, Any] = {
        "left_arm": left,
        "right_arm": right,
        "dataset_count": len(task_payloads),
        "left_wins": left_wins,
        "right_wins": right_wins,
        "ties": ties,
        "mean_rank_lower_is_better": {
            left: (left_wins + 2 * right_wins + 1.5 * ties) / total_rank,
            right: (right_wins + 2 * left_wins + 1.5 * ties) / total_rank,
        },
        "two_sided_exact_sign_test_p": float(sign_p),
        "rank_win_sign_statistics_are_scale_free": True,
    }
    if len(metrics) == 1:
        differences = np.asarray(paired_differences, dtype=np.float64)
        generator = np.random.default_rng(_BOOTSTRAP_SEED)
        selections = generator.integers(
            0,
            len(differences),
            size=(_BOOTSTRAP_RESAMPLES, len(differences)),
        )
        bootstrap_means = differences[selections].mean(axis=1)
        bootstrap_low, bootstrap_high = np.quantile(
            bootstrap_means, [0.025, 0.975]
        )
        comparison["raw_metric_error_difference"] = {
            "metric": next(iter(metrics)),
            "mean": float(differences.mean()),
            "direction": "left_metric_error_minus_right_metric_error",
            "negative_means_left_is_better": True,
            "paired_bootstrap_95ci": [
                float(bootstrap_low),
                float(bootstrap_high),
            ],
            "paired_bootstrap_seed": _BOOTSTRAP_SEED,
            "paired_bootstrap_resamples": _BOOTSTRAP_RESAMPLES,
        }
    else:
        comparison["raw_metric_error_difference"] = None
        comparison["raw_difference_omission_reason"] = (
            "mixed metric scales; inspect comparison_by_metric"
        )
    return comparison


def aggregate_run(
    run_root: Path,
    *,
    pair: PairManifest,
    roster: Roster,
    plan: ShardPlan,
) -> dict[str, Any]:
    """Validate a complete run and atomically publish a deterministic aggregate."""
    _real_directory(run_root, label="run_root")
    contract = _read_json(run_root / "run.json", label="run contract")
    expected_contract = _run_contract(
        pair=pair,
        roster=roster,
        plan=plan,
        provenance=code_provenance(**contract.get("code_provenance", {})),
        runtime_environment=contract.get("runtime_environment"),
        python_environment_contract=contract.get("python_environment_contract"),
    )
    if contract != expected_contract:
        raise ValueError("run.json differs from the supplied pair or roster")

    expected_task_directories = {
        task_dirname(Task(index=index, name=name))
        for index, name in enumerate(roster.names)
    }
    tasks_root = run_root / "tasks"
    _real_directory(tasks_root, label="tasks control directory")
    task_entries = list(tasks_root.iterdir())
    if any(path.is_symlink() for path in task_entries):
        raise ValueError("task artifact set contains a symlink")
    observed_task_directories = {path.name for path in task_entries}
    non_directories = [path.name for path in task_entries if not path.is_dir()]
    if observed_task_directories != expected_task_directories or non_directories:
        raise ValueError(
            "task artifact set is incomplete or contains extras: "
            f"missing={sorted(expected_task_directories - observed_task_directories)}, "
            f"extra={sorted(observed_task_directories - expected_task_directories) + non_directories}"
        )

    staging_root = _real_directory(
        run_root / ".staging", label="staging control directory"
    )
    if any(True for _ in staging_root.iterdir()):
        raise ValueError("staging control directory must be empty before aggregation")
    for name in (".runtime", ".slurm-runtime", "gpu-monitor"):
        optional = run_root / name
        if optional.exists() or optional.is_symlink():
            _real_directory(optional, label=f"{name} runtime directory")
            if any(path.is_symlink() for path in optional.rglob("*")):
                raise ValueError(f"{name} runtime directory must not contain symlinks")

    payloads: list[dict[str, Any]] = []
    task_artifacts: list[dict[str, Any]] = []
    for index, name in enumerate(roster.names):
        task = Task(index=index, name=name)
        path = tasks_root / task_dirname(task)
        payloads.append(_validate_task_directory(path, task=task, run_contract=contract))
        task_artifacts.append(
            {
                "directory": path.name,
                "manifest_sha256": sha256_file(path / "manifest.json"),
                "manifest_size_bytes": (path / "manifest.json").stat().st_size,
            }
        )

    shards_root = run_root / "shards"
    if _read_json(run_root / "canary.json", label="canary marker") != (
        _expected_canary_marker(
            run_root=run_root,
            pair=pair,
            roster=roster,
            plan=plan,
        )
    ):
        raise ValueError("canary marker does not attest current task bytes")
    expected_marker_names = {
        f"shard-{index:05d}-of-{plan.shard_count:05d}.json"
        for index in range(plan.shard_count)
    }
    _real_directory(shards_root, label="shards control directory")
    marker_entries = list(shards_root.iterdir())
    if any(path.is_symlink() for path in marker_entries):
        raise ValueError("shard marker set contains a symlink")
    observed_markers = {path.name for path in marker_entries}
    unexpected_marker_entries = [path.name for path in marker_entries if not path.is_file()]
    if observed_markers != expected_marker_names or unexpected_marker_entries:
        raise ValueError("shard marker set is incomplete or contains extras")
    for index in range(plan.shard_count):
        marker_path = (
            shards_root / f"shard-{index:05d}-of-{plan.shard_count:05d}.json"
        )
        if _read_json(marker_path, label="shard marker") != _expected_marker(
            run_root=run_root,
            pair=pair,
            roster=roster,
            plan=plan,
            shard_index=index,
        ):
            raise ValueError(f"shard marker {index} does not attest current task bytes")

    regimes = sorted({str(payload["task"]["split_regime"]) for payload in payloads})
    problems = sorted({str(payload["task"]["problem_type"]) for payload in payloads})
    metrics = sorted({str(payload["task"]["metric"]) for payload in payloads})
    aggregate = {
        "schema_version": 1,
        "kind": "two_arm_full_suite_aggregate",
        "complete": True,
        "pair_id": pair.pair_id,
        "formal_eligible": pair.formal_eligible,
        "pair_manifest_sha256": pair.sha256,
        "roster_sha256": roster.sha256,
        "shard_plan_sha256": plan.sha256,
        "assignment_sha256": plan.assignment_sha256,
        "canary": {
            "task_count": len(plan.canary_names),
            "names": list(plan.canary_names),
            "requirements": dict(plan.canary_requirements),
        },
        "suite": roster.suite_id,
        "subset": "lite",
        "task_count": len(payloads),
        "result_count": len(payloads) * 2,
        "arm_order": list(pair.arm_order),
        "checkpoint_contract": dict(pair.checkpoint_contract),
        "comparison": _comparison(payloads, pair.arm_order),
        "comparison_by_split_regime": {
            regime: _comparison(
                [
                    payload
                    for payload in payloads
                    if payload["task"]["split_regime"] == regime
                ],
                pair.arm_order,
            )
            for regime in regimes
        },
        "comparison_by_problem_type": {
            problem: _comparison(
                [
                    payload
                    for payload in payloads
                    if payload["task"]["problem_type"] == problem
                ],
                pair.arm_order,
            )
            for problem in problems
        },
        "comparison_by_metric": {
            metric: _comparison(
                [
                    payload
                    for payload in payloads
                    if payload["task"]["metric"] == metric
                ],
                pair.arm_order,
            )
            for metric in metrics
        },
        "checkpoints": {
            arm.arm_id: {
                "display_name": arm.display_name,
                "sha256": arm.checkpoint_sha256,
                "size_bytes": arm.checkpoint_size_bytes,
            }
            for arm in pair.arms
        },
        "inference": dict(pair.inference),
        "code_provenance": dict(contract["code_provenance"]),
        "python_environment_contract": dict(
            contract["python_environment_contract"]
        ),
        "task_artifacts": task_artifacts,
        "datasets": [
            {
                "task": payload["task"],
                "results": payload["results"],
                "fallback": payload["fallback"],
            }
            for payload in payloads
        ],
    }
    _atomic_publish_json(run_root / "aggregate.json", aggregate)
    return aggregate


__all__ = [
    "Arm",
    "PairManifest",
    "PairTaskOOM",
    "Roster",
    "ShardPlan",
    "Task",
    "_FIXED_CLASSIFIER_OPTIONS",
    "aggregate_run",
    "build_task_payload",
    "code_provenance",
    "canary_tasks",
    "initialize_run",
    "load_pair_manifest",
    "load_roster",
    "load_shard_plan",
    "run_canary",
    "run_shard",
    "sha256_file",
    "shard_tasks",
    "task_dirname",
    "validate_task_payload",
    "validate_checkpoint_pair",
    "validate_plan_metadata",
]
