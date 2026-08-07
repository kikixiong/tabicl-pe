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
from pathlib import Path
import re
import secrets
import stat
import sys
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

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_METRIC_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_SLURM_DURATION = re.compile(
    r"^(?:(?P<days>[1-9][0-9]{0,3})-)?"
    r"(?P<hours>[0-9]{2}):(?P<minutes>[0-5][0-9]):(?P<seconds>[0-5][0-9])$"
)

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
    "checkpoint_ceiling_bytes",
    "static_protocol_sha256_by_stage",
    "time_limit_by_stage",
    "predecessor_acceptance_sha256_by_seed",
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
    if isinstance(value, bool) or not isinstance(value, int) or value not in FORMAL_SEEDS:
        raise ValueError(f"{where} must be exactly one of 42, 43, or 44")
    return value


def _slurm_duration(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a canonical Slurm duration")
    match = _SLURM_DURATION.fullmatch(value)
    if match is None or int(match.group("hours")) > 23:
        raise ValueError(f"{where} must be a canonical Slurm duration")
    seconds = (
        (int(match.group("days") or "0") * 24 + int(match.group("hours")))
        * 3600
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


def _stat_signature(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
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
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["kind"] != expected_kind:
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
            raise ValueError("published temporary manifest is not a stable regular file")
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
            "gpu_model",
        },
        "campaign H100 gate",
    )
    h100_sha256 = _digest(gate["attestation_sha256"], "campaign H100 attestation")
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
        time_limit = _slurm_duration(stage["time_limit"], f"{expected_stage} time limit")
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
            "gpu_model": gate["gpu_model"],
        },
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
    """Build a campaign only after the exact H100 attestation validates.

    When ``exact_root`` is supplied, the exact-candidate H100 validator is
    invoked and proves the full twelve-case attestation.  Tests and offline
    schema tooling may omit it, but production campaign publication requires
    it through :func:`publish_campaign`.
    """

    source = _validated_training_source(training_source)
    h100_sha = _digest(expected_h100_sha256, "expected H100 attestation")
    ceiling = _positive_int(checkpoint_ceiling_bytes, "checkpoint ceiling")
    if exact_root is not None:
        root = _absolute_path(exact_root, "exact candidate root")
        helper_path = root / "scripts/run_h100_identity_validation.py"
        spec = importlib.util.spec_from_file_location(
            "_formal_campaign_h100_validation", helper_path
        )
        if spec is None or spec.loader is None:
            raise ValueError("cannot load exact-T H100 validation helper")
        helper = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = helper
        spec.loader.exec_module(helper)
        helper.validate_smoke_attestation(
            h100_attestation,
            expected_sha256=h100_sha,
            expected_commit_sha=source["commit_sha"],
            expected_tree_sha=source["tree_sha"],
            expected_environment_sha256=source["environment_sha256"],
            expected_source_manifest_sha256=source["source_manifest_sha256"],
            expected_gpu_model=expected_gpu_model,
            expected_checkpoint_ceiling_bytes=ceiling,
        )
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
    if observed_max > ceiling:
        raise ValueError("H100 observed checkpoint maximum exceeds campaign ceiling")
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
    manifest = make_manifest(
        "formal_campaign",
        {
            "campaign_id": _safe_id(campaign_id, "campaign_id"),
            "training_source": source,
            "h100_gate": {
                "attestation_sha256": h100_sha,
                "checkpoint_ceiling_bytes": ceiling,
                "observed_checkpoint_max_bytes": observed_max,
                "gpu_model": expected_gpu_model,
            },
            "supported_seeds": list(FORMAL_SEEDS),
            "formal_arms": list(ARMS),
            "stages": normalized_stages,
        },
    )
    validate_campaign_manifest(manifest)
    return manifest


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
        raise ValueError("campaign binding predecessor set does not match selected seed")
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
    acceptance_ceiling = _positive_int(
        acceptance_ceiling_bytes, "acceptance ceiling"
    )
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
    previous = {f"seed-{value}-accepted.json" for value in FORMAL_SEEDS if value < selected}
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
        "manifest_ceiling_bytes",
        "sacct_sha256",
        "path_format",
        "terminal_verified",
        "terminal_queries",
        "jobs",
    }
    payload = _exact(payload, required, "formal terminal payload")
    selected_seed = _formal_seed(seed)
    expected_study = f"{campaign_value['campaign_id']}-seed{selected_seed}"
    if (
        payload["study_id"] != expected_study
        or payload["seed"] != selected_seed
        or payload["source_commit_sha"]
        != campaign_value["training_source"]["commit_sha"]
        or payload["source_tree_sha"] != campaign_value["training_source"]["tree_sha"]
        or payload["campaign_binding"] != expected_campaign_binding
        or payload["terminal_verified"] is not True
        or payload["path_format"] != "artifact_root_relative_posix_v1"
    ):
        raise ValueError("formal terminal attestation campaign binding mismatch")
    _digest(payload["submission_receipt_sha256"], "terminal submission receipt")
    _digest(payload["transaction_ledger_sha256"], "terminal transaction ledger")
    jobs = payload["jobs"]
    expected_pairs = [(arm, stage) for stage, _ in STAGES for arm in ARMS]
    if not isinstance(jobs, list) or len(jobs) != len(expected_pairs):
        raise ValueError("formal terminal attestation must contain exactly nine jobs")
    observed_pairs: list[tuple[Any, Any]] = []
    for job in jobs:
        if not isinstance(job, Mapping):
            raise ValueError("formal terminal job must be an object")
        observed_pairs.append((job.get("arm"), job.get("stage")))
        if (
            job.get("state") != "COMPLETED"
            or job.get("exit_code") != "0:0"
            or job.get("derived_exit_code") != "0:0"
            or not isinstance(job.get("finalized_artifact"), Mapping)
        ):
            raise ValueError("formal terminal job is not independently successful")
        completion = job.get("completion")
        if (
            not isinstance(completion, Mapping)
            or completion.get("seed") != selected_seed
            or completion.get("repository_binding") != payload["repository_binding"]
        ):
            raise ValueError("formal terminal job completion binding mismatch")
    if observed_pairs != expected_pairs:
        raise ValueError("formal terminal job matrix/order is not canonical")
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


def _acceptance_payload(
    *,
    campaign: Mapping[str, Any],
    seed: int,
    predecessor_sha256: str | None,
    predecessor_map: Mapping[str, str],
    terminal_path: Path,
    terminal: Mapping[str, Any],
    evaluation_path: Path,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_campaign_manifest(campaign)
    binding = campaign_binding(
        campaign, predecessor_acceptance_sha256_by_seed=predecessor_map
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
    terminal_path = _absolute_path(terminal_binding["path"], "terminal attestation path")
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
    validate_campaign_manifest(campaign)
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
    terminal_attestation_path: str | os.PathLike[str],
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
    evaluation_path = _absolute_path(
        evaluation_receipt_path, "minimum evaluation receipt path"
    )
    if evaluation_path != evaluation_root / f"seed-{selected_seed}.json":
        raise ValueError("evaluation receipt path is not canonical for the selected seed")
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
        evaluation_path=evaluation_path,
        evaluation=evaluation,
    )
    acceptance = make_manifest("formal_seed_acceptance", payload)
    output = registry / current_name
    publish_manifest_no_replace(
        output, acceptance, max_bytes=acceptance_max_bytes
    )
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


def _read_plain_canonical_json(path: Path, *, max_bytes: int) -> Mapping[str, Any]:
    """Read a canonical non-manifest JSON spec with the manifest reader rules."""

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
            "h100_attestation_path",
            "h100_attestation_sha256",
            "h100_attestation_ceiling_bytes",
            "checkpoint_ceiling_bytes",
            "expected_gpu_model",
            "stages",
        },
        "formal campaign spec",
    )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

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
    accept.add_argument("--terminal-attestation", required=True, type=Path)
    accept.add_argument("--evaluation-receipt", required=True, type=Path)

    capacity = subparsers.add_parser("capacity")
    capacity.add_argument("--fragment-size", required=True, type=int)

    args = parser.parse_args(argv)
    if args.command == "authorize-seed":
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
            terminal_attestation_path=args.terminal_attestation,
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
