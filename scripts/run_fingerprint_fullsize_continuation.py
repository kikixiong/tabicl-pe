#!/usr/bin/env python3
"""Validate and run the exploratory full-size Fingerprint/RoPE continuation.

The original 5k pilot was deliberately built with a 5k scheduler horizon.  Its
terminal state is nevertheless identical to the 500k Stage-1 schedule at the
warmup boundary.  This runner constructs the Trainer with the 500k horizon,
fully restores the checkpoint, verifies the rebound scheduler, and only then
shortens the *loop* stop for a bounded Slurm segment.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from multiprocessing import set_start_method
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 1
STUDY = "tabiclv2-fullsize-rope-fingerprint-continuation-v1"
FORMAL_ELIGIBLE = False
SEED = 42
SCHEDULER_HORIZON_STEPS = 500_000
WARMUP_STEPS = 5_000
BASE_LR = 8e-4
LR_END = 1e-7
CHECKPOINT_CEILING_BYTES = 300_000_000
PRIOR_MANIFEST_SHA256 = (
    "e1e5f45cf434f8f6253507d33edf82572a088f7c44ededfb07ab27aacac9ab7d"
)
PRIOR_SCHEMA_SHA256 = "fa235438ee87d85c4d99439dc6dc7480b90fd54c3660819a739611d9ec390892"
ORIGIN_SOURCE_COMMIT = "6ec533890e93b8efa0f373bdf48eb7b1ffbb702f"
ALLOWED_SEGMENTS = {
    (5_000, 5_001),
    (5_000, 20_000),
    (20_000, 35_000),
    (35_000, 50_000),
}
ENVIRONMENT_KIND = "fingerprint_fullsize_continuation_environment"

ARM_CONTRACTS = {
    "rope": {
        "row_identity_mode": "rope",
        "row_fingerprint": False,
        "identity_manifest_sha256": (
            "1860cc20b54e95e0d0a7b5beacfcd993d1191e8ffbda071b91fd85d7ba3cb207"
        ),
        "state_tensors": 391,
        "state_elements": 27_552_258,
        "optimizer_state_entries": 390,
        "origin_checkpoint_sha256": (
            "a737194f2dd3a1e496c962845a1c88793cf8ccd2f37834597f57d173dc7bf821"
        ),
        "origin_completion_sha256": (
            "0a3953363e2f543077945e6d793ef2b4f6e23d021f5e748e3049c71310bb8ff1"
        ),
    },
    "fingerprint": {
        "row_identity_mode": "none",
        "row_fingerprint": True,
        "identity_manifest_sha256": (
            "f018876e213b2da2d7bce18621d05c25b06de5099d67a119b74cd3cc27bd16f3"
        ),
        "state_tensors": 404,
        "state_elements": 27_576_832,
        "optimizer_state_entries": 404,
        "origin_checkpoint_sha256": (
            "1068cfee5acd0fa47ee1150f8476ab8f1d9a83fb5934a6fcee110675b5aa3f8d"
        ),
        "origin_completion_sha256": (
            "01bed4dc0de9683aa8b5d2900778c26df0e00f4456f2cf767b9b66f1de6e0003"
        ),
    },
}

BASE_MODEL_CONFIG = {
    "activation": "gelu",
    "bias_free_ln": False,
    "col_affine": False,
    "col_feature_group": "same",
    "col_feature_group_size": 3,
    "col_nhead": 8,
    "col_num_blocks": 3,
    "col_num_inds": 128,
    "col_ssmax": "qassmax-mlp-elementwise",
    "col_target_aware": True,
    "dropout": 0.0,
    "embed_dim": 128,
    "ff_factor": 2,
    "icl_nhead": 8,
    "icl_num_blocks": 12,
    "icl_ssmax": "qassmax-mlp-elementwise",
    "max_classes": 10,
    "norm_first": True,
    "num_quantiles": 999,
    "recompute": False,
    "row_fingerprint_dim": 16,
    "row_nhead": 8,
    "row_num_blocks": 3,
    "row_num_cls": 4,
    "row_rope_base": 100000.0,
    "row_rope_interleaved": False,
    "zero_init": False,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _read_json(path: Path, *, max_bytes: int) -> tuple[dict[str, Any], str]:
    from tabicl.train._provenance import strict_json_load_nofollow

    value, digest = strict_json_load_nofollow(path, max_bytes=max_bytes)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value, digest


def _publish_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _stage_checkpoint(source: Path, destination: Path, *, expected_sha256: str) -> None:
    """Publish a private copy whose bytes are bound to the parent digest."""
    from tabicl.train._provenance import (
        _open_regular_nofollow,
        _same_file_snapshot,
    )

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace staged checkpoint: {destination}")
    if not destination.parent.is_dir():
        raise ValueError("staged checkpoint parent directory does not exist")
    source_fd, before = _open_regular_nofollow(source)
    if before.st_size < 1 or before.st_size > CHECKPOINT_CEILING_BYTES:
        os.close(source_fd)
        raise ValueError("parent checkpoint is outside the byte ceiling")
    destination_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    observed = 0
    try:
        os.fchmod(destination_fd, 0o600)
        with os.fdopen(source_fd, "rb", closefd=False) as source_handle:
            destination_handle = os.fdopen(destination_fd, "wb", closefd=True)
            destination_fd = -1
            with destination_handle:
                while chunk := source_handle.read(1 << 20):
                    observed += len(chunk)
                    if observed > CHECKPOINT_CEILING_BYTES:
                        raise ValueError(
                            "parent checkpoint grew beyond the byte ceiling"
                        )
                    digest.update(chunk)
                    destination_handle.write(chunk)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
        _same_file_snapshot(before, os.fstat(source_fd), where=str(source))
        if observed != before.st_size:
            raise ValueError("parent checkpoint size changed while staging")
        if digest.hexdigest() != expected_sha256:
            raise ValueError("staged parent checkpoint SHA-256 mismatch")
        os.chmod(temporary, 0o400, follow_symlinks=False)
        os.link(temporary, destination)
        temporary.unlink()
        directory = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.close(source_fd)
        if destination_fd >= 0:
            os.close(destination_fd)
        if temporary.exists():
            temporary.unlink()


def _manifest(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "payload": dict(payload),
    }
    return {**body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def _environment_payload() -> dict[str, Any]:
    import numpy

    from tabicl.train._provenance import checkpoint_sha256

    packages = [
        list(item)
        for item in sorted(
            {
                (
                    str(distribution.metadata.get("Name") or distribution.name).lower(),
                    str(distribution.version),
                )
                for distribution in importlib.metadata.distributions()
            }
        )
    ]
    files = {}
    for name, value in {
        "python": Path(sys.executable),
        "torch": Path(torch.__file__),
        "numpy": Path(numpy.__file__),
    }.items():
        path = value.resolve()
        files[name] = {"path": str(path), "sha256": checkpoint_sha256(path)}
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_cache_tag": sys.implementation.cache_tag,
        "prefix": str(Path(sys.prefix).resolve()),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_cudnn_build": torch.backends.cudnn.version(),
        "numpy_version": numpy.__version__,
        "packages": packages,
        "files": files,
    }


def prepare_environment(args: argparse.Namespace) -> None:
    record = _manifest(ENVIRONMENT_KIND, _environment_payload())
    _publish_json(Path(args.output), record)
    print(record["sha256"], flush=True)


def validate_environment_manifest(path: Path) -> dict[str, Any]:
    record = _load_manifest(path)
    if record["kind"] != ENVIRONMENT_KIND:
        raise ValueError("unexpected continuation environment kind")
    expected = _manifest(ENVIRONMENT_KIND, _environment_payload())
    if record != expected:
        raise ValueError("continuation environment fingerprint mismatch")
    return record


def _load_manifest(path: Path) -> dict[str, Any]:
    record, _digest = _read_json(path, max_bytes=1_048_576)
    if set(record) != {"schema_version", "kind", "payload", "sha256"}:
        raise ValueError("continuation manifest has unexpected fields")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("continuation manifest schema mismatch")
    if not isinstance(record["payload"], dict):
        raise TypeError("continuation manifest payload is not an object")
    body = {key: record[key] for key in ("schema_version", "kind", "payload")}
    expected = hashlib.sha256(_canonical(body)).hexdigest()
    if record["sha256"] != expected:
        raise ValueError("continuation manifest self-hash mismatch")
    return record


def expected_lr(step: int) -> float:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if step < WARMUP_STEPS:
        return BASE_LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (SCHEDULER_HORIZON_STEPS - WARMUP_STEPS)
    if progress >= 1.0:
        return LR_END
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LR_END + (BASE_LR - LR_END) * cosine


def _close(actual: float, expected: float, *, label: str) -> None:
    if not math.isfinite(actual) or not math.isclose(
        actual, expected, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def _expected_model_config(arm: str) -> dict[str, Any]:
    contract = ARM_CONTRACTS[arm]
    return {
        **BASE_MODEL_CONFIG,
        "row_identity_mode": contract["row_identity_mode"],
        "row_fingerprint": contract["row_fingerprint"],
    }


def validate_checkpoint(
    path: Path,
    *,
    arm: str,
    expected_step: int,
    expected_sha256: str | None,
) -> dict[str, Any]:
    if arm not in ARM_CONTRACTS:
        raise ValueError(f"unsupported arm: {arm}")
    from tabicl.train._provenance import _load_checkpoint_and_hash

    checkpoint, digest, size = _load_checkpoint_and_hash(
        path, max_checkpoint_bytes=CHECKPOINT_CEILING_BYTES
    )
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"checkpoint SHA-256 mismatch: {digest} != {expected_sha256}")
    from tabicl.train._rng_state import validate_full_resume_checkpoint

    validate_full_resume_checkpoint(checkpoint)
    if checkpoint["curr_step"] != expected_step:
        raise ValueError(
            f"checkpoint step mismatch: {checkpoint['curr_step']} != {expected_step}"
        )
    if checkpoint["config"] != _expected_model_config(arm):
        raise ValueError("checkpoint model/treatment configuration mismatch")

    contract = ARM_CONTRACTS[arm]
    state = checkpoint["state_dict"]
    if len(state) != contract["state_tensors"]:
        raise ValueError("checkpoint state tensor count mismatch")
    state_elements = sum(value.numel() for value in state.values())
    if state_elements != contract["state_elements"]:
        raise ValueError("checkpoint state element count mismatch")

    optimizer = checkpoint["optimizer_state"]
    if len(optimizer["state"]) != contract["optimizer_state_entries"]:
        raise ValueError("checkpoint optimizer state entry count mismatch")
    if len(optimizer["param_groups"]) != 1:
        raise ValueError("checkpoint must contain one Muon parameter group")
    group = optimizer["param_groups"][0]
    expected_group = {
        "use_muon": True,
        "weight_decay": 0.01,
        "matched_adamw_rms": 0.2,
        "momentum": 0.9,
        "nesterov": True,
        "ns_steps": 5,
        "adamw_betas": (0.9, 0.999),
        "adamw_eps": 1e-8,
        "use_cautious_wd": False,
        "initial_lr": BASE_LR,
    }
    for key, expected in expected_group.items():
        if group.get(key) != expected:
            raise ValueError(f"optimizer group mismatch for {key}")
    _close(float(group["lr"]), expected_lr(expected_step), label="optimizer lr")

    scheduler = checkpoint["scheduler_state"]
    if scheduler.get("last_epoch") != expected_step:
        raise ValueError("scheduler last_epoch mismatch")
    if scheduler.get("_step_count") != expected_step + 1:
        raise ValueError("scheduler _step_count mismatch")
    if scheduler.get("base_lrs") != [BASE_LR]:
        raise ValueError("scheduler base_lrs mismatch")
    if (
        not isinstance(scheduler.get("_last_lr"), list)
        or len(scheduler["_last_lr"]) != 1
    ):
        raise ValueError("scheduler _last_lr malformed")
    _close(
        float(scheduler["_last_lr"][0]),
        expected_lr(expected_step),
        label="scheduler last lr",
    )

    prior = checkpoint["prior_stream"]
    expected_prior = {
        "schema_version": 1,
        "algorithm": "sha256-schema-seed-rank-logical-step-v1",
        "experiment_seed": SEED,
        "ddp_rank": 0,
        "world_size": 1,
        "cursor": expected_step,
        "manifest_sha256": PRIOR_MANIFEST_SHA256,
        "schema_sha256": PRIOR_SCHEMA_SHA256,
    }
    for key, expected in expected_prior.items():
        if prior.get(key) != expected:
            raise ValueError(f"prior stream mismatch for {key}")

    identity = checkpoint["identity_treatment"]
    expected_identity = {
        "schema_version": 1,
        "row_identity_mode": contract["row_identity_mode"],
        "identity_rng_seed": SEED,
        "seed_policy": "sha256-domain-separated-base-seed-and-rank-v1",
        "sampler_version": None,
        "world_size": 1,
        "manifest_sha256": contract["identity_manifest_sha256"],
    }
    if identity != expected_identity:
        raise ValueError("checkpoint identity treatment mismatch")
    if "identity_sampler" in checkpoint:
        raise ValueError("RoPE/Fingerprint checkpoint unexpectedly has sampler state")

    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "size_bytes": size,
        "curr_step": expected_step,
        "state_tensors": len(state),
        "state_elements": state_elements,
        "optimizer_state_entries": len(optimizer["state"]),
        "scheduler_last_epoch": scheduler["last_epoch"],
        "scheduler_last_lr": float(scheduler["_last_lr"][0]),
        "prior_manifest_sha256": prior["manifest_sha256"],
        "prior_schema_sha256": prior["schema_sha256"],
    }


def _validate_origin_completion(
    path: Path,
    *,
    expected_sha256: str,
    expected_source_commit: str,
    checkpoint_contract: Mapping[str, Any],
    arm: str,
) -> str:
    completion, digest = _read_json(path, max_bytes=1_048_576)
    if digest != expected_sha256:
        raise ValueError("origin completion SHA-256 mismatch")
    expected_fields = {
        "schema_version": 1,
        "study": "tabiclv2-fullsize-rope-fingerprint-pilot-v1",
        "formal_evidence": False,
        "arm": arm,
        "source_commit": expected_source_commit,
        "curr_step": 5_000,
    }
    for key, expected in expected_fields.items():
        if completion.get(key) != expected:
            raise ValueError(f"origin completion mismatch for {key}")
    origin_checkpoint = completion.get("checkpoint")
    if not isinstance(origin_checkpoint, dict):
        raise TypeError("origin completion checkpoint is malformed")
    for key in ("path", "sha256", "size_bytes", "state_elements"):
        if origin_checkpoint.get(key) != checkpoint_contract.get(key):
            raise ValueError(f"origin completion checkpoint mismatch for {key}")
    treatment = completion.get("treatment", {})
    contract = ARM_CONTRACTS[arm]
    if treatment != {
        "row_identity_mode": contract["row_identity_mode"],
        "row_fingerprint": contract["row_fingerprint"],
        "row_fingerprint_dim": 16,
    }:
        raise ValueError("origin completion treatment mismatch")
    optimizer_prefix = completion.get("optimizer_prefix", {})
    if optimizer_prefix.get("scheduler") != "cosine_with_restarts":
        raise ValueError("origin completion scheduler mismatch")
    if optimizer_prefix.get("scheduler_horizon_steps") != 5_000:
        raise ValueError("origin completion scheduler horizon mismatch")
    if optimizer_prefix.get("warmup_steps") != WARMUP_STEPS:
        raise ValueError("origin completion warmup mismatch")
    return digest


def prepare_parent(args: argparse.Namespace) -> None:
    contract = ARM_CONTRACTS[args.arm]
    if args.origin_source_commit != ORIGIN_SOURCE_COMMIT:
        raise ValueError("unexpected origin source commit")
    if args.checkpoint_sha256 != contract["origin_checkpoint_sha256"]:
        raise ValueError("unexpected origin checkpoint SHA-256")
    if args.origin_completion_sha256 != contract["origin_completion_sha256"]:
        raise ValueError("unexpected origin completion SHA-256")
    environment = validate_environment_manifest(Path(args.environment_manifest))
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_contract = validate_checkpoint(
        checkpoint,
        arm=args.arm,
        expected_step=5_000,
        expected_sha256=args.checkpoint_sha256,
    )
    completion_digest = _validate_origin_completion(
        Path(args.origin_completion),
        expected_sha256=args.origin_completion_sha256,
        expected_source_commit=args.origin_source_commit,
        checkpoint_contract=checkpoint_contract,
        arm=args.arm,
    )
    record = _manifest(
        "fingerprint_fullsize_continuation_parent",
        {
            "study": STUDY,
            "formal_eligible": FORMAL_ELIGIBLE,
            "arm": args.arm,
            "seed": SEED,
            "source_commit": args.origin_source_commit,
            "origin_completion_sha256": completion_digest,
            "origin_scheduler_horizon_steps": 5_000,
            "continuation_scheduler_horizon_steps": SCHEDULER_HORIZON_STEPS,
            "continuation_environment_sha256": environment["sha256"],
            "checkpoint": checkpoint_contract,
        },
    )
    _publish_json(Path(args.output), record)
    print(record["sha256"], flush=True)


def validate_parent_manifest(
    path: Path,
    *,
    arm: str,
    expected_step: int,
    expected_source_commit: str,
    expected_environment_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _load_manifest(path)
    if record["kind"] not in {
        "fingerprint_fullsize_continuation_parent",
        "fingerprint_fullsize_segment_completion",
    }:
        raise ValueError("unexpected continuation parent kind")
    payload = record["payload"]
    expected = {
        "study": STUDY,
        "formal_eligible": FORMAL_ELIGIBLE,
        "arm": arm,
        "seed": SEED,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"continuation parent mismatch for {key}")
    if record["kind"] == "fingerprint_fullsize_segment_completion":
        expected_payload_fields = {
            "study",
            "formal_eligible",
            "arm",
            "seed",
            "source_commit",
            "environment_sha256",
            "from_step",
            "to_step",
            "scheduler_horizon_steps",
            "warmup_steps",
            "parent_manifest_path",
            "parent_manifest_sha256",
            "parent_checkpoint_sha256",
            "checkpoint",
            "runtime",
        }
    else:
        expected_payload_fields = {
            "study",
            "formal_eligible",
            "arm",
            "seed",
            "source_commit",
            "origin_completion_sha256",
            "origin_scheduler_horizon_steps",
            "continuation_scheduler_horizon_steps",
            "continuation_environment_sha256",
            "checkpoint",
        }
    if set(payload) != expected_payload_fields:
        raise ValueError("continuation parent payload has unexpected fields")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise TypeError("continuation parent checkpoint is malformed")
    if checkpoint.get("curr_step") != expected_step:
        raise ValueError("continuation parent step mismatch")
    if record["kind"] == "fingerprint_fullsize_segment_completion":
        if payload.get("source_commit") != expected_source_commit:
            raise ValueError("segment completion source commit mismatch")
        if payload.get("environment_sha256") != expected_environment_sha256:
            raise ValueError("segment completion environment mismatch")
        if payload.get("to_step") != expected_step:
            raise ValueError("segment completion terminal step mismatch")
        boundary = (payload.get("from_step"), payload.get("to_step"))
        if boundary not in ALLOWED_SEGMENTS:
            raise ValueError("segment completion boundary is not approved")
        if payload.get("scheduler_horizon_steps") != SCHEDULER_HORIZON_STEPS:
            raise ValueError("segment completion scheduler horizon mismatch")
        if payload.get("warmup_steps") != WARMUP_STEPS:
            raise ValueError("segment completion warmup mismatch")
        parent_manifest_value = payload.get("parent_manifest_path")
        if not isinstance(parent_manifest_value, str):
            raise TypeError("segment parent manifest path is malformed")
        upstream_path = Path(parent_manifest_value)
        if not upstream_path.is_absolute() or str(upstream_path.resolve()) != str(
            upstream_path
        ):
            raise ValueError("segment parent manifest path is not canonical absolute")
        upstream_record, upstream_checkpoint = validate_parent_manifest(
            upstream_path,
            arm=arm,
            expected_step=payload["from_step"],
            expected_source_commit=expected_source_commit,
            expected_environment_sha256=expected_environment_sha256,
        )
        if payload.get("parent_manifest_sha256") != upstream_record["sha256"]:
            raise ValueError("segment upstream manifest SHA-256 mismatch")
        if payload.get("parent_checkpoint_sha256") != upstream_checkpoint["sha256"]:
            raise ValueError("segment upstream checkpoint SHA-256 mismatch")
        runtime = payload.get("runtime")
        if not isinstance(runtime, dict) or set(runtime) != {
            "python",
            "torch",
            "cuda",
            "gpu",
        }:
            raise TypeError("segment runtime record is malformed")
    else:
        if payload.get("source_commit") != ORIGIN_SOURCE_COMMIT:
            raise ValueError("origin parent source commit mismatch")
        if (
            payload.get("origin_completion_sha256")
            != ARM_CONTRACTS[arm]["origin_completion_sha256"]
        ):
            raise ValueError("origin parent completion SHA-256 mismatch")
        if payload.get("origin_scheduler_horizon_steps") != 5_000:
            raise ValueError("origin parent scheduler horizon mismatch")
        if (
            payload.get("continuation_scheduler_horizon_steps")
            != SCHEDULER_HORIZON_STEPS
        ):
            raise ValueError("origin parent continuation horizon mismatch")
        if (
            payload.get("continuation_environment_sha256")
            != expected_environment_sha256
        ):
            raise ValueError("origin parent continuation environment mismatch")
        if checkpoint.get("sha256") != ARM_CONTRACTS[arm]["origin_checkpoint_sha256"]:
            raise ValueError("origin parent checkpoint SHA-256 mismatch")
    actual = validate_checkpoint(
        Path(checkpoint["path"]),
        arm=arm,
        expected_step=expected_step,
        expected_sha256=checkpoint["sha256"],
    )
    if set(checkpoint) != set(actual):
        raise ValueError("parent checkpoint contract has unexpected fields")
    for key, value in actual.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"parent checkpoint contract mismatch for {key}")
    return record, actual


def _git_head(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout.strip()


def _training_config_args(
    *,
    arm: str,
    checkpoint_path: Path,
    checkpoint_dir: Path,
    wandb_dir: Path,
    save_temp_every: int,
) -> list[str]:
    contract = ARM_CONTRACTS[arm]
    return [
        "--wandb_log",
        "False",
        "--wandb_project",
        "TabICLv2-Fingerprint-Fullsize-Continuation",
        "--wandb_name",
        f"tabiclv2_fullsize_{arm}_stage1_continuation_seed42",
        "--wandb_mode",
        "disabled",
        "--wandb_dir",
        str(wandb_dir),
        "--device",
        "cuda",
        "--dtype",
        "float32",
        "--amp",
        "True",
        "--np_seed",
        str(SEED),
        "--torch_seed",
        str(SEED),
        "--max_steps",
        str(SCHEDULER_HORIZON_STEPS),
        "--batch_size",
        "64",
        "--micro_batch_size",
        "8",
        "--lr",
        str(BASE_LR),
        "--muon",
        "True",
        "--beta1",
        "0.9",
        "--weight_decay",
        "0.01",
        "--use_cautious_wd",
        "False",
        "--scheduler",
        "cosine_with_restarts",
        "--warmup_proportion",
        "-1",
        "--warmup_steps",
        str(WARMUP_STEPS),
        "--cosine_num_cycles",
        "1",
        "--cosine_amplitude_decay",
        "1",
        "--cosine_lr_end",
        str(LR_END),
        "--gradient_clipping",
        "10",
        "--fail_on_oom",
        "True",
        "--fail_on_nonfinite",
        "True",
        "--prior_type",
        "graph_scm",
        "--prior_device",
        "cpu",
        "--n_jobs",
        "48",
        "--batch_size_per_gp",
        "8",
        "--min_features",
        "1",
        "--max_features",
        "100",
        "--max_classes",
        "10",
        "--max_seq_len",
        "1024",
        "--min_train_size",
        "0.3",
        "--max_train_size",
        "0.9",
        "--seq_len_per_gp",
        "True",
        "--graph_noise",
        "False",
        "--filter_unpredictable_graphs",
        "True",
        "--filter_unpredictable_datasets",
        "True",
        "--allow_act_warping",
        "False",
        "--min_n_nodes",
        "2",
        "--max_n_nodes",
        "32",
        "--cauchy_dag_offset",
        "0",
        "--embed_dim",
        "128",
        "--col_num_blocks",
        "3",
        "--col_nhead",
        "8",
        "--col_num_inds",
        "128",
        "--col_affine",
        "False",
        "--col_feature_group",
        "same",
        "--col_feature_group_size",
        "3",
        "--col_target_aware",
        "True",
        "--col_ssmax",
        "True",
        "--row_num_blocks",
        "3",
        "--row_nhead",
        "8",
        "--row_num_cls",
        "4",
        "--row_rope_base",
        "100000",
        "--row_rope_interleaved",
        "False",
        "--row_identity_mode",
        str(contract["row_identity_mode"]),
        "--row_fingerprint",
        str(contract["row_fingerprint"]),
        "--row_fingerprint_dim",
        "16",
        "--icl_num_blocks",
        "12",
        "--icl_nhead",
        "8",
        "--icl_ssmax",
        "True",
        "--ssmax_type",
        "qassmax-mlp-elementwise",
        "--ff_factor",
        "2",
        "--norm_first",
        "True",
        "--zero_init",
        "False",
        "--use_flash_attn3",
        "False",
        "--checkpoint_path",
        str(checkpoint_path),
        "--only_load_model",
        "False",
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--save_temp_every",
        str(save_temp_every),
        "--save_perm_every",
        "10000",
        "--max_checkpoints",
        "1",
        "--max_checkpoint_bytes",
        str(CHECKPOINT_CEILING_BYTES),
        "--empty_cache_every",
        "0",
        "--progress_refresh_seconds",
        "10",
    ]


def _validate_loaded_trainer(trainer: Any, *, arm: str, from_step: int) -> None:
    if not getattr(trainer, "_loaded_full_resume", False):
        raise ValueError("Trainer did not perform a full-state resume")
    if trainer.curr_step != from_step or trainer.prior_cursor != from_step:
        raise ValueError("Trainer resume cursor mismatch")
    if trainer.config.only_load_model:
        raise ValueError("continuation must not use model-only loading")
    scheduler = trainer.scheduler
    if scheduler.last_epoch != from_step:
        raise ValueError("loaded Trainer scheduler boundary mismatch")
    if len(scheduler.lr_lambdas) != 1 or not isinstance(
        scheduler.lr_lambdas[0], functools.partial
    ):
        raise TypeError("loaded Trainer scheduler lambda is malformed")
    scheduler_keywords = scheduler.lr_lambdas[0].keywords
    expected_keywords = {
        "num_warmup_steps": WARMUP_STEPS,
        "num_training_steps": SCHEDULER_HORIZON_STEPS,
        "num_cycles": 1,
        "amplitude_decay": 1.0,
        "lr_end": LR_END,
        "lr_init": BASE_LR,
    }
    if scheduler_keywords != expected_keywords:
        raise ValueError("loaded Trainer scheduler horizon/protocol mismatch")
    _close(
        float(scheduler.get_last_lr()[0]),
        expected_lr(from_step),
        label="loaded Trainer lr",
    )
    rebound_next = BASE_LR * float(scheduler.lr_lambdas[0](from_step + 1))
    _close(
        rebound_next,
        expected_lr(from_step + 1),
        label="500k rebound next lr",
    )
    if trainer.model_config != _expected_model_config(arm):
        raise ValueError("loaded Trainer model configuration mismatch")


def run_segment(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).resolve()
    environment_path = Path(args.environment_manifest).resolve()
    environment = validate_environment_manifest(environment_path)
    expected_init = source_root / "src" / "tabicl" / "__init__.py"
    import tabicl
    from tabicl.train._run import Trainer
    from tabicl.train._train_config import build_parser

    actual_init = Path(tabicl.__file__).resolve()
    if actual_init != expected_init:
        raise ValueError(f"wrong tabicl source: {actual_init} != {expected_init}")
    if _git_head(source_root) != args.source_commit:
        raise ValueError("source checkout HEAD changed before training")
    boundary = (args.from_step, args.stop_after_step)
    if boundary not in ALLOWED_SEGMENTS:
        raise ValueError(f"continuation segment boundary is not approved: {boundary}")

    parent_path = Path(args.parent_manifest).resolve()
    parent, parent_checkpoint = validate_parent_manifest(
        parent_path,
        arm=args.arm,
        expected_step=args.from_step,
        expected_source_commit=args.source_commit,
        expected_environment_sha256=environment["sha256"],
    )
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    wandb_dir = Path(args.wandb_dir).resolve()
    completion_output = Path(args.completion_output).resolve()
    if checkpoint_dir.exists() or checkpoint_dir.is_symlink():
        raise FileExistsError("segment checkpoint directory must be fresh")
    if completion_output.exists() or completion_output.is_symlink():
        raise FileExistsError("segment completion output must be fresh")
    staged_parent = completion_output.parent / "parent.ckpt"
    _stage_checkpoint(
        Path(parent_checkpoint["path"]),
        staged_parent,
        expected_sha256=parent_checkpoint["sha256"],
    )
    staged_contract = validate_checkpoint(
        staged_parent,
        arm=args.arm,
        expected_step=args.from_step,
        expected_sha256=parent_checkpoint["sha256"],
    )
    for key, value in parent_checkpoint.items():
        if key != "path" and staged_contract.get(key) != value:
            raise ValueError(f"staged parent checkpoint mismatch for {key}")
    wandb_dir.mkdir(parents=True, exist_ok=False)

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    from tabicl.train._provenance import _open_regular_nofollow

    trusted_parent_fd, _trusted_parent_stat = _open_regular_nofollow(staged_parent)
    try:
        config = build_parser().parse_args(
            _training_config_args(
                arm=args.arm,
                checkpoint_path=Path(f"/proc/self/fd/{trusted_parent_fd}"),
                checkpoint_dir=checkpoint_dir,
                wandb_dir=wandb_dir,
                save_temp_every=(
                    1 if args.stop_after_step == args.from_step + 1 else 1000
                ),
            )
        )
        trainer = Trainer(config)
    finally:
        os.close(trusted_parent_fd)
    _validate_loaded_trainer(trainer, arm=args.arm, from_step=args.from_step)

    # The scheduler lambda was constructed above with the immutable 500k
    # horizon.  Only the loop boundary is shortened for this Slurm segment.
    trainer.config.max_steps = args.stop_after_step
    trainer.train()
    if trainer.curr_step != args.stop_after_step:
        raise RuntimeError("Trainer stopped before the requested segment boundary")

    final_path = checkpoint_dir / f"step-{args.stop_after_step}.ckpt"
    final_contract = validate_checkpoint(
        final_path,
        arm=args.arm,
        expected_step=args.stop_after_step,
        expected_sha256=None,
    )
    if _git_head(source_root) != args.source_commit:
        raise ValueError("source checkout HEAD changed during training")
    status = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if status.stdout:
        raise ValueError("source checkout became dirty during training")
    final_environment = validate_environment_manifest(environment_path)
    if final_environment["sha256"] != environment["sha256"]:
        raise ValueError("continuation environment changed during training")
    completion = _manifest(
        "fingerprint_fullsize_segment_completion",
        {
            "study": STUDY,
            "formal_eligible": FORMAL_ELIGIBLE,
            "arm": args.arm,
            "seed": SEED,
            "source_commit": args.source_commit,
            "environment_sha256": environment["sha256"],
            "from_step": args.from_step,
            "to_step": args.stop_after_step,
            "scheduler_horizon_steps": SCHEDULER_HORIZON_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "parent_manifest_path": str(parent_path),
            "parent_manifest_sha256": parent["sha256"],
            "parent_checkpoint_sha256": parent_checkpoint["sha256"],
            "checkpoint": final_contract,
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
        },
    )
    _publish_json(completion_output, completion)
    print(json.dumps({"completion_sha256": completion["sha256"]}), flush=True)


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    environment = subparsers.add_parser("prepare-environment")
    environment.add_argument("--output", required=True)
    environment.set_defaults(handler=prepare_environment)

    prepare = subparsers.add_parser("prepare-parent")
    prepare.add_argument("--arm", choices=tuple(ARM_CONTRACTS), required=True)
    prepare.add_argument("--checkpoint", required=True)
    prepare.add_argument("--checkpoint-sha256", required=True)
    prepare.add_argument("--origin-completion", required=True)
    prepare.add_argument("--origin-completion-sha256", required=True)
    prepare.add_argument("--origin-source-commit", required=True)
    prepare.add_argument("--environment-manifest", required=True)
    prepare.add_argument("--output", required=True)
    prepare.set_defaults(handler=prepare_parent)

    run = subparsers.add_parser("run-segment")
    run.add_argument("--arm", choices=tuple(ARM_CONTRACTS), required=True)
    run.add_argument("--source-root", required=True)
    run.add_argument("--source-commit", required=True)
    run.add_argument("--environment-manifest", required=True)
    run.add_argument("--parent-manifest", required=True)
    run.add_argument("--from-step", type=int, required=True)
    run.add_argument("--stop-after-step", type=int, required=True)
    run.add_argument("--checkpoint-dir", required=True)
    run.add_argument("--wandb-dir", required=True)
    run.add_argument("--completion-output", required=True)
    run.set_defaults(handler=run_segment)
    return parser


def main() -> None:
    args = build_cli().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
