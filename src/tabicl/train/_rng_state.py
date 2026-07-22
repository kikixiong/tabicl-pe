"""Capture and restore every stochastic source used by a training rank."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor


RANK_RNG_SCHEMA_VERSION = 1
ALL_RANK_RNG_SCHEMA_VERSION = 1


def _tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _manifest_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rank_manifest(state: dict[str, Any]) -> dict[str, Any]:
    numpy_state = state["numpy"]
    return {
        "schema_version": state["schema_version"],
        "rank": state["rank"],
        "python": state["python"],
        "numpy": {
            "bit_generator": numpy_state["bit_generator"],
            "keys_sha256": _tensor_sha256(numpy_state["keys"]),
            "position": numpy_state["position"],
            "has_gauss": numpy_state["has_gauss"],
            "cached_gaussian": numpy_state["cached_gaussian"],
        },
        "torch_cpu_sha256": _tensor_sha256(state["torch_cpu"]),
        "torch_cuda_sha256": [_tensor_sha256(value) for value in state["torch_cuda"]],
        "cuda_device_count": state["cuda_device_count"],
    }


def capture_rank_rng_state(*, rank: int = 0) -> dict[str, Any]:
    """Return a weights-only-loadable snapshot of Python, NumPy, and Torch RNGs."""
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "schema_version": RANK_RNG_SCHEMA_VERSION,
        "rank": rank,
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            # torch.uint32 storage does not round-trip through
            # all_gather_object on every supported PyTorch/Python pairing.
            # int64 is weights-only safe and exactly represents every MT key.
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [value.clone().cpu() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    state["manifest_sha256"] = _manifest_sha256(_rank_manifest(state))
    return state


def _validate_rank_rng_state(state: dict[str, Any]) -> None:
    if state.get("schema_version") != RANK_RNG_SCHEMA_VERSION:
        raise ValueError("rank RNG schema_version mismatch")
    required = {"rank", "python", "numpy", "torch_cpu", "torch_cuda", "cuda_device_count"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"rank RNG state is missing: {', '.join(missing)}")
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, dict) or not isinstance(numpy_state.get("keys"), Tensor):
        raise ValueError("rank RNG NumPy state is invalid")
    if not isinstance(state["torch_cpu"], Tensor) or not all(
        isinstance(value, Tensor) for value in state["torch_cuda"]
    ):
        raise ValueError("rank RNG Torch state is invalid")
    expected_hash = _manifest_sha256(_rank_manifest(state))
    if state.get("manifest_sha256") != expected_hash:
        raise ValueError("rank RNG state manifest hash mismatch")
    actual_cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if state["cuda_device_count"] != actual_cuda_count:
        raise ValueError(
            "rank RNG cuda_device_count mismatch: "
            f"checkpoint has {state['cuda_device_count']}, runtime has {actual_cuda_count}"
        )


def restore_rank_rng_state(state: dict[str, Any]) -> None:
    """Validate a rank snapshot completely, then atomically restore its RNG streams."""
    _validate_rank_rng_state(state)
    numpy_state = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=True),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all([value.detach().cpu() for value in state["torch_cuda"]])


def _all_rank_manifest(rank_states: dict[str, dict[str, Any]], world_size: int) -> dict[str, Any]:
    return {
        "schema_version": ALL_RANK_RNG_SCHEMA_VERSION,
        "world_size": world_size,
        "rank_state_hashes": {
            key: rank_states[key]["manifest_sha256"] for key in sorted(rank_states, key=int)
        },
    }


def make_all_rank_rng_bundle(
    states: list[dict[str, Any]], *, world_size: int | None = None
) -> dict[str, Any]:
    if not states:
        raise ValueError("all-rank RNG bundle cannot be empty")
    world_size = len(states) if world_size is None else world_size
    rank_states: dict[str, dict[str, Any]] = {}
    for state in states:
        _validate_rank_rng_state(state)
        rank_key = str(state["rank"])
        if rank_key in rank_states:
            raise ValueError(f"duplicate rank RNG state for rank {rank_key}")
        rank_states[rank_key] = state
    expected = {str(rank) for rank in range(world_size)}
    if set(rank_states) != expected:
        raise ValueError(
            f"all-rank RNG bundle must contain ranks {sorted(expected)}, got {sorted(rank_states)}"
        )
    manifest = _all_rank_manifest(rank_states, world_size)
    return {
        "schema_version": ALL_RANK_RNG_SCHEMA_VERSION,
        "world_size": world_size,
        "rank_states": rank_states,
        "manifest_sha256": _manifest_sha256(manifest),
    }


def select_rank_rng_state(
    bundle: dict[str, Any], *, rank: int, world_size: int
) -> dict[str, Any]:
    if bundle.get("schema_version") != ALL_RANK_RNG_SCHEMA_VERSION:
        raise ValueError("all-rank RNG schema_version mismatch")
    if bundle.get("world_size") != world_size:
        raise ValueError(
            f"all-rank RNG world_size mismatch: expected {world_size}, got {bundle.get('world_size')}"
        )
    rank_states = bundle.get("rank_states")
    if not isinstance(rank_states, dict):
        raise ValueError("all-rank RNG rank_states is missing")
    for outer_rank, state in rank_states.items():
        inner_rank = state.get("rank") if isinstance(state, dict) else None
        if outer_rank != str(inner_rank):
            raise ValueError(
                f"all-rank RNG outer rank key {outer_rank!r} does not match "
                f"inner rank {inner_rank!r}"
            )
    normalized = make_all_rank_rng_bundle(list(rank_states.values()), world_size=world_size)
    if normalized["manifest_sha256"] != bundle.get("manifest_sha256"):
        raise ValueError("all-rank RNG bundle manifest hash mismatch")
    try:
        return normalized["rank_states"][str(rank)]
    except KeyError as error:
        raise ValueError(f"all-rank RNG state is missing rank {rank}") from error


def gather_all_rank_rng_state(*, rank: int, world_size: int) -> dict[str, Any]:
    """Collect all rank stochastic states through the active process group."""
    local = capture_rank_rng_state(rank=rank)
    if world_size == 1 and not dist.is_initialized():
        return make_all_rank_rng_bundle([local], world_size=1)
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("world_size > 1 requires an initialized distributed process group")
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise RuntimeError("distributed process group does not match RNG rank/world_size")
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    if any(value is None for value in gathered):
        raise RuntimeError("distributed RNG gather returned an empty rank")
    return make_all_rank_rng_bundle(gathered, world_size=world_size)  # type: ignore[arg-type]


def restore_all_rank_rng_state(
    bundle: dict[str, Any], *, rank: int, world_size: int
) -> None:
    restore_rank_rng_state(select_rank_rng_state(bundle, rank=rank, world_size=world_size))


def validate_full_resume_checkpoint(checkpoint: dict[str, Any]) -> None:
    """Reject legacy/partial state before any training object is mutated."""
    required = (
        "state_dict",
        "optimizer_state",
        "scheduler_state",
        "curr_step",
        "identity_treatment",
        "rng_state",
        "prior_stream",
        "scaler_state",
    )
    missing = [field for field in required if field not in checkpoint]
    if missing:
        raise ValueError(f"full resume checkpoint is missing required keys: {', '.join(missing)}")
