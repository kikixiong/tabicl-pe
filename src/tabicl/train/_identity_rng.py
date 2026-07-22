"""Dedicated, checkpointable RNG for Temporary Table-wise Identity."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor


SAMPLER_VERSION = "tabicl-temporary-identity/randperm-cpu-v1"
STATE_SCHEMA_VERSION = 1
BUNDLE_SCHEMA_VERSION = 1
TREATMENT_SCHEMA_VERSION = 1
SAMPLING_ALGORITHM = "independent-torch-randperm-per-table"
DEVICE_POLICY = "sample-on-cpu-then-transfer"
SEED_POLICY = "sha256-domain-separated-base-seed-and-rank-v1"


def _require_int(name: str, value: object, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _state_sha256(state: Tensor) -> str:
    state = state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return hashlib.sha256(state.numpy().tobytes()).hexdigest()


def _manifest_sha256(manifest: dict[str, object]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _derive_rank_seed(base_seed: int, rank: int) -> int:
    payload = f"{SAMPLER_VERSION}\0base_seed={base_seed}\0rank={rank}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


class IdentityPermutationSampler:
    """Sample per-table permutations without advancing a process-global RNG."""

    def __init__(self, *, base_seed: int, rank: int, world_size: int) -> None:
        self.base_seed = _require_int("base_seed", base_seed, minimum=0, maximum=(1 << 63) - 1)
        self.rank = _require_int("rank", rank, minimum=0)
        self.world_size = _require_int("world_size", world_size, minimum=1)
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")
        self.derived_seed = _derive_rank_seed(self.base_seed, self.rank)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.derived_seed)
        self.draw_count = 0

    def sample(self, *, batch_size: int, num_features: int, device: torch.device | str) -> Tensor:
        batch_size = _require_int("batch_size", batch_size, minimum=1)
        num_features = _require_int("num_features", num_features, minimum=1)
        if num_features == 1:
            permutations = torch.zeros((batch_size, 1), dtype=torch.long)
        else:
            permutations = torch.stack(
                [torch.randperm(num_features, generator=self._generator) for _ in range(batch_size)]
            )
        self.draw_count += 1
        return permutations.to(device=device)

    def state_dict(self) -> dict[str, object]:
        generator_state = self._generator.get_state().clone()
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "sampler_version": SAMPLER_VERSION,
            "algorithm": SAMPLING_ALGORITHM,
            "device_policy": DEVICE_POLICY,
            "torch_version": str(torch.__version__),
            "base_seed": self.base_seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "derived_seed": self.derived_seed,
            "draw_count": self.draw_count,
            "generator_state": generator_state,
            "generator_state_sha256": _state_sha256(generator_state),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        expected = {
            "schema_version": STATE_SCHEMA_VERSION,
            "sampler_version": SAMPLER_VERSION,
            "algorithm": SAMPLING_ALGORITHM,
            "device_policy": DEVICE_POLICY,
            "torch_version": str(torch.__version__),
            "base_seed": self.base_seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "derived_seed": self.derived_seed,
        }
        for field, expected_value in expected.items():
            if state.get(field) != expected_value:
                raise ValueError(
                    f"identity sampler {field} mismatch: expected {expected_value!r}, "
                    f"got {state.get(field)!r}"
                )
        draw_count = _require_int("draw_count", state.get("draw_count"), minimum=0)
        generator_state = state.get("generator_state")
        if not isinstance(generator_state, Tensor):
            raise TypeError("identity sampler generator_state must be a tensor")
        generator_state = generator_state.detach().cpu().to(torch.uint8)
        actual_hash = _state_sha256(generator_state)
        if actual_hash != state.get("generator_state_sha256"):
            raise ValueError("identity sampler generator_state hash mismatch")
        self._generator.set_state(generator_state)
        self.draw_count = draw_count

    def audit_record(self) -> dict[str, object]:
        record = self.state_dict()
        del record["generator_state"]
        return record


def _bundle_manifest(rank_states: dict[str, dict[str, object]]) -> dict[str, object]:
    first = rank_states["0"]
    records = []
    for rank_key in sorted(rank_states, key=int):
        records.append({key: value for key, value in rank_states[rank_key].items() if key != "generator_state"})
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "sampler_version": SAMPLER_VERSION,
        "base_seed": first["base_seed"],
        "world_size": first["world_size"],
        "rank_states": records,
    }


def build_checkpoint_bundle(states: list[dict[str, object]]) -> dict[str, object]:
    """Validate and combine every rank's identity stream state."""
    if not states:
        raise ValueError("identity sampler states cannot be empty")
    validated: dict[str, dict[str, object]] = {}
    for state in states:
        sampler = IdentityPermutationSampler(
            base_seed=int(state["base_seed"]),
            rank=int(state["rank"]),
            world_size=int(state["world_size"]),
        )
        sampler.load_state_dict(state)
        rank_key = str(sampler.rank)
        if rank_key in validated:
            raise ValueError(f"duplicate identity sampler state for rank {sampler.rank}")
        validated[rank_key] = state

    first = states[0]
    world_size = int(first["world_size"])
    expected_ranks = {str(rank) for rank in range(world_size)}
    if set(validated) != expected_ranks:
        raise ValueError(
            "identity sampler checkpoint must contain every rank exactly once: "
            f"expected {sorted(expected_ranks)}, got {sorted(validated)}"
        )
    for state in states:
        if state["base_seed"] != first["base_seed"]:
            raise ValueError("identity sampler base_seed differs across ranks")
        if state["world_size"] != world_size:
            raise ValueError("identity sampler world_size differs across ranks")
        if state["draw_count"] != first["draw_count"]:
            raise ValueError("identity sampler draw_count differs across ranks")

    rank_states = {rank: validated[rank] for rank in sorted(validated, key=int)}
    manifest = _bundle_manifest(rank_states)
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "sampler_version": SAMPLER_VERSION,
        "base_seed": first["base_seed"],
        "world_size": world_size,
        "rank_states": rank_states,
        "manifest_sha256": _manifest_sha256(manifest),
    }


def restore_sampler_from_bundle(sampler: IdentityPermutationSampler, bundle: dict[str, object]) -> None:
    expected = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "sampler_version": SAMPLER_VERSION,
        "base_seed": sampler.base_seed,
        "world_size": sampler.world_size,
    }
    for field, expected_value in expected.items():
        if bundle.get(field) != expected_value:
            raise ValueError(
                f"identity sampler bundle {field} mismatch: expected {expected_value!r}, "
                f"got {bundle.get(field)!r}"
            )
    rank_states = bundle.get("rank_states")
    if not isinstance(rank_states, dict):
        raise ValueError("identity sampler bundle rank_states is missing")
    normalized = build_checkpoint_bundle(list(rank_states.values()))
    if normalized["manifest_sha256"] != bundle.get("manifest_sha256"):
        raise ValueError("identity sampler bundle manifest_sha256 mismatch")
    sampler.load_state_dict(rank_states[str(sampler.rank)])


def gather_checkpoint_bundle(sampler: IdentityPermutationSampler) -> dict[str, object]:
    """Collect all rank-local sampler states on every rank."""
    if sampler.world_size == 1 and not dist.is_initialized():
        return build_checkpoint_bundle([sampler.state_dict()])
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("identity sampler world_size > 1 requires an initialized distributed process group")
    if dist.get_rank() != sampler.rank or dist.get_world_size() != sampler.world_size:
        raise RuntimeError("distributed process group does not match identity sampler rank/world_size")
    gathered: list[dict[str, object] | None] = [None] * sampler.world_size
    dist.all_gather_object(gathered, sampler.state_dict())
    if any(state is None for state in gathered):
        raise RuntimeError("distributed identity sampler state gather returned an empty rank")
    return build_checkpoint_bundle(gathered)  # type: ignore[arg-type]


def make_identity_treatment(*, mode: str, seed: int, world_size: int) -> dict[str, object]:
    controller = TrainerIdentityRNG(identity_mode=mode, base_seed=seed, rank=0, world_size=world_size)
    return controller.treatment_manifest()


def validate_identity_treatment(
    treatment: dict[str, object], *, mode: str, seed: int, world_size: int
) -> None:
    controller = TrainerIdentityRNG(identity_mode=mode, base_seed=seed, rank=0, world_size=world_size)
    controller._validate_treatment(treatment, only_load_model=False)


class TrainerIdentityRNG:
    """Training adapter for explicit identity sampling and checkpoint restore."""

    def __init__(
        self,
        *,
        identity_mode: str | None = None,
        base_seed: int | None = None,
        rank: int,
        world_size: int,
        mode: str | None = None,
        seed: int | None = None,
    ) -> None:
        identity_mode = identity_mode if identity_mode is not None else mode
        base_seed = base_seed if base_seed is not None else seed
        if identity_mode not in {"rope", "temporary", "none"}:
            raise ValueError("identity_mode must be one of 'rope', 'temporary', or 'none'")
        if base_seed is None:
            raise TypeError("base_seed must be provided")
        self.identity_mode = identity_mode
        self.base_seed = _require_int("base_seed", base_seed, minimum=0, maximum=(1 << 63) - 1)
        self.rank = _require_int("rank", rank, minimum=0)
        self.world_size = _require_int("world_size", world_size, minimum=1)
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")
        self.sampler = (
            IdentityPermutationSampler(
                base_seed=self.base_seed,
                rank=self.rank,
                world_size=self.world_size,
            )
            if self.identity_mode == "temporary"
            else None
        )

    def sample_for_micro_batch(
        self, *, batch_size: int, num_identity_tokens: int, device: torch.device | str
    ) -> Tensor | None:
        if self.sampler is None:
            return None
        return self.sampler.sample(
            batch_size=batch_size,
            num_features=num_identity_tokens,
            device=device,
        )

    def sample_permutations(
        self, batch_size: int, num_features: int, device: torch.device | str = "cpu"
    ) -> Tensor | None:
        return self.sample_for_micro_batch(
            batch_size=batch_size,
            num_identity_tokens=num_features,
            device=device,
        )

    def state_dict(self) -> dict[str, object]:
        if self.sampler is None:
            raise RuntimeError("non-temporary identity treatments have no sampler state")
        return self.sampler.state_dict()

    def load_state_dict(self, state: dict[str, object]) -> None:
        if self.sampler is None:
            raise RuntimeError("non-temporary identity treatments have no sampler state")
        self.sampler.load_state_dict(state)

    def treatment_manifest(self) -> dict[str, object]:
        manifest: dict[str, object] = {
            "schema_version": TREATMENT_SCHEMA_VERSION,
            "row_identity_mode": self.identity_mode,
            "identity_rng_seed": self.base_seed,
            "seed_policy": SEED_POLICY,
            "sampler_version": SAMPLER_VERSION if self.sampler is not None else None,
            "world_size": self.world_size,
        }
        return {**manifest, "manifest_sha256": _manifest_sha256(manifest)}

    def _validate_treatment(self, treatment: dict[str, object], *, only_load_model: bool) -> None:
        payload = {field: value for field, value in treatment.items() if field != "manifest_sha256"}
        if _manifest_sha256(payload) != treatment.get("manifest_sha256"):
            raise ValueError("identity_treatment manifest hash mismatch")
        expected = self.treatment_manifest()
        locked = ("schema_version", "row_identity_mode", "seed_policy", "sampler_version")
        fields = locked if only_load_model else tuple(expected)
        for field in fields:
            if treatment.get(field) != expected[field]:
                raise ValueError(
                    f"identity_treatment {field} mismatch: expected {expected[field]!r}, "
                    f"got {treatment.get(field)!r}"
                )

    def checkpoint_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {"identity_treatment": self.treatment_manifest()}
        if self.sampler is not None:
            fields["identity_sampler"] = gather_checkpoint_bundle(self.sampler)
        return fields

    def restore_checkpoint(self, checkpoint: dict[str, Any], *, only_load_model: bool) -> None:
        treatment = checkpoint.get("identity_treatment")
        if not isinstance(treatment, dict):
            raise ValueError("checkpoint is missing identity_treatment manifest")
        self._validate_treatment(treatment, only_load_model=only_load_model)
        if only_load_model:
            return
        if self.sampler is None:
            if "identity_sampler" in checkpoint:
                raise ValueError("non-temporary checkpoint must not contain identity_sampler state")
            return
        sampler_bundle = checkpoint.get("identity_sampler")
        if not isinstance(sampler_bundle, dict):
            raise ValueError("full temporary-identity resume requires identity_sampler state")
        restore_sampler_from_bundle(self.sampler, sampler_bundle)
