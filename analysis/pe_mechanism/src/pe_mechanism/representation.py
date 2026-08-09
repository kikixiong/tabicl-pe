"""Small, deterministic autoencoders for activation analysis.

This module deliberately has no dependency on a cluster layout or model-specific
code.  Activation collectors can write a two-dimensional NumPy array and use the
functions here directly, or invoke :func:`run` through the package CLI.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .adapters.base import ACTIVATION_VECTOR_AXIS_NAMES
from .identifiers import require_portable_identifier, require_public_label
from .manifest import ArtifactDigest, InputDigest, RunManifest
from .provenance import (
    RunTransaction,
    VerifiedFile,
    assert_dataset_roster,
    assert_git_commit_is_ancestor,
    load_verified_json_config,
    load_verified_run_manifest,
    manifest_from_verified_inputs,
    verify_configured_run_inputs,
    verify_file,
    verify_run_directory,
    verified_dataset_roster,
)

SUPPORTED_TOP_K = (16, 32, 64)
MIN_HELDOUT_EXPLAINED_VARIANCE = 0.95
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_HEX = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_TALENT_INPUT_ROLE = re.compile(r"^talent\.([0-9]{4})\.([A-Za-z0-9_]+)$")
_OFFICIAL_REQUIRED_LOGICAL_INPUTS = {
    "info.json",
    "y_train.npy",
    "y_val.npy",
    "y_test.npy",
}
_OFFICIAL_ALLOWED_LOGICAL_INPUTS = {
    "info.json",
    *(f"{prefix}_{split}.npy" for prefix in ("N", "C", "y") for split in ("train", "val", "test")),
}


def set_deterministic_seed(seed: int) -> None:
    """Seed all RNGs used here and request deterministic torch kernels."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Evidence runs fail instead of silently falling back to a nondeterministic
    # kernel.  A caller that needs an unsupported accelerator path must change
    # the method and provenance explicitly, not receive a warning-only run.
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class MeanRMSNormalizer(nn.Module):
    """Feature-wise centering followed by root-mean-square scaling."""

    def __init__(self, mean: Tensor, rms: Tensor, *, eps: float = 1e-6) -> None:
        super().__init__()
        mean = torch.as_tensor(mean).detach().to(dtype=torch.float32)
        rms = torch.as_tensor(rms).detach().to(dtype=torch.float32)
        if mean.ndim != 1 or rms.shape != mean.shape:
            raise ValueError("mean and rms must be one-dimensional tensors with equal shape")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if not torch.isfinite(mean).all() or not torch.isfinite(rms).all():
            raise ValueError("normalizer statistics must be finite")
        if (rms <= 0).any():
            raise ValueError("all rms values must be positive")
        self.register_buffer("mean", mean)
        self.register_buffer("rms", rms)
        self.eps = float(eps)

    @classmethod
    def fit(cls, values: Tensor | np.ndarray, *, eps: float = 1e-6) -> "MeanRMSNormalizer":
        tensor = _as_activation_tensor(values)
        mean = tensor.mean(dim=0)
        rms = (tensor - mean).square().mean(dim=0).sqrt()
        # Constant coordinates should remain zero after normalization instead of
        # being amplified by a tiny divisor.
        rms = torch.where(rms > eps, rms, torch.ones_like(rms))
        return cls(mean, rms, eps=eps)

    def normalize(self, values: Tensor) -> Tensor:
        return (values - self.mean) / self.rms

    def denormalize(self, values: Tensor) -> Tensor:
        return values * self.rms + self.mean

    def forward(self, values: Tensor) -> Tensor:
        return self.normalize(values)


def _activation(name: str) -> nn.Module:
    normalized = name.lower().replace("-", "_")
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized in {"linear", "identity", "none"}:
        return nn.Identity()
    raise ValueError(f"unsupported activation: {name!r}")


class DenseAutoencoder(nn.Module):
    """A conventional bottleneck autoencoder used as a reconstruction baseline."""

    model_type = "dense"

    def __init__(self, input_dim: int, latent_dim: int, *, activation: str = "relu") -> None:
        super().__init__()
        if input_dim <= 0 or latent_dim <= 0:
            raise ValueError("input_dim and latent_dim must be positive")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.activation_name = activation
        self.encoder = nn.Linear(self.input_dim, self.latent_dim)
        self.activation = _activation(activation)
        self.decoder = nn.Linear(self.latent_dim, self.input_dim)

    def encode(self, values: Tensor) -> Tensor:
        return self.activation(self.encoder(values))

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        latents = self.encode(values)
        return self.decode(latents), latents

    def configuration(self) -> dict[str, Any]:
        return {
            "type": self.model_type,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "activation": self.activation_name,
        }


class TopKSparseAutoencoder(nn.Module):
    """An over-complete autoencoder retaining only the largest positive features."""

    model_type = "topk"

    def __init__(
        self,
        input_dim: int,
        *,
        expansion_factor: int = 8,
        top_k: int | None = None,
        latent_dim: int | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or expansion_factor <= 0:
            raise ValueError("input_dim and expansion_factor must be positive")
        resolved_latent_dim = int(latent_dim or (input_dim * expansion_factor))
        resolved_top_k = min(32, resolved_latent_dim) if top_k is None else int(top_k)
        if not 0 < resolved_top_k <= resolved_latent_dim:
            raise ValueError("top_k must be positive and no larger than latent_dim")
        self.input_dim = int(input_dim)
        self.expansion_factor = int(expansion_factor)
        self.latent_dim = resolved_latent_dim
        self.top_k = resolved_top_k
        self.encoder = nn.Linear(self.input_dim, self.latent_dim)
        self.decoder = nn.Linear(self.latent_dim, self.input_dim)

    def encode(self, values: Tensor) -> Tensor:
        positive = torch.relu(self.encoder(values))
        selected_values, selected_indices = positive.topk(self.top_k, dim=-1, sorted=False)
        latents = torch.zeros_like(positive)
        return latents.scatter(-1, selected_indices, selected_values)

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        latents = self.encode(values)
        return self.decode(latents), latents

    def configuration(self) -> dict[str, Any]:
        return {
            "type": self.model_type,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "expansion_factor": self.expansion_factor,
            "top_k": self.top_k,
        }


class PCARepresentation(nn.Module):
    """A deterministic, full-SVD PCA reconstruction baseline.

    PCA is fitted on the normalized training activations only.  The component
    signs are canonicalized after the SVD so repeated fits do not differ merely
    by the otherwise arbitrary sign of a singular vector.
    """

    model_type = "pca"
    solver = "full_svd"

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or latent_dim <= 0:
            raise ValueError("input_dim and latent_dim must be positive")
        if latent_dim > input_dim:
            raise ValueError("PCA latent_dim must not exceed input_dim")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.register_buffer("mean", torch.zeros(self.input_dim, dtype=torch.float32))
        self.register_buffer(
            "components",
            torch.zeros(self.latent_dim, self.input_dim, dtype=torch.float32),
        )
        self.register_buffer("is_fitted", torch.tensor(False, dtype=torch.bool))

    def fit(self, values: Tensor | np.ndarray) -> "PCARepresentation":
        """Fit exact principal components on a CPU activation matrix."""

        tensor = _as_activation_tensor(values)
        if tensor.shape[1] != self.input_dim:
            raise ValueError("PCA fit activations do not match input_dim")
        if tensor.shape[0] < 2:
            raise ValueError("PCA requires at least two training rows")
        rank_bound = min(tensor.shape[0] - 1, tensor.shape[1])
        if self.latent_dim > rank_bound:
            raise ValueError("PCA latent_dim exceeds the available training rank bound")

        # Float64 CPU SVD is used intentionally: this is an exact baseline, not
        # a randomized low-rank approximation whose RNG/version would need
        # separate provenance and reproducibility gates.
        fit_values = tensor.to(dtype=torch.float64, device="cpu")
        mean = fit_values.mean(dim=0)
        centered = fit_values - mean
        _, _, right_vectors = torch.linalg.svd(centered, full_matrices=False)
        components = right_vectors[: self.latent_dim].contiguous()

        # Each singular vector is equivalent up to sign.  Make the coordinate
        # with largest magnitude positive to fix that otherwise arbitrary bit.
        pivots = components.abs().argmax(dim=1, keepdim=True)
        pivot_values = components.gather(1, pivots)
        signs = torch.where(pivot_values < 0, -torch.ones_like(pivot_values), 1.0)
        components = components * signs

        self.mean.copy_(mean.to(dtype=self.mean.dtype))
        self.components.copy_(components.to(dtype=self.components.dtype))
        self.is_fitted.fill_(True)
        return self

    def _require_fitted(self) -> None:
        if not bool(self.is_fitted.item()):
            raise RuntimeError("PCA representation must be fitted before encode/decode")

    def encode(self, values: Tensor) -> Tensor:
        self._require_fitted()
        return (values - self.mean) @ self.components.transpose(0, 1)

    def decode(self, latents: Tensor) -> Tensor:
        self._require_fitted()
        return latents @ self.components + self.mean

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        latents = self.encode(values)
        return self.decode(latents), latents

    def configuration(self) -> dict[str, Any]:
        return {
            "type": self.model_type,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "solver": self.solver,
        }


def build_autoencoder(input_dim: int, config: Mapping[str, Any]) -> nn.Module:
    """Construct an autoencoder from its JSON-compatible configuration."""

    model_type = str(config.get("type", config.get("kind", "dense"))).lower().replace("-", "")
    if model_type in {"dense", "ae", "autoencoder"}:
        latent_dim = int(config.get("latent_dim", max(1, input_dim // 2)))
        return DenseAutoencoder(
            input_dim,
            latent_dim,
            activation=str(config.get("activation", "relu")),
        )
    if model_type in {"topk", "top_k", "sparse", "sae"}:
        expansion_factor = int(config.get("expansion_factor", 8))
        top_k_value = config.get("top_k", config.get("k"))
        if top_k_value is not None and int(top_k_value) not in SUPPORTED_TOP_K:
            raise ValueError(f"top_k must be one of {SUPPORTED_TOP_K}")
        latent_dim_value = config.get("latent_dim")
        return TopKSparseAutoencoder(
            input_dim,
            expansion_factor=expansion_factor,
            top_k=None if top_k_value is None else int(top_k_value),
            latent_dim=None if latent_dim_value is None else int(latent_dim_value),
        )
    if model_type in {"pca", "principalcomponents", "principal_component_analysis"}:
        solver = str(config.get("solver", PCARepresentation.solver)).lower().replace("-", "_")
        if solver != PCARepresentation.solver:
            raise ValueError("PCA solver must be 'full_svd' for deterministic evidence runs")
        latent_dim = int(config.get("latent_dim", max(1, input_dim // 2)))
        return PCARepresentation(input_dim, latent_dim)
    raise ValueError(f"unsupported autoencoder type: {model_type!r}")


@dataclass(frozen=True)
class BalancedConditionActivations:
    """Separate, condition-balanced training and validation activation pools."""

    training: Tensor
    validation: Tensor
    condition_names: tuple[str, ...]
    training_condition_indices: Tensor
    validation_condition_indices: Tensor
    training_rows_per_condition: int
    validation_rows_per_condition: int
    training_dataset_roster: tuple[str, ...]
    validation_dataset_roster: tuple[str, ...]
    training_rows_by_dataset: tuple[tuple[str, int], ...]
    validation_rows_by_dataset: tuple[tuple[str, int], ...]
    training_selection_sha256: str
    validation_selection_sha256: str


def balance_condition_activations(
    training_by_condition: Mapping[str, Any],
    validation_by_condition: Mapping[str, Any],
    *,
    seed: int = 42,
    training_rows_per_condition: int | None = None,
    validation_rows_per_condition: int | None = None,
    training_dataset_roster: Sequence[str] | None = None,
    validation_dataset_roster: Sequence[str] | None = None,
) -> BalancedConditionActivations:
    """Create deterministic, equally sized condition pools without split mixing.

    The strict input form is ``condition -> dataset_id -> activations``.  Every
    condition must contain exactly the same dataset roster within a split, and
    corresponding dataset arrays must have identical shapes.  A pre-pooled
    ``condition -> activations`` shorthand is accepted only when both split
    rosters are supplied explicitly.  Training and validation dataset IDs must
    always be disjoint.
    """

    resolved_seed = _strict_nonnegative_integer(seed, name="seed")
    training_conditions = _condition_roster(
        training_by_condition, name="training_by_condition"
    )
    validation_conditions = _condition_roster(
        validation_by_condition, name="validation_by_condition"
    )
    if training_conditions != validation_conditions:
        raise ValueError(
            "training and validation condition rosters must match exactly"
        )

    training_is_nested = _condition_values_are_nested(
        training_by_condition,
        training_conditions,
        name="training_by_condition",
    )
    validation_is_nested = _condition_values_are_nested(
        validation_by_condition,
        validation_conditions,
        name="validation_by_condition",
    )
    if training_is_nested != validation_is_nested:
        raise ValueError(
            "training and validation condition activations must use the same input form"
        )

    if training_is_nested:
        training_groups, inferred_training_roster, training_sources = (
            _resolve_nested_condition_groups(
                training_by_condition,
                training_conditions,
                split_name="training",
            )
        )
        validation_groups, inferred_validation_roster, validation_sources = (
            _resolve_nested_condition_groups(
                validation_by_condition,
                validation_conditions,
                split_name="validation",
            )
        )
        resolved_training_roster = _match_optional_dataset_roster(
            training_dataset_roster,
            inferred_training_roster,
            name="training_dataset_roster",
        )
        resolved_validation_roster = _match_optional_dataset_roster(
            validation_dataset_roster,
            inferred_validation_roster,
            name="validation_dataset_roster",
        )
    else:
        if training_dataset_roster is None or validation_dataset_roster is None:
            raise ValueError(
                "pooled condition activations require explicit training and validation "
                "dataset rosters"
            )
        resolved_training_roster = _dataset_roster(
            training_dataset_roster, name="training_dataset_roster"
        )
        resolved_validation_roster = _dataset_roster(
            validation_dataset_roster, name="validation_dataset_roster"
        )
        if len(resolved_training_roster) != 1 or len(resolved_validation_roster) != 1:
            raise ValueError(
                "pooled condition activations can represent exactly one dataset per split"
            )
        training_groups, training_sources = _resolve_pooled_condition_groups(
            training_by_condition,
            training_conditions,
            dataset_id=resolved_training_roster[0],
        )
        validation_groups, validation_sources = _resolve_pooled_condition_groups(
            validation_by_condition,
            validation_conditions,
            dataset_id=resolved_validation_roster[0],
        )

    overlap = sorted(set(resolved_training_roster) & set(resolved_validation_roster))
    if overlap:
        raise ValueError(
            f"training and validation dataset rosters must be disjoint; overlap: {overlap}"
        )
    for training_source in training_sources:
        for validation_source in validation_sources:
            if _shares_activation_memory(training_source, validation_source):
                raise ValueError(
                    "training and validation activations must not share source memory"
                )

    widths = {
        int(group.shape[1])
        for conditions in (training_groups, validation_groups)
        for datasets in conditions.values()
        for group in datasets.values()
    }
    if len(widths) != 1:
        raise ValueError("all condition activations must have the same feature width")

    resolved_training_rows, training_allocations = _rows_per_dataset(
        training_rows_per_condition,
        training_groups,
        resolved_training_roster,
        name="training_rows_per_condition",
    )
    resolved_validation_rows, validation_allocations = _rows_per_dataset(
        validation_rows_per_condition,
        validation_groups,
        resolved_validation_roster,
        name="validation_rows_per_condition",
    )
    training, training_indices, training_selection_sha256 = (
        _sample_aligned_condition_datasets(
            training_groups,
            training_conditions,
            resolved_training_roster,
            rows_by_dataset=training_allocations,
            seed=resolved_seed,
            split_number=0,
            split_name="training",
        )
    )
    validation, validation_indices, validation_selection_sha256 = (
        _sample_aligned_condition_datasets(
            validation_groups,
            validation_conditions,
            resolved_validation_roster,
            rows_by_dataset=validation_allocations,
            seed=resolved_seed,
            split_number=1,
            split_name="validation",
        )
    )
    return BalancedConditionActivations(
        training=training,
        validation=validation,
        condition_names=training_conditions,
        training_condition_indices=training_indices,
        validation_condition_indices=validation_indices,
        training_rows_per_condition=resolved_training_rows,
        validation_rows_per_condition=resolved_validation_rows,
        training_dataset_roster=resolved_training_roster,
        validation_dataset_roster=resolved_validation_roster,
        training_rows_by_dataset=tuple(training_allocations.items()),
        validation_rows_by_dataset=tuple(validation_allocations.items()),
        training_selection_sha256=training_selection_sha256,
        validation_selection_sha256=validation_selection_sha256,
    )


def _condition_roster(values: Mapping[str, Any], *, name: str) -> tuple[str, ...]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{name} must be a non-empty mapping")
    names = tuple(values.keys())
    if any(not isinstance(value, str) or not value.strip() for value in names):
        raise ValueError(f"{name} condition names must be non-empty strings")
    return tuple(sorted(names))


def _condition_values_are_nested(
    values: Mapping[str, Any],
    conditions: Sequence[str],
    *,
    name: str,
) -> bool:
    nested = tuple(isinstance(values[condition], Mapping) for condition in conditions)
    if any(nested) and not all(nested):
        raise ValueError(f"{name} must not mix pooled and per-dataset condition values")
    return all(nested)


def _dataset_roster(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError(f"{name} must be a non-empty sequence")
    roster = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in roster):
        raise ValueError(f"{name} entries must be non-empty strings")
    if len(set(roster)) != len(roster):
        raise ValueError(f"{name} must not contain duplicate dataset IDs")
    return tuple(sorted(roster))


def _match_optional_dataset_roster(
    supplied: Sequence[str] | None,
    inferred: tuple[str, ...],
    *,
    name: str,
) -> tuple[str, ...]:
    if supplied is None:
        return inferred
    resolved = _dataset_roster(supplied, name=name)
    if resolved != inferred:
        raise ValueError(f"{name} does not match the per-condition dataset roster")
    return resolved


def _resolve_nested_condition_groups(
    values: Mapping[str, Any],
    conditions: Sequence[str],
    *,
    split_name: str,
) -> tuple[dict[str, dict[str, Tensor]], tuple[str, ...], list[Any]]:
    groups: dict[str, dict[str, Tensor]] = {}
    sources: list[Any] = []
    expected_roster: tuple[str, ...] | None = None
    expected_shapes: dict[str, tuple[int, ...]] = {}
    for condition in conditions:
        datasets = values[condition]
        if not isinstance(datasets, Mapping) or not datasets:
            raise ValueError(
                f"{split_name} condition {condition!r} must have a non-empty dataset mapping"
            )
        roster = _condition_roster(datasets, name=f"{split_name}[{condition!r}]")
        if expected_roster is None:
            expected_roster = roster
        elif roster != expected_roster:
            raise ValueError(
                f"{split_name} dataset roster must match exactly across conditions"
            )

        groups[condition] = {}
        for dataset_id in roster:
            source = datasets[dataset_id]
            tensor = _as_activation_tensor(source)
            shape = tuple(tensor.shape)
            if condition == conditions[0]:
                expected_shapes[dataset_id] = shape
            elif shape != expected_shapes[dataset_id]:
                raise ValueError(
                    f"{split_name} dataset {dataset_id!r} activation shape must match "
                    "across conditions"
                )
            sources.append(source)
            groups[condition][dataset_id] = tensor
    assert expected_roster is not None
    return groups, expected_roster, sources


def _resolve_pooled_condition_groups(
    values: Mapping[str, Any], conditions: Sequence[str], *, dataset_id: str
) -> tuple[dict[str, dict[str, Tensor]], list[Any]]:
    groups: dict[str, dict[str, Tensor]] = {}
    sources: list[Any] = []
    for condition in conditions:
        source = values[condition]
        sources.append(source)
        groups[condition] = {dataset_id: _as_activation_tensor(source)}
    return groups, sources


def _shares_activation_memory(left: Any, right: Any) -> bool:
    if left is right:
        return True
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.shares_memory(left, right))
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        if left.device.type == "cpu" and right.device.type == "cpu":
            try:
                return bool(
                    np.shares_memory(left.detach().numpy(), right.detach().numpy())
                )
            except (RuntimeError, TypeError):
                # NumPy cannot expose every torch dtype (for example bfloat16).
                # Sharing a backing storage is conservatively treated as split
                # reuse even if two unusual views might not overlap byte-for-byte.
                pass
        return bool(
            left.device == right.device
            and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        )
    if isinstance(left, Tensor) and isinstance(right, np.ndarray):
        if left.device.type != "cpu":
            return False
        try:
            return bool(np.shares_memory(left.detach().numpy(), right))
        except (RuntimeError, TypeError):
            return False
    if isinstance(left, np.ndarray) and isinstance(right, Tensor):
        return _shares_activation_memory(right, left)
    return False


def _strict_nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{name} must be non-negative")
    return resolved


def _rows_per_dataset(
    requested: int | None,
    groups: Mapping[str, Mapping[str, Tensor]],
    roster: Sequence[str],
    *,
    name: str,
) -> tuple[int, dict[str, int]]:
    first_condition = sorted(groups)[0]
    available_by_dataset = {
        dataset: int(groups[first_condition][dataset].shape[0]) for dataset in roster
    }
    if requested is None:
        # Equal dataset contribution is the strict default.  Large datasets do
        # not silently dominate the learned representation.
        rows_each = min(available_by_dataset.values())
        allocations = {dataset: rows_each for dataset in roster}
        return rows_each * len(roster), allocations
    resolved = _strict_nonnegative_integer(requested, name=name)
    if resolved == 0:
        raise ValueError(f"{name} must be positive")
    if resolved > sum(available_by_dataset.values()):
        raise ValueError(f"{name} exceeds the available per-dataset activation rows")

    low, high = 0, max(available_by_dataset.values())
    while low < high:
        midpoint = (low + high + 1) // 2
        if sum(min(count, midpoint) for count in available_by_dataset.values()) <= resolved:
            low = midpoint
        else:
            high = midpoint - 1
    allocations = {
        dataset: min(available_by_dataset[dataset], low) for dataset in roster
    }
    remaining = resolved - sum(allocations.values())
    for dataset in roster:
        if remaining == 0:
            break
        if allocations[dataset] < available_by_dataset[dataset]:
            allocations[dataset] += 1
            remaining -= 1
    if remaining:
        raise AssertionError("balanced per-dataset allocation did not reach the requested total")
    return resolved, allocations


def _sample_aligned_condition_datasets(
    groups: Mapping[str, Mapping[str, Tensor]],
    conditions: Sequence[str],
    roster: Sequence[str],
    *,
    rows_by_dataset: Mapping[str, int],
    seed: int,
    split_number: int,
    split_name: str,
) -> tuple[Tensor, Tensor, str]:
    selected_by_dataset: dict[str, Tensor] = {}
    selection_record: dict[str, list[int]] = {}
    reference_condition = conditions[0]
    for dataset_index, dataset in enumerate(roster):
        row_count = int(groups[reference_condition][dataset].shape[0])
        requested = int(rows_by_dataset[dataset])
        if requested == row_count:
            selected = torch.arange(row_count, dtype=torch.long)
        else:
            generator = np.random.default_rng(
                np.random.SeedSequence(
                    entropy=seed, spawn_key=(split_number, dataset_index)
                )
            )
            chosen = generator.choice(row_count, size=requested, replace=False)
            selected = torch.from_numpy(np.sort(chosen).astype(np.int64, copy=False))
        selected_by_dataset[dataset] = selected
        selection_record[dataset] = selected.tolist()

    sampled: list[Tensor] = []
    condition_indices: list[Tensor] = []
    for condition_index, condition in enumerate(conditions):
        condition_blocks = [
            groups[condition][dataset].index_select(0, selected_by_dataset[dataset])
            for dataset in roster
        ]
        condition_pool = torch.cat(condition_blocks, dim=0).contiguous()
        sampled.append(condition_pool)
        condition_indices.append(
            torch.full((condition_pool.shape[0],), condition_index, dtype=torch.long)
        )
    selection_bytes = json.dumps(
        {"split": split_name, "selected_rows": selection_record},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        torch.cat(sampled, dim=0).contiguous(),
        torch.cat(condition_indices, dim=0),
        hashlib.sha256(selection_bytes).hexdigest(),
    )


def reconstruction_metrics(
    inputs: Tensor | np.ndarray,
    reconstructions: Tensor | np.ndarray,
    latents: Tensor | np.ndarray,
    *,
    active_threshold: float = 1e-8,
    eps: float = 1e-12,
) -> dict[str, float | int]:
    """Return reconstruction quality and sparse-feature utilization metrics."""

    values = _as_activation_tensor(inputs)
    restored = _as_activation_tensor(reconstructions)
    latent_values = _as_activation_tensor(latents)
    if values.shape != restored.shape:
        raise ValueError("inputs and reconstructions must have equal shape")
    if latent_values.shape[0] != values.shape[0]:
        raise ValueError("latents and inputs must contain the same number of rows")

    residual = values - restored
    mse = residual.square().mean()
    centered = values - values.mean(dim=0, keepdim=True)
    variance = centered.square().mean()
    residual_centered = residual - residual.mean(dim=0, keepdim=True)
    residual_variance = residual_centered.square().mean()
    if float(variance) <= eps:
        normalized_mse = 0.0 if float(mse) <= eps else float(mse / eps)
        explained_variance = 1.0 if float(residual_variance) <= eps else 0.0
    else:
        normalized_mse = float(mse / variance)
        explained_variance = float(1.0 - residual_variance / variance)

    active = latent_values.abs() > active_threshold
    per_feature_active = active.any(dim=0)
    per_row_active = active.sum(dim=1).to(dtype=torch.float32)
    dead_features = int((~per_feature_active).sum().item())
    latent_dim = int(latent_values.shape[1])
    mean_active = float(per_row_active.mean().item())
    return {
        "explained_variance": explained_variance,
        "normalized_mse": normalized_mse,
        "mse": float(mse),
        "dead_features": dead_features,
        "dead_feature_fraction": dead_features / latent_dim,
        "active_count": mean_active,
        "mean_active_features": mean_active,
    }


@dataclass
class RepresentationTrainingResult:
    model: nn.Module
    normalizer: MeanRMSNormalizer
    history: list[float]
    metrics: dict[str, float | int]
    training_metrics: dict[str, float | int]
    metric_split: str
    validation_metrics_by_condition: dict[str, dict[str, float | int]] | None = None


def representation_qualification(
    result_or_metadata: RepresentationTrainingResult | Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the pre-registered held-out activation fidelity gate."""

    if isinstance(result_or_metadata, RepresentationTrainingResult):
        metric_split = result_or_metadata.metric_split
        metrics = result_or_metadata.metrics
        condition_metrics = result_or_metadata.validation_metrics_by_condition or {}
    else:
        metric_split = str(result_or_metadata.get("metric_split", "unknown"))
        metrics = result_or_metadata.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise ValueError("representation metadata metrics must be a mapping")
        raw_condition_metrics = result_or_metadata.get("validation_metrics_by_condition")
        if not isinstance(raw_condition_metrics, Mapping) or not raw_condition_metrics:
            raise ValueError(
                "representation metadata requires non-empty validation metrics by condition"
            )
        condition_metrics: dict[str, Mapping[str, Any]] = {}
        for condition, values in raw_condition_metrics.items():
            portable_condition = require_portable_identifier(
                condition, name="validation condition"
            )
            if not isinstance(values, Mapping) or "explained_variance" not in values:
                raise ValueError(
                    "each validation condition requires reconstruction metrics"
                )
            condition_metrics[portable_condition] = values
        stored_metadata = result_or_metadata.get("metadata")
        if not isinstance(stored_metadata, Mapping):
            raise ValueError("representation metadata lacks source lineage")
        source_lineage = stored_metadata.get("source_lineage")
        if not isinstance(source_lineage, Mapping):
            raise ValueError("representation metadata lacks source lineage")
        checkpoints = source_lineage.get("condition_checkpoints_sha256")
        if not isinstance(checkpoints, Mapping) or set(checkpoints) != set(condition_metrics):
            raise ValueError(
                "validation condition roster differs from source checkpoint lineage"
            )
    explained_variance = float(metrics.get("explained_variance", float("nan")))
    finite = np.isfinite(explained_variance)
    held_out = metric_split == "validation"
    condition_explained_variance = {
        condition: float(values.get("explained_variance", float("nan")))
        for condition, values in sorted(condition_metrics.items())
    }
    condition_gate = all(
        np.isfinite(value) and value >= MIN_HELDOUT_EXPLAINED_VARIANCE
        for value in condition_explained_variance.values()
    )
    passed = bool(
        held_out
        and finite
        and explained_variance >= MIN_HELDOUT_EXPLAINED_VARIANCE
        and condition_gate
    )
    return {
        "metric_split": metric_split,
        "held_out_validation": held_out,
        "explained_variance": explained_variance,
        "minimum_explained_variance": MIN_HELDOUT_EXPLAINED_VARIANCE,
        "activation_fidelity_passed": passed,
        "validation_by_condition": {
            condition: dict(condition_metrics[condition])
            for condition in sorted(condition_metrics)
        },
        "worst_condition_explained_variance": (
            min(condition_explained_variance.values())
            if condition_explained_variance
            else explained_variance
        ),
        # This second, model-level gate is measured only after reconstruction is
        # patched through the native predictor.
        "native_score_gate": "pending",
    }


def train_autoencoder(
    activations: Tensor | np.ndarray,
    *,
    validation_activations: Tensor | np.ndarray | None = None,
    model_config: Mapping[str, Any] | None = None,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> RepresentationTrainingResult:
    """Train an autoencoder deterministically on a two-dimensional activation set."""

    values = _as_activation_tensor(activations)
    validation_values = (
        None if validation_activations is None else _as_activation_tensor(validation_activations)
    )
    if validation_values is not None and validation_values.shape[1] != values.shape[1]:
        raise ValueError("training and validation activations must have the same width")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid training hyperparameters")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    set_deterministic_seed(seed)
    normalizer = MeanRMSNormalizer.fit(values)
    normalized = normalizer.normalize(values).contiguous()
    model = build_autoencoder(values.shape[1], model_config or {})
    normalizer = normalizer.to(resolved_device)
    history: list[float] = []

    if isinstance(model, PCARepresentation):
        # Fitting stays on CPU even when evaluation is requested on CUDA.  That
        # keeps the exact baseline independent of accelerator SVD kernels.
        model.fit(normalized)
        model = model.to(resolved_device)
    else:
        model = model.to(resolved_device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

        model.train()
        row_count = normalized.shape[0]
        for _ in range(int(epochs)):
            permutation = torch.randperm(row_count, generator=generator)
            total_squared_error = 0.0
            total_elements = 0
            for start in range(0, row_count, int(batch_size)):
                indices = permutation[start : start + int(batch_size)]
                batch = normalized.index_select(0, indices).to(resolved_device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction, _ = model(batch)
                loss = torch.mean((reconstruction - batch).square())
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "autoencoder training produced a non-finite loss"
                    )
                loss.backward()
                optimizer.step()
                total_squared_error += float(
                    (reconstruction.detach() - batch).square().sum().cpu()
                )
                total_elements += batch.numel()
            history.append(total_squared_error / total_elements)

    model.eval()
    with torch.no_grad():
        normalized_device = normalized.to(resolved_device)
        reconstruction, latents = model(normalized_device)
        training_metrics = reconstruction_metrics(
            normalized_device.cpu(), reconstruction.cpu(), latents.cpu()
        )
        if validation_values is None:
            metrics = training_metrics
            metric_split = "training"
        else:
            validation_normalized = normalizer.normalize(
                validation_values.to(resolved_device)
            )
            validation_reconstruction, validation_latents = model(validation_normalized)
            metrics = reconstruction_metrics(
                validation_normalized.cpu(),
                validation_reconstruction.cpu(),
                validation_latents.cpu(),
            )
            metric_split = "validation"
    return RepresentationTrainingResult(
        model, normalizer, history, metrics, training_metrics, metric_split
    )


def save_representation_checkpoint(
    path: Path,
    result: RepresentationTrainingResult,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save a representation model and its normalization statistics."""

    model = result.model
    if not hasattr(model, "configuration"):
        raise TypeError("model does not expose a serializable configuration")
    payload = {
        "format_version": 1,
        "model": model.configuration(),
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "normalizer": {
            "mean": result.normalizer.mean.detach().cpu(),
            "rms": result.normalizer.rms.detach().cpu(),
            "eps": result.normalizer.eps,
        },
        "metrics": dict(result.metrics),
        "training_metrics": dict(result.training_metrics),
        "metric_split": result.metric_split,
        "qualification": representation_qualification(result),
        "validation_metrics_by_condition": dict(
            result.validation_metrics_by_condition or {}
        ),
        "history": list(result.history),
        "metadata": dict(metadata or {}),
    }
    _atomic_torch_save(path, payload)


def load_representation_checkpoint(
    path: Path | str, *, map_location: str | torch.device = "cpu"
) -> tuple[nn.Module, MeanRMSNormalizer, dict[str, Any]]:
    """Load a checkpoint created by :func:`save_representation_checkpoint`."""

    checkpoint_path = Path(path)
    return _load_representation_checkpoint_payload(
        checkpoint_path, map_location=map_location
    )


def load_verified_representation_checkpoint(
    verified_file: Any, *, map_location: str | torch.device = "cpu"
) -> tuple[nn.Module, MeanRMSNormalizer, dict[str, Any]]:
    """Load a representation from bytes read through ``VerifiedFile``."""

    return _load_representation_checkpoint_payload(
        io.BytesIO(verified_file.read_bytes()), map_location=map_location
    )


def _load_representation_checkpoint_payload(
    source: Any, *, map_location: str | torch.device
) -> tuple[nn.Module, MeanRMSNormalizer, dict[str, Any]]:
    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        if hasattr(source, "seek"):
            source.seek(0)
        payload = torch.load(source, map_location=map_location)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported representation checkpoint")
    model_config = payload["model"]
    model = build_autoencoder(int(model_config["input_dim"]), model_config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(map_location).eval()
    normalizer_state = payload["normalizer"]
    normalizer = MeanRMSNormalizer(
        normalizer_state["mean"], normalizer_state["rms"], eps=float(normalizer_state["eps"])
    ).to(map_location)
    metadata = {
        "model": dict(model_config),
        "metrics": dict(payload.get("metrics", {})),
        "training_metrics": dict(payload.get("training_metrics", payload.get("metrics", {}))),
        "metric_split": str(payload.get("metric_split", "training")),
        "qualification": dict(payload.get("qualification", {})),
        "validation_metrics_by_condition": dict(
            payload.get("validation_metrics_by_condition", {})
        ),
        "history": list(payload.get("history", [])),
        "metadata": dict(payload.get("metadata", {})),
    }
    return model, normalizer, metadata


def run(args: Any) -> int:
    """Train from a JSON config supplied by the common CLI."""

    config_path = Path(args.config)
    configuration = load_verified_json_config(config_path)
    config = dict(configuration.data)
    training_specs = config.get("training_by_condition")
    validation_specs = config.get("validation_by_condition")
    if not isinstance(training_specs, Mapping) or not isinstance(validation_specs, Mapping):
        raise ValueError(
            "train-repr requires training_by_condition and validation_by_condition mappings"
        )
    _validate_cli_representation_config(config)
    raw_reference_condition = config.get("reference_condition")
    if not isinstance(raw_reference_condition, str):
        raise ValueError("train-repr requires a string reference_condition")
    reference_condition = require_portable_identifier(
        raw_reference_condition, name="reference_condition"
    )
    paths: dict[str, Path] = {}
    expected_hashes: dict[str, str] = {}
    collect_runs: dict[Path, _CollectRunSource] = {}
    training_inputs = _condition_input_specs(
        training_specs,
        split="training",
        paths=paths,
        expected_hashes=expected_hashes,
        collect_runs=collect_runs,
    )
    validation_inputs = _condition_input_specs(
        validation_specs,
        split="validation",
        paths=paths,
        expected_hashes=expected_hashes,
        collect_runs=collect_runs,
    )
    model_config = config["model"]
    training = config["training"]
    assert isinstance(model_config, dict) and isinstance(training, dict)
    seed = int(training["seed"])
    context = verify_configured_run_inputs(
        configuration,
        command="train-repr",
        seed=seed,
        additional_input_paths=paths,
        expected_additional_sha256=expected_hashes,
    )
    all_references = tuple(
        reference
        for split_inputs in (training_inputs, validation_inputs)
        for datasets in split_inputs.values()
        for reference in datasets.values()
    )
    source_lineage = _assert_collect_lineage(
        collect_runs,
        all_references,
        context,
        reference_condition=reference_condition,
    )
    activation_digests = [
        context.additional_file(reference.activation_role).digest.sha256
        for reference in all_references
    ]
    if len(set(activation_digests)) != len(activation_digests):
        raise ValueError(
            "each representation condition/dataset/split requires distinct activation bytes"
        )
    training_by_condition = _load_verified_condition_inputs(training_inputs, context)
    validation_by_condition = _load_verified_condition_inputs(validation_inputs, context)
    sampling = config.get("sampling", {})
    if not isinstance(sampling, Mapping):
        raise ValueError("sampling must be a JSON object")
    balanced = balance_condition_activations(
        training_by_condition,
        validation_by_condition,
        seed=seed,
        training_rows_per_condition=_optional_positive_integer(
            sampling.get("training_rows_per_condition"),
            name="training_rows_per_condition",
        ),
        validation_rows_per_condition=_optional_positive_integer(
            sampling.get("validation_rows_per_condition"),
            name="validation_rows_per_condition",
        ),
    )
    assert_dataset_roster(
        context.inputs.dataset_manifest,
        balanced.training_dataset_roster,
        required_split="discovery",
    )
    assert_dataset_roster(
        context.inputs.dataset_manifest,
        balanced.validation_dataset_roster,
        required_split="validation",
    )

    result = train_autoencoder(
        balanced.training,
        validation_activations=balanced.validation,
        model_config=model_config,
        epochs=int(training.get("epochs", 20)),
        batch_size=int(training.get("batch_size", 256)),
        learning_rate=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        seed=seed,
        device=str(training.get("device", "cpu")),
    )
    result.validation_metrics_by_condition = _validation_metrics_by_condition(
        result,
        balanced.validation,
        balanced.validation_condition_indices,
        balanced.condition_names,
    )
    public_sampling = {
        "condition_names": list(balanced.condition_names),
        "training_dataset_roster": list(balanced.training_dataset_roster),
        "validation_dataset_roster": list(balanced.validation_dataset_roster),
        "training_rows_per_condition": balanced.training_rows_per_condition,
        "validation_rows_per_condition": balanced.validation_rows_per_condition,
        "training_rows_by_dataset": dict(balanced.training_rows_by_dataset),
        "validation_rows_by_dataset": dict(balanced.validation_rows_by_dataset),
        "training_selection_sha256": balanced.training_selection_sha256,
        "validation_selection_sha256": balanced.validation_selection_sha256,
    }
    roots = (
        context.inputs.training_code.root,
        context.inputs.model_code.root,
        context.inputs.analysis_code.root,
    )
    with RunTransaction(Path(args.output_dir), source_roots=roots) as transaction:
        checkpoint_path = transaction.staging_dir / "model.pt"
        metrics_path = transaction.staging_dir / "metrics.json"
        history_path = transaction.staging_dir / "history.json"
        save_representation_checkpoint(
            checkpoint_path,
            result,
            metadata={
                "seed": seed,
                "sampling": public_sampling,
                "source_lineage": source_lineage,
            },
        )
        _atomic_json_write(
            metrics_path,
            {
                "metric_split": result.metric_split,
                "selected_metrics": result.metrics,
                "training_metrics": result.training_metrics,
                "qualification": representation_qualification(result),
                "sampling": public_sampling,
                "source_lineage": source_lineage,
            },
        )
        _atomic_json_write(history_path, {"training_loss": result.history})
        artifacts = transaction.artifact_digests(("history.json", "metrics.json", "model.pt"))
        manifest = manifest_from_verified_inputs(
            context.inputs,
            artifacts=artifacts,
        )
        transaction.commit(manifest, verified_inputs=context.inputs)
    return 0


def _validate_cli_representation_config(config: Mapping[str, Any]) -> None:
    required_root = {
        "reference_condition",
        "training_by_condition",
        "validation_by_condition",
        "model",
        "training",
        "provenance",
    }
    allowed_root = required_root | {"sampling"}
    missing = sorted(required_root - set(config))
    unknown = sorted(set(config) - allowed_root)
    if missing or unknown:
        raise ValueError(
            f"train-repr config fields mismatch: missing={missing}, unknown={unknown}"
        )
    model = config["model"]
    training = config["training"]
    sampling = config.get("sampling", {})
    if not isinstance(model, Mapping) or not isinstance(training, Mapping):
        raise ValueError("model and training entries must be JSON objects")
    if not isinstance(sampling, Mapping):
        raise ValueError("sampling must be a JSON object")

    model_type = model.get("type")
    if model_type == "topk":
        _require_exact_mapping_fields(
            model,
            {"type", "expansion_factor", "top_k"},
            name="topk evidence model",
        )
        if model["expansion_factor"] != 8:
            raise ValueError("topk evidence models require expansion_factor=8")
        if model["top_k"] not in SUPPORTED_TOP_K:
            raise ValueError(f"topk evidence top_k must be one of {SUPPORTED_TOP_K}")
    elif model_type == "dense":
        _require_allowed_mapping_fields(
            model,
            required={"type", "latent_dim"},
            allowed={"type", "latent_dim", "activation"},
            name="dense baseline model",
        )
    elif model_type == "pca":
        _require_allowed_mapping_fields(
            model,
            required={"type", "latent_dim"},
            allowed={"type", "latent_dim", "solver"},
            name="PCA baseline model",
        )
        if model.get("solver", "full_svd") != "full_svd":
            raise ValueError("PCA evidence models require solver='full_svd'")
    else:
        raise ValueError("CLI model type must be exactly topk, dense, or pca")
    latent_dim = model.get("latent_dim")
    if latent_dim is not None and (
        isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0
    ):
        raise ValueError("model latent_dim must be a positive integer")

    _require_allowed_mapping_fields(
        training,
        required={"seed"},
        allowed={
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "seed",
            "device",
        },
        name="representation training",
    )
    seed = training["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in {42, 43, 44}:
        raise ValueError("representation evidence seed must be one of 42, 43, or 44")
    for field in ("epochs", "batch_size"):
        value = training.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"training {field} must be a positive integer")
    for field in ("learning_rate", "weight_decay"):
        value = training.get(field)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or (field == "learning_rate" and value <= 0)
            or (field == "weight_decay" and value < 0)
        ):
            raise ValueError(f"training {field} is invalid")
    device = training.get("device")
    if device is not None and not isinstance(device, str):
        raise ValueError("training device must be a string")

    _require_allowed_mapping_fields(
        sampling,
        required=set(),
        allowed={
            "training_rows_per_condition",
            "validation_rows_per_condition",
        },
        name="representation sampling",
    )
    for field, value in sampling.items():
        _optional_positive_integer(value, name=str(field))


def _require_allowed_mapping_fields(
    value: Mapping[str, Any],
    *,
    required: set[str],
    allowed: set[str],
    name: str,
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - allowed)
    if missing or unknown:
        raise ValueError(f"{name} fields mismatch: missing={missing}, unknown={unknown}")


def _validation_metrics_by_condition(
    result: RepresentationTrainingResult,
    validation: Tensor,
    condition_indices: Tensor,
    condition_names: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    if validation.shape[0] != condition_indices.shape[0]:
        raise ValueError("validation condition indices do not align to activations")
    model_device = next(result.model.buffers(), None)
    if model_device is None:
        first_parameter = next(result.model.parameters(), None)
        device = torch.device("cpu") if first_parameter is None else first_parameter.device
    else:
        device = model_device.device
    metrics: dict[str, dict[str, float | int]] = {}
    result.model.eval()
    with torch.no_grad():
        for condition_index, condition in enumerate(condition_names):
            selected = torch.nonzero(
                condition_indices == condition_index, as_tuple=False
            ).flatten()
            if selected.numel() == 0:
                raise ValueError("every condition requires held-out validation rows")
            values = validation.index_select(0, selected).to(device)
            normalized = result.normalizer.normalize(values)
            reconstruction, latents = result.model(normalized)
            metrics[condition] = reconstruction_metrics(
                normalized.cpu(), reconstruction.cpu(), latents.cpu()
            )
    return metrics


def _condition_input_specs(
    values: Mapping[str, Any],
    *,
    split: str,
    paths: dict[str, Path],
    expected_hashes: dict[str, str],
    collect_runs: dict[Path, "_CollectRunSource"],
    verify_complete_runs: bool = True,
) -> dict[str, dict[str, "_CollectActivationReference"]]:
    if not values:
        raise ValueError(f"{split}_by_condition must not be empty")
    if not all(isinstance(name, str) for name in values):
        raise ValueError(f"{split}_by_condition keys must be strings")
    resolved: dict[str, dict[str, _CollectActivationReference]] = {}
    for raw_condition in sorted(values):
        condition = require_portable_identifier(raw_condition, name=f"{split} condition")
        datasets = values[raw_condition]
        if not isinstance(datasets, Mapping) or not datasets or "run_dir" in datasets:
            raise ValueError(
                f"{split} condition {condition!r} must map dataset IDs to collect artifacts"
            )
        if not all(isinstance(name, str) for name in datasets):
            raise ValueError(f"{split} dataset IDs must be strings")
        resolved_datasets: dict[str, _CollectActivationReference] = {}
        for raw_dataset in sorted(datasets):
            dataset = require_public_label(raw_dataset, name=f"{split} dataset_id")
            run_dir, artifact_name, key = _collect_activation_spec(
                datasets[raw_dataset],
                name=f"{split}[{condition}][{dataset}]",
            )
            source = collect_runs.get(run_dir)
            if source is None:
                source = (
                    _verify_collect_run_source(run_dir)
                    if verify_complete_runs
                    else _verify_parent_bound_collect_run_source(run_dir)
                )
                collect_runs[run_dir] = source
                _register_additional_input(
                    paths,
                    expected_hashes,
                    role=source.manifest_role,
                    file=source.manifest_file,
                )
                _register_additional_input(
                    paths,
                    expected_hashes,
                    role=source.index_role,
                    file=source.index_file,
                )
            entry = _unique_index_entry(
                source.index,
                artifact_name=artifact_name,
                dataset_id=dataset,
            )
            declared = _declared_artifact(source.manifest, artifact_name)
            activation_path = source.directory / artifact_name
            activation_file = verify_file(
                activation_path, expected_sha256=declared.sha256
            )
            if activation_file.digest.size_bytes != declared.size_bytes:
                raise ValueError("collect activation artifact size does not match its manifest")
            role_identity = json.dumps(
                {
                    "split": split,
                    "condition": condition,
                    "dataset_id": dataset,
                    "parent_manifest_sha256": source.manifest_file.digest.sha256,
                    "artifact": artifact_name,
                    "site": entry.site,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            activation_role = (
                "source.activation."
                + hashlib.sha256(role_identity).hexdigest()
            )
            _register_additional_input(
                paths,
                expected_hashes,
                role=activation_role,
                file=activation_file,
            )
            resolved_datasets[dataset] = _CollectActivationReference(
                split=split,
                condition=condition,
                dataset_id=dataset,
                source_directory=source.directory,
                artifact_name=artifact_name,
                key=key,
                activation_role=activation_role,
                site=entry.site,
            )
        resolved[condition] = resolved_datasets
    return resolved


def _load_verified_condition_inputs(
    specs: Mapping[str, Mapping[str, "_CollectActivationReference"]], context: Any
) -> dict[str, dict[str, Tensor]]:
    loaded: dict[str, dict[str, Tensor]] = {}
    for condition, datasets in specs.items():
        loaded[condition] = {}
        for dataset, reference in datasets.items():
            loaded[condition][dataset] = load_verified_activation_array(
                context.additional_file(reference.activation_role), key=reference.key
            )
    return loaded


@dataclass(frozen=True)
class _ActivationIndexEntry:
    site: str
    dataset_id: str
    artifact_name: str


@dataclass(frozen=True)
class _CollectActivationIndex:
    kind: str
    seed: int
    datasets: tuple[str, ...]
    sites: tuple[str, ...]
    entries: tuple[_ActivationIndexEntry, ...]
    official: "_OfficialCollectContract | None" = None


@dataclass(frozen=True)
class _OfficialCollectContract:
    model_family: str
    model_revision: str
    condition: str
    checkpoint_sha256: str
    model_code_sha: str
    assignment_split: str
    evaluation_split: str
    fit_context: str
    inference_contract_sha256: str
    max_classes: int
    manifest_inputs: tuple[InputDigest, ...]
    alignment_payloads: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class _CollectRunSource:
    directory: Path
    manifest: RunManifest
    manifest_file: VerifiedFile
    manifest_role: str
    index_file: VerifiedFile
    index_role: str
    index: _CollectActivationIndex


@dataclass(frozen=True)
class _CollectActivationReference:
    split: str
    condition: str
    dataset_id: str
    source_directory: Path
    artifact_name: str
    key: str
    activation_role: str
    site: str


def _collect_activation_spec(
    spec: Any, *, name: str
) -> tuple[Path, str, str]:
    if not isinstance(spec, Mapping):
        raise ValueError(
            f"{name} must reference a completed collect run, not a bare activation path"
        )
    expected_fields = {"run_dir", "artifact", "key"}
    missing = sorted(expected_fields - set(spec))
    unknown = sorted(set(spec) - expected_fields)
    if missing or unknown:
        raise ValueError(
            f"{name} collect reference fields mismatch: missing={missing}, unknown={unknown}"
        )
    raw_run_dir = spec["run_dir"]
    raw_artifact = spec["artifact"]
    raw_key = spec["key"]
    if not isinstance(raw_run_dir, str) or not raw_run_dir:
        raise ValueError(f"{name} run_dir must be a non-empty string")
    run_dir = Path(raw_run_dir).expanduser()
    if not run_dir.is_absolute():
        raise ValueError(f"{name} run_dir must be absolute")
    if run_dir.is_symlink():
        raise ValueError(f"{name} run_dir must not be a symlink")
    run_dir = run_dir.resolve(strict=True)
    if not isinstance(raw_artifact, str) or not raw_artifact:
        raise ValueError(f"{name} artifact must be a non-empty string")
    if not isinstance(raw_key, str) or raw_key != "activations":
        raise ValueError(f"{name} key must be exactly 'activations'")
    return run_dir, raw_artifact, raw_key


def _verify_collect_run_source(run_dir: Path) -> _CollectRunSource:
    checked_manifest = verify_run_directory(run_dir)
    if checked_manifest.command != "collect":
        raise ValueError("train-repr inputs must come from a collect run")
    if checked_manifest.evidence_level != "strict":
        raise ValueError("train-repr inputs require strict collect evidence")
    manifest_file = verify_file(run_dir / "manifest.json")
    exact_manifest = load_verified_run_manifest(manifest_file)
    if not isinstance(exact_manifest, RunManifest) or exact_manifest != checked_manifest:
        raise RuntimeError("collect manifest changed during source verification")
    parent_token = manifest_file.digest.sha256
    index_artifact = _declared_artifact(exact_manifest, "activation-index.json")
    index_file = verify_file(
        run_dir / index_artifact.name,
        expected_sha256=index_artifact.sha256,
    )
    if index_file.digest.size_bytes != index_artifact.size_bytes:
        raise ValueError("collect activation index size does not match its manifest")
    index = _parse_collect_activation_index(index_file.read_bytes())
    indexed_artifacts = {"activation-index.json"} | {
        entry.artifact_name for entry in index.entries
    }
    declared_artifacts = {artifact.name for artifact in exact_manifest.artifacts}
    if indexed_artifacts != declared_artifacts:
        raise ValueError(
            "collect manifest artifacts do not match its path-free activation index"
        )
    return _CollectRunSource(
        directory=run_dir,
        manifest=exact_manifest,
        manifest_file=manifest_file,
        manifest_role=f"source.collect_manifest.{parent_token}",
        index_file=index_file,
        index_role=f"source.collect_index.{parent_token}",
        index=index,
    )


def _verify_parent_bound_collect_run_source(run_dir: Path) -> _CollectRunSource:
    """Verify collect metadata without opening unselected activation artifacts.

    This narrow loader is only for a descendant workflow that subsequently
    proves the manifest, index, and selected artifact roles are already bound
    by a completed strict parent.  Unlike :func:`_verify_collect_run_source`, it
    intentionally inventories but does not hash unrelated activation shards.
    """

    manifest_file = verify_file(run_dir / "manifest.json")
    manifest = load_verified_run_manifest(manifest_file)
    if not isinstance(manifest, RunManifest) or manifest.command != "collect":
        raise ValueError("parent-bound sources must come from a collect run")
    if manifest.evidence_level != "strict" or manifest.legacy_reasons:
        raise ValueError("parent-bound sources require strict collect evidence")
    parent_token = manifest_file.digest.sha256
    index_artifact = _declared_artifact(manifest, "activation-index.json")
    index_file = verify_file(
        run_dir / index_artifact.name,
        expected_sha256=index_artifact.sha256,
    )
    if index_file.digest.size_bytes != index_artifact.size_bytes:
        raise ValueError("collect activation index size does not match its manifest")
    index = _parse_collect_activation_index(index_file.read_bytes())
    indexed_artifacts = {"activation-index.json"} | {
        entry.artifact_name for entry in index.entries
    }
    declared_artifacts = {artifact.name for artifact in manifest.artifacts}
    if indexed_artifacts != declared_artifacts:
        raise ValueError(
            "collect manifest artifacts do not match its path-free activation index"
        )
    expected_names = {"manifest.json", *declared_artifacts}
    observed_names: set[str] = set()
    for entry in run_dir.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError("collect run must contain only flat regular files")
        observed_names.add(entry.name)
    if observed_names != expected_names:
        raise ValueError("collect run inventory differs from its manifest")
    manifest_file.assert_unchanged()
    index_file.assert_unchanged()
    return _CollectRunSource(
        directory=run_dir,
        manifest=manifest,
        manifest_file=manifest_file,
        manifest_role=f"source.collect_manifest.{parent_token}",
        index_file=index_file,
        index_role=f"source.collect_index.{parent_token}",
        index=index,
    )


def _register_additional_input(
    paths: dict[str, Path],
    expected_hashes: dict[str, str],
    *,
    role: str,
    file: VerifiedFile,
) -> None:
    previous_path = paths.get(role)
    if previous_path is not None:
        if previous_path != file.path:
            raise ValueError("collect source role collision across different files")
        return
    paths[role] = file.path
    expected_hashes[role] = file.digest.sha256


def _declared_artifact(manifest: RunManifest, artifact_name: str) -> ArtifactDigest:
    matches = [artifact for artifact in manifest.artifacts if artifact.name == artifact_name]
    if len(matches) != 1:
        raise ValueError(
            f"collect artifact must be declared exactly once: {artifact_name!r}"
        )
    return matches[0]


def _parse_collect_activation_index(raw: bytes) -> _CollectActivationIndex:
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("collect activation-index.json must contain one object")
    kind = payload.get("kind")
    if kind == "bounded_activation_index":
        return _parse_generic_collect_activation_index(payload)
    if kind == "official_tabicl_bounded_activation_index":
        return _parse_official_collect_activation_index(payload)
    raise ValueError(f"unsupported collect activation index kind: {kind!r}")


def _parse_generic_collect_activation_index(
    payload: Mapping[str, Any],
) -> _CollectActivationIndex:
    expected_top = {"schema_version", "kind", "seed", "datasets", "sites"}
    if set(payload) != expected_top:
        raise ValueError("collect activation index fields do not match the registered schema")
    if payload["schema_version"] != 1 or payload["kind"] != "bounded_activation_index":
        raise ValueError("unsupported collect activation index")
    seed = payload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("collect activation index seed must be a non-negative integer")
    raw_datasets = payload["datasets"]
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError("collect activation index datasets must be a non-empty list")
    if not all(isinstance(value, str) for value in raw_datasets):
        raise ValueError("collect activation index dataset IDs must be strings")
    datasets = tuple(
        require_portable_identifier(value, name="collect dataset_id")
        for value in raw_datasets
    )
    if datasets != tuple(sorted(set(datasets))):
        raise ValueError("collect activation index datasets must be sorted and unique")
    raw_sites = payload["sites"]
    if not isinstance(raw_sites, list) or not raw_sites:
        raise ValueError("collect activation index sites must be a non-empty list")
    entries: list[_ActivationIndexEntry] = []
    sites: list[str] = []
    for site_record in raw_sites:
        if not isinstance(site_record, Mapping):
            raise ValueError("collect activation index site entries must be objects")
        expected_site_fields = {
            "site",
            "axis_names",
            "feature_group_maps",
            "datasets",
        }
        if set(site_record) != expected_site_fields:
            raise ValueError("collect activation site fields do not match the schema")
        site = require_portable_identifier(site_record["site"], name="collect site")
        sites.append(site)
        axes = site_record["axis_names"]
        if not isinstance(axes, list) or not axes:
            raise ValueError("collect activation axis_names must be a non-empty list")
        for axis in axes:
            require_portable_identifier(axis, name="collect activation axis")
        group_maps = site_record["feature_group_maps"]
        if not isinstance(group_maps, Mapping):
            raise ValueError("collect feature_group_maps must be an object")
        for raw_dataset_id, groups in group_maps.items():
            if not isinstance(raw_dataset_id, str):
                raise ValueError("collect feature_group_maps keys must be dataset IDs")
            mapped_dataset_id = require_portable_identifier(
                raw_dataset_id, name="collect feature_group_maps dataset_id"
            )
            if mapped_dataset_id not in datasets:
                raise ValueError("collect feature_group_maps references an unknown dataset")
            if not isinstance(groups, list) or not all(
                isinstance(group, list)
                and all(
                    not isinstance(index, bool) and isinstance(index, int) and index >= 0
                    for index in group
                )
                for group in groups
            ):
                raise ValueError("collect feature_group_maps must contain integer index lists")
        raw_site_datasets = site_record["datasets"]
        if not isinstance(raw_site_datasets, list) or not raw_site_datasets:
            raise ValueError("collect site datasets must be a non-empty list")
        for dataset_record in raw_site_datasets:
            if not isinstance(dataset_record, Mapping):
                raise ValueError("collect dataset entries must be objects")
            expected_dataset_fields = {
                "dataset_id",
                "file",
                "feature_dim",
                "dtype",
                "seen_vectors",
                "retained_vectors",
            }
            if set(dataset_record) != expected_dataset_fields:
                raise ValueError("collect dataset fields do not match the schema")
            if not isinstance(dataset_record["dataset_id"], str):
                raise ValueError("collect activation dataset_id must be a string")
            dataset_id = require_portable_identifier(
                dataset_record["dataset_id"], name="collect dataset_id"
            )
            artifact_name = dataset_record["file"]
            if not isinstance(artifact_name, str):
                raise ValueError("collect activation artifact name must be a string")
            # The manifest schema accepts only a flat portable filename.  Use
            # the same constructor here rather than trusting an index path.
            ArtifactDigest(artifact_name, "0" * 64, 0)
            feature_dim = dataset_record["feature_dim"]
            seen = dataset_record["seen_vectors"]
            retained = dataset_record["retained_vectors"]
            if (
                isinstance(feature_dim, bool)
                or not isinstance(feature_dim, int)
                or feature_dim <= 0
                or isinstance(seen, bool)
                or not isinstance(seen, int)
                or isinstance(retained, bool)
                or not isinstance(retained, int)
                or retained <= 0
                or seen < retained
            ):
                raise ValueError("collect activation index contains invalid vector counts")
            dtype = dataset_record["dtype"]
            if not isinstance(dtype, str) or np.dtype(dtype).kind != "f":
                raise ValueError("collect activation dtype must be floating point")
            entries.append(_ActivationIndexEntry(site, dataset_id, artifact_name))
    if tuple(sites) != tuple(sorted(set(sites))):
        raise ValueError("collect activation index sites must be sorted and unique")
    entry_keys = [(entry.site, entry.dataset_id) for entry in entries]
    artifact_names = [entry.artifact_name for entry in entries]
    if len(set(entry_keys)) != len(entry_keys) or len(set(artifact_names)) != len(entries):
        raise ValueError("collect activation index dataset/site artifacts must be unique")
    if set(datasets) != {entry.dataset_id for entry in entries}:
        raise ValueError("collect activation index dataset roster is inconsistent")
    return _CollectActivationIndex(
        kind="bounded_activation_index",
        seed=seed,
        datasets=datasets,
        sites=tuple(sites),
        entries=tuple(entries),
    )


def _parse_official_collect_activation_index(
    payload: Mapping[str, Any],
) -> _CollectActivationIndex:
    _require_exact_mapping_fields(
        payload,
        {
            "schema_version",
            "kind",
            "model_family",
            "model_revision",
            "condition",
            "checkpoint_sha256",
            "model_code_sha",
            "seed",
            "assignment_split",
            "evaluation_split",
            "fit_context",
            "inference_contract_sha256",
            "max_classes",
            "datasets",
            "sites",
        },
        name="official collect activation index",
    )
    if payload["schema_version"] != 1:
        raise ValueError("official collect activation index schema_version must be 1")
    if payload["kind"] != "official_tabicl_bounded_activation_index":
        raise ValueError("unsupported official collect activation index kind")
    model_family = _required_string(payload["model_family"], name="model_family")
    if model_family != "tabicl-v2":
        raise ValueError("official collect model_family must be 'tabicl-v2'")
    model_revision = require_portable_identifier(
        _required_string(payload["model_revision"], name="model_revision"),
        name="model_revision",
    )
    condition = require_portable_identifier(
        _required_string(payload["condition"], name="condition"),
        name="condition",
    )
    if condition not in {"rope", "temporary", "none"}:
        raise ValueError("official collect condition is not registered")
    checkpoint_sha256 = _required_sha256(
        payload["checkpoint_sha256"], name="checkpoint_sha256"
    )
    model_code_sha = _required_git_sha(payload["model_code_sha"], name="model_code_sha")
    seed = _required_nonnegative_integer(payload["seed"], name="seed")
    assignment_split = require_portable_identifier(
        _required_string(payload["assignment_split"], name="assignment_split"),
        name="assignment_split",
    )
    evaluation_split = _required_string(
        payload["evaluation_split"], name="evaluation_split"
    )
    if evaluation_split not in {"val", "test"}:
        raise ValueError("official collect evaluation_split must be val or test")
    fit_context = _required_string(payload["fit_context"], name="fit_context")
    if fit_context != "train":
        raise ValueError("official collect fit_context must be train")
    inference_contract_sha256 = _required_sha256(
        payload["inference_contract_sha256"], name="inference_contract_sha256"
    )
    max_classes = _required_positive_integer(payload["max_classes"], name="max_classes")
    if max_classes > 10:
        raise ValueError("official collect max_classes cannot exceed 10")

    raw_datasets = payload["datasets"]
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError("official collect datasets must be a non-empty list")
    dataset_ids: list[str] = []
    dataset_alignment: dict[str, Mapping[str, Any]] = {}
    preprocessing_trace_by_dataset: dict[str, str] = {}
    manifest_inputs: list[InputDigest] = []
    for ordinal, raw_dataset in enumerate(raw_datasets):
        if not isinstance(raw_dataset, Mapping):
            raise ValueError("official collect dataset entries must be objects")
        dataset_id, alignment_payload, trace_sha, inputs = _parse_official_dataset(
            raw_dataset,
            ordinal=ordinal,
            assignment_split=assignment_split,
            evaluation_split=evaluation_split,
            fit_context=fit_context,
        )
        dataset_ids.append(dataset_id)
        dataset_alignment[dataset_id] = alignment_payload
        preprocessing_trace_by_dataset[dataset_id] = trace_sha
        manifest_inputs.extend(inputs)
    expected_dataset_order = sorted(dataset_ids, key=lambda value: (value.casefold(), value))
    if dataset_ids != expected_dataset_order or len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("official collect dataset roster must be sorted and unique")

    raw_sites = payload["sites"]
    if not isinstance(raw_sites, list) or not raw_sites:
        raise ValueError("official collect sites must be a non-empty list")
    sites: list[str] = []
    entries: list[_ActivationIndexEntry] = []
    alignment_payloads: dict[str, Mapping[str, Any]] = {}
    for raw_site in raw_sites:
        if not isinstance(raw_site, Mapping):
            raise ValueError("official collect site entries must be objects")
        _require_exact_mapping_fields(
            raw_site,
            {"site", "axis_names", "datasets"},
            name="official collect site entry",
        )
        site = require_portable_identifier(
            _required_string(raw_site["site"], name="site"), name="site"
        )
        if site in sites:
            raise ValueError("official collect sites must be unique")
        sites.append(site)
        axis_names = _portable_string_list(raw_site["axis_names"], name="axis_names")
        if not axis_names or axis_names[-1] not in ACTIVATION_VECTOR_AXIS_NAMES:
            raise ValueError(
                "official collect site axes must end in a supported vector axis"
            )
        if len(set(axis_names)) != len(axis_names):
            raise ValueError("official collect site axes must be unique")
        raw_site_datasets = raw_site["datasets"]
        if not isinstance(raw_site_datasets, list) or not raw_site_datasets:
            raise ValueError("official collect site datasets must be a non-empty list")
        site_dataset_ids: list[str] = []
        for raw_entry in raw_site_datasets:
            if not isinstance(raw_entry, Mapping):
                raise ValueError("official collect site dataset entries must be objects")
            entry, site_alignment = _parse_official_site_dataset(
                raw_entry,
                site=site,
                site_axis_names=axis_names,
                known_datasets=set(dataset_ids),
                preprocessing_trace_by_dataset=preprocessing_trace_by_dataset,
                dataset_alignment_by_dataset=dataset_alignment,
            )
            if entry.artifact_name in alignment_payloads:
                raise ValueError("official collect activation artifacts must be unique")
            site_dataset_ids.append(entry.dataset_id)
            entries.append(entry)
            alignment_payloads[entry.artifact_name] = {
                **dataset_alignment[entry.dataset_id],
                **site_alignment,
                "seed": seed,
            }
        if site_dataset_ids != dataset_ids:
            raise ValueError(
                "official collect every site must contain the exact sorted dataset roster"
            )

    entry_keys = [(entry.site, entry.dataset_id) for entry in entries]
    if len(set(entry_keys)) != len(entry_keys):
        raise ValueError("official collect dataset/site entries must be unique")
    sorted_inputs = tuple(sorted(manifest_inputs, key=lambda item: item.role))
    if len({item.role for item in sorted_inputs}) != len(sorted_inputs):
        raise ValueError("official collect TALENT source roles must be unique")
    official = _OfficialCollectContract(
        model_family=model_family,
        model_revision=model_revision,
        condition=condition,
        checkpoint_sha256=checkpoint_sha256,
        model_code_sha=model_code_sha,
        assignment_split=assignment_split,
        evaluation_split=evaluation_split,
        fit_context=fit_context,
        inference_contract_sha256=inference_contract_sha256,
        max_classes=max_classes,
        manifest_inputs=sorted_inputs,
        alignment_payloads=alignment_payloads,
    )
    return _CollectActivationIndex(
        kind="official_tabicl_bounded_activation_index",
        seed=seed,
        datasets=tuple(dataset_ids),
        sites=tuple(sites),
        entries=tuple(entries),
        official=official,
    )


def _parse_official_dataset(
    payload: Mapping[str, Any],
    *,
    ordinal: int,
    assignment_split: str,
    evaluation_split: str,
    fit_context: str,
) -> tuple[str, Mapping[str, Any], str, tuple[InputDigest, ...]]:
    _require_exact_mapping_fields(
        payload,
        {
            "dataset_id",
            "task_type",
            "assignment_split",
            "evaluation_split",
            "fit_context",
            "n_numeric_features",
            "n_categorical_features",
            "info_sha256",
            "inputs",
            "input_bundle_sha256",
            "preprocessing_trace_sha256",
            "sample_roster_sha256",
            "probabilities_sha256",
            "metrics",
            "exact_baseline_verified",
            "feature_group_mode",
            "local_feature_group_map",
            "official_forward_calls",
        },
        name="official collect dataset entry",
    )
    dataset_id = require_public_label(
        _required_string(payload["dataset_id"], name="dataset_id"), name="dataset_id"
    )
    task_type = _required_string(payload["task_type"], name="task_type")
    if task_type.strip().lower() not in {
        "binclass",
        "binary",
        "binary_classification",
        "classification",
        "multiclass",
        "multiclass_classification",
    }:
        raise ValueError("official collect supports only classification datasets")
    for field, expected in (
        ("assignment_split", assignment_split),
        ("evaluation_split", evaluation_split),
        ("fit_context", fit_context),
    ):
        if payload[field] != expected:
            raise ValueError(f"official dataset {field} differs from the index root")
    n_numeric = _required_nonnegative_integer(
        payload["n_numeric_features"], name="n_numeric_features"
    )
    n_categorical = _required_nonnegative_integer(
        payload["n_categorical_features"], name="n_categorical_features"
    )
    if n_numeric + n_categorical <= 0:
        raise ValueError("official dataset must contain at least one feature")
    info_sha256 = _required_sha256(payload["info_sha256"], name="info_sha256")
    input_bundle_sha256 = _required_sha256(
        payload["input_bundle_sha256"], name="input_bundle_sha256"
    )
    preprocessing_trace_sha256 = _required_sha256(
        payload["preprocessing_trace_sha256"], name="preprocessing_trace_sha256"
    )
    sample_roster_sha256 = _required_sha256(
        payload["sample_roster_sha256"], name="sample_roster_sha256"
    )
    _required_sha256(payload["probabilities_sha256"], name="probabilities_sha256")
    if payload["exact_baseline_verified"] is not True:
        raise ValueError("official dataset exact baseline must be verified")
    if payload["feature_group_mode"] != "same":
        raise ValueError("official dataset feature_group_mode must be same")
    local_feature_group_map = _integer_group_map(
        payload["local_feature_group_map"], name="local_feature_group_map"
    )
    forward_calls = _official_forward_calls(payload["official_forward_calls"])

    raw_inputs = payload["inputs"]
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise ValueError("official dataset inputs must be a non-empty list")
    parsed_inputs: list[InputDigest] = []
    logical_hashes: dict[str, str] = {}
    logical_names: list[str] = []
    for raw_input in raw_inputs:
        if not isinstance(raw_input, Mapping):
            raise ValueError("official dataset input entries must be objects")
        _require_exact_mapping_fields(
            raw_input,
            {"logical_name", "role", "sha256", "size_bytes"},
            name="official dataset input",
        )
        logical_name = _required_string(raw_input["logical_name"], name="logical_name")
        if logical_name not in _OFFICIAL_ALLOWED_LOGICAL_INPUTS:
            raise ValueError("official dataset contains an unknown logical input")
        role = _required_string(raw_input["role"], name="TALENT input role")
        match = _TALENT_INPUT_ROLE.fullmatch(role)
        expected_token = logical_name.removesuffix(".npy").replace(".", "_")
        if (
            match is None
            or int(match.group(1)) != ordinal
            or match.group(2) != expected_token
        ):
            raise ValueError("official dataset TALENT input role does not match its ordinal")
        digest = _required_sha256(raw_input["sha256"], name="input sha256")
        size = _required_nonnegative_integer(raw_input["size_bytes"], name="input size")
        if logical_name in logical_hashes:
            raise ValueError("official dataset logical input names must be unique")
        logical_names.append(logical_name)
        logical_hashes[logical_name] = digest
        parsed_inputs.append(InputDigest(role=role, sha256=digest, size_bytes=size))
    if logical_names != sorted(logical_names):
        raise ValueError("official dataset logical inputs must be sorted")
    missing_inputs = sorted(_OFFICIAL_REQUIRED_LOGICAL_INPUTS - set(logical_names))
    if missing_inputs:
        raise ValueError(f"official dataset is missing required inputs: {missing_inputs}")
    for split in ("train", "val", "test"):
        if not {f"N_{split}.npy", f"C_{split}.npy"} & set(logical_names):
            raise ValueError(
                f"official dataset is missing feature inputs for split {split!r}"
            )
    if logical_hashes["info.json"] != info_sha256:
        raise ValueError("official dataset info hash differs from its input roster")
    if _canonical_json_sha256(logical_hashes) != input_bundle_sha256:
        raise ValueError("official dataset input bundle hash is invalid")

    metrics = payload["metrics"]
    if not isinstance(metrics, Mapping):
        raise ValueError("official dataset metrics must be an object")
    _require_exact_mapping_fields(
        metrics, {"accuracy", "log_loss", "n_samples"}, name="official dataset metrics"
    )
    for name in ("accuracy", "log_loss"):
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"official dataset metric {name} must be finite")
    _required_positive_integer(metrics["n_samples"], name="metric n_samples")

    alignment_payload = {
        "dataset_id": dataset_id,
        "task_type": task_type,
        "assignment_split": assignment_split,
        "evaluation_split": evaluation_split,
        "fit_context": fit_context,
        "n_numeric_features": n_numeric,
        "n_categorical_features": n_categorical,
        "info_sha256": info_sha256,
        "inputs": [
            {
                "logical_name": logical_name,
                "sha256": logical_hashes[logical_name],
                "size_bytes": next(
                    item.size_bytes
                    for item in parsed_inputs
                    if item.role.endswith(
                        "." + logical_name.removesuffix(".npy").replace(".", "_")
                    )
                ),
            }
            for logical_name in logical_names
        ],
        "input_bundle_sha256": input_bundle_sha256,
        "sample_roster_sha256": sample_roster_sha256,
        "feature_group_mode": "same",
        "local_feature_group_map": local_feature_group_map,
        "official_forward_calls": forward_calls,
    }
    return (
        dataset_id,
        alignment_payload,
        preprocessing_trace_sha256,
        tuple(parsed_inputs),
    )


def _parse_official_site_dataset(
    payload: Mapping[str, Any],
    *,
    site: str,
    site_axis_names: tuple[str, ...],
    known_datasets: set[str],
    preprocessing_trace_by_dataset: Mapping[str, str],
    dataset_alignment_by_dataset: Mapping[str, Mapping[str, Any]],
) -> tuple[_ActivationIndexEntry, Mapping[str, Any]]:
    _require_exact_mapping_fields(
        payload,
        {
            "dataset_id",
            "file",
            "axis_names",
            "activation_shapes",
            "coordinate_axis_names",
            "vector_axis_name",
            "feature_group_token_offset",
            "cls_token_count",
            "feature_dim",
            "dtype",
            "seen_vectors",
            "retained_vectors",
            "preprocessing_trace_sha256",
        },
        name="official site dataset entry",
    )
    dataset_id = require_public_label(
        _required_string(payload["dataset_id"], name="dataset_id"), name="dataset_id"
    )
    if dataset_id not in known_datasets:
        raise ValueError("official site references an unknown dataset")
    artifact_name = _required_string(payload["file"], name="activation artifact")
    ArtifactDigest(artifact_name, "0" * 64, 0)
    axis_names = _portable_string_list(payload["axis_names"], name="axis_names")
    coordinate_axis_names = _portable_string_list(
        payload["coordinate_axis_names"], name="coordinate_axis_names"
    )
    vector_axis_name = require_portable_identifier(
        _required_string(payload["vector_axis_name"], name="vector_axis_name"),
        name="vector_axis_name",
    )
    if axis_names != site_axis_names:
        raise ValueError("official site dataset axes differ from the site axes")
    if coordinate_axis_names != axis_names[:-1] or vector_axis_name != axis_names[-1]:
        raise ValueError("official activation coordinate metadata is inconsistent")
    feature_dim = _required_positive_integer(payload["feature_dim"], name="feature_dim")
    raw_activation_shapes = payload["activation_shapes"]
    if not isinstance(raw_activation_shapes, list) or not raw_activation_shapes:
        raise ValueError("official activation_shapes must be a non-empty list")
    activation_shapes = [
        _positive_integer_list(shape, name="activation shape")
        for shape in raw_activation_shapes
    ]
    forward_calls = dataset_alignment_by_dataset[dataset_id]["official_forward_calls"]
    if len(activation_shapes) != len(forward_calls):
        raise ValueError("official activation_shapes must align to forward calls")
    if any(
        len(shape) != len(axis_names) or shape[-1] != feature_dim
        for shape in activation_shapes
    ):
        raise ValueError(
            "official activation_shapes rank/embedding width differs from site metadata"
        )
    feature_group_token_offset = _optional_nonnegative_integer(
        payload["feature_group_token_offset"], name="feature_group_token_offset"
    )
    cls_token_count = _optional_nonnegative_integer(
        payload["cls_token_count"], name="cls_token_count"
    )
    local_group_map = dataset_alignment_by_dataset[dataset_id][
        "local_feature_group_map"
    ]
    if "feature_group_or_cls" in axis_names:
        token_axis = axis_names.index("feature_group_or_cls")
        if feature_group_token_offset is None or cls_token_count is None:
            raise ValueError(
                "feature_group_or_cls axes require explicit token offset/CLS count"
            )
        if feature_group_token_offset != cls_token_count or any(
            shape[token_axis] != cls_token_count + len(local_group_map)
            for shape in activation_shapes
        ):
            raise ValueError(
                "official feature-group token offset is inconsistent with activation shapes"
            )
    elif "cls" in axis_names:
        cls_axis = axis_names.index("cls")
        if feature_group_token_offset is not None or cls_token_count is None:
            raise ValueError("cls axes require only an explicit cls_token_count")
        if cls_token_count <= 0 or any(
            shape[cls_axis] != cls_token_count for shape in activation_shapes
        ):
            raise ValueError("official cls_token_count differs from activation shapes")
    elif feature_group_token_offset is not None or cls_token_count is not None:
        raise ValueError("token offset metadata is only valid for group/CLS axes")
    dtype = _required_string(payload["dtype"], name="activation dtype")
    try:
        resolved_dtype = np.dtype(dtype)
    except TypeError as error:
        raise ValueError("official activation dtype is invalid") from error
    if resolved_dtype.kind != "f":
        raise ValueError("official activation dtype must be floating point")
    seen = _required_positive_integer(payload["seen_vectors"], name="seen_vectors")
    retained = _required_positive_integer(
        payload["retained_vectors"], name="retained_vectors"
    )
    if retained > seen:
        raise ValueError("official retained_vectors cannot exceed seen_vectors")
    measured_seen = sum(
        int(np.prod(shape[:-1], dtype=np.int64)) for shape in activation_shapes
    )
    if seen != measured_seen:
        raise ValueError("official seen_vectors differs from activation_shapes")
    trace_sha = _required_sha256(
        payload["preprocessing_trace_sha256"], name="preprocessing_trace_sha256"
    )
    if trace_sha != preprocessing_trace_by_dataset[dataset_id]:
        raise ValueError("official site preprocessing trace differs from its dataset")
    return (
        _ActivationIndexEntry(site, dataset_id, artifact_name),
        {
            "site": site,
            "axis_names": axis_names,
            "activation_shapes": activation_shapes,
            "coordinate_axis_names": coordinate_axis_names,
            "vector_axis_name": vector_axis_name,
            "feature_group_token_offset": feature_group_token_offset,
            "cls_token_count": cls_token_count,
            "feature_dim": feature_dim,
            "dtype": resolved_dtype.name,
            "seen_vectors": seen,
            "retained_vectors": retained,
        },
    )


def _official_forward_calls(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("official_forward_calls must be a non-empty list")
    parsed: list[Mapping[str, Any]] = []
    for expected_call_index, raw_call in enumerate(value):
        if not isinstance(raw_call, Mapping):
            raise ValueError("official forward calls must be objects")
        _require_exact_mapping_fields(
            raw_call,
            {
                "call_index",
                "norm_method",
                "norm_view_indices",
                "ensemble_indices",
                "feature_shuffles",
                "class_shuffles",
                "raw_input_shape",
                "train_size",
                "preprocessing_view_id",
                "post_filter_feature_group_maps",
                "feature_coordinate_system",
            },
            name="official forward call",
        )
        call_index = _required_nonnegative_integer(
            raw_call["call_index"], name="call_index"
        )
        if call_index != expected_call_index:
            raise ValueError("official forward call indices must be contiguous")
        norm_method = require_portable_identifier(
            _required_string(raw_call["norm_method"], name="norm_method"),
            name="norm_method",
        )
        raw_shape = _positive_integer_list(
            raw_call["raw_input_shape"], name="raw_input_shape"
        )
        if len(raw_shape) != 3:
            raise ValueError("official raw_input_shape must have rank three")
        train_size = _required_positive_integer(raw_call["train_size"], name="train_size")
        if train_size > raw_shape[1]:
            raise ValueError("official train_size exceeds the raw row count")
        view_id = _required_string(
            raw_call["preprocessing_view_id"], name="preprocessing_view_id"
        )
        if view_id != f"official-tabicl/{norm_method}/raw-call-{call_index:04d}":
            raise ValueError("official preprocessing view ID is inconsistent")
        if raw_call["feature_coordinate_system"] != (
            "official-encoded-after-constant-filter-before-view-shuffle"
        ):
            raise ValueError("official feature coordinate system is not registered")
        norm_views = _nonnegative_integer_list(
            raw_call["norm_view_indices"], name="norm_view_indices"
        )
        ensemble_indices = _nonnegative_integer_list(
            raw_call["ensemble_indices"], name="ensemble_indices"
        )
        feature_shuffles = _permutation_list(
            raw_call["feature_shuffles"], name="feature_shuffles"
        )
        class_shuffles = _permutation_list(
            raw_call["class_shuffles"], name="class_shuffles"
        )
        post_filter_maps = raw_call["post_filter_feature_group_maps"]
        if not isinstance(post_filter_maps, list):
            raise ValueError("post_filter_feature_group_maps must be a list")
        normalized_maps = [
            _integer_group_map(group_map, name="post_filter_feature_group_map")
            for group_map in post_filter_maps
        ]
        table_count = raw_shape[0]
        if not all(
            len(item) == table_count
            for item in (
                norm_views,
                ensemble_indices,
                feature_shuffles,
                class_shuffles,
                normalized_maps,
            )
        ):
            raise ValueError("official forward-call table metadata has a wrong count")
        parsed.append(
            {
                "call_index": call_index,
                "norm_method": norm_method,
                "norm_view_indices": norm_views,
                "ensemble_indices": ensemble_indices,
                "feature_shuffles": feature_shuffles,
                "class_shuffles": class_shuffles,
                "raw_input_shape": raw_shape,
                "train_size": train_size,
                "preprocessing_view_id": view_id,
                "post_filter_feature_group_maps": normalized_maps,
                "feature_coordinate_system": raw_call["feature_coordinate_system"],
            }
        )
    return parsed


def _require_exact_mapping_fields(
    value: Mapping[str, Any], expected: set[str], *, name: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        raise ValueError(f"{name} fields mismatch: missing={missing}, unknown={unknown}")


def _required_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _required_sha256(value: Any, *, name: str) -> str:
    resolved = _required_string(value, name=name)
    if _SHA256_HEX.fullmatch(resolved) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")
    return resolved


def _required_git_sha(value: Any, *, name: str) -> str:
    resolved = _required_string(value, name=name)
    if _GIT_SHA_HEX.fullmatch(resolved) is None:
        raise ValueError(f"{name} must be a Git commit digest")
    return resolved


def _required_nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_nonnegative_integer(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    return _required_nonnegative_integer(value, name=name)


def _required_positive_integer(value: Any, *, name: str) -> int:
    resolved = _required_nonnegative_integer(value, name=name)
    if resolved == 0:
        raise ValueError(f"{name} must be positive")
    return resolved


def _portable_string_list(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return tuple(
        require_portable_identifier(
            _required_string(item, name=name), name=name
        )
        for item in value
    )


def _nonnegative_integer_list(value: Any, *, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty integer list")
    return [_required_nonnegative_integer(item, name=name) for item in value]


def _positive_integer_list(value: Any, *, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty integer list")
    return [_required_positive_integer(item, name=name) for item in value]


def _integer_group_map(value: Any, *, name: str) -> list[list[int]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    groups: list[list[int]] = []
    for raw_group in value:
        groups.append(_nonnegative_integer_list(raw_group, name=name))
    return groups


def _permutation_list(value: Any, *, name: str) -> list[list[int]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    permutations: list[list[int]] = []
    for raw_permutation in value:
        permutation = _nonnegative_integer_list(raw_permutation, name=name)
        if sorted(permutation) != list(range(len(permutation))):
            raise ValueError(f"{name} entries must be permutations")
        permutations.append(permutation)
    return permutations


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_index_entry(
    index: _CollectActivationIndex, *, artifact_name: str, dataset_id: str
) -> _ActivationIndexEntry:
    matches = [entry for entry in index.entries if entry.artifact_name == artifact_name]
    if len(matches) != 1:
        raise ValueError(
            f"collect artifact must identify exactly one dataset/site: {artifact_name!r}"
        )
    entry = matches[0]
    if entry.dataset_id != dataset_id:
        raise ValueError(
            f"collect artifact dataset mismatch: expected {dataset_id!r}, "
            f"observed {entry.dataset_id!r}"
        )
    return entry


def _assert_collect_lineage(
    collect_runs: Mapping[Path, _CollectRunSource],
    references: Sequence[_CollectActivationReference],
    context: Any,
    *,
    reference_condition: str,
) -> dict[str, Any]:
    conditions = tuple(sorted({reference.condition for reference in references}))
    if reference_condition not in conditions:
        raise ValueError("reference_condition is absent from the activation conditions")
    if context.condition != reference_condition:
        raise ValueError(
            "train-repr provenance condition must equal reference_condition"
        )
    if len(context.sites) != 1:
        raise ValueError("train-repr requires exactly one activation site per dictionary")
    kinds = {source.index.kind for source in collect_runs.values()}
    if len(kinds) != 1:
        raise ValueError("train-repr must not mix generic and official collect schemas")
    source_kind = next(iter(kinds))
    if len(conditions) > 1 and source_kind != (
        "official_tabicl_bounded_activation_index"
    ):
        raise ValueError(
            "multi-condition representation training requires official coordinate evidence"
        )

    references_by_source: dict[Path, list[_CollectActivationReference]] = {
        source: [] for source in collect_runs
    }
    for reference in references:
        references_by_source[reference.source_directory].append(reference)
    checkpoint_digests: dict[str, set[tuple[str, int]]] = {
        condition: set() for condition in conditions
    }
    parent_manifest_digests: dict[str, set[str]] = {
        condition: set() for condition in conditions
    }
    alignment_by_group: dict[
        tuple[str, str, str], dict[str, str]
    ] = {}
    inference_contracts: set[str] = set()
    official_evaluation_splits: set[str] = set()
    official_max_classes: set[int] = set()
    dataset_roster = verified_dataset_roster(context.inputs.dataset_manifest)
    for directory, source in collect_runs.items():
        exact_manifest = load_verified_run_manifest(
            context.additional_file(source.manifest_role)
        )
        if not isinstance(exact_manifest, RunManifest) or exact_manifest != source.manifest:
            raise RuntimeError("collect manifest changed after directory verification")
        if exact_manifest.command != "collect" or exact_manifest.evidence_level != "strict":
            raise ValueError("train-repr inputs require a strict collect manifest")
        source_conditions = {item.condition for item in references_by_source[directory]}
        if source_conditions != {exact_manifest.condition}:
            raise ValueError("collect manifest condition does not match its representation input")
        if exact_manifest.model_family != context.model_family:
            raise ValueError("collect manifest model_family does not match train-repr")
        if exact_manifest.model_revision != context.model_revision:
            raise ValueError("collect manifest model_revision does not match train-repr")
        selected_site = context.sites[0]
        if selected_site not in exact_manifest.sites:
            raise ValueError(
                "collect manifest sites do not match train-repr provenance: "
                "selected site is absent"
            )
        if exact_manifest.dataset_manifest != context.inputs.dataset_manifest.digest:
            raise ValueError("collect manifest dataset manifest does not match train-repr")
        if exact_manifest.training_code_sha != context.inputs.training_code.head_sha:
            raise ValueError("collect manifest training code does not match train-repr")
        if exact_manifest.model_code_sha != context.inputs.model_code.head_sha:
            raise ValueError("collect manifest model code does not match train-repr")
        assert_git_commit_is_ancestor(
            context.inputs.analysis_code, exact_manifest.analysis_code_sha
        )

        index_artifact = _declared_artifact(exact_manifest, "activation-index.json")
        index_file = context.additional_file(source.index_role)
        if (
            index_file.digest.sha256 != index_artifact.sha256
            or index_file.digest.size_bytes != index_artifact.size_bytes
        ):
            raise ValueError("verified collect activation index differs from its manifest")
        index = _parse_collect_activation_index(index_file.read_bytes())
        if index.seed != exact_manifest.seed or index.seed != context.inputs.contract.seed:
            raise ValueError("collect activation index seed does not match its manifest")
        if tuple(sorted(index.sites)) != tuple(sorted(exact_manifest.sites)):
            raise ValueError("collect activation index sites do not match its manifest")
        if index.kind == "bounded_activation_index":
            expected_source_roles = tuple(
                f"activation.{index_number:06d}"
                for index_number in range(len(exact_manifest.inputs))
            )
            if not exact_manifest.inputs or tuple(
                item.role for item in exact_manifest.inputs
            ) != expected_source_roles:
                raise ValueError("collect manifest has invalid raw activation source lineage")
        else:
            official = index.official
            if official is None:
                raise AssertionError("official collect index omitted its strict contract")
            if exact_manifest.inputs != official.manifest_inputs:
                raise ValueError(
                    "official collect manifest TALENT inputs differ from its index"
                )
            if official.model_family != exact_manifest.model_family or (
                official.model_family != context.model_family
            ):
                raise ValueError("official index model_family does not match provenance")
            if official.model_revision != exact_manifest.model_revision or (
                official.model_revision != context.model_revision
            ):
                raise ValueError("official index model_revision does not match provenance")
            if official.condition != exact_manifest.condition:
                raise ValueError("official index condition does not match its manifest")
            if official.checkpoint_sha256 != exact_manifest.checkpoint.sha256:
                raise ValueError("official index checkpoint does not match its manifest")
            if official.model_code_sha != exact_manifest.model_code_sha or (
                official.model_code_sha != context.inputs.model_code.head_sha
            ):
                raise ValueError("official index model code does not match provenance")
            inference_contracts.add(official.inference_contract_sha256)
            official_evaluation_splits.add(official.evaluation_split)
            official_max_classes.add(official.max_classes)

        source_condition = next(iter(source_conditions))
        checkpoint_digests[source_condition].add(
            (
                exact_manifest.checkpoint.sha256,
                exact_manifest.checkpoint.size_bytes,
            )
        )
        parent_manifest_digests[source_condition].add(
            source.manifest_file.digest.sha256
        )
        for reference in references_by_source[directory]:
            entry = _unique_index_entry(
                index,
                artifact_name=reference.artifact_name,
                dataset_id=reference.dataset_id,
            )
            if entry.site != reference.site or entry.site not in context.sites:
                raise ValueError("collect activation artifact site does not match train-repr")
            declared = _declared_artifact(exact_manifest, reference.artifact_name)
            activation_file = context.additional_file(reference.activation_role)
            if (
                activation_file.digest.sha256 != declared.sha256
                or activation_file.digest.size_bytes != declared.size_bytes
            ):
                raise ValueError("verified collect activation differs from its manifest")
            if index.official is not None:
                expected_assignment = (
                    "discovery" if reference.split == "training" else "validation"
                )
                if index.official.assignment_split != expected_assignment:
                    raise ValueError(
                        "official collect assignment split does not match train-repr split"
                    )
                if dataset_roster.get(reference.dataset_id) != expected_assignment:
                    raise ValueError(
                        "official collect assignment differs from the dataset manifest"
                    )
                alignment = _official_alignment_sha256(
                    index,
                    reference=reference,
                    activation_file=activation_file,
                )
                group = (reference.split, reference.dataset_id, reference.site)
                condition_alignment = alignment_by_group.setdefault(group, {})
                if reference.condition in condition_alignment:
                    raise ValueError("duplicate condition in official alignment group")
                condition_alignment[reference.condition] = alignment

    resolved_checkpoints: dict[str, str] = {}
    for condition in conditions:
        observed = checkpoint_digests[condition]
        if len(observed) != 1:
            raise ValueError(
                "all training/validation collect parents within one condition must use "
                "the same checkpoint"
            )
        resolved_checkpoints[condition] = next(iter(observed))[0]
    reference_digest = next(iter(checkpoint_digests[reference_condition]))
    if reference_digest != (
        context.inputs.checkpoint.digest.sha256,
        context.inputs.checkpoint.digest.size_bytes,
    ):
        raise ValueError(
            "train-repr checkpoint must equal the reference-condition checkpoint"
        )

    public_alignment: dict[str, dict[str, dict[str, str]]] = {}
    if source_kind == "official_tabicl_bounded_activation_index":
        if len(inference_contracts) != 1:
            raise ValueError("official collect inference contracts must match exactly")
        if len(official_evaluation_splits) != 1:
            raise ValueError("official collect evaluation splits must match exactly")
        if len(official_max_classes) != 1:
            raise ValueError("official collect max_classes must match exactly")
        expected_conditions = set(conditions)
        for (split, dataset_id, site), values in sorted(alignment_by_group.items()):
            if set(values) != expected_conditions:
                raise ValueError(
                    "every official dataset/site alignment group must contain all conditions"
                )
            unique = set(values.values())
            if len(unique) != 1:
                raise ValueError(
                    "official activation coordinates differ across conditions"
                )
            public_alignment.setdefault(split, {}).setdefault(dataset_id, {})[
                site
            ] = next(iter(unique))

    return {
        "schema_version": 1,
        "source_kind": source_kind,
        "reference_condition": reference_condition,
        "condition_checkpoints_sha256": dict(sorted(resolved_checkpoints.items())),
        "collect_parent_manifests_sha256": {
            condition: sorted(parent_manifest_digests[condition])
            for condition in conditions
        },
        "alignment_sha256": public_alignment,
        "inference_contract_sha256": (
            next(iter(inference_contracts)) if inference_contracts else None
        ),
        "evaluation_split": (
            next(iter(official_evaluation_splits))
            if official_evaluation_splits
            else None
        ),
        "max_classes": (
            next(iter(official_max_classes)) if official_max_classes else None
        ),
    }


def _official_alignment_sha256(
    index: _CollectActivationIndex,
    *,
    reference: _CollectActivationReference,
    activation_file: VerifiedFile,
) -> str:
    official = index.official
    if official is None:
        raise ValueError("official alignment requires an official collect index")
    base = official.alignment_payloads.get(reference.artifact_name)
    if base is None:
        raise ValueError("official activation artifact lacks alignment metadata")
    source = io.BytesIO(activation_file.read_bytes())
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != {"activations", "call_index", "axis_coordinates"}:
            raise ValueError("official activation shard fields do not match the schema")
        activations = np.asarray(archive["activations"])
        call_index = np.asarray(archive["call_index"])
        axis_coordinates = np.asarray(archive["axis_coordinates"])
    retained = int(base["retained_vectors"])
    feature_dim = int(base["feature_dim"])
    coordinate_axes = tuple(base["coordinate_axis_names"])
    if (
        activations.ndim != 2
        or activations.shape != (retained, feature_dim)
        or np.dtype(activations.dtype).name != base["dtype"]
        or not np.isfinite(activations).all()
    ):
        raise ValueError("official activation bytes differ from index shape/dtype metadata")
    if (
        call_index.ndim != 1
        or call_index.shape[0] != retained
        or call_index.dtype.kind not in {"i", "u"}
        or np.any(call_index < 0)
    ):
        raise ValueError("official call_index coordinates are invalid")
    if (
        axis_coordinates.ndim != 2
        or axis_coordinates.shape != (retained, len(coordinate_axes))
        or axis_coordinates.dtype.kind not in {"i", "u"}
        or np.any(axis_coordinates < 0)
    ):
        raise ValueError("official axis_coordinates are invalid")
    activation_shapes = base["activation_shapes"]
    if int(call_index.max(initial=-1)) >= len(activation_shapes):
        raise ValueError("official call_index references an unknown forward call")
    for observed_call in np.unique(call_index):
        mask = call_index == observed_call
        bounds = activation_shapes[int(observed_call)][:-1]
        if len(bounds) != axis_coordinates.shape[1]:
            raise ValueError("official coordinate rank differs from activation shape metadata")
        if any(
            np.any(axis_coordinates[mask, axis] >= int(bound))
            for axis, bound in enumerate(bounds)
        ):
            raise ValueError("official axis coordinate exceeds its activation shape bound")
    payload = {
        **base,
        "call_index_sha256": _array_content_sha256(call_index),
        "axis_coordinates_sha256": _array_content_sha256(axis_coordinates),
    }
    return _canonical_json_sha256(payload)


def _array_content_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(
        header + b"\0" + contiguous.tobytes(order="C")
    ).hexdigest()


def _optional_positive_integer(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer when provided")
    return value


def load_activation_array(spec: Any, *, base_dir: Path | None = None) -> Tensor:
    """Load a finite activation matrix from ``.npy``, ``.npz``, or tensor ``.pt``."""

    key: str | None = None
    if isinstance(spec, Mapping):
        if "path" not in spec:
            raise ValueError("activation object must contain a path")
        path = Path(str(spec["path"]))
        key_value = spec.get("key")
        key = None if key_value is None else str(key_value)
    else:
        path = Path(str(spec))
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        values: Any = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key is None:
                if len(archive.files) != 1:
                    raise ValueError("npz activation input needs an explicit key")
                key = archive.files[0]
            values = archive[key]
    elif suffix in {".pt", ".pth"}:
        try:
            values = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older torch
            values = torch.load(path, map_location="cpu")
        if isinstance(values, Mapping):
            if key is None:
                tensor_keys = [name for name, value in values.items() if isinstance(value, Tensor)]
                if len(tensor_keys) != 1:
                    raise ValueError("tensor checkpoint needs an explicit activation key")
                key = tensor_keys[0]
            values = values[key]
    else:
        raise ValueError(f"unsupported activation file extension: {suffix!r}")
    return _as_activation_tensor(values)


def load_verified_activation_array(verified_file: Any, *, key: str | None = None) -> Tensor:
    """Load activation bytes through a ``VerifiedFile`` without reopening by path."""

    suffix = verified_file.path.suffix.lower()
    source = io.BytesIO(verified_file.read_bytes())
    if suffix == ".npy":
        values: Any = np.load(source, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            resolved_key = key
            if resolved_key is None:
                if len(archive.files) != 1:
                    raise ValueError("npz activation input needs an explicit key")
                resolved_key = archive.files[0]
            values = archive[resolved_key]
    elif suffix in {".pt", ".pth"}:
        try:
            values = torch.load(source, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older torch
            source.seek(0)
            values = torch.load(source, map_location="cpu")
        if isinstance(values, Mapping):
            resolved_key = key
            if resolved_key is None:
                tensor_keys = [
                    name for name, value in values.items() if isinstance(value, Tensor)
                ]
                if len(tensor_keys) != 1:
                    raise ValueError("tensor checkpoint needs an explicit activation key")
                resolved_key = tensor_keys[0]
            values = values[resolved_key]
    else:
        raise ValueError(f"unsupported activation file extension: {suffix!r}")
    return _as_activation_tensor(values)


def _as_activation_tensor(values: Tensor | np.ndarray | Sequence[Sequence[float]]) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).detach().cpu()
    if tensor.ndim < 2:
        raise ValueError("activations must have at least two dimensions")
    tensor = tensor.reshape(-1, tensor.shape[-1]).contiguous()
    if tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError("activations must be non-empty")
    if not torch.isfinite(tensor).all():
        raise ValueError("activations must be finite")
    return tensor


def _atomic_json_write(path: Path, payload: Any) -> None:
    def writer(handle: Any) -> None:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    _atomic_text_write(path, writer)


def _atomic_text_write(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


__all__ = [
    "MIN_HELDOUT_EXPLAINED_VARIANCE",
    "SUPPORTED_TOP_K",
    "BalancedConditionActivations",
    "DenseAutoencoder",
    "MeanRMSNormalizer",
    "PCARepresentation",
    "RepresentationTrainingResult",
    "TopKSparseAutoencoder",
    "balance_condition_activations",
    "build_autoencoder",
    "load_activation_array",
    "load_representation_checkpoint",
    "load_verified_activation_array",
    "load_verified_representation_checkpoint",
    "reconstruction_metrics",
    "representation_qualification",
    "run",
    "save_representation_checkpoint",
    "set_deterministic_seed",
    "train_autoencoder",
]
