"""In-memory TabICLv2 classification evaluation for Row-RoPE ablations.

This module deliberately stops at the raw numerical model boundary.  Dataset
preprocessing remains the caller's responsibility, and no function here writes files
or downloads weights.  The returned objects are suitable inputs for the later
``collect`` and ``ablate`` command layers.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .adapters.base import ActivationRecord
from .adapters.tabicl import FrequencyBand, TabICLAdapter


@dataclass(frozen=True)
class TabICLRawBatch(Mapping[str, Any]):
    """One numerical table in the exact raw shape consumed by ``TabICL.forward``.

    Iterating this object exposes only model inputs.  Evaluation labels and the
    original class vocabulary remain metadata and are never passed to the model.
    """

    X: torch.Tensor
    y_train: torch.Tensor
    y_test: np.ndarray
    classes: tuple[Any, ...]

    def __getitem__(self, key: str) -> Any:
        if key == "X":
            return self.X
        if key == "y_train":
            return self.y_train
        if key == "return_logits":
            return False
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("X", "y_train", "return_logits"))

    def __len__(self) -> int:
        return 3

    @property
    def train_size(self) -> int:
        return int(self.y_train.shape[1])

    @property
    def test_size(self) -> int:
        return int(self.y_test.shape[0])

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def decode(self, encoded: Any) -> np.ndarray:
        """Map contiguous integer predictions back to the caller's label values."""

        indices = np.asarray(encoded)
        if indices.dtype.kind not in "iu":
            raise TypeError("encoded class indices must be integers")
        if np.any(indices < 0) or np.any(indices >= self.n_classes):
            raise ValueError("encoded class index is outside the class vocabulary")
        return np.asarray(self.classes, dtype=object)[indices]


@dataclass(frozen=True)
class RopeCondition:
    """One reversible Row-RoPE policy passed directly to ``TabICLAdapter``."""

    name: str
    blocks: tuple[int, ...] | None = None
    rotate_queries: bool = True
    rotate_keys: bool = True
    phase_strength: float = 1.0
    frequency_band: FrequencyBand = "all"
    heads: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("condition name must be a non-empty string")
        if self.blocks is not None:
            blocks = tuple(self.blocks)
            if any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in blocks
            ):
                raise TypeError("condition blocks must be integer indices")
            object.__setattr__(self, "blocks", blocks)
        if self.heads is not None:
            heads = tuple(self.heads)
            if any(
                isinstance(index, bool) or not isinstance(index, int) for index in heads
            ):
                raise TypeError("condition heads must be integer indices")
            object.__setattr__(self, "heads", heads)
        if not isinstance(self.rotate_queries, bool) or not isinstance(
            self.rotate_keys, bool
        ):
            raise TypeError("rotate_queries and rotate_keys must be booleans")
        if isinstance(self.phase_strength, bool) or not isinstance(
            self.phase_strength, (int, float)
        ):
            raise TypeError("phase_strength must be a number")
        if not np.isfinite(self.phase_strength) or self.phase_strength < 0:
            raise ValueError("phase_strength must be finite and non-negative")
        object.__setattr__(self, "phase_strength", float(self.phase_strength))
        band = self.frequency_band
        if isinstance(band, list):
            band = tuple(band)
            object.__setattr__(self, "frequency_band", band)
        if band not in ("all", "high", "low") and not (
            isinstance(band, tuple)
            and len(band) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool) for value in band
            )
        ):
            raise ValueError(
                "frequency_band must be all/high/low or a (start, stop) pair"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RopeCondition:
        """Build a condition from a JSON-compatible mapping."""

        allowed = {
            "name",
            "blocks",
            "rotate_queries",
            "rotate_keys",
            "phase_strength",
            "frequency_band",
            "heads",
        }
        unknown = value.keys() - allowed
        if unknown:
            raise ValueError(f"unsupported RoPE condition fields: {sorted(unknown)}")
        if "name" not in value:
            raise ValueError("RoPE condition requires a name")
        return cls(**dict(value))

    @property
    def is_full_policy(self) -> bool:
        """Whether this policy must be an exact no-op relative to baseline."""

        return (
            self.rotate_queries
            and self.rotate_keys
            and self.phase_strength == 1.0
            and self.frequency_band == "all"
            and self.heads is None
        )

    def policy_kwargs(self) -> dict[str, Any]:
        return {
            "blocks": self.blocks,
            "rotate_queries": self.rotate_queries,
            "rotate_keys": self.rotate_keys,
            "phase_strength": self.phase_strength,
            "frequency_band": self.frequency_band,
            "heads": self.heads,
        }


@dataclass(frozen=True)
class TabICLEvaluationResult:
    """Predictions, scalar metrics, and optional captured activations."""

    condition: str
    probabilities: np.ndarray
    accuracy: float
    log_loss: float
    activations: Mapping[str, ActivationRecord] | None = None


def default_rope_conditions(
    *,
    blocks: Sequence[int] | None = None,
    phase_strength: float = 0.5,
    head_selection: Sequence[int] | None = (0,),
) -> tuple[RopeCondition, ...]:
    """Return the standard mechanism-ablation roster (baseline is added separately)."""

    selected_blocks = None if blocks is None else tuple(blocks)
    selected_heads = None if head_selection is None else tuple(head_selection)
    conditions = [
        RopeCondition("full", blocks=selected_blocks),
        RopeCondition(
            "rope_off",
            blocks=selected_blocks,
            rotate_queries=False,
            rotate_keys=False,
        ),
        RopeCondition(
            "query_only",
            blocks=selected_blocks,
            rotate_queries=True,
            rotate_keys=False,
        ),
        RopeCondition(
            "key_only",
            blocks=selected_blocks,
            rotate_queries=False,
            rotate_keys=True,
        ),
        RopeCondition(
            f"phase_strength_{float(phase_strength):g}",
            blocks=selected_blocks,
            phase_strength=phase_strength,
        ),
        RopeCondition("frequency_high", blocks=selected_blocks, frequency_band="high"),
        RopeCondition("frequency_low", blocks=selected_blocks, frequency_band="low"),
    ]
    if selected_heads is not None:
        label = "_".join(str(index) for index in selected_heads)
        conditions.append(
            RopeCondition(
                f"heads_{label}", blocks=selected_blocks, heads=selected_heads
            )
        )
    return tuple(conditions)


def _as_two_dimensional_features(name: str, values: Any) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a numerical numpy-compatible array"
        ) from error
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (rows, features)")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one row and feature")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_one_dimensional_labels(name: str, values: Any, rows: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] != rows:
        raise ValueError(f"{name} must have shape ({rows},)")
    for value in array.tolist():
        if value is None:
            raise ValueError(f"{name} must not contain missing labels")
        try:
            if bool(value != value):
                raise ValueError(f"{name} must not contain missing labels")
        except (TypeError, ValueError):
            raise ValueError(f"{name} labels must have scalar equality") from None
        if isinstance(value, (float, complex)) and not np.isfinite(value):
            raise ValueError(f"{name} must contain only finite labels")
    return array


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def prepare_tabicl_raw_batch(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    *,
    device: str | torch.device = "cpu",
    max_classes: int | None = None,
) -> TabICLRawBatch:
    """Convert one numerical classification split into a raw TabICLv2 batch.

    The class vocabulary is learned from training labels only.  It is sorted using
    NumPy's deterministic ``unique`` order and mapped to contiguous ``0..C-1`` values.
    Every test class must therefore already occur in the training split.
    """

    train_features = _as_two_dimensional_features("X_train", X_train)
    test_features = _as_two_dimensional_features("X_test", X_test)
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("X_train and X_test must have the same number of features")
    train_labels = _as_one_dimensional_labels(
        "y_train", y_train, train_features.shape[0]
    )
    test_labels = _as_one_dimensional_labels("y_test", y_test, test_features.shape[0])
    try:
        unique = np.unique(train_labels)
    except TypeError as error:
        raise ValueError(
            "training labels must be mutually comparable scalar values"
        ) from error
    classes = tuple(_python_scalar(value) for value in unique)
    if len(classes) < 2:
        raise ValueError("classification requires at least two training classes")
    if max_classes is not None:
        if isinstance(max_classes, bool) or not isinstance(max_classes, int):
            raise TypeError("max_classes must be an integer")
        if max_classes < 1:
            raise ValueError("max_classes must be positive for classification")
        if len(classes) > max_classes:
            raise ValueError(
                f"dataset has {len(classes)} classes but model supports {max_classes}"
            )

    try:
        class_to_index = {label: index for index, label in enumerate(classes)}
        encoded_train = np.asarray(
            [class_to_index[_python_scalar(value)] for value in train_labels],
            dtype=np.int64,
        )
        encoded_test = np.asarray(
            [class_to_index[_python_scalar(value)] for value in test_labels],
            dtype=np.int64,
        )
    except (KeyError, TypeError) as error:
        missing = _python_scalar(error.args[0]) if error.args else "unknown"
        raise ValueError(
            f"test labels contain class {missing!r} absent from the training split"
        ) from None

    combined = np.concatenate((train_features, test_features), axis=0)
    return TabICLRawBatch(
        X=torch.as_tensor(combined, dtype=torch.float32, device=device).unsqueeze(0),
        y_train=torch.as_tensor(
            encoded_train, dtype=torch.long, device=device
        ).unsqueeze(0),
        y_test=encoded_test,
        classes=classes,
    )


def _model_max_classes(model: Any) -> int:
    raw = getattr(model, "model_", model)
    value = getattr(raw, "max_classes", None)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("TabICL model must expose an integer max_classes")
    if value < 1:
        raise ValueError("TabICL evaluation requires a classification checkpoint")
    return value


def _probability_array(output: Any, batch: TabICLRawBatch) -> np.ndarray:
    if isinstance(output, torch.Tensor):
        values = output.detach().to(device="cpu", dtype=torch.float64).numpy()
    else:
        values = np.asarray(output, dtype=np.float64)
    expected = (1, batch.test_size, batch.n_classes)
    if values.shape != expected:
        raise ValueError(
            f"TabICL probabilities must have shape {expected}, got {values.shape}"
        )
    values = values[0].copy()
    if not np.isfinite(values).all():
        raise ValueError("TabICL probabilities must contain only finite values")
    tolerance = 1e-6
    if np.any(values < -tolerance) or np.any(values > 1 + tolerance):
        raise ValueError("TabICL returned values outside the probability interval")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=tolerance):
        raise ValueError("TabICL probability rows must sum to one")
    return values


def _metrics(probabilities: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    accuracy = float(np.mean(np.argmax(probabilities, axis=1) == targets))
    true_probabilities = probabilities[np.arange(targets.shape[0]), targets]
    log_loss = float(-np.log(np.clip(true_probabilities, 1e-15, 1.0)).mean())
    return accuracy, log_loss


def _model_modules(model: Any) -> tuple[Any, ...]:
    """Return wrapper and raw modules once, preserving mixed train/eval states."""

    candidates = (model, getattr(model, "model_", model))
    modules: list[Any] = []
    seen: set[int] = set()
    for candidate in candidates:
        contained = (
            candidate.modules() if hasattr(candidate, "modules") else (candidate,)
        )
        for module in contained:
            if id(module) not in seen and hasattr(module, "training"):
                modules.append(module)
                seen.add(id(module))
    return tuple(modules)


@contextmanager
def _preserve_training_states(model: Any) -> Iterator[None]:
    modules = _model_modules(model)
    states = tuple(bool(module.training) for module in modules)
    try:
        yield None
    finally:
        # Assign directly so a parent's recursive ``train`` call cannot erase a
        # deliberately different child state.
        for module, state in zip(modules, states, strict=True):
            module.training = state


def _coerce_conditions(
    conditions: Sequence[RopeCondition | Mapping[str, Any]] | None,
) -> tuple[RopeCondition, ...]:
    if conditions is None:
        return default_rope_conditions()
    resolved = tuple(
        value if isinstance(value, RopeCondition) else RopeCondition.from_mapping(value)
        for value in conditions
    )
    names = [condition.name for condition in resolved]
    if "baseline" in names:
        raise ValueError("baseline is reserved and is always evaluated automatically")
    if len(set(names)) != len(names):
        raise ValueError("RoPE condition names must be unique")
    return resolved


def evaluate_tabicl_rope_conditions(
    adapter: TabICLAdapter,
    model: Any,
    batch: TabICLRawBatch,
    *,
    conditions: Sequence[RopeCondition | Mapping[str, Any]] | None = None,
    capture_sites: Sequence[str] = (),
    model_sha: str = "",
    checkpoint_sha: str = "",
    preprocessing_view_id: str = "raw",
    feature_group_map: tuple[tuple[int, ...], ...] | None = None,
) -> dict[str, TabICLEvaluationResult]:
    """Evaluate baseline and scoped Row-RoPE policies without changing the model.

    An explicit full policy (queries and keys, strength one, all frequencies and
    heads) is checked with exact array equality against baseline.  A mismatch raises
    instead of silently admitting misaligned inference into a causal comparison.
    """

    if not isinstance(batch, TabICLRawBatch):
        raise TypeError("batch must be produced by prepare_tabicl_raw_batch")
    max_classes = _model_max_classes(model)
    if batch.n_classes > max_classes:
        raise ValueError(
            f"dataset has {batch.n_classes} classes but model supports {max_classes}"
        )
    requested_sites = tuple(capture_sites)
    if len(set(requested_sites)) != len(requested_sites):
        raise ValueError("capture_sites must not contain duplicates")
    resolved_conditions = _coerce_conditions(conditions)
    results: dict[str, TabICLEvaluationResult] = {}

    def evaluate_one(condition: RopeCondition | None) -> TabICLEvaluationResult:
        name = "baseline" if condition is None else condition.name
        policy = (
            nullcontext()
            if condition is None
            else adapter.rope_policy(model, **condition.policy_kwargs())
        )
        capture = (
            adapter.capture(
                model,
                sites=requested_sites,
                model_sha=model_sha,
                checkpoint_sha=checkpoint_sha,
                preprocessing_view_id=preprocessing_view_id,
                feature_group_map=feature_group_map,
            )
            if requested_sites
            else nullcontext(None)
        )
        with policy:
            with capture as captured:
                output = adapter.predict(model, batch)
        probabilities = _probability_array(output, batch)
        accuracy, log_loss = _metrics(probabilities, batch.y_test)
        activations = None if captured is None else dict(captured)
        if activations is not None:
            missing = sorted(set(requested_sites) - activations.keys())
            if missing:
                raise RuntimeError(f"capture sites produced no activation: {missing}")
            if any(
                not isinstance(record, ActivationRecord)
                for record in activations.values()
            ):
                raise TypeError(
                    "adapter capture values must be ActivationRecord objects"
                )
        return TabICLEvaluationResult(
            condition=name,
            probabilities=probabilities,
            accuracy=accuracy,
            log_loss=log_loss,
            activations=activations,
        )

    with _preserve_training_states(model):
        results["baseline"] = evaluate_one(None)
        for condition in resolved_conditions:
            result = evaluate_one(condition)
            if condition.is_full_policy and not np.array_equal(
                result.probabilities, results["baseline"].probabilities
            ):
                raise RuntimeError(
                    f"full RoPE condition {condition.name!r} is not exactly equal to baseline"
                )
            results[condition.name] = result
    return results


__all__ = [
    "RopeCondition",
    "TabICLEvaluationResult",
    "TabICLRawBatch",
    "default_rope_conditions",
    "evaluate_tabicl_rope_conditions",
    "prepare_tabicl_raw_batch",
]
