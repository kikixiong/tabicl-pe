"""Shared, model-neutral interface for mechanism-analysis adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ContextManager


# Activation analysis treats the final tensor dimension as one learned vector.
# Most internal transformer sites call that axis ``embedding``.  TabICLv2's
# complete RowInteraction output instead concatenates its CLS tokens and names
# the resulting 512-wide axis ``row_representation``.  Keep those semantics
# distinct while sharing the same vector-sampling and representation pipeline.
ACTIVATION_VECTOR_AXIS_NAMES = frozenset({"embedding", "row_representation"})


@dataclass(frozen=True)
class ActivationSite:
    """A stable activation name and its documented tensor-axis semantics."""

    name: str
    axis_names: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class ActivationRecord:
    """One captured tensor with enough metadata to interpret it safely."""

    tensor: Any
    site: str
    axis_names: tuple[str, ...]
    shape: tuple[int, ...]
    model_sha: str
    checkpoint_sha: str
    preprocessing_view_id: str
    feature_group_map: tuple[tuple[int, ...], ...] | None = None


Intervention = Callable[[ActivationRecord], Any]
CaptureBuffer = Mapping[str, ActivationRecord]


class ModelAdapter(ABC):
    """Minimal boundary shared by TabICL and TabPFN model adapters."""

    @property
    @abstractmethod
    def model_family(self) -> str:
        """Return a portable model-family identifier."""

    @abstractmethod
    def load_model(
        self,
        checkpoint: Path,
        *,
        device: str,
        options: Mapping[str, Any] | None = None,
    ) -> Any:
        """Load an exact checkpoint without implicit downloading."""

    @abstractmethod
    def list_sites(self, model: Any) -> Mapping[str, ActivationSite]:
        """Return every supported capture/intervention site by stable name."""

    @abstractmethod
    def capture(
        self,
        model: Any,
        *,
        sites: Sequence[str],
        model_sha: str,
        checkpoint_sha: str,
        preprocessing_view_id: str,
        feature_group_map: tuple[tuple[int, ...], ...] | None = None,
    ) -> ContextManager[CaptureBuffer]:
        """Install temporary hooks and yield captured, axis-labelled records."""

    @abstractmethod
    def intervene(
        self,
        model: Any,
        interventions: Mapping[str, Intervention],
    ) -> ContextManager[None]:
        """Install reversible, scoped activation interventions."""

    @abstractmethod
    def predict(self, model: Any, batch: Any) -> Any:
        """Run the adapter's aligned prediction path for one prepared batch."""
