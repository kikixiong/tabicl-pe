"""Strict official TabPFN v2.6 inference for position-component ablations.

The checkpoint is loaded once through TabPFN's own low-level loader with
downloading disabled, then supplied to the public ``TabPFNClassifier`` as a
``ClassifierModelSpecs`` object.  All predictions still use the fitted public
scikit-learn estimator.  The four position conditions are scoped hooks on that
same fitted model; no condition refits or clones the estimator.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import inspect
from pathlib import Path
import threading
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from .adapters.tabpfn_v26 import PositionComponent, TabPFNV26Adapter
from .official_tabicl import RawTalentDataset
from .provenance import GitEvidence, VerifiedFile, verify_file, verify_git_tree


_COMPONENTS: tuple[PositionComponent, ...] = (
    "full",
    "weight",
    "bias",
    "none",
)
_PREDICTION_ORDER: tuple[PositionComponent, ...] = (
    "weight",
    "bias",
    "none",
    "full",
)
_CLASSIFICATION_TASKS = frozenset(
    {
        "binclass",
        "binary",
        "binary_classification",
        "classification",
        "multiclass",
        "multiclass_classification",
    }
)


@dataclass(frozen=True)
class TabPFNV26ConditionResult:
    """Official probabilities and metrics for one position component."""

    component: PositionComponent
    probabilities: np.ndarray
    accuracy: float
    log_loss: float
    n_samples: int


@dataclass(frozen=True)
class OfficialTabPFNV26Result:
    """Paired four-condition result from one fitted official classifier."""

    dataset_name: str
    split: Literal["val", "test"]
    classes: np.ndarray
    native_probabilities: np.ndarray
    conditions: Mapping[PositionComponent, TabPFNV26ConditionResult]
    model_code_sha: str
    loaded_checkpoint_sha256: str
    seed: int
    exact_full_native_verified: bool
    policies_restored_verified: bool


@dataclass(frozen=True)
class _TabPFNRuntime:
    classifier_class: type[Any]
    model_specs_class: type[Any]
    load_model_criterion_config: Callable[..., Any]


def _load_tabpfn_runtime() -> _TabPFNRuntime:
    try:
        from tabpfn import TabPFNClassifier
        from tabpfn.base import ClassifierModelSpecs
        from tabpfn.model_loading import load_model_criterion_config
    except ImportError as error:
        raise ImportError(
            "official TabPFN v2.6 inference requires the public tabpfn 7.1.1 "
            "package and its scikit-learn dependencies"
        ) from error
    return _TabPFNRuntime(
        classifier_class=TabPFNClassifier,
        model_specs_class=ClassifierModelSpecs,
        load_model_criterion_config=load_model_criterion_config,
    )


class OfficialTabPFNV26Driver:
    """Evaluate position components on one immutable, fitted TabPFN estimator."""

    def __init__(
        self,
        estimator: Any,
        dataset: RawTalentDataset,
        *,
        loaded_model: Any,
        adapter: TabPFNV26Adapter,
        model_code: GitEvidence,
        checkpoint: VerifiedFile,
        seed: int,
    ) -> None:
        self.estimator = estimator
        self.dataset = dataset
        self.adapter = adapter
        self.model_code_sha = model_code.head_sha
        self.loaded_checkpoint_sha256 = checkpoint.digest.sha256
        self.seed = seed
        self._loaded_model = loaded_model
        self._model_code = model_code
        self._checkpoint = checkpoint
        self._lock = threading.RLock()
        self._projection = _position_projection(loaded_model)
        self._hook_state = _hook_state(self._projection)
        self._classes = _validated_classes(estimator)
        self._validate_fitted_model()

    def evaluate(
        self, split: Literal["val", "test"] = "test"
    ) -> OfficialTabPFNV26Result:
        """Run native plus Wp+b/Wp/b/zero on one validation or test split."""

        if split not in ("val", "test"):
            raise ValueError("TabPFN evaluation split must be 'val' or 'test'")
        selected = getattr(self.dataset, split)
        labels = _target_vector(selected.y)
        with self._lock:
            self._assert_inputs_unchanged()
            self._validate_fitted_model()
            native = self._predict(selected.X, expected_rows=labels.shape[0])
            measured: dict[PositionComponent, np.ndarray] = {}
            for component in _PREDICTION_ORDER:
                with self._restored_policy(component):
                    measured[component] = self._predict(
                        selected.X,
                        expected_rows=labels.shape[0],
                    )
            if not _arrays_exact(measured["full"], native):
                raise RuntimeError(
                    "TabPFN full position policy is not byte-exact to native "
                    "official predict_proba"
                )
            self._validate_fitted_model()
            self._assert_inputs_unchanged()

        conditions = {
            component: _condition_result(
                component,
                measured[component],
                labels,
                self._classes,
            )
            for component in _COMPONENTS
        }
        return OfficialTabPFNV26Result(
            dataset_name=self.dataset.name,
            split=split,
            classes=_frozen_array(self._classes),
            native_probabilities=_frozen_array(native),
            conditions=MappingProxyType(conditions),
            model_code_sha=self.model_code_sha,
            loaded_checkpoint_sha256=self.loaded_checkpoint_sha256,
            seed=self.seed,
            exact_full_native_verified=True,
            policies_restored_verified=True,
        )

    def _predict(self, X: Any, *, expected_rows: int) -> np.ndarray:
        probabilities = np.asarray(self.estimator.predict_proba(deepcopy(X)))
        return _validated_probabilities(
            probabilities,
            expected_rows=expected_rows,
            expected_classes=self._classes.shape[0],
        )

    @contextmanager
    def _restored_policy(self, component: PositionComponent):
        before = _hook_state(self._projection)
        if before != self._hook_state:
            raise RuntimeError("TabPFN position projection hooks changed before policy")
        try:
            with self.adapter.position_policy(self.estimator, component):
                yield
        finally:
            after = _hook_state(self._projection)
            if after != before or after != self._hook_state:
                raise RuntimeError(
                    f"TabPFN {component} position policy did not restore all hooks"
                )

    def _validate_fitted_model(self) -> None:
        models = getattr(self.estimator, "models_", None)
        if not isinstance(models, (list, tuple)) or len(models) != 1:
            raise RuntimeError("official TabPFN classifier must retain exactly one model")
        if models[0] is not self._loaded_model:
            raise RuntimeError("official TabPFN classifier replaced the loaded model")
        if _position_projection(models[0]) is not self._projection:
            raise RuntimeError("TabPFN position projection was replaced")
        if _hook_state(self._projection) != self._hook_state:
            raise RuntimeError("TabPFN position policy was not fully restored")
        if int(getattr(self.estimator, "n_estimators", -1)) != 1:
            raise RuntimeError("official TabPFN classifier changed n_estimators")
        if getattr(self.estimator, "random_state", None) != self.seed:
            raise RuntimeError("official TabPFN classifier changed its fixed seed")
        observed_classes = _validated_classes(self.estimator)
        if not _arrays_exact(observed_classes, self._classes):
            raise RuntimeError("official TabPFN classifier classes changed")

    def _assert_inputs_unchanged(self) -> None:
        self._checkpoint.assert_unchanged()
        self._model_code.assert_unchanged()


def fit_official_tabpfn_v26_driver(
    dataset: RawTalentDataset,
    checkpoint: str | Path,
    *,
    checkpoint_sha256: str,
    model_sha: str,
    expected_source_root: str | Path,
    seed: int,
    device: str = "cpu",
) -> OfficialTabPFNV26Driver:
    """Load an exact local v2.6 checkpoint and fit only on TALENT train rows.

    The public classifier receives an already-loaded ``ClassifierModelSpecs``.
    This is the only supported route because TabPFN 7.1.1 otherwise enables
    downloading for a missing path internally.
    """

    _validate_dataset(dataset)
    model_sha = _hex_digest("model_sha", model_sha, length=40)
    checkpoint_sha256 = _hex_digest(
        "checkpoint_sha256", checkpoint_sha256, length=64
    )
    seed = _seed(seed)
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string")

    verified_checkpoint = verify_file(
        checkpoint,
        expected_sha256=checkpoint_sha256,
    )
    source_evidence = verify_git_tree(
        expected_source_root,
        expected_sha=model_sha,
    )
    if source_evidence.evidence_level != "strict":
        raise RuntimeError("TabPFN source evidence must be a clean Git checkout")

    runtime = _load_tabpfn_runtime()
    _require_source_object(
        "TabPFNClassifier", runtime.classifier_class, source_evidence
    )
    _require_source_object(
        "ClassifierModelSpecs", runtime.model_specs_class, source_evidence
    )
    _require_source_object(
        "load_model_criterion_config",
        runtime.load_model_criterion_config,
        source_evidence,
    )

    loaded = runtime.load_model_criterion_config(
        model_path=verified_checkpoint.path,
        check_bar_distribution_criterion=False,
        cache_trainset_representation=False,
        which="classifier",
        version="v2.6",
        download_if_not_exists=False,
    )
    try:
        models, _criterion, configs, inference_config = loaded
    except (TypeError, ValueError) as error:
        raise RuntimeError("TabPFN v2.6 loader returned an invalid result") from error
    if len(models) != 1 or len(configs) != 1:
        raise RuntimeError("exact TabPFN v2.6 checkpoint must load one model/config")
    loaded_model = models[0]
    config = configs[0]
    if (
        type(loaded_model).__name__ != "TabPFNV2p6"
        or getattr(config, "name", None) != "TabPFN-v2.6"
    ):
        raise RuntimeError(
            "local checkpoint did not resolve to the exact TabPFN-v2.6 architecture"
        )
    _require_source_object("TabPFNV2p6", loaded_model.__class__, source_evidence)
    verified_checkpoint.assert_unchanged()

    model_specs = runtime.model_specs_class(
        loaded_model,
        config,
        inference_config,
    )
    categorical_indices = list(
        range(
            dataset.n_numeric_features,
            dataset.n_numeric_features + dataset.n_categorical_features,
        )
    )
    classifier = runtime.classifier_class(
        n_estimators=1,
        categorical_features_indices=categorical_indices,
        model_path=model_specs,
        device=device,
        ignore_pretraining_limits=False,
        inference_precision="auto",
        fit_mode="fit_preprocessors",
        memory_saving_mode=False,
        random_state=seed,
        n_jobs=None,
        n_preprocessing_jobs=1,
        eval_metric=None,
        tuning_config=None,
    )
    classifier.fit(deepcopy(dataset.train.X), np.array(dataset.train.y, copy=True))
    models_after_fit = getattr(classifier, "models_", None)
    if (
        not isinstance(models_after_fit, (list, tuple))
        or len(models_after_fit) != 1
        or models_after_fit[0] is not loaded_model
    ):
        raise RuntimeError(
            "official TabPFN classifier did not retain the exact preloaded checkpoint"
        )
    verified_checkpoint.assert_unchanged()
    source_evidence.assert_unchanged()
    return OfficialTabPFNV26Driver(
        classifier,
        dataset,
        loaded_model=loaded_model,
        adapter=TabPFNV26Adapter(),
        model_code=source_evidence,
        checkpoint=verified_checkpoint,
        seed=seed,
    )


def _validate_dataset(dataset: RawTalentDataset) -> None:
    if not isinstance(dataset, RawTalentDataset):
        raise TypeError("dataset must be a loaded RawTalentDataset")
    if str(dataset.task_type).strip().lower() not in _CLASSIFICATION_TASKS:
        raise ValueError("official TabPFN driver supports classification only")
    numeric = int(dataset.n_numeric_features)
    categorical = int(dataset.n_categorical_features)
    if numeric < 0 or categorical < 0 or numeric + categorical < 1:
        raise ValueError("RawTalentDataset has invalid feature counts")
    expected_features = numeric + categorical
    for name in ("train", "val", "test"):
        split = getattr(dataset, name)
        shape = getattr(split.X, "shape", None)
        labels = _target_vector(split.y)
        if (
            shape is None
            or len(shape) != 2
            or int(shape[0]) != labels.shape[0]
            or int(shape[1]) != expected_features
            or labels.shape[0] == 0
        ):
            raise ValueError(f"RawTalentDataset {name} split shape is inconsistent")


def _require_source_object(
    label: str, value: Any, evidence: GitEvidence
) -> None:
    try:
        source = Path(inspect.getfile(value)).resolve(strict=True)
    except (OSError, TypeError) as error:
        raise RuntimeError(f"cannot locate imported {label} source") from error
    if not source.is_relative_to(evidence.root):
        raise RuntimeError(f"imported {label} is outside expected_source_root")


def _position_projection(model: Any) -> Any:
    projection = getattr(model, "feature_positional_embedding_embeddings", None)
    if projection is None or not callable(projection):
        raise RuntimeError("TabPFN v2.6 position projection is unavailable")
    if not hasattr(projection, "_forward_hooks"):
        raise RuntimeError("TabPFN v2.6 position projection is not a torch module")
    return projection


def _hook_state(projection: Any) -> tuple[tuple[Any, int], ...]:
    hooks = getattr(projection, "_forward_hooks", None)
    if hooks is None or not hasattr(hooks, "items"):
        raise RuntimeError("cannot verify TabPFN position hook restoration")
    return tuple((key, id(hook)) for key, hook in hooks.items())


def _validated_classes(estimator: Any) -> np.ndarray:
    if not hasattr(estimator, "classes_"):
        raise RuntimeError("official TabPFN classifier is not fitted")
    classes = np.asarray(estimator.classes_)
    if classes.ndim != 1 or classes.shape[0] < 2:
        raise RuntimeError("official TabPFN classifier returned invalid classes")
    tokens = tuple(_label_token(value) for value in classes)
    if len(set(tokens)) != len(tokens):
        raise RuntimeError("official TabPFN classifier classes are not unique")
    return np.array(classes, copy=True)


def _validated_probabilities(
    probabilities: np.ndarray,
    *,
    expected_rows: int,
    expected_classes: int,
) -> np.ndarray:
    if probabilities.shape != (expected_rows, expected_classes):
        raise RuntimeError("official TabPFN predict_proba returned the wrong shape")
    if not np.issubdtype(probabilities.dtype, np.floating):
        raise RuntimeError("official TabPFN probabilities must be floating point")
    values = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("official TabPFN probabilities contain non-finite values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise RuntimeError("official TabPFN probabilities are outside [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-6, atol=1e-7):
        raise RuntimeError("official TabPFN probability rows do not sum to one")
    return np.array(probabilities, copy=True)


def _condition_result(
    component: PositionComponent,
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
) -> TabPFNV26ConditionResult:
    class_indices = {_label_token(value): index for index, value in enumerate(classes)}
    try:
        target = np.asarray(
            [class_indices[_label_token(value)] for value in labels],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("evaluation split contains a class absent from train") from error
    values = np.asarray(probabilities, dtype=np.float64)
    selected = values[np.arange(target.shape[0]), target]
    return TabPFNV26ConditionResult(
        component=component,
        probabilities=_frozen_array(probabilities),
        accuracy=float(np.mean(np.argmax(values, axis=1) == target)),
        log_loss=float(
            -np.log(np.clip(selected, np.finfo(np.float64).tiny, 1.0)).mean()
        ),
        n_samples=int(target.shape[0]),
    )


def _target_vector(values: Any) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1:
        raise ValueError("TALENT labels must be a vector")
    return labels


def _label_token(value: Any) -> tuple[str, str]:
    scalar = value.item() if isinstance(value, np.generic) else value
    return type(scalar).__qualname__, repr(scalar)


def _arrays_exact(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.array_equal(left, right)
    )


def _frozen_array(values: np.ndarray) -> np.ndarray:
    frozen = np.array(values, copy=True)
    frozen.setflags(write=False)
    return frozen


def _hex_digest(name: str, value: str, *, length: int) -> str:
    normalized = str(value).lower()
    if len(normalized) != length or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a {length}-character hexadecimal digest")
    return normalized


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(value)
    if seed < 0 or seed >= 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    return seed


__all__ = [
    "OfficialTabPFNV26Driver",
    "OfficialTabPFNV26Result",
    "TabPFNV26ConditionResult",
    "fit_official_tabpfn_v26_driver",
]
