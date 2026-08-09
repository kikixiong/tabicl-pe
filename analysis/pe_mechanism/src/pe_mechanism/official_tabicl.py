"""Official TabICL preprocessing and ensemble inference with scoped hooks.

This module deliberately delegates prediction to a fitted
``tabicl.TabICLClassifier``.  It does not reproduce the classifier's numerical
preprocessing, ensemble construction, class-unshuffle, or aggregation.  Instead,
it temporarily wraps the *same* raw model's ``forward`` method so activation
hooks run inside every official, cache-free raw-model call.

The wrapper is intentionally fail-closed.  Cached inference, many-class
recursion, all-missing feature masks, unsupported feature grouping, and any
drift from the fitted classifier's expected raw-call schedule are rejected.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import inspect
import json
from pathlib import Path
import threading
from typing import Any

import numpy as np

from .adapters.base import ActivationRecord, ModelAdapter
from .adapters.tabicl import TabICLAdapter, same_feature_group_map
from .provenance import GitEvidence, verify_file, verify_git_tree


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
_SPLITS = ("train", "val", "test")
_ACTIVE_MODEL_IDS: set[int] = set()
_ACTIVE_LOCK = threading.Lock()
_TORCH_THREAD_LOCK = threading.RLock()
OFFICIAL_INFERENCE_PROTOCOL = "official-tabicl-sklearn-raw-cache-free-v1"


def official_inference_contract_sha256(
    model_code_sha: str, estimator_options: Mapping[str, Any]
) -> str:
    """Hash the single canonical official-inference contract.

    The evaluation split is intentionally absent: a representation selected on
    validation rows can be applied to held-out test rows without changing the
    fitted estimator or its inference semantics.
    """

    _require_hex_digest("model_code_sha", model_code_sha, length=40)
    if not isinstance(estimator_options, Mapping):
        raise TypeError("estimator_options must be a mapping")
    payload = {
        "protocol": OFFICIAL_INFERENCE_PROTOCOL,
        "model_code_sha": model_code_sha,
        "fit_context": "train",
        "estimator_options": dict(estimator_options),
        "locked_options": {
            "allow_auto_download": False,
            "feature_group": "same",
            "kv_cache": False,
            "support_many_classes": False,
        },
    }
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "estimator_options must contain finite JSON-compatible values"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RawTalentSplit:
    """One untouched TALENT split in a mixed-type DataFrame."""

    X: Any
    y: np.ndarray


@dataclass(frozen=True)
class RawTalentDataset:
    """Raw TALENT train/validation/test data before TabICL preprocessing."""

    name: str
    task_type: str
    train: RawTalentSplit
    val: RawTalentSplit
    test: RawTalentSplit
    n_numeric_features: int
    n_categorical_features: int
    info_sha256: str
    input_sha256: Mapping[str, str]


@dataclass(frozen=True)
class OfficialForwardMetadata:
    """Sidecar metadata for one official raw-model call.

    ``post_filter_feature_group_maps`` has one entry per table on the raw
    tensor's table axis.  Its integers refer to the coordinate system *after*
    official mixed-type encoding and constant-feature filtering, but before the
    current ensemble view's feature shuffle.  Mapping those coordinates back to
    original DataFrame column names is deliberately outside this driver.
    """

    call_index: int
    norm_method: str
    norm_view_indices: tuple[int, ...]
    ensemble_indices: tuple[int, ...]
    feature_shuffles: tuple[tuple[int, ...], ...]
    class_shuffles: tuple[tuple[int, ...], ...]
    raw_input_shape: tuple[int, int, int]
    train_size: int
    preprocessing_view_id: str
    post_filter_feature_group_maps: tuple[
        tuple[tuple[int, ...], ...], ...
    ]
    feature_coordinate_system: str = (
        "official-encoded-after-constant-filter-before-view-shuffle"
    )


@dataclass(frozen=True)
class OfficialForwardCapture:
    """Captured activations and table-axis metadata for one raw call."""

    metadata: OfficialForwardMetadata
    activations: Mapping[str, ActivationRecord]


@dataclass(frozen=True)
class OfficialInferenceMetrics:
    """Small, dependency-free classification sanity metrics."""

    accuracy: float
    log_loss: float
    n_samples: int


@dataclass(frozen=True)
class OfficialInferenceResult:
    """Official probabilities, paired baseline, and optional hook records."""

    probabilities: np.ndarray
    baseline_probabilities: np.ndarray
    classes: np.ndarray
    forward_calls: tuple[OfficialForwardCapture, ...]
    metrics: OfficialInferenceMetrics | None
    exact_baseline_verified: bool
    source_evidence_level: str


OfficialIntervention = Callable[[ActivationRecord, OfficialForwardMetadata], Any]


@dataclass(frozen=True)
class _ExpectedForward:
    norm_method: str
    norm_view_indices: tuple[int, ...]
    ensemble_indices: tuple[int, ...]
    feature_shuffles: tuple[tuple[int, ...], ...]
    class_shuffles: tuple[tuple[int, ...], ...]


class OfficialTabICLDriver:
    """Instrument a fitted classifier without replacing official inference."""

    def __init__(
        self,
        estimator: Any,
        *,
        adapter: ModelAdapter | None = None,
        model_sha: str,
        checkpoint_sha: str,
        fit_context: str = "caller-provided-train",
        _source_evidence: GitEvidence | None = None,
    ) -> None:
        _require_hex_digest("model_sha", model_sha, length=40)
        _require_hex_digest("checkpoint_sha", checkpoint_sha, length=64)
        if not isinstance(fit_context, str) or not fit_context.strip():
            raise ValueError("fit_context must be a non-empty string")
        if _source_evidence is not None:
            if not isinstance(_source_evidence, GitEvidence):
                raise TypeError("strict source evidence must be measured GitEvidence")
            if (
                _source_evidence.evidence_level != "strict"
                or _source_evidence.head_sha != model_sha
            ):
                raise ValueError(
                    "strict source evidence must be clean and match model_sha"
                )
            _source_evidence.assert_unchanged()
        self.estimator = estimator
        self.adapter = adapter or TabICLAdapter()
        self.model_sha = model_sha
        self.checkpoint_sha = checkpoint_sha
        self.fit_context = fit_context
        self.source_evidence_level = (
            "strict" if _source_evidence is not None else "exploratory_unverified"
        )
        self._source_evidence = _source_evidence
        self._raw_model = _validate_fitted_estimator(estimator)
        self._raw_model_id = id(self._raw_model)

    def predict_proba(
        self,
        X: Any,
        *,
        y: Any | None = None,
        sites: Sequence[str] = (),
        interventions: Mapping[str, OfficialIntervention] | None = None,
        require_exact_baseline: bool | None = None,
    ) -> OfficialInferenceResult:
        """Run the unchanged official prediction path with optional raw hooks.

        Captures observe post-intervention activations when the same site is both
        captured and edited.  A separate, uninstrumented official prediction is
        paired with every instrumented call.  Capture-only runs require byte-exact
        probabilities by default.
        """

        raw = self._validated_raw_model()
        with _exclusive_model(raw):
            return self._predict_proba_exclusive(
                X,
                y=y,
                sites=sites,
                interventions=interventions,
                require_exact_baseline=require_exact_baseline,
            )

    @contextmanager
    def paired_session(self) -> Iterator[OfficialTabICLSession]:
        """Hold model exclusivity across several RNG-paired conditions."""

        raw = self._validated_raw_model()
        with _exclusive_model(raw):
            session = OfficialTabICLSession(self, raw)
            try:
                yield session
            finally:
                session._active = False

    def _validated_raw_model(self) -> Any:
        raw = _validate_fitted_estimator(self.estimator)
        if id(raw) != self._raw_model_id or raw is not self._raw_model:
            raise RuntimeError("the fitted classifier's raw model was replaced")
        return raw

    def _predict_proba_exclusive(
        self,
        X: Any,
        *,
        y: Any | None,
        sites: Sequence[str],
        interventions: Mapping[str, OfficialIntervention] | None,
        require_exact_baseline: bool | None,
    ) -> OfficialInferenceResult:
        raw = self._validated_raw_model()
        metric_targets = _encoded_metric_targets(
            self.estimator, y, expected_rows=_num_rows(X)
        )
        requested_sites = tuple(sites)
        if len(requested_sites) != len(set(requested_sites)):
            raise ValueError("capture sites must not contain duplicates")
        edits = dict(interventions or {})
        if any(not callable(callback) for callback in edits.values()):
            raise TypeError("every official TabICL intervention must be callable")
        if _all_missing_feature_mask(X).any():
            raise ValueError(
                "instrumented official inference does not support all-missing "
                "feature masks because their shuffle/group remapping is transient"
            )

        instrumented = bool(requested_sites or edits)
        if require_exact_baseline is None:
            require_exact_baseline = not edits
        if require_exact_baseline and not instrumented:
            require_exact_baseline = False

        if not instrumented:
            initial_identity_state = _identity_rng_state(raw)
            try:
                probabilities = _validated_probabilities(
                    _official_predict_proba(self.estimator, deepcopy(X)),
                    n_rows=_num_rows(X),
                    n_classes=int(self.estimator.n_classes_),
                )
                return OfficialInferenceResult(
                    probabilities=probabilities,
                    baseline_probabilities=probabilities.copy(),
                    classes=np.asarray(self.estimator.classes_).copy(),
                    forward_calls=(),
                    metrics=_metrics(probabilities, metric_targets),
                    exact_baseline_verified=True,
                    source_evidence_level=self.source_evidence_level,
                )
            except BaseException:
                _restore_identity_rng_state(raw, initial_identity_state)
                raise

        schedule = _expected_forward_schedule(self.estimator)
        initial_identity_state = _identity_rng_state(raw)
        baseline_probabilities: np.ndarray
        baseline_final_identity_state: Any | None = None
        try:
            baseline_probabilities = _validated_probabilities(
                _official_predict_proba(self.estimator, deepcopy(X)),
                n_rows=_num_rows(X),
                n_classes=int(self.estimator.n_classes_),
            )
            baseline_final_identity_state = _identity_rng_state(raw)
            _restore_identity_rng_state(raw, initial_identity_state)
            probabilities, calls = self._instrumented_predict(
                deepcopy(X),
                schedule=schedule,
                sites=requested_sites,
                interventions=edits,
            )
            probabilities = _validated_probabilities(
                probabilities,
                n_rows=_num_rows(X),
                n_classes=int(self.estimator.n_classes_),
            )
            _require_same_identity_state(raw, baseline_final_identity_state)
            exact = np.array_equal(probabilities, baseline_probabilities)
            if require_exact_baseline and not exact:
                raise RuntimeError(
                    "instrumented official TabICL probabilities differ from the "
                    "paired direct predict_proba baseline"
                )
            return OfficialInferenceResult(
                probabilities=probabilities,
                baseline_probabilities=baseline_probabilities,
                classes=np.asarray(self.estimator.classes_).copy(),
                forward_calls=tuple(calls),
                metrics=_metrics(probabilities, metric_targets),
                exact_baseline_verified=exact,
                source_evidence_level=self.source_evidence_level,
            )
        except BaseException:
            _restore_identity_rng_state(raw, initial_identity_state)
            raise

    def _instrumented_predict(
        self,
        X: Any,
        *,
        schedule: tuple[_ExpectedForward, ...],
        sites: tuple[str, ...],
        interventions: Mapping[str, OfficialIntervention],
    ) -> tuple[np.ndarray, list[OfficialForwardCapture]]:
        raw = self._raw_model
        if "forward" in vars(raw):
            raise RuntimeError(
                "raw model already has an instance-level forward override; "
                "refusing to replace it"
            )
        original_forward = raw.forward
        calls: list[OfficialForwardCapture] = []
        call_index = 0

        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_index
            if call_index >= len(schedule):
                raise RuntimeError(
                    "official classifier made more raw-model calls than expected"
                )
            expected = schedule[call_index]
            metadata, local_group_map = _metadata_for_call(
                original_forward,
                args,
                kwargs,
                expected=expected,
                call_index=call_index,
                raw=raw,
                estimator=self.estimator,
            )
            counts = {site: 0 for site in interventions}
            bound_edits: dict[str, Callable[[ActivationRecord], Any]] = {}
            for site, callback in interventions.items():

                def apply_edit(
                    record: ActivationRecord,
                    *,
                    name: str = site,
                    operation: OfficialIntervention = callback,
                ) -> Any:
                    counts[name] += 1
                    if counts[name] > 1:
                        raise RuntimeError(
                            f"intervention site {name!r} ran more than once in "
                            "one official raw-model call"
                        )
                    return operation(record, metadata)

                bound_edits[site] = apply_edit

            with ExitStack() as stack:
                if bound_edits:
                    stack.enter_context(self.adapter.intervene(raw, bound_edits))
                capture_buffer: Mapping[str, ActivationRecord]
                if sites:
                    capture_buffer = stack.enter_context(
                        self.adapter.capture(
                            raw,
                            sites=sites,
                            model_sha=self.model_sha,
                            checkpoint_sha=self.checkpoint_sha,
                            preprocessing_view_id=metadata.preprocessing_view_id,
                            feature_group_map=local_group_map,
                        )
                    )
                else:
                    capture_buffer = {}
                output = original_forward(*args, **kwargs)
                missing_captures = sorted(set(sites) - capture_buffer.keys())
                if missing_captures:
                    raise RuntimeError(
                        "capture sites did not run exactly once in the official "
                        f"raw-model call: {missing_captures}"
                    )
                missing_edits = sorted(
                    site for site, count in counts.items() if count != 1
                )
                if missing_edits:
                    raise RuntimeError(
                        "intervention sites did not run exactly once in the official "
                        f"raw-model call: {missing_edits}"
                    )
                calls.append(
                    OfficialForwardCapture(
                        metadata=metadata,
                        activations=dict(capture_buffer),
                    )
                )
            call_index += 1
            return output

        try:
            raw.forward = wrapped_forward
            probabilities = _official_predict_proba(self.estimator, X)
            if call_index != len(schedule):
                raise RuntimeError(
                    "official classifier made fewer raw-model calls than expected: "
                    f"{call_index} != {len(schedule)}"
                )
        finally:
            if "forward" in vars(raw):
                delattr(raw, "forward")
        if not _same_bound_method(raw.forward, original_forward):
            raise RuntimeError("failed to restore the raw model's original forward method")
        return np.asarray(probabilities), calls


class OfficialTabICLSession:
    """Exclusive multi-condition session used for exact identity-RNG replay."""

    def __init__(self, driver: OfficialTabICLDriver, raw: Any) -> None:
        self._driver = driver
        self._raw = raw
        self._active = True
        self._owner_thread = threading.get_ident()
        self._in_call = False
        self._policy_active = False

    def snapshot_identity_rng(self) -> Any | None:
        self._require_idle()
        return _identity_rng_state(self._raw)

    def restore_identity_rng(self, state: Any | None) -> None:
        self._require_idle()
        _restore_identity_rng_state(self._raw, state)

    @contextmanager
    def rope_policy(self, **policy: Any) -> Iterator[None]:
        """Apply one reversible RowInteraction RoPE policy inside this session.

        Keeping the policy under the session's model-exclusivity lock prevents a
        second caller from observing the temporary attention hooks.  Prediction is
        intentionally allowed while the policy is active, but nested policies and
        cross-thread use fail closed.
        """

        self._require_idle()
        if self._policy_active:
            raise RuntimeError("official TabICL RoPE policies cannot be nested")
        policy_factory = getattr(self._driver.adapter, "rope_policy", None)
        if not callable(policy_factory):
            raise TypeError("the official TabICL adapter has no rope_policy context")
        self._policy_active = True
        try:
            with policy_factory(self._raw, **policy):
                yield None
        finally:
            self._policy_active = False

    def predict_proba(
        self,
        X: Any,
        *,
        y: Any | None = None,
        sites: Sequence[str] = (),
        interventions: Mapping[str, OfficialIntervention] | None = None,
        require_exact_baseline: bool | None = None,
    ) -> OfficialInferenceResult:
        self._require_idle()
        self._in_call = True
        try:
            return self._driver._predict_proba_exclusive(
                X,
                y=y,
                sites=sites,
                interventions=interventions,
                require_exact_baseline=require_exact_baseline,
            )
        finally:
            self._in_call = False

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("official TabICL paired session is no longer active")
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError(
                "official TabICL paired session cannot cross threads"
            )
        if self._driver._validated_raw_model() is not self._raw:
            raise RuntimeError("official TabICL paired session raw model changed")

    def _require_idle(self) -> None:
        self._require_active()
        if self._in_call:
            raise RuntimeError("official TabICL paired session is not reentrant")


def fit_official_tabicl_driver(
    checkpoint: str | Path,
    X_train: Any,
    y_train: Any,
    *,
    device: str = "cpu",
    model_sha: str,
    estimator_options: Mapping[str, Any] | None = None,
    adapter: ModelAdapter | None = None,
    expected_source_root: str | Path | None = None,
    fit_context: str = "caller-provided-train",
) -> OfficialTabICLDriver:
    """Fit the public classifier on raw data with downloading and caches disabled."""

    verified_checkpoint = verify_file(checkpoint)
    checkpoint_path = verified_checkpoint.path
    _require_hex_digest("model_sha", model_sha, length=40)
    try:
        from tabicl import TabICLClassifier
    except ImportError as error:
        raise ImportError(
            "official TabICL inference requires the public `tabicl` package and "
            "its pandas/scikit-learn dependencies"
        ) from error

    source_evidence = None
    if expected_source_root is not None:
        source_evidence = verify_git_tree(
            expected_source_root,
            expected_sha=model_sha,
        )
        source_file = Path(inspect.getfile(TabICLClassifier)).resolve(strict=True)
        if not source_file.is_relative_to(source_evidence.root):
            raise RuntimeError(
                "imported TabICLClassifier is outside expected_source_root"
            )

    options = dict(estimator_options or {})
    locked = {
        "model_path",
        "allow_auto_download",
        "kv_cache",
        "support_many_classes",
        "device",
    }
    conflicts = sorted(options.keys() & locked)
    if conflicts:
        raise ValueError(
            "estimator_options cannot override aligned safety settings: "
            f"{conflicts}"
        )
    classifier = TabICLClassifier(
        **options,
        model_path=checkpoint_path,
        allow_auto_download=False,
        kv_cache=False,
        support_many_classes=False,
        device=device,
    )
    classifier.fit(X_train, y_train)
    verified_checkpoint.assert_unchanged()
    if source_evidence is not None:
        raw_source = Path(inspect.getfile(classifier.model_.__class__)).resolve(
            strict=True
        )
        if not raw_source.is_relative_to(source_evidence.root):
            raise RuntimeError("loaded raw TabICL model is outside expected_source_root")
    loaded_path = Path(classifier.model_path_).resolve(strict=True)
    if loaded_path != checkpoint_path:
        raise RuntimeError("official classifier loaded a different checkpoint")
    measured_model_sha = (
        source_evidence.head_sha if source_evidence is not None else model_sha
    )
    return OfficialTabICLDriver(
        classifier,
        adapter=adapter,
        model_sha=measured_model_sha,
        checkpoint_sha=verified_checkpoint.digest.sha256,
        fit_context=fit_context,
        _source_evidence=source_evidence,
    )


def load_raw_talent_splits(
    dataset_dir: str | Path, *, trusted_pickle: bool = False
) -> RawTalentDataset:
    """Load raw TALENT arrays without encoding, scaling, or imputation.

    Object-backed NumPy arrays require Python pickle.  Because pickle can execute
    code, callers must opt in with ``trusted_pickle=True`` for a trusted local
    TALENT corpus.  Every consumed file is identity-checked and hashed.
    """

    if not isinstance(trusted_pickle, bool):
        raise TypeError("trusted_pickle must be a boolean")
    root = Path(dataset_dir).expanduser()
    if not root.is_absolute():
        raise ValueError("TALENT dataset_dir must be an absolute path")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("TALENT dataset_dir must be a directory")
    info_path = root / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError("TALENT dataset is missing info.json")
    expected_array_names = [
        f"{prefix}_{split}.npy"
        for prefix in ("N", "C", "y")
        for split in _SPLITS
    ]
    verified_inputs = {
        name: verify_file(root / name)
        for name in expected_array_names
        if (root / name).is_file()
    }
    verified_info = verify_file(info_path)
    verified_inputs[info_path.name] = verified_info
    input_bytes = {
        name: verified.read_bytes()
        for name, verified in verified_inputs.items()
    }
    info_bytes = input_bytes[info_path.name]
    try:
        info = json.loads(info_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("TALENT info.json is invalid") from error
    if not isinstance(info, Mapping):
        raise TypeError("TALENT info.json must contain a JSON object")
    task_type = str(info.get("task_type", "")).strip().lower()
    if task_type not in _CLASSIFICATION_TASKS:
        raise ValueError(f"unsupported TALENT task_type: {task_type!r}")

    y = {
        split: _load_target(
            input_bytes.get(f"y_{split}.npy"),
            split,
            allow_pickle=trusted_pickle,
        )
        for split in _SPLITS
    }
    row_counts = {split: int(values.shape[0]) for split, values in y.items()}
    if any(count == 0 for count in row_counts.values()):
        raise ValueError("TALENT train/val/test splits must all be non-empty")
    numeric = _load_raw_feature_family(
        input_bytes, "N", row_counts, allow_pickle=False
    )
    categorical = _load_raw_feature_family(
        input_bytes, "C", row_counts, allow_pickle=trusted_pickle
    )
    n_numeric = numeric["train"].shape[1]
    n_categorical = categorical["train"].shape[1]
    if n_numeric + n_categorical == 0:
        raise ValueError("TALENT dataset has no features")

    _validate_labels(y)
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError(
            "raw mixed-type TALENT loading requires pandas so the official "
            "TabICL transformer can detect categorical columns"
        ) from error

    splits: dict[str, RawTalentSplit] = {}
    numeric_columns = [f"numerical_{index}" for index in range(n_numeric)]
    categorical_columns = [
        f"categorical_{index}" for index in range(n_categorical)
    ]
    for split in _SPLITS:
        numeric_frame = pd.DataFrame(
            numeric[split].copy(), columns=numeric_columns
        )
        categorical_frame = pd.DataFrame(
            categorical[split].copy(), columns=categorical_columns, dtype=object
        )
        frame = pd.concat((numeric_frame, categorical_frame), axis=1)
        splits[split] = RawTalentSplit(X=frame, y=y[split].copy())

    for verified in verified_inputs.values():
        verified.assert_unchanged()
    return RawTalentDataset(
        name=root.name,
        task_type=task_type,
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
        n_numeric_features=n_numeric,
        n_categorical_features=n_categorical,
        info_sha256=verified_info.digest.sha256,
        input_sha256={
            name: verified.digest.sha256
            for name, verified in sorted(verified_inputs.items())
        },
    )


def fit_official_talent_driver(
    dataset: RawTalentDataset,
    checkpoint: str | Path,
    *,
    context_split: str,
    device: str = "cpu",
    model_sha: str,
    estimator_options: Mapping[str, Any] | None = None,
    adapter: ModelAdapter | None = None,
    expected_source_root: str | Path | None = None,
) -> OfficialTabICLDriver:
    """Fit on raw TALENT train or explicit train+validation context."""

    if context_split == "train":
        X_context = dataset.train.X
        y_context = dataset.train.y
    elif context_split == "train+val":
        try:
            import pandas as pd
        except ImportError as error:
            raise ImportError("train+val TALENT context requires pandas") from error
        X_context = pd.concat(
            (dataset.train.X, dataset.val.X), ignore_index=True
        )
        y_context = np.concatenate((dataset.train.y, dataset.val.y))
    else:
        raise ValueError("context_split must be 'train' or 'train+val'")
    return fit_official_tabicl_driver(
        checkpoint,
        X_context,
        y_context,
        device=device,
        model_sha=model_sha,
        estimator_options=estimator_options,
        adapter=adapter,
        expected_source_root=expected_source_root,
        fit_context=f"talent-{context_split}",
    )


def _validate_fitted_estimator(estimator: Any) -> Any:
    required = (
        "model_",
        "classes_",
        "n_classes_",
        "ensemble_generator_",
        "predict_proba",
    )
    missing = [name for name in required if not hasattr(estimator, name)]
    if missing:
        raise TypeError(f"classifier is not fitted; missing attributes: {missing}")
    if getattr(estimator, "kv_cache", None) is not False:
        raise ValueError("official mechanism inference requires kv_cache=False")
    if getattr(estimator, "model_kv_cache_", None) is not None:
        raise ValueError("official mechanism inference refuses classifier KV caches")
    if getattr(estimator, "support_many_classes", None) is not False:
        raise ValueError("official mechanism inference disables many-class recursion")
    raw = estimator.model_
    if getattr(raw, "_cache", None) is not None:
        raise ValueError("official mechanism inference refuses raw-model caches")
    if getattr(raw, "training", None) is not False:
        raise ValueError("official mechanism inference requires model.eval()")
    n_classes = int(estimator.n_classes_)
    max_classes = int(getattr(raw, "max_classes", -1))
    if n_classes < 2 or max_classes < n_classes:
        raise ValueError(
            "class count must be within the raw model's native max_classes"
        )
    return raw


def _expected_forward_schedule(estimator: Any) -> tuple[_ExpectedForward, ...]:
    generator = estimator.ensemble_generator_
    required = ("ensemble_configs_", "feature_shuffles_", "class_shuffles_")
    missing = [name for name in required if not hasattr(generator, name)]
    if missing:
        raise TypeError(f"official ensemble generator is missing: {missing}")
    configs = generator.ensemble_configs_
    feature_shuffles = generator.feature_shuffles_
    class_shuffles = generator.class_shuffles_
    keys = list(configs)
    if keys != list(feature_shuffles) or keys != list(class_shuffles):
        raise RuntimeError("official ensemble mapping orders do not agree")
    if not keys:
        raise RuntimeError("official ensemble schedule is empty")

    batch_size = getattr(estimator, "batch_size", None)
    if batch_size is not None and (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, (int, np.integer))
        or int(batch_size) <= 0
    ):
        raise ValueError("official classifier batch_size must be positive or None")
    n_features = int(getattr(generator, "n_features_in_", -1))
    n_classes = int(estimator.n_classes_)
    if n_features <= 0:
        raise RuntimeError("official ensemble has no retained features")

    schedule: list[_ExpectedForward] = []
    global_offset = 0
    for norm_method in keys:
        raw_configs = tuple(configs[norm_method])
        n_views = len(raw_configs)
        features = tuple(
            _permutation_tuple(value, n_features, "feature")
            for value in feature_shuffles[norm_method]
        )
        classes = tuple(
            _permutation_tuple(value, n_classes, "class")
            for value in class_shuffles[norm_method]
        )
        if n_views == 0 or len(features) != n_views or len(classes) != n_views:
            raise RuntimeError(
                f"official ensemble view counts disagree for {norm_method!r}"
            )
        configured_features: list[tuple[int, ...]] = []
        configured_classes: list[tuple[int, ...]] = []
        for config in raw_configs:
            try:
                feature_config, class_config = config
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    "official ensemble config must contain one feature and one "
                    "class permutation"
                ) from error
            configured_features.append(
                _permutation_tuple(feature_config, n_features, "feature")
            )
            configured_classes.append(
                _permutation_tuple(class_config, n_classes, "class")
            )
        if tuple(configured_features) != features:
            raise RuntimeError(
                "official ensemble_configs_ feature permutations differ from "
                "feature_shuffles_"
            )
        if tuple(configured_classes) != classes:
            raise RuntimeError(
                "official ensemble_configs_ class permutations differ from "
                "class_shuffles_"
            )
        effective_batch = n_views if batch_size is None else int(batch_size)
        n_batches = int(np.ceil(n_views / effective_batch))
        chunks = np.array_split(np.arange(n_views, dtype=np.int64), n_batches)
        for chunk in chunks:
            local = tuple(int(index) for index in chunk)
            schedule.append(
                _ExpectedForward(
                    norm_method=str(norm_method),
                    norm_view_indices=local,
                    ensemble_indices=tuple(
                        global_offset + index for index in local
                    ),
                    feature_shuffles=tuple(features[index] for index in local),
                    class_shuffles=tuple(classes[index] for index in local),
                )
            )
        global_offset += n_views
    return tuple(schedule)


def _metadata_for_call(
    original_forward: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    expected: _ExpectedForward,
    call_index: int,
    raw: Any,
    estimator: Any,
) -> tuple[OfficialForwardMetadata, tuple[tuple[int, ...], ...]]:
    try:
        bound = inspect.signature(original_forward).bind_partial(*args, **kwargs)
    except TypeError as error:
        raise RuntimeError("could not bind the official raw forward call") from error
    if "X" not in bound.arguments or "y_train" not in bound.arguments:
        raise RuntimeError("official raw forward call omitted X or y_train")
    X = bound.arguments["X"]
    y_train = bound.arguments["y_train"]
    X_shape = tuple(int(value) for value in getattr(X, "shape", ()))
    y_shape = tuple(int(value) for value in getattr(y_train, "shape", ()))
    if len(X_shape) != 3 or len(y_shape) != 2:
        raise RuntimeError("official raw X/y_train ranks changed unexpectedly")
    if X_shape[0] != len(expected.norm_view_indices) or y_shape[0] != X_shape[0]:
        raise RuntimeError("official raw ensemble batch size differs from schedule")
    if y_shape[1] <= 0 or y_shape[1] > X_shape[1]:
        raise RuntimeError("official raw train size is invalid")

    actual_shuffles = bound.arguments.get("feature_shuffles")
    if actual_shuffles is None:
        raise RuntimeError("official raw call omitted expected feature shuffles")
    normalized_shuffles = tuple(
        tuple(int(index) for index in shuffle) for shuffle in actual_shuffles
    )
    if normalized_shuffles != expected.feature_shuffles:
        raise RuntimeError("official raw feature shuffles differ from schedule")
    if X_shape[2] != len(expected.feature_shuffles[0]):
        raise RuntimeError("official raw feature width differs from fitted ensemble")
    fitted_y = np.asarray(estimator.ensemble_generator_.y_)
    if fitted_y.ndim != 1 or fitted_y.shape[0] != y_shape[1]:
        raise RuntimeError("official fitted ensemble labels have an invalid shape")
    expected_y = np.stack(
        [
            np.asarray(class_shuffle)[fitted_y.astype(np.int64)]
            for class_shuffle in expected.class_shuffles
        ],
        axis=0,
    )
    actual_y = y_train.detach().to(device="cpu").numpy()
    if not np.array_equal(actual_y, expected_y.astype(actual_y.dtype, copy=False)):
        raise RuntimeError(
            "official raw y_train differs from the scheduled class shuffles"
        )
    if bool(bound.arguments.get("return_logits", True)) != bool(
        estimator.average_logits
    ):
        raise RuntimeError("official raw return_logits setting drifted")
    actual_temperature = float(
        bound.arguments.get("softmax_temperature", 0.9)
    )
    if actual_temperature != float(estimator.softmax_temperature):
        raise RuntimeError("official raw softmax temperature drifted")
    if (
        "inference_config" in bound.arguments
        and bound.arguments["inference_config"] is not estimator.inference_config_
    ):
        raise RuntimeError("official raw inference_config identity drifted")

    local_group_map = _local_feature_group_map(raw, X_shape[2])
    post_filter_maps = tuple(
        tuple(
            tuple(shuffle[index] for index in group)
            for group in local_group_map
        )
        for shuffle in expected.feature_shuffles
    )
    view_id = (
        f"official-tabicl/{expected.norm_method}/raw-call-{call_index:04d}"
    )
    return (
        OfficialForwardMetadata(
            call_index=call_index,
            norm_method=expected.norm_method,
            norm_view_indices=expected.norm_view_indices,
            ensemble_indices=expected.ensemble_indices,
            feature_shuffles=expected.feature_shuffles,
            class_shuffles=expected.class_shuffles,
            raw_input_shape=X_shape,
            train_size=y_shape[1],
            preprocessing_view_id=view_id,
            post_filter_feature_group_maps=post_filter_maps,
        ),
        local_group_map,
    )


def _local_feature_group_map(
    raw: Any, n_features: int
) -> tuple[tuple[int, ...], ...]:
    col = getattr(raw, "col_embedder", None)
    if col is None:
        raise RuntimeError("raw TabICL model is missing col_embedder")
    mode = getattr(col, "feature_group", None)
    if mode is True or mode == "same":
        group_size = int(getattr(col, "feature_group_size", 0))
        return same_feature_group_map(n_features, group_size)
    raise ValueError(
        f"unsupported TabICL feature_group mode for exact mapping: {mode!r}"
    )


def _identity_rng_state(raw: Any) -> Any | None:
    row = getattr(raw, "row_interactor", None)
    raw_mode = getattr(raw, "row_identity_mode", None)
    row_mode = getattr(row, "identity_mode", None)
    if raw_mode is not None and row_mode is not None and raw_mode != row_mode:
        raise RuntimeError("raw and RowInteraction identity modes disagree")
    mode = raw_mode if raw_mode is not None else row_mode
    if mode != "temporary":
        return None
    generator = getattr(row, "_identity_generator", None)
    if generator is None or not hasattr(generator, "get_state"):
        raise RuntimeError("Temporary Identity RNG is unavailable")
    return generator.get_state().clone()


def _restore_identity_rng_state(raw: Any, state: Any | None) -> None:
    if state is None:
        return
    generator = getattr(raw.row_interactor, "_identity_generator", None)
    if generator is None or not hasattr(generator, "set_state"):
        raise RuntimeError("Temporary Identity RNG cannot be restored")
    generator.set_state(state.clone())


def _require_same_identity_state(raw: Any, expected: Any | None) -> None:
    if expected is None:
        return
    actual = _identity_rng_state(raw)
    try:
        import torch
    except ImportError as error:
        raise ImportError("Temporary Identity state comparison requires torch") from error
    if actual is None or not torch.equal(actual, expected):
        raise RuntimeError(
            "instrumented prediction advanced Temporary Identity RNG differently "
            "from the direct official baseline"
        )


def _encoded_metric_targets(
    estimator: Any, y: Any | None, *, expected_rows: int
) -> np.ndarray | None:
    if y is None:
        return None
    labels = np.asarray(y)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1 or labels.shape[0] != expected_rows:
        raise ValueError("metric labels must be a vector matching prediction rows")
    encoded = np.asarray(estimator.y_encoder_.transform(labels), dtype=np.int64)
    if encoded.shape != (expected_rows,) or np.any(encoded < 0) or np.any(
        encoded >= int(estimator.n_classes_)
    ):
        raise RuntimeError("official label encoder returned invalid class indices")
    return encoded


def _metrics(
    probabilities: np.ndarray, encoded: np.ndarray | None
) -> OfficialInferenceMetrics | None:
    if encoded is None:
        return None
    predicted = np.argmax(probabilities, axis=1)
    selected = probabilities[np.arange(encoded.shape[0]), encoded]
    tiny = np.finfo(probabilities.dtype).tiny
    return OfficialInferenceMetrics(
        accuracy=float(np.mean(predicted == encoded)),
        log_loss=float(-np.log(np.clip(selected, tiny, 1.0)).mean()),
        n_samples=int(encoded.shape[0]),
    )


def _official_predict_proba(estimator: Any, X: Any) -> Any:
    """Restore PyTorch's process-global thread setting even on hook failure."""

    n_jobs = getattr(estimator, "n_jobs", None)
    if n_jobs is None:
        return estimator.predict_proba(X)
    try:
        import torch
    except ImportError as error:
        raise ImportError("official TabICL inference requires torch") from error
    with _TORCH_THREAD_LOCK:
        previous_threads = torch.get_num_threads()
        try:
            return estimator.predict_proba(X)
        finally:
            torch.set_num_threads(previous_threads)


def _validated_probabilities(
    values: Any, *, n_rows: int, n_classes: int
) -> np.ndarray:
    probabilities = np.asarray(values)
    if probabilities.shape != (n_rows, n_classes):
        raise RuntimeError(
            "official predict_proba returned unexpected shape "
            f"{probabilities.shape}, expected {(n_rows, n_classes)}"
        )
    if not np.issubdtype(probabilities.dtype, np.floating):
        raise RuntimeError("official predict_proba did not return floating probabilities")
    if not np.isfinite(probabilities).all():
        raise RuntimeError("official predict_proba returned non-finite probabilities")
    if np.any(probabilities < 0.0):
        raise RuntimeError("official predict_proba returned negative probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-7):
        raise RuntimeError("official predict_proba rows are not normalized")
    return probabilities.copy()


def _all_missing_feature_mask(X: Any) -> np.ndarray:
    if hasattr(X, "isna"):
        mask = np.asarray(X.isna().all(axis=0), dtype=bool)
    else:
        values = np.asarray(X)
        if values.ndim != 2:
            raise ValueError("prediction features must be a two-dimensional table")
        if np.issubdtype(values.dtype, np.number):
            mask = np.isnan(values).all(axis=0)
        else:
            try:
                import pandas as pd
            except ImportError as error:
                raise ImportError(
                    "object-valued official TabICL inputs require pandas"
                ) from error
            mask = np.asarray(pd.isna(values).all(axis=0), dtype=bool)
    if mask.ndim != 1:
        raise ValueError("could not determine prediction feature mask")
    return mask


def _num_rows(X: Any) -> int:
    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("prediction features must be a two-dimensional table")
    return int(shape[0])


def _permutation_tuple(value: Any, size: int, kind: str) -> tuple[int, ...]:
    result = tuple(int(index) for index in value)
    if len(result) != size or set(result) != set(range(size)):
        raise RuntimeError(f"official {kind} shuffle is not a permutation")
    return result


def _same_bound_method(left: Any, right: Any) -> bool:
    return (
        getattr(left, "__self__", None) is getattr(right, "__self__", None)
        and getattr(left, "__func__", left) is getattr(right, "__func__", right)
    )


@contextmanager
def _exclusive_model(raw: Any) -> Iterator[None]:
    """Hold exclusivity across the complete paired prediction transaction."""

    model_id = id(raw)
    with _ACTIVE_LOCK:
        if model_id in _ACTIVE_MODEL_IDS:
            raise RuntimeError(
                "official TabICL inference is already active for this raw model"
            )
        _ACTIVE_MODEL_IDS.add(model_id)
    try:
        yield
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_MODEL_IDS.discard(model_id)


def _require_hex_digest(name: str, value: str, *, length: int) -> None:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a {length}-character lowercase hex digest")


def _load_target(
    raw: bytes | None, split: str, *, allow_pickle: bool
) -> np.ndarray:
    if raw is None:
        raise FileNotFoundError(f"TALENT is missing y_{split}.npy")
    try:
        values = np.load(io.BytesIO(raw), allow_pickle=allow_pickle)
    except ValueError as error:
        if "Object arrays cannot be loaded" in str(error):
            raise ValueError(
                "TALENT object labels require trusted_pickle=True for a trusted corpus"
            ) from error
        raise
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError(f"TALENT y_{split}.npy must be a vector")
    return values


def _load_raw_feature_family(
    inputs: Mapping[str, bytes],
    prefix: str,
    row_counts: Mapping[str, int],
    *,
    allow_pickle: bool,
) -> dict[str, np.ndarray]:
    names = {split: f"{prefix}_{split}.npy" for split in _SPLITS}
    present = {split: name in inputs for split, name in names.items()}
    if any(present.values()) and not all(present.values()):
        raise FileNotFoundError(f"TALENT has incomplete {prefix}_* feature splits")
    if not any(present.values()):
        return {
            split: np.empty((row_counts[split], 0), dtype=object)
            for split in _SPLITS
        }

    arrays: dict[str, np.ndarray] = {}
    width: int | None = None
    for split in _SPLITS:
        try:
            values = np.load(
                io.BytesIO(inputs[names[split]]), allow_pickle=allow_pickle
            )
        except ValueError as error:
            if "Object arrays cannot be loaded" in str(error):
                raise ValueError(
                    "TALENT object features require trusted_pickle=True for a "
                    "trusted corpus"
                ) from error
            raise
        if values.ndim != 2:
            raise ValueError(f"TALENT {prefix}_{split}.npy must be a matrix")
        if values.shape[0] != row_counts[split]:
            raise ValueError(f"TALENT {prefix}_{split}.npy row count disagrees")
        if width is None:
            width = int(values.shape[1])
        elif values.shape[1] != width:
            raise ValueError(f"TALENT {prefix}_* feature widths disagree")
        if prefix == "N":
            if not np.issubdtype(values.dtype, np.number):
                raise TypeError("TALENT numerical arrays must have numeric dtype")
            if np.isinf(values).any():
                raise ValueError(f"TALENT {prefix}_{split}.npy contains infinity")
        elif _object_contains_infinity(values):
            raise ValueError(f"TALENT {prefix}_{split}.npy contains infinity")
        arrays[split] = values
    return arrays


def _object_contains_infinity(values: np.ndarray) -> bool:
    for value in values.flat:
        if isinstance(value, (float, np.floating)) and np.isinf(value):
            return True
    return False


def _validate_labels(labels: Mapping[str, np.ndarray]) -> None:
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError("raw TALENT label validation requires pandas") from error
    for split, values in labels.items():
        if np.asarray(pd.isna(values)).any():
            raise ValueError(f"TALENT y_{split}.npy contains missing labels")
    train_tokens = {_label_token(value) for value in labels["train"]}
    if len(train_tokens) < 2:
        raise ValueError("TALENT training split has fewer than two classes")
    for split in ("val", "test"):
        split_tokens = {_label_token(value) for value in labels[split]}
        if split_tokens - train_tokens:
            raise ValueError(f"TALENT {split} split contains unseen labels")


def _label_token(value: Any) -> tuple[str, str]:
    scalar = value.item() if isinstance(value, np.generic) else value
    return type(scalar).__qualname__, repr(scalar)


__all__ = [
    "OFFICIAL_INFERENCE_PROTOCOL",
    "OfficialForwardCapture",
    "OfficialForwardMetadata",
    "OfficialInferenceMetrics",
    "OfficialInferenceResult",
    "OfficialIntervention",
    "OfficialTabICLDriver",
    "OfficialTabICLSession",
    "RawTalentDataset",
    "RawTalentSplit",
    "fit_official_tabicl_driver",
    "fit_official_talent_driver",
    "load_raw_talent_splits",
    "official_inference_contract_sha256",
]
