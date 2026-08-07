"""Model-causal AE/SAE edits through official TabICL sklearn inference.

All conditions use :class:`~pe_mechanism.official_tabicl.OfficialTabICLDriver`,
so mixed-type encoding, normalization ensembles, feature/class shuffles,
temperature, and aggregation remain owned by the public classifier.  Activation
replacements are indexed by the classifier's exact raw-call schedule rather than
merging or replaying ensemble tables independently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .adapters.base import ActivationRecord
from .causal import (
    activation_frequency,
    intervene_latents,
    matched_random_control_features,
    model_decoder_feature_norms,
)
from .identifiers import require_portable_identifier, require_public_label
from .manifest import RunManifest
from .model_causal import (
    CausalPredictionResult,
    PredictionMetrics,
    _condition_result,
    _decode_activation,
    _encode_activation,
    _normalizer_statistics,
    _prediction_metrics,
    _preserve_training_states,
    _resolve_latent_baseline,
)
from .official_tabicl import (
    OFFICIAL_INFERENCE_PROTOCOL,
    OfficialForwardCapture,
    OfficialForwardMetadata,
    OfficialInferenceResult,
    OfficialTabICLDriver,
    OfficialTabICLSession,
    fit_official_talent_driver,
    load_raw_talent_splits,
    official_inference_contract_sha256,
)
from .provenance import (
    RunTransaction,
    assert_dataset_roster,
    assert_git_commit_is_ancestor,
    load_verified_json_config,
    manifest_from_verified_inputs,
    verify_configured_run_inputs,
    verify_file,
    verify_run_directory,
)
from .representation import (
    MIN_HELDOUT_EXPLAINED_VARIANCE,
    MeanRMSNormalizer,
    load_verified_representation_checkpoint,
)


@dataclass(frozen=True)
class OfficialCausalEvaluation:
    """In-memory causal results paired to one official native prediction."""

    site: str
    dataset_id: str
    sample_ids: tuple[str | int, ...]
    model_sha: str
    checkpoint_sha: str
    source_evidence_level: str
    fit_context: str
    classes: np.ndarray
    native_prediction: PredictionMetrics
    native_forward_calls: tuple[OfficialForwardCapture, ...]
    conditions: Mapping[str, CausalPredictionResult]
    matched_control_features: tuple[int, ...]
    no_op_reconstruction_mse: float
    no_op_accuracy_drop: float
    no_op_accuracy_absolute_difference: float
    no_op_probability_max_abs_difference: float
    representation_qualification: Mapping[str, Any]
    capture_exact_to_direct: bool
    mechanistic_rescue_passed: bool
    mechanistic_rescue_status: str


@dataclass(frozen=True)
class _EncodedCall:
    capture: OfficialForwardCapture
    record: ActivationRecord
    raw: Tensor
    latents: Tensor
    reconstruction: Tensor


class _ReplacementPlan:
    """Apply exactly one shape- and view-matched replacement per raw call."""

    def __init__(
        self,
        baseline_calls: tuple[OfficialForwardCapture, ...],
        replacements: tuple[Tensor, ...],
        *,
        site: str,
    ) -> None:
        if len(baseline_calls) != len(replacements) or not baseline_calls:
            raise ValueError("replacement plan must align to every baseline raw call")
        self.baseline_calls = baseline_calls
        self.replacements = replacements
        self.site = site
        self.next_call = 0

    def __call__(
        self, record: ActivationRecord, metadata: OfficialForwardMetadata
    ) -> Tensor:
        if self.next_call >= len(self.baseline_calls):
            raise RuntimeError("replacement plan received an extra raw-model call")
        expected = self.baseline_calls[self.next_call]
        baseline = expected.activations[self.site]
        if metadata != expected.metadata:
            raise RuntimeError(
                "official raw-call schedule/view metadata changed across conditions"
            )
        if (
            record.site != self.site
            or record.axis_names != baseline.axis_names
            or tuple(record.shape) != tuple(baseline.shape)
        ):
            raise RuntimeError(
                "official activation site/shape/axis changed across conditions"
            )
        live = torch.as_tensor(record.tensor).detach().to(device="cpu")
        captured = torch.as_tensor(baseline.tensor).detach().to(device="cpu")
        if not torch.equal(live, captured):
            raise RuntimeError(
                "paired condition did not reproduce the captured native activation"
            )
        replacement = self.replacements[self.next_call]
        if tuple(replacement.shape) != tuple(record.shape):
            raise RuntimeError("decoded replacement shape differs from activation")
        self.next_call += 1
        original = torch.as_tensor(record.tensor)
        return replacement.to(device=original.device, dtype=original.dtype)

    def verify_complete(self) -> None:
        if self.next_call != len(self.baseline_calls):
            raise RuntimeError(
                "replacement plan did not consume every official raw-model call"
            )


def run_official_tabicl_causal_edits(
    driver: OfficialTabICLDriver,
    X: Any,
    y: Any,
    *,
    site: str,
    autoencoder: nn.Module,
    normalizer: MeanRMSNormalizer,
    target_features: Sequence[int],
    dataset_id: str,
    sample_ids: Sequence[str | int],
    representation_qualification: Mapping[str, Any],
    control_features: Sequence[int] | None = None,
    latent_baseline: Any = "mean",
    random_seed: int = 42,
    random_candidate_pool_size: int = 8,
    max_no_op_reconstruction_mse: float = 0.01,
    max_no_op_probability_deviation: float = 0.02,
    max_no_op_accuracy_drop: float = 0.005,
) -> OfficialCausalEvaluation:
    """Run no-op, target, matched-random, and round-trip control edits.

    Every condition begins from the same Temporary Identity RNG state.  On
    success, the generator is left at the state produced by one native
    prediction; on failure, its entry state is restored.
    """

    if not isinstance(driver, OfficialTabICLDriver):
        raise TypeError("driver must be an OfficialTabICLDriver")
    if not isinstance(site, str) or not site:
        raise ValueError("site must be a non-empty activation name")
    qualification = _validated_qualification(
        representation_qualification,
        driver=driver,
        site=site,
    )
    latent_dim = _positive_int_attribute(autoencoder, "latent_dim")
    _positive_int_attribute(autoencoder, "input_dim")
    features = _validated_features(target_features, latent_dim)
    frozen_controls = (
        None
        if control_features is None
        else _validated_features(control_features, latent_dim)
    )
    if frozen_controls is not None and (
        len(frozen_controls) != len(features)
        or set(frozen_controls) & set(features)
    ):
        raise ValueError(
            "control_features must be disjoint from and match the size of "
            "target_features"
        )
    if (
        isinstance(random_candidate_pool_size, bool)
        or not isinstance(random_candidate_pool_size, int)
        or random_candidate_pool_size <= 0
    ):
        raise ValueError("random_candidate_pool_size must be positive")
    if (
        not np.isfinite(max_no_op_reconstruction_mse)
        or max_no_op_reconstruction_mse < 0
    ):
        raise ValueError(
            "max_no_op_reconstruction_mse must be finite and non-negative"
        )
    if (
        not np.isfinite(max_no_op_probability_deviation)
        or max_no_op_probability_deviation < 0
    ):
        raise ValueError(
            "max_no_op_probability_deviation must be finite and non-negative"
        )
    if (
        not np.isfinite(max_no_op_accuracy_drop)
        or max_no_op_accuracy_drop < 0
    ):
        raise ValueError("max_no_op_accuracy_drop must be finite and non-negative")
    resolved_dataset_id = require_public_label(dataset_id, name="dataset_id")
    targets = _encoded_targets(driver, y, expected_rows=_num_rows(X))
    resolved_sample_ids = _sample_ids(sample_ids, expected=targets.shape[0])

    conditions: dict[str, CausalPredictionResult] = {}
    with _preserve_training_states(autoencoder, normalizer):
        autoencoder.eval()
        normalizer.eval()
        with torch.inference_mode(), driver.paired_session() as session:
            entry_state = session.snapshot_identity_rng()
            final_state = None
            try:
                session.restore_identity_rng(entry_state)
                native_result = session.predict_proba(
                    X,
                    y=y,
                    sites=(site,),
                    require_exact_baseline=True,
                )
                final_state = session.snapshot_identity_rng()
                _validate_native_capture(native_result, driver=driver, site=site)
                native = _prediction_metrics(
                    native_result.probabilities,
                    targets,
                    output_kind="probabilities",
                )
                encoded_calls = tuple(
                    _encode_call(
                        call,
                        site=site,
                        autoencoder=autoencoder,
                        normalizer=normalizer,
                    )
                    for call in native_result.forward_calls
                )
                reconstruction_mse = _reconstruction_mse(encoded_calls)
                if reconstruction_mse > max_no_op_reconstruction_mse:
                    raise RuntimeError(
                        f"no-op reconstruction MSE {reconstruction_mse:.6g} "
                        f"exceeds limit {max_no_op_reconstruction_mse:.6g}"
                    )

                no_op_replacements = tuple(
                    call.reconstruction for call in encoded_calls
                )
                no_op_result, no_op = _run_condition(
                    session,
                    X,
                    y,
                    site=site,
                    entry_state=entry_state,
                    expected_final_state=final_state,
                    native_result=native_result,
                    replacements=no_op_replacements,
                    targets=targets,
                )
                no_op_accuracy_drop = native.accuracy - no_op.accuracy
                no_op_accuracy_difference = abs(no_op_accuracy_drop)
                no_op_probability_difference = float(
                    np.max(
                        np.abs(
                            no_op.probabilities.astype(np.float64)
                            - native.probabilities.astype(np.float64)
                        )
                    )
                )
                if no_op_accuracy_difference > max_no_op_accuracy_drop:
                    raise RuntimeError(
                        "no-op reconstruction absolute accuracy difference "
                        f"{no_op_accuracy_difference:.6g} exceeds limit "
                        f"{max_no_op_accuracy_drop:.6g}"
                    )
                if (
                    no_op_probability_difference
                    > max_no_op_probability_deviation
                ):
                    raise RuntimeError(
                        "no-op reconstruction maximum absolute probability "
                        f"difference {no_op_probability_difference:.6g} exceeds "
                        f"limit {max_no_op_probability_deviation:.6g}"
                    )
                conditions["no_op_reconstruction"] = _condition_result(
                    "no_op_reconstruction",
                    no_op,
                    reference_scope="primary",
                    model_baseline=native,
                    reconstruction_baseline=no_op,
                    target_features=features,
                )

                all_latents = torch.cat(
                    [call.latents for call in encoded_calls], dim=0
                )
                baseline_value = _resolve_latent_baseline(
                    latent_baseline, all_latents
                )
                target_latents = tuple(
                    intervene_latents(
                        call.latents,
                        features,
                        mode="baseline",
                        baseline=baseline_value,
                    )
                    for call in encoded_calls
                )
                target_replacements = _decode_calls(
                    encoded_calls,
                    target_latents,
                    autoencoder=autoencoder,
                    normalizer=normalizer,
                )
                _, target_prediction = _run_condition(
                    session,
                    X,
                    y,
                    site=site,
                    entry_state=entry_state,
                    expected_final_state=final_state,
                    native_result=native_result,
                    replacements=target_replacements,
                    targets=targets,
                )
                conditions["target_baseline_edit"] = _condition_result(
                    "target_baseline_edit",
                    target_prediction,
                    reference_scope="primary",
                    model_baseline=native,
                    reconstruction_baseline=no_op,
                    target_features=features,
                )

                controls = (
                    frozen_controls
                    if frozen_controls is not None
                    else tuple(
                        matched_random_control_features(
                            features,
                            activation_frequency(all_latents),
                            model_decoder_feature_norms(autoencoder),
                            seed=random_seed,
                            candidate_pool_size=random_candidate_pool_size,
                        )
                    )
                )
                random_latents = tuple(
                    intervene_latents(
                        call.latents,
                        controls,
                        mode="baseline",
                        baseline=baseline_value,
                    )
                    for call in encoded_calls
                )
                random_replacements = _decode_calls(
                    encoded_calls,
                    random_latents,
                    autoencoder=autoencoder,
                    normalizer=normalizer,
                )
                _, random_prediction = _run_condition(
                    session,
                    X,
                    y,
                    site=site,
                    entry_state=entry_state,
                    expected_final_state=final_state,
                    native_result=native_result,
                    replacements=random_replacements,
                    targets=targets,
                )
                conditions["matched_random_edit"] = _condition_result(
                    "matched_random_edit",
                    random_prediction,
                    reference_scope="primary",
                    model_baseline=native,
                    reconstruction_baseline=no_op,
                    target_features=features,
                    control_features=controls,
                )

                restored_latents = []
                for call, edited in zip(
                    encoded_calls, target_latents, strict=True
                ):
                    restored = edited.clone()
                    restored[:, list(features)] = call.latents[:, list(features)]
                    if not torch.equal(restored, call.latents):
                        raise RuntimeError(
                            "round-trip control did not restore native latent values"
                        )
                    restored_latents.append(restored)
                restore_replacements = _decode_calls(
                    encoded_calls,
                    tuple(restored_latents),
                    autoencoder=autoencoder,
                    normalizer=normalizer,
                )
                for restored, no_op_replacement in zip(
                    restore_replacements, no_op_replacements, strict=True
                ):
                    if not torch.equal(restored, no_op_replacement):
                        raise RuntimeError(
                            "round-trip control did not restore no-op reconstruction"
                        )
                restore_result, restore_prediction = _run_condition(
                    session,
                    X,
                    y,
                    site=site,
                    entry_state=entry_state,
                    expected_final_state=final_state,
                    native_result=native_result,
                    replacements=restore_replacements,
                    targets=targets,
                )
                if not np.array_equal(
                    restore_result.probabilities, no_op_result.probabilities
                ):
                    raise RuntimeError(
                        "round-trip control prediction does not exactly match no-op"
                    )
                conditions["roundtrip_restore_control"] = _condition_result(
                    "roundtrip_restore_control",
                    restore_prediction,
                    reference_scope="primary",
                    model_baseline=native,
                    reconstruction_baseline=no_op,
                    target_features=features,
                )
                session.restore_identity_rng(final_state)
            except BaseException:
                session.restore_identity_rng(entry_state)
                raise

    return OfficialCausalEvaluation(
        site=site,
        dataset_id=resolved_dataset_id,
        sample_ids=resolved_sample_ids,
        model_sha=driver.model_sha,
        checkpoint_sha=driver.checkpoint_sha,
        source_evidence_level=driver.source_evidence_level,
        fit_context=driver.fit_context,
        classes=np.asarray(native_result.classes).copy(),
        native_prediction=native,
        native_forward_calls=native_result.forward_calls,
        conditions=conditions,
        matched_control_features=controls,
        no_op_reconstruction_mse=reconstruction_mse,
        no_op_accuracy_drop=no_op_accuracy_drop,
        no_op_accuracy_absolute_difference=no_op_accuracy_difference,
        no_op_probability_max_abs_difference=no_op_probability_difference,
        representation_qualification=qualification,
        capture_exact_to_direct=(
            native_result.exact_baseline_verified
            and np.array_equal(
                native_result.probabilities,
                native_result.baseline_probabilities,
            )
        ),
        mechanistic_rescue_passed=False,
        mechanistic_rescue_status="paired_rescue_not_run",
    )


def _run_condition(
    session: OfficialTabICLSession,
    X: Any,
    y: Any,
    *,
    site: str,
    entry_state: Any | None,
    expected_final_state: Any | None,
    native_result: OfficialInferenceResult,
    replacements: tuple[Tensor, ...],
    targets: np.ndarray,
) -> tuple[OfficialInferenceResult, PredictionMetrics]:
    session.restore_identity_rng(entry_state)
    plan = _ReplacementPlan(
        native_result.forward_calls, replacements, site=site
    )
    result = session.predict_proba(
        X,
        y=y,
        sites=(site,),
        interventions={site: plan},
        require_exact_baseline=False,
    )
    plan.verify_complete()
    _require_same_rng_state(
        session.snapshot_identity_rng(), expected_final_state
    )
    _validate_condition_capture(
        result,
        native_result=native_result,
        replacements=replacements,
        site=site,
    )
    return result, _prediction_metrics(
        result.probabilities,
        targets,
        output_kind="probabilities",
    )


def _validate_native_capture(
    result: OfficialInferenceResult,
    *,
    driver: OfficialTabICLDriver,
    site: str,
) -> None:
    if not result.exact_baseline_verified or not np.array_equal(
        result.probabilities, result.baseline_probabilities
    ):
        raise RuntimeError(
            "capture-only official inference is not byte-exact to direct inference"
        )
    if not result.forward_calls:
        raise RuntimeError("official baseline capture has no raw-model calls")
    for expected_index, call in enumerate(result.forward_calls):
        if call.metadata.call_index != expected_index:
            raise RuntimeError("official raw-call indices are not contiguous")
        if set(call.activations) != {site}:
            raise RuntimeError("official baseline did not capture exactly one site")
        record = call.activations[site]
        if (
            record.site != site
            or record.model_sha != driver.model_sha
            or record.checkpoint_sha != driver.checkpoint_sha
            or record.preprocessing_view_id
            != call.metadata.preprocessing_view_id
            or tuple(record.shape)[0] != call.metadata.raw_input_shape[0]
        ):
            raise RuntimeError(
                "official baseline activation metadata is internally inconsistent"
            )


def _validate_condition_capture(
    result: OfficialInferenceResult,
    *,
    native_result: OfficialInferenceResult,
    replacements: tuple[Tensor, ...],
    site: str,
) -> None:
    if not np.array_equal(
        result.baseline_probabilities, native_result.probabilities
    ):
        raise RuntimeError(
            "paired condition native baseline differs after identity-RNG replay"
        )
    if len(result.forward_calls) != len(native_result.forward_calls):
        raise RuntimeError("official raw-call count changed across conditions")
    for condition_call, native_call, replacement in zip(
        result.forward_calls,
        native_result.forward_calls,
        replacements,
        strict=True,
    ):
        if condition_call.metadata != native_call.metadata:
            raise RuntimeError(
                "official raw-call schedule/view metadata changed across conditions"
            )
        if set(condition_call.activations) != {site}:
            raise RuntimeError("condition did not capture exactly the edited site")
        observed = condition_call.activations[site]
        native = native_call.activations[site]
        if (
            observed.axis_names != native.axis_names
            or observed.shape != native.shape
            or observed.feature_group_map != native.feature_group_map
            or observed.preprocessing_view_id
            != native.preprocessing_view_id
        ):
            raise RuntimeError(
                "condition activation shape/axis/view differs from native capture"
            )
        expected = replacement.detach().to(
            device="cpu", dtype=torch.as_tensor(observed.tensor).dtype
        )
        if not torch.equal(torch.as_tensor(observed.tensor), expected):
            raise RuntimeError(
                "post-intervention capture differs from decoded replacement"
            )


def _encode_call(
    capture: OfficialForwardCapture,
    *,
    site: str,
    autoencoder: nn.Module,
    normalizer: MeanRMSNormalizer,
) -> _EncodedCall:
    record = capture.activations[site]
    encoded = _encode_activation(
        record,
        site=site,
        autoencoder=autoencoder,
        normalizer=normalizer,
    )
    return _EncodedCall(
        capture=capture,
        record=record,
        raw=encoded.raw,
        latents=encoded.latents,
        reconstruction=encoded.reconstruction,
    )


def _decode_calls(
    calls: tuple[_EncodedCall, ...],
    latents: tuple[Tensor, ...],
    *,
    autoencoder: nn.Module,
    normalizer: MeanRMSNormalizer,
) -> tuple[Tensor, ...]:
    if len(calls) != len(latents):
        raise ValueError("edited latents must align to raw calls")
    decoded = []
    for call, values in zip(calls, latents, strict=True):
        mean, rms = _normalizer_statistics(
            normalizer,
            device=values.device,
            dtype=values.dtype,
        )
        decoded.append(
            _decode_activation(
                values,
                shape=tuple(call.raw.shape),
                autoencoder=autoencoder,
                mean=mean,
                rms=rms,
            )
        )
    return tuple(decoded)


def _reconstruction_mse(calls: tuple[_EncodedCall, ...]) -> float:
    squared_error = 0.0
    count = 0
    for call in calls:
        difference = call.reconstruction.detach().to(device="cpu") - call.raw.to(
            device="cpu"
        )
        squared_error += float(difference.square().sum())
        count += difference.numel()
    value = squared_error / count
    if not np.isfinite(value):
        raise RuntimeError("no-op reconstruction MSE is non-finite")
    return value


def _validated_qualification(
    values: Mapping[str, Any],
    *,
    driver: OfficialTabICLDriver,
    site: str,
) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError("representation_qualification must be a mapping")
    if values.get("metric_split") != "validation" or (
        values.get("activation_fidelity_passed") is not True
    ):
        raise RuntimeError(
            "representation failed the held-out activation-fidelity qualification gate"
        )
    expected_binding = {
        "model_sha": driver.model_sha,
        "checkpoint_sha": driver.checkpoint_sha,
        "site": site,
    }
    mismatches = {
        name: (values.get(name), expected)
        for name, expected in expected_binding.items()
        if values.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(
            "representation qualification is not bound to this model/checkpoint/site: "
            f"{mismatches}"
        )
    return dict(values)


def _positive_int_attribute(module: nn.Module, name: str) -> int:
    value = getattr(module, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"autoencoder must expose a positive integer {name}")
    return value


def _validated_features(values: Sequence[int], latent_dim: int) -> tuple[int, ...]:
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("target_features must contain integer indices")
    result = tuple(values)
    if not result:
        raise ValueError("target_features must not be empty")
    if len(set(result)) != len(result):
        raise ValueError("target_features must be unique")
    if any(value < 0 or value >= latent_dim for value in result):
        raise IndexError("target feature is outside the latent dimension")
    return result


def _encoded_targets(
    driver: OfficialTabICLDriver, y: Any, *, expected_rows: int
) -> np.ndarray:
    values = np.asarray(y)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or values.shape[0] != expected_rows:
        raise ValueError("labels must align exactly to prediction rows")
    encoded = np.asarray(
        driver.estimator.y_encoder_.transform(values), dtype=np.int64
    )
    if encoded.shape != (expected_rows,):
        raise RuntimeError("official label encoder returned an unexpected shape")
    return encoded


def _sample_ids(
    values: Sequence[str | int], *, expected: int
) -> tuple[str | int, ...]:
    if isinstance(values, (str, bytes)) or len(values) != expected:
        raise ValueError("sample_ids must align exactly to prediction rows")
    result: list[str | int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError("sample_ids must contain strings or integers")
        result.append(
            require_portable_identifier(value, name="sample_id")
            if isinstance(value, str)
            else value
        )
    if len(set(result)) != len(result):
        raise ValueError("sample_ids must be unique")
    return tuple(result)


def _num_rows(X: Any) -> int:
    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("official causal features must be a two-dimensional table")
    return int(shape[0])


def _require_same_rng_state(actual: Any | None, expected: Any | None) -> None:
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise RuntimeError("Temporary Identity RNG availability changed")
        return
    if not torch.equal(actual, expected):
        raise RuntimeError(
            "paired condition advanced Temporary Identity RNG differently"
        )


_MODEL_CAUSAL_TOP_LEVEL = {
    "provenance",
    "representation_run_dir",
    "expected_parent_manifest_sha256",
    "dataset",
    "intervention",
    "official_classifier",
}
_PREPROCESSING_PROTOCOL = OFFICIAL_INFERENCE_PROTOCOL
_MAX_FORMAL_CLASSES = 10
_MAX_FORMAL_NO_OP_RECONSTRUCTION_MSE = 0.01
_MAX_FORMAL_NO_OP_PROBABILITY_DEVIATION = 0.02
_MAX_FORMAL_NO_OP_ACCURACY_DIFFERENCE = 0.005
_PUBLIC_REPRESENTATION_QUALIFICATION_FIELDS = {
    "metric_split",
    "held_out_validation",
    "explained_variance",
    "minimum_explained_variance",
    "activation_fidelity_passed",
    "validation_by_condition",
    "worst_condition_explained_variance",
    "native_score_gate",
}
_CANONICAL_CONDITIONS = {"rope", "temporary", "none"}
_ROSTER_SPLITS = {"discovery", "validation", "held_out"}


def run(args: Any) -> int:
    """Run and atomically publish one strict official model-causal evaluation."""

    configuration = load_verified_json_config(Path(args.config))
    config = _exact_object(
        configuration.data,
        label="model-causal config",
        required=_MODEL_CAUSAL_TOP_LEVEL - {"expected_parent_manifest_sha256"},
        optional={"expected_parent_manifest_sha256"},
    )
    dataset_config = _exact_object(
        config["dataset"],
        label="dataset",
        required={
            "dataset_id",
            "dataset_dir",
            "fit_split",
            "roster_split",
            "sample_roster_path",
            "trusted_pickle",
        },
        optional={"evaluation_split", "expected_sample_roster_sha256"},
    )
    intervention = _exact_object(
        config["intervention"],
        label="intervention",
        required={
            "site",
            "target_features",
            "control_features",
            "latent_baseline",
            "freeze_artifact_path",
            "random_seed",
            "random_candidate_pool_size",
            "max_no_op_reconstruction_mse",
            "max_no_op_probability_deviation",
            "max_no_op_accuracy_difference",
        },
        optional={
            "expected_freeze_artifact_sha256",
            "selection_run_dir",
            "expected_selection_manifest_sha256",
        },
    )
    classifier_config = _exact_object(
        config["official_classifier"],
        label="official_classifier",
        required={"device", "estimator_options"},
    )

    dataset_id = require_public_label(
        _required_string(dataset_config, "dataset_id"), name="dataset_id"
    )
    if dataset_config["fit_split"] != "train":
        raise ValueError("model-causal fit_split must be exactly 'train'")
    roster_split = dataset_config["roster_split"]
    if roster_split not in _ROSTER_SPLITS:
        raise ValueError(
            "roster_split must be discovery, validation, or held_out"
        )
    evaluation_split = dataset_config.get("evaluation_split", "test")
    if evaluation_split not in {"val", "test"}:
        raise ValueError("evaluation_split must be 'val' or 'test', never train")
    required_evaluation_split = "test" if roster_split == "held_out" else "val"
    if evaluation_split != required_evaluation_split:
        raise ValueError(
            f"roster_split={roster_split!r} requires "
            f"evaluation_split={required_evaluation_split!r}"
        )
    evidence_scope = _evidence_scope_for_roster_split(roster_split)
    trusted_pickle = dataset_config["trusted_pickle"]
    if not isinstance(trusted_pickle, bool):
        raise TypeError("trusted_pickle must be a JSON boolean")

    site = require_portable_identifier(
        _required_string(intervention, "site"), name="site"
    )
    random_seed = _non_negative_integer(
        intervention["random_seed"], name="random_seed"
    )
    random_pool = _positive_integer(
        intervention["random_candidate_pool_size"],
        name="random_candidate_pool_size",
    )
    mse_limit = _finite_non_negative(
        intervention["max_no_op_reconstruction_mse"],
        name="max_no_op_reconstruction_mse",
    )
    probability_limit = _finite_non_negative(
        intervention["max_no_op_probability_deviation"],
        name="max_no_op_probability_deviation",
    )
    accuracy_limit = _finite_non_negative(
        intervention["max_no_op_accuracy_difference"],
        name="max_no_op_accuracy_difference",
    )
    if probability_limit >= 1.0 or accuracy_limit >= 1.0:
        raise ValueError("probability and accuracy no-op limits are invalid")
    formal_limits = {
        "max_no_op_reconstruction_mse": (
            mse_limit,
            _MAX_FORMAL_NO_OP_RECONSTRUCTION_MSE,
        ),
        "max_no_op_probability_deviation": (
            probability_limit,
            _MAX_FORMAL_NO_OP_PROBABILITY_DEVIATION,
        ),
        "max_no_op_accuracy_difference": (
            accuracy_limit,
            _MAX_FORMAL_NO_OP_ACCURACY_DIFFERENCE,
        ),
    }
    relaxed = {
        name: (value, maximum)
        for name, (value, maximum) in formal_limits.items()
        if value > maximum
    }
    if relaxed:
        raise ValueError(
            f"formal no-op thresholds exceed protocol maxima: {relaxed}"
        )
    target_features = intervention["target_features"]
    if not isinstance(target_features, list):
        raise TypeError("target_features must be a JSON list")
    configured_controls = intervention["control_features"]
    if configured_controls is not None and not isinstance(
        configured_controls, list
    ):
        raise TypeError("control_features must be a JSON list or null")
    latent_baseline = intervention["latent_baseline"]
    if roster_split == "held_out":
        if configured_controls is None or not configured_controls:
            raise ValueError(
                "held_out evaluation requires frozen control_features"
            )
        latent_baseline = _frozen_numerical_baseline(latent_baseline)

    selection_parent: RunManifest | None = None
    selection_manifest_file = None
    selection_summary_file = None
    raw_selection_dir = intervention.get("selection_run_dir")
    expected_selection_manifest = intervention.get(
        "expected_selection_manifest_sha256"
    )
    if roster_split == "held_out":
        if raw_selection_dir is None or expected_selection_manifest is None:
            raise ValueError(
                "held_out evaluation requires a digest-bound validation "
                "selection_run_dir"
            )
        selection_dir = _absolute_directory(
            raw_selection_dir, name="selection_run_dir"
        )
        selection_parent = verify_run_directory(selection_dir)
        selection_manifest_file = verify_file(
            selection_dir / "manifest.json",
            expected_sha256=_optional_sha256(
                intervention, "expected_selection_manifest_sha256"
            ),
        )
        if _manifest_from_verified_bytes(selection_manifest_file) != (
            selection_parent
        ):
            raise RuntimeError(
                "selection manifest changed during directory verification"
            )
        selection_summary_artifact = next(
            (
                artifact
                for artifact in selection_parent.artifacts
                if artifact.name == "summary.json"
            ),
            None,
        )
        if selection_summary_artifact is None:
            raise ValueError("selection parent must declare summary.json")
        selection_summary_file = verify_file(
            selection_dir / "summary.json",
            expected_sha256=selection_summary_artifact.sha256,
        )
        if (
            selection_summary_file.digest.size_bytes
            != selection_summary_artifact.size_bytes
        ):
            raise ValueError("selection summary size differs from its manifest")
    elif raw_selection_dir is not None or expected_selection_manifest is not None:
        raise ValueError(
            "validation selection lineage is only valid for held_out evaluation"
        )

    device = _required_string(classifier_config, "device")
    estimator_options = classifier_config["estimator_options"]
    if not isinstance(estimator_options, Mapping):
        raise TypeError("official_classifier.estimator_options must be an object")
    estimator_options = dict(estimator_options)
    estimator_random_state = estimator_options.get("random_state")
    if (
        isinstance(estimator_random_state, bool)
        or not isinstance(estimator_random_state, int)
        or estimator_random_state < 0
    ):
        raise ValueError(
            "official_classifier.estimator_options.random_state must be an "
            "explicit non-negative integer"
        )
    if estimator_random_state != random_seed:
        raise ValueError(
            "official classifier random_state must equal intervention.random_seed"
        )

    parent_dir = _absolute_directory(
        config["representation_run_dir"], name="representation_run_dir"
    )
    parent = verify_run_directory(parent_dir)
    parent_manifest_file = verify_file(
        parent_dir / "manifest.json",
        expected_sha256=_optional_sha256(
            config, "expected_parent_manifest_sha256"
        ),
    )
    parent_from_bound_bytes = _manifest_from_verified_bytes(parent_manifest_file)
    if parent_from_bound_bytes != parent:
        raise RuntimeError("parent manifest changed during directory verification")
    _validate_parent_manifest(parent, site=site)
    model_artifact = next(
        (artifact for artifact in parent.artifacts if artifact.name == "model.pt"),
        None,
    )
    if model_artifact is None:
        raise ValueError("train-repr parent manifest does not declare model.pt")
    representation_file = verify_file(
        parent_dir / "model.pt", expected_sha256=model_artifact.sha256
    )
    if representation_file.digest.size_bytes != model_artifact.size_bytes:
        raise ValueError("train-repr model.pt size differs from its manifest")

    sample_roster_file = verify_file(
        _absolute_file(
            dataset_config["sample_roster_path"], name="sample_roster_path"
        ),
        expected_sha256=_optional_sha256(
            dataset_config, "expected_sample_roster_sha256"
        ),
    )
    sample_roster = _load_sample_roster(
        sample_roster_file,
        dataset_id=dataset_id,
        evaluation_split=evaluation_split,
    )
    freeze_file = None
    raw_freeze_path = intervention["freeze_artifact_path"]
    if roster_split == "held_out" and (
        intervention.get("expected_freeze_artifact_sha256") is None
    ):
        raise ValueError(
            "held_out evaluation requires an expected freeze artifact digest"
        )
    if raw_freeze_path is not None:
        freeze_file = verify_file(
            _absolute_file(
                raw_freeze_path, name="freeze_artifact_path"
            ),
            expected_sha256=_optional_sha256(
                intervention, "expected_freeze_artifact_sha256"
            ),
        )
    elif "expected_freeze_artifact_sha256" in intervention:
        raise ValueError(
            "expected_freeze_artifact_sha256 requires freeze_artifact_path"
        )
    if roster_split == "held_out" and freeze_file is None:
        raise ValueError(
            "held_out evaluation requires a verified freeze artifact"
        )
    raw_dataset = load_raw_talent_splits(
        _absolute_directory(dataset_config["dataset_dir"], name="dataset_dir"),
        trusted_pickle=trusted_pickle,
    )
    evaluated_split = getattr(raw_dataset, evaluation_split)
    row_indices = sample_roster["row_indices"]
    if max(row_indices) >= len(evaluated_split.y):
        raise IndexError("sample roster row index is outside the evaluation split")

    raw_root = _absolute_directory(dataset_config["dataset_dir"], name="dataset_dir")
    additional_paths: dict[str, Path] = {
        "representation.parent_manifest": parent_manifest_file.path,
        "representation.model": representation_file.path,
        "samples.roster": sample_roster_file.path,
    }
    expected_additional: dict[str, str] = {
        "representation.parent_manifest": parent_manifest_file.digest.sha256,
        "representation.model": representation_file.digest.sha256,
        "samples.roster": sample_roster_file.digest.sha256,
    }
    if freeze_file is not None:
        additional_paths["intervention.freeze"] = freeze_file.path
        expected_additional["intervention.freeze"] = (
            freeze_file.digest.sha256
        )
    if selection_manifest_file is not None:
        assert selection_summary_file is not None
        additional_paths["selection.parent_manifest"] = (
            selection_manifest_file.path
        )
        additional_paths["selection.summary"] = selection_summary_file.path
        expected_additional["selection.parent_manifest"] = (
            selection_manifest_file.digest.sha256
        )
        expected_additional["selection.summary"] = (
            selection_summary_file.digest.sha256
        )
    for name, digest in sorted(raw_dataset.input_sha256.items()):
        role = f"talent.raw.{name}"
        additional_paths[role] = raw_root / name
        expected_additional[role] = digest

    context = verify_configured_run_inputs(
        configuration,
        command="model-causal",
        seed=random_seed,
        additional_input_paths=additional_paths,
        expected_additional_sha256=expected_additional,
    )
    if context.inputs.evidence_level != "strict":
        raise RuntimeError("model-causal runs require strict Git evidence")
    if context.model_family != "tabicl-v2":
        raise ValueError("official model-causal supports only tabicl-v2")
    if context.sites != (site,):
        raise ValueError("provenance sites must contain exactly intervention.site")
    if context.condition not in _CANONICAL_CONDITIONS:
        raise ValueError(
            "model-causal condition must be one of rope, temporary, or none"
        )
    assert_dataset_roster(
        context.inputs.dataset_manifest,
        (dataset_id,),
        required_split=roster_split,
    )

    bound_representation = context.additional_file("representation.model")
    autoencoder, normalizer, representation_metadata = (
        load_verified_representation_checkpoint(bound_representation)
    )
    inference_contract_sha256 = _model_causal_inference_contract_sha256(
        context=context,
        estimator_options=estimator_options,
    )
    source_lineage = _validated_representation_source_lineage(
        representation_metadata,
        parent=parent,
        context=context,
        inference_contract_sha256=inference_contract_sha256,
    )
    _validate_parent_lineage(
        parent,
        context=context,
        seed=random_seed,
        reference_condition=source_lineage["reference_condition"],
    )
    if selection_parent is not None:
        assert selection_manifest_file is not None
        assert selection_summary_file is not None
        _validate_selection_parent(
            selection_parent,
            selection_summary_file,
            context=context,
            dataset_id=dataset_id,
            site=site,
            random_seed=random_seed,
            target_features=target_features,
            control_features=configured_controls,
            latent_baseline=latent_baseline,
            representation_model_sha256=bound_representation.digest.sha256,
            representation_parent_manifest_sha256=context.additional_file(
                "representation.parent_manifest"
            ).digest.sha256,
            inference_contract_sha256=inference_contract_sha256,
            source_lineage=source_lineage,
        )
    bound_freeze = (
        None
        if freeze_file is None
        else context.additional_file("intervention.freeze")
    )
    if bound_freeze is not None:
        _validate_freeze_artifact(
            bound_freeze,
            condition=context.condition,
            site=site,
            random_seed=random_seed,
            sample_roster_sha256=context.additional_file(
                "samples.roster"
            ).digest.sha256,
            target_features=target_features,
            control_features=configured_controls,
            latent_baseline=latent_baseline,
            representation_model_sha256=bound_representation.digest.sha256,
            representation_parent_manifest_sha256=context.additional_file(
                "representation.parent_manifest"
            ).digest.sha256,
            model_sha=context.inputs.model_code.head_sha,
            checkpoint_sha256=context.inputs.checkpoint.digest.sha256,
            inference_contract_sha256=inference_contract_sha256,
            selection_parent_manifest_sha256=(
                None
                if selection_manifest_file is None
                else selection_manifest_file.digest.sha256
            ),
            selection_summary_sha256=(
                None
                if selection_summary_file is None
                else selection_summary_file.digest.sha256
            ),
        )
    preprocessing_roster = {
        "fit_split": "train",
        "evaluation_split": evaluation_split,
        "raw_input_sha256": dict(sorted(raw_dataset.input_sha256.items())),
        "sample_roster_sha256": context.additional_file(
            "samples.roster"
        ).digest.sha256,
        "protocol": _PREPROCESSING_PROTOCOL,
        "estimator_options": estimator_options,
    }
    preprocessing_roster_sha256 = _canonical_sha256(preprocessing_roster)
    qualification = _bound_qualification(
        representation_metadata,
        {
            "model_sha": context.inputs.model_code.head_sha,
            "checkpoint_sha": context.inputs.checkpoint.digest.sha256,
            "site": site,
            "representation_model_sha256": bound_representation.digest.sha256,
            "parent_manifest_sha256": context.additional_file(
                "representation.parent_manifest"
            ).digest.sha256,
            "fit_context": "talent-train",
            "preprocessing_protocol": _PREPROCESSING_PROTOCOL,
            "preprocessing_roster_sha256": preprocessing_roster_sha256,
            "sample_roster_sha256": context.additional_file(
                "samples.roster"
            ).digest.sha256,
            "inference_contract_sha256": inference_contract_sha256,
            "freeze_artifact_sha256": (
                None
                if bound_freeze is None
                else bound_freeze.digest.sha256
            ),
        },
        expected_conditions=set(
            source_lineage["condition_checkpoints_sha256"]
        ),
    )

    driver = fit_official_talent_driver(
        raw_dataset,
        context.inputs.checkpoint.path,
        context_split="train",
        device=device,
        model_sha=context.inputs.model_code.head_sha,
        estimator_options=estimator_options,
        expected_source_root=context.inputs.model_code.root,
    )
    _validate_official_driver(driver, context=context)

    selected_X = evaluated_split.X.iloc[list(row_indices)].copy()
    selected_y = np.asarray(evaluated_split.y)[list(row_indices)].copy()
    evaluation = run_official_tabicl_causal_edits(
        driver,
        selected_X,
        selected_y,
        site=site,
        autoencoder=autoencoder,
        normalizer=normalizer,
        target_features=target_features,
        dataset_id=dataset_id,
        sample_ids=sample_roster["sample_ids"],
        representation_qualification=qualification,
        control_features=configured_controls,
        latent_baseline=latent_baseline,
        random_seed=random_seed,
        random_candidate_pool_size=random_pool,
        max_no_op_reconstruction_mse=mse_limit,
        max_no_op_probability_deviation=probability_limit,
        max_no_op_accuracy_drop=accuracy_limit,
    )
    if evaluation.source_evidence_level != "strict":
        raise RuntimeError("official classifier source evidence was not strict")

    predictions = _predictions_payload(
        evaluation,
        true_labels=selected_y,
        evaluation_split=evaluation_split,
        roster_split=roster_split,
        evidence_scope=evidence_scope,
    )
    summary = _summary_payload(
        evaluation,
        evaluation_split=evaluation_split,
        roster_split=roster_split,
        evidence_scope=evidence_scope,
        raw_dataset=raw_dataset,
        parent_manifest_sha256=context.additional_file(
            "representation.parent_manifest"
        ).digest.sha256,
        representation_model_sha256=bound_representation.digest.sha256,
        preprocessing_roster_sha256=preprocessing_roster_sha256,
        inference_contract_sha256=inference_contract_sha256,
        freeze_artifact_sha256=(
            None if bound_freeze is None else bound_freeze.digest.sha256
        ),
        selection_parent_manifest_sha256=(
            None
            if selection_manifest_file is None
            else selection_manifest_file.digest.sha256
        ),
        source_lineage=source_lineage,
        target_features=target_features,
        control_features=configured_controls,
        latent_baseline=latent_baseline,
        random_seed=random_seed,
        thresholds={
            "max_no_op_reconstruction_mse": mse_limit,
            "max_no_op_probability_deviation": probability_limit,
            "max_no_op_accuracy_difference": accuracy_limit,
        },
    )

    roots = tuple(
        dict.fromkeys(
            (
                context.inputs.training_code.root,
                context.inputs.model_code.root,
                context.inputs.analysis_code.root,
            )
        )
    )
    with RunTransaction(Path(args.output_dir), source_roots=roots) as transaction:
        _write_json_artifact(
            transaction.staging_dir / "predictions.json", predictions
        )
        _write_json_artifact(transaction.staging_dir / "summary.json", summary)
        artifacts = transaction.artifact_digests(
            ("predictions.json", "summary.json")
        )
        manifest = manifest_from_verified_inputs(
            context.inputs, artifacts=artifacts
        )
        transaction.commit(manifest, verified_inputs=context.inputs)
    return 0


def _exact_object(
    value: Any,
    *,
    label: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    optional = optional or set()
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing or unknown:
        raise ValueError(
            f"{label} fields mismatch: missing={missing}, unknown={unknown}"
        )
    return dict(value)


def _required_string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{key} must be a non-empty string")
    return value


def _optional_sha256(values: Mapping[str, Any], key: str) -> str | None:
    value = values.get(key)
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    return value


def _absolute_directory(value: Any, *, name: str) -> Path:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string path")
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    if raw.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    path = raw.resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"{name} must be a directory")
    return path


def _absolute_file(value: Any, *, name: str) -> Path:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string path")
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    if raw.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    path = raw.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{name} must be a regular file")
    return path


def _non_negative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: Any, *, name: str) -> int:
    result = _non_negative_integer(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_non_negative(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numerical")
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _frozen_numerical_baseline(value: Any) -> float | list[float]:
    if isinstance(value, bool):
        raise TypeError("held_out latent_baseline must be numerical")
    if isinstance(value, (int, float)):
        return _finite_number(value, name="latent_baseline")
    if not isinstance(value, list) or not value:
        raise TypeError(
            "held_out latent_baseline must be a number or non-empty number list"
        )
    return [
        _finite_number(item, name="latent_baseline item") for item in value
    ]


def _evidence_scope_for_roster_split(roster_split: str) -> str:
    try:
        return {
            "discovery": "exploratory-discovery",
            "validation": "exploratory-feature-selection",
            "held_out": "confirmatory-held-out",
        }[roster_split]
    except KeyError as error:
        raise ValueError("roster_split is not registered") from error


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numerical")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _manifest_from_verified_bytes(verified: Any) -> RunManifest:
    try:
        payload = json.loads(verified.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("parent manifest is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("parent manifest must contain a JSON object")
    return RunManifest.from_dict(payload)


def _validate_parent_manifest(parent: RunManifest, *, site: str) -> None:
    if parent.command != "train-repr":
        raise ValueError("representation parent command must be train-repr")
    if parent.evidence_level != "strict" or parent.legacy_reasons:
        raise ValueError("representation parent must have strict evidence")
    if parent.sites != (site,):
        raise ValueError("representation parent site differs from intervention.site")
    if sum(artifact.name == "model.pt" for artifact in parent.artifacts) != 1:
        raise ValueError("representation parent must declare exactly one model.pt")


def _validate_parent_lineage(
    parent: RunManifest,
    *,
    context: Any,
    seed: int,
    reference_condition: str,
) -> None:
    assert_git_commit_is_ancestor(
        context.inputs.analysis_code, parent.analysis_code_sha
    )
    expected = {
        "model_family": context.model_family,
        "model_revision": context.model_revision,
        "training_code_sha": context.inputs.training_code.head_sha,
        "model_code_sha": context.inputs.model_code.head_sha,
        "dataset_manifest": context.inputs.dataset_manifest.digest,
        "condition": reference_condition,
        "sites": context.sites,
        "seed": seed,
    }
    mismatches = {
        name: (getattr(parent, name), value)
        for name, value in expected.items()
        if getattr(parent, name) != value
    }
    if mismatches:
        raise ValueError(
            f"representation parent lineage differs from model-causal run: {mismatches}"
        )


def _validated_representation_source_lineage(
    metadata: Mapping[str, Any],
    *,
    parent: RunManifest,
    context: Any,
    inference_contract_sha256: str,
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError("representation checkpoint metadata must be an object")
    public_metadata = metadata.get("metadata")
    if not isinstance(public_metadata, Mapping):
        raise ValueError("representation checkpoint lacks public metadata")
    lineage = _exact_object(
        public_metadata.get("source_lineage"),
        label="representation source_lineage",
        required={
            "schema_version",
            "source_kind",
            "reference_condition",
            "condition_checkpoints_sha256",
            "collect_parent_manifests_sha256",
            "alignment_sha256",
            "inference_contract_sha256",
            "evaluation_split",
            "max_classes",
        },
    )
    if lineage["schema_version"] != 1 or lineage["source_kind"] != (
        "official_tabicl_bounded_activation_index"
    ):
        raise ValueError(
            "model-causal requires official coordinate-aligned representation sources"
        )
    reference_condition = lineage["reference_condition"]
    if reference_condition not in _CANONICAL_CONDITIONS:
        raise ValueError("representation reference_condition is not canonical")
    checkpoints = _condition_sha256_mapping(
        lineage["condition_checkpoints_sha256"],
        name="condition_checkpoints_sha256",
    )
    parent_manifests = _condition_digest_list_mapping(
        lineage["collect_parent_manifests_sha256"]
    )
    if set(checkpoints) != set(parent_manifests):
        raise ValueError(
            "representation condition checkpoint and collect-parent rosters differ"
        )
    if reference_condition not in checkpoints:
        raise ValueError("reference_condition is absent from checkpoint lineage")
    if context.condition not in checkpoints:
        raise ValueError("causal condition is absent from representation lineage")
    if checkpoints[reference_condition] != parent.checkpoint.sha256:
        raise ValueError(
            "reference-condition checkpoint differs from train-repr parent manifest"
        )
    if checkpoints[context.condition] != context.inputs.checkpoint.digest.sha256:
        raise ValueError(
            "causal checkpoint differs from its representation condition lineage"
        )

    registered_collect_inputs: set[str] = set()
    for item in parent.inputs:
        prefix = "source.collect_manifest."
        if not item.role.startswith(prefix):
            continue
        suffix = item.role.removeprefix(prefix)
        if suffix != item.sha256:
            raise ValueError(
                "train-repr collect-manifest input role/digest is inconsistent"
            )
        registered_collect_inputs.add(item.sha256)
    claimed_collect_inputs = {
        digest for values in parent_manifests.values() for digest in values
    }
    if claimed_collect_inputs != registered_collect_inputs:
        raise ValueError(
            "representation collect-parent lineage differs from hashed manifest inputs"
        )

    if lineage["inference_contract_sha256"] != inference_contract_sha256:
        raise ValueError(
            "causal official inference contract differs from representation sources"
        )
    if lineage["evaluation_split"] != "val":
        raise ValueError(
            "representation sources must use validation-row activation evidence"
        )
    max_classes = _positive_integer(
        lineage["max_classes"], name="source_lineage.max_classes"
    )
    if max_classes > _MAX_FORMAL_CLASSES:
        raise ValueError("representation source max_classes exceeds protocol")
    _validate_alignment_roster(
        lineage["alignment_sha256"],
        site=context.sites[0],
    )
    lineage["condition_checkpoints_sha256"] = checkpoints
    lineage["collect_parent_manifests_sha256"] = parent_manifests
    return lineage


def _condition_sha256_mapping(value: Any, *, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty object")
    result: dict[str, str] = {}
    for raw_condition, digest in value.items():
        if not isinstance(raw_condition, str) or raw_condition not in (
            _CANONICAL_CONDITIONS
        ):
            raise ValueError(f"{name} contains a non-canonical condition")
        result[raw_condition] = _required_sha256_value(
            digest, name=f"{name}.{raw_condition}"
        )
    return dict(sorted(result.items()))


def _condition_digest_list_mapping(value: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(
            "collect_parent_manifests_sha256 must be a non-empty object"
        )
    result: dict[str, tuple[str, ...]] = {}
    for raw_condition, raw_digests in value.items():
        if not isinstance(raw_condition, str) or raw_condition not in (
            _CANONICAL_CONDITIONS
        ):
            raise ValueError(
                "collect parent lineage contains a non-canonical condition"
            )
        if not isinstance(raw_digests, list) or not raw_digests:
            raise ValueError("each condition requires collect parent manifests")
        digests = tuple(
            _required_sha256_value(
                digest, name="collect parent manifest digest"
            )
            for digest in raw_digests
        )
        if digests != tuple(sorted(set(digests))):
            raise ValueError(
                "collect parent manifest digests must be sorted and unique"
            )
        result[raw_condition] = digests
    return dict(sorted(result.items()))


def _required_sha256_value(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_alignment_roster(value: Any, *, site: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"training", "validation"}:
        raise ValueError(
            "official representation alignment must contain training and validation"
        )
    for split, raw_datasets in value.items():
        if not isinstance(raw_datasets, Mapping) or not raw_datasets:
            raise ValueError(f"{split} alignment dataset roster must be non-empty")
        for dataset_id, raw_sites in raw_datasets.items():
            require_public_label(dataset_id, name="alignment dataset_id")
            if not isinstance(raw_sites, Mapping) or set(raw_sites) != {site}:
                raise ValueError("alignment site roster differs from causal site")
            _required_sha256_value(
                raw_sites[site], name="alignment digest"
            )


def _load_sample_roster(
    verified: Any, *, dataset_id: str, evaluation_split: str
) -> dict[str, tuple[Any, ...]]:
    try:
        payload = json.loads(verified.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("sample roster is not valid JSON") from error
    roster = _exact_object(
        payload,
        label="sample roster",
        required={"dataset_id", "split", "row_indices", "sample_ids"},
    )
    if roster["dataset_id"] != dataset_id or roster["split"] != evaluation_split:
        raise ValueError("sample roster dataset_id/split differs from the config")
    indices = roster["row_indices"]
    sample_ids = roster["sample_ids"]
    if (
        not isinstance(indices, list)
        or not indices
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        )
    ):
        raise ValueError("sample roster row_indices must be non-negative integers")
    if len(set(indices)) != len(indices):
        raise ValueError("sample roster row_indices must be unique")
    resolved_ids = _sample_ids(sample_ids, expected=len(indices))
    return {"row_indices": tuple(indices), "sample_ids": resolved_ids}


def _bound_qualification(
    metadata: Mapping[str, Any],
    bindings: Mapping[str, Any],
    *,
    expected_conditions: set[str],
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError("representation checkpoint metadata must be an object")
    qualification = metadata.get("qualification")
    if not isinstance(qualification, Mapping):
        raise ValueError("representation checkpoint lacks qualification metadata")
    conflicts = {
        name: (qualification[name], value)
        for name, value in bindings.items()
        if name in qualification and qualification[name] != value
    }
    if conflicts:
        raise ValueError(
            f"representation qualification binding conflicts: {conflicts}"
        )
    missing = sorted(_PUBLIC_REPRESENTATION_QUALIFICATION_FIELDS - set(qualification))
    unknown = sorted(set(qualification) - _PUBLIC_REPRESENTATION_QUALIFICATION_FIELDS)
    if missing or unknown:
        raise ValueError(
            "representation qualification fields mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    result = {
        name: qualification[name]
        for name in sorted(_PUBLIC_REPRESENTATION_QUALIFICATION_FIELDS)
    }
    if (
        result["metric_split"] != "validation"
        or result["held_out_validation"] is not True
        or result["activation_fidelity_passed"] is not True
        or result["native_score_gate"] != "pending"
    ):
        raise ValueError(
            "representation parent did not pass held-out activation fidelity"
        )
    for name in ("explained_variance", "minimum_explained_variance"):
        _finite_non_negative(result[name], name=name)
    explained = float(result["explained_variance"])
    minimum = float(result["minimum_explained_variance"])
    if minimum < MIN_HELDOUT_EXPLAINED_VARIANCE or explained < minimum:
        raise ValueError(
            "representation qualification explained variance is inconsistent "
            "with the registered held-out threshold"
        )
    condition_metrics = result["validation_by_condition"]
    if not isinstance(condition_metrics, Mapping) or set(condition_metrics) != (
        expected_conditions
    ):
        raise ValueError(
            "representation qualification condition roster differs from "
            "source lineage"
        )
    metric_fields = {
        "explained_variance",
        "normalized_mse",
        "mse",
        "dead_features",
        "dead_feature_fraction",
        "active_count",
        "mean_active_features",
    }
    condition_explained: list[float] = []
    normalized_metrics: dict[str, dict[str, float | int]] = {}
    for condition in sorted(expected_conditions):
        values = condition_metrics[condition]
        if not isinstance(values, Mapping) or set(values) != metric_fields:
            raise ValueError(
                f"validation_by_condition.{condition} metric fields mismatch"
            )
        for name in metric_fields - {"dead_features"}:
            _finite_non_negative(
                values[name], name=f"validation_by_condition.{condition}.{name}"
            )
        dead_features = values["dead_features"]
        if (
            isinstance(dead_features, bool)
            or not isinstance(dead_features, int)
            or dead_features < 0
        ):
            raise ValueError(
                f"validation_by_condition.{condition}.dead_features must be "
                "a non-negative integer"
            )
        condition_ev = float(values["explained_variance"])
        if condition_ev < minimum:
            raise ValueError(
                f"validation_by_condition.{condition} explained variance is "
                "below the registered threshold"
            )
        condition_explained.append(condition_ev)
        normalized_metrics[condition] = dict(values)
    worst = _finite_non_negative(
        result["worst_condition_explained_variance"],
        name="worst_condition_explained_variance",
    )
    if worst < minimum or worst != min(condition_explained):
        raise ValueError(
            "worst-condition explained variance is inconsistent with "
            "validation_by_condition"
        )
    result["validation_by_condition"] = normalized_metrics
    result["worst_condition_explained_variance"] = worst
    result.update(bindings)
    return result


def _validate_official_driver(driver: OfficialTabICLDriver, *, context: Any) -> None:
    if driver.source_evidence_level != "strict":
        raise RuntimeError("formal driver source evidence is not strict")
    if driver.fit_context != "talent-train":
        raise RuntimeError("formal driver did not fit only the TALENT train split")
    if (
        driver.model_sha != context.inputs.model_code.head_sha
        or driver.checkpoint_sha != context.inputs.checkpoint.digest.sha256
    ):
        raise RuntimeError("formal driver model/checkpoint binding differs from inputs")
    n_classes = int(driver.estimator.n_classes_)
    if n_classes > _MAX_FORMAL_CLASSES:
        raise ValueError(
            f"formal driver supports at most {_MAX_FORMAL_CLASSES} native classes"
        )
    column_embedder = driver.estimator.model_.col_embedder
    feature_group = getattr(column_embedder, "feature_group", None)
    if feature_group is not True and feature_group != "same":
        raise ValueError("official driver requires same feature grouping")
    raw = driver.estimator.model_
    row = getattr(raw, "row_interactor", None)
    raw_mode = getattr(raw, "row_identity_mode", None)
    row_mode = getattr(row, "identity_mode", None)
    if raw_mode != row_mode:
        raise RuntimeError("raw and row-interactor identity modes disagree")
    if raw_mode != context.condition:
        raise RuntimeError(
            "checkpoint row identity mode differs from causal condition"
        )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_causal_inference_contract_sha256(
    *, context: Any, estimator_options: Mapping[str, Any]
) -> str:
    return official_inference_contract_sha256(
        context.inputs.model_code.head_sha,
        estimator_options,
    )


def _validate_selection_parent(
    parent: RunManifest,
    verified_summary: Any,
    *,
    context: Any,
    dataset_id: str,
    site: str,
    random_seed: int,
    target_features: Sequence[int],
    control_features: Sequence[int] | None,
    latent_baseline: Any,
    representation_model_sha256: str,
    representation_parent_manifest_sha256: str,
    inference_contract_sha256: str,
    source_lineage: Mapping[str, Any],
) -> None:
    """Bind held-out choices to an earlier strict validation evaluation."""

    if parent.command != "model-causal" or parent.evidence_level != "strict":
        raise ValueError(
            "held_out selection parent must be a strict model-causal run"
        )
    assert_git_commit_is_ancestor(
        context.inputs.analysis_code, parent.analysis_code_sha
    )
    expected_manifest = {
        "model_family": context.model_family,
        "model_revision": context.model_revision,
        "training_code_sha": context.inputs.training_code.head_sha,
        "model_code_sha": context.inputs.model_code.head_sha,
        "checkpoint": context.inputs.checkpoint.digest,
        "dataset_manifest": context.inputs.dataset_manifest.digest,
        "condition": context.condition,
        "sites": context.sites,
        "seed": random_seed,
    }
    mismatches = {
        name: (getattr(parent, name), expected)
        for name, expected in expected_manifest.items()
        if getattr(parent, name) != expected
    }
    if mismatches:
        raise ValueError(
            f"validation selection lineage differs from held_out run: {mismatches}"
        )
    registered_inputs = {item.role: item.sha256 for item in parent.inputs}
    expected_inputs = {
        "representation.model": representation_model_sha256,
        "representation.parent_manifest": (
            representation_parent_manifest_sha256
        ),
    }
    if any(
        registered_inputs.get(role) != digest
        for role, digest in expected_inputs.items()
    ):
        raise ValueError(
            "validation selection representation inputs differ from held_out run"
        )
    try:
        summary = json.loads(verified_summary.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("selection summary is not valid JSON") from error
    if not isinstance(summary, Mapping):
        raise ValueError("selection summary must contain a JSON object")
    expected_summary = {
        "analysis": "official-model-causal",
        "fit_split": "train",
        "evaluation_split": "val",
        "roster_split": "validation",
        "evidence_scope": "exploratory-feature-selection",
        "site": site,
        "source_evidence_level": "strict",
    }
    summary_mismatches = {
        name: (summary.get(name), expected)
        for name, expected in expected_summary.items()
        if summary.get(name) != expected
    }
    if summary_mismatches:
        raise ValueError(
            "validation selection summary scope differs from held_out run: "
            f"{summary_mismatches}"
        )
    selection_dataset_id = require_public_label(
        summary.get("dataset_id"), name="selection dataset_id"
    )
    if selection_dataset_id == dataset_id:
        raise ValueError(
            "validation selection and held_out evaluation must use disjoint datasets"
        )
    assert_dataset_roster(
        context.inputs.dataset_manifest,
        (selection_dataset_id,),
        required_split="validation",
    )
    bindings = summary.get("input_bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("validation selection summary lacks input_bindings")
    expected_bindings = {
        "model_sha": context.inputs.model_code.head_sha,
        "checkpoint_sha256": context.inputs.checkpoint.digest.sha256,
        "parent_manifest_sha256": representation_parent_manifest_sha256,
        "representation_model_sha256": representation_model_sha256,
        "inference_contract_sha256": inference_contract_sha256,
    }
    if any(bindings.get(name) != value for name, value in expected_bindings.items()):
        raise ValueError(
            "validation selection summary input bindings differ from held_out run"
        )
    expected_intervention = {
        "target_features": list(target_features),
        "control_features": (
            None if control_features is None else list(control_features)
        ),
        "latent_baseline": _json_safe(latent_baseline),
        "random_seed": random_seed,
    }
    if summary.get("intervention") != expected_intervention:
        raise ValueError(
            "held_out intervention was not frozen by the validation selection"
        )
    if summary.get("representation_source_lineage") != _json_safe(source_lineage):
        raise ValueError(
            "validation selection representation source lineage differs"
        )
    no_op = summary.get("no_op_gates")
    if not isinstance(no_op, Mapping) or no_op.get("passed") is not True:
        raise ValueError("validation selection did not pass the no-op gates")


def _validate_freeze_artifact(
    verified: Any,
    *,
    condition: str,
    site: str,
    random_seed: int,
    sample_roster_sha256: str,
    target_features: Sequence[int],
    control_features: Sequence[int] | None,
    latent_baseline: Any,
    representation_model_sha256: str,
    representation_parent_manifest_sha256: str,
    model_sha: str,
    checkpoint_sha256: str,
    inference_contract_sha256: str,
    selection_parent_manifest_sha256: str | None,
    selection_summary_sha256: str | None,
) -> None:
    try:
        payload = json.loads(verified.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("freeze artifact is not valid JSON") from error
    artifact = _exact_object(
        payload,
        label="freeze artifact",
        required={
            "schema_version",
            "evidence_scope",
            "condition",
            "site",
            "random_seed",
            "evaluation_sample_roster_sha256",
            "target_features",
            "control_features",
            "latent_baseline",
            "representation_model_sha256",
            "representation_parent_manifest_sha256",
            "model_sha",
            "checkpoint_sha256",
            "inference_contract_sha256",
            "selection_parent_manifest_sha256",
            "selection_summary_sha256",
        },
    )
    if artifact["schema_version"] != 1:
        raise ValueError("unsupported freeze artifact schema_version")
    expected = {
        "evidence_scope": "validation-frozen",
        "condition": condition,
        "site": site,
        "random_seed": random_seed,
        "evaluation_sample_roster_sha256": sample_roster_sha256,
        "target_features": list(target_features),
        "control_features": (
            None if control_features is None else list(control_features)
        ),
        "latent_baseline": _json_safe(latent_baseline),
        "representation_model_sha256": representation_model_sha256,
        "representation_parent_manifest_sha256": (
            representation_parent_manifest_sha256
        ),
        "model_sha": model_sha,
        "checkpoint_sha256": checkpoint_sha256,
        "inference_contract_sha256": inference_contract_sha256,
        "selection_parent_manifest_sha256": (
            selection_parent_manifest_sha256
        ),
        "selection_summary_sha256": selection_summary_sha256,
    }
    mismatches = {
        name: (artifact.get(name), value)
        for name, value in expected.items()
        if artifact.get(name) != value
    }
    if mismatches:
        raise ValueError(
            f"freeze artifact differs from resolved causal inputs: {mismatches}"
        )


def _predictions_payload(
    evaluation: OfficialCausalEvaluation,
    *,
    true_labels: np.ndarray,
    evaluation_split: str,
    roster_split: str,
    evidence_scope: str,
) -> dict[str, Any]:
    classes = np.asarray(evaluation.classes)
    native = evaluation.native_prediction
    no_op = evaluation.conditions["no_op_reconstruction"].prediction
    conditions: dict[str, Any] = {}
    for name, condition in sorted(evaluation.conditions.items()):
        probabilities = condition.prediction.probabilities
        conditions[name] = {
            **_prediction_values(probabilities, classes),
            "true_class_log_loss": condition.prediction.true_class_log_loss,
            "delta_log_loss_vs_native": (
                condition.delta_log_loss_vs_model_baseline
            ),
            "delta_log_loss_vs_no_op": (
                condition.delta_log_loss_vs_reconstruction
            ),
            "probability_delta_vs_native": probabilities - native.probabilities,
            "probability_delta_vs_no_op": probabilities - no_op.probabilities,
            "target_features": condition.target_features,
            "control_features": condition.control_features,
        }
    return _json_safe(
        {
            "schema_version": 1,
            "dataset_id": evaluation.dataset_id,
            "evaluation_split": evaluation_split,
            "roster_split": roster_split,
            "evidence_scope": evidence_scope,
            "sample_ids": evaluation.sample_ids,
            "true_labels": true_labels,
            "classes": classes,
            "native": {
                **_prediction_values(native.probabilities, classes),
                "true_class_log_loss": native.true_class_log_loss,
            },
            "conditions": conditions,
        }
    )


def _prediction_values(
    probabilities: np.ndarray, classes: np.ndarray
) -> dict[str, Any]:
    predicted_indices = np.argmax(probabilities, axis=1)
    return {
        "probabilities": probabilities,
        "predicted_class_indices": predicted_indices,
        "predicted_labels": classes[predicted_indices],
    }


def _summary_payload(
    evaluation: OfficialCausalEvaluation,
    *,
    evaluation_split: str,
    roster_split: str,
    evidence_scope: str,
    raw_dataset: Any,
    parent_manifest_sha256: str,
    representation_model_sha256: str,
    preprocessing_roster_sha256: str,
    inference_contract_sha256: str,
    freeze_artifact_sha256: str | None,
    selection_parent_manifest_sha256: str | None,
    source_lineage: Mapping[str, Any],
    target_features: Sequence[int],
    control_features: Sequence[int] | None,
    latent_baseline: Any,
    random_seed: int,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    condition_metrics = {
        name: {
            "accuracy": condition.prediction.accuracy,
            "log_loss": condition.prediction.log_loss,
            "mean_delta_log_loss_vs_native": float(
                np.mean(condition.delta_log_loss_vs_model_baseline)
            ),
            "mean_delta_log_loss_vs_no_op": float(
                np.mean(condition.delta_log_loss_vs_reconstruction)
            ),
        }
        for name, condition in sorted(evaluation.conditions.items())
    }
    schedule = [
        {
            "call_index": call.metadata.call_index,
            "norm_method": call.metadata.norm_method,
            "norm_view_indices": call.metadata.norm_view_indices,
            "ensemble_indices": call.metadata.ensemble_indices,
            "raw_input_shape": call.metadata.raw_input_shape,
            "train_size": call.metadata.train_size,
            "preprocessing_view_id": call.metadata.preprocessing_view_id,
        }
        for call in evaluation.native_forward_calls
    ]
    return _json_safe(
        {
            "schema_version": 1,
            "analysis": "official-model-causal",
            "dataset_id": evaluation.dataset_id,
            "fit_split": "train",
            "evaluation_split": evaluation_split,
            "roster_split": roster_split,
            "evidence_scope": evidence_scope,
            "sample_count": len(evaluation.sample_ids),
            "site": evaluation.site,
            "native": {
                "accuracy": evaluation.native_prediction.accuracy,
                "log_loss": evaluation.native_prediction.log_loss,
            },
            "conditions": condition_metrics,
            "matched_control_features": evaluation.matched_control_features,
            "intervention": {
                "target_features": list(target_features),
                "control_features": (
                    None if control_features is None else list(control_features)
                ),
                "latent_baseline": latent_baseline,
                "random_seed": random_seed,
            },
            "mechanistic_rescue": {
                "passed": evaluation.mechanistic_rescue_passed,
                "status": evaluation.mechanistic_rescue_status,
                "roundtrip_restore_control_passed": True,
            },
            "no_op_gates": {
                "passed": True,
                "reconstruction_mse": evaluation.no_op_reconstruction_mse,
                "signed_accuracy_drop": evaluation.no_op_accuracy_drop,
                "absolute_accuracy_difference": (
                    evaluation.no_op_accuracy_absolute_difference
                ),
                "maximum_absolute_probability_difference": (
                    evaluation.no_op_probability_max_abs_difference
                ),
                "thresholds": dict(thresholds),
            },
            "official_scope": {
                "official_preprocessing_and_ensemble": True,
                "fit_context": evaluation.fit_context,
                "feature_group": "same",
                "kv_cache": False,
                "many_class_recursion": False,
                "maximum_native_classes": _MAX_FORMAL_CLASSES,
                "observed_classes": len(evaluation.classes),
                "preprocessing_protocol": _PREPROCESSING_PROTOCOL,
                "capture_exact_to_direct": evaluation.capture_exact_to_direct,
                "raw_call_schedule": schedule,
            },
            "input_bindings": {
                "model_sha": evaluation.model_sha,
                "checkpoint_sha256": evaluation.checkpoint_sha,
                "parent_manifest_sha256": parent_manifest_sha256,
                "representation_model_sha256": representation_model_sha256,
                "preprocessing_roster_sha256": preprocessing_roster_sha256,
                "inference_contract_sha256": inference_contract_sha256,
                "freeze_artifact_sha256": freeze_artifact_sha256,
                "selection_parent_manifest_sha256": (
                    selection_parent_manifest_sha256
                ),
                "raw_dataset_input_sha256": dict(
                    sorted(raw_dataset.input_sha256.items())
                ),
            },
            "representation_qualification": (
                evaluation.representation_qualification
            ),
            "representation_source_lineage": source_lineage,
            "source_evidence_level": evaluation.source_evidence_level,
        }
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("published JSON object keys must be strings")
            result[key] = _json_safe(child)
        return result
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("published JSON values must be finite")
        return value
    raise TypeError(f"unsupported published JSON value: {type(value).__name__}")


def _write_json_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            _json_safe(payload), sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


__all__ = [
    "OfficialCausalEvaluation",
    "run",
    "run_official_tabicl_causal_edits",
]
