"""Model-causal AE/SAE edits through official TabICL sklearn inference.

All conditions use :class:`~pe_mechanism.official_tabicl.OfficialTabICLDriver`,
so mixed-type encoding, normalization ensembles, feature/class shuffles,
temperature, and aggregation remain owned by the public classifier.  Activation
replacements are indexed by the classifier's exact raw-call schedule rather than
merging or replaying ensemble tables independently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import importlib
import inspect
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
    raw_space_decoder_feature_norms,
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
    verify_git_tree,
    verify_run_directory,
)
from .representation import (
    MIN_HELDOUT_EXPLAINED_VARIANCE,
    MeanRMSNormalizer,
    load_verified_representation_checkpoint,
)
from .statistics import (
    adjust_fdr_arbitrary_dependence,
    paired_bootstrap_ci,
    paired_sign_flip_p_value,
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
    paired_source_condition: str | None
    paired_source_model_sha: str | None
    paired_source_checkpoint_sha: str | None
    paired_source_native_prediction: PredictionMetrics | None
    paired_source_no_op_prediction: PredictionMetrics | None
    paired_source_no_op_reconstruction_mse: float | None
    paired_source_no_op_accuracy_absolute_difference: float | None
    paired_source_no_op_probability_max_abs_difference: float | None
    paired_alignment_verified: bool
    paired_reverse_patch_log_loss_improvement_vs_matched_random: (
        np.ndarray | None
    )
    paired_reverse_patch_log_loss_improvement_vs_no_op: np.ndarray | None
    paired_reverse_patch_log_loss_improvement_vs_target_baseline: (
        np.ndarray | None
    )
    paired_reverse_patch_native_distance_reduction_vs_target_baseline: (
        np.ndarray | None
    )
    paired_source_native_log_loss_improvement_vs_recipient_native: (
        np.ndarray | None
    )
    paired_source_no_op_log_loss_improvement_vs_recipient_no_op: (
        np.ndarray | None
    )
    paired_donor_shift_balance: Mapping[str, Any] | None
    paired_ablation_displacement_balance: Mapping[str, Any] | None
    paired_donor_displacement_balance: Mapping[str, Any] | None


@dataclass(frozen=True)
class PairedReversePatchSource:
    """One independently loaded official source model for reverse patching."""

    driver: OfficialTabICLDriver
    condition: str


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
    target_condition: str | None = None,
    paired_source: PairedReversePatchSource | None = None,
    maximum_symmetric_donor_shift_rms_ratio: float | None = None,
) -> OfficialCausalEvaluation:
    """Run no-op, target, matched-random, round-trip, and paired edits.

    Every condition begins from the same Temporary Identity RNG state.  On
    success, the generator is left at the state produced by one native
    prediction; on failure, its entry state is restored.  A paired reverse
    patch, when requested, comes from a second independently loaded official
    driver and is never synthesized from the target model's own forward pass.
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
    resolved_paired_source = _validated_paired_source(
        paired_source,
        target_driver=driver,
        target_condition=target_condition,
        qualification=qualification,
    )
    if resolved_paired_source is not None and site != "row_interactor":
        raise ValueError(
            "formal paired reverse patch is restricted to row_interactor "
            "because deeper token coordinates are not cross-condition stable"
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
    if resolved_paired_source is not None:
        if frozen_controls is None:
            raise ValueError(
                "paired reverse patch requires explicit frozen control_features"
            )
        if (
            maximum_symmetric_donor_shift_rms_ratio is None
            or isinstance(maximum_symmetric_donor_shift_rms_ratio, bool)
            or not isinstance(
                maximum_symmetric_donor_shift_rms_ratio, (int, float)
            )
            or not np.isfinite(maximum_symmetric_donor_shift_rms_ratio)
            or maximum_symmetric_donor_shift_rms_ratio < 1.0
            or maximum_symmetric_donor_shift_rms_ratio
            > _MAX_FORMAL_SYMMETRIC_DONOR_SHIFT_RMS_RATIO
        ):
            raise ValueError(
                "paired reverse patch requires a finite maximum symmetric "
                "donor-shift RMS ratio between one and the protocol maximum "
                f"{_MAX_FORMAL_SYMMETRIC_DONOR_SHIFT_RMS_RATIO}"
            )
        maximum_symmetric_donor_shift_rms_ratio = float(
            maximum_symmetric_donor_shift_rms_ratio
        )
    elif maximum_symmetric_donor_shift_rms_ratio is not None:
        raise ValueError(
            "maximum donor-shift RMS ratio requires a paired source"
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
    paired_improvement: np.ndarray | None = None
    paired_improvement_vs_no_op: np.ndarray | None = None
    paired_improvement_vs_target: np.ndarray | None = None
    paired_native_distance_reduction: np.ndarray | None = None
    paired_source_native_gap: np.ndarray | None = None
    paired_source_no_op_gap: np.ndarray | None = None
    paired_shift_balance: Mapping[str, Any] | None = None
    paired_ablation_displacement_balance: Mapping[str, Any] | None = None
    paired_donor_displacement_balance: Mapping[str, Any] | None = None
    paired_alignment_verified = False
    source_result: OfficialInferenceResult | None = None
    source_native: PredictionMetrics | None = None
    source_no_op: PredictionMetrics | None = None
    source_reconstruction_mse: float | None = None
    source_no_op_accuracy_difference: float | None = None
    source_no_op_probability_difference: float | None = None
    with _preserve_training_states(autoencoder, normalizer):
        autoencoder.eval()
        normalizer.eval()
        with torch.inference_mode(), ExitStack() as sessions:
            session = sessions.enter_context(driver.paired_session())
            source_session = (
                None
                if resolved_paired_source is None
                else sessions.enter_context(
                    resolved_paired_source.driver.paired_session()
                )
            )
            entry_state = session.snapshot_identity_rng()
            source_entry_state = (
                None
                if source_session is None
                else source_session.snapshot_identity_rng()
            )
            final_state = None
            source_final_state = None
            try:
                if source_session is not None:
                    assert resolved_paired_source is not None
                    source_session.restore_identity_rng(source_entry_state)
                    source_result = source_session.predict_proba(
                        X,
                        y=y,
                        sites=(site,),
                        require_exact_baseline=True,
                    )
                    source_final_state = source_session.snapshot_identity_rng()
                    _validate_native_capture(
                        source_result,
                        driver=resolved_paired_source.driver,
                        site=site,
                    )
                session.restore_identity_rng(entry_state)
                native_result = session.predict_proba(
                    X,
                    y=y,
                    sites=(site,),
                    require_exact_baseline=True,
                )
                final_state = session.snapshot_identity_rng()
                _validate_native_capture(native_result, driver=driver, site=site)
                if source_result is not None:
                    _validate_paired_capture_alignment(
                        target=native_result,
                        source=source_result,
                        site=site,
                    )
                    paired_alignment_verified = True
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
                source_calls = (
                    ()
                    if source_result is None
                    else tuple(
                        _encode_call(
                            call,
                            site=site,
                            autoencoder=autoencoder,
                            normalizer=normalizer,
                        )
                        for call in source_result.forward_calls
                    )
                )
                if source_result is not None:
                    source_native = _prediction_metrics(
                        source_result.probabilities,
                        targets,
                        output_kind="probabilities",
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
                if source_result is not None:
                    assert source_session is not None
                    source_reconstruction_mse = _reconstruction_mse(source_calls)
                    if source_reconstruction_mse > max_no_op_reconstruction_mse:
                        raise RuntimeError(
                            "paired source no-op reconstruction MSE "
                            f"{source_reconstruction_mse:.6g} exceeds limit "
                            f"{max_no_op_reconstruction_mse:.6g}"
                        )
                    _, source_no_op = _run_condition(
                        source_session,
                        X,
                        y,
                        site=site,
                        entry_state=source_entry_state,
                        expected_final_state=source_final_state,
                        native_result=source_result,
                        replacements=tuple(
                            call.reconstruction for call in source_calls
                        ),
                        targets=targets,
                    )
                    assert source_native is not None
                    source_no_op_accuracy_difference = abs(
                        source_native.accuracy - source_no_op.accuracy
                    )
                    source_no_op_probability_difference = float(
                        np.max(
                            np.abs(
                                source_no_op.probabilities.astype(np.float64)
                                - source_native.probabilities.astype(np.float64)
                            )
                        )
                    )
                    if (
                        source_no_op_accuracy_difference
                        > max_no_op_accuracy_drop
                    ):
                        raise RuntimeError(
                            "paired source no-op reconstruction absolute accuracy "
                            f"difference {source_no_op_accuracy_difference:.6g} "
                            f"exceeds limit {max_no_op_accuracy_drop:.6g}"
                        )
                    if (
                        source_no_op_probability_difference
                        > max_no_op_probability_deviation
                    ):
                        raise RuntimeError(
                            "paired source no-op reconstruction maximum absolute "
                            "probability difference "
                            f"{source_no_op_probability_difference:.6g} exceeds "
                            f"limit {max_no_op_probability_deviation:.6g}"
                        )
                    paired_source_native_gap = (
                        native.true_class_log_loss
                        - source_native.true_class_log_loss
                    )
                    paired_source_no_op_gap = (
                        no_op.true_class_log_loss
                        - source_no_op.true_class_log_loss
                    )
                    if not np.isfinite(paired_source_native_gap).all() or not (
                        np.isfinite(paired_source_no_op_gap).all()
                    ):
                        raise RuntimeError(
                            "paired source advantage contains non-finite values"
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

                if source_result is not None:
                    assert maximum_symmetric_donor_shift_rms_ratio is not None
                    paired_ablation_displacement_balance = (
                        _decoded_displacement_balance(
                            no_op_replacements,
                            target_replacements,
                            random_replacements,
                            maximum_symmetric_ratio=(
                                maximum_symmetric_donor_shift_rms_ratio
                            ),
                            label="ablation",
                        )
                    )
                    paired_shift_balance = _donor_shift_balance(
                        encoded_calls,
                        source_calls,
                        target_features=features,
                        control_features=controls,
                        maximum_symmetric_ratio=(
                            maximum_symmetric_donor_shift_rms_ratio
                        ),
                    )
                    paired_latents = _patched_latents_from_source(
                        encoded_calls,
                        source_calls,
                        features=features,
                    )
                    paired_replacements = _decode_calls(
                        encoded_calls,
                        paired_latents,
                        autoencoder=autoencoder,
                        normalizer=normalizer,
                    )
                    _, paired_prediction = _run_condition(
                        session,
                        X,
                        y,
                        site=site,
                        entry_state=entry_state,
                        expected_final_state=final_state,
                        native_result=native_result,
                        replacements=paired_replacements,
                        targets=targets,
                    )
                    conditions["paired_reverse_patch"] = _condition_result(
                        "paired_reverse_patch",
                        paired_prediction,
                        reference_scope="primary",
                        model_baseline=native,
                        reconstruction_baseline=no_op,
                        target_features=features,
                    )

                    paired_control_latents = _patched_latents_from_source(
                        encoded_calls,
                        source_calls,
                        features=controls,
                    )
                    paired_control_replacements = _decode_calls(
                        encoded_calls,
                        paired_control_latents,
                        autoencoder=autoencoder,
                        normalizer=normalizer,
                    )
                    paired_donor_displacement_balance = (
                        _decoded_displacement_balance(
                            no_op_replacements,
                            paired_replacements,
                            paired_control_replacements,
                            maximum_symmetric_ratio=(
                                maximum_symmetric_donor_shift_rms_ratio
                            ),
                            label="donor",
                        )
                    )
                    _, paired_control_prediction = _run_condition(
                        session,
                        X,
                        y,
                        site=site,
                        entry_state=entry_state,
                        expected_final_state=final_state,
                        native_result=native_result,
                        replacements=paired_control_replacements,
                        targets=targets,
                    )
                    conditions["paired_matched_random_patch"] = _condition_result(
                        "paired_matched_random_patch",
                        paired_control_prediction,
                        reference_scope="primary",
                        model_baseline=native,
                        reconstruction_baseline=no_op,
                        target_features=features,
                        control_features=controls,
                    )
                    paired_improvement = (
                        paired_control_prediction.true_class_log_loss
                        - paired_prediction.true_class_log_loss
                    )
                    paired_improvement_vs_no_op = (
                        no_op.true_class_log_loss
                        - paired_prediction.true_class_log_loss
                    )
                    paired_improvement_vs_target = (
                        target_prediction.true_class_log_loss
                        - paired_prediction.true_class_log_loss
                    )
                    paired_native_distance_reduction = np.abs(
                        target_prediction.true_class_log_loss
                        - native.true_class_log_loss
                    ) - np.abs(
                        paired_prediction.true_class_log_loss
                        - native.true_class_log_loss
                    )
                    paired_effects = (
                        paired_improvement,
                        paired_improvement_vs_no_op,
                        paired_improvement_vs_target,
                        paired_native_distance_reduction,
                    )
                    if any(not np.isfinite(effect).all() for effect in paired_effects):
                        raise RuntimeError(
                            "paired reverse-patch effect contains non-finite values"
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
                if source_session is not None:
                    source_session.restore_identity_rng(source_final_state)
            except BaseException:
                session.restore_identity_rng(entry_state)
                if source_session is not None:
                    source_session.restore_identity_rng(source_entry_state)
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
        mechanistic_rescue_status=(
            "paired_rescue_not_run"
            if resolved_paired_source is None
            else "paired_reverse_patch_completed_requires_paired_statistics"
        ),
        paired_source_condition=(
            None
            if resolved_paired_source is None
            else resolved_paired_source.condition
        ),
        paired_source_model_sha=(
            None
            if resolved_paired_source is None
            else resolved_paired_source.driver.model_sha
        ),
        paired_source_checkpoint_sha=(
            None
            if resolved_paired_source is None
            else resolved_paired_source.driver.checkpoint_sha
        ),
        paired_source_native_prediction=source_native,
        paired_source_no_op_prediction=source_no_op,
        paired_source_no_op_reconstruction_mse=source_reconstruction_mse,
        paired_source_no_op_accuracy_absolute_difference=(
            source_no_op_accuracy_difference
        ),
        paired_source_no_op_probability_max_abs_difference=(
            source_no_op_probability_difference
        ),
        paired_alignment_verified=paired_alignment_verified,
        paired_reverse_patch_log_loss_improvement_vs_matched_random=(
            paired_improvement
        ),
        paired_reverse_patch_log_loss_improvement_vs_no_op=(
            paired_improvement_vs_no_op
        ),
        paired_reverse_patch_log_loss_improvement_vs_target_baseline=(
            paired_improvement_vs_target
        ),
        paired_reverse_patch_native_distance_reduction_vs_target_baseline=(
            paired_native_distance_reduction
        ),
        paired_source_native_log_loss_improvement_vs_recipient_native=(
            paired_source_native_gap
        ),
        paired_source_no_op_log_loss_improvement_vs_recipient_no_op=(
            paired_source_no_op_gap
        ),
        paired_donor_shift_balance=paired_shift_balance,
        paired_ablation_displacement_balance=(
            paired_ablation_displacement_balance
        ),
        paired_donor_displacement_balance=paired_donor_displacement_balance,
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


def _validated_paired_source(
    source: PairedReversePatchSource | None,
    *,
    target_driver: OfficialTabICLDriver,
    target_condition: str | None,
    qualification: Mapping[str, Any],
) -> PairedReversePatchSource | None:
    if source is None:
        if target_condition is not None and target_condition not in (
            _CANONICAL_CONDITIONS
        ):
            raise ValueError("target_condition is not canonical")
        return None
    if not isinstance(source, PairedReversePatchSource):
        raise TypeError("paired_source must be a PairedReversePatchSource")
    if not isinstance(source.driver, OfficialTabICLDriver):
        raise TypeError("paired source driver must be an OfficialTabICLDriver")
    if target_condition not in _CANONICAL_CONDITIONS:
        raise ValueError("paired reverse patch requires a canonical target_condition")
    if source.condition not in _CANONICAL_CONDITIONS:
        raise ValueError("paired source condition is not canonical")
    if source.condition == target_condition:
        raise ValueError("paired source condition must differ from target condition")
    if source.driver is target_driver or source.driver.estimator is target_driver.estimator:
        raise ValueError("paired reverse patch requires an independent official driver")
    if source.driver.model_sha != target_driver.model_sha:
        raise ValueError(
            "paired source and target must use the same model-code commit"
        )
    if source.driver.checkpoint_sha == target_driver.checkpoint_sha:
        raise ValueError("paired source and target checkpoints must differ")
    if source.driver.fit_context != target_driver.fit_context:
        raise ValueError("paired source and target fit contexts must match")
    if (
        target_driver.source_evidence_level == "strict"
        and source.driver.source_evidence_level != "strict"
    ):
        raise RuntimeError("strict target requires strict paired-source evidence")
    checkpoint_lineage = qualification.get("condition_checkpoints_sha256")
    if not isinstance(checkpoint_lineage, Mapping):
        raise RuntimeError(
            "paired reverse patch requires shared representation checkpoint lineage"
        )
    expected = {
        target_condition: target_driver.checkpoint_sha,
        source.condition: source.driver.checkpoint_sha,
    }
    mismatches = {
        condition: (checkpoint_lineage.get(condition), checkpoint)
        for condition, checkpoint in expected.items()
        if checkpoint_lineage.get(condition) != checkpoint
    }
    if mismatches:
        raise RuntimeError(
            "paired source/target checkpoints are not bound by the shared "
            f"representation: {mismatches}"
        )
    return source


def _validate_paired_capture_alignment(
    *,
    target: OfficialInferenceResult,
    source: OfficialInferenceResult,
    site: str,
) -> None:
    """Require exact schedule and activation coordinates across two drivers."""

    if not np.array_equal(target.classes, source.classes):
        raise RuntimeError("paired source and target class rosters differ")
    if len(target.forward_calls) != len(source.forward_calls):
        raise RuntimeError("paired source and target raw-call schedules differ")
    for target_call, source_call in zip(
        target.forward_calls, source.forward_calls, strict=True
    ):
        if target_call.metadata != source_call.metadata:
            raise RuntimeError(
                "paired source and target ensemble schedules/coordinates differ"
            )
        target_record = target_call.activations[site]
        source_record = source_call.activations[site]
        coordinate_metadata = (
            "site",
            "axis_names",
            "shape",
            "preprocessing_view_id",
            "feature_group_map",
        )
        mismatches = {
            name: (getattr(target_record, name), getattr(source_record, name))
            for name in coordinate_metadata
            if getattr(target_record, name) != getattr(source_record, name)
        }
        if mismatches:
            raise RuntimeError(
                "paired source and target activation axes/coordinates differ: "
                f"{mismatches}"
            )


def _patched_latents_from_source(
    target_calls: tuple[_EncodedCall, ...],
    source_calls: tuple[_EncodedCall, ...],
    *,
    features: Sequence[int],
) -> tuple[Tensor, ...]:
    patched_calls: list[Tensor] = []
    for target_call, source_call in zip(target_calls, source_calls, strict=True):
        if target_call.latents.shape != source_call.latents.shape:
            raise RuntimeError("paired source and target latent coordinates differ")
        patched = target_call.latents.clone()
        patched[:, list(features)] = source_call.latents[:, list(features)]
        patched_calls.append(patched)
    return tuple(patched_calls)


def _donor_shift_balance(
    recipient_calls: tuple[_EncodedCall, ...],
    source_calls: tuple[_EncodedCall, ...],
    *,
    target_features: Sequence[int],
    control_features: Sequence[int],
    maximum_symmetric_ratio: float,
) -> dict[str, Any]:
    target_squares: list[Tensor] = []
    control_squares: list[Tensor] = []
    by_call: list[dict[str, Any]] = []
    for recipient, source in zip(recipient_calls, source_calls, strict=True):
        if recipient.latents.shape != source.latents.shape:
            raise RuntimeError("paired source and target latent coordinates differ")
        target_delta = source.latents[:, list(target_features)] - (
            recipient.latents[:, list(target_features)]
        )
        control_delta = source.latents[:, list(control_features)] - (
            recipient.latents[:, list(control_features)]
        )
        target_square = target_delta.detach().to(torch.float64).square().sum(dim=1)
        control_square = control_delta.detach().to(torch.float64).square().sum(dim=1)
        target_squares.append(target_square.to(device="cpu"))
        control_squares.append(control_square.to(device="cpu"))
        target_rms = float(torch.sqrt(target_square.mean()).item())
        control_rms = float(torch.sqrt(control_square.mean()).item())
        call_ratio = _symmetric_non_negative_ratio(target_rms, control_rms)
        if not np.isfinite(call_ratio) or call_ratio > maximum_symmetric_ratio:
            raise RuntimeError(
                "paired donor target/control latent-shift RMS ratio for call "
                f"{recipient.capture.metadata.call_index} is {call_ratio:.6g}, "
                f"exceeding frozen limit {maximum_symmetric_ratio:.6g}"
            )
        by_call.append(
            {
                "call_index": recipient.capture.metadata.call_index,
                "vector_count": int(target_square.numel()),
                "target_feature_shift_rms": target_rms,
                "control_feature_shift_rms": control_rms,
                "symmetric_rms_ratio": call_ratio,
            }
        )
    all_target = torch.cat(target_squares)
    all_control = torch.cat(control_squares)
    target_rms = float(torch.sqrt(all_target.mean()).item())
    control_rms = float(torch.sqrt(all_control.mean()).item())
    ratio = _symmetric_non_negative_ratio(target_rms, control_rms)
    if not np.isfinite(ratio) or ratio > maximum_symmetric_ratio:
        raise RuntimeError(
            "paired donor target/control latent-shift RMS ratio "
            f"{ratio:.6g} exceeds frozen limit {maximum_symmetric_ratio:.6g}"
        )
    return {
        "target_feature_shift_rms": target_rms,
        "control_feature_shift_rms": control_rms,
        "symmetric_rms_ratio": ratio,
        "maximum_symmetric_rms_ratio": maximum_symmetric_ratio,
        "passed": True,
        "by_call": by_call,
    }


def _decoded_displacement_balance(
    baseline_replacements: Sequence[Tensor],
    target_replacements: Sequence[Tensor],
    control_replacements: Sequence[Tensor],
    *,
    maximum_symmetric_ratio: float,
    label: str,
) -> dict[str, Any]:
    """Gate the actual decoded activation dose of target/control edits."""

    if not (
        len(baseline_replacements)
        == len(target_replacements)
        == len(control_replacements)
    ) or not baseline_replacements:
        raise RuntimeError(f"{label} displacement schedules do not align")
    target_squares: list[Tensor] = []
    control_squares: list[Tensor] = []
    by_call: list[dict[str, Any]] = []
    for call_index, (baseline, target, control) in enumerate(
        zip(
            baseline_replacements,
            target_replacements,
            control_replacements,
            strict=True,
        )
    ):
        if baseline.shape != target.shape or baseline.shape != control.shape:
            raise RuntimeError(f"{label} decoded activation shapes differ")
        target_square = (
            (target - baseline).detach().to(dtype=torch.float64, device="cpu").square()
        )
        control_square = (
            (control - baseline).detach().to(dtype=torch.float64, device="cpu").square()
        )
        target_squares.append(target_square.reshape(-1))
        control_squares.append(control_square.reshape(-1))
        target_rms = float(torch.sqrt(target_square.mean()).item())
        control_rms = float(torch.sqrt(control_square.mean()).item())
        ratio = _symmetric_non_negative_ratio(target_rms, control_rms)
        if not np.isfinite(ratio) or ratio > maximum_symmetric_ratio:
            raise RuntimeError(
                f"{label} decoded target/control displacement ratio for call "
                f"{call_index} is {ratio:.6g}, exceeding frozen limit "
                f"{maximum_symmetric_ratio:.6g}"
            )
        by_call.append(
            {
                "call_index": call_index,
                "vector_count": int(target_square.numel()),
                "target_displacement_rms": target_rms,
                "control_displacement_rms": control_rms,
                "symmetric_rms_ratio": ratio,
            }
        )
    all_target = torch.cat(target_squares)
    all_control = torch.cat(control_squares)
    target_rms = float(torch.sqrt(all_target.mean()).item())
    control_rms = float(torch.sqrt(all_control.mean()).item())
    ratio = _symmetric_non_negative_ratio(target_rms, control_rms)
    if not np.isfinite(ratio) or ratio > maximum_symmetric_ratio:
        raise RuntimeError(
            f"{label} decoded target/control displacement ratio {ratio:.6g} "
            f"exceeds frozen limit {maximum_symmetric_ratio:.6g}"
        )
    return {
        "target_displacement_rms": target_rms,
        "control_displacement_rms": control_rms,
        "symmetric_rms_ratio": ratio,
        "maximum_symmetric_rms_ratio": maximum_symmetric_ratio,
        "passed": True,
        "by_call": by_call,
    }


def _symmetric_non_negative_ratio(first: float, second: float) -> float:
    if first == 0.0 and second == 0.0:
        return 1.0
    if first <= 0.0 or second <= 0.0:
        return float("inf")
    return max(first / second, second / first)


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
    "checkpoint_study",
    "representation_run_dir",
    "expected_parent_manifest_sha256",
    "ranking_parent",
    "dataset",
    "intervention",
    "official_classifier",
    "paired_reverse_patch",
}
_RANKING_PARENT_FIELDS = {
    "run_dir",
    "expected_manifest_sha256",
    "expected_selection_sha256",
    "split_protocol_path",
    "expected_split_protocol_sha256",
    "directions",
}
_RANKING_SELECTION_FIELDS = {
    "schema_version",
    "analysis",
    "evidence_scope",
    "formal_claim",
    "site",
    "conditions",
    "directions",
    "maximum_symmetric_donor_shift_rms_ratio",
    "score_definition",
    "decoder_norm_space",
    "raw_activation_rms_definition",
    "representation_parent_manifest_sha256",
    "representation_model_sha256",
    "representation_source_lineage_sha256",
    "ranking_source_lineage_sha256",
    "split_protocol_id",
    "split_protocol_sha256",
    "ranking_dataset_roster_sha256",
    "activation_sha256_by_condition",
    "activation_binding_sha256",
    "inference_contract_sha256",
    "condition_checkpoints_sha256",
    "ranking_protocol",
    "ranking_protocol_sha256",
    "dataset_score_sha256",
    "raw_activation_rms_by_dataset",
    "median_scores",
    "median_scores_sha256",
    "nonzero_dataset_counts",
    "nonzero_dataset_counts_sha256",
    "activation_frequencies",
    "activation_frequencies_sha256",
    "decoder_norms",
    "decoder_norms_sha256",
    "top_features",
    "target_features",
    "control_features",
    "latent_baseline",
    "parent_seed",
}
_RANKING_PROTOCOL_FIELDS = {
    "dataset_ids",
    "minimum_nonzero_datasets",
    "target_count",
    "random_candidate_pool_size",
    "random_seed",
    "activation_frequency_threshold",
    "decoder_norm_space",
    "latent_baseline",
    "same_features_both_directions",
}
_RANKING_DIRECTIONS = ("rope_to_none", "none_to_rope")
_RANKING_CONDITIONS = ("rope", "none")
_RANKING_SCORE_ID = (
    "median_dataset_rms_latent_rope_minus_none_times_decoder_norm_"
    "divided_by_pooled_raw_activation_rms"
)
_RANKING_RAW_RMS_DEFINITION = (
    "root_mean_square_of_pooled_aligned_none_and_rope_raw_values"
)
_CHECKPOINT_STUDY_TRUST_FIELDS = {
    "checkpoint_path",
    "expected_checkpoint_sha256",
    "finalized_manifest_path",
    "expected_finalized_manifest_file_sha256",
    "expected_finalized_manifest_sha256",
    "transaction_ledger_path",
    "expected_transaction_ledger_file_sha256",
    "transaction_ledger_sha256",
    "artifact_root",
    "study_id",
    "arm",
    "stage",
    "upstream_identity",
    "artifact_identity",
}
_PREPROCESSING_PROTOCOL = OFFICIAL_INFERENCE_PROTOCOL
_MAX_FORMAL_CLASSES = 10
_MAX_FORMAL_NO_OP_RECONSTRUCTION_MSE = 0.01
_MAX_FORMAL_NO_OP_PROBABILITY_DEVIATION = 0.02
_MAX_FORMAL_NO_OP_ACCURACY_DIFFERENCE = 0.005
_MAX_FORMAL_SYMMETRIC_DONOR_SHIFT_RMS_RATIO = 1.25
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
_SELECTION_EVIDENCE_FAMILY = [
    {
        "name": "target_damage",
        "metric": "mean_delta_log_loss_vs_no_op",
        "direction": "positive",
        "scope": "candidate",
    },
    {
        "name": "ablation_specificity",
        "metric": "target_minus_matched_control_mean_delta_log_loss_vs_no_op",
        "direction": "positive",
        "scope": "candidate",
    },
    {
        "name": "donor_rescue",
        "metric": "mean_log_loss_improvement_vs_recipient_no_op",
        "direction": "positive",
        "scope": "candidate",
    },
    {
        "name": "donor_specificity",
        "metric": "mean_log_loss_improvement_vs_paired_matched_random_patch",
        "direction": "positive",
        "scope": "candidate",
    },
    {
        "name": "source_advantage",
        "metric": "intersection_source_native_and_no_op_advantage",
        "direction": "both_positive",
        "scope": "global",
    },
]


def run(args: Any) -> int:
    """Run and atomically publish one strict official model-causal evaluation."""

    configuration = load_verified_json_config(Path(args.config))
    config = _exact_object(
        configuration.data,
        label="model-causal config",
        required=_MODEL_CAUSAL_TOP_LEVEL
        - {
            "expected_parent_manifest_sha256",
            "ranking_parent",
            "paired_reverse_patch",
        },
        optional={
            "expected_parent_manifest_sha256",
            "ranking_parent",
            "paired_reverse_patch",
        },
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
    raw_ranking_config = config.get("ranking_parent")
    ranking_config = (
        None
        if raw_ranking_config is None
        else _exact_object(
            raw_ranking_config,
            label="ranking_parent",
            required=_RANKING_PARENT_FIELDS,
        )
    )
    raw_paired_config = config.get("paired_reverse_patch")
    paired_config = (
        None
        if raw_paired_config is None
        else _exact_object(
            raw_paired_config,
            label="paired_reverse_patch",
            required={
                "source_condition",
                "source_checkpoint_path",
                "expected_source_checkpoint_sha256",
                "source_model_code_root",
                "expected_source_model_code_sha",
                "source_code_attestation_path",
                "expected_source_code_attestation_sha256",
                "maximum_symmetric_donor_shift_rms_ratio",
            },
        )
    )
    checkpoint_study = _checkpoint_study_configuration(
        config["checkpoint_study"], paired_source_requested=paired_config is not None
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
    if checkpoint_study["scope"] == "exploratory_pilot":
        if roster_split != "discovery":
            raise ValueError(
                "exploratory_pilot checkpoints may only use the discovery roster; "
                "validation and held_out evidence require formal checkpoint trust"
            )
        evidence_scope = "exploratory-pilot"
    if ranking_config is not None and (
        checkpoint_study["scope"] != "exploratory_pilot"
        or roster_split != "discovery"
    ):
        raise ValueError(
            "ranking_parent is restricted to exploratory_pilot discovery runs"
        )
    if (
        checkpoint_study["scope"] == "exploratory_pilot"
        and paired_config is not None
        and ranking_config is None
    ):
        raise ValueError(
            "exploratory paired reverse patch requires a strict ranking_parent"
        )
    expected_sample_roster_sha256 = (
        _required_sha256(dataset_config, "expected_sample_roster_sha256")
        if ranking_config is not None
        else _optional_sha256(dataset_config, "expected_sample_roster_sha256")
    )
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
    selection_payload_file = None
    selection_freeze_file = None
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
        artifacts_by_name = {
            artifact.name: artifact for artifact in selection_parent.artifacts
        }
        if set(artifacts_by_name) != {
            "freeze.json",
            "selection.json",
            "summary.json",
        }:
            raise ValueError(
                "select-features parent must declare exactly freeze.json, "
                "selection.json, and summary.json"
            )
        verified_selection_artifacts = {}
        for artifact_name in sorted(artifacts_by_name):
            artifact = artifacts_by_name[artifact_name]
            verified = verify_file(
                selection_dir / artifact_name,
                expected_sha256=artifact.sha256,
            )
            if verified.digest.size_bytes != artifact.size_bytes:
                raise ValueError(
                    f"selection {artifact_name} size differs from its manifest"
                )
            verified_selection_artifacts[artifact_name] = verified
        selection_freeze_file = verified_selection_artifacts["freeze.json"]
        selection_payload_file = verified_selection_artifacts[
            "selection.json"
        ]
        selection_summary_file = verified_selection_artifacts["summary.json"]
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

    paired_checkpoint_file = None
    paired_code_attestation_file = None
    paired_source_condition = None
    paired_source_code_root = None
    paired_source_model_sha = None
    paired_shift_ratio_limit = None
    if paired_config is not None:
        paired_source_condition = _required_string(
            paired_config, "source_condition"
        )
        if paired_source_condition not in _CANONICAL_CONDITIONS:
            raise ValueError("paired source condition is not canonical")
        paired_checkpoint_file = verify_file(
            _absolute_file(
                paired_config["source_checkpoint_path"],
                name="paired source_checkpoint_path",
            ),
            expected_sha256=_required_sha256(
                paired_config, "expected_source_checkpoint_sha256"
            ),
        )
        paired_code_attestation_file = verify_file(
            _absolute_file(
                paired_config["source_code_attestation_path"],
                name="paired source_code_attestation_path",
            ),
            expected_sha256=_required_sha256(
                paired_config, "expected_source_code_attestation_sha256"
            ),
        )
        paired_source_code_root = _absolute_directory(
            paired_config["source_model_code_root"],
            name="paired source_model_code_root",
        )
        paired_source_model_sha = _required_git_sha(
            paired_config, "expected_source_model_code_sha"
        )
        paired_shift_ratio_limit = _finite_number(
            paired_config["maximum_symmetric_donor_shift_rms_ratio"],
            name="maximum_symmetric_donor_shift_rms_ratio",
        )
        if paired_shift_ratio_limit < 1.0:
            raise ValueError(
                "maximum_symmetric_donor_shift_rms_ratio must be at least one"
            )
        if paired_shift_ratio_limit > _MAX_FORMAL_SYMMETRIC_DONOR_SHIFT_RMS_RATIO:
            raise ValueError(
                "maximum_symmetric_donor_shift_rms_ratio exceeds protocol maximum "
                f"{_MAX_FORMAL_SYMMETRIC_DONOR_SHIFT_RMS_RATIO}"
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

    ranking_parent: RunManifest | None = None
    ranking_manifest_file = None
    ranking_selection_file = None
    ranking_split_protocol_file = None
    configured_ranking_directions: tuple[str, ...] | None = None
    if ranking_config is not None:
        configured_ranking_directions = _ranking_directions(
            ranking_config["directions"], name="ranking_parent.directions"
        )
        ranking_dir = _absolute_directory(
            ranking_config["run_dir"], name="ranking_parent.run_dir"
        )
        ranking_parent = verify_run_directory(ranking_dir)
        ranking_manifest_file = verify_file(
            ranking_dir / "manifest.json",
            expected_sha256=_required_sha256(
                ranking_config, "expected_manifest_sha256"
            ),
        )
        if _manifest_from_verified_bytes(ranking_manifest_file) != ranking_parent:
            raise RuntimeError(
                "ranking parent manifest changed during directory verification"
            )
        _validate_ranking_parent_manifest(ranking_parent)
        ranking_artifact = ranking_parent.artifacts[0]
        ranking_selection_file = verify_file(
            ranking_dir / "selection.json",
            expected_sha256=_required_sha256(
                ranking_config, "expected_selection_sha256"
            ),
        )
        if (
            ranking_selection_file.digest.sha256 != ranking_artifact.sha256
            or ranking_selection_file.digest.size_bytes
            != ranking_artifact.size_bytes
        ):
            raise ValueError(
                "ranking selection artifact differs from its completed manifest"
            )
        ranking_split_protocol_file = verify_file(
            _absolute_file(
                ranking_config["split_protocol_path"],
                name="ranking_parent.split_protocol_path",
            ),
            expected_sha256=_required_sha256(
                ranking_config, "expected_split_protocol_sha256"
            ),
        )

    sample_roster_file = verify_file(
        _absolute_file(
            dataset_config["sample_roster_path"], name="sample_roster_path"
        ),
        expected_sha256=expected_sample_roster_sha256,
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
    if selection_freeze_file is not None:
        if freeze_file is None:
            raise ValueError("select-features parent requires its freeze artifact")
        if (
            freeze_file.digest != selection_freeze_file.digest
            or (freeze_file.device, freeze_file.inode)
            != (selection_freeze_file.device, selection_freeze_file.inode)
        ):
            raise ValueError(
                "configured freeze artifact is not the exact select-features "
                "freeze.json input"
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
    if ranking_manifest_file is not None:
        assert ranking_selection_file is not None
        assert ranking_split_protocol_file is not None
        additional_paths.update(
            {
                "ranking.parent_manifest": ranking_manifest_file.path,
                "ranking.selection": ranking_selection_file.path,
                "ranking.split_protocol": ranking_split_protocol_file.path,
            }
        )
        expected_additional.update(
            {
                "ranking.parent_manifest": (
                    ranking_manifest_file.digest.sha256
                ),
                "ranking.selection": ranking_selection_file.digest.sha256,
                "ranking.split_protocol": (
                    ranking_split_protocol_file.digest.sha256
                ),
            }
        )
    if checkpoint_study["scope"] == "formal":
        common_ledger = checkpoint_study["recipient_chain"][0][
            "transaction_ledger_file"
        ]
        additional_paths["checkpoint_study.transaction_ledger"] = (
            common_ledger.path
        )
        expected_additional["checkpoint_study.transaction_ledger"] = (
            common_ledger.digest.sha256
        )
        for role_name in ("recipient", "source"):
            chain = checkpoint_study[f"{role_name}_chain"]
            if chain is None:
                continue
            for stage_index, trust in enumerate(chain):
                artifact_names = ["finalized_manifest"]
                if stage_index < len(chain) - 1:
                    artifact_names.append("checkpoint")
                for artifact_name in artifact_names:
                    verified = trust[f"{artifact_name}_file"]
                    role = (
                        f"checkpoint_study.{role_name}."
                        f"{stage_index}.{artifact_name}"
                    )
                    additional_paths[role] = verified.path
                    expected_additional[role] = verified.digest.sha256
    if paired_checkpoint_file is not None:
        assert paired_code_attestation_file is not None
        additional_paths["paired_source.checkpoint"] = (
            paired_checkpoint_file.path
        )
        additional_paths["paired_source.code_attestation"] = (
            paired_code_attestation_file.path
        )
        expected_additional["paired_source.checkpoint"] = (
            paired_checkpoint_file.digest.sha256
        )
        expected_additional["paired_source.code_attestation"] = (
            paired_code_attestation_file.digest.sha256
        )
    if freeze_file is not None:
        additional_paths["intervention.freeze"] = freeze_file.path
        expected_additional["intervention.freeze"] = (
            freeze_file.digest.sha256
        )
    if selection_manifest_file is not None:
        assert selection_summary_file is not None
        assert selection_payload_file is not None
        additional_paths["selection.parent_manifest"] = (
            selection_manifest_file.path
        )
        additional_paths["selection.summary"] = selection_summary_file.path
        additional_paths["selection.selection"] = selection_payload_file.path
        expected_additional["selection.parent_manifest"] = (
            selection_manifest_file.digest.sha256
        )
        expected_additional["selection.summary"] = (
            selection_summary_file.digest.sha256
        )
        expected_additional["selection.selection"] = (
            selection_payload_file.digest.sha256
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
    paired_source_git = None
    paired_source_binding = None
    if paired_config is not None:
        assert paired_source_condition is not None
        assert paired_checkpoint_file is not None
        assert paired_code_attestation_file is not None
        assert paired_source_code_root is not None
        assert paired_source_model_sha is not None
        if paired_source_condition == context.condition:
            raise ValueError(
                "paired source condition must differ from recipient condition"
            )
        bound_source_checkpoint = context.additional_file(
            "paired_source.checkpoint"
        )
        bound_source_attestation = context.additional_file(
            "paired_source.code_attestation"
        )
        if (
            bound_source_checkpoint.digest.sha256
            == context.inputs.checkpoint.digest.sha256
        ):
            raise ValueError(
                "paired source and recipient checkpoint hashes must differ"
            )
        if paired_source_model_sha != context.inputs.model_code.head_sha:
            raise ValueError(
                "paired source and recipient must use the same model-code SHA"
            )
        paired_source_git = verify_git_tree(
            paired_source_code_root,
            expected_sha=paired_source_model_sha,
        )
        if paired_source_git.evidence_level != "strict":
            raise RuntimeError("paired source model code evidence is not strict")
        source_inference_contract_sha256 = official_inference_contract_sha256(
            paired_source_git.head_sha, estimator_options
        )
        if source_inference_contract_sha256 != inference_contract_sha256:
            raise ValueError(
                "paired source and recipient inference contracts differ"
            )
        checkpoint_lineage = source_lineage["condition_checkpoints_sha256"]
        if checkpoint_lineage.get(paired_source_condition) != (
            bound_source_checkpoint.digest.sha256
        ):
            raise ValueError(
                "paired source checkpoint differs from shared representation lineage"
            )
        _validate_paired_source_code_attestation(
            bound_source_attestation,
            source_condition=paired_source_condition,
            model_code_sha=paired_source_git.head_sha,
            checkpoint_sha256=bound_source_checkpoint.digest.sha256,
            inference_contract_sha256=source_inference_contract_sha256,
        )
        paired_source_binding = {
            "source_condition": paired_source_condition,
            "recipient_condition": context.condition,
            "model_sha": paired_source_git.head_sha,
            "checkpoint_sha256": bound_source_checkpoint.digest.sha256,
            "code_attestation_sha256": (
                bound_source_attestation.digest.sha256
            ),
            "inference_contract_sha256": source_inference_contract_sha256,
            "sample_roster_sha256": context.additional_file(
                "samples.roster"
            ).digest.sha256,
            "representation_source_lineage_sha256": _canonical_sha256(
                source_lineage
            ),
            "maximum_symmetric_donor_shift_rms_ratio": (
                paired_shift_ratio_limit
            ),
        }
    ranking_parent_binding = None
    if ranking_parent is not None:
        assert ranking_selection_file is not None
        assert ranking_split_protocol_file is not None
        assert configured_ranking_directions is not None
        ranking_parent_binding = _validate_ranking_parent(
            ranking_parent,
            context.additional_file("ranking.selection"),
            context.additional_file("ranking.split_protocol"),
            context=context,
            ranking_parent_manifest_sha256=context.additional_file(
                "ranking.parent_manifest"
            ).digest.sha256,
            representation_parent=parent,
            representation_parent_manifest_sha256=context.additional_file(
                "representation.parent_manifest"
            ).digest.sha256,
            representation_model_sha256=bound_representation.digest.sha256,
            representation_model=autoencoder,
            representation_normalizer=normalizer,
            source_lineage=source_lineage,
            inference_contract_sha256=inference_contract_sha256,
            dataset_id=dataset_id,
            target_features=target_features,
            control_features=configured_controls,
            latent_baseline=latent_baseline,
            configured_directions=configured_ranking_directions,
            paired_source_binding=paired_source_binding,
            sample_roster_sha256=context.additional_file(
                "samples.roster"
            ).digest.sha256,
            sample_count=len(sample_roster["row_indices"]),
            evaluation_split=evaluation_split,
        )
    checkpoint_study_binding = _validated_checkpoint_study(
        checkpoint_study,
        context=context,
        paired_source_binding=paired_source_binding,
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
        assert selection_payload_file is not None
        assert selection_freeze_file is not None
        _validate_selection_parent(
            selection_parent,
            selection_summary_file,
            selection_payload_file,
            selection_freeze_file,
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
            sample_roster_sha256=context.additional_file(
                "samples.roster"
            ).digest.sha256,
            paired_source_binding=paired_source_binding,
            maximum_symmetric_donor_shift_rms_ratio=paired_shift_ratio_limit,
            checkpoint_study_sha256=checkpoint_study_binding["binding_sha256"],
        )
    bound_freeze = (
        None
        if freeze_file is None
        else context.additional_file("intervention.freeze")
    )
    if bound_freeze is not None and selection_parent is None:
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
            "condition_checkpoints_sha256": dict(
                source_lineage["condition_checkpoints_sha256"]
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

    paired_source = None
    if paired_source_binding is not None:
        assert paired_source_git is not None
        bound_source_checkpoint = context.additional_file(
            "paired_source.checkpoint"
        )
        source_driver = fit_official_talent_driver(
            raw_dataset,
            bound_source_checkpoint.path,
            context_split="train",
            device=device,
            model_sha=paired_source_git.head_sha,
            estimator_options=estimator_options,
            expected_source_root=paired_source_git.root,
        )
        _validate_official_driver(
            source_driver,
            context=context,
            expected_condition=paired_source_binding["source_condition"],
            expected_model_sha=paired_source_binding["model_sha"],
            expected_checkpoint_sha256=paired_source_binding[
                "checkpoint_sha256"
            ],
        )
        paired_source = PairedReversePatchSource(
            source_driver,
            paired_source_binding["source_condition"],
        )

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
        target_condition=context.condition,
        paired_source=paired_source,
        maximum_symmetric_donor_shift_rms_ratio=paired_shift_ratio_limit,
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
        paired_source_binding=paired_source_binding,
        ranking_parent_binding=ranking_parent_binding,
        checkpoint_study=checkpoint_study_binding,
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
    if paired_source_git is not None:
        paired_source_git.assert_unchanged()
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


def _required_sha256(values: Mapping[str, Any], key: str) -> str:
    value = _optional_sha256(values, key)
    if value is None:
        raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    return value


def _required_git_sha(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{key} must be a lowercase 40-character Git SHA")
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


def _checkpoint_study_configuration(
    value: Any, *, paired_source_requested: bool
) -> dict[str, Any]:
    study = _exact_object(
        value,
        label="checkpoint_study",
        required={"scope", "recipient_chain", "source_chain"},
    )
    scope = study["scope"]
    if scope == "exploratory_pilot":
        if study["recipient_chain"] is not None or study["source_chain"] is not None:
            raise ValueError("exploratory_pilot cannot carry formal checkpoint trust")
        return {
            "scope": scope,
            "recipient_chain": None,
            "source_chain": None,
        }
    if scope != "formal":
        raise ValueError("checkpoint_study.scope must be formal or exploratory_pilot")
    recipient_chain = _checkpoint_trust_chain(
        study["recipient_chain"], label="recipient checkpoint chain"
    )
    raw_source_chain = study["source_chain"]
    source_chain = (
        None
        if raw_source_chain is None
        else _checkpoint_trust_chain(
            raw_source_chain, label="source checkpoint chain"
        )
    )
    if paired_source_requested != (source_chain is not None):
        raise ValueError(
            "formal source checkpoint chain must align exactly with "
            "paired_reverse_patch"
        )
    if source_chain is not None:
        if len(source_chain) != len(recipient_chain):
            raise ValueError("formal source/recipient stage chains differ in length")
        if source_chain[0]["study_id"] != recipient_chain[0]["study_id"]:
            raise ValueError("formal source/recipient study ids differ")
        if source_chain[0]["arm"] == recipient_chain[0]["arm"]:
            raise ValueError("formal source/recipient arms must differ")
        recipient_ledger = recipient_chain[0]["transaction_ledger_file"]
        source_ledger = source_chain[0]["transaction_ledger_file"]
        if (
            recipient_chain[0]["transaction_ledger_sha256"]
            != source_chain[0]["transaction_ledger_sha256"]
            or recipient_chain[0]["artifact_root"]
            != source_chain[0]["artifact_root"]
            or recipient_ledger.digest != source_ledger.digest
            or (recipient_ledger.device, recipient_ledger.inode)
            != (source_ledger.device, source_ledger.inode)
        ):
            raise ValueError(
                "formal source/recipient must share one atomic cohort ledger"
            )
    return {
        "scope": scope,
        "recipient_chain": recipient_chain,
        "source_chain": source_chain,
    }


def _checkpoint_trust_chain(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value or len(value) > 3:
        raise ValueError(f"{label} must contain one to three formal stages")
    result: list[dict[str, Any]] = []
    for raw in value:
        trust = _exact_object(
            raw, label=f"{label} entry", required=_CHECKPOINT_STUDY_TRUST_FIELDS
        )
        checkpoint_file = verify_file(
            _absolute_file(trust["checkpoint_path"], name="checkpoint_path"),
            expected_sha256=_required_sha256(
                trust, "expected_checkpoint_sha256"
            ),
        )
        finalized_file = verify_file(
            _absolute_file(
                trust["finalized_manifest_path"],
                name="finalized_manifest_path",
            ),
            expected_sha256=_required_sha256(
                trust, "expected_finalized_manifest_file_sha256"
            ),
        )
        ledger_file = verify_file(
            _absolute_file(
                trust["transaction_ledger_path"],
                name="transaction_ledger_path",
            ),
            expected_sha256=_required_sha256(
                trust, "expected_transaction_ledger_file_sha256"
            ),
        )
        arm = _required_string(trust, "arm")
        stage = _required_string(trust, "stage")
        if arm not in _CANONICAL_CONDITIONS:
            raise ValueError("checkpoint trust arm is not canonical")
        if stage not in {"stage1", "stage2", "stage3"}:
            raise ValueError("checkpoint trust stage is not formal")
        result.append(
            {
                "checkpoint_file": checkpoint_file,
                "finalized_manifest_file": finalized_file,
                "expected_finalized_manifest_sha256": _required_sha256(
                    trust, "expected_finalized_manifest_sha256"
                ),
                "transaction_ledger_file": ledger_file,
                "transaction_ledger_sha256": _required_sha256(
                    trust, "transaction_ledger_sha256"
                ),
                "artifact_root": _absolute_directory(
                    trust["artifact_root"], name="checkpoint artifact_root"
                ),
                "study_id": require_portable_identifier(
                    _required_string(trust, "study_id"), name="study_id"
                ),
                "arm": arm,
                "stage": stage,
                "upstream_identity": require_portable_identifier(
                    _required_string(trust, "upstream_identity"),
                    name="upstream_identity",
                ),
                "artifact_identity": require_portable_identifier(
                    _required_string(trust, "artifact_identity"),
                    name="artifact_identity",
                ),
            }
        )
    expected_stages = ["stage1", "stage2", "stage3"][: len(result)]
    if [item["stage"] for item in result] != expected_stages:
        raise ValueError(f"{label} must be an unbroken stage1-to-current chain")
    for item in result:
        expected_upstream = (
            f"{item['study_id']}:{item['arm']}:{item['stage']}"
        )
        expected_artifact = (
            f"{item['study_id']}.{item['arm']}.{item['stage']}.final"
        )
        if (
            item["upstream_identity"] != expected_upstream
            or item["artifact_identity"] != expected_artifact
        ):
            raise ValueError(
                f"{label} identity does not match the formal overlay contract"
            )
    for field in (
        "study_id",
        "arm",
        "transaction_ledger_sha256",
        "artifact_root",
    ):
        if len({item[field] for item in result}) != 1:
            raise ValueError(f"{label} {field} must be constant")
    ledger_files = {
        (
            item["transaction_ledger_file"].digest.sha256,
            item["transaction_ledger_file"].device,
            item["transaction_ledger_file"].inode,
        )
        for item in result
    }
    if len(ledger_files) != 1:
        raise ValueError(f"{label} must use one immutable transaction ledger")
    return result


def _trusted_training_provenance_api(context: Any) -> tuple[Any, Any]:
    """Load the formal validator only from the bound clean training/model T."""

    try:
        module = importlib.import_module("tabicl.train._provenance")
    except ImportError as error:
        raise RuntimeError(
            "formal checkpoint validation requires the public TabICL training package"
        ) from error
    raw_module_file = getattr(module, "__file__", None)
    if not isinstance(raw_module_file, str) or not raw_module_file:
        raise RuntimeError(
            "formal training provenance module must have a concrete __file__"
        )
    if not os.path.isabs(raw_module_file):
        raise RuntimeError(
            "formal training provenance module path must be absolute"
        )
    lexical_module_file = os.path.abspath(raw_module_file)
    module_file = os.path.realpath(raw_module_file)
    if not os.path.isfile(module_file):
        raise RuntimeError(
            "formal training provenance module path is not a regular file"
        )

    training_code = context.inputs.training_code
    model_code = context.inputs.model_code
    for label, evidence in (
        ("training", training_code),
        ("model", model_code),
    ):
        if evidence.evidence_level != "strict" or evidence.legacy_reasons:
            raise RuntimeError(
                f"formal {label} checkout does not have strict clean evidence"
            )
        evidence.assert_unchanged()
    if training_code.head_sha != model_code.head_sha:
        raise RuntimeError(
            "formal training and model manifest SHAs do not identify one T"
        )

    bound_roots = {
        os.path.realpath(os.fspath(training_code.root)),
        os.path.realpath(os.fspath(model_code.root)),
    }
    canonical_module_files = {
        os.path.join(root, "src", "tabicl", "train", "_provenance.py")
        for root in bound_roots
    }
    if lexical_module_file not in canonical_module_files:
        raise RuntimeError(
            "formal training provenance module is not the canonical file in "
            "the bound T checkout"
        )
    try:
        module_is_contained = any(
            os.path.commonpath((root, module_file)) == root
            for root in bound_roots
        )
    except ValueError:
        module_is_contained = False
    if not module_is_contained:
        raise RuntimeError(
            "formal training provenance module symlink escapes the bound T checkout"
        )
    module_spec = getattr(module, "__spec__", None)
    module_origin = getattr(module_spec, "origin", None)
    if module_origin is not None and (
        not isinstance(module_origin, str)
        or os.path.realpath(module_origin) != module_file
    ):
        raise RuntimeError(
            "formal training provenance module origin differs from __file__"
        )
    module_git = verify_git_tree(
        Path(module_file).parent,
        expected_sha=training_code.head_sha,
    )
    module_root = os.path.realpath(os.fspath(module_git.root))
    if (
        module_git.evidence_level != "strict"
        or module_git.legacy_reasons
        or module_git.head_sha != model_code.head_sha
        or module_root not in bound_roots
    ):
        raise RuntimeError(
            "formal training provenance module Git evidence differs from "
            "manifest training/model T"
        )
    ParentTrust = getattr(module, "ParentTrust", None)
    validate_parent_trust = getattr(module, "validate_parent_trust", None)
    canonical_ledger_validator = getattr(
        module, "validate_canonical_transaction_ledger", None
    )
    if not callable(ParentTrust) or not callable(validate_parent_trust):
        raise RuntimeError(
            "formal training provenance module lacks the canonical trust API"
        )
    if not callable(canonical_ledger_validator):
        raise RuntimeError(
            "formal training provenance module lacks the canonical ledger API"
        )
    for label, symbol in (
        ("ParentTrust", ParentTrust),
        ("validate_parent_trust", inspect.unwrap(validate_parent_trust)),
        (
            "validate_canonical_transaction_ledger",
            inspect.unwrap(canonical_ledger_validator),
        ),
    ):
        try:
            symbol_file = os.path.realpath(inspect.getfile(symbol))
        except (TypeError, OSError) as error:
            raise RuntimeError(
                f"formal training provenance {label} has no source file"
            ) from error
        if symbol_file != module_file:
            raise RuntimeError(
                f"formal training provenance {label} is not defined by the "
                "canonical module file"
            )
    return ParentTrust, validate_parent_trust


def _validated_checkpoint_study(
    study: Mapping[str, Any],
    *,
    context: Any,
    paired_source_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if study["scope"] == "exploratory_pilot":
        payload = {
            "schema_version": 1,
            "scope": "exploratory_pilot",
            "formal_trust_verified": False,
            "model_evidence_scope": "exploratory-pilot",
        }
        return {**payload, "binding_sha256": _canonical_sha256(payload)}

    ParentTrust, validate_parent_trust = _trusted_training_provenance_api(
        context
    )

    recipient_chain = study["recipient_chain"]
    source_chain = study["source_chain"]
    assert isinstance(recipient_chain, list)
    expected_recipient_arm = context.condition
    if recipient_chain[-1]["arm"] != expected_recipient_arm:
        raise ValueError("recipient formal arm differs from model-causal condition")
    recipient_runtime = context.inputs.checkpoint
    recipient_bound = recipient_chain[-1]["checkpoint_file"]
    _require_same_verified_file(
        recipient_runtime,
        recipient_bound,
        label="recipient formal checkpoint",
    )
    if paired_source_binding is None:
        if source_chain is not None:
            raise ValueError("formal source chain requires paired_reverse_patch")
    else:
        if not isinstance(source_chain, list):
            raise ValueError("paired reverse patch requires a formal source chain")
        expected_source_arm = paired_source_binding["source_condition"]
        if source_chain[-1]["arm"] != expected_source_arm:
            raise ValueError("source formal arm differs from paired direction")
        source_runtime = context.additional_file("paired_source.checkpoint")
        source_bound = source_chain[-1]["checkpoint_file"]
        _require_same_verified_file(
            source_runtime,
            source_bound,
            label="source formal checkpoint",
        )

    def validate_chain(role_name: str, chain: list[dict[str, Any]]) -> tuple[
        list[dict[str, Any]], list[Mapping[str, Any]]
    ]:
        reports: list[dict[str, Any]] = []
        checkpoints: list[Mapping[str, Any]] = []
        previous_manifest = None
        for index, configured in enumerate(chain):
            checkpoint_file = (
                (
                    context.inputs.checkpoint
                    if role_name == "recipient"
                    else context.additional_file("paired_source.checkpoint")
                )
                if index == len(chain) - 1
                else context.additional_file(
                    f"checkpoint_study.{role_name}.{index}.checkpoint"
                )
            )
            finalized_file = context.additional_file(
                f"checkpoint_study.{role_name}.{index}.finalized_manifest"
            )
            ledger_file = context.additional_file(
                "checkpoint_study.transaction_ledger"
            )
            trust = ParentTrust(
                checkpoint_path=checkpoint_file.path,
                finalized_manifest_path=finalized_file.path,
                transaction_ledger_path=ledger_file.path,
                transaction_ledger_sha256=configured[
                    "transaction_ledger_sha256"
                ],
                study_id=configured["study_id"],
                arm=configured["arm"],
                parent_stage=configured["stage"],
                upstream_identity=configured["upstream_identity"],
                artifact_identity=configured["artifact_identity"],
                artifact_root=configured["artifact_root"],
            )
            validated = validate_parent_trust(trust)
            parent_record = validated.manifest["payload"]["parent"]
            if parent_record["finalized_manifest_sha256"] != configured[
                "expected_finalized_manifest_sha256"
            ]:
                raise ValueError("formal finalized manifest digest differs")
            if (
                validated.checkpoint_sha256 != checkpoint_file.digest.sha256
                or validated.checkpoint_size
                != checkpoint_file.digest.size_bytes
            ):
                raise ValueError(
                    "formal checkpoint validator report differs from bound bytes"
                )
            expected_terminal_steps = {
                "stage1": 500_000,
                "stage2": 40_000,
                "stage3": 10_000,
            }
            if parent_record["terminal_step"] != expected_terminal_steps[
                configured["stage"]
            ]:
                raise ValueError("formal checkpoint terminal step is not canonical")
            checkpoint = validated.checkpoint
            provenance = checkpoint["provenance"]
            manifests = provenance["manifests"]
            if manifests["source"]["payload"]["commit_sha"] != (
                context.inputs.model_code.head_sha
            ) or manifests["source"]["payload"]["commit_sha"] != (
                context.inputs.training_code.head_sha
            ):
                raise ValueError(
                    "formal checkpoint source commit differs from the clean "
                    "training/model checkout"
                )
            if previous_manifest is not None and manifests["parent"] != (
                previous_manifest
            ):
                raise ValueError(
                    "formal checkpoint parent does not equal the independently "
                    "validated preceding stage"
                )
            reports.append(
                _checkpoint_stage_report(
                    checkpoint,
                    parent_record=parent_record,
                    checkpoint_sha256=validated.checkpoint_sha256,
                    checkpoint_size=validated.checkpoint_size,
                    checkpoint_file_sha256=checkpoint_file.digest.sha256,
                    finalized_manifest_file_sha256=(
                        finalized_file.digest.sha256
                    ),
                    transaction_ledger_file_sha256=ledger_file.digest.sha256,
                )
            )
            checkpoints.append(checkpoint)
            previous_manifest = validated.manifest
        return reports, checkpoints

    recipient_reports, recipient_checkpoints = validate_chain(
        "recipient", recipient_chain
    )
    source_reports: list[dict[str, Any]] | None = None
    source_checkpoints: list[Mapping[str, Any]] | None = None
    if isinstance(source_chain, list):
        source_reports, source_checkpoints = validate_chain("source", source_chain)
        _validate_formal_paired_chains(
            recipient_reports,
            source_reports,
            recipient_checkpoints=recipient_checkpoints,
            source_checkpoints=source_checkpoints,
            recipient_condition=context.condition,
            source_condition=paired_source_binding["source_condition"],
        )
    payload = {
        "schema_version": 1,
        "scope": "formal",
        "formal_trust_verified": True,
        "model_evidence_scope": (
            "final-stage3"
            if recipient_reports[-1]["stage"] == "stage3"
            else "intermediate-stage-specific"
        ),
        "direction": (
            None
            if source_reports is None
            else {
                "source_condition": paired_source_binding["source_condition"],
                "recipient_condition": context.condition,
            }
        ),
        "recipient_chain": recipient_reports,
        "source_chain": source_reports,
    }
    return {**payload, "binding_sha256": _canonical_sha256(payload)}


def _require_same_verified_file(first: Any, second: Any, *, label: str) -> None:
    if first.digest != second.digest or (first.device, first.inode) != (
        second.device,
        second.inode,
    ):
        raise ValueError(f"{label} is not the exact runtime checkpoint input")


def _checkpoint_stage_report(
    checkpoint: Mapping[str, Any],
    *,
    parent_record: Mapping[str, Any],
    checkpoint_sha256: str,
    checkpoint_size: int,
    checkpoint_file_sha256: str,
    finalized_manifest_file_sha256: str,
    transaction_ledger_file_sha256: str,
) -> dict[str, Any]:
    provenance = checkpoint["provenance"]
    manifests = provenance["manifests"]
    operational = manifests["operational_config"]["payload"]["context"]
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size": checkpoint_size,
        "provenance_sha256": provenance["bundle_sha256"],
        "source_sha256": manifests["source"]["sha256"],
        "environment_sha256": manifests["environment"]["sha256"],
        "prior_sha256": manifests["prior"]["sha256"],
        "architecture_sha256": manifests["architecture"]["sha256"],
        "optimizer_sha256": manifests["optimizer"]["sha256"],
        "seed_sha256": manifests["seed"]["sha256"],
        "treatment_sha256": manifests["treatment"]["sha256"],
        "scientific_sha256": manifests["scientific_config"]["sha256"],
        "cohort_protocol_sha256": manifests["cohort_protocol"]["sha256"],
        "arm_protocol_sha256": manifests["arm_protocol"]["sha256"],
        "study_id": operational["study_id"],
        "output_id": operational["output_id"],
        "mode": operational["arm"],
        "stage": parent_record["stage"],
        "terminal_step": parent_record["terminal_step"],
        "max_checkpoint_bytes": parent_record["max_checkpoint_bytes"],
        "upstream_identity": parent_record["upstream_identity"],
        "artifact_identity": parent_record["artifact_identity"],
        "finalized_manifest_sha256": parent_record[
            "finalized_manifest_sha256"
        ],
        "transaction_ledger_sha256": parent_record[
            "transaction_ledger_sha256"
        ],
        "checkpoint_file_sha256": checkpoint_file_sha256,
        "finalized_manifest_file_sha256": finalized_manifest_file_sha256,
        "transaction_ledger_file_sha256": transaction_ledger_file_sha256,
    }


def _validate_formal_paired_chains(
    recipient_reports: Sequence[Mapping[str, Any]],
    source_reports: Sequence[Mapping[str, Any]],
    *,
    recipient_checkpoints: Sequence[Mapping[str, Any]],
    source_checkpoints: Sequence[Mapping[str, Any]],
    recipient_condition: str,
    source_condition: str,
) -> None:
    if not (
        len(recipient_reports)
        == len(source_reports)
        == len(recipient_checkpoints)
        == len(source_checkpoints)
    ):
        raise ValueError("formal paired checkpoint chains do not align")
    invariant_fields = (
        "study_id",
        "stage",
        "terminal_step",
        "source_sha256",
        "environment_sha256",
        "prior_sha256",
        "architecture_sha256",
        "optimizer_sha256",
        "seed_sha256",
        "scientific_sha256",
        "cohort_protocol_sha256",
        "transaction_ledger_sha256",
        "transaction_ledger_file_sha256",
        "max_checkpoint_bytes",
    )
    for index, (recipient, source, recipient_checkpoint, source_checkpoint) in enumerate(
        zip(
            recipient_reports,
            source_reports,
            recipient_checkpoints,
            source_checkpoints,
            strict=True,
        )
    ):
        mismatches = {
            name: (recipient[name], source[name])
            for name in invariant_fields
            if recipient[name] != source[name]
        }
        if mismatches:
            raise ValueError(
                f"formal paired cohort invariant mismatch at stage {index + 1}: "
                f"{mismatches}"
            )
        if recipient["mode"] != recipient_condition or source["mode"] != source_condition:
            raise ValueError("formal checkpoint modes reverse the paired direction")
        if (
            recipient["checkpoint_sha256"] == source["checkpoint_sha256"]
            or recipient["treatment_sha256"] == source["treatment_sha256"]
            or recipient["arm_protocol_sha256"] == source["arm_protocol_sha256"]
            or recipient["output_id"] == source["output_id"]
        ):
            raise ValueError(
                "formal paired checkpoints/treatments/arm protocols/output ids "
                "must differ"
            )
        recipient_treatment = recipient_checkpoint["provenance"]["manifests"][
            "treatment"
        ]["payload"]
        source_treatment = source_checkpoint["provenance"]["manifests"][
            "treatment"
        ]["payload"]
        treatment_fields = {
            "schema_version",
            "identity_rng_seed",
            "seed_policy",
            "world_size",
        }
        if any(
            recipient_treatment[name] != source_treatment[name]
            for name in treatment_fields
        ):
            raise ValueError(
                "formal paired treatment differs beyond row_identity_mode"
            )


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


def _validate_ranking_parent_manifest(parent: RunManifest) -> None:
    if parent.command != "rank-condition-shift":
        raise ValueError(
            "ranking parent command must be rank-condition-shift"
        )
    if parent.evidence_level != "strict" or parent.legacy_reasons:
        raise ValueError("ranking parent must have strict evidence")
    if parent.sites != ("row_interactor",):
        raise ValueError("ranking parent must select row_interactor")
    if tuple(artifact.name for artifact in parent.artifacts) != (
        "selection.json",
    ):
        raise ValueError(
            "ranking parent must declare exactly selection.json"
        )


def _ranking_directions(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON list")
    result = tuple(value)
    if result != _RANKING_DIRECTIONS:
        raise ValueError(
            f"{name} must equal the frozen bidirectional ordering "
            f"{list(_RANKING_DIRECTIONS)!r}"
        )
    return result


def _validate_ranking_parent(
    parent: RunManifest,
    verified_selection: Any,
    verified_split_protocol: Any,
    *,
    context: Any,
    ranking_parent_manifest_sha256: str,
    representation_parent: RunManifest,
    representation_parent_manifest_sha256: str,
    representation_model_sha256: str,
    representation_model: nn.Module,
    representation_normalizer: nn.Module,
    source_lineage: Mapping[str, Any],
    inference_contract_sha256: str,
    dataset_id: str,
    target_features: Sequence[int],
    control_features: Sequence[int] | None,
    latent_baseline: Any,
    configured_directions: Sequence[str],
    paired_source_binding: Mapping[str, Any] | None,
    sample_roster_sha256: str,
    sample_count: int,
    evaluation_split: str,
) -> dict[str, Any]:
    """Bind an exploratory causal run to one immutable discovery ranking."""

    _validate_ranking_parent_manifest(parent)
    assert_git_commit_is_ancestor(
        context.inputs.analysis_code, parent.analysis_code_sha
    )
    expected_parent_lineage = {
        "model_family": representation_parent.model_family,
        "model_revision": representation_parent.model_revision,
        "training_code_sha": representation_parent.training_code_sha,
        "model_code_sha": representation_parent.model_code_sha,
        "checkpoint": representation_parent.checkpoint,
        "dataset_manifest": representation_parent.dataset_manifest,
        "condition": representation_parent.condition,
        "sites": representation_parent.sites,
        "seed": representation_parent.seed,
    }
    lineage_mismatches = {
        name: (getattr(parent, name), expected)
        for name, expected in expected_parent_lineage.items()
        if getattr(parent, name) != expected
    }
    if lineage_mismatches:
        raise ValueError(
            "ranking parent lineage differs from its representation parent: "
            f"{lineage_mismatches}"
        )
    current_lineage = {
        "model_family": context.model_family,
        "model_revision": context.model_revision,
        "training_code_sha": context.inputs.training_code.head_sha,
        "model_code_sha": context.inputs.model_code.head_sha,
        "dataset_manifest": context.inputs.dataset_manifest.digest,
        "sites": context.sites,
        "seed": context.inputs.contract.seed,
    }
    current_mismatches = {
        name: (getattr(parent, name), expected)
        for name, expected in current_lineage.items()
        if getattr(parent, name) != expected
    }
    if current_mismatches:
        raise ValueError(
            "ranking parent lineage differs from model-causal inputs: "
            f"{current_mismatches}"
        )

    registered_inputs = {item.role: item.sha256 for item in parent.inputs}
    required_registered = {
        "representation.parent_manifest": (
            representation_parent_manifest_sha256
        ),
        "representation.model": representation_model_sha256,
        "ranking.split_protocol": verified_split_protocol.digest.sha256,
    }
    mismatched_inputs = {
        role: (registered_inputs.get(role), expected)
        for role, expected in required_registered.items()
        if registered_inputs.get(role) != expected
    }
    if mismatched_inputs:
        raise ValueError(
            "ranking parent does not bind the active representation/split "
            f"inputs: {mismatched_inputs}"
        )

    selection = _selection_json_object(
        verified_selection, label="ranking selection.json"
    )
    selection = _exact_object(
        selection,
        label="ranking selection.json",
        required=_RANKING_SELECTION_FIELDS,
    )
    split = _ranking_split_protocol(
        verified_split_protocol,
        dataset_id=dataset_id,
        sample_roster_sha256=sample_roster_sha256,
        sample_count=sample_count,
        evaluation_split=evaluation_split,
    )
    _validate_ranking_parent_input_schema(
        parent,
        source_lineage=source_lineage,
        ranking_dataset_count=len(split["feature_ranking_datasets"]),
    )
    common_expected = {
        "schema_version": 1,
        "analysis": "exploratory-condition-shift-ranking",
        "evidence_scope": "exploratory-pilot",
        "formal_claim": "forbidden",
        "site": "row_interactor",
        "conditions": list(_RANKING_CONDITIONS),
        "directions": list(_RANKING_DIRECTIONS),
        "maximum_symmetric_donor_shift_rms_ratio": split[
            "maximum_symmetric_donor_shift_rms_ratio"
        ],
        "score_definition": _RANKING_SCORE_ID,
        "decoder_norm_space": "raw_activation_after_denormalize",
        "raw_activation_rms_definition": _RANKING_RAW_RMS_DEFINITION,
        "representation_parent_manifest_sha256": (
            representation_parent_manifest_sha256
        ),
        "representation_model_sha256": representation_model_sha256,
        "representation_source_lineage_sha256": _canonical_sha256(
            source_lineage
        ),
        "split_protocol_id": split["protocol_id"],
        "split_protocol_sha256": verified_split_protocol.digest.sha256,
        "inference_contract_sha256": inference_contract_sha256,
        "condition_checkpoints_sha256": dict(
            source_lineage["condition_checkpoints_sha256"]
        ),
        "parent_seed": parent.seed,
    }
    selection_mismatches = {
        name: (selection.get(name), expected)
        for name, expected in common_expected.items()
        if selection.get(name) != expected
    }
    if selection_mismatches:
        raise ValueError(
            "ranking selection differs from verified model-causal lineage: "
            f"{selection_mismatches}"
        )
    _required_sha256_value(
        selection["ranking_source_lineage_sha256"],
        name="ranking_source_lineage_sha256",
    )
    if tuple(configured_directions) != tuple(selection["directions"]):
        raise ValueError(
            "configured ranking directions differ from selection.json"
        )
    if paired_source_binding is None:
        raise ValueError(
            "ranking-bound causal execution requires a paired reverse patch"
        )
    effective_direction = (
        f"{paired_source_binding['source_condition']}_to_{context.condition}"
    )
    if effective_direction not in configured_directions:
        raise ValueError(
            "paired source/recipient direction is absent from the ranking "
            "selection"
        )
    frozen_shift_limit = _finite_number(
        selection["maximum_symmetric_donor_shift_rms_ratio"],
        name="ranking maximum_symmetric_donor_shift_rms_ratio",
    )
    if frozen_shift_limit != paired_source_binding.get(
        "maximum_symmetric_donor_shift_rms_ratio"
    ):
        raise ValueError(
            "paired donor-shift threshold differs from ranking selection"
        )

    selection_targets = _ranking_feature_list(
        selection["target_features"], name="ranking target_features"
    )
    selection_controls = _ranking_feature_list(
        selection["control_features"], name="ranking control_features"
    )
    if list(target_features) != list(selection_targets):
        raise ValueError(
            "configured target_features differ from ranking selection"
        )
    if control_features is None or list(control_features) != list(
        selection_controls
    ):
        raise ValueError(
            "configured control_features differ from ranking selection"
        )
    if _json_safe(latent_baseline) != selection["latent_baseline"]:
        raise ValueError(
            "configured latent_baseline differs from ranking selection"
        )

    ranking_protocol = _exact_object(
        selection["ranking_protocol"],
        label="ranking selection protocol",
        required=_RANKING_PROTOCOL_FIELDS,
    )
    ranking_dataset_ids = tuple(split["feature_ranking_datasets"])
    expected_protocol = {
        "dataset_ids": list(ranking_dataset_ids),
        "minimum_nonzero_datasets": ranking_protocol[
            "minimum_nonzero_datasets"
        ],
        "target_count": len(selection_targets),
        "random_candidate_pool_size": ranking_protocol[
            "random_candidate_pool_size"
        ],
        "random_seed": parent.seed,
        "activation_frequency_threshold": 1e-8,
        "decoder_norm_space": "raw_activation_after_denormalize",
        "latent_baseline": selection["latent_baseline"],
        "same_features_both_directions": True,
    }
    if ranking_protocol != expected_protocol:
        raise ValueError(
            "ranking protocol differs from its frozen split/selection"
        )
    minimum_nonzero = _positive_integer(
        ranking_protocol["minimum_nonzero_datasets"],
        name="ranking minimum_nonzero_datasets",
    )
    if minimum_nonzero > len(ranking_dataset_ids):
        raise ValueError(
            "ranking minimum_nonzero_datasets exceeds its dataset roster"
        )
    candidate_pool_size = _positive_integer(
        ranking_protocol["random_candidate_pool_size"],
        name="ranking random_candidate_pool_size",
    )
    expected_feature_protocol = {
        "target_count": len(selection_targets),
        "score": (
            "median across feature-ranking datasets of RMS RoPE-minus-No-PE "
            "latent difference times decoder-direction norm divided by raw "
            "activation RMS"
        ),
        "minimum_nonzero_ranking_datasets": minimum_nonzero,
        "matched_control": (
            "activation-frequency and log-decoder-norm nearest-neighbour pool "
            f"of size {candidate_pool_size}, sampled once with seed "
            f"{parent.seed} without replacement"
        ),
        "same_features_both_directions": True,
    }
    if split["feature_protocol"] != expected_feature_protocol:
        raise ValueError(
            "ranking selection parameters differ from the frozen split protocol"
        )
    if selection["ranking_protocol_sha256"] != _canonical_sha256(
        ranking_protocol
    ):
        raise ValueError("ranking protocol digest is inconsistent")
    if selection["ranking_dataset_roster_sha256"] != _canonical_sha256(
        list(ranking_dataset_ids)
    ):
        raise ValueError("ranking dataset roster digest is inconsistent")

    latent_dim = _positive_int_attribute(representation_model, "latent_dim")
    if max((*selection_targets, *selection_controls)) >= latent_dim:
        raise ValueError(
            "ranking target/control features exceed representation latent_dim"
        )
    if set(selection_targets) & set(selection_controls) or len(
        selection_targets
    ) != len(selection_controls):
        raise ValueError(
            "ranking target/control features must be dose-matched and disjoint"
        )
    smallest_pool = latent_dim - 2 * len(selection_targets) + 1
    if candidate_pool_size > smallest_pool:
        raise ValueError(
            "ranking random_candidate_pool_size would be truncated for a later "
            "target"
        )
    median_scores = _ranking_float_array(
        selection["median_scores"],
        name="median_scores",
        expected=latent_dim,
        non_negative=True,
    )
    nonzero_counts = _ranking_integer_array(
        selection["nonzero_dataset_counts"],
        name="nonzero_dataset_counts",
        expected=latent_dim,
        maximum=len(ranking_dataset_ids),
    )
    activation_frequencies = _ranking_float_array(
        selection["activation_frequencies"],
        name="activation_frequencies",
        expected=latent_dim,
        non_negative=True,
    )
    decoder_norms = _ranking_float_array(
        selection["decoder_norms"],
        name="decoder_norms",
        expected=latent_dim,
        non_negative=True,
    )
    for name, values in (
        ("median_scores", median_scores),
        ("nonzero_dataset_counts", nonzero_counts),
        ("activation_frequencies", activation_frequencies),
        ("decoder_norms", decoder_norms),
    ):
        expected_digest = _ranking_numeric_array_sha256(values)
        if selection[f"{name}_sha256"] != expected_digest:
            raise ValueError(f"ranking {name} digest is inconsistent")
    observed_decoder_norms = raw_space_decoder_feature_norms(
        representation_model, representation_normalizer
    ).cpu().numpy()
    if not np.array_equal(decoder_norms, observed_decoder_norms):
        raise ValueError(
            "ranking decoder norms differ from the bound representation"
        )

    eligible = [
        index
        for index in range(latent_dim)
        if int(nonzero_counts[index]) >= minimum_nonzero
        and float(median_scores[index]) > 0.0
    ]
    eligible.sort(key=lambda index: (-float(median_scores[index]), index))
    if tuple(eligible[: len(selection_targets)]) != selection_targets:
        raise ValueError(
            "ranking target features are not the frozen top eligible features"
        )
    recomputed_controls = tuple(
        matched_random_control_features(
            selection_targets,
            activation_frequencies,
            decoder_norms,
            seed=parent.seed,
            candidate_pool_size=candidate_pool_size,
        )
    )
    if recomputed_controls != selection_controls:
        raise ValueError(
            "ranking matched controls are inconsistent with frozen statistics"
        )
    expected_top = [
        {
            "feature": index,
            "score": float(median_scores[index]),
            "nonzero_dataset_count": int(nonzero_counts[index]),
        }
        for index in sorted(
            range(latent_dim),
            key=lambda index: (-float(median_scores[index]), index),
        )[: min(10, latent_dim)]
    ]
    if selection["top_features"] != expected_top:
        raise ValueError("ranking top_features table is inconsistent")

    activation_binding = _ranking_activation_binding(
        selection["activation_sha256_by_condition"],
        dataset_ids=ranking_dataset_ids,
    )
    if selection["activation_binding_sha256"] != _canonical_sha256(
        activation_binding
    ):
        raise ValueError("ranking activation binding digest is inconsistent")
    registered_activations = sorted(
        digest
        for role, digest in registered_inputs.items()
        if role.startswith("source.activation.")
    )
    selected_activations = sorted(
        digest
        for condition in sorted(activation_binding)
        for digest in activation_binding[condition].values()
    )
    if registered_activations != selected_activations:
        raise ValueError(
            "ranking selection activations differ from manifest-bound inputs"
        )
    _ranking_dataset_scalar_mapping(
        selection["raw_activation_rms_by_dataset"],
        dataset_ids=ranking_dataset_ids,
        name="raw_activation_rms_by_dataset",
    )
    _ranking_dataset_digest_mapping(
        selection["dataset_score_sha256"],
        dataset_ids=ranking_dataset_ids,
        name="dataset_score_sha256",
    )

    return {
        "manifest_sha256": _required_sha256_value(
            ranking_parent_manifest_sha256,
            name="ranking parent manifest digest",
        ),
        "selection_sha256": verified_selection.digest.sha256,
        "split_protocol_sha256": verified_split_protocol.digest.sha256,
        "split_protocol_id": split["protocol_id"],
        "target_features": list(selection_targets),
        "control_features": list(selection_controls),
        "directions": list(configured_directions),
        "effective_direction": effective_direction,
        "evidence_scope": "exploratory-pilot",
        "formal_claim": "forbidden",
    }


def _ranking_split_protocol(
    verified: Any,
    *,
    dataset_id: str,
    sample_roster_sha256: str,
    sample_count: int,
    evaluation_split: str,
) -> dict[str, Any]:
    try:
        payload = json.loads(verified.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ranking split protocol is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise TypeError("ranking split protocol must contain one JSON object")
    required = {
        "schema_version",
        "protocol_id",
        "feature_ranking_datasets",
        "causal_test_datasets",
        "causal_sample_protocol",
        "feature_protocol",
        "causal_protocol",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(
            f"ranking split protocol missing required fields: {missing}"
        )
    if payload["schema_version"] != 1:
        raise ValueError("ranking split protocol schema_version must be one")
    protocol_id = require_public_label(
        payload["protocol_id"], name="ranking split protocol_id"
    )
    ranking = _ranking_dataset_ids(
        payload["feature_ranking_datasets"],
        name="feature_ranking_datasets",
    )
    causal = _ranking_dataset_ids(
        payload["causal_test_datasets"], name="causal_test_datasets"
    )
    if set(ranking) & set(causal):
        raise ValueError(
            "ranking and causal-test dataset rosters must be disjoint"
        )
    if dataset_id not in causal:
        raise ValueError(
            "model-causal dataset is absent from the frozen causal-test roster"
        )
    causal_samples = _exact_object(
        payload["causal_sample_protocol"],
        label="ranking causal_sample_protocol",
        required={
            "split",
            "maximum_rows_per_dataset",
            "selection",
            "sample_roster_sha256_by_dataset",
            "sample_count_by_dataset",
        },
    )
    if causal_samples["split"] != "val" or evaluation_split != "val":
        raise ValueError("ranking-bound causal samples must use the val split")
    maximum_rows = _positive_integer(
        causal_samples["maximum_rows_per_dataset"],
        name="causal_sample_protocol.maximum_rows_per_dataset",
    )
    selection_rule = causal_samples["selection"]
    if not isinstance(selection_rule, str) or not selection_rule:
        raise TypeError("causal_sample_protocol.selection must be a non-empty string")
    roster_digests = _ranking_causal_sample_digest_mapping(
        causal_samples["sample_roster_sha256_by_dataset"],
        dataset_ids=causal,
    )
    sample_counts = _ranking_causal_sample_count_mapping(
        causal_samples["sample_count_by_dataset"],
        dataset_ids=causal,
        maximum=maximum_rows,
    )
    if roster_digests[dataset_id] != _required_sha256_value(
        sample_roster_sha256, name="current sample roster digest"
    ):
        raise ValueError(
            "current sample roster digest differs from the frozen split protocol"
        )
    if sample_counts[dataset_id] != sample_count:
        raise ValueError(
            "current sample roster count differs from the frozen split protocol"
        )
    feature = payload["feature_protocol"]
    if not isinstance(feature, Mapping) or feature.get(
        "same_features_both_directions"
    ) is not True:
        raise ValueError(
            "ranking split protocol must freeze the same features both directions"
        )
    causal_protocol = payload["causal_protocol"]
    if not isinstance(causal_protocol, Mapping):
        raise TypeError("ranking causal_protocol must be a JSON object")
    if (
        causal_protocol.get("directions") != list(_RANKING_DIRECTIONS)
        or causal_protocol.get("checkpoint_scope") != "exploratory_pilot"
        or causal_protocol.get("formal_claim") != "forbidden"
    ):
        raise ValueError("ranking causal protocol is not the frozen pilot scope")
    shift_limit = _finite_number(
        causal_protocol.get("maximum_symmetric_donor_shift_rms_ratio"),
        name="causal protocol maximum_symmetric_donor_shift_rms_ratio",
    )
    if not 1.0 <= shift_limit <= _MAX_FORMAL_SYMMETRIC_DONOR_SHIFT_RMS_RATIO:
        raise ValueError("ranking causal protocol donor-shift limit is unsafe")
    return {
        "protocol_id": protocol_id,
        "feature_ranking_datasets": list(ranking),
        "causal_test_datasets": list(causal),
        "causal_sample_protocol": {
            "split": "val",
            "maximum_rows_per_dataset": maximum_rows,
            "selection": selection_rule,
            "sample_roster_sha256_by_dataset": roster_digests,
            "sample_count_by_dataset": sample_counts,
        },
        "feature_protocol": dict(feature),
        "maximum_symmetric_donor_shift_rms_ratio": shift_limit,
    }


def _ranking_dataset_ids(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty JSON list")
    result = tuple(
        require_public_label(item, name=f"{name} item") for item in value
    )
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _ranking_causal_sample_digest_mapping(
    value: Any, *, dataset_ids: Sequence[str]
) -> dict[str, str]:
    if not isinstance(value, Mapping) or tuple(sorted(value)) != tuple(dataset_ids):
        raise ValueError(
            "causal sample-roster digest mapping differs from the causal-test roster"
        )
    return {
        dataset_id: _required_sha256_value(
            value[dataset_id], name=f"causal sample roster {dataset_id}"
        )
        for dataset_id in dataset_ids
    }


def _ranking_causal_sample_count_mapping(
    value: Any, *, dataset_ids: Sequence[str], maximum: int
) -> dict[str, int]:
    if not isinstance(value, Mapping) or tuple(sorted(value)) != tuple(dataset_ids):
        raise ValueError(
            "causal sample-count mapping differs from the causal-test roster"
        )
    result = {
        dataset_id: _positive_integer(
            value[dataset_id], name=f"causal sample count {dataset_id}"
        )
        for dataset_id in dataset_ids
    }
    if any(count > maximum for count in result.values()):
        raise ValueError("causal sample count exceeds maximum_rows_per_dataset")
    return result


def _validate_ranking_parent_input_schema(
    parent: RunManifest,
    *,
    source_lineage: Mapping[str, Any],
    ranking_dataset_count: int,
) -> None:
    """Require the ranking producer's exact path-free input-role schema."""

    singleton_roles = {
        "ranking.split_protocol",
        "representation.parent_manifest",
        "representation.model",
    }
    prefixes = {
        "collect_manifest": "source.collect_manifest.",
        "collect_index": "source.collect_index.",
        "activation": "source.activation.",
    }
    grouped: dict[str, dict[str, Any]] = {name: {} for name in prefixes}
    unknown: list[str] = []
    for item in parent.inputs:
        if item.role in singleton_roles:
            continue
        matched = False
        for name, prefix in prefixes.items():
            if not item.role.startswith(prefix):
                continue
            suffix = item.role.removeprefix(prefix)
            _required_sha256_value(suffix, name=f"ranking input role {item.role}")
            grouped[name][suffix] = item
            matched = True
            break
        if not matched:
            unknown.append(item.role)
    if unknown:
        raise ValueError(f"ranking parent contains unknown input roles: {unknown}")
    observed_singletons = {
        item.role for item in parent.inputs if item.role in singleton_roles
    }
    if observed_singletons != singleton_roles:
        raise ValueError("ranking parent singleton input roles are incomplete")

    raw_collect_lineage = source_lineage.get("collect_parent_manifests_sha256")
    if not isinstance(raw_collect_lineage, Mapping) or set(raw_collect_lineage) != {
        "none",
        "rope",
    }:
        raise ValueError("ranking source collect lineage must contain none and rope")
    collect_lineage_by_condition: dict[str, set[str]] = {}
    for condition in ("none", "rope"):
        values = raw_collect_lineage[condition]
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(
                "ranking source lineage requires collect parents for each condition"
            )
        normalized = tuple(
            _required_sha256_value(
                value, name=f"ranking {condition} collect parent"
            )
            for value in values
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                "ranking source collect lineage must be unique per condition"
            )
        collect_lineage_by_condition[condition] = set(normalized)

    manifest_inputs = grouped["collect_manifest"]
    if len(manifest_inputs) != 2:
        raise ValueError(
            "ranking parent must select exactly one collect parent per condition"
        )
    if any(item.sha256 != suffix for suffix, item in manifest_inputs.items()):
        raise ValueError(
            "ranking collect-manifest role suffix differs from its file digest"
        )
    selected = set(manifest_inputs)
    selected_by_condition = {
        condition: selected & collect_lineage_by_condition[condition]
        for condition in ("none", "rope")
    }
    if any(len(values) != 1 for values in selected_by_condition.values()):
        raise ValueError(
            "ranking parent must select exactly one registered collect parent "
            "from each condition"
        )
    if set().union(*selected_by_condition.values()) != selected:
        raise ValueError(
            "ranking collect-manifest roles contain an unregistered parent"
        )
    if selected_by_condition["none"] & selected_by_condition["rope"]:
        raise ValueError(
            "ranking collect parent cannot belong to both conditions"
        )
    if set(grouped["collect_index"]) != set(manifest_inputs):
        raise ValueError(
            "ranking collect-index roles differ from collect-manifest parents"
        )
    expected_activation_count = 2 * ranking_dataset_count
    if len(grouped["activation"]) != expected_activation_count:
        raise ValueError(
            "ranking activation input count differs from two conditions times the "
            "feature-ranking roster"
        )
    expected_total = (
        len(singleton_roles) + 4 + expected_activation_count
    )
    if len(parent.inputs) != expected_total:
        raise ValueError("ranking parent input-role schema is not exact")


def _ranking_feature_list(value: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty JSON list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError(f"{name} must contain integer indices")
    result = tuple(value)
    if len(set(result)) != len(result) or min(result) < 0:
        raise ValueError(f"{name} must contain unique non-negative indices")
    return result


def _ranking_float_array(
    value: Any,
    *,
    name: str,
    expected: int,
    non_negative: bool,
) -> np.ndarray:
    if not isinstance(value, list):
        raise TypeError(f"ranking {name} must be a JSON list")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (expected,) or not np.isfinite(array).all():
        raise ValueError(f"ranking {name} shape or finiteness is invalid")
    if non_negative and (array < 0.0).any():
        raise ValueError(f"ranking {name} must be non-negative")
    return array


def _ranking_integer_array(
    value: Any,
    *,
    name: str,
    expected: int,
    maximum: int,
) -> np.ndarray:
    if (
        not isinstance(value, list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise TypeError(f"ranking {name} must contain JSON integers")
    array = np.asarray(value, dtype=np.int64)
    if (
        array.shape != (expected,)
        or (array < 0).any()
        or (array > maximum).any()
    ):
        raise ValueError(f"ranking {name} values are outside their roster")
    return array


def _ranking_numeric_array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype.kind == "f":
        canonical = np.asarray(array, dtype="<f8", order="C")
    elif array.dtype.kind in {"i", "u"}:
        canonical = np.asarray(array, dtype="<i8", order="C")
    else:
        raise TypeError("ranking numeric digest requires floats or integers")
    header = json.dumps(
        {"dtype": canonical.dtype.str, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(
        header + b"\0" + canonical.tobytes(order="C")
    ).hexdigest()


def _ranking_activation_binding(
    value: Any, *, dataset_ids: Sequence[str]
) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != {"none", "rope"}:
        raise ValueError(
            "ranking activation binding must contain exactly none and rope"
        )
    result: dict[str, dict[str, str]] = {}
    for condition in sorted(value):
        datasets = value[condition]
        if not isinstance(datasets, Mapping) or tuple(sorted(datasets)) != tuple(
            dataset_ids
        ):
            raise ValueError(
                "ranking activation dataset roster differs from the split"
            )
        result[condition] = {
            dataset_id: _required_sha256_value(
                datasets[dataset_id], name="ranking activation digest"
            )
            for dataset_id in dataset_ids
        }
    return result


def _ranking_dataset_scalar_mapping(
    value: Any, *, dataset_ids: Sequence[str], name: str
) -> dict[str, float]:
    if not isinstance(value, Mapping) or tuple(sorted(value)) != tuple(dataset_ids):
        raise ValueError(f"ranking {name} roster differs from the split")
    result = {
        dataset_id: _finite_number(value[dataset_id], name=f"{name} value")
        for dataset_id in dataset_ids
    }
    if any(item <= 0.0 for item in result.values()):
        raise ValueError(f"ranking {name} values must be positive")
    return result


def _ranking_dataset_digest_mapping(
    value: Any, *, dataset_ids: Sequence[str], name: str
) -> dict[str, str]:
    if not isinstance(value, Mapping) or tuple(sorted(value)) != tuple(dataset_ids):
        raise ValueError(f"ranking {name} roster differs from the split")
    return {
        dataset_id: _required_sha256_value(
            value[dataset_id], name=f"{name} value"
        )
        for dataset_id in dataset_ids
    }


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


def _validate_official_driver(
    driver: OfficialTabICLDriver,
    *,
    context: Any,
    expected_condition: str | None = None,
    expected_model_sha: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> None:
    expected_condition = expected_condition or context.condition
    expected_model_sha = expected_model_sha or context.inputs.model_code.head_sha
    expected_checkpoint_sha256 = (
        expected_checkpoint_sha256 or context.inputs.checkpoint.digest.sha256
    )
    if driver.source_evidence_level != "strict":
        raise RuntimeError("formal driver source evidence is not strict")
    if driver.fit_context != "talent-train":
        raise RuntimeError("formal driver did not fit only the TALENT train split")
    if (
        driver.model_sha != expected_model_sha
        or driver.checkpoint_sha != expected_checkpoint_sha256
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
    if raw_mode != expected_condition:
        raise RuntimeError(
            "checkpoint row identity mode differs from causal condition"
        )


def _validate_paired_source_code_attestation(
    verified: Any,
    *,
    source_condition: str,
    model_code_sha: str,
    checkpoint_sha256: str,
    inference_contract_sha256: str,
) -> None:
    try:
        payload = json.loads(verified.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("paired source code attestation is not valid JSON") from error
    attestation = _exact_object(
        payload,
        label="paired source code attestation",
        required={
            "schema_version",
            "evidence_kind",
            "source_condition",
            "model_code_sha",
            "checkpoint_sha256",
            "inference_contract_sha256",
        },
    )
    expected = {
        "schema_version": 1,
        "evidence_kind": "clean-git-checkout",
        "source_condition": source_condition,
        "model_code_sha": model_code_sha,
        "checkpoint_sha256": checkpoint_sha256,
        "inference_contract_sha256": inference_contract_sha256,
    }
    mismatches = {
        name: (attestation.get(name), value)
        for name, value in expected.items()
        if attestation.get(name) != value
    }
    if mismatches:
        raise ValueError(
            "paired source code attestation differs from verified inputs: "
            f"{mismatches}"
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
    verified_selection: Any,
    verified_freeze: Any,
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
    sample_roster_sha256: str,
    paired_source_binding: Mapping[str, Any] | None,
    maximum_symmetric_donor_shift_rms_ratio: float | None,
    checkpoint_study_sha256: str,
) -> None:
    """Bind held-out execution to one non-circular select-features freeze."""

    if parent.command != "select-features" or parent.evidence_level != "strict":
        raise ValueError("held_out parent must be a strict select-features run")
    if parent.legacy_reasons:
        raise ValueError("select-features parent must not contain legacy reasons")
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
            f"select-features lineage differs from held_out run: {mismatches}"
        )
    registered_inputs = {item.role: item.sha256 for item in parent.inputs}
    selection = _selection_json_object(
        verified_selection, label="selection.json"
    )
    freeze = _selection_json_object(verified_freeze, label="freeze.json")
    summary = _selection_json_object(verified_summary, label="summary.json")
    selection = _exact_object(
        selection,
        label="selection.json",
        required={
            "schema_version",
            "analysis",
            "evidence_scope",
            "condition",
            "site",
            "random_seed",
            "evidence_family",
            "statistics",
            "source_advantage_prerequisite",
            "common_lineage",
            "validation_dataset_ids",
            "validation_dataset_fingerprints_sha256",
            "validation_dataset_fingerprint_manifest_sha256",
            "heldout",
            "candidate_results",
            "selected_candidates",
        },
    )
    freeze = _exact_object(
        freeze,
        label="freeze.json",
        required={
            "schema_version",
            "evidence_scope",
            "condition",
            "site",
            "random_seed",
            "evaluation_sample_rosters_sha256",
            "validation_dataset_fingerprints_sha256",
            "validation_dataset_fingerprint_manifest_sha256",
            "selected_interventions",
            "representation_model_sha256",
            "representation_parent_manifest_sha256",
            "model_sha",
            "checkpoint_sha256",
            "inference_contract_sha256",
            "paired_source_direction",
            "paired_source_binding",
            "maximum_symmetric_donor_shift_rms_ratio",
            "source_advantage_prerequisite",
            "confirmation_protocol",
            "representation_source_lineage_sha256",
            "validation_selection_sha256",
            "checkpoint_study_sha256",
        },
    )
    summary = _exact_object(
        summary,
        label="selection summary",
        required={
            "schema_version",
            "analysis",
            "evidence_scope",
            "condition",
            "site",
            "random_seed",
            "evidence_family",
            "source_advantage_prerequisite",
            "confirmation_protocol",
            "selected_interventions",
            "selection_count",
            "validation_dataset_ids",
            "validation_dataset_fingerprints_sha256",
            "validation_dataset_fingerprint_manifest_sha256",
            "heldout_dataset_ids",
            "evaluation_sample_rosters_sha256",
            "validation_selection_sha256",
            "common_lineage",
            "checkpoint_study_sha256",
        },
    )
    common_scope = {
        "schema_version": 1,
        "analysis": "validation-feature-selection",
        "condition": context.condition,
        "site": site,
        "random_seed": random_seed,
    }
    for name, expected in common_scope.items():
        if selection.get(name) != expected or summary.get(name) != expected:
            raise ValueError(f"selection {name} differs from held_out inputs")
    if selection["evidence_scope"] != "validation-selection":
        raise ValueError("selection.json evidence scope is invalid")
    if (
        freeze["schema_version"] != 2
        or freeze["evidence_scope"] != "validation-frozen"
        or summary["evidence_scope"] != "validation-frozen"
    ):
        raise ValueError("selection freeze/summary schema or scope is invalid")
    for name in ("condition", "site", "random_seed"):
        if freeze[name] != common_scope[name]:
            raise ValueError(f"freeze {name} differs from held_out inputs")
    if selection["evidence_family"] != _SELECTION_EVIDENCE_FAMILY or (
        summary["evidence_family"] != _SELECTION_EVIDENCE_FAMILY
    ):
        raise ValueError("selection evidence family is not canonical")

    statistics = _validated_selection_statistics(selection["statistics"])
    source_advantage = _validated_source_advantage(
        selection["source_advantage_prerequisite"],
        statistics=statistics,
        random_seed=random_seed,
    )
    if (
        freeze["source_advantage_prerequisite"] != source_advantage
        or summary["source_advantage_prerequisite"] != source_advantage
    ):
        raise ValueError("source-advantage prerequisite differs across artifacts")

    selection_sha256 = verified_selection.digest.sha256
    if (
        freeze["validation_selection_sha256"] != selection_sha256
        or summary["validation_selection_sha256"] != selection_sha256
    ):
        raise ValueError("freeze does not bind the exact selection.json bytes")
    expected_lineage = _validated_selection_common_lineage(
        selection["common_lineage"],
        context=context,
        site=site,
        random_seed=random_seed,
        representation_model_sha256=representation_model_sha256,
        representation_parent_manifest_sha256=(
            representation_parent_manifest_sha256
        ),
        inference_contract_sha256=inference_contract_sha256,
        source_lineage=source_lineage,
        paired_source_binding=paired_source_binding,
        checkpoint_study_sha256=checkpoint_study_sha256,
    )
    summary_lineage_fields = {
        name: expected_lineage[name]
        for name in (
            "model_code_sha",
            "checkpoint_sha256",
            "representation_model_sha256",
            "representation_parent_manifest_sha256",
            "inference_contract_sha256",
            "paired_source_direction",
            "paired_source_binding",
            "checkpoint_study_sha256",
        )
    }
    if summary["common_lineage"] != summary_lineage_fields:
        raise ValueError("selection summary common lineage differs")
    if summary["checkpoint_study_sha256"] != checkpoint_study_sha256:
        raise ValueError("selection summary checkpoint study differs")

    expected_source_direction = expected_lineage["paired_source_direction"]
    expected_source_binding = expected_lineage["paired_source_binding"]
    if (
        freeze["paired_source_direction"] != expected_source_direction
        or freeze["paired_source_binding"] != expected_source_binding
    ):
        raise ValueError("freeze paired-source direction or binding differs")
    if maximum_symmetric_donor_shift_rms_ratio is None:
        raise ValueError("held_out donor evaluation requires a paired source")
    if paired_source_binding is None:
        raise ValueError("held_out donor evaluation lacks a paired-source binding")
    runtime_alignment = {
        "source_condition": paired_source_binding.get("source_condition"),
        "recipient_condition": paired_source_binding.get("recipient_condition"),
        "sample_roster_sha256": paired_source_binding.get(
            "sample_roster_sha256"
        ),
        "maximum_symmetric_donor_shift_rms_ratio": paired_source_binding.get(
            "maximum_symmetric_donor_shift_rms_ratio"
        ),
    }
    expected_runtime_alignment = {
        **expected_source_direction,
        "sample_roster_sha256": sample_roster_sha256,
        "maximum_symmetric_donor_shift_rms_ratio": (
            maximum_symmetric_donor_shift_rms_ratio
        ),
    }
    if runtime_alignment != expected_runtime_alignment:
        raise ValueError(
            "paired source direction, roster, or donor-shift threshold differs "
            "from the held_out run"
        )
    frozen_shift_limit = _finite_number(
        freeze["maximum_symmetric_donor_shift_rms_ratio"],
        name="maximum_symmetric_donor_shift_rms_ratio",
    )
    if (
        frozen_shift_limit != maximum_symmetric_donor_shift_rms_ratio
        or not 1.0
        <= frozen_shift_limit
        <= _MAX_FORMAL_SYMMETRIC_DONOR_SHIFT_RMS_RATIO
    ):
        raise ValueError("frozen donor-shift threshold differs or is unsafe")

    freeze_bindings = {
        "representation_model_sha256": representation_model_sha256,
        "representation_parent_manifest_sha256": (
            representation_parent_manifest_sha256
        ),
        "model_sha": context.inputs.model_code.head_sha,
        "checkpoint_sha256": context.inputs.checkpoint.digest.sha256,
        "inference_contract_sha256": inference_contract_sha256,
        "representation_source_lineage_sha256": _canonical_sha256(
            source_lineage
        ),
        "checkpoint_study_sha256": checkpoint_study_sha256,
    }
    binding_mismatches = {
        name: (freeze.get(name), expected)
        for name, expected in freeze_bindings.items()
        if freeze.get(name) != expected
    }
    if binding_mismatches:
        raise ValueError(f"freeze lineage differs: {binding_mismatches}")

    heldout = _exact_object(
        selection["heldout"],
        label="selection heldout",
        required={
            "dataset_ids",
            "evaluation_sample_rosters_sha256",
            "confirmation_protocol",
        },
    )
    roster_mapping = _selection_roster_mapping(
        freeze["evaluation_sample_rosters_sha256"]
    )
    heldout_ids = list(roster_mapping)
    frozen = _validated_frozen_interventions(freeze["selected_interventions"])
    confirmation = _validated_confirmation_protocol(
        freeze["confirmation_protocol"],
        expected_random_seed=random_seed,
        heldout_dataset_count=len(heldout_ids),
        expected_maximum_candidates=statistics["maximum_selections"],
        expected_frozen_count=len(frozen),
    )
    if summary["confirmation_protocol"] != confirmation:
        raise ValueError("selection summary confirmation protocol differs")
    if heldout["confirmation_protocol"] != confirmation:
        raise ValueError("selection heldout confirmation protocol differs")
    if len(heldout_ids) < max(
        confirmation["minimum_heldout_datasets"],
        confirmation["required_heldout_datasets"],
    ):
        raise ValueError("freeze contains too few held-out dataset rosters")
    if (
        heldout["dataset_ids"] != heldout_ids
        or heldout["evaluation_sample_rosters_sha256"] != roster_mapping
        or summary["heldout_dataset_ids"] != heldout_ids
        or summary["evaluation_sample_rosters_sha256"] != roster_mapping
    ):
        raise ValueError("held-out roster mappings differ across selection artifacts")
    if roster_mapping.get(dataset_id) != sample_roster_sha256:
        raise ValueError("current held-out sample roster is not exactly frozen")
    for roster_sha256 in roster_mapping.values():
        role = f"heldout.sample_roster.{roster_sha256}"
        if registered_inputs.get(role) != roster_sha256:
            raise ValueError(
                "a frozen held-out roster is absent from the selection manifest"
            )
    assert_dataset_roster(
        context.inputs.dataset_manifest,
        heldout_ids,
        required_split="held_out",
    )
    validation_ids = _selection_dataset_ids(
        selection["validation_dataset_ids"], name="validation_dataset_ids"
    )
    validation_fingerprints = _validated_dataset_fingerprint_mapping(
        selection["validation_dataset_fingerprints_sha256"],
        expected_dataset_ids=validation_ids,
        label="validation",
    )
    validation_fingerprint_manifest_sha256 = _required_sha256_value(
        selection["validation_dataset_fingerprint_manifest_sha256"],
        name="validation_dataset_fingerprint_manifest_sha256",
    )
    if (
        validation_fingerprint_manifest_sha256
        != _canonical_sha256(validation_fingerprints)
        or freeze["validation_dataset_fingerprints_sha256"]
        != validation_fingerprints
        or summary["validation_dataset_fingerprints_sha256"]
        != validation_fingerprints
        or freeze["validation_dataset_fingerprint_manifest_sha256"]
        != validation_fingerprint_manifest_sha256
        or summary["validation_dataset_fingerprint_manifest_sha256"]
        != validation_fingerprint_manifest_sha256
    ):
        raise ValueError(
            "validation dataset fingerprint mapping or manifest differs"
        )
    if (
        summary["validation_dataset_ids"] != validation_ids
        or set(validation_ids) & set(heldout_ids)
        or dataset_id in validation_ids
    ):
        raise ValueError("validation and held-out dataset rosters are inconsistent")
    assert_dataset_roster(
        context.inputs.dataset_manifest,
        validation_ids,
        required_split="validation",
    )
    if statistics["effective_validation_dataset_count"] != len(validation_ids):
        raise ValueError("selection statistics validation count differs")
    source_dataset_ids = [
        item["dataset_id"] for item in source_advantage["dataset_effects"]
    ]
    if source_dataset_ids != validation_ids:
        raise ValueError("source-advantage validation roster differs")

    if summary["selected_interventions"] != frozen or summary[
        "selection_count"
    ] != len(frozen):
        raise ValueError("selection summary frozen interventions differ")
    selected_ids = [item["candidate_id"] for item in frozen]
    if selection["selected_candidates"] != selected_ids:
        raise ValueError("selection candidate roster differs from freeze")
    expected_intervention = {
        "target_features": list(target_features),
        "control_features": (
            None if control_features is None else list(control_features)
        ),
        "latent_baseline": _json_safe(latent_baseline),
    }
    matches = [
        item
        for item in frozen
        if {
            "target_features": item["target_features"],
            "control_features": item["control_features"],
            "latent_baseline": item["latent_baseline"],
        }
        == expected_intervention
    ]
    if len(matches) != 1:
        raise ValueError("held_out intervention is not uniquely frozen")
    _validate_selected_candidate_results(
        selection["candidate_results"],
        selected_ids=selected_ids,
        statistics=statistics,
        source_advantage=source_advantage,
        validation_dataset_ids=validation_ids,
        frozen_interventions=frozen,
        random_seed=random_seed,
    )


def _selection_json_object(verified: Any, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(verified.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} must contain a JSON object")
    return dict(payload)


def _validated_selection_statistics(value: Any) -> dict[str, Any]:
    fields = {
        "multiplicity_method",
        "fdr_control_unit",
        "candidate_composite_method",
        "component_count_per_candidate",
        "fdr_alpha",
        "confidence_level",
        "bootstrap_method",
        "bootstrap_resamples",
        "p_value_method",
        "sign_flip_resamples",
        "minimum_validation_datasets",
        "minimum_positive_fraction",
        "maximum_selections",
        "ranking_rule",
        "family_size",
        "harmonic_factor",
        "rank_one_threshold",
        "exact_sign_flip_minimum_p_value",
        "required_validation_datasets",
        "effective_validation_dataset_count",
        "sign_flip_mode",
        "monte_carlo_minimum_p_value",
    }
    result = _exact_object(
        value, label="selection statistics", required=fields
    )
    expected_text = {
        "multiplicity_method": "benjamini-yekutieli",
        "fdr_control_unit": "candidate-intersection-union-hypothesis",
        "candidate_composite_method": "maximum-of-six-component-p-values",
        "bootstrap_method": "paired-dataset-bootstrap",
        "p_value_method": "one-sided-paired-sign-flip",
        "ranking_rule": "maximum_minimum_mean_evidence",
    }
    for name, expected in expected_text.items():
        if result[name] != expected:
            raise ValueError(f"selection statistics {name} is not canonical")
    alpha = _finite_number(result["fdr_alpha"], name="fdr_alpha")
    confidence = _finite_number(
        result["confidence_level"], name="confidence_level"
    )
    positive_fraction = _finite_number(
        result["minimum_positive_fraction"],
        name="minimum_positive_fraction",
    )
    if not 0.0 < alpha <= 1.0:
        raise ValueError("fdr_alpha must lie in (0, 1]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if not 0.0 <= positive_fraction <= 1.0:
        raise ValueError("minimum_positive_fraction must lie in [0, 1]")
    for name in (
        "component_count_per_candidate",
        "bootstrap_resamples",
        "sign_flip_resamples",
        "minimum_validation_datasets",
        "maximum_selections",
        "family_size",
        "required_validation_datasets",
        "effective_validation_dataset_count",
    ):
        result[name] = _positive_integer(result[name], name=name)
    if result["component_count_per_candidate"] != 6:
        raise ValueError("selection candidate composite must contain six components")
    if result["minimum_validation_datasets"] < 8:
        raise ValueError("selection requires at least eight validation datasets")
    if result["maximum_selections"] > 2:
        raise ValueError("selection may freeze at most two candidates")
    harmonic_factor = _finite_number(
        result["harmonic_factor"], name="harmonic_factor"
    )
    expected_harmonic = sum(
        1.0 / index for index in range(1, result["family_size"] + 1)
    )
    if harmonic_factor != expected_harmonic:
        raise ValueError("selection harmonic factor is inconsistent")
    rank_one = _finite_number(
        result["rank_one_threshold"], name="rank_one_threshold"
    )
    expected_rank_one = alpha / (result["family_size"] * harmonic_factor)
    if rank_one != expected_rank_one:
        raise ValueError("selection rank-one threshold is inconsistent")
    expected_required = max(
        8,
        int(
            np.ceil(
                np.log2(
                    result["family_size"] * harmonic_factor / alpha
                )
            )
        ),
    )
    if result["required_validation_datasets"] != expected_required:
        raise ValueError("required_validation_datasets is inconsistent")
    if result["effective_validation_dataset_count"] < max(
        result["minimum_validation_datasets"], expected_required
    ):
        raise ValueError("effective validation roster is below its frozen minimum")
    exact_minimum = _finite_number(
        result["exact_sign_flip_minimum_p_value"],
        name="exact_sign_flip_minimum_p_value",
    )
    effective_count = result["effective_validation_dataset_count"]
    if exact_minimum != 2.0 ** (-effective_count):
        raise ValueError("exact sign-flip minimum p-value is inconsistent")
    expected_mode = (
        "exact-enumeration"
        if effective_count <= 20
        else "fixed-seed-monte-carlo"
    )
    if result["sign_flip_mode"] != expected_mode:
        raise ValueError("selection sign-flip mode is inconsistent")
    monte_carlo_minimum = result["monte_carlo_minimum_p_value"]
    if expected_mode == "exact-enumeration":
        if monte_carlo_minimum is not None:
            raise ValueError("exact sign-flip selection cannot have an MC floor")
        attainable_minimum = exact_minimum
    else:
        expected_mc = 1.0 / (result["sign_flip_resamples"] + 1)
        if _finite_number(
            monte_carlo_minimum, name="monte_carlo_minimum_p_value"
        ) != expected_mc:
            raise ValueError("Monte Carlo p-value floor is inconsistent")
        attainable_minimum = expected_mc
    if attainable_minimum > rank_one:
        raise ValueError("selection sign-flip test cannot resolve rank-one threshold")
    return result


def _validated_source_advantage(
    value: Any, *, statistics: Mapping[str, Any], random_seed: int
) -> dict[str, Any]:
    fields = {
        "name",
        "metric",
        "direction",
        "scope",
        "effective_dataset_count",
        "dataset_effects",
        "components",
        "minimum_component_mean_effect",
        "composite_p_value",
        "direction_ci_replication_gates_passed",
        "p_value_gate_deferred_to_candidate_composite",
    }
    result = _exact_object(
        value, label="source-advantage prerequisite", required=fields
    )
    if {
        name: result[name] for name in ("name", "metric", "direction", "scope")
    } != _SELECTION_EVIDENCE_FAMILY[-1]:
        raise ValueError("source-advantage evidence definition is not canonical")
    count = _positive_integer(
        result["effective_dataset_count"],
        name="source advantage effective_dataset_count",
    )
    effects = result["dataset_effects"]
    if not isinstance(effects, list) or len(effects) != count:
        raise ValueError("source-advantage dataset effects are incomplete")
    normalized_effects: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in effects:
        item = _exact_object(
            raw,
            label="source-advantage dataset effect",
            required={
                "dataset_id",
                "source_native_advantage",
                "source_no_op_advantage",
            },
        )
        dataset_id = require_public_label(
            item["dataset_id"], name="source-advantage dataset_id"
        )
        if dataset_id in seen:
            raise ValueError("source-advantage dataset ids must be unique")
        seen.add(dataset_id)
        normalized_effects.append(
            {
                "dataset_id": dataset_id,
                "source_native_advantage": _finite_number(
                    item["source_native_advantage"],
                    name="source native advantage",
                ),
                "source_no_op_advantage": _finite_number(
                    item["source_no_op_advantage"],
                    name="source no-op advantage",
                ),
            }
        )
    if [item["dataset_id"] for item in normalized_effects] != sorted(seen):
        raise ValueError("source-advantage dataset effects must be sorted")
    result["dataset_effects"] = normalized_effects
    component_fields = {
        "name",
        "metric",
        "direction",
        "mean_effect",
        "median_effect",
        "confidence_low",
        "confidence_high",
        "positive_fraction",
        "p_value",
        "passes_direction_ci_replication",
    }
    component_definitions = (
        {
            "name": "source_native_advantage",
            "metric": (
                "mean_log_loss_improvement_of_source_native_vs_recipient_native"
            ),
            "direction": "positive",
            "effect_field": "source_native_advantage",
        },
        {
            "name": "source_no_op_advantage",
            "metric": (
                "mean_log_loss_improvement_of_source_no_op_vs_recipient_no_op"
            ),
            "direction": "positive",
            "effect_field": "source_no_op_advantage",
        },
    )
    raw_components = result["components"]
    if not isinstance(raw_components, list) or len(raw_components) != 2:
        raise ValueError("source advantage requires exactly two IUT components")
    components: list[dict[str, Any]] = []
    global_offset = statistics["family_size"] * (
        len(_SELECTION_EVIDENCE_FAMILY) - 1
    )
    for component_offset, (raw_component, definition) in enumerate(
        zip(raw_components, component_definitions, strict=True)
    ):
        component = _exact_object(
            raw_component,
            label="source-advantage component",
            required=component_fields,
        )
        if {
            name: component[name] for name in ("name", "metric", "direction")
        } != {name: definition[name] for name in ("name", "metric", "direction")}:
            raise ValueError("source-advantage component definition is not canonical")
        for name in (
            "mean_effect",
            "median_effect",
            "confidence_low",
            "confidence_high",
            "positive_fraction",
            "p_value",
        ):
            component[name] = _finite_number(component[name], name=name)
        values = np.asarray(
            [item[definition["effect_field"]] for item in normalized_effects],
            dtype=np.float64,
        )
        if component["mean_effect"] != float(np.mean(values)) or component[
            "median_effect"
        ] != float(np.median(values)):
            raise ValueError("source-advantage component aggregate is inconsistent")
        expected_low, expected_high = paired_bootstrap_ci(
            values,
            confidence=statistics["confidence_level"],
            n_resamples=statistics["bootstrap_resamples"],
            seed=random_seed + global_offset + component_offset,
        )
        expected_p_value = paired_sign_flip_p_value(
            values,
            n_resamples=statistics["sign_flip_resamples"],
            seed=(
                random_seed
                + 100_000
                + global_offset
                + component_offset
            ),
        )
        expected_replication = float(np.mean(values > 0.0))
        expected_statistics = {
            "confidence_low": expected_low,
            "confidence_high": expected_high,
            "positive_fraction": expected_replication,
            "p_value": expected_p_value,
        }
        if any(
            not np.isclose(
                component[name], expected, rtol=1e-15, atol=0.0
            )
            for name, expected in expected_statistics.items()
        ):
            raise ValueError(
                "source-advantage component statistics are inconsistent"
            )
        for name in ("positive_fraction", "p_value"):
            if not 0.0 <= component[name] <= 1.0:
                raise ValueError(f"source-advantage component {name} is invalid")
        expected_component_gate = (
            component["confidence_low"] > 0.0
            and component["positive_fraction"]
            >= statistics["minimum_positive_fraction"]
        )
        if component["passes_direction_ci_replication"] is not (
            expected_component_gate
        ):
            raise ValueError(
                "source-advantage component direction/CI/replication gate "
                "is inconsistent"
            )
        components.append(component)
    result["components"] = components
    result["minimum_component_mean_effect"] = _finite_number(
        result["minimum_component_mean_effect"],
        name="minimum_component_mean_effect",
    )
    if result["minimum_component_mean_effect"] != min(
        component["mean_effect"] for component in components
    ):
        raise ValueError("source-advantage minimum component mean is inconsistent")
    result["composite_p_value"] = _finite_number(
        result["composite_p_value"], name="source composite_p_value"
    )
    if not 0.0 <= result["composite_p_value"] <= 1.0:
        raise ValueError("source composite_p_value must lie in [0, 1]")
    if result["composite_p_value"] != max(
        component["p_value"] for component in components
    ):
        raise ValueError(
            "source-advantage IUT composite is not max(component p)"
        )
    expected_direction_gates = all(
        component["passes_direction_ci_replication"]
        for component in components
    )
    if (
        result["direction_ci_replication_gates_passed"]
        is not expected_direction_gates
    ):
        raise ValueError(
            "source-advantage direction/CI/replication gate is inconsistent"
        )
    if result["p_value_gate_deferred_to_candidate_composite"] is not True:
        raise ValueError(
            "source-advantage p-value gate must be deferred to each candidate IUT"
        )
    return result


def _validated_selection_common_lineage(
    value: Any,
    *,
    context: Any,
    site: str,
    random_seed: int,
    representation_model_sha256: str,
    representation_parent_manifest_sha256: str,
    inference_contract_sha256: str,
    source_lineage: Mapping[str, Any],
    paired_source_binding: Mapping[str, Any] | None,
    checkpoint_study_sha256: str,
) -> dict[str, Any]:
    fields = {
        "model_family",
        "model_revision",
        "training_code_sha",
        "model_code_sha",
        "parent_analysis_code_sha",
        "checkpoint_sha256",
        "dataset_manifest_sha256",
        "condition",
        "site",
        "random_seed",
        "representation_model_sha256",
        "representation_parent_manifest_sha256",
        "inference_contract_sha256",
        "representation_source_lineage",
        "paired_source_direction",
        "paired_source_binding",
        "checkpoint_study_sha256",
    }
    result = _exact_object(value, label="selection common lineage", required=fields)
    expected = {
        "model_family": context.model_family,
        "model_revision": context.model_revision,
        "training_code_sha": context.inputs.training_code.head_sha,
        "model_code_sha": context.inputs.model_code.head_sha,
        "checkpoint_sha256": context.inputs.checkpoint.digest.sha256,
        "dataset_manifest_sha256": context.inputs.dataset_manifest.digest.sha256,
        "condition": context.condition,
        "site": site,
        "random_seed": random_seed,
        "representation_model_sha256": representation_model_sha256,
        "representation_parent_manifest_sha256": (
            representation_parent_manifest_sha256
        ),
        "inference_contract_sha256": inference_contract_sha256,
        "representation_source_lineage": _json_safe(source_lineage),
        "checkpoint_study_sha256": checkpoint_study_sha256,
    }
    mismatches = {
        name: (result.get(name), expected_value)
        for name, expected_value in expected.items()
        if result.get(name) != expected_value
    }
    if mismatches:
        raise ValueError(f"selection common lineage differs: {mismatches}")
    parent_analysis_sha = result["parent_analysis_code_sha"]
    if (
        not isinstance(parent_analysis_sha, str)
        or len(parent_analysis_sha) != 40
        or any(character not in "0123456789abcdef" for character in parent_analysis_sha)
    ):
        raise ValueError("selection parent analysis SHA is invalid")
    assert_git_commit_is_ancestor(
        context.inputs.analysis_code, parent_analysis_sha
    )
    if paired_source_binding is None:
        raise ValueError("selection common lineage requires a paired source")
    direction = _exact_object(
        result["paired_source_direction"],
        label="selection paired-source direction",
        required={"source_condition", "recipient_condition"},
    )
    expected_direction = {
        "source_condition": paired_source_binding.get("source_condition"),
        "recipient_condition": context.condition,
    }
    if direction != expected_direction or direction["source_condition"] == direction[
        "recipient_condition"
    ]:
        raise ValueError("selection paired-source direction is reversed or invalid")
    binding = _exact_object(
        result["paired_source_binding"],
        label="selection paired-source binding",
        required={
            "model_sha",
            "checkpoint_sha256",
            "code_attestation_sha256",
            "inference_contract_sha256",
            "representation_source_lineage_sha256",
        },
    )
    expected_binding = {
        name: paired_source_binding.get(name)
        for name in binding
    }
    if binding != expected_binding:
        raise ValueError("selection paired-source checkpoint/code binding differs")
    source_checkpoints = source_lineage.get("condition_checkpoints_sha256")
    if (
        not isinstance(source_checkpoints, Mapping)
        or source_checkpoints.get(direction["source_condition"])
        != binding["checkpoint_sha256"]
        or source_checkpoints.get(direction["recipient_condition"])
        != context.inputs.checkpoint.digest.sha256
    ):
        raise ValueError("selection source/recipient checkpoint direction differs")
    return result


def _validated_confirmation_protocol(
    value: Any,
    *,
    expected_random_seed: int,
    heldout_dataset_count: int,
    expected_maximum_candidates: int,
    expected_frozen_count: int,
) -> dict[str, Any]:
    result = _exact_object(
        value,
        label="held-out confirmation protocol",
        required={
            "alpha",
            "confidence_level",
            "sign_flip_resamples",
            "bootstrap_resamples",
            "bootstrap_method",
            "random_seed",
            "minimum_positive_fraction",
            "minimum_heldout_datasets",
            "required_heldout_datasets",
            "maximum_confirmation_candidates",
            "frozen_candidate_count",
            "preregistered_holm_rank_one_threshold",
            "actual_holm_rank_one_threshold",
            "sign_flip_minimum_p_value",
            "sign_flip_mode",
            "candidate_test",
            "multiplicity_method",
            "top_k_after_freeze",
        },
    )
    alpha = _finite_number(result["alpha"], name="confirmation alpha")
    confidence = _finite_number(
        result["confidence_level"], name="confirmation confidence_level"
    )
    replication = _finite_number(
        result["minimum_positive_fraction"],
        name="confirmation minimum_positive_fraction",
    )
    if not 0.0 < alpha <= 1.0 or not 0.0 < confidence < 1.0:
        raise ValueError("confirmation alpha/confidence is invalid")
    if not 0.0 <= replication <= 1.0:
        raise ValueError("confirmation replication fraction is invalid")
    sign_flip_resamples = _positive_integer(
        result["sign_flip_resamples"], name="confirmation sign_flip_resamples"
    )
    result["sign_flip_resamples"] = sign_flip_resamples
    result["bootstrap_resamples"] = _positive_integer(
        result["bootstrap_resamples"], name="confirmation bootstrap_resamples"
    )
    if result["bootstrap_method"] != "paired-dataset-bootstrap":
        raise ValueError("held-out bootstrap method is not canonical")
    if _non_negative_integer(
        result["random_seed"], name="confirmation random_seed"
    ) != expected_random_seed:
        raise ValueError("held-out confirmation random_seed differs from freeze")
    minimum_heldout_datasets = _positive_integer(
        result["minimum_heldout_datasets"],
        name="minimum_heldout_datasets",
    )
    result["minimum_heldout_datasets"] = minimum_heldout_datasets
    if minimum_heldout_datasets < 8:
        raise ValueError("confirmation requires at least eight held-out datasets")
    required_heldout_datasets = _positive_integer(
        result["required_heldout_datasets"],
        name="required_heldout_datasets",
    )
    maximum_candidates = _positive_integer(
        result["maximum_confirmation_candidates"],
        name="maximum_confirmation_candidates",
    )
    frozen_count = _non_negative_integer(
        result["frozen_candidate_count"], name="frozen_candidate_count"
    )
    expected_required = max(
        8, int(np.ceil(np.log2(maximum_candidates / alpha)))
    )
    if (
        maximum_candidates > 2
        or maximum_candidates != expected_maximum_candidates
        or frozen_count > maximum_candidates
        or frozen_count != expected_frozen_count
        or required_heldout_datasets != expected_required
    ):
        raise ValueError("confirmation candidate/count contract is invalid")
    preregistered_threshold = _finite_number(
        result["preregistered_holm_rank_one_threshold"],
        name="preregistered_holm_rank_one_threshold",
    )
    raw_actual_threshold = result["actual_holm_rank_one_threshold"]
    actual_threshold = (
        None
        if raw_actual_threshold is None
        else _finite_number(
            raw_actual_threshold, name="actual_holm_rank_one_threshold"
        )
    )
    expected_actual_threshold = (
        None if frozen_count == 0 else alpha / frozen_count
    )
    if not np.isclose(
        preregistered_threshold,
        alpha / maximum_candidates,
        rtol=1e-15,
        atol=0.0,
    ) or (
        (actual_threshold is None) != (expected_actual_threshold is None)
    ):
        raise ValueError("confirmation Holm thresholds are invalid")
    if (
        actual_threshold is not None
        and expected_actual_threshold is not None
        and not np.isclose(
            actual_threshold,
            expected_actual_threshold,
            rtol=1e-15,
            atol=0.0,
        )
    ):
        raise ValueError("confirmation actual Holm threshold is invalid")
    expected_sign_flip_mode = (
        "exact-enumeration"
        if heldout_dataset_count <= 20
        else "fixed-seed-monte-carlo"
    )
    minimum_p_value = _finite_number(
        result["sign_flip_minimum_p_value"],
        name="sign_flip_minimum_p_value",
    )
    expected_minimum_p_value = (
        2.0 ** (-heldout_dataset_count)
        if expected_sign_flip_mode == "exact-enumeration"
        else 1.0 / (sign_flip_resamples + 1)
    )
    if (
        result["sign_flip_mode"] != expected_sign_flip_mode
        or not np.isclose(
            minimum_p_value,
            expected_minimum_p_value,
            rtol=1e-15,
            atol=0.0,
        )
        or not 0.0 < minimum_p_value <= preregistered_threshold
    ):
        raise ValueError("confirmation sign-flip resolution is invalid")
    expected = {
        "candidate_test": "intersection-union-max-p",
        "multiplicity_method": "holm",
        "top_k_after_freeze": False,
    }
    if any(result[name] != expected_value for name, expected_value in expected.items()):
        raise ValueError("held-out confirmation protocol is not canonical")
    return result


def _selection_roster_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("held-out roster mapping must be a non-empty object")
    result: dict[str, str] = {}
    for raw_dataset_id, raw_digest in value.items():
        dataset_id = require_public_label(
            raw_dataset_id, name="held-out dataset_id"
        )
        result[dataset_id] = _required_sha256_value(
            raw_digest, name=f"held-out roster {dataset_id}"
        )
    if list(value) != sorted(result):
        raise ValueError("held-out roster mapping must be sorted")
    return dict(sorted(result.items()))


def _validated_dataset_fingerprint_mapping(
    value: Any,
    *,
    expected_dataset_ids: Sequence[str],
    label: str,
) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} dataset fingerprints must be an object")
    result: dict[str, dict[str, str]] = {}
    for raw_dataset_id, raw_fingerprints in value.items():
        dataset_id = require_public_label(
            raw_dataset_id, name=f"{label} dataset_id"
        )
        fingerprints = _exact_object(
            raw_fingerprints,
            label=f"{label} dataset fingerprints",
            required={
                "raw_dataset_content_sha256",
                "invariant_prediction_content_sha256",
                "combined_dataset_fingerprint_sha256",
            },
        )
        raw_digest = _required_sha256_value(
            fingerprints["raw_dataset_content_sha256"],
            name="raw_dataset_content_sha256",
        )
        prediction_digest = _required_sha256_value(
            fingerprints["invariant_prediction_content_sha256"],
            name="invariant_prediction_content_sha256",
        )
        combined_digest = _required_sha256_value(
            fingerprints["combined_dataset_fingerprint_sha256"],
            name="combined_dataset_fingerprint_sha256",
        )
        expected_combined = _canonical_sha256(
            {
                "raw_dataset_content_sha256": raw_digest,
                "invariant_prediction_content_sha256": prediction_digest,
            }
        )
        if combined_digest != expected_combined:
            raise ValueError(f"{label} combined dataset fingerprint differs")
        result[dataset_id] = {
            "raw_dataset_content_sha256": raw_digest,
            "invariant_prediction_content_sha256": prediction_digest,
            "combined_dataset_fingerprint_sha256": combined_digest,
        }
    if list(result) != list(expected_dataset_ids):
        raise ValueError(f"{label} dataset fingerprint roster differs")
    for name in (
        "raw_dataset_content_sha256",
        "invariant_prediction_content_sha256",
        "combined_dataset_fingerprint_sha256",
    ):
        digests = [fingerprints[name] for fingerprints in result.values()]
        if len(digests) != len(set(digests)):
            raise ValueError(f"{label} dataset fingerprints contain an alias")
    return result


def _selection_dataset_ids(value: Any, *, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = [require_public_label(item, name=name) for item in value]
    if result != sorted(set(result)):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _selection_feature_indices(value: Any, *, name: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in value
        )
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{name} must contain unique non-negative integers")
    return list(value)


def _validated_frozen_interventions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("freeze must contain at least one selected intervention")
    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_specs: set[str] = set()
    for raw in value:
        item = _exact_object(
            raw,
            label="frozen intervention",
            required={
                "candidate_id",
                "target_features",
                "control_features",
                "latent_baseline",
            },
        )
        candidate_id = require_portable_identifier(
            item["candidate_id"], name="candidate_id"
        )
        if candidate_id in seen_ids:
            raise ValueError("frozen candidate ids must be unique")
        seen_ids.add(candidate_id)
        targets = _selection_feature_indices(
            item["target_features"], name="frozen target_features"
        )
        controls = _selection_feature_indices(
            item["control_features"], name="frozen control_features"
        )
        if len(targets) != len(controls) or set(targets) & set(controls):
            raise ValueError("frozen target/control features are not matched")
        baseline = _frozen_numerical_baseline(item["latent_baseline"])
        normalized = {
            "candidate_id": candidate_id,
            "target_features": targets,
            "control_features": controls,
            "latent_baseline": baseline,
        }
        spec_digest = _canonical_sha256(normalized)
        if spec_digest in seen_specs:
            raise ValueError("frozen interventions must be unique")
        seen_specs.add(spec_digest)
        result.append(normalized)
    return result


def _validated_hypothesis(
    value: Any,
    *,
    expected_definition: Mapping[str, str],
    statistics: Mapping[str, Any],
    expected_count: int,
) -> dict[str, Any]:
    fields = {
        "name",
        "metric",
        "direction",
        "scope",
        "effective_dataset_count",
        "mean_effect",
        "median_effect",
        "confidence_low",
        "confidence_high",
        "positive_fraction",
        "p_value",
        "passes_direction_ci_replication",
    }
    result = _exact_object(value, label="candidate hypothesis", required=fields)
    if {name: result[name] for name in expected_definition} != expected_definition:
        raise ValueError("candidate evidence definition is not canonical")
    if _positive_integer(
        result["effective_dataset_count"], name="effective_dataset_count"
    ) != expected_count:
        raise ValueError("candidate hypothesis dataset count differs")
    for name in (
        "mean_effect",
        "median_effect",
        "confidence_low",
        "confidence_high",
        "positive_fraction",
        "p_value",
    ):
        result[name] = _finite_number(result[name], name=name)
    for name in ("positive_fraction", "p_value"):
        if not 0.0 <= result[name] <= 1.0:
            raise ValueError(f"candidate hypothesis {name} is outside [0, 1]")
    expected_gate = (
        result["confidence_low"] > 0.0
        and result["positive_fraction"]
        >= statistics["minimum_positive_fraction"]
    )
    if result["passes_direction_ci_replication"] is not expected_gate:
        raise ValueError(
            "candidate hypothesis direction/CI/replication gate is inconsistent"
        )
    return result


def _validate_selected_candidate_results(
    value: Any,
    *,
    selected_ids: Sequence[str],
    statistics: Mapping[str, Any],
    source_advantage: Mapping[str, Any],
    validation_dataset_ids: Sequence[str],
    frozen_interventions: Sequence[Mapping[str, Any]],
    random_seed: int,
) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("selection candidate_results must be non-empty")
    candidate_fields = {
        "candidate_id",
        "target_features",
        "control_features",
        "latent_baseline",
        "effective_dataset_count",
        "dataset_effects",
        "hypotheses",
        "minimum_mean_evidence",
        "intersection_union_composite_p_value",
        "by_adjusted_composite_p_value",
        "all_direction_ci_replication_gates_passed",
        "eligible",
        "selected",
    }
    dataset_fields = {
        "dataset_id",
        "target_damage",
        "matched_control_damage",
        "ablation_specificity",
        "donor_rescue",
        "donor_specificity",
        "donor_shift_balance_ratio",
    }
    candidate_definitions = _SELECTION_EVIDENCE_FAMILY[:-1]
    normalized_selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    candidate_records: list[dict[str, Any]] = []
    for candidate_offset, raw in enumerate(value):
        item = _exact_object(
            raw, label="selection candidate result", required=candidate_fields
        )
        candidate_id = require_portable_identifier(
            item["candidate_id"], name="candidate_id"
        )
        if candidate_id in seen_ids:
            raise ValueError("candidate result ids must be unique")
        seen_ids.add(candidate_id)
        targets = _selection_feature_indices(
            item["target_features"], name="candidate target_features"
        )
        controls = _selection_feature_indices(
            item["control_features"], name="candidate control_features"
        )
        if len(targets) != len(controls) or set(targets) & set(controls):
            raise ValueError("candidate target/control features are not matched")
        baseline = _frozen_numerical_baseline(item["latent_baseline"])
        count = _positive_integer(
            item["effective_dataset_count"], name="effective_dataset_count"
        )
        if count != len(validation_dataset_ids):
            raise ValueError("candidate validation dataset count differs")
        raw_effects = item["dataset_effects"]
        if not isinstance(raw_effects, list) or len(raw_effects) != count:
            raise ValueError("candidate dataset effects are incomplete")
        effect_ids: list[str] = []
        normalized_effects: list[dict[str, Any]] = []
        for raw_effect in raw_effects:
            effect = _exact_object(
                raw_effect,
                label="candidate dataset effect",
                required=dataset_fields,
            )
            effect_id = require_public_label(
                effect["dataset_id"], name="validation dataset_id"
            )
            effect_ids.append(effect_id)
            normalized_effect = {"dataset_id": effect_id}
            for name in dataset_fields - {"dataset_id"}:
                number = _finite_number(effect[name], name=name)
                normalized_effect[name] = number
                if name == "donor_shift_balance_ratio" and not (
                    1.0
                    <= number
                    <= _MAX_FORMAL_SYMMETRIC_DONOR_SHIFT_RMS_RATIO
                ):
                    raise ValueError("candidate donor shift ratio is unsafe")
            if not np.isclose(
                normalized_effect["ablation_specificity"],
                normalized_effect["target_damage"]
                - normalized_effect["matched_control_damage"],
                rtol=1e-15,
                atol=0.0,
            ):
                raise ValueError("candidate ablation specificity is inconsistent")
            normalized_effects.append(normalized_effect)
        if effect_ids != list(validation_dataset_ids):
            raise ValueError("candidate validation dataset roster differs")
        hypotheses = item["hypotheses"]
        if not isinstance(hypotheses, list) or len(hypotheses) != len(
            candidate_definitions
        ):
            raise ValueError("candidate mechanism evidence family is incomplete")
        normalized_hypotheses = [
            _validated_hypothesis(
                hypothesis,
                expected_definition=definition,
                statistics=statistics,
                expected_count=count,
            )
            for hypothesis, definition in zip(
                hypotheses, candidate_definitions, strict=True
            )
        ]
        evidence_fields = (
            "target_damage",
            "ablation_specificity",
            "donor_rescue",
            "donor_specificity",
        )
        for evidence_offset, (hypothesis, effect_field) in enumerate(
            zip(normalized_hypotheses, evidence_fields, strict=True)
        ):
            observed = np.asarray(
                [effect[effect_field] for effect in normalized_effects],
                dtype=np.float64,
            )
            hypothesis_offset = (
                candidate_offset * len(candidate_definitions) + evidence_offset
            )
            expected_low, expected_high = paired_bootstrap_ci(
                observed,
                confidence=statistics["confidence_level"],
                n_resamples=statistics["bootstrap_resamples"],
                seed=random_seed + hypothesis_offset,
            )
            expected_statistics = {
                "mean_effect": float(np.mean(observed)),
                "median_effect": float(np.median(observed)),
                "confidence_low": expected_low,
                "confidence_high": expected_high,
                "positive_fraction": float(np.mean(observed > 0.0)),
                "p_value": paired_sign_flip_p_value(
                    observed,
                    n_resamples=statistics["sign_flip_resamples"],
                    seed=random_seed + 100_000 + hypothesis_offset,
                ),
            }
            if any(
                not np.isclose(
                    hypothesis[name], expected, rtol=1e-15, atol=0.0
                )
                for name, expected in expected_statistics.items()
            ):
                raise ValueError("candidate hypothesis statistics are inconsistent")
        minimum_mean = _finite_number(
            item["minimum_mean_evidence"], name="minimum_mean_evidence"
        )
        if minimum_mean != min(
            hypothesis["mean_effect"] for hypothesis in normalized_hypotheses
        ):
            raise ValueError("candidate minimum evidence is inconsistent")
        composite_p_value = _finite_number(
            item["intersection_union_composite_p_value"],
            name="intersection_union_composite_p_value",
        )
        expected_composite = max(
            [hypothesis["p_value"] for hypothesis in normalized_hypotheses]
            + [
                component["p_value"]
                for component in source_advantage["components"]
            ]
        )
        if (
            not 0.0 <= composite_p_value <= 1.0
            or composite_p_value != expected_composite
        ):
            raise ValueError(
                "candidate IUT composite is not max of all six component p-values"
            )
        adjusted_composite = _finite_number(
            item["by_adjusted_composite_p_value"],
            name="by_adjusted_composite_p_value",
        )
        if not 0.0 <= adjusted_composite <= 1.0:
            raise ValueError("candidate BY-adjusted composite is outside [0, 1]")
        direction_gates = all(
            hypothesis["passes_direction_ci_replication"]
            for hypothesis in normalized_hypotheses
        ) and source_advantage["direction_ci_replication_gates_passed"]
        if item["all_direction_ci_replication_gates_passed"] is not (
            direction_gates
        ):
            raise ValueError(
                "candidate aggregate direction/CI/replication gate is inconsistent"
            )
        if not isinstance(item["eligible"], bool):
            raise TypeError("candidate eligible flag must be a boolean")
        if not isinstance(item["selected"], bool):
            raise TypeError("candidate selected flag must be a boolean")
        candidate_records.append(
            {
                "item": item,
                "candidate_id": candidate_id,
                "minimum_mean_evidence": minimum_mean,
                "composite_p_value": composite_p_value,
                "adjusted_composite_p_value": adjusted_composite,
                "direction_gates": direction_gates,
                "frozen_specification": {
                    "candidate_id": candidate_id,
                    "target_features": targets,
                    "control_features": controls,
                    "latent_baseline": baseline,
                },
            }
        )
    if [record["candidate_id"] for record in candidate_records] != sorted(
        seen_ids
    ):
        raise ValueError("candidate results must be sorted by candidate_id")
    expected_family_size = len(value)
    if statistics["family_size"] != expected_family_size:
        raise ValueError("selection evidence family size is inconsistent")
    expected_adjusted = adjust_fdr_arbitrary_dependence(
        record["composite_p_value"] for record in candidate_records
    )
    eligible_records: list[dict[str, Any]] = []
    for record, expected_adjusted_p in zip(
        candidate_records, expected_adjusted, strict=True
    ):
        if not np.isclose(
            record["adjusted_composite_p_value"],
            expected_adjusted_p,
            rtol=1e-15,
            atol=0.0,
        ):
            raise ValueError("candidate BY-adjusted composite is inconsistent")
        expected_eligible = bool(
            record["direction_gates"]
            and expected_adjusted_p <= statistics["fdr_alpha"]
        )
        if record["item"]["eligible"] is not expected_eligible:
            raise ValueError("candidate eligibility is inconsistent")
        if expected_eligible:
            eligible_records.append(record)
    eligible_records.sort(
        key=lambda record: (
            -record["minimum_mean_evidence"],
            record["candidate_id"],
        )
    )
    expected_selected_set = {
        record["candidate_id"]
        for record in eligible_records[: statistics["maximum_selections"]]
    }
    selected_from_results: list[str] = []
    for record in candidate_records:
        expected_selected = record["candidate_id"] in expected_selected_set
        if record["item"]["selected"] is not expected_selected:
            raise ValueError("candidate selection/ranking flag is inconsistent")
        if expected_selected:
            selected_from_results.append(record["candidate_id"])
            normalized_selected.append(record["frozen_specification"])
    if selected_from_results != list(selected_ids):
        raise ValueError("selected candidate ids differ from candidate results")
    if normalized_selected != list(frozen_interventions):
        raise ValueError("frozen interventions differ from selected candidates")


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
    paired_effects = None
    if (
        evaluation.paired_reverse_patch_log_loss_improvement_vs_no_op
        is not None
    ):
        assert (
            evaluation.paired_reverse_patch_log_loss_improvement_vs_matched_random
            is not None
        )
        assert (
            evaluation.paired_reverse_patch_log_loss_improvement_vs_target_baseline
            is not None
        )
        assert (
            evaluation.paired_reverse_patch_native_distance_reduction_vs_target_baseline
            is not None
        )
        assert (
            evaluation.paired_source_native_log_loss_improvement_vs_recipient_native
            is not None
        )
        assert (
            evaluation.paired_source_no_op_log_loss_improvement_vs_recipient_no_op
            is not None
        )
        paired_effects = {
            "log_loss_improvement_vs_recipient_no_op": (
                evaluation.paired_reverse_patch_log_loss_improvement_vs_no_op
            ),
            "log_loss_improvement_vs_paired_matched_random_patch": (
                evaluation.paired_reverse_patch_log_loss_improvement_vs_matched_random
            ),
            "log_loss_improvement_vs_target_baseline_edit": (
                evaluation.paired_reverse_patch_log_loss_improvement_vs_target_baseline
            ),
            "native_distance_reduction_vs_target_baseline_edit": (
                evaluation.paired_reverse_patch_native_distance_reduction_vs_target_baseline
            ),
            "log_loss_improvement_of_source_native_vs_recipient_native": (
                evaluation.paired_source_native_log_loss_improvement_vs_recipient_native
            ),
            "log_loss_improvement_of_source_no_op_vs_recipient_no_op": (
                evaluation.paired_source_no_op_log_loss_improvement_vs_recipient_no_op
            ),
        }
    paired_source_native = None
    if evaluation.paired_source_native_prediction is not None:
        source_native = evaluation.paired_source_native_prediction
        assert evaluation.paired_source_no_op_prediction is not None
        paired_source_native = {
            **_prediction_values(source_native.probabilities, classes),
            "true_class_log_loss": source_native.true_class_log_loss,
            "no_op_probabilities": (
                evaluation.paired_source_no_op_prediction.probabilities
            ),
            "no_op_true_class_log_loss": (
                evaluation.paired_source_no_op_prediction.true_class_log_loss
            ),
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
            "paired_reverse_patch_measured_effects": paired_effects,
            "paired_source_native": paired_source_native,
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
    paired_source_binding: Mapping[str, Any] | None,
    ranking_parent_binding: Mapping[str, Any] | None,
    checkpoint_study: Mapping[str, Any],
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
    paired_measured_effects = None
    if (
        evaluation.paired_reverse_patch_log_loss_improvement_vs_no_op
        is not None
    ):
        assert (
            evaluation.paired_reverse_patch_log_loss_improvement_vs_matched_random
            is not None
        )
        assert (
            evaluation.paired_reverse_patch_log_loss_improvement_vs_target_baseline
            is not None
        )
        assert (
            evaluation.paired_reverse_patch_native_distance_reduction_vs_target_baseline
            is not None
        )
        assert (
            evaluation.paired_source_native_log_loss_improvement_vs_recipient_native
            is not None
        )
        assert (
            evaluation.paired_source_no_op_log_loss_improvement_vs_recipient_no_op
            is not None
        )
        mean_source_native_advantage = float(
            np.mean(
                evaluation.paired_source_native_log_loss_improvement_vs_recipient_native
            )
        )
        mean_source_no_op_advantage = float(
            np.mean(
                evaluation.paired_source_no_op_log_loss_improvement_vs_recipient_no_op
            )
        )
        mean_gap_closure = float(
            np.mean(
                evaluation.paired_reverse_patch_log_loss_improvement_vs_no_op
            )
        )
        paired_measured_effects = {
            "mean_log_loss_improvement_vs_recipient_no_op": float(
                np.mean(
                    evaluation.paired_reverse_patch_log_loss_improvement_vs_no_op
                )
            ),
            "mean_log_loss_improvement_vs_paired_matched_random_patch": float(
                np.mean(
                    evaluation.paired_reverse_patch_log_loss_improvement_vs_matched_random
                )
            ),
            "mean_log_loss_improvement_vs_target_baseline_edit": float(
                np.mean(
                    evaluation.paired_reverse_patch_log_loss_improvement_vs_target_baseline
                )
            ),
            "mean_native_distance_reduction_vs_target_baseline_edit": float(
                np.mean(
                    evaluation.paired_reverse_patch_native_distance_reduction_vs_target_baseline
                )
            ),
            "mean_log_loss_improvement_of_source_native_vs_recipient_native": (
                mean_source_native_advantage
            ),
            "mean_log_loss_improvement_of_source_no_op_vs_recipient_no_op": (
                mean_source_no_op_advantage
            ),
            "donor_patch_gap_closure_denominator": (
                "source_no_op_vs_recipient_no_op"
            ),
            "donor_patch_gap_closure_fraction": (
                None
                if mean_source_no_op_advantage <= 0.0
                else mean_gap_closure / mean_source_no_op_advantage
            ),
        }
    paired_direction = (
        None
        if paired_source_binding is None
        else {
            "source_condition": paired_source_binding["source_condition"],
            "recipient_condition": paired_source_binding[
                "recipient_condition"
            ],
        }
    )
    paired_binding_payload = (
        None
        if paired_source_binding is None
        else {
            "model_sha": paired_source_binding["model_sha"],
            "checkpoint_sha256": paired_source_binding["checkpoint_sha256"],
            "code_attestation_sha256": paired_source_binding[
                "code_attestation_sha256"
            ],
            "inference_contract_sha256": paired_source_binding[
                "inference_contract_sha256"
            ],
            "representation_source_lineage_sha256": paired_source_binding[
                "representation_source_lineage_sha256"
            ],
        }
    )
    sample_roster_sha256 = evaluation.representation_qualification[
        "sample_roster_sha256"
    ]
    paired_summary = {
        "status": (
            "not_run"
            if paired_source_binding is None
            else (
                "measured_exploratory_diagnostic_only"
                if checkpoint_study["scope"] == "exploratory_pilot"
                else "measured_diagnostic_only"
            )
        ),
        "evidence_scope": "single_dataset_measurement",
        "direction": paired_direction,
        "condition_names": {
            "source_target_feature_patch": "paired_reverse_patch",
            "source_control_feature_patch": "paired_matched_random_patch",
        },
        "alignment": {
            "verified": evaluation.paired_alignment_verified,
            "sample_roster_sha256": sample_roster_sha256,
            "inference_contract_sha256": inference_contract_sha256,
        },
        "source_binding": paired_binding_payload,
        "measured_effects": paired_measured_effects,
        "donor_shift_balance": evaluation.paired_donor_shift_balance,
        "ablation_displacement_balance": (
            evaluation.paired_ablation_displacement_balance
        ),
        "donor_displacement_balance": (
            evaluation.paired_donor_displacement_balance
        ),
        "source_native": (
            None
            if evaluation.paired_source_native_prediction is None
            else {
                "accuracy": evaluation.paired_source_native_prediction.accuracy,
                "log_loss": evaluation.paired_source_native_prediction.log_loss,
            }
        ),
        "source_no_op_gates": (
            None
            if evaluation.paired_source_no_op_prediction is None
            else {
                "passed": True,
                "reconstruction_mse": (
                    evaluation.paired_source_no_op_reconstruction_mse
                ),
                "absolute_accuracy_difference": (
                    evaluation.paired_source_no_op_accuracy_absolute_difference
                ),
                "maximum_absolute_probability_difference": (
                    evaluation.paired_source_no_op_probability_max_abs_difference
                ),
                "thresholds": dict(thresholds),
            }
        ),
    }
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
                "status": (
                    "paired_exploratory_diagnostic_only"
                    if paired_source_binding is not None
                    and checkpoint_study["scope"] == "exploratory_pilot"
                    else evaluation.mechanistic_rescue_status
                ),
                "roundtrip_restore_control_passed": True,
            },
            "paired_reverse_patch": paired_summary,
            "ranking_parent": ranking_parent_binding,
            "checkpoint_study": checkpoint_study,
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
                "ranking_parent_manifest_sha256": (
                    None
                    if ranking_parent_binding is None
                    else ranking_parent_binding["manifest_sha256"]
                ),
                "ranking_selection_sha256": (
                    None
                    if ranking_parent_binding is None
                    else ranking_parent_binding["selection_sha256"]
                ),
                "ranking_split_protocol_sha256": (
                    None
                    if ranking_parent_binding is None
                    else ranking_parent_binding["split_protocol_sha256"]
                ),
                "checkpoint_study_sha256": checkpoint_study[
                    "binding_sha256"
                ],
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
