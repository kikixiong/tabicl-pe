from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import types

import numpy as np
import pytest
import torch
from torch import nn

import pe_mechanism.official_causal as official_causal
from pe_mechanism.adapters.base import ActivationRecord
from pe_mechanism.adapters.tabicl import TabICLAdapter
from pe_mechanism.causal import (
    model_decoder_feature_norms,
    raw_space_decoder_feature_norms,
)
from pe_mechanism.manifest import (
    ArtifactDigest,
    FileDigest,
    InputDigest,
    new_manifest,
)
from pe_mechanism.official_causal import (
    PairedReversePatchSource,
    run_official_tabicl_causal_edits,
)
from pe_mechanism.official_tabicl import (
    OfficialTabICLDriver,
    official_inference_contract_sha256,
)
from pe_mechanism.provenance import (
    VerifiedRunContext,
    load_verified_run_manifest,
    verify_file,
    verify_git_tree,
    verify_run_directory,
    verify_run_inputs,
)
from pe_mechanism.representation import (
    DenseAutoencoder,
    MeanRMSNormalizer,
    PCARepresentation,
    TopKSparseAutoencoder,
)
from test_official_tabicl import FakeOfficialClassifier, make_driver


class ExactOvercompleteAutoencoder(nn.Module):
    input_dim = 1
    latent_dim = 3

    def __init__(self, *, negate_decode: bool = False) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, 3, bias=False)
        self.decoder = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.tensor([[1.0], [1.0], [-0.25]]))
            sign = -1.0 if negate_decode else 1.0
            self.decoder.weight.copy_(torch.tensor([[0.5 * sign, 0.5 * sign, 0.0]]))

    def encode(self, values):
        return self.encoder(values)

    def decode(self, latents):
        return self.decoder(latents)


class ImbalancedExactAutoencoder(nn.Module):
    input_dim = 1
    latent_dim = 3

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, 3, bias=False)
        self.decoder = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.tensor([[8.0], [1.0], [0.0]]))
            self.decoder.weight.copy_(torch.tensor([[1.0 / 9.0, 1.0 / 9.0, 0.0]]))

    def encode(self, values):
        return self.encoder(values)

    def decode(self, latents):
        return self.decoder(latents)


@pytest.fixture
def causal_inputs():
    X = np.array(
        [[0.1, 0.6, -0.3, 0.8], [0.9, -0.2, 0.4, 0.3]],
        dtype=np.float32,
    )
    y = np.array([0, 1])
    return X, y


def qualification(passed=True):
    return {
        "metric_split": "validation",
        "activation_fidelity_passed": passed,
        "held_out_explained_variance": 0.99,
        "model_sha": "a" * 40,
        "checkpoint_sha": "b" * 64,
        "site": "row_interactor",
    }


def test_official_causal_runner_preserves_ensemble_and_rng(causal_inputs):
    X, y = causal_inputs
    classifier = FakeOfficialClassifier(temporary=True)
    driver = make_driver(classifier)
    generator = classifier.model_.row_interactor._identity_generator
    entry_state = generator.get_state().clone()
    direct = classifier.predict_proba(X.copy())
    expected_final_state = generator.get_state().clone()
    generator.set_state(entry_state)

    evaluation = run_official_tabicl_causal_edits(
        driver,
        X,
        y,
        site="row_interactor",
        autoencoder=ExactOvercompleteAutoencoder(),
        normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
        target_features=(0,),
        dataset_id="toy-official",
        sample_ids=("row-0", "row-1"),
        representation_qualification=qualification(),
        random_candidate_pool_size=2,
    )

    assert evaluation.capture_exact_to_direct
    assert np.array_equal(evaluation.native_prediction.probabilities, direct)
    assert torch.equal(generator.get_state(), expected_final_state)
    assert [
        call.metadata.raw_input_shape[0]
        for call in evaluation.native_forward_calls
    ] == [2, 1, 2]
    assert set(evaluation.conditions) == {
        "no_op_reconstruction",
        "target_baseline_edit",
        "matched_random_edit",
        "roundtrip_restore_control",
    }
    assert evaluation.no_op_reconstruction_mse == 0.0
    assert evaluation.no_op_accuracy_drop == 0.0
    assert np.array_equal(
        evaluation.conditions["no_op_reconstruction"].prediction.probabilities,
        evaluation.native_prediction.probabilities,
    )
    assert not np.array_equal(
        evaluation.conditions["target_baseline_edit"].prediction.probabilities,
        evaluation.native_prediction.probabilities,
    )
    assert np.array_equal(
        evaluation.conditions["roundtrip_restore_control"].prediction.probabilities,
        evaluation.conditions["no_op_reconstruction"].prediction.probabilities,
    )
    assert evaluation.mechanistic_rescue_passed is False
    assert evaluation.mechanistic_rescue_status == "paired_rescue_not_run"
    for condition in evaluation.conditions.values():
        assert condition.delta_log_loss_vs_model_baseline.shape == (2,)
        assert condition.delta_log_loss_vs_reconstruction.shape == (2,)
    assert len(evaluation.matched_control_features) == 1


def _paired_driver(
    classifier: FakeOfficialClassifier, *, checkpoint_sha: str
) -> OfficialTabICLDriver:
    return OfficialTabICLDriver(
        classifier,
        adapter=TabICLAdapter(),
        model_sha="a" * 40,
        checkpoint_sha=checkpoint_sha,
    )


def _paired_qualification() -> dict[str, object]:
    result: dict[str, object] = qualification()
    result["condition_checkpoints_sha256"] = {
        "temporary": "b" * 64,
        "none": "c" * 64,
    }
    return result


def test_paired_reverse_patch_uses_independent_source_and_preserves_both_rngs(
    causal_inputs,
) -> None:
    X, y = causal_inputs
    target_classifier = FakeOfficialClassifier(temporary=True)
    source_classifier = FakeOfficialClassifier(temporary=False)
    target = _paired_driver(target_classifier, checkpoint_sha="b" * 64)
    source = _paired_driver(source_classifier, checkpoint_sha="c" * 64)
    target_generator = target_classifier.model_.row_interactor._identity_generator
    source_generator = source_classifier.model_.row_interactor._identity_generator
    target_entry = target_generator.get_state().clone()
    source_entry = source_generator.get_state().clone()
    target_classifier.predict_proba(X.copy())
    target_final = target_generator.get_state().clone()
    source_classifier.predict_proba(X.copy())
    source_final = source_generator.get_state().clone()
    target_generator.set_state(target_entry)
    source_generator.set_state(source_entry)

    evaluation = run_official_tabicl_causal_edits(
        target,
        X,
        y,
        site="row_interactor",
        autoencoder=ExactOvercompleteAutoencoder(),
        normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
        target_features=(0,),
        dataset_id="toy-official",
        sample_ids=("row-0", "row-1"),
        representation_qualification=_paired_qualification(),
        random_candidate_pool_size=2,
        control_features=(1,),
        target_condition="temporary",
        paired_source=PairedReversePatchSource(source, "none"),
        maximum_symmetric_donor_shift_rms_ratio=1.25,
    )

    assert "paired_reverse_patch" in evaluation.conditions
    assert "paired_matched_random_patch" in evaluation.conditions
    assert evaluation.conditions["paired_reverse_patch"].reference_scope == "primary"
    assert evaluation.paired_source_condition == "none"
    assert evaluation.paired_source_checkpoint_sha == "c" * 64
    assert evaluation.paired_alignment_verified
    assert evaluation.mechanistic_rescue_passed is False
    assert evaluation.mechanistic_rescue_status == (
        "paired_reverse_patch_completed_requires_paired_statistics"
    )
    assert (
        evaluation.paired_reverse_patch_log_loss_improvement_vs_matched_random
        is not None
    )
    assert (
        evaluation.paired_reverse_patch_log_loss_improvement_vs_matched_random.shape
        == (2,)
    )
    assert (
        evaluation.paired_reverse_patch_log_loss_improvement_vs_target_baseline
        is not None
    )
    assert (
        evaluation.paired_reverse_patch_log_loss_improvement_vs_no_op
        is not None
    )
    assert (
        evaluation.paired_reverse_patch_native_distance_reduction_vs_target_baseline
        is not None
    )
    assert evaluation.paired_source_native_prediction is not None
    assert evaluation.paired_source_no_op_prediction is not None
    assert (
        evaluation.paired_source_native_log_loss_improvement_vs_recipient_native
        is not None
    )
    assert (
        evaluation.paired_source_no_op_log_loss_improvement_vs_recipient_no_op
        is not None
    )
    assert evaluation.paired_donor_shift_balance is not None
    assert evaluation.paired_donor_shift_balance["passed"] is True
    assert evaluation.paired_ablation_displacement_balance is not None
    assert evaluation.paired_ablation_displacement_balance["passed"] is True
    assert evaluation.paired_donor_displacement_balance is not None
    assert evaluation.paired_donor_displacement_balance["passed"] is True
    assert torch.equal(target_generator.get_state(), target_final)
    assert torch.equal(source_generator.get_state(), source_final)


@pytest.mark.parametrize(
    ("source_condition", "source_checkpoint", "lineage_checkpoint", "message"),
    [
        ("temporary", "c" * 64, "c" * 64, "must differ"),
        ("none", "b" * 64, "b" * 64, "checkpoints must differ"),
        ("none", "c" * 64, "d" * 64, "not bound"),
    ],
)
def test_paired_reverse_patch_rejects_non_independent_or_unbound_source(
    causal_inputs,
    source_condition: str,
    source_checkpoint: str,
    lineage_checkpoint: str,
    message: str,
) -> None:
    X, y = causal_inputs
    target = _paired_driver(
        FakeOfficialClassifier(temporary=True), checkpoint_sha="b" * 64
    )
    source = _paired_driver(
        FakeOfficialClassifier(), checkpoint_sha=source_checkpoint
    )
    evidence = _paired_qualification()
    evidence["condition_checkpoints_sha256"] = {
        "temporary": "b" * 64,
        "none": lineage_checkpoint,
    }

    with pytest.raises((ValueError, RuntimeError), match=message):
        run_official_tabicl_causal_edits(
            target,
            X,
            y,
            site="row_interactor",
            autoencoder=ExactOvercompleteAutoencoder(),
            normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
            target_features=(0,),
            dataset_id="toy-official",
            sample_ids=(0, 1),
            representation_qualification=evidence,
            target_condition="temporary",
            paired_source=PairedReversePatchSource(
                source, source_condition
            ),
        )


def test_paired_reverse_patch_schedule_drift_fails_and_rolls_back_both_rngs(
    causal_inputs,
) -> None:
    X, y = causal_inputs
    target_classifier = FakeOfficialClassifier(temporary=True)
    source_classifier = FakeOfficialClassifier(temporary=False)
    changed_shuffle = [
        1,
        0,
        2,
        3,
    ]
    source_classifier.ensemble_generator_.feature_shuffles_["none"][0] = (
        changed_shuffle
    )
    _, class_shuffle = (
        source_classifier.ensemble_generator_.ensemble_configs_["none"][0]
    )
    source_classifier.ensemble_generator_.ensemble_configs_["none"][0] = (
        changed_shuffle,
        class_shuffle,
    )
    target_generator = target_classifier.model_.row_interactor._identity_generator
    source_generator = source_classifier.model_.row_interactor._identity_generator
    target_entry = target_generator.get_state().clone()
    source_entry = source_generator.get_state().clone()

    with pytest.raises(RuntimeError, match="schedules/coordinates differ"):
        run_official_tabicl_causal_edits(
            _paired_driver(target_classifier, checkpoint_sha="b" * 64),
            X,
            y,
            site="row_interactor",
            autoencoder=ExactOvercompleteAutoencoder(),
            normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
            target_features=(0,),
            dataset_id="toy-official",
            sample_ids=(0, 1),
            representation_qualification=_paired_qualification(),
            control_features=(1,),
            target_condition="temporary",
            paired_source=PairedReversePatchSource(
                _paired_driver(source_classifier, checkpoint_sha="c" * 64),
                "none",
            ),
            maximum_symmetric_donor_shift_rms_ratio=1.25,
        )

    assert torch.equal(target_generator.get_state(), target_entry)
    assert torch.equal(source_generator.get_state(), source_entry)


def test_paired_reverse_patch_is_restricted_to_coordinate_stable_site(
    causal_inputs,
) -> None:
    X, y = causal_inputs
    target = _paired_driver(
        FakeOfficialClassifier(temporary=True), checkpoint_sha="b" * 64
    )
    source = _paired_driver(
        FakeOfficialClassifier(temporary=False), checkpoint_sha="c" * 64
    )
    qualification = _paired_qualification()
    qualification["site"] = "deeper_token_site"

    with pytest.raises(ValueError, match="restricted to row_interactor"):
        run_official_tabicl_causal_edits(
            target,
            X,
            y,
            site="deeper_token_site",
            autoencoder=ExactOvercompleteAutoencoder(),
            normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
            target_features=(0,),
            control_features=(1,),
            dataset_id="toy-official",
            sample_ids=(0, 1),
            representation_qualification=qualification,
            target_condition="temporary",
            paired_source=PairedReversePatchSource(source, "none"),
            maximum_symmetric_donor_shift_rms_ratio=1.25,
        )


def test_paired_reverse_patch_rejects_class_roster_drift_and_rolls_back(
    causal_inputs,
) -> None:
    X, y = causal_inputs
    target_classifier = FakeOfficialClassifier(temporary=True)
    source_classifier = FakeOfficialClassifier(temporary=False)
    source_classifier.classes_ = np.array([1, 0])
    target_generator = target_classifier.model_.row_interactor._identity_generator
    source_generator = source_classifier.model_.row_interactor._identity_generator
    target_entry = target_generator.get_state().clone()
    source_entry = source_generator.get_state().clone()

    with pytest.raises(RuntimeError, match="class rosters differ"):
        run_official_tabicl_causal_edits(
            _paired_driver(target_classifier, checkpoint_sha="b" * 64),
            X,
            y,
            site="row_interactor",
            autoencoder=ExactOvercompleteAutoencoder(),
            normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
            target_features=(0,),
            dataset_id="toy-official",
            sample_ids=(0, 1),
            representation_qualification=_paired_qualification(),
            control_features=(1,),
            target_condition="temporary",
            paired_source=PairedReversePatchSource(
                _paired_driver(source_classifier, checkpoint_sha="c" * 64),
                "none",
            ),
            maximum_symmetric_donor_shift_rms_ratio=1.25,
        )

    assert torch.equal(target_generator.get_state(), target_entry)
    assert torch.equal(source_generator.get_state(), source_entry)


def test_paired_reverse_patch_requires_frozen_shift_matched_controls(
    causal_inputs,
) -> None:
    X, y = causal_inputs
    target_classifier = FakeOfficialClassifier(temporary=True)
    source_classifier = FakeOfficialClassifier(temporary=False)
    target = _paired_driver(target_classifier, checkpoint_sha="b" * 64)
    source = _paired_driver(source_classifier, checkpoint_sha="c" * 64)
    target_generator = target_classifier.model_.row_interactor._identity_generator
    source_generator = source_classifier.model_.row_interactor._identity_generator
    target_entry = target_generator.get_state().clone()
    source_entry = source_generator.get_state().clone()

    with pytest.raises(ValueError, match="explicit frozen control_features"):
        run_official_tabicl_causal_edits(
            target,
            X,
            y,
            site="row_interactor",
            autoencoder=ExactOvercompleteAutoencoder(),
            normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
            target_features=(0,),
            dataset_id="toy-official",
            sample_ids=(0, 1),
            representation_qualification=_paired_qualification(),
            target_condition="temporary",
            paired_source=PairedReversePatchSource(source, "none"),
            maximum_symmetric_donor_shift_rms_ratio=1.25,
        )

    with pytest.raises(RuntimeError, match="decoded target/control displacement"):
        run_official_tabicl_causal_edits(
            target,
            X,
            y,
            site="row_interactor",
            autoencoder=ExactOvercompleteAutoencoder(),
            normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
            target_features=(0,),
            control_features=(2,),
            dataset_id="toy-official",
            sample_ids=(0, 1),
            representation_qualification=_paired_qualification(),
            target_condition="temporary",
            paired_source=PairedReversePatchSource(source, "none"),
            maximum_symmetric_donor_shift_rms_ratio=1.25,
        )
    assert torch.equal(target_generator.get_state(), target_entry)
    assert torch.equal(source_generator.get_state(), source_entry)


def test_paired_reverse_patch_can_use_shrink_only_decoded_dose_matching(
    causal_inputs,
) -> None:
    X, y = causal_inputs
    target_classifier = FakeOfficialClassifier(temporary=True)
    source_classifier = FakeOfficialClassifier(temporary=False)
    target = _paired_driver(target_classifier, checkpoint_sha="b" * 64)
    source = _paired_driver(source_classifier, checkpoint_sha="c" * 64)
    target_generator = target_classifier.model_.row_interactor._identity_generator
    source_generator = source_classifier.model_.row_interactor._identity_generator
    target_entry = target_generator.get_state().clone()
    source_entry = source_generator.get_state().clone()
    target_classifier.predict_proba(X.copy())
    target_final = target_generator.get_state().clone()
    source_classifier.predict_proba(X.copy())
    source_final = source_generator.get_state().clone()
    target_generator.set_state(target_entry)
    source_generator.set_state(source_entry)

    evaluation = run_official_tabicl_causal_edits(
        target,
        X,
        y,
        site="row_interactor",
        autoencoder=ImbalancedExactAutoencoder(),
        normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
        target_features=(0,),
        control_features=(1,),
        dataset_id="toy-official",
        sample_ids=("row-0", "row-1"),
        representation_qualification=_paired_qualification(),
        target_condition="temporary",
        paired_source=PairedReversePatchSource(source, "none"),
        maximum_symmetric_donor_shift_rms_ratio=1.25,
        matched_control_dose=(
            "per_call_decoded_rms_clip_to_smaller_without_amplification"
        ),
    )

    assert evaluation.matched_control_dose == (
        "per_call_decoded_rms_clip_to_smaller_without_amplification"
    )
    for balance in (
        evaluation.paired_ablation_displacement_balance,
        evaluation.paired_donor_displacement_balance,
    ):
        assert balance is not None
        assert balance["passed"] is True
        assert balance["symmetric_rms_ratio"] == pytest.approx(1.0, abs=1e-5)
        matching = balance["dose_matching"]
        assert matching["amplification_allowed"] is False
        assert matching["full_edit"]["symmetric_rms_ratio"] == pytest.approx(
            8.0, rel=1e-5
        )
        assert matching["dose_matched_partial_edit"][
            "symmetric_rms_ratio"
        ] == pytest.approx(1.0, abs=1e-5)
        for call in matching["by_call"]:
            assert call["target_scale"] == pytest.approx(0.125, rel=1e-5)
            assert call["control_scale"] == 1.0
            assert call["scaled_symmetric_rms_ratio"] == pytest.approx(
                1.0, abs=1e-5
            )
    assert evaluation.paired_donor_shift_balance is not None
    assert evaluation.paired_donor_shift_balance["symmetric_rms_ratio"] == (
        pytest.approx(1.0, abs=1e-5)
    )
    assert np.array_equal(
        evaluation.conditions["roundtrip_restore_control"].prediction.probabilities,
        evaluation.conditions["no_op_reconstruction"].prediction.probabilities,
    )
    assert torch.equal(target_generator.get_state(), target_final)
    assert torch.equal(source_generator.get_state(), source_final)


def test_decoded_dose_matching_rejects_zero_control_dose_and_rolls_back(
    causal_inputs,
) -> None:
    X, y = causal_inputs
    target_classifier = FakeOfficialClassifier(temporary=True)
    source_classifier = FakeOfficialClassifier(temporary=False)
    target_generator = target_classifier.model_.row_interactor._identity_generator
    source_generator = source_classifier.model_.row_interactor._identity_generator
    target_entry = target_generator.get_state().clone()
    source_entry = source_generator.get_state().clone()

    with pytest.raises(RuntimeError, match="finite positive"):
        run_official_tabicl_causal_edits(
            _paired_driver(target_classifier, checkpoint_sha="b" * 64),
            X,
            y,
            site="row_interactor",
            autoencoder=ExactOvercompleteAutoencoder(),
            normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
            target_features=(0,),
            control_features=(2,),
            dataset_id="toy-official",
            sample_ids=("row-0", "row-1"),
            representation_qualification=_paired_qualification(),
            target_condition="temporary",
            paired_source=PairedReversePatchSource(
                _paired_driver(source_classifier, checkpoint_sha="c" * 64),
                "none",
            ),
            maximum_symmetric_donor_shift_rms_ratio=1.25,
            matched_control_dose=(
                "per_call_decoded_rms_clip_to_smaller_without_amplification"
            ),
        )

    assert torch.equal(target_generator.get_state(), target_entry)
    assert torch.equal(source_generator.get_state(), source_entry)


def test_dose_matching_preserves_the_smaller_full_edit_exactly() -> None:
    record = ActivationRecord(
        tensor=torch.tensor([[0.1234567], [-0.9876543]], dtype=torch.float32),
        site="row_interactor",
        axis_names=("row", "row_representation"),
        shape=(2, 1),
        model_sha="a" * 40,
        checkpoint_sha="b" * 64,
        preprocessing_view_id="toy-view",
    )
    autoencoder = ImbalancedExactAutoencoder()
    normalizer = MeanRMSNormalizer(torch.zeros(1), torch.ones(1))
    call = official_causal._encode_call(
        SimpleNamespace(
            activations={"row_interactor": record},
            metadata=SimpleNamespace(call_index=7),
        ),
        site="row_interactor",
        autoencoder=autoencoder,
        normalizer=normalizer,
    )
    target_latent = call.latents.clone()
    target_latent[:, 0] = 0.0
    control_latent = call.latents.clone()
    control_latent[:, 1] = 0.0
    target_replacement = official_causal._decode_calls(
        (call,),
        (target_latent,),
        autoencoder=autoencoder,
        normalizer=normalizer,
    )
    control_replacement = official_causal._decode_calls(
        (call,),
        (control_latent,),
        autoencoder=autoencoder,
        normalizer=normalizer,
    )

    (
        scaled_target_latents,
        scaled_control_latents,
        _scaled_target_replacements,
        scaled_control_replacements,
        matching,
    ) = official_causal._match_decoded_edit_doses_by_call(
        (call,),
        (target_latent,),
        (control_latent,),
        target_replacement,
        control_replacement,
        autoencoder=autoencoder,
        normalizer=normalizer,
    )

    assert not torch.equal(scaled_target_latents[0], target_latent)
    assert torch.equal(scaled_control_latents[0], control_latent)
    assert torch.equal(scaled_control_replacements[0], control_replacement[0])
    by_call = matching["by_call"][0]
    assert by_call["target_full_edit_preserved"] is False
    assert by_call["control_full_edit_preserved"] is True
    assert by_call["actual_no_amplification_verified"] is True
    assert matching["actual_no_amplification_verified"] is True
    assert by_call["scaled_target_displacement_rms"] <= (
        by_call["original_target_displacement_rms"]
    )
    assert by_call["scaled_control_displacement_rms"] == (
        by_call["original_control_displacement_rms"]
    )


def test_dose_matching_gates_live_activation_precision() -> None:
    record = ActivationRecord(
        tensor=torch.zeros((1, 1), dtype=torch.float16),
        site="row_interactor",
        axis_names=("row", "row_representation"),
        shape=(1, 1),
        model_sha="a" * 40,
        checkpoint_sha="b" * 64,
        preprocessing_view_id="toy-view",
    )
    call = official_causal._encode_call(
        SimpleNamespace(activations={"row_interactor": record}),
        site="row_interactor",
        autoencoder=ExactOvercompleteAutoencoder(),
        normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
    )
    assert call.reconstruction.dtype == torch.float16
    target_latent = call.latents.clone()
    target_latent[:, 0] = 1e-8
    control_latent = call.latents.clone()
    control_latent[:, 1] = 1.0
    autoencoder = ExactOvercompleteAutoencoder()
    normalizer = MeanRMSNormalizer(torch.zeros(1), torch.ones(1))
    target_replacement = official_causal._decode_calls(
        (call,),
        (target_latent,),
        autoencoder=autoencoder,
        normalizer=normalizer,
    )
    control_replacement = official_causal._decode_calls(
        (call,),
        (control_latent,),
        autoencoder=autoencoder,
        normalizer=normalizer,
    )
    assert target_replacement[0].dtype == torch.float16
    assert float(target_replacement[0].detach().abs().max()) == 0.0

    with pytest.raises(RuntimeError, match="finite positive"):
        official_causal._match_decoded_edit_doses_by_call(
            (call,),
            (target_latent,),
            (control_latent,),
            target_replacement,
            control_replacement,
            autoencoder=autoencoder,
            normalizer=normalizer,
        )


def test_qualification_fails_before_any_model_or_rng_change(causal_inputs):
    X, y = causal_inputs
    classifier = FakeOfficialClassifier(temporary=True)
    generator = classifier.model_.row_interactor._identity_generator
    initial = generator.get_state().clone()

    with pytest.raises(RuntimeError, match="qualification"):
        run_official_tabicl_causal_edits(
            make_driver(classifier),
            X,
            y,
            site="row_interactor",
            autoencoder=ExactOvercompleteAutoencoder(),
            normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
            target_features=(0,),
            dataset_id="toy-official",
            sample_ids=(0, 1),
            representation_qualification=qualification(False),
        )

    assert torch.equal(generator.get_state(), initial)


def test_qualification_identity_mismatch_fails_before_inference(causal_inputs):
    X, y = causal_inputs
    classifier = FakeOfficialClassifier(temporary=True)
    generator = classifier.model_.row_interactor._identity_generator
    initial = generator.get_state().clone()
    mismatched = qualification()
    mismatched["checkpoint_sha"] = "0" * 64

    with pytest.raises(RuntimeError, match="not bound"):
        run_official_tabicl_causal_edits(
            make_driver(classifier),
            X,
            y,
            site="row_interactor",
            autoencoder=ExactOvercompleteAutoencoder(),
            normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
            target_features=(0,),
            dataset_id="toy-official",
            sample_ids=(0, 1),
            representation_qualification=mismatched,
        )

    assert torch.equal(generator.get_state(), initial)


def test_no_op_native_accuracy_gate_is_enforced_and_rng_rolls_back(
    causal_inputs,
):
    X, _ = causal_inputs
    y = np.array([0, 0])
    classifier = FakeOfficialClassifier(temporary=True)
    generator = classifier.model_.row_interactor._identity_generator
    initial = generator.get_state().clone()

    with pytest.raises(RuntimeError, match="absolute accuracy difference"):
        run_official_tabicl_causal_edits(
            make_driver(classifier),
            X,
            y,
            site="row_interactor",
            autoencoder=ExactOvercompleteAutoencoder(negate_decode=True),
            normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
            target_features=(0,),
            dataset_id="toy-official",
            sample_ids=(0, 1),
            representation_qualification=qualification(),
            max_no_op_reconstruction_mse=10.0,
            max_no_op_probability_deviation=1.0,
            max_no_op_accuracy_drop=0.005,
        )

    assert torch.equal(generator.get_state(), initial)


def test_no_op_probability_gate_is_enforced_and_rng_rolls_back(
    causal_inputs,
):
    X, y = causal_inputs
    classifier = FakeOfficialClassifier(temporary=True)
    generator = classifier.model_.row_interactor._identity_generator
    initial = generator.get_state().clone()
    autoencoder = ExactOvercompleteAutoencoder()
    with torch.no_grad():
        autoencoder.decoder.weight[0, 0] = 0.99

    with pytest.raises(RuntimeError, match="probability difference"):
        run_official_tabicl_causal_edits(
            make_driver(classifier),
            X,
            y,
            site="row_interactor",
            autoencoder=autoencoder,
            normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
            target_features=(0,),
            dataset_id="toy-official",
            sample_ids=(0, 1),
            representation_qualification=qualification(),
            max_no_op_reconstruction_mse=10.0,
            max_no_op_probability_deviation=0.0,
            max_no_op_accuracy_drop=1.0,
        )

    assert torch.equal(generator.get_state(), initial)


def test_cross_condition_schedule_drift_fails_closed_and_rolls_back(
    causal_inputs,
):
    X, y = causal_inputs

    class DriftingClassifier(FakeOfficialClassifier):
        def __init__(self):
            super().__init__(temporary=True)
            self.predict_calls = 0

        def predict_proba(self, values):
            result = super().predict_proba(values)
            self.predict_calls += 1
            if self.predict_calls == 2:
                current = self.ensemble_generator_.feature_shuffles_["none"]
                self.ensemble_generator_.feature_shuffles_["none"] = list(
                    reversed(current)
                )
            return result

    classifier = DriftingClassifier()
    generator = classifier.model_.row_interactor._identity_generator
    initial = generator.get_state().clone()

    with pytest.raises(RuntimeError, match="feature permutations differ"):
        run_official_tabicl_causal_edits(
            make_driver(classifier),
            X,
            y,
            site="row_interactor",
            autoencoder=ExactOvercompleteAutoencoder(),
            normalizer=MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
            target_features=(0,),
            dataset_id="toy-official",
            sample_ids=(0, 1),
            representation_qualification=qualification(),
        )

    assert torch.equal(generator.get_state(), initial)


@pytest.mark.parametrize("model_kind", ["dense", "topk", "pca"])
def test_official_causal_matched_controls_support_all_representation_models(
    causal_inputs, model_kind: str
) -> None:
    X, y = causal_inputs

    class ThreeDimensionalColumnEmbedder(nn.Module):
        feature_group = "same"
        feature_group_size = 2

        def forward(self, values, **_kwargs):
            return torch.stack((values, values * 0.5, -values), dim=-1)

    classifier = FakeOfficialClassifier(temporary=True)
    classifier.model_.col_embedder = ThreeDimensionalColumnEmbedder()
    if model_kind == "dense":
        model = DenseAutoencoder(3, 3, activation="linear")
    elif model_kind == "topk":
        model = TopKSparseAutoencoder(3, latent_dim=3, top_k=2)
    else:
        model = PCARepresentation(3, 3).fit(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )

    evaluation = run_official_tabicl_causal_edits(
        make_driver(classifier),
        X,
        y,
        site="row_interactor",
        autoencoder=model,
        normalizer=MeanRMSNormalizer(torch.zeros(3), torch.ones(3)),
        target_features=(0,),
        dataset_id="toy-official",
        sample_ids=(0, 1),
        representation_qualification=qualification(),
        random_candidate_pool_size=2,
        max_no_op_reconstruction_mse=100.0,
        max_no_op_probability_deviation=1.0,
        max_no_op_accuracy_drop=1.0,
    )

    assert len(evaluation.matched_control_features) == 1
    assert model_decoder_feature_norms(model).shape == (3,)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _clean_repository(root: Path) -> str:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "user.email", "test@example.invalid")
    (root / "tracked.txt").write_text("strict source\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-q", "-m", "initial")
    return _git(root, "rev-parse", "HEAD")


def test_formal_provenance_module_must_come_from_bound_clean_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "bound-repository"
    _clean_repository(repository)
    module_file = repository / "src" / "tabicl" / "train" / "_provenance.py"
    module_file.parent.mkdir(parents=True)
    module_file.write_text(
        "class ParentTrust:\n"
        "    def __init__(self, **values):\n"
        "        self.__dict__.update(values)\n"
        "\n"
        "def validate_parent_trust(trust):\n"
        "    return trust\n"
        "\n"
        "def validate_canonical_transaction_ledger(payload, *, artifact_root):\n"
        "    return payload, artifact_root\n",
        encoding="utf-8",
    )
    outside_file = tmp_path / "environment-shadow" / "_provenance.py"
    outside_file.parent.mkdir()
    outside_file.write_text("# environment shadow\n", encoding="utf-8")
    escaping_link = repository / "src" / "escaping-provenance.py"
    escaping_link.symlink_to(outside_file)
    _git(repository, "add", "src")
    _git(repository, "commit", "-q", "-m", "add provenance module")
    evidence = verify_git_tree(repository)
    context = SimpleNamespace(
        inputs=SimpleNamespace(training_code=evidence, model_code=evidence)
    )
    module_spec = importlib.util.spec_from_file_location(
        "tabicl.train._provenance", module_file
    )
    assert module_spec is not None
    fake_module = types.ModuleType("tabicl.train._provenance")
    fake_module.__file__ = str(module_file)
    fake_module.__spec__ = module_spec
    exec(
        compile(module_file.read_text(encoding="utf-8"), str(module_file), "exec"),
        fake_module.__dict__,
    )
    monkeypatch.setitem(
        sys.modules, "tabicl.train._provenance", fake_module
    )

    ParentTrust, validate_parent_trust = (
        official_causal._trusted_training_provenance_api(context)
    )
    trust = ParentTrust(value=1)
    assert validate_parent_trust(trust).value == 1

    fake_module.__file__ = str(outside_file)
    with pytest.raises(RuntimeError, match="not the canonical file"):
        official_causal._trusted_training_provenance_api(context)

    fake_module.__file__ = str(escaping_link)
    with pytest.raises(RuntimeError, match="not the canonical file"):
        official_causal._trusted_training_provenance_api(context)

    symlink_repository = tmp_path / "symlink-repository"
    _clean_repository(symlink_repository)
    symlink_module_file = (
        symlink_repository / "src" / "tabicl" / "train" / "_provenance.py"
    )
    symlink_module_file.parent.mkdir(parents=True)
    symlink_module_file.symlink_to(outside_file)
    _git(symlink_repository, "add", "src")
    _git(symlink_repository, "commit", "-q", "-m", "add provenance symlink")
    symlink_evidence = verify_git_tree(symlink_repository)
    symlink_context = SimpleNamespace(
        inputs=SimpleNamespace(
            training_code=symlink_evidence,
            model_code=symlink_evidence,
        )
    )
    fake_module.__file__ = str(symlink_module_file)
    fake_module.__spec__ = importlib.util.spec_from_file_location(
        "tabicl.train._provenance", symlink_module_file
    )
    with pytest.raises(RuntimeError, match="symlink escapes"):
        official_causal._trusted_training_provenance_api(symlink_context)

    fake_module.__file__ = None
    with pytest.raises(RuntimeError, match="concrete __file__"):
        official_causal._trusted_training_provenance_api(context)

    forged_module = types.ModuleType("tabicl.train._provenance")
    forged_module.__file__ = str(module_file)
    forged_module.ParentTrust = lambda **values: SimpleNamespace(**values)
    forged_module.validate_parent_trust = lambda trust: trust
    forged_module.validate_canonical_transaction_ledger = (
        lambda payload, *, artifact_root: (payload, artifact_root)
    )
    monkeypatch.setitem(
        sys.modules, "tabicl.train._provenance", forged_module
    )
    with pytest.raises(RuntimeError, match="not defined by the canonical"):
        official_causal._trusted_training_provenance_api(context)


def _write_numeric_talent(root: Path) -> None:
    root.mkdir()
    (root / "info.json").write_text(
        json.dumps({"task_type": "binclass", "num_classes": 2}),
        encoding="utf-8",
    )
    arrays = {
        "train": np.array(
            [
                [0.2, 0.3, 0.7, 1.1],
                [1.0, -0.4, 0.5, 0.2],
                [-0.2, 0.8, 0.4, -0.1],
                [0.6, 0.9, -0.5, 0.3],
            ],
            dtype=np.float32,
        ),
        "val": np.array([[0.3, 0.4, 0.1, -0.2]], dtype=np.float32),
        "test": np.array(
            [[0.1, 0.6, -0.3, 0.8], [0.9, -0.2, 0.4, 0.3]],
            dtype=np.float32,
        ),
    }
    labels = {
        "train": np.array([0, 1, 0, 1]),
        "val": np.array([1]),
        "test": np.array([0, 1]),
    }
    for split in ("train", "val", "test"):
        np.save(root / f"N_{split}.npy", arrays[split])
        np.save(root / f"y_{split}.npy", labels[split])


def _workflow_fixture(tmp_path: Path) -> SimpleNamespace:
    repository = tmp_path / "repository"
    head = _clean_repository(repository)
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"temporary official checkpoint")
    reference_checkpoint = tmp_path / "reference-checkpoint.ckpt"
    reference_checkpoint.write_bytes(b"rope reference checkpoint")
    dataset_manifest = tmp_path / "dataset-manifest.json"
    dataset_manifest.write_text(
        json.dumps(
            {
                "assignments": [
                    {"name": "pilot-dataset", "split": "discovery"},
                    {"name": "selection-dataset", "split": "validation"},
                    *[
                        {"name": f"validation-{index:02d}", "split": "validation"}
                        for index in range(1, 8)
                    ],
                    {"name": "toy-official", "split": "held_out"},
                    *[
                        {"name": f"heldout-{index:02d}", "split": "held_out"}
                        for index in range(1, 8)
                    ],
                ]
            }
        ),
        encoding="utf-8",
    )
    raw_dataset = tmp_path / "raw-talent"
    _write_numeric_talent(raw_dataset)
    sample_roster = tmp_path / "sample-roster.json"
    sample_roster.write_text(
        json.dumps(
            {
                "dataset_id": "selection-dataset",
                "split": "val",
                "row_indices": [0],
                "sample_ids": ["val-0"],
            }
        ),
        encoding="utf-8",
    )

    parent_dir = tmp_path / "train-repr-run"
    parent_dir.mkdir()
    representation = parent_dir / "model.pt"
    representation.write_bytes(b"verified representation checkpoint")
    representation_digest = verify_file(representation).digest
    collect_digests = {"rope": "c" * 64, "temporary": "d" * 64}
    parent_inputs = tuple(
        sorted(
            (
                InputDigest(
                    role=f"source.collect_manifest.{digest}",
                    sha256=digest,
                    size_bytes=1,
                )
                for digest in collect_digests.values()
            ),
            key=lambda item: item.role,
        )
    )
    estimator_options = {
        "n_estimators": 5,
        "norm_methods": ["none", "power"],
        "random_state": 42,
    }
    inference_contract_sha256 = official_inference_contract_sha256(
        head, estimator_options
    )
    source_lineage = {
        "schema_version": 1,
        "source_kind": "official_tabicl_bounded_activation_index",
        "reference_condition": "rope",
        "condition_checkpoints_sha256": {
            "rope": verify_file(reference_checkpoint).digest.sha256,
            "temporary": verify_file(checkpoint).digest.sha256,
        },
        "collect_parent_manifests_sha256": {
            condition: [digest]
            for condition, digest in sorted(collect_digests.items())
        },
        "alignment_sha256": {
            "training": {
                "training-dataset": {"row_interactor": "e" * 64}
            },
            "validation": {
                "validation-dataset": {"row_interactor": "f" * 64}
            },
        },
        "inference_contract_sha256": inference_contract_sha256,
        "evaluation_split": "val",
        "max_classes": 10,
    }
    parent = new_manifest(
        command="train-repr",
        model_family="tabicl-v2",
        model_revision="step-210000",
        training_code_sha=head,
        model_code_sha=head,
        analysis_code_sha=head,
        configuration=FileDigest("1" * 64, 1),
        checkpoint=verify_file(reference_checkpoint).digest,
        dataset_manifest=verify_file(dataset_manifest).digest,
        inputs=parent_inputs,
        condition="rope",
        sites=("row_interactor",),
        seed=42,
        artifacts=(
            ArtifactDigest(
                "model.pt",
                representation_digest.sha256,
                representation_digest.size_bytes,
            ),
        ),
        created_at_utc="2026-08-07T12:00:00Z",
    )
    (parent_dir / "manifest.json").write_text(
        json.dumps(parent.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    parent_manifest_digest = verify_file(parent_dir / "manifest.json").digest

    selection_dir = tmp_path / "validation-selection-run"
    selection_dir.mkdir()
    selection_summary = {
        "schema_version": 1,
        "analysis": "official-model-causal",
        "dataset_id": "selection-dataset",
        "fit_split": "train",
        "evaluation_split": "val",
        "roster_split": "validation",
        "evidence_scope": "exploratory-feature-selection",
        "site": "row_interactor",
        "source_evidence_level": "strict",
        "intervention": {
            "target_features": [0],
            "control_features": [1],
            "latent_baseline": 0.0,
            "random_seed": 42,
        },
        "input_bindings": {
            "model_sha": head,
            "checkpoint_sha256": verify_file(checkpoint).digest.sha256,
            "parent_manifest_sha256": parent_manifest_digest.sha256,
            "representation_model_sha256": representation_digest.sha256,
            "inference_contract_sha256": inference_contract_sha256,
        },
        "representation_source_lineage": source_lineage,
        "no_op_gates": {"passed": True},
    }
    (selection_dir / "summary.json").write_text(
        json.dumps(selection_summary, sort_keys=True), encoding="utf-8"
    )
    selection_summary_digest = verify_file(
        selection_dir / "summary.json"
    ).digest
    selection_manifest = new_manifest(
        command="model-causal",
        model_family="tabicl-v2",
        model_revision="step-210000",
        training_code_sha=head,
        model_code_sha=head,
        analysis_code_sha=head,
        configuration=FileDigest("2" * 64, 1),
        checkpoint=verify_file(checkpoint).digest,
        dataset_manifest=verify_file(dataset_manifest).digest,
        inputs=tuple(
            sorted(
                (
                    InputDigest(
                        "representation.model",
                        representation_digest.sha256,
                        representation_digest.size_bytes,
                    ),
                    InputDigest(
                        "representation.parent_manifest",
                        parent_manifest_digest.sha256,
                        parent_manifest_digest.size_bytes,
                    ),
                ),
                key=lambda item: item.role,
            )
        ),
        condition="temporary",
        sites=("row_interactor",),
        seed=42,
        artifacts=(
            ArtifactDigest(
                "summary.json",
                selection_summary_digest.sha256,
                selection_summary_digest.size_bytes,
            ),
        ),
        created_at_utc="2026-08-07T12:30:00Z",
    )
    (selection_dir / "manifest.json").write_text(
        json.dumps(selection_manifest.to_dict(), sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    selection_manifest_digest = verify_file(
        selection_dir / "manifest.json"
    ).digest

    freeze_artifact = tmp_path / "intervention-freeze.json"
    freeze_artifact.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_scope": "validation-frozen",
                "condition": "temporary",
                "site": "row_interactor",
                "random_seed": 42,
                "evaluation_sample_roster_sha256": verify_file(
                    sample_roster
                ).digest.sha256,
                "target_features": [0],
                "control_features": [1],
                "latent_baseline": 0.0,
                "representation_model_sha256": representation_digest.sha256,
                "representation_parent_manifest_sha256": (
                    parent_manifest_digest.sha256
                ),
                "model_sha": head,
                "checkpoint_sha256": verify_file(checkpoint).digest.sha256,
                "inference_contract_sha256": inference_contract_sha256,
                "selection_parent_manifest_sha256": (
                    selection_manifest_digest.sha256
                ),
                "selection_summary_sha256": selection_summary_digest.sha256,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    transaction_ledger = tmp_path / "formal-transaction-ledger.json"
    transaction_ledger.write_text("{}\n", encoding="utf-8")
    recipient_finalized = tmp_path / "temporary-stage1-finalized.json"
    recipient_finalized.write_text("{}\n", encoding="utf-8")

    config = tmp_path / "model-causal.json"
    config.write_text(
        json.dumps(
            {
                "provenance": {
                    "model_family": "tabicl-v2",
                    "model_revision": "step-210000",
                    "condition": "temporary",
                    "sites": ["row_interactor"],
                    "checkpoint_path": str(checkpoint),
                    "dataset_manifest_path": str(dataset_manifest),
                    "training_code_root": str(repository),
                    "model_code_root": str(repository),
                    "analysis_code_root": str(repository),
                },
                "checkpoint_study": {
                    "scope": "formal",
                    "recipient_chain": [
                        {
                            "checkpoint_path": str(checkpoint),
                            "expected_checkpoint_sha256": verify_file(
                                checkpoint
                            ).digest.sha256,
                            "finalized_manifest_path": str(recipient_finalized),
                            "expected_finalized_manifest_file_sha256": verify_file(
                                recipient_finalized
                            ).digest.sha256,
                            "expected_finalized_manifest_sha256": "8" * 64,
                            "transaction_ledger_path": str(transaction_ledger),
                            "expected_transaction_ledger_file_sha256": verify_file(
                                transaction_ledger
                            ).digest.sha256,
                            "transaction_ledger_sha256": "9" * 64,
                            "artifact_root": str(tmp_path),
                            "study_id": "formal-study",
                            "arm": "temporary",
                            "stage": "stage1",
                            "upstream_identity": (
                                "formal-study:temporary:stage1"
                            ),
                            "artifact_identity": (
                                "formal-study.temporary.stage1.final"
                            ),
                        }
                    ],
                    "source_chain": None,
                },
                "representation_run_dir": str(parent_dir),
                "dataset": {
                    "dataset_id": "selection-dataset",
                    "dataset_dir": str(raw_dataset),
                    "fit_split": "train",
                    "roster_split": "validation",
                    "evaluation_split": "val",
                    "sample_roster_path": str(sample_roster),
                    "trusted_pickle": False,
                },
                "intervention": {
                    "site": "row_interactor",
                    "target_features": [0],
                    "control_features": [1],
                    "latent_baseline": 0.0,
                    "freeze_artifact_path": None,
                    "random_seed": 42,
                    "random_candidate_pool_size": 2,
                    "max_no_op_reconstruction_mse": 0.01,
                    "max_no_op_probability_deviation": 0.02,
                    "max_no_op_accuracy_difference": 0.005,
                },
                "official_classifier": {
                    "device": "cpu",
                    "estimator_options": estimator_options,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        repository=repository,
        head=head,
        checkpoint=checkpoint,
        reference_checkpoint=reference_checkpoint,
        dataset_manifest=dataset_manifest,
        parent_dir=parent_dir,
        representation=representation,
        source_lineage=source_lineage,
        freeze_artifact=freeze_artifact,
        selection_dir=selection_dir,
        transaction_ledger=transaction_ledger,
        config=config,
        output=tmp_path / "published-model-causal",
        recipient_condition="temporary",
    )


def _install_workflow_fakes(
    fixture: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    *,
    driver_temporary: bool = True,
) -> dict[str, object]:
    observed: dict[str, object] = {}

    def verify_context(
        configuration,
        *,
        command,
        seed,
        additional_input_paths,
        expected_additional_sha256,
    ):
        verified = verify_run_inputs(
            configuration_path=configuration.file.path,
            checkpoint_path=fixture.checkpoint,
            dataset_manifest_path=fixture.dataset_manifest,
            training_code_root=fixture.repository,
            model_code_root=fixture.repository,
            analysis_code_root=fixture.repository,
            command=command,
            model_family="tabicl-v2",
            model_revision="step-210000",
            condition=fixture.recipient_condition,
            sites=("row_interactor",),
            seed=seed,
            additional_input_paths=additional_input_paths,
            expected_additional_sha256=expected_additional_sha256,
        )
        return VerifiedRunContext(
            inputs=verified,
            model_family="tabicl-v2",
            model_revision="step-210000",
            condition=fixture.recipient_condition,
            sites=("row_interactor",),
        )

    def load_representation(verified, **_kwargs):
        observed["representation_digest"] = verified.digest.sha256
        assert verified.read_bytes() == b"verified representation checkpoint"
        metrics = {
            "explained_variance": 0.99,
            "normalized_mse": 0.01,
            "mse": 0.001,
            "dead_features": 0,
            "dead_feature_fraction": 0.0,
            "active_count": 2.0,
            "mean_active_features": 2.0,
        }
        return (
            ExactOvercompleteAutoencoder(),
            MeanRMSNormalizer(torch.zeros(1), torch.ones(1)),
            {
                "qualification": {
                    "metric_split": "validation",
                    "held_out_validation": True,
                    "activation_fidelity_passed": True,
                    "explained_variance": 0.99,
                    "minimum_explained_variance": 0.95,
                    "validation_by_condition": {
                        condition: dict(metrics)
                        for condition in fixture.source_lineage[
                            "condition_checkpoints_sha256"
                        ]
                    },
                    "worst_condition_explained_variance": 0.99,
                    "native_score_gate": "pending",
                },
                "metadata": {"source_lineage": fixture.source_lineage},
            },
        )

    def fit_driver(
        dataset,
        checkpoint,
        *,
        context_split,
        model_sha,
        **_kwargs,
    ):
        observed["fit_split"] = context_split
        observed["fit_rows"] = len(dataset.train.y)
        evidence = verify_git_tree(fixture.repository, expected_sha=model_sha)
        classifier = FakeOfficialClassifier(temporary=driver_temporary)
        if verify_file(checkpoint).digest == verify_file(
            fixture.reference_checkpoint
        ).digest:
            classifier = FakeOfficialClassifier(temporary=False)
            classifier.model_.row_identity_mode = "rope"
            classifier.model_.row_interactor.identity_mode = "rope"
            row_interactor = classifier.model_.row_interactor

            def rope_forward(self, embeddings, **_kwargs):
                return embeddings.mean(dim=2) + 0.125

            row_interactor.forward = types.MethodType(
                rope_forward, row_interactor
            )
        return OfficialTabICLDriver(
            classifier,
            adapter=TabICLAdapter(),
            model_sha=model_sha,
            checkpoint_sha=verify_file(checkpoint).digest.sha256,
            fit_context=f"talent-{context_split}",
            _source_evidence=evidence,
        )

    def validate_checkpoint_study(study, *, context, paired_source_binding):
        if study["scope"] == "exploratory_pilot":
            payload = {
                "schema_version": 1,
                "scope": "exploratory_pilot",
                "formal_trust_verified": False,
                "model_evidence_scope": "exploratory-pilot",
            }
            return {
                **payload,
                "binding_sha256": official_causal._canonical_sha256(payload),
            }
        recipient = {
            "study_id": "formal-study",
            "mode": "temporary",
            "stage": "stage1",
            "terminal_step": 500_000,
            "checkpoint_sha256": context.inputs.checkpoint.digest.sha256,
        }
        source = (
            None
            if paired_source_binding is None
            else [
                {
                    **recipient,
                    "mode": paired_source_binding["source_condition"],
                    "checkpoint_sha256": paired_source_binding[
                        "checkpoint_sha256"
                    ],
                }
            ]
        )
        payload = {
            "schema_version": 1,
            "scope": "formal",
            "formal_trust_verified": True,
            "model_evidence_scope": "intermediate-stage-specific",
            "direction": (
                None
                if source is None
                else {
                    "source_condition": paired_source_binding[
                        "source_condition"
                    ],
                    "recipient_condition": context.condition,
                }
            ),
            "recipient_chain": [recipient],
            "source_chain": source,
        }
        return {
            **payload,
            "binding_sha256": official_causal._canonical_sha256(payload),
        }

    monkeypatch.setattr(
        official_causal, "verify_configured_run_inputs", verify_context
    )
    monkeypatch.setattr(
        official_causal,
        "load_verified_representation_checkpoint",
        load_representation,
    )
    monkeypatch.setattr(
        official_causal, "fit_official_talent_driver", fit_driver
    )
    monkeypatch.setattr(
        official_causal,
        "_validated_checkpoint_study",
        validate_checkpoint_study,
    )
    return observed


def _enable_validation_paired_reverse_patch(
    fixture: SimpleNamespace, tmp_path: Path
) -> Path:
    sample_roster = tmp_path / "validation-sample-roster.json"
    sample_roster.write_text(
        json.dumps(
            {
                "dataset_id": "selection-dataset",
                "split": "val",
                "row_indices": [0],
                "sample_ids": ["val-0"],
            }
        ),
        encoding="utf-8",
    )
    checkpoint_sha256 = verify_file(fixture.reference_checkpoint).digest.sha256
    estimator_options = {
        "n_estimators": 5,
        "norm_methods": ["none", "power"],
        "random_state": 42,
    }
    inference_contract_sha256 = official_inference_contract_sha256(
        fixture.head, estimator_options
    )
    attestation = tmp_path / "paired-source-code-attestation.json"
    attestation.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_kind": "clean-git-checkout",
                "source_condition": "rope",
                "model_code_sha": fixture.head,
                "checkpoint_sha256": checkpoint_sha256,
                "inference_contract_sha256": inference_contract_sha256,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    source_finalized = tmp_path / "rope-stage1-finalized.json"
    source_finalized.write_text("{}\n", encoding="utf-8")
    config["dataset"].update(
        {
            "dataset_id": "selection-dataset",
            "roster_split": "validation",
            "evaluation_split": "val",
            "sample_roster_path": str(sample_roster),
        }
    )
    config["intervention"].update(
        {
            "freeze_artifact_path": None,
        }
    )
    for name in (
        "expected_freeze_artifact_sha256",
        "selection_run_dir",
        "expected_selection_manifest_sha256",
    ):
        config["intervention"].pop(name, None)
    config["paired_reverse_patch"] = {
        "source_condition": "rope",
        "source_checkpoint_path": str(fixture.reference_checkpoint),
        "expected_source_checkpoint_sha256": checkpoint_sha256,
        "source_model_code_root": str(fixture.repository),
        "expected_source_model_code_sha": fixture.head,
        "source_code_attestation_path": str(attestation),
        "expected_source_code_attestation_sha256": verify_file(
            attestation
        ).digest.sha256,
        "maximum_symmetric_donor_shift_rms_ratio": 1.25,
    }
    recipient_trust = config["checkpoint_study"]["recipient_chain"][0]
    config["checkpoint_study"]["source_chain"] = [
        {
            "checkpoint_path": str(fixture.reference_checkpoint),
            "expected_checkpoint_sha256": checkpoint_sha256,
            "finalized_manifest_path": str(source_finalized),
            "expected_finalized_manifest_file_sha256": verify_file(
                source_finalized
            ).digest.sha256,
            "expected_finalized_manifest_sha256": "7" * 64,
            "transaction_ledger_path": recipient_trust[
                "transaction_ledger_path"
            ],
            "expected_transaction_ledger_file_sha256": recipient_trust[
                "expected_transaction_ledger_file_sha256"
            ],
            "transaction_ledger_sha256": recipient_trust[
                "transaction_ledger_sha256"
            ],
            "artifact_root": recipient_trust["artifact_root"],
            "study_id": "formal-study",
            "arm": "rope",
            "stage": "stage1",
            "upstream_identity": "formal-study:rope:stage1",
            "artifact_identity": "formal-study.rope.stage1.final",
        }
    ]
    fixture.config.write_text(
        json.dumps(config, sort_keys=True), encoding="utf-8"
    )
    return attestation


def _enable_exploratory_ranking_paired(
    fixture: SimpleNamespace, tmp_path: Path
) -> SimpleNamespace:
    fixture.recipient_condition = "none"
    source_lineage = json.loads(json.dumps(fixture.source_lineage))
    source_lineage["condition_checkpoints_sha256"]["none"] = (
        source_lineage["condition_checkpoints_sha256"].pop("temporary")
    )
    source_lineage["collect_parent_manifests_sha256"]["none"] = (
        source_lineage["collect_parent_manifests_sha256"].pop("temporary")
    )
    fixture.source_lineage = source_lineage

    roster = tmp_path / "pilot-ranking-roster.json"
    roster.write_text(
        json.dumps(
            {
                "dataset_id": "pilot-dataset",
                "split": "val",
                "row_indices": [0],
                "sample_ids": ["pilot-0"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    roster_digest = verify_file(roster).digest
    split_protocol = tmp_path / "whole-row-causal-split.json"
    split_protocol.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": "toy-whole-row-split",
                "feature_ranking_datasets": ["ranking-dataset"],
                "causal_test_datasets": ["pilot-dataset"],
                "causal_sample_protocol": {
                    "split": "val",
                    "maximum_rows_per_dataset": 1,
                    "selection": "Use the frozen validation row.",
                    "sample_roster_sha256_by_dataset": {
                        "pilot-dataset": roster_digest.sha256
                    },
                    "sample_count_by_dataset": {"pilot-dataset": 1},
                },
                "feature_protocol": {
                    "target_count": 1,
                    "score": (
                        "median across feature-ranking datasets of RMS "
                        "RoPE-minus-No-PE latent difference times "
                        "decoder-direction norm divided by raw activation RMS"
                    ),
                    "minimum_nonzero_ranking_datasets": 1,
                    "matched_control": (
                        "activation-frequency and log-decoder-norm "
                        "nearest-neighbour pool of size 1, sampled once with "
                        "seed 42 without replacement"
                    ),
                    "same_features_both_directions": True,
                },
                "causal_protocol": {
                    "directions": ["rope_to_none", "none_to_rope"],
                    "maximum_symmetric_donor_shift_rms_ratio": 1.25,
                    "checkpoint_scope": "exploratory_pilot",
                    "formal_claim": "forbidden",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    split_digest = verify_file(split_protocol).digest
    dose_protocol = tmp_path / "whole-row-causal-dose-amendment.json"
    dose_protocol.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": (
                    "tabicl-step250k-whole-row-causal-dose-amendment-v1"
                ),
                "frozen_at": "2026-08-09T22:36:19+01:00",
                "parent_split_protocol_id": "toy-whole-row-split",
                "parent_split_protocol_sha256": split_digest.sha256,
                "scope": (
                    "exploratory_pilot_ranking_bound_paired_"
                    "row_interactor_only"
                ),
                "formal_claim": "forbidden",
                "trigger": (
                    "A decoded-dose balance gate failed before any target, control, "
                    "or donor intervention prediction was published."
                ),
                "model_outcomes_observed_before_freeze": False,
                "matched_control_dose": (
                    "per_call_decoded_rms_clip_to_smaller_without_amplification"
                ),
                "reference_activation": "recipient_no_op_reconstruction",
                "matching_unit": "official_raw_model_call",
                "dose_metric": (
                    "root_mean_square_of_decoded_activation_edit_after_cast_to_live_"
                    "activation_dtype_minus_recipient_no_op_reconstruction_"
                    "after_same_cast"
                ),
                "adjustment": (
                    "Set the common requested dose to the smaller full-edit dose, "
                    "retain the smaller latent edit and decoded activation byte-for-"
                    "byte, shrink only the larger latent edit by their ratio, decode "
                    "the changed edit again, cast both edit and no-op reconstruction "
                    "to the live activation dtype, reject any actual per-side dose "
                    "increase, and gate the actual injected values."
                ),
                "amplification_allowed": False,
                "zero_or_non_finite_dose_policy": "fail_closed",
                "maximum_post_adjustment_symmetric_rms_ratio": 1.25,
                "interpretation": (
                    "The executed interventions are dose-matched partial edits, not "
                    "complete deletion or complete transplantation. Target ablation "
                    "and donor patch families are matched only within their own "
                    "target/control pair and are not dose-comparable to each other."
                ),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    dose_digest = verify_file(dose_protocol).digest
    parent = verify_run_directory(fixture.parent_dir)
    parent_manifest_digest = verify_file(
        fixture.parent_dir / "manifest.json"
    ).digest
    representation_digest = verify_file(fixture.representation).digest
    model = ExactOvercompleteAutoencoder()
    normalizer = MeanRMSNormalizer(torch.zeros(1), torch.ones(1))
    decoder_norms = (
        raw_space_decoder_feature_norms(model, normalizer)
        .cpu()
        .numpy()
    )
    median_scores = np.asarray([3.0, 2.0, 1.0], dtype=np.float64)
    nonzero_counts = np.asarray([1, 1, 1], dtype=np.int64)
    activation_frequencies = np.asarray([0.5, 0.5, 0.1], dtype=np.float64)
    ranking_protocol = {
        "dataset_ids": ["ranking-dataset"],
        "minimum_nonzero_datasets": 1,
        "target_count": 1,
        "random_candidate_pool_size": 1,
        "random_seed": 42,
        "activation_frequency_threshold": 1e-8,
        "decoder_norm_space": "raw_activation_after_denormalize",
        "latent_baseline": 0.0,
        "same_features_both_directions": True,
    }
    activation_files = {}
    activation_binding = {}
    for condition in ("none", "rope"):
        path = tmp_path / f"ranking-{condition}-activation.npz"
        path.write_bytes(f"verified-{condition}-activation".encode())
        activation_files[condition] = path
        activation_binding[condition] = {
            "ranking-dataset": verify_file(path).digest.sha256
        }
    top_features = [
        {
            "feature": index,
            "score": float(median_scores[index]),
            "nonzero_dataset_count": int(nonzero_counts[index]),
        }
        for index in range(3)
    ]
    selection = {
        "schema_version": 1,
        "analysis": "exploratory-condition-shift-ranking",
        "evidence_scope": "exploratory-pilot",
        "formal_claim": "forbidden",
        "site": "row_interactor",
        "conditions": ["rope", "none"],
        "directions": ["rope_to_none", "none_to_rope"],
        "maximum_symmetric_donor_shift_rms_ratio": 1.25,
        "matched_control_dose": (
            "per_call_decoded_rms_clip_to_smaller_without_amplification"
        ),
        "dose_protocol_id": (
            "tabicl-step250k-whole-row-causal-dose-amendment-v1"
        ),
        "dose_protocol_sha256": dose_digest.sha256,
        "score_definition": (
            "median_dataset_rms_latent_rope_minus_none_times_decoder_norm_"
            "divided_by_pooled_raw_activation_rms"
        ),
        "decoder_norm_space": "raw_activation_after_denormalize",
        "raw_activation_rms_definition": (
            "root_mean_square_of_pooled_aligned_none_and_rope_raw_values"
        ),
        "representation_parent_manifest_sha256": (
            parent_manifest_digest.sha256
        ),
        "representation_model_sha256": representation_digest.sha256,
        "representation_source_lineage_sha256": (
            official_causal._canonical_sha256(source_lineage)
        ),
        "ranking_source_lineage_sha256": (
            official_causal._canonical_sha256(source_lineage)
        ),
        "split_protocol_id": "toy-whole-row-split",
        "split_protocol_sha256": split_digest.sha256,
        "ranking_dataset_roster_sha256": (
            official_causal._canonical_sha256(["ranking-dataset"])
        ),
        "activation_sha256_by_condition": activation_binding,
        "activation_binding_sha256": (
            official_causal._canonical_sha256(activation_binding)
        ),
        "inference_contract_sha256": source_lineage[
            "inference_contract_sha256"
        ],
        "condition_checkpoints_sha256": source_lineage[
            "condition_checkpoints_sha256"
        ],
        "ranking_protocol": ranking_protocol,
        "ranking_protocol_sha256": official_causal._canonical_sha256(
            ranking_protocol
        ),
        "dataset_score_sha256": {"ranking-dataset": "a" * 64},
        "raw_activation_rms_by_dataset": {"ranking-dataset": 1.0},
        "median_scores": median_scores.tolist(),
        "median_scores_sha256": (
            official_causal._ranking_numeric_array_sha256(median_scores)
        ),
        "nonzero_dataset_counts": nonzero_counts.tolist(),
        "nonzero_dataset_counts_sha256": (
            official_causal._ranking_numeric_array_sha256(nonzero_counts)
        ),
        "activation_frequencies": activation_frequencies.tolist(),
        "activation_frequencies_sha256": (
            official_causal._ranking_numeric_array_sha256(
                activation_frequencies
            )
        ),
        "decoder_norms": decoder_norms.tolist(),
        "decoder_norms_sha256": (
            official_causal._ranking_numeric_array_sha256(decoder_norms)
        ),
        "top_features": top_features,
        "target_features": [0],
        "control_features": [1],
        "latent_baseline": 0.0,
        "parent_seed": 42,
    }
    ranking_dir = tmp_path / "ranking-parent-run"
    ranking_dir.mkdir()
    selection_path = ranking_dir / "selection.json"
    selection_path.write_text(
        json.dumps(selection, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    selection_digest = verify_file(selection_path).digest
    ranking_inputs = [
        InputDigest(
            "ranking.split_protocol",
            split_digest.sha256,
            split_digest.size_bytes,
        ),
        InputDigest(
            "ranking.dose_protocol",
            dose_digest.sha256,
            dose_digest.size_bytes,
        ),
        InputDigest(
            "representation.parent_manifest",
            parent_manifest_digest.sha256,
            parent_manifest_digest.size_bytes,
        ),
        InputDigest(
            "representation.model",
            representation_digest.sha256,
            representation_digest.size_bytes,
        ),
    ]
    for index, digest in enumerate(
        sorted(
            value[0]
            for value in source_lineage[
                "collect_parent_manifests_sha256"
            ].values()
        )
    ):
        ranking_inputs.extend(
            (
                InputDigest(
                    f"source.collect_manifest.{digest}",
                    digest,
                    1,
                ),
                InputDigest(
                    f"source.collect_index.{digest}",
                    f"{index + 1}" * 64,
                    1,
                ),
            )
        )
    for index, condition in enumerate(("none", "rope")):
        digest = verify_file(activation_files[condition]).digest
        ranking_inputs.append(
            InputDigest(
                f"source.activation.{index:064x}",
                digest.sha256,
                digest.size_bytes,
            )
        )
    ranking_manifest = new_manifest(
        command="rank-condition-shift",
        model_family=parent.model_family,
        model_revision=parent.model_revision,
        training_code_sha=parent.training_code_sha,
        model_code_sha=parent.model_code_sha,
        analysis_code_sha=fixture.head,
        configuration=FileDigest("7" * 64, 1),
        checkpoint=parent.checkpoint,
        dataset_manifest=parent.dataset_manifest,
        inputs=tuple(sorted(ranking_inputs, key=lambda item: item.role)),
        condition=parent.condition,
        sites=parent.sites,
        seed=parent.seed,
        artifacts=(
            ArtifactDigest(
                "selection.json",
                selection_digest.sha256,
                selection_digest.size_bytes,
            ),
        ),
        created_at_utc="2026-08-09T20:00:00Z",
    )
    ranking_manifest_path = ranking_dir / "manifest.json"
    ranking_manifest_path.write_text(
        json.dumps(ranking_manifest.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    ranking_manifest_digest = verify_file(ranking_manifest_path).digest

    checkpoint_sha256 = verify_file(fixture.reference_checkpoint).digest.sha256
    attestation = tmp_path / "ranking-paired-source-attestation.json"
    attestation.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_kind": "clean-git-checkout",
                "source_condition": "rope",
                "model_code_sha": fixture.head,
                "checkpoint_sha256": checkpoint_sha256,
                "inference_contract_sha256": source_lineage[
                    "inference_contract_sha256"
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config["provenance"]["condition"] = "none"
    config["checkpoint_study"] = {
        "scope": "exploratory_pilot",
        "recipient_chain": None,
        "source_chain": None,
    }
    config["dataset"].update(
        {
            "dataset_id": "pilot-dataset",
            "roster_split": "discovery",
            "evaluation_split": "val",
            "sample_roster_path": str(roster),
            "expected_sample_roster_sha256": roster_digest.sha256,
        }
    )
    config["intervention"].update(
        {
            "target_features": [0],
            "control_features": [1],
            "latent_baseline": 0.0,
            "freeze_artifact_path": None,
            "matched_control_dose": (
                "per_call_decoded_rms_clip_to_smaller_without_amplification"
            ),
        }
    )
    config["paired_reverse_patch"] = {
        "source_condition": "rope",
        "source_checkpoint_path": str(fixture.reference_checkpoint),
        "expected_source_checkpoint_sha256": checkpoint_sha256,
        "source_model_code_root": str(fixture.repository),
        "expected_source_model_code_sha": fixture.head,
        "source_code_attestation_path": str(attestation),
        "expected_source_code_attestation_sha256": verify_file(
            attestation
        ).digest.sha256,
        "maximum_symmetric_donor_shift_rms_ratio": 1.25,
    }
    config["ranking_parent"] = {
        "run_dir": str(ranking_dir),
        "expected_manifest_sha256": ranking_manifest_digest.sha256,
        "expected_selection_sha256": selection_digest.sha256,
        "split_protocol_path": str(split_protocol),
        "expected_split_protocol_sha256": split_digest.sha256,
        "dose_protocol_path": str(dose_protocol),
        "expected_dose_protocol_sha256": dose_digest.sha256,
        "directions": ["rope_to_none", "none_to_rope"],
    }
    fixture.config.write_text(
        json.dumps(config, sort_keys=True), encoding="utf-8"
    )
    return SimpleNamespace(
        ranking_dir=ranking_dir,
        selection_path=selection_path,
        split_protocol=split_protocol,
        dose_protocol=dose_protocol,
        roster=roster,
        attestation=attestation,
    )


def _reseal_ranking_selection_and_split(
    fixture: SimpleNamespace, ranking: SimpleNamespace
) -> None:
    split_digest = verify_file(ranking.split_protocol).digest
    dose_protocol = json.loads(ranking.dose_protocol.read_text(encoding="utf-8"))
    dose_protocol["parent_split_protocol_sha256"] = split_digest.sha256
    ranking.dose_protocol.write_text(
        json.dumps(dose_protocol, sort_keys=True), encoding="utf-8"
    )
    dose_digest = verify_file(ranking.dose_protocol).digest
    selection = json.loads(ranking.selection_path.read_text(encoding="utf-8"))
    selection["split_protocol_sha256"] = split_digest.sha256
    selection["dose_protocol_sha256"] = dose_digest.sha256
    selection["ranking_protocol_sha256"] = official_causal._canonical_sha256(
        selection["ranking_protocol"]
    )
    ranking.selection_path.write_text(
        json.dumps(selection, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    selection_digest = verify_file(ranking.selection_path).digest

    manifest_path = ranking.ranking_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection_artifact = next(
        item for item in manifest["artifacts"] if item["name"] == "selection.json"
    )
    selection_artifact.update(
        {
            "sha256": selection_digest.sha256,
            "size_bytes": selection_digest.size_bytes,
        }
    )
    split_input = next(
        item
        for item in manifest["inputs"]
        if item["role"] == "ranking.split_protocol"
    )
    split_input.update(
        {"sha256": split_digest.sha256, "size_bytes": split_digest.size_bytes}
    )
    dose_input = next(
        item
        for item in manifest["inputs"]
        if item["role"] == "ranking.dose_protocol"
    )
    dose_input.update(
        {"sha256": dose_digest.sha256, "size_bytes": dose_digest.size_bytes}
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config["ranking_parent"].update(
        {
            "expected_manifest_sha256": verify_file(manifest_path).digest.sha256,
            "expected_selection_sha256": selection_digest.sha256,
            "expected_split_protocol_sha256": split_digest.sha256,
            "expected_dose_protocol_sha256": dose_digest.sha256,
        }
    )
    fixture.config.write_text(
        json.dumps(config, sort_keys=True), encoding="utf-8"
    )


def _enable_heldout_select_features(
    fixture: SimpleNamespace, tmp_path: Path
) -> SimpleNamespace:
    attestation = _enable_validation_paired_reverse_patch(fixture, tmp_path)
    heldout_roster = tmp_path / "heldout-sample-roster.json"
    heldout_roster.write_text(
        json.dumps(
            {
                "dataset_id": "toy-official",
                "split": "test",
                "row_indices": [0, 1],
                "sample_ids": ["test-0", "test-1"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    roster_digest = verify_file(heldout_roster).digest
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config["dataset"].update(
        {
            "dataset_id": "toy-official",
            "roster_split": "held_out",
            "evaluation_split": "test",
            "sample_roster_path": str(heldout_roster),
            "expected_sample_roster_sha256": roster_digest.sha256,
        }
    )

    validation_ids = ["selection-dataset"] + [
        f"validation-{index:02d}" for index in range(1, 8)
    ]
    validation_fingerprints = {}
    for dataset_id in validation_ids:
        raw_digest = hashlib.sha256(
            f"raw:{dataset_id}".encode("utf-8")
        ).hexdigest()
        prediction_digest = hashlib.sha256(
            f"prediction:{dataset_id}".encode("utf-8")
        ).hexdigest()
        validation_fingerprints[dataset_id] = {
            "raw_dataset_content_sha256": raw_digest,
            "invariant_prediction_content_sha256": prediction_digest,
            "combined_dataset_fingerprint_sha256": (
                official_causal._canonical_sha256(
                    {
                        "raw_dataset_content_sha256": raw_digest,
                        "invariant_prediction_content_sha256": prediction_digest,
                    }
                )
            ),
        }
    validation_fingerprint_manifest_sha256 = official_causal._canonical_sha256(
        validation_fingerprints
    )
    heldout_ids = [f"heldout-{index:02d}" for index in range(1, 8)] + [
        "toy-official"
    ]
    roster_mapping = {
        dataset_id: (
            roster_digest.sha256
            if dataset_id == "toy-official"
            else hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()
        )
        for dataset_id in heldout_ids
    }
    recipient_stage = {
        "study_id": "formal-study",
        "mode": "temporary",
        "stage": "stage1",
        "terminal_step": 500_000,
        "checkpoint_sha256": verify_file(fixture.checkpoint).digest.sha256,
    }
    source_stage = {
        **recipient_stage,
        "mode": "rope",
        "checkpoint_sha256": verify_file(
            fixture.reference_checkpoint
        ).digest.sha256,
    }
    checkpoint_study_payload = {
        "schema_version": 1,
        "scope": "formal",
        "formal_trust_verified": True,
        "model_evidence_scope": "intermediate-stage-specific",
        "direction": {
            "source_condition": "rope",
            "recipient_condition": "temporary",
        },
        "recipient_chain": [recipient_stage],
        "source_chain": [source_stage],
    }
    checkpoint_study_sha256 = official_causal._canonical_sha256(
        checkpoint_study_payload
    )
    source_binding = {
        "model_sha": fixture.head,
        "checkpoint_sha256": verify_file(
            fixture.reference_checkpoint
        ).digest.sha256,
        "code_attestation_sha256": verify_file(attestation).digest.sha256,
        "inference_contract_sha256": fixture.source_lineage[
            "inference_contract_sha256"
        ],
        "representation_source_lineage_sha256": (
            official_causal._canonical_sha256(fixture.source_lineage)
        ),
    }
    direction = {
        "source_condition": "rope",
        "recipient_condition": "temporary",
    }
    common_lineage = {
        "model_family": "tabicl-v2",
        "model_revision": "step-210000",
        "training_code_sha": fixture.head,
        "model_code_sha": fixture.head,
        "parent_analysis_code_sha": fixture.head,
        "checkpoint_sha256": verify_file(fixture.checkpoint).digest.sha256,
        "dataset_manifest_sha256": verify_file(
            fixture.dataset_manifest
        ).digest.sha256,
        "condition": "temporary",
        "site": "row_interactor",
        "random_seed": 42,
        "representation_model_sha256": verify_file(
            fixture.representation
        ).digest.sha256,
        "representation_parent_manifest_sha256": verify_file(
            fixture.parent_dir / "manifest.json"
        ).digest.sha256,
        "inference_contract_sha256": fixture.source_lineage[
            "inference_contract_sha256"
        ],
        "representation_source_lineage": fixture.source_lineage,
        "paired_source_direction": direction,
        "paired_source_binding": source_binding,
        "checkpoint_study_sha256": checkpoint_study_sha256,
    }
    evidence_family = official_causal._SELECTION_EVIDENCE_FAMILY
    component_p = 1.0 / 256.0
    source_advantage = {
        "name": "source_advantage",
        "metric": "intersection_source_native_and_no_op_advantage",
        "direction": "both_positive",
        "scope": "global",
        "effective_dataset_count": 8,
        "dataset_effects": [
            {
                "dataset_id": dataset_id,
                "source_native_advantage": 1.0,
                "source_no_op_advantage": 1.0,
            }
            for dataset_id in validation_ids
        ],
        "components": [
            {
                "name": "source_native_advantage",
                "metric": (
                    "mean_log_loss_improvement_of_source_native_vs_recipient_native"
                ),
                "direction": "positive",
                "mean_effect": 1.0,
                "median_effect": 1.0,
                "confidence_low": 1.0,
                "confidence_high": 1.0,
                "positive_fraction": 1.0,
                "p_value": component_p,
                "passes_direction_ci_replication": True,
            },
            {
                "name": "source_no_op_advantage",
                "metric": (
                    "mean_log_loss_improvement_of_source_no_op_vs_recipient_no_op"
                ),
                "direction": "positive",
                "mean_effect": 1.0,
                "median_effect": 1.0,
                "confidence_low": 1.0,
                "confidence_high": 1.0,
                "positive_fraction": 1.0,
                "p_value": component_p,
                "passes_direction_ci_replication": True,
            },
        ],
        "minimum_component_mean_effect": 1.0,
        "composite_p_value": component_p,
        "direction_ci_replication_gates_passed": True,
        "p_value_gate_deferred_to_candidate_composite": True,
    }
    hypothesis_means = {
        "target_damage": 0.5,
        "ablation_specificity": 0.4,
        "donor_rescue": 0.3,
        "donor_specificity": 0.2,
    }
    hypotheses = [
        {
            **definition,
            "effective_dataset_count": 8,
            "mean_effect": hypothesis_means[definition["name"]],
            "median_effect": hypothesis_means[definition["name"]],
            "confidence_low": hypothesis_means[definition["name"]],
            "confidence_high": hypothesis_means[definition["name"]],
            "positive_fraction": 1.0,
            "p_value": component_p,
            "passes_direction_ci_replication": True,
        }
        for definition in evidence_family[:-1]
    ]
    candidate = {
        "candidate_id": "candidate-0",
        "target_features": [0],
        "control_features": [1],
        "latent_baseline": 0.0,
        "effective_dataset_count": 8,
        "dataset_effects": [
            {
                "dataset_id": dataset_id,
                "target_damage": 0.5,
                "matched_control_damage": 0.1,
                "ablation_specificity": 0.4,
                "donor_rescue": 0.3,
                "donor_specificity": 0.2,
                "donor_shift_balance_ratio": 1.0,
            }
            for dataset_id in validation_ids
        ],
        "hypotheses": hypotheses,
        "minimum_mean_evidence": 0.2,
        "intersection_union_composite_p_value": component_p,
        "by_adjusted_composite_p_value": component_p,
        "all_direction_ci_replication_gates_passed": True,
        "eligible": True,
        "selected": True,
    }
    family_size = 1
    harmonic_factor = sum(1.0 / index for index in range(1, family_size + 1))
    confirmation = {
        "alpha": 0.05,
        "confidence_level": 0.95,
        "sign_flip_resamples": 1_000,
        "bootstrap_resamples": 1_000,
        "bootstrap_method": "paired-dataset-bootstrap",
        "random_seed": 42,
        "minimum_positive_fraction": 0.75,
        "minimum_heldout_datasets": 8,
        "required_heldout_datasets": 8,
        "maximum_confirmation_candidates": 1,
        "frozen_candidate_count": 1,
        "preregistered_holm_rank_one_threshold": 0.05,
        "actual_holm_rank_one_threshold": 0.05,
        "sign_flip_minimum_p_value": 1.0 / 256.0,
        "sign_flip_mode": "exact-enumeration",
        "candidate_test": "intersection-union-max-p",
        "multiplicity_method": "holm",
        "top_k_after_freeze": False,
    }
    selection = {
        "schema_version": 1,
        "analysis": "validation-feature-selection",
        "evidence_scope": "validation-selection",
        "condition": "temporary",
        "site": "row_interactor",
        "random_seed": 42,
        "evidence_family": evidence_family,
        "statistics": {
            "multiplicity_method": "benjamini-yekutieli",
            "fdr_control_unit": "candidate-intersection-union-hypothesis",
            "candidate_composite_method": "maximum-of-six-component-p-values",
            "component_count_per_candidate": 6,
            "fdr_alpha": 0.05,
            "confidence_level": 0.95,
            "bootstrap_method": "paired-dataset-bootstrap",
            "bootstrap_resamples": 1_000,
            "p_value_method": "one-sided-paired-sign-flip",
            "sign_flip_resamples": 1_000,
            "minimum_validation_datasets": 8,
            "minimum_positive_fraction": 0.75,
            "maximum_selections": 1,
            "ranking_rule": "maximum_minimum_mean_evidence",
            "family_size": family_size,
            "harmonic_factor": harmonic_factor,
            "rank_one_threshold": 0.05 / (family_size * harmonic_factor),
            "exact_sign_flip_minimum_p_value": component_p,
            "required_validation_datasets": 8,
            "effective_validation_dataset_count": 8,
            "sign_flip_mode": "exact-enumeration",
            "monte_carlo_minimum_p_value": None,
        },
        "source_advantage_prerequisite": source_advantage,
        "common_lineage": common_lineage,
        "validation_dataset_ids": validation_ids,
        "validation_dataset_fingerprints_sha256": validation_fingerprints,
        "validation_dataset_fingerprint_manifest_sha256": (
            validation_fingerprint_manifest_sha256
        ),
        "heldout": {
            "dataset_ids": heldout_ids,
            "evaluation_sample_rosters_sha256": roster_mapping,
            "confirmation_protocol": confirmation,
        },
        "candidate_results": [candidate],
        "selected_candidates": ["candidate-0"],
    }
    selection_dir = tmp_path / "select-features-run"
    selection_dir.mkdir()
    selection_path = selection_dir / "selection.json"
    selection_path.write_text(
        json.dumps(selection, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    selection_sha256 = verify_file(selection_path).digest.sha256
    frozen = [
        {
            "candidate_id": "candidate-0",
            "target_features": [0],
            "control_features": [1],
            "latent_baseline": 0.0,
        }
    ]
    freeze = {
        "schema_version": 2,
        "evidence_scope": "validation-frozen",
        "condition": "temporary",
        "site": "row_interactor",
        "random_seed": 42,
        "evaluation_sample_rosters_sha256": roster_mapping,
        "validation_dataset_fingerprints_sha256": validation_fingerprints,
        "validation_dataset_fingerprint_manifest_sha256": (
            validation_fingerprint_manifest_sha256
        ),
        "selected_interventions": frozen,
        "representation_model_sha256": common_lineage[
            "representation_model_sha256"
        ],
        "representation_parent_manifest_sha256": common_lineage[
            "representation_parent_manifest_sha256"
        ],
        "model_sha": fixture.head,
        "checkpoint_sha256": common_lineage["checkpoint_sha256"],
        "inference_contract_sha256": common_lineage[
            "inference_contract_sha256"
        ],
        "paired_source_direction": direction,
        "paired_source_binding": source_binding,
        "maximum_symmetric_donor_shift_rms_ratio": 1.25,
        "source_advantage_prerequisite": source_advantage,
        "confirmation_protocol": confirmation,
        "representation_source_lineage_sha256": (
            official_causal._canonical_sha256(fixture.source_lineage)
        ),
        "validation_selection_sha256": selection_sha256,
        "checkpoint_study_sha256": checkpoint_study_sha256,
    }
    summary_lineage_names = (
        "model_code_sha",
        "checkpoint_sha256",
        "representation_model_sha256",
        "representation_parent_manifest_sha256",
        "inference_contract_sha256",
        "paired_source_direction",
        "paired_source_binding",
        "checkpoint_study_sha256",
    )
    summary = {
        "schema_version": 1,
        "analysis": "validation-feature-selection",
        "evidence_scope": "validation-frozen",
        "condition": "temporary",
        "site": "row_interactor",
        "random_seed": 42,
        "evidence_family": evidence_family,
        "source_advantage_prerequisite": source_advantage,
        "confirmation_protocol": confirmation,
        "selected_interventions": frozen,
        "selection_count": 1,
        "validation_dataset_ids": validation_ids,
        "validation_dataset_fingerprints_sha256": validation_fingerprints,
        "validation_dataset_fingerprint_manifest_sha256": (
            validation_fingerprint_manifest_sha256
        ),
        "heldout_dataset_ids": heldout_ids,
        "evaluation_sample_rosters_sha256": roster_mapping,
        "validation_selection_sha256": selection_sha256,
        "common_lineage": {
            name: common_lineage[name] for name in summary_lineage_names
        },
        "checkpoint_study_sha256": checkpoint_study_sha256,
    }
    for name, payload in (("freeze.json", freeze), ("summary.json", summary)):
        (selection_dir / name).write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    artifacts = tuple(
        ArtifactDigest(name, digest.sha256, digest.size_bytes)
        for name in ("freeze.json", "selection.json", "summary.json")
        for digest in [verify_file(selection_dir / name).digest]
    )
    roster_inputs = tuple(
        InputDigest(
            role=f"heldout.sample_roster.{digest}",
            sha256=digest,
            size_bytes=(
                roster_digest.size_bytes
                if dataset_id == "toy-official"
                else 1
            ),
        )
        for dataset_id, digest in roster_mapping.items()
    )
    manifest = new_manifest(
        command="select-features",
        model_family="tabicl-v2",
        model_revision="step-210000",
        training_code_sha=fixture.head,
        model_code_sha=fixture.head,
        analysis_code_sha=fixture.head,
        configuration=FileDigest("6" * 64, 1),
        checkpoint=verify_file(fixture.checkpoint).digest,
        dataset_manifest=verify_file(fixture.dataset_manifest).digest,
        inputs=tuple(sorted(roster_inputs, key=lambda item: item.role)),
        condition="temporary",
        sites=("row_interactor",),
        seed=42,
        artifacts=artifacts,
        created_at_utc="2026-08-07T13:00:00Z",
    )
    manifest_path = selection_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_digest = verify_file(manifest_path).digest
    config["intervention"].update(
        {
            "freeze_artifact_path": str(selection_dir / "freeze.json"),
            "expected_freeze_artifact_sha256": verify_file(
                selection_dir / "freeze.json"
            ).digest.sha256,
            "selection_run_dir": str(selection_dir),
            "expected_selection_manifest_sha256": manifest_digest.sha256,
        }
    )
    fixture.config.write_text(
        json.dumps(config, sort_keys=True), encoding="utf-8"
    )
    return SimpleNamespace(
        selection_dir=selection_dir,
        selection=selection_path,
        freeze=selection_dir / "freeze.json",
        summary=selection_dir / "summary.json",
        manifest=manifest_path,
        roster=heldout_roster,
    )


def test_model_causal_workflow_publishes_verified_path_free_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    observed = _install_workflow_fakes(fixture, monkeypatch)

    assert (
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
        == 0
    )

    manifest = verify_run_directory(fixture.output)
    assert manifest.command == "model-causal"
    assert manifest.evidence_level == "strict"
    assert [artifact.name for artifact in manifest.artifacts] == [
        "predictions.json",
        "summary.json",
    ]
    roles = {item.role for item in manifest.inputs}
    assert {
        "representation.parent_manifest",
        "representation.model",
        "samples.roster",
        "talent.raw.info.json",
        "talent.raw.N_val.npy",
    } <= roles
    predictions = json.loads(
        (fixture.output / "predictions.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (fixture.output / "summary.json").read_text(encoding="utf-8")
    )
    assert predictions["sample_ids"] == ["val-0"]
    assert set(predictions["conditions"]) == {
        "matched_random_edit",
        "no_op_reconstruction",
        "roundtrip_restore_control",
        "target_baseline_edit",
    }
    assert summary["fit_split"] == "train"
    assert summary["evaluation_split"] == "val"
    assert summary["roster_split"] == "validation"
    assert summary["evidence_scope"] == "exploratory-feature-selection"
    assert predictions["evidence_scope"] == "exploratory-feature-selection"
    assert summary["mechanistic_rescue"] == {
        "passed": False,
        "status": "paired_rescue_not_run",
        "roundtrip_restore_control_passed": True,
    }
    assert summary["no_op_gates"]["passed"] is True
    assert summary["no_op_gates"]["thresholds"] == {
        "max_no_op_accuracy_difference": 0.005,
        "max_no_op_probability_deviation": 0.02,
        "max_no_op_reconstruction_mse": 0.01,
    }
    assert summary["official_scope"]["feature_group"] == "same"
    assert summary["official_scope"]["maximum_native_classes"] == 10
    checkpoints = summary["representation_source_lineage"][
        "condition_checkpoints_sha256"
    ]
    assert checkpoints["temporary"] == verify_file(
        fixture.checkpoint
    ).digest.sha256
    assert checkpoints["rope"] == verify_file(
        fixture.reference_checkpoint
    ).digest.sha256
    assert checkpoints["temporary"] != checkpoints["rope"]
    assert observed["fit_split"] == "train"
    assert observed["fit_rows"] == 4
    assert observed["representation_digest"] == hashlib.sha256(
        b"verified representation checkpoint"
    ).hexdigest()
    published = json.dumps(
        {
            "manifest": manifest.to_dict(),
            "predictions": predictions,
            "summary": summary,
        }
    )
    assert str(tmp_path) not in published
    assert not list(tmp_path.glob(".published-model-causal.*.staging"))


def test_model_causal_workflow_runs_independent_paired_reverse_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    attestation = _enable_validation_paired_reverse_patch(fixture, tmp_path)
    _install_workflow_fakes(fixture, monkeypatch)

    assert (
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
        == 0
    )

    manifest = verify_run_directory(fixture.output)
    roles = {item.role: item.sha256 for item in manifest.inputs}
    assert roles["paired_source.checkpoint"] == verify_file(
        fixture.reference_checkpoint
    ).digest.sha256
    assert roles["paired_source.code_attestation"] == verify_file(
        attestation
    ).digest.sha256
    predictions = json.loads(
        (fixture.output / "predictions.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (fixture.output / "summary.json").read_text(encoding="utf-8")
    )
    assert {
        "paired_reverse_patch",
        "paired_matched_random_patch",
    } <= set(predictions["conditions"])
    measured = predictions["paired_reverse_patch_measured_effects"]
    assert set(measured) == {
        "log_loss_improvement_vs_recipient_no_op",
        "log_loss_improvement_vs_paired_matched_random_patch",
        "log_loss_improvement_vs_target_baseline_edit",
        "native_distance_reduction_vs_target_baseline_edit",
        "log_loss_improvement_of_source_native_vs_recipient_native",
        "log_loss_improvement_of_source_no_op_vs_recipient_no_op",
    }
    assert all(len(values) == 1 for values in measured.values())
    paired = summary["paired_reverse_patch"]
    assert paired["status"] == "measured_diagnostic_only"
    assert paired["direction"] == {
        "source_condition": "rope",
        "recipient_condition": "temporary",
    }
    assert paired["alignment"]["verified"] is True
    assert paired["source_binding"]["checkpoint_sha256"] == verify_file(
        fixture.reference_checkpoint
    ).digest.sha256
    assert set(paired["measured_effects"]) == {
        "mean_log_loss_improvement_vs_recipient_no_op",
        "mean_log_loss_improvement_vs_paired_matched_random_patch",
        "mean_log_loss_improvement_vs_target_baseline_edit",
        "mean_native_distance_reduction_vs_target_baseline_edit",
        "mean_log_loss_improvement_of_source_native_vs_recipient_native",
        "mean_log_loss_improvement_of_source_no_op_vs_recipient_no_op",
        "donor_patch_gap_closure_denominator",
        "donor_patch_gap_closure_fraction",
    }
    assert "matched_control_dose" not in paired
    assert "ranking_dose_protocol_sha256" not in summary["input_bindings"]
    assert summary["intervention"] == {
        "target_features": [0],
        "control_features": [1],
        "latent_baseline": 0.0,
        "random_seed": 42,
    }
    assert paired["source_native"] is not None
    assert paired["source_no_op_gates"]["passed"] is True
    assert paired["donor_shift_balance"]["passed"] is True
    assert paired["ablation_displacement_balance"]["passed"] is True
    assert paired["donor_displacement_balance"]["passed"] is True
    assert summary["mechanistic_rescue"]["passed"] is False


def test_exploratory_paired_workflow_binds_completed_ranking_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    ranking = _enable_exploratory_ranking_paired(fixture, tmp_path)
    _install_workflow_fakes(fixture, monkeypatch, driver_temporary=False)

    assert official_causal.run(
        SimpleNamespace(config=fixture.config, output_dir=fixture.output)
    ) == 0

    manifest = verify_run_directory(fixture.output)
    roles = {item.role: item.sha256 for item in manifest.inputs}
    assert roles["ranking.parent_manifest"] == verify_file(
        ranking.ranking_dir / "manifest.json"
    ).digest.sha256
    assert roles["ranking.selection"] == verify_file(
        ranking.selection_path
    ).digest.sha256
    assert roles["ranking.split_protocol"] == verify_file(
        ranking.split_protocol
    ).digest.sha256
    assert roles["ranking.dose_protocol"] == verify_file(
        ranking.dose_protocol
    ).digest.sha256
    summary = json.loads(
        (fixture.output / "summary.json").read_text(encoding="utf-8")
    )
    predictions = json.loads(
        (fixture.output / "predictions.json").read_text(encoding="utf-8")
    )
    assert summary["ranking_parent"] == {
        "manifest_sha256": roles["ranking.parent_manifest"],
        "selection_sha256": roles["ranking.selection"],
        "split_protocol_sha256": roles["ranking.split_protocol"],
        "split_protocol_id": "toy-whole-row-split",
        "dose_protocol_sha256": verify_file(ranking.dose_protocol).digest.sha256,
        "dose_protocol_id": (
            "tabicl-step250k-whole-row-causal-dose-amendment-v1"
        ),
        "target_features": [0],
        "control_features": [1],
        "directions": ["rope_to_none", "none_to_rope"],
        "effective_direction": "rope_to_none",
        "matched_control_dose": (
            "per_call_decoded_rms_clip_to_smaller_without_amplification"
        ),
        "evidence_scope": "exploratory-pilot",
        "formal_claim": "forbidden",
    }
    assert summary["paired_reverse_patch"]["status"] == (
        "measured_exploratory_diagnostic_only"
    )
    dose_mode = "per_call_decoded_rms_clip_to_smaller_without_amplification"
    assert summary["intervention"] == {
        "target_features": [0],
        "control_features": [1],
        "latent_baseline": 0.0,
        "random_seed": 42,
        "matched_control_dose": dose_mode,
        "edit_semantics": "per_call_dose_matched_partial_edit",
    }
    paired = summary["paired_reverse_patch"]
    assert paired["matched_control_dose"] == dose_mode
    assert summary["input_bindings"]["ranking_dose_protocol_sha256"] == (
        verify_file(ranking.dose_protocol).digest.sha256
    )
    assert set(paired["measured_effects"]) == {
        "mean_log_loss_improvement_vs_recipient_no_op",
        "mean_log_loss_improvement_vs_paired_matched_random_patch",
        "mean_log_loss_improvement_of_source_native_vs_recipient_native",
        "mean_log_loss_improvement_of_source_no_op_vs_recipient_no_op",
        "donor_patch_gap_closure_denominator",
        "donor_patch_gap_closure_fraction",
        "target_ablation_and_donor_patch_dose_comparable",
    }
    assert paired["measured_effects"][
        "target_ablation_and_donor_patch_dose_comparable"
    ] is False
    assert set(predictions["paired_reverse_patch_measured_effects"]) == {
        "log_loss_improvement_vs_recipient_no_op",
        "log_loss_improvement_vs_paired_matched_random_patch",
        "log_loss_improvement_of_source_native_vs_recipient_native",
        "log_loss_improvement_of_source_no_op_vs_recipient_no_op",
    }
    assert summary["checkpoint_study"]["formal_trust_verified"] is False


def test_exploratory_paired_workflow_requires_ranking_parent(
    tmp_path: Path
) -> None:
    fixture = _workflow_fixture(tmp_path)
    _enable_exploratory_ranking_paired(fixture, tmp_path)
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config.pop("ranking_parent")
    fixture.config.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="requires a strict ranking_parent"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dose_metric", "a different but non-empty metric"),
        ("protocol_id", "renamed-dose-amendment"),
    ],
)
def test_exploratory_ranking_rejects_resealed_dose_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    fixture = _workflow_fixture(tmp_path)
    ranking = _enable_exploratory_ranking_paired(fixture, tmp_path)
    _install_workflow_fakes(fixture, monkeypatch, driver_temporary=False)
    dose = json.loads(ranking.dose_protocol.read_text(encoding="utf-8"))
    dose[field] = value
    ranking.dose_protocol.write_text(
        json.dumps(dose, sort_keys=True), encoding="utf-8"
    )
    _reseal_ranking_selection_and_split(fixture, ranking)

    with pytest.raises(ValueError, match="frozen contract"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_exploratory_ranking_requires_frozen_dose_mode(
    tmp_path: Path
) -> None:
    fixture = _workflow_fixture(tmp_path)
    _enable_exploratory_ranking_paired(fixture, tmp_path)
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config["intervention"].pop("matched_control_dose")
    fixture.config.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="requires frozen per-call dose matching"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target_features", [2], "target_features differ"),
        ("control_features", [2], "control_features differ"),
    ],
)
def test_exploratory_paired_workflow_rejects_config_selection_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: list[int],
    message: str,
) -> None:
    fixture = _workflow_fixture(tmp_path)
    _enable_exploratory_ranking_paired(fixture, tmp_path)
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config["intervention"][field] = value
    fixture.config.write_text(json.dumps(config), encoding="utf-8")
    _install_workflow_fakes(fixture, monkeypatch, driver_temporary=False)

    with pytest.raises(ValueError, match=message):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_exploratory_paired_workflow_rejects_direction_order_drift(
    tmp_path: Path
) -> None:
    fixture = _workflow_fixture(tmp_path)
    _enable_exploratory_ranking_paired(fixture, tmp_path)
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config["ranking_parent"]["directions"] = [
        "none_to_rope",
        "rope_to_none",
    ]
    fixture.config.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen bidirectional ordering"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_exploratory_ranking_requires_expected_sample_roster_digest(
    tmp_path: Path
) -> None:
    fixture = _workflow_fixture(tmp_path)
    _enable_exploratory_ranking_paired(fixture, tmp_path)
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config["dataset"].pop("expected_sample_roster_sha256")
    fixture.config.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="expected_sample_roster_sha256"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_exploratory_ranking_rejects_sample_roster_outside_frozen_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    ranking = _enable_exploratory_ranking_paired(fixture, tmp_path)
    ranking.roster.write_text(
        ranking.roster.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config["dataset"]["expected_sample_roster_sha256"] = verify_file(
        ranking.roster
    ).digest.sha256
    fixture.config.write_text(json.dumps(config), encoding="utf-8")
    _install_workflow_fakes(fixture, monkeypatch, driver_temporary=False)

    with pytest.raises(ValueError, match="frozen split protocol"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_exploratory_ranking_rejects_truncated_control_candidate_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    ranking = _enable_exploratory_ranking_paired(fixture, tmp_path)
    selection = json.loads(ranking.selection_path.read_text(encoding="utf-8"))
    selection["ranking_protocol"]["random_candidate_pool_size"] = 3
    ranking.selection_path.write_text(
        json.dumps(selection, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    split = json.loads(ranking.split_protocol.read_text(encoding="utf-8"))
    split["feature_protocol"]["matched_control"] = (
        "activation-frequency and log-decoder-norm nearest-neighbour pool of "
        "size 3, sampled once with seed 42 without replacement"
    )
    ranking.split_protocol.write_text(
        json.dumps(split, sort_keys=True), encoding="utf-8"
    )
    _reseal_ranking_selection_and_split(fixture, ranking)
    _install_workflow_fakes(fixture, monkeypatch, driver_temporary=False)

    with pytest.raises(ValueError, match="would be truncated"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_exploratory_ranking_rejects_outcome_or_unknown_parent_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    ranking = _enable_exploratory_ranking_paired(fixture, tmp_path)
    manifest_path = ranking.ranking_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"].append(
        {
            "role": "outcome.predictions",
            "sha256": "9" * 64,
            "size_bytes": 1,
        }
    )
    manifest["inputs"].sort(key=lambda item: item["role"])
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config["ranking_parent"]["expected_manifest_sha256"] = verify_file(
        manifest_path
    ).digest.sha256
    fixture.config.write_text(json.dumps(config), encoding="utf-8")
    _install_workflow_fakes(fixture, monkeypatch, driver_temporary=False)

    with pytest.raises(ValueError, match="unknown input roles"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_ranking_parent_schema_selects_one_discovery_collect_per_condition(
    tmp_path: Path,
) -> None:
    fixture = _workflow_fixture(tmp_path)
    ranking = _enable_exploratory_ranking_paired(fixture, tmp_path)
    parent = load_verified_run_manifest(ranking.ranking_dir / "manifest.json")
    source_lineage = copy.deepcopy(fixture.source_lineage)
    source_lineage["collect_parent_manifests_sha256"]["none"].append(
        "1" * 64
    )
    source_lineage["collect_parent_manifests_sha256"]["rope"].append(
        "2" * 64
    )

    official_causal._validate_ranking_parent_input_schema(
        parent,
        source_lineage=source_lineage,
        ranking_dataset_count=1,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda lineage, selected: lineage[
            "collect_parent_manifests_sha256"
        ]["none"].remove(selected["none"]),
        lambda lineage, selected: lineage[
            "collect_parent_manifests_sha256"
        ]["rope"].append(selected["none"]),
    ],
)
def test_ranking_parent_schema_rejects_missing_or_ambiguous_condition_collect(
    tmp_path: Path,
    mutation,
) -> None:
    fixture = _workflow_fixture(tmp_path)
    ranking = _enable_exploratory_ranking_paired(fixture, tmp_path)
    parent = load_verified_run_manifest(ranking.ranking_dir / "manifest.json")
    source_lineage = copy.deepcopy(fixture.source_lineage)
    selected = {
        condition: values[0]
        for condition, values in source_lineage[
            "collect_parent_manifests_sha256"
        ].items()
    }
    source_lineage["collect_parent_manifests_sha256"]["none"].append(
        "1" * 64
    )
    source_lineage["collect_parent_manifests_sha256"]["rope"].append(
        "2" * 64
    )
    mutation(source_lineage, selected)

    with pytest.raises(ValueError, match="exactly one registered collect parent"):
        official_causal._validate_ranking_parent_input_schema(
            parent,
            source_lineage=source_lineage,
            ranking_dataset_count=1,
        )


def test_model_causal_paired_reverse_patch_rejects_resealed_bad_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    attestation = _enable_validation_paired_reverse_patch(fixture, tmp_path)
    payload = json.loads(attestation.read_text(encoding="utf-8"))
    payload["source_condition"] = "none"
    attestation.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config["paired_reverse_patch"][
        "expected_source_code_attestation_sha256"
    ] = verify_file(attestation).digest.sha256
    fixture.config.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    _install_workflow_fakes(fixture, monkeypatch)

    with pytest.raises(ValueError, match="attestation differs"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_model_causal_workflow_rejects_tampered_parent_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    _install_workflow_fakes(fixture, monkeypatch)
    fixture.representation.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="hash or size mismatch"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )

    assert not fixture.output.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda lineage: lineage["condition_checkpoints_sha256"].__setitem__(
                "temporary", "0" * 64
            ),
            "causal checkpoint differs",
        ),
        (
            lambda lineage: lineage[
                "collect_parent_manifests_sha256"
            ].__setitem__("temporary", ["a" * 64]),
            "collect-parent lineage differs",
        ),
        (
            lambda lineage: lineage.__setitem__(
                "inference_contract_sha256", "0" * 64
            ),
            "inference contract differs",
        ),
    ],
)
def test_model_causal_rejects_tampered_multi_condition_source_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    fixture = _workflow_fixture(tmp_path)
    mutation(fixture.source_lineage)
    _install_workflow_fakes(fixture, monkeypatch)

    with pytest.raises(ValueError, match=message):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_model_causal_rejects_checkpoint_identity_mode_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    _install_workflow_fakes(
        fixture, monkeypatch, driver_temporary=False
    )

    with pytest.raises(RuntimeError, match="identity mode differs"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def _reseal_selection_artifact(
    fixture: SimpleNamespace,
    selection_bundle: SimpleNamespace,
    artifact_name: str,
) -> None:
    artifact_path = selection_bundle.selection_dir / artifact_name
    artifact_digest = verify_file(artifact_path).digest
    manifest = json.loads(
        selection_bundle.manifest.read_text(encoding="utf-8")
    )
    declared = next(
        item for item in manifest["artifacts"] if item["name"] == artifact_name
    )
    declared["sha256"] = artifact_digest.sha256
    declared["size_bytes"] = artifact_digest.size_bytes
    selection_bundle.manifest.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config["intervention"]["expected_selection_manifest_sha256"] = verify_file(
        selection_bundle.manifest
    ).digest.sha256
    if artifact_name == "freeze.json":
        config["intervention"]["expected_freeze_artifact_sha256"] = (
            artifact_digest.sha256
        )
    fixture.config.write_text(
        json.dumps(config, sort_keys=True), encoding="utf-8"
    )


def test_model_causal_consumes_exact_select_features_freeze_for_heldout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    selection_bundle = _enable_heldout_select_features(fixture, tmp_path)
    _install_workflow_fakes(fixture, monkeypatch)

    assert official_causal.run(
        SimpleNamespace(config=fixture.config, output_dir=fixture.output)
    ) == 0
    manifest = verify_run_directory(fixture.output)
    roles = {item.role for item in manifest.inputs}
    assert {
        "selection.parent_manifest",
        "selection.selection",
        "selection.summary",
        "intervention.freeze",
    } <= roles
    summary = json.loads(
        (fixture.output / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["roster_split"] == "held_out"
    assert summary["evidence_scope"] == "confirmatory-held-out"
    assert summary["input_bindings"]["selection_parent_manifest_sha256"] == (
        verify_file(selection_bundle.manifest).digest.sha256
    )


def test_model_causal_rejects_tampered_freeze_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    selection_bundle = _enable_heldout_select_features(fixture, tmp_path)
    _install_workflow_fakes(fixture, monkeypatch)
    selection_bundle.freeze.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash or size mismatch"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_model_causal_rejects_semantically_resealed_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    selection_bundle = _enable_heldout_select_features(fixture, tmp_path)
    freeze = json.loads(selection_bundle.freeze.read_text(encoding="utf-8"))
    freeze["selected_interventions"][0]["latent_baseline"] = 1.0
    selection_bundle.freeze.write_text(
        json.dumps(freeze, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    _reseal_selection_artifact(fixture, selection_bundle, "freeze.json")
    _install_workflow_fakes(fixture, monkeypatch)

    with pytest.raises(ValueError, match="frozen interventions differ"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_model_causal_rejects_tampered_validation_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    selection_bundle = _enable_heldout_select_features(fixture, tmp_path)
    _install_workflow_fakes(fixture, monkeypatch)
    selection_bundle.selection.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash or size mismatch"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_model_causal_rejects_semantically_resealed_validation_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    selection_bundle = _enable_heldout_select_features(fixture, tmp_path)
    summary = json.loads(selection_bundle.summary.read_text(encoding="utf-8"))
    summary["evidence_scope"] = "confirmatory-held-out"
    selection_bundle.summary.write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _reseal_selection_artifact(fixture, selection_bundle, "summary.json")
    _install_workflow_fakes(fixture, monkeypatch)

    with pytest.raises(ValueError, match="freeze/summary schema or scope"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda freeze: freeze.__setitem__(
            "paired_source_direction",
            {"source_condition": "temporary", "recipient_condition": "rope"},
        ),
        lambda freeze: freeze["paired_source_binding"].__setitem__(
            "checkpoint_sha256", freeze["checkpoint_sha256"]
        ),
    ],
)
def test_heldout_freeze_rejects_donor_direction_or_checkpoint_swapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation
) -> None:
    fixture = _workflow_fixture(tmp_path)
    selection_bundle = _enable_heldout_select_features(fixture, tmp_path)
    freeze = json.loads(selection_bundle.freeze.read_text(encoding="utf-8"))
    mutation(freeze)
    selection_bundle.freeze.write_text(
        json.dumps(freeze, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    _reseal_selection_artifact(fixture, selection_bundle, "freeze.json")
    _install_workflow_fakes(fixture, monkeypatch)

    with pytest.raises(ValueError, match="paired-source direction or binding"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )


def test_heldout_run_rejects_roster_not_frozen_by_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    _enable_heldout_select_features(fixture, tmp_path)
    replacement = tmp_path / "replacement-heldout-roster.json"
    replacement.write_text(
        json.dumps(
            {
                "dataset_id": "toy-official",
                "split": "test",
                "row_indices": [0],
                "sample_ids": ["different-test-0"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config["dataset"]["sample_roster_path"] = str(replacement)
    config["dataset"]["expected_sample_roster_sha256"] = verify_file(
        replacement
    ).digest.sha256
    fixture.config.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    _install_workflow_fakes(fixture, monkeypatch)

    with pytest.raises(ValueError, match="not exactly frozen"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )


def test_heldout_rejects_old_model_causal_selection_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    selection_bundle = _enable_heldout_select_features(fixture, tmp_path)
    manifest = json.loads(selection_bundle.manifest.read_text(encoding="utf-8"))
    manifest["command"] = "model-causal"
    selection_bundle.manifest.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config["intervention"]["expected_selection_manifest_sha256"] = verify_file(
        selection_bundle.manifest
    ).digest.sha256
    fixture.config.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    _install_workflow_fakes(fixture, monkeypatch)

    with pytest.raises(ValueError, match="strict select-features"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )


def test_model_causal_rejects_inconsistent_worst_condition_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    _install_workflow_fakes(fixture, monkeypatch)
    original_loader = official_causal.load_verified_representation_checkpoint

    def inconsistent_loader(*args, **kwargs):
        model, normalizer, metadata = original_loader(*args, **kwargs)
        metadata["qualification"]["worst_condition_explained_variance"] = 0.98
        return model, normalizer, metadata

    monkeypatch.setattr(
        official_causal,
        "load_verified_representation_checkpoint",
        inconsistent_loader,
    )

    with pytest.raises(ValueError, match="worst-condition"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_model_causal_rejects_unrelated_parent_analysis_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    manifest_path = fixture.parent_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["analysis_code_sha"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _install_workflow_fakes(fixture, monkeypatch)

    with pytest.raises(ValueError, match="analysis ancestry"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


def test_model_causal_requires_roster_split_and_evaluation_split_alignment(
    tmp_path: Path
) -> None:
    fixture = _workflow_fixture(tmp_path)
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config["dataset"]["roster_split"] = "held_out"
    fixture.config.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="requires evaluation_split='test'"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )
    assert not fixture.output.exists()


@pytest.mark.parametrize(
    ("roster_split", "scope"),
    [
        ("discovery", "exploratory-discovery"),
        ("validation", "exploratory-feature-selection"),
        ("held_out", "confirmatory-held-out"),
    ],
)
def test_model_causal_evidence_scope_is_explicit_and_validation_is_exploratory(
    roster_split: str, scope: str
) -> None:
    assert official_causal._evidence_scope_for_roster_split(roster_split) == scope
    if roster_split == "validation":
        assert "formal" not in scope and "confirmatory" not in scope


def test_exploratory_pilot_is_discovery_only_and_publishes_explicit_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    roster = tmp_path / "pilot-roster.json"
    roster.write_text(
        json.dumps(
            {
                "dataset_id": "pilot-dataset",
                "split": "val",
                "row_indices": [0],
                "sample_ids": ["pilot-0"],
            }
        ),
        encoding="utf-8",
    )
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config["checkpoint_study"] = {
        "scope": "exploratory_pilot",
        "recipient_chain": None,
        "source_chain": None,
    }
    config["dataset"].update(
        {
            "dataset_id": "pilot-dataset",
            "roster_split": "discovery",
            "evaluation_split": "val",
            "sample_roster_path": str(roster),
        }
    )
    fixture.config.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    _install_workflow_fakes(fixture, monkeypatch)

    assert official_causal.run(
        SimpleNamespace(config=fixture.config, output_dir=fixture.output)
    ) == 0
    summary = json.loads(
        (fixture.output / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["evidence_scope"] == "exploratory-pilot"
    assert summary["checkpoint_study"]["formal_trust_verified"] is False


def test_exploratory_pilot_cannot_enter_validation_or_heldout(tmp_path: Path) -> None:
    fixture = _workflow_fixture(tmp_path)
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    config["checkpoint_study"] = {
        "scope": "exploratory_pilot",
        "recipient_chain": None,
        "source_chain": None,
    }
    fixture.config.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="only use the discovery roster"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )


def test_formal_paired_config_requires_one_atomic_cohort_ledger(tmp_path: Path) -> None:
    fixture = _workflow_fixture(tmp_path)
    _enable_validation_paired_reverse_patch(fixture, tmp_path)
    config = json.loads(fixture.config.read_text(encoding="utf-8"))
    second_ledger = tmp_path / "copied-ledger.json"
    second_ledger.write_bytes(fixture.transaction_ledger.read_bytes())
    source = config["checkpoint_study"]["source_chain"][0]
    source["transaction_ledger_path"] = str(second_ledger)
    source["expected_transaction_ledger_file_sha256"] = verify_file(
        second_ledger
    ).digest.sha256

    with pytest.raises(ValueError, match="one atomic cohort ledger"):
        official_causal._checkpoint_study_configuration(
            config["checkpoint_study"], paired_source_requested=True
        )


def test_stage3_formal_study_revalidates_every_ancestor_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = "a" * 40
    ledger = tmp_path / "transaction-ledger.json"
    ledger.write_text("{}\n", encoding="utf-8")
    stage_specs = (
        ("stage1", 500_000),
        ("stage2", 40_000),
        ("stage3", 10_000),
    )
    raw_chain = []
    checkpoint_paths: dict[str, Path] = {}
    finalized_hashes: dict[str, str] = {}
    for index, (stage, _terminal_step) in enumerate(stage_specs, start=1):
        checkpoint = tmp_path / f"{stage}.ckpt"
        checkpoint.write_bytes(f"verified-{stage}".encode("utf-8"))
        finalized = tmp_path / f"{stage}-finalized.json"
        finalized.write_text("{}\n", encoding="utf-8")
        checkpoint_paths[stage] = checkpoint
        finalized_hashes[stage] = f"{index}" * 64
        raw_chain.append(
            {
                "checkpoint_path": str(checkpoint),
                "expected_checkpoint_sha256": verify_file(
                    checkpoint
                ).digest.sha256,
                "finalized_manifest_path": str(finalized),
                "expected_finalized_manifest_file_sha256": verify_file(
                    finalized
                ).digest.sha256,
                "expected_finalized_manifest_sha256": finalized_hashes[stage],
                "transaction_ledger_path": str(ledger),
                "expected_transaction_ledger_file_sha256": verify_file(
                    ledger
                ).digest.sha256,
                "transaction_ledger_sha256": "9" * 64,
                "artifact_root": str(tmp_path),
                "study_id": "formal-study",
                "arm": "temporary",
                "stage": stage,
                "upstream_identity": f"formal-study:temporary:{stage}",
                "artifact_identity": (
                    f"formal-study.temporary.{stage}.final"
                ),
            }
        )
    study = official_causal._checkpoint_study_configuration(
        {
            "scope": "formal",
            "recipient_chain": raw_chain,
            "source_chain": None,
        },
        paired_source_requested=False,
    )
    files = {
        "checkpoint_study.transaction_ledger": verify_file(ledger),
    }
    for index, configured in enumerate(study["recipient_chain"]):
        files[f"checkpoint_study.recipient.{index}.finalized_manifest"] = (
            configured["finalized_manifest_file"]
        )
        if index < len(stage_specs) - 1:
            files[f"checkpoint_study.recipient.{index}.checkpoint"] = (
                configured["checkpoint_file"]
            )
    context = SimpleNamespace(
        condition="temporary",
        inputs=SimpleNamespace(
            checkpoint=study["recipient_chain"][-1]["checkpoint_file"],
            model_code=SimpleNamespace(head_sha=head),
            training_code=SimpleNamespace(head_sha=head),
        ),
        additional_file=lambda role: files[role],
    )
    initial_digests = {
        path: verify_file(path).digest.sha256 for path in checkpoint_paths.values()
    }
    validated_manifests: dict[str, dict[str, object]] = {}
    reported_source_commit = head

    def fake_validate_parent_trust(trust):
        actual = verify_file(trust.checkpoint_path)
        if actual.digest.sha256 != initial_digests[trust.checkpoint_path]:
            raise ValueError("ancestor checkpoint bytes changed")
        stage = trust.parent_stage
        terminal_step = dict(stage_specs)[stage]
        parent_record = {
            "stage": stage,
            "terminal_step": terminal_step,
            "max_checkpoint_bytes": 1 << 20,
            "upstream_identity": f"formal-study:temporary:{stage}",
            "artifact_identity": f"formal-study.temporary.{stage}.final",
            "finalized_manifest_sha256": finalized_hashes[stage],
            "transaction_ledger_sha256": "9" * 64,
        }
        manifest = {"payload": {"parent": parent_record}}
        previous = {"stage2": "stage1", "stage3": "stage2"}.get(stage)
        source_manifest = {
            "sha256": "b" * 64,
            "payload": {"commit_sha": reported_source_commit},
        }
        checkpoint = {
            "provenance": {
                "bundle_sha256": "c" * 64,
                "manifests": {
                    "source": source_manifest,
                    "environment": {"sha256": "d" * 64},
                    "prior": {"sha256": "e" * 64},
                    "architecture": {"sha256": "f" * 64},
                    "optimizer": {"sha256": "1" * 64},
                    "seed": {"sha256": "2" * 64},
                    "treatment": {
                        "sha256": "3" * 64,
                        "payload": {
                            "schema_version": 1,
                            "row_identity_mode": "temporary",
                            "identity_rng_seed": 42,
                            "seed_policy": "shared",
                            "sampler_version": "temporary-v1",
                            "world_size": 1,
                            "manifest_sha256": "4" * 64,
                        },
                    },
                    "scientific_config": {"sha256": "5" * 64},
                    "cohort_protocol": {"sha256": "6" * 64},
                    "arm_protocol": {"sha256": "7" * 64},
                    "operational_config": {
                        "payload": {
                            "context": {
                                "study_id": "formal-study",
                                "arm": "temporary",
                                "output_id": f"formal-study-temporary-{stage}",
                            }
                        }
                    },
                    "parent": (
                        {"payload": {"parent": None}}
                        if previous is None
                        else validated_manifests[previous]
                    ),
                },
            }
        }
        validated_manifests[stage] = manifest
        return SimpleNamespace(
            manifest=manifest,
            checkpoint=checkpoint,
            checkpoint_sha256=actual.digest.sha256,
            checkpoint_size=actual.digest.size_bytes,
        )

    monkeypatch.setattr(
        official_causal,
        "_trusted_training_provenance_api",
        lambda _context: (
            lambda **values: SimpleNamespace(**values),
            fake_validate_parent_trust,
        ),
    )
    binding = official_causal._validated_checkpoint_study(
        study, context=context, paired_source_binding=None
    )
    assert binding["formal_trust_verified"] is True
    assert binding["model_evidence_scope"] == "final-stage3"

    reported_source_commit = "0" * 40
    with pytest.raises(ValueError, match="source commit differs"):
        official_causal._validated_checkpoint_study(
            study, context=context, paired_source_binding=None
        )

    reported_source_commit = head
    checkpoint_paths["stage1"].write_bytes(b"tampered-stage1")
    with pytest.raises(ValueError, match="ancestor checkpoint bytes changed"):
        official_causal._validated_checkpoint_study(
            study, context=context, paired_source_binding=None
        )


def test_formal_paired_chain_accepts_temporary_sampler_only_treatment_difference(
) -> None:
    shared = {
        "study_id": "formal-study",
        "stage": "stage1",
        "terminal_step": 500_000,
        "source_sha256": "1" * 64,
        "environment_sha256": "2" * 64,
        "prior_sha256": "3" * 64,
        "architecture_sha256": "4" * 64,
        "optimizer_sha256": "5" * 64,
        "seed_sha256": "6" * 64,
        "scientific_sha256": "7" * 64,
        "cohort_protocol_sha256": "8" * 64,
        "transaction_ledger_sha256": "9" * 64,
        "transaction_ledger_file_sha256": "a" * 64,
        "max_checkpoint_bytes": 1 << 20,
    }
    recipient = {
        **shared,
        "mode": "temporary",
        "checkpoint_sha256": "b" * 64,
        "treatment_sha256": "c" * 64,
        "arm_protocol_sha256": "d" * 64,
        "output_id": "temporary-stage1",
    }
    source = {
        **shared,
        "mode": "rope",
        "checkpoint_sha256": "e" * 64,
        "treatment_sha256": "f" * 64,
        "arm_protocol_sha256": "0" * 64,
        "output_id": "rope-stage1",
    }

    def checkpoint(mode: str, sampler_version: str) -> dict[str, object]:
        return {
            "provenance": {
                "manifests": {
                    "treatment": {
                        "payload": {
                            "schema_version": 1,
                            "row_identity_mode": mode,
                            "identity_rng_seed": 42,
                            "seed_policy": "shared",
                            "sampler_version": sampler_version,
                            "world_size": 1,
                        }
                    }
                }
            }
        }

    official_causal._validate_formal_paired_chains(
        [recipient],
        [source],
        recipient_checkpoints=[checkpoint("temporary", "temporary-v1")],
        source_checkpoints=[checkpoint("rope", "not-applicable")],
        recipient_condition="temporary",
        source_condition="rope",
    )

    source_with_other_ledger = {**source, "transaction_ledger_sha256": "f" * 64}
    with pytest.raises(ValueError, match="cohort invariant mismatch"):
        official_causal._validate_formal_paired_chains(
            [recipient],
            [source_with_other_ledger],
            recipient_checkpoints=[checkpoint("temporary", "temporary-v1")],
            source_checkpoints=[checkpoint("rope", "not-applicable")],
            recipient_condition="temporary",
            source_condition="rope",
        )


@pytest.mark.parametrize(
    ("field", "relaxed_value"),
    [
        ("max_no_op_reconstruction_mse", 0.010001),
        ("max_no_op_probability_deviation", 0.020001),
        ("max_no_op_accuracy_difference", 0.005001),
    ],
)
def test_model_causal_workflow_rejects_relaxed_protocol_thresholds(
    tmp_path: Path,
    field: str,
    relaxed_value: float,
) -> None:
    fixture = _workflow_fixture(tmp_path)
    payload = json.loads(fixture.config.read_text(encoding="utf-8"))
    payload["intervention"][field] = relaxed_value
    fixture.config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exceed protocol maxima"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )

    assert not fixture.output.exists()


@pytest.mark.parametrize("random_state", [None, True, 41])
def test_model_causal_workflow_requires_aligned_explicit_random_state(
    tmp_path: Path,
    random_state: object,
) -> None:
    fixture = _workflow_fixture(tmp_path)
    payload = json.loads(fixture.config.read_text(encoding="utf-8"))
    payload["official_classifier"]["estimator_options"][
        "random_state"
    ] = random_state
    fixture.config.write_text(json.dumps(payload), encoding="utf-8")

    expected = "explicit non-negative integer" if random_state in {None, True} else "must equal"
    with pytest.raises(ValueError, match=expected):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )

    assert not fixture.output.exists()


def test_model_causal_workflow_requires_strict_train_repr_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    _install_workflow_fakes(fixture, monkeypatch)
    manifest_path = fixture.parent_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["command"] = "collect"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="command must be train-repr"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )

    assert not fixture.output.exists()


def test_model_causal_workflow_rejects_train_evaluation_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    payload = json.loads(fixture.config.read_text(encoding="utf-8"))
    payload["dataset"]["evaluation_split"] = "train"
    fixture.config.write_text(json.dumps(payload), encoding="utf-8")
    _install_workflow_fakes(fixture, monkeypatch)

    with pytest.raises(ValueError, match="never train"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )

    assert not fixture.output.exists()


def test_model_causal_publication_failure_leaves_no_partial_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _workflow_fixture(tmp_path)
    _install_workflow_fakes(fixture, monkeypatch)
    original_write = official_causal._write_json_artifact
    calls = 0

    def fail_second_write(path, payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected publication failure")
        return original_write(path, payload)

    monkeypatch.setattr(
        official_causal, "_write_json_artifact", fail_second_write
    )
    with pytest.raises(RuntimeError, match="injected publication failure"):
        official_causal.run(
            SimpleNamespace(config=fixture.config, output_dir=fixture.output)
        )

    assert not fixture.output.exists()
    assert not list(tmp_path.glob(".published-model-causal.*.staging"))
