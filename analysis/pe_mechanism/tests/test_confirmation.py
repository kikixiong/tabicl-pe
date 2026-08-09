from __future__ import annotations

import copy
from pathlib import Path

import pytest

import pe_mechanism.confirmation as confirmation
import pe_mechanism.feature_selection as selection
from pe_mechanism.confirmation import (
    ConfirmationProtocol,
    FrozenCandidate,
    confirm_frozen_candidates,
)
from pe_mechanism.feature_selection import ValidationObservation


def protocol(
    *,
    dataset_count: int = 8,
    frozen_candidate_count: int = 1,
    maximum_confirmation_candidates: int = 2,
) -> ConfirmationProtocol:
    alpha = 0.05
    resamples = 999
    mode = (
        "exact-enumeration"
        if dataset_count <= 20
        else "fixed-seed-monte-carlo"
    )
    minimum_p = (
        2.0 ** (-dataset_count)
        if mode == "exact-enumeration"
        else 1.0 / (resamples + 1)
    )
    return ConfirmationProtocol(
        alpha=alpha,
        confidence_level=0.95,
        sign_flip_resamples=resamples,
        bootstrap_resamples=500,
        minimum_positive_fraction=0.75,
        minimum_heldout_datasets=8,
        required_heldout_datasets=8,
        maximum_confirmation_candidates=maximum_confirmation_candidates,
        frozen_candidate_count=frozen_candidate_count,
        preregistered_holm_rank_one_threshold=(
            alpha / maximum_confirmation_candidates
        ),
        actual_holm_rank_one_threshold=(
            None
            if frozen_candidate_count == 0
            else alpha / frozen_candidate_count
        ),
        sign_flip_minimum_p_value=minimum_p,
        sign_flip_mode=mode,
        random_seed=42,
    )


def candidate(name: str, feature: int) -> FrozenCandidate:
    return FrozenCandidate(
        candidate_id=name,
        target_features=(feature,),
        control_features=(feature + 10,),
        latent_baseline=0.0,
    )


def observations(
    candidate_id: str,
    *,
    count: int = 8,
    target: float = 0.5,
    control: float = 0.1,
    donor_rescue: float = 0.3,
    donor_specificity: float = 0.2,
    source_native: float = 0.4,
    source_no_op: float = 0.4,
) -> list[ValidationObservation]:
    return [
        ValidationObservation(
            candidate_id=candidate_id,
            dataset_id=f"heldout-{index:02d}",
            target_effect=target,
            control_effect=control,
            donor_rescue_effect=donor_rescue,
            donor_specificity_effect=donor_specificity,
            source_native_advantage_effect=source_native,
            source_no_op_advantage_effect=source_no_op,
            donor_shift_balance_ratio=1.1,
        )
        for index in range(count)
    ]


def test_confirmatory_iut_passes_all_six_components() -> None:
    frozen = candidate("candidate-a", 1)
    result = confirm_frozen_candidates(
        [frozen],
        observations(frozen.candidate_id),
        protocol=protocol(maximum_confirmation_candidates=1),
    )

    candidate_result = result["candidate_results"][0]
    assert candidate_result["confirmed"] is True
    assert candidate_result["intersection_union_composite_p_value"] == 1 / 256
    assert candidate_result["holm_adjusted_composite_p_value"] == 1 / 256
    assert result["any_candidate_confirmed"] is True
    assert result["all_frozen_candidates_confirmed"] is True
    source = result["source_advantage_prerequisite"]
    assert source["direction_ci_replication_gates_passed"] is True
    assert source["p_value_gate_deferred_to_candidate_composite"] is True
    assert "passed" not in source


def test_negative_target_fails_even_when_specificity_is_positive() -> None:
    frozen = candidate("candidate-a", 1)
    result = confirm_frozen_candidates(
        [frozen],
        observations(frozen.candidate_id, target=-0.1, control=-1.0),
        protocol=protocol(maximum_confirmation_candidates=1),
    )

    endpoints = {
        endpoint["name"]: endpoint
        for endpoint in result["candidate_results"][0]["endpoints"]
    }
    assert endpoints["ablation_specificity"]["mean_effect"] > 0.0
    assert endpoints["target_damage"]["passes_direction_ci_replication"] is False
    assert result["candidate_results"][0]["confirmed"] is False


def test_source_component_is_part_of_each_candidate_iut() -> None:
    frozen = candidate("candidate-a", 1)
    result = confirm_frozen_candidates(
        [frozen],
        observations(frozen.candidate_id, source_no_op=-0.2),
        protocol=protocol(maximum_confirmation_candidates=1),
    )

    assert result["source_advantage_prerequisite"][
        "direction_ci_replication_gates_passed"
    ] is False
    assert result["candidate_results"][0]["confirmed"] is False


def test_holm_tests_every_frozen_candidate_without_top_k() -> None:
    first = candidate("candidate-a", 1)
    second = candidate("candidate-b", 2)
    result = confirm_frozen_candidates(
        [first, second],
        [*observations(first.candidate_id), *observations(second.candidate_id)],
        protocol=protocol(frozen_candidate_count=2),
    )

    assert [item["candidate_id"] for item in result["candidate_results"]] == [
        "candidate-a",
        "candidate-b",
    ]
    assert all(item["confirmed"] for item in result["candidate_results"])
    assert result["family_candidate_count"] == 2
    assert result["top_k_after_freeze"] is False


def test_incomplete_duplicate_or_extra_heldout_runs_fail_closed() -> None:
    frozen = candidate("candidate-a", 1)
    complete = observations(frozen.candidate_id)
    with pytest.raises(ValueError, match="incomplete, duplicated, or misaligned"):
        confirm_frozen_candidates(
            [frozen], complete[:-1], protocol=protocol(maximum_confirmation_candidates=1)
        )
    with pytest.raises(ValueError, match="incomplete, duplicated, or misaligned"):
        confirm_frozen_candidates(
            [frozen],
            [*complete[:-1], complete[0]],
            protocol=protocol(maximum_confirmation_candidates=1),
        )
    extra = ValidationObservation(
        **{
            **complete[0].__dict__,
            "candidate_id": "not-frozen",
        }
    )
    with pytest.raises(ValueError, match="unfrozen candidate"):
        confirm_frozen_candidates(
            [frozen],
            [*complete, extra],
            protocol=protocol(maximum_confirmation_candidates=1),
        )


def test_full_24_dataset_roster_uses_frozen_monte_carlo() -> None:
    frozen = candidate("candidate-a", 1)
    first = confirm_frozen_candidates(
        [frozen],
        observations(frozen.candidate_id, count=24),
        protocol=protocol(
            dataset_count=24,
            maximum_confirmation_candidates=1,
        ),
    )
    second = confirm_frozen_candidates(
        [frozen],
        observations(frozen.candidate_id, count=24),
        protocol=protocol(
            dataset_count=24,
            maximum_confirmation_candidates=1,
        ),
    )
    assert first == second
    assert first["candidate_results"][0]["confirmed"] is True


def test_single_dataset_cannot_be_used_as_confirmation() -> None:
    frozen = candidate("candidate-a", 1)
    with pytest.raises(ValueError, match="incomplete, duplicated, or misaligned"):
        confirm_frozen_candidates(
            [frozen],
            observations(frozen.candidate_id, count=1),
            protocol=protocol(maximum_confirmation_candidates=1),
        )


def test_zero_frozen_candidates_terminate_without_heldout_evidence() -> None:
    frozen_protocol = protocol(frozen_candidate_count=0)
    result = confirm_frozen_candidates([], [], protocol=frozen_protocol)

    assert result == {
        "status": "no_frozen_candidates",
        "heldout_evidence_consumed": False,
        "source_advantage_prerequisite": None,
        "candidate_results": [],
        "family_candidate_count": 0,
        "multiplicity_method": None,
        "top_k_after_freeze": False,
        "any_candidate_confirmed": False,
        "all_frozen_candidates_confirmed": None,
    }
    assert confirmation._heldout_run_specifications([]) == ()

    with pytest.raises(ValueError, match="zero-candidate termination"):
        confirm_frozen_candidates(
            [],
            observations("not-frozen"),
            protocol=frozen_protocol,
        )


def test_confirmation_protocol_rejects_boolean_random_seed() -> None:
    payload = confirmation._protocol_payload(
        protocol(maximum_confirmation_candidates=1)
    )
    payload["random_seed"] = True

    with pytest.raises(ValueError, match="non-negative integer"):
        confirmation._confirmation_protocol(payload, random_seed=1)


def test_validation_and_heldout_dataset_aliases_are_rejected() -> None:
    validation = {
        "validation-a": {
            "raw_dataset_content_sha256": "a" * 64,
            "invariant_prediction_content_sha256": "b" * 64,
            "combined_dataset_fingerprint_sha256": "c" * 64,
        }
    }
    heldout = {
        "heldout-a": {
            "raw_dataset_content_sha256": "a" * 64,
            "invariant_prediction_content_sha256": "d" * 64,
            "combined_dataset_fingerprint_sha256": "e" * 64,
        }
    }
    with pytest.raises(ValueError, match="validation and heldout"):
        confirmation._assert_disjoint_dataset_fingerprints(validation, heldout)


def test_selection_pass_flags_and_ranking_are_semantically_recomputed() -> None:
    dataset_ids = tuple(f"validation-{index:02d}" for index in range(8))
    candidate_specification = selection.CandidateSpecification(
        candidate_id="candidate-a",
        target_features=(1,),
        control_features=(11,),
        latent_baseline=0.0,
        validation_runs=tuple(
            selection.ValidationRunSpecification(
                dataset_id=dataset_id,
                run_dir=Path("/frozen-validation"),
                expected_manifest_sha256="0" * 64,
            )
            for dataset_id in dataset_ids
        ),
    )
    validation_observations = observations("candidate-a")
    validation_observations = [
        ValidationObservation(
            **{
                **item.__dict__,
                "dataset_id": dataset_id,
            }
        )
        for item, dataset_id in zip(
            validation_observations, dataset_ids, strict=True
        )
    ]
    parameters = selection.SelectionParameters(
        fdr_alpha=0.05,
        confidence_level=0.95,
        bootstrap_resamples=500,
        sign_flip_resamples=999,
        random_seed=42,
        minimum_validation_datasets=8,
        minimum_positive_fraction=0.75,
        maximum_selections=1,
        source_condition="rope",
        recipient_condition="none",
        evidence_family=selection._EVIDENCE_FAMILY,
        maximum_symmetric_donor_shift_rms_ratio=1.25,
    )
    candidate_results = selection.aggregate_validation_candidates(
        [candidate_specification],
        validation_observations,
        parameters=parameters,
    )
    source = candidate_results[0].pop("source_advantage_prerequisite")
    artifact = {
        "candidate_results": candidate_results,
        "source_advantage_prerequisite": source,
        "selected_candidates": ["candidate-a"],
        "statistics": {
            "multiplicity_method": "benjamini-yekutieli",
            "fdr_control_unit": "candidate-intersection-union-hypothesis",
            "candidate_composite_method": "maximum-of-six-component-p-values",
            "component_count_per_candidate": 6,
            "fdr_alpha": 0.05,
            "confidence_level": 0.95,
            "bootstrap_method": "paired-dataset-bootstrap",
            "bootstrap_resamples": 500,
            "p_value_method": "one-sided-paired-sign-flip",
            "sign_flip_resamples": 999,
            "minimum_validation_datasets": 8,
            "minimum_positive_fraction": 0.75,
            "maximum_selections": 1,
            "ranking_rule": "maximum_minimum_mean_evidence",
            "family_size": 1,
            "harmonic_factor": 1.0,
            "rank_one_threshold": 0.05,
            "exact_sign_flip_minimum_p_value": 1 / 256,
            "required_validation_datasets": 8,
            "effective_validation_dataset_count": 8,
            "sign_flip_mode": "exact-enumeration",
            "monte_carlo_minimum_p_value": None,
        },
    }
    lineage = {
        "random_seed": 42,
        "paired_source_direction": {
            "source_condition": "rope",
            "recipient_condition": "none",
        },
    }
    recomputed = confirmation._recompute_validation_selection(
        artifact,
        validation_dataset_ids=dataset_ids,
        common_lineage=lineage,
        maximum_symmetric_donor_shift_rms_ratio=1.25,
    )
    assert recomputed["selected_candidate_ids"] == ["candidate-a"]

    tampered = copy.deepcopy(artifact)
    tampered["candidate_results"][0]["selected"] = False
    with pytest.raises(ValueError, match="do not recompute exactly"):
        confirmation._recompute_validation_selection(
            tampered,
            validation_dataset_ids=dataset_ids,
            common_lineage=lineage,
            maximum_symmetric_donor_shift_rms_ratio=1.25,
        )
