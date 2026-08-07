from __future__ import annotations

from pathlib import Path
import copy
import math

import pytest

import pe_mechanism.feature_selection as selection
from pe_mechanism.feature_selection import (
    CandidateSpecification,
    SelectionParameters,
    ValidationObservation,
    ValidationRunSpecification,
    aggregate_validation_candidates,
)


def parameters(**overrides: object) -> SelectionParameters:
    values: dict[str, object] = {
        "fdr_alpha": 0.05,
        "confidence_level": 0.95,
        "bootstrap_resamples": 1_000,
        "sign_flip_resamples": 1,
        "random_seed": 42,
        "minimum_validation_datasets": 8,
        "minimum_positive_fraction": 0.75,
        "maximum_selections": 1,
        "source_condition": "rope",
        "recipient_condition": "none",
        "evidence_family": selection._EVIDENCE_FAMILY,
        "maximum_symmetric_donor_shift_rms_ratio": 1.25,
    }
    values.update(overrides)
    return SelectionParameters(**values)  # type: ignore[arg-type]


def candidate(name: str = "feature-1", *, count: int = 8) -> CandidateSpecification:
    return CandidateSpecification(
        candidate_id=name,
        target_features=(1,),
        control_features=(7,),
        latent_baseline=0.0,
        validation_runs=tuple(
            ValidationRunSpecification(
                dataset_id=f"validation-{index}",
                run_dir=Path(f"/external/run-{name}-{index}"),
                expected_manifest_sha256=f"{index + 1:064x}",
            )
            for index in range(count)
        ),
    )


def observations(
    specification: CandidateSpecification,
    *,
    target: float = 0.5,
    control: float = 0.1,
    rescue: float = 0.3,
    donor_specificity: float = 0.2,
    source_advantage: float = 0.4,
) -> list[ValidationObservation]:
    return [
        ValidationObservation(
            candidate_id=specification.candidate_id,
            dataset_id=run.dataset_id,
            target_effect=target,
            control_effect=control,
            donor_rescue_effect=rescue,
            donor_specificity_effect=donor_specificity,
            source_native_advantage_effect=source_advantage,
            source_no_op_advantage_effect=source_advantage,
            donor_shift_balance_ratio=1.1,
        )
        for run in specification.validation_runs
    ]


def test_complete_four_part_family_passes_exact_fdr_and_freezes_top_candidate() -> None:
    specification = candidate()
    result = aggregate_validation_candidates(
        [specification], observations(specification), parameters=parameters()
    )[0]

    assert result["selected"] is True
    assert result["eligible"] is True
    assert result["effective_dataset_count"] == 8
    assert [item["name"] for item in result["hypotheses"]] == [
        "target_damage",
        "ablation_specificity",
        "donor_rescue",
        "donor_specificity",
    ]
    assert all(item["p_value"] == 1 / 256 for item in result["hypotheses"])
    assert result["intersection_union_composite_p_value"] == 1 / 256
    assert result["by_adjusted_composite_p_value"] == 1 / 256
    assert all(
        item["passes_direction_ci_replication"]
        for item in result["hypotheses"]
    )
    source = result["source_advantage_prerequisite"]
    assert source["composite_p_value"] == 1 / 256
    assert source["p_value_gate_deferred_to_candidate_composite"] is True
    assert "passed" not in source


def test_specificity_cannot_hide_a_target_edit_that_improves_loss() -> None:
    specification = candidate()
    result = aggregate_validation_candidates(
        [specification],
        observations(specification, target=-0.1, control=-1.0),
        parameters=parameters(),
    )[0]

    evidence = {item["name"]: item for item in result["hypotheses"]}
    assert evidence["ablation_specificity"]["mean_effect"] > 0.0
    assert evidence["target_damage"]["passes_direction_ci_replication"] is False
    assert result["eligible"] is False
    assert result["selected"] is False


def test_missing_donor_evidence_or_dataset_fails_closed() -> None:
    specification = candidate()
    weak_donor = aggregate_validation_candidates(
        [specification],
        observations(specification, rescue=-0.01),
        parameters=parameters(),
    )[0]
    assert weak_donor["eligible"] is False

    with pytest.raises(ValueError, match="dataset roster"):
        aggregate_validation_candidates(
            [specification],
            observations(specification)[:-1],
            parameters=parameters(),
        )


def test_by_family_covers_all_candidates_and_all_evidence() -> None:
    first = candidate("feature-1", count=10)
    second = candidate("feature-2", count=10)
    results = aggregate_validation_candidates(
        [first, second],
        [*observations(first), *observations(second, donor_specificity=0.0)],
        parameters=parameters(maximum_selections=2),
    )

    hypotheses = [item for result in results for item in result["hypotheses"]]
    assert len(hypotheses) == 8
    assert results[0]["selected"] is True
    assert results[1]["selected"] is False
    assert all("adjusted_p_value" not in item for item in hypotheses)
    assert all("by_adjusted_composite_p_value" in item for item in results)
    for result in results:
        source = result["source_advantage_prerequisite"]
        assert result["intersection_union_composite_p_value"] == max(
            [item["p_value"] for item in result["hypotheses"]]
            + [item["p_value"] for item in source["components"]]
        )


def test_selection_protocol_requires_eight_datasets_and_exact_family() -> None:
    config = {
        "metric": "mean_delta_log_loss_vs_no_op",
        "direction": "target_greater_than_matched_control",
        "fdr_alpha": 0.05,
        "confidence_level": 0.95,
        "bootstrap_resamples": 100,
        "sign_flip_resamples": 100,
        "random_seed": 42,
        "minimum_validation_datasets": 8,
        "minimum_positive_fraction": 0.75,
        "maximum_selections": 1,
        "ranking_rule": "maximum_minimum_mean_evidence",
        "evidence_family": list(selection._EVIDENCE_FAMILY),
        "paired_source_direction": {
            "source_condition": "rope",
            "recipient_condition": "none",
        },
        "maximum_symmetric_donor_shift_rms_ratio": 1.25,
        "multiplicity_method": "benjamini-yekutieli",
    }
    parsed = selection._selection_parameters(config)
    assert parsed.minimum_validation_datasets == 8
    assert len(parsed.evidence_family) == 5

    config["minimum_validation_datasets"] = 7
    with pytest.raises(ValueError, match="at least eight"):
        selection._selection_parameters(config)

    config["minimum_validation_datasets"] = 8
    config["evidence_family"] = list(selection._EVIDENCE_FAMILY[:-1])
    with pytest.raises(ValueError, match="complete canonical"):
        selection._selection_parameters(config)

    config["evidence_family"] = list(selection._EVIDENCE_FAMILY)
    config["maximum_selections"] = 3
    with pytest.raises(ValueError, match="at most two"):
        selection._selection_parameters(config)


def test_heldout_roster_parser_reads_only_ids_and_rejects_non_test_rows() -> None:
    payload = {
        "dataset_id": "heldout-1",
        "split": "test",
        "row_indices": [0, 4],
        "sample_ids": ["test-0", "test-4"],
    }
    assert selection._load_heldout_roster(selection._json_bytes(payload)) == {
        "dataset_id": "heldout-1"
    }
    payload["split"] = "val"
    with pytest.raises(ValueError, match="exactly 'test'"):
        selection._load_heldout_roster(selection._json_bytes(payload))


def test_heldout_confirmation_protocol_parses_bootstrap_count_and_strict_seed(
    tmp_path: Path,
) -> None:
    rosters = []
    for index in range(8):
        roster_path = tmp_path / f"heldout-{index}.json"
        roster_path.write_text("{}\n", encoding="utf-8")
        rosters.append(
            {
                "dataset_id": f"heldout-{index}",
                "sample_roster_path": str(roster_path),
                "expected_sample_roster_sha256": f"{index + 1:064x}",
            }
        )
    value = {
        "sample_rosters": rosters,
        "confirmation": {
            "alpha": 0.05,
            "confidence_level": 0.95,
            "sign_flip_resamples": 999,
            "bootstrap_resamples": 500,
            "minimum_positive_fraction": 0.75,
            "multiplicity_method": "holm",
            "minimum_heldout_datasets": 8,
            "random_seed": 42,
            "bootstrap_method": "paired-dataset-bootstrap",
        },
    }

    _, confirmation = selection._heldout_specifications(value, random_seed=42)
    assert confirmation.bootstrap_resamples == 500

    value["confirmation"]["random_seed"] = True
    with pytest.raises(ValueError, match="non-negative integer"):
        selection._heldout_specifications(value, random_seed=1)


def _prediction_values(probabilities: list[list[float]], labels: list[int]) -> dict:
    predicted = [max(range(2), key=lambda index: row[index]) for row in probabilities]
    return {
        "probabilities": probabilities,
        "predicted_class_indices": predicted,
        "predicted_labels": predicted,
        "true_class_log_loss": [
            -math.log(row[label])
            for row, label in zip(probabilities, labels, strict=True)
        ],
    }


def _prediction_payload() -> dict:
    labels = [0, 1]
    native = _prediction_values([[0.8, 0.2], [0.2, 0.8]], labels)
    probabilities = {
        "no_op_reconstruction": [[0.7, 0.3], [0.3, 0.7]],
        "target_baseline_edit": [[0.6, 0.4], [0.4, 0.6]],
        "matched_random_edit": [[0.65, 0.35], [0.35, 0.65]],
        "paired_reverse_patch": [[0.75, 0.25], [0.25, 0.75]],
        "paired_matched_random_patch": [[0.68, 0.32], [0.32, 0.68]],
        "roundtrip_restore_control": [[0.7, 0.3], [0.3, 0.7]],
    }
    conditions = {}
    no_op = _prediction_values(probabilities["no_op_reconstruction"], labels)
    for name, matrix in probabilities.items():
        record = _prediction_values(matrix, labels)
        controls = [7] if name in {
            "matched_random_edit",
            "paired_matched_random_patch",
        } else None
        conditions[name] = {
            **record,
            "delta_log_loss_vs_native": [
                value - baseline
                for value, baseline in zip(
                    record["true_class_log_loss"],
                    native["true_class_log_loss"],
                    strict=True,
                )
            ],
            "delta_log_loss_vs_no_op": [
                value - baseline
                for value, baseline in zip(
                    record["true_class_log_loss"],
                    no_op["true_class_log_loss"],
                    strict=True,
                )
            ],
            "probability_delta_vs_native": [
                [value - baseline for value, baseline in zip(row, native_row, strict=True)]
                for row, native_row in zip(matrix, native["probabilities"], strict=True)
            ],
            "probability_delta_vs_no_op": [
                [value - baseline for value, baseline in zip(row, no_op_row, strict=True)]
                for row, no_op_row in zip(matrix, no_op["probabilities"], strict=True)
            ],
            "target_features": [1],
            "control_features": controls,
        }
    donor = conditions["paired_reverse_patch"]["true_class_log_loss"]
    donor_control = conditions["paired_matched_random_patch"][
        "true_class_log_loss"
    ]
    target = conditions["target_baseline_edit"]["true_class_log_loss"]
    source_native = _prediction_values([[0.85, 0.15], [0.15, 0.85]], labels)
    source_no_op_probabilities = [[0.8, 0.2], [0.2, 0.8]]
    source_no_op_loss = [-math.log(0.8), -math.log(0.8)]
    return {
        "schema_version": 1,
        "dataset_id": "validation-0",
        "evaluation_split": "val",
        "roster_split": "validation",
        "evidence_scope": "exploratory-feature-selection",
        "sample_ids": ["sample-0", "sample-1"],
        "true_labels": labels,
        "classes": [0, 1],
        "native": native,
        "conditions": conditions,
        "paired_reverse_patch_measured_effects": {
            "log_loss_improvement_vs_recipient_no_op": [
                base - value
                for base, value in zip(
                    no_op["true_class_log_loss"], donor, strict=True
                )
            ],
            "log_loss_improvement_vs_paired_matched_random_patch": [
                base - value
                for base, value in zip(donor_control, donor, strict=True)
            ],
            "log_loss_improvement_vs_target_baseline_edit": [
                base - value for base, value in zip(target, donor, strict=True)
            ],
            "native_distance_reduction_vs_target_baseline_edit": [
                abs(base - native_loss) - abs(value - native_loss)
                for base, value, native_loss in zip(
                    target,
                    donor,
                    native["true_class_log_loss"],
                    strict=True,
                )
            ],
            "log_loss_improvement_of_source_native_vs_recipient_native": [
                recipient - source
                for recipient, source in zip(
                    native["true_class_log_loss"],
                    source_native["true_class_log_loss"],
                    strict=True,
                )
            ],
            "log_loss_improvement_of_source_no_op_vs_recipient_no_op": [
                recipient - source
                for recipient, source in zip(
                    no_op["true_class_log_loss"], source_no_op_loss, strict=True
                )
            ],
        },
        "paired_source_native": {
            **source_native,
            "no_op_probabilities": source_no_op_probabilities,
            "no_op_true_class_log_loss": source_no_op_loss,
        },
    }


def test_prediction_evidence_is_recomputed_from_probabilities_and_labels() -> None:
    payload = _prediction_payload()
    effects, alignment = selection._recomputed_prediction_effects(
        payload,
        dataset_id="validation-0",
        candidate=candidate(count=8),
    )
    assert effects["target_damage"] > 0.0
    assert len(alignment["invariant_prediction_content_sha256"]) == 64

    bad_probability = copy.deepcopy(payload)
    bad_probability["native"]["probabilities"][0][0] = 0.9
    with pytest.raises(ValueError, match="normalized probabilities"):
        selection._recomputed_prediction_effects(
            bad_probability,
            dataset_id="validation-0",
            candidate=candidate(count=8),
        )

    bad_loss = copy.deepcopy(payload)
    bad_loss["native"]["true_class_log_loss"][0] += 0.1
    with pytest.raises(ValueError, match="raw probabilities and true labels"):
        selection._recomputed_prediction_effects(
            bad_loss,
            dataset_id="validation-0",
            candidate=candidate(count=8),
        )

    bad_label = copy.deepcopy(payload)
    bad_label["native"]["predicted_labels"] = [1, 1]
    with pytest.raises(ValueError, match="predicted labels"):
        selection._recomputed_prediction_effects(
            bad_label,
            dataset_id="validation-0",
            candidate=candidate(count=8),
        )

    bad_delta = copy.deepcopy(payload)
    bad_delta["conditions"]["target_baseline_edit"][
        "delta_log_loss_vs_native"
    ][0] += 0.1
    with pytest.raises(ValueError, match="differs from raw probabilities"):
        selection._recomputed_prediction_effects(
            bad_delta,
            dataset_id="validation-0",
            candidate=candidate(count=8),
        )


def test_dataset_fingerprint_aliases_are_rejected() -> None:
    digest_a = "a" * 64
    digest_b = "b" * 64
    with pytest.raises(ValueError, match="fingerprint alias"):
        selection._dataset_fingerprint_mapping(
            {
                "dataset-a": {
                    "raw_dataset_content_sha256": digest_a,
                    "invariant_prediction_content_sha256": digest_b,
                },
                "dataset-b": {
                    "raw_dataset_content_sha256": digest_a,
                    "invariant_prediction_content_sha256": "c" * 64,
                },
            },
            label="validation",
        )


def _checkpoint_report(
    *, mode: str, checkpoint: str, treatment: str, stage: str = "stage1"
) -> dict:
    shared = "1" * 64
    terminal_step = {"stage1": 500_000, "stage2": 40_000, "stage3": 10_000}[
        stage
    ]
    return {
        "checkpoint_sha256": checkpoint,
        "checkpoint_size": 123,
        "provenance_sha256": "2" * 64,
        "source_sha256": shared,
        "environment_sha256": shared,
        "prior_sha256": shared,
        "architecture_sha256": shared,
        "optimizer_sha256": shared,
        "seed_sha256": shared,
        "treatment_sha256": treatment,
        "scientific_sha256": shared,
        "cohort_protocol_sha256": shared,
        "arm_protocol_sha256": treatment,
        "study_id": "formal-study",
        "output_id": f"formal-{mode}-{stage}",
        "mode": mode,
        "stage": stage,
        "terminal_step": terminal_step,
        "max_checkpoint_bytes": 1_000,
        "upstream_identity": f"formal-study:{mode}:{stage}",
        "artifact_identity": f"formal-study.{mode}.{stage}.final",
        "finalized_manifest_sha256": "3" * 64,
        "transaction_ledger_sha256": "4" * 64,
        "checkpoint_file_sha256": checkpoint,
        "finalized_manifest_file_sha256": "5" * 64,
        "transaction_ledger_file_sha256": "6" * 64,
    }


def test_checkpoint_study_must_be_formal_self_bound_and_directional() -> None:
    recipient_checkpoint = "a" * 64
    source_checkpoint = "b" * 64
    study = {
        "schema_version": 1,
        "scope": "formal",
        "formal_trust_verified": True,
        "model_evidence_scope": "intermediate-stage-specific",
        "direction": {
            "source_condition": "rope",
            "recipient_condition": "none",
        },
        "recipient_chain": [
            _checkpoint_report(
                mode="none",
                checkpoint=recipient_checkpoint,
                treatment="c" * 64,
            )
        ],
        "source_chain": [
            _checkpoint_report(
                mode="rope",
                checkpoint=source_checkpoint,
                treatment="d" * 64,
            )
        ],
    }
    binding = selection._canonical_sha256(study)
    study["binding_sha256"] = binding
    registered_inputs = {
        "checkpoint_study.transaction_ledger": "6" * 64,
        "checkpoint_study.recipient.0.finalized_manifest": "5" * 64,
        "checkpoint_study.source.0.finalized_manifest": "5" * 64,
        "paired_source.checkpoint": source_checkpoint,
    }
    assert selection._validated_checkpoint_study(
        study,
        input_bindings={"checkpoint_study_sha256": binding},
        registered_inputs=registered_inputs,
        source_condition="rope",
        recipient_condition="none",
        source_checkpoint_sha256=source_checkpoint,
        recipient_checkpoint_sha256=recipient_checkpoint,
    ) == binding

    tampered = copy.deepcopy(study)
    tampered["formal_trust_verified"] = False
    with pytest.raises(ValueError, match="formal paired checkpoint study"):
        selection._validated_checkpoint_study(
            tampered,
            input_bindings={"checkpoint_study_sha256": binding},
            registered_inputs=registered_inputs,
            source_condition="rope",
            recipient_condition="none",
            source_checkpoint_sha256=source_checkpoint,
            recipient_checkpoint_sha256=recipient_checkpoint,
        )


def test_checkpoint_study_binds_ledger_finalized_and_ancestor_inputs() -> None:
    recipient = [
        _checkpoint_report(
            mode="none", checkpoint="a" * 64, treatment="1" * 64
        ),
        _checkpoint_report(
            mode="none",
            checkpoint="b" * 64,
            treatment="2" * 64,
            stage="stage2",
        ),
    ]
    source = [
        _checkpoint_report(
            mode="rope", checkpoint="c" * 64, treatment="3" * 64
        ),
        _checkpoint_report(
            mode="rope",
            checkpoint="d" * 64,
            treatment="4" * 64,
            stage="stage2",
        ),
    ]
    # Cross-arm invariants include each stage's raw finalized/ledger bytes.
    source[0]["finalized_manifest_file_sha256"] = recipient[0][
        "finalized_manifest_file_sha256"
    ]
    source[1]["finalized_manifest_file_sha256"] = recipient[1][
        "finalized_manifest_file_sha256"
    ]
    study_payload = {
        "schema_version": 1,
        "scope": "formal",
        "formal_trust_verified": True,
        "model_evidence_scope": "intermediate-stage-specific",
        "direction": {
            "source_condition": "rope",
            "recipient_condition": "none",
        },
        "recipient_chain": recipient,
        "source_chain": source,
    }
    binding = selection._canonical_sha256(study_payload)
    study = {**study_payload, "binding_sha256": binding}
    registered = {
        "checkpoint_study.transaction_ledger": "6" * 64,
        "checkpoint_study.recipient.0.finalized_manifest": "5" * 64,
        "checkpoint_study.recipient.0.checkpoint": "a" * 64,
        "checkpoint_study.recipient.1.finalized_manifest": "5" * 64,
        "checkpoint_study.source.0.finalized_manifest": "5" * 64,
        "checkpoint_study.source.0.checkpoint": "c" * 64,
        "checkpoint_study.source.1.finalized_manifest": "5" * 64,
        "paired_source.checkpoint": "d" * 64,
    }
    arguments = {
        "input_bindings": {"checkpoint_study_sha256": binding},
        "registered_inputs": registered,
        "source_condition": "rope",
        "recipient_condition": "none",
        "source_checkpoint_sha256": "d" * 64,
        "recipient_checkpoint_sha256": "b" * 64,
    }
    assert selection._validated_checkpoint_study(study, **arguments) == binding

    for role, message in (
        ("checkpoint_study.transaction_ledger", "ledger bytes"),
        (
            "checkpoint_study.recipient.0.finalized_manifest",
            "finalized manifest",
        ),
        ("checkpoint_study.source.0.checkpoint", "ancestor checkpoint"),
        ("paired_source.checkpoint", "terminal source"),
    ):
        tampered_inputs = {**registered, role: "f" * 64}
        with pytest.raises(ValueError, match=message):
            selection._validated_checkpoint_study(
                study, **{**arguments, "registered_inputs": tampered_inputs}
            )
