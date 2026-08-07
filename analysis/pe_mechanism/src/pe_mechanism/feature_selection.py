"""Strict multi-dataset validation selection and held-out freezing.

This workflow is intentionally separated from model execution.  It consumes
only completed validation ``model-causal`` runs, aggregates one paired
target-versus-matched-control contrast per dataset, controls the false
discovery rate across every pre-registered candidate, and freezes the choices
before any held-out activation or prediction is read.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .identifiers import require_portable_identifier, require_public_label
from .manifest import RunManifest
from .provenance import (
    RunTransaction,
    assert_dataset_roster,
    assert_git_commit_is_ancestor,
    load_verified_json_config,
    load_verified_run_manifest,
    manifest_from_verified_inputs,
    verify_configured_run_inputs,
    verify_file,
    verify_run_directory,
)
from .statistics import (
    adjust_fdr_arbitrary_dependence,
    paired_bootstrap_ci,
    paired_sign_flip_p_value,
)


_METRIC = "mean_delta_log_loss_vs_no_op"
_DIRECTION = "target_greater_than_matched_control"
_RESCUE_METRIC = "mean_log_loss_improvement_vs_recipient_no_op"
_DONOR_SPECIFICITY_METRIC = (
    "mean_log_loss_improvement_vs_paired_matched_random_patch"
)
_EVIDENCE_FAMILY = (
    {
        "name": "target_damage",
        "metric": _METRIC,
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
        "metric": _RESCUE_METRIC,
        "direction": "positive",
        "scope": "candidate",
    },
    {
        "name": "donor_specificity",
        "metric": _DONOR_SPECIFICITY_METRIC,
        "direction": "positive",
        "scope": "candidate",
    },
    {
        "name": "source_advantage",
        "metric": "intersection_source_native_and_no_op_advantage",
        "direction": "both_positive",
        "scope": "global",
    },
)
_TARGET_CONDITION = "target_baseline_edit"
_CONTROL_CONDITION = "matched_random_edit"
_CANONICAL_CONDITIONS = {"rope", "temporary", "none"}
_TOP_LEVEL_FIELDS = {"provenance", "selection", "candidates", "heldout"}
_SELECTION_FIELDS = {
    "metric",
    "direction",
    "fdr_alpha",
    "confidence_level",
    "bootstrap_resamples",
    "sign_flip_resamples",
    "random_seed",
    "minimum_validation_datasets",
    "minimum_positive_fraction",
    "maximum_selections",
    "ranking_rule",
    "evidence_family",
    "paired_source_direction",
    "maximum_symmetric_donor_shift_rms_ratio",
    "multiplicity_method",
}
_CANDIDATE_FIELDS = {
    "candidate_id",
    "target_features",
    "control_features",
    "latent_baseline",
    "validation_runs",
}
_RUN_FIELDS = {"dataset_id", "run_dir", "expected_manifest_sha256"}
_HELDOUT_FIELDS = {"sample_rosters", "confirmation"}
_HELDOUT_ROSTER_FIELDS = {
    "dataset_id",
    "sample_roster_path",
    "expected_sample_roster_sha256",
}
_CONFIRMATION_FIELDS = {
    "alpha",
    "confidence_level",
    "sign_flip_resamples",
    "bootstrap_resamples",
    "minimum_positive_fraction",
    "multiplicity_method",
    "minimum_heldout_datasets",
    "random_seed",
    "bootstrap_method",
}
_CHECKPOINT_STAGE_REPORT_FIELDS = {
    "checkpoint_sha256",
    "checkpoint_size",
    "provenance_sha256",
    "source_sha256",
    "environment_sha256",
    "prior_sha256",
    "architecture_sha256",
    "optimizer_sha256",
    "seed_sha256",
    "treatment_sha256",
    "scientific_sha256",
    "cohort_protocol_sha256",
    "arm_protocol_sha256",
    "study_id",
    "output_id",
    "mode",
    "stage",
    "terminal_step",
    "max_checkpoint_bytes",
    "upstream_identity",
    "artifact_identity",
    "finalized_manifest_sha256",
    "transaction_ledger_sha256",
    "checkpoint_file_sha256",
    "finalized_manifest_file_sha256",
    "transaction_ledger_file_sha256",
}


@dataclass(frozen=True)
class CandidateSpecification:
    candidate_id: str
    target_features: tuple[int, ...]
    control_features: tuple[int, ...]
    latent_baseline: float | tuple[float, ...]
    validation_runs: tuple["ValidationRunSpecification", ...]


@dataclass(frozen=True)
class ValidationRunSpecification:
    dataset_id: str
    run_dir: Path
    expected_manifest_sha256: str


@dataclass(frozen=True)
class HeldoutRosterSpecification:
    dataset_id: str
    path: Path
    expected_sha256: str


@dataclass(frozen=True)
class PreparedParent:
    candidate: CandidateSpecification
    specification: ValidationRunSpecification
    manifest: RunManifest
    manifest_role: str
    summary_role: str
    predictions_role: str


@dataclass(frozen=True)
class ValidationObservation:
    candidate_id: str
    dataset_id: str
    target_effect: float
    control_effect: float
    donor_rescue_effect: float
    donor_specificity_effect: float
    source_native_advantage_effect: float
    source_no_op_advantage_effect: float
    donor_shift_balance_ratio: float

    @property
    def paired_effect(self) -> float:
        return self.target_effect - self.control_effect


@dataclass(frozen=True)
class SelectionParameters:
    fdr_alpha: float
    confidence_level: float
    bootstrap_resamples: int
    sign_flip_resamples: int
    random_seed: int
    minimum_validation_datasets: int
    minimum_positive_fraction: float
    maximum_selections: int
    source_condition: str
    recipient_condition: str
    evidence_family: tuple[Mapping[str, str], ...]
    maximum_symmetric_donor_shift_rms_ratio: float


@dataclass(frozen=True)
class ConfirmationParameters:
    alpha: float
    confidence_level: float
    sign_flip_resamples: int
    bootstrap_resamples: int
    minimum_positive_fraction: float
    minimum_heldout_datasets: int
    random_seed: int


def run(args: Any) -> int:
    """Aggregate strict validation runs and atomically freeze held-out choices."""

    configuration = load_verified_json_config(Path(args.config))
    config = _exact_object(
        configuration.data,
        label="select-features config",
        fields=_TOP_LEVEL_FIELDS,
    )
    parameters = _selection_parameters(config["selection"])
    candidates = _candidate_specifications(
        config["candidates"],
        minimum_validation_datasets=parameters.minimum_validation_datasets,
    )
    heldout, confirmation = _heldout_specifications(
        config["heldout"], random_seed=parameters.random_seed
    )
    if len(heldout) < confirmation.minimum_heldout_datasets:
        raise ValueError(
            "held-out roster is smaller than its frozen confirmation minimum"
        )

    additional_paths: dict[str, Path] = {}
    expected_additional: dict[str, str] = {}
    heldout_roles: dict[str, str] = {}
    for item in heldout:
        heldout_file = verify_file(
            item.path, expected_sha256=item.expected_sha256
        )
        heldout_roster = _load_heldout_roster(heldout_file.read_bytes())
        if heldout_roster["dataset_id"] != item.dataset_id:
            raise ValueError(
                "heldout roster dataset_id differs from its pre-registration"
            )
        role = f"heldout.sample_roster.{heldout_file.digest.sha256}"
        _register_input(
            role,
            heldout_file.path,
            heldout_file.digest.sha256,
            additional_paths,
            expected_additional,
        )
        heldout_roles[item.dataset_id] = role
    heldout_dataset_ids = tuple(sorted(heldout_roles))
    prepared: list[PreparedParent] = []
    seen_parent_files: set[tuple[int, int]] = set()
    validation_roster: tuple[str, ...] | None = None
    for candidate in candidates:
        candidate_roster = tuple(
            sorted(item.dataset_id for item in candidate.validation_runs)
        )
        if set(heldout_dataset_ids) & set(candidate_roster):
            raise ValueError(
                "validation and held-out dataset identities must be disjoint"
            )
        if validation_roster is None:
            validation_roster = candidate_roster
        elif candidate_roster != validation_roster:
            raise ValueError(
                "all candidates must use the same validation dataset roster"
            )
        for specification in candidate.validation_runs:
            parent, roles = _prepare_parent(
                candidate,
                specification,
                additional_paths=additional_paths,
                expected_additional=expected_additional,
            )
            identity = (
                verify_file(specification.run_dir / "manifest.json").device,
                verify_file(specification.run_dir / "manifest.json").inode,
            )
            if identity in seen_parent_files:
                raise ValueError("a validation run directory cannot be reused")
            seen_parent_files.add(identity)
            prepared.append(
                PreparedParent(
                    candidate=candidate,
                    specification=specification,
                    manifest=parent,
                    manifest_role=roles["manifest"],
                    summary_role=roles["summary"],
                    predictions_role=roles["predictions"],
                )
            )
    assert validation_roster is not None
    family_size = len(candidates)
    harmonic_factor = _harmonic_number(family_size)
    required_validation_datasets = max(
        8,
        math.ceil(
            math.log2(
                family_size * harmonic_factor / parameters.fdr_alpha
            )
        ),
    )
    if parameters.minimum_validation_datasets < required_validation_datasets:
        raise ValueError(
            "minimum_validation_datasets cannot attain the candidate-level "
            f"BY rank-one threshold for family_size={family_size}; require at least "
            f"{required_validation_datasets}"
        )
    if len(validation_roster) < required_validation_datasets:
        raise ValueError(
            "validation roster cannot attain the BY rank-one threshold"
        )
    if (
        len(validation_roster) > 20
        and 1.0 / (parameters.sign_flip_resamples + 1)
        > parameters.fdr_alpha / (family_size * harmonic_factor)
    ):
        raise ValueError(
            "sign_flip_resamples cannot resolve the BY rank-one threshold"
        )
    maximum_confirmation_candidates = parameters.maximum_selections
    required_heldout_datasets = max(
        8,
        math.ceil(
            math.log2(
                maximum_confirmation_candidates / confirmation.alpha
            )
        ),
    )
    if len(heldout) < max(
        confirmation.minimum_heldout_datasets, required_heldout_datasets
    ):
        raise ValueError(
            "held-out roster cannot attain the pre-registered Holm rank-one "
            "threshold"
        )
    confirmation_sign_flip_mode = (
        "exact-enumeration"
        if len(heldout) <= 20
        else "fixed-seed-monte-carlo"
    )
    confirmation_minimum_p_value = (
        2.0 ** (-len(heldout))
        if confirmation_sign_flip_mode == "exact-enumeration"
        else 1.0 / (confirmation.sign_flip_resamples + 1)
    )
    preregistered_holm_rank_one_threshold = (
        confirmation.alpha / maximum_confirmation_candidates
    )
    if confirmation_minimum_p_value > preregistered_holm_rank_one_threshold:
        raise ValueError(
            "held-out sign-flip test cannot resolve the pre-registered Holm "
            "rank-one threshold"
        )

    context = verify_configured_run_inputs(
        configuration,
        command="select-features",
        seed=parameters.random_seed,
        additional_input_paths=additional_paths,
        expected_additional_sha256=expected_additional,
    )
    if context.inputs.evidence_level != "strict":
        raise RuntimeError("select-features requires strict Git evidence")
    if context.condition not in _CANONICAL_CONDITIONS:
        raise ValueError("selection condition must be rope, temporary, or none")
    if context.condition != parameters.recipient_condition:
        raise ValueError(
            "selection condition differs from paired recipient pre-registration"
        )
    if context.sites != ("row_interactor",):
        raise ValueError(
            "formal paired selection requires exactly the row_interactor site"
        )
    assert_dataset_roster(
        context.inputs.dataset_manifest,
        validation_roster,
        required_split="validation",
    )
    assert_dataset_roster(
        context.inputs.dataset_manifest,
        heldout_dataset_ids,
        required_split="held_out",
    )
    heldout_rosters_sha256 = {
        dataset_id: context.additional_file(role).digest.sha256
        for dataset_id, role in sorted(heldout_roles.items())
    }

    observations: list[ValidationObservation] = []
    common_lineage: dict[str, Any] | None = None
    alignment_by_dataset: dict[str, dict[str, Any]] = {}
    for parent in prepared:
        manifest_file = context.additional_file(parent.manifest_role)
        bound_manifest = load_verified_run_manifest(manifest_file)
        if not isinstance(bound_manifest, RunManifest) or bound_manifest != parent.manifest:
            raise RuntimeError(
                "validation parent manifest changed during input verification"
            )
        summary_file = context.additional_file(parent.summary_role)
        observation, lineage, alignment = _validate_parent_and_extract(
            parent,
            bound_manifest,
            _load_json_object(summary_file.read_bytes(), label="validation summary"),
            predictions=_load_json_object(
                context.additional_file(parent.predictions_role).read_bytes(),
                label="validation predictions",
            ),
            context=context,
            parameters=parameters,
        )
        if common_lineage is None:
            common_lineage = lineage
        elif lineage != common_lineage:
            raise ValueError(
                "validation runs do not share exact representation/source lineage"
            )
        previous_alignment = alignment_by_dataset.setdefault(
            observation.dataset_id, alignment
        )
        if previous_alignment != alignment:
            raise ValueError(
                "candidate runs for one dataset have different raw/sample alignment"
            )
        observations.append(observation)
    assert common_lineage is not None
    validation_dataset_fingerprints = _dataset_fingerprint_mapping(
        alignment_by_dataset,
        label="validation",
    )
    validation_dataset_fingerprint_manifest_sha256 = _canonical_sha256(
        validation_dataset_fingerprints
    )

    candidate_results = aggregate_validation_candidates(
        candidates,
        observations,
        parameters=parameters,
    )
    source_advantage = candidate_results[0]["source_advantage_prerequisite"]
    for result in candidate_results:
        if result.pop("source_advantage_prerequisite") != source_advantage:
            raise RuntimeError("source advantage aggregation was inconsistent")
    selected = tuple(
        result
        for result in candidate_results
        if result["selected"] is True
    )
    confirmation_protocol = _confirmation_protocol_payload(
        confirmation,
        required_heldout_datasets=required_heldout_datasets,
        sign_flip_mode=confirmation_sign_flip_mode,
        maximum_confirmation_candidates=maximum_confirmation_candidates,
        frozen_candidate_count=len(selected),
        sign_flip_minimum_p_value=confirmation_minimum_p_value,
    )
    selection_payload = _selection_payload(
        condition=context.condition,
        site=context.sites[0],
        parameters=parameters,
        common_lineage=common_lineage,
        validation_dataset_ids=validation_roster,
        heldout_rosters_sha256=heldout_rosters_sha256,
        family_size=family_size,
        harmonic_factor=harmonic_factor,
        required_validation_datasets=required_validation_datasets,
        validation_dataset_fingerprints=validation_dataset_fingerprints,
        validation_dataset_fingerprint_manifest_sha256=(
            validation_dataset_fingerprint_manifest_sha256
        ),
        source_advantage=source_advantage,
        confirmation_protocol=confirmation_protocol,
        candidate_results=candidate_results,
        selected=selected,
    )
    selection_bytes = _json_bytes(selection_payload)
    selection_sha256 = hashlib.sha256(selection_bytes).hexdigest()
    frozen_interventions = [
        {
            "candidate_id": result["candidate_id"],
            "target_features": result["target_features"],
            "control_features": result["control_features"],
            "latent_baseline": result["latent_baseline"],
        }
        for result in selected
    ]
    freeze_payload = {
        "schema_version": 2,
        "evidence_scope": "validation-frozen",
        "condition": context.condition,
        "site": context.sites[0],
        "random_seed": parameters.random_seed,
        "evaluation_sample_rosters_sha256": heldout_rosters_sha256,
        "validation_dataset_fingerprints_sha256": (
            validation_dataset_fingerprints
        ),
        "validation_dataset_fingerprint_manifest_sha256": (
            validation_dataset_fingerprint_manifest_sha256
        ),
        "selected_interventions": frozen_interventions,
        "representation_model_sha256": common_lineage[
            "representation_model_sha256"
        ],
        "representation_parent_manifest_sha256": common_lineage[
            "representation_parent_manifest_sha256"
        ],
        "model_sha": common_lineage["model_code_sha"],
        "checkpoint_sha256": common_lineage["checkpoint_sha256"],
        "inference_contract_sha256": common_lineage[
            "inference_contract_sha256"
        ],
        "paired_source_direction": common_lineage["paired_source_direction"],
        "paired_source_binding": common_lineage["paired_source_binding"],
        "maximum_symmetric_donor_shift_rms_ratio": (
            parameters.maximum_symmetric_donor_shift_rms_ratio
        ),
        "source_advantage_prerequisite": source_advantage,
        "confirmation_protocol": confirmation_protocol,
        "representation_source_lineage_sha256": _canonical_sha256(
            common_lineage["representation_source_lineage"]
        ),
        "checkpoint_study_sha256": common_lineage[
            "checkpoint_study_sha256"
        ],
        "validation_selection_sha256": selection_sha256,
    }
    summary_payload = {
        "schema_version": 1,
        "analysis": "validation-feature-selection",
        "evidence_scope": "validation-frozen",
        "condition": context.condition,
        "site": context.sites[0],
        "random_seed": parameters.random_seed,
        "evidence_family": [dict(item) for item in parameters.evidence_family],
        "source_advantage_prerequisite": source_advantage,
        "confirmation_protocol": freeze_payload["confirmation_protocol"],
        "selected_interventions": frozen_interventions,
        "selection_count": len(frozen_interventions),
        "validation_dataset_ids": list(validation_roster),
        "heldout_dataset_ids": list(heldout_dataset_ids),
        "evaluation_sample_rosters_sha256": heldout_rosters_sha256,
        "validation_dataset_fingerprints_sha256": (
            validation_dataset_fingerprints
        ),
        "validation_dataset_fingerprint_manifest_sha256": (
            validation_dataset_fingerprint_manifest_sha256
        ),
        "validation_selection_sha256": selection_sha256,
        "checkpoint_study_sha256": common_lineage[
            "checkpoint_study_sha256"
        ],
        "common_lineage": {
            name: common_lineage[name]
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
        },
    }

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
        _write_bytes(transaction.staging_dir / "selection.json", selection_bytes)
        _write_bytes(
            transaction.staging_dir / "freeze.json", _json_bytes(freeze_payload)
        )
        _write_bytes(
            transaction.staging_dir / "summary.json", _json_bytes(summary_payload)
        )
        artifacts = transaction.artifact_digests(
            ("freeze.json", "selection.json", "summary.json")
        )
        manifest = manifest_from_verified_inputs(
            context.inputs, artifacts=artifacts
        )
        transaction.commit(manifest, verified_inputs=context.inputs)
    return 0


def aggregate_validation_candidates(
    candidates: Sequence[CandidateSpecification],
    observations: Sequence[ValidationObservation],
    *,
    parameters: SelectionParameters,
) -> list[dict[str, Any]]:
    """Return deterministic paired statistics and FDR decisions."""

    if not candidates:
        raise ValueError("selection requires candidate specifications")
    if parameters.maximum_selections < 1 or parameters.maximum_selections > 2:
        raise ValueError("maximum_selections must lie in [1, 2]")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate specifications must have unique IDs")
    by_candidate: dict[str, list[ValidationObservation]] = {
        candidate.candidate_id: [] for candidate in candidates
    }
    for observation in observations:
        try:
            by_candidate[observation.candidate_id].append(observation)
        except KeyError as error:
            raise ValueError("observation references an unknown candidate") from error
    results: list[dict[str, Any]] = []
    candidate_family = tuple(
        item for item in parameters.evidence_family if item["scope"] == "candidate"
    )
    global_family = tuple(
        item for item in parameters.evidence_family if item["scope"] == "global"
    )
    if len(global_family) != 1 or global_family[0]["name"] != "source_advantage":
        raise ValueError("evidence family must contain one global source advantage")
    specifications = {candidate.candidate_id: candidate for candidate in candidates}
    for offset, candidate_id in enumerate(sorted(specifications)):
        specification = specifications[candidate_id]
        values = sorted(by_candidate[candidate_id], key=lambda item: item.dataset_id)
        expected_datasets = [
            item.dataset_id for item in specification.validation_runs
        ]
        if [item.dataset_id for item in values] != sorted(expected_datasets):
            raise ValueError(
                f"candidate {candidate_id!r} observations do not match its dataset roster"
            )
        evidence_vectors = {
            "target_damage": np.asarray(
                [item.target_effect for item in values], dtype=np.float64
            ),
            "ablation_specificity": np.asarray(
                [item.paired_effect for item in values], dtype=np.float64
            ),
            "donor_rescue": np.asarray(
                [item.donor_rescue_effect for item in values], dtype=np.float64
            ),
            "donor_specificity": np.asarray(
                [item.donor_specificity_effect for item in values],
                dtype=np.float64,
            ),
        }
        if len(values) < parameters.minimum_validation_datasets:
            raise ValueError(
                f"candidate {candidate_id!r} has too few validation datasets"
            )
        candidate_hypotheses: list[dict[str, Any]] = []
        for evidence_offset, evidence in enumerate(candidate_family):
            evidence_name = evidence["name"]
            observed = evidence_vectors[evidence_name]
            if not np.isfinite(observed).all():
                raise ValueError("validation evidence must be finite")
            hypothesis_offset = offset * len(candidate_family) + evidence_offset
            low, high = paired_bootstrap_ci(
                observed,
                confidence=parameters.confidence_level,
                n_resamples=parameters.bootstrap_resamples,
                seed=parameters.random_seed + hypothesis_offset,
            )
            hypothesis = {
                **dict(evidence),
                "effective_dataset_count": int(observed.size),
                "mean_effect": float(np.mean(observed)),
                "median_effect": float(np.median(observed)),
                "confidence_low": low,
                "confidence_high": high,
                "positive_fraction": float(np.mean(observed > 0.0)),
                "p_value": paired_sign_flip_p_value(
                    observed,
                    n_resamples=parameters.sign_flip_resamples,
                    seed=parameters.random_seed + 100_000 + hypothesis_offset,
                ),
            }
            hypothesis["passes_direction_ci_replication"] = bool(
                hypothesis["confidence_low"] > 0.0
                and hypothesis["positive_fraction"]
                >= parameters.minimum_positive_fraction
            )
            candidate_hypotheses.append(hypothesis)
        results.append(
            {
                "candidate_id": candidate_id,
                "target_features": list(specification.target_features),
                "control_features": list(specification.control_features),
                "latent_baseline": _json_safe_baseline(
                    specification.latent_baseline
                ),
                "effective_dataset_count": len(values),
                "dataset_effects": [
                    {
                        "dataset_id": item.dataset_id,
                        "target_damage": item.target_effect,
                        "matched_control_damage": item.control_effect,
                        "ablation_specificity": item.paired_effect,
                        "donor_rescue": item.donor_rescue_effect,
                        "donor_specificity": item.donor_specificity_effect,
                        "donor_shift_balance_ratio": (
                            item.donor_shift_balance_ratio
                        ),
                    }
                    for item in values
                ],
                "hypotheses": candidate_hypotheses,
            }
        )

    source_native_by_dataset: dict[str, set[float]] = {}
    source_no_op_by_dataset: dict[str, set[float]] = {}
    for observation in observations:
        source_native_by_dataset.setdefault(observation.dataset_id, set()).add(
            observation.source_native_advantage_effect
        )
        source_no_op_by_dataset.setdefault(observation.dataset_id, set()).add(
            observation.source_no_op_advantage_effect
        )
    if any(len(values) != 1 for values in source_native_by_dataset.values()) or any(
        len(values) != 1 for values in source_no_op_by_dataset.values()
    ):
        raise ValueError(
            "source advantage differs across candidates for one dataset"
        )
    expected_global_roster = {
        item.dataset_id for item in candidates[0].validation_runs
    }
    if (
        set(source_native_by_dataset) != expected_global_roster
        or set(source_no_op_by_dataset) != expected_global_roster
    ):
        raise ValueError("source advantage dataset roster is incomplete")
    global_offset = len(results) * len(candidate_family)
    components: list[dict[str, Any]] = []
    component_specs = (
        (
            "source_native_advantage",
            "mean_log_loss_improvement_of_source_native_vs_recipient_native",
            source_native_by_dataset,
        ),
        (
            "source_no_op_advantage",
            "mean_log_loss_improvement_of_source_no_op_vs_recipient_no_op",
            source_no_op_by_dataset,
        ),
    )
    for component_offset, (name, metric, by_dataset) in enumerate(component_specs):
        values = np.asarray(
            [next(iter(by_dataset[item])) for item in sorted(by_dataset)],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("source advantage evidence must be finite")
        low, high = paired_bootstrap_ci(
            values,
            confidence=parameters.confidence_level,
            n_resamples=parameters.bootstrap_resamples,
            seed=parameters.random_seed + global_offset + component_offset,
        )
        components.append(
            {
                "name": name,
                "metric": metric,
                "direction": "positive",
                "mean_effect": float(np.mean(values)),
                "median_effect": float(np.median(values)),
                "confidence_low": low,
                "confidence_high": high,
                "positive_fraction": float(np.mean(values > 0.0)),
                "p_value": paired_sign_flip_p_value(
                    values,
                    n_resamples=parameters.sign_flip_resamples,
                    seed=(
                        parameters.random_seed
                        + 100_000
                        + global_offset
                        + component_offset
                    ),
                ),
            }
        )
    source_advantage = {
        **dict(global_family[0]),
        "effective_dataset_count": len(source_native_by_dataset),
        "dataset_effects": [
            {
                "dataset_id": name,
                "source_native_advantage": next(
                    iter(source_native_by_dataset[name])
                ),
                "source_no_op_advantage": next(
                    iter(source_no_op_by_dataset[name])
                ),
            }
            for name in sorted(source_native_by_dataset)
        ],
        "components": components,
        "minimum_component_mean_effect": min(
            item["mean_effect"] for item in components
        ),
        "composite_p_value": max(item["p_value"] for item in components),
        "direction_ci_replication_gates_passed": all(
            item["confidence_low"] > 0.0
            and item["positive_fraction"]
            >= parameters.minimum_positive_fraction
            for item in components
        ),
        "p_value_gate_deferred_to_candidate_composite": True,
    }
    for component in components:
        component["passes_direction_ci_replication"] = bool(
            component["confidence_low"] > 0.0
            and component["positive_fraction"]
            >= parameters.minimum_positive_fraction
        )
    composite_p_values: list[float] = []
    for result in results:
        composite_p = max(
            [hypothesis["p_value"] for hypothesis in result["hypotheses"]]
            + [source_advantage["composite_p_value"]]
        )
        result["minimum_mean_evidence"] = min(
            hypothesis["mean_effect"] for hypothesis in result["hypotheses"]
        )
        result["intersection_union_composite_p_value"] = composite_p
        result["all_direction_ci_replication_gates_passed"] = bool(
            all(
                hypothesis["passes_direction_ci_replication"]
                for hypothesis in result["hypotheses"]
            )
            and source_advantage["direction_ci_replication_gates_passed"]
        )
        composite_p_values.append(composite_p)
    adjusted = adjust_fdr_arbitrary_dependence(
        composite_p_values
    )
    eligible: list[dict[str, Any]] = []
    for result, adjusted_p in zip(results, adjusted, strict=True):
        result["by_adjusted_composite_p_value"] = float(adjusted_p)
        result["eligible"] = bool(
            result["all_direction_ci_replication_gates_passed"]
            and adjusted_p <= parameters.fdr_alpha
        )
        result["selected"] = False
        result["source_advantage_prerequisite"] = source_advantage
        if result["eligible"]:
            eligible.append(result)
    eligible.sort(
        key=lambda item: (
            -float(item["minimum_mean_evidence"]),
            item["candidate_id"],
        )
    )
    for result in eligible[: parameters.maximum_selections]:
        result["selected"] = True
    return results


def _prepare_parent(
    candidate: CandidateSpecification,
    specification: ValidationRunSpecification,
    *,
    additional_paths: dict[str, Path],
    expected_additional: dict[str, str],
) -> tuple[RunManifest, dict[str, str]]:
    parent = verify_run_directory(specification.run_dir)
    manifest_file = verify_file(
        specification.run_dir / "manifest.json",
        expected_sha256=specification.expected_manifest_sha256,
    )
    parsed = load_verified_run_manifest(manifest_file)
    if not isinstance(parsed, RunManifest) or parsed != parent:
        raise RuntimeError("validation parent changed during directory verification")
    summary_artifact = next(
        (item for item in parent.artifacts if item.name == "summary.json"), None
    )
    if summary_artifact is None:
        raise ValueError("validation parent does not declare summary.json")
    if not any(item.name == "predictions.json" for item in parent.artifacts):
        raise ValueError("validation parent does not declare predictions.json")
    token = manifest_file.digest.sha256
    manifest_role = f"validation.manifest.{token}"
    summary_role = f"validation.summary.{token}"
    predictions_role = f"validation.predictions.{token}"
    _register_input(
        manifest_role,
        manifest_file.path,
        manifest_file.digest.sha256,
        additional_paths,
        expected_additional,
    )
    for index, artifact in enumerate(parent.artifacts):
        artifact_file = verify_file(
            specification.run_dir / artifact.name,
            expected_sha256=artifact.sha256,
        )
        role = (
            summary_role
            if artifact.name == "summary.json"
            else predictions_role
            if artifact.name == "predictions.json"
            else f"validation.artifact.{token}.{index}.{artifact.sha256}"
        )
        _register_input(
            role,
            artifact_file.path,
            artifact.sha256,
            additional_paths,
            expected_additional,
        )
    return parent, {
        "manifest": manifest_role,
        "summary": summary_role,
        "predictions": predictions_role,
    }


def _register_input(
    role: str,
    path: Path,
    digest: str,
    paths: dict[str, Path],
    expected: dict[str, str],
) -> None:
    if role in paths:
        raise ValueError("validation input role collision")
    paths[role] = path
    expected[role] = digest


def _validate_parent_and_extract(
    prepared: PreparedParent,
    manifest: RunManifest,
    summary: Mapping[str, Any],
    *,
    predictions: Mapping[str, Any],
    context: Any,
    parameters: SelectionParameters,
) -> tuple[ValidationObservation, dict[str, Any], dict[str, Any]]:
    candidate = prepared.candidate
    dataset_id = prepared.specification.dataset_id
    if manifest.command != "model-causal":
        raise ValueError("selection inputs must be model-causal runs")
    if manifest.evidence_level != "strict" or manifest.legacy_reasons:
        raise ValueError("selection inputs must have strict evidence")
    assert_git_commit_is_ancestor(
        context.inputs.analysis_code, manifest.analysis_code_sha
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
        "seed": parameters.random_seed,
    }
    mismatches = {
        name: (getattr(manifest, name), expected)
        for name, expected in expected_manifest.items()
        if getattr(manifest, name) != expected
    }
    if mismatches:
        raise ValueError(f"validation parent manifest lineage mismatch: {mismatches}")

    expected_summary = {
        "analysis": "official-model-causal",
        "dataset_id": dataset_id,
        "fit_split": "train",
        "evaluation_split": "val",
        "roster_split": "validation",
        "evidence_scope": "exploratory-feature-selection",
        "site": context.sites[0],
        "source_evidence_level": "strict",
    }
    summary_mismatches = {
        name: (summary.get(name), expected)
        for name, expected in expected_summary.items()
        if summary.get(name) != expected
    }
    if summary_mismatches:
        raise ValueError(
            f"validation summary scope mismatch: {summary_mismatches}"
        )
    no_op = summary.get("no_op_gates")
    if not isinstance(no_op, Mapping) or no_op.get("passed") is not True:
        raise ValueError("validation parent did not pass no-op gates")
    intervention = summary.get("intervention")
    expected_intervention = {
        "target_features": list(candidate.target_features),
        "control_features": list(candidate.control_features),
        "latent_baseline": _json_safe_baseline(candidate.latent_baseline),
        "random_seed": parameters.random_seed,
    }
    if intervention != expected_intervention:
        raise ValueError(
            "validation intervention differs from pre-registered candidate"
        )
    if summary.get("matched_control_features") != list(
        candidate.control_features
    ):
        raise ValueError(
            "validation matched controls differ from configured controls"
        )
    conditions = summary.get("conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("validation summary lacks condition metrics")
    target_summary = _condition_metric(conditions, _TARGET_CONDITION)
    control_summary = _condition_metric(conditions, _CONTROL_CONDITION)

    bindings = summary.get("input_bindings")
    source_lineage = summary.get("representation_source_lineage")
    if not isinstance(bindings, Mapping) or not isinstance(source_lineage, Mapping):
        raise ValueError("validation summary lacks representation lineage")
    registered = {item.role: item.sha256 for item in manifest.inputs}
    representation_model_sha256 = _required_sha256(
        bindings.get("representation_model_sha256"),
        name="representation_model_sha256",
    )
    representation_parent_sha256 = _required_sha256(
        bindings.get("parent_manifest_sha256"),
        name="parent_manifest_sha256",
    )
    if registered.get("representation.model") != representation_model_sha256:
        raise ValueError("representation model binding differs from parent manifest")
    if (
        registered.get("representation.parent_manifest")
        != representation_parent_sha256
    ):
        raise ValueError(
            "representation parent binding differs from parent manifest"
        )
    checkpoint_sha256 = _required_sha256(
        bindings.get("checkpoint_sha256"), name="checkpoint_sha256"
    )
    inference_contract_sha256 = _required_sha256(
        bindings.get("inference_contract_sha256"),
        name="inference_contract_sha256",
    )
    if bindings.get("model_sha") != context.inputs.model_code.head_sha:
        raise ValueError("validation model SHA differs from manifest lineage")
    if checkpoint_sha256 != context.inputs.checkpoint.digest.sha256:
        raise ValueError("validation checkpoint binding differs from manifest")
    recomputed, prediction_alignment = _recomputed_prediction_effects(
        predictions,
        dataset_id=dataset_id,
        candidate=candidate,
    )
    _require_close(
        target_summary, recomputed["target_damage"], name="target damage"
    )
    _require_close(
        control_summary,
        recomputed["matched_control_damage"],
        name="matched control damage",
    )
    (
        rescue_summary,
        donor_specificity_summary,
        source_native_advantage_summary,
        source_no_op_advantage_summary,
        donor_shift_ratio,
        paired_source_binding,
    ) = _paired_reverse_evidence(
        summary,
        parameters=parameters,
        registered_inputs=registered,
        recipient_model_sha=context.inputs.model_code.head_sha,
        recipient_inference_contract_sha256=inference_contract_sha256,
        representation_source_lineage=source_lineage,
    )
    checkpoint_study_sha256 = _validated_checkpoint_study(
        summary.get("checkpoint_study"),
        input_bindings=bindings,
        registered_inputs=registered,
        source_condition=parameters.source_condition,
        recipient_condition=parameters.recipient_condition,
        source_checkpoint_sha256=paired_source_binding["checkpoint_sha256"],
        recipient_checkpoint_sha256=checkpoint_sha256,
    )
    for name, summary_value in (
        ("donor_rescue", rescue_summary),
        ("donor_specificity", donor_specificity_summary),
        ("source_native_advantage", source_native_advantage_summary),
        ("source_no_op_advantage", source_no_op_advantage_summary),
    ):
        _require_close(summary_value, recomputed[name], name=name)
    lineage = {
        "model_family": manifest.model_family,
        "model_revision": manifest.model_revision,
        "training_code_sha": manifest.training_code_sha,
        "model_code_sha": manifest.model_code_sha,
        "parent_analysis_code_sha": manifest.analysis_code_sha,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_manifest_sha256": manifest.dataset_manifest.sha256,
        "condition": manifest.condition,
        "site": manifest.sites[0],
        "random_seed": manifest.seed,
        "representation_model_sha256": representation_model_sha256,
        "representation_parent_manifest_sha256": representation_parent_sha256,
        "inference_contract_sha256": inference_contract_sha256,
        "representation_source_lineage": _json_public_copy(source_lineage),
        "paired_source_direction": {
            "source_condition": parameters.source_condition,
            "recipient_condition": parameters.recipient_condition,
        },
        "paired_source_binding": paired_source_binding,
        "checkpoint_study_sha256": checkpoint_study_sha256,
    }
    raw_input_bindings = _validated_raw_input_bindings(
        bindings.get("raw_dataset_input_sha256"),
        registered_inputs=registered,
    )
    return (
        ValidationObservation(
            candidate_id=candidate.candidate_id,
            dataset_id=dataset_id,
            target_effect=recomputed["target_damage"],
            control_effect=recomputed["matched_control_damage"],
            donor_rescue_effect=recomputed["donor_rescue"],
            donor_specificity_effect=recomputed["donor_specificity"],
            source_native_advantage_effect=recomputed[
                "source_native_advantage"
            ],
            source_no_op_advantage_effect=recomputed[
                "source_no_op_advantage"
            ],
            donor_shift_balance_ratio=donor_shift_ratio,
        ),
        lineage,
        {
            "sample_roster_sha256": registered["samples.roster"],
            "preprocessing_roster_sha256": _required_sha256(
                bindings.get("preprocessing_roster_sha256"),
                name="preprocessing_roster_sha256",
            ),
            "raw_dataset_input_sha256": raw_input_bindings,
            "raw_dataset_content_sha256": _canonical_sha256(
                raw_input_bindings
            ),
            **prediction_alignment,
        },
    )


def _validated_checkpoint_study(
    value: Any,
    *,
    input_bindings: Mapping[str, Any],
    registered_inputs: Mapping[str, str],
    source_condition: str,
    recipient_condition: str,
    source_checkpoint_sha256: str,
    recipient_checkpoint_sha256: str,
) -> str:
    """Validate the published formal checkpoint attestation and return its hash."""

    study = _exact_object(
        value,
        label="checkpoint_study",
        fields={
            "schema_version",
            "scope",
            "formal_trust_verified",
            "model_evidence_scope",
            "direction",
            "recipient_chain",
            "source_chain",
            "binding_sha256",
        },
    )
    expected_direction = {
        "source_condition": source_condition,
        "recipient_condition": recipient_condition,
    }
    if (
        study["schema_version"] != 1
        or study["scope"] != "formal"
        or study["formal_trust_verified"] is not True
        or study["model_evidence_scope"]
        not in {"final-stage3", "intermediate-stage-specific"}
        or study["direction"] != expected_direction
    ):
        raise ValueError("selection requires a formal paired checkpoint study")
    binding = _required_sha256(
        study["binding_sha256"], name="checkpoint study binding_sha256"
    )
    unhashed = {name: child for name, child in study.items() if name != "binding_sha256"}
    if _canonical_sha256(unhashed) != binding:
        raise ValueError("checkpoint study binding does not match its payload")
    if input_bindings.get("checkpoint_study_sha256") != binding:
        raise ValueError("checkpoint study is not bound by input_bindings")

    recipient_chain = _validated_checkpoint_chain(
        study["recipient_chain"],
        label="recipient checkpoint chain",
        expected_condition=recipient_condition,
    )
    source_chain = _validated_checkpoint_chain(
        study["source_chain"],
        label="source checkpoint chain",
        expected_condition=source_condition,
    )
    if len(recipient_chain) != len(source_chain):
        raise ValueError("formal source and recipient checkpoint chains differ in length")
    if recipient_chain[-1]["checkpoint_sha256"] != recipient_checkpoint_sha256:
        raise ValueError("formal recipient chain does not end at the evaluated checkpoint")
    if source_chain[-1]["checkpoint_sha256"] != source_checkpoint_sha256:
        raise ValueError("formal source chain does not end at the donor checkpoint")

    ledger_roles = sorted(
        role for role in registered_inputs if "transaction_ledger" in role
    )
    if ledger_roles != ["checkpoint_study.transaction_ledger"]:
        raise ValueError(
            "formal checkpoint study requires exactly one canonical ledger input"
        )
    ledger_file_sha256 = registered_inputs[
        "checkpoint_study.transaction_ledger"
    ]
    expected_study_roles = {"checkpoint_study.transaction_ledger"}
    for chain_name, chain in (
        ("recipient", recipient_chain),
        ("source", source_chain),
    ):
        for stage_index, report in enumerate(chain):
            finalized_role = (
                f"checkpoint_study.{chain_name}.{stage_index}."
                "finalized_manifest"
            )
            expected_study_roles.add(finalized_role)
            if registered_inputs.get(finalized_role) != report[
                "finalized_manifest_file_sha256"
            ]:
                raise ValueError(
                    "formal checkpoint finalized manifest is not manifest-bound"
                )
            if report["transaction_ledger_file_sha256"] != ledger_file_sha256:
                raise ValueError(
                    "formal checkpoint ledger bytes are not manifest-bound"
                )
            if stage_index < len(chain) - 1:
                checkpoint_role = (
                    f"checkpoint_study.{chain_name}.{stage_index}.checkpoint"
                )
                expected_study_roles.add(checkpoint_role)
                if registered_inputs.get(checkpoint_role) != report[
                    "checkpoint_file_sha256"
                ]:
                    raise ValueError(
                        "formal ancestor checkpoint is not manifest-bound"
                    )
    actual_study_roles = {
        role
        for role in registered_inputs
        if role.startswith("checkpoint_study.")
    }
    if actual_study_roles != expected_study_roles:
        raise ValueError("formal checkpoint study input role set is not canonical")
    if recipient_chain[-1]["checkpoint_file_sha256"] != (
        recipient_checkpoint_sha256
    ):
        raise ValueError("formal terminal recipient is not the manifest checkpoint")
    if registered_inputs.get("paired_source.checkpoint") != source_chain[-1][
        "checkpoint_file_sha256"
    ]:
        raise ValueError("formal terminal source checkpoint is not manifest-bound")
    shared_fields = {
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
    }
    for recipient, source in zip(recipient_chain, source_chain, strict=True):
        if any(recipient[name] != source[name] for name in shared_fields):
            raise ValueError("formal paired checkpoint cohort invariants differ")
        if (
            recipient["checkpoint_sha256"] == source["checkpoint_sha256"]
            or recipient["treatment_sha256"] == source["treatment_sha256"]
            or recipient["arm_protocol_sha256"] == source["arm_protocol_sha256"]
            or recipient["output_id"] == source["output_id"]
        ):
            raise ValueError("formal paired checkpoint arms are not independent")
    return binding


def _validated_checkpoint_chain(
    value: Any, *, label: str, expected_condition: str
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value or len(value) > 3:
        raise ValueError(f"{label} must contain one to three stage reports")
    result: list[dict[str, Any]] = []
    expected_stages = ("stage1", "stage2", "stage3")[: len(value)]
    terminal_steps = {"stage1": 500_000, "stage2": 40_000, "stage3": 10_000}
    digest_fields = {
        name for name in _CHECKPOINT_STAGE_REPORT_FIELDS if name.endswith("sha256")
    }
    for index, raw in enumerate(value):
        report = _exact_object(
            raw, label=f"{label} entry", fields=_CHECKPOINT_STAGE_REPORT_FIELDS
        )
        for name in digest_fields:
            _required_sha256(report[name], name=f"{label}.{name}")
        for name in ("checkpoint_size", "max_checkpoint_bytes"):
            _positive_int(report[name], name=f"{label}.{name}")
        if (
            report["stage"] != expected_stages[index]
            or report["terminal_step"] != terminal_steps[report["stage"]]
            or report["mode"] != expected_condition
        ):
            raise ValueError(f"{label} stage, step, or arm is not canonical")
        for name in (
            "study_id",
            "output_id",
            "upstream_identity",
            "artifact_identity",
        ):
            require_portable_identifier(report[name], name=f"{label}.{name}")
        if report["checkpoint_file_sha256"] != report["checkpoint_sha256"]:
            raise ValueError(f"{label} checkpoint file digest is inconsistent")
        result.append(report)
    for name in ("study_id", "transaction_ledger_sha256"):
        if len({report[name] for report in result}) != 1:
            raise ValueError(f"{label} {name} changes between stages")
    return result


def _condition_metric(conditions: Mapping[str, Any], name: str) -> float:
    values = conditions.get(name)
    if not isinstance(values, Mapping) or set(values) != {
        "accuracy",
        "log_loss",
        "mean_delta_log_loss_vs_native",
        "mean_delta_log_loss_vs_no_op",
    }:
        raise ValueError(f"validation condition {name!r} metric fields mismatch")
    value = values[_METRIC]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"validation metric {_METRIC!r} must be numerical")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"validation metric {_METRIC!r} must be finite")
    return result


def _recomputed_prediction_effects(
    value: Mapping[str, Any],
    *,
    dataset_id: str,
    candidate: CandidateSpecification,
    evaluation_split: str = "val",
    roster_split: str = "validation",
    evidence_scope: str = "exploratory-feature-selection",
) -> tuple[dict[str, float], dict[str, Any]]:
    predictions = _exact_object(
        value,
        label="validation predictions",
        fields={
            "schema_version",
            "dataset_id",
            "evaluation_split",
            "roster_split",
            "evidence_scope",
            "sample_ids",
            "true_labels",
            "classes",
            "native",
            "conditions",
            "paired_reverse_patch_measured_effects",
            "paired_source_native",
        },
    )
    expected_scope = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "evaluation_split": evaluation_split,
        "roster_split": roster_split,
        "evidence_scope": evidence_scope,
    }
    if any(predictions[name] != expected for name, expected in expected_scope.items()):
        raise ValueError("validation predictions scope differs from its run")
    sample_ids = predictions["sample_ids"]
    true_labels = predictions["true_labels"]
    if (
        not isinstance(sample_ids, list)
        or not sample_ids
        or len({_canonical_json(item) for item in sample_ids}) != len(sample_ids)
        or not isinstance(true_labels, list)
        or len(true_labels) != len(sample_ids)
    ):
        raise ValueError("validation prediction sample roster is invalid")
    classes, true_class_indices = _validated_classes_and_labels(
        predictions["classes"], true_labels
    )
    native = _prediction_record(
        predictions["native"],
        label="native prediction",
        expected=len(sample_ids),
        classes=classes,
        true_class_indices=true_class_indices,
    )
    conditions = predictions["conditions"]
    if not isinstance(conditions, Mapping):
        raise ValueError("validation predictions conditions must be an object")
    required_conditions = {
        "no_op_reconstruction",
        "target_baseline_edit",
        "matched_random_edit",
        "paired_reverse_patch",
        "paired_matched_random_patch",
        "roundtrip_restore_control",
    }
    if set(conditions) != required_conditions:
        raise ValueError("validation predictions conditions are not canonical")
    parsed_conditions = {
        name: _condition_prediction_record(
            conditions[name],
            label=name,
            expected=len(sample_ids),
            classes=classes,
            true_class_indices=true_class_indices,
        )
        for name in required_conditions
    }
    no_op = parsed_conditions["no_op_reconstruction"]
    target = parsed_conditions["target_baseline_edit"]
    control = parsed_conditions["matched_random_edit"]
    donor = parsed_conditions["paired_reverse_patch"]
    donor_control = parsed_conditions["paired_matched_random_patch"]
    for name, condition in parsed_conditions.items():
        _validate_condition_prediction_deltas(
            condition,
            native=native,
            no_op=no_op,
            label=name,
        )
    expected_feature_metadata = {
        "no_op_reconstruction": (list(candidate.target_features), None),
        "target_baseline_edit": (list(candidate.target_features), None),
        "matched_random_edit": (
            list(candidate.target_features),
            list(candidate.control_features),
        ),
        "paired_reverse_patch": (list(candidate.target_features), None),
        "paired_matched_random_patch": (
            list(candidate.target_features),
            list(candidate.control_features),
        ),
        "roundtrip_restore_control": (list(candidate.target_features), None),
    }
    for name, (expected_targets, expected_controls) in (
        expected_feature_metadata.items()
    ):
        condition = parsed_conditions[name]
        if (
            condition["target_features"] != expected_targets
            or condition["control_features"] != expected_controls
        ):
            raise ValueError(
                f"prediction feature metadata for {name!r} differs from candidate"
            )

    source = _exact_object(
        predictions["paired_source_native"],
        label="paired_source_native predictions",
        fields={
            "probabilities",
            "predicted_class_indices",
            "predicted_labels",
            "true_class_log_loss",
            "no_op_probabilities",
            "no_op_true_class_log_loss",
        },
    )
    source_native = _prediction_record(
        {
            "probabilities": source["probabilities"],
            "predicted_class_indices": source["predicted_class_indices"],
            "predicted_labels": source["predicted_labels"],
            "true_class_log_loss": source["true_class_log_loss"],
        },
        label="source native prediction",
        expected=len(sample_ids),
        classes=classes,
        true_class_indices=true_class_indices,
    )
    source_no_op_probabilities = _probability_matrix(
        source["no_op_probabilities"],
        name="source no-op probabilities",
        expected_rows=len(sample_ids),
        expected_columns=len(classes),
    )
    source_no_op_loss = _validated_true_class_log_loss(
        source["no_op_true_class_log_loss"],
        name="source no-op true_class_log_loss",
        probabilities=source_no_op_probabilities,
        true_class_indices=true_class_indices,
    )
    vectors = {
        "target_damage": target["loss"] - no_op["loss"],
        "matched_control_damage": control["loss"] - no_op["loss"],
        "donor_rescue": no_op["loss"] - donor["loss"],
        "donor_specificity": donor_control["loss"] - donor["loss"],
        "source_native_advantage": native["loss"] - source_native["loss"],
        "source_no_op_advantage": no_op["loss"] - source_no_op_loss,
    }
    diagnostic_vectors = {
        **vectors,
        "donor_improvement_vs_target": target["loss"] - donor["loss"],
        "donor_native_distance_reduction_vs_target": (
            np.abs(target["loss"] - native["loss"])
            - np.abs(donor["loss"] - native["loss"])
        ),
    }
    explicit = _exact_object(
        predictions["paired_reverse_patch_measured_effects"],
        label="paired reverse prediction effects",
        fields={
            "log_loss_improvement_vs_recipient_no_op",
            "log_loss_improvement_vs_paired_matched_random_patch",
            "log_loss_improvement_vs_target_baseline_edit",
            "native_distance_reduction_vs_target_baseline_edit",
            "log_loss_improvement_of_source_native_vs_recipient_native",
            "log_loss_improvement_of_source_no_op_vs_recipient_no_op",
        },
    )
    explicit_pairs = {
        "log_loss_improvement_vs_recipient_no_op": "donor_rescue",
        "log_loss_improvement_vs_paired_matched_random_patch": (
            "donor_specificity"
        ),
        "log_loss_improvement_vs_target_baseline_edit": (
            "donor_improvement_vs_target"
        ),
        "native_distance_reduction_vs_target_baseline_edit": (
            "donor_native_distance_reduction_vs_target"
        ),
        "log_loss_improvement_of_source_native_vs_recipient_native": (
            "source_native_advantage"
        ),
        "log_loss_improvement_of_source_no_op_vs_recipient_no_op": (
            "source_no_op_advantage"
        ),
    }
    for published_name, derived_name in explicit_pairs.items():
        published = _float_vector(
            explicit[published_name],
            name=published_name,
            expected=len(sample_ids),
        )
        if not np.allclose(
            published, diagnostic_vectors[derived_name], rtol=1e-12, atol=1e-12
        ):
            raise ValueError("published paired effect differs from raw losses")
    return (
        {name: float(np.mean(vector)) for name, vector in vectors.items()},
        {
            "sample_ids_sha256": _canonical_sha256(sample_ids),
            "true_labels_sha256": _canonical_sha256(true_labels),
            "classes_sha256": _canonical_sha256(predictions["classes"]),
            "sample_count": len(sample_ids),
            "invariant_prediction_content_sha256": _canonical_sha256(
                {
                    "sample_ids": sample_ids,
                    "true_labels": true_labels,
                    "classes": predictions["classes"],
                    "native": predictions["native"],
                    "no_op_reconstruction": {
                        name: conditions["no_op_reconstruction"][name]
                        for name in (
                            "probabilities",
                            "predicted_class_indices",
                            "predicted_labels",
                            "true_class_log_loss",
                        )
                    },
                    "paired_source_native": predictions[
                        "paired_source_native"
                    ],
                }
            ),
        },
    )


def _prediction_record(
    value: Any,
    *,
    label: str,
    expected: int,
    classes: Sequence[Any],
    true_class_indices: np.ndarray,
) -> dict[str, Any]:
    record = _exact_object(
        value,
        label=label,
        fields={
            "probabilities",
            "predicted_class_indices",
            "predicted_labels",
            "true_class_log_loss",
        },
    )
    probabilities = _probability_matrix(
        record["probabilities"],
        name=f"{label} probabilities",
        expected_rows=expected,
        expected_columns=len(classes),
    )
    predicted_indices = _integer_vector(
        record["predicted_class_indices"],
        name=f"{label} predicted_class_indices",
        expected=expected,
    )
    recomputed_indices = np.argmax(probabilities, axis=1)
    if not np.array_equal(predicted_indices, recomputed_indices):
        raise ValueError(f"{label} predicted indices differ from probabilities")
    predicted_labels = record["predicted_labels"]
    if not isinstance(predicted_labels, list) or len(predicted_labels) != expected:
        raise ValueError(f"{label} predicted_labels must be a complete vector")
    expected_label_keys = [
        _label_key(classes[index], name=f"{label} expected label")
        for index in recomputed_indices
    ]
    observed_label_keys = [
        _label_key(item, name=f"{label} predicted label")
        for item in predicted_labels
    ]
    if observed_label_keys != expected_label_keys:
        raise ValueError(f"{label} predicted labels differ from probabilities")
    return {
        "probabilities": probabilities,
        "loss": _validated_true_class_log_loss(
            record["true_class_log_loss"],
            name=f"{label} true_class_log_loss",
            probabilities=probabilities,
            true_class_indices=true_class_indices,
        ),
    }


def _condition_prediction_record(
    value: Any,
    *,
    label: str,
    expected: int,
    classes: Sequence[Any],
    true_class_indices: np.ndarray,
) -> dict[str, Any]:
    record = _exact_object(
        value,
        label=f"{label} prediction",
        fields={
            "probabilities",
            "predicted_class_indices",
            "predicted_labels",
            "true_class_log_loss",
            "delta_log_loss_vs_native",
            "delta_log_loss_vs_no_op",
            "probability_delta_vs_native",
            "probability_delta_vs_no_op",
            "target_features",
            "control_features",
        },
    )
    prediction = _prediction_record(
        {
            name: record[name]
            for name in (
                "probabilities",
                "predicted_class_indices",
                "predicted_labels",
                "true_class_log_loss",
            )
        },
        label=f"{label} prediction",
        expected=expected,
        classes=classes,
        true_class_indices=true_class_indices,
    )
    return {
        **prediction,
        "delta_log_loss_vs_native": _float_vector(
            record["delta_log_loss_vs_native"],
            name=f"{label} delta_log_loss_vs_native",
            expected=expected,
        ),
        "delta_log_loss_vs_no_op": _float_vector(
            record["delta_log_loss_vs_no_op"],
            name=f"{label} delta_log_loss_vs_no_op",
            expected=expected,
        ),
        "probability_delta_vs_native": _finite_matrix(
            record["probability_delta_vs_native"],
            name=f"{label} probability_delta_vs_native",
            expected_rows=expected,
            expected_columns=len(classes),
        ),
        "probability_delta_vs_no_op": _finite_matrix(
            record["probability_delta_vs_no_op"],
            name=f"{label} probability_delta_vs_no_op",
            expected_rows=expected,
            expected_columns=len(classes),
        ),
        "target_features": record["target_features"],
        "control_features": record["control_features"],
    }


def _validate_condition_prediction_deltas(
    condition: Mapping[str, Any],
    *,
    native: Mapping[str, np.ndarray],
    no_op: Mapping[str, np.ndarray],
    label: str,
) -> None:
    expected = {
        "delta_log_loss_vs_native": condition["loss"] - native["loss"],
        "delta_log_loss_vs_no_op": condition["loss"] - no_op["loss"],
        "probability_delta_vs_native": (
            condition["probabilities"] - native["probabilities"]
        ),
        "probability_delta_vs_no_op": (
            condition["probabilities"] - no_op["probabilities"]
        ),
    }
    for name, recomputed in expected.items():
        if not np.allclose(
            condition[name], recomputed, rtol=1e-12, atol=1e-12
        ):
            raise ValueError(f"{label} {name} differs from raw probabilities")


def _validated_classes_and_labels(
    raw_classes: Any, raw_labels: Any
) -> tuple[tuple[Any, ...], np.ndarray]:
    if not isinstance(raw_classes, list) or not raw_classes:
        raise ValueError("prediction classes must be a non-empty list")
    classes = tuple(raw_classes)
    keys = [_label_key(item, name="prediction class") for item in classes]
    if len(keys) != len(set(keys)):
        raise ValueError("prediction classes must be unique")
    for left_index, left in enumerate(classes):
        for right in classes[left_index + 1 :]:
            try:
                ambiguous = bool(left == right)
            except (TypeError, ValueError):
                ambiguous = False
            if ambiguous:
                raise ValueError("prediction classes contain type-ambiguous duplicates")
    if not isinstance(raw_labels, list) or not raw_labels:
        raise ValueError("true_labels must be a non-empty list")
    index_by_key = {key: index for index, key in enumerate(keys)}
    indices: list[int] = []
    for label in raw_labels:
        key = _label_key(label, name="true label")
        if key not in index_by_key:
            raise ValueError("true label is not present in prediction classes")
        indices.append(index_by_key[key])
    return classes, np.asarray(indices, dtype=np.int64)


def _label_key(value: Any, *, name: str) -> str:
    if value is None or isinstance(value, (list, dict)):
        raise ValueError(f"{name} must be a scalar JSON label")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if not isinstance(value, (bool, int, float, str)):
        raise ValueError(f"{name} has an unsupported type")
    return f"{type(value).__name__}:{_canonical_json(value)}"


def _validated_true_class_log_loss(
    value: Any,
    *,
    name: str,
    probabilities: np.ndarray,
    true_class_indices: np.ndarray,
) -> np.ndarray:
    published = _float_vector(value, name=name, expected=probabilities.shape[0])
    recomputed = -np.log(
        np.clip(
            probabilities[np.arange(probabilities.shape[0]), true_class_indices],
            1e-15,
            1.0,
        )
    )
    if not np.allclose(published, recomputed, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{name} differs from raw probabilities and true labels")
    return recomputed


def _probability_matrix(
    value: Any, *, name: str, expected_rows: int, expected_columns: int
) -> np.ndarray:
    array = _finite_matrix(
        value,
        name=name,
        expected_rows=expected_rows,
        expected_columns=expected_columns,
    )
    if (
        np.any(array < 0.0)
        or np.any(array > 1.0)
        or not np.allclose(
            np.sum(array, axis=1), 1.0, rtol=1e-7, atol=1e-7
        )
    ):
        raise ValueError(f"{name} must contain normalized probabilities")
    return array


def _finite_matrix(
    value: Any, *, name: str, expected_rows: int, expected_columns: int
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.shape != (expected_rows, expected_columns)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a complete finite matrix")
    return array


def _integer_vector(value: Any, *, name: str, expected: int) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != expected
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{name} must be a complete integer vector")
    return np.asarray(value, dtype=np.int64)


def _float_vector(value: Any, *, name: str, expected: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size != expected or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a complete aligned vector")
    return array


def _require_close(actual: float, expected: float, *, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{name} summary differs from recomputed predictions")


def _validated_raw_input_bindings(
    value: Any, *, registered_inputs: Mapping[str, str]
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("raw_dataset_input_sha256 must be a non-empty object")
    result: dict[str, str] = {}
    for raw_name, raw_digest in value.items():
        name = require_portable_identifier(raw_name, name="raw dataset input name")
        digest = _required_sha256(raw_digest, name="raw dataset input SHA-256")
        if registered_inputs.get(f"talent.raw.{name}") != digest:
            raise ValueError("raw dataset input is not manifest-bound")
        result[name] = digest
    normalized = dict(sorted(result.items()))
    registered_raw = {
        role.removeprefix("talent.raw."): digest
        for role, digest in registered_inputs.items()
        if role.startswith("talent.raw.")
    }
    if normalized != dict(sorted(registered_raw.items())):
        raise ValueError("raw dataset input mapping is incomplete or has extra entries")
    return normalized


def _dataset_fingerprint_mapping(
    alignment_by_dataset: Mapping[str, Mapping[str, Any]], *, label: str
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for dataset_id, alignment in sorted(alignment_by_dataset.items()):
        raw_digest = _required_sha256(
            alignment.get("raw_dataset_content_sha256"),
            name=f"{label} raw dataset fingerprint",
        )
        prediction_digest = _required_sha256(
            alignment.get("invariant_prediction_content_sha256"),
            name=f"{label} prediction content fingerprint",
        )
        result[dataset_id] = {
            "raw_dataset_content_sha256": raw_digest,
            "invariant_prediction_content_sha256": prediction_digest,
            "combined_dataset_fingerprint_sha256": _canonical_sha256(
                {
                    "raw_dataset_content_sha256": raw_digest,
                    "invariant_prediction_content_sha256": prediction_digest,
                }
            ),
        }
    for name in (
        "raw_dataset_content_sha256",
        "invariant_prediction_content_sha256",
        "combined_dataset_fingerprint_sha256",
    ):
        values = [fingerprints[name] for fingerprints in result.values()]
        if len(values) != len(set(values)):
            raise ValueError(f"{label} datasets contain a fingerprint alias")
    return result


def _paired_reverse_evidence(
    summary: Mapping[str, Any],
    *,
    parameters: SelectionParameters,
    registered_inputs: Mapping[str, str],
    recipient_model_sha: str,
    recipient_inference_contract_sha256: str,
    representation_source_lineage: Mapping[str, Any],
) -> tuple[float, float, float, float, float, dict[str, Any]]:
    paired = _exact_object(
        summary.get("paired_reverse_patch"),
        label="paired_reverse_patch",
        fields={
            "status",
            "evidence_scope",
            "direction",
            "condition_names",
            "alignment",
            "source_binding",
            "measured_effects",
            "donor_shift_balance",
            "source_native",
            "source_no_op_gates",
            "ablation_displacement_balance",
            "donor_displacement_balance",
        },
    )
    if (
        paired["status"] != "measured_diagnostic_only"
        or paired["evidence_scope"] != "single_dataset_measurement"
    ):
        raise ValueError("selection requires measured paired donor evidence")
    source_native = _exact_object(
        paired["source_native"],
        label="paired source_native",
        fields={"accuracy", "log_loss"},
    )
    source_accuracy = _finite_float(
        source_native["accuracy"], name="source native accuracy"
    )
    source_log_loss = _finite_float(
        source_native["log_loss"], name="source native log_loss"
    )
    if not 0.0 <= source_accuracy <= 1.0 or source_log_loss < 0.0:
        raise ValueError("paired source native metrics are outside valid bounds")
    _validate_source_no_op_gates(paired["source_no_op_gates"])
    direction = _exact_object(
        paired["direction"],
        label="paired reverse direction",
        fields={"source_condition", "recipient_condition"},
    )
    expected_direction = {
        "source_condition": parameters.source_condition,
        "recipient_condition": parameters.recipient_condition,
    }
    if direction != expected_direction:
        raise ValueError(
            "paired reverse direction differs from pre-registration"
        )
    condition_names = _exact_object(
        paired["condition_names"],
        label="paired reverse condition_names",
        fields={"source_target_feature_patch", "source_control_feature_patch"},
    )
    if condition_names != {
        "source_target_feature_patch": "paired_reverse_patch",
        "source_control_feature_patch": "paired_matched_random_patch",
    }:
        raise ValueError("paired reverse condition names are not canonical")
    alignment = _exact_object(
        paired["alignment"],
        label="paired reverse alignment",
        fields={
            "verified",
            "sample_roster_sha256",
            "inference_contract_sha256",
        },
    )
    if alignment["verified"] is not True:
        raise ValueError("paired reverse source alignment was not verified")
    sample_roster_sha256 = _required_sha256(
        alignment["sample_roster_sha256"], name="paired sample_roster_sha256"
    )
    if registered_inputs.get("samples.roster") != sample_roster_sha256:
        raise ValueError("paired source sample roster differs from recipient roster")
    source_binding = _exact_object(
        paired["source_binding"],
        label="paired reverse source_binding",
        fields={
            "model_sha",
            "checkpoint_sha256",
            "code_attestation_sha256",
            "inference_contract_sha256",
            "representation_source_lineage_sha256",
        },
    )
    normalized_source = {
        "model_sha": _required_git_sha(
            source_binding["model_sha"], name="paired source model_sha"
        ),
        "checkpoint_sha256": _required_sha256(
            source_binding["checkpoint_sha256"],
            name="paired source checkpoint_sha256",
        ),
        "code_attestation_sha256": _required_sha256(
            source_binding["code_attestation_sha256"],
            name="paired source code_attestation_sha256",
        ),
        "inference_contract_sha256": _required_sha256(
            source_binding["inference_contract_sha256"],
            name="paired source inference_contract_sha256",
        ),
        "representation_source_lineage_sha256": _required_sha256(
            source_binding["representation_source_lineage_sha256"],
            name="paired source representation_source_lineage_sha256",
        ),
    }
    if normalized_source["model_sha"] != recipient_model_sha:
        raise ValueError("paired source model SHA differs from recipient model code")
    if registered_inputs.get("paired_source.checkpoint") != (
        normalized_source["checkpoint_sha256"]
    ):
        raise ValueError("paired source checkpoint is not manifest-bound")
    if registered_inputs.get("paired_source.code_attestation") != (
        normalized_source["code_attestation_sha256"]
    ):
        raise ValueError("paired source code attestation is not manifest-bound")
    if (
        alignment["inference_contract_sha256"]
        != normalized_source["inference_contract_sha256"]
        or normalized_source["inference_contract_sha256"]
        != recipient_inference_contract_sha256
    ):
        raise ValueError("paired source inference contract is not aligned")
    if normalized_source["representation_source_lineage_sha256"] != (
        _canonical_sha256(representation_source_lineage)
    ):
        raise ValueError("paired source representation lineage is not shared")
    condition_checkpoints = representation_source_lineage.get(
        "condition_checkpoints_sha256"
    )
    if (
        not isinstance(condition_checkpoints, Mapping)
        or condition_checkpoints.get(parameters.source_condition)
        != normalized_source["checkpoint_sha256"]
    ):
        raise ValueError(
            "paired source checkpoint differs from representation condition lineage"
        )
    effects = _exact_object(
        paired["measured_effects"],
        label="paired reverse measured_effects",
        fields={
            "mean_log_loss_improvement_vs_recipient_no_op",
            "mean_log_loss_improvement_vs_paired_matched_random_patch",
            "mean_log_loss_improvement_vs_target_baseline_edit",
            "mean_native_distance_reduction_vs_target_baseline_edit",
            "mean_log_loss_improvement_of_source_native_vs_recipient_native",
            "mean_log_loss_improvement_of_source_no_op_vs_recipient_no_op",
            "donor_patch_gap_closure_denominator",
            "donor_patch_gap_closure_fraction",
        },
    )
    source_native_advantage = _finite_float(
        effects[
            "mean_log_loss_improvement_of_source_native_vs_recipient_native"
        ],
        name="mean_log_loss_improvement_of_source_native_vs_recipient_native",
    )
    source_no_op_advantage = _finite_float(
        effects[
            "mean_log_loss_improvement_of_source_no_op_vs_recipient_no_op"
        ],
        name="mean_log_loss_improvement_of_source_no_op_vs_recipient_no_op",
    )
    if effects["donor_patch_gap_closure_denominator"] != (
        "source_no_op_vs_recipient_no_op"
    ):
        raise ValueError("donor gap-closure denominator is not canonical")
    closure_fraction = effects["donor_patch_gap_closure_fraction"]
    if source_no_op_advantage > 0.0:
        _finite_float(
            closure_fraction, name="donor_patch_gap_closure_fraction"
        )
    elif closure_fraction is not None:
        raise ValueError(
            "donor patch gap-closure fraction requires positive source advantage"
        )
    shift_ratio = _validated_donor_shift_balance(
        paired["donor_shift_balance"], parameters=parameters
    )
    _validated_displacement_balance(
        paired["ablation_displacement_balance"],
        parameters=parameters,
        label="ablation_displacement_balance",
    )
    _validated_displacement_balance(
        paired["donor_displacement_balance"],
        parameters=parameters,
        label="donor_displacement_balance",
    )
    return (
        _finite_float(effects[_RESCUE_METRIC], name=_RESCUE_METRIC),
        _finite_float(
            effects[_DONOR_SPECIFICITY_METRIC],
            name=_DONOR_SPECIFICITY_METRIC,
        ),
        source_native_advantage,
        source_no_op_advantage,
        shift_ratio,
        normalized_source,
    )


def _validate_source_no_op_gates(value: Any) -> None:
    gates = _exact_object(
        value,
        label="paired source_no_op_gates",
        fields={
            "passed",
            "reconstruction_mse",
            "absolute_accuracy_difference",
            "maximum_absolute_probability_difference",
            "thresholds",
        },
    )
    if gates["passed"] is not True:
        raise ValueError("paired source no-op gates did not pass")
    thresholds = _exact_object(
        gates["thresholds"],
        label="paired source no-op thresholds",
        fields={
            "max_no_op_reconstruction_mse",
            "max_no_op_probability_deviation",
            "max_no_op_accuracy_difference",
        },
    )
    maxima = {
        "max_no_op_reconstruction_mse": 0.01,
        "max_no_op_probability_deviation": 0.02,
        "max_no_op_accuracy_difference": 0.005,
    }
    normalized: dict[str, float] = {}
    for name, maximum in maxima.items():
        normalized[name] = _finite_float(thresholds[name], name=name)
        if not 0.0 <= normalized[name] <= maximum:
            raise ValueError("paired source no-op threshold exceeds protocol")
    measured = {
        "reconstruction_mse": normalized["max_no_op_reconstruction_mse"],
        "maximum_absolute_probability_difference": normalized[
            "max_no_op_probability_deviation"
        ],
        "absolute_accuracy_difference": normalized[
            "max_no_op_accuracy_difference"
        ],
    }
    for name, limit in measured.items():
        observed = _finite_float(gates[name], name=name)
        if observed < 0.0 or observed > limit:
            raise ValueError("paired source no-op measurement exceeds threshold")


def _validated_donor_shift_balance(
    value: Any, *, parameters: SelectionParameters
) -> float:
    balance = _exact_object(
        value,
        label="donor_shift_balance",
        fields={
            "target_feature_shift_rms",
            "control_feature_shift_rms",
            "symmetric_rms_ratio",
            "maximum_symmetric_rms_ratio",
            "passed",
            "by_call",
        },
    )
    threshold = _finite_float(
        balance["maximum_symmetric_rms_ratio"],
        name="maximum_symmetric_rms_ratio",
    )
    if threshold != parameters.maximum_symmetric_donor_shift_rms_ratio:
        raise ValueError(
            "donor shift threshold differs from selection pre-registration"
        )
    ratio = _finite_float(
        balance["symmetric_rms_ratio"], name="symmetric_rms_ratio"
    )
    for name in ("target_feature_shift_rms", "control_feature_shift_rms"):
        if _finite_float(balance[name], name=name) < 0.0:
            raise ValueError("donor shift RMS values must be non-negative")
    if ratio < 1.0 or ratio > threshold or balance["passed"] is not True:
        raise ValueError("donor shift balance did not pass its frozen threshold")
    by_call = balance["by_call"]
    if not isinstance(by_call, list) or not by_call:
        raise ValueError("donor shift balance requires per-call diagnostics")
    seen_calls: set[int] = set()
    for raw_call in by_call:
        call = _exact_object(
            raw_call,
            label="donor shift call",
            fields={
                "call_index",
                "vector_count",
                "target_feature_shift_rms",
                "control_feature_shift_rms",
                "symmetric_rms_ratio",
            },
        )
        call_index = _non_negative_int(call["call_index"], name="call_index")
        if call_index in seen_calls:
            raise ValueError("donor shift call indices must be unique")
        seen_calls.add(call_index)
        _positive_int(call["vector_count"], name="vector_count")
        for name in ("target_feature_shift_rms", "control_feature_shift_rms"):
            if _finite_float(call[name], name=name) < 0.0:
                raise ValueError("donor shift RMS values must be non-negative")
        call_ratio = _finite_float(
            call["symmetric_rms_ratio"], name="symmetric_rms_ratio"
        )
        if call_ratio < 1.0 or call_ratio > threshold:
            raise ValueError("a donor shift call exceeds the frozen threshold")
    if seen_calls != set(range(len(seen_calls))):
        raise ValueError("donor shift call indices must be contiguous")
    return ratio


def _validated_displacement_balance(
    value: Any, *, parameters: SelectionParameters, label: str
) -> float:
    balance = _exact_object(
        value,
        label=label,
        fields={
            "target_displacement_rms",
            "control_displacement_rms",
            "symmetric_rms_ratio",
            "maximum_symmetric_rms_ratio",
            "passed",
            "by_call",
        },
    )
    threshold = _finite_float(
        balance["maximum_symmetric_rms_ratio"],
        name=f"{label}.maximum_symmetric_rms_ratio",
    )
    ratio = _finite_float(
        balance["symmetric_rms_ratio"], name=f"{label}.symmetric_rms_ratio"
    )
    if (
        threshold != parameters.maximum_symmetric_donor_shift_rms_ratio
        or not 1.0 <= ratio <= threshold
        or balance["passed"] is not True
    ):
        raise ValueError(f"{label} did not pass the frozen threshold")
    for name in ("target_displacement_rms", "control_displacement_rms"):
        if _finite_float(balance[name], name=f"{label}.{name}") < 0.0:
            raise ValueError(f"{label} RMS values must be non-negative")
    by_call = balance["by_call"]
    if not isinstance(by_call, list) or not by_call:
        raise ValueError(f"{label} requires per-call diagnostics")
    seen: set[int] = set()
    for raw_call in by_call:
        call = _exact_object(
            raw_call,
            label=f"{label} call",
            fields={
                "call_index",
                "vector_count",
                "target_displacement_rms",
                "control_displacement_rms",
                "symmetric_rms_ratio",
            },
        )
        call_index = _non_negative_int(call["call_index"], name="call_index")
        if call_index in seen:
            raise ValueError(f"{label} call indices must be unique")
        seen.add(call_index)
        _positive_int(call["vector_count"], name="vector_count")
        for name in ("target_displacement_rms", "control_displacement_rms"):
            if _finite_float(call[name], name=f"{label}.{name}") < 0.0:
                raise ValueError(f"{label} RMS values must be non-negative")
        call_ratio = _finite_float(
            call["symmetric_rms_ratio"], name=f"{label}.symmetric_rms_ratio"
        )
        if not 1.0 <= call_ratio <= threshold:
            raise ValueError(f"a {label} call exceeds the frozen threshold")
    if seen != set(range(len(seen))):
        raise ValueError(f"{label} call indices must be contiguous")
    return ratio


def _selection_parameters(value: Any) -> SelectionParameters:
    values = _exact_object(value, label="selection", fields=_SELECTION_FIELDS)
    if values["metric"] != _METRIC:
        raise ValueError(f"selection metric must be exactly {_METRIC!r}")
    if values["direction"] != _DIRECTION:
        raise ValueError(f"selection direction must be exactly {_DIRECTION!r}")
    if values["ranking_rule"] != "maximum_minimum_mean_evidence":
        raise ValueError(
            "ranking_rule must be exactly 'maximum_minimum_mean_evidence'"
        )
    if values["multiplicity_method"] != "benjamini-yekutieli":
        raise ValueError(
            "multiplicity_method must be exactly 'benjamini-yekutieli'"
        )
    raw_family = values["evidence_family"]
    if not isinstance(raw_family, list) or raw_family != list(_EVIDENCE_FAMILY):
        raise ValueError(
            "evidence_family must declare the complete canonical mechanism family"
        )
    paired_direction = _exact_object(
        values["paired_source_direction"],
        label="paired_source_direction",
        fields={"source_condition", "recipient_condition"},
    )
    source_condition = paired_direction["source_condition"]
    recipient_condition = paired_direction["recipient_condition"]
    if (
        source_condition not in _CANONICAL_CONDITIONS
        or recipient_condition not in _CANONICAL_CONDITIONS
        or source_condition == recipient_condition
    ):
        raise ValueError(
            "paired source and recipient must be distinct canonical conditions"
        )
    alpha = _finite_float(values["fdr_alpha"], name="fdr_alpha")
    confidence = _finite_float(
        values["confidence_level"], name="confidence_level"
    )
    replication = _finite_float(
        values["minimum_positive_fraction"],
        name="minimum_positive_fraction",
    )
    if not 0.0 < alpha <= 1.0:
        raise ValueError("fdr_alpha must lie in (0, 1]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if not 0.0 <= replication <= 1.0:
        raise ValueError("minimum_positive_fraction must lie in [0, 1]")
    minimum_datasets = _positive_int(
        values["minimum_validation_datasets"],
        name="minimum_validation_datasets",
    )
    if minimum_datasets < 8:
        raise ValueError("minimum_validation_datasets must be at least eight")
    shift_ratio = _finite_float(
        values["maximum_symmetric_donor_shift_rms_ratio"],
        name="maximum_symmetric_donor_shift_rms_ratio",
    )
    if not 1.0 <= shift_ratio <= 1.25:
        raise ValueError(
            "maximum_symmetric_donor_shift_rms_ratio must lie in [1, 1.25]"
        )
    maximum_selections = _positive_int(
        values["maximum_selections"], name="maximum_selections"
    )
    if maximum_selections > 2:
        raise ValueError("maximum_selections must be at most two")
    return SelectionParameters(
        fdr_alpha=alpha,
        confidence_level=confidence,
        bootstrap_resamples=_positive_int(
            values["bootstrap_resamples"], name="bootstrap_resamples"
        ),
        sign_flip_resamples=_positive_int(
            values["sign_flip_resamples"], name="sign_flip_resamples"
        ),
        random_seed=_non_negative_int(values["random_seed"], name="random_seed"),
        minimum_validation_datasets=minimum_datasets,
        minimum_positive_fraction=replication,
        maximum_selections=maximum_selections,
        source_condition=source_condition,
        recipient_condition=recipient_condition,
        evidence_family=tuple(dict(item) for item in raw_family),
        maximum_symmetric_donor_shift_rms_ratio=shift_ratio,
    )


def _candidate_specifications(
    value: Any, *, minimum_validation_datasets: int
) -> tuple[CandidateSpecification, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("candidates must be a non-empty JSON list")
    result: list[CandidateSpecification] = []
    seen_ids: set[str] = set()
    for raw in value:
        candidate = _exact_object(raw, label="candidate", fields=_CANDIDATE_FIELDS)
        candidate_id = require_portable_identifier(
            candidate["candidate_id"], name="candidate_id"
        )
        if candidate_id in seen_ids:
            raise ValueError("candidate_id values must be unique")
        seen_ids.add(candidate_id)
        targets = _feature_indices(candidate["target_features"], name="target_features")
        controls = _feature_indices(
            candidate["control_features"], name="control_features"
        )
        if len(targets) != len(controls) or set(targets) & set(controls):
            raise ValueError(
                "target/control features must be disjoint and have equal size"
            )
        baseline = _numerical_baseline(candidate["latent_baseline"])
        raw_runs = candidate["validation_runs"]
        if not isinstance(raw_runs, list) or len(raw_runs) < minimum_validation_datasets:
            raise ValueError(
                "each candidate needs at least minimum_validation_datasets runs"
            )
        runs: list[ValidationRunSpecification] = []
        seen_datasets: set[str] = set()
        for raw_run in raw_runs:
            run = _exact_object(raw_run, label="validation run", fields=_RUN_FIELDS)
            dataset_id = require_public_label(
                run["dataset_id"], name="validation dataset_id"
            )
            if dataset_id in seen_datasets:
                raise ValueError(
                    "validation dataset identities must be unique per candidate"
                )
            seen_datasets.add(dataset_id)
            path = _absolute_directory(run["run_dir"], name="run_dir")
            runs.append(
                ValidationRunSpecification(
                    dataset_id=dataset_id,
                    run_dir=path,
                    expected_manifest_sha256=_required_sha256(
                        run["expected_manifest_sha256"],
                        name="expected_manifest_sha256",
                    ),
                )
            )
        result.append(
            CandidateSpecification(
                candidate_id=candidate_id,
                target_features=targets,
                control_features=controls,
                latent_baseline=baseline,
                validation_runs=tuple(runs),
            )
        )
    return tuple(result)


def _heldout_specifications(
    value: Any, *, random_seed: int
) -> tuple[tuple[HeldoutRosterSpecification, ...], ConfirmationParameters]:
    values = _exact_object(value, label="heldout", fields=_HELDOUT_FIELDS)
    raw_rosters = values["sample_rosters"]
    if not isinstance(raw_rosters, list) or not raw_rosters:
        raise ValueError("heldout.sample_rosters must be a non-empty list")
    result: list[HeldoutRosterSpecification] = []
    seen: set[str] = set()
    for raw in raw_rosters:
        roster = _exact_object(
            raw, label="heldout roster", fields=_HELDOUT_ROSTER_FIELDS
        )
        dataset_id = require_public_label(
            roster["dataset_id"], name="heldout dataset_id"
        )
        if dataset_id in seen:
            raise ValueError("heldout dataset identities must be unique")
        seen.add(dataset_id)
        result.append(
            HeldoutRosterSpecification(
                dataset_id=dataset_id,
                path=_absolute_file(
                    roster["sample_roster_path"], name="sample_roster_path"
                ),
                expected_sha256=_required_sha256(
                    roster["expected_sample_roster_sha256"],
                    name="expected_sample_roster_sha256",
                ),
            )
        )
    confirmation_values = _exact_object(
        values["confirmation"],
        label="heldout confirmation",
        fields=_CONFIRMATION_FIELDS,
    )
    if confirmation_values["multiplicity_method"] != "holm":
        raise ValueError("heldout confirmation multiplicity_method must be 'holm'")
    if confirmation_values["bootstrap_method"] != "paired-dataset-bootstrap":
        raise ValueError(
            "heldout confirmation bootstrap_method must be canonical"
        )
    confirmation_seed = _non_negative_int(
        confirmation_values["random_seed"],
        name="confirmation random_seed",
    )
    if confirmation_seed != random_seed:
        raise ValueError(
            "heldout confirmation random_seed must equal selection random_seed"
        )
    alpha = _finite_float(confirmation_values["alpha"], name="confirmation alpha")
    confidence = _finite_float(
        confirmation_values["confidence_level"],
        name="confirmation confidence_level",
    )
    replication = _finite_float(
        confirmation_values["minimum_positive_fraction"],
        name="confirmation minimum_positive_fraction",
    )
    if not 0.0 < alpha <= 1.0:
        raise ValueError("confirmation alpha must lie in (0, 1]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confirmation confidence_level must lie in (0, 1)")
    if not 0.0 <= replication <= 1.0:
        raise ValueError(
            "confirmation minimum_positive_fraction must lie in [0, 1]"
        )
    minimum = _positive_int(
        confirmation_values["minimum_heldout_datasets"],
        name="minimum_heldout_datasets",
    )
    if minimum < 8:
        raise ValueError("minimum_heldout_datasets must be at least eight")
    return (
        tuple(result),
        ConfirmationParameters(
            alpha=alpha,
            confidence_level=confidence,
            sign_flip_resamples=_positive_int(
                confirmation_values["sign_flip_resamples"],
                name="confirmation sign_flip_resamples",
            ),
            bootstrap_resamples=_positive_int(
                confirmation_values["bootstrap_resamples"],
                name="confirmation bootstrap_resamples",
            ),
            minimum_positive_fraction=replication,
            minimum_heldout_datasets=minimum,
            random_seed=confirmation_seed,
        ),
    )


def _load_heldout_roster(raw: bytes) -> dict[str, Any]:
    roster = _exact_object(
        _load_json_object(raw, label="heldout sample roster"),
        label="heldout sample roster",
        fields={"dataset_id", "split", "row_indices", "sample_ids"},
    )
    dataset_id = require_public_label(
        roster["dataset_id"], name="heldout dataset_id"
    )
    if roster["split"] != "test":
        raise ValueError("heldout sample roster split must be exactly 'test'")
    rows = roster["row_indices"]
    sample_ids = roster["sample_ids"]
    if (
        not isinstance(rows, list)
        or not rows
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in rows)
        or len(rows) != len(set(rows))
    ):
        raise ValueError("heldout row_indices must be unique non-negative integers")
    if not isinstance(sample_ids, list) or len(sample_ids) != len(rows):
        raise ValueError("heldout sample_ids must align with row_indices")
    normalized_ids = [
        require_public_label(item, name="heldout sample_id") for item in sample_ids
    ]
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("heldout sample_ids must be unique")
    return {"dataset_id": dataset_id}


def _selection_payload(
    *,
    condition: str,
    site: str,
    parameters: SelectionParameters,
    common_lineage: Mapping[str, Any],
    validation_dataset_ids: Sequence[str],
    heldout_rosters_sha256: Mapping[str, str],
    family_size: int,
    harmonic_factor: float,
    required_validation_datasets: int,
    validation_dataset_fingerprints: Mapping[str, Mapping[str, str]],
    validation_dataset_fingerprint_manifest_sha256: str,
    source_advantage: Mapping[str, Any],
    confirmation_protocol: Mapping[str, Any],
    candidate_results: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "analysis": "validation-feature-selection",
        "evidence_scope": "validation-selection",
        "condition": condition,
        "site": site,
        "random_seed": parameters.random_seed,
        "evidence_family": [dict(item) for item in parameters.evidence_family],
        "statistics": {
            "multiplicity_method": "benjamini-yekutieli",
            "fdr_control_unit": "candidate-intersection-union-hypothesis",
            "candidate_composite_method": "maximum-of-six-component-p-values",
            "component_count_per_candidate": 6,
            "fdr_alpha": parameters.fdr_alpha,
            "confidence_level": parameters.confidence_level,
            "bootstrap_method": "paired-dataset-bootstrap",
            "bootstrap_resamples": parameters.bootstrap_resamples,
            "p_value_method": "one-sided-paired-sign-flip",
            "sign_flip_resamples": parameters.sign_flip_resamples,
            "minimum_validation_datasets": parameters.minimum_validation_datasets,
            "minimum_positive_fraction": parameters.minimum_positive_fraction,
            "maximum_selections": parameters.maximum_selections,
            "ranking_rule": "maximum_minimum_mean_evidence",
            "family_size": family_size,
            "harmonic_factor": harmonic_factor,
            "rank_one_threshold": parameters.fdr_alpha
            / (family_size * harmonic_factor),
            "exact_sign_flip_minimum_p_value": 2.0
            ** (-len(validation_dataset_ids)),
            "required_validation_datasets": required_validation_datasets,
            "effective_validation_dataset_count": len(validation_dataset_ids),
            "sign_flip_mode": (
                "exact-enumeration"
                if len(validation_dataset_ids) <= 20
                else "fixed-seed-monte-carlo"
            ),
            "monte_carlo_minimum_p_value": (
                None
                if len(validation_dataset_ids) <= 20
                else 1.0 / (parameters.sign_flip_resamples + 1)
            ),
        },
        "source_advantage_prerequisite": dict(source_advantage),
        "common_lineage": _json_public_copy(common_lineage),
        "validation_dataset_ids": list(validation_dataset_ids),
        "validation_dataset_fingerprints_sha256": _json_public_copy(
            validation_dataset_fingerprints
        ),
        "validation_dataset_fingerprint_manifest_sha256": (
            validation_dataset_fingerprint_manifest_sha256
        ),
        "heldout": {
            "dataset_ids": sorted(heldout_rosters_sha256),
            "evaluation_sample_rosters_sha256": dict(
                sorted(heldout_rosters_sha256.items())
            ),
            "confirmation_protocol": dict(confirmation_protocol),
        },
        "candidate_results": list(candidate_results),
        "selected_candidates": [item["candidate_id"] for item in selected],
    }


def _confirmation_protocol_payload(
    parameters: ConfirmationParameters,
    *,
    required_heldout_datasets: int,
    sign_flip_mode: str,
    maximum_confirmation_candidates: int,
    frozen_candidate_count: int,
    sign_flip_minimum_p_value: float,
) -> dict[str, Any]:
    if sign_flip_mode not in {
        "exact-enumeration",
        "fixed-seed-monte-carlo",
    }:
        raise ValueError("confirmation sign-flip mode is not canonical")
    return {
        "alpha": parameters.alpha,
        "confidence_level": parameters.confidence_level,
        "sign_flip_resamples": parameters.sign_flip_resamples,
        "bootstrap_resamples": parameters.bootstrap_resamples,
        "bootstrap_method": "paired-dataset-bootstrap",
        "random_seed": parameters.random_seed,
        "minimum_positive_fraction": parameters.minimum_positive_fraction,
        "minimum_heldout_datasets": parameters.minimum_heldout_datasets,
        "required_heldout_datasets": required_heldout_datasets,
        "maximum_confirmation_candidates": maximum_confirmation_candidates,
        "frozen_candidate_count": frozen_candidate_count,
        "preregistered_holm_rank_one_threshold": (
            parameters.alpha / maximum_confirmation_candidates
        ),
        "actual_holm_rank_one_threshold": (
            None
            if frozen_candidate_count == 0
            else parameters.alpha / frozen_candidate_count
        ),
        "sign_flip_minimum_p_value": sign_flip_minimum_p_value,
        "sign_flip_mode": sign_flip_mode,
        "candidate_test": "intersection-union-max-p",
        "multiplicity_method": "holm",
        "top_k_after_freeze": False,
    }


def _exact_object(value: Any, *, label: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        raise ValueError(
            f"{label} fields mismatch: missing={missing}, unknown={unknown}"
        )
    return dict(value)


def _load_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _feature_indices(value: Any, *, name: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
    ):
        raise ValueError(f"{name} must be a non-empty list of non-negative integers")
    result = tuple(value)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _numerical_baseline(value: Any) -> float | tuple[float, ...]:
    if isinstance(value, bool):
        raise TypeError("latent_baseline must be numerical")
    if isinstance(value, (int, float)):
        return _finite_float(value, name="latent_baseline")
    if not isinstance(value, list) or not value:
        raise TypeError("latent_baseline must be a number or non-empty number list")
    return tuple(
        _finite_float(item, name="latent_baseline item") for item in value
    )


def _json_safe_baseline(value: float | tuple[float, ...]) -> float | list[float]:
    if isinstance(value, tuple):
        return list(value)
    return value


def _positive_int(value: Any, *, name: str) -> int:
    result = _non_negative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numerical")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _required_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _required_git_sha(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase Git object ID")
    return value


def _absolute_directory(value: Any, *, name: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{name} must be an absolute non-symlink directory")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{name} must be an existing directory")
    return resolved


def _absolute_file(value: Any, *, name: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{name} must be an absolute non-symlink file")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{name} must be an existing regular file")
    return resolved


def _json_public_copy(value: Any) -> Any:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return json.loads(encoded)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _harmonic_number(count: int) -> float:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("harmonic count must be a positive integer")
    return float(sum(1.0 / rank for rank in range(1, count + 1)))


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


__all__ = [
    "CandidateSpecification",
    "HeldoutRosterSpecification",
    "SelectionParameters",
    "ValidationObservation",
    "ValidationRunSpecification",
    "aggregate_validation_candidates",
    "run",
]
