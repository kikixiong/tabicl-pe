"""Confirm validation-frozen mechanism candidates on a held-out roster."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import feature_selection as selection
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
from .statistics import adjust_holm, paired_bootstrap_ci, paired_sign_flip_p_value


_TOP_LEVEL_FIELDS = {
    "provenance",
    "selection_run_dir",
    "expected_selection_manifest_sha256",
    "heldout_runs",
}
_HELDOUT_RUN_FIELDS = {
    "candidate_id",
    "dataset_id",
    "run_dir",
    "expected_manifest_sha256",
}
_SELECTION_ARTIFACTS = {"freeze.json", "selection.json", "summary.json"}
_ENDPOINT_METRICS = {
    "target_damage": "mean_delta_log_loss_vs_no_op",
    "ablation_specificity": (
        "target_minus_matched_control_mean_delta_log_loss_vs_no_op"
    ),
    "donor_rescue": "mean_log_loss_improvement_vs_recipient_no_op",
    "donor_specificity": (
        "mean_log_loss_improvement_vs_paired_matched_random_patch"
    ),
    "source_native_advantage": (
        "mean_log_loss_improvement_of_source_native_vs_recipient_native"
    ),
    "source_no_op_advantage": (
        "mean_log_loss_improvement_of_source_no_op_vs_recipient_no_op"
    ),
}


@dataclass(frozen=True)
class ConfirmationProtocol:
    alpha: float
    confidence_level: float
    sign_flip_resamples: int
    bootstrap_resamples: int
    minimum_positive_fraction: float
    minimum_heldout_datasets: int
    required_heldout_datasets: int
    maximum_confirmation_candidates: int
    frozen_candidate_count: int
    preregistered_holm_rank_one_threshold: float
    actual_holm_rank_one_threshold: float | None
    sign_flip_minimum_p_value: float
    sign_flip_mode: str
    random_seed: int


@dataclass(frozen=True)
class FrozenCandidate:
    candidate_id: str
    target_features: tuple[int, ...]
    control_features: tuple[int, ...]
    latent_baseline: float | tuple[float, ...]


@dataclass(frozen=True)
class HeldoutRunSpecification:
    candidate_id: str
    dataset_id: str
    run_dir: Path
    expected_manifest_sha256: str


@dataclass(frozen=True)
class PreparedHeldoutRun:
    specification: HeldoutRunSpecification
    manifest: RunManifest
    manifest_role: str
    summary_role: str
    predictions_role: str


def run(args: Any) -> int:
    """Strictly aggregate the complete frozen held-out Cartesian roster."""

    configuration = load_verified_json_config(Path(args.config))
    config = selection._exact_object(
        configuration.data,
        label="confirm-features config",
        fields=_TOP_LEVEL_FIELDS,
    )
    selection_dir = selection._absolute_directory(
        config["selection_run_dir"], name="selection_run_dir"
    )
    expected_selection_manifest = selection._required_sha256(
        config["expected_selection_manifest_sha256"],
        name="expected_selection_manifest_sha256",
    )
    selection_parent = verify_run_directory(selection_dir)
    selection_manifest_file = verify_file(
        selection_dir / "manifest.json",
        expected_sha256=expected_selection_manifest,
    )
    parsed_parent = load_verified_run_manifest(selection_manifest_file)
    if not isinstance(parsed_parent, RunManifest) or parsed_parent != selection_parent:
        raise RuntimeError("selection parent changed during directory verification")
    if (
        selection_parent.command != "select-features"
        or selection_parent.evidence_level != "strict"
        or selection_parent.legacy_reasons
    ):
        raise ValueError("confirmation requires a strict select-features parent")
    if {item.name for item in selection_parent.artifacts} != _SELECTION_ARTIFACTS:
        raise ValueError("selection parent artifacts are not canonical")

    additional_paths: dict[str, Path] = {
        "selection.parent_manifest": selection_manifest_file.path
    }
    expected_additional: dict[str, str] = {
        "selection.parent_manifest": selection_manifest_file.digest.sha256
    }
    for artifact in selection_parent.artifacts:
        verified = verify_file(
            selection_dir / artifact.name, expected_sha256=artifact.sha256
        )
        role = f"selection.{artifact.name.removesuffix('.json')}"
        additional_paths[role] = verified.path
        expected_additional[role] = verified.digest.sha256
    selection_payload = selection._load_json_object(
        verify_file(
            selection_dir / "selection.json",
            expected_sha256=expected_additional["selection.selection"],
        ).read_bytes(),
        label="selection.json",
    )
    freeze_payload = selection._load_json_object(
        verify_file(
            selection_dir / "freeze.json",
            expected_sha256=expected_additional["selection.freeze"],
        ).read_bytes(),
        label="freeze.json",
    )
    summary_payload = selection._load_json_object(
        verify_file(
            selection_dir / "summary.json",
            expected_sha256=expected_additional["selection.summary"],
        ).read_bytes(),
        label="selection summary",
    )
    frozen = _validated_selection_artifacts(
        selection_payload,
        freeze_payload,
        summary_payload,
        selection_sha256=expected_additional["selection.selection"],
    )
    candidates = frozen["candidates"]
    protocol = frozen["protocol"]
    roster_mapping = frozen["roster_mapping"]
    validation_dataset_fingerprints = frozen[
        "validation_dataset_fingerprints"
    ]
    specifications = _heldout_run_specifications(config["heldout_runs"])
    expected_pairs = {
        (candidate.candidate_id, dataset_id)
        for candidate in candidates
        for dataset_id in roster_mapping
    }
    observed_pairs = {
        (item.candidate_id, item.dataset_id) for item in specifications
    }
    if observed_pairs != expected_pairs or len(specifications) != len(expected_pairs):
        raise ValueError(
            "heldout runs must be the exact frozen candidate-by-dataset Cartesian set"
        )

    prepared: list[PreparedHeldoutRun] = []
    seen_parent_files: set[tuple[int, int]] = set()
    for specification in specifications:
        parent, roles = _prepare_heldout_parent(
            specification,
            additional_paths=additional_paths,
            expected_additional=expected_additional,
        )
        manifest_file = verify_file(specification.run_dir / "manifest.json")
        identity = (manifest_file.device, manifest_file.inode)
        if identity in seen_parent_files:
            raise ValueError("a heldout parent manifest cannot be reused")
        seen_parent_files.add(identity)
        prepared.append(
            PreparedHeldoutRun(
                specification=specification,
                manifest=parent,
                manifest_role=roles["manifest"],
                summary_role=roles["summary"],
                predictions_role=roles["predictions"],
            )
        )

    context = verify_configured_run_inputs(
        configuration,
        command="confirm-features",
        seed=freeze_payload["random_seed"],
        additional_input_paths=additional_paths,
        expected_additional_sha256=expected_additional,
    )
    if context.inputs.evidence_level != "strict":
        raise RuntimeError("confirm-features requires strict Git evidence")
    _validate_selection_manifest_lineage(
        selection_parent,
        context=context,
        freeze=freeze_payload,
    )
    assert_dataset_roster(
        context.inputs.dataset_manifest,
        tuple(roster_mapping),
        required_split="held_out",
    )

    observations: list[selection.ValidationObservation] = []
    candidate_map = {item.candidate_id: item for item in candidates}
    alignment_by_dataset: dict[str, dict[str, Any]] = {}
    for item in prepared:
        bound_manifest = load_verified_run_manifest(
            context.additional_file(item.manifest_role)
        )
        if not isinstance(bound_manifest, RunManifest) or bound_manifest != item.manifest:
            raise RuntimeError("heldout parent changed during input verification")
        observation, alignment = _extract_heldout_observation(
            item,
            manifest=bound_manifest,
            summary=selection._load_json_object(
                context.additional_file(item.summary_role).read_bytes(),
                label="heldout summary",
            ),
            predictions=selection._load_json_object(
                context.additional_file(item.predictions_role).read_bytes(),
                label="heldout predictions",
            ),
            candidate=candidate_map[item.specification.candidate_id],
            context=context,
            freeze=freeze_payload,
            selection_manifest_sha256=context.additional_file(
                "selection.parent_manifest"
            ).digest.sha256,
            selection_summary_sha256=context.additional_file(
                "selection.summary"
            ).digest.sha256,
            freeze_sha256=context.additional_file("selection.freeze").digest.sha256,
            roster_sha256=roster_mapping[item.specification.dataset_id],
        )
        previous = alignment_by_dataset.setdefault(
            item.specification.dataset_id, alignment
        )
        if previous != alignment:
            raise ValueError(
                "heldout candidates for one dataset have different raw/sample alignment"
            )
        observations.append(observation)

    heldout_evidence_consumed = bool(candidates)
    heldout_dataset_fingerprints = (
        selection._dataset_fingerprint_mapping(
            alignment_by_dataset,
            label="heldout",
        )
        if heldout_evidence_consumed
        else None
    )
    if heldout_dataset_fingerprints is not None:
        _assert_disjoint_dataset_fingerprints(
            validation_dataset_fingerprints,
            heldout_dataset_fingerprints,
        )
    heldout_fingerprint_manifest_sha256 = (
        selection._canonical_sha256(heldout_dataset_fingerprints)
        if heldout_dataset_fingerprints is not None
        else None
    )

    confirmation = confirm_frozen_candidates(
        candidates,
        observations,
        protocol=protocol,
    )
    confirmation_payload = {
        "schema_version": 1,
        "analysis": "heldout-feature-confirmation",
        "evidence_scope": (
            "confirmatory-cross-dataset"
            if heldout_evidence_consumed
            else "selection-terminated-no-heldout-evidence"
        ),
        "condition": context.condition,
        "site": context.sites[0],
        "random_seed": freeze_payload["random_seed"],
        "selection_parent_manifest_sha256": context.additional_file(
            "selection.parent_manifest"
        ).digest.sha256,
        "validation_selection_sha256": context.additional_file(
            "selection.selection"
        ).digest.sha256,
        "freeze_sha256": context.additional_file("selection.freeze").digest.sha256,
        "checkpoint_study_sha256": freeze_payload["checkpoint_study_sha256"],
        "preregistered_heldout_dataset_ids": list(roster_mapping),
        "heldout_dataset_ids": (
            list(roster_mapping) if heldout_evidence_consumed else []
        ),
        "heldout_dataset_fingerprints_sha256": heldout_dataset_fingerprints,
        "heldout_dataset_fingerprint_manifest_sha256": (
            heldout_fingerprint_manifest_sha256
        ),
        "confirmation_protocol": _protocol_payload(protocol),
        **confirmation,
    }
    summary = {
        "schema_version": 1,
        "analysis": "heldout-feature-confirmation",
        "evidence_scope": (
            "confirmatory-cross-dataset"
            if heldout_evidence_consumed
            else "selection-terminated-no-heldout-evidence"
        ),
        "status": confirmation["status"],
        "heldout_evidence_consumed": heldout_evidence_consumed,
        "condition": context.condition,
        "site": context.sites[0],
        "preregistered_heldout_dataset_count": len(roster_mapping),
        "heldout_dataset_count": (
            len(roster_mapping) if heldout_evidence_consumed else 0
        ),
        "candidate_count": len(candidates),
        "heldout_dataset_fingerprint_manifest_sha256": (
            heldout_fingerprint_manifest_sha256
        ),
        "confirmed_candidate_ids": [
            item["candidate_id"]
            for item in confirmation["candidate_results"]
            if item["confirmed"] is True
        ],
        "all_frozen_candidates_tested": True,
        "top_k_after_freeze": False,
        "any_candidate_confirmed": confirmation["any_candidate_confirmed"],
        "all_frozen_candidates_confirmed": confirmation[
            "all_frozen_candidates_confirmed"
        ],
        "selection_parent_manifest_sha256": context.additional_file(
            "selection.parent_manifest"
        ).digest.sha256,
        "freeze_sha256": context.additional_file("selection.freeze").digest.sha256,
        "checkpoint_study_sha256": freeze_payload["checkpoint_study_sha256"],
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
        selection._write_bytes(
            transaction.staging_dir / "confirmation.json",
            selection._json_bytes(confirmation_payload),
        )
        selection._write_bytes(
            transaction.staging_dir / "summary.json",
            selection._json_bytes(summary),
        )
        artifacts = transaction.artifact_digests(
            ("confirmation.json", "summary.json")
        )
        manifest = manifest_from_verified_inputs(context.inputs, artifacts=artifacts)
        transaction.commit(manifest, verified_inputs=context.inputs)
    return 0


def confirm_frozen_candidates(
    candidates: Sequence[FrozenCandidate],
    observations: Sequence[selection.ValidationObservation],
    *,
    protocol: ConfirmationProtocol,
) -> dict[str, Any]:
    """Intersection-union tests with Holm correction and no reselection."""

    candidate_ids = {item.candidate_id for item in candidates}
    if len(candidate_ids) != len(candidates):
        raise ValueError("confirmation candidate IDs must be unique")
    required = max(
        8,
        int(
            np.ceil(
                np.log2(
                    protocol.maximum_confirmation_candidates / protocol.alpha
                )
            )
        ),
    )
    if (
        protocol.required_heldout_datasets != required
        or protocol.maximum_confirmation_candidates > 2
        or not np.isclose(
            protocol.preregistered_holm_rank_one_threshold,
            protocol.alpha / protocol.maximum_confirmation_candidates,
            rtol=1e-15,
            atol=0.0,
        )
        or protocol.sign_flip_minimum_p_value
        > protocol.preregistered_holm_rank_one_threshold
    ):
        raise ValueError("confirmation protocol power metadata is inconsistent")
    if not candidates:
        if (
            observations
            or protocol.frozen_candidate_count != 0
            or protocol.actual_holm_rank_one_threshold is not None
        ):
            raise ValueError(
                "zero-candidate termination differs from the frozen protocol"
            )
        return {
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
    by_candidate = {candidate_id: [] for candidate_id in candidate_ids}
    for observation in observations:
        if observation.candidate_id not in by_candidate:
            raise ValueError("heldout observation references an unfrozen candidate")
        by_candidate[observation.candidate_id].append(observation)
    dataset_rosters = {
        candidate_id: tuple(
            sorted(item.dataset_id for item in candidate_observations)
        )
        for candidate_id, candidate_observations in by_candidate.items()
    }
    expected_roster = dataset_rosters[sorted(candidate_ids)[0]]
    if (
        len(expected_roster)
        < max(
            protocol.minimum_heldout_datasets,
            protocol.required_heldout_datasets,
        )
        or any(roster != expected_roster for roster in dataset_rosters.values())
        or any(len(set(roster)) != len(roster) for roster in dataset_rosters.values())
    ):
        raise ValueError("heldout dataset roster is incomplete, duplicated, or misaligned")
    expected_mode = (
        "exact-enumeration"
        if len(expected_roster) <= 20
        else "fixed-seed-monte-carlo"
    )
    expected_minimum_p = (
        2.0 ** (-len(expected_roster))
        if expected_mode == "exact-enumeration"
        else 1.0 / (protocol.sign_flip_resamples + 1)
    )
    if (
        protocol.sign_flip_mode != expected_mode
        or not np.isclose(
            protocol.sign_flip_minimum_p_value,
            expected_minimum_p,
            rtol=1e-15,
            atol=0.0,
        )
        or len(candidates) != protocol.frozen_candidate_count
        or len(candidates) > protocol.maximum_confirmation_candidates
        or protocol.actual_holm_rank_one_threshold is None
        or not np.isclose(
            protocol.actual_holm_rank_one_threshold,
            protocol.alpha / len(candidates),
            rtol=1e-15,
            atol=0.0,
        )
    ):
        raise ValueError("heldout roster or candidate family differs from freeze")

    source_native: dict[str, set[float]] = {}
    source_no_op: dict[str, set[float]] = {}
    for observation in observations:
        source_native.setdefault(observation.dataset_id, set()).add(
            observation.source_native_advantage_effect
        )
        source_no_op.setdefault(observation.dataset_id, set()).add(
            observation.source_no_op_advantage_effect
        )
    if any(len(item) != 1 for item in source_native.values()) or any(
        len(item) != 1 for item in source_no_op.values()
    ):
        raise ValueError("source advantage differs across frozen candidates")
    source_components = [
        _confirmation_endpoint(
            name="source_native_advantage",
            values=[next(iter(source_native[name])) for name in expected_roster],
            protocol=protocol,
            seed_offset=100_000,
        ),
        _confirmation_endpoint(
            name="source_no_op_advantage",
            values=[next(iter(source_no_op[name])) for name in expected_roster],
            protocol=protocol,
            seed_offset=100_001,
        ),
    ]
    source_prerequisite = {
        "effective_dataset_count": len(expected_roster),
        "dataset_effects": [
            {
                "dataset_id": name,
                "source_native_advantage": next(iter(source_native[name])),
                "source_no_op_advantage": next(iter(source_no_op[name])),
            }
            for name in expected_roster
        ],
        "components": source_components,
        "composite_p_value": max(item["p_value"] for item in source_components),
        "direction_ci_replication_gates_passed": all(
            item["passes_direction_ci_replication"]
            for item in source_components
        ),
        "p_value_gate_deferred_to_candidate_composite": True,
    }

    candidate_results: list[dict[str, Any]] = []
    for candidate_offset, candidate in enumerate(
        sorted(candidates, key=lambda item: item.candidate_id)
    ):
        values = sorted(
            by_candidate[candidate.candidate_id], key=lambda item: item.dataset_id
        )
        vectors = {
            "target_damage": [item.target_effect for item in values],
            "ablation_specificity": [item.paired_effect for item in values],
            "donor_rescue": [item.donor_rescue_effect for item in values],
            "donor_specificity": [item.donor_specificity_effect for item in values],
        }
        endpoints = [
            _confirmation_endpoint(
                name=name,
                values=observed,
                protocol=protocol,
                seed_offset=candidate_offset * 10 + offset,
            )
            for offset, (name, observed) in enumerate(vectors.items())
        ]
        composite_p = max(
            [item["p_value"] for item in endpoints]
            + [source_prerequisite["composite_p_value"]]
        )
        candidate_results.append(
            {
                "candidate_id": candidate.candidate_id,
                "target_features": list(candidate.target_features),
                "control_features": list(candidate.control_features),
                "latent_baseline": selection._json_safe_baseline(
                    candidate.latent_baseline
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
                "endpoints": endpoints,
                "intersection_union_composite_p_value": composite_p,
                "all_direction_ci_replication_gates_passed": bool(
                    all(
                        item["passes_direction_ci_replication"]
                        for item in endpoints
                    )
                    and source_prerequisite[
                        "direction_ci_replication_gates_passed"
                    ]
                ),
            }
        )
    adjusted = adjust_holm(
        item["intersection_union_composite_p_value"] for item in candidate_results
    )
    for result, adjusted_p in zip(candidate_results, adjusted, strict=True):
        result["holm_adjusted_composite_p_value"] = float(adjusted_p)
        result["confirmed"] = bool(
            result["all_direction_ci_replication_gates_passed"]
            and adjusted_p <= protocol.alpha
        )
    confirmed = [item["confirmed"] is True for item in candidate_results]
    return {
        "status": "completed",
        "heldout_evidence_consumed": True,
        "source_advantage_prerequisite": source_prerequisite,
        "candidate_results": candidate_results,
        "family_candidate_count": len(candidate_results),
        "multiplicity_method": "holm",
        "top_k_after_freeze": False,
        "any_candidate_confirmed": any(confirmed),
        "all_frozen_candidates_confirmed": all(confirmed),
    }


def _confirmation_endpoint(
    *,
    name: str,
    values: Sequence[float],
    protocol: ConfirmationProtocol,
    seed_offset: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 1
        or array.size < protocol.minimum_heldout_datasets
        or not np.isfinite(array).all()
    ):
        raise ValueError("confirmation endpoint dataset vector is invalid")
    low, high = paired_bootstrap_ci(
        array,
        confidence=protocol.confidence_level,
        n_resamples=protocol.bootstrap_resamples,
        seed=protocol.random_seed + seed_offset,
    )
    positive_fraction = float(np.mean(array > 0.0))
    p_value = paired_sign_flip_p_value(
        array,
        n_resamples=protocol.sign_flip_resamples,
        seed=protocol.random_seed + 100_000 + seed_offset,
    )
    return {
        "name": name,
        "metric": _ENDPOINT_METRICS[name],
        "direction": "positive",
        "effective_dataset_count": int(array.size),
        "mean_effect": float(np.mean(array)),
        "median_effect": float(np.median(array)),
        "confidence_low": low,
        "confidence_high": high,
        "positive_fraction": positive_fraction,
        "p_value": p_value,
        "passes_direction_ci_replication": bool(
            low > 0.0
            and positive_fraction >= protocol.minimum_positive_fraction
        ),
    }


def _validated_selection_artifacts(
    selection_payload: Mapping[str, Any],
    freeze_payload: Mapping[str, Any],
    summary_payload: Mapping[str, Any],
    *,
    selection_sha256: str,
) -> dict[str, Any]:
    selection_object = selection._exact_object(
        selection_payload,
        label="selection.json",
        fields={
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
    freeze = selection._exact_object(
        freeze_payload,
        label="freeze.json",
        fields={
            "schema_version",
            "evidence_scope",
            "condition",
            "site",
            "random_seed",
            "evaluation_sample_rosters_sha256",
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
            "validation_dataset_fingerprints_sha256",
            "validation_dataset_fingerprint_manifest_sha256",
        },
    )
    summary = selection._exact_object(
        summary_payload,
        label="selection summary",
        fields={
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
            "heldout_dataset_ids",
            "evaluation_sample_rosters_sha256",
            "validation_selection_sha256",
            "common_lineage",
            "checkpoint_study_sha256",
            "validation_dataset_fingerprints_sha256",
            "validation_dataset_fingerprint_manifest_sha256",
        },
    )
    if (
        selection_object["schema_version"] != 1
        or selection_object["analysis"] != "validation-feature-selection"
        or selection_object["evidence_scope"] != "validation-selection"
        or freeze["schema_version"] != 2
        or freeze["evidence_scope"] != "validation-frozen"
        or summary["schema_version"] != 1
        or summary["analysis"] != "validation-feature-selection"
        or summary["evidence_scope"] != "validation-frozen"
    ):
        raise ValueError("selection artifact schema or evidence scope is invalid")
    for name in ("condition", "site", "random_seed"):
        if not (
            selection_object[name] == freeze[name] == summary[name]
        ):
            raise ValueError(f"selection artifact {name} values differ")
    if (
        selection_object["evidence_family"] != list(selection._EVIDENCE_FAMILY)
        or summary["evidence_family"] != list(selection._EVIDENCE_FAMILY)
    ):
        raise ValueError("selection evidence family is not canonical")
    if (
        freeze["validation_selection_sha256"] != selection_sha256
        or summary["validation_selection_sha256"] != selection_sha256
    ):
        raise ValueError("freeze does not bind exact selection bytes")
    common_lineage = _validated_common_lineage(
        selection_object["common_lineage"], freeze=freeze, summary=summary
    )
    validation_dataset_ids = _dataset_ids(
        selection_object["validation_dataset_ids"],
        name="validation_dataset_ids",
    )
    if summary["validation_dataset_ids"] != list(validation_dataset_ids):
        raise ValueError("validation dataset rosters differ across selection artifacts")
    validation_fingerprints = _validated_dataset_fingerprints(
        selection_object["validation_dataset_fingerprints_sha256"],
        expected_dataset_ids=validation_dataset_ids,
        label="validation",
    )
    fingerprint_manifest_sha256 = selection._canonical_sha256(
        validation_fingerprints
    )
    if (
        selection_object["validation_dataset_fingerprint_manifest_sha256"]
        != fingerprint_manifest_sha256
        or freeze["validation_dataset_fingerprints_sha256"]
        != validation_fingerprints
        or summary["validation_dataset_fingerprints_sha256"]
        != validation_fingerprints
        or freeze["validation_dataset_fingerprint_manifest_sha256"]
        != fingerprint_manifest_sha256
        or summary["validation_dataset_fingerprint_manifest_sha256"]
        != fingerprint_manifest_sha256
    ):
        raise ValueError("validation dataset fingerprints differ across artifacts")
    recomputed = _recompute_validation_selection(
        selection_object,
        validation_dataset_ids=validation_dataset_ids,
        common_lineage=common_lineage,
        maximum_symmetric_donor_shift_rms_ratio=selection._finite_float(
            freeze["maximum_symmetric_donor_shift_rms_ratio"],
            name="maximum_symmetric_donor_shift_rms_ratio",
        ),
    )
    if (
        freeze["source_advantage_prerequisite"] != recomputed["source"]
        or summary["source_advantage_prerequisite"] != recomputed["source"]
    ):
        raise ValueError("validation source-advantage artifacts differ")
    rosters = _sha256_mapping(freeze["evaluation_sample_rosters_sha256"])
    heldout = selection._exact_object(
        selection_object["heldout"],
        label="selection heldout",
        fields={
            "dataset_ids",
            "evaluation_sample_rosters_sha256",
            "confirmation_protocol",
        },
    )
    if (
        heldout["dataset_ids"] != list(rosters)
        or heldout["evaluation_sample_rosters_sha256"] != rosters
        or summary["heldout_dataset_ids"] != list(rosters)
        or summary["evaluation_sample_rosters_sha256"] != rosters
    ):
        raise ValueError("selection heldout roster mappings differ")
    protocol = _confirmation_protocol(
        freeze["confirmation_protocol"], random_seed=freeze["random_seed"]
    )
    if (
        heldout["confirmation_protocol"] != _protocol_payload(protocol)
        or summary["confirmation_protocol"] != _protocol_payload(protocol)
    ):
        raise ValueError("confirmation protocol differs across selection artifacts")
    candidates = _frozen_candidates(freeze["selected_interventions"])
    if (
        summary["selected_interventions"]
        != [
            {
                "candidate_id": item.candidate_id,
                "target_features": list(item.target_features),
                "control_features": list(item.control_features),
                "latent_baseline": selection._json_safe_baseline(item.latent_baseline),
            }
            for item in candidates
        ]
        or summary["selection_count"] != len(candidates)
        or [item.candidate_id for item in candidates]
        != recomputed["selected_candidate_ids"]
    ):
        raise ValueError("frozen candidate rosters differ across artifacts")
    if (
        protocol.frozen_candidate_count != len(candidates)
        or protocol.maximum_confirmation_candidates
        != recomputed["parameters"].maximum_selections
        or len(rosters)
        < max(
            protocol.minimum_heldout_datasets,
            protocol.required_heldout_datasets,
        )
    ):
        raise ValueError("frozen confirmation family or heldout power differs")
    expected_mode = (
        "exact-enumeration" if len(rosters) <= 20 else "fixed-seed-monte-carlo"
    )
    expected_minimum_p = (
        2.0 ** (-len(rosters))
        if expected_mode == "exact-enumeration"
        else 1.0 / (protocol.sign_flip_resamples + 1)
    )
    if (
        protocol.sign_flip_mode != expected_mode
        or not np.isclose(
            protocol.sign_flip_minimum_p_value,
            expected_minimum_p,
            rtol=1e-15,
            atol=0.0,
        )
    ):
        raise ValueError("frozen heldout sign-flip mode or resolution differs")
    return {
        "candidates": candidates,
        "protocol": protocol,
        "roster_mapping": rosters,
        "validation_dataset_fingerprints": validation_fingerprints,
    }


def _validated_common_lineage(
    value: Any, *, freeze: Mapping[str, Any], summary: Mapping[str, Any]
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
    lineage = selection._exact_object(
        value, label="selection common_lineage", fields=fields
    )
    direction = selection._exact_object(
        lineage["paired_source_direction"],
        label="paired_source_direction",
        fields={"source_condition", "recipient_condition"},
    )
    if (
        direction["source_condition"] == direction["recipient_condition"]
        or direction["recipient_condition"] != freeze["condition"]
    ):
        raise ValueError("selection paired source direction is invalid")
    binding = selection._exact_object(
        lineage["paired_source_binding"],
        label="paired_source_binding",
        fields={
            "model_sha",
            "checkpoint_sha256",
            "code_attestation_sha256",
            "inference_contract_sha256",
            "representation_source_lineage_sha256",
        },
    )
    for name in binding:
        if name == "model_sha":
            selection._required_git_sha(binding[name], name=name)
        else:
            selection._required_sha256(binding[name], name=name)
    source_checkpoints = lineage["representation_source_lineage"]
    if isinstance(source_checkpoints, Mapping):
        source_checkpoints = source_checkpoints.get(
            "condition_checkpoints_sha256"
        )
    if (
        binding["model_sha"] != lineage["model_code_sha"]
        or binding["inference_contract_sha256"]
        != lineage["inference_contract_sha256"]
        or binding["representation_source_lineage_sha256"]
        != selection._canonical_sha256(
            lineage["representation_source_lineage"]
        )
        or not isinstance(source_checkpoints, Mapping)
        or source_checkpoints.get(direction["source_condition"])
        != binding["checkpoint_sha256"]
        or source_checkpoints.get(direction["recipient_condition"])
        != lineage["checkpoint_sha256"]
    ):
        raise ValueError("selection paired source binding differs from common lineage")
    sha_fields = {
        "training_code_sha": "git",
        "model_code_sha": "git",
        "parent_analysis_code_sha": "git",
        "checkpoint_sha256": "sha256",
        "dataset_manifest_sha256": "sha256",
        "representation_model_sha256": "sha256",
        "representation_parent_manifest_sha256": "sha256",
        "inference_contract_sha256": "sha256",
        "checkpoint_study_sha256": "sha256",
    }
    for name, kind in sha_fields.items():
        if kind == "git":
            selection._required_git_sha(lineage[name], name=name)
        else:
            selection._required_sha256(lineage[name], name=name)
    expected_freeze = {
        "condition": lineage["condition"],
        "site": lineage["site"],
        "random_seed": lineage["random_seed"],
        "model_sha": lineage["model_code_sha"],
        "checkpoint_sha256": lineage["checkpoint_sha256"],
        "representation_model_sha256": lineage["representation_model_sha256"],
        "representation_parent_manifest_sha256": lineage[
            "representation_parent_manifest_sha256"
        ],
        "inference_contract_sha256": lineage["inference_contract_sha256"],
        "paired_source_direction": direction,
        "paired_source_binding": binding,
        "representation_source_lineage_sha256": selection._canonical_sha256(
            lineage["representation_source_lineage"]
        ),
        "checkpoint_study_sha256": lineage["checkpoint_study_sha256"],
    }
    if any(freeze.get(name) != expected for name, expected in expected_freeze.items()):
        raise ValueError("selection common lineage differs from freeze")
    expected_summary_lineage = {
        name: lineage[name]
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
    if (
        summary["common_lineage"] != expected_summary_lineage
        or summary["checkpoint_study_sha256"]
        != lineage["checkpoint_study_sha256"]
        or freeze["checkpoint_study_sha256"]
        != lineage["checkpoint_study_sha256"]
    ):
        raise ValueError("selection summary common lineage differs")
    return lineage


def _dataset_ids(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(require_public_label(item, name=name) for item in value)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _validated_dataset_fingerprints(
    value: Any, *, expected_dataset_ids: Sequence[str], label: str
) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} dataset fingerprints must be an object")
    result: dict[str, dict[str, str]] = {}
    for raw_dataset_id, raw_fingerprints in value.items():
        dataset_id = require_public_label(
            raw_dataset_id, name=f"{label} dataset_id"
        )
        fingerprints = selection._exact_object(
            raw_fingerprints,
            label=f"{label} dataset fingerprints",
            fields={
                "raw_dataset_content_sha256",
                "invariant_prediction_content_sha256",
                "combined_dataset_fingerprint_sha256",
            },
        )
        raw_digest = selection._required_sha256(
            fingerprints["raw_dataset_content_sha256"],
            name="raw_dataset_content_sha256",
        )
        prediction_digest = selection._required_sha256(
            fingerprints["invariant_prediction_content_sha256"],
            name="invariant_prediction_content_sha256",
        )
        combined = selection._required_sha256(
            fingerprints["combined_dataset_fingerprint_sha256"],
            name="combined_dataset_fingerprint_sha256",
        )
        if combined != selection._canonical_sha256(
            {
                "raw_dataset_content_sha256": raw_digest,
                "invariant_prediction_content_sha256": prediction_digest,
            }
        ):
            raise ValueError(f"{label} combined dataset fingerprint differs")
        result[dataset_id] = {
            "raw_dataset_content_sha256": raw_digest,
            "invariant_prediction_content_sha256": prediction_digest,
            "combined_dataset_fingerprint_sha256": combined,
        }
    if tuple(result) != tuple(expected_dataset_ids):
        raise ValueError(f"{label} dataset fingerprint roster differs")
    for name in next(iter(result.values())):
        digests = [fingerprints[name] for fingerprints in result.values()]
        if len(digests) != len(set(digests)):
            raise ValueError(f"{label} dataset fingerprints contain an alias")
    return result


def _assert_disjoint_dataset_fingerprints(
    validation: Mapping[str, Mapping[str, str]],
    heldout: Mapping[str, Mapping[str, str]],
) -> None:
    for name in (
        "raw_dataset_content_sha256",
        "invariant_prediction_content_sha256",
        "combined_dataset_fingerprint_sha256",
    ):
        validation_values = {
            fingerprints[name] for fingerprints in validation.values()
        }
        heldout_values = {
            fingerprints[name] for fingerprints in heldout.values()
        }
        if validation_values & heldout_values:
            raise ValueError("validation and heldout datasets contain an alias")


def _selection_parameters_from_artifact(
    value: Any,
    *,
    candidate_count: int,
    validation_dataset_count: int,
    common_lineage: Mapping[str, Any],
    maximum_symmetric_donor_shift_rms_ratio: float,
) -> selection.SelectionParameters:
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
    statistics = selection._exact_object(
        value, label="selection statistics", fields=fields
    )
    if (
        statistics["multiplicity_method"] != "benjamini-yekutieli"
        or statistics["fdr_control_unit"]
        != "candidate-intersection-union-hypothesis"
        or statistics["candidate_composite_method"]
        != "maximum-of-six-component-p-values"
        or statistics["component_count_per_candidate"] != 6
        or statistics["bootstrap_method"] != "paired-dataset-bootstrap"
        or statistics["p_value_method"] != "one-sided-paired-sign-flip"
        or statistics["ranking_rule"] != "maximum_minimum_mean_evidence"
    ):
        raise ValueError("selection statistics methods are not canonical")
    alpha = selection._finite_float(statistics["fdr_alpha"], name="fdr_alpha")
    confidence = selection._finite_float(
        statistics["confidence_level"], name="confidence_level"
    )
    positive_fraction = selection._finite_float(
        statistics["minimum_positive_fraction"],
        name="minimum_positive_fraction",
    )
    if (
        not 0.0 < alpha <= 1.0
        or not 0.0 < confidence < 1.0
        or not 0.0 <= positive_fraction <= 1.0
    ):
        raise ValueError("selection statistics numerical bounds are invalid")
    minimum = selection._positive_int(
        statistics["minimum_validation_datasets"],
        name="minimum_validation_datasets",
    )
    maximum = selection._positive_int(
        statistics["maximum_selections"], name="maximum_selections"
    )
    bootstrap_resamples = selection._positive_int(
        statistics["bootstrap_resamples"], name="bootstrap_resamples"
    )
    sign_flip_resamples = selection._positive_int(
        statistics["sign_flip_resamples"], name="sign_flip_resamples"
    )
    family_size = candidate_count
    harmonic = selection._harmonic_number(family_size)
    required = max(8, int(np.ceil(np.log2(family_size * harmonic / alpha))))
    expected_mode = (
        "exact-enumeration"
        if validation_dataset_count <= 20
        else "fixed-seed-monte-carlo"
    )
    expected_mc_minimum = (
        None
        if expected_mode == "exact-enumeration"
        else 1.0 / (sign_flip_resamples + 1)
    )
    exact_minimum = 2.0 ** (-validation_dataset_count)
    numerical_equalities = {
        "harmonic_factor": harmonic,
        "rank_one_threshold": alpha / (family_size * harmonic),
        "exact_sign_flip_minimum_p_value": exact_minimum,
    }
    if (
        minimum < 8
        or minimum < required
        or validation_dataset_count < max(minimum, required)
        or maximum > 2
        or statistics["family_size"] != family_size
        or statistics["required_validation_datasets"] != required
        or statistics["effective_validation_dataset_count"]
        != validation_dataset_count
        or statistics["sign_flip_mode"] != expected_mode
        or statistics["monte_carlo_minimum_p_value"] != expected_mc_minimum
        or any(
            not np.isclose(
                selection._finite_float(statistics[name], name=name),
                expected,
                rtol=1e-15,
                atol=0.0,
            )
            for name, expected in numerical_equalities.items()
        )
        or (
            expected_mc_minimum is not None
            and expected_mc_minimum > alpha / (family_size * harmonic)
        )
    ):
        raise ValueError("selection statistics power or family metadata is inconsistent")
    direction = common_lineage["paired_source_direction"]
    return selection.SelectionParameters(
        fdr_alpha=alpha,
        confidence_level=confidence,
        bootstrap_resamples=bootstrap_resamples,
        sign_flip_resamples=sign_flip_resamples,
        random_seed=selection._non_negative_int(
            common_lineage["random_seed"], name="random_seed"
        ),
        minimum_validation_datasets=minimum,
        minimum_positive_fraction=positive_fraction,
        maximum_selections=maximum,
        source_condition=direction["source_condition"],
        recipient_condition=direction["recipient_condition"],
        evidence_family=selection._EVIDENCE_FAMILY,
        maximum_symmetric_donor_shift_rms_ratio=(
            maximum_symmetric_donor_shift_rms_ratio
        ),
    )


def _recompute_validation_selection(
    selection_object: Mapping[str, Any],
    *,
    validation_dataset_ids: Sequence[str],
    common_lineage: Mapping[str, Any],
    maximum_symmetric_donor_shift_rms_ratio: float,
) -> dict[str, Any]:
    raw_results = selection_object["candidate_results"]
    if not isinstance(raw_results, list) or not raw_results:
        raise ValueError("selection candidate_results must be non-empty")
    parameters = _selection_parameters_from_artifact(
        selection_object["statistics"],
        candidate_count=len(raw_results),
        validation_dataset_count=len(validation_dataset_ids),
        common_lineage=common_lineage,
        maximum_symmetric_donor_shift_rms_ratio=(
            maximum_symmetric_donor_shift_rms_ratio
        ),
    )
    source = selection._exact_object(
        selection_object["source_advantage_prerequisite"],
        label="source_advantage_prerequisite",
        fields={
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
        },
    )
    source_effects = source["dataset_effects"]
    if not isinstance(source_effects, list):
        raise ValueError("source advantage dataset_effects must be a list")
    source_by_dataset: dict[str, tuple[float, float]] = {}
    for raw in source_effects:
        effect = selection._exact_object(
            raw,
            label="source advantage dataset effect",
            fields={
                "dataset_id",
                "source_native_advantage",
                "source_no_op_advantage",
            },
        )
        dataset_id = require_public_label(
            effect["dataset_id"], name="validation dataset_id"
        )
        if dataset_id in source_by_dataset:
            raise ValueError("source advantage dataset IDs are duplicated")
        source_by_dataset[dataset_id] = (
            selection._finite_float(
                effect["source_native_advantage"],
                name="source_native_advantage",
            ),
            selection._finite_float(
                effect["source_no_op_advantage"],
                name="source_no_op_advantage",
            ),
        )
    if tuple(source_by_dataset) != tuple(validation_dataset_ids):
        raise ValueError("source advantage validation roster differs")

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
    candidates: list[selection.CandidateSpecification] = []
    observations: list[selection.ValidationObservation] = []
    candidate_ids: list[str] = []
    for raw in raw_results:
        result = selection._exact_object(
            raw, label="selection candidate result", fields=candidate_fields
        )
        candidate_id = require_portable_identifier(
            result["candidate_id"], name="candidate_id"
        )
        candidate_ids.append(candidate_id)
        targets = selection._feature_indices(
            result["target_features"], name="target_features"
        )
        controls = selection._feature_indices(
            result["control_features"], name="control_features"
        )
        if len(targets) != len(controls) or set(targets) & set(controls):
            raise ValueError("selection target/control features are invalid")
        baseline = selection._numerical_baseline(result["latent_baseline"])
        raw_effects = result["dataset_effects"]
        if not isinstance(raw_effects, list):
            raise ValueError("candidate dataset_effects must be a list")
        effect_ids: list[str] = []
        for raw_effect in raw_effects:
            effect = selection._exact_object(
                raw_effect,
                label="candidate dataset effect",
                fields=dataset_fields,
            )
            dataset_id = require_public_label(
                effect["dataset_id"], name="validation dataset_id"
            )
            effect_ids.append(dataset_id)
            values = {
                name: selection._finite_float(effect[name], name=name)
                for name in dataset_fields - {"dataset_id"}
            }
            if not np.isclose(
                values["ablation_specificity"],
                values["target_damage"] - values["matched_control_damage"],
                rtol=1e-15,
                atol=0.0,
            ):
                raise ValueError("published ablation specificity is inconsistent")
            if not 1.0 <= values["donor_shift_balance_ratio"] <= (
                maximum_symmetric_donor_shift_rms_ratio
            ):
                raise ValueError("published donor shift balance is unsafe")
            source_native, source_no_op = source_by_dataset[dataset_id]
            observations.append(
                selection.ValidationObservation(
                    candidate_id=candidate_id,
                    dataset_id=dataset_id,
                    target_effect=values["target_damage"],
                    control_effect=values["matched_control_damage"],
                    donor_rescue_effect=values["donor_rescue"],
                    donor_specificity_effect=values["donor_specificity"],
                    source_native_advantage_effect=source_native,
                    source_no_op_advantage_effect=source_no_op,
                    donor_shift_balance_ratio=values[
                        "donor_shift_balance_ratio"
                    ],
                )
            )
        if effect_ids != list(validation_dataset_ids):
            raise ValueError("candidate validation dataset roster differs")
        if result["effective_dataset_count"] != len(validation_dataset_ids):
            raise ValueError("candidate effective dataset count differs")
        candidates.append(
            selection.CandidateSpecification(
                candidate_id=candidate_id,
                target_features=targets,
                control_features=controls,
                latent_baseline=baseline,
                validation_runs=tuple(
                    selection.ValidationRunSpecification(
                        dataset_id=dataset_id,
                        run_dir=Path("/frozen-validation"),
                        expected_manifest_sha256="0" * 64,
                    )
                    for dataset_id in validation_dataset_ids
                ),
            )
        )
    if candidate_ids != sorted(set(candidate_ids)):
        raise ValueError("selection candidate results must be sorted and unique")
    recomputed_results = selection.aggregate_validation_candidates(
        candidates, observations, parameters=parameters
    )
    recomputed_source = recomputed_results[0].pop(
        "source_advantage_prerequisite"
    )
    for result in recomputed_results[1:]:
        if result.pop("source_advantage_prerequisite") != recomputed_source:
            raise RuntimeError("recomputed source advantage was inconsistent")
    if recomputed_results != raw_results or recomputed_source != source:
        raise ValueError("selection statistics or decisions do not recompute exactly")
    selected_ids = [
        result["candidate_id"]
        for result in recomputed_results
        if result["selected"] is True
    ]
    if selection_object["selected_candidates"] != selected_ids:
        raise ValueError("selected candidate IDs differ from recomputed ranking")
    return {
        "source": recomputed_source,
        "selected_candidate_ids": selected_ids,
        "parameters": parameters,
    }


def _confirmation_protocol(
    value: Any, *, random_seed: int
) -> ConfirmationProtocol:
    protocol = selection._exact_object(
        value,
        label="confirmation protocol",
        fields={
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
    protocol_seed = selection._non_negative_int(
        protocol["random_seed"], name="confirmation random_seed"
    )
    if (
        protocol["sign_flip_mode"]
        not in {"exact-enumeration", "fixed-seed-monte-carlo"}
        or protocol["bootstrap_method"] != "paired-dataset-bootstrap"
        or protocol_seed != random_seed
        or protocol["candidate_test"] != "intersection-union-max-p"
        or protocol["multiplicity_method"] != "holm"
        or protocol["top_k_after_freeze"] is not False
    ):
        raise ValueError("confirmation protocol is not canonical")
    alpha = selection._finite_float(protocol["alpha"], name="confirmation alpha")
    confidence = selection._finite_float(
        protocol["confidence_level"], name="confirmation confidence_level"
    )
    replication = selection._finite_float(
        protocol["minimum_positive_fraction"],
        name="confirmation minimum_positive_fraction",
    )
    minimum = selection._positive_int(
        protocol["minimum_heldout_datasets"], name="minimum_heldout_datasets"
    )
    required = selection._positive_int(
        protocol["required_heldout_datasets"], name="required_heldout_datasets"
    )
    maximum_candidates = selection._positive_int(
        protocol["maximum_confirmation_candidates"],
        name="maximum_confirmation_candidates",
    )
    frozen_count = selection._non_negative_int(
        protocol["frozen_candidate_count"], name="frozen_candidate_count"
    )
    preregistered_threshold = selection._finite_float(
        protocol["preregistered_holm_rank_one_threshold"],
        name="preregistered_holm_rank_one_threshold",
    )
    raw_actual_threshold = protocol["actual_holm_rank_one_threshold"]
    actual_threshold = (
        None
        if raw_actual_threshold is None
        else selection._finite_float(
            raw_actual_threshold, name="actual_holm_rank_one_threshold"
        )
    )
    minimum_p = selection._finite_float(
        protocol["sign_flip_minimum_p_value"],
        name="sign_flip_minimum_p_value",
    )
    if (
        not 0.0 < alpha <= 1.0
        or not 0.0 < confidence < 1.0
        or not 0.0 <= replication <= 1.0
        or minimum < 8
        or required
        != max(8, int(np.ceil(np.log2(maximum_candidates / alpha))))
        or maximum_candidates > 2
        or frozen_count > maximum_candidates
        or not np.isclose(
            preregistered_threshold,
            alpha / maximum_candidates,
            rtol=1e-15,
            atol=0.0,
        )
        or (
            frozen_count == 0
            and actual_threshold is not None
        )
        or (
            frozen_count > 0
            and (
                actual_threshold is None
                or not np.isclose(
                    actual_threshold,
                    alpha / frozen_count,
                    rtol=1e-15,
                    atol=0.0,
                )
            )
        )
        or not 0.0 < minimum_p <= preregistered_threshold
    ):
        raise ValueError("confirmation protocol numerical bounds are invalid")
    return ConfirmationProtocol(
        alpha=alpha,
        confidence_level=confidence,
        sign_flip_resamples=selection._positive_int(
            protocol["sign_flip_resamples"], name="sign_flip_resamples"
        ),
        bootstrap_resamples=selection._positive_int(
            protocol["bootstrap_resamples"], name="bootstrap_resamples"
        ),
        minimum_positive_fraction=replication,
        minimum_heldout_datasets=minimum,
        required_heldout_datasets=required,
        maximum_confirmation_candidates=maximum_candidates,
        frozen_candidate_count=frozen_count,
        preregistered_holm_rank_one_threshold=preregistered_threshold,
        actual_holm_rank_one_threshold=actual_threshold,
        sign_flip_minimum_p_value=minimum_p,
        sign_flip_mode=protocol["sign_flip_mode"],
        random_seed=protocol_seed,
    )


def _protocol_payload(protocol: ConfirmationProtocol) -> dict[str, Any]:
    return {
        "alpha": protocol.alpha,
        "confidence_level": protocol.confidence_level,
        "sign_flip_resamples": protocol.sign_flip_resamples,
        "bootstrap_resamples": protocol.bootstrap_resamples,
        "bootstrap_method": "paired-dataset-bootstrap",
        "random_seed": protocol.random_seed,
        "minimum_positive_fraction": protocol.minimum_positive_fraction,
        "minimum_heldout_datasets": protocol.minimum_heldout_datasets,
        "required_heldout_datasets": protocol.required_heldout_datasets,
        "maximum_confirmation_candidates": (
            protocol.maximum_confirmation_candidates
        ),
        "frozen_candidate_count": protocol.frozen_candidate_count,
        "preregistered_holm_rank_one_threshold": (
            protocol.preregistered_holm_rank_one_threshold
        ),
        "actual_holm_rank_one_threshold": (
            protocol.actual_holm_rank_one_threshold
        ),
        "sign_flip_minimum_p_value": protocol.sign_flip_minimum_p_value,
        "sign_flip_mode": protocol.sign_flip_mode,
        "candidate_test": "intersection-union-max-p",
        "multiplicity_method": "holm",
        "top_k_after_freeze": False,
    }


def _frozen_candidates(value: Any) -> tuple[FrozenCandidate, ...]:
    if not isinstance(value, list):
        raise ValueError("selected_interventions must be a list")
    result: list[FrozenCandidate] = []
    seen: set[str] = set()
    for raw in value:
        item = selection._exact_object(
            raw,
            label="frozen intervention",
            fields={
                "candidate_id",
                "target_features",
                "control_features",
                "latent_baseline",
            },
        )
        candidate_id = require_portable_identifier(
            item["candidate_id"], name="candidate_id"
        )
        if candidate_id in seen:
            raise ValueError("frozen candidate IDs must be unique")
        seen.add(candidate_id)
        targets = selection._feature_indices(
            item["target_features"], name="target_features"
        )
        controls = selection._feature_indices(
            item["control_features"], name="control_features"
        )
        if len(targets) != len(controls) or set(targets) & set(controls):
            raise ValueError("frozen target/control features are invalid")
        result.append(
            FrozenCandidate(
                candidate_id=candidate_id,
                target_features=targets,
                control_features=controls,
                latent_baseline=selection._numerical_baseline(
                    item["latent_baseline"]
                ),
            )
        )
    if [item.candidate_id for item in result] != sorted(seen):
        raise ValueError("frozen candidates must be sorted by candidate_id")
    return tuple(result)


def _sha256_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or len(value) < 8:
        raise ValueError("heldout roster mapping must contain at least 8 datasets")
    result = {
        require_public_label(name, name="heldout dataset_id"): (
            selection._required_sha256(digest, name="heldout roster SHA-256")
        )
        for name, digest in value.items()
    }
    if list(result) != sorted(result):
        raise ValueError("heldout roster mapping must be sorted")
    return result


def _heldout_run_specifications(value: Any) -> tuple[HeldoutRunSpecification, ...]:
    if not isinstance(value, list):
        raise ValueError("heldout_runs must be a list")
    result: list[HeldoutRunSpecification] = []
    seen: set[tuple[str, str]] = set()
    for raw in value:
        item = selection._exact_object(
            raw, label="heldout run", fields=_HELDOUT_RUN_FIELDS
        )
        candidate_id = require_portable_identifier(
            item["candidate_id"], name="candidate_id"
        )
        dataset_id = require_public_label(
            item["dataset_id"], name="heldout dataset_id"
        )
        key = (candidate_id, dataset_id)
        if key in seen:
            raise ValueError("heldout candidate/dataset pairs must be unique")
        seen.add(key)
        result.append(
            HeldoutRunSpecification(
                candidate_id=candidate_id,
                dataset_id=dataset_id,
                run_dir=selection._absolute_directory(
                    item["run_dir"], name="heldout run_dir"
                ),
                expected_manifest_sha256=selection._required_sha256(
                    item["expected_manifest_sha256"],
                    name="expected_manifest_sha256",
                ),
            )
        )
    return tuple(result)


def _prepare_heldout_parent(
    specification: HeldoutRunSpecification,
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
        raise RuntimeError("heldout parent changed during directory verification")
    if not {"summary.json", "predictions.json"}.issubset(
        item.name for item in parent.artifacts
    ):
        raise ValueError("heldout parent lacks summary or predictions")
    token = manifest_file.digest.sha256
    roles = {
        "manifest": f"heldout.manifest.{token}",
        "summary": f"heldout.summary.{token}",
        "predictions": f"heldout.predictions.{token}",
    }
    _register(
        roles["manifest"],
        manifest_file.path,
        manifest_file.digest.sha256,
        additional_paths,
        expected_additional,
    )
    for index, artifact in enumerate(parent.artifacts):
        verified = verify_file(
            specification.run_dir / artifact.name,
            expected_sha256=artifact.sha256,
        )
        role = (
            roles["summary"]
            if artifact.name == "summary.json"
            else roles["predictions"]
            if artifact.name == "predictions.json"
            else f"heldout.artifact.{token}.{index}.{artifact.sha256}"
        )
        _register(
            role,
            verified.path,
            verified.digest.sha256,
            additional_paths,
            expected_additional,
        )
    return parent, roles


def _register(
    role: str,
    path: Path,
    digest: str,
    paths: dict[str, Path],
    expected: dict[str, str],
) -> None:
    if role in paths:
        raise ValueError("confirmation input role collision")
    paths[role] = path
    expected[role] = digest


def _validate_selection_manifest_lineage(
    parent: RunManifest, *, context: Any, freeze: Mapping[str, Any]
) -> None:
    assert_git_commit_is_ancestor(context.inputs.analysis_code, parent.analysis_code_sha)
    expected = {
        "model_family": context.model_family,
        "model_revision": context.model_revision,
        "training_code_sha": context.inputs.training_code.head_sha,
        "model_code_sha": context.inputs.model_code.head_sha,
        "checkpoint": context.inputs.checkpoint.digest,
        "dataset_manifest": context.inputs.dataset_manifest.digest,
        "condition": context.condition,
        "sites": context.sites,
        "seed": freeze["random_seed"],
    }
    mismatches = {
        name: (getattr(parent, name), expected_value)
        for name, expected_value in expected.items()
        if getattr(parent, name) != expected_value
    }
    if mismatches:
        raise ValueError(f"selection parent lineage differs: {mismatches}")
    if (
        context.sites != ("row_interactor",)
        or freeze["model_sha"] != context.inputs.model_code.head_sha
        or freeze["checkpoint_sha256"] != context.inputs.checkpoint.digest.sha256
    ):
        raise ValueError("confirmation model/checkpoint/site differs from freeze")


def _extract_heldout_observation(
    prepared: PreparedHeldoutRun,
    *,
    manifest: RunManifest,
    summary: Mapping[str, Any],
    predictions: Mapping[str, Any],
    candidate: FrozenCandidate,
    context: Any,
    freeze: Mapping[str, Any],
    selection_manifest_sha256: str,
    selection_summary_sha256: str,
    freeze_sha256: str,
    roster_sha256: str,
) -> tuple[selection.ValidationObservation, dict[str, Any]]:
    specification = prepared.specification
    if (
        manifest.command != "model-causal"
        or manifest.evidence_level != "strict"
        or manifest.legacy_reasons
    ):
        raise ValueError("heldout parents must be strict model-causal runs")
    assert_git_commit_is_ancestor(context.inputs.analysis_code, manifest.analysis_code_sha)
    expected_manifest = {
        "model_family": context.model_family,
        "model_revision": context.model_revision,
        "training_code_sha": context.inputs.training_code.head_sha,
        "model_code_sha": context.inputs.model_code.head_sha,
        "checkpoint": context.inputs.checkpoint.digest,
        "dataset_manifest": context.inputs.dataset_manifest.digest,
        "condition": context.condition,
        "sites": context.sites,
        "seed": freeze["random_seed"],
    }
    if any(getattr(manifest, name) != value for name, value in expected_manifest.items()):
        raise ValueError("heldout parent lineage differs from confirmation")
    scope = {
        "analysis": "official-model-causal",
        "dataset_id": specification.dataset_id,
        "fit_split": "train",
        "evaluation_split": "test",
        "roster_split": "held_out",
        "evidence_scope": "confirmatory-held-out",
        "site": context.sites[0],
        "source_evidence_level": "strict",
    }
    if any(summary.get(name) != value for name, value in scope.items()):
        raise ValueError("heldout summary scope differs from frozen run")
    registered = {item.role: item.sha256 for item in manifest.inputs}
    required_inputs = {
        "samples.roster": roster_sha256,
        "intervention.freeze": freeze_sha256,
        "selection.parent_manifest": selection_manifest_sha256,
        "selection.summary": selection_summary_sha256,
    }
    if any(registered.get(role) != digest for role, digest in required_inputs.items()):
        raise ValueError("heldout parent is not bound to frozen selection inputs")
    intervention = summary.get("intervention")
    if intervention != {
        "target_features": list(candidate.target_features),
        "control_features": list(candidate.control_features),
        "latent_baseline": selection._json_safe_baseline(candidate.latent_baseline),
        "random_seed": freeze["random_seed"],
    }:
        raise ValueError("heldout intervention differs from frozen candidate")
    if summary.get("matched_control_features") != list(candidate.control_features):
        raise ValueError("heldout matched controls differ from freeze")
    no_op = summary.get("no_op_gates")
    if not isinstance(no_op, Mapping) or no_op.get("passed") is not True:
        raise ValueError("heldout no-op gates did not pass")
    conditions = summary.get("conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("heldout summary lacks conditions")
    target_summary = selection._condition_metric(conditions, "target_baseline_edit")
    control_summary = selection._condition_metric(conditions, "matched_random_edit")
    surrogate = selection.CandidateSpecification(
        candidate_id=candidate.candidate_id,
        target_features=candidate.target_features,
        control_features=candidate.control_features,
        latent_baseline=candidate.latent_baseline,
        validation_runs=(
            selection.ValidationRunSpecification(
                dataset_id=specification.dataset_id,
                run_dir=specification.run_dir,
                expected_manifest_sha256=specification.expected_manifest_sha256,
            ),
        ),
    )
    recomputed, prediction_alignment = selection._recomputed_prediction_effects(
        predictions,
        dataset_id=specification.dataset_id,
        candidate=surrogate,
        evaluation_split="test",
        roster_split="held_out",
        evidence_scope="confirmatory-held-out",
    )
    selection._require_close(
        target_summary, recomputed["target_damage"], name="target damage"
    )
    selection._require_close(
        control_summary,
        recomputed["matched_control_damage"],
        name="matched control damage",
    )
    bindings = summary.get("input_bindings")
    source_lineage = summary.get("representation_source_lineage")
    if not isinstance(bindings, Mapping) or not isinstance(source_lineage, Mapping):
        raise ValueError("heldout summary lacks input lineage")
    if (
        bindings.get("freeze_artifact_sha256") != freeze_sha256
        or bindings.get("selection_parent_manifest_sha256")
        != selection_manifest_sha256
        or bindings.get("checkpoint_sha256") != freeze["checkpoint_sha256"]
        or bindings.get("representation_model_sha256")
        != freeze["representation_model_sha256"]
        or bindings.get("parent_manifest_sha256")
        != freeze["representation_parent_manifest_sha256"]
        or bindings.get("inference_contract_sha256")
        != freeze["inference_contract_sha256"]
        or bindings.get("checkpoint_study_sha256")
        != freeze["checkpoint_study_sha256"]
        or selection._canonical_sha256(source_lineage)
        != freeze["representation_source_lineage_sha256"]
    ):
        raise ValueError("heldout summary lineage differs from freeze")
    checkpoint_study_sha256 = selection._validated_checkpoint_study(
        summary.get("checkpoint_study"),
        input_bindings=bindings,
        source_condition=freeze["paired_source_direction"]["source_condition"],
        recipient_condition=freeze["paired_source_direction"][
            "recipient_condition"
        ],
        source_checkpoint_sha256=freeze["paired_source_binding"][
            "checkpoint_sha256"
        ],
        recipient_checkpoint_sha256=freeze["checkpoint_sha256"],
    )
    if checkpoint_study_sha256 != freeze["checkpoint_study_sha256"]:
        raise ValueError("heldout checkpoint study differs from freeze")
    parameters = selection.SelectionParameters(
        fdr_alpha=0.05,
        confidence_level=0.95,
        bootstrap_resamples=1,
        sign_flip_resamples=1,
        random_seed=freeze["random_seed"],
        minimum_validation_datasets=8,
        minimum_positive_fraction=0.0,
        maximum_selections=1,
        source_condition=freeze["paired_source_direction"]["source_condition"],
        recipient_condition=freeze["paired_source_direction"]["recipient_condition"],
        evidence_family=selection._EVIDENCE_FAMILY,
        maximum_symmetric_donor_shift_rms_ratio=freeze[
            "maximum_symmetric_donor_shift_rms_ratio"
        ],
    )
    (
        rescue,
        donor_specificity,
        source_native_advantage,
        source_no_op_advantage,
        donor_shift_ratio,
        source_binding,
    ) = selection._paired_reverse_evidence(
        summary,
        parameters=parameters,
        registered_inputs=registered,
        recipient_model_sha=freeze["model_sha"],
        recipient_inference_contract_sha256=freeze["inference_contract_sha256"],
        representation_source_lineage=source_lineage,
    )
    if source_binding != freeze["paired_source_binding"]:
        raise ValueError("heldout paired source differs from freeze")
    for name, actual in (
        ("donor_rescue", rescue),
        ("donor_specificity", donor_specificity),
        ("source_native_advantage", source_native_advantage),
        ("source_no_op_advantage", source_no_op_advantage),
    ):
        selection._require_close(actual, recomputed[name], name=name)
    raw_input_bindings = selection._validated_raw_input_bindings(
        bindings.get("raw_dataset_input_sha256"),
        registered_inputs=registered,
    )
    alignment = {
        "sample_roster_sha256": roster_sha256,
        "preprocessing_roster_sha256": selection._required_sha256(
            bindings.get("preprocessing_roster_sha256"),
            name="preprocessing_roster_sha256",
        ),
        "raw_dataset_input_sha256": raw_input_bindings,
        "raw_dataset_content_sha256": selection._canonical_sha256(
            raw_input_bindings
        ),
        **prediction_alignment,
    }
    return (
        selection.ValidationObservation(
            candidate_id=candidate.candidate_id,
            dataset_id=specification.dataset_id,
            target_effect=recomputed["target_damage"],
            control_effect=recomputed["matched_control_damage"],
            donor_rescue_effect=recomputed["donor_rescue"],
            donor_specificity_effect=recomputed["donor_specificity"],
            source_native_advantage_effect=recomputed[
                "source_native_advantage"
            ],
            source_no_op_advantage_effect=recomputed["source_no_op_advantage"],
            donor_shift_balance_ratio=donor_shift_ratio,
        ),
        alignment,
    )


__all__ = [
    "ConfirmationProtocol",
    "FrozenCandidate",
    "confirm_frozen_candidates",
    "run",
]
