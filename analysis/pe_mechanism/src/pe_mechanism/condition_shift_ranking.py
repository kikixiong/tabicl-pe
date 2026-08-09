"""Discovery-only ranking of aligned RoPE/No-PE representation shifts.

The workflow deliberately consumes activation evidence rather than model-causal
outcomes.  It selects one shared target/control feature set for later paired
interventions, but does not itself measure a causal or downstream effect.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .causal import (
    activation_frequency,
    matched_random_control_features,
    raw_space_decoder_feature_norms,
)
from .identifiers import require_public_label
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
from .representation import (
    _assert_collect_lineage,
    _condition_input_specs,
    _load_verified_condition_inputs,
    load_verified_representation_checkpoint,
    representation_qualification,
)


_CONDITIONS = ("none", "rope")
_SITE = "row_interactor"
_SCORE_DESCRIPTION = (
    "median across feature-ranking datasets of RMS RoPE-minus-No-PE latent "
    "difference times decoder-direction norm divided by raw activation RMS"
)
_SCORE_ID = (
    "median_dataset_rms_latent_rope_minus_none_times_decoder_norm_"
    "divided_by_pooled_raw_activation_rms"
)
_MATCHED_CONTROL_TEMPLATE = (
    "activation-frequency and log-decoder-norm nearest-neighbour pool of size "
    "{candidate_pool_size}, sampled once with seed {random_seed} without replacement"
)
_TOP_LEVEL_FIELDS = {
    "provenance",
    "representation_run_dir",
    "expected_parent_manifest_sha256",
    "activation_sources_by_condition",
    "split_protocol",
    "ranking",
}
_RANKING_FIELDS = {
    "dataset_ids",
    "minimum_nonzero_datasets",
    "target_count",
    "random_candidate_pool_size",
    "random_seed",
}
_SPLIT_PROTOCOL_REFERENCE_FIELDS = {"path", "expected_sha256"}
_SOURCE_LINEAGE_FIELDS = {
    "schema_version",
    "source_kind",
    "reference_condition",
    "condition_checkpoints_sha256",
    "collect_parent_manifests_sha256",
    "alignment_sha256",
    "inference_contract_sha256",
    "evaluation_split",
    "max_classes",
}


@dataclass(frozen=True)
class RankingParameters:
    """Frozen numerical choices for one condition-shift ranking."""

    dataset_ids: tuple[str, ...]
    minimum_nonzero_datasets: int
    target_count: int
    candidate_pool_size: int
    random_seed: int


@dataclass(frozen=True)
class ConditionShiftRanking:
    """Path-free deterministic output of the numerical ranking kernel."""

    median_scores: np.ndarray
    nonzero_dataset_counts: np.ndarray
    activation_frequencies: np.ndarray
    decoder_norms: np.ndarray
    raw_activation_rms_by_dataset: Mapping[str, float]
    dataset_score_sha256: Mapping[str, str]
    target_features: tuple[int, ...]
    control_features: tuple[int, ...]


def rank_condition_shift(
    activations_by_condition: Mapping[str, Mapping[str, Tensor]],
    *,
    model: nn.Module,
    normalizer: nn.Module,
    parameters: RankingParameters,
) -> ConditionShiftRanking:
    """Rank aligned condition shifts with one equal-weight score per dataset."""

    if set(activations_by_condition) != set(_CONDITIONS):
        raise ValueError("condition-shift ranking requires exactly none and rope")
    expected_roster = parameters.dataset_ids
    for condition in _CONDITIONS:
        condition_roster = activations_by_condition[condition]
        if len(condition_roster) != len(expected_roster) or set(
            condition_roster
        ) != set(expected_roster):
            raise ValueError(
                f"{condition} activation roster differs from ranking.dataset_ids"
            )
    _validate_parameters(parameters, latent_dim=None)
    if not hasattr(model, "encode"):
        raise TypeError("representation model must expose encode")
    if not hasattr(normalizer, "normalize"):
        raise TypeError("representation normalizer must expose normalize")

    model = model.to("cpu").eval()
    normalizer = normalizer.to("cpu").eval()
    decoder_norms = raw_space_decoder_feature_norms(
        model, normalizer
    ).to(dtype=torch.float64)
    latent_dim = int(decoder_norms.numel())
    _validate_parameters(parameters, latent_dim=latent_dim)

    dataset_scores: list[np.ndarray] = []
    dataset_score_sha256: dict[str, str] = {}
    raw_rms_by_dataset: dict[str, float] = {}
    weighted_frequency_sum = torch.zeros(latent_dim, dtype=torch.float64)
    frequency_row_count = 0
    with torch.no_grad():
        for dataset_id in expected_roster:
            none_values = _activation_matrix(
                activations_by_condition["none"][dataset_id],
                name=f"none[{dataset_id}]",
            )
            rope_values = _activation_matrix(
                activations_by_condition["rope"][dataset_id],
                name=f"rope[{dataset_id}]",
            )
            if none_values.shape != rope_values.shape:
                raise ValueError(
                    f"aligned activation shape differs for dataset {dataset_id!r}"
                )
            normalized_none = normalizer.normalize(none_values)
            normalized_rope = normalizer.normalize(rope_values)
            none_latents = _latent_matrix(
                model.encode(normalized_none),
                rows=none_values.shape[0],
                latent_dim=latent_dim,
                name=f"none[{dataset_id}]",
            )
            rope_latents = _latent_matrix(
                model.encode(normalized_rope),
                rows=rope_values.shape[0],
                latent_dim=latent_dim,
                name=f"rope[{dataset_id}]",
            )
            latent_difference_rms = (
                (rope_latents.to(torch.float64) - none_latents.to(torch.float64))
                .square()
                .mean(dim=0)
                .sqrt()
            )
            raw_rms = torch.cat(
                (none_values.to(torch.float64), rope_values.to(torch.float64)),
                dim=0,
            ).square().mean().sqrt()
            if not torch.isfinite(raw_rms) or float(raw_rms) <= 0.0:
                raise ValueError(
                    f"raw activation RMS must be finite and positive for {dataset_id!r}"
                )
            score = latent_difference_rms * decoder_norms / raw_rms
            if not torch.isfinite(score).all() or (score < 0).any():
                raise ValueError("condition-shift scores must be finite and non-negative")
            score_array = score.cpu().numpy().astype(np.float64, copy=False)
            dataset_scores.append(score_array)
            dataset_score_sha256[dataset_id] = _numeric_array_sha256(score_array)
            raw_rms_by_dataset[dataset_id] = float(raw_rms)

            for latents in (none_latents, rope_latents):
                row_count = int(latents.shape[0])
                weighted_frequency_sum += (
                    activation_frequency(latents).to(torch.float64) * row_count
                )
                frequency_row_count += row_count

    if frequency_row_count <= 0:
        raise RuntimeError("ranking observed no latent rows")
    score_matrix = np.stack(dataset_scores, axis=0)
    median_scores = np.median(score_matrix, axis=0)
    nonzero_counts = np.count_nonzero(score_matrix > 0.0, axis=0).astype(np.int64)
    activation_frequencies = (
        weighted_frequency_sum / frequency_row_count
    ).cpu().numpy()
    decoder_norm_array = decoder_norms.cpu().numpy()
    if not all(
        np.isfinite(values).all()
        for values in (median_scores, activation_frequencies, decoder_norm_array)
    ):
        raise RuntimeError("ranking aggregates contain non-finite values")

    eligible = [
        index
        for index in range(latent_dim)
        if int(nonzero_counts[index]) >= parameters.minimum_nonzero_datasets
        and float(median_scores[index]) > 0.0
    ]
    eligible.sort(key=lambda index: (-float(median_scores[index]), index))
    if len(eligible) < parameters.target_count:
        raise ValueError(
            "too few features satisfy minimum_nonzero_datasets for target_count"
        )
    targets = tuple(eligible[: parameters.target_count])
    controls = tuple(
        matched_random_control_features(
            targets,
            activation_frequencies,
            decoder_norm_array,
            seed=parameters.random_seed,
            candidate_pool_size=parameters.candidate_pool_size,
        )
    )
    if len(controls) != len(targets) or set(controls) & set(targets):
        raise RuntimeError("matched controls must be unique and disjoint from targets")
    return ConditionShiftRanking(
        median_scores=np.asarray(median_scores, dtype=np.float64),
        nonzero_dataset_counts=nonzero_counts,
        activation_frequencies=np.asarray(activation_frequencies, dtype=np.float64),
        decoder_norms=np.asarray(decoder_norm_array, dtype=np.float64),
        raw_activation_rms_by_dataset=raw_rms_by_dataset,
        dataset_score_sha256=dataset_score_sha256,
        target_features=targets,
        control_features=controls,
    )


def run(args: Any) -> int:
    """Verify discovery sources, rank their condition shifts, and publish atomically."""

    configuration = load_verified_json_config(Path(args.config))
    config = _exact_object(
        configuration.data,
        label="rank-condition-shift config",
        fields=_TOP_LEVEL_FIELDS,
    )
    parameters = _ranking_parameters(config["ranking"])
    split_reference = _exact_object(
        config["split_protocol"],
        label="split_protocol",
        fields=_SPLIT_PROTOCOL_REFERENCE_FIELDS,
    )
    split_protocol_file = verify_file(
        _absolute_file(split_reference["path"], name="split_protocol.path"),
        expected_sha256=_sha256(
            split_reference["expected_sha256"],
            name="split_protocol.expected_sha256",
        ),
    )
    split_protocol = _load_split_protocol(
        split_protocol_file.read_bytes(), parameters=parameters
    )

    parent_dir = _absolute_directory(
        config["representation_run_dir"], name="representation_run_dir"
    )
    parent = verify_run_directory(parent_dir)
    parent_manifest_file = verify_file(
        parent_dir / "manifest.json",
        expected_sha256=_sha256(
            config["expected_parent_manifest_sha256"],
            name="expected_parent_manifest_sha256",
        ),
    )
    exact_parent = load_verified_run_manifest(parent_manifest_file)
    if not isinstance(exact_parent, RunManifest) or exact_parent != parent:
        raise RuntimeError("representation parent changed during verification")
    _validate_parent_manifest(parent)
    model_artifact = next(
        artifact for artifact in parent.artifacts if artifact.name == "model.pt"
    )
    model_file = verify_file(
        parent_dir / "model.pt", expected_sha256=model_artifact.sha256
    )
    if model_file.digest.size_bytes != model_artifact.size_bytes:
        raise ValueError("representation model size differs from its parent manifest")

    raw_sources = config["activation_sources_by_condition"]
    if not isinstance(raw_sources, Mapping) or set(raw_sources) != set(_CONDITIONS):
        raise ValueError(
            "activation_sources_by_condition requires exactly none and rope"
        )
    for condition in _CONDITIONS:
        values = raw_sources[condition]
        if (
            not isinstance(values, Mapping)
            or not all(isinstance(dataset_id, str) for dataset_id in values)
            or tuple(sorted(values)) != parameters.dataset_ids
        ):
            raise ValueError(
                f"{condition} sources must equal the declared ranking dataset roster"
            )

    additional_paths: dict[str, Path] = {
        "ranking.split_protocol": split_protocol_file.path,
        "representation.parent_manifest": parent_manifest_file.path,
        "representation.model": model_file.path,
    }
    expected_additional = {
        "ranking.split_protocol": split_protocol_file.digest.sha256,
        "representation.parent_manifest": parent_manifest_file.digest.sha256,
        "representation.model": model_file.digest.sha256,
    }
    collect_runs: dict[Path, Any] = {}
    source_specs = _condition_input_specs(
        raw_sources,
        split="training",
        paths=additional_paths,
        expected_hashes=expected_additional,
        collect_runs=collect_runs,
        verify_complete_runs=False,
    )
    context = verify_configured_run_inputs(
        configuration,
        command="rank-condition-shift",
        seed=parameters.random_seed,
        additional_input_paths=additional_paths,
        expected_additional_sha256=expected_additional,
    )
    if context.inputs.evidence_level != "strict":
        raise RuntimeError("rank-condition-shift requires strict Git evidence")
    _validate_parent_lineage(parent, context=context, seed=parameters.random_seed)
    assert_dataset_roster(
        context.inputs.dataset_manifest,
        parameters.dataset_ids,
        required_split="discovery",
    )
    assert_dataset_roster(
        context.inputs.dataset_manifest,
        split_protocol["causal_test_datasets"],
        required_split="discovery",
    )

    model, normalizer, metadata = load_verified_representation_checkpoint(
        context.additional_file("representation.model"), map_location="cpu"
    )
    source_lineage = _validated_parent_source_lineage(
        metadata, parent=parent, context=context, dataset_ids=parameters.dataset_ids
    )
    references = tuple(
        reference
        for condition in _CONDITIONS
        for reference in source_specs[condition].values()
    )
    observed_lineage = _assert_collect_lineage(
        collect_runs,
        references,
        context,
        reference_condition=source_lineage["reference_condition"],
    )
    _assert_sources_are_parent_bound(
        context=context,
        parent=parent,
        source_specs=source_specs,
        parent_lineage=source_lineage,
        observed_lineage=observed_lineage,
        dataset_ids=parameters.dataset_ids,
    )
    qualification = representation_qualification(metadata)
    if metadata.get("qualification") != qualification:
        raise ValueError(
            "stored representation qualification differs from recomputed metadata"
        )
    if qualification["activation_fidelity_passed"] is not True:
        raise ValueError("representation parent did not pass activation fidelity")
    if set(qualification["validation_by_condition"]) != set(_CONDITIONS):
        raise ValueError("representation fidelity condition roster must be none and rope")

    activations = _load_verified_condition_inputs(source_specs, context)
    ranking = rank_condition_shift(
        activations,
        model=model,
        normalizer=normalizer,
        parameters=parameters,
    )
    activation_sha256 = {
        condition: {
            dataset_id: context.additional_file(
                source_specs[condition][dataset_id].activation_role
            ).digest.sha256
            for dataset_id in parameters.dataset_ids
        }
        for condition in _CONDITIONS
    }
    selection = _selection_payload(
        parameters=parameters,
        ranking=ranking,
        parent=parent,
        parent_manifest_sha256=parent_manifest_file.digest.sha256,
        model_sha256=model_file.digest.sha256,
        split_protocol=split_protocol,
        split_protocol_sha256=split_protocol_file.digest.sha256,
        activation_sha256=activation_sha256,
        source_lineage=source_lineage,
        observed_lineage=observed_lineage,
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
        _write_bytes(
            transaction.staging_dir / "selection.json", _json_bytes(selection)
        )
        artifacts = transaction.artifact_digests(("selection.json",))
        manifest = manifest_from_verified_inputs(context.inputs, artifacts=artifacts)
        transaction.commit(manifest, verified_inputs=context.inputs)
    return 0


def _ranking_parameters(value: Any) -> RankingParameters:
    raw = _exact_object(value, label="ranking", fields=_RANKING_FIELDS)
    dataset_ids = _dataset_ids(raw["dataset_ids"], name="ranking.dataset_ids")
    parameters = RankingParameters(
        dataset_ids=dataset_ids,
        minimum_nonzero_datasets=_positive_integer(
            raw["minimum_nonzero_datasets"], name="minimum_nonzero_datasets"
        ),
        target_count=_positive_integer(raw["target_count"], name="target_count"),
        candidate_pool_size=_positive_integer(
            raw["random_candidate_pool_size"], name="random_candidate_pool_size"
        ),
        random_seed=_nonnegative_integer(raw["random_seed"], name="random_seed"),
    )
    _validate_parameters(parameters, latent_dim=None)
    return parameters


def _validate_parameters(
    parameters: RankingParameters, *, latent_dim: int | None
) -> None:
    if not all(isinstance(dataset_id, str) for dataset_id in parameters.dataset_ids):
        raise ValueError("ranking dataset_ids must contain strings")
    if not parameters.dataset_ids or parameters.dataset_ids != tuple(
        sorted(set(parameters.dataset_ids))
    ):
        raise ValueError("ranking dataset_ids must be sorted and unique")
    if not 1 <= parameters.minimum_nonzero_datasets <= len(parameters.dataset_ids):
        raise ValueError(
            "minimum_nonzero_datasets must lie within the ranking roster"
        )
    for name in ("target_count", "candidate_pool_size"):
        value = getattr(parameters, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(parameters.random_seed, bool)
        or not isinstance(parameters.random_seed, int)
        or not 0 <= parameters.random_seed < 2**63
    ):
        raise ValueError("random_seed must lie in [0, 2**63)")
    if latent_dim is not None:
        if latent_dim < 2 * parameters.target_count:
            raise ValueError("latent dimension cannot provide disjoint matched controls")
        smallest_pool = latent_dim - 2 * parameters.target_count + 1
        if parameters.candidate_pool_size > smallest_pool:
            raise ValueError(
                "random_candidate_pool_size would be truncated for a later target"
            )


def _load_split_protocol(raw: bytes, *, parameters: RankingParameters) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("split protocol is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("split protocol must contain one JSON object")
    required = {
        "schema_version",
        "protocol_id",
        "feature_ranking_datasets",
        "causal_test_datasets",
        "feature_protocol",
        "causal_protocol",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"split protocol missing required fields: {missing}")
    if payload["schema_version"] != 1:
        raise ValueError("split protocol schema_version must be one")
    if not isinstance(payload["protocol_id"], str):
        raise ValueError("protocol_id must be a string")
    protocol_id = require_public_label(payload["protocol_id"], name="protocol_id")
    ranking_ids = _dataset_ids(
        payload["feature_ranking_datasets"],
        name="feature_ranking_datasets",
    )
    causal_ids = _dataset_ids(
        payload["causal_test_datasets"], name="causal_test_datasets"
    )
    if set(ranking_ids) & set(causal_ids):
        raise ValueError("feature-ranking and causal-test rosters must be disjoint")
    if ranking_ids != parameters.dataset_ids:
        raise ValueError(
            "config ranking.dataset_ids differs from the frozen split protocol"
        )
    feature = _exact_object(
        payload["feature_protocol"],
        label="feature_protocol",
        fields={
            "target_count",
            "score",
            "minimum_nonzero_ranking_datasets",
            "matched_control",
            "same_features_both_directions",
        },
    )
    expected_feature = {
        "target_count": parameters.target_count,
        "score": _SCORE_DESCRIPTION,
        "minimum_nonzero_ranking_datasets": (
            parameters.minimum_nonzero_datasets
        ),
        "matched_control": _MATCHED_CONTROL_TEMPLATE.format(
            candidate_pool_size=parameters.candidate_pool_size,
            random_seed=parameters.random_seed,
        ),
        "same_features_both_directions": True,
    }
    if feature != expected_feature:
        raise ValueError("ranking parameters differ from the frozen feature protocol")
    causal = _exact_object(
        payload["causal_protocol"],
        label="causal_protocol",
        fields={
            "directions",
            "maximum_symmetric_donor_shift_rms_ratio",
            "checkpoint_scope",
            "formal_claim",
        },
    )
    if causal["directions"] != ["rope_to_none", "none_to_rope"]:
        raise ValueError("causal protocol must freeze both paired directions")
    donor_shift_limit = _finite_number(
        causal["maximum_symmetric_donor_shift_rms_ratio"],
        name="maximum_symmetric_donor_shift_rms_ratio",
    )
    if not 1.0 <= donor_shift_limit <= 1.25:
        raise ValueError("symmetric donor-shift ratio lies outside [1, 1.25]")
    if causal["checkpoint_scope"] != "exploratory_pilot" or causal[
        "formal_claim"
    ] != "forbidden":
        raise ValueError("rank-condition-shift is restricted to exploratory pilot scope")
    return {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "feature_ranking_datasets": list(ranking_ids),
        "causal_test_datasets": list(causal_ids),
        "feature_protocol": expected_feature,
        "directions": list(causal["directions"]),
        "maximum_symmetric_donor_shift_rms_ratio": donor_shift_limit,
        "checkpoint_scope": "exploratory_pilot",
        "formal_claim": "forbidden",
    }


def _validate_parent_manifest(parent: RunManifest) -> None:
    if parent.command != "train-repr":
        raise ValueError("representation parent command must be train-repr")
    if parent.evidence_level != "strict" or parent.legacy_reasons:
        raise ValueError("representation parent must have strict evidence")
    if parent.sites != (_SITE,):
        raise ValueError("rank-condition-shift requires row_interactor representation")
    if sum(item.name == "model.pt" for item in parent.artifacts) != 1:
        raise ValueError("representation parent must declare exactly one model.pt")


def _validate_parent_lineage(parent: RunManifest, *, context: Any, seed: int) -> None:
    assert_git_commit_is_ancestor(context.inputs.analysis_code, parent.analysis_code_sha)
    expected = {
        "model_family": context.model_family,
        "model_revision": context.model_revision,
        "training_code_sha": context.inputs.training_code.head_sha,
        "model_code_sha": context.inputs.model_code.head_sha,
        "dataset_manifest": context.inputs.dataset_manifest.digest,
        "condition": context.condition,
        "sites": context.sites,
        "seed": seed,
        "checkpoint": context.inputs.checkpoint.digest,
    }
    mismatches = {
        key: (getattr(parent, key), value)
        for key, value in expected.items()
        if getattr(parent, key) != value
    }
    if mismatches:
        raise ValueError(f"representation parent lineage differs: {mismatches}")


def _validated_parent_source_lineage(
    metadata: Mapping[str, Any],
    *,
    parent: RunManifest,
    context: Any,
    dataset_ids: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError("representation checkpoint metadata must be an object")
    public_metadata = metadata.get("metadata")
    if not isinstance(public_metadata, Mapping):
        raise ValueError("representation checkpoint lacks public metadata")
    if public_metadata.get("seed") != parent.seed:
        raise ValueError("representation checkpoint seed differs from its manifest")
    lineage = _exact_object(
        public_metadata.get("source_lineage"),
        label="representation source_lineage",
        fields=_SOURCE_LINEAGE_FIELDS,
    )
    if lineage["schema_version"] != 1 or lineage["source_kind"] != (
        "official_tabicl_bounded_activation_index"
    ):
        raise ValueError("ranking requires official coordinate-aligned sources")
    checkpoints = _condition_sha256_mapping(
        lineage["condition_checkpoints_sha256"],
        name="condition_checkpoints_sha256",
    )
    parents = _condition_digest_lists(
        lineage["collect_parent_manifests_sha256"]
    )
    if set(checkpoints) != set(_CONDITIONS) or set(parents) != set(_CONDITIONS):
        raise ValueError("representation source conditions must be exactly none and rope")
    reference_condition = lineage["reference_condition"]
    if reference_condition not in _CONDITIONS:
        raise ValueError("representation reference_condition must be none or rope")
    if checkpoints[reference_condition] != parent.checkpoint.sha256:
        raise ValueError("reference checkpoint differs from train-repr parent")
    if (
        context.condition != reference_condition
        or context.inputs.checkpoint.digest.sha256 != checkpoints[reference_condition]
    ):
        raise ValueError("ranking provenance differs from reference-condition lineage")
    registered_collect_manifests: set[str] = set()
    for item in parent.inputs:
        prefix = "source.collect_manifest."
        if not item.role.startswith(prefix):
            continue
        if item.role.removeprefix(prefix) != item.sha256:
            raise ValueError("parent collect-manifest input role/digest is inconsistent")
        registered_collect_manifests.add(item.sha256)
    claimed_collect_manifests = {
        digest for condition in _CONDITIONS for digest in parents[condition]
    }
    if registered_collect_manifests != claimed_collect_manifests:
        raise ValueError("parent collect lineage differs from hashed manifest inputs")
    inference_contract = _sha256(
        lineage["inference_contract_sha256"], name="inference_contract_sha256"
    )
    if lineage["evaluation_split"] != "val":
        raise ValueError("representation sources must evaluate validation rows")
    max_classes = _positive_integer(lineage["max_classes"], name="max_classes")
    if max_classes > 10:
        raise ValueError("representation source max_classes exceeds ten")
    alignment = _alignment_mapping(lineage["alignment_sha256"], site=_SITE)
    missing = sorted(set(dataset_ids) - set(alignment["training"]))
    if missing:
        raise ValueError(f"ranking datasets absent from parent alignment: {missing}")
    return {
        "schema_version": 1,
        "source_kind": "official_tabicl_bounded_activation_index",
        "reference_condition": reference_condition,
        "condition_checkpoints_sha256": checkpoints,
        "collect_parent_manifests_sha256": {
            condition: list(parents[condition]) for condition in _CONDITIONS
        },
        "alignment_sha256": alignment,
        "inference_contract_sha256": inference_contract,
        "evaluation_split": "val",
        "max_classes": max_classes,
    }


def _assert_sources_are_parent_bound(
    *,
    context: Any,
    parent: RunManifest,
    source_specs: Mapping[str, Mapping[str, Any]],
    parent_lineage: Mapping[str, Any],
    observed_lineage: Mapping[str, Any],
    dataset_ids: Sequence[str],
) -> None:
    parent_inputs = {
        item.role: (item.sha256, item.size_bytes) for item in parent.inputs
    }
    source_roles = {
        reference.activation_role
        for condition in _CONDITIONS
        for reference in source_specs[condition].values()
    }
    source_roles.update(
        role
        for role in (
            item.role for item in context.inputs.additional_inputs
        )
        if role.startswith("source.collect_manifest.")
        or role.startswith("source.collect_index.")
    )
    for role in source_roles:
        verified = context.additional_file(role)
        if parent_inputs.get(role) != (
            verified.digest.sha256,
            verified.digest.size_bytes,
        ):
            raise ValueError("configured activation source was not bound by train-repr")

    if observed_lineage["source_kind"] != parent_lineage["source_kind"]:
        raise ValueError("ranking source kind differs from representation parent")
    for field in (
        "reference_condition",
        "condition_checkpoints_sha256",
        "inference_contract_sha256",
        "evaluation_split",
        "max_classes",
    ):
        if observed_lineage[field] != parent_lineage[field]:
            raise ValueError(f"ranking source {field} differs from representation parent")
    for condition in _CONDITIONS:
        observed = set(
            observed_lineage["collect_parent_manifests_sha256"][condition]
        )
        registered = set(
            parent_lineage["collect_parent_manifests_sha256"][condition]
        )
        if not observed or not observed <= registered:
            raise ValueError("ranking collect parent was not bound by train-repr")
    expected_alignment = {
        dataset_id: parent_lineage["alignment_sha256"]["training"][dataset_id]
        for dataset_id in dataset_ids
    }
    if observed_lineage["alignment_sha256"] != {"training": expected_alignment}:
        raise ValueError("ranking source coordinates differ from train-repr alignment")


def _selection_payload(
    *,
    parameters: RankingParameters,
    ranking: ConditionShiftRanking,
    parent: RunManifest,
    parent_manifest_sha256: str,
    model_sha256: str,
    split_protocol: Mapping[str, Any],
    split_protocol_sha256: str,
    activation_sha256: Mapping[str, Mapping[str, str]],
    source_lineage: Mapping[str, Any],
    observed_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    ranking_protocol = {
        "dataset_ids": list(parameters.dataset_ids),
        "minimum_nonzero_datasets": parameters.minimum_nonzero_datasets,
        "target_count": parameters.target_count,
        "random_candidate_pool_size": parameters.candidate_pool_size,
        "random_seed": parameters.random_seed,
        "activation_frequency_threshold": 1e-8,
        "decoder_norm_space": "raw_activation_after_denormalize",
        "latent_baseline": 0.0,
        "same_features_both_directions": True,
    }
    ordered_features = sorted(
        range(ranking.median_scores.size),
        key=lambda index: (-float(ranking.median_scores[index]), index),
    )
    activation_binding = {
        condition: {
            dataset_id: activation_sha256[condition][dataset_id]
            for dataset_id in parameters.dataset_ids
        }
        for condition in _CONDITIONS
    }
    return {
        "schema_version": 1,
        "analysis": "exploratory-condition-shift-ranking",
        "evidence_scope": "exploratory-pilot",
        "formal_claim": "forbidden",
        "site": _SITE,
        "conditions": ["rope", "none"],
        "directions": list(split_protocol["directions"]),
        "maximum_symmetric_donor_shift_rms_ratio": split_protocol[
            "maximum_symmetric_donor_shift_rms_ratio"
        ],
        "score_definition": _SCORE_ID,
        "decoder_norm_space": "raw_activation_after_denormalize",
        "raw_activation_rms_definition": (
            "root_mean_square_of_pooled_aligned_none_and_rope_raw_values"
        ),
        "representation_parent_manifest_sha256": parent_manifest_sha256,
        "representation_model_sha256": model_sha256,
        "representation_source_lineage_sha256": _canonical_sha256(source_lineage),
        "ranking_source_lineage_sha256": _canonical_sha256(observed_lineage),
        "split_protocol_id": split_protocol["protocol_id"],
        "split_protocol_sha256": split_protocol_sha256,
        "ranking_dataset_roster_sha256": _canonical_sha256(
            list(parameters.dataset_ids)
        ),
        "activation_sha256_by_condition": activation_binding,
        "activation_binding_sha256": _canonical_sha256(activation_binding),
        "inference_contract_sha256": source_lineage[
            "inference_contract_sha256"
        ],
        "condition_checkpoints_sha256": source_lineage[
            "condition_checkpoints_sha256"
        ],
        "ranking_protocol": ranking_protocol,
        "ranking_protocol_sha256": _canonical_sha256(ranking_protocol),
        "dataset_score_sha256": dict(ranking.dataset_score_sha256),
        "raw_activation_rms_by_dataset": dict(
            ranking.raw_activation_rms_by_dataset
        ),
        "median_scores": ranking.median_scores.tolist(),
        "median_scores_sha256": _numeric_array_sha256(ranking.median_scores),
        "nonzero_dataset_counts": ranking.nonzero_dataset_counts.tolist(),
        "nonzero_dataset_counts_sha256": _numeric_array_sha256(
            ranking.nonzero_dataset_counts
        ),
        "activation_frequencies": ranking.activation_frequencies.tolist(),
        "activation_frequencies_sha256": _numeric_array_sha256(
            ranking.activation_frequencies
        ),
        "decoder_norms": ranking.decoder_norms.tolist(),
        "decoder_norms_sha256": _numeric_array_sha256(ranking.decoder_norms),
        "top_features": [
            {
                "feature": index,
                "score": float(ranking.median_scores[index]),
                "nonzero_dataset_count": int(
                    ranking.nonzero_dataset_counts[index]
                ),
            }
            for index in ordered_features[: min(10, len(ordered_features))]
        ],
        "target_features": list(ranking.target_features),
        "control_features": list(ranking.control_features),
        "latent_baseline": 0.0,
        "parent_seed": parent.seed,
    }


def _alignment_mapping(value: Any, *, site: str) -> dict[str, dict[str, dict[str, str]]]:
    if not isinstance(value, Mapping) or set(value) != {"training", "validation"}:
        raise ValueError("parent alignment must contain training and validation")
    result: dict[str, dict[str, dict[str, str]]] = {}
    for split in ("training", "validation"):
        datasets = value[split]
        if not isinstance(datasets, Mapping) or not datasets:
            raise ValueError(f"{split} alignment roster must be non-empty")
        normalized: dict[str, dict[str, str]] = {}
        for raw_dataset_id, raw_sites in datasets.items():
            if not isinstance(raw_dataset_id, str):
                raise ValueError("alignment dataset IDs must be strings")
            dataset_id = require_public_label(raw_dataset_id, name="alignment dataset_id")
            if not isinstance(raw_sites, Mapping) or set(raw_sites) != {site}:
                raise ValueError("parent alignment site roster differs")
            normalized[dataset_id] = {
                site: _sha256(raw_sites[site], name="alignment digest")
            }
        result[split] = dict(sorted(normalized.items()))
    return result


def _condition_sha256_mapping(value: Any, *, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(_CONDITIONS):
        raise ValueError(f"{name} must contain exactly none and rope")
    return {
        condition: _sha256(value[condition], name=f"{name}.{condition}")
        for condition in _CONDITIONS
    }


def _condition_digest_lists(value: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or set(value) != set(_CONDITIONS):
        raise ValueError("collect parent mapping must contain exactly none and rope")
    result: dict[str, tuple[str, ...]] = {}
    for condition in _CONDITIONS:
        values = value[condition]
        if not isinstance(values, list) or not values:
            raise ValueError("each condition requires collect parent manifests")
        digests = tuple(_sha256(item, name="collect parent digest") for item in values)
        if digests != tuple(sorted(set(digests))):
            raise ValueError("collect parent digests must be sorted and unique")
        result[condition] = digests
    return result


def _activation_matrix(value: Any, *, name: str) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} activations must be a non-empty matrix")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} activations must be finite")
    return tensor.contiguous()


def _latent_matrix(
    value: Any, *, rows: int, latent_dim: int, name: str
) -> Tensor:
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.shape != (rows, latent_dim):
        raise ValueError(f"{name} encoded latent shape differs from the representation")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} encoded latents must be finite")
    return tensor.contiguous()


def _dataset_ids(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} entries must be strings")
    result = tuple(require_public_label(item, name=name) for item in value)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _exact_object(value: Any, *, label: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        raise ValueError(f"{label} fields mismatch: missing={missing}, unknown={unknown}")
    return dict(value)


def _absolute_file(value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{name} must be an absolute non-symlink file")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{name} must be a regular file")
    return resolved


def _absolute_directory(value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{name} must be an absolute non-symlink directory")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{name} must be a directory")
    return resolved


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numerical")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _numeric_array_sha256(value: np.ndarray | Sequence[Any]) -> str:
    array = np.asarray(value)
    if array.dtype.kind == "f":
        canonical = np.asarray(array, dtype="<f8", order="C")
    elif array.dtype.kind in {"i", "u"}:
        canonical = np.asarray(array, dtype="<i8", order="C")
    else:
        raise TypeError("numeric digest accepts only integer or floating arrays")
    header = json.dumps(
        {"dtype": canonical.dtype.str, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + canonical.tobytes(order="C")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


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
    "ConditionShiftRanking",
    "RankingParameters",
    "rank_condition_shift",
    "run",
]
