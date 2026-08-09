"""Strict matched-step fixed-weight localization on official TabICL inference.

This workflow is deliberately exploratory: it compares the content-verified
same-step Stable-RoPE and No-PE pilot checkpoints on the TALENT discovery split.
It executes RoPE changes inside the fitted public classifier, records paired raw
probabilities privately, and publishes a path-free digest-bound run manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .collect import (
    DEFAULT_MAX_PRIVATE_BYTES,
    DEFAULT_MIN_FREE_BYTES,
    _check_budget,
    _publish_json,
    _publish_npz,
)
from .engine import RopeCondition
from .identifiers import require_portable_identifier, require_public_label
from .official_collect import (
    DEFAULT_MAX_CLASSES,
    _DatasetSpec,
    _assert_dataset_inputs,
    _dataset_input_bindings,
    _enforce_class_limit,
    _private_study_root,
    _resolved_estimator_options,
    _validate_configuration as _validate_collect_configuration,
)
from .official_tabicl import (
    OfficialInferenceResult,
    OfficialTabICLDriver,
    _expected_forward_schedule,
    fit_official_talent_driver,
    load_raw_talent_splits,
    official_inference_contract_sha256,
)
from .provenance import (
    RunTransaction,
    VerifiedConfiguration,
    VerifiedFile,
    VerifiedRunContext,
    assert_dataset_roster,
    load_verified_json_config,
    manifest_from_verified_inputs,
    verify_configured_run_inputs,
)
from .statistics import summarize_paired, summary_dict


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROOT_FIELDS = {
    "schema_version",
    "private_study_root",
    "datasets",
    "roster_split",
    "evaluation_split",
    "trusted_pickle",
    "seed",
    "max_classes",
    "device",
    "estimator_options",
    "max_private_bytes",
    "min_free_bytes",
    "checkpoint_study",
    "comparison_step",
    "rope_conditions",
    "secondary_checkpoint_path",
    "expected_secondary_checkpoint_sha256",
    "snapshot_manifest_path",
    "expected_snapshot_manifest_sha256",
    "provenance",
}
_COLLECT_PROJECTION_FIELDS = {
    "schema_version",
    "private_study_root",
    "datasets",
    "roster_split",
    "evaluation_split",
    "trusted_pickle",
    "seed",
    "max_classes",
    "device",
    "estimator_options",
    "max_private_bytes",
    "min_free_bytes",
    "provenance",
}
_REQUIRED_ESTIMATOR_OPTIONS = {
    "n_estimators",
    "batch_size",
    "use_amp",
    "use_fa3",
    "random_state",
}
_ROPE_FREQUENCY_STATE_KEY = "row_interactor.tf_row.rope.freqs"


def evaluate_official_tabicl_rope_conditions(
    driver: OfficialTabICLDriver,
    X: Any,
    y: Any,
    conditions: Sequence[RopeCondition | Mapping[str, Any]],
) -> dict[str, OfficialInferenceResult]:
    """Run native plus reversible RoPE policies with identical identity RNG state."""

    resolved = tuple(
        item if isinstance(item, RopeCondition) else RopeCondition.from_mapping(item)
        for item in conditions
    )
    names = [item.name for item in resolved]
    if "native" in names or len(names) != len(set(names)):
        raise ValueError("RoPE condition names must be unique and cannot use 'native'")

    results: dict[str, OfficialInferenceResult] = {}
    with driver.paired_session() as session:
        initial_state = session.snapshot_identity_rng()
        try:
            results["native"] = session.predict_proba(X, y=y)
            for condition in resolved:
                session.restore_identity_rng(initial_state)
                with session.rope_policy(**condition.policy_kwargs()):
                    result = session.predict_proba(X, y=y)
                if condition.is_full_policy and not np.array_equal(
                    result.probabilities, results["native"].probabilities
                ):
                    raise RuntimeError(
                        f"full RoPE condition {condition.name!r} differs from native"
                    )
                if not np.array_equal(result.classes, results["native"].classes):
                    raise RuntimeError("class order changed across RoPE conditions")
                results[condition.name] = result
        finally:
            session.restore_identity_rng(initial_state)
    return results


def _validate_configuration(
    configuration: VerifiedConfiguration,
) -> tuple[dict[str, Any], tuple[_DatasetSpec, ...], tuple[RopeCondition, ...]]:
    config = dict(configuration.data)
    unknown = sorted(set(config) - _ROOT_FIELDS)
    if unknown:
        raise ValueError(f"unknown localization configuration fields: {unknown}")
    required = {
        "schema_version",
        "private_study_root",
        "datasets",
        "roster_split",
        "checkpoint_study",
        "comparison_step",
        "rope_conditions",
        "secondary_checkpoint_path",
        "snapshot_manifest_path",
        "provenance",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing localization configuration fields: {missing}")
    projected = {
        key: value for key, value in config.items() if key in _COLLECT_PROJECTION_FIELDS
    }
    _, specs = _validate_collect_configuration(
        VerifiedConfiguration(data=projected, file=configuration.file)
    )
    provenance = config["provenance"]
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be an object")
    required_assertions = {
        "expected_checkpoint_sha256",
        "expected_dataset_manifest_sha256",
        "expected_training_code_sha",
        "expected_model_code_sha",
        "expected_analysis_code_sha",
    }
    missing_assertions = sorted(required_assertions - set(provenance))
    if missing_assertions:
        raise ValueError(
            "localization provenance requires immutable expected hashes: "
            f"{missing_assertions}"
        )
    if provenance.get("allow_exploratory_legacy", False) is not False:
        raise ValueError("fixed-weight localization forbids legacy Git evidence")
    if config["checkpoint_study"] != "exploratory_pilot":
        raise ValueError("matched pilot localization requires checkpoint_study='exploratory_pilot'")
    if config["roster_split"] != "discovery":
        raise ValueError("exploratory pilot checkpoints may be used only on discovery datasets")
    if config.get("evaluation_split", "val") != "val":
        raise ValueError("the localization pass reserves TALENT test rows and evaluates val")
    step = config["comparison_step"]
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("comparison_step must be a positive integer")
    for field in ("secondary_checkpoint_path", "snapshot_manifest_path"):
        value = config[field]
        if not isinstance(value, str) or not Path(value).expanduser().is_absolute():
            raise ValueError(f"{field} must be an absolute path string")
    for field in (
        "expected_secondary_checkpoint_sha256",
        "expected_snapshot_manifest_sha256",
    ):
        value = config.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"{field} must be a lowercase SHA-256 string")
    estimator_options = config.get("estimator_options")
    if not isinstance(estimator_options, Mapping):
        raise ValueError("estimator_options must explicitly freeze localization inference")
    missing_estimator = sorted(_REQUIRED_ESTIMATOR_OPTIONS - set(estimator_options))
    unknown_estimator = sorted(set(estimator_options) - _REQUIRED_ESTIMATOR_OPTIONS)
    if missing_estimator or unknown_estimator:
        raise ValueError(
            "estimator_options fields mismatch: "
            f"missing={missing_estimator}, unknown={unknown_estimator}"
        )
    for field in ("n_estimators", "batch_size"):
        value = estimator_options[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"estimator_options.{field} must be a positive integer")
    for field in ("use_amp", "use_fa3"):
        if not isinstance(estimator_options[field], bool):
            raise TypeError(f"estimator_options.{field} must be a JSON boolean")
    if estimator_options["random_state"] != config.get("seed", 42):
        raise ValueError("estimator_options.random_state must equal seed")
    raw_conditions = config["rope_conditions"]
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("rope_conditions must be a non-empty list")
    conditions = tuple(
        RopeCondition.from_mapping(item)
        if isinstance(item, Mapping)
        else (_raise_condition_type())
        for item in raw_conditions
    )
    names = [condition.name for condition in conditions]
    if len(names) != len(set(names)) or "native" in names:
        raise ValueError("rope_conditions names must be unique and cannot use 'native'")
    for name in names:
        require_public_label(name, name="RoPE condition")
    if sum(condition.is_full_policy for condition in conditions) != 1:
        raise ValueError("rope_conditions must contain exactly one explicit full no-op")
    full = next(condition for condition in conditions if condition.is_full_policy)
    if full.blocks is not None:
        raise ValueError("the explicit full no-op must cover all RowInteraction blocks")
    if not any(
        condition.blocks is None
        and not condition.rotate_queries
        and not condition.rotate_keys
        for condition in conditions
    ):
        raise ValueError("rope_conditions must contain an all-block RoPE-off condition")
    if not any(
        condition.blocks is not None
        and not condition.rotate_queries
        and not condition.rotate_keys
        for condition in conditions
    ):
        raise ValueError("rope_conditions must contain at least one block-local RoPE-off condition")
    return config, specs, conditions


def _raise_condition_type() -> RopeCondition:
    raise TypeError("each rope_conditions entry must be an object")


def _validate_matched_checkpoint_pair_schema(
    *,
    rope_config: Mapping[str, Any],
    none_config: Mapping[str, Any],
    rope_schema: Mapping[str, tuple[tuple[int, ...], str]],
    none_schema: Mapping[str, tuple[tuple[int, ...], str]],
) -> None:
    """Validate a path-free Stable-RoPE/No-PE checkpoint schema contract."""

    if rope_config.get("row_identity_mode") != "rope":
        raise ValueError("RoPE checkpoint config has an unexpected row_identity_mode")
    if none_config.get("row_identity_mode") != "none":
        raise ValueError("No-PE checkpoint config has an unexpected row_identity_mode")

    rope_common = {
        key: value for key, value in rope_config.items() if key != "row_identity_mode"
    }
    none_common = {
        key: value for key, value in none_config.items() if key != "row_identity_mode"
    }
    if rope_common != none_common:
        raise ValueError("matched checkpoints differ outside row_identity_mode")

    expected_rope_only = {_ROPE_FREQUENCY_STATE_KEY}
    if set(rope_schema) - set(none_schema) != expected_rope_only:
        raise ValueError("RoPE checkpoint has an unexpected state-schema difference")
    if set(none_schema) - set(rope_schema):
        raise ValueError("No-PE checkpoint has an unexpected state-schema difference")
    for key in none_schema:
        if rope_schema[key] != none_schema[key]:
            raise ValueError("matched checkpoint tensor schemas differ")

    embed_dim = rope_common.get("embed_dim")
    row_nhead = rope_common.get("row_nhead")
    if (
        type(embed_dim) is not int
        or type(row_nhead) is not int
        or embed_dim <= 0
        or row_nhead <= 0
        or embed_dim % row_nhead != 0
        or (embed_dim // row_nhead) % 2 != 0
    ):
        raise ValueError("checkpoint config cannot derive the RoPE frequency shape")
    expected_frequency_schema = (
        (embed_dim // row_nhead // 2,),
        "torch.float32",
    )
    if rope_schema[_ROPE_FREQUENCY_STATE_KEY] != expected_frequency_schema:
        raise ValueError("RoPE frequency state does not match the checkpoint config")


def _verify_snapshot_pair(
    context: VerifiedRunContext,
    *,
    config: Mapping[str, Any],
    secondary: VerifiedFile,
    snapshot: VerifiedFile,
) -> dict[str, Any]:
    try:
        payload = json.loads(snapshot.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("snapshot manifest is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("snapshot manifest must contain one JSON object")
    expected_fields = {
        "schema_version",
        "kind",
        "formal_eligible",
        "comparison_step",
        "seed",
        "pilot_source_commit",
        "captured_at_utc",
        "notes",
        "arms",
    }
    if set(payload) != expected_fields or payload.get("schema_version") != 1:
        raise ValueError("snapshot manifest schema or field roster is not exact")
    if payload.get("kind") != "exploratory_same_step_pilot_checkpoint_pair":
        raise ValueError("snapshot manifest is not the registered matched pilot pair")
    if payload.get("formal_eligible") is not False:
        raise ValueError("pilot snapshot must explicitly remain formal_eligible=false")
    if payload.get("comparison_step") != config["comparison_step"]:
        raise ValueError("snapshot comparison step differs from the localization config")
    if payload.get("seed") != config.get("seed", 42):
        raise ValueError("snapshot seed differs from the localization config")
    source_commit = payload.get("pilot_source_commit")
    if source_commit not in {
        context.inputs.training_code.head_sha,
        context.inputs.model_code.head_sha,
    } or context.inputs.training_code.head_sha != context.inputs.model_code.head_sha:
        raise ValueError("snapshot source commit differs from the verified model/training code")
    arms = payload.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {"rope", "none"}:
        raise ValueError("snapshot manifest must contain exactly rope and none arms")
    expected = {
        "rope": context.inputs.checkpoint,
        "none": secondary,
    }
    identities: dict[str, tuple[dict[str, Any], dict[str, tuple[tuple[int, ...], str]]]] = {}
    for mode, verified in expected.items():
        arm = arms[mode]
        if not isinstance(arm, Mapping):
            raise ValueError("snapshot arm metadata must be an object")
        expected_arm_fields = {
            "bytes",
            "curr_step",
            "row_identity_mode",
            "sha256",
            "snapshot_checkpoint",
            "source_checkpoint",
            "source_job_id",
            "state_dict_tensor_count",
        }
        if set(arm) != expected_arm_fields:
            raise ValueError("snapshot arm field roster is not exact")
        if arm.get("row_identity_mode") != mode:
            raise ValueError("snapshot arm identity mode is inconsistent")
        if arm.get("curr_step") != config["comparison_step"]:
            raise ValueError("snapshot arm is not at the configured comparison step")
        if arm.get("sha256") != verified.digest.sha256:
            raise ValueError("snapshot arm SHA-256 differs from the verified checkpoint")
        if arm.get("bytes") != verified.digest.size_bytes:
            raise ValueError("snapshot arm size differs from the verified checkpoint")
        snapshot_name = arm.get("snapshot_checkpoint")
        if (
            not isinstance(snapshot_name, str)
            or Path(snapshot_name).name != snapshot_name
            or snapshot_name != verified.path.name
        ):
            raise ValueError("snapshot arm filename differs from the verified checkpoint")
        identity = _verify_checkpoint_identity(
            verified,
            expected_mode=mode,
            expected_step=int(config["comparison_step"]),
        )
        if arm.get("state_dict_tensor_count") != len(identity[1]):
            raise ValueError("snapshot arm tensor count differs from the checkpoint")
        identities[mode] = identity

    rope_config, rope_schema = identities["rope"]
    none_config, none_schema = identities["none"]
    _validate_matched_checkpoint_pair_schema(
        rope_config=rope_config,
        none_config=none_config,
        rope_schema=rope_schema,
        none_schema=none_schema,
    )
    return rope_config


def _verify_checkpoint_identity(
    checkpoint: VerifiedFile, *, expected_mode: str, expected_step: int
) -> tuple[dict[str, Any], dict[str, tuple[tuple[int, ...], str]]]:
    import torch

    payload = torch.load(checkpoint.path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("TabICL checkpoint must contain a mapping")
    config = payload.get("config")
    if not isinstance(config, Mapping) or config.get("row_identity_mode") != expected_mode:
        raise ValueError("checkpoint row_identity_mode differs from its arm")
    if payload.get("curr_step") != expected_step:
        raise ValueError("checkpoint curr_step differs from comparison_step")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint state_dict is missing or empty")
    schema: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, tensor in state.items():
        if not isinstance(name, str) or not hasattr(tensor, "shape") or not hasattr(
            tensor, "dtype"
        ):
            raise ValueError("checkpoint state_dict must contain named tensors")
        schema[name] = (tuple(int(size) for size in tensor.shape), str(tensor.dtype))
    checkpoint.assert_unchanged()
    return dict(config), schema


def _validate_condition_bounds(
    conditions: Sequence[RopeCondition], *, checkpoint_config: Mapping[str, Any]
) -> int:
    n_blocks = checkpoint_config.get("row_num_blocks")
    n_heads = checkpoint_config.get("row_nhead")
    embed_dim = checkpoint_config.get("embed_dim")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (n_blocks, n_heads, embed_dim)
    ):
        raise ValueError("checkpoint has invalid RowInteraction dimensions")
    assert isinstance(n_blocks, int)
    assert isinstance(n_heads, int)
    assert isinstance(embed_dim, int)
    if embed_dim % n_heads or (embed_dim // n_heads) % 2:
        raise ValueError("checkpoint RowInteraction head dimension is not even")
    pair_count = embed_dim // n_heads // 2
    for condition in conditions:
        if condition.blocks is not None:
            if not condition.blocks or len(set(condition.blocks)) != len(condition.blocks):
                raise ValueError("condition blocks must be non-empty and unique")
            if any(block < 0 or block >= n_blocks for block in condition.blocks):
                raise ValueError("condition block is outside the checkpoint architecture")
        if condition.heads is not None:
            if not condition.heads or len(set(condition.heads)) != len(condition.heads):
                raise ValueError("condition heads must be non-empty and unique")
            if any(head < 0 or head >= n_heads for head in condition.heads):
                raise ValueError("condition head is outside the checkpoint architecture")
        band = condition.frequency_band
        if isinstance(band, tuple):
            start, stop = band
            if start < 0 or stop <= start or stop > pair_count:
                raise ValueError("condition frequency band is outside the RoPE dimension")
    return n_blocks


def _validate_context(
    context: VerifiedRunContext,
    *,
    conditions: Sequence[RopeCondition],
    n_blocks: int,
) -> None:
    if context.inputs.evidence_level != "strict":
        raise RuntimeError("fixed-weight localization requires strict Git evidence")
    if context.model_family != "tabicl-v2":
        raise ValueError("this localization runner supports model_family='tabicl-v2'")
    if context.condition != "matched-step-rope-none":
        raise ValueError("condition must be 'matched-step-rope-none'")
    if context.inputs.training_code.head_sha != context.inputs.model_code.head_sha:
        raise ValueError("pilot training and model code must be the same exact commit")
    if not context.sites or len(set(context.sites)) != len(context.sites):
        raise ValueError("provenance sites must be non-empty and unique")
    selected_blocks: set[int] = set()
    for condition in conditions:
        selected_blocks.update(
            range(n_blocks) if condition.blocks is None else condition.blocks
        )
    required_sites = {
        f"row_interactor.tf_row.blocks.{block}" for block in selected_blocks
    }
    if set(context.sites) != required_sites:
        raise ValueError("provenance sites must exactly cover every RoPE intervention block")
    for site in context.sites:
        require_portable_identifier(site, name="localization site")


def _validate_driver(
    driver: OfficialTabICLDriver,
    *,
    context: VerifiedRunContext,
    checkpoint: VerifiedFile,
    expected_mode: str,
    expected_n_classes: int,
) -> None:
    if driver.model_sha != context.inputs.model_code.head_sha:
        raise RuntimeError("official driver model SHA differs from verified model code")
    if driver.checkpoint_sha != checkpoint.digest.sha256:
        raise RuntimeError("official driver loaded a different checkpoint")
    if driver.fit_context != "talent-train" or driver.source_evidence_level != "strict":
        raise RuntimeError("official driver must be strict and fit TALENT train only")
    estimator = driver.estimator
    if getattr(estimator, "kv_cache", None) is not False:
        raise ValueError("localization requires kv_cache=False")
    if getattr(estimator, "model_kv_cache_", None) is not None:
        raise ValueError("localization refuses fitted classifier caches")
    if getattr(estimator, "support_many_classes", None) is not False:
        raise ValueError("localization requires support_many_classes=False")
    raw = getattr(estimator, "model_", None)
    if raw is None or getattr(raw, "_cache", None) is not None:
        raise ValueError("localization requires an uncached fitted raw model")
    row = getattr(raw, "row_interactor", None)
    raw_mode = getattr(raw, "row_identity_mode", None)
    row_mode = getattr(row, "identity_mode", raw_mode)
    if raw_mode != expected_mode or row_mode != expected_mode:
        raise RuntimeError("loaded checkpoint identity mode differs from its arm")
    if getattr(raw, "training", None) is not False:
        raise ValueError("localization requires model.eval()")
    col_embedder = getattr(raw, "col_embedder", None)
    if getattr(col_embedder, "feature_group", None) != "same":
        raise ValueError("localization requires checkpoint feature_group='same'")
    if int(getattr(estimator, "n_classes_", -1)) != expected_n_classes:
        raise RuntimeError("fitted class count differs from the TALENT train split")


def _ensemble_schedule_sha256(driver: OfficialTabICLDriver) -> str:
    """Return a canonical digest of the fitted official ensemble schedule."""

    schedule = _expected_forward_schedule(driver.estimator)
    payload = [asdict(call) for call in schedule]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype.hasobject:
        raise TypeError("object arrays cannot be content-hashed from process-local pointers")
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + contiguous.tobytes(order="C")).hexdigest()


def _metric_entry(
    result: OfficialInferenceResult,
    *,
    native: OfficialInferenceResult,
    array_key: str,
) -> dict[str, Any]:
    if result.metrics is None or native.metrics is None:
        raise RuntimeError("localization requires labelled classification metrics")
    # Hash the exact float32 bytes published in predictions.npz, not a wider
    # temporary representation that cannot be recovered from the artifact.
    probabilities = np.asarray(result.probabilities, dtype=np.float32)
    native_probabilities = np.asarray(native.probabilities, dtype=np.float32)
    if probabilities.shape != native_probabilities.shape:
        raise RuntimeError("condition probabilities are not paired to native samples")
    return {
        "accuracy": float(result.metrics.accuracy),
        "log_loss": float(result.metrics.log_loss),
        "delta_accuracy_vs_rope_native": float(
            result.metrics.accuracy - native.metrics.accuracy
        ),
        "delta_log_loss_vs_rope_native": float(
            result.metrics.log_loss - native.metrics.log_loss
        ),
        "max_probability_delta_vs_rope_native": float(
            np.max(np.abs(probabilities - native_probabilities))
        ),
        "probability_sha256": _array_sha256(probabilities),
        "prediction_array_key": array_key,
    }


def _label_token(value: Any) -> tuple[str, str]:
    scalar = value.item() if isinstance(value, np.generic) else value
    return type(scalar).__qualname__, repr(scalar)


def _encode_labels(
    labels: np.ndarray, classes: np.ndarray
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Encode arbitrary labels into the exact published probability-column order."""

    class_tokens = [_label_token(value) for value in np.asarray(classes).reshape(-1)]
    if len(class_tokens) < 2 or len(set(class_tokens)) != len(class_tokens):
        raise RuntimeError("official class order must contain unique class labels")
    lookup = {token: index for index, token in enumerate(class_tokens)}
    try:
        encoded = np.asarray(
            [lookup[_label_token(value)] for value in np.asarray(labels).reshape(-1)],
            dtype=np.int64,
        )
    except KeyError as error:
        raise RuntimeError("evaluation labels differ from the official class order") from error
    portable = [
        {"python_type": python_type, "repr": representation}
        for python_type, representation in class_tokens
    ]
    return encoded, portable


def _portable_summary(
    datasets: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    names = sorted(datasets[0]["conditions"])
    if any(sorted(dataset["conditions"]) != names for dataset in datasets):
        raise RuntimeError("condition roster changed between datasets")
    native_accuracy = [dataset["conditions"]["rope_native"]["accuracy"] for dataset in datasets]
    native_loss = [dataset["conditions"]["rope_native"]["log_loss"] for dataset in datasets]
    conditions: dict[str, Any] = {}
    for offset, name in enumerate(names):
        accuracy = [dataset["conditions"][name]["accuracy"] for dataset in datasets]
        loss = [dataset["conditions"][name]["log_loss"] for dataset in datasets]
        conditions[name] = {
            "accuracy_effect": summary_dict(
                summarize_paired(
                    native_accuracy,
                    accuracy,
                    higher_is_better=True,
                    seed=seed + offset,
                )
            ),
            "log_loss_effect": summary_dict(
                summarize_paired(
                    native_loss,
                    loss,
                    higher_is_better=False,
                    seed=seed + 10_000 + offset,
                )
            ),
        }
    return {
        "schema_version": 1,
        "kind": "matched_step_tabicl_fixed_weight_localization_summary",
        "checkpoint_study": "exploratory_pilot",
        "formal_eligible": False,
        "effect_direction": "positive_means_condition_better_than_rope_native",
        "dataset_count": len(datasets),
        "conditions": conditions,
    }


def _condition_definitions(
    conditions: Sequence[RopeCondition],
) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = [
        {"result_condition": "rope_native", "checkpoint_arm": "rope", "policy": None}
    ]
    for condition in conditions:
        definitions.append(
            {
                "result_condition": f"rope_{condition.name}",
                "checkpoint_arm": "rope",
                "policy": {
                    "blocks": None if condition.blocks is None else list(condition.blocks),
                    "rotate_queries": condition.rotate_queries,
                    "rotate_keys": condition.rotate_keys,
                    "phase_strength": condition.phase_strength,
                    "frequency_band": (
                        list(condition.frequency_band)
                        if isinstance(condition.frequency_band, tuple)
                        else condition.frequency_band
                    ),
                    "heads": None if condition.heads is None else list(condition.heads),
                },
            }
        )
    definitions.append(
        {"result_condition": "none_native", "checkpoint_arm": "none", "policy": None}
    )
    return definitions


def _runtime_attestation(*, device: str) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for distribution in (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = "not-installed"
    import torch

    attestation: dict[str, Any] = {
        "python_version": platform.python_version(),
        "packages": packages,
        "requested_device": device,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        attestation["cuda_device_name"] = str(torch.cuda.get_device_name(index))
        attestation["cuda_device_capability"] = list(
            torch.cuda.get_device_capability(index)
        )
    return attestation


def run_localization(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    driver_factory: Callable[..., OfficialTabICLDriver] = fit_official_talent_driver,
) -> Mapping[str, Any]:
    configuration = load_verified_json_config(config_path)
    config, specs, conditions = _validate_configuration(configuration)
    additional_paths, expected_hashes, role_maps = _dataset_input_bindings(specs)
    missing_dataset_hashes = sorted(set(additional_paths) - set(expected_hashes))
    if missing_dataset_hashes:
        raise ValueError(
            "localization requires a precommitted SHA-256 for every TALENT input: "
            f"{missing_dataset_hashes}"
        )
    additional_paths.update(
        {
            "checkpoint.none": Path(config["secondary_checkpoint_path"]),
            "checkpoint_pair_manifest": Path(config["snapshot_manifest_path"]),
        }
    )
    if "expected_secondary_checkpoint_sha256" in config:
        expected_hashes["checkpoint.none"] = config[
            "expected_secondary_checkpoint_sha256"
        ]
    if "expected_snapshot_manifest_sha256" in config:
        expected_hashes["checkpoint_pair_manifest"] = config[
            "expected_snapshot_manifest_sha256"
        ]
    seed = int(config.get("seed", 42))
    context = verify_configured_run_inputs(
        configuration,
        command="localize",
        seed=seed,
        additional_input_paths=additional_paths,
        expected_additional_sha256=expected_hashes,
    )
    secondary = context.additional_file("checkpoint.none")
    snapshot = context.additional_file("checkpoint_pair_manifest")
    checkpoint_config = _verify_snapshot_pair(
        context,
        config=config,
        secondary=secondary,
        snapshot=snapshot,
    )
    n_blocks = _validate_condition_bounds(
        conditions,
        checkpoint_config=checkpoint_config,
    )
    _validate_context(context, conditions=conditions, n_blocks=n_blocks)
    dataset_ids = tuple(spec.dataset_id for spec in specs)
    assert_dataset_roster(
        context.inputs.dataset_manifest,
        dataset_ids,
        required_split="discovery",
    )

    dataset_results: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    estimator_options = _resolved_estimator_options(config)
    evaluation_split = "val"
    checkpoints = {
        "rope": context.inputs.checkpoint,
        "none": secondary,
    }
    for spec in specs:
        dataset = load_raw_talent_splits(
            spec.directory,
            trusted_pickle=bool(config.get("trusted_pickle", False)),
        )
        _assert_dataset_inputs(
            dataset,
            spec=spec,
            context=context,
            role_map=role_maps[spec.dataset_id],
        )
        n_classes = _enforce_class_limit(
            dataset, int(config.get("max_classes", DEFAULT_MAX_CLASSES))
        )
        split = getattr(dataset, evaluation_split)

        drivers: dict[str, OfficialTabICLDriver] = {}
        for mode, checkpoint in checkpoints.items():
            driver = driver_factory(
                dataset,
                checkpoint.path,
                context_split="train",
                device=str(config.get("device", "cpu")),
                model_sha=context.inputs.model_code.head_sha,
                estimator_options=estimator_options,
                expected_source_root=context.inputs.model_code.root,
            )
            _validate_driver(
                driver,
                context=context,
                checkpoint=checkpoint,
                expected_mode=mode,
                expected_n_classes=n_classes,
            )
            drivers[mode] = driver

        schedule_hashes = {
            mode: _ensemble_schedule_sha256(driver)
            for mode, driver in drivers.items()
        }
        if len(set(schedule_hashes.values())) != 1:
            raise RuntimeError(
                "RoPE and No-PE official ensemble schedules are not exactly paired"
            )

        rope = evaluate_official_tabicl_rope_conditions(
            drivers["rope"], split.X, split.y, conditions
        )
        none = evaluate_official_tabicl_rope_conditions(
            drivers["none"], split.X, split.y, ()
        )["native"]
        for mode, checkpoint in checkpoints.items():
            _validate_driver(
                drivers[mode],
                context=context,
                checkpoint=checkpoint,
                expected_mode=mode,
                expected_n_classes=n_classes,
            )
        native = rope["native"]
        if not np.array_equal(none.classes, native.classes):
            raise RuntimeError("RoPE and No-PE class orders differ")
        ordinal = len(dataset_results)
        prefix = f"d{ordinal:04d}"
        labels, class_tokens = _encode_labels(np.asarray(split.y), native.classes)
        arrays[f"{prefix}_labels"] = labels
        conditions_payload: dict[str, Any] = {}
        for name, result in rope.items():
            public_name = "rope_native" if name == "native" else f"rope_{name}"
            require_public_label(public_name, name="localization condition")
            key = f"{prefix}_{public_name}"
            arrays[key] = np.asarray(result.probabilities, dtype=np.float32)
            conditions_payload[public_name] = _metric_entry(
                result, native=native, array_key=key
            )
        none_key = f"{prefix}_none_native"
        arrays[none_key] = np.asarray(none.probabilities, dtype=np.float32)
        conditions_payload["none_native"] = _metric_entry(
            none, native=native, array_key=none_key
        )
        dataset_results.append(
            {
                "dataset_id": require_public_label(dataset.name, name="dataset_id"),
                "evaluation_split": evaluation_split,
                "n_samples": int(labels.shape[0]),
                "n_classes": n_classes,
                "labels_sha256": _array_sha256(labels),
                "label_encoding": "zero_based_index_into_probability_columns",
                "probability_columns": class_tokens,
                "ensemble_schedule_sha256": schedule_hashes["rope"],
                "conditions": conditions_payload,
            }
        )

    condition_definitions = _condition_definitions(conditions)
    runtime_attestation = _runtime_attestation(
        device=str(config.get("device", "cpu"))
    )
    results = {
        "schema_version": 1,
        "kind": "matched_step_tabicl_fixed_weight_localization",
        "checkpoint_study": "exploratory_pilot",
        "formal_eligible": False,
        "comparison_step": int(config["comparison_step"]),
        "seed": seed,
        "assignment_split": "discovery",
        "evaluation_split": evaluation_split,
        "device": str(config.get("device", "cpu")),
        "estimator_options": estimator_options,
        "condition_definitions": condition_definitions,
        "runtime": runtime_attestation,
        "inference_contract_sha256": official_inference_contract_sha256(
            context.inputs.model_code.head_sha,
            estimator_options,
        ),
        "datasets": dataset_results,
    }
    summary = _portable_summary(dataset_results, seed=seed)
    summary["condition_definitions"] = condition_definitions
    summary["runtime"] = runtime_attestation
    private_root = _private_study_root(config["private_study_root"])
    resolved_output = Path(output_dir).expanduser()
    if not resolved_output.is_absolute():
        raise ValueError("output_dir must be absolute")
    resolved_output = resolved_output.resolve(strict=False)
    if not resolved_output.is_relative_to(private_root):
        raise ValueError("localization output must be inside private_study_root")
    projected_bytes = (
        sum(int(array.nbytes) for array in arrays.values())
        + len(json.dumps(results, allow_nan=False))
        + len(json.dumps(summary, allow_nan=False))
        + (8 << 20)
    )
    _check_budget(
        private_root,
        projected_bytes=projected_bytes,
        max_private_bytes=int(
            config.get("max_private_bytes", DEFAULT_MAX_PRIVATE_BYTES)
        ),
        min_free_bytes=int(config.get("min_free_bytes", DEFAULT_MIN_FREE_BYTES)),
    )
    roots = (
        context.inputs.training_code.root,
        context.inputs.model_code.root,
        context.inputs.analysis_code.root,
    )
    with RunTransaction(resolved_output, source_roots=roots) as transaction:
        _publish_npz(transaction.staging_dir / "predictions.npz", **arrays)
        _publish_json(transaction.staging_dir / "results.json", results)
        _publish_json(transaction.staging_dir / "summary.json", summary)
        artifacts = transaction.artifact_digests(
            ("predictions.npz", "results.json", "summary.json")
        )
        manifest = manifest_from_verified_inputs(
            context.inputs,
            artifacts=artifacts,
        )
        transaction.commit(manifest, verified_inputs=context.inputs)
    return summary


def run(args: argparse.Namespace) -> int:
    run_localization(args.config, args.output_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run matched-step official TabICL fixed-weight localization"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "evaluate_official_tabicl_rope_conditions",
    "main",
    "run",
    "run_localization",
]
