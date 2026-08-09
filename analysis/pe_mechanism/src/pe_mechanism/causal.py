"""Distribution-aware latent interventions and paired effect records."""

from __future__ import annotations

import json
import hashlib
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .representation import (
    load_activation_array,
    load_verified_activation_array,
    load_verified_representation_checkpoint,
)
from .identifiers import require_portable_identifier, validate_public_value
from .provenance import (
    RunTransaction,
    assert_dataset_roster,
    load_verified_json_config,
    manifest_from_verified_inputs,
    verify_file,
    verify_configured_run_inputs,
    verify_run_directory,
)


def activation_frequency(latents: Tensor, *, threshold: float = 1e-8) -> Tensor:
    """Fraction of rows in which each latent feature is active."""

    values = _as_latents(latents)
    return (values.abs() > threshold).to(dtype=torch.float32).mean(dim=0)


def decoder_feature_norms(decoder: nn.Linear | Tensor) -> Tensor:
    """Return the output-space norm of every decoder feature direction."""

    weight = decoder.weight if isinstance(decoder, nn.Linear) else torch.as_tensor(decoder)
    if weight.ndim != 2:
        raise ValueError("decoder weight must have shape [output_dim, latent_dim]")
    return weight.detach().to(dtype=torch.float32, device="cpu").norm(dim=0)


def raw_space_decoder_feature_norms(
    model: nn.Module, normalizer: nn.Module
) -> Tensor:
    """Return decoder norms after mapping directions to raw activation units.

    The float64 scaling followed by the canonical float32 norm is intentional.
    Ranking producers and causal consumers must use these exact operations so an
    immutable ranking can be compared bit-for-bit with its bound representation.
    """

    decoder = getattr(model, "decoder", None)
    if isinstance(decoder, nn.Linear):
        normalized_directions = decoder.weight.detach().to(
            dtype=torch.float64, device="cpu"
        )
    else:
        components = getattr(model, "components", None)
        if not isinstance(components, Tensor) or components.ndim != 2:
            raise TypeError(
                "representation model does not expose linear decoder directions"
            )
        normalized_directions = (
            components.detach()
            .to(dtype=torch.float64, device="cpu")
            .transpose(0, 1)
        )
    rms = getattr(normalizer, "rms", None)
    if not isinstance(rms, Tensor) or rms.ndim != 1:
        raise TypeError("representation normalizer must expose one-dimensional rms")
    raw_scale = rms.detach().to(dtype=torch.float64, device="cpu")
    if normalized_directions.shape[0] != raw_scale.numel():
        raise ValueError("decoder output dimension differs from normalizer rms")
    raw_directions = normalized_directions * raw_scale[:, None]
    return decoder_feature_norms(raw_directions).to(dtype=torch.float64)


def matched_random_control_features(
    target_features: Sequence[int],
    frequencies: Tensor | np.ndarray,
    decoder_norms: Tensor | np.ndarray,
    *,
    seed: int = 42,
    candidate_pool_size: int = 8,
    excluded_features: Sequence[int] = (),
) -> list[int]:
    """Sample one frequency-and-norm matched control for each target feature.

    Matching is done in standardized activation-frequency and log decoder-norm
    space.  Sampling from a small nearest-neighbour pool keeps the control random
    without permitting grossly mismatched features.
    """

    frequency_values = torch.as_tensor(frequencies, dtype=torch.float64).flatten()
    norm_values = torch.as_tensor(decoder_norms, dtype=torch.float64).flatten()
    if frequency_values.shape != norm_values.shape or frequency_values.numel() == 0:
        raise ValueError("frequencies and decoder_norms must be equal non-empty vectors")
    if not torch.isfinite(frequency_values).all() or not torch.isfinite(norm_values).all():
        raise ValueError("matching statistics must be finite")
    if (frequency_values < 0).any() or (frequency_values > 1).any() or (norm_values < 0).any():
        raise ValueError("invalid activation frequencies or decoder norms")
    if candidate_pool_size <= 0:
        raise ValueError("candidate_pool_size must be positive")

    latent_dim = int(frequency_values.numel())
    targets = _validated_features(target_features, latent_dim)
    unavailable = set(targets) | {int(value) for value in excluded_features}
    if latent_dim - len(unavailable) < len(targets):
        raise ValueError("not enough non-target features for a matched random control")

    frequency_scale = frequency_values.std(unbiased=False).clamp_min(1e-8)
    log_norms = torch.log(norm_values.clamp_min(1e-12))
    norm_scale = log_norms.std(unbiased=False).clamp_min(1e-8)
    rng = np.random.default_rng(int(seed))
    controls: list[int] = []
    all_indices = range(latent_dim)
    for target in targets:
        candidates = [index for index in all_indices if index not in unavailable]
        distances = []
        for candidate in candidates:
            frequency_distance = (frequency_values[candidate] - frequency_values[target]) / frequency_scale
            norm_distance = (log_norms[candidate] - log_norms[target]) / norm_scale
            distance = float(torch.sqrt(frequency_distance.square() + norm_distance.square()))
            distances.append((distance, candidate))
        distances.sort(key=lambda item: (item[0], item[1]))
        pool = [candidate for _, candidate in distances[: min(candidate_pool_size, len(distances))]]
        selected = int(rng.choice(pool))
        controls.append(selected)
        unavailable.add(selected)
    return controls


# A shorter alias is useful for programmatic callers.
select_matched_random_features = matched_random_control_features


def intervene_latents(
    latents: Tensor,
    feature_indices: Sequence[int],
    *,
    mode: str,
    baseline: Tensor | np.ndarray | float | None = None,
    paired_latents: Tensor | np.ndarray | None = None,
) -> Tensor:
    """Return a copy of ``latents`` with the requested feature intervention."""

    values = _as_latents(latents)
    features = _validated_features(feature_indices, values.shape[1])
    normalized_mode = mode.lower().replace("-", "_")
    result = values.clone()
    if normalized_mode in {"noop", "no_op", "none"}:
        return result
    if not features:
        return result
    if normalized_mode in {"baseline", "ablate"}:
        replacement = 0.0 if baseline is None else baseline
        result[:, features] = _selected_replacement(replacement, values, features)
        return result
    if normalized_mode in {"paired", "paired_replace", "pair"}:
        if paired_latents is None:
            raise ValueError("paired intervention requires paired_latents")
        result[:, features] = _selected_replacement(paired_latents, values, features)
        return result
    raise ValueError(f"unsupported intervention mode: {mode!r}")


def no_op_reconstruction(model: nn.Module, normalized_inputs: Tensor) -> tuple[Tensor, Tensor]:
    """Encode and decode without changing any latent feature."""

    if not hasattr(model, "encode") or not hasattr(model, "decode"):
        raise TypeError("model must expose encode and decode methods")
    latents = model.encode(normalized_inputs)
    return model.decode(latents), latents


def paired_effect_records(
    baseline_values: Tensor | np.ndarray | Sequence[float],
    intervened_values: Tensor | np.ndarray | Sequence[float],
    *,
    condition: str,
    sample_ids: Sequence[str | int] | None = None,
    target_features: Sequence[int] = (),
    control_features: Sequence[int] = (),
    outcome_name: str = "outcome",
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build one paired before/after effect record per sample."""

    before = torch.as_tensor(baseline_values, dtype=torch.float64).flatten()
    after = torch.as_tensor(intervened_values, dtype=torch.float64).flatten()
    if before.shape != after.shape or before.numel() == 0:
        raise ValueError("baseline and intervened values must be equal non-empty vectors")
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise ValueError("effect values must be finite")
    if sample_ids is None:
        resolved_ids: Sequence[str | int] = list(range(before.numel()))
    else:
        if len(sample_ids) != before.numel():
            raise ValueError("sample_ids length must match effect vectors")
        normalized_ids: list[str | int] = []
        for sample_id in sample_ids:
            if isinstance(sample_id, bool) or not isinstance(sample_id, (str, int)):
                raise ValueError("sample_ids must contain portable strings or integers")
            normalized_ids.append(
                require_portable_identifier(sample_id, name="sample_id")
                if isinstance(sample_id, str)
                else sample_id
            )
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("sample_ids must be unique")
        resolved_ids = normalized_ids

    shared: dict[str, Any] = {
        "condition": require_portable_identifier(condition, name="condition"),
        "outcome": require_portable_identifier(outcome_name, name="outcome"),
        "target_features": [int(value) for value in target_features],
        "control_features": [int(value) for value in control_features],
    }
    if metadata:
        validate_public_value(dict(metadata), name="metadata")
        shared["metadata"] = dict(metadata)
    records = []
    for row, sample_id in enumerate(resolved_ids):
        baseline_value = float(before[row])
        intervened_value = float(after[row])
        records.append(
            {
                **shared,
                "sample_id": sample_id,
                "baseline_value": baseline_value,
                "intervened_value": intervened_value,
                "effect": intervened_value - baseline_value,
            }
        )
    return records


make_paired_effect_records = paired_effect_records


def summarize_effect_records(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Summarize paired effects separately for every intervention condition."""

    grouped: dict[str, list[float]] = {}
    for record in records:
        condition = str(record["condition"])
        effect = float(record["effect"])
        if not math.isfinite(effect):
            raise ValueError("effect records must be finite")
        grouped.setdefault(condition, []).append(effect)
    summary: dict[str, dict[str, float | int]] = {}
    for condition, effects in sorted(grouped.items()):
        values = np.asarray(effects, dtype=np.float64)
        summary[condition] = {
            "count": int(values.size),
            "mean_effect": float(values.mean()),
            "median_effect": float(np.median(values)),
            "mean_absolute_effect": float(np.abs(values).mean()),
        }
    return summary


def run(args: Any) -> int:
    """Run reconstruction-space sensitivity diagnostics from a JSON config."""

    config_path = Path(args.config)
    configuration = load_verified_json_config(config_path)
    config = dict(configuration.data)
    seed = int(config.get("seed", 42))
    dataset_id = require_portable_identifier(config.get("dataset_id", ""), name="dataset_id")
    sample_ids = _run_sample_ids(config.get("sample_ids"), name="sample_ids")
    primary_binding = _activation_identity_digest(dataset_id, sample_ids)
    primary_role = f"activations.primary.{primary_binding}"

    representation_run_value = config.get("representation_run_dir")
    activation_spec = config.get("activations", config.get("activation_path", config.get("input")))
    if representation_run_value is None or activation_spec is None:
        raise ValueError("config must define representation_run_dir and activations")
    representation_run_dir = Path(str(representation_run_value)).expanduser()
    if not representation_run_dir.is_absolute():
        raise ValueError("representation_run_dir must be an absolute path")
    parent_manifest = verify_run_directory(representation_run_dir)
    if parent_manifest.command != "train-repr":
        raise ValueError("representation parent run command must be train-repr")
    if parent_manifest.evidence_level != "strict":
        raise ValueError("representation parent run must have strict evidence")
    parent_model_artifacts = [
        artifact for artifact in parent_manifest.artifacts if artifact.name == "model.pt"
    ]
    if len(parent_model_artifacts) != 1:
        raise ValueError("representation parent run must declare exactly one model.pt artifact")
    parent_model_artifact = parent_model_artifacts[0]
    checkpoint_path = representation_run_dir.resolve(strict=True) / "model.pt"
    parent_manifest_path = representation_run_dir.resolve(strict=True) / "manifest.json"
    parent_manifest_file = verify_file(parent_manifest_path)
    activation_path, activation_key, activation_expected = _verified_file_spec(
        activation_spec, name="activations", base_dir=config_path.parent
    )
    additional_paths: dict[str, Path] = {
        "representation.model": checkpoint_path,
        "representation.parent_manifest": parent_manifest_path,
        primary_role: activation_path,
    }
    expected_hashes: dict[str, str] = {
        "representation.model": parent_model_artifact.sha256,
        "representation.parent_manifest": parent_manifest_file.digest.sha256,
    }
    if activation_expected is not None:
        expected_hashes[primary_role] = activation_expected

    paired_spec = config.get("paired_activations")
    paired_key: str | None = None
    paired_role: str | None = None
    if paired_spec is not None:
        paired_dataset_id = require_portable_identifier(
            config.get("paired_dataset_id", ""), name="paired_dataset_id"
        )
        paired_sample_ids = _run_sample_ids(
            config.get("paired_sample_ids"), name="paired_sample_ids"
        )
        if paired_dataset_id != dataset_id or paired_sample_ids != sample_ids:
            raise ValueError(
                "paired activations require the same dataset and exactly aligned sample IDs"
            )
        paired_binding = _activation_identity_digest(
            paired_dataset_id, paired_sample_ids
        )
        paired_role = f"activations.paired.{paired_binding}"
        paired_path, paired_key, paired_expected = _verified_file_spec(
            paired_spec, name="paired_activations", base_dir=config_path.parent
        )
        additional_paths[paired_role] = paired_path
        if paired_expected is not None:
            expected_hashes[paired_role] = paired_expected

    baseline_spec = config.get("baseline", "mean")
    baseline_input = _baseline_file_spec(baseline_spec, base_dir=config_path.parent)
    if baseline_input is not None:
        baseline_path, _, baseline_expected = baseline_input
        additional_paths["latents.baseline"] = baseline_path
        if baseline_expected is not None:
            expected_hashes["latents.baseline"] = baseline_expected

    context = verify_configured_run_inputs(
        configuration,
        command="reconstruction-sensitivity",
        seed=seed,
        additional_input_paths=additional_paths,
        expected_additional_sha256=expected_hashes,
    )
    _assert_representation_parent_lineage(parent_manifest, context)
    model, normalizer, representation_metadata = load_verified_representation_checkpoint(
        context.additional_file("representation.model")
    )
    qualification = representation_metadata.get("qualification", {})
    if not isinstance(qualification, Mapping) or (
        qualification.get("metric_split") != "validation"
        or qualification.get("activation_fidelity_passed") is not True
    ):
        raise ValueError(
            "representation failed the held-out activation-fidelity qualification gate"
        )
    raw_inputs = load_verified_activation_array(
        context.additional_file(primary_role), key=activation_key
    )
    if raw_inputs.shape[1] != model.input_dim:
        raise ValueError("activation width does not match representation checkpoint")

    features = _validated_features(
        config.get("feature_indices", config.get("features", [])), model.latent_dim
    )
    if not features:
        raise ValueError("reconstruction sensitivity requires at least one feature index")
    if len(sample_ids) != raw_inputs.shape[0]:
        raise ValueError("sample_ids must be a list aligned to activation rows")
    assert_dataset_roster(context.inputs.dataset_manifest, (dataset_id,))
    verified_baseline_spec: Any = baseline_spec
    if baseline_input is not None:
        _, baseline_key, _ = baseline_input
        verified_baseline_spec = load_verified_activation_array(
            context.additional_file("latents.baseline"), key=baseline_key
        )
    model.eval()
    with torch.no_grad():
        normalized = normalizer.normalize(raw_inputs)
        no_op_normalized, latents = no_op_reconstruction(model, normalized)
        no_op_raw = normalizer.denormalize(no_op_normalized)
        no_op_scores = (no_op_raw - raw_inputs).square().mean(dim=1)
        records = paired_effect_records(
            no_op_scores,
            no_op_scores,
            condition="no_op",
            target_features=features,
            outcome_name="reconstruction_mse",
            sample_ids=sample_ids,
        )

        baseline = _resolve_baseline(verified_baseline_spec, latents, config_path.parent)
        baseline_latents = intervene_latents(
            latents, features, mode="baseline", baseline=baseline
        )
        baseline_raw = normalizer.denormalize(model.decode(baseline_latents))
        baseline_scores = (baseline_raw - raw_inputs).square().mean(dim=1)
        records.extend(
            paired_effect_records(
                no_op_scores,
                baseline_scores,
                condition="feature_baseline",
                target_features=features,
                outcome_name="reconstruction_mse",
                sample_ids=sample_ids,
            )
        )

        if paired_spec is not None:
            assert paired_role is not None
            paired_inputs = load_verified_activation_array(
                context.additional_file(paired_role), key=paired_key
            )
            if paired_inputs.shape != raw_inputs.shape:
                raise ValueError("paired activations must match the primary activation shape")
            paired_latents = model.encode(normalizer.normalize(paired_inputs))
            replaced = intervene_latents(
                latents, features, mode="paired", paired_latents=paired_latents
            )
            paired_raw = normalizer.denormalize(model.decode(replaced))
            paired_scores = (paired_raw - raw_inputs).square().mean(dim=1)
            records.extend(
                paired_effect_records(
                    no_op_scores,
                    paired_scores,
                    condition="paired_replacement",
                    target_features=features,
                    outcome_name="reconstruction_mse",
                    sample_ids=sample_ids,
                )
            )

        random_config = config.get("random_control", {})
        if random_config is not False:
            if not isinstance(random_config, Mapping):
                raise ValueError("random_control must be false or a JSON object")
            controls = matched_random_control_features(
                features,
                activation_frequency(latents),
                model_decoder_feature_norms(model),
                seed=seed,
                candidate_pool_size=int(random_config.get("candidate_pool_size", 8)),
            )
            control_latents = intervene_latents(
                latents, controls, mode="baseline", baseline=baseline
            )
            control_raw = normalizer.denormalize(model.decode(control_latents))
            control_scores = (control_raw - raw_inputs).square().mean(dim=1)
            records.extend(
                paired_effect_records(
                    no_op_scores,
                    control_scores,
                    condition="matched_random_control",
                    target_features=features,
                    control_features=controls,
                    outcome_name="reconstruction_mse",
                    sample_ids=sample_ids,
                )
            )

    for record in records:
        record["dataset_id"] = dataset_id
    roots = (
        context.inputs.training_code.root,
        context.inputs.model_code.root,
        context.inputs.analysis_code.root,
    )
    with RunTransaction(Path(args.output_dir), source_roots=roots) as transaction:
        effects_path = transaction.staging_dir / "effects.jsonl"
        summary_path = transaction.staging_dir / "summary.json"
        _atomic_json_lines_write(effects_path, records)
        _atomic_json_write(summary_path, summarize_effect_records(records))
        artifacts = transaction.artifact_digests(("effects.jsonl", "summary.json"))
        manifest = manifest_from_verified_inputs(
            context.inputs,
            artifacts=artifacts,
        )
        transaction.commit(manifest, verified_inputs=context.inputs)
    return 0


def _assert_representation_parent_lineage(parent_manifest: Any, context: Any) -> None:
    """Bind this run to the strict train-repr run that produced its model."""

    expected_pairs = (
        ("model_family", parent_manifest.model_family, context.model_family),
        ("model_revision", parent_manifest.model_revision, context.model_revision),
        (
            "training_code_sha",
            parent_manifest.training_code_sha,
            context.inputs.training_code.head_sha,
        ),
        ("model_code_sha", parent_manifest.model_code_sha, context.inputs.model_code.head_sha),
        ("checkpoint", parent_manifest.checkpoint, context.inputs.checkpoint.digest),
        (
            "dataset_manifest",
            parent_manifest.dataset_manifest,
            context.inputs.dataset_manifest.digest,
        ),
        ("sites", parent_manifest.sites, context.sites),
    )
    mismatched = [name for name, expected, observed in expected_pairs if expected != observed]
    if mismatched:
        raise ValueError(
            f"representation parent lineage differs from current run: {mismatched}"
        )
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(context.inputs.analysis_code.root),
            "merge-base",
            "--is-ancestor",
            parent_manifest.analysis_code_sha,
            context.inputs.analysis_code.head_sha,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            "representation parent analysis commit is not an ancestor of current analysis code"
        )


def _resolve_baseline(spec: Any, latents: Tensor, base_dir: Path) -> Tensor | float:
    if isinstance(spec, str):
        strategy = spec.lower().replace("-", "_")
        if strategy == "mean":
            return latents.mean(dim=0)
        if strategy in {"zero", "zeros"}:
            return 0.0
        return load_activation_array(spec, base_dir=base_dir)
    if isinstance(spec, Mapping):
        if "path" in spec:
            return load_activation_array(spec, base_dir=base_dir)
        strategy = str(spec.get("strategy", "mean")).lower()
        if strategy == "mean":
            return latents.mean(dim=0)
        if strategy in {"zero", "zeros"}:
            return 0.0
        raise ValueError(f"unsupported baseline strategy: {strategy!r}")
    if isinstance(spec, (int, float, list)):
        return torch.as_tensor(spec, dtype=latents.dtype)
    raise ValueError("unsupported baseline specification")


def _selected_replacement(
    replacement: Tensor | np.ndarray | float,
    reference: Tensor,
    features: Sequence[int],
) -> Tensor | float:
    values = torch.as_tensor(replacement, dtype=reference.dtype, device=reference.device)
    if values.ndim == 0:
        return float(values)
    if values.ndim == 1:
        if values.numel() == reference.shape[1]:
            return values[list(features)]
        if values.numel() in {1, len(features)}:
            return values
    if values.ndim == 2:
        if values.shape[1] == reference.shape[1] and values.shape[0] in {1, reference.shape[0]}:
            return values[:, list(features)]
        if values.shape[1] == len(features) and values.shape[0] in {1, reference.shape[0]}:
            return values
    raise ValueError("replacement values cannot be broadcast to the selected latent features")


def _as_latents(values: Tensor | np.ndarray) -> Tensor:
    tensor = torch.as_tensor(values)
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32)
    if tensor.ndim < 2:
        raise ValueError("latents must have at least two dimensions")
    tensor = tensor.reshape(-1, tensor.shape[-1])
    if tensor.shape[0] == 0 or tensor.shape[1] == 0 or not torch.isfinite(tensor).all():
        raise ValueError("latents must be non-empty and finite")
    return tensor


def _validated_features(features: Sequence[int], latent_dim: int) -> list[int]:
    resolved = [int(value) for value in features]
    if len(set(resolved)) != len(resolved):
        raise ValueError("feature indices must be unique")
    if any(value < 0 or value >= latent_dim for value in resolved):
        raise IndexError("feature index is outside the latent dimension")
    return resolved


def _verified_file_spec(
    spec: Any, *, name: str, base_dir: Path
) -> tuple[Path, str | None, str | None]:
    key: str | None = None
    expected: str | None = None
    if isinstance(spec, Mapping):
        unknown = sorted(set(spec) - {"path", "key", "expected_sha256"})
        if unknown or "path" not in spec:
            raise ValueError(f"{name} input fields mismatch: unknown={unknown}")
        raw_path = spec["path"]
        raw_key = spec.get("key")
        raw_expected = spec.get("expected_sha256")
        if raw_key is not None and not isinstance(raw_key, str):
            raise ValueError(f"{name} key must be a string")
        if raw_expected is not None and not isinstance(raw_expected, str):
            raise ValueError(f"{name} expected_sha256 must be a string")
        key = raw_key
        expected = raw_expected
    else:
        raw_path = spec
    if raw_path is None:
        raise ValueError(f"{name} is missing a path")
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path, key, expected


def _run_sample_ids(value: Any, *, name: str) -> list[str | int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result: list[str | int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise ValueError(f"{name} must contain strings or integers")
        result.append(
            require_portable_identifier(item, name=name) if isinstance(item, str) else item
        )
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


def _activation_identity_digest(dataset_id: str, sample_ids: Sequence[str | int]) -> str:
    payload = json.dumps(
        {"dataset_id": dataset_id, "sample_ids": list(sample_ids)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _baseline_file_spec(
    spec: Any, *, base_dir: Path
) -> tuple[Path, str | None, str | None] | None:
    if isinstance(spec, str) and spec.lower().replace("-", "_") in {"mean", "zero", "zeros"}:
        return None
    if isinstance(spec, Mapping) and "path" not in spec:
        unknown = sorted(set(spec) - {"strategy"})
        if unknown:
            raise ValueError(f"baseline strategy fields mismatch: unknown={unknown}")
        return None
    if isinstance(spec, (str, Mapping)):
        return _verified_file_spec(spec, name="baseline", base_dir=base_dir)
    return None


def model_decoder_feature_norms(model: nn.Module) -> Tensor:
    """Return decoder-direction norms for dense, Top-K, or PCA models."""

    decoder = getattr(model, "decoder", None)
    if decoder is not None:
        return decoder_feature_norms(decoder)
    components = getattr(model, "components", None)
    if isinstance(components, Tensor) and components.ndim == 2:
        return decoder_feature_norms(components.transpose(0, 1))
    raise TypeError("representation model does not expose linear decoder directions")


def _atomic_json_lines_write(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    def writer(handle: Any) -> None:
        for record in records:
            json.dump(dict(record), handle, sort_keys=True, allow_nan=False)
            handle.write("\n")

    _atomic_text_write(path, writer)


def _atomic_json_write(path: Path, payload: Any) -> None:
    def writer(handle: Any) -> None:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    _atomic_text_write(path, writer)


def _atomic_text_write(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


__all__ = [
    "activation_frequency",
    "decoder_feature_norms",
    "intervene_latents",
    "make_paired_effect_records",
    "matched_random_control_features",
    "model_decoder_feature_norms",
    "no_op_reconstruction",
    "paired_effect_records",
    "raw_space_decoder_feature_norms",
    "run",
    "select_matched_random_features",
    "summarize_effect_records",
]
