"""Model-in-the-loop causal edits for learned activation representations.

Unlike :mod:`pe_mechanism.causal`, which studies reconstruction-space effects, this
module puts decoded activations back into a live model through ``ModelAdapter`` and
measures prediction changes.  It is intentionally an in-memory API: callers decide
whether and how to publish the returned records.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn

from .adapters.base import ActivationRecord, ModelAdapter
from .identifiers import require_portable_identifier, require_public_label
from .representation import MeanRMSNormalizer


OutputKind = Literal["probabilities", "logits"]


@dataclass(frozen=True)
class PredictionMetrics:
    """Predictions and both aggregate and per-sample classification metrics."""

    probabilities: np.ndarray
    true_class_log_loss: np.ndarray
    accuracy: float
    log_loss: float


@dataclass(frozen=True)
class CausalPredictionResult:
    """One activation-edit outcome paired to two explicit references."""

    condition: str
    reference_scope: Literal["primary", "paired"]
    prediction: PredictionMetrics
    delta_log_loss_vs_model_baseline: np.ndarray
    delta_log_loss_vs_reconstruction: np.ndarray
    target_features: tuple[int, ...]
    control_features: tuple[int, ...] = ()


@dataclass(frozen=True)
class ModelCausalEvaluation:
    """Complete in-memory activation-edit and prediction result."""

    site: str
    dataset_id: str
    sample_ids: tuple[str | int, ...]
    baseline: PredictionMetrics
    baseline_activation: ActivationRecord
    conditions: Mapping[str, CausalPredictionResult]
    matched_control_features: tuple[int, ...]
    no_op_reconstruction_mse: float
    no_op_accuracy_drop: float
    representation_qualification: Mapping[str, Any]
    paired_baseline: PredictionMetrics | None = None
    paired_activation: ActivationRecord | None = None
    paired_dataset_id: str | None = None
    paired_sample_ids: tuple[str | int, ...] | None = None


@dataclass(frozen=True)
class _EncodedActivation:
    raw: Tensor
    latents: Tensor
    reconstruction: Tensor


def _module_device_and_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    for value in (*tuple(module.parameters()), *tuple(module.buffers())):
        dtype = value.dtype if value.is_floating_point() else torch.float32
        return value.device, dtype
    return torch.device("cpu"), torch.float32


def _validate_activation_record(record: ActivationRecord, *, site: str) -> Tensor:
    if not isinstance(record, ActivationRecord):
        raise TypeError("adapter capture must return ActivationRecord values")
    if record.site != site:
        raise ValueError(
            f"captured activation site {record.site!r} does not match {site!r}"
        )
    tensor = torch.as_tensor(record.tensor)
    if tuple(tensor.shape) != tuple(record.shape):
        raise ValueError("ActivationRecord shape metadata does not match its tensor")
    if tensor.ndim < 2 or tensor.shape[-1] == 0 or tensor.numel() == 0:
        raise ValueError(
            "captured activation must have a non-empty representation axis"
        )
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError("captured activation must contain only finite values")
    return tensor.detach()


def _normalizer_statistics(
    normalizer: MeanRMSNormalizer, *, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    if not hasattr(normalizer, "mean") or not hasattr(normalizer, "rms"):
        raise TypeError("normalizer must expose mean and rms tensors")
    mean = torch.as_tensor(normalizer.mean, device=device, dtype=dtype)
    rms = torch.as_tensor(normalizer.rms, device=device, dtype=dtype)
    if mean.ndim != 1 or rms.shape != mean.shape:
        raise ValueError(
            "normalizer mean and rms must be equal one-dimensional tensors"
        )
    if (
        not torch.isfinite(mean).all()
        or not torch.isfinite(rms).all()
        or (rms <= 0).any()
    ):
        raise ValueError("normalizer statistics must be finite with positive rms")
    return mean, rms


def _encode_activation(
    record: ActivationRecord,
    *,
    site: str,
    autoencoder: nn.Module,
    normalizer: MeanRMSNormalizer,
) -> _EncodedActivation:
    raw = _validate_activation_record(record, site=site)
    input_dim = getattr(autoencoder, "input_dim", None)
    if isinstance(input_dim, bool) or not isinstance(input_dim, int):
        raise TypeError("autoencoder must expose an integer input_dim")
    if raw.shape[-1] != input_dim:
        raise ValueError(
            f"activation width {raw.shape[-1]} does not match autoencoder input_dim {input_dim}"
        )
    if not hasattr(autoencoder, "encode") or not hasattr(autoencoder, "decode"):
        raise TypeError("autoencoder must expose encode and decode methods")
    device, dtype = _module_device_and_dtype(autoencoder)
    flattened = raw.reshape(-1, raw.shape[-1]).to(device=device, dtype=dtype)
    mean, rms = _normalizer_statistics(normalizer, device=device, dtype=dtype)
    if mean.shape[0] != input_dim:
        raise ValueError("normalizer width does not match autoencoder input_dim")
    normalized = (flattened - mean) / rms
    latents = autoencoder.encode(normalized)
    if (
        not isinstance(latents, Tensor)
        or latents.ndim != 2
        or latents.shape[0] != flattened.shape[0]
    ):
        raise ValueError("autoencoder encode must return [activation_rows, latent_dim]")
    latent_dim = getattr(autoencoder, "latent_dim", None)
    if isinstance(latent_dim, bool) or not isinstance(latent_dim, int):
        raise TypeError("autoencoder must expose an integer latent_dim")
    if latents.shape[1] != latent_dim or not torch.isfinite(latents).all():
        raise ValueError("encoded latents have an invalid shape or non-finite values")
    reconstruction = _decode_activation(
        latents,
        shape=tuple(raw.shape),
        autoencoder=autoencoder,
        mean=mean,
        rms=rms,
    )
    return _EncodedActivation(raw=raw, latents=latents, reconstruction=reconstruction)


def _decode_activation(
    latents: Tensor,
    *,
    shape: tuple[int, ...],
    autoencoder: nn.Module,
    mean: Tensor,
    rms: Tensor,
) -> Tensor:
    if latents.ndim != 2 or not torch.isfinite(latents).all():
        raise ValueError("edited latents must be a finite matrix")
    normalized = autoencoder.decode(latents)
    expected = (int(np.prod(shape[:-1])), shape[-1])
    if not isinstance(normalized, Tensor) or tuple(normalized.shape) != expected:
        actual = getattr(normalized, "shape", None)
        raise ValueError(
            f"autoencoder decode must return shape {expected}, got {actual}"
        )
    decoded = normalized * rms + mean
    if not torch.isfinite(decoded).all():
        raise ValueError("decoded activation contains non-finite values")
    return decoded.reshape(shape)


def _resolve_latent_baseline(value: Any, latents: Tensor) -> Tensor | float:
    if isinstance(value, str):
        strategy = value.lower().replace("-", "_")
        if strategy == "mean":
            return latents.mean(dim=0)
        if strategy in {"zero", "zeros"}:
            return 0.0
        raise ValueError("latent_baseline must be mean, zero, or numerical values")
    resolved = torch.as_tensor(value, dtype=latents.dtype, device=latents.device)
    if not torch.isfinite(resolved).all():
        raise ValueError("latent baseline values must be finite")
    return resolved


def _targets_from_batch(prepared_batch: Any, supplied: Any | None) -> np.ndarray:
    values = supplied
    if values is None and hasattr(prepared_batch, "y_test"):
        values = prepared_batch.y_test
    if values is None and isinstance(prepared_batch, Mapping):
        for key in ("y_test", "targets", "labels"):
            if key in prepared_batch:
                values = prepared_batch[key]
                break
    if values is None:
        raise ValueError(
            "encoded test targets must be supplied or present on the prepared batch"
        )
    if isinstance(values, Tensor):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            "encoded test targets must be a non-empty one-dimensional vector"
        )
    if array.dtype.kind not in "iu":
        if (
            array.dtype.kind != "f"
            or not np.isfinite(array).all()
            or not np.equal(array, np.floor(array)).all()
        ):
            raise TypeError("encoded test targets must be integer class indices")
    return array.astype(np.int64, copy=False)


def _validated_sample_ids(
    values: Sequence[str | int], *, expected: int, name: str
) -> tuple[str | int, ...]:
    if isinstance(values, (str, bytes)) or len(values) != expected:
        raise ValueError(f"{name} must align exactly to the test samples")
    resolved: list[str | int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(f"{name} must contain strings or integers")
        resolved.append(
            require_portable_identifier(value, name=name)
            if isinstance(value, str)
            else value
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must be unique")
    return tuple(resolved)


def _prediction_metrics(
    output: Any, targets: np.ndarray, *, output_kind: OutputKind
) -> PredictionMetrics:
    if isinstance(output, Tensor):
        values = output.detach().to(device="cpu", dtype=torch.float64).numpy()
    else:
        values = np.asarray(output, dtype=np.float64)
    if values.ndim == 3:
        if values.shape[0] != 1:
            raise ValueError(
                "model prediction batch axis must contain exactly one table"
            )
        values = values[0]
    if values.ndim != 2 or values.shape[0] != targets.shape[0] or values.shape[1] < 2:
        raise ValueError("model predictions must have shape [test_samples, classes]")
    if not np.isfinite(values).all():
        raise ValueError("model predictions must contain only finite values")
    if output_kind == "logits":
        shifted = values - values.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    elif output_kind == "probabilities":
        probabilities = values.copy()
        tolerance = 1e-6
        if np.any(probabilities < -tolerance) or np.any(probabilities > 1 + tolerance):
            raise ValueError("model probabilities are outside [0, 1]")
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=tolerance):
            raise ValueError("model probability rows must sum to one")
    else:
        raise ValueError("output_kind must be probabilities or logits")
    if np.any(targets < 0) or np.any(targets >= probabilities.shape[1]):
        raise ValueError("encoded test target is outside the prediction class range")
    true_probabilities = probabilities[np.arange(targets.shape[0]), targets]
    per_sample = -np.log(np.clip(true_probabilities, 1e-15, 1.0))
    return PredictionMetrics(
        probabilities=probabilities,
        true_class_log_loss=per_sample,
        accuracy=float(np.mean(probabilities.argmax(axis=1) == targets)),
        log_loss=float(per_sample.mean()),
    )


def _capture_prediction(
    adapter: ModelAdapter,
    model: Any,
    prepared_batch: Any,
    targets: np.ndarray,
    *,
    site: str,
    output_kind: OutputKind,
    model_sha: str,
    checkpoint_sha: str,
    preprocessing_view_id: str,
    feature_group_map: tuple[tuple[int, ...], ...] | None,
) -> tuple[PredictionMetrics, ActivationRecord]:
    with adapter.capture(
        model,
        sites=(site,),
        model_sha=model_sha,
        checkpoint_sha=checkpoint_sha,
        preprocessing_view_id=preprocessing_view_id,
        feature_group_map=feature_group_map,
    ) as captured:
        output = adapter.predict(model, prepared_batch)
    if site not in captured:
        raise RuntimeError(f"capture site {site!r} produced no activation")
    return _prediction_metrics(output, targets, output_kind=output_kind), captured[site]


class _SlicedReplacement:
    """Feed a merged captured activation back through one or repeated hook calls."""

    def __init__(self, replacement: Tensor, *, site: str) -> None:
        if replacement.ndim < 2 or not torch.isfinite(replacement).all():
            raise ValueError(
                "replacement activation must be finite with a representation axis"
            )
        self.replacement = replacement.detach()
        self.site = site
        self.offset = 0

    def __call__(self, record: ActivationRecord) -> Tensor:
        original = torch.as_tensor(record.tensor)
        if original.ndim != self.replacement.ndim or tuple(original.shape[1:]) != tuple(
            self.replacement.shape[1:]
        ):
            raise ValueError(
                f"runtime shape {tuple(original.shape)} is incompatible with captured "
                f"replacement {tuple(self.replacement.shape)}"
            )
        stop = self.offset + original.shape[0]
        if stop > self.replacement.shape[0]:
            raise RuntimeError(
                f"site {self.site!r} was called more often than during capture"
            )
        selected = self.replacement[self.offset : stop]
        self.offset = stop
        return selected.to(device=original.device, dtype=original.dtype)

    def verify_complete(self) -> None:
        if self.offset != self.replacement.shape[0]:
            raise RuntimeError(
                f"site {self.site!r} consumed {self.offset} of "
                f"{self.replacement.shape[0]} captured rows"
            )


def _intervened_prediction(
    adapter: ModelAdapter,
    model: Any,
    prepared_batch: Any,
    targets: np.ndarray,
    *,
    site: str,
    replacement: Tensor,
    output_kind: OutputKind,
) -> PredictionMetrics:
    intervention = _SlicedReplacement(replacement, site=site)
    with adapter.intervene(model, {site: intervention}):
        output = adapter.predict(model, prepared_batch)
    intervention.verify_complete()
    return _prediction_metrics(output, targets, output_kind=output_kind)


def _condition_result(
    name: str,
    prediction: PredictionMetrics,
    *,
    reference_scope: Literal["primary", "paired"],
    model_baseline: PredictionMetrics,
    reconstruction_baseline: PredictionMetrics,
    target_features: tuple[int, ...],
    control_features: tuple[int, ...] = (),
) -> CausalPredictionResult:
    if (
        prediction.true_class_log_loss.shape != model_baseline.true_class_log_loss.shape
        or (
            prediction.true_class_log_loss.shape
            != reconstruction_baseline.true_class_log_loss.shape
        )
    ):
        raise ValueError("prediction references must contain the same test samples")
    return CausalPredictionResult(
        condition=name,
        reference_scope=reference_scope,
        prediction=prediction,
        delta_log_loss_vs_model_baseline=(
            prediction.true_class_log_loss - model_baseline.true_class_log_loss
        ),
        delta_log_loss_vs_reconstruction=(
            prediction.true_class_log_loss - reconstruction_baseline.true_class_log_loss
        ),
        target_features=target_features,
        control_features=control_features,
    )


def _all_modules(*roots: Any) -> tuple[Any, ...]:
    modules: list[Any] = []
    seen: set[int] = set()
    for root in roots:
        candidates = (root, getattr(root, "model_", root))
        for candidate in candidates:
            contained = (
                candidate.modules() if hasattr(candidate, "modules") else (candidate,)
            )
            for module in contained:
                if id(module) not in seen and hasattr(module, "training"):
                    modules.append(module)
                    seen.add(id(module))
    return tuple(modules)


@contextmanager
def _preserve_training_states(*roots: Any) -> Iterator[None]:
    modules = _all_modules(*roots)
    states = tuple(bool(module.training) for module in modules)
    try:
        yield None
    finally:
        for module, state in zip(modules, states, strict=True):
            module.training = state


def run_model_causal_edits(
    adapter: ModelAdapter,
    model: Any,
    prepared_batch: Any,
    *,
    site: str,
    autoencoder: nn.Module,
    normalizer: MeanRMSNormalizer,
    target_features: Sequence[int],
    dataset_id: str,
    sample_ids: Sequence[str | int],
    representation_qualification: Mapping[str, Any],
    targets: Any | None = None,
    latent_baseline: Any = "mean",
    random_seed: int = 42,
    random_candidate_pool_size: int = 8,
    paired_batch: Any | None = None,
    paired_dataset_id: str | None = None,
    paired_sample_ids: Sequence[str | int] | None = None,
    paired_targets: Any | None = None,
    output_kind: OutputKind = "probabilities",
    model_sha: str = "",
    checkpoint_sha: str = "",
    preprocessing_view_id: str = "raw",
    paired_preprocessing_view_id: str = "paired-raw",
    feature_group_map: tuple[tuple[int, ...], ...] | None = None,
    paired_feature_group_map: tuple[tuple[int, ...], ...] | None = None,
    max_no_op_reconstruction_mse: float | None = None,
    max_no_op_accuracy_drop: float = 0.005,
) -> ModelCausalEvaluation:
    """Run learned-latent activation edits and put every result back into the model.

    ``target_baseline_edit`` and ``matched_random_edit`` are compared to both the
    untouched model and the AE/SAE no-op reconstruction.  When ``paired_batch`` is
    supplied, forward and reverse target-feature swaps are run symmetrically.  Rescue
    conditions must reproduce their corresponding no-op reconstruction exactly.
    """

    if not isinstance(site, str) or not site:
        raise ValueError("site must be a non-empty adapter activation name")
    if not isinstance(representation_qualification, Mapping):
        raise TypeError("representation_qualification must be checkpoint metadata")
    if representation_qualification.get("metric_split") != "validation" or (
        representation_qualification.get("activation_fidelity_passed") is not True
    ):
        raise RuntimeError(
            "representation failed the held-out activation-fidelity qualification gate"
        )
    latent_dim = getattr(autoencoder, "latent_dim", None)
    if (
        isinstance(latent_dim, bool)
        or not isinstance(latent_dim, int)
        or latent_dim <= 0
    ):
        raise TypeError("autoencoder must expose a positive integer latent_dim")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in target_features
    ):
        raise TypeError("target_features must contain integer indices")
    features = tuple(target_features)
    if not features:
        raise ValueError("target_features must not be empty")
    if len(set(features)) != len(features):
        raise ValueError("target_features must be unique")
    if any(value < 0 or value >= latent_dim for value in features):
        raise IndexError("target feature is outside the autoencoder latent dimension")
    if (
        isinstance(random_candidate_pool_size, bool)
        or not isinstance(random_candidate_pool_size, int)
        or random_candidate_pool_size <= 0
    ):
        raise ValueError("random_candidate_pool_size must be positive")
    if paired_batch is None and paired_targets is not None:
        raise ValueError("paired_targets requires paired_batch")
    if max_no_op_reconstruction_mse is not None and (
        not np.isfinite(max_no_op_reconstruction_mse)
        or max_no_op_reconstruction_mse < 0
    ):
        raise ValueError("max_no_op_reconstruction_mse must be finite and non-negative")
    if not np.isfinite(max_no_op_accuracy_drop) or max_no_op_accuracy_drop < 0:
        raise ValueError("max_no_op_accuracy_drop must be finite and non-negative")
    primary_targets = _targets_from_batch(prepared_batch, targets)
    resolved_dataset_id = require_public_label(dataset_id, name="dataset_id")
    resolved_sample_ids = _validated_sample_ids(
        sample_ids, expected=primary_targets.size, name="sample_ids"
    )
    if paired_batch is None and (
        paired_dataset_id is not None or paired_sample_ids is not None
    ):
        raise ValueError("paired identity fields require paired_batch")
    resolved_paired_targets = (
        None
        if paired_batch is None
        else _targets_from_batch(paired_batch, paired_targets)
    )
    resolved_paired_dataset_id = None
    resolved_paired_sample_ids = None
    if paired_batch is not None:
        if paired_dataset_id is None or paired_sample_ids is None:
            raise ValueError(
                "paired_batch requires paired_dataset_id and paired_sample_ids"
            )
        resolved_paired_dataset_id = require_public_label(
            paired_dataset_id, name="paired_dataset_id"
        )
        assert resolved_paired_targets is not None
        resolved_paired_sample_ids = _validated_sample_ids(
            paired_sample_ids,
            expected=resolved_paired_targets.size,
            name="paired_sample_ids",
        )
        if resolved_paired_dataset_id != resolved_dataset_id:
            raise ValueError("paired edits require the same dataset_id")
        if resolved_paired_sample_ids != resolved_sample_ids:
            raise ValueError("paired edits require exactly aligned sample IDs")

    conditions: dict[str, CausalPredictionResult] = {}
    paired_baseline = None
    paired_record = None
    # Keep this dependency lazy so the existing reconstruction-space causal CLI can
    # import this model-in-the-loop module without creating an import cycle.
    from .causal import (
        activation_frequency,
        decoder_feature_norms,
        intervene_latents,
        matched_random_control_features,
    )

    with _preserve_training_states(model, autoencoder, normalizer):
        autoencoder.eval()
        normalizer.eval()
        with torch.inference_mode():
            baseline, baseline_record = _capture_prediction(
                adapter,
                model,
                prepared_batch,
                primary_targets,
                site=site,
                output_kind=output_kind,
                model_sha=model_sha,
                checkpoint_sha=checkpoint_sha,
                preprocessing_view_id=preprocessing_view_id,
                feature_group_map=feature_group_map,
            )
            encoded = _encode_activation(
                baseline_record,
                site=site,
                autoencoder=autoencoder,
                normalizer=normalizer,
            )
            reconstruction_mse = float(
                (encoded.reconstruction.to(encoded.raw.device) - encoded.raw)
                .square()
                .mean()
            )
            if not np.isfinite(reconstruction_mse):
                raise ValueError("no-op reconstruction MSE is non-finite")
            if (
                max_no_op_reconstruction_mse is not None
                and reconstruction_mse > max_no_op_reconstruction_mse
            ):
                raise RuntimeError(
                    f"no-op reconstruction MSE {reconstruction_mse:.6g} exceeds "
                    f"limit {max_no_op_reconstruction_mse:.6g}"
                )
            no_op = _intervened_prediction(
                adapter,
                model,
                prepared_batch,
                primary_targets,
                site=site,
                replacement=encoded.reconstruction,
                output_kind=output_kind,
            )
            no_op_accuracy_drop = baseline.accuracy - no_op.accuracy
            if no_op_accuracy_drop > max_no_op_accuracy_drop:
                raise RuntimeError(
                    f"no-op reconstruction accuracy drop {no_op_accuracy_drop:.6g} exceeds "
                    f"limit {max_no_op_accuracy_drop:.6g}"
                )
            conditions["no_op_reconstruction"] = _condition_result(
                "no_op_reconstruction",
                no_op,
                reference_scope="primary",
                model_baseline=baseline,
                reconstruction_baseline=no_op,
                target_features=features,
            )

            baseline_value = _resolve_latent_baseline(latent_baseline, encoded.latents)
            edited_latents = intervene_latents(
                encoded.latents,
                features,
                mode="baseline",
                baseline=baseline_value,
            )
            mean, rms = _normalizer_statistics(
                normalizer,
                device=encoded.latents.device,
                dtype=encoded.latents.dtype,
            )
            target_replacement = _decode_activation(
                edited_latents,
                shape=tuple(encoded.raw.shape),
                autoencoder=autoencoder,
                mean=mean,
                rms=rms,
            )
            target_prediction = _intervened_prediction(
                adapter,
                model,
                prepared_batch,
                primary_targets,
                site=site,
                replacement=target_replacement,
                output_kind=output_kind,
            )
            conditions["target_baseline_edit"] = _condition_result(
                "target_baseline_edit",
                target_prediction,
                reference_scope="primary",
                model_baseline=baseline,
                reconstruction_baseline=no_op,
                target_features=features,
            )

            if not hasattr(autoencoder, "decoder"):
                raise TypeError(
                    "autoencoder must expose decoder weights for matched controls"
                )
            controls = tuple(
                matched_random_control_features(
                    features,
                    activation_frequency(encoded.latents),
                    decoder_feature_norms(autoencoder.decoder),
                    seed=random_seed,
                    candidate_pool_size=random_candidate_pool_size,
                )
            )
            random_latents = intervene_latents(
                encoded.latents,
                controls,
                mode="baseline",
                baseline=baseline_value,
            )
            random_replacement = _decode_activation(
                random_latents,
                shape=tuple(encoded.raw.shape),
                autoencoder=autoencoder,
                mean=mean,
                rms=rms,
            )
            random_prediction = _intervened_prediction(
                adapter,
                model,
                prepared_batch,
                primary_targets,
                site=site,
                replacement=random_replacement,
                output_kind=output_kind,
            )
            conditions["matched_random_edit"] = _condition_result(
                "matched_random_edit",
                random_prediction,
                reference_scope="primary",
                model_baseline=baseline,
                reconstruction_baseline=no_op,
                target_features=features,
                control_features=controls,
            )

            rescued_latents = edited_latents.clone()
            rescued_latents[:, list(features)] = encoded.latents[:, list(features)]
            if not torch.equal(rescued_latents, encoded.latents):
                raise RuntimeError(
                    "target rescue did not restore the original latent values"
                )
            rescue_replacement = _decode_activation(
                rescued_latents,
                shape=tuple(encoded.raw.shape),
                autoencoder=autoencoder,
                mean=mean,
                rms=rms,
            )
            if not torch.equal(rescue_replacement, encoded.reconstruction):
                raise RuntimeError(
                    "target rescue did not restore the no-op reconstruction"
                )
            rescue_prediction = _intervened_prediction(
                adapter,
                model,
                prepared_batch,
                primary_targets,
                site=site,
                replacement=rescue_replacement,
                output_kind=output_kind,
            )
            if not np.array_equal(rescue_prediction.probabilities, no_op.probabilities):
                raise RuntimeError(
                    "target rescue prediction does not exactly match no-op"
                )
            conditions["rescue"] = _condition_result(
                "rescue",
                rescue_prediction,
                reference_scope="primary",
                model_baseline=baseline,
                reconstruction_baseline=no_op,
                target_features=features,
            )

            if paired_batch is not None:
                assert resolved_paired_targets is not None
                paired_baseline, paired_record = _capture_prediction(
                    adapter,
                    model,
                    paired_batch,
                    resolved_paired_targets,
                    site=site,
                    output_kind=output_kind,
                    model_sha=model_sha,
                    checkpoint_sha=checkpoint_sha,
                    preprocessing_view_id=paired_preprocessing_view_id,
                    feature_group_map=paired_feature_group_map,
                )
                paired_raw = _validate_activation_record(paired_record, site=site)
                if tuple(paired_raw.shape) != tuple(encoded.raw.shape):
                    raise ValueError(
                        "paired activation must exactly match primary activation shape"
                    )
                if paired_record.axis_names != baseline_record.axis_names:
                    raise ValueError(
                        "paired activation axis semantics must match primary activation"
                    )
                paired_encoded = _encode_activation(
                    paired_record,
                    site=site,
                    autoencoder=autoencoder,
                    normalizer=normalizer,
                )
                paired_no_op = _intervened_prediction(
                    adapter,
                    model,
                    paired_batch,
                    resolved_paired_targets,
                    site=site,
                    replacement=paired_encoded.reconstruction,
                    output_kind=output_kind,
                )
                conditions["paired_no_op_reconstruction"] = _condition_result(
                    "paired_no_op_reconstruction",
                    paired_no_op,
                    reference_scope="paired",
                    model_baseline=paired_baseline,
                    reconstruction_baseline=paired_no_op,
                    target_features=features,
                )

                paired_into_primary = intervene_latents(
                    encoded.latents,
                    features,
                    mode="paired",
                    paired_latents=paired_encoded.latents,
                )
                paired_edit_replacement = _decode_activation(
                    paired_into_primary,
                    shape=tuple(encoded.raw.shape),
                    autoencoder=autoencoder,
                    mean=mean,
                    rms=rms,
                )
                paired_edit_prediction = _intervened_prediction(
                    adapter,
                    model,
                    prepared_batch,
                    primary_targets,
                    site=site,
                    replacement=paired_edit_replacement,
                    output_kind=output_kind,
                )
                conditions["paired_activation_edit"] = _condition_result(
                    "paired_activation_edit",
                    paired_edit_prediction,
                    reference_scope="primary",
                    model_baseline=baseline,
                    reconstruction_baseline=no_op,
                    target_features=features,
                )

                primary_into_paired = intervene_latents(
                    paired_encoded.latents,
                    features,
                    mode="paired",
                    paired_latents=encoded.latents,
                )
                paired_mean, paired_rms = _normalizer_statistics(
                    normalizer,
                    device=paired_encoded.latents.device,
                    dtype=paired_encoded.latents.dtype,
                )
                reverse_replacement = _decode_activation(
                    primary_into_paired,
                    shape=tuple(paired_encoded.raw.shape),
                    autoencoder=autoencoder,
                    mean=paired_mean,
                    rms=paired_rms,
                )
                reverse_prediction = _intervened_prediction(
                    adapter,
                    model,
                    paired_batch,
                    resolved_paired_targets,
                    site=site,
                    replacement=reverse_replacement,
                    output_kind=output_kind,
                )
                conditions["paired_reverse"] = _condition_result(
                    "paired_reverse",
                    reverse_prediction,
                    reference_scope="paired",
                    model_baseline=paired_baseline,
                    reconstruction_baseline=paired_no_op,
                    target_features=features,
                )

                paired_rescued_latents = primary_into_paired.clone()
                paired_rescued_latents[:, list(features)] = paired_encoded.latents[
                    :, list(features)
                ]
                if not torch.equal(paired_rescued_latents, paired_encoded.latents):
                    raise RuntimeError(
                        "paired rescue did not restore paired latent values"
                    )
                paired_rescue_replacement = _decode_activation(
                    paired_rescued_latents,
                    shape=tuple(paired_encoded.raw.shape),
                    autoencoder=autoencoder,
                    mean=paired_mean,
                    rms=paired_rms,
                )
                if not torch.equal(
                    paired_rescue_replacement, paired_encoded.reconstruction
                ):
                    raise RuntimeError(
                        "paired rescue did not restore paired reconstruction"
                    )
                paired_rescue_prediction = _intervened_prediction(
                    adapter,
                    model,
                    paired_batch,
                    resolved_paired_targets,
                    site=site,
                    replacement=paired_rescue_replacement,
                    output_kind=output_kind,
                )
                if not np.array_equal(
                    paired_rescue_prediction.probabilities,
                    paired_no_op.probabilities,
                ):
                    raise RuntimeError(
                        "paired rescue prediction does not exactly match no-op"
                    )
                conditions["paired_rescue"] = _condition_result(
                    "paired_rescue",
                    paired_rescue_prediction,
                    reference_scope="paired",
                    model_baseline=paired_baseline,
                    reconstruction_baseline=paired_no_op,
                    target_features=features,
                )

    return ModelCausalEvaluation(
        site=site,
        dataset_id=resolved_dataset_id,
        sample_ids=resolved_sample_ids,
        baseline=baseline,
        baseline_activation=baseline_record,
        conditions=conditions,
        matched_control_features=controls,
        no_op_reconstruction_mse=reconstruction_mse,
        no_op_accuracy_drop=no_op_accuracy_drop,
        representation_qualification=dict(representation_qualification),
        paired_baseline=paired_baseline,
        paired_activation=paired_record,
        paired_dataset_id=resolved_paired_dataset_id,
        paired_sample_ids=resolved_paired_sample_ids,
    )


__all__ = [
    "CausalPredictionResult",
    "ModelCausalEvaluation",
    "OutputKind",
    "PredictionMetrics",
    "run_model_causal_edits",
]
