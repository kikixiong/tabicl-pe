"""Fail-closed utilities for TabICL fingerprint causal interventions."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


FINGERPRINT_INTERVENTIONS = ("correct", "zero", "permuted", "collapsed")
PERMUTATION_ALGORITHM = "cyclic_shift_left_one_v1"
ALIGNMENT_FIELDS = (
    "prediction_shape",
    "encoded_target_sha256",
    "row_fingerprint_sha256",
    "test_target_sha256",
    "class_labels_sha256",
    "train_content_sha256",
    "test_content_sha256",
)
_FULLSIZE_ARCHITECTURE = {
    "embed_dim": 128,
    "col_num_blocks": 3,
    "col_nhead": 8,
    "col_num_inds": 128,
    "row_num_blocks": 3,
    "row_nhead": 8,
    "icl_num_blocks": 12,
    "icl_nhead": 8,
}


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise TypeError("object arrays cannot be hashed safely")
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def cyclic_derangement_indices(num_tokens: int) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Return a deterministic cyclic derangement, explicitly flagging H=1."""
    if isinstance(num_tokens, bool) or not isinstance(num_tokens, int) or num_tokens < 1:
        raise ValueError("num_tokens must be a positive integer")
    indices = tuple(range(1, num_tokens)) + (0,)
    fixed_points = sum(index == value for index, value in enumerate(indices))
    degenerate = num_tokens == 1
    if not degenerate and fixed_points:
        raise RuntimeError("cyclic permutation unexpectedly contains a fixed point")
    encoded = np.asarray(indices, dtype=np.dtype("<i8"))
    return indices, {
        "algorithm": PERMUTATION_ALGORITHM,
        "token_count": num_tokens,
        "indices_sha256": _array_sha256(encoded),
        "fixed_point_count": fixed_points,
        "effective": not degenerate,
        "degenerate_reason": (
            "single_feature_token_has_no_derangement" if degenerate else None
        ),
    }


class FingerprintForwardPreHook:
    """Inject one causal condition into raw ``TabICL.forward`` calls."""

    def __init__(self, intervention: str) -> None:
        if intervention not in FINGERPRINT_INTERVENTIONS:
            raise ValueError(f"unknown fingerprint intervention: {intervention}")
        self.intervention = intervention
        self.call_count = 0
        self._token_count: int | None = None
        self._permutation: dict[str, Any] | None = None

    def __call__(
        self,
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        import torch

        if "fingerprint_intervention" in kwargs or "fingerprint_permutation" in kwargs:
            raise RuntimeError("fingerprint intervention kwargs were already supplied")
        X = kwargs.get("X", args[0] if args else None)
        if not isinstance(X, torch.Tensor) or X.ndim != 3:
            raise RuntimeError("fingerprint hook requires a three-dimensional X tensor")
        if not getattr(module, "row_fingerprint", False):
            raise RuntimeError("fingerprint hook was attached to a non-fingerprint model")
        if getattr(module, "row_identity_mode", None) != "none":
            raise RuntimeError("fingerprint model must use row_identity_mode='none'")
        token_count = int(module._num_row_identity_tokens(int(X.shape[-1])))
        if self._token_count is None:
            self._token_count = token_count
        elif self._token_count != token_count:
            raise RuntimeError("feature-token count changed across prediction batches")

        updated = dict(kwargs)
        updated["fingerprint_intervention"] = self.intervention
        if self.intervention == "permuted":
            indices, metadata = cyclic_derangement_indices(token_count)
            if self._permutation is None:
                self._permutation = metadata
            elif self._permutation != metadata:
                raise RuntimeError("fingerprint permutation changed across batches")
            permutation = torch.tensor(indices, dtype=torch.long, device=X.device)
            updated["fingerprint_permutation"] = permutation.unsqueeze(0).expand(
                int(X.shape[0]), -1
            )
        self.call_count += 1
        return args, updated

    def metadata(self) -> dict[str, Any]:
        if self.call_count < 1 or self._token_count is None:
            raise RuntimeError("fingerprint intervention hook did not observe inference")
        return {
            "intervention": self.intervention,
            "forward_call_count": self.call_count,
            "feature_token_count": self._token_count,
            "permutation": self._permutation,
        }


def fingerprint_checkpoint_contract(
    checkpoint: Path,
    *,
    comparison_step: int = 50_000,
) -> dict[str, Any]:
    """Validate the exact treatment and architecture needed by the causal study."""
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("fingerprint checkpoint payload must be an object")
    if payload.get("curr_step") != comparison_step:
        raise ValueError(f"fingerprint checkpoint is not at step {comparison_step}")
    config = payload.get("config")
    state_dict = payload.get("state_dict")
    if not isinstance(config, Mapping) or not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("fingerprint checkpoint lacks model config/state")
    prior = payload.get("prior_stream")
    if not isinstance(prior, Mapping) or any(
        prior.get(key) != expected
        for key, expected in (
            ("cursor", comparison_step),
            ("experiment_seed", 42),
            ("ddp_rank", 0),
            ("world_size", 1),
        )
    ):
        raise ValueError("fingerprint checkpoint prior stream is invalid")
    if (
        config.get("row_identity_mode") != "none"
        or config.get("row_fingerprint") is not True
        or config.get("row_fingerprint_dim") != 16
        or config.get("col_feature_group") != "same"
        or config.get("col_target_aware") is not True
    ):
        raise ValueError("fingerprint checkpoint treatment is invalid")
    observed_architecture = {
        key: config.get(key) for key in _FULLSIZE_ARCHITECTURE
    }
    if observed_architecture != _FULLSIZE_ARCHITECTURE:
        raise ValueError("fingerprint checkpoint is not the expected fullsize architecture")
    return {
        "curr_step": int(payload["curr_step"]),
        "model_state_tensors": len(state_dict),
        "model_state_elements": sum(value.numel() for value in state_dict.values()),
        "model_config": dict(config),
        "prior_stream": dict(prior),
    }


def make_fingerprint_causal_system(
    external_system_model: type,
    base_factory: Any,
) -> type:
    """Create a TabArena external system with a scoped raw-model pre-hook."""
    base = base_factory(external_system_model)

    class FingerprintCausalSystem(base):
        def __init__(self, *, intervention: str, **kwargs: Any) -> None:
            if intervention not in FINGERPRINT_INTERVENTIONS:
                raise ValueError(f"unknown fingerprint intervention: {intervention}")
            classifier_options = kwargs.get("classifier_options")
            if not isinstance(classifier_options, Mapping) or (
                classifier_options.get("kv_cache") is not False
            ):
                raise ValueError("fingerprint causal evaluation requires kv_cache=False")
            self.intervention = intervention
            self._fingerprint_hook: FingerprintForwardPreHook | None = None
            self._fingerprint_hook_handle: Any | None = None
            super().__init__(**kwargs)

        def _fit_system(self, X: Any, y: Any, **kwargs: Any) -> Any:
            if self._fingerprint_hook_handle is not None:
                raise RuntimeError("fingerprint hook is already registered")
            fitted = super()._fit_system(X, y, **kwargs)
            classifier = self.model
            raw_model = getattr(classifier, "model_", None)
            if raw_model is None:
                raise RuntimeError("fitted classifier lacks its raw TabICL model")
            if (
                getattr(raw_model, "row_fingerprint", None) is not True
                or getattr(raw_model, "row_identity_mode", None) != "none"
            ):
                raise RuntimeError("loaded model is not the fingerprint treatment")
            if getattr(classifier, "model_kv_cache_", None) is not None:
                raise RuntimeError("kv-cache state exists despite kv_cache=False")
            if getattr(raw_model, "_forward_pre_hooks", None):
                raise RuntimeError("raw TabICL model already has a forward pre-hook")
            hook = FingerprintForwardPreHook(self.intervention)
            handle = raw_model.register_forward_pre_hook(hook, with_kwargs=True)
            registered = getattr(raw_model, "_forward_pre_hooks", {})
            if len(registered) != 1 or handle.id not in registered:
                handle.remove()
                raise RuntimeError("fingerprint intervention hook registration is ambiguous")
            self._fingerprint_hook = hook
            self._fingerprint_hook_handle = handle
            return fitted

        def causal_metadata(self) -> dict[str, Any]:
            if self._fingerprint_hook is None or self._fingerprint_hook_handle is None:
                raise RuntimeError("fingerprint intervention hook is not active")
            raw_model = getattr(self.model, "model_", None)
            registered = getattr(raw_model, "_forward_pre_hooks", {})
            if (
                len(registered) != 1
                or self._fingerprint_hook_handle.id not in registered
            ):
                raise RuntimeError("fingerprint intervention hook was removed or duplicated")
            return self._fingerprint_hook.metadata()

        def cleanup(self) -> None:
            handle = self._fingerprint_hook_handle
            self._fingerprint_hook_handle = None
            self._fingerprint_hook = None
            try:
                if handle is not None:
                    handle.remove()
            finally:
                super().cleanup()

    FingerprintCausalSystem.__name__ = "FingerprintCausalSystem"
    FingerprintCausalSystem.__qualname__ = "FingerprintCausalSystem"
    return FingerprintCausalSystem


def _write_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("causal metadata destination must be fresh")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _prediction_evidence(path: Path) -> dict[str, Any]:
    expected = {
        "probabilities",
        "encoded_targets",
        "row_fingerprints",
        "test_target_fingerprints",
        "class_labels_utf8",
        "train_content_sha256",
        "test_content_sha256",
    }
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != expected:
            raise ValueError("causal prediction NPZ fields differ from the frozen schema")
        arrays = {name: np.ascontiguousarray(payload[name]) for name in expected}
    probabilities = arrays["probabilities"]
    encoded = arrays["encoded_targets"]
    rows = arrays["row_fingerprints"]
    targets = arrays["test_target_fingerprints"]
    labels = arrays["class_labels_utf8"]
    train_digest = arrays["train_content_sha256"]
    test_digest = arrays["test_content_sha256"]
    if probabilities.dtype != np.dtype("<f4") or probabilities.ndim != 2:
        raise ValueError("causal probabilities must be a float32 matrix")
    if encoded.dtype != np.dtype("<i8") or encoded.shape != (probabilities.shape[0],):
        raise ValueError("causal encoded targets are misaligned")
    if rows.dtype != np.uint8 or rows.shape != (probabilities.shape[0], 32):
        raise ValueError("causal row fingerprints are misaligned")
    if targets.dtype != np.uint8 or targets.shape != (probabilities.shape[0], 32):
        raise ValueError("causal target fingerprints are misaligned")
    if labels.dtype != np.uint8 or labels.ndim != 1:
        raise ValueError("causal class label evidence is invalid")
    if train_digest.dtype != np.uint8 or train_digest.shape != (32,):
        raise ValueError("causal training-table digest is invalid")
    if test_digest.dtype != np.uint8 or test_digest.shape != (32,):
        raise ValueError("causal test-table digest is invalid")
    if probabilities.shape[0] < 1 or probabilities.shape[1] < 2:
        raise ValueError("causal probability matrix is empty")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0) or np.any(
        probabilities > 1.0
    ):
        raise ValueError("causal probabilities are not finite values in [0,1]")
    if not np.allclose(
        probabilities.sum(axis=1, dtype=np.float64), 1.0, rtol=1e-6, atol=1e-6
    ):
        raise ValueError("causal probability rows do not sum to one")
    return {
        "prediction_shape": list(probabilities.shape),
        "probability_dtype": probabilities.dtype.str,
        "probabilities_sha256": _array_sha256(probabilities),
        "encoded_target_sha256": _array_sha256(encoded),
        "row_fingerprint_sha256": _array_sha256(rows),
        "test_target_sha256": _array_sha256(targets),
        "class_labels_sha256": _array_sha256(labels),
        "train_content_sha256": train_digest.tobytes().hex(),
        "test_content_sha256": test_digest.tobytes().hex(),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def capture_causal_prediction(
    probabilities: Any,
    *,
    train_features: Any,
    train_targets: Any,
    test_features: Any,
    test_targets: Any,
    dataset: str,
    intervention: str,
    causal_metadata: Mapping[str, Any],
    capture_root: Path,
) -> dict[str, Any]:
    """Write one private float32 prediction matrix plus public hash evidence."""
    import pandas as pd
    from pe_mechanism.tabarena_formal_runner import (
        _canonical_scalar,
        _class_label_evidence,
        _feature_table_fingerprint,
        _row_fingerprints,
        _table_content_fingerprint,
        _test_target_fingerprints,
    )

    if intervention not in FINGERPRINT_INTERVENTIONS:
        raise ValueError("unknown fingerprint intervention")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("causal dataset name must be non-empty")
    if not isinstance(probabilities, pd.DataFrame):
        raise TypeError("causal probabilities must be a pandas DataFrame")
    values = np.ascontiguousarray(probabilities.to_numpy(copy=False), dtype=np.dtype("<f4"))
    class_bytes, _ = _class_label_evidence(list(probabilities.columns))
    rows = _row_fingerprints(list(probabilities.index))
    targets = _test_target_fingerprints(test_targets, expected_index=probabilities.index)
    canonical_columns = [
        json.dumps(
            _canonical_scalar(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for value in probabilities.columns
    ]
    class_to_index = {value: index for index, value in enumerate(canonical_columns)}
    try:
        encoded_targets = np.asarray(
            [
                class_to_index[
                    json.dumps(
                        _canonical_scalar(value),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                ]
                for value in test_targets
            ],
            dtype=np.dtype("<i8"),
        )
    except KeyError as error:
        raise ValueError("test target is absent from prediction classes") from error
    train_digest = bytes.fromhex(_table_content_fingerprint(train_features, train_targets))
    test_digest = bytes.fromhex(_feature_table_fingerprint(test_features, domain="test"))
    dataset_digest = hashlib.sha256(dataset.encode("utf-8")).hexdigest()
    directory = capture_root / intervention / dataset_digest
    output = directory / "predictions.npz"
    if output.exists() or output.is_symlink():
        raise FileExistsError("causal prediction destination must be fresh")
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".predictions.", suffix=".tmp", dir=directory
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as handle:
            np.savez_compressed(
                handle,
                probabilities=values,
                encoded_targets=encoded_targets,
                row_fingerprints=rows,
                test_target_fingerprints=targets,
                class_labels_utf8=np.frombuffer(class_bytes, dtype=np.uint8),
                train_content_sha256=np.frombuffer(train_digest, dtype=np.uint8),
                test_content_sha256=np.frombuffer(test_digest, dtype=np.uint8),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        output.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)
    evidence = _prediction_evidence(output)
    record = {
        "schema_version": 1,
        "dataset": dataset,
        "dataset_sha256": dataset_digest,
        "intervention": intervention,
        "causal_metadata": dict(causal_metadata),
        "prediction_file": "predictions.npz",
        "prediction_evidence": evidence,
    }
    _write_private_json(directory / "capture.json", record)
    return record


def make_causal_prediction_runner(external_runner: type) -> type:
    """Capture raw predictions while TabArena still retains them in memory."""

    class CausalPredictionRunner(external_runner):
        def __init__(
            self,
            *,
            causal_capture_root: str,
            causal_intervention: str,
            **kwargs: Any,
        ) -> None:
            root = Path(causal_capture_root)
            if not root.is_absolute():
                raise ValueError("causal capture root must be absolute")
            if causal_intervention not in FINGERPRINT_INTERVENTIONS:
                raise ValueError("unknown causal intervention")
            self.causal_capture_root = root
            self.causal_intervention = causal_intervention
            super().__init__(**kwargs)

        def post_evaluate(self, out: dict[str, Any]) -> dict[str, Any]:
            result = super().post_evaluate(out)
            probabilities = result.get("probabilities")
            if probabilities is None:
                raise RuntimeError("classification result lacks probabilities")
            if self.model is None or not hasattr(self.model, "causal_metadata"):
                raise RuntimeError("causal runner lacks intervention metadata")
            task = result.get("task_metadata")
            if not isinstance(task, Mapping) or not isinstance(task.get("name"), str):
                raise RuntimeError("causal runner lacks task metadata")
            train_X, train_y, test_X, test_y = self._train_test_split()
            capture_causal_prediction(
                probabilities,
                train_features=train_X,
                train_targets=train_y,
                test_features=test_X,
                test_targets=test_y,
                dataset=task["name"],
                intervention=self.causal_intervention,
                causal_metadata=self.model.causal_metadata(),
                capture_root=self.causal_capture_root,
            )
            return result

    CausalPredictionRunner.__name__ = "CausalPredictionRunner"
    CausalPredictionRunner.__qualname__ = "CausalPredictionRunner"
    return CausalPredictionRunner


def load_and_validate_captures(
    capture_root: Path,
    *,
    roster: Sequence[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Load the 4-way capture grid and verify all non-treatment evidence aligns."""
    if len(roster) < 1 or len(set(roster)) != len(roster):
        raise ValueError("causal roster must contain unique dataset names")
    captures: dict[str, dict[str, dict[str, Any]]] = {}
    for intervention in FINGERPRINT_INTERVENTIONS:
        by_dataset: dict[str, dict[str, Any]] = {}
        for dataset in roster:
            digest = hashlib.sha256(dataset.encode("utf-8")).hexdigest()
            directory = capture_root / intervention / digest
            record_path = directory / "capture.json"
            prediction_path = directory / "predictions.npz"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if (
                record.get("dataset") != dataset
                or record.get("dataset_sha256") != digest
                or record.get("intervention") != intervention
                or record.get("prediction_file") != "predictions.npz"
            ):
                raise ValueError("causal capture identity mismatch")
            observed = _prediction_evidence(prediction_path)
            if record.get("prediction_evidence") != observed:
                raise ValueError("causal prediction evidence changed after capture")
            metadata = record.get("causal_metadata")
            if not isinstance(metadata, Mapping) or metadata.get("intervention") != intervention:
                raise ValueError("causal intervention metadata mismatch")
            if intervention == "permuted":
                permutation = metadata.get("permutation")
                if not isinstance(permutation, Mapping):
                    raise ValueError("permuted condition lacks permutation evidence")
                token_count = permutation.get("token_count")
                if token_count == 1:
                    if (
                        permutation.get("effective") is not False
                        or permutation.get("fixed_point_count") != 1
                        or permutation.get("degenerate_reason")
                        != "single_feature_token_has_no_derangement"
                    ):
                        raise ValueError("H=1 permutation is not explicitly degenerate")
                elif (
                    not isinstance(token_count, int)
                    or token_count < 2
                    or permutation.get("effective") is not True
                    or permutation.get("fixed_point_count") != 0
                    or permutation.get("degenerate_reason") is not None
                ):
                    raise ValueError("permuted condition is not a fixed-point-free derangement")
            elif metadata.get("permutation") is not None:
                raise ValueError("non-permuted condition contains permutation evidence")
            by_dataset[dataset] = record
        captures[intervention] = by_dataset

    for dataset in roster:
        reference = captures["correct"][dataset]["prediction_evidence"]
        reference_tokens = captures["correct"][dataset]["causal_metadata"].get(
            "feature_token_count"
        )
        for intervention in FINGERPRINT_INTERVENTIONS[1:]:
            evidence = captures[intervention][dataset]["prediction_evidence"]
            mismatched = [
                field for field in ALIGNMENT_FIELDS if evidence.get(field) != reference.get(field)
            ]
            if mismatched:
                raise ValueError(
                    f"causal captures are misaligned for {dataset}: {mismatched}"
                )
            token_count = captures[intervention][dataset]["causal_metadata"].get(
                "feature_token_count"
            )
            if token_count != reference_tokens:
                raise ValueError("feature-token count differs across causal conditions")
    return captures


def exact_sign_test(left_wins: int, right_wins: int) -> float:
    if min(left_wins, right_wins) < 0:
        raise ValueError("win counts must be non-negative")
    non_ties = left_wins + right_wins
    if not non_ties:
        return 1.0
    smaller = min(left_wins, right_wins)
    tail = sum(math.comb(non_ties, value) for value in range(smaller + 1))
    return float(min(1.0, 2.0 * tail / (2**non_ties)))


def aggregate_causal_results(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    roster: Sequence[str],
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare correct against each intervention without mixing metric scales."""
    from pe_mechanism.statistics import paired_bootstrap_ci

    if len(roster) < 1 or len(set(roster)) != len(roster):
        raise ValueError("causal roster must contain unique dataset names")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    for dataset in roster:
        metrics = set()
        for intervention in FINGERPRINT_INTERVENTIONS:
            row = rows.get((intervention, dataset))
            if not isinstance(row, Mapping):
                raise ValueError("causal result grid is incomplete")
            metric = row.get("metric")
            error = row.get("metric_error")
            if not isinstance(metric, str) or not metric:
                raise ValueError("causal result metric is invalid")
            if isinstance(error, bool):
                raise ValueError("causal metric error is invalid")
            try:
                numeric_error = float(error)
            except (TypeError, ValueError) as cause:
                raise ValueError("causal metric error is invalid") from cause
            if not math.isfinite(numeric_error):
                raise ValueError("causal metric error is invalid")
            metrics.add(metric)
        if len(metrics) != 1:
            raise ValueError("metric differs across causal conditions")

    interventions = FINGERPRINT_INTERVENTIONS[1:]
    overall: list[dict[str, Any]] = []
    for intervention in interventions:
        deltas = np.asarray(
            [
                float(rows[(intervention, dataset)]["metric_error"])
                - float(rows[("correct", dataset)]["metric_error"])
                for dataset in roster
            ],
            dtype=np.float64,
        )
        correct_wins = int(np.count_nonzero(deltas > 0.0))
        intervention_wins = int(np.count_nonzero(deltas < 0.0))
        overall.append(
            {
                "left_arm": "correct",
                "right_arm": intervention,
                "dataset_count": len(roster),
                "left_wins": correct_wins,
                "right_wins": intervention_wins,
                "ties": int(np.count_nonzero(deltas == 0.0)),
                "two_sided_exact_sign_test_p": exact_sign_test(
                    correct_wins, intervention_wins
                ),
                "scale_free": True,
            }
        )

    metric_groups: dict[str, Any] = {}
    metrics = sorted({str(rows[("correct", dataset)]["metric"]) for dataset in roster})
    for metric_index, metric in enumerate(metrics):
        metric_roster = tuple(
            dataset
            for dataset in roster
            if rows[("correct", dataset)]["metric"] == metric
        )
        comparisons = []
        for intervention_index, intervention in enumerate(interventions, start=1):
            deltas = np.asarray(
                [
                    float(rows[(intervention, dataset)]["metric_error"])
                    - float(rows[("correct", dataset)]["metric_error"])
                    for dataset in metric_roster
                ],
                dtype=np.float64,
            )
            low, high = paired_bootstrap_ci(
                deltas,
                n_resamples=n_resamples,
                seed=seed + metric_index * 100 + intervention_index,
            )
            correct_wins = int(np.count_nonzero(deltas > 0.0))
            intervention_wins = int(np.count_nonzero(deltas < 0.0))
            comparisons.append(
                {
                    "left_arm": "correct",
                    "right_arm": intervention,
                    "effect": "intervention_metric_error_minus_correct_metric_error",
                    "positive_means_correct_is_better": True,
                    "dataset_count": len(metric_roster),
                    "mean_raw_metric_error_delta": float(deltas.mean()),
                    "median_raw_metric_error_delta": float(np.median(deltas)),
                    "paired_bootstrap_95ci": [low, high],
                    "correct_wins": correct_wins,
                    "intervention_wins": intervention_wins,
                    "ties": int(np.count_nonzero(deltas == 0.0)),
                    "two_sided_exact_sign_test_p": exact_sign_test(
                        correct_wins, intervention_wins
                    ),
                }
            )
        metric_groups[metric] = {
            "dataset_count": len(metric_roster),
            "raw_metric_error_delta_correct_vs_interventions": comparisons,
        }
    return {
        "overall_scale_free_correct_vs_interventions": overall,
        "metric_groups": metric_groups,
    }
