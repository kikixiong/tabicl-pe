"""Strict same-step TabICL evaluation on the TabArena v0.1 classification roster.

This workflow deliberately avoids TabArena leaderboard caches.  It runs one fixed
split for every official classification task and compares a content-verified
RoPE/No-PE pilot pair with the released TabICL reference under the same inference
budget.  The pilot pair remains exploratory and can never be promoted to formal
three-arm evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import gc
import gzip
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import io
import json
import math
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .provenance import (
    GitEvidence,
    RunTransaction,
    VerifiedFile,
    VerifiedRunContext,
    load_verified_json_config,
    manifest_from_verified_inputs,
    verify_configured_run_inputs,
    verify_git_tree,
)
from .statistics import paired_bootstrap_ci, paired_sign_flip_p_value


_TOP_LEVEL_FIELDS = {
    "schema_version",
    "study",
    "checkpoints",
    "benchmark",
    "provenance",
}
_STUDY_FIELDS = {
    "comparison_step",
    "seed",
    "checkpoint_study",
    "formal_eligible",
    "task_subset",
    "problem_types",
    "device",
    "n_estimators",
    "augmentation",
    "ensemble_size",
    "cache_mode",
    "debug_mode",
    "bootstrap_resamples",
    "classifier_options",
}
_CHECKPOINT_FIELDS = {
    "pair_manifest_path",
    "expected_pair_manifest_sha256",
    "pair_checksums_path",
    "expected_pair_checksums_sha256",
    "none_path",
    "expected_none_sha256",
    "released_path",
    "expected_released_sha256",
}
_BENCHMARK_FIELDS = {
    "tabarena_code_root",
    "expected_tabarena_code_sha",
    "openml_cache_root",
    "suite_version",
    "expected_task_count",
    "expected_result_count",
}
_PROBLEM_TYPES = ("binary", "multiclass")
_ARM_ORDER = ("rope", "none", "released")
_ROPE_ONLY_STATE_KEYS = frozenset({"row_interactor.tf_row.rope.freqs"})
_SHA256 = frozenset("0123456789abcdef")
_CLASSIFIER_FIELDS = {
    "norm_methods",
    "feat_shuffle_method",
    "class_shuffle_method",
    "outlier_threshold",
    "softmax_temperature",
    "average_logits",
    "support_many_classes",
    "batch_size",
    "kv_cache",
    "use_amp",
    "use_fa3",
    "offload_mode",
    "verbose",
}
_FIXED_CLASSIFIER_OPTIONS = {
    "norm_methods": ["none", "power"],
    "feat_shuffle_method": "latin",
    "class_shuffle_method": "shift",
    "outlier_threshold": 4.0,
    "softmax_temperature": 0.9,
    "average_logits": True,
    "support_many_classes": True,
    "batch_size": 8,
    "kv_cache": False,
    "use_amp": "auto",
    "use_fa3": False,
    "offload_mode": "auto",
    "verbose": False,
}
_RESULT_FIELDS = {
    "experiment_metadata",
    "framework",
    "memory_usage",
    "metric",
    "metric_error",
    "problem_type",
    "simulation_artifacts",
    "task_metadata",
    "time_infer_s",
    "time_train_s",
}
_TASK_FIELDS = {"tid", "name", "fold", "repeat", "sample", "split_idx"}
_EXPECTED_COMPARISON_STEP = 250_000
_EXPECTED_SEED = 42
_EXPECTED_TASK_COUNT = 38
_EXPECTED_RESULT_COUNT = 114
_EXPECTED_PAIR_MANIFEST_SHA256 = (
    "14a8c5069f1b6609ac5020caf9c544d98fb276b910028203912379e47b483447"
)
_EXPECTED_ROSTER_FILE_SHA256 = (
    "3c26133c8b986aba530624b5c7a5dee42e4b29938bc09afa04f76b6462c4adf1"
)
_EXPECTED_RELEASED_SHA256 = (
    "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
)
_EXPECTED_RELEASED_SIZE_BYTES = 110_368_038


@dataclass(frozen=True)
class EvaluationSpec:
    comparison_step: int
    seed: int
    device: str
    bootstrap_resamples: int
    classifier_options: Mapping[str, Any]
    pair_manifest_path: Path
    pair_manifest_sha256: str
    pair_checksums_path: Path
    pair_checksums_sha256: str
    none_path: Path
    none_sha256: str
    released_path: Path
    released_sha256: str
    tabarena_code_root: Path
    tabarena_code_sha: str
    openml_cache_root: Path
    expected_task_count: int
    expected_result_count: int


def _exact_object(
    value: object,
    *,
    label: str,
    fields: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        raise ValueError(
            f"{label} fields mismatch: missing={missing}, unknown={unknown}"
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _git_sha(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in (40, 64)
        or any(character not in _SHA256 for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase Git object id")
    return value


def _absolute_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an absolute path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return path


def _parse_config(data: Mapping[str, Any]) -> EvaluationSpec:
    config = _exact_object(data, label="TabArena config", fields=_TOP_LEVEL_FIELDS)
    if config["schema_version"] != 1:
        raise ValueError("TabArena config schema_version must be 1")
    study = _exact_object(config["study"], label="study", fields=_STUDY_FIELDS)
    checkpoints = _exact_object(
        config["checkpoints"], label="checkpoints", fields=_CHECKPOINT_FIELDS
    )
    benchmark = _exact_object(
        config["benchmark"], label="benchmark", fields=_BENCHMARK_FIELDS
    )
    _ = _exact_object(
        config["provenance"],
        label="provenance",
        fields={
            "model_family",
            "model_revision",
            "condition",
            "sites",
            "checkpoint_path",
            "dataset_manifest_path",
            "training_code_root",
            "model_code_root",
            "analysis_code_root",
            "expected_checkpoint_sha256",
            "expected_dataset_manifest_sha256",
            "expected_training_code_sha",
            "expected_model_code_sha",
            "expected_analysis_code_sha",
            "allow_exploratory_legacy",
        },
    )

    comparison_step = _positive_int(study["comparison_step"], name="comparison_step")
    seed = _positive_int(study["seed"], name="seed")
    if comparison_step != _EXPECTED_COMPARISON_STEP or seed != _EXPECTED_SEED:
        raise ValueError("this frozen comparison requires step 250000 and seed 42")
    if study["checkpoint_study"] != "exploratory_same_step_pilot":
        raise ValueError("checkpoint_study must be exploratory_same_step_pilot")
    if study["formal_eligible"] is not False:
        raise ValueError(
            "same-step pilot TabArena evaluation must remain formal_eligible=false"
        )
    if study["task_subset"] != "lite":
        raise ValueError("task_subset must be the single-split lite protocol")
    if study["problem_types"] != list(_PROBLEM_TYPES):
        raise ValueError("problem_types must be exactly ['binary', 'multiclass']")
    if study["device"] != "cuda":
        raise ValueError("the scheduled TabArena comparison requires device='cuda'")
    if study["n_estimators"] != 1:
        raise ValueError(
            "n_estimators must be exactly 1 for the fixed inference budget"
        )
    if study["augmentation"] != "none" or study["ensemble_size"] != 1:
        raise ValueError("the baseline comparison forbids inference augmentation")
    if study["cache_mode"] != "ignore" or study["debug_mode"] is not True:
        raise ValueError(
            "the run must use a fresh cache and in-process debug execution"
        )
    classifier_options = _exact_object(
        study["classifier_options"],
        label="classifier_options",
        fields=_CLASSIFIER_FIELDS,
    )
    if dict(classifier_options) != _FIXED_CLASSIFIER_OPTIONS:
        raise ValueError("classifier_options differ from the frozen inference contract")
    bootstrap_resamples = _positive_int(
        study["bootstrap_resamples"], name="bootstrap_resamples"
    )
    if benchmark["suite_version"] != "v0.1":
        raise ValueError("suite_version must be v0.1")
    expected_task_count = _positive_int(
        benchmark["expected_task_count"], name="expected_task_count"
    )
    expected_result_count = _positive_int(
        benchmark["expected_result_count"], name="expected_result_count"
    )
    if expected_result_count != expected_task_count * len(_ARM_ORDER):
        raise ValueError(
            "expected_result_count must equal three times expected_task_count"
        )
    if (
        expected_task_count != _EXPECTED_TASK_COUNT
        or expected_result_count != _EXPECTED_RESULT_COUNT
    ):
        raise ValueError(
            "the frozen TabArena classification matrix must be exactly 38x3"
        )

    provenance = config["provenance"]
    expected_revision = f"step-{comparison_step}-plus-released"
    if (
        provenance["model_family"] != "tabicl-v2"
        or provenance["model_revision"] != expected_revision
        or provenance["condition"] != "tabarena-rope-none-released"
        or provenance["sites"] != ["tabarena-v0.1-classification"]
        or provenance["allow_exploratory_legacy"] is not False
    ):
        raise ValueError("provenance labels do not match the fixed TabArena study")

    pair_manifest_sha256 = _sha256(
        checkpoints["expected_pair_manifest_sha256"],
        name="expected_pair_manifest_sha256",
    )
    released_sha256 = _sha256(
        checkpoints["expected_released_sha256"],
        name="expected_released_sha256",
    )
    if pair_manifest_sha256 != _EXPECTED_PAIR_MANIFEST_SHA256:
        raise ValueError("checkpoint pair is not the frozen step-250000 pilot pair")
    if released_sha256 != _EXPECTED_RELEASED_SHA256:
        raise ValueError(
            "released checkpoint digest is not the frozen TabICL reference"
        )

    return EvaluationSpec(
        comparison_step=comparison_step,
        seed=seed,
        device=str(study["device"]),
        bootstrap_resamples=bootstrap_resamples,
        classifier_options=dict(classifier_options),
        pair_manifest_path=_absolute_path(
            checkpoints["pair_manifest_path"], name="pair_manifest_path"
        ),
        pair_manifest_sha256=pair_manifest_sha256,
        pair_checksums_path=_absolute_path(
            checkpoints["pair_checksums_path"], name="pair_checksums_path"
        ),
        pair_checksums_sha256=_sha256(
            checkpoints["expected_pair_checksums_sha256"],
            name="expected_pair_checksums_sha256",
        ),
        none_path=_absolute_path(checkpoints["none_path"], name="none_path"),
        none_sha256=_sha256(
            checkpoints["expected_none_sha256"], name="expected_none_sha256"
        ),
        released_path=_absolute_path(
            checkpoints["released_path"], name="released_path"
        ),
        released_sha256=released_sha256,
        tabarena_code_root=_absolute_path(
            benchmark["tabarena_code_root"], name="tabarena_code_root"
        ),
        tabarena_code_sha=_git_sha(
            benchmark["expected_tabarena_code_sha"],
            name="expected_tabarena_code_sha",
        ),
        openml_cache_root=_absolute_path(
            benchmark["openml_cache_root"], name="openml_cache_root"
        ),
        expected_task_count=expected_task_count,
        expected_result_count=expected_result_count,
    )


def _json_object(file: VerifiedFile, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(file.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _validate_roster(
    roster_file: VerifiedFile,
    *,
    benchmark_sha: str,
    expected_count: int,
) -> tuple[str, ...]:
    if roster_file.digest.sha256 != _EXPECTED_ROSTER_FILE_SHA256:
        raise ValueError("dataset manifest is not the frozen public TabArena roster")
    payload = _exact_object(
        _json_object(roster_file, label="TabArena roster"),
        label="TabArena roster",
        fields={
            "schema_version",
            "kind",
            "suite",
            "version",
            "task_type",
            "count",
            "names",
            "roster_hash_algorithm",
            "roster_sha256",
            "source_commit",
            "source_url",
        },
    )
    if (
        payload["schema_version"] != 1
        or payload["kind"] != "dataset_roster"
        or payload["suite"] != "TabArena"
        or payload["version"] != "v0.1"
        or payload["task_type"] != "classification"
        or payload["roster_hash_algorithm"] != "sha256_newline_join_sorted_names"
        or payload["source_commit"] != benchmark_sha
    ):
        raise ValueError("TabArena roster metadata differs from the frozen suite")
    names = payload["names"]
    if (
        not isinstance(names, list)
        or not all(isinstance(name, str) and name for name in names)
        or len(names) != len(set(names))
        or names != sorted(names)
        or payload["count"] != len(names)
        or len(names) != expected_count
    ):
        raise ValueError(
            "TabArena classification roster is incomplete or non-canonical"
        )
    observed_hash = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    if observed_hash != payload["roster_sha256"]:
        raise ValueError("TabArena roster content hash does not match its names")
    return tuple(names)


def _load_checkpoint_payload(file: VerifiedFile, *, label: str) -> Mapping[str, Any]:
    import torch

    try:
        payload = torch.load(
            io.BytesIO(file.read_bytes()), map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise ValueError(
            f"{label} is not a loadable weights-only checkpoint"
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain one checkpoint mapping")
    return payload


def _checkpoint_contract(
    file: VerifiedFile,
    *,
    arm: str,
    step: int,
) -> tuple[dict[str, Any], dict[str, tuple[tuple[int, ...], str]]]:
    import torch

    payload = _load_checkpoint_payload(file, label=f"{arm} checkpoint")
    config = payload.get("config")
    state_dict = payload.get("state_dict")
    if (
        not isinstance(config, dict)
        or not isinstance(state_dict, dict)
        or not state_dict
        or payload.get("curr_step") != step
        or config.get("row_identity_mode") != arm
    ):
        raise ValueError(f"{arm} checkpoint violates the matched-step contract")
    signature: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, tensor in state_dict.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{arm} state_dict must be tensor-only")
        signature[name] = (tuple(tensor.shape), str(tensor.dtype))
    return dict(config), signature


def _validate_checkpoint_pair(
    context: VerifiedRunContext,
    spec: EvaluationSpec,
) -> dict[str, dict[str, int | str]]:
    manifest_file = context.additional_file("pair_manifest")
    checksums_file = context.additional_file("pair_checksums")
    none_file = context.additional_file("none_checkpoint")
    released_file = context.additional_file("released_checkpoint")
    rope_file = context.inputs.checkpoint
    payload = _json_object(manifest_file, label="matched checkpoint manifest")

    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "exploratory_same_step_pilot_checkpoint_pair"
        or payload.get("formal_eligible") is not False
        or payload.get("comparison_step") != spec.comparison_step
        or payload.get("seed") != spec.seed
        or payload.get("pilot_source_commit") != context.inputs.model_code.head_sha
        or context.inputs.training_code.head_sha != context.inputs.model_code.head_sha
    ):
        raise ValueError(
            "matched checkpoint manifest violates the fixed pilot contract"
        )
    arms = payload.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {"rope", "none"}:
        raise ValueError(
            "matched checkpoint manifest must contain exactly RoPE and No-PE"
        )
    files = {"rope": rope_file, "none": none_file}
    for arm, file in files.items():
        entry = arms[arm]
        if not isinstance(entry, Mapping):
            raise ValueError(f"invalid checkpoint manifest entry: {arm}")
        filename = entry.get("snapshot_checkpoint")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or file.path.name != filename
            or entry.get("sha256") != file.digest.sha256
            or entry.get("bytes") != file.digest.size_bytes
            or entry.get("curr_step") != spec.comparison_step
            or entry.get("row_identity_mode") != arm
        ):
            raise ValueError(f"checkpoint manifest entry does not match {arm} bytes")

    expected_checksum_lines = {
        f"{files[arm].digest.sha256}  {files[arm].path.name}"
        for arm in ("none", "rope")
    }
    expected_checksum_lines.add(
        f"{manifest_file.digest.sha256}  {manifest_file.path.name}"
    )
    try:
        observed_checksum_lines = set(
            checksums_file.read_bytes().decode("ascii").splitlines()
        )
    except UnicodeDecodeError as error:
        raise ValueError("checkpoint checksum marker must be ASCII") from error
    if observed_checksum_lines != expected_checksum_lines:
        raise ValueError("checkpoint checksum marker does not match the verified pair")

    configs: dict[str, dict[str, Any]] = {}
    signatures: dict[str, dict[str, tuple[tuple[int, ...], str]]] = {}
    for arm in ("none", "rope"):
        configs[arm], signatures[arm] = _checkpoint_contract(
            files[arm], arm=arm, step=spec.comparison_step
        )
        if arms[arm].get("state_dict_tensor_count") != len(signatures[arm]):
            raise ValueError(f"checkpoint tensor count differs for {arm}")
    comparable = {}
    for arm, config in configs.items():
        value = dict(config)
        value.pop("row_identity_mode")
        comparable[arm] = value
    if comparable["none"] != comparable["rope"]:
        raise ValueError("RoPE and No-PE configs differ beyond row_identity_mode")
    none_state = signatures["none"]
    rope_state = signatures["rope"]
    if (
        set(none_state) - set(rope_state)
        or set(rope_state) - set(none_state) != _ROPE_ONLY_STATE_KEYS
    ):
        raise ValueError(
            "RoPE and No-PE state keys differ beyond the RoPE frequency buffer"
        )
    for name in set(none_state) & set(rope_state):
        if none_state[name] != rope_state[name]:
            raise ValueError("RoPE and No-PE tensor signatures differ")
    embed_dim = comparable["rope"].get("embed_dim")
    row_nhead = comparable["rope"].get("row_nhead")
    if (
        isinstance(embed_dim, bool)
        or not isinstance(embed_dim, int)
        or isinstance(row_nhead, bool)
        or not isinstance(row_nhead, int)
        or embed_dim <= 0
        or row_nhead <= 0
        or embed_dim % row_nhead
        or (embed_dim // row_nhead) % 2
    ):
        raise ValueError("checkpoint config cannot derive the RoPE frequency shape")
    expected_frequency = ((embed_dim // row_nhead // 2,), "torch.float32")
    if rope_state["row_interactor.tf_row.rope.freqs"] != expected_frequency:
        raise ValueError("RoPE frequency buffer shape differs from the model config")

    released = _load_checkpoint_payload(released_file, label="released checkpoint")
    released_config = released.get("config")
    released_state = released.get("state_dict")
    if (
        not isinstance(released_config, dict)
        or not isinstance(released_state, dict)
        or not released_state
        or "row_interactor.tf_row.rope.freqs" not in released_state
        or released_file.digest.sha256 != _EXPECTED_RELEASED_SHA256
        or released_file.digest.size_bytes != _EXPECTED_RELEASED_SIZE_BYTES
    ):
        raise ValueError(
            "released reference is not a compatible RoPE TabICL checkpoint"
        )
    return {
        "rope": {
            "sha256": rope_file.digest.sha256,
            "size_bytes": rope_file.digest.size_bytes,
        },
        "none": {
            "sha256": none_file.digest.sha256,
            "size_bytes": none_file.digest.size_bytes,
        },
        "released": {
            "sha256": released_file.digest.sha256,
            "size_bytes": released_file.digest.size_bytes,
        },
    }


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _require_module_under(module: Any, root: Path, *, name: str) -> None:
    raw = getattr(module, "__file__", None)
    if not isinstance(raw, str):
        raise RuntimeError(f"{name} import has no source file")
    source = Path(raw).resolve(strict=True)
    if not _is_within(source, root.resolve(strict=True)):
        raise RuntimeError(f"{name} imported outside its verified source checkout")


def _import_runtime(tabarena_root: Path, model_root: Path) -> Mapping[str, Any]:
    for prefix in ("tabarena", "tabicl"):
        if any(name == prefix or name.startswith(f"{prefix}.") for name in sys.modules):
            raise RuntimeError(f"{prefix} was imported before its verified source path")
    tabarena_src = (tabarena_root / "packages" / "tabarena" / "src").resolve(
        strict=True
    )
    model_src = (model_root / "src").resolve(strict=True)
    sys.path.insert(0, str(model_src))
    sys.path.insert(0, str(tabarena_src))
    importlib.invalidate_caches()

    tabarena = importlib.import_module("tabarena")
    tabicl = importlib.import_module("tabicl")
    _require_module_under(tabarena, tabarena_src, name="tabarena")
    _require_module_under(tabicl, model_src, name="tabicl")
    return {
        "tabarena": tabarena,
        "tabicl": tabicl,
        "ExternalSystemModel": importlib.import_module(
            "tabarena.benchmark.exec_models"
        ).ExternalSystemModel,
        "TabArenaV0pt1ExperimentBundle": importlib.import_module(
            "tabarena.benchmark.experiment"
        ).TabArenaV0pt1ExperimentBundle,
        "CacheConfig": importlib.import_module("tabarena.caching").CacheConfig,
        "TabArenaContext": importlib.import_module("tabarena.contexts").TabArenaContext,
        "SystemConfigGenerator": importlib.import_module(
            "tabarena.utils.config_utils"
        ).SystemConfigGenerator,
    }


def _make_system_model(external_system_model: type) -> type:
    class FixedTabICLSystem(external_system_model):
        def __init__(
            self,
            *,
            checkpoint: str,
            arm: str,
            device: str,
            n_estimators: int,
            seed: int,
            classifier_options: Mapping[str, Any],
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.checkpoint = checkpoint
            self.arm = arm
            self.device = device
            self.n_estimators = n_estimators
            self.seed = seed
            self.classifier_options = dict(classifier_options)
            self.model: Any | None = None

        def _fit_system(
            self,
            X: Any,
            y: Any,
            *,
            target_name: str,
            problem_type: str,
            eval_metric: Any,
            validation_metadata: Any,
            num_cpus: int | None,
            num_gpus: int | None,
            memory_limit: float | None,
            time_limit: float | None,
            random_state: int | None,
        ) -> "FixedTabICLSystem":
            del target_name, eval_metric, validation_metadata, num_gpus, memory_limit
            del time_limit, random_state
            if problem_type not in _PROBLEM_TYPES:
                raise ValueError(
                    "fixed TabICL TabArena runner supports classification only"
                )
            from tabicl import TabICLClassifier

            checkpoint = Path(self.checkpoint)
            if checkpoint.is_symlink() or not checkpoint.is_file():
                raise FileNotFoundError("verified TabICL checkpoint disappeared")
            self.model = TabICLClassifier(
                model_path=str(checkpoint),
                allow_auto_download=False,
                device=self.device,
                n_estimators=self.n_estimators,
                random_state=self.seed,
                n_jobs=max(1, int(num_cpus or 1)),
                **self.classifier_options,
            )
            self.model.fit(X, y)
            return self

        def _predict(self, X: Any) -> Any:
            if self.model is None:
                raise RuntimeError("TabICL system is not fitted")
            import pandas as pd

            return pd.Series(self.model.predict(X), index=X.index)

        def _predict_proba(self, X: Any) -> Any:
            if self.model is None:
                raise RuntimeError("TabICL system is not fitted")
            import pandas as pd

            values = self.model.predict_proba(X)
            return pd.DataFrame(values, index=X.index, columns=self.model.classes_)

        def cleanup(self) -> None:
            self.model = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:  # pragma: no cover - torch is a declared dependency
                pass

    FixedTabICLSystem.__name__ = "FixedTabICLSystem"
    FixedTabICLSystem.__qualname__ = "FixedTabICLSystem"
    return FixedTabICLSystem


def _framework_roots(step: int) -> dict[str, str]:
    return {
        "rope": f"TabICL_RoPE_Step{step}",
        "none": f"TabICL_NoPE_Step{step}",
        "released": "TabICL_Released_Reference",
    }


def _validate_context_tasks(
    context: Any, roster: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    available = set(context.task_metadata_collection.dataset_names())
    missing = sorted(set(roster) - available)
    if missing:
        raise ValueError(f"TabArena source checkout lacks roster tasks: {missing}")
    metadata = context.task_metadata
    selected = metadata[metadata["dataset_name"].isin(roster)]
    observed_names = set(selected["dataset_name"].astype(str))
    observed_types = set(selected["problem_type"].astype(str))
    if observed_names != set(roster) or not observed_types.issubset(
        set(_PROBLEM_TYPES)
    ):
        raise ValueError(
            "TabArena task metadata differs from the classification roster"
        )
    expected: dict[str, dict[str, Any]] = {}
    for row in selected.to_dict(orient="records"):
        name = str(row["dataset_name"])
        expected[name] = {
            "task_id": int(row["tid"]),
            "problem_type": str(row["problem_type"]),
            "metric": str(row["eval_metric"]),
        }
    if set(expected) != set(roster):
        raise ValueError("TabArena task metadata is not one-to-one with the roster")
    return expected


def _execute_tabarena(
    *,
    runtime: Mapping[str, Any],
    spec: EvaluationSpec,
    context: VerifiedRunContext,
    roster: tuple[str, ...],
    results_root: Path,
    scratch_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, dict[str, Any]]]:
    model_cls = _make_system_model(runtime["ExternalSystemModel"])
    checkpoints = {
        "rope": context.inputs.checkpoint.path,
        "none": context.additional_file("none_checkpoint").path,
        "released": context.additional_file("released_checkpoint").path,
    }
    framework_to_arm: dict[str, str] = {}
    experiments = []
    for arm in _ARM_ORDER:
        generator = runtime["SystemConfigGenerator"](
            model_cls=model_cls,
            name=_framework_roots(spec.comparison_step)[arm],
            manual_configs=[
                {
                    "checkpoint": str(checkpoints[arm]),
                    "arm": arm,
                    "device": spec.device,
                    "n_estimators": 1,
                    "seed": spec.seed,
                    "classifier_options": dict(spec.classifier_options),
                }
            ],
        )
        built = runtime["TabArenaV0pt1ExperimentBundle"](
            models=[(generator, 0)], system_experiments=True
        ).build_experiments()
        if len(built) != 1 or built[0].name in framework_to_arm:
            raise RuntimeError(
                "TabArena did not construct one unique framework per arm"
            )
        framework_to_arm[built[0].name] = arm
        experiments.extend(built)

    cache = runtime["CacheConfig"](
        openml=spec.openml_cache_root,
        huggingface=scratch_root / "huggingface",
        data_foundry=scratch_root / "data-foundry",
        tabarena=scratch_root / "tabarena",
        results=results_root,
        apply_on_run=True,
        scope_openml=True,
    )
    arena = runtime["TabArenaContext"](methods=[], backend="native", cache_config=cache)
    expected_tasks = _validate_context_tasks(arena, roster)
    jobs = arena.build_jobs(
        experiments,
        subset=["lite"],
        dataset_names=list(roster),
        problem_types=list(_PROBLEM_TYPES),
    )
    if len(jobs) != spec.expected_result_count:
        raise RuntimeError(
            f"TabArena constructed {len(jobs)} jobs; expected {spec.expected_result_count}"
        )
    expected_job_keys = {
        (framework, dataset, 0, 0)
        for framework in framework_to_arm
        for dataset in roster
    }
    observed_job_keys = {
        (
            job.experiment.name,
            job.task.dataset,
            job.task.fold,
            job.task.repeat,
        )
        for job in jobs
    }
    if observed_job_keys != expected_job_keys or len(observed_job_keys) != len(jobs):
        raise RuntimeError("TabArena job matrix differs from the frozen 38x3 lite grid")
    results = arena.run_jobs(
        jobs,
        expname=results_root,
        register=False,
        cache_mode="ignore",
        debug_mode=True,
        raise_on_failure=True,
    )
    if not all(isinstance(item, dict) for item in results):
        raise RuntimeError("TabArena returned a non-dictionary result")
    return list(results), framework_to_arm, expected_tasks


def _finite_number(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        raise ValueError(
            f"{name} must be finite" + (" and non-negative" if nonnegative else "")
        )
    return number


def _normalize_result(
    result: Mapping[str, Any],
    *,
    framework_to_arm: Mapping[str, str],
) -> dict[str, Any]:
    missing = sorted(_RESULT_FIELDS - set(result))
    unknown = sorted(set(result) - _RESULT_FIELDS)
    if missing or unknown:
        raise ValueError(
            f"TabArena result fields mismatch: missing={missing}, unknown={unknown}"
        )
    if result["simulation_artifacts"] is not None:
        raise ValueError(
            "external-system TabArena result unexpectedly contains simulation artifacts"
        )
    framework = result["framework"]
    if not isinstance(framework, str) or framework not in framework_to_arm:
        raise ValueError("TabArena result has an unknown framework")
    task = result["task_metadata"]
    if not isinstance(task, Mapping):
        raise ValueError("TabArena result task_metadata must be an object")
    if set(task) != _TASK_FIELDS:
        raise ValueError("TabArena task_metadata fields differ from the frozen schema")
    if any(task[name] != 0 for name in ("fold", "repeat", "sample", "split_idx")):
        raise ValueError("TabArena result is not the frozen lite split")
    if isinstance(task["tid"], bool) or not isinstance(task["tid"], (int, np.integer)):
        raise ValueError("TabArena task id must be an integer")
    if not isinstance(task["name"], str) or not task["name"]:
        raise ValueError("TabArena task name must be a non-empty string")
    problem_type = result["problem_type"]
    metric = result["metric"]
    if problem_type not in _PROBLEM_TYPES or not isinstance(metric, str) or not metric:
        raise ValueError("TabArena result metric or problem type is invalid")
    return {
        "arm": framework_to_arm[framework],
        "framework": framework,
        "dataset": task["name"],
        "task_id": int(task["tid"]),
        "fold": 0,
        "repeat": 0,
        "sample": 0,
        "split_idx": 0,
        "problem_type": problem_type,
        "metric": metric,
        "metric_error": _finite_number(
            result["metric_error"], name="metric_error", nonnegative=True
        ),
        "time_train_s": _finite_number(
            result["time_train_s"], name="time_train_s", nonnegative=True
        ),
        "time_infer_s": _finite_number(
            result["time_infer_s"], name="time_infer_s", nonnegative=True
        ),
    }


def _normalize_results(
    results: Sequence[Mapping[str, Any]],
    *,
    framework_to_arm: Mapping[str, str],
    roster: tuple[str, ...],
    expected_count: int,
    expected_tasks: Mapping[str, Mapping[str, Any]],
    arm_order: Sequence[str] = _ARM_ORDER,
) -> list[dict[str, Any]]:
    arms = tuple(arm_order)
    if (
        len(arms) < 2
        or len(set(arms)) != len(arms)
        or not all(isinstance(arm, str) and arm for arm in arms)
    ):
        raise ValueError("TabArena arm order must contain distinct non-empty labels")
    if (
        len(framework_to_arm) != len(arms)
        or set(framework_to_arm.values()) != set(arms)
    ):
        raise ValueError("TabArena framework mapping differs from the arm order")
    if expected_count != len(roster) * len(arms):
        raise ValueError("TabArena expected result count differs from its arm grid")
    normalized = [
        _normalize_result(result, framework_to_arm=framework_to_arm)
        for result in results
    ]
    if len(normalized) != expected_count:
        raise ValueError(
            f"received {len(normalized)} results; expected {expected_count}"
        )
    keys = [(item["arm"], item["dataset"]) for item in normalized]
    if len(keys) != len(set(keys)):
        raise ValueError("TabArena results contain duplicate arm/dataset pairs")
    expected_keys = {(arm, dataset) for arm in arms for dataset in roster}
    missing = sorted(expected_keys - set(keys))
    extra = sorted(set(keys) - expected_keys)
    if missing or extra:
        raise ValueError(
            f"TabArena result matrix mismatch: missing={missing}, extra={extra}"
        )

    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for item in normalized:
        by_dataset.setdefault(str(item["dataset"]), []).append(item)
    for dataset, rows in by_dataset.items():
        if len({row["task_id"] for row in rows}) != 1:
            raise ValueError(f"task id differs across arms for {dataset}")
        if len({row["problem_type"] for row in rows}) != 1:
            raise ValueError(f"problem type differs across arms for {dataset}")
        if len({row["metric"] for row in rows}) != 1:
            raise ValueError(f"metric differs across arms for {dataset}")
        expected = expected_tasks[dataset]
        if any(
            row["task_id"] != expected["task_id"]
            or row["problem_type"] != expected["problem_type"]
            or row["metric"] != expected["metric"]
            for row in rows
        ):
            raise ValueError(
                f"result metadata differs from official TabArena metadata for {dataset}"
            )
        if expected["metric"] == "roc_auc" and any(
            not 0.0 <= row["metric_error"] <= 1.0 for row in rows
        ):
            raise ValueError(f"ROC-AUC error is outside [0, 1] for {dataset}")
    return sorted(normalized, key=lambda item: (str(item["dataset"]), str(item["arm"])))


def _load_cached_results(results_root: Path) -> list[Mapping[str, Any]]:
    if results_root.is_symlink() or not results_root.is_dir():
        raise ValueError("TabArena result cache must be a real directory")
    all_files = sorted(path for path in results_root.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in results_root.rglob("*")):
        raise ValueError("TabArena result cache must not contain symlinks")
    unexpected = [path for path in all_files if path.name != "results.pkl"]
    if unexpected:
        raise ValueError("TabArena result cache contains unexpected files")
    loaded: list[Mapping[str, Any]] = []
    for path in all_files:
        try:
            with path.open("rb") as probe:
                compressed = probe.read(2) == b"\x1f\x8b"
            opener = gzip.open if compressed else open
            with opener(path, "rb") as handle:
                item = pickle.load(handle)
        except Exception as error:
            raise ValueError(
                "fresh TabArena result cache contains an unreadable pickle"
            ) from error
        if not isinstance(item, Mapping):
            raise ValueError("fresh TabArena cache item is not a result object")
        loaded.append(item)
    return loaded


def _paired_comparison(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    roster: tuple[str, ...],
    left: str,
    right: str,
    seed: int,
    n_resamples: int,
) -> dict[str, Any]:
    left_errors = np.asarray(
        [rows[(left, dataset)]["metric_error"] for dataset in roster], dtype=np.float64
    )
    right_errors = np.asarray(
        [rows[(right, dataset)]["metric_error"] for dataset in roster], dtype=np.float64
    )
    improvement = right_errors - left_errors
    low, high = paired_bootstrap_ci(improvement, n_resamples=n_resamples, seed=seed)
    left_wins = int(np.count_nonzero(improvement > 0.0))
    right_wins = int(np.count_nonzero(improvement < 0.0))
    non_ties = left_wins + right_wins
    if non_ties:
        smaller = min(left_wins, right_wins)
        sign_tail = sum(math.comb(non_ties, value) for value in range(smaller + 1))
        two_sided_sign_p = min(1.0, 2.0 * sign_tail / (2**non_ties))
    else:
        two_sided_sign_p = 1.0
    return {
        "left_arm": left,
        "right_arm": right,
        "effect": "right_metric_error_minus_left_metric_error",
        "positive_means_left_is_better": True,
        "dataset_count": len(roster),
        "mean_metric_error_improvement": float(improvement.mean()),
        "median_metric_error_improvement": float(np.median(improvement)),
        "paired_bootstrap_95ci": [low, high],
        "left_wins": left_wins,
        "right_wins": right_wins,
        "ties": int(np.count_nonzero(improvement == 0.0)),
        "two_sided_exact_sign_test_p": float(two_sided_sign_p),
        "one_sided_paired_sign_flip_p": paired_sign_flip_p_value(
            improvement, n_resamples=n_resamples, seed=seed + 100_000
        ),
    }


def _scale_free_comparison(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    roster: tuple[str, ...],
    left: str,
    right: str,
) -> dict[str, Any]:
    left_wins = 0
    right_wins = 0
    ties = 0
    for dataset in roster:
        left_error = float(rows[(left, dataset)]["metric_error"])
        right_error = float(rows[(right, dataset)]["metric_error"])
        if left_error < right_error:
            left_wins += 1
        elif right_error < left_error:
            right_wins += 1
        else:
            ties += 1
    non_ties = left_wins + right_wins
    if non_ties:
        smaller = min(left_wins, right_wins)
        tail = sum(math.comb(non_ties, value) for value in range(smaller + 1))
        p_value = min(1.0, 2.0 * tail / (2**non_ties))
    else:
        p_value = 1.0
    return {
        "left_arm": left,
        "right_arm": right,
        "dataset_count": len(roster),
        "left_wins": left_wins,
        "right_wins": right_wins,
        "ties": ties,
        "two_sided_exact_sign_test_p": float(p_value),
        "scale_free": True,
    }


def _mean_ranks(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    roster: tuple[str, ...],
    arm_order: Sequence[str] = _ARM_ORDER,
) -> dict[str, float]:
    arms = tuple(arm_order)
    if (
        len(arms) < 2
        or len(set(arms)) != len(arms)
        or not all(isinstance(arm, str) and arm for arm in arms)
    ):
        raise ValueError("TabArena arm order must contain distinct non-empty labels")
    ranks = {arm: [] for arm in arms}
    for dataset in roster:
        ordered = sorted(
            ((float(rows[(arm, dataset)]["metric_error"]), arm) for arm in arms),
            key=lambda item: (item[0], item[1]),
        )
        start = 0
        while start < len(ordered):
            end = start + 1
            while end < len(ordered) and ordered[end][0] == ordered[start][0]:
                end += 1
            average_rank = ((start + 1) + end) / 2.0
            for _, arm in ordered[start:end]:
                ranks[arm].append(average_rank)
            start = end
    return {arm: float(np.mean(values)) for arm, values in ranks.items()}


def _aggregate_summary(
    normalized: Sequence[Mapping[str, Any]],
    *,
    roster: tuple[str, ...],
    spec: EvaluationSpec,
    framework_to_arm: Mapping[str, str],
    checkpoint_digests: Mapping[str, Mapping[str, int | str]],
    context: VerifiedRunContext,
    benchmark_git: GitEvidence,
) -> dict[str, Any]:
    rows = {(str(item["arm"]), str(item["dataset"])): item for item in normalized}
    datasets = []
    for dataset in roster:
        first = rows[("rope", dataset)]
        datasets.append(
            {
                "dataset": dataset,
                "task_id": first["task_id"],
                "problem_type": first["problem_type"],
                "metric": first["metric"],
                "arms": {
                    arm: {
                        "metric_error": rows[(arm, dataset)]["metric_error"],
                        "time_train_s": rows[(arm, dataset)]["time_train_s"],
                        "time_infer_s": rows[(arm, dataset)]["time_infer_s"],
                    }
                    for arm in _ARM_ORDER
                },
            }
        )
    framework_by_arm = {arm: framework for framework, arm in framework_to_arm.items()}
    metric_groups: dict[str, Any] = {}
    for metric in sorted({str(item["metric"]) for item in normalized}):
        metric_roster = tuple(
            dataset for dataset in roster if rows[("rope", dataset)]["metric"] == metric
        )
        metric_groups[metric] = {
            "dataset_count": len(metric_roster),
            "macro_mean_metric_error": {
                arm: float(
                    np.mean(
                        [
                            rows[(arm, dataset)]["metric_error"]
                            for dataset in metric_roster
                        ]
                    )
                )
                for arm in _ARM_ORDER
            },
            "paired_comparisons": [
                _paired_comparison(
                    rows,
                    roster=metric_roster,
                    left=left,
                    right=right,
                    seed=spec.seed + 10 * offset,
                    n_resamples=spec.bootstrap_resamples,
                )
                for offset, (left, right) in enumerate(
                    (("rope", "none"), ("rope", "released"), ("none", "released")),
                    start=1,
                )
            ],
        }
    return {
        "schema_version": 1,
        "study": "tabicl-step250k-tabarena-v0.1-classification",
        "evidence_scope": "exploratory_same_step_pilot_plus_released_reference",
        "formal_eligible": False,
        "formal_claim": "forbidden",
        "leaderboard_replication": False,
        "comparison_step": spec.comparison_step,
        "seed": spec.seed,
        "task_subset": "lite",
        "task_count": len(roster),
        "result_count": len(normalized),
        "inference_budget": {
            "n_estimators": 1,
            "augmentation": "none",
            "ensemble_size": 1,
            "classifier_options": dict(spec.classifier_options),
        },
        "coverage": {
            "classification_roster": "complete",
            "model_constraints_applied": False,
            "tasks_above_500_native_features": [
                "hiva_agnostic",
                "kddcup09_appetency",
            ],
        },
        "framework_by_arm": framework_by_arm,
        "checkpoint_digests": checkpoint_digests,
        "code_provenance": {
            "training_code_sha": context.inputs.training_code.head_sha,
            "model_code_sha": context.inputs.model_code.head_sha,
            "analysis_code_sha": context.inputs.analysis_code.head_sha,
            "benchmark_code_sha": benchmark_git.head_sha,
            "checkpoint_pair_manifest_sha256": context.additional_file(
                "pair_manifest"
            ).digest.sha256,
            "dataset_roster_sha256": context.inputs.dataset_manifest.digest.sha256,
        },
        "overall_mean_rank_lower_is_better": _mean_ranks(rows, roster=roster),
        "overall_scale_free_comparisons": [
            _scale_free_comparison(rows, roster=roster, left="rope", right="none"),
            _scale_free_comparison(rows, roster=roster, left="rope", right="released"),
            _scale_free_comparison(rows, roster=roster, left="none", right="released"),
        ],
        "metric_groups": metric_groups,
        "datasets": datasets,
        "statistical_note": (
            "Metric error is lower-is-better. Cross-metric overall comparisons use only "
            "per-dataset ranks and wins. Raw error means, bootstrap intervals, and paired "
            "sign-flip tests are reported only within one metric group; their p-values are "
            "unadjusted exploratory summaries. This is not a normalized TabArena leaderboard."
        ),
    }


def _publish_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def _archive_directory(source: Path, destination: Path) -> None:
    entries = sorted(source.rglob("*"), key=lambda path: path.as_posix())
    if any(path.is_symlink() for path in entries):
        raise ValueError("cannot archive a result cache containing symlinks")
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(mode="w", fileobj=compressed) as archive:
                for path in entries:
                    relative = Path("tabarena-results") / path.relative_to(source)
                    info = archive.gettarinfo(str(path), arcname=relative.as_posix())
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    if path.is_file():
                        with path.open("rb") as handle:
                            archive.addfile(info, handle)
                    elif path.is_dir():
                        archive.addfile(info)
                    else:
                        raise ValueError(
                            "result cache contains a non-file, non-directory entry"
                        )


def _runtime_summary(
    *,
    runtime: Mapping[str, Any],
    benchmark_sha: str,
    duration_seconds: float,
    result_count: int,
) -> dict[str, Any]:
    import pandas as pd
    import sklearn
    import torch

    try:
        driver = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.splitlines()[0]
    except (FileNotFoundError, IndexError, subprocess.CalledProcessError):
        driver = "unknown"

    return {
        "schema_version": 1,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "openml": importlib_metadata.version("openml"),
        "autogluon_core": importlib_metadata.version("autogluon.core"),
        "torch": torch.__version__,
        "tabarena": str(getattr(runtime["tabarena"], "__version__", "unknown")),
        "tabicl": str(getattr(runtime["tabicl"], "__version__", "unknown")),
        "benchmark_code_sha": benchmark_sha,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_device_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_driver": driver,
        "duration_seconds": float(duration_seconds),
        "result_count": result_count,
    }


def run_evaluation(config_path: Path, output_dir: Path) -> dict[str, Any]:
    configuration = load_verified_json_config(config_path)
    spec = _parse_config(configuration.data)
    context = verify_configured_run_inputs(
        configuration,
        command="tabarena-evaluate",
        seed=spec.seed,
        additional_input_paths={
            "none_checkpoint": spec.none_path,
            "pair_checksums": spec.pair_checksums_path,
            "pair_manifest": spec.pair_manifest_path,
            "released_checkpoint": spec.released_path,
        },
        expected_additional_sha256={
            "none_checkpoint": spec.none_sha256,
            "pair_checksums": spec.pair_checksums_sha256,
            "pair_manifest": spec.pair_manifest_sha256,
            "released_checkpoint": spec.released_sha256,
        },
    )
    if context.inputs.checkpoint.digest.sha256 == spec.none_sha256:
        raise ValueError("provenance checkpoint must be the RoPE member, not No-PE")
    benchmark_git = verify_git_tree(
        spec.tabarena_code_root, expected_sha=spec.tabarena_code_sha
    )
    roster = _validate_roster(
        context.inputs.dataset_manifest,
        benchmark_sha=benchmark_git.head_sha,
        expected_count=spec.expected_task_count,
    )
    checkpoint_digests = _validate_checkpoint_pair(context, spec)

    if spec.openml_cache_root.is_symlink() or not spec.openml_cache_root.is_dir():
        raise ValueError("openml_cache_root must be an existing real directory")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "the scheduled TabArena evaluation requires one CUDA accelerator"
        )
    roots = (
        context.inputs.training_code.root,
        context.inputs.model_code.root,
        context.inputs.analysis_code.root,
        benchmark_git.root,
    )
    legacy_inputs = replace(
        context.inputs,
        evidence_level="exploratory_legacy",
        legacy_reasons=("legacy_pilot_checkpoint",),
    )

    with RunTransaction(output_dir, source_roots=roots) as transaction:
        results_root = transaction.staging_dir / "tabarena-results"
        scratch_root = transaction.staging_dir / "runtime-cache"
        runtime = _import_runtime(benchmark_git.root, context.inputs.model_code.root)
        started = time.monotonic()
        returned, framework_to_arm, expected_tasks = _execute_tabarena(
            runtime=runtime,
            spec=spec,
            context=context,
            roster=roster,
            results_root=results_root,
            scratch_root=scratch_root,
        )
        duration = time.monotonic() - started
        normalized_returned = _normalize_results(
            returned,
            framework_to_arm=framework_to_arm,
            roster=roster,
            expected_count=spec.expected_result_count,
            expected_tasks=expected_tasks,
        )
        cached = _load_cached_results(results_root)
        normalized_cached = _normalize_results(
            cached,
            framework_to_arm=framework_to_arm,
            roster=roster,
            expected_count=spec.expected_result_count,
            expected_tasks=expected_tasks,
        )
        if normalized_cached != normalized_returned:
            raise RuntimeError(
                "returned TabArena results differ from the fresh result cache"
            )

        summary = _aggregate_summary(
            normalized_returned,
            roster=roster,
            spec=spec,
            framework_to_arm=framework_to_arm,
            checkpoint_digests=checkpoint_digests,
            context=context,
            benchmark_git=benchmark_git,
        )
        _publish_json(transaction.staging_dir / "summary.json", summary)
        _publish_json(
            transaction.staging_dir / "runtime.json",
            _runtime_summary(
                runtime=runtime,
                benchmark_sha=benchmark_git.head_sha,
                duration_seconds=duration,
                result_count=len(normalized_returned),
            ),
        )
        _archive_directory(results_root, transaction.staging_dir / "results.tar.gz")
        shutil.rmtree(results_root)
        if scratch_root.exists():
            shutil.rmtree(scratch_root)
        artifacts = transaction.artifact_digests(
            ("results.tar.gz", "runtime.json", "summary.json")
        )
        manifest = manifest_from_verified_inputs(legacy_inputs, artifacts=artifacts)
        benchmark_git.assert_unchanged()
        transaction.commit(manifest, verified_inputs=legacy_inputs)
    return summary


def run(args: argparse.Namespace) -> int:
    run_evaluation(Path(args.config), Path(args.output_dir))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a strict same-step TabICL comparison on TabArena classification"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EvaluationSpec",
    "build_parser",
    "main",
    "run",
    "run_evaluation",
]
