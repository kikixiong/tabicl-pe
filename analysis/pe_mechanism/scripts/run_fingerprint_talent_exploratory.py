#!/usr/bin/env python3
"""Evaluate matched full-size RoPE/Fingerprint checkpoints on TALENT.

The protocol deliberately reuses the frozen mechanism-study discovery roster:
fit the official TabICL estimator on each TALENT train split and evaluate the
corresponding validation split.  Validation and held-out *dataset rosters* are
not opened.  This run is exploratory and cannot create formal campaign evidence.
"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


ARMS = ("rope", "fingerprint", "released")
MATCHED_ARMS = ("rope", "fingerprint")
EXPECTED_DISCOVERY_COUNT = 109
EXPECTED_MANIFEST_SHA256 = (
    "1d82bf8dd40528bb9937a20204f9b23d6c48696a5c1b245304387d3b47e1ea38"
)
RELEASED_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
EXCLUDED_NON_NATIVE_CLASS_COUNTS = frozenset(
    {"texture", "walking-activity", "kr-vs-k", "letter", "UJI_Pen_Characters"}
)
ESTIMATOR_OPTIONS: Mapping[str, Any] = {
    "n_estimators": 2,
    "norm_methods": ["none", "power"],
    "feat_shuffle_method": "latin",
    "class_shuffle_method": "shift",
    "outlier_threshold": 4.0,
    "softmax_temperature": 0.9,
    "average_logits": True,
    # Ensemble members are independent; process one at a time so the frozen
    # full-row protocol fits a 22 GiB A10 without changing the ensemble roster.
    "batch_size": 1,
    "random_state": 42,
    "n_jobs": 16,
    "use_amp": False,
    "use_fa3": False,
    "offload_mode": "auto",
    "verbose": False,
}
DATASET_ARTIFACTS = frozenset({"manifest.json", "predictions.npz", "result.json"})
LINEAGE_STUDY = "tabiclv2-fullsize-rope-fingerprint-pilot-v1"
LINEAGE_ARCHITECTURE = {
    "embed_dim": 128,
    "col_num_blocks": 3,
    "col_nhead": 8,
    "col_num_inds": 128,
    "row_num_blocks": 3,
    "row_nhead": 8,
    "icl_num_blocks": 12,
    "icl_nhead": 8,
}
PREDICTION_CHUNK_ROWS = 8_192


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--talent-root", required=True)
    parser.add_argument("--rope-checkpoint", required=True)
    parser.add_argument("--rope-sha256", required=True)
    parser.add_argument("--fingerprint-checkpoint", required=True)
    parser.add_argument("--fingerprint-sha256", required=True)
    parser.add_argument("--released-checkpoint", required=True)
    parser.add_argument("--comparison-step", required=True, type=int)
    parser.add_argument("--expected-model-sha", required=True)
    parser.add_argument("--expected-analysis-sha", required=True)
    parser.add_argument("--submission-receipt", required=True)
    parser.add_argument("--rope-launch-receipt", required=True)
    parser.add_argument("--rope-completion-receipt", required=True)
    parser.add_argument("--fingerprint-launch-receipt", required=True)
    parser.add_argument("--fingerprint-completion-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", action="append", default=[])
    return parser


def _absolute(value: str, *, name: str, directory: bool = False) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    resolved = path.resolve(strict=True)
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    if directory and not resolved.is_dir():
        raise ValueError(f"{name} must be a directory")
    if not directory and not resolved.is_file():
        raise ValueError(f"{name} must be a regular file")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_document_sha256(payload: object) -> str:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_head(root: Path, *, expected: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("expected Git SHA must be 40 lowercase hexadecimal characters")
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise ValueError(f"source checkout is not clean: {root}")
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected:
        raise ValueError(f"source checkout HEAD differs from expected SHA: {root}")
    symbolic = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "-q", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if symbolic.returncode == 0:
        raise ValueError(f"source checkout must be detached at the exact SHA: {root}")
    if symbolic.returncode != 1:
        raise RuntimeError(f"could not establish detached HEAD state: {root}")
    return head


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class _RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.descriptor: int | None = None

    def __enter__(self) -> "_RunLock":
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self.descriptor = os.open(self.path, flags, 0o600)
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(self.descriptor)
            self.descriptor = None
            raise RuntimeError("another evaluator owns this output lock") from error
        owner = json.dumps(
            {
                "pid": os.getpid(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "acquired_unix_seconds": time.time(),
            },
            sort_keys=True,
        ).encode("utf-8")
        os.ftruncate(self.descriptor, 0)
        os.write(self.descriptor, owner)
        os.fsync(self.descriptor)
        return self

    def __exit__(self, *_: object) -> None:
        assert self.descriptor is not None
        fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        os.close(self.descriptor)
        self.descriptor = None


def _load_mapping(path: Path, *, name: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return payload


def _require_values(
    payload: Mapping[str, Any], expected: Mapping[str, Any], *, name: str
) -> None:
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{name} contract mismatch: {mismatches}")


def _lineage_contract(
    *,
    receipts: Mapping[str, Path],
    checkpoints: Mapping[str, Path],
    checkpoint_digests: Mapping[str, str],
    comparison_step: int,
    model_sha: str,
) -> dict[str, Any]:
    payloads = {name: _load_mapping(path, name=name) for name, path in receipts.items()}
    submission = payloads["submission"]
    _require_values(
        submission,
        {
            "schema_version": 1,
            "study": LINEAGE_STUDY,
            "formal_evidence": False,
            "fresh_from_scratch": True,
            "max_steps": comparison_step,
            "seed": 42,
            "source_commit": model_sha,
        },
        name="submission receipt",
    )
    jobs = submission.get("jobs")
    if not isinstance(jobs, Mapping) or set(jobs) != set(MATCHED_ARMS):
        raise ValueError("submission receipt job roster is invalid")

    launch_facts: dict[str, Any] = {}
    completion_facts: dict[str, Any] = {}
    for arm in MATCHED_ARMS:
        launch = payloads[f"{arm}_launch"]
        completion = payloads[f"{arm}_completion"]
        _require_values(
            launch,
            {
                "schema_version": 1,
                "study": LINEAGE_STUDY,
                "arm": arm,
                "formal_evidence": False,
                "fresh_from_scratch": True,
                "max_steps": comparison_step,
                "seed": 42,
                "source_commit": model_sha,
                "batch_size": 64,
                "batch_size_per_gp": 8,
                "micro_batch_size": 8,
                "n_jobs": 48,
                "gpu": "NVIDIA H100 80GB HBM3",
            },
            name=f"{arm} launch receipt",
        )
        if str(jobs[arm]) != str(launch.get("slurm_job_id")):
            raise ValueError(f"{arm} launch does not belong to the submitted cohort")
        launch_architecture = launch.get("architecture")
        expected_launch_architecture = {
            key: LINEAGE_ARCHITECTURE[key]
            for key in (
                "embed_dim",
                "col_num_blocks",
                "row_num_blocks",
                "icl_num_blocks",
            )
        }
        if launch_architecture != expected_launch_architecture:
            raise ValueError(f"{arm} launch architecture is invalid")

        _require_values(
            completion,
            {
                "schema_version": 1,
                "study": LINEAGE_STUDY,
                "arm": arm,
                "formal_evidence": False,
                "curr_step": comparison_step,
                "source_commit": model_sha,
                "architecture": LINEAGE_ARCHITECTURE,
            },
            name=f"{arm} completion receipt",
        )
        checkpoint = completion.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"{arm} completion checkpoint record is invalid")
        _require_values(
            checkpoint,
            {
                "sha256": checkpoint_digests[arm],
                "size_bytes": checkpoints[arm].stat().st_size,
            },
            name=f"{arm} completion checkpoint",
        )
        treatment = completion.get("treatment")
        expected_treatment = {
            "row_identity_mode": "rope" if arm == "rope" else "none",
            "row_fingerprint": arm == "fingerprint",
            "row_fingerprint_dim": 16,
        }
        if treatment != expected_treatment:
            raise ValueError(f"{arm} completion treatment is invalid")
        launch_facts[arm] = {
            key: launch[key]
            for key in (
                "architecture",
                "batch_size",
                "batch_size_per_gp",
                "micro_batch_size",
                "n_jobs",
                "gpu",
                "python",
                "torch",
                "cuda_build",
            )
        }
        completion_facts[arm] = {
            "architecture": completion["architecture"],
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_size_bytes": checkpoint["size_bytes"],
            "state_elements": checkpoint.get("state_elements"),
            "optimizer_prefix": completion.get("optimizer_prefix"),
            "treatment": treatment,
        }
    shared_launch_keys = (
        "architecture",
        "batch_size",
        "batch_size_per_gp",
        "micro_batch_size",
        "n_jobs",
        "gpu",
        "python",
        "torch",
        "cuda_build",
    )
    for key in shared_launch_keys:
        if launch_facts["rope"][key] != launch_facts["fingerprint"][key]:
            raise ValueError(f"matched arms differ in launch field {key}")
    if (
        completion_facts["rope"]["optimizer_prefix"]
        != completion_facts["fingerprint"]["optimizer_prefix"]
    ):
        raise ValueError("matched arms differ in optimizer prefix")
    return {
        "schema_version": 1,
        "study": LINEAGE_STUDY,
        "formal_evidence": False,
        "fresh_from_scratch": True,
        "source_commit": model_sha,
        "seed": 42,
        "max_steps": comparison_step,
        "receipt_sha256": {
            name: _sha256(path) for name, path in sorted(receipts.items())
        },
        "shared_launch": launch_facts["rope"],
        "arms": completion_facts,
    }


def _stage_checkpoints(
    checkpoints: Mapping[str, Path], expected_digests: Mapping[str, str]
) -> tuple[Path, dict[str, Path]]:
    temporary_parent = os.environ.get("SLURM_TMPDIR")
    if temporary_parent and not Path(temporary_parent).is_dir():
        raise ValueError("SLURM_TMPDIR is not an existing directory")
    directory = Path(
        tempfile.mkdtemp(prefix="fingerprint-talent-checkpoints-", dir=temporary_parent)
    )
    atexit.register(shutil.rmtree, directory, True)
    staged: dict[str, Path] = {}
    try:
        for arm, source in checkpoints.items():
            destination = directory / f"{arm}.ckpt"
            shutil.copyfile(source, destination)
            with destination.open("rb") as handle:
                os.fsync(handle.fileno())
            if _sha256(destination) != expected_digests[arm]:
                raise RuntimeError(f"staged {arm} checkpoint digest mismatch")
            destination.chmod(0o400)
            staged[arm] = destination
        _fsync_directory(directory)
        return directory, staged
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def _runtime_environment(analysis_root: Path, model_root: Path) -> dict[str, Any]:
    import pandas
    import pe_mechanism
    import sklearn
    import tabicl
    import torch

    origins = {
        "pe_mechanism": Path(pe_mechanism.__file__).resolve(strict=True),
        "tabicl": Path(tabicl.__file__).resolve(strict=True),
    }
    if not origins["pe_mechanism"].is_relative_to(analysis_root):
        raise RuntimeError("pe_mechanism was imported outside the exact analysis tree")
    if not origins["tabicl"].is_relative_to(model_root):
        raise RuntimeError("tabicl was imported outside the exact model tree")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("evaluation requires exactly one visible CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    packages = {}
    for distribution in ("numpy", "pandas", "scikit-learn", "torch"):
        packages[distribution] = importlib.metadata.version(distribution)
    return {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "packages": packages,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [properties.major, properties.minor],
        },
        "imports": {name: str(path) for name, path in origins.items()},
        "official_inference_contract_sha256": _sha256(
            analysis_root / "analysis/pe_mechanism/src/pe_mechanism/official_tabicl.py"
        ),
    }


def _checkpoint_contract(
    rope: Path, fingerprint: Path, *, comparison_step: int
) -> dict[str, Any]:
    import torch

    if comparison_step <= 0:
        raise ValueError("comparison_step must be positive")
    payloads = {
        arm: torch.load(path, map_location="cpu", weights_only=True)
        for arm, path in (("rope", rope), ("fingerprint", fingerprint))
    }
    for arm, payload in payloads.items():
        if payload.get("curr_step") != comparison_step:
            raise ValueError(f"{arm} checkpoint is at the wrong step")
        if not isinstance(payload.get("config"), dict) or not isinstance(
            payload.get("state_dict"), dict
        ):
            raise ValueError(f"{arm} checkpoint lacks config/state_dict")
        identity = payload.get("identity_treatment")
        expected_identity = "rope" if arm == "rope" else "none"
        if not isinstance(identity, dict) or {
            key: identity.get(key)
            for key in (
                "schema_version",
                "row_identity_mode",
                "identity_rng_seed",
                "sampler_version",
                "world_size",
            )
        } != {
            "schema_version": 1,
            "row_identity_mode": expected_identity,
            "identity_rng_seed": 42,
            "sampler_version": None,
            "world_size": 1,
        }:
            raise ValueError(f"{arm} identity treatment metadata is invalid")
        stream = payload.get("prior_stream")
        if not isinstance(stream, dict) or (
            stream.get("cursor") != comparison_step
            or stream.get("experiment_seed") != 42
            or stream.get("ddp_rank") != 0
            or stream.get("world_size") != 1
        ):
            raise ValueError(f"{arm} checkpoint prior stream is invalid")
    if payloads["rope"]["prior_stream"] != payloads["fingerprint"]["prior_stream"]:
        raise ValueError("checkpoint prior streams are not exactly matched")
    configs = {arm: dict(payload["config"]) for arm, payload in payloads.items()}
    if (
        configs["rope"].get("row_identity_mode") != "rope"
        or configs["rope"].get("row_fingerprint") is not False
    ):
        raise ValueError("RoPE checkpoint treatment is invalid")
    if (
        configs["fingerprint"].get("row_identity_mode") != "none"
        or configs["fingerprint"].get("row_fingerprint") is not True
        or configs["fingerprint"].get("row_fingerprint_dim") != 16
    ):
        raise ValueError("Fingerprint checkpoint treatment is invalid")
    for arm, config in configs.items():
        if {
            key: config.get(key) for key in LINEAGE_ARCHITECTURE
        } != LINEAGE_ARCHITECTURE:
            raise ValueError(f"{arm} checkpoint is not full-size")
    treatment = {"row_identity_mode", "row_fingerprint"}
    if {
        key: value for key, value in configs["rope"].items() if key not in treatment
    } != {
        key: value
        for key, value in configs["fingerprint"].items()
        if key not in treatment
    }:
        raise ValueError("checkpoint configs differ outside the treatment")
    state_keys = {arm: set(payloads[arm]["state_dict"]) for arm in MATCHED_ARMS}
    if "row_interactor.tf_row.rope.freqs" not in state_keys["rope"]:
        raise ValueError("RoPE checkpoint lacks Row-RoPE state")
    if any("fingerprint_" in key for key in state_keys["rope"]):
        raise ValueError("RoPE checkpoint unexpectedly contains Fingerprint state")
    required_fingerprint_fragments = (
        "row_interactor.fingerprint_q_gates",
        "row_interactor.fingerprint_k_gates",
        "row_interactor.fingerprint_q_projections.",
        "row_interactor.fingerprint_k_projections.",
    )
    if "row_interactor.tf_row.rope.freqs" in state_keys["fingerprint"] or not all(
        any(fragment in key for key in state_keys["fingerprint"])
        for fragment in required_fingerprint_fragments
    ):
        raise ValueError("Fingerprint checkpoint state does not encode the treatment")
    return {
        arm: {
            "curr_step": comparison_step,
            "model_config": configs[arm],
            "model_state_tensors": len(payloads[arm]["state_dict"]),
            "model_state_elements": sum(
                value.numel() for value in payloads[arm]["state_dict"].values()
            ),
            "identity_treatment": payloads[arm]["identity_treatment"],
        }
        for arm in MATCHED_ARMS
    }


def _discovery_roster(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    assignments = manifest.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("TALENT manifest assignments are invalid")
    names = tuple(
        str(item["name"])
        for item in assignments
        if isinstance(item, Mapping)
        and item.get("split") == "discovery"
        and item.get("name") not in EXCLUDED_NON_NATIVE_CLASS_COUNTS
    )
    if len(names) != EXPECTED_DISCOVERY_COUNT or len(names) != len(set(names)):
        raise ValueError("native TALENT discovery roster is not the expected 109 tasks")
    return names


def _class_tokens(values: Sequence[Any]) -> list[str]:
    return [f"{type(value).__name__}:{value!r}" for value in values]


def _assert_treatment(
    driver: Any,
    arm: str,
    *,
    expected_model_sha: str | None = None,
    expected_checkpoint_sha: str | None = None,
) -> dict[str, Any]:
    if expected_model_sha is not None and driver.model_sha != expected_model_sha:
        raise RuntimeError(f"{arm} driver model SHA changed")
    if (
        expected_checkpoint_sha is not None
        and driver.checkpoint_sha != expected_checkpoint_sha
    ):
        raise RuntimeError(f"{arm} driver checkpoint SHA changed")
    if getattr(driver, "source_evidence_level", None) != "strict":
        raise RuntimeError(f"{arm} driver lost strict source evidence")
    raw = driver.estimator.model_
    row = raw.row_interactor
    observed = {
        "row_identity_mode": getattr(raw, "row_identity_mode", None),
        "row_interactor_identity_mode": getattr(row, "identity_mode", None),
        "row_fingerprint": getattr(raw, "row_fingerprint", None),
        "row_fingerprint_dim": getattr(raw, "row_fingerprint_dim", None),
        "row_rope_present": getattr(getattr(row, "tf_row", None), "rope", None)
        is not None,
        "fingerprint_q_gates_present": getattr(row, "fingerprint_q_gates", None)
        is not None,
        "fingerprint_k_gates_present": getattr(row, "fingerprint_k_gates", None)
        is not None,
        "fingerprint_q_projections_present": getattr(
            row, "fingerprint_q_projections", None
        )
        is not None,
        "fingerprint_k_projections_present": getattr(
            row, "fingerprint_k_projections", None
        )
        is not None,
    }
    expected_identity = "none" if arm == "fingerprint" else "rope"
    expected_fingerprint = arm == "fingerprint"
    if (
        observed["row_identity_mode"] != expected_identity
        or observed["row_interactor_identity_mode"] != expected_identity
        or observed["row_fingerprint"] is not expected_fingerprint
        or observed["row_rope_present"] is expected_fingerprint
        or any(
            observed[key] is not expected_fingerprint
            for key in (
                "fingerprint_q_gates_present",
                "fingerprint_k_gates_present",
                "fingerprint_q_projections_present",
                "fingerprint_k_projections_present",
            )
        )
    ):
        raise RuntimeError(f"{arm} inference treatment is misaligned: {observed}")
    if expected_fingerprint and observed["row_fingerprint_dim"] != 16:
        raise RuntimeError("Fingerprint inference dimension is not 16")
    return observed


def _exact_sign_p(
    left: np.ndarray, right: np.ndarray, *, lower_is_better: bool
) -> dict[str, Any]:
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired metric arrays must be matching vectors")
    delta = right - left if lower_is_better else left - right
    wins = int(np.count_nonzero(delta > 0))
    losses = int(np.count_nonzero(delta < 0))
    ties = int(delta.size - wins - losses)
    n = wins + losses
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, index) for index in range(min(wins, losses) + 1))
        p_value = min(1.0, 2.0 * tail / (2**n))
    return {
        "left_wins": wins,
        "right_wins": losses,
        "ties": ties,
        "two_sided_exact_sign_test_p": p_value,
    }


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate an empty result set")
    arm_metrics = {
        arm: {
            metric: np.asarray(
                [record["arms"][arm][metric] for record in records], dtype=np.float64
            )
            for metric in ("accuracy", "log_loss")
        }
        for arm in ARMS
    }
    rope_accuracy = arm_metrics["rope"]["accuracy"]
    fingerprint_accuracy = arm_metrics["fingerprint"]["accuracy"]
    rope_loss = arm_metrics["rope"]["log_loss"]
    fingerprint_loss = arm_metrics["fingerprint"]["log_loss"]
    primary = {
        "left": "fingerprint",
        "right": "rope",
        "accuracy": {
            **_exact_sign_p(fingerprint_accuracy, rope_accuracy, lower_is_better=False),
            "mean_improvement_left_over_right": float(
                np.mean(fingerprint_accuracy - rope_accuracy)
            ),
            "inferential_role": "primary",
        },
        "log_loss": {
            **_exact_sign_p(fingerprint_loss, rope_loss, lower_is_better=True),
            "mean_improvement_left_over_right": float(
                np.mean(rope_loss - fingerprint_loss)
            ),
            "inferential_role": "supportive_unadjusted",
        },
    }
    external = {}
    for arm in MATCHED_ARMS:
        external[f"{arm}_vs_released"] = {
            "accuracy_mean_improvement_arm_over_released": float(
                np.mean(
                    arm_metrics[arm]["accuracy"] - arm_metrics["released"]["accuracy"]
                )
            ),
            "log_loss_mean_improvement_arm_over_released": float(
                np.mean(
                    arm_metrics["released"]["log_loss"] - arm_metrics[arm]["log_loss"]
                )
            ),
            "inferential_role": "descriptive_only_not_treatment_matched",
        }
    return {
        "dataset_count": len(records),
        "macro_mean": {
            arm: {metric: float(np.mean(values)) for metric, values in metrics.items()}
            for arm, metrics in arm_metrics.items()
        },
        "primary_matched_comparison": primary,
        "external_released_reference": external,
        "multiplicity_policy": (
            "Fingerprint-vs-RoPE accuracy is the sole primary test; paired log-loss "
            "is supportive and unadjusted; released comparisons are descriptive only."
        ),
    }


def _validated_probabilities(result: Any, *, rows: int) -> np.ndarray:
    probabilities = np.asarray(result.probabilities, dtype=np.float32)
    if probabilities.ndim != 2 or probabilities.shape[0] != rows:
        raise RuntimeError("predict_proba returned the wrong shape")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise RuntimeError("predict_proba returned invalid probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise RuntimeError("predict_proba rows are not normalized")
    return probabilities


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + contiguous.tobytes(order="C")).hexdigest()


def _row_roster_sha256(dataset: str, split: str, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    header = json.dumps(
        {
            "dataset": dataset,
            "split": split,
            "row_policy": "all_rows_in_original_order",
            "row_count": int(len(labels)),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(header + b"\0")
    for index, value in enumerate(np.asarray(labels).reshape(-1)):
        token = f"{type(value).__name__}:{value!r}"
        digest.update(f"{index}:".encode("ascii"))
        digest.update(token.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _forward_schedule(entries: Sequence[Any]) -> dict[str, Any]:
    calls = []
    for entry in entries:
        payload = asdict(entry) if is_dataclass(entry) else vars(entry)
        calls.append(payload)
    encoded = json.dumps(
        calls, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return {
        "call_count": len(calls),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _row_slice(values: Any, start: int, stop: int) -> Any:
    if hasattr(values, "iloc"):
        return values.iloc[start:stop]
    return values[start:stop]


def _all_missing_columns(values: Any) -> np.ndarray:
    if hasattr(values, "isna"):
        return np.asarray(values.isna().all(axis=0), dtype=bool)
    import pandas as pd

    return np.asarray(pd.isna(np.asarray(values))).all(axis=0)


def _predict_in_chunks(
    driver: Any,
    *,
    X: Any,
    y: np.ndarray,
    arm: str,
    model_sha: str,
    checkpoint_sha256: str,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    import torch

    probabilities = []
    encoded_targets = []
    expected_classes: list[str] | None = None
    chunk_count = 0
    full_missing_columns = _all_missing_columns(X)
    for start in range(0, len(y), PREDICTION_CHUNK_ROWS):
        stop = min(len(y), start + PREDICTION_CHUNK_ROWS)
        chunk_y = np.asarray(y[start:stop])
        chunk_X = _row_slice(X, start, stop)
        if not np.array_equal(_all_missing_columns(chunk_X), full_missing_columns):
            raise RuntimeError(
                "prediction chunk changes the official all-missing feature mask"
            )
        result = driver.predict_proba(chunk_X, y=chunk_y)
        _assert_treatment(
            driver,
            arm,
            expected_model_sha=model_sha,
            expected_checkpoint_sha=checkpoint_sha256,
        )
        if (
            not result.exact_baseline_verified
            or result.source_evidence_level != "strict"
        ):
            raise RuntimeError(f"{arm} official inference evidence is not strict")
        if result.forward_calls:
            raise RuntimeError(
                "direct exploratory inference unexpectedly installed hooks"
            )
        current = _validated_probabilities(result, rows=len(chunk_y))
        baseline = np.asarray(result.baseline_probabilities, dtype=np.float32)
        if not np.array_equal(current, baseline):
            raise RuntimeError(f"{arm} probabilities differ from the direct baseline")
        classes = _class_tokens(result.classes)
        if expected_classes is None:
            expected_classes = classes
        elif classes != expected_classes:
            raise RuntimeError(f"{arm} class encoding changed between chunks")
        target = np.asarray(
            driver.estimator.y_encoder_.transform(chunk_y), dtype=np.int64
        )
        selected = current[np.arange(target.size), target]
        accuracy = float(np.mean(np.argmax(current, axis=1) == target))
        log_loss = float(
            -np.log(np.clip(selected, np.finfo(np.float32).tiny, 1.0)).mean()
        )
        if result.metrics is None or not (
            math.isclose(accuracy, result.metrics.accuracy, abs_tol=1e-7)
            and math.isclose(log_loss, result.metrics.log_loss, abs_tol=1e-6)
        ):
            raise RuntimeError("chunk metrics disagree with independent recomputation")
        probabilities.append(current)
        encoded_targets.append(target)
        chunk_count += 1
        del result, baseline
        gc.collect()
        torch.cuda.empty_cache()
    if expected_classes is None or not probabilities:
        raise RuntimeError("validation split produced no prediction chunks")
    return (
        np.concatenate(probabilities, axis=0),
        np.concatenate(encoded_targets, axis=0),
        expected_classes,
        {
            "chunk_count": chunk_count,
            "maximum_rows_per_call": PREDICTION_CHUNK_ROWS,
            "all_chunks_exact_baseline_verified": True,
            "source_evidence_level": "strict",
        },
    )


def _evaluate_dataset(
    *,
    name: str,
    ordinal: int,
    talent_root: Path,
    checkpoints: Mapping[str, Path],
    checkpoint_digests: Mapping[str, str],
    model_root: Path,
    model_sha: str,
    run_contract_sha256: str,
    destination: Path,
) -> dict[str, Any]:
    import torch
    from pe_mechanism.official_tabicl import (
        _expected_forward_schedule,
        fit_official_tabicl_driver,
        load_raw_talent_splits,
    )

    dataset = load_raw_talent_splits(talent_root / name, trusted_pickle=True)
    train_classes = np.unique(dataset.train.y)
    if not 2 <= len(train_classes) <= 10:
        raise RuntimeError(f"{name} is outside the native class-count contract")
    arm_records: dict[str, Any] = {}
    probabilities: dict[str, np.ndarray] = {}
    encoded_target: np.ndarray | None = None
    expected_classes: list[str] | None = None
    expected_schedule: dict[str, Any] | None = None
    for arm in ARMS:
        fit_started = time.monotonic()
        driver = fit_official_tabicl_driver(
            checkpoints[arm],
            dataset.train.X,
            dataset.train.y,
            device="cuda",
            model_sha=model_sha,
            estimator_options=ESTIMATOR_OPTIONS,
            expected_source_root=model_root,
            fit_context="talent-train",
        )
        fit_seconds = time.monotonic() - fit_started
        treatment = _assert_treatment(
            driver,
            arm,
            expected_model_sha=model_sha,
            expected_checkpoint_sha=checkpoint_digests[arm],
        )
        schedule = _forward_schedule(_expected_forward_schedule(driver.estimator))
        if expected_schedule is None:
            expected_schedule = schedule
        elif schedule != expected_schedule:
            raise RuntimeError(
                "paired arms used different official inference schedules"
            )
        predict_started = time.monotonic()
        current, target, class_tokens, prediction_contract = _predict_in_chunks(
            driver,
            X=dataset.val.X,
            y=np.asarray(dataset.val.y),
            arm=arm,
            model_sha=model_sha,
            checkpoint_sha256=checkpoint_digests[arm],
        )
        predict_seconds = time.monotonic() - predict_started
        if expected_classes is None:
            expected_classes = class_tokens
            encoded_target = target
        elif class_tokens != expected_classes or not np.array_equal(
            target, encoded_target
        ):
            raise RuntimeError("paired arms used different class/target encodings")
        selected = current[np.arange(target.size), target]
        accuracy = float(np.mean(np.argmax(current, axis=1) == target))
        log_loss = float(
            -np.log(np.clip(selected, np.finfo(np.float32).tiny, 1.0)).mean()
        )
        probabilities[arm] = current
        arm_records[arm] = {
            "accuracy": accuracy,
            "log_loss": log_loss,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "treatment": treatment,
            "checkpoint_sha256": driver.checkpoint_sha,
            "model_sha": driver.model_sha,
            "source_evidence_level": prediction_contract["source_evidence_level"],
            "exact_baseline_verified": prediction_contract[
                "all_chunks_exact_baseline_verified"
            ],
            "probabilities_sha256": _array_sha256(current),
            "official_forward_schedule": schedule,
            "prediction_chunking": prediction_contract,
            "reference_role": (
                "external_released_reference_descriptive_only"
                if arm == "released"
                else "matched_step5000_treatment"
            ),
        }
        del driver
        gc.collect()
        torch.cuda.empty_cache()

    assert encoded_target is not None and expected_classes is not None
    staging = Path(
        tempfile.mkdtemp(prefix=f".{ordinal:04d}.tmp-", dir=destination.parent)
    )
    try:
        np.savez_compressed(
            staging / "predictions.npz",
            target=encoded_target,
            rope=probabilities["rope"],
            fingerprint=probabilities["fingerprint"],
            released=probabilities["released"],
        )
        with (staging / "predictions.npz").open("rb") as handle:
            os.fsync(handle.fileno())
        record = {
            "schema_version": 1,
            "run_contract_sha256": run_contract_sha256,
            "ordinal": ordinal,
            "dataset": name,
            "fit_split": "train",
            "evaluation_split": "val",
            "n_train": int(len(dataset.train.y)),
            "n_evaluation": int(len(dataset.val.y)),
            "n_features": int(
                dataset.n_numeric_features + dataset.n_categorical_features
            ),
            "n_classes": len(expected_classes),
            "row_sampling": "none_full_split_original_order",
            "prediction_chunk_rows": PREDICTION_CHUNK_ROWS,
            "row_roster_sha256": {
                "train": _row_roster_sha256(name, "train", dataset.train.y),
                "val": _row_roster_sha256(name, "val", dataset.val.y),
            },
            "class_tokens": expected_classes,
            "info_sha256": dataset.info_sha256,
            "input_sha256": dict(sorted(dataset.input_sha256.items())),
            "arms": arm_records,
        }
        _atomic_json(staging / "result.json", record)
        _atomic_json(
            staging / "manifest.json",
            {
                "result.json": _sha256(staging / "result.json"),
                "predictions.npz": _sha256(staging / "predictions.npz"),
            },
        )
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
        return record
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _cleanup_transients(work: Path, results_root: Path) -> list[str]:
    cleaned: list[str] = []
    for entry in work.iterdir():
        if entry.name.startswith(".") and ".tmp-" in entry.name:
            if entry.is_symlink() or not entry.is_file():
                raise RuntimeError(f"unsafe top-level transient artifact: {entry.name}")
            entry.unlink()
            cleaned.append(entry.name)
    for entry in results_root.iterdir():
        if re.fullmatch(r"\.\d{4}\.tmp-.+", entry.name):
            if entry.is_symlink() or not entry.is_dir():
                raise RuntimeError(f"unsafe dataset transient artifact: {entry.name}")
            shutil.rmtree(entry)
            cleaned.append(f"datasets/{entry.name}")
    if cleaned:
        _fsync_directory(work)
        _fsync_directory(results_root)
    return cleaned


def _validated_cached_record(
    *,
    destination: Path,
    name: str,
    ordinal: int,
    run_contract_sha256: str,
    talent_root: Path,
) -> Mapping[str, Any]:
    if destination.is_symlink() or not destination.is_dir():
        raise RuntimeError(f"cached dataset path is unsafe for {name}")
    entries = {entry.name for entry in destination.iterdir()}
    if entries != DATASET_ARTIFACTS:
        raise RuntimeError(f"cached dataset artifact roster is invalid for {name}")
    for entry in destination.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise RuntimeError(f"cached dataset artifact is unsafe for {name}")
    manifest = _load_mapping(destination / "manifest.json", name="dataset manifest")
    if set(manifest) != {"result.json", "predictions.npz"}:
        raise RuntimeError(f"cached dataset manifest is not exact for {name}")
    for filename, digest in manifest.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"cached dataset manifest digest is invalid for {name}")
        if _sha256(destination / filename) != digest:
            raise RuntimeError(f"cached result digest mismatch for {name}")
    record = _load_mapping(destination / "result.json", name="dataset result")
    _require_values(
        record,
        {
            "schema_version": 1,
            "run_contract_sha256": run_contract_sha256,
            "dataset": name,
            "ordinal": ordinal,
            "fit_split": "train",
            "evaluation_split": "val",
            "row_sampling": "none_full_split_original_order",
            "prediction_chunk_rows": PREDICTION_CHUNK_ROWS,
        },
        name=f"cached dataset {name}",
    )
    input_digests = record.get("input_sha256")
    if not isinstance(input_digests, Mapping) or not input_digests:
        raise RuntimeError(f"cached dataset input contract is missing for {name}")
    dataset_root = (talent_root / name).resolve(strict=True)
    if dataset_root.is_symlink() or not dataset_root.is_dir():
        raise RuntimeError(f"TALENT dataset root is unsafe for {name}")
    for filename, digest in input_digests.items():
        if not isinstance(filename, str) or not isinstance(digest, str):
            raise RuntimeError(f"cached TALENT input contract is invalid for {name}")
        source = dataset_root / filename
        if source.is_symlink() or not source.is_file() or _sha256(source) != digest:
            raise RuntimeError(f"TALENT input bytes changed for {name}/{filename}")
    arms = record.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
        raise RuntimeError(f"cached arm roster is invalid for {name}")
    n_evaluation = record.get("n_evaluation")
    if not isinstance(n_evaluation, int) or isinstance(n_evaluation, bool):
        raise RuntimeError(f"cached evaluation row count is invalid for {name}")
    if n_evaluation <= 0:
        raise RuntimeError(f"cached evaluation row count is invalid for {name}")
    expected_chunking = {
        "chunk_count": math.ceil(n_evaluation / PREDICTION_CHUNK_ROWS),
        "maximum_rows_per_call": PREDICTION_CHUNK_ROWS,
        "all_chunks_exact_baseline_verified": True,
        "source_evidence_level": "strict",
    }
    for arm in ARMS:
        arm_record = arms[arm]
        if not isinstance(arm_record, Mapping):
            raise RuntimeError(f"cached arm record is invalid for {name}/{arm}")
        if arm_record.get("prediction_chunking") != expected_chunking:
            raise RuntimeError(
                f"cached prediction chunk contract is invalid for {name}/{arm}"
            )
    with np.load(destination / "predictions.npz", allow_pickle=False) as predictions:
        if set(predictions.files) != {"target", *ARMS}:
            raise RuntimeError(f"cached prediction roster is invalid for {name}")
        target = np.asarray(predictions["target"])
        if target.shape != (n_evaluation,):
            raise RuntimeError(f"cached target shape is invalid for {name}")
        for arm in ARMS:
            probabilities = np.asarray(predictions[arm])
            if probabilities.shape != (
                n_evaluation,
                record.get("n_classes"),
            ):
                raise RuntimeError(
                    f"cached probability shape is invalid for {name}/{arm}"
                )
            _validated_probabilities(
                type("CachedResult", (), {"probabilities": probabilities})(),
                rows=len(target),
            )
            if _array_sha256(probabilities) != arms[arm].get("probabilities_sha256"):
                raise RuntimeError(
                    f"cached probability digest changed for {name}/{arm}"
                )
            selected = probabilities[np.arange(target.size), target]
            accuracy = float(np.mean(np.argmax(probabilities, axis=1) == target))
            log_loss = float(
                -np.log(np.clip(selected, np.finfo(np.float32).tiny, 1.0)).mean()
            )
            if not (
                math.isclose(accuracy, arms[arm].get("accuracy"), abs_tol=1e-7)
                and math.isclose(log_loss, arms[arm].get("log_loss"), abs_tol=1e-6)
            ):
                raise RuntimeError(f"cached metrics changed for {name}/{arm}")
    return record


def _artifact_manifest(work: Path, roster: Sequence[str]) -> dict[str, Any]:
    expected_top = {
        "attempts.json",
        "datasets",
        "environment-contract.json",
        "run-contract.json",
        "runtime.json",
        "summary.json",
    }
    observed_top = {
        entry.name for entry in work.iterdir() if entry.name != "manifest.json"
    }
    if observed_top != expected_top:
        raise RuntimeError("work directory contains unexpected top-level artifacts")
    artifacts: dict[str, Any] = {}
    for filename in sorted(expected_top - {"datasets"}):
        path = work / filename
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"unsafe top-level artifact: {filename}")
        artifacts[filename] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    results_root = work / "datasets"
    expected_directories = {f"{ordinal:04d}" for ordinal in range(len(roster))}
    observed_directories = {entry.name for entry in results_root.iterdir()}
    if observed_directories != expected_directories:
        raise RuntimeError(
            "dataset artifact directory roster is incomplete or unexpected"
        )
    for directory in sorted(expected_directories):
        dataset_root = results_root / directory
        if dataset_root.is_symlink() or not dataset_root.is_dir():
            raise RuntimeError(f"unsafe dataset artifact directory: {directory}")
        entries = {entry.name for entry in dataset_root.iterdir()}
        if entries != DATASET_ARTIFACTS:
            raise RuntimeError(f"unexpected artifact roster in datasets/{directory}")
        for filename in sorted(entries):
            path = dataset_root / filename
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"unsafe artifact: datasets/{directory}/{filename}")
            logical = f"datasets/{directory}/{filename}"
            artifacts[logical] = {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
    return {"schema_version": 1, "artifacts": artifacts}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_locked(
    args: argparse.Namespace,
    *,
    analysis_root: Path,
    model_root: Path,
    talent_root: Path,
    output: Path,
) -> int:
    analysis_sha = _git_head(analysis_root, expected=args.expected_analysis_sha)
    model_sha = _git_head(model_root, expected=args.expected_model_sha)
    checkpoints = {
        "rope": _absolute(args.rope_checkpoint, name="rope_checkpoint"),
        "fingerprint": _absolute(
            args.fingerprint_checkpoint, name="fingerprint_checkpoint"
        ),
        "released": _absolute(args.released_checkpoint, name="released_checkpoint"),
    }
    expected_digests = {
        "rope": args.rope_sha256,
        "fingerprint": args.fingerprint_sha256,
        "released": RELEASED_SHA256,
    }
    if any(
        not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in expected_digests.values()
    ):
        raise ValueError("checkpoint SHA-256 values must be lowercase hexadecimal")
    observed_digests = {arm: _sha256(path) for arm, path in checkpoints.items()}
    if observed_digests != expected_digests:
        raise ValueError(f"checkpoint digest mismatch: {observed_digests}")
    checkpoint_contract = _checkpoint_contract(
        checkpoints["rope"],
        checkpoints["fingerprint"],
        comparison_step=args.comparison_step,
    )
    receipts = {
        "submission": _absolute(args.submission_receipt, name="submission_receipt"),
        "rope_launch": _absolute(args.rope_launch_receipt, name="rope_launch_receipt"),
        "rope_completion": _absolute(
            args.rope_completion_receipt, name="rope_completion_receipt"
        ),
        "fingerprint_launch": _absolute(
            args.fingerprint_launch_receipt, name="fingerprint_launch_receipt"
        ),
        "fingerprint_completion": _absolute(
            args.fingerprint_completion_receipt,
            name="fingerprint_completion_receipt",
        ),
    }
    lineage = _lineage_contract(
        receipts=receipts,
        checkpoints=checkpoints,
        checkpoint_digests=expected_digests,
        comparison_step=args.comparison_step,
        model_sha=model_sha,
    )
    for arm in MATCHED_ARMS:
        if (
            lineage["arms"][arm]["state_elements"]
            != checkpoint_contract[arm]["model_state_elements"]
        ):
            raise ValueError(f"{arm} completion receipt state size is invalid")

    manifest_path = (
        analysis_root
        / "analysis/pe_mechanism/manifests/talent-classification-split-v1.json"
    )
    if _sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("TALENT split manifest digest changed")
    manifest = _load_mapping(manifest_path, name="TALENT split manifest")
    roster = _discovery_roster(manifest)
    if args.dataset:
        unknown = sorted(set(args.dataset) - set(roster))
        if unknown:
            raise ValueError(
                f"datasets are outside the frozen discovery roster: {unknown}"
            )
        requested = set(args.dataset)
        roster = tuple(name for name in roster if name in requested)
    if not roster:
        raise ValueError("evaluation roster must not be empty")

    environment = _runtime_environment(analysis_root, model_root)
    contract = {
        "schema_version": 2,
        "study": "fullsize-fingerprint-talent-discovery-exploratory",
        "formal_eligible": False,
        "analysis_sha": analysis_sha,
        "model_sha": model_sha,
        "environment_contract_sha256": _json_document_sha256(environment),
        "comparison_step": args.comparison_step,
        "checkpoint_sha256": observed_digests,
        "training_lineage": lineage,
        "talent_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "roster": list(roster),
        "fit_split": "train",
        "evaluation_split": "val",
        "row_sampling": "none_full_split_original_order",
        "prediction_chunk_rows": PREDICTION_CHUNK_ROWS,
        "test_split_policy": (
            "raw loader integrity-checks local test bytes; test labels and rows are not "
            "used for fitting, metrics, model selection, or dataset selection"
        ),
        "dataset_roster_policy": (
            "frozen discovery only; protected validation and held-out dataset rosters "
            "are not evaluated"
        ),
        "trusted_local_talent_pickle": True,
        "estimator_options": dict(ESTIMATOR_OPTIONS),
        "comparison_roles": {
            "rope": "matched_step5000_treatment",
            "fingerprint": "matched_step5000_treatment",
            "released": "external_550000_step_reference_descriptive_only",
        },
    }
    run_contract_sha256 = _json_document_sha256(contract)
    work = output.with_name(f".{output.name}.work")
    contract_path = work / "run-contract.json"
    environment_path = work / "environment-contract.json"
    if work.exists():
        if work.is_symlink() or not work.is_dir():
            raise ValueError("existing work path is not a safe resumable directory")
        if not contract_path.is_file() or not environment_path.is_file():
            raise ValueError("existing work path lacks its immutable contracts")
        if _load_mapping(contract_path, name="run contract") != contract:
            raise ValueError("existing work contract differs from this invocation")
        if _load_mapping(environment_path, name="environment contract") != environment:
            raise ValueError("existing work environment differs from this invocation")
    else:
        work.mkdir()
        _fsync_directory(work.parent)
        _atomic_json(environment_path, environment)
        _atomic_json(contract_path, contract)
    if _sha256(contract_path) != run_contract_sha256:
        raise RuntimeError("serialized run contract digest is not canonical")
    results_root = work / "datasets"
    results_root.mkdir(exist_ok=True)
    if results_root.is_symlink() or not results_root.is_dir():
        raise RuntimeError("dataset work directory is unsafe")
    cleaned_transients = _cleanup_transients(work, results_root)

    attempts_path = work / "attempts.json"
    if attempts_path.exists():
        attempts = json.loads(attempts_path.read_text(encoding="utf-8"))
        if not isinstance(attempts, list):
            raise RuntimeError("attempt history is invalid")
    else:
        attempts = []
    attempt = {
        "attempt": len(attempts) + 1,
        "started_at_utc": _utc_now(),
        "status": "started",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cleaned_transients": cleaned_transients,
    }
    attempts.append(attempt)
    _atomic_json(attempts_path, attempts)

    started = time.monotonic()
    stage_directory: Path | None = None
    try:
        stage_directory, staged_checkpoints = _stage_checkpoints(
            checkpoints, expected_digests
        )
        records = []
        for ordinal, name in enumerate(roster):
            destination = results_root / f"{ordinal:04d}"
            if destination.exists():
                record = _validated_cached_record(
                    destination=destination,
                    name=name,
                    ordinal=ordinal,
                    run_contract_sha256=run_contract_sha256,
                    talent_root=talent_root,
                )
            else:
                record = _evaluate_dataset(
                    name=name,
                    ordinal=ordinal,
                    talent_root=talent_root,
                    checkpoints=staged_checkpoints,
                    checkpoint_digests=expected_digests,
                    model_root=model_root,
                    model_sha=model_sha,
                    run_contract_sha256=run_contract_sha256,
                    destination=destination,
                )
            records.append(record)
            print(
                json.dumps(
                    {
                        "completed": ordinal + 1,
                        "total": len(roster),
                        "dataset": name,
                        "rope_accuracy": record["arms"]["rope"]["accuracy"],
                        "fingerprint_accuracy": record["arms"]["fingerprint"][
                            "accuracy"
                        ],
                        "released_accuracy": record["arms"]["released"]["accuracy"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if _git_head(analysis_root, expected=analysis_sha) != analysis_sha:
            raise RuntimeError("analysis source changed during evaluation")
        if _git_head(model_root, expected=model_sha) != model_sha:
            raise RuntimeError("model source changed during evaluation")
        if {
            arm: _sha256(path) for arm, path in checkpoints.items()
        } != expected_digests:
            raise RuntimeError("original checkpoint bytes changed during evaluation")
        if {name: _sha256(path) for name, path in receipts.items()} != lineage[
            "receipt_sha256"
        ]:
            raise RuntimeError(
                "training lineage receipt bytes changed during evaluation"
            )
        if _sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
            raise RuntimeError("TALENT manifest changed during evaluation")
        if _runtime_environment(analysis_root, model_root) != environment:
            raise RuntimeError("runtime environment changed during evaluation")

        summary = {
            **contract,
            "run_contract_sha256": run_contract_sha256,
            "dataset_count": len(records),
            "prediction_rows": sum(int(record["n_evaluation"]) for record in records),
            "aggregate": _aggregate(records),
            "checkpoint_contract": checkpoint_contract,
            "checkpoint_size_bytes": {
                arm: path.stat().st_size for arm, path in checkpoints.items()
            },
            "datasets": records,
        }
        _atomic_json(work / "summary.json", summary)
        duration = time.monotonic() - started
        attempt.update(
            {
                "status": "completed",
                "completed_at_utc": _utc_now(),
                "duration_seconds": duration,
                "completed_datasets": len(records),
            }
        )
        _atomic_json(attempts_path, attempts)
        _atomic_json(
            work / "runtime.json",
            {
                "schema_version": 1,
                "attempt_count": len(attempts),
                "completed_at_utc": attempt["completed_at_utc"],
                "duration_seconds_final_attempt": duration,
                "environment_contract_sha256": contract["environment_contract_sha256"],
            },
        )
        _atomic_json(work / "manifest.json", _artifact_manifest(work, roster))
        os.replace(work, output)
        _fsync_directory(output.parent)
    except BaseException as error:
        attempt.update(
            {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "duration_seconds": time.monotonic() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        if work.is_dir() and not work.is_symlink():
            _atomic_json(attempts_path, attempts)
        raise
    finally:
        if stage_directory is not None:
            shutil.rmtree(stage_directory, ignore_errors=True)
    print(json.dumps({"output_dir": str(output), "dataset_count": len(roster)}))
    return 0


def main() -> int:
    args = _parser().parse_args()
    analysis_root = _absolute(args.analysis_root, name="analysis_root", directory=True)
    model_root = _absolute(args.model_root, name="model_root", directory=True)
    talent_root = _absolute(args.talent_root, name="talent_root", directory=True)
    output_argument = Path(args.output_dir)
    if not output_argument.is_absolute():
        raise ValueError("output_dir must be absolute")
    output_parent = output_argument.parent.resolve(strict=True)
    output = output_parent / output_argument.name
    if output.exists() or output.is_symlink():
        raise ValueError("output_dir must be absent")
    lock_path = output.with_name(f".{output.name}.lock")
    with _RunLock(lock_path):
        return _run_locked(
            args,
            analysis_root=analysis_root,
            model_root=model_root,
            talent_root=talent_root,
            output=output,
        )


if __name__ == "__main__":
    raise SystemExit(main())
