#!/usr/bin/env python3
"""Validate and transact the fresh three-arm formal submission overlay.

The immutable transaction ledger intentionally uses exactly the schema consumed
by :mod:`tabicl.train._provenance`.  Scheduler job IDs live only in a separate
submission receipt or rollback-recovery record; future checkpoint hashes are
not knowable, and therefore are not present, at initial submission time.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import signal
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
ARMS = ("rope", "temporary", "none")
STAGES = (("stage1", 500_000), ("stage2", 40_000), ("stage3", 10_000))
FORMAL_SEEDS = frozenset({42, 43, 44})
WORLD_SIZE = 1
CUDA_DEVICE_COUNT = 1
TREATMENT_SCHEMA_VERSION = 1
SEED_POLICY = "sha256-domain-separated-base-seed-and-rank-v1"
SAMPLER_VERSION = "tabicl-temporary-identity/randperm-cpu-v1"
SCHEDULER_COMMANDS = ("sbatch", "scontrol", "scancel", "squeue", "sacct")
TRANSACTION_JOURNAL_CEILING_BYTES = 2_000_000
RUNTIME_COMPLETION_CEILING_BYTES = 65_536
TERMINAL_LOG_ATTESTATION_CEILING_BYTES = 131_072
SCHEDULER_RECONCILIATION_ATTEMPTS = 3
TERMINAL_SCHEDULER_STATES = frozenset(
    {
        "CANCELLED",
        "COMPLETED",
        "FAILED",
        "TIMEOUT",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
        "REVOKED",
    }
)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_JOB_ID = re.compile(r"^[1-9][0-9]{0,19}$")
_SLURM_DURATION = re.compile(
    r"^(?:(?P<days>[1-9][0-9]{0,3})-)?"
    r"(?P<hours>[0-9]{2}):(?P<minutes>[0-5][0-9]):(?P<seconds>[0-5][0-9])$"
)
_SBATCH_RESULT = re.compile(
    r"^([1-9][0-9]{0,19})(?:;([A-Za-z0-9][A-Za-z0-9._-]{0,63}))?$"
)
_LEDGER_ENTRY_KEYS = {
    "arm",
    "stage",
    "terminal_step",
    "upstream_identity",
    "artifact_identity",
    "checkpoint_relpath",
    "finalized_manifest_relpath",
    "np_seed",
    "torch_seed",
    "identity_rng_seed",
    "world_size",
    "cuda_device_count",
    "max_checkpoint_bytes",
    "source_sha256",
    "environment_sha256",
    "prior_sha256",
    "architecture_sha256",
    "optimizer_sha256",
    "scientific_sha256",
    "cohort_protocol_sha256",
    "arm_protocol_sha256",
}


def _normalize(value: Any, *, where: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where} contains a non-finite number")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or "\x00" in key:
                raise ValueError(f"{where} has an invalid object key")
            result[key] = _normalize(item, where=f"{where}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _normalize(item, where=f"{where}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{where} contains unsupported type {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def make_manifest(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(kind, str) or not kind:
        raise ValueError("manifest kind is invalid")
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "payload": _normalize(payload, where=f"{kind}.payload"),
    }
    return {**body, "sha256": canonical_sha256(body)}


def _exact_keys(value: Any, expected: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{where} keys mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _git_oid(value: Any, where: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        raise ValueError(f"{where} must be a lowercase 40-character Git object ID")
    return value


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _safe_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{where} is invalid")
    return value


def _safe_text(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(character in value for character in ("\x00", "\n", "\r", ","))
    ):
        raise ValueError(f"{where} is invalid")
    return value


def _slurm_duration(value: Any, where: str) -> str:
    """Validate one canonical, finite Slurm ``TimeLimit`` duration.

    Durations below one day use ``HH:MM:SS``.  Durations of one day or more
    use ``D-HH:MM:SS``.  Requiring the normalized form makes the submitted
    value directly comparable with ``scontrol show job -o`` output.
    """

    if not isinstance(value, str):
        raise ValueError(f"{where} must be a canonical Slurm duration")
    match = _SLURM_DURATION.fullmatch(value)
    if match is None:
        raise ValueError(f"{where} must be a canonical Slurm duration")
    days = int(match.group("days") or "0")
    hours = int(match.group("hours"))
    if hours > 23:
        raise ValueError(f"{where} must be normalized as D-HH:MM:SS")
    total_seconds = (
        (days * 24 + hours) * 60 * 60
        + int(match.group("minutes")) * 60
        + int(match.group("seconds"))
    )
    if total_seconds <= 0:
        raise ValueError(f"{where} must be greater than zero")
    return value


def _absolute_path(value: Any, where: str) -> Path:
    raw = _safe_text(value, where)
    path = Path(raw)
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{where} must be an absolute normalized path")
    return path


def _relative_path(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError(f"{where} is invalid")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{where} must be normalized and relative")
    if pure.as_posix() != value:
        raise ValueError(f"{where} must be normalized")
    return value


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return true when either normalized absolute path contains the other."""

    return left == right or left in right.parents or right in left.parents


def _open_directory_nofollow(path: Path, *, where: str) -> int:
    """Open every directory component without following symbolic links."""
    if (
        not path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
    ):
        raise ValueError(f"{where} must be a normalized no-follow absolute directory")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    fd = os.open(os.path.sep, flags)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except Exception:
        os.close(fd)
        raise


def _require_physical_directory(path: Path, *, where: str) -> None:
    try:
        fd = _open_directory_nofollow(path, where=where)
    except OSError as error:
        raise ValueError(f"{where} must exist without symbolic-link components") from error
    os.close(fd)


def _load_filesystem_isolation_helper(exact_root: Path):
    path = exact_root / "scripts/verify_filesystem_isolation.py"
    spec = importlib.util.spec_from_file_location(
        "_formal_submit_filesystem_isolation", path
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load exact-T filesystem isolation helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_git_repository_helper(root: Path):
    path = root / "scripts/verify_git_repository.py"
    spec = importlib.util.spec_from_file_location(
        "_formal_submit_git_repository", path
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load exact-T Git repository helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _require_work_artifact_filesystem_isolation(
    plan: Mapping[str, Any], *, artifact_directory: Path
) -> dict[str, Any]:
    helper = _load_filesystem_isolation_helper(Path(plan["exact_root"]))
    return dict(
        helper.require_distinct_filesystems(
            Path(plan["runtime"]["job_work_root"]),
            artifact_directory,
            work_label="formal job work root",
            artifact_label="formal artifact filesystem",
        )
    )


def _formal_seed(value: Any, where: str = "formal seed") -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in FORMAL_SEEDS
    ):
        raise ValueError(f"{where} must be exactly one of 42, 43, or 44")
    return value


def _identity_treatment(mode: str, *, seed: int) -> dict[str, Any]:
    body = {
        "schema_version": TREATMENT_SCHEMA_VERSION,
        "row_identity_mode": mode,
        "identity_rng_seed": _formal_seed(seed),
        "seed_policy": SEED_POLICY,
        "sampler_version": SAMPLER_VERSION if mode == "temporary" else None,
        "world_size": WORLD_SIZE,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _expected_protocols(
    *,
    seed: int,
    stage: str,
    terminal_step: int,
    source_sha256: str,
    environment_sha256: str,
    prior_sha256: str,
    architecture_sha256: str,
    optimizer_sha256: str,
    scientific_sha256: str,
) -> tuple[str, dict[str, str]]:
    selected_seed = _formal_seed(seed)
    seed_manifest = make_manifest(
        "seed",
        {
            "np_seed": selected_seed,
            "torch_seed": selected_seed,
            "identity_rng_seed": selected_seed,
            "world_size": WORLD_SIZE,
        },
    )
    cohort = make_manifest(
        "cohort_protocol",
        {
            "stage": stage,
            "terminal_step": terminal_step,
            "source_sha256": source_sha256,
            "environment_sha256": environment_sha256,
            "architecture_sha256": architecture_sha256,
            "prior_sha256": prior_sha256,
            "optimizer_sha256": optimizer_sha256,
            "seed_sha256": seed_manifest["sha256"],
            "scientific_config_sha256": scientific_sha256,
        },
    )
    arms = {}
    for mode in ARMS:
        treatment = make_manifest(
            "treatment", _identity_treatment(mode, seed=selected_seed)
        )
        arms[mode] = make_manifest(
            "arm_protocol",
            {
                "cohort_protocol_sha256": cohort["sha256"],
                "mode": mode,
                "treatment_sha256": treatment["sha256"],
            },
        )["sha256"]
    return cohort["sha256"], arms


def _entry_identity(study_id: str, arm: str, stage: str) -> tuple[str, str]:
    return (
        f"{study_id}:{arm}:{stage}",
        f"{study_id}.{arm}.{stage}.final",
    )


def _entry_paths(arm: str, stage: str, terminal_step: int) -> tuple[str, str]:
    prefix = f"arms/{arm}/{stage}"
    return (
        f"{prefix}/step-{terminal_step}.ckpt",
        f"{prefix}/finalized-checkpoint.json",
    )


def validate_overlay(
    overlay: Mapping[str, Any], *, exact_root: str | os.PathLike[str]
) -> dict[str, Any]:
    """Return a canonical nine-job plan after strict pure validation."""
    overlay = _exact_keys(
        overlay,
        {
            "schema_version",
            "run_policy",
            "seed",
            "study_id",
            "artifact_root",
            "source",
            "smoke",
            "capacity",
            "runtime",
            "scheduler",
            "stages",
        },
        "formal overlay",
    )
    if (
        isinstance(overlay["schema_version"], bool)
        or not isinstance(overlay["schema_version"], int)
        or overlay["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("formal overlay schema_version mismatch")
    if overlay["run_policy"] != "fresh":
        raise ValueError("formal overlay is fresh-only")
    seed = _formal_seed(overlay["seed"])
    study_id = _safe_id(overlay["study_id"], "study_id")
    if not study_id.endswith(f"-seed{seed}"):
        raise ValueError("study_id must end with the selected formal seed namespace")
    artifact_root = _absolute_path(overlay["artifact_root"], "artifact_root")
    if artifact_root.name != study_id:
        raise ValueError("artifact_root basename must exactly match study_id")
    root = Path(exact_root)
    if not root.is_absolute():
        raise ValueError("exact_root must be absolute")

    source = _exact_keys(
        overlay["source"],
        {
            "commit_sha",
            "tree_sha",
            "manifest_path",
            "manifest_sha256",
            "environment_sha256",
            "candidate_repository",
            "candidate_ref",
        },
        "source",
    )
    commit_sha = _git_oid(source["commit_sha"], "source commit")
    tree_sha = _git_oid(source["tree_sha"], "source tree")
    source_manifest = _absolute_path(source["manifest_path"], "source manifest")
    source_sha256 = _digest(source["manifest_sha256"], "source manifest")
    environment_sha256 = _digest(
        source["environment_sha256"], "environment sha256"
    )
    candidate_repository = _safe_text(
        source["candidate_repository"], "candidate_repository"
    )
    candidate_ref = _safe_text(source["candidate_ref"], "candidate_ref")
    if candidate_repository != "https://github.com/kikixiong/tabicl-pe.git":
        raise ValueError("candidate_repository must be the canonical public GitHub URL")
    if candidate_ref != "refs/heads/codex/position-identity-v1":
        raise ValueError("candidate_ref must be the canonical training branch")

    smoke = _exact_keys(
        overlay["smoke"],
        {"attestation_path", "expected_sha256", "expected_gpu_model"},
        "smoke",
    )
    smoke_path = _absolute_path(smoke["attestation_path"], "smoke attestation")
    smoke_sha256 = _digest(smoke["expected_sha256"], "smoke expected sha256")
    smoke_gpu_model = _safe_text(smoke["expected_gpu_model"], "smoke GPU model")
    if "H100" not in smoke_gpu_model:
        raise ValueError("smoke GPU model must identify an H100")

    capacity = _exact_keys(
        overlay["capacity"],
        {
            "checkpoint_ceiling_bytes",
            "durable_log_allowance_bytes",
            "run_log_ceiling_bytes",
            "attestation_ceiling_bytes",
            "manifest_ceiling_bytes",
            "protocol_metadata_allowance_bytes",
        },
        "capacity",
    )
    checkpoint_ceiling = _positive_int(
        capacity["checkpoint_ceiling_bytes"], "checkpoint ceiling"
    )
    durable_log_allowance = _positive_int(
        capacity["durable_log_allowance_bytes"], "durable log allowance"
    )
    run_log_ceiling = _positive_int(
        capacity["run_log_ceiling_bytes"], "run log ceiling"
    )
    attestation_ceiling = _positive_int(
        capacity["attestation_ceiling_bytes"], "attestation ceiling"
    )
    manifest_ceiling = _positive_int(
        capacity["manifest_ceiling_bytes"], "manifest ceiling"
    )
    protocol_metadata_allowance = _positive_int(
        capacity["protocol_metadata_allowance_bytes"],
        "protocol metadata allowance",
    )
    assigned_durable = (
        4 * len(ARMS) * len(STAGES) * run_log_ceiling
        + 2 * len(ARMS) * len(STAGES) * attestation_ceiling
        + len(ARMS) * len(STAGES) * manifest_ceiling
        + protocol_metadata_allowance
    )
    if assigned_durable > durable_log_allowance:
        raise ValueError("formal durable-output sub-budgets exceed aggregate allowance")

    runtime = _exact_keys(
        overlay["runtime"],
        {"python", "git", "git_sha256", "nvidia_smi", "job_work_root"},
        "runtime",
    )
    python = _absolute_path(runtime["python"], "runtime python")
    git = _absolute_path(runtime["git"], "runtime git")
    git_sha256 = _digest(runtime["git_sha256"], "runtime git sha256")
    nvidia_smi = _absolute_path(runtime["nvidia_smi"], "runtime nvidia_smi")
    job_work_root = _absolute_path(runtime["job_work_root"], "job work root")
    if _paths_overlap(job_work_root, root):
        raise ValueError("job_work_root must be disjoint from exact_root")
    if _paths_overlap(job_work_root, artifact_root):
        raise ValueError("job_work_root must be disjoint from artifact_root")

    scheduler = _exact_keys(
        overlay["scheduler"],
        {
            "partition",
            "qos",
            "cpus_per_task",
            "memory_mb",
            "gpus_per_job",
            "time_limit_by_stage",
            "commands",
        },
        "scheduler",
    )
    partition = _safe_text(scheduler["partition"], "scheduler partition")
    if partition != "h100":
        raise ValueError("formal scheduler partition must be exactly h100")
    qos = _safe_text(scheduler["qos"], "scheduler qos")
    if qos != "long":
        raise ValueError("formal scheduler qos must be exactly long")
    cpus_per_task = _positive_int(
        scheduler["cpus_per_task"], "scheduler cpus_per_task"
    )
    if cpus_per_task != 64:
        raise ValueError("formal scheduler cpus_per_task must be exactly 64")
    memory_mb = _positive_int(scheduler["memory_mb"], "scheduler memory_mb")
    if memory_mb != 131_072:
        raise ValueError("formal scheduler memory_mb must be exactly 131072")
    if (
        isinstance(scheduler["gpus_per_job"], bool)
        or not isinstance(scheduler["gpus_per_job"], int)
        or scheduler["gpus_per_job"] != 1
    ):
        raise ValueError("formal production jobs require exactly one H100")
    raw_time_limits = _exact_keys(
        scheduler["time_limit_by_stage"],
        {stage for stage, _terminal_step in STAGES},
        "scheduler time_limit_by_stage",
    )
    time_limit_by_stage = {
        stage: _slurm_duration(
            raw_time_limits[stage], f"scheduler {stage} time limit"
        )
        for stage, _terminal_step in STAGES
    }
    raw_commands = _exact_keys(
        scheduler["commands"], set(SCHEDULER_COMMANDS), "scheduler commands"
    )
    scheduler_commands: dict[str, dict[str, str]] = {}
    for command_name in SCHEDULER_COMMANDS:
        command = _exact_keys(
            raw_commands[command_name], {"path", "sha256"},
            f"scheduler command {command_name}",
        )
        scheduler_commands[command_name] = {
            "path": str(
                _absolute_path(command["path"], f"scheduler {command_name} path")
            ),
            "sha256": _digest(
                command["sha256"], f"scheduler {command_name} sha256"
            ),
        }

    repository_helper = _load_git_repository_helper(Path(__file__).parents[1])
    repository_binding = dict(
        repository_helper.expected_repository_binding(
            expected_commit_sha=commit_sha,
            git_sha256=git_sha256,
        )
    )

    raw_stages = overlay["stages"]
    if not isinstance(raw_stages, list) or len(raw_stages) != len(STAGES):
        raise ValueError("stages must contain exactly stage1, stage2, and stage3")
    stages_by_name: dict[str, Mapping[str, Any]] = {}
    for stage_value in raw_stages:
        stage_value = _exact_keys(
            stage_value,
            {
                "stage",
                "terminal_step",
                "prior_sha256",
                "architecture_sha256",
                "optimizer_sha256",
                "scientific_sha256",
                "cohort_protocol_sha256",
                "arm_protocol_sha256",
            },
            "stage protocol",
        )
        stage_name = stage_value["stage"]
        if stage_name in stages_by_name:
            raise ValueError("stages contain a duplicate stage")
        stages_by_name[stage_name] = stage_value
    if set(stages_by_name) != {stage for stage, _ in STAGES}:
        raise ValueError("stages must be the exact canonical stage set")

    entries: list[dict[str, Any]] = []
    stage_protocols: dict[str, dict[str, Any]] = {}
    for stage_name, budget in STAGES:
        stage_value = stages_by_name[stage_name]
        if stage_value["terminal_step"] != budget:
            raise ValueError(f"{stage_name} budget must be exactly {budget}")
        prior = _digest(stage_value["prior_sha256"], f"{stage_name} prior")
        architecture = _digest(
            stage_value["architecture_sha256"], f"{stage_name} architecture"
        )
        optimizer = _digest(
            stage_value["optimizer_sha256"], f"{stage_name} optimizer"
        )
        scientific = _digest(
            stage_value["scientific_sha256"], f"{stage_name} scientific"
        )
        supplied_cohort = _digest(
            stage_value["cohort_protocol_sha256"],
            f"{stage_name} cohort_protocol",
        )
        arm_protocol = _exact_keys(
            stage_value["arm_protocol_sha256"], set(ARMS), f"{stage_name} arms"
        )
        expected_cohort, expected_arms = _expected_protocols(
            seed=seed,
            stage=stage_name,
            terminal_step=budget,
            source_sha256=source_sha256,
            environment_sha256=environment_sha256,
            prior_sha256=prior,
            architecture_sha256=architecture,
            optimizer_sha256=optimizer,
            scientific_sha256=scientific,
        )
        if supplied_cohort != expected_cohort:
            raise ValueError(
                f"{stage_name} cohort_protocol_sha256 is not the canonical shared protocol"
            )
        supplied_arms = {
            arm: _digest(arm_protocol[arm], f"{stage_name} {arm} arm_protocol")
            for arm in ARMS
        }
        if supplied_arms != expected_arms:
            raise ValueError(
                f"{stage_name} arm_protocol_sha256 is not derived solely from "
                "the cohort and canonical identity treatment"
            )
        stage_protocols[stage_name] = {
            "terminal_step": budget,
            "prior_sha256": prior,
            "architecture_sha256": architecture,
            "optimizer_sha256": optimizer,
            "scientific_sha256": scientific,
            "cohort_protocol_sha256": supplied_cohort,
            "arm_protocol_sha256": supplied_arms,
        }

    for stage_name, budget in STAGES:
        for arm in ARMS:
            upstream, artifact = _entry_identity(study_id, arm, stage_name)
            checkpoint_relpath, final_relpath = _entry_paths(
                arm, stage_name, budget
            )
            protocol = stage_protocols[stage_name]
            entries.append(
                {
                    "arm": arm,
                    "stage": stage_name,
                    "terminal_step": budget,
                    "upstream_identity": _safe_id(upstream, "upstream_identity"),
                    "artifact_identity": _safe_id(artifact, "artifact_identity"),
                    "checkpoint_relpath": _relative_path(
                        checkpoint_relpath, "checkpoint_relpath"
                    ),
                    "finalized_manifest_relpath": _relative_path(
                        final_relpath, "finalized_manifest_relpath"
                    ),
                    "np_seed": seed,
                    "torch_seed": seed,
                    "identity_rng_seed": seed,
                    "world_size": WORLD_SIZE,
                    "cuda_device_count": CUDA_DEVICE_COUNT,
                    "max_checkpoint_bytes": checkpoint_ceiling,
                    "source_sha256": source_sha256,
                    "environment_sha256": environment_sha256,
                    "prior_sha256": protocol["prior_sha256"],
                    "architecture_sha256": protocol["architecture_sha256"],
                    "optimizer_sha256": protocol["optimizer_sha256"],
                    "scientific_sha256": protocol["scientific_sha256"],
                    "cohort_protocol_sha256": protocol[
                        "cohort_protocol_sha256"
                    ],
                    "arm_protocol_sha256": protocol["arm_protocol_sha256"][arm],
                }
            )
    if any(set(entry) != _LEDGER_ENTRY_KEYS for entry in entries):
        raise AssertionError("internal ledger schema drift")
    checkpoint_paths = {entry["checkpoint_relpath"] for entry in entries}
    final_paths = {entry["finalized_manifest_relpath"] for entry in entries}
    if len(checkpoint_paths) != 9 or len(final_paths) != 9:
        raise ValueError("formal artifact paths must be unique")

    ledger = make_manifest(
        "transaction_ledger", {"study_id": study_id, "entries": entries}
    )
    ledger_path = artifact_root / "transaction-ledger.json"
    scheduler_log_root = artifact_root / "scheduler-logs"
    completion_root = artifact_root / "runtime-completions"
    terminal_log_attestation_path = artifact_root / "terminal-scheduler-logs.json"
    entries_by_key = {(entry["arm"], entry["stage"]): entry for entry in entries}
    jobs = []
    for stage_index, (stage_name, budget) in enumerate(STAGES, start=1):
        predecessor = STAGES[stage_index - 2][0] if stage_index > 1 else None
        for arm in ARMS:
            entry = entries_by_key[(arm, stage_name)]
            checkpoint_dir = artifact_root / "arms" / arm / stage_name
            job_slug = f"{arm}-{stage_name}"
            scheduler_stdout = scheduler_log_root / f"{job_slug}.out"
            scheduler_stderr = scheduler_log_root / f"{job_slug}.err"
            completion_path = completion_root / f"{job_slug}.json"
            exports = {
                "RUN_POLICY": "fresh",
                "MODE": arm,
                "STAGE": str(stage_index),
                "NUM_GPUS": "1",
                "PYTHON": str(python),
                "GIT": str(git),
                "NVIDIA_SMI": str(nvidia_smi),
                "PYTHONPATH": str(root / "src"),
                "PYTHONNOUSERSITE": "1",
                "CANDIDATE_REPOSITORY": candidate_repository,
                "CANDIDATE_REPOSITORY_REF": candidate_ref,
                "FORMAL_SOURCE_COMMIT_SHA": commit_sha,
                "FORMAL_SOURCE_TREE_SHA": tree_sha,
                "FORMAL_GIT_SHA256": git_sha256,
                "FORMAL_REPOSITORY_IDENTITY_SHA256": repository_binding[
                    "repository_identity_sha256"
                ],
                "FORMAL_REPOSITORY_QUERY_SHA256": repository_binding[
                    "query_sha256"
                ],
                "FORMAL_JOB_WORK_ROOT": str(job_work_root),
                "FORMAL_ARTIFACT_ROOT": str(artifact_root),
                "FORMAL_SUBMISSION_EXACT_ROOT": str(root),
                "FORMAL_CHECKPOINT_DIR": str(checkpoint_dir),
                "FORMAL_SOURCE_MANIFEST": str(source_manifest),
                "FORMAL_SOURCE_SHA256": source_sha256,
                "FORMAL_ENVIRONMENT_SHA256": environment_sha256,
                "FORMAL_SEED": str(seed),
                "FORMAL_STUDY_ID": study_id,
                "FORMAL_OUTPUT_ID": f"{study_id}-{arm}-{stage_name}",
                "FORMAL_PRIOR_SHA256": entry["prior_sha256"],
                "FORMAL_ARCHITECTURE_SHA256": entry["architecture_sha256"],
                "FORMAL_OPTIMIZER_SHA256": entry["optimizer_sha256"],
                "FORMAL_SCIENTIFIC_SHA256": entry["scientific_sha256"],
                "FORMAL_COHORT_PROTOCOL_SHA256": entry[
                    "cohort_protocol_sha256"
                ],
                "FORMAL_ARM_PROTOCOL_SHA256": entry["arm_protocol_sha256"],
                "FORMAL_TRANSACTION_LEDGER": str(ledger_path),
                "FORMAL_TRANSACTION_LEDGER_SHA256": ledger["sha256"],
                "FORMAL_SUBMISSION_RECEIPT": str(
                    artifact_root / "submission-receipt.json"
                ),
                "FORMAL_COMPLETION_EVIDENCE": str(completion_path),
                "FORMAL_RUNTIME_COMPLETION_CEILING_BYTES": str(
                    RUNTIME_COMPLETION_CEILING_BYTES
                ),
                "FORMAL_SCHEDULER_STDOUT": str(scheduler_stdout),
                "FORMAL_SCHEDULER_STDERR": str(scheduler_stderr),
                "FORMAL_TIME_LIMIT": time_limit_by_stage[stage_name],
                "FORMAL_SCONTROL": scheduler_commands["scontrol"]["path"],
                "FORMAL_SCONTROL_SHA256": scheduler_commands["scontrol"][
                    "sha256"
                ],
                "FORMAL_UPSTREAM_IDENTITY": entry["upstream_identity"],
                "FORMAL_ARTIFACT_IDENTITY": entry["artifact_identity"],
                "CHECKPOINT_CEILING_BYTES": str(checkpoint_ceiling),
                "DURABLE_LOG_ALLOWANCE_BYTES": str(durable_log_allowance),
                "FORMAL_RUN_LOG_CEILING_BYTES": str(run_log_ceiling),
                "FORMAL_ATTESTATION_CEILING_BYTES": str(attestation_ceiling),
                "FORMAL_MANIFEST_CEILING_BYTES": str(manifest_ceiling),
                "FORMAL_PROTOCOL_METADATA_ALLOWANCE_BYTES": str(
                    protocol_metadata_allowance
                ),
            }
            parent_job = None
            if predecessor is not None:
                parent = entries_by_key[(arm, predecessor)]
                exports.update(
                    {
                        "FORMAL_PARENT_CHECKPOINT": str(
                            artifact_root / parent["checkpoint_relpath"]
                        ),
                        "FORMAL_PARENT_FINALIZED_MANIFEST": str(
                            artifact_root / parent["finalized_manifest_relpath"]
                        ),
                        "FORMAL_PARENT_UPSTREAM_IDENTITY": parent[
                            "upstream_identity"
                        ],
                        "FORMAL_PARENT_ARTIFACT_IDENTITY": parent[
                            "artifact_identity"
                        ],
                    }
                )
                parent_job = (arm, predecessor)
            if any(
                not isinstance(value, str)
                or not value
                or any(c in value for c in ("\x00", "\n", "\r", ","))
                for value in exports.values()
            ):
                raise ValueError("job export values must be non-empty and comma-free")
            jobs.append(
                {
                    "arm": arm,
                    "stage": stage_name,
                    "stage_index": stage_index,
                    "terminal_step": budget,
                    "time_limit": time_limit_by_stage[stage_name],
                    "gpus": 1,
                    "parent_job": parent_job,
                    "scheduler_stdout": str(scheduler_stdout),
                    "scheduler_stderr": str(scheduler_stderr),
                    "completion_path": str(completion_path),
                    "exports": exports,
                }
            )

    plan = {
        "schema_version": SCHEMA_VERSION,
        "run_policy": "fresh",
        "seed": seed,
        "study_id": study_id,
        "artifact_root": str(artifact_root),
        "ledger_path": str(ledger_path),
        "rollback_path": str(artifact_root / "rollback-incomplete.json"),
        "submission_receipt_path": str(
            artifact_root / "submission-receipt.json"
        ),
        "terminal_log_attestation_path": str(terminal_log_attestation_path),
        "transaction_commit_path": str(
            artifact_root / "transaction-committed.json"
        ),
        "exact_root": str(root),
        "source": {
            "commit_sha": commit_sha,
            "tree_sha": tree_sha,
            "manifest_path": str(source_manifest),
            "manifest_sha256": source_sha256,
            "environment_sha256": environment_sha256,
        },
        "repository_binding": repository_binding,
        "smoke": {
            "attestation_path": str(smoke_path),
            "expected_sha256": smoke_sha256,
            "expected_gpu_model": smoke_gpu_model,
        },
        "capacity": {
            "checkpoint_ceiling_bytes": checkpoint_ceiling,
            "durable_log_allowance_bytes": durable_log_allowance,
            "run_log_ceiling_bytes": run_log_ceiling,
            "attestation_ceiling_bytes": attestation_ceiling,
            "manifest_ceiling_bytes": manifest_ceiling,
            "protocol_metadata_allowance_bytes": protocol_metadata_allowance,
            "runtime_completion_ceiling_bytes": RUNTIME_COMPLETION_CEILING_BYTES,
            "terminal_log_attestation_ceiling_bytes": (
                TERMINAL_LOG_ATTESTATION_CEILING_BYTES
            ),
        },
        "runtime": {
            "python": str(python),
            "git": str(git),
            "git_sha256": git_sha256,
            "nvidia_smi": str(nvidia_smi),
            "job_work_root": str(job_work_root),
        },
        "scheduler": {
            "partition": partition,
            "qos": qos,
            "cpus_per_task": cpus_per_task,
            "memory_mb": memory_mb,
            "gpus_per_job": 1,
            "time_limit_by_stage": time_limit_by_stage,
            "commands": scheduler_commands,
        },
        "ledger": ledger,
        "jobs": jobs,
    }
    _validate_protocol_metadata_budget(plan)
    return plan


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("short write")
        offset += written


def _publish_no_replace(
    path: Path, payload: bytes, *, fault: str | None = None
) -> Path:
    if fault not in {None, "write", "fsync", "rename", "dir_fsync"}:
        raise ValueError("unknown publication fault")
    parent = path.parent
    dir_fd = _open_directory_nofollow(parent, where="publication parent")
    temp_name = f".{path.name}.{secrets.token_hex(16)}.tmp"
    file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        file_flags |= os.O_CLOEXEC
    fd = -1
    published = False
    try:
        fd = os.open(temp_name, file_flags, 0o444, dir_fd=dir_fd)
        os.fchmod(fd, 0o444)
        if fault == "write":
            raise OSError("injected ledger write failure")
        _write_all(fd, payload)
        if fault == "fsync":
            raise OSError("injected ledger fsync failure")
        os.fsync(fd)
        os.close(fd)
        fd = -1
        if fault == "rename":
            raise OSError("injected ledger rename failure")
        os.link(
            temp_name,
            path.name,
            src_dir_fd=dir_fd,
            dst_dir_fd=dir_fd,
            follow_symlinks=False,
        )
        published = True
        if fault == "dir_fsync":
            raise OSError("injected ledger directory fsync failure")
        os.fsync(dir_fd)
        os.unlink(temp_name, dir_fd=dir_fd)
        os.fsync(dir_fd)
        return path
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temp_name, dir_fd=dir_fd)
        except FileNotFoundError:
            pass
        finally:
            # If publication succeeded and the directory fsync failed, the
            # caller still receives failure and must roll jobs back.  Never
            # remove the published, complete, write-once final.
            _ = published
            os.close(dir_fd)


def publish_ledger(plan: Mapping[str, Any], *, fault: str | None = None) -> Path:
    path = Path(plan["ledger_path"])
    return _publish_no_replace(
        path, canonical_json_bytes(plan["ledger"]) + b"\n", fault=fault
    )


def _job_ids(values: Sequence[str], where: str) -> list[str]:
    result = list(values)
    if any(not isinstance(value, str) or _JOB_ID.fullmatch(value) is None for value in result):
        raise ValueError(f"{where} contains an invalid job ID")
    if len(result) != len(set(result)):
        raise ValueError(f"{where} contains duplicate job IDs")
    return result


def _parse_sbatch_result(raw: str) -> tuple[str, str | None]:
    """Normalize documented ``--parsable`` JOBID[;CLUSTER] output."""

    match = _SBATCH_RESULT.fullmatch(raw)
    if match is None:
        raise ValueError("sbatch returned an empty or malformed job ID")
    return match.group(1), match.group(2)


def _submission_receipt(
    plan: Mapping[str, Any], submitted: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    job_ids = [item["job_id"] for item in submitted]
    _job_ids(job_ids, "submission receipt job_ids")
    jobs = []
    for item in submitted:
        job = item["job"]
        jobs.append(
            {
                "arm": job["arm"],
                "stage": job["stage"],
                "terminal_step": job["terminal_step"],
                "time_limit": job["time_limit"],
                "job_id": item["job_id"],
                "cluster": item.get("cluster"),
                "job_name": item["job_name"],
                "parent_job_id": item["parent_job_id"],
                "scheduler_stdout": job["scheduler_stdout"],
                "scheduler_stderr": job["scheduler_stderr"],
                "completion_path": job["completion_path"],
            }
        )
    return make_manifest(
        "held_submission_receipt",
        {
            "study_id": plan["study_id"],
            "seed": plan["seed"],
            "transaction_id": plan.get("transaction_id", "f" * 32),
            "transaction_ledger_sha256": plan["ledger"]["sha256"],
            "source_commit_sha": plan["source"]["commit_sha"],
            "source_tree_sha": plan["source"]["tree_sha"],
            "repository_binding": dict(plan["repository_binding"]),
            "jobs_held_at_publication": True,
            "run_log_ceiling_bytes": plan["capacity"]["run_log_ceiling_bytes"],
            "manifest_ceiling_bytes": plan["capacity"]["manifest_ceiling_bytes"],
            "runtime_completion_ceiling_bytes": RUNTIME_COMPLETION_CEILING_BYTES,
            "terminal_log_attestation_path": plan[
                "terminal_log_attestation_path"
            ],
            "transaction_commit_path": plan["transaction_commit_path"],
            "terminal_log_attestation_ceiling_bytes": (
                TERMINAL_LOG_ATTESTATION_CEILING_BYTES
            ),
            "scheduler": {
                key: plan["scheduler"][key]
                for key in (
                    "partition",
                    "qos",
                    "cpus_per_task",
                    "memory_mb",
                    "gpus_per_job",
                    "time_limit_by_stage",
                )
            }
            | {
                "sacct_path": plan["scheduler"]["commands"]["sacct"]["path"],
                "sacct_sha256": plan["scheduler"]["commands"]["sacct"][
                    "sha256"
                ],
                "scontrol_sha256": plan["scheduler"]["commands"]["scontrol"][
                    "sha256"
                ]
            },
            "job_ids": job_ids,
            "jobs": jobs,
        },
    )


def _submission_commit(
    plan: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    submitted: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    job_ids = [item["job_id"] for item in submitted]
    _job_ids(job_ids, "submission commit job_ids")
    return make_manifest(
        "formal_submission_commit",
        {
            "study_id": plan["study_id"],
            "transaction_id": plan.get("transaction_id", "f" * 32),
            "submission_receipt_sha256": receipt["sha256"],
            "transaction_ledger_sha256": plan["ledger"]["sha256"],
            "job_ids": job_ids,
        },
    )


def _rollback_record(
    plan: Mapping[str, Any],
    *,
    accepted: Sequence[str],
    cancelled: Sequence[str],
    remaining: Sequence[str],
    reason: str,
    ledger_published: bool,
    unresolved_job_names: Sequence[str] = (),
) -> dict[str, Any]:
    return make_manifest(
        "rollback_incomplete",
        {
            "study_id": plan["study_id"],
            "transaction_id": plan.get("transaction_id", "f" * 32),
            "reason": _safe_id(reason, "rollback reason"),
            "accepted_job_ids": list(accepted),
            "cancelled_job_ids": list(cancelled),
            "remaining_job_ids": list(remaining),
            "unresolved_job_names": list(unresolved_job_names),
            "ledger_published": bool(ledger_published),
        },
    )


def _terminal_artifact_relative_path(
    plan: Mapping[str, Any], value: Any, *, where: str
) -> str:
    """Encode one canonical internal artifact path independently of its parent."""

    root = _absolute_path(plan["artifact_root"], "artifact_root")
    path = _absolute_path(value, where)
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{where} escaped the formal artifact namespace") from error
    return _relative_path(relative.as_posix(), where)


def _maximal_terminal_attestation(
    plan: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    submitted: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Synthesize the largest schema-valid terminal record for this plan.

    The terminal controller records only bounded strings, digests, sizes, and
    canonical artifact-root-relative paths.  Constructing every one of those
    fields here makes the 128-KiB publication limit a submission-time gate,
    rather than a possible failure after all nine allocations have finished.
    """

    if len(submitted) != len(ARMS) * len(STAGES):
        raise ValueError("terminal attestation preflight requires exactly nine jobs")
    entries = plan["ledger"]["payload"]["entries"]
    entries_by_pair = {
        (entry["arm"], entry["stage"]): entry for entry in entries
    }
    submitted_by_pair = {
        (item["job"]["arm"], item["job"]["stage"]): item
        for item in submitted
    }
    expected_pairs = {(arm, stage) for stage, _step in STAGES for arm in ARMS}
    if (
        len(entries_by_pair) != len(expected_pairs)
        or set(entries_by_pair) != expected_pairs
        or len(submitted_by_pair) != len(expected_pairs)
        or set(submitted_by_pair) != expected_pairs
    ):
        raise ValueError("terminal attestation preflight matrix is incomplete")

    digest = "f" * 64
    cluster = "c" * 64
    scheduler_resources = {
        key: plan["scheduler"][key]
        for key in (
            "partition",
            "qos",
            "cpus_per_task",
            "memory_mb",
            "gpus_per_job",
        )
    }
    repository_binding = dict(plan["repository_binding"])
    checkpoint_ceiling = plan["capacity"]["checkpoint_ceiling_bytes"]
    manifest_ceiling = plan["capacity"]["manifest_ceiling_bytes"]
    run_log_ceiling = plan["capacity"]["run_log_ceiling_bytes"]

    def binding(entry: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "finalized_manifest_path": _relative_path(
                entry["finalized_manifest_relpath"],
                "terminal finalized manifest path",
            ),
            "finalized_manifest_sha256": digest,
            "checkpoint_path": _relative_path(
                entry["checkpoint_relpath"], "terminal checkpoint path"
            ),
            "checkpoint_sha256": digest,
            "checkpoint_size": checkpoint_ceiling,
            "provenance_sha256": digest,
            "seed_sha256": digest,
            "treatment_sha256": digest,
        }

    final_jobs: list[dict[str, Any]] = []
    for stage_index, (stage, _terminal_step) in enumerate(STAGES):
        for arm in ARMS:
            item = submitted_by_pair[(arm, stage)]
            job = item["job"]
            entry = entries_by_pair[(arm, stage)]
            completion_scheduler = {
                **scheduler_resources,
                "time_limit": job["time_limit"],
                "query_sha256": digest,
            }
            artifact_binding = binding(entry)
            completion_path = _terminal_artifact_relative_path(
                plan, job["completion_path"], where="terminal completion path"
            )
            scheduler_logs = []
            for stream, key in (
                ("stdout", "scheduler_stdout"),
                ("stderr", "scheduler_stderr"),
            ):
                scheduler_logs.append(
                    {
                        "stream": stream,
                        "path": _terminal_artifact_relative_path(
                            plan,
                            job[key],
                            where=f"terminal scheduler {stream} path",
                        ),
                        "size": run_log_ceiling,
                        "sha256": digest,
                        "fatal_signature_scan_passed": True,
                        "stable_reads": 2,
                    }
                )
            parent_lineage = None
            if stage_index:
                parent_stage = STAGES[stage_index - 1][0]
                parent = submitted_by_pair[(arm, parent_stage)]
                parent_lineage = {
                    **binding(entries_by_pair[(arm, parent_stage)]),
                    "job_id": parent["job_id"],
                    "stage": parent_stage,
                }
            final_jobs.append(
                {
                    "arm": arm,
                    "stage": stage,
                    "job_id": item["job_id"],
                    "cluster": cluster,
                    "job_name": item["job_name"],
                    "state": "COMPLETED",
                    "exit_code": "0:0",
                    "derived_exit_code": "0:0",
                    "completion": {
                        "path": completion_path,
                        "manifest_sha256": digest,
                        "file_sha256": digest,
                        "stdout_observed_size": run_log_ceiling,
                        "stderr_observed_size": run_log_ceiling,
                        "seed": plan["seed"],
                        "scheduler": completion_scheduler,
                        "cuda_visible_devices": "d" * 128,
                        # The terminal reader bounds this field by Unicode
                        # code points.  JSON control-character escaping is
                        # six bytes per accepted character, so NUL is the
                        # conservative serialized maximum.
                        "gpu_name": "H100" + "\x00" * 252,
                        "gpu_uuid": "GPU-" + "u" * 192,
                        "repository_binding": repository_binding,
                        "manifest_ceiling_bytes": manifest_ceiling,
                    },
                    "scheduler_logs": scheduler_logs,
                    "finalized_artifact": {
                        "binding": artifact_binding,
                        "finalized_manifest_file_sha256": digest,
                        "finalized_manifest_size": manifest_ceiling,
                        "independent_stable_validations": 2,
                        "initial_validation_sha256": digest,
                        "publication_validation_sha256": digest,
                    },
                    "parent_lineage": parent_lineage,
                }
            )

    maximal_ids = [item["job_id"] for item in submitted]
    payload = {
        "study_id": plan["study_id"],
        "seed": plan["seed"],
        "transaction_id": receipt["payload"]["transaction_id"],
        "submission_receipt_sha256": receipt["sha256"],
        "transaction_ledger_sha256": plan["ledger"]["sha256"],
        "source_commit_sha": plan["source"]["commit_sha"],
        "source_tree_sha": plan["source"]["tree_sha"],
        "repository_binding": repository_binding,
        "manifest_ceiling_bytes": manifest_ceiling,
        "sacct_sha256": plan["scheduler"]["commands"]["sacct"]["sha256"],
        "path_format": "artifact_root_relative_posix_v1",
        "terminal_verified": True,
        "terminal_queries": [
            {
                "cluster": cluster,
                "job_ids": maximal_ids,
                "query_argv_sha256": digest,
                "stdout_sha256": digest,
                "stderr_sha256": digest,
            }
        ],
        "jobs": final_jobs,
    }
    return make_manifest("formal_terminal_scheduler_logs", payload)


def _validate_protocol_metadata_budget(plan: Mapping[str, Any]) -> None:
    # Slurm's public job identifier is bounded to twenty decimal digits here;
    # construct the largest durable success and recovery records up front so
    # the overlay's L partition covers every controller-produced metadata byte.
    maximal_ids = [str(10_000_000_000_000_000_000 + index) for index in range(9)]
    ids_by_key: dict[tuple[str, str], str] = {}
    submitted = []
    for job, job_id in zip(plan["jobs"], maximal_ids):
        parent_key = job["parent_job"]
        parent_id = ids_by_key[parent_key] if parent_key is not None else None
        ids_by_key[(job["arm"], job["stage"])] = job_id
        submitted.append(
            {
                "job": job,
                "job_id": job_id,
                "cluster": "cluster-max",
                "job_name": f"tabicl-{'f' * 32}-{job['arm']}-s{job['stage_index']}",
                "parent_job_id": parent_id,
            }
        )
    receipt = _submission_receipt(plan, submitted)
    commit = _submission_commit(plan, receipt=receipt, submitted=submitted)
    terminal_attestation = _maximal_terminal_attestation(
        plan, receipt=receipt, submitted=submitted
    )
    terminal_attestation_bytes = len(
        canonical_json_bytes(terminal_attestation)
    ) + 1
    if terminal_attestation_bytes > TERMINAL_LOG_ATTESTATION_CEILING_BYTES:
        raise ValueError(
            "terminal scheduler-log attestation maximum exceeds its byte "
            f"ceiling ({terminal_attestation_bytes} > "
            f"{TERMINAL_LOG_ATTESTATION_CEILING_BYTES})"
        )
    recovery = max(
        (
            _rollback_record(
                plan,
                accepted=maximal_ids,
                cancelled=(),
                remaining=tuple(reversed(maximal_ids)),
                reason=reason,
                ledger_published=True,
                unresolved_job_names=(f"tabicl-{'f' * 32}-rope-s1",),
            )
            for reason in (
                "sbatch_failed",
                "capacity_recheck_failed",
                "ledger_publication_failed",
                "receipt_publication_failed",
                "release_failed",
                "journal_seal_failed",
                "commit_publication_failed",
            )
        ),
        key=lambda value: len(canonical_json_bytes(value)),
    )
    required = (
        TRANSACTION_JOURNAL_CEILING_BYTES
        + len(plan["jobs"]) * RUNTIME_COMPLETION_CEILING_BYTES
        + TERMINAL_LOG_ATTESTATION_CEILING_BYTES
        + sum(
        len(canonical_json_bytes(value)) + 1
        for value in (plan["ledger"], receipt, commit, recovery)
        )
    )
    allowance = plan["capacity"]["protocol_metadata_allowance_bytes"]
    if required > allowance:
        raise ValueError(
            "protocol metadata allowance is smaller than the maximum durable "
            f"controller metadata ({required} bytes)"
        )


def publish_rollback_incomplete(
    plan: Mapping[str, Any],
    *,
    accepted_job_ids: Sequence[str],
    cancelled_job_ids: Sequence[str],
    remaining_job_ids: Sequence[str],
    reason: str,
    ledger_published: bool,
    unresolved_job_names: Sequence[str] = (),
) -> Path:
    accepted = _job_ids(accepted_job_ids, "accepted_job_ids")
    cancelled = _job_ids(cancelled_job_ids, "cancelled_job_ids")
    remaining = _job_ids(remaining_job_ids, "remaining_job_ids")
    if set(cancelled) | set(remaining) != set(accepted):
        raise ValueError("cancelled and remaining IDs must partition accepted IDs")
    if set(cancelled) & set(remaining):
        raise ValueError("cancelled and remaining IDs overlap")
    unresolved = [
        _safe_id(name, "unresolved job name") for name in unresolved_job_names
    ]
    if len(unresolved) != len(set(unresolved)):
        raise ValueError("unresolved job names contain duplicates")
    record = _rollback_record(
        plan,
        accepted=accepted,
        cancelled=cancelled,
        remaining=remaining,
        reason=reason,
        ledger_published=ledger_published,
        unresolved_job_names=unresolved,
    )
    return _publish_no_replace(
        Path(plan["rollback_path"]), canonical_json_bytes(record) + b"\n"
    )


def _publish_submission_receipt(
    plan: Mapping[str, Any], submitted: Sequence[Mapping[str, Any]]
) -> tuple[Path, dict[str, Any]]:
    receipt = _submission_receipt(plan, submitted)
    path = _publish_no_replace(
        Path(plan["submission_receipt_path"]),
        canonical_json_bytes(receipt) + b"\n",
    )
    return path, receipt


def _publish_submission_commit(
    plan: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    submitted: Sequence[Mapping[str, Any]],
    fault: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    commit = _submission_commit(plan, receipt=receipt, submitted=submitted)
    path = _publish_no_replace(
        Path(plan["transaction_commit_path"]),
        canonical_json_bytes(commit) + b"\n",
        fault=fault,
    )
    return path, commit


def _remove_submission_commit(plan: Mapping[str, Any]) -> None:
    """Durably revoke a commit marker exposed by a failed publication."""

    path = Path(plan["transaction_commit_path"])
    parent_fd = _open_directory_nofollow(
        path.parent, where="transaction commit parent"
    )
    try:
        try:
            metadata = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("transaction commit marker is not a regular file")
        os.unlink(path.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _minimal_env(*, path: str, pythonpath: str | None = None) -> dict[str, str]:
    environment = {"PATH": path, "LC_ALL": "C", "LANG": "C"}
    if pythonpath is not None:
        environment.update(
            {"PYTHONPATH": pythonpath, "PYTHONNOUSERSITE": "1"}
        )
    return environment


def _run(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    where: str,
    allowed_returncodes: set[int] = {0},
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(argv),
        env=dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode not in allowed_returncodes:
        raise RuntimeError(f"{where} failed with status {completed.returncode}")
    return completed


def _attest_exact_checkout(plan: Mapping[str, Any]) -> None:
    root = Path(plan["exact_root"])
    physical_root = root.resolve(strict=True)
    if physical_root != root or not physical_root.is_dir():
        raise ValueError("exact-T root must be an absolute physical directory")
    _require_physical_directory(root, where="exact-T root")
    expected_helper = root / "scripts" / "verify_formal_overlay.py"
    if Path(__file__).resolve(strict=True) != expected_helper.resolve(strict=True):
        raise ValueError("formal overlay verifier is not executing from exact T")

    python = Path(plan["runtime"]["python"])
    if not python.is_absolute() or not os.access(python, os.X_OK):
        raise ValueError("formal runtime Python is not executable")
    if Path(sys.executable).resolve(strict=True) != python.resolve(strict=True):
        raise ValueError("formal controller Python differs from overlay runtime")
    git = Path(plan["runtime"]["git"])
    if not git.is_absolute() or not os.access(git, os.X_OK):
        raise ValueError("formal runtime Git is not executable")
    _sha256_regular_executable(
        git,
        plan["runtime"]["git_sha256"],
        where="formal runtime Git",
    )
    nvidia_smi = Path(plan["runtime"]["nvidia_smi"])
    if not nvidia_smi.is_absolute() or not os.access(nvidia_smi, os.X_OK):
        raise ValueError("formal runtime NVIDIA_SMI is not executable")
    _require_physical_directory(
        Path(plan["runtime"]["job_work_root"]), where="formal job work root"
    )

    path = os.environ.get("PATH", "")
    git_env = _minimal_env(path=path)
    head = _run(
        [str(git), "-C", str(root), "rev-parse", "HEAD^{commit}"],
        env=git_env,
        where="exact-T HEAD attestation",
    ).stdout.strip()
    tree = _run(
        [str(git), "-C", str(root), "rev-parse", "HEAD^{tree}"],
        env=git_env,
        where="exact-T tree attestation",
    ).stdout.strip()
    if head != plan["source"]["commit_sha"]:
        raise ValueError("exact-T checkout commit mismatch")
    if tree != plan["source"]["tree_sha"]:
        raise ValueError("exact-T checkout tree mismatch")
    status = _run(
        [
            str(git),
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        env=git_env,
        where="exact-T clean-tree attestation",
    ).stdout
    if status:
        raise ValueError("exact-T checkout is dirty")
    symbolic = _run(
        [str(git), "-C", str(root), "symbolic-ref", "-q", "HEAD"],
        env=git_env,
        where="exact-T detached-HEAD attestation",
        allowed_returncodes={0, 1},
    )
    if symbolic.returncode == 0:
        raise ValueError("exact-T checkout must be detached")

    source = plan["source"]
    verifier = root / "scripts" / "verify_runtime_source.py"
    result = _run(
        [
            str(python),
            "-I",
            "-B",
            str(verifier),
            "--archive-root",
            str(root),
            "--source-manifest",
            source["manifest_path"],
            "--expected-manifest-sha256",
            source["manifest_sha256"],
            "--expected-commit-sha",
            source["commit_sha"],
            "--expected-tree-sha",
            source["tree_sha"],
        ],
        env=_minimal_env(path=path, pythonpath=str(root / "src")),
        where="exact-T source/import attestation",
    )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("exact-T source verifier returned invalid JSON") from error
    expected_report = {
        "schema_version": 1,
        "commit_sha": source["commit_sha"],
        "tree_sha": source["tree_sha"],
        "source_manifest_sha256": source["manifest_sha256"],
        "import_relative_path": "src/tabicl/__init__.py",
    }
    if report != expected_report:
        raise ValueError("exact-T source verifier returned unexpected evidence")
    repository_helper = _load_git_repository_helper(root)
    observed_repository = repository_helper.query_exact_repository(
        git=git,
        git_sha256=plan["runtime"]["git_sha256"],
        expected_commit_sha=source["commit_sha"],
    )
    if observed_repository != plan["repository_binding"]:
        raise ValueError("GitHub repository binding differs from the formal plan")
    plan["filesystem_isolation"] = _require_work_artifact_filesystem_isolation(
        plan, artifact_directory=Path(plan["artifact_root"]).parent
    )


def _validate_h100_smoke(plan: Mapping[str, Any]) -> None:
    root = Path(plan["exact_root"])
    python = plan["runtime"]["python"]
    source = plan["source"]
    smoke = plan["smoke"]
    _run(
        [
            python,
            "-I",
            "-B",
            str(root / "scripts" / "run_h100_identity_validation.py"),
            "--validate-attestation",
            smoke["attestation_path"],
            "--expected-attestation-sha256",
            smoke["expected_sha256"],
            "--expected-commit-sha",
            source["commit_sha"],
            "--expected-tree-sha",
            source["tree_sha"],
            "--expected-environment-sha256",
            source["environment_sha256"],
            "--expected-source-manifest-sha256",
            source["manifest_sha256"],
            "--expected-git-sha256",
            plan["repository_binding"]["git_sha256"],
            "--expected-repository-identity-sha256",
            plan["repository_binding"]["repository_identity_sha256"],
            "--expected-repository-query-sha256",
            plan["repository_binding"]["query_sha256"],
            "--expected-gpu-model",
            smoke["expected_gpu_model"],
            "--expected-checkpoint-ceiling-bytes",
            str(plan["capacity"]["checkpoint_ceiling_bytes"]),
            "--max-input-bytes",
            str(plan["capacity"]["attestation_ceiling_bytes"]),
        ],
        env=_minimal_env(
            path=os.environ.get("PATH", ""), pythonpath=str(root / "src")
        ),
        where="external digest-bound H100 smoke attestation",
    )


def _run_submit_capacity_gate(
    plan: Mapping[str, Any], *, artifact_directory: Path | None = None
) -> None:
    root = Path(plan["exact_root"])
    artifact_root = Path(plan["artifact_root"])
    target = artifact_root.parent if artifact_directory is None else artifact_directory
    _require_physical_directory(target, where="capacity artifact filesystem")
    if artifact_directory is None and (artifact_root.exists() or artifact_root.is_symlink()):
        raise FileExistsError("fresh study namespace already exists")
    capacity = plan["capacity"]
    _run(
        [
            plan["runtime"]["python"],
            "-I",
            "-B",
            str(root / "scripts" / "check_formal_capacity.py"),
            "--artifact-root",
            str(target),
            "--checkpoint-ceiling-bytes",
            str(capacity["checkpoint_ceiling_bytes"]),
            "--durable-log-allowance-bytes",
            str(capacity["durable_log_allowance_bytes"]),
            "--remaining-checkpoints",
            "15",
            "--run-log-ceiling-bytes",
            str(capacity["run_log_ceiling_bytes"]),
            "--attestation-ceiling-bytes",
            str(capacity["attestation_ceiling_bytes"]),
            "--manifest-ceiling-bytes",
            str(capacity["manifest_ceiling_bytes"]),
            "--protocol-metadata-allowance-bytes",
            str(capacity["protocol_metadata_allowance_bytes"]),
        ],
        env=_minimal_env(
            path=os.environ.get("PATH", ""), pythonpath=str(root / "src")
        ),
        where="fresh-cohort capacity gate",
    )


def _create_fresh_namespace(plan: Mapping[str, Any]) -> None:
    root = Path(plan["artifact_root"])
    parent = root.parent
    parent_fd = _open_directory_nofollow(
        parent, where="artifact namespace parent"
    )
    try:
        os.mkdir(root.name, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    root_fd = _open_directory_nofollow(root, where="fresh artifact namespace")
    try:
        for child in ("scheduler-logs", "runtime-completions"):
            os.mkdir(child, 0o700, dir_fd=root_fd)
        os.mkdir("arms", 0o700, dir_fd=root_fd)
        arms_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            arms_flags |= os.O_NOFOLLOW
        arms_fd = os.open("arms", arms_flags, dir_fd=root_fd)
        try:
            for arm in ARMS:
                os.mkdir(arm, 0o700, dir_fd=arms_fd)
            os.fsync(arms_fd)
        finally:
            os.close(arms_fd)
        os.fsync(root_fd)
    finally:
        os.close(root_fd)


class TransactionJournal:
    """A bounded, hash-chained and fsync'd submission event journal."""

    def __init__(self, path: Path, transaction_id: str) -> None:
        self.path = path
        self.transaction_id = _safe_id(transaction_id, "transaction_id")
        self.sequence = 0
        self.previous_sha256: str | None = None
        self.bytes_written = 0

    def append(self, event: str, payload: Mapping[str, Any]) -> None:
        event = _safe_id(event, "journal event")
        body = {
            "schema_version": SCHEMA_VERSION,
            "transaction_id": self.transaction_id,
            "sequence": self.sequence,
            "event": event,
            "previous_sha256": self.previous_sha256,
            "payload": _normalize(payload, where=f"journal {event}"),
        }
        record = {**body, "sha256": canonical_sha256(body)}
        raw = canonical_json_bytes(record) + b"\n"
        if self.bytes_written + len(raw) > TRANSACTION_JOURNAL_CEILING_BYTES:
            raise RuntimeError("transaction journal exceeded its precommitted ceiling")
        parent_fd = _open_directory_nofollow(
            self.path.parent, where="transaction journal parent"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        flags |= getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.path.name, flags, 0o600, dir_fd=parent_fd)
            try:
                metadata = os.fstat(fd)
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError("transaction journal must remain a regular file")
                if metadata.st_size != self.bytes_written:
                    raise RuntimeError("transaction journal changed outside the controller")
                _write_all(fd, raw)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        self.bytes_written += len(raw)
        self.previous_sha256 = record["sha256"]
        self.sequence += 1

    def seal(self, *, fault: str | None = None) -> None:
        if fault not in {None, "fchmod", "fsync", "dir_fsync"}:
            raise ValueError("unknown journal seal fault")
        parent_fd = _open_directory_nofollow(
            self.path.parent, where="transaction journal parent"
        )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.path.name, flags, dir_fd=parent_fd)
            try:
                if fault == "fchmod":
                    raise OSError("injected journal fchmod failure")
                os.fchmod(fd, 0o400)
                if fault == "fsync":
                    raise OSError("injected journal fsync failure")
                os.fsync(fd)
            finally:
                os.close(fd)
            if fault == "dir_fsync":
                raise OSError("injected journal directory fsync failure")
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)


def _new_transaction_id(plan: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(plan["ledger"]))
    digest.update(secrets.token_bytes(32))
    return digest.hexdigest()[:32]


def _sha256_regular_executable(path: Path, expected_sha256: str, *, where: str) -> str:
    """Hash a no-follow regular executable and reject path substitution."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    parent_fd = _open_directory_nofollow(path.parent, where=f"{where} parent")
    try:
        fd = os.open(path.name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError(f"{where} must be an existing no-follow executable") from error
    finally:
        os.close(parent_fd)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 128 * 1024 * 1024:
            raise ValueError(f"{where} must be a bounded regular file")
        if metadata.st_mode & 0o111 == 0:
            raise ValueError(f"{where} must have an executable mode bit")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        observed = digest.hexdigest()
    finally:
        os.close(fd)
    if observed != expected_sha256:
        raise ValueError(f"{where} SHA-256 mismatch")
    return str(path)


def _scheduler_commands(plan: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    commands: dict[str, str] = {}
    for name in SCHEDULER_COMMANDS:
        specification = plan["scheduler"]["commands"][name]
        commands[name] = _sha256_regular_executable(
            Path(specification["path"]),
            specification["sha256"],
            where=f"scheduler {name}",
        )
    return commands, _minimal_env(path="/usr/bin:/bin")


def _export_argument(exports: Mapping[str, str]) -> str:
    if "ALL" in exports or "PYTHONHOME" in exports:
        raise ValueError("formal job export allowlist is contaminated")
    items = []
    for key in sorted(exports):
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError("formal job export name is invalid")
        value = exports[key]
        if any(character in value for character in (",", "\x00", "\n", "\r")):
            raise ValueError("formal job export value is invalid")
        items.append(f"{key}={value}")
    return "--export=" + ",".join(items)


def _sbatch_argv(
    plan: Mapping[str, Any],
    job: Mapping[str, Any],
    parent_job_id: str | None,
    job_name: str,
) -> list[str]:
    scheduler = plan["scheduler"]
    argv = [
        "--parsable",
        "--hold",
        f"--partition={scheduler['partition']}",
        f"--qos={scheduler['qos']}",
        "--nodes=1",
        "--gres=gpu:1",
        f"--cpus-per-task={scheduler['cpus_per_task']}",
        f"--mem={scheduler['memory_mb']}M",
        f"--time={_slurm_duration(job['time_limit'], 'formal job time limit')}",
        f"--job-name={_safe_id(job_name, 'scheduler job name')}",
        f"--output={job['scheduler_stdout']}",
        f"--error={job['scheduler_stderr']}",
        f"--chdir={plan['artifact_root']}",
        "--open-mode=truncate",
    ]
    if parent_job_id is not None:
        argv.extend(
            [
                f"--dependency=afterok:{parent_job_id}",
                "--kill-on-invalid-dep=yes",
            ]
        )
    argv.extend(
        [
            _export_argument(job["exports"]),
            str(Path(plan["exact_root"]) / "scripts/slurm_h100_identity_formal.sh"),
        ]
    )
    return argv


def _ledger_is_published(plan: Mapping[str, Any]) -> bool:
    path = Path(plan["ledger_path"])
    try:
        return path.read_bytes() == canonical_json_bytes(plan["ledger"]) + b"\n"
    except OSError:
        return False


class UnresolvedSubmission(RuntimeError):
    def __init__(
        self,
        job_name: str,
        message: str,
        *,
        known_job_id: str | None = None,
        cluster: str | None = None,
        name_remains_unresolved: bool = True,
    ) -> None:
        super().__init__(message)
        self.job_name = job_name
        self.known_job_id = known_job_id
        self.cluster = cluster
        self.name_remains_unresolved = name_remains_unresolved


class SubmissionInterrupted(RuntimeError):
    pass


def _cluster_argv(cluster: str | None) -> list[str]:
    return [f"--clusters={_safe_id(cluster, 'scheduler cluster')}"] if cluster else []


def _query_jobs_by_name(
    *,
    squeue: str,
    scheduler_env: Mapping[str, str],
    job_name: str,
    cluster: str | None,
    expected_job_id: str | None = None,
) -> list[dict[str, str]]:
    completed = _run(
        [
            squeue,
            *_cluster_argv(cluster),
            "--noheader",
            f"--name={_safe_id(job_name, 'scheduler job name')}",
            "--format=%i|%j|%T|%r",
        ],
        env=scheduler_env,
        where=f"scheduler query for {job_name}",
    )
    if len(completed.stdout.encode("utf-8")) > 1_000_000:
        raise ValueError("scheduler query response exceeded the safety ceiling")
    result: list[dict[str, str]] = []
    for raw_line in completed.stdout.splitlines():
        if not raw_line:
            continue
        fields = raw_line.split("|")
        purports_name = job_name in raw_line
        purports_id = expected_job_id is not None and re.search(
            rf"(?<![0-9]){re.escape(expected_job_id)}(?![0-9])", raw_line
        ) is not None
        if len(fields) != 4:
            if purports_name or purports_id:
                raise ValueError("scheduler query returned a malformed transaction row")
            # Login banners can be injected into scheduler command stdout.  A
            # non-record line that names neither this transaction job nor its
            # trusted ID is unrelated transport noise, not scheduler state.
            continue
        job_id, observed_name, state, reason = fields
        if observed_name != job_name:
            if purports_id:
                raise ValueError("scheduler query bound the transaction ID to another name")
            continue
        _job_ids((job_id,), "scheduler query")
        if re.fullmatch(r"[A-Z_]+", state) is None:
            raise ValueError("scheduler query returned an invalid state")
        if any(character in reason for character in ("\x00", "\n", "\r", "|")):
            raise ValueError("scheduler query returned an invalid reason")
        result.append(
            {"job_id": job_id, "job_name": observed_name, "state": state, "reason": reason}
        )
    return result


def _reconcile_held_submission(
    *,
    completed: subprocess.CompletedProcess[str],
    squeue: str,
    scheduler_env: Mapping[str, str],
    job_name: str,
    journal: TransactionJournal,
) -> tuple[str, str | None]:
    parsed_id: str | None = None
    cluster: str | None = None
    try:
        candidate_id, candidate_cluster = _parse_sbatch_result(
            completed.stdout.strip()
        )
        if completed.returncode != 0:
            raise ValueError("nonzero sbatch response is not authoritative")
        parsed_id, cluster = candidate_id, candidate_cluster
    except ValueError:
        journal.append(
            "submit_response_untrusted",
            {
                "job_name": job_name,
                "returncode": completed.returncode,
                "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
            },
        )
    observations: list[dict[str, str]] = []
    for _attempt in range(3):
        try:
            observations = _query_jobs_by_name(
                squeue=squeue,
                scheduler_env=scheduler_env,
                job_name=job_name,
                cluster=cluster,
                expected_job_id=parsed_id,
            )
        except Exception:
            observations = []
        if observations:
            break
    if len(observations) != 1:
        if parsed_id is not None and completed.returncode == 0:
            journal.append(
                "submit_response_known_id_unverified",
                {"job_name": job_name, "job_id": parsed_id, "cluster": cluster},
            )
        raise UnresolvedSubmission(
            job_name,
            "submission response could not be reconciled to exactly one held job",
            known_job_id=(parsed_id if completed.returncode == 0 else None),
            cluster=cluster,
            name_remains_unresolved=len(observations) > 1 or parsed_id is None,
        )
    observed = observations[0]
    if observed["state"] != "PENDING" or not observed["reason"].startswith("JobHeld"):
        raise UnresolvedSubmission(
            job_name,
            "reconciled submission is not uniquely held",
            known_job_id=observed["job_id"],
            cluster=cluster,
            name_remains_unresolved=False,
        )
    if parsed_id is not None and parsed_id != observed["job_id"]:
        journal.append(
            "submit_response_id_reconciled",
            {
                "job_name": job_name,
                "response_job_id": parsed_id,
                "observed_job_id": observed["job_id"],
            },
        )
    journal.append(
        "submit_accepted",
        {
            "job_name": job_name,
            "job_id": observed["job_id"],
            "cluster": cluster,
            "state": observed["state"],
            "reason": observed["reason"],
        },
    )
    return observed["job_id"], cluster


def _verify_released(
    *,
    squeue: str,
    scheduler_env: Mapping[str, str],
    item: Mapping[str, Any],
) -> bool:
    try:
        observations = _query_jobs_by_name(
            squeue=squeue,
            scheduler_env=scheduler_env,
            job_name=item["job_name"],
            cluster=item.get("cluster"),
            expected_job_id=item["job_id"],
        )
    except Exception:
        return False
    return (
        len(observations) == 1
        and observations[0]["job_id"] == item["job_id"]
        and not observations[0]["reason"].startswith("JobHeld")
        and observations[0]["state"]
        in {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING"}
    )


def _scheduler_state_token(raw: str) -> str | None:
    fields = raw.split()
    token = fields[0].split("+", 1)[0].upper() if fields else ""
    return token if re.fullmatch(r"[A-Z][A-Z_]*", token) is not None else None


def _query_scheduler_state(
    *,
    squeue: str,
    sacct: str,
    scheduler_env: Mapping[str, str],
    item: Mapping[str, Any],
) -> tuple[str, str, str]:
    """Resolve one accepted job by ID, falling back from queue to accounting."""

    job_id = item["job_id"]
    cluster_argv = _cluster_argv(item.get("cluster"))
    queue = subprocess.run(
        [
            squeue,
            *cluster_argv,
            "--noheader",
            f"--jobs={job_id}",
            "--format=%i|%T|%r",
        ],
        env=dict(scheduler_env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if (
        len(queue.stdout.encode("utf-8")) > 1_000_000
        or len(queue.stderr.encode("utf-8")) > 1_000_000
    ):
        raise RuntimeError("squeue cancellation reconciliation exceeded its ceiling")
    queue_rows: list[tuple[str, str]] = []
    if queue.returncode == 0:
        for raw_line in queue.stdout.splitlines():
            fields = [field.strip() for field in raw_line.split("|", 2)]
            if len(fields) != 3 or fields[0] != job_id:
                if re.search(
                    rf"(?<![0-9]){re.escape(job_id)}(?![0-9])", raw_line
                ):
                    raise RuntimeError(
                        "squeue cancellation reconciliation returned a malformed job row"
                    )
                continue
            state = _scheduler_state_token(fields[1])
            if state is None:
                raise RuntimeError(
                    "squeue cancellation reconciliation returned an invalid state"
                )
            queue_rows.append((state, fields[2]))
    if len(queue_rows) > 1:
        raise RuntimeError("squeue cancellation reconciliation is ambiguous")
    if len(queue_rows) == 1:
        state, reason = queue_rows[0]
        return state, reason, "squeue"

    accounting = subprocess.run(
        [
            sacct,
            *cluster_argv,
            "--noheader",
            "--allocations",
            f"--jobs={job_id}",
            "--format=JobIDRaw,State",
            "--parsable2",
        ],
        env=dict(scheduler_env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if (
        len(accounting.stdout.encode("utf-8")) > 1_000_000
        or len(accounting.stderr.encode("utf-8")) > 1_000_000
    ):
        raise RuntimeError("sacct cancellation reconciliation exceeded its ceiling")
    accounting_rows: list[str] = []
    if accounting.returncode == 0:
        for raw_line in accounting.stdout.splitlines():
            fields = [field.strip() for field in raw_line.split("|", 2)]
            if len(fields) < 2 or fields[0] != job_id:
                if re.search(
                    rf"(?<![0-9]){re.escape(job_id)}(?![0-9])", raw_line
                ):
                    raise RuntimeError(
                        "sacct cancellation reconciliation returned a malformed job row"
                    )
                continue
            state = _scheduler_state_token(fields[1])
            if state is None:
                raise RuntimeError(
                    "sacct cancellation reconciliation returned an invalid state"
                )
            accounting_rows.append(state)
    if len(accounting_rows) > 1:
        raise RuntimeError("sacct cancellation reconciliation is ambiguous")
    if len(accounting_rows) == 1:
        return accounting_rows[0], "", "sacct"
    raise RuntimeError(f"cannot reconcile scheduler state for job {job_id}")


def _verify_cancelled(
    *,
    squeue: str,
    sacct: str,
    scheduler_env: Mapping[str, str],
    item: Mapping[str, Any],
) -> tuple[bool, str | None, str | None]:
    last_state: str | None = None
    last_source: str | None = None
    for _attempt in range(SCHEDULER_RECONCILIATION_ATTEMPTS):
        try:
            state, _reason, source = _query_scheduler_state(
                squeue=squeue,
                sacct=sacct,
                scheduler_env=scheduler_env,
                item=item,
            )
        except Exception:
            continue
        last_state, last_source = state, source
        if state in TERMINAL_SCHEDULER_STATES:
            return True, state, source
    return False, last_state, last_source


def _safe_journal_append(
    journal: TransactionJournal, event: str, payload: Mapping[str, Any]
) -> bool:
    """Keep cleanup progressing when the diagnostic journal is unavailable."""

    try:
        journal.append(event, payload)
    except Exception:
        return False
    return True


def _rollback(
    *,
    scancel: str,
    squeue: str,
    sacct: str,
    scheduler_env: Mapping[str, str],
    submitted: Sequence[Mapping[str, Any]],
    journal: TransactionJournal,
) -> tuple[list[str], list[str]]:
    cancelled: list[str] = []
    remaining: list[str] = []
    for item in reversed(submitted):
        job_id = item["job_id"]
        try:
            completed = subprocess.run(
                [scancel, *_cluster_argv(item.get("cluster")), job_id],
                env=dict(scheduler_env),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except BaseException as cancel_error:
            remaining.append(job_id)
            _safe_journal_append(
                journal,
                "cancel_exec_failed",
                {
                    "job_name": item["job_name"],
                    "job_id": job_id,
                    "error_type": type(cancel_error).__name__,
                },
            )
            continue
        if completed.returncode != 0:
            remaining.append(job_id)
            _safe_journal_append(
                journal,
                "cancel_observed",
                {
                    "job_name": item["job_name"],
                    "job_id": job_id,
                    "returncode": completed.returncode,
                    "verified": False,
                    "state": None,
                    "source": None,
                },
            )
            continue
        verified, state, source = _verify_cancelled(
            squeue=squeue,
            sacct=sacct,
            scheduler_env=scheduler_env,
            item=item,
        )
        _safe_journal_append(
            journal,
            "cancel_observed",
            {
                "job_name": item["job_name"],
                "job_id": job_id,
                "returncode": completed.returncode,
                "verified": verified,
                "state": state,
                "source": source,
            },
        )
        if verified:
            cancelled.append(job_id)
        else:
            remaining.append(job_id)
    return cancelled, remaining


def submit_overlay(
    overlay_path: Path,
    *,
    exact_root: Path,
    ledger_fault: str | None = None,
    journal_seal_fault: str | None = None,
    commit_fault: str | None = None,
    process_scoped_signals: bool = False,
    test_signal_after_commit: bool = False,
) -> dict[str, Any]:
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise ValueError("formal submit controller requires Python -I -B")
    if test_signal_after_commit and not process_scoped_signals:
        raise ValueError("post-commit signal injection requires process-scoped signals")
    plan = validate_overlay(_load_strict_json(overlay_path), exact_root=exact_root)
    artifact_root = Path(plan["artifact_root"])
    if artifact_root == exact_root or exact_root in artifact_root.parents:
        raise ValueError("formal artifact namespace must be outside exact T")
    _attest_exact_checkout(plan)
    _validate_h100_smoke(plan)
    _run_submit_capacity_gate(plan)
    commands, scheduler_env = _scheduler_commands(plan)
    _create_fresh_namespace(plan)
    transaction_id = _new_transaction_id(plan)
    plan["transaction_id"] = transaction_id
    plan["rollback_path"] = str(
        artifact_root / f"rollback-incomplete-{transaction_id}.json"
    )
    journal = TransactionJournal(
        artifact_root / f"transaction-journal-{transaction_id}.jsonl",
        transaction_id,
    )
    journal.append(
        "transaction_started",
        {
            "study_id": plan["study_id"],
            "seed": plan["seed"],
            "ledger_sha256": plan["ledger"]["sha256"],
        },
    )
    accepted_ids: list[str] = []
    submitted: list[dict[str, Any]] = []
    observed_clusters: set[str | None] = set()
    job_ids_by_key: dict[tuple[str, str], str] = {}
    failure_phase = "sbatch_failed"
    unresolved_job_names: list[str] = []
    pending_submit_name: str | None = None
    prior_handlers: dict[int, Any] = {}

    def interrupt_handler(signum: int, _frame: Any) -> None:
        raise SubmissionInterrupted(f"signal_{signum}")

    if process_scoped_signals:
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            prior_handlers[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, interrupt_handler)
    try:
        for job in plan["jobs"]:
            parent_key = job["parent_job"]
            parent_id = job_ids_by_key[parent_key] if parent_key is not None else None
            job_name = f"tabicl-{transaction_id}-{job['arm']}-s{job['stage_index']}"
            job["exports"]["FORMAL_TRANSACTION_ID"] = transaction_id
            job["exports"]["FORMAL_EXPECTED_JOB_NAME"] = job_name
            pending_submit_name = job_name
            journal.append(
                "submit_intent",
                {
                    "arm": job["arm"],
                    "stage": job["stage"],
                    "job_name": job_name,
                    "parent_job_id": parent_id,
                },
            )
            completed = subprocess.run(
                [
                    commands["sbatch"],
                    *_sbatch_argv(plan, job, parent_id, job_name),
                ],
                env=dict(scheduler_env),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            try:
                job_id, cluster = _reconcile_held_submission(
                    completed=completed,
                    squeue=commands["squeue"],
                    scheduler_env=scheduler_env,
                    job_name=job_name,
                    journal=journal,
                )
            except UnresolvedSubmission as unresolved:
                if (
                    unresolved.known_job_id is not None
                    and unresolved.known_job_id not in accepted_ids
                ):
                    accepted_ids.append(unresolved.known_job_id)
                    submitted.append(
                        {
                            "job": job,
                            "job_id": unresolved.known_job_id,
                            "cluster": unresolved.cluster,
                            "job_name": job_name,
                            "parent_job_id": parent_id,
                        }
                    )
                    pending_submit_name = None
                else:
                    unresolved.name_remains_unresolved = True
                raise
            if job_id in accepted_ids:
                unresolved_job_names.append(job_name)
                raise UnresolvedSubmission(
                    job_name, "scheduler reconciled two transaction names to one job ID"
                )
            accepted_ids.append(job_id)
            job_ids_by_key[(job["arm"], job["stage"])] = job_id
            submitted.append(
                {
                    "job": job,
                    "job_id": job_id,
                    "cluster": cluster,
                    "job_name": job_name,
                    "parent_job_id": parent_id,
                }
            )
            pending_submit_name = None
            observed_clusters.add(cluster)
            if len(observed_clusters) != 1:
                raise RuntimeError(
                    "formal transaction jobs resolved to different scheduler clusters"
                )

        # All nine allocations remain held.  Reopen both roots without
        # following path components, prove their device identities have not
        # changed, and repeat the conservative capacity gate on the filesystem
        # that will actually receive the cohort before any release.
        failure_phase = "capacity_recheck_failed"
        repository_recheck = _load_git_repository_helper(
            Path(plan["exact_root"])
        ).query_exact_repository(
            git=Path(plan["runtime"]["git"]),
            git_sha256=plan["runtime"]["git_sha256"],
            expected_commit_sha=plan["source"]["commit_sha"],
        )
        if repository_recheck != plan["repository_binding"]:
            raise ValueError("GitHub repository binding changed before formal release")
        isolation_recheck = _require_work_artifact_filesystem_isolation(
            plan, artifact_directory=artifact_root
        )
        initial_isolation = plan["filesystem_isolation"]
        if (
            isolation_recheck["work_device"] != initial_isolation["work_device"]
            or isolation_recheck["artifact_device"]
            != initial_isolation["artifact_device"]
        ):
            raise ValueError(
                "formal work/artifact filesystem identity changed before release"
            )
        plan["filesystem_isolation"] = isolation_recheck
        _run_submit_capacity_gate(plan, artifact_directory=artifact_root)
        journal.append(
            "capacity_rechecked",
            {
                "work_device": isolation_recheck["work_device"],
                "artifact_device": isolation_recheck["artifact_device"],
            },
        )

        failure_phase = "ledger_publication_failed"
        publish_ledger(plan, fault=ledger_fault)
        journal.append("ledger_published", {"sha256": plan["ledger"]["sha256"]})
        failure_phase = "receipt_publication_failed"
        _receipt_path, receipt = _publish_submission_receipt(plan, submitted)
        journal.append("receipt_published", {"sha256": receipt["sha256"]})
        failure_phase = "release_failed"
        for item in submitted:
            journal.append(
                "release_intent",
                {"job_name": item["job_name"], "job_id": item["job_id"]},
            )
            completed = subprocess.run(
                [
                    commands["scontrol"],
                    *_cluster_argv(item.get("cluster")),
                    "release",
                    item["job_id"],
                ],
                env=dict(scheduler_env),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            released = _verify_released(
                squeue=commands["squeue"],
                scheduler_env=scheduler_env,
                item=item,
            )
            journal.append(
                "release_observed",
                {
                    "job_name": item["job_name"],
                    "job_id": item["job_id"],
                    "returncode": completed.returncode,
                    "verified": released,
                },
            )
            if not released:
                raise RuntimeError(
                    f"release of transaction job {item['job_name']} was not verified"
                )
        journal.append("transaction_released", {"job_ids": accepted_ids})
        failure_phase = "journal_seal_failed"
        journal.seal(fault=journal_seal_fault)
        # From this point the standalone controller owns signal disposition
        # until process exit.  There is deliberately no post-commit handler
        # restoration window in which a signal could turn a committed cohort
        # into an outward CLI failure.
        if process_scoped_signals:
            for signal_number in prior_handlers:
                signal.signal(signal_number, signal.SIG_IGN)
        failure_phase = "commit_publication_failed"
        _publish_submission_commit(
            plan,
            receipt=receipt,
            submitted=submitted,
            fault=commit_fault,
        )
        if test_signal_after_commit:
            os.kill(os.getpid(), signal.SIGTERM)
    except BaseException as error:
        # Cleanup is a non-interruptible fail-closed region.  A second
        # SIGINT/SIGTERM must not strand a partially released cohort.
        if process_scoped_signals:
            for signal_number in prior_handlers:
                signal.signal(signal_number, signal.SIG_IGN)
        if pending_submit_name is not None:
            unresolved_job_names.append(pending_submit_name)
        if isinstance(error, UnresolvedSubmission) and error.name_remains_unresolved:
            unresolved_job_names.append(error.job_name)
        if failure_phase == "commit_publication_failed":
            try:
                _remove_submission_commit(plan)
            except Exception:
                # A durable rollback record makes every consumer reject a
                # marker that could not itself be durably revoked.
                unresolved_job_names.append("transaction-commit-marker")
        unresolved_job_names = list(dict.fromkeys(unresolved_job_names))
        _safe_journal_append(
            journal,
            "transaction_failure",
            {
                "phase": failure_phase,
                "error_type": type(error).__name__,
                "unresolved_job_names": unresolved_job_names,
            },
        )
        try:
            cancelled, remaining = _rollback(
                scancel=commands["scancel"],
                squeue=commands["squeue"],
                sacct=commands["sacct"],
                scheduler_env=scheduler_env,
                submitted=submitted,
                journal=journal,
            )
        except BaseException as rollback_error:
            cancelled, remaining = [], list(accepted_ids)
            _safe_journal_append(
                journal,
                "rollback_controller_failed",
                {"error_type": type(rollback_error).__name__},
            )
        if remaining or unresolved_job_names:
            try:
                recovery = publish_rollback_incomplete(
                    plan,
                    accepted_job_ids=accepted_ids,
                    cancelled_job_ids=cancelled,
                    remaining_job_ids=remaining,
                    reason=failure_phase,
                    ledger_published=_ledger_is_published(plan),
                    unresolved_job_names=unresolved_job_names,
                )
            except Exception as recovery_error:
                raise RuntimeError(
                    "formal submission failed and rollback remains incomplete; "
                    "recovery record failed"
                ) from recovery_error
            _safe_journal_append(
                journal,
                "recovery_published",
                {
                    "path": str(recovery),
                    "remaining_job_ids": remaining,
                    "unresolved_job_names": unresolved_job_names,
                },
            )
            try:
                journal.seal()
            except Exception:
                pass
            raise RuntimeError(
                "formal submission failed and rollback remains incomplete; "
                f"recovery record: {recovery}"
            ) from error
        _safe_journal_append(
            journal, "transaction_rolled_back", {"cancelled_job_ids": cancelled}
        )
        try:
            journal.seal()
        except Exception:
            pass
        raise RuntimeError("formal submission failed; accepted jobs were rolled back") from error
    return receipt


def _load_strict_json(path: Path) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=pairs, parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(f"invalid number {raw}")))
    if not isinstance(value, Mapping):
        raise ValueError("overlay must contain a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-overlay", type=Path)
    group.add_argument("--submit-overlay", type=Path)
    parser.add_argument("--exact-root", required=True, type=Path)
    parser.add_argument(
        "--fault-ledger-stage", choices=("write", "fsync", "rename", "dir_fsync")
    )
    parser.add_argument(
        "--fault-journal-seal-stage",
        choices=("fchmod", "fsync", "dir_fsync"),
    )
    parser.add_argument(
        "--fault-commit-stage",
        choices=("write", "fsync", "rename", "dir_fsync"),
    )
    parser.add_argument(
        "--test-signal-after-commit",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.validate_overlay:
        if (
            args.fault_ledger_stage
            or args.fault_journal_seal_stage
            or args.fault_commit_stage
        ):
            parser.error("fault injection is valid only with --submit-overlay")
        result = validate_overlay(
            _load_strict_json(args.validate_overlay), exact_root=args.exact_root
        )
    else:
        result = submit_overlay(
            args.submit_overlay,
            exact_root=args.exact_root,
            ledger_fault=args.fault_ledger_stage,
            journal_seal_fault=args.fault_journal_seal_stage,
            commit_fault=args.fault_commit_stage,
            process_scoped_signals=True,
            test_signal_after_commit=args.test_signal_after_commit,
        )
    try:
        print(canonical_json_bytes(result).decode("utf-8"))
    except OSError:
        # The immutable commit marker is the authoritative success boundary;
        # a closed diagnostic stdout must not misreport it as rollback-worthy.
        pass
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"formal overlay verification failed: {error}", file=sys.stderr)
        raise SystemExit(2)
