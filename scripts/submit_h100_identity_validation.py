#!/usr/bin/env python3
"""Submit the exact twelve-case H100 gate as one held Slurm transaction.

This controller uses only the standard library before it attests the clean,
detached candidate.  It never executes a case from the controller checkout:
every Slurm wrapper clones and detaches the exact candidate independently.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_JOB_ID = re.compile(r"^[1-9][0-9]{0,19}$")
_SBATCH_RESULT = re.compile(r"^([1-9][0-9]{0,19})(?:;[A-Za-z0-9._-]+)?$")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_ENVIRONMENT_COMPLETION_MAX_BYTES = 1 << 20
_ENVIRONMENT_MANIFEST_MAX_BYTES = 1 << 20
_ENVIRONMENT_INVENTORY_MAX_BYTES = 128 << 20
SCHEDULER_COMMAND_TIMEOUT_SECONDS = 30
LOCAL_GIT_CONFIG_OVERRIDES = (
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.filemode=true",
)
_ONE_GPU_WRAPPER = "scripts/slurm_h100_identity_maxseq_smoke.sh"
_TWO_GPU_WRAPPER = "scripts/slurm_h100_identity_nccl_smoke.sh"
_SCHEDULER = {
    "partition": "h100",
    "qos": "short",
    "time_limit": "03:00:00",
    "resources_by_world_size": {
        "1": {
            "gpus": 1,
            "cpus_per_task": 32,
            "memory_mb": 131072,
            "wrapper": _ONE_GPU_WRAPPER,
        },
        "2": {
            "gpus": 2,
            "cpus_per_task": 64,
            "memory_mb": 131072,
            "wrapper": _TWO_GPU_WRAPPER,
        },
    },
}


def _fail(message: str) -> "None":
    raise ValueError(message)


def _normalize(value: Any, where: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(f"{where} contains a non-finite float")
        return value
    if isinstance(value, list):
        return [_normalize(item, f"{where}[]") for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or "\x00" in key:
                _fail(f"{where} contains an invalid key")
            result[key] = _normalize(item, f"{where}.{key}")
        return result
    _fail(f"{where} contains unsupported type {type(value).__name__}")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _exact_keys(value: Any, expected: set[str], where: str) -> None:
    actual = set(value) if isinstance(value, Mapping) else set()
    if actual != expected:
        _fail(
            f"{where} keys mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        _fail(f"{where} must be a lowercase SHA-256 digest")
    return value


def _positive(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _fail(f"{where} must be a positive integer")
    return value


_FILE_SLOTS_BY_CATEGORY = {
    "checkpoint": 9,
    "case_log": 12,
    "case_attestation": 36,
    "final_attestation": 1,
    "case_result": 12,
    "gpu_raw": 12,
    "gpu_summary": 6,
    "controller_receipt": 4,
    "scheduler_log": 24,
}
_DIRECTORY_SLOT_COUNT = 28
_ARTIFACT_ROOT_DIRENT_SLOTS = 1
_CHILD_DIRECTORY_DIRENT_SLOTS = 27
# All twelve cases may have an atomic publication's temporary and final names
# visible concurrently; the capacity gate must not assume scheduler serialization.
_ATOMIC_EXTRA_DIRENT_SLOTS = 13


def _round_up_allocation(value: int, allocation_unit_bytes: int) -> int:
    return ((value + allocation_unit_bytes - 1) // allocation_unit_bytes) * allocation_unit_bytes


def gate_capacity_budget(
    limits: Mapping[str, int], *, allocation_unit_bytes: int
) -> dict[str, Any]:
    """Return the logical and physical budget for one twelve-case gate."""

    allocation_unit = _positive(allocation_unit_bytes, "filesystem allocation unit")
    ceilings = {
        "checkpoint": _positive(
            limits["checkpoint_ceiling_bytes"], "checkpoint ceiling"
        ),
        "case_log": _positive(limits["run_log_ceiling_bytes"], "run-log ceiling"),
        "case_attestation": _positive(
            limits["attestation_ceiling_bytes"], "attestation ceiling"
        ),
        "final_attestation": _positive(
            limits["attestation_ceiling_bytes"], "attestation ceiling"
        ),
        "case_result": 1 << 20,
        "gpu_raw": _positive(
            limits["gpu_monitor_ceiling_bytes"], "GPU-monitor ceiling"
        ),
        "gpu_summary": _positive(
            limits["run_log_ceiling_bytes"], "run-log ceiling"
        ),
        "controller_receipt": _positive(
            limits["receipt_ceiling_bytes"], "receipt ceiling"
        ),
        "scheduler_log": _positive(
            limits["scheduler_log_ceiling_bytes"], "scheduler-log ceiling"
        ),
    }
    logical = {
        category: _FILE_SLOTS_BY_CATEGORY[category] * ceiling
        for category, ceiling in ceilings.items()
    }
    physical = {
        category: _FILE_SLOTS_BY_CATEGORY[category]
        * _round_up_allocation(ceiling, allocation_unit)
        for category, ceiling in ceilings.items()
    }
    total_file_slots = sum(_FILE_SLOTS_BY_CATEGORY.values())
    if total_file_slots != 116:
        _fail("H100 gate file-slot accounting drifted")
    artifact_root_dirents = _ARTIFACT_ROOT_DIRENT_SLOTS
    child_directory_dirents = _CHILD_DIRECTORY_DIRENT_SLOTS
    file_dirents = total_file_slots
    atomic_extra_dirents = _ATOMIC_EXTRA_DIRENT_SLOTS
    total_dirents = (
        artifact_root_dirents
        + child_directory_dirents
        + file_dirents
        + atomic_extra_dirents
    )
    if _DIRECTORY_SLOT_COUNT != 28 or total_dirents != 157:
        _fail("H100 gate directory/dirent accounting drifted")
    base_directory_physical = _DIRECTORY_SLOT_COUNT * allocation_unit
    dirent_physical = total_dirents * allocation_unit
    total_directory_physical = base_directory_physical + dirent_physical
    total_file_physical = sum(physical.values())
    worst_logical = sum(logical.values())
    worst_physical = total_file_physical + total_directory_physical
    reserve = 20 * (1 << 30)
    reserve_physical = _round_up_allocation(reserve, allocation_unit)
    return {
        "allocation_unit_bytes": allocation_unit,
        "reserve_bytes": reserve,
        "reserve_physical_bytes": reserve_physical,
        "file_slots_by_category": dict(_FILE_SLOTS_BY_CATEGORY),
        "total_file_slots": total_file_slots,
        **{f"{category}_bytes": value for category, value in logical.items()},
        "worst_case_gate_bytes": worst_logical,
        **{
            f"{category}_physical_bytes": value
            for category, value in physical.items()
        },
        "total_file_physical_bytes": total_file_physical,
        "directory_slot_count": _DIRECTORY_SLOT_COUNT,
        "base_directory_physical_bytes": base_directory_physical,
        "artifact_root_dirent_slots": artifact_root_dirents,
        "child_directory_dirent_slots": child_directory_dirents,
        "file_dirent_slots": file_dirents,
        "atomic_extra_dirent_slots": atomic_extra_dirents,
        "total_dirent_slots": total_dirents,
        "dirent_physical_bytes": dirent_physical,
        "total_directory_physical_bytes": total_directory_physical,
        "worst_case_gate_physical_bytes": worst_physical,
        "required_free_bytes": reserve_physical + worst_physical,
    }


def capacity_preflight(
    artifact_parent: Path,
    limits: Mapping[str, int],
    *,
    statvfs_fn=None,
    stat_fn=None,
) -> dict[str, Any]:
    """Take a stable parent/filesystem snapshot and enforce the gate reserve."""

    statvfs_fn = os.statvfs if statvfs_fn is None else statvfs_fn
    stat_fn = os.stat if stat_fn is None else stat_fn
    before = stat_fn(artifact_parent, follow_symlinks=False)
    filesystem = statvfs_fn(artifact_parent)
    blocks = int(filesystem.f_bavail)
    fragment_size = int(filesystem.f_frsize)
    if blocks < 0 or fragment_size < 1:
        _fail("artifact filesystem returned an invalid capacity snapshot")
    available = blocks * fragment_size
    after = stat_fn(artifact_parent, follow_symlinks=False)
    snapshot = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in snapshot):
        _fail("artifact parent changed during the capacity snapshot")
    budget = gate_capacity_budget(
        limits, allocation_unit_bytes=fragment_size
    )
    if available < budget["required_free_bytes"]:
        _fail(
            "insufficient space for H100 gate: "
            f"available={available} required={budget['required_free_bytes']}"
        )
    return {
        **budget,
        "available_blocks": blocks,
        "available_bytes": available,
    }


def _load_strict_json(path: Path, *, max_bytes: int = 8 << 20) -> Mapping[str, Any]:
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        _fail("gate overlay exceeds the input byte ceiling")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: _fail(f"invalid JSON constant: {token}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid H100 gate overlay JSON") from error
    if not isinstance(value, Mapping) or raw != _canonical(value) + b"\n":
        _fail("H100 gate overlay must be canonical newline-terminated JSON")
    return value


def _strict_json_bytes(raw: bytes, *, where: str, max_bytes: int) -> Mapping[str, Any]:
    if len(raw) > max_bytes:
        _fail(f"{where} exceeds its byte ceiling")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: _fail(f"invalid JSON constant: {token}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {where} JSON") from error
    if not isinstance(value, Mapping) or raw != _canonical(value) + b"\n":
        _fail(f"{where} must be canonical newline-terminated JSON")
    return value


def _absolute(path: Any, where: str) -> Path:
    if not isinstance(path, str) or not path:
        _fail(f"{where} must be a non-empty absolute path")
    value = Path(path)
    if not value.is_absolute() or any(part in {"", ".", ".."} for part in value.parts[1:]):
        _fail(f"{where} must be a normalized absolute path")
    return value


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        _fail("no-follow directory traversal is unavailable")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _open_directory_nofollow(path: Path, *, where: str) -> int:
    path = _absolute(os.fspath(path), where)
    fd = os.open(os.path.sep, _directory_flags())
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, _directory_flags(), dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except OSError as error:
        os.close(fd)
        raise ValueError(f"{where} contains a symlink or invalid component") from error


def _require_directory(path: Path, *, where: str) -> Path:
    fd = _open_directory_nofollow(path, where=where)
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            _fail(f"{where} is not a directory")
    finally:
        os.close(fd)
    return path


def _require_executable(path: Any, where: str) -> Path:
    value = _absolute(path, where)
    if not value.is_file() or not os.access(value, os.X_OK):
        _fail(f"{where} must be an executable regular file")
    return value


def _open_digest_bound_file(
    path: Any,
    expected_sha256: Any,
    where: str,
    *,
    require_executable: bool,
) -> tuple[str, int]:
    """Open/hash one file and retain that exact inode for later consumption."""

    value = _absolute(
        os.fspath(path) if isinstance(path, os.PathLike) else path, where
    )
    expected = _digest(expected_sha256, f"{where} digest")
    parent_fd = _open_directory_nofollow(value.parent, where=f"{where} parent")
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(value.name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError(f"{where} is not a no-follow regular file") from error
    finally:
        os.close(parent_fd)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or (require_executable and before.st_mode & 0o111 == 0)
            or before.st_size > 128 * 1024 * 1024
        ):
            _fail(
                f"{where} is not a bounded regular"
                + (" executable" if require_executable else " file")
            )
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                _fail(f"{where} was truncated while being attested")
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, field) != getattr(after, field) for field in fields):
            _fail(f"{where} changed while being attested")
        if digest.hexdigest() != expected:
            _fail(f"{where} digest mismatch")
        os.lseek(fd, 0, os.SEEK_SET)
        return f"/proc/self/fd/{fd}", fd
    except BaseException:
        os.close(fd)
        raise


def _open_digest_bound_executable(
    path: Any, expected_sha256: Any, where: str
) -> tuple[str, int]:
    return _open_digest_bound_file(
        path,
        expected_sha256,
        where,
        require_executable=True,
    )


def _open_digest_bound_regular_file(
    path: Any, expected_sha256: Any, where: str
) -> tuple[str, int]:
    return _open_digest_bound_file(
        path,
        expected_sha256,
        where,
        require_executable=False,
    )


def _open_scheduler_commands(
    specifications: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, str], tuple[int, ...]]:
    commands: dict[str, str] = {}
    descriptors: list[int] = []
    try:
        for name in sorted(specifications):
            command, descriptor = _open_digest_bound_executable(
                specifications[name]["path"],
                specifications[name]["sha256"],
                f"scheduler {name}",
            )
            commands[name] = command
            descriptors.append(descriptor)
    except BaseException:
        for descriptor in descriptors:
            os.close(descriptor)
        raise
    return commands, tuple(descriptors)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixed helper {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _require_work_artifact_filesystem_isolation(
    *, exact_root: Path, work_root: Path, artifact_directory: Path
) -> dict[str, Any]:
    helper = _load_module(
        "_h100_submit_filesystem_isolation",
        exact_root / "scripts/verify_filesystem_isolation.py",
    )
    return dict(
        helper.require_distinct_filesystems(
            work_root,
            artifact_directory,
            work_label="H100 case work root",
            artifact_label="H100 artifact filesystem",
        )
    )


def _query_repository_binding(
    *, exact_root: Path, git: Path, git_sha256: str, commit_sha: str
) -> dict[str, Any]:
    helper = _load_module(
        "_h100_submit_git_repository",
        exact_root / "scripts/verify_git_repository.py",
    )
    return dict(
        helper.query_exact_repository(
            git=git,
            git_sha256=git_sha256,
            expected_commit_sha=commit_sha,
        )
    )


def _run(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    where: str,
    check: bool = True,
    pass_fds: Sequence[int] = (),
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(argv),
        env=None if env is None else dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=SCHEDULER_COMMAND_TIMEOUT_SECONDS,
        pass_fds=tuple(pass_fds),
    )
    if check and completed.returncode != 0:
        raise RuntimeError(f"{where} failed with status {completed.returncode}: {completed.stderr.strip()}")
    return completed


def _verify_environment_transaction(
    *,
    exact_root: Path,
    python: Path,
    source_commit_sha: str,
    source_tree_sha: str,
    git_sha256: str,
    environments: Mapping[str, str],
    specification: Mapping[str, Any],
    verifier_sha256: str,
) -> dict[str, Any]:
    completion_path = _absolute(
        specification["completion_path"], "environment completion marker"
    )
    verifier_command, verifier_fd = _open_digest_bound_regular_file(
        exact_root / "scripts/verify_formal_environment_transaction.py",
        verifier_sha256,
        "formal environment transaction verifier",
    )
    try:
        completed = _run(
            [
                os.fspath(python),
                "-I",
                "-B",
                verifier_command,
                "--completion",
                os.fspath(completion_path),
                "--expected-completion-sha256",
                specification["completion_sha256"],
                "--expected-transaction-sha256",
                specification["transaction_sha256"],
                "--expected-one-gpu-environment-sha256",
                environments["1"],
                "--expected-two-gpu-environment-sha256",
                environments["2"],
                "--expected-inventory-sha256",
                specification["inventory_sha256"],
                "--expected-source-commit-sha",
                source_commit_sha,
                "--expected-source-tree-sha",
                source_tree_sha,
                "--expected-git-sha256",
                git_sha256,
                "--completion-max-bytes",
                str(specification["completion_max_bytes"]),
                "--manifest-max-bytes",
                str(specification["manifest_max_bytes"]),
                "--inventory-max-bytes",
                str(specification["inventory_max_bytes"]),
            ],
            env={
                "PATH": "/usr/bin:/bin",
                "LC_ALL": "C",
                "LANG": "C",
                "PYTHONPATH": os.fspath(exact_root / "src"),
                "PYTHONNOUSERSITE": "1",
            },
            where="formal environment transaction verification",
            pass_fds=(verifier_fd,),
        )
    finally:
        os.close(verifier_fd)
    if completed.stderr:
        _fail("formal environment transaction verifier emitted stderr")
    summary = _strict_json_bytes(
        completed.stdout.encode("utf-8"),
        where="formal environment transaction verifier output",
        max_bytes=1 << 20,
    )
    _exact_keys(
        summary,
        {
            "schema_version",
            "kind",
            "completion_raw_sha256",
            "transaction_sha256",
            "source_commit_sha",
            "source_tree_sha",
            "git_sha256",
            "one_gpu_environment_sha256",
            "two_gpu_environment_sha256",
            "inventory_sha256",
            "installed_distributions_sha256",
            "capture_visible_cuda_device_count",
            "output_raw_sha256_by_role",
        },
        "formal environment transaction verification summary",
    )
    output_digests = summary["output_raw_sha256_by_role"]
    _exact_keys(
        output_digests,
        {"one_gpu", "two_gpu", "inventory"},
        "formal environment transaction raw-output map",
    )
    for role, digest in output_digests.items():
        _digest(digest, f"formal environment {role} raw output")
    capture_count = summary["capture_visible_cuda_device_count"]
    if (
        isinstance(capture_count, bool)
        or not isinstance(capture_count, int)
        or capture_count < 0
    ):
        _fail("formal environment capture GPU count is invalid")
    expected = {
        "schema_version": 1,
        "kind": "formal_environment_transaction_verification",
        "completion_raw_sha256": specification["completion_sha256"],
        "transaction_sha256": specification["transaction_sha256"],
        "source_commit_sha": source_commit_sha,
        "source_tree_sha": source_tree_sha,
        "git_sha256": git_sha256,
        "one_gpu_environment_sha256": environments["1"],
        "two_gpu_environment_sha256": environments["2"],
        "inventory_sha256": specification["inventory_sha256"],
    }
    if any(summary[key] != value for key, value in expected.items()):
        _fail("formal environment transaction verifier returned a binding mismatch")
    _digest(
        summary["installed_distributions_sha256"],
        "formal environment installed-distribution inventory",
    )
    return {
        **dict(summary),
        "output_raw_sha256_by_role": dict(output_digests),
    }


def _git(git_command: str, git_fd: int, root: Path, *args: str) -> str:
    return _run(
        [git_command, *LOCAL_GIT_CONFIG_OVERRIDES, "-C", os.fspath(root), *args],
        env={
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "LANG": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_COUNT": "0",
            "GIT_CEILING_DIRECTORIES": "/",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_PROTOCOL_FROM_USER": "0",
            "GIT_ALLOW_PROTOCOL": "https",
        },
        where=f"git {' '.join(args)}",
        pass_fds=(git_fd,),
    ).stdout.strip()


def _attest_exact_checkout(
    *,
    root: Path,
    git_command: str,
    git_fd: int,
    commit_sha: str,
    tree_sha: str,
    source: Mapping[str, Any],
) -> Mapping[str, Any]:
    _require_directory(root, where="exact candidate root")
    if Path(__file__).resolve(strict=True) != (root / "scripts/submit_h100_identity_validation.py").resolve(strict=True):
        _fail("submission controller is not running from exact candidate root")
    if _git(git_command, git_fd, root, "rev-parse", "HEAD") != commit_sha:
        _fail("exact checkout commit mismatch")
    if _git(git_command, git_fd, root, "rev-parse", "HEAD^{tree}") != tree_sha:
        _fail("exact checkout tree mismatch")
    if _git(
        git_command,
        git_fd,
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ):
        _fail("exact checkout is dirty")
    symbolic = _run(
        [
            git_command,
            *LOCAL_GIT_CONFIG_OVERRIDES,
            "-C",
            os.fspath(root),
            "symbolic-ref",
            "-q",
            "HEAD",
        ],
        env={
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "LANG": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_COUNT": "0",
            "GIT_CEILING_DIRECTORIES": "/",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_PROTOCOL_FROM_USER": "0",
            "GIT_ALLOW_PROTOCOL": "https",
        },
        where="detached checkout check",
        check=False,
        pass_fds=(git_fd,),
    )
    if symbolic.returncode == 0:
        _fail("exact checkout must be detached")
    if symbolic.returncode != 1:
        _fail("cannot prove exact checkout is detached")

    verifier = _load_module("_h100_submit_source_verifier", root / "scripts/verify_runtime_source.py")
    manifest = verifier.load_and_validate_source_manifest(
        Path(source["manifest_path"]), source["manifest_sha256"], commit_sha, tree_sha
    )
    required = {
        "scripts/submit_h100_identity_validation.py",
        "scripts/submit_h100_identity_validation.sh",
        "scripts/run_h100_identity_validation.py",
        "scripts/run_h100_identity_maxseq_smoke.sh",
        "scripts/run_slurm_h100_identity_case.sh",
        "scripts/formal_run_with_gpu_monitor.sh",
        "scripts/run_with_durable_log.sh",
        "scripts/exec_digest_bound_git.py",
        "scripts/exec_digest_bound_nvidia_smi.py",
        "scripts/verify_formal_environment.py",
        "scripts/verify_formal_environment_transaction.py",
        _ONE_GPU_WRAPPER,
        _TWO_GPU_WRAPPER,
    }
    verifier.require_manifest_coverage(
        manifest,
        required_paths=required,
        required_code_roots=("scripts", "src/tabicl"),
    )
    verifier.verify_archive(root, manifest)
    return manifest


def _scheduler_environment() -> dict[str, str]:
    return {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"}


def _export_argument(exports: Mapping[str, str]) -> str:
    if "ALL" in exports or "PYTHONHOME" in exports:
        _fail("job export allowlist is contaminated")
    items: list[str] = []
    for key in sorted(exports):
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is None:
            _fail("job export name is malformed")
        value = exports[key]
        if not value or any(character in value for character in (",", "\x00", "\n", "\r")):
            _fail("job export value is empty or unsafe")
        items.append(f"{key}={value}")
    return "--export=" + ",".join(items)


def _make_envelope(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    body = {"schema_version": 1, "kind": kind, "payload": dict(payload)}
    return {**body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def _write_all(fd: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        written = os.write(fd, raw[offset:])
        if written <= 0:
            raise OSError("short write")
        offset += written


class _PublicationUncertain(RuntimeError):
    def __init__(self, path: Path):
        super().__init__(f"publication visibility could not be revoked: {path.name}")
        self.path = path


class _PublicationInterrupted(BaseException):
    """A process interrupt observed after publication became durable."""

    def __init__(self, path: Path, interruption: BaseException):
        super().__init__(f"publication committed while interrupted: {path.name}")
        self.path = path
        self.interruption = interruption


def _publication_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_published_bytes(parent_fd: int, name: str, *, max_bytes: int) -> bytes:
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    fd = os.open(name, flags, dir_fd=parent_fd)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > max_bytes:
            raise ValueError("published artifact is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                raise ValueError("published artifact was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if _publication_signature(os.fstat(fd)) != _publication_signature(opened):
            raise ValueError("published artifact changed during verification")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _publish_no_replace(
    path: Path,
    value: Mapping[str, Any],
    *,
    max_bytes: int,
    fault: str | None = None,
) -> Path:
    raw = _canonical(value) + b"\n"
    if len(raw) > max_bytes:
        _fail(f"publication exceeds ceiling: {path.name}")
    parent_fd = _open_directory_nofollow(path.parent, where="publication parent")
    temporary = f".{path.name}.{secrets.token_hex(16)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    fd = -1
    linked = False
    durably_committed = False
    durably_revoked = False
    try:
        fd = os.open(temporary, flags, 0o400, dir_fd=parent_fd)
        _write_all(fd, raw)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.link(temporary, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd, follow_symlinks=False)
        linked = True
        if fault == "after_link":
            raise OSError("injected publication failure after link")
        os.fsync(parent_fd)
        if fault == "after_link_fsync":
            raise OSError("injected publication failure after link fsync")
        os.unlink(temporary, dir_fd=parent_fd)
        if fault == "after_temp_unlink":
            raise OSError("injected publication failure after temporary unlink")
        os.fsync(parent_fd)
        durably_committed = True
        return path
    except BaseException as error:
        if not linked:
            # Python may deliver a process signal after link(2) committed but
            # before the following assignment ran.  Recover that exact window
            # only when the final and private temporary names are hard links
            # to the same regular inode; an unrelated pre-existing final must
            # remain a no-replace collision.
            try:
                temporary_info = os.stat(
                    temporary, dir_fd=parent_fd, follow_symlinks=False
                )
                published_info = os.stat(
                    path.name, dir_fd=parent_fd, follow_symlinks=False
                )
            except OSError:
                pass
            else:
                linked = (
                    stat.S_ISREG(temporary_info.st_mode)
                    and stat.S_ISREG(published_info.st_mode)
                    and temporary_info.st_dev == published_info.st_dev
                    and temporary_info.st_ino == published_info.st_ino
                )
        if linked:
            completion_error: BaseException | None = None
            try:
                if _read_published_bytes(
                    parent_fd, path.name, max_bytes=max_bytes
                ) != raw:
                    raise ValueError("published artifact differs from intended bytes")
                # A second directory sync closes a one-shot post-link failure.
                os.fsync(parent_fd)
                try:
                    os.unlink(temporary, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
                os.fsync(parent_fd)
                durably_committed = True
            except BaseException as caught:
                completion_error = caught
            if completion_error is None:
                if not isinstance(error, Exception):
                    raise _PublicationInterrupted(path, error) from error
                return path
            else:
                removed = False
                try:
                    os.unlink(path.name, dir_fd=parent_fd)
                    removed = True
                except FileNotFoundError:
                    removed = True
                except OSError:
                    pass
                if removed:
                    try:
                        os.fsync(parent_fd)
                    except BaseException as revoke_error:
                        raise _PublicationUncertain(path) from revoke_error
                    durably_committed = False
                    durably_revoked = True
                else:
                    raise _PublicationUncertain(path) from completion_error
        raise error
    finally:
        active_error = sys.exc_info()[1]
        cleanup_error: BaseException | None = None
        if fd >= 0:
            try:
                os.close(fd)
            except BaseException as error:
                cleanup_error = error
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        try:
            os.close(parent_fd)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        if cleanup_error is not None:
            if durably_committed:
                if not isinstance(cleanup_error, Exception):
                    raise _PublicationInterrupted(path, cleanup_error) from cleanup_error
                # The final name and removal of the private temporary name
                # were already directory-synced.  A redundant cleanup/close
                # error cannot turn that durable commit into a rollback.
            elif active_error is None:
                if linked and not durably_revoked:
                    raise _PublicationUncertain(path) from cleanup_error
                raise cleanup_error


class _Journal:
    def __init__(self, path: Path, transaction_id: str, max_bytes: int):
        self.path = path
        self.transaction_id = transaction_id
        self.max_bytes = max_bytes
        self.sequence = 0
        self.head = "0" * 64

    def append(self, event: str, payload: Mapping[str, Any]) -> None:
        self.sequence += 1
        body = {
            "schema_version": 1,
            "transaction_id": self.transaction_id,
            "sequence": self.sequence,
            "event": event,
            "payload": dict(payload),
            "previous_sha256": self.head,
        }
        record = {**body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}
        raw = _canonical(record) + b"\n"
        parent_fd = _open_directory_nofollow(self.path.parent, where="journal parent")
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        fd = -1
        try:
            fd = os.open(self.path.name, flags, 0o600, dir_fd=parent_fd)
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size + len(raw) > self.max_bytes:
                _fail("transaction journal exceeds its byte ceiling")
            _write_all(fd, raw)
            os.fsync(fd)
            os.close(fd)
            fd = -1
            os.fsync(parent_fd)
        finally:
            if fd >= 0:
                os.close(fd)
            os.close(parent_fd)
        self.head = record["sha256"]

    def snapshot(self) -> tuple[str, int]:
        fd = os.open(
            self.path,
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        try:
            digest = hashlib.sha256()
            while True:
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
        finally:
            os.close(fd)
        return digest.hexdigest(), self.sequence


def _safe_journal(journal: _Journal, event: str, payload: Mapping[str, Any]) -> None:
    try:
        journal.append(event, payload)
    except Exception:
        # Rollback must continue even if the diagnostic journal itself has
        # reached its precommitted ceiling or the filesystem has failed.
        pass


def _cluster_argv(cluster: str | None) -> list[str]:
    return [] if cluster is None else [f"--clusters={cluster}"]


def _scheduler_state_token(raw: str) -> str | None:
    token = raw.split()[0].split("+")[0].upper() if raw.split() else ""
    return token if re.fullmatch(r"[A-Z][A-Z_]*", token) is not None else None


def _scheduler_state(
    commands: Mapping[str, str],
    scheduler_env: Mapping[str, str],
    scheduler_fds: Sequence[int],
    *,
    job_id: str,
    cluster: str | None,
) -> tuple[str, str, str]:
    query = _run(
        [
            commands["squeue"],
            *_cluster_argv(cluster),
            "--noheader",
            f"--jobs={job_id}",
            "--format=%i|%T|%r",
        ],
        env=scheduler_env,
        where=f"squeue reconciliation {job_id}",
        check=False,
        pass_fds=scheduler_fds,
    )
    queue_rows: list[tuple[str, str]] = []
    if query.returncode == 0:
        for row in query.stdout.splitlines():
            fields = [field.strip() for field in row.strip().split("|", 2)]
            if len(fields) != 3 or fields[0] != job_id:
                continue
            state = _scheduler_state_token(fields[1])
            if state is not None:
                queue_rows.append((state, fields[2]))
    if len(queue_rows) > 1:
        raise RuntimeError(f"ambiguous squeue state rows for job {job_id}")
    if len(queue_rows) == 1:
        state, reason = queue_rows[0]
        return state, reason, "squeue"
    accounting = _run(
        [
            commands["sacct"],
            *_cluster_argv(cluster),
            "--noheader",
            "--allocations",
            f"--jobs={job_id}",
            "--format=JobIDRaw,State",
            "--parsable2",
        ],
        env=scheduler_env,
        where=f"sacct reconciliation {job_id}",
        check=False,
        pass_fds=scheduler_fds,
    )
    accounting_rows: list[str] = []
    if accounting.returncode == 0:
        for row in accounting.stdout.splitlines():
            fields = [field.strip() for field in row.strip().split("|", 2)]
            if len(fields) < 2 or fields[0] != job_id:
                continue
            state = _scheduler_state_token(fields[1])
            if state is not None:
                accounting_rows.append(state)
    if len(accounting_rows) > 1:
        raise RuntimeError(f"ambiguous sacct state rows for job {job_id}")
    if len(accounting_rows) == 1:
        # Accounting proves lifecycle state but supplies no authoritative
        # pending reason.  In particular, PENDING here cannot prove that a
        # release removed JobHeldUser/JobHeldAdmin.
        return accounting_rows[0], "", "sacct"
    raise RuntimeError(f"cannot reconcile scheduler state for job {job_id}")


def _scheduler_state_bounded_retry(
    commands: Mapping[str, str],
    scheduler_env: Mapping[str, str],
    scheduler_fds: Sequence[int],
    *,
    job_id: str,
    cluster: str | None,
    acceptable,
    attempts: int = 3,
) -> tuple[str, str, str]:
    """Reconcile bounded scheduler propagation without weakening fail-closed state."""

    last: tuple[str, str, str] | None = None
    last_error: Exception | None = None
    for _attempt in range(attempts):
        try:
            last = _scheduler_state(
                commands,
                scheduler_env,
                scheduler_fds,
                job_id=job_id,
                cluster=cluster,
            )
            if acceptable(*last):
                return last
        except Exception as error:
            last_error = error
    if last is not None:
        raise RuntimeError(
            f"scheduler state for job {job_id} did not converge after {attempts} queries: "
            f"{last[0]}"
        )
    raise RuntimeError(
        f"scheduler state for job {job_id} was unavailable after {attempts} queries"
    ) from last_error


def _released_state_is_acceptable(state: str, reason: str, source: str) -> bool:
    return state in _TERMINAL_STATES | {"CONFIGURING", "COMPLETING", "RUNNING"} or (
        state == "PENDING"
        and source == "squeue"
        and bool(reason)
        and "held" not in reason.lower()
    )


def _terminal_state_is_acceptable(state: str, _reason: str, _source: str) -> bool:
    return state in _TERMINAL_STATES


def _transaction_job_name(transaction_id: str, position: int) -> str:
    if re.fullmatch(r"[0-9a-f]{32}", transaction_id or "") is None:
        _fail("transaction ID is malformed for scheduler job name")
    if isinstance(position, bool) or not isinstance(position, int) or not 1 <= position <= 99:
        _fail("scheduler job position is invalid")
    return f"tabicl-gate-{transaction_id}-{position:02d}"


def _discover_transaction_jobs(
    commands: Mapping[str, str],
    scheduler_env: Mapping[str, str],
    scheduler_fds: Sequence[int],
    *,
    job_name: str,
) -> list[dict[str, str | None]]:
    """Find jobs after an sbatch response-loss window by exact unique name."""

    query = _run(
        [
            commands["squeue"],
            "--noheader",
            f"--name={job_name}",
            "--format=%i|%j|%T|%r",
        ],
        env=scheduler_env,
        where=f"squeue response-loss discovery {job_name}",
        check=False,
        pass_fds=scheduler_fds,
    )
    found: dict[str, dict[str, str | None]] = {}
    if query.returncode == 0:
        for row in query.stdout.splitlines():
            fields = [field.strip() for field in row.strip().split("|", 3)]
            if (
                len(fields) != 4
                or _JOB_ID.fullmatch(fields[0]) is None
                or fields[1] != job_name
            ):
                continue
            state = _scheduler_state_token(fields[2])
            if state is None:
                continue
            found[fields[0]] = {
                "job_id": fields[0],
                "cluster": None,
                "state": state,
                "reason": fields[3],
                "source": "squeue",
            }
    if found:
        return [found[job_id] for job_id in sorted(found, key=int)]

    accounting = _run(
        [
            commands["sacct"],
            "--noheader",
            "--allocations",
            f"--name={job_name}",
            "--format=JobIDRaw,JobName,State",
            "--parsable2",
        ],
        env=scheduler_env,
        where=f"sacct response-loss discovery {job_name}",
        check=False,
        pass_fds=scheduler_fds,
    )
    if accounting.returncode == 0:
        for row in accounting.stdout.splitlines():
            fields = [field.strip() for field in row.strip().split("|", 3)]
            if (
                len(fields) < 3
                or _JOB_ID.fullmatch(fields[0]) is None
                or fields[1] != job_name
            ):
                continue
            state = _scheduler_state_token(fields[2])
            if state is None:
                continue
            found[fields[0]] = {
                "job_id": fields[0],
                "cluster": None,
                "state": state,
                "reason": "",
                "source": "sacct",
            }
    return [found[job_id] for job_id in sorted(found, key=int)]


_TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}


def _create_fresh_namespace(root: Path) -> Path:
    if root.exists() or root.is_symlink():
        _fail("H100 validation artifact namespace is fresh-only")
    parent_fd = _open_directory_nofollow(root.parent, where="artifact namespace parent")
    try:
        os.mkdir(root.name, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    root_fd = _open_directory_nofollow(root, where="artifact namespace")
    try:
        os.mkdir("cases", 0o700, dir_fd=root_fd)
        os.mkdir("scheduler-logs", 0o700, dir_fd=root_fd)
        os.mkdir("job-cwd", 0o700, dir_fd=root_fd)
        os.fsync(root_fd)
    finally:
        os.close(root_fd)
    return root / "cases"


def _parse_sbatch(raw: str) -> tuple[str, str | None]:
    match = _SBATCH_RESULT.fullmatch(raw)
    if match is None:
        _fail("sbatch returned an empty or malformed JOBID[;CLUSTER]")
    parts = raw.split(";", 1)
    return match.group(1), (parts[1] if len(parts) == 2 else None)


def _rollback(
    commands: Mapping[str, str],
    scheduler_env: Mapping[str, str],
    scheduler_fds: Sequence[int],
    accepted: Sequence[Mapping[str, Any]],
    journal: _Journal,
) -> tuple[list[str], list[str]]:
    cancelled: list[str] = []
    remaining: list[str] = []
    for item in reversed(accepted):
        job_id = item["job_id"]
        cluster = item["cluster"]
        _safe_journal(journal, "cancel_attempt", {"job_id": job_id, "cluster": cluster})
        try:
            completed = _run(
                [commands["scancel"], *_cluster_argv(cluster), job_id],
                env=scheduler_env,
                where=f"rollback cancellation {job_id}",
                check=False,
                pass_fds=scheduler_fds,
            )
        except BaseException as cancel_error:
            remaining.append(job_id)
            _safe_journal(
                journal,
                "cancel_exec_failed",
                {
                    "job_id": job_id,
                    "cluster": cluster,
                    "error_type": type(cancel_error).__name__,
                },
            )
            continue
        if completed.returncode != 0:
            remaining.append(job_id)
            _safe_journal(journal, "cancel_command_failed", {"job_id": job_id, "cluster": cluster})
            continue
        try:
            state, reason, source = _scheduler_state_bounded_retry(
                commands,
                scheduler_env,
                scheduler_fds,
                job_id=job_id,
                cluster=cluster,
                acceptable=_terminal_state_is_acceptable,
            )
        except Exception:
            remaining.append(job_id)
            _safe_journal(journal, "cancel_reconcile_failed", {"job_id": job_id, "cluster": cluster})
            continue
        if state in _TERMINAL_STATES:
            cancelled.append(job_id)
            _safe_journal(
                journal,
                "cancel_reconciled",
                {
                    "job_id": job_id,
                    "cluster": cluster,
                    "state": state,
                    "reason": reason,
                    "source": source,
                },
            )
        else:
            remaining.append(job_id)
            _safe_journal(
                journal,
                "cancel_not_terminal",
                {
                    "job_id": job_id,
                    "cluster": cluster,
                    "state": state,
                    "reason": reason,
                    "source": source,
                },
            )
    return cancelled, remaining


def _validate_overlay(value: Mapping[str, Any], *, exact_root: Path) -> dict[str, Any]:
    _exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "run_policy",
            "validation_id",
            "artifact_root",
            "source",
            "environment_sha256_by_world_size",
            "environment_transaction",
            "runtime",
            "limits",
            "scheduler_commands",
        },
        "H100 gate overlay",
    )
    if value["schema_version"] != 1 or value["kind"] != "h100_identity_gate_submit":
        _fail("H100 gate overlay schema/kind mismatch")
    if value["run_policy"] != "fresh":
        _fail("H100 gate overlay is fresh-only")
    validation_id = value["validation_id"]
    if not isinstance(validation_id, str) or _SAFE_ID.fullmatch(validation_id) is None:
        _fail("validation ID is malformed")
    artifact_root = _absolute(value["artifact_root"], "artifact root")
    if artifact_root == exact_root or exact_root in artifact_root.parents:
        _fail("H100 validation artifacts must be outside exact candidate")
    _require_directory(artifact_root.parent, where="artifact namespace parent")

    source = value["source"]
    _exact_keys(
        source,
        {
            "candidate_repository",
            "candidate_ref",
            "commit_sha",
            "tree_sha",
            "manifest_path",
            "manifest_sha256",
        },
        "overlay source",
    )
    commit_sha = source["commit_sha"]
    tree_sha = source["tree_sha"]
    if not isinstance(commit_sha, str) or _HEX40.fullmatch(commit_sha) is None:
        _fail("source commit is malformed")
    if not isinstance(tree_sha, str) or _HEX40.fullmatch(tree_sha) is None:
        _fail("source tree is malformed")
    _absolute(source["manifest_path"], "source manifest")
    source_sha = _digest(source["manifest_sha256"], "source manifest")
    repository = source["candidate_repository"]
    repository_ref = source["candidate_ref"]
    if repository != "https://github.com/kikixiong/tabicl-pe.git":
        _fail("candidate repository must be the canonical public GitHub repository")
    if repository_ref != "refs/heads/codex/position-identity-v1":
        _fail("candidate ref must be the canonical immutable-training branch")

    environments = value["environment_sha256_by_world_size"]
    _exact_keys(environments, {"1", "2"}, "environment digest map")
    environments = {key: _digest(item, f"{key}-GPU expected environment") for key, item in environments.items()}
    if environments["1"] == environments["2"]:
        _fail("one- and two-GPU environment digests must differ")

    runtime = value["runtime"]
    _exact_keys(
        runtime,
        {
            "python", "git", "git_sha256", "nvidia_smi",
            "nvidia_smi_sha256", "case_work_root",
        },
        "overlay runtime",
    )
    python = _require_executable(runtime["python"], "runtime Python")
    git = _absolute(runtime["git"], "runtime Git")
    git_sha256 = _digest(runtime["git_sha256"], "runtime Git")
    nvidia_smi = _absolute(runtime["nvidia_smi"], "runtime nvidia-smi")
    nvidia_smi_sha256 = _digest(
        runtime["nvidia_smi_sha256"], "runtime nvidia-smi"
    )
    _nvidia_command, nvidia_fd = _open_digest_bound_executable(
        nvidia_smi, nvidia_smi_sha256, "runtime nvidia-smi"
    )
    os.close(nvidia_fd)
    case_work_root = _require_directory(
        _absolute(runtime["case_work_root"], "case work root"), where="case work root"
    )
    if (
        case_work_root == exact_root
        or exact_root in case_work_root.parents
        or case_work_root in exact_root.parents
        or case_work_root == artifact_root
        or artifact_root in case_work_root.parents
        or case_work_root in artifact_root.parents
    ):
        _fail("case work root must not overlap exact source or artifact namespace")

    environment_transaction = value["environment_transaction"]
    _exact_keys(
        environment_transaction,
        {
            "completion_path",
            "completion_sha256",
            "transaction_sha256",
            "inventory_sha256",
            "completion_max_bytes",
            "manifest_max_bytes",
            "inventory_max_bytes",
        },
        "formal environment transaction",
    )
    completion_path = _absolute(
        environment_transaction["completion_path"],
        "formal environment completion marker",
    )
    completion_parent = _require_directory(
        completion_path.parent, where="formal environment transaction directory"
    )
    for protected_root, label in (
        (exact_root, "exact source"),
        (artifact_root, "artifact namespace"),
        (case_work_root, "case work root"),
    ):
        if (
            completion_parent == protected_root
            or protected_root in completion_parent.parents
            or completion_parent in protected_root.parents
        ):
            _fail(
                "formal environment transaction directory must not overlap "
                f"{label}"
            )
    fixed_environment_transaction = {
        "completion_path": os.fspath(completion_path),
        "completion_sha256": _digest(
            environment_transaction["completion_sha256"],
            "formal environment completion marker",
        ),
        "transaction_sha256": _digest(
            environment_transaction["transaction_sha256"],
            "formal environment transaction",
        ),
        "inventory_sha256": _digest(
            environment_transaction["inventory_sha256"],
            "formal environment inventory",
        ),
        "completion_max_bytes": _positive(
            environment_transaction["completion_max_bytes"],
            "formal environment completion ceiling",
        ),
        "manifest_max_bytes": _positive(
            environment_transaction["manifest_max_bytes"],
            "formal environment manifest ceiling",
        ),
        "inventory_max_bytes": _positive(
            environment_transaction["inventory_max_bytes"],
            "formal environment inventory ceiling",
        ),
    }
    expected_environment_ceilings = {
        "completion_max_bytes": _ENVIRONMENT_COMPLETION_MAX_BYTES,
        "manifest_max_bytes": _ENVIRONMENT_MANIFEST_MAX_BYTES,
        "inventory_max_bytes": _ENVIRONMENT_INVENTORY_MAX_BYTES,
    }
    if any(
        fixed_environment_transaction[key] != expected
        for key, expected in expected_environment_ceilings.items()
    ):
        _fail("formal environment transaction ceilings are not canonical")

    scheduler_commands = value["scheduler_commands"]
    command_names = {"sbatch", "scontrol", "scancel", "squeue", "sacct"}
    _exact_keys(scheduler_commands, command_names, "scheduler command map")
    fixed_commands: dict[str, dict[str, str]] = {}
    scheduler_digests: dict[str, str] = {}
    for name in sorted(command_names):
        entry = scheduler_commands[name]
        _exact_keys(entry, {"path", "sha256"}, f"scheduler command {name}")
        fixed_commands[name] = {
            "path": os.fspath(_absolute(entry["path"], f"scheduler {name}")),
            "sha256": _digest(entry["sha256"], f"scheduler {name}"),
        }
        scheduler_digests[name] = fixed_commands[name]["sha256"]

    limits = value["limits"]
    _exact_keys(
        limits,
        {
            "checkpoint_ceiling_bytes",
            "run_log_ceiling_bytes",
            "gpu_monitor_ceiling_bytes",
            "attestation_ceiling_bytes",
            "receipt_ceiling_bytes",
            "scheduler_log_ceiling_bytes",
        },
        "overlay limits",
    )
    limits = {key: _positive(item, key) for key, item in limits.items()}

    git_command, git_fd = _open_digest_bound_executable(
        git, git_sha256, "runtime Git"
    )
    try:
        manifest = _attest_exact_checkout(
            root=exact_root,
            git_command=git_command,
            git_fd=git_fd,
            commit_sha=commit_sha,
            tree_sha=tree_sha,
            source=source,
        )
    finally:
        os.close(git_fd)
    if manifest["sha256"] != source_sha:
        _fail("validated source manifest digest mismatch")
    manifest_entries = {
        entry["path"]: entry for entry in manifest["payload"]["entries"]
    }
    environment_verifier_sha256 = _digest(
        manifest_entries["scripts/verify_formal_environment_transaction.py"][
            "sha256"
        ],
        "formal environment transaction verifier source",
    )
    repository_binding = _query_repository_binding(
        exact_root=exact_root,
        git=git,
        git_sha256=git_sha256,
        commit_sha=commit_sha,
    )
    environment_transaction_summary = _verify_environment_transaction(
        exact_root=exact_root,
        python=python,
        source_commit_sha=commit_sha,
        source_tree_sha=tree_sha,
        git_sha256=git_sha256,
        environments=environments,
        specification=fixed_environment_transaction,
        verifier_sha256=environment_verifier_sha256,
    )
    filesystem_isolation = _require_work_artifact_filesystem_isolation(
        exact_root=exact_root,
        work_root=case_work_root,
        artifact_directory=artifact_root.parent,
    )
    matrix = _load_module(
        "_h100_submit_matrix", exact_root / "scripts/run_h100_identity_validation.py"
    )
    cases = matrix.build_matrix()
    if len(cases) != 12 or sum(case.world_size == 2 for case in cases) != 1:
        _fail("canonical H100 matrix resource composition has drifted")
    if next(case for case in cases if case.world_size == 2).case_id != "nccl_2gpu":
        _fail("only nccl_2gpu may request two GPUs")

    jobs: list[dict[str, Any]] = []
    for position, case in enumerate(cases, start=1):
        world_size = case.world_size
        environment_sha = environments[str(world_size)]
        requested_resource = matrix.gate_requested_resource(case)
        requested_resource_sha = matrix.gate_requested_resource_sha256(case)
        identity = matrix.submission_artifact_identity(
            validation_id=validation_id,
            case_id=case.case_id,
            world_size=world_size,
            commit_sha=commit_sha,
            tree_sha=tree_sha,
            source_manifest_sha256=source_sha,
            expected_environment_sha256=environment_sha,
            requested_resource_sha256=requested_resource_sha,
            checkpoint_ceiling_bytes=limits["checkpoint_ceiling_bytes"],
        )
        exports = {
            "RUN_POLICY": "fresh",
            "VALIDATION_CASE_ID": case.case_id,
            "FORMAL_EXPECTED_GPUS": str(world_size),
            "CANDIDATE_REPOSITORY": repository,
            "CANDIDATE_REPOSITORY_REF": repository_ref,
            "FORMAL_SOURCE_COMMIT_SHA": commit_sha,
            "FORMAL_SOURCE_TREE_SHA": tree_sha,
            "FORMAL_SOURCE_MANIFEST": source["manifest_path"],
            "FORMAL_SOURCE_SHA256": source_sha,
            "FORMAL_GIT_SHA256": git_sha256,
            "FORMAL_REPOSITORY_IDENTITY_SHA256": repository_binding[
                "repository_identity_sha256"
            ],
            "FORMAL_REPOSITORY_QUERY_SHA256": repository_binding["query_sha256"],
            "FORMAL_EXPECTED_ENVIRONMENT_SHA256": environment_sha,
            "FORMAL_REQUESTED_RESOURCE_SHA256": requested_resource_sha,
            "FORMAL_REQUESTED_PARTITION": requested_resource["partition"],
            "FORMAL_REQUESTED_QOS": requested_resource["qos"],
            "FORMAL_REQUESTED_TIME_LIMIT": requested_resource["time_limit"],
            "FORMAL_REQUESTED_NODES": str(requested_resource["nodes"]),
            "FORMAL_REQUESTED_GPUS": str(requested_resource["gpus"]),
            "FORMAL_REQUESTED_CPUS": str(requested_resource["cpus_per_task"]),
            "FORMAL_REQUESTED_MEMORY_MB": str(requested_resource["memory_mb"]),
            "VALIDATION_ARTIFACT_IDENTITY_SHA256": identity,
            "H100_CASE_WORK_ROOT": os.fspath(case_work_root),
            "H100_VALIDATION_ROOT": os.fspath(artifact_root / "cases"),
            "CHECKPOINT_CEILING_BYTES": str(limits["checkpoint_ceiling_bytes"]),
            "FORMAL_RUN_LOG_CEILING_BYTES": str(limits["run_log_ceiling_bytes"]),
            "FORMAL_GPU_MONITOR_CEILING_BYTES": str(limits["gpu_monitor_ceiling_bytes"]),
            "FORMAL_ATTESTATION_CEILING_BYTES": str(limits["attestation_ceiling_bytes"]),
            "PYTHON": os.fspath(python),
            "GIT": os.fspath(git),
            "NVIDIA_SMI": os.fspath(nvidia_smi),
            "FORMAL_NVIDIA_SMI_SHA256": nvidia_smi_sha256,
            "PYTHONPATH": os.fspath(exact_root / "src"),
            "PYTHONNOUSERSITE": "1",
            "PATH": "/usr/bin:/bin",
            "FORMAL_SUBMISSION_EXACT_ROOT": os.fspath(exact_root),
        }
        _export_argument(exports)
        jobs.append(
            {
                "case_id": case.case_id,
                "position": position,
                "world_size": world_size,
                "artifact_identity_sha256": identity,
                "requested_resource_sha256": requested_resource_sha,
                "requested_resource": requested_resource,
                "exports": exports,
            }
        )

    wrapper_commands: dict[str, str] = {}
    wrapper_fds: list[int] = []
    try:
        for world_size, wrapper in (
            ("1", _ONE_GPU_WRAPPER),
            ("2", _TWO_GPU_WRAPPER),
        ):
            command, descriptor = _open_digest_bound_executable(
                exact_root / wrapper,
                manifest_entries[wrapper]["sha256"],
                f"{world_size}-GPU Slurm spool wrapper",
            )
            wrapper_commands[world_size] = command
            wrapper_fds.append(descriptor)
    except BaseException:
        for descriptor in wrapper_fds:
            os.close(descriptor)
        raise

    return {
        "validation_id": validation_id,
        "artifact_root": artifact_root,
        "exact_root": exact_root,
        "source": {
            "commit_sha": commit_sha,
            "tree_sha": tree_sha,
            "manifest_sha256": source_sha,
        },
        "environment_sha256_by_world_size": environments,
        "environment_transaction": environment_transaction_summary,
        "nvidia_smi_sha256": nvidia_smi_sha256,
        "limits": limits,
        "scheduler_commands": fixed_commands,
        "scheduler_command_sha256": scheduler_digests,
        "repository_binding": repository_binding,
        "filesystem_isolation": filesystem_isolation,
        "jobs": jobs,
        "wrapper_commands_by_world_size": wrapper_commands,
        "wrapper_fds": tuple(wrapper_fds),
    }


def _sbatch_argv(plan: Mapping[str, Any], job: Mapping[str, Any]) -> list[str]:
    resources = job["requested_resource"]
    log_root = Path(plan["artifact_root"]) / "scheduler-logs"
    return [
        "--parsable",
        "--hold",
        "--partition=h100",
        "--qos=short",
        "--time=03:00:00",
        "--nodes=1",
        f"--gres=gpu:{resources['gpus']}",
        f"--cpus-per-task={resources['cpus_per_task']}",
        f"--mem={resources['memory_mb']}M",
        f"--chdir={Path(plan['artifact_root']) / 'job-cwd'}",
        f"--output={log_root / (job['case_id'] + '.out')}",
        f"--error={log_root / (job['case_id'] + '.err')}",
        "--open-mode=truncate",
        f"--job-name={_transaction_job_name(plan['transaction_id'], job['position'])}",
        _export_argument(job["exports"]),
        plan["wrapper_commands_by_world_size"][str(job["world_size"])],
    ]


def _transaction_payload(plan: Mapping[str, Any], submitted: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "transaction_id": plan["transaction_id"],
        "validation_id": plan["validation_id"],
        "commit_sha": plan["source"]["commit_sha"],
        "tree_sha": plan["source"]["tree_sha"],
        "source_manifest_sha256": plan["source"]["manifest_sha256"],
        "repository_binding": dict(plan["repository_binding"]),
        "environment_sha256_by_world_size": dict(plan["environment_sha256_by_world_size"]),
        "environment_transaction": dict(plan["environment_transaction"]),
        "nvidia_smi_sha256": plan["nvidia_smi_sha256"],
        "checkpoint_ceiling_bytes": plan["limits"]["checkpoint_ceiling_bytes"],
        "limits": dict(plan["limits"]),
        "capacity": dict(plan["capacity"]),
        "scheduler": _SCHEDULER,
        "scheduler_command_sha256": dict(plan["scheduler_command_sha256"]),
        "jobs_held_at_publication": True,
        "cases": [
            {
                "case_id": item["job"]["case_id"],
                "world_size": item["job"]["world_size"],
                "artifact_identity_sha256": item["job"]["artifact_identity_sha256"],
                "requested_resource": item["job"]["requested_resource"],
                "requested_resource_sha256": item["job"]["requested_resource_sha256"],
                "job_id": item["job_id"],
                "cluster": item["cluster"],
            }
            for item in submitted
        ],
    }


def _submit_transaction(
    plan: dict[str, Any],
    *,
    commands: Mapping[str, str],
    scheduler_fds: Sequence[int],
    wrapper_fds: Sequence[int],
    process_scoped_signals: bool = False,
) -> Mapping[str, Any]:
    scheduler_env = _scheduler_environment()
    sbatch_fds = (*scheduler_fds, *wrapper_fds)
    plan["transaction_id"] = hashlib.sha256(
        secrets.token_bytes(32)
        + plan["validation_id"].encode("ascii")
        + plan["source"]["commit_sha"].encode("ascii")
    ).hexdigest()[:32]
    plan["capacity"] = capacity_preflight(
        plan["artifact_root"].parent, plan["limits"]
    )
    _create_fresh_namespace(plan["artifact_root"])

    journal = _Journal(
        plan["artifact_root"] / "transaction-journal.jsonl",
        plan["transaction_id"],
        plan["limits"]["receipt_ceiling_bytes"],
    )
    journal.append(
        "transaction_started",
        {
            "validation_id": plan["validation_id"],
            "commit_sha": plan["source"]["commit_sha"],
            "case_count": 12,
        },
    )
    submitted: list[dict[str, Any]] = []
    phase = "sbatch_failed"
    untrusted_result_position: int | None = None
    inflight_position: int | None = None
    response_loss_job_name: str | None = None
    response_loss_discovery: list[dict[str, str | None]] = []
    response_loss_resolution: str | None = None
    held_plan: Mapping[str, Any] | None = None
    final_receipt: Mapping[str, Any] | None = None
    held_plan_published = False
    receipt_published = False
    held_plan_visibility = "absent"
    receipt_visibility = "absent"
    previous_handlers: dict[int, Any] = {}

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"scheduler transaction interrupted by signal {signum}")

    if process_scoped_signals:
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupted)
    try:
        for position, job in enumerate(plan["jobs"], start=1):
            journal.append(
                "sbatch_attempt",
                {"position": position, "case_id": job["case_id"]},
            )
            inflight_position = position
            completed = _run(
                [commands["sbatch"], *_sbatch_argv(plan, job)],
                env=scheduler_env,
                where=f"held H100 validation submission {position}/12",
                check=False,
                pass_fds=sbatch_fds,
            )
            if completed.returncode != 0:
                untrusted_result_position = position
                phase = "sbatch_nonzero_with_unknown_result"
                raise RuntimeError(
                    f"held H100 validation submission {position}/12 failed with status {completed.returncode}"
                )
            try:
                job_id, cluster = _parse_sbatch(completed.stdout.strip())
            except BaseException:
                untrusted_result_position = position
                phase = "sbatch_result_untrusted"
                raise
            if any(item["job_id"] == job_id for item in submitted):
                untrusted_result_position = position
                phase = "sbatch_result_untrusted"
                _fail("sbatch returned a duplicate job ID")
            item = {"job": job, "job_id": job_id, "cluster": cluster}
            submitted.append(item)
            inflight_position = None
            journal.append(
                "sbatch_accepted",
                {
                    "position": position,
                    "case_id": job["case_id"],
                    "job_id": job_id,
                    "cluster": cluster,
                },
            )

        # The jobs are still held. Recheck the same conservative budget after
        # scheduler round trips and bind this freshest stable snapshot into the
        # held plan; a shared-filesystem drop rolls the batch back before any
        # case can start.
        phase = "capacity_recheck_failed"
        repository_recheck = _query_repository_binding(
            exact_root=plan["exact_root"],
            git=Path(plan["jobs"][0]["exports"]["GIT"]),
            git_sha256=plan["jobs"][0]["exports"]["FORMAL_GIT_SHA256"],
            commit_sha=plan["source"]["commit_sha"],
        )
        if repository_recheck != plan["repository_binding"]:
            _fail("GitHub repository binding changed before H100 gate release")
        isolation_recheck = _require_work_artifact_filesystem_isolation(
            exact_root=plan["exact_root"],
            work_root=Path(plan["filesystem_isolation"]["work_root"]),
            artifact_directory=plan["artifact_root"],
        )
        if (
            isolation_recheck["work_device"]
            != plan["filesystem_isolation"]["work_device"]
            or isolation_recheck["artifact_device"]
            != plan["filesystem_isolation"]["artifact_device"]
        ):
            _fail("H100 work/artifact filesystem identity changed before release")
        plan["filesystem_isolation"] = isolation_recheck
        initial_capacity_model = {
            key: value
            for key, value in plan["capacity"].items()
            if key not in {"available_blocks", "available_bytes"}
        }
        rechecked_capacity = capacity_preflight(
            plan["artifact_root"], plan["limits"]
        )
        if {
            key: value
            for key, value in rechecked_capacity.items()
            if key not in {"available_blocks", "available_bytes"}
        } != initial_capacity_model:
            _fail("H100 gate physical allocation model changed while jobs were held")
        plan["capacity"] = rechecked_capacity
        journal.append("capacity_rechecked", dict(plan["capacity"]))
        phase = "held_plan_publication_failed"
        held_plan = _make_envelope(
            "h100_gate_held_plan", _transaction_payload(plan, submitted)
        )
        previous_mask = None
        if process_scoped_signals:
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK, tuple(previous_handlers)
            )
        try:
            _publish_no_replace(
                plan["artifact_root"] / "held-plan.json",
                held_plan,
                max_bytes=plan["limits"]["receipt_ceiling_bytes"],
            )
            held_plan_published = True
            held_plan_visibility = "durable"
        finally:
            if previous_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        journal.append("held_plan_published", {"sha256": held_plan["sha256"]})

        phase = "release_failed"
        released_states: list[dict[str, Any]] = []
        for item in submitted:
            job_id, cluster = item["job_id"], item["cluster"]
            journal.append("release_attempt", {"job_id": job_id, "cluster": cluster})
            _run(
                [commands["scontrol"], *_cluster_argv(cluster), "release", job_id],
                env=scheduler_env,
                where=f"release held H100 gate job {job_id}",
                pass_fds=scheduler_fds,
            )
            state, reason, source = _scheduler_state_bounded_retry(
                commands,
                scheduler_env,
                scheduler_fds,
                job_id=job_id,
                cluster=cluster,
                acceptable=_released_state_is_acceptable,
            )
            released = {
                "job_id": job_id,
                "cluster": cluster,
                "state": state,
                "reason": reason,
                "source": source,
            }
            released_states.append(released)
            journal.append("release_reconciled", released)

        phase = "receipt_publication_failed"
        journal.append("all_releases_reconciled", {"jobs": released_states})
        journal_sha256, journal_events = journal.snapshot()
        receipt_payload = {
            **_transaction_payload(plan, submitted),
            "all_jobs_released": True,
            "held_plan_sha256": held_plan["sha256"],
            "transaction_journal_sha256": journal_sha256,
            "transaction_journal_events": journal_events,
        }
        final_receipt = _make_envelope(
            "h100_gate_submission_receipt", receipt_payload
        )
        if process_scoped_signals:
            for signum in previous_handlers:
                signal.signal(signum, signal.SIG_IGN)
        _publish_no_replace(
            plan["artifact_root"] / "submission-receipt.json",
            final_receipt,
            max_bytes=plan["limits"]["receipt_ceiling_bytes"],
        )
        receipt_published = True
        receipt_visibility = "durable"
        return final_receipt
    except BaseException as error:
        if isinstance(error, _PublicationInterrupted):
            interruption = error.interruption
            if error.path.name == "held-plan.json":
                held_plan_published = True
                held_plan_visibility = "durable"
                _safe_journal(
                    journal,
                    "held_plan_publication_interrupted",
                    {"sha256": held_plan["sha256"] if held_plan else None},
                )
                error = interruption
            elif error.path.name == "submission-receipt.json":
                receipt_published = True
                receipt_visibility = "durable"
                if process_scoped_signals:
                    for signum, handler in previous_handlers.items():
                        signal.signal(signum, handler)
                raise interruption
        elif isinstance(error, _PublicationUncertain):
            if error.path.name == "held-plan.json":
                held_plan_visibility = "uncertain"
            elif error.path.name == "submission-receipt.json":
                receipt_visibility = "uncertain"
        if inflight_position is not None:
            if untrusted_result_position is None:
                untrusted_result_position = inflight_position
                phase = "sbatch_interrupted_with_unknown_result"
        if process_scoped_signals:
            for signum in previous_handlers:
                signal.signal(signum, signal.SIG_IGN)
        if (
            isinstance(error, _PublicationUncertain)
            and error.path.name == "submission-receipt.json"
        ):
            raise RuntimeError(
                "released H100 gate receipt visibility is uncertain; jobs were "
                "left released to avoid contradicting a visible receipt"
            ) from error
        if untrusted_result_position is not None:
            response_loss_job_name = _transaction_job_name(
                plan["transaction_id"], untrusted_result_position
            )
            try:
                response_loss_discovery = _discover_transaction_jobs(
                    commands,
                    scheduler_env,
                    scheduler_fds,
                    job_name=response_loss_job_name,
                )
            except Exception:
                response_loss_discovery = []
                response_loss_resolution = "discovery_failed"
            if response_loss_resolution is None:
                if len(response_loss_discovery) == 1 and all(
                    item["job_id"] != response_loss_discovery[0]["job_id"]
                    for item in submitted
                ):
                    discovered = response_loss_discovery[0]
                    submitted.append(
                        {
                            "job": plan["jobs"][untrusted_result_position - 1],
                            "job_id": discovered["job_id"],
                            "cluster": discovered["cluster"],
                        }
                    )
                    response_loss_resolution = "unique_job_absorbed"
                elif not response_loss_discovery:
                    response_loss_resolution = "no_job_visible"
                elif len(response_loss_discovery) > 1:
                    response_loss_resolution = "ambiguous_jobs_visible"
                else:
                    response_loss_resolution = "job_id_conflicts_with_accepted"
            _safe_journal(
                journal,
                "sbatch_response_loss_reconciled",
                {
                    "position": untrusted_result_position,
                    "job_name": response_loss_job_name,
                    "resolution": response_loss_resolution,
                    "matches": response_loss_discovery,
                },
            )
        try:
            cancelled, remaining = _rollback(
                commands,
                scheduler_env,
                scheduler_fds,
                submitted,
                journal,
            )
        except BaseException as rollback_error:
            cancelled = []
            remaining = [item["job_id"] for item in submitted]
            _safe_journal(
                journal,
                "rollback_controller_failed",
                {"error_type": type(rollback_error).__name__},
            )
        _safe_journal(
            journal,
            "rollback_finished",
            {
                "reason": phase,
                "cancelled_job_ids": cancelled,
                "remaining_job_ids": remaining,
                "untrusted_sbatch_result_position": untrusted_result_position,
                "response_loss_job_name": response_loss_job_name,
                "response_loss_resolution": response_loss_resolution,
                "response_loss_discovery": response_loss_discovery,
            },
        )
        try:
            journal_sha256, journal_events = journal.snapshot()
        except Exception:
            journal_sha256, journal_events = None, None
        recovery = _make_envelope(
            "h100_gate_rollback_recovery",
            {
                "transaction_id": plan["transaction_id"],
                "validation_id": plan["validation_id"],
                "reason": phase,
                "accepted_jobs": [
                    {"job_id": item["job_id"], "cluster": item["cluster"]}
                    for item in submitted
                ],
                "cancelled_job_ids": cancelled,
                "remaining_job_ids": remaining,
                "untrusted_sbatch_result_position": untrusted_result_position,
                "response_loss_job_name": response_loss_job_name,
                "response_loss_resolution": response_loss_resolution,
                "response_loss_discovery": response_loss_discovery,
                "held_plan_published": held_plan_published,
                "submission_receipt_published": receipt_published,
                "held_plan_visibility": held_plan_visibility,
                "submission_receipt_visibility": receipt_visibility,
                "transaction_journal_sha256": journal_sha256,
                "transaction_journal_events": journal_events,
            },
        )
        try:
            _publish_no_replace(
                plan["artifact_root"] / "rollback-recovery.json",
                recovery,
                max_bytes=plan["limits"]["receipt_ceiling_bytes"],
            )
        except Exception as recovery_error:
            raise RuntimeError(
                "H100 gate transaction failed and recovery publication failed; "
                f"remaining={','.join(remaining)} unknown={untrusted_result_position}"
            ) from recovery_error
        if remaining or (
            untrusted_result_position is not None
            and response_loss_resolution != "unique_job_absorbed"
        ):
            raise RuntimeError(
                "H100 gate transaction failed with unresolved scheduler state; "
                f"recovery={plan['artifact_root'] / 'rollback-recovery.json'}"
            ) from error
        raise RuntimeError(
            "H100 gate transaction failed; rollback reconciled and recovery was published"
        ) from error


def submit(
    overlay: Path,
    *,
    exact_root: Path,
    process_scoped_signals: bool = False,
) -> Mapping[str, Any]:
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        _fail("H100 gate submitter requires Python -I -B")
    if (
        os.environ.get("PYTHONNOUSERSITE") != "1"
        or os.environ.get("PYTHONPATH") != os.fspath(exact_root / "src")
    ):
        _fail("H100 gate submitter requires a closed exact-source import environment")
    plan = _validate_overlay(_load_strict_json(overlay), exact_root=exact_root)
    wrapper_fds = plan["wrapper_fds"]
    try:
        commands, scheduler_fds = _open_scheduler_commands(
            plan["scheduler_commands"]
        )
    except BaseException:
        for wrapper_fd in wrapper_fds:
            os.close(wrapper_fd)
        raise
    try:
        return _submit_transaction(
            plan,
            commands=commands,
            scheduler_fds=scheduler_fds,
            wrapper_fds=wrapper_fds,
            process_scoped_signals=process_scoped_signals,
        )
    finally:
        for scheduler_fd in scheduler_fds:
            os.close(scheduler_fd)
        for wrapper_fd in wrapper_fds:
            os.close(wrapper_fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--exact-root", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.overlay.is_absolute() or not args.exact_root.is_absolute():
        parser.error("overlay and exact root must be absolute paths")
    receipt = submit(
        args.overlay,
        exact_root=args.exact_root,
        process_scoped_signals=True,
    )
    try:
        sys.stdout.buffer.write(_canonical(receipt) + b"\n")
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"H100 gate submission failed: {error}", file=sys.stderr)
        raise SystemExit(2)
