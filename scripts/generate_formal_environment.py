#!/usr/bin/env python3
"""Publish canonical one/two-GPU environment commitments from exact T."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
from typing import Any, Mapping


_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_SIBLING = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_GIT_MAX_BYTES = 128 << 20
_GIT_OUTPUT_MAX_BYTES = 1 << 20
LOCAL_GIT_CONFIG_OVERRIDES = (
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.filemode=true",
)
_FAILURE_INJECTION_ENABLE = "TABICL_ENABLE_TEST_FAILURE_INJECTION"
_FAILURE_INJECTION_POINT = "TABICL_TEST_FAIL_ENVIRONMENT_PUBLICATION_AT"


def _absolute_normalized(path: Path, *, where: str) -> Path:
    if (
        not path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or os.path.abspath(os.fspath(path)) != os.fspath(path)
    ):
        raise ValueError(f"{where} must be a normalized absolute path")
    return path


def _open_trusted_git(path: Path, *, expected_sha256: str) -> int:
    path = _absolute_normalized(path, where="Git executable")
    if _HEX64.fullmatch(expected_sha256) is None:
        raise ValueError("expected Git SHA-256 is malformed")
    parent_fd = _open_directory_nofollow(path.parent)
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(path.name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError("Git must be a no-follow regular executable") from error
    finally:
        os.close(parent_fd)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_mode & 0o111 == 0
            or before.st_size > _GIT_MAX_BYTES
        ):
            raise ValueError("Git is not a bounded regular executable")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                raise ValueError("Git was truncated while it was hashed")
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        if (
            (before.st_dev, before.st_ino, before.st_mode, before.st_size,
             before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_mode, after.st_size,
                after.st_mtime_ns, after.st_ctime_ns)
        ):
            raise ValueError("Git changed while it was hashed")
        if digest.hexdigest() != expected_sha256:
            raise ValueError("Git executable differs from its external commitment")
        os.lseek(fd, 0, os.SEEK_SET)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _assert_git_path_identity(git_fd: int, path: Path) -> None:
    try:
        path_metadata = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ValueError("Git executable path changed after attestation") from error
    fd_metadata = os.fstat(git_fd)
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(
        getattr(path_metadata, field) != getattr(fd_metadata, field)
        for field in fields
    ):
        raise ValueError("Git executable path changed after attestation")


def _git(git_fd: int, root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = {
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
    }
    return subprocess.run(
        [
            f"/proc/self/fd/{git_fd}",
            *LOCAL_GIT_CONFIG_OVERRIDES,
            "-C",
            os.fspath(root),
            *arguments,
        ],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=30,
        pass_fds=(git_fd,),
    )


def _attest_checkout(
    *,
    root: Path,
    git_fd: int,
    git_path: Path,
    expected_commit: str,
    expected_tree: str,
) -> None:
    checks = (
        (("rev-parse", "HEAD^{commit}"), expected_commit, 0),
        (("rev-parse", "HEAD^{tree}"), expected_tree, 0),
        (("status", "--porcelain=v1", "--untracked-files=all"), "", 0),
    )
    for arguments, expected_stdout, expected_returncode in checks:
        _assert_git_path_identity(git_fd, git_path)
        completed = _git(git_fd, root, *arguments)
        if (
            completed.returncode != expected_returncode
            or completed.stderr
            or len(completed.stdout.encode("utf-8")) > _GIT_OUTPUT_MAX_BYTES
            or completed.stdout.strip() != expected_stdout
        ):
            raise ValueError("exact-T Git attestation failed")
    _assert_git_path_identity(git_fd, git_path)
    symbolic = _git(git_fd, root, "symbolic-ref", "-q", "HEAD")
    if symbolic.returncode != 1 or symbolic.stdout or symbolic.stderr:
        raise ValueError("exact-T checkout must be detached")
    provenance_path = root / "src/tabicl/train/_provenance.py"
    provenance_raw = _read_no_follow(provenance_path, max_bytes=_GIT_MAX_BYTES)
    committed = _git(
        git_fd,
        root,
        "show",
        f"{expected_commit}:src/tabicl/train/_provenance.py",
    )
    if (
        committed.returncode != 0
        or committed.stderr
        or committed.stdout.encode("utf-8") != provenance_raw
    ):
        raise ValueError("formal provenance helper differs from committed source")
    _assert_git_path_identity(git_fd, git_path)


def _open_directory_nofollow(path: Path) -> int:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ValueError("publication parent must be a normalized absolute path")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(os.path.sep, flags)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except OSError as error:
        os.close(fd)
        raise ValueError("publication parent contains a symlink") from error


def _publish_no_replace(path: Path, raw: bytes, *, max_bytes: int) -> None:
    if max_bytes < 1 or len(raw) > max_bytes:
        raise ValueError("environment artifact exceeds its byte ceiling")
    if path.name in {"", ".", ".."}:
        raise ValueError("environment artifact must name a file")
    parent_fd = _open_directory_nofollow(path.parent)
    temporary = f".{path.name}.{secrets.token_hex(16)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    fd = -1
    linked = False
    try:
        fd = os.open(temporary, flags, 0o400, dir_fd=parent_fd)
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written < 1:
                raise OSError("short write while publishing environment artifact")
            view = view[written:]
        os.fsync(fd)
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != len(raw):
            raise ValueError("environment publication temporary is malformed")
        os.link(
            temporary,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        linked = True
        os.unlink(temporary, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        if not linked:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _read_no_follow(path: Path, *, max_bytes: int) -> bytes:
    if max_bytes < 1:
        raise ValueError("environment artifact byte ceiling must be positive")
    parent_fd = _open_directory_nofollow(path.parent)
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        try:
            fd = os.open(path.name, flags, dir_fd=parent_fd)
        except OSError as error:
            raise ValueError("environment artifact is missing or is a symlink") from error
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
                raise ValueError("environment artifact is not a bounded regular file")
            chunks: list[bytes] = []
            remaining = max_bytes + 1
            while remaining:
                chunk = os.read(fd, min(1 << 20, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            if len(raw) > max_bytes:
                raise ValueError("environment artifact exceeds its byte ceiling")
            after = os.fstat(fd)
            fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(before, field) != getattr(after, field) for field in fields):
                raise ValueError("environment artifact changed while it was read")
            return raw
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def _publish_or_verify(path: Path, raw: bytes, *, max_bytes: int) -> None:
    try:
        _publish_no_replace(path, raw, max_bytes=max_bytes)
    except FileExistsError:
        if _read_no_follow(path, max_bytes=max_bytes) != raw:
            raise ValueError("existing environment transaction artifact differs")


def _read_if_exists(path: Path, *, max_bytes: int) -> bytes | None:
    try:
        return _read_no_follow(path, max_bytes=max_bytes)
    except ValueError:
        parent_fd = _open_directory_nofollow(path.parent)
        try:
            try:
                os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return None
        finally:
            os.close(parent_fd)
        raise


def _inject_failure(point: str) -> None:
    requested = os.environ.get(_FAILURE_INJECTION_POINT)
    enabled = os.environ.get(_FAILURE_INJECTION_ENABLE)
    if requested is None and enabled is None:
        return
    if enabled != "1" or requested not in {
        "after-staging",
        "after-one-gpu",
        "after-two-gpu",
        "after-inventory",
        "after-completion",
    }:
        raise ValueError("invalid formal environment failure-injection request")
    if requested == point:
        raise OSError(f"injected formal environment publication failure: {point}")


def _transaction_completion(
    *,
    source_commit_sha: str,
    source_tree_sha: str,
    artifacts: tuple[tuple[str, Path, bytes, int], ...],
) -> tuple[dict[str, Any], str]:
    descriptor = {
        "schema_version": 1,
        "source_commit_sha": source_commit_sha,
        "source_tree_sha": source_tree_sha,
        "outputs": [
            {
                "role": role,
                "name": path.name,
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            for role, path, raw, _ceiling in artifacts
        ],
    }
    transaction_sha256 = hashlib.sha256(_canonical(descriptor)).hexdigest()
    return (
        {
            "schema_version": 1,
            "kind": "formal_environment_generation_completion",
            "transaction_sha256": transaction_sha256,
            "source_commit_sha": source_commit_sha,
            "source_tree_sha": source_tree_sha,
            "outputs": descriptor["outputs"],
        },
        transaction_sha256,
    )


def _cleanup_transaction_stages(
    stage_paths: tuple[tuple[str, Path, bytes, int], ...]
) -> None:
    if not stage_paths:
        return
    parent_fd = _open_directory_nofollow(stage_paths[0][1].parent)
    try:
        for _role, stage, raw, ceiling in stage_paths:
            existing = _read_if_exists(stage, max_bytes=ceiling)
            if existing is None:
                continue
            if existing != raw:
                raise ValueError("environment transaction staging artifact differs")
            try:
                os.unlink(stage.name, dir_fd=parent_fd)
            except FileNotFoundError:
                continue
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _publish_transaction(
    *,
    source_commit_sha: str,
    source_tree_sha: str,
    artifacts: tuple[tuple[str, Path, bytes, int], ...],
    completion_output: Path,
    completion_max_bytes: int,
) -> dict[str, Any]:
    completion, transaction_sha256 = _transaction_completion(
        source_commit_sha=source_commit_sha,
        source_tree_sha=source_tree_sha,
        artifacts=artifacts,
    )
    completion_raw = _canonical(completion)
    if len(completion_raw) > completion_max_bytes:
        raise ValueError("environment completion marker exceeds its byte ceiling")
    stage_paths = tuple(
        (
            role,
            completion_output.parent
            / f".formal-environment-{transaction_sha256[:32]}-{role}.stage",
            raw,
            ceiling,
        )
        for role, _path, raw, ceiling in artifacts
    )
    existing_completion = _read_if_exists(
        completion_output, max_bytes=completion_max_bytes
    )
    if existing_completion is not None:
        if existing_completion != completion_raw:
            raise ValueError("existing environment completion marker differs")
        for _role, final, raw, ceiling in artifacts:
            if _read_no_follow(final, max_bytes=ceiling) != raw:
                raise ValueError("completed environment transaction output differs")
        _cleanup_transaction_stages(stage_paths)
        return completion
    # Deterministic staging names make an interrupted transaction retryable.
    # Final names are authoritative only after the marker is published last.
    for _role, stage, raw, ceiling in stage_paths:
        _publish_or_verify(stage, raw, max_bytes=ceiling)
    _inject_failure("after-staging")
    for (role, final, raw, ceiling), (_stage_role, stage, _stage_raw, _stage_ceiling) in zip(
        artifacts, stage_paths
    ):
        if role != _stage_role:
            raise ValueError("environment transaction role mismatch")
        parent_fd = _open_directory_nofollow(final.parent)
        try:
            try:
                os.link(
                    stage.name,
                    final.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                os.fsync(parent_fd)
            except FileExistsError:
                if _read_no_follow(final, max_bytes=ceiling) != raw:
                    raise ValueError("existing environment output differs")
        finally:
            os.close(parent_fd)
        if _read_no_follow(final, max_bytes=ceiling) != raw:
            raise ValueError("published environment output differs")
        _inject_failure(f"after-{role.replace('_', '-')}")
    _publish_or_verify(
        completion_output, completion_raw, max_bytes=completion_max_bytes
    )
    _inject_failure("after-completion")

    # Cleanup is intentionally after the durable marker. A failure before this
    # point leaves deterministic staging names that the next invocation can
    # verify and reuse; a cleanup failure cannot invalidate the committed set.
    _cleanup_transaction_stages(stage_paths)
    return completion


def _import_exact_provenance(root: Path) -> Any:
    preloaded = sorted(
        name for name in sys.modules if name == "tabicl" or name.startswith("tabicl.")
    )
    if preloaded:
        raise ValueError("tabicl was imported before exact-T isolation")
    expected = root / "src" / "tabicl" / "train" / "_provenance.py"
    spec = importlib.util.spec_from_file_location(
        "_formal_environment_exact_provenance", expected
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load exact formal provenance helper")
    provenance = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = provenance
    spec.loader.exec_module(provenance)
    raw_file = getattr(provenance, "__file__", None)
    if not isinstance(raw_file, str) or Path(raw_file) != expected:
        raise ValueError("formal environment generator imported tabicl outside exact T")
    actual = Path(raw_file).resolve(strict=True)
    if actual != expected.resolve(strict=True) or not actual.is_file():
        raise ValueError("formal environment generator imported tabicl outside exact T")
    return provenance


def _build_artifacts(
    *,
    captured_environment: Mapping[str, Any],
    fingerprint_preimage: Mapping[str, Any],
    source_commit_sha: str,
    source_tree_sha: str,
    git_sha256: str,
    make_manifest: Any,
    canonical_sha256: Any,
    validate_environment_payload: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    captured_payload = captured_environment["payload"]
    inventory_sha256 = captured_payload["installed_distributions_sha256"]
    if canonical_sha256(fingerprint_preimage) != inventory_sha256:
        raise ValueError("private inventory preimage does not match environment digest")
    environments: dict[str, dict[str, Any]] = {}
    for world_size in (1, 2):
        payload = copy.deepcopy(captured_payload)
        payload["visible_cuda_device_count"] = world_size
        validate_environment_payload(payload, require_formal_runtime=True)
        environments[str(world_size)] = make_manifest("environment", payload)
    one_gpu = environments["1"]
    two_gpu = environments["2"]
    if one_gpu["sha256"] == two_gpu["sha256"]:
        raise ValueError("one/two-GPU environment commitments must differ")
    inventory = make_manifest(
        "formal_environment_inventory",
        {
            "source_commit_sha": source_commit_sha,
            "source_tree_sha": source_tree_sha,
            "git_sha256": git_sha256,
            "capture_visible_cuda_device_count": captured_payload[
                "visible_cuda_device_count"
            ],
            "installed_distributions_sha256": inventory_sha256,
            "environment_sha256_by_world_size": {
                key: environments[key]["sha256"] for key in ("1", "2")
            },
            "fingerprint_preimage": copy.deepcopy(fingerprint_preimage),
        },
    )
    return one_gpu, two_gpu, inventory


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-root", required=True, type=Path)
    parser.add_argument("--git", required=True, type=Path)
    parser.add_argument("--expected-git-sha256", required=True)
    parser.add_argument("--expected-commit-sha", required=True)
    parser.add_argument("--expected-tree-sha", required=True)
    parser.add_argument("--one-gpu-output", required=True, type=Path)
    parser.add_argument("--two-gpu-output", required=True, type=Path)
    parser.add_argument("--inventory-output", required=True, type=Path)
    parser.add_argument("--completion-output", required=True, type=Path)
    parser.add_argument("--manifest-max-bytes", required=True, type=int)
    parser.add_argument("--inventory-max-bytes", required=True, type=int)
    parser.add_argument("--completion-max-bytes", required=True, type=int)
    args = parser.parse_args(argv)
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise ValueError("formal environment generation requires Python -I -B")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise ValueError("formal environment generation requires PYTHONNOUSERSITE=1")
    root = Path(__file__).resolve(strict=True).parent.parent
    if not args.exact_root.is_absolute() or args.exact_root.resolve(strict=True) != root:
        raise ValueError("formal environment generator is not running from exact T")
    if _HEX40.fullmatch(args.expected_commit_sha) is None:
        raise ValueError("expected commit SHA is malformed")
    if _HEX40.fullmatch(args.expected_tree_sha) is None:
        raise ValueError("expected tree SHA is malformed")
    git_fd = _open_trusted_git(
        args.git, expected_sha256=args.expected_git_sha256
    )
    try:
        _attest_checkout(
            root=root,
            git_fd=git_fd,
            git_path=args.git,
            expected_commit=args.expected_commit_sha,
            expected_tree=args.expected_tree_sha,
        )
    finally:
        os.close(git_fd)
    provenance = _import_exact_provenance(root)
    captured, preimage = provenance.runtime_environment_snapshot(
        require_formal_runtime=True
    )
    one_gpu, two_gpu, inventory = _build_artifacts(
        captured_environment=captured,
        fingerprint_preimage=preimage,
        source_commit_sha=args.expected_commit_sha,
        source_tree_sha=args.expected_tree_sha,
        git_sha256=args.expected_git_sha256,
        make_manifest=provenance.make_manifest,
        canonical_sha256=provenance.canonical_sha256,
        validate_environment_payload=provenance.validate_environment_payload,
    )
    output_values = (
        ("one_gpu", args.one_gpu_output, one_gpu, args.manifest_max_bytes),
        ("two_gpu", args.two_gpu_output, two_gpu, args.manifest_max_bytes),
        ("inventory", args.inventory_output, inventory, args.inventory_max_bytes),
    )
    all_paths = tuple(path for _role, path, _value, _ceiling in output_values) + (
        args.completion_output,
    )
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("formal environment output paths must be distinct")
    for path in all_paths:
        _absolute_normalized(path, where="formal environment output")
        if _SAFE_SIBLING.fullmatch(path.name) is None:
            raise ValueError("formal environment output basename is not a safe sibling")
    parents = {path.parent for path in all_paths}
    if len(parents) != 1:
        raise ValueError("formal environment transaction outputs must share one parent")
    output_parent = next(iter(parents))
    parent_fd = _open_directory_nofollow(output_parent)
    os.close(parent_fd)
    physical_parent = output_parent.resolve(strict=True)
    physical_root = root.resolve(strict=True)
    for path in all_paths:
        try:
            (physical_parent / path.name).relative_to(physical_root)
        except ValueError:
            pass
        else:
            raise ValueError("formal environment outputs must be outside exact T")
    artifacts = tuple(
        (role, path, _canonical(value), ceiling)
        for role, path, value, ceiling in output_values
    )
    completion = _publish_transaction(
        source_commit_sha=args.expected_commit_sha,
        source_tree_sha=args.expected_tree_sha,
        artifacts=artifacts,
        completion_output=args.completion_output,
        completion_max_bytes=args.completion_max_bytes,
    )
    sys.stdout.buffer.write(
        _canonical(
            {
                "schema_version": 1,
                "kind": "formal_environment_generation",
                "one_gpu_environment_sha256": one_gpu["sha256"],
                "two_gpu_environment_sha256": two_gpu["sha256"],
                "inventory_sha256": inventory["sha256"],
                "transaction_sha256": completion["transaction_sha256"],
                "completion_output_sha256": hashlib.sha256(
                    _canonical(completion)
                ).hexdigest(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"formal environment generation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
