#!/usr/bin/env python3
"""Independently verify one completed formal-environment publication set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Mapping


_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_SIBLING = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ROLES = ("one_gpu", "two_gpu", "inventory")


def _canonical(value: Any, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _git_oid(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        raise ValueError(f"{where} must be a lowercase Git object ID")
    return value


def _positive_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _exact(value: Any, keys: set[str], *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    if set(value) != keys:
        raise ValueError(
            f"{where} keys mismatch; missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )
    return value


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_json(raw: bytes, *, where: str) -> Any:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{where} is not strict UTF-8 JSON") from error
    if raw != _canonical(value, newline=True):
        raise ValueError(f"{where} is not canonical JSON")
    return value


def _normalized_absolute(path: Path, *, where: str) -> Path:
    raw = os.fspath(path)
    if (
        not path.is_absolute()
        or any(character in raw for character in ("\x00", "\n", "\r"))
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or os.path.abspath(raw) != raw
    ):
        raise ValueError(f"{where} must be a normalized absolute path")
    return path


def _open_directory_nofollow(path: Path) -> int:
    path = _normalized_absolute(path, where="environment transaction directory")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ValueError("no-follow directory traversal is unavailable")
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    fd = os.open(os.path.sep, flags)
    try:
        for component in path.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _safe_sibling(value: Any, *, completion_name: str) -> str:
    if (
        not isinstance(value, str)
        or _SAFE_SIBLING.fullmatch(value) is None
        or PurePosixPath(value).name != value
        or value == completion_name
    ):
        raise ValueError("environment transaction output name is not a safe sibling")
    return value


def _read_sibling(parent_fd: int, name: str, *, max_bytes: int, where: str) -> bytes:
    _positive_int(max_bytes, where=f"{where} byte ceiling")
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError(f"{where} is missing or is a symlink") from error
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise ValueError(f"{where} is not a bounded regular file")
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
            raise ValueError(f"{where} exceeds its byte ceiling")
        after = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in fields):
            raise ValueError(f"{where} changed while it was read")
        return raw
    finally:
        os.close(fd)


def _manifest(raw: bytes, *, kind: str, expected_sha256: str, where: str) -> Mapping[str, Any]:
    value = _strict_json(raw, where=where)
    envelope = _exact(
        value,
        {"schema_version", "kind", "payload", "sha256"},
        where=where,
    )
    if envelope["schema_version"] != 1 or envelope["kind"] != kind:
        raise ValueError(f"{where} schema or kind mismatch")
    body = {key: envelope[key] for key in ("schema_version", "kind", "payload")}
    actual = _sha256_value(body)
    if envelope["sha256"] != actual:
        raise ValueError(f"{where} self-hash mismatch")
    if actual != _digest(expected_sha256, where=f"expected {where}"):
        raise ValueError(f"{where} does not match its external expected digest")
    if not isinstance(envelope["payload"], Mapping):
        raise ValueError(f"{where} payload must be an object")
    return envelope


def _environment(
    raw: bytes,
    *,
    expected_sha256: str,
    expected_count: int,
    where: str,
) -> Mapping[str, Any]:
    envelope = _manifest(
        raw,
        kind="environment",
        expected_sha256=expected_sha256,
        where=where,
    )
    payload = envelope["payload"]
    count = payload.get("visible_cuda_device_count")
    if isinstance(count, bool) or count != expected_count:
        raise ValueError(f"{where} visible CUDA device count mismatch")
    if payload.get("environment_fingerprint_schema_version") != 2:
        raise ValueError(f"{where} fingerprint schema mismatch")
    _digest(
        payload.get("installed_distributions_sha256"),
        where=f"{where} installed-distribution digest",
    )
    return envelope


def _verify(
    *,
    completion_path: Path,
    expected_completion_sha256: str,
    expected_transaction_sha256: str,
    expected_one_gpu_environment_sha256: str,
    expected_two_gpu_environment_sha256: str,
    expected_inventory_sha256: str,
    expected_source_commit_sha: str,
    expected_source_tree_sha: str,
    expected_git_sha256: str,
    completion_max_bytes: int,
    manifest_max_bytes: int,
    inventory_max_bytes: int,
) -> dict[str, Any]:
    completion_path = _normalized_absolute(
        completion_path, where="environment completion marker"
    )
    completion_name = _safe_sibling(
        completion_path.name, completion_name="__not_the_completion__"
    )
    parent_fd = _open_directory_nofollow(completion_path.parent)
    try:
        completion_raw = _read_sibling(
            parent_fd,
            completion_name,
            max_bytes=completion_max_bytes,
            where="environment completion marker",
        )
        completion_raw_sha256 = hashlib.sha256(completion_raw).hexdigest()
        if completion_raw_sha256 != _digest(
            expected_completion_sha256, where="expected completion marker"
        ):
            raise ValueError("environment completion marker raw SHA-256 mismatch")
        completion = _exact(
            _strict_json(completion_raw, where="environment completion marker"),
            {
                "schema_version",
                "kind",
                "transaction_sha256",
                "source_commit_sha",
                "source_tree_sha",
                "outputs",
            },
            where="environment completion marker",
        )
        if (
            completion["schema_version"] != 1
            or completion["kind"] != "formal_environment_generation_completion"
        ):
            raise ValueError("environment completion marker schema or kind mismatch")
        source_commit = _git_oid(
            completion["source_commit_sha"], where="completion source commit"
        )
        source_tree = _git_oid(
            completion["source_tree_sha"], where="completion source tree"
        )
        if source_commit != _git_oid(
            expected_source_commit_sha, where="expected source commit"
        ) or source_tree != _git_oid(
            expected_source_tree_sha, where="expected source tree"
        ):
            raise ValueError("environment completion source commit/tree mismatch")
        outputs = completion["outputs"]
        if not isinstance(outputs, list) or len(outputs) != len(_ROLES):
            raise ValueError("environment completion output set is invalid")
        normalized_outputs: list[dict[str, Any]] = []
        raw_by_role: dict[str, bytes] = {}
        seen_names: set[str] = set()
        for expected_role, raw_entry in zip(_ROLES, outputs):
            entry = _exact(
                raw_entry,
                {"role", "name", "size", "sha256"},
                where="environment completion output",
            )
            if entry["role"] != expected_role:
                raise ValueError("environment completion output order is invalid")
            name = _safe_sibling(entry["name"], completion_name=completion_name)
            if name in seen_names:
                raise ValueError("environment completion output name is duplicated")
            seen_names.add(name)
            size = _positive_int(entry["size"], where=f"{expected_role} output size")
            ceiling = inventory_max_bytes if expected_role == "inventory" else manifest_max_bytes
            if size > ceiling:
                raise ValueError(f"{expected_role} output exceeds its byte ceiling")
            expected_raw_sha = _digest(
                entry["sha256"], where=f"{expected_role} output raw digest"
            )
            raw = _read_sibling(
                parent_fd,
                name,
                max_bytes=ceiling,
                where=f"{expected_role} environment output",
            )
            if len(raw) != size or hashlib.sha256(raw).hexdigest() != expected_raw_sha:
                raise ValueError(f"{expected_role} environment output differs from marker")
            normalized_outputs.append(dict(entry))
            raw_by_role[expected_role] = raw
        descriptor = {
            "schema_version": 1,
            "source_commit_sha": source_commit,
            "source_tree_sha": source_tree,
            "outputs": normalized_outputs,
        }
        transaction_sha256 = hashlib.sha256(
            _canonical(descriptor, newline=True)
        ).hexdigest()
        if (
            completion["transaction_sha256"] != transaction_sha256
            or transaction_sha256
            != _digest(expected_transaction_sha256, where="expected transaction")
        ):
            raise ValueError("environment transaction descriptor SHA-256 mismatch")

        one = _environment(
            raw_by_role["one_gpu"],
            expected_sha256=expected_one_gpu_environment_sha256,
            expected_count=1,
            where="one-GPU environment",
        )
        two = _environment(
            raw_by_role["two_gpu"],
            expected_sha256=expected_two_gpu_environment_sha256,
            expected_count=2,
            where="two-GPU environment",
        )
        one_payload = dict(one["payload"])
        two_payload = dict(two["payload"])
        one_payload.pop("visible_cuda_device_count")
        two_payload.pop("visible_cuda_device_count")
        if one_payload != two_payload:
            raise ValueError("one/two-GPU environment payloads differ beyond CUDA count")

        inventory = _manifest(
            raw_by_role["inventory"],
            kind="formal_environment_inventory",
            expected_sha256=expected_inventory_sha256,
            where="formal environment inventory",
        )
        inventory_payload = _exact(
            inventory["payload"],
            {
                "source_commit_sha",
                "source_tree_sha",
                "git_sha256",
                "capture_visible_cuda_device_count",
                "installed_distributions_sha256",
                "environment_sha256_by_world_size",
                "fingerprint_preimage",
            },
            where="formal environment inventory payload",
        )
        capture_count = inventory_payload["capture_visible_cuda_device_count"]
        if isinstance(capture_count, bool) or not isinstance(capture_count, int) or capture_count < 0:
            raise ValueError("inventory capture visible CUDA count is invalid")
        installed_sha256 = _digest(
            inventory_payload["installed_distributions_sha256"],
            where="inventory installed-distribution digest",
        )
        fingerprint = _exact(
            inventory_payload["fingerprint_preimage"],
            {
                "visible_distribution_multiset",
                "effective_formal_runtime_distributions",
            },
            where="inventory fingerprint preimage",
        )
        if not isinstance(fingerprint["visible_distribution_multiset"], list) or not isinstance(
            fingerprint["effective_formal_runtime_distributions"], list
        ):
            raise ValueError("inventory fingerprint preimage lists are invalid")
        environment_map = _exact(
            inventory_payload["environment_sha256_by_world_size"],
            {"1", "2"},
            where="inventory environment map",
        )
        expected_git = _digest(expected_git_sha256, where="expected Git executable")
        if (
            inventory_payload["source_commit_sha"] != source_commit
            or inventory_payload["source_tree_sha"] != source_tree
            or inventory_payload["git_sha256"] != expected_git
            or environment_map["1"] != one["sha256"]
            or environment_map["2"] != two["sha256"]
            or one["payload"]["installed_distributions_sha256"] != installed_sha256
            or two["payload"]["installed_distributions_sha256"] != installed_sha256
            or _sha256_value(fingerprint) != installed_sha256
        ):
            raise ValueError("formal environment inventory binding mismatch")
    finally:
        os.close(parent_fd)

    return {
        "schema_version": 1,
        "kind": "formal_environment_transaction_verification",
        "completion_raw_sha256": completion_raw_sha256,
        "transaction_sha256": transaction_sha256,
        "source_commit_sha": source_commit,
        "source_tree_sha": source_tree,
        "git_sha256": expected_git,
        "one_gpu_environment_sha256": one["sha256"],
        "two_gpu_environment_sha256": two["sha256"],
        "inventory_sha256": inventory["sha256"],
        "installed_distributions_sha256": installed_sha256,
        "capture_visible_cuda_device_count": capture_count,
        "output_raw_sha256_by_role": {
            entry["role"]: entry["sha256"] for entry in normalized_outputs
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completion", required=True, type=Path)
    parser.add_argument("--expected-completion-sha256", required=True)
    parser.add_argument("--expected-transaction-sha256", required=True)
    parser.add_argument("--expected-one-gpu-environment-sha256", required=True)
    parser.add_argument("--expected-two-gpu-environment-sha256", required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--expected-source-commit-sha", required=True)
    parser.add_argument("--expected-source-tree-sha", required=True)
    parser.add_argument("--expected-git-sha256", required=True)
    parser.add_argument("--completion-max-bytes", required=True, type=int)
    parser.add_argument("--manifest-max-bytes", required=True, type=int)
    parser.add_argument("--inventory-max-bytes", required=True, type=int)
    args = parser.parse_args(argv)
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise ValueError("formal environment transaction verification requires Python -I -B")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise ValueError(
            "formal environment transaction verification requires PYTHONNOUSERSITE=1"
        )
    summary = _verify(
        completion_path=args.completion,
        expected_completion_sha256=args.expected_completion_sha256,
        expected_transaction_sha256=args.expected_transaction_sha256,
        expected_one_gpu_environment_sha256=args.expected_one_gpu_environment_sha256,
        expected_two_gpu_environment_sha256=args.expected_two_gpu_environment_sha256,
        expected_inventory_sha256=args.expected_inventory_sha256,
        expected_source_commit_sha=args.expected_source_commit_sha,
        expected_source_tree_sha=args.expected_source_tree_sha,
        expected_git_sha256=args.expected_git_sha256,
        completion_max_bytes=args.completion_max_bytes,
        manifest_max_bytes=args.manifest_max_bytes,
        inventory_max_bytes=args.inventory_max_bytes,
    )
    sys.stdout.buffer.write(_canonical(summary, newline=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"formal environment transaction rejected: {error}", file=sys.stderr)
        raise SystemExit(1)
