"""Strict, path-redacted contracts for a venv Python entry point."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
from typing import Any, Mapping


CONTRACT_KIND = "strict-venv-python-environment-v1"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_fd_bytes(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _normalized_absolute(path: Path, *, label: str) -> None:
    if not path.is_absolute() or Path(os.path.normpath(path)) != path:
        raise ValueError(f"{label} must be an absolute normalized path")


def _open_absolute_directory_nofollow(path: Path) -> int:
    _normalized_absolute(path, label="directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            replacement = os.open(
                component, flags | nofollow, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = replacement
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_at(
    parent_descriptor: int, name: str, *, label: str, executable: bool = False
) -> tuple[bytes, os.stat_result]:
    before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        opened_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or before.st_dev != opened_before.st_dev
            or before.st_ino != opened_before.st_ino
            or (executable and opened_before.st_mode & 0o111 == 0)
        ):
            raise ValueError(f"{label} is not the expected regular file")
        raw = _read_fd_bytes(descriptor)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if not (
        _stat_identity(before)
        == _stat_identity(opened_before)
        == _stat_identity(opened_after)
        == _stat_identity(after)
        and len(raw) == opened_after.st_size
    ):
        raise RuntimeError(f"{label} changed while it was read")
    return raw, opened_after


def _read_regular_path_nofollow(
    path: Path, *, label: str, executable: bool = False
) -> tuple[bytes, os.stat_result]:
    _normalized_absolute(path, label=label)
    parent_descriptor = _open_absolute_directory_nofollow(path.parent)
    try:
        return _read_regular_at(
            parent_descriptor, path.name, label=label, executable=executable
        )
    finally:
        os.close(parent_descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry_facts(entry: Path) -> tuple[dict[str, Any], Path, Path]:
    _normalized_absolute(entry, label="Python entry")
    parent_descriptor = _open_absolute_directory_nofollow(entry.parent)
    try:
        entry_before = os.stat(
            entry.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if not stat.S_ISLNK(entry_before.st_mode):
            raise ValueError("Python venv entry must be a final-component symlink")
        link_target = os.readlink(entry.name, dir_fd=parent_descriptor)
        if not link_target or "\0" in link_target:
            raise ValueError("Python venv entry symlink target is invalid")
    finally:
        os.close(parent_descriptor)
    target_candidate = Path(link_target)
    if not target_candidate.is_absolute():
        target_candidate = entry.parent / target_candidate
    real_executable = target_candidate.resolve(strict=True)
    executable_raw, executable_metadata = _read_regular_path_nofollow(
        real_executable, label="Python entry target", executable=True
    )
    venv_root = entry.parent.parent
    venv_descriptor = _open_absolute_directory_nofollow(venv_root)
    try:
        config_raw, config_metadata = _read_regular_at(
            venv_descriptor, "pyvenv.cfg", label="Python venv pyvenv.cfg"
        )
    finally:
        os.close(venv_descriptor)
    parent_descriptor = _open_absolute_directory_nofollow(entry.parent)
    try:
        entry_after = os.stat(
            entry.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        link_target_after = os.readlink(entry.name, dir_fd=parent_descriptor)
    finally:
        os.close(parent_descriptor)
    if (
        _stat_identity(entry_before) != _stat_identity(entry_after)
        or link_target_after != link_target
        or target_candidate.resolve(strict=True) != real_executable
    ):
        raise RuntimeError("Python venv entry changed while it was inspected")
    facts = {
        "entry_kind": "symlink",
        "entry_path_sha256": _sha256_bytes(str(entry).encode("utf-8")),
        "entry_link_target_sha256": _sha256_bytes(link_target.encode("utf-8")),
        "entry_symlink_size": entry_after.st_size,
        "real_executable_path_sha256": _sha256_bytes(
            str(real_executable).encode("utf-8")
        ),
        "real_executable_size": executable_metadata.st_size,
        "real_executable_sha256": _sha256_bytes(executable_raw),
        "pyvenv_cfg_size": config_metadata.st_size,
        "pyvenv_cfg_sha256": _sha256_bytes(config_raw),
        # Device/inode/mtime are intentionally transient: shared filesystems may
        # expose different st_dev values on login and compute nodes. They guard
        # each local inspection and /proc binding but are not frozen publicly.
        "_process_image": {
            "device": executable_metadata.st_dev,
            "inode": executable_metadata.st_ino,
            "size": executable_metadata.st_size,
            "mtime_ns": executable_metadata.st_mtime_ns,
            "sha256": _sha256_bytes(executable_raw),
        },
    }
    return facts, real_executable, venv_root


def _proc_self_exe_facts() -> dict[str, Any]:
    descriptor = os.open("/proc/self/exe", os.O_RDONLY)
    try:
        before = os.fstat(descriptor)
        raw = _read_fd_bytes(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after) or len(raw) != after.st_size:
        raise RuntimeError("current process executable changed while it was read")
    return {
        "device": after.st_dev,
        "inode": after.st_ino,
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": _sha256_bytes(raw),
    }


def _installed_distributions() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if not isinstance(name, str) or not name or not isinstance(version, str):
            raise ValueError("installed distribution metadata is incomplete")
        direct_url = distribution.read_text("direct_url.json")
        record_text = distribution.read_text("RECORD")
        records.append(
            {
                "name": name.casefold(),
                "version": version,
                "direct_url_sha256": (
                    hashlib.sha256(direct_url.encode("utf-8")).hexdigest()
                    if direct_url is not None
                    else ""
                ),
                "record_sha256": (
                    hashlib.sha256(record_text.encode("utf-8")).hexdigest()
                    if record_text is not None
                    else ""
                ),
            }
        )
    return sorted(
        records,
        key=lambda item: (
            item["name"],
            item["version"],
            item["direct_url_sha256"],
            item["record_sha256"],
        ),
    )


def _current_probe() -> dict[str, Any]:
    distributions = _installed_distributions()
    python_runtime = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "version_info": list(sys.version_info[:5]),
        "cache_tag": sys.implementation.cache_tag,
        "abi_flags": getattr(sys, "abiflags", ""),
        "prefix_semantics": (
            "venv_distinct_from_base"
            if sys.prefix != sys.base_prefix
            else "base_environment"
        ),
    }
    environment_payload = {
        "python_runtime": python_runtime,
        "installed_distributions": distributions,
    }
    return {
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "sys_base_prefix": sys.base_prefix,
        "proc_self_exe": _proc_self_exe_facts(),
        "python_runtime": python_runtime,
        "python_runtime_sha256": hashlib.sha256(_canonical(python_runtime)).hexdigest(),
        "installed_distribution_count": len(distributions),
        "installed_distribution_names": sorted(
            {record["name"] for record in distributions}
        ),
        "installed_distributions_sha256": hashlib.sha256(
            _canonical(distributions)
        ).hexdigest(),
        "environment_sha256": hashlib.sha256(
            _canonical(environment_payload)
        ).hexdigest(),
    }


def _payload_from_probe(
    *,
    entry: Path,
    facts: Mapping[str, Any],
    venv_root: Path,
    probe: Mapping[str, Any],
    required_distributions: tuple[str, ...],
) -> dict[str, Any]:
    if (
        Path(str(probe.get("sys_executable"))) != entry
        or Path(str(probe.get("sys_prefix"))) != venv_root
        or probe.get("sys_prefix") == probe.get("sys_base_prefix")
    ):
        raise ValueError("Python entry did not preserve its venv sys.prefix")
    expected_process_image = facts.get("_process_image")
    if probe.get("proc_self_exe") != expected_process_image:
        raise ValueError("current /proc/self/exe differs from the bound Python executable")
    runtime = probe.get("python_runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("Python runtime probe is malformed")
    installed_names = probe.get("installed_distribution_names")
    if (
        not isinstance(installed_names, list)
        or any(not isinstance(name, str) for name in installed_names)
        or not set(required_distributions).issubset(installed_names)
    ):
        raise ValueError("Python venv lacks a required installed distribution")
    return {
        "schema_version": 1,
        **{key: value for key, value in facts.items() if not key.startswith("_")},
        "venv_prefix_verified": True,
        "proc_self_exe_verified": True,
        "python_runtime": dict(runtime),
        "python_runtime_sha256": probe.get("python_runtime_sha256"),
        "installed_distribution_count": probe.get("installed_distribution_count"),
        "installed_distributions_sha256": probe.get("installed_distributions_sha256"),
        "environment_sha256": probe.get("environment_sha256"),
        "required_distributions": list(required_distributions),
    }


def _document(payload: Mapping[str, Any]) -> dict[str, Any]:
    base = {"schema_version": 1, "kind": CONTRACT_KIND, "payload": dict(payload)}
    return {**base, "sha256": hashlib.sha256(_canonical(base)).hexdigest()}


def build_python_environment_contract(
    entry: Path, *, required_distributions: tuple[str, ...] = ()
) -> dict[str, Any]:
    normalized_required = tuple(
        sorted({name.casefold() for name in required_distributions})
    )
    facts, _, venv_root = _entry_facts(entry)
    completed = subprocess.run(
        [str(entry), "-I", str(Path(__file__).resolve()), "--probe-current"],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        probe = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Python entry environment probe returned invalid JSON"
        ) from error
    if not isinstance(probe, Mapping):
        raise ValueError("Python entry environment probe is malformed")
    facts_after, _, venv_root_after = _entry_facts(entry)
    if facts_after != facts or venv_root_after != venv_root:
        raise RuntimeError("Python venv entry changed during environment probing")
    return _document(
        _payload_from_probe(
            entry=entry,
            facts=facts,
            venv_root=venv_root,
            probe=probe,
            required_distributions=normalized_required,
        )
    )


def verify_current_python_environment(entry: Path, document: Mapping[str, Any]) -> None:
    facts, _, venv_root = _entry_facts(entry)
    payload = document.get("payload")
    base = {"schema_version": 1, "kind": CONTRACT_KIND, "payload": payload}
    if (
        document.get("schema_version") != 1
        or document.get("kind") != CONTRACT_KIND
        or not isinstance(payload, Mapping)
        or set(document) != {"schema_version", "kind", "payload", "sha256"}
        or document.get("sha256") != hashlib.sha256(_canonical(base)).hexdigest()
    ):
        raise ValueError("Python environment contract self-hash is invalid")
    required = payload.get("required_distributions")
    if (
        not isinstance(required, list)
        or any(
            not isinstance(name, str) or name != name.casefold() for name in required
        )
        or required != sorted(set(required))
    ):
        raise ValueError("Python environment contract requirements are malformed")
    observed = _payload_from_probe(
        entry=entry,
        facts=facts,
        venv_root=venv_root,
        probe=_current_probe(),
        required_distributions=tuple(required),
    )
    facts_after, _, venv_root_after = _entry_facts(entry)
    if facts_after != facts or venv_root_after != venv_root:
        raise RuntimeError("Python venv entry changed during environment verification")
    if dict(payload) != observed:
        raise ValueError("Python venv entry/config/runtime environment changed")


def verify_python_environment_contract_file(
    *,
    entry: Path,
    contract_path: Path,
    expected_file_sha256: str,
    expected_document_sha256: str | None = None,
) -> None:
    if len(expected_file_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_file_sha256
    ):
        raise ValueError("expected Python contract file SHA-256 is malformed")
    raw, _ = _read_regular_path_nofollow(
        contract_path, label="Python environment contract"
    )
    if _sha256_bytes(raw) != expected_file_sha256:
        raise ValueError("Python contract file SHA-256 changed")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Python environment contract is invalid JSON") from error
    if not isinstance(document, Mapping):
        raise ValueError("Python environment contract root is malformed")
    if (
        expected_document_sha256 is not None
        and document.get("sha256") != expected_document_sha256
    ):
        raise ValueError("Python contract document SHA-256 changed")
    verify_current_python_environment(entry, document)


def main() -> int:
    if sys.argv[1:] != ["--probe-current"]:
        raise SystemExit("usage: python_environment.py --probe-current")
    print(json.dumps(_current_probe(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
