#!/usr/bin/env python3
"""Verify an extracted candidate before importing any TabICL module.

This entry point intentionally imports only the Python standard library at
module scope.  Run it in a fresh isolated interpreter with ``python -I -B``,
``PYTHONPATH=<archive>/src`` and ``PYTHONNOUSERSITE=1``.  The external expected
manifest digest/commit/tree are trust inputs; the copy inside the mutable
archive is never treated as an authority.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys


_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_MODES = frozenset({"100644", "100755"})
_CODE_SUFFIXES = frozenset(
    {".py", ".pyi", ".pyc", ".so", ".pyd", ".dylib", ".dll", ".pth"}
)


def _fail(message: str) -> "None":
    raise ValueError(message)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _normalize(value, where="value"):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(f"{where} contains a non-finite float")
        return value
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or "\x00" in key:
                _fail(f"{where} has an invalid object key")
            result[key] = _normalize(item, f"{where}.{key}")
        return result
    if isinstance(value, list):
        return [
            _normalize(item, f"{where}[{index}]") for index, item in enumerate(value)
        ]
    _fail(f"{where} has unsupported type {type(value).__name__}")


def _canonical(value):
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _exact_keys(value, keys, where):
    if not isinstance(value, dict) or set(value) != set(keys):
        actual = set(value) if isinstance(value, dict) else set()
        _fail(
            f"{where} keys mismatch; missing={sorted(set(keys) - actual)}, "
            f"extra={sorted(actual - set(keys))}"
        )


def _absolute_components(path, where):
    path = Path(path)
    if not path.is_absolute():
        _fail(f"{where} must be an absolute path")
    parts = path.parts
    if not parts or parts[0] != os.path.sep:
        _fail(f"{where} is not a supported absolute path")
    if any(part in {"", ".", ".."} for part in parts[1:]):
        _fail(f"{where} must be lexically normalized")
    if not hasattr(os, "O_NOFOLLOW") or os.open not in os.supports_dir_fd:
        _fail("no-follow path-component traversal is unavailable")
    return parts[1:]


def _directory_flags():
    if not hasattr(os, "O_DIRECTORY"):
        _fail("no-follow directory traversal is unavailable")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _open_directory_absolute(path, where):
    parts = _absolute_components(path, where)
    flags = _directory_flags()
    fd = None
    try:
        fd = os.open(os.path.sep, flags)
        for part in parts:
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            _fail(f"{where} is not a directory")
        return fd
    except OSError as error:
        if fd is not None:
            os.close(fd)
        raise ValueError(
            f"{where} has a symlink or invalid path component: {path}"
        ) from error


def _open_regular(path):
    parts = _absolute_components(path, "source path")
    if not parts:
        _fail("source path must name a regular file")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    parent = Path(os.path.sep, *parts[:-1])
    parent_fd = _open_directory_absolute(parent, "source path parent")
    try:
        fd = os.open(parts[-1], flags, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError(f"refusing symlink/non-regular source path: {path}") from error
    finally:
        os.close(parent_fd)
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        os.close(fd)
        _fail(f"source path is not a regular file: {path}")
    return fd, before


def _unchanged(before, after, where):
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        _fail(f"{where} changed while being read")


def _read_regular(path, max_bytes=None):
    fd, before = _open_regular(path)
    try:
        chunks = []
        total = 0
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                _fail(f"file exceeds size limit: {path}")
        _unchanged(before, os.fstat(fd), str(path))
    finally:
        os.close(fd)
    return b"".join(chunks), before


def _open_regular_beneath(root, relative):
    """Open a regular entry without following any path-component symlink."""
    directory_flags = _directory_flags()
    file_flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        file_flags |= os.O_CLOEXEC

    directory_fds = []
    try:
        directory_fds.append(_open_directory_absolute(root, "archive root"))
        if not stat.S_ISDIR(os.fstat(directory_fds[-1]).st_mode):
            _fail(f"archive root is not a directory: {root}")
        parts = PurePosixPath(relative).parts
        for part in parts[:-1]:
            directory_fds.append(
                os.open(part, directory_flags, dir_fd=directory_fds[-1])
            )
            if not stat.S_ISDIR(os.fstat(directory_fds[-1]).st_mode):
                _fail(f"tracked source parent is not a directory: {relative}")
        fd = os.open(parts[-1], file_flags, dir_fd=directory_fds[-1])
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            os.close(fd)
            _fail(f"tracked source is not a regular file: {relative}")
        return fd, before
    except OSError as error:
        raise ValueError(
            f"tracked source has a symlink or invalid path component: {relative}"
        ) from error
    finally:
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)


def _read_regular_beneath(root, relative):
    fd, before = _open_regular_beneath(root, relative)
    try:
        chunks = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        _unchanged(before, os.fstat(fd), relative)
    finally:
        os.close(fd)
    return b"".join(chunks), before


def _strict_json_file(path):
    raw, _ = _read_regular(path, max_bytes=32 << 20)
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: _fail(f"non-finite JSON constant: {token}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid strict JSON: {path}") from error


def _relative(raw, where):
    if not isinstance(raw, str) or not raw or "\x00" in raw or "\\" in raw:
        _fail(f"{where} path is invalid")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        _fail(f"{where} path must be normalized and relative")
    if path.as_posix() != raw:
        _fail(f"{where} path must be normalized")
    return raw


def _inside(path, root, where):
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{where} escapes exact archive") from error


def load_and_validate_source_manifest(
    path, expected_sha, expected_commit, expected_tree
):
    manifest = _strict_json_file(path)
    _exact_keys(
        manifest, {"schema_version", "kind", "payload", "sha256"}, "source manifest"
    )
    if manifest["schema_version"] != 1 or manifest["kind"] != "source":
        _fail("source manifest schema/kind mismatch")
    body = {
        "schema_version": manifest["schema_version"],
        "kind": manifest["kind"],
        "payload": manifest["payload"],
    }
    actual = hashlib.sha256(_canonical(body)).hexdigest()
    if manifest["sha256"] != actual:
        _fail("source manifest sha256 mismatch")
    if _HEX_64.fullmatch(expected_sha or "") is None or actual != expected_sha:
        _fail("source manifest does not match external expected manifest sha256")
    payload = manifest["payload"]
    _exact_keys(
        payload, {"commit_sha", "tree_sha", "code_roots", "entries"}, "source payload"
    )
    if (
        _HEX_40.fullmatch(expected_commit or "") is None
        or payload["commit_sha"] != expected_commit
    ):
        _fail("source commit does not match external expected commit")
    if (
        _HEX_40.fullmatch(expected_tree or "") is None
        or payload["tree_sha"] != expected_tree
    ):
        _fail("source tree does not match external expected tree")
    roots = payload["code_roots"]
    if not isinstance(roots, list) or roots != sorted(set(roots)):
        _fail("source code roots must be sorted and unique")
    for root in roots:
        _relative(root, "source code root")
    entries = payload["entries"]
    if not isinstance(entries, list):
        _fail("source entries must be a list")
    seen = set()
    last = None
    for entry in entries:
        _exact_keys(entry, {"path", "mode", "size", "sha256"}, "source entry")
        relative = _relative(entry["path"], "source entry")
        if relative in seen:
            _fail(f"duplicate source path: {relative}")
        if last is not None and relative < last:
            _fail("source entries must be sorted")
        seen.add(relative)
        last = relative
        if entry["mode"] == "120000":
            _fail(f"tracked symlink entries are forbidden: {relative}")
        if entry["mode"] not in _MODES:
            _fail(f"source mode is invalid for {relative}")
        if (
            isinstance(entry["size"], bool)
            or not isinstance(entry["size"], int)
            or entry["size"] < 0
        ):
            _fail(f"source size is invalid for {relative}")
        if _HEX_64.fullmatch(str(entry["sha256"])) is None:
            _fail(f"source digest is invalid for {relative}")
    return manifest


def verify_archive(root, manifest):
    root_fd = _open_directory_absolute(root, "archive root")
    os.close(root_fd)
    tracked = set()
    for entry in manifest["payload"]["entries"]:
        relative = entry["path"]
        tracked.add(relative)
        mode = entry["mode"]
        raw, info = _read_regular_beneath(root, relative)
        executable = bool(info.st_mode & stat.S_IXUSR)
        if executable != (mode == "100755"):
            _fail(f"tracked source executable mode mismatch: {relative}")
        if len(raw) != entry["size"]:
            _fail(f"tracked source size mismatch: {relative}")
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            _fail(f"tracked source content digest mismatch: {relative}")

    # ``<archive>/src`` is inserted as one importable sys.path entry below.
    # Scan that entire entry even if a malformed-but-externally-signed manifest
    # lists only ``src/tabicl``: otherwise an untracked top-level ``torch.py``
    # or package could shadow a trusted dependency before attestation returns.
    importable_roots = set(manifest["payload"]["code_roots"])
    importable_roots.add("src")
    for relative_root in sorted(importable_roots):
        code_root = root.joinpath(*PurePosixPath(relative_root).parts)
        if not code_root.exists():
            continue
        if code_root.is_symlink():
            _fail(f"tracked code root must not be a symlink: {relative_root}")
        _inside(code_root.resolve(strict=True), root, f"code root {relative_root}")
        for directory, names, files in os.walk(code_root, followlinks=False):
            directory_path = Path(directory)
            for name in names:
                candidate = directory_path / name
                if candidate.is_symlink():
                    _fail(f"unexpected symlink directory under code root: {candidate}")
            for name in files:
                candidate = directory_path / name
                suffix = candidate.suffix.lower()
                if suffix not in _CODE_SUFFIXES:
                    continue
                relative = candidate.relative_to(root).as_posix()
                if relative not in tracked:
                    _fail(f"unexpected code file under tracked code root: {relative}")
    return root


def validate_environment(root):
    expected_src = root / "src"
    raw_pythonpath = os.environ.get("PYTHONPATH")
    if raw_pythonpath != str(expected_src):
        _fail(f"PYTHONPATH must be exactly {expected_src}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        _fail("PYTHONNOUSERSITE must be exactly 1")
    if "tabicl" in sys.modules or any(
        name.startswith("tabicl.") for name in sys.modules
    ):
        _fail("tabicl must not be preloaded before source attestation")
    return expected_src


def attest_loaded_tabicl(root, expected_src):
    tabicl = sys.modules.get("tabicl")
    if tabicl is None:
        _fail("tabicl has not been imported from the attested checkout")
    expected_init = expected_src / "tabicl" / "__init__.py"
    module_file = Path(tabicl.__file__).resolve(strict=True)
    if module_file != expected_init.resolve(strict=True):
        _fail("tabicl.__file__ is not the exact attested checkout")
    paths = [Path(item).resolve(strict=True) for item in tabicl.__path__]
    if paths != [(expected_src / "tabicl").resolve(strict=True)]:
        _fail("tabicl.__path__ must contain exactly the attested package directory")
    for name, module in list(sys.modules.items()):
        if name != "tabicl" and not name.startswith("tabicl."):
            continue
        module_path = getattr(module, "__file__", None)
        if module_path is None:
            continue
        resolved = Path(module_path).resolve(strict=True)
        _inside(resolved, expected_src, f"loaded module {name}")
    return module_file.relative_to(root).as_posix()


def import_attested_tabicl(root, expected_src):
    # Isolated-mode Python ignores PYTHONPATH by design, so insert only the
    # already-attested exact source root and do not inherit any checkout path.
    sys.path.insert(0, str(expected_src))
    importlib.import_module("tabicl")
    return attest_loaded_tabicl(root, expected_src)


def run_attested_trainer(root, expected_src, manifest, args, trainer_argv):
    bootstrap_relative = "scripts/verify_runtime_source.py"
    expected_bootstrap = (root / bootstrap_relative).resolve(strict=True)
    actual_bootstrap = Path(__file__).resolve(strict=True)
    tracked = {entry["path"] for entry in manifest["payload"]["entries"]}
    if actual_bootstrap != expected_bootstrap or bootstrap_relative not in tracked:
        _fail("compute bootstrap is not the tracked exact-T verifier")
    config_module = importlib.import_module("tabicl.train._train_config")
    run_module = importlib.import_module("tabicl.train._run")
    provenance_module = importlib.import_module("tabicl.train._provenance")
    fixed_modules = {
        config_module: "src/tabicl/train/_train_config.py",
        run_module: "src/tabicl/train/_run.py",
        provenance_module: "src/tabicl/train/_provenance.py",
    }
    for module, relative in fixed_modules.items():
        actual = Path(module.__file__).resolve(strict=True)
        expected = (root / relative).resolve(strict=True)
        if actual != expected:
            _fail(f"fixed trainer module is not from exact checkout: {relative}")
    attest_loaded_tabicl(root, expected_src)

    config = config_module.build_parser().parse_args(trainer_argv)
    trust_inputs = {
        "formal_source_manifest": os.fspath(args.source_manifest),
        "formal_source_sha256": manifest["sha256"],
        "formal_source_commit_sha": manifest["payload"]["commit_sha"],
        "formal_source_tree_sha": manifest["payload"]["tree_sha"],
    }
    if getattr(config, "formal_training", None) is not True:
        _fail("restricted bootstrap requires --formal_training true")
    for name, expected in trust_inputs.items():
        if getattr(config, name, None) != expected:
            _fail(f"Trainer {name} does not match attested bootstrap input")

    trainer = run_module.Trainer(config)
    attest_loaded_tabicl(root, expected_src)
    trainer.train()
    attest_loaded_tabicl(root, expected_src)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-commit-sha", required=True)
    parser.add_argument("--expected-tree-sha", required=True)
    parser.add_argument("--run-trainer", action="store_true")
    return parser


def main(argv=None):
    if not sys.flags.isolated:
        _fail("runtime source verification requires isolated Python (-I)")
    if not sys.dont_write_bytecode:
        _fail("runtime source verification requires bytecode writes disabled (-B)")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    delimiters = [index for index, item in enumerate(raw_argv) if item == "--"]
    if len(delimiters) > 1:
        _fail("only one trainer argument delimiter is allowed")
    if delimiters:
        split = delimiters[0]
        bootstrap_argv = raw_argv[:split]
        trainer_argv = raw_argv[split + 1 :]
    else:
        bootstrap_argv = raw_argv
        trainer_argv = None
    args = build_parser().parse_args(bootstrap_argv)
    if args.run_trainer != (trainer_argv is not None):
        _fail("--run-trainer requires a literal -- trainer argument delimiter")
    manifest = load_and_validate_source_manifest(
        args.source_manifest,
        args.expected_manifest_sha256,
        args.expected_commit_sha,
        args.expected_tree_sha,
    )
    root = verify_archive(args.archive_root, manifest)
    expected_src = validate_environment(root)
    imported = import_attested_tabicl(root, expected_src)
    if args.run_trainer:
        run_attested_trainer(root, expected_src, manifest, args, trainer_argv)
    result = {
        "schema_version": 1,
        "commit_sha": manifest["payload"]["commit_sha"],
        "tree_sha": manifest["payload"]["tree_sha"],
        "source_manifest_sha256": manifest["sha256"],
        "import_relative_path": imported,
    }
    if args.run_trainer:
        result["compute"] = "completed"
    sys.stdout.buffer.write(_canonical(result) + b"\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"runtime source verification failed: {error}", file=sys.stderr)
        raise SystemExit(2)
