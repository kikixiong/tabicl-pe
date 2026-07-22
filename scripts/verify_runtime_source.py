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
_MODES = frozenset({"100644", "100755", "120000"})
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


def _open_regular(path):
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"refusing symlink/non-regular source path: {path}") from error
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
    if root.is_symlink() or not root.is_dir():
        _fail("archive root must be a real directory, not a symlink")
    root = root.resolve(strict=True)
    tracked = set()
    for entry in manifest["payload"]["entries"]:
        relative = entry["path"]
        tracked.add(relative)
        path = root.joinpath(*PurePosixPath(relative).parts)
        mode = entry["mode"]
        if mode == "120000":
            if not path.is_symlink():
                _fail(f"tracked symlink is missing or changed: {relative}")
            target = os.readlink(path)
            raw = os.fsencode(target)
            try:
                resolved = path.resolve(strict=True)
            except OSError as error:
                raise ValueError(
                    f"tracked symlink target is unavailable: {relative}"
                ) from error
            _inside(resolved, root, f"tracked symlink {relative}")
        else:
            if path.is_symlink():
                _fail(
                    f"tracked regular source unexpectedly became a symlink: {relative}"
                )
            raw, info = _read_regular(path)
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


def import_attested_tabicl(root, expected_src):
    # Isolated-mode Python ignores PYTHONPATH by design, so insert only the
    # already-attested exact source root and do not inherit any checkout path.
    sys.path.insert(0, str(expected_src))
    tabicl = importlib.import_module("tabicl")
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


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-commit-sha", required=True)
    parser.add_argument("--expected-tree-sha", required=True)
    return parser


def main(argv=None):
    if not sys.flags.isolated:
        _fail("runtime source verification requires isolated Python (-I)")
    if not sys.dont_write_bytecode:
        _fail("runtime source verification requires bytecode writes disabled (-B)")
    args = build_parser().parse_args(argv)
    manifest = load_and_validate_source_manifest(
        args.source_manifest,
        args.expected_manifest_sha256,
        args.expected_commit_sha,
        args.expected_tree_sha,
    )
    root = verify_archive(args.archive_root, manifest)
    expected_src = validate_environment(root)
    imported = import_attested_tabicl(root, expected_src)
    result = {
        "schema_version": 1,
        "commit_sha": manifest["payload"]["commit_sha"],
        "tree_sha": manifest["payload"]["tree_sha"],
        "source_manifest_sha256": manifest["sha256"],
        "import_relative_path": imported,
    }
    sys.stdout.buffer.write(_canonical(result) + b"\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"runtime source verification failed: {error}", file=sys.stderr)
        raise SystemExit(2)
