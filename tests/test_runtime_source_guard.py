from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tabicl.train._provenance import (
    build_source_manifest_from_files,
    canonical_json_bytes,
    make_manifest,
)


SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_runtime_source.py"


def _archive(tmp_path: Path):
    root = tmp_path / "candidate"
    package = root / "src" / "tabicl"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("SOURCE = 'candidate'\n")
    (package / "module.py").write_text("VALUE = 1\n")
    manifest = build_source_manifest_from_files(
        root,
        commit_sha="1" * 40,
        tree_sha="2" * 40,
        tracked={
            "src/tabicl/__init__.py": "100644",
            "src/tabicl/module.py": "100644",
        },
        code_roots=("src/tabicl",),
    )
    manifest_path = tmp_path / "trusted-source.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return root, manifest, manifest_path


def _run(
    root,
    manifest,
    manifest_path,
    *,
    pythonpath=None,
    commit=None,
    tree=None,
    extra_env=None,
    bytecode_disabled=True,
    trainer_args=None,
    bootstrap_args=None,
    script=SCRIPT,
):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") if pythonpath is None else str(pythonpath)
    env["PYTHONNOUSERSITE"] = "1"
    env.update(extra_env or {})
    interpreter = [sys.executable, "-I"]
    if bytecode_disabled:
        interpreter.append("-B")
    command = [
        *interpreter,
        str(script),
        "--archive-root",
        str(root),
        "--source-manifest",
        str(manifest_path),
        "--expected-manifest-sha256",
        manifest["sha256"],
        "--expected-commit-sha",
        commit or "1" * 40,
        "--expected-tree-sha",
        tree or "2" * 40,
    ]
    command.extend(bootstrap_args or [])
    if trainer_args is not None:
        command.extend(["--run-trainer", "--", *trainer_args])
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=env,
        cwd=root,
    )


def test_exact_archive_and_safe_pythonpath_import_pass(tmp_path):
    root, manifest, manifest_path = _archive(tmp_path)
    result = _run(root, manifest, manifest_path)
    assert result.returncode == 0, result.stderr
    attestation = json.loads(result.stdout)
    assert attestation["source_manifest_sha256"] == manifest["sha256"]
    assert attestation["import_relative_path"] == "src/tabicl/__init__.py"


def test_isolated_guard_manual_prepend_wins_over_visible_editable_install(tmp_path):
    root, manifest, manifest_path = _archive(tmp_path)
    visible = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import pathlib,tabicl; print(pathlib.Path(tabicl.__file__).resolve())",
        ],
        text=True,
        capture_output=True,
    )
    assert visible.returncode == 0, visible.stderr
    assert Path(visible.stdout.strip()) != (root / "src/tabicl/__init__.py").resolve()

    result = _run(root, manifest, manifest_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["import_relative_path"] == "src/tabicl/__init__.py"


def test_restricted_bootstrap_runs_exact_candidate_trainer_in_same_process(tmp_path):
    root = tmp_path / "candidate"
    package = root / "src" / "tabicl"
    train = package / "train"
    train.mkdir(parents=True)
    (package / "__init__.py").write_text("SOURCE = 'candidate'\n")
    (train / "__init__.py").write_text("")
    (train / "_provenance.py").write_text("SOURCE = 'candidate-provenance'\n")
    (train / "_train_config.py").write_text(
        "import argparse\n"
        "def _bool(value):\n"
        "    return value.lower() == 'true'\n"
        "def build_parser():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--marker', required=True)\n"
        "    parser.add_argument('--formal_training', type=_bool, required=True)\n"
        "    parser.add_argument('--formal_source_manifest', required=True)\n"
        "    parser.add_argument('--formal_source_sha256', required=True)\n"
        "    parser.add_argument('--formal_source_commit_sha', required=True)\n"
        "    parser.add_argument('--formal_source_tree_sha', required=True)\n"
        "    return parser\n"
    )
    (train / "_run.py").write_text(
        "from pathlib import Path\n"
        "import __main__\n"
        "from tabicl.train._provenance import SOURCE\n"
        "class Trainer:\n"
        "    def __init__(self, config):\n"
        "        self.config = config\n"
        "    def train(self):\n"
        "        Path(self.config.marker).write_text(\n"
        "            SOURCE + ':' + __file__ + ':' + __main__.__file__\n"
        "        )\n"
    )
    bootstrap = root / "scripts" / "verify_runtime_source.py"
    bootstrap.parent.mkdir()
    bootstrap.write_bytes(SCRIPT.read_bytes())
    bootstrap.chmod(0o755)
    tracked = {
        path.relative_to(root).as_posix(): "100644"
        for path in sorted(package.rglob("*.py"))
    }
    tracked["scripts/verify_runtime_source.py"] = "100755"
    manifest = build_source_manifest_from_files(
        root,
        commit_sha="1" * 40,
        tree_sha="2" * 40,
        tracked=tracked,
        code_roots=("scripts", "src/tabicl"),
    )
    manifest_path = tmp_path / "trusted-source.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    marker = tmp_path / "trainer-ran"

    visible = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import pathlib,tabicl; print(pathlib.Path(tabicl.__file__).resolve())",
        ],
        text=True,
        capture_output=True,
    )
    assert visible.returncode == 0, visible.stderr
    assert Path(visible.stdout.strip()) != (package / "__init__.py").resolve()

    stale_bootstrap = _run(
        root,
        manifest,
        manifest_path,
        trainer_args=[
            "--marker",
            str(marker),
            "--formal_training",
            "true",
            "--formal_source_manifest",
            str(manifest_path),
            "--formal_source_sha256",
            manifest["sha256"],
            "--formal_source_commit_sha",
            "1" * 40,
            "--formal_source_tree_sha",
            "2" * 40,
        ],
    )
    assert stale_bootstrap.returncode != 0
    assert "tracked exact-T verifier" in stale_bootstrap.stderr
    assert not marker.exists()

    result = _run(
        root,
        manifest,
        manifest_path,
        script=bootstrap,
        trainer_args=[
            "--marker",
            str(marker),
            "--formal_training",
            "true",
            "--formal_source_manifest",
            str(manifest_path),
            "--formal_source_sha256",
            manifest["sha256"],
            "--formal_source_commit_sha",
            "1" * 40,
            "--formal_source_tree_sha",
            "2" * 40,
        ],
    )

    assert result.returncode == 0, result.stderr
    assert marker.read_text().startswith("candidate-provenance:")
    assert str(train.resolve()) in marker.read_text()
    assert str(bootstrap.resolve()) in marker.read_text()
    assert json.loads(result.stdout)["compute"] == "completed"


@pytest.mark.parametrize("option", ["--module", "--command"])
def test_restricted_bootstrap_rejects_arbitrary_execution_options(tmp_path, option):
    root, manifest, manifest_path = _archive(tmp_path)
    marker = tmp_path / "must-not-run"

    result = _run(
        root,
        manifest,
        manifest_path,
        bootstrap_args=[option, f"touch {marker}"],
    )

    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr
    assert not marker.exists()


def test_tracked_byte_edit_is_rejected(tmp_path):
    root, manifest, manifest_path = _archive(tmp_path)
    (root / "src/tabicl/module.py").write_text("VALUE = 2\n")
    result = _run(root, manifest, manifest_path)
    assert result.returncode != 0
    assert "content digest" in result.stderr


def test_manifest_and_archive_rehash_still_fail_external_digest(tmp_path):
    root, original, manifest_path = _archive(tmp_path)
    (root / "src/tabicl/module.py").write_text("VALUE = 2\n")
    attacker = build_source_manifest_from_files(
        root,
        commit_sha="1" * 40,
        tree_sha="2" * 40,
        tracked={
            "src/tabicl/__init__.py": "100644",
            "src/tabicl/module.py": "100644",
        },
        code_roots=("src/tabicl",),
    )
    manifest_path.write_bytes(canonical_json_bytes(attacker))
    result = _run(root, original, manifest_path)
    assert result.returncode != 0
    assert "external expected manifest sha256" in result.stderr


def test_symlink_escape_is_rejected_before_import(tmp_path):
    root, manifest, manifest_path = _archive(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("SOURCE = 'outside'\n")
    init = root / "src/tabicl/__init__.py"
    init.unlink()
    init.symlink_to(outside)
    result = _run(root, manifest, manifest_path)
    assert result.returncode != 0
    assert "symlink" in result.stderr


def test_archive_root_parent_symlink_is_rejected(tmp_path):
    actual_parent = tmp_path / "actual"
    root, manifest, manifest_path = _archive(actual_parent)
    alias = tmp_path / "alias"
    alias.symlink_to(actual_parent, target_is_directory=True)

    result = _run(alias / root.name, manifest, manifest_path)

    assert result.returncode != 0
    assert "symlink" in result.stderr


def test_source_manifest_parent_symlink_is_rejected(tmp_path):
    root, manifest, _manifest_path = _archive(tmp_path)
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    manifest_path = trusted / "source.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    alias = tmp_path / "trusted-alias"
    alias.symlink_to(trusted, target_is_directory=True)

    result = _run(root, manifest, alias / manifest_path.name)

    assert result.returncode != 0
    assert "symlink" in result.stderr


def test_regular_entry_parent_symlink_escape_is_rejected(tmp_path):
    root, _manifest, _manifest_path = _archive(tmp_path)
    docs = root / "docs"
    docs.mkdir()
    (docs / "note.txt").write_text("trusted bytes\n")
    manifest = build_source_manifest_from_files(
        root,
        commit_sha="1" * 40,
        tree_sha="2" * 40,
        tracked={
            "docs/note.txt": "100644",
            "src/tabicl/__init__.py": "100644",
            "src/tabicl/module.py": "100644",
        },
        code_roots=("src/tabicl",),
    )
    manifest_path = tmp_path / "parent-symlink-source.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    outside = tmp_path / "outside-docs"
    outside.mkdir()
    (outside / "note.txt").write_text("trusted bytes\n")
    (docs / "note.txt").unlink()
    docs.rmdir()
    docs.symlink_to(outside, target_is_directory=True)

    result = _run(root, manifest, manifest_path)
    assert result.returncode != 0
    assert "symlink" in result.stderr


def test_tracked_code_symlink_to_untracked_internal_payload_is_rejected_before_execution(
    tmp_path,
):
    root, _manifest, _manifest_path = _archive(tmp_path)
    marker = tmp_path / "tracked-symlink-executed"
    payload = root / "src/payload.txt"
    payload.write_text(
        "from pathlib import Path\n"
        "import os\n"
        "Path(os.environ['SYMLINK_MARKER']).write_text('executed')\n"
    )
    module = root / "src/tabicl/module.py"
    module.unlink()
    module.symlink_to("../payload.txt")
    (root / "src/tabicl/__init__.py").write_text("from . import module\n")
    init_bytes = (root / "src/tabicl/__init__.py").read_bytes()
    link_bytes = os.fsencode("../payload.txt")
    manifest = make_manifest(
        "source",
        {
            "commit_sha": "1" * 40,
            "tree_sha": "2" * 40,
            "code_roots": ["src/tabicl"],
            "entries": [
                {
                    "path": "src/tabicl/__init__.py",
                    "mode": "100644",
                    "size": len(init_bytes),
                    "sha256": hashlib.sha256(init_bytes).hexdigest(),
                },
                {
                    "path": "src/tabicl/module.py",
                    "mode": "120000",
                    "size": len(link_bytes),
                    "sha256": hashlib.sha256(link_bytes).hexdigest(),
                },
            ],
        },
    )
    manifest_path = tmp_path / "tracked-symlink-source.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    result = _run(
        root,
        manifest,
        manifest_path,
        extra_env={"SYMLINK_MARKER": str(marker)},
    )
    assert result.returncode != 0
    assert "symlink" in result.stderr
    assert not marker.exists()


def test_inherited_or_multi_entry_pythonpath_is_rejected(tmp_path):
    root, manifest, manifest_path = _archive(tmp_path)
    other = tmp_path / "other" / "src"
    (other / "tabicl").mkdir(parents=True)
    (other / "tabicl/__init__.py").write_text("SOURCE = 'other'\n")
    result = _run(
        root,
        manifest,
        manifest_path,
        pythonpath=f"{other}{os.pathsep}{root / 'src'}",
    )
    assert result.returncode != 0
    assert "PYTHONPATH" in result.stderr


def test_unexpected_python_or_native_extension_is_rejected(tmp_path):
    root, manifest, manifest_path = _archive(tmp_path)
    (root / "src/tabicl/injected.py").write_text("PWNED = True\n")
    result = _run(root, manifest, manifest_path)
    assert result.returncode != 0
    assert "unexpected code file" in result.stderr


def test_untracked_top_level_src_shadow_is_rejected_before_import(tmp_path):
    root, manifest, manifest_path = _archive(tmp_path)
    marker = tmp_path / "shadow-executed"
    (root / "src/evil.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        "Path(os.environ['SHADOW_MARKER']).write_text('executed')\n"
    )
    (root / "src/tabicl/__init__.py").write_text("import evil\n")
    manifest = build_source_manifest_from_files(
        root,
        commit_sha="1" * 40,
        tree_sha="2" * 40,
        tracked={
            "src/tabicl/__init__.py": "100644",
            "src/tabicl/module.py": "100644",
        },
        code_roots=("src/tabicl",),
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    result = _run(
        root,
        manifest,
        manifest_path,
        extra_env={"SHADOW_MARKER": str(marker)},
    )

    assert result.returncode != 0
    assert "unexpected code file" in result.stderr
    assert not marker.exists()


@pytest.mark.parametrize("suffix", [".pyi", ".so", ".pyd", ".dylib", ".pth"])
def test_unexpected_code_like_extensions_are_rejected(tmp_path, suffix):
    root, manifest, manifest_path = _archive(tmp_path)
    (root / f"src/tabicl/injected{suffix}").write_bytes(b"injected")
    result = _run(root, manifest, manifest_path)
    assert result.returncode != 0
    assert "unexpected code file" in result.stderr


def test_tracked_deletion_and_executable_mode_drift_are_rejected(tmp_path):
    root, manifest, manifest_path = _archive(tmp_path)
    (root / "src/tabicl/module.py").unlink()
    assert _run(root, manifest, manifest_path).returncode != 0

    root, manifest, manifest_path = _archive(tmp_path / "mode")
    tracked = root / "src/tabicl/module.py"
    tracked.chmod(0o755)
    result = _run(root, manifest, manifest_path)
    assert result.returncode != 0
    assert "executable mode" in result.stderr


def test_python_no_user_site_is_required(tmp_path):
    root, manifest, manifest_path = _archive(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env.pop("PYTHONNOUSERSITE", None)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(SCRIPT),
            "--archive-root",
            str(root),
            "--source-manifest",
            str(manifest_path),
            "--expected-manifest-sha256",
            manifest["sha256"],
            "--expected-commit-sha",
            "1" * 40,
            "--expected-tree-sha",
            "2" * 40,
        ],
        text=True,
        capture_output=True,
        env=env,
        cwd=root,
    )
    assert result.returncode != 0
    assert "PYTHONNOUSERSITE" in result.stderr


def test_non_isolated_python_invocation_is_rejected(tmp_path):
    root, manifest, manifest_path = _archive(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--archive-root",
            str(root),
            "--source-manifest",
            str(manifest_path),
            "--expected-manifest-sha256",
            manifest["sha256"],
            "--expected-commit-sha",
            "1" * 40,
            "--expected-tree-sha",
            "2" * 40,
        ],
        text=True,
        capture_output=True,
        env=env,
        cwd=root,
    )
    assert result.returncode != 0
    assert "isolated" in result.stderr


def test_bytecode_writes_are_disabled_and_repeated_attestation_is_clean(tmp_path):
    root, manifest, manifest_path = _archive(tmp_path)
    unsafe = _run(
        root,
        manifest,
        manifest_path,
        bytecode_disabled=False,
    )
    assert unsafe.returncode != 0
    assert "bytecode" in unsafe.stderr
    assert not list((root / "src").rglob("__pycache__"))

    first = _run(root, manifest, manifest_path)
    second = _run(root, manifest, manifest_path)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert not list((root / "src").rglob("__pycache__"))


def test_mixed_loaded_tabicl_submodule_outside_exact_tree_is_rejected(tmp_path):
    root, _manifest, _manifest_path = _archive(tmp_path)
    outside = tmp_path / "outside_submodule.py"
    outside.write_text("VALUE = 'outside'\n")
    init = root / "src/tabicl/__init__.py"
    init.write_text(
        "import importlib.util, os, sys\n"
        "spec = importlib.util.spec_from_file_location('tabicl.mixed', os.environ['MIXED_PATH'])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules['tabicl.mixed'] = module\n"
        "spec.loader.exec_module(module)\n"
    )
    manifest = build_source_manifest_from_files(
        root,
        commit_sha="1" * 40,
        tree_sha="2" * 40,
        tracked={
            "src/tabicl/__init__.py": "100644",
            "src/tabicl/module.py": "100644",
        },
        code_roots=("src/tabicl",),
    )
    manifest_path = tmp_path / "mixed-source.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env["PYTHONNOUSERSITE"] = "1"
    env["MIXED_PATH"] = str(outside)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(SCRIPT),
            "--archive-root",
            str(root),
            "--source-manifest",
            str(manifest_path),
            "--expected-manifest-sha256",
            manifest["sha256"],
            "--expected-commit-sha",
            "1" * 40,
            "--expected-tree-sha",
            "2" * 40,
        ],
        text=True,
        capture_output=True,
        env=env,
        cwd=root,
    )
    assert result.returncode != 0
    assert "loaded module tabicl.mixed escapes exact archive" in result.stderr


def test_strict_json_rejects_duplicate_object_keys(tmp_path):
    root, manifest, manifest_path = _archive(tmp_path)
    raw = canonical_json_bytes(manifest).decode()
    raw = raw.replace('"kind":"source"', '"kind":"source","kind":"source"', 1)
    manifest_path.write_text(raw)
    result = _run(root, manifest, manifest_path)
    assert result.returncode != 0
    assert "duplicate JSON key" in result.stderr


@pytest.mark.parametrize(
    "bad_entries,match",
    [
        (
            [{"path": "../escape.py", "mode": "100644", "size": 0, "sha256": "0" * 64}],
            "path",
        ),
        (
            [
                {
                    "path": "src/tabicl/__init__.py",
                    "mode": "100644",
                    "size": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                },
                {
                    "path": "src/tabicl/__init__.py",
                    "mode": "100644",
                    "size": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                },
            ],
            "duplicate",
        ),
        (
            [{"path": "/absolute.py", "mode": "100644", "size": 0, "sha256": "0" * 64}],
            "path",
        ),
        (
            [{"path": "a/../b.py", "mode": "100644", "size": 0, "sha256": "0" * 64}],
            "path",
        ),
        (
            [{"path": "a//b.py", "mode": "100644", "size": 0, "sha256": "0" * 64}],
            "path",
        ),
        (
            [{"path": "a\x00b.py", "mode": "100644", "size": 0, "sha256": "0" * 64}],
            "path",
        ),
    ],
)
def test_manifest_rejects_path_traversal_and_duplicates(tmp_path, bad_entries, match):
    root, manifest, manifest_path = _archive(tmp_path)
    body = dict(manifest["payload"])
    body["entries"] = bad_entries
    from tabicl.train._provenance import make_manifest

    bad = make_manifest("source", body)
    manifest_path.write_bytes(canonical_json_bytes(bad))
    result = _run(root, bad, manifest_path)
    assert result.returncode != 0
    assert match in result.stderr


def test_wrong_external_commit_or_tree_is_rejected(tmp_path):
    root, manifest, manifest_path = _archive(tmp_path)
    assert _run(root, manifest, manifest_path, commit="3" * 40).returncode != 0
    assert _run(root, manifest, manifest_path, tree="4" * 40).returncode != 0


def test_guard_has_no_tabicl_import_before_attestation():
    source = SCRIPT.read_text()
    prefix = source.split("def import_attested_tabicl", 1)[0]
    assert "import tabicl" not in prefix
