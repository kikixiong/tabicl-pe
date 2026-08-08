from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import sys
import textwrap

import pytest

from tabicl.train._provenance import (
    _FORMAL_RUNTIME_DISTRIBUTIONS,
    _FORMAL_RUNTIME_MODULES,
    canonical_sha256,
    make_manifest,
    validate_environment_payload,
)


ROOT = Path(__file__).parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_environment_generator_rejects_fifo_git_without_blocking(tmp_path):
    fifo = tmp_path / "git"
    os.mkfifo(fifo)
    script = ROOT / "scripts/generate_formal_environment.py"
    code = (
        "import importlib.util,pathlib,sys\n"
        "spec=importlib.util.spec_from_file_location('fifo_env_generator',sys.argv[1])\n"
        "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n"
        "module._open_trusted_git(pathlib.Path(sys.argv[2]),expected_sha256='0'*64)\n"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(script), str(fifo)],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert completed.returncode != 0
    assert "bounded regular executable" in completed.stderr


def _run(
    *argv: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=30,
    )


def _git(*argv: str, cwd: Path) -> str:
    completed = _run("git", *argv, cwd=cwd)
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    return completed.stdout.strip()


def _make_generator_checkout(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "exact-t"
    (root / "scripts").mkdir(parents=True)
    package = root / "src/tabicl/train"
    package.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/generate_formal_environment.py", root / "scripts")
    (root / "src/tabicl/__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "_provenance.py").write_text(
        textwrap.dedent(
            """
            import hashlib
            import json

            def canonical_sha256(value):
                raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
                return hashlib.sha256(raw).hexdigest()

            def make_manifest(kind, payload):
                body = {"schema_version": 1, "kind": kind, "payload": payload}
                return {**body, "sha256": canonical_sha256(body)}

            def validate_environment_payload(payload, *, require_formal_runtime):
                if payload["visible_cuda_device_count"] not in (1, 2):
                    raise ValueError("bad fake CUDA count")

            def runtime_environment_snapshot(*, require_formal_runtime):
                preimage = {
                    "visible_distribution_multiset": [{"name": "fake"}],
                    "effective_formal_runtime_distributions": [{"name": "fake"}],
                }
                payload = {
                    "installed_distributions_sha256": canonical_sha256(preimage),
                    "environment_fingerprint_schema_version": 2,
                    "visible_cuda_device_count": 7,
                    "software": "fixed",
                }
                return make_manifest("environment", payload), preimage
            """
        ).lstrip()
    )
    _git("init", "-q", cwd=root)
    _git("config", "user.name", "Test", cwd=root)
    _git("config", "user.email", "test@example.invalid", cwd=root)
    _git("add", ".", cwd=root)
    _git("commit", "-qm", "exact T", cwd=root)
    commit = _git("rev-parse", "HEAD^{commit}", cwd=root)
    tree = _git("rev-parse", "HEAD^{tree}", cwd=root)
    _git("checkout", "-q", "--detach", commit, cwd=root)
    return root, commit, tree


def _generator_argv(root: Path, commit: str, tree: str, output: Path) -> list[str]:
    git = Path(shutil.which("git") or "").resolve(strict=True)
    git_sha256 = hashlib.sha256(git.read_bytes()).hexdigest()
    return [
        sys.executable,
        "-I",
        "-B",
        str(root / "scripts/generate_formal_environment.py"),
        "--exact-root",
        str(root),
        "--git",
        str(git),
        "--expected-git-sha256",
        git_sha256,
        "--expected-commit-sha",
        commit,
        "--expected-tree-sha",
        tree,
        "--one-gpu-output",
        str(output / "one.json"),
        "--two-gpu-output",
        str(output / "two.json"),
        "--inventory-output",
        str(output / "inventory.json"),
        "--completion-output",
        str(output / "complete.json"),
        "--manifest-max-bytes",
        str(1 << 20),
        "--inventory-max-bytes",
        str(1 << 20),
        "--completion-max-bytes",
        str(1 << 20),
    ]


def _verify_transaction_argv(
    output: Path,
    *,
    expected_commit: str | None = None,
) -> tuple[list[str], dict]:
    completion_raw = (output / "complete.json").read_bytes()
    completion = json.loads(completion_raw)
    one = json.loads((output / "one.json").read_text())
    two = json.loads((output / "two.json").read_text())
    inventory = json.loads((output / "inventory.json").read_text())
    values = {
        "completion_raw_sha256": hashlib.sha256(completion_raw).hexdigest(),
        "transaction_sha256": completion["transaction_sha256"],
        "one_gpu_environment_sha256": one["sha256"],
        "two_gpu_environment_sha256": two["sha256"],
        "inventory_sha256": inventory["sha256"],
        "source_commit_sha": completion["source_commit_sha"],
        "source_tree_sha": completion["source_tree_sha"],
        "git_sha256": inventory["payload"]["git_sha256"],
    }
    return (
        [
            sys.executable,
            "-I",
            "-B",
            str(ROOT / "scripts/verify_formal_environment_transaction.py"),
            "--completion",
            str(output / "complete.json"),
            "--expected-completion-sha256",
            values["completion_raw_sha256"],
            "--expected-transaction-sha256",
            values["transaction_sha256"],
            "--expected-one-gpu-environment-sha256",
            values["one_gpu_environment_sha256"],
            "--expected-two-gpu-environment-sha256",
            values["two_gpu_environment_sha256"],
            "--expected-inventory-sha256",
            values["inventory_sha256"],
            "--expected-source-commit-sha",
            expected_commit or values["source_commit_sha"],
            "--expected-source-tree-sha",
            values["source_tree_sha"],
            "--expected-git-sha256",
            values["git_sha256"],
            "--completion-max-bytes",
            str(1 << 20),
            "--manifest-max-bytes",
            str(1 << 20),
            "--inventory-max-bytes",
            str(1 << 20),
        ],
        values,
    )


def _generate_fake_transaction(tmp_path: Path) -> tuple[Path, list[str], dict]:
    root, commit, tree = _make_generator_checkout(tmp_path)
    output = tmp_path / "durable"
    output.mkdir()
    generated = _run(
        *_generator_argv(root, commit, tree, output),
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    assert generated.returncode == 0, generated.stderr
    argv, expected = _verify_transaction_argv(output)
    return output, argv, expected


def _environment_payload(*, visible_cuda_device_count: int) -> dict:
    return {
        "python_version": "3.10.0",
        "python_implementation": "CPython",
        "python_executable_sha256": "e" * 64,
        "python_cache_tag": "cpython-310",
        "python_soabi": "cpython-310-x86_64-linux-gnu",
        "platform_system": "Linux",
        "platform_release": "test",
        "platform_machine": "x86_64",
        "torch_version": "2.10.0+cu128",
        "numpy_version": "2.2.6",
        "cuda_runtime_version": "12.8",
        "cudnn_version": 91002,
        "environment_fingerprint_schema_version": 2,
        "installed_distributions_sha256": "d" * 64,
        "formal_runtime_distributions": [
            {
                "name": name,
                "version": "1.0",
                "metadata_sha256": "a" * 64,
                "record_sha256": "b" * 64,
                "wheel_sha256": "c" * 64,
                "module": _FORMAL_RUNTIME_MODULES[name],
                "module_version": "1.0",
                "module_origin_relative_path": f"{name}/__init__.py",
                "module_origin_sha256": "f" * 64,
                "record_verified_file_count": 1,
                "record_verified_total_bytes": 1,
                "record_verified_files_sha256": "9" * 64,
                "record_pyc_mismatch_count": 0,
            }
            for name in _FORMAL_RUNTIME_DISTRIBUTIONS
        ],
        "unavailable_formal_runtime_distributions": [],
        "flash_attn3_available": True,
        "nccl_version": [2, 27, 5],
        "visible_cuda_device_count": visible_cuda_device_count,
    }


def test_generator_derives_only_cuda_count_and_binds_private_preimage():
    generator = _load(
        "formal_environment_generator_test",
        "scripts/generate_formal_environment.py",
    )
    preimage = {
        "visible_distribution_multiset": [
            {
                "name": "example",
                "version": "1.0",
                "metadata_sha256": "1" * 64,
                "record_sha256": "2" * 64,
                "wheel_sha256": "3" * 64,
            }
        ],
        "effective_formal_runtime_distributions": [],
    }
    payload = _environment_payload(visible_cuda_device_count=4)
    payload["installed_distributions_sha256"] = canonical_sha256(preimage)
    captured = make_manifest("environment", payload)

    one, two, inventory = generator._build_artifacts(
        captured_environment=captured,
        fingerprint_preimage=preimage,
        source_commit_sha="1" * 40,
        source_tree_sha="2" * 40,
        git_sha256="3" * 64,
        make_manifest=make_manifest,
        canonical_sha256=canonical_sha256,
        validate_environment_payload=validate_environment_payload,
    )

    one_payload = copy.deepcopy(one["payload"])
    two_payload = copy.deepcopy(two["payload"])
    assert one_payload.pop("visible_cuda_device_count") == 1
    assert two_payload.pop("visible_cuda_device_count") == 2
    assert one_payload == two_payload
    assert one["sha256"] != two["sha256"]
    private = inventory["payload"]
    assert private["capture_visible_cuda_device_count"] == 4
    assert private["fingerprint_preimage"] == preimage
    assert private["environment_sha256_by_world_size"] == {
        "1": one["sha256"],
        "2": two["sha256"],
    }


def test_generator_rejects_inventory_preimage_mismatch():
    generator = _load(
        "formal_environment_generator_mismatch_test",
        "scripts/generate_formal_environment.py",
    )
    captured = make_manifest(
        "environment", _environment_payload(visible_cuda_device_count=1)
    )
    with pytest.raises(ValueError, match="preimage"):
        generator._build_artifacts(
            captured_environment=captured,
            fingerprint_preimage={"different": True},
            source_commit_sha="1" * 40,
            source_tree_sha="2" * 40,
            git_sha256="3" * 64,
            make_manifest=make_manifest,
            canonical_sha256=canonical_sha256,
            validate_environment_payload=validate_environment_payload,
        )


def test_verifier_checks_digest_and_gpu_count(monkeypatch):
    verifier = _load(
        "formal_environment_verifier_test",
        "scripts/verify_formal_environment.py",
    )
    payload = _environment_payload(visible_cuda_device_count=1)
    environment = make_manifest("environment", payload)
    from tabicl.train import _provenance as provenance

    monkeypatch.setattr(
        verifier.sys,
        "flags",
        SimpleNamespace(isolated=1),
    )
    monkeypatch.setattr(verifier.sys, "dont_write_bytecode", True)
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setattr(
        provenance,
        "runtime_environment_manifest",
        lambda *, require_formal_runtime: environment,
    )
    monkeypatch.setattr(verifier, "_import_exact_provenance", lambda _root: provenance)

    arguments = [
        "--exact-root",
        str(ROOT),
        "--expected-sha256",
        environment["sha256"],
        "--expected-gpus",
        "1",
    ]
    assert verifier.main(arguments) == 0
    with pytest.raises(ValueError, match="visible CUDA"):
        verifier.main([*arguments[:-1], "2"])
    with pytest.raises(ValueError, match="precommitted digest"):
        verifier.main(
            [
                *arguments[:3],
                "0" * 64,
                *arguments[4:],
            ]
        )


def test_verifier_rejects_nonisolated_python():
    verifier = _load(
        "formal_environment_verifier_nonisolated_test",
        "scripts/verify_formal_environment.py",
    )
    with pytest.raises(ValueError, match="Python -I -B"):
        verifier.main(
            [
                "--exact-root",
                str(ROOT),
                "--expected-sha256",
                "0" * 64,
                "--expected-gpus",
                "1",
            ]
        )


def test_generator_cli_recovers_transaction_after_injected_failure(tmp_path):
    root, commit, tree = _make_generator_checkout(tmp_path)
    output = tmp_path / "durable"
    output.mkdir()
    argv = _generator_argv(root, commit, tree, output)
    environment = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "TABICL_ENABLE_TEST_FAILURE_INJECTION": "1",
        "TABICL_TEST_FAIL_ENVIRONMENT_PUBLICATION_AT": "after-one-gpu",
    }

    interrupted = _run(*argv, env=environment)
    assert interrupted.returncode == 1
    assert "injected formal environment publication failure" in interrupted.stderr
    assert (output / "one.json").is_file()
    assert not (output / "two.json").exists()
    assert not (output / "inventory.json").exists()
    assert not (output / "complete.json").exists()
    assert len(list(output.glob(".formal-environment-*.stage"))) == 3

    recovered = _run(*argv, env={**os.environ, "PYTHONNOUSERSITE": "1"})
    assert recovered.returncode == 0, recovered.stderr
    assert recovered.stderr == ""
    report = json.loads(recovered.stdout)
    for name in ("one.json", "two.json", "inventory.json", "complete.json"):
        assert (output / name).is_file()
    assert not list(output.glob(".formal-environment-*.stage"))
    completion = json.loads((output / "complete.json").read_text())
    assert completion["kind"] == "formal_environment_generation_completion"
    assert completion["transaction_sha256"] == report["transaction_sha256"]
    before = {
        name: (output / name).stat().st_ino
        for name in ("one.json", "two.json", "inventory.json", "complete.json")
    }

    repeated = _run(*argv, env={**os.environ, "PYTHONNOUSERSITE": "1"})
    assert repeated.returncode == 0, repeated.stderr
    assert before == {
        name: (output / name).stat().st_ino
        for name in ("one.json", "two.json", "inventory.json", "complete.json")
    }

    (output / "two.json").unlink()
    corrupted = _run(*argv, env={**os.environ, "PYTHONNOUSERSITE": "1"})
    assert corrupted.returncode == 1
    assert "missing or is a symlink" in corrupted.stderr
    assert not (output / "two.json").exists()


def test_generator_cli_rejects_outputs_inside_exact_checkout(tmp_path):
    root, commit, tree = _make_generator_checkout(tmp_path)
    output = root / "generated"
    output.mkdir()
    completed = _run(
        *_generator_argv(root, commit, tree, output),
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    assert completed.returncode == 1
    assert "outside exact T" in completed.stderr
    assert not list(output.iterdir())


@pytest.mark.parametrize(
    ("option", "basename"),
    [
        ("--completion-output", ".complete.json"),
        ("--one-gpu-output", "one gpu.json"),
    ],
)
def test_generator_rejects_output_names_the_transaction_verifier_cannot_read(
    tmp_path, option, basename
):
    root, commit, tree = _make_generator_checkout(tmp_path)
    output = tmp_path / "durable"
    output.mkdir()
    argv = _generator_argv(root, commit, tree, output)
    argv[argv.index(option) + 1] = str(output / basename)

    completed = _run(
        *argv,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )

    assert completed.returncode == 1
    assert "output basename is not a safe sibling" in completed.stderr
    assert not list(output.iterdir())


def test_generator_rejects_replaced_git_path(tmp_path):
    generator = _load(
        "formal_environment_generator_git_swap_test",
        "scripts/generate_formal_environment.py",
    )
    root, commit, tree = _make_generator_checkout(tmp_path)
    trusted_git = tmp_path / "trusted-git"
    shutil.copy2(Path(shutil.which("git") or "").resolve(strict=True), trusted_git)
    digest = hashlib.sha256(trusted_git.read_bytes()).hexdigest()
    git_fd = generator._open_trusted_git(trusted_git, expected_sha256=digest)
    replacement = tmp_path / "replacement-git"
    replacement.write_bytes(b"#!/bin/sh\nexit 99\n")
    replacement.chmod(0o755)
    os.replace(replacement, trusted_git)
    try:
        with pytest.raises(ValueError, match="path changed"):
            generator._attest_checkout(
                root=root,
                git_fd=git_fd,
                git_path=trusted_git,
                expected_commit=commit,
                expected_tree=tree,
            )
    finally:
        os.close(git_fd)


def _make_verifier_root(tmp_path: Path) -> Path:
    root = tmp_path / "verifier-root"
    (root / "scripts").mkdir(parents=True)
    package = root / "src/tabicl/train"
    package.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/verify_formal_environment.py", root / "scripts")
    (root / "src/tabicl/__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "_provenance.py").write_text(
        "def runtime_environment_manifest(*, require_formal_runtime):\n"
        "    return {'payload': {'visible_cuda_device_count': 1}, "
        "'sha256': 'a' * 64}\n"
    )
    return root


def test_verifier_real_cli_accepts_exact_module_and_rejects_preloaded_pollution(
    tmp_path,
):
    root = _make_verifier_root(tmp_path)
    script = root / "scripts/verify_formal_environment.py"
    arguments = [
        "--exact-root",
        str(root),
        "--expected-sha256",
        "a" * 64,
        "--expected-gpus",
        "1",
    ]
    environment = {**os.environ, "PYTHONNOUSERSITE": "1"}
    exact = _run(sys.executable, "-I", "-B", str(script), *arguments, env=environment)
    assert exact.returncode == 0, exact.stderr
    assert exact.stdout == "" and exact.stderr == ""

    contaminant = tmp_path / "contaminant.py"
    contaminant.write_text("CONTAMINANT = True\n")
    launcher = textwrap.dedent(
        f"""
        import runpy
        import sys
        import types
        tabicl = types.ModuleType("tabicl")
        tabicl.__path__ = []
        train = types.ModuleType("tabicl.train")
        train.__path__ = []
        provenance = types.ModuleType("tabicl.train._provenance")
        provenance.__file__ = {str(contaminant)!r}
        sys.modules["tabicl"] = tabicl
        sys.modules["tabicl.train"] = train
        sys.modules["tabicl.train._provenance"] = provenance
        sys.argv = [{str(script)!r}, *{arguments!r}]
        runpy.run_path({str(script)!r}, run_name="__main__")
        """
    )
    polluted = _run(
        sys.executable, "-I", "-B", "-c", launcher, env=environment
    )
    assert polluted.returncode == 1
    assert "imported before exact-T isolation" in polluted.stderr


def test_environment_transaction_verifier_accepts_complete_bound_set(tmp_path):
    _output, argv, expected = _generate_fake_transaction(tmp_path)

    completed = _run(*argv, env={**os.environ, "PYTHONNOUSERSITE": "1"})

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    summary = json.loads(completed.stdout)
    assert summary["kind"] == "formal_environment_transaction_verification"
    assert summary["completion_raw_sha256"] == expected["completion_raw_sha256"]
    assert summary["transaction_sha256"] == expected["transaction_sha256"]
    assert summary["one_gpu_environment_sha256"] == expected[
        "one_gpu_environment_sha256"
    ]
    assert summary["two_gpu_environment_sha256"] == expected[
        "two_gpu_environment_sha256"
    ]
    assert summary["inventory_sha256"] == expected["inventory_sha256"]


def test_environment_transaction_verifier_rejects_missing_marker(tmp_path):
    output, argv, _expected = _generate_fake_transaction(tmp_path)
    for name in ("one.json", "two.json", "inventory.json", "complete.json"):
        (output / name).unlink()

    completed = _run(*argv, env={**os.environ, "PYTHONNOUSERSITE": "1"})

    assert completed.returncode == 1
    assert "completion marker is missing" in completed.stderr


def test_environment_transaction_verifier_rejects_markerless_finals(tmp_path):
    output, argv, _expected = _generate_fake_transaction(tmp_path)
    (output / "complete.json").unlink()
    assert all(
        (output / name).is_file()
        for name in ("one.json", "two.json", "inventory.json")
    )

    completed = _run(*argv, env={**os.environ, "PYTHONNOUSERSITE": "1"})

    assert completed.returncode == 1
    assert "completion marker is missing" in completed.stderr


def test_environment_transaction_verifier_rejects_recommitted_marker_tamper(
    tmp_path,
):
    output, _argv, _expected = _generate_fake_transaction(tmp_path)
    marker_path = output / "complete.json"
    marker = json.loads(marker_path.read_text())
    marker["transaction_sha256"] = "0" * 64
    marker_path.unlink()
    marker_path.write_bytes(
        json.dumps(
            marker,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        + b"\n"
    )
    argv, _expected = _verify_transaction_argv(output)

    completed = _run(*argv, env={**os.environ, "PYTHONNOUSERSITE": "1"})

    assert completed.returncode == 1
    assert "descriptor SHA-256 mismatch" in completed.stderr


def test_environment_transaction_verifier_rejects_output_byte_tamper(tmp_path):
    output, argv, _expected = _generate_fake_transaction(tmp_path)
    one_path = output / "one.json"
    raw = bytearray(one_path.read_bytes())
    raw[0] = ord("[")
    one_path.unlink()
    one_path.write_bytes(raw)

    completed = _run(*argv, env={**os.environ, "PYTHONNOUSERSITE": "1"})

    assert completed.returncode == 1
    assert "differs from marker" in completed.stderr


def test_environment_transaction_verifier_rejects_output_symlink(tmp_path):
    output, argv, _expected = _generate_fake_transaction(tmp_path)
    one_path = output / "one.json"
    relocated = tmp_path / "relocated-one.json"
    relocated.write_bytes(one_path.read_bytes())
    one_path.unlink()
    one_path.symlink_to(relocated)

    completed = _run(*argv, env={**os.environ, "PYTHONNOUSERSITE": "1"})

    assert completed.returncode == 1
    assert "missing or is a symlink" in completed.stderr


def test_environment_transaction_verifier_rejects_external_source_mismatch(
    tmp_path,
):
    output, _argv, _expected = _generate_fake_transaction(tmp_path)
    argv, _expected = _verify_transaction_argv(output, expected_commit="f" * 40)

    completed = _run(*argv, env={**os.environ, "PYTHONNOUSERSITE": "1"})

    assert completed.returncode == 1
    assert "source commit/tree mismatch" in completed.stderr
