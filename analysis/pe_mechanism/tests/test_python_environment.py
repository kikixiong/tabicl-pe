from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import venv

import pytest

from pe_mechanism import python_environment
from pe_mechanism.python_environment import build_python_environment_contract


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
VERIFIER = PACKAGE_ROOT / "scripts" / "verify_python_environment.py"


def _venv(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(root)
    entry = root / "bin" / "python"
    if not entry.is_symlink():
        pytest.skip("strict contract requires the platform's standard symlink venv")
    site = (
        root
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    dist = site / "autogluon.tabular-99.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: autogluon.tabular\nVersion: 99.0\n",
        encoding="utf-8",
    )
    (dist / "direct_url.json").write_text(
        '{"url":"https://example.invalid/unit"}\n', encoding="utf-8"
    )
    (dist / "RECORD").write_text("METADATA,,\ndirect_url.json,,\n", encoding="utf-8")
    return entry, dist


def _write_contract(tmp_path: Path, contract: dict[str, object]) -> tuple[Path, str]:
    raw = (
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    path = tmp_path / "python-contract.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _verify(
    entry: Path, contract: Path, file_sha: str
) -> subprocess.CompletedProcess[str]:
    document_sha = json.loads(contract.read_text(encoding="utf-8"))["sha256"]
    return subprocess.run(
        [
            str(entry),
            "-I",
            "-B",
            str(VERIFIER),
            "--entry",
            str(entry),
            "--contract",
            str(contract),
            "--expected-file-sha256",
            file_sha,
            "--expected-document-sha256",
            document_sha,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_symlink_venv_entry_retains_prefix_and_has_path_redacted_contract(
    tmp_path: Path,
) -> None:
    entry, _ = _venv(tmp_path)
    probe = subprocess.run(
        [
            str(entry),
            "-I",
            "-c",
            "import json,sys; print(json.dumps([sys.executable,sys.prefix,sys.base_prefix]))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    executable, prefix, base_prefix = json.loads(probe.stdout)
    assert executable == str(entry)
    assert prefix == str(entry.parent.parent)
    assert prefix != base_prefix

    contract = build_python_environment_contract(
        entry, required_distributions=("autogluon.tabular",)
    )
    path, file_sha = _write_contract(tmp_path, contract)
    assert _verify(entry, path, file_sha).returncode == 0
    assert str(tmp_path) not in json.dumps(contract, sort_keys=True)
    assert contract["payload"]["venv_prefix_verified"] is True


def test_python_contract_rejects_final_target_and_config_tamper(tmp_path: Path) -> None:
    entry, _ = _venv(tmp_path)
    contract = build_python_environment_contract(entry)
    path, file_sha = _write_contract(tmp_path, contract)

    original_target = os.readlink(entry)
    real_target = entry.resolve(strict=True)
    entry.unlink()
    replacement = str(real_target)
    if replacement == original_target:
        replacement = os.path.relpath(real_target, entry.parent)
    entry.symlink_to(replacement)
    assert _verify(entry, path, file_sha).returncode != 0

    entry.unlink()
    entry.symlink_to(original_target)
    config = entry.parent.parent / "pyvenv.cfg"
    config.write_text(
        config.read_text(encoding="utf-8") + "# tampered\n", encoding="utf-8"
    )
    assert _verify(entry, path, file_sha).returncode != 0


def test_python_contract_rejects_real_executable_content_tamper(tmp_path: Path) -> None:
    entry, _ = _venv(tmp_path)
    private_target = tmp_path / "bound-python"
    shutil.copy2(entry.resolve(strict=True), private_target)
    entry.unlink()
    entry.symlink_to(private_target)
    contract = build_python_environment_contract(entry)
    path, file_sha = _write_contract(tmp_path, contract)
    with private_target.open("ab") as handle:
        handle.write(b"\ncontent-tamper\n")
    assert _verify(entry, path, file_sha).returncode != 0


@pytest.mark.parametrize("metadata_name", ["RECORD", "direct_url.json"])
def test_python_contract_rejects_installed_environment_tamper(
    tmp_path: Path, metadata_name: str
) -> None:
    entry, dist = _venv(tmp_path)
    contract = build_python_environment_contract(
        entry, required_distributions=("autogluon.tabular",)
    )
    path, file_sha = _write_contract(tmp_path, contract)
    target = dist / metadata_name
    target.write_text(
        target.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8"
    )
    assert _verify(entry, path, file_sha).returncode != 0


def test_python_contract_requires_named_distribution(tmp_path: Path) -> None:
    entry, _ = _venv(tmp_path)
    with pytest.raises(ValueError, match="required installed distribution"):
        build_python_environment_contract(
            entry, required_distributions=("definitely.missing",)
        )


def test_bound_launcher_preserves_venv_and_postverifies(tmp_path: Path) -> None:
    entry, _ = _venv(tmp_path)
    contract = build_python_environment_contract(entry)
    path, file_sha = _write_contract(tmp_path, contract)
    command = [
        str(entry),
        "-I",
        "-B",
        str(VERIFIER),
        "--entry",
        str(entry),
        "--contract",
        str(path),
        "--expected-file-sha256",
        file_sha,
        "--expected-document-sha256",
        contract["sha256"],
        "--",
        "-c",
        "import sys; raise SystemExit(0 if sys.executable==sys.prefix+'/bin/python' else 3)",
    ]
    assert subprocess.run(command, check=False).returncode == 0
    tamper = [
        *command[:-2],
        "-c",
        f"from pathlib import Path; Path({str(path)!r}).write_text('tampered')",
    ]
    assert subprocess.run(tamper, check=False, capture_output=True).returncode != 0


def _fake_nvidia_smi(tmp_path: Path, *, fail_after_first: bool = False) -> Path:
    executable = tmp_path / "nvidia-smi"
    failure_guard = ""
    if fail_after_first:
        counter = tmp_path / "nvidia-smi-called"
        failure_guard = (
            f"if [ -e {str(counter)!r} ]; then exit 23; fi\n"
            f": > {str(counter)!r}\n"
        )
    executable.write_text(
        "#!/bin/sh\n"
        + failure_guard
        + "printf '0, NVIDIA H100 80GB HBM3, 87, 1024, 81559\\n'\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable


def _start_gpu_monitor(
    *,
    entry: Path,
    contract_path: Path,
    contract_file_sha256: str,
    contract_document_sha256: str,
    output_csv: Path,
    nvidia_smi: Path,
) -> subprocess.Popen[str]:
    loader = (
        "import importlib.util;"
        f"s=importlib.util.spec_from_file_location('bound_verifier',{str(VERIFIER)!r});"
        "m=importlib.util.module_from_spec(s);"
        "s.loader.exec_module(m);"
        f"m.NVIDIA_SMI={str(nvidia_smi)!r};"
        "raise SystemExit(m.main())"
    )
    return subprocess.Popen(
        [
            str(entry),
            "-I",
            "-B",
            "-c",
            loader,
            "--entry",
            str(entry),
            "--contract",
            str(contract_path),
            "--expected-file-sha256",
            contract_file_sha256,
            "--expected-document-sha256",
            contract_document_sha256,
            "--gpu-monitor",
            str(output_csv),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def _protocol_line(process: subprocess.Popen[str], *, timeout: float = 10) -> str:
    assert process.stdout is not None
    readable, _, _ = select.select((process.stdout,), (), (), timeout)
    assert readable, "timed out waiting for GPU monitor protocol output"
    return process.stdout.readline().rstrip("\n")


def _send_control(process: subprocess.Popen[str], control: str | None) -> None:
    assert process.stdin is not None
    if control is not None:
        process.stdin.write(control)
        process.stdin.flush()
    process.stdin.close()
    process.stdin = None


def test_in_process_gpu_monitor_ready_stop_complete_and_no_reopen_after_ready(
    tmp_path: Path,
) -> None:
    entry, _ = _venv(tmp_path)
    contract = build_python_environment_contract(entry)
    path, file_sha = _write_contract(tmp_path, contract)
    output_csv = tmp_path / "gpu.csv"
    process = _start_gpu_monitor(
        entry=entry,
        contract_path=path,
        contract_file_sha256=file_sha,
        contract_document_sha256=contract["sha256"],
        output_csv=output_csv,
        nvidia_smi=_fake_nvidia_smi(tmp_path),
    )
    replacement = tmp_path / "replacement-python"
    replacement.write_bytes(b"replacement must never be executed")
    try:
        assert _protocol_line(process) == "READY"
        entry.unlink()
        entry.symlink_to(replacement)
        config = entry.parent.parent / "pyvenv.cfg"
        config.write_text(
            config.read_text(encoding="utf-8") + "# post-ready tamper\n",
            encoding="utf-8",
        )
        _send_control(process, "STOP\n")
        assert _protocol_line(process) == "COMPLETE"
        assert process.wait(timeout=10) == 0
        lines = output_csv.read_text(encoding="utf-8").splitlines()
        assert lines[0] == (
            "timestamp,index,name,utilization_gpu_pct,memory_used_mib,"
            "memory_total_mib"
        )
        assert len(lines) == 3
        assert all(
            line.endswith(",0, NVIDIA H100 80GB HBM3, 87, 1024, 81559")
            for line in lines[1:]
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def test_gpu_monitor_refuses_existing_csv_without_overwrite(tmp_path: Path) -> None:
    entry, _ = _venv(tmp_path)
    contract = build_python_environment_contract(entry)
    path, file_sha = _write_contract(tmp_path, contract)
    output_csv = tmp_path / "gpu.csv"
    output_csv.write_text("do-not-overwrite\n", encoding="utf-8")
    process = _start_gpu_monitor(
        entry=entry,
        contract_path=path,
        contract_file_sha256=file_sha,
        contract_document_sha256=contract["sha256"],
        output_csv=output_csv,
        nvidia_smi=_fake_nvidia_smi(tmp_path),
    )
    stdout, _stderr = process.communicate(timeout=10)
    assert process.returncode != 0
    assert "READY" not in stdout
    assert output_csv.read_text(encoding="utf-8") == "do-not-overwrite\n"


@pytest.mark.parametrize("control", [None, "NOT_STOP\n"])
def test_gpu_monitor_rejects_stdin_eof_and_non_stop(
    tmp_path: Path, control: str | None
) -> None:
    entry, _ = _venv(tmp_path)
    contract = build_python_environment_contract(entry)
    path, file_sha = _write_contract(tmp_path, contract)
    process = _start_gpu_monitor(
        entry=entry,
        contract_path=path,
        contract_file_sha256=file_sha,
        contract_document_sha256=contract["sha256"],
        output_csv=tmp_path / "gpu.csv",
        nvidia_smi=_fake_nvidia_smi(tmp_path),
    )
    assert _protocol_line(process) == "READY"
    _send_control(process, control)
    assert process.wait(timeout=10) != 0
    assert _protocol_line(process, timeout=1) == ""


def test_gpu_monitor_final_sample_failure_has_no_complete_and_exits_nonzero(
    tmp_path: Path,
) -> None:
    entry, _ = _venv(tmp_path)
    contract = build_python_environment_contract(entry)
    path, file_sha = _write_contract(tmp_path, contract)
    process = _start_gpu_monitor(
        entry=entry,
        contract_path=path,
        contract_file_sha256=file_sha,
        contract_document_sha256=contract["sha256"],
        output_csv=tmp_path / "gpu.csv",
        nvidia_smi=_fake_nvidia_smi(tmp_path, fail_after_first=True),
    )
    assert _protocol_line(process) == "READY"
    _send_control(process, "STOP\n")
    assert process.wait(timeout=10) != 0
    assert _protocol_line(process, timeout=1) == ""


def test_contract_path_replacement_during_single_fd_read_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = (tmp_path / "contract.json").absolute()
    path.write_bytes(b'{"old":true}')
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b'{"new":true}')
    original_read = python_environment._read_fd_bytes
    replaced = False

    def replace_after_open(descriptor: int) -> bytes:
        nonlocal replaced
        if not replaced and os.readlink(f"/proc/self/fd/{descriptor}") == str(path):
            replaced = True
            path.rename(tmp_path / "old.json")
            replacement.rename(path)
        return original_read(descriptor)

    monkeypatch.setattr(python_environment, "_read_fd_bytes", replace_after_open)
    with pytest.raises(RuntimeError, match="changed while it was read"):
        python_environment._read_regular_path_nofollow(path, label="unit contract")


def test_build_rejects_entry_retargeted_during_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry, _ = _venv(tmp_path)
    original_run = python_environment.subprocess.run
    original_target = os.readlink(entry)
    real_target = entry.resolve(strict=True)

    def run_then_retarget(*args, **kwargs):
        completed = original_run(*args, **kwargs)
        entry.unlink()
        replacement = str(real_target)
        if replacement == original_target:
            replacement = os.path.relpath(real_target, entry.parent)
        entry.symlink_to(replacement)
        return completed

    monkeypatch.setattr(python_environment.subprocess, "run", run_then_retarget)
    with pytest.raises(RuntimeError, match="changed during environment probing"):
        build_python_environment_contract(entry)


def test_proc_binding_is_local_but_frozen_payload_is_cross_mount_portable(
    tmp_path: Path,
) -> None:
    entry, _ = _venv(tmp_path)
    facts, _, venv_root = python_environment._entry_facts(entry)
    completed = subprocess.run(
        [
            str(entry),
            "-I",
            str(Path(python_environment.__file__).resolve()),
            "--probe-current",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    probe = json.loads(completed.stdout)
    payload = python_environment._payload_from_probe(
        entry=entry,
        facts=facts,
        venv_root=venv_root,
        probe=probe,
        required_distributions=(),
    )
    changed_facts = dict(facts)
    changed_image = dict(facts["_process_image"])
    changed_image["device"] += 1000
    changed_image["inode"] += 1000
    changed_facts["_process_image"] = changed_image
    changed_probe = dict(probe)
    changed_probe["proc_self_exe"] = changed_image
    portable = python_environment._payload_from_probe(
        entry=entry,
        facts=changed_facts,
        venv_root=venv_root,
        probe=changed_probe,
        required_distributions=(),
    )
    assert portable == payload
    changed_probe["proc_self_exe"] = {**changed_image, "inode": 0}
    with pytest.raises(ValueError, match="/proc/self/exe"):
        python_environment._payload_from_probe(
            entry=entry,
            facts=changed_facts,
            venv_root=venv_root,
            probe=changed_probe,
            required_distributions=(),
        )
