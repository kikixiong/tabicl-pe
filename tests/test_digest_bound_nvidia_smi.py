from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/exec_digest_bound_nvidia_smi.py"
DURABLE = Path(__file__).parents[1] / "scripts/run_with_durable_log.sh"


def _load():
    spec = importlib.util.spec_from_file_location("digest_bound_nvidia_smi", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _executable(path: Path, text: str) -> str:
    path.write_text("#!/bin/sh\n" + text)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("mutation", ["delete", "swap"])
def test_verified_descriptor_survives_path_delete_or_swap(tmp_path: Path, mutation: str):
    module = _load()
    executable = tmp_path / "nvidia-smi"
    digest = _executable(executable, "printf 'verified-inode\\n'\n")
    fd = module._open_verified(executable, digest)
    try:
        if mutation == "delete":
            executable.unlink()
        else:
            replacement = tmp_path / "replacement"
            _executable(replacement, "printf 'replacement-path\\n'\n")
            replacement.replace(executable)
        completed = subprocess.run(
            [f"/proc/self/fd/{fd}"],
            check=True,
            capture_output=True,
            text=True,
            pass_fds=(fd,),
        )
        assert completed.stdout == "verified-inode\n"
    finally:
        os.close(fd)


def test_digest_mismatch_is_rejected(tmp_path: Path):
    executable = tmp_path / "nvidia-smi"
    _executable(executable, "exit 0\n")
    with pytest.raises(ValueError, match="SHA-256"):
        _load()._open_verified(executable, "0" * 64)


def test_fifo_is_rejected_without_blocking(tmp_path: Path):
    fifo = tmp_path / "nvidia-smi"
    os.mkfifo(fifo, stat.S_IRUSR | stat.S_IWUSR)
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(SCRIPT),
            "--nvidia-smi",
            str(fifo),
            "--expected-sha256",
            "0" * 64,
            "--",
            "/usr/bin/true",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 2
    assert "stable bounded executable" in completed.stderr


def test_allocation_scoped_query_has_a_fixed_wall_clock_timeout(tmp_path: Path):
    module = _load()
    executable = tmp_path / "nvidia-smi"
    digest = _executable(executable, "sleep 60\n")
    fd = module._open_verified(executable, digest)
    module.QUERY_TIMEOUT_SECONDS = 0.05
    started = time.monotonic()
    try:
        with pytest.raises(ValueError, match="timed out"):
            module._bounded_query(fd, "0", "uuid")
    finally:
        os.close(fd)
    assert time.monotonic() - started < 2


def test_allocation_scoped_query_is_bounded_and_uses_the_retained_inode(
    tmp_path: Path, capsys
):
    module = _load()
    executable = tmp_path / "nvidia-smi"
    digest = _executable(executable, "printf 'GPU-fixture\\n'\n")
    fd = module._open_verified(executable, digest)
    try:
        assert module._bounded_query(fd, "GPU-fixture", "uuid") == 0
    finally:
        os.close(fd)
    assert capsys.readouterr().out == "GPU-fixture\n"


def test_durable_logger_compute_reacquires_retained_descriptor_from_live_owner(
    tmp_path: Path,
):
    nvidia_smi = tmp_path / "nvidia-smi"
    digest = _executable(nvidia_smi, "printf 'verified-through-durable\\n'\n")
    result_path = tmp_path / "result"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import os, subprocess, sys\n"
        "fd = int(os.environ['FORMAL_NVIDIA_SMI_FD'])\n"
        "result = subprocess.run(\n"
        "    [os.environ['NVIDIA_SMI']], check=True, capture_output=True,\n"
        "    text=True, pass_fds=(fd,),\n"
        ")\n"
        "open(sys.argv[1], 'w').write(result.stdout)\n"
    )
    owner = tmp_path / "owner.py"
    owner.write_text(
        "import os, subprocess, sys\n"
        "fd = os.environ['FORMAL_NVIDIA_SMI_FD']\n"
        "owner = os.environ['FORMAL_NVIDIA_SMI_FD_OWNER_PID']\n"
        "os.unlink(sys.argv[3])\n"
        "subprocess.run([\n"
        "    os.environ['DURABLE'], sys.argv[1], '1000000',\n"
        "    sys.executable, '-I', '-B', os.environ['LAUNCHER'],\n"
        "    '--retained-fd-owner-pid', owner, '--retained-fd', fd,\n"
        "    '--expected-sha256', os.environ['FORMAL_NVIDIA_SMI_SHA256'], '--',\n"
        "    sys.executable, '-I', '-B', sys.argv[2], sys.argv[4],\n"
        "], check=True)\n"
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHON": sys.executable,
            "DURABLE": str(DURABLE),
            "LAUNCHER": str(SCRIPT),
            "FORMAL_NVIDIA_SMI_SHA256": digest,
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(SCRIPT),
            "--nvidia-smi",
            str(nvidia_smi),
            "--expected-sha256",
            digest,
            "--",
            sys.executable,
            "-I",
            "-B",
            str(owner),
            str(tmp_path / "durable.log"),
            str(worker),
            str(nvidia_smi),
            str(result_path),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert result_path.read_text() == "verified-through-durable\n"


def test_durable_logger_then_torchrun_ranks_reacquire_at_both_fd_boundaries(
    tmp_path: Path,
):
    pytest.importorskip("torch")
    nvidia_smi = tmp_path / "nvidia-smi"
    digest = _executable(nvidia_smi, "printf 'verified-through-torchrun\\n'\n")
    rank_worker = tmp_path / "rank-worker.sh"
    rank_worker.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "rank=${LOCAL_RANK:?}\n"
        "if [[ $FORMAL_NVIDIA_SMI_FD_OWNER_PID != $BASHPID ]]; then\n"
        "  printf 'closed-by-torchrun\\n' > \"$RESULT_ROOT/$rank.before\"\n"
        "  exec \"$PYTHON\" -I -B \"$LAUNCHER\" "
        "--retained-fd-owner-pid \"$FORMAL_NVIDIA_SMI_FD_OWNER_PID\" "
        "--retained-fd \"$FORMAL_NVIDIA_SMI_FD\" "
        "--expected-sha256 \"$FORMAL_NVIDIA_SMI_SHA256\" -- \"$0\"\n"
        "fi\n"
        "[[ $NVIDIA_SMI == /proc/self/fd/$FORMAL_NVIDIA_SMI_FD ]]\n"
        "\"$PYTHON\" -I -B - \"$RESULT_ROOT/$rank.after\" <<'PY'\n"
        "import os\n"
        "from pathlib import Path\n"
        "import subprocess\n"
        "fd = int(os.environ['FORMAL_NVIDIA_SMI_FD'])\n"
        "result = subprocess.run(\n"
        "    [os.environ['NVIDIA_SMI']],\n"
        "    check=True, capture_output=True, text=True, pass_fds=(fd,),\n"
        ")\n"
        "Path(__import__('sys').argv[1]).write_text(result.stdout)\n"
        "PY\n"
    )
    rank_worker.chmod(0o700)
    owner = tmp_path / "owner.py"
    owner.write_text(
        "import os, subprocess, sys\n"
        "fd = int(os.environ['FORMAL_NVIDIA_SMI_FD'])\n"
        "assert os.environ['FORMAL_NVIDIA_SMI_FD_OWNER_PID'] == str(os.getpid())\n"
        "os.unlink(sys.argv[2])\n"
        "subprocess.run([\n"
        "    os.environ['DURABLE'], os.environ['DURABLE_LOG'], '1000000',\n"
        "    sys.executable, '-I', '-B', os.environ['LAUNCHER'],\n"
        "    '--retained-fd-owner-pid', str(os.getpid()),\n"
        "    '--retained-fd', str(fd), '--expected-sha256',\n"
        "    os.environ['FORMAL_NVIDIA_SMI_SHA256'], '--',\n"
        "    sys.executable, '-I', '-B', '-m', 'torch.distributed.run',\n"
        "    '--standalone', '--nproc_per_node=2', '--no-python', sys.argv[1],\n"
        "], check=True)\n"
        "assert os.fstat(fd).st_ino\n"
    )
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHON": sys.executable,
            "DURABLE": str(DURABLE),
            "DURABLE_LOG": str(tmp_path / "durable.log"),
            "LAUNCHER": str(SCRIPT),
            "RESULT_ROOT": str(tmp_path),
            "FORMAL_NVIDIA_SMI_SHA256": digest,
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(SCRIPT),
            "--nvidia-smi",
            str(nvidia_smi),
            "--expected-sha256",
            digest,
            "--",
            sys.executable,
            "-I",
            "-B",
            str(owner),
            str(rank_worker),
            str(nvidia_smi),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    for rank in range(2):
        assert (tmp_path / f"{rank}.before").read_text() == "closed-by-torchrun\n"
        assert (tmp_path / f"{rank}.after").read_text() == "verified-through-torchrun\n"
