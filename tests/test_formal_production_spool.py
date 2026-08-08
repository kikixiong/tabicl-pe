from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

import pytest


SPOOL_SOURCE = Path(__file__).parents[1] / "scripts" / "slurm_h100_identity_formal.sh"


def _git(*args, stdout=False):
    return subprocess.run(
        ["/usr/bin/git", *map(str, args)],
        check=True,
        text=True,
        stdout=subprocess.PIPE if stdout else subprocess.DEVNULL,
    )


def _fixture(tmp_path: Path, exit_code: int):
    candidate = tmp_path / "candidate"
    scripts = candidate / "scripts"
    scripts.mkdir(parents=True)
    runner = scripts / "run_formal_identity_production_job.py"
    runner.write_text(
        "from pathlib import Path\n"
        "import os\n"
        "Path(os.environ['RUNTIME_MARKER']).write_text(str(Path(__file__).parents[1]))\n"
        "raise SystemExit(int(os.environ['RUNTIME_EXIT']))\n"
    )
    shutil.copy2(
        Path(__file__).parents[1] / "scripts/verify_filesystem_isolation.py",
        scripts / "verify_filesystem_isolation.py",
    )
    shutil.copy2(
        Path(__file__).parents[1] / "scripts/verify_git_repository.py",
        scripts / "verify_git_repository.py",
    )
    shutil.copy2(
        Path(__file__).parents[1] / "scripts/exec_digest_bound_nvidia_smi.py",
        scripts / "exec_digest_bound_nvidia_smi.py",
    )
    environment_verifier = scripts / "verify_formal_environment.py"
    environment_verifier.write_text(
        "import argparse, os, sys\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--exact-root', required=True)\n"
        "p.add_argument('--expected-sha256', required=True)\n"
        "p.add_argument('--expected-gpus', required=True, type=int)\n"
        "a=p.parse_args()\n"
        "assert sys.flags.isolated and sys.dont_write_bytecode\n"
        "assert os.environ.get('PYTHONNOUSERSITE') == '1'\n"
        "assert a.expected_gpus == 1 and len(a.expected_sha256) == 64\n"
    )
    environment_verifier.chmod(0o755)
    shutil.copy2(SPOOL_SOURCE, scripts / "slurm_h100_identity_formal.sh")
    _git("init", "-q", candidate)
    _git("-C", candidate, "config", "user.name", "fixture")
    _git("-C", candidate, "config", "user.email", "fixture@example.invalid")
    _git("-C", candidate, "add", "scripts")
    _git("-C", candidate, "commit", "-q", "-m", "fixture")
    commit = _git("-C", candidate, "rev-parse", "HEAD", stdout=True).stdout.strip()
    tree = _git("-C", candidate, "rev-parse", "HEAD^{tree}", stdout=True).stdout.strip()
    _git("-C", candidate, "checkout", "-q", "--detach", commit)

    spool_dir = tmp_path / "slurm-spool"
    spool_dir.mkdir()
    spool = spool_dir / "job-script"
    shutil.copy2(SPOOL_SOURCE, spool)
    spool.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    work_root = None
    for candidate_parent in (Path("/dev/shm"), Path("/tmp"), SPOOL_SOURCE.parent):
        if (
            candidate_parent.is_dir()
            and os.access(candidate_parent, os.W_OK | os.X_OK)
            and candidate_parent.stat().st_dev != artifact_root.stat().st_dev
        ):
            work_root = Path(
                tempfile.mkdtemp(prefix="tabicl-formal-job-work-", dir=candidate_parent)
            )
            break
    if work_root is None:
        pytest.skip("test host has no writable second filesystem")
    marker = tmp_path / "runtime-marker"
    nvidia_args = tmp_path / "nvidia-args"
    nvidia = tmp_path / "nvidia-smi"
    nvidia.write_text(
        "#!/bin/bash\n"
        f"printf '%s\\n' \"$@\" > {str(nvidia_args)!r}\n"
        "printf '%s\\n' 'NVIDIA H100 80GB HBM3, GPU-fixture, 570.00'\n"
    )
    nvidia.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    fake_git = tmp_path / "git"
    fake_git.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "if [[ \"${1:-}\" == ls-remote ]]; then\n"
        f"  printf '%s\\t%s\\n' '{commit}' 'refs/heads/codex/position-identity-v1'\n"
        "else\n"
        "  args=(\"$@\")\n"
        "  is_clone=0\n"
        "  checkout_parent=''\n"
        "  for index in \"${!args[@]}\"; do\n"
        "    if [[ \"${args[$index]}\" == '-C' ]]; then\n"
        "      next_index=$((index + 1))\n"
        "      checkout_parent=\"${args[$next_index]}\"\n"
        "    elif [[ \"${args[$index]}\" == 'clone' ]]; then\n"
        "      is_clone=1\n"
        "    fi\n"
        f"    [[ \"${{args[$index]}}\" != 'https://github.com/kikixiong/tabicl-pe.git' ]] || args[$index]='{candidate}'\n"
        "  done\n"
        "  if [[ \"$is_clone\" -eq 1 ]]; then\n"
        "    [[ -n \"$checkout_parent\" ]]\n"
        "    last_index=$((${#args[@]} - 1))\n"
        "    destination=\"${args[$last_index]}\"\n"
        "    env -u GIT_ALLOW_PROTOCOL /usr/bin/git "
        "-c protocol.file.allow=always \"${args[@]}\"\n"
        "    /usr/bin/git -C \"$checkout_parent/$destination\" remote set-url "
        "origin 'https://github.com/kikixiong/tabicl-pe.git'\n"
        "    /usr/bin/git -C \"$checkout_parent/$destination\" update-ref "
        f"refs/remotes/origin/codex/position-identity-v1 '{commit}'\n"
        "    exit 0\n"
        "  fi\n"
        "  exec /usr/bin/git \"${args[@]}\"\n"
        "fi\n"
    )
    fake_git.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    git_sha256 = hashlib.sha256(fake_git.read_bytes()).hexdigest()
    helper_spec = importlib.util.spec_from_file_location(
        "formal_spool_repository_helper",
        Path(__file__).parents[1] / "scripts/verify_git_repository.py",
    )
    assert helper_spec is not None and helper_spec.loader is not None
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    repository_binding = helper.expected_repository_binding(
        expected_commit_sha=commit, git_sha256=git_sha256
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "MODE": "rope",
        "STAGE": "1",
        "NUM_GPUS": "1",
        "FORMAL_SEED": "42",
        "CANDIDATE_REPOSITORY": "https://github.com/kikixiong/tabicl-pe.git",
        "CANDIDATE_REPOSITORY_REF": "refs/heads/codex/position-identity-v1",
        "FORMAL_SOURCE_COMMIT_SHA": commit,
        "FORMAL_SOURCE_TREE_SHA": tree,
        "FORMAL_ENVIRONMENT_SHA256": "e" * 64,
        "FORMAL_EXPECTED_GPU_MODEL": "NVIDIA H100 80GB HBM3",
        "FORMAL_EXPECTED_DRIVER_VERSION": "570.00",
        "FORMAL_GIT_SHA256": git_sha256,
        "FORMAL_REPOSITORY_IDENTITY_SHA256": repository_binding[
            "repository_identity_sha256"
        ],
        "FORMAL_REPOSITORY_QUERY_SHA256": repository_binding["query_sha256"],
        "FORMAL_JOB_WORK_ROOT": str(work_root),
        "FORMAL_ARTIFACT_ROOT": str(artifact_root),
        "FORMAL_SUBMISSION_EXACT_ROOT": str(candidate),
        "PYTHON": sys.executable,
        "GIT": str(fake_git),
        "NVIDIA_SMI": str(nvidia),
        "FORMAL_NVIDIA_SMI_SHA256": hashlib.sha256(
            nvidia.read_bytes()
        ).hexdigest(),
        "CUDA_VISIBLE_DEVICES": "7",
        "RUNTIME_MARKER": str(marker),
        "RUNTIME_EXIT": str(exit_code),
        "NVIDIA_ARGS": str(nvidia_args),
    }
    return spool, work_root, marker, nvidia_args, environment


@pytest.mark.parametrize("exit_code", [0, 7])
def test_spooled_wrapper_uses_allocated_gpu_and_cleans_checkout_on_all_exits(
    tmp_path, exit_code
):
    spool, work_root, marker, nvidia_args, environment = _fixture(
        tmp_path, exit_code
    )
    completed = subprocess.run(
        ["/bin/bash", str(spool)],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == exit_code
    checkout = Path(marker.read_text())
    assert not checkout.exists()
    assert list(work_root.iterdir()) == []
    assert "--id=7" in nvidia_args.read_text().splitlines()
    assert (
        "--query-gpu=name,uuid,driver_version"
        in nvidia_args.read_text().splitlines()
    )
    assert list(spool.parent.iterdir()) == [spool]
    work_root.rmdir()


def test_spooled_wrapper_rejects_leading_dash_repository_before_clone(tmp_path):
    spool, work_root, marker, _nvidia_args, environment = _fixture(tmp_path, 0)
    environment["CANDIDATE_REPOSITORY"] = "--upload-pack=poison"
    completed = subprocess.run(
        ["/bin/bash", str(spool)],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "not canonical" in completed.stderr
    assert not marker.exists()
    assert list(work_root.iterdir()) == []
    work_root.rmdir()


def test_spooled_wrapper_rejects_work_on_artifact_filesystem_before_clone(tmp_path):
    spool, work_root, marker, _nvidia_args, environment = _fixture(tmp_path, 0)
    environment["FORMAL_JOB_WORK_ROOT"] = environment["FORMAL_ARTIFACT_ROOT"]
    completed = subprocess.run(
        ["/bin/bash", str(spool)],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "different filesystem" in completed.stderr
    assert not marker.exists()
    assert list(work_root.iterdir()) == []
    work_root.rmdir()


@pytest.mark.parametrize(
    "helper_name",
    (
        "exec_digest_bound_nvidia_smi.py",
        "verify_git_repository.py",
        "verify_filesystem_isolation.py",
        "verify_formal_environment.py",
    ),
)
def test_trusted_spool_rejects_modified_exact_helper_without_executing_it(
    tmp_path, helper_name
):
    spool, work_root, runtime_marker, _nvidia_args, environment = _fixture(
        tmp_path, 0
    )
    malicious_marker = tmp_path / "modified-helper-executed"
    helper = (
        Path(environment["FORMAL_SUBMISSION_EXACT_ROOT"])
        / "scripts"
        / helper_name
    )
    helper.write_text(
        "from pathlib import Path\n"
        f"Path({str(malicious_marker)!r}).write_text('executed')\n"
        "raise SystemExit(97)\n"
    )
    helper.chmod(0o755)
    completed = subprocess.run(
        ["/bin/bash", str(spool)],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "trusted static production bootstrap failed: exact root is dirty" in (
        completed.stderr
    )
    assert not malicious_marker.exists()
    assert not runtime_marker.exists()
    assert list(work_root.iterdir()) == []
    work_root.rmdir()
