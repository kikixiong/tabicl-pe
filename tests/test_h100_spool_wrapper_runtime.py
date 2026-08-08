from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

import pytest


ROOT = Path(__file__).parents[1]


def _run(argv, **kwargs):
    return subprocess.run(argv, check=True, text=True, **kwargs)


@pytest.fixture()
def spool_runtime(tmp_path):
    exact = tmp_path / "exact"
    scripts = exact / "scripts"
    external = tmp_path / "external"
    spool = tmp_path / "slurm-spool"
    for path in (scripts, external / "cases", spool):
        path.mkdir(parents=True)

    case_work = None
    artifact_device = (external / "cases").stat().st_dev
    for candidate_parent in (Path("/dev/shm"), Path("/tmp"), ROOT.parent):
        if (
            candidate_parent.is_dir()
            and os.access(candidate_parent, os.W_OK | os.X_OK)
            and candidate_parent.stat().st_dev != artifact_device
        ):
            case_work = Path(
                tempfile.mkdtemp(prefix="tabicl-h100-case-work-", dir=candidate_parent)
            )
            break
    if case_work is None:
        pytest.skip("test host has no writable second filesystem")

    for name in (
        "exec_digest_bound_git.py",
        "exec_digest_bound_nvidia_smi.py",
        "slurm_h100_identity_maxseq_smoke.sh",
        "slurm_h100_identity_nccl_smoke.sh",
        "run_slurm_h100_identity_case.sh",
        "verify_filesystem_isolation.py",
        "verify_formal_environment.py",
        "verify_git_repository.py",
    ):
        shutil.copy2(ROOT / "scripts" / name, scripts / name)
        if name.endswith(".sh"):
            (scripts / name).chmod((scripts / name).stat().st_mode | stat.S_IXUSR)

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

    workload = scripts / "run_h100_identity_maxseq_smoke.sh"
    workload.write_text(
        """#!/bin/bash
set -euo pipefail
[[ "${BASH_SOURCE[0]}" == "$TABICL_EXACT_ROOT/scripts/run_h100_identity_maxseq_smoke.sh" ]]
[[ "$FORMAL_VISIBLE_GPU_TOKENS" == "$TEST_EXPECTED_GPU_TOKEN" ]]
[[ "$FORMAL_VISIBLE_GPU_UUIDS" == "GPU-fixture" ]]
printf 'ready\\n' > "$TEST_MARKER"
case "$TEST_MODE" in
  success) exit 0 ;;
  failure) exit 37 ;;
  wait) while :; do /usr/bin/sleep 1; done ;;
  *) exit 98 ;;
esac
"""
    )
    workload.chmod(0o755)

    _run(["/usr/bin/git", "init", "-q", str(exact)])
    _run(["/usr/bin/git", "-C", str(exact), "config", "user.name", "fixture"])
    _run(
        [
            "/usr/bin/git",
            "-C",
            str(exact),
            "config",
            "user.email",
            "fixture@example.invalid",
        ]
    )
    _run(["/usr/bin/git", "-C", str(exact), "add", "scripts"])
    _run(["/usr/bin/git", "-C", str(exact), "commit", "-q", "-m", "fixture"])
    commit = _run(
        ["/usr/bin/git", "-C", str(exact), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
    ).stdout.strip()
    tree = _run(
        ["/usr/bin/git", "-C", str(exact), "rev-parse", "HEAD^{tree}"],
        stdout=subprocess.PIPE,
    ).stdout.strip()
    _run(["/usr/bin/git", "-C", str(exact), "checkout", "-q", "--detach", commit])

    # Model Slurm's copied spool script: there is deliberately no sibling
    # common runner beside this file.
    spool_wrapper = spool / "slurm_script"
    shutil.copy2(scripts / "slurm_h100_identity_maxseq_smoke.sh", spool_wrapper)
    spool_wrapper.chmod(0o755)
    assert {path.name for path in spool.iterdir()} == {"slurm_script"}

    nvidia_log = external / "nvidia-argv.log"
    nvidia_exit = external / "nvidia-exit-code"
    nvidia_exit.write_text("0\n")
    nvidia_smi = external / "nvidia-smi"
    nvidia_smi.write_text(
        f"""#!/bin/bash
set -euo pipefail
printf '%s\\n' "$*" >> {str(nvidia_log)!r}
printf '%s\\n' 'GPU-fixture, NVIDIA H100 80GB HBM3, 570.00'
exit "$(< {str(nvidia_exit)!r})"
"""
    )
    nvidia_smi.chmod(0o755)
    fake_git = external / "git"
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
        f"    [[ \"${{args[$index]}}\" != 'https://github.com/kikixiong/tabicl-pe.git' ]] || args[$index]='{exact}'\n"
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
    fake_git.chmod(0o755)
    git_sha256 = hashlib.sha256(fake_git.read_bytes()).hexdigest()
    helper_spec = importlib.util.spec_from_file_location(
        "h100_spool_repository_helper", ROOT / "scripts/verify_git_repository.py"
    )
    assert helper_spec is not None and helper_spec.loader is not None
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    repository_binding = helper.expected_repository_binding(
        expected_commit_sha=commit, git_sha256=git_sha256
    )
    source_manifest = external / "source.json"
    source_manifest.write_text("{}\n")

    def environment(*, token: str, mode: str, marker: Path) -> dict[str, str]:
        return {
            "PATH": "/usr/bin:/bin",
            "FORMAL_EXPECTED_GPUS": "1",
            "FORMAL_REQUESTED_GPUS": "1",
            "VALIDATION_CASE_ID": "stage1_rope_one_step",
            "CANDIDATE_REPOSITORY": "https://github.com/kikixiong/tabicl-pe.git",
            "CANDIDATE_REPOSITORY_REF": "refs/heads/codex/position-identity-v1",
            "FORMAL_SOURCE_COMMIT_SHA": commit,
            "FORMAL_SOURCE_TREE_SHA": tree,
            "FORMAL_SOURCE_MANIFEST": str(source_manifest),
            "FORMAL_SOURCE_SHA256": "a" * 64,
            "FORMAL_GIT_SHA256": git_sha256,
            "FORMAL_REPOSITORY_IDENTITY_SHA256": repository_binding[
                "repository_identity_sha256"
            ],
            "FORMAL_REPOSITORY_QUERY_SHA256": repository_binding["query_sha256"],
            "FORMAL_EXPECTED_ENVIRONMENT_SHA256": "b" * 64,
            "VALIDATION_ARTIFACT_IDENTITY_SHA256": "c" * 64,
            "H100_CASE_WORK_ROOT": str(case_work),
            "H100_VALIDATION_ROOT": str(external / "cases"),
            "CHECKPOINT_CEILING_BYTES": "1000",
            "FORMAL_RUN_LOG_CEILING_BYTES": "1000",
            "FORMAL_GPU_MONITOR_CEILING_BYTES": "1000",
            "FORMAL_ATTESTATION_CEILING_BYTES": "1000",
            "PYTHON": str(Path(sys.executable).resolve()),
            "GIT": str(fake_git),
            "NVIDIA_SMI": str(nvidia_smi),
            "FORMAL_NVIDIA_SMI_SHA256": hashlib.sha256(
                nvidia_smi.read_bytes()
            ).hexdigest(),
            "RUN_POLICY": "fresh",
            "FORMAL_SUBMISSION_EXACT_ROOT": str(exact),
            "SLURM_JOB_ID": "901",
            "CUDA_VISIBLE_DEVICES": token,
            "FORMAL_REQUESTED_RESOURCE_SHA256": "d" * 64,
            "FORMAL_REQUESTED_PARTITION": "h100",
            "FORMAL_REQUESTED_QOS": "short",
            "FORMAL_REQUESTED_TIME_LIMIT": "03:00:00",
            "FORMAL_REQUESTED_NODES": "1",
            "FORMAL_REQUESTED_CPUS": "32",
            "FORMAL_REQUESTED_MEMORY_MB": "131072",
            "SLURM_JOB_PARTITION": "h100",
            "SLURM_CPUS_PER_TASK": "32",
            "SLURM_MEM_PER_NODE": "131072",
            "SLURM_CLUSTER_NAME": "cluster-a",
            "TEST_MODE": mode,
            "TEST_MARKER": str(marker),
            "TEST_EXPECTED_GPU_TOKEN": token,
        }

    yield {
        "wrapper": spool_wrapper,
        "exact": exact,
        "fake_git": fake_git,
        "case_work": case_work,
        "nvidia_log": nvidia_log,
        "nvidia_exit": nvidia_exit,
        "environment": environment,
    }
    shutil.rmtree(case_work, ignore_errors=True)


def test_valid_gpu_row_followed_by_nonzero_nvidia_status_is_rejected(
    spool_runtime, tmp_path
):
    marker = tmp_path / "workload-marker"
    environment = spool_runtime["environment"](
        token="0", mode="success", marker=marker
    )
    spool_runtime["nvidia_exit"].write_text("23\n")
    completed = subprocess.run(
        ["/bin/bash", str(spool_runtime["wrapper"])],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "CUDA-visible nvidia-smi query failed" in completed.stderr
    assert not marker.exists()
    assert list(spool_runtime["case_work"].iterdir()) == []


@pytest.mark.parametrize(
    ("token", "expected_gpus", "case_id"),
    (
        ("0,", "1", "stage1_rope_one_step"),
        ("0,1,", "2", "nccl_2gpu"),
    ),
)
def test_trailing_comma_gpu_list_is_rejected_before_query_checkout_or_workload(
    spool_runtime, tmp_path, token, expected_gpus, case_id
):
    if expected_gpus == "1":
        trusted_wrapper = spool_runtime["wrapper"]
    else:
        trusted_wrapper = tmp_path / "trusted-nccl-spool-wrapper"
        shutil.copy2(
            spool_runtime["exact"]
            / "scripts/slurm_h100_identity_nccl_smoke.sh",
            trusted_wrapper,
        )
    marker = tmp_path / "workload-marker"
    environment = spool_runtime["environment"](
        token=token, mode="success", marker=marker
    )
    environment.update(
        {
            "FORMAL_EXPECTED_GPUS": expected_gpus,
            "FORMAL_REQUESTED_GPUS": expected_gpus,
            "VALIDATION_CASE_ID": case_id,
        }
    )
    completed = subprocess.run(
        ["/bin/bash", str(trusted_wrapper)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "CUDA_VISIBLE_DEVICES is not a canonical" in completed.stderr
    assert not marker.exists()
    assert not spool_runtime["nvidia_log"].exists()
    assert list(spool_runtime["case_work"].iterdir()) == []


@pytest.mark.parametrize(
    ("wrapper_name", "expected_gpus", "case_id"),
    (
        ("slurm_h100_identity_maxseq_smoke.sh", "1", "stage1_rope_one_step"),
        ("slurm_h100_identity_nccl_smoke.sh", "2", "nccl_2gpu"),
    ),
)
def test_static_wrapper_never_executes_replaced_git_before_digest_rejection(
    spool_runtime, tmp_path, wrapper_name, expected_gpus, case_id
):
    marker = tmp_path / "malicious-git-executed"
    fake_git = spool_runtime["fake_git"]
    fake_git.rename(fake_git.with_name("git.verified-inode"))
    fake_git.write_text(
        "#!/bin/bash\n"
        f"printf executed > {str(marker)!r}\n"
        "exit 97\n"
    )
    fake_git.chmod(0o755)
    environment = spool_runtime["environment"](
        token="0", mode="success", marker=tmp_path / "workload-marker"
    )
    environment.update(
        {
            "FORMAL_EXPECTED_GPUS": expected_gpus,
            "FORMAL_REQUESTED_GPUS": expected_gpus,
            "VALIDATION_CASE_ID": case_id,
        }
    )
    completed = subprocess.run(
        [
            "/bin/bash",
            str(spool_runtime["exact"] / "scripts" / wrapper_name),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "trusted bootstrap Git digest mismatch" in completed.stderr
    assert not marker.exists()


@pytest.mark.parametrize(
    ("wrapper_name", "expected_gpus", "case_id"),
    (
        ("slurm_h100_identity_maxseq_smoke.sh", "1", "stage1_rope_one_step"),
        ("slurm_h100_identity_nccl_smoke.sh", "2", "nccl_2gpu"),
    ),
)
def test_trusted_static_wrapper_never_executes_modified_shared_runner(
    spool_runtime, tmp_path, wrapper_name, expected_gpus, case_id
):
    if expected_gpus == "1":
        trusted_wrapper = spool_runtime["wrapper"]
    else:
        trusted_wrapper = tmp_path / "trusted-nccl-spool-wrapper"
        shutil.copy2(
            spool_runtime["exact"] / "scripts" / wrapper_name,
            trusted_wrapper,
        )
    marker = tmp_path / "modified-shared-runner-executed"
    shared_runner = (
        spool_runtime["exact"] / "scripts/run_slurm_h100_identity_case.sh"
    )
    shared_runner.write_text(
        "#!/bin/bash\n"
        f"printf executed > {str(marker)!r}\n"
        "exit 97\n"
    )
    shared_runner.chmod(0o755)
    environment = spool_runtime["environment"](
        token="0", mode="success", marker=tmp_path / "workload-marker"
    )
    environment.update(
        {
            "FORMAL_EXPECTED_GPUS": expected_gpus,
            "FORMAL_REQUESTED_GPUS": expected_gpus,
            "VALIDATION_CASE_ID": case_id,
        }
    )
    completed = subprocess.run(
        ["/bin/bash", str(trusted_wrapper)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "trusted bootstrap exact root is dirty" in completed.stderr
    assert not marker.exists()


@pytest.mark.parametrize(
    ("wrapper_name", "expected_gpus", "case_id"),
    (
        ("slurm_h100_identity_maxseq_smoke.sh", "1", "stage1_rope_one_step"),
        ("slurm_h100_identity_nccl_smoke.sh", "2", "nccl_2gpu"),
    ),
)
def test_trusted_static_wrapper_rejects_fifo_shared_runner_without_blocking(
    spool_runtime, tmp_path, wrapper_name, expected_gpus, case_id
):
    if expected_gpus == "1":
        trusted_wrapper = spool_runtime["wrapper"]
    else:
        trusted_wrapper = tmp_path / "trusted-nccl-spool-wrapper"
        shutil.copy2(
            spool_runtime["exact"] / "scripts" / wrapper_name,
            trusted_wrapper,
        )
    shared_runner = (
        spool_runtime["exact"] / "scripts/run_slurm_h100_identity_case.sh"
    )
    shared_runner.unlink()
    os.mkfifo(shared_runner, stat.S_IRUSR | stat.S_IWUSR)
    environment = spool_runtime["environment"](
        token="0", mode="success", marker=tmp_path / "workload-marker"
    )
    environment.update(
        {
            "FORMAL_EXPECTED_GPUS": expected_gpus,
            "FORMAL_REQUESTED_GPUS": expected_gpus,
            "VALIDATION_CASE_ID": case_id,
        }
    )
    completed = subprocess.run(
        ["/bin/bash", str(trusted_wrapper)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 2
    assert "trusted bootstrap file is not a stable bounded executable" in completed.stderr


@pytest.mark.parametrize(
    "helper_name",
    (
        "exec_digest_bound_nvidia_smi.py",
        "verify_git_repository.py",
        "verify_filesystem_isolation.py",
        "verify_formal_environment.py",
    ),
)
def test_modified_assume_unchanged_helper_is_blob_rejected_before_execution(
    spool_runtime, tmp_path, helper_name
):
    exact = spool_runtime["exact"]
    relative = f"scripts/{helper_name}"
    _run(
        [
            "/usr/bin/git",
            "-C",
            str(exact),
            "update-index",
            "--assume-unchanged",
            relative,
        ]
    )
    marker = tmp_path / "modified-helper-executed"
    helper = exact / relative
    helper.write_text(
        "#!/bin/bash\n"
        f"printf executed > {str(marker)!r}\n"
        "exit 97\n"
    )
    helper.chmod(0o755)
    completed = subprocess.run(
        ["/bin/bash", str(spool_runtime["wrapper"])],
        env=spool_runtime["environment"](
            token="0", mode="success", marker=tmp_path / "workload-marker"
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 2
    assert "trusted H100 helper differs from committed source" in completed.stderr
    assert not marker.exists()


def test_local_fsmonitor_config_cannot_execute_during_static_or_runner_git_checks(
    spool_runtime, tmp_path
):
    exact = spool_runtime["exact"]
    marker = tmp_path / "fsmonitor-executed"
    hook = tmp_path / "fsmonitor-hook"
    hook.write_text(
        "#!/bin/bash\n"
        f"printf executed > {str(marker)!r}\n"
        "printf '0\\n'\n"
    )
    hook.chmod(0o755)
    _run(
        [
            "/usr/bin/git",
            "-C",
            str(exact),
            "config",
            "core.fsmonitor",
            str(hook),
        ]
    )
    workload_marker = tmp_path / "workload-marker"
    completed = subprocess.run(
        ["/bin/bash", str(spool_runtime["wrapper"])],
        env=spool_runtime["environment"](
            token="0", mode="success", marker=workload_marker
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert workload_marker.exists()
    assert not marker.exists()


@pytest.mark.parametrize("mutation", ("dirty", "attached"))
def test_shared_digest_bound_body_rejects_nonexact_submission_checkout(
    spool_runtime, tmp_path, mutation
):
    exact = spool_runtime["exact"]
    if mutation == "dirty":
        (exact / "untracked-marker").write_text("dirty\n")
        expected_error = "trusted bootstrap exact root is dirty"
    else:
        _run(["/usr/bin/git", "-C", str(exact), "checkout", "-q", "master"])
        expected_error = "trusted bootstrap exact root is not detached"
    marker = tmp_path / "workload-marker"
    completed = subprocess.run(
        ["/bin/bash", str(spool_runtime["wrapper"])],
        env=spool_runtime["environment"](token="0", mode="success", marker=marker),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert expected_error in completed.stderr
    assert not marker.exists()


@pytest.mark.parametrize("token", ["0", "GPU-assigned_1", "MIG-assigned_1"])
def test_spooled_wrapper_uses_exact_root_runner_and_scoped_gpu_token(
    spool_runtime, tmp_path, token
):
    marker = tmp_path / (token.replace("/", "_") + ".ready")
    completed = subprocess.run(
        ["/bin/bash", str(spool_runtime["wrapper"])],
        env=spool_runtime["environment"](token=token, mode="success", marker=marker),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr
    assert marker.read_text() == "ready\n"
    assert list(spool_runtime["case_work"].iterdir()) == []
    assert f"--id={token}" in spool_runtime["nvidia_log"].read_text().splitlines()[-1]


def test_spooled_wrapper_cleans_detached_checkout_after_workload_failure(
    spool_runtime, tmp_path
):
    marker = tmp_path / "failure.ready"
    completed = subprocess.run(
        ["/bin/bash", str(spool_runtime["wrapper"])],
        env=spool_runtime["environment"](token="0", mode="failure", marker=marker),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 37
    assert marker.is_file()
    assert list(spool_runtime["case_work"].iterdir()) == []


def test_spooled_wrapper_rejects_work_on_artifact_filesystem_before_clone(
    spool_runtime, tmp_path
):
    marker = tmp_path / "same-device.ready"
    environment = spool_runtime["environment"](
        token="0", mode="success", marker=marker
    )
    environment["H100_CASE_WORK_ROOT"] = environment["H100_VALIDATION_ROOT"]
    completed = subprocess.run(
        ["/bin/bash", str(spool_runtime["wrapper"])],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode != 0
    assert "different filesystem" in completed.stderr
    assert not marker.exists()


def test_spooled_wrapper_cleans_detached_checkout_after_sigterm(spool_runtime, tmp_path):
    marker = tmp_path / "signal.ready"
    process = subprocess.Popen(
        ["/bin/bash", str(spool_runtime["wrapper"])],
        env=spool_runtime["environment"](token="0", mode="wait", marker=marker),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    deadline = time.monotonic() + 10
    while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert marker.exists(), process.stderr.read() if process.poll() is not None else ""
    os.killpg(process.pid, signal.SIGTERM)
    process.communicate(timeout=10)
    assert process.returncode != 0
    assert list(spool_runtime["case_work"].iterdir()) == []


def test_case_artifact_root_is_created_with_atomic_fresh_only_mkdir(tmp_path):
    source = (ROOT / "scripts/run_h100_identity_maxseq_smoke.sh").read_text()
    assert "os.mkdir(case_id, mode=0o700, dir_fd=directory_fd)" in source
    assert "os.O_NOFOLLOW" in source
    assert 'if ! mkdir -- "$CASE_ROOT"; then' not in source
    assert 'mkdir -p "$CASE_ROOT"' not in source
    assert '[[ ! -e "$CASE_ROOT"' not in source

    validation_root = tmp_path / "cases"
    case_root = validation_root / "stage1_rope_one_step"
    case_root.mkdir(parents=True)
    sentinel = case_root / "sentinel"
    sentinel.write_text("preserve\n")
    environment = {
        "PATH": "/usr/bin:/bin",
            "TABICL_EXACT_ROOT": str(ROOT),
            "PYTHON": str(Path(sys.executable).resolve()),
            "NVIDIA_SMI": "/usr/bin/true",
            "FORMAL_NVIDIA_SMI_SHA256": hashlib.sha256(
                Path("/usr/bin/true").read_bytes()
            ).hexdigest(),
        "FORMAL_SOURCE_MANIFEST": str(tmp_path / "source.json"),
        "FORMAL_SOURCE_SHA256": "a" * 64,
        "FORMAL_SOURCE_COMMIT_SHA": "b" * 40,
        "FORMAL_SOURCE_TREE_SHA": "c" * 40,
        "VALIDATION_ARTIFACT_IDENTITY_SHA256": "d" * 64,
        "FORMAL_EXPECTED_ENVIRONMENT_SHA256": "e" * 64,
        "H100_VALIDATION_ROOT": str(validation_root),
        "FORMAL_RUN_LOG_CEILING_BYTES": "1000",
        "FORMAL_GPU_MONITOR_CEILING_BYTES": "1000",
        "FORMAL_ATTESTATION_CEILING_BYTES": "1000",
        "CHECKPOINT_CEILING_BYTES": "1000",
    }
    completed = subprocess.run(
        [
            str(Path(sys.executable).resolve()),
            "-I",
            "-B",
            str(ROOT / "scripts/exec_digest_bound_nvidia_smi.py"),
            "--nvidia-smi",
            "/usr/bin/true",
            "--expected-sha256",
            environment["FORMAL_NVIDIA_SMI_SHA256"],
            "--",
            "/bin/bash",
            str(ROOT / "scripts/run_h100_identity_maxseq_smoke.sh"),
            "stage1_rope_one_step",
        ],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode != 0
    assert "fresh-only" in completed.stderr
    assert sentinel.read_text() == "preserve\n"


def test_case_artifact_root_rejects_symlinked_parent_component(tmp_path):
    physical_parent = tmp_path / "physical-parent"
    validation_root = physical_parent / "cases"
    validation_root.mkdir(parents=True)
    parent_alias = tmp_path / "parent-alias"
    parent_alias.symlink_to(physical_parent, target_is_directory=True)
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text("{}\n")
    environment = {
        "PATH": "/usr/bin:/bin",
        "TABICL_EXACT_ROOT": str(ROOT),
        "PYTHON": str(Path(sys.executable).resolve()),
        "NVIDIA_SMI": "/usr/bin/true",
        "FORMAL_NVIDIA_SMI_SHA256": hashlib.sha256(
            Path("/usr/bin/true").read_bytes()
        ).hexdigest(),
        "FORMAL_SOURCE_MANIFEST": str(source_manifest),
        "FORMAL_SOURCE_SHA256": "a" * 64,
        "FORMAL_SOURCE_COMMIT_SHA": "b" * 40,
        "FORMAL_SOURCE_TREE_SHA": "c" * 40,
        "VALIDATION_ARTIFACT_IDENTITY_SHA256": "d" * 64,
        "FORMAL_EXPECTED_ENVIRONMENT_SHA256": "e" * 64,
        "H100_VALIDATION_ROOT": str(parent_alias / "cases"),
        "FORMAL_RUN_LOG_CEILING_BYTES": "1000",
        "FORMAL_GPU_MONITOR_CEILING_BYTES": "1000",
        "FORMAL_ATTESTATION_CEILING_BYTES": "1000",
        "CHECKPOINT_CEILING_BYTES": "1000",
    }
    completed = subprocess.run(
        [
            str(Path(sys.executable).resolve()),
            "-I",
            "-B",
            str(ROOT / "scripts/exec_digest_bound_nvidia_smi.py"),
            "--nvidia-smi",
            "/usr/bin/true",
            "--expected-sha256",
            environment["FORMAL_NVIDIA_SMI_SHA256"],
            "--",
            "/bin/bash",
            str(ROOT / "scripts/run_h100_identity_maxseq_smoke.sh"),
            "stage1_rope_one_step",
        ],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode != 0
    assert "physical validation-root traversal" in completed.stderr
    assert not (validation_root / "stage1_rope_one_step").exists()

    unnormalized_environment = dict(environment)
    unnormalized_environment["H100_VALIDATION_ROOT"] = (
        f"{validation_root}/../cases"
    )
    completed = subprocess.run(
        [
            str(Path(sys.executable).resolve()),
            "-I",
            "-B",
            str(ROOT / "scripts/exec_digest_bound_nvidia_smi.py"),
            "--nvidia-smi",
            "/usr/bin/true",
            "--expected-sha256",
            unnormalized_environment["FORMAL_NVIDIA_SMI_SHA256"],
            "--",
            "/bin/bash",
            str(ROOT / "scripts/run_h100_identity_maxseq_smoke.sh"),
            "stage1_rope_one_step",
        ],
        env=unnormalized_environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode != 0
    assert "normalized absolute path" in completed.stderr
    assert not (validation_root / "stage1_rope_one_step").exists()
