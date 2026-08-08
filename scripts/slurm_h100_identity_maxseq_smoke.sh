#!/usr/bin/env bash
#SBATCH --job-name=tabicl-identity-gate-1g
#SBATCH --partition=h100
#SBATCH --qos=short
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail

# Keep the static Slurm contract and the controller's command-line resource
# request independently fail-closed.  The shared body also checks the actual
# visible H100 inventory before it creates a checkout or an artifact directory.
[[ "${FORMAL_EXPECTED_GPUS:-}" == "1" ]] || {
  echo "one-GPU validation wrapper requires FORMAL_EXPECTED_GPUS=1" >&2
  exit 2
}
[[ "${VALIDATION_CASE_ID:-}" != "nccl_2gpu" ]] || {
  echo "nccl_2gpu must use the dedicated two-GPU wrapper" >&2
  exit 2
}
: "${FORMAL_SUBMISSION_EXACT_ROOT:?FORMAL_SUBMISSION_EXACT_ROOT is required}"
: "${GIT:?GIT is required}"
[[ "$FORMAL_SUBMISSION_EXACT_ROOT" == /* && "$GIT" == /* ]] || {
  echo "invalid exact-root bootstrap contract" >&2
  exit 2
}
: "${PYTHON:?PYTHON is required for the trusted static bootstrap}"
: "${FORMAL_GIT_SHA256:?FORMAL_GIT_SHA256 is required}"
: "${FORMAL_SOURCE_COMMIT_SHA:?FORMAL_SOURCE_COMMIT_SHA is required}"
: "${FORMAL_SOURCE_TREE_SHA:?FORMAL_SOURCE_TREE_SHA is required}"
exec "$PYTHON" -I -B - 1 <<'PY'
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys


MAX_BYTES = 128 * 1024 * 1024
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TRUSTED_FILES = {
    "FORMAL_TRUSTED_RUNNER_FD": ("run_slurm_h100_identity_case.sh", True),
    "FORMAL_H100_NVIDIA_LAUNCHER_FD": ("exec_digest_bound_nvidia_smi.py", False),
    "FORMAL_H100_VERIFY_GIT_FD": ("verify_git_repository.py", False),
    "FORMAL_H100_VERIFY_FILESYSTEM_FD": ("verify_filesystem_isolation.py", False),
    "FORMAL_H100_VERIFY_ENVIRONMENT_FD": ("verify_formal_environment.py", False),
}


def fail(message):
    print(f"trusted static H100 bootstrap failed: {message}", file=sys.stderr)
    raise SystemExit(2)


def signature(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def open_directory(path):
    raw = os.fspath(path)
    if not path.is_absolute() or raw != os.path.abspath(raw):
        fail("trusted bootstrap path is not normalized absolute")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    fd = os.open("/", flags)
    try:
        for component in PurePosixPath(raw).parts[1:]:
            next_fd = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def open_file(path, *, executable):
    parent_fd = open_directory(path.parent)
    try:
        before = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        fd = os.open(path.name, flags, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    opened = os.fstat(fd)
    if (
        signature(opened) != signature(before)
        or not stat.S_ISREG(opened.st_mode)
        or (executable and opened.st_mode & 0o111 == 0)
        or opened.st_size > MAX_BYTES
    ):
        os.close(fd)
        fail("trusted bootstrap file is not a stable bounded executable")
    return fd, opened


def read_stable(fd, opened):
    digest = hashlib.sha256()
    chunks = []
    remaining = opened.st_size
    while remaining:
        chunk = os.read(fd, min(1 << 20, remaining))
        if not chunk:
            fail("trusted bootstrap file was truncated")
        chunks.append(chunk)
        digest.update(chunk)
        remaining -= len(chunk)
    if signature(os.fstat(fd)) != signature(opened):
        fail("trusted bootstrap file changed while hashing")
    os.lseek(fd, 0, os.SEEK_SET)
    return b"".join(chunks), digest.hexdigest()


root = Path(os.environ["FORMAL_SUBMISSION_EXACT_ROOT"])
git_path = Path(os.environ["GIT"])
expected_git = os.environ["FORMAL_GIT_SHA256"]
commit = os.environ["FORMAL_SOURCE_COMMIT_SHA"]
tree = os.environ["FORMAL_SOURCE_TREE_SHA"]
if HEX64.fullmatch(expected_git) is None or HEX40.fullmatch(commit) is None or HEX40.fullmatch(tree) is None:
    fail("trusted bootstrap digest or source identity is malformed")

git_fd, git_stat = open_file(git_path, executable=True)
trusted_fds = {}
try:
    _, git_digest = read_stable(git_fd, git_stat)
    if git_digest != expected_git:
        fail("trusted bootstrap Git digest mismatch")
    trusted_raw = {}
    for environment_name, (filename, executable) in TRUSTED_FILES.items():
        fd, opened = open_file(
            root / "scripts" / filename, executable=executable
        )
        raw, _ = read_stable(fd, opened)
        trusted_fds[environment_name] = fd
        trusted_raw[filename] = raw

    git_command = f"/proc/self/fd/{git_fd}"
    git_environment = {
        "PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_COUNT": "0", "GIT_CEILING_DIRECTORIES": "/",
        "GIT_TERMINAL_PROMPT": "0", "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_ALLOW_PROTOCOL": "https",
    }

    def run_git(*arguments, check=True):
        result = subprocess.run(
            [
                git_command,
                "-c", "core.fsmonitor=false",
                "-c", "core.hooksPath=/dev/null",
                "-c", "core.filemode=true",
                "-C", os.fspath(root), *arguments,
            ],
            env=git_environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, pass_fds=(git_fd,), timeout=30,
        )
        if len(result.stdout) > MAX_BYTES or len(result.stderr) > (1 << 20):
            fail("trusted bootstrap Git output exceeds its ceiling")
        if check and result.returncode != 0:
            fail("trusted bootstrap Git query failed")
        return result

    if run_git("rev-parse", "HEAD").stdout.strip() != commit.encode():
        fail("trusted bootstrap exact-root commit mismatch")
    if run_git("rev-parse", "HEAD^{tree}").stdout.strip() != tree.encode():
        fail("trusted bootstrap exact-root tree mismatch")
    if run_git("status", "--porcelain=v1", "--untracked-files=all").stdout:
        fail("trusted bootstrap exact root is dirty")
    symbolic = run_git("symbolic-ref", "-q", "HEAD", check=False)
    if symbolic.returncode != 1:
        fail("trusted bootstrap exact root is not detached")
    for filename, raw in trusted_raw.items():
        committed = run_git("show", f"{commit}:scripts/{filename}")
        if committed.stdout != raw:
            fail(f"trusted H100 helper differs from committed source: {filename}")
    os.set_inheritable(git_fd, True)
    for fd in trusted_fds.values():
        os.set_inheritable(fd, True)
    environment = dict(os.environ)
    environment["FORMAL_GIT_FD"] = str(git_fd)
    environment["FORMAL_GIT_FD_OWNER_PID"] = str(os.getpid())
    environment["FORMAL_GIT_COMMAND"] = git_command
    for name, fd in trusted_fds.items():
        environment[name] = str(fd)
    runner_fd = trusted_fds["FORMAL_TRUSTED_RUNNER_FD"]
    runner_command = f"/proc/self/fd/{runner_fd}"
    os.execve("/bin/bash", ["/bin/bash", runner_command, sys.argv[1]], environment)
finally:
    os.close(git_fd)
    for fd in trusted_fds.values():
        os.close(fd)
PY
