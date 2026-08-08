#!/usr/bin/env bash
#SBATCH --job-name=tabicl-identity-formal
#SBATCH --partition=h100
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=131072M

set -euo pipefail
export PATH=/usr/bin:/bin
: "${MODE:?MODE must be rope, temporary, or none}"
: "${STAGE:?STAGE must be 1, 2, or 3}"
: "${NUM_GPUS:?NUM_GPUS must be exactly 1}"
: "${FORMAL_SEED:?FORMAL_SEED must be exactly 42, 43, or 44}"
case "$MODE" in rope|temporary|none) ;; *) echo "invalid MODE" >&2; exit 2 ;; esac
case "$STAGE" in 1|2|3) ;; *) echo "invalid STAGE" >&2; exit 2 ;; esac
case "$FORMAL_SEED" in
  42|43|44) ;;
  *) echo "FORMAL_SEED must be exactly 42, 43, or 44" >&2; exit 2 ;;
esac
readonly FORMAL_SEED
[[ "$NUM_GPUS" -eq 1 ]] || { echo "all nine production jobs require exactly one GPU" >&2; exit 2; }
: "${CANDIDATE_REPOSITORY:?}"
: "${CANDIDATE_REPOSITORY_REF:?}"
[[ "$CANDIDATE_REPOSITORY" == "https://github.com/kikixiong/tabicl-pe.git" ]] || {
  echo "formal candidate repository is not canonical" >&2; exit 2;
}
[[ "$CANDIDATE_REPOSITORY_REF" == "refs/heads/codex/position-identity-v1" ]] || {
  echo "formal candidate ref is not canonical" >&2; exit 2;
}
: "${FORMAL_SOURCE_COMMIT_SHA:?}"
: "${FORMAL_SOURCE_TREE_SHA:?}"
: "${FORMAL_ENVIRONMENT_SHA256:?}"
: "${FORMAL_EXPECTED_GPU_MODEL:?}"
: "${FORMAL_EXPECTED_DRIVER_VERSION:?}"
: "${FORMAL_GIT_SHA256:?}"
: "${FORMAL_REPOSITORY_IDENTITY_SHA256:?}"
: "${FORMAL_REPOSITORY_QUERY_SHA256:?}"
: "${FORMAL_JOB_WORK_ROOT:?}"
: "${FORMAL_ARTIFACT_ROOT:?}"
: "${FORMAL_SUBMISSION_EXACT_ROOT:?}"
: "${PYTHON:?}"
: "${GIT:?GIT must name the trusted absolute git executable}"
[[ "$PYTHON" == /* && -x "$PYTHON" && "$GIT" == /* ]] || {
  echo "invalid trusted production bootstrap executable" >&2; exit 2;
}
if [[ -z "${FORMAL_PRODUCTION_STATIC_BOOTSTRAPPED:-}" ]]; then
  exec "$PYTHON" -I -B - "${BASH_SOURCE[0]}" <<'PY'
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
HELPERS = {
    "FORMAL_PRODUCTION_NVIDIA_LAUNCHER_FD": "exec_digest_bound_nvidia_smi.py",
    "FORMAL_PRODUCTION_VERIFY_GIT_FD": "verify_git_repository.py",
    "FORMAL_PRODUCTION_VERIFY_FILESYSTEM_FD": "verify_filesystem_isolation.py",
    "FORMAL_PRODUCTION_VERIFY_ENVIRONMENT_FD": "verify_formal_environment.py",
}


def fail(message):
    print(f"trusted static production bootstrap failed: {message}", file=sys.stderr)
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
        fail("bootstrap path is not normalized absolute")
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


def open_file(path=None, *, parent_fd=None, name=None):
    owned_parent = None
    if path is not None:
        owned_parent = open_directory(path.parent)
        parent_fd = owned_parent
        name = path.name
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        fd = os.open(name, flags, dir_fd=parent_fd)
    finally:
        if owned_parent is not None:
            os.close(owned_parent)
    opened = os.fstat(fd)
    if (
        signature(opened) != signature(before)
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_size > MAX_BYTES
    ):
        os.close(fd)
        fail("bootstrap file is not stable, bounded, and regular")
    return fd, opened


def read_stable(fd, opened):
    chunks = []
    digest = hashlib.sha256()
    remaining = opened.st_size
    while remaining:
        chunk = os.read(fd, min(1 << 20, remaining))
        if not chunk:
            fail("bootstrap file was truncated")
        chunks.append(chunk)
        digest.update(chunk)
        remaining -= len(chunk)
    if signature(os.fstat(fd)) != signature(opened):
        fail("bootstrap file changed while hashing")
    os.lseek(fd, 0, os.SEEK_SET)
    return b"".join(chunks), digest.hexdigest()


root = Path(os.environ["FORMAL_SUBMISSION_EXACT_ROOT"])
git_path = Path(os.environ["GIT"])
git_digest = os.environ["FORMAL_GIT_SHA256"]
commit = os.environ["FORMAL_SOURCE_COMMIT_SHA"]
tree = os.environ["FORMAL_SOURCE_TREE_SHA"]
spool_script = Path(sys.argv[1])
if (
    HEX64.fullmatch(git_digest) is None
    or HEX40.fullmatch(commit) is None
    or HEX40.fullmatch(tree) is None
    or not spool_script.is_absolute()
):
    fail("bootstrap digest, source identity, or spool path is malformed")

git_fd, git_stat = open_file(git_path)
helper_fds = {}
try:
    _, observed_git_digest = read_stable(git_fd, git_stat)
    if git_stat.st_mode & 0o111 == 0 or observed_git_digest != git_digest:
        fail("Git executable or digest mismatch")
    scripts_fd = open_directory(root / "scripts")
    try:
        helper_raw = {}
        for environment_name, filename in HELPERS.items():
            fd, info = open_file(parent_fd=scripts_fd, name=filename)
            raw, _ = read_stable(fd, info)
            helper_fds[environment_name] = fd
            helper_raw[filename] = raw
    finally:
        os.close(scripts_fd)

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
            fail("Git output exceeds its bootstrap ceiling")
        if check and result.returncode != 0:
            fail("Git bootstrap query failed")
        return result

    if run_git("rev-parse", "HEAD").stdout.strip() != commit.encode():
        fail("exact-root commit mismatch")
    if run_git("rev-parse", "HEAD^{tree}").stdout.strip() != tree.encode():
        fail("exact-root tree mismatch")
    if run_git("status", "--porcelain=v1", "--untracked-files=all").stdout:
        fail("exact root is dirty")
    if run_git("symbolic-ref", "-q", "HEAD", check=False).returncode != 1:
        fail("exact root is not detached")
    for filename, raw in helper_raw.items():
        committed = run_git("show", f"{commit}:scripts/{filename}").stdout
        if committed != raw:
            fail(f"exact helper differs from committed source: {filename}")

    os.set_inheritable(git_fd, True)
    for fd in helper_fds.values():
        os.set_inheritable(fd, True)
    environment = dict(os.environ)
    environment["FORMAL_PRODUCTION_STATIC_BOOTSTRAPPED"] = "1"
    environment["FORMAL_PRODUCTION_STATIC_OWNER_PID"] = str(os.getpid())
    environment["FORMAL_GIT_FD"] = str(git_fd)
    environment["FORMAL_GIT_FD_OWNER_PID"] = str(os.getpid())
    environment["FORMAL_GIT_COMMAND"] = git_command
    for name, fd in helper_fds.items():
        environment[name] = str(fd)
    os.execve("/bin/bash", ["/bin/bash", os.fspath(spool_script)], environment)
finally:
    os.close(git_fd)
    for fd in helper_fds.values():
        os.close(fd)
PY
fi
[[ "$FORMAL_PRODUCTION_STATIC_BOOTSTRAPPED" == 1 ]] || {
  echo "trusted production bootstrap marker is missing" >&2; exit 2;
}
[[ "$FORMAL_PRODUCTION_STATIC_OWNER_PID" == "$BASHPID" ]] || {
  echo "trusted production bootstrap owner differs" >&2; exit 2;
}
[[ "$FORMAL_GIT_COMMAND" == "/proc/self/fd/$FORMAL_GIT_FD" ]] || {
  echo "trusted production Git descriptor binding differs" >&2; exit 2;
}
for FD_NAME in FORMAL_PRODUCTION_NVIDIA_LAUNCHER_FD \
  FORMAL_PRODUCTION_VERIFY_GIT_FD FORMAL_PRODUCTION_VERIFY_FILESYSTEM_FD \
  FORMAL_PRODUCTION_VERIFY_ENVIRONMENT_FD; do
  [[ "${!FD_NAME}" =~ ^[0-9]+$ ]] || {
    echo "trusted production helper descriptor is invalid" >&2; exit 2;
  }
done
readonly FORMAL_GIT_FD FORMAL_GIT_FD_OWNER_PID FORMAL_GIT_COMMAND
hermetic_git() {
  env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C \
    GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_COUNT=0 \
    GIT_CEILING_DIRECTORIES=/ GIT_TERMINAL_PROMPT=0 \
    GIT_PROTOCOL_FROM_USER=0 GIT_ALLOW_PROTOCOL=https \
    "$FORMAL_GIT_COMMAND" \
      -c core.fsmonitor=false \
      -c core.hooksPath=/dev/null \
      -c core.filemode=true "$@"
}
: "${NVIDIA_SMI:?NVIDIA_SMI must name the trusted absolute executable}"
: "${FORMAL_NVIDIA_SMI_SHA256:?FORMAL_NVIDIA_SMI_SHA256 is required}"
if [[ -z "${FORMAL_NVIDIA_SMI_FD:-}" ]]; then
  [[ "$NVIDIA_SMI" == /* ]] || { echo "invalid NVIDIA_SMI" >&2; exit 2; }
  exec "$PYTHON" -I -B \
    "/proc/self/fd/$FORMAL_PRODUCTION_NVIDIA_LAUNCHER_FD" \
    --nvidia-smi "$NVIDIA_SMI" \
    --expected-sha256 "$FORMAL_NVIDIA_SMI_SHA256" -- \
    "${BASH_SOURCE[0]}"
fi
[[ "$NVIDIA_SMI" == "/proc/self/fd/$FORMAL_NVIDIA_SMI_FD" ]] || {
  echo "NVIDIA_SMI is not the verified open descriptor" >&2; exit 2;
}
[[ "${FORMAL_NVIDIA_SMI_FD_OWNER_PID:-}" == "$BASHPID" ]] || {
  echo "NVIDIA_SMI descriptor owner differs from production wrapper" >&2; exit 2;
}
export PYTHONNOUSERSITE=1 RUN_POLICY=fresh

[[ "$FORMAL_SUBMISSION_EXACT_ROOT" == /* ]] || {
  echo "invalid formal exact-root bootstrap binding" >&2
  exit 2
}
"$PYTHON" -I -B \
  "/proc/self/fd/$FORMAL_PRODUCTION_VERIFY_GIT_FD" \
  --git "$GIT" \
  --git-sha256 "$FORMAL_GIT_SHA256" \
  --expected-commit-sha "$FORMAL_SOURCE_COMMIT_SHA" \
  --expected-identity-sha256 "$FORMAL_REPOSITORY_IDENTITY_SHA256" \
  --expected-query-sha256 "$FORMAL_REPOSITORY_QUERY_SHA256" >/dev/null
[[ "$(hermetic_git -C "$FORMAL_SUBMISSION_EXACT_ROOT" rev-parse HEAD)" == "$FORMAL_SOURCE_COMMIT_SHA" ]]
[[ "$(hermetic_git -C "$FORMAL_SUBMISSION_EXACT_ROOT" rev-parse 'HEAD^{tree}')" == "$FORMAL_SOURCE_TREE_SHA" ]]
[[ -z "$(hermetic_git -C "$FORMAL_SUBMISSION_EXACT_ROOT" status --porcelain=v1 --untracked-files=all)" ]]
if hermetic_git -C "$FORMAL_SUBMISSION_EXACT_ROOT" symbolic-ref -q HEAD >/dev/null; then
  echo "formal submission exact root must remain detached" >&2
  exit 2
fi

: "${CUDA_VISIBLE_DEVICES:?Slurm must provide exactly one CUDA_VISIBLE_DEVICES token}"
case "$CUDA_VISIBLE_DEVICES" in
  *,*|''|*[!A-Za-z0-9_.:-]*) echo "invalid CUDA_VISIBLE_DEVICES allocation" >&2; exit 2 ;;
esac
GPU_ROWS="$(
  "$PYTHON" -I -B "/proc/self/fd/$FORMAL_PRODUCTION_NVIDIA_LAUNCHER_FD" \
    --retained-fd-owner-pid "$FORMAL_NVIDIA_SMI_FD_OWNER_PID" \
    --retained-fd "$FORMAL_NVIDIA_SMI_FD" \
    --expected-sha256 "$FORMAL_NVIDIA_SMI_SHA256" \
    --query-token "$CUDA_VISIBLE_DEVICES" \
    --query-fields name,uuid,driver_version
)"
VISIBLE_GPUS="$(printf '%s\n' "$GPU_ROWS" | sed '/^[[:space:]]*$/d' | wc -l | tr -d '[:space:]')"
[[ "$VISIBLE_GPUS" -eq 1 ]] || { echo "production job requires exactly one visible GPU" >&2; exit 1; }
GPU_NAME="${GPU_ROWS%%,*}"
GPU_UUID_AND_DRIVER="${GPU_ROWS#*,}"
GPU_UUID="${GPU_UUID_AND_DRIVER%%,*}"
GPU_DRIVER="${GPU_UUID_AND_DRIVER#*,}"
GPU_NAME="${GPU_NAME#${GPU_NAME%%[![:space:]]*}}"; GPU_NAME="${GPU_NAME%${GPU_NAME##*[![:space:]]}}"
GPU_UUID="${GPU_UUID#${GPU_UUID%%[![:space:]]*}}"; GPU_UUID="${GPU_UUID%${GPU_UUID##*[![:space:]]}}"
GPU_DRIVER="${GPU_DRIVER#${GPU_DRIVER%%[![:space:]]*}}"; GPU_DRIVER="${GPU_DRIVER%${GPU_DRIVER##*[![:space:]]}}"
[[ "$GPU_NAME" == "$FORMAL_EXPECTED_GPU_MODEL" ]] || { echo "production GPU model differs from H100 gate" >&2; exit 1; }
[[ "$GPU_DRIVER" == "$FORMAL_EXPECTED_DRIVER_VERSION" ]] || { echo "production GPU driver differs from H100 gate" >&2; exit 1; }
[[ "$GPU_UUID" == GPU-* || "$GPU_UUID" == MIG-* ]] || { echo "invalid allocated GPU UUID" >&2; exit 1; }
export FORMAL_VISIBLE_GPU_NAME="$GPU_NAME" FORMAL_VISIBLE_GPU_UUID="$GPU_UUID"
export FORMAL_VISIBLE_GPU_DRIVER_VERSION="$GPU_DRIVER"

"$PYTHON" -I -B \
  "/proc/self/fd/$FORMAL_PRODUCTION_VERIFY_FILESYSTEM_FD" \
  --work-root "$FORMAL_JOB_WORK_ROOT" \
  --artifact-root "$FORMAL_ARTIFACT_ROOT" \
  --work-label "formal job work root" \
  --artifact-label "formal artifact filesystem" >/dev/null
"$PYTHON" -I -B \
  "/proc/self/fd/$FORMAL_PRODUCTION_VERIFY_ENVIRONMENT_FD" \
  --exact-root "$FORMAL_SUBMISSION_EXACT_ROOT" \
  --expected-sha256 "$FORMAL_ENVIRONMENT_SHA256" \
  --expected-gpus 1 >/dev/null

CHECKOUT_PARENT="$(mktemp -d "$FORMAL_JOB_WORK_ROOT/${MODE}-stage${STAGE}.XXXXXX")"
CHECKOUT="$CHECKOUT_PARENT/candidate"
cleanup() { rm -rf "$CHECKOUT_PARENT"; }
trap cleanup EXIT HUP INT TERM
hermetic_git -C "$CHECKOUT_PARENT" clone --quiet --no-hardlinks --no-checkout -- \
  "$CANDIDATE_REPOSITORY" candidate
[[ "$(hermetic_git -C "$CHECKOUT" remote get-url origin)" == "$CANDIDATE_REPOSITORY" ]]
[[ "$(hermetic_git -C "$CHECKOUT" rev-parse refs/remotes/origin/codex/position-identity-v1)" == "$FORMAL_SOURCE_COMMIT_SHA" ]]
hermetic_git -C "$CHECKOUT" checkout --quiet --detach "$FORMAL_SOURCE_COMMIT_SHA"
[[ "$(hermetic_git -C "$CHECKOUT" rev-parse HEAD)" == "$FORMAL_SOURCE_COMMIT_SHA" ]]
[[ "$(hermetic_git -C "$CHECKOUT" rev-parse 'HEAD^{tree}')" == "$FORMAL_SOURCE_TREE_SHA" ]]
[[ -z "$(hermetic_git -C "$CHECKOUT" status --porcelain=v1 --untracked-files=all)" ]]
if hermetic_git -C "$CHECKOUT" symbolic-ref -q HEAD >/dev/null; then
  echo "formal checkout must be detached" >&2
  exit 1
fi
export TABICL_EXACT_ROOT="$CHECKOUT" PYTHONPATH="$CHECKOUT/src"

# Production jobs intentionally do not write one-second GPU CSVs for hours or
# days.  The six bounded max-sequence jobs own the >=80% utilization claim;
# production observability uses scheduler/log/checkpoint state through the
# independent read-only monitor, with no utilization claim from production.
# Slurm owns the two bounded external bootstrap logs.  Once checkout succeeds,
# the long-running exact-T runner writes only the bounded stage logs and its
# durable completion evidence; no further bytes are sent to the spool handles.
exec >/dev/null 2>&1
set +e
"$PYTHON" -I -B "$CHECKOUT/scripts/run_formal_identity_production_job.py" \
  --exact-root "$CHECKOUT" --mode "$MODE" --stage "$STAGE"
STATUS=$?
set -e
exit "$STATUS"
