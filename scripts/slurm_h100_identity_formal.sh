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
: "${FORMAL_GIT_SHA256:?}"
: "${FORMAL_REPOSITORY_IDENTITY_SHA256:?}"
: "${FORMAL_REPOSITORY_QUERY_SHA256:?}"
: "${FORMAL_JOB_WORK_ROOT:?}"
: "${FORMAL_ARTIFACT_ROOT:?}"
: "${FORMAL_SUBMISSION_EXACT_ROOT:?}"
: "${PYTHON:?}"
: "${GIT:?GIT must name the trusted absolute git executable}"
[[ "$GIT" == /* && -x "$GIT" ]] || { echo "invalid GIT" >&2; exit 2; }
hermetic_git() {
  env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C \
    GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_COUNT=0 \
    GIT_CEILING_DIRECTORIES=/ GIT_TERMINAL_PROMPT=0 \
    GIT_PROTOCOL_FROM_USER=0 GIT_ALLOW_PROTOCOL=https \
    "$GIT" "$@"
}
: "${NVIDIA_SMI:?NVIDIA_SMI must name the trusted absolute executable}"
[[ "$NVIDIA_SMI" == /* && -x "$NVIDIA_SMI" ]] || { echo "invalid NVIDIA_SMI" >&2; exit 2; }
export PYTHONNOUSERSITE=1 RUN_POLICY=fresh

[[ "$FORMAL_SUBMISSION_EXACT_ROOT" == /* ]] || {
  echo "invalid formal exact-root bootstrap binding" >&2
  exit 2
}
"$PYTHON" -I -B \
  "$FORMAL_SUBMISSION_EXACT_ROOT/scripts/verify_git_repository.py" \
  --git "$GIT" \
  --git-sha256 "$FORMAL_GIT_SHA256" \
  --expected-commit-sha "$FORMAL_SOURCE_COMMIT_SHA" \
  --expected-identity-sha256 "$FORMAL_REPOSITORY_IDENTITY_SHA256" \
  --expected-query-sha256 "$FORMAL_REPOSITORY_QUERY_SHA256" >/dev/null
[[ "$(hermetic_git -C "$FORMAL_SUBMISSION_EXACT_ROOT" rev-parse HEAD)" == "$FORMAL_SOURCE_COMMIT_SHA" ]]
[[ "$(hermetic_git -C "$FORMAL_SUBMISSION_EXACT_ROOT" rev-parse 'HEAD^{tree}')" == "$FORMAL_SOURCE_TREE_SHA" ]]
[[ -z "$(hermetic_git -C "$FORMAL_SUBMISSION_EXACT_ROOT" status --porcelain=v1 --untracked-files=all)" ]]

: "${CUDA_VISIBLE_DEVICES:?Slurm must provide exactly one CUDA_VISIBLE_DEVICES token}"
case "$CUDA_VISIBLE_DEVICES" in
  *,*|''|*[!A-Za-z0-9_.:-]*) echo "invalid CUDA_VISIBLE_DEVICES allocation" >&2; exit 2 ;;
esac
GPU_ROWS="$("$NVIDIA_SMI" --id="$CUDA_VISIBLE_DEVICES" --query-gpu=name,uuid --format=csv,noheader,nounits)"
VISIBLE_GPUS="$(printf '%s\n' "$GPU_ROWS" | sed '/^[[:space:]]*$/d' | wc -l | tr -d '[:space:]')"
[[ "$VISIBLE_GPUS" -eq 1 ]] || { echo "production job requires exactly one visible GPU" >&2; exit 1; }
GPU_NAME="${GPU_ROWS%%,*}"
GPU_UUID="${GPU_ROWS#*,}"
GPU_NAME="${GPU_NAME#${GPU_NAME%%[![:space:]]*}}"; GPU_NAME="${GPU_NAME%${GPU_NAME##*[![:space:]]}}"
GPU_UUID="${GPU_UUID#${GPU_UUID%%[![:space:]]*}}"; GPU_UUID="${GPU_UUID%${GPU_UUID##*[![:space:]]}}"
[[ "$GPU_NAME" == *H100* ]] || { echo "production job requires an NVIDIA H100" >&2; exit 1; }
[[ "$GPU_UUID" == GPU-* || "$GPU_UUID" == MIG-* ]] || { echo "invalid allocated GPU UUID" >&2; exit 1; }
export FORMAL_VISIBLE_GPU_NAME="$GPU_NAME" FORMAL_VISIBLE_GPU_UUID="$GPU_UUID"

"$PYTHON" -I -B \
  "$FORMAL_SUBMISSION_EXACT_ROOT/scripts/verify_filesystem_isolation.py" \
  --work-root "$FORMAL_JOB_WORK_ROOT" \
  --artifact-root "$FORMAL_ARTIFACT_ROOT" \
  --work-label "formal job work root" \
  --artifact-label "formal artifact filesystem" >/dev/null

CHECKOUT_PARENT="$(mktemp -d "$FORMAL_JOB_WORK_ROOT/${MODE}-stage${STAGE}.XXXXXX")"
CHECKOUT="$CHECKOUT_PARENT/candidate"
cleanup() { rm -rf "$CHECKOUT_PARENT"; }
trap cleanup EXIT INT TERM
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
