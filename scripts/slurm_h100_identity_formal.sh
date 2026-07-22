#!/usr/bin/env bash
#SBATCH --job-name=tabicl-identity-formal
#SBATCH --partition=h100
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail
: "${MODE:?MODE must be rope, temporary, or none}"
: "${STAGE:?STAGE must be 1, 2, or 3}"
: "${NUM_GPUS:?NUM_GPUS must be exactly 1}"
case "$MODE" in rope|temporary|none) ;; *) echo "invalid MODE" >&2; exit 2 ;; esac
case "$STAGE" in 1|2|3) ;; *) echo "invalid STAGE" >&2; exit 2 ;; esac
[[ "$NUM_GPUS" -eq 1 ]] || { echo "all nine production jobs require exactly one GPU" >&2; exit 2; }
: "${CANDIDATE_REPOSITORY:?}"
: "${FORMAL_SOURCE_COMMIT_SHA:?}"
: "${FORMAL_SOURCE_TREE_SHA:?}"
: "${FORMAL_JOB_WORK_ROOT:?}"
: "${PYTHON:?}"
: "${GIT:?GIT must name the trusted absolute git executable}"
[[ "$GIT" == /* && -x "$GIT" ]] || { echo "invalid GIT" >&2; exit 2; }
: "${NVIDIA_SMI:?NVIDIA_SMI must name the trusted absolute executable}"
[[ "$NVIDIA_SMI" == /* && -x "$NVIDIA_SMI" ]] || { echo "invalid NVIDIA_SMI" >&2; exit 2; }
export PYTHONNOUSERSITE=1 RUN_POLICY=fresh

GPU_ROWS="$("$NVIDIA_SMI" --query-gpu=name,uuid --format=csv,noheader,nounits)"
VISIBLE_GPUS="$(printf '%s\n' "$GPU_ROWS" | sed '/^[[:space:]]*$/d' | wc -l | tr -d '[:space:]')"
[[ "$VISIBLE_GPUS" -eq 1 ]] || { echo "production job requires exactly one visible GPU" >&2; exit 1; }
GPU_NAME="${GPU_ROWS%%,*}"
[[ "$GPU_NAME" == *H100* ]] || { echo "production job requires an NVIDIA H100" >&2; exit 1; }

CHECKOUT_PARENT="$(mktemp -d "$FORMAL_JOB_WORK_ROOT/${MODE}-stage${STAGE}.XXXXXX")"
CHECKOUT="$CHECKOUT_PARENT/candidate"
cleanup() { rm -rf "$CHECKOUT_PARENT"; }
trap cleanup EXIT INT TERM
"$GIT" clone --quiet --no-hardlinks --no-checkout "$CANDIDATE_REPOSITORY" "$CHECKOUT"
"$GIT" -C "$CHECKOUT" checkout --quiet --detach "$FORMAL_SOURCE_COMMIT_SHA"
[[ "$("$GIT" -C "$CHECKOUT" rev-parse HEAD)" == "$FORMAL_SOURCE_COMMIT_SHA" ]]
[[ "$("$GIT" -C "$CHECKOUT" rev-parse 'HEAD^{tree}')" == "$FORMAL_SOURCE_TREE_SHA" ]]
[[ -z "$("$GIT" -C "$CHECKOUT" status --porcelain=v1 --untracked-files=all)" ]]
if "$GIT" -C "$CHECKOUT" symbolic-ref -q HEAD >/dev/null; then
  echo "formal checkout must be detached" >&2
  exit 1
fi
export TABICL_EXACT_ROOT="$CHECKOUT" PYTHONPATH="$CHECKOUT/src"

# Production jobs intentionally do not write one-second GPU CSVs for hours or
# days.  The six bounded max-sequence jobs own the >=80% utilization claim;
# production observability uses scheduler/log/checkpoint state through the
# independent read-only monitor, with no utilization claim from production.
exec "$CHECKOUT/scripts/formal_train_v2_clf_identity_stage${STAGE}.sh" "$MODE"
