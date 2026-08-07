#!/usr/bin/env bash
#SBATCH --job-name=tabicl-identity-gate-2g
#SBATCH --partition=h100
#SBATCH --qos=short
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G

set -euo pipefail

[[ "${FORMAL_EXPECTED_GPUS:-}" == "2" ]] || {
  echo "two-GPU validation wrapper requires FORMAL_EXPECTED_GPUS=2" >&2
  exit 2
}
[[ "${VALIDATION_CASE_ID:-}" == "nccl_2gpu" ]] || {
  echo "the two-GPU wrapper is reserved for nccl_2gpu" >&2
  exit 2
}
: "${FORMAL_SUBMISSION_EXACT_ROOT:?FORMAL_SUBMISSION_EXACT_ROOT is required}"
: "${GIT:?GIT is required}"
[[ "$FORMAL_SUBMISSION_EXACT_ROOT" == /* && "$GIT" == /* && -x "$GIT" ]] || {
  echo "invalid exact-root bootstrap contract" >&2
  exit 2
}
[[ "$("$GIT" -C "$FORMAL_SUBMISSION_EXACT_ROOT" rev-parse HEAD)" == "$FORMAL_SOURCE_COMMIT_SHA" ]]
[[ "$("$GIT" -C "$FORMAL_SUBMISSION_EXACT_ROOT" rev-parse 'HEAD^{tree}')" == "$FORMAL_SOURCE_TREE_SHA" ]]
[[ -z "$("$GIT" -C "$FORMAL_SUBMISSION_EXACT_ROOT" status --porcelain=v1 --untracked-files=all)" ]]
exec "$FORMAL_SUBMISSION_EXACT_ROOT/scripts/run_slurm_h100_identity_case.sh" 2
