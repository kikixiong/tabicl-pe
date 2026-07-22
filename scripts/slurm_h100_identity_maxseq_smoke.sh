#!/usr/bin/env bash
#SBATCH --job-name=tabicl-identity-gate
#SBATCH --partition=h100
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

set -euo pipefail
: "${VALIDATION_CASE_ID:?}"
: "${CANDIDATE_REPOSITORY:?}"
: "${FORMAL_SOURCE_COMMIT_SHA:?}"
: "${FORMAL_SOURCE_TREE_SHA:?}"
: "${H100_CASE_WORK_ROOT:?}"
: "${H100_VALIDATION_ROOT:?}"
: "${PYTHON:?}"
: "${GIT:?GIT must name the trusted absolute git executable}"
[[ "$GIT" == /* && -x "$GIT" ]] || { echo "invalid GIT" >&2; exit 2; }
: "${NVIDIA_SMI:?NVIDIA_SMI must name the trusted absolute executable}"
[[ "$NVIDIA_SMI" == /* && -x "$NVIDIA_SMI" ]] || { echo "invalid NVIDIA_SMI" >&2; exit 2; }
export PYTHONNOUSERSITE=1

case "$VALIDATION_CASE_ID" in
  nccl_2gpu) EXPECTED_GPUS=2 ;;
  stage1_rope_one_step|stage1_temporary_one_step|stage1_none_one_step|\
  stage2_rope_maxseq10240|stage2_temporary_maxseq10240|stage2_none_maxseq10240|\
  stage3_rope_maxseq60000_recompute|stage3_temporary_maxseq60000_recompute|stage3_none_maxseq60000_recompute|\
  temporary_cuda_rng_resume|prior_dataloader_resume) EXPECTED_GPUS=1 ;;
  *) echo "unknown validation case" >&2; exit 2 ;;
esac
VISIBLE_GPUS="$("$NVIDIA_SMI" --query-gpu=uuid --format=csv,noheader,nounits | wc -l | tr -d '[:space:]')"
[[ "$VISIBLE_GPUS" -eq "$EXPECTED_GPUS" ]] || {
  echo "validation case requires exactly $EXPECTED_GPUS visible H100 GPU(s)" >&2; exit 1;
}

CHECKOUT_PARENT="$(mktemp -d "$H100_CASE_WORK_ROOT/${VALIDATION_CASE_ID}.XXXXXX")"
CHECKOUT="$CHECKOUT_PARENT/candidate"
cleanup() { rm -rf "$CHECKOUT_PARENT"; }
trap cleanup EXIT INT TERM
"$GIT" clone --quiet --no-hardlinks --no-checkout "$CANDIDATE_REPOSITORY" "$CHECKOUT"
"$GIT" -C "$CHECKOUT" checkout --quiet --detach "$FORMAL_SOURCE_COMMIT_SHA"
[[ "$("$GIT" -C "$CHECKOUT" rev-parse HEAD)" == "$FORMAL_SOURCE_COMMIT_SHA" ]]
[[ "$("$GIT" -C "$CHECKOUT" rev-parse 'HEAD^{tree}')" == "$FORMAL_SOURCE_TREE_SHA" ]]
[[ -z "$("$GIT" -C "$CHECKOUT" status --porcelain=v1 --untracked-files=all)" ]]
if "$GIT" -C "$CHECKOUT" symbolic-ref -q HEAD >/dev/null; then
  echo "validation checkout must be detached" >&2
  exit 1
fi

export TABICL_EXACT_ROOT="$CHECKOUT"
export PYTHONPATH="$CHECKOUT/src"
exec "$CHECKOUT/scripts/run_h100_identity_maxseq_smoke.sh" "$VALIDATION_CASE_ID"
