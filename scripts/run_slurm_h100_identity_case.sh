#!/usr/bin/env bash
set -euo pipefail

EXPECTED_GPUS="${1:-}"
case "$EXPECTED_GPUS" in 1|2) ;; *) echo "expected GPU count must be 1 or 2" >&2; exit 2 ;; esac
[[ "${PATH:-}" == "/usr/bin:/bin" ]] || {
  echo "validation batch PATH must be exactly /usr/bin:/bin" >&2
  exit 2
}
: "${FORMAL_EXPECTED_GPUS:?}"
[[ "$FORMAL_EXPECTED_GPUS" == "$EXPECTED_GPUS" ]] || {
  echo "wrapper GPU count differs from the submitted resource contract" >&2
  exit 2
}
[[ "${FORMAL_REQUESTED_GPUS:-}" == "$EXPECTED_GPUS" ]] || {
  echo "requested GPU resource differs from wrapper" >&2; exit 2;
}
: "${VALIDATION_CASE_ID:?}"
case "$VALIDATION_CASE_ID" in
  nccl_2gpu) CASE_GPUS=2 ;;
  stage1_rope_one_step|stage1_temporary_one_step|stage1_none_one_step|\
  stage2_rope_maxseq10240|stage2_temporary_maxseq10240|stage2_none_maxseq10240|\
  stage3_rope_maxseq60000_recompute|stage3_temporary_maxseq60000_recompute|stage3_none_maxseq60000_recompute|\
  temporary_cuda_rng_resume|prior_dataloader_resume) CASE_GPUS=1 ;;
  *) echo "unknown validation case" >&2; exit 2 ;;
esac
[[ "$CASE_GPUS" == "$EXPECTED_GPUS" ]] || {
  echo "validation case was submitted through the wrong GPU wrapper" >&2
  exit 2
}

for NAME in CANDIDATE_REPOSITORY CANDIDATE_REPOSITORY_REF \
  FORMAL_SOURCE_COMMIT_SHA FORMAL_SOURCE_TREE_SHA \
  FORMAL_SOURCE_MANIFEST FORMAL_SOURCE_SHA256 FORMAL_EXPECTED_ENVIRONMENT_SHA256 \
  FORMAL_GIT_SHA256 FORMAL_REPOSITORY_IDENTITY_SHA256 \
  FORMAL_REPOSITORY_QUERY_SHA256 \
  VALIDATION_ARTIFACT_IDENTITY_SHA256 H100_CASE_WORK_ROOT H100_VALIDATION_ROOT \
  CHECKPOINT_CEILING_BYTES FORMAL_RUN_LOG_CEILING_BYTES \
  FORMAL_GPU_MONITOR_CEILING_BYTES FORMAL_ATTESTATION_CEILING_BYTES PYTHON GIT \
  NVIDIA_SMI RUN_POLICY FORMAL_SUBMISSION_EXACT_ROOT SLURM_JOB_ID \
  CUDA_VISIBLE_DEVICES FORMAL_REQUESTED_RESOURCE_SHA256 \
  FORMAL_REQUESTED_PARTITION FORMAL_REQUESTED_QOS FORMAL_REQUESTED_TIME_LIMIT \
  FORMAL_REQUESTED_NODES FORMAL_REQUESTED_GPUS FORMAL_REQUESTED_CPUS \
  FORMAL_REQUESTED_MEMORY_MB SLURM_JOB_PARTITION SLURM_CPUS_PER_TASK \
  SLURM_MEM_PER_NODE SLURM_CLUSTER_NAME; do
  [[ -n "${!NAME:-}" ]] || { echo "$NAME is required" >&2; exit 2; }
done
[[ "$RUN_POLICY" == "fresh" ]] || { echo "H100 validation cases are fresh-only" >&2; exit 2; }
[[ "$GIT" == /* && -x "$GIT" ]] || { echo "invalid GIT" >&2; exit 2; }
hermetic_git() {
  env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C \
    GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_COUNT=0 \
    GIT_CEILING_DIRECTORIES=/ GIT_TERMINAL_PROMPT=0 \
    GIT_PROTOCOL_FROM_USER=0 GIT_ALLOW_PROTOCOL=https \
    "$GIT" "$@"
}
[[ "$PYTHON" == /* && -x "$PYTHON" ]] || { echo "invalid PYTHON" >&2; exit 2; }
[[ "$NVIDIA_SMI" == /* && -x "$NVIDIA_SMI" ]] || { echo "invalid NVIDIA_SMI" >&2; exit 2; }
export PYTHONNOUSERSITE=1

[[ "${BASH_SOURCE[0]}" == "$FORMAL_SUBMISSION_EXACT_ROOT/scripts/run_slurm_h100_identity_case.sh" ]] || {
  echo "validation runner is not the exact-root absolute bootstrap" >&2
  exit 2
}
[[ "$CANDIDATE_REPOSITORY" == "https://github.com/kikixiong/tabicl-pe.git" ]] || {
  echo "validation candidate repository is not canonical" >&2
  exit 2
}
[[ "$CANDIDATE_REPOSITORY_REF" == "refs/heads/codex/position-identity-v1" ]] || {
  echo "validation candidate ref is not canonical" >&2
  exit 2
}
"$PYTHON" -I -B \
  "$FORMAL_SUBMISSION_EXACT_ROOT/scripts/verify_git_repository.py" \
  --git "$GIT" \
  --git-sha256 "$FORMAL_GIT_SHA256" \
  --expected-commit-sha "$FORMAL_SOURCE_COMMIT_SHA" \
  --expected-identity-sha256 "$FORMAL_REPOSITORY_IDENTITY_SHA256" \
  --expected-query-sha256 "$FORMAL_REPOSITORY_QUERY_SHA256" >/dev/null

[[ "$SLURM_JOB_ID" =~ ^[1-9][0-9]{0,19}$ ]] || { echo "invalid SLURM_JOB_ID" >&2; exit 2; }
IFS=',' read -r -a GPU_TOKENS <<<"$CUDA_VISIBLE_DEVICES"
[[ "${#GPU_TOKENS[@]}" -eq "$EXPECTED_GPUS" ]] || {
  echo "CUDA_VISIBLE_DEVICES count differs from requested GPUs" >&2
  exit 1
}
GPU_UUIDS=()
declare -A SEEN_GPU_UUIDS=()
for GPU_TOKEN in "${GPU_TOKENS[@]}"; do
  [[ "$GPU_TOKEN" =~ ^([0-9]+|GPU-[A-Za-z0-9._-]+|MIG-[A-Za-z0-9._-]+)$ ]] || {
    echo "invalid CUDA_VISIBLE_DEVICES token" >&2; exit 1;
  }
  mapfile -t TOKEN_ROWS < <(
    "$NVIDIA_SMI" --id="$GPU_TOKEN" \
      --query-gpu=uuid,name,driver_version --format=csv,noheader,nounits |
      sed '/^[[:space:]]*$/d'
  )
  [[ "${#TOKEN_ROWS[@]}" -eq 1 ]] || {
    echo "CUDA-visible token did not resolve to exactly one GPU" >&2; exit 1;
  }
  IFS=',' read -r GPU_UUID GPU_NAME GPU_DRIVER <<<"${TOKEN_ROWS[0]}"
  GPU_UUID="${GPU_UUID//[[:space:]]/}"
  [[ -n "$GPU_UUID" && "$GPU_NAME" == *H100* && -n "$GPU_DRIVER" ]] || {
    echo "CUDA-visible allocation contains a non-H100 or malformed GPU row" >&2
    exit 1
  }
  [[ -z "${SEEN_GPU_UUIDS[$GPU_UUID]+x}" ]] || {
    echo "CUDA-visible GPU UUIDs are not unique" >&2; exit 1;
  }
  SEEN_GPU_UUIDS["$GPU_UUID"]=1
  GPU_UUIDS+=("$GPU_UUID")
done
export FORMAL_VISIBLE_GPU_TOKENS="$CUDA_VISIBLE_DEVICES"
export FORMAL_VISIBLE_GPU_UUIDS="$(IFS=,; echo "${GPU_UUIDS[*]}")"

"$PYTHON" -I -B \
  "$FORMAL_SUBMISSION_EXACT_ROOT/scripts/verify_filesystem_isolation.py" \
  --work-root "$H100_CASE_WORK_ROOT" \
  --artifact-root "$H100_VALIDATION_ROOT" \
  --work-label "H100 case work root" \
  --artifact-label "H100 artifact filesystem" >/dev/null

CHECKOUT_PARENT="$(mktemp -d "$H100_CASE_WORK_ROOT/${VALIDATION_CASE_ID}.XXXXXX")"
CHECKOUT="$CHECKOUT_PARENT/candidate"
cleanup() { rm -rf -- "$CHECKOUT_PARENT"; }
trap 'status=$?; trap - EXIT INT TERM; cleanup; exit "$status"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
hermetic_git -C "$CHECKOUT_PARENT" clone --quiet --no-hardlinks --no-checkout -- \
  "$CANDIDATE_REPOSITORY" candidate
[[ "$(hermetic_git -C "$CHECKOUT" remote get-url origin)" == "$CANDIDATE_REPOSITORY" ]]
[[ "$(hermetic_git -C "$CHECKOUT" rev-parse refs/remotes/origin/codex/position-identity-v1)" == "$FORMAL_SOURCE_COMMIT_SHA" ]]
hermetic_git -C "$CHECKOUT" checkout --quiet --detach "$FORMAL_SOURCE_COMMIT_SHA"
[[ "$(hermetic_git -C "$CHECKOUT" rev-parse HEAD)" == "$FORMAL_SOURCE_COMMIT_SHA" ]]
[[ "$(hermetic_git -C "$CHECKOUT" rev-parse 'HEAD^{tree}')" == "$FORMAL_SOURCE_TREE_SHA" ]]
[[ -z "$(hermetic_git -C "$CHECKOUT" status --porcelain=v1 --untracked-files=all)" ]]
if hermetic_git -C "$CHECKOUT" symbolic-ref -q HEAD >/dev/null; then
  echo "validation checkout must be detached" >&2
  exit 1
fi

export TABICL_EXACT_ROOT="$CHECKOUT"
export PYTHONPATH="$CHECKOUT/src"
"$CHECKOUT/scripts/run_h100_identity_maxseq_smoke.sh" "$VALIDATION_CASE_ID"
