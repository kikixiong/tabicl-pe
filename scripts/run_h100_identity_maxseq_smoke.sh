#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--rank-worker" ]]; then
  shift
  CASE_ID="${1:?case ID required}"
  ROOT="${TABICL_EXACT_ROOT:?}"
  RANK_ID="${LOCAL_RANK:?}"
  exec "$PYTHON" -I -B "$ROOT/scripts/run_exact_tabicl.py" \
    --archive-root "$ROOT" --source-manifest "$FORMAL_SOURCE_MANIFEST" \
    --expected-manifest-sha256 "$FORMAL_SOURCE_SHA256" \
    --expected-commit-sha "$FORMAL_SOURCE_COMMIT_SHA" \
    --expected-tree-sha "$FORMAL_SOURCE_TREE_SHA" \
    --action h100-validation \
    --source-attestation-output "$CASE_ROOT/source-attestation.json" \
    --source-attestation-max-bytes "$FORMAL_ATTESTATION_CEILING_BYTES" \
    --runtime-evidence-output "$CASE_ROOT/runtime-evidence.json" \
    --completion-receipt-output "$CASE_ROOT/action-completion.json" \
    --case-id "$CASE_ID" --artifact-identity-sha256 "$VALIDATION_ARTIFACT_IDENTITY_SHA256" -- \
    --output-dir "$CASE_ROOT/compute" \
    --checkpoint-ceiling-bytes "$CHECKPOINT_CEILING_BYTES"
fi

CASE_ID="${1:-}"
case "$CASE_ID" in
  stage1_rope_one_step|stage1_temporary_one_step|stage1_none_one_step)
    UTILIZATION_REQUIRED=false; EXPECTED_GPUS=1 ;;
  stage2_rope_maxseq10240|stage2_temporary_maxseq10240|stage2_none_maxseq10240|\
  stage3_rope_maxseq60000_recompute|stage3_temporary_maxseq60000_recompute|stage3_none_maxseq60000_recompute)
    UTILIZATION_REQUIRED=true; EXPECTED_GPUS=1 ;;
  temporary_cuda_rng_resume|prior_dataloader_resume)
    UTILIZATION_REQUIRED=false; EXPECTED_GPUS=1 ;;
  nccl_2gpu)
    UTILIZATION_REQUIRED=false; EXPECTED_GPUS=2 ;;
  *) echo "unknown H100 validation case: $CASE_ID" >&2; exit 2 ;;
esac

ROOT="${TABICL_EXACT_ROOT:?}"; ROOT="$(cd "$ROOT" && pwd -P)"
: "${PYTHON:?}"; [[ "$PYTHON" == /* && -x "$PYTHON" ]] || { echo "invalid PYTHON" >&2; exit 2; }
: "${NVIDIA_SMI:?}"; [[ "$NVIDIA_SMI" == /* && -x "$NVIDIA_SMI" ]] || { echo "invalid NVIDIA_SMI" >&2; exit 2; }
export PYTHONPATH="$ROOT/src" PYTHONNOUSERSITE=1
for NAME in FORMAL_SOURCE_MANIFEST FORMAL_SOURCE_SHA256 FORMAL_SOURCE_COMMIT_SHA \
  FORMAL_SOURCE_TREE_SHA VALIDATION_ARTIFACT_IDENTITY_SHA256 \
  H100_VALIDATION_ROOT FORMAL_RUN_LOG_CEILING_BYTES FORMAL_GPU_MONITOR_CEILING_BYTES \
  FORMAL_ATTESTATION_CEILING_BYTES; do
  [[ -n "${!NAME:-}" ]] || { echo "$NAME is required" >&2; exit 2; }
done
: "${CHECKPOINT_CEILING_BYTES:?CHECKPOINT_CEILING_BYTES is required}"

CASE_ROOT="$H100_VALIDATION_ROOT/$CASE_ID"
export CASE_ROOT TABICL_EXACT_ROOT="$ROOT" TABICL_ATTESTED_CASE_ID="$CASE_ID"
export TABICL_ATTESTED_ARTIFACT_SHA256="$VALIDATION_ARTIFACT_IDENTITY_SHA256"
export FORMAL_GPU_EXPECTED_COUNT="$EXPECTED_GPUS"
[[ ! -e "$CASE_ROOT" && ! -L "$CASE_ROOT" ]] || { echo "validation case is fresh-only" >&2; exit 1; }
mkdir -p "$CASE_ROOT"
CSV_PATH="$CASE_ROOT/gpu.csv"
JSONL_PATH="$CASE_ROOT/gpu.jsonl"
COMPUTE_LOG="$CASE_ROOT/compute.log"

if [[ "$CASE_ID" == "nccl_2gpu" ]]; then
  COMPUTE=(
    "$PYTHON" -I -B -m torch.distributed.run --standalone --nproc_per_node=2
    --no-python "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh"
    --rank-worker "$CASE_ID"
  )
else
  COMPUTE=(
    "$PYTHON" -I -B "$ROOT/scripts/run_exact_tabicl.py"
    --archive-root "$ROOT" --source-manifest "$FORMAL_SOURCE_MANIFEST"
    --expected-manifest-sha256 "$FORMAL_SOURCE_SHA256"
    --expected-commit-sha "$FORMAL_SOURCE_COMMIT_SHA"
    --expected-tree-sha "$FORMAL_SOURCE_TREE_SHA"
    --action h100-validation
    --source-attestation-output "$CASE_ROOT/source-attestation.json"
    --source-attestation-max-bytes "$FORMAL_ATTESTATION_CEILING_BYTES"
    --runtime-evidence-output "$CASE_ROOT/runtime-evidence.json"
    --completion-receipt-output "$CASE_ROOT/action-completion.json"
    --case-id "$CASE_ID" --artifact-identity-sha256 "$VALIDATION_ARTIFACT_IDENTITY_SHA256" --
    --output-dir "$CASE_ROOT/compute"
    --checkpoint-ceiling-bytes "$CHECKPOINT_CEILING_BYTES"
  )
fi

if [[ "$UTILIZATION_REQUIRED" == true ]]; then
  PYTHON="$PYTHON" "$ROOT/scripts/formal_run_with_gpu_monitor.sh" \
    "$CSV_PATH" "$JSONL_PATH" "$FORMAL_GPU_MONITOR_CEILING_BYTES" -- \
    env PYTHON="$PYTHON" "$ROOT/scripts/run_with_durable_log.sh" \
      "$COMPUTE_LOG" "$FORMAL_RUN_LOG_CEILING_BYTES" "${COMPUTE[@]}"
else
  [[ ! -e "$CSV_PATH" && ! -L "$CSV_PATH" && ! -e "$JSONL_PATH" && ! -L "$JSONL_PATH" ]] || {
    echo "functional case must not have GPU monitor artifacts" >&2; exit 1;
  }
  PYTHON="$PYTHON" "$ROOT/scripts/run_with_durable_log.sh" \
    "$COMPUTE_LOG" "$FORMAL_RUN_LOG_CEILING_BYTES" "${COMPUTE[@]}"
  [[ ! -e "$CSV_PATH" && ! -L "$CSV_PATH" && ! -e "$JSONL_PATH" && ! -L "$JSONL_PATH" ]] || {
    echo "functional case created unbound GPU monitor artifacts" >&2; exit 1;
  }
fi

"$PYTHON" -I -B "$ROOT/scripts/reject_nonfinite_log.py" \
  "$COMPUTE_LOG" --max-bytes "$FORMAL_RUN_LOG_CEILING_BYTES"

if [[ "$UTILIZATION_REQUIRED" == true ]]; then
  mapfile -t GPU_UUIDS < <("$NVIDIA_SMI" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${#GPU_UUIDS[@]}" -eq "$EXPECTED_GPUS" ]] || { echo "unexpected visible GPU count" >&2; exit 1; }
  SUMMARY_ARGS=()
  for GPU_UUID in "${GPU_UUIDS[@]}"; do SUMMARY_ARGS+=(--expected-gpu-uuid "$GPU_UUID"); done
  PYTHON="$PYTHON" "$ROOT/scripts/run_with_durable_log.sh" \
    "$CASE_ROOT/gpu-summary.json" "$FORMAL_RUN_LOG_CEILING_BYTES" \
    "$PYTHON" -I -B "$ROOT/scripts/summarize_formal_gpu_usage.py" "$CSV_PATH" \
      --utilization-required --expected-gpus "$EXPECTED_GPUS" "${SUMMARY_ARGS[@]}"
fi
