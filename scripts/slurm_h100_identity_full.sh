#!/usr/bin/env bash
#SBATCH --job-name=tabicl-identity-full
#SBATCH --partition=h100
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --output=artifacts/logs/%x-%j.out
#SBATCH --error=artifacts/logs/%x-%j.err

set -euo pipefail

: "${MODE:?MODE must be rope or none}"
: "${STAGE:?STAGE must be 1, 2, or 3}"
: "${NUM_GPUS:?NUM_GPUS must be 1, 2, or 4}"
: "${TABICL_SOURCE_SHA:?TABICL_SOURCE_SHA must be the submitted commit}"
case "$MODE" in rope|none) ;; *) echo "invalid MODE=$MODE" >&2; exit 2 ;; esac
case "$STAGE" in 1|2|3) ;; *) echo "invalid STAGE=$STAGE" >&2; exit 2 ;; esac
case "$NUM_GPUS" in 1|2|4) ;; *) echo "invalid NUM_GPUS=$NUM_GPUS" >&2; exit 2 ;; esac

ROOT="$(cd "${TABICL_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}" && pwd -P)"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$TABICL_SOURCE_SHA" ]] || {
  echo "source checkout does not match TABICL_SOURCE_SHA" >&2
  exit 1
}
[[ -z "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "source checkout is not clean" >&2
  exit 1
}
MONITOR_ROOT="${TABICL_RUN_ROOT:-$ROOT/artifacts}"
MONITOR="$MONITOR_ROOT/gpu-monitor/${MODE}-stage${STAGE}-${SLURM_JOB_ID}.csv"
source "$ROOT/scripts/configure_wandb_node_local.sh"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export N_JOBS="${N_JOBS:-48}"
export CKPT_ROOT="${TABICL_CKPT_ROOT:-$ROOT/artifacts/tabiclv2-clf-identity}"
export GPU_MONITOR_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$STAGE" -eq 3 ]]; then
  export RECOMPUTE="${RECOMPUTE:-True}"
fi

cd "$ROOT"
if [[ "$STAGE" -ge 2 ]]; then
  .venv/bin/python -c \
    'from tabicl._model.attention import HAS_FLASH_ATTN3; assert HAS_FLASH_ATTN3, "FlashAttention-3 is required for FP32 Stage 2/3 on H100"'
fi
scripts/run_with_gpu_monitor.sh \
  "$MONITOR" \
  "scripts/train_v2_clf_identity_stage${STAGE}.sh" "$MODE"
summary_status=0
.venv/bin/python scripts/summarize_gpu_usage.py "$MONITOR" \
  --threshold 80 --start-after-active --end-after-active \
  --expected-gpus "$NUM_GPUS" || summary_status=$?
if [[ "$summary_status" -ne 0 ]]; then
  if [[ "$STAGE" -eq 1 ]]; then
    exit "$summary_status"
  fi
  echo "warning: Stage $STAGE GPU utilization is diagnostic-only" >&2
fi
