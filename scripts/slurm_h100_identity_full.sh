#!/usr/bin/env bash
#SBATCH --job-name=tabicl-identity-full
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=/slurm-storage/jiaxio/ws/TabFM/train/tabicl-v2-pretrain/artifacts/logs/%x-%j.out
#SBATCH --error=/slurm-storage/jiaxio/ws/TabFM/train/tabicl-v2-pretrain/artifacts/logs/%x-%j.err

set -euo pipefail

: "${MODE:?MODE must be rope or none}"
: "${STAGE:?STAGE must be 1, 2, or 3}"
: "${NUM_GPUS:?NUM_GPUS must be 1, 2, or 4}"
case "$MODE" in rope|none) ;; *) echo "invalid MODE=$MODE" >&2; exit 2 ;; esac
case "$STAGE" in 1|2|3) ;; *) echo "invalid STAGE=$STAGE" >&2; exit 2 ;; esac
case "$NUM_GPUS" in 1|2|4) ;; *) echo "invalid NUM_GPUS=$NUM_GPUS" >&2; exit 2 ;; esac

ROOT=/slurm-storage/jiaxio/ws/TabFM/train/tabicl-v2-pretrain
MONITOR="$ROOT/artifacts/gpu-monitor/${MODE}-stage${STAGE}-${SLURM_JOB_ID}.csv"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export N_JOBS="${N_JOBS:-12}"
export CKPT_ROOT="$ROOT/artifacts/tabiclv2-clf-identity"
export GPU_MONITOR_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
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
.venv/bin/python scripts/summarize_gpu_usage.py "$MONITOR" \
  --threshold 80 --expected-gpus "$NUM_GPUS"
