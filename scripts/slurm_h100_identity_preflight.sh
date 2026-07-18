#!/usr/bin/env bash
#SBATCH --job-name=tabicl-none-preflight
#SBATCH --partition=h100
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --output=/slurm-storage/jiaxio/ws/TabFM/train/tabicl-v2-pretrain/artifacts/logs/%x-%j.out
#SBATCH --error=/slurm-storage/jiaxio/ws/TabFM/train/tabicl-v2-pretrain/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT=/slurm-storage/jiaxio/ws/TabFM/train/tabicl-v2-pretrain
MONITOR="$ROOT/artifacts/gpu-monitor/preflight-${SLURM_JOB_ID}.csv"
: "${NUM_GPUS:?NUM_GPUS must be 1, 2, or 4}"
case "$NUM_GPUS" in 1|2|4) ;; *) echo "invalid NUM_GPUS=$NUM_GPUS" >&2; exit 2 ;; esac
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MAX_STEPS="${MAX_STEPS:-100}"
export N_JOBS="${N_JOBS:-48}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
export BATCH_SIZE_PER_GP="${BATCH_SIZE_PER_GP:-$MICRO_BATCH_SIZE}"
export CKPT_ROOT="$ROOT/artifacts/preflight/${NUM_GPUS}gpu-mb${MICRO_BATCH_SIZE}"
export SAVE_TEMP_EVERY="$MAX_STEPS"
export SAVE_PERM_EVERY="$MAX_STEPS"
export MAX_CHECKPOINTS=1
export GPU_MONITOR_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

cd "$ROOT"
scripts/run_with_gpu_monitor.sh "$MONITOR" scripts/train_v2_clf_identity_stage1.sh none
.venv/bin/python scripts/summarize_gpu_usage.py "$MONITOR" \
  --threshold 80 --start-after-active --warmup-samples 2 \
  --min-samples 12 --expected-gpus "$NUM_GPUS"
