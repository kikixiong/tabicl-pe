#!/usr/bin/env bash
#SBATCH --job-name=tabicl-id-s1
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --array=0-2%1
#SBATCH --output=/slurm-storage/jiaxio/ws/TabFM/train/tabicl-v2-pretrain/artifacts/logs/%A_%a.out
#SBATCH --error=/slurm-storage/jiaxio/ws/TabFM/train/tabicl-v2-pretrain/artifacts/logs/%A_%a.err

set -euo pipefail

ROOT=/slurm-storage/jiaxio/ws/TabFM/train/tabicl-v2-pretrain
MODES=(rope temporary none)
MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"
MAX_STEPS="${MAX_STEPS:-20000}"

cd "$ROOT"
NUM_GPUS=4 MAX_STEPS="$MAX_STEPS" scripts/train_v2_clf_identity_stage1.sh "$MODE"
