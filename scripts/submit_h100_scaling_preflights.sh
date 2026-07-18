#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p artifacts/logs artifacts/gpu-monitor

job_1=$(sbatch --parsable --qos=short --time=02:00:00 --gres=gpu:1 --cpus-per-task=16 --mem=128G \
  --export=ALL,NUM_GPUS=1,N_JOBS=12,MICRO_BATCH_SIZE=8,MAX_STEPS=100 scripts/slurm_h100_identity_preflight.sh)
job_2=$(sbatch --parsable --dependency="afterany:$job_1" --qos=short --time=02:00:00 \
  --gres=gpu:2 --cpus-per-task=32 --mem=256G \
  --export=ALL,NUM_GPUS=2,N_JOBS=12,MICRO_BATCH_SIZE=8,MAX_STEPS=100 scripts/slurm_h100_identity_preflight.sh)
job_4=$(sbatch --parsable --dependency="afterany:$job_2" --qos=short --time=02:00:00 \
  --gres=gpu:4 --cpus-per-task=64 --mem=512G \
  --export=ALL,NUM_GPUS=4,N_JOBS=12,MICRO_BATCH_SIZE=8,MAX_STEPS=100 scripts/slurm_h100_identity_preflight.sh)

printf 'H100 scaling preflights: 1gpu=%s 2gpu=%s 4gpu=%s\n' "$job_1" "$job_2" "$job_4"
