#!/usr/bin/env bash
set -euo pipefail

GPU_COUNT="${1:?usage: $0 {1|2|4}}"
case "$GPU_COUNT" in 1|2|4) ;; *) echo "invalid GPU_COUNT=$GPU_COUNT" >&2; exit 2 ;; esac
CPU_COUNT=$((GPU_COUNT * 16))
MEMORY_GB=$((GPU_COUNT * 128))
if [[ "$GPU_COUNT" -le 2 ]]; then
  QOS=long
  TIME_ARGS=()
else
  QOS=medium
  TIME_ARGS=(--time=23:30:00)
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p artifacts/logs artifacts/gpu-monitor

submit_stage() {
  local mode="$1"
  local stage="$2"
  local dependency="${3:-}"
  local args=(--parsable --qos="$QOS" --gres="gpu:$GPU_COUNT" \
    --cpus-per-task="$CPU_COUNT" --mem="${MEMORY_GB}G" "${TIME_ARGS[@]}")
  if [[ -n "$dependency" ]]; then
    args+=(--dependency="afterok:$dependency")
  fi
  sbatch "${args[@]}" \
    --export="ALL,MODE=$mode,STAGE=$stage,NUM_GPUS=$GPU_COUNT,N_JOBS=12,MICRO_BATCH_SIZE=8,BATCH_SIZE_PER_GP=8" \
    scripts/slurm_h100_identity_full.sh
}

none_s1=$(submit_stage none 1)
none_s2=$(submit_stage none 2 "$none_s1")
none_s3=$(submit_stage none 3 "$none_s2")

rope_s1=$(submit_stage rope 1)
rope_s2=$(submit_stage rope 2 "$rope_s1")
rope_s3=$(submit_stage rope 3 "$rope_s2")

printf 'none: stage1=%s stage2=%s stage3=%s\n' "$none_s1" "$none_s2" "$none_s3"
printf 'rope: stage1=%s stage2=%s stage3=%s\n' "$rope_s1" "$rope_s2" "$rope_s3"
