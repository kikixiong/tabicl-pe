#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 OUTPUT.csv COMMAND [ARG ...]" >&2
  exit 2
fi

OUTPUT="$1"
shift
mkdir -p "$(dirname "$OUTPUT")"

NVIDIA_SMI_ARGS=()
if [[ -n "${GPU_MONITOR_DEVICES:-}" ]]; then
  NVIDIA_SMI_ARGS=(-i "$GPU_MONITOR_DEVICES")
fi
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-5}"

nvidia-smi "${NVIDIA_SMI_ARGS[@]}" \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit \
  --format=csv,noheader,nounits \
  --loop="$GPU_MONITOR_INTERVAL" >"$OUTPUT" &
MONITOR_PID=$!

cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"$@"
