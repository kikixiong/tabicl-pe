#!/usr/bin/env bash
#SBATCH --job-name=tabicl-fingerprint-h100-2k
#SBATCH --partition=h100
#SBATCH --qos=medium
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --no-requeue

set -euo pipefail

: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${EXPECTED_SOURCE_SHA:?EXPECTED_SOURCE_SHA is required}"
: "${PILOT_ARTIFACT_ROOT:?PILOT_ARTIFACT_ROOT is required}"
: "${PYTHON:?PYTHON is required}"
[[ "$SOURCE_ROOT" == /* && -d "$SOURCE_ROOT/.git" ]] || {
  # Git worktrees use a .git file rather than a directory.
  [[ "$SOURCE_ROOT" == /* && -f "$SOURCE_ROOT/.git" ]] || { echo "invalid SOURCE_ROOT" >&2; exit 2; }
}
[[ "$EXPECTED_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "EXPECTED_SOURCE_SHA must be a full SHA" >&2; exit 2; }
[[ "$PILOT_ARTIFACT_ROOT" == /* ]] || { echo "PILOT_ARTIFACT_ROOT must be absolute" >&2; exit 2; }

ACTUAL_SHA="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_SHA" == "$EXPECTED_SOURCE_SHA" ]] || {
  echo "source SHA mismatch: expected $EXPECTED_SOURCE_SHA, got $ACTUAL_SHA" >&2
  exit 1
}
[[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]] || {
  echo "source checkout is not clean" >&2
  exit 1
}

ARMS=(rope fingerprint)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
MAX_STEPS=2000
N_JOBS=48
BATCH_SIZE=64
MICRO_BATCH_SIZE=8
BATCH_SIZE_PER_GP=8
SAVE_TEMP_EVERY=2000
SAVE_PERM_EVERY=2000
SEED="${SEED:-42}"
RUN_ROOT="$PILOT_ARTIFACT_ROOT/arms/$ARM/seed-$SEED"
mkdir -p "$RUN_ROOT/resource"

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq 1 ]] || { echo "expected exactly one visible GPU" >&2; exit 1; }
[[ "${GPU_NAMES[0]}" == *"NVIDIA H100"* ]] || {
  echo "pilot requires NVIDIA H100, got ${GPU_NAMES[0]}" >&2
  exit 1
}

export ARM EXPECTED_SOURCE_SHA ACTUAL_SHA RUN_ROOT MAX_STEPS N_JOBS
export BATCH_SIZE MICRO_BATCH_SIZE BATCH_SIZE_PER_GP
export SAVE_TEMP_EVERY SAVE_PERM_EVERY SEED
"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path
import platform
import torch

record = {
    "schema_version": 1,
    "study": "tabiclv2-fingerprint-h100-2k-pilot-v1",
    "formal_evidence": False,
    "arm": os.environ["ARM"],
    "source_commit": os.environ["ACTUAL_SHA"],
    "seed": int(os.environ["SEED"]),
    "max_steps": int(os.environ["MAX_STEPS"]),
    "batch_size": int(os.environ["BATCH_SIZE"]),
    "micro_batch_size": int(os.environ["MICRO_BATCH_SIZE"]),
    "batch_size_per_gp": int(os.environ["BATCH_SIZE_PER_GP"]),
    "n_jobs": int(os.environ["N_JOBS"]),
    "save_temp_every": int(os.environ["SAVE_TEMP_EVERY"]),
    "save_perm_every": int(os.environ["SAVE_PERM_EVERY"]),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
}
path = Path(os.environ["RUN_ROOT"]) / "resource" / "launch.json"
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
tmp.replace(path)
PY

GPU_CSV="$RUN_ROOT/resource/gpu.csv"
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 30 >"$GPU_CSV" &
MONITOR_PID=$!
cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$SOURCE_ROOT"
srun --kill-on-bad-exit=1 \
  "$SOURCE_ROOT/scripts/train_fingerprint_pilot_stage1.sh" "$ARM"
