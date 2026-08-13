#!/usr/bin/env bash
#SBATCH --job-name=tabicl-fullsize-pe-pilot
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --no-requeue

set -euo pipefail

: "${ARM:?ARM is required}"
: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${EXPECTED_SOURCE_SHA:?EXPECTED_SOURCE_SHA is required}"
: "${PILOT_ARTIFACT_ROOT:?PILOT_ARTIFACT_ROOT is required}"
: "${PYTHON:?PYTHON is required}"
: "${MAX_STEPS:?MAX_STEPS is required}"
case "$ARM" in rope|fingerprint) ;; *) echo "invalid ARM=$ARM" >&2; exit 2 ;; esac
case "$MAX_STEPS" in 1|5000) ;; *) echo "MAX_STEPS must be 1 or 5000" >&2; exit 2 ;; esac
[[ "$SOURCE_ROOT" == /* && ( -d "$SOURCE_ROOT/.git" || -f "$SOURCE_ROOT/.git" ) ]] || {
  echo "invalid SOURCE_ROOT" >&2
  exit 2
}
[[ "$EXPECTED_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "EXPECTED_SOURCE_SHA must be a full SHA" >&2; exit 2; }
[[ "$PILOT_ARTIFACT_ROOT" == /* ]] || { echo "PILOT_ARTIFACT_ROOT must be absolute" >&2; exit 2; }
[[ "$PYTHON" == /* && -x "$PYTHON" ]] || { echo "invalid PYTHON" >&2; exit 2; }

ACTUAL_SHA="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_SHA" == "$EXPECTED_SOURCE_SHA" ]] || {
  echo "source SHA mismatch: expected $EXPECTED_SOURCE_SHA, got $ACTUAL_SHA" >&2
  exit 1
}
[[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]] || {
  echo "source checkout is not clean" >&2
  exit 1
}

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq 1 ]] || { echo "expected exactly one visible GPU" >&2; exit 1; }
[[ "${GPU_NAMES[0]}" == *"NVIDIA H100"* ]] || {
  echo "full-size pilot requires NVIDIA H100, got ${GPU_NAMES[0]}" >&2
  exit 1
}

SEED="${SEED:-42}"
BATCH_SIZE=64
MICRO_BATCH_SIZE=8
BATCH_SIZE_PER_GP=8
N_JOBS=48
RUN_ROOT="$PILOT_ARTIFACT_ROOT/arms/$ARM/seed-$SEED"
[[ ! -e "$RUN_ROOT" ]] || { echo "fresh arm namespace already exists: $RUN_ROOT" >&2; exit 1; }
mkdir -p "$RUN_ROOT/resource"

export ARM ACTUAL_SHA RUN_ROOT MAX_STEPS SEED
export BATCH_SIZE MICRO_BATCH_SIZE BATCH_SIZE_PER_GP N_JOBS
"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path
import platform
import torch

record = {
    "schema_version": 1,
    "study": "tabiclv2-fullsize-rope-fingerprint-pilot-v1",
    "formal_evidence": False,
    "fresh_from_scratch": True,
    "arm": os.environ["ARM"],
    "source_commit": os.environ["ACTUAL_SHA"],
    "seed": int(os.environ["SEED"]),
    "max_steps": int(os.environ["MAX_STEPS"]),
    "batch_size": int(os.environ["BATCH_SIZE"]),
    "micro_batch_size": int(os.environ["MICRO_BATCH_SIZE"]),
    "batch_size_per_gp": int(os.environ["BATCH_SIZE_PER_GP"]),
    "n_jobs": int(os.environ["N_JOBS"]),
    "architecture": {
        "embed_dim": 128,
        "col_num_blocks": 3,
        "row_num_blocks": 3,
        "icl_num_blocks": 12,
    },
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
path = Path(os.environ["RUN_ROOT"]) / "resource" / "launch.json"
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
tmp.replace(path)
PY

GPU_CSV="$RUN_ROOT/resource/gpu.csv"
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits -l 30 >"$GPU_CSV" &
MONITOR_PID=$!
cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$SOURCE_ROOT"
srun --kill-on-bad-exit=1 \
  "$SOURCE_ROOT/scripts/train_fingerprint_fullsize_pilot_stage1.sh" "$ARM"

FINAL_CKPT="$RUN_ROOT/checkpoints/step-$MAX_STEPS.ckpt"
[[ -f "$FINAL_CKPT" ]] || { echo "missing final checkpoint: $FINAL_CKPT" >&2; exit 1; }
export FINAL_CKPT EXPECTED_SOURCE_SHA
"$PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import torch

arm = os.environ["ARM"]
expected_steps = int(os.environ["MAX_STEPS"])
path = Path(os.environ["FINAL_CKPT"])
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
config = checkpoint["config"]
expected_architecture = {
    "embed_dim": 128,
    "col_num_blocks": 3,
    "col_nhead": 8,
    "col_num_inds": 128,
    "row_num_blocks": 3,
    "row_nhead": 8,
    "icl_num_blocks": 12,
    "icl_nhead": 8,
}
for key, expected in expected_architecture.items():
    if config.get(key) != expected:
        raise SystemExit(f"architecture mismatch for {key}: {config.get(key)!r} != {expected!r}")
expected_treatment = ("rope", False) if arm == "rope" else ("none", True)
actual_treatment = (config.get("row_identity_mode"), config.get("row_fingerprint"))
if actual_treatment != expected_treatment:
    raise SystemExit(f"treatment mismatch: {actual_treatment!r} != {expected_treatment!r}")
if checkpoint.get("curr_step") != expected_steps:
    raise SystemExit(f"step mismatch: {checkpoint.get('curr_step')!r} != {expected_steps!r}")
state_elements = sum(value.numel() for value in checkpoint["state_dict"].values())
if state_elements < 27_500_000:
    raise SystemExit(f"checkpoint is not full-size: only {state_elements} state elements")

digest = hashlib.sha256(path.read_bytes()).hexdigest()
record = {
    "schema_version": 1,
    "study": "tabiclv2-fullsize-rope-fingerprint-pilot-v1",
    "formal_evidence": False,
    "arm": arm,
    "source_commit": os.environ["EXPECTED_SOURCE_SHA"],
    "curr_step": expected_steps,
    "checkpoint": {
        "path": str(path),
        "sha256": digest,
        "size_bytes": path.stat().st_size,
        "state_elements": state_elements,
    },
    "treatment": {
        "row_identity_mode": config["row_identity_mode"],
        "row_fingerprint": config["row_fingerprint"],
        "row_fingerprint_dim": config["row_fingerprint_dim"],
    },
    "architecture": expected_architecture,
    "optimizer_prefix": {
        "scheduler": "cosine_with_restarts",
        "scheduler_horizon_steps": expected_steps,
        "warmup_steps": 5000,
        "note": "The first 5000 learning-rate updates match the formal Stage-1 warmup prefix.",
    },
}
output = Path(os.environ["RUN_ROOT"]) / "resource" / "completion.json"
tmp = output.with_suffix(".json.tmp")
tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
tmp.replace(output)
print(json.dumps(record, sort_keys=True), flush=True)
PY
