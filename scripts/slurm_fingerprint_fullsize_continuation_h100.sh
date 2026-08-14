#!/usr/bin/env bash
#SBATCH --job-name=tabicl-fullsize-pe-cont
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --no-requeue

set -euo pipefail

required=(
  ARM SOURCE_ROOT EXPECTED_SOURCE_SHA CONTINUATION_ARTIFACT_ROOT PYTHON
  FROM_STEP TO_STEP PARENT_MANIFEST ENVIRONMENT_MANIFEST
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing $name" >&2; exit 2; }
done
case "$ARM" in rope|fingerprint) ;; *) echo "invalid ARM=$ARM" >&2; exit 2 ;; esac
[[ "$SOURCE_ROOT" == /* && ( -d "$SOURCE_ROOT/.git" || -f "$SOURCE_ROOT/.git" ) ]] || {
  echo "SOURCE_ROOT must be an absolute Git checkout" >&2
  exit 2
}
[[ "$EXPECTED_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid source SHA" >&2; exit 2; }
[[ "$CONTINUATION_ARTIFACT_ROOT" == /* ]] || { echo "artifact root must be absolute" >&2; exit 2; }
[[ "$PARENT_MANIFEST" == /* && -f "$PARENT_MANIFEST" ]] || { echo "invalid parent manifest" >&2; exit 2; }
[[ "$ENVIRONMENT_MANIFEST" == /* && -f "$ENVIRONMENT_MANIFEST" ]] || { echo "invalid environment manifest" >&2; exit 2; }
[[ "$PYTHON" == /* && -x "$PYTHON" ]] || { echo "invalid PYTHON" >&2; exit 2; }
[[ "$FROM_STEP" =~ ^[0-9]+$ && "$TO_STEP" =~ ^[0-9]+$ ]] || {
  echo "segment steps must be integers" >&2
  exit 2
}
(( FROM_STEP >= 5000 && TO_STEP > FROM_STEP && TO_STEP <= 500000 )) || {
  echo "invalid segment boundary $FROM_STEP->$TO_STEP" >&2
  exit 2
}

ACTUAL_SHA="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_SHA" == "$EXPECTED_SOURCE_SHA" ]] || {
  echo "source SHA mismatch: $ACTUAL_SHA != $EXPECTED_SOURCE_SHA" >&2
  exit 1
}
[[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]] || {
  echo "source checkout is not clean" >&2
  exit 1
}
if git -C "$SOURCE_ROOT" symbolic-ref -q HEAD >/dev/null; then
  echo "source checkout must be detached" >&2
  exit 1
fi

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq 1 ]] || { echo "expected exactly one visible GPU" >&2; exit 1; }
[[ "${GPU_NAMES[0]}" == *"NVIDIA H100"* ]] || {
  echo "continuation requires NVIDIA H100, got ${GPU_NAMES[0]}" >&2
  exit 1
}

printf -v FROM_LABEL '%06d' "$FROM_STEP"
printf -v TO_LABEL '%06d' "$TO_STEP"
SEGMENT_ROOT="$CONTINUATION_ARTIFACT_ROOT/arms/$ARM/seed-42/segments/${FROM_LABEL}-${TO_LABEL}"
[[ ! -e "$SEGMENT_ROOT" ]] || { echo "segment namespace already exists: $SEGMENT_ROOT" >&2; exit 1; }
mkdir -p "$SEGMENT_ROOT/resource"

GPU_CSV="$SEGMENT_ROOT/resource/gpu.csv"
MONITOR_INTERVAL=30
if (( TO_STEP == FROM_STEP + 1 )); then
  MONITOR_INTERVAL=1
fi
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits -l "$MONITOR_INTERVAL" >"$GPU_CSV" &
MONITOR_PID=$!
cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export PYTHONPATH="$SOURCE_ROOT/src"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0

cd "$SOURCE_ROOT"
srun --kill-on-bad-exit=1 "$PYTHON" -B \
  "$SOURCE_ROOT/scripts/run_fingerprint_fullsize_continuation.py" run-segment \
  --arm "$ARM" \
  --source-root "$SOURCE_ROOT" \
  --source-commit "$EXPECTED_SOURCE_SHA" \
  --environment-manifest "$ENVIRONMENT_MANIFEST" \
  --parent-manifest "$PARENT_MANIFEST" \
  --from-step "$FROM_STEP" \
  --stop-after-step "$TO_STEP" \
  --checkpoint-dir "$SEGMENT_ROOT/checkpoints" \
  --wandb-dir "$SEGMENT_ROOT/wandb" \
  --completion-output "$SEGMENT_ROOT/resource/completion.json"
