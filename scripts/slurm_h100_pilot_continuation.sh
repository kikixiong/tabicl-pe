#!/usr/bin/env bash
#SBATCH --partition=h100
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G

# Targeted continuation wrapper for the historical seed-42 RoPE/No-PE pilots.
# It deliberately does not implement the Temporary arm or claim formal status.
set -euo pipefail

: "${PILOT_ROOT:?PILOT_ROOT is required}"
: "${PILOT_ACTION:?PILOT_ACTION is required}"
: "${PILOT_CONTINUATION_ID:?PILOT_CONTINUATION_ID is required}"
: "${PILOT_SOURCE_SHA:?PILOT_SOURCE_SHA is required}"
: "${SLURM_JOB_ID:?must run inside a Slurm allocation}"

case "$PILOT_ACTION" in
  rope-stage1-479k-to-500k|none-stage2-6200-to-40k|rope-stage2-after-500k) ;;
  *) echo "unsupported PILOT_ACTION=$PILOT_ACTION" >&2; exit 2 ;;
esac
[[ "$PILOT_CONTINUATION_ID" =~ ^[a-z0-9][a-z0-9._-]{0,95}$ ]] || {
  echo "unsafe PILOT_CONTINUATION_ID" >&2; exit 2;
}
[[ "$PILOT_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || {
  echo "PILOT_SOURCE_SHA must be a full Git SHA" >&2; exit 2;
}

ROOT="$(cd "$PILOT_ROOT" && pwd -P)"
[[ "$(git -C "$ROOT" rev-parse --show-toplevel)" == "$ROOT" ]] || {
  echo "PILOT_ROOT is not the expected Git root" >&2; exit 2;
}
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$PILOT_SOURCE_SHA" ]] || {
  echo "pilot source HEAD differs from the submitted SHA" >&2; exit 1;
}
[[ -z "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "pilot continuation refuses a dirty source checkout" >&2; exit 1;
}

# Freeze the legacy training semantics even if this operational wrapper is
# later moved to another branch.  These are the exact raw-69e6d3d file bytes.
verify_file_digest() {
  local expected="$1"
  local path="$2"
  local observed
  observed="$(sha256sum "$ROOT/$path" | awk '{print $1}')"
  [[ "$observed" == "$expected" ]] || {
    echo "legacy pilot source digest mismatch for $path" >&2
    exit 1
  }
}
verify_file_digest 9fe98bd389e473511d1005327cae0dbd1a173e33203dfb9a6e690d1b3c919b19 src/tabicl/train/_run.py
verify_file_digest d03124e2b98a4a1c263bd7b01a7c4eae7464df703ac63758f34d8081e1ae8913 scripts/train_v2_clf_identity_stage1.sh
verify_file_digest c6139d3b5b81b75f526ed9f022d0548be6f07f346ac49da7741e98d7edf432a9 scripts/train_v2_clf_identity_stage2.sh

PYTHON="$ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo "pilot Python environment is unavailable" >&2; exit 1; }
VALIDATOR=("$PYTHON" "$ROOT/scripts/validate_pilot_checkpoint.py")
SNAPSHOTTER=("$PYTHON" "$ROOT/scripts/ensure_pilot_checkpoint_snapshot.py")
CKPT_ROOT="$ROOT/artifacts/tabiclv2-clf-identity"
SNAPSHOT_ROOT="$ROOT/artifacts/pilot-continuation-snapshots/$PILOT_CONTINUATION_ID"
MONITOR="$ROOT/artifacts/gpu-monitor/pilot-${PILOT_CONTINUATION_ID}-${PILOT_ACTION}-${SLURM_JOB_ID}.csv"

available_kib="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
reserve_kib=$((20 * 1024 * 1024))
warning_kib=$((22 * 1024 * 1024))
[[ "$available_kib" =~ ^[0-9]+$ ]] || { echo "could not read disk capacity" >&2; exit 1; }
if (( available_kib < reserve_kib )); then
  echo "less than 20 GiB is available; refusing pilot continuation" >&2
  exit 1
elif (( available_kib < warning_kib )); then
  echo "warning: less than 22 GiB is available" >&2
fi

source "$ROOT/scripts/configure_wandb_node_local.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 N_JOBS=48
export CKPT_ROOT PYTHON NUM_GPUS=1
export GPU_MONITOR_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

run_monitored() {
  scripts/run_with_gpu_monitor.sh "$MONITOR" "$@"
}

summarize_stage1_hard_gate() {
  "$PYTHON" "$ROOT/scripts/summarize_gpu_usage.py" "$MONITOR" \
    --threshold 80 --start-after-active --end-after-active --expected-gpus 1
}

summarize_stage2_diagnostic() {
  local status=0
  "$PYTHON" "$ROOT/scripts/summarize_gpu_usage.py" "$MONITOR" \
    --threshold 80 --start-after-active --end-after-active --expected-gpus 1 || status=$?
  if (( status != 0 )); then
    echo "warning: Stage 2 GPU utilization diagnostic returned $status; " \
      "the validated Stage 2 checkpoint remains a successful exploratory-pilot output" >&2
  fi
}

cd "$ROOT"
case "$PILOT_ACTION" in
  rope-stage1-479k-to-500k)
    START="$CKPT_ROOT/rope/seed-42/stage1/step-479000.ckpt"
    "${VALIDATOR[@]}" --checkpoint "$START" --mode rope --step 479000 \
      --expected-sha256 7f36e035e586ddbd2d33fa0e0cb7f649456946569dbaeb9335a4e7907654e117 \
      --require-latest-in-dir
    export MICRO_BATCH_SIZE=8 BATCH_SIZE_PER_GP=8
    run_monitored "$ROOT/scripts/train_v2_clf_identity_stage1.sh" rope
    FINAL="$CKPT_ROOT/rope/seed-42/stage1/step-500000.ckpt"
    "${VALIDATOR[@]}" --checkpoint "$FINAL" --mode rope --step 500000 \
      --require-latest-in-dir
    "${SNAPSHOTTER[@]}" --checkpoint "$FINAL" \
      --snapshot-dir "$SNAPSHOT_ROOT/rope/stage1-step500000" \
      --continuation-id "$PILOT_CONTINUATION_ID" \
      --continuation-source-commit "$PILOT_SOURCE_SHA" \
      --mode rope --stage stage1 --step 500000
    summarize_stage1_hard_gate
    ;;

  none-stage2-6200-to-40k)
    PARENT="$CKPT_ROOT/none/seed-42/stage1/step-500000.ckpt"
    START="$CKPT_ROOT/none/seed-42/stage2/step-6200.ckpt"
    "${VALIDATOR[@]}" --checkpoint "$PARENT" --mode none --step 500000 \
      --expected-sha256 14efa93349cb7b6f7d458508012a084526d970aeb00749bbb1cebbbe38aaec87
    "${VALIDATOR[@]}" --checkpoint "$START" --mode none --step 6200 \
      --expected-sha256 34564fe378d57d89b10e800b01ab63185170df25ad9210175f162c31749a16be \
      --require-latest-in-dir
    "${SNAPSHOTTER[@]}" --checkpoint "$PARENT" \
      --snapshot-dir "$SNAPSHOT_ROOT/none/stage1-step500000" \
      --continuation-id "$PILOT_CONTINUATION_ID" \
      --continuation-source-commit "$PILOT_SOURCE_SHA" \
      --mode none --stage stage1 --step 500000 \
      --expected-sha256 14efa93349cb7b6f7d458508012a084526d970aeb00749bbb1cebbbe38aaec87
    "$PYTHON" -c \
      'from tabicl._model.attention import HAS_FLASH_ATTN3; assert HAS_FLASH_ATTN3, "FlashAttention-3 is required"'
    run_monitored "$ROOT/scripts/train_v2_clf_identity_stage2.sh" none
    FINAL="$CKPT_ROOT/none/seed-42/stage2/step-40000.ckpt"
    "${VALIDATOR[@]}" --checkpoint "$FINAL" --mode none --step 40000 \
      --require-latest-in-dir
    summarize_stage2_diagnostic
    ;;

  rope-stage2-after-500k)
    PARENT="$CKPT_ROOT/rope/seed-42/stage1/step-500000.ckpt"
    CHILD_DIR="$CKPT_ROOT/rope/seed-42/stage2"
    "${VALIDATOR[@]}" --checkpoint "$PARENT" --mode rope --step 500000 \
      --require-latest-in-dir
    if [[ -d "$CHILD_DIR" ]] && find "$CHILD_DIR" -maxdepth 1 -type f -name 'step-*.ckpt' -print -quit | grep -q .; then
      echo "RoPE Stage 2 continuation is fresh-child-only and found an existing child checkpoint" >&2
      exit 1
    fi
    "${SNAPSHOTTER[@]}" --checkpoint "$PARENT" \
      --snapshot-dir "$SNAPSHOT_ROOT/rope/stage1-step500000" \
      --continuation-id "$PILOT_CONTINUATION_ID" \
      --continuation-source-commit "$PILOT_SOURCE_SHA" \
      --mode rope --stage stage1 --step 500000
    "$PYTHON" -c \
      'from tabicl._model.attention import HAS_FLASH_ATTN3; assert HAS_FLASH_ATTN3, "FlashAttention-3 is required"'
    run_monitored "$ROOT/scripts/train_v2_clf_identity_stage2.sh" rope
    FINAL="$CKPT_ROOT/rope/seed-42/stage2/step-40000.ckpt"
    "${VALIDATOR[@]}" --checkpoint "$FINAL" --mode rope --step 40000 \
      --require-latest-in-dir
    summarize_stage2_diagnostic
    ;;
esac
