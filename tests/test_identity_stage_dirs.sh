#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TMPDIR:?TMPDIR must be configured}"
TEST_ROOT="$(mktemp -d "$TMPDIR/tabicl-stage-dirs.XXXXXX")"
trap 'find "$TEST_ROOT" -depth -delete' EXIT

for stage in 1 2 3; do
  checkpoint_root="$TEST_ROOT/checkpoints-stage${stage}"
  wandb_dir="$TEST_ROOT/wandb-stage${stage}"

  env \
    PYTHON=/usr/bin/true \
    NUM_GPUS=1 \
    CKPT_ROOT="$checkpoint_root" \
    WANDB_DIR="$wandb_dir" \
    bash "$ROOT/scripts/train_v2_clf_identity_stage${stage}.sh" rope

  expected_checkpoint_dir="$checkpoint_root/rope/seed-42/stage${stage}"
  test -d "$expected_checkpoint_dir" || {
    echo "checkpoint directory was not created: $expected_checkpoint_dir" >&2
    exit 1
  }
  test -d "$wandb_dir" || {
    echo "W&B directory was not created: $wandb_dir" >&2
    exit 1
  }
done

resume_checkpoint="$TEST_ROOT/step-35800.ckpt"
touch "$resume_checkpoint"
resume_output="$({
  env \
    PYTHON=/bin/echo \
    NUM_GPUS=1 \
    CKPT_ROOT="$TEST_ROOT/resume-checkpoints" \
    WANDB_DIR="$TEST_ROOT/resume-wandb" \
    RESUME_CHECKPOINT="$resume_checkpoint" \
    bash "$ROOT/scripts/train_v2_clf_identity_stage2.sh" rope
} 2>&1)"
grep -F -- "--checkpoint_path $resume_checkpoint" <<<"$resume_output" >/dev/null
if grep -F -- "--only_load_model True" <<<"$resume_output" >/dev/null; then
  echo "Stage 2 resume unexpectedly requested model-only loading" >&2
  exit 1
fi

echo "identity stage directories are created before launch"
