#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/tabicl-stage-dirs.XXXXXX")"

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

echo "identity stage directories are created before launch"
