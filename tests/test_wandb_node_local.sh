#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TMPDIR:?TMPDIR must be configured}"
TEST_ROOT="$(mktemp -d "$TMPDIR/tabicl-wandb-local.XXXXXX")"
trap 'find "$TEST_ROOT" -depth -delete' EXIT

env \
  -u SLURM_TMPDIR \
  -u WANDB_LOG \
  -u WANDB_MODE \
  -u WANDB_PROJECT \
  -u WANDB_DIR \
  -u WANDB_CACHE_DIR \
  -u WANDB_DATA_DIR \
  -u GPU_MONITOR_INTERVAL \
  TMPDIR="$TEST_ROOT" \
  SLURM_JOB_ID=12345 \
  bash -c '
    set -euo pipefail
    source "$1"
    expected_root="$TMPDIR/tabicl-wandb-$SLURM_JOB_ID"
    test "$WANDB_LOG" = True
    test "$WANDB_MODE" = online
    test "$WANDB_PROJECT" = TabICLv2-Identity
    test "$WANDB_DIR" = "$expected_root/run"
    test "$WANDB_CACHE_DIR" = "$expected_root/cache"
    test "$WANDB_DATA_DIR" = "$expected_root/data"
    test "$GPU_MONITOR_INTERVAL" = 30
    test -d "$WANDB_DIR"
    test -d "$WANDB_CACHE_DIR"
    test -d "$WANDB_DATA_DIR"
  ' _ "$ROOT/scripts/configure_wandb_node_local.sh"

echo "W&B runtime storage is isolated on node-local temporary storage"
