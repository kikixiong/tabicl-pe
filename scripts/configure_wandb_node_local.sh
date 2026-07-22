#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"

LOCAL_WANDB_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/tabicl-wandb-${SLURM_JOB_ID}"
export WANDB_LOG="${WANDB_LOG:-True}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-TabICLv2-Identity}"
export WANDB_DIR="${WANDB_DIR:-$LOCAL_WANDB_ROOT/run}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$LOCAL_WANDB_ROOT/cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$LOCAL_WANDB_ROOT/data}"
export GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-30}"

mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
