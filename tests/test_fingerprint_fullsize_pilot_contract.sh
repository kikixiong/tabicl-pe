#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TRAIN="$ROOT/scripts/train_fingerprint_fullsize_pilot_stage1.sh"
SLURM="$ROOT/scripts/slurm_fingerprint_fullsize_h100_pilot.sh"
SUBMIT="$ROOT/scripts/submit_fingerprint_fullsize_h100_pilot.sh"

for script in "$TRAIN" "$SLURM" "$SUBMIT"; do
  bash -n "$script"
done

contains() {
  local path="$1"
  local expected="$2"
  grep -F -- "$expected" "$path" >/dev/null || {
    echo "missing contract text in $path: $expected" >&2
    exit 1
  }
}

contains "$TRAIN" '--embed_dim 128 --col_num_blocks 3 --col_nhead 8 --col_num_inds 128'
contains "$TRAIN" '--row_num_blocks 3 --row_nhead 8 --row_num_cls 4'
contains "$TRAIN" '--icl_num_blocks 12 --icl_nhead 8 --icl_ssmax True'
contains "$TRAIN" '--max_seq_len 1024'
contains "$TRAIN" '--scheduler cosine_with_restarts --warmup_proportion -1 --warmup_steps 5000'
contains "$TRAIN" 'ROW_IDENTITY_MODE="rope"'
contains "$TRAIN" 'ROW_IDENTITY_MODE="none"'
contains "$TRAIN" 'ROW_FINGERPRINT="True"'
contains "$TRAIN" '--row_fingerprint_dim 16'
contains "$TRAIN" '--fail_on_oom True --fail_on_nonfinite True'
contains "$SLURM" '[[ "${GPU_NAMES[0]}" == *"NVIDIA H100"* ]]'
contains "$SUBMIT" '--hold'
contains "$SUBMIT" 'for ARM in rope fingerprint'
contains "$SUBMIT" 'QOS=medium'
contains "$SUBMIT" 'TIME_LIMIT=08:00:00'

echo "full-size fingerprint pilot launcher contract: PASS"
