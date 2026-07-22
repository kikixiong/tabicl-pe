#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/tabicl-submit.XXXXXX")"
mkdir -p "$TEST_ROOT/bin"
ln -s /usr/bin/true "$TEST_ROOT/bin/sbatch"

output="$(PATH="$TEST_ROOT/bin:$PATH" bash "$ROOT/scripts/submit_h100_rope_none_full.sh" 1)"
test "$output" = $'none: stage1= stage2= stage3=\nrope: stage1= stage2= stage3='

echo "single-GPU identity chains accept GPU_COUNT=1"
