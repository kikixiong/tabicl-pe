#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
for script in \
  scripts/slurm_h100_pilot_continuation.sh \
  scripts/submit_h100_pilot_continuation.sh \
  scripts/slurm_monitor_pilot_continuation.sh; do
  bash -n "$ROOT/$script"
done

! grep -q 'submit_h100_rope_none_full.sh' "$ROOT/scripts/submit_h100_pilot_continuation.sh"
grep -q 'rope-stage1-479k-to-500k' "$ROOT/scripts/submit_h100_pilot_continuation.sh"
grep -q 'rope-stage2-after-500k' "$ROOT/scripts/submit_h100_pilot_continuation.sh"
grep -q 'none-stage2-6200-to-40k' "$ROOT/scripts/submit_h100_pilot_continuation.sh"
grep -q "dependency='afterok:<rope-stage1-job>'" "$ROOT/scripts/submit_h100_pilot_continuation.sh"
grep -q 'Stage 2 GPU utilization diagnostic returned' "$ROOT/scripts/slurm_h100_pilot_continuation.sh"
grep -q -- '--interval-seconds 1800' "$ROOT/scripts/slurm_monitor_pilot_continuation.sh"

TEMP_ROOT="$(mktemp -d)"
trap 'rm -rf -- "$TEMP_ROOT"' EXIT
OUTPUT="$TEMP_ROOT/dry-run.txt"
"$ROOT/scripts/submit_h100_pilot_continuation.sh" --dry-run \
  --receipt "$TEMP_ROOT/receipt.json" >"$OUTPUT"

[[ ! -e "$TEMP_ROOT/receipt.json" ]]
[[ "$(grep -c '^sbatch ' "$OUTPUT")" -eq 3 ]]
grep -q 'tabicl-pilot-rope-s1-500k' "$OUTPUT"
grep -q 'tabicl-pilot-rope-s2-40k' "$OUTPUT"
grep -q 'tabicl-pilot-none-s2-40k' "$OUTPUT"
! grep -q 'stage3' "$OUTPUT"
! grep -q 'temporary' "$OUTPUT"
grep -q 'Dry run only; no Slurm jobs or artifacts will be created.' "$OUTPUT"
