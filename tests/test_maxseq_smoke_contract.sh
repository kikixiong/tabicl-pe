#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
fail() { echo "maxseq smoke contract failed: $*" >&2; exit 1; }
contains() { grep -F -- "$2" "$1" >/dev/null || fail "$(basename "$1") lacks $2"; }

for SCRIPT in \
  "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" \
  "$ROOT/scripts/slurm_h100_identity_maxseq_smoke.sh" \
  "$ROOT/scripts/formal_run_with_gpu_monitor.sh"; do
  [[ -x "$SCRIPT" ]] || fail "$SCRIPT is not executable"
  bash -n "$SCRIPT"
done

contains "$ROOT/scripts/slurm_h100_identity_maxseq_smoke.sh" '#SBATCH --qos=short'
contains "$ROOT/scripts/slurm_h100_identity_maxseq_smoke.sh" 'checkout --quiet --detach'
contains "$ROOT/scripts/slurm_h100_identity_maxseq_smoke.sh" ': "${GIT:?'
contains "$ROOT/scripts/slurm_h100_identity_maxseq_smoke.sh" '"$GIT" clone'
contains "$ROOT/scripts/slurm_h100_identity_maxseq_smoke.sh" ': "${NVIDIA_SMI:?'
contains "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" '"$NVIDIA_SMI" --query-gpu=uuid'
contains "$ROOT/scripts/slurm_h100_identity_maxseq_smoke.sh" 'EXPECTED_GPUS=2'
contains "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" '--nproc_per_node=2'
if grep -F 'MAXSEQ_REPEAT_STEPS' "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" >/dev/null; then
  fail "hostile repeat-step override is exposed"
fi
contains "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" '--utilization-required'
contains "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" 'functional case created unbound GPU monitor artifacts'
contains "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" '--checkpoint-ceiling-bytes'
contains "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" '--runtime-evidence-output'
contains "$ROOT/scripts/formal_run_with_gpu_monitor.sh" 'summarize_formal_gpu_usage.py'
contains "$ROOT/scripts/formal_run_with_gpu_monitor.sh" 'FORMAL_GPU_ACTIVE_START_SIGNAL='
contains "$ROOT/scripts/formal_run_with_gpu_monitor.sh" 'FORMAL_GPU_ACTIVE_END_SIGNAL='
contains "$ROOT/scripts/summarize_formal_gpu_usage.py" 'interval_seconds != 1.0'

MATRIX="$($PYTHON -I -B "$ROOT/scripts/run_h100_identity_validation.py" --print-matrix)"
COUNT="$($PYTHON -I -B -c 'import json,sys; print(len(json.load(sys.stdin)))' <<<"$MATRIX")"
[[ "$COUNT" -eq 12 ]] || fail "matrix count is $COUNT, expected 12"

for CASE_ID in \
  stage1_rope_one_step stage1_temporary_one_step stage1_none_one_step \
  stage2_rope_maxseq10240 stage2_temporary_maxseq10240 stage2_none_maxseq10240 \
  stage3_rope_maxseq60000_recompute stage3_temporary_maxseq60000_recompute \
  stage3_none_maxseq60000_recompute temporary_cuda_rng_resume nccl_2gpu \
  prior_dataloader_resume; do
  "$PYTHON" -I -B "$ROOT/scripts/run_h100_identity_validation.py" \
    --dry-run-case "$CASE_ID" --root "$ROOT" >/dev/null
done

if "$ROOT/scripts/run_h100_identity_maxseq_smoke.sh" invalid-case >/dev/null 2>&1; then
  fail "invalid validation case was accepted"
fi

echo "maxseq smoke contracts passed"
