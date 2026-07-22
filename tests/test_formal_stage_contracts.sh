#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
fail() { echo "formal stage contract failed: $*" >&2; exit 1; }
contains() { grep -F -- "$2" "$1" >/dev/null || fail "$(basename "$1") lacks $2"; }
not_contains() { ! grep -F -- "$2" "$1" >/dev/null || fail "$(basename "$1") contains forbidden $2"; }

for SCRIPT in \
  "$ROOT/scripts/check_formal_capacity.py" \
  "$ROOT/scripts/run_exact_tabicl.py" \
  "$ROOT/scripts/reject_nonfinite_log.py" \
  "$ROOT/scripts/prune_identity_stage_checkpoints.py" \
  "$ROOT/scripts/run_with_durable_log.sh" \
  "$ROOT/scripts/formal_train_v2_clf_identity_stage1.sh" \
  "$ROOT/scripts/formal_train_v2_clf_identity_stage2.sh" \
  "$ROOT/scripts/formal_train_v2_clf_identity_stage3.sh" \
  "$ROOT/scripts/slurm_h100_identity_formal.sh"; do
  [[ -x "$SCRIPT" ]] || fail "$SCRIPT is not executable"
done
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" ': "${GIT:?'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" '"$GIT" clone'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" ': "${NVIDIA_SMI:?'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" '"$NVIDIA_SMI" --query-gpu=name,uuid'

for STAGE in 1 2 3; do
  SCRIPT="$ROOT/scripts/formal_train_v2_clf_identity_stage${STAGE}.sh"
  bash -n "$SCRIPT"
  contains "$SCRIPT" 'RUN_POLICY'
  contains "$SCRIPT" 'fresh-only'
  contains "$SCRIPT" 'check_formal_capacity.py'
  contains "$SCRIPT" 'run_exact_tabicl.py'
  contains "$SCRIPT" 'PYTHONNOUSERSITE=1'
  contains "$SCRIPT" '--max_checkpoints 1'
  contains "$SCRIPT" '--cohort-protocol-sha256'
  contains "$SCRIPT" '--arm-protocol-sha256'
  contains "$SCRIPT" 'reject_nonfinite_log.py'
  contains "$SCRIPT" 'prune_identity_stage_checkpoints.py'
  contains "$SCRIPT" 'finalized-checkpoint.json'
  contains "$SCRIPT" '--audit-tree'
  contains "$SCRIPT" '--wandb_log false'
  not_contains "$SCRIPT" '${WANDB_LOG'
  contains "$SCRIPT" 'FORMAL_ATTESTATION_CEILING_BYTES'
  contains "$SCRIPT" 'FORMAL_PROTOCOL_METADATA_ALLOWANCE_BYTES'
  not_contains "$SCRIPT" '${MAX_STEPS'
  not_contains "$SCRIPT" '--max_checkpoints "${'
done

contains "$ROOT/scripts/formal_train_v2_clf_identity_stage1.sh" '--max_steps 500000'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage1.sh" '--remaining-from-audit'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage1.sh" 'step-500000.ckpt'
not_contains "$ROOT/scripts/formal_train_v2_clf_identity_stage1.sh" '--checkpoint_path'

contains "$ROOT/scripts/formal_train_v2_clf_identity_stage2.sh" '--max_steps 40000'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage2.sh" '--remaining-from-audit'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage2.sh" 'step-40000.ckpt'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage2.sh" '--formal_parent_stage stage1'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage2.sh" '--checkpoint_path "$FORMAL_PARENT_CHECKPOINT"'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage2.sh" '--only_load_model true'

contains "$ROOT/scripts/formal_train_v2_clf_identity_stage3.sh" '--max_steps 10000'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage3.sh" '--remaining-from-audit'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage3.sh" 'step-10000.ckpt'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage3.sh" '--formal_parent_stage stage2'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage3.sh" '--recompute true'

bash -n "$ROOT/scripts/slurm_h100_identity_formal.sh"
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" '#SBATCH --gres=gpu:1'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'NUM_GPUS must be exactly 1'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'checkout --quiet --detach'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'status --porcelain=v1 --untracked-files=all'
if grep -F 'formal_run_with_gpu_monitor.sh' "$ROOT/scripts/slurm_h100_identity_formal.sh" >/dev/null; then
  fail "production job must not write a whole-run one-second GPU CSV"
fi
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'bounded max-sequence jobs own'

# Existing pilot entry points are outside formal T and must remain byte-identical.
for RAW in \
  scripts/train_v2_clf_identity_stage1.sh \
  scripts/train_v2_clf_identity_stage2.sh \
  scripts/train_v2_clf_identity_stage3.sh \
  scripts/slurm_h100_identity_full.sh \
  scripts/run_with_gpu_monitor.sh \
  scripts/summarize_gpu_usage.py; do
  git -C "$ROOT" diff --quiet 8513d8a19afd8b301bc08ab05dbec9bd34e09cc6 -- "$RAW" || \
    fail "pre-existing pilot changed: $RAW"
done

echo "formal stage contracts passed"
