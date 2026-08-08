#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
fail() { echo "formal stage contract failed: $*" >&2; exit 1; }
contains() { grep -F -- "$2" "$1" >/dev/null || fail "$(basename "$1") lacks $2"; }
not_contains() { ! grep -F -- "$2" "$1" >/dev/null || fail "$(basename "$1") contains forbidden $2"; }

for SCRIPT in \
  "$ROOT/scripts/check_formal_capacity.py" \
  "$ROOT/scripts/generate_formal_environment.py" \
  "$ROOT/scripts/verify_formal_environment_transaction.py" \
  "$ROOT/scripts/run_exact_tabicl.py" \
  "$ROOT/scripts/reject_nonfinite_log.py" \
  "$ROOT/scripts/verify_formal_environment.py" \
  "$ROOT/scripts/prune_identity_stage_checkpoints.py" \
  "$ROOT/scripts/run_with_durable_log.sh" \
  "$ROOT/scripts/run_formal_identity_production_job.py" \
  "$ROOT/scripts/formal_train_v2_clf_identity_stage1.sh" \
  "$ROOT/scripts/formal_train_v2_clf_identity_stage2.sh" \
  "$ROOT/scripts/formal_train_v2_clf_identity_stage3.sh" \
  "$ROOT/scripts/slurm_h100_identity_formal.sh"; do
  [[ -x "$SCRIPT" ]] || fail "$SCRIPT is not executable"
done
contains "$ROOT/scripts/generate_formal_environment.py" '--completion-output'
contains "$ROOT/scripts/generate_formal_environment.py" 'formal_environment_generation_completion'
contains "$ROOT/scripts/generate_formal_environment.py" '/proc/self/fd/'
contains "$ROOT/scripts/verify_formal_environment_transaction.py" '--expected-completion-sha256'
contains "$ROOT/scripts/verify_formal_environment_transaction.py" '--expected-transaction-sha256'
contains "$ROOT/scripts/verify_formal_environment_transaction.py" 'formal_environment_inventory'
contains "$ROOT/scripts/verify_formal_environment.py" 'imported before exact-T isolation'
contains "$ROOT/scripts/verify_formal_environment.py" '_provenance.py'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" ': "${GIT:?'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'hermetic_git() {'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'GIT_ALLOW_PROTOCOL=https'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'hermetic_git -C "$CHECKOUT_PARENT" clone --quiet --no-hardlinks --no-checkout --'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'RUNTIME_HOME="$CHECKOUT_PARENT/home"'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'export HOME="$RUNTIME_HOME" USER=tabicl LOGNAME=tabicl'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'export XDG_CACHE_HOME="$RUNTIME_HOME/.cache"'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'remote get-url origin'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" ': "${NVIDIA_SMI:?'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" '--query-token "$CUDA_VISIBLE_DEVICES"'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" '--query-fields name,uuid,driver_version'
contains "$ROOT/scripts/exec_digest_bound_nvidia_smi.py" 'QUERY_TIMEOUT_SECONDS = 15'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'FORMAL_EXPECTED_GPU_MODEL'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'FORMAL_EXPECTED_DRIVER_VERSION'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'verify_filesystem_isolation.py'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'verify_formal_environment.py'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" '--expected-sha256 "$FORMAL_ENVIRONMENT_SHA256"'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" ': "${FORMAL_SUBMISSION_EXACT_ROOT:?}"'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'clone --quiet --no-hardlinks --no-checkout --'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'run_formal_identity_production_job.py'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'STATUS=$?'
not_contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'exec "$CHECKOUT/scripts/formal_train'

for STAGE in 1 2 3; do
  SCRIPT="$ROOT/scripts/formal_train_v2_clf_identity_stage${STAGE}.sh"
  bash -n "$SCRIPT"
  contains "$SCRIPT" 'RUN_POLICY'
  contains "$SCRIPT" 'fresh-only'
  contains "$SCRIPT" 'check_formal_capacity.py'
  contains "$SCRIPT" 'run_exact_tabicl.py'
  contains "$SCRIPT" 'PYTHONNOUSERSITE=1'
  contains "$SCRIPT" '--max_checkpoints 1'
  contains "$SCRIPT" '--max_checkpoint_bytes "$CHECKPOINT_CEILING_BYTES"'
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
  contains "$SCRIPT" 'FORMAL_SEED'
  contains "$SCRIPT" '42|43|44'
  contains "$SCRIPT" '--np_seed "$FORMAL_SEED"'
  contains "$SCRIPT" '--torch_seed "$FORMAL_SEED"'
  contains "$SCRIPT" '--identity_rng_seed "$FORMAL_SEED"'
  contains "$SCRIPT" '--np-seed "$FORMAL_SEED"'
  contains "$SCRIPT" '--torch-seed "$FORMAL_SEED"'
  contains "$SCRIPT" '--identity-seed "$FORMAL_SEED"'
  not_contains "$SCRIPT" '--np_seed 42'
  not_contains "$SCRIPT" '--torch_seed 42'
  not_contains "$SCRIPT" '--identity_rng_seed 42'
  not_contains "$SCRIPT" '--np-seed 42'
  not_contains "$SCRIPT" '--torch-seed 42'
  not_contains "$SCRIPT" '--identity-seed 42'
  not_contains "$SCRIPT" '${MAX_STEPS'
  not_contains "$SCRIPT" '--max_checkpoints "${'
  contains "$SCRIPT" 'mkdir -- "$FORMAL_CHECKPOINT_DIR"'
  not_contains "$SCRIPT" 'mkdir -p "$FORMAL_CHECKPOINT_DIR"'
done

# The capacity snapshot intentionally precedes namespace creation.  Simulate a
# competing writer landing either a directory or symlink in that interval and
# prove the stage exits at the atomic mkdir without invoking a second command.
STAGE_TMP="$(mktemp -d)"
trap 'rm -rf "$STAGE_TMP"' EXIT
ARTIFACT_ROOT="$STAGE_TMP/artifacts"
CHECKPOINT_PARENT="$ARTIFACT_ROOT/arms/rope"
CHECKPOINT_DIR="$CHECKPOINT_PARENT/stage1"
mkdir -p "$CHECKPOINT_PARENT" "$STAGE_TMP/foreign"
FAKE_PYTHON="$STAGE_TMP/fake-python"
cat >"$FAKE_PYTHON" <<'EOF'
#!/bin/bash
set -eu
COUNT=0
[[ ! -f "$RACE_COUNT_FILE" ]] || read -r COUNT < "$RACE_COUNT_FILE"
COUNT=$((COUNT + 1))
printf '%s\n' "$COUNT" > "$RACE_COUNT_FILE"
if [[ "$*" == *check_formal_capacity.py* ]]; then
  case "${RACE_CREATE_KIND:-}" in
    directory) mkdir -- "$FORMAL_CHECKPOINT_DIR"; printf 'foreign\n' > "$FORMAL_CHECKPOINT_DIR/foreign" ;;
    symlink) ln -s -- "$RACE_TARGET" "$FORMAL_CHECKPOINT_DIR" ;;
  esac
  exit 0
fi
exit 97
EOF
chmod 700 "$FAKE_PYTHON"
COMMON_ENV=(
  RUN_POLICY=fresh TABICL_EXACT_ROOT="$ROOT" PYTHON="$FAKE_PYTHON"
  FORMAL_SEED=42 FORMAL_ARTIFACT_ROOT="$ARTIFACT_ROOT"
  FORMAL_CHECKPOINT_DIR="$CHECKPOINT_DIR" FORMAL_SOURCE_MANIFEST=/fixture/source.json
  FORMAL_SOURCE_SHA256=source FORMAL_SOURCE_COMMIT_SHA=commit FORMAL_SOURCE_TREE_SHA=tree
  FORMAL_ENVIRONMENT_SHA256=environment FORMAL_STUDY_ID=study-seed42
  FORMAL_OUTPUT_ID=study-seed42-rope-stage1 FORMAL_PRIOR_SHA256=prior
  FORMAL_ARCHITECTURE_SHA256=architecture FORMAL_OPTIMIZER_SHA256=optimizer
  FORMAL_SCIENTIFIC_SHA256=scientific FORMAL_COHORT_PROTOCOL_SHA256=cohort
  FORMAL_ARM_PROTOCOL_SHA256=arm FORMAL_TRANSACTION_LEDGER=/fixture/ledger.json
  FORMAL_TRANSACTION_LEDGER_SHA256=ledger FORMAL_UPSTREAM_IDENTITY=upstream
  FORMAL_ARTIFACT_IDENTITY=artifact CHECKPOINT_CEILING_BYTES=1
  DURABLE_LOG_ALLOWANCE_BYTES=100 FORMAL_RUN_LOG_CEILING_BYTES=1
  FORMAL_ATTESTATION_CEILING_BYTES=1 FORMAL_MANIFEST_CEILING_BYTES=1
  FORMAL_PROTOCOL_METADATA_ALLOWANCE_BYTES=1 RACE_TARGET="$STAGE_TMP/foreign"
)

mkdir -- "$CHECKPOINT_DIR"
if env "${COMMON_ENV[@]}" RACE_COUNT_FILE="$STAGE_TMP/existing.count" \
  "$ROOT/scripts/formal_train_v2_clf_identity_stage1.sh" rope >/dev/null 2>&1; then
  fail "pre-existing formal checkpoint directory was accepted"
fi
[[ ! -e "$STAGE_TMP/existing.count" ]] || fail "existing namespace reached capacity command"
rmdir "$CHECKPOINT_DIR"
ln -s -- "$STAGE_TMP/foreign" "$CHECKPOINT_DIR"
if env "${COMMON_ENV[@]}" RACE_COUNT_FILE="$STAGE_TMP/symlink.count" \
  "$ROOT/scripts/formal_train_v2_clf_identity_stage1.sh" rope >/dev/null 2>&1; then
  fail "pre-existing formal checkpoint symlink was accepted"
fi
[[ ! -e "$STAGE_TMP/symlink.count" ]] || fail "symlink namespace reached capacity command"
rm "$CHECKPOINT_DIR"

for KIND in directory symlink; do
  COUNT_FILE="$STAGE_TMP/race-$KIND.count"
  if env "${COMMON_ENV[@]}" RACE_COUNT_FILE="$COUNT_FILE" RACE_CREATE_KIND="$KIND" \
    "$ROOT/scripts/formal_train_v2_clf_identity_stage1.sh" rope >/dev/null 2>&1; then
    fail "racing $KIND checkpoint namespace was accepted"
  fi
  [[ "$(<"$COUNT_FILE")" == 1 ]] || fail "racing $KIND escaped atomic mkdir"
  if [[ "$KIND" == directory ]]; then
    [[ -f "$CHECKPOINT_DIR/foreign" ]] || fail "racing directory was replaced"
    rm -r "$CHECKPOINT_DIR"
  else
    [[ -L "$CHECKPOINT_DIR" ]] || fail "racing symlink was replaced"
    rm "$CHECKPOINT_DIR"
  fi
done

contains "$ROOT/scripts/formal_train_v2_clf_identity_stage1.sh" '--max_steps 500000'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage1.sh" '--progress_refresh_seconds 30'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage1.sh" '--remaining-from-audit'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage1.sh" 'step-500000.ckpt'
not_contains "$ROOT/scripts/formal_train_v2_clf_identity_stage1.sh" '--checkpoint_path'

contains "$ROOT/scripts/formal_train_v2_clf_identity_stage2.sh" '--max_steps 40000'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage2.sh" '--progress_refresh_seconds 30'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage2.sh" '--remaining-from-audit'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage2.sh" 'step-40000.ckpt'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage2.sh" '--formal_parent_stage stage1'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage2.sh" '--checkpoint_path "$FORMAL_PARENT_CHECKPOINT"'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage2.sh" '--only_load_model true'

contains "$ROOT/scripts/formal_train_v2_clf_identity_stage3.sh" '--max_steps 10000'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage3.sh" '--progress_refresh_seconds 30'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage3.sh" '--remaining-from-audit'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage3.sh" 'step-10000.ckpt'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage3.sh" '--formal_parent_stage stage2'
contains "$ROOT/scripts/formal_train_v2_clf_identity_stage3.sh" '--recompute true'

bash -n "$ROOT/scripts/slurm_h100_identity_formal.sh"
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" '#SBATCH --gres=gpu:1'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" '#SBATCH --partition=h100'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" '#SBATCH --qos=long'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" '#SBATCH --cpus-per-task=64'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'export PATH=/usr/bin:/bin'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" '#SBATCH --mem=131072M'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'NUM_GPUS must be exactly 1'
contains "$ROOT/scripts/slurm_h100_identity_formal.sh" 'FORMAL_SEED must be exactly 42, 43, or 44'
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
