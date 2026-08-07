#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
case "$MODE" in rope|temporary|none) ;; *) echo "usage: $0 {rope|temporary|none}" >&2; exit 2 ;; esac
: "${RUN_POLICY:?RUN_POLICY must be fresh}"
[[ "$RUN_POLICY" == "fresh" ]] || { echo "formal runs are fresh-only" >&2; exit 2; }
ROOT="${TABICL_EXACT_ROOT:?}"; ROOT="$(cd "$ROOT" && pwd -P)"
: "${PYTHON:?}"; [[ "$PYTHON" == /* && -x "$PYTHON" ]] || { echo "invalid PYTHON" >&2; exit 2; }
export PYTHONPATH="$ROOT/src" PYTHONNOUSERSITE=1

for NAME in FORMAL_ARTIFACT_ROOT FORMAL_CHECKPOINT_DIR FORMAL_SOURCE_MANIFEST \
  FORMAL_SOURCE_SHA256 FORMAL_SOURCE_COMMIT_SHA FORMAL_SOURCE_TREE_SHA \
  FORMAL_ENVIRONMENT_SHA256 FORMAL_SEED FORMAL_STUDY_ID FORMAL_OUTPUT_ID \
  FORMAL_PRIOR_SHA256 FORMAL_ARCHITECTURE_SHA256 FORMAL_OPTIMIZER_SHA256 \
  FORMAL_SCIENTIFIC_SHA256 FORMAL_COHORT_PROTOCOL_SHA256 FORMAL_ARM_PROTOCOL_SHA256 \
  FORMAL_TRANSACTION_LEDGER FORMAL_TRANSACTION_LEDGER_SHA256 \
  FORMAL_UPSTREAM_IDENTITY FORMAL_ARTIFACT_IDENTITY \
  FORMAL_PARENT_CHECKPOINT FORMAL_PARENT_FINALIZED_MANIFEST \
  FORMAL_PARENT_UPSTREAM_IDENTITY FORMAL_PARENT_ARTIFACT_IDENTITY \
  CHECKPOINT_CEILING_BYTES DURABLE_LOG_ALLOWANCE_BYTES \
  FORMAL_RUN_LOG_CEILING_BYTES FORMAL_ATTESTATION_CEILING_BYTES \
  FORMAL_MANIFEST_CEILING_BYTES FORMAL_PROTOCOL_METADATA_ALLOWANCE_BYTES; do
  [[ -n "${!NAME:-}" ]] || { echo "$NAME is required" >&2; exit 2; }
done
case "$FORMAL_SEED" in
  42|43|44) ;;
  *) echo "FORMAL_SEED must be exactly 42, 43, or 44" >&2; exit 2 ;;
esac
readonly FORMAL_SEED
[[ ! -e "$FORMAL_CHECKPOINT_DIR" && ! -L "$FORMAL_CHECKPOINT_DIR" ]] || {
  echo "fresh stage namespace already exists" >&2; exit 1;
}
[[ -f "$FORMAL_PARENT_CHECKPOINT" && -f "$FORMAL_PARENT_FINALIZED_MANIFEST" ]] || {
  echo "explicit validated Stage 1 parent is required" >&2; exit 1;
}

"$PYTHON" -I -B "$ROOT/scripts/check_formal_capacity.py" \
  --artifact-root "$FORMAL_ARTIFACT_ROOT" \
  --checkpoint-ceiling-bytes "$CHECKPOINT_CEILING_BYTES" \
  --durable-log-allowance-bytes "$DURABLE_LOG_ALLOWANCE_BYTES" \
  --audit-tree --remaining-from-audit \
  --run-log-ceiling-bytes "$FORMAL_RUN_LOG_CEILING_BYTES" \
  --attestation-ceiling-bytes "$FORMAL_ATTESTATION_CEILING_BYTES" \
  --manifest-ceiling-bytes "$FORMAL_MANIFEST_CEILING_BYTES" \
  --protocol-metadata-allowance-bytes "$FORMAL_PROTOCOL_METADATA_ALLOWANCE_BYTES"
mkdir -- "$FORMAL_CHECKPOINT_DIR"
TRAIN_LOG="$FORMAL_CHECKPOINT_DIR/train.log"
SOURCE_ATTESTATION="$FORMAL_CHECKPOINT_DIR/source-attestation-trainer.json"
VALIDATOR_ATTESTATION="$FORMAL_CHECKPOINT_DIR/source-attestation-validator.json"
VALIDATION_REPORT="$FORMAL_CHECKPOINT_DIR/validation-report.json"
FINALIZED_MANIFEST="$FORMAL_CHECKPOINT_DIR/finalized-checkpoint.json"
FINAL_CHECKPOINT="$FORMAL_CHECKPOINT_DIR/step-40000.ckpt"

PYTHON="$PYTHON" "$ROOT/scripts/run_with_durable_log.sh" \
  "$TRAIN_LOG" "$FORMAL_RUN_LOG_CEILING_BYTES" \
  "$PYTHON" -I -B "$ROOT/scripts/run_exact_tabicl.py" \
    --archive-root "$ROOT" --source-manifest "$FORMAL_SOURCE_MANIFEST" \
    --expected-manifest-sha256 "$FORMAL_SOURCE_SHA256" \
    --expected-commit-sha "$FORMAL_SOURCE_COMMIT_SHA" \
    --expected-tree-sha "$FORMAL_SOURCE_TREE_SHA" \
    --action trainer --source-attestation-output "$SOURCE_ATTESTATION" \
    --source-attestation-max-bytes "$FORMAL_ATTESTATION_CEILING_BYTES" -- \
    --formal_training true --formal_stage stage2 \
    --formal_source_manifest "$FORMAL_SOURCE_MANIFEST" \
    --formal_source_sha256 "$FORMAL_SOURCE_SHA256" \
    --formal_source_commit_sha "$FORMAL_SOURCE_COMMIT_SHA" \
    --formal_source_tree_sha "$FORMAL_SOURCE_TREE_SHA" \
    --formal_environment_sha256 "$FORMAL_ENVIRONMENT_SHA256" \
    --formal_study_id "$FORMAL_STUDY_ID" --formal_output_id "$FORMAL_OUTPUT_ID" \
    --formal_transaction_ledger "$FORMAL_TRANSACTION_LEDGER" \
    --formal_transaction_ledger_sha256 "$FORMAL_TRANSACTION_LEDGER_SHA256" \
    --formal_parent_finalized_manifest "$FORMAL_PARENT_FINALIZED_MANIFEST" \
    --formal_parent_stage stage1 \
    --formal_parent_upstream_identity "$FORMAL_PARENT_UPSTREAM_IDENTITY" \
    --formal_parent_artifact_identity "$FORMAL_PARENT_ARTIFACT_IDENTITY" \
    --formal_artifact_root "$FORMAL_ARTIFACT_ROOT" \
    --checkpoint_path "$FORMAL_PARENT_CHECKPOINT" --only_load_model true \
    --device cuda --dtype float32 --amp true \
    --wandb_log false --wandb_project TabICLv2-Identity \
    --wandb_name "$FORMAL_OUTPUT_ID" --wandb_mode disabled \
    --wandb_dir "$FORMAL_CHECKPOINT_DIR/wandb" \
    --np_seed "$FORMAL_SEED" --torch_seed "$FORMAL_SEED" \
    --identity_rng_seed "$FORMAL_SEED" --max_steps 40000 \
    --progress_refresh_seconds 30 \
    --batch_size 64 --micro_batch_size 1 --lr 1e-4 \
    --muon true --beta1 0.9 --weight_decay 0.01 --use_cautious_wd false \
    --scheduler cosine_with_restarts --warmup_proportion 0.01 \
    --cosine_num_cycles 1 --cosine_amplitude_decay 1 --cosine_lr_end 1e-7 \
    --gradient_clipping 10 --prior_type graph_scm --prior_device cpu --n_jobs 16 \
    --batch_size_per_gp 1 --min_features 1 --max_features 100 --max_classes 10 \
    --min_seq_len 400 --max_seq_len 10240 --log_seq_len true --replay_small false \
    --min_train_size 0.79 --max_train_size 0.81 --seq_len_per_gp true \
    --graph_noise false --filter_unpredictable_graphs true \
    --filter_unpredictable_datasets true --allow_act_warping false \
    --min_n_nodes 2 --max_n_nodes 32 --cauchy_dag_offset 0 \
    --embed_dim 128 --col_num_blocks 3 --col_nhead 8 --col_num_inds 128 \
    --col_affine false --col_feature_group same --col_feature_group_size 3 \
    --col_target_aware true --col_ssmax true \
    --row_num_blocks 3 --row_nhead 8 --row_num_cls 4 \
    --row_rope_base 100000 --row_rope_interleaved false --row_identity_mode "$MODE" \
    --icl_num_blocks 12 --icl_nhead 8 --icl_ssmax true \
    --ssmax_type qassmax-mlp-elementwise --ff_factor 2 \
    --norm_first true --zero_init false --use_flash_attn3 true --recompute false \
    --checkpoint_dir "$FORMAL_CHECKPOINT_DIR" \
    --max_checkpoint_bytes "$CHECKPOINT_CEILING_BYTES" --save_temp_every 200 \
    --save_perm_every 40000 --max_checkpoints 1 --empty_cache_every 0

"$PYTHON" -I -B "$ROOT/scripts/reject_nonfinite_log.py" "$TRAIN_LOG" \
  --max-bytes "$FORMAL_RUN_LOG_CEILING_BYTES"
PYTHON="$PYTHON" "$ROOT/scripts/run_with_durable_log.sh" \
  "$VALIDATION_REPORT" "$FORMAL_RUN_LOG_CEILING_BYTES" \
  "$PYTHON" -I -B "$ROOT/scripts/run_exact_tabicl.py" \
    --archive-root "$ROOT" --source-manifest "$FORMAL_SOURCE_MANIFEST" \
    --expected-manifest-sha256 "$FORMAL_SOURCE_SHA256" \
    --expected-commit-sha "$FORMAL_SOURCE_COMMIT_SHA" \
    --expected-tree-sha "$FORMAL_SOURCE_TREE_SHA" \
    --action checkpoint-validator --source-attestation-output "$VALIDATOR_ATTESTATION" \
    --source-attestation-max-bytes "$FORMAL_ATTESTATION_CEILING_BYTES" -- \
    --checkpoint "$FINAL_CHECKPOINT" --mode "$MODE" \
    --np-seed "$FORMAL_SEED" --torch-seed "$FORMAL_SEED" \
    --identity-seed "$FORMAL_SEED" \
    --stage stage2 --terminal-step 40000 --source-sha256 "$FORMAL_SOURCE_SHA256" \
    --environment-sha256 "$FORMAL_ENVIRONMENT_SHA256" --prior-sha256 "$FORMAL_PRIOR_SHA256" \
    --architecture-sha256 "$FORMAL_ARCHITECTURE_SHA256" \
    --optimizer-sha256 "$FORMAL_OPTIMIZER_SHA256" \
    --scientific-sha256 "$FORMAL_SCIENTIFIC_SHA256" \
    --cohort-protocol-sha256 "$FORMAL_COHORT_PROTOCOL_SHA256" \
    --arm-protocol-sha256 "$FORMAL_ARM_PROTOCOL_SHA256" \
    --world-size 1 --cuda-device-count 1 --max-checkpoint-bytes "$CHECKPOINT_CEILING_BYTES" \
    --parent-checkpoint "$FORMAL_PARENT_CHECKPOINT" \
    --finalized-parent-manifest "$FORMAL_PARENT_FINALIZED_MANIFEST" \
    --transaction-ledger "$FORMAL_TRANSACTION_LEDGER" \
    --transaction-ledger-sha256 "$FORMAL_TRANSACTION_LEDGER_SHA256" \
    --study-id "$FORMAL_STUDY_ID" --parent-stage stage1 \
    --upstream-identity "$FORMAL_PARENT_UPSTREAM_IDENTITY" \
    --artifact-identity "$FORMAL_PARENT_ARTIFACT_IDENTITY" \
    --artifact-root "$FORMAL_ARTIFACT_ROOT" \
    --finalize-output "$FINALIZED_MANIFEST" \
    --finalization-transaction-ledger "$FORMAL_TRANSACTION_LEDGER" \
    --finalization-transaction-ledger-sha256 "$FORMAL_TRANSACTION_LEDGER_SHA256" \
    --finalization-study-id "$FORMAL_STUDY_ID" \
    --finalization-upstream-identity "$FORMAL_UPSTREAM_IDENTITY" \
    --finalization-artifact-identity "$FORMAL_ARTIFACT_IDENTITY" \
    --finalization-artifact-root "$FORMAL_ARTIFACT_ROOT"

"$PYTHON" -I -B "$ROOT/scripts/prune_identity_stage_checkpoints.py" \
  --checkpoint-dir "$FORMAL_CHECKPOINT_DIR" --terminal-step 40000 \
  --finalized-manifest "$FINALIZED_MANIFEST" \
  --checkpoint-ceiling-bytes "$CHECKPOINT_CEILING_BYTES"

"$PYTHON" -I -B "$ROOT/scripts/check_formal_capacity.py" \
  --artifact-root "$FORMAL_ARTIFACT_ROOT" --audit-tree --remaining-from-audit \
  --checkpoint-ceiling-bytes "$CHECKPOINT_CEILING_BYTES" \
  --durable-log-allowance-bytes "$DURABLE_LOG_ALLOWANCE_BYTES" \
  --run-log-ceiling-bytes "$FORMAL_RUN_LOG_CEILING_BYTES" \
  --attestation-ceiling-bytes "$FORMAL_ATTESTATION_CEILING_BYTES" \
  --manifest-ceiling-bytes "$FORMAL_MANIFEST_CEILING_BYTES" \
  --protocol-metadata-allowance-bytes "$FORMAL_PROTOCOL_METADATA_ALLOWANCE_BYTES"
