#!/usr/bin/env bash
set -euo pipefail

ARM="${1:-}"
case "$ARM" in
  rope)
    ROW_IDENTITY_MODE="rope"
    ROW_FINGERPRINT="False"
    ;;
  fingerprint)
    ROW_IDENTITY_MODE="none"
    ROW_FINGERPRINT="True"
    ;;
  *)
    echo "usage: $0 {rope|fingerprint}" >&2
    exit 2
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
: "${PYTHON:?PYTHON must name the validated Python interpreter}"
: "${PILOT_ARTIFACT_ROOT:?PILOT_ARTIFACT_ROOT must be an absolute external artifact directory}"
[[ "$PYTHON" == /* && -x "$PYTHON" ]] || { echo "invalid PYTHON: $PYTHON" >&2; exit 2; }
[[ "$PILOT_ARTIFACT_ROOT" == /* ]] || { echo "PILOT_ARTIFACT_ROOT must be absolute" >&2; exit 2; }

MAX_STEPS="${MAX_STEPS:-5000}"
case "$MAX_STEPS" in
  1|5000) ;;
  *) echo "MAX_STEPS must be 1 or 5000 for this bounded pilot" >&2; exit 2 ;;
esac

BATCH_SIZE="${BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
N_JOBS="${N_JOBS:-48}"
BATCH_SIZE_PER_GP="${BATCH_SIZE_PER_GP:-8}"
SEED="${SEED:-42}"
CKPT_DIR="$PILOT_ARTIFACT_ROOT/arms/$ARM/seed-$SEED/checkpoints"
WANDB_DIR="$PILOT_ARTIFACT_ROOT/arms/$ARM/seed-$SEED/wandb"

# This pilot is fresh-only. Reusing a directory would make an implicit resume
# possible and invalidate the paired from-scratch comparison.
[[ ! -e "$CKPT_DIR" ]] || { echo "fresh checkpoint directory already exists: $CKPT_DIR" >&2; exit 1; }
mkdir -p "$CKPT_DIR" "$WANDB_DIR"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0

"$PYTHON" - <<'PY'
import pathlib
import tabicl

expected = pathlib.Path.cwd().resolve() / "src" / "tabicl" / "__init__.py"
actual = pathlib.Path(tabicl.__file__).resolve()
if actual != expected:
    raise SystemExit(f"wrong tabicl source: expected {expected}, got {actual}")
print(f"tabicl_source={actual}", flush=True)
PY

# This is the full released-size Stage-1 architecture. The two arms differ
# only in the Row-RoPE/fingerprint treatment and the parameters induced by it.
exec "$PYTHON" -m tabicl.train \
  --wandb_log "${WANDB_LOG:-False}" \
  --wandb_project "${WANDB_PROJECT:-TabICLv2-Fingerprint-Fullsize-Pilot}" \
  --wandb_name "tabiclv2_fullsize_${ARM}_${MAX_STEPS}step_seed${SEED}" \
  --wandb_mode "${WANDB_MODE:-offline}" --wandb_dir "$WANDB_DIR" \
  --device cuda --dtype float32 --amp True \
  --np_seed "$SEED" --torch_seed "$SEED" --max_steps "$MAX_STEPS" \
  --batch_size "$BATCH_SIZE" --micro_batch_size "$MICRO_BATCH_SIZE" \
  --lr 8e-4 --muon True --beta1 0.9 --weight_decay 0.01 --use_cautious_wd False \
  --scheduler cosine_with_restarts --warmup_proportion -1 --warmup_steps 5000 \
  --cosine_num_cycles 1 --cosine_amplitude_decay 1 --cosine_lr_end 1e-7 \
  --gradient_clipping 10 --fail_on_oom True --fail_on_nonfinite True \
  --prior_type graph_scm --prior_device cpu --n_jobs "$N_JOBS" \
  --batch_size_per_gp "$BATCH_SIZE_PER_GP" \
  --min_features 1 --max_features 100 --max_classes 10 --max_seq_len 1024 \
  --min_train_size 0.3 --max_train_size 0.9 --seq_len_per_gp True \
  --graph_noise False --filter_unpredictable_graphs True \
  --filter_unpredictable_datasets True --allow_act_warping False \
  --min_n_nodes 2 --max_n_nodes 32 --cauchy_dag_offset 0 \
  --embed_dim 128 --col_num_blocks 3 --col_nhead 8 --col_num_inds 128 \
  --col_affine False --col_feature_group same --col_feature_group_size 3 \
  --col_target_aware True --col_ssmax True \
  --row_num_blocks 3 --row_nhead 8 --row_num_cls 4 \
  --row_rope_base 100000 --row_rope_interleaved False \
  --row_identity_mode "$ROW_IDENTITY_MODE" \
  --row_fingerprint "$ROW_FINGERPRINT" --row_fingerprint_dim 16 \
  --icl_num_blocks 12 --icl_nhead 8 --icl_ssmax True \
  --ssmax_type qassmax-mlp-elementwise --ff_factor 2 \
  --norm_first True --zero_init False --use_flash_attn3 False \
  --checkpoint_dir "$CKPT_DIR" \
  --save_temp_every "${SAVE_TEMP_EVERY:-1000}" \
  --save_perm_every "$MAX_STEPS" --max_checkpoints 2 \
  --empty_cache_every 0 --progress_refresh_seconds 10
