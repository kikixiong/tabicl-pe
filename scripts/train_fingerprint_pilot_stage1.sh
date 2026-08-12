#!/usr/bin/env bash
set -euo pipefail

ARM="${1:-}"
case "$ARM" in
  rope)
    ROW_IDENTITY_MODE="rope"
    ROW_FINGERPRINT="False"
    ;;
  none)
    ROW_IDENTITY_MODE="none"
    ROW_FINGERPRINT="False"
    ;;
  fingerprint)
    ROW_IDENTITY_MODE="none"
    ROW_FINGERPRINT="True"
    ;;
  *)
    echo "usage: $0 {rope|none|fingerprint}" >&2
    exit 2
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
: "${PYTHON:?PYTHON must name the validated Python interpreter}"
: "${PILOT_ARTIFACT_ROOT:?PILOT_ARTIFACT_ROOT must be an absolute external artifact directory}"
[[ "$PYTHON" == /* && -x "$PYTHON" ]] || { echo "invalid PYTHON: $PYTHON" >&2; exit 2; }
[[ "$PILOT_ARTIFACT_ROOT" == /* ]] || { echo "PILOT_ARTIFACT_ROOT must be absolute" >&2; exit 2; }

MAX_STEPS="${MAX_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
N_JOBS="${N_JOBS:-8}"
SEED="${SEED:-42}"
CKPT_DIR="$PILOT_ARTIFACT_ROOT/arms/$ARM/seed-$SEED/checkpoints"
WANDB_DIR="$PILOT_ARTIFACT_ROOT/arms/$ARM/seed-$SEED/wandb"
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

exec "$PYTHON" -m tabicl.train \
  --wandb_log "${WANDB_LOG:-False}" \
  --wandb_project "${WANDB_PROJECT:-TabICLv2-Fingerprint-Pilot}" \
  --wandb_name "tabiclv2_fingerprint_pilot_${ARM}_seed${SEED}" \
  --wandb_mode "${WANDB_MODE:-offline}" --wandb_dir "$WANDB_DIR" \
  --device cuda --dtype float16 --amp True \
  --np_seed "$SEED" --torch_seed "$SEED" --max_steps "$MAX_STEPS" \
  --batch_size "$BATCH_SIZE" --micro_batch_size "$MICRO_BATCH_SIZE" \
  --lr 8e-4 --muon True --beta1 0.9 --weight_decay 0.01 --use_cautious_wd False \
  --scheduler cosine_warmup --warmup_proportion -1 --warmup_steps 500 \
  --gradient_clipping 10 --fail_on_oom True --fail_on_nonfinite True \
  --prior_type graph_scm --prior_device cpu --n_jobs "$N_JOBS" \
  --batch_size_per_gp 4 \
  --min_features 1 --max_features 100 --max_classes 10 --max_seq_len 1024 \
  --min_train_size 0.3 --max_train_size 0.9 --seq_len_per_gp True \
  --graph_noise False --filter_unpredictable_graphs True \
  --filter_unpredictable_datasets True --allow_act_warping False \
  --min_n_nodes 2 --max_n_nodes 32 --cauchy_dag_offset 0 \
  --embed_dim 96 --col_num_blocks 2 --col_nhead 6 --col_num_inds 64 \
  --col_affine False --col_feature_group same --col_feature_group_size 3 \
  --col_target_aware True --col_ssmax True \
  --row_num_blocks 3 --row_nhead 6 --row_num_cls 4 \
  --row_rope_base 100000 --row_rope_interleaved False \
  --row_identity_mode "$ROW_IDENTITY_MODE" \
  --row_fingerprint "$ROW_FINGERPRINT" --row_fingerprint_dim 16 \
  --icl_num_blocks 6 --icl_nhead 6 --icl_ssmax True \
  --ssmax_type qassmax-mlp-elementwise --ff_factor 2 \
  --norm_first True --zero_init False --use_flash_attn3 False \
  --checkpoint_dir "$CKPT_DIR" \
  --save_temp_every 5000 --save_perm_every 5000 --max_checkpoints 0 \
  --empty_cache_every 0 --progress_refresh_seconds 10
