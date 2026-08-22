#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-rope}"
case "$MODE" in
  rope|temporary|none) ;;
  *) echo "usage: $0 {rope|temporary|none}" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
NUM_GPUS="${NUM_GPUS:-4}"
N_JOBS="${N_JOBS:-16}"
SEED="${SEED:-42}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/artifacts/tabiclv2-clf-identity}"
STAGE1_CKPT="$CKPT_ROOT/$MODE/seed-$SEED/stage1/step-500000.ckpt"
CKPT_DIR="$CKPT_ROOT/$MODE/seed-$SEED/stage2"
WANDB_DIR="${WANDB_DIR:-$ROOT/artifacts/wandb}"
mkdir -p "$CKPT_DIR" "$WANDB_DIR"

RESUME_ARGS=(--checkpoint_path "$STAGE1_CKPT" --only_load_model True)
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  [[ -f "$RESUME_CHECKPOINT" ]] || {
    echo "resume checkpoint does not exist: $RESUME_CHECKPOINT" >&2
    exit 1
  }
  RESUME_ARGS=(--checkpoint_path "$RESUME_CHECKPOINT")
elif compgen -G "$CKPT_DIR/step-*.ckpt" >/dev/null; then
  RESUME_ARGS=()
fi

if [[ "$NUM_GPUS" -eq 1 ]]; then
  LAUNCHER=("$PYTHON" -m tabicl.train)
else
  LAUNCHER=("$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" -m tabicl.train)
fi

"${LAUNCHER[@]}" \
  --wandb_log "${WANDB_LOG:-False}" \
  --wandb_project "${WANDB_PROJECT:-TabICLv2-Identity}" \
  --wandb_name "tabiclv2_clf_stage2_${MODE}_seed${SEED}" \
  --wandb_mode "${WANDB_MODE:-offline}" --wandb_dir "$WANDB_DIR" \
  --device cuda --dtype float32 --amp True \
  --np_seed "$SEED" --torch_seed "$SEED" --max_steps 40000 \
  --batch_size 64 --micro_batch_size 1 --lr 1e-4 \
  --muon True --beta1 0.9 --weight_decay 0.01 --use_cautious_wd False \
  --scheduler cosine_with_restarts --warmup_proportion 0.01 \
  --cosine_num_cycles 1 --cosine_amplitude_decay 1 --cosine_lr_end 1e-7 \
  --gradient_clipping 10 \
  --prior_type graph_scm --prior_device cpu --n_jobs "$N_JOBS" --batch_size_per_gp 1 \
  --min_features 1 --max_features 100 --max_classes 10 \
  --min_seq_len 400 --max_seq_len 10240 --log_seq_len True \
  --min_train_size 0.79 --max_train_size 0.81 --seq_len_per_gp True \
  --graph_noise False --filter_unpredictable_graphs True \
  --filter_unpredictable_datasets True --allow_act_warping False \
  --min_n_nodes 2 --max_n_nodes 32 --cauchy_dag_offset 0 \
  --embed_dim 128 --col_num_blocks 3 --col_nhead 8 --col_num_inds 128 \
  --col_affine False --col_feature_group same --col_feature_group_size 3 \
  --col_target_aware True --col_ssmax True \
  --row_num_blocks 3 --row_nhead 8 --row_num_cls 4 \
  --row_rope_base 100000 --row_rope_interleaved False --row_identity_mode "$MODE" \
  --icl_num_blocks 12 --icl_nhead 8 --icl_ssmax True \
  --ssmax_type qassmax-mlp-elementwise --ff_factor 2 \
  --norm_first True --zero_init False --use_flash_attn3 True \
  --checkpoint_dir "$CKPT_DIR" "${RESUME_ARGS[@]}" \
  --save_temp_every "${SAVE_TEMP_EVERY:-200}" \
  --save_perm_every "${SAVE_PERM_EVERY:-40000}" \
  --max_checkpoints "${MAX_CHECKPOINTS:-2}" \
  --empty_cache_every "${EMPTY_CACHE_EVERY:-0}"
