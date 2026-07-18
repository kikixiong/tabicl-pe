#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-rope}"
case "$MODE" in
  rope|temporary|none) ;;
  *) echo "usage: $0 {rope|temporary|none}" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CKPT_DIR="${CKPT_DIR:-$ROOT/artifacts/identity-smoke/$MODE}"

"$PYTHON" -m tabicl.train \
  --wandb_log False --device cuda --dtype float32 --amp False \
  --np_seed 42 --torch_seed 42 --max_steps 1 \
  --batch_size 4 --micro_batch_size 2 --lr 8e-4 \
  --muon True --scheduler constant --warmup_proportion 0 --gradient_clipping 10 \
  --prior_type graph_scm --prior_device cpu --n_jobs 1 --batch_size_per_gp 2 \
  --min_features 2 --max_features 6 --max_classes 3 \
  --min_seq_len 24 --max_seq_len 32 --min_train_size 0.5 --max_train_size 0.75 \
  --seq_len_per_gp True --graph_noise False \
  --filter_unpredictable_graphs True --filter_unpredictable_datasets True \
  --allow_act_warping False --min_n_nodes 2 --max_n_nodes 8 \
  --embed_dim 32 --col_num_blocks 1 --col_nhead 4 --col_num_inds 8 \
  --col_affine False --col_feature_group same --col_feature_group_size 3 \
  --col_target_aware True --col_ssmax True \
  --row_num_blocks 1 --row_nhead 4 --row_num_cls 2 \
  --row_rope_base 100000 --row_rope_interleaved False \
  --row_identity_mode "$MODE" \
  --icl_num_blocks 2 --icl_nhead 4 --icl_ssmax True \
  --ssmax_type qassmax-mlp-elementwise --ff_factor 2 \
  --norm_first True --zero_init False --use_flash_attn3 False \
  --checkpoint_dir "$CKPT_DIR" --save_temp_every 1 --save_perm_every 1
