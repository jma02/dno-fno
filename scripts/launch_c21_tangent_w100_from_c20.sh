#!/bin/bash
# C21: diagnostic high-weight localized-tangent continuation from C20.

set -euo pipefail
cd /home/johnma/dno-fno

SOURCE_CKPT="${SOURCE_CKPT:-outputs/c20_mode_balanced_full_20260713_050000/best_val_ckpt}"
RUN_NAME="${RUN_NAME:-c21_tangent_w100_from_c20_$(date +%Y%m%d_%H%M%S)}"
TRANSLATION_TANGENT_WEIGHT="${TRANSLATION_TANGENT_WEIGHT:-100}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-2e-6}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python train-jax-10m/1d_dno_fno_jax.py \
  --model cs_dno \
  --norm scale \
  --dataset combined_dataset_v9.npz \
  --modes 64 \
  --width 512 \
  --n_blocks 8 \
  --latent 256 \
  --sobolev_k 1 \
  --cs_n_polys 3 \
  --cs_mult_hidden 128 \
  --cs_use_g1_baseline \
  --cs_g1_k_cut 0 \
  --cs_g1_fft_fp64 \
  --cs_tie_xi_out_mult \
  --cs_phi_bias_free \
  --cs_residual_eta_order 2 \
  --translation_tangent_weight "$TRANSLATION_TANGENT_WEIGHT" \
  --translation_tangent_window_depths 1 \
  --translation_tangent_energy_floor_relative 1e-3 \
  --mode_balanced_weight 6 \
  --mode_balanced_warmup_steps 1 \
  --mode_balanced_k_max 128 \
  --mode_balanced_active_scale_relative 1e-4 \
  --mode_balanced_denominator_floor_relative 1e-6 \
  --hadamard_weight 1e-2 \
  --hadamard_interval 16 \
  --hadamard_microbatch 8 \
  --hadamard_warmup_steps 1 \
  --hadamard_k_max 128 \
  --hadamard_sobolev_order 1 \
  --hadamard_relative_eps_min 1e-3 \
  --hadamard_relative_eps_max 3e-3 \
  --hadamard_eta_scale_floor 1e-3 \
  --hadamard_denominator_floor 1e-12 \
  --batch_size 1024 \
  --lr "$LR" \
  --lr_warmup_steps 200 \
  --weight_decay 1e-4 \
  --epochs "$EPOCHS" \
  --total_epochs "$EPOCHS" \
  --resume_from "$SOURCE_CKPT" \
  --reset_opt_state \
  --skip_dno_eval \
  --skip_plots \
  --run_name "$RUN_NAME" 2>&1 | tee "/tmp/$RUN_NAME.log"
