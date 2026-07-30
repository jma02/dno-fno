#!/bin/bash
# C22: replace the old relative-speed tangent objective by the geometrically
# derived gamma^2 phase-growth objective in a controlled C20 continuation.

set -euo pipefail
cd /home/johnma/dno-fno

MODE="${1:-full}"
SOURCE_CKPT="${SOURCE_CKPT:-outputs/c20_mode_balanced_full_20260713_050000/best_val_ckpt}"
# The 1%-v9 C20 smoke measured a raw phase-growth loss of 4.80e-6, so weight
# 10 contributes approximately 0.7% of the converged composite objective.
PHASE_GROWTH_WEIGHT="${PHASE_GROWTH_WEIGHT:-10}"
TRANSLATION_TANGENT_WEIGHT="${TRANSLATION_TANGENT_WEIGHT:-0}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-2e-6}"

case "$MODE" in
  smoke)
    DATA_FRACTION="${DATA_FRACTION:-0.01}"
    LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-50}"
    RUN_PREFIX="c22_phase_growth_smoke"
    ;;
  full)
    DATA_FRACTION="${DATA_FRACTION:-1.0}"
    LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"
    RUN_PREFIX="c22_phase_growth_full"
    ;;
  *)
    echo "usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python train-jax-10m/1d_dno_fno_jax.py \
  --model cs_dno \
  --norm scale \
  --dataset combined_dataset_v9.npz \
  --data_fraction "$DATA_FRACTION" \
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
  --phase_growth_weight "$PHASE_GROWTH_WEIGHT" \
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
  --lr_warmup_steps "$LR_WARMUP_STEPS" \
  --weight_decay 1e-4 \
  --epochs "$EPOCHS" \
  --total_epochs "$EPOCHS" \
  --resume_from "$SOURCE_CKPT" \
  --reset_opt_state \
  --skip_dno_eval \
  --skip_plots \
  --run_name "$RUN_NAME" 2>&1 | tee "/tmp/$RUN_NAME.log"
