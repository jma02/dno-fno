#!/bin/bash
# C24: replace C21's localized tangent term with centered modal phase-rate
# supervision.  The analytic backbone remains exactly G0+G1.

set -euo pipefail
cd /home/johnma/dno-fno

MODE="${1:-full}"
SOURCE_CKPT="${SOURCE_CKPT:-outputs/c21_tangent_w100_from_c20_20260714_171442/final_ckpt}"
DATASET="${DATASET:-combined_dataset_v9.npz}"
MODAL_PHASE_RATE_WEIGHT="${MODAL_PHASE_RATE_WEIGHT:-1}"
LR="${LR:-2e-6}"

case "$MODE" in
  full)
    DATA_FRACTION="${DATA_FRACTION:-1.0}"
    EPOCHS="${EPOCHS:-1}"
    LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"
    RUN_PREFIX="c24_modal_phase_rate_from_c21"
    ;;
  smoke)
    DATA_FRACTION="${DATA_FRACTION:-0.01}"
    EPOCHS="${EPOCHS:-1}"
    LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-50}"
    RUN_PREFIX="c24_modal_phase_rate_smoke_from_c21"
    ;;
  *)
    echo "usage: $0 [full|smoke]" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python train-jax-10m/1d_dno_fno_jax.py \
  --model cs_dno \
  --norm scale \
  --dataset "$DATASET" \
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
  --modal_phase_rate_weight "$MODAL_PHASE_RATE_WEIGHT" \
  --modal_phase_rate_k_min 1 \
  --modal_phase_rate_k_max 128 \
  --modal_phase_rate_active_scale_relative 1e-4 \
  --modal_phase_rate_denominator_floor_relative 1e-6 \
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
  --run_name "$RUN_NAME" 2>&1 | tee "/tmp/${RUN_NAME}.log"
