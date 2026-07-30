#!/bin/bash
# Fresh C20 retrain: accepted C16 null-structured CS-DNO plus universal
# mode-balanced complex supervision and the corrected localized tangent loss.

set -euo pipefail
cd /home/johnma/dno-fno

MODE="${1:-full}"
DATASET="${DATASET:-combined_dataset_v9.npz}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LR="${LR:-2e-5}"
HADAMARD_WEIGHT="${HADAMARD_WEIGHT:-1e-2}"
HADAMARD_INTERVAL="${HADAMARD_INTERVAL:-16}"
TRANSLATION_TANGENT_WEIGHT="${TRANSLATION_TANGENT_WEIGHT:-10}"
MODE_BALANCED_WEIGHT="${MODE_BALANCED_WEIGHT:-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-}"
MODE_BALANCED_WARMUP_STEPS="${MODE_BALANCED_WARMUP_STEPS:-}"

case "$MODE" in
  full)
    DATA_FRACTION="${DATA_FRACTION:-1.0}"
    EPOCHS="${EPOCHS:-40}"
    LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}"
    MODE_BALANCED_WARMUP_STEPS="${MODE_BALANCED_WARMUP_STEPS:-500}"
    RUN_PREFIX="c20_mode_balanced_full"
    ;;
  smoke)
    DATA_FRACTION="${DATA_FRACTION:-0.01}"
    EPOCHS="${EPOCHS:-1}"
    LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-50}"
    MODE_BALANCED_WARMUP_STEPS="${MODE_BALANCED_WARMUP_STEPS:-50}"
    RUN_PREFIX="c20_mode_balanced_smoke"
    ;;
  *)
    echo "usage: $0 [full|smoke]" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-$EPOCHS}"

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
  --translation_tangent_weight "$TRANSLATION_TANGENT_WEIGHT" \
  --translation_tangent_window_depths 1 \
  --translation_tangent_energy_floor_relative 1e-3 \
  --mode_balanced_weight "$MODE_BALANCED_WEIGHT" \
  --mode_balanced_warmup_steps "$MODE_BALANCED_WARMUP_STEPS" \
  --mode_balanced_k_max 128 \
  --mode_balanced_active_scale_relative 1e-4 \
  --mode_balanced_denominator_floor_relative 1e-6 \
  --hadamard_weight "$HADAMARD_WEIGHT" \
  --hadamard_interval "$HADAMARD_INTERVAL" \
  --hadamard_microbatch 8 \
  --hadamard_warmup_steps 500 \
  --hadamard_k_max 128 \
  --hadamard_sobolev_order 1 \
  --hadamard_relative_eps_min 1e-3 \
  --hadamard_relative_eps_max 3e-3 \
  --hadamard_eta_scale_floor 1e-3 \
  --hadamard_denominator_floor 1e-12 \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --lr_warmup_steps "$LR_WARMUP_STEPS" \
  --weight_decay 1e-4 \
  --epochs "$EPOCHS" \
  --total_epochs "$TOTAL_EPOCHS" \
  --skip_dno_eval \
  --skip_plots \
  --run_name "$RUN_NAME" 2>&1 | tee "/tmp/${RUN_NAME}.log"
