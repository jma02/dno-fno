#!/bin/bash
# C23: replace the instantaneous eta-only phase proxy by a sparse, paired,
# production-GL2 full-state finite-time objective.

set -euo pipefail
cd /home/johnma/dno-fno

MODE="${1:-full}"
SOURCE_CKPT="${SOURCE_CKPT:-outputs/c21_tangent_w100_from_c20_20260714_171442/final_ckpt}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-2e-6}"

case "$MODE" in
  smoke)
    DATA_FRACTION="${DATA_FRACTION:-0.01}"
    LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-50}"
    FINITE_TIME_PHASE_WEIGHT="${FINITE_TIME_PHASE_WEIGHT:-1}"
    FINITE_TIME_PHASE_INTERVAL="${FINITE_TIME_PHASE_INTERVAL:-4}"
    FINITE_TIME_PHASE_WARMUP_STEPS="${FINITE_TIME_PHASE_WARMUP_STEPS:-1}"
    RUN_PREFIX="c23_finite_time_phase_smoke_from_c21"
    ;;
  full)
    DATA_FRACTION="${DATA_FRACTION:-1.0}"
    LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"
    # The 1%-v9 smoke measured mean active loss 1.02103e-3.  Weight 0.1 at
    # interval 16 is about 1.5% of an active batch and 0.09% epoch-average,
    # matching the accepted sparse Hadamard term without rare large updates.
    FINITE_TIME_PHASE_WEIGHT="${FINITE_TIME_PHASE_WEIGHT:-0.1}"
    FINITE_TIME_PHASE_INTERVAL="${FINITE_TIME_PHASE_INTERVAL:-16}"
    FINITE_TIME_PHASE_WARMUP_STEPS="${FINITE_TIME_PHASE_WARMUP_STEPS:-200}"
    RUN_PREFIX="c23_finite_time_phase_from_c21"
    ;;
  *)
    echo "usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python train-jax-10m/1d_dno_fno_jax.py \
  --model cs_dno \
  --norm scale \
  --precision fp32 \
  --dataset combined_dataset_v9.npz \
  --data_fraction "$DATA_FRACTION" \
  --modes 64 \
  --width 512 \
  --n_blocks 8 \
  --latent 256 \
  --sobolev_k 1 \
  --cs_n_polys 3 \
  --cs_use_first_deriv \
  --cs_use_second_deriv \
  --cs_use_half_deriv \
  --cs_use_hilbert \
  --no-cs_use_g0_eta \
  --no-cs_use_g0_eta_dx \
  --cs_mult_hidden 128 \
  --cs_use_g1_baseline \
  --cs_g1_k_cut 0 \
  --cs_g1_fft_fp64 \
  --cs_tie_xi_out_mult \
  --cs_phi_bias_free \
  --cs_residual_eta_order 2 \
  --cs_block_k_cut 0 \
  --filter_gxi_fraction 1 \
  --filter_shape hard \
  --pushforward_gravity 1 \
  --translation_tangent_weight 0 \
  --phase_growth_weight 0 \
  --finite_time_phase_weight "$FINITE_TIME_PHASE_WEIGHT" \
  --finite_time_phase_interval "$FINITE_TIME_PHASE_INTERVAL" \
  --finite_time_phase_microbatch 4 \
  --finite_time_phase_warmup_steps "$FINITE_TIME_PHASE_WARMUP_STEPS" \
  --finite_time_phase_dt 0.01 \
  --finite_time_phase_substeps 8 \
  --finite_time_phase_picard 4 \
  --finite_time_phase_filter_fraction 0.25 \
  --finite_time_phase_reference_order 6 \
  --finite_time_phase_reference_pad 8 \
  --finite_time_phase_k_max 128 \
  --finite_time_phase_source_ids 5,6,14,7,8,9 \
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
