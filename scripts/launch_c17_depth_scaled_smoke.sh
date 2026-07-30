#!/bin/bash
# Matched 1%-v9 C16 smoke with only the learned residual depth-normalized.

set -euo pipefail
cd /home/johnma/dno-fno

RUN_NAME="${RUN_NAME:-c17_depth_scaled_residual_smoke_$(date +%Y%m%d_%H%M%S)}"
WAIT_FILE="${WAIT_FILE:-}"
if [[ -n "$WAIT_FILE" ]]; then
  echo "Waiting for $WAIT_FILE"
  while [[ ! -f "$WAIT_FILE" ]]; do
    sleep 10
  done
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python train-jax-10m/1d_dno_fno_jax.py \
  --model cs_dno \
  --norm scale \
  --dataset combined_dataset_v9.npz \
  --data_fraction 0.01 \
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
  --cs_depth_scaled_residual \
  --hadamard_weight 1e-2 \
  --hadamard_interval 16 \
  --hadamard_microbatch 8 \
  --hadamard_warmup_steps 500 \
  --hadamard_k_max 128 \
  --hadamard_sobolev_order 1 \
  --hadamard_relative_eps_min 1e-3 \
  --hadamard_relative_eps_max 3e-3 \
  --hadamard_eta_scale_floor 1e-3 \
  --hadamard_denominator_floor 1e-12 \
  --batch_size 1024 \
  --lr 2e-5 \
  --lr_warmup_steps 50 \
  --weight_decay 1e-4 \
  --epochs 1 \
  --total_epochs 40 \
  --skip_dno_eval \
  --skip_plots \
  --run_name "$RUN_NAME" 2>&1 | tee "/tmp/$RUN_NAME.log"

CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
uv run python scripts/probe_tanaka_case27_onestep.py \
  --checkpoint final \
  outputs/c16_hadamard_g01_oeta2_smoke_20260709_221222 \
  "outputs/$RUN_NAME" \
  2>&1 | tee "outputs/$RUN_NAME/case27_onestep_probe.log"
