#!/bin/bash
# C2 = stronger C1: same v8.5b base + v8 dataset, but reg turned up.
#   match:  0.001 → 0.003 (3× C1)
#   gain:   0.0   → 0.001 absolute λ_gain (enable the JCP09 tangent-gain term)
#   epochs: 5     → 8
# Batch increased 128 → 1024 to improve 2-GPU utilization. Stage-reg interval
# tightened 32 → 4 so the regularizer fires the same number of times per data
# pass despite the 8× fewer optimizer steps per epoch.
#
# Base: outputs/cs_dno_w512b8_l256_v8p5b_tiexo_20260622_234309/best_val_ckpt
# Architecture-critical: base is tied M_xi=M_out, so keep --cs_tie_xi_out_mult.
# Data: combined_dataset_v8.npz (NOT v9 — keeps direct C1 comparison)
# Devices: both local RTX 6000 Ada
#
# Expected wall: 5ep on 1 GPU C1 was ~10h; 8ep on 2 GPUs ≈ same wall.

set -euo pipefail
cd /home/johnma/dno-fno

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME=c2_stage_match_gain_from_v85b_$STAMP
BASE_CKPT=/home/johnma/dno-fno/outputs/cs_dno_w512b8_l256_v8p5b_tiexo_20260622_234309/best_val_ckpt

UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib JAX_PLATFORMS=cuda uv run python train-jax-10m/1d_dno_fno_jax.py \
    --model cs_dno --norm scale --dataset combined_dataset_v8.npz \
    --modes 64 --width 512 --n_blocks 8 --latent 256 \
    --sobolev_k 1 --cs_n_polys 3 --cs_mult_hidden 128 \
    --cs_tie_xi_out_mult \
    --batch_size 1024 --lr 2e-5 --weight_decay 1e-4 \
    --epochs 8 \
    --resume_from "$BASE_CKPT" \
    --stage_reg_weight 0.003 \
    --stage_reg_gain_weight 0.001 \
    --stage_reg_interval 4 \
    --stage_reg_microbatch 8 \
    --stage_reg_warmup_steps 500 \
    --stage_reg_k_lo 32.0 \
    --stage_reg_k_hi 128.0 \
    --stage_reg_k_low_hi 32.0 \
    --stage_reg_taper_lo 8.0 \
    --stage_reg_taper_hi 16.0 \
    --stage_reg_taper_low 8.0 \
    --stage_reg_eps_min 1e-6 \
    --stage_reg_eps_max 1e-3 \
    --stage_reg_gain_margin_rel 0.05 \
    --stage_reg_gain_margin_abs 1e-3 \
    --stage_reg_response_floor 1e-8 \
    --stage_reg_reference_order 6 \
    --stage_reg_reference_pad 8 \
    --stage_reg_reference_picard 1 \
    --stage_reg_dt 0.01 \
    --stage_reg_filter_fraction 0.6666666666666666 \
    --run_name "$RUN_NAME" 2>&1 | tee "/tmp/${RUN_NAME}.log"
