#!/bin/bash
# C8 = C2 final + stronger GL2 stage tangent-gain regularization.
#
# Intent:
#   Attack the model-side mid/high-band tangent amplification found in the
#   2026-07-08 NPZ NaN investigation, without repeating the bad Tanaka
#   pre-cascade fine-tune loop (C6/C7).
#
# Differences from C2:
#   - resume from corrected C2 final checkpoint, params only
#   - reset optimizer and use smaller lr
#   - no harvested Tanaka target data; dataset remains combined_dataset_v8.npz
#   - widen tangent band from k>=32 to k>=16 because failures already ignite in
#     the 16..32 band
#   - reduce match weight and increase/tighten gain hinge
#
# Eval after training:
#   1. no-limiter tanaka_g0 + bf_g1, final checkpoint
#   2. optional guard eval via scripts/eval_c2_nan_guard.sh-equivalent settings

set -euo pipefail
cd /home/johnma/dno-fno

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME=c8_stage_gain_k16_from_c2_$STAMP
BASE_CKPT=/home/johnma/dno-fno/outputs/c2_stage_match_gain_from_v85b_20260707_053707/final_ckpt

UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib JAX_PLATFORMS=cuda \
uv run python train-jax-10m/1d_dno_fno_jax.py \
    --model cs_dno --norm scale --dataset combined_dataset_v8.npz \
    --modes 64 --width 512 --n_blocks 8 --latent 256 \
    --sobolev_k 1 --cs_n_polys 3 --cs_mult_hidden 128 \
    --cs_tie_xi_out_mult \
    --batch_size 1024 --lr 5e-6 --weight_decay 1e-4 \
    --epochs 4 \
    --resume_from "$BASE_CKPT" \
    --reset_opt_state \
    --stage_reg_weight 0.001 \
    --stage_reg_gain_weight 0.02 \
    --stage_reg_interval 4 \
    --stage_reg_microbatch 8 \
    --stage_reg_warmup_steps 250 \
    --stage_reg_k_lo 16.0 \
    --stage_reg_k_hi 128.0 \
    --stage_reg_k_low_hi 16.0 \
    --stage_reg_taper_lo 8.0 \
    --stage_reg_taper_hi 16.0 \
    --stage_reg_taper_low 8.0 \
    --stage_reg_eps_min 1e-6 \
    --stage_reg_eps_max 1e-3 \
    --stage_reg_gain_margin_rel 0.0 \
    --stage_reg_gain_margin_abs 0.0 \
    --stage_reg_response_floor 1e-8 \
    --stage_reg_reference_order 6 \
    --stage_reg_reference_pad 8 \
    --stage_reg_reference_picard 1 \
    --stage_reg_dt 0.01 \
    --stage_reg_filter_fraction 0.6666666666666666 \
    --skip_dno_eval \
    --run_name "$RUN_NAME" 2>&1 | tee "/tmp/${RUN_NAME}.log"
