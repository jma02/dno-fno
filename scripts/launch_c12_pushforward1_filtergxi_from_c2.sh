#!/bin/bash
# C12 = C2 final + one-step pushforward fine-tune with production gxi filtering.
#
# This is the controlled closed-loop fallback if C10 is negative. It exposes the
# model to one small surrogate-advanced state, then trains against a fresh
# Craig-Sulem target on that state. The eval should use --filter_gxi, matching
# the train-time filtered prediction path.

set -euo pipefail
cd /home/johnma/dno-fno

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME=c12_pushforward1_filtergxi_from_c2_$STAMP
BASE_CKPT=/home/johnma/dno-fno/outputs/c2_stage_match_gain_from_v85b_20260707_053707/final_ckpt

UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib JAX_PLATFORMS=cuda \
uv run python train-jax-10m/1d_dno_fno_jax.py \
    --model cs_dno --norm scale --dataset combined_dataset_v8.npz \
    --modes 64 --width 512 --n_blocks 8 --latent 256 \
    --sobolev_k 1 --cs_n_polys 3 --cs_mult_hidden 128 \
    --cs_tie_xi_out_mult \
    --batch_size 1024 --lr 2e-6 --weight_decay 1e-4 \
    --epochs 2 \
    --resume_from "$BASE_CKPT" \
    --reset_opt_state \
    --filter_gxi_fraction 0.25 \
    --pushforward_steps 1 \
    --pushforward_weight 0.25 \
    --pushforward_dt 0.01 \
    --pushforward_order 4 \
    --pushforward_pad_factor 4 \
    --skip_dno_eval \
    --run_name "$RUN_NAME" 2>&1 | tee "/tmp/${RUN_NAME}.log"
