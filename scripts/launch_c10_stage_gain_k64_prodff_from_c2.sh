#!/bin/bash
# C10 = C2 final + production-filtered GL2 stage tangent-gain regularization.
#
# Why this exists:
#   C8 regularized the stage map with filter_fraction=2/3, while the failing
#   Tanaka/BF rollout regimes use the production GL2 filter_fraction=0.25.
#   The latest frame-by-frame scan points at abnormal temporal growth in
#   eta[k=64..128], so this run focuses the stage-gain regularizer on that
#   band under the same filtered map used by production eval.
#
# Controlled differences from C8:
#   - stage_reg_filter_fraction: 2/3 -> 0.25
#   - stage_reg band: k=16..128 -> k=64..128
#   - no harvested/model-generated pre-cascade data
#   - no train-time hard gxi limiter or static high-band penalty

set -euo pipefail
cd /home/johnma/dno-fno

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME=c10_stage_gain_k64_prodff_from_c2_$STAMP
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
    --stage_reg_k_lo 64.0 \
    --stage_reg_k_hi 128.0 \
    --stage_reg_k_low_hi 64.0 \
    --stage_reg_taper_lo 16.0 \
    --stage_reg_taper_hi 1.0 \
    --stage_reg_taper_low 16.0 \
    --stage_reg_eps_min 1e-6 \
    --stage_reg_eps_max 1e-3 \
    --stage_reg_gain_margin_rel 0.0 \
    --stage_reg_gain_margin_abs 0.0 \
    --stage_reg_response_floor 1e-8 \
    --stage_reg_reference_order 6 \
    --stage_reg_reference_pad 8 \
    --stage_reg_reference_picard 1 \
    --stage_reg_dt 0.01 \
    --stage_reg_filter_fraction 0.25 \
    --skip_dno_eval \
    --run_name "$RUN_NAME" 2>&1 | tee "/tmp/${RUN_NAME}.log"
