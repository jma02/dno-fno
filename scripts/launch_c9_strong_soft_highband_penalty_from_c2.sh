#!/bin/bash
# C9 = C2 final fine-tune with a stronger soft learned-gxi high-band excess penalty.
#
# This is not a repeat of C4: C4 used λ_hi=3e-5, which contributed <1% of the
# supervised loss scale and did not move the high-band diagnostic. C9 tests
# whether the full-output envelope penalty has enough leverage when weighted
# at a meaningful scale, without hard cs_block_k_cut and without contaminated
# C2-harvested pre-cascade target rows.

set -euo pipefail
cd /home/johnma/dno-fno

STAMP=$(date +%Y%m%d_%H%M%S)
GXI_HI_WEIGHT=${GXI_HI_WEIGHT:-1e-3}
EPOCHS=${EPOCHS:-2}
LR=${LR:-5e-6}
WEIGHT_TAG=${GXI_HI_WEIGHT//./p}
WEIGHT_TAG=${WEIGHT_TAG//-/m}
RUN_NAME=c9_strong_soft_hiband_w${WEIGHT_TAG}_from_c2_$STAMP
BASE_CKPT=/home/johnma/dno-fno/outputs/c2_stage_match_gain_from_v85b_20260707_053707/final_ckpt

UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib JAX_PLATFORMS=cuda uv run python train-jax-10m/1d_dno_fno_jax.py \
    --model cs_dno --norm scale --dataset combined_dataset_v8.npz \
    --modes 64 --width 512 --n_blocks 8 --latent 256 \
    --sobolev_k 1 --cs_n_polys 3 --cs_mult_hidden 128 \
    --cs_tie_xi_out_mult \
    --batch_size 1024 --lr "$LR" --weight_decay 1e-4 \
    --epochs "$EPOCHS" \
    --resume_from "$BASE_CKPT" \
    --reset_opt_state \
    --gxi_highband_penalty_weight "$GXI_HI_WEIGHT" \
    --gxi_highband_penalty_k_cut 32.0 \
    --gxi_highband_penalty_r_target 1e-2 \
    --gxi_highband_penalty_abs_floor 5.0 \
    --gxi_highband_penalty_temperature 1e-2 \
    --gxi_highband_penalty_gate_sharpness 10.0 \
    --gxi_highband_penalty_warmup_steps 500 \
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
    --skip_dno_eval \
    --run_name "$RUN_NAME" 2>&1 | tee "/tmp/${RUN_NAME}.log"
