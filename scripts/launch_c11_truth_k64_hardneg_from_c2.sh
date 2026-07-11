#!/bin/bash
# C11 = C2 final + clean truth-generated k=64..128 hard negatives.
#
# This is the controlled data-side fallback if C10 is negative. It avoids the
# bad C6/C7 axis by building hard negatives from cached f64 truth trajectories,
# not from model-generated pre-NaN states.

set -euo pipefail
cd /home/johnma/dno-fno

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME=c11_truth_k64_hardneg_from_c2_$STAMP
BASE_CKPT=/home/johnma/dno-fno/outputs/c2_stage_match_gain_from_v85b_20260707_053707/final_ckpt
TARGET_PACK=data/tanaka_truth_k64_hardneg_v1.npz
BALANCED_DATA=data/balanced_truth_k64_hardneg_v1_100k.npz

if [[ ! -f "$TARGET_PACK" ]]; then
    UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib JAX_PLATFORMS=cuda \
    uv run python scripts/build_truth_tanaka_k64_hardneg.py \
        --output "$TARGET_PACK"
fi

if [[ ! -f "$BALANCED_DATA" ]]; then
    UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib \
    uv run python scripts/build_balanced_precascade_finetune.py \
        --target "$TARGET_PACK" \
        --output "$BALANCED_DATA" \
        --n_base 90000 \
        --n_target 10000 \
        --target_source_id 51
fi

UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib JAX_PLATFORMS=cuda \
uv run python train-jax-10m/1d_dno_fno_jax.py \
    --model cs_dno --norm scale --dataset "$(basename "$BALANCED_DATA")" \
    --modes 64 --width 512 --n_blocks 8 --latent 256 \
    --sobolev_k 1 --cs_n_polys 3 --cs_mult_hidden 128 \
    --cs_tie_xi_out_mult \
    --batch_size 1024 --lr 5e-7 --weight_decay 1e-4 \
    --epochs 12 \
    --resume_from "$BASE_CKPT" \
    --reset_opt_state \
    --skip_dno_eval \
    --run_name "$RUN_NAME" 2>&1 | tee "/tmp/${RUN_NAME}.log"
