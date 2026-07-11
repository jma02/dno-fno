#!/bin/bash
# C13 = C10 final + clean truth hard negatives for the remaining failure set.
#
# Purpose:
#   Do not repeat broad Tanaka fine-tuning or C2-harvested contaminated rows.
#   C10 already fixed part of the failure mechanism but still NaNs on Tanaka
#   cases 5 and 22, with case 27 finite but high-error. This fine-tune starts
#   from C10 and only adds clean truth-state, k=64..128 perturbation coverage
#   around those observed failure windows, with heavy v8 replay.

set -euo pipefail
cd /home/johnma/dno-fno

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME=c13_truth_remaining_hardneg_from_c10_$STAMP
BASE_CKPT=/home/johnma/dno-fno/outputs/c10_stage_gain_k64_prodff_from_c2_20260709_055358/final_ckpt
TARGET_PACK=data/tanaka_truth_k64_remaining_hardneg_v1.npz
BALANCED_DATA=data/balanced_truth_k64_remaining_hardneg_v1_132k.npz

if [[ ! -d "$BASE_CKPT" ]]; then
    echo "missing base checkpoint: $BASE_CKPT" >&2
    exit 1
fi

if [[ ! -f "$TARGET_PACK" ]]; then
    UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib JAX_PLATFORMS=cuda \
    uv run python scripts/build_truth_tanaka_k64_hardneg.py \
        --output "$TARGET_PACK" \
        --case_ids 5 22 27 \
        --snapshot_times 24.0 28.0 32.0 33.6 36.0 40.0 44.0 48.0 52.0 56.0 80.0 120.0 160.0 188.0 \
        --n_perturb_per_state 256 \
        --perturb_k_lo 64.0 \
        --perturb_k_hi 128.0 \
        --perturb_rel_min 1e-6 \
        --perturb_rel_max 1e-3 \
        --filter_fraction 0.25
fi

if [[ ! -f "$BALANCED_DATA" ]]; then
    UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib \
    uv run python scripts/build_balanced_precascade_finetune.py \
        --target "$TARGET_PACK" \
        --output "$BALANCED_DATA" \
        --n_base 120000 \
        --n_target 12000 \
        --target_source_id 52
fi

UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib \
TF_GPU_ALLOCATOR=cuda_malloc_async JAX_PLATFORMS=cuda \
uv run python train-jax-10m/1d_dno_fno_jax.py \
    --model cs_dno --norm scale --dataset "$(basename "$BALANCED_DATA")" \
    --modes 64 --width 512 --n_blocks 8 --latent 256 \
    --sobolev_k 1 --cs_n_polys 3 --cs_mult_hidden 128 \
    --cs_tie_xi_out_mult \
    --batch_size 1024 --lr 2e-7 --weight_decay 1e-4 \
    --epochs 8 \
    --resume_from "$BASE_CKPT" \
    --reset_opt_state \
    --skip_dno_eval \
    --skip_plots \
    --run_name "$RUN_NAME" 2>&1 | tee "/tmp/${RUN_NAME}.log"
