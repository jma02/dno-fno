#!/bin/bash
# C14 = first principled bounded-output CS-DNO retrain.
#
# Goal:
#   Stop the Tanaka NaN loop by making excessive full-output high-band Gxi
#   structurally inadmissible inside the checkpoint, rather than relying on
#   Tanaka hard negatives, eval-only guards, or auxiliary penalties.
#
# Architecture:
#   - exact G0 baseline
#   - one full-size CS block with the original eta feature stack
#   - self-adjoint multiplier tying
#   - bias-free eta/phi trunk so the residual vanishes exactly at eta=0
#   - no hard cs_block_k_cut, no G1, no stage regularizer, no hard-negative data
#   - full-output high-band envelope after baseline + residual
#
# Override DATA_FRACTION/EPOCHS/RUN_NAME for smoke runs, e.g.
#   DATA_FRACTION=0.01 EPOCHS=1 RUN_NAME=c14_smoke ./scripts/launch_c14_output_bounded_1block.sh

set -euo pipefail
cd /home/johnma/dno-fno

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME=${RUN_NAME:-c14_output_bounded_1block_${STAMP}}
DATASET=${DATASET:-combined_dataset_v9.npz}
DATA_FRACTION=${DATA_FRACTION:-1.0}
EPOCHS=${EPOCHS:-40}
BATCH_SIZE=${BATCH_SIZE:-1024}
LR=${LR:-2e-4}

UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib JAX_PLATFORMS=cuda \
uv run python train-jax-10m/1d_dno_fno_jax.py \
    --model cs_dno --norm scale --dataset "$DATASET" \
    --data_fraction "$DATA_FRACTION" \
    --modes 64 --width 512 --n_blocks 1 --latent 256 \
    --sobolev_k 1 --cs_n_polys 3 --cs_mult_hidden 128 \
    --cs_tie_xi_out_mult \
    --cs_phi_bias_free \
    --cs_output_highband_cap \
    --cs_output_highband_cap_k_cut 32.0 \
    --cs_output_highband_cap_r_max 1e-2 \
    --cs_output_highband_cap_abs_floor 5.0 \
    --batch_size "$BATCH_SIZE" --lr "$LR" --weight_decay 1e-4 \
    --epochs "$EPOCHS" \
    --skip_dno_eval \
    --skip_plots \
    --run_name "$RUN_NAME" 2>&1 | tee "/tmp/${RUN_NAME}.log"
