#!/bin/bash
# v11-E: exact v10 arch + block_k_cut=64. ONE variable changed vs v10.
#
# v10 (baseline, taken from outputs/cs_dno_w512b8_l256_v9_h100x2 config.json):
#   width 512, n_blocks 8, latent 256, cs_mult_hidden 128
#   cs_n_polys 3, all derivs on (first, second, half, Hilbert)
#   G_0 only (cs_use_g1_baseline=False), tie_xi_out_mult=False, phi_bias_free=False
#   batch 256, lr 2e-4, wd 1e-4, sobolev_k 1, fp32
#
# v11-D confounded the block_k_cut experiment by ALSO shrinking the arch 80x.
# v11-E isolates: same 1.13M params as v10, just add block_k_cut=64 hard
# low-pass on m_xi and out_hat inside every block.
#
# NO stage_reg, NO PSD hinge, NO G_1. Nothing but the block_k_cut delta.

set -euo pipefail
cd /home/johnma/dno-fno

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME=v11e_v10arch_blockkcut64_$STAMP

JAX_PLATFORMS=cuda uv run python train-jax-10m/1d_dno_fno_jax.py \
    --model cs_dno --norm scale --dataset combined_dataset_v9.npz \
    --modes 64 --width 512 --n_blocks 8 --latent 256 \
    --cs_mult_hidden 128 \
    --cs_n_polys 3 \
    --cs_use_first_deriv \
    --cs_use_second_deriv \
    --cs_use_half_deriv \
    --cs_use_hilbert \
    --cs_block_k_cut 64 \
    --sobolev_k 1 \
    --batch_size 256 --lr 2e-4 --weight_decay 1e-4 \
    --epochs 40 \
    --run_name "$RUN_NAME" 2>&1 | tee "/tmp/${RUN_NAME}.log"

echo
echo "===== v11-E full train DONE ====="
echo "  run_dir: outputs/${RUN_NAME}"
