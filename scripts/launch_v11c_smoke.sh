#!/bin/bash
# v11-C-revised smoke: 100 batches on 1 GPU. Verify loss descends and no NaN.
#
# Recipe:
#   - 8 blocks (keep cascade), latent=8, mult_hidden=8 (shrink each block hard)
#   - phi_bias_free (η=0 exact linear limit)
#   - tie_xi_out_mult (self-adjoint)
#   - block_k_cut=64 (hard bandlimit on the learned correction)
#   - use_g1_baseline + g1_k_cut=64 (closed-form G_1 already baked in)
#   - Features: {η, η², |D|^{1/2}η} (drop d_x, d_xx, H — from weight audit)
#
# ~2,200 params total vs v10's 8M (3,600× shrink).

set -euo pipefail
cd /home/johnma/dno-fno

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME=v11c_smoke_$STAMP

CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda uv run python train-jax-10m/1d_dno_fno_jax.py \
    --model cs_dno --norm scale --dataset combined_dataset_v9.npz \
    --modes 64 --width 32 --n_blocks 8 --latent 8 \
    --cs_mult_hidden 8 \
    --cs_n_polys 2 \
    --no-cs_use_first_deriv \
    --no-cs_use_second_deriv \
    --cs_use_half_deriv \
    --no-cs_use_hilbert \
    --cs_use_g1_baseline \
    --cs_g1_k_cut 64 \
    --cs_tie_xi_out_mult \
    --cs_phi_bias_free \
    --cs_block_k_cut 64 \
    --sobolev_k 1 \
    --batch_size 128 --lr 2e-4 --weight_decay 1e-4 \
    --epochs 1 \
    --data_fraction 0.01 \
    --run_name "$RUN_NAME" 2>&1 | tee "/tmp/${RUN_NAME}.log"
