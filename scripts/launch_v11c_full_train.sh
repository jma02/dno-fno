#!/bin/bash
# v11-C-revised full train: 2 GPU, 40 epochs, full combined_dataset_v9.
#
# Arch spec (from block-7 ablation-informed design in notes/v11_light_arch_design.md):
#   - 8 blocks (preserve compensating-cascade structure, ablation-validated as essential)
#   - latent=8, mult_hidden=8, width=32 (drastic per-block shrink)
#   - phi_bias_free, tie_xi_out_mult (structural: η=0 exact, self-adjoint)
#   - block_k_cut=64 (hard bandlimit on learned correction, kills mid-k noise cascade)
#   - use_g1_baseline + g1_k_cut=64 (analytic G_1 as strong prior)
#   - Features: {η, η², |D|^{1/2}η}
#
# ~2,224 trainable params (3,600× smaller than v10's 8M).
#
# Recipe matches v10 (batch=128, lr=2e-4, wd=1e-4, sobolev_k=1) so any delta is
# purely architectural. Estimated wall on 2× RTX 6000 Ada: ~5h for 40 epochs
# (~7 min/epoch based on smoke timing).

set -euo pipefail
cd /home/johnma/dno-fno

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME=v11c_shrunk8blocks_$STAMP

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
    --epochs 40 \
    --run_name "$RUN_NAME" 2>&1 | tee "/tmp/${RUN_NAME}.log"

echo
echo "===== v11-C-revised full train DONE ====="
echo "  run_dir: outputs/${RUN_NAME}"
