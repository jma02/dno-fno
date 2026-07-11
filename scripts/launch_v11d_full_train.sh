#!/bin/bash
# v11-D full train: v11-C shrunk arch + stage-tangent regularizer (C1) + PSD hinge.
#
# What changed vs v11-C-revised:
#   + Stage-tangent regularizer (C1 recipe from GL2 residual patch queue) —
#     ONLY proven NaN cure so far (3/6 baseline NaN cured on FT v8.5b).
#   + PSD hinge (weight 1e-3, warmup 10k steps) — enforces <xi, G(eta) xi> >= 0.
#   - Removed CUDA_VISIBLE_DEVICES=0: use BOTH GPUs (GPU 1 now free).
#
# NOT included: cs_fft_fp64. Verified 2026-07-06 in project_mixed_precision_fft.md
# that fp64 FFT casting does NOT cure tanaka NaN (v8_fp64 eval showed NaN=0.125,
# same as fp32). Do not re-add.
#
# Arch spec (v11-C shrunk, G_0 only — G_1 baseline REMOVED per user directive
# "we only want up to G_0, even G_1 is pushing it"):
#   - 8 blocks, latent=32, mult_hidden=32, width=32
#   - phi_bias_free, tie_xi_out_mult
#   - block_k_cut=64 (hard low-pass on learned correction)
#   - G_0 linear baseline only (no analytic G_1)
#   - Features {eta, eta^2, |D|^{1/2} eta}
#
# ~25k trainable params (45x smaller than v10's 1.13M; earlier 2224-param
# version was killed as under-parameterized per user pushback).
#
# Recipe: batch=2048 (16x v10 baseline; model is tiny so we push hard),
# lr=8e-4 (sqrt-16 scaling of v10's 2e-4), wd=1e-4, sobolev_k=1.
# Memory footprint per GPU is dominated by XLA preallocator (~75% of 49GB),
# not activations; bs=2048 activations still fit easily. If OOM drop to 1024.

set -euo pipefail
cd /home/johnma/dno-fno

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME=v11d_shrunk8blocks_stagereg_psd_$STAMP

JAX_PLATFORMS=cuda uv run python train-jax-10m/1d_dno_fno_jax.py \
    --model cs_dno --norm scale --dataset combined_dataset_v9.npz \
    --modes 64 --width 32 --n_blocks 8 --latent 32 \
    --cs_mult_hidden 32 \
    --cs_n_polys 2 \
    --no-cs_use_first_deriv \
    --no-cs_use_second_deriv \
    --cs_use_half_deriv \
    --no-cs_use_hilbert \
    --cs_tie_xi_out_mult \
    --cs_phi_bias_free \
    --cs_block_k_cut 64 \
    --sobolev_k 1 \
    --batch_size 2048 --lr 8e-4 --weight_decay 1e-4 \
    --epochs 40 \
    --psd_hinge_weight 1e-3 \
    --psd_hinge_warmup_steps 10000 \
    --stage_reg_weight 0.001 \
    --stage_reg_interval 32 \
    --stage_reg_microbatch 8 \
    --stage_reg_warmup_steps 500 \
    --stage_reg_k_lo 32 \
    --stage_reg_k_hi 128 \
    --stage_reg_k_low_hi 32 \
    --stage_reg_taper_lo 8 \
    --stage_reg_taper_hi 16 \
    --stage_reg_taper_low 8 \
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
    --run_name "$RUN_NAME" 2>&1 | tee "/tmp/${RUN_NAME}.log"

echo
echo "===== v11-D full train DONE ====="
echo "  run_dir: outputs/${RUN_NAME}"
