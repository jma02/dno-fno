#!/usr/bin/env bash
# v9 cs_dno retrain with Jacobian regularization.
# Attacks the model's positive Lyapunov exponent on the Tanaka manifold (λ~0.07/s
# in mid-k for a/h>0.28). Single-step loss has no preference between two models
# of identical accuracy with stable vs unstable Jacobian — this term breaks the tie.
#
# Same architecture as v8 fp32 baseline; only adds the band-restricted Hutchinson
# penalty E_v[||P_hi · (∂M/∂η) · P_hi v||²] for |k|>=32. Warmup over 500 steps so
# data fit stabilizes before the penalty kicks in.
set -e
cd "$(dirname "$0")"
mkdir -p logs
RUN_NAME="cs_dno_w512b8_l256_v8_jacreg_$(date +%Y%m%d_%H%M%S)"
JAX_PLATFORMS=cuda nohup uv run python train-jax-10m/1d_dno_fno_jax.py \
    --model cs_dno \
    --norm scale \
    --dataset combined_dataset_v8.npz \
    --modes 64 --width 512 --n_blocks 8 --latent 256 \
    --sobolev_k 1 \
    --batch_size 256 \
    --lr 0.0002 \
    --weight_decay 0.0001 \
    --epochs 40 \
    --cs_n_polys 3 \
    --cs_mult_hidden 128 \
    --jac_reg_lambda 0.001 \
    --jac_reg_kcut 32 \
    --jac_reg_warmup_steps 500 \
    --run_name "$RUN_NAME" \
    > "logs/${RUN_NAME}.log" 2>&1 &
echo "Launched v9 PID $! ($RUN_NAME)"
