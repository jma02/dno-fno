#!/usr/bin/env bash
# v10 cs_dno retrain on combined_dataset_v9 (v8 + steep_tanaka_v2 as source 14).
# Same architecture as v8 fp32 baseline; no jacreg (isolate the effect of the
# steep_tanaka data addition). steep_tanaka_v2 fills the h ∈ [0.20, 0.35],
# a/h ∈ [0.25, 0.45] hole where every prior CS-DNO NaN'd (see NAN_INVESTIGATION.md).
set -e
cd "$(dirname "$0")"
mkdir -p logs
RUN_NAME="cs_dno_w512b8_l256_v9_$(date +%Y%m%d_%H%M%S)"
XLA_FLAGS="--xla_gpu_triton_gemm_any=true --xla_gpu_enable_latency_hiding_scheduler=true" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
JAX_PLATFORMS=cuda nohup uv run python train-jax-10m/1d_dno_fno_jax.py \
    --model cs_dno \
    --norm scale \
    --dataset combined_dataset_v9.npz \
    --modes 64 --width 512 --n_blocks 8 --latent 256 \
    --sobolev_k 1 \
    --batch_size 768 \
    --lr 0.0002 \
    --weight_decay 0.0001 \
    --epochs 40 \
    --cs_n_polys 3 \
    --cs_mult_hidden 128 \
    --run_name "$RUN_NAME" \
    > "logs/${RUN_NAME}.log" 2>&1 &
echo "Launched v10 PID $! ($RUN_NAME)"
