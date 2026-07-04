#!/usr/bin/env bash
# v10 CS-DNO training on Modal H100:4 — resume of the H100:2 run started at 03:57 EDT.
# Auto-resume picks up the ckpt inside /outputs/<RUN_NAME>/latest_ckpt (see
# 1d_dno_fno_jax.py:790 — same run_name → auto-resume kicks in).
set -e
cd "$(dirname "$0")"
mkdir -p logs

RUN_NAME="cs_dno_w512b8_l256_v9_h100x2_20260703_035745"
LOG="logs/${RUN_NAME}_h100x4_resume_$(date +%Y%m%d_%H%M%S).log"

MODAL_GPU="H100:8" nohup uv run modal run --detach modal_train.py::train \
    --dataset combined_dataset_v9.npz \
    --test-dataset test_dno_rescaled.npz \
    --run-name "$RUN_NAME" \
    --model-kind cs_dno \
    --norm scale \
    --precision fp32 \
    --modes 64 --width 512 --n-blocks 8 --latent 256 \
    --sobolev-k 1 \
    --batch-size 256 \
    --lr 0.0002 --weight-decay 0.0001 \
    --epochs 40 \
    --cs-mult-hidden 128 \
    > "$LOG" 2>&1 &

echo "Launched v10 H100:4 resume (PID $!, run_name=$RUN_NAME)"
echo "Log: $LOG"
