#!/usr/bin/env bash
# v10 CS-DNO training on Modal H100:2 (v9 dataset = v8 + steep_tanaka_v2 as source 14).
# No jacreg — clean single-variable ablation vs v8. Data upload is CPU-only (see
# `modal run modal_train.py::upload_data`). Trainer resume-safe via Modal Retries +
# per-epoch checkpoints written to /data/outputs/<run>/latest_ckpt.
set -e
cd "$(dirname "$0")"
mkdir -p logs

RUN_NAME="cs_dno_w512b8_l256_v9_h100x2_$(date +%Y%m%d_%H%M%S)"
LOG="logs/${RUN_NAME}.log"

MODAL_GPU="H100:2" nohup uv run modal run --detach modal_train.py::train \
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

echo "Launched v10 on Modal H100:2 (PID $!, run_name=$RUN_NAME)"
echo "Log: $LOG"
