#!/usr/bin/env bash
# 2-epoch fp64 smoke run on Modal H100:2 to measure post-jit-warmup min/epoch.
# Matches cs_dno_w512b8_l256_v5 hyperparameters except epochs=2 and precision=fp64.
set -euo pipefail

RUN_NAME="cs_dno_w512b8_l256_v7_fp64_smoke_h100x2_$(date +%Y%m%d_%H%M%S)"
echo "run_name: $RUN_NAME"

MODAL_GPU="H100:2" uv run modal run modal_train.py::train \
  --dataset combined_dataset_v7_trim.npz \
  --run-name "$RUN_NAME" \
  --precision fp64 \
  --model-kind cs_dno \
  --epochs 2 \
  --batch-size 256 \
  --lr 0.0002 \
  --weight-decay 0.0001 \
  --modes 64 \
  --width 512 \
  --n-blocks 8 \
  --sobolev-k 1 \
  --norm scale \
  --seed 0 \
  --skip-dno-eval \
  --latent 256 \
  --cs-mult-hidden 128 \
  --cs-g1-k-cut 128
