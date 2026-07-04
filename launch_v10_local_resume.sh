#!/usr/bin/env bash
# v10 CS-DNO local resume of the H100:2 Modal run (ep10 latest_ckpt mirrored to
# outputs/cs_dno_w512b8_l256_v9_h100x2_20260703_035745/latest_ckpt). Same run_name
# → trainer's auto-resume path (1d_dno_fno_jax.py:794) kicks in and starts at ep11.
set -e
cd "$(dirname "$0")"
mkdir -p logs

RUN_NAME="cs_dno_w512b8_l256_v9_h100x2_20260703_035745"
LOG="logs/${RUN_NAME}_local_resume_$(date +%Y%m%d_%H%M%S).log"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
JAX_PLATFORMS=cuda nohup uv run python train-jax-10m/1d_dno_fno_jax.py \
    --model cs_dno --norm scale --dataset combined_dataset_v9.npz \
    --modes 64 --width 512 --n_blocks 8 --latent 256 --sobolev_k 1 \
    --batch_size 256 --lr 0.0002 --weight_decay 0.0001 --epochs 40 \
    --cs_n_polys 3 --cs_mult_hidden 128 --run_name "$RUN_NAME" \
    > "$LOG" 2>&1 &

echo "Launched v10 local resume PID $! ($RUN_NAME)"
echo "Log: $LOG"
