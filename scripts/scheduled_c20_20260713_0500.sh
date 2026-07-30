#!/bin/bash
# Submitted to at(1): start the fresh C20 retrain at 05:00 EDT.

cd /home/johnma/dno-fno || exit 1
export PATH="/home/johnma/.local/bin:/usr/local/bin:/usr/bin:/bin"

SCHEDULER_LOG="/tmp/c20_mode_balanced_full_20260713_050000.scheduler.log"
date '+%Y-%m-%d %H:%M:%S %Z C20 scheduled launch starting' >>"$SCHEDULER_LOG"
nvidia-smi --query-gpu=index,utilization.gpu,power.draw,memory.used \
  --format=csv,noheader >>"$SCHEDULER_LOG" 2>&1

exec env \
  CUDA_VISIBLE_DEVICES=0,1 \
  BATCH_SIZE=1024 \
  RUN_NAME=c20_mode_balanced_full_20260713_050000 \
  bash scripts/launch_c20_mode_balanced_retrain.sh full \
  >>"$SCHEDULER_LOG" 2>&1
