#!/bin/bash
# Submitted to at(1): launch the C21 tangent-weight diagnostic immediately.

cd /home/johnma/dno-fno || exit 1
export PATH="/home/johnma/.local/bin:/usr/local/bin:/usr/bin:/bin"

SCHEDULER_LOG="/tmp/c21_tangent_w100_from_c20_20260714_171442.scheduler.log"
date '+%Y-%m-%d %H:%M:%S %Z C21 scheduled launch starting' >>"$SCHEDULER_LOG"
nvidia-smi --query-gpu=index,utilization.gpu,power.draw,memory.used \
  --format=csv,noheader >>"$SCHEDULER_LOG" 2>&1

exec env \
  CUDA_VISIBLE_DEVICES=0,1 \
  RUN_NAME=c21_tangent_w100_from_c20_20260714_171442 \
  bash scripts/launch_c21_tangent_w100_from_c20.sh \
  >>"$SCHEDULER_LOG" 2>&1
