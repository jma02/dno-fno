#!/bin/bash
# Durable host-scheduled C22 full-v9 continuation and matched Tanaka handoff.

set -euo pipefail
cd /home/johnma/dno-fno
export PATH="/home/johnma/.local/bin:/usr/local/bin:/usr/bin:/bin"

RUN_NAME="c22_phase_growth_w10_from_c20_20260715_012811"
RUN_DIR="outputs/$RUN_NAME"
SCHEDULER_LOG="/tmp/$RUN_NAME.scheduler.log"

date '+%Y-%m-%d %H:%M:%S %Z launching C22 phase-growth training' \
  >>"$SCHEDULER_LOG"
env \
  CUDA_VISIBLE_DEVICES=0,1 \
  RUN_NAME="$RUN_NAME" \
  PHASE_GROWTH_WEIGHT=10 \
  bash scripts/launch_c22_phase_growth_from_c20.sh full \
  >>"$SCHEDULER_LOG" 2>&1

if [[ ! -f "$RUN_DIR/final_ckpt/metadata.json" ]]; then
  date '+%Y-%m-%d %H:%M:%S %Z C22 exited without a final checkpoint' \
    >>"$SCHEDULER_LOG"
  exit 1
fi

date '+%Y-%m-%d %H:%M:%S %Z launching matched C22 Tanaka evaluation' \
  >>"$SCHEDULER_LOG"
env CHECKPOINT=final \
  bash scripts/eval_c16_soliton_spectral_guard.sh "$RUN_DIR" \
  >>"$SCHEDULER_LOG" 2>&1
date '+%Y-%m-%d %H:%M:%S %Z C22 training and Tanaka evaluation complete' \
  >>"$SCHEDULER_LOG"
