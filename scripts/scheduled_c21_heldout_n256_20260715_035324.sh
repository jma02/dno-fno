#!/bin/bash
# Durable C21 evaluation on ten fresh, trajectory-disjoint n=256 IC panels.

set -euo pipefail
cd /home/johnma/dno-fno
export PATH="/home/johnma/.local/bin:/usr/local/bin:/usr/bin:/bin"

RUN_DIR="outputs/c21_tangent_w100_from_c20_20260714_171442"
IC_PANEL_DIR="/home/johnma/dno-fno/data/manuscript_ic_panels_v1_20260715"
OUT_DIR="$RUN_DIR/eval_final_heldout_unguarded_n256_20260715_035324"
SCHEDULER_LOG="/tmp/c21_heldout_unguarded_n256_20260715_035324.scheduler.log"

date '+%Y-%m-%d %H:%M:%S %Z launching C21 held-out unguarded n=256 suite' \
  >>"$SCHEDULER_LOG"
env \
  CHECKPOINT=final \
  N_ICS=256 \
  ROLLOUT_BATCH_SIZE=128 \
  STAMP=20260715_035324 \
  OUT_DIR="$OUT_DIR" \
  IC_PANEL_DIR="$IC_PANEL_DIR" \
  GUARD_MODE=off \
  bash scripts/eval_manuscript_suite_n256.sh "$RUN_DIR" \
  >>"$SCHEDULER_LOG" 2>&1
date '+%Y-%m-%d %H:%M:%S %Z C21 held-out suite wrapper complete' \
  >>"$SCHEDULER_LOG"
