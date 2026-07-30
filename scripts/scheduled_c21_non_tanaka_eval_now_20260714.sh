#!/bin/bash
# Submitted to at(1): launch C21 non-Tanaka evaluation despite GPU sharing.

cd /home/johnma/dno-fno || exit 1
export PATH="/home/johnma/.local/bin:/usr/local/bin:/usr/bin:/bin"

RUN_DIR="outputs/c21_tangent_w100_from_c20_20260714_171442"
EVAL_LOG="/tmp/c21_tangent_w100_from_c20_20260714_171442.non_tanaka_immediate.log"
date '+%Y-%m-%d %H:%M:%S %Z launching shared-GPU C21 non-Tanaka suite' >>"$EVAL_LOG"
nvidia-smi --query-gpu=index,utilization.gpu,power.draw,memory.used \
  --format=csv,noheader >>"$EVAL_LOG" 2>&1

exec env CHECKPOINT=final \
  bash scripts/eval_guarded_non_tanaka_suite_n32.sh "$RUN_DIR" \
  >>"$EVAL_LOG" 2>&1
