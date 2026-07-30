#!/bin/bash
# Submitted to at(1): evaluate both Tanaka panels immediately after C21 exits.

set -euo pipefail
cd /home/johnma/dno-fno
export PATH="/home/johnma/.local/bin:/usr/local/bin:/usr/bin:/bin"

RUN_DIR="outputs/c21_tangent_w100_from_c20_20260714_171442"
TRAIN_PID=505668
HANDOFF_LOG="/tmp/c21_tangent_w100_from_c20_20260714_171442.eval_handoff.log"

date '+%Y-%m-%d %H:%M:%S %Z waiting for C21 training' >>"$HANDOFF_LOG"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 20
done

if [[ ! -f "$RUN_DIR/final_ckpt/metadata.json" ]]; then
  date '+%Y-%m-%d %H:%M:%S %Z C21 exited without a final checkpoint' >>"$HANDOFF_LOG"
  exit 1
fi

date '+%Y-%m-%d %H:%M:%S %Z launching matched C21 Tanaka evaluation' >>"$HANDOFF_LOG"
exec env CHECKPOINT=final \
  bash scripts/eval_c16_soliton_spectral_guard.sh "$RUN_DIR" \
  >>"$HANDOFF_LOG" 2>&1
