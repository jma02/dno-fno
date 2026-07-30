#!/bin/bash
# Wait for both GPUs to be stably idle, then run the C21 non-Tanaka suite.

set -euo pipefail
cd /home/johnma/dno-fno

RUN_DIR="${RUN_DIR:-outputs/c21_tangent_w100_from_c20_20260714_171442}"
CHECK_SECONDS="${CHECK_SECONDS:-20}"
REQUIRED_IDLE_CHECKS="${REQUIRED_IDLE_CHECKS:-3}"
QUEUE_LOG="${QUEUE_LOG:-/tmp/c21_tangent_w100_from_c20_20260714_171442.non_tanaka_queue.log}"
LOCK_FILE="/tmp/c21_tangent_w100_from_c20_20260714_171442.non_tanaka_queue.lock"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "A C21 non-Tanaka evaluation watcher is already active."
  exit 1
fi

idle_checks=0
last_state=""
while (( idle_checks < REQUIRED_IDLE_CHECKS )); do
  active_processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
  gpu_utilization="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
  if [[ -z "$active_processes" ]] && ! awk '$1 > 5 { busy = 1 } END { exit busy }' <<<"$gpu_utilization"; then
    ((idle_checks += 1))
    state="idle ${idle_checks}/${REQUIRED_IDLE_CHECKS}"
  else
    idle_checks=0
    state="busy"
  fi
  if [[ "$state" != "$last_state" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S %Z') GPUs $state" >>"$QUEUE_LOG"
    last_state="$state"
  fi
  if (( idle_checks < REQUIRED_IDLE_CHECKS )); then
    sleep "$CHECK_SECONDS"
  fi
done

echo "$(date '+%Y-%m-%d %H:%M:%S %Z') launching C21 non-Tanaka suite" >>"$QUEUE_LOG"
exec env CHECKPOINT=final \
  bash scripts/eval_guarded_non_tanaka_suite_n32.sh "$RUN_DIR" \
  >>"$QUEUE_LOG" 2>&1
