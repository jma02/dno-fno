#!/bin/bash
# Wait for a stable simultaneous idle window, then replace this watcher with
# the full two-GPU C20 training process.

set -euo pipefail
cd /home/johnma/dno-fno

RUN_NAME="${RUN_NAME:-c20_mode_balanced_full_$(date +%Y%m%d_%H%M%S)}"
CHECK_SECONDS="${CHECK_SECONDS:-20}"
REQUIRED_IDLE_CHECKS="${REQUIRED_IDLE_CHECKS:-3}"
LOCK_FILE="/tmp/c20_mode_balanced_full.queue.lock"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "A C20 full-run watcher is already active."
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
    echo "$(date '+%Y-%m-%d %H:%M:%S') GPUs $state; C20 run name: $RUN_NAME"
    last_state="$state"
  fi
  if (( idle_checks < REQUIRED_IDLE_CHECKS )); then
    sleep "$CHECK_SECONDS"
  fi
done

echo "$(date '+%Y-%m-%d %H:%M:%S') launching C20 on GPUs 0,1"
export CUDA_VISIBLE_DEVICES=0,1
export BATCH_SIZE=1024
export RUN_NAME
exec bash scripts/launch_c20_mode_balanced_retrain.sh full
