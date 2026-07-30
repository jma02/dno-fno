#!/bin/bash
# Wait for both GPUs, then run the one-epoch C27 directional screen.

set -euo pipefail
cd /home/johnma/dno-fno

PID_FILE="logs/c27_l2_finetune_queue.pid"
LOCK_FILE="logs/c27_gpu_pair_queue.lock"
mkdir -p logs

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "$(date '+%Y-%m-%d %H:%M:%S %Z') C27 fine-tune queue already active"
  exit 0
fi
trap 'rm -f "$PID_FILE"' EXIT
echo "$$" > "$PID_FILE"

idle_polls=0
while (( idle_polls < 2 )); do
  if ! compute_pids="$(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | sed '/^[[:space:]]*$/d'
  )"; then
    idle_polls=0
    echo "$(date '+%Y-%m-%d %H:%M:%S %Z') GPU probe failed; refusing to launch"
  elif [[ -z "$compute_pids" ]]; then
    ((idle_polls += 1))
  else
    idle_polls=0
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S %Z') idle_polls=$idle_polls compute_pids=${compute_pids//$'\n'/,}"
  if (( idle_polls < 2 )); then
    sleep 30
  fi
done

echo "$(date '+%Y-%m-%d %H:%M:%S %Z') launching one-epoch C27 L2 fine-tune"
CUDA_VISIBLE_DEVICES=0,1 \
RUN_NAME="c27_l2_finetune_from_c25_$(date +%Y%m%d_%H%M%S)" \
  bash scripts/launch_c27_l2_finetune_from_c25.sh
