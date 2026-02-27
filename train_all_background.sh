#!/bin/bash
# Script to run FNO grid search + evaluation in the background

REPO_ROOT=$(dirname "$(readlink -f "$0")")
CAMPAIGN_TAG="gridfno_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$REPO_ROOT/outputs/${CAMPAIGN_TAG}.log"
CSV_FILE="$REPO_ROOT/outputs/${CAMPAIGN_TAG}_results.csv"

mkdir -p "$REPO_ROOT/outputs"

echo "Starting grid search campaign..."
echo "Campaign tag: $CAMPAIGN_TAG"
echo "Logs will be written to $LOG_FILE"
echo "Results CSV will be written to $CSV_FILE"

nohup bash -lc "cd \"$REPO_ROOT\" && \
  source \"$REPO_ROOT/.venv/bin/activate\" && \
  export PYTHONPATH=\"$REPO_ROOT:\$PYTHONPATH\" && \
  python -u \"$REPO_ROOT/train/train_all.py\" --device cuda:0 --epochs 350 --plot-every 100 --campaign-tag \"$CAMPAIGN_TAG\" && \
  python -u \"$REPO_ROOT/notebooks/eval_grid_search.py\" --name-contains \"$CAMPAIGN_TAG\" --device cuda:0 --csv-out \"$CSV_FILE\"" \
  > "$LOG_FILE" 2>&1 &

echo "Grid search started with PID $!"
