#!/usr/bin/env bash
# Run eval_suite_f64h on v8 best_val_ckpt, then chain gif generation per regime.
# Usage: nohup bash scripts/eval_v8_chain.sh > logs/eval_v8_chain.log 2>&1 &
set -euo pipefail

RUN_DIR="outputs/cs_dno_w512b8_l256_v8_2gpu_20260619_024621"
EVAL_DIR="${RUN_DIR}/eval_suite_f64h"

echo "=== [$(date)] eval_suite_f64h ==="
uv run python -m solver.evals.eval_suite \
    --run_dir "${RUN_DIR}" \
    --checkpoint best \
    --f64_harness \
    --gpu

echo "=== [$(date)] animate trajs ==="
shopt -s nullglob
for trajs in "${EVAL_DIR}"/*_trajs.npz; do
    echo "--- animating ${trajs}"
    uv run python -m solver.evals.animate_trajs \
        --trajs "${trajs}" \
        --format gif \
        --fps 24
done
echo "=== [$(date)] done ==="
