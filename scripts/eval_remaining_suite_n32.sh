#!/bin/bash
# Evaluate the eight registered families not covered by eval_final_2reg.sh.

set -euo pipefail
cd /home/johnma/dno-fno

RUN_DIR="${1:?usage: scripts/eval_remaining_suite_n32.sh RUN_DIR}"
CHECKPOINT="${CHECKPOINT:-best}"
TRUTH_CACHE="${TRUTH_CACHE:-outputs/cs_dno_w512b8_l256_v9_h100x2_20260703_035745/eval_suite_f64h_n32}"
WAIT_DIR="${WAIT_DIR:-}"
if [[ -n "$WAIT_DIR" ]]; then
    echo "Waiting for the two-regime evaluation in $WAIT_DIR"
    while [[ ! -f "$WAIT_DIR/tanaka_g0/tanaka_g0_summary.json" ||
             ! -f "$WAIT_DIR/bf_g1/bf_g1_summary.json" ]]; do
        sleep 10
    done
fi
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$RUN_DIR/eval_${CHECKPOINT}_remaining_suite_n32_f64h_batched_cached_$STAMP"
mkdir -p "$OUT_DIR"
printf '%s\n' "$OUT_DIR" > "$RUN_DIR/last_eval_remaining_suite_n32.txt"

COMMON_ARGS=(
    --run_dir "$RUN_DIR"
    --checkpoint "$CHECKPOINT"
    --n_ics 32
    --f64_harness
    --batched_surrogate
    --truth_cache "$TRUTH_CACHE"
)

run_pair() {
    local regime0="$1"
    local regime1="$2"
    local out0="$OUT_DIR/$regime0"
    local out1="$OUT_DIR/$regime1"
    mkdir -p "$out0" "$out1"

    CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
        uv run python -m solver.evals.eval_suite \
        "${COMMON_ARGS[@]}" --regimes "$regime0" --output_dir "$out0" \
        2>&1 | tee "$out0/eval.log" &
    local pid0=$!

    CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda \
        uv run python -m solver.evals.eval_suite \
        "${COMMON_ARGS[@]}" --regimes "$regime1" --output_dir "$out1" \
        2>&1 | tee "$out1/eval.log" &
    local pid1=$!

    wait "$pid0" "$pid1"
}

run_pair tanaka_g1 bf_g0
run_pair bf_modal random_sea_deep
run_pair random_sea_finite linear
run_pair stokes_deep stokes_finite

uv run python - "$OUT_DIR" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summaries: dict[str, dict[str, object]] = {}
for path in sorted(root.glob("*/*_summary.json")):
    regime = path.stem.removesuffix("_summary")
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    summaries[regime] = summary
    print(
        f"{regime}: NaN={summary['nan_rate']:.3f} "
        f"div={summary['divergence_rate_final']:.3f} "
        f"eta_med_tf={summary['rel_l2_eta_median_tfinal']:.6g} "
        f"eta_p95_tf={summary['rel_l2_eta_p95_tfinal']:.6g}"
    )

with (root / "all_summaries.json").open("w", encoding="utf-8") as handle:
    json.dump(summaries, handle, indent=2)
PY

echo "Output: $OUT_DIR"
