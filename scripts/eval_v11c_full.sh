#!/bin/bash
# Eval v11-C-revised: batched surrogate tanaka_g0 + bf_g1 f64_harness at n_ics=32.
# Uses the same eval CLI as the 2×2 (matching baseline v10's off-arm) so numbers
# are directly comparable.
#
# Run this after v11c_shrunk8blocks training completes.

set -euo pipefail
cd /home/johnma/dno-fno

RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR=$(ls -td outputs/v11c_shrunk8blocks_* 2>/dev/null | head -1)
    if [[ -z "$RUN_DIR" ]]; then
        echo "No v11c_shrunk8blocks_* run_dir found" >&2
        exit 1
    fi
fi

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$RUN_DIR/eval_v11c_$STAMP"
mkdir -p "$OUT_DIR"

echo "===== v11-C-revised eval ====="
echo "  run_dir: $RUN_DIR"
echo "  out_dir: $OUT_DIR"

TANAKA_OUT="$OUT_DIR/tanaka_g0"
BF_OUT="$OUT_DIR/bf_g1"
mkdir -p "$TANAKA_OUT" "$BF_OUT"

# Split across both GPUs — tanaka_g0 on 0, bf_g1 on 1, run in parallel.
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda uv run python solver/evals/eval_suite.py \
    --run_dir "$RUN_DIR" --checkpoint best --regimes tanaka_g0 --n_ics 32 \
    --f64_harness --batched_surrogate --output_dir "$TANAKA_OUT" \
    > "$TANAKA_OUT/eval.log" 2>&1 &
TANAKA_PID=$!
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda uv run python solver/evals/eval_suite.py \
    --run_dir "$RUN_DIR" --checkpoint best --regimes bf_g1 --n_ics 32 \
    --f64_harness --batched_surrogate --output_dir "$BF_OUT" \
    > "$BF_OUT/eval.log" 2>&1 &
BF_PID=$!
wait $TANAKA_PID $BF_PID

echo
echo "===== v11-C-revised eval DONE ====="
echo
echo "Baselines (v10 tanaka_g0 4/32 NaN med 0.01232, bf 2/32 NaN med 0.009):"
echo "Comparison at:"
echo "  outputs/cs_dno_w512b8_l256_v9_h100x2_20260703_035745/eval_2x2v3_off_20260706_070633/"
echo
grep -E "\[tanaka_g0\] done|\[bf_g1\] done" "$OUT_DIR/eval.log"
