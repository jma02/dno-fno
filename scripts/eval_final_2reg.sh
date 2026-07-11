#!/bin/bash
# Generic final-checkpoint eval on the two decisive rollout regimes.
#
# Usage:
#   scripts/eval_final_2reg.sh RUN_DIR [extra eval_suite args...]
#
# Example:
#   scripts/eval_final_2reg.sh outputs/c12_pushforward1_filtergxi_from_c2_... --filter_gxi

set -euo pipefail
cd /home/johnma/dno-fno

RUN_DIR="${1:?usage: scripts/eval_final_2reg.sh RUN_DIR [extra eval args...]}"
shift || true
TRUTH_CACHE="${TRUTH_CACHE:-outputs/cs_dno_w512b8_l256_v9_h100x2_20260703_035745/eval_2x2v3_off_20260706_070633}"
CHECKPOINT="${CHECKPOINT:-final}"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$RUN_DIR/eval_${CHECKPOINT}_2reg_f64h_batched_cached_$STAMP"
TANAKA_OUT="$OUT_DIR/tanaka_g0"
BF_OUT="$OUT_DIR/bf_g1"
mkdir -p "$TANAKA_OUT" "$BF_OUT"
printf '%s\n' "$OUT_DIR" > "$RUN_DIR/last_eval_final_2reg.txt"

COMMON_ARGS=(
    --checkpoint "$CHECKPOINT"
    --n_ics 32
    --f64_harness
    --batched_surrogate
    --truth_cache "$TRUTH_CACHE"
)

echo "===== final 2-regime eval ====="
echo "run_dir:     $RUN_DIR"
echo "checkpoint:  $CHECKPOINT"
echo "truth_cache: $TRUTH_CACHE"
echo "out_dir:     $OUT_DIR"
echo "extra_args:  $*"

CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda uv run python -m solver.evals.eval_suite \
    --run_dir "$RUN_DIR" \
    --regimes tanaka_g0 \
    --output_dir "$TANAKA_OUT" \
    "${COMMON_ARGS[@]}" "$@" \
    2>&1 | tee "$TANAKA_OUT/eval.log" &
TANAKA_PID=$!

CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda uv run python -m solver.evals.eval_suite \
    --run_dir "$RUN_DIR" \
    --regimes bf_g1 \
    --output_dir "$BF_OUT" \
    "${COMMON_ARGS[@]}" "$@" \
    2>&1 | tee "$BF_OUT/eval.log" &
BF_PID=$!

wait "$TANAKA_PID" "$BF_PID"

echo
echo "===== final 2-regime eval DONE ====="
for reg_dir in "$TANAKA_OUT" "$BF_OUT"; do
    reg=$(basename "$reg_dir")
    summary="$reg_dir/${reg}_summary.json"
    if [[ -f "$summary" ]]; then
        uv run python - "$summary" "$reg" <<'PY'
import json
import sys

summary_path, regime = sys.argv[1], sys.argv[2]
with open(summary_path, encoding="utf-8") as f:
    s = json.load(f)
print(
    f"{regime}: NaN={s['nan_rate']:.3f} "
    f"div={s['divergence_rate_final']:.3f} "
    f"eta_med_tf={s['rel_l2_eta_median_tfinal']:.6g} "
    f"eta_p95_tf={s['rel_l2_eta_p95_tfinal']:.6g}"
)
PY
    else
        echo "$reg: missing summary at $summary"
    fi
done

echo "Output: $OUT_DIR"
