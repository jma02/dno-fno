#!/bin/bash
# Eval C2 final with temporal eta high-band growth guard.

set -euo pipefail
cd /home/johnma/dno-fno

RUN_DIR="${1:-outputs/c2_stage_match_gain_from_v85b_20260707_053707}"
TRUTH_CACHE="${2:-outputs/cs_dno_w512b8_l256_v9_h100x2_20260703_035745/eval_2x2v3_off_20260706_070633}"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$RUN_DIR/eval_eta_growth_guard_$STAMP"
TANAKA_OUT="$OUT_DIR/tanaka_g0"
BF_OUT="$OUT_DIR/bf_g1"
mkdir -p "$TANAKA_OUT" "$BF_OUT"

COMMON_ARGS=(
    --checkpoint final
    --n_ics 32
    --f64_harness
    --batched_surrogate
    --truth_cache "$TRUTH_CACHE"
    --eta_growth_guard
    --eta_growth_guard_k_lo 64
    --eta_growth_guard_k_hi 128
    --eta_growth_guard_abs_floor 1e-4
    --eta_growth_guard_growth_factor 100
    --eta_growth_guard_k_eff 64
)

echo "===== C2 eta-growth-guard eval ====="
echo "run_dir:     $RUN_DIR"
echo "truth_cache: $TRUTH_CACHE"
echo "out_dir:     $OUT_DIR"

CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda uv run python -m solver.evals.eval_suite \
    --run_dir "$RUN_DIR" \
    --regimes tanaka_g0 \
    --output_dir "$TANAKA_OUT" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$TANAKA_OUT/eval.log" &
TANAKA_PID=$!

CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda uv run python -m solver.evals.eval_suite \
    --run_dir "$RUN_DIR" \
    --regimes bf_g1 \
    --output_dir "$BF_OUT" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$BF_OUT/eval.log" &
BF_PID=$!

wait "$TANAKA_PID" "$BF_PID"

echo
echo "===== C2 eta-growth-guard eval DONE ====="
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
