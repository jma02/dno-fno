#!/bin/bash
# Third 2x2 relaunch — after switching containment from fleet-wide halving to
# per-sample masking-only (2026-07-06 07:xx). Prior two launches (05:11, 06:13)
# both showed baseline-on arm = 100% NaN because a single Mode 1 sample drove
# fleet-wide dt halving to exhaustion, then the scalar `failed` flag NaN'd all
# 32 samples. New code (`_safe_gl2_substep`) masks only the offending samples.
#
# Arms (all n_ics=32, f64_harness, batched_surrogate):
#   1. baseline v10 × patch off   (sanity: should reproduce ~6 NaN prior)
#   2. baseline v10 × patch on    (with fix: expect ~6 isolated per-sample NaNs)
#   3. C1 × patch off             (sanity: should reproduce ~3 NaN prior)
#   4. C1 × patch on              (with fix: expect ~3 isolated per-sample NaNs)
#
# All arms → GPU 0 (GPU 1 is currently in use by another user's benchmark).

set -euo pipefail
cd /home/johnma/dno-fno

STAMP=20260706_070633
BASELINE=outputs/cs_dno_w512b8_l256_v9_h100x2_20260703_035745
C1=outputs/c1_stage_match_from_v85b_20260704_210952

COMMON_ARGS=(
    --gpu
    --regimes tanaka_g0 bf_g1
    --n_ics 32
    --f64_harness
    --batched_surrogate
    --checkpoint best
)

# Arm 1: baseline v10 patch OFF
CUDA_VISIBLE_DEVICES=0 uv run python -m solver.evals.eval_suite \
    --run_dir "$BASELINE" \
    "${COMMON_ARGS[@]}" \
    --output_dir "$BASELINE/eval_2x2v3_off_$STAMP" \
    2>&1 | tee /tmp/2x2v3_baseline_off.log

# Arm 2: baseline v10 patch ON
CUDA_VISIBLE_DEVICES=0 uv run python -m solver.evals.eval_suite \
    --run_dir "$BASELINE" \
    "${COMMON_ARGS[@]}" \
    --gl2_residual_check \
    --output_dir "$BASELINE/eval_2x2v3_on_$STAMP" \
    2>&1 | tee /tmp/2x2v3_baseline_on.log

# Arm 3: C1 patch OFF
CUDA_VISIBLE_DEVICES=0 uv run python -m solver.evals.eval_suite \
    --run_dir "$C1" \
    "${COMMON_ARGS[@]}" \
    --output_dir "$C1/eval_2x2v3_off_$STAMP" \
    2>&1 | tee /tmp/2x2v3_c1_off.log

# Arm 4: C1 patch ON
CUDA_VISIBLE_DEVICES=0 uv run python -m solver.evals.eval_suite \
    --run_dir "$C1" \
    "${COMMON_ARGS[@]}" \
    --gl2_residual_check \
    --output_dir "$C1/eval_2x2v3_on_$STAMP" \
    2>&1 | tee /tmp/2x2v3_c1_on.log

echo ""
echo "===== 2x2 relaunch DONE ====="
for arm in baseline_off baseline_on c1_off c1_on; do
    case $arm in
        baseline_off) DIR="$BASELINE/eval_2x2v3_off_$STAMP" ;;
        baseline_on) DIR="$BASELINE/eval_2x2v3_on_$STAMP" ;;
        c1_off) DIR="$C1/eval_2x2v3_off_$STAMP" ;;
        c1_on) DIR="$C1/eval_2x2v3_on_$STAMP" ;;
    esac
    echo ""
    echo "$arm -> $DIR"
    for reg in tanaka_g0 bf_g1; do
        SUMM="$DIR/${reg}_summary.json"
        if [ -f "$SUMM" ]; then
            python -c "
import json, math
s = json.load(open('$SUMM'))
nr = s['nan_rate']
dr = s['divergence_rate_final']
m50 = s.get('rel_l2_eta_median_t50p', float('nan'))
mtf = s.get('rel_l2_eta_median_tfinal', float('nan'))
print(f'  {\"$reg\":10s}  NaN={nr:.2f}  div={dr:.2f}  eta_med_t50p={m50}  eta_med_tf={mtf}')
"
        else
            echo "  $reg: MISSING SUMMARY"
        fi
    done
done
