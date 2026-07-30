#!/bin/bash
# Matched 32-IC Tanaka evaluation of the soliton-selective spectral growth guard.

set -euo pipefail
cd /home/johnma/dno-fno

RUN_DIR="${1:-outputs/c16_fixed_replay_ep16_to32_20260711_014635}"
CHECKPOINT="${CHECKPOINT:-best}"
TRUTH_CACHE_G0="${TRUTH_CACHE_G0:-outputs/cs_dno_w512b8_l256_v9_h100x2_20260703_035745/eval_2x2v3_off_20260706_070633}"
TRUTH_CACHE_G1="${TRUTH_CACHE_G1:-outputs/cs_dno_w512b8_l256_v9_h100x2_20260703_035745/eval_suite_f64h_n32}"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$RUN_DIR/eval_${CHECKPOINT}_soliton_spectral_guard_$STAMP"
mkdir -p "$OUT_DIR/tanaka_g0" "$OUT_DIR/tanaka_g1"

COMMON_ARGS=(
  --run_dir "$RUN_DIR"
  --checkpoint "$CHECKPOINT"
  --n_ics 32
  --f64_harness
  --batched_surrogate
  --eta_growth_guard
  --eta_growth_guard_abs_floor 5e-7
  --eta_growth_guard_growth_factor 100
  --eta_growth_guard_houli_a 0.69
  --eta_growth_guard_houli_m 4
  --eta_growth_guard_k_eff 128
  --eta_growth_guard_kh_eff 12
  --eta_growth_guard_soliton_only
  --eta_growth_guard_negative_energy_threshold 1e-3
)

CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
uv run python -m solver.evals.eval_suite \
  "${COMMON_ARGS[@]}" \
  --truth_cache "$TRUTH_CACHE_G0" \
  --regimes tanaka_g0 \
  --output_dir "$OUT_DIR/tanaka_g0" \
  2>&1 | tee "$OUT_DIR/tanaka_g0/eval.log" &
PID_G0=$!

CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda \
uv run python -m solver.evals.eval_suite \
  "${COMMON_ARGS[@]}" \
  --truth_cache "$TRUTH_CACHE_G1" \
  --regimes tanaka_g1 \
  --output_dir "$OUT_DIR/tanaka_g1" \
  2>&1 | tee "$OUT_DIR/tanaka_g1/eval.log" &
PID_G1=$!

wait "$PID_G0" "$PID_G1"

uv run python - "$OUT_DIR" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for regime in ("tanaka_g0", "tanaka_g1"):
    path = root / regime / f"{regime}_summary.json"
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    print(
        f"{regime}: NaN={summary['nan_rate']:.3f} "
        f"div={summary['divergence_rate_final']:.3f} "
        f"eta_med={summary['rel_l2_eta_median_tfinal']:.6g} "
        f"eta_p95={summary['rel_l2_eta_p95_tfinal']:.6g}"
    )
PY

printf '%s\n' "$OUT_DIR" > "$RUN_DIR/last_eval_soliton_spectral_guard.txt"
echo "Output: $OUT_DIR"
