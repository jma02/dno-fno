#!/bin/bash
# Evaluate the eight non-Tanaka families under the deployed spectral guard.

set -euo pipefail
cd /home/johnma/dno-fno

RUN_DIR="${1:?usage: scripts/eval_guarded_non_tanaka_suite_n32.sh RUN_DIR}"
CHECKPOINT="${CHECKPOINT:-final}"
TRUTH_CACHE="${TRUTH_CACHE:-outputs/cs_dno_w512b8_l256_v9_h100x2_20260703_035745/eval_suite_f64h_n32}"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$RUN_DIR/eval_${CHECKPOINT}_guarded_non_tanaka_suite_n32_$STAMP"
mkdir -p "$OUT_DIR"
printf '%s\n' "$OUT_DIR" > "$RUN_DIR/last_eval_guarded_non_tanaka_suite_n32.txt"

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

truth_cache_for_regime() {
  local regime="$1"
  if [[ -d "$TRUTH_CACHE/$regime" ]]; then
    printf '%s\n' "$TRUTH_CACHE/$regime"
  else
    printf '%s\n' "$TRUTH_CACHE"
  fi
}

run_pair() {
  local regime0="$1"
  local regime1="$2"
  local out0="$OUT_DIR/$regime0"
  local out1="$OUT_DIR/$regime1"
  mkdir -p "$out0" "$out1"

  CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
    uv run python -m solver.evals.eval_suite \
      "${COMMON_ARGS[@]}" \
      --truth_cache "$(truth_cache_for_regime "$regime0")" \
      --regimes "$regime0" --output_dir "$out0" \
      2>&1 | tee "$out0/eval.log" &
  local pid0=$!

  CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda \
    uv run python -m solver.evals.eval_suite \
      "${COMMON_ARGS[@]}" \
      --truth_cache "$(truth_cache_for_regime "$regime1")" \
      --regimes "$regime1" --output_dir "$out1" \
      2>&1 | tee "$out1/eval.log" &
  local pid1=$!

  wait "$pid0" "$pid1"
}

run_pair bf_g0 bf_g1
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
regimes = (
    "bf_g0",
    "bf_g1",
    "bf_modal",
    "random_sea_deep",
    "random_sea_finite",
    "linear",
    "stokes_deep",
    "stokes_finite",
)
for regime in regimes:
    path = root / regime / f"{regime}_summary.json"
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    summaries[regime] = summary
    print(
        f"{regime}: NaN={summary['nan_rate']:.3f} "
        f"div={summary['divergence_rate_final']:.3f} "
        f"eta_med={summary['rel_l2_eta_median_tfinal']:.6g} "
        f"eta_p95={summary['rel_l2_eta_p95_tfinal']:.6g}"
    )

with (root / "all_summaries.json").open("w", encoding="utf-8") as handle:
    json.dump(summaries, handle, indent=2)
PY

echo "Output: $OUT_DIR"
