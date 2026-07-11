#!/bin/bash
# Evaluate C10. If it is negative by the hard NaN-focused criterion, run C11
# and C12 sequentially and evaluate each.

set -euo pipefail
cd /home/johnma/dno-fno

C10_RUN_DIR="${1:?usage: scripts/run_c10_eval_then_fallbacks.sh outputs/c10_...}"
QUEUE_SUMMARY="$C10_RUN_DIR/fallback_queue_summary.txt"

: > "$QUEUE_SUMMARY"

summarize_eval() {
    local label="$1"
    local eval_dir="$2"

    uv run python - "$label" "$eval_dir" <<'PY' | tee -a "$QUEUE_SUMMARY"
import json
import sys
from pathlib import Path
from typing import Any

label = sys.argv[1]
eval_dir = Path(sys.argv[2])


def load(regime: str) -> dict[str, Any]:
    path = eval_dir / regime / f"{regime}_summary.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


tanaka = load("tanaka_g0")
bf = load("bf_g1")
print(
    f"{label}_EVAL eval_dir={eval_dir} "
    f"tanaka_nan={float(tanaka['nan_rate']):.6g} "
    f"tanaka_div={float(tanaka['divergence_rate_final']):.6g} "
    f"tanaka_eta_med={float(tanaka['rel_l2_eta_median_tfinal']):.6g} "
    f"tanaka_eta_p95={float(tanaka['rel_l2_eta_p95_tfinal']):.6g} "
    f"bf_nan={float(bf['nan_rate']):.6g} "
    f"bf_div={float(bf['divergence_rate_final']):.6g} "
    f"bf_eta_med={float(bf['rel_l2_eta_median_tfinal']):.6g} "
    f"bf_eta_p95={float(bf['rel_l2_eta_p95_tfinal']):.6g}"
)
PY
}

./scripts/eval_final_2reg.sh "$C10_RUN_DIR"
C10_EVAL_DIR=$(< "$C10_RUN_DIR/last_eval_final_2reg.txt")
summarize_eval "C10" "$C10_EVAL_DIR"

if uv run python - "$C10_EVAL_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

eval_dir = Path(sys.argv[1])

def load(regime: str) -> dict:
    path = eval_dir / regime / f"{regime}_summary.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

try:
    tanaka = load("tanaka_g0")
    bf = load("bf_g1")
except FileNotFoundError as exc:
    print(f"C10_NEGATIVE missing summary: {exc}")
    raise SystemExit(1)

tanaka_nan = float(tanaka["nan_rate"])
tanaka_div = float(tanaka["divergence_rate_final"])
tanaka_p95 = float(tanaka["rel_l2_eta_p95_tfinal"])
bf_nan = float(bf["nan_rate"])
bf_div = float(bf["divergence_rate_final"])
bf_med = float(bf["rel_l2_eta_median_tfinal"])

positive = (
    tanaka_nan < (3.0 / 32.0)
    and tanaka_div <= (4.0 / 32.0)
    and math.isfinite(tanaka_p95)
    and tanaka_p95 <= 0.5
    and bf_nan <= 1e-12
    and bf_div <= 1e-12
    and math.isfinite(bf_med)
    and bf_med <= 0.012
)

print(
    "C10_METRICS "
    f"tanaka_nan={tanaka_nan:.6g} tanaka_div={tanaka_div:.6g} "
    f"tanaka_p95={tanaka_p95:.6g} bf_nan={bf_nan:.6g} "
    f"bf_div={bf_div:.6g} bf_med={bf_med:.6g}"
)
if positive:
    print("C10_POSITIVE stopping fallback queue")
    raise SystemExit(0)
print("C10_NEGATIVE launching C11 and C12 fallback queue")
raise SystemExit(1)
PY
then
    echo "C10_POSITIVE stopping fallback queue" | tee -a "$QUEUE_SUMMARY"
    exit 0
fi

echo "C10_NEGATIVE launching C11 and C12 fallback queue" | tee -a "$QUEUE_SUMMARY"

./scripts/launch_c11_truth_k64_hardneg_from_c2.sh
C11_RUN_DIR=$(ls -td outputs/c11_truth_k64_hardneg_from_c2_* | head -n 1)
./scripts/eval_final_2reg.sh "$C11_RUN_DIR"
C11_EVAL_DIR=$(< "$C11_RUN_DIR/last_eval_final_2reg.txt")
summarize_eval "C11" "$C11_EVAL_DIR"

./scripts/launch_c12_pushforward1_filtergxi_from_c2.sh
C12_RUN_DIR=$(ls -td outputs/c12_pushforward1_filtergxi_from_c2_* | head -n 1)
./scripts/eval_final_2reg.sh "$C12_RUN_DIR" --filter_gxi
C12_EVAL_DIR=$(< "$C12_RUN_DIR/last_eval_final_2reg.txt")
summarize_eval "C12" "$C12_EVAL_DIR"

echo "Fallback queue complete:"
echo "  C10 eval: $C10_EVAL_DIR"
echo "  C11 run:  $C11_RUN_DIR"
echo "  C12 run:  $C12_RUN_DIR"
