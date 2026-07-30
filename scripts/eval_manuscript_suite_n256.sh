#!/bin/bash
# Resumable two-GPU manuscript evaluation on ten held-out IC panels.

set -euo pipefail
cd /home/johnma/dno-fno

RUN_DIR="${1:?usage: scripts/eval_manuscript_suite_n256.sh RUN_DIR}"
CHECKPOINT="${CHECKPOINT:-final}"
N_ICS="${N_ICS:-256}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/eval_${CHECKPOINT}_manuscript_suite_n${N_ICS}_$STAMP}"
IC_PANEL_DIR="${IC_PANEL_DIR:?set IC_PANEL_DIR to the held-out panel directory}"
GUARD_MODE="${GUARD_MODE:-off}"

if [[ ! -s "$IC_PANEL_DIR/manifest.json" ]]; then
  printf 'held-out panel manifest is missing: %s/manifest.json\n' "$IC_PANEL_DIR" >&2
  exit 1
fi

CHECKPOINT_SUBDIR="final_ckpt"
if [[ "$CHECKPOINT" == "best" ]]; then
  CHECKPOINT_SUBDIR="best_val_ckpt"
fi
CHECKPOINT_DIR="$RUN_DIR/$CHECKPOINT_SUBDIR"
CHECKPOINT_SHA256="${CHECKPOINT_SHA256:-$(
  JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/matplotlib uv run python -c \
    'import sys; from pathlib import Path; from solver.evals.eval_suite import _directory_sha256; print(_directory_sha256(Path(sys.argv[1])))' \
    "$CHECKPOINT_DIR"
)}"

mkdir -p "$OUT_DIR"
printf '%s\n' "$OUT_DIR" > "$RUN_DIR/last_eval_manuscript_suite_n${N_ICS}.in_progress.txt"

COMMON_ARGS=(
  --run_dir "$RUN_DIR"
  --checkpoint "$CHECKPOINT"
  --n_ics "$N_ICS"
  --rollout_batch_size "$ROLLOUT_BATCH_SIZE"
  --ic_panel_dir "$IC_PANEL_DIR"
  --gpu
  --f64_harness
  --batched_surrogate
)

if [[ "$GUARD_MODE" == "tanaka_soliton" ]]; then
  COMMON_ARGS+=(
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
elif [[ "$GUARD_MODE" != "off" ]]; then
  printf 'unknown GUARD_MODE=%s (expected off or tanaka_soliton)\n' "$GUARD_MODE" >&2
  exit 1
fi

is_complete() {
  local regime="$1"
  local out="$OUT_DIR/$regime"
  local summary="$out/${regime}_summary.json"
  local archive="$out/${regime}_trajs.npz"
  local panel_sha
  local guard_enabled=false
  if [[ "$GUARD_MODE" == "tanaka_soliton" ]]; then
    guard_enabled=true
  fi
  panel_sha="$(sha256sum "$IC_PANEL_DIR/${regime}_ics.npz" | awk '{print $1}')"
  [[ -s "$summary" && -s "$archive" ]] &&
    jq -e --arg regime "$regime" --argjson n "$N_ICS" \
      --argjson rollout_batch "$ROLLOUT_BATCH_SIZE" --arg panel_sha "$panel_sha" \
      --arg checkpoint "$CHECKPOINT" --argjson guard "$guard_enabled" \
      --arg checkpoint_sha "$CHECKPOINT_SHA256" \
      '.regime == $regime and .n_ics_attempted == $n and
       .ic_panel_source.sha256 == $panel_sha and
       .checkpoint_source.selection == $checkpoint and
       .checkpoint_source.sha256 == $checkpoint_sha and
       .f64_harness == true and .batched_surrogate == true and
       .rollout_batch_size == $rollout_batch and
       .eta_growth_guard_enabled == $guard and
       ($guard == false or
         (.eta_growth_guard_abs_floor == 5e-7 and
          .eta_growth_guard_growth_factor == 100 and
          .eta_growth_guard_houli_a == 0.69 and
          .eta_growth_guard_houli_m == 4 and
          .eta_growth_guard_k_eff == 128 and
          .eta_growth_guard_kh_eff == 12 and
          .eta_growth_guard_soliton_only == true and
          .eta_growth_guard_negative_energy_threshold == 1e-3))' \
      "$summary" >/dev/null
}

run_one() {
  local gpu="$1"
  local regime="$2"
  local out="$OUT_DIR/$regime"
  mkdir -p "$out"

  if is_complete "$regime"; then
    printf '[%s] resume: complete archive already present; skipping\n' "$regime"
    return 0
  fi

  printf '[%s] launching on physical GPU %s at %s\n' \
    "$regime" "$gpu" "$(date '+%Y-%m-%d %H:%M:%S %Z')"
  CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run python -m solver.evals.eval_suite \
      "${COMMON_ARGS[@]}" \
      --truth_cache "$out" \
      --regimes "$regime" \
      --output_dir "$out" \
      2>&1 | tee "$out/eval.log"
}

wait_pair() {
  local pid0="$1"
  local pid1="$2"
  local status0=0
  local status1=0
  wait "$pid0" || status0=$?
  wait "$pid1" || status1=$?
  if (( status0 != 0 || status1 != 0 )); then
    printf 'parallel wave failed: status0=%d status1=%d\n' "$status0" "$status1" >&2
    return 1
  fi
}

run_pair() {
  run_one 0 "$1" &
  local pid0=$!
  run_one 1 "$2" &
  local pid1=$!
  wait_pair "$pid0" "$pid1"
}

# Start with short families: this is the full requested n=256 evaluation and
# also validates batch-256 memory before entering the long-horizon waves.
run_pair linear stokes_deep

# Five long-horizon families.  The final odd family is paired with the three
# remaining short families on the other GPU to avoid serial idle time.
run_pair tanaka_g0 tanaka_g1
run_pair bf_g0 bf_g1

run_one 0 bf_modal &
pid0=$!
(
  run_one 1 random_sea_deep
  run_one 1 random_sea_finite
  run_one 1 stokes_finite
) &
pid1=$!
wait_pair "$pid0" "$pid1"

JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/matplotlib uv run python - \
  "$OUT_DIR" "$N_ICS" "$IC_PANEL_DIR" "$CHECKPOINT" "$GUARD_MODE" \
  "$ROLLOUT_BATCH_SIZE" \
  "$CHECKPOINT_SHA256" <<'PY'
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

from solver.evals.eval_suite import _write_json, compute_macro_summary

root = Path(sys.argv[1])
expected_n = int(sys.argv[2])
panel_root = Path(sys.argv[3])
expected_checkpoint = sys.argv[4]
expected_guard = sys.argv[5] == "tanaka_soliton"
expected_rollout_batch = int(sys.argv[6])
expected_checkpoint_sha = sys.argv[7]
regimes = (
    "tanaka_g0",
    "tanaka_g1",
    "bf_g0",
    "bf_g1",
    "bf_modal",
    "linear",
    "random_sea_deep",
    "random_sea_finite",
    "stokes_deep",
    "stokes_finite",
)
summaries: dict[str, dict[str, object]] = {}
for regime in regimes:
    path = root / regime / f"{regime}_summary.json"
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    panel_sha = hashlib.sha256((panel_root / f"{regime}_ics.npz").read_bytes()).hexdigest()
    if (
        summary.get("regime") != regime
        or summary.get("n_ics_attempted") != expected_n
        or summary.get("ic_panel_source", {}).get("sha256") != panel_sha
        or summary.get("checkpoint_source", {}).get("selection") != expected_checkpoint
        or summary.get("checkpoint_source", {}).get("sha256") != expected_checkpoint_sha
        or summary.get("f64_harness") is not True
        or summary.get("batched_surrogate") is not True
        or summary.get("rollout_batch_size") != expected_rollout_batch
        or summary.get("eta_growth_guard_enabled") is not expected_guard
    ):
        raise SystemExit(f"invalid summary identity: {path}")
    summaries[regime] = summary
    print(
        f"{regime}: valid={summary['n_truth_valid']}/{summary['n_ics_attempted']} "
        f"nonfinite={summary['model_nonfinite_any_count_truth_valid']} "
        f"fail@1={summary['terminal_eta_failure_count_tau_1']} "
        f"eta_med={summary['rel_l2_eta_median_conditional_finite_tfinal']} "
        f"eta_p95={summary['rel_l2_eta_p95_conditional_finite_tfinal']}"
    )

_write_json(root / "all_summaries.json", summaries)
_write_json(root / "macro_summary.json", compute_macro_summary(summaries))
PY

printf '%s\n' "$OUT_DIR" > "$RUN_DIR/last_eval_manuscript_suite_n${N_ICS}.txt"
date '+%Y-%m-%d %H:%M:%S %Z manuscript n=256 suite complete'
echo "Output: $OUT_DIR"
