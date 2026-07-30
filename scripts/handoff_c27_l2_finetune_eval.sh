#!/bin/bash
# Wait for the one-epoch C27 fine-tune, verify its committed result, then run
# the matched fixed-64 Tanaka evaluation and paired C25 comparison.

set -euo pipefail
cd /home/johnma/dno-fno
export PATH="/home/johnma/.local/bin:/usr/local/bin:/usr/bin:/bin"

RUN_NAME="c27_l2_finetune_from_c25_20260717_191338"
RUN_DIR="outputs/$RUN_NAME"
TRAIN_LOG="logs/$RUN_NAME.log"
HANDOFF_LOG="logs/c27_l2_finetune_eval_handoff_20260717.log"
BASELINE_EVAL="outputs/c25_capacity125_full_20260716_022603/eval_best_soliton_spectral_guard_20260717_133536"
PAIRED_OUTPUT="notes/c25_c27_l2_finetune_fixed64_paired_20260717.json"
LOCK_FILE="logs/c27_l2_finetune_eval_handoff.lock"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "$(date '+%Y-%m-%d %H:%M:%S %Z') evaluation handoff already active"
  exit 0
fi

echo "$(date '+%Y-%m-%d %H:%M:%S %Z') waiting for $RUN_NAME" >>"$HANDOFF_LOG"
while pgrep -f "$RUN_NAME" >/dev/null; do
  sleep 20
done

for checkpoint in best_val_ckpt latest_ckpt final_ckpt; do
  if [[ ! -f "$RUN_DIR/$checkpoint/metadata.json" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S %Z') no committed $checkpoint" >>"$HANDOFF_LOG"
    exit 1
  fi
done
if ! grep -Fq "C27 one-epoch L2 fine-tune guard passed" "$TRAIN_LOG"; then
  echo "$(date '+%Y-%m-%d %H:%M:%S %Z') post-training guard did not pass" >>"$HANDOFF_LOG"
  exit 1
fi

uv run python - "$RUN_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
metadata = {
    checkpoint: json.loads((run_dir / checkpoint / "metadata.json").read_text())
    for checkpoint in ("best_val_ckpt", "latest_ckpt", "final_ckpt")
}
records = [
    json.loads(line)
    for line in (run_dir / "train_log.jsonl").read_text().splitlines()
]
epochs = {checkpoint: item.get("epoch") for checkpoint, item in metadata.items()}
if set(epochs.values()) != {1} or len(records) != 1 or records[0].get("epoch") != 1:
    raise SystemExit(f"unexpected epoch metadata: {epochs}, {records}")
if any(
    not math.isfinite(value)
    for record in records
    for value in record.values()
    if isinstance(value, (int, float))
):
    raise SystemExit("nonfinite training scalar")
PY

echo "$(date '+%Y-%m-%d %H:%M:%S %Z') launching fixed-64 evaluation" >>"$HANDOFF_LOG"
CHECKPOINT=final \
TRUTH_CACHE_G0="$BASELINE_EVAL/tanaka_g0" \
TRUTH_CACHE_G1="$BASELINE_EVAL/tanaka_g1" \
  bash scripts/eval_c16_soliton_spectral_guard.sh "$RUN_DIR" \
  >>"$HANDOFF_LOG" 2>&1

CANDIDATE_EVAL="$(cat "$RUN_DIR/last_eval_soliton_spectral_guard.txt")"
uv run python - "$BASELINE_EVAL" "$CANDIDATE_EVAL" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

baseline, candidate = map(Path, sys.argv[1:])
for regime in ("tanaka_g0", "tanaka_g1"):
    log_text = (candidate / regime / "eval.log").read_text(errors="replace")
    if "truth cache HIT" not in log_text:
        raise SystemExit(f"{regime}: C25 truth cache was not reused")
    summary = json.loads(
        (candidate / regime / f"{regime}_summary.json").read_text()
    )
    if summary["n_truth_valid"] != 32 or summary["n_model_finite_truth_valid"] != 32:
        raise SystemExit(f"{regime}: invalid/nonfinite candidate cases: {summary}")
    if summary["truth_wall_s"] != 0.0:
        raise SystemExit(f"{regime}: unexpected truth recomputation")
    checkpoint = summary.get("checkpoint_source", {})
    if checkpoint.get("selection") != "final" or checkpoint.get("epoch") != 1:
        raise SystemExit(f"{regime}: unexpected checkpoint source: {checkpoint}")
    with (
        np.load(baseline / regime / f"{regime}_trajs.npz") as before,
        np.load(candidate / regime / f"{regime}_trajs.npz") as after,
    ):
        for field in (
            "times", "depths", "case_ids", "truth_eta", "truth_xi", "truth_gxi",
            "truth_valid", "truth_protocol_sha256",
        ):
            if not np.array_equal(before[field], after[field]):
                raise SystemExit(f"{regime}: paired field differs: {field}")
        for field in ("pred_eta", "pred_xi", "pred_gxi"):
            if not np.all(np.isfinite(after[field])):
                raise SystemExit(f"{regime}: nonfinite values in {field}")
PY

uv run python scripts/compare_paired_translation_errors.py \
  --baseline-eval-dir "$BASELINE_EVAL" \
  --candidate-eval-dir "$CANDIDATE_EVAL" \
  --output "$PAIRED_OUTPUT" \
  >>"$HANDOFF_LOG" 2>&1

echo "$CANDIDATE_EVAL" >"$RUN_DIR/last_fixed64_paired_eval.txt"
echo "$(date '+%Y-%m-%d %H:%M:%S %Z') evaluation and comparison complete" >>"$HANDOFF_LOG"
