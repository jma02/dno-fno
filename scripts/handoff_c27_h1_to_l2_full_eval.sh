#!/bin/bash
# After the fresh 40-epoch C27 L2 ablation completes successfully, evaluate
# both validation-best and final checkpoints on the matched fixed Tanaka panel.

set -euo pipefail
cd /home/johnma/dno-fno
export PATH="/home/johnma/.local/bin:/usr/local/bin:/usr/bin:/bin"

RUN_NAME="c27_h1_to_l2_full_20260717_212550"
RUN_DIR="outputs/$RUN_NAME"
TRAIN_LOG="logs/$RUN_NAME.log"
HANDOFF_LOG="logs/c27_h1_to_l2_full_eval_handoff_20260717.log"
BASELINE_EVAL="outputs/c25_capacity125_full_20260716_022603/eval_best_soliton_spectral_guard_20260717_133536"
LOCK_FILE="logs/c27_h1_to_l2_full_eval_handoff.lock"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "$(date '+%Y-%m-%d %H:%M:%S %Z') evaluation handoff already active"
  exit 0
fi

echo "$(date '+%Y-%m-%d %H:%M:%S %Z') waiting for $RUN_NAME" >>"$HANDOFF_LOG"
while pgrep -f "$RUN_NAME" >/dev/null; do
  sleep 30
done

if ! grep -Fq \
  "C27 guard passed: C25 objective with only H1 changed to L2" \
  "$TRAIN_LOG"; then
  echo "$(date '+%Y-%m-%d %H:%M:%S %Z') training guard did not pass" \
    >>"$HANDOFF_LOG"
  exit 1
fi

uv run python - "$RUN_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
records = [
    json.loads(line)
    for line in (run_dir / "train_log.jsonl").read_text().splitlines()
]
if [record.get("epoch") for record in records] != list(range(1, 41)):
    raise SystemExit("training log is not a complete ordered 40-epoch run")
if any(
    not math.isfinite(value)
    for record in records
    for value in record.values()
    if isinstance(value, (int, float))
):
    raise SystemExit("training log contains a nonfinite scalar")
for record in records:
    if record.get("hadamard_active_batches") != record.get(
        "hadamard_expected_batches"
    ):
        raise SystemExit(
            f"epoch {record['epoch']}: Hadamard accounting mismatch"
        )

for selection in ("best_val_ckpt", "latest_ckpt", "final_ckpt"):
    checkpoint_dir = run_dir / selection
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text())
    epoch = int(metadata["epoch"])
    if selection in ("latest_ckpt", "final_ckpt") and epoch != 40:
        raise SystemExit(f"{selection}: expected epoch 40, got {epoch}")
    payload = checkpoint_dir / f"ckpt_{epoch}" / "_CHECKPOINT_METADATA"
    if not payload.is_file():
        raise SystemExit(f"{selection}: missing committed epoch-{epoch} payload")
PY

validate_eval() {
  local selection="$1"
  local candidate="$2"
  uv run python - "$BASELINE_EVAL" "$candidate" "$selection" "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

baseline, candidate = map(Path, sys.argv[1:3])
selection = sys.argv[3]
run_dir = Path(sys.argv[4])
checkpoint_dir = {
    "best": run_dir / "best_val_ckpt",
    "final": run_dir / "final_ckpt",
}[selection]
expected_epoch = json.loads(
    (checkpoint_dir / "metadata.json").read_text()
)["epoch"]
for regime in ("tanaka_g0", "tanaka_g1"):
    log_text = (candidate / regime / "eval.log").read_text(errors="replace")
    if "truth cache HIT" not in log_text:
        raise SystemExit(f"{selection}/{regime}: C25 truth cache was not reused")
    summary = json.loads(
        (candidate / regime / f"{regime}_summary.json").read_text()
    )
    if summary["n_truth_valid"] != 32:
        raise SystemExit(f"{selection}/{regime}: invalid truth panel")
    if summary["n_model_finite_truth_valid"] != 32:
        raise SystemExit(f"{selection}/{regime}: nonfinite model rollout")
    if summary["truth_wall_s"] != 0.0:
        raise SystemExit(f"{selection}/{regime}: unexpected truth recomputation")
    checkpoint = summary.get("checkpoint_source", {})
    if (
        checkpoint.get("selection") != selection
        or checkpoint.get("epoch") != expected_epoch
    ):
        raise SystemExit(
            f"{selection}/{regime}: unexpected checkpoint source {checkpoint}"
        )
    with (
        np.load(baseline / regime / f"{regime}_trajs.npz") as before,
        np.load(candidate / regime / f"{regime}_trajs.npz") as after,
    ):
        for field in (
            "times", "depths", "case_ids", "truth_eta", "truth_xi",
            "truth_gxi", "truth_valid", "truth_protocol_sha256",
        ):
            if not np.array_equal(before[field], after[field]):
                raise SystemExit(
                    f"{selection}/{regime}: paired field differs: {field}"
                )
        for field in ("pred_eta", "pred_xi", "pred_gxi"):
            if not np.all(np.isfinite(after[field])):
                raise SystemExit(
                    f"{selection}/{regime}: nonfinite values in {field}"
                )
PY
}

for selection in best final; do
  echo "$(date '+%Y-%m-%d %H:%M:%S %Z') launching $selection fixed-64 evaluation" \
    >>"$HANDOFF_LOG"
  CHECKPOINT="$selection" \
  TRUTH_CACHE_G0="$BASELINE_EVAL/tanaka_g0" \
  TRUTH_CACHE_G1="$BASELINE_EVAL/tanaka_g1" \
    bash scripts/eval_c16_soliton_spectral_guard.sh "$RUN_DIR" \
    >>"$HANDOFF_LOG" 2>&1

  candidate_eval="$(cat "$RUN_DIR/last_eval_soliton_spectral_guard.txt")"
  validate_eval "$selection" "$candidate_eval"
  uv run python scripts/compare_paired_translation_errors.py \
    --baseline-eval-dir "$BASELINE_EVAL" \
    --candidate-eval-dir "$candidate_eval" \
    --output "notes/c25_c27_l2_full_${selection}_fixed64_paired_20260717.json" \
    >>"$HANDOFF_LOG" 2>&1
  echo "$candidate_eval" >"$RUN_DIR/last_fixed64_${selection}_eval.txt"
done

echo "$(date '+%Y-%m-%d %H:%M:%S %Z') best/final evaluation complete" \
  >>"$HANDOFF_LOG"
