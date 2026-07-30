#!/bin/bash
# Evaluate the C28 epoch-40 final checkpoint after its exact mode-off retrain.

set -euo pipefail
cd /home/johnma/dno-fno
export PATH="/home/johnma/.local/bin:/usr/local/bin:/usr/bin:/bin"

RUN_NAME="${RUN_NAME:-c28_l2_mode_off_full_20260720_050017}"
RUN_DIR="outputs/$RUN_NAME"
C27_RUN="outputs/c27_h1_to_l2_full_20260717_212550"
C27_TANAKA="$C27_RUN/eval_final_soliton_spectral_guard_20260719_191228"
C27_NON_TANAKA="$C27_RUN/eval_final_guarded_non_tanaka_suite_n32_20260719_221813"
LOCK_FILE="logs/${RUN_NAME}_eval_handoff.lock"
ACCEPTANCE_OUTPUT="$RUN_DIR/c27_final_n32_mode_ablation_acceptance.json"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "$(date '+%Y-%m-%d %H:%M:%S %Z') C28 evaluation handoff already active"
  exit 0
fi

if [[ -f "$ACCEPTANCE_OUTPUT" ]]; then
  echo "$(date '+%Y-%m-%d %H:%M:%S %Z') C28 evaluation already complete"
  exit 0
fi

training_active() {
  pgrep -f \
    "train-jax-10m/1d_dno_fno_jax.py.*--run_name[[:space:]]+$RUN_NAME([[:space:]]|$)" \
    >/dev/null
}

launcher_active() {
  pgrep -f "[b]ash scripts/launch_c28_l2_mode_off_full.sh" >/dev/null
}

if [[ ! -d "$RUN_DIR" ]] && ! training_active && ! launcher_active; then
  echo "C28 has not been launched: $RUN_DIR" >&2
  exit 1
fi

echo "$(date '+%Y-%m-%d %H:%M:%S %Z') waiting for $RUN_NAME"
while training_active || launcher_active; do
  sleep 30
done

uv run python - "$C27_RUN" "$RUN_DIR" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


parent_dir, candidate_dir = map(Path, sys.argv[1:])
parent = json.loads((parent_dir / "config.json").read_text(encoding="utf-8"))
candidate = json.loads(
    (candidate_dir / "config.json").read_text(encoding="utf-8")
)
differences = {
    key: (parent.get(key), candidate.get(key))
    for key in sorted(set(parent) | set(candidate))
    if parent.get(key) != candidate.get(key)
}
if differences != {"mode_balanced_weight": (6.0, 0.0)}:
    raise SystemExit(f"unexpected C27/C28 config differences: {differences}")

records = [
    json.loads(line)
    for line in (candidate_dir / "train_log.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
]
if [record.get("epoch") for record in records] != list(range(1, 41)):
    raise SystemExit("training log is not a complete ordered 40-epoch run")
nonfinite = {
    f"epoch_{record['epoch']}.{key}": value
    for record in records
    for key, value in record.items()
    if isinstance(value, (int, float)) and not math.isfinite(value)
}
if nonfinite:
    raise SystemExit(f"training log contains nonfinite scalars: {nonfinite}")
for record in records:
    if record.get("hadamard_active_batches") != record.get(
        "hadamard_expected_batches"
    ):
        raise SystemExit(
            f"epoch {record['epoch']}: Hadamard batch accounting mismatch"
        )
    if record.get("mode_balanced_weight_eff") not in (None, 0.0):
        raise SystemExit(
            f"epoch {record['epoch']}: mode loss acquired nonzero weight"
        )
    if record.get("mode_balanced_extra") not in (None, 0.0):
        raise SystemExit(
            f"epoch {record['epoch']}: mode loss contributed to the objective"
        )

summary = json.loads(
    (candidate_dir / "summary.json").read_text(encoding="utf-8")
)
if int(summary.get("epochs_completed", -1)) != 40:
    raise SystemExit("summary does not report 40 completed epochs")

for selection in ("best_val_ckpt", "latest_ckpt", "final_ckpt"):
    checkpoint_dir = candidate_dir / selection
    metadata = json.loads(
        (checkpoint_dir / "metadata.json").read_text(encoding="utf-8")
    )
    epoch = int(metadata["epoch"])
    if selection in ("latest_ckpt", "final_ckpt") and epoch != 40:
        raise SystemExit(f"{selection}: expected epoch 40, got {epoch}")
    if not 1 <= epoch <= 40:
        raise SystemExit(f"{selection}: invalid epoch {epoch}")
    payload = checkpoint_dir / f"ckpt_{epoch}" / "_CHECKPOINT_METADATA"
    if not payload.is_file():
        raise SystemExit(f"{selection}: missing committed epoch-{epoch} payload")

print("C28 training guard passed: exact finite epoch-40 mode-only deletion")
PY

echo "$(date '+%Y-%m-%d %H:%M:%S %Z') launching cached final Tanaka-64"
CHECKPOINT=final \
TRUTH_CACHE_G0="$C27_TANAKA/tanaka_g0" \
TRUTH_CACHE_G1="$C27_TANAKA/tanaka_g1" \
  bash scripts/eval_c16_soliton_spectral_guard.sh "$RUN_DIR"
C28_TANAKA="$(<"$RUN_DIR/last_eval_soliton_spectral_guard.txt")"

uv run python scripts/compare_paired_translation_errors.py \
  --baseline-eval-dir "$C27_TANAKA" \
  --candidate-eval-dir "$C28_TANAKA" \
  --output "$RUN_DIR/c27_final_tanaka_paired_translation.json"

echo "$(date '+%Y-%m-%d %H:%M:%S %Z') launching cached final eight-family suite"
CHECKPOINT=final \
TRUTH_CACHE="$C27_NON_TANAKA" \
  bash scripts/eval_guarded_non_tanaka_suite_n32.sh "$RUN_DIR"
C28_NON_TANAKA="$(<"$RUN_DIR/last_eval_guarded_non_tanaka_suite_n32.txt")"

uv run python scripts/compare_c27_c28_mode_ablation.py \
  --baseline-tanaka-dir "$C27_TANAKA" \
  --baseline-non-tanaka-dir "$C27_NON_TANAKA" \
  --candidate-tanaka-dir "$C28_TANAKA" \
  --candidate-non-tanaka-dir "$C28_NON_TANAKA" \
  --candidate-run-dir "$RUN_DIR" \
  --output "$ACCEPTANCE_OUTPUT"

printf '%s\n' "$C28_TANAKA" >"$RUN_DIR/last_final_tanaka64_eval.txt"
printf '%s\n' "$C28_NON_TANAKA" >"$RUN_DIR/last_final_non_tanaka_n32_eval.txt"
echo "$(date '+%Y-%m-%d %H:%M:%S %Z') C28 final evaluation complete"
echo "Acceptance summary: $ACCEPTANCE_OUTPUT"
