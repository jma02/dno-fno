#!/bin/bash
# Exact C27 deletion ablation: remove only the mode-balanced loss.

set -euo pipefail
cd /home/johnma/dno-fno

RUN_NAME="${RUN_NAME:-c28_l2_mode_off_full_20260720_050017}"
PARENT_RUN="outputs/c27_h1_to_l2_full_20260717_212550"
RUN_DIR="outputs/$RUN_NAME"

if [[ -e "$RUN_DIR" ]]; then
  echo "refusing to resume or overwrite existing run: $RUN_DIR" >&2
  exit 1
fi

RUN_NAME="$RUN_NAME" \
DATASET=combined_dataset_v9.npz \
DATA_FRACTION=1.0 \
BATCH_SIZE=1024 \
LR=2e-5 \
LR_WARMUP_STEPS=500 \
HADAMARD_WEIGHT=1e-2 \
HADAMARD_INTERVAL=16 \
TRANSLATION_TANGENT_WEIGHT=10 \
MODE_BALANCED_WEIGHT=0 \
MODE_BALANCED_WARMUP_STEPS=500 \
EPOCHS=40 \
TOTAL_EPOCHS=40 \
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/launch_c27_h1_to_l2_ablation.sh full

uv run python - "$PARENT_RUN" "$RUN_DIR" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

parent_dir, candidate_dir = map(Path, sys.argv[1:])
parent = json.loads((parent_dir / "config.json").read_text())
candidate = json.loads((candidate_dir / "config.json").read_text())

expected_difference = {"mode_balanced_weight": (6.0, 0.0)}
keys = set(parent) | set(candidate)
differences = {
    key: (parent.get(key), candidate.get(key))
    for key in sorted(keys)
    if parent.get(key) != candidate.get(key)
}
if differences != expected_difference:
    raise SystemExit(f"C28 config differs from C27 unexpectedly: {differences}")

print("C28 guard passed: fresh C27 replica with only mode weight 6 -> 0")
PY
