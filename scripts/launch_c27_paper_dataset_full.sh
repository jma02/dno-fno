#!/bin/bash
# Fresh C27-derived all-family tangent run on the paper dataset.

set -euo pipefail
cd "$(dirname "$0")/.."

RUN_NAME="${RUN_NAME:-c27_all_family_tangent_paper_dataset_full_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="outputs/$RUN_NAME"
DATASET_PATH="/home/johnma/dno-fno/outputs/paper_dataset_literature_aligned_v1/combined/c16384_v01024_t01024/paper_dataset_all_splits_c16384.dataset.json"

if [[ -e "$RUN_DIR" ]]; then
  echo "refusing to resume or overwrite existing run: $RUN_DIR" >&2
  exit 1
fi

uv run python - "$DATASET_PATH" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

dataset_path = Path(sys.argv[1])
manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
if sum(record["n_rows"] for record in manifest["dataset_shards"]) != 7_686_144:
    raise SystemExit("paper-dataset manifest does not contain the expected rows")
print("paper-dataset structure preflight passed")
PY

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exit 0
fi

RUN_NAME="$RUN_NAME" \
DATASET="$DATASET_PATH" \
BATCH_SIZE=1024 \
LR=2e-5 \
LR_WARMUP_STEPS=500 \
HADAMARD_WEIGHT=1e-2 \
HADAMARD_INTERVAL=16 \
TRANSLATION_TANGENT_WEIGHT=10 \
MODE_BALANCED_WEIGHT=6 \
MODE_BALANCED_WARMUP_STEPS=500 \
EPOCHS=40 \
TOTAL_EPOCHS=40 \
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/launch_c27_h1_to_l2_ablation.sh

uv run python - "$RUN_DIR" "$DATASET_PATH" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

run_dir, dataset_path = map(Path, sys.argv[1:])
candidate = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

expected = {
    "model": "cs_dno",
    "width": 640,
    "n_blocks": 8,
    "latent": 320,
    "batch_size": 1024,
    "device_count": 2,
    "lr": 2e-5,
    "lr_warmup_steps": 500,
    "weight_decay": 1e-4,
    "epochs": 40,
    "total_epochs": 40,
    "translation_tangent_weight": 10.0,
    "translation_tangent_scope": "all_nonflat_rows",
    "mode_balanced_weight": 6.0,
    "mode_balanced_warmup_steps": 500,
    "hadamard_weight": 1e-2,
    "hadamard_interval": 16,
    "param_count": 1_342_400,
    "dataset": str(dataset_path),
    "train_examples": 6_832_128,
    "val_examples": 427_008,
    "eta_scale": 0.15583430230617523,
    "xi_scale": 0.09394174814224243,
    "target_scale": 0.12198150902986526,
}
mismatches = {
    key: (candidate.get(key), value)
    for key, value in expected.items()
    if candidate.get(key) != value
}
if mismatches:
    raise SystemExit(f"new-dataset handoff guard failed: {mismatches}")

records = [
    json.loads(line)
    for line in (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
]
if len(records) != 40 or int(records[-1]["epoch"]) != 40:
    raise SystemExit(f"expected 40 complete epochs, got {len(records)}")
nonfinite = {
    f"epoch_{record['epoch']}.{key}": value
    for record in records
    for key, value in record.items()
    if isinstance(value, (int, float)) and not math.isfinite(value)
}
if nonfinite:
    raise SystemExit(f"nonfinite training scalars: {nonfinite}")
print("all-family tangent/new-dataset guard passed: recipe and dataset match")
PY
