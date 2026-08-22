#!/bin/bash
# Fresh C27 replica on the authenticated four-family paper corpus.

set -euo pipefail
cd "$(dirname "$0")/.."

RUN_NAME="${RUN_NAME:-c27_paper_corpus_full_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="outputs/$RUN_NAME"
DATASET_PATH="/home/johnma/dno-fno/outputs/paper_corpus_literature_aligned_v1/combined/c16384_v01024_t01024/paper_corpus_all_splits_c16384.dataset.json"
HANDOFF_PATH="outputs/paper_corpus_final_postcompletion/32f764cf865c90892ee0329e48fe992cbc24df2707e05fd1af2ee9cd2b84307c/training_handoff_audit.json"

if [[ -e "$RUN_DIR" ]]; then
  echo "refusing to resume or overwrite existing run: $RUN_DIR" >&2
  exit 1
fi

uv run python - "$DATASET_PATH" "$HANDOFF_PATH" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from solver.gen_data.pipeline.quota_driver import canonical_json_sha256

dataset_path, handoff_path = map(Path, sys.argv[1:])
handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
normalization = handoff["training_normalization"]
stats_path = dataset_path.with_suffix(".stats.json")
stats = json.loads(stats_path.read_text(encoding="utf-8"))

expected = {
    "status": "complete",
    "combined_summary_sha256": (
        "32f764cf865c90892ee0329e48fe992cbc24df2707e05fd1af2ee9cd2b84307c"
    ),
    "manifest_sha256": (
        "dbb76ed1cba4d0146f3c9643a73d168d3a9fa97bc27c37fe0fd7ad08554bd734"
    ),
    "stats_fingerprint": (
        "4e8ec96947bc05cee3b375467499b0ae00a7814f9982614084efd859efe7c3b8"
    ),
    "train_rows": 6_832_128,
    "validation_rows": 427_008,
}
rows_by_split = handoff["final_contract"]["observed"][
    "accepted_rows_by_split_and_family"
]
observed = {
    "status": handoff["status"],
    "combined_summary_sha256": handoff["combined_summary"]["sha256"],
    "manifest_sha256": handoff["dataset_view"]["manifest"]["sha256"],
    "stats_fingerprint": canonical_json_sha256(stats),
    "train_rows": sum(rows_by_split["train"].values()),
    "validation_rows": sum(rows_by_split["validation"].values()),
}
if observed != expected:
    raise SystemExit(
        f"paper-corpus/C27 preflight mismatch: expected={expected}, observed={observed}"
    )
if stats != normalization["stats"]:
    raise SystemExit("trainer statistics cache differs from authenticated handoff")
print("paper-corpus/C27 preflight passed")
PY

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exit 0
fi

RUN_NAME="$RUN_NAME" \
DATASET="$DATASET_PATH" \
DATA_FRACTION=1.0 \
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
  bash scripts/launch_c27_h1_to_l2_ablation.sh full

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
    raise SystemExit(f"new-corpus handoff guard failed: {mismatches}")

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
print("fresh C27/new-corpus guard passed: frozen recipe and corpus are authenticated")
PY
