#!/bin/bash
# Fresh C27-derived all-family tangent run on the paper dataset.

set -euo pipefail
cd "$(dirname "$0")/.."

RUN_NAME="${RUN_NAME:-c27_all_family_tangent_paper_dataset_full_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="outputs/$RUN_NAME"
DATASET="${DATASET:-outputs/paper_dataset/arrays}"

if [[ -e "$RUN_DIR" ]]; then
  echo "refusing to resume or overwrite existing run: $RUN_DIR" >&2
  exit 1
fi

JAX_PLATFORMS=cpu uv run python - "$DATASET" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("train-jax-10m").resolve()))
from util import build_dataset_split_indices, get_batches, load_dataset_arrays, load_or_compute_stats, make_normalizers

dataset_path = Path(sys.argv[1]).expanduser().resolve()
dataset = load_dataset_arrays(dataset_path)
train, validation, test = build_dataset_split_indices(dataset)
if not train.size or not validation.size:
    raise SystemExit("training requires nonempty train and validation splits")
stats = load_or_compute_stats(dataset_path, dataset, indices=train)
norm_inputs, norm_targets, _ = make_normalizers(stats, mode="scale")
eta, xi, gxi, depth, _ = next(get_batches(
    dataset["eta"], dataset["xi"], dataset["gxi"], dataset["depth"], train,
    batch_size=1024, rng=None,
))
if not all(np.isfinite(array).all() for array in (norm_inputs(eta, xi), norm_targets(gxi), depth)):
    raise SystemExit("dataset has a nonfinite normalized example batch")
print(f"dataset preflight passed: train={train.size}, validation={validation.size}, test={test.size} rows")
PY

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exit 0
fi

RUN_NAME="$RUN_NAME" \
DATASET="$DATASET" \
  bash scripts/launch_c27_h1_to_l2_ablation.sh
