#!/bin/bash
# C27 full-dataset run with translation loss on Tanaka rows only.

set -euo pipefail
cd /home/johnma/dno-fno

DATASET="${DATASET:-outputs/paper_dataset/arrays}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LR="${LR:-2e-5}"
HADAMARD_WEIGHT="${HADAMARD_WEIGHT:-1e-2}"
HADAMARD_INTERVAL="${HADAMARD_INTERVAL:-16}"
TRANSLATION_TANGENT_WEIGHT="${TRANSLATION_TANGENT_WEIGHT:-10}"
MODE_BALANCED_WEIGHT="${MODE_BALANCED_WEIGHT:-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}"
MODE_BALANCED_WARMUP_STEPS="${MODE_BALANCED_WARMUP_STEPS:-500}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
EPOCHS="${EPOCHS:-40}"
RUN_PREFIX="c27_tanaka_tangent_full"

RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)}"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python train-jax-10m/1d_dno_fno_jax.py \
  --model cs_dno \
  --norm scale \
  --dataset "$DATASET" \
  --width 640 \
  --n_blocks 8 \
  --latent 320 \
  --cs_n_polys 3 \
  --cs_mult_hidden 160 \
  --translation_tangent_weight "$TRANSLATION_TANGENT_WEIGHT" \
  --translation_tangent_smoothing_scale 1 \
  --translation_tangent_denominator_eps 1e-3 \
  --mode_balanced_weight "$MODE_BALANCED_WEIGHT" \
  --mode_balanced_warmup_steps "$MODE_BALANCED_WARMUP_STEPS" \
  --mode_balanced_k_max 128 \
  --mode_balanced_activity_threshold 1e-4 \
  --mode_balanced_denominator_eps 1e-6 \
  --hadamard_weight "$HADAMARD_WEIGHT" \
  --hadamard_interval "$HADAMARD_INTERVAL" \
  --hadamard_microbatch 8 \
  --hadamard_warmup_steps 500 \
  --hadamard_k_max 128 \
  --hadamard_sobolev_order 1 \
  --hadamard_fd_step_min 1e-3 \
  --hadamard_fd_step_max 3e-3 \
  --hadamard_min_surface_rms 1e-3 \
  --hadamard_denominator_eps 1e-12 \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --lr_warmup_steps "$LR_WARMUP_STEPS" \
  --weight_decay 1e-4 \
  --epochs "$EPOCHS" \
  --run_name "$RUN_NAME" 2>&1 | tee "/tmp/${RUN_NAME}.log"

uv run python - \
  "$RUN_NAME" "$BATCH_SIZE" "$HADAMARD_WEIGHT" "$HADAMARD_INTERVAL" \
  "$TRANSLATION_TANGENT_WEIGHT" "$MODE_BALANCED_WEIGHT" \
  "$MODE_BALANCED_WARMUP_STEPS" "$CUDA_DEVICES" <<'PY'
import json
import math
import sys
from pathlib import Path

(
    run_name,
    batch_size,
    hadamard_weight,
    hadamard_interval,
    tangent_weight,
    mode_weight,
    mode_warmup_steps,
    cuda_devices,
) = sys.argv[1:]
run_dir = Path("outputs") / run_name
config = json.loads((run_dir / "config.json").read_text())
expected = {
    "model": "cs_dno",
    "width": 640,
    "n_blocks": 8,
    "latent": 320,
    "batch_size": int(batch_size),
    "device_count": len(cuda_devices.split(",")),
    "cs_n_polys": 3,
    "cs_use_first_deriv": True,
    "cs_use_second_deriv": True,
    "cs_use_half_deriv": True,
    "cs_use_hilbert": True,
    "cs_mult_hidden": 160,
    "translation_tangent_weight": float(tangent_weight),
    "translation_tangent_scope": "tanaka",
    "translation_tangent_smoothing_scale": 1.0,
    "translation_tangent_denominator_eps": 1e-3,
    "mode_balanced_weight": float(mode_weight),
    "mode_balanced_warmup_steps": int(mode_warmup_steps),
    "mode_balanced_k_max": 128.0,
    "mode_balanced_activity_threshold": 1e-4,
    "mode_balanced_denominator_eps": 1e-6,
    "hadamard_weight": float(hadamard_weight),
    "hadamard_interval": int(hadamard_interval),
    "hadamard_microbatch": 8,
    "hadamard_warmup_steps": 500,
    "hadamard_k_max": 128.0,
    "hadamard_sobolev_order": 1,
    "hadamard_fd_step_min": 1e-3,
    "hadamard_fd_step_max": 3e-3,
    "hadamard_min_surface_rms": 1e-3,
    "hadamard_denominator_eps": 1e-12,
    "param_count": 1_342_400,
}
mismatches = {
    key: (config.get(key), value)
    for key, value in expected.items()
    if config.get(key) != value
}
if mismatches:
    raise SystemExit(f"Tanaka tangent configuration guard failed: {mismatches}")

records = [json.loads(line) for line in (run_dir / "train_log.jsonl").read_text().splitlines()]
if not records:
    raise SystemExit("Tanaka tangent configuration guard failed: empty training log")
nonfinite = {
    f"epoch_{record.get('epoch', index + 1)}.{key}": value
    for index, record in enumerate(records)
    for key, value in record.items()
    if isinstance(value, (int, float)) and not math.isfinite(value)
}
if nonfinite:
    raise SystemExit(f"Tanaka tangent finiteness guard failed: {nonfinite}")
print(
    "C27 Tanaka-only tangent guard passed; "
    "1,342,400 parameters; all logged scalars finite"
)
PY
