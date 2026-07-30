#!/bin/bash
# Fresh C25 retrain: C20's accepted objective with a balanced 1.25x increase
# in the learned full-DNO trunk, branch rank, and depth-multiplier width.

set -euo pipefail
cd /home/johnma/dno-fno

MODE="${1:-full}"
DATASET="${DATASET:-combined_dataset_v9.npz}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LR="${LR:-2e-5}"
HADAMARD_WEIGHT="${HADAMARD_WEIGHT:-1e-2}"
HADAMARD_INTERVAL="${HADAMARD_INTERVAL:-16}"
TRANSLATION_TANGENT_WEIGHT="${TRANSLATION_TANGENT_WEIGHT:-10}"
MODE_BALANCED_WEIGHT="${MODE_BALANCED_WEIGHT:-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-}"
MODE_BALANCED_WARMUP_STEPS="${MODE_BALANCED_WARMUP_STEPS:-}"

case "$MODE" in
  full)
    DATA_FRACTION="${DATA_FRACTION:-1.0}"
    EPOCHS="${EPOCHS:-40}"
    LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}"
    MODE_BALANCED_WARMUP_STEPS="${MODE_BALANCED_WARMUP_STEPS:-500}"
    RUN_PREFIX="c25_capacity125_full"
    ;;
  smoke)
    DATA_FRACTION="${DATA_FRACTION:-0.01}"
    EPOCHS="${EPOCHS:-1}"
    LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-50}"
    MODE_BALANCED_WARMUP_STEPS="${MODE_BALANCED_WARMUP_STEPS:-50}"
    RUN_PREFIX="c25_capacity125_smoke"
    ;;
  *)
    echo "usage: $0 [full|smoke]" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-$EPOCHS}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python train-jax-10m/1d_dno_fno_jax.py \
  --model cs_dno \
  --norm scale \
  --dataset "$DATASET" \
  --data_fraction "$DATA_FRACTION" \
  --modes 64 \
  --width 640 \
  --n_blocks 8 \
  --latent 320 \
  --sobolev_k 1 \
  --cs_n_polys 3 \
  --cs_mult_hidden 160 \
  --cs_use_g1_baseline \
  --cs_g1_k_cut 0 \
  --cs_g1_fft_fp64 \
  --cs_tie_xi_out_mult \
  --cs_phi_bias_free \
  --cs_residual_eta_order 2 \
  --translation_tangent_weight "$TRANSLATION_TANGENT_WEIGHT" \
  --translation_tangent_window_depths 1 \
  --translation_tangent_energy_floor_relative 1e-3 \
  --mode_balanced_weight "$MODE_BALANCED_WEIGHT" \
  --mode_balanced_warmup_steps "$MODE_BALANCED_WARMUP_STEPS" \
  --mode_balanced_k_max 128 \
  --mode_balanced_active_scale_relative 1e-4 \
  --mode_balanced_denominator_floor_relative 1e-6 \
  --hadamard_weight "$HADAMARD_WEIGHT" \
  --hadamard_interval "$HADAMARD_INTERVAL" \
  --hadamard_microbatch 8 \
  --hadamard_warmup_steps 500 \
  --hadamard_k_max 128 \
  --hadamard_sobolev_order 1 \
  --hadamard_relative_eps_min 1e-3 \
  --hadamard_relative_eps_max 3e-3 \
  --hadamard_eta_scale_floor 1e-3 \
  --hadamard_denominator_floor 1e-12 \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --lr_warmup_steps "$LR_WARMUP_STEPS" \
  --weight_decay 1e-4 \
  --epochs "$EPOCHS" \
  --total_epochs "$TOTAL_EPOCHS" \
  --skip_dno_eval \
  --skip_plots \
  --run_name "$RUN_NAME" 2>&1 | tee "/tmp/${RUN_NAME}.log"

python3 - "$RUN_NAME" <<'PY'
import json
import sys
from pathlib import Path

run_name = sys.argv[1]
config_path = Path("outputs") / run_name / "config.json"
config = json.loads(config_path.read_text())
expected = {
    "model": "cs_dno",
    "width": 640,
    "n_blocks": 8,
    "latent": 320,
    "cs_mult_hidden": 160,
    "cs_use_g1_baseline": True,
    "cs_g1_k_cut": 0,
    "cs_g1_fft_fp64": True,
    "cs_tie_xi_out_mult": True,
    "cs_phi_bias_free": True,
    "cs_residual_eta_order": 2,
    "cs_depth_scaled_residual": False,
    "cs_block_k_cut": 0,
    "cs_residual_highband_cap": False,
    "cs_output_highband_cap": False,
    "param_count": 1_342_400,
}
mismatches = {
    key: (config.get(key), value)
    for key, value in expected.items()
    if config.get(key) != value
}
if mismatches:
    raise SystemExit(f"C25 architecture guard failed: {mismatches}")
print("C25 architecture guard passed: 1,342,400 parameters")
PY
