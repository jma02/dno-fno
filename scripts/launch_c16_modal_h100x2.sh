#!/usr/bin/env bash
# Launch the clean C16 retrain on the personal jma02 Modal workspace.

set -euo pipefail
cd /home/johnma/dno-fno

PROFILE="$(uv run modal profile current)"
if [[ "$PROFILE" != "jma02" ]]; then
  echo "refusing to launch under Modal profile '$PROFILE'; activate jma02" >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-c16_hadamard_g01_oeta2_modal_h100x2_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-16}"
mkdir -p logs
LOG="logs/${RUN_NAME}.log"

TRAINER_ARGS="--cs_g1_fft_fp64 --cs_phi_bias_free --cs_residual_eta_order 2 --hadamard_weight 1e-2 --hadamard_interval 16 --hadamard_microbatch 8 --hadamard_warmup_steps 500 --hadamard_k_max 128 --hadamard_sobolev_order 1 --hadamard_relative_eps_min 1e-3 --hadamard_relative_eps_max 3e-3 --hadamard_eta_scale_floor 1e-3 --hadamard_denominator_floor 1e-12 --lr_warmup_steps 500 --skip_plots"

MODAL_GPU="H100:2" uv run modal run --detach modal_train.py::train \
  --dataset combined_dataset_v9.npz \
  --test-dataset test_dno_rescaled.npz \
  --run-name "$RUN_NAME" \
  --model-kind cs_dno \
  --norm scale \
  --precision fp32 \
  --modes 64 \
  --width 512 \
  --n-blocks 8 \
  --latent 256 \
  --sobolev-k 1 \
  --batch-size 1024 \
  --lr 2e-5 \
  --weight-decay 1e-4 \
  --epochs "$EPOCHS" \
  --cs-mult-hidden 128 \
  --cs-use-g1-baseline \
  --cs-g1-k-cut 0 \
  --cs-tie-xi-out-mult \
  --skip-dno-eval \
  --trainer-args "$TRAINER_ARGS" 2>&1 | tee "$LOG"

echo "run_name=$RUN_NAME"
echo "log=$LOG"
