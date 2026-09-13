#!/usr/bin/env bash
# Fresh C27 training with Tanaka-only translation loss on two Modal A100s.

set -euo pipefail
cd /home/johnma/dno-fno

export MODAL_PROFILE="${MODAL_PROFILE:-sciml-at-ud}"
PROFILE="$(modal profile current)"
if [[ "$PROFILE" != "sciml-at-ud" ]]; then
  echo "refusing to launch under Modal profile '$PROFILE'; use sciml-at-ud" >&2
  exit 1
fi

GPU_SPEC="${GPU_SPEC:-A100-80GB:2}"
GPU_TAG="${GPU_SPEC//:/x}"
GPU_TAG="${GPU_TAG,,}"
RUN_NAME="${RUN_NAME:-c27_tanaka_tangent_modal_${GPU_TAG}_$(date +%Y%m%d_%H%M%S)}"
DATASET="${DATASET:-/data/outputs/paper_dataset/arrays}"
EPOCHS="${EPOCHS:-40}"
mkdir -p logs
LOG="logs/${RUN_NAME}.log"

TRAINER_ARGS="--cs_n_polys 3 --translation_tangent_weight 10 --translation_tangent_smoothing_scale 1 --translation_tangent_denominator_eps 1e-3 --mode_balanced_weight 6 --mode_balanced_warmup_steps 500 --mode_balanced_k_max 128 --mode_balanced_activity_threshold 1e-4 --mode_balanced_denominator_eps 1e-6 --hadamard_weight 1e-2 --hadamard_interval 16 --hadamard_microbatch 8 --hadamard_warmup_steps 500 --hadamard_k_max 128 --hadamard_sobolev_order 1 --hadamard_fd_step_min 1e-3 --hadamard_fd_step_max 3e-3 --hadamard_min_surface_rms 1e-3 --hadamard_denominator_eps 1e-12 --lr_warmup_steps 500"

MODAL_GPU="$GPU_SPEC" modal run --detach scripts/modal_train.py::train \
  --dataset "$DATASET" \
  --run-name "$RUN_NAME" \
  --model-kind cs_dno \
  --norm scale \
  --seed 0 \
  --width 640 \
  --n-blocks 8 \
  --latent 320 \
  --batch-size 1024 \
  --lr 2e-5 \
  --weight-decay 1e-4 \
  --epochs "$EPOCHS" \
  --cs-mult-hidden 160 \
  --spawn \
  --trainer-args "$TRAINER_ARGS" 2>&1 | tee "$LOG"

echo "run_name=$RUN_NAME"
echo "gpu_spec=$GPU_SPEC"
echo "log=$LOG"
