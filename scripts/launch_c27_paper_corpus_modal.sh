#!/usr/bin/env bash
# Fresh exact-C27 training on the authenticated paper corpus using two Modal A100s.

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
RUN_NAME="${RUN_NAME:-c27_paper_corpus_full_modal_${GPU_TAG}_$(date +%Y%m%d_%H%M%S)}"
DATASET="/data/outputs/paper_corpus_literature_aligned_v1/combined/c16384_v01024_t01024/paper_corpus_all_splits_c16384.dataset.json"
mkdir -p logs
LOG="logs/${RUN_NAME}.log"

TRAINER_ARGS="--cs_n_polys 3 --translation_tangent_weight 10 --translation_tangent_window_depths 1 --translation_tangent_energy_floor_relative 1e-3 --mode_balanced_weight 6 --mode_balanced_warmup_steps 500 --mode_balanced_k_max 128 --mode_balanced_active_scale_relative 1e-4 --mode_balanced_denominator_floor_relative 1e-6 --hadamard_weight 1e-2 --hadamard_interval 16 --hadamard_microbatch 8 --hadamard_warmup_steps 500 --hadamard_k_max 128 --hadamard_sobolev_order 1 --hadamard_relative_eps_min 1e-3 --hadamard_relative_eps_max 3e-3 --hadamard_eta_scale_floor 1e-3 --hadamard_denominator_floor 1e-12 --lr_warmup_steps 500 --data_fraction 1.0"

MODAL_GPU="$GPU_SPEC" modal run --detach scripts/modal_train.py::train \
  --dataset "$DATASET" \
  --run-name "$RUN_NAME" \
  --model-kind cs_dno \
  --norm scale \
  --precision fp32 \
  --seed 0 \
  --width 640 \
  --n-blocks 8 \
  --latent 320 \
  --batch-size 1024 \
  --lr 2e-5 \
  --weight-decay 1e-4 \
  --epochs 40 \
  --total-epochs 40 \
  --cs-mult-hidden 160 \
  --spawn \
  --trainer-args "$TRAINER_ARGS" 2>&1 | tee "$LOG"

echo "run_name=$RUN_NAME"
echo "gpu_spec=$GPU_SPEC"
echo "log=$LOG"
