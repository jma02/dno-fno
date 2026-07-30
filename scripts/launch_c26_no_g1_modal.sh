#!/usr/bin/env bash
# Matched C25 component ablation: remove only the analytic G1 term while
# retaining the identical learned O(eta^2) residual. Everything else is fixed.

set -euo pipefail
cd /home/johnma/dno-fno

export MODAL_PROFILE="${MODAL_PROFILE:-sciml-at-ud}"
PROFILE="$(modal profile current)"
if [[ "$PROFILE" != "sciml-at-ud" ]]; then
  echo "refusing to launch under Modal profile '$PROFILE'; use sciml-at-ud" >&2
  exit 1
fi

GPU_SPEC="${GPU_SPEC:-L40S:2}"
GPU_TAG="${GPU_SPEC//:/x}"
GPU_TAG="${GPU_TAG,,}"

MODE="${1:-full}"
case "$MODE" in
  full)
    DATA_FRACTION="${DATA_FRACTION:-1.0}"
    EPOCHS="${EPOCHS:-40}"
    LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}"
    MODE_WARMUP_STEPS="${MODE_WARMUP_STEPS:-500}"
    RUN_PREFIX="c26_no_g1_oeta2_full_modal_${GPU_TAG}"
    DETACH=(--detach)
    ;;
  smoke)
    DATA_FRACTION="${DATA_FRACTION:-0.01}"
    EPOCHS="${EPOCHS:-1}"
    LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-50}"
    MODE_WARMUP_STEPS="${MODE_WARMUP_STEPS:-50}"
    RUN_PREFIX="c26_no_g1_oeta2_smoke_modal_${GPU_TAG}"
    DETACH=()
    ;;
  *)
    echo "usage: $0 [full|smoke]" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p logs
LOG="logs/${RUN_NAME}.log"

TRAINER_ARGS="--cs_n_polys 3 --cs_g1_fft_fp64 --cs_phi_bias_free --cs_residual_eta_order 2 --translation_tangent_weight 10 --translation_tangent_window_depths 1 --translation_tangent_energy_floor_relative 1e-3 --mode_balanced_weight 6 --mode_balanced_warmup_steps ${MODE_WARMUP_STEPS} --mode_balanced_k_max 128 --mode_balanced_active_scale_relative 1e-4 --mode_balanced_denominator_floor_relative 1e-6 --hadamard_weight 1e-2 --hadamard_interval 16 --hadamard_microbatch 8 --hadamard_warmup_steps 500 --hadamard_k_max 128 --hadamard_sobolev_order 1 --hadamard_relative_eps_min 1e-3 --hadamard_relative_eps_max 3e-3 --hadamard_eta_scale_floor 1e-3 --hadamard_denominator_floor 1e-12 --lr_warmup_steps ${LR_WARMUP_STEPS} --data_fraction ${DATA_FRACTION} --skip_plots"

MODAL_GPU="$GPU_SPEC" modal run "${DETACH[@]}" modal_train.py::train \
  --dataset combined_dataset_v9.npz \
  --test-dataset test_dno_rescaled.npz \
  --run-name "$RUN_NAME" \
  --model-kind cs_dno \
  --norm scale \
  --precision fp32 \
  --seed 0 \
  --modes 64 \
  --width 640 \
  --n-blocks 8 \
  --latent 320 \
  --sobolev-k 1 \
  --batch-size 1024 \
  --lr 2e-5 \
  --weight-decay 1e-4 \
  --epochs "$EPOCHS" \
  --total-epochs "$EPOCHS" \
  --cs-mult-hidden 160 \
  --cs-g1-k-cut 0 \
  --cs-tie-xi-out-mult \
  --skip-dno-eval \
  --trainer-args "$TRAINER_ARGS" 2>&1 | tee "$LOG"

echo "run_name=$RUN_NAME"
echo "log=$LOG"
