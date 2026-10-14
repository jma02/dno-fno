#!/usr/bin/env bash
set -euo pipefail
cd /home/johnma/dno-fno

run_name=c27_nested_b16_ema_low_lr_continue_20261014
run_dir=outputs/$run_name
source_dir=outputs/c27_phase1_lr8_momentum_half_ema995_grad05_adam05_b2099_batch16_hd04_band128_nested1_gamma32_full_20261012
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
gpu_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)
[[ -z "$gpu_processes" ]] || { printf 'GPU 0 is occupied; continuation not started\n' >&2; exit 1; }
mkdir "$run_dir"
exec >> "$run_dir/console.log" 2>&1
# Keep the winning raw parameters, optimizer, gradient EMA, and weight EMA together.
cp -a "$source_dir/latest_ckpt" "$run_dir/latest_ckpt"
export JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1
export UV_CACHE_DIR=/tmp/dno-uv-cache MPLCONFIGDIR=/tmp/dno-matplotlib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.40

for epoch in 2 3; do
  printf '%s continuing candidate through epoch %s at LR 8e-6\n' "$(date +%H:%M)" "$epoch"
  uv run --no-sync python train-jax-10m/1d_dno_fno_jax_ema_grad.py \
    --model cs_dno --norm scale \
    --dataset outputs/paper_dataset_full_equal_tanaka_hard128_20260918/arrays \
    --width 320 --n_blocks 8 --latent 4 --cs_mult_hidden 80 --cs_n_polys 3 \
    --cs_normalize_surface_features --cs_surface_context_width 8 \
    --cs_nested_blocks 1 --cs_baseline_order 1 --cs_baseline_true_depth \
    --batch_size 16 --epochs 3 --stop_after_epoch "$epoch" --seed 0 \
    --train_fraction 1 --validation_fraction 1 --early_stopping_patience 0 \
    --lr 8e-5 --lr_warmup_steps 500 --schedule_epochs 1 --lr_continuation 8e-6 \
    --gradient_ema_decay 0.5 --adam_b1 0.5 --adam_b2 0.99 \
    --weight_ema_decay 0.995 --weight_ema_start_fraction 0.8 \
    --data_normalization target --data_k_max 128 \
    --mode_balanced_weight 6 --mode_balanced_warmup_steps 500 \
    --translation_tangent_weight 10 --translation_tangent_scope tanaka \
    --phase_mechanism all_momentum --phase_mechanism_weight 0.002574349830754761 \
    --hadamard_weight 0.04 --hadamard_warmup_steps 500 --hadamard_interval 16 \
    --val_hadamard_interval 16 --hadamard_microbatch 8 --metric_interval 16 \
    --rollout_training_weight 1 --rollout_training_loss full \
    --rollout_training_start_epoch 1 --rollout_training_ramp_epochs 1 \
    --rollout_training_interval 16 --rollout_training_microbatch 8 --rollout_training_steps 10 \
    --run_name "$run_name"
  printf '%s evaluating epoch %s weight EMA on all 128 rollouts\n' "$(date +%H:%M)" "$epoch"
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_experimental_stream_annotation=true \
    uv run --no-sync python scripts/eval_fast_rollouts.py \
      --run-dir "$run_dir" --checkpoint weight_ema --fuse-nested-fft --batch-context-fft \
      --reference-dir "$source_dir/eval_fast_weight_ema_nestedfused_contextbatch" \
      --output-dir "$run_dir/eval_epoch${epoch}_weight_ema"
done
printf '%s both continuation epochs and weight-EMA rollouts complete\n' "$(date +%H:%M)"
