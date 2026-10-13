#!/usr/bin/env bash
set -euo pipefail
cd /home/johnma/dno-fno

run_name=c27_nested_b16_gamma32_ema_ep120_20261013
run_dir=outputs/$run_name
reference_dir=outputs/c27_phase1_lr8_momentum_half_ema995_grad05_adam05_b2099_batch16_hd04_band128_nested1_gamma32_full_20261012/eval_fast_weight_ema_nestedfused_contextbatch
mode=${1:-fresh}
case "$mode" in
  fresh) mkdir "$run_dir" ;;
  --resume-after-epoch-7) old_train_pid=${2:?Current trainer PID required} ;;
  *) printf 'Use no arguments for fresh training, or --resume-after-epoch-7 TRAINER_PID\n' >&2; exit 2 ;;
esac
exec >> "$run_dir/console.log" 2>&1
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
export JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1
export UV_CACHE_DIR=/tmp/dno-uv-cache MPLCONFIGDIR=/tmp/dno-matplotlib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.40

ema_start_fraction=0.8
early_stopping_patience=0
if [[ "$mode" == --resume-after-epoch-7 ]]; then
  printf '%s waiting for completed epoch 7 before approved restart\n' "$(date +%H:%M)"
  while true; do
    saved_epoch=$(uv run --no-sync python -c \
      'import json, sys; print(json.load(open(sys.argv[1]))["epoch"])' \
      "$run_dir/latest_ckpt/metadata.json")
    if (( saved_epoch >= 7 )); then
      [[ "$saved_epoch" == 7 ]] || { printf 'Expected epoch 7; found %s\n' "$saved_epoch"; exit 1; }
      # Metadata is published atomically only after the complete checkpoint is saved.
      old_command=$(ps -p "$old_train_pid" -o args=)
      [[ "$old_command" == *1d_dno_fno_jax_ema_grad.py*"$run_name"* ]] || exit 1
      kill -TERM "$old_train_pid"
      while kill -0 "$old_train_pid" 2>/dev/null; do sleep 1; done
      break
    fi
    kill -0 "$old_train_pid" || { printf 'Trainer exited before epoch 7 was saved\n'; exit 1; }
    sleep 5
  done
  # 7/120 puts the EMA start exactly at restored step 5,161,065. The next
  # update seeds the shadow from trained parameters, not its dormant initial values.
  ema_start_fraction=0.058333333333333334
  early_stopping_patience=5
fi

printf '%s queued on GPU 0; waiting for an idle device\n' "$(date +%H:%M)"
while true; do
  gpu_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)
  if [[ -z "$gpu_processes" ]]; then
    sleep 30
    gpu_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)
    [[ -z "$gpu_processes" ]] && break
  fi
  sleep 30
done
if [[ "$mode" == --resume-after-epoch-7 ]]; then
  printf '%s evaluating pre-EMA epoch 7 raw weights on all 128 rollouts\n' "$(date +%H:%M)"
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_experimental_stream_annotation=true \
    uv run --no-sync python scripts/eval_fast_rollouts.py \
      --run-dir "$run_dir" --checkpoint latest --fuse-nested-fft --batch-context-fft \
      --output-dir "$run_dir/eval_epoch7_raw_pre_ema" --reference-dir "$reference_dir"
fi
printf '%s starting candidate training (%s), maximum 120 epochs, early-stop patience %s\n' "$(date +%H:%M)" "$mode" "$early_stopping_patience"
uv run --no-sync python train-jax-10m/1d_dno_fno_jax_ema_grad.py \
  --model cs_dno --norm scale \
  --dataset outputs/paper_dataset_full_equal_tanaka_hard128_20260918/arrays \
  --width 320 --n_blocks 8 --latent 4 --cs_mult_hidden 80 --cs_n_polys 3 \
  --cs_normalize_surface_features --cs_surface_context_width 8 \
  --cs_nested_blocks 1 --cs_baseline_order 1 --cs_baseline_true_depth \
  --batch_size 16 --epochs 120 --stop_after_epoch 120 --seed 0 \
  --train_fraction 1 --validation_fraction 1 --early_stopping_patience "$early_stopping_patience" \
  --lr 8e-5 --lr_warmup_steps 500 --gradient_ema_decay 0.5 \
  --adam_b1 0.5 --adam_b2 0.99 --weight_ema_decay 0.995 --weight_ema_start_fraction "$ema_start_fraction" \
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

printf '%s evaluating final weight-EMA checkpoint on all 128 rollouts\n' "$(date +%H:%M)"
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_experimental_stream_annotation=true \
  uv run --no-sync python scripts/eval_fast_rollouts.py \
    --run-dir "$run_dir" --checkpoint weight_ema --fuse-nested-fft --batch-context-fft \
    --reference-dir "$reference_dir"
printf '%s training and final weight-EMA rollouts complete\n' "$(date +%H:%M)"
