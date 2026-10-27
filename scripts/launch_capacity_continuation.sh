#!/usr/bin/env bash
# Extended training of the one-epoch blocks16 capacity arm with the hero37 continuation recipe
# (launch_nested_candidate_continuation.sh): resume its epoch-1 state, train one epoch per call at the held
# continuation LR, evaluate the weight EMA on all 128 rollouts after each epoch, keep the checkpoint with the
# lowest pooled terminal error as rollout_best_epoch<N>_ckpt, and stop after five epochs without
# EMA-validation improvement.
set -euo pipefail
cd /home/samzheng/dno-fno

run_name=c27_capacity_blocks16_low_lr_ep60_20261027
run_dir=outputs/$run_name
source_dir=outputs/c27_capacity_blocks16_scratch1_20261027
shared=/home/johnma/dno-fno/outputs/shared_samzheng_20261026
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-GPU-c44b86bf-148d-aa7e-32ec-77d4f2f7ce71}
gpu_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)
[[ -z "$gpu_processes" ]] || { printf 'GPU is occupied; continuation not started\n' >&2; exit 1; }
mkdir "$run_dir"
exec >> "$run_dir/console.log" 2>&1
# Keep the epoch-1 raw parameters, optimizer, gradient EMA and weight EMA together.
cp -a "$source_dir/latest_ckpt" "$run_dir/latest_ckpt"
export JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.40

best_rollout=inf
for ((epoch=2; epoch<=60; epoch++)); do
  printf '%s continuing blocks16 through epoch %s at LR 8e-6\n' "$(date +%H:%M)" "$epoch"
  uv run --no-sync python train-jax-10m/1d_dno_fno_jax_ema_grad.py \
    --model cs_dno --norm scale --dataset "$shared/data" \
    --width 320 --n_blocks 16 --latent 4 --cs_mult_hidden 80 --cs_n_polys 3 \
    --cs_normalize_surface_features --cs_surface_context_width 8 \
    --cs_nested_blocks 1 --cs_baseline_order 1 --cs_baseline_true_depth \
    --batch_size 16 --epochs 60 --stop_after_epoch "$epoch" --seed 0 \
    --train_fraction 1 --validation_fraction 1 --early_stopping_patience 5 \
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
  read -r completed best_epoch < <(uv run --no-sync python -c \
    'import json, sys; m=json.load(open(sys.argv[1])); print(m["epoch"], m["best_epoch"])' \
    "$run_dir/latest_ckpt/metadata.json")
  (( completed == epoch )) || break
  printf '%s evaluating epoch %s weight EMA on all 128 rollouts\n' "$(date +%H:%M)" "$epoch"
  eval_dir=$run_dir/eval_epoch${epoch}_weight_ema
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_experimental_stream_annotation=true \
    uv run --no-sync python scripts/eval_fast_rollouts.py \
      --run-dir "$run_dir" --checkpoint weight_ema --fuse-nested-fft --batch-context-fft \
      --reference-dir "$shared/reference" --output-dir "$eval_dir"
  rollout=$(uv run --no-sync python -c '
import sys
import numpy as np
errors = []
for family in ("stokes", "tanaka", "benjamin_feir", "jonswap_tma"):
    with np.load(f"{sys.argv[1]}/{family}_trajs.npz") as data:
        errors.append(data["rel_l2_eta"][-1])
print(float(np.exp(np.log(np.concatenate(errors)).mean())))' "$eval_dir")
  printf '%s epoch %s pooled terminal GM %s (best so far %s)\n' "$(date +%H:%M)" "$epoch" "$rollout" "$best_rollout"
  if python3 -c 'import sys; sys.exit(not float(sys.argv[1]) < float(sys.argv[2]))' "$rollout" "$best_rollout"; then
    rm -rf "$run_dir"/rollout_best_epoch*_ckpt
    cp -a "$run_dir/latest_ckpt" "$run_dir/rollout_best_epoch${epoch}_ckpt"
    best_rollout=$rollout
    printf '%s kept epoch %s as the rollout-best checkpoint\n' "$(date +%H:%M)" "$epoch"
  fi
  if (( completed - best_epoch >= 5 )); then
    printf '%s early stopping: five epochs without EMA-validation improvement\n' "$(date +%H:%M)"
    break
  fi
done
printf '%s continuation complete at epoch %s; best EMA-validation epoch %s; rollout-best GM %s\n' \
  "$(date +%H:%M)" "$completed" "$best_epoch" "$best_rollout"
