#!/usr/bin/env bash
set -euo pipefail
cd /home/johnma/dno-fno

screen_prefix=${1:?Run-name prefix required}
queued_control_pid=${2:?Waiting control-launcher PID required}
reference_run=outputs/c27_w320_b4_h80_ema_grad09_20260927
dataset=outputs/paper_dataset_full_equal_tanaka_hard128_20260918/arrays
export UV_CACHE_DIR=/tmp/dno-uv-cache CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.45

# Hold only our waiting launcher; release it on completion or failure.
trap 'kill -CONT "$queued_control_pid" 2>/dev/null || true' EXIT
kill -STOP "$queued_control_pid"
for mode in control scrambled; do
  run_name=${screen_prefix}_${mode}
  trial_dir=outputs/$run_name
  mkdir "$trial_dir"
  printf '%s training %s on GPU1 with a 45%% JAX memory pool\n' "$(date +%H:%M)" "$run_name"
  uv run --no-sync python train-jax-10m/1d_dno_fno_jax_ema_grad.py \
    --model cs_dno --norm scale --dataset "$dataset" \
    --width 320 --n_blocks 8 --latent 4 --cs_mult_hidden 80 --cs_n_polys 3 \
    --batch_size 1024 --epochs 1 --stop_after_epoch 1 --seed 0 \
    --lr 2e-5 --lr_warmup_steps 500 --gradient_ema_decay 0.9 --cs_baseline_true_depth \
    --data_normalization target --mode_balanced_weight 6 --mode_balanced_warmup_steps 500 \
    --translation_tangent_weight 10 --translation_tangent_scope tanaka \
    --hadamard_weight 0.01 --hadamard_warmup_steps 500 --hadamard_interval 16 \
    --val_hadamard_interval 16 --hadamard_microbatch 8 --metric_interval 16 \
    --dirichlet_probe_mode "$mode" --dirichlet_probe_fraction 0.25 \
    --run_name "$run_name" > "$trial_dir/console.log" 2>&1

  printf '%s rollouts %s\n' "$(date +%H:%M)" "$run_name"
  for group in 0 1; do
    if (( group == 0 )); then families=(stokes tanaka); else families=(jonswap_tma benjamin_feir); fi
    eval_dir="$trial_dir/eval_epoch1_test_stratified_n32_gpu$group"
    mkdir "$eval_dir"
    uv run --no-sync python -m solver.evals.eval_suite \
      --run_dir "$trial_dir" --checkpoint latest --gpu --dataset "$dataset" \
      --families "${families[@]}" --n_ics 32 --ic_selection stratified --rollout_batch_size 32 \
      --truth_cache "$reference_run/eval_best_current_test_stratified_n32_fp32net_fp64solver_gpu$group" \
      --output_dir "$eval_dir" > "$eval_dir/console.log" 2>&1
  done
done
printf '%s paired probe screen complete\n' "$(date +%H:%M)"
