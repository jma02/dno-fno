#!/usr/bin/env bash
set -euo pipefail
cd /home/johnma/dno-fno

# Each subset starts from the same weights, not from the preceding trial.
sweep_dir=outputs/c27_ema_loss_subsets_20260928
source_run=outputs/c27_w320_b4_h80_ema_grad09_20260927
export UV_CACHE_DIR=/tmp/dno-uv-cache JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1
train_args=(
  --model cs_dno --norm scale
  --dataset outputs/paper_dataset_full_equal_tanaka_hard128_20260918/arrays
  --width 320 --n_blocks 8 --latent 4 --cs_mult_hidden 80 --cs_n_polys 3
  --batch_size 1024 --epochs 120 --stop_after_epoch 1 --seed 0
  --lr 5e-7 --lr_warmup_steps 500 --gradient_ema_decay 0.9
  --mode_balanced_warmup_steps 0 --hadamard_warmup_steps 0
  --hadamard_interval 16 --val_hadamard_interval 16 --hadamard_microbatch 8
  --metric_interval 16 --nonlinear_reference 0.002 --nonlinear_weight_mean 1.463313
  --init_checkpoint "$source_run/best_val_ckpt/ckpt_120" --output_root "$sweep_dir"
)

mkdir -p "$sweep_dir"
for subset in mb_tg mb_hd tg_hd mb tg hd data; do
  mode_weight=0 tangent_weight=0 hadamard_weight=0
  case "$subset" in *mb*) mode_weight=6 ;; esac
  case "$subset" in *tg*) tangent_weight=10 ;; esac
  case "$subset" in *hd*) hadamard_weight=0.01 ;; esac
  trial_dir="$sweep_dir/$subset"
  mkdir "$trial_dir"
  printf '%s %s training\n' "$(date +%H:%M)" "$subset"
  CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync python train-jax-10m/1d_dno_fno_jax_ema_grad.py \
    "${train_args[@]}" --run_name "$subset" \
    --mode_balanced_weight "$mode_weight" --translation_tangent_weight "$tangent_weight" \
    --hadamard_weight "$hadamard_weight" > "$trial_dir/console.log" 2>&1

  printf '%s %s rollouts\n' "$(date +%H:%M)" "$subset"
  eval_pids=()
  for gpu in 0 1; do
    if (( gpu == 0 )); then families=(stokes tanaka); else families=(jonswap_tma benjamin_feir); fi
    eval_dir="$trial_dir/eval_epoch1_test_stratified_n32_gpu$gpu"
    mkdir "$eval_dir"
    CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m solver.evals.eval_suite \
      --run_dir "$trial_dir" --checkpoint latest --gpu \
      --dataset outputs/paper_dataset_full_equal_tanaka_hard128_20260918/arrays \
      --families "${families[@]}" --n_ics 32 --ic_selection stratified --rollout_batch_size 32 \
      --truth_cache "$source_run/eval_best_current_test_stratified_n32_fp32net_fp64solver_gpu$gpu" \
      --output_dir "$eval_dir" > "$eval_dir/console.log" 2>&1 &
    eval_pids+=("$!")
  done
  eval_failed=0
  for pid in "${eval_pids[@]}"; do wait "$pid" || eval_failed=1; done
  (( eval_failed == 0 ))
  printf '%s %s complete\n' "$(date +%H:%M)" "$subset"
done
