#!/usr/bin/env bash
set -euo pipefail
cd /home/johnma/dno-fno

gpu=${1:?GPU index required}
shift
[[ "$gpu" == 0 || "$gpu" == 1 ]] || { printf 'Use local GPU 0 or 1\n' >&2; exit 2; }
(( $# )) || set -- control reflection polynomial_skip momentum_half momentum_quarter mb2 even_features lr_low lr_high multiplier_skip
run_date=$(date +%Y%m%d)
reference_run=outputs/c27_w320_b4_h80_ema_grad09_20260927
export CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1
export UV_CACHE_DIR=/tmp/dno-uv-cache MPLCONFIGDIR=/tmp/dno-matplotlib
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}

for variant in "$@"; do
  variant_args=()
  # Coefficients are calibrated at fresh initialization on training rows only.
  case "$variant" in
    control|control_ema) ;;
    reflection) variant_args=(--reflection_augmentation) ;;
    polynomial_skip) variant_args=(--cs_surface_polynomial_skip) ;;
    multiplier_skip) variant_args=(--cs_multiplier_skip) ;;
    momentum_half) variant_args=(--phase_mechanism all_momentum --phase_mechanism_weight 0.002574349830754761) ;;
    momentum_quarter) variant_args=(--phase_mechanism all_momentum --phase_mechanism_weight 0.0012871749153773805) ;;
    mb2) variant_args=(--mode_balanced_weight 2) ;;
    even_features) variant_args=(--no-cs_use_first_deriv --no-cs_use_hilbert) ;;
    lr_low) variant_args=(--lr 1e-5) ;;
    lr_high) variant_args=(--lr 4e-5) ;;
    lr_high_momentum_half) variant_args=(--lr 4e-5 --phase_mechanism all_momentum --phase_mechanism_weight 0.002574349830754761) ;;
    lr_high_momentum_quarter) variant_args=(--lr 4e-5 --phase_mechanism all_momentum --phase_mechanism_weight 0.0012871749153773805) ;;
    lr6_momentum_half) variant_args=(--lr 6e-5 --phase_mechanism all_momentum --phase_mechanism_weight 0.002574349830754761) ;;
    lr6_momentum_half_ema995) variant_args=(--lr 6e-5 --phase_mechanism all_momentum --phase_mechanism_weight 0.002574349830754761 --weight_ema_decay 0.995) ;;
    lr6_momentum_half_ema995_hold) variant_args=(--lr 6e-5 --phase_mechanism all_momentum --phase_mechanism_weight 0.002574349830754761 --weight_ema_decay 0.995 --lr_hold_fraction 0.5) ;;
    lr6_momentum_half_ema995_momentum_ramp) variant_args=(--lr 6e-5 --phase_mechanism all_momentum --phase_mechanism_weight 0.002574349830754761 --weight_ema_decay 0.995 --phase_mechanism_ramp_epochs 0.5) ;;
    lr6_momentum_half_ema995_grad05) variant_args=(--lr 6e-5 --phase_mechanism all_momentum --phase_mechanism_weight 0.002574349830754761 --weight_ema_decay 0.995 --gradient_ema_decay 0.5) ;;
    lr6_momentum_half_ema995_head2) variant_args=(--lr 6e-5 --phase_mechanism all_momentum --phase_mechanism_weight 0.002574349830754761 --weight_ema_decay 0.995 --surface_head_lr_multiplier 2) ;;
    lr6_momentum_half_ema995_band128) variant_args=(--lr 6e-5 --phase_mechanism all_momentum --phase_mechanism_weight 0.002574349830754761 --weight_ema_decay 0.995 --data_k_max 128) ;;
    lr6_momentum_half_ema995_log_inputs) variant_args=(--lr 6e-5 --phase_mechanism all_momentum --phase_mechanism_weight 0.002574349830754761 --weight_ema_decay 0.995 --cs_log_multiplier_inputs) ;;
    lr_high_momentum_half_ema995) variant_args=(--lr 4e-5 --phase_mechanism all_momentum --phase_mechanism_weight 0.002574349830754761 --weight_ema_decay 0.995) ;;
    lr_high_momentum_half_ramp025) variant_args=(--lr 4e-5 --phase_mechanism all_momentum --phase_mechanism_weight 0.002574349830754761 --rollout_training_ramp_epochs 0.25) ;;
    stokes) variant_args=(--phase_mechanism stokes_carrier --phase_mechanism_weight 0.002797671729980144) ;;
    tanaka) variant_args=(--phase_mechanism tanaka_momentum --phase_mechanism_weight 0.01384754275706006) ;;
    bf) variant_args=(--phase_mechanism bf_sideband --phase_mechanism_weight 0.00013728115168189654) ;;
    jonswap) variant_args=(--jonswap_mode_balanced_scale 0) ;;
    nested) variant_args=(--cs_nested_blocks 2) ;;
    nested4) variant_args=(--cs_nested_blocks 4) ;;
    momentum) variant_args=(--phase_mechanism all_momentum --phase_mechanism_weight 0.005148699661509522) ;;
    nested_momentum) variant_args=(--cs_nested_blocks 2 --phase_mechanism all_momentum --phase_mechanism_weight 0.005148699661509522) ;;
    *) printf 'Unsupported phase experiment: %s\n' "$variant" >&2; exit 2 ;;
  esac
  run_name=c27_phase1_${variant}_full_${run_date}
  trial_dir=outputs/$run_name
  printf '%s preparing Ada GPU%s for %s\n' "$(date +%H:%M)" "$gpu" "$variant"
  while true; do
    if [[ -d "$trial_dir" ]]; then
      printf '%s skipping claimed experiment %s\n' "$(date +%H:%M)" "$run_name"
      continue 2
    fi
    if [[ ${DNO_SHARE_GPU:-0} == 1 ]]; then
      if mkdir "$trial_dir"; then break; fi
      [[ -d "$trial_dir" ]] || exit 1
      continue
    fi
    gpu_processes=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
    if [[ -z "$gpu_processes" ]]; then
      sleep 30
      gpu_processes=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
      if [[ -z "$gpu_processes" ]]; then
        if mkdir "$trial_dir"; then break; fi
        [[ -d "$trial_dir" ]] || exit 1
      fi
    fi
    sleep 30
  done
  printf '%s fresh training %s; final 128-case weight-EMA rollouts follow\n' "$(date +%H:%M)" "$run_name"
  uv run --no-sync python train-jax-10m/1d_dno_fno_jax_ema_grad.py \
    --model cs_dno --norm scale \
    --dataset outputs/paper_dataset_full_equal_tanaka_hard128_20260918/arrays \
    --width 320 --n_blocks 8 --latent 4 --cs_mult_hidden 80 --cs_n_polys 3 \
    --cs_normalize_surface_features --cs_surface_context_width 8 --cs_baseline_true_depth \
    --batch_size 1024 --epochs 1 --stop_after_epoch 1 --seed 0 \
    --train_fraction 1 --validation_fraction 1 \
    --lr 2e-5 --lr_warmup_steps 500 --gradient_ema_decay 0.9 \
    --weight_ema_decay 0.999 --weight_ema_start_fraction 0.8 \
    --data_normalization target --mode_balanced_weight 6 --mode_balanced_warmup_steps 500 \
    --translation_tangent_weight 10 --translation_tangent_scope tanaka \
    --hadamard_weight 0.01 --hadamard_warmup_steps 500 --hadamard_interval 16 \
    --val_hadamard_interval 16 --hadamard_microbatch 8 --metric_interval 16 \
    --rollout_training_weight 1 --rollout_training_loss full \
    --rollout_training_start_epoch 1 --rollout_training_ramp_epochs 1 \
    --rollout_training_interval 16 --rollout_training_microbatch 8 --rollout_training_steps 10 \
    --rollout_epochs 1 --rollout_truth_run "$reference_run" \
    --run_name "$run_name" "${variant_args[@]}" > "$trial_dir/console.log" 2>&1
  uv run --no-sync paper-plots/plot_neural_advantage.py --one-epoch >> "$trial_dir/console.log" 2>&1 \
    || printf '%s Figure12b refresh failed; see %s/console.log\n' "$(date +%H:%M)" "$trial_dir" >&2
  if [[ ${DNO_SHARE_GPU:-0} == 1 ]]; then
    printf '%s %s complete; isolated timings deferred while sharing GPU\n' "$(date +%H:%M)" "$run_name"
    continue
  fi
  printf '%s %s training and rollouts complete; measuring single-rollout runtime\n' "$(date +%H:%M)" "$run_name"
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_experimental_stream_annotation=true \
    uv run --no-sync scripts/benchmark_dno_fusion.py --run "$trial_dir" --checkpoint weight_ema \
      --variants original fourier_fused_features_product_overlap --repeats 100 --ept 8 \
      --output "$trial_dir/timing/forward" >> "$trial_dir/console.log" 2>&1
  uv run --no-sync python -c \
    'import json, sys; from pathlib import Path; result = json.loads(Path(sys.argv[1]).read_text()); assert result["variants"]["fourier_fused_features_product_overlap"]["validation"]["passed"], "Optimized inference failed parity"' \
    "$trial_dir/timing/forward/results.json"
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_experimental_stream_annotation=true \
    uv run --no-sync scripts/time_single_rollouts.py --candidate-run "$trial_dir" --checkpoint weight_ema \
      --reference cached_M6 --repeats 3 --pad-factor 2 --ept 8 \
      --methods fourier_fused_features_product_overlap_spectral_all_boundaries_stages_carry_local \
      --output "$trial_dir/timing/timings.json" >> "$trial_dir/console.log" 2>&1
  printf '%s %s complete\n' "$(date +%H:%M)" "$run_name"
done
