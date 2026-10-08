#!/usr/bin/env bash
set -euo pipefail
cd /home/johnma/dno-fno

gpu=${1:?GPU index required}
shift
[[ "$gpu" == 0 || "$gpu" == 1 ]] || { printf 'Use local GPU 0 or 1\n' >&2; exit 2; }
(( $# )) || set -- stokes tanaka bf jonswap
run_date=$(date +%Y%m%d)
reference_run=outputs/c27_w320_b4_h80_ema_grad09_20260927
export CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1
export UV_CACHE_DIR=/tmp/dno-uv-cache MPLCONFIGDIR=/tmp/dno-matplotlib
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.80

for variant in "$@"; do
  variant_args=()
  # Fixed at 10% of the target family's data-gradient norm at fresh initialization.
  case "$variant" in
    control) ;;
    stokes) variant_args=(--phase_mechanism stokes_carrier --phase_mechanism_weight 0.002797671729980144) ;;
    tanaka) variant_args=(--phase_mechanism tanaka_momentum --phase_mechanism_weight 0.01384754275706006) ;;
    bf) variant_args=(--phase_mechanism bf_sideband --phase_mechanism_weight 0.00013728115168189654) ;;
    jonswap) variant_args=(--jonswap_mode_balanced_scale 0) ;;
    *) printf 'Unsupported phase experiment: %s\n' "$variant" >&2; exit 2 ;;
  esac
  run_name=c27_phase40_${variant}_subset16_${run_date}
  trial_dir=outputs/$run_name
  printf '%s waiting for idle Ada GPU%s for %s\n' "$(date +%H:%M)" "$gpu" "$variant"
  while true; do
    if [[ -d "$trial_dir" ]]; then
      printf '%s skipping claimed experiment %s\n' "$(date +%H:%M)" "$run_name"
      continue 2
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
  printf '%s fresh training %s; final 128-case rollouts follow\n' "$(date +%H:%M)" "$run_name"
  uv run --no-sync python train-jax-10m/1d_dno_fno_jax_ema_grad.py \
    --model cs_dno --norm scale \
    --dataset outputs/paper_dataset_full_equal_tanaka_hard128_20260918/arrays \
    --width 320 --n_blocks 8 --latent 4 --cs_mult_hidden 80 --cs_n_polys 3 \
    --cs_normalize_surface_features --cs_surface_context_width 8 --cs_baseline_true_depth \
    --batch_size 1024 --epochs 40 --stop_after_epoch 40 --seed 0 \
    --train_fraction 0.0625 --validation_fraction 0.0625 \
    --lr 2e-5 --lr_warmup_steps 500 --gradient_ema_decay 0.9 \
    --data_normalization target --mode_balanced_weight 6 --mode_balanced_warmup_steps 500 \
    --translation_tangent_weight 10 --translation_tangent_scope tanaka \
    --hadamard_weight 0.01 --hadamard_warmup_steps 500 --hadamard_interval 16 \
    --val_hadamard_interval 16 --hadamard_microbatch 8 --metric_interval 16 \
    --rollout_training_weight 1 --rollout_training_loss full \
    --rollout_training_start_epoch 20 --rollout_training_ramp_epochs 2 \
    --rollout_training_interval 16 --rollout_training_microbatch 8 --rollout_training_steps 10 \
    --rollout_epochs 40 --rollout_truth_run "$reference_run" \
    --run_name "$run_name" "${variant_args[@]}" > "$trial_dir/console.log" 2>&1
  printf '%s %s training and rollouts complete\n' "$(date +%H:%M)" "$run_name"
done
