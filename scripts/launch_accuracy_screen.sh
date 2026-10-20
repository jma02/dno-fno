#!/usr/bin/env bash
set -euo pipefail
cd /home/johnma/dno-fno

export CUDA_VISIBLE_DEVICES=GPU-c44b86bf-148d-aa7e-32ec-77d4f2f7ce71
export JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1
export UV_CACHE_DIR=/tmp/dno-uv-cache MPLCONFIGDIR=/tmp/dno-matplotlib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
control_dir=outputs/c27_phase1_lr8_momentum_half_ema995_grad05_adam05_b2099_batch16_hd04_band128_nested1_gamma32_full_20261012
reference_dir=$control_dir/eval_fast_weight_ema_nestedfused_contextbatch
run_date=$(date +%Y%m%d)

if (( $# == 0 )); then
    printf 'Usage: %s full_q_teacher|rhs_teacher|parity_products|phase_direction|phase_rank_one|phase_wavenumber|surface_amplitude_fp32|nested_features|cs_terms|cs_terms_m1|cs_terms_w160 [...]\n' "$0" >&2
    exit 2
fi
for arm in "$@"; do
    export JAX_PLATFORMS=cuda
    eval_args=(--fuse-nested-fft --batch-context-fft)
    case "$arm" in
        full_q_teacher) arm_args=(--rollout_training_full_spectrum) ;;
        rhs_teacher) arm_args=(--rollout_training_full_spectrum --rollout_training_rhs) ;;
        parity_products) arm_args=(--cs_surface_parity_products) ;;
        phase_direction) arm_args=(--data_phase_projection_weight 1) ;;
        phase_rank_one) arm_args=(--cs_phase_rank_one) ;;
        phase_wavenumber) arm_args=(--cs_phase_wavenumber_hidden 8) ;;
        surface_amplitude_fp32) arm_args=(--cs_surface_amplitude_normalization) ;;
        nested_features) arm_args=(--cs_nested_surface_features) ;;
        cs_terms|cs_terms_m1|cs_terms_w160)
            arm_args=(--cs_terms_residual --cs_terms_cpu_prefetch --cs_nested_blocks 0 --cs_baseline_order 3)
            eval_args=(--cs-terms-variant terms_interleaved_features_cufftdx)
            if [[ "$arm" == cs_terms_m1 ]]; then
                arm_args+=(--cs_baseline_order 1)
            fi
            if [[ "$arm" == cs_terms_w160 ]]; then
                arm_args+=(--width 160)
            fi
            export JAX_PLATFORMS=cuda,cpu
            ;;
        *) printf 'Unknown experiment: %s\n' "$arm" >&2; exit 2 ;;
    esac
    run_name=c27_${arm}_scratch1_${run_date}
    run_dir=outputs/$run_name
    mkdir "$run_dir"
    while :; do
        gpu_pids=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)
        [[ -z "$gpu_pids" ]] && break
        sleep 30
    done
    printf '%s starting %s from scratch\n' "$(date +%H:%M)" "$arm"
    uv run --no-sync python train-jax-10m/1d_dno_fno_jax_ema_grad.py \
        --model cs_dno --norm scale \
        --dataset outputs/paper_dataset_full_equal_tanaka_hard128_20260918/arrays \
        --width 320 --n_blocks 8 --latent 4 --cs_mult_hidden 80 --cs_n_polys 3 \
        --cs_normalize_surface_features --cs_surface_context_width 8 --cs_nested_blocks 1 \
        --cs_baseline_order 1 --cs_baseline_true_depth \
        --batch_size 16 --epochs 1 --stop_after_epoch 1 --schedule_epochs 1 --seed 0 \
        --train_fraction 1 --validation_fraction 1 --lr 8e-5 --lr_warmup_steps 500 \
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
        --run_name "$run_name" "${arm_args[@]}" > "$run_dir/console.log" 2>&1
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_experimental_stream_annotation=true \
        uv run --no-sync python scripts/eval_fast_rollouts.py \
        --run-dir "$run_dir" --checkpoint weight_ema "${eval_args[@]}" \
        --reference-dir "$reference_dir" --output-dir "$run_dir/eval_weight_ema_standard" \
        > "$run_dir/eval_weight_ema_standard.log" 2>&1
    printf '%s finished %s and weight-EMA rollouts\n' "$(date +%H:%M)" "$arm"
done
