#!/usr/bin/env bash
# One-epoch scratch capacity screens with the hero recipe (launch_accuracy_screen.sh), on the shared
# read-only dataset, followed by 128-case weight-EMA rollouts against the shared M6 reference.
set -euo pipefail
cd /home/samzheng/dno-fno

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-GPU-c44b86bf-148d-aa7e-32ec-77d4f2f7ce71}
export JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
shared=/home/johnma/dno-fno/outputs/shared_samzheng_20261026
run_date=$(date +%Y%m%d)

if (( $# == 0 )); then
    printf 'Usage: %s control|width512|blocks16 [...]\n' "$0" >&2
    exit 2
fi
for arm in "$@"; do
    case "$arm" in
        control) arm_args=() ;;
        width512) arm_args=(--width 512) ;;
        blocks16) arm_args=(--n_blocks 16) ;;
        *) printf 'Unknown experiment: %s\n' "$arm" >&2; exit 2 ;;
    esac
    run_name=c27_capacity_${arm}_scratch1_${run_date}
    run_dir=outputs/$run_name
    mkdir "$run_dir"
    while :; do
        gpu_pids=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)
        [[ -z "$gpu_pids" ]] && break
        sleep 30
    done
    printf '%s starting %s from scratch\n' "$(date +%H:%M)" "$arm"
    uv run --no-sync python train-jax-10m/1d_dno_fno_jax_ema_grad.py \
        --model cs_dno --norm scale --dataset "$shared/data" \
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
        --run-dir "$run_dir" --checkpoint weight_ema --fuse-nested-fft --batch-context-fft \
        --reference-dir "$shared/reference" --output-dir "$run_dir/eval_weight_ema_standard" \
        > "$run_dir/eval_weight_ema_standard.log" 2>&1
    printf '%s finished %s and weight-EMA rollouts\n' "$(date +%H:%M)" "$arm"
done
