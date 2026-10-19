#!/usr/bin/env bash
set -euo pipefail
cd /home/johnma/dno-fno

modal_pid=${1:?Modal client PID required}
queue_pid=${2:?local queue PID required}
run_dir=outputs/c27_phase_direction_seed1_a100_scratch1_20261018
reference_dir=outputs/c27_phase1_lr8_momentum_half_ema995_grad05_adam05_b2099_batch16_hd04_band128_nested1_gamma32_seed1_full_20261012/eval_fast_weight_ema
export CUDA_VISIBLE_DEVICES=GPU-c44b86bf-148d-aa7e-32ec-77d4f2f7ce71
export JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1
export UV_CACHE_DIR=/tmp/dno-uv-cache MPLCONFIGDIR=/tmp/dno-matplotlib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_experimental_stream_annotation=true

printf '%s waiting for Modal training and local rollout queue\n' "$(date +%H:%M)"
while kill -0 "$modal_pid" 2>/dev/null; do sleep 30; done
rg -q 'training finished in [0-9]+s, returncode=0' outputs/modal_phase_seed1_20261018.log
for artifact in config.json summary.json train_log.jsonl weight_ema_ckpt; do
    MODAL_PROFILE=jma02 /home/johnma/.local/bin/modal volume get dno-fno-train-data \
        "/$run_dir/$artifact" "$run_dir/" --env main
done
while kill -0 "$queue_pid" 2>/dev/null; do sleep 30; done
while :; do
    gpu_pids=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)
    [[ -z "$gpu_pids" ]] && break
    sleep 30
done
printf '%s starting Modal seed1 primary-EMA rollouts on GPU0\n' "$(date +%H:%M)"
uv run --no-sync python scripts/eval_fast_rollouts.py \
    --run-dir "$run_dir" --checkpoint weight_ema --reference-dir "$reference_dir" \
    --output-dir "$run_dir/eval_fast_weight_ema" > "$run_dir/eval_fast_weight_ema.log" 2>&1
printf '%s finished Modal seed1 primary-EMA rollouts\n' "$(date +%H:%M)"
