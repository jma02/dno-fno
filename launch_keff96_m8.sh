#!/usr/bin/env bash
# Arm B: Hou-Li gxi filter, k_eff=96 (filter_fraction=0.1875 of k_max=512), m=8 smoothness
# m=8 is sharper than m=4: c(80)~0.997 c(96)=0.5 c(128)~0 — preserves carriers more
# while killing noise floor harder.
set -e
cd "$(dirname "$0")"
mkdir -p logs
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=1 nohup uv run python -m solver.evals.eval_suite \
    --run_dir outputs/cs_dno_w512b8_l256_v8_fftfp64_20260625_012616 \
    --checkpoint best --gpu --f64_harness \
    --filter_gxi --filter_shape houli --houli_a 0.69 --houli_m 8 \
    --filter_fraction 0.1875 \
    --regimes tanaka_g0 tanaka_g1 \
    > logs/eval_v8_fftfp64_keff96_m8.log 2>&1 &
echo "Launched arm B PID $! (k_eff=96 m=8) on GPU 1"
