#!/usr/bin/env bash
# Arm A: Hou-Li gxi filter, k_eff=96 (filter_fraction=0.1875 of k_max=512), m=4 smoothness
# Geometry that prior tests missed: above all carrier harmonics (g0 6th=84, g1 6th=66)
# but below the noise floor band peak (~k=128).
set -e
cd "$(dirname "$0")"
mkdir -p logs
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 nohup uv run python -m solver.evals.eval_suite \
    --run_dir outputs/cs_dno_w512b8_l256_v8_fftfp64_20260625_012616 \
    --checkpoint best --gpu --f64_harness \
    --filter_gxi --filter_shape houli --houli_a 0.69 --houli_m 4 \
    --filter_fraction 0.1875 \
    --output_dir outputs/cs_dno_w512b8_l256_v8_fftfp64_20260625_012616/eval_keff96_m4_f64h \
    --regimes tanaka_g0 tanaka_g1 \
    > logs/eval_v8_fftfp64_keff96_m4.log 2>&1 &
echo "Launched arm A PID $! (k_eff=96 m=4) on GPU 0"
