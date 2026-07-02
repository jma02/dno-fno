#!/usr/bin/env bash
# Eval v8 baseline with Hou-Li smooth filter at k_eff=64 (filter_fraction=0.125).
# Applies to BOTH state filter (eta_t/xi_t every substep) AND gxi filter (--filter_gxi).
# tanaka_g0 on GPU 0, tanaka_g1 on GPU 1, in parallel.

set -u
cd /home/johnma/dno-fno

RUN_DIR="outputs/cs_dno_w512b8_l256_v8_2gpu_20260619_024621"
OUT_DIR="$RUN_DIR/eval_tanaka_houli_keff64"
HOULI_A=0.69
HOULI_M=4
FF=0.125

mkdir -p logs

echo "[chain] $(date +%H:%M:%S) launching Hou-Li keff=64 tanaka_g0+g1 in parallel"
echo "[chain] filter_fraction=$FF (k_eff=64), state+gxi Hou-Li (a=$HOULI_A m=$HOULI_M)"

CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
  uv run python -m solver.evals.eval_suite \
    --run_dir "$RUN_DIR" \
    --regimes tanaka_g0 \
    --f64_harness --gpu \
    --filter_gxi --filter_shape houli --houli_a $HOULI_A --houli_m $HOULI_M \
    --filter_fraction $FF \
    --output_dir "$OUT_DIR" \
  > logs/houli_keff64_g0.log 2>&1 &
PID_G0=$!

CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda \
  uv run python -m solver.evals.eval_suite \
    --run_dir "$RUN_DIR" \
    --regimes tanaka_g1 \
    --f64_harness --gpu \
    --filter_gxi --filter_shape houli --houli_a $HOULI_A --houli_m $HOULI_M \
    --filter_fraction $FF \
    --output_dir "$OUT_DIR" \
  > logs/houli_keff64_g1.log 2>&1 &
PID_G1=$!

echo "[chain] $(date +%H:%M:%S) g0 pid=$PID_G0 (GPU0)  g1 pid=$PID_G1 (GPU1)"

wait $PID_G0
RC_G0=$?
echo "[chain] $(date +%H:%M:%S) g0 exit=$RC_G0"

wait $PID_G1
RC_G1=$?
echo "[chain] $(date +%H:%M:%S) g1 exit=$RC_G1"

echo "[chain] done. results -> $OUT_DIR"
grep -E "NaN=|η_med_tf=|η_p95_tf=" logs/houli_keff64_g0.log logs/houli_keff64_g1.log || true
