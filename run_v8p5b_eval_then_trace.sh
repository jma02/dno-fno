#!/usr/bin/env bash
# Parallel eval (g0 on GPU 0, g1 on GPU 1), then NaN trace on whichever GPU.
# Trace targets v8 baseline (known NaN) to characterize the cascade mechanism.

set -u

cd /home/johnma/dno-fno

RUN_DIR_V85B="outputs/cs_dno_w512b8_l256_v8p5b_tiexo_20260622_234309"
RUN_DIR_V8="outputs/cs_dno_w512b8_l256_v8_2gpu_20260619_024621"

echo "[chain] $(date +%H:%M:%S) launching parallel tanaka f64h eval (g0=GPU0, g1=GPU1)"
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
  uv run python -m solver.evals.eval_suite \
    --run_dir "$RUN_DIR_V85B" \
    --regimes tanaka_g0 \
    --f64_harness --gpu \
    --output_dir "$RUN_DIR_V85B/eval_tanaka_f64h" \
  > logs/v8p5b_eval_tanaka_g0.log 2>&1 &
PID_G0=$!

CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda \
  uv run python -m solver.evals.eval_suite \
    --run_dir "$RUN_DIR_V85B" \
    --regimes tanaka_g1 \
    --f64_harness --gpu \
    --output_dir "$RUN_DIR_V85B/eval_tanaka_f64h" \
  > logs/v8p5b_eval_tanaka_g1.log 2>&1 &
PID_G1=$!

echo "[chain] $(date +%H:%M:%S) g0 pid=$PID_G0 (GPU0)  g1 pid=$PID_G1 (GPU1)"

wait $PID_G0
RC_G0=$?
echo "[chain] $(date +%H:%M:%S) g0 exit=$RC_G0"

wait $PID_G1
RC_G1=$?
echo "[chain] $(date +%H:%M:%S) g1 exit=$RC_G1"

echo "[chain] $(date +%H:%M:%S) launching NaN trace on v8 cid=5 (fp64 harness, GPU 1)"
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python -m solver.evals.trace_nan_cascade \
    --run_dir "$RUN_DIR_V8" \
    --npz data/tanaka_2_adaptive_g0.npz \
    --case_id 5 \
    --max_substeps 20000 --every 200 \
  > logs/trace_v8_nan_cascade.log 2>&1
TRACE_RC=$?
echo "[chain] $(date +%H:%M:%S) trace exit=$TRACE_RC"
echo "[chain] done."
