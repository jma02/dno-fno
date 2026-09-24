# DNO forward-pass profile — September 24, 2026

The main bottleneck is **many small FFTs and their associated overhead**, not
the large surface-network matrix multiply. The slowest individual kernels are
the FP64 FFTs in the analytic `G1` baseline. No model or solver code was changed.
**This report profiles only our neural model's forward pass, not the time
integrator or a classical DNO solver.** `G0 + G1` is part of `CraigSulemDNO.__call__`.

[Roofline and kernel timeline](../outputs/dno_profile_20260924/roofline.png)
([PDF](../outputs/dno_profile_20260924/roofline.pdf)).

## Measurements

GPU 0, RTX 6000 Ada, JAX 0.9.2; batch **one**, 1,024 grid points. Inputs and
weights remain on the device. The 37,632-parameter checkpoint has four branches
in each of eight groups, 160 surface channels and multiplier hidden width 80.
The comparator is the existing 172,608-parameter, 16-branches-per-group model.
Both use their saved checkpoints; this experiment does not compare accuracy
after equal training. GPU 1's existing job was left untouched.

After compilation and 50 warm-up calls, 200 synchronized calls gave:

| Forward pass, depth fixed | Median | Interquartile range |
| --- | ---: | ---: |
| 37,632 parameters | 390.98 µs | 389.87–392.06 µs |
| 172,608 parameters | 416.51 µs | 415.50–417.83 µs |

These are standalone DNO latencies, **not trajectory timings**. The smaller
model saves 6.1% here despite 4.4× fewer compiler-estimated FLOPs. Passing depth as a runtime
input instead gave 465.08 µs for the compact model; this is a different compiled
execution path, not an isolated measurement of the depth MLP's cost.

The fixed-depth wrapper reproduced the existing predictor exactly for the
profile input. All profiled forward outputs were finite.

## Exactly where time goes

Nsight Systems recorded 100 warmed compact-model calls. Each launched **113
CUDA kernels and 25 device-to-device copies**. All 25 copies were correlated
with inverse-FFT scopes. Mean GPU execution time per forward pass:

| Work | Kernels or copies | GPU time | Share of GPU execution |
| --- | ---: | ---: | ---: |
| FP32 FFTs and inverse normalization | 52 | 90.06 µs | 32.9% |
| FP64 FFTs and inverse normalization | 11 | 74.81 µs | 27.3% |
| Other fused/elementwise/reduction kernels | 38 | 50.40 µs | 18.4% |
| Dense matrix multiplies | 12 | 32.58 µs | 11.9% |
| Device-to-device copies | 25 | 25.76 µs | 9.4% |

The 172,608-parameter model also launches **113 kernels, including 38 FFTs,
and 25 copies**. Its combined FFT/scaling/copy time is 191.05 µs versus
190.63 µs for the compact model: reducing branch width barely changes this cost.

Thus FFTs, normalization and their copies account for **69.7% of GPU execution**.
This denominator excludes idle gaps and host work; it is not 69.7% of clean
end-to-end latency. The trace has a median first-to-last GPU-operation span of
514.34 µs, versus 390.98 µs without profiling. Its 240.67 µs median gap time
includes profiler overhead and must not be presented as uninstrumented launch
overhead.

The [FP64 baseline](../models/dno-net/dno_net_v2.py#L114) performs seven FFTs:
four inverse kernels at **10.10 µs each**, and three forward kernels at
**9.90 µs each**. These are the longest individual forward-pass kernels.
Each launches just **one 64-thread block** on a GPU with 142 SMs. The FP32 FFTs
launch 1–4 blocks and take about 2.1–2.2 µs. This directly demonstrates small
grids, not a measured occupancy or bandwidth limit.

In contrast, the 160-to-160 surface mixing matrix multiply takes only
**3.14 µs**. Eight small depth-output GEMMs together take 24.15 µs. Their
compiled names and source scopes are retained in `fixed_capture.hlo` and the
Nsight report. Shrinking channels reduces arithmetic but leaves the repeated
FFT/library-call structure largely intact.

## Roofline: useful estimate, not hardware-counter measurement

Nsight Compute returned `ERR_NVGPUCTRPERM`: this host restricts GPU performance
counters to administrators. No driver permissions or clocks were changed.
The [NVIDIA explanation](https://developer.nvidia.com/ERR_NVGPUCTRPERM) describes
the required administrator action for a future counter-based measurement.

The plotted points use `compiled.cost_analysis()` and clean measured latency:

| Model | Estimated FLOPs | Compiler-counted traffic | Estimated FLOP/byte | Work / measured time |
| --- | ---: | ---: | ---: | ---: |
| 37,632 parameters | 79.67 million | 17.28 MB | 4.61 | 0.204 TFLOP/s |
| 172,608 parameters | 349.31 million | 37.57 MB | 9.30 | 0.839 TFLOP/s |

Compiler-counted traffic is **not measured DRAM traffic**; cache reuse matters
(this GPU has 96 MiB L2), and XLA's operation counts need not equal executed
instructions. Transcendental estimates are recorded separately in the raw JSON.
The curve uses the published 91.1 TFLOP/s CUDA-core FP32 ceiling and about
960 GB/s memory bandwidth, not tensor-core performance. These are reference
ceilings, not measured sustained limits, and the network contains FP64 work.
[NVIDIA specifications](https://www.nvidia.com/en-us/products/workstations/rtx-6000/).

Consequently, the figure does **not** establish DRAM-bandwidth saturation or a
precise percentage of peak efficiency. Together with the tiny grids and the
kernel trace, it supports targeting FFT execution structure and dispatch
overhead before further channel reduction.

## Ranked next changes — proposed, not implemented or timed

1. **Pack the correction groups' transforms.** The eight independent groups
   currently execute 24 FFT calls after the shared input transform. Stack their
   branch channels for one batched inverse FFT and one batched forward FFT;
   sum all weighted spectra before one final inverse FFT. This changes those
   24 calls to three while retaining all branches, weights and normalization.
   It batches branches of one sample, not independent trajectories.
2. **Combine the final `G1` inverse transforms.** Compute
   `irfft(g0_term_hat + dx_term_hat)` instead of two inverse transforms followed
   by addition. Retain FP64; also consider batching the intermediate transforms.
3. **Precompute depth-only multipliers** once per checkpoint/grid/depth for
   inference. The model-only trace shows the depth hidden GEMM on all 100 calls,
   despite identical fixed depth; fixing depth has not eliminated this work.

These preserve the mathematical model in exact arithmetic, so they do not
require retraining. Floating-point equivalence and full-rollout accuracy still
need verification. No speedup for these changes has yet been measured. In
particular, blindly switching `G1` to FP32 is not justified by this profile.

## Reproduce

Use `scripts/profile_dno.py` with `CUDA_VISIBLE_DEVICES=0` and the project
environment. Direct execution measures clean latency; `--depth-mode dynamic`
keeps depth as a runtime argument. Use `--run` and
`--output` for the comparison checkpoint. For a warmed trace:

```sh
CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/tmp/codex-uv-cache nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-event-trace=false \
  --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output outputs/dno_profile_20260924/forward \
  uv run --no-sync scripts/profile_dno.py --capture 100
```

Export each `.nsys-rep` to SQLite with `nsys export --type sqlite`. Repeat with
the 16-branch checkpoint and output directory `branches16` for its model-only
comparison trace. Then
`JAX_PLATFORMS=cpu uv run --no-sync scripts/profile_dno.py --analyze` regenerates
the summary and figure. All raw measurements, HLO and traces are under
`outputs/dno_profile_20260924/`. Profiler versions: Systems 2025.1.3,
Compute 2025.2.1. Scoped Ruff and Pyright checks passed.
