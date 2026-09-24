# DNO forward-pass profile — September 24, 2026

The main bottleneck is **many small FFTs and their associated overhead**, not
the large surface-network matrix multiply. The slowest individual kernels are
the FP64 FFTs in the analytic `G1` baseline. No model or solver code was changed.
**The initial profile covers only our neural model's forward pass, not the time
integrator or a classical DNO solver.** `G0 + G1` is part of `CraigSulemDNO.__call__`.
A separate full-rollout follow-up appears near the end of this report.

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

## Ranked changes from the initial profile

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
need verification. These were hypotheses at the time of the initial profile;
the following prototype now measures them. Blindly switching `G1` to FP32 is
not justified by this profile.

## Measured inference prototype

[Timing figure](../outputs/dno_fusion_20260924/forward_fusion.png)
([PDF](../outputs/dno_fusion_20260924/forward_fusion.pdf)); prototype:
`scripts/benchmark_dno_fusion.py`. The production model remains unchanged.

The first rearrangement is valid because these groups are **parallel and
summed**, not sequential layers. With group index `g` and branch index `r`, set

\[
z_{gr}=m_{gr}\,\mathcal F_x
\left[a_{gr}\,\mathcal F_x^{-1}(m_{gr}\widehat\xi)\right].
\]

The existing correction is
`sum_g irfft(sum_r z_gr)`. It equals `irfft(sum_(g,r) z_gr)` in exact arithmetic.
Stacking all 32 branch channels lets the earlier transforms run together along
**the spatial axis only**. Every branch retains its own learned multiplier
and spatial coefficient, and the original square-root normalization is kept.
This does not average branches or batch different trajectories.

The new comparison interleaved 600 synchronized, warmed batch-one calls per
variant, alternating execution order. Compilation and depth-cache construction
were excluded from latency. All variants use the same 37,632 trained weights,
the same input, and the original FP64 `G1` precision; no retraining occurred.
These are within-run comparisons, not comparisons against the earlier 391 µs
measurement from a separate run.

| Change | Median latency | Interquartile range | Speedup vs original |
| --- | ---: | ---: | ---: |
| Original | 426.11 µs | 421.91–431.05 µs | 1.00× |
| Combine final `G1` inverse transforms only | 411.17 µs | 407.78–416.56 µs | 1.04× |
| Pack branch transforms only | 262.13 µs | 260.72–265.84 µs | 1.63× |
| Cache depth multipliers only | 427.65 µs | 424.20–432.36 µs | 1.00× |
| All three | 234.68 µs | 232.46–239.22 µs | 1.82× |

Packing is the largest demonstrated benefit. Caching alone gave no measurable
improvement; the combined result does not establish a separate caching speedup
because compiler interactions make these changes non-additive.

Across 48 saved reference states (four trajectories and three times in each of
Stokes, Tanaka, JONSWAP/TMA and Benjamin–Feir), the combined variant's maximum
relative difference from the unchanged predictor was **2.806e-8**, and maximum
absolute difference was **7.451e-9**. This is an output-equivalence check, not a
new DNO-accuracy evaluation against the reference solver. Validation used a
48-state batch; **all latency measurements used batch one**. That forward-only
experiment did not test trajectory equivalence; see the separate follow-up below.

The compiler FFT counts are 38 original, 37 with `G1` combination alone,
17 with packing alone and 16 with all three. A separate 100-call Nsight capture
of the combined prototype measured **43 kernels and 10 device copies per call**,
versus 113 kernels and 25 copies in the original model-only profile. This
confirms a reduction in launches, not merely a FLOP estimate.

Cached multipliers must be rebuilt when checkpoint weights, depth or grid
change. They are inference constants, not values to reuse through training
updates. The prototype intercepts the existing Flax computation at tracing
time; unused original branch FFTs are eliminated by XLA. No production source
or checkpoint was replaced. Raw timings, HLO and trace are in
`outputs/dno_fusion_20260924/`.

## Follow-up: full single-trajectory timing

At the user's subsequent request, the same 37,632-parameter checkpoint was
tested **inside the unchanged integrator**. One Stokes initial condition
(simulation 517582), Nx=1024, T=20, dt=0.01, 2,000 internal steps, GL2 with four
fixed-point iterations, padding factor 8, hard-filter fraction 0.25. GPU0, batch one;
three synchronized full-length repeats per version in alternating order.
Compilation, cache setup, warm-up and host transfers were excluded.

| Version | Median full-rollout time | Three runs |
| --- | ---: | --- |
| Unchanged model | 12.5265 s | 12.6359, 12.5265, 12.3727 s |
| All three inference optimizations | 9.4842 s | 9.5380, 9.4842, 9.3468 s |

This is **1.32x end-to-end speedup, or 24.3% less time**, not the 1.82x
standalone-forward speedup. Full-shape compilation took 3.0347/1.9221 s and
solver/cache setup took 0.3014/0.5330 s, respectively; these are not complete
cold-process timings. Both trajectories were finite and retained positive
saved fluid depth. Against the unchanged model, maximum saved-frame relative
differences were 3.560e-7 for eta, 1.565e-7 for xi, and 4.037e-6 for Gxi;
terminal eta difference was 2.850e-7. These measure implementation equivalence,
not accuracy against M6. The additional T200 tests below were completed later;
production inference remains unchanged.

Raw results: `outputs/dno_fusion_20260924/rollout_timing.json`. Reproduce with:

```sh
CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/tmp/codex-uv-cache \
  uv run --no-sync scripts/time_single_rollouts.py --families stokes --repeats 3 \
  --candidate-run outputs/c27_w320_b4_h80_tanaka_hard128_20260924 \
  --methods compact fused --reference compact \
  --output outputs/dno_fusion_20260924/rollout_timing.json
```

### Remaining full-model families

Using the same grid, time step, checkpoint and batch-one protocol, three full
repeats per version gave:

| Family | Horizon | Before FFT changes | After FFT changes | Maximum saved-frame relative eta difference |
| --- | ---: | ---: | ---: | ---: |
| JONSWAP/TMA | 20 | 12.3799 s | 9.3523 s | 9.050e-7 |
| Tanaka | 200 | 123.0641 s | 93.2098 s | 3.300e-5 |
| Benjamin–Feir | 200 | 123.7326 s | 93.8028 s | 3.934e-4 |

All trajectories were finite with positive saved fluid depth. Differences
grow over the long horizon: Benjamin–Feir's terminal surface difference is
0.0393%, and its maximum relative Gxi difference is 1.184e-3. These are not
bitwise-identical rollouts or a replacement for accuracy testing against M6.
The model weights were unchanged; "after" means packed FFTs, the combined G1
inverse transform, and cached fixed-depth multipliers, not further training.

The initial Benjamin–Feir attempt overlapped a newly started GPU0 training
job; its 149.09/169.79 s fused timings were rejected for performance comparison.
At the user's relaunch request, only our benchmark was stopped. The table
uses the fresh isolated-GPU0 rerun; the unrelated GPU1 job was left untouched.
Raw results are `rollout_remaining_families.json` (JONSWAP/TMA and Tanaka) and
`rollout_benjamin_feir_retiming.json`, both under
`outputs/dno_fusion_20260924/`. Figure12 uses these plus the earlier Stokes file;
the orange lines indicate runtime only, not an unevaluated 32-case error median.

### Full rollouts without model G0+G1

The same checkpoint and integrator were timed after omitting only the model's
analytic baselines before compilation. Three synchronized full repeats per
family on GPU0 gave these medians:

| Family | With model G0+G1 | Without model G0+G1 | Baseline-free saved states finite |
| --- | ---: | ---: | --- |
| Stokes | 9.4842 s | 7.7456 s | Yes |
| JONSWAP/TMA | 9.3523 s | 7.6911 s | No |
| Tanaka | 93.2098 s | 77.0396 s | No |
| Benjamin–Feir | 93.8028 s | 77.1251 s | No |

All requested steps executed, including after non-finite states appeared.
This isolates runtime cost, not a usable water-wave model: the integrator's
own linear-flow/G0 operations remain unchanged. A captured-branch reconstruction
verified the nonzero learned correction was retained (relative difference
1.245e-6 for the packed variant); its compiled forward has nine FFTs and no
baseline FFT scopes. Production inference and weights are unchanged.

Raw results: `outputs/dno_fusion_20260924/rollout_no_baselines.json`. Reproduce
using the single-rollout command above with all four families,
`--methods fused --reference fused --without-baselines`. Figure12 includes both
runtime guides for every family; numerical outcomes are recorded here and in
the raw results, with no explanatory callouts or footer on the figure.

## Analytic FFT follow-up

The retained change batches the two FP64 inverse transforms inside G1 and
the two transforms of their products with eta. G0 stays in its original
precision. This reduces the previous fused model from16 to14 compiler FFTs.
Two600-call interleaved screens on idle GPU0 measured224.18→201.09 us and
227.05→204.09 us. Full batch-one rollouts gave:

| Family | Previous fused | Batched G1 | Reduction |
| --- | ---: | ---: | ---: |
| Stokes | 9.3404 s | 8.8495 s | 5.26% |
| Tanaka | 93.1609 s | 88.3122 s | 5.20% |
| Benjamin–Feir | 93.1297 s | 88.2849 s | 5.20% |
| JONSWAP/TMA | 9.3929 s | 8.8986 s | 5.26% |

Stokes uses three repeats; the other rows use one full trajectory per version.
All saved eta, xi and Gxi arrays were bitwise identical to the previous fused
implementation. These are the same checkpoint and full horizons as above.
Raw reports are `baseline_joint/rollout_stokes.json` and
`baseline_joint/rollout_remaining.json` under `outputs/dno_fusion_20260924/`.
Use `--methods fused fused_g1 --reference fused` in the single-rollout timer.

Two more aggressive candidates were rejected. Reusing the FP64 G0 evaluation
inside G1 in place of the original FP32 G0 changed the48-state forward result
by up to1.74e-4 relative. Merging FP32 G0 with the learned correction's final
inverse transform passed that screen (1.63e-7 maximum relative difference),
but its T200 Benjamin–Feir surface differed by0.1954% from the previous
implementation, for only about1% additional rollout speed. Figure12 uses
neither rejected candidate. Production model weights and both trainers are
unchanged. Rejected trial implementations are retained in git history only.

### Applying the same optimization to the classical solver

The classical recurrence already cached the depth symbol and accumulated
G0 through GM spectrally before the final inverse transform. Its remaining
independent transforms are now batched too: one padded inverse transform
for eta, dx(xi) and G0(xi), and one batch of product transforms per recurrence
order. Padding8, precision, truncation orders and the time integrator stay
unchanged.

CPU comparisons against the pre-change implementation passed84 combinations
of orders0–6, grids64/128, padding1/2/8 and batched/unbatched inputs (maximum
relative difference7.06e-17). On GPU0, all48 saved states produced bitwise
identical outputs for every order1–6. In300 interleaved warmed batch-one
calls, M1 went369.20→204.09 us and M6 went2385.26→1245.55 us; compiler FFT
counts fell8→5 and48→25. Raw data: `baseline_joint/classical_screen.json`.
The independent G0+G1 identity test, both dealiasing tests, and23 integrator/
evaluator tests pass. Full trajectories are measured separately in
`baseline_joint/classical_rollouts.json`; Figure12 uses those runtimes.

The full single-rollout sweep completed for all24 family/order combinations:

| Family | M1 | M2 | M3 | M4 | M5 | M6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stokes | 4.30 s | 8.52 s | 12.69 s | 16.88 s | 21.09 s | 25.27 s |
| Tanaka | 42.79 s | 84.70 s | 126.22 s | 167.91 s | 209.71 s | 251.71 s |
| Benjamin–Feir | 42.78 s | 84.70 s | 126.21 s | 167.89 s | 209.71 s | 251.79 s |
| JONSWAP/TMA | 4.30 s | 8.51 s | 12.68 s | 16.87 s | 21.07 s | 25.29 s |

Each entry is one warmed full trajectory on uncontended GPU0, with the same
case and numerical settings used by the prior single-rollout benchmark.
All saved states are finite with positive fluid depth. The largest M6
relative difference from the cached reference is4.99e-12 for eta and
9.94e-12 for Gxi, both on Tanaka. Figure12 retains the prior32-case error
statistics and original small/full neural points, but replaces every
classical runtime with this sweep and adds the candidate's measured5.2–5.3%
incremental runtime savings. The PDF and PNG are regenerated; neither the
manuscript nor the ZIP is changed.

## Network input-transform packing

The next network-only change keeps every feature and learned weight, but
groups independent transforms earlier in the forward pass. First, the four
surface-feature inverse FFTs (first, second and half derivatives, plus the
Hilbert transform) are batched. Then one paired forward transform computes
the normalized surface and physical Dirichlet spectra, and one batched inverse
transform produces the four surface features, G0, and all32 filtered branch
inputs. The FP64 G1 computation is unchanged. G0 is not combined with the
final learned correction, unlike the previously rejected trial.

The existing benchmark script implements these as `surface` and `front`;
the rollout timer exposes `fused_surface` and `fused_front`. Both trainers,
the production model, and all checkpoint weights remain unchanged.

The latest600-call interleaved, warmed batch-one screen on GPU0 gave:

| Implementation | Median forward latency | Compiler FFT calls |
| --- | ---: | ---: |
| Previous retained G1 batching | 201.67 us | 14 |
| Also batch surface features | 178.50 us | 11 |
| Pack the complete input stage | 162.10 us | 8 |

The input-stage version reduces standalone latency by19.6%. Both new variants
pass the48-state check across all four families, with maximum relative
difference2.806e-8 from the unchanged production predictor. On Stokes,
three alternating full single-rollout repeats give8.8729→8.2496 seconds
(7.0% less time); all saved eta, xi and Gxi values are bitwise identical to
the previous retained version. Surface batching alone gave8.8624→8.5302 seconds
in its separate three-repeat trial. Remaining full-family results follow below.

The fresh pre-change100-call trace contains40 kernels and9 device copies per
forward call. Its three dense matrix multiplies account for6.77 us of traced
GPU work, while FP32 transforms/scaling take29.24 us and FP64 transforms/scaling
take42.44 us. This supports grouping transforms before reducing channels again.
Trace timings include profiler overhead; the latency table is uninstrumented.

The classical rollout already passes Fourier-space states directly into
`_dno_series_hat`, so it has no analogous pair of physical-input FFTs to remove
at each step. Its independent padded input transforms were already batched in
the preceding classical optimization; the new surface features and learned
branch inputs exist only in the neural model. Figure12 retains that optimized
classical sweep.

Raw artifacts are under `outputs/dno_fusion_20260924/network/`: `batched.nsys-rep`,
`batched.sqlite`, `surface/results.json`, `front/results.json`,
`rollout_stokes.json`, `front_rollout_stokes.json`, and
`front_rollout_remaining.json`.

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
