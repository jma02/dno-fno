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
transform produces the four surface features, G0, and all 32 filtered branch
inputs. The FP64 G1 computation is unchanged. G0 is not combined with the
final learned correction, unlike the previously rejected trial.

The existing benchmark script implements these as `surface` and `front`;
the rollout timer exposes `fused_surface` and `fused_front`. Both trainers,
the production model, and all checkpoint weights remain unchanged.

The latest 600-call interleaved, warmed batch-one screen on GPU0 gave:

| Implementation | Median forward latency | Compiler FFT calls |
| --- | ---: | ---: |
| Previous retained G1 batching | 201.67 us | 14 |
| Also batch surface features | 178.50 us | 11 |
| Pack the complete input stage | 162.10 us | 8 |

The input-stage version reduces standalone latency by 19.6%. Both new variants
pass the 48-state check across all four families, with maximum relative
difference 2.806e-8 from the unchanged production predictor. On Stokes,
three alternating full single-rollout repeats give 8.8729→8.2496 seconds
(7.0% less time); all saved eta, xi and Gxi values are bitwise identical to
the previous retained version. Surface batching alone gave 8.8624→8.5302 seconds
in its separate three-repeat trial. Remaining full-family results follow below.

All four full-trajectory comparisons completed:

| Family | Previous | Packed input stage | Runtime reduction |
| --- | ---: | ---: | ---: |
| Stokes | 8.8729 s | 8.2496 s | 7.0% |
| Tanaka | 88.3657 s | 82.1713 s | 7.0% |
| Benjamin–Feir | 88.4216 s | 82.1740 s | 7.1% |
| JONSWAP/TMA | 8.9033 s | 8.2703 s | 7.1% |

Stokes uses the median of three alternating full runs; other families use one
full run per version. All saved eta, xi and Gxi values agree exactly, all
trajectories are finite, and the minimum saved fluid depth is positive
(0.05002645 over these four cases). Figure12 PNG/PDF now use these timings and
annotate the incremental reduction from the previous G1-batched version.

The fresh pre-change 100-call trace contains 40 kernels and 9 device copies per
forward call. Its three dense matrix multiplies account for 6.77 us of traced
GPU work, while FP32 transforms/scaling take 29.24 us and FP64 transforms/scaling
take 42.44 us. This supports grouping transforms before reducing channels again.
Trace timings include profiler overhead; the latency table is uninstrumented.

The final 100-call capture confirms 28 kernels and 4 copies per forward call,
down from 40 and 9. FP32 FFT/scaling work is now 10.81 us; dense multiplies
take 6.91 us. The unchanged FP64 G1 transforms/scaling take 42.51 us and now
account for about half of the traced GPU execution time. Further shrinking
the dense layers is unlikely to remove the largest remaining cost.

Six additional CPU comparisons activate nonzero branch weights in a small
model and vary grids 64/128/256, batches 1/3/2, domain lengths 2pi/4.3,
depths spanning exp(-2) to exp(2), and analytic baselines on/off. The `front`
and previous `batched` implementations agree bitwise in all six cases.

The classical rollout already passes Fourier-space states directly into
`_dno_series_hat`, so it has no analogous pair of physical-input FFTs to remove
at each step. Its independent padded input transforms were already batched in
the preceding classical optimization; the new surface features and learned
branch inputs exist only in the neural model. Figure12 retains that optimized
classical sweep.

Raw artifacts are under `outputs/dno_fusion_20260924/network/`: `batched.nsys-rep`,
`batched.sqlite`, `surface/results.json`, `front/results.json`,
`rollout_stokes.json`, `front_rollout_stokes.json`, and
`front_rollout_remaining.json`. The final trace is
`front_capture.nsys-rep` / `front_capture.sqlite`.

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

## Why the Modal Blackwell run is slower — September 25

The measured penalty has two components: **slower 1024-point FP64 cuFFT
kernels**, especially in the neural rollout adapter, and **higher host
submission/synchronization overhead**. It is not primarily the learned dense
layers or FP32 FFTs. Swapping the two CUDA12 cuFFT versions in both directions
does not remove the difference. The later CUDA13.1 test below also finds no gain
for these workloads.

Three agents split the Ada GPU0 measurements, personal-account Modal Blackwell
measurements, and independent trace/source review. The common diagnostic uses
the same checkpoint and first Stokes state, batch one, N=1024, FP64 integration,
FP32 learned inference, dt=0.01 and four GL2 iterations. It times 16 actual
steps after warming the exact executable, plus standalone and dependent-loop
forward calls. These short diagnostics explain execution costs; they are not
replacement full-trajectory timings for Figure12. No training was run.

Clean neural step time is 3.975 ms on Ada versus 6.160 ms on Blackwell. The
corresponding M2 times are 4.327 versus 5.356 ms. In the GPU traces, the neural
step breaks down as follows (FFT categories include their scaling kernels):

| Traced time per neural step | Ada (ms) | Blackwell (ms) |
| --- | ---: | ---: |
| Ordinary FP64 FFTs/scaling | 2.626 | 4.266 |
| Fused cuFFTDx G1 | 0.267 | 0.309 |
| Remaining GPU work | 0.707 | 0.733 |
| Gaps between GPU operations | 0.495 | 1.058 |
| First-to-last GPU-operation span | 4.095 | 6.366 |

Ordinary FP64 FFTs account for **72.2% of the traced span increase**, gaps for
**24.8%**, and everything else for 3.0%. These shares describe the traces, not
an exact partition of uninstrumented wall time. Trace overhead is similar in
the two 16-step runs (about 3–4%). The operation counts agree between GPUs.

The repeated FP64 complex FFT of length 1024 takes 10.150 us on Ada versus
19.014 us on Blackwell, across 2144 calls in each trace. The real forward and
inverse transforms of the same length similarly rise from 9.324/9.514 us to
17.470/17.929 us. Those three transform types alone explain 66.8% of the traced
increase. FP32 FFT/scaling work is almost unchanged (0.104 versus 0.107 ms per
step); dense work is 0.134 versus 0.139 ms. The neural adapter executes many
more of the affected small FP64 transforms than the classical spectral path,
explaining why the neural rollout suffers the larger cross-machine penalty.

The host-side penalty is independently visible in a one-kernel control: its
GPU execution takes 1.093 us on Ada and 1.073 us on Blackwell, while the
synchronized call takes 62.00 versus 157.06 us. A standalone neural forward
has only 60.91 versus 66.92 us of traced GPU work, despite clean synchronized
latencies of 157.14 versus 275.27 us. This identifies host/driver dispatch and
synchronization as a substantial separate cost; it does not distinguish
gVisor overhead from host CPU or driver effects.

### Controls

- Active SM clocks were stable at 2670 MHz on Ada and 2287 MHz on Blackwell in
  the paired trace. Power was approximately 65–85 W of 300 W and 89–107 W of 600 W,
  respectively. Neither was sitting at its idle clock during measurement.
  Blackwell's lower clock is consistent with the roughly15% G1/2048-point FFT
  penalty, but does not explain the nearly90% 1024-point FFT penalty alone.
- **cuFFT version swap:** on the same Modal GPU/container, 11.4.1.4 versus
  11.3.3.83 gives neural step times 6.08515 versus 6.08441 ms and M2 times 5.24037
  versus 5.23495 ms. The 1024-point FP64 kernel remains 18.505 versus 18.488 us.
  Conversely, loading 11.4.1.4 on Ada leaves that kernel at 10.154 us and the
  neural step at 3.965 ms. Actual loaded library paths were read from the
  benchmark process, not inferred from package metadata. The initial Ada
  `ada_cufft114` attempt still loaded the old library and is not a valid control;
  use `ada_cufft114_loaded`.
- **CUDA graph loop setting:** adding `+WHILE` with loop unrolling disabled
  changes neural step time by less than 0.2% on either GPU. Ordinary FFT thunks
  are not supported by XLA command-buffer conversion, so the setting cannot
  capture a complete FFT-containing loop. Existing traces confirm that cuFFT
  kernels remain outside graphs. This negative result does not rule out
  dispatch overhead. [XLA conversion implementation](https://github.com/openxla/xla/blob/main/xla/backends/gpu/runtime/command_buffer_conversion_pass.cc).
- All diagnostic outputs are finite. The optimized forward differs from the
  production predictor by 1.81e-9 relative on the shared input. No model weights,
  solver equations, precision, production dependencies, or GPU clock settings
  were changed. GPU1 was untouched. Every diagnostic Modal app finished.

The remaining unisolated detail is why the Blackwell/driver execution of these
specific FP64 kernels takes more cycles; this experiment does not attribute it
to a particular GPU instruction or claim that all Blackwell FP64 arithmetic
is slower. The actionable target is the small FP64 transform path and its
repeated launches, not shrinking the learned network further. No new speedup
was accepted, so Figure12 remains unchanged.

Reproduce the common diagnostic with
`CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/tmp/codex-uv-cache uv run --no-sync scripts/profile_dno.py --hardware-diagnosis --repeats 200 --output outputs/hardware_diagnosis_20260925/ada_default`.
Use a fresh output directory for a repeat. Saved `hardware.json` contains clean
timings and operating samples; `hardware_trace.json` contains kernel counts,
durations and gaps, with the raw JAX trace alongside it.

Raw results:

- `outputs/hardware_diagnosis_20260925/{ada_default,ada_while,ada_cufft114_loaded}/`
- `outputs/modal_rtx6000_20260925_hardware/{default,while}/`
- `outputs/modal_rtx6000_20260925_cufft_control/{default,old_cufft}/`
- `outputs/modal_rtx6000_20260925_environment/environment.json`

### CUDA13.1 cuFFT follow-up — September 25

NVIDIA's [CUDA13.1 release notes](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html#cufft-release-13-1)
advertise Blackwell improvements for power-of-two FFT sizes, including FP64.
Tested cuFFT11.4.1.4 versus **12.1.0.31**, in separate processes on the same
Modal RTX PRO6000 and driver580.95.05. Explicit library paths and
`cufftGetVersion` confirm the native versions; JAX process mappings confirm
`libcufft.so.11` versus `libcufft.so.12`. No SONAME substitutions were used.

The native test bypasses JAX, using driver-API allocations/events and CUDA
graphs of256 FFT executions, seven repeats after0.2s warm-up per case. An
old/new/old sequence gives essentially unchanged performance:

| Workload | CUDA12 cuFFT | CUDA13.1 cuFFT |
| --- | ---: | ---: |
| FP64 complex FFT1024, batch1 | 18.589 us | 18.631 us |
| FP64 real forward FFT1024, batch1 | 16.902 us | 16.892 us |
| FP64 real inverse FFT1024, batch1 | 17.312 us | 17.322 us |
| FP64 real inverse FFT2048, batch1 | 21.013 us | 21.024 us |
| Neural GL2 step, warmed16-step diagnostic | 6.1135 ms | 6.1023 ms |
| Classical M2 GL2 step, same diagnostic | 5.2667 ms | 5.2684 ms |

The repeated old-library complex FFT1024 measures18.568 us. Batches2/4 show
the same lack of improvement. All13 native cases pass random-input numerical
checks: maximum relative error3.46e-16 in FP64 and1.83e-7 in the FP32 control.
Zero timing inputs avoid destructive C2R input changes during graph replay.

Integrated tests use separate CUDA12/CUDA13 JAX0.9.2 environments, the same
checkpoint/input, unchanged precision and cuFFTDx G1 kernel. Other CUDA
dependencies necessarily differ in that comparison; the direct cuFFT test
isolates the FFT path. Active clocks stay2340–2347 MHz. Both neural forwards
differ from production by1.81e-9 relative, and all diagnostic outputs are finite.
Traced FP64 FFT/scaling work is likewise unchanged:4.164 versus4.165 ms/step.

**Decision: no speedup accepted.** Do not replace Figure12 with extrapolated
short-step timings or run full trajectories merely to chase a0.18% difference.
Production dependencies, trainers, checkpoints and local GPU jobs are untouched.
The isolated Modal environment is selectable with `DNO_MODAL_CUDA13=1`;
`/opt/cuda13/bin/python` selects its CUDA13 JAX backend. The native benchmark
is `scripts/benchmark_cufft_library.py --library /absolute/libcufft.so --output results.json`.
Raw results: `outputs/modal_rtx6000_20260925_cuda131/` (three native JSONs,
CUDA12/CUDA13 timing reports, HLO and traces). The Modal app completed normally.

### A100 comparison — September 25

Ran the existing native FFT benchmark, shared inference/16-step diagnostic,
and three alternating full Stokes rollouts per method on a Modal
**A100-SXM4-40GB**. The CUDA12 image, JAX0.9.2, cuFFT11.4.1.4 and driver580.95.05
match the Blackwell CUDA12 comparison. Only the cuFFTDx binary is rebuilt for
SM80; weights, precision, equations and benchmark inputs are unchanged.

| Measurement | RTX PRO6000 Blackwell | A100 |
| --- | ---: | ---: |
| Native FP64 complex FFT1024, batch1 | 18.568 us | 5.372 us |
| Native FP64 real forward FFT1024, batch1 | 16.892 us | 3.828 us |
| Native FP64 real inverse FFT1024, batch1 | 17.328 us | 3.892 us |
| Native FP32 real forward FFT1024, batch32 | 2.027 us | 2.984 us |
| Dependent network forward,64-call diagnostic | 96.574 us | 93.566 us |
| Neural GL2 step,16-step diagnostic | 6.1135 ms | 3.0477 ms |
| M2 GL2 step,16-step diagnostic | 5.2667 ms | 1.8947 ms |
| Full Stokes neural rollout, median of3 | 12.6299 s | 5.9798 s |
| Full Stokes M2 rollout, median of3 | 10.7075 s | 3.6477 s |

Full rollouts use the same Stokes case517582, T20,2000 steps, dt0.01,
batch1, four GL2 iterations and warm-up protocol. A100 gives a2.11x neural
speedup and2.94x M2 speedup against the repeated Blackwell Stokes measurements;
M2 is1.64x faster than the neural model on A100. These are measured full
trajectories, not scaled short-step estimates. The standalone synchronized
forward remains host-sensitive (321.01 us on A100 versus275.76 us on Blackwell).

All13 native correctness cases pass. The shared optimized forward equals the
production forward on this A100 input. Full trajectories remain finite with
positive depth. Terminal surface errors against cached M6 are1.07998e-4 for
neural and1.31888e-6 for M2; corresponding Blackwell values are1.08046e-4 and
1.31888e-6. A100's profiler substantially perturbs the short-step run
(traced span6.97 ms/step versus clean3.05 ms/step), so its traced gap shares
must not be interpreted as clean execution overhead.

Raw results: `outputs/modal_a100_20260925/{native.json,hardware/,stokes_rollouts.json}`.
Blackwell full-rollout comparator: `outputs/modal_rtx6000_20260925_repeat_stokes/rollouts.json`.
Reused existing scripts with `DNO_MODAL_GPU=A100` and cuFFTDx `--sm 80`;
no training or production dependency changes. The Modal app completed and
released its GPU. These initial Stokes measurements motivated the complete
four-family comparison below.

### A100 full four-family panel — September 25

Completed three full single-rollout repeats for classical M1–M6 and the
unchanged candidate in every family: 84 timings total. Three personal Modal
A100 jobs ran concurrently on separate GPUs, with one rollout per GPU at a
time. Stokes/JONSWAP use T20 and2000 steps; Tanaka/Benjamin–Feir use T200 and
20000 steps. All retain N1024, dt0.01, four GL2 iterations, FP64 classical
arithmetic/integration, FP32 learned inference, and the shared optimized
baseline implementation. Timings exclude compilation and transfers.

| Family | M1 | M2 | M3 | M4 | M5 | M6 | Candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stokes | 2.0415 | 3.6303 | 5.2251 | 6.9552 | 8.5531 | 10.1694 | 5.9464 |
| Tanaka | 20.6404 | 36.5594 | 53.3955 | 73.6432 | 89.5786 | 104.3034 | 66.0836 |
| Benjamin–Feir | 20.4769 | 36.5521 | 52.3885 | 69.6011 | 85.7606 | 101.8188 | 59.0919 |
| JONSWAP/TMA | 2.0413 | 3.6383 | 5.2347 | 6.9715 | 8.5646 | 10.1669 | 5.9514 |

Entries are median seconds. The candidate's A100 runtime lies between M3 and
M4 in every family. All returned trajectories are finite with positive fluid
depth; M6's maximum relative surface discrepancy against cached M6 is
4.991e-12 across the four families. Figure12 now contains complete four-family
blocks for Blackwell, Ada and A100. Classical errors use cached M6; the
candidate remains a vertical timing line, without an accuracy point.

Raw results: `outputs/modal_a100_20260925_{short,tanaka,bf}_full/rollouts.json`.
The short file contains Stokes and JONSWAP/TMA. All three Modal apps stopped
with zero tasks. No training, local GPU work or production-code changes.

### Spectral adapter and FFT-packing search — September 25

The major remaining cost was not the learned branches: the neural rollout
adapter repeatedly converted fields between physical and Fourier space.
`scripts/benchmark_surrogate_rhs.py` now mirrors the classical spectral RHS,
retaining the same learned predictor, FP64 integration, two-times nonlinear
grid, filtering, Nyquist handling and four GL2 iterations. Its `all` packing
mode batches two inverse FFTs for the network inputs and three inverse FFTs
for the padded nonlinear terms. M2 receives the same nonlinear batching.
The adapter reduces FFT calls from approximately21 to4 per RHS evaluation;
this count excludes FFTs inside the network and step-boundary transforms.

Five-repeat Stokes screens,320 actual steps, median seconds:

| Hardware | Previous M2 | Packed M2 | Spectral + packed neural | Speedup over packed M2 |
| --- | ---: | ---: | ---: | ---: |
| Ada | 1.353607 | 1.214538 | 0.531959 | 2.283x |
| Blackwell | 1.665180 | 1.477257 | 0.732492 | 2.017x |
| A100 | 0.594482 | 0.526033 | 0.488130 | 1.078x |

These short runs select settings, not Figure12 runtimes. The plain spectral
adapter alone nearly tied M2 on A100; batching the remaining FFT calls closed
that gap. Actual-checkpoint RHS checks cover48 states from all four families,
filtered and unfiltered: maximum relative discrepancy2.22e-12. All four16-step
trajectory checks pass, with maximum surface discrepancy2.74e-15. Synthetic
Fourier-edge/Nyquist checks pass. Packing adds no observed discrepancy; the
shared classical packed RHS is bitwise identical on48 states with ramp on/off.

Rejected whole learned-branch sandwich fusion, EPT4/8/16, joint G1 and dense
combinations. Pure branch fusion passes48-state checks (about2.9e-8 maximum
relative forward difference), but adds no material integrated speedup on
any GPU. Fused dense/activation composition fails the1e-5 forward-equivalence
threshold at1.22e-5. Trial wiring was removed; pre-existing CUDA prototype
files were left untouched. No rank reduction, precision change or retraining.

Raw search data: `outputs/knob_search_20260925/{ada_kernels,ada_forward}/`,
`ada_steps.json`, `ada_packing.json`, and
`{blackwell,a100}_{screen,packing}/steps.json` under the same directory.

Full single-rollout validation uses three repeats for each optimized method
and one old-neural control per family:84 full timings across three GPUs.
Entries below are median seconds; all four families retain their complete
T20/200 horizons, N1024, dt0.01 and model G0+G1 terms.

| Hardware | Family | New neural | Packed M2 | Speedup over packed M2 |
| --- | --- | ---: | ---: | ---: |
| Ada | Stokes | 3.3045 | 7.5900 | 2.297x |
| Ada | Tanaka | 32.9617 | 76.0299 | 2.307x |
| Ada | Benjamin–Feir | 32.9994 | 76.0820 | 2.306x |
| Ada | JONSWAP/TMA | 3.3145 | 7.6079 | 2.295x |
| Blackwell | Stokes | 4.6630 | 9.5350 | 2.045x |
| Blackwell | Tanaka | 46.3671 | 95.2873 | 2.055x |
| Blackwell | Benjamin–Feir | 46.3739 | 95.2980 | 2.055x |
| Blackwell | JONSWAP/TMA | 4.6708 | 9.5354 | 2.041x |
| A100 | Stokes | 3.0387 | 3.2732 | 1.077x |
| A100 | Tanaka | 30.0608 | 32.3962 | 1.078x |
| A100 | Benjamin–Feir | 29.7409 | 32.2572 | 1.085x |
| A100 | JONSWAP/TMA | 3.0453 | 3.2358 | 1.063x |

All36 returned method/family records are finite with positive fluid depth.
Long trajectories are not bitwise identical: the maximum relative surface
change against the old neural execution is9.14e-4 on Ada/A100 and3.94e-4 on
Blackwell, both in Benjamin–Feir. Its cached-M6 terminal surface error changes
from0.974247% to1.050564% on Ada/A100 and0.810612% to0.840767% on Blackwell.
Thus these are execution-speed gains with unchanged weights and arithmetic
precision, not a claim of improved learned accuracy.

**Keep** the spectral adapter and all-FFT packing; **reject** the extra
learned-branch/dense variants. The shared nonlinear packing is measured in M2,
not withheld from the comparator. Figure12 now shows these full measured
candidate timing lines and new M2 runtimes; its other classical measurements
and accuracy coordinates are unchanged. The obsolete baseline-free overlay
is removed. No training or production trainer edits; the accepted path is
selected with `fused_cufftdx_spectral_all` and `M2_packed` in the benchmark.

Full results: `outputs/knob_search_20260925/ada_full.json`,
`blackwell_full/rollouts.json` and `a100_full/rollouts.json` under that directory.
All personal Modal apps stopped; local GPU0 finished, and GPU1 was untouched.

## Correcting the classical comparison — September 26

The preceding comparison updated only M2's shared nonlinear FFT packing;
M1 and the other classical orders retained older timings. It also retained
8× classical padding, whereas the candidate's G1 operates on the unpadded
1024-point model grid. Thus the apparent candidate win over M1 did not compare
identical G0+G1 implementations, and the earlier three-GPU speedup numbers
must not be read as comparisons against the reduced-padding classical solver.

### Padding and execution controls

In `_dno_series_hat`, every multiplication is binary and its result is
projected back to base modes before further products. With zero Nyquist mode,
each factor has support through K=N/2−1; its product has support through 2K,
which fits on a 2N grid. This remains true at every recurrence order because
of the intermediate projections. Therefore pad2 suffices for this implemented
recurrence; pad8 is redundant. The standard 3/2 rule also suffices for the
retained modes ([spectral-method notes](https://kth-nek5000.github.io/kthNekBook/_md/spectral/pseudo.html)).
This argument does not apply to evaluating an unprojected high-degree product
in one operation. The separate 2× padding of the nonlinear Zakharov RHS is
unchanged.

CPU checks on 48 saved states across all four families and M1–M6 give a
maximum pad2/pad8 relative difference of 7.74344e-12. Four near-Nyquist stress
cases agree to 1.84e-17; simply removing padding gives up to 6.81% error.
Random binary products agree to 5.43e-16 with 3/2 or 2× padding, while unpadded
products differ by 58.79%. Evidence is in
`outputs/fair_classical_20260925/padding_check.json`.

Every ordinary M1–M6 benchmark now uses the shared packed spectral RHS;
`--pad-factor 2` selects the reduced DNO padding. Packed/unpacked classical
RHS outputs are bitwise equal for all six orders across four family states
and two filter choices. The production solver defaults and trainers remain
unchanged.

The new `baseline-only` control uses the candidate's own normalization,
depth clipping, G0 arithmetic and cuFFTDx G1, with its learned correction
disabled. It is not interchangeable with classical M1, which uses the actual
depth, FP64 inputs and padded products. The control is bitwise identical to
the candidate with zeroed correction parameters on 48 states; compiled HLO
contains no neural GEMMs or surface/head projections. Its teal line and the
full candidate's orange line show runtime only, not accuracy points.

Full measurements use three single-rollout repeats for all eight methods,
N=1024, dt=0.01, four GL2 iterations, and the original T=20/200 horizons.
Compilation, setup, warm-up and transfers are excluded. Raw files are
`outputs/fair_classical_20260925/ada_full.json` and
`{blackwell,a100}_{short,tanaka,bf}/rollouts.json` under that directory.
The Blackwell long groups were restarted before completion with a 3600-second
function limit: the full three-repeat sweeps exceeded the previous 1800-second
limit. No completed result files were deleted.

### Completed full-rollout comparison

All 288 recorded full-rollout timings are complete (three repeats for eight
methods, four families and three GPUs). The 96 returned method/family outputs
are finite with positive depth. The largest M6 surface discrepancy from the
cached pad8 reference is 5.08e-12 relative. Figure12 now uses the new times for
every classical order and both candidate controls.

Median seconds per complete batch-one rollout:

| GPU | Family | Classical M1 | Classical M2 | Candidate | Candidate G0+G1 only |
| --- | --- | ---: | ---: | ---: | ---: |
| Ada | Stokes | 2.2152 | 4.0475 | 3.3119 | 2.7109 |
| Ada | Tanaka | 22.1440 | 40.5050 | 32.9625 | 26.9964 |
| Ada | Benjamin–Feir | 22.1476 | 40.4702 | 32.9832 | 27.0004 |
| Ada | JONSWAP/TMA | 2.2171 | 4.0476 | 3.3175 | 2.7143 |
| Blackwell | Stokes | 2.8577 | 5.1439 | 4.6212 | 3.8389 |
| Blackwell | Tanaka | 28.5624 | 51.4613 | 46.1134 | 38.2823 |
| Blackwell | Benjamin–Feir | 28.7831 | 51.5973 | 46.3711 | 38.4221 |
| Blackwell | JONSWAP/TMA | 2.8527 | 5.1441 | 4.6059 | 3.8176 |
| A100 | Stokes | 1.4878 | 2.2774 | 3.2668 | 2.3944 |
| A100 | Tanaka | 12.9420 | 22.5851 | 30.1196 | 20.0544 |
| A100 | Benjamin–Feir | 13.0310 | 22.7947 | 30.3640 | 20.1040 |
| A100 | JONSWAP/TMA | 1.4838 | 2.3262 | 3.1538 | 2.2390 |

The candidate takes 18.04–18.62% less time than M2 on Ada and 10.13–10.46%
less on Blackwell, but 33.21–43.45% more on A100. This supersedes the earlier
all-three-GPU win: reducing redundant classical padding improves M2 enough
to reverse the A100 comparison. Classical M1 and the candidate's analytic-only
control are faster than the full candidate in every panel.

Some A100 short-family repeats fluctuate: Stokes M6 ranges 6.8390–8.8988 s
(median 7.9417 s), Stokes candidate 3.2455–3.5751 s, and JONSWAP analytic-only
2.0067–2.7375 s. All repeats are retained; no selective reruns or small-difference
speed claims. The long-family measurements are stable and show the same
A100/M2 ordering. No new training or production changes were made.

## Sharing the solver M1 baseline — September 26

The baseline-only candidate is slower than classical M1, motivating a direct
replacement rather than estimating the saving by subtracting separate rollout
times. `shared_m1_spectral_all` uses `_dno_series_hat(order=1, pad_factor=2)`
on the integrator's FP64 spectra, then adds the unchanged learned correction.
Only the correction passes through model normalization and depth conditioning.
The baseline uses actual depth rather than the model's cap, padded products,
and FP64 inputs. This changes the discrete DNO; it is not merely kernel fusion.
The original baseline_order=1 surface features and checkpoint are unchanged.
All changes are confined to benchmark scripts, not training or production code.

The 48-state validation spans four families and depths 0.0126–46.93. Composite
predictions agree with independently evaluated M1 plus correction to 1.56e-13
relative. With zero correction, the RHS (including ramp checks) and short GL2
trajectory reproduce classical M1 bitwise. Saved Gxi is verified to contain
the complete composite, not M1 alone; as before, neural saved Gxi is unfiltered
while classical saved Gxi is filtered, so trajectory comparisons use eta/xi.
Correction extraction changes summation rounding by at most 4.67e-8 relative
to the full DNO. Family-pooled new-versus-old DNO differences range from
1.55e-5 to 1.21e-4. Raw checks: `outputs/shared_m1_20260926/checks.json`.

Full tests compare M1, M2, the previous candidate and the shared-baseline
candidate with three complete batch-one repeats each, plus one M6 reference
trajectory per family. They retain N=1024, dt=0.01, four GL2 iterations and
T=20/200. Raw results are `outputs/shared_m1_20260926/ada_full.json` and
`{a100,blackwell}/rollouts.json` under that directory. No training is performed.

The separate baseline-only rollout is not an additive cost decomposition.
CPU-only lowering of the old and shared spectral RHS finds eight versus ten
live FFT operations and one versus zero cuFFTDx sandwich calls. The front
FP32 inverse-FFT batch shrinks only from37 to36 channels. The shared path
replaces the sandwich with a padded FP64 inverse batch of3×2048 and a forward
batch of2×2048. It retains a separate3×2048 inverse batch for the nonlinear
water-wave RHS; the xi derivative occurs in both batches. No duplicate complete
baseline is present. These are compiler-IR observations, not measured GPU
launch counts or an attribution of elapsed time to individual operations.
The initial sandboxed CPU restore stalled; an isolated unsandboxed CPU retry
completed without GPU use.

After the Ada full sweep finished, a separate fixed-state RHS diagnostic on
GPU0 used300 alternating warmed calls: median240.07→257.82 microseconds
(+7.4%). A separate20-call-per-method trace measured33→38 GPU kernels and
4→5 device copies per call. Old FP64 FFT/scaling plus cuFFTDx work totaled
89.87 microseconds, versus102.79 microseconds of FP64 FFT/scaling in the
replacement. Neural FP32 FFT and GEMM costs were essentially unchanged.
Total traced GPU busy time increased148.94→161.20 microseconds. Thus padded
transforms explain the added device work in this trace; the profile is not
used as a substitute for full-rollout times. Evidence:
`outputs/shared_m1_20260926/profile/results.json` and the adjacent raw trace.

### Full-rollout result: retain the fused baseline

All 156 full-rollout timings are complete. All 60 returned method/family
outputs are finite with positive depth; the largest M6 surface discrepancy
from the cached reference is 5.08e-12 relative. Median batch-one seconds:

| GPU | Family | Fused candidate | Shared solver M1 candidate | Change |
| --- | --- | ---: | ---: | ---: |
| Ada | Stokes | 3.3059 | 3.6567 | +10.61% |
| Ada | Tanaka | 32.9320 | 36.2907 | +10.20% |
| Ada | Benjamin–Feir | 32.9254 | 36.2977 | +10.24% |
| Ada | JONSWAP/TMA | 3.3106 | 3.6630 | +10.64% |
| Blackwell | Stokes | 4.5631 | 5.1688 | +13.27% |
| Blackwell | Tanaka | 45.3951 | 51.0719 | +12.51% |
| Blackwell | Benjamin–Feir | 45.3877 | 51.0657 | +12.51% |
| Blackwell | JONSWAP/TMA | 4.5625 | 5.1592 | +13.08% |
| A100 | Stokes | 3.3665 | 3.9252 | +16.59% |
| A100 | Tanaka | 33.0728 | 38.3421 | +15.93% |
| A100 | Benjamin–Feir | 31.7440 | 37.0417 | +16.69% |
| A100 | JONSWAP/TMA | 3.2021 | 3.7568 | +17.32% |

Reject the direct replacement as a speed optimization: it is slower in all
12 comparisons. Retain the original fused candidate as the default and the
shared variant only as a benchmark option. The earlier saving estimated by
subtracting separate baseline-only runtimes did not hold for the combined
network execution. Compare paired measurements here, not absolute cloud
runtimes across separate allocations.

The single Benjamin–Feir case's terminal surface error improves from
1.0506% to 0.7850% on Ada, 0.8408% to 0.7856% on Blackwell, and 1.0691% to
0.7848% on A100. Other changes are small. These are timing-checkpoint results,
not trained-model aggregate accuracy; Figure12 retains timing-only candidate
lines. No training or production changes were made.

The final Figure12 audit checks all 12 candidate line pairs and slowdown
annotations, all 72 classical coordinates, and unchanged classical error
sources. M1/M2 use this experiment's medians; M3–M6 retain the preceding
three-repeat measurements rather than using this experiment's single M6
accuracy run. All jobs completed; GPU1 was untouched.

## A100 baseline FFT backend — September 26

Compared the existing packed cuFFT `front` implementation against cuFFTDx
inside the current spectral/all-packed adapter. The one-line timer selector
`fused_front_spectral_all` changes only the G1 execution backend. Both retain
the model-local unpadded 1024-point FP64 baseline, depth clipping, normalization,
G0, learned branches and checkpoint. This is not the shared-solver M1 trial.

On a personal Modal A100-SXM4-40GB, 52 full batch-one timings cover three
repeats each of both candidates and M1/M2, plus one M6 reference per family.
N1024, dt0.01, four GL2 iterations and full T20/200 are unchanged; classical
padding is2. Median rollout seconds:

| Family | cuFFTDx G1 | cuFFT G1 | cuFFT slowdown | M2 |
| --- | ---: | ---: | ---: | ---: |
| Stokes | 2.9756 | 3.5626 | 19.73% | 2.2216 |
| Tanaka | 29.5806 | 35.3613 | 19.54% | 22.2025 |
| Benjamin–Feir | 29.5520 | 35.3438 | 19.60% | 22.1498 |
| JONSWAP/TMA | 3.0023 | 3.5643 | 18.72% | 2.2216 |

Both forward variants pass the48-state comparison against the production
predictor with maximum relative difference2.81e-8. All20 rollout outputs are
finite with positive depth; maximum cached-M6 surface discrepancy is5.08e-12.
Long trajectories are not bitwise equivalent: the BF terminal surface error
against M6 is1.0691% with cuFFTDx versus0.9575% with cuFFT. This is one timing
checkpoint/case, not a trained-model aggregate accuracy comparison.

The separate200-repeat model-only screen gives258.35 microseconds for cuFFTDx
and325.37 for cuFFT; those host-synchronized latencies are not used to infer
rollout speed. Raw measurements and forward HLO are under
`outputs/a100_g1_backend_20260926/`. Reject the cuFFT replacement as a speed
optimization; keep cuFFTDx and leave Figure12 unchanged. No training,
production edits, spectral-interface implementation or local GPU use occurred.
The Modal app completed and stopped after returning results.
