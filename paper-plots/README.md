# Epoch-40 C27 / hard-P128 draft paper figures

## Drop into a LaTeX project

Use `c27-hard128-latex.zip`: it contains only `draft.tex` and `figures/` with the
nine plot PDFs. Extract both into your project's root and select `draft.tex`
as the main document in your editor or Overleaf project. Keep your existing
`PRIMEarxiv.sty` and `references.bib` in that root. The draft also compiles
without those files, using a standard article layout and a bibliography note.

`draft.tex` follows the pasted manuscript, inserts all nine figures, and adapts
the numerical text to the actual preliminary results. Unfinished method
sections and author notes remain marked. `draft.pdf` is a compiled preview.

## Analysis pack

Nine real-data figures corresponding to the manuscript placeholders, exported
as PDF and PNG in `figures/`. Open `preview.pdf` to browse them, or `overview.png`
for a contact sheet. `c27-hard128-plot-pack.zip` contains the complete local pack.

This is a visual first pass, not the final experimental study. No manuscript,
checkpoint, or dataset was uploaded anywhere.

Paths below are relative to the repository root:

- Model run: `outputs/c27_tanaka_hard128_full_equal_local_20260918/`
- Checkpoint: `best_val_ckpt/ckpt_40` within that run.
- Dataset: `outputs/paper_dataset_full_equal_tanaka_hard128_20260918/arrays/`.

The saved checkpoint normalization is used. Learned inference uses FP32;
the analytic physics, time integrator, saved trajectories, and independently
evaluated Hamiltonians use FP64.

Important qualifications:

- No matched vanilla-FNO comparison is included. None is invented or substituted.
- DNO accuracy uses 512 training and 512 test rows per family (4,096 snapshots
  total), sampled from the dataset's saved **simulation-disjoint split**.
  These sampled errors do not cover the full 14,745,600-row dataset.
- Long trajectories reuse the completed current-TEST panels
  `eval_best_current_test_stratified_n32_fp32net_fp64solver_gpu{0,1}` under
  the model run. There are 32 parameter-stratified simulations per family,
  128 total. All references are valid; all learned saved states are finite
  with positive fluid depth. This is not an exhaustive internal-stage audit.
- These rollouts have **no adaptive damping or soliton-selective guard**.
- Their reference uses CS order 6, padding 8, hard cutoff mode 128, GL2 with
  four fixed-point iterations, and dt=0.01. Stokes/JONSWAP-TMA horizons are 20;
  Tanaka/Benjamin-Feir horizons are 200. There are 251 saved states per case:
  output intervals 0.08/0.8, respectively (8/80 internal steps).
- Physical Hamiltonians were independently recomputed with CS-6 on both
  trajectories in float64, at 26 times per case. Median terminal absolute
  C27 drift is about 8.14e-5 versus 7.36e-11 for reference states: **not** at
  the solver floor. The numerical reference itself is not exact conservation.
- Mean linearity residuals are 3.37e-6 to 1.50e-4 in the deployed mixed
  precision, not double-precision roundoff. No negative quadratic form was
  found in the 100 matched-snapshot trials per family; this is not a proof.
- Fresh timings are **CPU only** and do not establish GPU performance.
  The short timing rollout uses
  N=256, T=1, dt=0.01 and cutoff mode 64, separately from the long-time panel.

## Reproduce

Only two scripts and one consolidated `plot-data.npz` are needed. That archive
contains the plotted arrays, row/case identifiers, numerical settings, timings,
and metadata (JSON strings), including 100 structural trials per family.
No full dataset or checkpoint is copied into the ZIP.

From the repository root, render without model/dataset access:

```sh
UV_CACHE_DIR=/tmp/dno-fno-uv-cache uv run --no-sync paper-plots/render_figures.py
```

To regenerate the numerical data, run these stages **sequentially**; each updates
the same archive. The timing run used logical CPUs 8–15. The local sandbox's
async checkpoint loading hangs; use the normal local terminal for these stages.

```sh
UV_CACHE_DIR=/tmp/dno-fno-uv-cache OPENBLAS_NUM_THREADS=1 taskset -c 8-15 uv run --no-sync paper-plots/build_data.py rollouts
UV_CACHE_DIR=/tmp/dno-fno-uv-cache OPENBLAS_NUM_THREADS=1 taskset -c 8-15 uv run --no-sync paper-plots/build_data.py snapshots --samples 512
UV_CACHE_DIR=/tmp/dno-fno-uv-cache OPENBLAS_NUM_THREADS=1 taskset -c 8-15 uv run --no-sync paper-plots/build_data.py benchmarks
UV_CACHE_DIR=/tmp/dno-fno-uv-cache uv run --no-sync paper-plots/render_figures.py
```

Use the individual PDFs with `\includegraphics`; full draft captions follow.

## Figure captions

### 01-wave-regimes

Four training snapshots with a spectral cutoff at mode 128, chosen at the median sampled learned-DNO error within each family. The peak wavenumber is the strongest nonzero surface Fourier mode; a is half the surface range. Snapshot row IDs are in plot-data.npz.

### 02-dno-class-errors

Learned-DNO relative L2 error against the stored DNO targets with a spectral cutoff at mode 128: 512 uniformly sampled rows per family from each of the saved training and test splits. Complete simulations belong to only one split. Points and bars show medians and interquartile ranges. Right panel uses test rows only, grouped by the stated peak-wavenumber depth bins; missing groups contain no sampled rows. These are sampled errors, not exhaustive dataset statistics.

### 03-trajectory-errors

Learned-DNO trajectory errors on 32 parameter-stratified simulations per family from the test split (128 total). The black curve and shading are the pooled median and interquartile range; thin colored curves are family medians. Times are normalized separately by T=20 for Stokes/JONSWAP-TMA and T=200 for Tanaka/Benjamin–Feir. All 128 reference solves are valid and saved predictions are finite with positive fluid depth. Learned inference uses FP32 with FP64 physics and integration, without adaptive damping. This is a selected test panel, not the complete test set. No matched vanilla-FNO baseline is included.

### 04-traveling-wave-accuracy

Phase drift and final unaligned/aligned surface profiles for median-terminal-error Stokes and isolated Tanaka cases in the held-out test panel. Isolated Tanaka cases are identified from their single-wave parameter groups. Translation minimizes the periodic surface L2 discrepancy by Fourier interpolation. For Stokes the shift is unwrapped modulo one carrier wavelength, resolving the equivalent periodic minimizers. Alignment is diagnostic only; the raw errors are shown as well.

### 05-tanaka-collision

Representative two-wave Tanaka interaction from the held-out test panel, h=0.0393919. The displayed event is an early local maximum of crest amplification, not a tracked-crest estimate of collision time. All panels use identical axes and the same fixed spatial recentering around the reference event crest; the learned profiles are not individually aligned. This supplies the draft's interaction illustration without claiming an unverified collision-time statistic.

### 06-spectral-transfer

Quadratic modal-energy evolution in representative Benjamin–Feir and JONSWAP/TMA test cases. Each column shares one color scale across the two methods, normalized by its joint maximum E*. One-sided Fourier energies include conjugate-mode multiplicities. The Benjamin–Feir carrier and strongest symmetric seeded sideband pair are identified from the initial energy spectrum. Sea bands are fixed at n<0.5np, 0.5np<=n<=1.5np and n>1.5np, excluding the zero mode. Solid/dashed band curves denote the Craig–Sulem reference and learned DNO, respectively.

### 07-hamiltonian-drift

Independent physical Hamiltonian check: order-6 Craig–Sulem with padding factor 8 is evaluated on both the reference and learned-DNO predicted states at 26 saved times per case. The pooled panel shows signed median and interquartile range on a symmetric-log scale with linear threshold 1e-8; the four examples use linear signed axes. Saved states and Hamiltonian evaluation use float64. Energy drift is not assumed to equal the CS numerical floor; the underlying arrays are included in plot-data.npz.

### 08-dno-complexity-scaling

CPU-only, batch-one DNO latency on an AMD EPYC 9124 host with an eight-logical-CPU affinity. Learned inference uses FP32; CS and physics use FP64. Seven synchronized measurements follow JIT warm-up at each resolution and order. One held-out Stokes initial condition is Fourier-resampled. CS padding is 8. Fitted log-log exponents over N=128..2048 are empirical finite-range summaries, not asymptotic proofs. The learned DNO does not depend on truncation order M. These CPU measurements do not establish GPU performance.

### 09-runtime-tradeoff

Short-horizon CPU benchmark on one held-out Stokes initial condition, Fourier-resampled to N=256: T=1, dt=0.01, GL2 with four fixed-point iterations, hard cutoff at mode 64, g=1 and padding factor 8. All methods use the same surrogate-callback integration path, including CS actions, and save the same 51 states. Learned inference uses FP32 with FP64 CS and integration. Timings are medians of three warmed synchronized runs. Errors compare surface trajectories with CS-10; the CS-10 zero self-error is omitted from the log error plot. This illustration is distinct from the long-horizon evaluation panel.

## Additional neural-versus-classical comparison (outside the draft bundle)

[10-neural-vs-classical.png](figures/10-neural-vs-classical.png) and its
[vector PDF](figures/10-neural-vs-classical.pdf) show the archived **batch-32**
runtimes against median terminal surface relative-L2 error (%). Lower and left
is better. The small neural model uses 16 branches per group (172,608 parameters);
the full model uses 320 (1,342,400 parameters). All errors use the same 32 test
cases per wave family. M6 is the reference, so its zero error is omitted from
the log axis; a dotted line shows its runtime. Axis ranges differ by family.

The archived neural timer includes compilation and host transfers; the
classical timer does not. Figure 10 therefore does not establish execution-only
speed differences. It is retained separately from the single-rollout benchmark.

The single-rollout benchmark uses the first case in each saved test panel,
with shape `(1, 1024)`, not a batch time divided by 32. Every method runs the full
T=20 (Stokes/JONSWAP-TMA) or T=200 (Tanaka/Benjamin–Feir) trajectory, saving the
same 251 states. It uses the same dt=0.01, four-iteration GL2 integrator, hard
mode-128 cutoff and CS padding 8. Classical arithmetic and integration are FP64;
learned inference is FP32. Each whole rollout is compiled explicitly and warmed
using four output intervals before one synchronized timing measurement. Loading,
compilation, warm-up, transfers and error calculations are outside that timer.
These are individual timing samples, not averages across the full test panel.

Render the simple batch plot, or collect and render the single-rollout version
(`11-single-rollout-warm.png` and `.pdf`) on an idle GPU:

```sh
UV_CACHE_DIR=/tmp/dno-fno-uv-cache uv run --no-sync paper-plots/plot_neural_advantage.py
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/time_single_rollouts.py
UV_CACHE_DIR=/tmp/dno-fno-uv-cache uv run --no-sync paper-plots/plot_neural_advantage.py --timings outputs/single_rollout_timing_20260923.json
```

The second plot combines the new single-case timings with the existing 32-case
error statistics; it does not present one case's error as a family median.
Neither figure changes the manuscript or its ZIP. No data are uploaded.

Completed single-rollout figure: [PNG](figures/11-single-rollout-warm.png) /
[PDF](figures/11-single-rollout-warm.pdf). On the RTX 6000 Ada, the full neural
model takes about 15.2 seconds for T=20 and 150.3 seconds for T=200, versus
48.2 and 480.3 seconds for M6. It beats M3 in both runtime and median final
surface error on Benjamin–Feir; the small model also beats M2 on Benjamin–Feir
and Tanaka. M2 is faster and more accurate than the full neural model on
Stokes and JONSWAP/TMA. These conclusions use one timing sample per method/family.

### 12-potential-new-nn

[PNG](figures/12-potential-new-nn.png) / [PDF](figures/12-potential-new-nn.pdf).
[October 1 meeting notes](meeting-notes-2026-10-01.pdf)
([LaTeX source](meeting-notes-2026-10-01.tex)) summarize the accuracy results and current experiments.
Blackwell/Modal panels are on top, Ada/local in the middle, and A100/Modal below.
The meeting layout places the four families in columns, with logarithmic runtime
axes and identical axis limits within each column.
Repeated savings annotations and redundant transformer timing lines are removed;
transformer diamonds retain both runtime and accuracy. All three hardware rows
show orange runtime/error points for the epoch-120 gradient-EMA checkpoint,
measured with full single rollouts of the optimized network. The separate cloud
timing and saved-accuracy lines have been replaced by jointly measured points.
The candidate retains G0+G1 baselines, cuFFTDx G1 evaluation,
cached depth multipliers and packed network branches. Its spectral adapter
removes repeated physical/Fourier round trips and batches independent inverse
FFTs. Every classical order M1–M6 receives the shared nonlinear FFT packing.
All three GPUs now use the same FP64 cuFFTDx classical implementation:
M1 uses unpadded DNO products with EPT8, M2 uses the specialized recurrence
with unpadded products and EPT16, and M3–M6 use 2× padding with EPT8.
M3–M6 interleave surface-power construction with the recurrence, reducing
M6 FFT dispatches from 22 to 12. Ada's M1–M6 and the missing cloud M3–M5
timings have been refreshed; cloud M1/M2 and M6 retain their preceding measurements.
The unpadded M1/M2 products exploit the existing 128-mode state filter.
Each binary product is projected back to the base modes before reuse;
the earlier 2×/8× check on 48 states across all six orders had maximum
relative difference 7.75e-12; see
`outputs/fair_classical_20260925/padding_check.json`. This does not change the
separate 2× padding in the nonlinear water-wave RHS.

Ada's orange points are labeled `New candidate optimized network` and use the
best epoch-120 checkpoint in `outputs/c27_w320_b4_h80_ema_grad09_20260927/`.
Median rollout times are 1.285576, 12.621518, 12.331878, and 1.254769 seconds
for Stokes, Tanaka, Benjamin–Feir, and JONSWAP/TMA, respectively. For the
Benjamin–Feir case, product fusion saves 0.567% against the preceding fusion
but increases final surface error from 1.448749% to 1.904739%; the plotted point
uses the latter measured error.
Cloud points use October 1 full-rollout measurements of the same trained
checkpoint in `outputs/ema_grad_timing_20261001/{blackwell,a100_sxm}/timings.json`.
Each point pairs the median of three warmed repeats with the final surface error
against cached M6 from that execution, not a saved local accuracy reference.
The old small/full neural references are removed. All panels use single-case errors.

Ada additionally shows pink plus markers labeled `Rollout-supervised network`
for the latest epoch-120 checkpoint in
`outputs/c27_rollout_teacher_m6_ep120_gpu0_20261001/`. Each marker pairs a fresh
full single-rollout timing with the error from that same optimized execution,
against the same cached M6 reference case used by the other methods. These are
not the 32-case median errors reported in the training comparison. Measurements
are saved in `outputs/rollout_teacher_timing_20261002/ada/timings.json`;
the Blackwell and A100 points remain unchanged.

October 1 median warm rollout times (seconds):

| GPU | Stokes | Tanaka | Benjamin–Feir | JONSWAP/TMA |
| --- | ---: | ---: | ---: | ---: |
| A100-SXM4-80GB | 0.934274 | 9.078840 | 9.081056 | 0.933793 |
| RTX PRO 6000 Blackwell | 1.660089 | 16.191948 | 16.179543 | 1.658278 |

Both runs used `jma02`, the existing CUDA12.9/JAX0.9.2 image, EPT8, FP64
integration/G1, FP32 learned inference, and the unchanged full-horizon cases.
All eight trajectories remained finite with positive depth. The Blackwell BF
error is 0.861117%, versus 1.904739% on A100/Ada; each plotted dot uses its own
measured error rather than transferring accuracy across GPUs.

The Ada row replaces the obsolete local-tangent and modal-phase overlays with
`RMS + Fourier context` (blue squares), measured on October 7 for
`outputs/c27_surface_rms_context8_rollout_ep120_gpu0_20261006/`.
These use its latest epoch-120 checkpoint, not best epoch119. The optimized spectral
adapter applies the fixed training-feature RMS scales and the learned width-8
Fourier-context residual; its depth-dependent context multiplier is cached.
The native fused forward passed a 48-state comparison against the ordinary model
(maximum relative difference `7.29e-7`). Three warmed full single-rollout repeats
per family are stored in `outputs/rms_context_timing_20261007/ada/timings.json`.
All four trajectories remain finite with positive depth. Blackwell/A100 measurements
stay unchanged.

| Family | Median runtime (s) | Final surface error (%) |
| --- | ---: | ---: |
| Stokes | 1.479432 | 0.002083 |
| Tanaka | 14.291180 | 3.357832 |
| Benjamin–Feir | 14.360399 | 0.324359 |
| JONSWAP/TMA | 1.460715 | 0.233506 |

Each error belongs to the measured single case, not the 32-case family median.

Purple diamonds show the trained spectral transformer's epoch-4 weight-EMA checkpoint.
Diamonds give that single case's final surface error on each GPU. The separate full current-test suite
is saved under `outputs/torch_ema_20260926/evaluation/` (128/128 finite).
These timings use BF16/FP32 neural inference, FP64 integration, and a reusable
CUDA graph of the unchanged GL2 step; capture and warm-up are excluded.

All three hardware panels use the same
`fourier_fused_features_product_overlap_spectral_all_boundaries_stages_carry_local`
candidate, with EPT8 throughout. It fuses the learned spatial multiplication,
forward FFT and outer filter in one kernel, avoiding the intermediate weighted
branch buffer. The head output already has the required physical layout;
this is not a gain from removing a head transpose. It retains paired GL2 stages
and G1/learned-correction stream overlap, carries spectral states between
internal timesteps, caches linear propagators at step-relative stage offsets,
and writes surface-network features directly from the field inverse FFT.
Physical states are materialized at saved times. Cached propagators use each
saved interval's actual dt; any ramp still uses absolute physical time.
Spectral carry and local propagators also apply to every classical order M1–M6,
alongside the previous stage pairing and boundary FFT packing. FP64 integration/G1,
FP32 learned inference, weights and iteration count are unchanged.

Classical timings come from
`outputs/classical_all_orders_20260928/{ada,blackwell,a100_sxm}/classical.json`.
Ada uses a fresh all-order sweep. Cloud M3–M5 use new `sciml-at-ud` runs;
M1/M2 retain `outputs/classical_figure12_20260927/` measurements, and M6
retains `outputs/classical_interleaved_20260927/` measurements. These cloud
points therefore span multiple allocations. Each M3–M6 optimization comparison
uses paired old/new implementations, with three alternating full-rollout repeats
after full-executable warm-up. Paired M6 savings remain in the source records;
they are not differences between allocations. All errors are referenced to the cached
single-case M6 trajectory. Ada candidate measurements are under
`outputs/ema_grad_timing_20260928/ada/timings.json`; cloud measurements are under
`outputs/ema_grad_timing_20261001/{blackwell,a100_sxm}/timings.json`.
Classical and neural runs retain the same integration settings and packed
boundaries. The fused inference path is unchanged; all three GPUs use the trained
gradient-EMA checkpoint. Spectral EMA measurements are unchanged.
The unused FP32 FFT trial has been removed from Figure 12. Its combined FP32 G1,
reused xi FFT and FP32 input inverse-FFT measurements remain archived in
`outputs/g1_fp32_tuning_20260928/ada/gpu{0,1}.json`:
Stokes 1.0696 s, Tanaka 10.3357 s, Benjamin–Feir 10.0451 s and JONSWAP/TMA 1.0395 s.
Comparisons against the original FP64 control in those same files show
(17.2–18.5% less time); the incremental saving over FP32 G1 alone is 6.3–6.5%.
The trial fails strict 48-state forward equivalence and changes Benjamin–Feir
terminal error from 1.9047% to 0.6746%, with a maximum saved-surface relative
difference of 1.30% against the original path. Existing accuracy points remain
unchanged: this is a timing-only precision trial, not a default precision switch.
Balanced feature stores showed no useful additional gain. Half-spectrum GL2
carry was also tested for both methods and rejected; the selected neural and
classical paths both retain full-spectrum carry.

Product fusion passes the 48-state forward checks with maximum relative
difference 2.01e-9, but long trajectories are not bitwise equivalent. For
Benjamin–Feir in the earlier product-fusion runs, terminal surface error against M6 changes from 1.022% to 1.010%
on Ada/A100 and from 0.888% to 0.976% on Blackwell. The largest relative saved
surface difference against the preceding candidate is 0.001223 (Blackwell BF).
Those preceding timing-only comparisons remain archived; all hardware rows now
show measured candidate points. Spectral EMA overlays are retained;
production/training code is untouched. Trial details, numerical comparisons and decisions are in
`experiments/experiments-2026-09-27.md`.

The preceding FFT round used `fourier_transforms_e16_spectral_all` on Ada and
`fourier_transforms_spectral_all` on Blackwell/A100.
These replace the network's
FP64 G1 input forward FFT and packed FP32 field inverse FFT with cuFFTDx;
the original FP32 input forward FFT, weights and G0/G1 formulation are retained.
Ada used 16 elements per thread; Blackwell/A100 used 8. Annotations compared
three-repeat full-rollout medians against `fourier_input_spectral_all` in the
same run, alongside fresh M2 timings. Measurements are under
`outputs/fourier_ffts_20260926/`: `ada_full.json`,
`blackwell_full/rollouts.json` and `a100_sxm_full/rollouts.json`.
The separate `a100_full/` run used an A100 PCIe and is not mixed into the SXM4
panel. The broader FP32 input-FFT replacement
failed accuracy checks; extra product fusion was not selected. The preceding
input-inverse optimization remains recorded under `outputs/fourier_reuse_20260926/`.
Across the four families, that round reduced rollout time by
7.78–7.85% on Ada, 13.41–13.54% on Blackwell, and 13.77–13.96% on A100 SXM4.
It takes 40.76–41.31%, 39.52–40.18%, and 12.29–14.07% less time than fresh M2,
respectively. These are runtime-only candidate lines; observed rounding-sensitive
trajectory changes and complete numerical comparisons are recorded in
`notes/dno_profile_20260924.md`.

The preceding Fourier-I/O round is saved in
`outputs/fourier_io_20260926/{ada,blackwell,a100}_full/rollouts.json`.
That candidate preserves physical-space FP32 input rounding, returns the
learned correction and G0 directly in Fourier space, and uses a spectral-output
cuFFTDx G1 kernel. Runtime decreases by 18.85–19.24% on Ada, 16.28–16.90% on
Blackwell, and 19.86–20.39% on A100 versus `fused_cufftdx_spectral_all`.
The A100 allocation is the 80GB SXM4 variant; the classical series retains that
run's three repeats for M1 and one M6 reference rollout per family, with M2
refreshed by the latest paired comparison. Unmeasured
M3–M5 were omitted in that round; earlier 40GB A100 timings were not mixed into it.

In that round, Ada and Blackwell M1 medians retained their preceding Fourier-I/O paired runs;
M3–M6 retained the earlier three-repeat measurements on the same hardware.
M2 was remeasured with the old and new candidates on each GPU. Classical math
and the shared nonlinear RHS are unchanged; these replacements concern the
network's unpadded, normalized inputs and learned fields, absent from the
spectral classical recurrence.

Each Fourier-I/O comparison uses three full batch-one rollouts of
M1, M2, the previous candidate and the Fourier-output candidate, plus one M6
trajectory for accuracy checks. Horizons are T=20 for
Stokes/JONSWAP and T=200 for Tanaka/Benjamin–Feir, with dt=0.01 and four GL2
iterations. Compilation, setup, warm-up and host transfers are excluded.
The checkpoint, learned-branch precision and integration settings are unchanged;
no training is performed.

Historical baseline-replacement measurements remain under `outputs/shared_m1_20260926/`:
`ada_full.json`, `blackwell/rollouts.json` and `a100/rollouts.json`; they are no longer plotted.
Those Ada/Blackwell M3–M6 measurements retained the preceding
three-repeat medians from `outputs/fair_classical_20260925/`, including the
M6 runtime reference. Ada/Blackwell classical error coordinates are unchanged: Ada retains
the earlier 32-case medians and small/full neural points; the Modal panels use
single-case errors against cached M6. Blue/purple horizontal lines on the cloud
panels restore the trained small/full models' saved final errors for the same
simulation IDs and horizons. They are accuracy-only references from the existing
Ada evaluations, not new cloud measurements; no cloud runtime is assigned.
The retained and rejected methods are
`fused_cufftdx_spectral_all` and `shared_m1_spectral_all`, respectively.

The replacement changes the discrete baseline rather than merely its execution:
actual depth replaces clipped depth, padded products replace model-grid products,
and the baseline no longer passes through model input normalization. Across48
saved states, independent M1-plus-correction checks agree to1.56e-13 relative,
and zero correction reproduces classical M1. See `checks.json` in the new
measurement directory. Full-rollout accuracy is recorded alongside timing.

The shared-baseline candidate is slower in all twelve comparisons:
10.2–10.6% on Ada, 12.5–13.3% on Blackwell and 15.9–17.3% on A100.
The original fused candidate remains the default; the shared variant is retained
only as a benchmark option. Both are isolated in `scripts/benchmark_surrogate_rhs.py`
and selected by the rollout timer; production trainers and checkpoints are
untouched. Search results, numerical checks and merge decisions are recorded
in `notes/dno_profile_20260924.md` and the September 25–26 logs under `experiments/`.
This update does not regenerate figure 11, the manuscript or the ZIP. After
building cuFFTDx for the selected GPU, reproduce the classical sweep with:

```sh
XLA_FLAGS=--xla_gpu_experimental_stream_annotation=true \
  UV_CACHE_DIR=/tmp/codex-uv-cache uv run --no-sync scripts/benchmark_classical.py \
  --sweep --rollouts --repeats 3 \
  --output outputs/classical_sweep/sweep.json
```

Add `--sweep-orders 3 4 5` for the cloud refresh; Ada uses all six orders.
Reproduce the retained cloud M6 interleaving comparison with:

```sh
XLA_FLAGS=--xla_gpu_experimental_stream_annotation=true \
  UV_CACHE_DIR=/tmp/codex-uv-cache uv run --no-sync scripts/benchmark_classical.py \
  --order 6 --control cufftdx_all_fp64_p2_e8 \
  --variants interleaved_cufftdx_fp64_p2 --epts 8 --rollouts --repeats 3 \
  --output outputs/classical_interleaved/rollouts.json
```

The selected records populate `classical.json`; the preceding `neural.json`
files are unchanged. For the epoch-7 Ada refresh, render after the epoch's
EMA rollouts and `fig12_epoch7_timings.json` are complete. Stokes and Tanaka
reuse the clean timings of the same nested/context-batched inference path;
BF and JONSWAP use fresh full-rollout measurements. The Ada errors come from
epoch 7 at the timed simulation IDs. Blackwell and A100 retain their saved
measurements and accuracies.

```sh
UV_CACHE_DIR=/tmp/codex-uv-cache uv run --no-sync paper-plots/plot_neural_advantage.py \
  --stack-hardware \
  --timings outputs/single_rollout_timing_20260923.json \
  --candidate-timings \
    outputs/c27_phase1_lr8_momentum_half_ema995_grad05_adam05_b2099_batch16_hd04_band128_nested1_gamma32_full_20261012/eval_fast_weight_ema_nestedfused_contextbatch/single_rollout_timings.json \
    outputs/c27_nested_b16_ema_low_lr_ep60_20261014/fig12_epoch7_timings.json \
  --candidate-method fourier_fused_features_product_overlap_nestedfused_contextbatch_spectral_all_boundaries_stages_carry_local \
  --candidate-eval outputs/c27_nested_b16_ema_low_lr_ep60_20261014/eval_epoch7_weight_ema \
  --candidate-label "New candidate optimized network · epoch 7 EMA" \
  --trained-timings outputs/rollout_teacher_timing_20261002/ada/timings.json \
    outputs/rms_context_timing_20261007/ada/timings.json \
  --spectral-timings outputs/torch_ema_20260926/blackwell_timing/timings.json \
    outputs/torch_ema_20260926/a100_80gb_timing/timings.json \
    outputs/torch_ema_20260926/ada_timing/timings.json
```

Figure 12c compares the Ada classical solver, epoch-10 EMA candidate, and older
rollout-supervised network. Epoch 10 is the best overall evaluated checkpoint
through epoch 15 by pooled paired terminal surface error across the 128 cases;
each panel still displays the fixed timed case, not the family median. Runtime
measurements are unchanged for the same inference path. It omits the RMS/context
and spectral-DNO models and historical timing annotations:

```sh
UV_CACHE_DIR=/tmp/dno-uv-cache uv run --no-sync paper-plots/plot_neural_advantage.py \
  --single-case --timings outputs/classical_all_orders_20260928/ada/classical.json \
  --candidate-timings \
    outputs/c27_phase1_lr8_momentum_half_ema995_grad05_adam05_b2099_batch16_hd04_band128_nested1_gamma32_full_20261012/eval_fast_weight_ema_nestedfused_contextbatch/single_rollout_timings.json \
    outputs/c27_nested_b16_ema_low_lr_ep60_20261014/fig12_epoch7_timings.json \
  --candidate-method fourier_fused_features_product_overlap_nestedfused_contextbatch_spectral_all_boundaries_stages_carry_local \
  --candidate-eval outputs/c27_nested_b16_ema_low_lr_ep60_20261014/eval_epoch10_weight_ema \
  --candidate-label "New candidate optimized network · epoch 10 EMA" \
  --trained-timings outputs/rollout_teacher_timing_20261002/ada/timings.json \
  --hide-timing-changes --output-name 12c-ada-runtime-vs-error
```

To time the spectral transformer locally, point `DNO_TORCH_WORKTREE` at its
worktree and select an idle GPU. Use an isolated environment: otherwise the
repository's installed PyTorch can override the requested version.

```sh
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=PHB uv run --isolated --no-project --python 3.11 \
  --with modal --with torch==2.14.0 --with numpy==2.4.2 python -c \
  'from scripts.benchmark_torch_rollouts import main; main(output="outputs/torch_ema_ada", batch_size=1, repeats=3, local=True)'
```

Before rendering the new overlay, check the new architecture and measure full
single rollouts on a free Ada. The first command includes a 48-state comparison
against the ordinary model; the second measures all four families with three
warmed repeats and records accuracy from those same executions:

```sh
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_FLAGS=--xla_gpu_experimental_stream_annotation=true UV_CACHE_DIR=/tmp/dno-uv-cache \
  uv run --no-sync scripts/benchmark_dno_fusion.py \
  --run outputs/c27_surface_rms_context8_rollout_ep120_gpu0_20261006 \
  --checkpoint latest --variants original fourier_fused_features_product_overlap \
  --repeats 3 --ept 8 --output outputs/rms_context_timing_20261007/ada/forward
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_FLAGS=--xla_gpu_experimental_stream_annotation=true UV_CACHE_DIR=/tmp/dno-uv-cache \
  uv run --no-sync scripts/time_single_rollouts.py \
  --candidate-run outputs/c27_surface_rms_context8_rollout_ep120_gpu0_20261006 \
  --checkpoint latest --reference cached_M6 --repeats 3 --pad-factor 2 --ept 8 \
  --methods fourier_fused_features_product_overlap_spectral_all_boundaries_stages_carry_local \
  --output outputs/rms_context_timing_20261007/ada/timings.json
```

Reproduce the preceding rollout-supervised model's measurements on a free GPU with:

```sh
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_FLAGS=--xla_gpu_experimental_stream_annotation=true UV_CACHE_DIR=/tmp/dno-uv-cache \
  uv run --no-sync scripts/time_single_rollouts.py \
  --candidate-run outputs/c27_rollout_teacher_m6_ep120_gpu0_20261001 \
  --checkpoint latest --reference cached_M6 --repeats 3 --pad-factor 2 --ept 8 \
  --methods fourier_fused_features_product_overlap_spectral_all_boundaries_stages_carry_local \
  --output outputs/rollout_teacher_timing_20261002/ada/timings.json
```

The preceding inference-optimization comparison can be reproduced with:

```sh
CUDA_VISIBLE_DEVICES=1 XLA_FLAGS=--xla_gpu_experimental_stream_annotation=true \
  uv run --no-sync scripts/time_single_rollouts.py \
  --methods fourier_fused_features_overlap_spectral_all_boundaries_stages_carry_local \
    fourier_fused_features_product_overlap_spectral_all_boundaries_stages_carry_local M2_boundaries_stages_carry_local \
  --reference fourier_fused_features_overlap_spectral_all_boundaries_stages_carry_local --repeats 3 --pad-factor 2 \
  --candidate-run outputs/c27_w320_b4_h80_tanaka_hard128_20260924 \
  --output outputs/half_spectrum_20260927/ada_full.json
```

The cuFFTDx prototype is inference-only, outside both production trainers.
It uses NVIDIA MathDx25.12.1 CUDA12 (cuFFTDx1.6.1), CUDA12.9 and SM89.
To build locally, unpack the [NVIDIA archive](https://developer.nvidia.com/downloads/compute/cuFFTDx/redist/cuFFTDx/cuda12/nvidia-mathdx-25.12.1-cuda12.tar.gz)
under `outputs/cufftdx_20260925/`, then run:

```sh
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/benchmark_cufftdx.py --build
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/benchmark_cufftdx.py --classical
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/benchmark_dno_fusion.py \
  --variants original front cufftdx --output outputs/cufftdx_20260925/network
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/time_single_rollouts.py \
  --methods fused_front fused_cufftdx --reference fused_front \
  --candidate-run outputs/c27_w320_b4_h80_tanaka_hard128_20260924 \
  --output outputs/cufftdx_20260925/rollouts.json
```

### 12-potential-new-nn-modal

[PNG](figures/12-potential-new-nn-modal.png) / [PDF](figures/12-potential-new-nn-modal.pdf).
Personal `jma02` Modal benchmarks on NVIDIA RTX PRO 6000 Blackwell Server Edition,
CUDA12.9/SM120 and JAX0.9.2. This separate figure contains no local-GPU timings.
It uses one full trajectory per family, N1024, dt0.01 and four GL2 iterations.
Classical errors use the cached M6 reference; the neural candidate is shown
only as a vertical timing line, with no one-epoch accuracy points. Classical M1
means G0+G1. The candidate is the one-epoch checkpoint in
`outputs/c27_w320_b4_h80_tanaka_hard128_20260924`.

Before→cuFFTDx seconds: Stokes13.851→12.630, Tanaka137.045→125.434,
Benjamin–Feir136.781→125.400, JONSWAP13.752→12.587 (8.3–8.8% reductions).
The Stokes neural timing line uses a three-repeat median; other timings use one measurement.
Timings exclude compilation/transfers and follow a separate five-frame warm-up;
the full-executable first use is timed. Stokes repetitions put the classical
first-call overhead near0.5%. Saved Gxi differs in filtering between the two
implementations; the classical accuracy curve uses eta only. All measured trajectories are finite
with positive fluid depth. M2 is faster and more accurate than this candidate
in all four cases; the candidate's Tanaka error is98.866% before and98.866% after cuFFTDx.

Additional joint-G1/product fusions were slower; dense fusion failed the fixed
forward-equivalence tolerance. EPT4 and calibrated low-rank mixing brought no
consistent additional rollout benefit. Rank16 also increased BF surface error
from0.811% to6.534%, so no fine-tuning or production replacement was made.
The classical shared product/rFFT fusion was slower at every order; its faster
existing batched backend is used in the figure. Reproduce the plot with:

```sh
UV_CACHE_DIR=/tmp/codex-uv-cache uv run --no-sync paper-plots/plot_neural_advantage.py \
  --single-case \
  --timings outputs/modal_rtx6000_20260925_classical_short/classical_rollouts.json \
  --classical-timings outputs/modal_rtx6000_20260925_classical_tanaka/classical_rollouts.json \
    outputs/modal_rtx6000_20260925_classical_bf/classical_rollouts.json \
  --candidate-timings outputs/modal_rtx6000_20260925_tuning/rollouts.json \
    outputs/modal_rtx6000_20260925_repeat_stokes/rollouts.json \
  --candidate-method fused_cufftdx --candidate-previous-method fused_front
```

`scripts/modal_benchmark.py` honors the selected `MODAL_PROFILE`,
uploads only selected code/checkpoint/reference cases, and downloads result
files under a fresh `--output` directory. No manuscript or notes are uploaded.

### 12b-one-epoch-full-data

[PNG](figures/12b-one-epoch-full-data.png) / [PDF](figures/12b-one-epoch-full-data.pdf).
The Oct9–10 screens train from scratch for exactly one epoch on the full training
and validation splits, with gradient and weight EMA. Detached rollout supervision
ramps up through the epoch. The comparison includes the control, reflection
augmentation, direct polynomial shortcut, two weaker universal-momentum weights,
global mode-balancing weight2, reflection-even features, and learning rates
1e-5/4e-5. Default learning rate is2e-5 with500-update warmup and one-epoch cosine.
The Oct10 optimizer follow-ups share LR 6e-5, half-strength universal momentum,
and weight EMA 0.995; gradient EMA, Adam beta1, and LR 8e-5 changes are labeled
separately. Adam beta1 remains 0.9 unless stated otherwise. The 20-step teacher
changes only the detached rollout horizon from 0.1 to 0.2, keeping dt=0.01 and
the same generated-state losses. `Combined` denotes LR 4e-5 plus half momentum;
`Parent · seed 1` is the LR 6e-5/weight EMA 0.995 recipe with gradient EMA 0.9.
Runs enter the plot only after all four weight-EMA family evaluations finish.
Raw-checkpoint results are excluded.
Each panel reports the median terminal surface relative L2 error over the
same32 full test trajectories, in percent. The test cases are not subsampled.

The horizontal coordinate is warm batch-one inference latency, not training
duration or the batch32 evaluation wall time. Unchanged inference architectures
share the existing three-repeat Ada measurement in
`outputs/rms_context_timing_20261007/ada/timings.json`:1.4794/14.2912/14.3604/1.4607s
for Stokes/Tanaka/BF/JONSWAP. These are architecture-cost coordinates, not fresh
checkpoint-specific timings; the plot labels this reuse and does not jitter
overlapping points. The polynomial shortcut and even-feature models require
their own `timing/timings.json` before appearing; the shared-GPU training queue
does not generate isolated timings. Other runs also use their own timings when
available. The ordinary-inference32-case errors are separate from
those optimized single-case timing measurements. M2/M3 errors use the same
32-case medians from `outputs/cs_order_sweep_20260923`, with their measured
single-rollout times from `outputs/classical_all_orders_20260928/ada/classical.json`.
All timings use dt.01, four GL2 iterations, FP64 integration, and T20/200;
the learned path is FP32. No forty-epoch subset or120-epoch model accuracies enter
this figure. The old subset40 PNG/PDF remain separate historical artifacts.

```sh
UV_CACHE_DIR=/tmp/dno-uv-cache uv run --no-sync paper-plots/plot_neural_advantage.py --one-epoch
```

### 13-model-roofline

[PNG](figures/13-model-roofline.png) / [PDF](figures/13-model-roofline.pdf).
Native batch-one, 1,024-point forward passes for the Fourier candidate, previous
candidate, and trained small/full models on A100 80 GB, RTX PRO 6000 Blackwell,
and RTX 6000 Ada. Roofline points use measured scalar FP32, scalar FP64, and
TF32 tensor work separately, with published peak reference ceilings; each
precision independently aggregates its participating kernels. Nsight Compute
kernel replay supplies FLOPs, DRAM/L2 traffic, and duration for eight A100/Blackwell
captures. Ada counter access is denied (`ERR_NVGPUCTRPERM`); its measured kernel
profile is shown without substituting estimated roofline points. Lower panels
show separate JAX-traced kernel times; clean forward latencies, replay settings,
and raw per-kernel metrics are in `outputs/model_roofline_20260926/analysis.json`.
Rebuild with `uv run --no-sync scripts/profile_dno.py --analyze --output outputs/model_roofline_20260926`.

### 15–18: Phase and local-potential diagnostics

- [Translation-drift share PNG](figures/15-phase-speed-error.png) / [PDF](figures/15-phase-speed-error.pdf)
- [Fourier-phase PNG](figures/16-fourier-phase-error.png) / [PDF](figures/16-fourier-phase-error.pdf)
- [Local-perturbation PNG](figures/17-local-perturbation-error.png) / [PDF](figures/17-local-perturbation-error.pdf)
- [Perturbation examples PNG](figures/18-local-perturbation-examples.png) / [PDF](figures/18-local-perturbation-examples.pdf)

These show the original epoch-120 EMA model, labeled “Neural network”:
`outputs/c27_w320_b4_h80_ema_grad09_20260927`, using its uncentered
`eval_best_current_test_stratified_n32_fp32net_fp64solver_gpu*` rollouts.
Lines/points are case medians; shading/error bars are interquartile ranges.
M2 is included in the local-perturbation comparison.

**15:** Plot the percentage of squared surface error removed by one rigid
translation: `100*(1 - min_s ||pred(x+s)-ref(x)||^2 / ||pred-ref||^2)`.
Use continuous Fourier alignment, including zero shift as a candidate, on
every fifth saved frame after the initial state. All 32 Stokes cases and the
11 isolated Tanaka cases (groups containing `_m1_`) are included. This
diagnoses accumulated translation drift; it does not divide a speed error
by a surface error. A value of 100% means translation alone removes all
surface error, while 0% means alignment cannot improve it.

**16:** Split squared surface error into Fourier-amplitude and phase parts.
With predicted/reference Fourier coefficients P/R, the phase share is
`100*(1 - sum(w*(abs(P)-abs(R))^2) / sum(w*abs(P-R)^2))`.
All rFFT coefficients are included with Hermitian weights (1 at DC/Nyquist,
2 elsewhere), so the denominator equals physical squared L2 error up to
the common FFT normalization. Equivalently, each mode's phase component is
`2*abs(P)*abs(R)*(1-cos(arg(P*conj(R))))`. Each family has 32 cases.
Both figures compute shares per case before taking median/IQR, use a 0–100%
axis, and omit the zero-error initial frame where the ratio is undefined.
Translation-removable error cannot exceed Fourier-phase error: the former
allows one common shift, the latter independent phase correction per mode.
These are overlapping decompositions, not percentages to add together.

**17:** Reuses `outputs/operator_probe_a100_20261001/operator_probes.npz`.
At fixed surface and depth compare original xi with `xi + 0.1*zeta`, where
zeta preserves xi's retained Fourier amplitudes but randomizes its phases.
Thus `norm(0.1*zeta)/norm(xi) = 0.1`. For each of 32 cases per family,
divide the median DNO relative L2 error over four perturbations by that case's
original error, then compute the family median/IQR of these paired ratios.
The dashed 1x line means unchanged error; 10x means ten times the original
relative error. Each input has its own FP64 M6/padding-8 reference output
and denominator; M2 is included as a baseline. These are error multipliers,
not absolute accuracy rankings: a small original error can yield a large
multiplier. This measures nearby-input sensitivity, not surface-shape
derivatives or error in the incremental response. Fully scrambled
potentials are excluded.

**18:** The actual first perturbation direction from the audit, reconstructed
with its saved seed and original full-array random-draw order. Show the first
case of each family (simulation IDs 517582, 3686647, 3714872, 3735800), with
original and perturbed potential above, and the added change below. Each
column uses the same vertical scale in both rows, in units of the original
potential's RMS. The added change is not magnified; its RMS is exactly 10%
of the original. Surface eta and depth h are unchanged. “Local” here means
nearby in function space, not a spatially confined bump. These are four
actual examples from the probes summarized in Figure 17, not new experiments.

All four figures are CPU-only reductions of existing saved results:

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 UV_CACHE_DIR=/tmp/dno-uv-cache \
  uv run --no-sync paper-plots/plot_phase_diagnostics.py
```

### 19–20b: Hero-run phase drift

- [Translation drift PNG](figures/19-hero-translation-drift.png) / [PDF](figures/19-hero-translation-drift.pdf)
- [Fourier phase PNG](figures/20-hero-fourier-phase.png) / [PDF](figures/20-hero-fourier-phase.pdf)
- [Fourier phase share PNG](figures/20b-hero-fourier-phase-share.png) / [PDF](figures/20b-hero-fourier-phase-share.pdf)

Compare primary weight-EMA rollouts at epochs 1, 6 and 7 of the current nested
candidate/low-LR continuation. Lines are case medians; shading is the
interquartile range. Figures 20 and 20b shade only epoch 7; epoch 1 is dashed.
No models, solvers or GPUs are invoked.

**19:** The top row shows absolute continuous translation displacement as a
percentage of domain length. The bottom row shows the percentage of squared
surface error removed by that alignment. Include all 32 Stokes cases and the
11 isolated Tanaka cases; evaluate every tenth saved frame after time zero.
Stokes displacements are unwrapped modulo the initial dominant carrier
wavelength to resolve equivalent periodic alignments. These are accumulated
displacements, not instantaneous speed errors.

**20:** Include all 32 cases per family and every saved nonzero time, in four
larger panels. The y axis, “Relative surface error from phase mismatch (%)”,
measures `100*sqrt(phase_energy/reference_energy)` on a logarithmic scale.
For example, 0.01% means the phase-related surface mismatch is 0.01% of the
reference wave's L2 size, not 0.01% of the total prediction error.

**20b:** Plot `100*phase_energy/total_error_energy`, the phase share of squared
surface error, separately on linear 0–100% axes for the same cases and times.
Use the same Hermitian-weighted Parseval decomposition as Figure 16.
Translation-removable and Fourier-phase shares overlap; they must not be
added. A larger share alone does not mean a larger absolute error.

The current hero recipe has a direct local translation-tangent term only on
Tanaka (weight 10). All-family momentum-production matching (weight
0.002574349830754761) provides indirect phase control; mode balancing (weight
6) penalizes complex coefficient errors but does not isolate phase. Explicit
Stokes-carrier, BF-sideband and coupled-modal phase penalties are inactive.
Rollout-generated states receive the same data/MB/TG/momentum objective,
with unchanged family masks and no backpropagation through trajectories.
Hadamard self-consistency remains active on its scheduled training subset,
not as an additional rollout-endpoint calculation.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/dno-uv-cache \
  uv run --no-sync paper-plots/plot_phase_diagnostics.py --hero
```

### 21: Actual phase drift in one wave

[PNG](figures/21-phase-drift-example.png) / [PDF](figures/21-phase-drift-example.pdf)

Epoch-7 EMA Stokes case 1901833, the case nearest the 32-case median terminal
Fourier-phase error. At time 20, the first two panels show the actual reference,
prediction and translation-aligned prediction, then a tight zoom around a
reference zero crossing. Fourier interpolation resolves the sub-grid offset;
the prediction's displacement is not artificially enlarged. The local crossing
offset is 7.21735e-7 spatial units, versus a globally fitted shift of 7.09653e-7.

The third panel shows pointwise error before and after alignment; alignment
removes 99.8039% of this case's squared surface error. No Fourier-angle panel
is included.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/dno-uv-cache \
  uv run --no-sync paper-plots/plot_phase_diagnostics.py --phase-example
```

### 22: Phase error across training

[PNG](figures/22-phase-error-vs-training.png) / [PDF](figures/22-phase-error-vs-training.pdf)

Terminal Fourier-phase error for primary weight-EMA checkpoints at every epoch
from 1 through 15, using the same 32 cases per family. The y axis uses Figure 20's
relative-error magnitude, not the percentage of total squared error. Each point
is the median of `100*sqrt(phase_energy/reference_energy)` at the final rollout
time. At epoch 15, the reductions versus epoch 1 are 54.3%, 64.4%, 2.8% and
67.7% for Stokes, Tanaka, Benjamin–Feir and JONSWAP/TMA. All fifteen measurements
are shown; the curves are not monotonic. From epoch 14 to 15, the Stokes, Tanaka
and JONSWAP/TMA medians decrease by 17.0%, 0.9% and 7.7%, while Benjamin–Feir
increases by 29.0%. Epoch-15 median errors are 0.000798205%, 0.0113509%,
0.0522900% and 0.00701770% in the family order above. These are family medians;
epoch 10 remains best by pooled paired terminal surface error across all cases.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/dno-uv-cache \
  uv run --no-sync paper-plots/plot_phase_diagnostics.py --phase-training
```
