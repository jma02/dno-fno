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
Blackwell/Modal panels are on top, Ada/local in the middle, and A100/Modal below.
All three hardware sections show the candidate's measured runtimes as orange
vertical lines, without assigning accuracy to the one-epoch timing checkpoint.
The candidate retains the same checkpoint, G0+G1 baselines, cuFFTDx G1 evaluation,
cached depth multipliers and packed network branches. Its spectral adapter
removes repeated physical/Fourier round trips and batches independent inverse
FFTs. Every classical order M1–M6 now receives the shared nonlinear FFT packing.
Classical DNO products use 2× padding instead of 8×: the implementation projects
each binary product back to the base modes before reuse, so 2× is sufficient
for dealiasing at every order. On 48 states across all six orders, the maximum
2×/8× relative difference is 7.75e-12; see
`outputs/fair_classical_20260925/padding_check.json`. This does not change the
separate 2× padding in the nonlinear water-wave RHS.

The orange line shows the retained fused candidate. The rejected shared-M1
candidate line and its slowdown annotations have been removed; its measured
results remain in the experiment records below. No candidate accuracy
coordinate is assigned.

Purple lines show the trained spectral transformer's epoch-4 weight-EMA checkpoint.
Cloud diamonds give that single case's final surface error; Ada shows runtime only
because its error panel uses 32-case medians. The separate full current-test suite
is saved under `outputs/torch_ema_20260926/evaluation/` (128/128 finite).
These timings use BF16/FP32 neural inference, FP64 integration, and a reusable
CUDA graph of the unchanged GL2 step; capture and warm-up are excluded.

All three hardware panels use the same
`fourier_fused_features_overlap_spectral_all_boundaries_stages_carry_local`
candidate, with EPT8 throughout. It retains the previous paired GL2 stages
and G1/learned-correction stream overlap, carries spectral states between
internal timesteps, caches linear propagators at step-relative stage offsets,
and writes surface-network features directly from the field inverse FFT.
Physical states are materialized at saved times. Cached propagators use each
saved interval's actual dt; any ramp still uses absolute physical time.
Spectral carry and local propagators also apply to every classical order M1–M6,
alongside the previous stage pairing and boundary FFT packing. FP64 integration/G1,
FP32 learned inference, weights and iteration count are unchanged.

Annotations compare against `fourier_fused_fields_overlap_spectral_all_boundaries_stages`
in the same three-repeat, four-family full-rollout benchmark. Sources under
`outputs/spectral_carry_20260927/` are `ada_full.json`,
`blackwell_full/rollouts.json`, and `a100_combined_sxm/rollouts.json`.
Classical M1–M6 use fresh full-rollout measurements from `ada_classical.json`,
`blackwell_classical/classical.json`, and `a100_combined_sxm/classical.json`;
M2 uses the three-repeat median from the paired neural comparison, with its
accuracy still taken from the classical M6 reference. Ada paired timings use GPU1;
the all-order sweep uses GPU0. Across the four families, the common candidate
uses 11.2–11.9% less time on Ada, 11.4–12.0% on Blackwell, and 13.2–13.9% on A100
than its paired control. It uses 34.1–35.3%, 29.8–31.1%, and 15.0–18.1% less time
than equally optimized M2, respectively. Most of the gain is in shared integrator
execution; feature fusion adds a further A100 gain and is nearly neutral on Blackwell.

Native feature arrays and the 48-state forward checks are exact, but changing
the integration's rounding path changes long trajectories. For Benjamin–Feir,
the candidate's terminal surface error against M6 changes from 0.839% to 1.022%
on Ada/A100 and from 0.812% to 0.888% on Blackwell. The separate Ada four-way
diagnostic attributes most sensitivity to spectral carry; local propagators
alone give 0.836%. These remain runtime-only candidate lines. Trained-model
accuracy references and EMA overlays are retained; production/training code
is untouched. Trial details, numerical comparisons and decisions are in
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
This update does not regenerate figure 11, the manuscript or the ZIP. Reproduce with:

```sh
UV_CACHE_DIR=/tmp/codex-uv-cache uv run --no-sync paper-plots/plot_neural_advantage.py \
  --stack-hardware \
  --timings outputs/single_rollout_timing_20260923.json \
  --candidate-timings outputs/spectral_carry_20260927/ada_full.json \
  --spectral-timings outputs/torch_ema_20260926/blackwell_timing/timings.json \
    outputs/torch_ema_20260926/a100_80gb_timing/timings.json \
    outputs/torch_ema_20260926/ada_timing/timings.json
```

To time the spectral transformer locally, point `DNO_TORCH_WORKTREE` at its
worktree and select an idle GPU. Use an isolated environment: otherwise the
repository's installed PyTorch can override the requested version.

```sh
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=PHB uv run --isolated --no-project --python 3.11 \
  --with modal --with torch==2.14.0 --with numpy==2.4.2 python -c \
  'from scripts.benchmark_torch_rollouts import main; main(output="outputs/torch_ema_ada", batch_size=1, repeats=3, local=True)'
```

Reproduce the local full-rollout measurements on a free GPU with:

```sh
CUDA_VISIBLE_DEVICES=1 XLA_FLAGS=--xla_gpu_experimental_stream_annotation=true \
  uv run --no-sync scripts/time_single_rollouts.py \
  --methods fourier_fused_fields_overlap_spectral_all_boundaries_stages \
    fourier_fused_fields_overlap_spectral_all_boundaries_stages_carry_local \
    fourier_fused_features_overlap_spectral_all_boundaries_stages_carry_local M2_boundaries_stages_carry_local \
  --reference fourier_fused_fields_overlap_spectral_all_boundaries_stages --repeats 3 --pad-factor 2 \
  --candidate-run outputs/c27_w320_b4_h80_tanaka_hard128_20260924 \
  --output outputs/spectral_carry_20260927/ada_full.json
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

`scripts/modal_benchmark.py` runs explicit commands in the personal account,
uploads only selected code/checkpoint/reference cases, and downloads result
files under a fresh `--output` directory. No manuscript or notes are uploaded.

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
