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
The candidate retains the same checkpoint, G0+G1 baselines, cuFFTDx G1 kernel,
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

Each baseline-replacement comparison uses three full batch-one rollouts of
M1, M2, the previous candidate and the shared-baseline candidate, plus one M6
trajectory for accuracy checks. Horizons are T=20 for
Stokes/JONSWAP and T=200 for Tanaka/Benjamin–Feir, with dt=0.01 and four GL2
iterations. Compilation, setup, warm-up and host transfers are excluded.
The checkpoint, learned-branch precision and integration settings are unchanged;
no training is performed.

Replacement measurements are under `outputs/shared_m1_20260926/`:
`ada_full.json`, `blackwell/rollouts.json` and `a100/rollouts.json`. They provide
the retained candidate line and fresh M1/M2 runtime medians. M3–M6 retain the preceding
three-repeat medians from `outputs/fair_classical_20260925/`, including the
M6 runtime reference. Classical error coordinates are unchanged: Ada retains
the earlier 32-case medians and small/full neural points; the Modal panels use
single-case errors against cached M6. The retained and rejected methods are
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
  --candidate-timings outputs/shared_m1_20260926/ada_full.json
```

Reproduce the local full-rollout measurements on a free GPU with:

```sh
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/time_single_rollouts.py \
  --methods M1 M2 M6 fused_cufftdx_spectral_all shared_m1_spectral_all \
  --reference M6 --reference-once --repeats 3 --pad-factor 2 \
  --candidate-run outputs/c27_w320_b4_h80_tanaka_hard128_20260924 \
  --output outputs/shared_m1_20260926/ada_full.json
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
