# C27 / v9 draft paper figures

Nine real-data figures corresponding to the manuscript placeholders, exported
as PDF and PNG in `figures/`. Open `preview.pdf` to browse them, or `overview.png`
for a contact sheet. `c27-v9-plot-pack.zip` contains the complete local pack.

This is a visual first pass, not the final experimental study. No manuscript,
checkpoint, or dataset was uploaded anywhere.

Paths below are relative to the repository root:

- Model run: `outputs/c27_h1_to_l2_full_20260717_212550/`
- Checkpoint: `outputs/c27_h1_to_l2_full_20260717_212550/final_ckpt/` (epoch 40).
- Dataset: `data/combined_dataset_v9.npz`.

The saved checkpoint normalization is used; fresh evaluation uses float64.

Important qualifications:

- No matched vanilla FNO trained on v9 was found. None is invented or substituted.
- DNO accuracy uses 512 training and 512 held-out rows per main family (4,096
  snapshots total), reconstructing the July seed-0 80/10/10 **row split**.
  This is not a trajectory-disjoint test. The extra linear/manufactured v9
  source families are not included in the four-family plots.
- Long trajectories reuse the real July C27 final-checkpoint **guarded**
  evaluation panels: `eval_final_guarded_non_tanaka_suite_n32_20260719_221813`
  and `eval_final_soliton_spectral_guard_20260719_191228` under the model run.
  These are historical source-family panels, not a newly generated held-out v9
  test set. There are 288 attempted cases; 285 have valid CS references.
  Finite output and positive depth were checked at the saved times only.
- Their reference uses CS order 6, padding 8, hard cutoff mode 128, GL2 with
  four fixed-point iterations, and dt=0.01. Stokes/sea horizons are 20;
  Tanaka/BF-inspired horizons are 200. Stored arrays are float32.
- Physical Hamiltonians were independently recomputed with CS-6 on both
  trajectories in float64, at 26 times per case. Median terminal absolute
  C27 drift is about 3.60e-4 versus 2.09e-9 for reference states: **not** at
  the solver floor. The numerical reference itself is not exact conservation.
- Fresh timings are **CPU only**, not historical GPU timings. C27 is slower
  than the measured CS orders on this CPU. The short timing rollout uses
  N=256, T=1, dt=0.01 and cutoff mode 64, separately from the long-time panel.
- The v9 names “random sea” and “BF-inspired” are retained; these are not
  silently relabeled as the newer JONSWAP/TMA or canonical BF datasets.

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

Four actual v9 training snapshots, chosen at the median sampled C27 error within each family. The peak wavenumber is the strongest nonzero surface Fourier mode; a is half the surface range. Snapshot row IDs are in plot-data.npz. The historical random-sea and BF-inspired generators are not relabeled as the newer JONSWAP/TMA or canonical Benjamin–Feir datasets.

### 02-dno-class-errors

Measured C27 relative L2 error against the stored v9 DNO targets: 512 uniformly sampled rows per family per split, with the exact July seed-0 80/10/10 row split reconstructed. Points and bars show medians and interquartile ranges. Right panel uses held-out rows only, grouped by the stated peak-wavenumber depth bins; missing groups contain no sampled rows. This historical snapshot split does not establish trajectory-disjoint generalization. Auxiliary v9 source families are outside these four panels.

### 03-trajectory-errors

C27 trajectory errors over 285 historical reference-valid cases: 63 Stokes, 64 Tanaka, 64 random-sea and 94 BF-inspired cases. The black curve and shading are the case-pooled median and interquartile range; thin colored curves are family medians. Times are normalized separately by T=20 for Stokes/sea and T=200 for Tanaka/BF-inspired. Three of 288 reference solves were invalid and are explicitly excluded from accuracy statistics. Saved predictions in all 285 paired cases are finite with positive fluid depth; this is a saved-state check, not a new exhaustive failure audit. No matched vanilla-FNO baseline is available.

### 04-traveling-wave-accuracy

Phase drift and final unaligned/aligned surface profiles for median-terminal-error historical Stokes and single-crest Tanaka cases. Translation minimizes the periodic surface L2 discrepancy by Fourier interpolation. For Stokes the shift is unwrapped modulo one carrier wavelength, resolving the equivalent periodic minimizers. Alignment is diagnostic only; the raw errors are shown as well. Representatives were chosen within their profile class rather than to minimize model error.

### 05-tanaka-collision

Representative two-crest Tanaka interaction, historical tanaka_g0 case 30, h=0.125923. The displayed event is an early local maximum of crest amplification, not a tracked-crest estimate of collision time. All panels use identical axes and the same fixed spatial recentering around the reference event crest; the learned profiles are not individually aligned. This supplies the draft's interaction illustration without claiming an unverified collision-time statistic.

### 06-spectral-transfer

Quadratic modal-energy evolution in representative historical BF-inspired and random-sea cases. Each column shares one color scale across the two methods, normalized by its joint maximum E*. One-sided Fourier energies include conjugate-mode multiplicities. The BF-inspired carrier and strongest symmetric seeded sideband pair are identified from the initial energy spectrum. Sea bands are fixed at n<0.5np, 0.5np<=n<=1.5np and n>1.5np, excluding the zero mode. Solid/dashed band curves denote CS/C27. These are diagnostics of the old generators, not evidence for all canonical Benjamin–Feir or JONSWAP/TMA regimes.

### 07-hamiltonian-drift

Independent physical Hamiltonian check: order-6 Craig–Sulem with padding factor 8 is freshly evaluated on both the reference and C27 predicted states at 26 saved times per case. It is not the archived learned-DNO energy diagnostic. The pooled panel shows signed median and interquartile range on a symmetric-log scale with linear threshold 1e-8; the four examples use linear signed axes. Saved states have float32 storage, evaluated here in float64. Energy drift is not assumed to equal the CS numerical floor; the underlying arrays are included in plot-data.npz.

### 08-dno-complexity-scaling

Fresh CPU-only, float64, batch-one DNO latency on an AMD EPYC 9124 host with an eight-logical-CPU affinity. Seven synchronized measurements follow JIT warm-up at each resolution and order. The same historical Stokes initial condition is Fourier-resampled. CS padding is 8. Fitted log-log exponents over N=128..2048 are empirical finite-range summaries, not asymptotic proofs. C27 has no M input, but is slower than the measured CS orders on this CPU; no GPU speedup is claimed.

### 09-runtime-tradeoff

Fresh short-horizon CPU benchmark on one historical Stokes initial condition, Fourier-resampled to N=256: T=1, dt=0.01, GL2 with four fixed-point iterations, hard cutoff at mode 64, g=1 and padding factor 8. All methods use the same surrogate-callback integration path, including CS actions, and save the same 51 states. Timings are medians of three warmed synchronized runs. Errors compare surface trajectories with CS-10; the CS-10 zero self-error is omitted from the log error plot. This illustration is distinct from the long-horizon historical evaluation panel.
