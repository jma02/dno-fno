# Dataset coverage behind the September C27 failures

Audited 2026-09-13, 19:16–19:26. CPU only; no model, dataset, generator or
training changes. The original July C27 and balanced September run have the
same main architecture and intended loss recipe. Missing-data causality is
not established until an intervention improves held-out rollouts.

## Main finding

The balanced dataset covers steep solitary waves, but has almost no other
wave shapes at comparable depth and height.

The two failing ICs have depth 0.2804/0.2680 and actual crest-to-trough
height divided by depth 0.4019/0.4375. An exhaustive scan of current TRAIN
rows at depth 0.20–0.35 gives:

| Family | Largest height/depth | Rows with height/depth 0.35–0.50 |
| --- | ---: | ---: |
| Stokes | 0.1061 | 0 |
| Tanaka | 0.4498 | 20,974 |
| Benjamin–Feir | No rows at these depths | 0 |
| JONSWAP/TMA | 0.2055 | 0 |

Of those 20,974 rows, 20,969 are steep single-crest Tanaka and five are main
two-crest Tanaka. They represent 1,316 simulations. Equal global family
proportions therefore did not supply different kinds of large waves here.

This is not simply insufficient steepness or too few Fourier modes *within*
Tanaka. In a 4096-row-per-population audit, the current height-matched Tanaka
median maximum slope was 0.153; the failing waves are 0.154/0.174.
Their low-mode spectra also overlap the new Tanaka distribution.

## What the omitted mixed waves supply

Old mixed wavetrains contain independently chosen low Fourier components,
amplitudes, phases and propagation directions. Their evolved states overlap
the failing depth/height range but have different shapes and eta/xi pairings.

Among 117 sampled old TRAIN packet rows with height/depth 0.35–0.50,
median surface power in modes 5–16 is 7.22%, versus 1.97% among height-matched
current Tanaka. This is a low-mode shape difference, not evidence that adding
arbitrary near-cutoff noise is the remedy.

A separate label-only check asks how much instantaneous surface change cannot
be explained by shifting the surface left or right. Fit the best scalar c in
Gxi ≈ c*eta_x, after removing the mean and modes above 128. The remaining
relative RMS is:

| Dataset subset | Rows | Median unexplained fraction |
| --- | ---: | ---: |
| All current depth/height-matched TRAIN rows | 20,974 | 0.0122% |
| Height-matched old packet sample | 117 | 10.45% |
| Height-matched fine-tune packet sample | 132 | 38.26% |

The packet rows are selected examples, not exhaustive population statistics.
The old/fine-tune packet medians should not be interpreted as a causal
comparison between those splits. This measures instantaneous **surface**
evolution, not physical energy or proof that the entire eta/xi state travels
rigidly.

One old TRAIN packet example has depth 0.2567, height/depth 0.4114 and
maximum slope 0.1770, close to failing 16624's0.2680/0.4375/0.1744.
Its surface-change residual is 92.2%, unlike the nearly translating Tanaka IC.
This is a nearby example, not the nearest wave in full function space.

## Alternatives weakened by the audit

- **Broken new labels:** 128 sampled old/new TRAIN labels all passed the
  retained-mode order-convergence screen. Stored-versus-recomputed order-six
  relative errors have medians about 1.2e-6 for both Tanaka datasets.
  Order 8-to-10 disagreement is at most 3.56e-6. Higher-order corrections beyond
  G0+G1 are present and comparable. No labeling regression found in this sample.
- **All random-sea diversity disappeared:**false. New JONSWAP retains independent
  right/left phases and, in this depth slice, has more intermediate-mode power
  than the old Gaussian seas. Its near-depth exposure is 0.894% of all TRAIN
  rows versus 0.597% for old Gaussian. Its amplitudes here are the limitation.
- **Old Gaussian or single-mode data directly fills the strong-wave gap:**
  not supported. Both old Gaussian and current finite JONSWAP prescribe
  significant heights 0.005–0.03; the shipped old single-mode amplitude cap
  also lies far below the failing height/depth at depth 0.27.
- **No other distribution changes:**also false. Old Gaussian includes early,
  unadjusted evolution; current JONSWAP excludes its 20-period adjustment.
  BF lost independent sideband phases/amplitudes, but neither old nor new BF
  directly covers these depths. These remain secondary hypotheses.

## Next direction

The running packet-only fine-tune is testing the most directly relevant
missing population. First evaluate its unchanged 128-IC FP64, damping-off panel.

If helpful, test a compute-matched ordinary training mixture with the old
mid-depth, larger-amplitude mixed waves added back. Reuse accepted existing
simulations first; do not simply increase steep Tanaka counts or all random-sea
counts. Preserve simulation-disjoint splits and assess other families for
regression. A null five-epoch fine-tune would not rule out an effect from
including these data during training from scratch.

No additional training or restart campaign was launched for this audit.

## Evidence and reproducibility

- [Full experiment record and executed code](../experiments/experiments-2026-09-13.md)
- [Coverage descriptors and provenance](../outputs/c27_balanced_tanaka_tangent_modal_20260913/diagnostics_dataset_coverage_20260913/summary.json)
- [Exhaustive current depth/height slice](../outputs/c27_balanced_tanaka_tangent_modal_20260913/diagnostics_dataset_coverage_20260913/exhaustive_balanced_height_slice.json)
- [Surface-translation fit](../outputs/c27_balanced_tanaka_tangent_modal_20260913/diagnostics_dataset_coverage_20260913/translation_fit_extended.json)
- [Teacher-label audit](../outputs/c27_balanced_tanaka_tangent_modal_20260913/diagnostics_dataset_labels_20260913/summary.json)

Wave arrays were stored float32 and promoted for CPU calculations. Matched
intervals describe depth and height, not equality of complete wave profiles.
Old TRAIN membership follows July's exact seed-0 row split; new datasets use
whole-simulation splits. Correlated snapshots are not independent experiments.
