# Mid-k stability training recommendation

Date: 2026-07-01

## Problem

The Tanaka NaNs have two stages:

1. The learned closed-loop dynamics slowly amplify nonphysical content in
   approximately `k=64:128`.
2. Once that state is sufficiently out of distribution, the GL2 stage
   fixed-point map becomes non-contractive and one substep blows up.

Static output/state filtering is not an acceptable model fix because physical
Tanaka harmonics occupy the same band. Mixed-precision FFTs also did not reduce
the aggregate NaN rate. The existing Jacobian regularizer is directionally
useful but incomplete: it only penalizes the high-to-high block of
`dG_theta/deta`, not the tangent gain of the complete `(eta, xi)` rollout map.

## Ideal objective

Let `S_theta` be one `dt=0.01` surrogate integration substep and `S_ref` the
reference substep. For a soft band projector `P_B`, initially covering
`32 <= |k| <= 160`, penalize randomized tangent mismatch:

```text
L_tangent = E_z,v [
    ||P_B (J S_theta(z) - J S_ref(z)) v||^2
    / (||P_B J S_ref(z) v||^2 + eps)
]
```

Perturb both `eta` and `xi`, and sample both low-to-mid and mid-to-mid
directions. A one-sided gain hinge can additionally penalize model gain above
the reference gain plus a small margin. Matching the reference is preferable
to blindly forcing contraction because the physical flow may legitimately
have neutral or transiently amplifying directions.

## Cost

The naive implementation is too expensive for full training:

- Four-iteration GL2 evaluates the model about ten times per substep.
- Differentiating through 4–8 substeps would require roughly 40–80 model
  evaluations per example before accounting for JVP/backprop overhead.
- Keeping every unrolled state for reverse-mode differentiation would also
  substantially increase accelerator memory.

Do not run this objective on every full-size training batch.

## Cost-controlled implementation

1. **Offline hard-negative data (recommended first).**
   Add band-limited perturbations to clean training states at amplitudes
   spanning the observed precursor range, roughly `1e-6` to `1e-3` relative
   state norm. Recompute reference `G(eta)xi` labels, or preferably one
   reference substep, offline. Sample these rows as 5–10% of fine-tuning
   batches and add an explicit `k=32:160` error term. This uses ordinary
   supervised training and should add approximately 5–15% wall time after the
   data are generated.

2. **Sparse closed-loop tangent penalty.**
   Evaluate one substep and one randomized JVP on a small microbatch every
   16–32 optimizer steps. This verifies the actual failure mechanism without
   paying its cost on every example. Measure the real wall-time multiplier in
   a short A/B run; target no more than 1.5x.

3. **Only then consider short unrolling.**
   If hard negatives plus sparse tangent matching reduce one-step band gain but
   do not fix rollout alarms, use 2–4 substeps on the same sparse microbatch.
   Do not begin with an 8-step full-batch unroll.

The existing output-only Jacobian penalty can remain as a cheap auxiliary, but
it should be expanded to include `xi`, low-to-mid coupling, and a finite upper
band instead of treating every mode above one cutoff identically.

## Data-coverage audit

The data hypothesis is credible, but perturbation fine-tuning is not an
untried strategy and should not displace the tangent objective as the main
next mechanism:

- v8 contains 2,000,000 Tanaka rows, but only 8.13% have `h >= 0.23`.
  Sampling 20,000 of those rows found 14.77% with proxy `a/h >= 0.27`,
  giving approximately 1.20% joint coverage, or about 24,000 rows. This is
  only 0.32% of the complete 7.43M-row v8 corpus.
- v8's added `shallow_steep_wide` family does not fill the spectral hole.
  Its deep, high-amplitude states can have large physical slopes, but their
  `k=64:128` content remains near zero.
- For physical deep-and-steep Tanaka training rows, sampled `k=64:128`
  eta-band RMS has median `3.56e-8`, p99 `2.31e-7`, and maximum `3.87e-6`.
  On v8 g0 cid5, the predicted value reaches `3.93e-7` at t=128,
  `2.58e-6` at t=132.8, and `8.10e-5` at t=136. Thus the clean initial state
  is represented, but the precursor moves through the extreme tail and then
  about 20x beyond the sampled physical envelope.
- `steep_tanaka_v2` and `build_combined_v9.py` already target
  `h=0.20:0.35, a/h=0.25:0.45`, but both planned shards currently contain
  zero generated samples.

Prior experiments show a stability/fidelity tradeoff rather than a clean data
win:

- `tanaka_perturbed_v1` fine-tuning still produced `1/16` g0 NaNs and worsened
  median error and energy drift.
- `tanaka_perturbed_v2` eliminated g0 NaNs/divergences (`0/16`), but worsened
  median final eta error from `0.0420` to `0.0761` and median energy drift from
  `0.00174` to `0.00537`. Only g0 was evaluated.
- Generic model-drift fine-tuning retained the same `2/16` g0 NaNs while
  worsening median error from `0.0118` to `0.0313` and p95 from `0.278` to
  `0.425`.
- One-step pushforward was tried on FNO/SpectralDNO, not CS-DNO. It improved
  several rollout regimes but substantially regressed random-sea accuracy and
  increased some NaN counts.

The exact narrow recipe proposed below was not tested: historical perturbations
covered every mode 1–128 at relative amplitude `1e-3:3e-2`, much stronger and
broader than the observed ignition, and were fine-tuned from older v3/v5
checkpoints rather than v8.5b.

If a final cheap data ablation is desired, test a small targeted pack rather
than the existing broad perturbation recipe:

- clean states concentrated at `h=0.23:0.30, a/h=0.27:0.35`;
- paired perturbations restricted primarily to `k=64:128`;
- perturbation RMS log-uniform from `1e-6` to `1e-3` of state amplitude,
  which spans the measured precursor without starting at already-corrupted
  amplitudes;
- reference labels recomputed on the perturbed states with the existing
  order-6/pad-8 tail-rejection check;
- mix only 5–10% into a short v8.5b fine-tune.

Do not launch another large data-only fine-tune. Run this as a short controlled
A/B beside the sparse one-substep tangent regularizer. The tangent objective is
the chief untried mechanism because it directly constrains amplification,
whereas the completed data experiments mostly traded NaNs for worse fidelity.
If the narrow data arm removes spectral-growth alarms without degrading clean
controls, it can replace the more expensive regularizer.

## Validation

Use a matched f64 harness and require all of the following:

- zero NaNs on the current 96-trajectory diagnostic set;
- zero alarms from the truth-free `k=64:128` growth detector;
- model/reference one-substep tangent-gain agreement in the failure band;
- GL2 fourth-iteration residual remains below tolerance throughout rollout;
- no regression in median and p95 Tanaka error, especially shallow cases with
  legitimate broadband harmonics.

GL2 residual detection plus substep rejection/retry should be implemented
separately as runtime containment. It prevents a non-contractive solve from
producing NaNs, but it does not remove the learned mid-k instability.
