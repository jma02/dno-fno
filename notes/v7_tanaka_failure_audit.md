# v5 Tanaka Failure Audit

Audit of the four failing Tanaka cases under `eval_suite_f64h` for the v5 CS-DNO
(`outputs/cs_dno_w512b8_l256_v5_2gpu_20260615_141717/`). All figures and the
machine-readable per-case summary live under `notes/figures/tanaka_audit/`.

The eval harness is f64 in the time integrator + spectral ops, with the model
itself applied in f32 (and gxi cast back to f64 for the implicit step). Outer
`dt=0.8`, `substeps=80`, `gl2_if`, `M=6`, `filter_fraction=0.25` (`k_cut=128`).

| label                | h      | a      | a/h    | ka    | kh    | NaN / div t | max rel_l2 | max truth drift |
|----------------------|--------|--------|--------|-------|-------|-------------|------------|-----------------|
| tanaka_g0_cid5       | 0.276  | 0.0915 | 0.331  | 0.274 | 0.83  | NaN @ 41.6  | 14.63      | 1.6e-7          |
| tanaka_g0_cid11      | 0.234  | 0.0640 | 0.274  | 0.207 | 0.76  | NaN @ 91.2  | 0.567      | 2.6e-7          |
| tanaka_g1_cid1000006 | 0.293  | 0.0893 | 0.305  | 0.243 | 0.80  | NaN @ 96.8  | 0.244      | 1.6e-7          |
| tanaka_g1_cid1000011 | 0.011  | 0.0022 | 0.194  | 0.112 | 0.58  | div @ 121.6 | 1.436      | 4.1e-7          |

`a`, `ka` and `kh` are estimated from the IC's half-width
(`k_eff = pi / half_width`). Specs and full per-case stats are in
`notes/figures/tanaka_audit/audit_summary.json`.

## 1. Failing Tanaka ICs - are they reasonable?

### Reference: JCP09 (Xu & Guyenne, 2009)

JCP09 sec 4.2.2 runs a single progressive Tanaka soliton at h=1 on a fixed
[0,82]x[0,1.6] domain:

- `a=0.3, h=1` (a/h=0.3) with `M=4, dt=0.01`, no filter, no instability over t in [0,1000].
- `a=0.6, h=1` (a/h=0.6) with `M=6, dt=0.01` plus the ideal `m=0.9` filter,
  also stable up to t=1000. Energy drift O(1e-4).
- The paper does not explicitly show a failure threshold, but the literature
  Tanaka stability boundary sits at a/h around 0.83 (max breaking height);
  a/h=0.6 is in the "highly nonlinear but resolvable" regime.

Therefore the *physics* (Tanaka stability envelope alone) gives plenty of room
for any of our four ICs.

### Per-IC verdict

See `notes/figures/tanaka_audit/ic_profiles.png` for IC `eta(x)` overlaid with a
KdV-soliton at matched a/h (qualitative reference; KdV underestimates the
Tanaka peak/narrowness at large a/h or small h).

- **cid5** (h=0.276, a/h=0.331): a/h well below JCP09's stable a/h=0.6. The
  shape matches a slightly-steeper-than-KdV sech^2. The catch is the *periodic
  fit*: at h=0.276 the soliton's 1% support is roughly `9.5*h ~ 2.6`, which is
  on order half the periodic domain. The visible eta wraps and leaves a
  residual ~0.04 bump at the opposite end (vs ~0 for cid11 / cid1000006). The
  generator's 3-copy periodic extension is exact only when
  `9.5*h <<  L=2*pi`; at h>=0.23 this assumption breaks (the dataset comment
  in `solver/gen_data/generate_tanaka_dataset_v2.py:338` literally says
  `depth_max=0.08` is the safe choice but the dataset uses depth_max=0.30).
  This IC is *physically OK* but *numerically marginal for a periodic FFT
  domain at L=2pi*.
- **cid11** (h=0.234, a/h=0.274): visibly a clean sech^2 well inside the
  domain; periodic wrap is essentially zero. Comfortably within JCP09's
  envelope. The IC itself is well-posed.
- **cid1000006** (h=0.293, a/h=0.305): same geometry concern as cid5 - 
  visible wrap ~0.02 at the far edge. a/h still below JCP09.
- **cid1000011** (h=0.011, a/h=0.194): in absolute units the soliton is
  amplitude 0.0022 and width ~14 grid cells. The KdV reference is much
  wider than this IC, confirming the exact Tanaka soliton at h=0.011 is
  steeper than KdV predicts. The IC is well-resolved (the truth integrator
  handles it cleanly, see below) but is sitting near the f32 dynamic-range
  floor when fed into the model.

### Truth-validity gate (see `notes/figures/tanaka_audit/energy_drift.png`)

`max |truth energy drift|` for all four cases is **between 1.6e-7 and 4.1e-7**,
three orders of magnitude below the eval gate of 1e-3. **The truth is pristine
for all four ICs**. These failures are **model-side**, not truth-limited,
ruling out the "saved-MATLAB shadowing" / truth-blowup hypotheses that have
shown up in previous family failures (`project_truth_validity_audit`).

The eval_suite `truth_invalid_ics` list is empty for both `tanaka_g0` and
`tanaka_g1`.

### Coverage check (training set distribution)

The training set `combined_dataset_v5.npz` contains 2M Tanaka rows
(1M g0 + 1M g1). Per-IC distribution of `(h, a/h)`:

- Tanaka g0/g1 depths sampled `log_uniform(0.01, 0.30)`. Median h = 0.05-0.06,
  99% percentile h = 0.29. So **the failing deep cases (h ~ 0.23-0.29) sit at
  the top 8% of training depths**.
- Among Tanaka ICs, only **1.0% have both h>=0.23 AND a/h>=0.27** (the
  cid5/cid11/cid1000006 regime).
- For cid1000011 (h<0.02, a/h>=0.18): 7.3% of training Tanaka ICs cover this
  regime, but at the extreme small-amplitude end where f32 noise becomes
  comparable to signal.

The deep+steep failure regime IS thinly covered; the very-shallow-amplitude
regime is better covered numerically but is f32-precision-limited at runtime.

## 2. Spectral error attribution

See `notes/figures/tanaka_audit/spec_heatmap_<label>.png` for log10 |hat_err|
and log10 |hat_truth| as `(t, k)` heatmaps, and
`notes/figures/tanaka_audit/banded_spectral.png` for L2 norms per band
(k<10 / 10<=k<30 / k>=30) overlaid with truth band norms.

| label      | dominant err band at end | err_high crosses truth_high at t |
|------------|--------------------------|----------------------------------|
| cid5       | **high (k>=30)**         | **1.6**                          |
| cid11      | **high**                 | **2.4**                          |
| cid1000006 | **high**                 | **2.4**                          |
| cid1000011 | carrier (10<=k<30)       | 102.4                            |

Early-step band ratios `err_band(t) / truth_band(t)` for v5 (full numbers in
`notes/figures/tanaka_audit/v5_early_step_diagnostics.json`):

| label      | high@t=2.4 | high@t=4.8 | high@t=8.0 | carrier@t=8.0 |
|------------|------------|------------|------------|---------------|
| cid5       | 1.29       | 2.25       | 3.83       | 0.006         |
| cid11      | 1.23       | 1.57       | 1.72       | 0.003         |
| cid1000006 | 1.09       | 1.52       | 1.62       | 0.010         |
| cid1000011 | 0.007      | 0.008      | 0.013      | 0.005         |

At t=8.0 the deep-water cases have `err_carrier << err_high` (3x to 600x
smaller, depending on case). The error cascade originates in and is
dominated by the high-k band; the carrier-band error stays well below the
truth-carrier amplitude until well past the crossover. This is consistent
with the order-6 CS-series amplification (`G_m ~ k^m * eta^(m-1)`)
preferentially exciting high-k.

The three deep-water cases (cid5, cid11, cid1000006) have an identical
spectral signature:

1. At t=0 the integrator feeds the model the truth IC, error is identically
   zero in every k.
2. After the first outer integrator step (t=0.8), the *peak* error sits in
   the low-k carrier band (`max|err|(t=0.8) ~ 3-5e-6`) while the high-k
   band err is still **at half the truth_high level** (ratio 0.4-0.5).
   The model isn't injecting white noise on step 1.
3. **At t=1.6-2.4 (steps 2-3) the high-k err crosses the truth_high level**
   (ratio 1.1-1.3x). Truth's high-k content for these deep cases is itself
   tiny (`|hat_truth|(k>=30) ~ 2-6e-7`) because the soliton's Fourier energy
   is concentrated at k<10. Once err_high exceeds this thin truth_high
   floor, there is no physical content to "anchor" the high-k modes - the
   error is free to grow on every subsequent step.
4. The high-k band then grows monotonically (cid5 banded plot shows
   err_high doubling roughly every t=2). The amplification comes from the
   order-6 CS series, where each `G_m` term scales like `k^m * eta^(m-1)`;
   at k=128 (the filter cut) the k^6 weight is `~7e12`, so a `1e-6` gxi
   error at `eta ~ 0.1` accumulates rapidly over 80 substeps per outer dt.
5. NaN time scales **inversely with a/h** in this set: cid5 (a/h=0.33) goes
   NaN at t=41.6; cid11 (a/h=0.27) at t=91.2; cid1000006 (a/h=0.305) at t=96.8.
   The cid5 vs cid1000006 ordering breaks pure a/h monotonicity. The cid5
   IC's residual periodic-wrap bump (section 1) gives it a slightly higher
   k>0 baseline at t=0 which is the most plausible discriminator.

cid1000011 (h=0.011) is qualitatively different and matches the f32 dynamic
range floor hypothesis:

- Truth has high-k content `~1e-4` (the very narrow soliton has wide-band
  Fourier support), so the model's high-k noise injection does not exceed
  truth_high until t=102.
- Final `alpha = -0.996`: the model's `eta(T)` is **anti-correlated** with
  the truth - i.e. the model has reproduced essentially the same shape but
  flipped sign (probably amplitude-only drift, not phase drift). `coh_l2`
  and `inc_l2` are within 4% of each other at the end (see
  `coherent_incoherent.png`).
- The rel_l2 "blowup" is a normalization artifact - absolute |err| is small,
  but the truth has |eta|_max=0.0022 so rel_l2=1.4 with abs_err ~3e-3.

The `coherent_incoherent.png` figure makes this concrete: for the three
deep-water cases, both coherent and incoherent errors climb together with
the incoherent error larger by ~3x at the end (shape/phase dominates). For
cid1000011, coherent and incoherent are tied.

## 3. Implication for v7 retrain

What this audit says about v5's failure mode:

1. **It is NOT a truth-validity issue** (truth drift < 5e-7 across all four
   failing ICs). The "saved-MATLAB shadowing" / "truth blowup" defenses do
   not apply.
2. **It is NOT a physics-stability issue** (all four ICs have a/h <= 0.33,
   well inside JCP09's stable a/h <= 0.6 envelope at M=6).
3. The failure is an **error cascade in the high-k band of the model's gxi
   prediction**, seeded by f32 precision in the very first integrator step
   and amplified by (i) the order-M CS series and (ii) the orbital
   displacement per outer dt at deep h (cid5: ~80 cells per dt). Once
   `|hat_err|(k>30)` exceeds `|hat_truth|(k>30)`, the cascade is essentially
   irreversible because the truth has no carrier in that band.
4. Two of the three deep-water failures (cid5, cid1000006) sit at h close
   to the dataset's `depth_max=0.30`, where the soliton's 1% periodic support
   is no longer `<< L=2*pi` (the generator's own safety comment flags
   `depth_max=0.08`). The IC itself is OK but the **eval is sampling a regime
   the training distribution covers thinly (1% of Tanaka rows)** while
   simultaneously being numerically marginal for the periodic FFT.

### What to look for when v7 lands at ~7am

- **Few-step diagnostic**: dump `err_high(t=2.4) / truth_high(t=2.4)` for the
  three deep cases (cid5, cid11, cid1000006). v5 ratio at t=2.4: 1.09-1.29x.
  Pass criterion for v7: < 0.1x. Below 0.1 the high-k band stays
  truth-anchored and the cascade cannot seed.
- **Per-band error at t=2.4** is a faster-to-evaluate proxy for "will this
  rollout survive?" than running the full t=200 rollout. Three steps is
  enough to predict the deep-water failures.
- **Stratify the Tanaka eval by (h, a/h) bin**: cid6 (h=0.133, a/h=0.30) and
  cid12 (h=0.089, a/h=0.256) PASS while cid5/cid11/cid1000006 FAIL with
  similar a/h - the discriminator is h, not a/h. v7's improvement on the
  three deep cases is the key.
- For cid1000011 (very-shallow): the alpha=-0.996 result suggests v5 learned
  the right *shape* but not the right *amplitude scaling* at h<0.02. v7
  metric to watch: alpha_final at h<0.02 ICs. If still negative, the model
  is amplitude-saturating not phase-drifting.

### Model-side vs data-side fixes

**Model-side fixes (no data change needed)**:

- **f32 -> bf16 mixed precision or f32 gxi residual head**: the model's
  injected high-k noise is at `eps_f32 ~ 1e-7`. If we ran gxi in f32 but
  applied a low-pass at the same `filter_fraction=0.25` (k_cut=128) on the
  *model output* (the eval already has the `filter_gxi` switch but it's
  off), the high-k err injection would be killed at source. That's a one-line
  eval flag, not a retrain.
- **Train with `filter_gxi` ON**: the v5 training labels are already filtered
  at `nu=0.25`; if we filter the model's gxi prediction the same way during
  training (and eval), the model never has to learn k>128 content and the
  cascade can't seed.
- **Pushforward / multi-step training**: v5 has `pushforward_steps=0`. A
  pushforward of even 1 step at training time would let the model see what
  its own first-step error looks like and learn to suppress it. (Caveat:
  see `project_pushforward_experiment.md` - we tried this on the DNO; the
  experiment claimed pushforward FAILED. Worth re-trying with the
  filter_gxi-on combination since the failure mode was different.)

**Data-side fixes (requires re-mix)**:

- **Up-weight the deep-Tanaka regime**: 1% coverage of `h>=0.23 AND a/h>=0.27`
  is thin. A v8 mix with the deep-Tanaka tail tripled (3-4%) would help
  without inflating dataset size much.
- **Trim or relabel the h>0.23 Tanaka cases**: the generator's own safety
  threshold is `depth_max=0.08` - either lower `depth_max` to 0.20 (drop the
  marginal periodic-wrap regime entirely) or regenerate with a larger
  effective domain via wider periodic extension (5-copy instead of 3-copy).
  This is a generator change, not a retrain change.
- **Add a few exact Tanaka labels at h=1 (JCP09's canonical setting) on a
  bigger domain**: gives the model a "ground truth at the canonical Tanaka
  configuration" that it currently never sees.

### Recommendation

When v7 lands, run the full audit script
(`notes/figures/tanaka_audit/_run_audit.py`, parameterized on the output
directory) against `v7/eval_suite_f64h/tanaka_{g0,g1}_trajs.npz`. The
key diagnostics in priority order:

1. `err_high(t=2.4) / truth_high(t=2.4)` on cid5, cid11, cid1000006. v5 ratios:
   1.09-1.29x. Pass criterion for v7: < 0.1.
2. `alpha_final` on cid1000011. v5: -0.996. Pass criterion: > 0.5.
3. NaN rate on Tanaka.

If (1) is the same as v5 (around 1-4x truth_high at t=2.4-8.0), v7's training
improvements did not address the model-output high-k floor; we should try
the eval-side `filter_gxi=true` flag as the fastest available mitigation
before considering an v8 retrain with filtered-output training.

`amin`, `amax_eta`, the per-IC spec JSON and all band ratios at every
saved time are in `notes/figures/tanaka_audit/audit_summary.json` and
`v5_early_step_diagnostics.json` for direct numerical comparison.
