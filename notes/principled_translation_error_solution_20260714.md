# A principled treatment of the remaining translation error

## Executive conclusion

**Design constraint.**  The deployed surrogate's analytic Craig--Sulem
backbone stops at \(G_0+G_1\).  Higher-order series terms may be used only by
the order-six truth/reference solver; they are not candidates for the learned
model, its rollout integrator, or an inference-time correction.

The remaining Tanaka tail is no longer a NaN or spectral-stability problem.
C20 is finite on all 320 fixed-suite rollouts.  C21's stronger tangent loss is
also finite, and its six Tanaka cases with final relative elevation error above
`0.25` consist of four almost rigid translations and two differential
multi-crest translations.  They remain spectrally smooth and well resolved.

The present localized tangent regularizer estimates a physically meaningful
local speed defect, but normalizes it as a *relative speed error*.  The reported
rollout metric is instead a *relative state error after a finite horizon*.
Those are not equivalent.  For a translated profile, the conversion factor is

\[
    \kappa(\eta)=\frac{\|\eta_x\|_2}{\|\eta\|_2},
\]

which varies by almost two orders of magnitude across the 64 Tanaka cases and
is strongly coupled to depth.  The most phase-sensitive case is precisely the
old C20 case 27; the difficult C21 two-crest case 24 is above the 90th
percentile.  The current loss divides this sensitivity away, so it permits the
optimizer to trade a small error reduction on broad waves for a damaging speed
error on narrow/shallow waves.

The smallest principled correction is therefore to replace the current
per-state relative-speed tangent loss by the instantaneous growth rate of the
actual relative-elevation error:

\[
    \boxed{
    \gamma_b^2
      = \frac{\int B_b(x)\,\delta c_b(x)^2\,dx}
                   {\int \eta_b(x)^2\,dx+\varepsilon}
      = gh_b\,\kappa_b^2\,\ell_{\mathrm{tan},b}.
    }
\]

Here \(B_b=K_{h_b}*(\eta_{x,b}^2)\), \(\delta c_b\) is the already implemented
local speed-defect estimate, and \(\ell_{\mathrm{tan},b}\) is the current
per-state tangent loss.  This is not hard-negative mining, a special Tanaka
case weight, or a new data pack.  It is the linearized map from an operator
defect to the quantity reported at rollout time.  It adds no network call and
no inference cost.

C22 has now executed the recommended isolated experiment: one
low-learning-rate epoch from C20 with the mean `gamma^2` loss contributing
`0.585%` of the train objective and included in validation.  It strongly repairs
the rigid case 27 but does not repair differential cases 22/24.  A completed
frame-zero discriminator further shows that the implemented spatial-variance
component is dominated by within-crest variation on case 27.  Therefore a
naive fourth moment or unnormalized CVaR of the current `gamma_diff_sq` is not
the next experiment.  First formulate and validate a coherent packet/crest
statistic or a capped, sample-balanced truth-trajectory tail statistic.

## 1. What is and is not failing

### 1.1 C21's remaining Tanaka tail

C21 improves final raw error in 41 of the 64 Tanaka cases and aligned error in
39 of 64.  Its six cases with raw final error above `0.25` separate cleanly:

- four single-pulse cases are nearly pure rigid translations; periodic
  alignment removes `99.1--99.95%` of their squared error;
- two same-going multi-crest cases have different accumulated shifts on
  different crests, while each crest retains nearly the correct local shape.

The decisive multi-crest measurements are:

| Case | Final crest shifts (grid points) | Individually aligned local errors |
|---|---:|---:|
| `tanaka_g1 / 1000024` | `[+6.21, -0.24]` | `[3.1%, 0.2%]` |
| `tanaka_g1 / 1000022` | `[+3.98, -1.91, +0.34]` | `[3.7%, 1.8%, 1.0%]` |

For case 24 the differential split is negligible near `t=20`, becomes
`[+1.48, -0.19]` at `t=100` and `[+3.43, -0.28]` at `t=150`, and reaches
`[+6.21, -0.24]` at `t=200`.  The aligned error grows smoothly rather than
jumping.  This is a secular local-speed allocation error.

Across the six tails:

- final high-band energy is `0.83--1.07` times truth;
- final maximum-slope ratios are `0.93--1.02` times truth;
- learned-Hamiltonian drift remains within approximately `+/-4.4%`;
- every saved `eta`, `xi`, and `Gxi` value is finite.

Thus this is not the old mid/high-band feedback cascade, an unresolved spatial
mode, or a failed GL2 solve.  It is also not primarily deformation: the
prediction is usually close to the correct group orbit but occupies the wrong
point on that orbit.

### 1.2 Why Hamiltonian control is only a partial answer

Spatial translation is a continuous symmetry of the Hamiltonian.  Two waves
that differ only by translation have the same Hamiltonian.  Consequently a
surrogate can conserve energy while accumulating an arbitrarily large phase
error.  The observed bounded Hamiltonian drift is useful evidence against an
amplitude blow-up, but it cannot identify the correct translation phase or
make two crests propagate at their correct separate local speeds.

### 1.3 Mechanisms already ruled out

The repository evidence rules out the tempting simple alternatives:

- **Integrator refinement:** earlier `80/160/320` substep sweeps did not move
  failure times monotonically, and the present tails have no abrupt
  non-contractive signature.
- **Spectral damping or a hard cutoff:** C21 tail spectra remain physical; the
  prior `k` cut removed NaNs only by destroying shallow-water signal.
- **A missing shallow-depth family:** sources 5, 6, and 14 contain 2.2 million
  Tanaka/steep-Tanaka rows, including dense shallow coverage.
- **Broken translation equivariance:** the architecture is convolutional and
  Fourier based, hence equivariant to a common periodic shift.  Equivariance
  says a translated input gets a translated output; it does not make the
  learned vector field advance at exactly the right speed.
- **More of the same global tangent weight:** C21 lowers the dataset tangent
  statistic by 17% and repairs case 27, but worsens case 24 and rigid g1 case 2.
  It redistributes phase error rather than eliminating the relevant risk.
- **Training modulo translation:** aligning the target inside the loss would
  explicitly forgive the tangent defect that causes the rollout error.

## 2. Translation error as a neutral-direction defect

Let \(T_s\eta(x)=\eta(x-s)\).  For a small displacement \(s\),

\[
    T_s\eta-\eta=-s\eta_x+\mathcal O(s^2),
\]

and therefore

\[
    \frac{\|T_s\eta-\eta\|_2}{\|\eta\|_2}
      = |s|\frac{\|\eta_x\|_2}{\|\eta\|_2}
        +\mathcal O(s^2)
      = |s|\kappa(\eta)+\mathcal O(s^2).
\]

If the model has local phase-speed defect \(\delta c(t)\), then

\[
    s(T)=\int_0^T\delta c(t)\,dt,
    \qquad
    e_\eta(T)\simeq
      \left|\int_0^T\delta c(t)\,dt\right|\kappa(\eta(T)).
\]

For an approximately coherent bias this becomes

\[
    e_\eta(T)\simeq T|\delta c|\kappa.
\]

This explains three otherwise confusing observations:

1. a one-step DNO error can be tiny while the final raw error is large;
2. the growth is secular rather than explosive;
3. narrow profiles are much less tolerant of the same speed error than broad
   profiles.

For C20 case 27, the final truth profile has \(\kappa=33.33393313\).  The
linearized shift producing relative error `0.25` is therefore
`0.25/kappa = 0.00750`; exact Fourier translation and bisection give `0.007566`.
The first-order formula is accurate to about one percent at the operational
threshold.  This final-profile value is distinct from the frame-zero
\(\kappa\) used in the loss-factor audit below.

## 3. What the current regularizer actually minimizes

Write

\[
    e_q=q_\theta-q_*,\qquad q=G(\eta)\xi,
\]

and let \(K_h\) be the periodic Gaussian with Fourier multiplier
\(\exp[-(hk)^2/2]\).  The current code forms

\[
\begin{aligned}
    A(x) &= K_h*(\eta_x e_q),\\
    B(x) &= \max\{K_h*(\eta_x^2),0\},\\
    \delta c_h(x) &= -\frac{A(x)}
      {B(x)+10^{-3}\max_y B(y)+10^{-30}},\\
    d\mu(x)&=\frac{B(x)}{\int B(y)\,dy}\,dx.
\end{aligned}
\]

Its per-state objective is

\[
    \ell_{\mathrm{tan}}
      =\int \left(\frac{\delta c_h(x)}{\sqrt{gh}}\right)^2d\mu(x).
\]

This is a sound positive-semidefinite, translation-invariant seminorm of the
DNO error.  The physical quantities, Gaussian scale, spatial normalization,
source selection, and multi-device selected-count correction are all correct.
Non-Tanaka rows do not dilute the selected-sample mean.  The problem is the
meaning of the normalization: \(\sqrt{gh}\) makes a dimensionless *fractional
speed error*, but the rollout score is not fractional speed error.

Because a normalized Gaussian preserves the integral,

\[
    \int B\,dx=\int\eta_x^2\,dx.
\]

It follows exactly that

\[
\begin{aligned}
    gh\,\kappa^2\ell_{\mathrm{tan}}
      &=gh\frac{\int B\,dx}{\int\eta^2\,dx}
        \int\frac{B}{\int B}
           \frac{\delta c_h^2}{gh}\,dx\\
      &=\frac{\int B(x)\delta c_h(x)^2\,dx}
              {\int\eta(x)^2\,dx}
       =:\gamma^2.
\end{aligned}
\]

For the tangent defect \(e_{\eta,t}\simeq-\delta c_h\eta_x\), \(\gamma^2\)
is the squared instantaneous growth rate of relative elevation error, with the
same local-window approximation already accepted in C19/C20.  A horizon-aware
version is

\[
    z_T=T_{\mathrm{ref}}^2\gamma^2
       =\left(T_{\mathrm{ref}}\sqrt{gh}\kappa\right)^2
        \ell_{\mathrm{tan}}.
\]

The global factor \(T_{\mathrm{ref}}^2\) can be absorbed into the loss
coefficient; the essential correction is \(gh\kappa^2\).

## 4. Quantitative evidence for the missing normalization

On the 64 matched C20 Tanaka initial states:

| Quantity | Result |
|---|---:|
| Frame-zero `kappa` minimum / median / p90 / maximum | `0.4819860400 / 6.3151146892 / 19.3302364980 / 35.2952751152` |
| Frame-zero Pearson correlation of `log(h)` and `log(kappa)` | `-0.9706889755` |
| Frame-zero `gh*kappa^2` minimum / median / maximum | `0.06240194586 / 2.028865511 / 14.45894588` |
| Pearson `corr(log(ell_tan), log(final raw eta error))` | `0.6502210258` |
| Pearson `corr(log(gamma^2), log(final raw eta error))` | `0.7369574728` |
| Raw-scale Pearson correlation, `ell_tan / gamma^2` | `0.796616 / 0.794757` |

The final truth profiles, which enter the translation-tolerance calculation
rather than the frame-zero loss factors, have `kappa` minimum / median / p90 /
maximum `0.4075981345 / 6.318302638 / 20.50914397 / 33.33393313` and Pearson
`corr(log(h), log(kappa)) = -0.9620172732`.  Thus the apparent improvement from
`0.6502210258` to `0.7369574728` is explicitly a log--log tail-ordering result.
The raw linear correlation does not improve.

Case 27 receives the largest physically derived factor (`14.46`).  Case 24's
factor is `5.85`, ranking seventh of 64.  Across the C21 final profiles, the
horizon sensitivity

\[
    S=T\sqrt{gh}\kappa
\]

has Spearman correlation `0.705` with final raw error
(`p=7.6e-11`).  The six C21 errors above `0.25` lie at sensitivity percentiles
`78, 89, 91, 92, 95, 100`.  This is much stronger ordering than a simple
initial global fitted-speed bias.

The result also explains the tolerance scale.  To keep linearized raw error
below `0.25` at `T=200`, the allowed mean speed error relative to
\(\sqrt{gh}\) is roughly `0.088%` at the median state, `0.035%` for case 27,
and `0.049%` for case 24.  These tolerances are substantially below the
remaining trajectory-averaged biases.

This is not proof from a single frame: temporal sign coherence and learned
trajectory states still matter.  It is, however, an exact derivation supported
by the correct case ordering, exact shift bisection, and the complete 64-case
tail audit.

## 5. Rigid and differential phase without crest detection

Peak finding is fragile when crests merge, split, or exchange order.  The
local speed field already permits a coordinate-free decomposition.  Let

\[
    \overline{\delta c}=\int\delta c_h\,d\mu.
\]

Then

\[
\begin{aligned}
    \gamma_{\mathrm{rigid}}^2
      &=\kappa^2\overline{\delta c}^{2},\\
    \gamma_{\mathrm{diff}}^2
      &=\kappa^2\int(\delta c_h-\overline{\delta c})^2d\mu,\\
    \gamma^2
      &=\gamma_{\mathrm{rigid}}^2+\gamma_{\mathrm{diff}}^2.
\end{aligned}
\]

The first term is the squared energy-weighted spatial mean of the speed defect;
the second is its spatial variance.  The latter can respond to different crest
speeds, but it is not an inter-crest variance: variation within one narrow
crest also enters.  That distinction is decisive in the completed
common-initial-state discriminator:

| C20 initial truth state | `gamma_diff_sq` rank | Exact `gamma_diff_sq` | C22 final geometry |
|---|---:|---:|---|
| `tanaka_g0 / case 27` | 1 | `2.24346294649e-3` | predominantly rigid |
| `tanaka_g1 / case 24` | 7 | `8.62902609365e-6` | two-crest differential |
| `tanaka_g1 / case 22` | 9 | `2.58167939588e-6` | three-crest differential |

The current statistic does put cases 22/24 in the upper truth-state tail, but
case 27 is about 260 times larger than case 24.  It classifies `97.47%` of case
27's implemented `gamma^2` as differential even though one rigid shift removes
`97.78%` of its C22 final squared elevation error.  C22 preserves ranks
`1/7/9`, and all three frame-zero scores increase slightly despite its large
case-27 rollout improvement.  The implemented soft spatial variance is thus
primarily detecting within-crest variation on this narrow pulse, not cleanly
isolating inter-crest phase allocation.

Consequently, do **not** apply a naive fourth moment or unnormalized CVaR to the
current `gamma_diff_sq`; either would further overweight the already-repaired
single-crest case.  Before another training run, formulate and validate either
a coherent packet/crest-speed statistic or a capped, sample-balanced tail
statistic over held-out truth trajectories.  It must rank the secular aligned
growth of cases 22/24 without being dominated by case 27.

## 6. Code audit

The current implementation is in
`train-jax-10m/translation_tangent_regularizer.py:39--117` and is called from
`train-jax-10m/1d_dno_fno_jax.py:1269--1300`.

Two implementation facts matter:

1. **There is no source-count dilution bug.**  Sources 5, 6, and 14 contain
   2.2 million rows, and the observed 1,759,644 selected training rows are their
   exact 80% train split.  The cross-device `shard_weight` produces the exact
   global selected-row mean.
2. **The tangent loss is absent from validation.**  The validation body at
   `train-jax-10m/1d_dno_fno_jax.py:1651--1680` returns only H1 and modal loss.
   Consequently checkpoint selection cannot see whether the new quantity
   improves or regresses.  Any next experiment must pass `source` into the
   validation step, compute the phase loss, log it, and include its full-weight
   value in the checkpoint composite.

There is a second, longer-term validation limitation.  `build_split_indices`
in `train-jax-10m/util.py:146--154` randomly permutes rows.  The v9 archive has
`source`, `depth`, and `time`, but no trajectory/case identifier, so different
frames of one generated trajectory can enter different splits.  This is
adequate for an IID one-step interpolation score but not a strong held-orbit
test of secular phase accuracy.  For the final manuscript run, retain a fixed
held-out IC rollout panel and, when the dataset is next rebuilt, store a case
ID and split by case rather than by row.

### Minimal implementation

After the existing `per_sample_loss` is computed, form

```python
slope_energy = jnp.sum(local_energy, axis=-1)
eta_energy = jnp.sum(eta**2, axis=-1)
rollout_sensitivity_sq = (
    gravity * depth * slope_energy / (eta_energy + eta_energy_floor)
)
per_sample_gamma_sq = rollout_sensitivity_sq * per_sample_loss
```

Then take the selected-sample mean using the existing source mask and existing
multi-device correction.  Keep the old fractional-speed loss as a diagnostic.
Also log `kappa`, `gamma_sq`, and the rigid/differential decomposition.

The denominator floor should be scale-aware and tested on the selected v9
distribution.  A fixed sensitivity cap should not be introduced unless an
offline scan reveals near-zero-elevation numerical outliers; the observed
physical factors are moderate and carry the desired signal.

### Required invariants

- exact prediction gives zero loss;
- a pure traveling defect `e_q=-delta_c eta_x` gives
  `gamma_sq = kappa_sq * delta_c_sq` up to the documented local-energy floor;
- profiles with wavenumbers `k1` and `k2` receive the expected
  `(k2/k1)^2` phase-sensitivity ratio;
- common periodic translation leaves the loss unchanged;
- consistent physical amplitude scaling leaves relative-error growth
  unchanged;
- two separated crests with opposite local defects do not cancel;
- adding nonselected samples or changing their device allocation does not
  change the selected loss;
- zero/small-amplitude inputs and their gradients remain finite;
- validation and training use the same loss definition and full coefficient.

## 7. Why not start with a full Hamiltonian-vector tangent

The full Zakharov vector field is

\[
\begin{aligned}
    F_\eta &= q,\\
    F_\xi &=-g\eta-\frac12\xi_x^2
       +\frac{(q+\eta_x\xi_x)^2}{2(1+\eta_x^2)}.
\end{aligned}
\]

One can nondimensionalize \(F_\eta\) by \(\sqrt{gh}\), \(F_\xi\) by \(gh\),
and project \(F_\theta-F_*\) onto
\((\eta_x,\xi_x/\sqrt{gh})\).  This is a useful offline diagnostic of group
tangent error.

It should not be the first new training loss.  The only learned output is
\(q\); the \(F_\xi\) difference is a deterministic nonlinear reuse of the same
\(q\) label.  A full-vector penalty therefore adds no independent target, may
double-count the same operator defect, and requires a less canonical choice of
state-space metric and localization.  The proposed `gamma^2` has a direct
first-order identity with the reported relative-eta rollout metric.  Use the
full-vector projection only if an offline truth-orbit audit ranks long-time
tails materially better than `gamma^2`.

## 8. Controlled experiment sequence

### Phase A: offline calibration

1. Evaluate old `ell_tan`, `gamma_rigid_sq`, `gamma_diff_sq`, and `gamma_sq` on
   representative C20 batches and a fixed held-out Tanaka truth-orbit panel.
2. Record mean, p90, p99, maximum, and source/depth breakdown.
3. Choose the coefficient so the *weighted* `gamma_sq` term is approximately
   `0.5--1.0%` of converged H1, with a short warmup.
4. Do not transfer weight 100.  On the 64 C20 initial states, mean
   `gamma_sq=5.18e-5` versus old mean `ell_tan=4.18e-6`; a panel-based estimate
   gives a coefficient near `1.3` for a 1% contribution, but the full-v9 batch
   calibration is authoritative.

### Phase B: one-variable continuation

Run one low-learning-rate full-v9 epoch from the same C20 checkpoint with:

- architecture, H1, modal, and Hadamard settings unchanged;
- old tangent objective replaced by mean `gamma_sq`;
- reset Adam, as in C21;
- phase loss present in validation/checkpoint selection;
- no new data, guard, cutoff, damping, architecture, or risk term.

Evaluate the identical 64 Tanaka cases first.  This isolates whether the
missing rollout sensitivity is causal.

### Phase C: differential-tail fallback

The C22 outcome satisfies the premise: rigid case 27 improves while cases 22/24
retain differential error.  The frame-zero discriminator rules out a naive
fourth moment or unnormalized CVaR of the current spatial-variance component.
The next controlled step is offline, not training: construct a coherent
packet/crest-speed statistic or a capped, sample-balanced truth-trajectory tail
statistic, and require it to predict cases 22/24 without case-27 domination.

### Phase D: off-trajectory fallback

Only if the offline audit shows that phase risk is small on truth states and
blooms only on learned states, add sparse signed one-substep tubular training:

1. take detached `dt=+/-0.01` surrogate steps from truth states;
2. evaluate the exact `M=6` DNO label at those perturbed states;
3. apply only the `gamma_sq` loss on a small microbatch, e.g. 8 examples every
   16 steps.

This directly addresses local covariate shift around the truth manifold and is
substantially cheaper and easier to explain than unrolling the full GL2 map.
It is unnecessary unless the truth-state diagnostic fails.

### Phase E: fresh run only after the diagnostic passes

A fresh 40-epoch retrain is justified only after a one-epoch continuation
improves both rigid and differential held-out tails without harming other
families.  Confirm the final configuration with a second seed or a genuinely
case-held-out selection panel before manuscript claims.

## 9. Acceptance criteria

For the one-epoch diagnostic, require all of the following:

- 0 nonfinite rollouts;
- preserve C21 case 27: raw `<=0.37`, aligned `<=0.06`;
- recover case 24: raw `<=0.50`, aligned `<=0.15`;
- combined Tanaka counts no worse than C21's `6/1/0/0` above
  `0.25/0.50/0.75/1.00`;
- g1 aligned p95 approximately `<=0.107`;
- H1 validation no more than 0.5% above C20 and modal validation no more than
  5% above C20;
- on the eight non-Tanaka families, median/p95 do not regress by more than
  approximately 5%/10%;
- the held-out rigid and differential `gamma` tails both improve, rather than
  only the named cases.

These gates prevent another apparent case repair that merely transports the
phase defect elsewhere.

## 10. Bottom line

The stability problem that produced NaNs has been solved structurally.  The
remaining Tanaka error is a neutral-direction accuracy problem.  C22 confirms
that the `gamma^2` normalization can repair a rigid phase tail, but its dataset
mean does not control the differential multi-crest tail.  The present
`gamma_diff_sq` is also not a clean inter-crest statistic, so naive
fourth-moment/CVaR reweighting is contraindicated.  The next step is to validate
a coherent packet/crest predictor or capped, sample-balanced truth-trajectory
tail before spending another optimization run; use tubular labels only if no
truth-state predictor anticipates the subsequent growth.
