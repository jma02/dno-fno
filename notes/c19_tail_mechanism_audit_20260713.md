# C19 tail-mechanism audit and next-step recommendation

## Executive conclusion

The 16 truth-valid C19 rollouts with final raw relative elevation error above
`0.25` are not 16 versions of the former NaN mechanism.  All states remain
finite and smooth.  Their errors grow secularly rather than explosively, and
none shows a precursor high-frequency cascade or loss of GL2 stability.

The population separates into:

- **9 rigid phase errors:** the predicted wave is nearly the correct wave at
  the wrong periodic translation;
- **5 differential phase/shape errors:** different crests or carrier modes
  accumulate different phase errors; and
- **2 modal-amplitude errors:** the nonlinear transfer among resolved modes is
  wrong and no translation can repair the result.

The natural next step is therefore **not more spectral damping, a harder
cutoff, more GL2 substeps, or another targeted-data pack**.  First evaluate the
already-retained C19 epoch-1 checkpoint on this fixed 16-case panel.  It is the
cleanest zero-training test of whether the second fine-tuning epoch simply
overstepped.  If epoch 1 does not preserve the zero-divergence result while
recovering the C19 regressions, perform one clean 40-epoch retrain of the C16
architecture on v9 with the corrected localized tangent loss present from the
start at its existing weight 10, rather than applying it as a late two-epoch
fine-tune.

This is principally a training-objective and checkpoint-selection problem.
The soliton-selective spectral guard remains useful insurance against the
separate, already-diagnosed transverse cascade, but it is inactive on every
tail studied here and cannot improve these errors.

## Scope and diagnostics

The audit covers every truth-valid trajectory in the final guarded C19 suite
whose final

\[
e_{\mathrm{raw}}
=\frac{\|\eta_\theta(T)-\eta_*(T)\|_2}{\|\eta_*(T)\|_2}
\]

exceeds `0.25`.  There are 16 such cases among 305 truth-valid trajectories.
For each final state, periodic translation alignment minimizes

\[
e_{\mathrm{align}}
=\min_s
\frac{\|T_s\eta_\theta(T)-\eta_*(T)\|_2}{\|\eta_*(T)\|_2}.
\]

An exhaustive integer circular correlation was refined to a continuous shift
using Fourier translation.  The fraction of squared discrepancy explained by
a rigid translation is

\[
R_{\mathrm{phase}}
=1-\left(\frac{e_{\mathrm{align}}}{e_{\mathrm{raw}}}\right)^2.
\]

The audit also tracks crest locations and heights, modal amplitudes and phases,
spectral-band energies, maximum slope, learned-Hamiltonian drift, frame-to-frame
growth, and the activation of the deployed soliton guard.  A positive listed
shift moves the prediction to the right; one grid point is
`2*pi/1024 = 0.006136` in physical coordinates.

## All 16 cases

| Family / case | C16 raw | C19 raw | C19 aligned | Shift (grid points) | Squared error removed | Learned-H drift | Diagnosis |
|---|---:|---:|---:|---:|---:|---:|---|
| `bf_g0 / 23` | 0.8680 | 0.9120 | 0.8775 | +2.95 | 7.4% | -2.47% | Incorrect in-band BF sideband transfer and recurrence timing |
| `tanaka_g1 / 1000029` | 0.6067 | 0.7920 | 0.0344 | +4.98 | 99.81% | +1.87% | Rigid single-soliton phase/speed error |
| `tanaka_g1 / 1000011` | 0.5231 | 0.7141 | 0.0288 | +4.48 | 99.84% | +2.87% | Rigid single-soliton phase/speed error |
| `tanaka_g0 / 4` | 0.4537 | 0.5155 | 0.0210 | +3.75 | 99.83% | +3.16% | Rigid single-soliton phase/speed error |
| `tanaka_g1 / 1000005` | 0.4951 | 0.4998 | 0.0264 | +11.21 | 99.72% | +4.98% | Rigid single-soliton phase/speed error |
| `tanaka_g1 / 1000024` | 0.7254 | 0.4734 | 0.1273 | +3.85 | 92.77% | -6.07% | Two crests have different phase errors, with secondary amplitude drift |
| `tanaka_g1 / 1000001` | 0.3609 | 0.4326 | 0.4042 | +2.03 | 12.72% | +2.15% | Overlapping crests move in opposite relative directions; genuine interaction/shape error |
| `tanaka_g1 / 1000028` | 0.4600 | 0.4178 | 0.0985 | +4.61 | 94.44% | +2.55% | Differential two-crest phase error |
| `tanaka_g0 / 27` | 1.1673 | 0.3860 | 0.0529 | -1.94 | 98.12% | +0.37% | Rigid phase divergence strongly repaired by C19 |
| `bf_g0 / 0` | 0.3639 | 0.3799 | 0.0393 | -12.34 | 98.93% | -1.65% | Rigid carrier-phase error |
| `tanaka_g0 / 8` | 0.3874 | 0.3623 | 0.0155 | +3.15 | 99.82% | +2.67% | Rigid single-soliton phase/speed error |
| `bf_g1 / 1000017` | 0.3598 | 0.3620 | 0.2070 | +5.25 | 67.32% | -6.59% | Differential modal phase plus amplitude depletion |
| `bf_modal / 2000004` | 0.3261 | 0.3331 | 0.0405 | -10.71 | 98.52% | +0.03% | Rigid carrier-phase error |
| `tanaka_g0 / 9` | 0.2974 | 0.3147 | 0.0753 | -7.65 | 94.28% | -2.34% | Differential two-crest phase error |
| `linear / 14` | 0.1371 | 0.2968 | 0.2602 | non-unique | 23.11% | +0.14% | Incorrect nonlinear harmonic redistribution and modal timing |
| `tanaka_g0 / 17` | 0.5768 | 0.2678 | 0.0280 | +1.69 | 98.91% | -2.84% | Rigid phase tail strongly repaired by C19 |

Nine cases are well described by a single translation.  Five require relative
motion between crests or modes.  The two clean modal-amplitude failures are
`bf_g0 / 23` and `linear / 14`; `tanaka_g1 / 1000001` is also a genuine local
interaction/shape error, but is grouped with the differential multi-crest
cases because the defect is opposing crest motion rather than a global
amplitude-transfer bias.

## What C19 changed relative to C16

Across all 305 truth-valid cases, the raw-error tail changed as follows:

| Threshold | C16 | C19 |
|---|---:|---:|
| `> 0.25` | 18 | 16 |
| `> 0.50` | 7 | 4 |
| `> 0.75` | 2 | 2 |
| `> 1.00` | 1 | 0 |

C19 therefore is a real net improvement and achieves the stated stability
goal.  It removed three C16 cases above `0.25`, introduced one new case
(`linear / 14`), and reduced the counts above `0.50` and `1.00`.  However, among
the 16 cases that remain in the C19 tail, only five improve and eleven regress
relative to C16.  The large improvements are concentrated in Tanaka cases 27,
17, and 24.  The largest regressions are Tanaka cases 29 and 11 and linear case
14.  Mean non-Tanaka final error changes by only about `+0.0016`, so this is not
a broad collapse; it is a small shared-parameter displacement with a few large
tail consequences.

This pattern is consistent with a two-epoch late fine-tune moving the learned
phase-speed bias around the state space rather than uniformly eliminating it.
C19's second epoch improves its aggregate localized-speed statistic by only
about 2.6%, while its one-step validation loss remains essentially tied with
C16.  Aggregate validation therefore cannot select the better long-time phase
map.

## The genuine non-phase cases

### Benjamin--Feir `bf_g0 / 23`

This is the largest remaining error and is already present in C16.  The
dominant carrier remains reasonable, but the sideband exchange is mistimed:
truth mode 17 continues growing through `t=200`, whereas the prediction peaks
near `t=105`; truth mode 21 decays after approximately `t=143`, whereas the
prediction keeps growing.  At the final time, prediction/truth amplitude ratios
for modes 17, 19, and 21 are approximately `0.53`, `1.03`, and `7.25`.

This is an in-band nonlinear recurrence defect.  It is not a high-frequency
instability, and damping it would suppress physical sideband dynamics present
in the truth.

### Tanaka `g1 / 1000001`

The two overlapping crests have offsets of opposite sign (approximately -5
and +5 grid points) and peak ratios about `1.084` and `0.930`.  A global shift
can remove only 12.7% of the squared error.  This is an interaction error, not
a scalar traveling-wave speed error, and explains why simply increasing the
old global tangent weight is not a general solution.

### `linear / 14`

This case has small learned-H drift but substantially wrong harmonic magnitude
and phase.  It is direct evidence that Hamiltonian conservation alone does not
imply trajectory fidelity: a state can remain close to the correct energy
level while redistributing that energy incorrectly among modes.

There is also an evaluation-semantics issue.  The `linear` registry entry in
`solver/evals/eval_suite.py` leaves `truth_kind` at its default `"nonlinear"`.
The implemented `"linear_analytic"` branch is therefore not used.  The present
case is a nonlinear evolution of a shallow single-mode initial condition, not
an analytic linear-dispersion test, despite older documentation calling this
the linear regime.  Before optimizing against case 14, either set
`truth_kind="linear_analytic"` if that was the intended test, or rename the
current family as a shallow nonlinear harmonic-generation benchmark.

## Mechanisms ruled out

### Renewed NaN or high-frequency cascade

All predicted `eta`, `xi`, and `gxi` arrays are finite for all 320 suite
trajectories.  The maximum error increment between saved frames in this tail is
only about `0.019`.  Predicted frame-to-frame state jumps track truth within
about 2.3%, and maximum slopes remain between approximately 0.84 and 1.17 times
their truth values.  Fifteen cases attain their maximum raw error at the final
frame; the remaining BF case peaks only slightly before the end.  This is
smooth secular accumulation, not terminal blow-up.

### Spectral-guard failure

Every Tanaka tail remains at only about 1--1.6% of its guard trigger, with
maximum sigmoid activation below `1.3e-18`.  The non-Tanaka selector is false.
The guard is consequently an identity map on these trajectories.  Strengthening
it would either do nothing or damage physical high-band evolution.

### GL2 time discretization

The NPZs do not contain direct Picard-residual telemetry because residual
checking was disabled, so one cannot infer contraction constants from them.
Nevertheless, none exhibits the abrupt slope or high-band surge associated
with the former non-contractive cases.  Earlier 4x substep-refinement tests also
failed to move NaN times monotonically.  There is no evidence that more GL2
substeps will reduce these secular operator errors.

### A simple missing-depth data gap

The v9 dataset contains 7,633,989 rows.  Tanaka sources 5 and 6 contribute one
million rows each and source 14 contributes 200,000.  Sources 5 and 6 contain
about 198,400 and 206,600 rows below `h=0.02`, respectively, directly covering
the shallowest tail cluster.  The tail depths and amplitudes lie inside dense
training support.  The Tanaka and BF evaluation ICs also originate from the
same trajectory archives used to build v9, so adding more copies of the same
pointwise states is unlikely to address the accumulated dynamics.

This does not prove that all useful trajectory neighborhoods are covered, but
it rules out the simple explanation that the model has never seen these depths
or wave families.  Historical targeted and hard-negative additions also did
not provide a robust, manuscript-defensible cure.

### A `modes=64` bandwidth bottleneck

Although the CLI records `--modes 64`, `CraigSulemDNO.modes` is explicitly
unused in `models/dno-net/dno_net_v2.py`; its depth-aware spectral multipliers
operate over the full represented rFFT grid.  Raising that flag to 128 would
not change the model and is not an experiment.

### More Hamiltonian weight alone

The worst learned-H drift in the tail is bounded at about 6.6%, but drift does
not order the errors.  `linear / 14` has only `+0.14%` drift while its raw error
more than doubles relative to C16.  Hamiltonian control is important for
stability but does not identify phase on a symmetry orbit or the correct modal
energy distribution.

## Training-side interpretation

The corrected localized tangent term is physically appropriate for the
dominant Tanaka mechanism, but C19 applies it only to source IDs 5, 6, and 14
and only during a two-epoch fine-tune from an already-converged C16 checkpoint.
At weight 10, its weighted contribution is approximately 0.1% of total
training loss.  Teacher-forced probes show that C19 lowers several global
speed projections, which explains the repaired phase tails, but does not
uniformly reduce the local DNO defect and slightly worsens the largest remaining
single-soliton tails.  This is evidence against merely increasing its weight.

The more natural optimization problem is to let the representation accommodate
the pointwise, Hadamard, and localized-tangent objectives jointly throughout
training.  A late low-learning-rate displacement can improve the target
subpopulation while degrading untargeted shared features; a fresh mixed-objective
run need not make that compromise in the same direction.

## Recommended sequence

### 1. Zero-training checkpoint test

Evaluate the already-retained checkpoint
`outputs/c19_localized_tangent_full_20260712_232407/best_val_ckpt/ckpt_1` on
these 16 fixed cases with exactly the C19 guard and harness.  Epoch 1 is an
actual smaller optimization update, so it is a cleaner first test than an
arbitrary parameter interpolation.  Its validation loss (`0.00673739`) is
essentially C16's (`0.00673693`).

Accept epoch 1 for a full-suite run only if it:

- preserves finite predictions and no raw error above 1;
- retains the major C19 repairs on Tanaka cases 27, 17, and 24;
- reduces the regressions on Tanaka cases 29 and 11 and `linear / 14`; and
- does not worsen the `>0.25`, `>0.50`, and `>0.75` tail counts.

This test requires no training and should be done before launching another
day-long run.

### 2. Durable experiment if epoch 1 is not enough

Train one fresh 40-epoch model with:

- the exact accepted C16 architecture and v9 dataset;
- the existing pointwise/Sobolev objective;
- the existing Hadamard weight `0.01` and schedule;
- the corrected h-localized tangent loss at its current weight `10`, active
  from the beginning with a short first-epoch warmup; and
- no new hard-negative data, hard cutoff, larger tangent weight, mode-count
  change, or architecture change.

Keep multiple late checkpoints.  Select among them using a fixed rollout
validation panel that reports raw and translation-aligned soliton error,
per-crest error for multi-solitons, BF modal amplitude/phase error, nonlinear
harmonic error, and ordinary one-step validation.  Confirm the selected
checkpoint on separate seeds so the 16-case panel does not become another
training target.  Continue to deploy the independently motivated
soliton-selective guard as inactive-until-needed safety insurance.

Based on the C16 wall time, this full run is approximately 24--25 hours on the
two local RTX 6000 Ada GPUs.  It is the simplest manuscript-defensible next
training experiment because it changes one thing: when the already-validated
localized tangent objective enters optimization.

### 3. Only if the modal outlier survives

A fresh pointwise retrain may still leave `bf_g0 / 23`, because its error is a
long-time in-band recurrence mismatch already visible in C16 and C19.  If the
full mixed-objective run leaves that isolated mechanism unchanged, the next
separate study should be a very low-weight one-substep trajectory/modal-
consistency loss applied across physical families, with random-sea accuracy as
an explicit rejection criterion.  Historical pushforward training improved
some coherent-wave regimes but badly regressed random seas and increased NaNs,
so it should not be bundled into the primary retrain without a controlled
ablation.

## Bottom line

C19 plus the guard already solves the NaN objective on this suite.  The
remaining `>0.25` population is mostly a phase-accuracy tail, not concealed
instability.  The immediate action is to evaluate C19 epoch 1.  The durable
next training action, if needed, is one clean C16-architecture retrain with the
correct localized tangent objective present conservatively from epoch 1 and
rollout-aware checkpoint selection.  Do not add damping or targeted data until
that controlled experiment has been performed.
