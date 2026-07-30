# Finite-depth Stokes case 98

Date: 2026-07-23

## Result

Case 98 should not have been generated. It is not a bad shuffle, a rescaling
error, an interpolation error, or a failure introduced by the time integrator.
The finite-depth fifth-order Stokes formula was evaluated where its higher
harmonics are no longer small corrections to its leading harmonic.

## Exact source

The case is global row 98 of `data/test_dno_rescaled.npz`. Reversing the
seed-42 test-set shuffle maps it to finite-depth Stokes parameter draw 316 and
saved phase 34. Replaying the seed-43 generator gives
\[
  L=164,\qquad h=1,\qquad n=14,\qquad
  k=\frac{2\pi n}{L}=0.5363694774,
\]
\[
  a=0.2254346639,\qquad kh=0.5363694774,\qquad
  ka=0.1209162729.
\]
The reconstructed \(\eta\) and \(\xi\) agree with the archived row to
approximately \(3\times10^{-6}\) relative error.

The legacy sampler in `solver/gen_data/regenerate_dno_stokes.py` accepts an
amplitude whenever
\[
  ka\leq 0.15.
\]
This condition alone does not control finite-depth coefficient growth.

## Why the profile is sharp

Write the generated surface as
\[
  \eta(\theta)=\sum_{m=1}^{5} E_m\cos(m\theta).
\]
For case 98, the five harmonic amplitudes are
\[
  (E_1,E_2,E_3,E_4,E_5)
  =
  (0.247496,0.135204,0.070063,0.054616,0.032900).
\]
Consequently,
\[
  \frac{|E_2|}{|E_1|}=0.5463,
  \qquad
  \frac{\sum_{m=2}^{5}|E_m|}{|E_1|}=1.1830.
\]
The combined second through fifth harmonics are larger than the fundamental.
The crest height grows from \(0.225h\) at first order to \(0.540h\) at fifth
order, and the crest-to-trough height is \(0.706h\). The visible sharp crest
and smaller oscillations are therefore present in the analytic five-mode
formula; they are not grid noise.

This behavior is expected from the coefficients in
`solver/data/stokes_truth_jax.py`. They contain inverse powers of
\(\tanh(kh)\), \(\sinh(kh)\), and \(\cosh(2kh)-1\). As \(kh\) decreases,
these coefficients grow rapidly, so the deep-water condition \(ka\ll1\) no
longer orders the finite-depth expansion by itself.

## Direct water-wave-equation check

For an exact traveling wave, the kinematic equation requires
\[
  \eta_t=G(\eta;h)\xi.
\]
Directly differentiating the analytic Stokes formula and evaluating the
repository's Dirichlet--Neumann series gives
\[
  \frac{\|\eta_t-G_6(\eta;h)\xi\|_2}{\|\eta_t\|_2}=0.2699.
\]
Increasing the Dirichlet--Neumann order to nine leaves the defect at
approximately \(0.2702\). The missing accuracy is therefore not supplied by
more Craig--Sulem terms. The initial fifth-order Stokes state itself is not a
close traveling solution of the numerical water-wave equations.

The corresponding C27 truth rollout remains finite but has maximum relative
Hamiltonian drift \(0.02219\), compared with the declared acceptance tolerance
\(0.001\). This is why the evaluation rejects it.

## A pre-generation condition

The simplest condition tied directly to the declared fifth-order construction
is:
\[
  \boxed{\sum_{m=2}^{5}|E_m|\leq |E_1|.}
\]
The coefficients \(E_m\) are explicit functions of \((k,a,h)\), so this is
checked before constructing a grid function or starting a rollout. It is not
a visual roughness filter. It says only that all nominal corrections combined
may not exceed the leading wave.

Applied retrospectively:

| coefficient bound | current 500,000-row finite-Stokes archive removed | legacy parent waves removed | fresh 256-case rollout panel |
| --- | ---: | ---: | ---: |
| higher/fundamental \(\leq1\) | 11,154 (2.23%) | 16/392 (4.08%) | all 7 invalid and 1 valid |
| higher/fundamental \(\leq0.75\) | 21,153 (4.23%) | 29/392 (7.40%) | all 7 invalid and 4 valid |
| higher/fundamental \(\leq0.5\) | 39,856 (7.97%) | 63/392 (16.07%) | all 7 invalid and 14 valid |

The threshold one is the least destructive statement of asymptotic ordering.
It rejects case 98 and every truth-invalid finite-Stokes case in the fresh
256-case panel.

An input-only conservative alternative is
\[
  ka\leq \frac{(kh)^3}{3}.
\]
At the same \(k\) and \(h\), this would cap case 98 at
\[
  a=0.095897,\qquad ka=0.051436.
\]
The resulting profile has higher-harmonic ratio \(0.383\), crest height
\(0.134h\), and kinematic defect \(0.0068\). It is visibly and numerically
well ordered, but this bound removes \(8.96\%\) of the current 500,000-row
archive, compared with \(2.23\%\) for the direct coefficient-order condition.

## Recommended corpus change

For the paper corpus, sample \((k,a,h)\), compute the five elevation
coefficients, and accept the parameters only when
\[
  ka\leq0.15
  \quad\hbox{and}\quad
  \sum_{m=2}^{5}|E_m|\leq |E_1|.
\]
The first condition limits nominal steepness. The second makes that condition
meaningful at finite depth. Store one parameter draw as one case; if it is
invalid, remove all of its translated phases rather than deleting individual
rows.

The diagnostic figure and arrays are in
`outputs/stokes_case98_audit_20260723/`.
