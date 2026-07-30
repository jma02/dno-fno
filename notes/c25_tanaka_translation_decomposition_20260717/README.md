# C25 Tanaka translation/shape decomposition

This is a CPU-only analysis of all 64 fixed-panel C25 Tanaka trajectories
(32 `tanaka_g0`, 32 `tanaka_g1`).  All 64 reference trajectories pass the
stored truth-validity gate, and all saved surrogate fields are finite.  The
seven-case table is the preselected set of largest or near-tail C25 cases from
the terminal raw-error inventory: g0 indices 4, 21, 27 and g1 indices 11, 16,
24, 29.  It is not a threshold-selected set; in particular, g0 case 21 ends at
raw error 0.2156.

## Definitions

Let `T_d f(x)=f(x-d)` on the periodic interval of length `L`.  At each saved
time and for each case, the continuous displacement is

\[
 d(t)\in\mathop{\rm argmin}_{d\in\mathbb R/L\mathbb Z}
 \|T_{-d}\eta_p(t)-\eta_y(t)\|_2^2.
\]

The minimization uses cyclic correlation to locate candidate basins and
bounded continuous Fourier interpolation to refine the minimum below one grid
cell.  The raw and residual shape errors are

\[
 E_{\rm raw}=\frac{\|\eta_p-\eta_y\|_2}{\|\eta_y\|_2},\qquad
 E_{\rm shape}=\frac{\|T_{-d}\eta_p-\eta_y\|_2}{\|\eta_y\|_2}.
\]

The scalar error removable by a single translation is defined exactly by

\[
 E_{\rm trans}=\sqrt{\max(E_{\rm raw}^2-E_{\rm shape}^2,0)}.
\]

This is an error-budget definition, not a claim that the nonlinear translation
orbit is a linear orthogonal subspace.  The same elevation-fitted displacement
is applied to `xi` and `q=G(eta)xi`; those fields do not receive independent
best shifts.  The `xi` comparison removes its spatial mean because that mode is
a gauge.

For the exact alignment velocity, write `S_d f(x)=f(x+d)`,
`a=S_d eta_p`, and `e=a-eta_y`.  At a differentiable isolated minimizer,
`<e,a_x>=0`.  Differentiating this condition and using `eta_t=q` gives

\[
 d'=-\frac{\langle S_dq_p-q_y,a_x\rangle
              +\langle e,S_d(q_p)_x\rangle}
             {\langle a_x,a_x\rangle+\langle e,a_{xx}\rangle}.
\]

The code evaluates both numerator terms, the denominator, the stationarity
residual, the correlation with a centered finite difference of the unwrapped
`d(t)`, and the trapezoidal integral of `d'`.

## All 64 cases

| Panel | Terminal raw median / p95 / max | Terminal shape median / p95 / max | Raw crossings above 0.25 | Shape crossings above 0.25 | Translation-dominated at raw onset | Median one-grid lead to raw 0.25 |
|---|---:|---:|---:|---:|---:|---:|
| g0 | 0.01254 / 0.24116 / 0.66399 | 0.00524 / 0.03616 / 0.04401 | 2 | 0 | 2/2 | 36.0 |
| g1 | 0.03623 / 0.51080 / 1.14192 | 0.00727 / 0.10792 / 0.58067 | 4 | 1 | 3/4 | 28.8 |

Thus five of the six raw 0.25 crossings are primarily one-shift translation
events at onset.  The exception is the g1 multi-crest tail represented by case
24.  A one-grid displacement precedes the raw 0.25 crossing by tens of time
units, so the terminal event is not abrupt.

Across all saved frames, the identity velocity and numerical displacement
velocity have pooled Pearson/Spearman correlations 0.872/0.822 in g0 and
0.627/0.694 in g1.  Median terminal integral closure is 0.0036 and 0.0085 grid
cells, respectively.  These population correlations include cases where the
global alignment changes between competing local minima.  In g1 case 8, for
example, branch changes produce a 7.16-grid-cell integral discrepancy even
though each selected frame is stationary and well conditioned.  The identity
is a differential statement along one isolated smooth minimizer branch; it
does not include jumps caused by changing which minimizer is globally best.

## Seven largest or near-tail cases

| Case | Raw | Shape | Squared error removed | Shift (grid cells) | corr(identity `d'`, numerical `d'`) | Integral closure (grid cells) |
|---|---:|---:|---:|---:|---:|---:|
| g0 / 4 | 0.27235 | 0.00665 | 99.94% | -1.93 | 0.943 | -0.004 |
| g0 / 21 | 0.21565 | 0.03614 | 97.19% | +3.48 | 0.222 | -0.102 |
| g0 / 27 | 0.66399 | 0.04401 | 99.56% | -3.55 | 0.973 | +0.031 |
| g1 / 1000011 | 0.36204 | 0.01229 | 99.88% | -2.13 | 0.952 | +0.005 |
| g1 / 1000016 | 0.33675 | 0.01126 | 99.89% | +5.21 | 0.998 | +0.008 |
| g1 / 1000024 | 1.14192 | 0.58067 | 74.14% | -15.60 | 0.9994 | -0.011 |
| g1 / 1000029 | 0.69261 | 0.02051 | 99.91% | -4.19 | 0.959 | +0.005 |

Six of seven cases are overwhelmingly coherent translations after one common
shift.  Case 24 is qualitatively different: translation is substantial, but
the remaining shape error 0.581 is itself severe.  The exact velocity identity
closes the observed displacement particularly well in this case despite its
deformation, which confirms that the saved `q` fields propagate the fitted
translation coordinate rather than merely correlating with the final picture.

The direct and geometry contributions to `d'` can be individually much larger
than their sum and often cancel.  The direct term uses the total aligned
`q_p-q_y`, which contains both state separation and operator error; it must not
be described as a same-state neural-operator defect.

## Artifacts and verification

Each panel has a JSON report, framewise compressed NPZ, all-case CSV,
threshold-onset CSV, correlation CSV, and focused-case CSV in this directory.
The implementation is `scripts/analyze_rollout_translation_decomposition.py`.
Five synthetic CPU invariants pass: exact rigid-shift removal, rejection of a
pure amplitude/shape error as translation, the exact two-speed velocity
identity and integral, common alignment of `eta/xi/q`, and an end-to-end
archive/output test.  `py_compile`, Ruff, and `git diff --check` pass.
