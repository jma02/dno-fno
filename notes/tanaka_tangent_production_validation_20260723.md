# Tangent-aware Tanaka construction

## Question

The May-v2 Tanaka constructor first computes a smooth solitary-wave profile
parametrically and then places it on the periodic numerical grid.  The old
placement used piecewise-linear interpolation.  Its corners are small in the
surface elevation, but they leave Fourier coefficients that are amplified by
the order-six Dirichlet--Neumann calculation.

The exact case-31 experiment showed that replacing only this interpolation
removes the oscillatory tail.  This note records the production implementation
and asks whether it remains accurate over the full data range.

## Construction

The Tanaka solve returns knots

\[
    (x_j,\eta_j,\theta_j),\qquad j=0,\ldots,J-1.
\]

Its parametric equations give the physical surface slope directly:

\[
    \frac{d\eta}{dx}(x_j)=\tan\theta_j.
\]

On every interval \([x_j,x_{j+1}]\), the new constructor uses the unique cubic
that matches the two endpoint values and these two endpoint slopes.  The
profile is zero outside the computed Tanaka support.  Its endpoint values and
slopes are checked before they are set exactly to zero.

Both \(x\) and \(\eta\) are multiplied by the case depth \(h\).  Therefore

\[
    \frac{d(h\eta)}{d(hx)}=\frac{d\eta}{dx},
\]

so the Hermite slopes are not rescaled separately.

The periodic field is the finite sum of every translated compact-support
profile that intersects \([0,2\pi]\).  This replaces the old fixed three-copy
sum.  The fixed sum was adequate for case 31 but was not uniformly converged
at the broad, weak-wave corner: after projection to modes \(|k|\le128\), three
copies differed from the complete sum by \(2.96\times10^{-4}\) at
\(h=0.30\), \(a/h=0.05\).  Summing all intersecting copies removes the arbitrary
image truncation.

The remaining construction is unchanged:

1. evaluate on the \(8N\) placement grid;
2. restrict by Fourier truncation to the \(N\)-point grid;
3. construct the surface potential \(\xi\);
4. project to modes \(|k|\le128\);
5. use the order-six, pad-eight Dirichlet--Neumann series and the existing GL2
   rollout.

New archives identify this construction as
`periodic_tangent_hermite_tan_theta_v1`.  An incomplete archive made with a
different construction cannot be resumed, which prevents mixed datasets.

## Focused implementation tests

The direct CPU test
`solver/gen_data/tests_tanaka_tangent_hermite.py` checks:

- exact interpolation of a cubic on nonuniform knots;
- the prescribed derivative at every knot;
- zero value and derivative outside the support;
- increasing Tanaka \(x\)-knots and \(\cos\theta>0\);
- nonnegativity and half-profile monotonicity for
  \(a/h\in\{0.05,0.25,0.45\}\);
- convergence of the periodic image sum;
- periodic translation and direction-reversal identities;
- the exact case-31 Fourier tail; and
- one complete GL2 output interval.

All checks pass.  A two-sample end-to-end generator smoke test also writes
finite \(\eta\), \(\xi\), and \(G(\eta;h)\xi\) arrays with the new construction
metadata.

## Twelve-case static panel

The predeclared panel covers:

- the May-v2 range \(h\in[0.01,0.30]\) and total steepness
  \(S\in[0.10,0.35]\);
- one-, two-, and three-crest states;
- the observed component tail below the nominal \(0.05\) floor;
- the steep-Tanaka range up to \(h=0.35\), \(a/h=0.45\);
- seam placement, half-period translation, and direction reversal; and
- the exact May-v2 case 31.

Each state was independently constructed at
\(N=1024,2048,4096\), always compared on the fixed band
\(|k|\le128\).  The 257-knot Tanaka solve was also compared with a 513-knot
solve.

All 180 static gates pass.  The largest grid defect is
\(1.032\times10^{-7}\), attained by \(G(\eta;h)\xi\) for the narrow, steep
case; the declared label budget is \(10^{-3}\).  The largest 257-versus-513
defects are \(4.175\times10^{-4}\) in modes \(0{:}32\) and
\(4.166\times10^{-4}\) in modes \(0{:}128\), both in \(\xi\) for the same
case.  Their budgets are \(10^{-3}\) and \(2\times10^{-3}\).

The minimum radicand normalized by \(c^2\) is \(0.372>0\), and the largest
algebraic Bernoulli residual is \(4.20\times10^{-15}\).  Half-period
translation errors in \(\eta,\xi,G(\eta;h)\xi\) are

\[
    1.54\times10^{-16},\quad
    2.17\times10^{-16},\quad
    3.51\times10^{-11}.
\]

Direction-reversal errors are zero to stored precision.

For exact case 31, the ratios of the piecewise-linear and Hermite
mode-\(80{:}128\) norms are

\[
    91.95\quad(\eta),\qquad
    112.53\quad(\xi),\qquad
    141.91\quad(G(\eta;h)\xi).
\]

The corresponding relative changes in modes \(0{:}32\) are only

\[
    1.56\times10^{-4},\qquad
    1.61\times10^{-4},\qquad
    1.86\times10^{-4}.
\]

Thus the correction removes the unresolved tail without materially changing
the resolved initial condition.

## Dynamic check

A paired four-case GL2 rollout through \(T=20\) was run for:

- the narrow, steep corner \(h=0.01,\ a/h=0.35\);
- the broad main-dataset corner \(h=0.30,\ a/h=0.35\);
- the steep-Tanaka corner \(h=0.35,\ a/h=0.45\); and
- exact May-v2 case 31.

Each case has a piecewise-linear and a tangent-Hermite arm, with identical
projection, time step, eight substeps per saved interval, four fixed-point
sweeps, and order-six pad-eight Dirichlet--Neumann evaluation.

All 25 dynamic gates pass.  Every saved \(\eta\), \(\xi\), and
\(G(\eta;h)\xi\) value is finite.  The largest Hamiltonian drift among the
eight arms is

\[
    9.76\times10^{-9},
\]

and the minimum water-column fraction is

\[
    \min_{x,t}\frac{h+\eta(x,t)}{h}=0.9844.
\]

At \(T=20\), the relative changes between the linear and Hermite arms in modes
\(0{:}32\) are:

| case | \(\eta\) | \(\xi\) |
| --- | ---: | ---: |
| narrow, steep | \(3.51\times10^{-4}\) | \(1.84\times10^{-4}\) |
| broad main corner | \(1.49\times10^{-4}\) | \(8.83\times10^{-5}\) |
| steep-Tanaka corner | \(1.42\times10^{-4}\) | \(8.32\times10^{-5}\) |
| exact case 31 | \(1.55\times10^{-4}\) | \(1.67\times10^{-4}\) |

For the narrow \(h=0.01\) wave, modes \(80{:}128\) contain genuine resolved
wave structure, and Hermite leaves their norms essentially unchanged.  For
the broad cases, where the old interpolation tail dominates that band, the
Hermite construction reduces it by roughly two orders of magnitude.  For
case 31 at \(T=20\), the mode-\(80{:}128\) norm of
\(G(\eta;h)\xi\) is \(2.95\times10^{-6}\) for the linear arm and
\(3.01\times10^{-8}\) for the Hermite arm.

## Decision

Accept `periodic_tangent_hermite_tan_theta_v1` for new Tanaka data.  The
correction removes the interpolation artifact, passes the fixed-band grid and
collocation comparisons, and leaves the resolved dynamics and invariants
unchanged at the measured scale.

Keep the current 257 Tanaka knots.  The 513-knot comparison is already well
inside the declared \(10^{-3}\) and \(2\times10^{-3}\) fixed-band budgets, so
increasing the collocation count is not needed for this correction.

The accepted machine-readable results are:

- [`static_summary.json`](../outputs/tanaka_tangent_stratified_panel_20260723/static_summary.json);
- [`rollout_summary.json`](../outputs/tanaka_tangent_stratified_panel_20260723/rollout_summary.json).

## Figure

The exact case-31 comparison at the cancellation time \(t=76.32\) is
[`before_after_t76p32.png`](../outputs/tanaka_case31_tangent_resampling_trial_20260723/before_after_t76p32.png).
The surface is visually unchanged, while the sign count in
\(G(\eta;h)\xi\) falls from \(162\) to \(4\).
