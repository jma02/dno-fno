# Case-31 tangent-aware pre-smoothing trial

Date: 2026-07-23

Status: complete

## Question

Does replacing the piecewise-linear placement of a Tanaka profile by a
tangent-aware smooth reconstruction remove the artificial oscillations in
\(G(\eta;h)\xi\) without materially changing the intended solitary waves or
their Gauss--Legendre evolution?

This is a pre-smoothing experiment in the operational sense: the initial
profile is reconstructed smoothly before it is supplied to the Fourier
solver. It is not a low-pass filter. There is no smoothing strength or fitted
spectral cutoff.

## Exact initial condition

The trial uses case 31 from batch 0 of
`data/tanaka_2_adaptive_g0.npz`. Its pre-storage double-precision depth is

\[
h=0.26861433760407505.
\]

The two component waves are:

| \(a/h\) | center | direction |
| ---: | ---: | ---: |
| \(0.05548354495289499\) | \(3.5548496920089176\) | \(-1\) |
| \(0.053531413616445936\) | \(5.032703928715591\) | \(+1\) |

The archived maximum sign-count row is the 77th saved sample of this case:
dense rollout index 954, \(t=76.32\).

## Paired construction

The control arm reproduces the May-v2 procedure:

1. solve each Tanaka profile on the default transformed collocation grid;
2. place it on an \(8N\) grid with piecewise-linear interpolation;
3. Fourier-truncate to \(N=1024\);
4. construct \(\xi\), sum the two crests, and project to \(|k|\leq128\).

The trial arm changes only step 2. At every Tanaka profile knot, the conformal
solution supplies

\[
\frac{d\eta}{dx}=\tan\theta.
\]

The trial uses cubic Hermite reconstruction through the same profile values
with these slopes, followed by the same three-copy periodic placement,
Fourier truncation, construction of \(\xi\), and \(|k|\leq128\) projection.

Both arms are then advanced together with the production reference protocol:
order-six Craig--Sulem DNO, padding factor eight, two-stage fourth-order
Gauss--Legendre integrating-factor method, four fixed-point sweeps,
inner step \(0.01\), output interval \(0.08\), and \(T=200\).

## Comparisons fixed before running

The trial will report:

1. reproduction error between the control initial condition and the archived
   frame at \(t=0\);
2. full-state and low-mode differences between the two initial conditions;
3. mode-band norms of \(\eta\), \(\xi\), and \(G(\eta;h)\xi\);
4. the single-crest traveling-wave residual
   \(G^{(6)}(\eta;h)\xi+c_{\rm signed}\eta_x\);
5. the full time history of the historical sign-transition count;
6. absolute high-band norms, rather than only their ratios to the cancelling
   low modes;
7. Hamiltonian drift and minimum water-column height;
8. fields and spectra at the original offending time \(t=76.32\).

The reconstruction will be considered promising only if it reduces the
artificial retained-band floor while preserving the low modes, the intended
wave amplitude, the traveling-wave residual, and the long-time invariants.
The sign count alone is not an acceptance criterion because physical
cancellation can make any relative sign statistic ill-conditioned.

## Execution update

A \(T=0.16\) execution smoke passed before the full launch. The control
reproduces the archived frame-zero float32 arrays bit-for-bit for
\(\eta\), \(\xi\), and \(G(\eta;h)\xi\). Relative differences between the
tangent-Hermite and control initial states are

\[
1.6265\times10^{-4}\quad\hbox{in }\eta,
\qquad
1.6082\times10^{-4}\quad\hbox{in }\xi.
\]

The mode-80--128 coefficient norm of the initial surface changes from

\[
3.51994\times10^{-7}
\quad\hbox{to}\quad
5.19705\times10^{-9},
\]

a factor of \(67.7\). The corresponding norm in \(G(\eta;h)\xi\) changes from
\(6.12371\times10^{-7}\) to \(1.26482\times10^{-8}\), a factor of \(48.4\).
The single-crest traveling-wave residuals improve slightly, from
\(0.02425,0.02392\) to \(0.02304,0.02268\). The full paired CPU rollout was
then launched with the unchanged production protocol.

## Full-rollout result

The control arm reproduces not only the initial archive row but also the
offending \(t=76.32\) row:

| field | relative \(L^2\) difference from archive at \(t=76.32\) |
| --- | ---: |
| \(\eta\) | \(2.22\times10^{-8}\) |
| \(\xi\) | \(2.53\times10^{-8}\) |
| \(G(\eta;h)\xi\) | \(2.33\times10^{-8}\) |

Hence the paired control follows the original case to storage precision.

The sign-transition histories separate completely:

| quantity | piecewise-linear control | tangent Hermite |
| --- | ---: | ---: |
| maximum count | \(192\) | \(6\) |
| time of maximum | \(4.56\) | \(16.48\) |
| count at \(t=76.32\) | \(162\) | \(4\) |
| dense frames with count \(>10\) | \(1597/2501\) | \(0/2501\) |

This is not merely a change in the relative diagnostic. At the original
offending time, the absolute mode-80--128 coefficient norms are:

| field | piecewise-linear control | tangent Hermite | reduction factor |
| --- | ---: | ---: | ---: |
| \(\eta\) | \(2.45763\times10^{-7}\) | \(4.06797\times10^{-9}\) | \(60.4\) |
| \(\xi\) | \(2.73923\times10^{-8}\) | \(3.69571\times10^{-10}\) | \(74.1\) |
| \(G(\eta;h)\xi\) | \(2.73909\times10^{-6}\) | \(3.51520\times10^{-8}\) | \(77.9\) |

The low-frequency physical evolution is essentially unchanged. The relative
surface difference between the arms is \(1.65\times10^{-4}\) at \(t=76.32\)
and \(1.76\times10^{-4}\) at \(T=200\); the optimal integer translation is
zero at both times. At \(T=200\), the relative differences are
\(2.94\times10^{-4}\) in \(\xi\) and \(9.56\times10^{-4}\) in
\(G(\eta;h)\xi\). The apparently larger relative \(G(\eta;h)\xi\) difference
of \(3.33\times10^{-2}\) at \(t=76.32\) occurs because that field nearly
cancels there; the absolute high-band comparison above is the meaningful
quantity.

The reconstruction does not introduce detectable dissipation:

| invariant diagnostic | piecewise-linear control | tangent Hermite |
| --- | ---: | ---: |
| maximum relative Hamiltonian drift | \(2.3801\times10^{-12}\) | \(2.3865\times10^{-12}\) |
| minimum \(h+\eta\) | \(0.270342\) | \(0.270340\) |

The paired CPU wall time was \(49{:}53\): \(47{:}58\) for the GL rollout,
\(1{:}42\) for dense DNO postprocessing, and \(0{:}13\) for setup.

## Decision

For this exact case, tangent-aware pre-smoothing removes the numerical
oscillation without damping the GL evolution or changing the resolved
surface dynamics materially. The result supports replacing piecewise-linear
Tanaka placement in the generator.

It does not yet justify immediate dataset regeneration. The next implementation
step is to move the tested Hermite evaluator into the Tanaka construction,
add its knot/symmetry/periodicity tests, and repeat the comparison on a
depth-and-steepness-stratified panel. The existing 257 nonnegative Tanaka
collocation nodes are sufficient for this intervention; increasing the
collocation count was not the source of the improvement.

Artifacts:

- `outputs/tanaka_case31_tangent_resampling_trial_20260723/summary.json`
- `outputs/tanaka_case31_tangent_resampling_trial_20260723/diagnostics.png`
- `outputs/tanaka_case31_tangent_resampling_trial_20260723/diagnostics.npz`
- `outputs/tanaka_case31_tangent_resampling_trial_20260723/paired_trajectories.npz`
