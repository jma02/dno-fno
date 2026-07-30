# Fixed-band refinement as the replacement data-quality rule

Date: 2026-07-22

## Decision

The historical sign-transition count should not be the primary data-quality
criterion in the manuscript.  It remains part of the exact historical record of the
April Tanaka pilot corpus.  The final methodology should instead separate:

1. prevention of nonperiodic embedding by construction;
2. independent fixed-band grid-consistency of the initial state and DNO label;
3. finiteness, bottom-clearance, and Hamiltonian-drift checks during rollout.

This distinction also avoids an inaccurate claim: the final v9 Tanaka data were
regenerated with a corrected periodic builder and were not obtained by applying
the historical sign filter to the old archive.

## Three-grid DNO discrepancy

Let \(\vartheta\) be the latent specification of one initial condition.  Build
the same specification independently in float64 on
\(n_j=2^jN\), \(j=0,1,2\).  Do not Fourier-interpolate the \(N\)-grid state;
that would hide the construction error under investigation.  At a fixed
physical cutoff \(K\), define

\[
q_j^K=P_K\left[G_{n_j}^{(6)}(P_K\eta_{n_j})(P_K\xi_{n_j})\right].
\]

Let \(Q_j^K\) contain the consistently normalized Fourier coefficients for
\(|k|\le K\).  The primary audit statistic is

\[
\Delta_K(\vartheta)
=\max_{0\le r<s\le2}
\frac{\|Q_r^K-Q_s^K\|_{\ell^2}}
     {\|Q_s^K\|_{\ell^2}+10^{-30}}.
\]

The proposed relative label budget is

\[
\Delta_K(\vartheta)\le 10^{-3}.
\]

The maximum over all three pairs is deliberately simpler than imposing a
fitted convergence-rate threshold.  The observed contraction ratio remains a
diagnostic.  Some acceptable historical controls have small absolute defects
but slow ratios, so a hard requirement such as
\(d_{2N,4N}/d_{N,2N}\le1/2\) would reject converged states unnecessarily.

This statistic measures only grid consistency of the fixed order-six target
on the delivered band.  It does not estimate Craig--Sulem order-truncation
error, prove physical admissibility, or validate the time integrator.

## Completed \(N,2N,4N\) calibration

The CPU calibration uses \(N=1024\), \(2N=2048\), \(4N=4096\), and the
historical delivered band \(K=341\).  The deterministic panel contains:

- all 12 reconstructable historical frame-zero sign rejections;
- four clean interior historical controls;
- the same 12 rejected crest specifications rebuilt with the corrected
  three-copy periodic construction.

The maximum three-grid relative \(L^2\) discrepancy separates the matched
populations by more than two orders of magnitude:

| stratum | number | range relevant to decision |
| --- | ---: | ---: |
| corrected matched counterfactuals | 12 | \(\Delta_K\le1.10\times10^{-5}\) |
| clean historical controls | 4 | \(\Delta_K\le1.64\times10^{-4}\) |
| clipped historical constructions | 12 | \(\Delta_K\ge2.33\times10^{-2}\) |

The decision is identical at thresholds
\(3\times10^{-4}\), \(10^{-3}\), and \(3\times10^{-3}\).  There are no
disagreements between the provisional \(N\)-versus-\(2N\) decision and the
direct \(N\)-versus-\(4N\) decision on the 28 cases.

For the matched periodic counterfactuals, median relative DNO discrepancies
are

\[
d_{N,2N}=7.03\times10^{-6},
\qquad
d_{2N,4N}=2.18\times10^{-6}.
\]

For the clipped construction they are \(0.0827\) and \(0.147\), respectively.
The increasing second discrepancy in the clipped cases confirms that the
two-grid result was not an accidental cancellation against an unconverged
\(2N\) reference.

## Cheaper per-trajectory screen

Evaluating the order-six DNO at three resolutions for every stored row is
unnecessary.  A candidate trajectory can first be screened once, before
rollout, using only independently constructed initial states:

\[
\Delta_{\mathrm{IC},K}
=\max_{u\in\{\eta,\xi\}}
\max_{0\le r<s\le2}
\frac{\|U_{u,r}^K-U_{u,s}^K\|_{\ell^2}}
     {\|U_{u,s}^K\|_{\ell^2}+10^{-30}}.
\]

The expensive DNO discrepancy can then be evaluated on a predeclared,
source-stratified validation panel.  On the complete 254-case historical
batch-zero audit, the two-grid \(\eta\)-construction defect at a retrospective
threshold \(4\times10^{-4}\) detects all 41 cases with DNO defect above
\(10^{-3}\) and flags 4 of the 213 passing cases.  This supports the state
criterion but does not yet freeze its threshold; that requires a cross-family
\(N,2N,4N\) calibration.

## Why not use a more elaborate one-grid proxy?

- Endpoint mismatch directly detects the old seam, but it depends on the
  chosen coordinate cut and can flag a valid wave crossing that cut.
- A spectral-tail threshold works extremely well inside the old Tanaka family
  but rejects resolved current multi-crest and Stokes states.  It measures
  spectral morphology, not construction convergence.
- Pad-8 versus pad-16 agreement tests the pseudospectral product evaluation.
  It can agree even when independently generated \(N\)- and \(2N\)-grid inputs
  disagree, so it cannot diagnose the old embedding defect.
- Hamiltonian drift tests the subsequent discrete trajectory.  A corrupted
  initial state can still evolve consistently under the same discrete
  equations.

## Remaining publication requirement

The completed matched Tanaka calibration is sufficient to replace the sign
count in the explanation of the historical generator revision.  A claim that
the entire final corpus was filtered by \(\Delta_K\) would be false.  Before
calling it a corpus-wide acceptance rule, apply the state screen to every
candidate trajectory and run the DNO screen on a predeclared stratified panel
from every physical family at its delivered cutoff.  Until then, the correct
term in the manuscript is **fixed-band resolution audit**.

Reproduction:

```bash
taskset -c 32-47 env JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 \
  uv run python scripts/calibrate_signflip_refinement_n2n4n.py
```

Primary artifact:

- `notes/signflip_refinement_n2n4n_calibration_20260722.json`
