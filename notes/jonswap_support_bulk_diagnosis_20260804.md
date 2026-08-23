> **Historical / superseded diagnosis.** This note analyzes the revision-3
> absolute-band prefix and its retrospective RMS-slope proposal; that proposal
> is not the current JONSWAP/TMA support law.  The current family map is static
> Stokes revision 2, corrected Tanaka revision 3, Benjamin--Feir revision 4,
> and JONSWAP/TMA revision 4.  Use the
> [revision-4 relative-frequency decision](jonswap_defensible_support_review_20260804.md),
> [data-generation contract](../solver/gen_data/README.md), and
> [release-readiness map](paper_dataset_generation_readiness_20260726.md) for
> the adopted support and release path; the body below is retained only as
> evidence for why the revision-3 rule was rejected.

# JONSWAP support diagnosis from the revision-3 bulk prefix

## Question

The revision-3 sampler imposed

\[
\epsilon_p=\frac{k_pH_s}{2}\leq 0.08,
\]

but the first 1,600 durable bulk attempts contained 159 numerical
rejections (9.94%).  The one-case-per-category smoke was therefore too small
to characterize the population rejection rate.  This note asks whether a
different, input-only support condition better separates the numerically
resolved random seas from the unstable tail.

## A bandwidth-aware input quantity

Let \(\varpi_j\) denote the normalized energy fraction assigned by the actual
discrete JONSWAP/TMA construction to wavenumber \(k_j\), so that
\(\sum_j\varpi_j=1\).  The construction normalizes the surface variance to

\[
m_0=\left(\frac{H_s}{4}\right)^2.
\]

Consequently, averaging over the Fourier phases gives the root-mean-square
surface slope

\[
s_{\rm rms}
 = \left(\mathbb E_x|\eta_x|^2\right)^{1/2}
 = \frac{H_s}{4}
   \left(\sum_j k_j^2\varpi_j\right)^{1/2}.
\]

This is known before a trajectory is integrated.  Unlike \(\epsilon_p\), it
accounts for the complete resolved bandwidth: energy at wavenumber \(k_j\)
is weighted by \(k_j^2\).  For a monochromatic spectrum it reduces to
\(k_pH_s/4\).  Define also

\[
\epsilon_2=2s_{\rm rms}
 = \frac{H_s}{2}
   \left(\sum_j k_j^2\varpi_j\right)^{1/2}.
\]

## Retrospective result

Conditioning the 1,600-attempt revision-3 prefix on

\[
\boxed{s_{\rm rms}\leq 0.06}
\qquad\text{or, equivalently,}\qquad
\boxed{\epsilon_2\leq 0.12}
\]

retains 1,188 attempts (74.25%).  Six are rejected (0.505%; 95% Wilson
interval 0.232%--1.097%).  All six are finite GL2 residual misses; there are
no nonfinite adjustment or production failures.  The zero-nonfinite count
has a one-sided 95% binomial upper bound of 0.252%.  All 27 sample cells
remain represented, with at least 32 retained attempts per cell.

The result is stable under elementary splits:

| subset | retained | rejected | rate |
| --- | ---: | ---: | ---: |
| attempts 0--799 | 584 | 3 | 0.514% |
| attempts 800--1599 | 604 | 3 | 0.497% |
| even attempt index | 603 | 5 | 0.829% |
| odd attempt index | 585 | 1 | 0.171% |

The six remaining failures are distributed across the support rather than
forming one missing category: shallow 4/416, finite 1/383, deep 1/389;
left-moving 3/414, bidirectional 0/376, and right-moving 3/398.

The threshold also agrees with the older, wider-support revision-2 archive.
It retains 1,112 of 2,071 attempts and none of the 23 numerical failures.
A dense deterministic scan over the declared depth, peak-wavenumber, and
peak-enhancement boxes found

\[
\frac{(\sum_j k_j^2\varpi_j)^{1/2}}{k_p}\geq 1.54837.
\]

Thus \(\epsilon_2\leq0.12\) implies \(\epsilon_p\lesssim0.07751\) on the
implemented parameter boxes.  The new condition therefore subsumes the old
0.08 peak-steepness cutoff; retaining both would be redundant.

The observed breakpoint is reasonably sharp.  On the same 1,600 attempts,
\(\epsilon_2\leq0.125\) retains 1,249 with 8 residual-only rejections
(0.641%).  At \(0.13\), the first two nonfinite failures enter and the rate
rises to 15/1,304 (1.15%).

## Recommendation

Replace the peak-only rule by the single support condition

\[
s_{\rm rms}\leq0.06.
\]

This is simpler than cell-, direction-, or peak-enhancement-dependent bounds
and has a direct interpretation: exclude spectra whose expected surface
slope is too large for the fixed numerical method.  Treat 0.06 as a
numerically calibrated applicability boundary, not as a physical breaking
theorem.

Because the boundary was selected using the present bulk prefix, validate it
on a fresh deterministic stream before accepting a new dataset.  The fresh
run should keep the spectrum, nonlinear adjustment, spatial cutoff, GL2
method, and complete-trajectory health checks unchanged.  Do not post-filter
and mix the current accepted rows with the new population.
