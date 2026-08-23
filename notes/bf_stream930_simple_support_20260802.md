# Simple Benjamin--Feir support suggested by stream 930

## Scope and notation

This note only reads the four immutable proposal/result pairs from validation
stream 930.  It does not change the sampler or declare a final population.
The first batch contains one proposal from each of the 66 equally weighted
integer pairs \((n_c,\Delta n)\).  Later batches retry only categories that
failed, so the first 66 proposals are the useful balanced diagnostic; the
complete 96-proposal prefix is outcome-adaptive and must not be treated as an
i.i.d. binomial sample.

Write

\[
  \epsilon_c=k_c a_c,
  \qquad
  \rho=\frac{a_s}{a_c},
  \qquad
  \beta=
  \frac{\Delta n/n_c}{2\sqrt{2}\,\epsilon_c}.
\]

Here \(\rho\) is the first-harmonic amplitude of each sideband relative to the
carrier.  The leading deep-water NLS instability condition is
\(0<\beta<1\).  In a fixed cell, the current sampler is

\[
  \epsilon_c\sim
  \operatorname{Unif}(a_{n_c,\Delta n},0.13],
  \quad
  a_{n_c,\Delta n}
  =\max\!\left(0.05,
    \frac{\Delta n}{2\sqrt{2}\,n_c}\right),
  \qquad
  \rho\sim\operatorname{Unif}[0.05,0.20].
\]

## What the 96 calculations say

The first balanced batch accepted 46 of 66 proposals.  Across all retries,
the durable prefix accepted 65 of 96 and rejected 31; 29 rejections became
nonfinite or unsolved and two remained finite but failed the numerical-health
test.

The entries below are ``accepted / attempted''.  The left count is the
balanced first batch and the parenthesized count uses all 96 attempts.

| Variable range | Accepted / attempted |
|---|---:|
| \(0.05\leq\epsilon_c<0.09\) | 12/12 (25/25) |
| \(0.09\leq\epsilon_c<0.11\) | 11/18 (17/28) |
| \(0.11\leq\epsilon_c\leq0.13\) | 23/36 (23/43) |
| \(0.05\leq\rho<0.10\) | 22/24 (29/34) |
| \(0.10\leq\rho<0.15\) | 9/17 (15/28) |
| \(0.15\leq\rho\leq0.20\) | 15/25 (21/34) |
| \(0<\beta<0.4\) | 4/15 (8/23) |
| \(0.4\leq\beta<0.8\) | 14/23 (27/43) |
| \(0.8\leq\beta<1\) | 28/28 (30/30) |
| \(4\leq n_c\leq9\) | 7/11 (11/17) |
| \(10\leq n_c\leq14\) | 13/20 (19/33) |
| \(15\leq n_c\leq20\) | 26/35 (35/46) |
| \(\Delta n=1\) | 7/17 (17/32) |
| \(\Delta n=2\) | 7/15 (14/26) |
| \(\Delta n\geq3\) | 32/34 (34/38) |

Thus neither \(n_c\) nor \(\rho\) supplies a safe monotone cutoff.  In
particular, five proposals with \(\rho\leq0.10\) still failed in the complete
prefix, including a nonfinite case at \(\rho=0.0599\).  Small sideband
amplitude does not rescue an arbitrarily steep carrier.  The two clearest
coordinates in this smoke are \(\epsilon_c\) and the normalized sideband
location \(\beta\).

## Elementary candidate restrictions

The theoretical retained fraction below is the exact probability under the
current law: average the retained conditional interval length over the 66
equally weighted cells and multiply by the retained \(\rho\)-interval
fraction.  ``Cells'' counts mode-pair categories having nonempty support.
The interval is the ordinary two-sided 95% Wilson interval computed from the
balanced retained proposals.

| Candidate rule | Retained probability | Cells | Balanced accepted / rejected | Wilson pass interval | All attempts accepted / rejected |
|---|---:|---:|---:|---:|---:|
| no restriction | 100.00% | 66 | 46 / 20 | 57.8--79.4% | 65 / 31 |
| \(\epsilon_c\leq0.09\) | 26.05% | 45 | 12 / 0 | 75.8--100% | 25 / 0 |
| \(\epsilon_c\leq0.10\) | 37.35% | 49 | 16 / 6 | 51.8--86.8% | 33 / 6 |
| \(\rho\leq0.10\) | 33.33% | 66 | 22 / 2 | 74.2--97.7% | 29 / 5 |
| \(\epsilon_c\leq0.11\) and \(\rho\leq0.10\) | 17.18% | 55 | 10 / 0 | 72.2--100% | 17 / 0 |
| \(\beta\geq0.80\) | 42.67% | 51 | 28 / 0 | 87.9--100% | 30 / 0 |
| \(\epsilon_c\leq0.09\) or \(\beta\geq0.80\) | 60.88% | 66 | 34 / 0 | 89.8--100% | 48 / 0 |

The Wilson intervals are descriptive, not confirmatory: the restrictions were
chosen after looking at stream 930, and success probabilities can differ by
cell.  The all-attempt counts have no binomial interpretation because retries
were triggered by earlier failures.

## Recommendation to test

If the shortest possible population definition is the overriding goal, use

\[
  0.05\leq\epsilon_c\leq0.09,
  \qquad 0.05\leq\rho\leq0.20,
\]

and recompute the feasible mode pairs.  This is easy to explain as a moderate
carrier-steepness population, but it leaves only 45 mode-pair categories and
retains about 26% of the current law.

A less wasteful candidate, still using only the standard instability-band
coordinate, is

\[
  \boxed{\epsilon_c\leq0.09
  \quad\text{or}\quad
  \frac{\Delta n/n_c}{2\sqrt{2}\,\epsilon_c}\geq0.80.}
\]

In words: a carrier may be steeper than 0.09 only when the sidebands lie in
the outer 20% of the leading instability band.  This removes the observed
high-steepness/interior-band failure region, retains about 61% of the original
population, leaves \(\rho\) unchanged, and keeps all 66 integer categories
nonempty.  The closest rejected proposal had \(\epsilon_c=0.094906\), and the
largest rejected value of \(\beta\) was 0.744762, so the proposed decimal
thresholds are not placed at floating-point distance from an observed
failure.

This boxed rule is only a hypothesis.  Before changing the population, freeze
it and run a fresh full-horizon validation.  A useful test consists of at
least one independently sampled case in every one of the 66 cells, plus a
boundary case near the largest allowed \(\epsilon_c\) with \(\rho=0.20\).
The already designed fixed-capacity probe of \((n_c,\Delta n)=(12,2)\) samples
the old full steepness interval; it diagnoses that category but does not by
itself validate either restriction above.

## A single NLS focused-steepness proxy

The preceding disjunction can be replaced by one physically motivated scalar.
Define

\[
  \epsilon_f
  =\epsilon_c\left(1+2\sqrt{1-\beta^2}\right)
  =\epsilon_c+
    \sqrt{4\epsilon_c^2-\frac12(\Delta n/n_c)^2}.
\]

This is the carrier steepness plus the leading-NLS sideband contribution at a
focused phase.  We use it only as a screening proxy; NLS does not prove that a
full water-wave trajectory satisfying an \(\epsilon_f\) bound remains regular
or numerically converged.

For a cell, put

\[
  \ell=\frac{\Delta n}{2\sqrt{2}\,n_c}.
\]

Since \(\epsilon_f=\epsilon_c+2\sqrt{\epsilon_c^2-\ell^2}\) is increasing in
\(\epsilon_c\), the condition \(\epsilon_f\leq F\) is exactly the cellwise
upper bound

\[
  \epsilon_c\leq
  u_F(\ell)
  =\frac{-F+2\sqrt{F^2+3\ell^2}}{3}.
\]

The actual upper endpoint is \(\min(0.13,u_F)\).  This makes the retained
probability analytic under the existing cellwise-uniform sampling law.

| Proxy bound | Retained probability | Nonempty cells | Cellwise upper endpoints | Balanced accepted / rejected | Balanced Wilson interval | All attempts accepted / rejected |
|---|---:|---:|---:|---:|---:|---:|
| \(\epsilon_f\leq0.24\) | 56.86% | 66 | 0.08130--0.13000 | 30 / 0 | 88.6--100% | 44 / 0 |
| \(\epsilon_f\leq0.25\) | 61.21% | 66 | 0.08458--0.13000 | 32 / 0 | 89.3--100% | 47 / 0 |
| \(\epsilon_f\leq0.26\) | 65.65% | 66 | 0.08786--0.13000 | 34 / 0 | 89.8--100% | 52 / 0 |

Before clipping at the existing 0.13 ceiling, the corresponding upper-endpoint
ranges are 0.08130--0.13826, 0.08458--0.13986, and 0.08786--0.14155.  Nine
cells therefore retain the full upper endpoint 0.13 under each candidate.

Every rejected stream-930 proposal has \(\epsilon_f\geq0.263815\).  Thus all
three displayed bounds remove every observed failure.  The value 0.26 leaves
only 0.003815 between its cutoff and the smallest observed failure.  The value
0.25 leaves a more meaningful 0.013815 gap while retaining essentially the
same population mass as the earlier disjunction:

| Rule | Retained probability | Cells | Balanced accepted / rejected | All attempts accepted / rejected |
|---|---:|---:|---:|---:|
| \(\epsilon_c\leq0.09\) or \(\beta\geq0.80\) | 60.88% | 66 | 34 / 0 | 48 / 0 |
| \(\epsilon_f\leq0.25\) | 61.21% | 66 | 32 / 0 | 47 / 0 |

The \(\epsilon_f\leq0.25\) rule is therefore the cleaner next hypothesis:
one inequality replaces a piecewise ``or'' rule, its coordinate has a direct
leading-NLS interpretation, it leaves the complete \(\rho\in[0.05,0.20]\)
range untouched, and its theoretical retained mass differs from the
disjunction by only 0.33 percentage points.  It remains a post-hoc hypothesis
and requires a fresh predeclared full-horizon validation before adoption.

### A non-decimal reference value

Instead of choosing 0.24 or 0.25 directly, one can set a reference carrier
steepness \(\epsilon_\star=0.10\) and evaluate the proxy at the maximally
unstable leading-NLS band coordinate \(\beta=1/\sqrt{2}\).  This gives

\[
  F_0
  =\epsilon_\star
    \left(1+2\sqrt{1-\frac12}\right)
  =\frac{1+\sqrt{2}}{10}
  =0.24142135623730948.
\]

The exact stream-930 comparison is

| Proxy bound | Retained probability | Cells | Cellwise upper endpoints | Balanced accepted / rejected | All attempts accepted / rejected |
|---|---:|---:|---:|---:|---:|
| \(F=0.24\) | 56.8602% | 66 | 0.081297--0.130000 | 30 / 0 | 44 / 0 |
| \(F=F_0\) | 57.4726% | 66 | 0.081763--0.130000 | 30 / 0 | 44 / 0 |
| \(F=0.25\) | 61.2090% | 66 | 0.084579--0.130000 | 32 / 0 | 47 / 0 |

For \(F_0\), the unclipped per-cell upper endpoints range from
0.081763 to 0.138484; nine cells reach the existing ceiling 0.13.  The
balanced 30/30 successes have the descriptive 95% Wilson pass interval
88.6--100%.  The complete 44/44 count has no ordinary binomial interpretation
because later attempts were outcome-adaptive.

The smallest failed value remains \(\epsilon_f=0.263815\), so \(F_0\) has a
gap of 0.022394 to the nearest observed failure.  It retains 0.61 percentage
points more probability than the decimal value 0.24 and 3.74 percentage
points less than 0.25, with the same observed retained cases as 0.24.
Consequently \(F_0\) has the cleanest definition: it says that the NLS
focused-steepness proxy may not exceed the value attained by
\(\epsilon_c=0.10\) at the maximally unstable sideband location.  The
reference value is an explainable modeling choice, not an NLS regularity
theorem or a literature-derived breaking boundary.

## Cross-check against the revision-2 archive

The preceding candidate choice was suggested by the then-current revision-3
stream 930.  A separate read-only calculation checked it against every durable
revision-2
Benjamin--Feir proposal/result transaction. These older calculations used

\[
  M=6,\qquad K=128,\qquad \Delta t=0.01,\qquad T=200,
\]

with four GL2 fixed-point iterations and a uniformly sampled common
sideband phase. They therefore cannot be pooled statistically with stream
930, which uses the fixed JCP09 phase, internal \(M=4,K=256\), and a horizon
of 100 carrier periods. They are nevertheless an independent numerical
cross-check of the same scalar initial-condition parameters.

There are 6,570 result-bearing revision-2 attempts: 4,522 attempts in the
three completed 2,048/1,024/1,024 train/validation/test roots and 2,048
additional atomically committed attempts in the interrupted continuation of
the training split. Their outcomes under four nearby proxy bounds are

| Proxy bound | Accepted / retained attempts | Nonfinite retained failures | Represented cells |
|---|---:|---:|---:|
| \(\epsilon_f\leq0.24\) | 3,700 / 3,700 | 0 | 66 |
| \(\epsilon_f\leq F_0\) | 3,743 / 3,743 | 0 | 66 |
| \(\epsilon_f\leq0.25\) | 3,965 / 3,968 | 0 | 66 |
| \(\epsilon_f\leq0.26\) | 4,242 / 4,248 | 0 | 66 |

The three retained failures at 0.25 and the six at 0.26 are finite
trajectories whose largest recorded GL2 residuals lie between
\(1.005\times10^{-8}\) and \(2.922\times10^{-8}\), just above the declared
\(10^{-8}\) tolerance. Thus 0.25 and 0.26 do not admit an archived NaN in
this older calculation, but they do lose the exact numerical-acceptance
property. In contrast, both 0.24 and \(F_0\) retain no failed attempt.

The 4,096 accepted trajectories in the three complete revision-2 roots also
have stored fields at 200 retained times. Scanning every retained frame and
then selecting the 2,577 trajectories satisfying \(\epsilon_f\leq F_0\)
gives

\[
  \max_t \frac{|H(t)-H(0)|}{|H(0)|}
    =4.330\times10^{-4},
  \qquad
  Q_{0.99}=4.626\times10^{-6}.
\]

None exceeds the current \(10^{-3}\) Hamiltonian threshold. For the
fraction of delivered \(G(\eta)\xi\) Fourier energy in modes 96--128, the
maximum is 0.04817 and the 99th percentile is
\(8.505\times10^{-4}\); eight of 2,577 trajectories exceed 0.01. These
high-band figures are diagnostic rather than an acceptance rule, but the
proxy removes the much larger revision-2 tail: before restriction, the
maximum high-band fraction is 0.12447 and 107 of 4,096 accepted trajectories
exceed the current Hamiltonian threshold.

Among the candidates tested here, \(F_0\) is therefore the best next
hypothesis. It has a pre-numerical definition, preserves all 66 integer
cells, retains 57.47% of the declared sampling law, passes every retained
attempt in both numerical contracts, and is slightly less restrictive than
0.24. This is retrospective screening evidence, not final validation. The
next test must sample directly from the conditioned \(\epsilon_f\leq F_0\)
population in a fresh revision-3 stream and run the exact current full-horizon
contract.

## Proposed paper-dataset regime

For the new dataset, the cleanest hypothesis to validate is

\[
  0<\beta<1,
  \qquad
  \epsilon_c\left(1+2\sqrt{1-\beta^2}\right)
    \leq \frac{1+\sqrt2}{10},
  \qquad
  0.05\leq\rho\leq0.10.
\]

This uses only the three ordinary Benjamin--Feir parameters already present
in the construction.  It introduces no property measured after a rollout.
Conditional on an integer pair \((n_c,\Delta n)\), we will draw
\(\epsilon_c\) uniformly between its instability-band lower endpoint and
the upper endpoint defined by the displayed inequality.  We will draw
\(\rho\) uniformly on \([0.05,0.10]\), and retain the uniform translation.
All 66 integer pairs remain possible and continue to receive equal numbers of
accepted dataset cases.

The amplification factor is not fitted to stream 930.  In the standard
Akhmediev-breather notation, the parameter relation

\[
  a_{\rm AB}=\frac{1-\beta^2}{2}
\]

gives the focused envelope amplification
\(1+2\sqrt{2a_{\rm AB}}=1+2\sqrt{1-\beta^2}\).  Andrade and Stuhlmeier state
the relation between sideband location and \(a_{\rm AB}\) explicitly in their
equation (5.9).  It follows the exact Akhmediev family.  The original exact
solution and a convenient form of its focused profile are given by
Akhmediev, Eleonskii, and Kulagin and by Akhmediev, Ankiewicz, and Taki:

- https://doi.org/10.1007/BF01017105
- https://doi.org/10.1016/j.physleta.2008.12.036
- https://doi.org/10.1017/jfm.2023.96

The right-hand side is anchored by an independent moderate recurrence case,
not by the smallest failed value in our smoke.  Yang and Liu use
\(\epsilon_c=0.10\), five-percent sidebands, relative phase \(-\pi/4\), and
the maximum-growth frequency coordinate
\(\Delta\omega/(\omega_c\epsilon_c)=1\).  Under the leading deep-water
narrowband relation
\(\Delta k/k_c\simeq2\Delta\omega/\omega_c\), their coordinate corresponds
to \(\beta\simeq1/\sqrt2\) in our symmetric-wavenumber normalization.  We
therefore use \((1+\sqrt2)/10\) as the NLS-motivated project anchor; it is not
an exact identity between the two finite-amplitude constructions.  Their
calculation exhibits repeated focusing and recurrence over a
1000-carrier-period simulation:

- https://doi.org/10.1017/jfm.2024.604, Section 4.3.1.

The sideband interval joins that five-percent seed to the ten-percent seed in
Xu and Guyenne's equation-(33) calculation:

- https://doi.org/10.1016/j.jcp.2009.08.015, Section 4.2.3.

Uniform sampling between these two published amplitudes is our population
choice; neither paper asserts that every intermediate full-Euler trajectory
is regular.  We do not retain the former upper value \(\rho=0.20\): it adds
finite-seed diversity, but is harder to describe as a weak sideband
perturbation.  Finite-seed NLS calculations likewise show that the seed
changes the focusing time and that sufficiently large seeds depart from the
ideal Akhmediev orbit (Chin, Ashour, and Belić,
https://doi.org/10.1103/PhysRevE.92.063202).

The quantity \(\epsilon_f\) is only an NLS focused-*envelope* proxy.  It is
not the maximum physical slope, a wave-breaking criterion, or a regularity
bound.  Our initial condition is a finite three-mode Stokes--Airy state, not
an exact time slice of the Akhmediev solution; the latter contains a full
sideband ladder at focus.  Bound harmonics, finite sideband amplitude,
higher-order water-wave effects, and the finite horizon can all change the
realized maximum.  For those reasons the complete-trajectory finiteness,
stage-residual, water-column, and Hamiltonian checks remain part of dataset
generation.  At this decision point, a fresh one-draw-per-cell
current-contract calculation was the necessary final test before this support
could replace the previous law.  The later exact-source stream-935 result is
recorded at the end of this note.

## Fresh conditioned validation

Validation stream 934 fixed its complete parameter plan before numerical
execution.  Parameter-only rejection sampling examined 369 independent
original-law draws and selected one draw in each of the 66 cells satisfying
the proposed focused-steepness and sideband-ratio restrictions.  All 66
selected trajectories were then advanced together under the unchanged
revision-3 contract

\[
  K=256,\qquad M=4,\qquad \Delta t=0.01,
\]

for 100 carrier periods.  Numerical outcomes caused neither replacement nor
early stopping.  The predeclared all-pass decision succeeded:

\[
  66\ \text{attempted},\qquad
  66\ \text{accepted},\qquad
  0\ \text{rejected}.
\]

Every stored value of \(\eta\), \(\xi\), and \(G(\eta)\xi\) is finite.  The
largest internal Hamiltonian drift was \(5.38713\times10^{-5}\), the largest
GL2 stage residual was \(9.999984\times10^{-9}\), and the smallest internal
water column was 4.958158.  All 66 canonical cells occur exactly once, and
all 13,200 retained frames have authenticated proposal and trajectory
ownership.  The result therefore supports adopting the proposed population
law.  It does not change the interpretation of \(\epsilon_f\): the scalar
remains a leading-NLS screening proxy, and the full-trajectory numerical
health tests remain mandatory.

## Exact-source production-law gate (2026-08-02)

The subsequent stream-935 calculation drew directly from the implemented
revision-4 population law rather than conditioning draws from an older law.
It attempted and accepted the first proposal in each of the 66 categories,
with no rejection, and retained 13,200 rows.  Every current-source,
specification, clock, retained-time, ownership, and trajectory-health check
passed.  The maximum GL2 stage residual was
\(9.999967276\times10^{-9}\), the maximum internal Hamiltonian drift was
\(2.952009333\times10^{-5}\), and the minimum internal water column was
\(4.960597932\).  The audited source fingerprint was
`d1ba68f08a5316d0b9ce0e399d48896d766044615504e461592eaee7ff3325ce`.

The fixed release audit is
`outputs/nondataset_bf_revision4_jonswap_revision3_category_smoke_20260802/category_smoke_release_audit_v1.json`
with SHA-256
`62fb5fe3b6be7df0d21784b1ce590c44cc484bf5fdf84ad6097fbed6f655f684`.
Its `release_gate_passed` value is true.  This closes adoption of the exact
revision-4 population law and supplies category-coverage and numerical-health
evidence; it is not a population rejection-rate estimate.
