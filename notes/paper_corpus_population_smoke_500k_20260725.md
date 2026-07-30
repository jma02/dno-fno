# Population-law smoke with 500,000 parameter specifications

Date: 2026-07-25.

## 1. Scope and result

This run sampled 500,000 parameter specifications: 125,000 from each of four
initial-condition families. It tested allocation counts, parameter support,
declared one-dimensional sampling laws, deterministic replay, and archive
integrity.

This was **not** a 500,000-trajectory data-generation run. Let \(\eta\) denote
surface elevation, let \(\xi\) denote surface potential, and let \(G(\eta)\)
denote the Dirichlet--Neumann operator (DNO) at surface \(\eta\). In
particular, the run did not construct all 500,000 surface fields, evaluate
\(G(\eta)\xi\), generate DNO labels, advance the two-stage Gauss--Legendre
(GL2) solver, or estimate trajectory acceptance rates. The field galleries
reconstruct only a small deterministic subset after the population smoke.

The run completed with the following exact counts.

| Family | Allocation cells | Required count per cell | Attempted specifications | Accepted specifications | Rejected specifications | Support violations |
|---|---:|---:|---:|---:|---:|---:|
| Fifth-order Stokes | 4 | \(31{,}250\) in every cell | 125,000 | 125,000 | 0 | 0 |
| Tanaka-profile sums | 4 | \(31{,}250\) in every cell | 125,000 | 125,000 | 0 | 0 |
| Benjamin--Feir | 66 | \(1{,}894\) in 62 cells and \(1{,}893\) in 4 cells | 125,000 | 125,000 | 0 | 0 |
| JONSWAP/TMA | 27 | \(4{,}630\) in 17 cells and \(4{,}629\) in 10 cells | 125,000 | 125,000 | 0 | 0 |
| **Total** | 101 | -- | **500,000** | **500,000** | **0** | **0** |

The unequal counts in the last two rows are the unique quotient--remainder
allocations whose cell counts differ by at most one. Thus the allocation
result is exact, rather than an approximate consequence of random cell
selection.

## 2. Parameter laws that were exercised

All four families use the periodic interval of length \(L=2\pi\). A Fourier
mode \(n\) therefore has wavenumber \(k=2\pi n/L=n\).

### 2.1 Fifth-order Stokes waves

Let \(h\) denote depth, \(a\) the first-harmonic amplitude, and \(ka\) the
carrier steepness. The four cells are the Cartesian product of two water-depth
branches and two steepness intervals:

\[
  0.005\leq ka<0.03,
  \qquad
  0.03\leq ka\leq0.15.
\]

In the finite-depth branch, \(n\) is uniform on the integers from 14 through
26. The depth is log-uniform on the intersection of
\(0.02\leq h\leq1.5\) and \(0.5\leq kh\leq5\). In the deep-water branch,
\(h\) is log-uniform on the intersection of \(4\leq h\leq50\) and
\(kh\geq5\). The feasible deep-water modes are \(1,\ldots,20\) in the lower
steepness cell and \(3,\ldots,20\) in the upper cell.

Conditional on \(k\), the amplitude is uniform on the intersection of
\[
  0.000766\leq a\leq0.011494
\]
and the assigned steepness interval. The phase is uniform on
\([0,2\pi)\). For finite-depth cases, \(\mathrm{Ur}_+\) denotes the
conservative fifth-order Ursell support quantity computed by the shared
Stokes routine. An amplitude is accepted only if
\(\mathrm{Ur}_+\leq26\); failure redraws the amplitude without changing the
cell, mode, depth, or phase.

### 2.2 Tanaka-profile sums

Let \(m\) be the number of crests, \(a_i\) the dimensional amplitude of crest
\(i\), and
\[
  \alpha_i=\frac{a_i}{h}
\]
its dimensionless amplitude. The three main cells use \(m=1,2,3\),
log-uniform depth \(0.01\leq h\leq0.30\), and a total dimensionless
amplitude
\[
  A=\sum_{i=1}^{m}\alpha_i
\]
that is uniform on \([0.10,0.35]\). For \(m>1\), \(A\) is split by a
\(\operatorname{Dirichlet}(1,\ldots,1)\) vector. The fourth cell has one
crest, log-uniform depth \(0.20\leq h\leq0.35\), and
\(\alpha_1\) uniform on \([0.25,0.45]\).

Let \(x_i\) be the periodic center of crest \(i\), and let \(g_i\) be the
cyclic gap from one sorted center to the next. For \(m>1\), the construction
enforces
\[
  g_i\geq3h
\]
by drawing nonnegative weights \(w_i\) whose sum is one and setting
\[
  g_i=3h+(L-3mh)w_i.
\]
An independent uniform rotation makes the center law translation invariant.
Each crest direction is independently chosen from \(\{-1,+1\}\) with equal
probability.

### 2.3 Benjamin--Feir wave groups

Let \(n_c\) be the carrier mode, \(\Delta n\) the positive symmetric sideband
offset, \(\varepsilon_c\) the carrier steepness, \(\rho\) the
sideband-to-carrier amplitude ratio, and \(\phi\) the common sideband phase.
There are 66 feasible integer cells \((n_c,\Delta n)\), with
\(4\leq n_c\leq20\). Define the leading instability-band fraction
\[
  f=
  \frac{\Delta n/n_c}{2\sqrt{2}\,\varepsilon_c}.
\]
Only cells that intersect \(0<f<1\) are included. Conditional on a cell,
\[
  \max\!\left(0.05,\frac{\Delta n}{2\sqrt{2}\,n_c}\right)
  <\varepsilon_c\leq0.13,
  \qquad
  0.05\leq\rho\leq0.20,
  \qquad
  0\leq\phi<2\pi,
\]
and each of these three continuous variables is uniform on its stated
interval.

### 2.4 JONSWAP/TMA random seas

Let \(h\) denote depth, \(H_s\) significant wave height, \(k_p\) peak
wavenumber, \(\gamma\) the JONSWAP peak-enhancement factor, and \(r_d\) the
fraction of phase-independent linear wave energy assigned to right-moving
modes. The 27 cells are the Cartesian product of three depth strata,
\(\gamma\in\{1,3.3,5\}\), and \(r_d\in\{0,0.5,1\}\).

For the shallow stratum, the peak mode \(n_p\) is uniform on
\(\{16,18,20,22,24\}\). Define peak depth
\[
  \chi=k_p h
\]
and relative height
\[
  \delta=\frac{H_s}{2h}.
\]
The pair \((\chi,\delta)\) is uniform in area on
\[
  0.2\leq\chi\leq1.5,\qquad
  0.03\leq\delta\leq0.16,\qquad
  \chi\delta\leq0.15.
\]
The physical parameters are then \(k_p=2\pi n_p/L\),
\(h=\chi/k_p\), and \(H_s=2h\delta\).

For the finite-depth stratum, \(k_p\), \(h\), and \(H_s\) are independent and
uniform on \([2,12]\), \([0.1,1.5]\), and \([0.005,0.03]\), respectively.
For the deep-water stratum, the corresponding intervals are
\([2,12]\), \([5,25]\), and \([0.005,0.03]\). Each specification also has
128 independent phases in \([0,2\pi)\) for each propagation direction.

## 3. Distances to the declared support boundaries

The following margins are dimensionless and nonnegative on the declared
support. A value near zero means that the parameter specification is close to
a boundary; it is not a support violation.

For a finite-depth Stokes case, define
\[
  M_{\mathrm S}=1-\frac{\mathrm{Ur}_+}{26}.
\]
For a multi-crest Tanaka case, define
\[
  M_{\mathrm T}=\frac{\min_i g_i-3h}{L}.
\]
For a Benjamin--Feir case, define
\[
  M_{\mathrm{BF}}=1-f.
\]
For a shallow JONSWAP/TMA case, define the normalized steepness margin
\[
  M_{\mathrm{sh}}=
  \frac{0.15-k_pH_s/2}{0.15}.
\]

It remains to define a margin for the rectangular finite- and deep-water
JONSWAP/TMA cells. For a variable \(z\) in an interval \([a,b]\), define its
normalized coordinate \(u(z;a,b)=(z-a)/(b-a)\). For the three variables
\(h\), \(H_s\), and \(k_p\), using the bounds of the applicable stratum,
define
\[
  M_{\mathrm{rect}}=
  \min\{u_h,1-u_h,u_H,1-u_H,u_k,1-u_k\},
\]
where \(u_h\), \(u_H\), and \(u_k\) are the normalized coordinates of
\(h\), \(H_s\), and \(k_p\), respectively.

| Margin | Number of cases | Minimum | Fraction below \(10^{-3}\) | Fraction below \(10^{-2}\) | Fraction below \(5\times10^{-2}\) |
|---|---:|---:|---:|---:|---:|
| \(M_{\mathrm S}\), finite-depth Stokes | 62,500 | \(3.8335\times10^{-5}\) | 0.0128% | 0.1712% | 0.8400% |
| \(M_{\mathrm T}\), multi-crest Tanaka | 62,500 | \(3.4295\times10^{-6}\) | 0.4272% | 4.5184% | 21.5632% |
| \(M_{\mathrm{BF}}\) | 125,000 | \(1.8060\times10^{-6}\) | 0.4688% | 4.6072% | 16.4368% |
| \(M_{\mathrm{sh}}\) | 41,670 | \(6.3519\times10^{-5}\) | 0.0312% | 0.4824% | 2.5150% |
| \(M_{\mathrm{rect}}\) | 83,330 | \(6.6892\times10^{-8}\) | 0.6336% | 5.8694% | 27.0443% |

The small rectangular margin is expected to occur often because it is the
minimum distance to six independently sampled faces. The value is useful for
locating boundary cases, but it does not by itself indicate a bad sample.

Three other tails should be retained for the field and trajectory smoke:

1. Stokes sampling made 141,999 amplitude proposals. It rejected 16,999
   proposals internally, accepted every assigned specification, exhausted no
   specification, and required at most 73 redraws for one case.
2. The Tanaka Dirichlet split permits a crest to have arbitrarily small
   positive amplitude. Among 218,750 individual crests, 5.6123% have
   \(\alpha_i<0.01\), 0.5518% have \(\alpha_i<0.001\), and the minimum is
   \(2.3132\times10^{-6}\). These cases satisfy the stated law, but some are
   effectively lower-crest-count cases.
3. The maximum shallow random-sea peak steepness \(k_pH_s/2\) is
   0.14999047, below the limit 0.15. The observed maximum peak wavenumber 24
   comes from the discrete shallow peak-mode set and is not clipping of the
   finite/deep interval \([2,12]\).

## 4. Uniform-coordinate diagnostics

Suppose a scalar random variable \(X\) is declared uniform on an interval
\([a,b]\). Its uniform coordinate is
\[
  U=\frac{X-a}{b-a}.
\]
If \(X\) is declared log-uniform on a positive interval \([a,b]\), its
uniform coordinate is
\[
  U=\frac{\log X-\log a}{\log b-\log a}.
\]
Cell- or mode-conditioned bounds are substituted case by case.

The shallow JONSWAP/TMA support is not rectangular. For a fixed peak depth
\(\chi\), define
\[
  \delta_{\max}(\chi)=\min\!\left(0.16,\frac{0.15}{\chi}\right).
\]
The conditional relative-height coordinate is
\[
  U_\delta=\frac{\delta-0.03}{\delta_{\max}(\chi)-0.03}.
\]
The peak-depth coordinate is \(U_\chi=F_\chi(\chi)\), where the exact
marginal distribution function under the uniform-area law is
\[
  F_\chi(x)=
  \frac{\displaystyle\int_{0.2}^{x}
    \bigl(\delta_{\max}(s)-0.03\bigr)\,ds}
  {\displaystyle\int_{0.2}^{1.5}
    \bigl(\delta_{\max}(s)-0.03\bigr)\,ds}.
\]
Here \(s\) is the integration variable.

For \(n\) observed coordinates \(U_1,\ldots,U_n\), define the empirical
distribution function
\[
  F_n(u)=\frac{1}{n}\sum_{j=1}^{n}\mathbf{1}_{\{U_j\leq u\}}
\]
and the uniform discrepancy
\[
  D_n=\sup_{0\leq u\leq1}|F_n(u)-u|.
\]
With Dvoretzky--Kiefer--Wolfowitz tail probability
\(\alpha_{\mathrm{DKW}}=0.05\), the reported 95% half-width is
\[
  \epsilon_n=
  \sqrt{\frac{\log(2/\alpha_{\mathrm{DKW}})}{2n}}.
\]

| Family and transformed coordinate | \(n\) | \(D_n\) | \(\epsilon_n\) |
|---|---:|---:|---:|
| Stokes, conditional log-depth | 125,000 | 0.002829 | 0.003841 |
| Stokes, phase | 125,000 | 0.002023 | 0.003841 |
| Stokes, every conditional amplitude proposal | 141,999 | 0.002537 | 0.003604 |
| Tanaka, conditional log-depth | 125,000 | 0.002129 | 0.003841 |
| Tanaka, conditional total dimensionless amplitude | 125,000 | 0.002341 | 0.003841 |
| Benjamin--Feir, conditional carrier steepness | 125,000 | 0.001842 | 0.003841 |
| Benjamin--Feir, perturbation ratio | 125,000 | 0.001688 | 0.003841 |
| Benjamin--Feir, sideband phase | 125,000 | 0.002099 | 0.003841 |
| JONSWAP/TMA, finite/deep depth | 83,330 | 0.002796 | 0.004705 |
| JONSWAP/TMA, finite/deep significant height | 83,330 | 0.001831 | 0.004705 |
| JONSWAP/TMA, finite/deep peak wavenumber | 83,330 | 0.002637 | 0.004705 |
| JONSWAP/TMA, shallow peak depth \(U_\chi\) | 41,670 | 0.005596 | 0.006653 |
| JONSWAP/TMA, shallow conditional height \(U_\delta\) | 41,670 | 0.003165 | 0.006653 |
| JONSWAP/TMA, first right-moving phase | 125,000 | 0.001900 | 0.003841 |
| JONSWAP/TMA, first left-moving phase | 125,000 | 0.002440 | 0.003841 |

Every listed discrepancy is smaller than its corresponding 95% DKW
half-width. This is a finite-sample diagnostic consistent with the declared
one-dimensional laws. It is neither a proof of the random-number generator
nor a test of every joint independence relation. The two JONSWAP/TMA phase
histograms additionally contain all 16,000,000 phases per propagation
direction. For phase values \(\phi_1,\ldots,\phi_N\), define their circular
resultant by
\[
  R=\left|\frac{1}{N}\sum_{j=1}^{N}
  \exp(\mathrm{i}\phi_j)\right|,
\]
where \(\mathrm{i}^2=-1\). The right- and left-moving values of \(R\) are
\(3.4209\times10^{-4}\) and \(3.7896\times10^{-4}\), respectively.

## 5. Deterministic replay, files, and timing

The run used the test split, root seed 2026072205, revision 1, and stream
identifier 500000. JAX used a single CPU device with 64-bit arithmetic
enabled. The configuration fingerprint is
`b85aa699658b3848d333e1c1ceb3704dff0f1ca18213bf7ffe5715d748fbbba1`.
All 303 deterministic sentinel specifications reproduced their archived
records exactly.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `stokes_samples.npz` | 15,736,941 | `63d75d515e649520391bfff66ed7ccaf257e0f2732c569dd5b0edbc8c65f0d1a` |
| `tanaka_samples.npz` | 18,253,606 | `aa16ae40571dce19ee9b9994d5c7dc8be98ee0644312d499f8b583b8de654d55` |
| `benjamin_feir_samples.npz` | 10,753,452 | `a6ca72e18e6b3ac9efee1cc8de7341377dca1b909830cf0355fc3477dec220cf` |
| `jonswap_tma_samples.npz` | 17,507,382 | `de8efa28c314d9b7ff29a7c8b2c40d6fd004ccdbe83e8bf4c63ee410f24c6842` |
| `sentinel_records.json` | 1,065,541 | `34bfabcf65ad6561d486c1f42715947f47979eabaa72b709bf5581be4cbe79ae` |
| `summary.json` | 30,986 | `ed15a485ee1a1800309c680d621f3b3db57d0f5b9d1007bc58a28bd91de26201` |

The summary also records the SHA-256 digest of every sampling-law source used
by the run. Sampling times were 41.775 s for Stokes, 7.364 s for Tanaka,
4.391 s for Benjamin--Feir, and 20.789 s for JONSWAP/TMA. Archive writing and
sentinel replay brought the total wall time to 74.741 s.

The machine-readable result is
[summary.json](../outputs/paper_corpus_population_smoke_500k_20260725/summary.json).

## 6. Figures

Every specification contributes to an exact count, histogram, hexagonal bin,
empirical distribution, or accumulated phase histogram in the population
figures. The field galleries and cell atlases instead reconstruct explicitly
selected cases with the production initial-condition routines.

- [Exact allocations and support overview](../outputs/paper_corpus_population_smoke_500k_20260725/figures/overview_counts_support.png)
- [Stokes parameter population](../outputs/paper_corpus_population_smoke_500k_20260725/figures/stokes_population.png)
- [Tanaka parameter population](../outputs/paper_corpus_population_smoke_500k_20260725/figures/tanaka_population.png)
- [Benjamin--Feir parameter population](../outputs/paper_corpus_population_smoke_500k_20260725/figures/benjamin_feir_population.png)
- [JONSWAP/TMA parameter population](../outputs/paper_corpus_population_smoke_500k_20260725/figures/jonswap_tma_population.png)
- [Stokes selected-field gallery](../outputs/paper_corpus_population_smoke_500k_20260725/figures/stokes_field_gallery.png)
- [Tanaka selected-field gallery](../outputs/paper_corpus_population_smoke_500k_20260725/figures/tanaka_field_gallery.png)
- [Benjamin--Feir selected-field gallery](../outputs/paper_corpus_population_smoke_500k_20260725/figures/benjamin_feir_field_gallery.png)
- [JONSWAP/TMA selected-field gallery](../outputs/paper_corpus_population_smoke_500k_20260725/figures/jonswap_tma_field_gallery.png)
- [Stokes cell atlas](../outputs/paper_corpus_population_smoke_500k_20260725/figures/stokes_cell_atlas.png)
- [Tanaka cell atlas](../outputs/paper_corpus_population_smoke_500k_20260725/figures/tanaka_cell_atlas.png)
- [Benjamin--Feir cell atlas](../outputs/paper_corpus_population_smoke_500k_20260725/figures/benjamin_feir_cell_atlas.png)
- [JONSWAP/TMA cell atlas](../outputs/paper_corpus_population_smoke_500k_20260725/figures/jonswap_tma_cell_atlas.png)
- [Machine-readable figure definitions and selected-case records](../outputs/paper_corpus_population_smoke_500k_20260725/figures/plot_summary.json)

## 7. What this smoke does not establish

1. Parameter support does not imply that every constructed field is smooth
   enough for the intended discretization.
2. A valid initial field does not imply an accepted GL2 trajectory.
3. This run does not measure nonfinite DNO labels, temporal defects, or
   long-time error growth.
4. The uniform-coordinate checks are marginal checks. They do not exhaust
   all joint or conditional independence statements.
5. The Tanaka small-component tail and the cases nearest the Stokes,
   Benjamin--Feir, and shallow-sea boundaries should be included deliberately
   in the next field-construction and short-trajectory smoke.

The result therefore supports the implementation of the four parameter laws
and their exact allocation. It does not yet support a claim about the
numerical yield of the final trajectory corpus.
