# Paper-dataset generation readiness

Date: 2026-07-27

## Final authenticated release record

The literature-aligned replacement dataset is complete and source-authenticated.
Every family contains 16,384 training, 1,024 validation, and 1,024 test
cases.  The final release retained 73,728 cases from 74,045 attempts,
recorded 317 zero-row rejections, and stores 7,686,144 rows:

| Family | Revision | Attempted | Accepted | Rejected | Rows |
|---|---:|---:|---:|---:|---:|
| fifth-order Stokes | 2 | 18,432 | 18,432 | 0 | 18,432 |
| corrected Tanaka | 3 | 18,432 | 18,432 | 0 | 3,686,400 |
| Benjamin--Feir | 4 | 18,455 | 18,432 | 23 | 3,686,400 |
| JONSWAP/TMA | 4 | 18,726 | 18,432 | 294 | 294,912 |

The four cumulative combined-summary SHA-256 identities, at 2,048, 4,096,
8,192, and 16,384 training cases per family, are
`2265cb71270accb3f6d9103e90433b664ca813989dfab63403d8612af22d28c3`,
`7feb018ee4d94eab10278917c145262026cccab6169eaaa2d8877f7c7b28c9da`,
`43b1eb91198e724c6f32b871c2d0efd0d8355533789919df5897ad588e7e85f6`,
and
`32f764cf865c90892ee0329e48fe992cbc24df2707e05fd1af2ee9cd2b84307c`.
The final 26-source manifest and trajectory-map SHA-256 identities are
`dbb76ed1cba4d0146f3c9643a73d168d3a9fa97bc27c37fe0fd7ad08554bd734`
and
`f26888ad0a2c738405fb1662b9410337e1babfe48f7913805b5243ae0f279cc8`.

The immutable document handoff is
`outputs/paper_dataset_final_postcompletion/32f764cf865c90892ee0329e48fe992cbc24df2707e05fd1af2ee9cd2b84307c/paper_dataset_document_handoff.json`
(SHA-256
`b21830cf56f9c17a1ee165624c71fdc0e286727747d364330fbace19e5ef6053`).
It binds all family audits, all cumulative views, every physical shard, the
postcompletion diagnostics, 94,666,557,624 measured shard bytes, and the
training-only normalization scales.  The authenticated family illustration
is installed at `notes/figures/parameterized_dataset_case_examples.{pdf,png}`;
its PDF and PNG SHA-256 identities are
`16309a687960d02e879e1e646b7854dcc53336b1935b0e81dde738e64d9b7011`
and
`88b9687e40344984b84e0deb601fcb16db5c89720c29b2ffa592aa311b0f1dfa`.
The superseded Tanaka population and all adoption gates remain historical
evidence outside the release.  No reported model has been trained on the
replacement dataset.

> **Historical readiness snapshot.** This note predates the final
> JONSWAP/TMA numerical contract and must not be used as a launch recipe.  The
> current random-sea method uses 2048 internal points, sharp
> \(|k|\leq704\), order four, padding factor eight, GL2 step \(0.01\), saved
> spacing \(0.08\), residual tolerance \(10^{-8}\), and an iteration cap of
> five.  It adjusts for 20 peak periods and then evolves autonomously for 16
> peak periods.  Its present input population also requires
> \(k_pH_s/2\leq0.08\) in every depth stratum; this moderate-sea restriction
> supersedes the old shallow \(0.15\) bound stated later in this historical
> note.  Stored fields use 1024 points, \(P_{128}\), and a freshly
> evaluated order-six target.  See
> [the current data-generation contract](../solver/gen_data/README.md) and
> [the condensed construction contract](paper_dataset_generation_condensed_20260726.tex).
> The later formulas also predate the Tanaka revision-3 conditional-depth law
> and the Benjamin--Feir revision-4 focused-steepness support.  Any later use
> of "current", "final", or "required" refers only to the July snapshot.
> The guarded category calculations described below are diagnostic.  Later
> dated corrections record the exact-source successors and launches that were
> current at those checkpoints; only the final authenticated release record above records
> present status.  The
> measurements and commands below are retained as dated historical evidence
> and are not a launch recipe.

> **Correction, 2 August 2026.** The revision-2 Stokes dataset is complete
> at 16,384 training, 1,024 validation, and 1,024 test cases and is preserved.
> The historical Tanaka run also contains 4,096 accepted trajectories, but it
> is not retained for the paper dataset.  Its constructor inflated one small
> crest in 64 cases because the old solve had an effective amplitude floor of
> \(9.9478\times10^{-4}\).  The corrected solve widens the bracket, uses 48
> bisections, and checks every requested crest height.  All 4,096 unchanged
> revision-3 parameter specifications must be replayed in a fresh root; old
> and corrected Tanaka rows must not be mixed.  The Benjamin--Feir
> revision-4 and JONSWAP/TMA revision-3 exact-source category gates have now
> passed.  Both methods
> sharply project the state after every completed GL2 step, to \(P_{256}\)
> and \(P_{704}\), respectively; they use no Hou--Li or other smooth damping.
>
> For the Benjamin--Feir population specified at that checkpoint, let
> \(n_c\in\{4,\ldots,20\}\)
> be the carrier mode, let \(\Delta n\) be the positive symmetric sideband
> separation, let \(\epsilon_c\) be the steepness of the first carrier
> harmonic, and let \(\rho\) be the amplitude of each sideband divided by the
> first carrier-harmonic amplitude.  Define
> \[
> \beta=\frac{\Delta n/n_c}{2\sqrt2\,\epsilon_c},\qquad
> \epsilon_f=\epsilon_c\left(1+2\sqrt{1-\beta^2}\right).
> \]
> The current pre-rollout support is
> \[
> 0<\beta<1,\qquad
> \epsilon_f\leq\frac{1+\sqrt2}{10},\qquad
> 0.05\leq\rho\leq0.10.
> \]
> Conditional on \((n_c,\Delta n)\), \(\epsilon_c\) is uniform on the
> nonempty interval satisfying these inequalities and
> \(0.05\leq\epsilon_c\leq0.13\).  All 66 mode-pair categories remain
> possible.  Stream 934 fixed one parameter draw per category before
> numerical execution and accepted all 66 attempts with no replacement.  All
> 13,200 retained frames were finite; the largest relative internal
> Hamiltonian drift was \(5.38713\times10^{-5}\), the largest GL2 stage
> residual was \(9.999984\times10^{-9}\), and the smallest water column was
> \(4.958158\).  The quantity \(\epsilon_f\) is a leading nonlinear
> Schrödinger
> focused-envelope proxy, not a physical-slope, breaking, or regularity bound.
> The complete-trajectory numerical checks therefore remain required.  These
> statements supersede later Benjamin--Feir status and support statements in
> this dated snapshot.  Revision 4 changes the Benjamin--Feir sampling law but
> retains the revision-3 numerical evolution method; revision-3 proposals and
> validation artifacts cannot be counted toward a revision-4 dataset quota.
>
> Support-validation stream 934 is retained as independent evidence for the
> focused Benjamin--Feir support: it conditioned the preceding sampling law
> before integration and accepted all 66 selected trajectories.  The direct
> production-law gate is instead revision-4 stream 935.  It accepted all
> (66/66) first proposals, rejected none, and retained 13,200 rows.  Its
> maximum GL2 residual was \(9.999967\times10^{-9}\), maximum relative
> internal-Hamiltonian drift was \(2.952009\times10^{-5}\), and minimum water
> column was \(4.960598\).
>
> JONSWAP/TMA revision-3 stream 933 accepted one trajectory in every one of
> its 27 categories after 29 attempts and retained 432 rows.  The first
> proposals in `finite__gamma_3p3__right_1` and
> `shallow__gamma_1__right_0p5` failed during nonlinear adjustment, retained
> zero rows, and were replaced by passing canonical second proposals in the
> same categories.  Over the accepted cases, the maximum GL2 residual was
> \(9.999783\times10^{-9}\), maximum relative internal-Hamiltonian drift was
> \(5.908800\times10^{-5}\), and minimum water column was \(0.0127322\).
> These one-case-per-category calculations establish coverage and accepted
> full-horizon health, not a population rejection-rate estimate.
>
> The exact independent audit passed at
> `outputs/nondataset_bf_revision4_jonswap_revision3_category_smoke_20260802/category_smoke_release_audit_v1.json`;
> its SHA-256 is
> `62fb5fe3b6be7df0d21784b1ce590c44cc484bf5fdf84ad6097fbed6f655f684`.
> Fresh bulk generation was launched at 22:26 EDT on 2 August 2026 under
> `outputs/paper_dataset_bf_revision4_jonswap_revision3_literature_aligned_v1`.

## Decision at the time of this note

The population laws, exact numerical target, accepted-case quota driver,
finite attempt bound, restart logic, combined dataset view, and case-balanced
loader are implemented. The earlier exact-contract four-family smoke used the
paired method-validation path; the current one-rollout executor has focused
software-test coverage. Large-scale trajectory generation should begin only
after one accepted case from every remaining trajectory parameter category has
been run on the intended GPU platform with the final bounded scheduler and
the single-\(0.01\)-rollout production path. Family construction is decided
before integration and is separate from the complete-trajectory decision.
That historical gate has since been superseded by the current-revision
calculations recorded above.  The Benjamin--Feir and JONSWAP/TMA exact-source
gates passed, and their fresh bulk lanes were launched at 22:26 EDT on
2 August 2026.  The replacement dataset remains incomplete until all declared
chunks and the final combined-view audit finish.

The production target uses \(C_{\mathrm{tr}}=16384\) accepted training cases
per family and \(C_{\mathrm{va}}=C_{\mathrm{te}}=1024\) accepted validation
and test cases per family. Training is generated cumulatively at
\(C_{\mathrm{tr}}\in\{2048,4096,8192,16384\}\), so \(2048\) is the first
checkpoint rather than the final dataset. Here \(C_{\mathrm{tr}}\),
\(C_{\mathrm{va}}\), and \(C_{\mathrm{te}}\) denote case counts within one
physical family, not row counts.

## 1. Frozen common numerical contract

Let \(L=2\pi\) be the periodic domain length, \(N=1024\) the number of spatial
grid points, \(M=6\) the highest Craig--Sulem order, \(p=8\) the
pseudospectral padding factor, and \(K=128\) the largest delivered
wavenumber. The nondimensional gravitational acceleration is \(g=1\).
Let \(P_K\) denote Fourier projection onto \(|k|\leq K\), and let
\(P_K^\circ\) additionally remove the zero Fourier mode. Let
\(\mathcal G^{N,p}_M\) denote the order-\(M\) pseudospectral Craig--Sulem
recursion on \(N\) points with padding factor \(p\). For surface elevation
\(\eta\), surface potential \(\xi\), and depth \(h\), the stored target is

\[
G_{\mathrm{ref}}(\eta)\xi
=P_K^\circ\!\left[
  \mathcal G^{N,p}_M(P_K\eta;h)(P_K\xi)
\right].
\]

All constructors, rollouts, and target evaluations use float64 arithmetic.
The final \(\eta\), \(\xi\), and \(G_{\mathrm{ref}}(\eta)\xi\) arrays are stored as
float32; depth and time are stored as float64. Every retained row has three
fields of length \(N\).

The trajectory integrator is the two-stage Gauss--Legendre method, denoted
GL2. Production saves a state every \(0.08\) time units and takes eight equal
GL2 substeps between saved states. Hence the actual production step is

\[
\Delta t_{\mathrm{GL2}}=\frac{0.08}{8}=0.01.
\]

Each production case is rolled out once at this step. Each implicit stage has
residual tolerance \(10^{-8}\) and at most four fixed-point updates.
Steps \(0.005\) and \(0.0025\) belong to a separate method-level refinement
utility. They are not additional production rollouts.

## 2. Four declared populations

The notation \(X\sim\operatorname{Unif}[a,b]\) means that \(X\) is uniformly
distributed on the stated interval. A positive variable is log-uniform on
\([a,b]\) when its logarithm is uniform on
\([\log a,\log b]\). The parameter categories below receive accepted-case counts
that differ by at most one.

### 2.1 Fifth-order Stokes: 4 parameter categories

Let \(n\) be an integer carrier mode, \(k=2\pi n/L\) its wavenumber, \(a\)
the leading amplitude, \(h\) the depth, and \(\phi\) the spatial phase. The
four categories are

\[
\{\text{finite},\text{deep}\}
\times
\{0.005\leq ka<0.03,\;0.03\leq ka\leq0.15\}.
\]

The common amplitude interval is
\(a\in[0.000766,0.011494]\). In the finite branch,
\(n\in\{14,\ldots,26\}\), \(h\in[0.02,1.5]\), and
\(0.5\leq kh\leq5\). In the deep branch,
\(n\in\{1,\ldots,20\}\), \(h\in[4,50]\), and \(kh\geq5\).
The sampler first finds the integer modes for which the assigned depth,
amplitude, and steepness intervals intersect. It chooses uniformly among
those modes, chooses \(h\) log-uniformly in the resulting
mode-conditioned interval, chooses
\(\phi\sim\operatorname{Unif}[0,2\pi)\), and chooses \(a\) uniformly in the
intersection of the amplitude interval with its assigned steepness category.

For the finite branch, let \(H_+\) be the implemented upper bound on the
crest-to-trough height of the complete fifth-order surface and let
\(\lambda=2\pi/k\). The declared support condition is

\[
\mathrm{Ur}_+=\frac{H_+\lambda^2}{h^3}\leq26.
\]

If this fails, only \(a\) is redrawn in the same category; \(n,h,\phi\), split,
and case identity remain fixed. The production limit is the initial draw
plus at most 1000 redraws. Exhaustion is a recorded zero-row support
rejection. An accepted Stokes state stores one row at \(t=0\).

### 2.2 Tangent-Hermite Tanaka profiles: 11 parameter categories

Let \(m\) be the number of crests and let
\(\alpha_j=a_j/h\) be the amplitude of crest \(j\) divided by depth. The
three main structural groups have \(m=1,2,3\),

\[
U_h\sim\operatorname{Unif}[0,1],
\qquad
h=(0.01)^{1-U_h}(0.30)^{U_h},
\qquad
A=\sum_{j=1}^m\alpha_j
\sim\operatorname{Unif}[0.10,0.35].
\]

This formula is geometric interpolation between the endpoint depths. At
fixed \(\alpha_j\), multiplying \(h\) by a constant multiplies both the
height and width of a Tanaka component by that constant. Equal increments of
\(U_h\) therefore cover equal multiplicative changes in profile scale. This
is a numerical coverage rule, not a probability model for physical water
depths.

For \(m>1\), the fractions
\((\alpha_1/A,\ldots,\alpha_m/A)\) are uniform on the simplex, equivalently
a \(\operatorname{Dirichlet}(1,\ldots,1)\) draw. The steep structural group has
\(m=1\),
\(h=(0.20)^{1-U_h}(0.35)^{U_h}\) for a fresh
\(U_h\sim\operatorname{Unif}[0,1]\), and
\(\alpha_1\sim\operatorname{Unif}[0.25,0.45]\).

Let
\[
q=\#\{j:s_j=+1\}
\]
be the number of right-moving crests. For each main group, the pairs
\((m,q)\), with \(q=0,\ldots,m\), define \(2+3+4=9\) parameter categories.
The steep one-crest group adds the two categories \(q=0,1\). Conditional on
\(q\), the sampler chooses uniformly which \(q\) of the \(m\) exchangeable
crest labels move right; the remainder move left.

The accepted-case count is divided as evenly as possible over all eleven
categories. This gives equal coverage to the possible direction
compositions, rather than the binomial frequencies generated by independent
fair signs. Left and right directions remain balanced marginally in the ideal
equal-category population. For a finite quota, category counts differ by at
most one and a one-case direction imbalance is possible; the fixed order
gives a quota of 1024 one extra left-moving one-crest case. The main
\(m=1,2,3\) groups consequently have aggregate shares \(2/11,3/11,4/11\),
and the steep group has share \(2/11\). This prospective law replaces the old
four-group independent-sign mixture and deliberately gives more total weight
to two- and three-crest states.

The periodic crest centers are sampled through cyclic gaps

\[
g_j=3h+(L-3mh)w_j,\qquad
(w_1,\ldots,w_m)\sim\operatorname{Dirichlet}(1,\ldots,1),
\]

followed by a uniform rotation of all centers. Thus every pair of neighboring
centers is separated by at least \(3h\), and the law is invariant under
periodic translation. The profile constructor is the tangent-Hermite
periodic construction. A finite negative surface-potential radicand is outside
the real-valued construction and is rejected before a square root is taken.
A nonfinite radicand or an exception in the wave-speed calculation stops the
run; it is not relabeled as an out-of-support case and replaced.

At this July snapshot, the direction-aware Tanaka law advanced the then-global
generator revision to 2 and used the version-2 stored Tanaka parameter record.
Revision-1 parameter smokes, dry runs, and all earlier family pilot artifacts
did not satisfy that historical revision-2 run specification.  The planned
104-category trajectory gate and dataset were then expected to use revision 2.
That expectation is superseded by the current family-local revision tuple
\((2,3,4,4)\).

### 2.3 Benjamin--Feir carrier and sidebands: 66 parameter categories

Let \(n_c\in\{4,\ldots,20\}\) be the carrier mode,
\(\Delta n\) the positive symmetric sideband offset,
\(\epsilon_c\) the steepness of the first elevation harmonic, \(\rho\) the
sideband-to-carrier amplitude ratio, and \(x_0\) a global spatial translation.
The leading deep-water instability condition is

\[
0<
\frac{\Delta n/n_c}{2\sqrt{2}\,\epsilon_c}
<1.
\]

Each of the 66 integer pairs \((n_c,\Delta n)\) that intersects this
condition for \(\epsilon_c\leq0.13\) is one parameter category. Conditional on
the pair, define

\[
\epsilon_{\min}
=\max\!\left\{0.05,
  \frac{\Delta n}{2\sqrt{2}\,n_c}\right\}.
\]

The continuous laws are

\[
\epsilon_c\sim\operatorname{Unif}(\epsilon_{\min},0.13],\qquad
\rho\sim\operatorname{Unif}[0.05,0.20],\qquad
x_0\sim\operatorname{Unif}[0,L).
\]

The depth is \(h=5L/(2\pi)=5\), so even the fundamental mode has \(kh=5\).
The constructor uses the project's fifth-order deep-water carrier plus the
symmetric equation-(33) sidebands. With \(y=x-x_0\), the carrier phase is zero
in the translated coordinate and both sidebands have the fixed relative phase
\(\theta=-\pi/4\). Thus \(x_0\) translates the complete state without changing
the quartet phase \(2\theta=-\pi/2\).

For \(k_c=2\pi n_c/L\), define the carrier period and realized horizon by

\[
T_c=\frac{2\pi}{\sqrt{gk_c}},\qquad
T_{\mathrm{BF}}=0.08\left\lfloor\frac{100T_c}{0.08}\right\rfloor.
\]

Every case is therefore observed for 100 linear carrier periods, up to the
saved-grid floor.

### 2.4 Windowed JONSWAP/TMA random seas: 27 parameter categories

Let \(k_p\) be the peak wavenumber, \(H_s\) the significant wave height,
\(\gamma\) the peak-enhancement factor, and \(r_d\) the fraction of linear
wave energy traveling to the right. The categories are

\[
\{\text{shallow},\text{finite},\text{deep}\}
\times\{1,3.3,5\}
\times\{0,\tfrac12,1\},
\]

where the last two factors are \(\gamma\) and \(r_d\).

In the shallow stratum, choose the peak mode
\(n_p\) uniformly from \(\{16,18,20,22,24\}\), set
\(k_p=2\pi n_p/L\), and draw

\[
\chi_p=k_ph\in[0.2,1.5],\qquad
\delta_s=\frac{H_s}{2h}\in[0.03,0.16]
\]

uniformly with respect to area on the subset
\(\chi_p\delta_s\leq0.08\). Then
\(h=\chi_p/k_p\) and \(H_s=2h\delta_s\). In the finite stratum,
\(k_p\), \(h\), and \(H_s\) are independent uniforms on
\([2,12]\), \([0.1,1.5]\), and \([0.005,0.03]\), conditioned on
\(k_pH_s/2\leq0.08\). In the deep stratum, the
corresponding intervals are \([2,12]\), \([5,25]\), and
\([0.005,0.03]\), with the same condition. Both right- and left-going phase
arrays are sampled and
stored in the case specification before construction.

The constructor uses cell-integrated JONSWAP density, the TMA finite-depth
factor, 16-point quadrature in each Fourier cell, and the frozen density
window

\[
W(k)=
\begin{cases}
1,&0\leq k\leq96,\\
\cos^2\!\left[\dfrac{\pi}{2}\dfrac{k-96}{32}\right],
  &96<k<128,\\
0,&k\geq128.
\end{cases}
\]

This window is part of the population definition, not a realized-wave
filter. To see why the transition begins at 96, define the finite-depth
dispersion relation
\(\omega(k,h)=\sqrt{gk\tanh(kh)}\). For \(r\geq1\),

\[
\frac{\omega(rk_p,h)}{\omega(k_p,h)}
=\left[
r\frac{\tanh(rk_ph)}{\tanh(k_ph)}
\right]^{1/2}
\geq\sqrt r.
\]

Every allowed case has \(k_p\leq24\). Hence
\(96=4\max k_p\) leaves all frequencies through \(2\omega_p\) untapered,
where \(\omega_p=\omega(k_p,h)\), and the remaining 32 integer wavenumbers
give a smooth transition to \(K=128\).

The frozen-window audit used 864 deterministic specifications, 32 from each
parameter category, and did not integrate them in time. Relative to a practically
untapered reference on \(k\leq128\), the current window retained at least
98.469% of spectral mass in the shallow stratum and at least 99.653% in the
finite and deep strata. The median spectral total variations were 0.767%,
0.0869%, and 0.0861%, respectively. Moving the taper to \(80\)--\(112\)
changed a typical shallow surface by 7.14% in relative \(L^2\) while reducing
ensemble root-mean-square slope by only 4.33%. Independent
\(N=512\)-versus-\(1024\) fixed-band target disagreement was at most
\(4.11\times10^{-4}\), and the three full-horizon random-sea panel cases had
temporal defects between \(4.52\times10^{-6}\) and
\(5.63\times10^{-6}\). These checks support freezing \(96\)--\(128\);
they do not justify rejecting an individual rough-looking realization.

After the one-time spectral projection, a realization is sent to the solver
only if both state fields are finite and \(\min_x(h+\eta_0)>0\). This is the
graph-domain requirement that the free surface lie above the flat bottom. If
one case in a batch fails it, only that case becomes a zero-row attempt and is
replaced in the same parameter category; valid batch companions are retained.
The record also stores the discrete peak cell, half-maximum spectral width,
RMS values of \(\eta_0\) and \(\xi_0\), realized significant-height ratio,
linear-Hamiltonian consistency, and minimum water column. These are
diagnostics, not extra thresholds.

## 3. Declared support and numerical acceptance are different decisions

The population conditions in Section 2 answer whether an attempted
specification belongs to the named family. They are fixed before observing a
rollout. The trajectory calculation then asks whether a numerical reference
exists over the complete requested interval.

There is one post-rollout production condition:

\[
\boxed{\text{reject if and only if no complete admissible trajectory exists
on }[0,T].}
\]

Here a complete admissible trajectory reaches \(T\), has finite
\(\eta\), \(\xi\), and \(G_{\mathrm{ref}}(\eta)\xi\) at every saved time,
satisfies \(h+\eta>0\) at every saved state, and solves every implicit GL2
stage to the declared residual tolerance. A nonfinite field, nonfinite
target, unsolved stage, or loss of the graph domain is a diagnostic cause of
this one failure condition. It is not a separate empirical filter. There is
no sign-change, slope, extremum-count, absolute target-amplitude,
Hamiltonian-drift, or visual-appearance rejection rule. Rejected trajectories
own zero rows; no finite prefix or isolated frame is kept.

The production step was selected before bulk generation. In the May study,
the rollout interface used a saved interval of \(0.08\) with eight GL2
substeps, so the actual step tested and frozen was \(0.01\). The July
full-horizon panel then compared actual steps \(0.01\) and \(0.005\). For
completeness, if \(r=0\) denotes the \(0.01\) trajectory and \(r=1\) the
\(0.005\) trajectory, define at saved time \(t_j\)

\[
Y_r^{(\eta)}(t_j)=\frac{P_K\eta_r(t_j)}{h},\qquad
Y_r^{(\xi)}(t_j)=\frac{P_K\xi_r(t_j)}{h\sqrt{gh}},\qquad
Y_r^{(G)}(t_j)=
\frac{G_{\mathrm{ref}}(\eta_r(t_j))\xi_r(t_j)}{\sqrt{gh}},
\]

and let

\[
E=
\max_{a\in\{\eta,\xi,G\}}
\frac{
  \max_j\|Y_1^{(a)}(t_j)-Y_0^{(a)}(t_j)\|_{L^2}
}{
  \max_j\|Y_1^{(a)}(t_j)\|_{L^2}+10^{-12}
}.
\]

The \(L^2\) norm is the normalized periodic grid norm. All 10 panel cases
that completed at both steps had
\(E\leq1.126337\times10^{-5}\), well below the declared validation tolerance
\(10^{-3}\). The steep Tanaka and canonical Benjamin--Feir stress cases
became incomplete at essentially the same physical time at both steps. Thus
halving the step changed no decision, and no \(0.0025\) diagnostic retry ran.
The \(0.01/0.005\) comparison and optional \(0.0025\) retry remain available
as method-validation utilities only. They are not run for every production
case, do not determine which of two computed trajectories is stored, and do
not define a second production rejection condition.

Static Stokes states do not undergo a fictitious temporal check. They require
declared Stokes support, a finite constructed state, positive water column,
and finite common target. These properties are checked on successfully
returned arrays. An unexpected constructor or target-evaluation exception
stops the run rather than being recorded as a rejected numerical sample.

## 4. Horizons, retained rows, and exact dataset size

Tanaka is integrated on \(0\leq t\leq200\), with the dense result saved every
\(0.08\), and contributes 200 activity-selected times. Benjamin--Feir is
integrated through its case-dependent \(T_{\mathrm{BF}}\) above and contributes
200 approximately uniform saved-grid times, including both endpoints.
Selection occurs only after acceptance and changes storage, not quality.

For a random sea, let

\[
T_p=\frac{2\pi}{\omega(k_p,h)},\qquad
H=16T_p,\qquad
T_s=0.08\left\lfloor\frac{H}{0.08}\right\rfloor.
\]

The case is integrated through \(T_s\leq H\) and contributes 16 times,
uniformly spaced in saved-grid index and including both endpoints.

The retained rows per accepted case are therefore

| family | rows per case |
| --- | ---: |
| Stokes | 1 |
| Tanaka | 200 |
| Benjamin--Feir | 200 |
| JONSWAP/TMA | 16 |

If every family contributes \(C\) accepted cases to one split, that split has

\[
4C\ \text{accepted cases},\qquad
(1+200+200+16)C=417C\ \text{rows}.
\]

Across all three splits the exact row count is

\[
417(C_{\mathrm{tr}}+C_{\mathrm{va}}+C_{\mathrm{te}}).
\]

## 5. Quotas, restart, view construction, and loading

Suppose a family has \(m\) ordered parameter categories and needs \(C\)
accepted cases. Write \(C=bm+r\), where \(0\leq r<m\). The first \(r\)
categories receive \(b+1\) accepted cases and the remaining categories
receive \(b\). An unsuccessful attempt does not advance the accepted count
of its category, so its replacement is scheduled in the same category.

The family sampling laws describe attempted specifications, not the
unconditional law of loader-visible cases. If \(X\) is a complete attempted
specification in preassigned category \(i\), \(A_f\) is the declared complete
case-acceptance event for family \(f\), and \(w_i\) is the accepted-quota
weight, then

\[
p_{\mathrm{release}}(i,x)
=w_i p_{\mathrm{proposal}}(x\mid i,A_f=1).
\]

For Stokes, \(A_f\) comprises declared static support, a finite state, positive
water column, and a finite target. For Tanaka, it comprises declared
construction support followed by complete autonomous-trajectory acceptance;
for Benjamin--Feir, initial-state validity followed by complete autonomous-
trajectory acceptance. For JONSWAP/TMA, it comprises initial construction and
graph validity followed by nonlinear adjustment and complete autonomous-
trajectory acceptance. Thus the accepted quota fixes each category marginal,
not the unconditioned proposal law within the category. This identity is a
release-construction statement, not a claim that a within-category shift has
been measured for every family. Rejected proposals remain recorded, and no
retrospective parameter cutoff is applied.

The outer loop is finite. If category \(i\) has accepted target \(Q_i\), its
attempt ceiling is

\[
C_i=4Q_i.
\]

An attempt means one case assignment in a durable proposal. It does not mean
an internal Stokes amplitude redraw or one of the eight GL2 substeps between
saved states. A
proposal reserves its attempt exactly once, including after an interruption,
and a pending proposal at the ceiling must be replayed. If the pending work
has resolved, the accepted count is still below \(Q_i\), and the durable
attempt count equals \(C_i\), the run stops and names the exhausted
category. It cannot schedule a fifth attempt for a one-case category. The factor four
and the resulting per-category ceilings are part of the run fingerprint; changing
them cannot silently resume the same output root.

This ceiling is a computational fail-safe, not another sample filter. If it
is reached, the entire family run is incomplete and cannot enter a combined
dataset. If it is not reached, it has no effect on the deterministic sequence
of proposed or accepted cases. The factor four bounds worst-case attempted
cases by four times the requested accepted count while leaving three
replacement opportunities for a one-case pilot category.

Each attempted batch is a transaction:

```text
proposal NPZ -> optional accepted-case field shard NPZ -> result JSON
```

The proposal, including split, parameter category, random-stream coordinates, and complete
sampled specification, is written before construction. The result JSON is
the commit marker. Proposal-only and proposal-plus-shard interruptions replay
the identical attempt. A result that names a missing shard or a shard with a
different SHA-256 digest is treated as corruption. The quota driver performs
one full scan when it starts and advances committed batches in memory; a
restart performs the full integrity scan again.

Every run fingerprint binds its quota, family, split, batch size, platform,
random stream, exact numerical settings, Python/JAX/JAXLIB/NumPy versions,
dependency-file hashes, and source-file hashes. A changed command or
environment cannot silently resume an old run.

The combined view does not copy field arrays. It verifies every proposal,
result, and shard hash and writes a compact row-to-case map. Within each
included split it requires all four families, equal accepted counts, and
gap-free nested chunks beginning at zero. It also requires compatible common
target, production-integration, method-audit, storage, dependency, and source
identities.

The schema-v2 training loader uses the split assigned before numerical work.
In each training epoch, it selects exactly one stored time from every
accepted training case. Thus every family and every case has equal epoch
weight even though the stored row counts differ. Validation uses one fixed,
deterministic stored time from every accepted validation case, so checkpoint
values are comparable across epochs. Normalization extrema are computed over
all stored rows in the training split, rather than over the one-time-per-case
epoch draw, and the cached statistics are bound to the dataset identity.

## 6. Split sizes and nested training chunks

Validation and test are fixed at 1024 accepted cases per family. Training is
an additive learning curve:

| chunk | accepted cases before, \(C_0\) | new accepted cases, \(A\) | cumulative cases per family | stream ID |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 2048 | 2048 | 0 |
| 2 | 2048 | 2048 | 4096 | 1 |
| 3 | 4096 | 4096 | 8192 | 2 |
| 4 | 8192 | 8192 | 16384 | 3 |

Let \(B_i(C)\) be the balanced cumulative quota of category \(i\) at family count
\(C\). A chunk beginning at \(C_0\) and adding \(A\) cases receives

\[
B_i(C_0+A)-B_i(C_0)
\]

cases from category \(i\). This difference is essential for the 66- and
27-category
families: balancing every increment independently would repeatedly favor the
same remainder categories. Every additive chunk uses a distinct output root and
stream ID. Validation and test each use one fixed, non-nested chunk.

### Storage estimates

The current writer uses uncompressed `np.savez` storage. Approximate archive,
row-map, and current eager-loader sizes are:

| \(C_{\mathrm{tr}}\) per family | total retained rows, including fixed validation/test | archive plus row map | steady loaded arrays | estimated loader peak |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 1,708,032 | 19.62 GiB | 19.63 GiB | 39.6 GiB |
| 4096 | 2,562,048 | about 29.4 GiB | about 29.4 GiB | about 59.4 GiB |
| 8192 | 4,270,080 | about 49.1 GiB | about 49.1 GiB | about 99.0 GiB |
| 16384 | 7,686,144 | about 88.3 GiB | about 88.3 GiB | about 178 GiB |

These estimates include
\(C_{\mathrm{va}}=C_{\mathrm{te}}=1024\) per family. Actual proposal and
decision overhead depends on the number of rejected attempts. The audited
workstation had 6.2 TiB of free disk and 417 GiB of available host memory at
the final readiness check. It therefore has ample capacity for the complete
16384-case-per-family target. That target is comparable in snapshot count to
the historical v9 dataset: 7,686,144 rather than 7,633,989 rows, a difference
of 52,155 rows (0.68%).

## 7. Evidence already complete

- A CPU population-law smoke sampled 500,000 revision-1 specifications:
  125,000 from each family. It found exact then-current category quotas, no
  support violations, no outer case rejection, and no finite-Stokes redraw
  exhaustion. It is historical evidence and does not test the revision-2
  eleven-category Tanaka law.
- Under bounded run-spec schema 2 and generator revision 1, the dedicated
  static-Stokes quota launcher accepted one case in every Stokes category at
  \((N,M,p,K)=(1024,6,8,128)\), wrote the complete transaction and view, and
  loaded four finite rows. The current unified launcher independently
  accepted one exact `finite_low` Stokes case. Their output root is
  `outputs/paper_dataset_bounded_scheduler_smoke_20260726`. These artifacts
  remain historical transaction evidence.
- The generator-revision-2 exact Stokes pilot accepted one case in
  each of the four categories, with `failed_bits=0` for every case, in
  \(4.6875\) seconds. Its output root is
  `outputs/static_stokes_exact_target_pilot_revision2_20260727`, and its
  configuration fingerprint is `0efa9d71...a0dec`.
- Under the final bounded scheduler but generator revision 1, the reduced
  real-GL2 transaction took one case from each trajectory family through
  proposal, the then-current paired validation path, selection, archive,
  view, and loader. All three were accepted. Its reduced spatial and temporal
  settings make it historical software-wiring evidence, not evidence for
  exact-contract accuracy or single-rollout production timing.
- The 12-case exact full-horizon method panel compared actual GL2 steps
  \(0.01\) and \(0.005\). All 10 cases that completed at both steps had
  \(E\leq1.126337\times10^{-5}\). The steep Tanaka and equation-(33)
  Benjamin--Feir stress cases became incomplete at essentially the same
  physical time at both steps. No decision changed and no \(0.0025\)
  diagnostic retry ran. This corroborates the production step; it is not an
  estimate of population rejection probability. Across this panel, the
  recorded \(0.01\)-arm compute time was \(4026.06\) seconds and the
  \(0.005\)-arm compute time was \(7798.47\) seconds. The former pair
  therefore used \(11824.53\) arm-seconds, whereas one \(0.01\) arm used
  34.05% of that total. This measures the redundant work removed on this
  panel; it is not an exact-contract GPU throughput estimate.
- Before the one-rollout source change, the command

  ```bash
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
    uv run python -m unittest discover -s solver/gen_data -p 'test*.py'
  ```

  ran 181 tests and all passed. Separately, the unified launcher tests passed
  5/5 and the combined-view tests passed 4/4. The three training integration
  scripts for dataset identity/view, case-level split, and case-balanced
  validation each printed `PASS`. After the one-rollout source change, the
  complete discovery command passed 185/185 tests in \(107.617\) seconds.
  After the final contract-field and constant-name cleanup, the focused
  production, writer, quota, launcher, and view tests passed 27/27.
- The variable-horizon refinement test verifies that the method-level
  validation utility gives batched JONSWAP cases the same accepted decisions,
  retry routes, retained fields, and prefix telemetry as separate serial
  calls, even when values after a shorter case's declared horizon are
  deliberately poisoned. Production uses the corresponding variable-horizon
  one-rollout path instead.
- The first real four-summary assembly exposed an immutable-JSON boundary:
  parameter-category lists and nested dictionaries are frozen inside a run specification.
  The builder now validates the strict thawed JSON record while the quota
  scanner retains the immutable object. A regression test covers this
  boundary, and the real four-family view then built successfully.
- One inline, read-only reduced CPU benchmark at \(N=32\) and \(M=0\) gave
  serial repeat times \([3.731,3.898,3.719]\) seconds and batched repeat
  times \([2.043,1.861,1.901]\) seconds. The median ratio is
  \(3.731/1.901=1.963\). No benchmark artifact was saved. This number supports
  batching as an implementation direction, but it is not an estimate of
  exact-contract GPU throughput.

## 8. Exact-contract smoke results

The following four-family calculation completed immediately before the
attempt ceiling was added and used the former pair-per-case executor. Its
\(0.01/0.005\) defects are useful method-validation evidence, and its
transactions test the physical constructors, target, field writer, combined
view, and loader. The reported rollout wall times include both time-step arms,
and the listed \(0.005\) step is the trajectory retained by that old
validation path. Neither number is a production timing measurement under the
single-\(0.01\)-rollout policy. The changed run specification means these
run-spec-v1 artifacts cannot resume under the current code.

| item | result |
| --- | --- |
| output roots and identities | family runs: `outputs/paper_dataset_exact_smoke_final_20260726`; combined view: `outputs/paper_dataset_exact_smoke_final_view_20260726`; the four run fingerprints are `cbe29081...a189b7`, `45110059...98ebc`, `e6c83d27...a3788`, and `33f8e3c8...f6f4`; the combined dataset-contract fingerprint is `160b1702...2527` |
| Stokes attempted / accepted / rows / wall time | \(1/1/1\), \(3.782\) seconds |
| Tanaka attempted / accepted / rows / validation \(E\) / old retained step / paired wall time | \(1/1/200\), \(3.1953\times10^{-8}\), \(0.005\), \(5555.35\) seconds |
| Benjamin--Feir attempted / accepted / rows / validation \(E\) / old retained step / paired wall time | \(1/1/200\), \(3.7058\times10^{-8}\), \(0.005\), \(5561.11\) seconds |
| JONSWAP/TMA attempted / accepted / rows / validation \(E\) / old retained step / paired wall time | \(1/1/16\), \(7.6600\times10^{-7}\), \(0.005\), \(835.19\) seconds; realized horizon \(24.88\) |
| combined dataset-schema-v2 view (generator revision 1) | 4 accepted cases, 417 finite rows on 1024 points, all assigned to validation; manifest SHA-256 `b5533dd1...5089a` |
| loader audit | physical-family row counts \(1,200,200,16\); one accepted case per family; one selected time per case; fixed validation count 4; all fields finite; map and dataset identities verified |
| exact smoke decision | **PASS for the exact target, paired method-validation, transaction, combined-view, integrity, and loader behavior tested under run-spec v1.** It did not test the then-new one-rollout production executor. At this July checkpoint, the all-category pilot in Section 9 was intended to be that executor's first exact trajectory-population calculation. |

The following bounded-scheduler checks also predate the one-rollout
correction. They remain transaction and restart evidence, but the trajectory
smoke is not current production-timing evidence:

| item | bounded run-spec-schema-2, generator-revision-1 result |
| --- | --- |
| unified exact Stokes launcher | `outputs/paper_dataset_bounded_scheduler_smoke_20260726/unified_stokes`; attempted/accepted/rows \(1/1/1\); fingerprint `5dbae7f7...a70e4`; complete in \(3.06\) seconds |
| exact four-category Stokes quota | `outputs/paper_dataset_bounded_scheduler_smoke_20260726/stokes_all_cells`; attempted/accepted \(4/4\), exactly one in every category; all fields finite and trainer load passed; fingerprint `5d698085...8e20` |
| reduced real-GL2 trajectory transaction | `outputs/paper_dataset_bounded_scheduler_smoke_20260726/trajectory_reduced`; former paired path; attempted/accepted \(3/3\); strict summary SHA-256 `e17dbb13...8db5` |
| bounded restart tests | a resolved deficient category stops exactly at its ceiling; a pending final proposal replays once; a changed multiplier fails the configuration fingerprint; a batch cannot cross a per-category ceiling |

## 9. Historical all-category trajectory GPU pilot plan (do not run)

The following commands record the July plan.  They are not compatible with
the current family revisions or the dedicated adjusted JONSWAP/TMA launcher;
use the final authenticated release record above instead.  At that snapshot, static Stokes was
cleared by the generator-revision-2 exact four-category CPU pilot. The
remaining gate requested exactly one accepted case from every
trajectory parameter category: 11 Tanaka, 66 Benjamin--Feir, and 27
JONSWAP/TMA cases, for 104 cases in total. It is
deliberately separate from the training dataset and must not be included in the
final combined view. The proposed commands used batch size four so that the
random-sea run would exercise variable-horizon batching; the summaries were
intended to determine whether four was an appropriate production batch size.
A batch would have combined distinct cases in one accelerator call. Each case
would still have been integrated only once, with
saved interval \(0.08\), eight GL2 substeps per saved interval, and actual
step \(0.01\).

At 06:53 EDT on 2026-07-26, both local GPUs were at 100% utilization with
17.7 GiB and 25.3 GiB in use by other projects. The pilot was therefore not
launched. The plan was to start it only after both devices became idle, so its
memory and timing measurements would be attributable to this generator.

```bash
uv run python scripts/run_paper_dataset_quota.py \
  --family tanaka \
  --split validation \
  --accepted-cases 11 \
  --accepted-cases-before 0 \
  --batch-size 4 \
  --maximum-attempts-per-accepted-case 4 \
  --stream-id 900 \
  --output-root outputs/paper_dataset_all_category_gpu_pilot_20260727/tanaka \
  --platform gpu \
  --execute

uv run python scripts/run_paper_dataset_quota.py \
  --family benjamin_feir \
  --split validation \
  --accepted-cases 66 \
  --accepted-cases-before 0 \
  --batch-size 4 \
  --maximum-attempts-per-accepted-case 4 \
  --stream-id 900 \
  --output-root outputs/paper_dataset_all_category_gpu_pilot_20260727/benjamin_feir \
  --platform gpu \
  --execute

uv run python scripts/run_paper_dataset_quota.py \
  --family jonswap_tma \
  --split validation \
  --accepted-cases 27 \
  --accepted-cases-before 0 \
  --batch-size 4 \
  --maximum-attempts-per-accepted-case 4 \
  --stream-id 900 \
  --output-root outputs/paper_dataset_all_category_gpu_pilot_20260727/jonswap_tma \
  --platform gpu \
  --execute
```

The revision-1 read-only forms of the three predecessor commands were executed
on 2026-07-26.
This was an unsaved terminal observation under the former production source,
not a durable run artifact. It reported the intended two-GPU backend, no
prior transactions, quota exactly one in each of the then-current 4, 66, and
27 categories, and attempt ceiling four in every nonzero category. At that
checkpoint, the Tanaka law had advanced to revision 2 with eleven categories,
and the production executor had changed from paired refinement to one rollout. Those old
fingerprints are therefore obsolete. The July plan required a new read-only
preflight before adding `--execute`.

Under that plan, each command would first be run unchanged without `--execute`
to obtain a read-only preflight. The gate would pass only if:

1. every summary is complete, with no pending transaction, batch failure, or
   attempt-limit failure;
2. every category has accepted count one;
3. all retained arrays and required decision fields are finite;
4. every rejected attempt is either a declared finite Tanaka-radicand
   construction exclusion or has decoded failed bits identifying why no
   complete admissible trajectory existed; an unexpected construction or
   target exception stops the run, and no per-case refinement defect or retry
   is required;
5. the dependency environment, common numerical target, and storage contract
   agree; hashes agree for source paths shared across families; and every
   family-specific source map is constant within that family and revision;
6. each trajectory case records the frozen saved interval \(0.08\), eight
   substeps, and actual production step \(0.01\);
7. wall time from each summary and peak GPU memory from an external monitor
   are recorded so the production batch size is chosen from measurement.

The pilot summaries have unequal family counts by design, so item 5 is a
separate summary audit rather than a call to the equal-family combined-view
builder. The generator recorded wall time but not peak GPU memory. The proposed
external monitor was:

```bash
mkdir -p outputs/paper_dataset_all_category_gpu_pilot_20260727
nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv \
  --loop=5 \
  > outputs/paper_dataset_all_category_gpu_pilot_20260727/gpu_telemetry.csv
```

## 10. Historical combined-view command templates (do not run)

The paths below record the July layout and are not the current literature-
aligned roots.  At that snapshot, the following output-root layout was
proposed for the first checkpoint view,
\((C_{\mathrm{tr}},C_{\mathrm{va}},C_{\mathrm{te}})
=(2048,1024,1024)\):

```text
outputs/paper_dataset/train/<family>/chunk_00000_02048
outputs/paper_dataset/validation/<family>/c01024
outputs/paper_dataset/test/<family>/c01024
```

The July plan called for running the read-only form of the following command
after all twelve checkpoint summaries were complete, then repeating it with
`--execute` to write the first all-split manifest:

```bash
uv run python scripts/build_paper_dataset_view.py \
  --chunk-summary outputs/paper_dataset/train/stokes/chunk_00000_02048/paper_dataset_stokes_train.summary.json \
  --chunk-summary outputs/paper_dataset/train/tanaka/chunk_00000_02048/paper_dataset_tanaka_train.summary.json \
  --chunk-summary outputs/paper_dataset/train/benjamin_feir/chunk_00000_02048/paper_dataset_benjamin_feir_train.summary.json \
  --chunk-summary outputs/paper_dataset/train/jonswap_tma/chunk_00000_02048/paper_dataset_jonswap_tma_train.summary.json \
  --chunk-summary outputs/paper_dataset/validation/stokes/c01024/paper_dataset_stokes_validation.summary.json \
  --chunk-summary outputs/paper_dataset/validation/tanaka/c01024/paper_dataset_tanaka_validation.summary.json \
  --chunk-summary outputs/paper_dataset/validation/benjamin_feir/c01024/paper_dataset_benjamin_feir_validation.summary.json \
  --chunk-summary outputs/paper_dataset/validation/jonswap_tma/c01024/paper_dataset_jonswap_tma_validation.summary.json \
  --chunk-summary outputs/paper_dataset/test/stokes/c01024/paper_dataset_stokes_test.summary.json \
  --chunk-summary outputs/paper_dataset/test/tanaka/c01024/paper_dataset_tanaka_test.summary.json \
  --chunk-summary outputs/paper_dataset/test/benjamin_feir/c01024/paper_dataset_benjamin_feir_test.summary.json \
  --chunk-summary outputs/paper_dataset/test/jonswap_tma/c01024/paper_dataset_jonswap_tma_test.summary.json \
  --output-root outputs/paper_dataset/combined/c02048_v01024_t01024 \
  --name paper_dataset_all_splits_c02048 \
  --execute
```

For the proposed final \(C_{\mathrm{tr}}=16384\) view, the template included
all four training chunks for every family and reused the fixed validation and
test summaries:

```bash
uv run python scripts/build_paper_dataset_view.py \
  --chunk-summary outputs/paper_dataset/train/stokes/chunk_00000_02048/paper_dataset_stokes_train.summary.json \
  --chunk-summary outputs/paper_dataset/train/stokes/chunk_02048_02048/paper_dataset_stokes_train.summary.json \
  --chunk-summary outputs/paper_dataset/train/stokes/chunk_04096_04096/paper_dataset_stokes_train.summary.json \
  --chunk-summary outputs/paper_dataset/train/stokes/chunk_08192_08192/paper_dataset_stokes_train.summary.json \
  --chunk-summary outputs/paper_dataset/train/tanaka/chunk_00000_02048/paper_dataset_tanaka_train.summary.json \
  --chunk-summary outputs/paper_dataset/train/tanaka/chunk_02048_02048/paper_dataset_tanaka_train.summary.json \
  --chunk-summary outputs/paper_dataset/train/tanaka/chunk_04096_04096/paper_dataset_tanaka_train.summary.json \
  --chunk-summary outputs/paper_dataset/train/tanaka/chunk_08192_08192/paper_dataset_tanaka_train.summary.json \
  --chunk-summary outputs/paper_dataset/train/benjamin_feir/chunk_00000_02048/paper_dataset_benjamin_feir_train.summary.json \
  --chunk-summary outputs/paper_dataset/train/benjamin_feir/chunk_02048_02048/paper_dataset_benjamin_feir_train.summary.json \
  --chunk-summary outputs/paper_dataset/train/benjamin_feir/chunk_04096_04096/paper_dataset_benjamin_feir_train.summary.json \
  --chunk-summary outputs/paper_dataset/train/benjamin_feir/chunk_08192_08192/paper_dataset_benjamin_feir_train.summary.json \
  --chunk-summary outputs/paper_dataset/train/jonswap_tma/chunk_00000_02048/paper_dataset_jonswap_tma_train.summary.json \
  --chunk-summary outputs/paper_dataset/train/jonswap_tma/chunk_02048_02048/paper_dataset_jonswap_tma_train.summary.json \
  --chunk-summary outputs/paper_dataset/train/jonswap_tma/chunk_04096_04096/paper_dataset_jonswap_tma_train.summary.json \
  --chunk-summary outputs/paper_dataset/train/jonswap_tma/chunk_08192_08192/paper_dataset_jonswap_tma_train.summary.json \
  --chunk-summary outputs/paper_dataset/validation/stokes/c01024/paper_dataset_stokes_validation.summary.json \
  --chunk-summary outputs/paper_dataset/validation/tanaka/c01024/paper_dataset_tanaka_validation.summary.json \
  --chunk-summary outputs/paper_dataset/validation/benjamin_feir/c01024/paper_dataset_benjamin_feir_validation.summary.json \
  --chunk-summary outputs/paper_dataset/validation/jonswap_tma/c01024/paper_dataset_jonswap_tma_validation.summary.json \
  --chunk-summary outputs/paper_dataset/test/stokes/c01024/paper_dataset_stokes_test.summary.json \
  --chunk-summary outputs/paper_dataset/test/tanaka/c01024/paper_dataset_tanaka_test.summary.json \
  --chunk-summary outputs/paper_dataset/test/benjamin_feir/c01024/paper_dataset_benjamin_feir_test.summary.json \
  --chunk-summary outputs/paper_dataset/test/jonswap_tma/c01024/paper_dataset_jonswap_tma_test.summary.json \
  --output-root outputs/paper_dataset/combined/c16384_v01024_t01024 \
  --name paper_dataset_all_splits_c16384 \
  --execute
```

The proposed \(C_{\mathrm{tr}}=4096\) and \(8192\) views used the same
template, stopping after the second and third training chunks, respectively.
The July plan did not enlarge a completed quota in place.

## 11. After generation, before training

At \(C_{\mathrm{tr}}=2048\), schema-v2 training presents

\[
4C_{\mathrm{tr}}=8192
\]

case samples per epoch. With global batch size 1024, this is only eight
optimizer steps per epoch. Therefore a fixed 40-epoch schedule would give
only 320 updates at the smallest learning-curve point, while the
\(C_{\mathrm{tr}}=16384\) target would give 2560 updates.

Learning-curve models must be compared at a fixed optimizer-step budget, not
at 40 fixed epochs. The current audit uses 10,240 updates as the common
comparison budget:

| \(C_{\mathrm{tr}}\) per family | steps per epoch at batch 1024 | epochs for 10,240 updates |
| ---: | ---: | ---: |
| 2048 | 8 | 1280 |
| 4096 | 16 | 640 |
| 8192 | 32 | 320 |
| 16384 | 64 | 160 |

The final training launch must record both epochs and optimizer steps and
must make the learning-rate schedule a function of the common step budget.
Keep normalization extrema computed from every stored training row and keep
the fixed one-time-per-case validation draw. These two choices are distinct:
the former covers the full stored training support, while the latter gives
equal case and family weight to checkpoint comparison.

Use final checkpoints, or compare checkpoints at the same predeclared update
numbers. Choosing the best validation value after every epoch would give the
2048-case run eight times as many selection opportunities as the 16384-case run,
even though their optimizer budgets match.

The current general test path evaluates stored rows, so with equal case
counts its approximate family weights are 0.24% Stokes, 47.96% Tanaka,
47.96% Benjamin--Feir, and 3.84% JONSWAP/TMA. That row-weighted scalar is not
a paper-level four-family result. Report test errors by case within each
family and then give each family equal weight. These training and reporting
requirements do not alter the generation acceptance decision.
