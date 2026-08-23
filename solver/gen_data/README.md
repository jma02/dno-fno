# Data-generation contract

The physical families have different initial-condition mathematics.  They
share the delivered DNO label, whole-case quality rule, storage format, and
case-level split rule.  Constructor sources and rollout settings are fixed
separately for each `(family, revision)` pair.  The present dataset uses static
Stokes revision 2, corrected Tanaka revision 3, Benjamin--Feir revision 4, and
JONSWAP/TMA revision 4.  Earlier family revisions and the superseded Tanaka
rows remain historical evidence and cannot be mixed with their replacements.
A generator shard is not a physical family: historical names such as `g0`,
`g1`, and `modal` identify independent runs only.

Five terms are used consistently here:

- a **constructor** maps one stored case specification to its periodic initial
  state;
- a **parameter category** is a predeclared part of one family's parameter
  range with its own accepted-case quota (the code stores its name as
  `cell_id`);
- a **validation panel** is a small, predeclared set of cases used to test a
  numerical contract and does not produce training data;
- a **writer** stores proposals, immutable field shards, decisions, and failure
  records;
- the **dataset** is the collection of accepted cases exposed to the loader by
  a dataset view.

## Order of operations

For rollout-derived data, the unit of generation and rejection is a complete
candidate trajectory:

1. assign the attempted case its physical family, generator revision,
   parameter category, train/validation/test split, and random-stream key;
2. sample its family-specific parameters and atomically store the complete
   proposal before constructing a field;
3. construct the periodic initial state on the declared grid, rejecting only
   a declared failure of the family's mathematical construction domain;
4. for JONSWAP/TMA only, evolve the linear random sea through the declared
   nonlinear-adjustment interval, retain its complete \(|k|\leq704\) endpoint,
   and reject the attempt if that handoff is not finite, graph-valid, and
   stage-converged;
5. start the autonomous clock at zero and run the family-revision integrator
   with GL2 step \(0.01\), saving every \(0.08\);
6. evaluate the common delivered target and the required internal numerical
   diagnostics on the saved autonomous grid;
7. reject the case if and only if no complete admissible trajectory exists on
   its requested interval;
8. select training times only from the retained accepted trajectory;
9. take the selected
   \((\eta,\xi,G_{\mathrm{ref}}(\eta)\xi)\) rows;
10. commit only complete accepted trajectories, while retaining a decision for
   every attempted case;
11. build a dataset view that uses the split assigned in step 1 rather than
   making a row-level split.

Static Stokes cases use the same target and source-record schema, but each
independent state is its own quality unit.  Airy states appear only in
method-validation tests; they are not a fifth dataset family.  Activity-based
temporal sampling changes which accepted frames are stored; it is not a
quality filter.  The paper dataset target is the float64 order-six, pad-eight
Craig--Sulem value projected to the fixed delivered band \(|k|\leq128\), then
stored in float32.

## Quality semantics

`pipeline.QualityDecision` keeps three masks:

- `required`: checks that define acceptance under the declared policy;
- `evaluated`: checks that were actually run;
- `failed`: evaluated checks that failed.

A unit is accepted only when every required check was evaluated and no
required check failed.  Separate masks prevent old, unevaluated data from
being described as having passed a newer audit.  The stable reason vocabulary
is defined in `pipeline/quality.py`.

After a proposal is stored, acceptance has two stages.  First, the
family-specific formulas must produce an initial state in their declared
mathematical construction domain.  Second, a trajectory family must produce
one complete admissible trajectory under the frozen numerical method.
Unexpected constructor or solver errors stop generation; they are not
converted into rejected cases.

Every initial condition is sampled from a parameter set declared before
generation.  For finite-depth fifth-order Stokes states, let \(H\) be the
physical crest-to-trough height of the complete implemented series and
\(\lambda\) its carrier wavelength.  The code evaluates the series on 4096
equally spaced phases and adds the Taylor bound
\(\frac14(2\pi/4096)^2\sum_{m=1}^5m^2|E_m|\), where \(E_m\) is the
\(m\)-th elevation harmonic.  Denote the resulting upper bound by \(H_+\).
The support condition is

\[
\mathrm{Ur}_+=\frac{H_+\lambda^2}{h^3}\leq26.
\]

Taylor's theorem gives \(H\leq H_+\), so this conservatively enforces the
physical Ursell condition.  Every rejected amplitude and its
\(\mathrm{Ur}_+\) value are stored with the accepted case.  This is part of
the definition of the Stokes family, not an outcome-dependent cleaning rule.
Harmonic ordering remains a recorded diagnostic but is not a support gate.

The Stokes sampler draws the spatial phase directly as
\(\phi\sim\operatorname{Unif}[0,2\pi)\) and constructs the snapshot at
\(kx+\phi\); it does not use a sampled time as a proxy for phase.  Before the
draw, the common scheduler assigns one of four parameter categories:
finite/deep crossed with
\(ka\in[0.005,0.03)\) or \(ka\in[0.03,0.15]\).  The sampler chooses a
feasible integer carrier mode, a mode-conditioned depth, the phase, and then
an amplitude inside the assigned category.  An Ursell rejection redraws only
the amplitude, so the carrier, depth, phase, split, and category do not change.
The float64 constructed surface potential has zero spatial mean; float32
storage preserves this only to rounding error.

NumPy PCG64 uses all five coordinates of the stored case key: root seed,
physical-family identifier, generator revision, stream identifier, and
attempt index.  This is a reproducible stratified pseudorandom sampler, not a
low-discrepancy sampler.  If the initial same-category amplitude draw and all
1000 allowed redraws fail, the sampler raises an exception whose strict record
contains the fixed parameters and every rejected amplitude and
\(\mathrm{Ur}_+\) value.  This is a declared case rejection, not a fatal batch
error: `stokes_quota_executor.py` puts that record in the proposal, commits a
zero-row `OUTSIDE_SUPPORT` decision, keeps valid siblings, and schedules a
replacement in the same category.  An unclassified sampler error propagates
and is not relabeled as a physical rejection.  The analytic deep-water branch
additionally requires \(kh\geq5\); the finite-depth branch uses the
conservative Ursell condition above.

For a main Tanaka case, first draw the total dimensionless amplitude

\[
A\sim\operatorname{Unif}[0.10,0.35].
\]

If the case has \(m\in\{1,2,3\}\) crests, draw
\((W_1,\ldots,W_m)\sim\operatorname{Dirichlet}(1,\ldots,1)\) and set
\(\alpha_j=AW_j\).  For \(m=1\), this means \(W_1=1\).  Thus
\(\sum_j\alpha_j=A\).  Define

\[
\alpha_{\max}=\max_j\alpha_j,
\qquad
\kappa_{\max}=\frac{\sqrt{3\alpha_{\max}}}{2h},
\qquad
R=\frac{128}{\kappa_{\max}}.
\]

Here \(\kappa_{\max}\) is the inverse-width scale obtained from the
small-amplitude KdV solitary-wave formula.  It is used only as an elementary
resolution proxy for the narrowest requested crest; it is not asserted to be
the exact width of a finite-amplitude Tanaka wave.  Revision 3 requires
\(R\geq10\).  With \(h_{\rm base}=0.01\), \(h_{\max}=0.30\), define

\[
h_{\min}
=\max\!\left(h_{\rm base},\frac{10\sqrt{3\alpha_{\max}}}{256}\right),
\qquad
h=h_{\min}^{1-U}h_{\max}^{U},
\qquad U\sim\operatorname{Unif}[0,1].
\]

The steep one-crest branch uses the same construction with
\(A=\alpha_1\sim\operatorname{Unif}[0.25,0.45]\),
\(h_{\rm base}=0.20\), and \(h_{\max}=0.35\).  Its conditional term is
always below \(0.20\), so its depth law remains the ordinary geometric
interpolation from \(0.20\) to \(0.35\).

Let \(s_j\in\{-1,+1\}\) be the direction of crest \(j\), with \(s_j=+1\)
denoting rightward travel, and let
\(q=\#\{j:s_j=+1\}\) be the number of right-moving crests.  The main pairs
\((m,q)\), \(q=0,\ldots,m\), define \(2+3+4=9\) parameter categories.
The steep one-crest branch adds the two categories \(q=0,1\).  Conditional
on \(q\), the sampler chooses uniformly which \(q\) of the \(m\) exchangeable
crest labels move right.  Accepted quotas are divided as evenly as possible
across all eleven categories, so their counts differ by at most one.  The
possible direction compositions receive equal coverage in the ideal category
law.  The quotient--remainder
assignment can introduce a one-case direction imbalance for a finite quota;
with the fixed category order, a quota of 1024 has one extra left-moving
one-crest case.  This replaces the old four-group independent-sign mixture
and deliberately gives more aggregate weight to two- and three-crest cases.

The direction categories were introduced in revision 2.  Revision 3 retains
all eleven categories and changes the amplitude--depth draw: amplitudes are
drawn first, then depth is drawn directly from the interval satisfying
\(R\geq10\).  Its stored specification is `tanaka_sample_spec_v3`.
A revision-2 Tanaka proposal remains exactly replayable, but it cannot be
resumed into a revision-3 run.

A full-horizon boundary panel used
\(\alpha\in\{0.225,0.30,0.35\}\), \(R\in\{6,8,10\}\), and internal cutoffs
\(J\in\{224,256\}\).  At saved time \(t\), define

\[
D_J(t)=P_{128}^{\circ}\!\left[
\mathcal G_6^{1024,8}(\eta_J(t);h)\xi_J(t)
\right],
\qquad
Q_J(t)=G_{\mathrm{ref}}(P_{128}\eta_J(t);h)P_{128}\xi_J(t),
\]

and, with \(\|\cdot\|_{2,N}\) denoting the root-mean-square norm on the
spatial grid,

\[
C_J=
\frac{\max_t\|D_J(t)-Q_J(t)\|_{2,N}}
     {\max_s\|Q_J(s)\|_{2,N}+10^{-12}}.
\]

The reported boundary statistic is \(\max_{J\in\{224,256\}}C_J\).  The three
\(R=6\) cases exceeded the \(10^{-3}\) budget (worst value
\(2.91\times10^{-3}\)).  All three \(R=8\) cases passed, but the worst value
was \(6.31\times10^{-4}\), leaving little margin.  At \(R=10\), the worst
value was \(1.40\times10^{-4}\).  Comparing the same cutoff-224 and cutoff-256
histories directly gave the same ordering: \(R=6\) failed, while the worst
delivered-field discrepancies at \(R=8\) and \(R=10\) were respectively
\(7.34\times10^{-4}\) and \(1.35\times10^{-4}\).  The revision-3 lower bound
therefore uses 10 rather than the merely passing value 8.

The traveling-wave surface-potential formula contains a
square root.  When every wave speed and radicand entry is finite, a component
with a negative radicand is outside that formula's declared construction
domain and is recorded as a zero-row `OUTSIDE_SUPPORT` case before the square
root is taken.  A nonfinite radicand, a nonpositive or nonfinite wave speed,
or an unexpected constructor exception instead stops generation.  Such
failures are not treated as new random draws from the Tanaka population.

The constructed Tanaka fields are sharply projected once to \(P_{128}\)
before rollout and then zero-filled into the wider \(K=256\) evolution band.
Thus modes \(129\leq |k|\leq256\) are zero initially but may be generated by
nonlinear evolution.

After construction succeeds, every production trajectory is integrated
once.  Revision-3 Tanaka evolution retains internal modes \(|k|\leq256\).
After every GL2 step it multiplies the Fourier coefficients of both \(\eta\)
and \(\xi\) by

\[
\sigma_k=\exp\!\left[-36\left(\frac{|k|}{256}\right)^{36}\right].
\]

The exact revision-3 path is therefore \(P_{128}\) construction, zero-filled
lift to the \(K=256\) evolution band, this Hou--Li multiplier on both state
fields after every GL2 step, and \(P_{128}\) delivery to the dataset.

For the revision-4 Benjamin--Feir accepted-quota population, let \(n_c\) be
the carrier mode, \(\Delta n\) the symmetric sideband offset,
\(\varepsilon_c\) the carrier steepness, and \(\rho\) the sideband-to-carrier
amplitude ratio.  Define

\[
\beta=\frac{\Delta n/n_c}{2\sqrt{2}\,\varepsilon_c},
\qquad
\varepsilon_f=\varepsilon_c\left(1+2\sqrt{1-\beta^2}\right),
\qquad
F_0=\frac{1+\sqrt2}{10}.
\]

The declared support is \(n_c\in\{4,\ldots,20\}\),
\(\Delta n\in\{1,\ldots,n_c-1\}\),
\(\varepsilon_c\in[0.05,0.13]\), \(0<\beta<1\),
\(\varepsilon_f\leq F_0\), \(\rho\in[0.05,0.10]\), and
\(x_0\in[0,L)\).  Exactly 66 integer pairs have nonempty support.  Conditional
on one such pair, set

\[
\ell=\frac{\Delta n}{2\sqrt2\,n_c},\qquad
\varepsilon_{\min}=\max\{0.05,\ell\},\qquad
\varepsilon_{\max}=\min\left\{0.13,
\frac{-F_0+2\sqrt{F_0^2+3\ell^2}}{3}\right\}.
\]

The implementation draws
\(\varepsilon_c\sim\operatorname{Unif}
(\varepsilon_{\min},\varepsilon_{\max}]\),
\(\rho\sim\operatorname{Unif}[0.05,0.10)\), and
\(x_0\sim\operatorname{Unif}[0,L)\), with fixed relative sideband phase
\(-\pi/4\).  Each feasible pair is an accepted-quota category, and a rejected
attempt is replaced in the same pair.  This population sampler is distinct
from the generic standalone Benjamin--Feir archive helper described below.

Revision-4 Benjamin--Feir trajectories use 1024 internal grid points, a sharp
Galerkin band \(|k|\leq256\), and DNO-series order \(M=4\).  Revision-4
JONSWAP/TMA trajectories instead use 2048 internal grid points, the sharp band
\(|k|\leq704\), and order \(M=4\).  After every completed GL2 step, the two
state fields are sharply projected to \(P_{256}\) or \(P_{704}\), respectively.
These repeated sharp projections enforce the declared finite-dimensional
evolution space; they are not Hou--Li filters and apply no smooth damping
inside the retained band.  Their constructed \(P_{128}\) states are embedded
in the corresponding internal grid and band.  JONSWAP/TMA first performs the
nonlinear adjustment specified below; Benjamin--Feir starts autonomous
evolution immediately.
Revision-2 Stokes states remain static.  In every family, the arrays exposed
to the dataset have 1024 points and contain \(P_{128}\eta\), \(P_{128}\xi\),
and the common order-six \(G_{\mathrm{ref}}(\eta;h)\xi\) defined below.  The
wider internal bands and the order-four internal DNO are numerical workspace,
not additional learned features or targets.

The saved interval is
\(\Delta t_s=0.08\), and GL2 takes eight equal substeps between saved states.
Thus the actual GL2 step is

\[
\Delta t_{\mathrm{GL2}}=\frac{0.08}{8}=0.01.
\]

This is the same step frozen by the May time-step study.  That study varied
the saved interval while retaining eight substeps, so its reported choice
\(0.08\) corresponds to an actual GL2 step of \(0.01\).  Production does not
repeat that solver-selection study for every sampled case.

A production trajectory is admissible when it reaches the requested terminal
time, every \(\eta\), \(\xi\), and delivered target value is finite, every
saved state satisfies \(h+\eta>0\), and every implicit GL2 stage is solved to
the declared residual tolerance.  Revision-4 Benjamin--Feir and JONSWAP/TMA
add one numerical condition.  Let \(K_F=256\) for Benjamin--Feir and
\(K_F=704\) for JONSWAP/TMA.  On every saved autonomous state, compute the
internal order-four Hamiltonian in that family's sharp internal band,

\[
H_{K_F}(t)=\frac12\int_0^{2\pi}
\left\{\xi\,G^{K_F}_4(\eta;h)\xi+g\eta^2\right\}\,dx,
\qquad
d_{H,K_F}=\max_j
\frac{|H_{K_F}(t_j)-H_{K_F}(0)|}
{\max\{|H_{K_F}(0)|,\varepsilon_{64}\}},
\]

where \(\varepsilon_{64}\) is the smallest positive normal float64 number.
For these nonzero-energy states, the denominator is \(|H_{K_F}(0)|\).  These
trajectories require \(d_{H,K_F}\leq10^{-3}\).  This is evaluated on
the complete saved grid before training-time selection, not merely on the
selected rows or after projection to \(P_{128}\).  The post-rollout condition
is therefore:

\[
\text{reject the case if and only if no complete admissible trajectory
exists on }[0,T].
\]

A nonfinite field, nonfinite target, unsolved stage, loss of the graph domain,
or a required Hamiltonian-conservation failure is a recorded cause of that
single failure condition.  These are not empirical shape filters.  A rejected
trajectory owns zero rows; no finite prefix or isolated frame is kept.  The
accepted-quota scheduler replaces it deterministically in the same parameter
category.

Time-step comparison remains a method-level validation calculation.  It is
run on a small, predeclared panel rather than on every production case.  If
\(P_K\) denotes projection to the delivered band and \(P_K^\circ\)
additionally removes the mean, define

\[
G_{\mathrm{ref}}(\eta;h)\xi
=P_K^\circ\!\left[
\mathcal G_6^{1024,8}(P_K\eta;h)(P_K\xi)
\right].
\]

Here \(g=1\) is gravitational acceleration.  The dimensionless delivered
fields are

\[
Y_r(t)=\left(
\frac{P_K\eta_r(t)}{h},
\frac{P_K\xi_r(t)}{h\sqrt{gh}},
\frac{G_{\mathrm{ref}}(\eta_r(t);h)\xi_r(t)}{\sqrt{gh}}
\right),
\qquad \Delta t_r=2^{-r}\Delta t_0.
\]

Writing \(\mathcal U_r\) for the three components of \(Y_r\), the single
validation statistic is

\[
E_r=
\max_{u\in\mathcal U_r}
\frac{\max_j\lVert u_{r+1}(t_j)-u_r(t_j)\rVert_{L^2}}
     {\max_j\lVert u_{r+1}(t_j)\rVert_{L^2}+10^{-12}}.
\]

The validation pair uses actual GL2 steps \(0.01\) and \(0.005\).
The \(0.0025\) calculation is available only as a diagnostic retry in this
method-level utility; neither the second arm nor the retry is part of
production generation or either production acceptance stage.

The historical revision-2 12-case full-horizon panel corroborated the frozen
\(0.01\) production step.  All 10 cases that completed at both steps had
\(E_0\leq1.126337\times10^{-5}\), almost two orders of magnitude below
\(10^{-3}\).  The steep upper-seam Tanaka case and the equation-(33)
Benjamin--Feir stress case became incomplete at essentially the same physical
times under both steps.  No case changed decision under refinement, and no
\(0.0025\) retry ran.  This panel is retained as time-step evidence, not as
revision-3 Tanaka population evidence or an estimate of rejection rates.

Static Stokes cases do not undergo a fictitious temporal check.  They retain
only \(t=0\) after their declared support and static finite-state,
positive-water-column, and finite-target checks pass.
A returned nonfinite array therefore gives a zero-row decision, whereas an
unexpected exception from the Stokes constructor or target evaluator
propagates and stops generation.

Historical and method-validation panels may also report mass, spectra, and
stage iteration counts.  Those quantities are diagnostics only.  Hamiltonian
drift is a required production metric only for revision-4 Benjamin--Feir and
the autonomous part of revision-4 JONSWAP/TMA; it remains diagnostic for the
other family revisions.  The arbitrary half-depth margin is removed.

The three-grid fixed-band calculation and the other cutoff comparisons are
method-level tests, not outcome-dependent row filters.  Each result applies
only to the family, revision, and parameter panel named with it.  In
particular, the historical revision-3 JONSWAP/TMA cutoff sentinel does not by
itself validate the revision-4 relative-frequency population.

The old Tanaka sign-transition count is historical evidence for the clipped
zero-extension defect.  It is not part of the production acceptance rule.
Neither a slope/extremum count nor an absolute cap on
\(\lvert G(\eta;h)\xi\rvert\) is a numerical-health test.

The CPU smoke test exercises the finite-Stokes constructor and two actual
GL2 refinement pairs as method-level validation:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python -m unittest solver.gen_data.tests_paper_acceptance_smoke
```

This older smoke verifies the refinement-audit decision layer.  It does not
describe the one-rollout production path.  The production executor records
whether every implicit stage was solved; its focused tests are listed below.

Two additional method-level smokes are intentionally separate from
per-trajectory acceptance:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python -m unittest solver.gen_data.tests_paper_acceptance_cross_family

JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python -m unittest solver.gen_data.tests_stokes_spatial_smoke
```

The first checks constructor finiteness, graph validity, band limitation,
translation covariance of \(G(\eta;h)\xi\), and one actual \(\Delta t\) versus
\(\Delta t/2\) GL2 pair for the current finite-Stokes, tangent-Hermite Tanaka,
and Benjamin--Feir paths.  The Benjamin--Feir constructor
is `benjamin_feir_jcp09.py`.  It samples one of the 66 feasible
carrier--sideband integer pairs, the steepness of the carrier's first
elevation harmonic, a sideband amplitude ratio, and a global translation
\(x_0\in[0,L)\).  In the translated coordinate \(y=x-x_0\), the carrier phase
is zero and both sidebands have the fixed Xu--Guyenne relative phase
\(-\pi/4\).  A case is integrated through the last saved-grid time not
exceeding 100 linear carrier periods and stores 200 approximately uniform
saved-grid times.  The second test independently
constructs one Stokes state on \(N=32\) and \(N=64\) grids and compares
consistently normalized Fourier coefficients on the fixed band
\(|m|\leq12\).  It validates the retained reference formulas; it does not
reject individual samples.

Accepted-quota proposals use the population-record schema
`benjamin_feir_sample_spec_v3`; standalone archive members use the
parameter-record schema `bf_jcp09_parameter_spec_v2`.  Both use the constructor
identifier `jcp09_equation_33_with_project_fifth_order_carrier_v2`.  These
names separate the fixed \(-\pi/4\)-phase, globally translated construction
from historical records that used the v1 labels.  A v1 standalone archive or
Modal shard is not resumable or mergeable into v2 output.  The standalone path
binds its constructor, support, random stream, grid, solver, time selection,
storage, and run plan to immutable fingerprints.  It also reconciles mutable
progress with the exact member inventory and per-member SHA-256 digests before
continuing.  The accepted-quota paper path has its own stronger revision and
source-bound transaction contract described below.

The revision-4 random-sea paper constructor is `jonswap_tma.py`.  Let

\[
\omega(k,h)=\sqrt{k\tanh(kh)},\qquad \omega_p=\omega(k_p,h).
\]

It forms a cell-integrated JONSWAP density with the standard TMA depth factor
and retains the fixed relative-frequency interval

\[
\frac12\leq\frac{\omega(k,h)}{\omega_p}\leq\frac52.
\]

The indicator is evaluated at the 16 Gauss--Legendre nodes in each integer
Fourier cell, so a boundary cell can be only partly retained.  The support
also requires \(\omega(128,h)\geq(5/2)\omega_p\), ensuring that the complete
declared interval fits inside the delivered band.  This replaces the earlier
fixed-wavenumber cosine taper.  No realization-dependent spectral, slope,
crest/trough-height, or appearance *support* gate is used; the separate
graph-domain check below still requires \(\min_x(h+\eta_0)>0\).

Every depth stratum additionally requires
\(\epsilon_p=k_pH_s/2\leq0.08\).  This is a pre-phase restriction to the
moderate input-sea regime, not a wave-breaking test.  The shallow proposal sampler is
uniform on the part of its \((k_ph,H_s/(2h))\) rectangle satisfying both this
bound and the relative-frequency fit condition.  The finite and deep samplers
are uniform on the corresponding admissible parts of their original
\((k_p,h,H_s)\) boxes.  Random phases and later focusing can still make a
trajectory fail, so these parameter-support restrictions do not replace the
complete numerical calculation.

After the one-time projection, a JONSWAP/TMA realization is passed to the
solver only when both initial fields are finite and
\(\min_x(h+\eta_0)>0\).  This is the graph-domain requirement that the free
surface lie above the bottom, not an empirical roughness rule.  An invalid
realization becomes one zero-row attempt and is replaced only within its own
category; valid cases from the same numerical batch are preserved.  The
record also stores the realized discrete peak cell, half-maximum spectral
width, RMS fields, significant-height ratio, linear-Hamiltonian consistency,
and minimum water column as diagnostics rather than thresholds.

For JONSWAP/TMA specifically, these uniform parameter and phase laws govern
attempted specifications.  Its complete-case acceptance event includes
initial construction and graph validity, nonlinear adjustment, and autonomous
trajectory acceptance.  Accepted-case quotas preserve the category counts,
not the unconditioned continuous density inside a category.  This conditioning
is a consequence of the declared case decision, not an additional gate or a
retrospective parameter cutoff; every rejected proposal remains recorded.

Revision-4 JONSWAP/TMA then applies a nonlinear adjustment before data
collection.  With peak period \(T_p\), it sets

\[
A(t)=1-\exp[-(t/(10T_p))^4]
\]

in the Dommermuth ramped equations and evolves to the last saved-grid time not
exceeding \(20T_p\).  The complete 2048-point, \(K=704\), order-four endpoint
\((\eta,\xi)\) is handed directly to the autonomous solver: it is not first
projected to \(P_{128}\), and neither component is rescaled.  The autonomous
clock and Hamiltonian reference are both restarted at zero.  Because the
adjusted equations depend explicitly on time, Hamiltonian conservation is not
required during adjustment; finiteness, positive water depth, and successful
implicit stages are required.  The intended autonomous horizon is \(16T_p\);
the realized endpoint is the last saved-grid time not exceeding it and is
subject to the dense internal Hamiltonian condition above.

The GL2 production mode is residual-controlled.  JONSWAP/TMA allows at most
five simultaneous fixed-point updates per GL2 stage; the other trajectory
contracts retain their declared cap of four:

```python
rollout(
    ...,
    implicit_iterations=5,  # JONSWAP/TMA
    implicit_residual_tolerance=1e-8,
)
```

It stores per-step residual, convergence, iteration-cap, and finiteness
telemetry.  The frozen delivered target implementation is
`pipeline.reference.PAPER_DNO_TARGET`, namely
\((L,M,N,p,K)=(2\pi,6,1024,8,128)\).  Its evaluator requires JAX float64
mode and promotes stored float32 inputs before applying the target.  For a
JONSWAP/TMA saved state, normalized Fourier coefficients are formed as
\(\widehat f_k^{(N)}=N^{-1}\sum_{j=0}^{N-1}f_j e^{-2\pi i jk/N}\).
The coefficients of \(\eta\) and \(\xi\) with \(|k|\leq128\) are copied from
the 2048-point internal grid to the 1024-point storage grid and reconstructed
there.  The common order-six target is then recomputed from those two
1024-point fields.  The internal value of \(G(\eta;h)\xi\) is never resampled
or used as the stored target.

The
rollout settings are `pipeline.refinement.PAPER_TANAKA_GL2_CONTRACT` for
revision-3 Tanaka,
`pipeline.refinement.PAPER_BENJAMIN_FEIR_GL2_CONTRACT` for revision-4
Benjamin--Feir, and `pipeline.refinement.PAPER_JONSWAP_GL2_CONTRACT` for
revision-4 JONSWAP/TMA.  The latter two contracts explicitly separate the
sharp internal evolution from the delivered target.  Benjamin--Feir uses
\((N,K,M)=(1024,256,4)\), JONSWAP/TMA uses
\((N,K,M)=(2048,704,4)\), and both deliver the common
\((N,K,M)=(1024,128,6)\) target.  Revision-2 Stokes uses the static target
transaction and has no rollout.

The final JONSWAP/TMA cutoff was selected on the predeclared shallow sentinel
by independently adjusting and evolving adjacent sharp cutoffs on the
2048-point grid.  Comparing \(K=704\) with \(K=768\), the largest relative
whole-field discrepancies over the complete autonomous history were
\(4.64267\times10^{-4}\) for \(\eta\), \(1.93035\times10^{-4}\) for \(\xi\),
and \(7.91931\times10^{-4}\) for \(G(\eta;h)\xi\).  The corresponding
\(K=640\) versus \(K=768\) comparison exceeded the \(10^{-3}\) requirement.
Diagnostics restricted to modes \(96\leq|k|\leq128\) were about
\(3\times10^{-3}\), so the calculation certifies only the stated global
whole-field errors; it does not claim bandwise convergence.  Some stage
residuals approach the required \(10^{-8}\) tolerance, and this is one
worst-case sentinel rather than a population test.  The separate revision-4
27-category population gate described below was therefore required before
bulk JONSWAP/TMA generation and has passed.

As a current-source handoff check, the production executor replayed all 289
autonomous \(K=704\) saved frames from the archived full internal endpoint.
Against the archived canonical arrays, the maximum relative \(L^2\) errors
were \(3.99\times10^{-16}\), \(4.66\times10^{-16}\), and
\(1.17\times10^{-15}\) for \(\eta\), \(\xi\), and the recomputed stored target.
The replay was accepted, solved every stage, had maximum residual
\(9.994198\times10^{-9}\), Hamiltonian drift
\(1.831187\times10^{-5}\), and minimum water column \(0.0387699\).  Its CPU
wall time was 197.355 seconds.  This verifies the source-to-artifact handoff;
it is not a substitute for the 27-category GPU smoke.

The executable method-level pair/retry utility is
`pipeline.refinement.execute_residual_controlled_refinement`.  It is used for
the predeclared validation panel, not for routine dataset generation.  The
fixed- and variable-horizon production entry points are
`pipeline.refinement.execute_production_trajectory` and
`pipeline.refinement.execute_variable_horizon_production_trajectory`.
They take one residual-controlled GL2 rollout with actual step \(0.01\),
record complete-run telemetry, and expose fields through
`pipeline.trajectory_writer.outcomes_from_production` only when the complete
trajectory is admissible.  A rejected case retains its quality masks and GL2
telemetry but cannot expose a numerical prefix.

Paper-dataset JONSWAP/TMA generation is available only through
`scripts/run_paper_dataset_jonswap_bucketed.py`.  That launcher records the
adjustment both in the trajectory-execution record and in the complete
horizon-sorted numerical-batch policy, and binds both records into the run
fingerprint.  It uses `HorizonBucketedJonswapQuotaExecutor`.  The general
`scripts/run_paper_dataset_quota.py` path fails closed for JONSWAP/TMA, so a
paper run cannot silently omit the adjustment.  The combined-view preflight
requires the current execution record in the summary and every proposal, and
requires the complete current bucketing policy.  Benjamin--Feir continues to
use the general launcher.

Post-acceptance time selection is implemented in
`pipeline.time_selection`.  Tanaka uses relative change in surface-gradient
energy, with midpoint-quantile indices projected onto the endpoint-pinned
strictly increasing integer grid.  Benjamin--Feir uses 200 approximately
uniform saved-grid indices over
\(T_{\mathrm{BF}}=0.08\lfloor100T_c/0.08\rfloor\), where
\(T_c=2\pi/\sqrt{gk_c}\).  Random seas have peak period
\(T_p=2\pi/\omega_p\), intended horizon
\(H=16T_p\), and saved spacing \(\Delta t_s=0.08\).  They are integrated to
the last saved-grid time
\(T_s=\Delta t_s\lfloor H/\Delta t_s\rfloor\leq H\).  With
\(J=T_s/\Delta t_s\), the 16 retained indices are the nearest integers
\(j_\ell=\lfloor \ell J/15+1/2\rfloor\), \(0\leq\ell\leq15\).
These functions do not inspect model error.

## Artifact layout

Large field arrays remain immutable.  A dataset view is a JSON manifest that
points to those arrays and to a compact row-to-case map.  Passing a historical
raw NPZ to the loader preserves its historical row-split behavior; passing a
schema-v2 manifest uses the case split fixed in the proposal.

Each attempted batch is a transaction:

```text
proposal NPZ -> optional accepted-case shard NPZ -> result JSON
```

The proposal is written before numerical work.  The result JSON is the commit
marker.  A proposal without a result, or a proposal and shard without a result,
can be replayed exactly.  A result that names a missing or hash-mismatched
shard is corruption.  A rejected case owns zero rows; every accepted case owns
one complete, contiguous row block.

The schema-v2 row map contains

```text
trajectory_index, frame_index, shard_index, shard_row
```

and its attempted-case table contains

```text
trajectory_family_id, trajectory_revision_id, trajectory_split_id
trajectory_case_id, trajectory_cell_id
trajectory_accepted, trajectory_required_bits
trajectory_evaluated_bits, trajectory_failed_bits
trajectory_first_row, trajectory_row_count
```

The manifest records every proposal, result, and immutable shard path and
SHA-256 digest.  Each batch retains its run-configuration fingerprint; these
fingerprints normally differ across families and splits because they bind the
quota, random stream, and split assignment.  A separate dataset-contract
fingerprint binds the common DNO target and stored dtypes.  Numerical and
source contracts are compared within each `(family, revision)` pair, so a
revision-3 trajectory rollout need not equal its revision-2 predecessor.  The
persisted trajectory contract records the single production step \(0.01\);
audit-only half steps and discrepancy tolerances are not production
settings.  A combined view contains exactly one revision for each family
and rejects inconsistent numerical or source contracts for that
family-revision pair.  Rejected candidate specifications and reasons remain in
the proposal and result records even though they contribute no training rows.
The schema-v2 training sampler uses exactly one stored time from every
accepted case in an epoch and requires equal accepted-case counts across
physical families.

Across all four families, `proposal_law_applies_to = attempted_specifications`.
Within a preassigned category,
`released_case_law = proposal_conditioned_on_complete_case_acceptance_within_preassigned_cell`:
the loader-visible law is the proposal conditioned on the declared family
complete-case acceptance event within that category.  The event is frozen by
family and revision.  Accepted quotas fix the category marginals; they do not
restore the unconditioned within-category proposal density.  This is the
common quota/executor semantics, not a post-hoc parameter gate.

The family-independent quota, transaction, view, and loader checks are
reproduced by

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python -m unittest \
    solver.gen_data.pipeline.test_production \
    solver.gen_data.pipeline.test_archive \
    solver.gen_data.pipeline.test_manifest \
    solver.gen_data.pipeline.test_writer \
    solver.gen_data.pipeline.test_quota_driver \
    solver.gen_data.pipeline.test_refinement \
    solver.gen_data.pipeline.test_time_selection \
    solver.gen_data.pipeline.test_trajectory_writer

JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python train-jax-10m/tests_paper_dataset_view.py
```

These are tests of the common storage and sampling foundation.  The common
accepted-quota driver reconstructs exact per-category counts from committed
artifacts, replays proposal-only and proposal-plus-shard interruptions, and
uses a nonblocking single-writer lock.  A rejected attempt does not advance
its accepted-category count.  The source-authenticated release is complete at
16384 training, 1024 validation, and 1024 test cases in each of four families:
73728 accepted cases from 74045 attempts, 317 zero-row rejections, and
7686144 retained rows.  Stokes revision 2 retained 18432/18432 attempts;
corrected Tanaka revision 3 retained 18432/18432; Benjamin--Feir revision 4
retained 18432/18455; and JONSWAP/TMA revision 4 retained 18432/18726.  The
historical 4096-case Tanaka output remains excluded because its old
solitary-wave inversion did not realize 64 requested small-amplitude
components.  The final combined summary is
`outputs/paper_dataset_literature_aligned_v1/combined/c16384_v01024_t01024/paper_dataset_all_splits_c16384.summary.json`
(SHA-256
`32f764cf865c90892ee0329e48fe992cbc24df2707e05fd1af2ee9cd2b84307c`).
The completed Benjamin--Feir family is independently certified by
`outputs/paper_dataset_bf_revision4_jonswap_revision3_literature_aligned_v1/benjamin_feir_completion_audit.json`
(SHA-256
`c155b35276d0cfef6844f6b0db3e91da3ce15579ad8b0f0ca482f44cc912753c`).
That full-dataset audit binds the exact 66-category taxonomy and split quotas,
reconstructed attempted/accepted/rejected counts in every category, proposal
replay, transaction and array hashes, accepted and rejected row ownership, the
200 endpoint-pinned, approximately uniform selections from each saved grid,
stored depths and water columns, and the numerical acceptance conditions.  It
also binds the two recovered shared-source snapshots to all six chunk source
maps and rehashes the other 17 recorded generation sources against the current
repository.  It is stronger than the earlier one-case-per-category gate.

Two shared modules subsequently changed for JONSWAP-specific work and their
Benjamin--Feir run bytes were no longer present in Git history.  Their exact
recorded versions are preserved inertly under
`reproducibility/source_snapshots/benjamin_feir_revision4_e16773f/`, together
with a `SHA256SUMS` manifest.  This closes the missing-byte gap for
`pipeline/production.py` and `trajectory_family_adapters.py`; the immutable
chunk summaries and completion audit bind the other 17 recorded source files
to their current repository bytes, along with the dependencies, run
specifications, and dataset data.  The snapshot is not a current import path or
a self-contained executable checkout.

The completed revision-2 Stokes release is analogously source-closed by
`outputs/paper_dataset_cap4_revision2_20260728/stokes_completion_binding.json`
(SHA-256
`d2223fddeedebd541ecdb19752ca547bbd0325ef886c6d97de821b24935dd4f5`).
All six chunks have the same exact 14-path map and source fingerprint.  Eleven
paths still match current repository bytes; the exact historical
`run_paper_dataset_quota.py`, `pipeline/manifest.py`, and
`pipeline/production.py` bytes from commit
`60a28ffae394465c6ea295eb4ed6c075fbc756a4` are preserved inertly with
`SHA256SUMS` under
`reproducibility/source_snapshots/stokes_revision2_60a28ff/`.  The binding also
requires exact zero-rejection attempted/accepted counts in each of the four
Stokes categories.  Neither historical snapshot directory is an import path.

The three rollout-family adapters in `trajectory_family_adapters.py` enforce
the first three operations:

```text
sample complete specifications
-> persist the proposal
-> construct initial fields
```

A constructor accepts only a durable proposal token and rechecks the proposal
state, file hash, case identities, and stored specifications before numerical
construction.  The shared executor, time selector, and whole-case writer
enforce the remaining integration, selection, and commit operations.
`trajectory_quota_executor.py` connects those operations to accepted
per-category quotas and replay.  A corrected real-GL2 CPU smoke at
\((N,M,p,K)=(64,0,1,16)\) took one Tanaka, one Benjamin--Feir, and one
JONSWAP/TMA case through proposal, the then-current paired validation path,
selection, whole-case
commit, schema-v2 view, and the training loader.  All three cases were
accepted and contributed three finite rows each.  This historical run is
wiring evidence for those components, not a production timing measurement.
The Tanaka executor records only the finite-negative-radicand failure
described above as a zero-row `OUTSIDE_SUPPORT` case.  Its classifier verifies
the recorded identities \(c^2=|c|^2\) and \(R_{\min}/c^2\), and verifies that
the negative-entry count agrees with the sign of the finite minimum.  A
nonfinite, malformed, or otherwise unclassified construction failure leaves
the durable transaction pending for replay, while only an explicit
`DeclaredTrajectoryFatalError` writes a fatal sidecar.  These reduced
calculations establish software wiring only; they are not spatial,
full-horizon, or population validation.

Exact-contract generation is launched one family and split at a time.  Stokes,
Tanaka, and Benjamin--Feir use `scripts/run_paper_dataset_quota.py`;
JONSWAP/TMA uses the dedicated adjusted launcher named above.  The default
action is a read-only preflight: it prints the ordered parameter-category
quotas, exact numerical contract, expected retained-row count, source and
dependency hashes, output namespace, and any resumable state.  Numerical work
requires the explicit `--execute` flag.

The mixed Benjamin--Feir-r4/JONSWAP-r3 launcher is superseded and remains
available only through repository history.  Current JONSWAP/TMA generation
uses `scripts/launch_revision4_jonswap_bulk.sh`.

The exact revision-3 Tanaka category check was run as complementary
validation-tagged shards: six cases in cells 0--5 on stream 300 and five cases
in cells 6--10 on stream 301.  All 11 attempts were accepted, no attempt was
rejected, and the trajectories contributed 2200 rows.  These are diagnostic
pilot rows in dedicated output roots; they are excluded from the final dataset
view, training, validation statistics, and normalization despite their
validation split tags.  The pilot used the exact \(P_{128}\to K=256\)
Hou--Li contract described above.

The three trajectory-family category count is \(11+66+27=104\); including the
four static Stokes categories gives 108 across all four families.  The
trajectory checks are implemented separately for Tanaka, Benjamin--Feir, and
JONSWAP/TMA.  Passing Tanaka does not establish either of the other trajectory
constructors.  The Benjamin--Feir and JONSWAP/TMA checks remain separate.  The
four static Stokes categories have passed their exact revision-2 calculation,
with all four cases accepted and `failed_bits=0`; that output is stored at
`outputs/static_stokes_exact_target_pilot_revision2_20260727`.

The interrupted Benjamin--Feir run at
`outputs/paper_dataset_cap4_revision2_20260728/train/benjamin_feir/chunk_02048_02048`
is quarantined as historical revision-2 output.  Its eight committed batches
contain 2048 attempted cases, of which 1856 were accepted and 192 rejected,
for 371200 retained rows.  The next proposal contains 192 uncommitted cases;
it has no result, promoted shard, or manifest.  None of these artifacts may be
resumed into or counted toward the revision-4 dataset.

The direct revision-4 Benjamin--Feir source gate accepted all 66 first
proposals, one in every carrier--sideband category.  The direct revision-4
JONSWAP/TMA population gate fixed 20 accepted cases in each of its 27
categories and retained 540 trajectories from 548 attempts.  Its eight
zero-row rejections were incomplete numerical trajectories, giving an
observed rejection rate of \(8/548=1.46\%\), below the predeclared two-percent
bound.  That criterion was the already-completed construction-adoption gate;
the revision-4 bulk rejection rate is reported diagnostically and is not a
final release threshold.  These population gates complement rather than
replace the independent
\(K=704\) versus \(K=768\) whole-field sentinel.  The earlier revision-3
JONSWAP/TMA stream-933 category calculation remains historical evidence for
the inherited adjustment and evolution method; it does not validate the
revision-4 relative-frequency population.

The superseded deterministic CPU preflight exercised persisted proposals and
retained-time writing, checked the Benjamin--Feir Fourier amplitudes and
phases, evaluated parameter-support boundaries, and verified that one failed
JONSWAP realization did not discard valid batch companions.  Its record is
preserved in `EXPERIMENTS.md` and repository history.  The full GPU gates
above, rather than that CPU preflight or unit tests, supplied the
population-level launch evidence.

A historical revision-3 Tanaka run completed 2048 training, 1024 validation,
and 1024 test cases with 200 rows per case.  Those rows are excluded from the
replacement dataset.  The old solitary-wave inversion had an effective crest
amplitude floor of \(9.9478\times10^{-4}\); 64 cases requested one component
below that floor, and the constructed initial state therefore did not match
its recorded parameters.  The corrected inversion widens the root bracket,
uses 48 bisections, and verifies every realized crest height.  Because this
changes the initial condition and nonlinear trajectory, the old rows cannot
be repaired or mixed with corrected rows.  The final revision-3 Tanaka dataset
was consequently generated afresh from new deterministic streams.  It
retained all 18432 attempts, stored 3686400 rows, and shares no historical
Tanaka shard with the release.

CPU is the fail-safe default.  `--platform gpu` selects the accelerator path
before JAX initializes.  Platform, quota, batch size, stream coordinates,
attempt ceiling, paper contract, family-specific source hashes,
Python/JAX/JAXLIB/NumPy versions, and the `pyproject.toml` and `uv.lock`
hashes are all bound by the configuration fingerprint.  Consequently a
resume must use the same command-defining values and dependency environment.
Use a different output root for a pilot and a later larger quota; an accepted
quota cannot be enlarged in place.  The general launcher accepts
`--family stokes`, `--family tanaka`, and `--family benjamin_feir`, but refuses
paper-dataset JONSWAP/TMA.  Stokes uses the static one-row transaction; the
other three families use their declared whole-trajectory integration and
time-selection contracts.

Learning-curve corpora are additive chunks, not enlargements of an existing
run.  Let \(B_i(C)\) be the balanced quota of category \(i\) after \(C\) accepted
cases.  A chunk beginning at cumulative count \(C_0\) and containing \(A\)
new accepted cases receives

\[
  B_i(C_0+A)-B_i(C_0)
\]

cases from category \(i\).  The launcher implements this rule through
`--accepted-cases-before C0`; its default is zero for a standalone pilot.
This subtraction matters for the 66-category Benjamin--Feir and 27-category
JONSWAP/TMA families: balancing every chunk independently would repeatedly
favor the first remainder categories.  The frozen nested learning curve is:

| Chunk | `--accepted-cases-before` | `--accepted-cases` | Cumulative count | Suggested `--stream-id` |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 2048 | 2048 | 0 |
| 2 | 2048 | 2048 | 4096 | 1 |
| 3 | 4096 | 4096 | 8192 | 2 |
| 4 | 8192 | 8192 | 16384 | 3 |

Stokes, corrected Tanaka, and Benjamin--Feir use these four execution shards
directly.  To balance the two remaining GPU lanes, the last JONSWAP/TMA
increment is physically divided into `[8192,12288)`, `[12288,14336)`, and
`[14336,16384)` with streams 3, 4, and 5.  The additive quota differences
preserve exactly the same final per-category allocation; an execution-shard
boundary is not a new population category or learning-curve checkpoint.

Every chunk must use a distinct output root and stream ID.  The output root
is the transaction namespace; the stream ID is part of every case identity
and random seed.  `--first-attempt-index` normally remains zero within each
new stream.  The preflight prints the cumulative before, chunk, and
cumulative after quota for every category so this assignment can be checked
without constructing a field.

Batch size changes transaction and memory granularity, not the population
law.  A larger batch can improve accelerator utilization, but increases peak
memory and the amount of work replayed after an interruption.  The fresh
Tanaka plans use 256-case transactions.  JONSWAP/TMA uses 32-case transactions
whose numerical solves are sorted by horizon and executed in groups of eight.
Both choices are part of their run fingerprints and do not set another
family's batch size.

The accepted-quota loop is also bounded.  If a category requires \(Q_i\) accepted
cases, the default `--maximum-attempts-per-accepted-case 4` permits at most
\(4Q_i\) durable case proposals in that category.  A pending proposal already
owns one of those attempt slots and is replayed rather than counted twice.  If
its result leaves the category short at the ceiling, the run stops with the
accepted, attempted, target, and ceiling counts instead of drawing forever.
This outer proposal count is distinct from the JONSWAP/TMA cap of five
fixed-point updates per GL2 stage.  It does not include GL2 substeps or the
finite-Stokes sampler's internal amplitude redraws.  The multiplier is
immutable within an output root.

The long-running replacement queue is supervised by
`scripts/launch_tanaka_after_revision4_jonswap.sh`.  It exact-resumes a
missing JONSWAP or Tanaka generator only within a fixed retry budget.  After
each family closes its lightweight quota gate, it runs the corresponding
CPU-only source-bound completion audit,
`scripts/audit_completed_jonswap_revision4.py` or
`scripts/audit_completed_tanaka_revision3.py`.  The final builder is not
called unless both audit artifacts report the exact split counts, row counts,
family revision, source identity, transaction integrity, row ownership, and
numerical-health postconditions.

Before a missing Tanaka run is launched, the supervisor sources
`scripts/check_tanaka_gpu_admission.sh` and waits until physical GPUs 0 and 1
(or the two explicitly configured physical indices) have no compute processes
and each reports at least 40,960 MiB free.  That floor leaves 5,935 MiB above
the measured 35,025-MiB batch-256 peak.  The check only reads `nvidia-smi` and
never signals a foreign process.  Invalid or unavailable telemetry fails
closed; ordinary occupancy or low free memory is logged once per minute and
retried.  The direct Tanaka launcher checks once before preflights and once
again immediately before forking its two lanes, closing the claim-during-
preflight race without changing any numerical, source, dependency, or run
configuration fingerprint.  An already valid Tanaka completion gate bypasses
the GPU wait because only the CPU completion audit remains.

`scripts/watch_paper_dataset_workers.py` supplies the separate silent-hang
guard.  It does not import or modify generator code.  Instead, it matches only
the declared user, checkout, complete worker command, GPU, PID start time, and
chunk root, and watches each lane's immutable proposal/result/shard tokens.
The production interval is one minute and the no-progress threshold is six
hours for both remaining families.  That threshold was selected only after
including a valid 4.218-hour revision-4 JONSWAP transaction; it is not a
solver time step or a trajectory acceptance rule.  A stale exact worker gets
TERM, then KILL after 120 seconds if its identity is still unchanged, after
which the bounded exact-resume supervisor handles its pending transaction.
The later CPU audits and final view build have independent one-hour and
30-minute limits, respectively.

After every train, validation, and test run is complete and those family
audits pass, build the four loader-facing learning-curve views from their
exact completion summaries:

```bash
bash scripts/build_literature_aligned_paper_dataset_views.sh
```

The guarded script first performs the same read-only preflight used by the
individual builder, then writes metadata-only views at 2048, 4096, 8192, and
16384 accepted training cases per family with the full fixed validation and
test splits.  Within every included split it requires exactly the four
physical families, gap-free cumulative intervals beginning at zero, distinct
stream IDs and roots within each family, and equal final accepted-case counts
across families.  It permits one revision per family, requires the same
family revision across that family's chunks and splits, and compares source
and numerical contracts within the resulting family-revision group.  It also
requires one dependency environment across the complete view.  It passes the
exact set of run fingerprints to `build_dataset_view` and validates every
manifest split count.  If \(C_{\rm tr},C_{\rm va},C_{\rm te}\) are the accepted counts per
family, the result contains
\(4(C_{\rm tr}+C_{\rm va}+C_{\rm te})\) accepted cases and
\(417(C_{\rm tr}+C_{\rm va}+C_{\rm te})\) rows.  Validation and test normally
use one fixed chunk per family rather than the four-checkpoint training schedule.
A single-split view remains available for diagnostics, but training should
consume the combined all-split manifest.

The separate CPU-only postcompletion path is
`scripts/run_paper_dataset_postcompletion.py`, launched by
`scripts/launch_paper_dataset_postcompletion.sh`.  It first authenticates the
Stokes binding and the exact Benjamin--Feir, JONSWAP/TMA, and Tanaka completion
audit records against the ordered 26 source summaries in the final combined
view.  It independently rehashes both the four-artifact Stokes and
three-artifact Benjamin--Feir historical source sets, retaining family-scoped
identities so their `SHA256SUMS` and `production.py` names cannot collide.
Only then does it run the non-gating worst-case renderer, JONSWAP order
diagnostic, training-normalization and loader-handoff audit, and deterministic
family-example figure.  These diagnostics cannot change trajectory
acceptance or retroactively change the release status of a completed view.

The release-specific integration surface is listed in
`reproducibility/paper_dataset_release_files.json`.  Its focused test checks
that every named implementation and test file exists, that Python and JSON
parse, and that the five shell entrypoints retain mode `0755`.  Numerical
transitive dependencies are not duplicated in this inventory because each
family run specification already records and verifies that curated generation
source-identity map.

Static Stokes is connected end to end by `stokes_static_pipeline.py`: it
writes the proposal before construction, evaluates the common target, stores
exactly one \(t=0\) row for an accepted case, and gives a rejected attempt
zero rows.  `stokes_quota_executor.py` connects that transaction to the
accepted-quota driver.  If the finite-depth sampler exhausts its declared
same-category amplitude redraws, the executor stores the complete draw ledger as
a zero-row `OUTSIDE_SUPPORT` attempt, keeps valid siblings in the batch, and
schedules a replacement only in the missing category.  Its production default is
the frozen paper target.  A reduced target must be labeled
`reduced_wiring_evidence_only`.  The historical exact-target pilot ran the
accepted-quota path with one validation case in each of the four Stokes
categories at \((N,M,p,K)=(1024,6,8,128)\).  The 2026-07-25 run accepted all
four cases and wrote a proposal, shard, result, manifest, and trajectory map
with recorded SHA-256 hashes; the training loader returned four finite one-row
cases.  That generator-revision-1 entrypoint remains available through
repository history.  The current generator-revision-2 run on 2026-07-27 again
accepted all four cases, with `failed_bits=0`; it is stored at
`outputs/static_stokes_exact_target_pilot_revision2_20260727`.

The four declared population laws and both quota executors are checked by:

```bash
uv run python -m unittest \
  solver.gen_data.tests_stokes_sampling \
  solver.gen_data.tests_stokes_static_pipeline \
  solver.gen_data.tests_stokes_quota_executor \
  solver.gen_data.tests_tanaka_sampling \
  solver.gen_data.tests_tanaka_potential_radicand \
  solver.gen_data.tests_benjamin_feir_sampling \
  solver.gen_data.tests_jonswap_tma_sampling \
  solver.gen_data.tests_trajectory_family_adapters \
  solver.gen_data.tests_trajectory_quota_executor \
  scripts.test_run_paper_dataset_quota \
  scripts.test_build_paper_dataset_view \
  scripts.test_run_trajectory_quota_real_gl2_smoke
```

## Legacy v9

`combined_dataset_v9.npz` is frozen as a historical artifact.  Its first ten
sources were row-filtered and globally shuffled before four later source
additions were appended.  The flat archive discarded original case IDs, so it
must not itself be described as uniformly trajectory-filtered.

The trajectory-filtered v9 view reconstructs compound trajectory identity from the raw
source archives and the recorded shuffle/append lineage.  It masks complete
trajectories whenever a source row was historically rejected, and it splits
the retained trajectories by physical family.  Checks unavailable from the
saved artifacts remain explicitly unevaluated.  A future generator revision
must use this contract directly rather than reconstruct it after assembly.
