# Paper-dataset confidence tests

Date: 2026-07-24; updated 2026-07-25

> **Historical note — superseded production contract.** This note records
> revision-1 validation measurements and is retained only as historical
> evidence. The current generator uses revision 2. In particular, Tanaka has
> eleven parameter categories indexed by structural regime, crest count
> \(m\), and right-moving crest count
> \(q=\#\{j:s_j=+1\}\); direction is therefore part of the category. No
> revision-1 artifact can be resumed into the revision-2 gate or final
> dataset. Use the
> [current readiness note](paper_dataset_generation_readiness_20260726.md)
> and the
> [condensed construction contract](paper_dataset_generation_condensed_20260726.tex)
> for production decisions. The measurements below have not been rewritten.

## Purpose

These tests ask whether the proposed initial-condition constructors and common
reference solver behave consistently. They do not introduce additional
per-row rejection rules.

In this note, a **constructor** maps a stored case specification to an initial
state; a **validation panel** is a small, fixed set of cases used to test the
numerical method; a **writer** stores proposals, field shards, and decisions;
and the **dataset** is the collection of accepted cases presented to the
training loader. The calculation in Section 7 is a validation panel, not
dataset generation.

The production acceptance decision remains:

1. sample an initial condition from a family defined before generation;
2. for Tanaka, Benjamin--Feir, and JONSWAP/TMA, retain the finer member of a
   consecutive time-refinement pair only when the dimensionless delivered
   fields agree to \(10^{-3}\).

A static Stokes training case has no time trajectory. It is accepted by its
declared parameter support and static finite-state, positive-water-column, and
finite-target checks, and it retains only \(t=0\).

Finiteness, positive water depth, and successful implicit solves determine
whether the numerical trajectory exists. Hamiltonian drift, spectra, slopes,
and the figures below remain diagnostics.

## 1. Finite-depth Stokes construction

The collaborator's MATLAB code and the JAX constructor implement the same
fifth-order perturbation formula. The issue found in the shallow cases is not
a disagreement between those implementations. It is the use of the
fifth-order formula outside its stated parameter regime.

This was checked against data produced independently by the MATLAB generator,
not only by comparing source expressions. For seven archived parent waves
(three deep and four finite), the JAX constructor was compared at
\(t=0,6.8,20\). Across the 21 snapshots, the relative discrepancies were

\[
\begin{aligned}
  1.7\times10^{-16}
  &\leq
  \frac{\|\eta_{\mathrm{JAX}}-\eta_{\mathrm{MATLAB}}\|_2}
       {\|\eta_{\mathrm{MATLAB}}\|_2}
  \leq1.1\times10^{-14},\\
  1.3\times10^{-16}
  &\leq
  \frac{\|\xi_{\mathrm{JAX}}-\xi_{\mathrm{MATLAB}}\|_2}
       {\|\xi_{\mathrm{MATLAB}}\|_2}
  \leq1.0\times10^{-14},\\
  4.7\times10^{-15}
  &\leq
  \frac{\|[G(\eta;h)\xi]_{\mathrm{JAX}}
             -[G(\eta;h)\xi]_{\mathrm{MATLAB}}\|_2}
       {\|[G(\eta;h)\xi]_{\mathrm{MATLAB}}\|_2}
  \leq9.0\times10^{-13}.
\end{aligned}
\]

The comparison includes a sharp finite-depth MATLAB state whose correction
harmonics already exceed its fundamental. Thus neither that shape nor the
Ursell diagnosis is created by the JAX port.

For a constructed surface \(\eta_0\), define its crest-to-trough height

\[
  H=\max_x\eta_0(x)-\min_x\eta_0(x).
\]

If \(k\) is the carrier wavenumber, its wavelength is
\(\lambda=2\pi/k\). The physical Ursell number is

\[
  \mathrm{Ur}=\frac{H\lambda^2}{h^3}.
\]

Let \(\theta\in\mathbb R/(2\pi\mathbb Z)\) denote the carrier phase, and let
\(E_m\) denote the coefficient of the \(m\)-th elevation harmonic. Write the
implemented degree-five series as
\(\eta_0(\theta)=\sum_{m=1}^5E_m\cos(m\theta)\). Let \(H_{4096}\) be its
sampled height on 4096 equally spaced values of \(\theta\), and define

\[
  H_+=H_{4096}
  +\frac14\left(\frac{2\pi}{4096}\right)^2
   \sum_{m=1}^{5}m^2|E_m|.
\]

Because \(|\eta_0''|\leq\sum m^2|E_m|\), Taylor's theorem gives
\(H\leq H_+\). The proposed finite-depth Stokes family requires the
conservative sufficient condition

\[
  \boxed{\mathrm{Ur}_+=\frac{H_+\lambda^2}{h^3}\leq26.}
\]

This replaces the coefficient-order rule previously proposed in this report.
Fenton identifies \(ka/(kh)^3\) as the effective shallow-water expansion
parameter for Stokes theory
([Fenton, 1985](https://doi.org/10.1061/(ASCE)0733-950X(1985)111:2(216))).
Zhao, Wang, and Liu use \(\mathrm{Ur}=26\) to separate the regime in which a
fifth-order Stokes description is recommended from the shallower cnoidal
regime
([Zhao, Wang, and Liu, 2024](https://doi.org/10.1016/j.coastaleng.2023.104432)).

Let \(a\) denote the leading-order amplitude supplied to the fifth-order
constructor. Since \(H=2a[1+O((ka)^2)]\),

\[
  \mathrm{Ur}
  =
  8\pi^2\frac{ka}{(kh)^3}[1+O((ka)^2)].
\]

Thus the leading-amplitude approximation to the physical
\(\mathrm{Ur}\leq26\) condition is

\[
  ka\lesssim\frac{26}{8\pi^2}(kh)^3
  =0.32929\,(kh)^3.
\]

The implemented family decision uses the conservative height \(H_+\), not
this leading approximation. It is evaluated after constructing the analytic
profile but before time integration, and it is independent of phase.

### Retrospective diagnosis

The rule was evaluated on all 256 independently seeded finite-depth Stokes
initial conditions in the completed nonlinear rollout panel. That panel
contains four nonfinite reference trajectories and three finite trajectories
whose stored relative Hamiltonian drift exceeds \(10^{-3}\).

| quantity used for the support check | outside support | unhealthy caught | numerically healthy but outside support |
| --- | ---: | ---: | ---: |
| conservative \(H_+\) from the analytic profile | 25/256 | 7/7 | 18/249 |
| leading proxy \(8\pi^2ka/(kh)^3\) | 23/256 | 7/7 | 16/249 |

Every unhealthy case is far outside the published boundary:

\[
  80.50\leq\mathrm{Ur}_+\leq161.83.
\]

Historical case 98 has \(\mathrm{Ur}_+=96.88\); the leading-amplitude proxy
is \(61.87\).
Both diagnose the case as an unsupported use of fifth-order Stokes theory.
Across the panel, the Spearman correlation between
\(\log\mathrm{Ur}_+\) and the logarithm of maximum stored truth-energy drift is
\(0.376\), with \(p=1.5\times10^{-9}\).

The boundary is not an exact numerical-failure classifier. Case 99000002 is
healthy at \(\mathrm{Ur}_+=25.56\), and case 99000166 is also healthy at
\(\mathrm{Ur}_+=27.07\). More generally, 18 trajectories outside the
declared support happened to remain numerically healthy. This is not evidence that
their fifth-order initial states are accurate traveling waves: a general
initial state may evolve finitely even when its intended traveling-wave
interpretation is poor. The Ursell condition defines the named family;
consecutive time refinement separately tests the numerical evolution.

Rejected amplitudes are resampled inside the same preassigned stratum and
their values are retained in the case record. Clipping them to
\(\mathrm{Ur}_+=26\) would create an artificial point mass at the support
boundary.

### Harmonic ordering remains a diagnostic

Write the fifth-order elevation as

\[
  \eta(x)=\sum_{m=1}^{5}E_m\cos(m\theta).
\]

Here \(\theta=kx+\phi\) is the carrier phase and \(E_m\) is the coefficient
of the \(m\)-th elevation harmonic.

The earlier audit used

\[
  R=\frac{\sum_{m=2}^{5}|E_m|}{|E_1|}.
\]

No cited Stokes-wave analysis recommends \(R\leq1\) as an applicability
boundary, so it is no longer a sampling condition. It remains useful for
showing what went wrong geometrically: case 98 has \(R=1.182981\), so its
nominal correction harmonics collectively exceed its fundamental.

The deterministic implementation tests still establish that:

- scalar and batched parameter evaluations produce the same five
  coefficients;
- the normalized coefficients depend on \(kh\) and \(ka\), rather than on
  the dimensional representation of the same parameters;
- integer-grid translation changes the sampled phase exactly, and at zero
  phase the elevation is even while the surface potential is odd.

The batched test found and corrected a broadcasting error in the new harmonic
helper: the scalar calculation worked, but a vector of amplitudes could not
be multiplied by its \(B\times5\) coefficient array.

For historical context, \(R\leq1\) excluded \(2.23\%\) of the old
amplitude-uniform finite-Stokes archive, whereas the leading Ursell
approximation excludes approximately \(9\%\). The latter is deliberately more
conservative because it represents a published theory-applicability boundary
rather than an empirical separator fitted to the seven observed failures.

## 2. Constructor and translation tests

The tangent-Hermite Tanaka tests cover:

- exact Hermite values and tangents;
- strictly increasing profile knots;
- nonnegative, monotone single-crest profiles;
- all periodic images that intersect the domain;
- translation by one period and by one grid cell;
- reversal of propagation direction;
- the historical case-31 high-frequency tail;
- one production GL2 output interval.

All six tests pass. In the full twelve-case static panel, the previously
recorded half-period translation errors are

\[
  2.0\times10^{-16}\quad\hbox{in }\eta,
  \qquad
  1.7\times10^{-16}\quad\hbox{in }\xi.
\]

A separate four-code-path test constructs linear, finite-Stokes,
tangent-Hermite Tanaka, and historical Benjamin--Feir states. It then translates
each state by seven grid points and recomputes the order-six, pad-eight DNO.
The translated value of \(G(\eta;h)\xi\) agrees with the translated original
target to the test tolerance. This checks the periodic indexing and label
construction; it is not a morphology filter.

## 3. Consecutive time refinement

For one state from each of the four code paths above, the same \(N=256\)
batch was evolved from \(t=0\) to \(t=0.08\) using:

- coarse step \(\Delta t=0.01\);
- fine step \(\Delta t=0.005\);
- order-six DNO with pad factor eight;
- eight GL2 fixed-point sweeps;
- delivered band \(|k|\leq32\).

The relative errors are:

| Constructor | \(\eta\) | \(\xi\) | \(G(\eta;h)\xi\) | maximum |
| --- | ---: | ---: | ---: | ---: |
| linear | \(1.04\times10^{-12}\) | \(6.43\times10^{-13}\) | \(1.71\times10^{-12}\) | \(1.71\times10^{-12}\) |
| finite Stokes | \(1.48\times10^{-12}\) | \(7.99\times10^{-13}\) | \(2.54\times10^{-12}\) | \(2.54\times10^{-12}\) |
| tangent-Hermite Tanaka | \(1.90\times10^{-12}\) | \(2.89\times10^{-13}\) | \(8.51\times10^{-12}\) | \(8.51\times10^{-12}\) |
| historical Benjamin--Feir | \(4.14\times10^{-11}\) | \(2.69\times10^{-11}\) | \(5.52\times10^{-11}\) | \(5.52\times10^{-11}\) |

All four pass \(10^{-3}\). The intentionally coarse negative control has
maximum error \(3.31\times10^{-3}\) and is rejected.

This is a short-horizon integration smoke. It does not calibrate the
\(10^{-3}\) tolerance for every full production trajectory.

## 4. Independent spatial refinement

Temporal agreement cannot detect a spatial error shared by both trajectories.
A separate generator-revision test therefore constructs the same ordered
finite-Stokes state independently at \(N=32\) and \(N=64\).

For a real periodic field \(u\), define

\[
  c_m^{(N)}=\frac{1}{N}\operatorname{rfft}(u_N)_m,
  \qquad 0\leq m\leq12.
\]

The fixed-band norm is

\[
  \|c\|_K^2=|c_0|^2+2\sum_{m=1}^{K}|c_m|^2.
\]

The \(N\)-versus-\(2N\) relative errors are

\[
\begin{aligned}
  \Delta_\eta &=2.98\times10^{-16},\\
  \Delta_\xi &=4.46\times10^{-11},\\
  \Delta_{G(\eta;h)\xi}&=1.34\times10^{-8}.
\end{aligned}
\]

The test requires only that their maximum remain below \(10^{-6}\). It
qualifies a common generator revision and is not applied as a per-row gate.

## 5. Visual stress cases

The [shape-stress figure](../outputs/paper_dataset_shape_stress_20260724/paper_dataset_shape_stress.png)
shows:

1. finite-Stokes case 99000224, with \(R=1.630\) and
   \(\mathrm{Ur}_+=161.83\);
2. finite-Stokes case 99000221, with \(R=0.9617\) but
   \(\mathrm{Ur}=70.70\);
3. Tanaka case 31 before and after tangent-Hermite placement;
4. the smooth single-mode Airy case 95000030, whose minimum water column is
   \(0.458h\) and which demonstrates why the old \(h/2\) margin was not a
   valid shape test.

The second Stokes profile is important retrospectively: it passed the old
harmonic-order rule but lies outside the adopted Ursell support. Thus the
figure should not be read as an accepted-versus-rejected comparison under
the final rule.

The [finite-Stokes boundary figure](../outputs/paper_dataset_shape_stress_20260724/finite_stokes_boundary_panel.png)
compares profiles with \(R=1\) and \(R=0.75\) at four shallow values of
\(kh\). It is retained as evidence that harmonic ordering alone did not
control traveling-wave accuracy; it is not the adopted support boundary.

For an analytic traveling profile define the kinematic defect

\[
  D_{\mathrm{kin}}
  =
  \frac{\|\eta_t-G_6(\eta;h)\xi\|_{L^2}}
       {\|\eta_t\|_{L^2}},
\]

where \(\eta_t\) is the exact time derivative of the implemented fifth-order
formula. At the \(R=1\) boundary,

\[
  0.165\leq D_{\mathrm{kin}}\leq0.189
  \qquad (0.5\leq kh\leq0.625).
\]

At \(R=0.75\),

\[
  0.068\leq D_{\mathrm{kin}}\leq0.081.
\]

Thus harmonic ordering alone does not guarantee an accurate traveling-wave
approximation. This failure motivated replacing \(R\) by the literature-based
Ursell support above.

Exact figure values are in
`outputs/paper_dataset_shape_stress_20260724/shape_stress_metrics.json`.

## 6. Implemented constructors and numerical definitions

The following definitions now replace the provisional items in the original
version of this note.

### 6.1 Stokes support and sampling

Let \(L\) be the spatial period, let \(x\in[0,L)\) be position, let \(h>0\)
be the depth, let \(n\) be an integer carrier mode, and set
\(k=2\pi n/L\) and \(\lambda=2\pi/k\). The parameter \(a\) is the
leading-order amplitude, and \(\theta=kx+\phi\) is the carrier phase, where the
translation \(\phi\) is sampled directly from
\(\operatorname{Unif}[0,2\pi)\). The surface potential \(\xi\) is evaluated
at this prescribed phase and its spatial mean is removed.
The finite-depth branch uses the Ursell condition below. The deep branch
instead requires \(kh\geq5\), so its infinite-depth formula is not assigned
to the historical \(n=1,\ 4\leq h<5\) corner.
For 40 random finite- and deep-water parameter sets, prescribing a phase
directly gives the same elevation as choosing the corresponding phase in the
historical time-based constructor; after removing the constant-potential
gauge, the largest pointwise surface-potential discrepancy is
\(4.34\times10^{-19}\).

For the complete implemented elevation \(\eta\), define its physical height
by \(H=\max_x\eta(x)-\min_x\eta(x)\). The public sampler computes the upper
bound \(H_+\) below and enforces \(\mathrm{Ur}_+\leq26\).
Let \(E_m\) denote the coefficient
of the \(m\)-th elevation harmonic. If \(H_{4096}\) is the sampled height on
4096 equally spaced phases, it uses

\[
 H_+=H_{4096}
 +\frac14\left(\frac{2\pi}{4096}\right)^2
  \sum_{m=1}^{5}m^2|E_m|.
\]

Taylor's theorem gives \(H\leq H_+\). The implemented support condition is

\[
  \mathrm{Ur}_+=\frac{H_+\lambda^2}{h^3}\leq 26.
\]

Before drawing \(a\), the sampler intersects the requested dimensional
amplitude interval with the two steepness cells

\[
  0.005\leq ka<0.03,
  \qquad
  0.03\leq ka\leq0.15.
\]

Crossing the finite/deep branch with these two intervals gives four allocation
cells. The common scheduler assigns a cell before sampling. Conditional on
that cell, the current sampler chooses the carrier mode uniformly from the
integer modes with nonempty depth and amplitude support, chooses depth
log-uniformly on its mode-conditioned interval, chooses phase uniformly on
\([0,2\pi)\), and chooses \(a\) uniformly from the intersection of the
global amplitude interval and its assigned steepness cell. If a finite-depth
amplitude violates the Ursell inequality, only \(a\) is redrawn; the carrier,
depth, phase, split, and cell remain fixed. Every rejected amplitude and its
\(\mathrm{Ur}_+\) value, followed by the accepted amplitude, is stored in the
case record.

The per-case PCG64 stream uses five stored integers: split root, physical
family, generator revision, stream identifier, and attempted-case index.
Changing any one changes the random stream, while replaying all five
reproduces the specification exactly. This is seeded stratified
pseudorandom sampling, not a low-discrepancy construction.

All four cells passed 1024 independent support draws. A separate
finite-moderate stress calculation accepted all 16,384 attempts; 1935 needed
at least one amplitude redraw, the largest redraw count was 57, and no case
exhausted the cap of 1000. A forced-exhaustion test verifies that the sampler
raises with a strict record containing the fixed parameters and every
amplitude--\(\mathrm{Ur}_+\) pair. The remaining outer-driver requirement is
to store it as a zero-row out-of-support attempt and schedule a same-cell
replacement. It is not a fatal batch error. The
[replacement Ursell figure](../outputs/paper_dataset_shape_stress_20260724/finite_stokes_ursell_panel.png)
shows the published boundary together with the earlier truth diagnostics.
The boundary defines the family support; it is not claimed to classify every
possible numerical failure.

The common static transaction was then run at the exact paper target
\((N,M,p,K)=(1024,6,8,128)\) in float64. One validation attempt was assigned
to each of the four finite/deep and low/moderate cells, and every proposal was
stored before construction. All four cases were accepted: the required and
evaluated masks were both 263, the failed mask was zero, the state and target
were finite, the water column was positive, and the fixed-band support check
had zero violations. The complete proposal, target evaluation, commit, and
dataset-view construction took 2.584 seconds on CPU. The machine-readable
record is
[`summary.json`](../outputs/static_stokes_exact_target_pilot_20260725/summary.json).

### 6.2 Periodic Tanaka population

The four allocation cells are the main family with one, two, or three crests
and the steep one-crest family. In a main cell, depth \(h\) is log-uniform on
\([0.01,0.30]\), the total amplitude ratio
\(\sum_{i=1}^m\alpha_i\) is uniform on \([0.10,0.35]\), and a
\(\operatorname{Dirichlet}(1,\ldots,1)\) draw divides that total among the
\(m\) crests. For \(m>1\), a second uniform-simplex vector
\((w_1,\ldots,w_m)\) defines the cyclic gaps

\[
  g_i=3h+(2\pi-3mh)w_i,\qquad i=1,\ldots,m.
\]

One uniform global rotation places the resulting centers on the periodic
domain, and the propagation directions are independent fair signs. The steep
cell has \(m=1\), log-uniform \(h\in[0.20,0.35]\), and uniform
\(\alpha_1\in[0.25,0.45]\).

A stress calculation drew 4096 cases in every cell. All 16,384 samples obeyed
the depth, amplitude, direction, and cyclic-separation conditions. The
ordered adapter stores the complete specification before tangent-Hermite
construction and reproduces the constructed fields exactly. It has not yet
run a real-GL2 transaction because the constructor still clips a negative
surface-potential radicand. The required fix is to reject such a case
explicitly; the fixed upper-seam validation case has a positive scaled margin
of \(0.372151\), so this change does not alter that case.

### 6.3 Deep-water Benjamin--Feir family

The frozen family is the equation-(33) form of Xu and Guyenne with a
fifth-order analytic deep-water carrier, equal-amplitude sidebands, one
common sideband phase, and the sideband-specific factors
\(\sqrt{g/k_\pm}\exp(k_\pm\eta)\). It contains no empirical cross terms and no
special rewrite of a mode pair. Let \(x\in[0,L)\) be the periodic spatial
coordinate, where \(L\) is the period, and let \(g\) be gravitational
acceleration. Let \(n_c\) be the integer carrier mode, let
\(\Delta n\) be the positive integer separation between the carrier and each
sideband, let \(\epsilon_c\) be the prescribed carrier steepness, and let
\(\rho\) be the ratio of each sideband amplitude to the leading carrier
amplitude. Define

\[
k_c=\frac{2\pi n_c}{L},\qquad
k_\pm=\frac{2\pi(n_c\pm\Delta n)}{L},\qquad
a=\frac{\epsilon_c}{k_c}.
\]

Thus \(a\) is the leading carrier amplitude. If
\((\eta_c,\xi_c)\) is the analytic fifth-order carrier and
\(\phi\in\mathbb R/(2\pi\mathbb Z)\) is the common sideband phase, the
initial state is

\[
\eta=\eta_c+\rho a\{
\cos(k_-x+\phi)+\cos(k_+x+\phi)\},
\]

\[
\xi=\xi_c+\rho a\left\{
\sqrt{\frac{g}{k_-}}e^{k_-\eta}\sin(k_-x+\phi)
+
\sqrt{\frac{g}{k_+}}e^{k_+\eta}\sin(k_+x+\phi)
\right\},
\]

followed by removal of the spatial mean of \(\xi\). The parameter support is

\[
\begin{gathered}
4\leq n_c\leq20,\qquad
0.05\leq\epsilon_c\leq0.13,\qquad
0.05\leq\rho\leq0.20,\\
1\leq\Delta n<n_c,\qquad
0<
\frac{\Delta n/n_c}{2\sqrt2\,\epsilon_c}
<1.
\end{gathered}
\]

The numerical depth is \(h=5L/(2\pi)\). Six focused formula and support tests
pass. Static formula comparisons are at machine precision and a four-case
generator smoke is finite. The carrier is explicitly the project's analytic
fifth-order carrier, not the numerically exact steady carrier used in JCP09.
A finite-depth Benjamin--Feir family would require a separate definition.

The population sampler treats each feasible integer pair
\((n_c,\Delta n)\) as one of 66 parameter cells. In a 20,000-case schedule,
the quotient--remainder rule assigns 303 or 304 attempts to every pair.
Conditional on the pair, the sampler draws \(\epsilon_c\) uniformly over its
admissible interval, \(\rho\) uniformly on \([0.05,0.20]\), and \(\phi\)
uniformly on \([0,2\pi)\). A stress calculation drew 128 cases from every
pair, for 8448 cases in total; all satisfied the stated support. The ordered
adapter stores the pair and continuous parameters before construction. A
reduced real-GL2 case then passed through the common refinement, temporal
selection, whole-case archive, and split-aware dataset view. The reduced
calculation checks software wiring, not full-horizon accuracy.

### 6.4 Resolved-band JONSWAP/TMA family

The public random-sea constructor uses cell-integrated JONSWAP frequency
density, the standard TMA depth factor, and the group-velocity Jacobian. Its
returned state contains both phase arrays and assigns the right-going
linear-energy fraction exactly. The 27-case pilot field NPZ does not store
those arrays; its JSON record stores a case seed from which the current
phases replay. The replacement sampler instead stores both arrays directly in
the durable proposal. The constructor uses a cosine-squared density window
that is one through
\(K_0=96\) and zero at \(K=128\). No realization-dependent slope, height,
water-margin, or appearance test is used.

The deterministic pilot contains nine shallow, nine finite-depth, and nine
deep cases. All 27 cases are finite and in their declared parameter support.
The largest relative error in the linear-energy identity is
\(1.97\times10^{-15}\). The largest fixed-band \(N=512\) versus \(N=1024\)
DNO discrepancies are \(4.11\times10^{-4}\), \(7.76\times10^{-5}\), and
\(6.07\times10^{-5}\) for the shallow, finite-depth, and deep strata. A short
\(\Delta t=0.01\) versus \(0.005\) check on the steepest case in each stratum
has maximum discrepancy \(3.85\times10^{-8}\). The complete case ledger and
parameters are in
[`summary.json`](../outputs/jonswap_tma_pilot_20260725/summary.json), and the
[representative stress-case figure](../outputs/jonswap_tma_pilot_20260725/representative_stress_cases.png)
shows the most demanding shallow, finite-depth, and deep constructions used
by the pilot.

The population sampler implements 27 allocation cells: depth stratum,
\(\gamma\in\{1,3.3,5\}\), and right-moving energy fraction
\(r_d\in\{0,1/2,1\}\). It stores both realized phase arrays in the durable
pre-construction proposal. A reduced real-GL2 case has passed from this
proposal through the common refinement, 16-point post-acceptance selection,
whole-case archive, and split-aware view. This is an end-to-end software test;
the full-horizon cases in Section 7 test the paper numerical contract.

### 6.5 Residual-controlled GL2

For GL2 stage \(i\), component \(u\in\{\eta,\xi\}\), returned stage
\(V_i^u\), and fixed-point image \(\Phi_i^u\), the implemented residual is

\[
  R_n=
  \max_{i,u}
  \frac{\|\Phi_i^u-V_i^u\|_{\ell^2}}
       {\max\{\|\Phi_i^u\|_{\ell^2},
              \|V_i^u\|_{\ell^2},
              \operatorname{tiny}\}}.
\]

Here \(\|\cdot\|_{\ell^2}\) is the Euclidean norm of one component's Fourier
coefficients, and \(\operatorname{tiny}\) is the smallest positive normal
number in the working floating-point type. The two physical components are
normalized separately. With tolerance
\(10^{-8}\) and an eight-iteration cap, the integrator records every
substep's residual, iteration count, convergence, stage finiteness, and state
finiteness. A finite iteration-cap failure is distinct from a nonfinite
stage. Seven focused tests, 38 broader solver/data tests, and 20 existing
training-side GL2 tests pass. A real four-code-path \(N=256\) smoke converged all
32 sampled steps in two updates; its largest residual was
\(2.09\times10^{-9}\).

### 6.6 Frozen learning target

For a periodic field, \(P_{128}\) retains Fourier modes with
\(|k|\leq128\), and \(P_{128}^{\circ}\) additionally removes the spatial
mean. The symbol \(\mathcal G^{1024,8}_6\) denotes the order-six
Craig--Sulem recursion on 1024 points with pad factor eight. The paper's
target, including its period, is exactly

\[
q_{\mathrm{ref}}
=P_{128}^{\circ}
\left[
\mathcal G^{1024,8}_{6}
(P_{128}\eta;h)(P_{128}\xi)
\right],
\qquad L=2\pi.
\]

The evaluator requires JAX float64 mode and promotes its inputs to float64;
the delivered arrays may then be stored in float32. Tests fix the period and
all four numerical parameters, remove unresolved input
modes, enforce the zero-mean gauge, and verify translation covariance. This
defines the reproducible discrete operator to be learned by the
regenerated-dataset model. The paper does not infer continuum-DNO accuracy
from fixed-order grid agreement.

The static Stokes archives store \(\eta\), \(\xi\), and \(q_{\mathrm{ref}}\)
in float32 but retain their generating parameters and grid in float64.
Re-evaluating every case in the final shallow, deep, and partial-batch smokes
and then applying the declared float32 conversion reproduces every stored
field bit-for-bit. The archive metadata stores both its complete
configuration and the SHA-256 digest of that configuration; resume recomputes
the digest rather than trusting the stored digest string alone.

## 7. Full-horizon validation calculation

The predeclared boundary panel uses full-horizon \(\Delta t=0.01\) versus
\(0.005\) refinement with residual-controlled GL2. A case that fails this
comparison is recomputed at \(0.0025\) only if its \(0.005\) trajectory is
finite, remains in the graph domain, and solves every GL2 stage; the final
comparison is then \(0.005\) versus \(0.0025\). No further halving is allowed.
The horizons are \(T=200\) for Tanaka and Benjamin--Feir and \(T=20\) for the
Stokes propagation diagnostic. For JONSWAP/TMA, let \(\omega_p\) be the peak
angular frequency and let \(T_p=2\pi/\omega_p\) be the peak period. The
intended endpoint is \(H=16T_p\). Since fields are saved every \(0.08\), the
last saved time is

\[
  T_s=0.08\left\lfloor\frac{H}{0.08}\right\rfloor\leq16T_p.
\]

Coarse and fine trajectories use the same \(T_s\) and identical saved times.
The earlier \(t=0.08\) values in Section 3 remain an implementation smoke and
are not substituted for this calculation.

The two Stokes propagation cases are complete. Both primary
\(\Delta t=0.01\) versus \(0.005\) comparisons pass, so neither invoked the
conditional final halving. The largest normalized delivered-field
discrepancy is \(5.42\times10^{-8}\) for the finite-depth case and
\(8.24\times10^{-8}\) for the deep case. Every one of the 6000 GL2
substeps per case solved its implicit stages, with no nonfinite state and no
iteration-cap event.

Five of the seven Tanaka and Benjamin--Feir cases pass, with maximum paired
defect at most \(1.13\times10^{-5}\). The seam-centered steep Tanaka case and
the equation-(33) Benjamin--Feir stress case fail at essentially the same
physical times under both step sizes and are therefore rejected without the
ineligible final halving. The shallow and finite-depth JONSWAP/TMA cases pass
through their last saved time \(T_s\leq16T_p\), with maximum defects
\(4.52\times10^{-6}\) and \(5.63\times10^{-6}\). The deep case also passes
through \(T_s\), with maximum defect \(4.75\times10^{-6}\). Every implicit
stage in all six random-sea arms is solved. The complete panel therefore
accepts 10 of 12 fixed cases: both Stokes cases, three of four Tanaka cases,
two of three Benjamin--Feir cases, and all three random-sea cases. No case
invokes the final \(0.0025\) retry. This is evidence at selected support
points, not an estimate of population acceptance rates.

### 7.1 What happens before the two coarse failures

At both \(\Delta t=0.01\) and \(0.005\), five of the seven
Tanaka/Benjamin--Feir cases reach \(T=200\) with finite states and converged
GL2 stages. The seam-centered steep Tanaka case first has a nonfinite stage at
\(t=119.48\) under both step sizes. The equation-(33) Benjamin--Feir stress
case first fails at \(t=108.77\) and \(108.78\), respectively. The
\(0.005\) trajectories are themselves incomplete and contain unsolved or
nonfinite stages, so the \(0.0025\) retry is ineligible under the predeclared
rule. Both attempted cases are rejected.

Both initial states were reconstructed independently before interpreting the
late failures. The Tanaka case is the declared upper corner
\((h,\alpha)=(0.35,0.45)\). Its \(N=1024,2048,4096\) constructions agree on
\(|k|\leq128\) to at most \(6.38\times10^{-10}\), and a half-period
translation changes \(\eta,\xi,q_{\mathrm{ref}}\) by only
\(1.54\times10^{-16}\), \(2.17\times10^{-16}\), and
\(3.51\times10^{-11}\), respectively. The Benjamin--Feir state also agrees
across the three grids to at most \(1.2\times10^{-14}\). Its exact
equation-(33) sideband pattern is applied to the project's analytic
fifth-order carrier, so it is not described as the exact Fenton carrier used
by Xu and Guyenne. These checks and every parameter value are recorded in
`notes/full_horizon_boundary_case_support_audit_20260725.md`.

The Tanaka surface-potential formula contains a square root. For a component
moving with signed speed \(c\), define its argument by

\[
  \mathcal R(x)
  =
  \bigl(1+\eta_x(x)^2\bigr)
  \bigl(c^2-2g\eta(x)\bigr),
\]

where \(x\) is position, \(\eta_x\) is the surface slope, and \(g\) is
gravitational acceleration. A real-valued initial potential requires
\(\mathcal R(x)\geq0\) for every \(x\). The direct pre-square-root audit of
the upper-seam case satisfies

\[
  \min_x\frac{\mathcal R(x)}{c^2}=0.372151>0.
\]

Thus the historical clamp did not change this validation case and cannot
explain its late failure. The production constructor now records and rejects
a specification with \(\min_x\mathcal R(x)<0\), because replacing a negative
value by zero would silently change the sampled initial condition.

The saved frames determine whether the stage failure is the beginning or the
end of the numerical deterioration. For a saved periodic field \(f\), let
\(\widehat f_k\) be its Fourier coefficient. Define the delivered positive
modes and the high band by

\[
  D=\{1,\ldots,128\},
  \qquad
  B=\{80,\ldots,128\}.
\]

For either set \(I\), define its energy by

\[
  E_I(f,t)=\sum_{k\in I}|\widehat f_k(t)|^2.
\]

The generated high-band tail is

\[
  G_B(f,t)
  =
  \frac{\max\{E_B(f,t)-E_B(f,0),0\}}{E_D(f,0)}.
\]

This quantity is zero when the high-band energy has not increased from its
initial value. It is normalized by the initial energy in all delivered
nonzero modes, so it can be compared across time without dividing by a
possibly tiny high-band value.

Across the five complete coarse trajectories, the largest values of \(G_B\)
for \(\eta,\xi,q_{\mathrm{ref}}\) are respectively
\(3.14\times10^{-5}\), \(1.62\times10^{-5}\), and
\(3.20\times10^{-3}\). Their largest surface-slope amplification,

\[
  \frac{\max_x|\eta_x(x,t)|}{\max_x|\eta_x(x,0)|},
\]

is \(1.548\), and their largest relative Hamiltonian drift is
\(1.43\times10^{-7}\).

The two failures depart from that control envelope long before GL2 stops.
For the Tanaka case, \(G_B(q_{\mathrm{ref}},t)\) first exceeds \(0.01\) at
\(t=104.16\) and reaches \(0.308\); the slope is amplified by \(5.95\), and
the Hamiltonian drift reaches \(7.68\times10^{-2}\). For the
Benjamin--Feir case, the same tail first exceeds \(0.01\) at \(t=101.28\)
and reaches \(0.281\); the slope is amplified by \(10.18\), and the
Hamiltonian drift reaches \(1.58\times10^{-2}\). Their minimum water-column
fractions remain \(0.9841\) and \(0.9932\), respectively.

Thus the implicit failure is a terminal symptom, not an isolated first
event. The trajectory first develops a sustained cascade within the
delivered band toward the cutoff \(K=128\), accompanied by rapid surface
steepening. Halving the time step from \(0.01\) to \(0.005\) reproduces the
Tanaka failure at \(t=119.48\) and moves the Benjamin--Feir failure only from
\(t=108.77\) to \(108.78\). Ordinary time truncation at this scale therefore
does not explain either cascade. The next controlled calculation is the
one-factor grid, padding, cutoff, and series-order comparison, not an
empirical row filter.

![Framewise diagnosis of the two coarse failures.](../outputs/full_horizon_refinement_panel_20260725/coarse_failure_spectral_audit.png)

The numerical values, source hash, definitions, threshold-crossing times,
figure hashes, and exact failed-step telemetry are stored in
`outputs/full_horizon_refinement_panel_20260725/coarse_failure_spectral_audit_metrics.json`.
The figure is reproduced by
`scripts/plot_full_horizon_coarse_failures.py`.

### 7.2 Static Craig--Sulem term ordering before the cascade

A separate CPU calculation asks whether the order-six Craig--Sulem sum is
already disordered at states that precede the rapid high-band growth. It does
not advance either state in time. Let \(P=P_{128}^{\circ}\) be projection to
Fourier modes \(|k|\leq128\) followed by removal of the spatial mean. At one
fixed state \((\eta,\xi,h)\), define the projected partial sum through order
\(m\) by

\[
  S_m
  =
  P\!\left[
  \mathcal G_m^{1024,8}(P\eta;h)(P\xi)
  \right],
  \qquad 0\leq m\leq6.
\]

Here \(\mathcal G_m^{1024,8}\) is the Craig--Sulem sum through order \(m\),
evaluated on 1024 grid points with pad factor eight. Define its individual
terms by

\[
  T_0=S_0,
  \qquad
  T_m=S_m-S_{m-1}\quad(1\leq m\leq6).
\]

For grid values \(f(x_j)\), where \(x_j=2\pi j/1024\), define

\[
  \|f\|_{\mathrm{rms}}
  =
  \left(\frac{1}{1024}\sum_{j=0}^{1023}|f(x_j)|^2\right)^{1/2}.
\]

Three dimensionless quantities summarize the ordering:

\[
  \rho_m=\frac{\|T_m\|_{\mathrm{rms}}}
                 {\|T_{m-1}\|_{\mathrm{rms}}},
  \qquad
  \delta_6=\frac{\|T_6\|_{\mathrm{rms}}}{\|S_6\|_{\mathrm{rms}}},
  \qquad
  C_6=\frac{\sum_{m=0}^{6}\|T_m\|_{\mathrm{rms}}}
             {\|S_6\|_{\mathrm{rms}}}.
\]

The ratio \(\rho_m\) tests whether one term is smaller than the preceding
term. The ratio \(\delta_6\) measures the size of the last retained term
relative to the delivered sum. The cancellation index \(C_6\) equals one
when the term norms add without cancellation and becomes large only when
large terms substantially cancel in their sum.

The source states are saved from the \(\Delta t=0.01\) coarse trajectories.
The Tanaka state is taken at \(t=80\), and the equation-(33)
Benjamin--Feir state is taken at \(t=88\). At those times the generated
high-band quantity \(G_B(q_{\mathrm{ref}},t)\), defined in Section 7.1, is
still below the predeclared \(10^{-4}\) restart bound.
For Tanaka, \((\rho_1,\rho_2)=(0.333,0.110)\); for Benjamin--Feir,
\((\rho_1,\rho_2)=(0.0618,0.299)\). The remaining ratios and two sum
diagnostics are

| case and time | \(\rho_3\) | \(\rho_4\) | \(\rho_5\) | \(\rho_6\) | \(\delta_6\) | \(C_6\) | \(G_B(q_{\mathrm{ref}},t)\) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tanaka, \(t=80\) | \(0.337\) | \(0.233\) | \(0.238\) | \(0.268\) | \(1.41\times10^{-4}\) | \(1.060\) | \(4.65\times10^{-5}\) |
| Benjamin--Feir, \(t=88\) | \(0.0786\) | \(0.329\) | \(0.168\) | \(0.321\) | \(2.46\times10^{-5}\) | \(1.036\) | \(6.26\times10^{-5}\) |

Every successive term is smaller than its predecessor through order six, the
sixth term is small relative to the order-six sum, and neither state requires
large cancellation among the seven terms. This static test therefore finds no
loss of Craig--Sulem term ordering before the observed cascade. It does not
establish convergence of the time evolution or rule out later sensitivity to
the order, grid, padding, or delivered cutoff.

The dynamic calculation begins from a saved state rather than from the
original initial condition.  Let \(t_r\) denote its restart time, let
\(B=\{80,\ldots,128\}\), and let \(D=\{1,\ldots,128\}\).  Using the energy
\(E_I\) defined in Section 7.1, define the restart-band growth by

\[
  R_B(q,t;t_r)
  =
  \frac{\max\{E_B(q,t)-E_B(q,t_r),0\}}{E_D(q,t_r)},
  \qquad t\geq t_r.
\]

This is not the full-trajectory quantity \(G_B\), whose reference time is
zero.

In the dynamic Tanaka restart, the baseline \(N=1024\) arm fails again at
\(t=119.48\). The \(N=2048\), \(M=5\), and \(M=4\) arms remain finite with
every GL2 stage solved through \(t=122\), whereas doubling product padding
reproduces the failure at \(t=119.48\). The maximum values of the generated
restart-band quantity \(R_B(q_{\rm ref},t;80)\) are
\(0.562\), \(0.463\), and \(0.390\) for the \(N=2048\), \(M=5\), and \(M=4\)
arms. Their threshold times are essentially unchanged from the baseline.
These controls prevent the terminal nonfinite state without removing the
preceding cascade.

Changing only the Tanaka delivered cutoff from \(K=128\) to \(K=192\) keeps
the arm finite and stage-solved through \(t=122\). In this arm,
\(R_B(q_{\rm ref},t;80)\) stays below \(3.20\times10^{-5}\). For
\(A=\{129,\ldots,192\}\) and \(D=\{1,\ldots,128\}\), the largest added-band
ratio

\[
  \frac{\max_t\sum_{k\in A}|\widehat q_{{\rm ref},k}(t)|^2}
       {\sum_{k\in D}|\widehat q_{{\rm ref},k}(80)|^2}
\]

is \(2.80\times10^{-4}\). This is evidence of cutoff crowding in the Tanaka
continuation.

The completed Benjamin--Feir controls do not support a common cutoff remedy.
The baseline fails at \(t=108.78\), the \(N=2048\) arm at \(t=109.175\), the
pad-factor-16 arm at \(t=108.78\), and the \(K=192\) arm earlier at
\(t=103.755\). The three \(K=128,M=6\) arms cross the recorded \(R_B\)
thresholds at the same saved times. The \(M=5\) and \(M=4\) arms remain
finite and stage-solved through \(t=112\), but their maximum restart growth
is \(0.182\) and \(0.283\); both cross \(10^{-4}\), \(10^{-3}\), and
\(10^{-2}\) at the same saved times as the baseline. Lowering the order
therefore changes the terminal solve but does not remove the preceding
growth. In the finite prefix of the \(K=192\) arm, the maximum added-band
ratio for \(129\leq k\leq192\), normalized by the nonzero delivered-band
energy at \(t_r=88\), is \(4.82\times10^{-2}\). No common reference-contract
change is supported by these restart calculations.

The complete static values and the current dynamic results are stored in
`outputs/precascade_spatial_order_diagnostic_20260725/`. The live
`summary.json` records the baseline source-archive configuration fingerprint
`91eb6c187fd079030fef08b8963b50b1c237d3e9d8940c172777a806fb997b5e`
and script SHA-256
`98d67a44b5c23a76e0e0ae08d78dd119aaa2c6c5c984fd97463ced5d3ee999a2`.
The completed summary has file SHA-256
`84c890138fbcef3cb345d083facc4c07a9d816f5508a1335481b7f41d1529948`.

## Reproduction

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python -m solver.gen_data.generate_stokes_dataset \
    --output outputs/paper_dataset_stokes_ursell_smoke_20260725.npz \
    --regime shallow --target_samples 16 --batch_size 16 --seed 42 \
    --overwrite

JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python -m solver.gen_data.generate_stokes_dataset \
    --output outputs/paper_dataset_stokes_deep_smoke_20260725.npz \
    --regime deep --target_samples 16 --batch_size 16 --seed 43 \
    --overwrite

JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python -m unittest solver.gen_data.tests.test_paper_acceptance_smoke

JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python -m unittest solver.gen_data.tests.test_paper_acceptance_cross_family

JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python -m unittest solver.gen_data.tests.test_stokes_spatial_smoke

JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python solver/gen_data/tests/test_tanaka_tangent_hermite.py

JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/plot_paper_dataset_shape_stress.py

JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python -m unittest \
    solver.gen_data.tests.test_benjamin_feir_jcp09 \
    solver.gen_data.tests.test_jonswap_tma \
    solver.gen_data.pipeline.tests.test_dno_target \
    solver.gen_data.pipeline.tests.test_acceptance \
    solver.solvers.test_time_integrator_telemetry \
    solver.gen_data.tests_stokes_ursell_sampling \
    solver.gen_data.tests_stokes_archive_contract

JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/pilot_jonswap_tma_random_sea.py

JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/plot_stokes_ursell_diagnosis.py

CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_ENABLE_X64=True \
  uv run python scripts/run_full_horizon_refinement_panel.py --overwrite

JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/plot_full_horizon_coarse_failures.py

JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/test_precascade_spatial_order_diagnostic.py

JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/run_precascade_spatial_order_diagnostic.py \
    --static-only --overwrite

JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python -m unittest \
    solver.gen_data.pipeline.tests.test_case_allocation \
    solver.gen_data.pipeline.tests.test_archive \
    solver.gen_data.pipeline.tests.test_build_dataset_view

JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python -m unittest \
    solver.gen_data.tests_stokes_population \
    solver.gen_data.tests.test_stokes_static_pipeline \
    solver.gen_data.tests_tanaka_population \
    solver.gen_data.tests_benjamin_feir_population \
    solver.gen_data.tests_jonswap_tma_population \
    solver.gen_data.tests.test_trajectory_family_adapters

JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/run_static_stokes_exact_target_pilot.py \
    --output-dir outputs/static_stokes_exact_target_pilot_reproduction

JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python train-jax-10m/tests_paper_dataset_view.py
```
