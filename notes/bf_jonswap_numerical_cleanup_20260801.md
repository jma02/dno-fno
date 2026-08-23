> **Historical / superseded development note.** This file records numerical
> cleanup reasoning that preceded the final paper-dataset contracts; it is not
> the current release specification.  The current family map is static Stokes
> revision 2, corrected Tanaka revision 3, Benjamin--Feir revision 4, and
> JONSWAP/TMA revision 4.  Use the
> [data-generation contract](../solver/gen_data/README.md) and
> [release-readiness map](paper_dataset_generation_readiness_20260726.md) for
> the operational contract, the
> [Benjamin--Feir support decision](bf_stream930_simple_support_20260802.md)
> and [JONSWAP/TMA support decision](jonswap_defensible_support_review_20260804.md)
> for the adopted populations, and the revision-4 family completion auditors
> under [`scripts/`](../scripts/) for final release evidence.

# Benjamin--Feir and JONSWAP/TMA numerical cleanup

Date: 2026-08-01

## Decision in one paragraph

The visually smooth Benjamin--Feir (BF) and JONSWAP/TMA trajectories are not
automatically numerically trustworthy.  In both families, a small tail can
focus until the maximum surface slope is near one, transfer substantial energy
to the largest retained modes, and then accumulate percent-level Hamiltonian
error.  This is different from the old Tanaka interpolation artifact: it is a
whole-trajectory resolution/nonbreaking-validity problem, not an initially
jagged profile.  The resolved Benjamin--Feir candidate uses 1024 internal
points, the fixed sharp band \(K=256\), DNO order \(M=4\), and
\(\Delta t=0.01\).  The final JONSWAP/TMA contract uses 2048 internal points,
the sharp band \(K=704\), the same order and time step, and at most five
fixed-point updates per GL2 stage.  Both retain the fixed 1024-point,
\(P_{128},M=6\) learning target.  JONSWAP/TMA additionally receives the
20-peak-period nonlinear adjustment described below, and its input population
is restricted to the moderate range \(k_pH_s/2\leq0.08\).  This is a declared
range of study, not a wave-breaking theorem.  Every autonomous
Benjamin--Feir and JONSWAP trajectory must be finite,
solve every implicit stage, preserve positive water depth, and have dense
internal Hamiltonian drift at most \(10^{-3}\).  A failed attempt is recorded
and replaced from the same predeclared parameter category.  A sign-count or
visual-shape filter would not diagnose this failure.

## Quantities used below

For surface elevation \(\eta(x,t)\), surface potential \(\xi(x,t)\), and the
Dirichlet--Neumann operator \(G(\eta)\), the Hamiltonian is

\[
 H(t)=\frac12\int_0^{2\pi}
 \left\{\xi\,G(\eta)\xi+g\eta^2\right\}\,dx.
\]

For the revision-3 production check, \(G=G^K_4\) is evaluated in the internal
evolution band: \(K=256\) for Benjamin--Feir and \(K=704\) for JONSWAP/TMA.
The relative conservation error is

\[
 d_H=\max_j\frac{|H(t_j)-H(0)|}
 {\max\{|H(0)|,\varepsilon_{64}\}}.
\]

Here \(t_j\) ranges over every saved autonomous state and
\(\varepsilon_{64}\) is the smallest positive normal float64 number used by
the implementation.  For the present nonzero-energy waves, the denominator
is simply \(|H(0)|\).  This check precedes training-time subsampling.

For a field \(v\), the delivered high-band fraction is

\[
 E_{96:128}(v)=
 \frac{\sum_{96\le |k|\le128}|\widehat v_k|^2}
      {\sum_{1\le |k|\le128}|\widehat v_k|^2}.
\]

We report this quantity for \(v=G(\eta)\xi\).  It is a resolution diagnostic,
not an acceptance rule: a broadband random sea can have a large value while
remaining well resolved and conserving \(H\).

## What the literature actually supports

### Benjamin--Feir

Xu and Guyenne [JCP 228 (2009), 8446--8466](../JCP09.pdf) use the Hamiltonian
above as a global accuracy diagnostic.  Their canonical BF experiment has
carrier mode 9, sidebands 7 and 11, carrier steepness 0.13, sideband-to-carrier
amplitude ratio 0.10, common sideband phase \(-\pi/4\), an accurately computed
steady Stokes carrier, DNO order \(M=4\), and \(\Delta t=0.01\).  It conserves
energy to about \(10^{-5}\) through \(t/T=600\); the error rises toward
\(10^{-3}\) only near the stated computational breakdown.

Our constructor preserves the algebraic form of their equation (33), but our
population is an extrapolation rather than a reproduction of that experiment:
carrier modes range from 4 to 20, carrier and perturbation amplitudes vary,
perturbation ratios reach 0.20, and a global translation is uniform on the
periodic domain.  The relative sideband phase remains fixed at \(-\pi/4\).
The carrier is the project's analytic fifth-order Stokes approximation.  Those
extensions are reasonable objects to study, but they must be supported by our
own convergence evidence rather than attributed to JCP09.

The earlier [construction audit](benjamin_feir_jcp09_audit_20260723.md)
already separated this definitional difference from the breakdown mechanism.
For an archived failing case, the old empirical construction failed at
\(t=60.88\), whereas the literal equation-(33) and finite-depth-corrected
constructions failed at \(t=60.96\).  Extra GL2 iterations and halving
\(\Delta t\) exposed the same rapid Hamiltonian loss.  The current constructor
has removed the empirical cross-terms and uses each sideband's own
wavenumber.  Replacing the analytic fifth-order carrier by a numerical Fenton
carrier would make the canonical benchmark more literal, but the available
matched calculation does not support it as the cure for the focusing tail.

JCP09 also explains that DNO-series accuracy improves with order only up to a
problem-dependent optimum.  They normally use \(M=4\), check orders below 10,
dealias all products, and use the Hou--Li power-36 filter when filtering is
needed.  Thus simply increasing \(M\) is not a resolution argument.

### JONSWAP/TMA

The current constructor correctly transforms the JONSWAP frequency density to
wavenumber density using the group velocity, applies the finite-depth TMA
factor, normalizes to the requested significant height, and constructs the
linear finite-depth pair \((\eta,\xi)\).  The continuous spectrum follows the
Joint North Sea Wave Project report and the finite-depth factor of Bouws et
al.; the project's finite Fourier window and cell integration are explicit
discretization choices.  The full formula audit is given in the
[condensed dataset account](paper_dataset_generation_condensed_20260726.tex).
The missing standard step is the nonlinear adjustment used by Dommermuth and
by HOS-Ocean.  It replaces the autonomous equations temporarily by

\[
 \eta_t=G_0\xi+A(t)\bigl(G_M(\eta)\xi-G_0\xi\bigr),
\]

\[
 \xi_t=-g\eta+A(t)\left[-\frac12\xi_x^2+
 \frac12\frac{(G_M(\eta)\xi+\eta_x\xi_x)^2}{1+\eta_x^2}\right],
 \qquad
 A(t)=1-\exp\!\left[-(t/T_a)^n\right].
\]

Here \(G_0\) is the flat-surface DNO, \(G_M\) is the order-\(M\)
Craig--Sulem approximation used by the nonlinear solver, \(T_a\) is the
adjustment time, and \(n\) is the ramp exponent.  The JONSWAP peak frequency
and period are
\(\omega_p=\sqrt{gk_p\tanh(k_p h)}\) and \(T_p=2\pi/\omega_p\), respectively.
HOS-Ocean uses this formula directly in its
[reference implementation](https://github.com/LHEEA/HOS-ocean/blob/master/sources/HOS/resol_HOS.f90#L1768-L1780)
and exposes both \(n\) and \(T_a\), measured in peak periods, as numerical
parameters.  Its current example uses \(n=2\) and \(T_a=10T_p\), while its
legacy example uses \(n=4\).  We use \(n=4\) and \(T_a=10T_p\) as an explicit
project choice and validate it by the adjustment and spatial-resolution
calculations reported below; neither value is attributed as a unique
literature prescription.  The purpose of the adjustment is to let bound
harmonics form gradually instead of exciting high-frequency free standing
waves by inserting a linear random sea directly into the fully nonlinear
equations [Dommermuth (2000)](https://doi.org/10.1016/S0165-2125(00)00047-0).

During this adjustment, ordinary Hamiltonian conservation is not expected:
the equations have an explicitly time-dependent Hamiltonian.  Per-trajectory
acceptance checks finiteness, implicit-stage convergence, and positive water
depth.  Spectral decay and cross-resolution agreement are fixed
method-validation diagnostics, not per-trajectory acceptance rules.  A
subsequent autonomous production trajectory must restart the Hamiltonian
reference at the adjusted state.

HOS-Ocean recommends approximately \(k_{\max}\simeq8k_p\), followed by an
explicit convergence study
([numerical-parameter guidance](https://lheea.gitlab.io/HOS-Ocean/choice-numerical-parameters.html)).
Our shallow support reaches \(k_p=24\), for which the historical revision-2
evolution band \(K=128\) gave only \(K/k_p=5.33\).  This observation motivated
the explicit internal-band comparison that ultimately selected \(K=704\) on
2048 points, while the delivered learning target remains
\(P_{128}G_6(P_{128}\eta)P_{128}\xi\).

The input peak steepness used below is

\[
 \epsilon_p=\frac{k_pH_s}{2}.
\]

Ducrozet et al. call \(\epsilon_p=0.08\) typical of a moderate JONSWAP sea
state in a HOS calculation
([Phys. Rev. Fluids 6, 064803 (2021)](https://doi.org/10.1103/PhysRevFluids.6.064803)).
Their separate applicability study shows why a steep sea that survives a
coarse HOS cutoff need not be accurately resolved: resolving more short waves
can expose the approach to breaking and reduce the range of successful
potential-flow calculations
([Ocean Engineering 142 (2017), 233--244](https://doi.org/10.1016/j.oceaneng.2017.07.003)).
We therefore use \(\epsilon_p\leq0.08\) as a conservative population bound
for this nonbreaking solver.  It is imposed before the phases are drawn and
does not replace the trajectory calculation.

## Evidence from the historical revision-2 dataset

The following statistics diagnose the numerical contract that revision 3
replaces; they are not revision-3 acceptance rates.  Across 4,096 completed
revision-2 BF trajectories:

- 107 (2.61%) have \(d_H>10^{-3}\);
- 542 (13.23%) have \(E_{96:128}>1\%\);
- 229 (5.59%) have \(E_{96:128}>5\%\);
- every Hamiltonian failure has \(E_{96:128}>1\%\), and 106 of 107 exceed
  5%; but 123 cases exceed 5% while still passing the Hamiltonian check.

This last fact is why high-band occupancy should trigger a convergence test
rather than reject a trajectory by itself.

The worst BF replay, case `2306062911539250218`, is reconstructed from its
durable proposal to float32 archive accuracy.  Its initial high-band fraction
is only \(5.20\times10^{-9}\).  Focusing later produces
\(E_{96:128}=12.447\%\), a nondecayed spectral wall at mode 128, maximum
surface slope 1.249, and \(d_H=2.7025\%\).  Thus the defect is generated during
evolution; it is not initial-data corruption.

An exact clean-initial-condition replay at internal \(K=256\), Hou--Li, and
\(M=4\) does not converge this case.  It becomes nonfinite at \(t=68.88\),
whereas the archived \(K=128\) arm reaches its focusing maximum near
\(t=74.48\).  At the last finite frame, \(\max|\eta_x|=1.728\); the internal
and delivered-\(P_{128}\) Hamiltonian errors have already reached 2.892% and
2.353%.  Before failure, the delivered \(P_{128}\) fields depart from the
archived arm by 18.0%, 10.3%, and 47.1% in relative \(L^2\) for
\(\eta,\xi,G(\eta)\xi\).  The discrepancy first exceeds \(10^{-3}\) near
\(t=59.36\), before the visible extreme.  A larger hidden band therefore does
not repair this trajectory; it demonstrates lack of cross-resolution
convergence.

Removing the Hou--Li filter does not change that verdict.  The matched sharp
\(K=256,M=4\) arm becomes nonfinite even earlier, at \(t=66.32\), with
\(\max|\eta_x|=1.959\), 18.17% of \(G(\eta)\xi\) energy in its internal edge
band, and 2.25% internal Hamiltonian error at the last finite frame.  Thus the
failure is not filter-induced dissipation: without damping, the unresolved
cascade reaches the wider spectral wall more strongly.

Two matched controls show that this is not a generic instability caused by
using \(K=256\).  Case `2306062911539249398` passes the old \(K=128\)
Hamiltonian rule with \(d_H=6.82\times10^{-4}\), but has 7.16% delivered
high-band energy.  Its sharp \(K=256,M=4\) replay becomes nonfinite at
\(t=96.40\), after the projected \(G(\eta)\xi\) has already departed from the
archive by 48.5% in relative \(L^2\).  Thus the old invariant measured behind
the \(K=128\) wall can give a false pass.

In contrast, quiet case `2306062911539249711` completes \(t=200\) at sharp
\(K=256,M=4\).  Every GL2 stage converges, internal
\(d_H=5.53\times10^{-9}\), and the internal upper-quarter energy is below
\(7.0\times10^{-21}\).  At the exact 200 archived times, the global relative
\(L^2\) differences between the archived \(K=128,M=6\) arm and the projected
\(K=256,M=4\) arm are \(8.05\times10^{-7}\), \(7.93\times10^{-7}\), and
\(8.31\times10^{-7}\) for \(\eta,\xi,G(\eta)\xi\), respectively; the largest
single-frame difference is \(1.10\times10^{-6}\).  The wider method therefore
reproduces a resolved BF trajectory while exposing the under-resolved
focusing cases.

Among the 4,096 completed revision-2 JONSWAP/TMA trajectories, median and
99th-percentile Hamiltonian errors are \(5.70\times10^{-8}\) and
\(1.01\times10^{-3}\), while the maximum is 2.521%.  The six largest errors
all occur when the maximum surface slope lies between 0.909 and 1.113.  In
contrast, many intentionally broadband shallow seas have large high-band
fractions and negligible Hamiltonian error.

## Controlled JONSWAP replays

All replays below reconstruct exact float64 proposals from revision 2 and then
compare candidate numerical methods.  They are method-selection evidence, not
rows retained for revision 3.

### Worst case 1675

This is a one-directional deep-water, \(\gamma=1\) realization with
\(k_p=10.8583\), \(H_s=0.0269280\), and
\(k_pH_s/2=0.14620\).  The archived \(K=128,M=6\) trajectory remains finite
but reaches maximum slope 1.113, \(E_{96:128}=14.72\%\), and
\(d_H=2.521\%\).

Increasing the internal band does not reveal a converged solution:

| Evolution | First nonfinite saved time |
| --- | ---: |
| \(K=256\), Hou--Li, \(M=6\) | 0.72 |
| \(K=256\), Hou--Li, \(M=4\) | 8.00 |
| \(K=256\), Hou--Li, \(M=3\) | 0.88 |

The standard \(n=4,T_a=10T_p\) adjustment remains finite for 16 peak
periods, but it ends with maximum slope 1.161 and 9.87% high-band energy.  An
autonomous restart then becomes nonfinite at time 1.60, after the slope reaches
1.546.  The adjustment is therefore not a cure for this realization; it
exposes that the trajectory has entered a steep focusing regime outside a
reliably resolved nonbreaking graph solution.

### Moderate control case 1324

This matched deep-water, \(\gamma=1\), one-directional case has
\(k_pH_s/2=0.05397\).  The same adjustment stays finite and has maximum slope
0.273.  The following autonomous 16-period trajectory has maximum slope 0.282,
all GL2 stages converged below \(1.31\times10^{-9}\), and
\(d_H=1.55\times10^{-7}\).  The unadjusted control also remains healthy, with
\(d_H=1.99\times10^{-7}\).  Hence the adjustment implementation is sound; the
worst-case failure is specific to the steep tail rather than a universal
failure of the initialization method.

### Shallow \(k_p=24\) resolution controls

The HOS-Ocean \(K\simeq8k_p\) recommendation is consequential, not merely
conservative.  Mild accepted case 1259 has \(k_pH_s/2=0.01066\).  Its archived
\(K=128\) arm looks healthy and has \(d_H=3.63\times10^{-9}\), but a sharp
\(K=192,M=6\) arm is also internally healthy and differs from the archived
delivered trajectory by as much as 3.71%, 1.13%, and 10.08% in relative
\(L^2\) for \(\eta,\xi,G(\eta)\xi\).  The disagreement begins by \(t=2.56\).
The wider arm has internal \(d_H=2.08\times10^{-9}\) and only
\(1.14\times10^{-3}\) of its \(G(\eta)\xi\) energy in the upper quarter of the
internal band, so
it is the resolved arm; visual smoothness and the \(K=128\) invariant alone
did not reveal the wrong dynamics.

For that mild case, \(K=192,M=4\) and \(M=6\) agree to
\(3.37\times10^{-8}\) over the whole trajectory.  In contrast, steep accepted
case 141 has \(k_pH_s/2=0.14906\).  Its sharp \(K=192,M=6\) arm reaches
internal \(d_H=5.09\times10^{-3}\), while \(M=4\) reduces this to
\(9.05\times10^{-4}\).  Halving \(\Delta t\) leaves the drift unchanged and
changes all fields by at most \(8.82\times10^{-6}\), ruling out temporal error.
The \(M=4\) and \(M=6\) arms disagree appreciably only in this steep tail,
consistent with JCP09's warning that the DNO series has a
steepness-dependent optimal truncation rather than monotone convergence in
\(M\).

The next spatial comparison makes a fixed rule possible.  For mild case 1259,
the sharp \(K=192\) and \(K=256\) arms differ by up to
\(1.09\times10^{-3}\), \(3.12\times10^{-4}\), and
\(2.55\times10^{-3}\) in delivered \(\eta,\xi,G(\eta)\xi\), whereas
\(K=256\) and \(K=320\) differ by only \(5.40\times10^{-6}\),
\(1.45\times10^{-6}\), and \(1.28\times10^{-5}\).  Thus \(K=256\) is
converged for the mild control.  Its internal Hamiltonian drift is
\(1.65\times10^{-9}\), and only \(6.72\times10^{-5}\) of its DNO energy lies
in modes 192--256.

The steep case does not become a valid \(K=256\) solution: its internal drift
rises to 1.278%, 6.15% of DNO energy remains in modes 192--256, and some GL2
stages miss the \(10^{-8}\) tolerance.  This is precisely the case that should
be recorded as a failed attempt and replaced.  Together these two controls
show that \(K=256\) resolves the mild realization and rejects the steep one.
They do not establish that \(K=256\) resolves every accepted realization; the
independent panel below supplies the missing counterexample.

The end-to-end method also passes a control.  Case 1259 completed a
20-\(T_p\) adjustment followed by 16 autonomous periods at sharp
\(K=256,M=4\).  The ramp ended at \(A=0.999999878\); every subsequent GL2
stage converged, internal \(d_H=1.71\times10^{-9}\), and the internal
upper-quarter fraction remained \(5.42\times10^{-5}\).  The adjustment changed
the realized RMS elevation by less than 1%, while allowing the bound
components to form as intended.

A repeated Hou--Li filter at \(K=192\) is not suitable when the delivered band
ends at 128.  Its mode-128 multiplier is 0.9999835 per step; over 3,872 steps
this compounds to 0.9382.  The observed 0.2% Hamiltonian drift in that arm is
filter accumulation, while the matched sharp arm conserves the internal
Hamiltonian to \(2.1\times10^{-9}\).  Use a sharp internal Galerkin wall and
require spectral decay before that wall, or place a smooth filter farther away.

### Revision-3 adjacent-cutoff panel

The complete revision-3 calculation compares independently evolved cutoff
pairs after the full JONSWAP adjustment.  Three Benjamin--Feir cases and the
finite- and deep-water JONSWAP cases agree between \(K=256\) and \(K=320\):
their largest \(P_{128}\) discrepancy in any of
\(\eta,\xi,G(\eta)\xi\) is respectively
\(9.02\times10^{-8}\), \(1.17\times10^{-6}\), and
\(5.26\times10^{-5}\).  The shallow \(k_p=24,\gamma=1\) realization does not.
At its adjusted handoff, the \(K=256/320\) discrepancies are

\[
 (e_\eta,e_\xi,e_{G\xi})(0)
 =(3.162,1.314,5.089)\times10^{-3}.
\]

They grow during the autonomous interval to maxima
\((2.736,1.148,4.438)\times10^{-2}\).  Modes 97--128 contain most of the
error.  Every stage in both arms is solved, their Hamiltonian drifts are
\(1.60\times10^{-5}\) and \(2.74\times10^{-5}\), and a best common spatial
translation removes only a small part of the discrepancy.  Thus neither the
integrator, the handoff, nor a rigid phase shift explains the failure.  It is
created by cutoff sensitivity during the nonlinear adjustment and then
amplified by the autonomous flow.

An independently adjusted \(K=384\) arm also rules out \(K=320\) as the
replacement.  For \(K=320/384\), the handoff errors decrease to

\[
 (7.514,3.028,11.954)\times10^{-4},
\]

but the last component still exceeds \(10^{-3}\), and the full-trajectory
maxima are \((1.509,0.631,2.485)\times10^{-2}\).  The \(K=384\) arm remains
finite, solves every stage, and has Hamiltonian drift
\(3.87\times10^{-5}\).  Fixed \(K=320\) is therefore not accepted.

The \(K=384/448\) handoff passes, with errors
\((1.283,0.540,2.149)\times10^{-4}\), but the 16-period maxima are still
\((0.783,0.329,1.297)\times10^{-2}\).  The \(K=448\) arm is healthy and only
\(2.26\times10^{-8}\) of its adjusted surface energy lies in modes 385--448.
The adjustment is therefore spectrally converging; the autonomous flow is
amplifying an increasingly small cutoff perturbation.  At that point,
\(K=448\) was already close to the \(N_x=1024\) Nyquist limit, so the next
comparison had to recompute both adjacent arms on a larger grid.  The result
of that calculation is recorded next.

### Final 2048-point cutoff calculation

The required larger-grid calculation is now complete for the same shallow
sentinel.  Both arms use \(N_x=2048\), order \(M=4\), padding factor eight,
step \(0.01\), saved spacing \(0.08\), residual tolerance \(10^{-8}\), and an
iteration cap of five.  Each arm is adjusted independently for 20 peak periods
and then evolved autonomously for 16 peak periods.  For a field \(f\) sampled
on \(N\) points, define the normalized coefficient

\[
 \widehat f_k^{(N)}=\frac1N\sum_{j=0}^{N-1}
 f_j\exp(-2\pi i jk/N).
\]

At every common saved time, the comparison copies the coefficients with
\(|k|\le128\) from each 2048-point state to a 1024-point grid, reconstructs
\(\eta\) and \(\xi\), and recomputes the order-six target there.  It does not
resample either arm's internal value of \(G(\eta)\xi\).

The \(K=704\) versus \(K=768\) maximum relative whole-field differences over
the complete autonomous history are

\[
 (e_\eta,e_\xi,e_{G\xi})
 = (4.64267,1.93035,7.91931)\times10^{-4}.
\]

All three are below the predeclared \(10^{-3}\) requirement.  The more widely
separated \(K=640\) versus \(K=768\) comparison does not pass, so \(K=640\) is not an
adequate production cutoff.  Diagnostics restricted to modes
\(96\le |k|\le128\) are about \(3\times10^{-3}\).  Hence this calculation
certifies only the stated global whole-field comparison; it does not establish
a separate bandwise error bound.  Some stage residuals are close to
\(10^{-8}\), and the calculation contains only one deliberately difficult
sentinel.  The 27-category GPU smoke therefore remains required before bulk
JONSWAP/TMA generation.

The current production source was also replayed from the archived full
internal \(K=704\) endpoint.  It reproduced all 289 autonomous saved frames
with maximum relative \(L^2\) differences
\((3.99\times10^{-16},4.66\times10^{-16},1.17\times10^{-15})\) in
\((\eta,\xi,G(\eta)\xi)\).  The replay was accepted, solved every stage, had
maximum residual \(9.994198\times10^{-9}\), Hamiltonian drift
\(1.831187\times10^{-5}\), minimum water column \(0.0387699\), and CPU wall
time 197.355 seconds.  This checks that the final source implements the
archived sentinel calculation; it does not broaden one sentinel into a
population result.

### Benjamin--Feir focusing controls

The fixed Xu--Guyenne-style point
\((n_c,\Delta n,\epsilon_c,\rho,\phi,h)=(9,2,.13,.10,-\pi/4,5)\) is a
validation case, not a forced dataset row.  Both its \(K=256\) and \(K=320\)
project implementations become nonfinite and are rejected.  This one result
does not define an a-priori parameter cutoff.

The milder point with \(\epsilon_c=1/9\) has the leading-NLS maximum-growth
detuning \(1/\sqrt2\) and passes.  Both cutoff arms complete \(T=200\), their
largest cross-resolution error is \(1.38\times10^{-8}\), and their Hamiltonian
drift is \(5.56\times10^{-5}\).  The carrier falls to 0.538 of its initial
amplitude while the two sidebands reach 6.066 and 4.758 times their initial
amplitudes.  This is a genuinely focusing positive control rather than a
nearly linear trajectory.  The Benjamin--Feir support remains unchanged;
numerically failed proposals are recorded and replaced within the same one of
the 66 carrier/sideband categories.

### JONSWAP category-smoke support correction

The first 27-category revision-3 smoke used the old support: shallow cases
allowed \(\epsilon_p\leq0.15\), while finite and deep cases had no common
peak-steepness bound.  It attempted 38 trajectories, accepted 26, rejected
12, and exhausted the four-attempt allowance in the category
"shallow__gamma_1__right_1".  The four proposals in that category had
\(\epsilon_p=0.09394,0.13336,0.08293,0.14694\).  Their initial fields were
finite, above the bottom, and satisfied the linear energy construction to
roundoff, but every nonlinear adjustment failed.  More attempts could
eventually draw a milder proposal, but raising the cap would leave the
unwanted steep tail in the population law and hide the support-design problem
behind repeated numerical rejections.

Applying \(\epsilon_p\leq0.08\) retrospectively removes all ten adjustment
failures, one of the two later autonomous failures, and three previously
successful attempts.  It retains 23 successful attempts and one autonomous
failure.  This is the intended division of labor: the input bound removes the
unsupported steep tail, while the full-trajectory conditions still reject a
rare phase-dependent evolution.  A Miche-style inequality accepted all 38
attempts and was therefore too weak to diagnose this tail.

The implementation uses one bound in shallow, finite, and deep water.  In
shallow water it chooses \(n_p\) uniformly, proposes
\((k_ph,H_s/(2h))\) uniformly on the old rectangle, and repeats the pair until
the bound holds.  Thus, for each \(n_p\), the retained dimensionless pair is
uniform with respect to area on the clipped rectangle.  In finite and deep
water it repeats uniform \((k_p,h,H_s)\) proposals from the old box, so the
retained triple is uniform with respect to volume on the clipped box.  A
deterministic CPU check passes 405 boundary cases (225 shallow, 90 finite, and
90 deep) and 22 just-outside probes.  A fresh stream-932, four-attempt,
one-case-per-category GPU smoke is the remaining check before bulk random-sea
generation.

## Fixed parts of the revision-3 contract

1. **Keep construction support separate from numerical acceptance.**  Draw the
   declared parameter category and random phases before the solver runs.  Do
   not inspect a realized profile and then rewrite its parameters.
2. **Benjamin--Feir evolution.**  Zero-fill the constructed \(P_{128}\) state
   into a sharp internal band and evolve with
   \(K=256,M=4,\Delta t=0.01\), without a repeated spectral filter or nonlinear
   adjustment.  The BF initial condition already consists of a nonlinear
   carrier and prescribed sidebands.  For carrier period
   \(T_c=2\pi/\sqrt{gk_c}\), evolve through the last saved-grid time not
   exceeding \(100T_c\), using saved spacing \(0.08\), and retain 200
   approximately uniform saved-grid states including both endpoints.
3. **JONSWAP/TMA adjustment and evolution.**  Starting from the linear random
   sea with \(k_pH_s/2\leq0.08\) on 2048 internal points, use the sharp cutoff
   \(K_{\rm sea}=704\), with
   \(M=4\), padding factor eight, \(\Delta t=0.01\), saved spacing \(0.08\),
   stage-residual tolerance \(10^{-8}\), an iteration cap of five, and

   \[
   A(t)=1-\exp[-(t/(10T_p))^4].
   \]

   For each case, the adjustment endpoint is the last saved-grid time not
   exceeding \(20T_p\).  Preserve the complete internal-band endpoint
   \((\eta,\xi)\); do not project it to \(P_{128}\), add separate Stokes
   harmonics, or rescale one component.  Start the autonomous 16-\(T_p\)
   trajectory from that pair with a fresh clock \(t=0\), the ramp disabled,
   and a new Hamiltonian reference.
4. **Delivered learning value.**  For both families, deliver 1024-point
   \(P_{128}\) states and evaluate the fixed order-six reference target there.
   For JONSWAP/TMA, copy normalized Fourier coefficients of \(\eta\) and
   \(\xi\) with \(|k|\le128\) from the 2048-point internal state to the
   1024-point grid, reconstruct those two fields, and recompute the target.
   Never resample the internal \(G(\eta)\xi\).  Thus \(M=4\) defines the
   internal dynamics and \(M=6\) defines the label; they are not competing
   approximations of one operation.
5. **Required autonomous check.**  Require finite internal states and DNO
   values, successful implicit stages, positive water depth, and
   \(d_H\le10^{-3}\) over every saved autonomous state.  Persist the initial
   Hamiltonian, maximum drift, minimum water column, finiteness indicators,
   and acceptance masks.  Hamiltonian is not checked during JONSWAP adjustment
   because the ramp makes the equations explicitly time dependent.  Nor is
   the Hamiltonian recomputed after \(P_{128}\) projection used as the
   conservation test: energy can cross that projection boundary.
6. **Replacement, not silent deletion.**  If adjustment or autonomous
   evolution fails its required checks, record a zero-row attempt and draw the
   next deterministic attempt in the same category.  This retains the
   predeclared accepted-case quota for every category.
7. **No morphology gate.**  Sign counts, local-slope limits, and high-band
   fractions remain explanatory diagnostics.  They do not replace the fixed
   numerical method and its Hamiltonian check.

## Implementation and artifact status

The corrected accepted-quota proposal uses the population-record schema
`benjamin_feir_sample_spec_v2`; each standalone archive specification uses
the parameter-record schema `bf_jcp09_parameter_spec_v2`.  Both use the
constructor identifier `jcp09_equation_33_with_project_fifth_order_carrier_v2`.
Historical v1 archives are not eligible for resume or shard merge.  The
standalone writer records separate fingerprints for the complete run and for
the dataset construction shared by Modal shards.  Resume checks the immutable
configuration together with the unique archive members, their SHA-256
digests, array contracts, case specifications, and exact relation between the
last committed batch and the mutable progress counters.  Modal performs the
same checks on every shard, requires disjoint gap-free batch intervals, and
validates a staged merged archive before publishing it.  This bookkeeping does
not alter the accepted-quota paper generator; it prevents a historical or
partial BF archive from being mistaken for output of the corrected
construction.

That archive audit also exposed a separate defect in the historical adaptive
time selector.  If almost all activity was concentrated at the terminal
time, the forward-only collision repair could return the terminal index many
times.  For a synthetic 2501-time, 200-row example with the Modal weight
\(\alpha=0.85\), only 31 indices were distinct.  A reverse repair now shifts
the saturated indices left, so every requested index is in range and strictly
increasing while the two endpoints remain fixed.  The revision-3 paper BF
contract already uses 200 approximately uniform times, so this correction is
for the standalone adaptive path and does not change the paper time-selection
rule.

The production contracts now store separate evolution and target DNO orders.
Before projection, the executor computes compact saved-frame telemetry for
the internal Hamiltonian, state and DNO finiteness, and minimum water column;
the writer persists these quantities and the corresponding decision bits.
Benjamin--Feir and JONSWAP/TMA both have generator revision 3 and the common
\((K,M)=(128,6)\) delivered target.  Benjamin--Feir has fixed
\((N,K,M)=(1024,256,4)\) internal evolution.  JONSWAP/TMA has fixed
\((N,K,M)=(2048,704,4)\) internal evolution and delivers the common
1024-point target by normalized Fourier projection of \(\eta\) and \(\xi\),
followed by a fresh target evaluation.

The JONSWAP/TMA executor performs each case's own floored 20-\(T_p\)
adjustment, hands the full internal-band endpoint to the autonomous solver,
and restarts the clock.  Its adjustment duration, ramp parameters, numerical
microbatch policy, and source files are bound into the run fingerprint.  The
same adjustment is an explicit field of the immutable trajectory-execution
record.  Final-view preflight checks that record in both the completion
summary and every committed proposal, then validates the complete current
bucketing policy rather than only its ramp parameters.
Paper-dataset JONSWAP/TMA generation must use
`scripts/run_paper_dataset_jonswap_bucketed.py`; the general quota launcher and
base executor fail closed rather than permit an unadjusted paper run.

Before the dynamic category calculations, a deterministic CPU gate checks
the literal revision-3 BF and JONSWAP execution records, including the JONSWAP
adjustment parameters.  It constructs BF states through a durable proposal,
checks the implemented \(\eta\) and \(\xi\) against an independent evaluation
of the stated formula at two translations, and sends all 66 BF horizons
through the production time-grid dispatch and retained-row writer.  It also
checks all declared JONSWAP support boundaries and places the support-valid
all-\(\pi\) bottom-crossing witness in the middle of a three-case proposal.
The production recovery path rejects only that middle case and returns both
independently constructed siblings byte for byte.  This static gate passes;
it does not replace the outstanding full-horizon GPU trajectories.

The old Benjamin--Feir root
`outputs/paper_dataset_cap4_revision2_20260728/train/benjamin_feir/chunk_02048_02048`
is quarantined as historical revision-2 evidence.  Its eight committed
batches contain 2048 attempted cases, 1856 accepted cases, 192 rejected cases,
and 371200 rows.  The following 192-case proposal has no result or promoted
shard.  No manifest was built.  This partial root is neither resumable into nor
counted toward revision 3.

## Remaining calculations

- Run the revision-3 Benjamin--Feir smoke over all 66 declared categories.
- Run the adjusted revision-3 JONSWAP/TMA smoke over all 27 categories at the
  fixed 2048-point, \(K=704\) method.  The guarded launcher for both smokes is
  `scripts/launch_revision3_bf_jonswap_category_smokes.sh`; it uses GPU 0 and
  stream 930 for the 66 Benjamin--Feir cases, GPU 1 and stream 931 for the 27
  JONSWAP/TMA cases, a JONSWAP solver microbatch of eight, and fresh nondataset
  output roots.  It does not launch bulk generation.
- Report the 66-cell and 27-cell rejection rates and retained-parameter
  summaries before starting the full dataset.

The canonical and milder mode-9 Benjamin--Feir validation calculations are
complete.  The former is a negative project-specific admissibility result;
the latter is a resolved focusing positive control.  The broader population
must still be described as a project extension of the named JCP09 example.
