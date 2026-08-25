# Support and construction audit for the two coarse-arm failures

Date: 2026-07-25

**Later update.** This note was originally frozen while the
\(\Delta t=0.005\) arm was active. Section 10 now records the completed
time-halving decision and all one-factor restart arms. Those results supersede
only the statements below about what the coarse arm could not decide; the
static support and construction measurements are unchanged.

## 1. Question and scope

The production full-horizon panel uses the periodic interval
\([0,L)\), with

\[
L=2\pi,\qquad N=1024,\qquad g=1.
\]

Here \(N\) is the number of spatial grid points and \(g\) is gravitational
acceleration. The reference calculation uses the order-six
Craig--Sulem Dirichlet--Neumann expansion, pad factor eight, and the retained
Fourier band \(|k|\leq K\), where \(K=128\). Time integration uses the
fourth-order, two-stage Gauss--Legendre method, abbreviated GL2.

The coarse arm used time step \(\Delta t=0.01\) through the requested terminal
time \(T=200\). Five of the seven Tanaka and Benjamin--Feir cases completed
this arm, while the following two did not:

1. `tanaka_steep_upper_seam`;
2. `bf_jcp09_canonical`.

This note answers three questions.

1. Does each initial condition lie inside its stated parameter family?
2. Is each initial condition constructed and spatially resolved as intended?
3. What can, and what cannot, be inferred from failure of the
   \(\Delta t=0.01\) arm?

Only the immutable case manifest and completed coarse-arm artifact are used.
The active \(\Delta t=0.005\) arm was not inspected. No result from that arm is
anticipated below.

## 2. Definitions used in the audit

### 2.1 State, target, and Fourier projection

Let

\[
x_j=\frac{Lj}{N},\qquad j=0,\ldots,N-1,
\]

be the uniform periodic grid. A water-wave state consists of the surface
elevation \(\eta(x_j)\), surface potential \(\xi(x_j)\), and depth \(h>0\).
For the order-six, pad-eight discrete Dirichlet--Neumann operator
\(G_N^{(6,8)}(\eta;h)\), define the stored reference target

\[
q_{\mathrm{ref}}
=P_K^\circ\!\left[
G_N^{(6,8)}(P_K\eta;h)(P_K\xi)
\right].
\]

Here \(P_K\) is Fourier projection onto modes \(|k|\leq K\), and
\(P_K^\circ\) additionally removes mode zero. The spatial mean of \(\xi\) is
removed before this evaluation. Thus
\(q_{\mathrm{ref}}\) always denotes the delivered target defined by the
display above, including pad factor eight and the final fixed-band,
mean-zero projection. It is not a new physical variable.

For a real grid function \(f=(f_j)_{j=0}^{N-1}\), define its normalized
Fourier coefficients by

\[
\widehat f_k=\frac1N\sum_{j=0}^{N-1}
f_j\exp\!\left(-\frac{2\pi i j k}{N}\right).
\]

For integers \(0\leq a\leq b<N/2\), define the real-field band norm

\[
\|f\|_{[a,b]}^2
=
\mathbf 1_{\{a=0\}}|\widehat f_0|^2
+2\sum_{k=\max(1,a)}^b|\widehat f_k|^2.
\]

The high-band fraction reported below is

\[
R_f^{80:128}
=
\frac{\|f\|_{[80,128]}}{\|f\|_{[0,128]}}.
\]

Thus \(R_f^{80:128}\) measures how much of the retained Fourier norm lies
near the upper end of the delivered band. It is a diagnostic, not an
acceptance rule.

### 2.2 Spatial derivatives and water column

The derivative \(f_x\) is evaluated spectrally by multiplying
\(\widehat f_k\) by \(ik\). We report the grid maximum

\[
\|f_x\|_\infty=\max_j |f_x(x_j)|.
\]

The minimum water column and its depth-normalized version are

\[
w_{\min}(t)=\min_j\{h+\eta(x_j,t)\},
\qquad
\widetilde w_{\min}(t)=\frac{w_{\min}(t)}h.
\]

A positive value means that the computed surface remains above the flat
bottom. Positivity does not, by itself, establish that the numerical
calculation is accurate. In the refinement decision, *graph-valued* means
that \(\eta\) remains a finite, single-valued periodic grid function and
\(h+\eta(x_j)>0\) at every grid point.

### 2.3 Fixed-band grid and collocation discrepancies

Suppose the same parameter case is independently constructed at spatial
sizes \(N_r\) and \(N_s\). For a field \(f\), define the fixed-band relative
discrepancy

\[
\delta_{r,s}(f)
=
\frac{\|P_K f_{N_r}-P_K f_{N_s}\|_{[0,K]}}
     {\|P_K f_{N_s}\|_{[0,K]}}.
\]

The three-grid discrepancy is the largest value among the pairs
\((1024,2048)\), \((2048,4096)\), and \((1024,4096)\). The Tanaka audit uses
budgets \(3\times10^{-4}\) for \(\eta,\xi\) and \(10^{-3}\) for
\(q_{\mathrm{ref}}\).

The Tanaka profile is obtained from a nonlinear collocation solve. Its
collocation discrepancy compares an otherwise identical construction using
257 profile knots with one using 513 profile knots. The reported full-band
quantity uses modes \(0,\ldots,128\). Its budget is \(2\times10^{-3}\).

### 2.4 Traveling-wave residual for a Tanaka component

Let \(c\) be the signed Tanaka wave speed. An exact traveling wave satisfies

\[
G(\eta;h)\xi=-c\,\eta_x.
\]

The static audit therefore defines

\[
R_{\mathrm{trav}}
=
\frac{\|q_{\mathrm{ref}}+c\eta_x\|_{\mathrm{rms}}}
     {\|c\eta_x\|_{\mathrm{rms}}},
\qquad
\|f\|_{\mathrm{rms}}
=
\left(\frac1N\sum_{j=0}^{N-1}|f_j|^2\right)^{1/2}.
\]

The periodic Tanaka constructor first obtains a candidate derivative
\(\xi_x^{\mathrm{cand}}\) from the steady Bernoulli relation and then removes
its mean so that a periodic antiderivative exists. The relative size of this
removed mean is

\[
R_{\mathrm{mean}}
=
\frac{|\operatorname{mean}(\xi_x^{\mathrm{cand}})|}
     {\|\xi_x^{\mathrm{cand}}\|_{\mathrm{rms}}}.
\]

Neither \(R_{\mathrm{trav}}\) nor \(R_{\mathrm{mean}}\) was used as an
absolute row-rejection rule. They quantify how far the periodic construction
is from the isolated traveling-wave identity.

### 2.5 Hamiltonian drift

At a saved time \(t\), define the discrete Hamiltonian

\[
H(t)
=
\frac{\Delta x}{2}\sum_{j=0}^{N-1}
\left[\xi(x_j,t)q_{\mathrm{ref}}(x_j,t)
+g\eta(x_j,t)^2\right],
\qquad
\Delta x=\frac LN.
\]

Its relative drift from the initial value is

\[
D_H(t)=\frac{|H(t)-H(0)|}{|H(0)|}.
\]

This is the same discrete Hamiltonian obtained from the unprojected internal
DNO value. Indeed, write that value as \(\widetilde q\). Because \(\xi\) is
mean-zero and \(P_K\)-bandlimited and because \(P_K^\circ\) is an orthogonal
Fourier projection,

\[
\langle \xi,P_K^\circ\widetilde q\rangle
=\langle P_K^\circ\xi,\widetilde q\rangle
=\langle \xi,\widetilde q\rangle.
\]

All initial Hamiltonians in this audit are nonzero, so no artificial
denominator floor is needed for the reported values.

### 2.6 GL2 stage residual

Let \(S_i^u\) be the current GL2 stage iterate and let
\(\Phi_i^u\) be the corresponding stage-map value, where
\(i\in\{1,2\}\) denotes the stage and \(u\in\{\eta,\xi\}\) denotes the
physical component. The recorded dimensionless residual is

\[
r_{\mathrm{GL}}
=
\max_{\substack{i\in\{1,2\}\\u\in\{\eta,\xi\}}}
\frac{\|\Phi_i^u-S_i^u\|_2}
     {\max\{\|\Phi_i^u\|_2,\|S_i^u\|_2,\mathrm{tiny}\}}.
\]

The production tolerance is \(10^{-8}\), and at most eight Picard updates
are permitted. A step is called converged only when all stage and returned
state values are finite and \(r_{\mathrm{GL}}\leq10^{-8}\).

## 3. Exact observations: upper-seam Tanaka case

### 3.1 Parameter support

The case has one right-moving crest with

\[
h=0.35,\qquad
\alpha=\frac ah=0.45,\qquad
x_0=0.
\]

Here \(a\) is the physical crest-height parameter, \(\alpha\) is the
dimensionless amplitude ratio, and \(x_0\) is the periodic crest center.
The stated steep single-crest stratum is

\[
h\in[0.20,0.35],
\qquad
\alpha\in[0.25,0.45].
\]

The case is therefore exactly at both upper endpoints. Its intended physical
crest height is

\[
a=h\alpha=0.1575,
\]

and the projected grid state has
\(\max_j\eta(x_j,0)=0.157500048238\).

The active case definition uses the identifier
`periodic_tangent_hermite_tan_theta_v1`. The constructor:

1. solves the dimensionless, unit-depth Tanaka problem;
2. rescales horizontal position and elevation by \(h\);
3. reconstructs each profile interval by the cubic Hermite polynomial whose
   nodal slope is \(\tan\theta\), where \(\theta\) is the angle returned by
   the Tanaka solve;
4. sums every translated compact-support image that intersects the periodic
   interval;
5. evaluates the result on an \(8N\) grid and restricts it in Fourier space;
6. constructs the periodic surface potential; and
7. applies the common projection \(P_{128}\).

The center \(x_0=0\) does not introduce a seam defect. Comparing this state
with its exact half-period translation gives relative differences

\[
1.54\times10^{-16}\quad(\eta),\qquad
2.17\times10^{-16}\quad(\xi),\qquad
3.51\times10^{-11}\quad(q_{\mathrm{ref}}).
\]

### 3.2 Initial spatial diagnostics

The independently constructed three-grid discrepancies are

| field \(f\) | \(\max_{r<s}\delta_{r,s}(f)\) | audit budget |
|---|---:|---:|
| \(\eta\) | \(2.92154\times10^{-11}\) | \(3\times10^{-4}\) |
| \(\xi\) | \(6.74071\times10^{-11}\) | \(3\times10^{-4}\) |
| \(q_{\mathrm{ref}}\) | \(6.38433\times10^{-10}\) | \(10^{-3}\) |

The 257-versus-513-knot full-band discrepancies are

| field \(f\) | collocation discrepancy | audit budget |
|---|---:|---:|
| \(\eta\) | \(3.33774\times10^{-4}\) | \(2\times10^{-3}\) |
| \(\xi\) | \(2.71911\times10^{-4}\) | \(2\times10^{-3}\) |
| \(q_{\mathrm{ref}}\) | \(2.73635\times10^{-4}\) | \(2\times10^{-3}\) |

The actual \(N=1024\) initial state has

\[
\|\eta_x(0)\|_\infty=0.181737,\qquad
\|q_{\mathrm{ref}}(0)\|_\infty=0.119383,
\]

\[
R_{q_{\mathrm{ref}}}^{80:128}(0)=2.37054\times10^{-6},\qquad
w_{\min}(0)=0.350225.
\]

For a component with signed speed \(c\), the surface-potential construction
uses the square-root argument

\[
\mathcal R(x)
=\bigl(1+\eta_x(x)^2\bigr)\bigl(c^2-2g\eta(x)\bigr).
\]

Before the current implementation applies its zero clamp, the upper-seam
case satisfies

\[
\min_x\frac{\mathcal R(x)}{c^2}=0.372151>0.
\]

The clamp therefore does not alter this audited case. It is nevertheless
wrong as a production support policy: a future case with
\(\min_x\mathcal R(x)<0\) has no real value under this construction and must
be recorded as rejected rather than silently changed.

Thus the initial target has negligible retained-band mass near modes
80--128. For comparison, the successful narrow Tanaka case has
\(R_{q_{\mathrm{ref}}}^{80:128}(0)=0.551061\), over five orders of magnitude larger.

The closest successful broad Tanaka control is `tanaka_main_upper_corner`,
with \(h=0.30\) and \(\alpha=0.35\). Its three-grid target discrepancy is
\(4.84598\times10^{-10}\), and its target collocation discrepancy is
\(2.72757\times10^{-4}\). These are the same scale as the upper-seam values.

### 3.3 Absolute traveling-wave caveat

For the upper-seam case,

\[
R_{\mathrm{trav}}=0.0726833,\qquad
R_{\mathrm{mean}}=0.550899.
\]

The successful broad control has

\[
R_{\mathrm{trav}}=0.0575197,\qquad
R_{\mathrm{mean}}=0.543931.
\]

The upper-seam static audit passed because tangent-Hermite reconstruction did
not worsen the corresponding piecewise-linear control by more than the
declared allowance. That test did not assert that
\(R_{\mathrm{trav}}\) was small in an absolute sense.

This distinction is important: the case is correctly constructed as a
periodic Tanaka-profile initial condition, but it is not an exact periodic
traveling wave of the discrete order-six equations.

### 3.4 Coarse-arm terminal behavior

The first failed GL2 step starts at \(t=119.48\). Its stage values are
nonfinite. The preceding step at \(t=119.47\) is finite and satisfies

\[
r_{\mathrm{GL}}=5.93028\times10^{-9}<10^{-8}.
\]

The last completely finite saved frame is at \(t=119.44\). Between \(t=0\)
and this frame,

| diagnostic | \(t=0\) | \(t=119.44\) |
|---|---:|---:|
| \(\|\eta_x\|_\infty\) | \(0.181737\) | \(1.08157\) |
| \(\|\partial_xq_{\mathrm{ref}}\|_\infty\) | \(0.590830\) | \(13.1181\) |
| \(R_{q_{\mathrm{ref}}}^{80:128}\) | \(2.37054\times10^{-6}\) | \(0.538060\) |
| \(w_{\min}\) | \(0.350225\) | \(0.345167\) |
| \(D_H\) | \(0\) | \(0.0768268\) |

The minimum water column at the last finite frame is
\(\widetilde w_{\min}=0.986193\), so bottom contact is not the event.
Hamiltonian drift first exceeds \(10^{-3}\) at \(t=115.28\) and \(10^{-2}\)
at \(t=118.00\).

These values show that substantial gradients and high-band target content
are generated dynamically before stage failure. They are not present in the
initial state.

## 4. Exact observations: Xu--Guyenne diagnostic Benjamin--Feir case

### 4.1 Parameter support and formula

Let \(n_c\) be the carrier mode, let \(\Delta n\) be the integer sideband
offset, let \(\epsilon_c=k_ca\) be the prescribed carrier steepness, let
\(\rho\) be the ratio of each sideband amplitude to the leading carrier
amplitude, and let \(\phi\) be their common phase. On \(L=2\pi\), define

\[
k_c=n_c,\qquad
k_\pm=n_c\pm\Delta n,\qquad
a=\frac{\epsilon_c}{k_c}.
\]

The case parameters are

\[
n_c=9,\qquad
\Delta n=2,\qquad
\epsilon_c=0.13,\qquad
\rho=0.10,\qquad
\phi=-\frac{\pi}{4}.
\]

The sideband modes are consequently \(k_-=7\) and \(k_+=11\). The declared
family support is

\[
4\leq n_c\leq20,\qquad
0.05\leq\epsilon_c\leq0.13,\qquad
0.05\leq\rho\leq0.20,\qquad
1\leq\Delta n<n_c,
\]

together with the leading deep-water instability condition

\[
0<\beta<1,\qquad
\beta=
\frac{\Delta n/n_c}{2\sqrt2\,\epsilon_c}.
\]

For this case,

\[
\beta=0.604364770245.
\]

The carrier steepness is at its inclusive upper endpoint, but the case is
not close to the instability-band boundary \(\beta=1\). The phase
\(-\pi/4\) is valid because phase is defined modulo \(2\pi\).

The numerical depth is

\[
h=\frac{5L}{2\pi}=5.
\]

Thus the carrier and sideband depth parameters are

\[
k_ch=45,\qquad k_-h=35,\qquad k_+h=55.
\]

The leading carrier amplitude and each sideband amplitude are

\[
a=\frac{0.13}{9}=0.0144444444\ldots,
\qquad
\rho a=0.0014444444\ldots.
\]

Let \((\eta_c,\xi_c)\) denote the project's analytic fifth-order deep-water
carrier whose first elevation harmonic has amplitude \(a\). The implemented
initial condition is

\[
\eta
=
\eta_c+\rho a
\left[\cos(k_-x+\phi)+\cos(k_+x+\phi)\right],
\]

\[
\xi
=
\xi_c+\rho a
\left[
\sqrt{\frac g{k_-}}e^{k_-\eta}\sin(k_-x+\phi)
+
\sqrt{\frac g{k_+}}e^{k_+\eta}\sin(k_+x+\phi)
\right],
\]

followed by removal of the mean of \(\xi\). The two sideband coefficients are
different, and the implementation contains neither empirical cross terms nor
an exceptional rewrite of the mode pair.

The focused formula test reconstructs this expression independently and
agrees with the constructor to an absolute and relative tolerance of
\(2\times10^{-14}\). All six focused Benjamin--Feir tests pass.

This named case uses the parameter values in equation (33) of Xu and Guyenne,
but its carrier is not identical to the numerically exact steady carrier used
in that paper. It is equation-(33) sidebands paired with the project's stated
analytic fifth-order carrier.

### 4.2 Initial spatial diagnostics

The following table compares the failed equation-(33) stress case with the two
Benjamin--Feir cases that completed the coarse arm. The quantity
\(\Delta_{q_{\mathrm{ref}}}^{512:2048}\) is the largest fixed-\(P_{128}\) target discrepancy
among grids 512, 1024, and 2048.

| case | coarse status | \(\|q_{\mathrm{ref}}\|_{\mathrm{rms}}\) | \(\|q_{\mathrm{ref}}\|_\infty\) | \(\|\eta_x\|_\infty\) | \(R_{q_{\mathrm{ref}}}^{80:128}\) | \(\Delta_{q_{\mathrm{ref}}}^{512:2048}\) |
|---|---|---:|---:|---:|---:|---:|
| equation-(33) \(9;2\) stress case | failed | \(0.0315034\) | \(0.0514777\) | \(0.154137\) | \(8.99144\times10^{-8}\) | \(1.19821\times10^{-14}\) |
| high-mode boundary \(20;7\) | complete | \(0.0207288\) | \(0.0376251\) | \(0.168775\) | \(2.80108\times10^{-3}\) | \(2.08054\times10^{-14}\) |
| low-carrier boundary \(4;1\) | complete | \(0.0327311\) | \(0.0543795\) | \(0.109473\) | \(6.63545\times10^{-15}\) | \(8.31871\times10^{-15}\) |

The equation-(33) stress-case initial target is therefore neither unusually large nor
unresolved compared with the successful cases. Replacing the proxy depth
\(h=5\) by \(h=20\) changes its initial target by only

\[
\frac{\|q_{\mathrm{ref},h=5}-q_{\mathrm{ref},h=20}\|_2}
{\|q_{\mathrm{ref},h=20}\|_2}
=8.25344\times10^{-12}.
\]

### 4.3 Relation to the published diagnostic calculation

Xu and Guyenne used \(M=4\), \(N=64\), \(\Delta t=0.01\), and no filtering.
Here \(M\) is the DNO-series order. They report the first carrier minimum
near 60 carrier periods and eventual computation breakdown after about
716 carrier periods.

An earlier repository control used the same analytic fifth-order carrier as
the current family and the literal equation-(33) sidebands at \(N=64\). It
remained finite through 67 carrier periods and reached its first carrier
minimum at 56.827 periods.

For comparison, define the approximate nonlinear carrier frequency and
period by

\[
\omega_c
=
\sqrt{k_c}
\left(1+\frac{\epsilon_c^2}{2}
+\frac{5\epsilon_c^4}{8}\right),
\qquad
T_c=\frac{2\pi}{\omega_c}.
\]

For these parameters, \(T_c=2.07647820\). The active coarse failure
at \(t=108.77\) therefore occurs at \(t/T_c=52.3820\).

This comparison does not make the current and published calculations
identical: they use different DNO orders, spatial bands, filters, and carrier
representations. It does show that the parameter tuple and equation-(33)
sidebands are standard, meaningful diagnostic data rather than an accidental
out-of-support draw.

### 4.4 Coarse-arm terminal behavior

The first failed step starts at \(t=108.77\). Its stage and returned state are
finite, but eight Picard updates leave

\[
r_{\mathrm{GL}}=3.89314\times10^{-7}>10^{-8}.
\]

The preceding step at \(t=108.76\) converges with
\(r_{\mathrm{GL}}=2.52312\times10^{-9}\).

The last completely finite saved frame is at \(t=108.72\). Between \(t=0\)
and this frame,

| diagnostic | \(t=0\) | \(t=108.72\) |
|---|---:|---:|
| \(\|\eta_x\|_\infty\) | \(0.154137\) | \(1.56940\) |
| \(\|\partial_xq_{\mathrm{ref}}\|_\infty\) | \(0.610319\) | \(29.0545\) |
| \(R_{q_{\mathrm{ref}}}^{80:128}\) | \(8.99144\times10^{-8}\) | \(0.372267\) |
| \(w_{\min}\) | \(4.98423\) | \(4.96623\) |
| \(D_H\) | \(0\) | \(0.0157893\) |

Hamiltonian drift first exceeds \(10^{-3}\) at \(t=106.88\). The water
column remains far from zero. As for the Tanaka case, the terminal event is
preceded by dynamically generated gradients and high-band content rather
than by a poorly resolved initial target.

## 5. Comparison with all five coarse-arm completions

For context, the endpoint diagnostics of the five completed cases are:

| case | endpoint | \(\|\eta_x\|_\infty\) | \(\|\partial_xq_{\mathrm{ref}}\|_\infty\) | \(R_{q_{\mathrm{ref}}}^{80:128}\) | maximum \(D_H\) |
|---|---:|---:|---:|---:|---:|
| narrow Tanaka | \(200\) | \(0.101155\) | \(0.919406\) | \(0.495536\) | \(1.00400\times10^{-8}\) |
| broad Tanaka | \(200\) | \(0.127794\) | \(0.363122\) | \(0.00262429\) | \(5.76513\times10^{-8}\) |
| Tanaka case 31 | \(200\) | \(0.00933154\) | \(0.0105956\) | \(7.77808\times10^{-6}\) | \(2.36890\times10^{-11}\) |
| high-mode BF boundary | \(200\) | \(0.198994\) | \(1.53460\) | \(0.0112358\) | \(1.43427\times10^{-7}\) |
| low-carrier BF boundary | \(200\) | \(0.150506\) | \(0.423335\) | \(3.53475\times10^{-13}\) | \(3.08924\times10^{-8}\) |

The completed narrow Tanaka case demonstrates that a large high-band
fraction by itself is not sufficient to predict failure. The failed cases
differ more clearly in the simultaneous growth of derivative magnitude,
high-band content, Hamiltonian drift, and GL2 stage difficulty.

## 6. Interpretation, kept separate from the observations

### 6.1 What the observations rule out

The measurements do not support any of the following diagnoses.

- The Tanaka failure is caused by placing a crest on the periodic seam.
- Either failed case begins with an unresolved \(P_{128}\) target.
- The Benjamin--Feir tuple is outside the declared instability family.
- Bottom contact causes either coarse failure.
- Either failure is already present in the initial state.

The Tanaka seam symmetry, independent spatial constructions, collocation
comparison, and Benjamin--Feir formula test directly contradict these
explanations.

### 6.2 What the coarse arm does not decide

The coarse arm alone does not distinguish among:

1. an insufficient time step for a still-resolvable trajectory;
2. dynamically generated structure that is too demanding for the frozen
   \(M=6\), \(K=128\) reference representation;
3. a parameter point that belongs to the static family but is too aggressive
   for a complete, numerically accepted trajectory through \(T=200\).

This is especially important for the upper Tanaka point. It is inside the
declared stress stratum, but it is not an exact periodic traveling wave and
has a non-negligible absolute traveling-wave residual before integration.

### 6.3 Predeclared decision rule

The completed refinement calculations should be interpreted as follows.

- If the \(\Delta t=0.005\) trajectory is finite, graph-valued, and
  stage-converged, and the predeclared
  \(0.005\)-versus-\(0.0025\) comparison is at most \(10^{-3}\), then the
  \(\Delta t=0.01\) failure is evidence that the coarse time step was
  insufficient.
- If the \(\Delta t=0.005\) trajectory fails in the same neighborhood, the
  attempted complete trajectory should be rejected for this reference
  contract and horizon. That result would not make the initial condition
  malformed. It would instead motivate either reducing the dynamic support
  or changing and revalidating the frozen reference discretization.

No finite prefix should be retained as an accepted trajectory.

## 7. Post-run code follow-ups and completed document correction

The code changes should wait until the active run has finished because the
runner source participates in its configuration fingerprint.

1. **Enforce the stated support in the runner.** The current Tanaka predicate
   checks only positive amplitude and positive depth. It should also check
   the main or steep depth interval, amplitude-ratio interval, crest count,
   and reconstruction identifier. The Benjamin--Feir predicate checks modes,
   steepness, amplitude ratio, and instability fraction, but the runner
   should additionally check that the phase is finite and that the case depth
   equals the declared deep-water proxy depth.
2. **Hash all files that determine the initial arrays.** The current source
   fingerprint includes `generate_tanaka_dataset_v2.py` but omits the
   indirectly imported Tanaka solver and multi-crest specification modules.
   Add `solver/tanaka_ICs/modified_tanaka.py` and
   `solver/gen_data/multi_crest.py`.
3. **Completed document correction.** The plan and manuscript now describe
   the implemented sum over every compact-support image that intersects the
   periodic interval, rather than a fixed number of images.

## 8. Reproduction commands

All commands below are read-only and run from the repository root.

### 8.1 Verify that the active source hashes match the recorded run

```bash
jq '{fingerprint:.config_fingerprint, source:.source.files}' \
  outputs/full_horizon_refinement_panel_20260725/summary.json

sha256sum \
  scripts/run_full_horizon_refinement_panel.py \
  solver/gen_data/generate_tanaka_dataset_v2.py \
  solver/gen_data/benjamin_feir_jcp09.py \
  solver/data/stokes_truth_jax.py \
  solver/solvers/dno_series_jax.py \
  solver/solvers/time_integrator.py \
  solver/gen_data/jonswap_tma.py \
  solver/gen_data/pipeline/acceptance.py
```

### 8.2 Re-run the focused Benjamin--Feir formula and support tests on CPU

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=True CUDA_VISIBLE_DEVICES='' \
  uv run python -m unittest solver.gen_data.tests.test_benjamin_feir_jcp09
```

The recorded result is six passing tests.

### 8.3 Extract the Tanaka static evidence

```bash
jq '
  {
    case: [.cases[] |
      select(.name=="steep_upper_seam" or .name=="main_upper_corner")],
    grid: [.static.grid_consistency[] |
      select(.case=="steep_upper_seam" or .case=="main_upper_corner")],
    collocation: [.static.collocation_consistency[] |
      select(.case=="steep_upper_seam" or .case=="main_upper_corner")],
    physics: [.static.component_physics[] |
      select(.case=="steep_upper_seam" or .case=="main_upper_corner")],
    symmetry: .static.symmetry
  }
' outputs/tanaka_tangent_stratified_panel_20260723/static_summary.json
```

### 8.4 Recompute the coarse-arm field, derivative, band, and Hamiltonian values

```bash
uv run python - <<'PY'
import numpy as np

path = (
    "outputs/full_horizon_refinement_panel_20260725/"
    "tanaka_benjamin_feir_dt_0p010.npz"
)
with np.load(path, allow_pickle=False) as archive:
    case_ids = archive["case_id"]
    times = archive["times"]
    depths = archive["depth"]
    eta = archive["eta"]
    xi = archive["xi"]
    q_ref = archive["gxi"]
    residual = archive["gl2_stage_residual"]
    converged = archive["gl2_converged"]
    step_times = archive["gl2_step_times"]

nx = eta.shape[-1]
length = 2.0 * np.pi
wavenumbers = 2.0 * np.pi * np.fft.rfftfreq(nx, d=length / nx)


def spectral_derivative(field: np.ndarray) -> np.ndarray:
    return np.fft.irfft(
        1j * wavenumbers * np.fft.rfft(field),
        n=nx,
    )


def band_norm(field: np.ndarray, lower: int, upper: int) -> float:
    coefficients = np.fft.rfft(field) / nx
    weights = np.full(coefficients.size, 2.0)
    weights[0] = 1.0
    weights[-1] = 1.0
    selected = (wavenumbers >= lower) & (wavenumbers <= upper)
    return float(
        np.sqrt(
            np.sum(weights[selected] * np.abs(coefficients[selected]) ** 2)
        )
    )


hamiltonian = (
    0.5 * length / nx * np.sum(xi * q_ref + eta * eta, axis=-1)
)

for case_index, case_id in enumerate(case_ids):
    finite_saved = (
        np.isfinite(eta[:, case_index]).all(axis=1)
        & np.isfinite(xi[:, case_index]).all(axis=1)
        & np.isfinite(q_ref[:, case_index]).all(axis=1)
    )
    first_nonfinite = np.flatnonzero(~finite_saved)
    last = int(first_nonfinite[0] - 1) if first_nonfinite.size else -1
    first_failed_step = np.flatnonzero(~converged[:, case_index])
    first_failed_step = (
        int(first_failed_step[0]) if first_failed_step.size else -1
    )
    h0 = hamiltonian[0, case_index]
    drift = np.abs(hamiltonian[:, case_index] - h0) / abs(h0)

    for label, frame in (("initial", 0), ("last_finite", last)):
        eta_x = spectral_derivative(eta[frame, case_index])
        q_ref_x = spectral_derivative(q_ref[frame, case_index])
        high_fraction = (
            band_norm(q_ref[frame, case_index], 80, 128)
            / band_norm(q_ref[frame, case_index], 0, 128)
        )
        print(
            case_id,
            label,
            "time",
            times[frame],
            "max_abs_eta_x",
            np.max(np.abs(eta_x)),
            "max_abs_q_ref_x",
            np.max(np.abs(q_ref_x)),
            "q_ref_high_fraction",
            high_fraction,
            "minimum_water",
            np.min(depths[case_index] + eta[frame, case_index]),
            "hamiltonian_drift",
            drift[frame],
        )

    if first_failed_step >= 0:
        print(
            case_id,
            "first_failed_step",
            first_failed_step,
            "time",
            step_times[first_failed_step],
            "residual",
            residual[first_failed_step, case_index],
            "previous_residual",
            residual[first_failed_step - 1, case_index],
        )
PY
```

### 8.5 Extract the relevant Xu--Guyenne PDF passage without OCR

```bash
uv run python - <<'PY'
from pypdf import PdfReader

reader = PdfReader("JCP09.pdf")
for page_number in (12, 13, 14):
    print(f"\nPAGE {page_number}\n")
    print(reader.pages[page_number - 1].extract_text())
PY
```

Pages 12--14 contain equation (33), the numerical settings, the first
focusing time, and the authors' discussion of eventual high-wavenumber
breakdown.

## 9. Evidence files

The numerical values in this note come from:

- `outputs/full_horizon_refinement_panel_20260725/case_manifest.npz`,
  which stores the exact initial arrays;
- `outputs/full_horizon_refinement_panel_20260725/`
  `tanaka_benjamin_feir_dt_0p010.npz`, which stores the completed coarse arm
  and all per-step GL2 telemetry;
- `outputs/tanaka_tangent_stratified_panel_20260723/static_summary.json`,
  which stores the independent Tanaka grid, collocation, symmetry, and
  traveling-wave checks;
- `outputs/bf_jcp09_audit_20260723/summary.json`, which stores the earlier
  equation-(33) sideband comparison;
- `outputs/precascade_spatial_order_diagnostic_20260725/summary.json`, which
  stores the one-factor restart results; and
- `JCP09.pdf`, the Xu--Guyenne reference.

## 10. Completed time-halving decision and spatial restart update

The \(\Delta t=0.005\) arm reproduces both failures. The steep upper-seam
Tanaka case first fails at \(t=119.48\) at both step sizes. The
equation-(33) Benjamin--Feir stress case first fails at \(t=108.77\) for
\(\Delta t=0.01\) and \(t=108.78\) for \(\Delta t=0.005\). In each case the
fine trajectory is incomplete and has unsolved or nonfinite GL2 stages, so the
predeclared \(0.0025\) retry is ineligible. Both attempted complete
trajectories are rejected. This rules out ordinary time-step truncation
between \(0.01\) and \(0.005\) as the explanation.

The one-factor diagnosis restarts the saved coarse state before the cascade:
\(t_r=80\) for Tanaka and \(t_r=88\) for Benjamin--Feir. Let

\[
  D=\{1,\ldots,128\},\qquad B=\{80,\ldots,128\},
\]

and, for a set of positive Fourier modes \(I\), let

\[
  E_I(q,t)=\sum_{k\in I}|\widehat q_k(t)|^2.
\]

The source guard and the restart comparison have different reference times,
so we distinguish them.  The source guard is

\[
  S_B(q,t_r)=
  \frac{\max\{E_B(q,t_r)-E_B(q,0),0\}}{E_D(q,0)}.
\]

For a continuation that begins at \(t_r\), its subsequent old-band growth is

\[
  R_B(q,t;t_r)=
  \frac{\max\{E_B(q,t)-E_B(q,t_r),0\}}{E_D(q,t_r)},
  \qquad t\geq t_r.
\]

Both quantities measure an increase near the old cutoff, divided by the
nonzero delivered-band energy at the corresponding reference time.  The
Tanaka source satisfies
\(S_B(q_{\mathrm{ref}},80)=4.649\times10^{-5}<10^{-4}\). The completed
Tanaka arms are listed below. In the table, \(p\) is the product-padding
factor and \(M\) is the Craig--Sulem series order.

| arm | \((N,p,K,M)\) | first failed stage | result through \(t=122\) |
|---|---:|---:|---|
| baseline | \((1024,8,128,6)\) | \(119.48\) | nonfinite |
| doubled grid | \((2048,8,128,6)\) | none | finite; all stages solved |
| doubled padding | \((1024,16,128,6)\) | \(119.48\) | nonfinite |
| enlarged cutoff | \((1024,8,192,6)\) | none | finite; all stages solved |
| order five | \((1024,8,128,5)\) | none | finite; all stages solved |
| order four | \((1024,8,128,4)\) | none | finite; all stages solved |

The baseline, doubled-grid, doubled-padding, order-five, and order-four arms
cross the recorded high-band thresholds at essentially the same saved times.
Doubling product padding therefore does not change the failure. Doubling the
base grid prevents the terminal nonfinite state over this short continuation
but leaves \(\max_t R_B=0.562\). Reducing the series order also prevents the
terminal failure, but leaves \(\max_t R_B=0.463\) for \(M=5\) and \(0.390\)
for \(M=4\). Hence the order-five and order-four controls do not remove the
preceding pile-up near \(K=128\).

In the Tanaka \(K=192\) arm, however, \(R_B(q_{\rm ref},t;80)\) remains below
\(3.20\times10^{-5}\) and never crosses \(10^{-4}\). Let
\(A=\{129,\ldots,192\}\). The largest added-band ratio
\(\max_t E_A(q_{\rm ref},t)/E_D(q_{\rm ref},80)\) is
\(2.80\times10^{-4}\), and the maximum GL2 residual is
\(6.67\times10^{-9}\). This is evidence of crowding against the old
\(K=128\) boundary in the Tanaka continuation.

The completed Benjamin--Feir arms give a different result:

| arm | \((N,p,K,M)\) | first failed stage | result through \(t=112\) |
|---|---:|---:|---|
| baseline | \((1024,8,128,6)\) | \(108.78\) | incomplete |
| doubled grid | \((2048,8,128,6)\) | \(109.175\) | incomplete |
| doubled padding | \((1024,16,128,6)\) | \(108.78\) | incomplete |
| enlarged cutoff | \((1024,8,192,6)\) | \(103.755\) | incomplete |
| order five | \((1024,8,128,5)\) | none | finite; all stages solved |
| order four | \((1024,8,128,4)\) | none | finite; all stages solved |

The baseline, doubled-grid, and doubled-padding arms cross
\(R_B=10^{-4},10^{-3},10^{-2}\), and \(10^{-1}\) at the same saved times to
the precision recorded by the run. The \(M=5\) and \(M=4\) arms also cross
the first three of those thresholds at the same times, and their maximum
restart growth is \(0.182\) and \(0.283\), respectively. Lowering the order
therefore prevents the terminal stage failure without removing the preceding
growth.

Enlarging the cutoff does not cure this case and instead produces an earlier
failed stage. Before that failure, the maximum added-band ratio
\(E_A(q_{\rm ref},t)/E_D(q_{\rm ref},88)\) is \(4.82\times10^{-2}\), about
172 times the corresponding Tanaka maximum. Therefore \(K=192\) is not a
common remedy for the two families.

The baseline source archive recorded in the summary has configuration
fingerprint
`91eb6c187fd079030fef08b8963b50b1c237d3e9d8940c172777a806fb997b5e`
and script SHA-256
`98d67a44b5c23a76e0e0ae08d78dd119aaa2c6c5c984fd97463ced5d3ee999a2`.
The completed summary has file SHA-256
`84c890138fbcef3cb345d083facc4c07a9d816f5508a1335481b7f41d1529948`.
Neither restart is a complete-horizon \(K=192\) calculation. The current
\(N=1024,K=128\) complete trajectories remain rejected regardless of these
diagnostic outcomes.
