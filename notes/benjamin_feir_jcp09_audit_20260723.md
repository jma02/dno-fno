# Benjamin--Feir data audit against Xu--Guyenne (JCP 2009)

Date: 2026-07-23

## Question

Does the Benjamin--Feir data generator implement the initial condition and
time integration used in Section 4.2.3 of Xu and Guyenne (2009), and can a
construction correction analogous to the Tanaka Hermite correction remove the
nonfinite trajectories?

## Reference construction

Let the spatial period be \(L=2\pi\), let \(g=1\), and write
\[
    k_n = \frac{2\pi n}{L}.
\]
Xu and Guyenne use carrier mode \(n_c=9\), sideband modes
\(n_-=7\) and \(n_+=11\), carrier steepness \(k_c a=0.13\), and
relative sideband amplitude \(\rho=0.1\). Their initial condition is
\[
  \eta(x,0)
  =
  \eta_c(x)
  +
  \rho a\left[
      \cos(k_-x-\pi/4)+\cos(k_+x-\pi/4)
  \right],
\]
\[
  \xi(x,0)
  =
  \xi_c(x)
  +
  \rho a\left[
      k_-^{-1/2}e^{k_-\eta(x,0)}\sin(k_-x-\pi/4)
      +
      k_+^{-1/2}e^{k_+\eta(x,0)}\sin(k_+x-\pi/4)
  \right].
\]
Here \(\eta_c\) and \(\xi_c\) are the surface elevation and surface
potential of a numerically computed deep-water Stokes wave. The factor
\(k_\pm^{-1/2}\) is different for the two sidebands.

The paper uses a fourth-order Gauss--Legendre method with
\(\Delta t=0.01\), a fourth-order Dirichlet--Neumann expansion, and
\(N=64\) points. It reports the first carrier minimum near 60 carrier
periods.

## Differences in the generator audited here

This section describes the now-retired standalone Benjamin--Feir generator,
not the revision-4 paper-dataset implementation that replaced it.

1. The carrier and sidebands use deep-water formulas even when the generated
   rollout has finite depth. The initial-condition function does not receive
   the sampled depth.

2. Both sideband potentials use the carrier coefficient
   \(\sqrt{g/k_c}/k_c=1/\sqrt{k_c}\), rather than the individual
   coefficients \(1/\sqrt{k_-}\) and \(1/\sqrt{k_+}\). In the canonical
   \(7,9,11\) case, the two coefficient errors are approximately \(11.8\%\)
   and \(10.6\%\).

3. The default construction adds empirical second- and third-order
   cross-terms in both \(\eta\) and \(\xi\). These terms are not present in
   the initial condition stated in the paper.

4. The carrier potential is evaluated on the combined elevation, including
   the sidebands, rather than retaining the carrier surface potential
   \(\xi_c\) and adding the linear sideband potentials.

5. The sampler contains a special rule replacing the pair
   \((n_c,\Delta n)=(10,2)\) by \((10,3)\), because the former produced
   nonfinite trajectories. This is an empirical exception, not a distinct
   resonance condition: every symmetric pair
   \((n_c-\Delta n,n_c,n_c+\Delta n)\) satisfies the same integer
   four-wave relation.

For a genuinely finite-depth generalization, a linear sideband of amplitude
\(a_s\), wavenumber \(k_s\), and phase \(\phi_s\) should use
\[
  \omega_s^2 = gk_s\tanh(k_s h),
\]
\[
  \eta_s(x)=a_s\cos(k_sx+\phi_s),
\qquad
  \xi_s(x)=
  a_s\frac{g}{\omega_s}
  \frac{\cosh(k_s(h+\eta(x)))}{\cosh(k_sh)}
  \sin(k_sx+\phi_s).
\]

## Static construction test

The audit script
`scripts/audit_bf_jcp09.py` independently constructs four versions:

1. the current empirical construction;
2. the current construction with the empirical cross-terms removed;
3. the literal deep-water initial condition from the paper;
4. the corresponding finite-depth linear-sideband construction.

Each construction was repeated at \(N=512,1024,2048\), and the same fixed
Fourier band was compared. All relative differences are at the
\(10^{-14}\) level or smaller. Thus these initial conditions are analytic
and periodic. There is no endpoint interpolation or seam error analogous to
the old Tanaka construction.

Relative to the literal deep-water formula, the current empirical
construction changes the initial fields by:

| case | \(\eta\) | \(\xi\) | \(G(\eta;h)\xi\) |
| --- | ---: | ---: | ---: |
| canonical \(7,9,11\) | 0.00913 | 0.02097 | 0.02405 |
| archived failing case 1002084 | 0.01624 | 0.02797 | 0.03820 |
| selected finite-depth corner | 0.01857 | 0.10766 | 0.09092 |

The finite-depth correction matters for a clean definition of the finite-depth
family, but the initial states remain fully resolved under all four formulas.

## Canonical recurrence test

At the paper's \(N=64\), with \(\Delta t=0.01\), the literal construction is
finite through 67 carrier periods and reaches its first carrier minimum at
56.83 periods. The current empirical construction reaches the minimum at
57.07 periods. The small difference from the reported value near 60 periods
is consistent with the fact that the audit uses the repository's explicit
fifth-order carrier rather than the paper's numerically computed Fenton
carrier.

Changing four fixed-point sweeps to eight changes the state at \(t=20\) by
approximately \(10^{-14}\). Halving the time step changes it by approximately
\(10^{-8}\). The canonical first focusing event therefore does not depend
materially on either the empirical cross-terms or an under-converged implicit
solve.

An exploratory \(N=128\) run without filtering has much larger Hamiltonian
drift than the \(N=64\) run. Increasing the spatial grid without controlling
the admitted high-wavenumber part of the truncated Dirichlet--Neumann series
is not automatically a refinement.

## Archived nonfinite case

Case 1002084 is one of the archived trajectories that becomes nonfinite. On a
fixed mode-128 representation at \(N=256\), all four initial-condition
constructions fail:

| construction | first nonfinite time |
| --- | ---: |
| current empirical | 60.88 |
| current without empirical cross-terms | 60.88 |
| literal deep-water formula | 60.96 |
| finite-depth-consistent sidebands | 60.96 |

The formula correction delays failure by only \(0.08\). Immediately before
failure, the four trajectories have the same rapid growth in
\(G(\eta;h)\xi\). Hence the empirical construction errors are real, but they
do not cause this terminal event.

An independent replay from the archived \(N=1024\) state gives:

| GL setting | first nonfinite time |
| --- | ---: |
| four sweeps, \(\Delta t=0.01\) | 66.80 |
| eight sweeps, \(\Delta t=0.01\) | 66.79 |
| eight sweeps, \(\Delta t=0.005\) | 66.88 |

At the strongest saved focusing frame, the true fixed-point residual after
four sweeps is \(1.98\times10^{-6}\), and the four- versus eight-sweep state
difference is \(6.45\times10^{-8}\) relative to the state. All three replay
arms follow the same Hamiltonian drift: approximately \(1.04\%\) at
\(t=66.69\), \(10.6\%\) at \(t=66.75\), and order one at \(t=66.78\).
Thus GL iteration count is not the cause. Step halving only postpones the same
breakdown.

## Population evidence

The three archived Benjamin--Feir sources contain 10,240 trajectories. Of
these, 184 become nonfinite. Every failure has sideband offset
\(\Delta n=2\); none with \(\Delta n=3\) or \(4\) fails. No case with
carrier steepness below \(0.11\) fails. The failure rate rises from zero below
\(0.11\) to \(12.9\%\) in the interval \(0.125\leq k_ca\leq0.13\).
Failures concentrate at carrier modes 8 through 17 and are almost independent
of the sampled depth.

The quantity
\[
    \mu=\frac{\Delta n}{n_c}
\]
is the relative sideband separation. At leading nonlinear-Schrödinger order,
the unstable deep-water band satisfies
\[
    \mu < 2\sqrt{2}\,k_ca.
\]
Only 4,784 of the 10,240 archived trajectories, or \(46.7\%\), lie in this
band. The present data family is therefore a broad family of sideband-perturbed
Stokes waves, not a uniformly defined Benjamin--Feir family. The nonfinite
cases are concentrated in the strongest-focusing part of that family.

## Conclusion

There is no Benjamin--Feir analogue of the Tanaka Hermite fix. The
Benjamin--Feir initial conditions are already analytic periodic Fourier
series. Correcting the sideband coefficients and removing the empirical
cross-terms makes the construction match the paper, but does not cure a known
failing trajectory. More GL iterations and step halving also do not cure it.

For a clean new dataset:

1. Define a deep-water Benjamin--Feir family directly from the paper's
   Stokes carrier plus two linear sidebands.
2. Sample \((n_c,\Delta n,k_ca)\) so that the sidebands lie in a declared
   subinterval of the leading instability band, instead of drawing
   \(\Delta n\) independently and adding exceptional mode-pair rules.
3. Remove the empirical cross-terms and the special
   \((n_c,\Delta n)=(10,2)\) replacement.
4. Treat a terminal high-wavenumber focusing event as a failed reference
   trajectory. Reject the complete trajectory after recording fixed-point
   residual and a \(\Delta t/2\) replay; do not retain its finite prefix.
5. If finite-depth Benjamin--Feir data are needed, define them as a separate
   family with the finite-depth dispersion and vertical structure stated
   above.

The decisive artifacts are in:

- `outputs/bf_jcp09_audit_20260723/`
- `outputs/bf_jcp09_failure_audit_n256_20260723/`
