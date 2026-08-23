# A defensible numerical definition of the JONSWAP/TMA family

Date: 2026-08-04

## Decision

Do not adopt the retrospective condition

\[
  s_{\rm rms}\leq 0.06
\]

as the paper's definition of admissible JONSWAP/TMA sea states.  The quantity
is useful diagnostically, but its value depends materially on the arbitrary
high-frequency cutoff used to discretize the JONSWAP tail.

Instead, define the numerical spectrum on the relative frequency interval

\[
  0.5\,\omega_p\leq \omega\leq 2.5\,\omega_p,
\]

set it to zero outside this interval, and normalize the retained spectrum to
the prescribed significant height.  Require that the upper endpoint is
represented in the delivered initial-state band,

\[
  \omega(K_0;h)\geq 2.5\,\omega_p,
  \qquad K_0=128,
\]

where \(\omega(k;h)=\sqrt{gk\tanh(kh)}\).  Keep the already declared
moderate-sea support

\[
  \epsilon_p=\frac{k_pH_s}{2}\leq 0.08.
\]

This gives one published spectral interval, one elementary resolution check,
and one standard dimensionless steepness.  It does not add a
realization-dependent rejection rule.

At this decision point the construction was candidate population revision 4.
Its paired mechanism gate had passed, but final dataset adoption still awaited
the fresh two-stream population gate described below.  That gate subsequently
passed; its immutable result is recorded at the end of this note.  The
revision-3 outcomes were generated from a different spectrum and cannot
establish the rejection rate of the proposed family.

## Why the RMS-slope cutoff is not self-contained

In deep water, the high-frequency JONSWAP density has the asymptotic form

\[
  S(\omega)\sim C\omega^{-5},
  \qquad k(\omega)=\frac{\omega^2}{g}.
\]

Consequently, the linear mean-square surface slope below frequency
\(\Omega\) behaves as

\[
  \mathbb E|\eta_x|^2
  =\int^{\Omega}k(\omega)^2S(\omega)\,d\omega
  \sim \frac{C}{g^2}\int^{\Omega}\frac{d\omega}{\omega}
  =\frac{C}{g^2}\log\Omega+\text{constant}.
\]

Thus mean-square slope is a low-pass quantity: its numerical value is not a
property of \((H_s,k_p,\gamma,h)\) alone.  Ocean-slope studies likewise state
the cutoff explicitly when reporting mean-square slope [Hwang (2005),
doi:10.1029/2005JC003002](https://doi.org/10.1029/2005JC003002).

Our revision-3 constructor uses one absolute window for every sea state: it is
one through \(k=96\), tapers, and is zero at \(k=128\).  In deep water, the
upper frequency ratio is approximately

\[
  \frac{\omega(128)}{\omega_p}=\sqrt{\frac{128}{k_p}}.
\]

It is therefore 8 when \(k_p=2\), but only 3.27 when \(k_p=12\).  The
constructor assigns different relative short-wave ranges to otherwise
similarly parameterized spectra.  An RMS threshold computed after that step
inherits this grid dependence.

## Literature basis for a relative frequency interval

The interval \([0.5\omega_p,2.5\omega_p]\) is not a breaking theorem.  It is a
standard, explicit definition of the finite set of components used to realize
an irregular JONSWAP sea numerically.

- Draycott et al. prescribe exactly
  \(0.5f_p\leq f\leq2.5f_p\) when comparing phase-resolved irregular JONSWAP
  waves in a fully nonlinear potential-flow solver, a two-phase CFD solver,
  and laboratory experiments
  [Journal of Ocean Engineering and Marine Energy (2026),
  doi:10.1007/s40722-026-00495-0](https://doi.org/10.1007/s40722-026-00495-0).
- Grice, Taylor, and Eatock Taylor use an upper cutoff
  \(\omega_{\max}=2.5\omega_p\) for random JONSWAP realizations
  [Philosophical Transactions of the Royal Society A 373 (2015),
  doi:10.1098/rsta.2014.0113](https://doi.org/10.1098/rsta.2014.0113).
- Bonar et al. use a truncated JONSWAP spectrum with
  \(f_{\min}=0.5f_p\) and \(f_{\max}=3f_p\) for finite-depth irregular waves
  [Journal of Ocean Engineering and Marine Energy 7 (2021), 145--155,
  doi:10.1007/s40722-021-00192-0](https://doi.org/10.1007/s40722-021-00192-0).
- Vasarmidis et al. use the same interval for nonlinear numerical generation
  of finite-depth irregular short-crested JONSWAP waves
  [Ocean Engineering 219 (2021), 108303,
  doi:10.1016/j.oceaneng.2020.108303](https://doi.org/10.1016/j.oceaneng.2020.108303).
- McAllister et al. use the same interval when varying JONSWAP bandwidth from
  \(\gamma=1\) to 4
  [Journal of Fluid Mechanics 971 (2023), A11,
  doi:10.1017/jfm.2023.645](https://doi.org/10.1017/jfm.2023.645).

The HOS-Ocean numerical guidance independently recommends approximately
\(k_{\max}=8k_p\) for nonlinear wave calculations and requires a convergence
study.  In deep water this corresponds to
\(\omega_{\max}/\omega_p=\sqrt{8}=2.83\), close to the upper endpoint above
([HOS-Ocean numerical-parameter guidance](https://lheea.gitlab.io/HOS-Ocean/choice-numerical-parameters.html)).

The literature therefore supports a relative cutoff, but does not prescribe a
unique endpoint: both \(2.5f_p\) and \(3f_p\) are in use.  We choose
\(2.5f_p\) because it is published and is compatible with almost all of our
declared support at \(K_0=128\).  A universal \(3f_p\) interval is not: only
1,544 of the 1,664 archived specifications satisfy
\(\omega(128;h)\geq3\omega_p\), and the minimum ratio in the shallow stratum
is 2.434.  Claiming \(3f_p\) without changing either the grid or the support
would therefore reproduce the hidden grid dependence that this revision is
intended to remove.

At \(2.5f_p\), 1,656 of 1,664 archived specifications satisfy the resolution
condition; the eight exceptions are shallow-water proposals.  The condition
is imposed before phases are drawn, so those parameter values can be excluded
from the sampling support rather than called failed trajectories.  This is
preferable to selecting an absolute Fourier endpoint from the machine grid
and then using a statistic computed from that endpoint to define
admissibility.

## Literature basis for the remaining steepness bound

The dimensionless peak steepness

\[
  \epsilon_p=\frac{k_pH_s}{2}
\]

is the conventional coordinate used to classify JONSWAP/HOS sea states.
Ducrozet et al. explicitly describe \(\epsilon_p=0.08\) as typical of a
moderate sea state
([Physical Review Fluids 6 (2021), 064803](https://doi.org/10.1103/PhysRevFluids.6.064803)).
Luenser et al. study order-four HOS JONSWAP calculations over
\(0.025\leq\epsilon_p\leq0.125\), including \(\gamma=1\), and report that
broader spectra generate more short waves and more locally steep events
([Fluids 7 (2022), 243](https://doi.org/10.3390/fluids7070243)).  Their result
agrees with the strong \(\gamma=1\) effect in our archive, but it does not
provide a universal gamma-dependent rejection boundary.

Physical breaking envelopes are not useful for the present numerical tail.
The published deep- and finite-depth limits examined in the archive admit
essentially every revision-3 failure.  Narrowing \(\epsilon_p\) to 0.05 would
reduce our historical rejection rate, but the literature does not identify
0.05 as a boundary for order-four HOS.  It would therefore be an empirical
restriction that discards useful moderate-sea cases, not a better physical
argument.

## What the archive says and does not say

The completed revision-3 prefix contains 1,664 attempted trajectories and 167
numerical failures.  A retrospective reconstruction gives the following for a
sharp upper frequency cutoff, before renormalizing to the requested \(H_s\):

| upper endpoint | median retained variance | least retained variance | median reduction in slope variance |
| --- | ---: | ---: | ---: |
| \(2.5\omega_p\) | 97.86% | 82.81% | 35.39% |
| \(3\omega_p\) | 99.19% | 88.75% | 22.30% |

The \(2.5\omega_p\) endpoint preserves 97.86% of the old unnormalized variance
in the median case while removing the part of the ideal tail that
disproportionately controls surface slope.  However, the fraction of slope
variance it removes has almost no ability to classify the old outcomes (AUC
0.522).  We must not claim from this calculation that changing the interval
has already fixed the integration failures.  The old outcomes belong to the
old initial conditions.

## Required validation before adoption

1. Implement the relative frequency interval and its deterministic resolution
   condition only in a diagnostic branch;
   keep the TMA factor, random phases, height normalization, nonlinear
   adjustment, GL2 method, evolution band, and trajectory health checks fixed.
2. Replay a predeclared panel of twelve failed revision-3 specifications and
   twelve matched accepted controls.  Proceed only if at least nine of the
   failures are repaired and all twelve controls still pass.  This tests the
   proposed mechanism without selecting cases after seeing the new outcomes.
3. If the paired panel passes, request twenty accepted fresh trajectories in
   each of the 27 cells (540 accepted trajectories in total).  Keep every
   rejected attempt in the audit trail and report adjustment and autonomous
   failures separately.  This is the ordinary production quota mechanism and
   uses no separate phase-valued support gate: phases are sampled in each
   proposal, and a rejected realization is replaced only by the next
   independently seeded attempt in the same cell.  The loader-visible phase
   law is therefore the proposal phase law conditional on full case
   acceptance, not the unconditioned proposal law.
4. Adopt the construction only if no cell exhausts its four-attempt-per-case
   ceiling and at most 2 percent of all fresh attempts are rejected.  If a
   numerical support bound is still needed, define and calibrate it only after
   the relative band is fixed; do not reuse the value 0.06 obtained from the
   absolute-band archive.

Before observing any fresh outcome, the 540-case gate was divided into two
independent validation streams.  Stream 941 and stream 942 each request ten
accepted trajectories in every cell (270 accepted trajectories per stream),
use proposal batches of 32 and horizon-sorted solver batches of eight, and
retain the ordinary four-attempt-per-request ceiling.  The aggregate decision
uses all attempts from both streams and the criterion in item 4.

The clean paper statement would then be: JONSWAP/TMA spectra are sampled on
\([0.5\omega_p,2.5\omega_p]\), normalized to \(H_s\), required to fit in the
delivered initial-state band, restricted to the declared moderate range
\(k_pH_s/2\leq0.08\), nonlinearly adjusted, and retained only when the
predeclared numerical calculation completes its ordinary health checks.

## Paired mechanism result

The paired gate used twelve archived revision-3 failures, one from every
combination of four numerical failure classes and three depth strata.  Each
failure was paired with an unused accepted case from the same sample cell.
The old physical parameters and both phase arrays were held fixed.  Only the
spectral interval and the associated height normalization changed.

| quantity | predeclared requirement | observed |
| --- | ---: | ---: |
| old failures becoming healthy | at least 9 of 12 | 11 of 12 |
| accepted controls remaining healthy | 12 of 12 | 12 of 12 |

The one unrepaired case remained finite and passed nonlinear adjustment.  Its
autonomous calculation was rejected because the maximum GL2 stage residual
was `1.2473e-7`, above the fixed `1e-8` tolerance.  Thus the candidate removed
all nonfinite failures in this panel without causing a control regression.
The frozen plan and strict result are stored under
`outputs/jonswap_relative_band_paired_replay_20260804`; the plan SHA-256 is
`dedff31cbec323bd2d0acbdfee6300dc1da282b586df60f6b81980cb7e760e29`.

## Fresh two-stream population result (2026-08-04)

The predeclared streams 941 and 942 subsequently completed the ordinary
accepted-quota gate above.  Together they retained 540 trajectories from 548
attempts, with exactly 20 accepted trajectories in every one of the 27 cells.
Eight attempts were rejected as incomplete numerical trajectories, so the
observed rejection rate was

\[
  \frac{8}{548}=0.0145985\ldots<0.02.
\]

No cell exhausted the four-attempt-per-accepted-case ceiling.  The fixed result
has schema `jonswap_relative_band_fresh_gate_summary_v1`, status `pass`, and
`passed=true`.  It is stored at
`outputs/jonswap_relative_band_fresh_gate_20260804/jonswap_relative_band_fresh_gate.summary.json`
with SHA-256
`4228660e0db7fbf83caa05596d5409a93e183d59454453ded4b3aeee2670a9d8`.
This closes the predeclared revision-4 constructor-adoption gate.  The
two-percent criterion applies only to that completed pre-bulk gate; it is not a
release threshold for the final bulk dataset.  The completed gate introduced no
separate phase-valued support gate.  Nevertheless, its 540 released phase
pairs are conditional on full case acceptance: every rejected proposal,
including its phase values, remains in the attempt record but contributes no
loader-visible trajectory.
