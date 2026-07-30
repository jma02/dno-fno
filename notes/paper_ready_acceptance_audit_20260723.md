# Audit of rejection rules for the paper-ready dataset

Date: 2026-07-23

## Status of this audit

The two-decision conclusion below remains current.  One family-boundary
proposal in this note is obsolete: the harmonic-order condition in
“Membership in the declared family” was superseded by the conservative
fifth-order Ursell bound
\(\mathrm{Ur}_+=H_+\lambda^2/h^3\leq26\).  The current executable contract is
`solver/gen_data/README.md`; the harmonic-order ratio is now diagnostic only.

## Question

The historical datasets used several different cleaning rules.  Which rules
remove numerically bad reference data, and which rules remove finite,
potentially useful samples for no defensible reason?

The central conclusion is that **an evolved training row is not the correct
unit of rejection** and that the paper does not need a list of five or more
independent filters.  There are only two decisions:

1. Does the sampled parameter vector belong to the stated domain of its
   initial-condition constructor?
2. Does the complete numerical reference calculation converge under
   refinement?

The first decision defines the mathematical family; it is not after-the-fact
data cleaning.  The second is the only numerical acceptance test.  The new
dataset should accept or reject an independent static state or an entire
attempted trajectory.  It should not retain the finite prefix of a failed
rollout, and it should not delete isolated frames from an otherwise accepted
rollout.

## Historical rules

### Absolute DNO-output cap

The v9 assembler required

\[
  \max_x |G(\eta;h)\xi|\leq 1
\]

in addition to rowwise finiteness.  Across the three Benjamin--Feir source
archives, this cap removed exactly one finite row.  The following saved frame
from the same trajectory was nonfinite, so the complete trajectory should be
rejected, but the value `1` did not diagnose that failure: the last finite
frames of three other failing trajectories had maxima only
`0.273`, `0.322`, and `0.346`.  The cap therefore has neither scale
invariance nor predictive value.  It is removed from the forward policy.

### Tanaka sign-transition count

The April Tanaka cleaner removed a row when the cyclic derivative of
\(G(\eta;h)\xi\), after a 3% dead zone, changed sign more than ten times.  It
removed 548,360 of 4,261,232 finite rows (12.87%) and removed no nonfinite
rows.

This rule detected the clipped-profile construction then in use, but it is
not a validity condition for a resolved water wave.  A smooth physical state
can have many extrema.  In the present Tanaka case 31, the count reaches 162
even though the stored-frame Hamiltonian drift is below
\(4.81\times10^{-9}\) and independent \(N\)/\(2N\) evaluation of the fixed
band agrees to \(3.30\times10^{-14}\).  Tangent-Hermite construction removes
the interpolation tail that caused the large count without filtering an
evolved row.  The sign-transition rule is therefore removed from the forward
policy.

### Historical \(10^{-5}\) Hamiltonian gate

Three shallow-wavetrain runs attempted 13,824 trajectories.  Their generators
rejected 692 complete trajectories:

| source | rejected | no finite drift recorded | finite drift above \(10^{-5}\) |
| --- | ---: | ---: | ---: |
| standard | 253 | 47 | 206 |
| mid | 230 | 124 | 106 |
| wide | 209 | 65 | 144 |
| **total** | **692** | **236** | **456** |

Of the 456 finite rejections, 426 have maximum stored-frame relative
Hamiltonian drift at most \(10^{-3}\).  Thus the old \(10^{-5}\) threshold
discarded at least 426 cases that pass the established \(10^{-3}\) numerical
screen.  Because the old runs did not record converged GL2 stage residuals or
perform time-step retries, those cases cannot be accepted retroactively; the
result instead shows that \(10^{-5}\) is too aggressive as a production
rejection threshold.

The old steep-Tanaka generator used \(10^{-3}\).  It rejected 1,981 of 9,728
attempts: 1,786 had no finite drift recorded and the other 195 had drift above
\(10^{-3}\).  This does not exhibit the same overly strict threshold.
However, those attempts used the pre-Hermite initial-state constructor, so the
new corpus should regenerate them rather than inherit either decision.

## Audit of the proposed checks

A fresh panel contains 256 independently sampled cases from each of ten
initial-condition families, or 2,560 cases in total.  On the saved nominal
reference trajectories:

- 24 cases contain a nonfinite state or DNO evaluation;
- 28 more are finite but have stored-frame relative Hamiltonian drift above
  \(10^{-3}\);
- the remaining 2,508 pass this retrospective screen.

The 28 finite cases are **not final rejections**.  Three Benjamin--Feir cases
lie just above the boundary, with drifts `0.001017`, `0.001098`, and
`0.001217`.  One shallow Airy diagnostic has a transient drift
`0.001760` but final drift only `0.000155`.  These examples show why a
one-pass threshold would be overly zealous.  Each nominal failure must be
rerun at half time step and judged by agreement of two admissible,
consecutive refinements.

The 28 finite nominal failures, grouped by family, are:

- shallow Airy diagnostic:
  `95000044`, `95000124`, `95000163`, `95000176`, `95000204`,
  `95000209`, `95000242`, `95000243`;
- finite-depth Stokes:
  `99000118`, `99000138`, `99000201`;
- Benjamin--Feir, first shard:
  `92000055`, `92000091`, `92000112`, `92000204`, `92000240`;
- Benjamin--Feir, second shard:
  `93000085`, `93000115`, `93000123`, `93000207`, `93000226`,
  `93000252`;
- Benjamin--Feir, third shard:
  `94000009`, `94000036`, `94000120`, `94000173`, `94000175`,
  `94000200`.

The first refinement queue should begin with cases `93000085`, `92000091`,
and `93000207`, whose drifts are respectively `0.001017`, `0.001098`, and
`0.001217`, followed by the transient shallow diagnostic `95000044`.

Lowering the screen to \(10^{-5}\) would flag 142 otherwise finite cases in
this panel, compared with 28 at \(10^{-3}\).  The extra 114 cases are spread
across linear, Stokes, Benjamin--Feir, and random-sea families.  This is a
second independent reason not to use \(10^{-5}\).

The water-column condition

\[
  \min_{x,t}\frac{h+\eta(x,t)}{h}\geq\frac12
\]

adds no rejection among the nominally valid 2,508 cases.  It catches five
already-invalid shallow Airy diagnostics.  No overly strict rejection by this
condition was observed in the panel.

## Construction restrictions are not row filters

The finite-depth fifth-order Stokes expansion is retained only when its
elevation coefficients satisfy

\[
  \sum_{m=2}^{5}|E_m|\leq |E_1|.
\]

This excludes all seven invalid finite-depth Stokes cases in the fresh panel
and one case that passes the stored Hamiltonian screen.  Direct substitution
of that extra case into the kinematic surface condition gives relative defect
`0.219`, so it is not a credible fifth-order traveling Stokes state.  The
condition is therefore a justified restriction on the parameterized
constructor, not an overly zealous visual filter.

The three-grid fixed-band calculation is similarly used to validate each
generator revision and freeze its supported parameter cells before
production.  It is not applied after the fact to select attractive individual
rows.

## Why the previous list looked like too many rules

The earlier wording mixed mathematical definitions, solver implementation
details, diagnostics, and actual data decisions.  They have different roles:

| Previous item | What it does | Example it would flag | Final role |
| --- | --- | --- | --- |
| Family parameter bounds | Defines which waves the named constructor claims to represent. | Finite-depth Stokes case 98 has \(\sum_{m=2}^5|E_m|/|E_1|=1.183\), outside the ordered fifth-order regime. | Keep as the definition of the family, not a cleaning rule. |
| Finite values | Detects that a numerical calculation ceased to return real numbers. | Benjamin--Feir case `1002084` becomes nonfinite near \(t=66.8\). | Fold into convergence: a nonfinite calculation has infinite refinement error. |
| Water column at least \(h/2\) | Imposes an extra safety margin above the bottom. | Shallow diagnostic `95000000` reaches \(0.387h\). It is already nonfinite with energy drift \(0.444\). | Remove. The only mathematical domain condition is \(h+\eta>0\); the \(h/2\) threshold added no rejection among 2,508 otherwise-valid fresh cases. |
| GL2 stage residual below \(10^{-8}\) | Determines whether the nonlinear equations defining one implicit step were solved. | Saved Benjamin--Feir case `1002084` has four-sweep residual \(1.98\times10^{-6}\), but continuing to eight sweeps reduces it to \(6.65\times10^{-10}\). | Keep inside the integrator. Continue solving or refine the step; do not reject a physical parameter draw from an unfinished stage solve. |
| Hamiltonian drift below a threshold | Checks an invariant as an indirect warning about the trajectory. | Fresh case `93000085` has maximum drift \(1.0168\times10^{-3}\), only \(1.68\%\) above the proposed cutoff. | Report as a diagnostic and use it to prioritize refinement. Do not reject from this scalar alone. |
| Time-step refinement | Directly tests whether changing the numerical discretization changes the fields that will be stored. | In case `1002084`, \(\Delta t=0.01\) and \(0.005\) both fail and the last common states differ by \(0.0368\). | Keep as the only numerical acceptance test. |
| Three-grid fixed-band audit | Tests a constructor and target implementation on an independently generated resolution panel. | Historical clipped Tanaka case 11 has fixed-band DNO discrepancy \(0.0754\), far above \(10^{-3}\). | Use before production to reject or revise a generator version, not to delete individual rows. |
| Sign count or \(\max|G(\eta;h)\xi|\) | Attempts to identify bad-looking individual rows from one sampled field. | Tanaka case 31 has sign count 162 despite a converged trajectory; BF case `2004594` exceeds the unit cap while other failing BF cases remain below \(0.35\). | Remove completely. |

## The two forward decisions

### 1. Membership in the declared family

For each family \(f\), define a parameter set \(\Theta_f\) before generation.
A sampled vector \(\theta\) is used only when \(\theta\in\Theta_f\).  This is
no different from saying that a variable is sampled from an interval: points
outside the interval are not members of the proposed distribution.

For finite-depth fifth-order Stokes states, membership includes

\[
  \sum_{m=2}^{5}|E_m(\theta)|\leq |E_1(\theta)|.
\]

Case 98 is the concrete excluded example.  Its ratio is `1.183`, its direct
kinematic defect is `0.2699`, and increasing the DNO order to nine does not
repair it.  It is therefore not a trustworthy fifth-order traveling Stokes
state.  The corresponding figure is
`outputs/stokes_case98_audit_20260723/stokes_case98_original_vs_shallow_cap.png`.
Replacing this definition by the simpler rectangle \(kh\geq0.75\),
\(ka\leq0.15\) would discard 87,795/500,000 archived draws (17.56%), including
42 truth-valid cases in the fresh panel.  The coefficient definition discards
11,154/500,000 (2.23%) and therefore avoids that unnecessary loss of shallow
coverage.  We also tested the elementary envelope
\(ka\leq\min\{0.15,0.6(kh)^3\}\).  It nearly reproduces the coefficient
condition, but the constant `0.6` was obtained by numerical maximization
rather than an analytic bound.  The coefficient condition is therefore the
cleaner mathematical definition.

The Tanaka Hermite reconstruction, literal Benjamin--Feir formula, and
JONSWAP/TMA parameter ranges are likewise parts of their constructors.  They
are not additional predicates applied after looking at generated outcomes.

### 2. Convergence of the complete reference calculation

Let \(z_r(t)=(\eta_r(t),\xi_r(t))\) be the reference trajectory computed with
\(\Delta t_r=\Delta t_0/2^r\).  At each common saved time define the three
dimensionless delivered fields

\[
  Y_r(t)=\left(
    \frac{P_K\eta_r(t)}{h},\,
    \frac{P_K\xi_r(t)}{h\sqrt{gh}},\,
    \frac{P_K[G_M(\eta_r(t);h)\xi_r(t)]}{\sqrt{gh}}
  \right).
\]

Write \(\mathcal U_r\) for the set containing the three components of \(Y_r\).
For one attempted case \(\theta\), define the single refinement error

\[
  E_r(\theta)=
  \max_{u\in\mathcal U_r}
  \frac{\displaystyle
    \max_j\|u_{r+1}(t_j)-u_r(t_j)\|_{L^2}}
       {\displaystyle
    \max_j\|u_{r+1}(t_j)\|_{L^2}+10^{-12}}.
\]

The maximum in the denominator is taken over the complete trajectory, rather
than normalizing at each instant.  This avoids falsely rejecting a
Benjamin--Feir or counterpropagating Tanaka state merely because
\(G(\eta;h)\xi\) temporarily passes near zero.

If either calculation does not reach the requested final time, contains a
nonfinite value, leaves the mathematical water-wave domain \(h+\eta>0\), or
cannot solve an implicit GL2 stage, set \(E_r(\theta)=+\infty\).  These are
not four more thresholds; they mean that the trajectory needed to evaluate
the one convergence quantity does not exist.

Accept the case when

\[
  E_r(\theta)\leq 10^{-3}
\]

for a pair of consecutive resolutions, and store the finer trajectory.  Run
\(r=0,1\) for every case.  If \(E_0>10^{-3}\), allow one final calculation at
\(r=2\) and accept only if \(E_1\leq10^{-3}\).  Otherwise reject the complete
case.  Benjamin--Feir case `1002084` is a concrete rejection:
\(\Delta t=0.01\) and \(0.005\) become nonfinite at `66.79` and `66.88`, and
their last common states differ by `0.0368`.

For a static state, the same statement is used with independently constructed
\(N\)- and \(2N\)-point fields instead of two time steps.  A
family-stratified spatial panel validates the fixed production grid before
rollout generation; it is not another rowwise filter.

There is therefore one family definition and one numerical acceptance
quantity.  Hamiltonian drift, mass, stage iteration counts, and spectra are
reported to demonstrate that the chosen reference method behaves sensibly,
but none is fitted as another accept/reject threshold.  Failed attempted
cases and the value of \(E_r\) remain in the generation ledger, while only
complete accepted cases contribute training rows.

## Consequence for existing v9 data

The immutable v9 archive is a historical artifact and cannot be certified
under the forward rule because it lacks internal-step and refinement
diagnostics.  A retrospective whole-trajectory mask based only on saved
frames would remove 67,779 rows currently present in v9 (0.888%): 67,649
Benjamin--Feir rows and 130 random-sea rows.  This view is useful for audits,
but the paper-ready dataset should be regenerated directly under the forward
contract rather than described as a cleaned v9 file.
