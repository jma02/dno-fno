# Historical sign-flip filter and the proposed resolution test

Date: 2026-07-22

## Question and answer

The historical sign-flip filter was genuinely used.  It is not merely a rule
that appeared in an unused script.  It produced the 3,712,872-row parent
archive from which the 500,000-row Tanaka dataset used by the April FNO runs
was sampled.

The proposed fixed-band resolution test is **not** a set-theoretic superset of
the sign-flip rule on arbitrary valid wave data.  Nor should it be forced to
be one: the sign counter also counts resolved physical extrema.  A spectral
tail threshold low enough to contain every sign-flip rejection rejects almost
the entire sign-flip-negative population.

There is nevertheless a useful transition result.  On the exactly
reconstructed frame-zero cases from the first historical batch, the
fixed-band resolution test catches all 12 rows removed by the sign rule and
also catches 29 of all 242 rows that the sign rule retained.  Conversely, three
sign-positive initial conditions from the corrected periodic builder are well
resolved and pass the resolution test by more than two orders of magnitude.
Thus the resolution test detects old clipped tails missed by the sign rule,
without treating every oscillatory state as invalid.

The soliton-guard selector in the manuscript is a third, separate object.  It
acts on the initial surface shape to decide whether a numerical stabilizer is
eligible; it is not a data-cleaning rule and is not claimed to contain the
historical sign-flip rejects.

## 1. What the historical rule was

For a stored row, let

\[
q_j = [G(\eta)\xi](x_j), \qquad
d_j = \frac{q_{j+1}-q_j}{\Delta x},
\]

with periodic indexing.  Set \(\rho=0.03\), discard every \(d_j\) satisfying

\[
|d_j|\leq \rho\max_\ell |d_\ell|,
\]

and write \(s_1,\ldots,s_m\in\{-1,1\}\) for the signs that remain in cyclic
spatial order.  The cleaner computed

\[
C_\rho(q)
  = \sum_{i=1}^{m}
      \mathbf 1\{s_i\neq s_{i+1}\},
\qquad s_{m+1}=s_1,
\]

and rejected the row when

\[
C_{0.03}(q)>10.
\]

The retired cleaner's saved summary,
`data/old_tanaka_1_clean.clean.json`, records

| quantity | rows |
| --- | ---: |
| before the sign filter | 4,261,232 |
| retained | 3,712,872 |
| removed as nonfinite | 0 |
| removed for more than 10 sign changes | 548,360 |

Hence the rule removed 12.8686% of the available rows.  The chain of use is
also recorded on disk:

1. `data/old_tanaka_1_clean_sub500k.summary.json` says that the 500,000-row
   subset was sampled from the 3,712,872-row clean archive.
2. Multiple April FNO configurations and checkpoint metadata, for example
   `outputs/fno_jax_10m_20260405_191326/config.json`, explicitly name
   `tanaka_1_clean_sub500k.npz` as their training set.

The current v9 dataset is different.  Its Tanaka sources are
`tanaka_2_adaptive_g0.npz` and `tanaka_2_adaptive_g1.npz`, as specified in
`solver/gen_data/combine_datasets.py`; those sources were not passed through
the historical sign filter.

## Mechanistic interpretation of why the historical rule worked

The rule was not a general smoothness criterion, but it was not devoid of
physical meaning for the old Tanaka generator.  Write

\[
q=G(\eta)\xi=\eta_t.
\]

For an exactly translating profile \(\eta(x,t)=\bar\eta(x-ct)\),

\[
q=-c\bar\eta_x,
\qquad
q_x=-c\bar\eta_{xx}.
\]

Thus the dead-zoned sign transitions of the forward difference of \(q\)
approximately count transitions between the principal curvature lobes of the
surface.  A smooth unimodal solitary crest has two inflection points.  The old
generator placed one to three crests in a row, so an undistorted translating
configuration has a nominal count of roughly two to six.  The cutoff at ten
left room for crest interaction and discretization, while the 3% dead zone
prevented very small curvature fluctuations from controlling the count.

The old builder used zero extension outside the tabulated solitary-wave
profile.  A translated crest near a periodic endpoint could therefore have a
non-negligible tail cut to zero.  The resulting seam mismatch has slowly
decaying Fourier coefficients, and the derivatives in the order-six DNO
expansion amplify the associated ringing.  This creates many alternating
curvature lobes in \(q\), which is exactly the morphology measured by
\(C_{0.03}\).

This interpretation can be checked directly on all 254 reconstructable
frame-zero cases from historical batch 0.  Define the normalized endpoint
mismatch

\[
J_\eta=
\frac{|\eta(x_0)-\eta(x_{N-1})|}{\|\eta\|_\infty}.
\]

The twelve frame-zero sign rejections have median \(J_\eta=0.219\), with range
\([0.051,0.998]\), whereas the 242 retained cases have median
\(2.50\times10^{-6}\).  Every rejected case has \(J_\eta>0.05\).  The
Mann--Whitney test comparing the two populations gives \(p=3.9\times10^{-8}\).
Endpoint mismatch is itself strongly anticorrelated with the minimum
crest-to-boundary clearance (Spearman \(\rho=-0.967\),
\(p=5\times10^{-151}\)).

The filter's behavior over the complete 200 saved rows of each reconstructed
case shows the same mechanism.  The per-case fraction of rows removed has
Spearman correlation \(-0.701\) with initial crest-to-boundary clearance
(\(p=8.4\times10^{-39}\)).  Removal rates stratified by clearance are

| minimum boundary clearance | rows rejected by the sign rule |
| ---: | ---: |
| \(<5\) | 5,377 / 5,400 (99.57%) |
| \(5\)--\(10\) | 64.83% |
| \(10\)--\(20\) | 20.40% |
| \(20\)--\(40\) | 5.91% |
| \(\geq40\) | 0.26% |

There is also a direct construction-level counterfactual.  Holding the twelve
rejected frame-zero crest specifications fixed and replacing only the old
zero-extended interpolation by the three-copy periodic construction changes
all twelve counts from 12--38 to 2--8.  Their fixed-band DNO defects change
from 0.0101--0.465 to \(5.79\times10^{-6}\)--\(1.10\times10^{-5}\).  This
intervention is stronger evidence than the boundary correlation alone: it
isolates the nonperiodic embedding as the cause of both the sign-count response
and the grid-refinement failure.

These results explain the rule's useful positive decisions: in its intended
one-to-three-crest pilot family, a large count was strong evidence for the old
boundary-truncation artifact.  They do not make the count complete.  For
example, retained historical case 163 has only two dead-zoned sign transitions
despite \(J_\eta=1\) and a fixed-band refinement defect of 0.463.  Nor do they
make it portable to other wave families: resolved multi-crest and Stokes waves
can have more than ten physical transitions.  The journal-safe claim is
therefore that the sign rule was a family-specific curvature screen that
successfully removed many outputs corrupted by the old embedding, after which
the final generator removed that embedding defect by construction.

## 2. Why a spectral-tail threshold cannot meaningfully contain the rule

For Fourier cutoff \(K\), define the derivative-tail energy fraction

\[
S_K^{(1)}(q)
  =
  \frac{\displaystyle\sum_{|k|>K}|k|^2|\widehat q_k|^2}
       {\displaystyle\sum_{k\neq0}|k|^2|\widehat q_k|^2}.
\]

The sign counter and \(S_K^{(1)}\) measure different properties.  A precise
zero-count argument makes the distinction clear.  Let \(f=D_hq\) be the
periodic forward difference and suppose

\[
\|(I-P_K)f\|_\infty
  < \rho\|f\|_\infty.
\]

At every point retained by the dead zone, \(P_Kf\) then has the same sign as
\(f\).  Every counted cyclic sign transition forces a zero of the
trigonometric polynomial \(P_Kf\).  Since a nonzero trigonometric polynomial
of degree \(K\) has at most \(2K\) zeros,

\[
C_\rho(q)\leq 2K.
\]

Consequently, \(C_{0.03}(q)>10\) guarantees some significant content only
above mode 5.  It does not guarantee contamination near modes 64, 80, or 128.
For example, \(D_hq=\sin(6x)\) has 12 sign transitions and no Fourier energy
above mode 6.

### Population tests

A CPU scan applied the exact historical predicate to all 2,000,000 rows of
the two corrected, nonadaptive Tanaka-v2 archives.  It labeled 33,054 rows
(1.6527%) positive.  Among 17 one-grid diagnostics, the best simple proxy was
the fraction of \(q\)-energy above mode 16:

| operating point | recall of sign-positive rows | false-positive rate on sign-negative rows |
| --- | ---: | ---: |
| threshold chosen for complete containment | 100% | 99.833% |
| threshold chosen for 99.9% recall | 99.9% | 88.466% |
| threshold chosen for 1% false positives | 23.017% | 1% |

The same conclusion holds on data used by v9.  On the first 51,200 rows of
each adaptive Tanaka source, the sign rule labels 1,031 g0 rows and 803 g1
rows.  A derivative-tail threshold above mode 80 that contains every one of
those rows also labels 99.783% and 99.393%, respectively, of the rows that
pass the sign rule.  The overlap is displayed in
`notes/signflip_spectral_overlap_20260722.png`.

This is not a threshold-tuning failure.  It is the expected consequence of
using an extrema counter as a resolution diagnostic.  Finite-depth Stokes
waves give the complementary example: all 8,192 sampled rows have more than
10 sign changes after the common \(|k|\leq128\) projection, while an
independently regenerated 256-case Stokes panel has fixed-band \(N\)-versus-
\(2N\) defects with median \(6.25\times10^{-15}\), 95th percentile
\(9.72\times10^{-15}\), and maximum \(4.20\times10^{-7}\).

## 3. Fixed-band independent-resolution defect

The replacement audit asks whether the numerical construction and the
order-six DNO agree when the grid is refined while the physical band is held
fixed.  Independently construct the same initial condition on grids of size
\(N\) and \(2N\).  Let \(P_K\) be the projection onto \(|k|\leq K\), and let
\(R_{2N\to N}\) be exact Fourier restriction.  Define

\[
q_N^K
  = P_K\!\left[G_N^{(6)}(P_K\eta_N)(P_K\xi_N)\right],
\]

and analogously \(q_{2N}^K\).  The measured defect is

\[
\delta_K
  =
  \frac{\|q_N^K-R_{2N\to N}q_{2N}^K\|_2}
       {\|R_{2N\to N}q_{2N}^K\|_2+\epsilon}.
\]

Here the notation \(G(\eta)\xi\) is understood in the definition of
\(q_N^K\): both the surface and Dirichlet datum are projected before the DNO
is evaluated, and the output is projected afterward.  Merely Fourier-
interpolating an already constructed \(N\)-grid state is not an independent
resolution test.

The current experiment uses \(N=1024\), \(2N=2048\), \(K=341\) for the
historical two-thirds-band solver, and \(K=128\) for the current protocol.  A
provisional screen labels \(\delta_K>10^{-3}\).  This tolerance is an audit
value, not yet a frozen production threshold; it should be calibrated on an
\(N,2N,4N\) panel.

### Historical reconstruction

The original pre-filter archive is no longer present, so an exhaustive
rowwise test of all 548,360 removed rows is impossible.  Frame-zero cases in
the first batch can nevertheless be reconstructed exactly from the archived
seed and generator parameters.  This required care because deduplication
compacted the saved specification list: 254 compact specifications match the
256 seeded cases one-to-one, with original case IDs 195 and 238 already absent
before the sign filter.  Those two cases are excluded from the comparison.

Among the remaining 254 cases, 12 frame-zero rows were removed by the sign
rule and 242 were retained.  The completed deterministic refinement panel
contains all 254 rows.

As a mapping and precision check, the reconstructed sign counts agree exactly
with the archived counts for all 242 retained rows in both float32 and
float64.  Their counts range from 2 to 10.  The 12 removed cases have
reconstructed counts between 12 and 38, with no precision-dependent threshold
crossing.

At the provisional tolerance:

| population | number tested | \(\delta_K>10^{-3}\) |
| --- | ---: | ---: |
| historical, sign-rejected | 12 | 12 (100%) |
| historical, sign-retained | 242 | 29 (11.98%) |
| current periodic-builder, sign-positive | 3 | 0 (0%) |

Every historical sign-rejected defect is at least \(1.01\times10^{-2}\);
the largest is 0.465.  Relative to this provisional refinement label, the
sign screen has 100% precision but only \(12/41=29.3\%\) recall.  The retained
cases caught by refinement are not small threshold ambiguities.  For example,
historical case 163 has only two dead-zoned sign changes but
\(\delta_K=0.463\); a crest center lies only 0.228 from the boundary of the old
length-164 construction.  This is a direct example of a clipped or
grid-inconsistent tail missed by the sign rule.

The three current sign-positive cases have 12, 18, and 20 sign changes but
defects between \(4.90\times10^{-6}\) and \(7.17\times10^{-6}\).  They are
resolved under the proposed test and would be false rejections under the old
rule.  The full case-level transition plot is
`notes/signflip_refinement_full_batch0_20260722.png`.

## 4. Decision

1. Preserve the historical fact: the 3%-dead-zone, 10-transition rule was
   used to produce the old FNO training dataset and removed 548,360 rows.
2. Do not claim that a one-grid tail-energy threshold supersets it.  Complete
   containment has a practically 100% false-positive rate.
3. Do not apply the historical sign rule to current periodic data.  It rejects
   resolved multi-crest and wavetrain geometry.
4. Use fixed-band independent refinement as the primary proposed validity
   test for newly generated data, together with finiteness and reference-
   invariant checks.  Calibrate and freeze its threshold on an
   \(N,2N,4N\) panel before production use.
5. If exact backward compatibility is required for an audit, the only literal
   superset is

   \[
   \{C_{0.03}>10\}\ \cup\ \{\delta_K>\tau\}.
   \]

   This union should not be presented as the scientifically preferred current
   filter because it deliberately inherits the sign counter's false
   positives.

## 5. Reproduction and artifacts

CPU commands:

```bash
JAX_PLATFORMS=cpu uv run python scripts/audit_signflip_spectral_overlap.py
JAX_PLATFORMS=cpu uv run python scripts/audit_signflip_refinement_transition.py
```

Primary artifacts:

- `notes/signflip_spectral_overlap_20260722.{json,npz,png}`
- `notes/signflip_spectral_full_v2_audit_20260722.json`
- `notes/signflip_spectral_full_v2_metric_summary_20260722.json`
- `notes/signflip_refinement_transition_20260722.{json,png}`
- `notes/signflip_refinement_full_batch0_20260722.{json,png}`
- `notes/signflip_boundary_mechanism_20260722.png`
- `notes/signflip_boundary_bins_20260722.tsv`
- `notes/signflip_periodization_counterfactual_20260722.json`
- `notes/signflip_valid_ic_spectral_screen_20260722.json`
- `notes/signflip_v9_sample_spectral_screen_20260722.json`
- `notes/signflip_stokes_independent_convergence_20260722.npy`
