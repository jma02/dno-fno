# Translation-tail investigation, 2026-07-17

## Conclusion

The current localized tangent loss is implemented correctly and does affect
the single-crest translation mechanism. It is not, however, a reliable
population-tail objective. Increasing its weight from 10 to 100 improved the
paired Tanaka-g0 panel but did not improve Tanaka-g1, and the loss is dominated by a
handful of rows. Increasing the tangent or \(\gamma^2\) weight again is therefore
not the recommended next experiment.

The evidence instead points to a small, bounded error in the complete learned
DNO, amplified over a long neutral direction. The immediate experiment is
already running: finish C25 and evaluate whether moderate additional capacity
reduces the full DNO defect. If C25 does not improve both Tanaka families, run
a matched screening ablation in which every arm has at most three loss terms.
Do not add a fifth loss to C25.

## Quantities used in the audit

Let \(q_\theta=G_\theta(\eta)\xi\) and let \(q_6\) denote the order-six numerical
reference. At terminal time, define

\[
 E_{\rm raw}
 =\frac{\|\eta_\theta-\eta_6\|_2}{\|\eta_6\|_2}.
\]

For periodic translation \(T_d f(x)=f(x-d)\), the fitted displacement and
aligned error are

\[
 d_*=\arg\min_d\|T_{-d}\eta_\theta-\eta_6\|_2,
 \qquad
 E_{\rm align}
 =\frac{\|T_{-d_*}\eta_\theta-\eta_6\|_2}{\|\eta_6\|_2}.
\]

The translation component reported below is

\[
 E_{\rm trans}
 =\sqrt{\max(E_{\rm raw}^2-E_{\rm align}^2,0)}.
\]

This is a diagnostic decomposition of terminal elevation error. It is not an
orthogonal decomposition of the full Zakharov state.

For centered DNO error \(e=q_\theta-q_6\), the implemented tangent loss first
fits a local speed defect,

\[
 \delta c_h(x)
 =-\frac{B_h(\eta_x e)(x)}
 {B_h(\eta_x^2)(x)+10^{-3}\max_y B_h(\eta_x^2)(y)+10^{-30}},
\]

where \(B_h\) is Gaussian convolution with Fourier multiplier
\(\exp[-(hk)^2/2]\). It then forms

\[
 \ell_{\rm tan}
 =\int
 \frac{B_h(\eta_x^2)}{\int B_h(\eta_x^2)}
 \frac{\delta c_h^2}{gh}\,dx.
\]

The phase-growth version is

\[
 \gamma^2=gh\,\kappa(\eta)^2\ell_{\rm tan},
 \qquad
 \kappa(\eta)^2=\frac{\|\eta_x\|_2^2}{\|\eta\|_2^2}.
\]

For a pure defect \(e=-\delta c\eta_x\), \(\gamma^2\) is the squared
instantaneous growth rate of relative elevation error. For a general DNO
defect it is only a projected surrogate.

## Correctness checks

All focused CPU tests passed:

- localized tangent regularizer: 9/9;
- modal phase-rate regularizer: 9/9;
- finite-time phase regularizer and production-GL2 parity: 6/6;
- Hadamard regularizer: 7/7;
- mode-balanced regularizer: 6/6.

The tangent gradient also agrees with a centered finite difference to relative
error \(4.6\times10^{-10}\). Its sampled Hessian is positive semidefinite, its
constant-mode gradient is zero to \(2.95\times10^{-16}\), and the selected-row
multi-device reduction is the exact global selected-row mean. The defect is
therefore not a sign, source-mask, gradient, or cross-device averaging bug.

The two analysis programs introduced for this audit pass py_compile and Ruff.
Their machine-readable results are:

- notes/c21_translation_tail_population_audit_20260717.json;
- notes/c20_c21_paired_translation_comparison_20260717.json.

## What the 512 fresh Tanaka rollouts show

There are 43 terminal Tanaka cases with \(E_{\rm raw}>0.25\). A single global
shift removes more than 95% of squared error in 31 of them. Geometry matters:

- all 25 single-crest cases clear the 95% removal test, with median removal
  99.861% and median aligned error 0.02146;
- only 6 of 18 multi-crest cases clear it, with median removal 91.87% and
  median aligned error 0.14848.

Thus “the translation tail” contains two problems: nearly rigid drift of a
single crest and differential displacement/deformation of multiple crests.

The five Tanaka cases with raw error above one develop smoothly. A quadratic
fit of displacement over \(20\le t\le200\) has \(R^2\) between 0.99920 and
0.999999. A persistent one-grid-cell displacement appears 13--35 time units
before raw error first crosses 0.25. There is no abrupt terminal event,
high-band cascade, or failed GL2 solve in these trajectories.

The population is strongly depth dependent. For Tanaka-g0, all 13 cases with
\(E_{\rm raw}>0.25\) and more than 90% translation removal have \(h<0.028\).
Their rates by increasing depth quartile are 17.19%, 3.13%, 0%, and 0%. For
Tanaka-g1, the corresponding rates are 21.88%, 7.81%, 3.13%, and 1.56%.

This is not a missing-depth data gap. Sources 5, 6, and 14 contain 2.2 million
Tanaka rows; sources 5 and 6 contain about 198,400 and 206,600 rows below
\(h=0.02\). The problem is that the rollout tolerance becomes much tighter as
\(\kappa(\eta)\) grows, not that shallow states are absent.

## Which one-step scores predict the translation component?

The following table uses the frame-zero score on each fresh C21 initial state.
The correlation is Spearman correlation with terminal \(E_{\rm trans}\). The
last two columns give the number of translation tails captured by the worst
10% of the score.

| Score | Tanaka-g0 correlation | Tanaka-g1 correlation | Tanaka-g0 recall | Tanaka-g1 recall |
|---|---:|---:|---:|---:|
| Relative full-\(q\) \(L^2\), squared | 0.522 | 0.474 | 9/13 | 9/22 |
| Exact kinematic rate \(\|q_\theta-q_6\|_2^2/\|\eta\|_2^2\) | 0.724 | 0.712 | 9/13 | 9/22 |
| Relative full-\(q\) \(H^1\), squared | -0.471 | -0.509 | 2/13 | 1/22 |
| Mode-balanced loss | 0.602 | 0.593 | 8/13 | 8/22 |
| Local tangent loss | 0.593 | 0.557 | 9/13 | 9/22 |
| \(\gamma^2\) | 0.729 | 0.716 | 9/13 | 9/22 |
| \(h\kappa^2\) times full-\(q\) \(L^2\) | 0.721 | 0.716 | 9/13 | 9/22 |
| \(h\kappa^2\) times mode-balanced loss | 0.699 | 0.699 | 9/13 | 9/22 |
| \(\Omega_{\rm eff}^2\) times mode-balanced loss | 0.697 | 0.701 | 9/13 | 9/22 |

Two conclusions follow.

First, ordinary relative \(H^1\) error orders the shallow translation cases in
the wrong direction. A lower validation \(H^1\) loss is therefore not evidence
that long-time phase tails must improve.

Second, almost all of the ranking improvement in \(\gamma^2\) can be reproduced
by multiplying a nonprojected full-\(q\) error by the known sensitivity
\(h\kappa^2\). The local tangent projection does not improve top-decile tail
recall. If another sensitivity-weighted loss is tested, the full DNO error is
more consistent with the oracle evidence than another projected speed loss.
The exact common-state kinematic score makes the same point without a
traveling-wave approximation: for a short time \(\tau\),

\[
 \frac{\|\eta_\theta(\tau)-\eta_6(\tau)\|_2^2}
      {\tau^2\|\eta(0)\|_2^2}
 \longrightarrow
 \frac{\|q_\theta(0)-q_6(0)\|_2^2}{\|\eta(0)\|_2^2}.
\]

The scores are useful for severe cases: the \(\gamma^2\) top decile contains all
eight raw errors above 0.75 and all five above one. It captures only about 53%
of the 43 errors above 0.25, so it is not a sufficient population-tail gate.

## The mean tangent loss is effectively a handful of rows

On the fresh initial states, the upper tail supplies most of the mean:

| Score | Tanaka-g0, top 1% / top 5% | Tanaka-g1, top 1% / top 5% | Effective \(N\) out of 256 |
|---|---:|---:|---:|
| Tangent | 72.8% / 96.3% | 61.6% / 91.9% | 4.10 / 7.02 |
| \(\gamma^2\) | 81.3% / 98.9% | 69.4% / 96.3% | 3.18 / 5.47 |
| \(h\kappa^2\) times full-\(q\) \(L^2\) | 76.0% / 98.2% | 59.5% / 95.4% | 4.10 / 6.61 |
| \(h\kappa^2\) times mode-balanced loss | 29.9% / 66.6% | 22.5% / 61.1% | 19.83 / 25.41 |
| \(\Omega_{\rm eff}^2\) times mode-balanced loss | 29.7% / 66.1% | 22.2% / 60.9% | 20.05 / 25.70 |

The issue is not that a mean ignores large rows. The issue is that whichever
few rows are currently largest determine the auxiliary gradient. Reducing
that mean can exchange one set of tail cases for another without controlling
the held-out family tail.

The sensitivity-weighted mode loss is the exception worth screening. It has
nearly the same terminal-translation ranking and exactly the same top-decile
recall as \(\gamma^2\), but its soft per-mode normalization raises the effective
sample size from about 3--5 to about 20--25. It remains tail weighted without
being numerically determined by only a few samples.

For use outside the shallow Tanaka regime, define the sensitivity by the
truth-elevation-weighted linear dispersion,

\[
 \Omega_{\rm eff}^2(\eta,h)
 =\frac{\sum_k g|k|\tanh(h|k|)|\widehat\eta_k|^2}
        {\sum_k|\widehat\eta_k|^2}.
\]

This reduces to \(gh\kappa^2\) in shallow water but remains meaningful in deep
water. On the fresh Tanaka panel it is numerically almost identical to the
shallow expression; the reason to prefer it is the broader physical domain,
not a better retrospective fit.

The v9 archive also has no trajectory identifier. It is therefore not
possible to form a genuine per-trajectory accumulated signed-phase objective
from the combined archive without rebuilding its row-to-trajectory map. Row-order
heuristics should not be used for a manuscript experiment.

## Did the tangent intervention help?

C20 to C21 is the closest available intervention: one additional low-rate
epoch from C20 while raising tangent weight from 10 to 100. Continuous
alignment gives:

- Tanaka-g0: translation error improves in 21/32 cases, Wilcoxon two-sided
  \(p=0.0118\); the maximum translation component falls from 1.126 to 0.335;
- Tanaka-g1: translation error improves in 17/32 cases, \(p=0.651\); the maximum
  rises from 0.435 to 0.612.

This is evidence that the intervention affects single-crest \(g=0\) drift. It
is not evidence of a family-independent cure. C21 also performs an extra
epoch of every ordinary loss, so the absence of a tangent-zero extra-epoch
control prevents attributing the entire \(g=0\) change to the tangent term.

At weight 10, the tangent contribution is only 0.0829% of the C20 objective and
about 0.076% of C25 at epoch 34. At weight 100 it is 0.683% of C21 and is large
enough to move the model. C21 lowers the training tangent mean by 17%, but the
fixed held-out frame-zero mean rises by 17.2% in Tanaka-g0 and 14.2% in Tanaka-g1.
This is an optimization/generalization failure, not an inactive-term bug.

## Why a static phase projection is incomplete

At a fixed Zakharov state, let \(d=q_\theta-q_6\). To first order, this DNO
defect changes the full vector field by

\[
 F_\theta-F_6
 =\left(d,\,B_*d\right)+O(d^2),
 \qquad
 B_*=\frac{q_6+\eta_x\xi_x}{1+\eta_x^2}.
\]

The current loss uses only the first component and projects it onto
\(\eta_x\). A full-state energy-metric projection changes the fitted
coefficient by roughly a factor of two at frame zero and lowers phase-rate RMSE
along each of the five severe paths by about 40%. It still has essentially
zero correlation with the signed terminal displacement when evaluated only
at the initial state. The missing object is not merely a better static inner
product; it is propagation through the state-dependent linearized dynamics.

The exact same-state split confirms this. In the five severe cases, the
learned same-state DNO defect remains bounded at about 0.4--1.2%, while the
order-six response to the already separated state grows to about 3.1--11.7%.
The latter is downstream of the earlier learned defect, not a second model
error.

Horizontal momentum is not the overlooked scalar cure. The frame-zero
momentum-rate defect has correlation 0.539/0.468 and top-decile recall 6/13 and
4/22, worse than \(\gamma^2\). Momentum separation later in the rollout is a
moderate downstream marker, not an initial discriminator.

Finally, the stagewise oracle is decisive: different reduced defect subspaces
repair cases 24 and 27, while injecting the complete same-state DNO defect
reduces both terminal errors to about \(2\times10^{-8}\). A universal fix must
improve the complete coupled DNO error or use the actual finite-time
sensitivity; another fixed \(q\)-space projection is not supported.

## Recommended experimental sequence

### 1. Finish and evaluate C25 unchanged

C25 tests the simplest hypothesis consistent with the two-geometry oracle:
the structurally correct \(G_0+G_1+O(\eta^2)\) residual needs more approximation
capacity. Evaluate its final checkpoint first on the paired 64-case Tanaka
panel and then on the fresh 256-per-family panels. Report raw, aligned, and
translation-component errors separately for Tanaka-g0 and Tanaka-g1.

Do not interpret C25 as evidence for tangent loss. At epoch 34 its tangent
term contributes only about 0.076% of the objective, while mode-balanced and
Hadamard contributions are about 2.8% and 1.7%.

### 2. If C25 is insufficient, select one third loss by a matched screen

Use one common structural checkpoint, identical sample order, optimizer reset,
learning rate, batch size, and one-epoch budget. Every arm has at most three
terms:

\[
\begin{aligned}
A_0 &: L_{H^1}+\lambda_H L_{\rm Had},\\
A_M &: L_{H^1}+\lambda_H L_{\rm Had}+\lambda_M L_{\rm mode},\\
A_S &: L_{H^1}+\lambda_H L_{\rm Had}
       +\lambda_S\,\Omega_{\rm eff}^2L_{\rm mode}.
\end{aligned}
\]

The third arm uses the complete complex DNO defect with the same analytically
derived shallow/narrow sensitivity as \(\gamma^2\). It replaces, rather than
duplicates, the unweighted mode term. It is preferred to another local tangent
or gamma arm because it does not discard the components required by the
full-defect oracle and is materially less concentrated. Its offline score does
not improve extreme-tail recall, so this remains a falsifiable screening
experiment, not a claimed solution.

Calibrate \(\lambda_M\) and \(\lambda_S\) once on fixed batches to the same
parameter-gradient fraction, then freeze them before any rollout is inspected.
Use final checkpoints. Reject an arm unless it improves both Tanaka families
without BF-modal or linear regression. If neither auxiliary arm wins, retain
\(A_0\) and treat the problem as approximation capacity rather than adding more
losses.

### 3. Only then consider a finite-time adjoint phase term

A terminal phase can be defined by a full-state alignment condition and
differentiated backward through the actual GL2 stages. That gives the correct
discrete finite-time phase covector. Before training it, run a one-dimensional
stagewise oracle on cases 24 and 27. If the adjoint projection does not repair
both at equal correction budget, reject phase-specific training and return to
full-DNO capacity.

This is more principled than the present instantaneous tangent term, but also
more expensive and substantially harder to explain and validate. It is a
fallback, not the next run.
