# Simple CS-DNO architecture ablation

**Recorded:** 2026-07-29
**Status:** Proposed only. This ablation has not been implemented or trained.

## Objective

Determine whether the unusual part of the learned CS-DNO correction is
actually necessary. Keep the comparison as simple as possible: change one
operation, while holding the analytic terms, learned parameters, data, loss,
and training procedure fixed.

## Elementary description of the current model

Write

\[
G_\theta(\eta;h)\xi
=
G_0(h)\xi+G_1(\eta;h)\xi+R_\theta(\eta;h)\xi .
\]

The first two terms are known analytically. The learned correction is a sum of
parallel channels:

\[
R_\theta(\eta;h)\xi
=
\frac{1}{\sqrt r}
\sum_{j=1}^{r}
F_j(h)\!\left[a_j(\eta)\,F_j(h)\xi\right],
\qquad r=8\mathbin{\times}320=2560.
\]

Here \(F_j(h)\) is a learned Fourier filter and \(a_j(\eta)\) is a real
surface-dependent function forced to satisfy \(a_j(\eta)=O(\eta^2)\). Thus
each channel performs four elementary operations:

1. filter \(\xi\);
2. multiply the result pointwise by a function computed from \(\eta\);
3. apply the same filter again;
4. add the channel to the other parallel channels.

The eight code-level “blocks” are not eight sequential layers. They only
divide the 2560 parallel channels into eight groups. The manuscript should not
present this grouping as a mathematical principle.

Using the same filter before and after multiplication gives the reciprocal
identity

\[
\int u\,F_j(a_jF_jv)\,dx
=
\int (F_ju)\,a_j(F_jv)\,dx
=
\int v\,F_j(a_jF_ju)\,dx .
\]

This is the elementary reason the current learned correction has the same
symmetry in its two potential arguments as the true DNO.

## Primary ablation: remove the second filter

Compare the current correction with

\[
R_\theta^{\mathrm{one}}(\eta;h)\xi
=
\frac{1}{\sqrt r}
\sum_{j=1}^{r}
a_j(\eta)\,F_j(h)\xi .
\]

This changes only step 3 above. It preserves:

- the exact \(G_0+G_1\) terms;
- linearity in \(\xi\);
- the constraint \(R_\theta=O(\eta^2)\);
- the surface-feature network and depth dependence;
- all \(8\times320\) channels;
- all 1,342,400 active trainable parameters;
- the initial function \(G_0+G_1\), because the learned correction is
  initialized to zero.

It removes the second learned filtering operation and, consequently, the
exact reciprocal identity. For this reason it should be called the
**second-filter deletion** or **one-filter ablation**, rather than described
as a pure symmetry ablation.

Train the two models freshly for the same number of optimizer steps, using the
same initialization seed, data order, loss, optimizer, and evaluation panels.
Do not fine-tune the one-filter model from a trained two-filter checkpoint.
Compare:

- one-step validation error;
- terminal surface-error median and upper quantiles;
- raw and translation-aligned errors;
- nonfinite and divergent rollout counts;
- reciprocal-identity defect;
- training and inference time.

Interpretation:

- If the one-filter model is noninferior, prefer it because it is simpler and
  cheaper.
- If it is materially worse, retain the two-filter model and use this result
  as direct evidence that the reciprocal construction improves learned DNO
  rollouts.

No previous tracked experiment removed this second learned filter. Earlier
cutoff, output-cap, multiplier-tying, feature, and block-count experiments are
different questions.

## Secondary ablation: number of parallel channels

Only after resolving the primary question, test whether all eight groups are
needed. Changing from eight groups to one while retaining 320 channels per
group changes the parameter count from

\[
1{,}342{,}400 \quad\text{to}\quad 259{,}360.
\]

This is a capacity and compression experiment, not a test of the mathematical
form. An older three-epoch experiment found that one group increased
validation loss by about \(9\%\) while reducing wall time by about a factor of
five, but it did not use the current \(G_0+G_1+O(\eta^2)\) recipe or receive a
long-rollout evaluation.

## Preferred manuscript language

Avoid “self-adjoint sandwich” and avoid treating the eight implementation
groups as depth. A concise description is:

> The learned correction is a sum of parallel channels. In each channel we
> filter the surface potential, multiply it by a coefficient computed from the
> surface profile, and apply the same filter once more. Reusing the filter on
> both sides guarantees the reciprocal identity of the DNO.
