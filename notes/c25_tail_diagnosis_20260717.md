# C25 fixed-Tanaka tail diagnosis

## Question

The fixed C25 evaluation contains no nonfinite surrogate trajectory. The
remaining question is therefore not why C25 produces NaN, but why six of its
64 finite Tanaka trajectories have terminal relative elevation error greater
than 0.25, with one as large as 1.142. We tested three candidate mechanisms:

1. failure of the four-sweep implicit GL2 stage solve;
2. growth of unresolved mid/high Fourier modes;
3. a small error in the learned vector field
   $q_\theta=G_\theta(\eta)\xi$ that accumulates along the translation orbit
   or deforms the wave.

The evaluated model is the epoch-39 best validation checkpoint from
outputs/c25_capacity125_full_20260716_022603, with checkpoint payload SHA-256
a9d5dbbdd65f04983646ac969db0099c036e2151eac8a1933a58eaf496420140.
The protocol uses $T=200$, saved-frame spacing 0.8, 80 inner steps per saved
interval ($\Delta t=0.01$), four GL2 Picard sweeps, a hard $|k|\le128$
filter, and the float64 integration harness around the float32 network.

## Translation and deformation

For each saved time, define the fitted periodic displacement

\[
 d(t)\in\arg\min_{d\in\mathbb R/(2\pi\mathbb Z)}
 \|\eta_\theta(t,\cdot+d)-\eta_6(t,\cdot)\|_2,
\]

and the raw and translation-aligned errors

\[
 E_{\rm raw}(t)=
 \frac{\|\eta_\theta(t)-\eta_6(t)\|_2}{\|\eta_6(t)\|_2},\qquad
 E_{\rm shape}(t)=
 \frac{\|\eta_\theta(t,\cdot+d(t))-\eta_6(t)\|_2}
      {\|\eta_6(t)\|_2}.
\]

The continuous Fourier registration was applied at every one of the 251 saved
frames, not just at $T=200$. Five of the six cases with terminal
$E_{\rm raw}>0.25$ are translation dominated at the first 0.25 crossing. A
displacement of one grid cell precedes that crossing by a median of 36.0 time
units in g0 and 28.8 in g1. The failures grow smoothly; they do not appear as a
sudden terminal event.

| Family / index | $E_{\rm raw}(200)$ | $E_{\rm shape}(200)$ | squared error removed by one shift | fitted shift (grid cells) |
|---|---:|---:|---:|---:|
| g0 / 4  | 0.27235 | 0.00665 | 99.94% | -1.93 |
| g0 / 21 | 0.21565 | 0.03614 | 97.19% | +3.48 |
| g0 / 27 | 0.66399 | 0.04401 | 99.56% | -3.55 |
| g1 / 11 | 0.36204 | 0.01229 | 99.88% | -2.13 |
| g1 / 16 | 0.33675 | 0.01126 | 99.89% | +5.21 |
| g1 / 24 | 1.14192 | 0.58067 | 74.14% | -15.60 |
| g1 / 29 | 0.69261 | 0.02051 | 99.91% | -4.19 |

Thus g1/24 is qualitatively different from the other large-error cases: it
contains both a large translation and a large deformation. Treating every tail
as one rigid phase error is false.

## Why G(eta)xi drives the measured shift

Write $a(x,t)=\eta_\theta(x+d(t),t)$, $e=a-\eta_6$, and
$S_df(x)=f(x+d)$. At a smooth isolated least-squares minimizer,
$\langle e,a_x\rangle=0$. Differentiating this stationarity equation and using
$\eta_t=q=G(\eta)\xi$ gives the exact identity

\[
 d'(t)=-\frac{
 \langle S_dq_\theta-q_6,a_x\rangle
 +\langle e,S_d(q_\theta)_x\rangle}
 {\langle a_x,a_x\rangle+\langle e,a_{xx}\rangle}.
\]

This identity was evaluated from the saved fields. For g0/27, its velocity has
correlation 0.973 with the finite-difference velocity of the fitted shift, and
its time integral closes the terminal shift to 0.031 grid cells. For g1/24 the
correlation is 0.9994, with closure error 0.011 grid cells. The other five
focused correlations are 0.943, 0.222, 0.952, 0.998, and 0.959; even the noisy
g0/21 case has terminal integral closure within 0.102 grid cells.
Consequently, the translation coordinate is propagated by the discrepancy in
the saved $q$ fields. It is not an artifact of looking only at the final
pictures.

The total discrepancy $S_dq_\theta-q_6$ above contains both the same-state
operator error and the physical response to the already separated states. To
isolate the first term, we also evaluated C25 on every archived truth state:

\[
 \delta q_*(t)=G_\theta(\eta_6(t))\xi_6(t)
              -G_6(\eta_6(t))\xi_6(t).
\]

For $P_8$, the projection to nonzero modes $|k|\le8$, define

\[
 \delta c_8(t)=
 -\frac{\langle P_8\delta q_*(t),P_8\partial_x\eta_6(t)\rangle}
        {\|P_8\partial_x\eta_6(t)\|_2^2}.
\]

The sign-coherence ratio

\[
 \frac{\left|\int_0^{200}\delta c_8(t)\,dt\right|}
      {\int_0^{200}|\delta c_8(t)|\,dt}
\]

is at least 0.976 in all seven cases. Hence the truth-orbit tangent defect is
persistent rather than a large norm caused by sign-cancelling noise. However,
the sign of its integral agrees with the eventual fitted displacement in only
four of seven cases. In g0/27 every teacher-forced mode $1,\ldots,8$ predicts a
positive displacement while the free rollout ends at a negative displacement.
The bounded operator defect is therefore a persistent seed, but nonlinear
state separation can reverse or redistribute its effect. This directly
explains why a larger scalar tangent weight is not a universal cure.

## The GL2 stage solve is not failing

At every saved predicted state in the seven rows above (1,757 states total), we
replayed one production-size $\Delta t=0.01$ substep with four and eight Picard
sweeps. Let $r_4$ be the relative change of the GL2 stage on its fourth sweep.
The maximum over all states is

\[
 \max r_4=8.21\times10^{-8},
\]

and no state has $r_4\ge10^{-6}$. Moreover,

\[
 \max\frac{\|z^{(4)}_{n+1}-z^{(8)}_{n+1}\|}
               {\|z^{(8)}_{n+1}-z_n\|}
 =5.50\times10^{-6},\qquad
 \max\frac{\|z^{(4)}_{n+1}-z^{(8)}_{n+1}\|}
               {\|z^{(8)}_{n+1}\|}
 =1.13\times10^{-7}.
\]

These differences cannot explain terminal errors between 0.2 and 1.14.
Increasing the Picard iteration count is not a plausible remedy. The archive
has no internal-stage telemetry, so this test samples states every 0.8 time
units rather than every inner step. Smooth error growth, the uniformly tiny
sampled residuals, and the earlier same-GL2 full-defect oracle together make a
hidden intermittent stage failure very unlikely.

## There is no spectral cascade

The active evaluation guard monitors the elevation amplitude in
$64\le|k|<128$ and begins damping only after that amplitude exceeds both
$5\times10^{-7}$ and 100 times its initial value. In the seven cases, the
largest saved amplitude divided by this trigger lies between 0.0100 and
0.0143. The sigmoid gate is therefore effectively the identity throughout
these trajectories.

For g0/27, terminal predicted-to-truth elevation energy ratios in bands
$1{:}8$, $9{:}16$, $17{:}32$, $33{:}64$, and $65{:}128$ are respectively
0.998, 0.970, 0.959, 0.921, and 0.921. For g1/24 they are 0.885, 1.105, 0.831,
0.903, and 0.731. This is not explosive transfer to high modes. After
translation alignment, 77.3% of the g1/24 residual elevation energy below mode
128 lies in modes 9--32, confirming differential phase and shape error at
resolved modes rather than a cutoff-scale instability.

## Invariant and Hadamard evidence

The terminal absolute learned-Hamiltonian drift is strongly associated with
terminal raw elevation error across the fixed panel. Pearson/Spearman
correlations are 0.867/0.843 for g0 and 0.902/0.724 for g1. At $t=8$, the
Spearman correlations with terminal raw error are already 0.717 and 0.630.
The seven final signed drifts are

\[
 0.0083,\ 0.0174,\ -0.0448,\ 0.0116,\ -0.0210,\ -0.0733,\ 0.0227.
\]

This is evidence that vector-field consistency is relevant, but it is not a
proof that energy drift causes every phase error. An energy-conserving vector
field may still have the wrong frequency, and g1/29 has large translation
despite only moderate drift.

C25's network is linear and self-adjoint in $\xi$, so
$q_\theta=\delta_\xi K_\theta$ for

\[
 K_\theta(\eta,\xi)
 =\tfrac12\langle\xi,G_\theta(\eta)\xi\rangle.
\]

The production formula for $\xi_t$ is the exact water-wave formula and agrees
with $-\delta_\eta K_\theta-g\eta$ when the learned DNO has the correct
Hadamard shape derivative. C25 penalizes a randomized finite-secant version of
that identity only every 16 minibatches, on eight samples, with weight 0.01.
That is 2,984 one-direction probes per epoch, only 0.049% as many probes as
training rows. At epoch 39, the aggregate randomized relative defect RMS is
0.0882.

This aggregate number alone does not identify the Hadamard defect as the cause
of the tail:

- C16 already used the two-term H1 plus Hadamard objective at the smaller
  architecture. It removed all 320 observed nonfinites but retained one finite
  phase divergence in each Tanaka family.
- C16 simultaneously introduced the structured $G_0+G_1+O(\eta^2)$
  architecture and the dealiased right-hand side, so its result does not
  isolate the loss.
- The aggregate Hadamard metric is not monotone with the measured tails: it is
  0.0968 for C16, 0.0895 for C20, and 0.0882 for C25, while C25 worsens the g1
  extreme.
- The current Hadamard loss uses sampled finite secants, sampled states and
  directions, and projected modes. No uniform defect bound or stability result
  connects this scalar metric to the long rollout.

We therefore evaluated the same finite-secant residual on 17 evenly spaced
truth states per trajectory and 16 fixed random directions per state: 17,408
held-out probes. The split-half case-ranking correlation between the first and
second eight directions is 0.995, so Monte Carlo noise is not controlling the
result. The path-mean defect ranks four of the six raw-error tails in its top
six and five in its top sixteen; its tail-versus-nontail ROC AUC is 0.876.
However, it misses g1/16 badly (rank 41 of 64). Its Spearman correlations with
terminal raw, aligned, translation, and absolute-displacement errors are
0.408, 0.487, 0.363, and 0.200. Thus it is more informative about residual
shape error than about the translation coordinate itself.

## The direct supervised norm is the simpler missing issue

C25's main supervised term is the relative $H^1$ error

\[
 {\cal E}_{H^1}(q_\theta,q_6)=
 \left(
 \frac{\sum_{k\ge0}(1+k^2)|\widehat{q_\theta-q_6}(k)|^2}
      {\sum_{k\ge0}(1+k^2)|\widehat{q_6}(k)|^2}
 \right)^{1/2}.
\]

We evaluated C25 on all 251 saved truth states for all 64 cases. The path-mean
$H^1$ error has Spearman correlations -0.322, -0.206, and -0.330 with terminal
raw, aligned, and translation error. Its tail ROC AUC is only 0.681. The
g1/16 failure, whose raw error is 0.337 and whose error is almost entirely a
translation, has the smallest path-mean $H^1$ error of all 64 cases.

The same predictions tell a different story in the unweighted relative
$L^2$ norm:

\[
 {\cal E}_{L^2}(q_\theta,q_6)=
 \frac{\|q_\theta-q_6\|_2}{\|q_6\|_2}.
\]

Its path-mean correlations with terminal raw, aligned, and translation error
are 0.618, 0.791, and 0.552, and its tail AUC is 0.925. Restricting this
diagnostic to modes 1--8 gives correlations 0.683, 0.731, and 0.629 and AUC
0.963. This is already visible at the initial truth state: the full-$L^2$
score has raw/aligned/translation correlations 0.527/0.594/0.482 and tail AUC
0.920, while the initial $H^1$ score has correlations
-0.265/-0.201/-0.266 and AUC 0.644.

This is not only a shallow-depth ordering effect. After rank-residualizing
against family and depth, the $H^1$ correlations with raw/aligned/translation
error are -0.113/0.049/-0.141, whereas full $L^2$ gives
0.405/0.679/0.328 and modes-1--8 $L^2$ gives 0.493/0.566/0.433.

The Hadamard score does contain information beyond the current $H^1$ score:
after controlling ranks for family, depth, and $H^1$ error, its partial
correlations with raw/aligned/translation error are
0.258/0.302/0.226. But after controlling within-family ranks for full-$L^2$
error, the corresponding exploratory residual correlations are only
0.030/-0.009/0.047; controlling for modes-1--8 $L^2$ gives
-0.087/0.021/-0.075. In other words, the tail-specific signal in the
Hadamard audit is almost entirely shared with the low-frequency value error
that the present $H^1$ objective underweights.

This norm mismatch has a direct dynamical interpretation. Since
$\eta_t=q$, the same-state local elevation defect satisfies

\[
 \|\delta\eta(t+\Delta t)\|_{L^2}
 =\Delta t\,\|\delta q(t)\|_{L^2}+O(\Delta t^2)
\]

before state separation. The rollout is also judged primarily in relative
$L^2(\eta)$. By multiplying mode $k$ by approximately $k$, the current
$H^1$ loss allows phase-critical low-mode value errors to rank below
high-frequency discrepancies that do not drive these tails. The mode and
tangent terms were attempts to patch that mismatch with selected projections;
the cross-geometry failures show why those patches do not define one stable
objective.

## Conclusion and next experiment

The evidence rejects another projected phase/tangent loss and rejects an
integrator or high-frequency patch. C25's remaining problem is the accuracy of
the complete learned vector field along the trajectory. Most failures express
that error as coherent translation; g1/24 also shows resolved-mode
deformation. A fixed projection cannot cover both geometries, and the
teacher-forced sign reversals explain why such terms have repeatedly moved the
tail between cases.

The simplest principled candidate is not a fifth term or another calibrated
weight. It is a fresh C25-capacity retrain with the two-term objective

\[
 {\cal L}(\theta)
 ={\cal E}_{L^2}\!\left(G_\theta(\eta)\xi,G_6(\eta)\xi\right)
 +0.01\,{\cal L}_{\rm Had}(\theta).
\]

This means setting the existing Sobolev order from 1 to 0, removing both the
mode-balanced and localized-tangent terms, and initially retaining the
existing sparse Hadamard weight, interval, and microbatch. The direct term then
matches the state component and norm whose error actually accumulates, while
Hadamard retains the structural shape-derivative constraint associated with
the zero-nonfinite C16 result. Keeping its schedule fixed avoids conflating the
norm test with a 16-fold change in shape-probe coverage.

This proposal is supported by a ranking audit, not yet by an intervention.
Run a 1%-v9 smoke for finiteness and optimization scale, then train fresh rather
than presenting another targeted Tanaka fine-tune as a final model. The
predeclared gate remains the fixed 64-case raw/aligned evaluation. A clean
large-capacity $H^1+$Hadamard control would be needed later for a strict
one-variable manuscript ablation, but it need not block testing the simpler
candidate now. No loss-gradient calibration is needed.

## Artifacts

- Framewise translation analysis and definitions:
  notes/c25_tanaka_translation_decomposition_20260717/README.md
- All-64 and framewise arrays:
  notes/c25_tanaka_translation_decomposition_20260717/
- Truth-path same-state DNO audit:
  notes/c25_teacher_forced_gxi_fixed64_20260717.json and .npz
- Held-out Hadamard ranking audit:
  notes/c25_hadamard_tail_ranking_20260717.json and .npz
- Four-versus-eight-sweep GL2 replay:
  notes/c25_gl2_residual_probe_20260717.json and .npz
- Implementations:
  scripts/analyze_rollout_translation_decomposition.py,
  scripts/audit_teacher_forced_gxi.py, and
  scripts/audit_hadamard_tail_ranking.py,
  scripts/probe_c25_gl2_residuals.py
