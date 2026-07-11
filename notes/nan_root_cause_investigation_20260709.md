# NaN root-cause investigation, 2026-07-09

## Conclusion

The remaining Tanaka NaNs are not primarily caused by corrupt training rows,
insufficient pointwise data coverage, the GL2 time step, or a lack of model
parameters. The learned operator has the right values on the truth manifold but
the wrong derivative transverse to that manifold. In particular, it does not
satisfy the Dirichlet--Neumann shape derivative identity, so the two equations
used in the surrogate rollout are not the canonical derivatives of one learned
Hamiltonian.

The failure chain is:

1. A small rollout-state error enters the learned operator in a transverse
   direction that was not constrained by the pointwise supervised loss.
2. The model produces excessive same-sign high-mode sidebands, initially most
   visible in `k=32..64` and later in `k=64..128`.
3. The model response becomes thousands of times larger than its original
   truth-manifold bias while its local secant gain rises to about 7--8.
4. Base-grid products in the quadratic Zakharov RHS alias strongly only after
   this cascade is already established.
5. The four Picard iterations in GL2 cease to be contractive and produce the
   terminal nonfinite state.

Thus GL2 and RHS aliasing amplify and expose the failure, but the learned shape
derivative is the initiating defect.

## 1. Truth-manifold accuracy does not predict rollout stability

The aligned archives

- `outputs/cs_dno_w512b8_l256_v9_h100x2_20260703_035745/eval_suite_f64h/tanaka_g0_onestep.npz`
- `outputs/cs_dno_w512b8_l256_v9_h100x2_20260703_035745/eval_suite_f64h/tanaka_g0_trajs.npz`

contain model evaluations on exact truth states and on the model's own rollout
states, respectively. Define

\[
b_n = G_\theta(z_n^*)-G(z_n^*),\qquad
i_n = G_\theta(z_n)-G_\theta(z_n^*),
\]

and the measured secant gain

\[
q_n = \frac{\lVert i_n\rVert_2}{
              \lVert z_n-z_n^*\rVert_2}.
\]

The eventual failures remain extremely accurate when evaluated on truth
states: the median one-step relative `gxi` error is `1.87e-4` for case 5 and
`1.72e-4` for case 11. It is essentially constant in time. The response to the
rollout perturbation is not constant:

| Case | Time | state RMS error | truth-state bias RMS | induced response RMS | secant gain | induced / bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.8 | `1.62e-6` | `5.12e-6` | `5.56e-6` | 3.43 | 1.09 |
| 5 | 58.4 | `2.08e-3` | `5.13e-6` | `1.24e-2` | 5.95 | `2.41e3` |
| 5 | 66.4 | `1.67e-2` | `5.12e-6` | `1.35e-1` | 8.11 | `2.64e4` |
| 11 | 142.4 | `3.84e-3` | `3.15e-6` | `2.71e-2` | 7.04 | `8.60e3` |
| 11 | 150.4 | `9.02e-3` | `3.15e-6` | `7.59e-2` | 8.42 | `2.41e4` |
| healthy 0 | 200 | `5.91e-5` | `1.88e-7` | `1.96e-5` | 0.33 | 104 |
| healthy 6 | 200 | `3.76e-4` | `1.56e-6` | `6.71e-4` | 1.78 | 429 |

At the earlier pre-cascade frames of cases 5 and 11, the induced response is
concentrated in `k=32..64`. At the last finite frames it has spread through
`k=64..128`. This is direct evidence for a transverse/tangent instability, not
for missing pointwise labels along the truth trajectories.

## 2. The missing constraint is the DNO shape derivative identity

For a genuine DNO, let

\[
B = \frac{G(\eta)\xi+\eta_x\xi_x}{1+\eta_x^2},
\qquad V=\xi_x-B\eta_x.
\]

For every surface perturbation `zeta`, the Hadamard identity is

\[
D_\eta G(\eta)[\zeta]\xi
=-G(\eta)(\zeta B)-\partial_x(\zeta V).
\]

Equivalently, if

\[
K_\theta(\eta,\xi)=\frac12\langle\xi,
G_\theta(\eta)\xi\rangle,
\]

then the geometric Zakharov formula for `xi_t` must equal
`-delta K_theta / delta eta`. C2 is linear and self-adjoint in `xi`, so this is
a direct consistency check on the actual learned operator.

The following table reports the relative difference after the production
`k<=128` projection. `truth` means the model was evaluated at the exact truth
state; `pred` means the corresponding model-rollout state.

| Case / time | C2 value error on truth | C2 shape defect at truth | C2 shape defect at rollout | order-6 shape defect at truth |
| --- | ---: | ---: | ---: | ---: |
| healthy 0 / 200.0 | `6.47e-5` | 0.0896 | 0.2966 | `2.0e-9` |
| 5 / 33.6 | `2.34e-4` | 1.394 | 4.975 | `1.43e-5` |
| 11 / 189.6 | `2.15e-4` | 0.502 | 1.117 | `3.43e-6` |
| 22 / 45.6 | `5.54e-4` | 2.724 | 7.481 | `3.82e-6` |
| finite-divergent 27 / 200.0 | `2.73e-2` | 0.247 | 0.330 | `3.22e-6` |

The defect is already large before visible failure: it is 15.55 for case 5 at
`t=16`, 6.80 for case 11 at `t=160`, and 5.89 for case 22 at `t=32` on rollout
states. The exact order-6 implementation passes the same discrete check to
roughly `1e-5`, ruling out a normalization or differentiation artifact in the
diagnostic.

This also explains why GL2 does not supply the expected long-time protection.
GL2 is symplectic for the Hamiltonian vector field to which it is applied. The
rollout currently uses `eta_t=G_theta(eta)xi` but combines it with the exact
geometric formula for `xi_t`. Unless the learned operator obeys the identity
above, those two components are not derivatives of one Hamiltonian.

## 3. A concrete spectral mechanism: the missing first-order null form

The C2 transfer-kernel artifact

`outputs/c2_stage_match_gain_from_v85b_20260707_053707/transfer_kernel_truth_eta_case5_frame40/summary.json`

shows that, even at the truth surface, the learned operator maps the positive
mode `k_in=117` into the positive sideband `k_out=115` with raw FFT amplitude
`985.67`, while the order-6 reference amplitude is `1.47e-5` (ratio
`6.7e7`). This is not a large diagonal error; the flat `G0` diagonal there is
about `5.9e4`. It is a small but persistent spurious sideband, approximately
1.7% of the diagonal response, injected at every RHS evaluation.

The exact first Craig--Sulem correction is

\[
G_1(\eta)\xi=-G_0(\eta G_0\xi)-\partial_x(\eta\partial_x\xi).
\]

Its Fourier kernel is

\[
\left(k k' - g_0(k)g_0(k')\right)\widehat\eta_{k-k'}.
\]

In deep water this is exactly zero for same-sign `k,k'`. At the Tanaka depths
and `k` near 115, `tanh(h|k|)` is already indistinguishable from one, so the
same cancellation applies. The current learned blocks can approximate this
cancellation through multiple features and blocks, but do not enforce it. The
block-7 ablation and the failed `eta_xx` ablation show that the trained model is
using fragile inter-block and inter-feature cancellations instead.

On all C2 Tanaka truth snapshots, the analytic baselines have pointwise errors:

| Baseline | median relative `gxi` error | p95 |
| --- | ---: | ---: |
| `G0` | 0.1496 | 0.2089 |
| exact `G0+G1` | 0.00555 | 0.01373 |

Thus `G1` supplies roughly 90--97% of the nonlinear correction and, more
importantly, its high-frequency null structure exactly. A full CPU f64-harness
Tanaka rollout with exact `G0+G1`, no learned residual, and no output cap gave
`0/32` NaNs, `0/32` divergences, final eta median `0.2071`, and p95 `0.4096`.
It is stable but not accurate enough by itself. This is the appropriate stable
backbone around which to learn the remaining approximately one-percent
operator correction.

The experiment history does not contain a clean full-capacity test of that
architecture. The June 24 `G1` launch was killed before training, and the old
v3 and v11 variants changed several other architecture and regularization
variables simultaneously.

## 4. RHS dealiasing is required, but is not the cure

JCP09 Section 3.3 states that aliasing occurs both in the DNO and in the
equations of motion, and evaluates nonlinear terms in the equations of motion
on spectra extended by a factor of two. Our order-6 reference DNO uses padding,
but the quadratic Zakharov products are still evaluated at the base grid in
`solver/evals/model_rollout.py:268-290` and
`solver/solvers/time_integrator.py:211-231`.

A frame-by-frame comparison of the retained `k<=128` `xi_t` from current
base-grid products and 2x-padded products gives:

| Case | last finite time | relative RHS alias difference | first `>1e-3` | first `>0.1` |
| --- | ---: | ---: | ---: | ---: |
| healthy 0 | 200.0 | `2.71e-11` | never | never |
| healthy 6 | 200.0 | `1.78e-11` | never | never |
| 5 | 33.6 | 0.753 | 32.0 | 33.6 |
| 11 | 189.6 | 0.819 | 187.2 | 188.8 |
| 22 | 45.6 | 0.0829 | 45.6 | never |
| finite-divergent 27 | 200.0 | `7.27e-10` | never | never |

Replaying from each last finite C2 state confirms causality. Two-times
dealiasing leaves a healthy trajectory identical to about `3e-15` relative,
but only delays terminal failure:

| Case | base-grid first nonfinite | 2x-dealiased first nonfinite | delay |
| --- | ---: | ---: | ---: |
| 5 | 33.72 | 33.77 | 0.05 |
| 11 | 189.71 | 189.79 | 0.08 |
| 22 | 45.89 | 45.94 | 0.05 |

The JCP09 dealiasing rule should be implemented as a correctness fix and will
remove a selective terminal amplifier. It cannot repair a state that has
already accumulated the learned transverse error.

## 5. Why the recent remedies did not solve it

### More or targeted data

Random and source-stratified scans of v8/v9 found no nonfinite base rows. The
older coverage audit also placed the evaluation distributions inside the
training distribution except for modest tails. More decisively, the model is
already accurate on the exact failing truth trajectories. The missing object is
the derivative of the map around those trajectories, not another copy of the
same pointwise labels.

The C6/C7 pack was actively harmful because it used model-generated
pre-cascade states and contaminated spectral targets. Clean truth hard
negatives are not intrinsically invalid, but they are an inefficient surrogate
for enforcing the identity that all nearby states should satisfy.

### Smaller time steps and more Picard work

Earlier `80 -> 160 -> 320` substep sweeps moved failure times only slightly.
The inner traces and the replay above show that GL2 becomes noncontractive only
after the spectral state is already contaminated. Step halving remains a good
fail-safe after a residual check, not a model cure.

### Static caps and cutoffs

The C14/C15 full-output cap is not an absolute bound. It enforces

\[
\lVert g_{hi}\rVert\leq
\max(5,0.1\lVert g_{lo}\rVert).
\]

The dangerous `k=16..32` band is in the unbounded low part, and growth of that
part raises the permitted high-band output. The unnormalized FFT floor of 5 is
also grid-dependent and much larger than the physical deep-water tail. At the
last finite capped C14 frames, cases 3 and 5 are pinned near `hi/lo=0.1`; case
27 has `hi=2.81<5`, so the cap is completely inactive even though
`hi/lo=1.80`. C15 uses the same mechanism and cannot close this loophole by
adding blocks.

C15 itself stopped at epoch 1 step 165/5964, before a checkpoint or validation
result. It is interrupted and inconclusive as a capacity comparison, but the
mechanism is rejected by the frame evidence and should not be relaunched.

### Existing GL2 stage regularization

C10 was directionally related and improved Tanaka from `3/32` to `2/32` NaNs,
but it regularized an indirect one-Picard stage secant rather than the defining
DNO shape identity. The current implementation also has correctness
mismatches:

- `compute_stage_reg(..., dtype=compute_dtype)` makes the nominal order-6,
  pad-8 reference fp32 in every C2/C8/C10 run;
- `_promote_state` exists but is unused;
- the learned stage map does not perform the per-sample output mean removal
  used by production prediction;
- the perturbation size is absolute after unit-energy normalization rather
  than relative to the state scale.

These do not prove that a corrected stage loss would fail, but the direct shape
identity is cheaper, more diagnostic, and easier to explain.

## 6. Recommended durable model

Use a full-capacity, null-structured Hamiltonian CS-DNO:

\[
G_\theta(\eta)=G_0+G_1(\eta)+R_\theta(\eta),
\qquad R_\theta(\eta)=O(\eta^2).
\]

The requirements are:

1. Use the exact, uncut `G1` implementation in fp64 spectral arithmetic.
2. Make the learned residual vanish through first order at `eta=0`, so it
   cannot relearn and destroy the exact `G1` null form.
3. Retain the full eight-block capacity initially; the C14 one-block reduction
   was a smoke configuration, not evidence that fewer parameters are better.
4. Keep the residual self-adjoint in `xi`.
5. Replace unbounded raw derivative channels, especially raw `eta_xx`, with
   Sobolev-normalized/order-zero features, or otherwise constrain the learned
   residual to remain a first-order operator. Do this in a clean retrain rather
   than zeroing a feature in an existing checkpoint, because the current
   checkpoint relies on fragile learned cancellations.
6. Dealias nonlinear learned products and the Zakharov RHS; apply the production
   `k<=128` projection after padded products.

Train it with the ordinary clean v9 data plus a randomized Hadamard consistency
loss, not case-specific hard negatives. For a relative perturbation `epsilon`
and a Sobolev-scaled probe `zeta`, use the finite secant

\[
S_\theta=
\frac{G_\theta(\eta+\epsilon\zeta)\xi-G_\theta(\eta)\xi}{\epsilon}
+G_\theta(\eta)(\zeta B_\theta)
+\partial_x(\zeta V_\theta),
\]

and add

\[
\mathcal L
=\mathcal L_{data}
+\lambda_{shape}\,
\mathbb E_\zeta
\frac{\lVert W P_{\leq128} S_\theta\rVert_2^2}
     {\lVert W P_{\leq128}
       [G_\theta(\eta)(\zeta B_\theta)+
        \partial_x(\zeta V_\theta)]\rVert_2^2+\delta}.
\]

Here `W` is the same Sobolev weighting used by the supervised loss. Sampling a
mixture of low and `k=32..128` probes estimates a universal operator identity;
it is not Tanaka-specific hard-negative fitting. A finite secant avoids a
second derivative through the whole training loss. On a small microbatch every
four optimizer steps it should be materially cheaper than the current
order-6/pad-8 GL2 stage regularizer.

## 7. Minimal experiment sequence

1. Implement 2x-padded Zakharov RHS products and verify bit-level healthy-case
   parity plus the last-finite replay. Keep it regardless of the model result.
2. Add a read-only `shape_defect` evaluation metric. Confirm the exact order-6
   reference stays below `1e-4` and reproduce the C2 defect table above.
3. Implement exact `G0+G1 + O(eta^2) residual` at full capacity and run a short
   supervised smoke. Do not use a cap, hard `k` cutoff, PSD hinge, or targeted
   Tanaka pack.
4. Add the finite-secant Hadamard loss in f64 on a small periodic microbatch.
   Calibrate `lambda_shape` by the normalized loss and gradient norm rather
   than recycling the C10 stage-loss weight.
5. Full retrain on clean v9, then evaluate without runtime damping.

Acceptance requires all of:

- Tanaka `0/32` NaNs and `0/32` divergences;
- scored BF `0/31` NaNs and divergences;
- no material median or p95 regression relative to C2/C10;
- failing-case truth-state shape defect reduced by at least two orders of
  magnitude, with no pre-cascade secant-gain growth toward 7--8;
- GL2 fixed-point residual checked at every step, with adaptive halving only as
  a reported safety fallback.

This is a structural and journal-explainable correction: it restores the
first-order Craig--Sulem null form and the Hamiltonian compatibility that the
integrator assumes, instead of suppressing the spectral symptom after it has
appeared.
