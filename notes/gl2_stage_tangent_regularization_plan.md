# GL2 stage-tangent regularization implementation plan

Date: 2026-07-04

## 1. Objective

Prevent the learned `k≈64:128` feedback mode by constraining the local
sensitivity of the nonlinear GL2 stage map, rather than only fitting
`G_theta(eta) xi` at clean states.

For the two integrating-factor stages `V = (V1, V2)`, define one Picard map

```text
T_theta(V) = (v_n, v_n) + h (A tensor I) F_theta(V),
```

where `h=0.01`, `A` is the two-stage Gauss-Legendre matrix, and
`F_theta` is the integrating-factor nonlinear RHS using the learned DNO.

The regularizer should:

1. match the model and reference stage responses to small perturbations;
2. penalize stage gain above the reference gain;
3. cover both mid-to-mid and low-to-mid coupling;
4. act on both `eta` and `xi`;
5. run sparsely enough to keep training below approximately `1.5x` baseline
   wall time.

This is a training-time model fix. GL2 residual checking and rejected-step
retry are a separate runtime-containment change.

## 2. Loss

Keep the existing supervised loss and add

```text
L_total = L_data
        + lambda_match * warmup * L_stage_match
        + lambda_gain  * warmup * L_stage_gain.
```

For a stage perturbation `q`, use a finite secant initially:

```text
w_theta = [T_theta(V + eps q) - T_theta(V)] / eps
w_ref   = stop_gradient(
            [T_ref(V + eps q) - T_ref(V)] / eps
          )
```

Finite secants are preferred for the first implementation because they:

- use ordinary first-order backpropagation with respect to model parameters;
- exercise the finite perturbation amplitudes seen during rollout;
- avoid forward-over-reverse differentiation through a JVP;
- are easy to compare against an exact `jax.jvp` in tests.

Use the linear Hamiltonian norm

```text
||(deta, dxi)||_E^2
    = g ||deta||_2^2 + <dxi, G0 dxi>,
```

summed over both stages with GL2 weights `b1=b2=1/2`. The `xi` zero mode is
excluded. The exact linear flow preserves this norm.

Let `P_B` be a soft projector onto the unstable output band and `P_L` a
low-mode input projector. Initial defaults:

```text
P_B: roll on over |k|=24:32, full weight over 32:112,
     roll off over 112:128
P_L: |k| < 32 with a short complementary taper
```

The production state/RHS filter cuts at approximately `|k|=128`, so modes
above 128 are diagnostic rather than part of the primary stage-response norm.

Use two probe classes:

```text
mid-to-mid: q = P_B q_random
low-to-mid: q = P_L q_random, but score P_B w
```

Cycle through `eta`-only, `xi`-only, and joint probes, or sample these probe
types uniformly. Normalize every probe to unit energy norm before multiplying
by `eps`.

The matching term is

```text
L_stage_match =
    mean(
      ||P_B (w_theta - w_ref)||_E^2
      / (||P_B w_ref||_E^2 + response_floor)
    ).
```

The gain term is

```text
r_theta = ||P_B w_theta||_E / (||q||_E + eps_norm)
r_ref   = ||P_B w_ref||_E   / (||q||_E + eps_norm)

margin = gain_margin_rel * r_ref + gain_margin_abs

L_stage_gain = mean(relu(r_theta - r_ref - margin)^2).
```

Do not impose an unconditional `r_theta < 1` constraint. The reference flow
may have legitimate transient amplification; the model should match it.

Initial perturbation amplitude:

```text
eps ~ LogUniform(1e-6, 1e-3)
```

relative to the energy norm of the underlying physical state. This spans the
measured cid5 ignition window without starting from the overly strong
`1e-3:3e-2` perturbations used by the old perturbed-Tanaka fine-tunes.

## 3. Stage state construction

Treat each autonomous physical training state as a new local time origin:

```text
t_n = 0
v_n = u_n
```

The integrating-factor transformation can be restarted at each autonomous
step. Add a parity test showing that this local-origin construction is
equivalent, up to floating-point error and the expected conjugacy, to the
production absolute-time construction.

Use a reference-predicted stage center:

```text
V0    = (v_n, v_n)
V_bar = stop_gradient(T_ref(V0))
```

This costs one reference Picard update and places the regularizer closer to
the collocation stages than `V0`, without solving the complete reference
stage system. Evaluate both model and reference secants at `V_bar`.

If the first experiment shows that one predictor is insufficient, test two
reference Picard updates. Do not begin with a converged reference-stage solve.

The reference RHS must use:

- order-6 Craig-Sulem DNO;
- pad factor 8;
- the same RHS and post-RHS filter convention as production;
- float64 reference computation;
- per-example depth and `G0`;
- zero-mean `xi`.

## 4. Code organization

Add a focused helper module:

```text
train-jax-10m/stage_tangent_regularizer.py
```

It should contain pure functions for:

- GL2 coefficients;
- soft Fourier projectors;
- energy norm;
- probe construction and normalization;
- one model Picard map;
- one reference Picard map;
- secant responses;
- match/gain losses and diagnostic metrics.

Do not refactor the production integrator initially. Its trajectories are
numerically sensitive to operation ordering. Duplicate the small GL2
coefficient/map definition in the training helper and enforce parity through
tests.

Wire the helper into:

```text
train-jax-10m/1d_dno_fno_jax.py
```

Reuse the trainer's existing:

- model `apply_fn`;
- normalization and denormalization functions;
- physical `eta`, `xi`, and depth batch values;
- `k` grid;
- random-key folding across the `"batch"` device axis.

The regularizer must return zero without constructing its expensive branch
when its weight is zero.

## 5. CLI and configuration

Add these arguments with disabled defaults:

```text
--stage_reg_weight             0.0
--stage_reg_gain_weight        0.0
--stage_reg_interval           32
--stage_reg_microbatch         8
--stage_reg_warmup_steps       500
--stage_reg_k_lo               32
--stage_reg_k_hi               128
--stage_reg_eps_min            1e-6
--stage_reg_eps_max            1e-3
--stage_reg_gain_margin_rel    0.05
--stage_reg_gain_margin_abs    1e-3
--stage_reg_response_floor     1e-8
--stage_reg_reference_order    6
--stage_reg_reference_pad      8
--stage_reg_reference_picard   1
```

Persist every setting in `config.json`.

Interpret `stage_reg_microbatch` as the global microbatch size. Split it
across devices with a fixed static local shape. Require divisibility by device
count or fail with a clear error.

## 6. Sparse execution

Evaluate the regularizer only when

```text
optimizer_step % stage_reg_interval == 0.
```

Use a static-shaped `jax.lax.cond` so only the selected branch executes.
Select the microbatch deterministically from a random permutation of the
local batch; do not always take the first rows.

Initial schedule:

```text
global microbatch = 8
interval          = 32 optimizer steps
one mid probe + one low probe
warmup            = 500 optimizer steps
```

This is expected to add roughly 10–30% wall time. The implementation has a
hard go/no-go target of at most `1.5x` baseline wall time in a 200-step A/B.

If memory is excessive:

1. reduce the microbatch to 4;
2. use one probe class per selected step and alternate classes;
3. rematerialize model blocks in the secant branch;
4. increase the interval to 64.

Do not lower reference order or pad factor before trying those options.

## 7. Metrics

Log these metrics separately from the ordinary data loss:

```text
stage_reg_match
stage_reg_gain
stage_reg_model_gain_mean
stage_reg_model_gain_max
stage_reg_reference_gain_mean
stage_reg_violation_fraction
stage_reg_mid_to_mid
stage_reg_low_to_mid
stage_reg_wall_fraction
```

Also report the unweighted and weighted regularizer values. This prevents a
small configured coefficient from hiding an unstable raw gain.

Checkpoint selection must not use global validation loss alone. Add a small
fixed stability-validation batch containing:

- deep/steep Tanaka states;
- shallow broadband controls;
- early cid5/cid11-like precursor states if available;
- both mid-band and single-mode probes near `k=96:128`.

## 8. Tests

Add CPU tests with small grids and models.

### Mathematical tests

1. GL2 coefficients exactly match production values.
2. The training Picard map matches a manually evaluated production Picard
   update for the same RHS.
3. `apply_linear_flow_hat` preserves the discrete energy norm.
4. Local-time-origin and absolute-time stage constructions agree under the
   correct linear-flow conjugacy.
5. Projectors have the requested support/tapers and preserve real fields.
6. Probe normalization produces unit energy norm for `eta`-, `xi`-, and
   joint probes.

### Loss tests

1. `model == reference` gives matching loss near numerical zero.
2. Artificially multiplying the model's mid-band response increases both
   matching and gain losses.
3. A low-mode perturbation that generates a synthetic mid-band response is
   detected by the low-to-mid term.
4. Finite secants agree with `jax.jvp` over the selected epsilon range in a
   smooth test problem.
5. Loss and parameter gradients remain finite.
6. The regularizer produces a nonzero gradient on an intentionally unstable
   model.
7. With both weights zero, the original train loss and update are unchanged.

### Integration smoke tests

1. One CPU optimizer update with the regularizer enabled.
2. One multi-device update with correct RNG independence and reductions.
3. Save/resume preserves all configuration and optimizer behavior.
4. The expensive branch is skipped on non-interval steps.

Run available checks with:

```text
uv run ruff check <changed files>
uv run pyright <changed files>
uv run pytest <targeted tests>
```

If a tool is unavailable, run `py_compile` and the focused smoke script.

## 9. Experiment sequence

Log every experiment in `EXPERIMENTS.md`.

### Phase A: correctness

1. CPU mathematical/unit tests.
2. Small-model gradient smoke test.
3. Verify finite-secant/JVP agreement.
4. Confirm zero-weight numerical parity with baseline training.

### Phase B: cost calibration

Run matched 200-step jobs from the v8.5b tied checkpoint:

```text
A0: regularizer disabled
A1: microbatch=8, interval=32, match only
A2: microbatch=8, interval=32, match + gain
```

Record:

- steps/second;
- peak accelerator memory;
- regularizer wall fraction;
- data-loss trajectory;
- raw model/reference stage gains.

Stop if:

- wall time exceeds `1.5x` baseline;
- peak memory leaves insufficient headroom;
- gradients become non-finite;
- stage loss is dominated by the response floor.

### Phase C: short fine-tune

Fine-tune v8.5b for 3–5 epochs with a low learning rate. Use a matched
continued-training control so improvements are not attributed to extra
optimization alone.

Primary arms:

```text
C0: tied checkpoint continued, no stage regularizer
C1: stage matching only
C2: stage matching + gain hinge
```

Optional data comparison:

```text
C3: narrow k=64:128 perturbation data, no stage regularizer
```

Do not combine C2 and C3 until each has demonstrated an independent benefit.

### Phase D: rollout evaluation

Use the matched f64 harness and evaluate:

- all six saved Tanaka g0/g1 sets used by the 96-trajectory audit;
- the complete standard 10-regime suite for the winning arm;
- per-substep GL2 residual traces on cid5 and cid11;
- the truth-free `k=64:128` growth detector;
- model/reference stage-gain distributions.

## 10. Acceptance criteria

The change is accepted only if it satisfies all of:

1. zero NaNs on the current 96-trajectory diagnostic set;
2. zero truth-free spectral-growth alarms, or a clearly justified threshold
   recalibration on a held-out set;
3. no more than 10% regression in median or p95 Tanaka final error;
4. no material regression on shallow broadband and random-sea regimes;
5. fourth Picard residual stays within tolerance on traced rollouts;
6. model stage gain approaches reference stage gain in both probe classes;
7. training wall time is at most `1.5x` the matched control.

A reduction in NaNs accompanied by bounded-but-wrong rollouts is a rejection,
not a success.

## 11. Runtime containment follow-up

After the training experiment, implement a separate opt-in GL2 safeguard:

1. measure relative stage-update residual after every Picard iteration;
2. accept only if the residual is below tolerance and decreasing;
3. if it increases, reject the substep and retry with two half-steps;
4. cap retry depth and report a structured failure rather than emitting NaNs.

This safeguard should be evaluated with both the baseline and regularized
models. It protects production from catastrophic stage solves but must not be
used to claim that the learned mid-band instability has been cured.
