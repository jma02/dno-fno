# Tier 3H: linear-exact integrating-factor split for the Zakharov rollout

Doc reference: `notes/cs_dno_jcp09_improvement_plan.md` §3H; JCP09 §3.2 (eq 17-21).

## Goal

Decouple the high-`k` linear-dispersion band — which the model already gets exactly via the `G_0 ξ` baseline — from time-accumulated truncation error in the outer integrator. Equivalent to: time-step *only* the nonlinear residual `(G - G_0) ξ`, keep `G_0` evolution analytic.

Eval-side change; no retrain. Implementable against the existing v8 ckpt.

## Math

Linearize around `η = 0`. The Zakharov system reduces to

```
∂_t η̂_k = G_0(k, h) · ξ̂_k = |k| tanh(h|k|) · ξ̂_k
∂_t ξ̂_k = −g · η̂_k
```

Eigenvalues `λ² = −g · G_0`, frequencies `ω_k = √(g · |k| tanh(h|k|))` (deep-water gravity-wave dispersion). For `k > 0` the per-mode linear flow operator is the 2×2 rotation-like matrix

```
Φ_k(t) = [ cos(ωt)             (G_0/ω) sin(ωt) ]
         [ −(g/ω) sin(ωt)      cos(ωt)         ]
```

At `k = 0` the matrix degenerates: `η̂_0` is constant, `ξ̂_0(t) = ξ̂_0(0) − g · η̂_0 · t`.

This is **already implemented** in `solver/evals/eval_suite.py:truth_rollout_linear_analytic` (lines ~280-310) — that code is the analytic flow for the *linear* truth. We reuse the same closed form per substep.

## Splitting

Define `v̂ ≡ Φ_k(−t) û`. Then

```
∂_t v̂ = Φ_k(−t) · N(u)
```

where `N(u) = [(G(η) − G_0) ξ; nonlinear-ξ-RHS]` is the nonlinear residual. The model predicts `G(η) ξ` directly, so `(G − G_0) ξ = model_output − G_0 · ξ`.

The numerical scheme:

```
for substep i, t_i:
    1. û_i = Φ_k(t_i) · v̂_i                      (lift v to u)
    2. u_i = IFFT(û_i)                            (to physical)
    3. gxi = model(η_i, ξ_i)                      (model query)
    4. N_η = gxi − G_0 · ξ_i                      (subtract linear part)
       N_ξ = full nonlinear ξ-RHS (as today)
    5. N̂ = FFT([N_η, N_ξ])
    6. rhs_v̂ = Φ_k(−t_i) · N̂                    (project residual back to v frame)
    7. v̂_{i+1} = integrate(v̂_i, rhs_v̂, dt)      (RK4-IF / GL2 — anything;
                                                  the stiff linear part is gone)
```

Final state: `û_final = Φ_k(t_final) · v̂_final → IFFT → u_final`.

## Code changes

1. **`solver/solvers/time_integrator.py`** — add:
   - `linear_flow_matrix(k_grid, depth, gravity, t)` → returns Φ_k(t) for all k as a `(nx, 2, 2)` complex array (or four `(nx,)` arrays for the four matrix entries; latter is cleaner since Φ is real). Handle `k=0` branch.
   - `apply_linear_flow(eta_hat, xi_hat, k, depth, gravity, t)` → returns `(eta_hat', xi_hat')`.

2. **New rollout function** (`rollout_surrogate_lin_exact`) in `solver/solvers/time_integrator.py` or `solver/evals/model_rollout.py`:
   - Mirrors `rollout_surrogate` but operates on `v̂` between substeps.
   - Integrator choice: simplest is GL2 in `v` (existing structure already supports implicit iters via `gl2_iterations=impl`). The implicit step in `v` has no linear coupling so it's a plain fixed-point on the nonlinear residual.

3. **`solver/evals/eval_suite.py`** — add `--integrator={rk4_if, gl2_if, lin_exact}` flag. When `lin_exact`, dispatch to the new rollout. Per-IC jit signature unchanged.

4. **Diagnostic**: re-run v8 best ckpt eval with `--integrator=lin_exact --f64_harness` on tanaka_g0/g1 only. Compare:
   - NaN rate (does cascade survive at h ≥ 0.23?)
   - `err_high(t=2.4) / truth_high(t=2.4)` (JCP09-canonical diagnostic)

## Effort + dependencies

- ~250 LOC, ~1-2 days dev.
- No retraining needed — runs against any existing CS-DNO ckpt (v8, v8.5, etc.).
- Dependencies: existing `truth_rollout_linear_analytic` provides the closed-form Φ_k(t); reuse it.

## When to implement

Per the doc's decision tree (§4): implement only if A+B+C retrain (v8.5) still shows `err_high/truth_high > 0.1×` on cid5/cid11/cid1000006. Otherwise filter_gxi is sufficient and this is unnecessary complexity.

## Risk

- Subtle: `v` and `u` have the same DC mode (`Φ_0 = I` up to the `−g·η_0·t` shear). Need to verify the `k=0` branch doesn't drift across many substeps.
- The model is trained on `(η, ξ)` pairs, not `(η, ξ_v_frame)`. Step 1 lifts back to `u`-frame before model call so this is fine.
- Existing GL2 implicit iters apply to the full `(η, ξ)` state; we'd retain the implicit structure in `v̂` with a smaller stiff residual.
