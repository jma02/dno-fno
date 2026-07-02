# CS-DNO improvement plan, grounded in JCP09 (Xu & Guyenne 2009)

**Goal.** Make CS-DNO a better surrogate for the Dirichlet–Neumann operator as
formulated in `JCP09.pdf` (Xu & Guyenne, *Numerical simulation of
three-dimensional nonlinear water waves*, JCP 228 (2009) 8446–8466).

**Scope of this note.** Findings from reading JCP09, the CS-DNO architecture
reference (`notes/cs_dno_architecture.tex`), the ablation study
(`notes/cs_dno_ablation_study.md`), the v5 Tanaka failure audit
(`notes/v7_tanaka_failure_audit.md`), and the current implementation
(`models/dno-net/dno_net_v2.py`, `solver/evals/model_rollout.py`,
`train-jax-10m/1d_dno_fno_jax.py`). This is an analysis/handoff document; no
code changes were made.

---

## 1. What JCP09 actually relies on

The paper's DNO is the Craig–Sulem Taylor series

```
G(η) = Σ_{j≥0} G_j(η),   G_j homogeneous of degree j in η,
```

with the **adjoint recursion (12)/(13)** built from alternating
`G_0 = |D| tanh(h|D|)` and `∂_x` with powers of `η`. Four structural
properties are load-bearing in the paper:

1. **Self-adjointness is the efficiency argument** (§3.1). JCP09 uses the
   adjoint recursion *specifically* because `G` is self-adjoint, which lets
   `G_j` be stored/reused as vector ops on `ξ` instead of recomputed per
   order. Cost `O(M² N log N)`.
2. **Filtering is applied to the operator output every step** (§3.3, eq 28–29),
   not just the state. The paper uses both an ideal low-pass (`m=0.9`) and the
   smooth Hou–Li exponential filter `c_k = exp(-a (k/kmax)^{2m})` with
   `a=m=36`, applied to `η̂_k` and `ξ̂_k` at every time step.
3. **The filter is smooth (Hou–Li), not a hard cutoff.** JCP09 §4.1 documents
   that beyond an optimal truncation order `M` the expansion ill-conditioning
   amplifies high-`k` roundoff; the smooth filter is the stated remedy. A hard
   cutoff produces a Gibbs-like cliff right where `k^M` amplification is
   largest.
4. **Linear-exact time integration** (§3.2, eq 17–21). JCP09 splits
   `∂_t u = L(u) + N(u)` with `L = A u` the closed-form `G_0` block, solves
   the linear part *exactly* via the integrating-factor change of variables
   `û_k = Φ(t) v̂_k` (eq 18–19), and time-steps only the nonlinear residual
   `(G − G_0)ξ`. This is what makes the high-`k` linear dispersion band
   non-accumulating in time.

The paper also reports (§4.1) exponential convergence in `M` with an
*optimal* `M` beyond which roundoff takes over — `M=11` at `a=0.2`, smaller
`M` for finer resolution — and that de-aliasing pushes the optimal `M`
higher. This is the same ill-conditioning that drives the audit's high-`k`
cascade.

---

## 2. Where CS-DNO currently leaves this physics on the table

Cross-referencing `models/dno-net/dno_net_v2.py`,
`solver/evals/model_rollout.py`, and the parked-variants section
(`notes/cs_dno_architecture.tex` §"Parked architectural variants"):

| JCP09 property | CS-DNO status | Evidence |
|---|---|---|
| Self-adjoint `G` | **Not enforced; flag off, never ablated** | `tie_xi_out_mult=False` default; ablation study only tested `n_blocks` and feature set |
| Filter the operator output every step | **Not wired into rollout** | `model_rollout.py:211-213,267-270` filters `eta`/`xi` only; `gxi` output is unfiltered. `filter_gxi` flag exists in eval plumbing but is not applied to the model prediction |
| Smooth (Hou–Li) filter | **Hard `k_cut=128`** | `g1_k_cut=128`, `filter_fraction=0.25` ideal low-pass |
| Linear-exact time stepping | **No** | Rollout feeds `G_0 ξ + residual` (the sum) into a generic explicit integrator; high-`k` linear-dispersion error accumulates in time even though `G_0` is known analytically |
| Analyticity at `η=0` (series convergence) | **Only at init** | `phi_bias_free=False`; after training `Φ(0)≠0` so `G_θ(0;h)≠G_0(h)` exactly. Flag exists, parked, never ablated |
| Nested `G_0(η)` in the recursion | **Not exposed to the trunk** | η-feature stack is `[η, η², η³, ∂_x η, ∂_x² η, |D|^{1/2}η, ℋη]` — never `G_0(η)` or `∂_x G_0(η)`, so the model must learn what `G_0` acting on `η` looks like through the multiplier MLPs |
| Odd/imaginary multipliers (`D_x = −i∂_x`) | **Not representable** | Multipliers are real (zero-phase). Architecture doc §"Expressivity notes" flags that the `η_x ∂_x` part of `G_1` is only approximable |

### What the audit says is actually failing

From `notes/v7_tanaka_failure_audit.md`: the v5 failure mode on the three
deep Tanaka cases (cid5/cid11/cid1000006, `h≈0.23–0.29`, `a/h≈0.27–0.33`)
is **not** a truth-validity issue (truth energy drift `<5e-7`) and **not** a
physics-stability issue (well inside JCP09's stable `a/h≤0.6` envelope). It
is:

- An **error cascade in the high-`k` band of the model's `gxi` prediction**,
  seeded by f32 precision in the first integrator step.
- Amplified by (i) the order-`M` CS series (`G_m ~ k^m η^{m-1}`, `k^6 ~ 7e12`
  at `k=128`) and (ii) the orbital displacement per outer `dt` at deep `h`.
- Once `|hat_err|(k>30)` exceeds `|hat_truth|(k>30)` (by `t=2.4` in v5), the
  cascade is irreversible because truth has no carrier in that band.

The audit's #1 recommended mitigation is **`filter_gxi=true`** at eval and
training time — which is exactly JCP09 property #2 above and is currently
not wired in.

The fourth failing case (cid1000011, `h=0.011`, `a/h=0.194`) is a separate
f32 dynamic-range floor issue: truth `|eta|_max=0.0022`, model reproduces
the right *shape* but wrong *amplitude scaling* (`alpha_final = −0.996`,
anti-correlated). This is a data-coverage / precision issue, not the
high-`k` cascade.

---

## 3. Prioritized improvements

Ranked by (expected value × grounding in JCP09 × implementation cost).

### Tier 1 — directly from JCP09, cheap, unimplemented

**A. Wire `filter_gxi` into `solver/evals/model_rollout.py` and train with it on.**
- Labels are already lowpassed to `k≤128`; filtering the model's `gxi`
  prediction the same way means it never has to learn `k>128` content and
  the cascade cannot seed.
- Audit's #1 recommendation. One-line eval change plus a training flag.
- Files: `solver/evals/model_rollout.py` (apply `apply_lowpass` to `gxi`
  after the model call, before the integrator step);
  `train-jax-10m/1d_dno_fno_jax.py` (filter predicted `gxi` inside the loss).

**B. Replace the hard `k_cut=128` with the Hou–Li exponential filter (JCP09 eq 29).**
- `c_k = exp(-a (k/kmax)^{2m})`, `a=m=36`. Smooth roll-off kills the `k^6`
  amplification at the boundary without a Gibbs cliff.
- Apply to state *and* `gxi`.
- Files: `solver/evals/model_rollout.py` (new `apply_houli_filter` next to
  `apply_lowpass`); `train-jax-10m/1d_dno_fno_jax.py`.

**C. Ablate `tie_xi_out_mult=True`.**
- Enforces the self-adjoint symmetry JCP09 exploits; halves the multiplier
  parameter count (the dominant 47.6% of params per the architecture
  reference's parameter table); strong regularizer matching the true operator.
- Never tested — the ablation study only swept `n_blocks` and the feature set.
- File: `models/dno-net/dno_net_v2.py` (already implemented, just flip the
  default / pass through training); `train-jax-10m/1d_dno_fno_jax.py`.

### Tier 2 — architecture changes that respect the recursion (12)/(13)

**D. Add `G_0(η)` and `∂_x G_0(η)` to the η-feature stack.**
- The true `G_2, G_3` involve `G_0` *acting on η* (nested multipliers).
  Currently the trunk has pointwise powers + `∂_x η, ∂_x² η, |D|^{1/2}η, ℋη`
  but never `G_0(η)`, so the model must learn what `G_0(η)` looks like
  through the multiplier MLPs.
- Exposing it analytically is the same economy as the `G_0` baseline,
  applied to the η-side.
- File: `models/dno-net/dno_net_v2.py` `_eta_spatial_features` — add two
  channels using the existing `_g0_apply` helper with `η` as input.

**E. `phi_bias_free=True`.**
- Removes all biases from the η trunk and Φ projections so every learned
  term vanishes identically at `η=0`: `G_θ(0;h) = G_0(h)` becomes exact for
  *all* trained weights, not just at init.
- Matches the analyticity-at-`η=0` property JCP09 relies on for series
  convergence.
- Parked, never ablated. File: same as above.

**F. A few odd/imaginary multiplier branches.**
- The architecture doc notes the `η_x ∂_x` part of `G_1` is unreachable with
  zero-phase real multipliers (`D_x = −i∂_x` is imaginary).
- One or two complex branches close the gap cheaply. Doubles the parameter
  count of those branches only.
- File: `models/dno-net/dno_net_v2.py` `DepthAwareMultiplier` (complex output
  dtype) and `CraigSulemBlock` (complex pointwise product for those branches).

### Tier 3 — training & data

**G. `pushforward_steps > 0` combined with `filter_gxi` on.**
- Pushforward is implemented (`--pushforward_steps`) but 0 in baseline; an
  earlier "failure" was in a different failure mode.
- JCP09's point that errors accumulate through the nonlinearity in time is
  exactly what pushforward trains against.
- Re-try with the `filter_gxi`-on combination since the original failure-mode
  diagnosis was different.
- File: `train-jax-10m/1d_dno_fno_jax.py` (already plumbed).

**H. Linear-exact integrating-factor split in the rollout** (JCP09 eq 17–21).
- Time-step only `(G − G_0)ξ`, keep `G_0` exact in time via the fundamental
  matrix `Φ(t)` (eq 19).
- Decouples the high-`k` linear dispersion band (which the model already
  gets exactly) from time-accumulated error.
- Eval-side change, no retrain. Largest single change to the rollout loop.
- Files: `solver/evals/model_rollout.py` and the time-integrator module it
  calls.

**I. Data-side: add JCP09-canonical Tanaka at `h=1, a/h∈[0.3,0.6]` on a
larger domain; upweight the deep+steep tail.**
- The dataset samples `h∈[0.01,0.30]` — it *never* sees the paper's
  canonical `h=1` setting.
- The audit shows the failing regime (`h≥0.23 AND a/h≥0.27`) has **1%
  coverage** in training; the generator's own safety comment flags
  `depth_max=0.08` as the safe choice but the dataset uses `depth_max=0.30`.
- Either lower `depth_max` to 0.20, regenerate with 5-copy periodic
  extension instead of 3-copy, or add the canonical `h=1` family.
- Files: `solver/gen_data/generate_tanaka_dataset_v2.py` (per the audit);
  dataset rebuild into `data/combined_dataset_v8.npz`.

---

## 4. Recommended first experiment

**A + B + C together** (output filtering + smooth Hou–Li filter +
self-adjoint multipliers), then a retrain and the standard eval suite.

Rationale:
- Mutually reinforcing: filtering kills the cascade at source, the smooth
  filter removes the cliff that feeds `k^6` amplification, self-adjointness
  halves the dominant parameter block and matches the true operator.
- All three are JCP09-grounded and cheap (one eval-side wiring, one filter
  swap, one flag flip + retrain).
- Directly targets the audit's identified failure mode (high-`k` cascade
  seeded by f32 in step 1, amplified by `k^6` at the cutoff).

Then **D** (`G_0(η)` features) as the first architecture change, since it
closes the recursion-structure gap that the ablation study identified as the
load-bearing component of CS-DNO (the spatial η-feature stack).

**Diagnostics to watch** (from the audit, in priority order):
1. `err_high(t=2.4) / truth_high(t=2.4)` on cid5, cid11, cid1000006. v5
   ratios: 1.09–1.29×. Pass criterion: `< 0.1×`.
2. `alpha_final` on cid1000011. v5: `−0.996`. Pass criterion: `> 0.5`.
3. NaN rate on Tanaka.

If (1) is unchanged after A+B+C, the model-output high-`k` floor is not the
binding constraint and the next step is **H** (linear-exact rollout) rather
than further architecture changes.

---

## 5. Key file references

- `models/dno-net/dno_net_v2.py` — `CraigSulemDNO`, `CraigSulemBlock`,
  `DepthAwareMultiplier`, `_eta_spatial_features`, `_g0_apply`,
  `_g1_baseline`. Parked flags: `use_g1_baseline`, `tie_xi_out_mult`,
  `phi_bias_free`, `g1_k_cut`.
- `solver/evals/model_rollout.py` — rollout loop; filtering at lines
  211–213, 267–270 (state only, not `gxi`); model reconstruction at lines
  76–90.
- `train-jax-10m/1d_dno_fno_jax.py` — training entrypoint with all CS-DNO
  flags plumbed (lines 100–201) and pushforward args (lines 181–201).
- `notes/cs_dno_architecture.tex` — architecture reference and parked
  variants (§"Parked architectural variants", lines 363–386).
- `notes/cs_dno_ablation_study.md` — `n_blocks=1` Pareto win, feature stack
  is load-bearing.
- `notes/v7_tanaka_failure_audit.md` — failure mode analysis and recommended
  diagnostics.
- `JCP09.pdf` — §2.2 (DNO recursion eq 12–13), §3.1 (self-adjointness),
  §3.2 (linear-exact integrating factor eq 17–21), §3.3 (de-aliasing +
  Hou–Li filter eq 28–29), §4.1 (truncation convergence + optimal `M`).
