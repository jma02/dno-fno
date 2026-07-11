# v11: light DNO surrogate — design notes (2026-07-06)

Directives, verbatim from user:
- **"we want a neural network that is as light as possible."**
- **"we only want up to G_0, even G_1 is pushing it."**

Ground truth for what to build against:
- 3-4 tanaka_g0 samples NaN by t≈65-173 on v10 at f64_harness batched surrogate rollout.
- Failing ICs [5, 11, 22, 23] are moderate depth h ∈ [0.21, 0.28], NOT deep/shallow tails.
- C1 stage-tangent regularizer cures bf_g1 fully but only 1/4 tanaka NaNs — the tanaka
  cascade needs a **structural** fix, not a training signal.

## 1. What v10 (8-block × 256-branch × w512 CS-DNO, ~8M params) actually learned

From `notes/figures/cs_dno_v10_operator_viz/diagnostics.json` (n_ics=32 audit, 2026-07-06):

**Every block computes essentially the same k-symbol.**
- `xi_shape_corr_|k|` across the 8 blocks: {0.98, 0.96, 0.99, 0.99, 0.99, 0.99, 0.99, 0.93}.
- All 8 M_xi filters look like a scaled `G_0`. There is no depth-of-cascade being used;
  the 8 blocks are 8 near-copies of the same operator.

**M_out anti-correlated in the last block.**
- `out_shape_corr_|k|` per block: {0.96, 0.86, 0.99, 0.98, 0.80, 0.32, 0.12, **−0.88**}.
- Block 7's M_out symbol goes AGAINST |k| — i.e. adds gain at low-k and cuts high-k.
  Reading this as an amplitude-selective amplifier that fires on high-|ξ| samples.

**Cross-sample activation is nearly IC-independent.**
- Top-3 branches per block are the same set across every audited IC (healthy AND failing).
  Block 7 always fires [232, 120, 33/16]. Block 6 always [171, 192, 155/130/196].
- The model has collapsed to a fixed low-rank operator; the "n_branches=256" capacity is
  ~40 effective scalars (peak_gt_1e-1_of_max on M_out is 54/256 for block 7, 109/256 for 8).

**η-feature use is a 3-term ansatz.**
- Per-block L1 magnitudes are dominated by `{η, |D|^{1/2} η, η², η³}` (5-8, 2-5, 2-4, 2-4).
- `d_x η, d_xx η, H η` are all ≤ 1.5 in L1 — near noise. The model doesn't use them.

**Runaway signature (pre-NaN).**
- IC 22, t=166.4: block 7 `block_norm_over_xi_l2 = 2.84`, residual/G_0 = 1.15.
- IC 23, t=172.8: block 7 = 1.26, block 6 = 0.47, block 5 = 1.35.
- Block 7's contribution jumps 30-60× from t=0 to pre-NaN while G_0 grows only 4×.
  The correction magnitude has become *larger than the linear term*, and it's blowing up.

**Conclusion (revised 15:00 after block-7 ablation).** v10 has 8M params doing:
1. Applying `G_0(D, h) · ξ` (essentially the baseline).
2. Applying a coupled η-nonlinear correction from the sum of 8 blocks.

**Update 2026-07-06 15:00**: My initial reading — that block 7's anti-|k| M_out symbol
made it a "runaway amplifier" — was FALSIFIED by ablation. Zeroing cs_block_7/phi_proj
takes tanaka_g0 NaN from 4/32 to 28/32 and median error from 0.012 to 0.76. Block 7 is
a **load-bearing compensator**, not a parasite. The anti-|k| symbol *cancels* overshoot
from blocks 0-6. Consequently the 8-block cascade is a coupled system — the blocks are
NOT redundant copies of the same operator — and **v11-C's single-block replacement
premise is incorrect**. Revised recommendation is below.

## 2. Design constraints (all non-negotiable)

Every one of these attacks a specific failure mode we've verified:

1. **Analytic G_0 baseline. 0 params.** Every existing model already learned it; no reason to
   spend params rediscovering it.
2. **No block cascade.** The 8-block sum is empirically wasted — every block is the same
   operator. Use **at most one** learned correction block.
3. **Hard k-cutoff on the correction.** The v10 cascade tolerates mid-k noise; the light
   arch must not. Mask the correction output to `k < k_cut ≈ 64` (dt = 0.8 / nx = 1024 lets
   us keep k∈[0,64] with margin).
4. **Dealiased product.** `phi * xi_branched` in `dno_net_v2.py:175` is NOT dealiased — this
   is the source of the mid-k noise floor. The v11 correction must dealias the product
   (pad_factor ≥ 2 in the product step).
5. **Self-adjoint correction.** `M_xi = M_out` (tie the two multipliers) — matches the
   symmetry of the true G(η) and halves the multiplier params.
6. **Feature basis {η, η², |D|^{1/2} η}.** Everything else in the audit was ≤ 20% signal.
   Optional: add `η³` and `G_0(h) η` as ~free extras.
7. **Bias-free correction stack.** With no biases, `η=0 ⇒ correction=0` exactly, so the
   linear-regime limit is a structural property, not a trained one.
8. **Zero-init output.** Model returns exact G_0 at step 0; training only adds signal
   where it improves the loss.

## 3. Three candidates (pick one; recommend v11-B as first target)

Notation: nx = 1024, K = 513, L = 2π. Depth h ∈ [0.005, 5.0].

### v11-A: pure `G_0` baseline. **0 params.**

```
gxi(η, ξ, h) = G_0(D, h) · ξ    where    G_0 = |k| tanh(h|k|).
```

Serves as a floor. If this is competitive on smooth regimes, we've proven the 8M-param
architecture was ~0-value-added. Almost certainly worse on tanaka_g0 nonlinear regime,
but it *cannot* NaN — the operator is bounded, linear, and h-monotone.

**Expected result:** median H^1 error 2-3× worse than v10 on smooth regimes; NaN=0/32
on tanaka_g0 (but rel_l2 tail probably 0.5-1.0).

### v11-B: `G_0` + closed-form `G_1` with k-cutoff. **~5-30 params.**

```
gxi = G_0(D, h) · ξ + α(h) · Π_{k<k_cut} G_1(η, ξ, h)
```
where
```
G_1(η, ξ, h) = -G_0(η · G_0 ξ) - ∂_x(η · ∂_x ξ)              [closed form, JCP09 §3]
α(h)         = tiny MLP:  4 features (h, log h, tanh h, h·tanh h) → 8 hidden → 1 scalar
Π_{k<k_cut}  = hard low-pass mask at k = 64
```

Already implemented in `dno_net_v2.py:278` (`_g1_baseline`, with `g1_k_cut=128` default).
For v11-B we lower the cutoff to 64 and gate G_1 by a learned depth-dependent scalar
α(h) ∈ [0, 1] so the model can shut off G_1 in the flat / very-shallow regimes where
higher-order effects vanish.

Param count:
- α(h) MLP: 4 → 8 → 1 = 32 + 8 (kernel + bias) + 8 + 1 = 49 params (or 32 bias-free).

Total: **~30 params**.

**Expected result:** essentially the same median as v10 on smooth regimes (both learn a
G_1-shape correction). Tanaka cascade should NOT happen — G_1 is exact to first order,
mode-limited, and multiplied by α(h) ≤ 1, so residual/G_0 is structurally bounded.

### v11-C: `G_0` + `α(h) · G_1` + tiny learned 4-branch residual. **~500 params.**

```
gxi = G_0 · ξ + α_1(h) · Π_{k<64} G_1(η, ξ, h) + N(η, ξ, h)
```
where `N` is a **single** dealiased self-adjoint CS block:
- Features: `{η, η², |D|^{1/2} η}` (3 features, no biases).
- Feature projection: `Dense(3 → 4, kernel_init=zeros, use_bias=False)` = 12 params.
- Multiplier: shared M(k, h) via a `DepthAwareMultiplier(hidden=8, out=4, no bias)` =
  4·8 + 8·4 = 64 params.
- Dealiasing: pad_factor=2 on `phi · xi_branched` step (0 params).
- Hard mask at k<64 on the mult output.
- α_1(h) MLP: ~30 params.

Total: **~500 params** — 16,000× smaller than v10.

Structurally, `N` cannot be a runaway amplifier because:
- Zero-init output → training only turns it on where it helps.
- Bias-free → identically zero at η=0.
- Hard k<64 mask → cannot generate mid-k noise (kills the v10 cascade at source).
- Shared M_xi = M_out with only 4 branches × 8-hidden multiplier → capacity is small
  enough that runaway would require both features and mult to blow up jointly, which the
  loss doesn't reward at training time.

**Expected result:** matches or beats v10 medians on smooth regimes AND cures tanaka_g0
cascade. This is the target design.

## 4. Comparison table

| arm    | params  | linear-limit  | G_1-limit          | k-cutoff | dealiased | self-adjoint | runaway-safe |
|--------|---------|---------------|--------------------|----------|-----------|--------------|--------------|
| v10    | 8M      | learned       | learned            | none     | NO        | if tie set   | NO           |
| v11-A  | 0       | exact         | —                  | k=K/2    | n/a       | yes          | yes          |
| v11-B  | ~30     | exact         | exact (k<64)       | k=64     | inherit   | yes          | yes          |
| v11-C  | ~500    | exact         | exact + tiny res   | k=64     | YES       | yes          | yes          |

## 5. Implementation surface

Everything below is **already present** in `models/dno-net/dno_net_v2.py`. We do NOT need
a new module file. The design maps onto existing hyperparams:

```
--model cs_dno --width 32 --n_blocks 1 --latent 4 --modes 64
--cs_n_polys 2  (η, η²; add 3 if η³ helps in a smoke)
--use_first_deriv=False --use_second_deriv=False --use_hilbert=False
--use_half_deriv=True
--use_g0_eta=False --use_g0_eta_dx=False
--tie_xi_out_mult=True
--phi_bias_free=True
--use_g1_baseline=True
--g1_k_cut=64                    (lower from default 128)
--mult_hidden=8                  (down from default 32)
```

Plus a tiny CLI addition: an `α_1(h) ∈ [0, 1]` gate around the G_1 baseline. Cheap:
new field on `CraigSulemDNO`, `use_g1_scalar_gate: bool = True`. Adds ~30 params.

**Cost of adding the α_1(h) gate**: 30-line change in `dno_net_v2.py` — one `nn.Dense`
in `__call__`, applied as a multiplier on the `_g1_baseline` return. No dataset or
trainer changes needed.

**Cost of the dealiased product in the CS block**: 30-line change in `CraigSulemBlock`
— pad the physical product to 2N, multiply, rfft back, truncate. Standard 3/2 rule.
No new hyperparam needed if we hardcode pad_factor=2 for the block.

Total code delta: ~60 lines in one file.

## 6. Training plan

1. **Smoke** (1 hr, 1 GPU): 100 batches on v9 dataset with v11-C spec, verify loss
   descends and no NaN.
2. **Full train** (v11-C): local 2 GPU, `combined_dataset_v9.npz`, 40 epochs.
   Because the model is 16,000× smaller, expect batch=1024 easily; ~2-5 min/epoch total.
3. **Eval**: standard f64_harness batched surrogate rollout on tanaka_g0 (n_ics=32),
   plus bf_g1 as the sanity check. Include v11-A + v11-B as reference arms.

## 7. Fallback if v11-C misses tanaka

If tanaka NaN survives at n_ics=32:
- Bump k_cut down to 48.
- Add a residual amplitude clip: `N ← N · min(1, β · ||G_0 ξ||_2 / ||N||_2)` with β=0.3.
  This is a **structural** cap on residual/G_0, matching the v10 pre-NaN diagnostic ratio.

If tanaka NaN survives even with the clip, the failure is inside the truth (not the
model) at that h/a regime — which we'd verify by re-running the truth cache with tighter
substeps.

## 8. Non-goals (deliberately excluded)

- No G_2+ terms. User directive.
- No 8-block cascade. It doesn't buy anything in v10 and it makes the arch fragile.
- No new dataset. v9 was our last dataset move; the fix is architectural now.
- No FT. This is a fresh train from scratch — the smallness makes it cheap.
- No Jacobian regularization (v9 tried; didn't help enough — see [[project-v10-result]]).

## 9. Open decisions to run past user

**Revised after block-7 ablation (2026-07-06 15:00).** v11-C's single-block premise was
falsified. Two viable paths:

1. **v11-B (recommended): pure analytic, no learned residual.** `G_0 + α(h)·G_1`. ~30
   params. Trades median accuracy on nonlinear regimes for **structural stability**.
   The user directive "even G_1 is pushing it" plus "as light as possible" both point
   here. If tanaka H^1 median regresses 3-5× vs v10 but tanaka NaN goes to 0/32, that's
   a win under the stated priorities.

2. **v11-C-revised: keep the 8-block cascade but shrink each block drastically.** Since
   the blocks are coupled, we can't drop them, but we can shrink `latent 256 → 8` per
   block and `mult_hidden 32 → 8`. Would yield ~5-20k params (still 500× smaller than
   v10) while preserving the compensating cascade structure. Add hard k<64 cutoff on
   block outputs to attack the mid-k noise floor.

Reject v11-A (pure G_0, no G_1) — too weak on nonlinear regimes per the audit's
`residual_l2 / g0_xi_l2 ≈ 0.2` at t=0.

Reject the original v11-C (single-block replacement) — ablation says the cascade is
load-bearing.

## 10. Chosen path & launched run (2026-07-06 14:59)

User picked **v11-C-revised**. Launched full 40-epoch training on GPU 0 (GPU 1 taken).

**Actual spec (as launched)**: n_blocks=8, latent=8, mult_hidden=8, width=32,
cs_n_polys=2 (η, η²), no d_x/d_xx/H (audit-informed drop), use_half_deriv=True,
phi_bias_free, tie_xi_out_mult, block_k_cut=64, use_g1_baseline+g1_k_cut=64,
sobolev_k=1, batch=128, lr=2e-4, wd=1e-4.

**Actual model size**: 2,224 params (48+256 in eta trunk, 240×8 = 1920 in blocks).
**3,600× smaller** than v10's 8M.

**Run dir**: `outputs/v11c_shrunk8blocks_20260706_145903/`

**Code delta shipped**:
- `models/dno-net/dno_net_v2.py`: added `block_k_cut` field to `CraigSulemBlock` and
  `CraigSulemDNO`; applied hard low-pass mask to m_xi and out_hat inside the block.
- `train-jax-10m/1d_dno_fno_jax.py`: added `--cs_block_k_cut` CLI arg; persisted in
  config.json.
- `solver/evals/model_rollout.py`: `load_run` now reads `cs_block_k_cut` from config.
- All three edits total ~15 lines. All existing eval/train pipelines unchanged.

**Smoke result**: 1 epoch on 1% data (477 batches) descends loss to ~0.024 in 8s wall,
no NaN, no shape errors. Post-train auto-eval crashed with unrelated `input_proj`
kernel error (existing task #21 bug, not related to this arch change).

**Expected wall**: ~5-9h for 40 epochs on 1 GPU.

**Eval plan when training completes**: `scripts/eval_v11c_full.sh` — batched surrogate
f64_harness at n_ics=32 on tanaka_g0 + bf_g1. Success criterion: NaN ≤ 2/32 AND
median ≤ 3× v10 on smooth regimes.

---

Related: `notes/figures/cs_dno_v10_operator_viz/`, `dno_net_v2.py:278`,
`solver/solvers/dno_series_jax.py:105`, `project_cs_dno_load_bearing`.
