# NaN culling investigation, 2026-07-08

## Bottom line

The remaining rollout NaNs are not primarily an integrator bug and not solved by more vanilla fine-tuning data. They are a model-side spectral envelope failure in predicted `gξ`: the model can emit excessive mid/high-band `gξ` on visually small states. GL2 then amplifies this into a non-contractive substep and reports the terminal NaN.

The simplest realistic operational fix is to enable the full-output learned-`gξ` high-band limiter in hard rollouts:

```text
k_cut = 32
r_max = 1e-2
abs_floor = 5
```

This is not a clean model cure, but it is the best current NaN-culling guard. The durable model-side fix should make the predicted full `gξ` satisfy a hard or very strong differentiable spectral envelope before entering the RHS.

Do not run another Tanaka-only / harvested-pre-cascade fine-tune as the next step. We have effectively done that axis already: C6 and C7 are targeted Tanaka pre-cascade fine-tunes, and both made rollout stability worse. Repackaging the same idea with a small filter would still be a Tanaka fine-tune unless the target states are regenerated from clean truth-manifold neighborhoods and the full-output envelope is structural.

## Evidence scanned

Rollout NPZs inspected:

- C2 final baseline Tanaka: `outputs/c2_stage_match_gain_from_v85b_20260707_053707/eval_final_2reg_f64h_batched_cached_20260707_133545/tanaka_g0/tanaka_g0_trajs.npz`
- C2 final baseline BF: `outputs/c2_stage_match_gain_from_v85b_20260707_053707/eval_final_2reg_f64h_batched_cached_20260707_133545/bf_g1/bf_g1_trajs.npz`
- C2 final + full learned-`gξ` high-band limiter, Tanaka: `outputs/c2_stage_match_gain_from_v85b_20260707_053707/eval_final_tanaka_gxihbl_k32_r1e-2_abs5_20260707_1505/tanaka_g0_trajs.npz`
- C2 final + full learned-`gξ` high-band limiter, BF: `outputs/c2_stage_match_gain_from_v85b_20260707_053707/eval_final_bf_gxihbl_f64h_batched_cached_20260708_155438/bf_g1_trajs.npz`
- C5 residual high-band cap sweep: `outputs/c2_stage_match_gain_from_v85b_20260707_053707/eval_final_c5_rescap_hilo_beta_sweep_tanaka_cached_20260708_144857/b010/tanaka_g0_trajs.npz`
- C6 targeted pre-cascade pack fine-tune: `outputs/c6_precascade_pack_from_c2_20260708_151349/eval_final_tanaka_f64h_batched_cached_20260708_151617/tanaka_g0_trajs.npz`
- C7 balanced pre-cascade fine-tune: `outputs/c7_balanced_precascade_100k_from_c2_20260708_153259/eval_final_tanaka_f64h_batched_cached_20260708_153835/tanaka_g0_trajs.npz`

Data NPZs inspected:

- Original v8 distribution sample from `data/combined_dataset_v8.npz`
- Harvested target pack `data/tanaka_precascade_c2_v1.npz`
- Balanced fine-tune pack `data/balanced_precascade_c2_v1_100k.npz`

## Frame-by-frame rollout signature

C2 final baseline Tanaka has model NaNs in cases `[5, 11, 22]` and one additional finite divergence in case `27`.

First bad saved frames:

| case | first bad t | last finite t | last finite rel-L2 eta |
| ---: | ---: | ---: | ---: |
| 5 | 34.4 | 33.6 | 0.780 |
| 22 | 46.4 | 45.6 | 0.365 |
| 11 | 190.4 | 189.6 | 1.709 |

At the last finite frames, the low bands are not the main pathology; `gξ` has orders-of-magnitude excess in `k≈16..128`.

Representative predicted/truth `gξ` band amplitude ratios:

| case, last finite t | k=0..8 | 8..16 | 16..32 | 32..64 | 64..128 | 128..256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5, t=33.6 | 2.5 | 22 | 1.2e3 | 1.6e6 | 3.2e4 | 1.9e4 |
| 22, t=45.6 | 1.3 | 14 | 8.2e3 | 5.2e4 | 2.0e4 | 3.4e3 |
| 11, t=189.6 | 3.7 | 21 | 1.8e3 | 2.7e6 | 1.5e5 | 3.6e5 |

Case `27` is different: it is finite but divergent, with low/mid ratios close to one and only moderate tail excess. This separates two failure modes:

1. NaN mode: extreme mid/high-band `gξ` cascade.
2. Bounded-bad mode: lower-mode/tail phase/amplitude drift.

The NaN mode is what we can realistically cull with a spectral envelope.

## What the limiter proves

C2 final with the full-output learned-`gξ` high-band limiter:

- Tanaka: NaNs `3/32 -> 2/32`, divergence `4/32 -> 3/32`.
- Cures case `11`.
- Delays case `5` from first bad `t=34.4` to `t=65.6`.
- Delays case `22` from first bad `t=46.4` to `t=71.2`.
- BF: NaNs/divergence stay `0/31`, median stays essentially unchanged (`0.00735 -> 0.00734`), but p95 worsens (`0.186 -> 0.417`).

The capped Tanaka last-finite predicted `gξ` high-band amplitudes hit exactly the intended envelope:

| case | high amp | low amp | high/low |
| ---: | ---: | ---: | ---: |
| 22 | 6.36 | 63.6 | 0.10 |
| 5 | 5.95 | 59.5 | 0.10 |
| 11 | 3.54 | 22.1 | 0.16 |

This says the control variable is the full predicted `gξ` entering the RHS, not only the learned residual. C5 residual-only caps reduced some NaNs but created more bounded-bad trajectories because the full `baseline + residual` balance was still wrong.

## Why targeted fine-tuning got worse

C6/C7 were meant to test whether the model simply lacked data near pre-cascade states. They failed:

- C6 pack-only fine-tune: Tanaka NaNs `9/32`, div `14/32`; BF NaNs `1/31`, div `1/31`.
- C7 balanced fine-tune: Tanaka NaNs `7/32`, div `11/32`; BF NaNs `3/31`, div `4/31`.

This is worse than corrected C2 final without a limiter:

- Tanaka NaNs `3/32`, div `4/32`.
- BF NaNs/div `0/31`.

The target pack explains why. Its labels include states already in the pathological envelope:

| base case | t | rows | `gξ_hi` q50 | `gξ_hi` q95 | `gξ_hi` max | hi/lo q95 | eta_hi q95 | eta_lo q95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 28.8 | 257 | 2.095 | 2.942 | 3.433 | 0.146 | 0.393 | 26.827 |
| 5 | 32.0 | 257 | 17.138 | 17.264 | 17.445 | 0.779 | 1.739 | 27.500 |
| 5 | 33.6 | 257 | 557.354 | 558.584 | 563.847 | 13.840 | 3.923 | 28.924 |
| 11 | 180.0 | 257 | 8.097 | 8.231 | 8.525 | 0.586 | 1.113 | 18.095 |
| 22 | 40.0 | 257 | 5.211 | 5.408 | 5.910 | 0.339 | 0.564 | 29.687 |

For comparison, the original v8 Tanaka source distribution has `gξ_hi` max around `1.21` in a 10k sample and no rows active under the rollout limiter envelope. The harvested target pack has `gξ_hi` q95 around `557` and limiter-active fraction around `0.267`.

So C6/C7 did not teach the model clean missing physics. They taught it to reproduce contaminated C2 pre-NaN high-band outputs. That can create new NaNs in cases that C2 handled.

## What the model is missing

The model is missing a contractive/spectral admissibility constraint on the predicted full `gξ` before RHS evaluation. In practical terms:

```text
sqrt(sum_{|k|>=32} |gξ_hat_k|^2)
    <= max(abs_floor, sqrt(r_max) * sqrt(sum_{|k|<32} |gξ_hat_k|^2))
```

The successful runtime setting is:

```text
abs_floor = 5
r_max = 1e-2
```

The model may also be missing clean coverage of truth-manifold neighborhoods near pre-cascade states, but the current harvested pack is not that. Clean coverage means states from truth rollouts or accepted/reversible solver neighborhoods, not C2-contaminated states immediately before NaN.

## What the model has too much of

The model has too much learned mid/high-band `gξ` gain, especially in `k=16..128` and `k>=32`, relative to the low-band amplitude. It can create high-band `gξ` while `eta`/`xi` still look small in physical space. That is exactly the dangerous combination because GL2 sees the derivative-amplified RHS, not the visual smoothness of the state.

C7 also shows this is not isolated to the original three C2 failures:

- C7 Tanaka new NaN cases include `[6, 12, 13, 23, 29]`.
- C7 BF introduces NaNs in `[1000003, 1000023, 1000024]`.
- These new failures again show huge `gξ` high-band excess before NaN.

## Realistic options, ranked

### 1. Immediate operational guard: full-output `gξ` high-band limiter

Use the limiter in eval/rollout for hard regimes:

```text
--gxi_highband_limiter
--gxi_highband_k_cut 32
--gxi_highband_r_max 1e-2
--gxi_highband_abs_floor 5
```

Pros:

- Already tested.
- Culls/delays the known Tanaka NaNs.
- BF-safe in NaN/div/median.
- Cheap compared with retraining.

Cons:

- Does not fully cure cases 5/22.
- Worsens BF tail p95.
- It is a runtime guard, not a model fix.

### 2. Training-side hard envelope on full predicted `gξ`

Train/evaluate the architecture so the predicted full `gξ` satisfies the same envelope before it enters any rollout RHS or tangent/pushforward path.

This should be done on full output, not residual-only, because residual-only C5 distorted the low/high balance and still missed the actual RHS variable.

### 3. Clean data, not contaminated pre-NaN data

If adding data, regenerate neighborhoods from truth rollouts or solver-accepted states and reject labels/states that violate the admissible `gξ` envelope. The current C2-harvested pack should be filtered before any reuse.

Minimum culling rule for targeted packs:

```text
reject if gxi_hi > max(5, 0.1 * gxi_lo)
```

This would remove the worst case-5 t=33.6 rows and other C2-contaminated late rows.

### 4. GL2 residual containment / halving

Keep containment as a safety detector, but do not expect it to cure this class once the state is contaminated. Past substep sweeps and residual traces show the NaN time is not meaningfully fixed by finer substeps; the integrator is amplifying model-side high-band RHS error.

## Recommended next move

For culling NaNs now:

1. Treat C2 final as the model checkpoint baseline.
2. Enable full-output learned-`gξ` high-band limiting in Tanaka-like hard rollouts.
3. Do not use C6/C7.
4. Do not run another vanilla targeted fine-tune from `data/tanaka_precascade_c2_v1.npz`.

For the next training-side model experiment:

Do not make it a Tanaka fine-tune. The next trainable model-side experiment should target the actual instability mechanism: excessive local gain from small state perturbations into mid/high-band `gξ`.

Recommended C8 direction:

```text
C8 = C2 final/base recipe
     + stronger high-band tangent-gain regularization on ordinary v8/base states
     + no harvested Tanaka target-pack supervision
     + no C6/C7-style pre-NaN data fitting
```

The regularizer should penalize local response of `gξ_hi` in `k≈16..128` to admissible perturbations of `(η, ξ)`, rather than asking the model to match contaminated high-band `gξ` labels. This is the training-side analogue of the runtime guard: reduce the gain that creates envelope violations, not fit the envelope-violating outputs.

Secondary option:

- Use a strong full-distribution high-band envelope hinge on predictions, with active fraction and BF tail monitored. C4 tried a weak version and did not move the diagnostic, so this only makes sense if the hinge is strong enough to visibly reduce active excess early.

Data option, only if needed:

The next trainable model-side experiment should be a structural/full-output bounded-`gξ` architecture or loss path that is evaluated broadly, with any Tanaka-derived data treated only as a filtered diagnostic slice. If target data is used at all:

1. Regenerate or filter it by the same high-band admissibility envelope.
2. Keep ordinary v8 rows dominant as an anchor.
3. Apply the full-output envelope inside training/eval, not just as an auxiliary loss.
4. Evaluate against C2 final no-limiter and C2 final + eval-only limiter. The new checkpoint must beat the limiter on NaN/div without a BF p95 regression to be accepted.
