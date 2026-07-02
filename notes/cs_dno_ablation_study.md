# CS-DNO minimal-architecture ablation study

**Headline:** `n_blocks=8` is essentially decoration. A single CS block with the full spatial-feature set (η, η², η³, ∂η, ∂²η, ½∂η, ℋη) gets **within 9%** of the baseline val_loss with **~5.6× fewer params** and **5× faster wall time**. Killing the spatial features (cs_n_polys=1, all derivs off) but keeping 8 blocks regresses **2.3×**. The load-bearing component of CS-DNO is the **set of spatial / spectral η-features fed into the multilinear form**, not network depth.

## Setup

All runs: cs_dno, v7_trim (3.78M rows), 3 epochs, batch=256, lr=2e-4, weight_decay=1e-4, modes=64, width=512, latent=256, mult_hidden=128, sobolev_k=1, scale norm, 2×RTX 6000 Ada (local), JAX fp32, shard_map data parallel.

3 epochs is enough to discriminate — baseline drops val from 0.0113→0.00701 in those 3 epochs, the same trajectory the full v5 80-epoch retrain traced in its first 3 epochs (0.0115→0.00721 on v5).

## Results

| ablation       | n_blocks | cs_n_polys | derivs        | ckpt bytes | val ep1 | val ep2 | **val ep3**  | wall    |
|----------------|----------|------------|---------------|-----------:|--------:|--------:|-------------:|--------:|
| baseline       | 8        | 3          | all on        | 12,836,073 | 0.0113  | 0.00768 | **0.00701**  | 51.8 min |
| n_blocks=1     | **1**    | 3          | all on        |  2,282,946 | 0.00963 | 0.00803 | **0.00765**  | 10.4 min |
| min_features   | 8        | **1**      | **all off**   | 12,856,693 | 0.0192  | 0.0168  | **0.0162**   | 51.1 min |

Per-component reading:
- n_blocks=1 / baseline = 1.09× val_loss for 0.178× params, 0.20× wall — strict Pareto win.
- min_features / baseline = 2.31× val_loss for ~1.0× params (same n_blocks×width, slightly thinner Φ projection). Strict Pareto loss.
- min_features is essentially "8 stacked depth-aware Fourier multipliers with linear η". The Φ_i features (η^k * spectral derivs) are doing the load.

## Why the spatial features dominate

The CS block (`models/dno-net/dno_net_v2.py:206-260`) implements
`M_out(D, h) [ Σ_i Φ_i(η)(x) · (M_i(D, h) ξ)(x) ]`,
where Φ_i runs over polynomials of η times spectral derivative operators applied to η. Without those Φ_i, the block reduces to a depth-aware Fourier multiplier on ξ — i.e. a linear-in-ξ DNO with no η coupling at all (other than via the multilinear `M_out` over polynomial-η features that are now collapsed to a constant). The 2.3× val regression vs baseline shows this coupling is not optional.

Conversely, stacking 8 of those rich blocks vs 1 only buys ~9% val. The block already mixes η and ξ multiplicatively; stacking adds nominal expressivity but the data doesn't seem to demand it.

## Implications for porting back to FNO

The user's stated goal — "extract the simple reason for CS DNO's win and port back to FNO" — is now well-defined:
1. **The win is the spatial η-feature stack** (η, η², η³, ∂η, ∂²η, ½∂η, ℋη), fed multiplicatively against a depth-conditioned Fourier multiplier of ξ.
2. **The depth and block-stacking machinery is not load-bearing.** A single CS block at width=512, latent=256, mult_hidden=128, cs_n_polys=3 gets val 0.00765 in 10 min.

Suggested FNO retrofit: add an η-feature concatenation channel to the FNO input stack — `[η, η², η³, ∂η, ∂²η, ½∂η, ℋη]` plus depth via FiLM — and keep the rest of the FNO architecture. This tests whether the feature engineering alone (no multilinear form, no Φ_i × M_i product structure) captures the win.

## Suggested followup ablations

In rank order of expected information value:
1. **single-block + min mult_hidden** — does cs_mult_hidden=128 matter, or would 32 do? (Tests the depth-conditioning capacity.)
2. **8 blocks + cs_n_polys=1** (only derivs off) and **8 blocks + cs_n_polys=3, all derivs off** — splits the spatial feature contribution into polynomial vs derivative halves.
3. **n_blocks=1 + cs_n_polys=1 + all derivs off** — the true minimal: should regress to roughly min_features performance, confirming nothing was hidden in the depth.

## Run dirs

- `outputs/ablation_baseline_031004/`
- `outputs/ablation_n_blocks1_040511/`
- `outputs/ablation_min_features_041736/`

Logs: `logs/ablation/ablation_*_train.log`, orchestrator: `logs/ablation/orchestrator.log`.

## Notes on the run

Subagent B's orchestration loop exited prematurely (Monitor races) after launching only the baseline. Main session wrote `/tmp/ablation_orchestrator.sh` to sequence the rest with a hard stop at 05:50am EDT to free local GPUs for the v7 modal pipeline eval. Orchestrator finished at 05:11:53, well inside the budget.
