# CS-DNO v5 — learned operator visualization (experiment #7)

Date: 2026-06-19.
Source: `cs_dno_w512b8_l256_v5_2gpu_20260615_141717/best_val_ckpt`.
Script: `solver/evals/viz_cs_dno_v5_operators.py`.

## Findings

1. **All 8 blocks learned the same symbol shape.** Per-block branch-RMS M_xi(k, h) at h ≈ 0.3 has shape correlation **+0.95 to +1.00** with both `|k|` and the linear DNO symbol `G0(k, h) = |k| · tanh(h|k|)`, and ~0.00 with identity. M_out has the same shape, ~half amplitude. No block resembles ∂_x, identity, or a band-selector. The 8 blocks differ only in scalar magnitude (RMS at h≈0.3 ranges 0.11–0.26 for M_xi, 0.06–0.15 for M_out).

2. **`phi_proj` is feature-selective and block-identical.** Composed linear chain `W_phi @ W_mix @ W_proj` shows the dominant inputs are, consistently across blocks:
   - `η` (mean |w| = 1.8–4.3)
   - `|D|^{1/2} η` (1.6–3.5)
   - `η²` (0.8–2.0)
   - everything else (`η³`, `∂_x η`, `∂²_x η`, `ℋη`) ≲ 0.3, often ~1e-2.

3. **latent=256 is ~5× over-provisioned.** Block 0's composed 7×256 kernel sorted by column L2 shows only ~50 branches carry meaningful weight. The other ~200 columns are near-zero.

## Verdict

**Yes, the model is hand-codable.** Despite n_blocks=8 × latent=256 = 2048 branches, v5 condensed to a 3-term polynomial ansatz over `{η, |D|^{1/2} η, η²}` with G0-style Fourier multipliers per term. A compact closed-form expansion of the form

```
gxi ≈ a₀·G0(D,h)·η + a₁·G0(D,h)·(|D|^{1/2} η) + a₂·G0(D,h)·η² + …
```

with ~3–5 learned scalar amplitudes should reproduce most of v5's behavior.

## Caveats

- Shape correlation computed at h≈0.3 only; h-dependence (visible in plots) is consistent with `tanh(h|k|)` saturation but not strictly verified at all h.
- M_out and M_xi are *separately* G0-like; their product carries an extra factor that the analysis doesn't isolate.
- This is the v5 *checkpoint*. v8 has not been characterized yet.

## Recommended next moves

- **#4 (trunk_linear)** is now redundant — the trunk already collapsed to feature-selectivity on the 3 dominant inputs; replacing gelu with Dense should be a no-op test.
- Skip **#2 (polys_only)** — half-deriv `|D|^{1/2} η` IS one of the 3 load-bearing inputs.
- New move: **#11 (handcoded_3term)** — directly train a 3-term closed-form `gxi ≈ Σ aᵢ · G0(D,h) · fᵢ(η)` over `{η, |D|^{1/2} η, η²}` with learnable scalar amplitudes and (optionally) a single learned multiplier per term. If this matches v5 to within 30%, the architecture is solved.
- **#1 (feature LOO at b=1)** is now informed: only `|D|^{1/2} η` is worth ablating (the other 3 derivatives are essentially unused).
