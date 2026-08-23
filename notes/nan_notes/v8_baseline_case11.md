# v8 baseline — case 11 (j=11, h=0.234)

**Model:** cs_dno w512/b8/l256, fp32 baseline
**Trajs:** `outputs/cs_dno_w512b8_l256_v8_2gpu_20260619_024621/eval_suite_f64h/tanaka_g0_trajs.npz`
**Rollout:** `dt=0.8`, `nx=1024`, `L=2π`, `n_t=251`, `substeps=80`
**Truth:** finite for all fields, all frames.
**Note:** This case is a *cure candidate* — v8-fftfp64 and v9-jacreg both eliminate this NaN.

## 1. First-blowup frame

- **First NaN frame:** 216, `t = 172.80` (very late in the 200-s window).
- All three predicted fields (`pred_eta`, `pred_xi`, `pred_gxi`) go NaN in the *same* step (frame 216). We cannot separate them by field, so the blowup is a single-step joint saturation — consistent with fp32 arithmetic overflow inside the surrogate substep once mid-k energy has grown enough that the nonlinear terms exceed fp32 range.
- **Which |k| band ignites first:** the mid-band `32 ≤ |k| < 96` is where anomalous pred activity first departs the truth floor (ratio > 3000× as early as frame 195), followed by an aggressive downstream buildup at `96 ≤ |k| < 128` that becomes the largest absolute contributor by frame 210.

## 2. Growth trajectory (frames 185–215)

Two very different clocks are running.

**Physical envelope (low-k) is quiet through blowup:**

| frame | t     | max\|eta\|_p | max\|eta\|_t | max\|xi\|_p | max\|gxi\|_p | ‖eta‖₂ p/t | energy p/t |
|-------|-------|-------------|-------------|------------|-------------|-----------|-----------|
| 185   | 148.0 | 0.0641      | 0.0640      | 0.0448     | 0.0464      | 0.0547/0.0547 | 0.01750/0.01750 |
| 200   | 160.0 | 0.0642      | 0.0640      | 0.0448     | 0.0465      | 0.0547/0.0547 | 0.01750/0.01750 |
| 210   | 168.0 | 0.0639      | 0.0640      | 0.0458     | 0.0709      | 0.0551/0.0547 | 0.01770/0.01750 |
| 214   | 171.2 | 0.0620      | 0.0640      | 0.0493     | 0.1360      | 0.0559/0.0547 | 0.01820/0.01750 |
| 215   | 172.0 | 0.0620      | 0.0640      | 0.0498     | 0.1374      | 0.0567/0.0547 | 0.01870/0.01750 |
| 216   | 172.8 | NaN         | 0.0640      | NaN        | NaN         | NaN       | NaN       |

`max|eta|_pred` is essentially flat (0.062–0.065) all the way to the NaN. Exponential fit over frames 185..215 gives slope **−0.0007/s** — no envelope growth at all. Energy drift is under 7% at frame 215. This is *not* a low-k blowup.

**High-k band energies grow fast (fit over 200..215):**

| band            | slope (/s) | doubling |
|-----------------|-----------:|---------:|
| \|k\|<10        | +0.0022    | 314 s    |
| 10≤\|k\|<32     | +0.345     | **2.01 s** |
| 32≤\|k\|<96     | +0.430     | **1.61 s** |
| 96≤\|k\|<128    | +0.563     | **1.23 s** |
| \|k\|≥128       | +0.719     | **0.96 s** |

`max|gxi|_pred` grows at +0.089/s (doubling ≈ 7.8 s) — 3× its low-k physical value by frame 215 — consistent with a G-operator that has to differentiate the growing mid-k eta content.

## 3. Spectral cascade signature

Sum of `|eta_k|` over the truth-clean band (`|k| ≥ 32`) vs frame:

| frame | t     | 32≤\|k\|<64 pred/truth | 64≤\|k\|<96 pred/truth | 96≤\|k\|<128 pred/truth | \|k\|≥128 pred/truth |
|-------|-------|------------------------|------------------------|-------------------------|----------------------|
| 195   | 156.0 | 6.06e-2 / 2.0e-5 (×2968) | 1.34e-1 / 1.8e-4 (×748) | 6.82e-1 / 1.3e-3 (×521)  | 1.4e-2 / 4.7e-5      |
| 205   | 164.0 | 4.35e-1 / 2.3e-5 (×18823)| 1.93e+0 / 2.4e-4 (×8013)| 1.64e+0 / 1.4e-3 (×1200) | 1.8e-2 / 2.9e-5      |
| 210   | 168.0 | 1.75e+0 / 2.1e-5         | 1.24e+0 / 1.7e-4        | 8.59e+0 / 1.5e-3         | 1.2e-1 / 5.7e-5      |
| 214   | 171.2 | 2.75e+0 / 2.3e-5         | 2.64e+0 / 2.1e-4        | 2.01e+1 / 1.4e-3 (×14413)| 6.7e-1 / 9.3e-5      |
| 215   | 172.0 | 4.90e+0 / 2.2e-5         | 8.96e+0 / 1.9e-4        | **2.67e+1 / 1.1e-3 (×23544)** | 1.9e-1 / 1.7e-4  |

Top individual modes at frame 215 (excluding physical k=0..5):
- **k = ±122**, amp ≈ 1.01 (single largest non-physical mode)
- **k = ±116**, amp ≈ 0.68 by frame 214

Nyquist band is **empty**: `|k| ≥ 500` energy at frame 215 is 2.4e-18. `|k| ≥ 256` is 1.4e-16. **No aliasing to Nyquist.** Energy is not being folded onto ±k_max; it is being *deposited* at a specific mid-high-k window centered on |k| ≈ 96–128, with a distinct spectral bump at k ≈ ±120 that grows into a spatially localized bump-on-crest before the substep saturates.

This is a **mid/high-k Lyapunov mode**, not an aliased cascade.

## 4. Truth-vs-pred divergence

- Truth is finite everywhere and its spectrum stays clean at the numerical floor (`|k| ≥ 32` truth energy ≈ 1e-5 for all frames).
- Rel-L2 eta error trajectory: 0.001 at t=16, 0.005 at t=80, 0.010 at t=144, 0.015 at t=160 — a very slow linear-in-t coherent bias, matching the "rollout drift is linear-in-t" observation in the MEMORY note. Then it inflects sharply:
  - frame 209 (t=167.2): rel = 0.099
  - **frame 210 (t=168.0): rel = 0.114 — first exceedance of 0.10**
  - frame 212 (t=169.6): rel = 0.154
  - frame 216 (t=172.8): NaN
- The 10%-error threshold is crossed only **4.8 s before NaN** — the divergence is late and rapid. It appears in the mid-k band; low-k stays within 3% relative amplitude the whole time. The spatial residual at f=215 is 0.024 rms, well below the 0.062 crest amplitude, but concentrated near x ≈ 4.0 (not the crest at x ≈ 2.0) — the noise is riding on the trough side of the wave, not the crest.

## 5. Verdict — **(B) mid-k Lyapunov mode**

Growth is monotone and exponential in a narrow high-k window (peak at |k| ≈ 96–128, single-mode maxima at k ≈ ±116, ±122), with doubling time ≈ 1.2 s, while the physical envelope, wave energy and low-k spectrum stay within 3–7 % of truth right up to the failure step. Nyquist and near-Nyquist bands are empty (< 1e-16), ruling out aliasing (A). The blowup is preceded by ~30 s of coherent, well-behaved rollout, so it is not (D) sudden-single-step and not (C) low-k envelope amplification. The signature is a *narrow, additive, high-k Lyapunov instability* that the surrogate quietly amplifies until fp32 overflows on the substep of frame 216.

### Contrast with case 5 (early NaN in every model)

Case 5 (h=0.276, first NaN t=139.2 for this same v8-baseline; t=38–41 for v5/v7) NaNs orders of magnitude earlier and in every model. Case 11 by contrast is a *late-time* failure that gets cured by both fftfp64 and jacreg. That pattern is consistent with:

- **case 5** = a genuinely hard steep-wave regime that the model has not learned (early divergence, likely across all bands, not just mid-k),
- **case 11** = a *marginal* trajectory whose only failure mode is a mid-k Lyapunov mode slowly accumulating in fp32 arithmetic — precisely the mode that (i) fp64 FFT casting attenuates by reducing the fp32 noise floor at the FFT (aligns with the "mixed-precision FFT WINS" memory) and (ii) jacreg suppresses by penalizing directional derivative amplification of exactly those modes.

The cure signal is coherent: both fixes attack the *linear amplification of round-off/high-k noise*, which is the failure mechanism observed here but not (or not only) the failure mechanism on case 5.

## One-line summary
Late-time (t=172.8) NaN with flat physical envelope but a narrow mid/high-k (|k|≈96–128, peak modes ±116, ±122) Lyapunov instability doubling every ≈1.2 s — **verdict B**; consistent with the fftfp64/jacreg cure signature and distinct from the every-model early-blowup pattern on case 5.
