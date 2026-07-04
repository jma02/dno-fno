# v5 · case 11 (h ≈ 0.234) — NaN forensic report

Model: `cs_dno w512/b8/l256`, fp32, v5 dataset
Trajs: `outputs/cs_dno_w512b8_l256_v5_2gpu_20260615_141717/eval_suite_f64h/tanaka_g0_trajs.npz`
Rollout: `dt = 0.8`, `tmax = 200`, `n_t = 251`, `nx = 1024`, `L = 2π`, `substeps = 80`.

## 1. First-blowup frame

- **First NaN frame = 114, t = 91.20.**
- All three predicted fields (`pred_eta`, `pred_xi`, `pred_gxi`) NaN on the **same** frame (114). No single-field precursor NaN.
- Driver is unambiguously `pred_gxi`: at the last finite step (frame 113, t = 90.40)
  - `max|pred_eta|` = 6.463e-02 (truth 6.401e-02, +1%)
  - `max|pred_xi|`  = 4.341e-02 (truth 4.472e-02, −3%)
  - `max|pred_gxi|` = **1.128e+00** (truth 4.604e-02, **+24×**)
- Dominant unstable wavenumber at frame 113: **|k| = 126** in `pred_gxi` (Nyquist = 128), power 0.696. So the blow-up is **top-band, near-Nyquist**.

## 2. Growth trajectory (frames 80..113)

Truth is essentially stationary in every diagnostic across the whole window — `max|η|_truth ≈ 0.0640`, `max|ξ|_truth ≈ 0.0447`, `max|gξ|_truth ≈ 0.0460`, `‖η‖₂ ≈ 0.698`, `E ≈ 2.851`. The instability is entirely on the prediction side.

Prediction diagnostics, selected frames:

| f | t | max\|η\|_p | max\|gξ\|_p | \|η\|₂_p | E_p | b3 (32–96) η_p | b4 (\|k\|≥96) η_p |
|---|---|---|---|---|---|---|---|
| 80 | 64.00 | 6.40e-02 | 4.60e-02 | 0.6977 | 2.848 | 1.84e-07 | 2.08e-10 |
| 95 | 76.00 | 6.39e-02 | 4.61e-02 | 0.6975 | 2.848 | 2.12e-06 | 9.86e-09 |
| 105 | 84.00 | 6.41e-02 | 4.95e-02 | 0.6976 | 2.848 | 1.34e-05 | 2.38e-07 |
| 108 | 86.40 | 6.51e-02 | 6.06e-02 | 0.6978 | 2.849 | 9.11e-05 | 2.03e-04 |
| 110 | 88.00 | 6.49e-02 | 1.24e-01 | 0.7008 | 2.865 | 1.05e-04 | 2.96e-03 |
| 112 | 89.60 | 6.39e-02 | 4.25e-01 | 0.6993 | 2.847 | 4.04e-03 | 2.11e-02 |
| 113 | 90.40 | 6.46e-02 | 1.13e+00 | 0.7728 | 3.351 | 1.18e-02 | 1.02e-01 |
| 114 | 91.20 | NaN | NaN | NaN | NaN | — | — |

- **Exponential fit on `max|gξ|_pred` frames 80..113:** slope = 5.19e-02 / time-unit, doubling ≈ **13.4**.
- **Tail-only fit frames 100..113:** slope = **0.245 / time-unit, doubling ≈ 2.83**. So the growth is not a single exponential — it accelerates dramatically in the final ~10 frames (~8 time units).
- `max|η|_pred` grows only ~1% over the same window (fit slope 4.9e-04, doubling ≈ 1400). This is a **gξ-driven blow-up**, not an η envelope amplification.
- Energy budget: E_p only wobbles by 0.5% until frame 113, where it jumps 3.351 vs truth 2.851 (+18%). Energy accounting stays clean right up to the last finite step.

## 3. Spectral cascade signature

`|η_k|²` band energy in `pred_eta`:

- **frame 95 (t = 76.00):** b3 = 2.1e-06, b4 = 9.9e-09. Dominant `|k| ≥ 5` mode: **|k| = 5** (physical). Pred noise floor in b3 is already 1.5e6× truth (truth b3 ≈ 1e-12), but it's still tiny in absolute terms.
- **frame 105 (t = 84.00):** b3 = 1.3e-05, b4 = 2.4e-07. Still low-k-dominated.
- **frame 112 (t = 89.60):** b3 = 4.0e-03, b4 = **2.1e-02**. **Dominant `|k| ≥ 5` mode has jumped to |k| = 121** (near Nyquist). Power at that single mode = 1.43e-03 in η, 1.09e-01 in gξ.
- **frame 113 (t = 90.40):** b3 = 1.2e-02, b4 = **1.02e-01** (≈15% of the |k|<10 mass). Dominant mode still **|k| = 126**, power 0.696 in gξ.

The energy is piling up at the **top of the spectrum (|k| ≥ 96, near Nyquist)**, not at mid-k. It is not the classic k ≈ 30–60 Lyapunov signature.

## 4. Truth-vs-pred divergence

- Truth is completely well-behaved at t = 91.20 and beyond — every truth field remains at its steady-state amplitude (all bands stable to 4+ sig figs).
- Pred–truth divergence signature (band ratio `b_pred / b_truth`):

  | frame | t | \|k\|<10 | 10–32 | 32–96 | ≥ 96 |
  |---|---|---|---|---|---|
  | 95 | 76.00 | 1.00 | 0.97 | **1.5e+06** | 2.1e+02 |
  | 105 | 84.00 | 1.00 | 0.94 | **1.5e+07** | 6.6e+03 |
  | 108 | 86.40 | 1.00 | 2.59 | 7.3e+07 | 4.5e+06 |
  | 112 | 89.60 | 0.95 | 1.5e+02 | 2.3e+09 | 5.2e+08 |
  | 113 | 90.40 | 0.99 | 2.3e+02 | 9.1e+09 | 2.3e+09 |

  The **32–96 band was already 1.5e6× above truth by frame 95** — 15 time units before blowup — but only at absolute power ~1e-6, undetectable in physical diagnostics. It's the **|k| ≥ 96 (near-Nyquist) band that lights up latest but grows fastest**, catching up from 2e2× at frame 95 to 2e9× at frame 113.
- **First frame with `rel_l2_eta > 0.1`: frame 111, t = 88.80** (rel = 0.154). Only 3 frames / 2.4 time units before NaN. The rel-error series is (100 → 113): 0.0051, 0.0054, 0.0057, 0.0059, 0.0068, 0.0074, 0.0119, 0.0177, 0.0283, 0.0360, 0.0827, **0.1542, 0.2540, 0.5673**. Doubling every 1–2 frames near the end.

## 5. Verdict

**(A) aliased-cascade at Nyquist.**

Justification: `pred_gξ` grows from truth-level (~4.6e-2) to O(1) in 10 frames, doubling every ~2.8 time units in the tail. The dominant unstable mode sits at **|k| = 121–126** (Nyquist = 128) — the top band `|k| ≥ 96` carries 15% of η's total mass by the last finite frame while truth has ~5e-11 there. Mid-k (10–32) only reacts *after* the top band explodes (b2 lags b4 by ~4 frames). This is the classic fp32 near-Nyquist aliasing cascade — the same mechanism seen elsewhere in the CS-DNO tanaka failures, cured by fp64 FFT casting.

### v5 vs v7/v8 timing comment

v5's first_nan_t = 91.2 on case 11 is **~70 time units earlier** than v7 (160.8) and v8-baseline (172.8) on the *same* case. But the mechanism looks **qualitatively identical** — near-Nyquist gξ blow-up on a stationary truth. What differs is the seed noise floor: at frame 95 (t = 76), v5's pred_eta already has b3 ≈ 2.1e-6 (1.5e6× truth). This gives the cascade a hotter starting point, so the exponential runaway crosses O(1) sooner. So v5 is not exhibiting a *different* weakness in this regime; it's the same aliased-cascade failure, just with a worse (noisier) initial high-k floor, plausibly because v5 was trained on a smaller/less filtered dataset than v7/v8 and its gξ predictions carry more baseline high-k garbage. The cures identified for v8 (fftfp64, jacreg) should apply identically here.

---

**Verdict: (A) aliased-cascade at Nyquist. `pred_gξ` blows up at |k| ≈ 121–126 (Nyquist = 128), doubling every ~2.8 time units, while truth stays stationary and η/ξ remain quiescent until the last frame — same mechanism as v7/v8 case-11, arriving ~70 time units earlier because v5's baseline high-k gξ noise floor is higher.**
