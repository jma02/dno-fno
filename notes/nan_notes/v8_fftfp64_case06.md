## v8 fftfp64 · case 6 (h = 0.133, mid-shallow — regression unique to fftfp64)

Traj file: `outputs/cs_dno_w512b8_l256_v8_fftfp64_20260625_012616/eval_suite_f64h/tanaka_g0_trajs.npz`
Batch index j = 6, case_id = 6, h ≈ 0.1331, tmax = 200, dt = 0.8, nx = 1024, L = 2π.

### 1. First-blowup frame
- All three pred fields go NaN together at **frame 197, t = 157.60** (frame 196 is the last finite frame; every one of 1024 points becomes NaN at once).
- `pred_gxi` is the runaway field: at frame 196 `max|gxi|_pred = 0.362` vs `max|gxi|_truth = 0.163`, `rel_l2_gxi = 5.9`. `pred_eta` and `pred_xi` are still nearly correct (rel_l2 ≈ 1.8) at that last finite frame.
- Blowup is concentrated at **|k| = 128** (see §3), well below Nyquist (k = 512) and well above the FNO learned-band edge (`modes = 64`).

### 2. Growth trajectory (leading-up window)
Frames 175 → 196 (physical-space maxima; spectral energy in bands over `|eta_k|_pred`):

| frame | t | max|η|ₚ | max|η|ₜ | max|ξ|ₚ | max|gξ|ₚ | ‖η‖₂ | E_pred | band <10 | 10–32 | 32–96 | ≥96 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 175 | 140.00 | 0.0390 | 0.0400 | 0.0245 | 0.0465 | 0.325 | 1.0e-4 | 1.0e-4 | 4.7e-7 | 2.8e-10 | **2.65e-6** |
| 180 | 144.00 | 0.0388 | 0.0400 | 0.0277 | 0.0797 | 0.339 | 1.4e-4 | 1.1e-4 | 3.1e-7 | 6.8e-10 | **4.81e-6** |
| 185 | 148.00 | 0.0405 | 0.0400 | 0.0327 | 0.1025 | 0.378 | 1.5e-4 | 1.3e-4 | 1.1e-6 | 2.4e-9 | **8.46e-6** |
| 190 | 152.00 | 0.0362 | 0.0400 | 0.0339 | 0.1333 | 0.422 | 1.9e-4 | 1.6e-4 | 7.2e-7 | 3.1e-7 | **1.44e-5** |
| 193 | 154.40 | 0.0469 | 0.0400 | 0.0322 | 0.3148 | 0.459 | 2.3e-4 | 1.8e-4 | 6.9e-7 | 1.7e-7 | **2.55e-5** |
| 195 | 156.00 | 0.0475 | 0.0400 | 0.0319 | 0.3169 | 0.446 | 2.3e-4 | 1.6e-4 | 9.9e-7 | 2.0e-6 | **2.89e-5** |
| 196 | 156.80 | 0.0555 | 0.0400 | 0.0312 | **0.3622** | 0.471 | 2.4e-4 | 1.8e-4 | 1.8e-6 | 3.6e-6 | **3.20e-5** |

Exponential fits, last 30 frames (166 → 196):
- `max|η|_pred` slope = **0.0065/t**, doubling ≈ 107 t.u. — essentially flat.
- `max|gξ|_pred` slope = **0.093/t**, doubling **≈ 7.5 t.u.** — the runaway.
- Isolated |k| = 128 mode: `|η_k|_pred` slope 0.101/t (doubling **6.86 t.u.**), `|gξ_k|_pred` slope 0.101/t (doubling **6.85 t.u.**). The two are locked, growing at the *same* exponential rate over the entire 200-t.u. horizon (gain 4.5×10⁷ for gξ mode, 3.7×10⁶ for η mode).

Rel-L2 threshold crossings (η channel): `rel_l2_eta > 0.01` at t = 105.6, `> 0.05` at t = 129.6, `> 0.10` at t = **133.6**. rel_l2_gxi hits 1.0 already at t ≈ 135 and reaches 5.9 by t = 156.8.

Early behavior (frames 0..30): utterly quiet. `max|η|_pred = 0.0400` matches truth to 4 decimals, and the |k|≥96 band creeps from ~7e-13 (t = 0) to ~1e-15 (t = 24) — this is the f32 alias floor rising as roundoff accumulates, but far below anything dynamical. No early anomaly.

### 3. Spectral cascade signature
The blowup is a **single-mode Lyapunov spike at |k| = 128**, not a broadband cascade and not a Nyquist alias. Per-band log-slopes of `max|gξ_k|_pred` over frames 175–196:

| |k| band | log-slope /t | value at f=196 |
|---|---|---|
| 0–4 | +0.024 | 4.4e-3 |
| 4–10 | +0.036 | 6.6e-3 |
| 10–20 | +0.031 | 2.3e-3 |
| 20–40 | +0.116 | 2.2e-3 |
| 40–80 | +0.246 | 1.8e-3 |
| **80–160** | +0.055 | **2.9e-2** (already saturated early) |
| 160–320 | +0.165 | 2.0e-3 |
| 320–513 | +0.158 | 2.1e-3 |

The 80-160 band's *value* dwarfs everything else; its slope looks smaller because it was already ramping earlier — the isolated |k|=128 mode is on a 0.10/t slope from t=0 (see §2). Zooming into individual wavenumbers at frame 196 shows a **sharp peak with a cliff at 128**:

```
 |k|=124: 1.38e-2    |k|=127: 1.81e-2
 |k|=126: 1.38e-2    |k|=128: 2.92e-2   ← peak
 |k|=125: 8.29e-3    |k|=129: 1.74e-3   ← cliff (17× drop)
                     |k|=130: 5.86e-4
                     |k|=140: 3.22e-4
```

Truth `|gξ_k=128|` at the same frames is ≈ **8e-10** — pred is ~4×10⁷× larger. There is no truth energy above |k|≈30 to speak of (`|gξ_k=40|_truth ≈ 6e-7`, `|gξ_k=80|_truth ≈ 6e-9`).

### 4. Truth-vs-pred divergence
Truth is completely well-behaved throughout: `max|η|_truth = 0.0400`, its spectrum is stationary to five decimals from t = 0 out to and past t = 157.6. It's a mid-h, small-amplitude linear-dispersing case with essentially no nonlinear content — the shallow control case that should have been trivial.

Fourier of `(pred − truth)` at frame 196:

| field | k<10 | 10–32 | 32–96 | ≥96 |
|---|---|---|---|---|
| η   | 9.7e-3 | 8.9e-4 | 2.6e-4 | 2.4e-3 |
| ξ   | 2.0e-2 | 2.7e-4 | 2.5e-5 | 2.0e-4 |
| gξ  | 5.7e-3 | 3.4e-3 | 1.8e-3 | **2.92e-2** |

Error onset (pred/truth ratio in the 80–160 band exceeding 10×) is at **frame 9, t ≈ 7.2** — a single mode is misestimated in the very first surrogate step, then grows steadily. rel_l2_eta crosses 10% at **t = 133.6**, so from mode-onset to macroscopic divergence takes ~120 t.u., consistent with the 0.10/t = 6.9 t.u. doubling time (roughly 17 doublings).

### 5. Verdict — **(B) mid-k Lyapunov mode**

Single-mode, single-|k| exponential growth at **|k| = 128** (nx/8, not Nyquist and not the 2/3 dealiasing edge). rate ≈ 0.10/t (doubling 6.9 t.u.), locked between `pred_η` and `pred_gξ`, present from step 1 (t = 7 is when pred exceeds truth by 10× in that band), no ramp anywhere else in physical-space observables — `max|η|_pred` is flat to 3 decimals for the first 175 of 197 frames.

**Contrast with case 5 (h = 0.276).** case 5 is nonlinear (a/h ≈ 0.28, in the a/h ≥ 0.28 danger zone that every model fails on) with truth energy in mid-k where the surrogate has to work hard. Case 6 here is the opposite: a shallow, linear-dispersing case where truth spectrum stops around |k| ≈ 30 and there is *nothing* physical for the model to be resolving at |k| = 128. This blowup does not look like case 5. It looks like an fftfp64-specific spectral resonance — the fp64 FFT chain removed the f32 noise floor that was previously damping this mode by roundoff mixing, letting an isolated learned amplification at |k|=128 grow unimpeded. The training-time config confirms the model has `cs_g1_k_cut = 128` (currently inert because `use_g1_baseline = False` in this run, but the cutoff constant sits right on top of the runaway mode, which is suggestive that whatever set that constant is what the block ansatz has "learned" around).

**One-line summary:** Isolated |k| = 128 Lyapunov mode in `pred_gxi` (doubling 6.9 t.u., locked to `pred_eta`, present from t = 7) — verdict **B**, and clearly a *different* failure mode than case 5's nonlinear-regime cascade.
