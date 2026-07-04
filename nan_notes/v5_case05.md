# v5 · j=5 · case_id=5 · h=0.276 (a/h ≈ 0.28)

Trajs: `outputs/cs_dno_w512b8_l256_v5_2gpu_20260615_141717/eval_suite_f64h/tanaka_g0_trajs.npz`.
Rollout: `dt=0.8, tmax=200, n_t=251, nx=1024, x=linspace(0,2π,1024,endpoint=False)`.
Truth is finite for all 251 frames (max|η|_truth = 0.0915, max|gξ|_truth = 0.0666, stationary Stokes-like envelope). All blow-up is model-side.

## 1. First-blowup frame

- **First non-finite frame = 52, t = 41.60.** All three pred fields (`pred_eta`, `pred_xi`, `pred_gxi`) turn NaN in the same frame — consistent with a single surrogate substep whose output overflows fp32.
- The step before, frame 51 (t = 40.80), is still finite but already catastrophically out of physics: `max|η|_pred = 6.79` vs truth 0.091 (74×), `max|ξ|_pred = 23.6` vs 0.061 (390×), and **`max|gξ|_pred = 3.83e+07`** vs 0.067. So frame 51 is really the blowup; frame 52 is fp32 overflow of the resulting gξ.
- Which |k| band diverges first: **band 3 (32 ≤ |k| < 96) leads.** By frame 30 (t=24) band 3 pred = 2.9e-10 vs truth 1.3e-14 (already ~4 orders above truth's numerical floor); the tail is present from the very first frames. Band 2 (10 ≤ |k| < 32) only crosses 10× truth at frame 48 (t=38.4). By blow-up (frame 51), the energy hierarchy is inverted: band 4 (|k| ≥ 96) holds 1.55e-1, band 3 = 5.6e-2, band 2 = 7.4e-3, band 0 (|k|<10) = 1.5e-3.

## 2. Growth trajectory (frames 15…51)

Coherent low-k envelope is essentially unchanged (max|η|_pred stays 0.091 → 0.092 through frame 50, ‖η‖₂ stays 0.032, `energy ≈ 0.0013` vs truth 0.0014). The instability is entirely a small high-k tail growing under the surrogate.

Selected rows (see raw diagnostics in cell log; full 15..51 table generated in-run):

| frame | t | max\|η\|_p | max\|ξ\|_p | max\|gξ\|_p | ‖η\|₂_p | E_p | band3_p (32..96) | band4_p (\|k\|≥96) |
|---|---|---|---|---|---|---|---|---|
| 15 | 12.00 | 0.091 | 0.061 | 0.067 | 0.032 | 0.0014 | 4.31e-12 | 5.10e-13 |
| 30 | 24.00 | 0.091 | 0.061 | 0.067 | 0.032 | 0.0014 | 2.94e-10 | 1.33e-11 |
| 40 | 32.00 | 0.090 | 0.061 | 0.069 | 0.032 | 0.0013 | 1.06e-07 | 5.68e-08 |
| 45 | 36.00 | 0.090 | 0.061 | 0.067 | 0.032 | 0.0013 | 6.73e-07 | 3.83e-07 |
| 48 | 38.40 | 0.091 | 0.061 | 0.089 | 0.032 | 0.0013 | 8.80e-07 | 1.06e-06 |
| 49 | 39.20 | 0.092 | 0.061 | 0.094 | 0.032 | 0.0013 | 1.13e-06 | 2.50e-06 |
| 50 | 40.00 | 0.092 | 0.061 | **0.163** | 0.032 | 0.0013 | 1.71e-06 | **1.38e-05** |
| 51 | 40.80 | **6.79** | **23.65** | **3.83e+07** | **0.474** | **1.92** | **5.63e-02** | **1.55e-01** |

Doubling times (least-squares log-fit):

- max\|η\|_pred [15..51]: slope 0.0226/t → t_double ≈ 30.7 (slow uniform drift over the whole window).
- max\|η\|_pred [45..51]: slope 0.583/t → t_double ≈ 1.19 (final acceleration; one rollout dt = 0.8).
- max\|gξ\|_pred [15..51]: slope 0.117/t → t_double ≈ 5.94.
- max\|gξ\|_pred [45..51]: slope 2.78/t → t_double ≈ 0.25 (**~3 doublings per outer step by the end**).

The high-band energies grow ~4 orders across frames 30→45 (band3: 3e-10 → 7e-7), then explode ~5 more orders in the final 5 frames — a smooth exponential ramp that finally kicks into a runaway once the gξ operator (`|D|^{1/2}` applied to η) starts feeding on the |k|≈100 tail (which multiplies η_k by ~10).

## 3. Spectral cascade signature

Peak-k of |η_k| restricted to |k| ≥ 10 (i.e. where the new tail lives):

| frame | t | pred peak-k | truth peak-k |
|---|---|---|---|
| 30 | 24.00 | 10 | 10 |
| 37 | 29.60 | 10 | 10 |
| 38 | 30.40 | **80** | 10 |
| 40 | 32.00 | 112 | 10 |
| 43 | 34.40 | 40 | 10 |
| 48 | 38.40 | 40 | 10 |
| 49 | 39.20 | 104 | 10 |
| 51 | 40.80 | 127 | 10 |

The tail sits at **|k| ≈ 60–130**, i.e. deep in the high-k band, well short of the Nyquist wavenumber (|k|_max = 512 with nx=1024). At blow-up, band4 (|k|≥96) has more energy than band3, and band [128,256) has 3.99e-3 (much less than the [64,128) band 1.55e-1), so this is **not a pile-up at Nyquist / aliasing signature** — the energy is peaked at mid-high |k| ≈ 100 and *decays* toward Nyquist. The bin (256,512) stays at 1e-16 the entire run — no Nyquist reflection.

The truth spectrum, meanwhile, is essentially unchanged from IC across all 251 frames — a clean Stokes envelope with support only in |k| ≤ 8.

## 4. Truth-vs-pred divergence

- Truth is completely well-behaved at t = 41.60: max|η|=0.091, max|gξ|=0.067, all bands identical to their frame-15 values. (Confirmed by the truth-validity gate — this is a model failure, not a truth blow-up.)
- Band where pred first *departs* from truth: **band 3 (32 ≤ |k| < 96) is > 10× truth from frame 15 onward** (pred 4.3e-12 vs truth 1.1e-14). This is the fp32 noise floor of the surrogate — it's below O(1) energy so no dynamical effect yet, but it seeds the cascade.
- Band 2 (10 ≤ |k| < 32) first exceeds 10× truth at frame 48 (t = 38.4). By then it's already too late; the tail below it is O(1e-6).
- **First frame `rel_l2_eta > 0.1`: frame 50 (t = 40.00), rel_l2 = 0.141.** One outer step later (frame 51, t = 40.80) rel_l2 = 14.6 — one step from "5% error" to blow-up. This matches the fitted 1.19-time-unit doubling of max|η| near the end.

Sequence: fp32-noise tail at |k|~60–130 grows smoothly for ~30 outer steps at doubling time 6 (in gξ), the |D|^{1/2} weighting boosts gξ relative to η, once max|gξ|_pred crosses ~0.1 (frame 50, 2.4× truth) the surrogate is out of its training distribution and one step later the whole state is O(10) with fp32 overflow of gξ following.

## 5. Verdict

**(B) mid-k Lyapunov mode.** Growth is a smooth exponential in a band peaked at |k| ≈ 60–130 (well below Nyquist |k|=512), doubling time in gξ is ~6 t-units over the long ramp and ~0.25 at the end, energy stays away from |k| ≥ 256, and the low-k coherent envelope is unaffected until the very last step. This is the standard v5 `f32` mid-high-k cascade — the fp32 gξ noise floor sits above truth's high-k floor from t=0, and the surrogate has an unstable direction in that band that the |D|^{1/2} in the gξ operator amplifies over ~40 outer steps. The event at frame 52 is fp32 overflow of the already-blown-up state at frame 51, not a discontinuous jump.

Report ends.
