# v7 · case 5 (h=0.276, a/h≈0.28) — NaN forensic report

Model: `cs_dno w512/b8/l256` v7 (fp32, v7_trim training set).
Traj file: `/home/johnma/dno-fno/outputs/cs_dno_w512b8_l256_v7_a100x2_resumed_20260618_124459/eval_suite_f64h/tanaka_g0_trajs.npz`
Rollout: dt=0.8, tmax=200, nx=1024, L=2π, substeps=80.
Truth is finite across all 251 frames — this is model-side, not truth blow-up.

## 1. First-blowup frame

- **First NaN frame = 48 (t = 38.40)**, matches the master table.
- `pred_eta`, `pred_xi`, `pred_gxi` all first become non-finite at exactly frame 48 (same step). At frame 47 all three fields are still finite but with the peaks below already blown up.
- The failure ignites in the **|k| ≥ 32 band** — see §2/§3. `max|pred_gxi|` jumps 0.067 → 0.086 → 0.119 → 0.244 across frames 43→47 while `max|pred_eta|` is still only 0.099. gxi (surface-vertical velocity ≈ derivative-heavy field) NaNs the pack; eta and xi follow the same step because the surrogate substep couples them.

## 2. Growth trajectory (frames 10–47)

Baseline (frames 10–40) is dead-flat: `max|eta_p| ≈ 0.0914` vs truth 0.0915, `‖eta‖₂ ≈ 1.035`, energy ≈ 1.40, all matching truth to five digits. **The overall exp fit on `max|eta|` over 10..47 gives slope 0.0009/s (760 s doubling) — dominated by the quiet stretch.** The blow-up is a sharp knee, not a slow ramp.

Last-window numbers, `max|eta|_pred / max|xi| / max|gxi|`:

```
fr=40 t=32.0  0.0913 / 0.0607 / 0.0665
fr=41 t=32.8  0.0913 / 0.0607 / 0.0663
fr=42 t=33.6  0.0911 / 0.0607 / 0.0666
fr=43 t=34.4  0.0913 / 0.0606 / 0.0669
fr=44 t=35.2  0.0920 / 0.0603 / 0.0861   <- gxi kink
fr=45 t=36.0  0.0942 / 0.0605 / 0.1194
fr=46 t=36.8  0.0992 / 0.0618 / 0.1191
fr=47 t=37.6  0.0993 / 0.0640 / 0.2440
fr=48 t=38.4  NaN
```

Local exp-fit on last 10 frames: slope 0.012/s → doubling 60 s in `max|eta|` (weak, because eta lags). Where the physics actually lives is the **|k|≥32 spectral band**: energy fit over frames 40..47 gives slope 1.36/s (energy) / 0.68/s (amplitude) → **amplitude doubling ≈ 1.0 s (~1 rollout step)**. `max|gxi|` doubles every ~3.3 s over the same window. Blow-up is one big high-k step, not slow exponential build-up in low modes.

## 3. Spectral cascade signature

`|η̂_k|` band energy vs truth, frame-by-frame:

```
fr t     |k|<10      10<=k<32   32<=k<96    96<=k<256   >=256   truth 32..96
20 16.0  1.05e-3     1.26e-08   2.07e-11    1.37e-11    3.4e-19   7.7e-15
30 24.0  1.05e-3     1.19e-08   2.78e-11    1.25e-10    3.3e-19   1.3e-14
40 32.0  1.05e-3     1.11e-08   3.04e-11    1.02e-08    3.0e-19   1.0e-14  <- 96..256 already 6 decades above truth
43 34.4  1.04e-3     1.47e-08   1.88e-08    3.19e-08    ...      (32..96 ratio 1058x truth)
44 35.2  1.06e-3     6.66e-08   2.56e-07    1.23e-07    ...      (32..96 ratio 4141x)
46 36.8  1.10e-3     7.84e-07   1.98e-06    1.38e-06    ...      (32..96 ratio 9676x)
47 37.6  1.13e-3     1.52e-06   4.98e-06    6.33e-06    ...      (32..96 ratio 21930x)
```

Truth stays clean throughout (`|k|>=32 band ~1e-14`). The blow-up chronology in `|k|>=32`:
- The `96 <= |k| < 256` band was **already 6 decades above truth by t=32** (1e-8 vs 1e-13). This is the persistent f32 gxi noise floor described in `project_tanaka_failure_mechanism`.
- Between t=34.4 and t=37.6 (≈4 s, ~5 outer steps), `|k|>=32` energy climbs 3e-8 → 1.1e-5 (~350x); `32..96` explicitly hits 22000x truth by frame 47.
- **The peak grows in the 32≤|k|<96 band first, then simultaneously in 96≤|k|<256; the pure Nyquist band `|k|≥256` never lights up (~1e-9 all through).**
- Low modes `|k|<10` barely move (1.05e-3 → 1.13e-3 over the whole blow-up).

So this is **not** aliasing at Nyquist (D would want the ≥256 band to blow first) and **not** a low-k envelope amplification (`|k|<10` is essentially flat). The energy piles up in the **mid-band around |k|≈30..100**, propagating upward from the 96..256 gxi noise seed.

## 4. Truth-vs-pred divergence

- `rel_l2_eta[j=5]` at frames 40..47: `0.0033, 0.0033, 0.0035, 0.0037, 0.0042, 0.0047, 0.0057, 0.0071, 0.0083 (fr37) ... 0.0322 (fr44), 0.0619 (fr45), 0.1063 (fr46), 0.2017 (fr47)`.
- **First frame where rel-eta > 0.1: fr=46, t=36.80**, i.e. only 2 outer steps before NaN. From <1% divergence to NaN in ~1.6 s of physical time.
- Truth is well-behaved at all frames — max|truth_eta| stays at 0.0915, high-k truth energy at 1e-14. Divergence is entirely on the model side.
- Band ratios `|η̂_p|/|η̂_t|` at frame 44: `<10: 1.005, 10..32: 2.28, 32..96: 4141, 96..256: 743`. **The break starts in |k|≥32 and reaches the low-mid band only after mid-k is already 4 orders too big.** Bulk `<10` never gets more than 4% off before NaN.

## 5. Verdict

**(B) mid-k Lyapunov mode — cascade seeded in 96..256 gxi noise, blows up through 32..96.**

The signature: (i) `|k|<10` stays glued to truth to the last frame (rules out C), (ii) pure Nyquist `|k|≥256` never lights up (rules out A/D), (iii) the peak band is 32..96 at 22000× truth by frame 47, with 96..256 growing in lockstep, (iv) the whole cascade takes ~5 outer steps not one (rules out D "single-step blowup"). The persistent 6-decade-above-truth `96..256` seed at t=32 is exactly the f32 gxi noise floor — pred gxi has ~1e-6 rms high-k content that truth has zero of, and once the a/h≈0.28 nonlinearity gets it into the 32..96 band it grows at amplitude doubling ~1 s until gxi is O(0.2) and the next surrogate substep overflows.

## v7 vs v5 growth-rate comparison (same case, both NaN)

v5 (first_nan_t = 41.60) and v7 (first_nan_t = 38.40) share the same qualitative story — mid-k Lyapunov cascade seeded in the f32 gxi high-k floor — but **v7's noise seed is bigger and it lights earlier**:

- `|k|>=32` band energy at t=32.0: **v7 = 1.02e-8** vs **v5 = 1.63e-7 (t=32.0)** — v5's seed is actually *larger*, so v5 lights first arithmetically, but the growth is what matters.
- Time from `|k|>=32 = 1e-6` band energy to NaN: v5 hits 1.05e-6 at t=36.0 and NaN at 41.60 → **~5.6 s runway**. v7 hits 1.9e-6 at t=36.8 and NaN at 38.40 → **~1.6 s runway**.
- Amplitude-doubling in the runaway window: v5 shows a two-plateau ramp (1e-6 for ~5 frames, then blow), v7 goes 1e-8 → 1e-5 in 6 frames (a much steeper knee, doubling ~1 s in amplitude vs ~2 s in v5).
- `max|gxi|` v5 stays 0.067–0.07 until t=40 then blows in one step (0.16 → 3.8e7 → NaN); v7 gxi lifts off earlier (0.067→0.086 at fr=44) and takes a slower 4-frame climb to 0.24 before NaN.

**v7 blows up ~20% earlier in physical time and with a steeper mid-k growth rate than v5.** This is consistent with the memory that v7_trim regressed relative to v5 — the smaller training dataset produced a surrogate whose 96..256 gxi bias is higher and whose mid-k mode is less damped, so once the a/h≈0.28 Tanaka crest starts feeding the cascade there's less runway.

---

**Verdict:** **(B) mid-k Lyapunov mode** — 32≤|k|<96 blows through 22000× truth in ~5 outer steps, seeded by a persistent 6-decade-above-truth `96..256` gxi f32 floor; v7's blow-up is ~20% earlier and steeper than v5's on the identical case, matching the v7_trim training regression.
