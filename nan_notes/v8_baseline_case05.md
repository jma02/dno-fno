# v8 baseline (fp32) — Tanaka-g0 j=5 case_id=5

- Model: `cs_dno w512/b8/l256`, fp32 throughout, trained on combined_dataset_v8
- Trajs: `/home/johnma/dno-fno/outputs/cs_dno_w512b8_l256_v8_2gpu_20260619_024621/eval_suite_f64h/tanaka_g0_trajs.npz`
- Case: j=5, case_id=5, h=0.2761 (a/h ≈ 0.28)
- Rollout: dt=0.8, tmax=200, n_t=251, nx=1024, x∈[0,2π), substeps=80 → inner dt≈0.01

## 1. First-blowup frame

- First NaN at **frame 174, t=139.20**.
- `pred_eta`, `pred_xi`, `pred_gxi` all become non-finite at exactly the same frame 174 (any-NaN and all-NaN coincide). Truth stays finite throughout.
- **`pred_gxi` is the field that departs first in magnitude**: at fr=172 max|gxi_pred|/max|gxi_truth| = 1.43, at fr=173 it is 2.54, while max|eta| and max|xi| remain within 1% of truth. The blowup is written into gxi one solver-step ahead of eta/xi (which then follow through the substep coupling in the next macro step, hence the simultaneous NaN at frame 174).
- The excited |k| band is **32 ≤ |k| < 128, with a sharp peak at |k| ≈ 100–108** — not Nyquist. The bins with |k| ≥ 192 stay at fp32 rounding noise (~1e-11) all the way through frame 173.

## 2. Growth trajectory (frames 140–173)

Envelope observables are pinned to truth right up to the last frame:

| frame | t | max|η|p / t | ‖η‖₂ p / t | E_p / t |
|---|---|---|---|---|
| 140 | 112.00 | 0.0912 / 0.0915 | 1.033 / 1.035 | 6.099 / 6.119 |
| 160 | 128.00 | 0.0913 / 0.0915 | 1.033 / 1.035 | 6.097 / 6.119 |
| 170 | 136.00 | 0.0910 / 0.0915 | 1.032 / 1.035 | 6.075 / 6.119 |
| 173 | 138.40 | 0.0906 / 0.0915 | 1.035 / 1.035 | 6.103 / 6.119 |

max|η|_pred is **not** rising — an exp fit over frames [140,173] gives slope ≈ −4e-6 (flat/decaying).

The action is entirely in the high-|k| bands. Exponential fits of band-summed |η̂|² over frames [140,173]:

| band | rate b (1/time) | doubling time |
|---|---|---|
| 10 ≤ |k| < 32 | 0.133 | 5.21 |
| 32 ≤ |k| < 96 | 0.205 | 3.39 |
| \|k\| ≥ 96 | **0.574** | **1.21** |

The `|k| ≥ 96` band energy grows from 3e-13 (fr=140) → 3e-13 (fr=150) → 5.5e-12 (fr=160) → 8.3e-7 (fr=170) → 4.7e-6 (fr=173), i.e. **~7 orders of magnitude in ~26 time units**, with the last decade landing in the ~4 frames before NaN. The pred/truth ratio in that band evolves 0.68 → 1.0 (fr=150) → 12 (fr=160) → 416 (fr=165) → 1.2e6 (fr=170) → 8.7e6 (fr=173) — pure runaway relative to a quiescent truth.

Energy `0.5(‖ξ‖² + g‖η‖²)` stays within 0.5% of truth until fr=170, then drops ~0.9% by fr=171 before ticking back up at fr=173 — inconsistent with a genuine physical amplification, consistent with a numerical mode gaining amplitude in the wrong basis and then aliasing back.

## 3. Spectral cascade signature

Top-|k| structure of pred vs truth stays identical in the load-bearing low modes (k = 0…4 amplitudes are within 5% at fr=173). But the **pred − truth** difference spectrum bootstraps a distinct mid-k peak:

| fr | t | top-3 |k| of (pred − truth)_η by amplitude |
|---|---|---|
| 155 | 124.00 | k=2 (1.6e-4), k=1 (1.4e-4), k=3 (1.3e-4)  — low-k phase drift only |
| 160 | 128.00 | k=2 (1.7e-4), k=1 (1.6e-4), k=27 (9.8e-5) — mid-k mode appears |
| 165 | 132.00 | k=2 (1.9e-4), k=1 (1.8e-4), k=29 (7.9e-5) — mid-k around 25–31 growing |
| 170 | 136.00 | **k=101 (2.4e-4), k=102 (2.3e-4), k=25 (2.3e-4)** — new mode near k≈100 dominates |
| 172 | 137.60 | k=27 (4.4e-4), k=3 (4.0e-4), k=107 (3.3e-4) — k≈100 mode still growing |
| 173 | 138.40 | k=3 (9.2e-4), **k=105 (6.0e-4), k=104 (5.7e-4)**, k=27 (5.0e-4) |

In eta the peak of the high-band feature is at **|k|=105**; in gxi it is at **|k|=105** with amp 6.3e-3 (an order of magnitude larger than in η, as expected because gxi ~ |D|η in symbol). Nyquist bins (|k|=256, 384, 480, 512) never rise above fp32 rounding (~1e-11) at any frame — this is not an aliased-Nyquist cascade.

Fine band decomposition of `|η̂|²`:

```
fr  t     [0,8)   [8,16)  [16,32) [32,64) [64,96) [96,128) [128,192) [192,)
140 112.0 1.0e-3  1.4e-7  4.3e-9  5.9e-11 2.5e-13 3.0e-13  1.6e-15   ~1e-19
160 128.0 1.0e-3  1.4e-7  3.7e-8  3.1e-9  1.4e-11 5.5e-12  4.8e-14   ~1e-19
170 136.0 1.0e-3  1.2e-7  3.9e-7  3.4e-8  1.5e-8  8.3e-7   4.6e-10   ~1e-19
173 138.4 1.0e-3  3.4e-7  1.5e-6  5.9e-8  8.8e-8  4.7e-6   2.6e-9    ~1e-19
```

Signature: energy piles into `[96,128)` (peak at k≈105), with `[16,32)` and `[64,96)` shoulders — a **localized narrow-band Lyapunov mode**, not a Kolmogorov-style monotonic tail, and definitely not an aliasing cascade (bins ≥ 192 are quiet).

## 4. Truth-vs-pred divergence

Truth is completely well-behaved at fr=174 — all fields finite, spectral bands unchanged (`|k| ≥ 96` energy stays at ~5e-13, matching the fp32 label floor). The truth trajectory is not doing anything singular here.

Divergence structure:
- **First frame with rel_l2_eta > 0.1: never before NaN.** Rel error is 0.0146 at fr=160, 0.040 at fr=170, 0.060 at fr=172, 0.098 at fr=173, and NaN at fr=174. The overall envelope error crosses 10% only *at* the blowup frame.
- **In which |k| band does pred first depart by ≥2× from truth?**
  - `|k| < 10`: never (envelope stays clean).
  - `10 ≤ |k| < 32`: first at fr=148, t=118.40 (still tiny in absolute value; well below label floor influence).
  - `|k| ≥ 96`: first at fr=152, t=121.60, and grows explosively from there.

So the timeline is: at t≈118 a low-amplitude mid-k mode starts drifting from truth; by t≈122 the `|k|≥96` band has crossed truth; the band then grows at ~0.57 rate for ~17 time units and NaNs at t=139.20. The envelope error only becomes visible in the last ~4 frames.

## 5. Verdict

**B — mid-k Lyapunov mode, sharply localized at |k| ≈ 100–108** (band `[96,128)` dominates, k=105 the exact peak). Doubling time ≈ 1.2 time units, seeded around t≈118 while the envelope is still on-truth, most visible in `pred_gxi` (natural because gxi ~ |D|η amplifies high-k errors), and clearly not an aliased-Nyquist cascade (all bins |k| ≥ 192 stay at fp32 label noise ~1e-11 through frame 173).
