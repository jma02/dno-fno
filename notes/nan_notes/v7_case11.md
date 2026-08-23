# v7 — case 11 (h=0.234) NaN forensic report

**Model**: `cs_dno w512/b8/l256`, fp32, trained on `v7_trim`.
**Traj file**: `outputs/cs_dno_w512b8_l256_v7_a100x2_resumed_20260618_124459/eval_suite_f64h/tanaka_g0_trajs.npz`
**Case**: j = 11, `case_ids[j] = 11`, `depths[j] = 0.23385`.
**Rollout**: `dt=0.8, tmax=200, n_t=251, nx=1024, L=2π, endpoint=False`.

## 1. First-blowup frame

All three `pred` fields — `eta`, `xi`, `gxi` — go non-finite at the **same** frame:

- `first_nan_frame_pred_eta = 201`, `first_nan_frame_pred_xi = 201`, `first_nan_frame_pred_gxi = 201`
- `first_nan_t = 160.80`
- All three `truth_*` are finite over the full window (no truth blow-up — this is model-side).

The last finite frame is `fr=200, t=160.00`. At `fr=200`, `pred_gxi` is already visibly blown up: `max|gxi|_pred = 6.22e-1` vs `max|gxi|_truth = 4.60e-2` (13.5×), while `max|eta|_pred = 7.78e-2` vs truth `6.40e-2` (only 1.22×). So the field that lost containment *first* is **`gxi`** — the Hou-Li / `|D|^{1/2}η`-like surface-derivative field — and the eta/xi NaN one step later is the downstream consequence.

**Band where the departure originates: the high-k tail `|k|≥32`.** The pred−truth spectral energy in bands `32≤|k|<96` and `|k|≥96` lifts off the noise floor around `fr≈181-183` (t≈145-146), 15+ frames before the eta amplitude visibly departs.

## 2. Growth trajectory (frames 170..201, dt=0.8)

Key columns (pred/truth), truth is essentially frozen in this window:

| fr | t | max\|eta\|_p | max\|xi\|_p | max\|gxi\|_p | ‖eta‖₂_p | E_pred | E_truth |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 170 | 136.00 | 6.39e-2 | 4.45e-2 | 4.60e-2 | 5.46e-2 | 4.30e-3 | 4.32e-3 |
| 180 | 144.00 | 6.37e-2 | 4.50e-2 | 4.58e-2 | 5.45e-2 | 4.33e-3 | 4.32e-3 |
| 184 | 147.20 | 6.48e-2 | 4.50e-2 | **6.12e-2** | 5.53e-2 | 4.30e-3 | 4.32e-3 |
| 188 | 150.40 | 6.32e-2 | 4.82e-2 | **6.50e-2** | 5.51e-2 | 4.48e-3 | 4.32e-3 |
| 192 | 153.60 | 6.30e-2 | 5.12e-2 | **1.09e-1** | 5.80e-2 | 4.42e-3 | 4.32e-3 |
| 196 | 156.80 | 6.52e-2 | 5.71e-2 | **4.09e-1** | 6.23e-2 | 5.06e-3 | 4.32e-3 |
| 199 | 159.20 | 7.13e-2 | 6.46e-2 | **4.29e-1** | 7.08e-2 | 5.57e-3 | 4.32e-3 |
| 200 | 160.00 | 7.78e-2 | 6.18e-2 | **6.22e-1** | 8.22e-2 | 6.21e-3 | 4.32e-3 |
| 201 | 160.80 | NaN | NaN | NaN | — | — | 4.32e-3 |

- `max|gxi|_pred` doubles ~every 2 frames (~1.6 s) from fr=192 onward — this is the leading indicator.
- `max|eta|_pred` moves last: still within 22 % of truth at `fr=200`; a **single-step jump to NaN at 201**.

**Exponential rate on `max|eta|_pred`:**
- Full window 170..200:  λ ≈ 4.86e-3 / unit t  (doubling time ~143 s) — slow drift.
- Late window 190..200:  λ ≈ 3.05e-2 / unit t  (doubling time ~23 s) — superexponential.

So the ramp is neither uniform-exponential nor a single-step surprise; it's a **cascade that percolates for ~20 frames in `gxi` while `eta` stays quiet**, then the last two frames snap.

**Energy signature**: E_pred drifts from 4.32e-3 → 6.21e-3 (+44 %) in 30 frames — nearly all of the drift lands in `‖xi‖²` and in the high-k tail, not in `g‖eta‖²`.

**First frame where `rel_l2_eta` > 0.1**: `fr=189`, `t=151.2` (value 0.1144). By fr=200, `rel_l2_eta = 1.14` — pred has essentially lost eta before it NaN's.

## 3. Spectral cascade signature

`pred_eta` spectral bands (Parseval; truth is essentially constant across the window at `<10: 4.87e-1`, `10-32: 1.02e-5`, `32-96: ~1e-12`, `≥96: ~4e-11`):

| fr | t | P(<10) | P(10-32) | P(32-96) | P(≥96) |
|---:|---:|---:|---:|---:|---:|
| 180 | 144.00 | 4.85e-1 | 9.06e-6 | **3.69e-7** | **1.03e-5** |
| 184 | 147.20 | 4.97e-1 | 3.73e-5 | 3.56e-5 | 6.84e-5 |
| 188 | 150.40 | 4.95e-1 | 1.19e-4 | 9.65e-5 | 2.73e-4 |
| 192 | 153.60 | 5.46e-1 | 2.54e-4 | 7.07e-4 | 7.88e-4 |
| 196 | 156.80 | 6.22e-1 | 2.89e-3 | 4.09e-3 | 4.36e-3 |
| 200 | 160.00 | 1.02e+0 | 1.48e-2 | 2.04e-2 | 5.04e-2 |

Relative to the truth floors:
- `≥96` band jumps from **4e-11 → 1e-5** by fr=180 (5 orders above truth), then blows to `5e-2` — that's 9 orders above the truth floor in 20 frames.
- `32-96` band is a similar story: **1e-12 → 4e-7** by fr=180 (6 orders above truth), out to `2e-2` by fr=200.
- `<10` band inflates by only ~2× over the same span.

**`pred_gxi` is worse**: bands `32-96 / ≥96` blow up to **3.37 / 12.2** by fr=200 — the `|D|^{1/2}η`-type spectral weighting is amplifying the same tail structure by roughly k^{1/2}, producing multi-order-of-magnitude gxi high-k content.

**Peak-k location.** Through frames 180..197 the top |eta_k| peaks stay at k = ±1, ±2, ±3, ±4 (the Tanaka carrier). At frame 199 a **k = ±127 peak** shows up in the top-10; by frame 200 the top-10 now includes **k = ±127 and ±128** (Nyquist = 512, so this is not literal aliasing yet — but it is the far-tail Nyquist-band saturating up into the visible top-10). The blow-up is *not* mid-k (k≈30-60) — it is unambiguously high-k / Nyquist tail.

## 4. Truth-vs-pred divergence

- Truth is finite and quasi-stationary throughout: `max|eta|_truth ≈ 6.40e-2`, energy `4.32e-3`, all bands `≥32` stay at machine-noise levels (1e-11 to 1e-12).
- **Where in |k| pred first departs from truth**: pred−truth diff energy in `32-96` and `≥96` bands is already 1e-9 → 1e-6 by fr=170 (t=136) and lifts to 1e-5 by fr=180 (t=144). The `<10` diff band grows slower and lifts one order at a time.
- **When divergence exceeds 10 %**: `rel_l2_eta > 0.1` first at **fr=189, t=151.2**, about 9.6 s (12 frames) before NaN.

The truth manifold is well-behaved at every frame including the ones bracketing the NaN — this is a pure model-side failure.

## 5. Verdict

**(A) aliased-cascade at Nyquist.**

Pred-side spectral energy in `32-96` and `≥96` bands is 5-9 orders above truth for 20 frames before eta visibly leaves; peaks migrate into k=±127/±128 in the final frames; `gxi` (the `|D|^{1/2}η` field) explodes first and by an order of magnitude more than eta, consistent with a high-k tail being amplified by the surface-derivative operator until the DNO nonlinearity feeds it back into eta/xi.
