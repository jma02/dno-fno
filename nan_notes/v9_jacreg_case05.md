# v9 jacreg — case 5 (h=0.276, a/h≈0.28)

- Trajs: `outputs/cs_dno_w512b8_l256_v8_jacreg_20260629_142544/eval_modal_a100_naked/tanaka_g0_trajs.npz`
- Batch index j=5, case_id=5, h=0.2761, g·h=2.7088
- Rollout: dt=0.8, tmax=200, nx=1024, L=2π

## 1. First-blowup frame

- First NaN at frame **121, t = 96.80 s**.
- All three fields (`pred_eta`, `pred_xi`, `pred_gxi`) go NaN in the **same** frame, and the whole 1024-point grid is NaN at that frame (not localized) — the FFT-based operator poisoned everything at once.
- Truth is finite for the entire trajectory (frozen steady-wave, max|η|=0.0915 forever).
- The dominant `|k|` band at the last finite frame (120) is **k ≈ 100–110** (see §3).

## 2. Growth trajectory (frames 90–120)

Bounded oscillation for frames 90–111, then a fast ramp. Truth is a perfect traveling wave (max|η|=0.0915, energy 0.694 for every frame).

| frame | t | max\|η\|_p | max\|ξ\|_p | max\|gξ\|_p | ‖η‖₂ | E_p | E_band<10 | E_10-32 | E_32-96 | E_≥96 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 80.0 | 0.0928 | 0.0608 | 0.0747 | 0.0807 | 0.6887 | 2.16e-5 | 5.23e-5 | 4.22e-7 | ~0 |
| 110 | 88.0 | 0.0922 | 0.0609 | 0.0770 | 0.0805 | 0.6854 | 8.39e-5 | 1.49e-4 | 2.71e-6 | ~0 |
| 112 | 89.6 | 0.1050 | 0.0607 | 0.1491 | 0.0809 | 0.6899 | 2.99e-4 | 8.58e-4 | 1.51e-4 | ~0 |
| 115 | 92.0 | 0.0950 | 0.0614 | 0.1059 | 0.0800 | 0.6790 | 3.84e-4 | 2.74e-4 | 4.59e-5 | ~0 |
| 117 | 93.6 | 0.0951 | 0.0628 | 0.2013 | 0.0799 | 0.6762 | 8.86e-4 | 7.97e-4 | 2.28e-4 | ~0 |
| 118 | 94.4 | 0.1356 | 0.0629 | 0.5078 | 0.0835 | 0.7092 | 5.83e-3 | 8.42e-3 | 2.18e-3 | ~0 |
| 119 | 95.2 | 0.1378 | 0.0623 | 0.4875 | 0.0858 | 0.7415 | 3.84e-3 | 8.26e-3 | 4.02e-3 | ~0 |
| 120 | 96.0 | 0.1851 | 0.0619 | **0.9353** | 0.0890 | 0.7685 | 1.30e-2 | 9.64e-3 | 1.29e-2 | ~0 |
| 121 | 96.80 | **NaN** |

Exponential-rate fits on max|η|_pred:
- Frames 90–120 (30 frames):     a ≈ **0.030 /s** (broad window — mostly flat)
- Frames 111–120 (10 frames):    a ≈ **0.095 /s**, doubling ≈ **7.3 s**
- Frames 107–120 on high-k energy (|k|≥32): a ≈ **0.479 /s**, doubling ≈ **1.45 s** — the high-k spectral energy grows ~10× faster than the max amplitude.

`max|gξ|` leads the blowup: 0.077 → 0.20 → 0.51 → 0.94 in the last 4 frames — surface-gradient of the potential is the field that becomes ill-posed first (in amplitude, though NaN hits simultaneously).

## 3. Spectral cascade signature

Top-5 |k| by amplitude in `pred_eta` (all frames anchored on primary at k=1..4):

| frame | pred (top-5) | truth (top-5) |
|---:|---|---|
| 101 | k=0,1,2,3,4  |k=0,1,2,3,4 (identical to 5e-3) |
| 111 | k=0,1,2,3,4  | same |
| 116 | k=0,1,2,3,4 (k=3 now 5.9e-3 vs 5.68e-3 truth) | same |
| 119 | k=0,1,2,3,4 (k=3 = 7.07e-3, k=4 = 4.65e-3) | k=0,1,2,3,4 (unchanged) |
| 120 | k=0,1,2,3,4 (k=3 = 7.20e-3, k=4 = 5.22e-3) | k=0,1,2,3,4 (unchanged) |

But `pred_gxi` reveals the real cascade — an explicit mid-k band emerging:

- **frame 118**  top-5 |k| in `pred_gxi`: k=2 (1.10e-2), k=3 (1.02e-2), k=4 (8.79e-3), k=1 (7.70e-3), k=5 (7.51e-3), then a distinct band at **k=90–94 (~6.0e-3)** already tied for 6th–10th.
- **frame 120**  top-10 |k| in `pred_gxi`: **k=104,106,105,107,108,103,109,102,99,100** — all ~1.2e-2 to 1.7e-2. The mid-k band has completely taken over.

Band amplification ratios (√Σ|η_k|², frame 100 → frame 120):

| |k| band | A(100) | A(120) | ratio |
|---:|---:|---:|---:|
| 0–5    | 2.59e-2 | 2.68e-2 | 1.04 |
| 5–10   | 1.90e-3 | 5.78e-3 | 3.04 |
| 10–20  | 1.15e-4 | 2.85e-3 | 24.7 |
| 20–32  | 8.82e-5 | 2.13e-3 | 24.2 |
| 32–64  | 2.25e-4 | 2.19e-3 | 9.75 |
| **64–96**   | 2.05e-5 | 2.14e-3 | **105** |
| **96–150**  | 2.03e-5 | 3.55e-3 | **175** |
| 150–250 | 7.12e-10 | 8.76e-10 | 1.23 |
| 250–400 | 1.04e-9 | 9.83e-10 | 0.94 |
| 400–513 | 8.72e-10 | 9.60e-10 | 1.10 |

The band **k ∈ [64, 150]** grows 100–175× while everything at k ≥ 150 stays flat at the f32 noise floor (~1e-9). No pileup at Nyquist (k=512 has A~1e-10 at frame 120 — irrelevant). Fraction of |η|² in k≥256 at frame 120 = 2.3e-15. This is emphatically **not** an aliased-Nyquist cascade.

## 4. Truth-vs-pred divergence

Truth spectrum is invariant across the whole window (steady wave), and truth remains finite forever. Divergence is entirely on the pred side.

- Rel-L² divergence in η first exceeds 0.10 at **frame 116, t = 92.80 s** (rel = 0.115).
- Top-5 |k| in `|pred − truth|` at frame 116: **k = 1, 2, 3, 4, 5** — divergence _starts_ as a low-k sideband error on the primary carrier (k=1 dominant, amplitude 1.68e-3).
- Two frames later (frame 118, t=94.4) `pred_gxi` shows a distinct k≈90 band emerging; by frame 120 that band peaks at k≈104–108 and blows up.

rel_l2_eta trajectory: 0.038 (fr 90) → 0.045 (fr 100) → 0.058 (fr 108) → 0.068 (fr 111) → 0.082 (fr 112, first jump) → 0.10 (fr 115) → 0.12 (fr 116) → 0.24 (fr 118) → 0.31 (fr 119) → 0.43 (fr 120) → NaN.

So the sequence is:
1. Slow low-k phase/amplitude drift of the primary (5-week ramp from 0.04 → 0.07 rel-L²).
2. At t ≈ 89 (frame 112) a mid-k band k∈[64,150] starts amplifying at rate ~0.48/s (doubling 1.5 s).
3. Once the mid-k band amplitude reaches ~1e-2 in gξ (frame 118), max|gξ| ≈ 0.5 and the substep operator saturates → NaN one dt later.

## 5. Verdict

**B — mid-k Lyapunov mode (k ≈ 64–150, peaking at k ≈ 100–108).**

Justification: Growth is broadband but concentrated in a well-defined mid-to-upper-k band (100–175× amplification for k∈[64,150] over 20 frames, doubling ~1.5 s), while both the low-k primary carrier (k<5, ratio 1.04) and the Nyquist tail (k≥150, ratio ~1) are inert — spectral energy above k=150 stays glued to the f32 floor. That rules out (A) aliased-Nyquist and (C) low-k envelope. The 10-frame ramp in max|η| and 14-frame ramp in high-k energy rule out (D) single-step. The pred_gxi mid-k band explicitly locks onto k ≈ 90 (fr 118) → k ≈ 105 (fr 120), which is a coherent unstable mode, not broadband noise.
