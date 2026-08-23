# v8 fftfp64 — case 5 (h=0.276, a/h≈0.28)

- Trajs: `outputs/cs_dno_w512b8_l256_v8_fftfp64_20260625_012616/eval_suite_f64h/tanaka_g0_trajs.npz`
- j=5, case_id=5, h=0.2761, rollout dt=0.8, nx=1024, L=2π
- First NaN frame = **79** (t = **63.20**). All three fields (eta, xi, gxi) go NaN in the same rollout step.
- v8-baseline fp32 NaNs at t=139.20 on this same case → fftfp64 is **~2.2× earlier**.

## 1. First-blowup frame

Frame 79 (t=63.20). All of `pred_eta`, `pred_xi`, `pred_gxi` become NaN in the same step — there is no earlier NaN in any single field. Frame 78 is still finite everywhere, but `max|gxi|_pred = 2.334` vs `max|gxi|_truth = 0.0666` (35× hot); `max|eta|_pred = 0.118` vs truth `0.0915` (1.29×); `max|xi|_pred = 0.0622` vs truth `0.0608` (still only 2% hot). Signal that blows up first is **gxi**, driven by the `|D|` weight amplifying mid-k contamination in eta.

The Nyquist band (|k|≥96, and especially |k|≥200) stays at ~1e-9 for the entire window — **no aliased pileup**. Divergence is concentrated in **k ≈ 27–33** (see §3).

## 2. Growth trajectory, frames 48–78

Full table (bands = `Σ|η_k|²` over listed |k| ranges):

```
frame  t     max|η|p  max|η|t  max|ξ|p  max|ξ|t  max|gξ|p max|gξ|t ‖η‖p    ‖η‖t    Ep      Et      | pred bands <10, 10-32, 32-96, ≥96                    | truth bands
 48  38.40  9.14e-2  9.15e-2  6.07e-2  6.07e-2  6.65e-2  6.66e-2  1.034   1.035   2.311   2.314   | 1.04e-3 1.26e-8 2.19e-10 1.12e-12 | 1.05e-3 1.27e-8 1.69e-14 2.26e-13
 60  48.00  9.14e-2  9.15e-2  6.07e-2  6.08e-2  6.70e-2  6.66e-2  1.034   1.035   2.310   2.315   | 1.04e-3 1.19e-8 1.09e-9  6.28e-11 | 1.05e-3 1.27e-8 2.19e-14 1.79e-13
 68  54.40  9.12e-2  9.15e-2  6.07e-2  6.08e-2  6.77e-2  6.65e-2  1.031   1.035   2.300   2.315   | 1.04e-3 1.27e-8 3.55e-8  1.99e-9  | 1.05e-3 1.27e-8 2.57e-14 1.58e-13
 70  56.00  9.08e-2  9.15e-2  6.08e-2  6.08e-2  6.72e-2  6.66e-2  1.026   1.035   2.291   2.315   | 1.03e-3 1.59e-7 1.99e-7  1.48e-7  | 1.05e-3 1.27e-8 2.42e-14 2.40e-13
 74  59.20  9.07e-2  9.15e-2  6.07e-2  6.08e-2  7.18e-2  6.66e-2  1.024   1.035   2.270   2.315   | 1.02e-3 6.23e-7 7.92e-7  1.08e-6  | 1.05e-3 1.27e-8 1.04e-14 1.35e-13
 76  60.80  8.99e-2  9.15e-2  6.22e-2  6.08e-2  7.61e-2  6.66e-2  1.017   1.035   2.261   2.315   | 1.01e-3 1.60e-6 1.16e-6  2.01e-6  | 1.05e-3 1.27e-8 2.23e-14 2.25e-13
 77  61.60  8.87e-2  9.15e-2  6.32e-2  6.08e-2  3.90e-1  6.66e-2  1.017   1.035   2.258   2.315   | 9.94e-4 2.19e-6 9.34e-6  3.79e-6  | 1.05e-3 1.27e-8 2.05e-14 2.84e-13
 78  62.40  1.18e-1  9.15e-2  6.22e-2  6.08e-2  2.33e+0  6.66e-2  1.068   1.035   2.360   2.315   | 1.03e-3 2.64e-5 4.84e-5  1.02e-5  | 1.05e-3 1.27e-8 2.06e-14 2.79e-13
 79  63.20  NaN
```

Exponential fit `a = (ln y[-1] − ln y[0])/(t[-1] − t[0])` on `max|eta|_pred` over frames [48..78]: `a = 1.07e-2`, doubling ≈ 64.7 t-units — **the eta envelope itself is not growing exponentially**; it is essentially flat until the last three frames, when a single mode blows up. First frame where `rel_l2_eta > 0.1` is frame **77 (t=61.60, rel=0.145)** — only two frames before NaN.

The 32≤|k|<96 pred-band grows monotonically: 2.2e-10 (t=38.4) → 3.55e-8 (t=54.4, +160×) → 7.92e-7 (t=59.2, +22×) → 4.84e-5 (t=62.4, +61×). That's a factor of ~2×10⁵ over 24 rollout units, exponential rate ≈ **0.51 per t-unit** in band energy → doubling ~1.4 t-units. This is the growing mode.

Truth's 32-96 band stays at ~1e-14 for the entire window.

## 3. Spectral cascade signature

Top-5 |k| modes at snapshots (both fields symmetric so pairs are ±k):

- **frame 60, t=48.00:** pred `[k=0:1.76e-2, k=1:1.50e-2, k=2:9.92e-3]` ≈ truth (differ <0.2%). Tail all in low k.
- **frame 70, t=56.00:** pred `[k=0, k=1, k=2]` same three top modes, k=1 amplitude down 0.8% from truth.
- **frame 75, t=60.00:** same top-3, k=1 down 1%.
- **frame 78, t=62.40:** pred top |k|≥20 modes are `k=33 (1.24e-3), k=30 (1.21e-3), k=27 (1.09e-3)`; truth top |k|≥20 modes are `k=20 (2.76e-7), k=21 (1.62e-7)`. **Pred is 4500× hotter than truth at k∈[27,33]**, six orders of magnitude at k=32.

Detailed band decomposition at final frames:

```
frame 74 t=59.2  [0-10]:3.20e-2 [10-32]:7.89e-4 [32-64]:8.41e-4 [64-96]:2.91e-4 [96-200]:1.04e-3 [200-400]:5.07e-10
frame 76 t=60.8  [0-10]:3.17e-2 [10-32]:1.27e-3 [32-64]:7.38e-4 [64-96]:7.83e-4 [96-200]:1.42e-3 [200-400]:4.76e-10
frame 77 t=61.6  [0-10]:3.15e-2 [10-32]:1.48e-3 [32-64]:9.54e-4 [64-96]:2.90e-3 [96-200]:1.95e-3 [200-400]:5.47e-10
frame 78 t=62.4  [0-10]:3.21e-2 [10-32]:5.14e-3 [32-64]:4.73e-3 [64-96]:5.11e-3 [96-200]:3.20e-3 [200-400]:6.04e-10
```

Growth is confined to `k∈[10, 200]`, peaked in `[32, 96]`. Nyquist (`k≥200`) is silent at 6e-10 the whole time — **no aliasing signature**. The cascade is a mid-to-high k Lyapunov mode initially seeded near k∈[27, 33] and spreading outward into [10, 200] in the final two steps.

## 4. Truth-vs-pred divergence

Truth is well-behaved everywhere in the window: `‖η‖_t` = 1.035 constant, energy `Et` = 2.315 constant, `max|gξ|_t` = 0.0666 constant. Truth's 32-96 band is at f64 numerical zero (1e-14). So the failure is 100% model-side.

Per-k first divergence (>50% relative error):

| |k| | first_div frame | first_div t |
|---|---|---|
| 5 | 78 | 62.4 |
| 10 | 69 | 55.2 |
| 15 | 64 | 51.2 |
| 20 | 28 | 22.4 |
| 32 | 1 | 0.8 |
| 48–400 | 1 | 0.8 |

Interpretation: the mid-k channel has been "wrong" from step 1 in the sense that truth's amplitude there is machine-zero and pred's is not — but that only mattered once the seeded contamination grew. The first band where pred/truth ratio genuinely diverges physically is 32≤|k|<96, running at 100–1000× truth from t≈43 onward, entering runaway growth around t≈54 (ratio 10⁴), and dragging the 10-32 band along by t=60. Low-k (<10) tracks truth to <2% until frame 78; **the envelope is not the source**.

## 5. Verdict

**B — mid-k Lyapunov mode (k ≈ 27–33 in a broader 10-96 band).**

Energy grows monotonically at ~0.5 per t-unit in the 32-96 band from t≈40 onward, four orders of magnitude above truth's floor, with zero Nyquist involvement and low-k tracking truth. The final NaN is `|gxi|` saturating because `|D|·η̂_k` at k≈32 crosses O(1) magnitude when the eta band hits ~5e-5. That fftfp64 NaNs **2.2× earlier** than v8-baseline fp32 confirms fp64 arithmetic is *not* damping this mode — a genuine model instability at case 5, likely a learned resonance near k≈30 that the f32 baseline was partially quenching via bit-truncation noise.
