# Tanaka-g0 NaN investigation

Question: **why do the CS-DNO surrogates blow up on the Tanaka manifold?**

## NaN inventory (from `pred_eta` in `tanaka_g0_trajs.npz`)

| Model | Path | j | case_id | h | first_nan_t | note |
|---|---|---|---|---|---|---|
| v9 jacreg (2026-07-01) | `outputs/cs_dno_w512b8_l256_v8_jacreg_20260629_142544/eval_modal_a100_naked/` | 5 | 5 | 0.276 | 96.80 | 1/16 NaN — jacreg pushed NaN later, but still there |
| v8 fftfp64 (fp64 FFT) | `outputs/cs_dno_w512b8_l256_v8_fftfp64_20260625_012616/eval_suite_f64h/` | 5 | 5 | 0.276 | 63.20 | 2/16 NaN — cured case 11 but regressed case 6 |
| v8 fftfp64 | " | 6 | 6 | 0.133 | 157.60 | mid-h regression, unique to this model |
| v8 baseline (fp32) | `outputs/cs_dno_w512b8_l256_v8_2gpu_20260619_024621/eval_suite_f64h/` | 5 | 5 | 0.276 | 139.20 | 2/16 NaN |
| v8 baseline | " | 11 | 11 | 0.234 | 172.80 | 2/16 NaN |
| v7 | `outputs/cs_dno_w512b8_l256_v7_a100x2_resumed_20260618_124459/eval_suite_f64h/` | 5 | 5 | 0.276 | 38.40 | 2/16 NaN |
| v7 | " | 11 | 11 | 0.234 | 160.80 | 2/16 NaN |
| v5 | `outputs/cs_dno_w512b8_l256_v5_2gpu_20260615_141717/eval_suite_f64h/` | 5 | 5 | 0.276 | 41.60 | 2/16 NaN |
| v5 | " | 11 | 11 | 0.234 | 91.20 | 2/16 NaN |

- Rollout config: `dt=0.8, tmax=200, n_t=251, nx=1024, L=2π, substeps=80` → inner `dt≈0.01`.
- All truth `truth_eta` fields are finite for these cases (verified by earlier truth-validity gate).

## Cross-cuts across models

- **Case 5 (h≈0.276, a/h≈0.28) NaNs in every model** — the canonical failure. Test hypotheses that survive across the whole set here.
- **Case 11 (h≈0.234) NaNs in v5/v7/v8-baseline; cured by fftfp64 and by jacreg.** The two independent cures point at what the failure mode *needs* to happen.
- **Case 6 (h≈0.133) NaNs only in fftfp64** — a regression, worth understanding as it may sharpen the fftfp64 tradeoff.
- v9 jacreg first_nan_t=96.80 vs v8-baseline 139.20 on the same case-5. Jacreg *delayed* nothing — it fires later in absolute time but the model was arguably worse at case 5, so this is subtle. Compare growth rates.

## Analysis protocol

Each subagent writes its section to `nan_notes/<slug>.md`. Report has these required subsections:

1. **First-blowup frame** — time, which field NaN'd first (eta / xi / gxi), which |k| band.
2. **Growth trajectory (leading-up window)** — track max|eta|, max|xi|, max|gxi|, ‖eta‖₂, energy `0.5(‖xi‖²+g‖eta‖²)`, and high-k spectral energy `sum_{|k|>=32} |eta_k|²` over the last ~20 frames before blowup. Log the doubling-time / exponential rate if applicable.
3. **Spectral cascade signature** — plot `|eta_k|` vs |k| at t = {first_nan_t − 20, −10, −5, −1, first_nan_t − dt}. Is energy piling up at Nyquist (aliasing) or at mid-k (Lyapunov mode)?
4. **Truth-vs-pred divergence** — is truth well-behaved at first_nan_t? Where in |k| does pred − truth start? When (which t) does divergence exceed 10%?
5. **Verdict** — pick one: **(A) aliased-cascade at Nyquist**, **(B) mid-k Lyapunov mode (~k≈30–60)**, **(C) low-k envelope amplification (~k<10)**, **(D) sudden single-step blowup (no ramp)**, **(E) other — describe**.

Each subagent should run its own diagnostics with a small inline Python script — no shared helper needed. Data lives in the `trajs.npz`; `truth_eta/xi/gxi` and `pred_eta/xi/gxi` have shape `(n_t, NB, nx)`.

---

## Individual reports

_(one per (model, case) — filled in by subagents in `nan_notes/`)_

## Cross-case synthesis

_(final synthesis added after all forensic reports land)_
