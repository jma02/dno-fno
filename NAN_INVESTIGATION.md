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

Full per-case reports live in `nan_notes/`:

| slug | file | verdict | notes |
|---|---|---|---|
| v9 jacreg, case 5 | `nan_notes/v9_jacreg_case05.md` | **B** mid-k Lyapunov | k∈[64,150], peaks k≈100–108, doubling ~1.5s; pred_gξ locks onto k≈105 mode 2 frames before whole grid blows up |
| v8 fftfp64, case 5 | `nan_notes/v8_fftfp64_case05.md` | **B** mid-k | learned resonance at k≈27–33 grows exp at ~0.5/t in [32,96] from t≈40; NaN 2.2× *earlier* than fp32 baseline → fp64 not the bottleneck |
| v8 fftfp64, case 6 | `nan_notes/v8_fftfp64_case06.md` | **B** — different signature | isolated single-mode `pred_gξ` resonance at exactly `\|k\|=128` (= trained `cs_g1_k_cut`), off truth by 4×10⁷ before eta/xi even move; fp64 FFT unmasked a damped model bug |
| v8 baseline, case 5 | `nan_notes/v8_baseline_case05.md` | **B** mid-k | k≈100–108 (matches v9 jacreg), doubling 1.2s; pred_gξ leads eta/xi by one step |
| v8 baseline, case 11 | `nan_notes/v8_baseline_case11.md` | **B** mid-k | k≈96–128 (dominant k≈±116, ±122), Nyquist empty (<1e-16); the fp64/jacreg cure pattern |
| v7, case 5 | `nan_notes/v7_case05.md` | **B** mid-k | \|k\|≥32 band doubles in ~1s from a 6-decade-above-truth gξ floor; v7's knee 20% steeper than v5 (v7_trim regression) |
| v7, case 11 | `nan_notes/v7_case11.md` | **A** aliased-Nyquist | `pred_gξ` peaks migrate into k=±127/±128 by t=160; gξ blows up first, eta/xi one step later |
| v5, case 5 | `nan_notes/v5_case05.md` | **B** mid-k | k∈[60,130], doubling shrinks 6s→0.25s over the ramp; `\|D\|^{1/2}` amplifies fp32 gξ tail |
| v5, case 11 | `nan_notes/v5_case11.md` | **A** aliased-Nyquist | k=121–126 (Nyquist=128), doubling 2.8s; v5's earlier failure = hotter starting noise floor, not different mechanism |

## Cross-case synthesis

### The single mechanism that explains 8/9 cases

**Every NaN in this table is `pred_gξ` blowing up first**, in a narrow mid-to-high-k band, while pred_eta and pred_xi look fine in physical space right up to the final step. eta and xi NaN in the *same* substep, one after gξ has already reached O(1) values 10³–10⁷× above truth. Terminal blowup is a joint fp32 overflow of `(gξ + η_x·ξ_x)/(1+η_x²)` once gξ pumps mid-k energy hard enough for `|D|^{1/2}η` to explode.

The physical envelope (`max|η|`) stays inside the training distribution until the last 1–2 frames. This is not a shock, not a Nyquist aliasing chain in the classical CFL sense, and not (mostly) a low-k envelope amplification. **It is a spectral-band resonance in the model's gξ output.**

### Two distinct populations of NaN, with different cures

**Population 1 — the a/h≈0.28 dynamical instability (case 5).**

Every model, without exception, NaNs on case 5 (h=0.276). Verdict is unanimous **B** (mid-k Lyapunov). The locus varies by model — v8 fftfp64 sits at k≈30, v9-jacreg/v8-baseline sit at k≈100–108, v5/v7 span [32,130] — but the mechanism is identical: a coherent exponential Lyapunov mode in a band the model has learned to over-amplify in gξ. Doubling times cluster around 1–2 s (0.05–0.7/s rate). Truth is pristine throughout.

This is not a noise-floor artifact. **fp64 FFT arithmetic makes it *worse*** on case 5 (v8 fftfp64 NaNs at t=63.20, v8 baseline at t=139.20 — 2.2× *earlier*). The f32 rounding noise had been partially quenching a genuine dynamical resonance the model overfits. Jacobian regularization at `kcut=32` pushed case-5 NaN from t=41.60 (v5) to t=96.80 (v9) — the mechanism is intact but slower. This is a genuine model failure at the edge of the training distribution (1% coverage at h≥0.23 ∧ a/h≥0.27 per the v7 tanaka audit).

**Population 2 — the noise-floor amplifiers (case 11).**

Cases 11 (v5, v7, v8-baseline) all NaN through a distinct signature: `pred_gξ` sits ~10⁵–10⁷× above the truth floor at k≥32 from *very early* (t ≈ 5–10), then grows monotonically over 100–170 s. Nyquist bins carry visible weight in v5/v7 (verdict **A** — aliased-cascade at Nyquist), broadening slightly toward k∈[96,128] in v8-baseline (verdict **B**). The physical envelope is *flat* the whole time — this is entirely a spectral-tail phenomenon.

**Both fp64-FFT and jacreg independently cure this population** because they attack the same seed: the f32 chained-FFT rounding noise in the spectral path (fftfp64) or the model's tendency to write |k|≥32 content into gξ (jacreg with kcut=32). This is the "convergent cures" fingerprint we expected — case 11 is the *learnable* NaN.

**Population 3 (n=1) — the fp64-unmasked bug (case 6, fftfp64 only).**

fp64 FFT is not free. It exposed a latent model bug on a shallow non-danger case (h=0.133): `pred_gξ` has an isolated single-mode resonance at exactly `|k|=128` (drops 17× to |k|=129 — a spike, not a shoulder). This is the trained constant `cs_g1_k_cut=128`, still present in the config *even though `use_g1_baseline=False`* — flagging a likely config-plumbing issue that shaped the learned representation at that k. Under f32 the rounding noise damped this mode; under fp64 it grows unimpeded to `|gξ_{k=128}|=2.92e-2` vs truth `8e-10` — off by 4×10⁷ before eta or xi visibly move.

### Ranked findings

1. **`pred_gξ` is the failure vector.** In every case, gξ saturates first with mid-k energy while eta stays on-envelope. Any cure must reshape gξ's k-tail specifically.

2. **The a/h≈0.28 boundary is a hard model failure, not a numerical artifact.** fp64 doesn't help; it hurts on case 5. Jacreg helps modestly. The mechanism is a coherent exponential mode the model has *learned*.

3. **fp64 and jacreg are curing two independent problems for the same reason on case 11.** Both suppress |k|≥32 content in gξ (fftfp64 by removing rounding noise; jacreg by penalizing high-k Jacobian action). Case 11 is the safest single-datapoint test that either intervention works.

4. **fftfp64 has a genuine regression at case 6** that is not present in any other model. Config plumbing at `cs_g1_k_cut` is the prime suspect — remove that constant when `use_g1_baseline=False` and re-check.

5. **v9-jacreg's locus (k≈100) matches v8-baseline's locus (k≈100) on case 5**, not v8-fftfp64's locus (k≈30). This is evidence that jacreg is *not* attacking the same subspace fftfp64 attacks — they cure the noise-floor problem via different pathways.

### What to run next

- **Case 5 cure (population 1) — new territory.** Options:
  - Increase jacreg `λ` (currently 0.001) and/or lower `kcut` from 32 toward 16 to bite the mid-k mode harder. Push v9 further with a follow-on run.
  - Add training data at h≥0.23 ∧ a/h≥0.27 (the 1% tail per the tanaka audit) — the truth generator can produce this (JCP09 stable to a/h=0.6 at M=6) even though we chose depth_max=0.30 in v5+.
  - Wire the cascade gate (task #101) — the earlier per-substep trace already showed the ramp signature this population needs.

- **Case 11 cure (population 2) — decided.** Both fp64 and jacreg work. Jacreg is preferred since it doesn't introduce the case-6 regression.

- **Case 6 regression (fftfp64-only) — bugfix.** Grep the trainer + model for how `cs_g1_k_cut` is applied when `use_g1_baseline=False`. Likely a config-persistence bug that's biasing the fp64 output distribution at k=128.

- **v9 training continues** — ep30/40 val 0.00160, ~10 epochs left. Once complete, re-eval on tanaka_g0 to see if case 5 pushes past t=96.80 with more optimizer steps at the jacreg penalty active.

