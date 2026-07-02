# Rollout forensics: FNO baseline vs DNO+PF on 1D Zakharov

Run: 2026-06-04. Source eval suites:
- FNO baseline: `outputs/fno_w128b6_v3_hclip5_20260514_063811/eval_suite/`
- DNO+pushforward (40 ep): `outputs/dno_w128b6_l64_v3_pf1_20260603_064450/eval_suite/`

## Per-regime worst-IC table (rel-L2 η @ tmax; NaN = blow-up)

| regime | fno_med | dno_med | fno_worst | dno_worst | worst-5 overlap |
|---|---|---|---|---|---|
| tanaka_g0          | 0.246  | 0.099  | 1.520    | NaN(1) | 2 |
| tanaka_g1          | 0.605  | 0.255  | 1.783    | NaN(1) | 3 |
| bf_g0              | 0.174  | 0.130  | 0.523    | NaN(3) | 2 |
| bf_g1              | 0.193  | 0.130  | NaN(1)   | NaN(2) | 2 |
| bf_modal           | 0.245  | 0.127  | NaN(3)   | NaN(2) | 3 |
| linear             | 1.170  | 0.341  | 3.137    | NaN(4) | 3 |
| stokes_deep        | 0.0036 | 0.0010 | 0.005    | 0.002  | 2 |
| stokes_finite      | 0.0035 | 0.0056 | 0.064    | 0.034  | 4 |
| random_sea_deep    | 0.0041 | 0.120  | 0.026    | 0.624  | 4 |
| random_sea_finite  | 0.0042 | 0.091  | 0.208    | 0.872  | 4 |

Mean worst-5 IC overlap = **2.9/5** — the hard ICs are largely shared between
models, but DNO+PF introduces *new* failures on the random-sea regimes that
FNO did not have.

## Physics → failure correlation

Global Pearson r / Spearman ρ across all regimes (n ≈ 156 finite ICs):

- bandwidth: r=+0.38, ρ=+0.20 (DNO); +0.19, −0.11 (FNO)
- kh:        r=−0.21, ρ=−0.44 (FNO); −0.20, −0.34 (DNO) — small-kh = harder
- depth:     r=−0.26, ρ=−0.45 (FNO); −0.23, −0.34 (DNO)

Within-regime ρ — strongest predictors:

- tanaka_g0/g1: **bandwidth** ρ = +0.76/+0.78 (FNO), +0.84/+0.85 (DNO)
- random_sea_deep: **max steepness** ρ = +0.63/+0.84
- stokes_finite: **max steepness** ρ = +0.70/+0.83
- bf_g0/g1: **k_peak** ρ = +0.33–0.87

The global "low depth = bad" is confounding: Tanaka happens to have both
shallow water and broad spectra. **Within-regime, broader spectrum and
higher steepness predict failure.**

## Error-growth signature (worst-3 finite-error ICs by FNO)

| IC | FNO kind | r²_lin | r²_exp | FNO slope | DNO kind | DNO slope |
|---|---|---|---|---|---|---|
| linear/59         | **linear**  | 0.93 | 0.83 | 0.176/t | linear | 0.026/t |
| bf_modal/2000014  | **stepped** | 0.53 | 0.92 | 0.010/t | linear | 0.0004/t |
| bf_g1/1000001     | linear/exp  | 0.82 | 0.95 | 0.014/t | linear | 0.006/t  |

NaN blow-ups (4 ICs under FNO, all in bf_g1/modal) occur **mid-late rollout
(t = 81–189 of tmax=200)**, with |η|_max ≈ 0.04–0.12 right before the NaN —
i.e. *not* a steepening cascade, but a numerical divergence after extended
coherent drift.

Spectral signature:

- linear/59 @ t=2: 99% of error in sideband k=10–50 while truth has all
  power at k=6 → **FNO hallucinates nonlinear sidebands in the linear
  regime** (smoking gun for the linear case).
- bf_modal/2000014 and bf_g1/1000001: error peak band = truth peak band
  (cos-sim 0.57–1.00); error is dominated by **resolved physical modes**,
  not high-k Gibbs (high-k fraction ≈ 0.000–0.18).

**No regime shows high-k Gibbs dominance.** Rollout error is **not** a
sub-grid numerical artifact — it is coherent error inside the band the
model is supposed to predict.

## Where pushforward helps vs not (DNO+PF − FNO, median rel-L2 @ tmax)

- **Big wins:** tanaka_g0 −60%, tanaka_g1 −58%, linear −71%, bf_modal −48%,
  bf_g1 −33% — all long-horizon, broadband, shallow-water regimes.
  Linear-in-t slope cut by 5–25×.
- **Neutral:** stokes_deep (both near-zero). bf_g0 −25%.
- **Regressions:** **random_sea_deep +2799%, random_sea_finite +2088%,
  stokes_finite +60%.** Pushforward broke the random-sea regime where FNO
  had near-zero error (0.004 → 0.12).
- NaN rates also rise under DNO+PF in shallow regimes (bf_g0 19%, linear
  25%; FNO 0% on both).

## Synthesis

Rollout failure mode = **coherent, linear-in-t drift inside the physical
band**, driven by broadband / high-steepness ICs (within-regime ρ up to
+0.85). NOT high-k numerical noise. NOT distribution shift on the IC.

Pushforward training **does fix the linear-in-t drift** (slope 5–25×
smaller) in the regimes it was trained on, but **regresses on random_sea**.
Likely interpretation: the pushforward step trades robustness on
broadband-noise ICs for fidelity on coherent soliton-like ICs.

**Fix should target:**
1. Noise-robustness during pushforward (curriculum, or augmenting with
   random_sea ICs in the pushforward batch).
2. A regularizer that suppresses out-of-band energy in η_pred — the
   linear-case sideband generation suggests directly penalizing harmonic
   creation outside the IC band, especially for small-kh ICs.

## Caveats

1. Only 16 ICs per regime — within-regime ρ has wide CIs.
2. Random-sea regression may be partly explained by DNO+PF having trained
   40 epochs vs FNO's 80 (resume in progress).
3. ξ-error growth not separately analyzed — only η.
4. NaN blow-ups need a step-by-step trace to determine whether they
   originate in the φ or η update.

## Artifacts

- Raw data: `playground/diagnostics/rollout_forensics.json`
- Scripts: `rollout_forensics{,_part2,_part3,_table,_within}.py`
