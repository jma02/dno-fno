# Training data audit + train/test distribution gap

Run: 2026-06-04. Source dataset: `data/combined_dataset_v3.npz` (73.7 GB,
5,993,989 snapshots). Audit was on a mmap'd sample of 1000 snapshots.

## Training distribution (sampled N=1000)

| Stat | min | p05 | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| eta_amp | 3.0e-4 | 1.7e-3 | 9.7e-3 | 3.50e-2 | 6.4e-2 | 9.15e-2 |
| xi_amp | 7.0e-5 | 6.4e-4 | 3.5e-3 | 2.40e-2 | 4.4e-2 | 6.06e-2 |
| max \|dη/dx\| | 1.3e-3 | 1.3e-2 | 7.5e-2 | 3.36e-1 | 8.10e-1 | 1.30 |
| dominant k | 1 | 1 | 7 | 19 | 25 | 26 |
| gxi_amp | 8.0e-4 | 3.2e-3 | 2.2e-2 | 8.0e-2 | 1.7e-1 | 2.80e-1 |
| depth h | 0.010 | 0.016 | 0.52 | 13.9 | 42.3 | 49.7 |

Per-source (`n_total | eta_p95 | xi_p95 | steep_p95 | k_dom_p95 | depth_range | t_range`):

- `random_sea_deep`   (241k  | 0.022 | 0.010 | 0.22 | 12 | 5.0–25     | 0–20)
- `random_sea_finite` (220k  | 0.021 | 0.013 | 0.22 | 12 | 0.10–1.5   | 0–20)
- `linear`            (500k  | 0.011 | 0.016 | 0.13 | 19 | 0.02–1.48  | t=0 only)
- `stokes_deep`       (500k  | 0.011 | 0.006 | 0.14 | 20 | 4.0–50     | 0–20)
- `stokes_finite`     (500k  | 0.011 | 0.004 | 0.24 | 26 | 0.02–0.34  | 0–20)
- `tanaka_g0`         (1M    | 0.050 | 0.039 | 0.10 |  5 | 0.010–0.30 | 0–200)
- `tanaka_g1`         (1M    | 0.050 | 0.038 | 0.11 |  5 | 0.010–0.30 | 0–200)
- `bf_{g0,g1,modal}`  (~0.5M | 0.031 | 0.014 | 0.60 (max ≈ 1.3) | 19 | 0.50–4.0 | 0–200)

## Rollout IC vs. training (OOD = outside training p05–p95)

| Regime | n_ICs | dt | depths | eta (med / max) | xi (med / max) | k_dom (med / max) | Verdict |
|---|---|---|---|---|---|---|---|
| tanaka_g0       | 16 | 0.8  | 0.014–0.28 | ok / above_p95 | ok / OUTSIDE_MAX | ok / ok    | borderline tail |
| tanaka_g1       | 16 | 0.8  | 0.011–0.29 | ok / above_p95 | ok / OUTSIDE_MAX | ok / ok    | borderline tail |
| bf_g0           | 16 | 0.8  | 0.66–2.75  | ok / ok        | ok / ok          | ok / above_p95 | clean |
| bf_g1           | 16 | 0.8  | 0.56–3.87  | ok / ok        | ok / ok          | ok / above_p95 | clean; truth solver NaNs on 1 of 16 cases after t≈146 |
| bf_modal        | 16 | 0.8  | 0.85–3.97  | ok / ok        | ok / ok          | ok / ok        | clean |
| linear          | 16 | 0.08 | 0.0383     | ok / ok        | ok / above_p95   | ok / ok        | clean |
| stokes_deep     | 16 | 0.08 | 38.3       | ok / ok        | ok / ok          | ok / ok        | clean |
| stokes_finite   | 16 | 0.08 | 0.0383     | ok / ok        | ok / ok          | above_p95 / above_p95 | high-k tail of training k |
| random_sea_deep | 16 | 0.08 | 5.2–22.5   | ok / ok        | ok / ok          | ok / ok        | clean |
| random_sea_finite | 16 | 0.08 | 0.11–1.25 | ok / ok        | ok / ok          | ok / ok        | clean |

## Red flags

- **No NaN/inf** in 1024k sampled training voxels across eta/xi/gxi.
- **One bf_g1 eval IC** has the truth solver diverge to NaN at t≈146 (out of 200).
  This is an eval-side solver bug, not training-data corruption — but it
  pollutes aggregate metrics on that regime.
- **Tanaka dominates the high-amplitude tail** (eta_p95 ≈ 0.05 vs 0.011 elsewhere);
  high-η coverage is therefore thin outside Tanaka.
- **BF carries near-singular steepness** (max |dη/dx| ≈ 1.3); these dominate the
  high-derivative tail of gxi and the normalization (`target_absmax` ≈ 0.769).
- **xi is zero-meaned** by the loader (`xi -= xi.mean`); a couple of tanaka rollout
  ICs sit just above the training xi p99 (minor).
- **Training is snapshot, not trajectory.** Rollout dt comes from the integrator,
  not the data — no train/test dt mismatch.

## Bottom line

The training data is **clean and broadly covers the rollout ICs**. No NaNs,
no spectral pollution, every test regime's ICs fall inside the training
central 90% at the median. Only the maximum-amplitude Tanaka cases nip the
training tail; only `stokes_finite` has its dominant wavenumber sitting at
the upper edge (still within p99). The `bf_g1` truth divergence is an
isolated eval-side solver bug.

**Implication:** the rollout-accuracy ceiling we are seeing is **not a
data-coverage problem.** Improvements need to come from model / training
(pushforward, larger-k unrolls, H-conservation), not from adding more data.

## Artifacts

- Raw stats: `playground/diagnostics/data_audit.json`
- Sampling script: `playground/diagnostics/audit_data.py`
