# Tier 3I: canonical Tanaka h=1 data pack

Doc reference: `notes/cs_dno_jcp09_improvement_plan.md` §3I; `notes/v7_tanaka_failure_audit.md`.

## Goal

Add the JCP09-canonical Tanaka solitary-wave regime (`h = 1`, `a/h ∈ [0.3, 0.6]`) to the training corpus. The current dataset samples `h ∈ [0.01, 0.30]` exclusively — it *never* sees the paper's reference configuration. The audit shows the failing regime `(h ≥ 0.23 ∧ a/h ≥ 0.27)` has **~1% coverage** in v8.

## What to generate

| param | value | notes |
|---|---|---|
| depth | `h = 1.0` | Tanaka's canonical depth |
| amplitudes | `a/h ∈ {0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60}` | the high-`a/h` envelope JCP09 reports stable |
| domain | `L = 6.283185...` (2π) — same as v8 — OR `L = 4π` larger | larger domain reduces wrap-error from 3-copy periodic extension |
| copies | 5-copy periodic extension (audit recommends over current 3-copy) | matches generator's safety-margin comment |
| samples per `a/h` | ~5,000 trajectory snapshots | total ~35k samples — keep pack proportional to existing source-12 share |
| solver | `gl2_if`, dno_order=6, pad_factor=8, filter_fraction=0.25 | identical to v8 labels |

## Code changes

1. **`solver/gen_data/generate_tanaka_dataset_v2.py`** — add a `--depth_canonical` mode that overrides depth_max=1.0 and uses 5-copy extension. Audit (notes/v7_tanaka_failure_audit.md) already specifies the 5-copy code path.

2. **Run with**:
   ```bash
   uv run python -m solver.gen_data.generate_tanaka_dataset_v2 \
     --depth_canonical \
     --a_over_h 0.30 0.35 0.40 0.45 0.50 0.55 0.60 \
     --copies 5 \
     --output data/tanaka_h1_canonical.npz
   ```

3. **Dataset assembly** — combined_dataset_v9 = combined_dataset_v8 + tanaka_h1_canonical as source 14. Roughly +35k samples on 5.95M base (~0.6%).

## Effort + dependencies

- Generator extension: ~50 LOC, ~2h. Mostly an extension of v2 generator.
- Generation wall-time: hours on 2-GPU. Tanaka trajectories at `h=1` are *cheaper* than at h~0.05 (no orbital-amplitude blow-up), so likely <4h for 7 × 5k = 35k samples.
- Dataset assembly: minutes.
- Retrain v9: full 40-ep on 2-GPU = ~24h.

## When to generate

Per the doc's decision tree: only if v8.5 (A+B+C bundle) and v8.6 (A+B+C+D) both show residual tanaka failures. The hypothesis is that h=1 anchoring lets the model learn the deep-water tanh saturation correctly; if filtering + arch fixes already pass, the new data may not be needed.

If used: bundles with Tier 2D (`use_g0_eta`) and Tier 3H (`lin_exact` rollout) for v9.

## Diagnostic upgrade

After v9 trains, evaluate on:
- Existing 16-IC tanaka_g0/g1 set (h ∈ [0.01, 0.30])
- **New** 7-IC tanaka_h1_canonical eval set (`h = 1`, `a/h` sweep) — confirm we now match the paper's stability envelope.

JCP09 reports stable rollouts at `a/h ≤ 0.6` for `h = 1`; if v9 fails above `a/h = 0.5`, we've over-fit to the shallow regime and may need to upweight the new pack via sample-replication in v10.
