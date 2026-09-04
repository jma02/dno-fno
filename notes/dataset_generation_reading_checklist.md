# Dataset generation reading checklist

Read these in order. The first section covers the control flow and scientific
choices; the second section is implementation detail that can usually be
skimmed.

## Main path

- [x] `scripts/generate_paper_dataset.py` — command-line entry point and family dispatch
- [x] `solver/gen_data/pipeline/dataset_generation.py` — functional generation loop: balance accepted simulations, retry failures, and resume completed batches
- [x] `solver/gen_data/pipeline/trajectory_config.py` — the three fixed numerical rollout profiles
- [x] `solver/gen_data/stokes_sampling.py` — Stokes parameter distribution
- [x] `solver/gen_data/tanaka_sampling.py` — Tanaka parameter distribution
- [x] `solver/gen_data/benjamin_feir_sampling.py` — Benjamin--Feir parameter distribution
- [x] `solver/gen_data/jonswap_tma_sampling.py` — JONSWAP/TMA parameter distribution
- [x] `solver/gen_data/stokes_static_pipeline.py` — static Stokes construction, target, and checks
- [x] `solver/gen_data/stokes_batch_generator.py` — one function that samples, checks, and saves a Stokes batch
- [x] `solver/gen_data/trajectory_batch_generator.py` — functional Tanaka/BF/JONSWAP batch flow
- [x] `solver/gen_data/trajectory_family_adapters.py` — sampled parameters to solver-ready initial states
- [x] `solver/gen_data/jonswap_horizon_generator.py` — JONSWAP adjustment and memory-bounded grouping by rollout length
- [ ] `solver/gen_data/pipeline/trajectory_integration.py` — batched GL2 integration
- [ ] `solver/gen_data/pipeline/dno_target.py` — supervised DNO target construction
- [ ] `solver/gen_data/pipeline/trajectory_rollout.py` — rollout assembly and per-simulation results
- [ ] `solver/gen_data/pipeline/trajectory_checks.py` — numerical acceptance checks
- [ ] `solver/gen_data/pipeline/time_selection.py` — dense-time index selection
- [ ] `solver/gen_data/pipeline/trajectory_subsampling.py` — retained training rows per accepted simulation
- [ ] `solver/gen_data/pipeline/writer.py` — conversion of results into a completed batch
- [ ] `solver/gen_data/pipeline/build_dataset_view.py` — loader manifest and row-to-simulation map
- [ ] `scripts/build_paper_dataset_view.py` — combination of all family/split runs

## Physical constructors worth reading separately

- [ ] `solver/gen_data/tanaka_initial_conditions.py`
- [ ] `solver/gen_data/benjamin_feir_jcp09.py`
- [ ] `solver/gen_data/jonswap_tma.py`

## Optional implementation details

- [ ] `solver/gen_data/pipeline/simulation_checks.py` — small named result record
- [ ] `solver/gen_data/pipeline/batch_artifacts.py` — in-memory saved-batch layout
- [ ] `solver/gen_data/pipeline/batch_storage.py` — NPZ save/load boundary
- [ ] `solver/gen_data/pipeline/artifact_io.py` — atomic JSON/NPZ helpers
- [ ] `solver/gen_data/pipeline/types.py` — shared type aliases

Tests are best read beside the corresponding implementation file, not as a
separate sequence.
