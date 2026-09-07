# Dataset generation reading checklist

Read these in order. Checkmarks track the completed simplification pass.
The main path covers control flow and scientific choices; optional implementation
details can usually be skimmed.

## Main path

- [x] `scripts/generate_paper_dataset.py` — command-line entry point and family dispatch
- [x] `solver/gen_data/pipeline/dataset_generation.py` — read saved progress, then finish each parameter group in batches with replacement attempts for failures
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
- [x] `solver/gen_data/pipeline/trajectory_integration.py` — batched GL2 integration
- [x] `solver/gen_data/pipeline/dno_target.py` — supervised DNO target construction
- [x] `solver/gen_data/pipeline/trajectory_rollout.py` — rollout assembly and numerical acceptance checks
- [x] ~~`solver/gen_data/pipeline/trajectory_checks.py`~~ — checks were inlined into `trajectory_rollout.py`
- [x] `solver/gen_data/pipeline/time_selection.py` — saved-time grids, frame selection, and dataset rows
- [x] ~~`solver/gen_data/pipeline/trajectory_subsampling.py`~~ — folded into `time_selection.py`
- [x] ~~`solver/gen_data/pipeline/writer.py`~~ — folded into `batch_storage.py`
- [x] `solver/gen_data/pipeline/build_dataset_view.py` — loader manifest and row-to-simulation map
- [x] `scripts/build_paper_dataset_view.py` — combination of all family/split runs

## Physical constructors worth reading separately

- [x] `solver/gen_data/tanaka_initial_conditions.py`
- [x] `solver/gen_data/benjamin_feir_jcp09.py`
- [x] `solver/gen_data/jonswap_tma.py`

## Optional implementation details

- [x] ~~`solver/gen_data/pipeline/simulation_checks.py`~~ — acceptance is represented directly by rows or `None`
- [x] ~~`solver/gen_data/pipeline/batch_artifacts.py`~~ — folded into `batch_storage.py`
- [x] `solver/gen_data/pipeline/batch_storage.py` — NPZ save/load boundary
- [x] `solver/gen_data/pipeline/artifact_io.py` — atomic JSON/NPZ helpers
- [x] `solver/gen_data/pipeline/types.py` — shared type aliases

Tests are best read beside the corresponding implementation file, not as a
separate sequence.
