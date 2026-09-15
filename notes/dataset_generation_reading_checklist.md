# Dataset generation reading checklist

Read these in order. Checkmarks track the completed simplification pass.
The main path covers control flow and scientific choices; optional implementation
details can usually be skimmed.

Current workflow: choose simulation counts, generate without dataset splits,
pool completed NPZ batches, split whole simulations once, then train on shuffled
rows. The builder imposes no family quotas; its default is a global 80/10/10 split
with seed `42`. Generation uses seed `2026072210` by default, with no summary files.

## Main path

- [x] `scripts/generate_paper_dataset.py` — requested simulation count, sampling seed, and family dispatch
- [x] `solver/gen_data/pipeline/dataset_generation.py` — read saved progress, then finish each parameter group in batches with replacement attempts for failures
- [x] `solver/gen_data/pipeline/trajectory_config.py` — the three fixed numerical rollout profiles
- [x] `solver/gen_data/stokes_sampling.py` — Stokes parameter distribution
- [x] `solver/gen_data/tanaka_sampling.py` — Tanaka parameter distribution
- [x] `solver/gen_data/benjamin_feir_sampling.py` — Benjamin--Feir parameter distribution
- [x] `solver/gen_data/jonswap_tma_sampling.py` — JONSWAP/TMA parameter distribution
- [x] `solver/gen_data/stokes_batch_generator.py` — sample, construct, label, check, and save one Stokes batch
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
- [x] `scripts/build_paper_dataset.py` — read all `batch_*.npz` under repeated `--input-root` directories, split simulations, and write NPY arrays to `--output-root`; no generation-completion requirement

Continue into `train-jax-10m/util.py` for training-only normalization and shuffled
row batches, then `train-jax-10m/1d_dno_fno_jax.py` for the training loop.
`scripts/launch_c27_paper_dataset_full.sh` checks the arrays and launches C27;
its default dataset is `outputs/paper_dataset/arrays`. Training uses all retained
rows with no family reweighting; equal simulation counts need not mean equal row
counts.

## Physical constructors worth reading separately

- [x] `solver/gen_data/tanaka_initial_conditions.py`
- [x] `solver/gen_data/benjamin_feir_jcp09.py`
- [x] `solver/gen_data/jonswap_tma.py`

## Optional implementation details

- [x] ~~`solver/gen_data/pipeline/simulation_checks.py`~~ — acceptance is represented directly by rows or `None`
- [x] ~~`solver/gen_data/pipeline/batch_artifacts.py`~~ — folded into `batch_storage.py`
- [x] `solver/gen_data/pipeline/batch_storage.py` — atomic, resumable NPZ batch files
- [x] `solver/gen_data/pipeline/types.py` — shared type aliases

Tests are best read beside the corresponding implementation file, not as a
separate sequence.
