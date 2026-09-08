# Dataset generation

This directory builds the four physical families used by the current paper
dataset:

- finite-depth Stokes;
- Tanaka initial conditions;
- Benjamin--Feir wave groups;
- JONSWAP/TMA random seas.

The families use different initial-condition formulas, but share the same DNO
target, split rules, acceptance checks, batch format, and dataset-array format.
Older implementations remain available in Git history and should not be mixed
into a new dataset.

## Generation flow

Each attempted simulation is numbered in order within one family/split run. Its
family, split, and number determine its random draws. A run then:

1. samples parameters from one declared parameter group;
2. constructs the initial state;
3. runs the nonlinear adjustment when the JONSWAP family requires it;
4. evolves a complete trajectory for rollout families;
5. computes the common Craig--Sulem DNO target;
6. applies the required numerical checks;
7. selects the retained times from accepted trajectories;
8. atomically writes the completed batch; rejected attempts own no rows.

Unexpected errors in the constructor or solver stop the run. Declared numerical
rejections own no rows and are replaced from the same parameter group.

Generation first reads saved batches to recover progress, then finishes each
parameter group in dictionary order. New batches contain one group, up to the
batch-size limit or that group's remaining quota. A group stops the run if it
still needs successes after twice its requested count in attempts. This ordering
controls generation only; it does not balance training epochs.

All accepted trajectory rows stay together in one split. The generator never
makes row-level train/validation/test splits.

## Main modules

Family definitions:

- `stokes_sampling.py` samples finite-depth Stokes states.
- `tanaka_sampling.py` and `tanaka_initial_conditions.py` build Tanaka simulations.
- `benjamin_feir_sampling.py` and `benjamin_feir_jcp09.py` build modulated wave
  groups.
- `jonswap_tma_sampling.py` and `jonswap_tma.py` build random-sea simulations.

Execution:

- `stokes_batch_generator.py` evaluates one static Stokes batch.
- `trajectory_batch_generator.py` evaluates one rollout batch.
- `jonswap_horizon_generator.py` groups JONSWAP simulations by compatible integration
  length so they can run efficiently together.
- `trajectory_family_adapters.py` connects each rollout family to the shared
  generator.

Shared pipeline:

- `pipeline/types.py` defines shared dataset identifiers and saved-array types.
- `pipeline/trajectory_integration.py` runs batched GL2 integrations and constructs
  saved targets.
- `pipeline/trajectory_rollout.py` evaluates complete trajectories and their
  numerical acceptance checks.
- `pipeline/time_selection.py` selects frames and builds dataset rows.
- `pipeline/dno_target.py` computes the stored DNO target.
- `pipeline/batch_storage.py` validates, reads, and writes one completed NPZ per
  batch.
- `pipeline/artifact_io.py` performs atomic JSON and NPZ writes.
- `pipeline/build_dataset.py` combines saved batches into the arrays used by training.
- `pipeline/dataset_generation.py` counts completed batches and generates the
  remaining simulations needed by each parameter group.

## Acceptance

The rollout returns rows only when the simulation passes its numerical checks.
Those checks cover finite values, positive water height, complete time grids,
integration convergence, DNO values, and internal energy drift where applicable.

For JONSWAP, the nonlinear adjustment must first produce a valid handoff; the
autonomous production rollout is checked separately.

## Stored files

Each finished batch is one NPZ containing its family, split, parameter-group
assignments, and any accepted rows. Row ownership identifies accepted attempts;
a batch with no accepted simulations simply omits the row arrays.

The NPZ is written to a temporary sibling and published only after it is
complete. If generation stops before publication, the next run restarts the
same deterministic batch from the beginning.

The final dataset directory contains:

- `eta.npy`, `xi.npy`, `gxi.npy`: float32 fields, each shaped `(rows, grid_points)`;
- `depth.npy`, `time.npy`: float64 values per row;
- `family_id.npy`, `simulation_id.npy`: int16 family and int64 simulation IDs per row;
- `parameter_group_id.npy`, `dataset_split.npy`: parameter-group and split strings per row;
- `frame_index.npy`: int32 frame number per row;
- `x.npy`: the shared float64 spatial grid.

Simulation IDs are scoped to a family/split. Only accepted simulations contribute
rows. Training reads these arrays directly, selects the stored split, and shuffles
its rows together; there is no manifest, trajectory map, or per-family epoch sampling.

## Entrypoints

The supported generation entrypoints live in `scripts/`:

- `generate_paper_dataset.py` saves batches and a run summary for one family/split;
- `build_paper_dataset.py` combines completed runs into one training dataset directory.

The build command requires all 12 generation summaries: four families, each with
train, validation, and test. Pass each using `--run-summary` and supply
`--output-root`. Families must have equal successful simulation counts within each split.
Existing completed batches can be exported without regenerating simulations.
Use a new output directory (the C27 launchers expect `.../c16384_v01024_t01024/arrays`);
allow about 100 GB of additional disk for the full dataset.

## Development checks

From the repository root:

```bash
uv run ruff check solver/gen_data scripts/generate_paper_dataset.py \
  scripts/build_paper_dataset.py
uv run pyright solver/gen_data scripts/generate_paper_dataset.py \
  scripts/build_paper_dataset.py
uv run pytest solver/gen_data/tests solver/gen_data/pipeline/tests
```
