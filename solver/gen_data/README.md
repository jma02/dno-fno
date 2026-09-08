# Dataset generation

This directory builds the four physical families used by the current paper
dataset:

- finite-depth Stokes;
- Tanaka initial conditions;
- Benjamin--Feir wave groups;
- JONSWAP/TMA random seas.

The families use different initial-condition formulas, but share the same DNO
target, acceptance checks, batch format, and dataset-array format.
Older implementations remain available in Git history and should not be mixed
into a new dataset.

## Generation flow

Choose the number of accepted simulations for each family run. Generation divides
that count across the family's parameter groups; it imposes no cross-family quota.
Each attempt's family, integer seed, and attempt number determine its random draws.
The generation seed defaults to `2026072210`. A run then:

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

Generation does not assign dataset splits. After pooling completed batches, the
builder randomly splits whole accepted simulations once, globally: 80% train,
10% validation, and 10% test by default, with split seed `42`. Fractions are
applied to simulation counts, not row counts; every simulation's snapshots stay
together. There is no per-family stratification or balance requirement.

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
- `pipeline/dataset_generation.py` counts completed batches and generates the
  remaining simulations needed by each parameter group.

## Acceptance

The rollout returns rows only when the simulation passes its numerical checks.
Those checks cover finite values, positive water height, complete time grids,
integration convergence, DNO values, and internal energy drift where applicable.

For JONSWAP, the nonlinear adjustment must first produce a valid handoff; the
autonomous production rollout is checked separately.

## Stored files

Each finished batch is one NPZ containing its family, seed, parameter-group
assignments, and any accepted rows. Row ownership identifies accepted attempts;
a batch with no accepted simulations simply omits the row arrays.
Files live at `batches/<family>/batch_<number>.npz`; resume verifies the seed
before reusing them. Generation writes no summary files.

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

Simulation IDs are unique across the built dataset. Only accepted simulations
contribute rows. Training memory-maps these arrays, uses the saved splits, and
shuffles all training rows together each epoch. Normalization statistics come
only from training rows and are reused for validation/test. There is no manifest,
trajectory map, per-family reweighting, or per-simulation epoch sampling.

Retained frames per accepted simulation remain family-specific: Stokes contributes
1, Tanaka and Benjamin--Feir 200 each, and JONSWAP/TMA 16. Equal simulation counts
therefore do not mean equal row counts or equal training contributions.

## Entrypoints

The supported generation entrypoints live in `scripts/`:

- `generate_paper_dataset.py` takes a family, simulation count, and optional seed,
  then saves resumable NPZ batches;
- `build_paper_dataset.py` contains the assembly and split logic and writes the
  NPY arrays used by training.

Pass `--input-root` (repeat to pool directories) and `--output-root`. The builder
recursively reads all completed `batch_*.npz` present; generation need not have
reached its quota. No family balance is required. Separate runs of the same family
must use different seeds. The builder's `--seed`, `--validation-fraction`, and
`--test-fraction` control splitting independently of generation.

Existing completed batches can be exported without regenerating simulations;
write to a new output directory. The C27 launchers default to
`outputs/paper_dataset/arrays`. See [`scripts/README.md`](../../scripts/README.md)
for generation, build, and training commands.

## Development checks

From the repository root:

```bash
uv run ruff check solver/gen_data scripts/generate_paper_dataset.py \
  scripts/build_paper_dataset.py
uv run pyright solver/gen_data scripts/generate_paper_dataset.py \
  scripts/build_paper_dataset.py
uv run pytest solver/gen_data/tests solver/gen_data/pipeline/tests
```
