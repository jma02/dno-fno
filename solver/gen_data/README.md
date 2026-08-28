# Dataset generation

This directory builds the four physical families used by the current paper
dataset:

- finite-depth Stokes;
- Tanaka initial conditions;
- Benjamin--Feir wave groups;
- JONSWAP/TMA random seas.

The families use different initial-condition formulas, but share the same DNO
target, split rules, acceptance checks, batch format, and dataset-view format.
The active revisions are Stokes 2, Tanaka 3, Benjamin--Feir 4, and JONSWAP/TMA
4. Older implementations remain available in Git history and should not be
mixed into a new dataset.

## Generation flow

Each attempted case has a deterministic ID derived from its family, revision,
split, stream, and attempt index. A run then:

1. samples parameters from one declared parameter group;
2. writes the proposal;
3. constructs the initial state;
4. runs the nonlinear adjustment when the JONSWAP family requires it;
5. evolves a complete trajectory for rollout families;
6. computes the common Craig--Sulem DNO target;
7. applies the required numerical checks;
8. stores a complete accepted case or a zero-row rejection decision.

Errors in the constructor or solver stop the run. They are not silently turned
into rejected data. A rejected case is always tied to an explicit failed check.

All accepted trajectory rows stay together in one split. The generator never
makes row-level train/validation/test splits.

## Main modules

Family definitions:

- `stokes_sampling.py` samples finite-depth Stokes states.
- `tanaka_sampling.py` and `tanaka_initial_conditions.py` build Tanaka cases.
- `benjamin_feir_sampling.py` and `benjamin_feir_jcp09.py` build modulated wave
  groups.
- `jonswap_tma_sampling.py` and `jonswap_tma.py` build random-sea cases.

Execution:

- `stokes_batch_executor.py` evaluates one static Stokes batch.
- `trajectory_batch_executor.py` evaluates one rollout batch.
- `jonswap_horizon_executor.py` groups JONSWAP cases by compatible integration
  length so they can run efficiently together.
- `trajectory_family_adapters.py` connects each rollout family to the shared
  executor.

Shared pipeline:

- `pipeline/case_allocation.py` assigns attempts across parameter groups and
  defines deterministic split IDs.
- `pipeline/case_checks.py` defines the acceptance masks and failure reasons.
- `pipeline/trajectory_checks.py` evaluates complete-trajectory checks.
- `pipeline/trajectory_rollout.py` runs and samples trajectories.
- `pipeline/dno_target.py` computes the stored DNO target.
- `pipeline/batch_format.py` validates proposal and shard arrays.
- `pipeline/batch_storage.py` inspects committed or interrupted batches.
- `pipeline/artifact_io.py` performs atomic JSON and NPZ writes.
- `pipeline/build_dataset_view.py` creates the manifest and trajectory map used
  by training.
- `pipeline/valid_case_generation.py` resumes a run and continues until each
  parameter group reaches its accepted-case target.

## Acceptance

`CaseCheckResult` carries three masks:

- `required`: checks that must pass for this family and revision;
- `evaluated`: checks that actually ran;
- `failed`: evaluated checks that failed.

A case is accepted only when every required check ran and none failed. The
shared checks cover finite values, positive water depth, complete time grids,
stage convergence, DNO residuals, and internal energy drift where applicable.

JONSWAP uses two separate decisions: the nonlinear adjustment must first
produce a valid handoff, then the autonomous production rollout must pass its
own checks. This separation is why the JONSWAP executor has explicit
adjustment handling.

## Stored files

Each batch may contain:

- a proposal NPZ with case IDs, sampled parameters, and execution settings;
- a result JSON with one acceptance decision per attempt;
- an accepted-row NPZ shard, omitted when the whole batch is rejected;
- a failure JSON if execution stopped before a result could be committed.

Writes use a temporary sibling file, `fsync`, and `os.replace`. On restart,
the scanner either recognizes a complete batch, resumes from a stored
proposal, or reports the incomplete state. It does not rely on file digests or
source snapshots.

The dataset view contains:

- a JSON manifest listing shards, counts, grid settings, and numerical target
  settings;
- a trajectory-map NPZ mapping every attempted case to its acceptance decision
  and every stored row to its trajectory, frame, and shard row.

The training loader uses those two files directly.

## Entrypoints

The supported generation entrypoints live in `scripts/`:

- `generate_paper_dataset.py` handles Stokes, Tanaka, and Benjamin--Feir runs;
- `generate_paper_dataset_jonswap.py` handles JONSWAP/TMA runs;
- `build_paper_dataset_view.py` combines completed chunks into one training
  view.

The launch shell scripts are recipes for the completed paper dataset. They
delegate to these Python entrypoints rather than implementing another
generator.

## Development checks

From the repository root:

```bash
uv run ruff check solver/gen_data scripts/generate_paper_dataset.py \
  scripts/generate_paper_dataset_jonswap.py scripts/build_paper_dataset_view.py
uv run pyright solver/gen_data scripts/generate_paper_dataset.py \
  scripts/generate_paper_dataset_jonswap.py scripts/build_paper_dataset_view.py
uv run pytest solver/gen_data/tests solver/gen_data/pipeline/tests
```
