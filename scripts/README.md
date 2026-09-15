# Script entrypoints

There are two paper-workflow training entrypoints:

- `uv run python train-jax-10m/1d_dno_fno_jax.py ...` runs the canonical
  local JAX trainer.
- `uv run modal run scripts/modal_train.py::train ...` runs that same trainer
  on Modal and manages the persistent data/output volume.

The files under `train-jax-10m/` are the internal training engine and its
importable helpers. Superseded trainers remain available through Git history
rather than in the working tree.

The directory is intentionally limited to the current paper workflow:

- the locked C27 launcher and its current evaluation and rendering tools;
- the four-family dataset generators and the array-dataset builder.

A `launch_*` recipe configures an experiment but ultimately delegates
training to the canonical JAX engine or `modal_train.py`; it is not another
trainer implementation. Superseded exploratory and scheduled launch scripts
remain available through Git history rather than in the working tree.

Generate the requested number of accepted simulations for each chosen family.
Generation uses `--seed` (default `2026072210`), not a train/validation/test split:

```sh
uv run python scripts/generate_paper_dataset.py --family stokes \
  --num-simulations 100 --batch-size 32 \
  --output-root outputs/paper_dataset/generated
```

The count above is just a small example; choose each family's count explicitly.
Generation saves self-contained, resumable NPZ batches, not summary files.
For JONSWAP, make `--batch-size` large enough to sort a full parameter group by
rollout length, while `--solver-batch-size` limits how many run on the GPU at once.
For 18,432 simulations across the current 27 groups, use 683 and 128 respectively.
`build_paper_dataset.py` contains all assembly and split logic: it recursively
reads `batch_*.npz` under each `--input-root` (repeat to pool directories).
It uses every completed batch present, even before generation reaches its quota.
Whole accepted simulations get a global 80/10/10 train/validation/test split with
seed `42`; change it with `--seed`, `--validation-fraction`, and `--test-fraction`.

```sh
uv run python scripts/build_paper_dataset.py \
  --input-root outputs/paper_dataset/generated \
  --output-root outputs/paper_dataset/arrays
```

Export to a new directory; existing arrays are not overwritten. Existing completed
batches can be exported again without rerunning simulations. Storage depends on
the chosen simulation counts and retained frames: the three float32 fields alone
need `12 × rows × grid_points` bytes.

Local and Modal training take the array directory as `--dataset`. Normalization
is fitted on training rows only, then reused for validation/test. Each epoch
shuffles all retained training rows together, without family reweighting or
one-frame-per-simulation sampling. Simulation counts and retained-frame counts
therefore determine each family's contribution to training.

The C27 launchers default to `outputs/paper_dataset/arrays`:

```sh
PREFLIGHT_ONLY=1 bash scripts/launch_c27_paper_dataset_full.sh
bash scripts/launch_c27_paper_dataset_full.sh
```

The first command checks loading and training-only normalization without training.
The second starts a fresh run. `EPOCHS` (default 40) controls both the final
epoch and the learning-rate schedule. A resumed 40-epoch run still uses 40,
not the number of epochs remaining.

Upload the arrays, then run the same C27 model and losses on two Modal GPUs:

```sh
modal run scripts/modal_train.py::upload_dataset --dataset outputs/paper_dataset/arrays
bash scripts/launch_c27_paper_dataset_modal.sh
```

Compare two saved evaluation archives directly:

```sh
uv run python scripts/compare_paired_translation_errors.py \
  --baseline outputs/baseline/eval_suite/tanaka_trajs.npz \
  --candidate outputs/candidate/eval_suite/tanaka_trajs.npz \
  --output outputs/paired_translation_errors.json
```
