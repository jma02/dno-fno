# Script entrypoints

There are two paper-workflow training entrypoints:

- `uv run python train-jax-10m/1d_dno_fno_jax.py ...` runs the canonical
  local JAX trainer.
- `uv run modal run scripts/modal_train.py::train ...` runs that same trainer
  on Modal and manages the persistent data/output volume.

The files under `train-jax-10m/` are the internal training engine and its
importable helpers. Superseded trainers remain available through Git history
rather than in the working tree.
The separate `train-jax/` trainer remains available for baseline experiments.

The directory is intentionally limited to the current paper workflow:

- the locked C27 launcher and its current evaluation and rendering tools;
- the four-family dataset generators and the array-dataset builder.

A `launch_*` recipe configures an experiment but ultimately delegates
training to the canonical JAX engine or `modal_train.py`; it is not another
trainer implementation. Superseded exploratory and scheduled launch scripts
remain available through Git history rather than in the working tree.

Generation saves batches and a run summary. Combine completed runs into a dataset
directory with `scripts/build_paper_dataset.py`, repeating `--run-summary` for
each family/split run and choosing `--output-root` for the resulting NPY arrays.
Existing completed batches can be reused without rerunning simulations. Export
to a new directory; allow about 100 GB of additional disk for the full dataset:

```sh
uv run python scripts/build_paper_dataset.py --run-summary ... \
  --output-root outputs/paper_dataset_literature_aligned_v1/combined/c16384_v01024_t01024/arrays
```

Local paper training and Modal training take that directory as `--dataset`; no
manifest or trajectory map is needed. Upload its arrays for Modal training with:

```sh
modal run scripts/modal_train.py::upload_dataset --dataset outputs/.../arrays
modal run scripts/modal_train.py::train --dataset /data/outputs/.../arrays
```

Compare two saved evaluation archives directly:

```sh
uv run python scripts/compare_paired_translation_errors.py \
  --baseline outputs/baseline/eval_suite/tanaka_trajs.npz \
  --candidate outputs/candidate/eval_suite/tanaka_trajs.npz \
  --output outputs/paired_translation_errors.json
```
