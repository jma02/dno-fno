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
- the four-family dataset generators and the combined-view builder.

A `launch_*` recipe configures an experiment but ultimately delegates
training to the canonical JAX engine or `modal_train.py`; it is not another
trainer implementation. Superseded exploratory and scheduled launch scripts
remain available through Git history rather than in the working tree.

Modal training takes an explicit `--dataset` manifest path. Upload that view and
its referenced files with `modal run scripts/modal_train.py::upload_dataset_view
--dataset outputs/.../paper.dataset.json`; flat-NPZ uploads are not supported.

Compare two saved evaluation archives directly:

```sh
uv run python scripts/compare_paired_translation_errors.py \
  --baseline outputs/baseline/eval_suite/tanaka_trajs.npz \
  --candidate outputs/candidate/eval_suite/tanaka_trajs.npz \
  --output outputs/paired_translation_errors.json
```
