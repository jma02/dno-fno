# Script entrypoints

There are two supported training entrypoints:

- `uv run python train-jax-10m/1d_dno_fno_jax.py ...` runs the canonical
  local JAX trainer.
- `uv run modal run scripts/modal_train.py::train ...` runs that same trainer
  on Modal and manages the persistent data/output volume.

The files under `train-jax-10m/` are the internal training engine and its
importable helpers. Superseded trainers remain available through Git history
rather than in the working tree.

The directory is intentionally limited to the current paper workflow:

- the locked C27 launcher and its current evaluation and rendering tools;
- the four-family dataset generators and the combined-view builder.

A `launch_*` recipe configures an experiment but ultimately delegates
training to the canonical JAX engine or `modal_train.py`; it is not another
trainer implementation. Superseded exploratory and scheduled launch scripts
remain available through Git history rather than in the working tree.
