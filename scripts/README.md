# Script entrypoints

There are two supported training entrypoints:

- `uv run python scripts/train_dno.py ...` runs the canonical local JAX
  trainer.
- `uv run modal run scripts/modal_train.py::train ...` runs that same trainer
  on Modal and manages the persistent data/output volume.

The files under `train-jax-10m/` are the internal training engine and its
importable helpers. The older `train/`, `train-jax/`, and `train-dnonet-jax/`
directories are retained for experiment provenance; they are not current
training entrypoints.

The directory is intentionally limited to the current paper workflow:

- C27/C28/C29 launchers and their evaluation, ablation, and rendering tools;
- the frozen paper-corpus generation, completion-audit, and release-test
  closure listed by `reproducibility/paper_corpus_release_files.json`; and
- `modal_bf.py`, which remains covered by the Benjamin--Feir Modal contract
  test.

A `launch_*` recipe configures an experiment but ultimately delegates
training to `train_dno.py` or `modal_train.py`; it is not another trainer
implementation. Superseded exploratory and scheduled launch scripts remain
available through Git history rather than in the working tree.
