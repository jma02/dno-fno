# dno-fno

JAX FNO and CS-DNO surrogates for the 1D Dirichlet--Neumann operator.
Training requires an NVIDIA GPU.

For the current paper dataset and C27 training workflow, see
[`scripts/README.md`](scripts/README.md) and the
[dataset-generation guide](solver/gen_data/README.md). Generate NPZ batches, then
use `scripts/build_paper_dataset.py` to split whole simulations and assemble the
training arrays. C27 launchers default to `outputs/paper_dataset/arrays`.

## Setup

```bash
uv sync --python 3.11
```

## Train

Run the current C27 recipe against the assembled array dataset:

```bash
bash scripts/launch_c27_paper_dataset_full.sh
```

All local launchers use `train-jax-10m/1d_dno_fno_jax.py`. For a custom run:

```bash
uv run python train-jax-10m/1d_dno_fno_jax.py \
  --dataset outputs/paper_dataset/arrays \
  --model fno --norm scale --batch_size 256
```

Use `--help` for model and loss options. `scripts/modal_train.py` runs the same
trainer on Modal; see the [script guide](scripts/README.md). Retired trainers and
the CARBS workflow remain available in Git history.

## Outputs

Each run creates `outputs/<run_name>/` with:

- `config.json`
- `train_log.jsonl`
- `summary.json`
- `latest_ckpt/`
- `best_val_ckpt/`
- `final_ckpt/`

`best_val_ckpt/` stores the best validation model seen during training.
`final_ckpt/` stores the final model; `latest_ckpt/` supports automatic resumption
when the same run directory is reused.

## Split

The dataset builder assigns whole simulations to train/validation/test splits
(80/10/10 by default). The trainer uses those saved splits and fits normalization
on training rows only. Test evaluation is separate, through `solver/evals/eval_suite.py`.
