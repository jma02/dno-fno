# dno-fno

Minimal FNO training repo for the 1D Dirichlet--Neumann operator dataset.
This repo now assumes NVIDIA GPU execution for both the PyTorch and JAX paths.

## What is here

- `data/dno_dataset.npz`: unified NumPy dataset with `soliton`, `stokes`, and `linear` samples.
- `models/fno/fno1d.py`: PyTorch FNO model.
- `models/fno-jax/fno1d.py`: JAX/Flax FNO model.
- `models/fno-jax/losses.py`: JAX loss functions and parameter counting.
- `train/1d_dno_fno.py`: PyTorch training script.
- `train/carbs_fno.py`: CARBS runner for the PyTorch trainer.
- `train-jax/1d_dno_fno_jax.py`: JAX training script.
- `train-jax/carbs_fno_jax.py`: CARBS runner for the JAX trainer.
- `train/util.py`: PyTorch data loading, normalization, and plotting helpers.
- `train-jax/util.py`: JAX data loading, normalization, batching, and plotting helpers.

## Setup

```bash
uv sync --python 3.9
```

For the JAX path, use Python 3.11 or newer:

```bash
uv sync --python 3.11
```

## GPU Requirement

Both training scripts require GPU execution:

- PyTorch trainer: requires CUDA
- JAX trainer: requires a GPU-backed JAX install and will refuse to run on CPU

There is no Apple/MPS path in this repo anymore.

## Train with PyTorch

```bash
uv run --python 3.9 python train/1d_dno_fno.py \
  --dataset dno_dataset.npz \
  --sources all \
  --batch_size 128
```

## Train with JAX

```bash
uv run --python 3.11 python train-jax/1d_dno_fno_jax.py \
  --dataset dno_dataset.npz \
  --sources all \
  --batch_size 128
```

## Useful flags

- `--modes 128`
- `--width 64`
- `--n_blocks 10`
- `--loss sobolev|lp`
- `--batch_size 128`
- `--epochs 600`
- `--lr 5e-3`
- `--weight_decay 1e-4`
- `--sources all` or `--sources soliton,stokes`

## Outputs

Each run creates `outputs/fno_YYYYMMDD_HHMMSS/` with:

- `config.json`
- `train_log.json`
- `summary.json`
- `loss_curve.png`
- `best_val_ckpt.pt`
- `final_ckpt.pt`
- `plots/epoch_*.png`
- `plots/final.png`

`best_val_ckpt.pt` stores the best validation model seen during training.
`final_ckpt.pt` stores the model at the final epoch.
`loss_curve.png` is overwritten every epoch.

The JAX trainer writes the same structure under `outputs/fno_jax_YYYYMMDD_HHMMSS/`,
but uses `best_val_ckpt.pkl` and `final_ckpt.pkl`.

## CARBS

Both training paths can now be driven by CARBS.

The CARBS runners treat:

- `best_val_loss` as the objective to minimize
- `runtime_seconds` as the observed cost
- failed training runs as CARBS failures

PyTorch CARBS search:

```bash
uv run --python 3.11 python train/carbs_fno.py --trials 20
```

JAX CARBS search:

```bash
uv run --python 3.11 python train-jax/carbs_fno_jax.py --trials 20
```

Each CARBS runner writes:

- `carbs_history.json`
- `best_result.json`
- one subdirectory per trial containing the normal trainer outputs

The CARBS runners automatically disable plotting inside the trainer runs to keep search overhead low.

## Split

The trainer reserves an `80/10/10` split:

- `80%` train
- `10%` validation
- `10%` held out for later

The training script only uses the train and validation splits. It does not run test evaluation.
