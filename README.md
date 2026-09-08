# dno-fno

Minimal FNO training repo for the 1D Dirichlet--Neumann operator dataset.
This repo assumes NVIDIA GPU execution for JAX training.

For the current paper dataset and C27 training workflow, see
[`scripts/README.md`](scripts/README.md) and the
[dataset-generation guide](solver/gen_data/README.md). Generate simulations first,
split them once when building the arrays, then train on shuffled rows. The C27
launchers default to `outputs/paper_dataset/arrays`.

## Baseline workflow

- `data/dno_dataset.npz`: unified NumPy dataset with `soliton`, `stokes`, and `linear` samples.
- `models/fno-jax/fno1d.py`: JAX/Flax FNO model.
- `models/fno-jax/losses.py`: JAX loss functions and parameter counting.
- `train-jax/1d_dno_fno_jax.py`: JAX training script.
- `train-jax/carbs_fno_jax.py`: CARBS runner for the JAX trainer.
- `train-jax/util.py`: JAX data loading, normalization, batching, and plotting helpers.

## Setup

```bash
uv sync --python 3.11
```

## GPU Requirement

The JAX trainer requires a GPU-backed JAX install and will refuse to run on CPU.

## Train

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
- `--sobolev_k 1` (derivative orders to penalize)
- `--sobolev_weight 1.0` (0 = pure relative L2, higher = more derivative penalty)
- `--batch_size 128`
- `--epochs 600`
- `--lr 5e-3`
- `--weight_decay 1e-4`
- `--sources all` or `--sources soliton,stokes`

## Outputs

Each run creates `outputs/fno_jax_YYYYMMDD_HHMMSS/` with:

- `config.json`
- `train_log.json`
- `summary.json`
- `loss_curve.png`
- `best_val_ckpt/`
- `final_ckpt/`

`best_val_ckpt/` stores the best validation model seen during training.
`final_ckpt/` stores the model at the final epoch.
`loss_curve.png` is overwritten every epoch.

## CARBS

The CARBS runner treats:

- `best_val_loss` as the objective to minimize
- `runtime_seconds` as the observed cost
- failed training runs as CARBS failures

```bash
uv run --python 3.11 python train-jax/carbs_fno_jax.py --trials 20
```

Each CARBS run writes:

- `carbs_history.json`
- `best_result.json`
- one subdirectory per trial containing the normal trainer outputs

The CARBS runner automatically disables plotting inside trainer runs to keep search overhead low.

## Split

The trainer reserves an `80/10/10` split:

- `80%` train
- `10%` validation
- `10%` held out for later

The training script only uses the train and validation splits. It does not run test evaluation.
