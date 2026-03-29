# dno-fno

Minimal FNO training repo for the 1D Dirichlet--Neumann operator dataset.

## What is here

- `data/dno_dataset.npz`: unified NumPy dataset with `soliton`, `stokes`, and `linear` samples.
- `models/fno/fno1d.py`: the only model architecture used on this branch.
- `train/1d_dno_fno.py`: the main training script.
- `train/run_apple_fno.sh`: Apple Silicon shortcut.
- `train/init_env.sh`: optional backend setup helper.
- `train/util.py`: dataset loading, normalization, runtime setup, and plotting helpers.

## Setup

```bash
uv sync --python 3.9
```

Or initialize the repo for a specific backend:

```bash
source train/init_env.sh auto
```

Supported backends:

- `auto`
- `mps`
- `cuda`
- `cpu`

## Train on Apple Silicon

Default Apple/MPS run:

```bash
zsh train/run_apple_fno.sh
```

Generic run after initializing the backend:

```bash
uv run --python 3.9 python train/1d_dno_fno.py \
  --device auto \
  --dataset dno_dataset.npz \
  --sources all \
  --batch_size 128 \
  --num_workers 0
```

Or choose the backend explicitly:

```bash
uv run --python 3.9 python train/1d_dno_fno.py --device mps
```

Faster local iteration:

```bash
zsh train/run_apple_fno.sh --modes 64 --width 48 --n_blocks 6
```

## Useful flags

- `--device auto|mps|cuda|cpu`
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

## Split

The trainer reserves an `80/10/10` split:

- `80%` train
- `10%` validation
- `10%` held out for later

The training script only uses the train and validation splits. It does not run test evaluation.
