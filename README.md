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

Use `--help` for model and loss options. Run the same C27 recipe on Modal with
`bash scripts/launch_c27_paper_dataset_modal.sh` after uploading the arrays;
see the [script guide](scripts/README.md). Retired trainers and
the CARBS workflow remain available in Git history.

### Narrower branch correction

The smaller FFT branch model uses these options:

```bash
--model cs_dno --norm scale --width 128 --n_blocks 2 --latent 64 \
  --cs_mult_hidden 160 --cs_correction branches --cs_learned_grid 256 --cs_fuse_fft
```

`--width 128` gives 64 shared channels; two groups of 64 give 128 branches.
This keeps the branch architecture on the 256-point learned grid, with the
analytic baseline on the original input grid. Reducing channels and branches
does not impose the compact correction's fixed Fourier rank.

### Experimental compact correction

CS-DNO keeps its branch correction by default. To try the separate compact
matrix correction, add these flags to a training command and use a new run name:

```bash
--model cs_dno --norm scale --cs_correction compact \
  --cs_learned_grid 256 --cs_compact_rank 64 --cs_compact_hidden 128
```

This keeps the full-resolution analytic baseline and learns a symmetric matrix
acting on a fixed Fourier subspace. Rank 64 means 32 sine and 32 cosine modes;
rank 32 is also supported. The correction remains linear in the input potential
and quadratic near a flat surface. It cannot correct modes outside that subspace
and does not enforce translation equivariance. Treat it as an accuracy/performance
experiment, with validation and rollout checks before adoption.

`--cs_compact_hidden` controls the per-example conditioning network. Branch
settings (`--width`, `--n_blocks`, `--latent`, and spatial-feature/multiplier
options) do not affect this correction; omit `--cs_fuse_fft`. The architecture
settings are saved in `config.json` and restored by the rollout loader.
Select `--cs_correction branches` to use the existing architecture.

The [short H100 pilot](experiments/fewer_branches_20260916_1545_compact_h100.json)
measured 10.4x faster steps at batch 4096 for rank 32, but 2.44x higher validation
error after 1024 matched updates at batch 512. It is not an established replacement
for the branch model; reducing channel width is a separate option without this
fixed Fourier-subspace restriction.

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
