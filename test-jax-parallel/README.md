# JAX Parallel Smoke Test

This directory is a minimal multi-GPU JAX sanity test that is intentionally
separate from the main DNO/FNO training code.

It uses:
- `sklearn.datasets.load_digits()` as a small offline classification dataset
- a simple Flax MLP
- `jax.pmap` over all local GPUs

The goal is to answer one narrow question:

- does a basic multi-GPU `pmap` training loop behave normally on this machine?

## Run

The script defaults to `NCCL_P2P_LEVEL=PHB`, which is the stable multi-GPU
setting on this machine.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python test-jax-parallel/mlp_digits_pmap.py
```

Optional smaller run:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python test-jax-parallel/mlp_digits_pmap.py \
  --epochs 3 \
  --batch_size 128 \
  --hidden_dim 128
```

## Expected behavior

The script prints:
- device list
- warmup compile+execute time for `train_step`
- warmup compile+execute time for `eval_step`
- per-epoch train loss / validation accuracy

This is meant to be simple enough that if it stalls, the issue is probably with
the local JAX multi-GPU stack rather than our DNO training code.
