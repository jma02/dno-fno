# Trained spectral DNO checkpoints

These are unchanged copies of the September 26, 2026 training artifacts. All use
the 1024-point spectral model with width 256, two transformer blocks, four heads,
32 correction branches, BF16 attention/MLPs, and the 128-mode projection.

| File | Completed epochs | Full-validation mean relative L2 | Contents |
| --- | --- | --- | --- |
| [epoch3.pt](epoch3.pt) | 3 | 0.0001301254733 | Ordinary weights and training state |
| [epoch4.pt](epoch4.pt) | 4 | 0.0001084525520 | Ordinary weights, training state, and shadow weight EMA |
| [epoch4_ema.pt](epoch4_ema.pt) | 4 | 0.0000812487787 | Averaged weights for inference only |

Validation covers the same 1,474,440 held-out examples. These are one-step
operator errors, not rollout errors. Weight EMA uses decay 0.999 over the final
9,217 updates (20%) of epoch 4. The ordinary checkpoints include optimizer,
gradient-EMA, and sampler recovery state; the EMA export has no optimizer state.

[manifest.json](manifest.json) records SHA-256 hashes, sizes, steps, and original
paths in the personal `zhaleon` Modal workspace's `dno-fno-train-data` volume.
The three files total about 45 MiB.

From the repository root, evaluate a checkpoint with the existing rollout CLI:

```sh
uv run torch-attention/rollout_eval.py \
  --checkpoint torch-attention/checkpoints/epoch4_ema.pt \
  --input /path/to/initial_conditions.npz \
  --out /path/to/new_predictions.npz --device cuda
```

Input format and integrator details are documented in [the model README](../README.md).
