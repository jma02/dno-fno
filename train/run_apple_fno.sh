#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")/.."

exec uv run --python 3.9 python train/1d_dno_fno.py \
  --device mps \
  --dataset dno_dataset.npz \
  --sources all \
  --batch_size 128 \
  --num_workers 0 \
  "$@"
