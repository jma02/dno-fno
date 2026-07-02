#!/usr/bin/env bash
# Sequential FNO -> SpectralDNO training pair with --input_noise_sigma 0.02.
# Same hyperparameters as fno_w128b6_v3_hclip5_20260514_063811 except
# epochs halved to 40 and noise reg added.
set -e
cd /home/johnma/dno-fno

TS="$1"
if [ -z "$TS" ]; then
  TS=$(date +%Y%m%d_%H%M%S)
fi

mkdir -p logs

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65

uv run python train-jax-10m/1d_dno_fno_jax.py \
  --model fno \
  --norm scale \
  --dataset combined_dataset_v3.npz \
  --modes 64 --width 128 --n_blocks 6 \
  --sobolev_k 1 --batch_size 512 \
  --epochs 40 --lr 0.0002 --weight_decay 0.0001 \
  --input_noise_sigma 0.02 \
  --run_name "fno_w128b6_v3_noise02_${TS}" \
  > "logs/fno_w128b6_v3_noise02_${TS}.log" 2>&1

uv run python train-jax-10m/1d_dno_fno_jax.py \
  --model spectral_dno \
  --norm scale \
  --dataset combined_dataset_v3.npz \
  --modes 64 --width 128 --n_blocks 6 --latent 64 \
  --sobolev_k 1 --batch_size 512 \
  --epochs 40 --lr 0.0002 --weight_decay 0.0001 \
  --input_noise_sigma 0.02 \
  --run_name "dno_w128b6_l64_v3_noise02_${TS}" \
  > "logs/dno_w128b6_l64_v3_noise02_${TS}.log" 2>&1
