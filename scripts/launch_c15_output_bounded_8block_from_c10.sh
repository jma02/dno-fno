#!/bin/bash
# Full-capacity checkpoint-carrying output high-band cap fine-tune.
# Warm-starts from C10 and keeps the 8-block parameterization; unlike C14 this
# isolates the bounded-output mechanism without shrinking the model.

set -euo pipefail
cd /home/johnma/dno-fno

SOURCE_RUN="${SOURCE_RUN:-outputs/c10_stage_gain_k64_prodff_from_c2_20260709_055358}"
SOURCE_CKPT="${SOURCE_CKPT:-$SOURCE_RUN/best_val_ckpt}"
DATASET="${DATASET:-combined_dataset_v9.npz}"
RUN_NAME="${RUN_NAME:-c15_output_bounded_8block_from_c10_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LR="${LR:-2e-6}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python train-jax-10m/1d_dno_fno_jax.py \
  --model cs_dno \
  --norm scale \
  --dataset "$DATASET" \
  --data_fraction 1.0 \
  --modes 64 \
  --width 512 \
  --n_blocks 8 \
  --latent 256 \
  --sobolev_k 1 \
  --cs_n_polys 3 \
  --cs_mult_hidden 128 \
  --cs_tie_xi_out_mult \
  --cs_output_highband_cap \
  --cs_output_highband_cap_k_cut 32.0 \
  --cs_output_highband_cap_r_max 1e-2 \
  --cs_output_highband_cap_abs_floor 5.0 \
  --resume_from "$SOURCE_CKPT" \
  --reset_opt_state \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight_decay 1e-4 \
  --epochs "$EPOCHS" \
  --skip_dno_eval \
  --skip_plots \
  --run_name "$RUN_NAME" 2>&1 | tee "/tmp/${RUN_NAME}.log"
