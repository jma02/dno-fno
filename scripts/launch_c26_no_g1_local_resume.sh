#!/usr/bin/env bash
# Resume the matched C25 G1-deletion ablation from its downloaded Modal run.

set -euo pipefail
cd /home/johnma/dno-fno

RUN_NAME="${RUN_NAME:-c26_no_g1_oeta2_full_modal_l40sx4_20260716_211238}"
RUN_DIR="outputs/${RUN_NAME}"

if [[ ! -f "${RUN_DIR}/latest_ckpt/metadata.json" ]]; then
  echo "missing committed checkpoint metadata: ${RUN_DIR}/latest_ckpt/metadata.json" >&2
  exit 1
fi

RESUME_EPOCH="$(jq -r '.epoch' "${RUN_DIR}/latest_ckpt/metadata.json")"
if [[ ! -d "${RUN_DIR}/latest_ckpt/ckpt_${RESUME_EPOCH}" ]]; then
  echo "missing checkpoint payload: ${RUN_DIR}/latest_ckpt/ckpt_${RESUME_EPOCH}" >&2
  exit 1
fi

echo "resuming ${RUN_NAME} locally after committed epoch ${RESUME_EPOCH}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python train-jax-10m/1d_dno_fno_jax.py \
  --model cs_dno \
  --norm scale \
  --dataset combined_dataset_v9.npz \
  --data_fraction 1.0 \
  --modes 64 \
  --width 640 \
  --n_blocks 8 \
  --latent 320 \
  --sobolev_k 1 \
  --cs_n_polys 3 \
  --cs_mult_hidden 160 \
  --cs_g1_k_cut 0 \
  --cs_g1_fft_fp64 \
  --cs_tie_xi_out_mult \
  --cs_phi_bias_free \
  --cs_residual_eta_order 2 \
  --translation_tangent_weight 10 \
  --translation_tangent_window_depths 1 \
  --translation_tangent_energy_floor_relative 1e-3 \
  --mode_balanced_weight 6 \
  --mode_balanced_warmup_steps 500 \
  --mode_balanced_k_max 128 \
  --mode_balanced_active_scale_relative 1e-4 \
  --mode_balanced_denominator_floor_relative 1e-6 \
  --hadamard_weight 1e-2 \
  --hadamard_interval 16 \
  --hadamard_microbatch 8 \
  --hadamard_warmup_steps 500 \
  --hadamard_k_max 128 \
  --hadamard_sobolev_order 1 \
  --hadamard_relative_eps_min 1e-3 \
  --hadamard_relative_eps_max 3e-3 \
  --hadamard_eta_scale_floor 1e-3 \
  --hadamard_denominator_floor 1e-12 \
  --batch_size 1024 \
  --lr 2e-5 \
  --lr_warmup_steps 500 \
  --weight_decay 1e-4 \
  --epochs 40 \
  --total_epochs 40 \
  --skip_dno_eval \
  --skip_plots \
  --run_name "${RUN_NAME}" 2>&1 | tee -a "/tmp/${RUN_NAME}_local_resume.log"

uv run python - "${RUN_DIR}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
metadata = json.loads(
    (run_dir / "final_ckpt" / "metadata.json").read_text(encoding="utf-8")
)
expected = {
    "cs_use_g1_baseline": False,
    "cs_residual_eta_order": 2,
    "param_count": 1_342_400,
    "epochs": 40,
    "total_epochs": 40,
}
mismatches = {
    key: (config.get(key), value)
    for key, value in expected.items()
    if config.get(key) != value
}
if mismatches:
    raise SystemExit(f"C26 config guard failed: {mismatches}")
if int(metadata["epoch"]) != 40:
    raise SystemExit(f"C26 final checkpoint is epoch {metadata['epoch']}, expected 40")
print("C26 local completion guard passed: no G1, O(eta^2), epoch 40")
PY
