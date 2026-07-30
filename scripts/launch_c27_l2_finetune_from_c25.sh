#!/bin/bash
# Fast directional screen: continue C25 best for one full-v9 epoch after
# replacing only the supervised H1 norm by L2.

set -euo pipefail
cd /home/johnma/dno-fno

SOURCE_CKPT="outputs/c25_capacity125_full_20260716_022603/best_val_ckpt"
RUN_NAME="${RUN_NAME:-c27_l2_finetune_from_c25_$(date +%Y%m%d_%H%M%S)}"
EPOCHS=1
LR=2e-6
BATCH_SIZE=1024
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

if [[ -e "outputs/$RUN_NAME" ]]; then
  echo "refusing to reuse existing run directory: outputs/$RUN_NAME" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python train-jax-10m/1d_dno_fno_jax.py \
  --model cs_dno \
  --norm scale \
  --dataset combined_dataset_v9.npz \
  --modes 64 \
  --width 640 \
  --n_blocks 8 \
  --latent 320 \
  --sobolev_k 0 \
  --cs_n_polys 3 \
  --cs_mult_hidden 160 \
  --cs_use_g1_baseline \
  --cs_g1_k_cut 0 \
  --cs_g1_fft_fp64 \
  --cs_tie_xi_out_mult \
  --cs_phi_bias_free \
  --cs_residual_eta_order 2 \
  --translation_tangent_weight 10 \
  --translation_tangent_window_depths 1 \
  --translation_tangent_energy_floor_relative 1e-3 \
  --mode_balanced_weight 6 \
  --mode_balanced_warmup_steps 1 \
  --mode_balanced_k_max 128 \
  --mode_balanced_active_scale_relative 1e-4 \
  --mode_balanced_denominator_floor_relative 1e-6 \
  --hadamard_weight 1e-2 \
  --hadamard_interval 16 \
  --hadamard_microbatch 8 \
  --hadamard_warmup_steps 1 \
  --hadamard_k_max 128 \
  --hadamard_sobolev_order 1 \
  --hadamard_relative_eps_min 1e-3 \
  --hadamard_relative_eps_max 3e-3 \
  --hadamard_eta_scale_floor 1e-3 \
  --hadamard_denominator_floor 1e-12 \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --lr_warmup_steps 200 \
  --weight_decay 1e-4 \
  --epochs "$EPOCHS" \
  --total_epochs "$EPOCHS" \
  --resume_from "$SOURCE_CKPT" \
  --reset_opt_state \
  --skip_dno_eval \
  --skip_plots \
  --run_name "$RUN_NAME" 2>&1 | tee "/tmp/$RUN_NAME.log"

uv run python - "$RUN_NAME" "$SOURCE_CKPT" "$BATCH_SIZE" "$CUDA_DEVICES" <<'PY'
import json
import math
import sys
from pathlib import Path

run_name, source_ckpt, batch_size, cuda_devices = sys.argv[1:]
run_dir = Path("outputs") / run_name
config = json.loads((run_dir / "config.json").read_text())
expected = {
    "model": "cs_dno",
    "modes": 64,
    "width": 640,
    "n_blocks": 8,
    "latent": 320,
    "sobolev_k": 0,
    "batch_size": int(batch_size),
    "device_count": len(cuda_devices.split(",")),
    "dataset": "combined_dataset_v9.npz",
    "data_fraction": 1.0,
    "lr": 2e-6,
    "lr_warmup_steps": 200,
    "epochs": 1,
    "total_epochs": 1,
    "weight_decay": 1e-4,
    "cs_n_polys": 3,
    "cs_use_first_deriv": True,
    "cs_use_second_deriv": True,
    "cs_use_half_deriv": True,
    "cs_use_hilbert": True,
    "cs_use_g0_eta": False,
    "cs_use_g0_eta_dx": False,
    "cs_mult_hidden": 160,
    "cs_use_g1_baseline": True,
    "cs_g1_k_cut": 0,
    "cs_fft_fp64": False,
    "cs_g1_fft_fp64": True,
    "cs_tie_xi_out_mult": True,
    "cs_phi_bias_free": True,
    "cs_residual_eta_order": 2,
    "cs_depth_scaled_residual": False,
    "cs_block_k_cut": 0,
    "cs_residual_highband_cap": False,
    "cs_output_highband_cap": False,
    "translation_tangent_weight": 10.0,
    "phase_growth_weight": 0.0,
    "mode_balanced_weight": 6.0,
    "mode_balanced_warmup_steps": 1,
    "hadamard_weight": 0.01,
    "hadamard_interval": 16,
    "hadamard_microbatch": 8,
    "hadamard_warmup_steps": 1,
    "hadamard_k_max": 128.0,
    "hadamard_sobolev_order": 1,
    "hadamard_relative_eps_min": 1e-3,
    "hadamard_relative_eps_max": 3e-3,
    "hadamard_eta_scale_floor": 1e-3,
    "hadamard_denominator_floor": 1e-12,
    "pushforward_steps": 0,
    "hamiltonian_weight": 0.0,
    "modal_phase_rate_weight": 0.0,
    "finite_time_phase_weight": 0.0,
    "stage_reg_weight": 0.0,
    "stage_reg_gain_weight": 0.0,
    "jac_reg_lambda": 0.0,
    "psd_hinge_weight": 0.0,
    "input_noise_sigma": 0.0,
    "gxi_highband_limiter": False,
    "gxi_highband_penalty_weight": 0.0,
    "filter_gxi_fraction": 1.0,
    "param_count": 1_342_400,
    "resume_from": str(
        Path("outputs/c25_capacity125_full_20260716_022603/best_val_ckpt").resolve()
    ),
}
mismatches = {
    key: (config.get(key), value)
    for key, value in expected.items()
    if config.get(key) != value
}
if mismatches:
    raise SystemExit(f"C27 fine-tune configuration guard failed: {mismatches}")

records = [json.loads(line) for line in (run_dir / "train_log.jsonl").read_text().splitlines()]
nonfinite = {
    f"epoch_{record.get('epoch', index + 1)}.{key}": value
    for index, record in enumerate(records)
    for key, value in record.items()
    if isinstance(value, (int, float)) and not math.isfinite(value)
}
if not records or nonfinite:
    raise SystemExit(f"C27 fine-tune finiteness guard failed: {nonfinite}")
if len(records) != 1 or records[0].get("epoch") != 1:
    raise SystemExit(f"C27 fine-tune epoch guard failed: {records}")
if records[-1].get("hadamard_active_batches") != records[-1].get(
    "hadamard_expected_batches"
):
    raise SystemExit("C27 fine-tune Hadamard accounting guard failed")
log_text = Path(f"/tmp/{run_name}.log").read_text(errors="replace")
if "resumed params only" not in log_text or "--reset_opt_state" not in log_text:
    raise SystemExit("C27 fine-tune reset-optimizer restore guard failed")
print("C27 one-epoch L2 fine-tune guard passed")
PY
