#!/usr/bin/env bash
# Regenerate linear + stokes (deep + finite) using:
#   - MATLAB-style uniform-on-a0 sampling with rejection on steepness
#   - Zakharov-rescaled a0 range (matches dno_dataset.npz physics)
#   - Raised a0_min floor to eliminate tiny-amp tail
#   - Depth ranges that bracket MATLAB's fixed h after Zakharov rescaling
#     so dno_dataset (test set) sits inside our training distribution:
#       linear:        h=0.0383 (MATLAB h=1)        -> our [0.02, 1.5]
#       stokes_deep:   h=38.31  (MATLAB h=1000)     -> our [4.0, 50.0]
#       stokes_finite: h=0.0383 (MATLAB h=1)        -> our [0.02, 1.5]
#   - kh_min for stokes_finite kept >= 0.5 to avoid 5th-order Stokes blowup.
#
# Output combined dataset as combined_dataset_v3.npz.
set -euo pipefail
cd /home/johnma/dno-fno

echo "REFUSING historical v3 regeneration: generate_stokes_dataset now implements"
echo "the frozen paper target and paper steepness/Ursell sampler, not this script's"
echo "MATLAB-style raw-label recipe. Existing v3 artifacts are immutable."
echo "Use the paper-corpus generator and manifest instead."
exit 2

export JAX_ENABLE_X64=1        # generators compute and store fp64

# Per-regime a0 mins matching MATLAB after Zakharov rescaling (alpha = 164/(2pi)):
#   linear:  MATLAB a0_min=0.001 -> 3.83e-5  (steepness_min=5e-3 binds the effective floor)
#   stokes:  MATLAB a0_min=0.02  -> 7.66e-4
A0_MIN_LINEAR=1e-4       # below steepness-min binding (5e-3/n0_max=2.5e-4), recovers MATLAB tail
A0_MIN_STOKES=7.66e-4    # MATLAB 0.02 / alpha
A0_MAX=1.1494e-2         # Zakharov of MATLAB 0.30 = 0.3/alpha
STEEP_MIN_LINEAR=5e-3    # MATLAB linear steepness floor
STEEP_MAX=0.15           # MATLAB steepness cap (linear and stokes)
BS=1000

echo "=== linear (depth 0.02-1.5, kh in [0.02, 20]) -> 500K samples ==="
# kh_min must be <= MATLAB n0=1 kh (= 0.0383 after Zakharov) so dno_linear's
# long-wave samples (n0 in {1..7}, ~35% of dno_linear) sit inside our distribution.
uv run python -m solver.gen_data.generate_linear_dataset \
  --output data/linear.npz \
  --target_samples 500000 --batch_size $BS \
  --a0_min $A0_MIN_LINEAR --a0_max $A0_MAX \
  --steepness_min $STEEP_MIN_LINEAR --steepness_max $STEEP_MAX \
  --n0_min 1 --n0_max 20 \
  --depth_min 0.02 --depth_max 1.5 \
  --kh_min 0.02 --kh_max 20.0 \
  --overwrite

echo "=== stokes_deep (depth 4-50, MATLAB ichoi=0 deep-water formula) -> 500K samples ==="
uv run python -m solver.gen_data.generate_stokes_dataset \
  --output data/stokes_deep.npz --regime deep \
  --target_samples 500000 --batch_size $BS \
  --a0_min $A0_MIN_STOKES --a0_max $A0_MAX \
  --steepness_max $STEEP_MAX \
  --n0_min 1 --n0_max 20 \
  --depth_min 4.0 --depth_max 50.0 \
  --overwrite

echo "=== stokes_finite (depth 0.02-1.5, kh in [0.5, 5.0]) -> 500K samples ==="
uv run python -m solver.gen_data.generate_stokes_dataset \
  --output data/stokes_finite.npz --regime shallow \
  --target_samples 500000 --batch_size $BS \
  --a0_min $A0_MIN_STOKES --a0_max $A0_MAX \
  --steepness_max $STEEP_MAX \
  --n0_min 14 --n0_max 26 \
  --depth_min 0.02 --depth_max 1.5 \
  --kh_min 0.5 --kh_max 5.0 \
  --overwrite

echo "=== combine -> combined_dataset_v3.npz ==="
uv run python -m solver.gen_data.combine_datasets \
  --output data/combined_dataset_v3.npz --shuffle-seed 42

echo "DONE"
