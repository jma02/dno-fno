# CSDNO wavenumber conditioning (quick reference, 2026-08-17)

This note summarizes how CSDNO conditions on Fourier mode number `k` and where each
knob lives.

## 1) Core conditioning in the architecture

Conditioning is built into `models/dno-net/dno_net_v2.py`.

### 1.1 Depth-aware multiplier features

Each learned Fourier multiplier in a Craig–Sulem block is an MLP of physical-mode
inputs:

- `k` (mode magnitude)
- `h` (depth, as log depth, clipped by `h_clip_max`)
- `tanh(hk)`
- `k*tanh(hk)`

That is implemented in `DepthAwareMultiplier.__call__`:

- `k` is computed from the rFFT grid.
- multipliers are produced per mode, per branch, per batch.
- this is why CSDNO learns a depth-dependent symbol rather than a depth-agnostic one.

### 1.2 Multiplier and block hard cuts

Two explicit spectral cutoffs are in the forward pass:

- `g1_k_cut` (default `128`) masks the *analytic* `G1` output modes by applying
  a hard mask on G1 multipliers/transforms.
  - Defined in `CraigSulemDNO` and exposed by CLI `--cs_g1_k_cut`.
- `cs_block_k_cut` (default `0`, disabled) masks each learned block’s
  `m_xi` input multiplier and block output `out_hat` at `k < block_k_cut`.
  - Controlled by `--cs_block_k_cut`.
  - This enforces analytically bandlimited learned corrections.

### 1.3 Learned high-band caps

These are structural output guards (not training losses):

- `cs_residual_highband_cap`:
  only the residual is scaled down above `cs_residual_highband_cap_k_cut`
  if `||P_hi R||` exceeds `beta * ||P_lo R||` (with floor).
- `cs_output_highband_cap`:
  same idea on full output `gxi` above `cs_output_highband_cap_k_cut`.

Both are run-time, within the model call.

## 2) Wavenumber masks used in loss terms

Several losses are explicitly computed only on selected bands:

- `mode_balanced`:
  `0 < |k| <= mode_balanced_k_max` and `weight ∝` soft activity from reference
  physics scale.
- `modal_phase_rate`:
  `k_min <= |k| <= k_max` with default `1` to `128`.
- `finite_time_phase`:
  `0 < |k| <= k_max`.
- `hadamard`:
  probe and defect projection use `hadamard_k_max`.

All are configurable in `train-jax-10m/1d_dno_fno_jax.py` flags and carried into
corresponding configs.

## 3) Data/model filtering in the pipeline

Outside the model itself, training/eval filtering can also enforce `k` conditioning:

- `filter_gxi_fraction` (train/eval) applies a spectral filter to `gxi` predictions.
- `filter_shape` chooses hard cutoff vs Hou–Li spectral smoothing.
- Separate low/high `k` penalty and cap flags for `gxi` exist (`gxi_highband_*`).

These are separate from architectural conditioning and should be considered together.

## 4) Where to inspect

Primary docs:

- [paper/sections/04_csdno_architecture.tex](/home/johnma/dno-fno/paper/sections/04_csdno_architecture.tex)
- [notes/cs_dno_architecture/cs_dno_architecture.tex](/home/johnma/dno-fno/notes/cs_dno_architecture/cs_dno_architecture.tex)
- [notes/v11_light_arch_design.md](/home/johnma/dno-fno/notes/v11_light_arch_design.md)
- [notes/cs_dno_jcp09_improvement_plan.md](/home/johnma/dno-fno/notes/cs_dno_jcp09_improvement_plan.md)

Primary code:

- [models/dno-net/dno_net_v2.py](/home/johnma/dno-fno/models/dno-net/dno_net_v2.py)
- [train-jax-10m/1d_dno_fno_jax.py](/home/johnma/dno-fno/train-jax-10m/1d_dno_fno_jax.py)
- Loss implementations under `train-jax-10m/*_regularizer.py`.
