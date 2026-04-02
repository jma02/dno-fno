# Solver Session Notes: 2026-03-31

## Summary

This session implemented and reorganized the JAX water-wave solver stack, added implicit time stepping for the fast soliton cases, generated rollout GIFs for all local soliton files, and refactored `solver/` into a cleaner package layout.

## Current Layout

- `solver/solvers/`
  - `dno_series_jax.py`
  - `time_integrator.py`
- `solver/data/`
  - `solitary_loader_jax.py`
  - `stokes_truth_jax.py`
- `solver/evals/`
  - `compare_rollout_jax.py`
  - `compare_stokes_rollout.py`
  - `render_rollout_movie.py`
  - `render_all_soliton_rollouts.py`

Public imports are still re-exported from `solver/__init__.py`.

## Implemented Solvers

### DNO

- Ported the MATLAB DNO Taylor-series code to JAX.
- Uses periodic FFTs and zero-padded spectral multiplication.
- Main file: `solver/solvers/dno_series_jax.py`

### Time Integrators

Main file: `solver/solvers/time_integrator.py`

Implemented:

- `rk4_if_step`
  - explicit integrating-factor RK4
- `implicit_midpoint_if_step`
  - simple implicit midpoint in integrating-factor variables
- `gauss_legendre_2_if_step`
  - 2-stage fourth-order Gauss-Legendre implicit Runge-Kutta method

Important note:

- Gauss-Legendre is an implicit Runge-Kutta method, so it belongs in the time integrator module.
- The old filename `rk4_integrator_jax.py` was misleading, so it was renamed to `time_integrator.py`.

## Data / Truth Utilities

### Soliton Loader

File: `solver/data/solitary_loader_jax.py`

- Loads local files from `/home/johnma/dno_locl/soliton_data`
- The actual dataset has:
  - 30 waveform files
  - `Nx = 1024`
  - `L = 164`
  - `1001` saved time slices per file

### Stokes Truth

File: `solver/data/stokes_truth_jax.py`

- Added closed-form Stokes trajectory generation based on `dno_locl/stokes.m`
- Supports finite depth and deep-water modes

## Important Numerical Findings

### Periodic BCs

The solver uses periodic boundary conditions spectrally:

- periodic grid with no duplicated endpoint
- derivatives via `ifft(i k fft(u))`
- DNO applied as a Fourier multiplier / series operator
- nonlinear products computed with periodic zero-padding dealiasing

The soliton data themselves are only approximately periodic on the finite computational box, which is a separate issue from the implementation.

### Xi Drift

Observed on mild soliton rollouts:

- raw `xi` looked poor
- `xi - mean(xi)` looked very good
- `xi_x` looked very good
- `G(eta)xi` looked very good

Interpretation:

- this is mostly gauge drift in the spatial mean of `xi`
- the physically relevant part is still captured well
- `G(eta)xi` is insensitive to additive spatial constants in `xi`

Practical recommendation:

- if using `xi` directly, project out the zero Fourier mode / subtract the spatial mean
- otherwise use `xi_x`, `xi - mean(xi)`, or `G(eta)xi`

### Fast Soliton Family

The `f*` trajectories are much coarser in time than the other families:

- `f*`: `dt = 1.0`
- `s*`: `dt = 0.2`
- `h005005`: `dt = 0.08`

Observed behavior:

- explicit IF-RK4 blew up on `f*` cases even with aggressive substepping
- implicit midpoint was worse
- Gauss-Legendre implicit RK was better but still unstable by itself
- adding a spectral cutoff stabilized the fast cases

Working fast-case recipe:

- method: `gl2_if`
- internal substeps: `8`
- filter fraction: `2/3`
- DNO order: `6`

This was stable through the full `coll.anim_f04005` horizon, though long-time accuracy is still not great.

## JCP09 Takeaway

The 2009 JCP paper is not using plain explicit RK4 on the raw system.

Relevant points:

- DNO evaluated spectrally
- implicit symplectic fourth-order Runge-Kutta time stepping
- de-aliasing / filtering are part of the numerical method

That matches the practical behavior seen here:

- mild cases can tolerate explicit IF-RK4
- fast cases want an implicit RK method and filtering

## Defaults After This Session

### Soliton Batch Renderer

File: `solver/evals/render_all_soliton_rollouts.py`

Current defaults:

- `dno_order = 6`
- `default_method = "rk4_if"`
- `fast_method = "gl2_if"`
- `default_substeps = 1`
- `fast_substeps = 8`
- `filter_fraction` is provided by CLI when needed

### Movie Style

File: `solver/evals/render_rollout_movie.py`

Current movie style:

- top row: `eta`
- middle row: `xi`
- bottom row: `G(eta)xi`
- true = opaque blue
- ours = red drawn over the blue curve
- no fading water-fill effect in the `eta` panel

## Generated Outputs

### Soliton GIF Batch

Output directory:

- `soliton_rollouts/`

Rerun completed for all 30 files using:

- `dno_order=6`
- `gl2_if` for `f*`
- `rk4_if` for `h*` and `s*`
- `fast_substeps=8`
- `filter_fraction=2/3`

Total batch runtime:

- about `2210.87 s`
- about `36.8 min`

### Stokes Comparison Artifacts

Saved under:

- `outputs/solver_rollouts/`

Includes:

- Stokes rollout plots
- comparison NPZs
- summary JSONs
- example GIFs

## Useful Commands

### Run All Soliton GIFs

```bash
MPLCONFIGDIR=/tmp/matplotlib CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/johnma/dno-fno/.venv/bin/python -m solver.evals.render_all_soliton_rollouts \
  --output_dir soliton_rollouts \
  --dno_order 6 \
  --filter_fraction 0.6666666667 \
  --default_method rk4_if \
  --fast_method gl2_if \
  --default_substeps 1 \
  --fast_substeps 8 \
  --implicit_iterations 4 \
  --implicit_relaxation 1.0 \
  --fps 30 \
  --n_frames 200 \
  --dpi 120
```

### Render One Saved Comparison GIF

```bash
MPLCONFIGDIR=/tmp/matplotlib CUDA_VISIBLE_DEVICES=1 \
/home/johnma/dno-fno/.venv/bin/python -m solver.evals.render_rollout_movie \
  --comparison_npz outputs/solver_rollouts/rollout_compare_coll_anim_h005005_order2.npz \
  --output outputs/solver_rollouts/rollout_compare_coll_anim_h005005_order2.gif
```

### Compare Stokes Rollout

```bash
/home/johnma/dno-fno/.venv/bin/python -m solver.evals.compare_stokes_rollout \
  --ichoi 1 \
  --n0 14 \
  --a0 0.1 \
  --dno_order 6
```

## Verification Performed

- `python -m compileall solver`
- import smoke test under the project venv
- GPU rollout tests on `coll.anim_f04005`
- full GIF batch rerun completed successfully

## Open Issues / Next Steps

- Long-time `f*` accuracy is still rough even when stabilized.
- The `xi` gauge should likely be fixed explicitly by removing the spatial mean during rollout or evaluation.
- If a closer match to the original solitary-wave generator is needed, the next place to investigate is the exact filtering / stabilization used in the original production code.
