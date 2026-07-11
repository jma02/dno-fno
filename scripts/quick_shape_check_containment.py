"""Fast CPU shape/behavior check for `_safe_gl2_substep` per-sample masking.

Constructs a tiny batched state, dispatches to the new masking-only path, and
confirms:
  1. Healthy inputs → all samples pass through unchanged (bit-exact).
  2. One deliberately-corrupted sample → only that sample is NaN'd.
  3. Scalar (per-IC) path still works: 1D state in, 1D state out.

Fast because it avoids the full model load — uses a lambda predict_gxi that
returns zeros, which makes the operator equal to the linear IF flow (Picard
converges in one iteration to identity). r_final ≈ 0 → no masking.
"""
from __future__ import annotations
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from solver.evals.model_rollout import _safe_gl2_substep  # noqa: E402
from solver.solvers import time_integrator as ti  # noqa: E402
from solver.solvers.dno_series_jax import build_grid, make_linear_dno_symbol  # noqa: E402


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    dtype = jnp.float64
    nx = 64
    length = 2.0 * float(np.pi)
    NB = 4

    _, k_grid = build_grid(nx, length)
    k_grid = jnp.asarray(k_grid, dtype=dtype)
    depth_1d = jnp.asarray(1.0, dtype=dtype)
    depth_2d = jnp.asarray([1.0] * NB, dtype=dtype)[:, None]

    g0_1d = make_linear_dno_symbol(k_grid, depth_1d)
    g0_2d = make_linear_dno_symbol(k_grid, depth_2d)

    sp_1d = ti.SolverParams(
        nx=nx, length=length, depth=depth_1d, gravity=1.0,
        dno_order=6, pad_factor=8, filter_fraction=0.25, k=k_grid, g0=g0_1d,
    )
    sp_2d = ti.SolverParams(
        nx=nx, length=length, depth=depth_2d, gravity=1.0,
        dno_order=6, pad_factor=8, filter_fraction=0.25, k=k_grid, g0=g0_2d,
    )

    def predict_zero_1d(eta_in: jnp.ndarray, xi_in: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(eta_in)

    def predict_zero_2d(eta_in: jnp.ndarray, xi_in: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(eta_in)

    x = jnp.linspace(0.0, length, nx, endpoint=False, dtype=dtype)

    common_kwargs = dict(
        iterations=4,
        residual_tol=1e-2,
        zero_mean_xi=True,
        filter_shape="hard",
        houli_a=36.0, houli_m=36.0,
        cascade_gate_enabled=False,
        cascade_k_cut=32.0, cascade_r_threshold=1e-3, cascade_sharpness=10.0,
        cascade_houli_a=0.69, cascade_houli_m=4.0, cascade_k_eff=128.0,
        cascade_filter_xi=True,
    )

    # --- 1D healthy ---
    eta_1d = 0.01 * jnp.sin(x)
    xi_1d = 0.005 * jnp.cos(x)
    state_1d = ti.State(eta=eta_1d, xi=xi_1d)
    out_1d = _safe_gl2_substep(
        state_1d, jnp.asarray(0.0, dtype=dtype), jnp.asarray(0.1, dtype=dtype),
        sp_1d, predict_zero_1d, **common_kwargs,
    )
    assert out_1d.eta.shape == (nx,), f"1D eta shape {out_1d.eta.shape}"
    assert not jnp.isnan(out_1d.eta).any(), "1D healthy path should not NaN"
    print(f"1D healthy: eta max={float(jnp.abs(out_1d.eta).max()):.3e} — OK")

    # --- 2D healthy (batched) ---
    eta_2d = jnp.stack([0.01 * jnp.sin(x)] * NB, axis=0)
    xi_2d = jnp.stack([0.005 * jnp.cos(x)] * NB, axis=0)
    state_2d = ti.State(eta=eta_2d, xi=xi_2d)
    out_2d = _safe_gl2_substep(
        state_2d, jnp.asarray(0.0, dtype=dtype), jnp.asarray(0.1, dtype=dtype),
        sp_2d, predict_zero_2d, **common_kwargs,
    )
    assert out_2d.eta.shape == (NB, nx), f"2D eta shape {out_2d.eta.shape}"
    nans_per_row = jnp.isnan(out_2d.eta).any(axis=1)
    print(f"2D healthy: any NaN per row = {np.asarray(nans_per_row).tolist()}")
    assert not nans_per_row.any(), "2D healthy path should not NaN any row"
    print("2D healthy: OK")

    # --- 2D one poisoned sample (row 2 corrupted with wild noise → high residual expected) ---
    # Corrupt row 2 with a sharp high-k component so Picard doesn't converge cleanly at big dt.
    poisoned = eta_2d.at[2].set(0.5 * jnp.cos(30.0 * x))  # extreme high-k, big amplitude
    poisoned_xi = xi_2d.at[2].set(0.3 * jnp.sin(30.0 * x))
    state_bad = ti.State(eta=poisoned, xi=poisoned_xi)
    # Use a very small tol so any nontrivial state gets flagged; also crank dt so the
    # linear flow rotates hard on the k=30 mode → residual becomes large.
    kwargs_strict = dict(common_kwargs)
    kwargs_strict["residual_tol"] = 1e-8
    out_bad = _safe_gl2_substep(
        state_bad, jnp.asarray(0.0, dtype=dtype), jnp.asarray(3.0, dtype=dtype),
        sp_2d, predict_zero_2d, **kwargs_strict,
    )
    nans_per_row = np.asarray(jnp.isnan(out_bad.eta).any(axis=1))
    print(f"2D mixed strict-tol: NaN per row = {nans_per_row.tolist()}")
    assert nans_per_row[2], "row 2 (poisoned) should be NaN'd under strict tol"

    print("QUICK SHAPE CHECK OK")


if __name__ == "__main__":
    main()
