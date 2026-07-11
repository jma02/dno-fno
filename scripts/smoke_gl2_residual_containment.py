"""Force the GL2 residual guard to fire and verify per-sample NaN emission.

Constructs a pathological state that drives the CS-DNO Picard map non-contractive
within a single substep (large-amplitude high-k mode → the ``(1+eta_x²)`` term
blows up), and asserts:

  1. The guarded substep emits NaN on the pathological state
     rather than a silently-diverging state.
  2. Healthy states pass through without NaNs.

Uses the analytic ``dno_series_eval`` as the predictor so nothing about the
model is required. CPU-only, ~few seconds.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solver.evals.model_rollout import _safe_gl2_substep  # noqa: E402
from solver.solvers import time_integrator as ti  # noqa: E402
from solver.solvers.dno_series_jax import (  # noqa: E402
    build_grid,
    dno_series_eval,
    make_linear_dno_symbol,
)


def build_params(nx: int, length: float, depth: float, filt: float) -> ti.SolverParams:
    _, k = build_grid(nx, length)
    k = jnp.asarray(k, dtype=jnp.float64)
    g0 = make_linear_dno_symbol(k, depth)
    return ti.SolverParams(
        nx=nx, length=length, depth=depth, gravity=1.0,
        dno_order=6, pad_factor=8, filter_fraction=filt, k=k, g0=g0,
    )


def main() -> None:
    jax.config.update("jax_enable_x64", True)

    nx = 256
    length = 2.0 * float(np.pi)
    depth = 0.1
    params = build_params(nx, length, depth, filt=0.25)

    # Pathological state: heavy mid-k xi with a large-amplitude eta so
    # (1+eta_x²) reaches O(30). Order-6 series will amplify this into a
    # non-contractive stage system when dt is large.
    xg = jnp.linspace(0.0, length, nx, endpoint=False, dtype=jnp.float64)
    eta = jnp.asarray(0.4 * jnp.cos(6 * xg) + 0.05 * jnp.cos(30 * xg))
    xi = jnp.asarray(0.3 * jnp.cos(6 * xg) - 0.02 * jnp.sin(30 * xg))
    state = ti.State(eta=eta, xi=xi)

    def predict_gxi(eta_in: jnp.ndarray, xi_in: jnp.ndarray) -> jnp.ndarray:
        return dno_series_eval(eta_in, xi_in, params.k, params.depth, 6, pad_factor=8)

    kwargs = dict(
        params=params,
        predict_gxi=predict_gxi,
        iterations=4,
        residual_tol=1e-6,
        zero_mean_xi=True,
        filter_shape="hard", houli_a=36.0, houli_m=36.0,
        cascade_gate_enabled=False,
        cascade_k_cut=32.0, cascade_r_threshold=1e-3, cascade_sharpness=10.0,
        cascade_houli_a=0.69, cascade_houli_m=4.0, cascade_k_eff=128.0,
        cascade_filter_xi=True,
    )

    # dt large enough that GL2 Picard cannot contract on this pathological state.
    dt = 3.0

    print(f"Pathological state: max|eta|={float(jnp.max(jnp.abs(eta))):.3f}, "
          f"max|xi|={float(jnp.max(jnp.abs(xi))):.3f}, dt={dt}")

    print("Test 1: pathological state — expect NaN emission.")
    out0 = _safe_gl2_substep(state, jnp.asarray(0.0, dtype=jnp.float64),
                             jnp.asarray(dt, dtype=jnp.float64),
                             **kwargs)
    eta0 = np.asarray(out0.eta)
    xi0 = np.asarray(out0.xi)
    print(f"  max|eta_out|={np.nanmax(np.abs(eta0)):.3e}  any_nan={np.any(np.isnan(eta0))}")
    assert np.any(np.isnan(eta0)), "expected NaN output when halvings=0 and Picard fails"

    print("Test 2: repeated pathological state — never diverged finite garbage.")
    out3 = _safe_gl2_substep(state, jnp.asarray(0.0, dtype=jnp.float64),
                             jnp.asarray(dt, dtype=jnp.float64),
                             **kwargs)
    eta3 = np.asarray(out3.eta)
    xi3 = np.asarray(out3.xi)
    any_nan3 = bool(np.any(np.isnan(eta3)))
    max_eta3 = float(np.nanmax(np.abs(eta3))) if not any_nan3 else float("nan")
    print(f"  max|eta_out|={max_eta3:.3e}  any_nan={any_nan3}")
    if not any_nan3:
        # If contained (finite), the result must be a reasonable state — not a
        # blow-up to 1e6+. Loose bound: within 10× of the max IC amplitude.
        max_ic = float(np.max(np.abs(np.asarray(eta))))
        assert max_eta3 < 10 * max_ic, (
            f"finite output max={max_eta3:.3e} vastly exceeds IC scale {max_ic:.3e} — "
            f"containment let a diverged state through"
        )
        print(f"  contained within 10× IC scale ({max_ic:.3f}); guarded state accepted.")
    else:
        print("  guarded step emitted NaN, as intended.")

    print("Test 3: healthy state at moderate dt — should complete cleanly with no halving.")
    eta_h = jnp.asarray(0.01 * jnp.cos(2 * xg))
    xi_h = jnp.asarray(0.005 * jnp.sin(2 * xg))
    state_h = ti.State(eta=eta_h, xi=xi_h)
    dt_h = 0.05
    out_h = _safe_gl2_substep(state_h, jnp.asarray(0.0, dtype=jnp.float64),
                              jnp.asarray(dt_h, dtype=jnp.float64),
                              **kwargs)
    eta_h_out = np.asarray(out_h.eta)
    assert not np.any(np.isnan(eta_h_out)), "healthy state NaN'd — containment misfired"
    print(f"  max|eta_out|={np.max(np.abs(eta_h_out)):.3e}  healthy accept confirmed.")

    print("CONTAINMENT OK")


if __name__ == "__main__":
    main()
