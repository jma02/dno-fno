"""CPU regression test for the saved rollout DNO field.

Run directly; pytest is not required:

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
      .venv/bin/python solver/solvers/tests_rollout_saved_gxi.py
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from solver.solvers.dno_series_jax import dno_series_eval  # noqa: E402
from solver.solvers.time_integrator import (  # noqa: E402
    State,
    apply_lowpass,
    batched_rollout,
    make_solver_params,
)

jax.config.update("jax_enable_x64", True)


def test_saved_gxi_respects_rollout_lowpass() -> None:
    nx = 64
    length = 2.0 * np.pi
    x = length * jnp.arange(nx) / nx
    params = make_solver_params(
        nx=nx,
        length=length,
        depth=1.0,
        dno_order=0,
        pad_factor=1,
        filter_fraction=0.25,
    )
    initial_state = State(
        eta=jnp.zeros((2, nx)),
        xi=jnp.stack(
            (
                jnp.cos(3.0 * x) + 0.2 * jnp.cos(20.0 * x),
                jnp.sin(5.0 * x) + 0.1 * jnp.sin(18.0 * x),
            )
        ),
    )

    raw_gxi = dno_series_eval(
        initial_state.eta,
        initial_state.xi,
        params.k,
        params.depth,
        params.dno_order,
        pad_factor=params.pad_factor,
    )
    expected = apply_lowpass(raw_gxi, params.k, params.filter_fraction)
    result = batched_rollout(
        initial_state,
        jnp.asarray([0.0]),
        params,
        save_gxi=True,
    )

    np.testing.assert_allclose(
        np.asarray(result["gxi"][0]), np.asarray(expected), atol=1e-12
    )
    assert float(jnp.linalg.norm(raw_gxi - expected)) > 1.0


def main() -> None:
    test_saved_gxi_respects_rollout_lowpass()
    print("[PASS] saved gxi respects the rollout low-pass")


if __name__ == "__main__":
    main()
