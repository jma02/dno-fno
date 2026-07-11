"""CPU tests for two-times dealiasing of the nonlinear Zakharov RHS.

Run directly; pytest is not required:

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
      .venv/bin/python solver/solvers/tests_zakharov_dealias.py
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from solver.solvers.time_integrator import dealiased_zakharov_xi_rhs  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _base_grid_rhs(
    eta_x: jax.Array,
    xi_x: jax.Array,
    gxi: jax.Array,
) -> jax.Array:
    numerator = gxi + eta_x * xi_x
    return -0.5 * xi_x**2 + 0.5 * numerator**2 / (1.0 + eta_x**2)


def test_bandlimited_products_match_base_grid() -> None:
    nx = 64
    x = 2.0 * jnp.pi * jnp.arange(nx) / nx
    eta_x = jnp.zeros((2, nx))
    xi_x = jnp.stack(
        (
            0.20 * jnp.cos(3.0 * x) + 0.07 * jnp.sin(5.0 * x),
            0.13 * jnp.cos(4.0 * x) - 0.09 * jnp.sin(6.0 * x),
        )
    )
    gxi = jnp.stack(
        (0.11 * jnp.sin(2.0 * x), 0.08 * jnp.cos(5.0 * x))
    )

    expected = _base_grid_rhs(eta_x, xi_x, gxi)
    actual = jax.jit(dealiased_zakharov_xi_rhs)(eta_x, xi_x, gxi)
    assert actual.shape == (2, nx)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-7)


def test_high_band_product_alias_is_removed() -> None:
    nx = 64
    x = 2.0 * jnp.pi * jnp.arange(nx) / nx
    eta_x = jnp.zeros(nx)
    xi_x = jnp.cos(22.0 * x)
    gxi = jnp.zeros(nx)

    aliased = _base_grid_rhs(eta_x, xi_x, gxi)
    dealiased = dealiased_zakharov_xi_rhs(eta_x, xi_x, gxi)
    expected = -0.25 * jnp.ones(nx)

    np.testing.assert_allclose(np.asarray(dealiased), np.asarray(expected), atol=2e-6)
    assert float(jnp.linalg.norm(aliased - dealiased)) > 1.0


def main() -> None:
    test_bandlimited_products_match_base_grid()
    print("[PASS] bandlimited products preserve base-grid RHS")
    test_high_band_product_alias_is_removed()
    print("[PASS] high-band product alias is removed")


if __name__ == "__main__":
    main()
