"""Focused CPU invariants for the Craig--Sulem DNO architecture.

Run with::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=True .venv/bin/python \
        models/dno-net/tests_dno_net_v2.py
"""
from __future__ import annotations

import os
from collections.abc import Callable

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

from dno_net_v2 import CraigSulemDNO

jax.config.update("jax_enable_x64", True)


def _model(**overrides: object) -> CraigSulemDNO:
    config: dict[str, object] = {
        "modes": 16,
        "width": 32,
        "n_blocks": 2,
        "latent": 8,
        "n_polys": 3,
        "mult_hidden": 16,
        "use_g1_baseline": True,
        "g1_k_cut": 0,
        "g1_fft_fp64": True,
        "tie_xi_out_mult": True,
        "domain_length": 2.0 * jnp.pi,
    }
    config.update(overrides)
    return CraigSulemDNO(**config)


def _state(nx: int = 64) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x = jnp.linspace(0.0, 2.0 * jnp.pi, nx, endpoint=False, dtype=jnp.float64)
    eta = 0.04 * jnp.cos(3.0 * x)[None, :] + 0.015 * jnp.sin(5.0 * x)[None, :]
    xi = 0.07 * jnp.sin(2.0 * x)[None, :] - 0.02 * jnp.cos(7.0 * x)[None, :]
    depth = jnp.log(jnp.asarray([[0.8]], dtype=jnp.float64))
    return eta, xi, depth


def _activate_residual(variables: dict[str, object]) -> dict[str, object]:
    """Replace zero-init block projections to mimic a trained checkpoint."""
    mutable = unfreeze(variables)
    for block_idx in range(2):
        kernel = mutable["params"][f"cs_block_{block_idx}"]["phi_proj"]["kernel"]
        key = jax.random.PRNGKey(100 + block_idx)
        mutable["params"][f"cs_block_{block_idx}"]["phi_proj"]["kernel"] = (
            0.2 * jax.random.normal(key, kernel.shape, dtype=kernel.dtype)
        )
    return freeze(mutable)


def _learned_residual(
    model: CraigSulemDNO,
    variables: dict[str, object],
    eta: jnp.ndarray,
    xi: jnp.ndarray,
    depth: jnp.ndarray,
) -> jnp.ndarray:
    inputs = jnp.stack((eta, xi), axis=-1)
    output = model.apply(variables, inputs, depth)[..., 0]
    baseline = model._linear_baseline(xi, depth) + model._g1_baseline(eta, xi, depth)
    return output - baseline


def test_order_one_is_unchanged() -> None:
    """The default and explicit order-one models retain identical trees/outputs."""
    eta, xi, depth = _state()
    inputs = jnp.stack((eta, xi), axis=-1)
    default_model = _model(g1_fft_fp64=False)
    order_one_model = _model(g1_fft_fp64=False, residual_eta_order=1)
    default_vars = default_model.init(jax.random.PRNGKey(0), inputs, depth)
    order_one_vars = order_one_model.init(jax.random.PRNGKey(0), inputs, depth)

    default_shapes = jax.tree.map(lambda value: value.shape, default_vars)
    order_one_shapes = jax.tree.map(lambda value: value.shape, order_one_vars)
    assert default_shapes == order_one_shapes
    default_output = default_model.apply(default_vars, inputs, depth)
    order_one_output = order_one_model.apply(order_one_vars, inputs, depth)
    assert jnp.array_equal(default_output, order_one_output)


def test_order_two_has_zero_value_and_first_variation() -> None:
    """R(0,xi) and D_eta R(0,xi) are structural zeros after nonzero weights."""
    eta, xi, depth = _state()
    model = _model(residual_eta_order=2)
    inputs = jnp.stack((eta, xi), axis=-1)
    variables = _activate_residual(model.init(jax.random.PRNGKey(1), inputs, depth))

    def residual(surface: jnp.ndarray) -> jnp.ndarray:
        return _learned_residual(model, variables, surface, xi, depth)

    eta_zero = jnp.zeros_like(eta)
    value_zero, first_variation = jax.jvp(residual, (eta_zero,), (eta,))
    assert jnp.max(jnp.abs(value_zero)) < 1e-14
    assert jnp.max(jnp.abs(first_variation)) < 1e-14


def test_order_two_is_quadratic_near_zero() -> None:
    """The diagnostic ||R(eps*eta)||/eps tends to zero linearly in eps."""
    eta, xi, depth = _state()
    model = _model(residual_eta_order=2)
    inputs = jnp.stack((eta, xi), axis=-1)
    variables = _activate_residual(model.init(jax.random.PRNGKey(2), inputs, depth))
    epsilons = (0.25, 0.125, 0.0625)
    quotients = [
        jnp.linalg.norm(_learned_residual(model, variables, eps * eta, xi, depth)) / eps
        for eps in epsilons
    ]

    assert all(float(value) > 0.0 for value in quotients)
    assert float(quotients[1]) < 0.65 * float(quotients[0])
    assert float(quotients[2]) < 0.65 * float(quotients[1])


def test_order_two_residual_is_self_adjoint() -> None:
    """Tied real multiplier sandwiches remain self-adjoint after the lift."""
    eta, xi, depth = _state()
    psi = jnp.roll(xi, 9, axis=-1) + 0.03 * jnp.sin(
        jnp.linspace(0.0, 6.0 * jnp.pi, xi.shape[-1], endpoint=False)
    )[None, :]
    model = _model(residual_eta_order=2)
    inputs = jnp.stack((eta, xi), axis=-1)
    variables = _activate_residual(model.init(jax.random.PRNGKey(3), inputs, depth))

    residual_xi = _learned_residual(model, variables, eta, xi, depth)
    residual_psi = _learned_residual(model, variables, eta, psi, depth)
    lhs = jnp.vdot(psi, residual_xi)
    rhs = jnp.vdot(residual_psi, xi)
    scale = jnp.maximum(jnp.maximum(jnp.abs(lhs), jnp.abs(rhs)), 1e-14)
    assert float(jnp.abs(lhs - rhs) / scale) < 1e-11


def test_g1_only_fp64_is_isolated() -> None:
    """The narrow fp64 flag matches full-fp64 G1 and does not alter G0."""
    eta, xi, depth = _state()
    eta32, xi32, depth32 = eta.astype(jnp.float32), xi.astype(jnp.float32), depth.astype(jnp.float32)
    regular = _model(fft_fp64=False, g1_fft_fp64=False)
    g1_only = _model(fft_fp64=False, g1_fft_fp64=True)
    full = _model(fft_fp64=True, g1_fft_fp64=False)

    assert jnp.array_equal(
        regular._linear_baseline(xi32, depth32),
        g1_only._linear_baseline(xi32, depth32),
    )
    assert jnp.array_equal(
        g1_only._g1_baseline(eta32, xi32, depth32),
        full._g1_baseline(eta32, xi32, depth32),
    )


def main() -> int:
    tests: tuple[Callable[[], None], ...] = (
        test_order_one_is_unchanged,
        test_order_two_has_zero_value_and_first_variation,
        test_order_two_is_quadratic_near_zero,
        test_order_two_residual_is_self_adjoint,
        test_g1_only_fp64_is_isolated,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
