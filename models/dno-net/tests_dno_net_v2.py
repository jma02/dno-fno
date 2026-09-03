"""Focused CPU invariants for the Craig--Sulem DNO architecture.

Run with::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=True .venv/bin/python \
        models/dno-net/tests_dno_net_v2.py
"""

from __future__ import annotations

import os
from collections.abc import Callable
from math import pi
from typing import Any, cast

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from flax.typing import FrozenVariableDict

from dno_net_v2 import CraigSulemDNO, validate_fixed_craig_sulem_config

jax.config.update("jax_enable_x64", True)


VariableState = FrozenVariableDict | dict[str, Any]


def _model(
    *,
    n_polys: int = 3,
    use_first_deriv: bool = True,
    use_second_deriv: bool = True,
    use_half_deriv: bool = True,
    use_hilbert: bool = True,
) -> CraigSulemDNO:
    return CraigSulemDNO(
        width=32,
        n_blocks=2,
        latent=8,
        n_polys=n_polys,
        use_first_deriv=use_first_deriv,
        use_second_deriv=use_second_deriv,
        use_half_deriv=use_half_deriv,
        use_hilbert=use_hilbert,
        mult_hidden=16,
        domain_length=2.0 * pi,
    )


def _state(nx: int = 64) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x = jnp.linspace(0.0, 2.0 * jnp.pi, nx, endpoint=False, dtype=jnp.float64)
    eta = 0.04 * jnp.cos(3.0 * x)[None, :] + 0.015 * jnp.sin(5.0 * x)[None, :]
    xi = 0.07 * jnp.sin(2.0 * x)[None, :] - 0.02 * jnp.cos(7.0 * x)[None, :]
    depth = jnp.log(jnp.asarray([[0.8]], dtype=jnp.float64))
    return eta, xi, depth


def _activate_residual(variables: VariableState) -> FrozenVariableDict:
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
    variables: VariableState,
    eta: jnp.ndarray,
    xi: jnp.ndarray,
    depth: jnp.ndarray,
) -> jnp.ndarray:
    inputs = jnp.stack((eta, xi), axis=-1)
    output = cast(jax.Array, model.apply(variables, inputs, depth))[..., 0]
    baseline = model._linear_baseline(xi, depth) + model._g1_baseline(eta, xi, depth)
    return output - baseline


def test_order_two_has_zero_value_and_first_variation() -> None:
    """R(0,xi) and D_eta R(0,xi) are structural zeros after nonzero weights."""
    eta, xi, depth = _state()
    model = _model()
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
    model = _model()
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
    psi = (
        jnp.roll(xi, 9, axis=-1)
        + 0.03
        * jnp.sin(jnp.linspace(0.0, 6.0 * jnp.pi, xi.shape[-1], endpoint=False))[
            None, :
        ]
    )
    model = _model()
    inputs = jnp.stack((eta, xi), axis=-1)
    variables = _activate_residual(model.init(jax.random.PRNGKey(3), inputs, depth))

    residual_xi = _learned_residual(model, variables, eta, xi, depth)
    residual_psi = _learned_residual(model, variables, eta, psi, depth)
    lhs = jnp.vdot(psi, residual_xi)
    rhs = jnp.vdot(residual_psi, xi)
    scale = jnp.maximum(jnp.maximum(jnp.abs(lhs), jnp.abs(rhs)), 1e-14)
    assert float(jnp.abs(lhs - rhs) / scale) < 1e-11


def test_eta_feature_configuration_controls_trunk_shape() -> None:
    """Polynomial and derivative switches remain a live exploratory surface."""
    eta, xi, depth = _state()
    inputs = jnp.stack((eta, xi), axis=-1)
    configurations = (
        (_model(n_polys=1), 5),
        (_model(n_polys=2, use_second_deriv=False), 5),
        (
            _model(
                n_polys=3,
                use_first_deriv=False,
                use_second_deriv=False,
                use_half_deriv=False,
                use_hilbert=False,
            ),
            3,
        ),
    )
    for model, expected_channels in configurations:
        variables = model.init(jax.random.PRNGKey(expected_channels), inputs, depth)
        kernel = cast(jax.Array, variables["params"]["eta_feat_proj"]["kernel"])
        assert kernel.shape[0] == expected_channels
        output = cast(jax.Array, model.apply(variables, inputs, depth))
        assert jnp.all(jnp.isfinite(output))


def test_legacy_config_guard_accepts_only_c27_architecture() -> None:
    """Archived C27 configs load, while removed CS-DNO variants fail loudly."""
    validate_fixed_craig_sulem_config({"cs_use_g1_baseline": True})
    try:
        validate_fixed_craig_sulem_config({"cs_use_g1_baseline": False})
    except ValueError as exc:
        assert "removed experimental CS-DNO architecture" in str(exc)
    else:
        raise AssertionError("legacy non-C27 architecture was accepted")


def main() -> int:
    tests: tuple[Callable[[], None], ...] = (
        test_order_two_has_zero_value_and_first_variation,
        test_order_two_is_quadratic_near_zero,
        test_order_two_residual_is_self_adjoint,
        test_eta_feature_configuration_controls_trunk_shape,
        test_legacy_config_guard_accepts_only_c27_architecture,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
