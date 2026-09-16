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

from dno_net_v2 import CraigSulemDNO, fourier_resample_even

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


def test_coarse_correction_keeps_output_grid_and_linearity() -> None:
    eta, xi, depth = _state()
    model = _model().clone(learned_grid=32)
    inputs = jnp.stack((eta, xi), axis=-1)
    variables = _activate_residual(model.init(jax.random.PRNGKey(5), inputs, depth))
    output = model.apply(variables, inputs, depth)
    assert output.shape == (1, 64, 1)
    psi = jnp.roll(xi, 7, axis=-1)
    combined = _learned_residual(model, variables, eta, xi + psi, depth)
    separate = _learned_residual(model, variables, eta, xi, depth)
    separate += _learned_residual(model, variables, eta, psi, depth)
    assert jnp.allclose(combined, separate, rtol=1e-10, atol=1e-12)
    assert jnp.max(jnp.abs(_learned_residual(model, variables, jnp.zeros_like(eta), xi, depth))) < 1e-14
    roundtrip = fourier_resample_even(fourier_resample_even(inputs, 32), 64)
    assert jnp.allclose(roundtrip, inputs, rtol=1e-12, atol=1e-12)


def test_compact_correction_invariants_and_initial_gradient() -> None:
    eta, xi, depth = _state()
    inputs = jnp.stack((eta, xi), axis=-1)
    for rank in (8, 16):
        model = _model().clone(
            learned_grid=32, correction_kind="compact", compact_rank=rank, compact_hidden=16,
        )
        variables = model.init(jax.random.key(rank), inputs, depth)
        assert jnp.max(jnp.abs(_learned_residual(model, variables, eta, xi, depth))) < 1e-14
        gradient = jax.grad(lambda params: jnp.sum(
            (model.apply({"params": params}, inputs, depth)[..., 0] - xi)**2
        ))(variables["params"])
        initial_gradient = gradient["compact"]["matrix_right"]["kernel"]
        assert jnp.isfinite(initial_gradient).all() and jnp.linalg.norm(initial_gradient) > 0

        variables = unfreeze(variables)
        kernel = variables["params"]["compact"]["matrix_right"]["kernel"]
        variables["params"]["compact"]["matrix_right"]["kernel"] = 0.2 * jax.random.normal(
            jax.random.key(rank + 1), kernel.shape, dtype=kernel.dtype,
        )

        def residual(surface: jax.Array, potential: jax.Array) -> jax.Array:
            return _learned_residual(model, variables, surface, potential, depth)

        psi = jnp.roll(xi, 7, axis=-1)
        actual = residual(eta, xi)
        assert actual.shape == xi.shape and jnp.linalg.norm(actual) > 0
        assert jnp.allclose(residual(eta, 2 * xi - psi), 2 * actual - residual(eta, psi), atol=1e-12)
        assert jnp.allclose(jnp.vdot(psi, actual), jnp.vdot(residual(eta, psi), xi), atol=1e-12)
        assert jnp.max(jnp.abs(actual.mean(axis=-1))) < 1e-12
        assert jnp.max(jnp.abs(residual(eta, jnp.ones_like(xi)))) < 1e-12
        value, derivative = jax.jvp(lambda surface: residual(surface, xi), (jnp.zeros_like(eta),), (eta,))
        assert jnp.max(jnp.abs(value)) < 1e-14 and jnp.max(jnp.abs(derivative)) < 1e-14
        quadratic_ratio = jnp.linalg.norm(residual(0.02 * eta, xi)) / jnp.linalg.norm(residual(0.01 * eta, xi))
        assert 3.9 < float(quadratic_ratio) < 4.1
        outside_basis = jnp.sin((rank // 2 + 2) * 2 * jnp.pi * jnp.arange(64) / 64)[None, :]
        assert jnp.max(jnp.abs(residual(eta, outside_basis))) < 1e-12


def test_spectral_mlp_learns_full_grid_correction() -> None:
    eta, xi, depth = _state(256)
    model = _model().clone(
        learned_grid=128, correction_kind="spectral_mlp", spectral_hidden=16,
        spectral_layers=2, spectral_channels=4, spectral_decoder_hidden=8,
    )
    inputs = jnp.stack((eta, xi), axis=-1)
    variables = model.init(jax.random.key(41), inputs, depth)
    assert jnp.max(jnp.abs(_learned_residual(model, variables, eta, xi, depth))) < 1e-14
    gradients = jax.grad(lambda params: jnp.sum(
        (model.apply({"params": params}, inputs, depth)[..., 0] - xi)**2
    ))(variables["params"])
    assert all(jnp.isfinite(leaf).all() for leaf in jax.tree.leaves(gradients))
    assert jnp.linalg.norm(gradients["spectral_mlp"]["decoder_out"]["kernel"]) > 0
    variables = unfreeze(variables)
    kernel = variables["params"]["spectral_mlp"]["decoder_out"]["kernel"]
    variables["params"]["spectral_mlp"]["decoder_out"]["kernel"] = 0.1 * jax.random.normal(
        jax.random.key(42), kernel.shape, dtype=kernel.dtype,
    )
    residual = _learned_residual(model, variables, eta, xi, depth)
    assert residual.shape == xi.shape and jnp.linalg.norm(residual) > 0
    assert jnp.max(jnp.abs(residual.mean(axis=-1))) < 1e-12
    high_mode = jnp.sin(48 * 2 * jnp.pi * jnp.arange(256) / 256)[None, :]
    _, response = jax.jvp(
        lambda potential: _learned_residual(model, variables, eta, potential, depth),
        (xi,), (high_mode,),
    )
    assert jnp.linalg.norm(response) > 1e-8
    assert jnp.linalg.norm(jnp.fft.rfft(residual, axis=-1)[:, 33:64]) > 1e-8


def test_canonical_fno_gradient_and_translation() -> None:
    eta, xi, depth = _state()
    model = _model().clone(learned_grid=32, correction_kind="canonical_fno")
    inputs = jnp.stack((eta, xi), axis=-1)
    variables = model.init(jax.random.key(51), inputs, depth)
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(variables["params"]))
    assert model.apply(variables, inputs.astype(jnp.float32), depth.astype(jnp.float32)).dtype == jnp.float32
    assert jnp.max(jnp.abs(_learned_residual(model, variables, eta, xi, depth))) < 1e-14
    gradients = jax.grad(lambda params: jnp.sum(
        (model.apply({"params": params}, inputs, depth)[..., 0] - xi)**2
    ))(variables["params"])
    assert all(jnp.isfinite(leaf).all() for leaf in jax.tree.leaves(gradients))
    assert jnp.linalg.norm(gradients["canonical_fno"]["decoder_out"]["kernel"]) > 0
    variables = unfreeze(variables)
    kernel = variables["params"]["canonical_fno"]["decoder_out"]["kernel"]
    variables["params"]["canonical_fno"]["decoder_out"]["kernel"] = 0.1 * jax.random.normal(
        jax.random.key(52), kernel.shape, dtype=kernel.dtype,
    )
    residual = _learned_residual(model, variables, eta, xi, depth)
    assert residual.shape == xi.shape and jnp.linalg.norm(residual) > 0
    assert jnp.max(jnp.abs(residual.mean(axis=-1))) < 1e-12
    shifted = _learned_residual(
        model, variables, jnp.roll(eta, 8, axis=1), jnp.roll(xi, 8, axis=1), depth,
    )
    assert jnp.allclose(shifted, jnp.roll(residual, 8, axis=1), rtol=1e-9, atol=1e-11)
    for index in range(4):
        variables["params"]["canonical_fno"][f"spatial_bias_{index}"] = 0.01 * jax.random.normal(
            jax.random.key(60 + index), (32,), dtype=jnp.float32,
        )
    # A nonzero decoder and biases exercise every block's parameter/input gradients.
    for dtype in (jnp.float32, jnp.float64):
        fields, depths = inputs.astype(dtype), depth.astype(dtype)
        predictions = []
        derivatives = []
        configurations = (
            (False, "complex", "fft"), (True, "complex", "fft"),
            (True, "packed", "fft"), (True, "split", "fft"),
            (True, "packed", "fft_backward"), (True, "packed", "dft"),
            (True, "packed", "fft_channels_first"),
        )
        for folded, gemm, transform in configurations:
            candidate = model.clone(fno_fold_spatial=folded, fno_spectral_gemm=gemm, fno_transform=transform)
            predictions.append(candidate.apply(variables, fields, depths))
            derivatives.append(jax.grad(lambda params, values: jnp.sum(
                (candidate.apply({"params": params}, values, depths)[..., 0] - xi)**2
            ), argnums=(0, 1))(variables["params"], fields))
        for prediction, derivative in zip(predictions[1:], derivatives[1:], strict=True):
            assert jnp.allclose(predictions[0], prediction, rtol=1e-5, atol=1e-7)
            for original, candidate in zip(jax.tree.leaves(derivatives[0]), jax.tree.leaves(derivative), strict=True):
                assert jnp.allclose(original, candidate, rtol=1e-4, atol=1e-7)


def main() -> int:
    tests: tuple[Callable[[], None], ...] = (
        test_order_two_has_zero_value_and_first_variation,
        test_order_two_is_quadratic_near_zero,
        test_order_two_residual_is_self_adjoint,
        test_eta_feature_configuration_controls_trunk_shape,
        test_coarse_correction_keeps_output_grid_and_linearity,
        test_compact_correction_invariants_and_initial_gradient,
        test_spectral_mlp_learns_full_grid_correction,
        test_canonical_fno_gradient_and_translation,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
