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
from flax import linen as nn
from flax.core import freeze, unfreeze
from flax.typing import FrozenVariableDict

from dno_net_v2 import CraigSulemBlock, CraigSulemDNO, DepthAwareMultiplier
from solver.solvers.dno_series_jax import dno_series_eval

jax.config.update("jax_enable_x64", True)


VariableState = FrozenVariableDict | dict[str, Any]


def _model(
    *,
    baseline_order: int = 1,
    n_polys: int = 3,
    use_first_deriv: bool = True,
    use_second_deriv: bool = True,
    use_half_deriv: bool = True,
    use_hilbert: bool = True,
    use_relative_elevation: bool = False,
    use_learned_eta_filters: bool = False,
    surface_feature_scales: tuple[float, ...] | None = None,
    surface_context_width: int = 0,
    surface_context_gate: bool = False,
    surface_polynomial_skip: bool = False,
    nested_blocks: int = 0,
    multiplier_skip: bool = False,
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
        use_relative_elevation=use_relative_elevation,
        use_learned_eta_filters=use_learned_eta_filters,
        surface_feature_scales=surface_feature_scales,
        surface_context_width=surface_context_width,
        surface_context_gate=surface_context_gate,
        surface_polynomial_skip=surface_polynomial_skip,
        nested_blocks=nested_blocks,
        multiplier_skip=multiplier_skip,
        mult_hidden=16,
        baseline_order=baseline_order,
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
        block = mutable["params"][f"cs_block_{block_idx}"]
        key = jax.random.PRNGKey(100 + block_idx)
        if "gamma" in block:
            block["gamma"] = 0.2 * jax.random.normal(key, block["gamma"].shape, dtype=block["gamma"].dtype)
        else:
            kernel = block["phi_proj"]["kernel"]
            block["phi_proj"]["kernel"] = 0.2 * jax.random.normal(key, kernel.shape, dtype=kernel.dtype)
    if "eta_filter_out" in mutable["params"]:
        kernel = mutable["params"]["eta_filter_out"]["kernel"]
        mutable["params"]["eta_filter_out"]["kernel"] = 0.2 * jax.random.normal(
            jax.random.PRNGKey(102), kernel.shape, dtype=kernel.dtype,
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
    output, baseline, _ = cast(tuple[jax.Array, jax.Array, jax.Array], model.apply(
        variables, inputs, depth, return_baseline=True,
    ))
    return (output - baseline)[..., 0]


def test_default_matches_original_forward() -> None:
    """Both opt-in experiments leave the existing learned M1 forward unchanged."""
    eta, xi, depth = _state()
    model = _model()
    inputs = jnp.stack((eta, xi), axis=-1)
    variables = _activate_residual(model.init(jax.random.PRNGKey(0), inputs, depth))
    k = jnp.arange(eta.shape[-1] // 2 + 1, dtype=eta.dtype)
    eta_hat = jnp.fft.rfft(eta)
    features = jnp.stack([
        eta, eta**2, eta**3,
        *(jnp.fft.irfft(symbol * eta_hat, n=eta.shape[-1]) for symbol in
          (1j * k, -(k**2), jnp.sqrt(k), -1j * jnp.sign(k))),
    ], axis=-1)
    for name in ("eta_feat_proj", "eta_feat_mix"):
        features = nn.gelu(features @ cast(jax.Array, variables["params"][name]["kernel"]))
    features = features * jnp.tanh(features)
    residual = jnp.zeros_like(xi)
    for index in range(model.n_blocks):
        block = CraigSulemBlock(model.latent, model.domain_length, model.h_clip_max, model.mult_hidden)
        residual += cast(jax.Array, block.apply(
            {"params": variables["params"][f"cs_block_{index}"]}, features, xi, depth,
        ))
    expected = model._linear_baseline(xi, depth) + model._g1_baseline(eta, xi, depth)
    expected += residual / (model.n_blocks * model.latent) ** 0.5
    actual = cast(jax.Array, model.apply(variables, inputs, depth))[..., 0]
    assert jnp.array_equal(actual, expected)


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


def test_learned_operator_is_self_adjoint_and_linear() -> None:
    """Tied real filters preserve both properties with or without a constant input."""
    eta, xi, depth = _state()
    psi = (
        jnp.roll(xi, 9, axis=-1)
        + 0.03
        * jnp.sin(jnp.linspace(0.0, 6.0 * jnp.pi, xi.shape[-1], endpoint=False))[
            None, :
        ]
    )
    inputs = jnp.stack((eta, xi), axis=-1)
    for baseline_order, relative_elevation, learned_filters, context_width, context_gate in (
        (-1, False, False, 0, False), (0, False, False, 0, False), (1, False, False, 0, False),
        (1, True, False, 0, False), (2, False, False, 0, False), (1, False, True, 0, False),
        (1, False, False, 8, False), (1, False, False, 8, True),
    ):
        model = _model(baseline_order=baseline_order, use_relative_elevation=relative_elevation,
                       use_learned_eta_filters=learned_filters, surface_context_width=context_width,
                       surface_context_gate=context_gate)
        variables = _activate_residual(model.init(jax.random.PRNGKey(3), inputs, depth))
        residual_xi = _learned_residual(model, variables, eta, xi, depth)
        residual_psi = _learned_residual(model, variables, eta, psi, depth)
        lhs = jnp.vdot(psi, residual_xi)
        rhs = jnp.vdot(residual_psi, xi)
        scale = jnp.maximum(jnp.maximum(jnp.abs(lhs), jnp.abs(rhs)), 1e-14)
        assert float(jnp.abs(lhs - rhs) / scale) < 1e-11
        combined = _learned_residual(model, variables, eta, 0.3 * xi - 0.7 * psi, depth)
        expected = 0.3 * residual_xi - 0.7 * residual_psi
        assert float(jnp.linalg.norm(combined - expected) / jnp.linalg.norm(expected)) < 1e-11
        if baseline_order == 2:
            full_xi = cast(jax.Array, model.apply(variables, inputs, depth))[..., 0]
            full_psi = cast(jax.Array, model.apply(variables, jnp.stack((eta, psi), axis=-1), depth))[..., 0]
            assert jnp.allclose(jnp.vdot(psi, full_xi), jnp.vdot(full_psi, xi), rtol=1e-11, atol=1e-14)
            full_combined = cast(jax.Array, model.apply(variables, jnp.stack((eta, 0.3 * xi - 0.7 * psi), axis=-1), depth))[..., 0]
            assert jnp.allclose(full_combined, 0.3 * full_xi - 0.7 * full_psi, rtol=1e-11, atol=1e-14)


def test_baseline_free_flat_surface_is_trainable() -> None:
    """The full neural operator can learn nonzero value and first variation at eta=0."""
    eta, xi, depth = _state()
    model = _model(baseline_order=-1)
    inputs = jnp.stack((jnp.zeros_like(eta), xi), axis=-1)
    variables = model.init(jax.random.PRNGKey(4), inputs, depth)
    assert jnp.all(model.apply(variables, inputs, depth) == 0)
    target = model._linear_baseline(xi, depth)[..., None]

    def flat_loss(params: Any) -> jax.Array:
        prediction = cast(jax.Array, model.apply({"params": params}, inputs, depth))
        return jnp.mean((prediction - target) ** 2)

    loss, gradient = jax.value_and_grad(flat_loss)(variables["params"])
    updated = jax.tree.map(lambda p, g: p - 1e-3 * g, variables["params"], gradient)
    assert float(flat_loss(updated)) < float(loss)
    active = _activate_residual(variables)
    value, variation = jax.jvp(
        lambda surface: _learned_residual(model, active, surface, xi, depth),
        (jnp.zeros_like(eta),), (eta,),
    )
    assert float(jnp.linalg.norm(value)) > 1e-14
    assert float(jnp.linalg.norm(variation)) > 1e-14


def test_g0_only_preserves_flat_surface_and_allows_first_order_correction() -> None:
    """G0 remains exact at eta=0 while the learned first variation is nonzero."""
    eta, xi, depth = _state()
    model = _model(baseline_order=0)
    inputs = jnp.stack((eta, xi), axis=-1)
    variables = model.init(jax.random.PRNGKey(5), inputs, depth)
    assert jnp.array_equal(
        cast(jax.Array, model.apply(variables, inputs, depth))[..., 0], model._linear_baseline(xi, depth)
    )
    active = _activate_residual(variables)

    def residual(surface: jnp.ndarray) -> jnp.ndarray:
        return _learned_residual(model, active, surface, xi, depth)

    value, variation = jax.jvp(residual, (jnp.zeros_like(eta),), (eta,))
    assert jnp.max(jnp.abs(value)) < 1e-14
    assert float(jnp.linalg.norm(variation)) > 1e-14
    for epsilon in (1e-3, 5e-4):
        relative = jnp.linalg.norm(residual(epsilon * eta) / epsilon - variation) / jnp.linalg.norm(variation)
        assert float(relative) < 1e-3


def test_eta_feature_configuration_controls_trunk_shape() -> None:
    """Polynomial and derivative switches remain a live exploratory surface."""
    eta, xi, depth = _state()
    inputs = jnp.stack((eta, xi), axis=-1)
    configurations = (
        (_model(baseline_order=-1), 8),
        (_model(baseline_order=0), 7),
        (_model(use_relative_elevation=True), 8),
        (_model(use_learned_eta_filters=True), 7),
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


def test_relative_elevation_uses_physical_eta_and_unclipped_depth() -> None:
    eta, xi, _ = _state()
    depth = jnp.log(jnp.asarray([[0.1], [8.0]], dtype=eta.dtype))
    inputs = jnp.broadcast_to(jnp.stack((eta, xi), axis=-1), (2, eta.shape[-1], 2))
    model = _model(use_relative_elevation=True).clone(eta_scale=0.17)
    variables = unfreeze(model.init(jax.random.PRNGKey(8), inputs, depth))
    kernel = variables["params"]["eta_feat_proj"]["kernel"]
    variables["params"]["eta_feat_proj"]["kernel"] = jnp.zeros_like(kernel).at[-1, 0].set(1)
    _, captured = model.apply(
        variables, inputs, depth,
        capture_intermediates=lambda module, _: module.name == "eta_feat_proj",
    )
    projection = cast(jax.Array, captured["intermediates"]["eta_feat_proj"]["__call__"][0])
    assert jnp.allclose(projection[..., 0], inputs[..., 0] * model.eta_scale / jnp.exp(depth), rtol=1e-14)
    assert jnp.all(projection[..., 1:] == 0)


def test_m2_zero_init_matches_classical_batched_fp64() -> None:
    eta, xi, _ = _state()
    model = _model(baseline_order=2).clone(eta_scale=0.17, xi_scale=0.08, target_scale=0.13)
    for dtype in (jnp.float32, jnp.float64):
        inputs = jnp.stack((jnp.concatenate((eta, 0.7 * eta)), jnp.concatenate((xi, -xi))), axis=-1).astype(dtype)
        depth = jnp.log(jnp.asarray([[0.08], [12.0]], dtype=dtype))
        variables = model.init(jax.random.PRNGKey(9), inputs, depth)
        actual = cast(jax.Array, model.apply(variables, inputs, depth))[..., 0]
        expected = dno_series_eval(
            (inputs[..., 0] * model.eta_scale).astype(jnp.float64),
            (inputs[..., 1] * model.xi_scale).astype(jnp.float64),
            jnp.arange(eta.shape[-1] // 2 + 1, dtype=jnp.float64),
            jnp.exp(jnp.minimum(depth, jnp.log(model.h_clip_max))).astype(jnp.float64),
            order=2,
        ) / model.target_scale
        assert actual.dtype == jnp.float64
        assert jnp.array_equal(actual, expected)
        assert jnp.allclose(jnp.mean(actual, axis=-1), 0, atol=1e-14)
        assert jnp.allclose(jnp.fft.rfft(actual)[..., -1], 0, atol=1e-14)
        gradient = jax.grad(lambda params: jnp.mean(cast(jax.Array, model.apply({"params": params}, inputs, depth)) ** 2))(variables["params"])
        assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(gradient))
        assert sum(float(jnp.sum(leaf**2)) for leaf in jax.tree.leaves(gradient)) > 0


def test_learned_eta_filters_preserve_initial_features_and_shared_parameters() -> None:
    """The opt-in starts at the identical function, not just an identical baseline."""
    eta, xi, depth = _state()
    inputs = jnp.stack((eta, xi), axis=-1)
    fixed = _model()
    learned = _model(use_learned_eta_filters=True)
    original = fixed.init(jax.random.PRNGKey(13), inputs, depth)
    variables = learned.init(jax.random.PRNGKey(13), inputs, depth)
    for name, parameters in original["params"].items():
        assert all(jnp.array_equal(a, b) for a, b in zip(
            jax.tree.leaves(parameters), jax.tree.leaves(variables["params"][name]), strict=True,
        ))
    original = _activate_residual(original)
    active = unfreeze(_activate_residual(variables))
    active["params"]["eta_filter_out"] = variables["params"]["eta_filter_out"]
    outputs, features = [], []
    for model, state in ((fixed, original), (learned, active)):
        output, captured = model.apply(
            state, inputs, depth,
            capture_intermediates=lambda module, _: module.name == "eta_feat_proj",
        )
        outputs.append(output)
        features.append(captured["intermediates"]["eta_feat_proj"]["__call__"][0])
    assert jnp.array_equal(features[0], features[1])
    assert jnp.array_equal(outputs[0], outputs[1])
    for state in (active, _activate_residual(variables)):
        gradient = jax.grad(lambda params: jnp.mean(cast(jax.Array, learned.apply(
            {"params": params}, inputs, depth,
        )) ** 2))(state["params"])
        assert jnp.linalg.norm(gradient["eta_filter_out"]["kernel"]) > 0
        if state is not active:
            assert jnp.linalg.norm(gradient["eta_filter_hidden"]["kernel"]) > 0
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient))


def test_learned_eta_filters_change_spectral_shape_and_condition_on_depth() -> None:
    """A nonlinear frequency envelope expands the fixed power-law feature basis."""
    eta, xi, _ = _state(128)
    eta = jnp.zeros_like(eta).at[:, 0].set(0.01)
    inputs = jnp.broadcast_to(jnp.stack((eta, xi), axis=-1), (2, eta.shape[-1], 2))
    depth = jnp.log(jnp.asarray([[0.1], [0.8]], dtype=eta.dtype))
    model = _model(use_learned_eta_filters=True)
    variables = unfreeze(model.init(jax.random.PRNGKey(14), inputs, depth))
    params = jax.tree.map(lambda value: value.astype(eta.dtype), variables["params"])
    variables["params"] = params
    params["eta_filter_hidden"]["kernel"] = jnp.zeros_like(params["eta_filter_hidden"]["kernel"]).at[:, 0].set(jnp.asarray([2.0, 0.2, 0.3]))
    params["eta_filter_hidden"]["bias"] = jnp.zeros_like(params["eta_filter_hidden"]["bias"]).at[0].set(-6)
    params["eta_filter_out"]["kernel"] = jnp.zeros_like(params["eta_filter_out"]["kernel"]).at[0, :].set(jnp.asarray([0.8, -0.6]))
    params["eta_feat_proj"]["kernel"] = jnp.eye(7, model.width // 2)
    _, captured = model.apply(
        variables, inputs, depth,
        capture_intermediates=lambda module, _: module.name == "eta_feat_proj",
    )
    features = cast(jax.Array, captured["intermediates"]["eta_feat_proj"]["__call__"][0])
    k = jnp.arange(eta.shape[-1] // 2 + 1, dtype=eta.dtype)
    hidden = jnp.tanh(2.0 * jnp.log1p(k) + 0.2 * depth + 0.3 * jnp.tanh(jnp.exp(depth) * k) - 6)
    for channel, symbol, scale in ((4, -(k**2), 0.8), (5, jnp.sqrt(k), -0.6)):
        expected = symbol * (1 + 0.5 * jnp.tanh(scale * hidden))
        actual = jnp.fft.rfft(features[..., channel]) / jnp.fft.rfft(eta)
        assert jnp.allclose(actual, expected, rtol=1e-11, atol=1e-10)
        assert jnp.linalg.norm(actual[0] - actual[1]) > 0
    p = k[1:]
    old_basis = jnp.stack((jnp.ones_like(p), p**0.5, p, p**2, p**2.5, p**4), axis=-1)
    old_basis /= jnp.linalg.norm(old_basis, axis=0)
    quadratic_gain = p**4 * (1 + 0.5 * jnp.tanh(0.8 * hidden[0, 1:]))**2
    projection = old_basis @ jnp.linalg.lstsq(old_basis, quadratic_gain, rcond=None)[0]
    assert jnp.linalg.norm(quadratic_gain - projection) / jnp.linalg.norm(quadratic_gain) > 1e-5


def test_true_depth_analytic_baselines_and_returned_components() -> None:
    eta, xi, _ = _state()
    for order in (0, 1, 2):
        model = _model(baseline_order=order).clone(
            baseline_true_depth=True, domain_length=8 * pi,
            eta_scale=0.17, xi_scale=0.08, target_scale=0.13,
        )
        clipped = model.clone(baseline_true_depth=False)
        for dtype in (jnp.float32, jnp.float64):
            inputs = jnp.broadcast_to(jnp.stack((eta, xi), axis=-1), (2, eta.shape[-1], 2)).astype(dtype)
            depth = jnp.log(jnp.asarray([[0.8], [12.0]], dtype=dtype))
            variables = model.init(jax.random.PRNGKey(12), inputs, depth)
            prediction, baseline, m1 = cast(tuple[jax.Array, jax.Array, jax.Array], model.apply(
                variables, inputs, depth, return_baseline=True,
            ))
            physical = (
                (inputs[..., 0] * model.eta_scale).astype(jnp.float64),
                (inputs[..., 1] * model.xi_scale).astype(jnp.float64),
                0.25 * jnp.arange(eta.shape[-1] // 2 + 1, dtype=jnp.float64),
                jnp.exp(depth).astype(jnp.float64),
            )
            expected = dno_series_eval(*physical, order=order)[..., None] / model.target_scale
            expected_m1 = dno_series_eval(*physical, order=min(order, 1))[..., None] / model.target_scale
            assert jnp.array_equal(prediction, cast(jax.Array, model.apply(variables, inputs, depth)))
            assert jnp.array_equal(prediction, baseline)
            tolerance = 2e-6 if dtype == jnp.float32 else 1e-11
            assert float(jnp.linalg.norm(baseline - expected) / jnp.linalg.norm(expected)) < tolerance
            assert float(jnp.linalg.norm(m1 - expected_m1) / jnp.linalg.norm(expected_m1)) < tolerance
            if order == 2:
                assert jnp.array_equal(baseline, expected)
            clipped_prediction = cast(jax.Array, clipped.apply(variables, inputs, depth))
            assert jnp.array_equal(clipped_prediction[0], prediction[0])
            assert float(jnp.linalg.norm(clipped_prediction[1] - prediction[1])) > 1e-5
            assert jnp.array_equal(clipped_prediction, cast(jax.Array, model.apply(
                variables, inputs, jnp.minimum(depth, jnp.log(model.h_clip_max)),
            )))
            if dtype == jnp.float64:
                active = _activate_residual(variables)
                true_residual = _learned_residual(model, active, inputs[..., 0], inputs[..., 1], depth)
                clipped_residual = _learned_residual(clipped, active, inputs[..., 0], inputs[..., 1], depth)
                assert float(jnp.linalg.norm(true_residual)) > 0
                assert jnp.allclose(true_residual, clipped_residual, rtol=1e-9, atol=1e-14)


def test_experimental_residual_orders() -> None:
    eta, xi, depth = _state()
    inputs = jnp.stack((eta, xi), axis=-1)
    for model, order in (
        (_model(use_relative_elevation=True), 2), (_model(baseline_order=2), 3),
        (_model(use_learned_eta_filters=True), 2),
        (_model(surface_context_width=8), 2),
        (_model(surface_context_width=8, surface_feature_scales=(0.2, 0.04, 0.008, 2., 8., 0.7, 0.3)), 2),
        (_model(surface_context_width=8, surface_context_gate=True,
                surface_feature_scales=(0.2, 0.04, 0.008, 2., 8., 0.7, 0.3)), 2),
        (_model(baseline_order=2, surface_context_width=8, surface_context_gate=True), 3),
        (_model(nested_blocks=1, surface_context_width=8), 2),
        (_model(nested_blocks=2), 2),
        (_model(surface_polynomial_skip=True, surface_context_width=8), 2),
    ):
        variables = _activate_residual(model.init(jax.random.PRNGKey(10), inputs, depth))
        def residual(surface: jnp.ndarray) -> jnp.ndarray:
            return _learned_residual(model, variables, surface, xi, depth)

        value, variation = jax.jvp(residual, (jnp.zeros_like(eta),), (eta,))
        assert jnp.max(jnp.abs(value)) < 1e-14
        assert jnp.max(jnp.abs(variation)) < 1e-14
        if order == 3:
            second = jax.jvp(lambda surface: jax.jvp(residual, (surface,), (eta,))[1], (jnp.zeros_like(eta),), (eta,))[1]
            assert jnp.max(jnp.abs(second)) < 1e-14
        norms = [float(jnp.linalg.norm(residual(epsilon * eta))) for epsilon in (0.125, 0.0625, 0.03125)]
        assert all(value > 0 for value in norms)
        assert all(abs(norms[index] / norms[index + 1] / 2**order - 1) < 0.1 for index in (0, 1))


def test_surface_feature_scaling_folds_into_first_projection() -> None:
    """Fixed RMS normalization needs no extra inference work once its kernel is folded."""
    eta, xi, depth = _state()
    inputs = jnp.stack((eta, xi), axis=-1)
    scales = (0.2, 0.04, 0.008, 2., 8., 0.7, 0.3)
    for context_width in (0, 8):
        normalized = _model(surface_feature_scales=scales, surface_context_width=context_width)
        folded = normalized.clone(surface_feature_scales=None)
        variables = _activate_residual(normalized.init(jax.random.PRNGKey(15), inputs, depth))
        variables = jax.tree.map(lambda value: value.astype(eta.dtype), variables)
        folded_variables = unfreeze(variables)
        folded_variables["params"]["eta_feat_proj"]["kernel"] /= jnp.asarray(scales)[:, None]
        expected = cast(jax.Array, normalized.apply(variables, inputs, depth))
        actual = cast(jax.Array, folded.apply(folded_variables, inputs, depth))
        assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-14)
        assert jnp.linalg.norm(_learned_residual(normalized, variables, eta, xi, depth)) > 0


def test_surface_context_is_nonlocal_and_depth_conditioned() -> None:
    """The learned context transports geometry beyond the support of local features."""
    eta, xi, _ = _state()
    eta = jnp.zeros_like(eta).at[:, 0].set(0.04)
    inputs = jnp.broadcast_to(jnp.stack((eta, xi), axis=-1), (2, eta.shape[-1], 2))
    depth = jnp.log(jnp.asarray([[0.1], [0.8]], dtype=eta.dtype))
    model = _model(n_polys=1, use_first_deriv=False, use_second_deriv=False,
                   use_half_deriv=False, use_hilbert=False, surface_context_width=8)
    variables = _activate_residual(model.init(jax.random.PRNGKey(16), inputs, depth))
    output, captured = model.apply(
        variables, inputs, depth,
        capture_intermediates=lambda module, _: module.name == "eta_context_expand",
    )
    context = cast(jax.Array, captured["intermediates"]["eta_context_expand"]["__call__"][0])
    assert jnp.linalg.norm(context[:, 1:, :]) > 1e-8
    assert jnp.linalg.norm(context[0] - context[1]) > 1e-8
    translated = cast(jax.Array, model.apply(variables, jnp.roll(inputs, 7, axis=1), depth))
    assert jnp.allclose(translated, jnp.roll(output, 7, axis=1), rtol=1e-10, atol=1e-13)


def test_surface_context_has_live_gradients_and_unchanged_shared_initialization() -> None:
    eta, xi, depth = _state()
    inputs = jnp.stack((eta, xi), axis=-1)
    model = _model(surface_context_width=8)
    original = _model().init(jax.random.PRNGKey(17), inputs, depth)
    variables = model.init(jax.random.PRNGKey(17), inputs, depth)
    for name, parameters in original["params"].items():
        assert all(jnp.array_equal(a, b) for a, b in zip(
            jax.tree.leaves(parameters), jax.tree.leaves(variables["params"][name]), strict=True,
        ))
    variables = _activate_residual(variables)
    gradient = jax.grad(lambda params: jnp.mean(cast(jax.Array, model.apply(
        {"params": params}, inputs, depth,
    )) ** 2))(variables["params"])
    for name in ("eta_context_in", "eta_context_hidden", "eta_context_out", "eta_context_expand"):
        assert jnp.linalg.norm(gradient[name]["kernel"]) > 1e-15
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient))


def test_bounded_context_gate_reuses_context_and_preserves_initialization() -> None:
    eta, xi, depth = _state()
    inputs = jnp.stack((eta, xi), axis=-1)
    original = _model(surface_context_width=8)
    gated = original.clone(surface_context_gate=True)
    variables = original.init(jax.random.PRNGKey(18), inputs, depth)
    gated_variables = gated.init(jax.random.PRNGKey(18), inputs, depth)
    assert jax.tree.structure(variables) == jax.tree.structure(gated_variables)
    assert all(jnp.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(variables), jax.tree.leaves(gated_variables), strict=True,
    ))
    assert jnp.array_equal(
        cast(jax.Array, original.apply(variables, inputs, depth)),
        cast(jax.Array, gated.apply(variables, inputs, depth)),
    )
    variables = _activate_residual(variables)
    output, captured = gated.apply(
        variables, inputs, depth,
        capture_intermediates=lambda module, _: module.name in (
            "eta_feat_mix", "eta_context_expand", "phi_proj",
        ),
    )
    context = cast(jax.Array, captured["intermediates"]["eta_context_expand"]["__call__"][0])
    z = nn.gelu(cast(jax.Array, captured["intermediates"]["eta_feat_mix"]["__call__"][0]))
    features = z * jnp.tanh(z) * (1 + jnp.tanh(context))
    for index in range(gated.n_blocks):
        block = f"cs_block_{index}"
        expected = features @ cast(jax.Array, variables["params"][block]["phi_proj"]["kernel"])
        actual = cast(jax.Array, captured["intermediates"][block]["phi_proj"]["__call__"][0])
        assert jnp.array_equal(actual, expected)
    assert jnp.linalg.norm(output - original.apply(variables, inputs, depth)) > 1e-10
    translated = cast(jax.Array, gated.apply(variables, jnp.roll(inputs, 7, axis=1), depth))
    assert jnp.allclose(translated, jnp.roll(output, 7, axis=1), rtol=1e-10, atol=1e-13)
    gradient = jax.grad(lambda params: jnp.mean(cast(jax.Array, gated.apply(
        {"params": params}, inputs, depth,
    )) ** 2))(variables["params"])
    for name in ("eta_context_in", "eta_context_hidden", "eta_context_out", "eta_context_expand"):
        assert jnp.linalg.norm(gradient[name]["kernel"]) > 1e-15
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient))


def test_polynomial_skip_formula_initialization_and_gradients() -> None:
    eta, xi, depth = _state()
    inputs = jnp.stack((eta, xi), axis=-1)
    for scales in (None, (0.2, 0.04, 0.008, 2., 8., 0.7, 0.3)):
        original = _model(surface_context_width=8, surface_context_gate=True,
                          surface_feature_scales=scales)
        model = original.clone(surface_polynomial_skip=True)
        original_variables = original.init(jax.random.PRNGKey(21), inputs, depth)
        variables = model.init(jax.random.PRNGKey(21), inputs, depth)
        for name, parameters in original_variables["params"].items():
            if name.startswith("cs_block_"):
                assert cast(jax.Array, variables["params"][name]["phi_proj"]["kernel"]).shape == (18, model.latent)
                assert jnp.all(variables["params"][name]["phi_proj"]["kernel"] == 0)
                parameters = {"m_shared": parameters["m_shared"]}
            for key, value in parameters.items():
                assert all(jnp.array_equal(a, b) for a, b in zip(
                    jax.tree.leaves(value), jax.tree.leaves(variables["params"][name][key]), strict=True,
                ))
        assert jnp.array_equal(cast(jax.Array, model.apply(variables, inputs, depth)),
                               cast(jax.Array, original.apply(original_variables, inputs, depth)))
        variables = _activate_residual(variables)
        _, captured = model.apply(
            variables, inputs, depth,
            capture_intermediates=lambda module, _: module.name in (
                "eta_feat_mix", "eta_context_expand", "phi_proj",
            ),
        )
        context = cast(jax.Array, captured["intermediates"]["eta_context_expand"]["__call__"][0])
        z = nn.gelu(cast(jax.Array, captured["intermediates"]["eta_feat_mix"]["__call__"][0]))
        polynomial_features = jnp.stack((eta**2, eta**3), axis=-1)
        if scales is not None:
            polynomial_features /= jnp.asarray(scales[1:3])
        features = jnp.concatenate((z * jnp.tanh(z) * (1 + jnp.tanh(context)), polynomial_features), axis=-1)
        for index in range(model.n_blocks):
            block = f"cs_block_{index}"
            expected = features @ cast(jax.Array, variables["params"][block]["phi_proj"]["kernel"])
            actual = cast(jax.Array, captured["intermediates"][block]["phi_proj"]["__call__"][0])
            assert jnp.array_equal(actual, expected)
        gradient = jax.grad(lambda params: jnp.mean(cast(jax.Array, model.apply(
            {"params": params}, inputs, depth,
        )) ** 2))(variables["params"])
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient))
        for index in range(model.n_blocks):
            assert jnp.all(jnp.linalg.norm(gradient[f"cs_block_{index}"]["phi_proj"]["kernel"][-2:], axis=-1) > 0)
        psi = jnp.roll(xi, 9, axis=-1)
        actual = _learned_residual(model, variables, eta, xi, depth)
        other = _learned_residual(model, variables, eta, psi, depth)
        mixed = _learned_residual(model, variables, eta, 0.3 * xi - 0.7 * psi, depth)
        assert jnp.allclose(jnp.vdot(psi, actual), jnp.vdot(other, xi), rtol=1e-11, atol=1e-14)
        assert jnp.allclose(mixed, 0.3 * actual - 0.7 * other, rtol=1e-10, atol=1e-14)


def test_even_surface_features_enforce_reflection_covariance() -> None:
    eta, xi, depth = _state()
    inputs = jnp.stack((eta, xi), axis=-1)
    reflection = (-jnp.arange(eta.shape[-1])) % eta.shape[-1]
    for polynomial_skip in (False, True):
        model = _model(use_first_deriv=False, use_hilbert=False,
                       surface_feature_scales=(0.2, 0.04, 0.008, 8., 0.7),
                       surface_context_width=8, surface_context_gate=True,
                       surface_polynomial_skip=polynomial_skip)
        variables = _activate_residual(model.init(jax.random.PRNGKey(22), inputs, depth))
        output = cast(jax.Array, model.apply(variables, inputs, depth))
        reflected = cast(jax.Array, model.apply(variables, inputs[:, reflection], depth))
        assert jnp.allclose(reflected, output[:, reflection], rtol=1e-10, atol=1e-13)
        assert jnp.linalg.norm(_learned_residual(model, variables, eta, xi, depth)) > 0


def test_nested_branch_formula_and_gradients() -> None:
    eta, xi, depth = _state()
    features = jnp.stack((eta, eta**2), axis=-1)
    block = CraigSulemBlock(4, 2 * pi, 5.0, 16, nested=True)
    variables = unfreeze(block.init(jax.random.PRNGKey(19), features, xi, depth, eta_norm=eta))
    assert set(variables["params"]) == {"m_shared", "m_inner", "gamma"}
    assert jnp.all(block.apply(variables, features, xi, depth, eta_norm=eta) == 0)
    variables["params"]["gamma"] = jnp.asarray([0.1, -0.2, 0.3, -0.4], dtype=jnp.float32)
    multiplier = DepthAwareMultiplier(4, 2 * pi, 5.0, 16)
    outer, inner = (cast(jax.Array, multiplier.apply(
        {"params": variables["params"][name]}, depth, eta.shape[-1] // 2 + 1,
    )) for name in ("m_shared", "m_inner"))
    field = jnp.fft.irfft(outer * jnp.fft.rfft(xi)[..., None], n=eta.shape[-1], axis=1)
    field = jnp.fft.irfft(inner * jnp.fft.rfft(eta[..., None] * field, axis=1), n=eta.shape[-1], axis=1)
    expected = jnp.fft.irfft(jnp.sum(
        outer * jnp.fft.rfft(variables["params"]["gamma"] * eta[..., None] * field, axis=1), axis=-1,
    ), n=eta.shape[-1])
    actual = cast(jax.Array, block.apply(variables, features, xi, depth, eta_norm=eta))
    assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-15)
    gradient = jax.grad(lambda params: jnp.mean(cast(jax.Array, block.apply(
        {"params": params}, features, xi, depth, eta_norm=eta,
    )) ** 2))(variables["params"])
    assert jnp.linalg.norm(gradient["gamma"]) > 0
    for name in ("m_shared", "m_inner"):
        for layer in ("hidden", "out"):
            assert jnp.linalg.norm(gradient[name][layer]["kernel"]) > 0
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient))


def test_nested_model_replacement_preserves_structure() -> None:
    eta, xi, depth = _state()
    inputs = jnp.stack((eta, xi), axis=-1)
    scales = (0.2, 0.04, 0.008, 2., 8., 0.7, 0.3)
    original = _model(surface_context_width=8, surface_feature_scales=scales)
    original_variables = original.init(jax.random.PRNGKey(20), inputs, depth)
    for nested_blocks in (1, 2):
        model = original.clone(nested_blocks=nested_blocks)
        variables = model.init(jax.random.PRNGKey(20), inputs, depth)
        for name, parameters in original_variables["params"].items():
            if name.startswith("cs_block_") and int(name.rsplit("_", 1)[-1]) >= model.n_blocks - nested_blocks:
                parameters = {"m_shared": parameters["m_shared"]}
                assert set(variables["params"][name]) == {"m_shared", "m_inner", "gamma"}
            for key, value in parameters.items():
                assert all(jnp.array_equal(a, b) for a, b in zip(
                    jax.tree.leaves(value), jax.tree.leaves(variables["params"][name][key]), strict=True,
                ))
        assert jnp.array_equal(
            cast(jax.Array, model.apply(variables, inputs, depth)),
            cast(jax.Array, original.apply(original_variables, inputs, depth)),
        )
        variables = _activate_residual(variables)
        psi = jnp.roll(xi, 9, axis=-1)
        actual = _learned_residual(model, variables, eta, xi, depth)
        if nested_blocks == model.n_blocks:
            unscaled = _learned_residual(model.clone(surface_feature_scales=None), variables, eta, xi, depth)
            assert jnp.allclose(actual, unscaled / scales[0]**2, rtol=1e-10, atol=1e-14)
        other = _learned_residual(model, variables, eta, psi, depth)
        assert jnp.allclose(jnp.vdot(psi, actual), jnp.vdot(other, xi), rtol=1e-11, atol=1e-14)
        mixed = _learned_residual(model, variables, eta, 0.3 * xi - 0.7 * psi, depth)
        assert jnp.allclose(mixed, 0.3 * actual - 0.7 * other, rtol=1e-10, atol=1e-14)
        shifted = _learned_residual(model, variables, jnp.roll(eta, 7, axis=-1), jnp.roll(xi, 7, axis=-1), depth)
        assert jnp.allclose(shifted, jnp.roll(actual, 7, axis=-1), rtol=1e-10, atol=1e-14)


def test_multiplier_skip_symbol_basis_and_grid_independence() -> None:
    depth = jnp.log(jnp.asarray([[0.08], [8.0]], dtype=jnp.float64))
    for length in (2 * pi, 8 * pi):
        model = DepthAwareMultiplier(3, length, 5.0, multiplier_skip=True)
        original = model.clone(multiplier_skip=False)
        variables = unfreeze(model.init(jax.random.PRNGKey(23), depth, 257))
        original_variables = original.init(jax.random.PRNGKey(23), depth, 257)
        assert jnp.array_equal(cast(jax.Array, model.apply(variables, depth, 257)),
                               cast(jax.Array, original.apply(original_variables, depth, 257)))
        variables["params"]["out"] = jax.tree.map(jnp.zeros_like, variables["params"]["out"])
        variables["params"]["symbol_skip"]["kernel"] = jnp.eye(3)
        actual = cast(jax.Array, model.apply(variables, depth, 257))
        k = (2 * pi / length) * jnp.arange(257, dtype=jnp.float64)
        k_ref = 2 * pi * 128 / length
        k, h = jnp.broadcast_arrays(k[None, :], jnp.exp(jnp.minimum(depth, jnp.log(5.0))))
        expected = jnp.stack((jnp.ones_like(k), k * jnp.tanh(k * h) / k_ref,
                              (k / k_ref) ** 2), axis=-1)
        assert jnp.allclose(actual, expected, rtol=1e-14, atol=1e-14)
        assert jnp.array_equal(actual[:, :33], cast(jax.Array, model.apply(variables, depth, 33)))
        assert jnp.allclose(actual[:, 256, 2] / actual[:, 128, 2], 4.0)
        assert jnp.allclose(actual[:, 128, 2], 1.0)


def test_multiplier_skip_preserves_initialization_and_operator_structure() -> None:
    eta, xi, depth = _state()
    inputs = jnp.stack((eta, xi), axis=-1)
    for nested_blocks in (0, 1, 2):
        original = _model(nested_blocks=nested_blocks)
        model = original.clone(multiplier_skip=True)
        original_variables = original.init(jax.random.PRNGKey(24), inputs, depth)
        variables = model.init(jax.random.PRNGKey(24), inputs, depth)
        stripped = unfreeze(variables)
        for block_idx in range(model.n_blocks):
            block = stripped["params"][f"cs_block_{block_idx}"]
            for name in ("m_shared", "m_inner"):
                if name in block:
                    assert jnp.all(block[name].pop("symbol_skip")["kernel"] == 0)
        assert all(jnp.array_equal(a, b) for a, b in zip(
            jax.tree.leaves(original_variables), jax.tree.leaves(stripped), strict=True,
        ))
        variables = unfreeze(_activate_residual(variables))
        assert jnp.array_equal(cast(jax.Array, model.apply(variables, inputs, depth)),
                               cast(jax.Array, original.apply(_activate_residual(original_variables), inputs, depth)))
        gradient = jax.grad(lambda params: jnp.mean(cast(jax.Array, model.apply(
            {"params": params}, inputs, depth,
        )) ** 2))(variables["params"])
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient))
        for block_idx in range(model.n_blocks):
            block_name = f"cs_block_{block_idx}"
            for name in ("m_shared", "m_inner"):
                if name in variables["params"][block_name]:
                    assert jnp.linalg.norm(gradient[block_name][name]["symbol_skip"]["kernel"]) > 0
                    kernel = variables["params"][block_name][name]["symbol_skip"]["kernel"]
                    variables["params"][block_name][name]["symbol_skip"]["kernel"] = jnp.broadcast_to(
                        jnp.asarray([0.1, -0.15, 0.05], dtype=kernel.dtype)[:, None], kernel.shape,
                    )
        psi = jnp.roll(xi, 9, axis=-1)
        actual = _learned_residual(model, variables, eta, xi, depth)
        other = _learned_residual(model, variables, eta, psi, depth)
        mixed = _learned_residual(model, variables, eta, 0.3 * xi - 0.7 * psi, depth)
        assert jnp.allclose(jnp.vdot(psi, actual), jnp.vdot(other, xi), rtol=1e-11, atol=1e-14)
        assert jnp.allclose(mixed, 0.3 * actual - 0.7 * other, rtol=1e-10, atol=1e-14)


def test_log_multiplier_inputs_formula_initialization_and_gradients() -> None:
    depth = jnp.log(jnp.asarray([[0.02], [0.8], [5.0], [8.0]], dtype=jnp.float64))
    original = DepthAwareMultiplier(3, 2 * pi, 5.0, 16)
    variables = original.init(jax.random.PRNGKey(25), depth, 129)
    model = original.clone(log_multiplier_inputs=True)
    log_variables = model.init(jax.random.PRNGKey(25), depth, 129)
    assert jax.tree.structure(variables) == jax.tree.structure(log_variables)
    assert all(jnp.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(variables), jax.tree.leaves(log_variables), strict=True,
    ))
    k, h = jnp.broadcast_arrays(jnp.arange(129)[None, :], jnp.exp(jnp.minimum(depth, jnp.log(5.0))))
    hidden, output = (cast(dict[str, jax.Array], variables["params"][name]) for name in ("hidden", "out"))
    for enabled in (False, True):
        current = original.clone(log_multiplier_inputs=enabled)
        features = (jnp.stack((jnp.log1p(k), jnp.log(h), jnp.tanh(k * h),
                               jnp.log1p(k * jnp.tanh(k * h))), axis=-1) if enabled
                    else jnp.stack((k, h, jnp.tanh(k * h), k * jnp.tanh(k * h)), axis=-1))
        expected = nn.tanh(features @ hidden["kernel"] + hidden["bias"]) @ output["kernel"] + output["bias"]
        actual = cast(jax.Array, current.apply(variables, depth, 129))
        assert jnp.allclose(actual, expected, rtol=1e-13, atol=1e-14)
        assert jnp.array_equal(actual[2], actual[3])
        if not enabled:
            assert jnp.array_equal(actual, cast(jax.Array, original.apply(variables, depth, 129)))
        for n_freq in (1, 129):
            gradients = jax.grad(lambda params, values: jnp.sum(cast(jax.Array, current.apply(
                {"params": params}, values, n_freq,
            )) ** 2), argnums=(0, 1))(variables["params"], depth)
            assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradients))
            assert jnp.all(gradients[1][-1] == 0)
            assert jnp.linalg.norm(gradients[1][:-1]) > 0


def test_log_multiplier_inputs_preserve_zero_head_prediction_and_parameters() -> None:
    eta, xi, depth = _state()
    inputs = jnp.stack((eta, xi), axis=-1)
    for nested_blocks in (0, 1, 2):
        original = _model(nested_blocks=nested_blocks, surface_context_width=4)
        model = original.clone(log_multiplier_inputs=True)
        variables = original.init(jax.random.PRNGKey(26), inputs, depth)
        log_variables = model.init(jax.random.PRNGKey(26), inputs, depth)
        assert jax.tree.structure(variables) == jax.tree.structure(log_variables)
        assert all(jnp.array_equal(a, b) for a, b in zip(
            jax.tree.leaves(variables), jax.tree.leaves(log_variables), strict=True,
        ))
        assert jnp.array_equal(cast(jax.Array, model.apply(log_variables, inputs, depth)),
                               cast(jax.Array, original.apply(variables, inputs, depth)))
        activated = _activate_residual(variables)
        assert not jnp.allclose(_learned_residual(model, activated, eta, xi, depth),
                               _learned_residual(original, activated, eta, xi, depth), rtol=1e-3, atol=1e-12)


def main() -> int:
    tests: tuple[Callable[[], None], ...] = (
        test_default_matches_original_forward,
        test_order_two_has_zero_value_and_first_variation,
        test_order_two_is_quadratic_near_zero,
        test_learned_operator_is_self_adjoint_and_linear,
        test_baseline_free_flat_surface_is_trainable,
        test_g0_only_preserves_flat_surface_and_allows_first_order_correction,
        test_eta_feature_configuration_controls_trunk_shape,
        test_relative_elevation_uses_physical_eta_and_unclipped_depth,
        test_m2_zero_init_matches_classical_batched_fp64,
        test_learned_eta_filters_preserve_initial_features_and_shared_parameters,
        test_learned_eta_filters_change_spectral_shape_and_condition_on_depth,
        test_true_depth_analytic_baselines_and_returned_components,
        test_experimental_residual_orders,
        test_surface_feature_scaling_folds_into_first_projection,
        test_surface_context_is_nonlocal_and_depth_conditioned,
        test_surface_context_has_live_gradients_and_unchanged_shared_initialization,
        test_bounded_context_gate_reuses_context_and_preserves_initialization,
        test_polynomial_skip_formula_initialization_and_gradients,
        test_even_surface_features_enforce_reflection_covariance,
        test_nested_branch_formula_and_gradients,
        test_nested_model_replacement_preserves_structure,
        test_multiplier_skip_symbol_basis_and_grid_independence,
        test_multiplier_skip_preserves_initialization_and_operator_structure,
        test_log_multiplier_inputs_formula_initialization_and_gradients,
        test_log_multiplier_inputs_preserve_zero_head_prediction_and_parameters,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
