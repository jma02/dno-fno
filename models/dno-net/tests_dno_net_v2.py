"""Focused CPU invariants for the Craig--Sulem DNO architecture.

Run with::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=True .venv/bin/python \
        models/dno-net/tests_dno_net_v2.py
"""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import partial
from math import pi
from typing import Any, cast
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
from flax.typing import FrozenVariableDict

from dno_net_v2 import (
    CraigSulemBlock, CraigSulemDNO, CraigSulemTermsResidual, DepthAwareMultiplier,
    ParityDense, PhaseWavenumberMultiplier, mixed_surface_lift, parity_dense, parity_gelu,
    phase_rank_one_hat, phase_wavenumber_hat, surface_amplitude,
    surface_parity_products,
)
from solver.solvers.dno_series_jax import dno_series_eval

jax.config.update("jax_enable_x64", True)


VariableState = FrozenVariableDict | dict[str, Any]


def _model(
    *,
    baseline_order: int = 1,
    n_polys: int = 3,
    use_first_deriv: bool = True,
    use_second_deriv: bool = True,
    curvature_cutoff: float = 0.0,
    use_half_deriv: bool = True,
    use_hilbert: bool = True,
    use_relative_elevation: bool = False,
    use_learned_eta_filters: bool = False,
    surface_feature_scales: tuple[float, ...] | None = None,
    surface_feature_transform: tuple[tuple[float, ...], ...] | None = None,
    surface_context_width: int = 0,
    surface_context_gate: bool = False,
    surface_polynomial_skip: bool = False,
    surface_cubic_channels: int = 0,
    surface_parity_products: bool = False,
    surface_parity_streams: bool = False,
    surface_depth_gate: bool = False,
    nested_blocks: int = 0,
    nested_surface_modulation: bool = False,
    nested_surface_features: bool = False,
    multiplier_skip: bool = False,
) -> CraigSulemDNO:
    return CraigSulemDNO(
        width=32,
        n_blocks=2,
        latent=8,
        n_polys=n_polys,
        use_first_deriv=use_first_deriv,
        use_second_deriv=use_second_deriv,
        curvature_cutoff=curvature_cutoff,
        use_half_deriv=use_half_deriv,
        use_hilbert=use_hilbert,
        use_relative_elevation=use_relative_elevation,
        use_learned_eta_filters=use_learned_eta_filters,
        surface_feature_scales=surface_feature_scales,
        surface_feature_transform=surface_feature_transform,
        surface_context_width=surface_context_width,
        surface_context_gate=surface_context_gate,
        surface_polynomial_skip=surface_polynomial_skip,
        surface_cubic_channels=surface_cubic_channels,
        surface_parity_products=surface_parity_products,
        surface_parity_streams=surface_parity_streams,
        surface_depth_gate=surface_depth_gate,
        nested_blocks=nested_blocks,
        nested_surface_modulation=nested_surface_modulation,
        nested_surface_features=nested_surface_features,
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
    if "eta_depth_gate" in mutable["params"]:
        kernel = mutable["params"]["eta_depth_gate"]["kernel"]
        mutable["params"]["eta_depth_gate"]["kernel"] = 0.2 * jax.random.normal(
            jax.random.PRNGKey(103), kernel.shape, dtype=kernel.dtype,
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
    _, _, _, correction = cast(tuple[jax.Array, jax.Array, jax.Array, jax.Array], model.apply(
        variables, inputs, depth, return_baseline=True,
    ))
    return correction[..., 0]


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


def test_mixed_surface_lift_structure_and_orders() -> None:
    eta, xi, depth = _state()
    model = _model(surface_cubic_channels=4, surface_context_width=8, nested_blocks=1)
    inputs = jnp.stack((eta, xi), axis=-1)
    initial = model.init(jax.random.PRNGKey(83), inputs, depth)
    control = model.clone(surface_cubic_channels=0)
    original = control.init(jax.random.PRNGKey(83), inputs, depth)
    assert jax.tree.structure(initial) == jax.tree.structure(original)
    assert all(jnp.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(initial), jax.tree.leaves(original), strict=True,
    ))
    assert jnp.array_equal(cast(jax.Array, model.apply(initial, inputs, depth)),
                           cast(jax.Array, control.apply(original, inputs, depth)))
    features = jnp.linspace(-1.0, 1.0, 48).reshape(1, 3, 16)
    lifted = mixed_surface_lift(features, 4)
    assert jnp.array_equal(lifted[..., :12], features[..., :12] * jnp.tanh(features[..., :12]))
    assert jnp.allclose(lifted[..., 12:], features[..., 12:] * jnp.tanh(features[..., 12:])**2,
                        rtol=2e-16, atol=1e-17)
    variables = jax.tree.map(lambda a: a.astype(jnp.float64), _activate_residual(initial))
    q = _learned_residual(model, variables, eta, xi, depth)
    other_xi = jnp.roll(xi, 7, axis=-1)
    other_q = _learned_residual(model, variables, eta, other_xi, depth)
    assert jnp.allclose(_learned_residual(model, variables, eta, .3*xi-.7*other_xi, depth),
                        .3*q-.7*other_q, rtol=1e-10, atol=1e-14)
    assert jnp.allclose(jnp.vdot(other_xi, q), jnp.vdot(xi, other_q), rtol=1e-10, atol=1e-14)
    assert jnp.allclose(_learned_residual(
        model, variables, jnp.roll(eta, 9, axis=-1), jnp.roll(xi, 9, axis=-1), depth,
    ), jnp.roll(q, 9, axis=-1), rtol=1e-10, atol=1e-14)
    zero, derivative = jax.jvp(
        lambda e: _learned_residual(model, variables, e, xi, depth), (jnp.zeros_like(eta),), (eta,),
    )
    assert jnp.all(zero == 0) and jnp.all(derivative == 0)
    for cubic, expected in ((False, 4), (True, 8)):
        isolated = unfreeze(variables)
        isolated["params"]["cs_block_1"]["gamma"] = jnp.zeros_like(
            isolated["params"]["cs_block_1"]["gamma"],
        )
        kernel = isolated["params"]["cs_block_0"]["phi_proj"]["kernel"]
        mask = jnp.arange(16) >= 12
        isolated["params"]["cs_block_0"]["phi_proj"]["kernel"] = kernel * (
            mask if cubic else ~mask
        )[:, None]
        norms = [jnp.linalg.norm(_learned_residual(model, isolated, scale*eta, xi, depth))
                 for scale in (1/256, 1/512, 1/1024)]
        assert all(abs(float(a/b)/expected-1) < .01
                   for a, b in zip(norms[:-1], norms[1:], strict=True))


def test_mixed_surface_lift_fourier_inference() -> None:
    from scripts import benchmark_dno_fusion as fusion
    from solver.evals.model_rollout import LoadedRun

    eta, xi, _ = _state()
    eta = jnp.concatenate((eta, jnp.roll(eta, 7, axis=-1))).astype(jnp.float32)
    xi = jnp.concatenate((xi, .7*jnp.roll(xi, 3, axis=-1))).astype(jnp.float32)
    depth = jnp.log(jnp.asarray([[.8], [.04]], dtype=jnp.float32))
    model = _model(surface_cubic_channels=4, surface_context_width=8, nested_blocks=1,
                   surface_feature_scales=(.3, .1, .04, .7, 2., .5, .3))
    variables = _activate_residual(model.init(jax.random.PRNGKey(84), jnp.stack((eta, xi), axis=-1), depth))
    loaded = LoadedRun(model, unfreeze(variables)["params"], {},
                       {"feature_absmax": [1., 1.], "target_absmax": 1.}, "scale", 0)
    with patch.object(fusion, "sandwich", lambda surface, potential, symbols, **_: (
        symbols * jnp.fft.rfft(surface[:, None, :] * jnp.fft.irfft(symbols * potential[:, None, :], n=64))
    )):
        for batch in (1, 2):
            predict = fusion.build_variant(loaded, depth[:batch], 64, "fourier_round", return_components=True)
            correction_hat = predict(jnp.fft.rfft(eta[:batch]), jnp.fft.rfft(xi[:batch]))[1]
            native_hat = jnp.fft.rfft(_learned_residual(model, variables, eta[:batch], xi[:batch], depth[:batch]))
            native_hat = native_hat.at[..., 0].set(0)
            assert float(jnp.linalg.norm(correction_hat-native_hat)/jnp.linalg.norm(native_hat)) < 3e-6


def test_reflection_encoder_matches_full_prediction_average() -> None:
    """Average nonzero geometry heads, not two complete DNO evaluations."""
    from scripts import benchmark_dno_fusion as fusion
    from solver.evals.model_rollout import LoadedRun

    nx = 64
    eta, xi, _ = _state(nx)
    eta = jnp.concatenate((eta, jnp.roll(eta, 7, axis=-1)), axis=0)
    xi = jnp.concatenate((xi, 0.6 * jnp.roll(xi, 3, axis=-1)), axis=0)
    reflection = (-jnp.arange(nx)) % nx
    for batch in (1, 2):
        model = _model(
            nested_blocks=1, surface_context_width=8,
            surface_context_gate=batch == 2, surface_polynomial_skip=batch == 2,
            surface_feature_scales=(0.3, 0.1, 0.04, 0.7, 2.0, 0.5, 0.3),
            surface_feature_transform=(tuple(tuple(float(i == j) + 0.03 for j in range(7))
                                             for i in range(7)) if batch == 2 else None),
        )
        depth = jnp.log(jnp.asarray([[0.8], [0.04]], dtype=jnp.float32)[:batch])
        variables = _activate_residual(model.init(
            jax.random.PRNGKey(701), jnp.stack((eta[:batch], xi[:batch]), axis=-1).astype(jnp.float32), depth,
        ))
        loaded = LoadedRun(
            model, unfreeze(variables)["params"], {},
            {"feature_absmax": [1.0, 1.0], "target_absmax": 1.0}, "scale", 0,
        )
        # Only the G1 CUDA sandwich is substituted; all geometry, nested branches,
        # cached filters, reflection averaging, and output assembly are production.
        with patch.object(fusion, "sandwich", lambda surface, potential, symbols, **_: (
            symbols * jnp.fft.rfft(
                surface[:, None, :] * jnp.fft.irfft(symbols * potential[:, None, :], n=nx),
            )
        )):
            ordinary = fusion.build_variant(loaded, depth, nx, "fourier_round", return_components=True)
            disabled = fusion.build_variant(
                loaded, depth, nx, "fourier_round", return_components=True, reflection_encoder=False,
            )
            averaged = fusion.build_variant(
                loaded, depth, nx, "fourier_round", return_components=True, reflection_encoder=True,
            )
            eta_hat, xi_hat = jnp.fft.rfft(eta[:batch]), jnp.fft.rfft(xi[:batch])
            raw = ordinary(eta_hat, xi_hat)
            for actual, expected in zip(disabled(eta_hat, xi_hat), raw):
                assert jnp.array_equal(actual, expected)
            reflected = ordinary(jnp.conj(eta_hat), jnp.conj(xi_hat))
            predicted = averaged(eta_hat, xi_hat)
            assert float(jnp.linalg.norm(raw[1])) > 1e-6
            for actual, first, second in zip(predicted, raw, reflected):
                expected = 0.5 * (first + jnp.conj(second))
                assert float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)) < 3e-6
            for index, (actual, expected) in enumerate(zip(
                averaged(jnp.conj(eta_hat), jnp.conj(xi_hat)), predicted,
            )):
                relative = float(jnp.linalg.norm(actual - jnp.conj(expected)) / jnp.linalg.norm(expected))
                # Two independently rounded FP32 encoder evaluations contribute
                # to covariance; the direct-average check above keeps its 3e-6 budget.
                assert relative < 6e-6, (batch, index, relative)
            other_hat = jnp.fft.rfft(jnp.roll(xi[:batch], 11, axis=-1))
            other = averaged(eta_hat, other_hat)
            mixed = averaged(eta_hat, 0.3 * xi_hat - 0.7 * other_hat)
            for actual, first, second in zip(mixed, predicted, other):
                expected = 0.3 * first - 0.7 * second
                assert float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)) < 3e-6
            # Confirm that the Fourier conjugation above is x_j -> x_{-j}, not reversal.
            assert jnp.allclose(jnp.fft.irfft(jnp.conj(eta_hat), n=nx), eta[:batch, reflection], atol=1e-14)


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
    for baseline_order, relative_elevation, learned_filters, context_width, context_gate, cutoff in (
        (-1, False, False, 0, False, 0.), (0, False, False, 0, False, 0.), (1, False, False, 0, False, 0.),
        (1, True, False, 0, False, 0.), (2, False, False, 0, False, 0.), (1, False, True, 0, False, 0.),
        (1, False, False, 8, False, 0.), (1, False, False, 8, True, 0.),
        (1, False, False, 8, False, 22.727203594311888),
    ):
        model = _model(baseline_order=baseline_order, use_relative_elevation=relative_elevation,
                       use_learned_eta_filters=learned_filters, surface_context_width=context_width,
                       surface_context_gate=context_gate, curvature_cutoff=cutoff)
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
            prediction, baseline, m1, correction = cast(tuple[jax.Array, jax.Array, jax.Array, jax.Array], model.apply(
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
            assert jnp.array_equal(correction, jnp.zeros_like(correction))
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


def test_cs_term_fields_reconstruct_and_differentiate() -> None:
    eta, xi, _ = _state()
    eta, xi = (jnp.broadcast_to(value, (4, value.shape[-1])) for value in (eta, xi))
    depth = jnp.asarray([[0.02], [0.2], [2.0], [12.0]])
    k = jnp.arange(eta.shape[-1] // 2 + 1, dtype=jnp.float64)
    term_fn = partial(dno_series_eval, k=k, depth=depth, order=3, pad_factor=2, return_terms=True)
    terms = term_fn(eta, xi)
    assert terms.shape == (4, 4, eta.shape[-1])
    assert jnp.all(jnp.isfinite(terms))
    for order in range(4):
        expected = dno_series_eval(eta, xi, k, depth, order=order, pad_factor=2)
        assert jnp.allclose(terms[:, :order + 1].sum(axis=-2), expected, rtol=1e-10, atol=1e-12)
    scaled = term_fn(0.37 * eta, xi)
    assert jnp.allclose(scaled, terms * 0.37 ** jnp.arange(4)[None, :, None], rtol=1e-8, atol=1e-12)
    direction = jnp.roll(xi, 3, axis=-1)
    expected_direction = term_fn(eta, direction)
    _, tangent = jax.jvp(term_fn, (eta, xi), (jnp.zeros_like(eta), direction))
    assert jnp.allclose(tangent, expected_direction, rtol=1e-8, atol=1e-12)
    assert jnp.allclose(term_fn(eta, 0.3 * xi - 0.7 * direction),
                        0.3 * terms - 0.7 * expected_direction, rtol=1e-8, atol=1e-12)


def test_experimental_residual_orders() -> None:
    eta, xi, depth = _state()
    inputs = jnp.stack((eta, xi), axis=-1)
    for model, order in (
        (_model(use_relative_elevation=True), 2), (_model(baseline_order=2), 3),
        (_model(use_learned_eta_filters=True), 2),
        (_model(surface_context_width=8), 2),
        (_model(surface_context_width=8, surface_feature_scales=(0.2, 0.04, 0.008, 2., 8., 0.7, 0.3)), 2),
        (_model(curvature_cutoff=22.727203594311888, surface_context_width=8,
                surface_feature_scales=(0.2, 0.04, 0.008, 2., 8., 0.7, 0.3)), 2),
        (_model(surface_context_width=8, surface_context_gate=True,
                surface_feature_scales=(0.2, 0.04, 0.008, 2., 8., 0.7, 0.3)), 2),
        (_model(baseline_order=2, surface_context_width=8, surface_context_gate=True), 3),
        (_model(nested_blocks=1, surface_context_width=8), 2),
        (_model(nested_blocks=2), 2),
        (_model(surface_polynomial_skip=True, surface_context_width=8), 2),
        (_model(surface_depth_gate=True, surface_context_width=8), 2),
        (_model(surface_depth_gate=True, surface_context_width=8,
                surface_context_gate=True, surface_polynomial_skip=True), 2),
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
    for context_width, cutoff in ((0, 0.), (8, 0.), (8, 22.727203594311888)):
        normalized = _model(surface_feature_scales=scales, surface_context_width=context_width,
                            curvature_cutoff=cutoff)
        folded = normalized.clone(surface_feature_scales=None)
        variables = _activate_residual(normalized.init(jax.random.PRNGKey(15), inputs, depth))
        variables = jax.tree.map(lambda value: value.astype(eta.dtype), variables)
        folded_variables = unfreeze(variables)
        folded_variables["params"]["eta_feat_proj"]["kernel"] /= jnp.asarray(scales)[:, None]
        expected = cast(jax.Array, normalized.apply(variables, inputs, depth))
        actual = cast(jax.Array, folded.apply(folded_variables, inputs, depth))
        assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-14)
        assert jnp.linalg.norm(_learned_residual(normalized, variables, eta, xi, depth)) > 0


def test_softened_curvature_uses_physical_wavenumbers() -> None:
    """Only curvature changes; its mixed-mode symbol uses physical k, not indices."""
    _, xi, depth = _state(256)
    phase = 2 * jnp.pi * jnp.arange(xi.shape[-1]) / xi.shape[-1]
    eta = (0.01 + 0.02 * jnp.cos(4 * phase) + 0.003 * jnp.cos(96 * phase))[None, :]
    scales = (0.2, 0.04, 0.008, 2., 8., 0.7, 0.3)
    model = _model(curvature_cutoff=22.727203594311888, surface_context_width=8,
                   surface_feature_scales=scales).clone(domain_length=3 * pi)
    original = model.clone(curvature_cutoff=0.)
    inputs = jnp.stack((eta, xi), axis=-1)
    initial = model.init(jax.random.PRNGKey(31), inputs, depth)
    assert jnp.array_equal(model.apply(initial, inputs, depth), original.apply(initial, inputs, depth))
    variables = unfreeze(_activate_residual(initial))
    kernel = variables["params"]["eta_feat_proj"]["kernel"]
    variables["params"]["eta_feat_proj"]["kernel"] = jnp.zeros_like(kernel).at[:, :7].set(jnp.eye(7, dtype=kernel.dtype))
    projections = []
    for current in (model, original):
        _, captured = current.apply(
            variables, inputs, depth,
            capture_intermediates=lambda module, _: module.name == "eta_feat_proj",
        )
        projections.append(cast(jax.Array, captured["intermediates"]["eta_feat_proj"]["__call__"][0]))
    modes = jnp.asarray([4., 96.])
    k = (2 * pi / model.domain_length) * modes
    symbol = -(k**2) / jnp.sqrt(1 + (k / model.curvature_cutoff)**2)
    expected = jnp.sum(symbol[:, None] * jnp.asarray([0.02, 0.003])[:, None]
                       * jnp.cos(modes[:, None] * phase), axis=0) / scales[4]
    assert jnp.allclose(projections[0][..., 4], expected[None, :], rtol=1e-12, atol=1e-12)
    unchanged = jnp.asarray([0, 1, 2, 3, 5, 6])
    assert jnp.array_equal(projections[0][..., unchanged], projections[1][..., unchanged])
    assert jnp.linalg.norm(_learned_residual(model, variables, eta, xi, depth)) > 0
    actual = cast(jax.Array, model.apply(variables, inputs, depth))
    shifted = cast(jax.Array, model.apply(variables, jnp.roll(inputs, 11, axis=1), depth))
    assert jnp.all(jnp.isfinite(actual))
    assert jnp.allclose(shifted, jnp.roll(actual, 11, axis=1), rtol=1e-11, atol=1e-13)


def test_surface_feature_whitening_folds_and_preserves_operator_structure() -> None:
    eta, xi, depth = _state()
    inputs = jnp.stack((eta, xi), axis=-1)
    transform = jnp.diag(jnp.linspace(0.5, 1.5, 7)) + 0.1
    for polynomial_skip in (False, True):
        model = _model(
            surface_feature_scales=(0.2, 0.04, 0.008, 2., 8., 0.7, 0.3),
            surface_feature_transform=tuple(map(tuple, transform.tolist())),
            surface_context_width=8, surface_polynomial_skip=polynomial_skip,
        )
        folded = model.clone(surface_feature_transform=None)
        initial = model.init(jax.random.PRNGKey(27), inputs, depth)
        original = folded.init(jax.random.PRNGKey(27), inputs, depth)
        assert all(jnp.array_equal(a, b) for a, b in zip(jax.tree.leaves(initial), jax.tree.leaves(original), strict=True))
        variables = jax.tree.map(lambda value: value.astype(eta.dtype), _activate_residual(initial))
        folded_variables = unfreeze(variables)
        folded_variables["params"]["eta_feat_proj"]["kernel"] = (
            transform @ folded_variables["params"]["eta_feat_proj"]["kernel"]
        )

        def residual(surface: jnp.ndarray, potential: jnp.ndarray, log_depth: jnp.ndarray) -> jnp.ndarray:
            return _learned_residual(model, variables, surface, potential, log_depth)

        actual, derivative = jax.jvp(residual, (eta, xi, depth), (eta, xi, jnp.ones_like(depth)))
        expected, folded_derivative = jax.jvp(
            lambda surface, potential, log_depth: _learned_residual(
                folded, folded_variables, surface, potential, log_depth,
            ), (eta, xi, depth), (eta, xi, jnp.ones_like(depth)),
        )
        assert jnp.linalg.norm(actual) > 1e-10
        assert jnp.allclose(actual, expected, rtol=1e-11, atol=1e-14)
        assert jnp.allclose(derivative, folded_derivative, rtol=1e-11, atol=1e-14)
        zero, first = jax.jvp(lambda surface: residual(surface, xi, depth), (jnp.zeros_like(eta),), (eta,))
        assert jnp.max(jnp.abs(zero)) < 1e-14
        assert jnp.max(jnp.abs(first)) < 1e-14
        sizes = [jnp.linalg.norm(residual(epsilon * eta, xi, depth)) for epsilon in (0.01, 0.005)]
        assert abs(float(sizes[0] / sizes[1]) - 4.0) < 0.1
        psi = jnp.roll(xi, 7, axis=-1)
        response = residual(eta, psi, depth)
        assert jnp.allclose(jnp.vdot(psi, actual), jnp.vdot(response, xi), rtol=1e-11, atol=1e-14)
        assert jnp.allclose(residual(eta, 0.3 * xi - 0.7 * psi, depth),
                            0.3 * actual - 0.7 * response, rtol=1e-11, atol=1e-14)
        assert jnp.allclose(residual(jnp.roll(eta, 7, axis=-1), psi, depth),
                            jnp.roll(actual, 7, axis=-1), rtol=1e-11, atol=1e-14)


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


def test_depth_gate_initial_identity_shared_parameters_and_gradient() -> None:
    eta, xi, depth = _state()
    inputs = jnp.stack((eta, xi), axis=-1)
    original = _model(surface_context_width=8).clone(width=320, n_blocks=8, latent=4, mult_hidden=80)
    gated = original.clone(surface_depth_gate=True)
    variables = original.init(jax.random.PRNGKey(28), inputs, depth)
    gated_variables = unfreeze(gated.init(jax.random.PRNGKey(28), inputs, depth))
    kernel = gated_variables["params"].pop("eta_depth_gate")["kernel"]
    assert kernel.shape == (2, 160) and jnp.all(kernel == 0)
    assert all(jnp.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(variables), jax.tree.leaves(gated_variables), strict=True,
    ))
    for active in (False, True):
        original_variables = _activate_residual(variables) if active else variables
        gated_variables = unfreeze(original_variables)
        gated_variables["params"]["eta_depth_gate"] = {"kernel": kernel}
        assert jnp.array_equal(cast(jax.Array, original.apply(original_variables, inputs, depth)),
                               cast(jax.Array, gated.apply(gated_variables, inputs, depth)))
    gradients = jax.grad(lambda params, fields, log_depth: jnp.sum(cast(jax.Array, gated.apply(
        {"params": params}, fields, log_depth,
    )) ** 2), argnums=(0, 1, 2))(gated_variables["params"], inputs, depth)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradients))
    assert jnp.linalg.norm(gradients[0]["eta_depth_gate"]["kernel"]) > 0


def test_depth_gate_formula_clipping_and_feature_order() -> None:
    eta, xi, _ = _state()
    inputs = jnp.broadcast_to(jnp.stack((eta, xi), axis=-1), (4, eta.shape[-1], 2))
    depth = jnp.log(jnp.asarray([[0.08], [0.8], [5.0], [12.0]], dtype=eta.dtype))
    for polynomial_skip in (False, True):
        model = _model(surface_context_width=8, surface_context_gate=True,
                       surface_polynomial_skip=polynomial_skip, surface_depth_gate=True)
        variables = _activate_residual(model.init(jax.random.PRNGKey(29), inputs, depth))
        output, captured = model.apply(
            variables, inputs, depth,
            capture_intermediates=lambda module, _: module.name in (
                "eta_feat_mix", "eta_context_expand", "eta_depth_gate", "phi_proj",
            ),
        )
        context = cast(jax.Array, captured["intermediates"]["eta_context_expand"]["__call__"][0])
        z = nn.gelu(cast(jax.Array, captured["intermediates"]["eta_feat_mix"]["__call__"][0]))
        features = z * jnp.tanh(z) * (1 + jnp.tanh(context))
        if polynomial_skip:
            features = jnp.concatenate((features, inputs[..., :1] ** 2, inputs[..., :1] ** 3), axis=-1)
        t = jnp.minimum(depth, jnp.log(5.0)) / 4.0
        gate_logits = jnp.concatenate((t, t**2), axis=-1) @ cast(jax.Array, variables["params"]["eta_depth_gate"]["kernel"])
        assert jnp.array_equal(gate_logits, cast(jax.Array, captured["intermediates"]["eta_depth_gate"]["__call__"][0]))
        features *= (1 + 0.5 * jnp.tanh(gate_logits))[:, None, :]
        for index in range(model.n_blocks):
            block = f"cs_block_{index}"
            expected = features @ cast(jax.Array, variables["params"][block]["phi_proj"]["kernel"])
            assert jnp.array_equal(expected, cast(jax.Array, captured["intermediates"][block]["phi_proj"]["__call__"][0]))
        assert jnp.array_equal(output[2], output[3])
        derivative = jax.grad(lambda log_depth: jnp.sum(cast(jax.Array, model.apply(
            variables, inputs, log_depth,
        )) ** 2))(depth)
        assert jnp.all(jnp.isfinite(derivative)) and derivative[-1, 0] == 0
        assert jnp.linalg.norm(derivative[:2]) > 0


def test_depth_gate_changes_leading_quadratic_without_losing_structure() -> None:
    eta, xi, depth = _state()
    model = _model(surface_context_width=8, surface_depth_gate=True)
    variables = _activate_residual(model.init(jax.random.PRNGKey(30), jnp.stack((eta, xi), axis=-1), depth))
    coefficients = []
    for enabled in (False, True):
        current = model.clone(surface_depth_gate=enabled)

        def correction(surface: jnp.ndarray) -> jnp.ndarray:
            _, captured = current.apply(
                variables, jnp.stack((surface, xi), axis=-1), depth,
                capture_intermediates=lambda module, _: str(module.name).startswith("cs_block_"),
            )
            return sum(cast(jax.Array, captured["intermediates"][f"cs_block_{i}"]["__call__"][0])
                       for i in range(current.n_blocks)) / (current.n_blocks * current.latent) ** 0.5

        zero = jnp.zeros_like(eta)
        value, first = jax.jvp(correction, (zero,), (eta,))
        assert jnp.all(value == 0) and jnp.all(first == 0)
        second = jax.jvp(lambda surface: jax.jvp(correction, (surface,), (eta,))[1], (zero,), (eta,))[1]
        coefficients.append(0.5 * second)
        assert jnp.linalg.norm(correction(1e-4 * eta) / 1e-8 - coefficients[-1]) / jnp.linalg.norm(coefficients[-1]) < 1e-3
    assert jnp.linalg.norm(coefficients[1] - coefficients[0]) / jnp.linalg.norm(coefficients[0]) > 1e-4
    psi = jnp.roll(xi, 9, axis=-1)
    actual = _learned_residual(model, variables, eta, xi, depth)
    other = _learned_residual(model, variables, eta, psi, depth)
    mixed = _learned_residual(model, variables, eta, 0.3 * xi - 0.7 * psi, depth)
    assert jnp.allclose(jnp.vdot(psi, actual), jnp.vdot(other, xi), rtol=1e-11, atol=1e-14)
    assert jnp.allclose(mixed, 0.3 * actual - 0.7 * other, rtol=1e-10, atol=1e-14)
    shifted = _learned_residual(model, variables, jnp.roll(eta, 9, axis=-1), psi, depth)
    assert jnp.allclose(shifted, jnp.roll(actual, 9, axis=-1), rtol=1e-10, atol=1e-14)


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


def test_parity_products_formula_normalization_and_head() -> None:
    eta, xi, depth = _state()
    inputs = jnp.stack((eta, xi), axis=-1)
    k = jnp.arange(eta.shape[-1] // 2 + 1)
    eta_hat = jnp.fft.rfft(eta)
    dx, dxx, half, hilbert = (
        jnp.fft.irfft(symbol * eta_hat, n=eta.shape[-1])
        for symbol in (1j * k, -k**2, jnp.sqrt(k), -1j * jnp.sign(k))
    )
    raw = jnp.stack((eta, eta**2, eta**3, dx, dxx, half, hilbert), axis=-1)
    products = jnp.stack((eta, eta**2, eta**3, dx**2, dxx, half, hilbert**2, dx * hilbert), axis=-1)
    assert jnp.array_equal(surface_parity_products(raw), products)
    scales = (0.2, 0.04, 0.008, 4., 8., 0.7, 0.09, 0.6)
    normalized = products / jnp.asarray(scales)
    for polynomial_skip in (False, True):
        model = _model(
            surface_parity_products=True, surface_feature_scales=scales,
            surface_context_width=8, surface_context_gate=True,
            surface_polynomial_skip=polynomial_skip, nested_blocks=1,
        )
        initialized = model.init(jax.random.PRNGKey(71), inputs, depth)
        prediction, baseline, _, correction = cast(tuple[jax.Array, ...], model.apply(
            initialized, inputs, depth, return_baseline=True,
        ))
        assert jnp.array_equal(prediction, baseline)
        assert jnp.all(correction == 0)
        variables = _activate_residual(initialized)
        _, captured = model.apply(
            variables, inputs, depth,
            capture_intermediates=lambda module, _: module.name in (
                "eta_feat_proj", "eta_feat_mix", "eta_context_expand", "phi_proj",
            ),
        )
        intermediates = captured["intermediates"]
        assert jnp.allclose(
            cast(jax.Array, intermediates["eta_feat_proj"]["__call__"][0]),
            normalized @ cast(jax.Array, variables["params"]["eta_feat_proj"]["kernel"]), rtol=1e-12, atol=1e-14,
        )
        z = nn.gelu(cast(jax.Array, intermediates["eta_feat_mix"]["__call__"][0]))
        context = cast(jax.Array, intermediates["eta_context_expand"]["__call__"][0])
        skip = ((normalized[..., 1:3],) if polynomial_skip else ()) + (normalized[..., jnp.asarray((3, 6, 7))],)
        features = jnp.concatenate((z * jnp.tanh(z) * (1 + jnp.tanh(context)), *skip), axis=-1)
        assert jnp.array_equal(
            cast(jax.Array, intermediates["cs_block_0"]["phi_proj"]["__call__"][0]),
            features @ cast(jax.Array, variables["params"]["cs_block_0"]["phi_proj"]["kernel"]),
        )


def test_parity_products_structure_and_direct_quadratic_skip() -> None:
    eta, xi, depth = _state()
    eta, xi = (jnp.concatenate((value, jnp.roll(value, 7, axis=-1)), axis=0) for value in (eta, xi))
    depth = jnp.concatenate((depth, jnp.log(jnp.asarray([[0.3]]))), axis=0)
    model = _model(
        surface_parity_products=True, surface_feature_scales=(0.2, 0.04, 0.008, 4., 8., 0.7, 0.09, 0.6),
        surface_context_width=8, surface_context_gate=True, nested_blocks=1,
    )
    inputs = jnp.stack((eta, xi), axis=-1)
    initialized = model.init(jax.random.PRNGKey(72), inputs, depth)
    variables = _activate_residual(initialized)
    reflection = (-jnp.arange(eta.shape[-1])) % eta.shape[-1]
    output = cast(jax.Array, model.apply(variables, inputs, depth))
    reflected = cast(jax.Array, model.apply(variables, inputs[:, reflection], depth))
    assert jnp.allclose(reflected, output[:, reflection], rtol=1e-10, atol=1e-13)
    psi = jnp.roll(xi, 9, axis=-1)
    actual = _learned_residual(model, variables, eta, xi, depth)
    other = _learned_residual(model, variables, eta, psi, depth)
    mixed = _learned_residual(model, variables, eta, 0.3 * xi - 0.7 * psi, depth)
    assert jnp.allclose(mixed, 0.3 * actual - 0.7 * other, rtol=1e-10, atol=1e-14)
    assert jnp.allclose(jnp.vdot(psi, actual), jnp.vdot(other, xi), rtol=1e-10, atol=1e-14)
    residual = partial(_learned_residual, model, variables, xi=xi, depth=depth)
    value, variation = jax.jvp(residual, (jnp.zeros_like(eta),), (eta,))
    assert jnp.all(value == 0) and jnp.all(variation == 0)
    norms = [jnp.linalg.norm(residual(scale * eta)) for scale in (1 / 32, 1 / 64, 1 / 128)]
    assert all(abs(float(a / b) / 4 - 1) < 0.01 for a, b in zip(norms[:-1], norms[1:], strict=True))
    for channel in range(3):
        direct = unfreeze(initialized)
        kernel = direct["params"]["cs_block_0"]["phi_proj"]["kernel"]
        direct["params"]["cs_block_0"]["phi_proj"]["kernel"] = kernel.at[-3 + channel, 0].set(1)
        at_one = _learned_residual(model, direct, eta, xi, depth)
        at_half = _learned_residual(model, direct, eta / 2, xi, depth)
        assert jnp.linalg.norm(at_one) > 1e-8
        assert jnp.allclose(at_one, 4 * at_half, rtol=1e-12, atol=1e-14)


def test_parity_products_hero_parameter_count_and_default_identity() -> None:
    eta, xi, depth = _state(32)
    inputs = jnp.stack((eta, xi), axis=-1)
    original = CraigSulemDNO(
        width=320, n_blocks=8, latent=4, nested_blocks=1, mult_hidden=80,
        surface_context_width=8, surface_feature_scales=(1.,) * 7,
    )
    key = jax.random.PRNGKey(73)
    initialized = original.init(key, inputs, depth)
    explicit_default = original.clone(surface_parity_products=False).init(key, inputs, depth)
    assert all(jnp.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(initialized), jax.tree.leaves(explicit_default), strict=True,
    ))
    model = original.clone(surface_parity_products=True, surface_feature_scales=(1.,) * 8)
    enabled = model.init(key, inputs, depth)
    assert sum(value.size for value in jax.tree.leaves(enabled)) - sum(
        value.size for value in jax.tree.leaves(initialized)
    ) == 244
    assert cast(jax.Array, enabled["params"]["eta_feat_proj"]["kernel"]).shape == (8, 160)
    assert cast(jax.Array, enabled["params"]["cs_block_0"]["phi_proj"]["kernel"]).shape == (163, 4)
    assert jnp.array_equal(cast(jax.Array, original.apply(initialized, inputs, depth)),
                           cast(jax.Array, model.apply(enabled, inputs, depth)))
    streams = model.clone(surface_parity_streams=True, surface_feature_scales=(1.,) * 10)
    stream_state = streams.init(key, inputs, depth)
    assert sum(value.size for value in jax.tree.leaves(stream_state)) == 26068
    for name, shapes in (
        ("eta_feat_proj", ((8, 80), (2, 80))),
        ("eta_feat_mix", ((80, 80), (80, 80))),
        ("eta_context_in", ((80, 4), (80, 4))),
        ("eta_context_expand", ((4, 80), (4, 80))),
    ):
        for stream, shape in zip(("even_kernel", "odd_kernel"), shapes, strict=True):
            value = cast(jax.Array, stream_state["params"][name][stream])
            assert value.shape == shape and value.dtype == jnp.float32
    assert cast(jax.Array, stream_state["params"]["cs_block_0"]["phi_proj"]["kernel"]).shape == (163, 4)
    assert jnp.array_equal(cast(jax.Array, streams.apply(stream_state, inputs, depth)),
                           cast(jax.Array, original.apply(initialized, inputs, depth)))
    stream_default = model.clone(surface_parity_streams=False).init(key, inputs, depth)
    assert all(jnp.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(enabled), jax.tree.leaves(stream_default), strict=True,
    ))


def test_parity_streams_features_dense_and_activation() -> None:
    raw = jax.random.normal(jax.random.PRNGKey(74), (2, 16, 7), dtype=jnp.float64)
    features = surface_parity_products(raw, include_odd=True)
    assert jnp.array_equal(features[..., :8], surface_parity_products(raw))
    assert jnp.array_equal(features[..., 8:], raw[..., jnp.asarray((3, 6))])
    reflection = (-jnp.arange(raw.shape[1])) % raw.shape[1]
    for even_inputs, input_width in ((8, 10), (8, 16), (4, 8)):
        layer = ParityDense(features=16, even_inputs=even_inputs)
        x = jax.random.normal(jax.random.PRNGKey(input_width), (2, 16, input_width), dtype=jnp.float64)
        variables = layer.init(jax.random.PRNGKey(75), x)
        even_kernel, odd_kernel = (
            cast(jax.Array, variables["params"][name]) for name in ("even_kernel", "odd_kernel")
        )
        expected = jnp.concatenate((
            x[..., :even_inputs] @ even_kernel, x[..., even_inputs:] @ odd_kernel,
        ), axis=-1)
        actual = cast(jax.Array, layer.apply(variables, x))
        assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-14)
        assert jnp.array_equal(actual, parity_dense(x, even_kernel, odd_kernel))
        signs = jnp.concatenate((jnp.ones(even_inputs), -jnp.ones(input_width - even_inputs)))
        reflected = parity_gelu(cast(jax.Array, layer.apply(variables, x[:, reflection] * signs)))
        activated = parity_gelu(actual)
        assert jnp.allclose(
            reflected, activated[:, reflection] * jnp.repeat(jnp.asarray((1, -1)), 8),
            rtol=1e-12, atol=1e-14,
        )
        assert jnp.allclose(
            activated[..., :8] + activated[..., 8:],
            nn.gelu(actual[..., :8] + actual[..., 8:]), rtol=1e-12, atol=1e-14,
        )


def test_parity_streams_structure_and_live_gradients() -> None:
    eta, xi, depth = _state()
    eta, xi = (jnp.concatenate((v, jnp.roll(v, 7, axis=-1))) for v in (eta, xi))
    depth = jnp.concatenate((depth, jnp.log(jnp.asarray([[0.3]]))))
    model = _model(
        surface_parity_products=True, surface_parity_streams=True,
        surface_feature_scales=(0.2, 0.04, 0.008, 4., 8., 0.7, 0.09, 0.6, 2., 0.3),
        surface_context_width=8, nested_blocks=1,
    )
    inputs = jnp.stack((eta, xi), axis=-1)
    initialized = model.init(jax.random.PRNGKey(76), inputs, depth)
    variables = _activate_residual(initialized)
    reflection = (-jnp.arange(eta.shape[-1])) % eta.shape[-1]
    for batch_size in (1, 2):
        output = cast(jax.Array, model.apply(variables, inputs[:batch_size], depth[:batch_size]))
        reflected = cast(jax.Array, model.apply(
            variables, inputs[:batch_size, reflection], depth[:batch_size],
        ))
        assert jnp.allclose(reflected, output[:, reflection], rtol=1e-10, atol=1e-13)
    residual = partial(_learned_residual, model, variables, xi=xi, depth=depth)
    actual = residual(eta)
    assert jnp.linalg.norm(actual) > 1e-8
    psi = jnp.roll(xi, 9, axis=-1)
    other = _learned_residual(model, variables, eta, psi, depth)
    mixed = _learned_residual(model, variables, eta, 0.3 * xi - 0.7 * psi, depth)
    assert jnp.allclose(mixed, 0.3 * actual - 0.7 * other, rtol=1e-10, atol=1e-14)
    assert jnp.allclose(jnp.vdot(psi, actual), jnp.vdot(other, xi), rtol=1e-10, atol=1e-14)
    value, variation = jax.jvp(residual, (jnp.zeros_like(eta),), (eta,))
    assert jnp.all(value == 0) and jnp.all(variation == 0)
    norms = [jnp.linalg.norm(residual(scale * eta)) for scale in (1 / 32, 1 / 64, 1 / 128)]
    assert all(abs(float(a / b) / 4 - 1) < 0.01 for a, b in zip(norms[:-1], norms[1:], strict=True))
    gradients = jax.grad(lambda params: jnp.mean(_learned_residual(
        model, {"params": params}, eta, xi, depth,
    ) ** 2))(variables["params"])
    assert all(jnp.all(jnp.isfinite(v)) and jnp.linalg.norm(v) > 0 for v in jax.tree.leaves(gradients))
    initial_gradients = jax.grad(lambda params: jnp.mean(_learned_residual(
        model, {"params": params}, eta, xi, depth,
    ) * xi))(initialized["params"])
    assert jnp.linalg.norm(initial_gradients["cs_block_0"]["phi_proj"]["kernel"][8:16]) > 0
    assert jnp.all(initial_gradients["eta_feat_proj"]["odd_kernel"] == 0)


def test_parity_streams_restore_filtered_odd_quadratic() -> None:
    """A filtered Hilbert channel survives where every local product vanishes."""
    _, xi, depth = _state()
    x = 2 * jnp.pi * jnp.arange(xi.shape[-1]) / xi.shape[-1]
    eta = (jnp.sin(x) - 2 * jnp.sin(2 * x) + jnp.sin(3 * x))[None]
    model = _model(
        surface_parity_products=True, surface_parity_streams=True,
        surface_feature_scales=(1.,) * 10, surface_context_width=8,
    )
    variables = unfreeze(model.init(jax.random.PRNGKey(77), jnp.stack((eta, xi), axis=-1), depth))
    params = variables["params"]
    for name in ("eta_feat_proj", "eta_feat_mix", "eta_context_in", "eta_context_expand"):
        params[name] = jax.tree.map(jnp.zeros_like, params[name])
    params["eta_feat_proj"]["odd_kernel"] = params["eta_feat_proj"]["odd_kernel"].at[1, 0].set(1)
    for name in ("eta_feat_mix", "eta_context_in", "eta_context_expand"):
        params[name]["odd_kernel"] = params[name]["odd_kernel"].at[0, 0].set(1)
    for name in ("eta_context_hidden", "eta_context_out"):
        params[name] = jax.tree.map(jnp.zeros_like, params[name])
    params["eta_context_hidden"]["kernel"] = params["eta_context_hidden"]["kernel"].at[0, 0].set(1)
    params["eta_context_out"]["kernel"] = params["eta_context_out"]["kernel"].at[0, 4].set(1)
    params["cs_block_0"]["phi_proj"]["kernel"] = params["cs_block_0"]["phi_proj"]["kernel"].at[8, 0].set(1)
    modes = jnp.arange(1, 4, dtype=eta.dtype)
    symbol = 0.1 * jnp.tanh(jnp.tanh(jnp.log1p(modes)))
    filtered_at_zero = jnp.dot(jnp.asarray((-1., 2., -1.)), symbol)
    leading = filtered_at_zero**2 / 16
    assert leading > 1e-9
    for scale in (1e-2, 5e-3, 2.5e-3):
        _, captured = model.apply(
            variables, jnp.stack((scale * eta, xi), axis=-1), depth,
            capture_intermediates=lambda module, _: module.name == "phi_proj",
        )
        coefficient = cast(
            jax.Array, captured["intermediates"]["cs_block_0"]["phi_proj"]["__call__"][0],
        )[0, 0, 0]
        assert jnp.allclose(coefficient / scale**2, leading, rtol=1e-7, atol=1e-16)
    # At x=0, eta=eta_x=Heta=0, so the three local quadratic skips cannot supply this.
    assert abs(float(jnp.fft.irfft(-1j * jnp.sign(jnp.arange(33)) * jnp.fft.rfft(eta))[0, 0])) < 1e-14


def test_nested_branch_formula_and_gradients() -> None:
    eta, xi, depth = _state()
    features = jnp.stack((eta, eta**2), axis=-1)
    block = CraigSulemBlock(4, 2 * pi, 5.0, 16, nested=True)
    variables = unfreeze(block.init(jax.random.PRNGKey(19), features, xi, depth, eta_norm=eta))
    assert set(variables["params"]) == {"m_shared", "m_inner", "gamma"}
    assert variables["params"]["gamma"].dtype == jnp.float32
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


def test_nested_surface_modulation_identity_depth_and_gradient() -> None:
    eta, xi, _ = _state()
    inputs = jnp.broadcast_to(jnp.stack((eta + 0.03, xi), axis=-1), (2, eta.shape[-1], 2))
    depth = jnp.log(jnp.asarray([[0.08], [12.0]], dtype=eta.dtype))
    scales = (0.2, 0.04, 0.008, 2., 8., 0.7, 0.3)
    original = _model(nested_blocks=1, surface_context_width=8, surface_feature_scales=scales).clone(
        eta_scale=0.17, xi_scale=0.08, target_scale=0.13,
    )
    model = original.clone(nested_surface_modulation=True)
    original_variables = original.init(jax.random.PRNGKey(34), inputs, depth)
    variables = unfreeze(model.init(jax.random.PRNGKey(34), inputs, depth))
    gain = cast(jax.Array, variables["params"].pop("nested_surface_gain"))
    assert gain.shape == () and gain == 0
    assert all(jnp.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(original_variables), jax.tree.leaves(variables), strict=True,
    ))
    for active in (False, True):
        control = _activate_residual(original_variables) if active else original_variables
        variables = unfreeze(control)
        variables["params"]["nested_surface_gain"] = gain
        assert jnp.array_equal(cast(jax.Array, model.apply(variables, inputs, depth)),
                               cast(jax.Array, original.apply(control, inputs, depth)))
    for value in (0.0, 0.7):
        variables["params"]["nested_surface_gain"] = jnp.asarray(value, dtype=gain.dtype)
        gradient = jax.grad(lambda params: jnp.mean(cast(jax.Array, model.apply(
            {"params": params}, inputs, depth,
        )) ** 2))(variables["params"])
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient))
        assert jnp.abs(gradient["nested_surface_gain"]) > 0
    _, captured = model.apply(variables, inputs, depth, capture_intermediates=lambda module, _: module.name == "cs_block_1")
    actual = cast(jax.Array, captured["intermediates"]["cs_block_1"]["__call__"][0])
    surface = inputs[..., 0] / scales[0] * (
        1 + jnp.tanh(variables["params"]["nested_surface_gain"])
        * jnp.tanh(inputs[..., 0] * model.eta_scale / jnp.exp(depth))
    )
    block = CraigSulemBlock(model.latent, model.domain_length, model.h_clip_max, model.mult_hidden, nested=True)
    expected = cast(jax.Array, block.apply(
        {"params": variables["params"]["cs_block_1"]}, inputs, inputs[..., 1] * model.xi_scale,
        jnp.minimum(depth, jnp.log(model.h_clip_max)), eta_norm=surface,
    ))
    assert jnp.array_equal(actual, expected)


def test_nested_surface_features_initialization_and_channel_order() -> None:
    eta, xi, depth = _state()
    inputs = jnp.stack((eta, xi), axis=-1)
    scales = (0.2, 0.04, 0.008, 2., 8., 0.7, 0.3)
    original = _model(nested_blocks=1, surface_context_width=8, surface_feature_scales=scales).clone(
        width=320, n_blocks=8, latent=4, mult_hidden=80,
    )
    model = original.clone(nested_surface_features=True)
    control = unfreeze(original.init(jax.random.PRNGKey(36), inputs, depth))
    variables = unfreeze(model.init(jax.random.PRNGKey(36), inputs, depth))
    assert sum(leaf.size for leaf in jax.tree.leaves(control)) == 40384
    assert sum(leaf.size for leaf in jax.tree.leaves(variables)) == 40392
    mix = variables["params"]["cs_block_7"].pop("surface_mix")
    assert mix.shape == (2, 4) and mix.dtype == jnp.float32 and jnp.all(mix == 0)
    assert all(jnp.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(control), jax.tree.leaves(variables), strict=True,
    ))
    variables["params"]["cs_block_7"]["surface_mix"] = mix
    for state in (control, variables):
        state["params"]["cs_block_7"]["gamma"] = jnp.asarray([.1, -.2, .3, -.4], jnp.float32)
    for dtype in (jnp.float32, jnp.float64):
        assert jnp.array_equal(cast(jax.Array, original.apply(control, inputs.astype(dtype), depth.astype(dtype))),
                               cast(jax.Array, model.apply(variables, inputs.astype(dtype), depth.astype(dtype))))
    # Capture before whitening and the nonlinear trunk, with two channel layouts.
    for first_derivative in (False, True):
        current_scales = scales if first_derivative else (*scales[:3], *scales[4:])
        transform = 1.1 * jnp.eye(len(current_scales)) + .1
        current = _model(nested_blocks=1, nested_surface_features=True,
                         use_first_deriv=first_derivative, surface_feature_scales=current_scales,
                         surface_feature_transform=tuple(map(tuple, transform.tolist())))
        captured_features = []

        def intercept(next_fun: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], context: Any) -> Any:
            if isinstance(context.module, CraigSulemBlock) and context.module.nested:
                captured_features.append(kwargs["nested_features"])
            return next_fun(*args, **kwargs)

        with nn.intercept_methods(intercept):
            current.init(jax.random.PRNGKey(37), inputs, depth)
        k = jnp.arange(eta.shape[-1] // 2 + 1, dtype=eta.dtype)
        expected = jnp.stack((
            jnp.fft.irfft(jnp.sqrt(k) * jnp.fft.rfft(eta), n=eta.shape[-1]) / scales[5],
            jnp.fft.irfft(-k**2 * jnp.fft.rfft(eta), n=eta.shape[-1]) / scales[4],
        ), axis=-1)
        assert len(captured_features) == 1
        assert jnp.allclose(captured_features[0], expected, rtol=1e-12, atol=1e-14)


def test_nested_surface_features_preserve_structure_and_trainable_quadratic() -> None:
    eta, xi, depth = _state()
    model = _model(nested_blocks=2, nested_surface_features=True,
                   surface_feature_scales=(0.2, 0.04, 0.008, 2., 8., 0.7, 0.3)).clone(latent=4)
    inputs = jnp.stack((eta, xi), axis=-1)
    variables = unfreeze(_activate_residual(model.init(jax.random.PRNGKey(38), inputs, depth)))
    before = _learned_residual(model, variables, eta, xi, depth)
    for index in range(2):
        variables["params"][f"cs_block_{index}"]["surface_mix"] = jnp.asarray(
            [[.1, -.1, .05, .1], [.05, .1, -.1, .05]], dtype=jnp.float32,
        )
    actual = _learned_residual(model, variables, eta, xi, depth)
    assert jnp.linalg.norm(actual - before) > 1e-10
    psi = jnp.roll(xi, 9, axis=-1)
    other = _learned_residual(model, variables, eta, psi, depth)
    mixed = _learned_residual(model, variables, eta, .3 * xi - .7 * psi, depth)
    assert jnp.allclose(mixed, .3 * actual - .7 * other, rtol=1e-10, atol=1e-14)
    assert jnp.allclose(jnp.vdot(psi, actual), jnp.vdot(other, xi), rtol=1e-11, atol=1e-14)
    for amplitude in (.5, .25, .0625):
        assert jnp.allclose(_learned_residual(model, variables, amplitude * eta, xi, depth),
                            amplitude**2 * actual, rtol=1e-11, atol=1e-14)
    zero, first = jax.jvp(lambda surface: _learned_residual(model, variables, surface, xi, depth),
                          (jnp.zeros_like(eta),), (eta,))
    assert jnp.all(zero == 0) and jnp.all(first == 0)
    gradients = jax.grad(lambda params, fields, log_depth: jnp.sum(cast(jax.Array, model.apply(
        {"params": params}, fields, log_depth,
    )) ** 2), argnums=(0, 1, 2))(variables["params"], inputs, depth)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradients))
    for index in range(2):
        assert jnp.all(jnp.abs(gradients[0][f"cs_block_{index}"]["surface_mix"]) > 0)


def test_nested_surface_modulation_preserves_structure_and_adds_cubic_response() -> None:
    eta, xi, depth = _state()
    model = _model(nested_blocks=2, nested_surface_modulation=True, surface_context_width=8,
                   surface_feature_scales=(0.2, 0.04, 0.008, 2., 8., 0.7, 0.3))
    unmodulated = _activate_residual(model.init(jax.random.PRNGKey(35), jnp.stack((eta, xi), axis=-1), depth))
    variables = unfreeze(unmodulated)
    variables["params"]["nested_surface_gain"] = jnp.asarray(0.7, dtype=jnp.float32)
    psi = jnp.roll(xi, 9, axis=-1)
    actual = _learned_residual(model, variables, eta, xi, depth)
    other = _learned_residual(model, variables, eta, psi, depth)
    mixed = _learned_residual(model, variables, eta, 0.3 * xi - 0.7 * psi, depth)
    assert jnp.allclose(mixed, 0.3 * actual - 0.7 * other, rtol=1e-10, atol=1e-14)
    assert jnp.allclose(jnp.vdot(psi, actual), jnp.vdot(other, xi), rtol=1e-11, atol=1e-14)
    zero, first = jax.jvp(lambda surface: _learned_residual(model, variables, surface, xi, depth),
                          (jnp.zeros_like(eta),), (eta,))
    assert jnp.all(zero == 0) and jnp.linalg.norm(first) < 1e-14
    quadratic, cubic = [], []
    for amplitude in (0.25, 0.125, 0.0625):
        correction = _learned_residual(model, variables, amplitude * eta, xi, depth)
        reference = _learned_residual(model, unmodulated, amplitude * eta, xi, depth)
        quadratic.append(jnp.linalg.norm(correction))
        cubic.append(jnp.linalg.norm(correction - reference))
    assert jnp.all(jnp.asarray(cubic) > 0)
    assert jnp.allclose(jnp.asarray(quadratic[:-1]) / jnp.asarray(quadratic[1:]), 4.0, rtol=0.01)
    assert jnp.allclose(jnp.asarray(cubic[:-1]) / jnp.asarray(cubic[1:]), 8.0, rtol=0.01)


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


def _terms_model(baseline_order: int = 3) -> CraigSulemTermsResidual:
    return CraigSulemTermsResidual(
        width=32, surface_context_width=4, baseline_order=baseline_order,
        surface_feature_scales=(0.1, 0.01, 0.001, 0.2, 1.0, 0.3, 0.1),
        eta_scale=0.7, xi_scale=0.4, target_scale=0.6,
    )


def _active_terms_variables(
    model: CraigSulemTermsResidual, inputs: jax.Array, depth: jax.Array,
) -> FrozenVariableDict:
    variables = unfreeze(model.init(jax.random.PRNGKey(27), inputs, depth))
    kernel = variables["params"]["terms_head"]["kernel"]
    variables["params"]["terms_head"]["kernel"] = 0.2 * jax.random.normal(
        jax.random.PRNGKey(28), kernel.shape, dtype=kernel.dtype,
    )
    return freeze(variables)


def _physical_terms(model: CraigSulemTermsResidual, inputs: jax.Array, depth: jax.Array) -> jax.Array:
    k = (2 * jnp.pi / model.domain_length) * jnp.arange(inputs.shape[1] // 2 + 1, dtype=jnp.float64)
    return dno_series_eval(
        inputs[..., 0].astype(jnp.float64) * model.eta_scale,
        inputs[..., 1].astype(jnp.float64) * model.xi_scale,
        k, jnp.exp(depth.astype(jnp.float64)), order=model.baseline_order, pad_factor=2, return_terms=True,
    )


def test_terms_residual_zero_head_exact_baseline_and_split_initialization(baseline_order: int = 3) -> None:
    eta, xi, _ = _state()
    inputs = jnp.repeat(jnp.stack((eta, xi), axis=-1), 2, axis=0)
    depth = jnp.log(jnp.asarray([[0.8], [8.0]], dtype=jnp.float64))
    model = _terms_model(baseline_order)
    terms = _physical_terms(model, inputs, depth)
    key = jax.random.PRNGKey(27)
    variables = model.init(key, inputs, depth)
    assert cast(jax.Array, variables["params"]["terms_head"]["kernel"]).shape == (model.width // 2, baseline_order + 1)
    encoded_variables = model.init(key, inputs, depth, method=model.encode_geometry)
    assert jax.tree.structure(variables) == jax.tree.structure(encoded_variables)
    assert all(jnp.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(variables), jax.tree.leaves(encoded_variables), strict=True,
    ))
    assert not any(name.startswith("cs_block_") for name in variables["params"])
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(variables))
    assert cast(jax.Array, model.apply(
        variables, inputs, depth, method=model.encode_geometry,
    )).dtype == jnp.float32
    prediction, baseline, m1, correction = cast(tuple[jax.Array, ...], model.apply(
        variables, inputs, depth, return_baseline=True,
    ))
    assert prediction.dtype == jnp.float64
    assert jnp.array_equal(prediction, jnp.sum(terms, axis=-2)[..., None] / model.target_scale)
    assert jnp.array_equal(prediction, baseline)
    assert jnp.array_equal(m1, jnp.sum(terms[..., :2, :], axis=-2)[..., None] / model.target_scale)
    assert jnp.all(correction == 0)
    assert jnp.array_equal(prediction, cast(jax.Array, model.apply(
        variables, inputs, depth, terms_phys=terms,
    )))


def test_terms_residual_split_forward_and_all_gradients(baseline_order: int = 3) -> None:
    eta, xi, depth = _state(32)
    inputs = jnp.stack((eta, xi), axis=-1)
    model = _terms_model(baseline_order)
    variables = _active_terms_variables(model, inputs, depth)

    def split(params: Any, values: jax.Array, log_h: jax.Array) -> jax.Array:
        coefficients = model.apply({"params": params}, values, log_h, method=model.encode_geometry)
        return cast(jax.Array, model.apply(
            {}, coefficients, _physical_terms(model, values, log_h), method=model.decode_terms,
        ))

    def native_loss(params: Any, values: jax.Array, log_h: jax.Array) -> jax.Array:
        return jnp.sum(cast(jax.Array, model.apply({"params": params}, values, log_h)) ** 2)

    def split_loss(params: Any, values: jax.Array, log_h: jax.Array) -> jax.Array:
        return jnp.sum(split(params, values, log_h) ** 2)

    args = (variables["params"], inputs, depth)
    native = cast(jax.Array, model.apply(variables, inputs, depth))
    assert jnp.array_equal(native, split(*args))
    native_grads = jax.grad(native_loss, argnums=(0, 1, 2))(*args)
    split_grads = jax.grad(split_loss, argnums=(0, 1, 2))(*args)
    for expected, actual in zip(jax.tree.leaves(native_grads), jax.tree.leaves(split_grads), strict=True):
        assert jnp.all(jnp.isfinite(actual))
        assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-13)
    assert jnp.linalg.norm(native_grads[0]["eta_feat_proj"]["kernel"]) > 0
    assert jnp.linalg.norm(native_grads[1][..., 0]) > 0
    assert jnp.linalg.norm(native_grads[1][..., 1]) > 0
    assert jnp.linalg.norm(native_grads[2]) > 0
    provided_grads = jax.grad(lambda params, values, log_h: jnp.sum(cast(jax.Array, model.apply(
        {"params": params}, values, log_h, terms_phys=_physical_terms(model, values, log_h),
    )) ** 2), argnums=(0, 1, 2))(*args)
    for expected, actual in zip(jax.tree.leaves(native_grads), jax.tree.leaves(provided_grads), strict=True):
        assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-13)
    direction = jnp.stack((jnp.roll(eta, 3, axis=-1), jnp.zeros_like(xi)), axis=-1)
    native_tangent = jax.jvp(lambda values: cast(jax.Array, model.apply(variables, values, depth)),
                             (inputs,), (direction,))[1]
    provided_tangent = jax.jvp(lambda values: cast(jax.Array, model.apply(
        variables, values, depth, terms_phys=_physical_terms(model, values, depth),
    )), (inputs,), (direction,))[1]
    assert jnp.linalg.norm(native_tangent) > 0
    assert jnp.allclose(provided_tangent, native_tangent, rtol=1e-12, atol=1e-13)
    terms = _physical_terms(model, inputs, depth)
    term_grads = jax.grad(lambda q: jnp.sum(cast(tuple[jax.Array, ...], model.apply(
        variables, inputs, depth, terms_phys=q, return_baseline=True,
    ))[3] ** 2))(terms)
    assert jnp.all(jnp.isfinite(term_grads))
    assert jnp.all(jnp.linalg.norm(term_grads, axis=-1) > 0)


def test_terms_residual_formula_linearity_gauge_and_mean(baseline_order: int = 3) -> None:
    eta, xi, depth = _state()
    inputs = jnp.stack((eta, xi), axis=-1)
    model = _terms_model(baseline_order)
    variables = _active_terms_variables(model, inputs, depth)
    coefficients = cast(jax.Array, model.apply(variables, inputs, depth, method=model.encode_geometry))
    assert jnp.all(jnp.linalg.norm(coefficients, axis=1) > 0)
    terms = _physical_terms(model, inputs, depth)
    raw = jnp.sum(coefficients * jnp.swapaxes(terms, -2, -1), axis=-1)
    expected = (raw - jnp.mean(raw, axis=-1, keepdims=True)) / model.target_scale
    prediction, _, _, residual = cast(tuple[jax.Array, ...], model.apply(
        variables, inputs, depth, return_baseline=True,
    ))
    assert jnp.array_equal(residual[..., 0], expected)
    assert jnp.linalg.norm(residual) > 1e-10
    assert jnp.max(jnp.abs(jnp.mean(prediction, axis=1))) < 1e-14
    assert jnp.max(jnp.abs(jnp.mean(residual, axis=1))) < 1e-14
    shifted = inputs.at[..., 1].add(0.3)
    assert jnp.array_equal(coefficients, cast(jax.Array, model.apply(
        variables, shifted, depth, method=model.encode_geometry,
    )))
    assert jnp.allclose(prediction, cast(jax.Array, model.apply(variables, shifted, depth)), rtol=1e-11, atol=1e-13)
    psi = jnp.roll(xi, 9, axis=-1)
    other = cast(jax.Array, model.apply(variables, jnp.stack((eta, psi), axis=-1), depth))
    combined = cast(jax.Array, model.apply(
        variables, jnp.stack((eta, 0.3 * xi - 0.7 * psi), axis=-1), depth,
    ))
    assert jnp.allclose(combined, 0.3 * prediction - 0.7 * other, rtol=1e-11, atol=1e-13)
    constant = cast(jax.Array, model.apply(variables, inputs.at[..., 1].set(0.3), depth))
    assert jnp.max(jnp.abs(constant)) < 1e-13


def test_terms_residual_has_baseline_plus_one_order_onset(baseline_order: int = 3) -> None:
    eta, xi, depth = _state()
    inputs = jnp.stack((eta, xi), axis=-1)
    model = _terms_model(baseline_order)
    variables = _active_terms_variables(model, inputs, depth)
    norms = [jnp.linalg.norm(_learned_residual(model, variables, scale * eta, xi, depth))
             for scale in (0.04, 0.02, 0.01)]
    assert all(float(value) > 1e-16 for value in norms)
    assert all(abs(float(first / second) / 2 ** (baseline_order + 1) - 1) < 0.125
               for first, second in zip(norms[:-1], norms[1:], strict=True))
    terms = _physical_terms(model, inputs, depth)
    for scale in (-0.37, 0.37):
        scaled_terms = _physical_terms(model, inputs.at[..., 0].multiply(scale), depth)
        expected = terms * scale ** jnp.arange(baseline_order + 1)[None, :, None]
        assert jnp.allclose(scaled_terms, expected, rtol=1e-8, atol=1e-12)
    value, first_variation = jax.jvp(
        lambda surface: _learned_residual(model, variables, surface, xi, depth),
        (jnp.zeros_like(eta),), (eta,),
    )
    assert jnp.all(value == 0)
    assert jnp.all(first_variation == 0)


def test_terms_residual_m1_invariants() -> None:
    for test in (test_terms_residual_zero_head_exact_baseline_and_split_initialization,
                 test_terms_residual_split_forward_and_all_gradients,
                 test_terms_residual_formula_linearity_gauge_and_mean,
                 test_terms_residual_has_baseline_plus_one_order_onset):
        test(1)


def test_terms_residual_can_learn_at_surface_zero_crossings() -> None:
    nx = 64
    x = 2 * jnp.pi * jnp.arange(nx, dtype=jnp.float64) / nx
    nodes = jnp.arange(8, nx, 16)
    eta = (0.04 * jnp.cos(2 * x)).at[nodes].set(0.0)[None, :]
    xi = jnp.sin(2 * x)[None, :]
    depth = jnp.zeros((1, 1), dtype=jnp.float64)
    inputs = jnp.stack((eta, xi), axis=-1)
    model = _terms_model()
    variables = model.init(jax.random.PRNGKey(27), inputs, depth)
    terms = _physical_terms(model, inputs, depth)
    # Pointwise eta powers kill channels 0..2 here, and parity kills G3.
    assert jnp.max(jnp.abs(terms[:, 3, nodes])) < 1e-12
    probe = xi[:, nodes, None]
    gradients = jax.grad(lambda params: jnp.sum(probe * cast(tuple[jax.Array, ...], model.apply(
        {"params": params}, inputs, depth, terms_phys=terms, return_baseline=True,
    ))[3][:, nodes]))(variables["params"])
    assert jnp.linalg.norm(gradients["terms_head"]["kernel"][:, 0]) > 1e-10
    active = _active_terms_variables(model, inputs, depth)
    residual = cast(tuple[jax.Array, ...], model.apply(
        active, inputs, depth, terms_phys=terms, return_baseline=True,
    ))[3]
    assert jnp.ptp(residual[:, nodes]) > 1e-10


def _phase_rank_one_state(nx: int = 64) -> tuple[jax.Array, jax.Array, jax.Array]:
    phase = 2 * jnp.pi * jnp.arange(nx, dtype=jnp.float64) / nx
    nyquist = (-1.0) ** jnp.arange(nx)
    eta = jnp.asarray([[1.0], [0.7]]) * (
        0.01 + 0.04 * jnp.cos(3 * phase) + 0.015 * jnp.sin(5 * phase) + 0.003 * nyquist
    )
    xi = jnp.broadcast_to(
        0.3 + 0.07 * jnp.sin(3 * phase) - 0.02 * jnp.cos(5 * phase)
        + 0.03 * jnp.sin(7 * phase) + 0.005 * nyquist, eta.shape,
    )
    return eta, xi, jnp.log(jnp.asarray([[0.2], [12.0]]))


def test_phase_rank_one_spectrum_matches_physical_definition() -> None:
    length = 3 * pi
    coefficients = jnp.asarray([0.2, -0.3], dtype=jnp.float32)
    for nx in (64, 96):
        eta, xi, depth = _phase_rank_one_state(nx)
        phase = 2 * jnp.pi * jnp.arange(nx, dtype=jnp.float64) / nx
        k0 = 2 * pi / length
        # Analytic derivative ignores the deliberately nonzero DC/Nyquist modes.
        u = jnp.asarray([[1.0], [0.7]]) * (
            -0.04 * (3 * k0) ** 1.5 * jnp.sin(3 * phase)
            + 0.015 * (5 * k0) ** 1.5 * jnp.cos(5 * phase)
        )
        beta = coefficients[0] + coefficients[1] * jnp.log(k0 * jnp.exp(depth))
        expected = beta * u * jnp.mean(u * xi, axis=-1, keepdims=True)
        actual_hat = phase_rank_one_hat(
            jnp.fft.rfft(eta), jnp.fft.rfft(xi), depth, coefficients, length,
        )
        assert jnp.all(actual_hat[..., 0] == 0) and jnp.all(actual_hat[..., -1] == 0)
        assert jnp.allclose(jnp.fft.irfft(actual_hat, n=nx), expected, rtol=1e-11, atol=1e-13)
        assert jnp.allclose(actual_hat, jnp.fft.rfft(expected), rtol=1e-11, atol=1e-12)


def test_phase_rank_one_preserves_operator_structure_and_quadratic_order() -> None:
    eta, xi, depth = _phase_rank_one_state()
    coefficients = jnp.asarray([0.2, -0.3], dtype=jnp.float32)

    def response(surface: jax.Array, potential: jax.Array, log_depth: jax.Array = depth) -> jax.Array:
        return jnp.fft.irfft(phase_rank_one_hat(
            jnp.fft.rfft(surface), jnp.fft.rfft(potential), log_depth, coefficients, 3 * pi,
        ), n=surface.shape[-1])

    actual = response(eta, xi)
    psi = jnp.roll(xi, 3, axis=-1)
    other = response(eta, psi)
    assert jnp.all(jnp.linalg.norm(actual, axis=-1) > 1e-5)
    assert jnp.allclose(response(eta, 0.3 * xi - 0.7 * psi),
                        0.3 * actual - 0.7 * other, rtol=1e-11, atol=1e-13)
    assert jnp.allclose(jnp.mean(psi * actual, axis=-1),
                        jnp.mean(xi * other, axis=-1), rtol=1e-11, atol=1e-13)
    assert jnp.allclose(response(eta, xi + 2.3), actual, rtol=1e-11, atol=1e-13)
    assert jnp.max(jnp.abs(jnp.mean(actual, axis=-1))) < 1e-14
    reflected = (-jnp.arange(eta.shape[-1])) % eta.shape[-1]
    assert jnp.allclose(response(eta[:, reflected], xi[:, reflected]),
                        actual[:, reflected], rtol=1e-11, atol=1e-13)
    assert jnp.allclose(response(jnp.roll(eta, 7, axis=-1), jnp.roll(xi, 7, axis=-1)),
                        jnp.roll(actual, 7, axis=-1), rtol=1e-11, atol=1e-13)
    for scale in (-0.5, 0.3):
        assert jnp.allclose(response(scale * eta, xi), scale**2 * actual, rtol=1e-11, atol=1e-13)
    _, eta_tangent = jax.jvp(lambda surface: response(surface, xi), (eta,), (eta,))
    _, xi_tangent = jax.jvp(lambda potential: response(eta, potential), (xi,), (xi,))
    _, depth_tangent = jax.jvp(lambda log_h: response(eta, xi, log_h), (depth,), (jnp.ones_like(depth),))
    phase = 2 * jnp.pi * jnp.arange(eta.shape[-1], dtype=jnp.float64) / eta.shape[-1]
    u = jnp.asarray([[1.0], [0.7]]) * (
        -0.04 * 2**1.5 * jnp.sin(3 * phase) + 0.015 * (10 / 3)**1.5 * jnp.cos(5 * phase)
    )
    assert jnp.allclose(eta_tangent, 2 * actual, rtol=1e-11, atol=1e-13)
    assert jnp.allclose(xi_tangent, actual, rtol=1e-11, atol=1e-13)
    assert jnp.allclose(depth_tangent, coefficients[1] * u * jnp.mean(u * xi, axis=-1, keepdims=True),
                        rtol=1e-11, atol=1e-13)
    zero, first = jax.jvp(lambda surface: response(surface, xi), (jnp.zeros_like(eta),), (eta,))
    assert jnp.all(zero == 0) and jnp.all(first == 0)
    assert jnp.all(response(jnp.full_like(eta, 0.2), xi) == 0)


def test_phase_rank_one_zero_head_parity_normalization_and_trainability() -> None:
    eta, xi, log_depth = _phase_rank_one_state()
    for dtype in (jnp.float32, jnp.float64):
        for nested_blocks in (0, 1):
            original = _model(nested_blocks=nested_blocks, surface_context_width=8).clone(
                domain_length=3 * pi, eta_scale=0.3, xi_scale=1.7, target_scale=0.4,
            )
            model = original.clone(phase_rank_one=True)
            inputs = jnp.stack((eta / original.eta_scale, xi / original.xi_scale), axis=-1).astype(dtype)
            depth = log_depth.astype(dtype)
            plain = original.init(jax.random.PRNGKey(41), inputs, depth)
            enabled = unfreeze(model.init(jax.random.PRNGKey(41), inputs, depth))
            head = enabled["params"].pop("phase_rank_one_coefficients")
            assert head.shape == (2,) and head.dtype == jnp.float32 and jnp.all(head == 0)
            assert jax.tree.structure(plain) == jax.tree.structure(enabled)
            assert all(jnp.array_equal(a, b) for a, b in zip(
                jax.tree.leaves(plain), jax.tree.leaves(enabled), strict=True,
            ))
            shared = _activate_residual(plain)
            enabled = unfreeze(shared)
            enabled["params"]["phase_rank_one_coefficients"] = head
            expected_zero = cast(tuple[jax.Array, ...], original.apply(shared, inputs, depth, return_baseline=True))
            actual_zero = cast(tuple[jax.Array, ...], model.apply(enabled, inputs, depth, return_baseline=True))
            assert all(jnp.array_equal(a, b) for a, b in zip(expected_zero, actual_zero, strict=True))
            gradients = jax.grad(lambda params: jnp.mean(cast(tuple[jax.Array, ...], model.apply(
                {"params": params}, inputs, depth, return_baseline=True,
            ))[3][..., 0] * xi))(enabled["params"])
            assert jnp.all(jnp.abs(gradients["phase_rank_one_coefficients"]) > 1e-8)
            assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradients))
            coefficients = jnp.asarray([0.2, -0.3], dtype=jnp.float32)
            enabled["params"]["phase_rank_one_coefficients"] = coefficients
            actual = cast(tuple[jax.Array, ...], model.apply(enabled, inputs, depth, return_baseline=True))
            increment = jnp.fft.irfft(phase_rank_one_hat(
                jnp.fft.rfft(inputs[..., 0]) * model.eta_scale,
                jnp.fft.rfft(inputs[..., 1] * model.xi_scale), depth, coefficients, model.domain_length,
            ), n=eta.shape[-1])[..., None] / model.target_scale
            tolerance = 2e-7 if dtype == jnp.float32 else 2e-13
            assert jnp.allclose(actual[0] - actual_zero[0], increment, rtol=2e-5, atol=tolerance)
            assert jnp.allclose(actual[3] - actual_zero[3], increment, rtol=2e-5, atol=tolerance)
            assert jnp.array_equal(actual[1], actual_zero[1]) and jnp.array_equal(actual[2], actual_zero[2])
            clipped = phase_rank_one_hat(
                jnp.fft.rfft(eta), jnp.fft.rfft(xi), jnp.minimum(log_depth, jnp.log(5.0)),
                coefficients, model.domain_length,
            )
            assert not jnp.allclose(increment[-1, :, 0] * model.target_scale,
                                    jnp.fft.irfft(clipped, n=eta.shape[-1])[-1], rtol=1e-3, atol=1e-8)


def test_phase_wavenumber_matches_physical_formula_and_constant_head() -> None:
    length = 3 * pi
    module = PhaseWavenumberMultiplier(8, length)
    for nx in (64, 96):
        eta, xi, depth = _phase_rank_one_state(nx)
        variables = unfreeze(module.init(jax.random.PRNGKey(67), depth, nx // 2 + 1))
        assert jnp.all(module.apply(variables, depth, nx // 2 + 1) == 0)
        variables["params"]["out"]["kernel"] = jnp.linspace(-.15, .2, 8)[:, None]
        variables["params"]["out"]["bias"] = jnp.asarray([.03])
        multiplier = cast(jax.Array, module.apply(variables, depth, nx // 2 + 1))
        kh = jnp.exp(depth) * (2 * pi / length) * jnp.arange(nx // 2 + 1)
        hidden, out = variables["params"]["hidden"], variables["params"]["out"]
        expected_multiplier = (jnp.tanh(jnp.stack((jnp.log1p(kh), jnp.tanh(kh)), axis=-1)
                                         @ hidden["kernel"] + hidden["bias"])
                               @ out["kernel"] + out["bias"])[..., 0]
        assert jnp.allclose(multiplier, expected_multiplier, rtol=1e-13, atol=1e-15)
        assert not jnp.allclose(multiplier[-1], cast(jax.Array, module.apply(
            variables, jnp.minimum(depth, jnp.log(5.0)), nx // 2 + 1,
        ))[-1], rtol=1e-3, atol=1e-6)
        phase = 2 * pi * jnp.arange(nx) / nx
        components = jnp.asarray([[1.0], [.7]])[..., None] * jnp.stack((
            -.04 * (3 * 2 * pi / length)**1.5 * jnp.sin(3 * phase),
            .015 * (5 * 2 * pi / length)**1.5 * jnp.cos(5 * phase),
        ))
        u = jnp.sum(components, axis=1)
        v = jnp.sum(components * multiplier[:, jnp.asarray([3, 5]), None], axis=1)
        expected = .5 * (u * jnp.mean(v * xi, axis=-1, keepdims=True)
                         + v * jnp.mean(u * xi, axis=-1, keepdims=True))
        eta_hat, xi_hat = jnp.fft.rfft(eta), jnp.fft.rfft(xi)
        actual = phase_wavenumber_hat(eta_hat, xi_hat, multiplier, length)
        assert jnp.all(actual[..., 0] == 0) and jnp.all(actual[..., -1] == 0)
        assert jnp.allclose(actual, jnp.fft.rfft(expected), rtol=1e-11, atol=1e-13)
        constant = phase_wavenumber_hat(eta_hat, xi_hat, jnp.full_like(multiplier, .23), length)
        assert jnp.allclose(constant, phase_rank_one_hat(
            eta_hat, xi_hat, depth, jnp.asarray([.23, 0.0]), length,
        ), rtol=1e-12, atol=1e-14)


def test_phase_wavenumber_structure_order_and_active_gradients() -> None:
    eta, xi, depth = _phase_rank_one_state()
    module = PhaseWavenumberMultiplier(8, 3 * pi)
    variables = unfreeze(module.init(jax.random.PRNGKey(67), depth, eta.shape[-1] // 2 + 1))
    variables["params"]["out"]["kernel"] = jnp.linspace(-.15, .2, 8)[:, None]
    variables["params"]["out"]["bias"] = jnp.asarray([.03])

    def response(surface: jax.Array, potential: jax.Array, log_h: jax.Array = depth,
                 params: Any = variables["params"]) -> jax.Array:
        multiplier = cast(jax.Array, module.apply({"params": params}, log_h, surface.shape[-1] // 2 + 1))
        return jnp.fft.irfft(phase_wavenumber_hat(
            jnp.fft.rfft(surface), jnp.fft.rfft(potential), multiplier, 3 * pi,
        ), n=surface.shape[-1])

    actual = response(eta, xi)
    psi = jnp.roll(xi, 3, axis=-1)
    other = response(eta, psi)
    assert jnp.all(jnp.linalg.norm(actual, axis=-1) > 1e-6)
    assert jnp.allclose(response(eta, .3*xi-.7*psi), .3*actual-.7*other, rtol=1e-11, atol=1e-13)
    assert jnp.allclose(jnp.mean(psi*actual, axis=-1), jnp.mean(xi*other, axis=-1), rtol=1e-11, atol=1e-13)
    assert jnp.allclose(response(eta, xi+2.3), actual, rtol=1e-11, atol=1e-13)
    assert jnp.allclose(response(eta+.4, xi), actual, rtol=1e-11, atol=1e-13)
    assert jnp.max(jnp.abs(jnp.mean(actual, axis=-1))) < 1e-14
    reflected = (-jnp.arange(eta.shape[-1])) % eta.shape[-1]
    assert jnp.allclose(response(eta[:, reflected], xi[:, reflected]), actual[:, reflected], rtol=1e-11, atol=1e-13)
    assert jnp.allclose(response(jnp.roll(eta, 7, axis=-1), jnp.roll(xi, 7, axis=-1)),
                        jnp.roll(actual, 7, axis=-1), rtol=1e-11, atol=1e-13)
    for scale in (-.5, .3):
        assert jnp.allclose(response(scale*eta, xi), scale**2*actual, rtol=1e-11, atol=1e-13)
    _, eta_tangent = jax.jvp(lambda field: response(field, xi), (eta,), (eta,))
    _, xi_tangent = jax.jvp(lambda field: response(eta, field), (xi,), (xi,))
    _, depth_tangent = jax.jvp(lambda log_h: response(eta, xi, log_h), (depth,), (jnp.ones_like(depth),))
    assert jnp.allclose(eta_tangent, 2*actual, rtol=1e-11, atol=1e-13)
    assert jnp.allclose(xi_tangent, actual, rtol=1e-11, atol=1e-13)
    assert jnp.all(jnp.isfinite(depth_tangent)) and jnp.linalg.norm(depth_tangent) > 1e-8
    gradients = jax.grad(lambda params: jnp.mean(response(eta, xi, params=params)*xi))(variables["params"])
    assert all(jnp.all(jnp.isfinite(leaf)) and jnp.linalg.norm(leaf) > 1e-9
               for leaf in jax.tree.leaves(gradients))
    zero, first = jax.jvp(lambda field: response(field, xi), (jnp.zeros_like(eta),), (eta,))
    assert jnp.all(zero == 0) and jnp.all(first == 0)
    assert jnp.all(response(jnp.full_like(eta, .2), xi) == 0)


def test_phase_wavenumber_zero_init_native_scaling_and_trainability() -> None:
    eta, xi, depth = _phase_rank_one_state()
    for dtype in (jnp.float32, jnp.float64):
        for nested_blocks in (0, 1):
            plain = _model(nested_blocks=nested_blocks, surface_context_width=8).clone(
                domain_length=3*pi, eta_scale=.3, xi_scale=1.7, target_scale=.4,
            )
            model = plain.clone(phase_wavenumber_hidden=8)
            inputs = jnp.stack((eta/plain.eta_scale, xi/plain.xi_scale), axis=-1).astype(dtype)
            log_h = depth.astype(dtype)
            original = plain.init(jax.random.PRNGKey(41), inputs, log_h)
            variables = unfreeze(model.init(jax.random.PRNGKey(41), inputs, log_h))
            head = variables["params"].pop("phase_wavenumber")
            assert all(jnp.all(leaf == 0) for leaf in jax.tree.leaves(head["out"]))
            assert jax.tree.structure(original) == jax.tree.structure(variables)
            assert all(jnp.array_equal(a,b) for a,b in zip(jax.tree.leaves(original), jax.tree.leaves(variables), strict=True))
            shared = _activate_residual(original)
            variables = unfreeze(shared)
            variables["params"]["phase_wavenumber"] = head
            expected = cast(tuple[jax.Array, ...], plain.apply(shared, inputs, log_h, return_baseline=True))
            actual_zero = cast(tuple[jax.Array, ...], model.apply(variables, inputs, log_h, return_baseline=True))
            assert all(jnp.array_equal(a,b) for a,b in zip(expected, actual_zero, strict=True))
            gradients = jax.grad(lambda params: jnp.mean(cast(tuple[jax.Array, ...], model.apply(
                {"params": params}, inputs, log_h, return_baseline=True,
            ))[3][..., 0]*xi))(variables["params"])
            assert all(jnp.linalg.norm(leaf) > 1e-8 for leaf in jax.tree.leaves(gradients["phase_wavenumber"]["out"]))
            assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradients))
            head["out"]["kernel"] = jnp.linspace(-.15, .2, 8, dtype=jnp.float32)[:, None]
            head["out"]["bias"] = jnp.asarray([.03], dtype=jnp.float32)
            multiplier = cast(jax.Array, PhaseWavenumberMultiplier(8, model.domain_length).apply(
                {"params": head}, log_h, eta.shape[-1]//2+1,
            ))
            increment = jnp.fft.irfft(phase_wavenumber_hat(
                jnp.fft.rfft(inputs[..., 0])*model.eta_scale,
                jnp.fft.rfft(inputs[..., 1]*model.xi_scale), multiplier, model.domain_length,
            ), n=eta.shape[-1])[..., None]/model.target_scale
            actual = cast(tuple[jax.Array, ...], model.apply(variables, inputs, log_h, return_baseline=True))
            tolerance = 2e-7 if dtype == jnp.float32 else 2e-13
            assert jnp.allclose(actual[3]-actual_zero[3], increment, rtol=2e-5, atol=tolerance)
            assert jnp.allclose(actual[0]-actual_zero[0], increment, rtol=2e-5, atol=tolerance)
            assert jnp.array_equal(actual[1], actual_zero[1]) and jnp.array_equal(actual[2], actual_zero[2])


def test_surface_amplitude_preserves_training_dtype_under_x64() -> None:
    eta, xi, depth = _state()
    model = _model(nested_blocks=1, surface_context_width=8).clone(surface_amplitude_reference=.04)
    for dtype in (jnp.float32, jnp.float64):
        inputs = jnp.stack((eta, xi), axis=-1).astype(dtype)
        log_h = depth.astype(dtype)
        variables = model.init(jax.random.PRNGKey(79), inputs, log_h)
        assert variables["params"]["eta_amplitude_gain"].dtype == jnp.float32
        assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(variables))
        for state in (variables, _activate_residual(variables)):
            outputs = cast(tuple[jax.Array, ...], model.apply(state, inputs, log_h, return_baseline=True))
            assert all(value.dtype == dtype and jnp.all(jnp.isfinite(value)) for value in outputs)


def test_surface_amplitude_formula_and_original_nested_scale() -> None:
    eta, xi, depth = _state()
    plain = _model(nested_blocks=1, surface_feature_scales=(.2, .1, .06, .4, .8, .3, .2)).clone(
        domain_length=3*pi, eta_scale=.3, xi_scale=.7, target_scale=.8,
    )
    model = plain.clone(surface_amplitude_reference=.03)
    inputs = jnp.stack((eta, xi), axis=-1)
    initialized = unfreeze(model.init(jax.random.PRNGKey(52), inputs, depth))
    plain_initialized = plain.init(jax.random.PRNGKey(52), inputs, depth)
    gain = initialized["params"].pop("eta_amplitude_gain")
    assert jnp.all(gain == 0)
    assert all(jnp.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(initialized), jax.tree.leaves(plain_initialized), strict=True,
    ))
    initialized["params"]["eta_amplitude_gain"] = gain
    zero_head = cast(tuple[jax.Array, ...], model.apply(initialized, inputs, depth, return_baseline=True))
    assert jnp.array_equal(zero_head[0], zero_head[1])
    variables = unfreeze(_activate_residual(initialized))
    variables["params"]["eta_amplitude_gain"] = jnp.linspace(-.2, .3, model.width//2)
    a = jnp.sqrt(jnp.mean((eta/.03)**2, axis=-1, keepdims=True)+1e-6)
    assert jnp.array_equal(a, surface_amplitude(eta, .03))
    k = 2*pi/model.domain_length*jnp.arange(eta.shape[-1]//2+1)
    eta_hat = jnp.fft.rfft(eta)
    features = jnp.stack([
        eta, eta**2, eta**3,
        *(jnp.fft.irfft(symbol*eta_hat, n=eta.shape[-1]) for symbol in
          (1j*k, -k**2, jnp.sqrt(k), -1j*jnp.sign(k))),
    ], axis=-1)
    features = features / a[..., None]**jnp.asarray([1, 2, 3, 1, 1, 1, 1])
    features = features / jnp.asarray(model.surface_feature_scales)
    features = nn.gelu(features @ variables["params"]["eta_feat_proj"]["kernel"])
    features *= 1+jnp.log(a)[..., None]*variables["params"]["eta_amplitude_gain"]
    features = nn.gelu(features @ variables["params"]["eta_feat_mix"]["kernel"])
    features = a[..., None]**2*features*jnp.tanh(features)
    expected = jnp.zeros_like(xi)
    for index in range(model.n_blocks):
        block = CraigSulemBlock(model.latent, model.domain_length, model.h_clip_max,
                               model.mult_hidden, nested=index == 1)
        expected += cast(jax.Array, block.apply(
            {"params": variables["params"][f"cs_block_{index}"]},
            features, xi*model.xi_scale, depth, eta_norm=eta/.03,
        ))
    expected /= (model.n_blocks*model.latent)**.5*model.target_scale
    actual = _learned_residual(model, variables, eta, xi, depth)
    assert jnp.allclose(actual, expected, rtol=2e-12, atol=1e-14)
    variables["params"]["cs_block_0"]["phi_proj"]["kernel"] *= 0
    raw_nested = plain.clone(surface_feature_scales=(.03, .1, .06, .4, .8, .3, .2))
    assert jnp.array_equal(_learned_residual(model, variables, eta, xi, depth),
                           _learned_residual(raw_nested, variables, eta, xi, depth))


def test_surface_amplitude_structure_and_derivatives() -> None:
    eta, xi, depth = _state()
    eta = jnp.concatenate((eta, .04*eta))
    xi = jnp.concatenate((xi, jnp.roll(xi, 7, axis=-1)))
    depth = jnp.concatenate((depth, jnp.log(jnp.array([[.04]]))))
    model = _model(nested_blocks=1, surface_context_width=8,
                   surface_feature_scales=(.03, .002, .0002, .1, .4, .05, .03)).clone(
        surface_amplitude_reference=.04,
    )
    variables = unfreeze(_activate_residual(model.init(
        jax.random.PRNGKey(53), jnp.stack((eta, xi), axis=-1), depth,
    )))
    variables["params"]["eta_amplitude_gain"] = jnp.linspace(-.2, .3, model.width//2)

    @jax.jit
    def residual(surface: jax.Array, potential: jax.Array) -> jax.Array:
        return _learned_residual(model, variables, surface, potential, depth)

    other = jnp.roll(xi, 5, axis=-1)
    first, second = residual(eta, xi), residual(eta, other)
    assert jnp.allclose(residual(eta, 1.7*xi-.4*other), 1.7*first-.4*second, rtol=2e-11, atol=1e-13)
    assert jnp.allclose(jnp.sum(xi*second), jnp.sum(other*first), rtol=2e-11, atol=1e-13)
    assert jnp.allclose(residual(jnp.roll(eta, 9, axis=-1), jnp.roll(xi, 9, axis=-1)),
                        jnp.roll(first, 9, axis=-1), rtol=2e-11, atol=1e-13)
    zeros = jnp.zeros_like(eta)
    flat, first_variation = jax.jvp(lambda e: residual(e, xi), (zeros,), (eta,))
    assert jnp.max(jnp.abs(flat)) == 0 and jnp.max(jnp.abs(first_variation)) == 0
    second_variation = jax.jvp(
        lambda e: jax.jvp(lambda v: residual(v, xi), (e,), (eta,))[1], (zeros,), (eta,),
    )[1]
    errors = [jnp.linalg.norm(residual(scale*eta, xi)/scale**2-.5*second_variation)
              / jnp.linalg.norm(.5*second_variation) for scale in (1e-5, 1e-6, 1e-7)]
    assert errors[2] < .15*errors[1] < .15**2*errors[0]
    direction = jnp.roll(eta, 3, axis=-1)
    derivative = jax.jvp(lambda e: residual(e, xi), (eta,), (direction,))[1]
    fd = (residual(eta+1e-5*direction, xi)-residual(eta-1e-5*direction, xi))/2e-5
    assert jnp.linalg.norm(derivative-fd)/jnp.linalg.norm(derivative) < 2e-7
    gradients = jax.grad(lambda p: jnp.sum(_learned_residual(
        model, {"params": p}, eta, xi, depth,
    )*xi))(variables["params"])
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradients))
    assert jnp.linalg.norm(gradients["eta_amplitude_gain"]) > 1e-8


def main() -> int:
    tests: tuple[Callable[[], None], ...] = (
        test_default_matches_original_forward,
        test_mixed_surface_lift_structure_and_orders,
        test_mixed_surface_lift_fourier_inference,
        test_reflection_encoder_matches_full_prediction_average,
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
        test_cs_term_fields_reconstruct_and_differentiate,
        test_experimental_residual_orders,
        test_surface_feature_scaling_folds_into_first_projection,
        test_softened_curvature_uses_physical_wavenumbers,
        test_surface_feature_whitening_folds_and_preserves_operator_structure,
        test_surface_context_is_nonlocal_and_depth_conditioned,
        test_surface_context_has_live_gradients_and_unchanged_shared_initialization,
        test_bounded_context_gate_reuses_context_and_preserves_initialization,
        test_depth_gate_initial_identity_shared_parameters_and_gradient,
        test_depth_gate_formula_clipping_and_feature_order,
        test_depth_gate_changes_leading_quadratic_without_losing_structure,
        test_polynomial_skip_formula_initialization_and_gradients,
        test_even_surface_features_enforce_reflection_covariance,
        test_parity_products_formula_normalization_and_head,
        test_parity_products_structure_and_direct_quadratic_skip,
        test_parity_products_hero_parameter_count_and_default_identity,
        test_parity_streams_features_dense_and_activation,
        test_parity_streams_structure_and_live_gradients,
        test_parity_streams_restore_filtered_odd_quadratic,
        test_nested_branch_formula_and_gradients,
        test_nested_model_replacement_preserves_structure,
        test_nested_surface_modulation_identity_depth_and_gradient,
        test_nested_surface_features_initialization_and_channel_order,
        test_nested_surface_features_preserve_structure_and_trainable_quadratic,
        test_nested_surface_modulation_preserves_structure_and_adds_cubic_response,
        test_multiplier_skip_symbol_basis_and_grid_independence,
        test_multiplier_skip_preserves_initialization_and_operator_structure,
        test_log_multiplier_inputs_formula_initialization_and_gradients,
        test_log_multiplier_inputs_preserve_zero_head_prediction_and_parameters,
        test_terms_residual_zero_head_exact_baseline_and_split_initialization,
        test_terms_residual_split_forward_and_all_gradients,
        test_terms_residual_formula_linearity_gauge_and_mean,
        test_terms_residual_has_baseline_plus_one_order_onset,
        test_terms_residual_m1_invariants,
        test_terms_residual_can_learn_at_surface_zero_crossings,
        test_phase_rank_one_spectrum_matches_physical_definition,
        test_phase_rank_one_preserves_operator_structure_and_quadratic_order,
        test_phase_rank_one_zero_head_parity_normalization_and_trainability,
        test_phase_wavenumber_matches_physical_formula_and_constant_head,
        test_phase_wavenumber_structure_order_and_active_gradients,
        test_phase_wavenumber_zero_init_native_scaling_and_trainability,
        test_surface_amplitude_preserves_training_dtype_under_x64,
        test_surface_amplitude_formula_and_original_nested_scale,
        test_surface_amplitude_structure_and_derivatives,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
