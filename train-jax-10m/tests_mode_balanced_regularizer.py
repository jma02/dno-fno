"""CPU invariants for the universal mode-balanced complex DNO loss."""
from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from mode_balanced_regularizer import (
    ModeBalancedConfig,
    compute_mode_balanced_loss,
)


jax.config.update("jax_enable_x64", True)


def _grid(n: int = 512) -> tuple[jax.Array, jax.Array]:
    x = jnp.arange(n, dtype=jnp.float64) * (2.0 * jnp.pi / n)
    k_rfft = jnp.fft.rfftfreq(n, d=1.0 / n)
    return x, k_rfft


def _omega(mode: int, depth: float, gravity: float = 1.0) -> float:
    return float(np.sqrt(gravity * mode * np.tanh(mode * depth)))


def _loss(
    eta: jax.Array,
    prediction: jax.Array,
    target: jax.Array,
    depth: jax.Array,
    config: ModeBalancedConfig = ModeBalancedConfig(),
) -> tuple[jax.Array, dict[str, jax.Array]]:
    _, k_rfft = _grid(eta.shape[-1])
    return compute_mode_balanced_loss(
        eta, prediction, target, depth, k_rfft, config
    )


def test_exact_prediction_has_zero_loss() -> None:
    x, _ = _grid()
    eta = jnp.sin(5.0 * x)[None, :]
    target = jnp.cos(5.0 * x)[None, :]
    loss, diagnostics = _loss(
        eta, target, target, jnp.asarray([0.3], dtype=eta.dtype)
    )

    np.testing.assert_allclose(loss, 0.0, atol=1e-15)
    np.testing.assert_allclose(diagnostics["relative_error_rms"], 0.0, atol=1e-15)
    assert all(value.shape == () for value in diagnostics.values())


def test_complex_loss_detects_amplitude_and_phase_error() -> None:
    x, _ = _grid()
    depth_value = 0.4
    mode = 7
    omega = _omega(mode, depth_value)
    eta = jnp.cos(mode * x)[None, :]
    target = omega * jnp.cos(mode * x)[None, :]
    amplitude_prediction = 1.2 * target
    phase_prediction = omega * jnp.cos(mode * x + 0.2)[None, :]
    depth = jnp.asarray([depth_value], dtype=eta.dtype)

    amplitude_loss, _ = _loss(eta, amplitude_prediction, target, depth)
    phase_loss, _ = _loss(eta, phase_prediction, target, depth)

    assert float(amplitude_loss) > 1e-3
    assert float(phase_loss) > 1e-3


def test_sideband_fractional_error_has_comparable_weight_to_carrier() -> None:
    x, _ = _grid()
    depth_value = 0.7
    carrier_mode = 17
    sideband_mode = 19
    sideband_amplitude = 0.05
    carrier = jnp.cos(carrier_mode * x)
    sideband = sideband_amplitude * jnp.cos(sideband_mode * x)
    eta = (carrier + sideband)[None, :]
    target = (
        _omega(carrier_mode, depth_value) * carrier
        + _omega(sideband_mode, depth_value) * sideband
    )[None, :]
    carrier_error = 0.1 * _omega(carrier_mode, depth_value) * carrier
    sideband_error = 0.1 * _omega(sideband_mode, depth_value) * sideband
    depth = jnp.asarray([depth_value], dtype=eta.dtype)

    carrier_loss, _ = _loss(eta, target + carrier_error, target, depth)
    sideband_loss, _ = _loss(eta, target + sideband_error, target, depth)
    loss_ratio = float(sideband_loss / carrier_loss)

    assert 0.9 < loss_ratio < 1.1


def test_zero_and_out_of_band_modes_are_finite_and_inactive() -> None:
    x, _ = _grid()
    zeros = jnp.zeros((1, x.size), dtype=x.dtype)
    prediction = (2.0 + jnp.cos(150.0 * x))[None, :]
    loss, diagnostics = _loss(
        zeros,
        prediction,
        zeros,
        jnp.asarray([0.2], dtype=x.dtype),
    )

    np.testing.assert_allclose(loss, 0.0, atol=1e-15)
    np.testing.assert_allclose(diagnostics["active_modes"], 0.0, atol=1e-15)
    assert all(bool(jnp.isfinite(value)) for value in diagnostics.values())


def test_loss_is_translation_invariant() -> None:
    x, _ = _grid()
    eta = (jnp.cos(11.0 * x) + 0.1 * jnp.cos(13.0 * x))[None, :]
    target = (jnp.sin(11.0 * x) + 0.08 * jnp.sin(13.0 * x))[None, :]
    prediction = target + 0.03 * jnp.cos(13.0 * x)[None, :]
    depth = jnp.asarray([0.25], dtype=eta.dtype)
    shift = 73
    for dispersion_weighting in (False, True):
        config = replace(
            ModeBalancedConfig(),
            dispersion_weighting=dispersion_weighting,
        )
        loss, _ = _loss(eta, prediction, target, depth, config)
        shifted_loss, _ = _loss(
            jnp.roll(eta, shift, axis=-1),
            jnp.roll(prediction, shift, axis=-1),
            jnp.roll(target, shift, axis=-1),
            depth,
            config,
        )

        np.testing.assert_allclose(
            shifted_loss, loss, rtol=1e-12, atol=1e-15
        )


def test_prediction_gradient_is_finite_and_nonzero() -> None:
    x, k_rfft = _grid()
    eta = jnp.cos(9.0 * x)[None, :]
    target = jnp.sin(9.0 * x)[None, :]
    prediction = target + 0.02 * jnp.cos(9.0 * x)[None, :]
    depth = jnp.asarray([0.5], dtype=eta.dtype)

    def objective(value: jax.Array) -> jax.Array:
        return compute_mode_balanced_loss(
            eta,
            value,
            target,
            depth,
            k_rfft,
            ModeBalancedConfig(),
        )[0]

    gradient = jax.grad(objective)(prediction)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0


def test_dispersion_weighting_scales_single_mode_by_frequency_squared() -> None:
    x, _ = _grid()
    depth_value = 0.3
    mode = 11
    eta = jnp.cos(mode * x)[None, :]
    target = _omega(mode, depth_value) * jnp.sin(mode * x)[None, :]
    prediction = target + 0.02 * jnp.cos(mode * x)[None, :]
    depth = jnp.asarray([depth_value], dtype=eta.dtype)
    base_config = ModeBalancedConfig()

    unweighted_loss, _ = _loss(
        eta, prediction, target, depth, base_config
    )
    weighted_loss, diagnostics = _loss(
        eta,
        prediction,
        target,
        depth,
        replace(base_config, dispersion_weighting=True),
    )
    omega_squared = _omega(mode, depth_value) ** 2

    np.testing.assert_allclose(
        weighted_loss / unweighted_loss,
        omega_squared,
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        diagnostics["effective_frequency_squared_mean"],
        omega_squared,
        rtol=1e-12,
        atol=1e-14,
    )

    _, k_rfft = _grid()

    def objective(value: jax.Array, weighted: bool) -> jax.Array:
        return compute_mode_balanced_loss(
            eta,
            value,
            target,
            depth,
            k_rfft,
            replace(base_config, dispersion_weighting=weighted),
        )[0]

    unweighted_gradient = jax.grad(objective, argnums=0)(prediction, False)
    weighted_gradient = jax.grad(objective, argnums=0)(prediction, True)
    np.testing.assert_allclose(
        weighted_gradient,
        omega_squared * unweighted_gradient,
        rtol=1e-11,
        atol=1e-14,
    )


def test_dispersion_weighting_is_applied_before_batch_average() -> None:
    x, _ = _grid()
    modes = (2, 20)
    depth_values = (0.1, 1.0)
    fractional_errors = (0.3, 0.01)
    eta_rows = []
    target_rows = []
    prediction_rows = []
    individual_losses = []
    individual_frequencies = []
    base_config = ModeBalancedConfig()
    weighted_config = replace(base_config, dispersion_weighting=True)

    for mode, depth_value, fractional_error in zip(
        modes, depth_values, fractional_errors
    ):
        eta = jnp.cos(mode * x)[None, :]
        target = _omega(mode, depth_value) * jnp.sin(mode * x)[None, :]
        prediction = target * (1.0 + fractional_error)
        depth = jnp.asarray([depth_value], dtype=eta.dtype)
        loss, diagnostics = _loss(
            eta, prediction, target, depth, base_config
        )
        eta_rows.append(eta[0])
        target_rows.append(target[0])
        prediction_rows.append(prediction[0])
        individual_losses.append(loss)
        individual_frequencies.append(
            diagnostics["effective_frequency_squared_mean"]
        )

    batch_loss, _ = _loss(
        jnp.stack(eta_rows),
        jnp.stack(prediction_rows),
        jnp.stack(target_rows),
        jnp.asarray(depth_values, dtype=x.dtype),
        weighted_config,
    )
    expected = jnp.mean(
        jnp.stack(individual_losses) * jnp.stack(individual_frequencies)
    )
    incorrect_product_of_means = jnp.mean(
        jnp.stack(individual_losses)
    ) * jnp.mean(jnp.stack(individual_frequencies))

    np.testing.assert_allclose(batch_loss, expected, rtol=1e-12, atol=1e-15)
    assert not np.isclose(
        float(batch_loss),
        float(incorrect_product_of_means),
        rtol=1e-2,
        atol=1e-8,
    )


def test_effective_frequency_uses_scored_band_and_parseval_weights() -> None:
    x, _ = _grid(32)
    depth_value = 0.2
    amplitude_low = 0.7
    amplitude_nyquist = 0.2
    eta = (
        amplitude_low * jnp.cos(3.0 * x)
        + amplitude_nyquist * jnp.cos(16.0 * x)
    )[None, :]
    target = jnp.sin(3.0 * x)[None, :]
    depth = jnp.asarray([depth_value], dtype=x.dtype)
    config = replace(
        ModeBalancedConfig(k_max=16.0), dispersion_weighting=True
    )
    _, diagnostics = _loss(eta, target, target, depth, config)
    expected = (
        _omega(3, depth_value) ** 2 * amplitude_low**2 / 2.0
        + _omega(16, depth_value) ** 2 * amplitude_nyquist**2
    ) / (amplitude_low**2 / 2.0 + amplitude_nyquist**2)
    np.testing.assert_allclose(
        diagnostics["effective_frequency_squared_mean"],
        expected,
        rtol=1e-12,
        atol=1e-14,
    )

    x, _ = _grid()
    base_eta = jnp.cos(5.0 * x)[None, :]
    out_of_band_eta = base_eta + 100.0 * jnp.cos(150.0 * x)[None, :]
    target = jnp.sin(5.0 * x)[None, :]
    prediction = target + 0.1 * jnp.cos(5.0 * x)[None, :]
    depth = jnp.asarray([depth_value], dtype=x.dtype)
    base_loss, base_diagnostics = _loss(
        base_eta, prediction, target, depth, config
    )
    contaminated_loss, contaminated_diagnostics = _loss(
        out_of_band_eta, prediction, target, depth, config
    )
    np.testing.assert_allclose(
        contaminated_loss, base_loss, rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(
        contaminated_diagnostics["effective_frequency_squared_mean"],
        base_diagnostics["effective_frequency_squared_mean"],
        rtol=1e-12,
        atol=1e-14,
    )


def test_dispersion_weighting_is_finite_for_flat_surface() -> None:
    x, _ = _grid()
    eta = jnp.zeros((1, x.size), dtype=x.dtype)
    target = jnp.sin(5.0 * x)[None, :]
    prediction = target + 0.1 * jnp.cos(5.0 * x)[None, :]
    config = replace(ModeBalancedConfig(), dispersion_weighting=True)

    loss, diagnostics = _loss(
        eta,
        prediction,
        target,
        jnp.asarray([0.2], dtype=eta.dtype),
        config,
    )

    np.testing.assert_allclose(loss, 0.0, atol=1e-15)
    np.testing.assert_allclose(
        diagnostics["effective_frequency_squared_mean"], 0.0, atol=1e-15
    )
    assert all(bool(jnp.isfinite(value)) for value in diagnostics.values())


if __name__ == "__main__":
    tests = (
        test_exact_prediction_has_zero_loss,
        test_complex_loss_detects_amplitude_and_phase_error,
        test_sideband_fractional_error_has_comparable_weight_to_carrier,
        test_zero_and_out_of_band_modes_are_finite_and_inactive,
        test_loss_is_translation_invariant,
        test_prediction_gradient_is_finite_and_nonzero,
        test_dispersion_weighting_scales_single_mode_by_frequency_squared,
        test_dispersion_weighting_is_applied_before_batch_average,
        test_effective_frequency_uses_scored_band_and_parseval_weights,
        test_dispersion_weighting_is_finite_for_flat_surface,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
