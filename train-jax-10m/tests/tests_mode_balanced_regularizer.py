"""CPU invariants for the universal mode-balanced complex DNO loss."""

from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mode_balanced_regularizer import (  # noqa: E402
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
    *,
    k_max: float = 128.0,
    dispersion_weighting: bool = False,
) -> jax.Array:
    _, k_rfft = _grid(eta.shape[-1])
    return compute_mode_balanced_loss(
        eta,
        prediction,
        target,
        depth,
        k_rfft,
        k_max=k_max,
        dispersion_weighting=dispersion_weighting,
    )


def test_exact_prediction_has_zero_loss() -> None:
    x, _ = _grid()
    eta = jnp.sin(5.0 * x)[None, :]
    target = jnp.cos(5.0 * x)[None, :]
    loss = _loss(eta, target, target, jnp.asarray([0.3], dtype=eta.dtype))

    np.testing.assert_allclose(loss, 0.0, atol=1e-15)
    assert loss.shape == ()


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

    amplitude_loss = _loss(eta, amplitude_prediction, target, depth)
    phase_loss = _loss(eta, phase_prediction, target, depth)

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

    carrier_loss = _loss(eta, target + carrier_error, target, depth)
    sideband_loss = _loss(eta, target + sideband_error, target, depth)
    loss_ratio = float(sideband_loss / carrier_loss)

    assert 0.9 < loss_ratio < 1.1


def test_zero_and_out_of_band_modes_are_finite_and_inactive() -> None:
    x, _ = _grid()
    zeros = jnp.zeros((1, x.size), dtype=x.dtype)
    prediction = (2.0 + jnp.cos(150.0 * x))[None, :]
    loss = _loss(
        zeros,
        prediction,
        zeros,
        jnp.asarray([0.2], dtype=x.dtype),
    )

    np.testing.assert_allclose(loss, 0.0, atol=1e-15)


def test_loss_is_translation_invariant() -> None:
    x, _ = _grid()
    eta = (jnp.cos(11.0 * x) + 0.1 * jnp.cos(13.0 * x))[None, :]
    target = (jnp.sin(11.0 * x) + 0.08 * jnp.sin(13.0 * x))[None, :]
    prediction = target + 0.03 * jnp.cos(13.0 * x)[None, :]
    depth = jnp.asarray([0.25], dtype=eta.dtype)
    shift = 73
    for dispersion_weighting in (False, True):
        loss = _loss(
            eta,
            prediction,
            target,
            depth,
            dispersion_weighting=dispersion_weighting,
        )
        shifted_loss = _loss(
            jnp.roll(eta, shift, axis=-1),
            jnp.roll(prediction, shift, axis=-1),
            jnp.roll(target, shift, axis=-1),
            depth,
            dispersion_weighting=dispersion_weighting,
        )

        np.testing.assert_allclose(shifted_loss, loss, rtol=1e-12, atol=1e-15)


def test_prediction_gradient_is_finite_and_nonzero() -> None:
    x, k_rfft = _grid()
    eta = jnp.cos(9.0 * x)[None, :]
    target = jnp.sin(9.0 * x)[None, :]
    prediction = target + 0.02 * jnp.cos(9.0 * x)[None, :]
    depth = jnp.asarray([0.5], dtype=eta.dtype)

    gradient = jax.grad(compute_mode_balanced_loss, argnums=1)(
        eta, prediction, target, depth, k_rfft
    )
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
    unweighted_loss = _loss(eta, prediction, target, depth)
    weighted_loss = _loss(
        eta,
        prediction,
        target,
        depth,
        dispersion_weighting=True,
    )
    omega_squared = _omega(mode, depth_value) ** 2

    np.testing.assert_allclose(
        weighted_loss / unweighted_loss,
        omega_squared,
        rtol=1e-12,
        atol=1e-14,
    )

    gradient = jax.grad(_loss, argnums=1)
    unweighted_gradient = gradient(eta, prediction, target, depth)
    weighted_gradient = gradient(
        eta, prediction, target, depth, dispersion_weighting=True
    )
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
    for mode, depth_value, fractional_error in zip(
        modes, depth_values, fractional_errors
    ):
        eta = jnp.cos(mode * x)[None, :]
        target = _omega(mode, depth_value) * jnp.sin(mode * x)[None, :]
        prediction = target * (1.0 + fractional_error)
        depth = jnp.asarray([depth_value], dtype=eta.dtype)
        loss = _loss(eta, prediction, target, depth)
        eta_rows.append(eta[0])
        target_rows.append(target[0])
        prediction_rows.append(prediction[0])
        individual_losses.append(loss)
        individual_frequencies.append(_omega(mode, depth_value) ** 2)

    batch_loss = _loss(
        jnp.stack(eta_rows),
        jnp.stack(prediction_rows),
        jnp.stack(target_rows),
        jnp.asarray(depth_values, dtype=x.dtype),
        dispersion_weighting=True,
    )
    expected = jnp.mean(
        jnp.stack(individual_losses) * jnp.stack(individual_frequencies)
    )
    incorrect_product_of_means = jnp.mean(jnp.stack(individual_losses)) * jnp.mean(
        jnp.stack(individual_frequencies)
    )

    np.testing.assert_allclose(batch_loss, expected, rtol=1e-12, atol=1e-15)
    assert not np.isclose(
        float(batch_loss),
        float(incorrect_product_of_means),
        rtol=1e-2,
        atol=1e-8,
    )


def test_effective_frequency_uses_scored_band_and_parseval_weights() -> None:
    depth_value = 0.2
    amplitude_low = 0.7
    amplitude_high = 0.2
    for n in (31, 32):
        x, _ = _grid(n)
        highest_mode = n // 2
        eta = (
            amplitude_low * jnp.cos(3.0 * x)
            + amplitude_high * jnp.cos(highest_mode * x)
        )[None, :]
        target = jnp.sin(3.0 * x)[None, :]
        prediction = target + 0.1 * jnp.cos(3.0 * x)[None, :]
        depth = jnp.asarray([depth_value], dtype=x.dtype)
        unweighted_loss = _loss(
            eta, prediction, target, depth, k_max=float(highest_mode)
        )
        weighted_loss = _loss(
            eta,
            prediction,
            target,
            depth,
            k_max=float(highest_mode),
            dispersion_weighting=True,
        )
        high_energy = amplitude_high**2 / (2.0 if n % 2 else 1.0)
        expected = (
            _omega(3, depth_value) ** 2 * amplitude_low**2 / 2.0
            + _omega(highest_mode, depth_value) ** 2 * high_energy
        ) / (amplitude_low**2 / 2.0 + high_energy)
        np.testing.assert_allclose(
            weighted_loss / unweighted_loss, expected, rtol=1e-12, atol=1e-14
        )

    x, _ = _grid()
    base_eta = jnp.cos(5.0 * x)[None, :]
    out_of_band_eta = base_eta + 100.0 * jnp.cos(150.0 * x)[None, :]
    target = jnp.sin(5.0 * x)[None, :]
    prediction = target + 0.1 * jnp.cos(5.0 * x)[None, :]
    depth = jnp.asarray([depth_value], dtype=x.dtype)
    base_loss = _loss(base_eta, prediction, target, depth, dispersion_weighting=True)
    contaminated_loss = _loss(
        out_of_band_eta, prediction, target, depth, dispersion_weighting=True
    )
    np.testing.assert_allclose(contaminated_loss, base_loss, rtol=1e-12, atol=1e-15)


def test_dispersion_weighting_is_finite_for_flat_surface() -> None:
    x, k_rfft = _grid()
    eta = jnp.zeros((1, x.size), dtype=x.dtype)
    target = jnp.sin(5.0 * x)[None, :]
    prediction = target + 0.1 * jnp.cos(5.0 * x)[None, :]
    loss, gradient = jax.value_and_grad(compute_mode_balanced_loss, argnums=1)(
        eta,
        prediction,
        target,
        jnp.asarray([0.2], dtype=eta.dtype),
        k_rfft,
        dispersion_weighting=True,
    )

    np.testing.assert_allclose(loss, 0.0, atol=1e-15)
    np.testing.assert_allclose(gradient, 0.0, atol=1e-15)


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
