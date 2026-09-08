"""CPU invariants for the localized translation-tangent objective."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from translation_tangent_regularizer import (
    TranslationTangentConfig,
    compute_translation_tangent_loss,
)


jax.config.update("jax_enable_x64", True)


def _inputs() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    n = 256
    x = jnp.arange(n, dtype=jnp.float64) * (2.0 * jnp.pi / n)
    k = jnp.fft.fftfreq(n, d=1.0 / n)
    eta = jnp.sin(x)[None, :]
    eta_x = jnp.cos(x)[None, :]
    target = -0.4 * eta_x
    depth = jnp.asarray([0.16], dtype=jnp.float64)
    return eta, target, depth, k


def _batched_inputs() -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    n = 256
    x = jnp.arange(n, dtype=jnp.float64) * (2.0 * jnp.pi / n)
    k = jnp.fft.fftfreq(n, d=1.0 / n)
    modes = jnp.asarray([1.0, 2.0, 3.0], dtype=x.dtype)
    eta = jnp.sin(modes[:, None] * x[None, :])
    eta_x = modes[:, None] * jnp.cos(modes[:, None] * x[None, :])
    target = -0.4 * eta_x
    prediction = target + 0.01 * jnp.sin(
        (modes[:, None] + 1.0) * x[None, :]
    )
    depth = jnp.asarray([0.16, 0.21, 0.09], dtype=eta.dtype)
    return eta, prediction, target, depth, k


def _assert_results_allclose(
    actual: tuple[jax.Array, dict[str, jax.Array]],
    expected: tuple[jax.Array, dict[str, jax.Array]],
) -> None:
    actual_loss, actual_diagnostics = actual
    expected_loss, expected_diagnostics = expected
    np.testing.assert_allclose(actual_loss, expected_loss, rtol=1e-12, atol=1e-14)
    assert actual_diagnostics.keys() == expected_diagnostics.keys()
    for name in actual_diagnostics:
        np.testing.assert_allclose(
            actual_diagnostics[name],
            expected_diagnostics[name],
            rtol=1e-12,
            atol=1e-14,
            err_msg=name,
        )


def test_exact_prediction_has_zero_loss() -> None:
    eta, target, depth, k = _inputs()
    loss, diagnostics = compute_translation_tangent_loss(
        eta, target, target, depth, k, TranslationTangentConfig()
    )
    np.testing.assert_allclose(loss, 0.0, atol=1e-14)
    np.testing.assert_allclose(diagnostics["phase_growth_loss"], 0.0, atol=1e-14)
    np.testing.assert_allclose(
        diagnostics["rigid_phase_growth_loss"], 0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        diagnostics["differential_phase_growth_loss"], 0.0, atol=1e-14
    )
    assert all(value.shape == () for value in diagnostics.values())


def test_local_objective_detects_globally_cancelling_error() -> None:
    eta, target, depth, k = _inputs()
    x = jnp.arange(eta.shape[-1], dtype=eta.dtype) * (2.0 * jnp.pi / eta.shape[-1])
    eta_x = jnp.cos(x)[None, :]
    cancelling_error = 0.02 * jnp.sin(x)[None, :] * eta_x
    prediction = target + cancelling_error
    global_projection = jnp.sum(cancelling_error * eta_x)
    np.testing.assert_allclose(global_projection, 0.0, atol=1e-14)
    loss, diagnostics = compute_translation_tangent_loss(
        eta, prediction, target, depth, k, TranslationTangentConfig()
    )
    assert float(loss) > 1e-5
    assert float(diagnostics["differential_phase_growth_loss"]) > 100.0 * float(
        diagnostics["rigid_phase_growth_loss"]
    )
    np.testing.assert_allclose(
        diagnostics["phase_growth_loss"],
        diagnostics["rigid_phase_growth_loss"]
        + diagnostics["differential_phase_growth_loss"],
        rtol=1e-12,
        atol=1e-14,
    )


def test_rigid_phase_growth_matches_relative_elevation_growth() -> None:
    n = 256
    mode = 3
    speed_error = 0.02
    x = jnp.arange(n, dtype=jnp.float64) * (2.0 * jnp.pi / n)
    k = jnp.fft.fftfreq(n, d=1.0 / n)
    eta = jnp.sin(mode * x)[None, :]
    eta_x = mode * jnp.cos(mode * x)[None, :]
    target = -0.4 * eta_x
    prediction = target - speed_error * eta_x
    depth = jnp.asarray([0.16], dtype=eta.dtype)
    config = TranslationTangentConfig(denominator_eps=1e-12)

    loss, diagnostics = compute_translation_tangent_loss(
        eta, prediction, target, depth, k, config
    )
    expected_phase_growth = (mode * speed_error) ** 2

    np.testing.assert_allclose(
        diagnostics["phase_growth_loss"],
        expected_phase_growth,
        rtol=5e-10,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        diagnostics["rms_wave_number"], mode, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        diagnostics["phase_growth_loss"],
        diagnostics["phase_growth_multiplier"] * loss,
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(diagnostics["relative_speed_loss"], loss)
    np.testing.assert_allclose(
        diagnostics["rigid_phase_growth_loss"],
        diagnostics["phase_growth_loss"],
        rtol=5e-10,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        diagnostics["differential_phase_growth_loss"], 0.0, atol=1e-12
    )


def test_phase_growth_weights_narrower_profiles_by_wave_number_squared() -> None:
    n = 256
    x = jnp.arange(n, dtype=jnp.float64) * (2.0 * jnp.pi / n)
    k = jnp.fft.fftfreq(n, d=1.0 / n)
    modes = jnp.asarray([1.0, 4.0], dtype=x.dtype)
    eta = jnp.sin(modes[:, None] * x[None, :])
    eta_x = modes[:, None] * jnp.cos(modes[:, None] * x[None, :])
    target = -0.4 * eta_x
    prediction = target - 0.01 * eta_x
    depth = jnp.asarray([0.2, 0.2], dtype=eta.dtype)
    config = TranslationTangentConfig(denominator_eps=1e-12)

    def one_sample(index: int) -> tuple[float, float]:
        loss, diagnostics = compute_translation_tangent_loss(
            eta[index : index + 1],
            prediction[index : index + 1],
            target[index : index + 1],
            depth[index : index + 1],
            k,
            config,
        )
        return float(loss), float(diagnostics["phase_growth_loss"])

    broad_speed_loss, broad_phase_growth = one_sample(0)
    narrow_speed_loss, narrow_phase_growth = one_sample(1)
    np.testing.assert_allclose(
        narrow_speed_loss, broad_speed_loss, rtol=1e-9, atol=1e-14
    )
    np.testing.assert_allclose(
        narrow_phase_growth / broad_phase_growth, 16.0, rtol=1e-9
    )


def test_loss_is_translation_invariant() -> None:
    eta, target, depth, k = _inputs()
    prediction = target + 0.01 * jnp.sin(2.0 * jnp.pi * jnp.arange(256) / 256)[None, :]
    config = TranslationTangentConfig()
    loss, _ = compute_translation_tangent_loss(
        eta, prediction, target, depth, k, config
    )
    shifted_loss, _ = compute_translation_tangent_loss(
        jnp.roll(eta, 37, axis=-1),
        jnp.roll(prediction, 37, axis=-1),
        jnp.roll(target, 37, axis=-1),
        depth,
        k,
        config,
    )
    np.testing.assert_allclose(shifted_loss, loss, rtol=1e-12, atol=1e-14)


def test_per_sample_constant_offsets_leave_all_outputs_unchanged() -> None:
    eta, prediction, target, depth, k = _batched_inputs()
    config = TranslationTangentConfig()
    baseline = compute_translation_tangent_loss(
        eta, prediction, target, depth, k, config
    )
    prediction_offsets = jnp.asarray([3.5, -0.7, 11.0])[:, None]
    target_offsets = jnp.asarray([-2.0, 4.25, 0.3])[:, None]
    shifted = compute_translation_tangent_loss(
        eta,
        prediction + prediction_offsets,
        target + target_offsets,
        depth,
        k,
        config,
    )
    _assert_results_allclose(shifted, baseline)


def test_raw_and_precentered_inputs_have_identical_outputs() -> None:
    eta, prediction, target, depth, k = _batched_inputs()
    prediction = prediction + jnp.asarray([1.25, -8.0, 0.4])[:, None]
    target = target + jnp.asarray([-3.0, 2.5, 6.75])[:, None]
    config = TranslationTangentConfig()
    raw = compute_translation_tangent_loss(
        eta, prediction, target, depth, k, config
    )
    precentered = compute_translation_tangent_loss(
        eta,
        prediction - jnp.mean(prediction, axis=-1, keepdims=True),
        target - jnp.mean(target, axis=-1, keepdims=True),
        depth,
        k,
        config,
    )
    _assert_results_allclose(raw, precentered)


def test_flat_sample_is_gated_out() -> None:
    eta, target, depth, k = _inputs()
    eta = jnp.zeros_like(eta)
    loss, diagnostics = compute_translation_tangent_loss(
        eta,
        target + 0.01 * jnp.sin(2.0 * jnp.pi * jnp.arange(256) / 256)[None, :],
        target,
        depth,
        k,
        TranslationTangentConfig(),
    )
    np.testing.assert_allclose(loss, 0.0, atol=1e-14)
    np.testing.assert_allclose(diagnostics["phase_growth_loss"], 0.0, atol=1e-14)
    np.testing.assert_allclose(diagnostics["selected_samples"], 0.0, atol=1e-14)


def test_nonflat_sample_is_selected() -> None:
    eta, target, depth, k = _inputs()
    prediction = target - 0.02 * jnp.cos(
        2.0 * jnp.pi * jnp.arange(eta.shape[-1]) / eta.shape[-1]
    )[None, :]

    loss, diagnostics = compute_translation_tangent_loss(
        eta,
        prediction,
        target,
        depth,
        k,
        TranslationTangentConfig(),
    )
    assert float(loss) > 0.0
    np.testing.assert_allclose(
        diagnostics["selected_samples"], 1.0, atol=1e-14
    )


def test_gradient_is_finite_and_nonzero() -> None:
    eta, target, depth, k = _inputs()
    prediction = target + 0.01 * jnp.sin(2.0 * jnp.pi * jnp.arange(256) / 256)[None, :]

    def objective(value: jax.Array) -> jax.Array:
        return compute_translation_tangent_loss(
            eta, value, target, depth, k, TranslationTangentConfig()
        )[0]

    def phase_growth_objective(value: jax.Array) -> jax.Array:
        return compute_translation_tangent_loss(
            eta, value, target, depth, k, TranslationTangentConfig()
        )[1]["phase_growth_loss"]

    for gradient in (
        jax.grad(objective)(prediction),
        jax.grad(phase_growth_objective)(prediction),
    ):
        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert float(jnp.linalg.norm(gradient)) > 0.0


if __name__ == "__main__":
    tests = (
        test_exact_prediction_has_zero_loss,
        test_local_objective_detects_globally_cancelling_error,
        test_rigid_phase_growth_matches_relative_elevation_growth,
        test_phase_growth_weights_narrower_profiles_by_wave_number_squared,
        test_loss_is_translation_invariant,
        test_per_sample_constant_offsets_leave_all_outputs_unchanged,
        test_raw_and_precentered_inputs_have_identical_outputs,
        test_flat_sample_is_gated_out,
        test_nonflat_sample_is_selected,
        test_gradient_is_finite_and_nonzero,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
