"""CPU invariants for the localized translation-tangent objective."""

from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from translation_tangent_regularizer import (  # noqa: E402
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
    prediction = target + 0.01 * jnp.sin((modes[:, None] + 1.0) * x[None, :])
    depth = jnp.asarray([0.16, 0.21, 0.09], dtype=eta.dtype)
    return eta, prediction, target, depth, k


def test_exact_prediction_has_zero_loss() -> None:
    eta, target, depth, k = _inputs()
    loss, selected_count = compute_translation_tangent_loss(
        eta, target, target, depth, k
    )
    np.testing.assert_allclose(loss, 0.0, atol=1e-14)
    np.testing.assert_allclose(selected_count, 1.0)
    assert loss.shape == selected_count.shape == ()


def test_local_objective_detects_globally_cancelling_error() -> None:
    eta, target, depth, k = _inputs()
    x = jnp.arange(eta.shape[-1], dtype=eta.dtype) * (2.0 * jnp.pi / eta.shape[-1])
    eta_x = jnp.cos(x)[None, :]
    cancelling_error = 0.02 * jnp.sin(x)[None, :] * eta_x
    prediction = target + cancelling_error
    global_projection = jnp.sum(cancelling_error * eta_x)
    np.testing.assert_allclose(global_projection, 0.0, atol=1e-14)
    loss, _ = compute_translation_tangent_loss(eta, prediction, target, depth, k)
    assert float(loss) > 1e-5


def test_rigid_speed_error_matches_analytic_loss() -> None:
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
    gravity = 2.0

    loss, _ = compute_translation_tangent_loss(
        eta, prediction, target, depth, k, denominator_eps=1e-12, gravity=gravity
    )
    np.testing.assert_allclose(
        loss,
        speed_error**2 / (gravity * float(depth[0])),
        rtol=5e-10,
        atol=1e-14,
    )


def test_rigid_speed_loss_is_independent_of_wave_number() -> None:
    n = 256
    x = jnp.arange(n, dtype=jnp.float64) * (2.0 * jnp.pi / n)
    k = jnp.fft.fftfreq(n, d=1.0 / n)
    modes = jnp.asarray([1.0, 4.0], dtype=x.dtype)
    eta = jnp.sin(modes[:, None] * x[None, :])
    eta_x = modes[:, None] * jnp.cos(modes[:, None] * x[None, :])
    target = -0.4 * eta_x
    prediction = target - 0.01 * eta_x
    depth = jnp.asarray([0.2, 0.2], dtype=eta.dtype)
    losses = [
        compute_translation_tangent_loss(
            eta[index : index + 1],
            prediction[index : index + 1],
            target[index : index + 1],
            depth[index : index + 1],
            k,
            denominator_eps=1e-12,
        )[0]
        for index in range(2)
    ]
    np.testing.assert_allclose(losses[0], losses[1], rtol=1e-9, atol=1e-14)


def test_loss_is_translation_invariant() -> None:
    eta, target, depth, k = _inputs()
    prediction = target + 0.01 * jnp.sin(2.0 * jnp.pi * jnp.arange(256) / 256)[None, :]
    loss, _ = compute_translation_tangent_loss(eta, prediction, target, depth, k)
    shifted_loss, _ = compute_translation_tangent_loss(
        jnp.roll(eta, 37, axis=-1),
        jnp.roll(prediction, 37, axis=-1),
        jnp.roll(target, 37, axis=-1),
        depth,
        k,
    )
    np.testing.assert_allclose(shifted_loss, loss, rtol=1e-12, atol=1e-14)


def test_per_sample_constant_offsets_leave_all_outputs_unchanged() -> None:
    eta, prediction, target, depth, k = _batched_inputs()
    baseline = compute_translation_tangent_loss(eta, prediction, target, depth, k)
    prediction_offsets = jnp.asarray([3.5, -0.7, 11.0])[:, None]
    target_offsets = jnp.asarray([-2.0, 4.25, 0.3])[:, None]
    shifted = compute_translation_tangent_loss(
        eta,
        prediction + prediction_offsets,
        target + target_offsets,
        depth,
        k,
    )
    np.testing.assert_allclose(shifted, baseline, rtol=1e-12, atol=1e-14)


def test_raw_and_precentered_inputs_have_identical_outputs() -> None:
    eta, prediction, target, depth, k = _batched_inputs()
    prediction = prediction + jnp.asarray([1.25, -8.0, 0.4])[:, None]
    target = target + jnp.asarray([-3.0, 2.5, 6.75])[:, None]
    raw = compute_translation_tangent_loss(eta, prediction, target, depth, k)
    precentered = compute_translation_tangent_loss(
        eta,
        prediction - jnp.mean(prediction, axis=-1, keepdims=True),
        target - jnp.mean(target, axis=-1, keepdims=True),
        depth,
        k,
    )
    np.testing.assert_allclose(raw, precentered, rtol=1e-12, atol=1e-14)


def test_flat_sample_is_gated_out() -> None:
    eta, target, depth, k = _inputs()
    eta = jnp.zeros_like(eta)
    loss, selected_count = compute_translation_tangent_loss(
        eta,
        target + 0.01 * jnp.sin(2.0 * jnp.pi * jnp.arange(256) / 256)[None, :],
        target,
        depth,
        k,
    )
    np.testing.assert_allclose(loss, 0.0, atol=1e-14)
    np.testing.assert_allclose(selected_count, 0.0, atol=1e-14)


def test_flat_samples_do_not_change_selected_mean() -> None:
    eta, target, depth, k = _inputs()
    prediction = (
        target
        - 0.02
        * jnp.cos(2.0 * jnp.pi * jnp.arange(eta.shape[-1]) / eta.shape[-1])[None, :]
    )

    loss, selected_count = compute_translation_tangent_loss(
        eta,
        prediction,
        target,
        depth,
        k,
    )
    assert float(loss) > 0.0
    np.testing.assert_allclose(selected_count, 1.0, atol=1e-14)
    mixed_loss, mixed_count = compute_translation_tangent_loss(
        jnp.concatenate((eta, jnp.zeros_like(eta))),
        jnp.concatenate((prediction, prediction)),
        jnp.concatenate((target, target)),
        jnp.concatenate((depth, depth)),
        k,
    )
    np.testing.assert_allclose(mixed_count, selected_count)
    np.testing.assert_allclose(mixed_loss, loss, rtol=1e-12, atol=1e-14)


def test_gradient_is_finite_and_nonzero() -> None:
    eta, target, depth, k = _inputs()
    prediction = target + 0.01 * jnp.sin(2.0 * jnp.pi * jnp.arange(256) / 256)[None, :]

    gradient, _ = jax.jit(
        jax.grad(compute_translation_tangent_loss, argnums=1, has_aux=True)
    )(eta, prediction, target, depth, k)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0


if __name__ == "__main__":
    tests = (
        test_exact_prediction_has_zero_loss,
        test_local_objective_detects_globally_cancelling_error,
        test_rigid_speed_error_matches_analytic_loss,
        test_rigid_speed_loss_is_independent_of_wave_number,
        test_loss_is_translation_invariant,
        test_per_sample_constant_offsets_leave_all_outputs_unchanged,
        test_raw_and_precentered_inputs_have_identical_outputs,
        test_flat_sample_is_gated_out,
        test_flat_samples_do_not_change_selected_mean,
        test_gradient_is_finite_and_nonzero,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
