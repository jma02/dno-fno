"""CPU invariants for translation-tangent and coupled modal-phase objectives."""

from __future__ import annotations

from functools import partial
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from translation_tangent_regularizer import (  # noqa: E402
    _centered_error,
    compute_coherent_rollout_loss,
    compute_modal_phase_loss,
    compute_phase_mechanism_loss,
    compute_translation_tangent_loss,
)
from solver.solvers.time_integrator import (  # noqa: E402
    dealiased_zakharov_xi_rhs,
    spectral_dx,
)


jax.config.update("jax_enable_x64", True)


def test_subtract_before_centering_preserves_losses_and_gradients() -> None:
    for dtype in (jnp.float32, jnp.float64):
        n = 128
        x = jnp.arange(n, dtype=dtype) * (2.0 * jnp.pi / n)
        k = jnp.fft.fftfreq(n, d=1.0 / n).astype(dtype)
        eta = (0.03 * jnp.cos(3.0 * x) + 0.01 * jnp.sin(5.0 * x))[None, :]
        xi = 0.04 * jnp.sin(3.0 * x)[None, :]
        target = 0.1 + 0.05 * jnp.cos(3.0 * x)[None, :]
        prediction = target + (0.01 * jnp.sin(5.0 * x) + 0.003 * jnp.cos(3.0 * x))[None, :] + 0.2
        depth = jnp.asarray([0.2], dtype=dtype)
        for objective in (
            partial(compute_translation_tangent_loss, eta),
            partial(compute_modal_phase_loss, eta, xi),
            partial(compute_phase_mechanism_loss, eta, xi, mechanism="momentum"),
        ):
            evaluate = jax.jit(
                jax.value_and_grad(objective, has_aux=True),
                static_argnames=("subtract_before_centering",),
            )
            if dtype == jnp.float64:
                (expected, count), expected_gradient = evaluate(
                    prediction, target, depth, k, subtract_before_centering=False,
                )
                (actual, actual_count), gradient = evaluate(
                    prediction, target, depth, k, subtract_before_centering=True,
                )
                np.testing.assert_array_equal(actual_count, count)
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-17)
                np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-10, atol=1e-14)
            for scale in (0.0, 1e-6):
                (loss, count), gradient = evaluate(
                    target + scale * (prediction - target), target, depth, k,
                    subtract_before_centering=True,
                )
                assert np.isfinite(np.asarray(loss))
                assert np.all(np.isfinite(gradient))
                np.testing.assert_array_equal(count, 1.0)
                if scale == 0.0:
                    np.testing.assert_array_equal(loss, 0.0)
                    np.testing.assert_array_equal(gradient, 0.0)


def test_subtract_before_centering_reduces_float32_residual_roundoff() -> None:
    rng = np.random.default_rng(42)
    target = jnp.asarray(rng.normal(0.1, 0.2, (4, 256)), dtype=jnp.float32)
    prediction = target + jnp.asarray(rng.normal(0.0, 1e-6, target.shape), dtype=jnp.float32)
    difference64 = prediction.astype(jnp.float64) - target.astype(jnp.float64)
    reference = difference64 - jnp.mean(difference64, axis=-1, keepdims=True)
    evaluate = jax.jit(_centered_error, static_argnames=("subtract_before_centering",))
    old_error, old_target = evaluate(prediction, target, subtract_before_centering=False)
    new_error, new_target = evaluate(prediction, target, subtract_before_centering=True)
    np.testing.assert_array_equal(new_target, old_target)
    old_roundoff = np.linalg.norm(np.asarray(old_error, dtype=np.float64) - reference)
    new_roundoff = np.linalg.norm(np.asarray(new_error, dtype=np.float64) - reference)
    assert new_roundoff < 0.01 * old_roundoff


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


def test_sample_mask_matches_selected_loss_and_gradient() -> None:
    eta, prediction, target, depth, k = _batched_inputs()
    selected = jnp.asarray([0, 2])
    mask = jnp.asarray([True, False, True])
    loss_and_gradient = jax.jit(
        jax.value_and_grad(compute_translation_tangent_loss, argnums=1, has_aux=True)
    )
    for dtype in (jnp.float32, jnp.float64):
        inputs = tuple(
            value.astype(dtype) for value in (eta, prediction, target, depth)
        )
        (loss, count), gradient = loss_and_gradient(
            *inputs, k.astype(dtype), sample_mask=mask
        )
        (expected_loss, expected_count), expected_gradient = loss_and_gradient(
            *(value[selected] for value in inputs), k.astype(dtype)
        )
        tolerance = 1e-6 if dtype == jnp.float32 else 1e-12
        np.testing.assert_allclose(loss, expected_loss, rtol=tolerance, atol=1e-14)
        np.testing.assert_array_equal(count, expected_count)
        np.testing.assert_allclose(
            gradient[selected], expected_gradient, rtol=tolerance, atol=1e-14
        )
        np.testing.assert_array_equal(gradient[1], 0.0)


def test_mask_with_no_nonflat_samples_has_zero_loss_and_gradient() -> None:
    eta, prediction, target, depth, k = _batched_inputs()
    eta = eta.at[2].set(0.0)
    loss_and_gradient = jax.jit(
        jax.value_and_grad(compute_translation_tangent_loss, argnums=1, has_aux=True)
    )
    for mask in (jnp.zeros(3, dtype=jnp.bool_), jnp.asarray([False, False, True])):
        (loss, count), gradient = loss_and_gradient(
            eta, prediction, target, depth, k, sample_mask=mask
        )
        np.testing.assert_array_equal((loss, count), (0.0, 0.0))
        np.testing.assert_array_equal(gradient, 0.0)


def test_all_selected_mask_matches_default_exactly() -> None:
    eta, prediction, target, depth, k = _batched_inputs()
    eta = eta.at[2].set(0.0)
    loss_and_gradient = jax.jit(
        jax.value_and_grad(compute_translation_tangent_loss, argnums=1, has_aux=True)
    )
    for dtype in (jnp.float32, jnp.float64):
        inputs = tuple(
            value.astype(dtype) for value in (eta, prediction, target, depth, k)
        )
        (loss, count), gradient = loss_and_gradient(*inputs)
        (masked_loss, masked_count), masked_gradient = loss_and_gradient(
            *inputs, sample_mask=jnp.ones(3, dtype=jnp.bool_)
        )
        np.testing.assert_array_equal((masked_loss, masked_count), (loss, count))
        np.testing.assert_array_equal(masked_gradient, gradient)


def test_global_relative_loss_matches_traveling_gain() -> None:
    loss_and_gradient = jax.jit(
        jax.value_and_grad(compute_translation_tangent_loss, argnums=1, has_aux=True)
    )
    for dtype in (jnp.float32, jnp.float64):
        x = jnp.arange(256, dtype=dtype) * (2.0 * jnp.pi / 256)
        k = jnp.fft.fftfreq(256, d=1.0 / 256).astype(dtype)
        tolerance = 2e-5 if dtype == jnp.float32 else 1e-12
        for mode, speed, amplitude, gain in (
            (1, 0.4, 1.0, 0.03), (1, -0.4, 1.0, 0.03),
            (4, 0.4, 1.0, 0.03), (1, 0.8, 1.0, 0.03),
            (1, 0.4, 0.02, 0.03), (3, -0.15, 2.0, -0.03),
        ):
            eta = amplitude * jnp.sin(mode * x)[None, :]
            target = -speed * amplitude * mode * jnp.cos(mode * x)[None, :]
            (loss, count), gradient = loss_and_gradient(
                eta, (1.0 + gain) * target, target, jnp.asarray([0.2], dtype=dtype), k,
                global_relative_mask=jnp.ones(1, dtype=jnp.bool_),
            )
            np.testing.assert_allclose(loss, gain**2 / 1.001, rtol=tolerance)
            np.testing.assert_array_equal(count, 1.0)
            assert bool(jnp.all(jnp.isfinite(gradient)))
            assert float(jnp.linalg.norm(gradient)) > 0.0


def test_mixed_global_local_rows_match_individual_losses_and_gradients() -> None:
    eta, prediction, target, depth, k = _batched_inputs()
    prediction = prediction + 0.03 * target
    global_mask = jnp.asarray([True, False, True])
    loss_and_gradient = jax.jit(
        jax.value_and_grad(compute_translation_tangent_loss, argnums=1, has_aux=True)
    )
    for dtype in (jnp.float32, jnp.float64):
        inputs = tuple(value.astype(dtype) for value in (eta, prediction, target, depth))
        (loss, count), gradient = loss_and_gradient(
            *inputs, k.astype(dtype), global_relative_mask=global_mask
        )
        individual = [
            loss_and_gradient(
                *(value[index : index + 1] for value in inputs), k.astype(dtype),
                global_relative_mask=global_mask[index : index + 1],
            )
            for index in range(3)
        ]
        tolerance = 2e-6 if dtype == jnp.float32 else 1e-12
        np.testing.assert_array_equal(count, 3.0)
        np.testing.assert_allclose(
            loss, np.mean([result[0][0] for result in individual]), rtol=tolerance
        )
        np.testing.assert_allclose(
            gradient, np.concatenate([result[1] for result in individual]) / 3.0,
            rtol=tolerance, atol=1e-12,
        )
        assert bool(jnp.all(jnp.isfinite(gradient)))
        (exact_loss, _), exact_gradient = loss_and_gradient(
            inputs[0], inputs[2], inputs[2], inputs[3], k.astype(dtype),
            global_relative_mask=global_mask,
        )
        np.testing.assert_array_equal(exact_loss, 0.0)
        np.testing.assert_array_equal(exact_gradient, 0.0)


def test_charbonnier_selected_mean_matches_rowwise_formula() -> None:
    eta, _, target, depth, k = _batched_inputs()
    x = jnp.arange(eta.shape[-1], dtype=eta.dtype) * (2.0 * jnp.pi / eta.shape[-1])
    modes = jnp.asarray([1.0, 2.0, 3.0], dtype=eta.dtype)[:, None]
    gains = jnp.asarray([0.03, -0.05, 0.07], dtype=eta.dtype)[:, None]
    prediction = target * (1.0 + gains) + 0.02 * jnp.sin((modes + 2.0) * x[None, :])
    global_mask = jnp.asarray([True, False, True])
    sample_mask = jnp.asarray([True, True, False])
    loss_and_gradient = jax.jit(
        jax.value_and_grad(compute_translation_tangent_loss, argnums=1, has_aux=True),
        static_argnames=("loss_norm",),
    )
    for dtype in (jnp.float32, jnp.float64):
        inputs = tuple(
            value.astype(dtype) for value in (eta, prediction, target, depth)
        )
        kk = k.astype(dtype)
        (loss, count), gradient = loss_and_gradient(
            *inputs,
            kk,
            sample_mask=sample_mask,
            global_relative_mask=global_mask,
            loss_norm="charbonnier",
        )
        np.testing.assert_array_equal(count, 2.0)
        raws = []
        expected = []
        row_gradients = []
        for index in (0, 1):
            row_eta, row_prediction, row_target, row_depth = (
                value[index : index + 1] for value in inputs
            )
            raw, raw_count = compute_translation_tangent_loss(
                row_eta,
                row_prediction,
                row_target,
                row_depth,
                kk,
                global_relative_mask=global_mask[index : index + 1],
            )
            np.testing.assert_array_equal(raw_count, 1.0)
            raw_scalar = float(raw)
            raws.append(raw_scalar)
            expected.append(raw_scalar / (np.sqrt(raw_scalar + 1e-14) + 1e-7))
            (row_loss, _), row_gradient = loss_and_gradient(
                row_eta,
                row_prediction,
                row_target,
                row_depth,
                kk,
                global_relative_mask=global_mask[index : index + 1],
                loss_norm="charbonnier",
            )
            np.testing.assert_allclose(
                row_loss, expected[-1], rtol=2e-6 if dtype == jnp.float32 else 1e-12
            )
            row_gradients.append(np.asarray(row_gradient[0]))
        tolerance = 2e-5 if dtype == jnp.float32 else 1e-12
        atol = 1e-8 if dtype == jnp.float32 else 1e-14
        np.testing.assert_allclose(
            loss, np.mean(expected), rtol=tolerance, atol=atol
        )
        for index in (0, 1):
            np.testing.assert_allclose(
                gradient[index], row_gradients[index] / 2.0, rtol=tolerance, atol=atol
            )
        np.testing.assert_array_equal(gradient[2], 0.0)
        mean_raw_transform = np.mean(raws) / (np.sqrt(np.mean(raws) + 1e-14) + 1e-7)
        assert not np.isclose(float(loss), mean_raw_transform, rtol=1e-3)


def test_charbonnier_degenerate_cases_are_finite_or_zero() -> None:
    loss_and_gradient = jax.jit(
        jax.value_and_grad(compute_translation_tangent_loss, argnums=1, has_aux=True),
        static_argnames=("loss_norm",),
    )
    for dtype in (jnp.float32, jnp.float64):
        eta, prediction, target, depth, k = _batched_inputs()
        inputs = tuple(
            value.astype(dtype) for value in (eta, prediction, target, depth)
        )
        kk = k.astype(dtype)

        (loss, _), gradient = loss_and_gradient(
            inputs[0], inputs[2], inputs[2], inputs[3], kk, loss_norm="charbonnier"
        )
        np.testing.assert_array_equal(loss, 0.0)
        np.testing.assert_array_equal(gradient, 0.0)

        flat = jnp.zeros_like(inputs[0])
        (loss, count), gradient = loss_and_gradient(
            flat, inputs[1], inputs[2], inputs[3], kk, loss_norm="charbonnier"
        )
        np.testing.assert_array_equal((loss, count), (0.0, 0.0))
        np.testing.assert_array_equal(gradient, 0.0)

        mixed_eta = inputs[0].at[2].set(0.0)
        (loss, count), gradient = loss_and_gradient(
            mixed_eta, inputs[1], inputs[2], inputs[3], kk, loss_norm="charbonnier"
        )
        np.testing.assert_array_equal(count, 2.0)
        assert np.isfinite(np.asarray(loss))
        assert bool(jnp.all(jnp.isfinite(gradient)))
        np.testing.assert_array_equal(gradient[2], 0.0)
        assert float(jnp.linalg.norm(gradient[:2])) > 0.0

        (loss, count), gradient = loss_and_gradient(
            *inputs,
            kk,
            sample_mask=jnp.zeros(3, dtype=jnp.bool_),
            loss_norm="charbonnier",
        )
        np.testing.assert_array_equal((loss, count), (0.0, 0.0))
        np.testing.assert_array_equal(gradient, 0.0)

        x = jnp.arange(256, dtype=dtype) * (2.0 * jnp.pi / 256)
        eta_degenerate = jnp.stack((jnp.sin(x), jnp.sin(x)))
        target_degenerate = jnp.stack((jnp.zeros_like(x), jnp.sin(2.0 * x)))
        prediction_degenerate = target_degenerate + 0.02 * jnp.cos(x)[None, :]
        (loss, count), gradient = loss_and_gradient(
            eta_degenerate,
            prediction_degenerate,
            target_degenerate,
            jnp.full(2, 0.2, dtype=dtype),
            kk,
            global_relative_mask=jnp.ones(2, dtype=jnp.bool_),
            loss_norm="charbonnier",
        )
        np.testing.assert_array_equal(count, 2.0)
        assert np.isfinite(np.asarray(loss))
        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert float(jnp.linalg.norm(gradient)) > 0.0


def test_charbonnier_traveling_gain_is_nearly_linear_in_gain() -> None:
    loss_and_gradient = jax.jit(
        jax.value_and_grad(compute_translation_tangent_loss, argnums=1, has_aux=True),
        static_argnames=("loss_norm",),
    )
    for dtype in (jnp.float32, jnp.float64):
        x = jnp.arange(256, dtype=dtype) * (2.0 * jnp.pi / 256)
        k = jnp.fft.fftfreq(256, d=1.0 / 256).astype(dtype)
        squared_tolerance = 2e-5 if dtype == jnp.float32 else 1e-12
        for mode in (1, 3):
            eta = jnp.sin(mode * x)[None, :]
            target = -0.4 * mode * jnp.cos(mode * x)[None, :]
            depth = jnp.asarray([0.2], dtype=dtype)
            values = []
            gradient_norms = []
            for gain in (0.01, 0.02):
                prediction = (1.0 + gain) * target
                (squared, _), _ = loss_and_gradient(
                    eta,
                    prediction,
                    target,
                    depth,
                    k,
                    global_relative_mask=jnp.ones(1, dtype=jnp.bool_),
                )
                np.testing.assert_allclose(
                    squared, gain**2 / 1.001, rtol=squared_tolerance
                )
                (loss, _), gradient = loss_and_gradient(
                    eta,
                    prediction,
                    target,
                    depth,
                    k,
                    global_relative_mask=jnp.ones(1, dtype=jnp.bool_),
                    loss_norm="charbonnier",
                )
                raw = float(squared)
                expected = raw / (np.sqrt(raw + 1e-14) + 1e-7)
                np.testing.assert_allclose(loss, expected, rtol=2e-4)
                values.append(float(loss))
                gradient_norms.append(float(jnp.linalg.norm(gradient)))
            np.testing.assert_allclose(values[1], 2.0 * values[0], rtol=2e-4)
            np.testing.assert_allclose(gradient_norms[1], gradient_norms[0], rtol=2e-4)
            assert all(norm > 0.0 for norm in gradient_norms)
            assert all(np.isfinite(value) for value in values)


def test_degenerate_global_references_fall_back_to_finite_local_loss() -> None:
    loss_and_gradient = jax.jit(
        jax.value_and_grad(compute_translation_tangent_loss, argnums=1, has_aux=True)
    )
    for dtype in (jnp.float32, jnp.float64):
        x = jnp.arange(256, dtype=dtype) * (2.0 * jnp.pi / 256)
        k = jnp.fft.fftfreq(256, d=1.0 / 256).astype(dtype)
        eta = jnp.broadcast_to(jnp.sin(x), (4, 256)).at[2].set(0.0)
        target = jnp.stack((
            jnp.zeros_like(x), jnp.sin(2.0 * x), jnp.cos(x),
            jnp.sin(2.0 * x) + 0.001 * jnp.cos(x),
        ))
        prediction = target + 0.02 * jnp.cos(x)[None, :]
        inputs = (eta, prediction, target, jnp.full(4, 0.2, dtype=dtype), k)
        local, local_gradient = loss_and_gradient(*inputs)
        fallback, gradient = loss_and_gradient(
            *inputs, global_relative_mask=jnp.ones(4, dtype=jnp.bool_)
        )
        tolerance = 2e-6 if dtype == jnp.float32 else 1e-12
        np.testing.assert_allclose(fallback, local, rtol=tolerance)
        np.testing.assert_allclose(gradient, local_gradient, rtol=tolerance, atol=1e-12)
        np.testing.assert_array_equal(fallback[1], 3.0)
        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert np.isfinite(np.asarray(fallback)).all()
        np.testing.assert_array_equal(gradient[2], 0.0)


def test_modal_exact_and_flat_rows_have_finite_zero_gradients() -> None:
    loss_and_gradient = jax.jit(
        jax.value_and_grad(compute_modal_phase_loss, argnums=2, has_aux=True),
        static_argnames=("loss_norm",),
    )
    for dtype in (jnp.float32, jnp.float64):
        x = jnp.arange(128, dtype=dtype) * (2.0 * jnp.pi / 128)
        k = jnp.fft.fftfreq(128, d=1.0 / 128).astype(dtype)
        eta = jnp.stack((0.03 * jnp.cos(3.0 * x), jnp.zeros_like(x)))
        xi = jnp.stack((0.04 * jnp.sin(3.0 * x), jnp.zeros_like(x)))
        target = 0.05 * jnp.cos(3.0 * x)[None, :] * jnp.ones((2, 1), dtype=dtype)
        depth = jnp.asarray([0.2, 0.3], dtype=dtype)
        for norm in ("squared", "charbonnier"):
            (loss, count), gradient = loss_and_gradient(
                eta, xi, target, target, depth, k, loss_norm=norm
            )
            np.testing.assert_array_equal((loss, count), (0.0, 1.0))
            np.testing.assert_array_equal(gradient, 0.0)
            assert bool(jnp.all(jnp.isfinite(gradient)))
            (loss, count), gradient = loss_and_gradient(
                eta, xi, target * 1.1, target, depth, k,
                sample_mask=jnp.asarray([False, True]), loss_norm=norm,
            )
            np.testing.assert_array_equal((loss, count), (0.0, 0.0))
            np.testing.assert_array_equal(gradient, 0.0)


def test_modal_standing_wave_retains_both_opposite_phase_rates() -> None:
    # At an eta turning point the two traveling coordinates remain nonzero,
    # and their signed phase defects cancel if summed before squaring.
    for dtype in (jnp.float32, jnp.float64):
        n, mode = 128, 3
        x = jnp.arange(n, dtype=dtype) * (2.0 * jnp.pi / n)
        k = jnp.fft.fftfreq(n, d=1.0 / n).astype(dtype)
        eta = jnp.zeros((1, n), dtype=dtype)
        xi = 0.04 * jnp.cos(mode * x)[None, :]
        depth = jnp.asarray([0.2], dtype=dtype)
        g0 = mode * jnp.tanh(mode * depth[0])
        target = g0 * xi
        denominator_eps = 1e-3

        def gain_loss(gain: jax.Array) -> jax.Array:
            return compute_modal_phase_loss(
                eta, xi, (1.0 + gain) * target, target, depth, k,
                k_max=8.0, denominator_eps=denominator_eps,
            )[0]

        gain = jnp.asarray(0.03, dtype=dtype)
        loss, gradient = jax.jit(jax.value_and_grad(gain_loss))(gain)
        tolerance = 2e-5 if dtype == jnp.float32 else 1e-12
        np.testing.assert_allclose(loss, gain**2 / (1.0 + denominator_eps)**2, rtol=tolerance)
        np.testing.assert_allclose(gradient, 2.0 * gain / (1.0 + denominator_eps)**2, rtol=tolerance)
        _, count = compute_modal_phase_loss(eta, xi, target * 1.03, target, depth, k)
    np.testing.assert_array_equal(count, 1.0)


def test_modal_excludes_dc_nyquist_and_out_of_band_states() -> None:
    n = 128
    k = jnp.fft.fftfreq(n, d=1.0 / n)
    eta = 0.1 + jnp.tile(jnp.asarray([0.02, -0.02]), n // 2)[None, :]
    xi = jnp.tile(jnp.asarray([0.03, 0.0, -0.03, 0.0]), n // 4)[None, :]
    x = jnp.arange(n, dtype=jnp.float64) * (2.0 * jnp.pi / n)
    target = jnp.sin(3.0 * x)[None, :]
    (loss, count), gradient = jax.value_and_grad(
        compute_modal_phase_loss, argnums=2, has_aux=True
    )(
        eta, xi, 1.1 * target, target, jnp.asarray([0.2]), k,
        k_max=8.0, loss_norm="charbonnier",
    )
    np.testing.assert_array_equal((loss, count), (0.0, 0.0))
    np.testing.assert_array_equal(gradient, 0.0)


def test_modal_radial_error_has_no_instantaneous_phase_defect() -> None:
    n = 128
    x = jnp.arange(n, dtype=jnp.float64) * (2.0 * jnp.pi / n)
    k = jnp.fft.fftfreq(n, d=1.0 / n)
    eta = 0.03 * jnp.cos(3.0 * x)[None, :]
    zero = jnp.zeros_like(eta)
    loss, count = compute_modal_phase_loss(
        eta, zero, 0.2 * eta, zero, jnp.asarray([0.2]), k, k_max=8.0
    )
    np.testing.assert_allclose(loss, 0.0, atol=1e-28)
    np.testing.assert_array_equal(count, 1.0)


def test_modal_matches_dealiased_rhs_and_normal_coordinate_phase_difference() -> None:
    n = 128
    x = jnp.arange(n, dtype=jnp.float64) * (2.0 * jnp.pi / n)
    k = jnp.fft.fftfreq(n, d=1.0 / n)
    eta = (0.04 * jnp.cos(3.0 * x) + 0.02 * jnp.sin(5.0 * x))[None, :]
    xi = (0.03 * jnp.sin(3.0 * x) + 0.01 * jnp.cos(5.0 * x))[None, :]
    # Out-of-band q modes differ by mode 3: their product must survive into
    # the retained xi equation even though direct eta_t errors are filtered.
    target = (0.02 * jnp.cos(3.0 * x) + 0.04 * jnp.sin(17.0 * x))[None, :]
    direction = (0.013 * jnp.sin(5.0 * x) + 0.02 * jnp.sin(20.0 * x))[None, :]
    prediction = target + direction
    depth, gravity, floor_coefficient = 0.2, 1.7, 1e-3
    eta_x, xi_x = (spectral_dx(field, k) for field in (eta, xi))
    rhs_target, rhs_prediction = (
        -gravity * eta + dealiased_zakharov_xi_rhs(eta_x, xi_x, field)
        for field in (target, prediction)
    )
    modes = np.arange(1, 9)
    g0 = modes * np.tanh(modes * depth)
    omega = np.sqrt(gravity * g0)
    eta_hat, xi_hat, target_hat, prediction_hat, target_rhs_hat, prediction_rhs_hat = (
        np.fft.rfft(np.asarray(field[0]), norm="forward")[modes]
        for field in (eta, xi, target, prediction, rhs_target, rhs_prediction)
    )
    signs = np.asarray([1.0, -1.0])[:, None]
    coordinates = np.sqrt(gravity) * eta_hat + signs * 1j * np.sqrt(g0) * xi_hat
    rhs_coordinates = [
        np.sqrt(gravity) * q_hat + signs * 1j * np.sqrt(g0) * r_hat
        for q_hat, r_hat in ((target_hat, target_rhs_hat), (prediction_hat, prediction_rhs_hat))
    ]
    energy = np.abs(coordinates)**2
    denominator = energy + floor_coefficient * np.max(energy) + 1e-30
    numerator = np.imag(np.conj(coordinates) * (rhs_coordinates[1] - rhs_coordinates[0]))
    expected = np.sum(energy * (numerator / (omega * denominator))**2) / np.sum(energy)
    actual, _ = compute_modal_phase_loss(
        eta, xi, prediction, target, jnp.asarray([depth]), k,
        k_max=8.0, gravity=gravity, denominator_eps=floor_coefficient,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-18)

    # On occupied modes, the formula is the derivative of the actual complex
    # angle, not the real (amplitude) component of the RHS difference.
    occupied = energy > 1e-12 * np.max(energy)
    dt = 1e-6
    phase_rates = [
        (
            np.angle(1.0 + dt * rhs[occupied] / coordinates[occupied])
            - np.angle(1.0 - dt * rhs[occupied] / coordinates[occupied])
        ) / (2.0 * dt)
        for rhs in rhs_coordinates
    ]
    np.testing.assert_allclose(
        phase_rates[1] - phase_rates[0], numerator[occupied] / energy[occupied],
        rtol=2e-5, atol=1e-10,
    )

    def directional_loss(scale: jax.Array) -> jax.Array:
        return compute_modal_phase_loss(
            eta, xi, target + scale * direction, target, jnp.asarray([depth]), k,
            k_max=8.0, gravity=gravity, loss_norm="charbonnier",
        )[0]

    scale = jnp.asarray(1.0)
    derivative = jax.grad(directional_loss)(scale)
    finite_difference = (directional_loss(scale + 1e-5) - directional_loss(scale - 1e-5)) / 2e-5
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-7, atol=1e-10)
    assert float(actual) > 1e-6


def test_modal_masks_centering_and_translation_preserve_selected_mean() -> None:
    n = 128
    x = jnp.arange(n, dtype=jnp.float64) * (2.0 * jnp.pi / n)
    k = jnp.fft.fftfreq(n, d=1.0 / n)
    modes = jnp.asarray([2.0, 3.0, 5.0])[:, None]
    eta = 0.03 * jnp.cos(modes * x)
    xi = 0.04 * jnp.sin(modes * x)
    target = 0.05 * jnp.sin(modes * x)
    prediction = target * jnp.asarray([1.1, 1.2, 1.3])[:, None]
    depth = jnp.asarray([0.1, 0.2, 0.3])
    mask = jnp.asarray([True, False, True])
    evaluate = jax.jit(
        jax.value_and_grad(compute_modal_phase_loss, argnums=2, has_aux=True),
        static_argnames=("loss_norm",),
    )
    for norm in ("squared", "charbonnier"):
        (loss, count), gradient = evaluate(
            eta, xi, prediction, target, depth, k, sample_mask=mask, loss_norm=norm
        )
        individual_losses = []
        for index in (0, 2):
            (single, _), single_gradient = evaluate(
                eta[index:index + 1], xi[index:index + 1], prediction[index:index + 1],
                target[index:index + 1], depth[index:index + 1], k, loss_norm=norm,
            )
            individual_losses.append(single)
            np.testing.assert_allclose(gradient[index], single_gradient[0] / 2.0, rtol=1e-11, atol=1e-14)
        np.testing.assert_allclose(loss, np.mean(individual_losses), rtol=1e-12)
        np.testing.assert_array_equal(count, 2.0)
        np.testing.assert_array_equal(gradient[1], 0.0)
        offsets = jnp.asarray([0.7, -0.3, 0.2])[:, None]
        shifted_inputs = [
            jnp.roll(field, 19, axis=-1)
            for field in (eta, xi + offsets, prediction + offsets, target - 2.0 * offsets)
        ]
        (shifted_loss, shifted_count), shifted_gradient = evaluate(
            *shifted_inputs, depth, k, sample_mask=mask, loss_norm=norm
        )
        np.testing.assert_allclose((shifted_loss, shifted_count), (loss, count), rtol=1e-11)
        np.testing.assert_allclose(shifted_gradient, jnp.roll(gradient, 19, axis=-1), rtol=1e-10, atol=1e-13)


def test_carrier_phase_has_coupled_half_gain_and_correct_horizon() -> None:
    n, mode, gain, horizon = 128, 4, .03, 20.
    for dtype in (jnp.float32, jnp.float64):
        x = jnp.arange(n, dtype=dtype) * (2 * jnp.pi / n)
        k = jnp.fft.fftfreq(n, d=1. / n).astype(dtype)
        depth = jnp.asarray([.2], dtype=dtype)
        g0 = mode * jnp.tanh(mode * depth[0])
        eta = 1e-7 * jnp.cos(mode * x)[None]
        for direction in (-1., 1.):
            xi = direction * 1e-7 / jnp.sqrt(g0) * jnp.sin(mode * x)[None]
            target = g0 * xi
            loss, count = compute_phase_mechanism_loss(
                eta, xi, (1 + gain) * target, target, depth, k,
                mechanism="carrier", horizon=horizon,
            )
            expected_squared = (horizon * gain * float(jnp.sqrt(g0)) / 2)**2
            expected = expected_squared / (np.sqrt(expected_squared + 1e-14) + 1e-7)
            np.testing.assert_allclose(loss, expected, rtol=2e-5 if dtype == jnp.float32 else 1e-11)
            np.testing.assert_array_equal(count, 1.)


def test_sideband_relative_phase_cancels_translation_but_detects_differential_drift() -> None:
    n, horizon, dc = 128, 200., .002
    x = jnp.arange(n, dtype=jnp.float64) * (2 * jnp.pi / n)
    k = jnp.fft.fftfreq(n, d=1. / n)
    modes = jnp.asarray([4, 5, 6])
    amplitudes = 1e-8 * jnp.asarray([.1, 1., .1])
    g0 = modes * jnp.tanh(.2 * modes)
    eta = jnp.sum(amplitudes[:, None] * jnp.cos(modes[:, None] * x), axis=0)[None]
    xi = jnp.sum((amplitudes / jnp.sqrt(g0))[:, None] * jnp.sin(modes[:, None] * x), axis=0)[None]
    target = jnp.sum((amplitudes * jnp.sqrt(g0))[:, None] * jnp.sin(modes[:, None] * x), axis=0)[None]
    for mismatch in (0., .1):
        rates = dc * modes.astype(jnp.float64).at[-1].multiply(1 + mismatch)
        error = jnp.sum((2 * rates * amplitudes)[:, None] * jnp.sin(modes[:, None] * x), axis=0)[None]
        loss, count = compute_phase_mechanism_loss(
            eta, xi, target + error, target, jnp.asarray([.2]), k,
            mechanism="sideband", carrier_mode=jnp.asarray([5]), sideband_offset=jnp.asarray([1]), horizon=horizon,
        )
        squared = (horizon * dc * 6 * mismatch)**2
        expected = squared / (np.sqrt(squared + 1e-14) + 1e-7)
        np.testing.assert_allclose(loss, expected, rtol=1e-8, atol=1e-18)
        np.testing.assert_array_equal(count, 1.)
    # Fixed sidebands are not replaced by whichever neighbor has nonzero energy.
    one_mode_eta = 1e-8 * jnp.cos(5 * x)[None]
    loss, count = compute_phase_mechanism_loss(
        one_mode_eta, jnp.zeros_like(one_mode_eta), target + error, target, jnp.asarray([.2]), k,
        mechanism="sideband", carrier_mode=jnp.asarray([5]), sideband_offset=jnp.asarray([1]),
    )
    np.testing.assert_array_equal((loss, count), (0., 0.))


def test_momentum_defect_matches_dealiased_rhs_including_high_mode_products() -> None:
    n = 128
    x = jnp.arange(n, dtype=jnp.float64) * (2 * jnp.pi / n)
    k = jnp.fft.fftfreq(n, d=1. / n)
    eta = (.03 * jnp.sin(3 * x) + .01 * jnp.cos(5 * x))[None]
    xi = (.04 * jnp.sin(3 * x) - .02 * jnp.cos(4 * x))[None]
    target = (.05 * jnp.sin(3 * x) + .08 * jnp.sin(17 * x))[None]
    error = (.01 * jnp.cos(3 * x) + .02 * jnp.sin(20 * x))[None]
    eta_x, xi_x = (spectral_dx(field, k) for field in (eta, xi))
    xi_defect = (dealiased_zakharov_xi_rhs(eta_x, xi_x, target + error)
                 - dealiased_zakharov_xi_rhs(eta_x, xi_x, target))
    residual = 200 * jnp.mean(xi_x * error - eta_x * xi_defect) / jnp.sqrt(jnp.mean(eta**2) * jnp.mean(xi_x**2))
    expected = residual**2 / (jnp.sqrt(residual**2 + 1e-14) + 1e-7)
    loss, count = compute_phase_mechanism_loss(
        eta, xi, target + error, target, jnp.asarray([.2]), k,
        mechanism="momentum", horizon=200, k_max=8,
    )
    np.testing.assert_allclose(loss, expected, rtol=1e-12)
    np.testing.assert_array_equal(count, 1.)


def test_mechanism_losses_masks_gauge_translation_and_gradients() -> None:
    n = 128
    for dtype in (jnp.float32, jnp.float64):
        x = jnp.arange(n, dtype=dtype) * (2 * jnp.pi / n)
        k = jnp.fft.fftfreq(n, d=1. / n).astype(dtype)
        eta = (.03 * jnp.cos(5 * x) + .004 * jnp.cos(4 * x) + .003 * jnp.cos(6 * x))[None]
        xi = .02 * jnp.sin(5 * x)[None] + .1 * eta
        target = .05 * jnp.sin(5 * x)[None]
        prediction = target + .01 * jnp.cos(4 * x)[None] + .002 * jnp.cos(5 * x)[None]
        depth = jnp.asarray([.2], dtype=dtype)
        for mechanism in ("carrier", "sideband", "momentum"):
            evaluate = jax.jit(jax.value_and_grad(compute_phase_mechanism_loss, argnums=2, has_aux=True),
                               static_argnames=("mechanism",))
            options: dict[str, Any] = dict(mechanism=mechanism, carrier_mode=jnp.asarray([5]), sideband_offset=jnp.asarray([1]))
            (loss, count), gradient = evaluate(eta, xi, prediction, target, depth, k, **options)
            assert float(loss) > 0 and bool(jnp.isfinite(gradient).all()) and float(jnp.linalg.norm(gradient)) > 0
            np.testing.assert_array_equal(count, 1.)
            offsets = (0., .7, -.3, .5)
            shifted = tuple(jnp.roll(field + offset, 13, axis=-1)
                            for field, offset in zip((eta, xi, prediction, target), offsets))
            (shifted_loss, shifted_count), shifted_gradient = evaluate(*shifted, depth, k, **options)
            tolerance = 5e-4 if dtype == jnp.float32 else 1e-10
            np.testing.assert_allclose((shifted_loss, shifted_count), (loss, count), rtol=tolerance, atol=1e-10)
            np.testing.assert_allclose(shifted_gradient, jnp.roll(gradient, 13, axis=-1), rtol=tolerance, atol=1e-5 if dtype == jnp.float32 else 1e-11)
            for inputs, mask in (((eta, xi, target, target), None),
                                 ((eta, xi, prediction, target), jnp.asarray([False])),
                                 ((jnp.zeros_like(eta), jnp.zeros_like(xi), prediction, target), None)):
                (zero, selected), zero_gradient = evaluate(*inputs, depth, k, sample_mask=mask, **options)
                np.testing.assert_array_equal(zero, 0.)
                np.testing.assert_array_equal(zero_gradient, 0.)
                if inputs[0] is not eta or mask is not None:
                    np.testing.assert_array_equal(selected, 0.)
            if dtype == jnp.float64:
                direction = prediction - target
                def objective(scale: float | jax.Array) -> jax.Array:
                    return compute_phase_mechanism_loss(eta, xi, target + scale * direction,
                                                        target, depth, k, **options)[0]
                analytical = jax.grad(objective)(1.)
                finite = (objective(1. + 1e-5) - objective(1. - 1e-5)) / 2e-5
                np.testing.assert_allclose(analytical, finite, rtol=1e-6, atol=1e-10)


def test_mechanism_batched_selection_equals_individual_rows() -> None:
    n = 128
    x = jnp.arange(n, dtype=jnp.float64) * (2 * jnp.pi / n)
    k = jnp.fft.fftfreq(n, d=1. / n)
    carrier, offset = jnp.asarray([5, 7, 9]), jnp.asarray([1, 2, 3])
    modes = jnp.stack((carrier - offset, carrier, carrier + offset), axis=-1)
    eta = jnp.sum(jnp.asarray([.004, .03, .003])[None, :, None] * jnp.cos(modes[..., None] * x), axis=1)
    xi = jnp.sum(.02 * jnp.sin(modes[..., None] * x), axis=1) * jnp.asarray([1., -1., -1.])[:, None]
    target = .05 * jnp.sin(carrier[:, None] * x)
    prediction = target + .01 * jnp.cos((carrier - offset)[:, None] * x) + .002 * jnp.cos(carrier[:, None] * x)
    depth = jnp.asarray([.2, .3, .1])
    evaluate = jax.jit(jax.value_and_grad(compute_phase_mechanism_loss, argnums=2, has_aux=True), static_argnames=("mechanism",))
    for mechanism in ("carrier", "sideband", "momentum"):
        options: dict[str, Any] = dict(mechanism=mechanism, carrier_mode=carrier, sideband_offset=offset)
        (loss, count), gradient = evaluate(eta, xi, prediction, target, depth, k,
                                           sample_mask=jnp.asarray([True, False, True]), **options)
        singles = []
        for row in (0, 2):
            (single, _), single_gradient = evaluate(
                *(field[row:row + 1] for field in (eta, xi, prediction, target, depth)), k,
                mechanism=mechanism, carrier_mode=carrier[row:row + 1], sideband_offset=offset[row:row + 1],
            )
            singles.append(single)
            np.testing.assert_allclose(gradient[row], single_gradient[0] / 2, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(loss, np.mean(singles), rtol=1e-11)
        np.testing.assert_array_equal(count, 2.)
        np.testing.assert_array_equal(gradient[1], 0.)


def _coherent_inputs() -> tuple[jax.Array, ...]:
    n = 64
    x = jnp.arange(n, dtype=jnp.float64) * (2 * jnp.pi / n)
    phase = jnp.arange(12, dtype=jnp.float64).reshape(2, 2, 3, 1) / 17
    eta = .03 * jnp.cos(3 * x + phase)
    xi = .04 * jnp.sin(2 * x - phase)
    target = .1 * jnp.sin(4 * x + phase) + .05 * jnp.cos(19 * x)
    prediction = target + .02 * jnp.cos(23 * x + phase) + .005 * jnp.sin(6 * x)
    return (
        eta, xi, prediction, target, jnp.array([.1, .3, 1.2]),
        2 * jnp.fft.fftfreq(n, d=1 / n),
        jnp.array([[.02, .07], [.14, .23]]), jnp.array([[.03, .07], [.04, .11]]),
        eta[0, 0], xi[0, 0], jnp.array([True, False, True]),
    )


def _coherent_equation_reference(inputs: tuple[jax.Array, ...], gravity: float, k_max: float) -> float:
    eta, xi, pred, target, depth, k, times, weights, eta0, xi0, valid = map(np.asarray, inputs)
    n = eta.shape[-1]
    ka = abs(k[:n // 2 + 1])
    pred = pred - pred.mean(axis=-1, keepdims=True)
    target = target - target.mean(axis=-1, keepdims=True)
    spectra = np.fft.rfft(np.stack((eta, xi, pred, target), axis=-2), norm="forward")
    spectra[..., -1] = 0
    eta_pad = np.fft.irfft(spectra[..., 0, :], n=2 * n, norm="forward")
    eta_x, xi_x = (
        np.fft.irfft(1j * ka * spectra[..., index, :], n=2 * n, norm="forward")
        for index in (0, 1)
    )
    xi_rhs = [
        -gravity * eta_pad - .5 * xi_x**2
        + .5 * (np.fft.irfft(spectra[..., index, :], n=2 * n, norm="forward") + eta_x * xi_x)**2
        / (1 + eta_x**2)
        for index in (2, 3)
    ]
    dq = np.sqrt(gravity) * np.fft.rfft(pred - target, norm="forward")
    lam = ka * np.tanh(depth[:, None] * ka)
    dxi = np.sqrt(lam) * np.fft.rfft(xi_rhs[0] - xi_rhs[1], norm="forward")[..., :n // 2 + 1]
    angle = times[..., None, None] * np.sqrt(gravity * lam)
    c, s = np.cos(angle), np.sin(angle)
    accumulated = np.sum(weights[..., None, None, None] * np.stack((c * dq - s * dxi, s * dq + c * dxi), axis=-2), axis=(0, 1))
    band = (ka > 0) & (ka <= k_max)
    band[-1] = False
    norm = np.linalg.norm(accumulated[..., band], axis=(-2, -1))
    initial_energy = gravity * abs(np.fft.rfft(eta0, norm="forward"))**2 + lam * abs(np.fft.rfft(xi0, norm="forward"))**2
    denominator = np.sqrt(np.maximum(np.sum(initial_energy[..., band], axis=-1), 1e-30))
    return float(np.mean((norm / denominator)[valid])) if np.any(valid) else 0.0


def test_coherent_loss_matches_padded_equations_and_translation() -> None:
    inputs = _coherent_inputs()
    evaluate = jax.jit(jax.value_and_grad(compute_coherent_rollout_loss, argnums=2),
                       static_argnames=("gravity", "k_max"))
    loss, gradient = evaluate(*inputs, gravity=2.3, k_max=16.)
    np.testing.assert_allclose(loss, _coherent_equation_reference(inputs, 2.3, 16.), rtol=2e-12)
    shifted = tuple(jnp.roll(value, 7, axis=-1) if index in (0, 1, 2, 3, 8, 9) else value
                    for index, value in enumerate(inputs))
    shifted_loss, shifted_gradient = evaluate(*shifted, gravity=2.3, k_max=16.)
    np.testing.assert_allclose(shifted_loss, loss, rtol=2e-12)
    np.testing.assert_allclose(shifted_gradient, jnp.roll(gradient, 7, axis=-1), rtol=2e-11, atol=1e-14)
    # Separate constants and Nyquist components cannot enter the retained equations.
    nyquist = (-1.)**jnp.arange(inputs[0].shape[-1])
    offsets = tuple(value + (2. + .3 * nyquist if index == 2 else -.7 + .1 * nyquist)
                    if index in (2, 3) else value for index, value in enumerate(inputs))
    np.testing.assert_allclose(evaluate(*offsets, gravity=2.3, k_max=16.)[0], loss, rtol=2e-12)


def test_coherent_loss_stops_path_gradients_and_matches_finite_difference() -> None:
    inputs = _coherent_inputs()
    evaluate = jax.jit(jax.value_and_grad(compute_coherent_rollout_loss, argnums=tuple(range(10))),
                       static_argnames=("gravity", "k_max"))
    loss, gradients = evaluate(*inputs, gravity=2.3, k_max=16.)
    assert loss > 0
    for index, gradient in enumerate(gradients):
        assert np.isfinite(np.asarray(gradient)).all()
        if index != 2:
            np.testing.assert_array_equal(gradient, 0.)
    np.testing.assert_array_equal(gradients[2][:, :, 1], 0.)
    assert np.linalg.norm(gradients[2]) > 0
    direction = inputs[2] - inputs[3]
    epsilon = 1e-5
    numerical = []
    for sign in (-1, 1):
        changed = (*inputs[:2], inputs[2] + sign * epsilon * direction, *inputs[3:])
        numerical.append(_coherent_equation_reference(changed, 2.3, 16.))
    np.testing.assert_allclose(jnp.sum(gradients[2] * direction),
                               (numerical[1] - numerical[0]) / (2 * epsilon), rtol=2e-7)
    # The selected-row mean and gradient are independent of sanitized invalid rows.
    singles = []
    for row in (0, 2):
        single = tuple(value[:, :, row:row + 1] if index < 4 else
                       value[row:row + 1] if index in (4, 8, 9, 10) else value
                       for index, value in enumerate(inputs))
        value, grads = evaluate(*single, gravity=2.3, k_max=16.)
        singles.append(value)
        np.testing.assert_allclose(gradients[2][:, :, row], grads[2][:, :, 0] / 2, rtol=2e-12, atol=1e-15)
    np.testing.assert_allclose(loss, np.mean(singles), rtol=2e-12)


def test_coherent_loss_zero_mask_and_nonfinite_contracts() -> None:
    inputs = _coherent_inputs()
    evaluate = jax.jit(jax.value_and_grad(compute_coherent_rollout_loss, argnums=2))
    for dtype in (jnp.float32, jnp.float64):
        cast_inputs = tuple(value.astype(dtype) if value.dtype != jnp.bool_ else value for value in inputs)
        exact = (*cast_inputs[:2], cast_inputs[3], *cast_inputs[3:])
        zero = tuple(jnp.zeros_like(value) if index in (0, 1, 2, 3, 8, 9) else value
                     for index, value in enumerate(cast_inputs))
        masked = (*cast_inputs[:-1], jnp.zeros_like(cast_inputs[-1]))
        for case in (exact, zero, masked):
            loss, gradient = evaluate(*case)
            assert loss.dtype == jnp.float64 and gradient.dtype == dtype
            np.testing.assert_array_equal(loss, 0.)
            np.testing.assert_array_equal(gradient, 0.)
        for nonfinite in (jnp.nan, jnp.inf):
            bad = (*cast_inputs[:2], cast_inputs[2].at[0, 0, 0, 0].set(nonfinite), *cast_inputs[3:])
            loss, gradient = evaluate(*bad)
            assert not np.isfinite(float(loss))
            assert not np.isfinite(np.asarray(gradient)).all()


def test_coherent_loss_transport_cancellation_and_high_mode_coupling() -> None:
    n = 64
    x = jnp.arange(n, dtype=jnp.float64) * (2 * jnp.pi / n)
    k = jnp.fft.fftfreq(n, d=1 / n)
    zero = jnp.zeros((1, 2, 1, n))
    depth, mask = jnp.array([.4]), jnp.array([True])
    prediction = jnp.broadcast_to(.02 * jnp.cos(2 * x), zero.shape)
    initial = (.02 * jnp.cos(2 * x))[None]
    omega = jnp.sqrt(2 * jnp.tanh(2 * depth[0]))
    inputs = (zero, zero, prediction, -prediction, depth, k, jnp.array([[0., jnp.pi / omega]]),
              jnp.array([[.1, .1]]), initial, jnp.zeros_like(initial), mask)
    np.testing.assert_allclose(compute_coherent_rollout_loss(*inputs, k_max=8.), 0., atol=1e-15)
    coherent = (*inputs[:6], jnp.zeros_like(inputs[6]), *inputs[7:])
    np.testing.assert_allclose(compute_coherent_rollout_loss(*coherent, k_max=8.), .4, rtol=1e-12)
    # Both q fields have no retained modes, but their product has difference mode4.
    target = jnp.broadcast_to(.1 * jnp.cos(19 * x), zero.shape)
    prediction = target + .02 * jnp.cos(23 * x)
    coupled = (*coherent[:2], prediction, target, *coherent[4:])
    loss = compute_coherent_rollout_loss(*coupled, k_max=8.)
    assert loss > 1e-4
    np.testing.assert_allclose(loss, _coherent_equation_reference(coupled, 1., 8.), rtol=2e-12)
    projected = (*coupled[:2], zero, zero, *coupled[4:])
    np.testing.assert_array_equal(compute_coherent_rollout_loss(*projected, k_max=8.), 0.)


if __name__ == "__main__":
    tests = (
        test_coherent_loss_matches_padded_equations_and_translation,
        test_coherent_loss_stops_path_gradients_and_matches_finite_difference,
        test_coherent_loss_zero_mask_and_nonfinite_contracts,
        test_coherent_loss_transport_cancellation_and_high_mode_coupling,
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
        test_sample_mask_matches_selected_loss_and_gradient,
        test_mask_with_no_nonflat_samples_has_zero_loss_and_gradient,
        test_all_selected_mask_matches_default_exactly,
        test_global_relative_loss_matches_traveling_gain,
        test_mixed_global_local_rows_match_individual_losses_and_gradients,
        test_charbonnier_selected_mean_matches_rowwise_formula,
        test_charbonnier_degenerate_cases_are_finite_or_zero,
        test_charbonnier_traveling_gain_is_nearly_linear_in_gain,
        test_degenerate_global_references_fall_back_to_finite_local_loss,
        test_modal_exact_and_flat_rows_have_finite_zero_gradients,
        test_modal_standing_wave_retains_both_opposite_phase_rates,
        test_modal_excludes_dc_nyquist_and_out_of_band_states,
        test_modal_radial_error_has_no_instantaneous_phase_defect,
        test_modal_matches_dealiased_rhs_and_normal_coordinate_phase_difference,
        test_modal_masks_centering_and_translation_preserve_selected_mean,
        test_carrier_phase_has_coupled_half_gain_and_correct_horizon,
        test_sideband_relative_phase_cancels_translation_but_detects_differential_drift,
        test_momentum_defect_matches_dealiased_rhs_including_high_mode_products,
        test_mechanism_losses_masks_gauge_translation_and_gradients,
        test_mechanism_batched_selection_equals_individual_rows,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
