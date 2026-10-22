"""CPU invariants for native-loss second-order residual representations."""
from __future__ import annotations

import os
import sys
import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import hadamard_shape_regularizer as hd  # noqa: E402
import finetune_lbfgs as driver  # noqa: E402
from mode_balanced_regularizer import compute_mode_balanced_loss  # noqa: E402
from translation_tangent_regularizer import (  # noqa: E402
    compute_phase_mechanism_loss,
    compute_translation_tangent_loss,
)

jax.config.update("jax_enable_x64", True)
ArrayFn = Callable[[jax.Array], jax.Array]


def _check_gradient(native: ArrayFn, residual: ArrayFn, point: jax.Array,
                    *, atol: float = 2e-12) -> None:
    expected = jax.grad(native)(point)
    actual = jax.grad(lambda value: .5 * jnp.sum(jnp.abs(residual(value)) ** 2))(point)
    assert bool(jnp.all(jnp.isfinite(expected)) & jnp.all(jnp.isfinite(actual)))
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=atol)


def _fields() -> tuple[jax.Array, ...]:
    x = jnp.arange(32, dtype=jnp.float64) * (2 * jnp.pi / 32)
    k = jnp.fft.fftfreq(x.size, d=1 / x.size)
    amplitudes = jnp.array([.0, .06, .13])[:, None]
    eta = amplitudes * (jnp.cos(2 * x) + .2 * jnp.sin(3 * x))
    xi = (.1 + amplitudes) * (jnp.sin(2 * x) + .3 * jnp.cos(4 * x))
    target = (.2 + amplitudes) * (jnp.cos(2 * x) - .1 * jnp.sin(4 * x))
    return x, k, eta, xi, target, jnp.array([.3, .7, 1.2])


def test_mode_balanced_zero_cap_and_tie() -> None:
    target = 2 * jnp.array([[1., 0., -1., 0., 1., 0., -1., 0.]])
    zeros, depth = jnp.zeros_like(target), jnp.array([.5])
    k = jnp.arange(5, dtype=jnp.float64)

    def native(amplitude: jax.Array) -> jax.Array:
        return compute_mode_balanced_loss(
            zeros, (1 + amplitude) * target, target, depth, k, denominator_eps=0.,
        )

    def residual(amplitude: jax.Array) -> jax.Array:
        return compute_mode_balanced_loss(
            zeros, (1 + amplitude) * target, target, depth, k, denominator_eps=0.,
            return_irls_residuals=True,
        )

    for amplitude in (0., .3, 10., 12.):
        point = jnp.array(amplitude)
        _check_gradient(native, residual, point)
        derivative = (2 * amplitude / np.sqrt(1 + amplitude**2)
                      if amplitude < 10 else 10 / np.sqrt(101) if amplitude == 10 else 0.)
        np.testing.assert_allclose(jax.grad(native)(point), derivative, rtol=2e-13, atol=1e-14)
    assert native(jnp.array(12.)) > 0  # IRLS requires a constant offset above the cap.
    np.testing.assert_array_equal(residual(jnp.array(12.)), 0.)
    np.testing.assert_array_equal(residual(jnp.array(0.)), 0.)


def test_mode_balanced_fp32_inactive_rows_remain_finite() -> None:
    zeros = jnp.zeros((2, 16), dtype=jnp.float32)
    k = jnp.arange(9, dtype=jnp.float32)
    depth = jnp.array([.2, 1.], dtype=jnp.float32)

    def native(prediction: jax.Array) -> jax.Array:
        return compute_mode_balanced_loss(zeros, prediction, zeros, depth, k)

    def residual(prediction: jax.Array) -> jax.Array:
        return compute_mode_balanced_loss(zeros, prediction, zeros, depth, k, return_irls_residuals=True)

    for prediction in (zeros, jnp.arange(32, dtype=jnp.float32).reshape(2, 16)):
        _check_gradient(native, residual, prediction)
        np.testing.assert_array_equal(residual(prediction), 0.)


def test_mode_balanced_weights_and_dispersion() -> None:
    x, k, eta, _, target, depth = _fields()
    prediction = target + .03 * jnp.sin(3 * x) + .02 * jnp.cos(4 * x)
    for dispersion in (False, True):
        options: dict[str, Any] = dict(k_max=6., sample_weight=jnp.array([0., .3, 2.]),
                                      dispersion_weighting=dispersion)

        def native(value: jax.Array) -> jax.Array:
            return compute_mode_balanced_loss(eta, value, target, depth, k[:17], **options)

        def residual(value: jax.Array) -> jax.Array:
            return compute_mode_balanced_loss(
                eta, value, target, depth, k[:17], return_irls_residuals=True, **options)
        _check_gradient(native, residual, prediction)
        np.testing.assert_array_equal(jax.grad(native)(prediction)[0], 0.)
        np.testing.assert_array_equal(native(prediction), compute_mode_balanced_loss(
            eta, prediction, target, depth, k[:17], return_irls_residuals=False, **options))


def test_tangent_selected_signed_and_empty() -> None:
    x, k, eta, _, target, depth = _fields()
    error = .02 * jnp.sin(2 * x) + .01 * jnp.cos(3 * x)
    for norm in ("squared", "charbonnier"):
        options: dict[str, Any] = dict(sample_mask=jnp.array([True, True, False]), loss_norm=norm)

        def native(value: jax.Array) -> jax.Array:
            return compute_translation_tangent_loss(eta, value, target, depth, k, **options)[0]

        def residual(value: jax.Array) -> jax.Array:
            return compute_translation_tangent_loss(
                eta, value, target, depth, k, return_irls_residuals=True, **options)[0]
        _check_gradient(native, residual, target + error)
        _check_gradient(native, residual, target)
        returned, count = compute_translation_tangent_loss(
            eta, target + error, target, depth, k, return_irls_residuals=True, **options)
        assert count == 1
        np.testing.assert_allclose(returned, -residual(target - error), rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(returned[jnp.array([0, 2])], 0.)
        if norm == "squared":
            np.testing.assert_allclose(.5 * jnp.sum(returned**2), native(target + error), rtol=1e-12)
    def empty(value: jax.Array) -> jax.Array:
        return compute_translation_tangent_loss(
            eta, value, target, depth, k, sample_mask=jnp.zeros(3, dtype=bool))[0]

    def empty_residual(value: jax.Array) -> jax.Array:
        return compute_translation_tangent_loss(
            eta, value, target, depth, k, sample_mask=jnp.zeros(3, dtype=bool),
            return_irls_residuals=True)[0]
    _check_gradient(empty, empty_residual, target + error)
    np.testing.assert_array_equal(empty_residual(target + error), 0.)


def test_tangent_flat_rows_have_finite_forward_curvature() -> None:
    for dtype, tolerance in ((jnp.float32, 2e-5), (jnp.float64, 1e-10)):
        x, k, eta, _, target, depth = map(lambda value: value.astype(dtype), _fields())
        prediction = target + .02 * jnp.sin(2 * x)

        def residual(value: jax.Array) -> jax.Array:
            return compute_translation_tangent_loss(
                eta, value, target, depth, k, return_irls_residuals=True,
            )[0]

        v = jnp.broadcast_to(jnp.cos(2 * x) + .3 * jnp.sin(3 * x), prediction.shape)
        w = jnp.broadcast_to(jnp.sin(2 * x) - .2 * jnp.cos(4 * x), prediction.shape)
        _, jv = jax.jvp(residual, (prediction,), (v,))
        gv = driver.normal_product(residual, prediction, v)
        gw = driver.normal_product(residual, prediction, w)
        assert all(bool(jnp.all(jnp.isfinite(value))) for value in (jv, gv, gw))
        for value in (jv, gv, gw):
            np.testing.assert_array_equal(value[0], 0.)
        energy = jnp.sum(jv**2)
        assert energy > 0
        np.testing.assert_allclose(jnp.sum(v * gv), energy, rtol=tolerance)
        symmetry_scale = jnp.maximum(jnp.linalg.norm(v) * jnp.linalg.norm(gw),
                                     jnp.linalg.norm(w) * jnp.linalg.norm(gv))
        assert jnp.abs(jnp.sum(v * gw) - jnp.sum(w * gv)) <= tolerance * symmetry_scale


def test_momentum_full_batch_denominator() -> None:
    x, k, eta, xi, target, depth = _fields()
    prediction = target + .03 * jnp.cos(3 * x) + .01 * jnp.sin(2 * x)

    def native(value: jax.Array) -> jax.Array:
        loss, count = compute_phase_mechanism_loss(
            eta, xi, value, target, depth, k, mechanism="momentum", horizon=200.)
        return loss * count / eta.shape[0]

    def residual(value: jax.Array) -> jax.Array:
        result, count = compute_phase_mechanism_loss(
            eta, xi, value, target, depth, k, mechanism="momentum", horizon=200.,
            return_irls_residuals=True)
        return result * jnp.sqrt(count / eta.shape[0])

    _check_gradient(native, residual, prediction)
    _check_gradient(native, residual, target)
    selected_loss, count = compute_phase_mechanism_loss(
        eta[1:], xi[1:], prediction[1:], target[1:], depth[1:], k,
        mechanism="momentum", horizon=200.)
    assert count == 2
    np.testing.assert_allclose(native(prediction), selected_loss * 2 / 3, rtol=1e-12)
    np.testing.assert_array_equal(jax.grad(native)(prediction)[0], 0.)


def test_hadamard_includes_forcing_denominator_derivative() -> None:
    _, k, eta, xi, _, depth = _fields()

    def apply(variables: dict[str, Any], inputs: jax.Array, _: jax.Array) -> jax.Array:
        parameters = variables["params"]
        surface, potential = inputs[..., 0], inputs[..., 1]
        derivative = jnp.fft.ifft(1j * k * jnp.fft.fft(potential, axis=-1), axis=-1).real
        return (parameters[0] * potential + parameters[1] * surface * potential
                + parameters[2] * derivative)[..., None]

    options: dict[str, Any] = dict(
        rng=jax.random.PRNGKey(7), apply_fn=apply, eta_phys=eta, xi_phys=xi,
        batch_depth_local=jnp.log(depth)[:, None],
        norm_inputs_fn=lambda surface, potential: jnp.stack((surface, potential), axis=-1),
        denorm_targets_fn=lambda value: value, k=k, dtype=jnp.float64,
        k_max=7., fd_step_min=.002, fd_step_max=.002)

    def native(value: jax.Array) -> jax.Array:
        return hd.compute_hadamard_loss(model_params=value, **options)

    def residual(value: jax.Array) -> jax.Array:
        return hd.compute_hadamard_loss(model_params=value, return_irls_residuals=True, **options)
    point = jnp.array([1.1, .3, .2])
    _check_gradient(native, residual, point, atol=1e-11)
    np.testing.assert_allclose(.5 * jnp.sum(jnp.abs(residual(point))**2), native(point), rtol=1e-12)
    expected = jax.grad(native)(point)
    energy = hd.projected_sobolev_energy
    detach = iter((False, True))

    def frozen_forcing(field: jax.Array, modes: jax.Array, cutoff: float,
                       order: Literal[0, 1], *, return_weighted_spectrum: bool = False) -> jax.Array:
        value = energy(field, modes, cutoff, order, return_weighted_spectrum=return_weighted_spectrum)
        return jax.lax.stop_gradient(value) if next(detach) else value

    with patch.object(hd, "projected_sobolev_energy", frozen_forcing):
        incorrect = jax.grad(native)(point)
    assert np.linalg.norm(np.asarray(expected - incorrect)) > .01 * np.linalg.norm(np.asarray(expected))


def test_native_factory_teacher_and_hd_cadence() -> None:
    spec = importlib.util.spec_from_file_location(
        "second_order_test_trainer", Path(__file__).resolve().parents[1] / "1d_dno_fno_jax_ema_grad.py")
    assert spec is not None and spec.loader is not None
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    args = trainer.build_parser().parse_args(["--dataset", "unused", "--model", "cs_dno"])
    vars(args).update(norm="scale", data_normalization="target", data_k_max=6.,
                      translation_tangent_scope="tanaka", translation_tangent_weight=10.,
                      translation_tangent_loss_norm="squared", phase_mechanism="all_momentum",
                      phase_mechanism_weight=.002574349830754761, mode_balanced_weight=6.,
                      mode_balanced_k_max=6., hadamard_mode="self", hadamard_microbatch=8,
                      hadamard_k_max=6., hadamard_weight=.04)
    stats = {"feature_absmax": [.25, .5], "target_absmax": .5}
    normalize, _, denormalize = trainer.make_normalizers(stats, "scale", dtype=np.dtype(np.float64))
    x, _, _, _, _, _ = _fields()
    amplitudes = jnp.linspace(.03, .11, 16)[:, None]
    eta = amplitudes * (jnp.cos(2 * x) + .3 * jnp.sin(5 * x))
    xi = (.2 + amplitudes) * (jnp.sin(2 * x) + .2 * jnp.cos(4 * x))
    depth = jnp.log(jnp.linspace(.15, 7., 16))[:, None]
    families = jnp.tile(jnp.arange(1, 5), 4)

    def apply(variables: dict[str, Any], inputs: jax.Array, _: jax.Array,
              *, return_baseline: bool = False) -> Any:
        parameters = variables["params"]
        surface, potential = inputs[..., 0], inputs[..., 1]
        baseline = (.2 * potential)[..., None]
        correction = (parameters[0] * surface * potential
                      + parameters[1] * surface**2 * potential)[..., None]
        prediction = baseline + correction
        return (prediction, baseline, baseline, correction) if return_baseline else prediction

    point = jnp.array([.09, .025])
    target = denormalize(apply({"params": point + .01}, normalize(eta, xi), depth))[..., 0]
    batch = (eta, xi, target, depth, families)
    teacher = (eta[:8] + .005 * jnp.cos(3 * x), xi[:8] + .003 * jnp.sin(4 * x),
               target[:8] + .002 * jnp.cos(2 * x), depth[:8], families[:8])
    components, hadamard = trainer.build_loss_functions(
        args, None, apply, 32, 2 * np.pi, stats, jnp.float64, jnp.float64, 1)
    weights = jnp.array([1., 6., 10., args.phase_mechanism_weight])
    key = jax.random.PRNGKey(17)

    def native(value: jax.Array) -> jax.Array:
        clean = jax.vmap(lambda _: components(value, *batch), axis_name="batch")(jnp.arange(1))
        generated = jax.vmap(lambda _: components(value, *teacher), axis_name="batch")(jnp.arange(1))
        clean_loss = jnp.dot(jnp.stack(tuple(field[0] for field in clean[:4])), weights)
        generated_loss = jnp.dot(jnp.stack(tuple(field[0] for field in generated[:4])), weights)
        # Sixteen identical B16 clean batches; only one has separately averaged B8 teacher/HD.
        return clean_loss + generated_loss / 16 + .04 / 16 * hadamard(key, value, eta, xi, depth)

    def residual(value: jax.Array) -> jax.Array:
        return jnp.concatenate([
            jnp.sqrt(fraction) * driver.batch_residuals(
                value, batch, teacher, key, components=components, hadamard=hadamard,
                coefficients=weights, teacher_weight=1., hadamard_weight=.04,
                hd_active=active, teacher_active=active,
            ) for active, fraction in ((False, 15 / 16), (True, 1 / 16))
        ])

    _check_gradient(native, residual, point, atol=1e-10)

    def real_residual(value: jax.Array) -> jax.Array:
        result = residual(value)
        return jnp.concatenate((result.real, result.imag))

    vector, push = jax.linearize(real_residual, point)
    _, pull = jax.vjp(real_residual, point)
    np.testing.assert_allclose(pull(vector)[0], jax.grad(native)(point), rtol=2e-10, atol=1e-10)
    v, w = jnp.array([.3, -.7]), jnp.array([-.4, .2])
    gv, gw = pull(push(v))[0], pull(push(w))[0]
    np.testing.assert_allclose(driver.normal_product(real_residual, point, v), gv,
                               rtol=2e-11, atol=1e-11)
    np.testing.assert_allclose(v @ gw, w @ gv, rtol=2e-11, atol=1e-11)
    np.testing.assert_allclose(v @ gv, jnp.sum(push(v)**2), rtol=2e-11, atol=1e-11)
    assert v @ gv >= 0
    zero_targets = denormalize(apply({"params": point}, normalize(eta, xi), depth))[..., 0]
    zero_batch = (eta, xi, zero_targets, depth, families)

    def data_native(value: jax.Array) -> jax.Array:
        return jax.vmap(lambda _: components(value, *zero_batch), axis_name="batch")(jnp.arange(1))[0][0]

    def data_residual(value: jax.Array) -> jax.Array:
        return jax.vmap(lambda _: components(value, *zero_batch, return_irls_residuals=True),
                        axis_name="batch")(jnp.arange(1))[0][0]

    _check_gradient(data_native, data_residual, point)
    np.testing.assert_array_equal(data_residual(point), 0.)


def test_lm_native_acceptance_and_limits() -> None:
    for kind in ("least_squares", "robust", "nonfinite", "raised", "all_invalid",
                 "attempt_cap", "cg_cap", "curvature_nan"):
        initial = np.array([.1, 2.])
        accepted, ema = initial.copy(), initial.copy()
        accepted_values: list[float] = []
        centers: list[np.ndarray] = []

        def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
            invalid = abs(value[0]) > 3 or kind == "all_invalid" and not np.array_equal(
                value, initial.astype(np.float32).astype(float))
            if kind in ("nonfinite", "raised", "all_invalid") and invalid:
                if kind == "raised":
                    raise FloatingPointError("Rejected numerical trial")
                return np.nan, np.full_like(value, np.nan)
            residual = np.array([value[0]**2 - 1, value[1] - .5])
            derivative = np.array([2 * value[0], 1.])
            if kind == "robust":
                norm = np.sqrt(residual**2 + .1**2)
                return float(np.sum(norm - .1)), derivative * residual / norm
            return float(.5 * residual @ residual), derivative * residual

        def factory(value: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
            centers.append(value.copy())
            if kind == "curvature_nan":
                return lambda vector: np.full_like(vector, np.nan)
            diagonal = np.array([4 * value[0]**2, 1.])
            if kind == "robust":
                diagonal /= np.sqrt(np.array([value[0]**2 - 1, value[1] - .5])**2 + .1**2)
            return lambda vector: diagonal * vector

        def accept(value: np.ndarray, scalar: float, evaluations: int) -> None:
            nonlocal accepted, ema
            accepted, ema = value.copy(), .995 * ema + .005 * value
            accepted_values.append(scalar)

        options: dict[str, Any] = dict(maxaccepted=16, maxcg=8, maxattempts=40, seconds=5.,
                                      initial_damping=1e-4, cg_rtol=1e-10)
        if kind == "attempt_cap":
            options["maxattempts"] = 1
        if kind == "cg_cap":
            options.update(maxcg=1, maxaccepted=1)
        result = driver.run_lm(objective, factory, lambda value: value.astype(np.float32).astype(float),
                               initial, accept, **options)
        initial_value = objective(initial.astype(np.float32).astype(float))[0]
        assert np.all(np.diff([initial_value, *accepted_values]) < 0)
        assert len(accepted_values) == result["accepted_steps"] <= options["maxaccepted"]
        assert len(result["attempts"]) <= options["maxattempts"]
        assert result["cg_iterations"] <= len(result["attempts"]) * options["maxcg"]
        assert len(centers) <= result["accepted_steps"] + 1
        if kind in ("attempt_cap", "all_invalid", "curvature_nan"):
            assert not accepted_values
            np.testing.assert_array_equal(accepted, initial)
            np.testing.assert_array_equal(ema, initial)
        elif kind == "cg_cap":
            assert accepted_values and any(row.get("cg_info") == 1 for row in result["attempts"])
        else:
            assert result["accepted_objective"] < 1e-9
            np.testing.assert_allclose(accepted, [1., .5], atol=1e-5)
            assert any(not row["accepted"] for row in result["attempts"])


def test_lm_rounded_prediction_and_time_cap() -> None:
    accepted: list[np.ndarray] = []

    def quadratic(value: np.ndarray) -> tuple[float, np.ndarray]:
        return float(.5 * (value[0] - .36)**2), value - .36

    def identity(value: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        return lambda vector: vector

    result = driver.run_lm(
        quadratic, identity, lambda value: np.round(value, 1), np.zeros(1),
        lambda value, scalar, count: accepted.append(value),
        maxaccepted=1, maxcg=2, maxattempts=2, seconds=5.)
    row = result["attempts"][0]
    np.testing.assert_allclose(row["predicted_reduction"], row["actual_reduction"], rtol=1e-14)
    np.testing.assert_array_equal(accepted[0], [.4])
    accepted.clear()
    clock = [0.]

    def timed_objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        clock[0] += .6
        return quadratic(value)

    with patch.object(driver, "perf_counter", side_effect=lambda: clock[0]):
        result = driver.run_lm(
            timed_objective, identity, lambda value: value, np.zeros(1),
            lambda value, scalar, count: accepted.append(value),
            maxaccepted=1, maxcg=2, maxattempts=2, seconds=1.)
    assert not accepted and result["termination"] == "LM wall-time cap reached"
    assert result["evaluations"] == 2


def test_lbfgs_rejects_late_objective_result() -> None:
    clock = [0.]
    accepted: list[np.ndarray] = []

    def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        clock[0] += .6
        return float(.5 * value @ value), value

    with patch.object(driver, "perf_counter", side_effect=lambda: clock[0]):
        result = driver.run_lbfgs(
            objective, np.ones(1), .5,
            lambda value, scalar, count: accepted.append(value),
            maxiter=2, maxls=4, seconds=1.)
    # The second call reaches the exact minimizer, but returns after the deadline.
    assert result["evaluations"] == 2
    assert result["termination"] == "L-BFGS wall-time cap reached"
    assert not accepted


if __name__ == "__main__":
    for name, function in tuple(globals().items()):
        if name.startswith("test_"):
            function()
            print(f"PASS {name}", flush=True)
