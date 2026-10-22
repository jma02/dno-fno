"""CPU regressions for data normalization, gradients, and family reductions."""

import argparse
import ast
from collections.abc import Callable
from functools import partial
from itertools import product
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "models" / "fno-jax"))
sys.path.insert(0, str(REPO_ROOT / "models" / "dno-net"))
sys.path.insert(0, str(REPO_ROOT / "train-jax-10m"))

from losses import relative_l2_loss  # noqa: E402
from dno_net_v2 import (  # noqa: E402
    CraigSulemBlock, CraigSulemDNO, CraigSulemTermsResidual, DepthAwareMultiplier,
    PhaseWavenumberMultiplier,
)
from mode_balanced_regularizer import compute_mode_balanced_loss  # noqa: E402
from solver.gen_data.pipeline.dno_target import compute_dno_target  # noqa: E402
from solver.gen_data.pipeline.types import PhysicalFamilyId  # noqa: E402
from solver.solvers.time_integrator import dealiased_zakharov_xi_rhs  # noqa: E402
from translation_tangent_regularizer import (  # noqa: E402
    compute_coherent_rollout_loss,
    compute_modal_phase_loss,
    compute_phase_mechanism_loss,
    compute_rhs_relative_error,
    compute_translation_tangent_loss,
)

jax.config.update("jax_enable_x64", True)

TrainerLoss = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, tuple[jax.Array, tuple[jax.Array, ...]]],
]


def trainer_loss(
    config: SimpleNamespace, domain_length: float = 2 * np.pi,
    *, full_spectrum_rows: jax.Array | None = None, capture_modal_weights: bool = False,
) -> TrainerLoss:
    # Exercise the trainer's inline arithmetic, not a second implementation.
    config.data_xi_projection_weight = 0.0
    config.data_phase_projection_weight = 0.0
    config.data_modal_residual_floor = getattr(config, "data_modal_residual_floor", 0.0)
    config.subtract_before_fft = getattr(config, "subtract_before_fft", False)
    config.fp64_residual = False
    config.rollout_training_rhs = getattr(config, "rollout_training_rhs", False)
    path = REPO_ROOT / "train-jax-10m" / "1d_dno_fno_jax_ema_grad.py"
    source = ast.parse(path.read_text())
    component = next(node for node in ast.walk(source)
                     if isinstance(node, ast.FunctionDef) and node.name == "compute_loss_components")
    assignments = {node.targets[0].id: i for i, node in enumerate(component.body)
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    function = ast.parse(
        "def losses(predictions: jax.Array, batch_targets: jax.Array, baseline: jax.Array, "
        "m1_baseline: jax.Array, family: jax.Array) "
        "-> tuple[jax.Array, tuple[jax.Array, tuple[jax.Array, ...]]]: pass"
    ).body[0]
    assert isinstance(function, ast.FunctionDef)
    function.body = ast.parse(
        "k_rfft_jax = (2 * jnp.pi / domain_length) * jnp.arange(predictions.shape[1] // 2 + 1)"
    ).body + component.body[assignments["pred_hat"]:assignments["h_phys"]]
    function.body += ast.parse(
        "return data_loss, (full_l2_loss, (modal_weight,))" if capture_modal_weights
        else "return data_loss, (full_l2_loss, family_metrics)"
    ).body
    namespace: dict[str, Any] = {
        "jax": jax, "jnp": jnp, "args": config, "target_residual64": None,
        "full_spectrum_rows": full_spectrum_rows,
        "PhysicalFamilyId": PhysicalFamilyId, "collect_family_metrics": True, "domain_length": domain_length,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["losses"]


def test_exact_fit_and_mixed_batches_have_finite_gradients() -> None:
    differentiate = jax.value_and_grad(relative_l2_loss, argnums=(0, 1))
    for dtype in (jnp.float32, jnp.float64):
        target = jnp.arange(48, dtype=dtype).reshape(3, 16, 1).at[2].set(0)
        for evaluate in (differentiate, jax.jit(differentiate)):
            loss, gradients = evaluate(target, target)
            assert loss == 0.0
            assert loss.dtype == jnp.float64
            for gradient in gradients:
                np.testing.assert_array_equal(gradient, jnp.zeros_like(target))

            prediction = target.at[1].multiply(1.001)
            loss, gradients = evaluate(prediction, target)
            assert loss > 0.0
            for gradient in gradients:
                assert bool(jnp.all(jnp.isfinite(gradient)))
                np.testing.assert_array_equal(gradient[jnp.array([0, 2])], 0.0)
                assert bool(jnp.any(gradient[1] != 0.0))

            assert jnp.isnan(evaluate(target.at[0, 0, 0].set(jnp.nan), target)[0])
            assert not jnp.isfinite(
                evaluate(target.at[0, 0, 0].set(jnp.inf), target)[0]
            )


def test_rhs_energy_error_matches_padded_equations_and_stopped_denominator() -> None:
    nx, gravity = 512, 2.3
    x = jnp.arange(nx, dtype=jnp.float64) * (2 * jnp.pi / nx)
    k = jnp.fft.fftfreq(nx) * nx
    eta = jnp.stack((0.03 * jnp.cos(3 * x), 0.02 * jnp.sin(5 * x)))
    xi = jnp.stack((0.04 * jnp.sin(2 * x), 0.03 * jnp.cos(7 * x)))
    target = 0.1 * jnp.sin(4 * x)[None] + jnp.array([[0.07], [0.11]]) * jnp.cos(137 * x)
    prediction = target + 0.03 * jnp.cos(141 * x) + 0.01 * jnp.sin(6 * x)
    depth = jnp.array([0.07, 9.0])

    def reference(pred: jax.Array, truth: jax.Array, detach: bool = True) -> jax.Array:
        pred, truth = pred.astype(jnp.float64), truth.astype(jnp.float64)
        pred, truth = (q - jnp.mean(q, axis=-1, keepdims=True) for q in (pred, truth))
        ka = jnp.arange(nx // 2 + 1, dtype=jnp.float64)
        ex, xx = (
            jnp.fft.irfft((1j * ka * jnp.fft.rfft(f)).at[..., -1].set(0), n=nx)
            for f in (eta, xi)
        )
        true_rhs, pred_rhs = (
            -gravity * eta + dealiased_zakharov_xi_rhs(ex, xx, q) for q in (truth, pred)
        )
        q_error, rhs_error, q_hat, rhs_hat = (
            jnp.fft.rfft(f, norm="forward")[:, 1:129]
            for f in (pred - truth, pred_rhs - true_rhs, truth, true_rhs)
        )
        g0 = ka[None, 1:129] * jnp.tanh(depth[:, None] * ka[None, 1:129])
        denominator = jnp.sqrt(jnp.sum(gravity * jnp.abs(q_hat)**2 + g0 * jnp.abs(rhs_hat)**2, axis=-1))
        if detach:
            denominator = jax.lax.stop_gradient(denominator)
        return jnp.sqrt(jnp.sum(gravity * jnp.abs(q_error)**2 + g0 * jnp.abs(rhs_error)**2, axis=-1)) / denominator

    actual = partial(compute_rhs_relative_error, eta, xi, depth=depth, k=k, gravity=gravity)
    np.testing.assert_allclose(actual(prediction, target), reference(prediction, target), rtol=3e-13)
    for argument in (0, 1):
        actual_gradient = jax.grad(lambda p, t: jnp.sum(actual(p, t)), argnums=argument)(prediction, target)
        expected_gradient = jax.grad(lambda p, t: jnp.sum(reference(p, t)), argnums=argument)(prediction, target)
        np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=2e-11, atol=2e-13)
    unstopped = jax.grad(lambda t: jnp.sum(reference(prediction, t, False)))(target)
    assert float(jnp.linalg.norm(unstopped - expected_gradient)) > 1e-3
    direction = jnp.broadcast_to(jnp.cos(141 * x) + 0.4 * jnp.cos(137 * x), target.shape)
    step = 1e-5
    difference = jnp.sum(actual(prediction + step * direction, target)
                         - actual(prediction - step * direction, target)) / (2 * step)
    derivative = jnp.sum(jax.grad(lambda p: jnp.sum(actual(p, target)))(prediction) * direction)
    np.testing.assert_allclose(derivative, difference, rtol=2e-7)
    # Two out-of-band errors beat into retained modes even with a zero q target:
    # this requires the quadratic 0.5*error**2 term, not just a linearized defect.
    high_prediction = 0.02 * direction
    zero_target = jnp.zeros_like(target)
    high_loss = actual(high_prediction, zero_target)
    np.testing.assert_allclose(high_loss, reference(high_prediction, zero_target), rtol=5e-12)
    assert bool(jnp.all(high_loss > 1e-4))
    np.testing.assert_allclose(actual(prediction + 2.0, target - 3.0), actual(prediction, target), rtol=2e-13)
    # Nyquist-only prediction changes are absent from both retained eta_t and
    # the padded Bernoulli products, even if the requested cutoff includes it.
    nyquist = jnp.cos((nx // 2) * x)[None]
    projected = compute_rhs_relative_error(eta, xi, target + nyquist, target, depth, k, k_max=nx)
    np.testing.assert_allclose(projected, 0, atol=2e-14)
    for dtype in (jnp.float32, jnp.float64):
        for truth in (target.astype(dtype), jnp.zeros_like(target, dtype=dtype)):
            evaluate = jax.jit(jax.value_and_grad(lambda p: jnp.sum(
                compute_rhs_relative_error(eta.astype(dtype), xi.astype(dtype), p, truth,
                                           depth.astype(dtype), k.astype(dtype)),
            )))
            value, gradient = evaluate(truth)
            assert value.dtype == jnp.float64 and value == 0
            np.testing.assert_array_equal(gradient, 0)
            assert gradient.dtype == dtype and bool(jnp.all(jnp.isfinite(gradient)))
        rounded = tuple(f.astype(jnp.float32) for f in (eta, xi, prediction, target, depth, k))
        np.testing.assert_array_equal(compute_rhs_relative_error(*rounded),
                                      compute_rhs_relative_error(*(f.astype(jnp.float64) for f in rounded)))
        zero = jnp.zeros_like(target, dtype=dtype)
        value, gradient = jax.value_and_grad(lambda p: jnp.sum(compute_rhs_relative_error(
            zero, zero, p, zero, depth.astype(dtype), k.astype(dtype),
        )))(zero)
        assert value == 0 and bool(jnp.all(jnp.isfinite(gradient)))
        np.testing.assert_array_equal(gradient, 0)


def test_rhs_teacher_flag_requires_full_spectrum_and_active_teacher() -> None:
    main = trainer_function("main")
    names = {"--rollout_training_rhs", "--rollout_training_full_spectrum", "--rollout_training_weight"}
    options: list[ast.stmt] = [
        node for node in ast.walk(main) if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and node.value.args
        and isinstance(node.value.args[0], ast.Constant) and node.value.args[0].value in names
    ]
    checks: list[ast.stmt] = [
        node for node in main.body if isinstance(node, ast.If)
        and ast.unparse(node.test).startswith(("args.rollout_training_rhs and",
                                              "args.rollout_training_full_spectrum and"))
    ]
    parser = argparse.ArgumentParser(add_help=False)
    namespace: dict[str, Any] = {"parser": parser}
    exec(compile(ast.Module(body=options, type_ignores=[]), "rhs_teacher_options", "exec"), namespace)
    assert parser.parse_args([]).rollout_training_rhs is False
    for rhs, full, weight in product((False, True), (False, True), (0.0, 0.7)):
        flags = ["--rollout_training_weight", str(weight)]
        flags += ["--rollout_training_rhs"] if rhs else []
        flags += ["--rollout_training_full_spectrum"] if full else []
        namespace["args"] = parser.parse_args(flags)
        invalid = (rhs and not full) or (full and weight == 0)
        with patch.object(parser, "error", side_effect=ValueError) as error:
            try:
                exec(compile(ast.Module(body=checks, type_ignores=[]), "rhs_teacher_checks", "exec"), namespace)
            except ValueError:
                assert invalid
            assert error.called == invalid


def test_nonlinear_denominator_prioritizes_larger_corrections() -> None:
    config = SimpleNamespace(model="cs_dno", data_normalization="nonlinear", data_k_max=None, residual_floor=1e-6,
                             nonlinear_reference=0.002, nonlinear_weight_mean=1.463313)
    losses = trainer_loss(config)
    evaluate = jax.jit(jax.value_and_grad(losses, argnums=(0, 1, 2, 3), has_aux=True))
    family = jnp.array([PhysicalFamilyId.STOKES])
    ratios = np.array([0.0, 1e-5, 0.001, 0.002, 0.02, 1.0])
    for dtype in (jnp.float32, jnp.float64):
        x = jnp.arange(64, dtype=dtype) * (2 * jnp.pi / 64)
        target = jnp.sin(x)[None, :, None]
        prediction = target + 0.01 * jnp.cos(2 * x)[None, :, None]
        ordinary, ordinary_gradient = jax.jit(jax.value_and_grad(relative_l2_loss))(prediction, target)
        target_hat = np.asarray(jnp.fft.rfft(target, axis=1))
        weights, expected = [], []
        for ratio in ratios:
            baseline = (1 - jnp.asarray(ratio, dtype=dtype)) * target
            residual_hat = (target_hat - np.asarray(jnp.fft.rfft(baseline, axis=1))).astype(np.complex128)
            measured_ratio = np.linalg.norm(residual_hat) / np.linalg.norm(target_hat.astype(np.complex128))
            expected.append((1 + measured_ratio / (measured_ratio + config.nonlinear_reference))
                            / config.nonlinear_weight_mean)
            (weighted, (full, _)), gradients = evaluate(prediction, target, baseline, baseline, family)
            np.testing.assert_array_equal(full, ordinary)
            weight = float(weighted / ordinary)
            weights.append(weight)
            np.testing.assert_allclose(gradients[0], weight * ordinary_gradient, rtol=3e-5, atol=1e-8)
            np.testing.assert_array_equal(gradients[2], 0)
            np.testing.assert_array_equal(gradients[3], 0)
            scaled, _ = losses(7 * prediction, 7 * target, 7 * baseline, 7 * baseline, family)
            np.testing.assert_allclose(scaled, weighted, rtol=1e-5)
        assert np.all(np.diff(weights) > 0)
        np.testing.assert_allclose(weights, expected, rtol=5e-6)
        baselines = (1 - jnp.asarray(ratios, dtype=dtype)[:, None, None]) * target
        mixed, _ = losses(jnp.broadcast_to(prediction, baselines.shape),
                          jnp.broadcast_to(target, baselines.shape), baselines, baselines,
                          jnp.repeat(family, len(ratios)))
        np.testing.assert_allclose(mixed, ordinary * np.mean(weights), rtol=1e-6)
        for scale in (0.0, 1e-14, 1.0):
            for exact in (True, False):
                truth = target * scale
                pred = truth if exact else truth + 1e-8
                (value, (full, _)), gradients = evaluate(pred, truth, truth, truth, family)
                assert bool(jnp.isfinite(value) & jnp.isfinite(full))
                assert all(bool(jnp.all(jnp.isfinite(gradient))) for gradient in gradients)
                if exact:
                    assert value == full == 0
                    for gradient in gradients:
                        np.testing.assert_array_equal(gradient, 0)
        config.nonlinear_weight_mean = 1.0
        np.testing.assert_array_equal(losses(prediction, target, target, target, family)[0], ordinary)
        config.nonlinear_weight_mean = 1.463313
        config.model = "fno"
        np.testing.assert_array_equal(losses(prediction, target, target, target, family)[0], ordinary)
        config.model = "cs_dno"


def test_target_normalization_matches_unweighted_relative_l2() -> None:
    config = SimpleNamespace(model="cs_dno", data_normalization="target", data_k_max=None, residual_floor=1e-6)
    losses = trainer_loss(config)
    evaluate = jax.jit(jax.value_and_grad(losses, argnums=(0, 2, 3), has_aux=True))
    ordinary = jax.jit(jax.value_and_grad(relative_l2_loss))
    family = jnp.array([1, 2, 3, 4])
    for dtype in (jnp.float32, jnp.float64):
        x = jnp.arange(512, dtype=dtype) * (2 * jnp.pi / 512)
        target = jnp.arange(1, 5, dtype=dtype)[:, None, None] * jnp.sin(x)[None, :, None]
        # The old objective includes modes above the phase diagnostic's k=128 cutoff.
        prediction = target + 0.01 * jnp.cos(160 * x)[None, :, None]
        for scale in (0.0, 1e-14, 1.0):
            for pred in (scale * target, scale * prediction):
                truth = scale * target
                (value, (full, _)), gradients = evaluate(pred, truth, 0.7 * truth, 0.9 * truth, family)
                expected, expected_gradient = ordinary(pred, truth)
                np.testing.assert_array_equal(value, expected)
                np.testing.assert_array_equal(full, expected)
                np.testing.assert_array_equal(gradients[0], expected_gradient)
                assert bool(jnp.isfinite(value) & jnp.all(jnp.isfinite(gradients[0])))
                for gradient in gradients[1:]:
                    np.testing.assert_array_equal(gradient, 0)


def test_subtract_before_fft_changes_only_residual_arithmetic() -> None:
    nx = 1024
    x = np.arange(nx) * (2 * np.pi / nx)
    family = jnp.array([PhysicalFamilyId.STOKES])
    for dtype, cutoff, enabled in product((jnp.float32, jnp.float64), (None, 128.0), (False, True)):
        config = SimpleNamespace(model="cs_dno", data_normalization="target", data_k_max=cutoff,
                                 residual_floor=1e-6, subtract_before_fft=enabled)
        losses = trainer_loss(config)
        target = jnp.asarray(np.sin(3 * x) + 0.3 * np.cos(7 * x), dtype=dtype)[None, :, None]
        prediction = target + jnp.asarray(1e-7 * (np.cos(11 * x) + np.sin(160 * x)), dtype=dtype)[None, :, None]
        baseline = 0.9 * target
        target_hat = jnp.fft.rfft(target, axis=1)
        mask = jnp.ones(nx // 2 + 1, bool) if cutoff is None else jnp.arange(nx // 2 + 1) <= cutoff
        denominator = jnp.sqrt(jnp.sum(jnp.abs(target_hat[:, mask].astype(jnp.complex128)) ** 2))

        def expected_loss(pred: jax.Array) -> jax.Array:
            residual = (jnp.fft.rfft(pred - target, axis=1) if enabled
                        else jnp.fft.rfft(pred, axis=1) - target_hat)
            return jnp.sqrt(jnp.sum(jnp.abs(residual[:, mask].astype(jnp.complex128)) ** 2)) / denominator

        evaluate = jax.jit(jax.value_and_grad(losses, has_aux=True))
        (value, (full, _)), gradient = evaluate(prediction, target, baseline, baseline, family)
        expected, expected_gradient = jax.jit(jax.value_and_grad(expected_loss))(prediction)
        np.testing.assert_allclose(value, expected, rtol=1e-12)
        np.testing.assert_allclose(value, expected_loss(prediction), rtol=1e-12)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-6, atol=1e-9)
        if cutoff is None:
            np.testing.assert_array_equal(value, full)
        (zero, (zero_full, _)), zero_gradient = evaluate(target, target, baseline, baseline, family)
        np.testing.assert_array_equal((zero, zero_full), (0.0, 0.0))
        np.testing.assert_array_equal(zero_gradient, jnp.zeros_like(target))
        assert bool(jnp.all(jnp.isfinite(gradient)))
        if dtype == jnp.float32 and enabled:
            # Compare both FFT orders with an FP64 transform of the same rounded fields.
            reference = jnp.fft.rfft(prediction.astype(jnp.float64) - target.astype(jnp.float64), axis=1)
            before = jnp.fft.rfft(prediction - target, axis=1)
            after = jnp.fft.rfft(prediction, axis=1) - target_hat
            assert float(jnp.linalg.norm(before - reference)) < 0.01 * float(jnp.linalg.norm(after - reference))


def test_modal_residual_preserves_baseline_start_dc_masks_and_ordinary_metrics() -> None:
    nx = 64
    x = np.arange(nx) * (2 * np.pi / nx)
    family = jnp.arange(1, 5)
    for dtype, cutoff, full_rows in product(
        (jnp.float32, jnp.float64), (None, 8.0), (None, jnp.array([True, False, True, False])),
    ):
        config = SimpleNamespace(model="cs_dno", data_normalization="target",
                                 data_k_max=cutoff, residual_floor=1e-6, data_modal_residual_floor=0.0)
        ordinary = trainer_loss(config, full_spectrum_rows=full_rows)
        weighted = trainer_loss(SimpleNamespace(**{**vars(config), "data_modal_residual_floor": 0.05}),
                                full_spectrum_rows=full_rows)
        target = jnp.asarray(np.arange(1, 5)[:, None, None]
                             * (1.0 + np.sin(x) + 0.4 * np.cos(20 * x))[None, :, None], dtype)
        baseline = target - jnp.asarray(0.2 + 0.07 * np.cos(3 * x) + 0.3 * np.sin(18 * x), dtype)[None, :, None]
        original = ordinary(baseline, target, baseline, baseline, family)
        actual = weighted(baseline, target, baseline, baseline, family)
        # Normalizing modal weights preserves the initial scalar objective, including DC.
        np.testing.assert_allclose(actual[0], original[0], rtol=3e-7, atol=1e-15)
        np.testing.assert_array_equal(actual[1][0], original[1][0])
        np.testing.assert_array_equal(np.asarray(actual[1][1]).reshape(4, 5)[:, :4],
                                      np.asarray(original[1][1]).reshape(4, 5)[:, :4])
        pred_hat, target_hat = (jnp.fft.rfft(v, axis=1) for v in (baseline, target))
        mask = jnp.ones((4, nx // 2 + 1, 1), bool)
        if cutoff is not None:
            mask = mask & (jnp.arange(nx // 2 + 1) <= cutoff)[None, :, None]
        if full_rows is not None:
            mask = mask | full_rows[:, None, None]
        expected = jnp.mean(jnp.sqrt(jnp.sum(jnp.abs(jnp.where(
            mask, (pred_hat - target_hat).astype(jnp.complex128), 0.0,
        )) ** 2, axis=(1, 2))) / jnp.sqrt(jnp.sum(jnp.abs(jnp.where(
            mask, target_hat.astype(jnp.complex128), 0.0,
        )) ** 2, axis=(1, 2))))
        np.testing.assert_array_equal(original[0], expected)
        # A DC-only discrepancy is not reweighted even when the label residual is nonzero.
        dc_prediction = target + jnp.asarray(0.125, dtype)
        np.testing.assert_allclose(weighted(dc_prediction, target, baseline, baseline, family)[0],
                                   ordinary(dc_prediction, target, baseline, baseline, family)[0], rtol=3e-7)


def test_modal_residual_weights_gradients_and_symmetries() -> None:
    nx, size, alpha = 64, 3, 0.05
    x = np.arange(nx) * (2 * np.pi / nx)
    config = SimpleNamespace(model="cs_dno", data_normalization="target", data_k_max=12.0,
                             residual_floor=1e-6, data_modal_residual_floor=alpha)
    losses = trainer_loss(config, capture_modal_weights=True)
    target = jnp.asarray(np.arange(1, size + 1)[:, None, None]
                         * (0.3 + np.sin(x) + 0.2 * np.cos(19 * x))[None, :, None])
    baseline = target - (0.2 + 0.08 * jnp.cos(2 * x) + 0.001 * jnp.sin(7 * x))[None, :, None]
    prediction = target + (0.03 + 0.02 * jnp.sin(3 * x) + 0.01 * jnp.cos(20 * x))[None, :, None]
    family = jnp.arange(1, size + 1)
    (value, (_, (weight,))), gradients = jax.jit(jax.value_and_grad(
        losses, argnums=(0, 1, 2, 3), has_aux=True,
    ))(prediction, target, baseline, baseline, family)
    q_hat, b_hat, error = (
        np.fft.rfft(np.asarray(v), axis=1) for v in (target, baseline, prediction - target)
    )
    mask = (np.arange(nx // 2 + 1) <= config.data_k_max)[None, :, None]
    modal_mask = mask & (np.arange(nx // 2 + 1) > 0)[None, :, None]
    q_norm = np.sqrt(np.sum(np.where(mask, np.abs(q_hat)**2, 0), axis=(1, 2)))
    power = np.where(modal_mask, np.abs(q_hat - b_hat)**2, 0)
    inverse = 1 / np.sqrt(power + (
        alpha**2 * np.max(power, axis=(1, 2)) + (config.residual_floor * q_norm)**2
    )[:, None, None])
    expected_weights = np.where(modal_mask, inverse * (
        np.sum(power, axis=(1, 2)) / np.sum(inverse * power, axis=(1, 2))
    )[:, None, None], 1.0)
    np.testing.assert_allclose(weight, expected_weights, rtol=1e-13, atol=1e-13)
    active = np.asarray(weight)[:, 1:13]
    assert np.all(np.max(active, axis=1) / np.min(active, axis=1) <= np.sqrt(1 + alpha**2) / alpha)
    np.testing.assert_array_equal(np.asarray(weight)[:, [0, *range(13, nx // 2 + 1)]], 1)
    weighted_norm = np.sqrt(np.sum(np.where(mask, expected_weights * np.abs(error)**2, 0), axis=(1, 2)))
    np.testing.assert_allclose(value, np.mean(weighted_norm / q_norm), rtol=1e-13)
    # Analytic real-input gradient of the one-sided Fourier norm (no autodiff reference).
    half_weights = np.full(nx // 2 + 1, 0.5)
    half_weights[[0, -1]] = 1.0
    expected_gradient = nx * np.fft.irfft(np.where(
        mask, expected_weights * error * half_weights[None, :, None], 0,
    ), n=nx, axis=1) / (size * q_norm * weighted_norm)[:, None, None]
    np.testing.assert_allclose(gradients[0], expected_gradient, rtol=2e-12, atol=1e-14)
    np.testing.assert_array_equal(gradients[1], -gradients[0])
    for gradient in gradients[2:]:
        np.testing.assert_array_equal(gradient, 0)
    for scale, shift in ((0.1, 0), (-3.0, 0), (1.0, 7)):
        transformed = [scale * jnp.roll(v, shift, axis=1) for v in (prediction, target, baseline, baseline)]
        (changed, (_, (changed_weights,))), changed_gradient = jax.value_and_grad(
            losses, has_aux=True,
        )(*transformed, family)
        np.testing.assert_allclose(changed, value, rtol=2e-13)
        np.testing.assert_allclose(changed_weights, weight, rtol=2e-13, atol=1e-13)
        np.testing.assert_allclose(changed_gradient, jnp.roll(gradients[0], shift, axis=1) / scale,
                                   rtol=3e-12, atol=1e-13)


def test_modal_residual_exact_fit_zero_residual_and_nonfinite_inputs() -> None:
    config = SimpleNamespace(model="cs_dno", data_normalization="target", data_k_max=12.0,
                             residual_floor=1e-6, data_modal_residual_floor=0.05)
    losses = trainer_loss(config, capture_modal_weights=True)
    ordinary = trainer_loss(SimpleNamespace(**{**vars(config), "data_modal_residual_floor": 0.0}))
    family = jnp.array([1, 2, 3])
    evaluate = jax.jit(jax.value_and_grad(losses, argnums=(0, 1, 2, 3), has_aux=True))
    for dtype in (jnp.float32, jnp.float64):
        x = jnp.arange(64, dtype=dtype) * (2 * jnp.pi / 64)
        target = jnp.stack((jnp.sin(x), jnp.cos(3 * x), jnp.zeros_like(x)))[..., None]
        baseline = target.at[0].multiply(0.8)
        (zero, _), gradients = evaluate(target, target, baseline, baseline, family)
        np.testing.assert_array_equal(zero, 0)
        for gradient in gradients:
            np.testing.assert_array_equal(gradient, 0)
        prediction = target.at[1, :, 0].add(0.03 * jnp.cos(5 * x))
        (actual, (_, (weights,))), gradients = evaluate(prediction, target, target, target, family)
        expected = jax.value_and_grad(ordinary, has_aux=True)(prediction, target, target, target, family)
        np.testing.assert_array_equal(actual, expected[0][0])
        np.testing.assert_array_equal(weights, 1)
        np.testing.assert_allclose(gradients[0], expected[1], rtol=2e-6, atol=1e-10)
        for gradient in gradients:
            assert np.all(np.isfinite(gradient))
        np.testing.assert_array_equal(gradients[0][jnp.array([0, 2])], 0)
        for invalid in (jnp.nan, jnp.inf):
            bad = target.at[0, 3, 0].set(invalid)
            assert not np.isfinite(losses(bad, target, baseline, baseline, family)[0])
            assert not np.isfinite(losses(prediction, bad, baseline, baseline, family)[0])


def test_generated_full_spectrum_rows_preserve_fallback_losses_and_gradients() -> None:
    nx, size = 512, 4
    x = jnp.arange(nx, dtype=jnp.float64) * (2 * jnp.pi / nx)
    family = jnp.arange(1, size + 1)
    valid = jnp.array([True, False, True, False])
    for dtype, normalization in product((jnp.float32, jnp.float64), ("target", "residual")):
        config = SimpleNamespace(model="cs_dno", data_normalization=normalization,
                                 data_k_max=128.0, residual_floor=1e-6)
        target = jnp.asarray((jnp.arange(size) + 1)[:, None, None]
                             * (jnp.sin(3 * x) + 0.7 * jnp.cos(160 * x))[None, :, None], dtype=dtype)
        prediction = target + jnp.asarray(
            0.03 * jnp.cos(7 * x) + 0.09 * jnp.sin(170 * x), dtype=dtype,
        )[None, :, None]
        baseline = 0.8 * target
        ordinary = trainer_loss(config)
        mixed = trainer_loss(config, full_spectrum_rows=valid)
        full = trainer_loss(SimpleNamespace(**{**vars(config), "data_k_max": None}))
        evaluate = jax.jit(jax.value_and_grad(mixed, argnums=(0, 2, 3), has_aux=True))
        (value, (full_metric, family_metrics)), gradients = evaluate(
            prediction, target, baseline, baseline, family,
        )
        row_results = [
            jax.value_and_grad(full if valid[i] else ordinary, has_aux=True)(
                prediction[i:i + 1], target[i:i + 1], baseline[i:i + 1],
                baseline[i:i + 1], family[i:i + 1],
            ) for i in range(size)
        ]
        np.testing.assert_allclose(value, jnp.mean(jnp.asarray([row[0][0] for row in row_results])), rtol=2e-6)
        np.testing.assert_allclose(gradients[0], jnp.concatenate([row[1] for row in row_results]) / size,
                                   rtol=2e-6, atol=2e-9)
        for gradient in gradients[1:]:
            np.testing.assert_array_equal(gradient, 0)
        ordinary_metrics = ordinary(prediction, target, baseline, baseline, family)[1]
        np.testing.assert_array_equal(full_metric, ordinary_metrics[0])
        # Only the final data-objective entry in each family's five metrics changes.
        np.testing.assert_allclose(jnp.asarray(family_metrics).reshape(4, 5)[:, :4],
                                   jnp.asarray(ordinary_metrics[1]).reshape(4, 5)[:, :4], rtol=1e-12)
        gradient_hat = np.asarray(jnp.fft.rfft(gradients[0], axis=1))
        assert np.linalg.norm(gradient_hat[np.asarray(valid), 170]) > 1e-3
        np.testing.assert_allclose(gradient_hat[~np.asarray(valid), 129:], 0,
                                   atol=8 * np.finfo(np.dtype(dtype)).eps * np.max(np.abs(gradient_hat)))
        for mask, reference in ((jnp.zeros(size, bool), ordinary), (jnp.ones(size, bool), full)):
            masked = trainer_loss(config, full_spectrum_rows=mask)
            actual = jax.value_and_grad(masked, has_aux=True)(prediction, target, baseline, baseline, family)
            expected = jax.value_and_grad(reference, has_aux=True)(prediction, target, baseline, baseline, family)
            for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
                np.testing.assert_array_equal(a, b)
        (zero, _), zero_gradients = evaluate(target, target, baseline, baseline, family)
        assert zero == 0
        for gradient in zero_gradients:
            np.testing.assert_array_equal(gradient, 0)


def test_fp64_residual_uses_direct_correction_and_preserves_loss_scales() -> None:
    from util import make_normalizers

    nx = 64
    x = jnp.arange(nx, dtype=jnp.float64) * (2 * jnp.pi / nx)
    eta = (0.02 * jnp.cos(2 * x))[None]
    xi = (0.03 * jnp.sin(3 * x))[None]
    depth = jnp.log(jnp.asarray([[0.4]], dtype=jnp.float64))
    family = jnp.array([PhysicalFamilyId.STOKES])
    model = CraigSulemDNO(width=8, n_blocks=1, latent=2, mult_hidden=4,
                         eta_scale=2.0, xi_scale=3.0, target_scale=4.0,
                         baseline_true_depth=True)
    config = SimpleNamespace(
        model="cs_dno", data_normalization="target", data_k_max=16.0, residual_floor=1e-6,
        data_xi_projection_weight=0.0, data_phase_projection_weight=0.0, data_modal_residual_floor=0.0,
        mode_balanced_weight=6.0, mode_balanced_k_max=16.0,
        mode_balanced_activity_threshold=1e-4, mode_balanced_denominator_eps=1e-6,
        translation_tangent_weight=0.0, phase_mechanism="none", jonswap_mode_balanced_scale=1.0,
        subtract_before_fft=False, fp64_residual=True, subtract_before_centering=False,
    )
    mode = jnp.cos(5 * x).astype(jnp.float32)[None, :, None]
    calls = []

    def apply_model(
        variables: dict[str, jax.Array], inputs: jax.Array, log_depth: jax.Array, *, return_baseline: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        calls.append(inputs.dtype)
        baseline = (cast(jax.Array, model.apply({}, inputs[..., 1], log_depth, method=model._linear_baseline))
                    + cast(jax.Array, model.apply({}, inputs[..., 0], inputs[..., 1], log_depth,
                                                method=model._g1_baseline)))[..., None]
        correction = variables["params"] * mode
        return baseline + correction, baseline, baseline, correction

    baseline64 = (
        cast(jax.Array, model.apply({}, xi / 3, depth, method=model._linear_baseline))
        + cast(jax.Array, model.apply({}, eta / 2, xi / 3, depth, method=model._g1_baseline))
    )[..., None]
    target = 4 * (baseline64[..., 0] + 3e-11 * mode[..., 0].astype(jnp.float64))
    target32 = target.astype(jnp.float32)
    target_hat = jnp.fft.rfft(target32[..., None] / 4, axis=1).astype(jnp.complex128)
    denominator = jax.lax.stop_gradient(jnp.sqrt(jnp.sum(jnp.abs(target_hat[:, :17])**2)))
    normalize_inputs, normalize_targets, denormalize_targets = make_normalizers(
        {"feature_absmax": [2.0, 3.0], "target_absmax": 4.0}, mode="scale",
    )
    namespace: dict[str, Any] = dict(
        jax=jax, jnp=jnp, FlatParams=Any, cast=cast, args=config, model=model,
        training_dtype=jnp.float32, physical_dtype=jnp.float32, PhysicalFamilyId=PhysicalFamilyId, apply_model=apply_model,
        norm_inputs_jax=normalize_inputs, norm_targets_jax=normalize_targets,
        denorm_targets_jax=denormalize_targets,
        log_h_max=np.log(5), k_rfft_jax=jnp.arange(nx // 2 + 1, dtype=jnp.float32),
        compute_mode_balanced_loss=compute_mode_balanced_loss,
    )
    exec(compile(ast.Module(body=[trainer_function("compute_loss_components")], type_ignores=[]),
                 "fp64_residual_components", "exec"), namespace)
    parts = namespace["compute_loss_components"]

    def objective(p: jax.Array) -> jax.Array:
        values = parts(p, eta, xi, target, depth, family)
        return values[0] + 6 * values[1]

    def expected(p: jax.Array) -> jax.Array:
        residual = p.astype(jnp.float64) * mode - (target[..., None] / 4 - baseline64)
        error_hat = jnp.fft.rfft(residual, axis=1)
        data_loss = jnp.sqrt(jnp.sum(jnp.abs(error_hat[:, :17])**2)) / denominator
        mb = compute_mode_balanced_loss(
            eta.astype(jnp.float32), jnp.zeros_like(target32), target32,
            jnp.exp(depth.astype(jnp.float32)[:, 0]), jnp.arange(nx // 2 + 1, dtype=jnp.float32),
            k_max=16, gxi_error=4 * residual[..., 0],
        )
        return data_loss + 6 * mb

    p = jnp.float32(1e-10)
    calls.clear()
    objective(p)
    assert calls == [jnp.dtype(jnp.float32)]
    for actual, reference in zip(jax.jit(jax.value_and_grad(objective))(p),
                                 jax.jit(jax.value_and_grad(expected))(p)):
        np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=1e-13)
    # Validation/family metrics and generated labeled rows use this same function.
    ordinary = parts(p, eta, xi, target, depth, family)
    collected = parts(p, eta, xi, target, depth, family, collect_family_metrics=True)
    for actual, reference in zip(collected[:6], ordinary[:6]):
        np.testing.assert_array_equal(actual, reference)
    generated = parts(p, eta, xi, target, depth, family, n_data_only_rows=1)
    np.testing.assert_array_equal(generated[0], ordinary[0])
    np.testing.assert_array_equal(generated[1:5], jnp.zeros(4))
    config.__dict__.update(mode_balanced_warmup_steps=1, phase_mechanism_weight=0.0,
                           phase_mechanism_ramp_epochs=0.0, rollout_training_weight=0.7,
                           rollout_training_ramp_epochs=0.0, rollout_training_loss="full",
                           rollout_training_full_spectrum=False, rollout_coherent_weight=0.0, hadamard_weight=0.0)
    namespace.update(eta=eta, xi=xi, gxi=target, batch_depth=depth, family=family,
                     current_state=SimpleNamespace(step=100), n_data_only_rows=0, amplitude_batch=None, coherent_batch=None, terms_phys=None,
                     rollout_batch=(True, depth, family, eta, xi, target, jnp.float32(1.0)))
    exec(compile(ast.Module(body=[trainer_function("compute_training_loss")], type_ignores=[]),
                 "fp64_residual_teacher_loss", "exec"), namespace)
    (combined, metrics), combined_grad = jax.value_and_grad(namespace["compute_training_loss"], has_aux=True)(p)
    expected_value, expected_grad = jax.value_and_grad(expected)(p)
    np.testing.assert_allclose(combined, 1.7 * expected_value, rtol=2e-6)
    np.testing.assert_allclose(combined_grad, 1.7 * expected_grad, rtol=2e-6)
    np.testing.assert_allclose(metrics["rollout_data_loss"], ordinary[0], rtol=2e-6)
    np.testing.assert_allclose(metrics["rollout_mode_balanced_loss"], ordinary[1], rtol=2e-6)
    # Disabling the arm retains the old full-output FFT numerator, even when the
    # direct component is much smaller than a ULP of the full prediction.
    config.fp64_residual = False
    pred, _, _, direct = apply_model({"params": p}, jnp.stack((eta, xi), axis=-1).astype(jnp.float32)
                                     / jnp.asarray([2, 3]), depth.astype(jnp.float32), return_baseline=True)
    error = jnp.fft.rfft(pred, axis=1) - jnp.fft.rfft(target32[..., None] / 4, axis=1)
    expected_old = jnp.sqrt(jnp.sum(jnp.abs(error[:, :17].astype(jnp.complex128))**2)) / denominator
    np.testing.assert_allclose(parts(p, eta, xi, target, depth, family)[0], expected_old, rtol=1e-12)
    assert jnp.any(direct != 0)


def test_xi_projection_uses_production_loss_and_preserves_ordinary_metrics() -> None:
    config = SimpleNamespace(
        model="fno", data_k_max=None, residual_floor=1e-6, data_xi_projection_weight=0.0,
        mode_balanced_weight=0.0, translation_tangent_weight=0.0, phase_mechanism="none", subtract_before_fft=False,
        fp64_residual=False, subtract_before_centering=False, data_phase_projection_weight=0.0,
        data_modal_residual_floor=0.0,
    )
    nx = 1024
    x = jnp.arange(nx, dtype=jnp.float64) * (2 * jnp.pi / nx)
    family = jnp.arange(1, 5)
    amplitude = jnp.arange(1, 5)[:, None]

    def apply_model(
        variables: dict[str, jax.Array], inputs: jax.Array, depth: jax.Array, *, return_baseline: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        prediction = variables["params"][..., None] / 4
        return prediction, jnp.zeros_like(prediction), jnp.zeros_like(prediction), prediction

    namespace: dict[str, Any] = {
        "jax": jax, "jnp": jnp, "FlatParams": Any, "args": config,
        "PhysicalFamilyId": PhysicalFamilyId, "apply_model": apply_model, "log_h_max": np.log(5),
        "norm_inputs_jax": lambda eta, xi: jnp.stack((eta / 2, xi / 3), axis=-1),
        "norm_targets_jax": lambda q: q[..., None] / 4, "denorm_targets_jax": lambda q: q * 4,
    }
    exec(compile(ast.Module(body=[trainer_function("compute_loss_components")], type_ignores=[]),
                 "xi_projection_components", "exec"), namespace)

    def evaluate(prediction: jax.Array, xi: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        parts = namespace["compute_loss_components"](
            prediction, jnp.zeros_like(xi), xi, target, jnp.zeros((4, 1)), family,
            collect_family_metrics=True,
        )
        return parts[0], (parts[5], jnp.asarray(parts[6]).reshape(4, 5))

    for dtype in (jnp.float32, jnp.float64):
        namespace["training_dtype"] = dtype
        namespace["physical_dtype"] = dtype
        target = (amplitude * (jnp.cos(3 * x) + 0.3 * jnp.sin(7 * x))).astype(dtype)
        cases = (
            (2 * np.pi, None, jnp.cos(4 * x), jnp.cos(4 * x), np.sqrt(5)),
            (2 * np.pi, None, -3 * jnp.cos(4 * x), jnp.cos(4 * x), np.sqrt(5)),
            (2 * np.pi, None, 1e-8 * jnp.cos(4 * x), jnp.cos(4 * x), np.sqrt(5)),
            (2 * np.pi, None, 3 + jnp.cos(4 * x), jnp.cos(4 * x), np.sqrt(5)),
            (2 * np.pi, None, jnp.cos(4 * x), jnp.sin(4 * x), 1.0),
            (2 * np.pi, None, jnp.cos(4 * x), jnp.cos(5 * x), 1.0),
            (2 * np.pi, None, jnp.zeros_like(x), jnp.cos(4 * x), 1.0),
            (2 * np.pi, None, jnp.ones_like(x), jnp.cos(4 * x), 1.0),
            # Suppress roundoff in a mathematically absent retained direction.
            (2 * np.pi, None, jnp.cos(200 * x), jnp.cos(4 * x), 1.0),
            (4 * np.pi, None, jnp.cos(200 * x), jnp.cos(200 * x), np.sqrt(5)),
            (4 * np.pi, 5.0, jnp.cos(8 * x), jnp.cos(8 * x), np.sqrt(5)),
            (4 * np.pi, 5.0, jnp.cos(12 * x), jnp.cos(4 * x) + 0.3 * jnp.cos(12 * x), 1.0),
        )
        for length, cutoff, direction, error, factor in cases:
            config.data_k_max = cutoff
            namespace["k_rfft_jax"] = (2 * jnp.pi / length) * jnp.arange(nx // 2 + 1)
            xi = jnp.broadcast_to(direction, target.shape).astype(dtype)
            prediction = (target + 0.013 * amplitude * error).astype(dtype)
            results = []
            for weight in (0.0, 4.0):
                config.data_xi_projection_weight = weight
                differentiate = jax.jit(jax.value_and_grad(lambda p, z: evaluate(p, z), has_aux=True))
                results.append(differentiate(prediction, xi))
                (zero, _), zero_gradient = differentiate(target, xi)
                assert zero == 0 and bool(jnp.all(jnp.isfinite(zero_gradient)))
                np.testing.assert_array_equal(zero_gradient, 0)
            (ordinary, ordinary_aux), ordinary_gradient = results[0]
            (weighted, weighted_aux), weighted_gradient = results[1]
            tolerance = 2e-6 if dtype == jnp.float32 else 3e-12
            np.testing.assert_allclose(weighted, factor * ordinary, rtol=tolerance)
            assert ordinary * (1 - tolerance) <= weighted <= np.sqrt(5) * ordinary * (1 + tolerance)
            assert bool(jnp.all(jnp.isfinite(weighted_gradient)))
            np.testing.assert_array_equal(weighted_aux[0], ordinary_aux[0])
            np.testing.assert_array_equal(weighted_aux[1][:, :4], ordinary_aux[1][:, :4])
            if cutoff is None:
                reference, reference_gradient = jax.jit(jax.value_and_grad(
                    lambda p: relative_l2_loss(p[..., None] / 4, target[..., None] / 4),
                ))(prediction)
                np.testing.assert_array_equal(ordinary, reference)
                np.testing.assert_array_equal(ordinary_gradient, reference_gradient)

    config.data_k_max = None
    namespace["k_rfft_jax"] = jnp.arange(nx // 2 + 1)
    coefficients = jnp.array([0.013, 0.021])
    (value, _), gradient = jax.value_and_grad(lambda c: evaluate(
        target + amplitude * (c[0] * jnp.cos(4 * x) + c[1] * jnp.cos(5 * x)),
        jnp.broadcast_to(jnp.cos(4 * x), target.shape),
    ), has_aux=True)(coefficients)
    norm = jnp.sqrt(5 * coefficients[0]**2 + coefficients[1]**2)
    np.testing.assert_allclose(value, norm / np.sqrt(1.09), rtol=1e-12)
    np.testing.assert_allclose(gradient, coefficients * jnp.array([5, 1]) / (norm * np.sqrt(1.09)), rtol=1e-12)
    target = jnp.zeros_like(target)
    (value, _), gradient = jax.value_and_grad(evaluate, has_aux=True)(target, jnp.ones_like(target))
    assert value == 0 and bool(jnp.all(jnp.isfinite(gradient)))
    np.testing.assert_array_equal(gradient, 0)

    main = trainer_function("main")
    option = next(node for node in ast.walk(main) if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call) and node.value.args
                  and isinstance(node.value.args[0], ast.Constant)
                  and node.value.args[0].value == "--data_xi_projection_weight")
    validation = next(node for node in main.body if isinstance(node, ast.If)
                      and "args.data_xi_projection_weight" in ast.unparse(node.test))
    parser = argparse.ArgumentParser(add_help=False)
    namespace = {"parser": parser, "np": np}
    exec(compile(ast.Module(body=[option], type_ignores=[]), "xi_projection_option", "exec"), namespace)
    assert parser.parse_args([]).data_xi_projection_weight == 0.0
    assert parser.parse_args(["--data_xi_projection_weight", "4"]).data_xi_projection_weight == 4.0
    with patch.object(parser, "error", side_effect=ValueError):
        for invalid in (-1.0, np.nan, np.inf, -np.inf):
            namespace["args"] = SimpleNamespace(data_xi_projection_weight=invalid)
            with np.testing.assert_raises(ValueError):
                exec(compile(ast.Module(body=[validation], type_ignores=[]), "xi_projection_validation", "exec"), namespace)


def test_phase_projection_weights_angular_not_radial_errors_and_preserves_symmetries() -> None:
    config = SimpleNamespace(
        model="fno", data_k_max=None, residual_floor=1e-6, data_xi_projection_weight=0.0,
        data_phase_projection_weight=0.0, data_modal_residual_floor=0.0, mode_balanced_weight=0.0,
        translation_tangent_weight=0.0, phase_mechanism="none", subtract_before_fft=False,
        fp64_residual=False, subtract_before_centering=False,
    )
    nx = 512
    x = jnp.arange(nx, dtype=jnp.float64) * (2 * jnp.pi / nx)
    family = jnp.arange(1, 5)
    amplitude = jnp.arange(1, 5)[:, None]

    def apply_model(
        variables: dict[str, jax.Array], inputs: jax.Array, depth: jax.Array, *, return_baseline: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        prediction = variables["params"][..., None] / 4
        return prediction, jnp.zeros_like(prediction), jnp.zeros_like(prediction), prediction

    namespace: dict[str, Any] = {
        "jax": jax, "jnp": jnp, "FlatParams": Any, "args": config,
        "PhysicalFamilyId": PhysicalFamilyId, "apply_model": apply_model, "log_h_max": np.log(5),
        "norm_inputs_jax": lambda eta, xi: jnp.stack((eta / 2, xi / 3), axis=-1),
        "norm_targets_jax": lambda q: q[..., None] / 4, "denorm_targets_jax": lambda q: q * 4,
    }
    exec(compile(ast.Module(body=[trainer_function("compute_loss_components")], type_ignores=[]),
                 "phase_projection_components", "exec"), namespace)
    components = namespace["compute_loss_components"]

    def evaluate(prediction: jax.Array, eta: jax.Array, truth: jax.Array) -> tuple[jax.Array, jax.Array]:
        values = components(prediction, eta, jnp.zeros_like(eta), truth, jnp.zeros((4, 1)), family,
                            collect_family_metrics=True)
        return values[0], jnp.asarray((values[5], *values[6]))

    for dtype in (jnp.float32, jnp.float64):
        namespace["training_dtype"] = dtype
        namespace["physical_dtype"] = dtype
        tolerance = 5e-6 if dtype == jnp.float32 else 3e-12
        target = (amplitude * (jnp.cos(3 * x) + 0.3 * jnp.sin(7 * x))).astype(dtype)
        cases = (
            (2 * np.pi, None, jnp.cos(4 * x), jnp.sin(4 * x), 2.0),
            (2 * np.pi, None, jnp.cos(4 * x), jnp.cos(4 * x), 1.0),
            (2 * np.pi, None, -3 * jnp.cos(4 * x), jnp.sin(4 * x), 2.0),
            (2 * np.pi, None, 1e-8 * jnp.cos(4 * x), jnp.sin(4 * x), 2.0),
            (2 * np.pi, None, 3 + jnp.cos(4 * x), jnp.sin(4 * x), 2.0),
            (2 * np.pi, None, jnp.cos(4 * x), jnp.sin(5 * x), 1.0),
            (2 * np.pi, None, jnp.cos(4 * x) + 1e-8 * jnp.cos(5 * x), jnp.sin(5 * x), 1.0),
            (2 * np.pi, None, jnp.zeros_like(x), jnp.sin(4 * x), 1.0),
            (2 * np.pi, None, jnp.ones_like(x), jnp.sin(4 * x), 1.0),
            (2 * np.pi, None, jnp.cos(200 * x), jnp.sin(4 * x), 1.0),
            (2 * np.pi, None, jnp.cos(4 * x) + jnp.cos(200 * x), jnp.sin(200 * x), 1.0),
            (4 * np.pi, None, jnp.cos(200 * x), jnp.sin(200 * x), 2.0),
            (4 * np.pi, 5.0, jnp.cos(8 * x), jnp.sin(8 * x), 2.0),
            (4 * np.pi, 5.0, jnp.cos(4 * x) + jnp.cos(12 * x),
             jnp.cos(4 * x) + 0.3 * jnp.sin(12 * x), 1.0),
        )
        for length, cutoff, direction, error, factor in cases:
            config.data_k_max = cutoff
            namespace["k_rfft_jax"] = (2 * jnp.pi / length) * jnp.arange(nx // 2 + 1)
            eta = jnp.broadcast_to(direction, target.shape).astype(dtype)
            prediction = (target + 0.013 * amplitude * error).astype(dtype)
            results = []
            for weight in (0.0, 3.0):
                config.data_phase_projection_weight = weight
                differentiate = jax.jit(jax.value_and_grad(evaluate, argnums=(0, 1), has_aux=True))
                results.append(differentiate(prediction, eta, target))
                (zero, _), zero_gradients = differentiate(target, eta, target)
                assert zero == 0
                for gradient in zero_gradients:
                    np.testing.assert_array_equal(gradient, 0)
            (ordinary, ordinary_aux), ordinary_gradients = results[0]
            (weighted, weighted_aux), gradients = results[1]
            np.testing.assert_allclose(weighted, factor * ordinary, rtol=tolerance)
            assert ordinary * (1 - tolerance) <= weighted <= 2 * ordinary * (1 + tolerance)
            assert all(bool(jnp.all(jnp.isfinite(gradient))) for gradient in gradients)
            np.testing.assert_array_equal(weighted_aux[0], ordinary_aux[0])
            np.testing.assert_array_equal(weighted_aux[1:].reshape(4, 5)[:, :4],
                                          ordinary_aux[1:].reshape(4, 5)[:, :4])
            if cutoff is None:
                reference, reference_gradient = jax.value_and_grad(
                    lambda p: relative_l2_loss(p[..., None] / 4, target[..., None] / 4),
                )(prediction)
                np.testing.assert_allclose(ordinary, reference, rtol=tolerance)
                np.testing.assert_allclose(ordinary_gradients[0], reference_gradient, rtol=tolerance, atol=1e-9)

        config.data_k_max = 128.0
        namespace["k_rfft_jax"] = jnp.arange(nx // 2 + 1)
        eta = (amplitude * (jnp.cos(4 * x) + 0.3 * jnp.sin(7 * x))).astype(dtype)
        prediction = target + (amplitude * (0.013 * jnp.sin(4 * x) + 0.021 * jnp.cos(7 * x))).astype(dtype)
        (value, _), gradient = jax.value_and_grad(evaluate, has_aux=True)(prediction, eta, target)
        for indices in (jnp.roll(jnp.arange(nx), 37), (-jnp.arange(nx)) % nx):
            (transformed, _), transformed_gradient = jax.value_and_grad(evaluate, has_aux=True)(
                prediction[:, indices], eta[:, indices], target[:, indices],
            )
            np.testing.assert_allclose(transformed, value, rtol=tolerance)
            np.testing.assert_allclose(transformed_gradient, gradient[:, indices], rtol=3e-5,
                                       atol=1e-8 if dtype == jnp.float32 else 1e-12)
        original = components(prediction, eta, eta, target, jnp.zeros((4, 1)), family)
        relabeled = components(prediction, eta, eta, target, jnp.zeros((4, 1)), jnp.full_like(family, 99))
        np.testing.assert_array_equal(original[:6], relabeled[:6])

    main = trainer_function("main")
    option = next(node for node in ast.walk(main) if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call) and node.value.args
                  and isinstance(node.value.args[0], ast.Constant)
                  and node.value.args[0].value == "--data_phase_projection_weight")
    validation = next(node for node in main.body if isinstance(node, ast.If)
                      and "args.data_phase_projection_weight" in ast.unparse(node.test))
    parser = argparse.ArgumentParser(add_help=False)
    namespace = {"parser": parser, "np": np}
    exec(compile(ast.Module(body=[option], type_ignores=[]), "phase_projection_option", "exec"), namespace)
    assert parser.parse_args([]).data_phase_projection_weight == 0.0
    assert parser.parse_args(["--data_phase_projection_weight", "1"]).data_phase_projection_weight == 1.0
    with patch.object(parser, "error", side_effect=ValueError):
        for invalid in (-1.0, np.nan, np.inf, -np.inf):
            namespace["args"] = SimpleNamespace(data_phase_projection_weight=invalid)
            with np.testing.assert_raises(ValueError):
                exec(compile(ast.Module(body=[validation], type_ignores=[]), "phase_projection_validation", "exec"), namespace)


def test_retained_band_normalizers_gradients_and_full_metrics() -> None:
    nx = 64
    phase = jnp.arange(nx, dtype=jnp.float64) * (2 * jnp.pi / nx)
    amplitude = jnp.arange(1, 5, dtype=jnp.float64)[:, None, None]
    target = amplitude * (jnp.sin(4 * phase) + 0.8 * jnp.cos(20 * phase))[None, :, None]
    prediction = target + amplitude * (0.1 * jnp.cos(6 * phase) + 0.3 * jnp.cos(22 * phase))[None, :, None]
    baseline = 0.7 * target + 5 * jnp.cos(20 * phase)[None, :, None]
    m1 = 0.5 * target + 3 * jnp.cos(20 * phase)[None, :, None]
    family = jnp.arange(1, 5)
    # L=4pi: cutoff k=3 retains mode6, not just mode3.
    mask = (jnp.arange(nx // 2 + 1) <= 6)[None, :, None]
    projected = tuple(jnp.fft.irfft(jnp.where(mask, jnp.fft.rfft(value, axis=1), 0.0), n=nx, axis=1)
                      for value in (prediction, target, baseline, m1))
    for normalization in ("target", "residual", "bounded", "nonlinear"):
        config = SimpleNamespace(model="cs_dno", data_normalization=normalization, data_k_max=3.0,
                                 residual_floor=1e-6, nonlinear_reference=0.002, nonlinear_weight_mean=1.4)
        selected = trainer_loss(config, domain_length=4 * np.pi)
        full = trainer_loss(SimpleNamespace(**(vars(config) | {"data_k_max": None})))
        (actual, (full_l2, metrics)), gradients = jax.jit(jax.value_and_grad(
            selected, argnums=(0, 2, 3), has_aux=True,
        ))(prediction, target, baseline, m1, family)
        expected, (_, band_metrics) = full(projected[0], projected[1], projected[2], projected[3], family)
        _, (expected_full_l2, full_metrics) = full(prediction, target, baseline, m1, family)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)
        np.testing.assert_allclose(full_l2, expected_full_l2, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(metrics).reshape(4, 5)[:, :4],
                                   np.asarray(full_metrics).reshape(4, 5)[:, :4], rtol=1e-12)
        np.testing.assert_allclose(np.asarray(metrics).reshape(4, 5)[:, 4],
                                   np.asarray(band_metrics).reshape(4, 5)[:, 4], rtol=1e-12)
        gradient_hat = jnp.fft.rfft(gradients[0], axis=1)
        np.testing.assert_allclose(gradient_hat[:, 7:], 0.0, atol=1e-12)
        assert float(jnp.linalg.norm(gradient_hat[:, 6])) > 0.01
        for gradient in gradients[1:]:
            np.testing.assert_array_equal(gradient, 0.0)


def test_retained_band_zero_error_and_cutoff_validation() -> None:
    family = jnp.arange(1, 5)
    for dtype, normalization in product((jnp.float32, jnp.float64), ("target", "residual")):
        config = SimpleNamespace(model="cs_dno", data_normalization=normalization,
                                 data_k_max=8.0, residual_floor=1e-6)
        evaluate = jax.jit(jax.value_and_grad(trainer_loss(config), has_aux=True))
        target = jnp.ones((4, 64, 1), dtype=dtype)
        high = jnp.tile(jnp.array([1, 0, -1, 0], dtype=dtype), 16)[None, :, None]
        for prediction, truth in ((target + 0.125 * high, target), (target, target), (0 * target, 0 * target)):
            (loss, (full_l2, _)), gradient = evaluate(prediction, truth, truth, truth, family)
            assert loss == 0 and bool(jnp.all(jnp.isfinite(gradient)))
            np.testing.assert_array_equal(gradient, 0.0)
            assert full_l2 > 0 if bool(jnp.any(prediction != truth)) else full_l2 == 0
    main = trainer_function("main")
    option = next(node for node in ast.walk(main) if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call) and node.value.args
                  and isinstance(node.value.args[0], ast.Constant) and node.value.args[0].value == "--data_k_max")
    validation = next(node for node in ast.walk(main) if isinstance(node, ast.If)
                      and "args.data_k_max" in ast.unparse(node.test))
    parser = argparse.ArgumentParser(add_help=False)
    namespace: dict[str, Any] = {"parser": parser, "np": np}
    exec(compile(ast.Module(body=[option], type_ignores=[]), "data_cutoff_option", "exec"), namespace)
    assert parser.parse_args([]).data_k_max is None
    assert parser.parse_args(["--data_k_max", "128"]).data_k_max == 128
    with patch.object(parser, "error", side_effect=ValueError):
        for invalid in (0.0, -1.0, np.nan, np.inf, -np.inf):
            namespace["args"] = SimpleNamespace(data_k_max=invalid)
            with np.testing.assert_raises(ValueError):
                exec(compile(ast.Module(body=[validation], type_ignores=[]), "data_cutoff_validation", "exec"), namespace)


def test_bounded_weights_decrease_and_have_stopped_scale_invariant_gradients() -> None:
    config = SimpleNamespace(model="cs_dno", data_normalization="bounded", data_k_max=None, residual_floor=1e-6,
                             nonlinear_reference=0.002, nonlinear_weight_mean=1.4)
    losses = trainer_loss(config)
    evaluate = jax.jit(jax.value_and_grad(losses, argnums=(0, 1, 2, 3), has_aux=True))
    family = jnp.array([PhysicalFamilyId.TANAKA])
    ratios = np.array([0.0, 1e-5, 0.001, 0.002, 0.02, 1.0, 1e4])
    for dtype in (jnp.float32, jnp.float64):
        x = jnp.arange(64, dtype=dtype) * (2 * jnp.pi / 64)
        target = jnp.sin(x)[None, :, None]
        prediction = target + 0.01 * jnp.cos(2 * x)[None, :, None]
        ordinary, ordinary_gradient = jax.jit(jax.value_and_grad(relative_l2_loss))(prediction, target)
        target_hat = np.asarray(jnp.fft.rfft(target, axis=1))
        weights = []
        for ratio in ratios:
            baseline = (1 - jnp.asarray(ratio, dtype=dtype)) * target
            residual_hat = (target_hat - np.asarray(jnp.fft.rfft(baseline, axis=1))).astype(np.complex128)
            measured_ratio = np.linalg.norm(residual_hat) / np.linalg.norm(target_hat.astype(np.complex128))
            expected = (1 + config.nonlinear_reference
                        / (np.hypot(measured_ratio, config.residual_floor) + config.nonlinear_reference))
            expected /= config.nonlinear_weight_mean
            (weighted, (full, _)), gradients = evaluate(prediction, target, baseline, target, family)
            np.testing.assert_array_equal(full, ordinary)
            weight = float(weighted / ordinary)
            weights.append(weight)
            np.testing.assert_allclose(weight, expected, rtol=5e-6)
            np.testing.assert_allclose(gradients[0], weight * ordinary_gradient, rtol=3e-5, atol=1e-8)
            np.testing.assert_allclose(gradients[1], -gradients[0], rtol=1e-6)
            np.testing.assert_array_equal(gradients[2], 0)
            np.testing.assert_array_equal(gradients[3], 0)
            for scale in (0.1, 7.0):
                scaled, _ = losses(scale * prediction, scale * target, scale * baseline, scale * target, family)
                # Tiny residuals subtract nearly equal FP32 FFT coefficients.
                np.testing.assert_allclose(scaled, weighted, rtol=1e-4 if dtype == jnp.float32 else 1e-11)
            (changed_m1, _), _ = evaluate(prediction, target, baseline, 0.3 * target, family)
            np.testing.assert_array_equal(changed_m1, weighted)
        assert np.all(np.diff(weights) < 0)
        assert np.all(np.asarray(weights) >= 1 / config.nonlinear_weight_mean)
        assert np.all(np.asarray(weights) <= 2 / config.nonlinear_weight_mean)
        for scale in (0.0, 1e-14, 1.0):
            truth = scale * target
            for pred in (truth, truth + 1e-8):
                (value, (full, metrics)), gradients = evaluate(pred, truth, truth, truth, family)
                assert bool(jnp.isfinite(value) & jnp.isfinite(full))
                assert bool(jnp.all(jnp.isfinite(jnp.asarray(metrics))))
                assert all(bool(jnp.all(jnp.isfinite(gradient))) for gradient in gradients)
                if bool(jnp.array_equal(pred, truth)):
                    assert value == full == 0
                    for gradient in gradients:
                        np.testing.assert_array_equal(gradient, 0)


def test_residual_denominator_gradients_scaling_and_zero_head() -> None:
    config = SimpleNamespace(model="cs_dno", data_normalization="residual", data_k_max=None, residual_floor=1e-6)
    losses = trainer_loss(config)
    differentiate = jax.value_and_grad(losses, argnums=(0, 1, 2, 3), has_aux=True)
    family = jnp.array([PhysicalFamilyId.STOKES])
    for dtype in (jnp.float32, jnp.float64):
        x = jnp.arange(64, dtype=dtype) * (2 * jnp.pi / 64)
        target = jnp.sin(x)[None, :, None]
        baseline = 0.75 * target
        prediction = target + 0.02 * jnp.cos(2 * x)[None, :, None]
        for evaluate in (differentiate, jax.jit(differentiate)):
            (value, _), gradients = evaluate(prediction, target, baseline, target, family)
            assert value > 0
            assert all(bool(jnp.all(jnp.isfinite(gradient))) for gradient in gradients)
            np.testing.assert_array_equal(gradients[2], 0)
            np.testing.assert_array_equal(gradients[3], 0)
            np.testing.assert_allclose(gradients[1], -gradients[0], rtol=1e-6)
            for scale in (0.1, 7.0):
                (scaled, _), _ = evaluate(scale * prediction, scale * target,
                                          scale * baseline, scale * target, family)
                np.testing.assert_allclose(scaled, value, rtol=3e-6)
            doubled, _ = losses(target + 2 * (prediction - target), target, baseline, target, family)
            np.testing.assert_allclose(doubled, 2 * value, rtol=3e-6)
            for scale in (0.0, 1e-14, 1.0):
                truth = target * scale
                for pred in (truth, truth + 1e-8):
                    (result, (full, metrics)), grads = evaluate(pred, truth, truth, truth, family)
                    assert bool(jnp.isfinite(result) & jnp.isfinite(full))
                    assert bool(jnp.all(jnp.isfinite(jnp.asarray(metrics))))
                    assert all(bool(jnp.all(jnp.isfinite(gradient))) for gradient in grads)
                    if bool(jnp.array_equal(pred, truth)):
                        assert result == full == 0
                        for gradient in grads:
                            np.testing.assert_array_equal(gradient, 0)
        # A zero learned head predicts its own baseline. The floor can only lower its ratio.
        for ratio in (0.0, 1e-6, 0.001, 0.25):
            baseline = (1 - jnp.asarray(ratio, dtype=dtype)) * target
            value, _ = losses(baseline, target, baseline, jnp.zeros_like(target), family)
            residual = np.linalg.norm(np.asarray(jnp.fft.rfft(target, axis=1))
                                      - np.asarray(jnp.fft.rfft(baseline, axis=1)))
            target_norm = np.linalg.norm(np.asarray(jnp.fft.rfft(target, axis=1)))
            expected = residual / np.hypot(residual, config.residual_floor * target_norm)
            np.testing.assert_allclose(value, expected, rtol=2e-6, atol=1e-12)
            assert 0 <= value <= 1
            if ratio >= 0.001:
                np.testing.assert_allclose(value, 1.0, rtol=1e-6)


def test_data_normalization_uses_the_intended_baseline() -> None:
    config = SimpleNamespace(model="cs_dno", data_normalization="nonlinear", data_k_max=None, residual_floor=1e-6,
                             nonlinear_reference=0.002, nonlinear_weight_mean=1.463313)
    losses = trainer_loss(config)
    target = jnp.arange(16, dtype=jnp.float64)[None, :, None]
    prediction = 1.01 * target
    actual, m1 = 0.99 * target, 0.8 * target
    family = jnp.array([PhysicalFamilyId.TANAKA])
    nonlinear, _ = losses(prediction, target, actual, m1, family)
    np.testing.assert_array_equal(losses(prediction, target, target, m1, family)[0], nonlinear)
    assert losses(prediction, target, actual, target, family)[0] < nonlinear
    config.data_normalization = "residual"
    residual, _ = losses(prediction, target, actual, m1, family)
    np.testing.assert_array_equal(losses(prediction, target, actual, target, family)[0], residual)
    assert losses(prediction, target, m1, m1, family)[0] < residual


def test_family_metrics_recombine_uneven_counts_and_isolate_nan() -> None:
    config = SimpleNamespace(model="cs_dno", data_normalization="residual", data_k_max=None, residual_floor=1e-6)
    losses = jax.jit(trainer_loss(config))
    family = jnp.array([1, 1, 1, 2, 2, 2, 3, 3, 4, 1, 2, 4])
    errors = jnp.array([0.0, 0.125, 0.25, 0.5, 0.125, 0.0, 0.25, 0.5, 0.125, 0.5, 0.25, 0.0])
    target = jnp.broadcast_to(jnp.arange(16, dtype=jnp.float64)[None, :, None], (12, 16, 1))
    baseline = 0.75 * target
    prediction = (1 + errors[:, None, None]) * target
    data_loss, (_, flat) = losses(prediction, target, baseline, jnp.zeros_like(target), family)
    metrics = np.asarray(flat).reshape(-1, 5)
    for i, family_id in enumerate(PhysicalFamilyId):
        selected = np.asarray(family == family_id)
        expected = np.array([
            selected.mean(), np.asarray(errors)[selected].sum() / len(family),
            np.asarray(errors)[selected].sum() / len(family) / np.hypot(0.25, config.residual_floor),
            np.sum(np.asarray(errors)[selected] < 0.25) / len(family),
            np.asarray(errors)[selected].sum() / len(family) / np.hypot(0.25, config.residual_floor),
        ])
        np.testing.assert_allclose(metrics[i], expected, rtol=1e-12)
    np.testing.assert_allclose(metrics[:, 4].sum(), data_loss)

    # Extract the actual host epoch reduction, including sample counts and data shares.
    path = REPO_ROOT / "train-jax-10m" / "1d_dno_fno_jax_ema_grad.py"
    reduction = trainer_function("evaluate_validation")
    start = next(index for index, node in enumerate(reduction.body)
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id == "family_totals")
    function = ast.parse(
        "def aggregate(losses: np.ndarray, val_batch_sizes: list[int]) "
        "-> dict[str, float]: metrics = {}"
    ).body[0]
    assert isinstance(function, ast.FunctionDef)
    function.body += reduction.body[start:]
    namespace: dict[str, Any] = {"np": np, "PhysicalFamilyId": PhysicalFamilyId}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), namespace)
    aggregate = namespace["aggregate"]
    full_record = aggregate(np.concatenate((np.zeros(7), metrics.ravel()))[None, :], [12])
    for family_id in PhysicalFamilyId:
        selected_errors = np.asarray(errors)[np.asarray(family == family_id)]
        prefix = f"val_{family_id.name.lower()}"
        expected_record = {
            f"{prefix}_l2_loss": selected_errors.mean(),
            f"{prefix}_baseline_ratio_floored": selected_errors.mean() / np.hypot(0.25, config.residual_floor),
            f"{prefix}_baseline_win_fraction": np.mean(selected_errors < 0.25),
            f"{prefix}_data_loss_share": selected_errors.sum() / np.asarray(errors).sum(),
        }
        np.testing.assert_allclose([full_record[key] for key in expected_record],
                                   list(expected_record.values()), rtol=1e-12)
    for partitions in (((0, 8), (8, 12)), ((0, 7), (7, 12)), ((0, 4), (4, 8), (8, 10), (10, 12))):
        rows, sizes = [], []
        for start, end in partitions:
            _, (_, values) = losses(prediction[start:end], target[start:end], baseline[start:end],
                                    jnp.zeros_like(target[start:end]), family[start:end])
            rows.append(np.concatenate((np.zeros(7), np.asarray(values))))
            sizes.append(end - start)
        record = aggregate(np.stack(rows), sizes)
        assert record.keys() == full_record.keys()
        np.testing.assert_allclose(list(record.values()), list(full_record.values()), rtol=1e-12)
        if len(rows) == 4:
            # Equal-sized device shards are pmean'ed before weighting the uneven 8/4 batches.
            sharded = aggregate(np.stack((np.mean(rows[:2], axis=0), np.mean(rows[2:], axis=0))), [8, 4])
            np.testing.assert_allclose(list(sharded.values()), list(full_record.values()), rtol=1e-12)
    np.testing.assert_allclose(sum(value for key, value in full_record.items()
                                  if key.endswith("data_loss_share")), 1.0)
    failed_loss, (_, failed) = losses(prediction.at[0, 0, 0].set(jnp.nan), target, baseline,
                                     jnp.zeros_like(target), family)
    failed_metrics = np.asarray(failed).reshape(-1, 5)
    assert jnp.isnan(failed_loss)
    assert np.isnan(failed_metrics[0, [1, 2, 4]]).all()
    np.testing.assert_array_equal(failed_metrics[1:], metrics[1:])


def test_tangent_geometry_wiring_and_uneven_shard_reduction() -> None:
    path = REPO_ROOT / "train-jax-10m" / "1d_dno_fno_jax_ema_grad.py"
    component = next(node for node in ast.walk(ast.parse(path.read_text()))
                     if isinstance(node, ast.FunctionDef) and node.name == "compute_loss_components")
    config = SimpleNamespace(
        model="fno", data_k_max=None, residual_floor=1e-6, data_xi_projection_weight=0.0, mode_balanced_weight=0.0,
        translation_tangent_weight=0.3, translation_tangent_scope="tanaka",
        translation_tangent_geometry="coupled_modal", translation_tangent_k_max=1,
        translation_tangent_denominator_eps=1e-3, translation_tangent_loss_norm="charbonnier",
        translation_tangent_charbonnier_eps=1e-7, translation_tangent_smoothing_scale=1.0,
        stokes_relative_tangent=False, phase_mechanism="none", subtract_before_fft=False,
        fp64_residual=False, subtract_before_centering=False, data_phase_projection_weight=0.0,
        data_modal_residual_floor=0.0,
    )

    def apply_model(
        variables: dict[str, jax.Array], inputs: jax.Array, depth: jax.Array,
        *, return_baseline: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        prediction = (variables["params"][..., None] - 7.0) / 4.0
        return prediction, jnp.zeros_like(prediction), jnp.zeros_like(prediction), prediction

    nx = 32
    phase = jnp.arange(nx) * (2 * jnp.pi / nx)
    k = jnp.fft.fftfreq(nx) * nx / 4
    amplitude = jnp.arange(1, 7)[:, None] / 6
    eta = amplitude * (jnp.cos(phase) + 0.3 * jnp.cos(5 * phase))
    xi = amplitude * (0.4 * jnp.sin(phase) + 0.2 * jnp.cos(5 * phase))
    target = amplitude * (0.2 * jnp.cos(phase) + 0.3 * jnp.sin(5 * phase))
    prediction = target + amplitude * (0.03 * jnp.sin(phase) + 0.2 * jnp.cos(5 * phase))
    depth = jnp.array([8.0, 0.5, 2.0, 12.0, 0.4, 1.0])
    family = jnp.array([PhysicalFamilyId.TANAKA, PhysicalFamilyId.STOKES, PhysicalFamilyId.STOKES,
                        PhysicalFamilyId.TANAKA, PhysicalFamilyId.TANAKA, PhysicalFamilyId.STOKES])
    namespace: dict[str, Any] = {
        "jax": jax, "jnp": jnp, "FlatParams": Any, "args": config, "cast": cast,
        "training_dtype": jnp.float64, "physical_dtype": jnp.float64, "PhysicalFamilyId": PhysicalFamilyId,
        "norm_inputs_jax": lambda eta, xi: jnp.stack((eta / 2, xi / 3), axis=-1),
        "norm_targets_jax": lambda target: (target[..., None] - 7.0) / 4.0,
        "denorm_targets_jax": lambda target: target * 4.0 + 7.0,
        "apply_model": apply_model, "log_h_max": np.log(5.0), "k_grid_jax": k,
        "compute_modal_phase_loss": compute_modal_phase_loss,
        "compute_translation_tangent_loss": compute_translation_tangent_loss,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[component], type_ignores=[])), str(path), "exec"), namespace)
    losses = jax.vmap(namespace["compute_loss_components"], axis_name="batch")
    arguments = tuple(field.reshape((2, 3) + field.shape[1:])
                      for field in (eta, xi, target, jnp.log(depth)[:, None], family))
    predictions = prediction.reshape(2, 3, nx)
    target_shards = target.reshape(2, 3, nx)
    options: dict[str, Any] = dict(
        sample_mask=family == PhysicalFamilyId.TANAKA, denominator_eps=1e-3,
        loss_norm="charbonnier", charbonnier_eps=1e-7,
    )
    data_losses = []
    for geometry in ("coupled_modal", "local"):
        config.translation_tangent_geometry = geometry
        data, _, tangent, _, fraction, full, _ = losses(predictions, *arguments)
        data_losses.append(data)
        np.testing.assert_array_equal(data, full)
        np.testing.assert_allclose(fraction, [1 / 3, 2 / 3])
        if geometry == "coupled_modal":
            expected, count = compute_modal_phase_loss(
                eta, xi, prediction, target, depth, k, k_max=1, **options,
            )
            # These alternatives must be distinguishable in this fixture.
            for wrong_xi, wrong_depth, wrong_k_max in (
                (xi / 3, depth, 1), (xi, jnp.minimum(depth, 5.0), 1), (xi, depth, 128),
            ):
                wrong, _ = compute_modal_phase_loss(
                    eta, wrong_xi, prediction, target, wrong_depth, k,
                    k_max=wrong_k_max, **options,
                )
                assert not np.isclose(expected, wrong, rtol=1e-5, atol=1e-10)
        else:
            expected, count = compute_translation_tangent_loss(
                eta, prediction, target, jnp.minimum(depth, 5.0), k,
                smoothing_scale=1.0, **options,
            )
        assert count == 3
        np.testing.assert_allclose(jnp.mean(tangent), expected, rtol=1e-11)

        def objective(params: jax.Array) -> jax.Array:
            data, _, tangent, _, _, _, _ = losses(params, *arguments)
            return jnp.mean(data + config.translation_tangent_weight * tangent)

        evaluate = jax.jit(jax.value_and_grad(objective))
        for params in (predictions, target_shards):
            value, gradient = evaluate(params)
            assert bool(jnp.isfinite(value) & jnp.all(jnp.isfinite(gradient)))
            if params is target_shards:
                np.testing.assert_allclose(value, 0.0, atol=1e-14)
                np.testing.assert_allclose(gradient, 0.0, atol=1e-8)
    config.translation_tangent_weight = 0.0
    ordinary_data = losses(predictions, *arguments)[0]
    for data in data_losses:
        np.testing.assert_array_equal(data, ordinary_data)


def trainer_function(name: str) -> ast.FunctionDef:
    source = ast.parse((REPO_ROOT / "train-jax-10m" / "1d_dno_fno_jax_ema_grad.py").read_text())
    return next(node for node in ast.walk(source)
                if isinstance(node, ast.FunctionDef) and node.name == name)


def trainer_optimizer(main: ast.FunctionDef) -> list[ast.stmt]:
    start = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "inner_optimizer")
    end = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
               and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "optimizer")
    return main.body[start:end + 1]


def test_terms_selector_preserves_order_and_rejects_unused_blocks() -> None:
    main = trainer_function("main")
    options: list[ast.stmt] = [node for node in main.body if isinstance(node, ast.Expr)
                             and isinstance(node.value, ast.Call) and node.value.args
                             and isinstance(node.value.args[0], ast.Constant)
                             and node.value.args[0].value in ("--cs_terms_residual", "--cs_baseline_order")]
    validation = next(node for node in main.body if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "args.cs_terms_residual")
    parser = argparse.ArgumentParser(add_help=False)
    namespace: dict[str, Any] = {"parser": parser}
    exec(compile(ast.Module(body=options, type_ignores=[]), "terms_option", "exec"), namespace)
    assert not parser.parse_args([]).cs_terms_residual
    assert parser.parse_args([]).cs_baseline_order == 1
    assert parser.parse_args(["--cs_terms_residual"]).cs_terms_residual
    valid = dict(
        cs_terms_residual=True, model="cs_dno", norm="scale", cs_normalize_surface_features=True,
        cs_nested_blocks=0, cs_nested_zero_inner=False, cs_nested_surface_modulation=False,
        cs_nested_surface_features=False, cs_surface_context_gate=False, cs_surface_depth_gate=False,
        cs_surface_polynomial_skip=False, cs_multiplier_skip=False, cs_log_multiplier_inputs=False,
        surface_head_lr_multiplier=1.0, cs_baseline_order=1, cs_baseline_true_depth=False,
    )
    code = compile(ast.Module(body=[validation], type_ignores=[]), "terms_validation", "exec")
    for baseline_order in (1, 3):
        assert parser.parse_args(["--cs_terms_residual", "--cs_baseline_order", str(baseline_order)]).cs_baseline_order == baseline_order
        namespace["args"] = SimpleNamespace(**(valid | {"cs_baseline_order": baseline_order}))
        exec(code, namespace)
        assert namespace["args"].cs_baseline_order == baseline_order
        assert namespace["args"].cs_baseline_true_depth
    for baseline_order in (1, 2):
        namespace["args"] = SimpleNamespace(**(valid | {"cs_terms_residual": False, "cs_baseline_order": baseline_order}))
        before = vars(namespace["args"]).copy()
        exec(code, namespace)
        assert vars(namespace["args"]) == before
    with patch.object(parser, "error", side_effect=ValueError):
        namespace["args"] = SimpleNamespace(**(valid | {"cs_terms_residual": False, "cs_baseline_order": 3}))
        with np.testing.assert_raises(ValueError):
            exec(code, namespace)
        for baseline_order in (0, 4):
            with np.testing.assert_raises(ValueError):
                parser.parse_args(["--cs_baseline_order", str(baseline_order)])
        for name, value in (
            ("model", "fno"), ("norm", "minmax"), ("cs_normalize_surface_features", False),
            ("cs_baseline_order", 2),
            ("cs_nested_blocks", 1), ("cs_nested_zero_inner", True),
            ("cs_nested_surface_modulation", True), ("cs_nested_surface_features", True),
            ("cs_surface_context_gate", True), ("cs_surface_depth_gate", True),
            ("cs_surface_polynomial_skip", True), ("cs_multiplier_skip", True),
            ("cs_log_multiplier_inputs", True), ("surface_head_lr_multiplier", 2.0),
        ):
            namespace["args"] = SimpleNamespace(**(valid | {name: value}))
            with np.testing.assert_raises(ValueError):
                exec(code, namespace)

    option = next(node for node in main.body if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call) and node.value.args
                  and isinstance(node.value.args[0], ast.Constant)
                  and node.value.args[0].value == "--cs_terms_cpu_prefetch")
    validation = next(node for node in main.body if isinstance(node, ast.If)
                      and ast.unparse(node.test).startswith("args.cs_terms_cpu_prefetch and"))
    exec(compile(ast.Module(body=[option], type_ignores=[]), "cpu_terms_option", "exec"), namespace)
    assert not parser.parse_args([]).cs_terms_cpu_prefetch
    code = compile(ast.Module(body=[validation], type_ignores=[]), "cpu_terms_validation", "exec")
    valid_cpu = dict(cs_terms_cpu_prefetch=True, cs_terms_residual=True, datum_shift_rms=0.0,
                     reflection_augmentation=False, input_noise_rms=0.0, dirichlet_probe_mode="off")
    namespace["args"] = SimpleNamespace(**valid_cpu)
    exec(code, namespace)
    with patch.object(parser, "error", side_effect=ValueError):
        for name, value in (("cs_terms_residual", False), ("datum_shift_rms", 0.01),
                            ("reflection_augmentation", True), ("input_noise_rms", 0.01),
                            ("dirichlet_probe_mode", "scrambled")):
            namespace["args"] = SimpleNamespace(**(valid_cpu | {name: value}))
            with np.testing.assert_raises(ValueError):
                exec(code, namespace)


def test_terms_trainer_apply_preserves_native_and_hadamard_gradients(baseline_order: int = 1) -> None:
    from hadamard_shape_regularizer import compute_hadamard_loss

    nx = 16
    x = 2 * jnp.pi * jnp.arange(nx) / nx
    inputs = jnp.stack((0.02 * jnp.cos(x), 0.03 * jnp.sin(2 * x)), axis=-1)[None].astype(jnp.float64)
    depth = jnp.log(jnp.asarray([[0.712345678901]], dtype=jnp.float64))
    model = CraigSulemTermsResidual(width=8, baseline_order=baseline_order, surface_context_width=2,
                                   surface_feature_scales=(1.0,) * 7,
                                   eta_scale=0.134123456789, xi_scale=0.317123456789,
                                   target_scale=0.797123456789)
    params = dict(model.init(jax.random.PRNGKey(7), inputs, depth)["params"])
    kernel = cast(jax.Array, params["terms_head"]["kernel"])
    params["terms_head"]["kernel"] = 0.1 * jax.random.normal(
        jax.random.PRNGKey(8), kernel.shape, dtype=kernel.dtype,
    )
    namespace: dict[str, Any] = dict(
        jax=jax, jnp=jnp, Any=Any, cast=cast, model=model,
        args=SimpleNamespace(model="cs_dno", cs_terms_residual=True),
    )
    exec(compile(ast.Module(body=[trainer_function("apply_model")], type_ignores=[]),
                 "terms_trainer_apply", "exec"), namespace)
    apply_model = namespace["apply_model"]
    actual = apply_model({"params": params}, inputs, depth, return_baseline=True)
    expected = model.apply({"params": params}, inputs, depth, return_baseline=True)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(value, reference)

    def loss(parameters: dict[str, Any], *, wrapped: bool) -> jax.Array:
        apply = cast(Callable[..., jax.Array], apply_model if wrapped else model.apply)
        prediction = cast(jax.Array, apply({"params": parameters}, inputs, depth))
        hadamard = compute_hadamard_loss(
            rng=jax.random.PRNGKey(9), apply_fn=apply, model_params=parameters,
            eta_phys=inputs[..., 0] * model.eta_scale, xi_phys=inputs[..., 1] * model.xi_scale,
            batch_depth_local=depth,
            norm_inputs_fn=lambda eta, xi: jnp.stack((eta / model.eta_scale, xi / model.xi_scale), axis=-1),
            denorm_targets_fn=lambda values: values * model.target_scale,
            k=jnp.fft.fftfreq(nx) * nx, dtype=jnp.float64, k_max=5,
        )
        return jnp.sum(prediction**2) + 0.04 * hadamard

    evaluate = jax.jit(jax.value_and_grad(loss), static_argnames="wrapped")
    native = evaluate(params, wrapped=False)
    through_trainer = evaluate(params, wrapped=True)
    for value, reference in zip(jax.tree.leaves(through_trainer), jax.tree.leaves(native), strict=True):
        assert bool(jnp.all(jnp.isfinite(value)))
        np.testing.assert_array_equal(value, reference)
    assert float(jnp.linalg.norm(through_trainer[1]["terms_head"]["kernel"])) > 0
    assert float(jnp.linalg.norm(through_trainer[1]["eta_feat_proj"]["kernel"])) > 0


def test_terms_full_loss_preserves_physical_baseline_and_teacher_precision(baseline_order: int = 1) -> None:
    import optax
    from checkpoint_util import WeightEMATrainState
    from hadamard_shape_regularizer import compute_hadamard_loss
    from solver.evals.model_rollout import rollout_surrogate
    from solver.solvers import time_integrator as ti
    from solver.solvers.dno_series_jax import dno_series_eval
    from util import make_normalizers

    nx = 32
    x = 2 * jnp.pi * jnp.arange(nx, dtype=jnp.float64) / nx
    eta, xi = (0.012 * jnp.cos(x))[None], (0.021 * jnp.sin(x))[None]
    depth = jnp.log(jnp.asarray([[0.712345678901]], dtype=jnp.float64))
    model = CraigSulemTermsResidual(
        width=8, baseline_order=baseline_order, surface_context_width=2, surface_feature_scales=(1.0,) * 7,
        eta_scale=0.134123456789, xi_scale=0.317123456789, target_scale=0.797123456789,
    )
    norm_inputs, norm_targets, denorm_targets = make_normalizers(
        {"feature_absmax": [model.eta_scale, model.xi_scale], "target_absmax": model.target_scale},
        "scale", dtype=np.dtype(np.float64),
    )
    params = model.init(jax.random.PRNGKey(7), norm_inputs(eta, xi), depth)["params"]
    config = SimpleNamespace(
        model="cs_dno", cs_terms_residual=True, fp64_residual=False,
        data_k_max=128, data_normalization="target", residual_floor=1e-6,
        data_xi_projection_weight=0.0, data_phase_projection_weight=0.0, subtract_before_fft=False,
        data_modal_residual_floor=0.0,
        mode_balanced_weight=6.0, mode_balanced_k_max=128,
        mode_balanced_activity_threshold=1e-6, mode_balanced_denominator_eps=1e-12,
        jonswap_mode_balanced_scale=1.0, translation_tangent_weight=10.0,
        translation_tangent_scope="tanaka", translation_tangent_geometry="local",
        translation_tangent_denominator_eps=1e-12, translation_tangent_loss_norm="l2",
        translation_tangent_charbonnier_eps=1e-7, translation_tangent_smoothing_scale=1.0,
        stokes_relative_tangent=False, subtract_before_centering=False, phase_mechanism="all_momentum",
    )
    namespace: dict[str, Any] = dict(
        jax=jax, jnp=jnp, cast=cast, Any=Any, FlatParams=Any, args=config, model=model,
        training_dtype=jnp.float32, physical_dtype=jnp.float64, PhysicalFamilyId=PhysicalFamilyId,
        norm_inputs_jax=norm_inputs, norm_targets_jax=norm_targets, denorm_targets_jax=denorm_targets,
        k_grid_jax=jnp.fft.fftfreq(nx) * nx, k_rfft_jax=jnp.arange(nx // 2 + 1), log_h_max=np.log(5),
        compute_mode_balanced_loss=compute_mode_balanced_loss,
        compute_translation_tangent_loss=compute_translation_tangent_loss,
        compute_phase_mechanism_loss=compute_phase_mechanism_loss,
    )
    exec(compile(ast.Module(body=[trainer_function("apply_model"), trainer_function("compute_loss_components")],
                            type_ignores=[]), "terms_physical_loss", "exec"), namespace)
    components = jax.vmap(namespace["compute_loss_components"], in_axes=(None, 0, 0, 0, 0, 0), axis_name="batch")
    target = dno_series_eval(eta, xi, jnp.arange(nx // 2 + 1), jnp.exp(depth), order=baseline_order, pad_factor=2)
    family = jnp.asarray([PhysicalFamilyId.TANAKA])
    exact = components(params, *(value[None] for value in (eta, xi, target, depth, family)))
    assert float(exact[0][0]) < 1e-12 and float(exact[5][0]) < 1e-12

    trajectory = rollout_surrogate(
        ti.State(eta, xi), jnp.asarray([0.0, 0.02]), ti.make_solver_params(nx, 2 * np.pi, jnp.exp(depth)),
        lambda e, z: denorm_targets(cast(jax.Array, model.apply({"params": params}, norm_inputs(e, z), depth)))[..., 0],
        substeps=2,
    )
    generated = tuple(map(jax.lax.stop_gradient, compute_dno_target(
        trajectory["eta"][-1], trajectory["xi"][-1], jnp.exp(depth), nx=nx, length=2 * np.pi,
        dno_order=6, pad_factor=8, maximum_wavenumber=128.0,
    )))
    assert all(value.dtype == jnp.float64 for value in generated)

    params["terms_head"]["kernel"] = 0.01 * jax.random.normal(
        jax.random.PRNGKey(19), cast(jax.Array, params["terms_head"]["kernel"]).shape, dtype=jnp.float32,
    )
    cached_terms = jax.lax.stop_gradient(dno_series_eval(
        eta, xi, jnp.arange(nx // 2 + 1), jnp.exp(depth), order=baseline_order, pad_factor=2, return_terms=True,
    ))
    assert cached_terms.shape == (1, baseline_order + 1, nx)

    def loss(parameters: dict[str, Any], supplied_terms: jax.Array | None = None) -> jax.Array:
        data, mode, tangent, momentum, _, _, _ = components(
            parameters, *(value[None] for value in (*generated, depth, family)),
        )
        ordinary = components(
            parameters, *(value[None] for value in (eta, xi, target, depth, family)),
            **({"terms_phys": supplied_terms[None]} if supplied_terms is not None else {}),
        )
        hadamard = compute_hadamard_loss(
            rng=jax.random.PRNGKey(11), apply_fn=namespace["apply_model"], model_params=parameters,
            eta_phys=generated[0], xi_phys=generated[1], batch_depth_local=depth,
            norm_inputs_fn=norm_inputs, denorm_targets_fn=denorm_targets,
            k=namespace["k_grid_jax"], dtype=jnp.float64, k_max=12,
        )
        return jnp.sum(data + ordinary[0] + 6 * (mode + ordinary[1]) + 10 * (tangent + ordinary[2])
                       + 0.002574349830754761 * (momentum + ordinary[3])) + 0.04 * hadamard

    evaluate = jax.jit(jax.value_and_grad(loss))
    value, gradients = evaluate(params)
    prefetched = evaluate(params, cached_terms)
    for actual, expected in zip(jax.tree.leaves(prefetched), jax.tree.leaves((value, gradients)), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-10)
    assert jnp.isfinite(value) and value > 0
    assert all(jnp.all(jnp.isfinite(gradient)) for gradient in jax.tree.leaves(gradients))
    assert float(jnp.linalg.norm(gradients["terms_head"]["kernel"])) > 0
    assert all(gradient.dtype == jnp.float32 for gradient in jax.tree.leaves(gradients))
    state = WeightEMATrainState.create(
        apply_fn=model.apply, params=params,
        tx=optax.chain(optax.ema(0.5, debias=True), optax.adamw(8e-5, b1=0.5, b2=0.99, weight_decay=1e-4)),
        weight_ema_params=params, weight_ema_decay=0.995,
    )
    update = jax.jit(lambda current, gradient: current.apply_gradients(grads=gradient))
    state = update(state, jax.tree.map(lambda parameter: jnp.full_like(parameter, 0.01), params))
    native_state, prefetched_state = update(state, gradients), update(state, prefetched[1])
    assert int(native_state.step) == int(prefetched_state.step) == 2
    for actual, expected in zip(jax.tree.leaves(prefetched_state), jax.tree.leaves(native_state), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=1e-9)


def test_learning_rate_hold_schedule_and_validation() -> None:
    import optax

    main = trainer_function("main")
    option = next(node for node in ast.walk(main) if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call) and node.value.args
                  and isinstance(node.value.args[0], ast.Constant)
                  and node.value.args[0].value == "--lr_hold_fraction")
    validation = next(node for node in main.body if isinstance(node, ast.If)
                      and "np.isfinite(args.lr_hold_fraction)" in ast.unparse(node.test))
    start = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "schedule_steps")
    parser = argparse.ArgumentParser(add_help=False)
    namespace: dict[str, Any] = {"parser": parser, "np": np, "optax": optax,
                                 "train_steps_per_epoch": 10000, "schedule_epochs": 1}
    exec(compile(ast.Module(body=[option], type_ignores=[]), "lr_hold_option", "exec"), namespace)
    validate = compile(ast.Module(body=[validation], type_ignores=[]), "lr_hold_validation", "exec")
    configure = compile(ast.Module(body=main.body[start:start + 2], type_ignores=[]), "lr_hold_schedule", "exec")
    steps = jnp.array([0, 250, 500, 4999, 5000, 7500, 10000, 12500])
    for warmup, fraction in product((0, 500), (0.0, 0.5)):
        config = parser.parse_args([] if fraction == 0 else ["--lr_hold_fraction", str(fraction)])
        config.epochs, config.lr, config.lr_warmup_steps = 1, 6e-5, warmup
        namespace["args"] = config
        exec(validate, namespace)
        exec(configure, namespace)
        actual = jax.jit(jax.vmap(namespace["lr_schedule"]))(steps)
        if fraction == 0:
            reference = (optax.warmup_cosine_decay_schedule(0.0, config.lr, warmup, 10000, end_value=0.0)
                         if warmup else optax.cosine_decay_schedule(config.lr, 10000))
            np.testing.assert_array_equal(actual, jax.jit(jax.vmap(reference))(steps))
        else:
            expected = [0, 0.5, 1, 1, 1, 0.5, 0, 0] if warmup else [1, 1, 1, 1, 1, 0.5, 0, 0]
            np.testing.assert_allclose(actual, config.lr * np.asarray(expected), rtol=1e-7, atol=1e-12)
    config = namespace["args"]
    config.lr_hold_fraction = 0.05
    exec(configure, namespace)
    reference = optax.warmup_cosine_decay_schedule(0.0, config.lr, 500, 10000, end_value=0.0)
    np.testing.assert_array_equal(jax.vmap(namespace["lr_schedule"])(steps), jax.vmap(reference)(steps))
    with patch.object(parser, "error", side_effect=ValueError) as error:
        for invalid in (-0.1, 1.0, np.nan, np.inf, -np.inf):
            namespace["args"] = parser.parse_args([f"--lr_hold_fraction={invalid}"])
            with np.testing.assert_raises(ValueError):
                exec(validate, namespace)
            assert "finite and in [0, 1)" in error.call_args.args[0]
        namespace["args"] = config
        config.lr_hold_fraction = 0.025
        with np.testing.assert_raises(ValueError):
            exec(configure, namespace)
        assert "at or after --lr_warmup_steps" in error.call_args.args[0]


def test_surface_head_lr_multiplier_updates_and_weight_ema() -> None:
    import optax
    from checkpoint_util import WeightEMATrainState, training_counter_values

    main = trainer_function("main")
    option = next(node for node in ast.walk(main) if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call) and node.value.args
                  and isinstance(node.value.args[0], ast.Constant)
                  and node.value.args[0].value == "--surface_head_lr_multiplier")
    validation = next(node for node in main.body if isinstance(node, ast.If)
                      and "np.isfinite(args.surface_head_lr_multiplier)" in ast.unparse(node.test))
    params = {
        "cs_block_0": {"phi_proj": {"kernel": jnp.array([0.0, 0.1]), "bias": jnp.array([0.2])},
                       "m_shared": {"kernel": jnp.array([0.3, -0.4])}},
        "eta_feat_proj": {"kernel": jnp.array([0.5, -0.6])},
        "other_block": {"phi_proj": {"kernel": jnp.array([0.7, -0.8])}},
    }
    parser = argparse.ArgumentParser(add_help=False)
    schedule = optax.constant_schedule(6e-5)
    namespace: dict[str, Any] = {"parser": parser, "np": np, "jax": jax, "cast": cast,
                                 "optax": optax, "params": params, "lr_schedule": schedule,
                                 "head_transform": None}
    exec(compile(ast.Module(body=[option], type_ignores=[]), "head_lr_option", "exec"), namespace)
    validate = compile(ast.Module(body=[validation], type_ignores=[]), "head_lr_validation", "exec")
    configure = compile(ast.Module(body=trainer_optimizer(main), type_ignores=[]), "head_optimizer", "exec")
    for multiplier in (1.0, 2.0):
        config = parser.parse_args([] if multiplier == 1 else ["--surface_head_lr_multiplier", str(multiplier)])
        config.gradient_ema_decay, config.adam_b1, config.adam_b2, config.weight_decay = 0.9, 0.9, 0.999, 1e-4
        config.surface_head_muon = False
        namespace["args"] = config
        exec(validate, namespace)
        exec(configure, namespace)
        optimizer = namespace["optimizer"]
        reference = optax.chain(optax.ema(0.9, debias=True), optax.adamw(schedule, weight_decay=1e-4))
        actual_state, reference_state = optimizer.init(params), reference.init(params)
        assert len(actual_state) == (2 if multiplier == 1 else 3)
        for step in range(3):
            grads = jax.tree.map(lambda value: jnp.full_like(value, (-1.0) ** step * (step + 1)), params)
            actual, actual_state = optimizer.update(grads, actual_state, params)
            expected, reference_state = reference.update(grads, reference_state, params)
            expected = cast(dict[str, Any], expected)
            expected["cs_block_0"]["phi_proj"]["kernel"] *= multiplier
            for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
                np.testing.assert_array_equal(actual_leaf, expected_leaf)
            for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual_state[:2]), jax.tree.leaves(reference_state), strict=True):
                np.testing.assert_array_equal(actual_leaf, expected_leaf)
    state = WeightEMATrainState.create(
        apply_fn=Mock(), params=params, tx=namespace["optimizer"], weight_ema_params=params,
        weight_ema_decay=0.5, weight_ema_start_step=0,
    )
    update = jax.jit(lambda current, grads: current.apply_gradients(grads=grads))
    gradients = jax.tree.map(jnp.ones_like, params)
    first = update(state, gradients)
    second = update(first, gradients)
    assert training_counter_values(second.replace(opt_state=second.opt_state[1]), context="head multiplier") == (2, 2)
    for initial, actual, previous, expected in zip(
        jax.tree.leaves(first.weight_ema_params), jax.tree.leaves(second.weight_ema_params),
        jax.tree.leaves(first.params), jax.tree.leaves(second.params), strict=True,
    ):
        np.testing.assert_array_equal(initial, previous)
        np.testing.assert_allclose(actual, 0.5 * (previous + expected), rtol=1e-14)
    with patch.object(parser, "error", side_effect=ValueError) as error:
        for invalid in (0.0, -1.0, np.nan, np.inf, -np.inf):
            namespace["args"] = parser.parse_args([f"--surface_head_lr_multiplier={invalid}"])
            with np.testing.assert_raises(ValueError):
                exec(validate, namespace)
            assert "finite and positive" in error.call_args.args[0]


def test_adam_moment_options_preserve_both_emas() -> None:
    import optax
    from checkpoint_util import WeightEMATrainState

    main = trainer_function("main")
    options: list[ast.stmt] = [node for node in ast.walk(main) if isinstance(node, ast.Expr)
                              and isinstance(node.value, ast.Call) and node.value.args
                              and isinstance(node.value.args[0], ast.Constant)
                              and node.value.args[0].value in ("--adam_b1", "--adam_b2")]
    validations = {name: next(node for node in main.body if isinstance(node, ast.If)
                             and f"args.{name}" in ast.unparse(node.test))
                   for name in ("adam_b1", "adam_b2")}
    parser = argparse.ArgumentParser(add_help=False)
    params = {"kernel": jnp.array([0.0, 0.1], dtype=jnp.float32)}
    schedule = optax.constant_schedule(6e-5)
    namespace: dict[str, Any] = {"parser": parser, "optax": optax, "jax": jax,
                                 "params": params, "lr_schedule": schedule, "head_transform": None}
    exec(compile(ast.Module(body=options, type_ignores=[]), "adam_options", "exec"), namespace)
    validate = compile(ast.Module(body=list(validations.values()), type_ignores=[]), "adam_validation", "exec")
    configure = compile(ast.Module(body=trainer_optimizer(main), type_ignores=[]), "adam_optimizer", "exec")
    states = []
    for arguments, b1, b2 in (([], 0.9, 0.999), (["--adam_b1", "0.5"], 0.5, 0.999),
                              (["--adam_b2", "0.99"], 0.9, 0.99)):
        config = parser.parse_args(arguments)
        assert (config.adam_b1, config.adam_b2) == (b1, b2)
        config.gradient_ema_decay, config.weight_decay, config.surface_head_lr_multiplier = 0.5, 1e-4, 1.0
        config.surface_head_muon = False
        namespace["args"] = config
        exec(validate, namespace)
        exec(configure, namespace)
        states.append(WeightEMATrainState.create(
            apply_fn=Mock(), params=params, tx=namespace["optimizer"], weight_ema_params=params,
            weight_ema_decay=0.995, weight_ema_start_step=1,
        ))
    old_optimizer = optax.chain(optax.ema(0.5, debias=True), optax.adamw(schedule, weight_decay=1e-4))
    old_params, old_state = params, old_optimizer.init(params)
    gradient_ema = optax.ema(0.5, debias=True)
    gradient_state = gradient_ema.init(params)
    update = jax.jit(lambda state, grads: state.apply_gradients(grads=grads))
    for step, values in enumerate(((0.0, 0.0), (1.0, -2.0), (-2.0, 0.5), (0.1, -0.3)), start=1):
        grads = {"kernel": jnp.array(values, dtype=jnp.float32)}
        previous = states
        states = [update(state, grads) for state in states]
        _, gradient_state = gradient_ema.update(grads, gradient_state)
        old_updates, old_state = old_optimizer.update(grads, old_state, old_params)
        old_params = optax.apply_updates(old_params, old_updates)
        for actual, expected in zip(jax.tree.leaves((states[0].params, states[0].opt_state)),
                                    jax.tree.leaves((old_params, old_state)), strict=True):
            np.testing.assert_array_equal(actual, expected)
        for current, prior in zip(states, previous, strict=True):
            assert int(current.step) == step
            for actual, expected in zip(jax.tree.leaves(current.opt_state[0]),
                                        jax.tree.leaves(gradient_state), strict=True):
                np.testing.assert_array_equal(actual, expected)
            expected_shadow = (params if step == 1 else current.params if step == 2 else
                               optax.incremental_update(current.params, prior.weight_ema_params, 0.005))
            for actual, expected in zip(jax.tree.leaves(current.weight_ema_params),
                                        jax.tree.leaves(expected_shadow), strict=True):
                np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-12)
        adam_default, adam_b1_trial, adam_b2_trial = (state.opt_state[1][0] for state in states)
        np.testing.assert_array_equal(adam_default.nu["kernel"], adam_b1_trial.nu["kernel"])
        np.testing.assert_array_equal(adam_default.mu["kernel"], adam_b2_trial.mu["kernel"])
        assert int(adam_default.count) == int(adam_b1_trial.count) == int(adam_b2_trial.count) == step
        if step > 1:
            assert not np.array_equal(adam_default.mu["kernel"], adam_b1_trial.mu["kernel"])
            assert not np.array_equal(adam_default.nu["kernel"], adam_b2_trial.nu["kernel"])
    for trial in states[1:]:
        assert not np.array_equal(states[0].params["kernel"], trial.params["kernel"])
    with patch.object(parser, "error", side_effect=ValueError) as error:
        for name in validations:
            for invalid in (-0.1, 1.0, np.nan, np.inf, -np.inf):
                namespace["args"] = parser.parse_args([f"--{name}={invalid}"])
                with np.testing.assert_raises(ValueError):
                    exec(validate, namespace)
                assert f"--{name} must be in [0, 1)" in error.call_args.args[0]
            namespace["args"] = parser.parse_args([f"--{name}=0"])
            exec(validate, namespace)


def test_head_whitening_calibration_and_physical_optimizer_coordinates() -> None:
    import optax
    from checkpoint_util import WeightEMATrainState, training_counter_values

    main = trainer_function("main")
    option = next(node for node in ast.walk(main) if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call) and node.value.args
                  and isinstance(node.value.args[0], ast.Constant)
                  and node.value.args[0].value == "--surface_head_whitening")
    validation = next(node for node in main.body if isinstance(node, ast.If)
                      and ast.unparse(node.test).startswith("args.surface_head_whitening and"))
    parser = argparse.ArgumentParser(add_help=False)
    option_namespace: dict[str, Any] = {"parser": parser}
    exec(compile(ast.Module(body=[option], type_ignores=[]), "head_whitening_option", "exec"), option_namespace)
    assert not parser.parse_args([]).surface_head_whitening
    valid = dict(surface_head_whitening=True, model="cs_dno", cs_terms_residual=False,
                 cs_nested_blocks=1, n_blocks=8, init_checkpoint=None)
    validate = compile(ast.Module(body=[validation], type_ignores=[]), "head_whitening_validation", "exec")
    option_namespace["args"] = SimpleNamespace(**valid)
    exec(validate, option_namespace)
    with patch.object(parser, "error", side_effect=ValueError):
        for invalid in ({"model": "fno"}, {"cs_terms_residual": True},
                        {"cs_nested_blocks": 8}, {"init_checkpoint": "saved"}):
            option_namespace["args"] = SimpleNamespace(**(valid | invalid))
            with np.testing.assert_raises(ValueError):
                exec(validate, option_namespace)
    calibration = next(node for node in main.body if isinstance(node, ast.If)
                       and ast.unparse(node.test) == "args.surface_head_whitening")
    math_block = next(node for node in calibration.body if isinstance(node, ast.With))
    eigen_start = next(i for i, node in enumerate(math_block.body) if isinstance(node, ast.Assign)
                       and isinstance(node.targets[0], ast.Tuple))
    calibrate = compile(ast.Module(body=math_block.body[eigen_start:], type_ignores=[]),
                        "head_whitening_calibration", "exec")
    configure = compile(ast.Module(body=[trainer_function("transform_surface_heads"),
                                        *trainer_optimizer(main)], type_ignores=[]),
                        "head_whitening_optimizer", "exec")
    vectors, _ = np.linalg.qr(np.random.default_rng(41).normal(size=(4, 4)))
    covariance = (vectors * np.array([9.0, 1.0, 0.02, 0.0])) @ vectors.T
    for dtype, tolerance in ((jnp.float32, 3e-6), (jnp.float64, 2e-12)):
        namespace: dict[str, Any] = {
            "np": np, "jnp": jnp, "jax": jax, "Any": Any, "optax": optax,
            "covariance": covariance, "head_width": 4, "training_dtype": dtype,
        }
        exec(calibrate, namespace)
        transform = namespace["head_transform"]
        assert transform.dtype == dtype
        np.testing.assert_allclose(transform, transform.T, rtol=0, atol=tolerance)
        np.testing.assert_allclose(np.sum(np.asarray(transform)**2) / 4, 1.0, atol=tolerance)
        assert np.linalg.eigvalsh(transform).min() > 0
        np.testing.assert_allclose(np.linalg.cond(transform), np.sqrt(1000), rtol=tolerance)
        np.testing.assert_array_equal(transform @ jnp.zeros((4, 2), dtype), 0)
        namespace["covariance"] = 7.0 * covariance
        exec(calibrate, namespace)
        np.testing.assert_allclose(namespace["head_transform"], transform, rtol=tolerance)
        namespace["covariance"] = np.zeros((4, 4))
        with np.testing.assert_raises(FloatingPointError):
            exec(calibrate, namespace)

        params = {
            "cs_block_0": {"phi_proj": {"kernel": jnp.zeros((4, 2), dtype)},
                           "m_shared": {"kernel": jnp.ones((4, 2), dtype)}},
            "cs_block_1": {"phi_proj": {"kernel": jnp.arange(8, dtype=dtype).reshape(4, 2) / 10}},
            "cs_block_2": {"gamma": jnp.array([0.3, -0.2], dtype)},
            "eta_feat_proj": {"kernel": jnp.array([0.1, -0.4], dtype)},
            "other_block": {"phi_proj": {"kernel": jnp.array([0.2, -0.1], dtype)}},
        }

        def change_heads(tree: Any, matrix: jax.Array) -> Any:
            changed = jax.tree.map(lambda value: value, tree)
            for name in ("cs_block_0", "cs_block_1"):
                changed[name]["phi_proj"]["kernel"] = matrix @ tree[name]["phi_proj"]["kernel"]
            return changed

        schedule = optax.exponential_decay(1e-3, transition_steps=1, decay_rate=0.9)
        config = SimpleNamespace(gradient_ema_decay=0.5, adam_b1=0.5, adam_b2=0.99,
                                 weight_decay=0.07, surface_head_lr_multiplier=1.0, surface_head_muon=False)
        namespace.update(args=config, params=params, lr_schedule=schedule)
        for selected_transform in (None, transform):
            namespace["head_transform"] = selected_transform
            physical_initial = params if selected_transform is None else change_heads(params, transform)
            namespace["params"] = physical_initial
            exec(configure, namespace)
            actual = WeightEMATrainState.create(
                apply_fn=Mock(), params=physical_initial, tx=namespace["optimizer"],
                weight_ema_params=physical_initial, weight_ema_decay=0.995, weight_ema_start_step=1,
            )
            coordinates = params
            reference = WeightEMATrainState.create(
                apply_fn=Mock(), params=coordinates,
                tx=optax.chain(optax.ema(0.5, debias=True),
                               optax.adamw(schedule, b1=0.5, b2=0.99, weight_decay=0.07)),
                weight_ema_params=coordinates, weight_ema_decay=0.995, weight_ema_start_step=1,
            )
            update = jax.jit(lambda state, grads: state.apply_gradients(grads=grads))
            for step in range(1, 7):
                grads = jax.tree.map(
                    lambda value: (0.2 * jnp.sin(jnp.arange(value.size, dtype=dtype).reshape(value.shape)
                                                 + step) + 0.1 * value).astype(dtype), actual.params,
                )
                coordinate_grads = (grads if selected_transform is None
                                    else change_heads(grads, transform.T))
                actual = update(actual, grads)
                reference = update(reference, coordinate_grads)
                expected_params, expected_shadow = reference.params, reference.weight_ema_params
                if selected_transform is not None:
                    expected_params = change_heads(expected_params, transform)
                    expected_shadow = change_heads(expected_shadow, transform)
                for observed, expected in zip(jax.tree.leaves((actual.params, actual.weight_ema_params)),
                                               jax.tree.leaves((expected_params, expected_shadow)), strict=True):
                    assert observed.dtype == expected.dtype == dtype
                    if selected_transform is None:
                        np.testing.assert_array_equal(observed, expected)
                    else:
                        np.testing.assert_allclose(observed, expected, rtol=tolerance, atol=tolerance * 0.01)
                for observed, expected in zip(jax.tree.leaves(actual.opt_state[1]),
                                               jax.tree.leaves(reference.opt_state[1]), strict=True):
                    assert observed.dtype == expected.dtype
                    np.testing.assert_allclose(observed, expected, rtol=tolerance, atol=tolerance * 0.01)
                expected_gradient_ema = (actual.opt_state[0].ema if selected_transform is None
                                         else change_heads(actual.opt_state[0].ema, transform.T))
                for observed, expected in zip(jax.tree.leaves(expected_gradient_ema),
                                               jax.tree.leaves(reference.opt_state[0].ema), strict=True):
                    np.testing.assert_allclose(observed, expected, rtol=tolerance, atol=tolerance * 0.01)
                for name in ("cs_block_2", "eta_feat_proj", "other_block"):
                    jax.tree.map(np.testing.assert_array_equal, actual.params[name], reference.params[name])
                np.testing.assert_array_equal(actual.params["cs_block_0"]["m_shared"]["kernel"],
                                              reference.params["cs_block_0"]["m_shared"]["kernel"])
                assert training_counter_values(actual.replace(opt_state=actual.opt_state[1]),
                                               context="head whitening") == (step, step)
                assert int(actual.opt_state[0].count) == int(reference.opt_state[0].count) == step


def test_packed_surface_head_muon_and_unchanged_adam_emas_counters() -> None:
    import optax
    from checkpoint_util import WeightEMATrainState, read_parameter_update_count, training_counter_values

    main = trainer_function("main")
    options: list[ast.stmt] = [node for node in main.body if isinstance(node, ast.Expr)
               and isinstance(node.value, ast.Call) and node.value.args
               and isinstance(node.value.args[0], ast.Constant)
               and node.value.args[0].value in ("--surface_head_muon", "--surface_head_muon_scale")]
    checks: list[ast.stmt] = [node for node in main.body if isinstance(node, ast.If)
              and (ast.unparse(node.test).startswith("args.surface_head_muon and")
                   or ast.unparse(node.test).startswith("not np.isfinite(args.surface_head_muon_scale)"))]
    parser = argparse.ArgumentParser(add_help=False)
    namespace: dict[str, Any] = {"parser": parser, "np": np}
    exec(compile(ast.Module(body=options, type_ignores=[]), "muon_options", "exec"), namespace)
    assert not parser.parse_args([]).surface_head_muon
    assert parser.parse_args([]).surface_head_muon_scale == 20.0
    config = SimpleNamespace(
        surface_head_muon=True, surface_head_muon_scale=20.0, surface_head_whitening=False,
        model="cs_dno", cs_terms_residual=False, n_blocks=8, cs_nested_blocks=1,
        init_checkpoint=None, gradient_ema_decay=0.5, adam_b1=0.5, adam_b2=0.99,
        weight_decay=0.07, surface_head_lr_multiplier=1.0,
    )
    namespace["args"] = config
    validate = compile(ast.Module(body=checks, type_ignores=[]), "muon_validation", "exec")
    exec(validate, namespace)
    with patch.object(parser, "error", side_effect=ValueError):
        for change in ({"model": "fno"}, {"cs_terms_residual": True}, {"cs_nested_blocks": 8},
                       {"init_checkpoint": "saved"}, {"surface_head_whitening": True},
                       {"surface_head_lr_multiplier": 2.0},
                       *({"surface_head_muon_scale": value} for value in (0, -1, np.nan, np.inf))):
            namespace["args"] = SimpleNamespace(**(vars(config) | change))
            with np.testing.assert_raises(ValueError):
                exec(validate, namespace)

    names = tuple(f"cs_block_{i}" for i in range(7))
    params: dict[str, Any] = {name: {"phi_proj": {"kernel": jnp.zeros((160, 4), jnp.float32),
                                 "bias": jnp.array([0.2], jnp.float32)},
                     "m_shared": {"kernel": jnp.ones((4, 2), jnp.float32)}} for name in reversed(names)}
    params["cs_block_7"] = {"gamma": jnp.array([0.2, -0.1, 0.3, 0.4], jnp.float32)}
    params["eta_feat_proj"] = {"kernel": jnp.array([0.1, -0.3], jnp.float32)}
    params["other_block"] = {"phi_proj": {"kernel": jnp.ones((2, 2), jnp.float32)}}
    schedule = optax.exponential_decay(8e-5, transition_steps=1, decay_rate=0.9)
    namespace.update(args=config, params=params, lr_schedule=schedule, head_transform=None,
                     jax=jax, jnp=jnp, optax=optax, Any=Any, cast=cast, WeightEMATrainState=WeightEMATrainState,
                     training_counter_values=training_counter_values,
                     read_parameter_update_count=read_parameter_update_count)
    exec(compile(ast.Module(body=[*trainer_optimizer(main), trainer_function("check_training_counters")],
                            type_ignores=[]), "muon_optimizer", "exec"), namespace)
    common = dict(apply_fn=Mock(), weight_ema_decay=0.995, weight_ema_secondary_decay=0.9,
                  weight_ema_start_step=1)
    actual: Any = WeightEMATrainState.create(params=params, tx=namespace["optimizer"],
                                       weight_ema_params=params, weight_ema_secondary_params=params, **common)
    adam: Any = WeightEMATrainState.create(
        params=params, weight_ema_params=params, weight_ema_secondary_params=params,
        tx=optax.chain(optax.ema(0.5, debias=True),
                       optax.adamw(schedule, b1=0.5, b2=0.99, weight_decay=0.07)), **common,
    )
    packed = jnp.concatenate([params[name]["phi_proj"]["kernel"] for name in names], axis=-1)
    muon: Any = WeightEMATrainState.create(
        params=packed, weight_ema_params=packed, weight_ema_secondary_params=packed,
        tx=optax.chain(optax.ema(0.5, debias=True),
                       optax.contrib.scale_by_muon(beta=0.5, ns_steps=5, nesterov=True),
                       optax.scale(20), optax.add_decayed_weights(0.07),
                       optax.scale_by_learning_rate(schedule)), **common,
    )
    update = jax.jit(lambda state, grads: state.apply_gradients(grads=grads))
    check_counters = namespace["check_training_counters"]
    for step in range(1, 7):
        grads = jax.tree.map(lambda value: jnp.zeros_like(value) if step == 1 else
                             0.1 * jnp.sin(jnp.arange(value.size, dtype=jnp.float32).reshape(value.shape)
                                           + step), params)
        for index, name in enumerate(names):
            if step > 1:
                grads[name]["phi_proj"]["kernel"] = (index + 1) * jax.random.normal(
                    jax.random.fold_in(jax.random.PRNGKey(step), index), (160, 4), dtype=jnp.float32,
                )
        packed_grads = jnp.concatenate([grads[name]["phi_proj"]["kernel"] for name in names], axis=-1)
        actual, adam, muon = update(actual, grads), update(adam, grads), update(muon, packed_grads)
        assert check_counters(actual, context="Muon test") == (step, step)
        assert actual.opt_state[1][0][1].mu.shape == (160, 28)
        jax.tree.map(np.testing.assert_array_equal, actual.opt_state[0], adam.opt_state[0])
        np.testing.assert_array_equal(actual.opt_state[1][0][1].mu, muon.opt_state[1].mu)
        for attribute in ("params", "weight_ema_params", "weight_ema_secondary_params"):
            observed_leaves = jax.tree_util.tree_flatten_with_path(getattr(actual, attribute))[0]
            for (path, observed), expected in zip(observed_leaves, jax.tree.leaves(getattr(adam, attribute)), strict=True):
                assert np.isfinite(observed).all() and observed.dtype == jnp.float32
                if (len(path) == 3 and path[0].key in names
                        and path[1:] == (jax.tree_util.DictKey("phi_proj"), jax.tree_util.DictKey("kernel"))):
                    index = names.index(path[0].key)
                    expected = getattr(muon, attribute)[:, 4*index:4*index+4]
                    np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=1e-9)
                else:
                    np.testing.assert_array_equal(observed, expected)
        if step == 1:
            for name in names:
                np.testing.assert_array_equal(actual.params[name]["phi_proj"]["kernel"], 0)
    inner = actual.opt_state[1]
    bad_muon = (inner[0][0], inner[0][1]._replace(count=jnp.asarray(5, jnp.int32)))
    bad_adam = (inner[0][0]._replace(inner_state=inner[0][0].inner_state._replace(count=jnp.asarray(5, jnp.int32))),
                inner[0][1])
    for modified in ((bad_muon, *inner[1:]), (bad_adam, *inner[1:]),
                     (*inner[:-1], inner[-1]._replace(count=jnp.asarray(5, jnp.int32)))):
        with np.testing.assert_raises(RuntimeError):
            check_counters(actual.replace(opt_state=(actual.opt_state[0], modified)), context="bad count")
    with np.testing.assert_raises(RuntimeError):
        check_counters(actual.replace(step=actual.step + 1), context="bad TrainState")


def test_milestone_rollouts_use_only_started_weight_ema() -> None:
    main = trainer_function("main")
    validation = next(node for node in ast.walk(main) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "args.rollout_epochs and args.weight_ema_decay == 0.0")
    schedule_validation = next(node for node in ast.walk(main) if isinstance(node, ast.If)
                               and ast.unparse(node.test) == "args.rollout_epochs and min(args.rollout_epochs) <= schedule_epochs * args.weight_ema_start_fraction")
    milestone = next(node for node in ast.walk(main) if isinstance(node, ast.If)
                     and ast.unparse(node.test) == "epoch in args.rollout_epochs")
    checkpoint = next(node for node in ast.walk(main) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "args.weight_ema_decay > 0.0 and epoch_end_step > training_state.weight_ema_start_step")
    config = SimpleNamespace(rollout_epochs=[1], epochs=1, weight_ema_decay=0.999,
                             weight_ema_start_fraction=0.8, norm="scale",
                             rollout_truth_run="/tmp/ema-milestone-reference")
    state = SimpleNamespace(params=object(), weight_ema_params=object(), weight_ema_start_step=80)
    families = ("stokes", "tanaka", "jonswap_tma", "benjamin_feir")
    loaded, predict, evaluate, save, write = (Mock() for _ in range(5))
    namespace: dict[str, Any] = {
        "args": config, "training_state": state, "epoch": 1, "epoch_end_step": 100, "schedule_epochs": 1,
        "run_dir": Path("/tmp/ema-milestone-run"), "dataset_path": Path("/tmp/ema-milestone-data"),
        "model": object(), "config_payload": {}, "stats": {}, "history": [],
        "best_val_loss": 0.0, "best_epoch": 1, "checkpoint_async_manager": object(),
        "LoadedRun": loaded, "build_predict_gxi_batched": predict, "run_family": evaluate,
        "save_checkpoint": save, "_write_json": write, "compute_macro_summary": Mock(),
        "FAMILY_CONFIGS": dict.fromkeys(families), "Path": Path, "print": Mock(),
        "parser": SimpleNamespace(error=Mock(side_effect=ValueError)),
    }
    validate = compile(ast.Module(body=[validation, schedule_validation], type_ignores=[]), "milestone_validation", "exec")
    evaluate_milestone = compile(ast.Module(body=[checkpoint, milestone], type_ignores=[]), "ema_milestone", "exec")
    exec(validate, namespace)
    config.weight_ema_decay = 0.0
    with np.testing.assert_raises(ValueError):
        exec(validate, namespace)
    assert "requires --weight_ema_decay" in namespace["parser"].error.call_args.args[0]
    config.weight_ema_decay = 0.999
    config.epochs = 5
    namespace["schedule_epochs"] = 5
    for milestones in ([3], [4], [4, 5]):
        config.rollout_epochs = milestones
        with np.testing.assert_raises(ValueError):
            exec(validate, namespace)
        assert "must follow the weight-EMA start" in namespace["parser"].error.call_args.args[0]
    config.rollout_epochs = [5]
    exec(validate, namespace)
    config.epochs, config.rollout_epochs = 1, [1]
    with patch.object(Path, "mkdir"):
        exec(evaluate_milestone, namespace)
    assert loaded.call_args.args[1] is state.weight_ema_params
    assert loaded.call_args.args[1] is not state.params
    assert save.call_count == 1 and save.call_args.kwargs["weight_ema"] is True
    assert save.call_args.args[0] == namespace["run_dir"] / "weight_ema_ckpt"
    assert evaluate.call_count == 4 and write.call_count == 4
    for call, family in zip(evaluate.call_args_list, families, strict=True):
        group = 0 if family in ("stokes", "tanaka") else 1
        assert call.args[0] == family
        assert call.args[4] == namespace["run_dir"] / f"eval_epoch1_weight_ema_test_stratified_n32_gpu{group}"
        assert call.args[6] == call.args[9] == 32
        assert call.args[8] == {"run_dir": str(namespace["run_dir"]), "selection": "weight_ema",
                                "path": str(namespace["run_dir"] / "weight_ema_ckpt"), "epoch": 1}


def test_nested_zero_inner_initialization_and_gradient_activation() -> None:
    main = trainer_function("main")
    options: list[ast.stmt] = [node for node in ast.walk(main) if isinstance(node, ast.Expr)
                              and isinstance(node.value, ast.Call) and node.value.args
                              and isinstance(node.value.args[0], ast.Constant)
                              and node.value.args[0].value in ("--cs_nested_blocks", "--cs_nested_zero_inner")]
    validation = next(node for node in main.body if isinstance(node, ast.If)
                      and "args.cs_nested_zero_inner" in ast.unparse(node.test))
    initialization = next(node for node in main.body if isinstance(node, ast.With)
                          and any(isinstance(child, ast.Assign) and isinstance(child.targets[0], ast.Name)
                                  and child.targets[0].id == "params" for child in node.body))
    replacement = next(node for node in initialization.body if isinstance(node, ast.If)
                       and ast.unparse(node.test) == "args.cs_nested_zero_inner")
    parser = argparse.ArgumentParser(add_help=False)
    namespace: dict[str, Any] = {"parser": parser, "jax": jax, "jnp": jnp, "cast": cast}
    exec(compile(ast.Module(body=options, type_ignores=[]), "nested_zero_inner_options", "exec"), namespace)
    validate = compile(ast.Module(body=[validation], type_ignores=[]), "nested_zero_inner_validation", "exec")
    initialize = compile(ast.Module(body=[initialization.body[0], replacement], type_ignores=[]),
                         "nested_zero_inner_initialization", "exec")
    assert parser.parse_args([]).cs_nested_zero_inner is False
    namespace["args"] = parser.parse_args(["--cs_nested_zero_inner"])
    with patch.object(parser, "error", side_effect=ValueError):
        with np.testing.assert_raises(ValueError):
            exec(validate, namespace)
    nx = 32
    inputs = 0.05 * jax.random.normal(jax.random.PRNGKey(41), (2, nx, 2), dtype=jnp.float32)
    depth = jnp.log(jnp.asarray([[0.8], [8.0]], dtype=jnp.float32))
    for nested_blocks in (1, 2):
        model = CraigSulemDNO(
            width=8, n_blocks=2, nested_blocks=nested_blocks, latent=2, mult_hidden=4,
            surface_context_width=2, surface_feature_scales=(0.3, 0.1, 0.04, 0.7, 2.0, 0.5, 0.3),
            baseline_true_depth=True, eta_scale=0.13, xi_scale=0.08, target_scale=0.12,
        )
        namespace.update(model=model, nx=nx, training_dtype=jnp.float32)
        reference = model.init(jax.random.PRNGKey(0), jnp.zeros((1, nx, 2), dtype=jnp.float32),
                               jnp.zeros((1, 1), dtype=jnp.float32))["params"]
        baseline = cast(tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                        model.apply({"params": reference}, inputs, depth, return_baseline=True))[1]
        target = baseline + 0.02 * jax.random.normal(jax.random.PRNGKey(42), baseline.shape, dtype=baseline.dtype)
        for enabled in (False, True):
            config = parser.parse_args(["--cs_nested_blocks", str(nested_blocks)]
                                       + (["--cs_nested_zero_inner"] if enabled else []))
            config.seed, config.n_blocks = 0, 2
            namespace["args"] = config
            exec(validate, namespace)
            exec(initialize, namespace)
            params = namespace["params"]
            assert jax.tree.structure(params) == jax.tree.structure(reference)
            assert sum(leaf.size for leaf in jax.tree.leaves(params)) == sum(leaf.size for leaf in jax.tree.leaves(reference))
            for (path, actual), expected in zip(jax.tree.flatten_with_path(params)[0], jax.tree.leaves(reference), strict=True):
                assert actual.dtype == expected.dtype
                names = tuple(key.key for key in path)
                nested = names[0] in tuple(f"cs_block_{i}" for i in range(2 - nested_blocks, 2))
                if enabled and nested and names[1:] == ("gamma",):
                    np.testing.assert_array_equal(actual, jnp.ones_like(expected))
                elif enabled and nested and names[1:3] == ("m_inner", "out"):
                    np.testing.assert_array_equal(actual, jnp.zeros_like(expected))
                else:
                    np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(model.apply({"params": params}, inputs, depth), baseline)
            value, gradient = jax.value_and_grad(lambda parameters: relative_l2_loss(
                cast(jax.Array, model.apply({"params": parameters}, inputs, depth)), target,
            ))(params)
            assert value > 0 and all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(gradient))
            for index in range(2 - nested_blocks, 2):
                group = gradient[f"cs_block_{index}"]
                assert (float(jnp.linalg.norm(group["gamma"])) == 0) == enabled
                for leaf in jax.tree.leaves((group["m_shared"], group["m_inner"]["hidden"])):
                    np.testing.assert_array_equal(leaf, 0)
                for leaf in jax.tree.leaves(group["m_inner"]["out"]):
                    assert (float(jnp.linalg.norm(leaf)) > 0) == enabled


def test_phase_rank_one_option_requires_m1_cs_network() -> None:
    main = trainer_function("main")
    option = next(node for node in main.body if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call) and node.value.args
                  and isinstance(node.value.args[0], ast.Constant)
                  and node.value.args[0].value == "--cs_phase_rank_one")
    validation = next(node for node in main.body if isinstance(node, ast.If)
                      and ast.unparse(node.test).startswith("args.cs_phase_rank_one"))
    parser = argparse.ArgumentParser(add_help=False)
    namespace: dict[str, Any] = {"parser": parser}
    exec(compile(ast.Module(body=[option], type_ignores=[]), "phase_rank_one_option", "exec"), namespace)
    assert not parser.parse_args([]).cs_phase_rank_one
    assert parser.parse_args(["--cs_phase_rank_one"]).cs_phase_rank_one
    valid = dict(cs_phase_rank_one=True, model="cs_dno", cs_terms_residual=False, cs_baseline_order=1)
    code = compile(ast.Module(body=[validation], type_ignores=[]), "phase_rank_one_validation", "exec")
    namespace["args"] = SimpleNamespace(**valid)
    exec(code, namespace)
    with patch.object(parser, "error", side_effect=ValueError):
        for name, value in (("model", "fno"), ("cs_terms_residual", True), ("cs_baseline_order", 2)):
            namespace["args"] = SimpleNamespace(**(valid | {name: value}))
            with np.testing.assert_raises(ValueError):
                exec(code, namespace)
            namespace["args"].cs_phase_rank_one = False
            exec(code, namespace)


def test_packed_nested_branches_match_native_values_and_gradients() -> None:
    check_packed_nested_branches(phase_wavenumber_hidden=0)


def test_phase_wavenumber_option_requires_m1_and_excludes_rank_one() -> None:
    main = trainer_function("main")
    option = next(node for node in main.body if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call) and node.value.args
                  and isinstance(node.value.args[0], ast.Constant)
                  and node.value.args[0].value == "--cs_phase_wavenumber_hidden")
    validation: list[ast.stmt] = [node for node in main.body if isinstance(node, ast.If)
                                 and ast.unparse(node.test).startswith("args.cs_phase_wavenumber_hidden")]
    parser = argparse.ArgumentParser(add_help=False)
    namespace: dict[str, Any] = {"parser": parser}
    exec(compile(ast.Module(body=[option], type_ignores=[]), "phase_wavenumber_option", "exec"), namespace)
    assert parser.parse_args([]).cs_phase_wavenumber_hidden == 0
    assert parser.parse_args(["--cs_phase_wavenumber_hidden", "8"]).cs_phase_wavenumber_hidden == 8
    valid = dict(cs_phase_wavenumber_hidden=8, cs_phase_rank_one=False, model="cs_dno",
                 cs_terms_residual=False, cs_baseline_order=1)
    code = compile(ast.Module(body=validation, type_ignores=[]), "phase_wavenumber_validation", "exec")
    namespace["args"] = SimpleNamespace(**valid)
    exec(code, namespace)
    with patch.object(parser, "error", side_effect=ValueError):
        for name, value in (("cs_phase_wavenumber_hidden", -1), ("model", "fno"),
                            ("cs_terms_residual", True), ("cs_baseline_order", 2), ("cs_phase_rank_one", True)):
            namespace["args"] = SimpleNamespace(**(valid | {name: value}))
            with np.testing.assert_raises(ValueError):
                exec(code, namespace)
            namespace["args"].cs_phase_wavenumber_hidden = 0
            exec(code, namespace)


def test_packed_wavenumber_phase_matches_native_values_and_gradients() -> None:
    check_packed_nested_branches(phase_wavenumber_hidden=8)


def test_packed_surface_amplitude_matches_native_values_and_gradients() -> None:
    check_packed_nested_branches(phase_wavenumber_hidden=0, surface_amplitude_reference=0.07)


def test_packed_parity_streams_match_native_values_and_gradients() -> None:
    check_packed_nested_branches(phase_wavenumber_hidden=0, surface_parity_streams=True)


def test_pair_coupling_options_values_and_gradients() -> None:
    main = trainer_function("main")
    option = next(node for node in main.body if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call) and node.value.args
                  and isinstance(node.value.args[0], ast.Constant)
                  and node.value.args[0].value == "--cs_pair_coupling")
    validation = next(node for node in main.body if isinstance(node, ast.If)
                      and ast.unparse(node.test).startswith("args.cs_pair_coupling"))
    parser = argparse.ArgumentParser(add_help=False)
    namespace: dict[str, Any] = {"parser": parser}
    exec(compile(ast.Module(body=[option], type_ignores=[]), "pair_coupling_option", "exec"), namespace)
    assert parser.parse_args([]).cs_pair_coupling is False
    assert parser.parse_args(["--cs_pair_coupling"]).cs_pair_coupling is True
    code = compile(ast.Module(body=[validation], type_ignores=[]), "pair_coupling_validation", "exec")
    valid = dict(cs_pair_coupling=True, model="cs_dno", cs_terms_residual=False,
                 latent=4, n_blocks=8, cs_nested_blocks=1)
    for nested in (0, 1, 7):
        namespace["args"] = SimpleNamespace(**(valid | {"cs_nested_blocks": nested}))
        exec(code, namespace)
    with patch.object(parser, "error", side_effect=ValueError):
        for field, value in (("model", "fno"), ("cs_terms_residual", True), ("latent", 0),
                             ("latent", 1), ("latent", 3), ("cs_nested_blocks", 8)):
            namespace["args"] = SimpleNamespace(**(valid | {field: value}))
            with np.testing.assert_raises(ValueError):
                exec(code, namespace)
            namespace["args"].cs_pair_coupling = False
            exec(code, namespace)
    constructor = next(node for node in ast.walk(main) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Name) and node.func.id == "model_class")
    assert any(keyword.arg == "pair_coupling" and ast.unparse(keyword.value) == "args.cs_pair_coupling"
               for keyword in constructor.keywords)
    check_packed_nested_branches(phase_wavenumber_hidden=0, pair_coupling=True)


def test_mixed_surface_lift_options_values_and_gradients() -> None:
    main = trainer_function("main")
    option = next(node for node in main.body if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call) and node.value.args
                  and isinstance(node.value.args[0], ast.Constant)
                  and node.value.args[0].value == "--cs_surface_cubic_channels")
    validations: list[ast.stmt] = [node for node in main.body if isinstance(node, ast.If)
                                  and "args.cs_surface_cubic_channels" in ast.unparse(node.test)]
    parser = argparse.ArgumentParser(add_help=False)
    namespace: dict[str, Any] = {"parser": parser}
    exec(compile(ast.Module(body=[option], type_ignores=[]), "mixed_lift_option", "exec"), namespace)
    assert parser.parse_args([]).cs_surface_cubic_channels == 0
    assert parser.parse_args(["--cs_surface_cubic_channels", "32"]).cs_surface_cubic_channels == 32
    code = compile(ast.Module(body=validations, type_ignores=[]), "mixed_lift_validation", "exec")
    valid = dict(cs_surface_cubic_channels=2, width=8, model="cs_dno", cs_baseline_order=1,
                 cs_terms_residual=False, cs_surface_parity_streams=False, cs_surface_amplitude_normalization=False)
    for channels in (0, 2, 4):
        namespace["args"] = SimpleNamespace(**(valid | {"cs_surface_cubic_channels": channels}))
        exec(code, namespace)
    with patch.object(parser, "error", side_effect=ValueError):
        for field, value in (("cs_surface_cubic_channels", -1), ("cs_surface_cubic_channels", 5),
                             ("model", "fno"), ("cs_baseline_order", 2), ("cs_terms_residual", True),
                             ("cs_surface_parity_streams", True), ("cs_surface_amplitude_normalization", True)):
            namespace["args"] = SimpleNamespace(**(valid | {field: value}))
            with np.testing.assert_raises(ValueError):
                exec(code, namespace)
    check_packed_nested_branches(phase_wavenumber_hidden=0, surface_cubic_channels=2)


def check_packed_nested_branches(
    phase_wavenumber_hidden: int, surface_amplitude_reference: float = 0.0,
    surface_parity_streams: bool = False, surface_cubic_channels: int = 0,
    pair_coupling: bool = False,
) -> None:
    validation = next(node for node in trainer_function("main").body if isinstance(node, ast.If)
                      and ast.unparse(node.test).startswith("args.cs_nested_surface_features"))
    valid_config = dict(cs_nested_surface_features=True, cs_nested_blocks=1, cs_normalize_surface_features=True,
                        cs_use_half_deriv=True, cs_use_second_deriv=True, cs_use_learned_eta_filters=False,
                        cs_nested_surface_modulation=False)
    parser = SimpleNamespace(error=Mock(side_effect=ValueError))
    validation_namespace: dict[str, Any] = {"args": SimpleNamespace(**valid_config), "parser": parser}
    validation_code = compile(ast.Module(body=[validation], type_ignores=[]), "nested_features_validation", "exec")
    exec(validation_code, validation_namespace)
    for field, invalid in (("cs_nested_blocks", 0), ("cs_normalize_surface_features", False),
                           ("cs_use_half_deriv", False), ("cs_use_second_deriv", False),
                           ("cs_use_learned_eta_filters", True), ("cs_nested_surface_modulation", True)):
        validation_namespace["args"] = SimpleNamespace(**(valid_config | {field: invalid}))
        with np.testing.assert_raises(ValueError):
            exec(validation_code, validation_namespace)
    nx = 32
    inputs = 0.1 * jax.random.normal(jax.random.PRNGKey(7), (3, nx, 2), dtype=jnp.float32)
    depth = jnp.log(jnp.asarray([[0.1], [2.3], [8.0]], dtype=jnp.float32))
    probe = jax.random.normal(jax.random.PRNGKey(9), (3, nx, 1), dtype=jnp.float32)
    cases = [(nested, polynomial, multiplier, log_inputs, whiten, False, 2)
             for nested, polynomial, multiplier, log_inputs, whiten in (
        (0, False, False, False, False), (1, False, False, False, False),
        (2, False, False, False, False), (1, True, False, False, False),
        (0, False, True, False, False), (1, True, True, False, False),
        (0, False, False, True, False), (1, True, True, True, False),
        (0, False, False, False, True), (1, True, False, False, True),
    )]
    cases += [(0, polynomial, False, False, False, True, context)
              for polynomial, context in product((False, True), (0, 2))]
    precision_cases = [(*case, 0.0, jnp.float32, False) for case in cases]
    precision_cases += [(0, False, False, False, False, False, 2, 22.727203594311888, dtype, False)
                        for dtype in (jnp.float32, jnp.float64)]
    precision_cases += [(1, False, False, False, False, False, 2, 0.0, dtype, True)
                       for dtype in (jnp.float32, jnp.float64)]
    precision_cases = [(*case, False) for case in precision_cases]
    precision_cases += [(1, False, False, False, whiten, False, 2, 0.0, dtype, False, True)
                       for dtype, whiten in product((jnp.float32, jnp.float64), (False, True))]
    precision_cases = [(*case, False) for case in precision_cases]
    precision_cases += [(nested, False, False, False, False, False, 2, 0.0, dtype, False, False, True)
                       for dtype, nested in product((jnp.float32, jnp.float64), (0, 1))]
    if phase_wavenumber_hidden or surface_amplitude_reference or surface_parity_streams or surface_cubic_channels or pair_coupling:
        precision_cases = [(nested, False, False, False, False, False, 2, 0.0, dtype, False, False, False)
                           for dtype, nested in product((jnp.float32, jnp.float64), (0, 1))]
    branches = 4 if pair_coupling else 2
    for (nested_blocks, polynomial_skip, multiplier_skip, log_multiplier_inputs,
         whiten, depth_gate, context, cutoff, dtype, nested_modulation, nested_features, phase_rank_one) in precision_cases:
        case_inputs, case_depth = inputs.astype(dtype), depth.astype(dtype)
        model = CraigSulemDNO(
            width=8, n_blocks=2, nested_blocks=nested_blocks, latent=branches, mult_hidden=4,
            pair_coupling=pair_coupling,
            nested_surface_modulation=nested_modulation,
            nested_surface_features=nested_features,
            curvature_cutoff=cutoff,
            surface_feature_scales=((0.3, 0.1, 0.04, 0.7, 2.0, 0.5, 0.3, 0.2, 0.4, 0.6)
                                    if surface_parity_streams else (0.3, 0.1, 0.04, 0.7, 2.0, 0.5, 0.3)),
            surface_feature_transform=(tuple(map(tuple, np.eye(7) + 0.05 * np.ones((7, 7))))
                                       if whiten else None),
            surface_context_width=context, surface_context_gate=depth_gate and bool(context),
            surface_depth_gate=depth_gate, baseline_order=1, baseline_true_depth=True,
            surface_polynomial_skip=polynomial_skip, multiplier_skip=multiplier_skip,
            log_multiplier_inputs=log_multiplier_inputs, phase_rank_one=phase_rank_one,
            phase_wavenumber_hidden=phase_wavenumber_hidden,
            surface_amplitude_reference=surface_amplitude_reference,
            surface_parity_products=surface_parity_streams,
            surface_parity_streams=surface_parity_streams,
            surface_cubic_channels=surface_cubic_channels,
            domain_length=5 * np.pi if phase_rank_one or phase_wavenumber_hidden else 2 * np.pi,
            eta_scale=0.13, xi_scale=0.08, target_scale=0.12,
        )
        params = cast(dict[str, Any], model.init(jax.random.PRNGKey(3), inputs, depth)["params"])
        if surface_amplitude_reference:
            assert params["eta_amplitude_gain"].dtype == jnp.float32
            params["eta_amplitude_gain"] = jnp.asarray([0.2, -0.3, 0.1, 0.4], dtype=jnp.float32)
        if phase_rank_one:
            params["phase_rank_one_coefficients"] = jnp.array([0.25, -0.07], dtype=jnp.float32)
        if phase_wavenumber_hidden:
            params["phase_wavenumber"]["out"]["kernel"] = jnp.linspace(
                -0.15, 0.2, phase_wavenumber_hidden, dtype=jnp.float32,
            )[:, None]
            params["phase_wavenumber"]["out"]["bias"] = jnp.asarray([0.03], dtype=jnp.float32)
        if nested_modulation:
            params["nested_surface_gain"] = jnp.asarray(0.6, dtype=jnp.float32)
        if depth_gate:
            shape = cast(jax.Array, params["eta_depth_gate"]["kernel"]).shape
            params["eta_depth_gate"]["kernel"] = jnp.linspace(-0.6, 0.7, int(np.prod(shape)), dtype=jnp.float32).reshape(shape)
        for index in range(2):
            group = params[f"cs_block_{index}"]
            if index < 2 - nested_blocks:
                shape = cast(jax.Array, group["phi_proj"]["kernel"]).shape
                group["phi_proj"]["kernel"] = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) * 0.02 - 0.07
                if pair_coupling:
                    shape = cast(jax.Array, group["phi_cross"]["kernel"]).shape
                    group["phi_cross"]["kernel"] = jnp.linspace(
                        -0.13 + index * 0.03, 0.19 + index * 0.05, int(np.prod(shape)), dtype=jnp.float32,
                    ).reshape(shape)
            else:
                group["gamma"] = jnp.linspace(0.2, -0.3, branches, dtype=jnp.float32)
                if nested_features:
                    group["surface_mix"] = jnp.array([[0.1, -0.1], [0.05, 0.1]], dtype=jnp.float32)
            if multiplier_skip:
                for part in ("m_shared", "m_inner") if index >= 2 - nested_blocks else ("m_shared",):
                    shape = cast(jax.Array, group[part]["symbol_skip"]["kernel"]).shape
                    group[part]["symbol_skip"]["kernel"] = jnp.linspace(-0.2, 0.3, int(np.prod(shape)), dtype=jnp.float32).reshape(shape)
        namespace: dict[str, Any] = {
            "jax": jax, "jnp": jnp, "nn": nn, "Any": Any, "cast": cast, "model": model,
            "CraigSulemBlock": CraigSulemBlock, "DepthAwareMultiplier": DepthAwareMultiplier,
            "nx": nx, "domain_length": float(model.domain_length),
            "args": SimpleNamespace(model="cs_dno", cs_terms_residual=False, n_blocks=2, cs_nested_blocks=nested_blocks,
                                    latent=branches, cs_mult_hidden=4, cs_surface_polynomial_skip=polynomial_skip,
                                    cs_multiplier_skip=multiplier_skip, cs_log_multiplier_inputs=log_multiplier_inputs,
                                    cs_nested_surface_features=nested_features),
        }
        exec(compile(ast.Module(body=[trainer_function("apply_model")], type_ignores=[]),
                     "trainer_packed_apply", "exec"), namespace)
        packed_apply = namespace["apply_model"]
        native = cast(tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                      model.apply({"params": params}, case_inputs, case_depth, return_baseline=True))
        evaluate_packed = jax.jit(packed_apply, static_argnames="return_baseline")
        packed = evaluate_packed(
            {"params": params}, case_inputs, case_depth, return_baseline=True,
        )
        for actual, expected in zip(packed, native):
            assert actual.dtype == expected.dtype == case_inputs.dtype
            np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-7)
        np.testing.assert_array_equal(packed[1], packed[2])
        if phase_rank_one or phase_wavenumber_hidden:
            zero_phase = ({"phase_wavenumber": params["phase_wavenumber"] | {
                "out": jax.tree.map(jnp.zeros_like, params["phase_wavenumber"]["out"]),
            }} if phase_wavenumber_hidden else {"phase_rank_one_coefficients": jnp.zeros(2, dtype=jnp.float32)})
            disabled = evaluate_packed(
                {"params": params | zero_phase},
                case_inputs, case_depth, return_baseline=True,
            )
            k = (2 * jnp.pi / model.domain_length) * jnp.arange(nx // 2 + 1, dtype=dtype)
            u_hat = (1j * k * jnp.sqrt(k) * jnp.fft.rfft(case_inputs[..., 0] * model.eta_scale)).at[:, -1].set(0)
            u = jnp.fft.irfft(u_hat, n=nx)
            potential = case_inputs[..., 1] * model.xi_scale
            if phase_wavenumber_hidden:
                multiplier = cast(jax.Array, PhaseWavenumberMultiplier(phase_wavenumber_hidden, model.domain_length).apply(
                    {"params": params["phase_wavenumber"]}, case_depth, nx // 2 + 1,
                ))
                v = jnp.fft.irfft(multiplier * u_hat, n=nx)
                expected_phase = 0.5 * (u * jnp.mean(v * potential, axis=1, keepdims=True)
                                        + v * jnp.mean(u * potential, axis=1, keepdims=True))
            else:
                coefficients = cast(jax.Array, params["phase_rank_one_coefficients"])
                beta = coefficients[0] + coefficients[1] * (case_depth + jnp.log(2 * jnp.pi / model.domain_length))
                expected_phase = beta * u * jnp.mean(u * potential, axis=1, keepdims=True)
            expected_phase = expected_phase[..., None] / model.target_scale
            np.testing.assert_allclose(packed[3] - disabled[3], expected_phase, rtol=3e-5, atol=2e-7)
            np.testing.assert_allclose(packed[0] - disabled[0], expected_phase, rtol=3e-5, atol=2e-7)
            np.testing.assert_array_equal(packed[1], disabled[1])

        def objective(parameters: dict[str, Any], values: jax.Array, log_depth: jax.Array,
                      *, packed: bool) -> jax.Array:
            apply = packed_apply if packed else model.apply
            return jnp.sum(cast(jax.Array, apply({"params": parameters}, values, log_depth)) * probe)

        differentiate = jax.jit(jax.value_and_grad(objective, argnums=(0, 1, 2)), static_argnames="packed")
        expected_value, expected_grad = differentiate(params, case_inputs, case_depth, packed=False)
        actual_value, actual_grad = differentiate(params, case_inputs, case_depth, packed=True)
        np.testing.assert_allclose(actual_value, expected_value, rtol=3e-6, atol=2e-7)
        for actual, expected in zip(jax.tree.leaves(actual_grad), jax.tree.leaves(expected_grad)):
            assert actual.dtype == expected.dtype
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-7)
            assert bool(jnp.all(jnp.isfinite(actual)))
        if surface_amplitude_reference:
            assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(actual_grad[0]))
            assert actual_grad[1].dtype == actual_grad[2].dtype == dtype
            assert float(jnp.linalg.norm(actual_grad[0]["eta_amplitude_gain"])) > 0
        if surface_parity_streams:
            for name in ("eta_feat_proj", "eta_feat_mix", "eta_context_in", "eta_context_expand"):
                for parity in ("even_kernel", "odd_kernel"):
                    assert float(jnp.linalg.norm(actual_grad[0][name][parity])) > 0
        if surface_cubic_channels:
            for columns in (slice(None, -surface_cubic_channels), slice(-surface_cubic_channels, None)):
                assert float(jnp.linalg.norm(actual_grad[0]["eta_feat_mix"]["kernel"][:, columns])) > 0
                assert float(jnp.linalg.norm(actual_grad[0]["cs_block_0"]["phi_proj"]["kernel"][columns])) > 0
        if pair_coupling:
            for index in range(2 - nested_blocks):
                assert float(jnp.linalg.norm(actual_grad[0][f"cs_block_{index}"]["phi_cross"]["kernel"])) > 0
        for index in range(2 - nested_blocks, 2):
            group_grad = actual_grad[0][f"cs_block_{index}"]
            for part in ("gamma", "m_inner", "m_shared"):
                assert sum(float(jnp.linalg.norm(value)) for value in jax.tree.leaves(group_grad[part])) > 0
            if nested_features:
                assert bool(jnp.all(jnp.abs(group_grad["surface_mix"]) > 0))
        if polynomial_skip:
            assert float(jnp.linalg.norm(actual_grad[0]["cs_block_0"]["phi_proj"]["kernel"][-2:])) > 0
        if depth_gate:
            assert float(jnp.linalg.norm(actual_grad[0]["eta_depth_gate"]["kernel"])) > 0
        if nested_modulation:
            assert float(jnp.abs(actual_grad[0]["nested_surface_gain"])) > 0
        if phase_rank_one:
            assert bool(jnp.all(jnp.abs(actual_grad[0]["phase_rank_one_coefficients"]) > 0))
        if phase_wavenumber_hidden:
            for leaf in jax.tree.leaves(actual_grad[0]["phase_wavenumber"]):
                assert float(jnp.linalg.norm(leaf)) > 0
        if multiplier_skip:
            for index in range(2):
                for part in ("m_shared", "m_inner") if index >= 2 - nested_blocks else ("m_shared",):
                    assert float(jnp.linalg.norm(actual_grad[0][f"cs_block_{index}"][part]["symbol_skip"]["kernel"])) > 0


def test_all_momentum_ignores_family_labels_and_supervises_generated_rows() -> None:
    nx, size = 32, 8
    x = jnp.arange(nx, dtype=jnp.float64) * (2 * jnp.pi / nx)
    amplitude = jnp.arange(1, size + 1)[:, None] / size
    eta = amplitude * (0.03 * jnp.cos(x) + 0.01 * jnp.sin(2 * x))
    xi = amplitude * (0.02 * jnp.sin(x) + 0.01 * jnp.cos(3 * x))
    target = amplitude * (0.04 * jnp.sin(x) + 0.01 * jnp.cos(2 * x))
    depth = jnp.log(jnp.linspace(0.2, 8.0, size))[:, None]
    family = jnp.array([1, 2, 3, 4, 1, 2, 3, 4])
    config = SimpleNamespace(
        model="fno", data_k_max=None, residual_floor=1e-6, data_xi_projection_weight=0.0, mode_balanced_weight=0.0,
        translation_tangent_weight=0.0, phase_mechanism="all_momentum",
        phase_mechanism_weight=0.03, phase_mechanism_ramp_epochs=0.0,
        hadamard_weight=0.0, mode_balanced_warmup_steps=1,
        rollout_training_weight=0.7, rollout_training_ramp_epochs=0,
        rollout_training_loss="full", rollout_training_full_spectrum=False, rollout_training_rhs=False, rollout_coherent_weight=0.0,
        amplitude_probe_scale=0.0, subtract_before_fft=False,
        fp64_residual=False, subtract_before_centering=False, data_phase_projection_weight=0.0,
        data_modal_residual_floor=0.0,
    )

    def apply_model(
        variables: dict[str, jax.Array], inputs: jax.Array, depth: jax.Array, *, return_baseline: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        prediction = variables["params"][:, None, None] * jnp.sum(inputs, axis=-1, keepdims=True)
        return prediction, jnp.zeros_like(prediction), jnp.zeros_like(prediction), prediction

    namespace: dict[str, Any] = {
        "jax": jax, "jnp": jnp, "Any": Any, "cast": cast, "FlatParams": Any, "args": config,
        "training_dtype": jnp.float64, "physical_dtype": jnp.float64, "PhysicalFamilyId": PhysicalFamilyId,
        "norm_inputs_jax": lambda e, p: jnp.stack((e, p), axis=-1),
        "norm_targets_jax": lambda q: q[..., None], "denorm_targets_jax": lambda q: q,
        "log_h_max": np.log(5.0), "k_grid_jax": jnp.fft.fftfreq(nx) * nx,
        "k_rfft_jax": jnp.arange(nx // 2 + 1),
        "compute_phase_mechanism_loss": compute_phase_mechanism_loss, "apply_model": apply_model,
        "compute_rhs_relative_error": compute_rhs_relative_error,
        "current_state": SimpleNamespace(step=100), "n_data_only_rows": 0, "amplitude_batch": None, "coherent_batch": None, "terms_phys": None,
    }
    exec(compile(ast.Module(body=[trainer_function("compute_loss_components"),
                                 trainer_function("compute_training_loss")], type_ignores=[]),
                 "trainer_all_momentum", "exec"), namespace)
    components = namespace["compute_loss_components"]
    params = jnp.linspace(0.4, 0.9, size)
    fields = (eta, xi, target, depth, family)
    prediction = params[:, None] * (eta + xi)
    for mechanism in ("none", "tanaka_momentum", "all_momentum"):
        config.phase_mechanism = mechanism
        losses = components(params, *fields)
        if mechanism == "none":
            assert losses[3] == 0
            continue
        mask = family == PhysicalFamilyId.TANAKA if mechanism == "tanaka_momentum" else None
        expected, count = compute_phase_mechanism_loss(
            eta, xi, prediction, target, jnp.exp(depth[:, 0]), namespace["k_grid_jax"],
            mechanism="momentum", horizon=200.0, sample_mask=mask,
        )
        np.testing.assert_allclose(losses[3], expected * count / size, rtol=1e-12)
        gradient = jax.grad(lambda p: components(p, *fields)[3])(params)
        selected = np.asarray(family == PhysicalFamilyId.TANAKA) if mask is not None else np.ones(size, bool)
        assert np.all(np.abs(np.asarray(gradient)[selected]) > 1e-8)
        np.testing.assert_array_equal(np.asarray(gradient)[~selected], 0)
    relabeled = fields[:-1] + (jnp.full_like(family, 99),)
    for actual, expected in zip(components(params, *relabeled)[:6], components(params, *fields)[:6]):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(jax.grad(lambda p: components(p, *relabeled)[3])(params), gradient)
    generated = (1.5 * eta, 1.2 * xi, 0.8 * target, depth, family)
    base, teacher = components(params, *fields), components(params, *generated)
    namespace.update(eta=eta, xi=xi, gxi=target, batch_depth=depth, family=family)
    for cutoff, loss_mode, projection_weight, phase_weight in product(
        (None, 1.5), ("full", "data"), (0.0, 4.0), (0.0, 3.0),
    ):
        config.data_k_max = cutoff
        config.data_xi_projection_weight = projection_weight
        config.data_phase_projection_weight = phase_weight
        if cutoff is not None or projection_weight > 0 or phase_weight > 0:
            for inputs, ordinary in ((fields, base), (generated, teacher)):
                band = components(params, *inputs)
                assert not np.isclose(band[0], ordinary[0])
                np.testing.assert_array_equal(band[1:6], ordinary[1:6])
        config.rollout_training_loss = loss_mode
        namespace["rollout_batch"] = (True, depth, family, *generated[:3], jnp.float32(1.0))
        (actual, metrics), actual_gradient = jax.value_and_grad(
            namespace["compute_training_loss"], has_aux=True,
        )(params)

        expected_value, expected_gradient = jax.value_and_grad(lambda p: sum(
            value[0] * data_weight + value[3] * momentum_weight
            for value, data_weight, momentum_weight in (
                (components(p, *fields), 1.0, config.phase_mechanism_weight),
                (components(p, *generated), config.rollout_training_weight,
                 config.rollout_training_weight * config.phase_mechanism_weight if loss_mode == "full" else 0.0),
            )
        ))(params)
        np.testing.assert_allclose(actual, expected_value, rtol=1e-12)
        np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(metrics["phase_mechanism_loss"], base[3], rtol=1e-12)
        np.testing.assert_allclose(metrics["rollout_phase_mechanism_loss"],
                                   teacher[3] if loss_mode == "full" else 0.0, rtol=1e-12)
    config.rollout_training_full_spectrum = True
    config.rollout_training_loss = "full"
    config.data_xi_projection_weight = config.data_phase_projection_weight = 0.0
    valid = jnp.arange(size) % 2 == 0
    generated = (*generated[:2], generated[2] + valid[:, None] * 0.02 * jnp.sin(3 * x), *generated[3:])
    namespace["rollout_batch"] = (True, depth, family, *generated[:3], jnp.mean(valid), valid)
    (actual, metrics), actual_gradient = jax.value_and_grad(
        namespace["compute_training_loss"], has_aux=True,
    )(params)
    expected_value, expected_gradient = jax.value_and_grad(lambda p: sum(
        weight * (values[0] + config.phase_mechanism_weight * values[3])
        for weight, values in (
            (1.0, components(p, *fields)),
            (config.rollout_training_weight, components(p, *generated, full_spectrum_rows=valid)),
        )
    ))(params)
    np.testing.assert_allclose(actual, expected_value, rtol=1e-12)
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    expected_generated = components(params, *generated, full_spectrum_rows=valid)
    np.testing.assert_allclose(metrics["rollout_phase_mechanism_loss"], expected_generated[3], rtol=1e-12)
    assert not np.isclose(expected_generated[3], teacher[3])
    ordinary = jax.value_and_grad(lambda p: components(p, *fields)[0])(params)
    old_generated = jax.jacrev(lambda p: jnp.asarray(
        components(p, *generated, full_spectrum_rows=valid)[:6],
    ))(params)
    config.rollout_training_rhs = True
    new_generated = components(params, *generated, full_spectrum_rows=valid)
    new_derivatives = jax.jacrev(lambda p: jnp.asarray(
        components(p, *generated, full_spectrum_rows=valid)[:6],
    ))(params)
    np.testing.assert_array_equal(new_generated[1:6], expected_generated[1:6])
    np.testing.assert_array_equal(new_derivatives[1:6], old_generated[1:6])
    np.testing.assert_array_equal(new_derivatives[0, ~valid], old_generated[0, ~valid])
    for value, reference in zip(jax.value_and_grad(lambda p: components(p, *fields)[0])(params), ordinary):
        np.testing.assert_array_equal(value, reference)

    def expected_rhs_data(p: jax.Array) -> jax.Array:
        rhs = compute_rhs_relative_error(
            generated[0], generated[1], p[:, None] * (generated[0] + generated[1]),
            generated[2], jnp.exp(depth[:, 0]), namespace["k_grid_jax"],
        )
        fallback = jnp.stack([components(p[i:i + 1], *(f[i:i + 1] for f in generated))[0]
                              for i in range(size)])
        return jnp.mean(jnp.where(valid, rhs, fallback))

    expected_data, expected_gradient = jax.value_and_grad(expected_rhs_data)(params)
    np.testing.assert_allclose(new_generated[0], expected_data, rtol=1e-12)
    np.testing.assert_allclose(new_derivatives[0], expected_gradient, rtol=1e-12, atol=1e-13)
    (rhs_total, rhs_metrics), rhs_gradient = jax.value_and_grad(
        namespace["compute_training_loss"], has_aux=True,
    )(params)
    np.testing.assert_allclose(rhs_total - actual,
                               config.rollout_training_weight * (expected_rhs_data(params) - expected_generated[0]),
                               rtol=1e-12)
    np.testing.assert_allclose(rhs_gradient - actual_gradient, config.rollout_training_weight
                               * (expected_gradient - old_generated[0]), rtol=1e-12, atol=1e-13)
    for key in metrics:
        if key not in ("rollout_data_loss",):
            np.testing.assert_array_equal(rhs_metrics[key], metrics[key])
    for mask in (None, jnp.zeros(size, bool)):
        for value, reference in zip(jax.value_and_grad(
            lambda p: components(p, *fields, full_spectrum_rows=mask)[0],
        )(params), ordinary):
            np.testing.assert_array_equal(value, reference)


def test_datum_shift_geometry_clean_rows_rng_and_validation() -> None:
    main = trainer_function("main")
    option = next(node for node in ast.walk(main) if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call) and node.value.args
                  and isinstance(node.value.args[0], ast.Constant)
                  and node.value.args[0].value == "--datum_shift_rms")
    validation = next(node for node in main.body if isinstance(node, ast.If)
                      and ast.unparse(node.test).startswith("not np.isfinite(args.datum_shift_rms)"))
    parser = argparse.ArgumentParser(add_help=False)
    namespace: dict[str, Any] = {"parser": parser, "np": np}
    exec(compile(ast.Module(body=[option], type_ignores=[]), "datum_option", "exec"), namespace)
    assert parser.parse_args([]).datum_shift_rms == 0.0
    validate = compile(ast.Module(body=[validation], type_ignores=[]), "datum_validation", "exec")
    with patch.object(parser, "error", side_effect=ValueError):
        for scale in (-0.1, np.nan, np.inf, -np.inf):
            namespace["args"] = SimpleNamespace(datum_shift_rms=scale)
            with np.testing.assert_raises(ValueError):
                exec(validate, namespace)
        for scale in (0.0, 0.025, 100.0):
            namespace["args"] = SimpleNamespace(datum_shift_rms=scale)
            exec(validate, namespace)
    component = trainer_function("train_step")
    datum = next(node for node in component.body if isinstance(node, ast.If)
                 and "args.datum_shift_rms" in ast.unparse(node.test))
    assert component.body[0] is datum
    function = ast.parse(
        "def prepare(step, rng_key, eta, xi, gxi, batch_depth, family):\n"
        "    current_state = SimpleNamespace(step=step)"
    ).body[0]
    assert isinstance(function, ast.FunctionDef)
    function.body += [datum]
    function.body += ast.parse("return eta, xi, gxi, batch_depth, family, rng_key").body
    config = SimpleNamespace(datum_shift_rms=0.025)
    namespace.update(jax=jax, jnp=jnp, SimpleNamespace=SimpleNamespace, args=config)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 "trainer_datum", "exec"), namespace)
    nx, size = 64, 64
    x = jnp.arange(nx, dtype=jnp.float64) * (2 * jnp.pi / nx)
    h = jnp.geomspace(0.01, 100.0, size)[:, None]
    key = jax.random.PRNGKey(41)
    for dtype, scale in product((jnp.float32, jnp.float64), (0.025, 100.0)):
        config.datum_shift_rms = scale
        prepare = jax.vmap(namespace["prepare"], in_axes=(None, None, 0, 0, 0, 0, 0), axis_name="batch")
        eta = h * (0.1 + 0.2 * jnp.cos(x))
        eta = eta.at[:8].set(jnp.broadcast_to(0.1 * h[:8], (8, nx)))
        fields = (eta.astype(dtype), (0.02 * jnp.sin(x) + eta).astype(dtype),
                  (0.03 * jnp.cos(2 * x) - eta).astype(dtype), jnp.log(h).astype(dtype),
                  jnp.tile(jnp.arange(1, 5), size // 4))
        inputs = tuple(jnp.stack((value, value)) for value in fields)
        result = prepare(7, key, *inputs)
        repeated = prepare(7, key, *inputs)
        compiled = jax.jit(prepare)(7, key, *inputs)
        changed = prepare(8, key, *inputs)
        shifted = np.any(np.asarray(result[0]) != np.asarray(inputs[0]), axis=-1)
        assert size // 2 < np.sum(shifted) < 3 * size // 2
        assert not np.array_equal(shifted[0], shifted[1])
        assert not np.array_equal(result[0], changed[0])
        assert np.any(shifted[:, :8])  # Constant surfaces still have nonzero raw RMS.
        np.testing.assert_array_equal(np.asarray(result[0])[~shifted], np.asarray(inputs[0])[~shifted])
        np.testing.assert_array_equal(np.asarray(result[3])[~shifted], np.asarray(inputs[3])[~shifted])
        depth = np.exp(np.asarray(inputs[3], dtype=np.float64))
        new_depth = np.exp(np.asarray(result[3]))
        np.testing.assert_allclose(result[0] + new_depth, inputs[0].astype(jnp.float64) + depth,
                                   rtol=3e-15, atol=3e-15)
        assert np.all(new_depth >= 0.75 * depth * (1 - 1e-14))
        assert np.all(np.asarray(result[0]) + new_depth > 0)
        shift = np.asarray(inputs[0], dtype=np.float64) - np.asarray(result[0])
        limit = np.minimum(scale * np.sqrt(np.mean(np.asarray(inputs[0], dtype=np.float64)**2,
                                                   axis=-1, keepdims=True)), 0.25 * depth)
        assert np.all(np.abs(shift) <= limit * (1 + 1e-12))
        for actual, same, traced in zip(result, repeated, compiled):
            np.testing.assert_array_equal(actual, same)
            np.testing.assert_allclose(actual, traced, rtol=3e-15, atol=3e-15)
        for index in (1, 2, 4):
            np.testing.assert_array_equal(result[index], inputs[index])
            assert result[index].dtype == inputs[index].dtype
        np.testing.assert_array_equal(result[-1], jnp.broadcast_to(key, (2,) + key.shape))
        config.datum_shift_rms = 0.0
        for actual, original in zip(prepare(7, key, *inputs)[:5], inputs):
            np.testing.assert_array_equal(actual, original)
            assert actual.dtype == original.dtype
    assert "datum_shift_rms" not in ast.unparse(trainer_function("eval_step"))
    assert "datum_shift_rms" not in ast.unparse(trainer_function("compute_loss_components"))


def test_reflection_pairs_labels_preserves_rng_and_commutes_with_m6() -> None:
    component = trainer_function("train_step")
    reflection = next(node for node in component.body
                      if isinstance(node, ast.If) and "args.reflection_augmentation" in ast.unparse(node.test))
    function = ast.parse(
        "def prepare(step, rng_key, eta, xi, gxi, batch_depth, family):\n"
        "    current_state = SimpleNamespace(step=step)"
    ).body[0]
    assert isinstance(function, ast.FunctionDef)
    function.body += [reflection]
    function.body += ast.parse("return eta, xi, gxi, batch_depth, family, rng_key").body
    nx, size = 64, 32
    x = jnp.arange(nx, dtype=jnp.float64) * (2 * jnp.pi / nx)
    amplitude = jnp.linspace(0.5, 1.0, size)[:, None]
    eta = amplitude * (0.008 * jnp.cos(x) + 0.003 * jnp.sin(3 * x))
    xi = amplitude * (0.02 * jnp.sin(x) + 0.006 * jnp.cos(2 * x))
    depth = jnp.log(jnp.linspace(0.1, 8.0, size))[:, None]
    family = jnp.tile(jnp.arange(1, 5), size // 4)
    target = compute_dno_target(
        eta, xi, jnp.exp(depth), nx=nx, length=2 * np.pi,
        dno_order=6, pad_factor=8, maximum_wavenumber=128.0,
    )[2]
    fields = eta, xi, target, depth, family
    inputs = tuple(jnp.stack((value, value)) for value in fields)
    config = SimpleNamespace(reflection_augmentation=True)
    namespace: dict[str, Any] = {
        "jax": jax, "jnp": jnp, "SimpleNamespace": SimpleNamespace,
        "args": config, "nx": nx, "cast": cast,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 "trainer_reflection", "exec"), namespace)
    prepare = jax.vmap(namespace["prepare"], in_axes=(None, None, 0, 0, 0, 0, 0), axis_name="batch")
    key = jax.random.PRNGKey(41)
    result = prepare(7, key, *inputs)
    repeated = jax.jit(prepare)(7, key, *inputs)
    changed = prepare(8, key, *inputs)
    indices = (-jnp.arange(nx)) % nx
    flipped = np.asarray(result[0][..., 1] != inputs[0][..., 1])
    assert np.any(flipped) and not np.all(flipped)
    assert not np.array_equal(flipped[0], flipped[1])
    assert not np.array_equal(result[0], changed[0])
    for actual, original in zip(result[:3], inputs[:3]):
        expected = jnp.where(flipped[..., None], original[..., indices], original)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[..., 0], original[..., 0])
        # Periodic reflection conjugates each Fourier coefficient; no grid shift.
        expected_hat = jnp.fft.rfft(original, axis=-1)
        expected_hat = jnp.where(flipped[..., None], expected_hat.conj(), expected_hat)
        np.testing.assert_allclose(jnp.fft.rfft(actual, axis=-1), expected_hat, atol=2e-15)
    for actual, original in zip(result[3:5], inputs[3:]):
        np.testing.assert_array_equal(actual, original)
    for actual, same in zip(result, repeated):
        np.testing.assert_array_equal(actual, same)
    np.testing.assert_array_equal(result[-1], jnp.broadcast_to(key, (2,) + key.shape))
    _, _, regenerated = compute_dno_target(
        result[0], result[1], jnp.exp(result[3]), nx=nx, length=2 * np.pi,
        dno_order=6, pad_factor=8, maximum_wavenumber=128.0,
    )
    np.testing.assert_allclose(regenerated, result[2], rtol=1e-10, atol=2e-14)
    k = jnp.arange(nx // 2 + 1)
    derivative = jnp.fft.irfft(1j * k * jnp.fft.rfft(inputs[0], axis=-1), n=nx, axis=-1)
    reflected_derivative = jnp.fft.irfft(1j * k * jnp.fft.rfft(result[0], axis=-1), n=nx, axis=-1)
    np.testing.assert_allclose(
        reflected_derivative, jnp.where(flipped[..., None], -derivative[..., indices], derivative), atol=1e-14,
    )
    config.reflection_augmentation = False
    disabled = prepare(7, key, *inputs)
    for actual, original in zip(disabled[:5], inputs):
        np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(disabled[-1], result[-1])
    assert "reflection_augmentation" not in ast.unparse(trainer_function("eval_step"))


def test_input_noise_scale_spectrum_rng_and_depth_safety() -> None:
    function = ast.parse(
        "def prepare(step, rng_key, eta, xi, gxi, batch_depth):\n"
        "    current_state = SimpleNamespace(step=step)"
    ).body[0]
    assert isinstance(function, ast.FunctionDef)
    function.body += [next(node for node in trainer_function("train_step").body
                          if isinstance(node, ast.If) and "args.input_noise_rms" in ast.unparse(node.test))]
    function.body += ast.parse("return eta, xi, gxi, batch_depth").body
    nx = 512
    config = SimpleNamespace(input_noise_rms=0.01)
    namespace: dict[str, Any] = {
        "jax": jax, "jnp": jnp, "SimpleNamespace": SimpleNamespace,
        "args": config, "nx": nx, "k_rfft_jax": jnp.arange(nx // 2 + 1), "cast": cast,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "trainer_noise", "exec"), namespace)
    prepare = jax.vmap(namespace["prepare"], in_axes=(None, None, 0, 0, 0, 0), axis_name="batch")
    x = jnp.arange(nx, dtype=jnp.float32) * (2 * jnp.pi / nx)
    eta = jnp.stack((0.03 * jnp.cos(x), 0.004 * jnp.sin(7 * x), jnp.zeros_like(x)))
    xi = jnp.stack((0.05 * jnp.sin(2 * x), 0.02 * jnp.cos(3 * x), jnp.zeros_like(x)))
    inputs = tuple(jnp.stack((field, field)) for field in (eta, xi, xi * 2, jnp.zeros((3, 1), jnp.float32)))
    key = jax.random.PRNGKey(23)
    result = prepare(1, key, *inputs)
    compiled = jax.jit(prepare)(1, key, *inputs)
    for actual, traced in zip(result, compiled):
        np.testing.assert_allclose(actual, traced, rtol=1e-6, atol=1e-8)
    for actual, original in zip(result[2:], inputs[2:]):
        np.testing.assert_array_equal(actual, original)
    changes = []
    for actual, original in zip(result[:2], inputs[:2]):
        delta = np.asarray(actual - original)
        changes.append(delta)
        rms = np.sqrt(np.mean(np.asarray(original[:, :2]) ** 2, axis=-1))
        np.testing.assert_allclose(np.sqrt(np.mean(delta[:, :2] ** 2, axis=-1)), .01 * rms, rtol=2e-5)
        np.testing.assert_allclose(delta.mean(axis=-1), 0, atol=1e-9)
        np.testing.assert_allclose(np.fft.rfft(delta, axis=-1)[..., 129:], 0, atol=1e-7)
        np.testing.assert_array_equal(actual[:, 2], original[:, 2])
        assert not np.allclose(delta[0], delta[1], atol=1e-7)
    assert not np.allclose(changes[0], changes[1], atol=1e-7)
    repeated = prepare(1, key, *inputs)
    changed = prepare(2, key, *inputs)
    for original, same, different in zip(result[:2], repeated[:2], changed[:2]):
        np.testing.assert_array_equal(original, same)
        assert not np.allclose(original, different, atol=1e-7)
    config.input_noise_rms = 0.0
    for actual, original in zip(prepare(1, key, *inputs), inputs):
        np.testing.assert_array_equal(actual, original)
    config.input_noise_rms = 1e4
    # Huge mean-zero perturbations must be rejected before they dry a valid row.
    for actual, original in zip(prepare(1, key, *inputs), inputs):
        np.testing.assert_array_equal(actual, original)
    assert "input_noise" not in ast.unparse(trainer_function("eval_step"))
    assert "input_noise" not in ast.unparse(trainer_function("compute_loss_components"))


def test_dirichlet_probe_batch_pairing_spectrum_and_true_depth_labels() -> None:
    component = trainer_function("train_step")
    prelude_end = next(i for i, node in enumerate(component.body) if isinstance(node, ast.FunctionDef))
    function = ast.parse(
        "def prepare(step, rng_key, eta, xi, gxi, batch_depth, family):\n"
        "    current_state = SimpleNamespace(step=step)"
    ).body[0]
    assert isinstance(function, ast.FunctionDef)
    function.body += component.body[:prelude_end]
    function.body += ast.parse("return eta, xi, gxi, batch_depth, family, n_data_only_rows").body
    nx, size = 512, 8
    x = jnp.arange(nx, dtype=jnp.float64) * (2 * jnp.pi / nx)
    amplitude = jnp.arange(1, size + 1)[:, None]
    eta = (0.02 + amplitude * 0.0002 * jnp.cos(2 * x)).astype(jnp.float32)
    xi = (amplitude * (0.003 * jnp.sin(x) + 0.001 * jnp.cos(7 * x))).astype(jnp.float32)
    target = (0.2 * xi).astype(jnp.float32)
    depth = jnp.log(jnp.array([8.0, 0.3, 0.6, 1.0, 2.0, 12.0, 0.1, 0.4]))[:, None].astype(jnp.float32)
    family = jnp.array([1, 2, 3, 4, 1, 2, 3, 4])
    fields = (eta, xi, target, depth, family)
    before = tuple(np.array(value) for value in fields)
    config = SimpleNamespace(datum_shift_rms=0.0, reflection_augmentation=False, input_noise_rms=0.0,
                             dirichlet_probe_mode="off", dirichlet_probe_fraction=0.25,
                             rollout_training_weight=0.0)
    namespace: dict[str, Any] = {
        "jax": jax, "jnp": jnp, "SimpleNamespace": SimpleNamespace, "args": config,
        "nx": nx, "domain_length": 2 * np.pi, "k_rfft_jax": jnp.arange(nx // 2 + 1),
        "compute_dno_target": compute_dno_target,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "trainer_probe", "exec"), namespace)
    prepare = jax.vmap(namespace["prepare"], in_axes=(None, None, 0, 0, 0, 0, 0), axis_name="batch")
    inputs = tuple(value[None] for value in fields)
    key = jax.random.PRNGKey(17)
    for mode in ("off", "control", "scrambled"):
        config.dirichlet_probe_mode = mode
        result = tuple(value[0] for value in prepare(3, key, *inputs))
        repeated = tuple(value[0] for value in prepare(3, key, *inputs))
        for actual, same in zip(result, repeated):
            np.testing.assert_array_equal(actual, same)
        if mode == "off":
            assert result[-1] == 0
            for actual, original in zip(result, fields):
                np.testing.assert_array_equal(actual, original)
            continue
        assert result[-1] == 2
        for actual, original in zip(result, fields):
            np.testing.assert_array_equal(actual[:6], original[:6])
            assert actual.dtype == original.dtype
        for field in (0, 3, 4):
            np.testing.assert_array_equal(result[field][6:], fields[field][:2])
        original_hat = np.fft.rfft(np.asarray(xi[:2], dtype=np.float64), axis=-1)
        original_hat[:, 0] = 0
        original_hat[:, 129:] = 0
        actual_hat = np.fft.rfft(np.asarray(result[1][6:], dtype=np.float64), axis=-1)
        np.testing.assert_allclose(np.abs(actual_hat), np.abs(original_hat), rtol=1e-5, atol=2e-8)
        np.testing.assert_allclose(np.linalg.norm(result[1][6:], axis=-1), np.linalg.norm(xi[:2], axis=-1), rtol=1e-7)
        np.testing.assert_allclose(actual_hat[:, 0], 0, atol=2e-8)
        np.testing.assert_allclose(actual_hat[:, 129:], 0, atol=2e-8)
        _, _, expected = compute_dno_target(
            eta[:2], result[1][6:], jnp.exp(depth[:2].astype(jnp.float64)),
            nx=nx, length=2 * np.pi, dno_order=6, pad_factor=8, maximum_wavenumber=128.0,
        )
        np.testing.assert_allclose(result[2][6:], expected, rtol=2e-5, atol=2e-8)
        _, _, clipped = compute_dno_target(
            eta[:2], result[1][6:], jnp.minimum(jnp.exp(depth[:2].astype(jnp.float64)), 5),
            nx=nx, length=2 * np.pi, dno_order=6, pad_factor=8, maximum_wavenumber=128.0,
        )
        assert np.linalg.norm(np.asarray(expected[0] - clipped[0])) > 1e-6
        changed = prepare(4, key, *inputs)[1][0]
        if mode == "scrambled":
            assert not np.allclose(result[1][6:], xi[:2])
            assert not np.allclose(changed[6:], result[1][6:])
        else:
            np.testing.assert_allclose(result[1][6:], xi[:2], atol=1e-9)
            np.testing.assert_array_equal(changed, result[1])
    for original, unchanged in zip(fields, before):
        np.testing.assert_array_equal(original, unchanged)
    hadamard_call = next(node for node in ast.walk(component) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name) and node.func.id == "hadamard_loss_on_subset")
    assert [ast.unparse(argument) for argument in hadamard_call.args[-3:]] == [
        "eta[:n_auxiliary_rows]", "xi[:n_auxiliary_rows]", "batch_depth[:n_auxiliary_rows]",
    ]


def test_hadamard_flow_probes_route_only_training_physical_targets() -> None:
    main = trainer_function("main")
    options: list[ast.stmt] = [node for node in ast.walk(main) if isinstance(node, ast.Expr)
               and isinstance(node.value, ast.Call) and node.value.args
               and isinstance(node.value.args[0], ast.Constant)
               and node.value.args[0].value in ("--hadamard_flow_probes", "--hadamard_mode")]
    guard = next(node for node in ast.walk(main) if isinstance(node, ast.If)
                 and "args.hadamard_flow_probes" in ast.unparse(node.test))
    parser = argparse.ArgumentParser(add_help=False)
    namespace: dict[str, Any] = {"parser": parser}
    exec(compile(ast.Module(body=options, type_ignores=[]), "hadamard_options", "exec"), namespace)
    assert parser.parse_args([]).hadamard_flow_probes is False
    assert parser.parse_args(["--hadamard_flow_probes"]).hadamard_flow_probes is True
    validate = compile(ast.Module(body=[guard], type_ignores=[]), "hadamard_guard", "exec")
    for mode, enabled in product(("self", "secant", "teacher", "targeted"), (False, True)):
        namespace["args"] = SimpleNamespace(hadamard_mode=mode, hadamard_flow_probes=enabled)
        with patch.object(parser, "error", side_effect=ValueError):
            if enabled and mode != "self":
                with np.testing.assert_raises(ValueError):
                    exec(validate, namespace)
            else:
                exec(validate, namespace)

    config = SimpleNamespace(
        hadamard_mode="self", hadamard_flow_probes=False, hadamard_k_max=12.0,
        hadamard_sobolev_order=1, hadamard_fd_step_min=1e-3, hadamard_fd_step_max=3e-3,
        hadamard_min_surface_rms=1e-3, hadamard_denominator_eps=1e-12,
    )
    eta = jnp.arange(8 * 32, dtype=jnp.float64).reshape(8, 32) / 17
    xi, target, depth = 2 * eta, 1000 + 3 * eta, jnp.arange(8, dtype=jnp.float64)[:, None] / 7
    key = jax.random.PRNGKey(19)
    hadamard, local = Mock(return_value=jnp.float64(0.125)), Mock(return_value=jnp.float64(0.25))
    namespace.update(
        jax=jax, jnp=jnp, partial=partial, FlatParams=Any, args=config,
        training_dtype=jnp.float32, physical_dtype=jnp.float32, hadamard_microbatch_local=3, domain_length=2 * np.pi,
        compute_hadamard_loss=hadamard, compute_local_response_loss=local,
        apply_model=object(), norm_inputs_jax=object(), denorm_targets_jax=object(),
        k_grid_jax=jnp.fft.fftfreq(32) * 32, current_params=object(),
        eta=eta, xi=xi, gxi=target, batch_depth=depth, n_auxiliary_rows=5,
        rng=key, rng_key=key, batch_index=2,
    )
    exec(compile(ast.Module(body=[trainer_function("hadamard_loss_on_subset")], type_ignores=[]),
                 "hadamard_subset", "exec"), namespace)
    for route in ("train_step", "eval_step"):
        call = next(node for node in ast.walk(trainer_function(route)) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == "hadamard_loss_on_subset")
        invoke = compile(ast.Expression(body=call), route, "eval")
        for enabled in (False, True):
            config.hadamard_flow_probes = enabled
            with patch.object(jax.lax, "axis_index", return_value=0):
                assert eval(invoke, namespace) == 0.125
            recorded = hadamard.call_args.kwargs
            rng = key if route == "train_step" else jax.random.fold_in(jax.random.fold_in(key, 2), 0)
            perm_key, probe_key = jax.random.split(rng)
            indices = jax.random.permutation(perm_key, 5 if route == "train_step" else 8)[:3]
            np.testing.assert_array_equal(recorded["rng"], probe_key)
            for name, original in (("eta_phys", eta), ("xi_phys", xi), ("batch_depth_local", depth)):
                np.testing.assert_array_equal(recorded[name], original.astype(jnp.float32)[indices])
            assert recorded["batch_depth_local"].dtype == recorded["dtype"] == jnp.float64
            assert recorded["model_params"] is namespace["current_params"]
            if enabled and route == "train_step":
                np.testing.assert_array_equal(recorded["flow_target"], target[indices])
            else:
                assert recorded["flow_target"] is None
    for mode in ("secant", "teacher", "targeted"):
        config.hadamard_mode = mode
        assert namespace["hadamard_loss_on_subset"](key, {}, eta, xi, depth) == 0.25
        assert local.call_args.kwargs["mode"] == mode
        assert "flow_target" not in local.call_args.kwargs
    namespace["physical_dtype"] = jnp.float64
    config.hadamard_mode = "self"
    namespace["hadamard_loss_on_subset"](key, {}, eta, xi, depth)
    indices = jax.random.permutation(jax.random.split(key)[0], eta.shape[0])[:3]
    for name, original in (("eta_phys", eta), ("xi_phys", xi), ("batch_depth_local", depth)):
        np.testing.assert_array_equal(hadamard.call_args.kwargs[name], original[indices])


def test_dirichlet_data_uses_all_rows_but_auxiliary_gradients_use_only_prefix() -> None:
    component = trainer_function("compute_loss_components")
    config = SimpleNamespace(
        model="fno", data_k_max=None, residual_floor=1e-6, data_xi_projection_weight=0.0, mode_balanced_weight=6.0,
        mode_balanced_k_max=128.0, mode_balanced_activity_threshold=1e-4,
        mode_balanced_denominator_eps=1e-6, translation_tangent_weight=10.0,
        translation_tangent_scope="tanaka", translation_tangent_geometry="local",
        translation_tangent_denominator_eps=1e-3, translation_tangent_loss_norm="squared",
        translation_tangent_charbonnier_eps=1e-7, translation_tangent_smoothing_scale=1.0,
        stokes_relative_tangent=False, phase_mechanism="none", jonswap_mode_balanced_scale=1.0,
        subtract_before_fft=False, fp64_residual=False, subtract_before_centering=False,
        data_phase_projection_weight=0.0, data_modal_residual_floor=0.0,
    )
    nx = 64
    x = jnp.arange(nx, dtype=jnp.float64) * (2 * jnp.pi / nx)
    amplitude = jnp.arange(1, 9)[:, None] / 8
    eta = 0.02 + amplitude * (jnp.cos(x) + 0.2 * jnp.cos(3 * x))
    xi = amplitude * jnp.sin(x)
    target = amplitude * (0.2 * jnp.cos(x) + 0.1 * jnp.sin(3 * x))
    prediction = target + amplitude**2 * (0.03 * jnp.sin(x) + 0.04 * jnp.cos(3 * x))
    depth = jnp.log(jnp.arange(1, 9, dtype=jnp.float64))[:, None]
    family = jnp.array([2, 1, 2, 3, 2, 4, 2, 2])

    def apply_model(
        variables: dict[str, jax.Array], inputs: jax.Array, depth: jax.Array,
        *, return_baseline: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        result = variables["params"][..., None] / 4
        return result, jnp.zeros_like(result), jnp.zeros_like(result), result

    namespace: dict[str, Any] = {
        "jax": jax, "jnp": jnp, "FlatParams": Any, "args": config, "cast": cast,
        "training_dtype": jnp.float64, "physical_dtype": jnp.float64, "PhysicalFamilyId": PhysicalFamilyId,
        "norm_inputs_jax": lambda eta, xi: jnp.stack((eta / 2, xi / 3), axis=-1),
        "norm_targets_jax": lambda target: target[..., None] / 4,
        "denorm_targets_jax": lambda target: target * 4,
        "apply_model": apply_model, "log_h_max": np.log(5.0),
        "k_grid_jax": jnp.fft.fftfreq(nx) * nx, "k_rfft_jax": jnp.arange(nx // 2 + 1),
        "compute_mode_balanced_loss": compute_mode_balanced_loss,
        "compute_translation_tangent_loss": compute_translation_tangent_loss,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[component], type_ignores=[])), "trainer_components", "exec"), namespace)
    losses = namespace["compute_loss_components"]
    fields = (eta, xi, target, depth, family)

    def evaluate(params: jax.Array, size: int, n_data_only_rows: int = 0) -> tuple[jax.Array, ...]:
        return jax.vmap(lambda p, *values: losses(p, *values, n_data_only_rows=n_data_only_rows),
                        axis_name="batch")(params[None], *(value[None, :size] for value in fields))

    augmented = evaluate(prediction, 8, 2)
    prefix = evaluate(prediction[:6], 6)
    ordinary = evaluate(prediction, 8)
    np.testing.assert_allclose(augmented[0][0], relative_l2_loss(prediction[..., None] / 4, target[..., None] / 4), rtol=1e-12)
    np.testing.assert_array_equal(augmented[0], ordinary[0])
    np.testing.assert_array_equal(augmented[5], ordinary[5])
    for index in (1, 2):
        np.testing.assert_allclose(augmented[index], prefix[index], rtol=1e-12)
        assert not np.allclose(ordinary[index], prefix[index], rtol=1e-5, atol=1e-12)
        gradient = jax.grad(lambda p: jnp.sum(evaluate(p, 8, 2)[index]))(prediction)
        np.testing.assert_array_equal(gradient[6:], 0)
        assert np.linalg.norm(gradient[:6]) > 0
        validation_gradient = jax.grad(lambda p: jnp.sum(evaluate(p, 8)[index]))(prediction)
        assert np.linalg.norm(validation_gradient[6:]) > 0
    np.testing.assert_allclose(augmented[4] * 8, prefix[4] * 6)
    data_gradient = jax.grad(lambda p: jnp.sum(evaluate(p, 8, 2)[0]))(prediction)
    assert np.linalg.norm(data_gradient[6:]) > 0
    data_only = evaluate(prediction, 8, 8)
    for index in (0, 5):
        np.testing.assert_array_equal(data_only[index], ordinary[index])
    for index in (1, 2, 3, 4):
        np.testing.assert_array_equal(data_only[index], 0)
    np.testing.assert_array_equal(
        jax.grad(lambda p: jnp.sum(evaluate(p, 8, 8)[0]))(prediction),
        jax.grad(lambda p: jnp.sum(evaluate(p, 8)[0]))(prediction),
    )


def test_fractional_rollout_ramp_parsing_validation_and_weights() -> None:
    main = trainer_function("main")
    option = next(node for node in ast.walk(main) if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call) and node.value.args
                  and isinstance(node.value.args[0], ast.Constant)
                  and node.value.args[0].value == "--rollout_training_ramp_epochs")
    validation = next(node for node in ast.walk(main) if isinstance(node, ast.If)
                      and "args.rollout_training_ramp_epochs" in ast.unparse(node.test))
    weight = next(node for node in ast.walk(trainer_function("train_step"))
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id == "rollout_weight")
    parser = argparse.ArgumentParser(add_help=False)
    namespace: dict[str, Any] = {"parser": parser, "np": np, "jnp": jnp,
                                 "train_steps_per_epoch": 128, "current_state": SimpleNamespace(step=0)}
    exec(compile(ast.Module(body=[option], type_ignores=[]), "ramp_option", "exec"), namespace)
    validate = compile(ast.Module(body=[validation], type_ignores=[]), "ramp_validation", "exec")
    evaluate = compile(ast.Module(body=[weight], type_ignores=[]), "ramp_weight", "exec")
    for ramp, fractions in ((0.0, (1, 1, 1, 1, 1)), (0.25, (0, 0.5, 1, 1, 1)),
                            (1.0, (0, 0.125, 0.25, 1, 1))):
        config = parser.parse_args(["--rollout_training_ramp_epochs", str(ramp)])
        namespace["args"] = config
        exec(validate, namespace)
        assert config.rollout_training_ramp_epochs == ramp
        config.rollout_training_weight, config.rollout_training_start_epoch = 0.75, 1
        for step, fraction in zip((0, 16, 32, 128, 256), fractions, strict=True):
            namespace["current_state"].step = step
            exec(evaluate, namespace)
            np.testing.assert_allclose(namespace["rollout_weight"], 0.75 * fraction)
        if ramp > 0:
            namespace["current_state"].step = -16
            exec(evaluate, namespace)
            assert namespace["rollout_weight"] == 0.0
    with patch.object(parser, "error", side_effect=ValueError) as error:
        for invalid in (-0.25, np.nan, np.inf, -np.inf):
            namespace["args"] = SimpleNamespace(rollout_training_ramp_epochs=invalid)
            with np.testing.assert_raises(ValueError):
                exec(validate, namespace)
            assert "finite and nonnegative" in error.call_args.args[0]


def test_momentum_ramp_weights_original_and_generated_losses_and_gradients() -> None:
    main = trainer_function("main")
    option = next(node for node in ast.walk(main) if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call) and node.value.args
                  and isinstance(node.value.args[0], ast.Constant)
                  and node.value.args[0].value == "--phase_mechanism_ramp_epochs")
    validation = next(node for node in ast.walk(main) if isinstance(node, ast.If)
                      and "args.phase_mechanism_ramp_epochs" in ast.unparse(node.test))
    parser = argparse.ArgumentParser(add_help=False)
    namespace: dict[str, Any] = {"parser": parser, "np": np}
    exec(compile(ast.Module(body=[option], type_ignores=[]), "momentum_ramp_option", "exec"), namespace)
    assert parser.parse_args([]).phase_mechanism_ramp_epochs == 0.0
    validate = compile(ast.Module(body=[validation], type_ignores=[]), "momentum_ramp_validation", "exec")
    with patch.object(parser, "error", side_effect=ValueError):
        for invalid in (-0.5, np.nan, np.inf, -np.inf):
            namespace["args"] = SimpleNamespace(phase_mechanism_ramp_epochs=invalid)
            with np.testing.assert_raises(ValueError):
                exec(validate, namespace)
    config = parser.parse_args(["--phase_mechanism_ramp_epochs", "0.5"])
    config.mode_balanced_weight, config.mode_balanced_warmup_steps = 2.0, 1
    config.translation_tangent_weight, config.phase_mechanism_weight = 3.0, 4.0
    config.hadamard_weight = 0.0
    config.rollout_training_weight, config.rollout_training_ramp_epochs = 0.7, 0.0
    config.rollout_training_loss, config.amplitude_probe_scale = "full", 0.0
    config.rollout_training_full_spectrum = False
    config.rollout_coherent_weight = 0.0
    namespace["args"] = config
    exec(validate, namespace)

    def components(params: jax.Array, eta: jax.Array, *unused: Any, **options: Any) -> tuple[Any, ...]:
        losses = jnp.mean(eta) * jnp.arange(1, 5) * params ** jnp.arange(1, 5)
        return *losses, jnp.float64(1.0), losses[0], ()

    eta = jnp.ones((2, 4), dtype=jnp.float64)
    family = jnp.array([1, 4])
    namespace.update(
        jax=jax, jnp=jnp, FlatParams=Any, compute_loss_components=components,
        current_state=SimpleNamespace(step=0), train_steps_per_epoch=100,
        training_dtype=jnp.float64, physical_dtype=jnp.float64, eta=eta, xi=eta, gxi=eta,
        batch_depth=eta[:, :1], family=family, n_data_only_rows=0, amplitude_batch=None, coherent_batch=None, terms_phys=None,
    )
    exec(compile(ast.Module(body=[trainer_function("compute_training_loss")], type_ignores=[]),
                 "momentum_ramp_loss", "exec"), namespace)
    for ramp, step, generated in product((0.0, 0.5), (0, 25, 50, 100), (None, False, True)):
        config.phase_mechanism_ramp_epochs = ramp
        namespace["current_state"].step = step
        namespace["rollout_batch"] = (generated, eta[:, :1], family, 2 * eta, eta, eta,
                                       jnp.float32(1.0)) if generated is not None else None
        weights = jnp.array([1.0, 2.0 * min(step, 1), 3.0,
                             4.0 * min(step / (100 * ramp), 1.0) if ramp else 4.0])
        (actual, metrics), gradient = jax.value_and_grad(namespace["compute_training_loss"], has_aux=True)(jnp.float64(0.6))
        expected, expected_gradient = jax.value_and_grad(lambda p:
            jnp.dot(weights, jnp.asarray(components(p, eta)[:4])) * (1 + 1.4 * bool(generated))
        )(jnp.float64(0.6))
        np.testing.assert_allclose(actual, expected, rtol=1e-12)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-12)
        np.testing.assert_allclose(metrics["phase_mechanism_loss"], components(jnp.float64(0.6), eta)[3])


def test_rollout_weight_ramp_scales_only_scheduled_student_loss() -> None:
    scheduled = next(node for node in ast.walk(trainer_function("train_step"))
                     if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id == "rollout_scheduled")
    function = ast.parse(
        "def evaluate(current_params, step, enabled, eta, xi, gxi, batch_depth, family):\n"
        "    current_state = SimpleNamespace(step=step)"
    ).body[0]
    assert isinstance(function, ast.FunctionDef)
    function.body += [scheduled]
    function.body += ast.parse(
        "rollout_batch = (rollout_scheduled, batch_depth, family, 3 * eta, 2 * xi, 1.5 * gxi, "
        "jnp.float32(rollout_scheduled), "
        "*((family == 2,) if args.rollout_training_full_spectrum else ())) if enabled else None"
    ).body
    function.body += [trainer_function("compute_training_loss")]
    function.body += ast.parse("return compute_training_loss(current_params)").body
    config = SimpleNamespace(
        rollout_training_weight=0.75, rollout_training_start_epoch=60,
        rollout_training_ramp_epochs=5, rollout_training_interval=16,
        rollout_training_loss="full", rollout_training_full_spectrum=False, rollout_training_rhs=False, rollout_coherent_weight=0.0,
        amplitude_probe_scale=0.0,
        model="fno", data_k_max=None, residual_floor=1e-6, data_xi_projection_weight=0.0, mode_balanced_weight=6.0,
        mode_balanced_warmup_steps=4000, mode_balanced_k_max=128.0,
        mode_balanced_activity_threshold=1e-4, mode_balanced_denominator_eps=1e-6,
        translation_tangent_weight=10.0, translation_tangent_scope="tanaka",
        translation_tangent_geometry="local", translation_tangent_denominator_eps=1e-3,
        translation_tangent_loss_norm="squared", translation_tangent_charbonnier_eps=1e-7,
        translation_tangent_smoothing_scale=1.0, stokes_relative_tangent=False,
        hadamard_weight=0.0, phase_mechanism="none", phase_mechanism_weight=0.0,
        phase_mechanism_ramp_epochs=0.0,
        jonswap_mode_balanced_scale=1.0, subtract_before_fft=False,
        fp64_residual=False, subtract_before_centering=False, data_phase_projection_weight=0.0,
        data_modal_residual_floor=0.0,
    )
    nx = 32
    x = jnp.arange(nx, dtype=jnp.float64) * (2 * jnp.pi / nx)
    amplitude = jnp.arange(1, 9)[:, None] / 8
    eta = amplitude * (jnp.cos(x) + 0.2 * jnp.cos(3 * x))
    xi = amplitude * (jnp.sin(x) + 0.1 * jnp.cos(3 * x))
    target = amplitude * (0.2 * jnp.cos(x) + 0.1 * jnp.sin(3 * x))
    depth = jnp.linspace(0.3, 3.0, 8)
    family = jnp.array([1, 3, 4, 1, 2, 2, 2, 4])  # First shard has no Tanaka rows.
    k = jnp.fft.fftfreq(nx) * nx

    def apply_model(
        variables: dict[str, jax.Array], inputs: jax.Array, depth: jax.Array,
        *, return_baseline: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        prediction = variables["params"] * jnp.sum(inputs, axis=-1, keepdims=True) / 4
        return prediction, jnp.zeros_like(prediction), jnp.zeros_like(prediction), prediction

    def expected_components(params: jax.Array, scale: float) -> jax.Array:
        e, z, truth = scale * eta, (scale + 1) / 2 * xi, (scale + 3) / 4 * target
        prediction = params * (e / 2 + z / 3)
        return jnp.stack((
            relative_l2_loss(prediction[..., None] / 4, truth[..., None] / 4),
            compute_mode_balanced_loss(e, prediction, truth, depth, jnp.arange(nx // 2 + 1)),
            compute_translation_tangent_loss(e, prediction, truth, depth, k, sample_mask=family == 2)[0],
        ))

    namespace: dict[str, Any] = {
        "jax": jax, "jnp": jnp, "FlatParams": Any, "SimpleNamespace": SimpleNamespace, "cast": cast,
        "args": config, "train_steps_per_epoch": 32, "training_dtype": jnp.float64, "physical_dtype": jnp.float64,
        "n_data_only_rows": 0, "PhysicalFamilyId": PhysicalFamilyId, "amplitude_batch": None, "coherent_batch": None, "terms_phys": None,
        "norm_inputs_jax": lambda eta, xi: jnp.stack((eta / 2, xi / 3), axis=-1),
        "norm_targets_jax": lambda target: target[..., None] / 4,
        "denorm_targets_jax": lambda target: target * 4,
        "apply_model": apply_model, "log_h_max": np.log(5.0),
        "k_grid_jax": k, "k_rfft_jax": jnp.arange(nx // 2 + 1),
        "compute_mode_balanced_loss": compute_mode_balanced_loss,
        "compute_translation_tangent_loss": compute_translation_tangent_loss,
        "compute_rhs_relative_error": compute_rhs_relative_error,
    }
    module = ast.Module(body=[trainer_function("compute_loss_components"), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "trainer_ramp", "exec"), namespace)
    params = jnp.float64(0.7)
    assert not np.allclose(expected_components(params, 1.0), expected_components(params, 3.0))
    assert bool(jnp.all(jnp.abs(jax.jacfwd(expected_components)(params, 3.0)) > 1e-5))
    for shards, loss_mode, ramp in product((1, 2), ("full", "data"), (5, 0)):
        config.rollout_training_loss, config.rollout_training_ramp_epochs = loss_mode, ramp
        config.rollout_training_start_epoch = 60
        auxiliary_mask = jnp.array([1.0, float(loss_mode == "full"), float(loss_mode == "full")])
        fields = tuple(field.reshape((shards, 8 // shards) + field.shape[1:])
                       for field in (eta, xi, target, jnp.log(depth)[:, None], family))
        mapped = jax.vmap(namespace["evaluate"], in_axes=(None, None, None, 0, 0, 0, 0, 0), axis_name="batch")
        differentiate = jax.value_and_grad(
            lambda p, step, enabled: jax.tree.map(jnp.mean, mapped(p, step, enabled, *fields)),
            has_aux=True,
        )
        for evaluate in (differentiate, jax.jit(differentiate, static_argnums=2)):
            for step, fraction, active in (
                (59 * 32 - 16, 0.0, False), (59 * 32, 0.0, True),
                (59 * 32 + 16, 0.1, True), (61 * 32 + 16, 0.5, True),
                (64 * 32, 1.0, True), (70 * 32, 1.0, True), (64 * 32 + 1, 0.0, False),
            ):
                (loss, metrics), gradient = evaluate(params, step, True)
                weight = 0.75 * (float(active) if ramp == 0 else fraction)
                coefficients = jnp.array([1.0, 6.0 * min(step / 4000, 1), 10.0])
                expected, expected_gradient = jax.value_and_grad(
                    lambda p: coefficients @ (expected_components(p, 1.0)
                                              + weight * auxiliary_mask * expected_components(p, 3.0)),
                )(params)
                np.testing.assert_allclose((loss, gradient), (expected, expected_gradient), rtol=1e-10)
                np.testing.assert_allclose(metrics["rollout_weight"], weight, rtol=1e-7)
                assert metrics["rollout_active"] == active
                np.testing.assert_allclose([
                    metrics["rollout_data_loss"], metrics["rollout_mode_balanced_loss"],
                    metrics["rollout_translation_tangent_loss"],
                ], float(active) * auxiliary_mask * expected_components(params, 3.0), rtol=1e-10)
                np.testing.assert_allclose([
                    metrics["train_data_loss"], metrics["mode_balanced_loss"],
                    metrics["translation_tangent_loss"],
                ], expected_components(params, 1.0), rtol=1e-10)
            (loss, metrics), gradient = evaluate(params, 64 * 32, False)
            coefficients = jnp.array([1.0, 6.0 * 64 * 32 / 4000, 10.0])
            expected = jax.value_and_grad(lambda p: coefficients @ expected_components(p, 1.0))(params)
            np.testing.assert_allclose((loss, gradient), expected, rtol=1e-10)
            assert not any(key.startswith("rollout_") for key in metrics)
        if ramp == 0:
            config.rollout_training_start_epoch = 1
            (loss, metrics), gradient = differentiate(params, 0, True)
            coefficients = jnp.array([1.0, 0.0, 10.0])
            expected = jax.value_and_grad(
                lambda p: coefficients @ (expected_components(p, 1.0)
                                          + 0.75 * auxiliary_mask * expected_components(p, 3.0)),
            )(params)
            np.testing.assert_allclose((loss, gradient), expected, rtol=1e-10)
            assert metrics["rollout_active"] == 1 and metrics["rollout_weight"] == 0.75

    # The new data norm must not change original/generated MB or tangent gradients.
    config.rollout_training_loss = "full"
    config.rollout_training_start_epoch, config.rollout_training_ramp_epochs = 1, 0
    fields = tuple(value[None] for value in (eta, xi, target, jnp.log(depth)[:, None], family))
    mapped = jax.vmap(namespace["evaluate"], in_axes=(None, None, None, 0, 0, 0, 0, 0), axis_name="batch")
    controls = []
    for weight in (0.0, 4.0):
        config.data_xi_projection_weight = weight
        (loss, metrics), gradient = jax.value_and_grad(lambda p: jax.tree.map(
            jnp.mean, mapped(p, 32, True, *fields),
        ), has_aux=True)(params)
        metric_gradients = jax.jacfwd(lambda p: jax.tree.map(
            jnp.mean, mapped(p, 32, True, *fields),
        )[1])(params)
        controls.append((loss, metrics, gradient, metric_gradients))
    old_loss, old_metrics, old_gradient, old_derivatives = controls[0]
    new_loss, new_metrics, new_gradient, new_derivatives = controls[1]
    np.testing.assert_allclose(new_loss - old_loss, sum(
        factor * (new_metrics[key] - old_metrics[key])
        for key, factor in (("train_data_loss", 1), ("rollout_data_loss", 0.75))
    ), rtol=1e-10)
    np.testing.assert_allclose(new_gradient - old_gradient, sum(
        factor * (new_derivatives[key] - old_derivatives[key])
        for key, factor in (("train_data_loss", 1), ("rollout_data_loss", 0.75))
    ), rtol=1e-10)
    for key in ("mode_balanced_loss", "translation_tangent_loss", "train_l2_loss",
                "rollout_mode_balanced_loss", "rollout_translation_tangent_loss"):
        np.testing.assert_array_equal(new_metrics[key], old_metrics[key])
        np.testing.assert_array_equal(new_derivatives[key], old_derivatives[key])
    config.rollout_training_full_spectrum = True
    config.data_k_max, config.data_xi_projection_weight = 1.5, 0.0
    components = jax.vmap(
        namespace["compute_loss_components"], in_axes=(None, 0, 0, 0, 0, 0), axis_name="batch",
    )
    generated = tuple(value[None] for value in (3 * eta, 2 * xi, 1.5 * target, jnp.log(depth)[:, None], family))
    weights = jnp.array([1.0, 6.0, 10.0])
    expected, expected_gradient = jax.value_and_grad(lambda p: weights @ (
        jnp.asarray(components(p, *fields)[:3])[:, 0]
        + 0.75 * jnp.asarray(components(p, *generated, full_spectrum_rows=family[None] == 2)[:3])[:, 0]
    ))(params)
    (actual, metrics), gradient = jax.value_and_grad(lambda p: jax.tree.map(
        jnp.mean, mapped(p, 4000, True, *fields),
    ), has_aux=True)(params)
    np.testing.assert_allclose((actual, gradient), (expected, expected_gradient), rtol=1e-10)
    generated_losses = components(params, *generated, full_spectrum_rows=family[None] == 2)
    for index, key in enumerate(("rollout_data_loss", "rollout_mode_balanced_loss", "rollout_translation_tangent_loss")):
        np.testing.assert_allclose(metrics[key], generated_losses[index][0], rtol=1e-12)
    old_derivatives = jax.jacfwd(lambda p: jax.tree.map(
        jnp.mean, mapped(p, 4000, True, *fields),
    )[1])(params)
    config.rollout_training_rhs = True
    (_, rhs_metrics), _ = jax.value_and_grad(lambda p: jax.tree.map(
        jnp.mean, mapped(p, 4000, True, *fields),
    ), has_aux=True)(params)
    rhs_derivatives = jax.jacfwd(lambda p: jax.tree.map(
        jnp.mean, mapped(p, 4000, True, *fields),
    )[1])(params)
    assert not np.isclose(rhs_metrics["rollout_data_loss"], metrics["rollout_data_loss"])
    for key in metrics:
        if key != "rollout_data_loss":
            np.testing.assert_array_equal(rhs_metrics[key], metrics[key])
            np.testing.assert_array_equal(rhs_derivatives[key], old_derivatives[key])


def test_rollout_teacher_schedule_detachment_projection_and_rejection() -> None:
    from solver.solvers import time_integrator as ti

    component = trainer_function("train_step")
    prelude_end = next(i for i, node in enumerate(component.body) if isinstance(node, ast.FunctionDef))
    function = ast.parse(
        "def prepare(params, step, eta, xi, gxi, batch_depth, family):\n"
        "    current_state = SimpleNamespace(params=params, step=step)"
    ).body[0]
    assert isinstance(function, ast.FunctionDef)
    function.body += component.body[:prelude_end]
    function.body += ast.parse(
        "return (*rollout_batch[3:7], rollout_scheduled, "
        "*((rollout_batch[7],) if args.rollout_training_full_spectrum else ()))"
    ).body
    nx, size = 512, 4
    x = jnp.arange(nx, dtype=jnp.float64) * (2 * jnp.pi / nx)
    amplitude = jnp.arange(1, size + 1)[:, None]
    eta = (0.02 + amplitude * 0.0002 * jnp.cos(2 * x)).astype(jnp.float32)
    xi = (0.01 + amplitude * 0.003 * jnp.sin(x)).astype(jnp.float32)
    target = (0.2 * xi).astype(jnp.float32)
    depth = jnp.log(jnp.array([8.0, 0.3, 0.6, 1.0]))[:, None].astype(jnp.float32)
    family = jnp.arange(1, size + 1)
    fields = (eta, xi, target, depth, family)
    config = SimpleNamespace(
        datum_shift_rms=0.0, reflection_augmentation=False, input_noise_rms=0.0,
        dirichlet_probe_mode="off", dirichlet_probe_fraction=0.25,
        rollout_training_weight=1.0, rollout_training_start_epoch=60,
        rollout_training_interval=16, rollout_training_microbatch=size,
        rollout_training_steps=4, rollout_training_full_spectrum=False, rollout_coherent_weight=0.0,
        amplitude_probe_scale=0.0, poison=False, poison_target=np.nan, stage_poison=None,
    )

    def apply_model(variables: dict[str, jax.Array], inputs: jax.Array, depth: jax.Array) -> jax.Array:
        return variables["params"] * (inputs[..., :1] + inputs[..., 1:])

    def rollout(
        initial: ti.State, times: jax.Array, params: ti.SolverParams,
        predict: Callable[[jax.Array, jax.Array], jax.Array], *, substeps: int,
        return_stages: bool = False,
    ) -> dict[str, jax.Array]:
        assert substeps == 4
        velocity = predict(initial.eta, initial.xi)
        eta_new = initial.eta + times[-1] * velocity + 0.0001 * jnp.cos(140 * x)
        xi_new = initial.xi - jnp.mean(initial.xi, axis=-1, keepdims=True) + times[-1] * velocity
        xi_new += 0.0001 * jnp.sin(140 * x)
        if config.poison:
            eta_new = eta_new.at[1].set(jnp.nan).at[2].set(-2 * jnp.asarray(params.depth)[2])
        result = {"eta": jnp.stack((initial.eta, eta_new)), "xi": jnp.stack((initial.xi, xi_new))}
        if return_stages:
            stage_times = (jnp.arange(substeps)[:, None] + jnp.array([0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6])) * 0.01
            stage_eta = initial.eta + stage_times[..., None, None] * velocity + 0.0001 * jnp.cos(140 * x)
            stage_xi = initial.xi + stage_times[..., None, None] * velocity + 0.0001 * jnp.sin(140 * x)
            if config.stage_poison is not None:
                stage_eta = stage_eta.at[1, 0, 1, 5].set(config.stage_poison)
                stage_eta = stage_eta.at[2, 1, 2, 8].set(-2 * jnp.asarray(params.depth)[2, 0])
            result.update(stage_eta=stage_eta, stage_xi=stage_xi, stage_times=stage_times,
                          stage_weights=jnp.full_like(stage_times, 0.005))
        return result

    def stage_teacher(
        eta: jax.Array, xi: jax.Array, k: jax.Array, depth: jax.Array,
        order: int, *, pad_factor: int,
    ) -> jax.Array:
        assert order == 6 and pad_factor == 8 and k.dtype == eta.dtype == xi.dtype == depth.dtype == jnp.float64
        labels = 2 * xi + eta**2 + 0.1 * depth * xi
        safe_inputs = jnp.all(jnp.isfinite(eta) & jnp.isfinite(xi) & (eta + depth > 0))
        labels = jnp.where(safe_inputs, labels, jnp.nan)
        if config.stage_poison is not None:
            labels = labels.at[-1, 7].set(jnp.nan)
        return labels

    def teacher(eta: jax.Array, xi: jax.Array, depth: jax.Array, **options: Any) -> tuple[jax.Array, ...]:
        assert options.get("project_output", True) == (not config.rollout_training_full_spectrum)
        assert {key: value for key, value in options.items() if key != "project_output"} == dict(
            nx=nx, length=2 * np.pi, dno_order=6, pad_factor=8, maximum_wavenumber=128.0,
        )
        projected_eta, projected_xi, labels = compute_dno_target(eta, xi, depth, **options)
        if config.poison:
            safe_inputs = jnp.all(jnp.isfinite(eta) & jnp.isfinite(xi) & (eta + depth > 0))
            poison = config.poison_target * (jnp.cos(x) if config.rollout_training_full_spectrum else 1.0)
            labels = cast(jax.Array, jnp.where(safe_inputs, labels, jnp.nan)).at[3].set(poison)
        return projected_eta, projected_xi, labels

    namespace: dict[str, Any] = {
        "jax": jax, "jnp": jnp, "SimpleNamespace": SimpleNamespace, "args": config,
        "nx": nx, "domain_length": 2 * np.pi, "n_devices": 1, "train_steps_per_epoch": 16,
        "training_dtype": jnp.float32, "physical_dtype": jnp.float32, "apply_model": apply_model, "ti": ti, "cast": cast,
        "norm_inputs_jax": lambda eta, xi: jnp.stack((eta / 2, xi / 3), axis=-1),
        "denorm_targets_jax": lambda prediction: prediction * 4,
        "rollout_surrogate": rollout, "compute_dno_target": teacher,
        "dno_series_eval": stage_teacher, "k_grid_jax": jnp.fft.fftfreq(nx) * nx,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "trainer_rollout", "exec"), namespace)
    prepare = namespace["prepare"]
    params, active_step = jnp.float32(0.7), 59 * 16
    for step in (active_step - 16, active_step - 1, active_step + 1):
        result = prepare(params, step, *fields)
        for actual, original in zip(result[:3], fields[:3]):
            np.testing.assert_array_equal(actual, original)
        assert result[3] == 0 and not result[4]
    result = prepare(params, active_step, *fields)
    assert result[3] == 1 and result[4]
    raw = rollout(
        ti.State(eta.astype(jnp.float64), xi.astype(jnp.float64)), jnp.array([0.0, 0.04]),
        ti.make_solver_params(nx, 2 * np.pi, jnp.exp(depth.astype(jnp.float64))),
        lambda e, z: 4 * params * (e / 2 + z / 3 - jnp.mean(e / 2 + z / 3, axis=-1, keepdims=True)),
        substeps=4,
    )
    expected = teacher(raw["eta"][-1], raw["xi"][-1], jnp.exp(depth.astype(jnp.float64)),
                       nx=nx, length=2 * np.pi, dno_order=6, pad_factor=8, maximum_wavenumber=128.0,
                       project_output=True)
    for actual, projected in zip(result[:3], expected):
        assert actual.dtype == jnp.float32 and actual.shape == (size, nx)
        np.testing.assert_allclose(actual, projected, rtol=2e-5, atol=2e-8)
        np.testing.assert_allclose(np.fft.rfft(actual.astype(jnp.float64), axis=-1)[:, 129:], 0, atol=2e-7)
    np.testing.assert_allclose(result[0].mean(axis=-1), eta.mean(axis=-1), atol=1e-8)
    clipped = compute_dno_target(result[0], result[1], jnp.minimum(jnp.exp(depth.astype(jnp.float64)), 5),
                                 nx=nx, length=2 * np.pi, dno_order=6, pad_factor=8, maximum_wavenumber=128.0)[2]
    assert np.linalg.norm(np.asarray(result[2][0] - clipped[0])) > 1e-6
    detached_grad = jax.grad(lambda p: sum(jnp.sum(v**2) for v in prepare(p, active_step, *fields)[:3]))(params)
    np.testing.assert_array_equal(detached_grad, 0)

    def student_loss(p: jax.Array) -> jax.Array:
        e, z, labels, _, _ = prepare(p, active_step, *fields)
        prediction = apply_model({"params": p}, jnp.stack((e / 2, z / 3), axis=-1), depth)[..., 0] * 4
        return jnp.mean((prediction - labels) ** 2)

    value, gradient = jax.value_and_grad(student_loss)(params)
    compiled = jax.jit(jax.value_and_grad(student_loss))(params)
    np.testing.assert_allclose(compiled, (value, gradient), rtol=2e-5)
    inputs = result[0] / 2 + result[1] / 3
    expected_gradient = jnp.mean(2 * (4 * params * inputs - result[2]) * 4 * inputs)
    assert jnp.isfinite(value) and jnp.isfinite(gradient) and gradient != 0
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-6)
    config.poison = True
    for config.poison_target in (np.nan, 1e40):
        rejected = prepare(params, active_step, *fields)
        assert rejected[3] == 0.25
        for actual, original, clean in zip(rejected[:3], fields[:3], result[:3]):
            np.testing.assert_array_equal(actual[1:], original[1:])
            np.testing.assert_array_equal(actual[:1], clean[:1])
    assert "rollout" not in ast.unparse(trainer_function("eval_step"))
    assert "compute_dno_target" not in ast.unparse(trainer_function("eval_step"))

    # Full-band labels retain nonlinear harmonics; rejected rows retain their
    # original P128 target and must not inherit the valid rows' wider loss mask.
    config.rollout_training_full_spectrum, config.poison = True, False
    high_eta = eta + (0.0002 * jnp.cos(128 * x))[None].astype(jnp.float32)
    high_xi = xi + (0.003 * jnp.sin(70 * x))[None].astype(jnp.float32)
    high_fields = (high_eta, high_xi, target, depth, family)
    full_result = prepare(params, active_step, *high_fields)
    np.testing.assert_array_equal(full_result[5], jnp.ones(size, bool))
    raw_full = rollout(
        ti.State(high_eta.astype(jnp.float64), high_xi.astype(jnp.float64)), jnp.array([0.0, 0.04]),
        ti.make_solver_params(nx, 2 * np.pi, jnp.exp(depth.astype(jnp.float64))),
        lambda e, z: 4 * params * (e / 2 + z / 3 - jnp.mean(e / 2 + z / 3, axis=-1, keepdims=True)),
        substeps=4,
    )
    expected_full = compute_dno_target(
        raw_full["eta"][-1], raw_full["xi"][-1], jnp.exp(depth.astype(jnp.float64)),
        nx=nx, length=2 * np.pi, dno_order=6, pad_factor=8, maximum_wavenumber=128.0,
        project_output=False,
    )[2]
    expected_full -= expected_full.mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(full_result[2], expected_full.astype(jnp.float32), rtol=2e-6, atol=2e-8)
    assert np.linalg.norm(np.fft.rfft(full_result[2])[:, 129:]) > 1e-4
    for derivative in jax.jvp(
        lambda p: prepare(p, active_step, *high_fields)[:3], (params,), (jnp.ones_like(params),),
    )[1]:
        np.testing.assert_array_equal(derivative, 0)
    unscheduled = prepare(params, active_step + 1, *high_fields)
    np.testing.assert_array_equal(unscheduled[5], jnp.zeros(size, bool))
    for actual, original in zip(unscheduled[:3], high_fields[:3]):
        np.testing.assert_array_equal(actual, original)
    config.poison = True
    for config.poison_target in (np.nan, 1e40):
        rejected = prepare(params, active_step, *high_fields)
        np.testing.assert_array_equal(rejected[5], [True, False, False, False])
        assert rejected[3] == 0.25
        for actual, original, clean in zip(rejected[:3], high_fields[:3], full_result[:3]):
            np.testing.assert_array_equal(actual[1:], original[1:])
            np.testing.assert_array_equal(actual[:1], clean[:1])
    config.rollout_training_full_spectrum = False

    # Actual GL2 stages (including their high modes) have separate full-M6
    # labels; turning them on must not change the existing endpoint teacher.
    function.name = "prepare_coherent"
    function.body[-1] = ast.parse("return rollout_batch, coherent_batch").body[0]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 "trainer_coherent_batch", "exec"), namespace)
    prepare_coherent = namespace["prepare_coherent"]
    config.poison = False
    ordinary, absent = prepare_coherent(params, active_step, *fields)
    assert absent is None
    config.rollout_coherent_weight = 0.3
    endpoint, coherent = prepare_coherent(params, active_step, *fields)
    assert coherent[0]
    np.testing.assert_array_equal(coherent[1], depth)
    stages = coherent[2:]
    for actual, old in zip(endpoint, ordinary):
        np.testing.assert_array_equal(actual, old)
    physical_prediction = namespace["denorm_targets_jax"](apply_model(
        {"params": params}, namespace["norm_inputs_jax"](eta, xi), depth,
    ))[..., 0].astype(jnp.float64)
    frozen_velocity = physical_prediction - physical_prediction.mean(axis=-1, keepdims=True)
    raw = rollout(
        ti.State(eta.astype(jnp.float64), xi.astype(jnp.float64)), jnp.array([0.0, 0.04]),
        ti.make_solver_params(nx, 2 * np.pi, jnp.exp(depth.astype(jnp.float64))),
        lambda e, z: frozen_velocity,
        substeps=4, return_stages=True,
    )
    for actual, key in zip((*stages[:2], *stages[4:]), ("stage_eta", "stage_xi", "stage_times", "stage_weights")):
        # The independent eager FP32 predictor is not contracted like lax.cond.
        np.testing.assert_allclose(actual, raw[key], rtol=2e-7, atol=3e-10)
    physical_depth = jnp.broadcast_to(jnp.exp(depth.astype(jnp.float64)), stages[0].shape[:-1] + (1,))
    np.testing.assert_allclose(stages[2], stage_teacher(
        stages[0].reshape(-1, nx), stages[1].reshape(-1, nx), namespace["k_grid_jax"],
        physical_depth.reshape(-1, 1), 6, pad_factor=8,
    ).reshape(stages[0].shape), rtol=2e-15, atol=2e-17)
    assert np.all(stages[3]) and stages[0].shape == (4, 2, size, nx)
    for value in stages[:3]:
        assert np.linalg.norm(np.fft.rfft(value)[..., 129:]) > 0.01
    config.rollout_training_full_spectrum = True
    full_endpoint, full_stages = prepare_coherent(params, active_step, *high_fields)
    for actual, old in zip(full_endpoint[3:6], full_result[:3]):
        np.testing.assert_array_equal(actual, old)
    np.testing.assert_array_equal(full_endpoint[7], full_result[5])
    assert full_stages[2].shape == stages[0].shape
    config.rollout_training_full_spectrum = False
    for step in (active_step - 16, active_step + 1):
        inactive_endpoint, inactive_stages = prepare_coherent(params, step, *fields)
        assert not inactive_endpoint[0] and not inactive_stages[0]
        np.testing.assert_array_equal(inactive_stages[1], depth)
        for value in inactive_stages[2:]:
            np.testing.assert_array_equal(value, jnp.zeros_like(value))
    detached = jax.jvp(
        lambda p, e, z, q: prepare_coherent(p, active_step, e, z, q, depth, family)[1][2:5],
        (params, *fields[:3]), tuple(jnp.ones_like(value) for value in (params, *fields[:3])),
    )[1]
    for derivative in detached:
        np.testing.assert_array_equal(derivative, 0)
    for config.stage_poison in (np.nan, 1e40):
        unchanged_endpoint, rejected_stages = prepare_coherent(params, active_step, *fields)
        rejected_stages = rejected_stages[2:]
        for actual, old in zip(unchanged_endpoint, endpoint):
            np.testing.assert_array_equal(actual, old)
        np.testing.assert_array_equal(rejected_stages[3], [True, False, False, False])
        for rejected, clean in zip(rejected_stages[:3], stages[:3]):
            assert np.isfinite(rejected).all()
            np.testing.assert_allclose(rejected[:, :, 0], clean[:, :, 0], rtol=2e-15, atol=2e-17)
            np.testing.assert_array_equal(rejected[:, :, 1:], 0)
    config.stage_poison, config.rollout_coherent_weight = None, 0.0

    # The extra operator probes use original rows, not detached trajectory endpoints.
    function.name = "prepare_amplitude"
    function.body[-1] = ast.parse("return amplitude_batch, source_fields, rollout_depth").body[0]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 "trainer_amplitude_batch", "exec"), namespace)
    prepare_amplitude = namespace["prepare_amplitude"]
    config.poison, config.amplitude_probe_scale = False, 0.25
    probe, originals, probe_depth = prepare_amplitude(params, active_step, *fields)
    expected_probe = compute_dno_target(
        0.25 * eta, xi, jnp.exp(depth.astype(jnp.float64)), nx=nx, length=2 * np.pi,
        dno_order=6, pad_factor=8, maximum_wavenumber=128.0,
    )
    for actual, expected_field in zip(probe[:3], expected_probe):
        np.testing.assert_allclose(actual, expected_field, rtol=2e-6, atol=2e-8)
        np.testing.assert_allclose(np.fft.rfft(actual.astype(float))[:, 129:], 0, atol=2e-7)
    for actual, original in zip(originals, fields[:3]):
        np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(probe_depth, depth)
    assert np.all(probe[3]) and not np.allclose(probe[2], target)
    np.testing.assert_allclose(probe[1], xi, atol=2e-8)
    detached = jax.jvp(
        lambda e, z, q: prepare_amplitude(params, active_step, e, z, q, depth, family)[0][:3],
        fields[:3], tuple(jnp.ones_like(value) for value in fields[:3]),
    )[1]
    for derivative in detached:
        np.testing.assert_array_equal(derivative, 0)
    for step in (active_step - 16, active_step + 1):
        unscheduled = prepare_amplitude(params, step, *fields)[0]
        assert not np.any(unscheduled[3])
        for actual, original in zip(unscheduled[:3], fields[:3]):
            np.testing.assert_array_equal(actual, original)
    config.poison = True
    for config.poison_target in (np.nan, 1e40):
        rejected = prepare_amplitude(params, active_step, *fields)[0]
        np.testing.assert_array_equal(rejected[3], [True, True, True, False])
        for actual, clean, original in zip(rejected[:3], probe[:3], fields[:3]):
            np.testing.assert_array_equal(actual[:3], clean[:3])
            np.testing.assert_array_equal(actual[3], original[3])
    config.poison = False
    namespace["n_devices"] = 2
    assert prepare_amplitude(params, active_step, *fields)[0][0].shape == (size // 2, nx)
    namespace.update(n_devices=1, physical_dtype=jnp.float64)
    fields64 = (*map(jnp.float64, fields[:4]), family)
    output64 = prepare(params, active_step, *fields64)
    assert all(value.dtype == jnp.float64 for value in output64[:3])
    reference64 = compute_dno_target(
        output64[0], output64[1], jnp.exp(fields64[3]), nx=nx, length=2 * np.pi,
        dno_order=6, pad_factor=8, maximum_wavenumber=128.0,
    )[2]
    np.testing.assert_allclose(output64[2], reference64, rtol=1e-11, atol=1e-13)
    assert np.any(np.asarray(output64[2]) != np.asarray(output64[2].astype(jnp.float32)))
    for derivative in jax.jvp(
        lambda p: prepare(p, active_step, *fields64)[:3], (params,), (jnp.ones_like(params),),
    )[1]:
        np.testing.assert_array_equal(derivative, 0)
    config.amplitude_probe_scale = 0.0
    assert prepare_amplitude(params, active_step, *fields)[0] is None


def test_coherent_rollout_reprediction_schedule_ramp_and_endpoint_parity() -> None:
    nx, size = 32, 2
    x = jnp.arange(nx, dtype=jnp.float64) * (2 * jnp.pi / nx)
    eta = jnp.array([0.02, 0.04])[:, None] * jnp.cos(2 * x)
    xi = jnp.array([0.03, 0.05])[:, None] * jnp.sin(x)
    depth, family = jnp.log(jnp.array([[0.5], [8.0]])), jnp.array([1, 2])
    times = jnp.array([[0.002, 0.008], [0.012, 0.018]])
    stage_eta = eta + times[..., None, None] * xi
    stage_xi = xi - times[..., None, None] * eta
    stage_target = 0.4 * stage_xi + stage_eta**2
    weights, k = jnp.full_like(times, 0.005), jnp.fft.fftfreq(nx) * nx
    config = SimpleNamespace(
        rollout_coherent_weight=0.3, rollout_training_weight=0.75,
        rollout_training_start_epoch=2, rollout_training_ramp_epochs=1,
        rollout_training_loss="full", rollout_training_full_spectrum=False,
        mode_balanced_weight=6.0, mode_balanced_warmup_steps=1,
        translation_tangent_weight=10.0, phase_mechanism_weight=0.2,
        phase_mechanism_ramp_epochs=0.0, hadamard_weight=0.0,
    )

    def apply_model(variables: dict[str, jax.Array], inputs: jax.Array, depth: jax.Array) -> jax.Array:
        return variables["params"] * (inputs[..., :1] + inputs[..., 1:]) + 0.01 * depth[:, None, :]

    def components(p: jax.Array, e: jax.Array, *unused: Any, **options: Any) -> tuple[Any, ...]:
        losses = jnp.mean(e**2) * jnp.arange(1, 5) * p ** jnp.arange(1, 5)
        return *losses, jnp.float64(1.0), losses[0], ()

    namespace: dict[str, Any] = dict(
        jax=jax, jnp=jnp, cast=cast, FlatParams=Any, args=config, nx=nx,
        training_dtype=jnp.float64, physical_dtype=jnp.float64,
        current_state=SimpleNamespace(step=32), train_steps_per_epoch=32,
        compute_loss_components=components, compute_coherent_rollout_loss=compute_coherent_rollout_loss,
        apply_model=apply_model, norm_inputs_jax=lambda e, z: jnp.stack((e / 2, z / 3), axis=-1),
        denorm_targets_jax=lambda q: 4 * q, k_grid_jax=k,
        eta=eta, xi=xi, gxi=0.4 * xi, batch_depth=depth, family=family,
        n_data_only_rows=0, terms_phys=None, amplitude_batch=None, coherent_batch=None,
        rollout_batch=(True, depth, family, 2 * eta, xi, 0.4 * xi, jnp.float32(1)),
    )
    component = trainer_function("train_step")
    coherent_index = next(i for i, node in enumerate(component.body) if isinstance(node, ast.If)
                          and ast.unparse(node.test) == "coherent_batch is not None")
    function = ast.parse("def coherent_update(current_params):\n    current_state.params = current_params").body[0]
    assert isinstance(function, ast.FunctionDef)
    function.body += component.body[coherent_index - 1:coherent_index + 1]
    function.body += ast.parse("return (loss_value, metrics), grads").body
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[trainer_function("compute_training_loss"), function], type_ignores=[],
    )), "coherent_training_loss", "exec"), namespace)
    p = jnp.float64(0.6)
    ordinary = jax.value_and_grad(namespace["compute_training_loss"], has_aux=True)(p)
    disabled = namespace["coherent_update"](p)
    for actual, old in zip(jax.tree.leaves(disabled), jax.tree.leaves(ordinary)):
        np.testing.assert_array_equal(actual, old)
    stage_depth = jnp.broadcast_to(depth, stage_eta.shape[:-1] + (1,)).reshape(-1, 1)
    for valid in (jnp.ones(size, bool), jnp.array([True, False]), jnp.zeros(size, bool)):
        expected = jax.value_and_grad(lambda parameter: compute_coherent_rollout_loss(
            stage_eta, stage_xi,
            (4 * apply_model({"params": parameter}, jnp.stack((stage_eta.reshape(-1, nx) / 2,
             stage_xi.reshape(-1, nx) / 3), axis=-1), stage_depth))[..., 0].reshape(stage_eta.shape),
            stage_target, jnp.exp(depth).reshape(-1), k, times, weights, eta, xi, valid,
        ))(p)
        assert np.isfinite(expected).all()
        if not np.any(valid):
            np.testing.assert_array_equal(expected, (0.0, 0.0))
        else:
            assert expected[0] > 0 and expected[1] != 0
        for step in (16, 32, 48, 49, 64, 80):
            namespace["current_state"].step = step
            active = step >= 32 and step % 16 == 0
            weight = 0.75 * np.clip((step - 32) / 32, 0, 1)
            namespace["coherent_batch"] = (active, depth, stage_eta, stage_xi, stage_target, valid, times, weights)
            namespace["rollout_batch"] = (active, depth, family, 2 * eta, xi, 0.4 * xi, jnp.float32(active))
            (value, metrics), gradient = namespace["coherent_update"](p)
            (old, old_metrics), old_gradient = jax.value_and_grad(namespace["compute_training_loss"], has_aux=True)(p)
            np.testing.assert_allclose((value, gradient),
                np.asarray((old, old_gradient)) + active * weight * config.rollout_coherent_weight * np.asarray(expected),
                rtol=1e-12, atol=1e-15)
            for key in old_metrics:
                np.testing.assert_array_equal(metrics[key], old_metrics[key])
            np.testing.assert_allclose(metrics["rollout_coherent_loss"], active * expected[0], rtol=1e-12)
            np.testing.assert_array_equal(metrics["rollout_coherent_valid_fraction"], jnp.mean(valid.astype(jnp.float32)))
        if np.all(valid):
            compiled = jax.jit(namespace["coherent_update"])(p)
            for actual, eager in zip(jax.tree.leaves(compiled), jax.tree.leaves(((value, metrics), gradient))):
                np.testing.assert_allclose(actual, eager, rtol=1e-12, atol=1e-15)


def test_amplitude_probe_compensation_schedule_and_off_parity() -> None:
    main = trainer_function("main")
    option = next(node for node in ast.walk(main) if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call) and node.value.args
                  and isinstance(node.value.args[0], ast.Constant)
                  and node.value.args[0].value == "--amplitude_probe_scale")
    validations: list[ast.stmt] = [node for node in main.body if isinstance(node, ast.If)
                                  and "args.amplitude_probe_scale" in ast.unparse(node.test)]
    parser = argparse.ArgumentParser(add_help=False)
    namespace: dict[str, Any] = {"parser": parser, "np": np}
    exec(compile(ast.Module(body=[option], type_ignores=[]), "amplitude_option", "exec"), namespace)
    assert parser.parse_args([]).amplitude_probe_scale == 0.0
    validate = compile(ast.Module(body=validations, type_ignores=[]), "amplitude_validation", "exec")
    with patch.object(parser, "error", side_effect=ValueError):
        for scale, weight in ((-0.1, 1), (1.0, 1), (np.nan, 1), (np.inf, 1), (0.25, 0)):
            namespace["args"] = SimpleNamespace(amplitude_probe_scale=scale, rollout_training_weight=weight)
            with np.testing.assert_raises(ValueError):
                exec(validate, namespace)
    config = SimpleNamespace(
        amplitude_probe_scale=0.0, rollout_training_weight=0.75, rollout_training_loss="full",
        rollout_training_full_spectrum=False, rollout_coherent_weight=0.0,
        rollout_training_start_epoch=1, rollout_training_ramp_epochs=1,
        mode_balanced_weight=6.0, mode_balanced_warmup_steps=1, translation_tangent_weight=10.0,
        phase_mechanism_weight=0.1, phase_mechanism_ramp_epochs=0.0, hadamard_weight=0.0,
    )
    namespace["args"] = config
    exec(validate, namespace)
    x = jnp.arange(64, dtype=jnp.float64) * (2 * jnp.pi / 64)
    eta = jnp.array([0.1, 0.2])[:, None] * jnp.cos(x)
    xi = jnp.array([0.3, 0.4])[:, None] * jnp.sin(2 * x)
    target = xi * (1 + eta**2)
    depth, family = jnp.log(jnp.array([[0.5], [8.0]])), jnp.array([1, 4])

    def apply_model(variables: dict[str, jax.Array], inputs: jax.Array, depth: jax.Array) -> jax.Array:
        return inputs[..., 1:] * (1 + variables["params"] * inputs[..., :1]**2)

    def components(p: jax.Array, e: jax.Array, *unused: Any, **options: Any) -> tuple[Any, ...]:
        losses = jnp.mean(e**2) * jnp.arange(1, 5) * p ** jnp.arange(1, 5)
        return *losses, jnp.float64(1.0), losses[0], ()

    namespace.update(
        jax=jax, jnp=jnp, cast=cast, FlatParams=Any, training_dtype=jnp.float64, physical_dtype=jnp.float64,
        current_state=SimpleNamespace(step=32), train_steps_per_epoch=32,
        compute_loss_components=components, apply_model=apply_model,
        norm_inputs_jax=lambda e, z: jnp.stack((e, z), axis=-1),
        norm_targets_jax=lambda q: q[..., None], depth_sub=depth,
        k_rfft_jax=jnp.arange(eta.shape[1] // 2 + 1),
        eta=eta, xi=xi, gxi=target, batch_depth=depth, family=family, n_data_only_rows=0, terms_phys=None,
        amplitude_batch=None, coherent_batch=None,
        rollout_batch=(True, depth, family, 2 * eta, xi, target, jnp.float32(1)),
    )
    prior = trainer_function("compute_training_loss")
    prior.name = "prior_loss"
    for node in ast.walk(prior):
        if isinstance(node, (ast.If, ast.FunctionDef)):
            node.body = [item for item in node.body if not (
                isinstance(item, ast.If) and ast.unparse(item.test) == "amplitude_batch is not None"
            )]
    probe_function = trainer_function("compute_amplitude_loss")
    probe_branch = next(node for node in ast.walk(trainer_function("compute_training_loss"))
                        if isinstance(node, ast.If) and ast.unparse(node.test) == "amplitude_batch is not None")
    probe_function.body.insert(0, probe_branch.body[0])
    exec(compile(ast.Module(body=[prior, trainer_function("compute_training_loss"),
                                 probe_function], type_ignores=[]),
                 "amplitude_loss", "exec"), namespace)
    p = jnp.float64(0.6)
    actual = jax.value_and_grad(namespace["compute_training_loss"], has_aux=True)(p)
    expected = jax.value_and_grad(namespace["prior_loss"], has_aux=True)(p)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(a, b)
    for scale in (0.25, 0.0625):
        config.amplitude_probe_scale = scale
        namespace["amplitude_batch"] = (scale * eta, xi, xi * (1 + (scale * eta)**2), jnp.ones(2, bool))
        probe_loss = namespace["compute_amplitude_loss"]
        expected_probe = jax.value_and_grad(lambda a: relative_l2_loss(
            (xi * (1 + a * eta**2))[..., None], target[..., None],
        ))(p)
        np.testing.assert_allclose(jax.value_and_grad(probe_loss)(p), expected_probe, rtol=1e-10)
        np.testing.assert_allclose(jax.jit(jax.value_and_grad(probe_loss))(p), expected_probe, rtol=1e-10)
        np.testing.assert_array_equal(jax.value_and_grad(probe_loss)(jnp.float64(1)), (0.0, 0.0))
        for step in (0, 16, 32, 33):
            namespace["current_state"].step = step
            active, weight = step % 16 == 0, 0.75 * min(step / 32, 1.0)
            namespace["rollout_batch"] = (active, depth, family, 2 * eta, xi, target, jnp.float32(active))
            (value, metrics), gradient = jax.value_and_grad(namespace["compute_training_loss"], has_aux=True)(p)
            (ordinary, old_metrics), old_gradient = jax.value_and_grad(namespace["prior_loss"], has_aux=True)(p)
            np.testing.assert_allclose((value, gradient),
                                       np.asarray((ordinary, old_gradient)) + active * weight * np.asarray(expected_probe),
                                       rtol=1e-11)
            for key in old_metrics:
                np.testing.assert_array_equal(metrics[key], old_metrics[key])
            np.testing.assert_allclose(metrics["amplitude_data_loss"], active * expected_probe[0], rtol=1e-10)
        # Rejected rows are finite original inputs with zero contribution, including gradients.
        clean = namespace["amplitude_batch"]
        namespace["amplitude_batch"] = (*clean[:3], jnp.zeros(2, bool))
        np.testing.assert_array_equal(jax.value_and_grad(probe_loss)(p), (0.0, 0.0))


def test_amplitude_probe_retained_band_values_and_gradients() -> None:
    function = trainer_function("compute_amplitude_loss")
    branch = next(node for node in ast.walk(trainer_function("compute_training_loss"))
                  if isinstance(node, ast.If) and ast.unparse(node.test) == "amplitude_batch is not None")
    function.body.insert(0, branch.body[0])
    code = compile(ast.Module(body=[function], type_ignores=[]), "amplitude_band_loss", "exec")
    nx = 1024
    spectra = jnp.zeros((3, nx // 2 + 1), jnp.complex128).at[jnp.arange(3), jnp.array([128, 129, 257])].set(nx / 2)
    modes = jnp.fft.irfft(spectra, n=nx)
    eta, xi = jnp.ones((1, nx)), jnp.zeros((1, nx))

    def apply_model(variables: dict[str, jax.Array], inputs: jax.Array, depth: jax.Array) -> jax.Array:
        p, amplitude = variables["params"], inputs[..., :1]
        return (p[0] * amplitude**2 * modes[0, :, None]
                + amplitude * (p[1] * modes[1, :, None] + p[2] * modes[2, :, None]))

    namespace: dict[str, Any] = dict(
        jax=jax, jnp=jnp, cast=cast, FlatParams=Any, training_dtype=jnp.float64, physical_dtype=jnp.float64,
        apply_model=apply_model, norm_inputs_jax=lambda e, z: jnp.stack((e, z), axis=-1),
        norm_targets_jax=lambda q: q[..., None], depth_sub=jnp.zeros((1, 1)),
        # The denominator remains the full original-target norm, including mode 257.
        gxi=(modes[0] + 2 * modes[2])[None],
    )
    for scale, length in product((0.25, 0.0625), (2 * np.pi, 4 * np.pi)):
        namespace.update(
            args=SimpleNamespace(amplitude_probe_scale=scale),
            amplitude_batch=(scale * eta, xi, xi, jnp.ones(1, bool)),
            k_rfft_jax=(2 * np.pi / length) * jnp.arange(nx // 2 + 1),
        )
        exec(code, namespace)
        evaluate = jax.jit(jax.value_and_grad(namespace["compute_amplitude_loss"]))
        params = jnp.array([0.6, 1.7, 2.3])
        if length == 2 * np.pi:
            expected = (params[0] / np.sqrt(5), jnp.array([1 / np.sqrt(5), 0, 0]))
        else:
            norm = jnp.sqrt(params[0]**2 + (params[1] / scale)**2)
            expected = (norm / np.sqrt(5), jnp.array([params[0], params[1] / scale**2, 0]) / (norm * np.sqrt(5)))
        for actual, reference in zip(evaluate(params), expected):
            np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-13)
        # A G1-like O(lambda) tail above the physical cutoff has no loss or gradient.
        outside_only = jnp.array([0.0, 0.0, 2.3])
        value, gradient = evaluate(outside_only)
        np.testing.assert_allclose(value, 0, atol=1e-13)
        np.testing.assert_allclose(gradient[2], 0, atol=1e-13)
        assert np.isfinite(gradient).all()
        for actual, reference in zip(evaluate(params.at[2].set(0)), expected):
            np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-13)


if __name__ == "__main__":
    test_exact_fit_and_mixed_batches_have_finite_gradients()
    test_rhs_energy_error_matches_padded_equations_and_stopped_denominator()
    test_rhs_teacher_flag_requires_full_spectrum_and_active_teacher()
    test_nonlinear_denominator_prioritizes_larger_corrections()
    test_target_normalization_matches_unweighted_relative_l2()
    test_subtract_before_fft_changes_only_residual_arithmetic()
    test_modal_residual_preserves_baseline_start_dc_masks_and_ordinary_metrics()
    test_modal_residual_weights_gradients_and_symmetries()
    test_modal_residual_exact_fit_zero_residual_and_nonfinite_inputs()
    test_generated_full_spectrum_rows_preserve_fallback_losses_and_gradients()
    test_fp64_residual_uses_direct_correction_and_preserves_loss_scales()
    test_xi_projection_uses_production_loss_and_preserves_ordinary_metrics()
    test_phase_projection_weights_angular_not_radial_errors_and_preserves_symmetries()
    test_retained_band_normalizers_gradients_and_full_metrics()
    test_retained_band_zero_error_and_cutoff_validation()
    test_bounded_weights_decrease_and_have_stopped_scale_invariant_gradients()
    test_residual_denominator_gradients_scaling_and_zero_head()
    test_data_normalization_uses_the_intended_baseline()
    test_family_metrics_recombine_uneven_counts_and_isolate_nan()
    test_tangent_geometry_wiring_and_uneven_shard_reduction()
    test_terms_selector_preserves_order_and_rejects_unused_blocks()
    for baseline_order in (1, 3):
        test_terms_trainer_apply_preserves_native_and_hadamard_gradients(baseline_order)
        test_terms_full_loss_preserves_physical_baseline_and_teacher_precision(baseline_order)
    test_learning_rate_hold_schedule_and_validation()
    test_surface_head_lr_multiplier_updates_and_weight_ema()
    test_adam_moment_options_preserve_both_emas()
    test_head_whitening_calibration_and_physical_optimizer_coordinates()
    test_packed_surface_head_muon_and_unchanged_adam_emas_counters()
    test_milestone_rollouts_use_only_started_weight_ema()
    test_nested_zero_inner_initialization_and_gradient_activation()
    test_phase_rank_one_option_requires_m1_cs_network()
    test_phase_wavenumber_option_requires_m1_and_excludes_rank_one()
    test_packed_nested_branches_match_native_values_and_gradients()
    test_packed_wavenumber_phase_matches_native_values_and_gradients()
    test_packed_surface_amplitude_matches_native_values_and_gradients()
    test_packed_parity_streams_match_native_values_and_gradients()
    test_pair_coupling_options_values_and_gradients()
    test_mixed_surface_lift_options_values_and_gradients()
    test_all_momentum_ignores_family_labels_and_supervises_generated_rows()
    test_datum_shift_geometry_clean_rows_rng_and_validation()
    test_reflection_pairs_labels_preserves_rng_and_commutes_with_m6()
    test_input_noise_scale_spectrum_rng_and_depth_safety()
    test_dirichlet_probe_batch_pairing_spectrum_and_true_depth_labels()
    test_hadamard_flow_probes_route_only_training_physical_targets()
    test_dirichlet_data_uses_all_rows_but_auxiliary_gradients_use_only_prefix()
    test_fractional_rollout_ramp_parsing_validation_and_weights()
    test_momentum_ramp_weights_original_and_generated_losses_and_gradients()
    test_rollout_weight_ramp_scales_only_scheduled_student_loss()
    test_rollout_teacher_schedule_detachment_projection_and_rejection()
    test_coherent_rollout_reprediction_schedule_ramp_and_endpoint_parity()
    test_amplitude_probe_compensation_schedule_and_off_parity()
    test_amplitude_probe_retained_band_values_and_gradients()
    print("[PASS] data normalization, finite gradients, and family metric reductions")
