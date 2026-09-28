"""CPU regressions for supervised-loss gradients and nonlinear sample weighting."""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models" / "fno-jax"))

from losses import relative_l2_loss  # noqa: E402

jax.config.update("jax_enable_x64", True)


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


def test_nonlinear_denominator_prioritizes_larger_corrections() -> None:
    # Exercise the trainer's inline arithmetic, not a second implementation.
    path = Path(__file__).resolve().parents[1] / "1d_dno_fno_jax_ema_grad.py"
    source = ast.parse(path.read_text())
    component = next(node for node in ast.walk(source)
                     if isinstance(node, ast.FunctionDef) and node.name == "compute_loss_components")
    assignments = {node.targets[0].id: i for i, node in enumerate(component.body)
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    function = ast.parse(
        "def losses(predictions: jax.Array, batch_targets: jax.Array, baseline: jax.Array)"
        " -> tuple[jax.Array, jax.Array]: pass"
    ).body[0]
    assert isinstance(function, ast.FunctionDef)
    function.body = component.body[assignments["pred_hat"]:assignments["full_l2_loss"] + 1]
    function.body += ast.parse("return data_loss, full_l2_loss").body
    config = SimpleNamespace(model="cs_dno", nonlinear_reference=0.002, nonlinear_weight_mean=1.463313)
    namespace: dict[str, Any] = {"jax": jax, "jnp": jnp, "args": config}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), namespace)
    losses = namespace["losses"]
    evaluate = jax.jit(jax.value_and_grad(losses, argnums=(0, 1, 2), has_aux=True))
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
            (weighted, full), gradients = evaluate(prediction, target, baseline)
            np.testing.assert_array_equal(full, ordinary)
            weight = float(weighted / ordinary)
            weights.append(weight)
            np.testing.assert_allclose(gradients[0], weight * ordinary_gradient, rtol=3e-5, atol=1e-8)
            np.testing.assert_array_equal(gradients[2], 0)
            scaled, _ = losses(7 * prediction, 7 * target, 7 * baseline)
            np.testing.assert_allclose(scaled, weighted, rtol=1e-5)
        assert np.all(np.diff(weights) > 0)
        np.testing.assert_allclose(weights, expected, rtol=5e-6)
        baselines = (1 - jnp.asarray(ratios, dtype=dtype)[:, None, None]) * target
        mixed, _ = losses(jnp.broadcast_to(prediction, baselines.shape),
                          jnp.broadcast_to(target, baselines.shape), baselines)
        np.testing.assert_allclose(mixed, ordinary * np.mean(weights), rtol=1e-6)
        for scale in (0.0, 1e-14, 1.0):
            for exact in (True, False):
                truth = target * scale
                pred = truth if exact else truth + 1e-8
                (value, full), gradients = evaluate(pred, truth, truth)
                assert bool(jnp.isfinite(value) & jnp.isfinite(full))
                assert all(bool(jnp.all(jnp.isfinite(gradient))) for gradient in gradients)
                if exact:
                    assert value == full == 0
                    for gradient in gradients:
                        np.testing.assert_array_equal(gradient, 0)
        config.nonlinear_weight_mean = 1.0
        np.testing.assert_array_equal(losses(prediction, target, target)[0], ordinary)
        config.nonlinear_weight_mean = 1.463313
        config.model = "fno"
        np.testing.assert_array_equal(losses(prediction, target, jnp.zeros_like(target))[0], ordinary)
        config.model = "cs_dno"


if __name__ == "__main__":
    test_exact_fit_and_mixed_batches_have_finite_gradients()
    test_nonlinear_denominator_prioritizes_larger_corrections()
    print("[PASS] finite loss gradients and increasing nonlinear sample weights")
