"""CPU regressions for data normalization, gradients, and family reductions."""

import ast
from collections.abc import Callable
from itertools import product
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, cast

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
from dno_net_v2 import CraigSulemBlock, CraigSulemDNO, DepthAwareMultiplier  # noqa: E402
from mode_balanced_regularizer import compute_mode_balanced_loss  # noqa: E402
from solver.gen_data.pipeline.dno_target import compute_dno_target  # noqa: E402
from solver.gen_data.pipeline.types import PhysicalFamilyId  # noqa: E402
from translation_tangent_regularizer import (  # noqa: E402
    compute_modal_phase_loss,
    compute_phase_mechanism_loss,
    compute_translation_tangent_loss,
)

jax.config.update("jax_enable_x64", True)

TrainerLoss = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, tuple[jax.Array, tuple[jax.Array, ...]]],
]


def trainer_loss(config: SimpleNamespace) -> TrainerLoss:
    # Exercise the trainer's inline arithmetic, not a second implementation.
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
    function.body = component.body[assignments["pred_hat"]:assignments["h_phys"]]
    function.body += ast.parse("return data_loss, (full_l2_loss, family_metrics)").body
    namespace: dict[str, Any] = {
        "jax": jax, "jnp": jnp, "args": config,
        "PhysicalFamilyId": PhysicalFamilyId, "collect_family_metrics": True,
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


def test_nonlinear_denominator_prioritizes_larger_corrections() -> None:
    config = SimpleNamespace(model="cs_dno", data_normalization="nonlinear", residual_floor=1e-6,
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
    config = SimpleNamespace(model="cs_dno", data_normalization="target", residual_floor=1e-6)
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


def test_bounded_weights_decrease_and_have_stopped_scale_invariant_gradients() -> None:
    config = SimpleNamespace(model="cs_dno", data_normalization="bounded", residual_floor=1e-6,
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
    config = SimpleNamespace(model="cs_dno", data_normalization="residual", residual_floor=1e-6)
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
    config = SimpleNamespace(model="cs_dno", data_normalization="nonlinear", residual_floor=1e-6,
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
    config = SimpleNamespace(model="cs_dno", data_normalization="residual", residual_floor=1e-6)
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
    source = ast.parse(path.read_text())
    reduction = next(node for node in ast.walk(source) if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Name) and node.test.id == "val_batch_losses"
                     and any(isinstance(child, ast.Assign) and isinstance(child.targets[0], ast.Name)
                             and child.targets[0].id == "family_totals" for child in node.body))
    function = ast.parse(
        "def aggregate(losses: np.ndarray, val_batch_sizes: list[int]) "
        "-> dict[str, float]: epoch_record = {}"
    ).body[0]
    assert isinstance(function, ast.FunctionDef)
    function.body += reduction.body + ast.parse("return epoch_record").body
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
        model="fno", residual_floor=1e-6, mode_balanced_weight=0.0,
        translation_tangent_weight=0.3, translation_tangent_scope="tanaka",
        translation_tangent_geometry="coupled_modal", translation_tangent_k_max=1,
        translation_tangent_denominator_eps=1e-3, translation_tangent_loss_norm="charbonnier",
        translation_tangent_charbonnier_eps=1e-7, translation_tangent_smoothing_scale=1.0,
        stokes_relative_tangent=False, phase_mechanism="none",
    )

    def apply_model(
        variables: dict[str, jax.Array], inputs: jax.Array, depth: jax.Array,
        *, return_baseline: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        prediction = (variables["params"][..., None] - 7.0) / 4.0
        return prediction, jnp.zeros_like(prediction), jnp.zeros_like(prediction)

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
        "training_dtype": jnp.float64, "PhysicalFamilyId": PhysicalFamilyId,
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


def test_packed_nested_branches_match_native_values_and_gradients() -> None:
    nx = 32
    inputs = 0.1 * jax.random.normal(jax.random.PRNGKey(7), (3, nx, 2), dtype=jnp.float32)
    depth = jnp.log(jnp.asarray([[0.1], [2.3], [8.0]], dtype=jnp.float32))
    probe = jax.random.normal(jax.random.PRNGKey(9), (3, nx, 1), dtype=jnp.float32)
    for nested_blocks, polynomial_skip in ((0, False), (1, False), (2, False), (1, True)):
        model = CraigSulemDNO(
            width=8, n_blocks=2, nested_blocks=nested_blocks, latent=2, mult_hidden=4,
            surface_feature_scales=(0.3, 0.1, 0.04, 0.7, 2.0, 0.5, 0.3),
            surface_context_width=2, baseline_order=1, baseline_true_depth=True,
            surface_polynomial_skip=polynomial_skip,
            eta_scale=0.13, xi_scale=0.08, target_scale=0.12,
        )
        params = dict(model.init(jax.random.PRNGKey(3), inputs, depth)["params"])
        for index in range(2):
            group = params[f"cs_block_{index}"]
            if index < 2 - nested_blocks:
                shape = cast(jax.Array, group["phi_proj"]["kernel"]).shape
                group["phi_proj"]["kernel"] = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) * 0.02 - 0.07
            else:
                group["gamma"] = jnp.array([0.2, -0.3], dtype=jnp.float32)
        namespace: dict[str, Any] = {
            "jax": jax, "jnp": jnp, "nn": nn, "Any": Any, "cast": cast, "model": model,
            "CraigSulemBlock": CraigSulemBlock, "DepthAwareMultiplier": DepthAwareMultiplier,
            "nx": nx, "domain_length": float(model.domain_length),
            "args": SimpleNamespace(model="cs_dno", n_blocks=2, cs_nested_blocks=nested_blocks,
                                    latent=2, cs_mult_hidden=4, cs_surface_polynomial_skip=polynomial_skip),
        }
        exec(compile(ast.Module(body=[trainer_function("apply_model")], type_ignores=[]),
                     "trainer_packed_apply", "exec"), namespace)
        packed_apply = namespace["apply_model"]
        native = cast(tuple[jax.Array, jax.Array, jax.Array],
                      model.apply({"params": params}, inputs, depth, return_baseline=True))
        packed = jax.jit(packed_apply, static_argnames="return_baseline")(
            {"params": params}, inputs, depth, return_baseline=True,
        )
        for actual, expected in zip(packed, native):
            np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-7)
        np.testing.assert_array_equal(packed[1], packed[2])

        def objective(parameters: dict[str, Any], values: jax.Array, *, packed: bool) -> jax.Array:
            apply = packed_apply if packed else model.apply
            return jnp.sum(cast(jax.Array, apply({"params": parameters}, values, depth)) * probe)

        differentiate = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)), static_argnames="packed")
        expected_value, expected_grad = differentiate(params, inputs, packed=False)
        actual_value, actual_grad = differentiate(params, inputs, packed=True)
        np.testing.assert_allclose(actual_value, expected_value, rtol=3e-6, atol=2e-7)
        for actual, expected in zip(jax.tree.leaves(actual_grad), jax.tree.leaves(expected_grad)):
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-7)
            assert bool(jnp.all(jnp.isfinite(actual)))
        for index in range(2 - nested_blocks, 2):
            group_grad = actual_grad[0][f"cs_block_{index}"]
            for part in ("gamma", "m_inner", "m_shared"):
                assert sum(float(jnp.linalg.norm(value)) for value in jax.tree.leaves(group_grad[part])) > 0
        if polynomial_skip:
            assert float(jnp.linalg.norm(actual_grad[0]["cs_block_0"]["phi_proj"]["kernel"][-2:])) > 0


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
        model="fno", residual_floor=1e-6, mode_balanced_weight=0.0,
        translation_tangent_weight=0.0, phase_mechanism="all_momentum",
        phase_mechanism_weight=0.03, hadamard_weight=0.0, mode_balanced_warmup_steps=1,
        rollout_training_weight=0.7, rollout_training_ramp_epochs=0,
        rollout_training_loss="full",
    )

    def apply_model(
        variables: dict[str, jax.Array], inputs: jax.Array, depth: jax.Array, *, return_baseline: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        prediction = variables["params"][:, None, None] * jnp.sum(inputs, axis=-1, keepdims=True)
        return prediction, jnp.zeros_like(prediction), jnp.zeros_like(prediction)

    namespace: dict[str, Any] = {
        "jax": jax, "jnp": jnp, "Any": Any, "cast": cast, "FlatParams": Any, "args": config,
        "training_dtype": jnp.float64, "PhysicalFamilyId": PhysicalFamilyId,
        "norm_inputs_jax": lambda e, p: jnp.stack((e, p), axis=-1),
        "norm_targets_jax": lambda q: q[..., None], "denorm_targets_jax": lambda q: q,
        "log_h_max": np.log(5.0), "k_grid_jax": jnp.fft.fftfreq(nx) * nx,
        "compute_phase_mechanism_loss": compute_phase_mechanism_loss, "apply_model": apply_model,
        "current_state": SimpleNamespace(step=100), "n_data_only_rows": 0,
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
    for loss_mode in ("full", "data"):
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
    config = SimpleNamespace(reflection_augmentation=False, input_noise_rms=0.0,
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


def test_dirichlet_data_uses_all_rows_but_auxiliary_gradients_use_only_prefix() -> None:
    component = trainer_function("compute_loss_components")
    config = SimpleNamespace(
        model="fno", residual_floor=1e-6, mode_balanced_weight=6.0,
        mode_balanced_k_max=128.0, mode_balanced_activity_threshold=1e-4,
        mode_balanced_denominator_eps=1e-6, translation_tangent_weight=10.0,
        translation_tangent_scope="tanaka", translation_tangent_geometry="local",
        translation_tangent_denominator_eps=1e-3, translation_tangent_loss_norm="squared",
        translation_tangent_charbonnier_eps=1e-7, translation_tangent_smoothing_scale=1.0,
        stokes_relative_tangent=False, phase_mechanism="none", jonswap_mode_balanced_scale=1.0,
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
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        result = variables["params"][..., None] / 4
        return result, jnp.zeros_like(result), jnp.zeros_like(result)

    namespace: dict[str, Any] = {
        "jax": jax, "jnp": jnp, "FlatParams": Any, "args": config, "cast": cast,
        "training_dtype": jnp.float64, "PhysicalFamilyId": PhysicalFamilyId,
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
        "jnp.float32(rollout_scheduled)) if enabled else None"
    ).body
    function.body += [trainer_function("compute_training_loss")]
    function.body += ast.parse("return compute_training_loss(current_params)").body
    config = SimpleNamespace(
        rollout_training_weight=0.75, rollout_training_start_epoch=60,
        rollout_training_ramp_epochs=5, rollout_training_interval=16,
        rollout_training_loss="full",
        model="fno", residual_floor=1e-6, mode_balanced_weight=6.0,
        mode_balanced_warmup_steps=4000, mode_balanced_k_max=128.0,
        mode_balanced_activity_threshold=1e-4, mode_balanced_denominator_eps=1e-6,
        translation_tangent_weight=10.0, translation_tangent_scope="tanaka",
        translation_tangent_geometry="local", translation_tangent_denominator_eps=1e-3,
        translation_tangent_loss_norm="squared", translation_tangent_charbonnier_eps=1e-7,
        translation_tangent_smoothing_scale=1.0, stokes_relative_tangent=False,
        hadamard_weight=0.0, phase_mechanism="none", phase_mechanism_weight=0.0,
        jonswap_mode_balanced_scale=1.0,
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
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        prediction = variables["params"] * jnp.sum(inputs, axis=-1, keepdims=True) / 4
        return prediction, jnp.zeros_like(prediction), jnp.zeros_like(prediction)

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
        "args": config, "train_steps_per_epoch": 32, "training_dtype": jnp.float64,
        "n_data_only_rows": 0, "PhysicalFamilyId": PhysicalFamilyId,
        "norm_inputs_jax": lambda eta, xi: jnp.stack((eta / 2, xi / 3), axis=-1),
        "norm_targets_jax": lambda target: target[..., None] / 4,
        "denorm_targets_jax": lambda target: target * 4,
        "apply_model": apply_model, "log_h_max": np.log(5.0),
        "k_grid_jax": k, "k_rfft_jax": jnp.arange(nx // 2 + 1),
        "compute_mode_balanced_loss": compute_mode_balanced_loss,
        "compute_translation_tangent_loss": compute_translation_tangent_loss,
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
        "return rollout_eta, rollout_xi, rollout_target, rollout_valid, rollout_scheduled"
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
        reflection_augmentation=False, input_noise_rms=0.0,
        dirichlet_probe_mode="off", dirichlet_probe_fraction=0.25,
        rollout_training_weight=1.0, rollout_training_start_epoch=60,
        rollout_training_interval=16, rollout_training_microbatch=size,
        rollout_training_steps=4, poison=False, poison_target=np.nan,
    )

    def apply_model(variables: dict[str, jax.Array], inputs: jax.Array, depth: jax.Array) -> jax.Array:
        return variables["params"] * (inputs[..., :1] + inputs[..., 1:])

    def rollout(
        initial: ti.State, times: jax.Array, params: ti.SolverParams,
        predict: Callable[[jax.Array, jax.Array], jax.Array], *, substeps: int,
    ) -> dict[str, jax.Array]:
        assert substeps == 4
        velocity = predict(initial.eta, initial.xi)
        eta_new = initial.eta + times[-1] * velocity + 0.0001 * jnp.cos(140 * x)
        xi_new = initial.xi - jnp.mean(initial.xi, axis=-1, keepdims=True) + times[-1] * velocity
        xi_new += 0.0001 * jnp.sin(140 * x)
        if config.poison:
            eta_new = eta_new.at[1].set(jnp.nan).at[2].set(-2 * jnp.asarray(params.depth)[2])
        return {"eta": jnp.stack((initial.eta, eta_new)), "xi": jnp.stack((initial.xi, xi_new))}

    def teacher(eta: jax.Array, xi: jax.Array, depth: jax.Array, **options: Any) -> tuple[jax.Array, ...]:
        assert options == dict(nx=nx, length=2 * np.pi, dno_order=6, pad_factor=8, maximum_wavenumber=128.0)
        projected_eta, projected_xi, labels = compute_dno_target(eta, xi, depth, **options)
        if config.poison:
            safe_inputs = jnp.all(jnp.isfinite(eta) & jnp.isfinite(xi) & (eta + depth > 0))
            labels = cast(jax.Array, jnp.where(safe_inputs, labels, jnp.nan)).at[3].set(config.poison_target)
        return projected_eta, projected_xi, labels

    namespace: dict[str, Any] = {
        "jax": jax, "jnp": jnp, "SimpleNamespace": SimpleNamespace, "args": config,
        "nx": nx, "domain_length": 2 * np.pi, "n_devices": 1, "train_steps_per_epoch": 16,
        "training_dtype": jnp.float32, "apply_model": apply_model, "ti": ti, "cast": cast,
        "norm_inputs_jax": lambda eta, xi: jnp.stack((eta / 2, xi / 3), axis=-1),
        "denorm_targets_jax": lambda prediction: prediction * 4,
        "rollout_surrogate": rollout, "compute_dno_target": teacher,
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
                       nx=nx, length=2 * np.pi, dno_order=6, pad_factor=8, maximum_wavenumber=128.0)
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


if __name__ == "__main__":
    test_exact_fit_and_mixed_batches_have_finite_gradients()
    test_nonlinear_denominator_prioritizes_larger_corrections()
    test_target_normalization_matches_unweighted_relative_l2()
    test_bounded_weights_decrease_and_have_stopped_scale_invariant_gradients()
    test_residual_denominator_gradients_scaling_and_zero_head()
    test_data_normalization_uses_the_intended_baseline()
    test_family_metrics_recombine_uneven_counts_and_isolate_nan()
    test_tangent_geometry_wiring_and_uneven_shard_reduction()
    test_packed_nested_branches_match_native_values_and_gradients()
    test_all_momentum_ignores_family_labels_and_supervises_generated_rows()
    test_reflection_pairs_labels_preserves_rng_and_commutes_with_m6()
    test_input_noise_scale_spectrum_rng_and_depth_safety()
    test_dirichlet_probe_batch_pairing_spectrum_and_true_depth_labels()
    test_dirichlet_data_uses_all_rows_but_auxiliary_gradients_use_only_prefix()
    test_rollout_weight_ramp_scales_only_scheduled_student_loss()
    test_rollout_teacher_schedule_detachment_projection_and_rejection()
    print("[PASS] data normalization, finite gradients, and family metric reductions")
