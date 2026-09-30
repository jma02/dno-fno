"""CPU regressions for data normalization, gradients, and family reductions."""

import ast
from collections.abc import Callable
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "models" / "fno-jax"))
sys.path.insert(0, str(REPO_ROOT / "train-jax-10m"))

from losses import relative_l2_loss  # noqa: E402
from solver.gen_data.pipeline.types import PhysicalFamilyId  # noqa: E402
from translation_tangent_regularizer import (  # noqa: E402
    compute_modal_phase_loss,
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
    full_record = aggregate(np.concatenate((np.zeros(6), metrics.ravel()))[None, :], [12])
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
            rows.append(np.concatenate((np.zeros(6), np.asarray(values))))
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
        stokes_relative_tangent=False,
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
        "jax": jax, "jnp": jnp, "FlatParams": Any, "args": config,
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
        data, _, tangent, fraction, full, _ = losses(predictions, *arguments)
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
            data, _, tangent, _, _, _ = losses(params, *arguments)
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


if __name__ == "__main__":
    test_exact_fit_and_mixed_batches_have_finite_gradients()
    test_nonlinear_denominator_prioritizes_larger_corrections()
    test_residual_denominator_gradients_scaling_and_zero_head()
    test_data_normalization_uses_the_intended_baseline()
    test_family_metrics_recombine_uneven_counts_and_isolate_nan()
    test_tangent_geometry_wiring_and_uneven_shard_reduction()
    print("[PASS] data normalization, finite gradients, and family metric reductions")
