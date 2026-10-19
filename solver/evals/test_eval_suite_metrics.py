"""Focused regression tests for manuscript rollout metrics."""

from __future__ import annotations

import ast
from functools import partial
from importlib import import_module
from itertools import product
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Literal, cast
import unittest
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from solver.evals.eval_suite import (  # noqa: E402
    FAMILY_CONFIGS,
    IC,
    FamilyConfig,
    _json_ready,
    _load_paper_dataset_ics,
    _rollout_ic_chunks,
    _try_load_cached_truth,
    _write_truth_cache,
    _write_json,
    compute_macro_summary,
    compute_metrics,
    main as eval_main,
    run_family,
)
from solver.evals.model_rollout import LoadedRun, build_predict_gxi_batched, load_run  # noqa: E402
from dno_net_v2 import CraigSulemDNO, CraigSulemTermsResidual  # noqa: E402
from fno1d import SpectralConv1d  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from solver.gen_data.pipeline.batch_storage import save_completed_batch  # noqa: E402
from scripts.build_paper_dataset import build_dataset  # noqa: E402
from solver.gen_data.pipeline.types import (  # noqa: E402
    PhysicalFamilyId,
    SimulationRows,
)


class ComputeMetricsTest(unittest.TestCase):
    def test_trainer_milestones_evaluate_weight_ema_without_changing_training(self) -> None:
        path = Path(__file__).resolve().parents[2] / "train-jax-10m" / "1d_dno_fno_jax_ema_grad.py"
        milestone = next(
            node for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.If) and ast.unparse(node.test) == "epoch in args.rollout_epochs"
        )
        code = compile(ast.Module(body=[milestone], type_ignores=[]), str(path), "exec")
        model = CraigSulemDNO(width=8, n_blocks=1, latent=2, mult_hidden=4)
        inputs = 0.01 * jax.random.normal(jax.random.PRNGKey(4), (1, 16, 2), dtype=jnp.float32)
        depth = jnp.zeros((1, 1), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), inputs, depth)["params"]
        state = import_module("train-jax-10m.checkpoint_util").WeightEMATrainState.create(
            apply_fn=model.apply, params=params,
            tx=optax.chain(optax.ema(0.9, debias=True), optax.adamw(2e-5)),
            weight_ema_params=params, weight_ema_decay=0.5,
        )
        gradients = jax.tree_util.tree_map(jnp.ones_like, params)
        state = state.apply_gradients(grads=gradients)
        state = state.apply_gradients(grads=gradients)
        sharding = jax.sharding.NamedSharding(
            jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("data",)), jax.sharding.PartitionSpec(),
        )
        state = jax.device_put(state, sharding)
        self.assertTrue(any(
            not np.array_equal(raw, averaged)
            for raw, averaged in zip(jax.tree_util.tree_leaves(state.params),
                                     jax.tree_util.tree_leaves(state.weight_ema_params), strict=True)
        ))
        before = jax.tree_util.tree_map(lambda value: np.asarray(value).copy(), state)
        expected_next = state.apply_gradients(grads=gradients)
        config = {"model": "cs_dno"}
        stats = {"feature_absmax": [1.0, 1.0], "target_absmax": 1.0}
        train_rng = jax.random.PRNGKey(7)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for epoch in (9, 10, 11, 30, 60, 120):
                with self.subTest(epoch=epoch):
                    build = Mock(wraps=build_predict_gxi_batched)
                    evaluate = Mock(return_value={"epoch": epoch})
                    macro = Mock(return_value={"n_families": 2})
                    namespace: dict[str, Any] = {
                        "epoch": epoch,
                        "args": SimpleNamespace(rollout_epochs=[10, 30, 60],
                                                rollout_truth_run=str(directory / "truth"), norm="scale"),
                        "model": model, "training_state": state, "config_payload": config,
                        "stats": stats, "run_dir": directory, "dataset_path": directory / "dataset",
                        "LoadedRun": LoadedRun, "build_predict_gxi_batched": build,
                        "run_family": evaluate, "FAMILY_CONFIGS": FAMILY_CONFIGS,
                        "compute_macro_summary": macro, "Path": Path, "_write_json": _write_json,
                        "print": Mock(), "train_rng": train_rng,
                        "epoch_end_step": 2, "epoch_end_optimizer_count": 2,
                    }
                    exec(code, namespace)
                    self.assertIs(namespace["training_state"], state)
                    self.assertIs(namespace["train_rng"], train_rng)
                    self.assertEqual(namespace["epoch_end_step"], 2)
                    self.assertEqual(namespace["epoch_end_optimizer_count"], 2)
                    for actual, expected in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(before)):
                        np.testing.assert_array_equal(actual, expected)
                    if epoch not in (10, 30, 60):
                        build.assert_not_called()
                        evaluate.assert_not_called()
                        macro.assert_not_called()
                        continue
                    loaded = build.call_args.args[0]
                    build.assert_called_once_with(loaded)
                    self.assertIs(loaded.model, model)
                    self.assertIs(loaded.params, state.weight_ema_params)
                    self.assertIsNot(loaded.params, state.params)
                    self.assertIs(loaded.config, config)
                    self.assertIs(loaded.stats, stats)
                    self.assertEqual((loaded.epoch, loaded.norm_mode), (epoch, "scale"))
                    self.assertEqual(evaluate.call_count, 4)
                    for index, family in enumerate(("stokes", "tanaka", "jonswap_tma", "benjamin_feir")):
                        call = evaluate.call_args_list[index]
                        group = index // 2
                        out = directory / f"eval_epoch{epoch}_weight_ema_test_stratified_n32_gpu{group}"
                        cache = directory / "truth" / f"eval_best_current_test_stratified_n32_fp32net_fp64solver_gpu{group}"
                        self.assertEqual(call.args[:2], (family, FAMILY_CONFIGS[family]))
                        self.assertIs(call.args[2], loaded)
                        self.assertIs(call.args[3], evaluate.call_args_list[0].args[3])
                        self.assertEqual(call.args[4:], (
                            out, directory / "dataset", 32, cache,
                            {"run_dir": str(directory), "selection": "weight_ema",
                             "path": str(directory / "weight_ema_ckpt"), "epoch": epoch}, 32,
                        ))
                        self.assertEqual(call.kwargs, {"ic_selection": "stratified"})
                        self.assertEqual(json.loads((out / "all_summaries.json").read_text())[family], {"epoch": epoch})
                        self.assertEqual(json.loads((out / "macro_summary.json").read_text()), {"n_families": 2})
                    predictor = evaluate.call_args_list[0].args[3]
                    expected = build_predict_gxi_batched(loaded, center_xi=False)
                    np.testing.assert_array_equal(
                        predictor(inputs[..., 0], inputs[..., 1], depth),
                        expected(inputs[..., 0], inputs[..., 1], depth),
                    )
                    actual_next = namespace["training_state"].apply_gradients(grads=gradients)
                    for actual, expected in zip(jax.tree_util.tree_leaves(actual_next), jax.tree_util.tree_leaves(expected_next)):
                        np.testing.assert_array_equal(actual, expected)

    def test_checkpoint_loader_restores_cs_architecture_options(self) -> None:
        jax.devices()  # Initialize plugin discovery before mocking Path.read_text.
        config = {
            "model": "cs_dno", "norm": "scale", "width": 8, "n_blocks": 1,
            "latent": 2, "cs_n_polys": 3, "cs_mult_hidden": 4,
            "cs_use_first_deriv": True, "cs_use_second_deriv": True,
            "cs_use_half_deriv": True, "cs_use_hilbert": True,
            "domain_length": 2 * np.pi, "xi_scale": 1.0,
            "eta_scale": 1.0, "target_scale": 1.0,
        }
        for options in (
            {}, {"cs_use_relative_elevation": True}, {"cs_baseline_order": 2},
            {"cs_nested_blocks": 1, "cs_baseline_order": 1},
            {"cs_nested_blocks": 1, "cs_nested_surface_modulation": True},
            {"cs_nested_blocks": 1, "cs_nested_surface_features": True,
             "cs_surface_feature_scales": [0.1, 0.02, 0.003, 1.0, 10.0, 0.3, 0.1]},
            {"cs_baseline_true_depth": True},
            {"cs_use_learned_eta_filters": True},
            {"cs_curvature_cutoff": 22.727203594311888},
            {"cs_surface_feature_scales": [0.1, 0.02, 0.003, 1.0, 10.0, 0.3, 0.1]},
            {"cs_surface_feature_transform": (np.eye(7) + 0.1).tolist()},
            {"cs_surface_context_width": 4},
            {"cs_surface_context_width": 4, "cs_surface_context_gate": True},
            {"cs_surface_depth_gate": True},
            {"cs_surface_polynomial_skip": True},
            {"cs_multiplier_skip": True},
            {"cs_multiplier_skip": True, "cs_nested_blocks": 1},
            {"cs_log_multiplier_inputs": True},
            {"cs_log_multiplier_inputs": True, "cs_nested_blocks": 1},
            {"cs_surface_polynomial_skip": True, "cs_surface_feature_scales": [0.1, 0.02, 0.003, 10.0, 0.3],
             "cs_use_first_deriv": False, "cs_use_hilbert": False},
            {"cs_surface_feature_scales": [0.1, 0.02, 0.003, 1.0, 10.0, 0.3, 0.1],
             "cs_surface_context_width": 4},
            {"cs_use_relative_elevation": True, "cs_baseline_order": 2, "cs_baseline_true_depth": True},
        ):
            with self.subTest(options=options), patch(
                "solver.evals.model_rollout.Path.read_text",
                side_effect=[json.dumps({**config, **options}), json.dumps({"epoch": 1, "stats": {}})],
            ), patch(
                "solver.evals.model_rollout.checkpoints.restore_checkpoint",
                return_value={"params": {}},
            ), patch(
                "solver.evals.model_rollout.ocp.PyTreeCheckpointer",
            ):
                loaded = load_run("/tmp/dno-architecture-loader-test")
                self.assertEqual(loaded.model.use_relative_elevation, options.get("cs_use_relative_elevation", False))
                self.assertEqual(loaded.model.use_learned_eta_filters, options.get("cs_use_learned_eta_filters", False))
                self.assertEqual(loaded.model.curvature_cutoff, options.get("cs_curvature_cutoff", 0.0))
                self.assertEqual(loaded.model.baseline_order, options.get("cs_baseline_order", 1))
                self.assertEqual(loaded.model.nested_blocks, options.get("cs_nested_blocks", 0))
                self.assertEqual(loaded.model.nested_surface_modulation,
                                 options.get("cs_nested_surface_modulation", False))
                self.assertEqual(loaded.model.nested_surface_features,
                                 options.get("cs_nested_surface_features", False))
                self.assertEqual(loaded.model.baseline_true_depth, options.get("cs_baseline_true_depth", False))
                scales = cast(list[float] | None, options.get("cs_surface_feature_scales"))
                self.assertEqual(loaded.model.surface_feature_scales, None if scales is None else tuple(scales))
                transform = cast(list[list[float]] | None, options.get("cs_surface_feature_transform"))
                self.assertEqual(loaded.model.surface_feature_transform,
                                 None if transform is None else tuple(map(tuple, transform)))
                self.assertEqual(loaded.model.surface_context_width, options.get("cs_surface_context_width", 0))
                self.assertEqual(loaded.model.surface_context_gate, options.get("cs_surface_context_gate", False))
                self.assertEqual(loaded.model.surface_depth_gate, options.get("cs_surface_depth_gate", False))
                self.assertEqual(loaded.model.surface_polynomial_skip, options.get("cs_surface_polynomial_skip", False))
                self.assertEqual(loaded.model.multiplier_skip, options.get("cs_multiplier_skip", False))
                self.assertEqual(loaded.model.log_multiplier_inputs, options.get("cs_log_multiplier_inputs", False))
                self.assertEqual(loaded.model.use_first_deriv, options.get("cs_use_first_deriv", True))
                self.assertEqual(loaded.model.use_hilbert, options.get("cs_use_hilbert", True))

        selections: tuple[Literal["weight_ema", "weight_ema_secondary"], ...] = ("weight_ema", "weight_ema_secondary")
        for selection, value in zip(selections, (2.0, 3.0), strict=True):
            with patch("solver.evals.model_rollout.Path.read_text", side_effect=[
                json.dumps(config), json.dumps({"epoch": 1, "stats": {}}),
            ]), patch("solver.evals.model_rollout.checkpoints.restore_checkpoint", return_value={
                "params": {"value": np.array([value], dtype=np.float32)},
            }) as restore, patch("solver.evals.model_rollout.ocp.PyTreeCheckpointer"):
                loaded = load_run("/tmp/dno-architecture-loader-test", checkpoint=selection)
                self.assertEqual(restore.call_args.kwargs["ckpt_dir"].name, f"{selection}_ckpt")
            np.testing.assert_array_equal(loaded.params["value"], [value])
            self.assertEqual(loaded.epoch, 1)

        for checkpoint_dir in (None, Path("/tmp/dno-preserved-epoch37")):
            with self.subTest(checkpoint_dir=checkpoint_dir), patch(
                "solver.evals.model_rollout.Path.read_text", side_effect=[
                    json.dumps(config), json.dumps({"epoch": 37, "stats": {}}),
                ],
            ), patch("solver.evals.model_rollout.checkpoints.restore_checkpoint", return_value={
                "params": {"value": np.array([-1.0], dtype=np.float32)},
                "weight_ema_params": {"value": np.array([2.0], dtype=np.float32)},
            }) as restore, patch("solver.evals.model_rollout.ocp.PyTreeCheckpointer"):
                loaded = load_run("/tmp/dno-architecture-loader-test", checkpoint="best_weight_ema",
                                  checkpoint_dir=checkpoint_dir)
                self.assertEqual(restore.call_args.kwargs["ckpt_dir"], checkpoint_dir or
                                 Path("/tmp/dno-architecture-loader-test/best_val_ckpt"))
                self.assertEqual(restore.call_args.kwargs["step"], 37)
                np.testing.assert_array_equal(loaded.params["value"], [2.0])
                self.assertEqual(loaded.epoch, 37)

    def test_nested_checkpoint_restores_nonzero_correction_and_m1_baseline(self) -> None:
        model = CraigSulemDNO(width=8, n_blocks=1, nested_blocks=1, latent=2, mult_hidden=4,
                             baseline_order=1, nested_surface_modulation=True)
        inputs = 0.1 * jax.random.normal(jax.random.PRNGKey(4), (2, 32, 2), dtype=jnp.float32)
        depth = jnp.log(jnp.asarray([[0.2], [8.0]], dtype=jnp.float32))
        params = dict(model.init(jax.random.PRNGKey(0), inputs, depth)["params"])
        params["cs_block_0"]["gamma"] = jnp.full((2,), 0.1, dtype=jnp.float32)
        params["nested_surface_gain"] = jnp.asarray(0.6, dtype=jnp.float32)
        prediction, baseline, m1, correction = cast(tuple[jax.Array, jax.Array, jax.Array, jax.Array],
                                       model.apply({"params": params}, inputs, depth, return_baseline=True))
        np.testing.assert_array_equal(baseline, m1)
        np.testing.assert_array_equal(prediction, baseline + correction)
        self.assertGreater(float(jnp.linalg.norm(prediction - baseline)), 1e-7)
        config = {
            "model": "cs_dno", "norm": "scale", "width": 8, "n_blocks": 1,
            "cs_nested_blocks": 1, "latent": 2, "cs_n_polys": 3, "cs_mult_hidden": 4,
            "cs_nested_surface_modulation": True,
            "cs_use_first_deriv": True, "cs_use_second_deriv": True,
            "cs_use_half_deriv": True, "cs_use_hilbert": True,
            "cs_baseline_order": 1, "cs_baseline_true_depth": False,
            "domain_length": 2 * np.pi, "xi_scale": 1.0, "eta_scale": 1.0, "target_scale": 1.0,
        }
        stats = {"feature_absmax": [1.0, 1.0], "target_absmax": 1.0}
        with patch("solver.evals.model_rollout.Path.read_text", side_effect=[
            json.dumps(config), json.dumps({"epoch": 40, "stats": stats}),
        ]), patch("solver.evals.model_rollout.checkpoints.restore_checkpoint", return_value={
            "params": params,
        }), patch("solver.evals.model_rollout.ocp.PyTreeCheckpointer"):
            restored = load_run("/tmp/dno-nested-loader-test", checkpoint="latest")
        actual = build_predict_gxi_batched(restored)(inputs[..., 0], inputs[..., 1], depth)
        expected = prediction[..., 0].astype(jnp.float64)
        np.testing.assert_allclose(actual, expected - expected.mean(axis=-1, keepdims=True),
                                   rtol=2e-6, atol=1e-7)
        self.assertEqual(restored.model.nested_blocks, 1)
        self.assertTrue(restored.model.nested_surface_modulation)
        np.testing.assert_array_equal(restored.params["nested_surface_gain"], params["nested_surface_gain"])
        self.assertEqual(restored.model.baseline_order, 1)
        self.assertEqual(restored.epoch, 40)

    def test_terms_checkpoint_restores_active_primary_ema_and_native_predictor(self) -> None:
        surface_scales = (0.1, 0.02, 0.003, 1.0, 10.0, 0.3, 0.1)
        model = CraigSulemTermsResidual(
            width=8, n_blocks=1, nested_blocks=0, latent=2, mult_hidden=4,
            surface_context_width=4, surface_feature_scales=surface_scales,
            domain_length=4 * np.pi, eta_scale=0.5, xi_scale=0.25, target_scale=0.75,
        )
        x = 2 * jnp.pi * jnp.arange(32, dtype=jnp.float32) / 32
        inputs = jnp.stack((
            jnp.stack((0.1 * jnp.cos(x), 0.08 * jnp.sin(2 * x)), axis=0),
            jnp.stack((0.2 * jnp.sin(3 * x), 0.15 * jnp.cos(x)), axis=0),
        ), axis=-1)
        depth = jnp.log(jnp.asarray([[0.2], [8.0]], dtype=jnp.float64))
        model_depth = depth.astype(jnp.float32)
        raw_params = model.init(jax.random.PRNGKey(29), inputs, model_depth)["params"]
        kernel = cast(jax.Array, raw_params["terms_head"]["kernel"])
        ema_params = {**raw_params, "terms_head": {"kernel": jnp.linspace(
            -0.1, 0.2, kernel.size, dtype=jnp.float32,
        ).reshape(kernel.shape)}}
        expected_components = cast(tuple[jax.Array, jax.Array, jax.Array, jax.Array], model.apply(
            {"params": ema_params}, inputs, model_depth, return_baseline=True,
        ))
        prediction, baseline, m1, correction = expected_components
        np.testing.assert_array_equal(prediction, baseline + correction)
        self.assertGreater(float(jnp.linalg.norm(correction)), 1e-7)
        self.assertGreater(float(jnp.linalg.norm(baseline - m1)), 1e-7)
        config = {
            "model": "cs_dno", "cs_terms_residual": True, "norm": "scale",
            "width": 8, "n_blocks": 1, "cs_nested_blocks": 0, "latent": 2,
            "cs_n_polys": 3, "cs_mult_hidden": 4, "cs_surface_context_width": 4,
            "cs_surface_feature_scales": list(surface_scales),
            "cs_use_first_deriv": True, "cs_use_second_deriv": True,
            "cs_use_half_deriv": True, "cs_use_hilbert": True,
            "cs_baseline_order": 3, "cs_baseline_true_depth": True,
            "domain_length": 4 * np.pi, "eta_scale": 0.5, "xi_scale": 0.25, "target_scale": 0.75,
        }
        stats = {"feature_absmax": [0.5, 0.25], "target_absmax": 0.75}
        physical = inputs.astype(jnp.float64) * jnp.asarray(stats["feature_absmax"])
        reference = build_predict_gxi_batched(LoadedRun(model, ema_params, config, stats, "scale", 17))
        selections: tuple[Literal["weight_ema", "best_weight_ema"], ...] = ("weight_ema", "best_weight_ema")
        for selection in selections:
            checkpoint = ({"params": ema_params} if selection == "weight_ema" else
                          {"params": raw_params, "weight_ema_params": ema_params})
            with self.subTest(selection=selection), patch(
                "solver.evals.model_rollout.Path.read_text", side_effect=[
                    json.dumps(config), json.dumps({"epoch": 17, "stats": stats}),
                ],
            ), patch(
                "solver.evals.model_rollout.checkpoints.restore_checkpoint", return_value=checkpoint,
            ) as restore, patch("solver.evals.model_rollout.ocp.PyTreeCheckpointer"):
                loaded = load_run("/tmp/dno-terms-loader-test", checkpoint=selection)
            self.assertIsInstance(loaded.model, CraigSulemTermsResidual)
            self.assertEqual(loaded.model.baseline_order, 3)
            self.assertTrue(loaded.model.baseline_true_depth)
            self.assertEqual(loaded.model.nested_blocks, 0)
            self.assertEqual(loaded.epoch, 17)
            self.assertEqual(restore.call_args.kwargs["step"], 17)
            self.assertEqual(restore.call_args.kwargs["ckpt_dir"].name,
                             "weight_ema_ckpt" if selection == "weight_ema" else "best_val_ckpt")
            np.testing.assert_array_equal(loaded.params["terms_head"]["kernel"], ema_params["terms_head"]["kernel"])
            self.assertFalse(any(name.startswith("cs_block_") for name in loaded.params))
            actual_components = cast(tuple[jax.Array, jax.Array, jax.Array, jax.Array], loaded.model.apply(
                {"params": loaded.params}, inputs, model_depth, return_baseline=True,
            ))
            for actual, expected in zip(actual_components, expected_components, strict=True):
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(
                loaded.model.apply({"params": loaded.params}, inputs, model_depth), prediction,
            )
            actual = build_predict_gxi_batched(loaded)(physical[..., 0], physical[..., 1], depth)
            np.testing.assert_array_equal(actual, reference(physical[..., 0], physical[..., 1], depth))
            expected = prediction[..., 0] * stats["target_absmax"]
            np.testing.assert_allclose(actual, expected - expected.mean(axis=-1, keepdims=True),
                                       rtol=2e-6, atol=1e-8)

    def test_cached_fourier_multiplier_skip_matches_nonzero_native_branches(self) -> None:
        from scripts.benchmark_dno_fusion import build_variant

        nx = 64
        inputs = 0.1 * jax.random.normal(jax.random.PRNGKey(14), (2, nx, 2), dtype=jnp.float32)
        depth = jnp.log(jnp.asarray([[0.2], [8.0]], dtype=jnp.float32))
        stats = {"feature_absmax": [1.0, 1.0], "target_absmax": 1.0}
        # Use the identical sandwich algebra on CPU; only the native FFT binding is replaced.
        with patch("scripts.benchmark_dno_fusion.sandwich", side_effect=lambda eta, xi, symbols, **kwargs:
                   symbols * jnp.fft.rfft(eta[:, None, :] * jnp.fft.irfft(
                       symbols * xi[:, None, :], n=nx), n=nx)):
            cases = [(nested, log_inputs, whiten, False, whiten, 4 if whiten else 0, 2, 0.0)
                     for nested, log_inputs, whiten in ((0, False, False), (1, False, False), (0, True, False),
                                                        (1, True, False), (0, False, True), (1, False, True))]
            cases += [(0, False, False, True, polynomial, context, batch, 0.0)
                      for polynomial, context, batch in product((False, True), (0, 4), (1, 2))]
            cases += [(0, False, False, False, False, 4, batch, 22.727203594311888) for batch in (1, 2)]
            for nested, log_inputs, whiten, depth_gate, polynomial, context, batch, cutoff in cases:
                with self.subTest(nested=nested, log_inputs=log_inputs, whiten=whiten,
                                  depth_gate=depth_gate, polynomial=polynomial, context=context, batch=batch,
                                  curvature_cutoff=cutoff):
                    fields, log_depth = inputs[:batch], depth[:batch]
                    model = CraigSulemDNO(width=8, n_blocks=2, nested_blocks=nested, latent=2,
                                         mult_hidden=4, multiplier_skip=True, baseline_true_depth=True,
                                         curvature_cutoff=cutoff, domain_length=(3 if cutoff else 2) * np.pi,
                                         log_multiplier_inputs=log_inputs,
                                         surface_feature_transform=tuple(map(tuple, (np.eye(7) + 0.1).tolist())) if whiten else None,
                                         surface_feature_scales=(0.2, 0.04, 0.008, 2., 8., 0.7, 0.3) if whiten or cutoff else None,
                                         surface_context_width=context,
                                         surface_context_gate=depth_gate and bool(context),
                                         surface_depth_gate=depth_gate, surface_polynomial_skip=polynomial)
                    params = dict(model.init(jax.random.PRNGKey(0), fields, log_depth)["params"])
                    if depth_gate:
                        shape = cast(jax.Array, params["eta_depth_gate"]["kernel"]).shape
                        params["eta_depth_gate"]["kernel"] = jnp.linspace(
                            -0.6, 0.7, int(np.prod(shape)), dtype=jnp.float32,
                        ).reshape(shape)
                    for index in range(2):
                        block = params[f"cs_block_{index}"]
                        for name in ("m_shared", "m_inner") if index >= 2 - nested else ("m_shared",):
                            block[name]["symbol_skip"]["kernel"] = jnp.asarray(
                                [[0.4, -0.3], [1.0, 0.7], [-0.2, 0.5]], dtype=jnp.float32)
                        if index >= 2 - nested:
                            block["gamma"] = jnp.full((2,), 0.2, dtype=jnp.float32)
                        else:
                            block["phi_proj"]["kernel"] = jnp.full_like(cast(jax.Array, block["phi_proj"]["kernel"]), 0.1)
                    loaded = LoadedRun(model, params, {"model": "cs_dno"}, stats, "scale", 1)
                    expected = build_predict_gxi_batched(loaded)(fields[..., 0], fields[..., 1], log_depth)
                    transformed = build_variant(loaded, log_depth, nx, "fourier_round")
                    actual = jnp.fft.irfft(transformed(
                        jnp.fft.rfft(fields[..., 0].astype(jnp.float64)),
                        jnp.fft.rfft(fields[..., 1].astype(jnp.float64)),
                    ), n=nx)
                    relative = jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)
                    self.assertLess(float(relative), 5e-7)
                    if (whiten or depth_gate) and not nested:
                        for variant in ("cached", "surface", "front"):
                            physical = build_variant(loaded, log_depth, nx, variant)(fields[..., 0], fields[..., 1])
                            error = physical - expected
                            # FP32 regrouping near output zeros is measured against the field scale.
                            self.assertLess(float(jnp.linalg.norm(error) / jnp.linalg.norm(expected)), 5e-7)
                            self.assertLess(float(jnp.max(jnp.abs(error)) / jnp.sqrt(jnp.mean(expected**2))), 1e-6)

    def test_nested_fused_fourier_path_preserves_signed_nested_contribution(self) -> None:
        from contextlib import nullcontext
        from scripts import benchmark_dno_fusion as fusion

        nx = 1024

        def fft_cpu(a: jax.Array, n: int, *, inverse: bool = False,
                    b: jax.Array | None = None, outer: jax.Array | None = None,
                    channels_last: bool = False, **_: Any) -> jax.Array:
            if b is not None:
                a = (jnp.moveaxis(a, -1, -2) if channels_last else a) * b
            result = jnp.fft.irfft(a, n=n) if inverse else jnp.fft.rfft(a, n=n)
            return result if outer is None else result * outer

        def fields_cpu(inputs_hat: jax.Array, operators: jax.Array, filters: jax.Array,
                       *, n: int, normalized_inputs: jax.Array, **_: Any) -> tuple[jax.Array, jax.Array]:
            eta = normalized_inputs[:, 0]
            features = jnp.concatenate((jnp.stack((eta, eta**2, eta**3), axis=1),
                                        jnp.fft.irfft(operators * inputs_hat[:, :1], n=n)), axis=1)
            return features, jnp.fft.irfft(filters * inputs_hat[:, 1:2], n=n)

        def sandwich_cpu(eta: jax.Array, xi: jax.Array, symbols: jax.Array, **_: Any) -> jax.Array:
            return symbols * jnp.fft.rfft(
                eta[:, None, :] * jnp.fft.irfft(symbols * xi[:, None, :], n=nx), n=nx,
            )

        model = CraigSulemDNO(width=8, n_blocks=2, nested_blocks=1, latent=2, mult_hidden=4,
                             eta_scale=0.3, xi_scale=0.7, target_scale=0.8,
                             nested_surface_modulation=True,
                             surface_feature_scales=(0.2, 0.04, 0.008, 2., 8., 0.7, 0.3),
                             surface_context_width=4)
        params = cast(dict[str, Any], model.init(
            jax.random.PRNGKey(83), jnp.zeros((1, nx, 2), jnp.float32),
            jnp.zeros((1, 1), jnp.float32))["params"])
        params["nested_surface_gain"] = jnp.asarray(0.6, jnp.float32)
        params["cs_block_0"]["phi_proj"]["kernel"] = jnp.full_like(
            params["cs_block_0"]["phi_proj"]["kernel"], 0.1)
        block = params["cs_block_1"]
        block["gamma"] = jnp.asarray([0.13, -0.19], jnp.float32)
        for name, bias in (("m_shared", [0.6, -0.45]), ("m_inner", [-0.4, 0.7])):
            block[name]["out"]["kernel"] *= 0.03
            block[name]["out"]["bias"] = jnp.asarray(bias, jnp.float32)
        stats = {"feature_absmax": [0.3, 0.7], "target_absmax": 1.1}
        name = "fourier_fused_features_product_overlap"
        x = 2 * jnp.pi * jnp.arange(nx, dtype=jnp.float64) / nx
        with patch.object(fusion, "fft", side_effect=fft_cpu), patch.object(
            fusion, "field_irfft", side_effect=fields_cpu,
        ), patch.object(fusion, "sandwich", side_effect=sandwich_cpu), patch.object(
            fusion, "compute_on", return_value=nullcontext(),
        ):
            for surface_features, batch in product((False, True), (1, 32)):
                with self.subTest(surface_features=surface_features, batch=batch):
                    case_model = model.clone(nested_surface_features=surface_features,
                                             nested_surface_modulation=not surface_features)
                    case_block = {**block, "surface_mix": jnp.asarray([[0.17, -0.11], [-0.09, 0.23]], jnp.float32)} if surface_features else block
                    case_params = {**params, "cs_block_1": case_block}
                    zero_params = {**case_params, "cs_block_1": {**case_block, "gamma": jnp.zeros_like(block["gamma"])}}
                    loaded = LoadedRun(case_model, case_params, {"model": "cs_dno"}, stats, "scale", 1)
                    zero = loaded._replace(params=zero_params)
                    phase = jnp.arange(batch, dtype=jnp.float64)[:, None] / 7
                    eta = 0.015 + 0.04 * jnp.cos(2 * x + phase) + 0.01 * jnp.sin(5 * x - phase)
                    xi = 0.2 * jnp.sin(x + phase) - 0.05 * jnp.cos(3 * x - phase)
                    depth = jnp.log(jnp.linspace(0.08, 8., batch, dtype=jnp.float32))[:, None]
                    forward = fusion.build_variant(loaded, depth, nx, name)
                    fused = fusion.build_variant(loaded, depth, nx, name + "_nestedfused")
                    context_fused = fusion.build_variant(loaded, depth, nx, name + "_nestedfused_contextbatch")
                    without = fusion.build_variant(zero, depth, nx, name)
                    fused_without = fusion.build_variant(zero, depth, nx, name + "_nestedfused")
                    context_without = fusion.build_variant(zero, depth, nx, name + "_nestedfused_contextbatch")
                    eta_hat, xi_hat = jnp.fft.rfft(eta), jnp.fft.rfft(xi)
                    expected, actual = forward(eta_hat, xi_hat), fused(eta_hat, xi_hat)
                    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-7)
                    np.testing.assert_allclose(context_fused(eta_hat, xi_hat), expected, rtol=3e-6, atol=2e-7)
                    native = build_predict_gxi_batched(loaded)(eta, xi, depth)
                    self.assertLess(float(jnp.linalg.norm(jnp.fft.irfft(expected, n=nx) - native)
                                          / jnp.linalg.norm(native)), 3e-6)
                    multiplier = fusion.DepthAwareMultiplier(2, model.domain_length, model.h_clip_max, 4)
                    outer, inner = (jnp.moveaxis(cast(jax.Array, multiplier.apply(
                        {"params": block[key]}, depth, nx // 2 + 1)), -1, 1)
                        for key in ("m_shared", "m_inner"))
                    eta_norm = jax.jit(lambda value: jnp.fft.irfft(value, n=nx).astype(jnp.float32)
                                       / np.float32(0.3))(eta_hat)
                    surface = (eta_norm / np.float32(0.2))[:, None, :]
                    if surface_features:
                        k = jnp.arange(nx // 2 + 1, dtype=jnp.float32)
                        raw_features = jnp.stack((
                            jnp.fft.irfft(jnp.sqrt(k) * jnp.fft.rfft(eta_norm), n=nx) / np.float32(0.7),
                            jnp.fft.irfft(-k**2 * jnp.fft.rfft(eta_norm), n=nx) / np.float32(8.),
                        ), axis=-1)
                        surface = surface + jnp.moveaxis(raw_features @ case_block["surface_mix"], -1, 1)
                    else:
                        surface = surface * (1 + jnp.tanh(params["nested_surface_gain"])
                                             * jnp.tanh(eta_norm * model.eta_scale / jnp.exp(depth)))[:, None, :]
                    potential = (jnp.fft.irfft(xi_hat, n=nx).astype(jnp.float32)
                                 / np.float32(0.7)) * model.xi_scale
                    middle = jnp.fft.irfft(inner * jnp.fft.rfft(
                        surface * jnp.fft.irfft(outer * jnp.fft.rfft(potential)[:, None], n=nx)), n=nx)
                    nested = jnp.sum(outer * jnp.fft.rfft(block["gamma"][None, :, None] * surface * middle), axis=1)
                    nested = (nested / 2).astype(jnp.complex128) * (1.1 / model.target_scale)
                    nested = nested.at[:, 0].set(0).at[:, -1].set(nested[:, -1].real)
                    self.assertGreater(float(jnp.linalg.norm(nested)), 1e-3)
                    for result in (expected - without(eta_hat, xi_hat), actual - fused_without(eta_hat, xi_hat),
                                   context_fused(eta_hat, xi_hat) - context_without(eta_hat, xi_hat)):
                        self.assertLess(float(jnp.linalg.norm(result - nested) / jnp.linalg.norm(nested)), 3e-6)
                    stages_eta = jnp.stack((eta_hat, eta_hat))
                    stages_xi = jnp.stack((xi_hat, 1.3 * xi_hat))
                    stages = jax.jit(jax.vmap(fused))(stages_eta, stages_xi)
                    np.testing.assert_allclose(stages, jax.jit(jax.vmap(forward))(stages_eta, stages_xi),
                                               rtol=3e-6, atol=2e-7)
                    np.testing.assert_allclose(stages, jax.jit(jax.vmap(context_fused))(stages_eta, stages_xi),
                                               rtol=3e-6, atol=2e-7)
                    isolated = stages - jax.jit(jax.vmap(fused_without))(stages_eta, stages_xi)
                    reference = jnp.stack((nested, 1.3 * nested))
                    self.assertLess(float(jnp.linalg.norm(isolated - reference) / jnp.linalg.norm(reference)), 3e-6)
                    if surface_features:
                        np.testing.assert_array_equal(
                            context_fused(jnp.zeros_like(eta_hat), xi_hat),
                            context_without(jnp.zeros_like(eta_hat), xi_hat),
                        )
                        other_xi = 0.7 * xi + 0.1 * jnp.cos(4 * x - phase)
                        other_hat = jnp.fft.rfft(other_xi)
                        other_nested = context_fused(eta_hat, other_hat) - context_without(eta_hat, other_hat)
                        left = jnp.sum(other_xi * jnp.fft.irfft(nested, n=nx))
                        right = jnp.sum(xi * jnp.fft.irfft(other_nested, n=nx))
                        np.testing.assert_allclose(left, right, rtol=3e-6, atol=2e-7)

    def test_exact_solver_g0_uses_original_spectrum_without_rescaling_other_terms(self) -> None:
        from scripts.benchmark_dno_fusion import build_variant
        from solver.solvers import time_integrator as ti

        nx = 32
        x = 2 * jnp.pi * jnp.arange(nx, dtype=jnp.float64) / nx
        eta = jnp.broadcast_to(0.013 + 0.01 * jnp.cos(x), (2, nx))
        xi = jnp.stack((0.2 * jnp.sin(x) + 0.1 * jnp.cos(3 * x),
                        0.13 * jnp.cos(x) - 0.07 * jnp.sin(2 * x))) + 0.123456789
        xi_hat = jnp.fft.rfft(xi).at[..., -1].set(1e-8 + 2e-8j)
        depths = jnp.asarray([0.137123456789, 7.0000031415926], dtype=jnp.float64)
        model = CraigSulemDNO(width=8, n_blocks=1, latent=2, mult_hidden=4,
                             eta_scale=0.3, xi_scale=0.7, target_scale=0.8,
                             baseline_true_depth=True)
        params = model.init(jax.random.PRNGKey(7), jnp.zeros((1, nx, 2), jnp.float32),
                            jnp.zeros((1, 1), jnp.float32))["params"]
        params["cs_block_0"]["phi_proj"]["kernel"] = jnp.full_like(
            params["cs_block_0"]["phi_proj"]["kernel"], 0.2)
        stats = {"feature_absmax": [0.3, 0.7], "target_absmax": 2.5}
        loaded = LoadedRun(model, params, {"model": "cs_dno"}, stats, "scale", 1)
        variant = "fourier_round_exactg0"
        with patch("scripts.benchmark_dno_fusion.sandwich", side_effect=lambda e, q, symbols, **kwargs:
                   symbols * jnp.fft.rfft(e[:, None, :] * jnp.fft.irfft(
                       symbols * q[:, None, :], n=nx), n=nx)):
            for batch in (1, 2):
                with self.subTest(batch=batch):
                    log_depth = jnp.log(depths[:batch])
                    symbol = ti.make_solver_params(nx, 2 * np.pi, depths[:batch, None]).g0[..., :nx // 2 + 1]
                    forward = build_variant(loaded, log_depth, nx, variant, solver_g0=symbol)
                    nonlinear = build_variant(loaded, log_depth, nx, variant, solver_g0=jnp.zeros_like(symbol))
                    ehat, qhat = jnp.fft.rfft(eta[:batch]), xi_hat[:batch]
                    exact = (symbol * qhat).at[..., 0].set(0).at[..., -1].set((symbol * qhat)[..., -1].real)
                    np.testing.assert_allclose(forward(jnp.zeros_like(ehat), qhat), exact, rtol=2e-15, atol=2e-15)
                    actual, remainder = forward(ehat, qhat), nonlinear(ehat, qhat)
                    np.testing.assert_allclose(actual - remainder, exact, rtol=2e-14, atol=2e-14)
                    zero_head = jax.tree.map(jnp.zeros_like, params["cs_block_0"]["phi_proj"])
                    baseline = loaded._replace(params={**params, "cs_block_0": {
                        **params["cs_block_0"], "phi_proj": zero_head}})
                    baseline_forward = build_variant(baseline, log_depth, nx, variant, solver_g0=symbol)
                    self.assertGreater(float(jnp.linalg.norm(actual - baseline_forward(ehat, qhat))), 1e-8)
                    np.testing.assert_array_equal(actual[..., 0], 0)
                    np.testing.assert_array_equal(actual[..., -1].imag, 0)
            for candidate, name, kwargs in (
                (loaded, variant, {}),
                (loaded, "original_exactg0", {"solver_g0": symbol}),
                (loaded, variant, {"solver_g0": symbol, "include_baselines": False}),
                (loaded._replace(model=model.clone(baseline_order=2)), variant, {"solver_g0": symbol}),
            ):
                with self.subTest(invalid=name, kwargs=kwargs), self.assertRaises(ValueError):
                    build_variant(candidate, log_depth, nx, name, **kwargs)

    def test_spectral_rollout_projects_batched_momentum_after_each_substep(self) -> None:
        from scripts.benchmark_surrogate_rhs import rollout_if_trial
        from solver.solvers import time_integrator as ti

        for nx in (32, 64):
            with self.subTest(nx=nx):
                params = ti.make_solver_params(
                    nx, 2 * np.pi, jnp.asarray([[0.7], [1.4]]), filter_fraction=2 / 3,
                )
                x = 2 * jnp.pi * jnp.arange(nx, dtype=jnp.float64) / nx
                initial = ti.State(
                    eta=jnp.stack((0.03 + 0.12 * jnp.cos(x) + 0.04 * jnp.sin(2 * x),
                                   -0.02 + 0.09 * jnp.sin(x) + 0.03 * jnp.cos(3 * x))),
                    xi=jnp.stack((0.2 + 0.18 * jnp.sin(x) + 0.05 * jnp.cos(2 * x),
                                  -0.1 + 0.14 * jnp.cos(x) + 0.04 * jnp.sin(3 * x))),
                )

                def predict_hat(eta_hat: jax.Array, xi_hat: jax.Array) -> jax.Array:
                    return 0.7 * params.g0[..., :nx // 2 + 1] * xi_hat

                run = partial(rollout_if_trial, initial, params=params,
                              carry_spectral=True, local_if=True, predict_hat=predict_hat)
                times = jnp.arange(9, dtype=jnp.float64) / 32
                unprojected = run(times, substeps=1)
                projected = run(times, substeps=1, project_momentum=True)
                coarse = run(times[::2], substeps=2, project_momentum=True)
                momenta = [
                    np.mean(np.asarray(trajectory["eta"]) * np.fft.irfft(
                        1j * np.fft.rfftfreq(nx, d=1 / nx)
                        * np.fft.rfft(np.asarray(trajectory["xi"])), n=nx,
                    ), axis=-1)
                    for trajectory in (unprojected, projected)
                ]
                self.assertTrue(np.all(np.max(np.abs(momenta[0] - momenta[0][0]), axis=0) > 1e-7))
                np.testing.assert_allclose(
                    momenta[1], np.broadcast_to(momenta[1][0], momenta[1].shape),
                    rtol=0, atol=2e-14,
                )
                np.testing.assert_array_equal(projected["eta"][1], unprojected["eta"][1])
                for field in ("eta", "xi", "gxi"):
                    np.testing.assert_allclose(coarse[field], projected[field][::2], rtol=0, atol=2e-14)
                for trajectory in (unprojected, projected, coarse):
                    np.testing.assert_allclose(
                        np.mean(trajectory["eta"], axis=-1),
                        np.broadcast_to(np.mean(initial.eta, axis=-1), trajectory["eta"].shape[:-1]),
                        rtol=0, atol=2e-14,
                    )
                    np.testing.assert_allclose(np.mean(trajectory["xi"], axis=-1), 0, rtol=0, atol=2e-14)
                for carry_spectral, half_spectrum in ((False, False), (True, True)):
                    with self.subTest(carry_spectral=carry_spectral, half_spectrum=half_spectrum):
                        with self.assertRaisesRegex(ValueError, "requires full spectral carry"):
                            run(times, substeps=1, project_momentum=True,
                                carry_spectral=carry_spectral, half_spectrum=half_spectrum)

    def test_fast_evaluator_uses_candidate_ics_reference_truth_and_exact_spectral_execution(self) -> None:
        from scripts import eval_fast_rollouts as fast

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "candidate"
            reference = Path(temporary) / "reference"
            source = reference / "eval_epoch1_weight_ema_test_stratified_n32_gpu0"
            source.mkdir(parents=True)
            fields = np.broadcast_to(np.arange(32)[None, :, None], (251, 32, 8)).astype(np.float64)
            for family, cfg in FAMILY_CONFIGS.items():
                np.savez(source / f"{family}_trajs.npz", pred_eta=fields - 1000,
                         pred_xi=fields - 2000, truth_eta=fields - 3000,
                         truth_xi=fields - 4000, depths=np.full(32, 99.0),
                         times=np.arange(251) * cfg.dt * 2)
            ics = [IC(fields[0, i], fields[0, i] + 1, 0.1 + i / 32, i + 17)
                   for i in range(32)]
            loaded = LoadedRun(None, {}, {"dataset": "relative/arrays", "domain_length": 2 * np.pi},
                               {}, "scale", 1)
            execute = Mock(return_value={name: fields for name in ("eta", "xi", "gxi")})
            compiled = Mock()
            compiled.lower.return_value.compile.return_value = execute
            solver_g0 = jnp.arange(32 * 8, dtype=jnp.float64).reshape(32, 8)
            original_boundary = fast.ti._state_to_hat
            with patch.object(
                fast, "load_run", return_value=loaded,
            ) as load, patch.object(
                fast, "_load_paper_dataset_ics", return_value=(ics, {}, 8, 2 * np.pi),
            ) as load_ics, patch.object(fast, "build_variant") as build, patch.object(
                fast.jax, "jit", return_value=compiled,
            ) as jit, patch.object(fast.jax, "block_until_ready"), patch.object(
                fast.jax, "clear_caches",
            ), patch.object(
                fast.ti, "make_solver_params", return_value=SimpleNamespace(nx=8, g0=solver_g0),
            ) as make_params, patch.object(
                fast, "run_family", return_value={},
            ) as evaluate, patch.object(fast, "compute_macro_summary", return_value={}), patch.object(
                fast, "_write_json",
            ):
                for reference_args, offset, suffix, exact_g0, fuse_nested, selection, checkpoint_dir in (
                    (["--reference-run", str(reference)], 0.5, "", True, True, "best_weight_ema", directory / "rollout_best_epoch37_ckpt"),
                    (["--reference-run", str(reference)], 0.5, "", True, True, "weight_ema", None),
                    (["--reference-run", str(reference)], 0.5, "", True, False, "weight_ema", None),
                    (["--reference-run", str(reference)], 0.5, "", False, False, "weight_ema", None),
                    (["--reference-dir", str(source), "--ic-quantile-offset", "0.25"], 0.25, "_q0.25", False, False, "weight_ema", None),
                    (["--reference-dir", str(source), "--ic-quantile-offset", "0.25"], 0.25, "_q0.25", False, False, "weight_ema_secondary", None),
                ):
                    with self.subTest(offset=offset, selection=selection), patch("sys.argv", [
                        "eval_fast_rollouts", "--run-dir", str(directory), "--project-momentum", *reference_args,
                        *(["--exact-solver-g0"] if exact_g0 else []),
                        *(["--fuse-nested-fft"] if fuse_nested else []),
                        *(["--checkpoint-dir", str(checkpoint_dir), "--batch-context-fft"] if checkpoint_dir else []),
                        *(["--checkpoint", selection] if selection != "weight_ema" else []),
                    ]):
                        for mocked in (load, load_ics, build, jit, make_params, execute, evaluate):
                            mocked.reset_mock()
                        load.return_value = loaded._replace(epoch=37) if checkpoint_dir else loaded
                        fast.main()
                        load.assert_called_once_with(directory, checkpoint=selection, checkpoint_dir=checkpoint_dir)
                        for call in load_ics.call_args_list:
                            self.assertEqual(call.kwargs, {"ic_quantile_offset": offset})
                        for call in evaluate.call_args_list:
                            self.assertEqual(call.args[4], directory / (
                                f"eval_fast_{selection}{suffix}_momentum" + ("_exactg0" if exact_g0 else "")
                                + ("_nestedfused" if fuse_nested else "")
                                + ("_contextbatch" if checkpoint_dir else "")))
                            self.assertEqual(call.args[8]["selection"], selection)
                            self.assertEqual(call.args[8]["epoch"], 37 if checkpoint_dir else 1)
                            self.assertEqual(call.args[8]["path"], str(checkpoint_dir or directory / f"{selection}_ckpt"))
                            self.assertEqual(call.args[7], source)
                            self.assertEqual(call.kwargs["ic_quantile_offset"], offset)
                            self.assertEqual("exactg0" in call.args[8]["implementation"], exact_g0)
                            self.assertEqual("nestedfused" in call.args[8]["implementation"], fuse_nested)
                        for call in build.call_args_list:
                            self.assertEqual(call.args[3], "fourier_fused_features_product_overlap"
                                             + ("_exactg0" if exact_g0 else "")
                                             + ("_nestedfused" if fuse_nested else "")
                                             + ("_contextbatch" if checkpoint_dir else ""))
                            if exact_g0:
                                self.assertEqual(set(call.kwargs), {"fft_ept", "solver_g0"})
                                np.testing.assert_array_equal(call.kwargs["solver_g0"], solver_g0[..., :5])
                            else:
                                self.assertEqual(call.kwargs, {"fft_ept": 8, "solver_g0": None})
            load.assert_called_once_with(directory, checkpoint="weight_ema_secondary", checkpoint_dir=None)
            self.assertFalse(any(directory.glob("eval_epoch*_test_*")))
            self.assertIs(fast.ti._state_to_hat, original_boundary)
            self.assertEqual(load_ics.call_count, 4)
            for call, family in zip(load_ics.call_args_list, FAMILY_CONFIGS, strict=True):
                self.assertEqual(call.args, (fast.ROOT / "relative/arrays", family, 32, "stratified"))
                self.assertEqual(call.kwargs, {"ic_quantile_offset": 0.25})
            self.assertEqual(make_params.call_count, 4)
            for call, cfg in zip(make_params.call_args_list, FAMILY_CONFIGS.values(), strict=True):
                self.assertEqual(call.args[:2], (8, 2 * np.pi))
                np.testing.assert_array_equal(call.args[2], np.asarray([ic.depth for ic in ics])[:, None])
                self.assertEqual(call.kwargs, {"dno_order": 6, "pad_factor": 8,
                                              "filter_fraction": cfg.filter_fraction})
            self.assertEqual(build.call_count, 4)
            for call in build.call_args_list:
                self.assertIs(call.args[0], loaded)
                np.testing.assert_allclose(call.args[1], np.log([ic.depth for ic in ics]), rtol=1e-15)
                self.assertEqual(call.args[2:], (8, "fourier_fused_features_product_overlap"))
                self.assertEqual(call.kwargs, {"fft_ept": 8, "solver_g0": None})
            rollouts = [call.args[0] for call in jit.call_args_list
                        if getattr(call.args[0], "func", None) is fast.rollout_if_trial]
            self.assertEqual(len(rollouts), 4)
            for function, cfg in zip(rollouts, FAMILY_CONFIGS.values(), strict=True):
                self.assertIs(function.func, fast.rollout_if_trial)
                self.assertTrue(function.keywords["carry_spectral"])
                self.assertTrue(function.keywords["local_if"])
                self.assertTrue(function.keywords["project_momentum"])
                self.assertEqual(function.keywords["substeps"], cfg.substeps)
                self.assertIs(function.keywords["predict_hat"], build.return_value)
                self.assertIs(function.keywords["predict_gxi"], compiled)
            for call, cfg in zip(execute.call_args_list, FAMILY_CONFIGS.values(), strict=True):
                np.testing.assert_array_equal(call.args[0].eta, fields[0])
                np.testing.assert_array_equal(call.args[0].xi, fields[0] + 1)
                self.assertEqual(call.args[0].xi.dtype, jnp.float64)
                np.testing.assert_array_equal(call.args[1], np.arange(251) * cfg.dt)
            self.assertEqual(evaluate.call_count, 4)
            for call in evaluate.call_args_list:
                self.assertEqual(call.args[4], directory / "eval_fast_weight_ema_secondary_q0.25_momentum")
                self.assertEqual(call.args[5:8], (fast.ROOT / "relative/arrays", 32, source))
                self.assertEqual(call.args[8]["selection"], "weight_ema_secondary")
                self.assertTrue(call.args[8]["project_momentum"])
                self.assertIn("spectral carry", call.args[8]["implementation"])
                self.assertEqual(call.args[9], 32)
                self.assertEqual(call.kwargs["ic_selection"], "stratified")
                self.assertEqual(call.kwargs["ic_quantile_offset"], 0.25)
                self.assertEqual(call.kwargs["truth_batch_size"], 8)
                for name in ("eta", "xi", "gxi"):
                    np.testing.assert_array_equal(call.kwargs["pred"][name], fields)

    def test_precomputed_predictions_preserve_outputs_and_failure_metrics(self) -> None:
        values = np.ones((2, 2, 8), dtype=np.float64)
        truth = {"eta": values, "xi": values * 0, "gxi": values * 0, "wall_s": 1.0}
        pred = {**truth, "eta": values.copy()}
        pred["eta"][-1, 1, 0] = np.nan
        ics = [IC(values[0, i], values[0, i] * 0, 1.0, i + 17) for i in range(2)]
        loaded = LoadedRun(None, {}, {}, {}, "scale", 40)
        with tempfile.TemporaryDirectory() as temporary, patch(
            "solver.evals.eval_suite._load_paper_dataset_ics",
            return_value=(ics, {}, 8, 2 * np.pi),
        ), patch(
            "solver.evals.eval_suite._try_load_cached_truth", return_value=truth,
        ), patch(
            "solver.evals.eval_suite._rollout_ic_chunks", return_value=pred,
        ) as rollout:
            directory = Path(temporary)
            summaries, saved = [], []
            for reuse in (False, True):
                summaries.append(run_family(
                    "tanaka", FamilyConfig(2, 1.0, 1.0, 1), loaded, Mock(),
                    directory, directory, 2, directory, {}, 2,
                    center_xi=True,
                    pred=pred if reuse else None,
                ))
                with np.load(directory / "tanaka_trajs.npz") as archive:
                    saved.append({name: archive[name].copy() for name in archive.files})
            self.assertEqual(rollout.call_count, 1)
            self.assertEqual(_json_ready(summaries[0]), _json_ready(summaries[1]))
            for name in saved[0]:
                np.testing.assert_array_equal(saved[0][name], saved[1][name])
            self.assertEqual(summaries[1]["model_nonfinite_any_count_truth_valid"], 1)
            self.assertTrue(summaries[1]["center_xi"])

    def test_center_xi_cli_passes_option_to_predictor_and_summary(self) -> None:
        loaded = LoadedRun(None, {}, {}, {}, "scale", 120)
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as temporary:
                argv = ["eval_suite", "--run_dir", temporary, "--families", "tanaka"]
                if enabled:
                    argv.extend(["--center_xi", "--truth_batch_size", "8"])
                with patch("sys.argv", argv), patch(
                    "solver.evals.eval_suite.load_run", return_value=loaded,
                ), patch(
                    "solver.evals.eval_suite.build_predict_gxi_batched",
                ) as build_predictor, patch(
                    "solver.evals.eval_suite.run_family", return_value={"center_xi": enabled},
                ) as evaluate_family, patch(
                    "solver.evals.eval_suite.compute_macro_summary", return_value={},
                ), patch("solver.evals.eval_suite._write_json") as write_json:
                    eval_main()
                build_predictor.assert_called_once_with(loaded, center_xi=enabled)
                self.assertEqual(evaluate_family.call_args.kwargs["center_xi"], enabled)
                self.assertEqual(evaluate_family.call_args.kwargs["truth_batch_size"], 8 if enabled else None)
                self.assertIs(evaluate_family.call_args.args[3], build_predictor.return_value)
                self.assertEqual(write_json.call_args_list[0].args[1], {"tanaka": {"center_xi": enabled}})

    def test_evaluation_uses_float32_network_and_float64_output(self) -> None:
        model = CraigSulemDNO(width=8, n_blocks=1, latent=2, mult_hidden=4)
        inputs = 0.01 * jax.random.normal(jax.random.PRNGKey(3), (1, 16, 2), dtype=jnp.float32)
        depth = jnp.zeros((1, 1), dtype=jnp.float32)
        params = dict(model.init(jax.random.PRNGKey(0), inputs, depth)["params"])
        params["cs_block_0"]["phi_proj"]["kernel"] = jnp.full((4, 2), 0.1, dtype=jnp.float32)
        loaded = LoadedRun(
            model, params, {}, {"feature_absmax": [1.0, 1.0], "target_absmax": 1.0},
            "scale", 0,
        )
        output = build_predict_gxi_batched(loaded)(
            inputs[..., 0].astype(jnp.float64),
            inputs[..., 1].astype(jnp.float64),
            depth.astype(jnp.float64),
        )
        self.assertEqual(output.dtype, jnp.float64)
        expected = cast(jax.Array, model.apply(
            {"params": params}, inputs, depth
        ))[..., 0].astype(jnp.float64)
        np.testing.assert_allclose(output, expected - expected.mean(axis=-1, keepdims=True), rtol=1e-12, atol=1e-14)

    def test_center_xi_removes_offsets_before_float32_rounding(self) -> None:
        model = CraigSulemDNO(width=8, n_blocks=1, latent=2, mult_hidden=4)
        nx = 32
        x = jnp.arange(nx, dtype=jnp.float64) * (2.0 * jnp.pi / nx)
        eta = jnp.stack((0.03 * jnp.cos(x), 0.02 * jnp.sin(2.0 * x)))
        xi = jnp.tile(jnp.asarray([1.0, 0.0, -1.0, 0.0]), nx // 4)[None, :]
        xi = xi * jnp.asarray([2.0**-12, 2.0**-11])[:, None]
        log_depth = jnp.log(jnp.asarray([0.1, 0.8]))
        inputs = jnp.stack((eta, xi), axis=-1).astype(jnp.float32)
        params = dict(model.init(
            jax.random.PRNGKey(2), inputs, log_depth.astype(jnp.float32)[:, None]
        )["params"])
        params["cs_block_0"]["phi_proj"]["kernel"] = jnp.full((4, 2), 0.1, dtype=jnp.float32)
        for norm, stats in (
            ("scale", {"feature_absmax": [0.25, 2.0], "target_absmax": 3.0}),
            ("minmax", {"feature_min": [-0.25, -0.5], "feature_max": [0.75, 1.5],
                        "target_min": -1.0, "target_max": 2.0}),
        ):
            with self.subTest(norm=norm):
                loaded = LoadedRun(model, params, {}, stats, norm, 0)
                default = build_predict_gxi_batched(loaded)
                disabled = build_predict_gxi_batched(loaded, center_xi=False)
                centered = build_predict_gxi_batched(loaded, center_xi=True)
                expected = centered(eta, xi, log_depth)
                np.testing.assert_array_equal(default(eta, xi, log_depth), disabled(eta, xi, log_depth))
                for offsets in ([1e-5, -1e-4], [32768.0, -65536.0]):
                    shifted = xi + jnp.asarray(offsets)[:, None]
                    output = centered(eta, shifted, log_depth)
                    np.testing.assert_array_equal(output, expected)
                    np.testing.assert_allclose(jnp.mean(output, axis=-1), 0.0, atol=1e-15)
                    self.assertEqual(output.dtype, jnp.float64)
                self.assertGreater(
                    float(jnp.linalg.norm(disabled(eta, shifted, log_depth) - expected)), 1e-7
                )

    def test_fno_spectral_layer_preserves_evaluation_precision(self) -> None:
        layer = SpectralConv1d(in_channels=2, out_channels=2, modes=4)
        inputs = jax.random.normal(jax.random.PRNGKey(1), (1, 16, 2), dtype=jnp.float32)
        variables = layer.init(jax.random.PRNGKey(2), inputs)
        for input_dtype, param_dtype in (
            (jnp.float32, jnp.float32), (jnp.float32, jnp.float64), (jnp.float64, jnp.float64),
        ):
            typed_variables = jax.tree_util.tree_map(lambda value: value.astype(param_dtype), variables)
            output = cast(jax.Array, layer.apply(typed_variables, inputs.astype(input_dtype)))
            self.assertEqual(output.dtype, input_dtype)
            self.assertTrue(np.isfinite(output).all())

    def test_dataset_ics_select_test_initial_rows_with_global_simulation_ids(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batches = []
            rows = SimulationRows(
                eta=np.arange(16, dtype=np.float64).reshape(2, 8),
                xi=np.ones((2, 8)),
                gxi=np.zeros((2, 8)),
                depth=2.0,
                time=np.asarray((0.0, 1.0)),
            )
            for index, (family, accepted) in enumerate(
                (
                    (PhysicalFamilyId.STOKES, (rows,)),
                    (PhysicalFamilyId.TANAKA, (rows,)),
                    (PhysicalFamilyId.TANAKA, (None, rows, None, rows)),
                    (PhysicalFamilyId.TANAKA, (rows,)),
                )
            ):
                batch = root / f"batch_{index}.npz"
                save_completed_batch(
                    batch,
                    ("known",) * len(accepted),
                    accepted,
                    family_id=family,
                    seed=42,
                )
                batches.append(batch)
            dataset = build_dataset(
                root / "dataset", batches, validation_fraction=0.2, test_fraction=0.6
            )
            splits = np.load(dataset / "dataset_split.npy")
            np.testing.assert_array_equal(splits[::2], splits[1::2])
            np.testing.assert_array_equal(
                np.load(dataset / "simulation_id.npy"), np.repeat(np.arange(5), 2)
            )
            expected_rows = np.flatnonzero(
                (splits == "test")
                & (np.load(dataset / "family_id.npy") == PhysicalFamilyId.TANAKA)
                & (np.load(dataset / "frame_index.npy") == 0)
            )
            ics, source, nx, length = _load_paper_dataset_ics(
                dataset, "tanaka", len(expected_rows)
            )
            self.assertEqual([ic.simulation_id for ic in ics], list(expected_rows // 2))
            self.assertEqual(
                [ic.meta["dataset_row"] for ic in ics], list(expected_rows)
            )
            self.assertEqual(source["dataset"], str(dataset))
            self.assertEqual(nx, 8)
            self.assertAlmostEqual(length, 2.0 * np.pi)
            for ic in ics:
                np.testing.assert_array_equal(ic.eta, rows.eta[0])
                np.testing.assert_array_equal(ic.xi, rows.xi[0])
            with self.assertRaisesRegex(ValueError, f"only {len(expected_rows)}"):
                _load_paper_dataset_ics(dataset, "tanaka", len(expected_rows) + 1)

    def test_stratified_ics_spread_across_parameter_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            group_ids = np.repeat([f"g{index:02d}" for index in range(10)], 3)
            arrays = {
                "eta": np.arange(30 * 8, dtype=np.float64).reshape(30, 8),
                "xi": np.zeros((30, 8)),
                "depth": np.ones(30),
                "family_id": np.full(30, PhysicalFamilyId.TANAKA),
                "dataset_split": np.full(30, "test"),
                "simulation_id": np.arange(30),
                "frame_index": np.zeros(30, dtype=np.int64),
                "parameter_group_id": group_ids,
                "x": np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False),
            }
            for name, values in arrays.items():
                np.save(dataset / f"{name}.npy", values)

            first, _, _, _ = _load_paper_dataset_ics(dataset, "tanaka", 4)
            self.assertEqual([ic.simulation_id for ic in first], [0, 1, 2, 3])

            spread, source, _, _ = _load_paper_dataset_ics(
                dataset, "tanaka", 4, "stratified"
            )
            self.assertEqual([ic.simulation_id for ic in spread], [4, 10, 19, 25])
            self.assertEqual(
                source["selected_parameter_group_ids"],
                ["g01", "g03", "g06", "g08"],
            )

            covered, source, _, _ = _load_paper_dataset_ics(
                dataset, "tanaka", 12, "stratified"
            )
            self.assertEqual(
                [ic.simulation_id for ic in covered[:10]],
                [0, 3, 7, 10, 13, 16, 19, 22, 25, 28],
            )
            self.assertEqual([ic.simulation_id for ic in covered[10:]], [2, 5])
            self.assertEqual(
                source["selected_parameter_group_ids"],
                [f"g{index:02d}" for index in range(10)] + ["g00", "g01"],
            )

            bf_groups = [
                f"n_c_{mode:02d}__delta_n_{offset:02d}"
                for mode in range(4, 21)
                for offset in range(1, 2 if mode < 6 else 4)
            ]
            n_rows = len(bf_groups) * 3
            arrays.update(
                eta=np.zeros((n_rows, 8)),
                xi=np.zeros((n_rows, 8)),
                depth=np.ones(n_rows),
                family_id=np.full(n_rows, PhysicalFamilyId.BENJAMIN_FEIR),
                dataset_split=np.full(n_rows, "test"),
                simulation_id=np.arange(n_rows),
                frame_index=np.zeros(n_rows, dtype=np.int64),
                parameter_group_id=np.repeat(bf_groups, 3),
            )
            for name, values in arrays.items():
                np.save(dataset / f"{name}.npy", values)
            bf_ics, bf_source, _, _ = _load_paper_dataset_ics(
                dataset, "benjamin_feir", 32, "stratified"
            )
            self.assertEqual(len(bf_ics), 32)
            self.assertTrue(all(ic.simulation_id % 3 == 1 for ic in bf_ics))
            selected_groups = cast(list[str], bf_source["selected_parameter_group_ids"])
            for mode in range(4, 21):
                prefix = f"n_c_{mode:02d}__delta_n_"
                selected_offsets = [
                    group.removeprefix(prefix)
                    for group in selected_groups
                    if group.startswith(prefix)
                ]
                self.assertEqual(selected_offsets, ["01"] if mode < 6 else ["01", "03"])

    def test_truth_cache_uses_simulation_ids_consistently(self) -> None:
        values = np.ones((2, 1, 3), dtype=np.float64)
        truth = {
            "eta": values,
            "xi": 2.0 * values,
            "gxi": 3.0 * values,
            "wall_s": 1.0,
        }
        protocol = '{"family":"tanaka"}'

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            path = _write_truth_cache(
                directory,
                "tanaka",
                truth,
                times=np.asarray((0.0, 1.0)),
                depths=np.asarray((2.0,)),
                simulation_ids=np.asarray((17,)),
                truth_protocol_json=protocol,
            )
            with np.load(path, allow_pickle=False) as archive:
                self.assertIn("simulation_ids", archive.files)
                self.assertNotIn("case_ids", archive.files)

            loaded = _try_load_cached_truth(
                "tanaka",
                directory,
                expected_shape=values.shape,
                expected_simulation_ids=[17],
                expected_protocol_json=protocol,
            )

        self.assertIsNotNone(loaded)
        assert loaded is not None
        np.testing.assert_array_equal(loaded["eta"], values)

    def test_truth_cache_rejects_float32_despite_float64_protocol(self) -> None:
        values = np.ones((2, 1, 3))
        protocol = '{"dtype":"float64"}'
        for filename in ("tanaka_truth_cache.npz", "tanaka_trajs.npz"):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                np.savez(
                    directory / filename,
                    truth_eta=values.astype(np.float32), truth_xi=values, truth_gxi=values,
                    simulation_ids=np.asarray([17]), truth_protocol_json=np.asarray(protocol),
                )
                self.assertIsNone(_try_load_cached_truth(
                    "tanaka", directory, expected_shape=values.shape,
                    expected_simulation_ids=[17], expected_protocol_json=protocol,
                ))

    def test_rollout_chunks_preserve_panel_order_and_sum_wall_time(self) -> None:
        ics = [
            IC(
                eta=np.zeros(3),
                xi=np.zeros(3),
                depth=1.0,
                simulation_id=simulation_id,
            )
            for simulation_id in range(5)
        ]
        calls: list[list[int]] = []

        def rollout(chunk: list[IC]) -> dict[str, np.ndarray | float]:
            simulation_ids = [ic.simulation_id for ic in chunk]
            calls.append(simulation_ids)
            values = np.broadcast_to(
                np.asarray(simulation_ids, dtype=np.float32)[None, :, None],
                (2, len(chunk), 3),
            ).copy()
            return {
                "eta": values,
                "xi": values + 10.0,
                "gxi": values + 20.0,
                "wall_s": 0.25 * len(chunk),
            }

        result = _rollout_ic_chunks(ics, 2, rollout, label="test")

        self.assertEqual(calls, [[0, 1], [2, 3], [4]])
        np.testing.assert_array_equal(
            np.asarray(result["eta"])[0, :, 0],
            np.arange(5, dtype=np.float32),
        )
        self.assertEqual(np.asarray(result["eta"]).shape, (2, 5, 3))
        self.assertEqual(float(result["wall_s"]), 1.25)

    def test_truth_and_model_chunk_sizes_preserve_all_ordered_cases_and_frames(self) -> None:
        ics = [IC(np.full(8, index + 1.0), np.zeros(8), 1.0, index + 17) for index in range(32)]
        loaded = LoadedRun(None, {}, {}, {}, "scale", 1)

        def rollout(chunk: list[IC], times: jnp.ndarray, nx: int, *_: object) -> dict[str, np.ndarray | float]:
            eta = np.broadcast_to(np.stack([ic.eta for ic in chunk]), (len(times), len(chunk), nx)).copy()
            return {"eta": eta, "xi": eta * 0, "gxi": eta * 0, "wall_s": 0.25 * len(chunk)}

        for truth_batch_size in (None, 8):
            with self.subTest(truth_batch_size=truth_batch_size), tempfile.TemporaryDirectory() as temporary, patch(
                "solver.evals.eval_suite._load_paper_dataset_ics", return_value=(ics, {}, 8, 2 * np.pi),
            ), patch(
                "solver.evals.eval_suite.truth_rollout_batched", side_effect=rollout,
            ) as truth, patch(
                "solver.evals.eval_suite.surrogate_rollout_batched", side_effect=rollout,
            ) as model:
                directory = Path(temporary)
                summary = run_family(
                    "stokes", FamilyConfig(1, 0.08, 20.0, 8), loaded, Mock(),
                    directory, directory, 32, None, {}, 32, truth_batch_size=truth_batch_size,
                )
                self.assertEqual([len(call.args[0]) for call in truth.call_args_list],
                                 [8] * 4 if truth_batch_size == 8 else [32])
                self.assertEqual([len(call.args[0]) for call in model.call_args_list], [32])
                self.assertEqual(summary["rollout_batch_size"], 32)
                with np.load(directory / "stokes_trajs.npz") as archive:
                    np.testing.assert_array_equal(archive["simulation_ids"], np.arange(17, 49))
                    np.testing.assert_array_equal(archive["times"], np.arange(251) * 0.08)
                    for field in ("eta", "xi", "gxi"):
                        values = archive[f"truth_{field}"]
                        self.assertEqual(values.shape, (251, 32, 8))
                        self.assertEqual(values.dtype, np.float64)
                        np.testing.assert_array_equal(values, archive[f"pred_{field}"])
                    np.testing.assert_array_equal(archive["truth_eta"][:, :, 0],
                                                  np.broadcast_to(np.arange(1.0, 33.0), (251, 32)))

    def test_nonfinite_predictions_are_failures_and_truth_reasons_are_explicit(
        self,
    ) -> None:
        n_t, n_ics, nx = 11, 4, 2
        times = np.arange(n_t, dtype=np.float64)
        eta = np.ones((n_t, n_ics, nx), dtype=np.float64)
        xi = np.zeros_like(eta)
        gxi = np.zeros_like(eta)

        # IC 1 has non-finite truth. IC 2 is finite but has gross truth-energy
        # drift. Only ICs 0 and 3 are therefore valid model tests.
        eta[4, 1, 0] = np.nan
        eta[1:, 2, :] = 2.0
        truth = {"eta": eta, "xi": xi, "gxi": gxi}

        pred = {field: values.copy() for field, values in truth.items()}
        pred["eta"][-1, 0, :] = 1.3
        pred["eta"][-1, 3, :] = 1.1
        pred["xi"][3, 3, 0] = np.inf

        metrics = compute_metrics(truth, pred, times, length=2.0)

        self.assertEqual(metrics["n_ics_attempted"], 4)
        self.assertEqual(metrics["n_truth_valid"], 2)
        self.assertEqual(metrics["n_truth_invalid"], 2)
        self.assertEqual(metrics["truth_valid_ics"], [0, 3])
        reasons = {
            item["ic_index"]: item["reasons"]
            for item in metrics["truth_invalid_reasons"]
        }
        self.assertEqual(reasons[1], ["nonfinite_eta"])
        self.assertEqual(reasons[2], ["energy_drift_gt_tol"])

        self.assertEqual(metrics["model_nonfinite_any_ics_truth_valid"], [3])
        self.assertEqual(metrics["model_nonfinite_any_count_truth_valid"], 1)
        self.assertEqual(metrics["model_nonfinite_any_rate_truth_valid"], 0.5)
        # IC 0 exceeds 0.25; IC 3 fails every threshold because xi became
        # non-finite, even though its terminal eta error itself is small.
        self.assertEqual(metrics["terminal_eta_failure_rate_tau_0p25"], 1.0)
        self.assertEqual(metrics["terminal_eta_failure_rate_tau_0p5"], 0.5)
        self.assertEqual(metrics["terminal_eta_failure_rate_tau_0p75"], 0.5)
        self.assertEqual(metrics["terminal_eta_failure_rate_tau_1"], 0.5)
        # Quantiles are explicitly conditional on an all-field finite rollout,
        # so only IC 0 contributes at the terminal frame.
        self.assertEqual(metrics["rel_l2_eta_n_conditional_finite_tfinal"], 1)
        self.assertAlmostEqual(
            metrics["rel_l2_eta_median_conditional_finite_tfinal"], 0.3
        )

    def test_horizon_indices_include_both_saved_endpoints(self) -> None:
        values = np.ones((11, 1, 2), dtype=np.float64)
        fields = {"eta": values, "xi": values, "gxi": values}
        metrics = compute_metrics(
            fields,
            {name: value.copy() for name, value in fields.items()},
            np.arange(11, dtype=np.float64),
            length=2.0,
        )

        self.assertEqual(metrics["metric_frame_index_t10p"], 1)
        self.assertEqual(metrics["metric_frame_index_t50p"], 5)
        self.assertEqual(metrics["metric_frame_index_tfinal"], 10)
        self.assertEqual(metrics["metric_time_t10p"], 1.0)
        self.assertEqual(metrics["metric_time_t50p"], 5.0)

    def test_predicted_hamiltonian_drift_uses_its_own_initial_value(self) -> None:
        truth_eta = np.ones((3, 1, 2), dtype=np.float64)
        pred_eta = 2.0 * truth_eta
        zeros = np.zeros_like(truth_eta)
        metrics = compute_metrics(
            {"eta": truth_eta, "xi": zeros, "gxi": zeros},
            {"eta": pred_eta, "xi": zeros, "gxi": zeros},
            np.arange(3, dtype=np.float64),
            length=2.0,
        )

        self.assertEqual(
            metrics["hamiltonian_drift_pred_reference"],
            "predicted Hamiltonian at t=0",
        )
        self.assertAlmostEqual(
            metrics["hamiltonian_drift_pred_median_abs_conditional_finite_tfinal"],
            0.0,
        )
        self.assertAlmostEqual(
            metrics["energy_error_pred_vs_truth_median_abs_conditional_finite_tfinal"],
            3.0,
        )


class MacroSummaryTest(unittest.TestCase):
    def test_macro_rates_are_equal_family_and_micro_rates_are_pooled(self) -> None:
        summaries = {
            "small": {
                "regime": "small",
                "n_ics_attempted": 2,
                "n_truth_valid": 2,
                "n_truth_invalid": 0,
                "truth_invalid_rate_attempted": 0.0,
                "model_nonfinite_any_count_truth_valid": 1,
                "model_nonfinite_any_rate_truth_valid": 0.5,
                "terminal_eta_failure_count_tau_0p25": 1,
                "terminal_eta_failure_rate_tau_0p25": 0.5,
                "terminal_eta_failure_count_tau_0p5": 1,
                "terminal_eta_failure_rate_tau_0p5": 0.5,
                "terminal_eta_failure_count_tau_0p75": 1,
                "terminal_eta_failure_rate_tau_0p75": 0.5,
                "terminal_eta_failure_count_tau_1": 1,
                "terminal_eta_failure_rate_tau_1": 0.5,
                "rel_l2_eta_median_conditional_finite_tfinal": 0.1,
                "rel_l2_eta_p95_conditional_finite_tfinal": 0.2,
            },
            "large": {
                "regime": "large",
                "n_ics_attempted": 8,
                "n_truth_valid": 8,
                "n_truth_invalid": 0,
                "truth_invalid_rate_attempted": 0.0,
                "model_nonfinite_any_count_truth_valid": 0,
                "model_nonfinite_any_rate_truth_valid": 0.0,
                "terminal_eta_failure_count_tau_0p25": 0,
                "terminal_eta_failure_rate_tau_0p25": 0.0,
                "terminal_eta_failure_count_tau_0p5": 0,
                "terminal_eta_failure_rate_tau_0p5": 0.0,
                "terminal_eta_failure_count_tau_0p75": 0,
                "terminal_eta_failure_rate_tau_0p75": 0.0,
                "terminal_eta_failure_count_tau_1": 0,
                "terminal_eta_failure_rate_tau_1": 0.0,
                "rel_l2_eta_median_conditional_finite_tfinal": 0.3,
                "rel_l2_eta_p95_conditional_finite_tfinal": 0.4,
            },
        }

        macro = compute_macro_summary(summaries)

        self.assertEqual(macro["n_ics_attempted_total"], 10)
        self.assertEqual(macro["model_nonfinite_any_rate_truth_valid_macro"], 0.25)
        self.assertEqual(macro["model_nonfinite_any_rate_truth_valid_micro"], 0.1)
        self.assertEqual(macro["terminal_eta_failure_rate_tau_0p25_macro"], 0.25)
        self.assertEqual(macro["terminal_eta_failure_rate_tau_0p25_micro"], 0.1)
        self.assertEqual(macro["n_families"], 2)
        self.assertAlmostEqual(
            cast(
                float,
                macro["rel_l2_eta_median_conditional_finite_tfinal_macro_mean"],
            ),
            0.2,
        )

    def test_null_metrics_from_empty_cohort_remain_standard_json(self) -> None:
        summary = {
            "regime": "invalid",
            "n_ics_attempted": 2,
            "n_truth_valid": 0,
            "n_truth_invalid": 2,
            "truth_invalid_rate_attempted": 1.0,
            "model_nonfinite_any_count_truth_valid": 0,
            "model_nonfinite_any_rate_truth_valid": None,
            **{
                f"terminal_eta_failure_count_tau_{label}": 0
                for label in ("0p25", "0p5", "0p75", "1")
            },
            **{
                f"terminal_eta_failure_rate_tau_{label}": None
                for label in ("0p25", "0p5", "0p75", "1")
            },
            "rel_l2_eta_median_conditional_finite_tfinal": None,
            "rel_l2_eta_p95_conditional_finite_tfinal": None,
        }

        macro = compute_macro_summary({"invalid": summary})
        self.assertIsNone(
            _json_ready(macro["model_nonfinite_any_rate_truth_valid_micro"])
        )
        self.assertIsNone(_json_ready(float("nan")))


if __name__ == "__main__":
    unittest.main()
