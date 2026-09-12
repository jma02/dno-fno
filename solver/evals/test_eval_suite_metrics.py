"""Focused regression tests for manuscript rollout metrics."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import cast
import unittest
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from solver.evals.eval_suite import (  # noqa: E402
    IC,
    FamilyConfig,
    _json_ready,
    _load_paper_dataset_ics,
    _rollout_ic_chunks,
    _try_load_cached_truth,
    _write_truth_cache,
    compute_macro_summary,
    compute_metrics,
    run_family,
)
from solver.evals.model_rollout import LoadedRun, build_predict_gxi_batched  # noqa: E402
from dno_net_v2 import CraigSulemDNO  # noqa: E402
from fno1d import SpectralConv1d  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from solver.gen_data.pipeline.batch_storage import save_completed_batch  # noqa: E402
from scripts.build_paper_dataset import build_dataset  # noqa: E402
from solver.gen_data.pipeline.types import (  # noqa: E402
    PhysicalFamilyId,
    SimulationRows,
)


class ComputeMetricsTest(unittest.TestCase):
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
                    pred=pred if reuse else None,
                ))
                with np.load(directory / "tanaka_trajs.npz") as archive:
                    saved.append({name: archive[name].copy() for name in archive.files})
            self.assertEqual(rollout.call_count, 1)
            self.assertEqual(_json_ready(summaries[0]), _json_ready(summaries[1]))
            for name in saved[0]:
                np.testing.assert_array_equal(saved[0][name], saved[1][name])
            self.assertEqual(summaries[1]["model_nonfinite_any_count_truth_valid"], 1)

    def test_evaluation_promotes_float32_inputs_before_model_arithmetic(self) -> None:
        model = CraigSulemDNO(width=8, n_blocks=1, latent=2, mult_hidden=4)
        inputs = 0.01 * jax.random.normal(jax.random.PRNGKey(3), (1, 16, 2), dtype=jnp.float32)
        depth = jnp.zeros((1, 1), dtype=jnp.float32)
        params = jax.tree_util.tree_map(
            lambda value: value.astype(jnp.float64),
            model.init(jax.random.PRNGKey(0), inputs, depth)["params"],
        )
        params["cs_block_0"]["phi_proj"]["kernel"] = jnp.full((4, 2), 0.1, dtype=jnp.float64)
        loaded = LoadedRun(
            model, params, {}, {"feature_absmax": [1.0, 1.0], "target_absmax": 1.0},
            "scale", 0,
        )
        output = build_predict_gxi_batched(loaded)(inputs[..., 0], inputs[..., 1], depth)
        self.assertEqual(output.dtype, jnp.float64)
        expected = cast(jax.Array, model.apply(
            {"params": params}, inputs.astype(jnp.float64), depth.astype(jnp.float64)
        ))[..., 0]
        np.testing.assert_allclose(output, expected - expected.mean(axis=-1, keepdims=True), rtol=1e-12, atol=1e-14)

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
