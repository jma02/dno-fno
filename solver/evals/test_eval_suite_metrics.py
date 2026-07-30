"""Focused regression tests for manuscript rollout metrics."""
from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from solver.evals.eval_suite import (  # noqa: E402
    IC,
    _json_ready,
    _rollout_ic_chunks,
    compute_macro_summary,
    compute_metrics,
)


class ComputeMetricsTest(unittest.TestCase):
    def test_rollout_chunks_preserve_panel_order_and_sum_wall_time(self) -> None:
        ics = [
            IC(
                eta=np.zeros(3),
                xi=np.zeros(3),
                depth=1.0,
                case_id=case_id,
            )
            for case_id in range(5)
        ]
        calls: list[list[int]] = []

        def rollout(chunk: list[IC]) -> dict[str, np.ndarray | float]:
            case_ids = [ic.case_id for ic in chunk]
            calls.append(case_ids)
            values = np.broadcast_to(
                np.asarray(case_ids, dtype=np.float32)[None, :, None],
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

    def test_nonfinite_predictions_are_failures_and_truth_reasons_are_explicit(self) -> None:
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
        self.assertEqual(metrics["nan_rate"], 0.0)

        # IC 0 exceeds 0.25; IC 3 fails every threshold because xi became
        # non-finite, even though its terminal eta error itself is small.
        self.assertEqual(metrics["terminal_eta_failure_rate_tau_0p25"], 1.0)
        self.assertEqual(metrics["terminal_eta_failure_rate_tau_0p5"], 0.5)
        self.assertEqual(metrics["terminal_eta_failure_rate_tau_0p75"], 0.5)
        self.assertEqual(metrics["terminal_eta_failure_rate_tau_1"], 0.5)
        self.assertEqual(metrics["divergence_rate_final"], 0.0)

        # Quantiles are explicitly conditional on an all-field finite rollout,
        # so only IC 0 contributes at the terminal frame.
        self.assertEqual(metrics["rel_l2_eta_n_conditional_finite_tfinal"], 1)
        self.assertAlmostEqual(
            metrics["rel_l2_eta_median_conditional_finite_tfinal"], 0.3
        )
        self.assertAlmostEqual(metrics["rel_l2_eta_median_tfinal"], 0.2)

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
        self.assertEqual(
            metrics["energy_drift_pred_reference"],
            "truth Hamiltonian at t=0 (historical definition)",
        )
        self.assertAlmostEqual(metrics["energy_drift_pred_median_at_tfinal"], 3.0)
        self.assertAlmostEqual(
            metrics["energy_error_pred_vs_truth_median_abs_conditional_finite_tfinal"],
            3.0,
        )


class MacroSummaryTest(unittest.TestCase):
    def test_macro_rates_are_equal_family_and_micro_rates_are_pooled(self) -> None:
        summaries = {
            "small": {
                "regime": "small",
                "ic_panel_source": {"replicate_group": "shared"},
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
                "ic_panel_source": {"replicate_group": "shared"},
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
        self.assertEqual(macro["n_distribution_groups"], 1)
        self.assertEqual(
            macro["model_nonfinite_any_rate_truth_valid_distribution_group_macro"],
            0.1,
        )
        self.assertAlmostEqual(
            macro["rel_l2_eta_median_conditional_finite_tfinal_macro_mean"], 0.2
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
        self.assertIsNone(_json_ready(macro["model_nonfinite_any_rate_truth_valid_micro"]))
        self.assertIsNone(_json_ready(float("nan")))


if __name__ == "__main__":
    unittest.main()
