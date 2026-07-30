"""Focused CPU tests for the frozen full-horizon panel runner."""

from __future__ import annotations

from dataclasses import replace
import math
import os
import unittest
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np  # noqa: E402

from scripts.run_full_horizon_refinement_panel import (  # noqa: E402
    _arm_path,
    _subset_arm_cases,
    build_case_definitions,
    build_group_definitions,
    case_support_record,
    final_halving_warrant,
    production_contract,
    smoke_contract,
    summarize_stage_telemetry,
)


class FrozenProductionContractTest(unittest.TestCase):
    def test_numerical_contract_is_exact(self) -> None:
        contract = production_contract()
        self.assertEqual(contract.nx, 1024)
        self.assertEqual(contract.dno_order, 6)
        self.assertEqual(contract.pad_factor, 8)
        self.assertEqual(contract.delivered_wavenumber, 128.0)
        self.assertEqual(contract.dtype, "float64")
        self.assertEqual(contract.coarse_dt, 0.01)
        self.assertEqual(contract.fine_dt, 0.005)
        self.assertEqual(contract.final_retry_dt, 0.0025)
        self.assertEqual(contract.saved_dt, 0.08)
        self.assertEqual(contract.gl2_residual_tolerance, 1.0e-8)
        self.assertEqual(contract.gl2_iteration_cap, 8)
        self.assertEqual(contract.refinement_tolerance, 1.0e-3)
        self.assertFalse(contract.smoke)

    def test_every_predeclared_case_is_supported_once(self) -> None:
        contract = production_contract()
        cases = build_case_definitions()
        groups = build_group_definitions(cases, contract)
        assigned = [case_id for group in groups for case_id in group.case_ids]

        self.assertEqual(len(cases), 12)
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertCountEqual(assigned, [case.case_id for case in cases])
        self.assertTrue(
            all(case_support_record(case, contract)["inside"] for case in cases)
        )

    def test_support_checks_reject_parameters_outside_declared_laws(
        self,
    ) -> None:
        contract = production_contract()
        cases = {case.case_id: case for case in build_case_definitions()}

        stokes = cases["stokes_deep_98000193"]
        mode = int(stokes.parameters["n0"])
        boundary_depth = 5.0 * contract.length / (2.0 * math.pi * mode)
        self.assertTrue(
            case_support_record(
                replace(stokes, depth=boundary_depth),
                contract,
            )["inside"]
        )

        benjamin_feir = cases["bf_jcp09_canonical"]
        self.assertFalse(
            case_support_record(
                replace(benjamin_feir, depth=benjamin_feir.depth + 1.0e-10),
                contract,
            )["inside"]
        )
        nonfinite_phase = {
            **benjamin_feir.parameters,
            "phase": math.nan,
        }
        self.assertFalse(
            case_support_record(
                replace(benjamin_feir, parameters=nonfinite_phase),
                contract,
            )["inside"]
        )

        tanaka = cases["tanaka_may_case31"]
        wrong_reconstruction = {
            **tanaka.parameters,
            "profile_reconstruction": "piecewise_linear",
        }
        self.assertFalse(
            case_support_record(
                replace(tanaka, parameters=wrong_reconstruction),
                contract,
            )["inside"]
        )
        overlapping_crests = {
            **tanaka.parameters,
            "crests": [
                {
                    "amplitude": 0.055,
                    "center": 1.0,
                    "direction": -1,
                },
                {
                    "amplitude": 0.055,
                    "center": 1.1,
                    "direction": 1,
                },
            ],
        }
        self.assertFalse(
            case_support_record(
                replace(tanaka, parameters=overlapping_crests),
                contract,
            )["inside"]
        )

    def test_family_horizons_follow_the_declared_policy(self) -> None:
        contract = production_contract()
        groups = {
            group.name: group
            for group in build_group_definitions(build_case_definitions(), contract)
        }
        self.assertEqual(groups["stokes"].realized_terminal_time, 20.0)
        self.assertEqual(groups["tanaka_benjamin_feir"].realized_terminal_time, 200.0)
        for name in (
            "jonswap_tma_shallow",
            "jonswap_tma_finite",
            "jonswap_tma_deep",
        ):
            group = groups[name]
            self.assertLessEqual(
                group.realized_terminal_time, group.intended_terminal_time
            )
            self.assertLess(
                group.intended_terminal_time - group.realized_terminal_time,
                contract.saved_dt + 1.0e-12,
            )
            ratio = group.realized_terminal_time / contract.saved_dt
            self.assertTrue(math.isclose(ratio, round(ratio), abs_tol=1.0e-12))

    def test_smoke_keeps_the_refinement_and_stage_tolerances(self) -> None:
        production = production_contract()
        smoke = smoke_contract()
        self.assertEqual(smoke.coarse_dt, production.coarse_dt)
        self.assertEqual(smoke.fine_dt, production.fine_dt)
        self.assertEqual(smoke.final_retry_dt, production.final_retry_dt)
        self.assertEqual(
            smoke.gl2_residual_tolerance,
            production.gl2_residual_tolerance,
        )
        self.assertEqual(smoke.gl2_iteration_cap, production.gl2_iteration_cap)
        self.assertEqual(smoke.refinement_tolerance, production.refinement_tolerance)
        self.assertTrue(smoke.smoke)

    def test_final_halving_artifact_has_an_exact_unambiguous_name(self) -> None:
        path = _arm_path(Path("/tmp"), "panel_final_halving", 0.0025)
        self.assertEqual(path.name, "panel_final_halving_dt_0p0025.npz")


class TelemetrySummaryTest(unittest.TestCase):
    def test_summary_preserves_failures_and_iteration_histogram(self) -> None:
        arm = {
            "gl2_stage_residual": np.asarray([[1.0e-10], [2.0e-9], [3.0e-7]]),
            "gl2_iterations": np.asarray([[1], [2], [8]], dtype=np.int32),
            "gl2_converged": np.asarray([[True], [True], [False]], dtype=np.bool_),
            "gl2_stage_finite": np.ones((3, 1), dtype=np.bool_),
            "gl2_state_finite": np.ones((3, 1), dtype=np.bool_),
            "gl2_hit_iteration_cap": np.asarray(
                [[False], [False], [True]], dtype=np.bool_
            ),
            "gl2_step_times": np.asarray([0.0, 0.01, 0.02]),
            "gl2_step_dts": np.full(3, 0.01),
        }

        summary = summarize_stage_telemetry(arm, case_index=0, iteration_cap=8)

        self.assertFalse(summary["all_stages_converged"])
        self.assertEqual(summary["failed_stage_count"], 1)
        self.assertEqual(summary["iteration_cap_hit_count"], 1)
        self.assertEqual(summary["first_failed_step"], 2)
        self.assertEqual(summary["first_failed_step_time"], 0.02)
        self.assertEqual(summary["iteration_histogram"]["1"], 1)
        self.assertEqual(summary["iteration_histogram"]["2"], 1)
        self.assertEqual(summary["iteration_histogram"]["8"], 1)


class FinalHalvingPolicyTest(unittest.TestCase):
    @staticmethod
    def _fine_arm() -> dict[str, np.ndarray]:
        return {
            "times": np.asarray([0.0, 0.02]),
            "eta": np.zeros((2, 2, 4)),
            "xi": np.zeros((2, 2, 4)),
            "gxi": np.zeros((2, 2, 4)),
            "gl2_stage_residual": np.zeros((4, 2)),
            "gl2_iterations": np.ones((4, 2), dtype=np.int32),
            "gl2_converged": np.ones((4, 2), dtype=np.bool_),
            "gl2_stage_finite": np.ones((4, 2), dtype=np.bool_),
            "gl2_state_finite": np.ones((4, 2), dtype=np.bool_),
            "gl2_hit_iteration_cap": np.zeros((4, 2), dtype=np.bool_),
            "gl2_step_times": np.asarray([0.0, 0.005, 0.01, 0.015]),
            "gl2_step_dts": np.full(4, 0.005),
            "gl2_all_stages_converged": np.ones(2, dtype=np.bool_),
            "gl2_max_stage_residual": np.zeros(2),
            "gl2_first_failed_step": np.full(2, -1, dtype=np.int32),
        }

    @staticmethod
    def _primary_record(accepted: bool) -> dict[str, object]:
        return {
            "support": {"inside": True},
            "quality_decision": {"refinement_accepted": accepted},
        }

    def test_retry_requires_a_failed_primary_and_usable_fine_arm(self) -> None:
        arm = self._fine_arm()
        warranted, _ = final_halving_warrant(
            self._primary_record(False),
            arm,
            case_index=0,
            depth=1.0,
        )
        self.assertTrue(warranted)

        accepted, reason = final_halving_warrant(
            self._primary_record(True),
            arm,
            case_index=0,
            depth=1.0,
        )
        self.assertFalse(accepted)
        self.assertIn("accepted", reason)

        arm["gl2_converged"][0, 0] = False
        unsolved, reason = final_halving_warrant(
            self._primary_record(False),
            arm,
            case_index=0,
            depth=1.0,
        )
        self.assertFalse(unsolved)
        self.assertIn("unsolved", reason)

    def test_retry_is_not_run_when_dt005_is_nonfinite(self) -> None:
        arm = self._fine_arm()
        arm["gxi"][1, 0, 2] = np.nan
        warranted, reason = final_halving_warrant(
            self._primary_record(False),
            arm,
            case_index=0,
            depth=1.0,
        )
        self.assertFalse(warranted)
        self.assertIn("nonfinite", reason)

    def test_case_subset_preserves_coordinates_and_selects_batch_axis(self) -> None:
        arm = self._fine_arm()
        subset = _subset_arm_cases(arm, np.asarray([1], dtype=np.int64))
        np.testing.assert_array_equal(subset["times"], arm["times"])
        self.assertEqual(subset["eta"].shape, (2, 1, 4))
        self.assertEqual(subset["gl2_converged"].shape, (4, 1))
        self.assertEqual(subset["gl2_all_stages_converged"].shape, (1,))


if __name__ == "__main__":
    unittest.main()
