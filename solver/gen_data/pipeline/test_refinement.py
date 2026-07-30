"""CPU tests for production GL2 execution and refinement audits."""
from __future__ import annotations

import math
import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.pipeline.quality import QualityReason  # noqa: E402
from solver.gen_data.pipeline.refinement import (  # noqa: E402
    PAPER_GL2_CONTRACT,
    ResidualControlledArm,
    ResidualControlledGL2Contract,
    execute_production_trajectory,
    execute_residual_controlled_refinement,
    execute_variable_horizon_production_trajectory,
    execute_variable_horizon_residual_controlled_refinement,
)

jax.config.update("jax_enable_x64", True)


def cheap_contract(
    *,
    nx: int = 16,
    maximum_wavenumber: float = 3.0,
) -> ResidualControlledGL2Contract:
    return ResidualControlledGL2Contract(
        nx=nx,
        length=2.0 * math.pi,
        gravity=1.0,
        dno_order=0,
        pad_factor=1,
        maximum_wavenumber=maximum_wavenumber,
        production_dt=0.02,
        saved_dt=0.02,
        gl2_residual_tolerance=1.0e-8,
        gl2_iteration_cap=8,
        refinement_tolerance=1.0e-3,
        relative_floor=1.0e-12,
        target_time_chunk_size=2,
    )


class InjectedArmExecutor:
    """Deterministic arms that exercise all three routing outcomes."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, np.ndarray]] = []

    def __call__(
        self,
        *,
        eta0: np.ndarray,
        xi0: np.ndarray,
        depths: np.ndarray,
        saved_times: np.ndarray,
        contract: ResidualControlledGL2Contract,
        dt: float,
    ) -> ResidualControlledArm:
        del xi0, depths
        self.calls.append((dt, eta0[:, 0].copy()))
        nt = saved_times.size
        batch, nx = eta0.shape
        eta = np.broadcast_to(eta0, (nt, batch, nx)).copy()
        x = 2.0 * np.pi * np.arange(nx, dtype=np.float64) / nx

        if math.isclose(dt, contract.audit_dt):
            for case_index, marker in enumerate(eta0[:, 0]):
                if marker >= 1.0:
                    eta[:, case_index] += 0.02 * np.cos(x)
                if marker == 4.0:
                    eta[:, case_index] = -11.0
        elif math.isclose(dt, contract.audit_retry_dt):
            eta += 0.02 * np.cos(x)[None, None, :]

        complete = np.ones(batch, dtype=np.bool_)
        if math.isclose(dt, contract.audit_dt):
            complete[eta0[:, 0] == 2.0] = False
        step_shape = (1, batch)
        q_ref = np.zeros_like(eta)
        converged = np.ones(step_shape, dtype=np.bool_)
        residual = np.zeros(step_shape, dtype=np.float64)
        if math.isclose(dt, contract.audit_dt):
            q_ref[:, eta0[:, 0] == 3.0] = np.nan
            converged[:, eta0[:, 0] == 5.0] = False
            residual[:, eta0[:, 0] == 5.0] = 2.0e-8
        return ResidualControlledArm(
            dt=dt,
            times=saved_times.copy(),
            eta=eta,
            xi=np.zeros_like(eta),
            q_ref=q_ref,
            complete=complete,
            gl2_stage_residual=residual,
            gl2_iterations=np.ones(step_shape, dtype=np.int32),
            gl2_converged=converged,
            gl2_stage_finite=np.ones(step_shape, dtype=np.bool_),
            gl2_state_finite=np.ones(step_shape, dtype=np.bool_),
            gl2_hit_iteration_cap=np.zeros(step_shape, dtype=np.bool_),
        )


class VariableHorizonArmExecutor:
    """Independent synthetic cases with deliberate post-horizon failures."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, tuple[int, ...], float]] = []
        self.poisoned: list[tuple[float, int]] = []

    def __call__(
        self,
        *,
        eta0: np.ndarray,
        xi0: np.ndarray,
        depths: np.ndarray,
        saved_times: np.ndarray,
        contract: ResidualControlledGL2Contract,
        dt: float,
    ) -> ResidualControlledArm:
        del xi0, depths
        markers = np.rint(eta0[:, 0]).astype(int)
        self.calls.append((dt, tuple(markers.tolist()), float(saved_times[-1])))
        nt = saved_times.size
        batch, nx = eta0.shape
        eta = np.broadcast_to(eta0, (nt, batch, nx)).copy()
        x = 2.0 * np.pi * np.arange(nx, dtype=np.float64) / nx
        if math.isclose(dt, contract.production_dt, abs_tol=1.0e-15):
            eta[:, markers >= 2] += 0.02 * np.cos(x)
        xi = np.zeros_like(eta)
        q_ref = np.zeros_like(eta)

        substeps = int(round(contract.saved_dt / dt))
        step_count = (nt - 1) * substeps
        step_times = dt * np.arange(step_count, dtype=np.float64)
        step_shape = (step_count, batch)
        residual = np.zeros(step_shape, dtype=np.float64)
        converged = np.ones(step_shape, dtype=np.bool_)
        stage_finite = np.ones(step_shape, dtype=np.bool_)
        state_finite = np.ones(step_shape, dtype=np.bool_)
        horizon_by_marker = {1: 0.04, 2: 0.06, 3: 0.08}
        for case_index, marker in enumerate(markers):
            horizon = horizon_by_marker[int(marker)]
            saved_after = saved_times > horizon + 1.0e-14
            steps_after = step_times >= horizon - 1.0e-14
            if np.any(saved_after):
                self.poisoned.append((dt, int(marker)))
                eta[saved_after, case_index] = np.nan
                xi[saved_after, case_index] = np.nan
                q_ref[saved_after, case_index] = np.nan
                residual[steps_after, case_index] = np.inf
                converged[steps_after, case_index] = False
                stage_finite[steps_after, case_index] = False
                state_finite[steps_after, case_index] = False

        return ResidualControlledArm(
            dt=dt,
            times=saved_times.copy(),
            eta=eta,
            xi=xi,
            q_ref=q_ref,
            complete=np.ones(batch, dtype=np.bool_),
            gl2_stage_residual=residual,
            gl2_iterations=np.ones(step_shape, dtype=np.int32),
            gl2_converged=converged,
            gl2_stage_finite=stage_finite,
            gl2_state_finite=state_finite,
            gl2_hit_iteration_cap=np.zeros(step_shape, dtype=np.bool_),
        )


class ResidualControlledRefinementTest(unittest.TestCase):
    def test_production_defaults_are_frozen(self) -> None:
        contract = PAPER_GL2_CONTRACT
        self.assertEqual(
            (
                contract.nx,
                contract.dno_order,
                contract.pad_factor,
                contract.maximum_wavenumber,
                contract.dtype,
            ),
            (1024, 6, 8, 128.0, "float64"),
        )
        self.assertEqual(
            (
                contract.production_dt,
                contract.audit_dt,
                contract.audit_retry_dt,
                contract.gl2_residual_tolerance,
                contract.gl2_iteration_cap,
                contract.refinement_tolerance,
            ),
            (0.01, 0.005, 0.0025, 1.0e-8, 4, 1.0e-3),
        )
        self.assertEqual(contract.production_dt, 0.01)

    def test_production_uses_only_the_validated_arm(self) -> None:
        contract = cheap_contract(nx=8, maximum_wavenumber=2.0)
        executor = InjectedArmExecutor()
        eta0 = np.stack(
            tuple(marker * np.ones(contract.nx) for marker in (0, 1))
        )

        execution = execute_production_trajectory(
            eta0,
            np.zeros_like(eta0),
            np.full(2, 10.0),
            np.asarray([0.0, 0.02]),
            contract=contract,
            arm_executor=executor,
        )

        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(executor.calls[0][0], contract.production_dt)
        np.testing.assert_array_equal(
            execution.accepted_mask,
            np.asarray([True, True]),
        )
        self.assertTrue(
            all(
                case.dt == contract.production_dt
                and case.retained_trajectory is not None
                for case in execution.cases
            )
        )

    def test_variable_horizon_production_crops_post_horizon_failures(
        self,
    ) -> None:
        contract = cheap_contract(nx=8, maximum_wavenumber=2.0)
        eta0 = np.stack(
            tuple(marker * np.ones(contract.nx) for marker in (1, 2, 3))
        )
        grids = (
            np.arange(3, dtype=np.float64) * contract.saved_dt,
            np.arange(4, dtype=np.float64) * contract.saved_dt,
            np.arange(5, dtype=np.float64) * contract.saved_dt,
        )
        executor = VariableHorizonArmExecutor()

        execution = execute_variable_horizon_production_trajectory(
            eta0,
            np.zeros_like(eta0),
            np.full(3, 10.0),
            grids,
            contract=contract,
            arm_executor=executor,
        )

        self.assertEqual(
            executor.calls,
            [(contract.production_dt, (1, 2, 3), 0.08)],
        )
        np.testing.assert_array_equal(
            execution.accepted_mask,
            np.asarray([True, True, True]),
        )
        self.assertEqual(
            tuple(case.case_index for case in execution.cases),
            (0, 1, 2),
        )

    def test_actual_cpu_pair_accepts_and_q_ref_is_delivered_band(self) -> None:
        contract = cheap_contract()
        x = 2.0 * np.pi * np.arange(contract.nx) / contract.nx
        eta0 = np.stack(
            (0.001 * np.cos(x), 0.0015 * np.cos(2.0 * x))
        )
        xi0 = np.stack(
            (0.002 * np.sin(2.0 * x), 0.001 * np.sin(x))
        )
        execution = execute_residual_controlled_refinement(
            eta0,
            xi0,
            np.asarray([1.0, 1.5]),
            np.asarray([0.0, 0.02, 0.04]),
            contract=contract,
        )

        np.testing.assert_array_equal(
            execution.accepted_mask, np.asarray([True, True])
        )
        self.assertFalse(execution.retry_was_run)
        self.assertIsNone(execution.retry_timing)
        self.assertEqual(execution.retry_case_indices.size, 0)
        retained_q_ref = np.stack(
            tuple(
                case.retained_trajectory.gxi
                for case in execution.cases
                if case.retained_trajectory is not None
            ),
            axis=1,
        )
        self.assertEqual(retained_q_ref.shape, (3, 2, contract.nx))
        means = np.mean(retained_q_ref, axis=-1)
        self.assertLess(float(np.max(np.abs(means))), 1.0e-13)
        coefficients = np.fft.fft(retained_q_ref, axis=-1)
        modes = np.fft.fftfreq(contract.nx, d=1.0 / contract.nx)
        high_band = coefficients[..., np.abs(modes) > 3.0]
        self.assertLess(float(np.max(np.abs(high_band))), 1.0e-11)
        for case in execution.cases:
            self.assertEqual(case.retained_dt, contract.audit_dt)
            self.assertIsNotNone(case.retained_trajectory)
            self.assertEqual(case.primary.decision.required_bits, 1 << 5)

    def test_only_eligible_primary_failure_is_retried_once(self) -> None:
        contract = cheap_contract(nx=8, maximum_wavenumber=2.0)
        executor = InjectedArmExecutor()
        eta0 = np.stack(
            tuple(marker * np.ones(contract.nx) for marker in range(6))
        )
        execution = execute_residual_controlled_refinement(
            eta0,
            np.zeros_like(eta0),
            np.full(6, 10.0),
            np.asarray([0.0, 0.02]),
            contract=contract,
            arm_executor=executor,
        )

        self.assertEqual([call[0] for call in executor.calls], [0.02, 0.01, 0.005])
        np.testing.assert_array_equal(
            executor.calls[-1][1], np.asarray([1.0])
        )
        np.testing.assert_array_equal(
            execution.retry_case_indices, np.asarray([1])
        )
        np.testing.assert_array_equal(
            execution.accepted_mask,
            np.asarray([True, True, False, False, False, False]),
        )

        accepted_primary, accepted_retry, *rejected_cases = execution.cases
        self.assertFalse(accepted_primary.retry_eligible)
        self.assertIsNone(accepted_primary.retry)
        self.assertEqual(accepted_primary.retained_dt, contract.audit_dt)

        self.assertTrue(accepted_retry.retry_eligible)
        self.assertIsNotNone(accepted_retry.retry)
        self.assertEqual(accepted_retry.retained_dt, contract.audit_retry_dt)
        self.assertIsNotNone(accepted_retry.retained_trajectory)

        for rejected in rejected_cases:
            self.assertFalse(rejected.retry_eligible)
            self.assertIsNone(rejected.retry)
            self.assertIsNone(rejected.retained_dt)
            self.assertIsNone(rejected.retained_trajectory)

        incomplete, nonfinite, graph_invalid, unsolved = rejected_cases
        self.assertTrue(
            incomplete.effective_decision.evaluated
            & QualityReason.INCOMPLETE_TRAJECTORY
        )
        self.assertTrue(
            incomplete.effective_decision.failed
            & QualityReason.INCOMPLETE_TRAJECTORY
        )
        self.assertTrue(
            incomplete.effective_decision.failed
            & QualityReason.TEMPORAL_DEFECT
        )
        self.assertTrue(
            nonfinite.effective_decision.failed
            & QualityReason.NONFINITE_TARGET
        )
        self.assertTrue(
            graph_invalid.effective_decision.failed
            & QualityReason.BOTTOM_CLEARANCE
        )
        self.assertTrue(
            unsolved.effective_decision.failed
            & QualityReason.GL2_STAGE_RESIDUAL
        )

    def test_variable_horizons_equal_serial_with_prefix_poison_and_retry(
        self,
    ) -> None:
        contract = cheap_contract(nx=8, maximum_wavenumber=2.0)
        eta0 = np.stack(
            tuple(marker * np.ones(contract.nx) for marker in (1, 2, 3))
        )
        xi0 = np.zeros_like(eta0)
        depths = np.full(3, 10.0)
        grids = (
            np.arange(3, dtype=np.float64) * contract.saved_dt,
            np.arange(4, dtype=np.float64) * contract.saved_dt,
            np.arange(5, dtype=np.float64) * contract.saved_dt,
        )

        batched_arms = VariableHorizonArmExecutor()
        batched = execute_variable_horizon_residual_controlled_refinement(
            eta0,
            xi0,
            depths,
            grids,
            contract=contract,
            arm_executor=batched_arms,
        )
        serial_cases = []
        for case_index, grid in enumerate(grids):
            serial = execute_residual_controlled_refinement(
                eta0[case_index : case_index + 1],
                xi0[case_index : case_index + 1],
                depths[case_index : case_index + 1],
                grid,
                contract=contract,
                arm_executor=VariableHorizonArmExecutor(),
            )
            serial_cases.append(serial.cases[0])

        np.testing.assert_array_equal(
            batched.retry_case_indices,
            np.asarray([1, 2], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            batched.accepted_mask,
            np.asarray([True, True, True]),
        )
        self.assertEqual(
            batched_arms.calls,
            [
                (contract.production_dt, (1, 2, 3), 0.08),
                (contract.audit_dt, (1, 2, 3), 0.08),
                (contract.audit_retry_dt, (2, 3), 0.08),
            ],
        )
        self.assertIn((contract.production_dt, 1), batched_arms.poisoned)
        self.assertIn((contract.audit_dt, 1), batched_arms.poisoned)
        self.assertIn((contract.audit_retry_dt, 2), batched_arms.poisoned)

        for grid, batched_case, serial_case in zip(
            grids,
            batched.cases,
            serial_cases,
        ):
            self.assertEqual(batched_case.accepted, serial_case.accepted)
            self.assertEqual(
                batched_case.retry_eligible,
                serial_case.retry_eligible,
            )
            self.assertEqual(
                batched_case.retry_reason,
                serial_case.retry_reason,
            )
            self.assertEqual(
                batched_case.retained_dt,
                serial_case.retained_dt,
            )
            self.assertEqual(
                batched_case.primary.metrics,
                serial_case.primary.metrics,
            )
            self.assertEqual(
                batched_case.primary.decision,
                serial_case.primary.decision,
            )
            self.assertEqual(
                batched_case.retry is None,
                serial_case.retry is None,
            )
            attempt_pairs = [
                (batched_case.primary, serial_case.primary),
            ]
            if batched_case.retry is not None:
                assert serial_case.retry is not None
                self.assertEqual(
                    batched_case.retry.metrics,
                    serial_case.retry.metrics,
                )
                self.assertEqual(
                    batched_case.retry.decision,
                    serial_case.retry.decision,
                )
                attempt_pairs.append(
                    (batched_case.retry, serial_case.retry)
                )
            for batched_attempt, serial_attempt in attempt_pairs:
                for telemetry_name in (
                    "coarse_telemetry",
                    "fine_telemetry",
                ):
                    batched_telemetry = getattr(
                        batched_attempt,
                        telemetry_name,
                    )
                    serial_telemetry = getattr(
                        serial_attempt,
                        telemetry_name,
                    )
                    self.assertEqual(
                        batched_telemetry.all_stages_solved,
                        serial_telemetry.all_stages_solved,
                    )
                    self.assertEqual(
                        batched_telemetry.maximum_stage_residual,
                        serial_telemetry.maximum_stage_residual,
                    )
                    for array_name in (
                        "residual",
                        "iterations",
                        "converged",
                        "stage_finite",
                        "state_finite",
                        "hit_iteration_cap",
                    ):
                        np.testing.assert_array_equal(
                            getattr(batched_telemetry, array_name),
                            getattr(serial_telemetry, array_name),
                        )
            batched_trajectory = batched_case.retained_trajectory
            serial_trajectory = serial_case.retained_trajectory
            self.assertIsNotNone(batched_trajectory)
            self.assertIsNotNone(serial_trajectory)
            assert batched_trajectory is not None
            assert serial_trajectory is not None
            self.assertEqual(
                batched_trajectory.eta.shape,
                (grid.size, contract.nx),
            )
            np.testing.assert_array_equal(
                batched_trajectory.times,
                serial_trajectory.times,
            )
            np.testing.assert_array_equal(
                batched_trajectory.eta,
                serial_trajectory.eta,
            )
            np.testing.assert_array_equal(
                batched_trajectory.xi,
                serial_trajectory.xi,
            )
            np.testing.assert_array_equal(
                batched_trajectory.gxi,
                serial_trajectory.gxi,
            )


if __name__ == "__main__":
    unittest.main()
