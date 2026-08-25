"""CPU tests for production GL2 execution and nonlinear adjustment."""
from __future__ import annotations

from dataclasses import replace
import math
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.pipeline.quality import QualityReason  # noqa: E402
from solver.gen_data.pipeline.reference import (  # noqa: E402
    evaluate_discrete_dno_target,
)
from solver.gen_data.pipeline.refinement import (  # noqa: E402
    PAPER_BENJAMIN_FEIR_GL2_CONTRACT,
    PAPER_GL2_CONTRACT,
    PAPER_JONSWAP_GL2_CONTRACT,
    PAPER_TANAKA_GL2_CONTRACT,
    TANAKA_BUFFERED_HOU_LI_GL2_CONTRACT,
    TANAKA_HOU_LI_GL2_CONTRACT,
    _evaluate_saved_target,
    _project_fixed_band_to_target_grid,
    InternalTrajectoryTelemetry,
    NonlinearAdjustmentArm,
    PostStepStateFilter,
    ResidualControlledArm,
    ResidualControlledGL2Contract,
    execute_production_trajectory,
    execute_variable_horizon_nonlinear_adjustment,
    execute_variable_horizon_production_trajectory,
)

jax.config.update("jax_enable_x64", True)


def cheap_contract(
    *,
    nx: int = 16,
    target_nx: int | None = None,
    maximum_wavenumber: float = 3.0,
    post_step_state_filter: PostStepStateFilter = "sharp",
    target_dno_order: int | None = None,
    target_maximum_wavenumber: float | None = None,
    internal_hamiltonian_drift_threshold: float | None = None,
) -> ResidualControlledGL2Contract:
    return ResidualControlledGL2Contract(
        nx=nx,
        target_nx=target_nx,
        length=2.0 * math.pi,
        gravity=1.0,
        dno_order=0,
        target_dno_order=target_dno_order,
        pad_factor=1,
        maximum_wavenumber=maximum_wavenumber,
        target_maximum_wavenumber=target_maximum_wavenumber,
        production_dt=0.02,
        saved_dt=0.02,
        gl2_residual_tolerance=1.0e-8,
        gl2_iteration_cap=8,
        target_time_chunk_size=2,
        post_step_state_filter=post_step_state_filter,
        internal_hamiltonian_drift_threshold=(
            internal_hamiltonian_drift_threshold
        ),
    )


class InjectedArmExecutor:
    """Return deterministic finite production arms."""

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
        complete = np.ones(batch, dtype=np.bool_)
        step_shape = (1, batch)
        q_ref = np.zeros_like(eta)
        converged = np.ones(step_shape, dtype=np.bool_)
        residual = np.zeros(step_shape, dtype=np.float64)
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
        internal_hamiltonian = np.ones((nt, batch), dtype=np.float64)
        internal_state_finite = np.ones((nt, batch), dtype=np.bool_)
        internal_dno_finite = np.ones((nt, batch), dtype=np.bool_)
        minimum_water_column = np.full((nt, batch), 9.0, dtype=np.float64)
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
                internal_hamiltonian[saved_after, case_index] = np.nan
                internal_state_finite[saved_after, case_index] = False
                internal_dno_finite[saved_after, case_index] = False
                minimum_water_column[saved_after, case_index] = np.nan

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
            internal_telemetry=InternalTrajectoryTelemetry(
                hamiltonian=internal_hamiltonian,
                state_finite=internal_state_finite,
                dno_finite=internal_dno_finite,
                minimum_water_column=minimum_water_column,
            ),
        )


class InternalHealthArmExecutor:
    """Return five cases that isolate the four required internal gates."""

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
        saved_count = saved_times.size
        batch_size, nx = eta0.shape
        field_shape = (saved_count, batch_size, nx)
        step_shape = (
            (saved_count - 1) * int(round(contract.saved_dt / dt)),
            batch_size,
        )
        hamiltonian = np.ones((saved_count, batch_size), dtype=np.float64)
        hamiltonian[-1, 1] = 1.01
        state_finite = np.ones_like(hamiltonian, dtype=np.bool_)
        state_finite[-1, 2] = False
        dno_finite = np.ones_like(hamiltonian, dtype=np.bool_)
        dno_finite[-1, 3] = False
        minimum_water = np.ones_like(hamiltonian)
        minimum_water[-1, 4] = 0.0
        return ResidualControlledArm(
            dt=dt,
            times=saved_times.copy(),
            eta=np.broadcast_to(eta0, field_shape).copy(),
            xi=np.zeros(field_shape, dtype=np.float64),
            q_ref=np.zeros(field_shape, dtype=np.float64),
            complete=np.ones(batch_size, dtype=np.bool_),
            gl2_stage_residual=np.zeros(step_shape, dtype=np.float64),
            gl2_iterations=np.ones(step_shape, dtype=np.int32),
            gl2_converged=np.ones(step_shape, dtype=np.bool_),
            gl2_stage_finite=np.ones(step_shape, dtype=np.bool_),
            gl2_state_finite=np.ones(step_shape, dtype=np.bool_),
            gl2_hit_iteration_cap=np.zeros(step_shape, dtype=np.bool_),
            internal_telemetry=InternalTrajectoryTelemetry(
                hamiltonian=hamiltonian,
                state_finite=state_finite,
                dno_finite=dno_finite,
                minimum_water_column=minimum_water,
            ),
        )


class InjectedAdjustmentArmExecutor:
    """Return per-case endpoints and one deliberate within-horizon failure."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, int]] = []

    def __call__(
        self,
        *,
        eta0: np.ndarray,
        xi0: np.ndarray,
        depths: np.ndarray,
        saved_times: np.ndarray,
        contract: ResidualControlledGL2Contract,
        dt: float,
        nonlinear_ramp_times: np.ndarray,
        nonlinear_ramp_order: int,
    ) -> NonlinearAdjustmentArm:
        del depths
        self.calls.append((nonlinear_ramp_times.copy(), nonlinear_ramp_order))
        markers = np.rint(eta0[:, 0]).astype(int)
        saved_count = saved_times.size
        batch_size, nx = eta0.shape
        eta = np.broadcast_to(
            eta0[None, :, :],
            (saved_count, batch_size, nx),
        ).copy()
        xi = np.broadcast_to(
            xi0[None, :, :],
            (saved_count, batch_size, nx),
        ).copy()
        eta += saved_times[:, None, None]
        xi -= saved_times[:, None, None]
        substeps = int(round(contract.saved_dt / dt))
        step_shape = ((saved_count - 1) * substeps, batch_size)
        step_times = dt * np.arange(step_shape[0], dtype=np.float64)
        residual = np.zeros(step_shape, dtype=np.float64)
        converged = np.ones(step_shape, dtype=np.bool_)
        stage_finite = np.ones(step_shape, dtype=np.bool_)
        state_finite = np.ones(step_shape, dtype=np.bool_)
        horizon_by_marker = {1: 0.04, 2: 0.06, 3: 0.08}
        for case_index, marker in enumerate(markers):
            horizon = horizon_by_marker[int(marker)]
            saved_after = saved_times > horizon + 1.0e-14
            if np.any(saved_after):
                eta[saved_after, case_index] = np.nan
                xi[saved_after, case_index] = np.nan
                steps_after = step_times >= horizon - 1.0e-14
                residual[steps_after, case_index] = np.inf
                converged[steps_after, case_index] = False
            if marker == 3:
                residual[0, case_index] = 2.0 * contract.gl2_residual_tolerance
                converged[0, case_index] = False
        return NonlinearAdjustmentArm(
            dt=dt,
            times=saved_times.copy(),
            eta=eta,
            xi=xi,
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
        self.assertEqual(contract.production_dt, 0.01)
        self.assertEqual(contract.gl2_residual_tolerance, 1.0e-8)
        self.assertEqual(contract.gl2_iteration_cap, 4)
        self.assertEqual(contract.post_step_state_filter, "sharp")
        self.assertIsNone(contract.post_step_maximum_wavenumber)
        self.assertEqual(
            contract.post_step_filter_fraction,
            contract.filter_fraction,
        )
        self.assertEqual(
            (
                TANAKA_HOU_LI_GL2_CONTRACT.post_step_state_filter,
                TANAKA_HOU_LI_GL2_CONTRACT.hou_li_coefficient,
                TANAKA_HOU_LI_GL2_CONTRACT.hou_li_power,
            ),
            ("hou_li", 36.0, 36),
        )
        self.assertEqual(
            TANAKA_HOU_LI_GL2_CONTRACT.target_definition,
            PAPER_GL2_CONTRACT.target_definition,
        )
        buffered = TANAKA_BUFFERED_HOU_LI_GL2_CONTRACT
        self.assertEqual(buffered.maximum_wavenumber, 224.0)
        self.assertEqual(
            buffered.target_definition.maximum_wavenumber,
            128.0,
        )
        self.assertEqual(buffered.post_step_maximum_wavenumber, 224.0)
        self.assertEqual(buffered.post_step_state_filter, "hou_li")

        paper_tanaka = PAPER_TANAKA_GL2_CONTRACT
        self.assertEqual(
            (
                paper_tanaka.maximum_wavenumber,
                paper_tanaka.target_definition.maximum_wavenumber,
                paper_tanaka.post_step_maximum_wavenumber,
                paper_tanaka.post_step_state_filter,
                paper_tanaka.hou_li_coefficient,
                paper_tanaka.hou_li_power,
            ),
            (256.0, 128.0, 256.0, "hou_li", 36.0, 36),
        )
        paper_bf = PAPER_BENJAMIN_FEIR_GL2_CONTRACT
        self.assertEqual(
            (
                paper_bf.nx,
                paper_bf.target_nx,
                paper_bf.dno_order,
                paper_bf.maximum_wavenumber,
                paper_bf.target_definition.nx,
                paper_bf.target_definition.dno_order,
                paper_bf.target_definition.maximum_wavenumber,
                paper_bf.gl2_iteration_cap,
            ),
            (1024, None, 4, 256.0, 1024, 6, 128.0, 4),
        )
        paper_jonswap = PAPER_JONSWAP_GL2_CONTRACT
        self.assertEqual(
            (
                paper_jonswap.nx,
                paper_jonswap.target_nx,
                paper_jonswap.dno_order,
                paper_jonswap.maximum_wavenumber,
                paper_jonswap.target_definition.nx,
                paper_jonswap.target_definition.dno_order,
                paper_jonswap.target_definition.maximum_wavenumber,
                paper_jonswap.gl2_iteration_cap,
                paper_jonswap.post_step_state_filter,
                paper_jonswap.internal_hamiltonian_drift_threshold,
            ),
            (2048, 1024, 4, 704.0, 1024, 6, 128.0, 5, "sharp", 1.0e-3),
        )

    def test_dual_grid_delivery_preserves_mode_128_and_drops_higher_modes(
        self,
    ) -> None:
        contract = PAPER_JONSWAP_GL2_CONTRACT
        x = contract.length * np.arange(contract.nx) / contract.nx
        field = (
            0.7 * np.cos(128.0 * x)
            + 0.4 * np.sin(128.0 * x)
            + 0.5 * np.cos(129.0 * x)
            + 0.3 * np.cos(700.0 * x)
        )

        delivered = np.asarray(
            _project_fixed_band_to_target_grid(
                jnp.asarray(field, dtype=jnp.float64),
                contract=contract,
            ),
            dtype=np.float64,
        )

        coefficients = np.fft.rfft(delivered) / contract.delivered_nx
        self.assertEqual(delivered.shape, (1024,))
        self.assertAlmostEqual(abs(coefficients[128]), math.hypot(0.7, 0.4) / 2.0)
        self.assertLess(float(np.max(np.abs(coefficients[129:]))), 1.0e-14)

    def test_canonical_q_is_recomputed_from_resampled_eta_and_xi(self) -> None:
        contract = replace(
            PAPER_JONSWAP_GL2_CONTRACT,
            target_time_chunk_size=1,
            internal_hamiltonian_drift_threshold=None,
        )
        x = contract.length * np.arange(contract.nx) / contract.nx
        eta = (0.02 * np.cos(3.0 * x) + 0.001 * np.cos(129.0 * x))[
            None, None, :
        ]
        xi = (0.03 * np.sin(2.0 * x) + 0.002 * np.sin(200.0 * x))[
            None, None, :
        ]
        depths = jnp.asarray([5.0], dtype=jnp.float64)

        delivered_eta, delivered_xi, delivered_q, internal = (
            _evaluate_saved_target(
                jnp.asarray(eta),
                jnp.asarray(xi),
                depths,
                contract,
            )
        )
        target_eta = _project_fixed_band_to_target_grid(
            jnp.asarray(eta),
            contract=contract,
        )
        target_xi = _project_fixed_band_to_target_grid(
            jnp.asarray(xi),
            contract=contract,
        )
        expected_eta, expected_xi, expected_q = evaluate_discrete_dno_target(
            target_eta,
            target_xi,
            depths[:, None],
            definition=contract.target_definition,
        )

        self.assertIsNone(internal)
        assert delivered_eta is not None and delivered_xi is not None
        np.testing.assert_allclose(delivered_eta, np.asarray(expected_eta))
        np.testing.assert_allclose(delivered_xi, np.asarray(expected_xi))
        np.testing.assert_allclose(delivered_q, np.asarray(expected_q))
        self.assertEqual(delivered_q.shape, (1, 1, 1024))

    def test_internal_health_uses_internal_grid_but_cannot_supply_target_q(
        self,
    ) -> None:
        contract = replace(
            PAPER_JONSWAP_GL2_CONTRACT,
            target_time_chunk_size=1,
        )
        eta = jnp.zeros((1, 1, contract.nx), dtype=jnp.float64)
        xi = jnp.ones((1, 1, contract.nx), dtype=jnp.float64)
        calls: list[int] = []

        def fake_target(
            eta_value: jax.Array,
            xi_value: jax.Array,
            depth_value: jax.Array,
            *,
            definition: object,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            del depth_value
            nx = getattr(definition, "nx")
            calls.append(nx)
            marker = 7.0 if nx == contract.delivered_nx else 999.0
            return eta_value, xi_value, jnp.full_like(eta_value, marker)

        with patch(
            "solver.gen_data.pipeline.refinement.evaluate_discrete_dno_target",
            side_effect=fake_target,
        ):
            _, _, delivered_q, internal = _evaluate_saved_target(
                eta,
                xi,
                jnp.asarray([5.0], dtype=jnp.float64),
                contract,
            )

        self.assertEqual(calls, [1024, 2048])
        np.testing.assert_array_equal(delivered_q, 7.0)
        self.assertIsNotNone(internal)

    def test_target_order_is_separate_from_internal_hamiltonian_order(
        self,
    ) -> None:
        contract = cheap_contract(
            maximum_wavenumber=6.0,
            target_dno_order=1,
            target_maximum_wavenumber=3.0,
            internal_hamiltonian_drift_threshold=1.0,
        )
        x = 2.0 * math.pi * np.arange(contract.nx) / contract.nx
        eta = np.stack(
            (0.04 * np.cos(x), 0.05 * np.cos(x)),
        )[:, None, :]
        xi = np.stack(
            (0.08 * np.sin(2.0 * x), 0.07 * np.sin(2.0 * x)),
        )[:, None, :]
        depths = jnp.asarray([2.0], dtype=jnp.float64)

        _, _, delivered_q, internal = _evaluate_saved_target(
            jnp.asarray(eta),
            jnp.asarray(xi),
            depths,
            contract,
        )
        _, _, expected_target_q = evaluate_discrete_dno_target(
            jnp.asarray(eta),
            jnp.asarray(xi),
            depths[:, None],
            definition=contract.target_definition,
        )
        _, _, expected_internal_q = evaluate_discrete_dno_target(
            jnp.asarray(eta),
            jnp.asarray(xi),
            depths[:, None],
            definition=contract.internal_definition,
        )

        np.testing.assert_allclose(
            delivered_q,
            np.asarray(expected_target_q),
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        self.assertIsNotNone(internal)
        assert internal is not None
        expected_hamiltonian = (
            0.5
            * contract.length
            / contract.nx
            * np.sum(
                xi * np.asarray(expected_internal_q)
                + contract.gravity * eta**2,
                axis=-1,
            )
        )
        np.testing.assert_allclose(
            internal.hamiltonian,
            expected_hamiltonian,
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        self.assertGreater(
            float(
                np.linalg.norm(
                    delivered_q - np.asarray(expected_internal_q)
                )
            ),
            1.0e-8,
        )

    def test_production_requires_each_dense_internal_health_gate(self) -> None:
        contract = cheap_contract(
            nx=8,
            maximum_wavenumber=2.0,
            internal_hamiltonian_drift_threshold=1.0e-3,
        )
        execution = execute_production_trajectory(
            np.zeros((5, contract.nx), dtype=np.float64),
            np.zeros((5, contract.nx), dtype=np.float64),
            np.ones(5, dtype=np.float64),
            np.asarray([0.0, contract.saved_dt]),
            contract=contract,
            arm_executor=InternalHealthArmExecutor(),
        )

        np.testing.assert_array_equal(
            execution.accepted_mask,
            np.asarray([True, False, False, False, False]),
        )
        expected_failures = (
            QualityReason.HAMILTONIAN_DRIFT,
            QualityReason.NONFINITE_STATE,
            QualityReason.NONFINITE_TARGET,
            QualityReason.BOTTOM_CLEARANCE,
        )
        for case, reason in zip(execution.cases[1:], expected_failures):
            self.assertTrue(case.decision.required & reason)
            self.assertTrue(case.decision.evaluated & reason)
            self.assertTrue(case.decision.failed & reason)
            self.assertIsNone(case.retained_trajectory)

    def test_buffered_target_delivers_only_the_declared_lower_band(self) -> None:
        contract = ResidualControlledGL2Contract(
            nx=16,
            length=2.0 * math.pi,
            dno_order=0,
            pad_factor=1,
            maximum_wavenumber=6.0,
            target_maximum_wavenumber=3.0,
            production_dt=0.02,
            saved_dt=0.02,
            target_time_chunk_size=2,
            post_step_state_filter="hou_li",
            post_step_maximum_wavenumber=6.0,
        )
        x = 2.0 * math.pi * np.arange(16, dtype=np.float64) / 16.0
        eta = np.stack(
            (
                np.cos(2.0 * x) + 0.25 * np.cos(5.0 * x),
                np.cos(x) - 0.4 * np.sin(6.0 * x),
            )
        )[:, None, :]
        xi = np.stack(
            (
                np.sin(3.0 * x) + 0.3 * np.cos(6.0 * x),
                np.sin(2.0 * x) - 0.2 * np.sin(5.0 * x),
            )
        )[:, None, :]

        delivered_eta, delivered_xi, delivered_q, internal = (
            _evaluate_saved_target(
                jax.numpy.asarray(eta),
                jax.numpy.asarray(xi),
                jax.numpy.asarray([2.0]),
                contract,
            )
        )

        self.assertIsNotNone(delivered_eta)
        self.assertIsNotNone(delivered_xi)
        self.assertIsNone(internal)
        assert delivered_eta is not None
        assert delivered_xi is not None
        modes = np.fft.fftfreq(16, d=1.0 / 16.0)
        outside = np.abs(modes) > 3.0
        for field in (delivered_eta, delivered_xi, delivered_q):
            coefficients = np.fft.fft(field, axis=-1)
            self.assertLess(
                float(np.max(np.abs(coefficients[..., outside]))),
                1.0e-11,
            )

    def test_target_band_cannot_exceed_the_evolution_band(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "target_maximum_wavenumber",
        ):
            ResidualControlledGL2Contract(
                maximum_wavenumber=4.0,
                target_maximum_wavenumber=5.0,
            )

    def test_post_step_band_can_be_inside_evolution_band(self) -> None:
        contract = ResidualControlledGL2Contract(
            maximum_wavenumber=192.0,
            post_step_maximum_wavenumber=172.0,
        )

        self.assertEqual(contract.filter_fraction, 192.0 / 512.0)
        self.assertEqual(contract.post_step_filter_fraction, 172.0 / 512.0)
        self.assertEqual(
            contract.target_definition.maximum_wavenumber,
            192.0,
        )

        with self.assertRaisesRegex(
            ValueError,
            "no larger than maximum_wavenumber",
        ):
            ResidualControlledGL2Contract(
                maximum_wavenumber=192.0,
                post_step_maximum_wavenumber=193.0,
            )

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
        contract = cheap_contract(
            nx=8,
            maximum_wavenumber=2.0,
            internal_hamiltonian_drift_threshold=1.0e-3,
        )
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
        for case in execution.cases:
            self.assertTrue(
                case.decision.required & QualityReason.HAMILTONIAN_DRIFT
            )
            self.assertIsNotNone(case.internal_metrics)
            assert case.internal_metrics is not None
            self.assertEqual(
                case.internal_metrics.maximum_relative_hamiltonian_drift,
                0.0,
            )

    def test_nonlinear_adjustment_returns_each_endpoint_and_rejects_failure(
        self,
    ) -> None:
        contract = cheap_contract(nx=8, maximum_wavenumber=2.0)
        eta0 = np.stack(
            tuple(marker * np.ones(contract.nx) for marker in (1, 2, 3))
        )
        xi0 = np.zeros_like(eta0)
        grids = (
            np.arange(3, dtype=np.float64) * contract.saved_dt,
            np.arange(4, dtype=np.float64) * contract.saved_dt,
            np.arange(5, dtype=np.float64) * contract.saved_dt,
        )
        ramp_times = np.asarray([0.4, 0.6, 0.8], dtype=np.float64)
        arm_executor = InjectedAdjustmentArmExecutor()

        execution = execute_variable_horizon_nonlinear_adjustment(
            eta0,
            xi0,
            np.full(3, 10.0),
            grids,
            nonlinear_ramp_times=ramp_times,
            nonlinear_ramp_order=4,
            contract=contract,
            arm_executor=arm_executor,
        )

        self.assertEqual(len(arm_executor.calls), 1)
        np.testing.assert_array_equal(arm_executor.calls[0][0], ramp_times)
        self.assertEqual(arm_executor.calls[0][1], 4)
        np.testing.assert_array_equal(
            execution.accepted_mask,
            np.asarray([True, True, False]),
        )
        for index, expected_terminal in enumerate((1.04, 2.06)):
            case = execution.cases[index]
            self.assertIsNotNone(case.terminal_eta)
            self.assertIsNotNone(case.terminal_xi)
            assert case.terminal_eta is not None
            assert case.terminal_xi is not None
            np.testing.assert_allclose(case.terminal_eta, expected_terminal)
            np.testing.assert_allclose(
                case.terminal_xi,
                -float(grids[index][-1]),
            )
            self.assertTrue(case.telemetry.all_stages_solved)
            self.assertEqual(case.telemetry.maximum_stage_residual, 0.0)
        rejected = execution.cases[2]
        self.assertIsNone(rejected.terminal_eta)
        self.assertIsNone(rejected.terminal_xi)
        self.assertTrue(
            rejected.decision.failed & QualityReason.GL2_STAGE_RESIDUAL
        )
        self.assertTrue(
            rejected.decision.failed & QualityReason.INCOMPLETE_TRAJECTORY
        )


if __name__ == "__main__":
    unittest.main()
