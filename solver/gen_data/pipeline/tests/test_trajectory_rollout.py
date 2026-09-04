"""CPU tests for trajectory integration and nonlinear warm-up."""

from __future__ import annotations

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

from solver.gen_data.pipeline.dno_target import (  # noqa: E402
    compute_dno_target,
)
from solver.gen_data.pipeline.trajectory_config import (  # noqa: E402
    PAPER_ROLLOUT_NUMERICS,
    RolloutNumerics,
)
from solver.gen_data.pipeline.trajectory_integration import (  # noqa: E402
    compute_saved_targets,
    resample_to_target_grid,
    GL2BatchTelemetry,
    InternalHealthTelemetry,
    IntegratedAdjustmentBatch,
    IntegratedTrajectoryBatch,
)
from solver.gen_data.pipeline.trajectory_rollout import (  # noqa: E402
    execute_adjustment_batch,
    execute_trajectory_batch,
)

jax.config.update("jax_enable_x64", True)


def cheap_config(
    *,
    nx: int = 16,
    target_nx: int | None = None,
    maximum_wavenumber: float = 3.0,
    label_dno_order: int | None = None,
    target_maximum_wavenumber: float | None = None,
    internal_hamiltonian_drift_threshold: float | None = None,
) -> RolloutNumerics:
    resolved_target_nx = nx if target_nx is None else target_nx
    resolved_label_dno_order = 0 if label_dno_order is None else label_dno_order
    resolved_target_maximum_wavenumber = (
        maximum_wavenumber
        if target_maximum_wavenumber is None
        else target_maximum_wavenumber
    )
    return RolloutNumerics(
        nx=nx,
        target_nx=resolved_target_nx,
        length=2.0 * math.pi,
        gravity=1.0,
        integration_dno_order=0,
        label_dno_order=resolved_label_dno_order,
        pad_factor=1,
        maximum_wavenumber=maximum_wavenumber,
        target_maximum_wavenumber=resolved_target_maximum_wavenumber,
        saved_dt=0.02,
        substeps_per_saved_frame=1,
        gl2_residual_tolerance=1.0e-8,
        gl2_iteration_cap=8,
        internal_hamiltonian_drift_threshold=(internal_hamiltonian_drift_threshold),
    )


class InjectedRolloutIntegrator:
    """Return deterministic finite trajectory batches."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, np.ndarray]] = []

    def __call__(
        self,
        *,
        eta0: np.ndarray,
        xi0: np.ndarray,
        depths: np.ndarray,
        saved_times: np.ndarray,
        config: RolloutNumerics,
    ) -> IntegratedTrajectoryBatch:
        del xi0, depths
        self.calls.append((config.substeps_per_saved_frame, eta0[:, 0].copy()))
        nt = saved_times.size
        batch, nx = eta0.shape
        eta = np.broadcast_to(eta0, (nt, batch, nx)).copy()
        step_shape = (1, batch)
        converged = np.ones(step_shape, dtype=np.bool_)
        residual = np.zeros(step_shape, dtype=np.float64)
        return IntegratedTrajectoryBatch(
            eta=eta,
            xi=np.zeros_like(eta),
            gxi=np.zeros_like(eta),
            gl2=GL2BatchTelemetry(
                stage_residual=residual,
                converged=converged,
                stage_finite=np.ones(step_shape, dtype=np.bool_),
                state_finite=np.ones(step_shape, dtype=np.bool_),
            ),
        )


class VariableHorizonRolloutIntegrator:
    """Independent synthetic simulations with deliberate post-horizon failures."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, tuple[int, ...], float]] = []
        self.poisoned: list[tuple[int, int]] = []

    def __call__(
        self,
        *,
        eta0: np.ndarray,
        xi0: np.ndarray,
        depths: np.ndarray,
        saved_times: np.ndarray,
        config: RolloutNumerics,
    ) -> IntegratedTrajectoryBatch:
        del xi0, depths
        markers = np.rint(eta0[:, 0]).astype(int)
        self.calls.append(
            (
                config.substeps_per_saved_frame,
                tuple(markers.tolist()),
                float(saved_times[-1]),
            )
        )
        nt = saved_times.size
        batch, nx = eta0.shape
        eta = np.broadcast_to(eta0, (nt, batch, nx)).copy()
        x = 2.0 * np.pi * np.arange(nx, dtype=np.float64) / nx
        eta[:, markers >= 2] += 0.02 * np.cos(x)
        xi = np.zeros_like(eta)
        q_ref = np.zeros_like(eta)

        substeps = config.substeps_per_saved_frame
        step_count = (nt - 1) * substeps
        step_times = (
            config.saved_dt
            / config.substeps_per_saved_frame
            * np.arange(step_count, dtype=np.float64)
        )
        step_shape = (step_count, batch)
        residual = np.zeros(step_shape, dtype=np.float64)
        converged = np.ones(step_shape, dtype=np.bool_)
        stage_finite = np.ones(step_shape, dtype=np.bool_)
        state_finite = np.ones(step_shape, dtype=np.bool_)
        internal_hamiltonian = np.ones((nt, batch), dtype=np.float64)
        internal_state_finite = np.ones((nt, batch), dtype=np.bool_)
        internal_dno_output_finite = np.ones((nt, batch), dtype=np.bool_)
        minimum_water_column = np.full((nt, batch), 9.0, dtype=np.float64)
        horizon_by_marker = {1: 0.04, 2: 0.06, 3: 0.08}
        for simulation_index, marker in enumerate(markers):
            horizon = horizon_by_marker[int(marker)]
            saved_after = saved_times > horizon + 1.0e-14
            steps_after = step_times >= horizon - 1.0e-14
            if np.any(saved_after):
                self.poisoned.append((config.substeps_per_saved_frame, int(marker)))
                eta[saved_after, simulation_index] = np.nan
                xi[saved_after, simulation_index] = np.nan
                q_ref[saved_after, simulation_index] = np.nan
                residual[steps_after, simulation_index] = np.inf
                converged[steps_after, simulation_index] = False
                stage_finite[steps_after, simulation_index] = False
                state_finite[steps_after, simulation_index] = False
                internal_hamiltonian[saved_after, simulation_index] = np.nan
                internal_state_finite[saved_after, simulation_index] = False
                internal_dno_output_finite[saved_after, simulation_index] = False
                minimum_water_column[saved_after, simulation_index] = np.nan

        return IntegratedTrajectoryBatch(
            eta=eta,
            xi=xi,
            gxi=q_ref,
            gl2=GL2BatchTelemetry(
                stage_residual=residual,
                converged=converged,
                stage_finite=stage_finite,
                state_finite=state_finite,
            ),
            internal_telemetry=InternalHealthTelemetry(
                hamiltonian=internal_hamiltonian,
                state_finite=internal_state_finite,
                dno_output_finite=internal_dno_output_finite,
                minimum_water_column=minimum_water_column,
            ),
        )


class InternalHealthRolloutIntegrator:
    """Return five simulations that isolate the four required internal gates."""

    def __call__(
        self,
        *,
        eta0: np.ndarray,
        xi0: np.ndarray,
        depths: np.ndarray,
        saved_times: np.ndarray,
        config: RolloutNumerics,
    ) -> IntegratedTrajectoryBatch:
        del xi0, depths
        saved_count = saved_times.size
        batch_size, nx = eta0.shape
        field_shape = (saved_count, batch_size, nx)
        step_shape = (
            (saved_count - 1) * config.substeps_per_saved_frame,
            batch_size,
        )
        hamiltonian = np.ones((saved_count, batch_size), dtype=np.float64)
        hamiltonian[-1, 1] = 1.01
        state_finite = np.ones_like(hamiltonian, dtype=np.bool_)
        state_finite[-1, 2] = False
        dno_output_finite = np.ones_like(hamiltonian, dtype=np.bool_)
        dno_output_finite[-1, 3] = False
        minimum_water = np.ones_like(hamiltonian)
        minimum_water[-1, 4] = 0.0
        return IntegratedTrajectoryBatch(
            eta=np.broadcast_to(eta0, field_shape).copy(),
            xi=np.zeros(field_shape, dtype=np.float64),
            gxi=np.zeros(field_shape, dtype=np.float64),
            gl2=GL2BatchTelemetry(
                stage_residual=np.zeros(step_shape, dtype=np.float64),
                converged=np.ones(step_shape, dtype=np.bool_),
                stage_finite=np.ones(step_shape, dtype=np.bool_),
                state_finite=np.ones(step_shape, dtype=np.bool_),
            ),
            internal_telemetry=InternalHealthTelemetry(
                hamiltonian=hamiltonian,
                state_finite=state_finite,
                dno_output_finite=dno_output_finite,
                minimum_water_column=minimum_water,
            ),
        )


class InjectedAdjustmentRolloutIntegrator:
    """Return per-simulation endpoints and one deliberate within-horizon failure."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, int]] = []

    def __call__(
        self,
        *,
        eta0: np.ndarray,
        xi0: np.ndarray,
        depths: np.ndarray,
        saved_times: np.ndarray,
        config: RolloutNumerics,
        nonlinear_ramp_times: np.ndarray,
        nonlinear_ramp_order: int,
    ) -> IntegratedAdjustmentBatch:
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
        substeps = config.substeps_per_saved_frame
        step_shape = ((saved_count - 1) * substeps, batch_size)
        step_times = (
            config.saved_dt
            / config.substeps_per_saved_frame
            * np.arange(step_shape[0], dtype=np.float64)
        )
        residual = np.zeros(step_shape, dtype=np.float64)
        converged = np.ones(step_shape, dtype=np.bool_)
        stage_finite = np.ones(step_shape, dtype=np.bool_)
        state_finite = np.ones(step_shape, dtype=np.bool_)
        horizon_by_marker = {1: 0.04, 2: 0.06, 3: 0.08}
        for simulation_index, marker in enumerate(markers):
            horizon = horizon_by_marker[int(marker)]
            saved_after = saved_times > horizon + 1.0e-14
            if np.any(saved_after):
                eta[saved_after, simulation_index] = np.nan
                xi[saved_after, simulation_index] = np.nan
                steps_after = step_times >= horizon - 1.0e-14
                residual[steps_after, simulation_index] = np.inf
                converged[steps_after, simulation_index] = False
            if marker == 3:
                residual[0, simulation_index] = 2.0 * config.gl2_residual_tolerance
                converged[0, simulation_index] = False
        return IntegratedAdjustmentBatch(
            eta=eta,
            xi=xi,
            gl2=GL2BatchTelemetry(
                stage_residual=residual,
                converged=converged,
                stage_finite=stage_finite,
                state_finite=state_finite,
            ),
        )


class TrajectoryRolloutTest(unittest.TestCase):
    def test_dual_grid_delivery_preserves_mode_128_and_drops_higher_modes(
        self,
    ) -> None:
        rollout_config = PAPER_ROLLOUT_NUMERICS["jonswap_tma"]
        x = rollout_config.length * np.arange(rollout_config.nx) / rollout_config.nx
        field = (
            0.7 * np.cos(128.0 * x)
            + 0.4 * np.sin(128.0 * x)
            + 0.5 * np.cos(129.0 * x)
            + 0.3 * np.cos(700.0 * x)
        )

        delivered = np.asarray(
            resample_to_target_grid(
                jnp.asarray(field, dtype=jnp.float64),
                config=rollout_config,
            ),
            dtype=np.float64,
        )

        coefficients = np.fft.rfft(delivered) / rollout_config.target_nx
        self.assertEqual(delivered.shape, (1024,))
        self.assertAlmostEqual(abs(coefficients[128]), math.hypot(0.7, 0.4) / 2.0)
        self.assertLess(float(np.max(np.abs(coefficients[129:]))), 1.0e-14)

    def test_canonical_q_is_recomputed_from_resampled_eta_and_xi(self) -> None:
        rollout_config = PAPER_ROLLOUT_NUMERICS["jonswap_tma"]._replace(
            internal_hamiltonian_drift_threshold=None,
        )
        x = rollout_config.length * np.arange(rollout_config.nx) / rollout_config.nx
        eta = (0.02 * np.cos(3.0 * x) + 0.001 * np.cos(129.0 * x))[None, None, :]
        xi = (0.03 * np.sin(2.0 * x) + 0.002 * np.sin(200.0 * x))[None, None, :]
        depths = jnp.asarray([5.0], dtype=jnp.float64)

        delivered_eta, delivered_xi, delivered_q, internal = compute_saved_targets(
            jnp.asarray(eta),
            jnp.asarray(xi),
            depths,
            rollout_config,
        )
        target_eta = resample_to_target_grid(
            jnp.asarray(eta),
            config=rollout_config,
        )
        target_xi = resample_to_target_grid(
            jnp.asarray(xi),
            config=rollout_config,
        )
        expected_eta, expected_xi, expected_q = compute_dno_target(
            target_eta,
            target_xi,
            depths[:, None],
            nx=rollout_config.target_nx,
            length=rollout_config.length,
            dno_order=rollout_config.label_dno_order,
            pad_factor=rollout_config.pad_factor,
            maximum_wavenumber=rollout_config.target_maximum_wavenumber,
        )

        self.assertIsNone(internal)
        assert delivered_eta is not None and delivered_xi is not None
        np.testing.assert_allclose(
            delivered_eta,
            np.asarray(expected_eta),
            rtol=1.0e-13,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            delivered_xi,
            np.asarray(expected_xi),
            rtol=1.0e-13,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            delivered_q,
            np.asarray(expected_q),
            rtol=1.0e-13,
            atol=1.0e-15,
        )
        self.assertEqual(delivered_q.shape, (1, 1, 1024))

    def test_internal_health_uses_internal_grid_but_cannot_supply_target_q(
        self,
    ) -> None:
        rollout_config = PAPER_ROLLOUT_NUMERICS["jonswap_tma"]
        eta = jnp.zeros((1, 1, rollout_config.nx), dtype=jnp.float64)
        xi = jnp.ones((1, 1, rollout_config.nx), dtype=jnp.float64)
        calls: list[int] = []

        def fake_target(
            eta_value: jax.Array,
            xi_value: jax.Array,
            depth_value: jax.Array,
            *,
            nx: int,
            length: float,
            dno_order: int,
            pad_factor: int,
            maximum_wavenumber: float,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            del depth_value, length, dno_order, pad_factor, maximum_wavenumber
            calls.append(nx)
            marker = 7.0 if nx == rollout_config.target_nx else 999.0
            return eta_value, xi_value, jnp.full_like(eta_value, marker)

        with patch(
            "solver.gen_data.pipeline.trajectory_integration.compute_dno_target",
            side_effect=fake_target,
        ):
            _, _, delivered_q, internal = compute_saved_targets(
                eta,
                xi,
                jnp.asarray([5.0], dtype=jnp.float64),
                rollout_config,
            )

        self.assertEqual(calls, [1024, 2048])
        np.testing.assert_array_equal(delivered_q, 7.0)
        self.assertIsNotNone(internal)

    def test_target_order_is_separate_from_internal_hamiltonian_order(
        self,
    ) -> None:
        rollout_config = cheap_config(
            maximum_wavenumber=6.0,
            label_dno_order=1,
            target_maximum_wavenumber=3.0,
            internal_hamiltonian_drift_threshold=1.0,
        )
        x = 2.0 * math.pi * np.arange(rollout_config.nx) / rollout_config.nx
        eta = np.stack(
            (0.04 * np.cos(x), 0.05 * np.cos(x)),
        )[:, None, :]
        xi = np.stack(
            (0.08 * np.sin(2.0 * x), 0.07 * np.sin(2.0 * x)),
        )[:, None, :]
        depths = jnp.asarray([2.0], dtype=jnp.float64)

        _, _, delivered_q, internal = compute_saved_targets(
            jnp.asarray(eta),
            jnp.asarray(xi),
            depths,
            rollout_config,
        )
        _, _, expected_target_q = compute_dno_target(
            jnp.asarray(eta),
            jnp.asarray(xi),
            depths[:, None],
            nx=rollout_config.target_nx,
            length=rollout_config.length,
            dno_order=rollout_config.label_dno_order,
            pad_factor=rollout_config.pad_factor,
            maximum_wavenumber=rollout_config.target_maximum_wavenumber,
        )
        _, _, expected_internal_q = compute_dno_target(
            jnp.asarray(eta),
            jnp.asarray(xi),
            depths[:, None],
            nx=rollout_config.nx,
            length=rollout_config.length,
            dno_order=rollout_config.integration_dno_order,
            pad_factor=rollout_config.pad_factor,
            maximum_wavenumber=rollout_config.maximum_wavenumber,
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
            * rollout_config.length
            / rollout_config.nx
            * np.sum(
                xi * np.asarray(expected_internal_q) + rollout_config.gravity * eta**2,
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
            float(np.linalg.norm(delivered_q - np.asarray(expected_internal_q))),
            1.0e-8,
        )

    def test_trajectory_requires_each_dense_internal_health_gate(self) -> None:
        rollout_config = cheap_config(
            nx=8,
            maximum_wavenumber=2.0,
            internal_hamiltonian_drift_threshold=1.0e-3,
        )
        with patch(
            "solver.gen_data.pipeline.trajectory_rollout.integrate_batch",
            new=InternalHealthRolloutIntegrator(),
        ):
            execution = execute_trajectory_batch(
                np.zeros((5, rollout_config.nx), dtype=np.float64),
                np.zeros((5, rollout_config.nx), dtype=np.float64),
                np.ones(5, dtype=np.float64),
                (np.asarray([0.0, rollout_config.saved_dt]),) * 5,
                config=rollout_config,
            )

        self.assertEqual(
            tuple(simulation.decision.accepted for simulation in execution),
            (True, False, False, False, False),
        )
        expected_failures = (
            "hamiltonian_drift",
            "nonfinite_state",
            "nonfinite_target",
            "nonpositive_water_height",
        )
        for simulation, failed_check in zip(execution[1:], expected_failures):
            self.assertTrue(getattr(simulation.decision, failed_check))
            self.assertIsNone(simulation.trajectory)

    def test_target_delivers_only_the_declared_lower_band(self) -> None:
        rollout_config = RolloutNumerics(
            nx=16,
            target_nx=16,
            length=2.0 * math.pi,
            integration_dno_order=0,
            label_dno_order=0,
            pad_factor=1,
            maximum_wavenumber=6.0,
            target_maximum_wavenumber=3.0,
            saved_dt=0.02,
            substeps_per_saved_frame=1,
            gravity=1.0,
            gl2_residual_tolerance=1.0e-8,
            gl2_iteration_cap=8,
            internal_hamiltonian_drift_threshold=None,
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

        delivered_eta, delivered_xi, delivered_q, internal = compute_saved_targets(
            jax.numpy.asarray(eta),
            jax.numpy.asarray(xi),
            jax.numpy.asarray([2.0]),
            rollout_config,
        )

        self.assertIsNone(internal)
        modes = np.fft.fftfreq(16, d=1.0 / 16.0)
        outside = np.abs(modes) > 3.0
        for field in (delivered_eta, delivered_xi, delivered_q):
            coefficients = np.fft.fft(field, axis=-1)
            self.assertLess(
                float(np.max(np.abs(coefficients[..., outside]))),
                1.0e-11,
            )

    def test_trajectory_uses_only_the_validated_rollout(self) -> None:
        rollout_config = cheap_config(nx=8, maximum_wavenumber=2.0)
        generator = InjectedRolloutIntegrator()
        eta0 = np.stack(tuple(marker * np.ones(rollout_config.nx) for marker in (0, 1)))

        with patch(
            "solver.gen_data.pipeline.trajectory_rollout.integrate_batch",
            new=generator,
        ):
            execution = execute_trajectory_batch(
                eta0,
                np.zeros_like(eta0),
                np.full(2, 10.0),
                (np.asarray([0.0, 0.02]),) * 2,
                config=rollout_config,
            )

        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(generator.calls[0][0], rollout_config.substeps_per_saved_frame)
        self.assertEqual(
            tuple(simulation.decision.accepted for simulation in execution),
            (True, True),
        )
        self.assertTrue(
            all(simulation.trajectory is not None for simulation in execution)
        )

    def test_variable_duration_trajectory_ignores_later_failures(
        self,
    ) -> None:
        rollout_config = cheap_config(
            nx=8,
            maximum_wavenumber=2.0,
            internal_hamiltonian_drift_threshold=1.0e-3,
        )
        eta0 = np.stack(
            tuple(marker * np.ones(rollout_config.nx) for marker in (1, 2, 3))
        )
        grids = (
            np.arange(3, dtype=np.float64) * rollout_config.saved_dt,
            np.arange(4, dtype=np.float64) * rollout_config.saved_dt,
            np.arange(5, dtype=np.float64) * rollout_config.saved_dt,
        )
        generator = VariableHorizonRolloutIntegrator()

        with patch(
            "solver.gen_data.pipeline.trajectory_rollout.integrate_batch",
            new=generator,
        ):
            execution = execute_trajectory_batch(
                eta0,
                np.zeros_like(eta0),
                np.full(3, 10.0),
                grids,
                config=rollout_config,
            )

        self.assertEqual(
            generator.calls,
            [(rollout_config.substeps_per_saved_frame, (1, 2, 3), 0.08)],
        )
        self.assertEqual(
            tuple(simulation.decision.accepted for simulation in execution),
            (True, True, True),
        )
        for simulation in execution:
            self.assertFalse(simulation.decision.hamiltonian_drift)
            self.assertIsNotNone(simulation.health_metrics)
            assert simulation.health_metrics is not None
            self.assertEqual(
                simulation.health_metrics.maximum_relative_hamiltonian_drift,
                0.0,
            )

    def test_nonlinear_adjustment_returns_each_endpoint_and_rejects_failure(
        self,
    ) -> None:
        rollout_config = cheap_config(nx=8, maximum_wavenumber=2.0)
        eta0 = np.stack(
            tuple(marker * np.ones(rollout_config.nx) for marker in (1, 2, 3))
        )
        xi0 = np.zeros_like(eta0)
        grids = (
            np.arange(3, dtype=np.float64) * rollout_config.saved_dt,
            np.arange(4, dtype=np.float64) * rollout_config.saved_dt,
            np.arange(5, dtype=np.float64) * rollout_config.saved_dt,
        )
        ramp_times = np.asarray([0.4, 0.6, 0.8], dtype=np.float64)
        rollout_integrator = InjectedAdjustmentRolloutIntegrator()

        with patch(
            "solver.gen_data.pipeline.trajectory_rollout.integrate_adjustment_batch",
            new=rollout_integrator,
        ):
            execution = execute_adjustment_batch(
                eta0,
                xi0,
                np.full(3, 10.0),
                grids,
                nonlinear_ramp_times=ramp_times,
                nonlinear_ramp_order=4,
                config=rollout_config,
            )

        self.assertEqual(len(rollout_integrator.calls), 1)
        np.testing.assert_array_equal(rollout_integrator.calls[0][0], ramp_times)
        self.assertEqual(rollout_integrator.calls[0][1], 4)
        self.assertEqual(
            tuple(simulation.decision.accepted for simulation in execution),
            (True, True, False),
        )
        for index, expected_terminal in enumerate((1.04, 2.06)):
            simulation = execution[index]
            self.assertIsNotNone(simulation.terminal_eta)
            self.assertIsNotNone(simulation.terminal_xi)
            assert simulation.terminal_eta is not None
            assert simulation.terminal_xi is not None
            np.testing.assert_allclose(simulation.terminal_eta, expected_terminal)
            np.testing.assert_allclose(
                simulation.terminal_xi,
                -float(grids[index][-1]),
            )
            self.assertFalse(simulation.decision.integration_failure)
            self.assertEqual(simulation.maximum_gl2_stage_residual, 0.0)
        rejected = execution[2]
        self.assertIsNone(rejected.terminal_eta)
        self.assertIsNone(rejected.terminal_xi)
        self.assertTrue(rejected.decision.integration_failure)
        self.assertTrue(rejected.decision.incomplete_trajectory)


if __name__ == "__main__":
    unittest.main()
