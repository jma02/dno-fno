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

from solver.gen_data.pipeline.dno_target import compute_dno_target  # noqa: E402
from solver.gen_data.pipeline.trajectory_config import (  # noqa: E402
    PAPER_ROLLOUT_NUMERICS,
    RolloutNumerics,
)
from solver.gen_data.pipeline.trajectory_integration import (  # noqa: E402
    IntegratedAdjustmentBatch,
    IntegratedTrajectoryBatch,
    SolverGridHealth,
    integrate_batch,
    resample_to_target_grid,
)
from solver.gen_data.pipeline.trajectory_rollout import (  # noqa: E402
    execute_adjustment_batch,
    execute_trajectory_batch,
)

jax.config.update("jax_enable_x64", True)


def _config(
    *,
    nx: int = 16,
    target_nx: int | None = None,
    maximum_wavenumber: float = 3.0,
    label_dno_order: int = 0,
    target_maximum_wavenumber: float | None = None,
    health_threshold: float | None = None,
) -> RolloutNumerics:
    return RolloutNumerics(
        nx=nx,
        target_nx=nx if target_nx is None else target_nx,
        length=2.0 * math.pi,
        gravity=1.0,
        integration_dno_order=0,
        label_dno_order=label_dno_order,
        pad_factor=1,
        maximum_wavenumber=maximum_wavenumber,
        target_maximum_wavenumber=(
            maximum_wavenumber
            if target_maximum_wavenumber is None
            else target_maximum_wavenumber
        ),
        saved_dt=0.02,
        substeps_per_saved_frame=1,
        gl2_residual_tolerance=1.0e-8,
        gl2_iteration_cap=8,
        internal_hamiltonian_drift_threshold=health_threshold,
    )


class TrajectoryRolloutTest(unittest.TestCase):
    def test_dual_grid_delivery_preserves_mode_128_and_drops_higher_modes(
        self,
    ) -> None:
        config = PAPER_ROLLOUT_NUMERICS["jonswap_tma"]
        x = config.length * np.arange(config.nx) / config.nx
        field = (
            0.7 * np.cos(128.0 * x)
            + 0.4 * np.sin(128.0 * x)
            + 0.5 * np.cos(129.0 * x)
            + 0.3 * np.cos(700.0 * x)
        )

        delivered = np.asarray(
            resample_to_target_grid(jnp.asarray(field), config=config),
            dtype=np.float64,
        )

        coefficients = np.fft.rfft(delivered) / config.target_nx
        self.assertEqual(delivered.shape, (1024,))
        self.assertAlmostEqual(abs(coefficients[128]), math.hypot(0.7, 0.4) / 2.0)
        self.assertLess(float(np.max(np.abs(coefficients[129:]))), 1.0e-14)

    def test_target_q_is_recomputed_from_resampled_eta_and_xi(self) -> None:
        config = _config(
            target_nx=8,
            maximum_wavenumber=6.0,
            label_dno_order=1,
            target_maximum_wavenumber=3.0,
        )
        x = config.length * np.arange(config.nx) / config.nx
        eta = (0.02 * np.cos(2.0 * x) + 0.01 * np.cos(5.0 * x))[None, None]
        xi = (0.03 * np.sin(x) + 0.02 * np.sin(6.0 * x))[None, None]
        depths = np.asarray([2.0])
        with patch(
            "solver.gen_data.pipeline.trajectory_integration._integrate_gl2",
            return_value=(
                {
                    "eta": jnp.asarray(eta),
                    "xi": jnp.asarray(xi),
                    "gl2_converged": jnp.ones((0, 1), dtype=jnp.bool_),
                },
                jnp.asarray(depths),
            ),
        ):
            delivered = integrate_batch(
                eta0=eta[0],
                xi0=xi[0],
                depths=depths,
                saved_times=np.asarray([0.0]),
                config=config,
            )

        target_eta = resample_to_target_grid(jnp.asarray(eta), config=config)
        target_xi = resample_to_target_grid(jnp.asarray(xi), config=config)
        expected = compute_dno_target(
            target_eta,
            target_xi,
            jnp.asarray(depths)[:, None],
            nx=config.target_nx,
            length=config.length,
            dno_order=config.label_dno_order,
            pad_factor=config.pad_factor,
            maximum_wavenumber=config.target_maximum_wavenumber,
        )
        np.testing.assert_allclose(delivered.eta, np.asarray(expected[0])[:1])
        np.testing.assert_allclose(delivered.xi, np.asarray(expected[1])[:1])
        np.testing.assert_allclose(delivered.gxi, np.asarray(expected[2])[:1])
        self.assertEqual(delivered.gxi.shape, (1, 1, 8))

    def test_solver_health_and_labels_use_their_declared_dno_orders(self) -> None:
        config = _config(
            maximum_wavenumber=6.0,
            label_dno_order=1,
            target_maximum_wavenumber=3.0,
            health_threshold=1.0,
        )
        x = config.length * np.arange(config.nx) / config.nx
        eta = np.stack((0.04 * np.cos(x), 0.05 * np.cos(x)))[:, None]
        xi = np.stack((0.08 * np.sin(2.0 * x), 0.07 * np.sin(2.0 * x)))[:, None]
        depths = np.asarray([2.0])
        with patch(
            "solver.gen_data.pipeline.trajectory_integration._integrate_gl2",
            return_value=(
                {
                    "eta": jnp.asarray(eta),
                    "xi": jnp.asarray(xi),
                    "gl2_converged": jnp.ones((1, 1), dtype=jnp.bool_),
                },
                jnp.asarray(depths),
            ),
        ):
            delivered = integrate_batch(
                eta0=eta[0],
                xi0=xi[0],
                depths=depths,
                saved_times=np.asarray([0.0, config.saved_dt]),
                config=config,
            )

        expected_label_q = compute_dno_target(
            jnp.asarray(eta),
            jnp.asarray(xi),
            jnp.asarray(depths)[:, None],
            nx=config.target_nx,
            length=config.length,
            dno_order=config.label_dno_order,
            pad_factor=config.pad_factor,
            maximum_wavenumber=config.target_maximum_wavenumber,
        )[2]
        expected_solver_q = compute_dno_target(
            jnp.asarray(eta),
            jnp.asarray(xi),
            jnp.asarray(depths)[:, None],
            nx=config.nx,
            length=config.length,
            dno_order=config.integration_dno_order,
            pad_factor=config.pad_factor,
            maximum_wavenumber=config.maximum_wavenumber,
        )[2]
        np.testing.assert_allclose(delivered.gxi, np.asarray(expected_label_q))
        assert delivered.solver_grid_health is not None
        expected_hamiltonian = (
            0.5
            * config.length
            / config.nx
            * np.sum(
                xi * np.asarray(expected_solver_q) + config.gravity * eta**2,
                axis=-1,
            )
        )
        np.testing.assert_allclose(
            delivered.solver_grid_health.hamiltonian,
            expected_hamiltonian,
        )
        self.assertGreater(
            float(np.linalg.norm(delivered.gxi - np.asarray(expected_solver_q))),
            1.0e-8,
        )

    def test_trajectory_requires_all_four_solver_grid_health_gates(self) -> None:
        config = _config(
            nx=8,
            maximum_wavenumber=2.0,
            health_threshold=1.0e-3,
        )

        def integrate(
            *,
            eta0: np.ndarray,
            xi0: np.ndarray,
            depths: np.ndarray,
            saved_times: np.ndarray,
            config: RolloutNumerics,
        ) -> IntegratedTrajectoryBatch:
            del xi0, depths, config
            shape = (saved_times.size, eta0.shape[0], eta0.shape[1])
            hamiltonian = np.ones(shape[:2])
            hamiltonian[-1, 1] = 1.01
            state_finite = np.ones(shape[:2], dtype=np.bool_)
            state_finite[-1, 2] = False
            dno_finite = np.ones(shape[:2], dtype=np.bool_)
            dno_finite[-1, 3] = False
            water_column = np.ones(shape[:2])
            water_column[-1, 4] = 0.0
            return IntegratedTrajectoryBatch(
                np.broadcast_to(eta0, shape).copy(),
                np.zeros(shape),
                np.zeros(shape),
                np.ones((saved_times.size - 1, eta0.shape[0]), dtype=np.bool_),
                SolverGridHealth(
                    hamiltonian,
                    state_finite,
                    dno_finite,
                    water_column,
                ),
            )

        with patch(
            "solver.gen_data.pipeline.trajectory_rollout.integrate_batch",
            new=integrate,
        ):
            simulations = execute_trajectory_batch(
                np.zeros((5, config.nx)),
                np.zeros((5, config.nx)),
                np.ones(5),
                (np.asarray([0.0, config.saved_dt]),) * 5,
                config=config,
            )

        self.assertEqual(
            tuple(simulation is not None for simulation in simulations),
            (True, False, False, False, False),
        )
        self.assertTrue(all(simulation is None for simulation in simulations[1:]))

    def test_variable_horizons_ignore_failures_after_each_prefix(self) -> None:
        config = _config(
            nx=8,
            maximum_wavenumber=2.0,
            health_threshold=1.0e-3,
        )
        grids = tuple(
            np.arange(count, dtype=np.float64) * config.saved_dt for count in (3, 4, 5)
        )
        eta0 = np.repeat(np.arange(1.0, 4.0)[:, None], config.nx, axis=1)

        def integrate(
            *,
            eta0: np.ndarray,
            xi0: np.ndarray,
            depths: np.ndarray,
            saved_times: np.ndarray,
            config: RolloutNumerics,
        ) -> IntegratedTrajectoryBatch:
            del xi0, depths, config
            shape = (saved_times.size, eta0.shape[0], eta0.shape[1])
            eta = np.broadcast_to(eta0, shape).copy()
            xi = np.zeros(shape)
            gxi = np.zeros(shape)
            converged = np.ones((saved_times.size - 1, eta0.shape[0]), dtype=np.bool_)
            hamiltonian = np.ones(shape[:2])
            state_finite = np.ones(shape[:2], dtype=np.bool_)
            dno_finite = np.ones(shape[:2], dtype=np.bool_)
            water_column = np.ones(shape[:2])
            for index, grid in enumerate(grids):
                eta[grid.size :, index] = np.nan
                xi[grid.size :, index] = np.nan
                gxi[grid.size :, index] = np.nan
                converged[grid.size - 1 :, index] = False
                hamiltonian[grid.size :, index] = np.nan
                state_finite[grid.size :, index] = False
                dno_finite[grid.size :, index] = False
                water_column[grid.size :, index] = np.nan
            return IntegratedTrajectoryBatch(
                eta,
                xi,
                gxi,
                converged,
                SolverGridHealth(
                    hamiltonian,
                    state_finite,
                    dno_finite,
                    water_column,
                ),
            )

        with patch(
            "solver.gen_data.pipeline.trajectory_rollout.integrate_batch",
            new=integrate,
        ):
            simulations = execute_trajectory_batch(
                eta0,
                np.zeros_like(eta0),
                np.full(3, 10.0),
                grids,
                config=config,
            )

        self.assertTrue(all(result is not None for result in simulations))
        self.assertEqual(
            tuple(result.times.size for result in simulations if result is not None),
            (3, 4, 5),
        )

    def test_adjustment_returns_each_endpoint_and_rejects_failed_gl2(self) -> None:
        config = _config(nx=8, maximum_wavenumber=2.0)
        grids = tuple(
            np.arange(count, dtype=np.float64) * config.saved_dt for count in (3, 4, 5)
        )
        eta0 = np.repeat(np.arange(1.0, 4.0)[:, None], config.nx, axis=1)
        ramp_times = np.asarray([0.4, 0.6, 0.8])
        calls: list[tuple[np.ndarray, int]] = []

        def integrate(
            *,
            eta0: np.ndarray,
            xi0: np.ndarray,
            depths: np.ndarray,
            saved_times: np.ndarray,
            config: RolloutNumerics,
            nonlinear_ramp_times: np.ndarray,
            nonlinear_ramp_order: int,
        ) -> IntegratedAdjustmentBatch:
            del depths, config
            calls.append(
                (
                    nonlinear_ramp_times.copy(),
                    nonlinear_ramp_order,
                )
            )
            eta = np.broadcast_to(
                eta0,
                (saved_times.size, *eta0.shape),
            ).copy()
            xi = np.broadcast_to(
                xi0,
                (saved_times.size, *xi0.shape),
            ).copy()
            eta += saved_times[:, None, None]
            xi -= saved_times[:, None, None]
            converged = np.ones((saved_times.size - 1, eta0.shape[0]), dtype=np.bool_)
            converged[0, 2] = False
            return IntegratedAdjustmentBatch(eta, xi, converged)

        with patch(
            "solver.gen_data.pipeline.trajectory_rollout.integrate_adjustment_batch",
            new=integrate,
        ):
            simulations = execute_adjustment_batch(
                eta0,
                np.zeros_like(eta0),
                np.full(3, 10.0),
                grids,
                nonlinear_ramp_times=ramp_times,
                nonlinear_ramp_order=4,
                config=config,
            )

        np.testing.assert_array_equal(calls[0][0], ramp_times)
        self.assertEqual(calls[0][1], 4)
        self.assertEqual(
            tuple(result is not None for result in simulations),
            (True, True, False),
        )
        for index, expected in enumerate((1.04, 2.06)):
            endpoint = simulations[index]
            assert endpoint is not None
            terminal_eta, terminal_xi = endpoint
            np.testing.assert_allclose(terminal_eta, expected)
            np.testing.assert_allclose(terminal_xi, -grids[index][-1])
        self.assertIsNone(simulations[2])


if __name__ == "__main__":
    unittest.main()
