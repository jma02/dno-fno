"""Behavioral tests for grouped JONSWAP nonlinear adjustment."""

from __future__ import annotations

from collections.abc import Callable
import math
import os
import unittest
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

from solver.gen_data.jonswap_horizon_generator import (  # noqa: E402
    integrate_and_subsample_jonswap,
)
from solver.gen_data.pipeline.trajectory_config import (  # noqa: E402
    PAPER_ROLLOUT_NUMERICS,
    RolloutNumerics,
)
from solver.gen_data.pipeline.trajectory_integration import (  # noqa: E402
    IntegratedTrajectoryBatch,
    SolverGridHealth,
)
from solver.gen_data.trajectory_family_adapters import (  # noqa: E402
    TrajectoryInitialBatch,
)


def _config(
    *, internal_hamiltonian_drift_threshold: float | None = None
) -> RolloutNumerics:
    return RolloutNumerics(
        nx=64,
        target_nx=64,
        length=2.0 * math.pi,
        gravity=1.0,
        integration_dno_order=0,
        label_dno_order=0,
        pad_factor=1,
        maximum_wavenumber=16.0,
        target_maximum_wavenumber=16.0,
        saved_dt=0.08,
        substeps_per_saved_frame=2,
        gl2_residual_tolerance=1.0e-8,
        gl2_iteration_cap=2,
        internal_hamiltonian_drift_threshold=internal_hamiltonian_drift_threshold,
    )


def _grid(saved_count: int) -> np.ndarray:
    return 0.08 * np.arange(saved_count, dtype=np.float64)


def _production_integrator(
    *, hamiltonian_drift: float | None = None
) -> tuple[
    Callable[..., IntegratedTrajectoryBatch],
    list[tuple[int, int]],
    list[np.ndarray],
    list[np.ndarray],
]:
    calls: list[tuple[int, int]] = []
    initial_eta: list[np.ndarray] = []
    time_grids: list[np.ndarray] = []

    def integrate(
        *,
        eta0: np.ndarray,
        xi0: np.ndarray,
        depths: np.ndarray,
        saved_times: np.ndarray,
        config: RolloutNumerics,
    ) -> IntegratedTrajectoryBatch:
        del depths
        batch_size, internal_nx = eta0.shape
        saved_count = saved_times.size
        calls.append((batch_size, saved_count))
        initial_eta.append(np.asarray(eta0, dtype=np.float64).copy())
        time_grids.append(np.asarray(saved_times, dtype=np.float64).copy())
        if config.target_nx == internal_nx:
            delivered_eta0, delivered_xi0 = eta0, xi0
        else:
            delivered_eta0 = np.repeat(
                np.mean(eta0, axis=-1)[:, None], config.target_nx, axis=1
            )
            delivered_xi0 = np.repeat(
                np.mean(xi0, axis=-1)[:, None], config.target_nx, axis=1
            )
        internal = None
        if hamiltonian_drift is not None:
            hamiltonian = np.ones((saved_count, batch_size), dtype=np.float64)
            hamiltonian[-1] += hamiltonian_drift
            internal = SolverGridHealth(
                hamiltonian,
                np.ones_like(hamiltonian, dtype=np.bool_),
                np.ones_like(hamiltonian, dtype=np.bool_),
                np.ones_like(hamiltonian),
            )
        return IntegratedTrajectoryBatch(
            np.repeat(delivered_eta0[None], saved_count, axis=0),
            np.repeat(delivered_xi0[None], saved_count, axis=0),
            np.zeros((saved_count, batch_size, config.target_nx), dtype=np.float64),
            np.ones(
                (
                    (saved_count - 1) * config.substeps_per_saved_frame,
                    batch_size,
                ),
                dtype=np.bool_,
            ),
            internal,
        )

    return integrate, calls, initial_eta, time_grids


def _adjustment_integrator(
    *,
    failing_markers: tuple[float, ...] = (),
    terminal_addition: np.ndarray | None = None,
) -> tuple[
    Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
    list[tuple[int, int, np.ndarray, int]],
]:
    calls: list[tuple[int, int, np.ndarray, int]] = []

    def integrate(
        *,
        eta0: np.ndarray,
        xi0: np.ndarray,
        depths: np.ndarray,
        saved_times: np.ndarray,
        config: RolloutNumerics,
        nonlinear_ramp_times: np.ndarray,
        nonlinear_ramp_order: int,
        saved_time_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del depths, config
        batch_size, nx = eta0.shape
        calls.append(
            (
                batch_size,
                saved_times.size,
                np.asarray(nonlinear_ramp_times).copy(),
                nonlinear_ramp_order,
            )
        )
        eta = eta0 + saved_times[saved_time_counts - 1, None]
        if terminal_addition is not None:
            eta += terminal_addition[None, :]
        accepted = np.ones(batch_size, dtype=np.bool_)
        markers = eta0[:, 0]
        for marker in failing_markers:
            accepted[np.isclose(markers, marker, rtol=0.0, atol=1.0e-14)] = False
        return eta, xi0.copy(), accepted

    return integrate, calls


class JonswapHorizonGeneratorTests(unittest.TestCase):
    def test_each_adjustment_uses_its_peak_period_and_own_endpoint(self) -> None:
        numerical = _config()
        production, _, handed_off, production_times = _production_integrator()
        adjustment, adjustment_calls = _adjustment_integrator()
        markers = np.asarray((0.01, 0.02, 0.03), dtype=np.float64)
        peak_periods = np.asarray((0.31, 0.43, 0.57), dtype=np.float64)
        initial = TrajectoryInitialBatch(
            np.repeat(markers[:, None], numerical.nx, axis=1),
            np.zeros((3, numerical.nx), dtype=np.float64),
            np.ones(3, dtype=np.float64),
        )

        with (
            mock.patch(
                "solver.gen_data.pipeline.trajectory_rollout.integrate_batch",
                new=production,
            ),
            mock.patch(
                "solver.gen_data.pipeline.trajectory_rollout.integrate_adjustment_batch",
                new=adjustment,
            ),
        ):
            rows_by_simulation = integrate_and_subsample_jonswap(
                initial,
                (_grid(200),) * 3,
                peak_periods,
                numerical=numerical,
                solver_batch_size=3,
            )
            empty_rows = integrate_and_subsample_jonswap(
                TrajectoryInitialBatch(*(field[:0] for field in initial)),
                (),
                peak_periods[:0],
                numerical=numerical,
                solver_batch_size=3,
            )

        self.assertEqual(empty_rows, ())
        self.assertEqual(len(adjustment_calls), 1)
        realized = (
            np.floor(20.0 * peak_periods / numerical.saved_dt) * numerical.saved_dt
        )
        np.testing.assert_allclose(handed_off[0][:, 0], markers + realized)
        np.testing.assert_allclose(adjustment_calls[0][2], 10.0 * peak_periods)
        self.assertEqual(adjustment_calls[0][3], 4)
        self.assertEqual(float(production_times[0][0]), 0.0)
        self.assertTrue(all(rows is not None for rows in rows_by_simulation))

    def test_full_band_adjustment_endpoint_reaches_production_unprojected(self) -> None:
        numerical = PAPER_ROLLOUT_NUMERICS["jonswap_tma"]
        x = 2.0 * math.pi * np.arange(numerical.nx) / numerical.nx
        high_mode = 0.001 * np.cos(600.0 * x)
        production, _, handed_off, _ = _production_integrator(hamiltonian_drift=0.0)
        adjustment, _ = _adjustment_integrator(terminal_addition=high_mode)
        initial = TrajectoryInitialBatch(
            np.full((1, numerical.nx), 0.01, dtype=np.float64),
            np.zeros((1, numerical.nx), dtype=np.float64),
            np.ones(1, dtype=np.float64),
        )

        with (
            mock.patch(
                "solver.gen_data.pipeline.trajectory_rollout.integrate_batch",
                new=production,
            ),
            mock.patch(
                "solver.gen_data.pipeline.trajectory_rollout.integrate_adjustment_batch",
                new=adjustment,
            ),
        ):
            (rows,) = integrate_and_subsample_jonswap(
                initial,
                (_grid(200),),
                np.asarray([0.4]),
                numerical=numerical,
                solver_batch_size=1,
            )

        coefficient = np.fft.rfft(handed_off[0][0] - np.mean(handed_off[0][0]))[600]
        self.assertGreater(abs(coefficient), 0.4)
        self.assertEqual(handed_off[0].shape, (1, 2048))
        assert rows is not None
        self.assertEqual(rows.eta.shape[-1], 1024)

    def test_stable_groups_skip_burn_failures_and_restore_input_order(self) -> None:
        numerical = _config()
        production, production_calls, _, _ = _production_integrator()
        adjustment, adjustment_calls = _adjustment_integrator(
            failing_markers=(0.02, 0.05)
        )
        markers = np.arange(1, 6, dtype=np.float64) / 100.0
        peak_periods = np.asarray((0.31, 0.43, 0.57, 0.61, 0.73), dtype=np.float64)
        initial = TrajectoryInitialBatch(
            np.repeat(markers[:, None], numerical.nx, axis=1),
            np.zeros((5, numerical.nx), dtype=np.float64),
            np.ones(5, dtype=np.float64),
        )

        with (
            mock.patch(
                "solver.gen_data.pipeline.trajectory_rollout.integrate_batch",
                new=production,
            ),
            mock.patch(
                "solver.gen_data.pipeline.trajectory_rollout.integrate_adjustment_batch",
                new=adjustment,
            ),
        ):
            rows_by_simulation = integrate_and_subsample_jonswap(
                initial,
                tuple(map(_grid, (203, 200, 202, 200, 204))),
                peak_periods,
                numerical=numerical,
                solver_batch_size=2,
            )

        self.assertEqual(
            [rows is not None for rows in rows_by_simulation],
            [True, False, True, True, False],
        )
        self.assertEqual([call[0] for call in adjustment_calls], [2, 2, 1])
        np.testing.assert_allclose(
            np.concatenate([call[2] for call in adjustment_calls]),
            10.0 * peak_periods[[1, 3, 2, 0, 4]],
        )
        self.assertEqual(production_calls, [(1, 200), (2, 203)])
        realized = (
            np.floor(20.0 * peak_periods / numerical.saved_dt) * numerical.saved_dt
        )
        np.testing.assert_allclose(
            [rows.eta[0, 0] for rows in rows_by_simulation if rows is not None],
            (markers + realized)[[0, 2, 3]],
        )

    def test_production_health_gate_runs_after_an_accepted_burn(self) -> None:
        numerical = _config(internal_hamiltonian_drift_threshold=1.0e-3)
        production, _, _, _ = _production_integrator(hamiltonian_drift=1.0e-2)
        adjustment, _ = _adjustment_integrator()
        initial = TrajectoryInitialBatch(
            np.full((1, numerical.nx), 0.01, dtype=np.float64),
            np.zeros((1, numerical.nx), dtype=np.float64),
            np.ones(1, dtype=np.float64),
        )

        with (
            mock.patch(
                "solver.gen_data.pipeline.trajectory_rollout.integrate_batch",
                new=production,
            ),
            mock.patch(
                "solver.gen_data.pipeline.trajectory_rollout.integrate_adjustment_batch",
                new=adjustment,
            ),
        ):
            (rows,) = integrate_and_subsample_jonswap(
                initial,
                (_grid(200),),
                np.asarray([0.4]),
                numerical=numerical,
                solver_batch_size=1,
            )

        self.assertIsNone(rows)


if __name__ == "__main__":
    unittest.main()
