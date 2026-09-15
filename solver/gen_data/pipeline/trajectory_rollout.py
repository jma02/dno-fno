"""Integrate wave trajectories and retain numerically valid simulations."""

from __future__ import annotations

from typing import NamedTuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.trajectory_config import RolloutNumerics
from solver.gen_data.pipeline.trajectory_integration import (
    integrate_adjustment_batch,
    integrate_batch,
)

FloatArray: TypeAlias = NDArray[np.float64]


# Saved times, states, and DNO targets for one trajectory.
TrajectorySamples = NamedTuple(
    "TrajectorySamples",
    [
        ("times", FloatArray),
        ("eta", FloatArray),
        ("xi", FloatArray),
        ("gxi", FloatArray),
    ],
)


def execute_trajectory_batch(
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    time_grids: tuple[FloatArray, ...],
    *,
    config: RolloutNumerics,
) -> tuple[TrajectorySamples | None, ...]:
    """Integrate a batch, then evaluate each requested trajectory prefix."""

    integration_times = max(time_grids, key=len)
    rollout = integrate_batch(
        eta0=eta0,
        xi0=xi0,
        depths=depths,
        saved_times=integration_times,
        config=config,
    )
    health = rollout.solver_grid_health

    results: list[TrajectorySamples | None] = []
    for index, saved_times in enumerate(time_grids):
        saved_count = saved_times.size
        trajectory = TrajectorySamples(
            saved_times,
            rollout.eta[:saved_count, index],
            rollout.xi[:saved_count, index],
            rollout.gxi[:saved_count, index],
        )
        state_finite = bool(
            np.isfinite(trajectory.eta).all() and np.isfinite(trajectory.xi).all()
        )
        target_finite = bool(np.isfinite(trajectory.gxi).all())
        nonpositive_water_height = (
            state_finite and float(np.min(depths[index] + trajectory.eta)) <= 0.0
        )
        step_count = (saved_count - 1) * config.substeps_per_saved_frame
        integration_failure = not bool(
            np.all(rollout.gl2_converged[:step_count, index])
        )
        hamiltonian_drift = False

        if (
            config.internal_hamiltonian_drift_threshold is not None
            and health is not None
        ):
            internal_state_finite = bool(
                np.all(health.state_finite[:saved_count, index])
            )
            internal_target_finite = bool(
                np.all(health.dno_output_finite[:saved_count, index])
            )
            water_column = health.minimum_water_column[:saved_count, index]
            internal_nonpositive_water_height = (
                not np.isfinite(water_column).all()
                or float(np.min(water_column)) <= 0.0
            )
            hamiltonian = health.hamiltonian[:saved_count, index]
            hamiltonian_drift = not np.isfinite(hamiltonian).all()
            if not hamiltonian_drift:
                initial_hamiltonian = float(hamiltonian[0])
                hamiltonian_drift = bool(
                    np.max(
                        np.abs(hamiltonian - initial_hamiltonian)
                        / max(abs(initial_hamiltonian), np.finfo(np.float64).tiny)
                    )
                    > config.internal_hamiltonian_drift_threshold
                )
            state_finite = state_finite and internal_state_finite
            target_finite = target_finite and internal_target_finite
            nonpositive_water_height = (
                nonpositive_water_height or internal_nonpositive_water_height
            )

        results.append(
            trajectory
            if state_finite
            and target_finite
            and not nonpositive_water_height
            and not hamiltonian_drift
            and not integration_failure
            else None
        )
    return tuple(results)


def execute_adjustment_batch(
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    time_grids: tuple[FloatArray, ...],
    *,
    nonlinear_ramp_times: FloatArray,
    nonlinear_ramp_order: int,
    config: RolloutNumerics,
) -> tuple[tuple[FloatArray, FloatArray] | None, ...]:
    """Warm up JONSWAP simulations and return valid nonlinear endpoints."""

    integration_times = max(time_grids, key=len)

    eta, xi, accepted = integrate_adjustment_batch(
        eta0=eta0,
        xi0=xi0,
        depths=depths,
        saved_times=integration_times,
        config=config,
        nonlinear_ramp_times=nonlinear_ramp_times,
        nonlinear_ramp_order=nonlinear_ramp_order,
        saved_time_counts=np.fromiter(
            (times.size for times in time_grids),
            dtype=np.int32,
            count=len(time_grids),
        ),
    )
    return tuple(
        (eta[index].copy(), xi[index].copy()) if is_accepted else None
        for index, is_accepted in enumerate(accepted)
    )
