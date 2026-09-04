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
BoolArray: TypeAlias = NDArray[np.bool_]


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


def _validate_initial_conditions(
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    *,
    nx: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    eta = np.asarray(eta0, dtype=np.float64)
    xi = np.asarray(xi0, dtype=np.float64)
    depth_values = np.asarray(depths, dtype=np.float64)
    if eta.ndim != 2 or eta.shape[-1] != nx:
        raise ValueError(f"eta0 must have shape (batch, {nx}), got {eta.shape}")
    if xi.shape != eta.shape:
        raise ValueError("xi0 must have the same shape as eta0")
    if depth_values.shape != (eta.shape[0],):
        raise ValueError(f"depths must have shape ({eta.shape[0]},)")
    if not np.isfinite(depth_values).all() or np.any(depth_values <= 0.0):
        raise ValueError("depths must be finite and positive")
    return eta, xi, depth_values


def _prepare_time_grids(
    time_grids: tuple[FloatArray, ...],
    *,
    batch_size: int,
) -> tuple[tuple[FloatArray, ...], FloatArray]:
    if len(time_grids) != batch_size:
        raise ValueError("time_grids must contain one grid per input simulation")
    grids = tuple(np.asarray(times, dtype=np.float64) for times in time_grids)
    integration_times = max(grids, key=len)
    if any(
        not np.array_equal(times, integration_times[: times.size]) for times in grids
    ):
        raise ValueError("every time grid must be a prefix of the longest grid")
    return grids, integration_times


def _all_gl2_steps_converged(
    convergence: BoolArray,
    simulation_index: int,
    saved_time_count: int,
    substeps_per_saved_frame: int,
) -> bool:
    step_count = (saved_time_count - 1) * substeps_per_saved_frame
    return bool(np.all(convergence[:step_count, simulation_index]))


def execute_trajectory_batch(
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    time_grids: tuple[FloatArray, ...],
    *,
    config: RolloutNumerics,
) -> tuple[TrajectorySamples | None, ...]:
    """Integrate a batch, then evaluate each requested trajectory prefix."""

    eta, xi, depth_values = _validate_initial_conditions(
        eta0,
        xi0,
        depths,
        nx=config.nx,
    )
    grids, integration_times = _prepare_time_grids(
        time_grids,
        batch_size=eta.shape[0],
    )
    rollout = integrate_batch(
        eta0=eta,
        xi0=xi,
        depths=depth_values,
        saved_times=integration_times,
        config=config,
    )
    field_shape = (integration_times.size, eta.shape[0], config.target_nx)
    if any(
        np.shape(field) != field_shape
        for field in (rollout.eta, rollout.xi, rollout.gxi)
    ):
        raise ValueError(f"trajectory fields must have shape {field_shape}")
    step_shape = (
        (integration_times.size - 1) * config.substeps_per_saved_frame,
        eta.shape[0],
    )
    if np.shape(rollout.gl2_converged) != step_shape:
        raise ValueError(f"GL2 convergence must have shape {step_shape}")
    health = rollout.solver_grid_health
    health_shape = (integration_times.size, eta.shape[0])
    if config.internal_hamiltonian_drift_threshold is not None and (
        health is None or any(np.shape(field) != health_shape for field in health)
    ):
        raise ValueError(f"solver-grid health fields must have shape {health_shape}")

    results: list[TrajectorySamples | None] = []
    for index, saved_times in enumerate(grids):
        saved_count = saved_times.size
        trajectory = TrajectorySamples(
            np.asarray(saved_times, dtype=np.float64),
            np.asarray(rollout.eta[:saved_count, index], dtype=np.float64),
            np.asarray(rollout.xi[:saved_count, index], dtype=np.float64),
            np.asarray(rollout.gxi[:saved_count, index], dtype=np.float64),
        )
        state_finite = bool(
            np.isfinite(trajectory.eta).all() and np.isfinite(trajectory.xi).all()
        )
        target_finite = bool(np.isfinite(trajectory.gxi).all())
        nonpositive_water_height = (
            state_finite and float(np.min(depth_values[index] + trajectory.eta)) <= 0.0
        )
        integration_failure = not _all_gl2_steps_converged(
            rollout.gl2_converged,
            index,
            saved_count,
            config.substeps_per_saved_frame,
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

    eta, xi, depth_values = _validate_initial_conditions(
        eta0,
        xi0,
        depths,
        nx=config.nx,
    )
    grids, integration_times = _prepare_time_grids(
        time_grids,
        batch_size=eta.shape[0],
    )
    ramp_times = np.asarray(nonlinear_ramp_times, dtype=np.float64)
    if ramp_times.shape != depth_values.shape:
        raise ValueError(f"nonlinear_ramp_times must have shape {depth_values.shape}")
    if not np.isfinite(ramp_times).all() or np.any(ramp_times <= 0.0):
        raise ValueError("nonlinear_ramp_times must be finite and positive")
    if nonlinear_ramp_order < 1:
        raise ValueError("nonlinear_ramp_order must be positive")

    rollout = integrate_adjustment_batch(
        eta0=eta,
        xi0=xi,
        depths=depth_values,
        saved_times=integration_times,
        config=config,
        nonlinear_ramp_times=ramp_times,
        nonlinear_ramp_order=nonlinear_ramp_order,
    )
    field_shape = (integration_times.size, eta.shape[0], config.nx)
    if any(np.shape(field) != field_shape for field in (rollout.eta, rollout.xi)):
        raise ValueError(f"adjustment fields must have shape {field_shape}")
    step_shape = (
        (integration_times.size - 1) * config.substeps_per_saved_frame,
        eta.shape[0],
    )
    if np.shape(rollout.gl2_converged) != step_shape:
        raise ValueError(f"GL2 convergence must have shape {step_shape}")

    results: list[tuple[FloatArray, FloatArray] | None] = []
    for index, saved_times in enumerate(grids):
        saved_count = saved_times.size
        trajectory_eta = np.asarray(rollout.eta[:saved_count, index], dtype=np.float64)
        trajectory_xi = np.asarray(rollout.xi[:saved_count, index], dtype=np.float64)
        state_finite = bool(
            np.isfinite(trajectory_eta).all() and np.isfinite(trajectory_xi).all()
        )
        nonpositive_water_height = (
            state_finite and float(np.min(depth_values[index] + trajectory_eta)) <= 0.0
        )
        integration_failure = not _all_gl2_steps_converged(
            rollout.gl2_converged,
            index,
            saved_count,
            config.substeps_per_saved_frame,
        )
        results.append(
            (trajectory_eta[-1], trajectory_xi[-1])
            if state_finite and not nonpositive_water_height and not integration_failure
            else None
        )
    return tuple(results)
