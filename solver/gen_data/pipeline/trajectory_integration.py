"""Run batched GL2 wave integrations and construct DNO training targets."""

from __future__ import annotations

import math
from typing import NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.dno_target import compute_dno_target
from solver.gen_data.pipeline.trajectory_config import RolloutNumerics
from solver.solvers.dno_series_jax import build_grid, make_linear_dno_symbol
from solver.solvers.time_integrator import (
    SolverParams,
    State,
    rollout as integrate_trajectory,
)

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


# Acceptance data evaluated on the full solver grid.
SolverGridHealth = NamedTuple(
    "SolverGridHealth",
    [
        ("hamiltonian", FloatArray),
        ("state_finite", BoolArray),
        ("dno_output_finite", BoolArray),
        ("minimum_water_column", FloatArray),
    ],
)

# Projected trajectory fields and their numerical health data.
IntegratedTrajectoryBatch = NamedTuple(
    "IntegratedTrajectoryBatch",
    [
        ("eta", FloatArray),
        ("xi", FloatArray),
        ("gxi", FloatArray),
        ("gl2_converged", BoolArray),
        ("solver_grid_health", SolverGridHealth | None),
    ],
)

# States and GL2 convergence from a JONSWAP nonlinear warm-up.
IntegratedAdjustmentBatch = NamedTuple(
    "IntegratedAdjustmentBatch",
    [
        ("eta", FloatArray),
        ("xi", FloatArray),
        ("gl2_converged", BoolArray),
    ],
)


def resample_to_target_grid(field: jax.Array, *, config: RolloutNumerics) -> jax.Array:
    """Keep the target Fourier band and resample it onto the target grid."""

    source = jnp.asarray(field, dtype=jnp.float64)
    maximum_mode = int(
        math.floor(
            config.target_maximum_wavenumber * config.length / (2.0 * math.pi) + 1.0e-12
        )
    )
    source_coefficients = jnp.fft.rfft(source, axis=-1) / config.nx
    target_coefficients = jnp.zeros(
        (*source.shape[:-1], config.target_nx // 2 + 1),
        dtype=source_coefficients.dtype,
    )
    target_coefficients = target_coefficients.at[..., : maximum_mode + 1].set(
        source_coefficients[..., : maximum_mode + 1]
    )
    return jnp.fft.irfft(
        target_coefficients * config.target_nx,
        n=config.target_nx,
        axis=-1,
    )


def _integrate_gl2(
    eta: FloatArray,
    xi: FloatArray,
    depths: FloatArray,
    saved_times: FloatArray,
    config: RolloutNumerics,
    *,
    nonlinear_ramp_times: FloatArray | None = None,
    nonlinear_ramp_order: int = 4,
) -> tuple[dict[str, jax.Array], jax.Array]:
    depth_device = jnp.asarray(depths, dtype=jnp.float64)
    _, wavenumbers = build_grid(config.nx, config.length)
    k = jnp.asarray(wavenumbers, dtype=jnp.float64)
    depth_column = depth_device[:, None]
    payload = integrate_trajectory(
        State(
            eta=jnp.asarray(eta, dtype=jnp.float64),
            xi=jnp.asarray(xi, dtype=jnp.float64),
        ),
        jnp.asarray(saved_times, dtype=jnp.float64),
        SolverParams(
            nx=config.nx,
            length=config.length,
            depth=depth_column,
            gravity=config.gravity,
            dno_order=config.integration_dno_order,
            pad_factor=config.pad_factor,
            filter_fraction=(
                config.maximum_wavenumber / (math.pi * config.nx / config.length)
            ),
            k=k,
            g0=make_linear_dno_symbol(k, depth_column),
            nonlinear_ramp_time=(
                None
                if nonlinear_ramp_times is None
                else jnp.asarray(nonlinear_ramp_times, dtype=jnp.float64)
            ),
            nonlinear_ramp_order=nonlinear_ramp_order,
        ),
        save_gxi=False,
        substeps_per_interval=config.substeps_per_saved_frame,
        method="gl2_if",
        implicit_iterations=config.gl2_iteration_cap,
        implicit_residual_tolerance=config.gl2_residual_tolerance,
        implicit_relaxation=1.0,
        zero_mean_xi=True,
    )
    return payload, depth_device


def integrate_batch(
    *,
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    saved_times: FloatArray,
    config: RolloutNumerics,
) -> IntegratedTrajectoryBatch:
    """Integrate trajectories and compute stored targets in time chunks."""

    payload, depth_device = _integrate_gl2(
        eta0,
        xi0,
        depths,
        saved_times,
        config,
    )
    eta = payload["eta"]
    xi = payload["xi"]
    delivered_shape = (*eta.shape[:-1], config.target_nx)
    eta_result = np.empty(delivered_shape, dtype=np.float64)
    xi_result = np.empty(delivered_shape, dtype=np.float64)
    gxi_result = np.empty(delivered_shape, dtype=np.float64)
    health_shape = eta.shape[:2]
    hamiltonian_result = np.empty(health_shape, dtype=np.float64)
    state_finite_result = np.empty(health_shape, dtype=np.bool_)
    dno_finite_result = np.empty(health_shape, dtype=np.bool_)
    water_column_result = np.empty(health_shape, dtype=np.float64)
    evaluate_health = config.internal_hamiltonian_drift_threshold is not None

    chunk_size = 8
    for start in range(0, eta.shape[0], chunk_size):
        stop = min(start + chunk_size, eta.shape[0])
        count = stop - start
        padding = ((0, chunk_size - count), (0, 0), (0, 0))
        eta_chunk = jnp.pad(eta[start:stop], padding)
        xi_chunk = jnp.pad(xi[start:stop], padding)
        target_eta = eta_chunk
        target_xi = xi_chunk
        if config.target_nx != config.nx:
            target_eta = resample_to_target_grid(eta_chunk, config=config)
            target_xi = resample_to_target_grid(xi_chunk, config=config)
        target_eta, target_xi, target_gxi = compute_dno_target(
            target_eta,
            target_xi,
            depth_device[:, None],
            nx=config.target_nx,
            length=config.length,
            dno_order=config.label_dno_order,
            pad_factor=config.pad_factor,
            maximum_wavenumber=config.target_maximum_wavenumber,
        )
        eta_host, xi_host, gxi_host = jax.device_get(
            (target_eta[:count], target_xi[:count], target_gxi[:count])
        )
        eta_result[start:stop] = np.asarray(eta_host, dtype=np.float64)
        xi_result[start:stop] = np.asarray(xi_host, dtype=np.float64)
        gxi_result[start:stop] = np.asarray(gxi_host, dtype=np.float64)

        if evaluate_health:
            _, _, internal_gxi = compute_dno_target(
                eta_chunk,
                xi_chunk,
                depth_device[:, None],
                nx=config.nx,
                length=config.length,
                dno_order=config.integration_dno_order,
                pad_factor=config.pad_factor,
                maximum_wavenumber=config.maximum_wavenumber,
            )
            health = jax.device_get(
                (
                    0.5
                    * (config.length / config.nx)
                    * jnp.sum(
                        xi_chunk * internal_gxi + config.gravity * eta_chunk**2,
                        axis=-1,
                    ),
                    jnp.all(jnp.isfinite(eta_chunk) & jnp.isfinite(xi_chunk), axis=-1),
                    jnp.all(jnp.isfinite(internal_gxi), axis=-1),
                    jnp.min(
                        eta_chunk + depth_device[None, :, None],
                        axis=-1,
                    ),
                )
            )
            hamiltonian_result[start:stop] = np.asarray(
                health[0][:count], dtype=np.float64
            )
            state_finite_result[start:stop] = np.asarray(
                health[1][:count], dtype=np.bool_
            )
            dno_finite_result[start:stop] = np.asarray(
                health[2][:count], dtype=np.bool_
            )
            water_column_result[start:stop] = np.asarray(
                health[3][:count], dtype=np.float64
            )

    solver_grid_health = (
        SolverGridHealth(
            hamiltonian_result,
            state_finite_result,
            dno_finite_result,
            water_column_result,
        )
        if evaluate_health
        else None
    )
    return IntegratedTrajectoryBatch(
        eta_result,
        xi_result,
        gxi_result,
        np.asarray(jax.device_get(payload["gl2_converged"]), dtype=np.bool_),
        solver_grid_health,
    )


def integrate_adjustment_batch(
    *,
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    saved_times: FloatArray,
    config: RolloutNumerics,
    nonlinear_ramp_times: FloatArray,
    nonlinear_ramp_order: int,
) -> IntegratedAdjustmentBatch:
    """Warm up a JONSWAP batch by gradually enabling nonlinearity."""

    payload, _ = _integrate_gl2(
        eta0,
        xi0,
        depths,
        saved_times,
        config,
        nonlinear_ramp_times=nonlinear_ramp_times,
        nonlinear_ramp_order=nonlinear_ramp_order,
    )
    eta, xi, converged = jax.device_get(
        (payload["eta"], payload["xi"], payload["gl2_converged"])
    )
    return IntegratedAdjustmentBatch(
        np.asarray(eta, dtype=np.float64),
        np.asarray(xi, dtype=np.float64),
        np.asarray(converged, dtype=np.bool_),
    )
