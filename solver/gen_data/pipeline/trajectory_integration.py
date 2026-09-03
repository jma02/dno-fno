"""Run batched GL2 wave integrations and construct DNO training targets."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.dno_target import compute_dno_target
from solver.gen_data.pipeline.trajectory_config import RolloutConfig
from solver.solvers.dno_series_jax import build_grid, make_linear_dno_symbol
from solver.solvers.time_integrator import (
    SolverParams,
    State,
    rollout as integrate_trajectory,
)

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True)
class GL2BatchTelemetry:
    """Numerical health arrays with shape ``(substep, batch)``."""

    stage_residual: FloatArray
    converged: BoolArray
    stage_finite: BoolArray
    state_finite: BoolArray


@dataclass(frozen=True)
class InternalHealthTelemetry:
    """Health checks on the full solver grid before target projection."""

    hamiltonian: FloatArray
    state_finite: BoolArray
    dno_output_finite: BoolArray
    minimum_water_column: FloatArray


@dataclass(frozen=True)
class IntegratedTrajectoryBatch:
    """Projected trajectory fields and their solver diagnostics."""

    eta: FloatArray
    xi: FloatArray
    gxi: FloatArray
    gl2: GL2BatchTelemetry
    internal_telemetry: InternalHealthTelemetry | None = None


@dataclass(frozen=True)
class IntegratedAdjustmentBatch:
    """States and solver diagnostics from a JONSWAP nonlinear warm-up."""

    eta: FloatArray
    xi: FloatArray
    gl2: GL2BatchTelemetry


class BatchIntegrator(Protocol):
    """Callable that integrates one trajectory batch."""

    def __call__(
        self,
        *,
        eta0: FloatArray,
        xi0: FloatArray,
        depths: FloatArray,
        saved_times: FloatArray,
        config: RolloutConfig,
    ) -> IntegratedTrajectoryBatch: ...


class AdjustmentBatchIntegrator(Protocol):
    """Callable that performs one JONSWAP nonlinear warm-up."""

    def __call__(
        self,
        *,
        eta0: FloatArray,
        xi0: FloatArray,
        depths: FloatArray,
        saved_times: FloatArray,
        config: RolloutConfig,
        nonlinear_ramp_times: FloatArray,
        nonlinear_ramp_order: int,
    ) -> IntegratedAdjustmentBatch: ...


def resample_to_target_grid(field: jax.Array, *, config: RolloutConfig) -> jax.Array:
    """Keep the target Fourier band and resample it onto the target grid."""

    source = jnp.asarray(field, dtype=jnp.float64)
    if source.shape[-1] != config.nx:
        raise ValueError(f"internal field must end in {config.nx} grid points")
    maximum_mode = int(
        math.floor(
            config.target_maximum_wavenumber * config.length / (2.0 * math.pi) + 1.0e-12
        )
    )
    if maximum_mode >= min(config.nx, config.target_nx) // 2:
        raise ValueError("delivered band must lie below both grid Nyquists")
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


def _compute_solver_grid_diagnostics(
    eta: jax.Array,
    xi: jax.Array,
    depths: jax.Array,
    config: RolloutConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    _, _, gxi = compute_dno_target(
        eta,
        xi,
        depths[:, None],
        nx=config.nx,
        length=config.length,
        dno_order=config.dno_order,
        pad_factor=config.pad_factor,
        maximum_wavenumber=config.maximum_wavenumber,
    )
    state_finite = jnp.all(jnp.isfinite(eta) & jnp.isfinite(xi), axis=-1)
    dno_output_finite = jnp.all(jnp.isfinite(gxi), axis=-1)
    minimum_water_column = jnp.min(eta + depths[None, :, None], axis=-1)
    hamiltonian = (
        0.5
        * (config.length / config.nx)
        * jnp.sum(xi * gxi + config.gravity * eta**2, axis=-1)
    )
    return hamiltonian, state_finite, dno_output_finite, minimum_water_column


def compute_saved_targets(
    eta: jax.Array,
    xi: jax.Array,
    depths: jax.Array,
    config: RolloutConfig,
) -> tuple[FloatArray, FloatArray, FloatArray, InternalHealthTelemetry | None]:
    """Project saved states and compute their DNO targets in time chunks."""

    delivered_shape = (*eta.shape[:-1], config.target_nx)
    eta_result = np.empty(delivered_shape, dtype=np.float64)
    xi_result = np.empty(delivered_shape, dtype=np.float64)
    gxi_result = np.empty(delivered_shape, dtype=np.float64)
    evaluate_health = config.internal_hamiltonian_drift_threshold is not None
    health_shape = eta.shape[:2]
    hamiltonian_result = np.empty(health_shape, dtype=np.float64)
    state_finite_result = np.empty(health_shape, dtype=np.bool_)
    dno_finite_result = np.empty(health_shape, dtype=np.bool_)
    water_column_result = np.empty(health_shape, dtype=np.float64)

    chunk_size = config.target_time_chunk_size
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
            depths[:, None],
            nx=config.target_nx,
            length=config.length,
            dno_order=config.target_dno_order,
            pad_factor=config.pad_factor,
            maximum_wavenumber=config.target_maximum_wavenumber,
        )
        stop = start + count
        eta_host, xi_host, gxi_host = jax.device_get(
            (target_eta[:count], target_xi[:count], target_gxi[:count])
        )
        eta_result[start:stop] = np.asarray(eta_host, dtype=np.float64)
        xi_result[start:stop] = np.asarray(xi_host, dtype=np.float64)
        gxi_result[start:stop] = np.asarray(gxi_host, dtype=np.float64)
        if evaluate_health:
            health = jax.device_get(
                tuple(
                    value[:count]
                    for value in _compute_solver_grid_diagnostics(
                        eta_chunk,
                        xi_chunk,
                        depths,
                        config,
                    )
                )
            )
            hamiltonian_result[start:stop] = np.asarray(health[0], dtype=np.float64)
            state_finite_result[start:stop] = np.asarray(health[1], dtype=np.bool_)
            dno_finite_result[start:stop] = np.asarray(health[2], dtype=np.bool_)
            water_column_result[start:stop] = np.asarray(health[3], dtype=np.float64)

    internal_telemetry = None
    if evaluate_health:
        internal_telemetry = InternalHealthTelemetry(
            hamiltonian=hamiltonian_result,
            state_finite=state_finite_result,
            dno_output_finite=dno_finite_result,
            minimum_water_column=water_column_result,
        )
    return eta_result, xi_result, gxi_result, internal_telemetry


def _solver_parameters(
    config: RolloutConfig,
    depths: jax.Array,
    *,
    nonlinear_ramp_times: FloatArray | None,
    nonlinear_ramp_order: int,
) -> SolverParams:
    _, wavenumbers = build_grid(config.nx, config.length)
    k = jnp.asarray(wavenumbers, dtype=jnp.float64)
    depth_column = depths[:, None]
    return SolverParams(
        nx=config.nx,
        length=config.length,
        depth=depth_column,
        gravity=config.gravity,
        dno_order=config.dno_order,
        pad_factor=config.pad_factor,
        filter_fraction=config.filter_fraction,
        k=k,
        g0=make_linear_dno_symbol(k, depth_column),
        nonlinear_ramp_time=(
            None
            if nonlinear_ramp_times is None
            else jnp.asarray(nonlinear_ramp_times, dtype=jnp.float64)
        ),
        nonlinear_ramp_order=nonlinear_ramp_order,
    )


def _integrate_gl2(
    eta: FloatArray,
    xi: FloatArray,
    depths: FloatArray,
    saved_times: FloatArray,
    config: RolloutConfig,
    *,
    nonlinear_ramp_times: FloatArray | None = None,
    nonlinear_ramp_order: int = 4,
) -> tuple[dict[str, jax.Array], jax.Array]:
    depth_device = jnp.asarray(depths, dtype=jnp.float64)
    payload = integrate_trajectory(
        State(
            eta=jnp.asarray(eta, dtype=jnp.float64),
            xi=jnp.asarray(xi, dtype=jnp.float64),
        ),
        jnp.asarray(saved_times, dtype=jnp.float64),
        _solver_parameters(
            config,
            depth_device,
            nonlinear_ramp_times=nonlinear_ramp_times,
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
    jax.block_until_ready(payload["xi"])
    return payload, depth_device


def _extract_gl2_telemetry(payload: dict[str, jax.Array]) -> GL2BatchTelemetry:
    residual, converged, stage_finite, state_finite = jax.device_get(
        (
            payload["gl2_stage_residual"],
            payload["gl2_converged"],
            payload["gl2_stage_finite"],
            payload["gl2_state_finite"],
        )
    )
    return GL2BatchTelemetry(
        stage_residual=np.asarray(residual, dtype=np.float64),
        converged=np.asarray(converged, dtype=np.bool_),
        stage_finite=np.asarray(stage_finite, dtype=np.bool_),
        state_finite=np.asarray(state_finite, dtype=np.bool_),
    )


def integrate_batch(
    *,
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    saved_times: FloatArray,
    config: RolloutConfig,
) -> IntegratedTrajectoryBatch:
    """Integrate a validated trajectory batch and compute its DNO targets."""

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("residual-controlled paper-dataset GL2 requires float64")
    payload, depth_device = _integrate_gl2(
        eta0,
        xi0,
        depths,
        saved_times,
        config,
    )
    eta, xi, gxi, internal_telemetry = compute_saved_targets(
        payload["eta"],
        payload["xi"],
        depth_device,
        config,
    )
    return IntegratedTrajectoryBatch(
        eta=eta,
        xi=xi,
        gxi=gxi,
        gl2=_extract_gl2_telemetry(payload),
        internal_telemetry=internal_telemetry,
    )


def integrate_adjustment_batch(
    *,
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    saved_times: FloatArray,
    config: RolloutConfig,
    nonlinear_ramp_times: FloatArray,
    nonlinear_ramp_order: int,
) -> IntegratedAdjustmentBatch:
    """Warm up a validated JONSWAP batch by gradually enabling nonlinearity."""

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("nonlinear adjustment requires JAX float64")
    payload, _ = _integrate_gl2(
        eta0,
        xi0,
        depths,
        saved_times,
        config,
        nonlinear_ramp_times=nonlinear_ramp_times,
        nonlinear_ramp_order=nonlinear_ramp_order,
    )
    eta, xi = jax.device_get((payload["eta"], payload["xi"]))
    return IntegratedAdjustmentBatch(
        eta=np.asarray(eta, dtype=np.float64),
        xi=np.asarray(xi, dtype=np.float64),
        gl2=_extract_gl2_telemetry(payload),
    )


def _validate_gl2_telemetry_shapes(
    telemetry: GL2BatchTelemetry,
    expected_shape: tuple[int, int],
) -> None:
    for name, values in (
        ("stage_residual", telemetry.stage_residual),
        ("converged", telemetry.converged),
        ("stage_finite", telemetry.stage_finite),
        ("state_finite", telemetry.state_finite),
    ):
        if np.asarray(values).shape != expected_shape:
            raise ValueError(f"GL2 {name} must have shape {expected_shape}")


def validate_integrated_batch(
    rollout: IntegratedTrajectoryBatch,
    *,
    batch_size: int,
    saved_time_count: int,
    config: RolloutConfig,
) -> None:
    """Validate the arrays returned by a trajectory integrator."""

    field_shape = (saved_time_count, batch_size, config.target_nx)
    for name, field in (
        ("eta", rollout.eta),
        ("xi", rollout.xi),
        ("gxi", rollout.gxi),
    ):
        if np.asarray(field).shape != field_shape:
            raise ValueError(f"rollout {name} must have shape {field_shape}")
    _validate_gl2_telemetry_shapes(
        rollout.gl2,
        ((saved_time_count - 1) * config.substeps_per_saved_frame, batch_size),
    )
    internal = rollout.internal_telemetry
    if internal is None:
        return
    expected_health_shape = (saved_time_count, batch_size)
    for name in (
        "hamiltonian",
        "state_finite",
        "dno_output_finite",
        "minimum_water_column",
    ):
        if np.asarray(getattr(internal, name)).shape != expected_health_shape:
            raise ValueError(f"internal {name} must have shape {expected_health_shape}")


def validate_adjustment_rollout(
    rollout: IntegratedAdjustmentBatch,
    *,
    batch_size: int,
    saved_time_count: int,
    config: RolloutConfig,
) -> None:
    """Validate the arrays returned by a nonlinear-adjustment generator."""

    field_shape = (saved_time_count, batch_size, config.nx)
    for name, field in (("eta", rollout.eta), ("xi", rollout.xi)):
        if np.asarray(field).shape != field_shape:
            raise ValueError(f"adjustment rollout {name} must have shape {field_shape}")
    _validate_gl2_telemetry_shapes(
        rollout.gl2,
        ((saved_time_count - 1) * config.substeps_per_saved_frame, batch_size),
    )
