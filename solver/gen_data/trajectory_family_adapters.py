"""Construct solver-ready initial states for trajectory simulations."""

from __future__ import annotations

from typing import NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from solver.gen_data.benjamin_feir_jcp09 import (
    build_initial_conditions as build_benjamin_feir_initial_conditions,
    deep_water_proxy_depth,
)
from solver.gen_data.benjamin_feir_sampling import BenjaminFeirSample
from solver.gen_data.jonswap_tma import (
    JonswapTmaState,
    ResolvedBand,
    build_jonswap_tma_initial_condition,
)
from solver.gen_data.jonswap_tma_sampling import JonswapTmaSample
from solver.gen_data.pipeline.dno_target import project_fixed_band
from solver.gen_data.pipeline.trajectory_config import RolloutNumerics
from solver.gen_data.tanaka_initial_conditions import (
    build_tanaka_initial_conditions,
)
from solver.gen_data.tanaka_sampling import TanakaSample
from solver.solvers.dno_series_jax import build_grid
from solver.tanaka_ICs.modified_tanaka import make_default_tanaka_template


FloatArray: TypeAlias = NDArray[np.float64]

TrajectoryInitialBatch = NamedTuple(
    "TrajectoryInitialBatch",
    [("eta0", FloatArray), ("xi0", FloatArray), ("depths", FloatArray)],
)


class JonswapInitialStateDomainError(ValueError):
    """JONSWAP initial states outside the finite positive-water graph domain."""

    def __init__(
        self,
        invalid_simulation_indices: tuple[int, ...],
        nonfinite_state_flags: tuple[bool, ...],
        nonpositive_water_height_flags: tuple[bool, ...],
    ) -> None:
        self.invalid_simulation_indices = invalid_simulation_indices
        self.nonfinite_state_flags = nonfinite_state_flags
        self.nonpositive_water_height_flags = nonpositive_water_height_flags
        super().__init__(
            "JONSWAP/TMA initial states violate the graph domain at local indices "
            f"{invalid_simulation_indices}"
        )


def _project_initial_conditions(
    eta0: FloatArray | jax.Array,
    xi0: FloatArray | jax.Array,
    numerical: RolloutNumerics,
) -> tuple[FloatArray, FloatArray]:
    """Apply the common target-band projection and remove the xi zero mode."""

    _, wavenumbers = build_grid(numerical.nx, numerical.length)
    projected_eta = project_fixed_band(
        jnp.asarray(eta0, dtype=jnp.float64),
        jnp.asarray(wavenumbers, dtype=jnp.float64),
        maximum_wavenumber=numerical.target_maximum_wavenumber,
    )
    projected_xi = project_fixed_band(
        jnp.asarray(xi0, dtype=jnp.float64),
        jnp.asarray(wavenumbers, dtype=jnp.float64),
        maximum_wavenumber=numerical.target_maximum_wavenumber,
        remove_mean=True,
    )
    jax.block_until_ready(projected_xi)
    return (
        np.asarray(jax.device_get(projected_eta), dtype=np.float64),
        np.asarray(jax.device_get(projected_xi), dtype=np.float64),
    )


def construct_tanaka_trajectory_batch(
    samples: tuple[TanakaSample, ...],
    numerical: RolloutNumerics,
) -> TrajectoryInitialBatch:
    """Construct tangent-Hermite initial states for Tanaka simulations."""

    depths = np.asarray([sample.depth for sample in samples], dtype=np.float64)
    template = make_default_tanaka_template(
        depth=1.0,
        gravity=numerical.gravity,
        direction=1,
        nx=numerical.nx,
        length=numerical.length,
        center=0.0,
        dno_order=numerical.integration_dno_order,
        pad_factor=numerical.pad_factor,
    )
    eta0, xi0 = build_tanaka_initial_conditions(
        template,
        depths,
        tuple(sample.crests for sample in samples),
        length=numerical.length,
        nx=numerical.nx,
        gravity=numerical.gravity,
    )
    eta, xi = _project_initial_conditions(eta0, xi0, numerical)
    return TrajectoryInitialBatch(eta, xi, depths)


def construct_benjamin_feir_trajectory_batch(
    samples: tuple[BenjaminFeirSample, ...],
    numerical: RolloutNumerics,
) -> TrajectoryInitialBatch:
    """Construct Benjamin--Feir initial states."""

    x, _ = build_grid(numerical.nx, numerical.length)
    eta0, xi0 = build_benjamin_feir_initial_conditions(
        x=jnp.asarray(x, dtype=jnp.float64),
        carrier_modes=np.fromiter(
            (sample.carrier_mode for sample in samples), dtype=np.int32
        ),
        sideband_offsets=np.fromiter(
            (sample.sideband_offset for sample in samples), dtype=np.int32
        ),
        carrier_steepnesses=np.fromiter(
            (sample.carrier_steepness for sample in samples), dtype=np.float64
        ),
        perturbation_ratios=np.fromiter(
            (sample.perturbation_ratio for sample in samples), dtype=np.float64
        ),
        translations=np.fromiter(
            (sample.translation for sample in samples), dtype=np.float64
        ),
        length=numerical.length,
        gravity=numerical.gravity,
    )
    eta, xi = _project_initial_conditions(eta0, xi0, numerical)
    depths = np.full(
        len(samples), deep_water_proxy_depth(numerical.length), dtype=np.float64
    )
    return TrajectoryInitialBatch(eta, xi, depths)


def construct_jonswap_tma_trajectory_batch(
    samples: tuple[JonswapTmaSample, ...],
    numerical: RolloutNumerics,
    *,
    band: ResolvedBand,
) -> TrajectoryInitialBatch:
    """Construct resolved-band JONSWAP/TMA initial states."""

    x = numerical.length * np.arange(numerical.nx, dtype=np.float64) / numerical.nx
    states: tuple[JonswapTmaState, ...] = tuple(
        build_jonswap_tma_initial_condition(
            x,
            parameters=sample.parameters,
            phase_right=sample.phase_right,
            phase_left=sample.phase_left,
            band=band,
            gravity=numerical.gravity,
        )
        for sample in samples
    )
    eta0, xi0 = _project_initial_conditions(
        np.stack(tuple(state.eta for state in states)),
        np.stack(tuple(state.xi for state in states)),
        numerical,
    )
    depths = np.asarray(
        [sample.parameters.depth for sample in samples], dtype=np.float64
    )
    state_finite = (
        np.isfinite(eta0).all(axis=1)
        & np.isfinite(xi0).all(axis=1)
        & np.isfinite(depths)
        & (depths > 0.0)
    )
    minimum_water_columns = np.full(len(samples), np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(state_finite)
    minimum_water_columns[finite_indices] = np.min(
        depths[finite_indices, None] + eta0[finite_indices], axis=1
    )
    invalid = np.flatnonzero(
        ~state_finite
        | ~np.isfinite(minimum_water_columns)
        | (minimum_water_columns <= 0.0)
    )
    if invalid.size:
        indices = tuple(int(index) for index in invalid)
        raise JonswapInitialStateDomainError(
            indices,
            tuple(not bool(state_finite[index]) for index in invalid),
            tuple(bool(state_finite[index]) for index in invalid),
        )
    return TrajectoryInitialBatch(eta0, xi0, depths)
