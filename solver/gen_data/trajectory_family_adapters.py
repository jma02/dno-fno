"""Sample and construct initial states for trajectory batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.artifact_io import json_text
from solver.gen_data.benjamin_feir_jcp09 import (
    BENJAMIN_FEIR_CONSTRUCTOR,
    ParameterArrays,
    build_initial_conditions as build_benjamin_feir_initial_conditions,
    deep_water_proxy_depth,
)
from solver.gen_data.benjamin_feir_sampling import (
    BenjaminFeirSample,
    sample_benjamin_feir_simulation,
)
from solver.gen_data.tanaka_initial_conditions import (
    TANAKA_PROFILE_RECONSTRUCTION,
    build_per_simulation_initial_conditions,
)
from solver.gen_data.jonswap_tma import (
    PAPER_RELATIVE_FREQUENCY_MAXIMUM,
    PAPER_RELATIVE_FREQUENCY_MINIMUM,
    PAPER_RELATIVE_FREQUENCY_WINDOW,
    PAPER_RESOLVED_BAND_TRANSITION_FRACTION,
    ResolvedBand,
    JonswapTmaState,
    build_jonswap_tma_initial_condition,
)
from solver.gen_data.jonswap_tma_sampling import (
    JonswapTmaSample,
    sample_jonswap_tma_simulation,
)
from solver.gen_data.pipeline.types import DatasetSplit
from solver.gen_data.pipeline.dno_target import project_fixed_band
from solver.gen_data.pipeline.trajectory_config import RolloutNumerics
from solver.gen_data.pipeline.writer import JsonScalar
from solver.gen_data.tanaka_sampling import (
    TanakaSample,
    sample_tanaka_simulation,
)
from solver.solvers.dno_series_jax import build_grid
from solver.tanaka_ICs.modified_tanaka import make_default_tanaka_template


FloatArray: TypeAlias = NDArray[np.float64]
SpecificationRecord: TypeAlias = Mapping[str, object]
MetricsRecord: TypeAlias = Mapping[str, JsonScalar]
SampleT = TypeVar("SampleT")


@dataclass(frozen=True)
class SampledSimulations(Generic[SampleT]):
    """Complete sampled simulations and strict records, before numerical work."""

    parameter_group_ids: tuple[str, ...]
    samples: tuple[SampleT, ...]
    specification_records: tuple[SpecificationRecord, ...]
    config: RolloutNumerics
    construction_settings: SpecificationRecord

    def __post_init__(self) -> None:
        count = len(self.parameter_group_ids)
        if count == 0:
            raise ValueError("sampled trajectory simulations must not be empty")
        if len(self.samples) != count or len(self.specification_records) != count:
            raise ValueError(
                "parameter groups, samples, and records must have equal lengths"
            )
        for record in self.specification_records:
            json_text(record)


@dataclass(frozen=True)
class TrajectoryInitialBatch:
    """One constructor-ready batch and its persisted simulation specifications."""

    eta0: FloatArray
    xi0: FloatArray
    depths: FloatArray
    specification_records: tuple[SpecificationRecord, ...]
    construction_metrics: tuple[MetricsRecord, ...] = ()

    def __post_init__(self) -> None:
        if self.eta0.dtype != np.float64 or self.xi0.dtype != np.float64:
            raise TypeError("eta0 and xi0 must have dtype float64")
        if self.depths.dtype != np.float64:
            raise TypeError("depths must have dtype float64")
        if self.eta0.ndim != 2 or self.xi0.shape != self.eta0.shape:
            raise ValueError("eta0 and xi0 must have common shape (batch, nx)")
        if self.depths.shape != (self.eta0.shape[0],):
            raise ValueError("depths must contain one value per simulation")
        if len(self.specification_records) != self.eta0.shape[0]:
            raise ValueError(
                "specification_records must contain one record per simulation"
            )
        if self.construction_metrics and (
            len(self.construction_metrics) != self.eta0.shape[0]
        ):
            raise ValueError(
                "construction_metrics must contain one record per simulation"
            )
        for metrics in self.construction_metrics:
            json_text(metrics)
        if not np.isfinite(self.eta0).all() or not np.isfinite(self.xi0).all():
            raise ValueError("initial fields must be finite")
        if not np.isfinite(self.depths).all() or np.any(self.depths <= 0.0):
            raise ValueError("depths must be finite and positive")
        if np.any(self.depths[:, None] + self.eta0 <= 0.0):
            raise ValueError("initial surfaces must remain above the bottom")


@dataclass(frozen=True)
class JonswapInitialStateFailure:
    """One JONSWAP initial state outside the graph domain."""

    local_simulation_index: int
    state_finite: bool
    minimum_water_column: float | None

    def to_json_record(self) -> dict[str, object]:
        return {
            "local_simulation_index": self.local_simulation_index,
            "failure_reason": (
                "nonpositive_initial_water_column"
                if self.state_finite
                else "nonfinite_initial_state"
            ),
            "state_finite": self.state_finite,
            "minimum_water_column": self.minimum_water_column,
        }


class JonswapInitialStateDomainError(ValueError):
    """Per-simulation failure of the JONSWAP initial graph-domain conditions."""

    def __init__(self, failures: tuple[JonswapInitialStateFailure, ...]) -> None:
        self.failures = failures
        super().__init__(
            "JONSWAP/TMA initial states violate the graph domain at "
            f"constructor-sub-batch indices {self.invalid_simulation_indices}"
        )

    @property
    def invalid_simulation_indices(self) -> tuple[int, ...]:
        return tuple(failure.local_simulation_index for failure in self.failures)

    def to_json_record(self) -> dict[str, object]:
        return {
            "schema": "jonswap_initial_state_domain_failure_v1",
            "reason": "invalid_initial_graph_state",
            "index_space": "constructor_subbatch_local_index",
            "invalid_simulation_indices": list(self.invalid_simulation_indices),
            "simulations": [failure.to_json_record() for failure in self.failures],
        }


def _jonswap_initial_metrics(
    state: JonswapTmaState,
    eta: FloatArray,
    xi: FloatArray,
    *,
    depth: float,
    significant_height: float,
    length: float,
    gravity: float,
) -> MetricsRecord:
    """Return elementary diagnostics of one constructed random-sea state."""

    eta_rms = float(np.sqrt(np.mean((eta - np.mean(eta)) ** 2)))
    xi_rms = float(np.sqrt(np.mean((xi - np.mean(xi)) ** 2)))
    maximum_cell_energy = float(np.max(state.spectrum.energy_fractions))
    half_maximum_cells = int(
        np.count_nonzero(state.spectrum.energy_fractions >= 0.5 * maximum_cell_energy)
    )
    flat_wavenumbers = (
        2.0
        * np.pi
        * np.fft.rfftfreq(
            eta.size,
            d=length / eta.size,
        )
    )
    flat_symbol = flat_wavenumbers * np.tanh(flat_wavenumbers * depth)
    flat_dno_xi = np.fft.irfft(
        flat_symbol * np.fft.rfft(xi),
        n=eta.size,
    )
    realized_linear_hamiltonian = float(
        0.5 * length * np.mean(xi * flat_dno_xi + gravity * eta**2)
    )
    expected_linear_hamiltonian = gravity * length * (significant_height / 4.0) ** 2
    return {
        "initial_discrete_peak_wavenumber": float(
            state.spectrum.wavenumbers[int(np.argmax(state.spectrum.energy_fractions))]
        ),
        "initial_half_maximum_spectral_cell_count": half_maximum_cells,
        "initial_eta_rms": eta_rms,
        "initial_xi_rms": xi_rms,
        "initial_realized_height_ratio": 4.0 * eta_rms / significant_height,
        "initial_linear_hamiltonian": realized_linear_hamiltonian,
        "initial_expected_linear_hamiltonian": expected_linear_hamiltonian,
        "initial_linear_hamiltonian_relative_error": abs(
            realized_linear_hamiltonian - expected_linear_hamiltonian
        )
        / expected_linear_hamiltonian,
        "initial_minimum_water_column": float(np.min(depth + eta)),
    }


def resolved_band_for_config(
    config: RolloutNumerics,
    *,
    quadrature_order: int = 16,
) -> ResolvedBand:
    """Return the delivered random-sea band with transition at ``3 K / 4``."""

    maximum_wavenumber = config.target_maximum_wavenumber
    return ResolvedBand(
        length=config.length,
        maximum_wavenumber=maximum_wavenumber,
        transition_wavenumber=(
            PAPER_RESOLVED_BAND_TRANSITION_FRACTION * maximum_wavenumber
        ),
        quadrature_order=quadrature_order,
    )


def _require_parameter_group_ids(
    parameter_group_ids: Sequence[str],
) -> tuple[str, ...]:
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("trajectory constructors require JAX float64 mode")
    values = tuple(parameter_group_ids)
    if not values:
        raise ValueError("parameter_group_ids must not be empty")
    return values


def _strict_record(
    sample_record: Mapping[str, object],
    *,
    constructor: str,
    config: RolloutNumerics,
    constructor_settings: Mapping[str, object] | None = None,
) -> SpecificationRecord:
    initial_maximum_wavenumber = config.target_maximum_wavenumber
    record = {
        **sample_record,
        "initial_condition_constructor": constructor,
        "initial_projection": {
            "kind": "sharp_fixed_physical_wavenumber_band",
            "maximum_wavenumber": initial_maximum_wavenumber,
            "xi_zero_mode_removed": True,
        },
        "constructor_settings": dict(constructor_settings or {}),
    }
    json_text(record)
    return record


def _sampled_simulations(
    parameter_group_ids: tuple[str, ...],
    samples: tuple[SampleT, ...],
    records: tuple[SpecificationRecord, ...],
    *,
    config: RolloutNumerics,
    construction_settings: SpecificationRecord,
) -> SampledSimulations[SampleT]:
    return SampledSimulations(
        parameter_group_ids=parameter_group_ids,
        samples=samples,
        specification_records=records,
        config=config,
        construction_settings=construction_settings,
    )


def _project_once(
    eta0: FloatArray | jax.Array,
    xi0: FloatArray | jax.Array,
    *,
    config: RolloutNumerics,
) -> tuple[FloatArray, FloatArray]:
    """Apply the common fixed-band map, removing only the ``xi`` zero mode."""

    eta = jnp.asarray(eta0, dtype=jnp.float64)
    xi = jnp.asarray(xi0, dtype=jnp.float64)
    if eta.ndim != 2 or eta.shape != xi.shape or eta.shape[1] != config.nx:
        raise ValueError(f"raw fields must have common shape (batch, {config.nx})")
    _, wavenumbers = build_grid(config.nx, config.length)
    k = jnp.asarray(wavenumbers, dtype=jnp.float64)
    maximum_wavenumber = config.target_maximum_wavenumber
    projected_eta = project_fixed_band(
        eta,
        k,
        maximum_wavenumber=maximum_wavenumber,
    )
    projected_xi = project_fixed_band(
        xi,
        k,
        maximum_wavenumber=maximum_wavenumber,
        remove_mean=True,
    )
    jax.block_until_ready(projected_xi)
    return (
        np.asarray(jax.device_get(projected_eta), dtype=np.float64),
        np.asarray(jax.device_get(projected_xi), dtype=np.float64),
    )


def _resolve_selected_simulation_indices(
    selected_local_indices: Sequence[int] | None,
    *,
    simulation_count: int,
) -> tuple[int, ...]:
    """Return all simulation indices or validate an explicit ordered subset."""

    if selected_local_indices is None:
        return tuple(range(simulation_count))
    selected = tuple(selected_local_indices)
    if not selected:
        raise ValueError("selected_local_indices must not be empty")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in selected):
        raise TypeError("selected_local_indices must contain integers")
    if tuple(sorted(set(selected))) != selected:
        raise ValueError("selected_local_indices must be unique and increasing")
    if selected[0] < 0 or selected[-1] >= simulation_count:
        raise ValueError("selected_local_indices contains an out-of-range index")
    return selected


def _batch(
    eta0: FloatArray | jax.Array,
    xi0: FloatArray | jax.Array,
    depths: FloatArray,
    specification_records: tuple[SpecificationRecord, ...],
    *,
    config: RolloutNumerics,
) -> TrajectoryInitialBatch:
    eta, xi = _project_once(eta0, xi0, config=config)
    return TrajectoryInitialBatch(
        eta0=eta,
        xi0=xi,
        depths=np.asarray(depths, dtype=np.float64),
        specification_records=specification_records,
    )


def sample_tanaka_simulations(
    parameter_group_ids: Sequence[str],
    *,
    dataset_split: DatasetSplit,
    first_attempt_number: int,
    config: RolloutNumerics,
) -> SampledSimulations[TanakaSample]:
    """Sample complete Tanaka simulations without running the Tanaka solver."""

    attempted_groups = _require_parameter_group_ids(parameter_group_ids)
    samples = tuple(
        sample_tanaka_simulation(
            parameter_group_id,
            dataset_split=dataset_split,
            attempt_number=first_attempt_number + offset,
        )
        for offset, parameter_group_id in enumerate(attempted_groups)
    )
    settings: SpecificationRecord = {
        "dimensionless_template_depth": 1.0,
        "crest_spec_amplitude_is_alpha": True,
    }
    records = tuple(
        _strict_record(
            {
                "depth": sample.depth,
                "crests": [
                    {
                        "alpha": crest.alpha,
                        "center": crest.center,
                        "direction": crest.direction,
                    }
                    for crest in sample.crests
                ],
            },
            constructor=TANAKA_PROFILE_RECONSTRUCTION,
            config=config,
            constructor_settings=settings,
        )
        for sample in samples
    )
    return _sampled_simulations(
        attempted_groups,
        samples,
        records,
        config=config,
        construction_settings=settings,
    )


def construct_tanaka_trajectory_batch(
    sampled: SampledSimulations[TanakaSample],
    *,
    selected_local_indices: Sequence[int] | None = None,
) -> TrajectoryInitialBatch:
    """Construct tangent-Hermite states for the selected simulations.

    ``selected_local_indices`` retains valid siblings after a declared
    per-simulation construction failure.
    """

    samples = sampled.samples
    if not all(isinstance(sample, TanakaSample) for sample in samples):
        raise TypeError("Tanaka construction requires Tanaka samples")
    expected_records = sampled.specification_records
    selected = _resolve_selected_simulation_indices(
        selected_local_indices,
        simulation_count=len(samples),
    )

    selected_samples = tuple(samples[index] for index in selected)
    selected_records = tuple(expected_records[index] for index in selected)
    config = sampled.config
    depths = np.asarray(
        [sample.depth for sample in selected_samples],
        dtype=np.float64,
    )
    simulation_specs = [list(sample.crests) for sample in selected_samples]
    template = make_default_tanaka_template(
        depth=1.0,
        gravity=config.gravity,
        direction=1,
        nx=config.nx,
        length=config.length,
        center=0.0,
        dno_order=config.integration_dno_order,
        pad_factor=config.pad_factor,
    )
    eta0, xi0 = build_per_simulation_initial_conditions(
        template_params=template,
        simulation_h_ref=depths,
        simulation_specs=simulation_specs,
        length=config.length,
        nx=config.nx,
        gravity=config.gravity,
    )
    return _batch(
        eta0,
        xi0,
        depths,
        selected_records,
        config=config,
    )


def sample_benjamin_feir_simulations(
    parameter_group_ids: Sequence[str],
    *,
    dataset_split: DatasetSplit,
    first_attempt_number: int,
    config: RolloutNumerics,
) -> SampledSimulations[BenjaminFeirSample]:
    """Sample complete Benjamin--Feir simulations without numerical construction."""

    attempted_groups = _require_parameter_group_ids(parameter_group_ids)
    samples = tuple(
        sample_benjamin_feir_simulation(
            parameter_group_id,
            dataset_split=dataset_split,
            attempt_number=first_attempt_number + offset,
        )
        for offset, parameter_group_id in enumerate(attempted_groups)
    )
    records = tuple(
        _strict_record(
            {
                "carrier_mode": sample.carrier_mode,
                "sideband_offset": sample.sideband_offset,
                "carrier_steepness": sample.carrier_steepness,
                "perturbation_ratio": sample.perturbation_ratio,
                "translation": sample.translation,
            },
            constructor=BENJAMIN_FEIR_CONSTRUCTOR,
            config=config,
        )
        for sample in samples
    )
    return _sampled_simulations(
        attempted_groups,
        samples,
        records,
        config=config,
        construction_settings={},
    )


def construct_benjamin_feir_trajectory_batch(
    sampled: SampledSimulations[BenjaminFeirSample],
) -> TrajectoryInitialBatch:
    """Construct Benjamin--Feir initial states."""

    samples = sampled.samples
    if not all(isinstance(sample, BenjaminFeirSample) for sample in samples):
        raise TypeError("Benjamin--Feir construction requires Benjamin--Feir samples")
    config = sampled.config
    parameters: ParameterArrays = {
        "n_carr": np.fromiter((s.carrier_mode for s in samples), dtype=np.int32),
        "side_offset": np.fromiter(
            (s.sideband_offset for s in samples), dtype=np.int32
        ),
        "eps_carrier": np.fromiter(
            (s.carrier_steepness for s in samples), dtype=np.float64
        ),
        "eps_pert": np.fromiter(
            (s.perturbation_ratio for s in samples), dtype=np.float64
        ),
        "translation": np.fromiter((s.translation for s in samples), dtype=np.float64),
    }
    x, _ = build_grid(config.nx, config.length)
    eta0, xi0 = build_benjamin_feir_initial_conditions(
        x=jnp.asarray(x, dtype=jnp.float64),
        parameters=parameters,
        length=config.length,
        gravity=config.gravity,
        dtype=jnp.float64,
    )
    proxy_depth = deep_water_proxy_depth(config.length)
    depths = np.full(len(samples), proxy_depth, dtype=np.float64)
    return _batch(
        eta0,
        xi0,
        depths,
        sampled.specification_records,
        config=config,
    )


def sample_jonswap_tma_simulations(
    parameter_group_ids: Sequence[str],
    *,
    dataset_split: DatasetSplit,
    first_attempt_number: int,
    config: RolloutNumerics,
    quadrature_order: int = 16,
) -> SampledSimulations[JonswapTmaSample]:
    """Sample complete JONSWAP/TMA simulations and both phase arrays."""

    attempted_groups = _require_parameter_group_ids(parameter_group_ids)
    band = resolved_band_for_config(
        config,
        quadrature_order=quadrature_order,
    )
    samples = tuple(
        sample_jonswap_tma_simulation(
            parameter_group_id,
            dataset_split=dataset_split,
            attempt_number=first_attempt_number + offset,
            band=band,
        )
        for offset, parameter_group_id in enumerate(attempted_groups)
    )
    constructor = "relative_frequency_jonswap_tma_linear_state_v2"
    settings: SpecificationRecord = {
        "density_window": PAPER_RELATIVE_FREQUENCY_WINDOW,
        "relative_frequency_minimum": PAPER_RELATIVE_FREQUENCY_MINIMUM,
        "relative_frequency_maximum": PAPER_RELATIVE_FREQUENCY_MAXIMUM,
        "quadrature_order": band.quadrature_order,
    }
    records = tuple(
        _strict_record(
            {
                "depth": sample.parameters.depth,
                "significant_height": sample.parameters.significant_height,
                "peak_wavenumber": sample.parameters.peak_wavenumber,
                "peak_enhancement": sample.parameters.peak_enhancement,
                "right_moving_fraction": sample.parameters.right_moving_fraction,
                "phase_right": sample.phase_right.tolist(),
                "phase_left": sample.phase_left.tolist(),
            },
            constructor=constructor,
            config=config,
            constructor_settings=settings,
        )
        for sample in samples
    )
    return _sampled_simulations(
        attempted_groups,
        samples,
        records,
        config=config,
        construction_settings=settings,
    )


def construct_jonswap_tma_trajectory_batch(
    sampled: SampledSimulations[JonswapTmaSample],
    *,
    selected_local_indices: Sequence[int] | None = None,
) -> TrajectoryInitialBatch:
    """Construct resolved-band states for the selected simulations."""

    samples = sampled.samples
    if not all(isinstance(sample, JonswapTmaSample) for sample in samples):
        raise TypeError("JONSWAP/TMA construction requires JONSWAP/TMA samples")
    density_window = sampled.construction_settings.get("density_window")
    if density_window != PAPER_RELATIVE_FREQUENCY_WINDOW:
        raise ValueError("JONSWAP/TMA density window does not match the config")
    relative_minimum = sampled.construction_settings.get("relative_frequency_minimum")
    relative_maximum = sampled.construction_settings.get("relative_frequency_maximum")
    if (
        relative_minimum != PAPER_RELATIVE_FREQUENCY_MINIMUM
        or relative_maximum != PAPER_RELATIVE_FREQUENCY_MAXIMUM
    ):
        raise ValueError(
            "JONSWAP/TMA relative frequency interval does not match the config"
        )
    relative_frequency_interval = (
        PAPER_RELATIVE_FREQUENCY_MINIMUM,
        PAPER_RELATIVE_FREQUENCY_MAXIMUM,
    )
    quadrature_order = sampled.construction_settings.get("quadrature_order")
    if not isinstance(quadrature_order, int) or isinstance(quadrature_order, bool):
        raise TypeError("JONSWAP/TMA quadrature_order must be an integer")
    config = sampled.config
    band = resolved_band_for_config(
        config,
        quadrature_order=quadrature_order,
    )
    expected_records = sampled.specification_records
    selected = _resolve_selected_simulation_indices(
        selected_local_indices,
        simulation_count=len(samples),
    )

    selected_samples = tuple(samples[index] for index in selected)
    selected_records = tuple(expected_records[index] for index in selected)
    x = config.length * np.arange(config.nx, dtype=np.float64) / config.nx
    states = tuple(
        build_jonswap_tma_initial_condition(
            x,
            parameters=sample.parameters,
            phase_right=sample.phase_right,
            phase_left=sample.phase_left,
            band=band,
            gravity=config.gravity,
            relative_frequency_interval=relative_frequency_interval,
        )
        for sample in selected_samples
    )
    raw_eta0 = np.stack(tuple(state.eta for state in states))
    raw_xi0 = np.stack(tuple(state.xi for state in states))
    eta0, xi0 = _project_once(raw_eta0, raw_xi0, config=config)
    depths = np.asarray(
        [sample.parameters.depth for sample in selected_samples],
        dtype=np.float64,
    )

    state_finite = (
        np.isfinite(eta0).all(axis=1)
        & np.isfinite(xi0).all(axis=1)
        & np.isfinite(depths)
        & (depths > 0.0)
    )
    minimum_water_columns = np.full(len(selected), np.nan, dtype=np.float64)
    for index in np.flatnonzero(state_finite):
        minimum_water_columns[index] = np.min(depths[index] + eta0[index])
    invalid = np.flatnonzero(
        ~state_finite
        | ~np.isfinite(minimum_water_columns)
        | (minimum_water_columns <= 0.0)
    )
    if invalid.size:
        failures = tuple(
            JonswapInitialStateFailure(
                local_simulation_index=int(index),
                state_finite=bool(state_finite[index]),
                minimum_water_column=(
                    float(minimum_water_columns[index]) if state_finite[index] else None
                ),
            )
            for index in invalid
        )
        raise JonswapInitialStateDomainError(failures)

    return TrajectoryInitialBatch(
        eta0=eta0,
        xi0=xi0,
        depths=depths,
        specification_records=selected_records,
        construction_metrics=tuple(
            _jonswap_initial_metrics(
                state,
                eta0[index],
                xi0[index],
                depth=sample.parameters.depth,
                significant_height=sample.parameters.significant_height,
                length=config.length,
                gravity=config.gravity,
            )
            for index, (sample, state) in enumerate(zip(selected_samples, states))
        ),
    )
