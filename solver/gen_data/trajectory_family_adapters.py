"""Sample and construct initial states for trajectory batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
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
from solver.gen_data.pipeline.batch_storage import batch_path
from solver.gen_data.pipeline.simulation_allocation import AttemptAssignment
from solver.gen_data.pipeline.dno_target import project_fixed_band
from solver.gen_data.pipeline.trajectory_config import RolloutConfig
from solver.gen_data.pipeline.types import BatchPlanArrays
from solver.gen_data.pipeline.writer import (
    JsonScalar,
    build_batch_plan,
)
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

    assignments: tuple[AttemptAssignment, ...]
    samples: tuple[SampleT, ...]
    specification_records: tuple[SpecificationRecord, ...]
    contract: RolloutConfig
    construction_settings: SpecificationRecord

    def __post_init__(self) -> None:
        count = len(self.assignments)
        if count == 0:
            raise ValueError("sampled trajectory simulations must not be empty")
        if len(self.samples) != count or len(self.specification_records) != count:
            raise ValueError(
                "assignments, samples, and records must have equal lengths"
            )
        for record in self.specification_records:
            json_text(record)


@dataclass(frozen=True)
class PreparedTrajectoryBatch(Generic[SampleT]):
    """Sampled simulations and their eventual completed-batch path."""

    sampled: SampledSimulations[SampleT]
    path: Path
    batch_plan: BatchPlanArrays


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


class JonswapInitialStateDomainError(ValueError):
    """Per-simulation failure of the JONSWAP initial graph-domain conditions."""

    def __init__(self, failure_record: dict[str, object]) -> None:
        self.failure_record = failure_record
        invalid = failure_record.get("invalid_simulation_indices")
        super().__init__(
            "JONSWAP/TMA initial states violate the graph domain at "
            f"constructor-sub-batch indices {invalid}"
        )


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


def resolved_band_for_contract(
    contract: RolloutConfig,
    *,
    quadrature_order: int = 16,
) -> ResolvedBand:
    """Return the delivered random-sea band with transition at ``3 K / 4``."""

    maximum_wavenumber = contract.target_maximum_wavenumber
    return ResolvedBand(
        length=contract.length,
        maximum_wavenumber=maximum_wavenumber,
        transition_wavenumber=(
            PAPER_RESOLVED_BAND_TRANSITION_FRACTION * maximum_wavenumber
        ),
        quadrature_order=quadrature_order,
    )


def _require_assignments(
    assignments: Sequence[AttemptAssignment],
) -> tuple[AttemptAssignment, ...]:
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("trajectory constructors require JAX float64 mode")
    values = tuple(assignments)
    if not values:
        raise ValueError("assignments must not be empty")
    return values


def _strict_record(
    sample_record: Mapping[str, object],
    *,
    constructor: str,
    contract: RolloutConfig,
    constructor_settings: Mapping[str, object] | None = None,
) -> SpecificationRecord:
    initial_maximum_wavenumber = contract.target_maximum_wavenumber
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
    assignments: tuple[AttemptAssignment, ...],
    samples: tuple[SampleT, ...],
    records: tuple[SpecificationRecord, ...],
    *,
    contract: RolloutConfig,
    construction_settings: SpecificationRecord,
) -> SampledSimulations[SampleT]:
    return SampledSimulations(
        assignments=assignments,
        samples=samples,
        specification_records=records,
        contract=contract,
        construction_settings=construction_settings,
    )


def prepare_trajectory_batch(
    sampled: SampledSimulations[SampleT],
    *,
    root: Path,
    family_name: str,
    batch_id: int,
    metadata: Mapping[str, object],
) -> PreparedTrajectoryBatch[SampleT]:
    """Build the in-memory description for one sampled trajectory batch."""

    batch_plan = build_batch_plan(
        sampled.assignments,
        sampled.specification_records,
        metadata=metadata,
    )
    path = batch_path(
        root,
        family=family_name,
        split=sampled.assignments[0].simulation_key.dataset_split.value,
        batch_id=batch_id,
    )
    return PreparedTrajectoryBatch(
        sampled=sampled,
        path=path,
        batch_plan=batch_plan,
    )


def _project_once(
    eta0: FloatArray | jax.Array,
    xi0: FloatArray | jax.Array,
    *,
    contract: RolloutConfig,
) -> tuple[FloatArray, FloatArray]:
    """Apply the common fixed-band map, removing only the ``xi`` zero mode."""

    eta = jnp.asarray(eta0, dtype=jnp.float64)
    xi = jnp.asarray(xi0, dtype=jnp.float64)
    if eta.ndim != 2 or eta.shape != xi.shape or eta.shape[1] != contract.nx:
        raise ValueError(f"raw fields must have common shape (batch, {contract.nx})")
    _, wavenumbers = build_grid(contract.nx, contract.length)
    k = jnp.asarray(wavenumbers, dtype=jnp.float64)
    maximum_wavenumber = contract.target_maximum_wavenumber
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
    contract: RolloutConfig,
) -> TrajectoryInitialBatch:
    eta, xi = _project_once(eta0, xi0, contract=contract)
    return TrajectoryInitialBatch(
        eta0=eta,
        xi0=xi,
        depths=np.asarray(depths, dtype=np.float64),
        specification_records=specification_records,
    )


def sample_tanaka_simulations(
    assignments: Sequence[AttemptAssignment],
    *,
    contract: RolloutConfig,
) -> SampledSimulations[TanakaSample]:
    """Sample complete Tanaka simulations without running the Tanaka solver."""

    attempted = _require_assignments(assignments)
    samples = tuple(
        sample_tanaka_simulation(
            assignment,
            domain_length=contract.length,
        )
        for assignment in attempted
    )
    settings: SpecificationRecord = {
        "dimensionless_template_depth": 1.0,
        "crest_spec_amplitude_is_alpha": True,
    }
    records = tuple(
        _strict_record(
            sample.to_json_record(),
            constructor=TANAKA_PROFILE_RECONSTRUCTION,
            contract=contract,
            constructor_settings=settings,
        )
        for sample in samples
    )
    return _sampled_simulations(
        attempted,
        samples,
        records,
        contract=contract,
        construction_settings=settings,
    )


def construct_tanaka_trajectory_batch(
    proposed: PreparedTrajectoryBatch[TanakaSample],
    *,
    selected_local_indices: Sequence[int] | None = None,
) -> TrajectoryInitialBatch:
    """Construct tangent-Hermite states for the selected simulations.

    ``selected_local_indices`` retains valid siblings after a declared
    per-simulation construction failure.
    """

    sampled = proposed.sampled
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
    contract = sampled.contract
    depths = np.asarray(
        [sample.depth for sample in selected_samples],
        dtype=np.float64,
    )
    simulation_specs = [list(sample.crests) for sample in selected_samples]
    template = make_default_tanaka_template(
        depth=1.0,
        gravity=contract.gravity,
        direction=1,
        nx=contract.nx,
        length=contract.length,
        center=0.0,
        dno_order=contract.dno_order,
        pad_factor=contract.pad_factor,
    )
    eta0, xi0 = build_per_simulation_initial_conditions(
        template_params=template,
        simulation_h_ref=depths,
        simulation_specs=simulation_specs,
        length=contract.length,
        nx=contract.nx,
        gravity=contract.gravity,
    )
    return _batch(
        eta0,
        xi0,
        depths,
        selected_records,
        contract=contract,
    )


def sample_benjamin_feir_simulations(
    assignments: Sequence[AttemptAssignment],
    *,
    contract: RolloutConfig,
) -> SampledSimulations[BenjaminFeirSample]:
    """Sample complete Benjamin--Feir simulations without numerical construction."""

    attempted = _require_assignments(assignments)
    samples = tuple(
        sample_benjamin_feir_simulation(
            assignment,
            domain_length=contract.length,
        )
        for assignment in attempted
    )
    records = tuple(
        _strict_record(
            sample.to_json_record(),
            constructor=BENJAMIN_FEIR_CONSTRUCTOR,
            contract=contract,
        )
        for sample in samples
    )
    return _sampled_simulations(
        attempted,
        samples,
        records,
        contract=contract,
        construction_settings={},
    )


def construct_benjamin_feir_trajectory_batch(
    proposed: PreparedTrajectoryBatch[BenjaminFeirSample],
) -> TrajectoryInitialBatch:
    """Construct Benjamin--Feir initial states."""

    sampled = proposed.sampled
    samples = sampled.samples
    if not all(isinstance(sample, BenjaminFeirSample) for sample in samples):
        raise TypeError("Benjamin--Feir construction requires Benjamin--Feir samples")
    expected_records = sampled.specification_records
    contract = sampled.contract
    per_simulation_parameters = tuple(
        sample.to_parameter_arrays() for sample in samples
    )
    parameter_names = tuple(per_simulation_parameters[0])
    parameters: ParameterArrays = {
        name: np.concatenate(
            tuple(
                simulation_parameters[name]
                for simulation_parameters in per_simulation_parameters
            )
        )
        for name in parameter_names
    }
    x, _ = build_grid(contract.nx, contract.length)
    eta0, xi0 = build_benjamin_feir_initial_conditions(
        x=jnp.asarray(x, dtype=jnp.float64),
        parameters=parameters,
        length=contract.length,
        gravity=contract.gravity,
        dtype=jnp.float64,
    )
    depths = np.asarray([sample.depth for sample in samples], dtype=np.float64)
    return _batch(
        eta0,
        xi0,
        depths,
        expected_records,
        contract=contract,
    )


def sample_jonswap_tma_simulations(
    assignments: Sequence[AttemptAssignment],
    *,
    contract: RolloutConfig,
    quadrature_order: int = 16,
) -> SampledSimulations[JonswapTmaSample]:
    """Sample complete JONSWAP/TMA simulations and both phase arrays."""

    attempted = _require_assignments(assignments)
    band = resolved_band_for_contract(
        contract,
        quadrature_order=quadrature_order,
    )
    samples = tuple(
        sample_jonswap_tma_simulation(assignment, band=band) for assignment in attempted
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
            sample.to_json_record(),
            constructor=constructor,
            contract=contract,
            constructor_settings=settings,
        )
        for sample in samples
    )
    return _sampled_simulations(
        attempted,
        samples,
        records,
        contract=contract,
        construction_settings=settings,
    )


def construct_jonswap_tma_trajectory_batch(
    proposed: PreparedTrajectoryBatch[JonswapTmaSample],
    *,
    selected_local_indices: Sequence[int] | None = None,
) -> TrajectoryInitialBatch:
    """Construct resolved-band states for the selected simulations."""

    sampled = proposed.sampled
    samples = sampled.samples
    if not all(isinstance(sample, JonswapTmaSample) for sample in samples):
        raise TypeError("JONSWAP/TMA construction requires JONSWAP/TMA samples")
    density_window = sampled.construction_settings.get("density_window")
    if density_window != PAPER_RELATIVE_FREQUENCY_WINDOW:
        raise ValueError("JONSWAP/TMA density window does not match the contract")
    relative_minimum = sampled.construction_settings.get("relative_frequency_minimum")
    relative_maximum = sampled.construction_settings.get("relative_frequency_maximum")
    if (
        relative_minimum != PAPER_RELATIVE_FREQUENCY_MINIMUM
        or relative_maximum != PAPER_RELATIVE_FREQUENCY_MAXIMUM
    ):
        raise ValueError(
            "JONSWAP/TMA relative frequency interval does not match the contract"
        )
    relative_frequency_interval = (
        PAPER_RELATIVE_FREQUENCY_MINIMUM,
        PAPER_RELATIVE_FREQUENCY_MAXIMUM,
    )
    quadrature_order = sampled.construction_settings.get("quadrature_order")
    if not isinstance(quadrature_order, int) or isinstance(quadrature_order, bool):
        raise TypeError("JONSWAP/TMA quadrature_order must be an integer")
    contract = sampled.contract
    band = resolved_band_for_contract(
        contract,
        quadrature_order=quadrature_order,
    )
    expected_records = sampled.specification_records
    selected = _resolve_selected_simulation_indices(
        selected_local_indices,
        simulation_count=len(samples),
    )

    selected_samples = tuple(samples[index] for index in selected)
    selected_records = tuple(expected_records[index] for index in selected)
    x = contract.length * np.arange(contract.nx, dtype=np.float64) / contract.nx
    states = tuple(
        build_jonswap_tma_initial_condition(
            x,
            parameters=sample.parameters,
            phase_right=sample.phase_right,
            phase_left=sample.phase_left,
            band=band,
            gravity=contract.gravity,
            relative_frequency_interval=relative_frequency_interval,
        )
        for sample in selected_samples
    )
    raw_eta0 = np.stack(tuple(state.eta for state in states))
    raw_xi0 = np.stack(tuple(state.xi for state in states))
    eta0, xi0 = _project_once(raw_eta0, raw_xi0, contract=contract)
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
        simulations: list[dict[str, object]] = []
        for index in invalid:
            local_index = int(index)
            finite = bool(state_finite[local_index])
            simulations.append(
                {
                    "local_simulation_index": local_index,
                    "failure_reason": (
                        "nonpositive_initial_water_column"
                        if finite
                        else "nonfinite_initial_state"
                    ),
                    "state_finite": finite,
                    "minimum_water_column": (
                        float(minimum_water_columns[local_index]) if finite else None
                    ),
                }
            )
        failure_record: dict[str, object] = {
            "schema": "jonswap_initial_state_domain_failure_v1",
            "reason": "invalid_initial_graph_state",
            "index_space": "constructor_subbatch_local_index",
            "invalid_simulation_indices": [int(index) for index in invalid],
            "simulations": simulations,
        }
        raise JonswapInitialStateDomainError(failure_record)

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
                length=contract.length,
                gravity=contract.gravity,
            )
            for index, (sample, state) in enumerate(zip(selected_samples, states))
        ),
    )
