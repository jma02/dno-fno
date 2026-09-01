"""Generate and validate one trajectory batch for each rollout family."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np

from solver.gen_data.pipeline.artifact_io import json_text, parse_json
from solver.gen_data.benjamin_feir_sampling import (
    BENJAMIN_FEIR_PARAMETER_GROUPS,
    BenjaminFeirSample,
)
from solver.gen_data.jonswap_tma import finite_depth_angular_frequency
from solver.gen_data.jonswap_tma_sampling import (
    JONSWAP_TMA_PARAMETER_GROUPS,
    JonswapTmaSample,
)
from solver.gen_data.pipeline.simulation_allocation import (
    AttemptAssignment,
    PhysicalFamilyId,
)
from solver.gen_data.pipeline.simulation_checks import (
    SimulationCheckResult,
    SimulationCheck,
)
from solver.gen_data.pipeline.dataset_generation import DatasetChunkConfig
from solver.gen_data.pipeline.trajectory_config import (
    PAPER_BENJAMIN_FEIR_ROLLOUT_CONFIG,
    PAPER_JONSWAP_ROLLOUT_CONFIG,
    PAPER_TANAKA_ROLLOUT_CONFIG,
    RolloutConfig,
)
from solver.gen_data.pipeline.trajectory_integration import (
    BatchIntegrator,
    integrate_batch,
)
from solver.gen_data.pipeline.trajectory_rollout import (
    execute_trajectory_batch,
)
from solver.gen_data.pipeline.trajectory_subsampling import (
    TrajectoryFrameSelectionConfig,
    TrajectoryFamily,
    subsample_trajectories,
)
from solver.gen_data.pipeline.writer import (
    SimulationOutcome,
    commit_simulation_outcomes,
)
from solver.gen_data.tanaka_sampling import TANAKA_PARAMETER_GROUPS
from solver.gen_data.tanaka_sampling import TanakaSample
from solver.gen_data.trajectory_family_adapters import (
    JonswapInitialStateDomainError,
    PreparedTrajectoryBatch,
    SampledSimulations,
    TrajectoryInitialBatch,
    construct_benjamin_feir_trajectory_batch,
    construct_jonswap_tma_trajectory_batch,
    construct_tanaka_trajectory_batch,
    prepare_trajectory_batch,
    sample_benjamin_feir_simulations,
    sample_jonswap_tma_simulations,
    sample_tanaka_simulations,
)


ContractRole: TypeAlias = Literal[
    "paper_dataset",
    "reduced_wiring_evidence_only",
]
HorizonKind: TypeAlias = Literal[
    "fixed_terminal_time",
    "benjamin_feir_carrier_periods_floor_saved_grid",
    "jonswap_peak_periods_floor_saved_grid",
]
JsonScalar: TypeAlias = str | int | float | bool | None
JONSWAP_ADJUSTMENT_SCHEMA = "dommermuth_nonlinear_adjustment_v1"
JONSWAP_ADJUSTMENT_FORMULA = "A(t)=1-exp(-(t/T_a)^n)"
TRAJECTORY_REQUIRED_CHECKS = (
    SimulationCheck.OUTSIDE_SUPPORT | SimulationCheck.INCOMPLETE_TRAJECTORY
)
_TANAKA_FAILURE_REASONS = frozenset(
    {
        "negative_or_nonfinite_surface_potential_radicand",
        "nonpositive_or_nonfinite_surface_potential_speed_squared",
    }
)
_TANAKA_COMPONENT_FIELDS = frozenset(
    {
        "local_simulation_index",
        "component_within_simulation",
        "global_component_index",
        "alpha",
        "center",
        "direction",
        "depth",
        "unsigned_speed",
        "speed_squared",
        "minimum_radicand",
        "minimum_radicand_over_speed_squared",
        "minimum_radicand_grid_index",
        "minimum_radicand_x",
        "negative_count",
        "nonfinite_count",
        "first_nonfinite_grid_index",
        "first_nonfinite_x",
    }
)

_FAMILY_IDS = {
    "tanaka": PhysicalFamilyId.TANAKA,
    "benjamin_feir": PhysicalFamilyId.BENJAMIN_FEIR,
    "jonswap_tma": PhysicalFamilyId.JONSWAP_TMA,
}
_FAMILY_PARAMETER_GROUPS = {
    "tanaka": frozenset(TANAKA_PARAMETER_GROUPS),
    "benjamin_feir": frozenset(BENJAMIN_FEIR_PARAMETER_GROUPS),
    "jonswap_tma": frozenset(JONSWAP_TMA_PARAMETER_GROUPS),
}


def _paper_rollout_config(
    family: TrajectoryFamily,
) -> RolloutConfig:
    """Return the production rollout settings for one family."""

    return {
        "tanaka": PAPER_TANAKA_ROLLOUT_CONFIG,
        "benjamin_feir": PAPER_BENJAMIN_FEIR_ROLLOUT_CONFIG,
        "jonswap_tma": PAPER_JONSWAP_ROLLOUT_CONFIG,
    }[family]


def _strict_json_copy(value: Mapping[str, object]) -> Mapping[str, object]:
    parsed = parse_json(json_text(dict(value)))
    if not isinstance(parsed, dict):
        raise TypeError("value must encode a JSON object")
    return MappingProxyType(parsed)


@dataclass(frozen=True)
class TrajectoryHorizonPolicy:
    """Family-level terminal-time construction on the saved-time grid."""

    kind: HorizonKind
    fixed_terminal_time: float | None
    period_count: int | None

    def __post_init__(self) -> None:
        if self.kind == "fixed_terminal_time":
            if (
                self.fixed_terminal_time is None
                or not math.isfinite(self.fixed_terminal_time)
                or self.fixed_terminal_time <= 0.0
            ):
                raise ValueError("fixed terminal time must be finite and positive")
            if self.period_count is not None:
                raise ValueError("a fixed horizon cannot set period_count")
        elif self.kind in (
            "benjamin_feir_carrier_periods_floor_saved_grid",
            "jonswap_peak_periods_floor_saved_grid",
        ):
            if self.fixed_terminal_time is not None:
                raise ValueError("a period-count horizon cannot set a fixed time")
            if (
                isinstance(self.period_count, bool)
                or not isinstance(self.period_count, int)
                or self.period_count <= 0
            ):
                raise ValueError("period_count must be a positive integer")
        else:
            raise ValueError(f"unknown trajectory horizon kind: {self.kind}")

    def to_json_record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "fixed_terminal_time": self.fixed_terminal_time,
            "period_count": self.period_count,
            "saved_grid_rounding": (
                "exact" if self.kind == "fixed_terminal_time" else "floor"
            ),
        }


def fixed_time_horizon(terminal_time: float) -> TrajectoryHorizonPolicy:
    return TrajectoryHorizonPolicy(
        kind="fixed_terminal_time",
        fixed_terminal_time=terminal_time,
        period_count=None,
    )


def benjamin_feir_horizon(period_count: int = 100) -> TrajectoryHorizonPolicy:
    return TrajectoryHorizonPolicy(
        kind="benjamin_feir_carrier_periods_floor_saved_grid",
        fixed_terminal_time=None,
        period_count=period_count,
    )


def jonswap_horizon(period_count: int = 16) -> TrajectoryHorizonPolicy:
    return TrajectoryHorizonPolicy(
        kind="jonswap_peak_periods_floor_saved_grid",
        fixed_terminal_time=None,
        period_count=period_count,
    )


def _paper_horizon(family: TrajectoryFamily) -> TrajectoryHorizonPolicy:
    if family == "jonswap_tma":
        return jonswap_horizon()
    if family == "benjamin_feir":
        return benjamin_feir_horizon()
    return fixed_time_horizon(200.0)


@dataclass(frozen=True)
class JonswapNonlinearAdjustmentPolicy:
    """Pre-production nonlinear-adjustment contract for JONSWAP/TMA seas."""

    ramp_order: int
    ramp_time_peak_periods: int
    burn_peak_periods: int

    def __post_init__(self) -> None:
        values = (
            self.ramp_order,
            self.ramp_time_peak_periods,
            self.burn_peak_periods,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in values
        ):
            raise ValueError("JONSWAP nonlinear-adjustment integers must be positive")
        if self.burn_peak_periods < self.ramp_time_peak_periods:
            raise ValueError("JONSWAP adjustment burn must cover the ramp time")

    def to_json_record(self) -> dict[str, object]:
        return {
            "schema": JONSWAP_ADJUSTMENT_SCHEMA,
            "formula": JONSWAP_ADJUSTMENT_FORMULA,
            "ramp_application": "linear_rhs_plus_A_times_nonlinear_residual",
            "ramp_order": self.ramp_order,
            "ramp_time_peak_periods": self.ramp_time_peak_periods,
            "burn_peak_periods": self.burn_peak_periods,
            "burn_saved_grid_rounding": "floor",
            "handoff": "full_internal_band_at_each_simulation_endpoint",
            "production_ramp": "disabled",
            "autonomous_clock": "restart_at_zero",
            "hamiltonian_scope": "autonomous_production_only",
        }


PAPER_JONSWAP_ADJUSTMENT_POLICY = JonswapNonlinearAdjustmentPolicy(
    ramp_order=4,
    ramp_time_peak_periods=10,
    burn_peak_periods=20,
)


@dataclass(frozen=True)
class TrajectoryExecutionConfig:
    """Full numerical and retained-time contract for one rollout family."""

    family: TrajectoryFamily
    role: ContractRole
    numerical: RolloutConfig
    horizon: TrajectoryHorizonPolicy
    frame_selection: TrajectoryFrameSelectionConfig
    jonswap_quadrature_order: int | None
    jonswap_adjustment: JonswapNonlinearAdjustmentPolicy | None = None

    def __post_init__(self) -> None:
        if self.family not in _FAMILY_IDS:
            raise ValueError(f"unknown trajectory family: {self.family}")
        if self.role not in (
            "paper_dataset",
            "reduced_wiring_evidence_only",
        ):
            raise ValueError(f"unknown trajectory contract role: {self.role}")
        if self.family == "jonswap_tma":
            if (
                isinstance(self.jonswap_quadrature_order, bool)
                or not isinstance(self.jonswap_quadrature_order, int)
                or self.jonswap_quadrature_order < 2
            ):
                raise ValueError("JONSWAP/TMA quadrature_order must be at least two")
            if self.horizon.kind != "jonswap_peak_periods_floor_saved_grid":
                raise ValueError(
                    "JONSWAP/TMA requires a per-simulation peak-period horizon"
                )
        elif self.family == "benjamin_feir":
            if self.jonswap_quadrature_order is not None:
                raise ValueError("only JONSWAP/TMA may set a quadrature order")
            if self.horizon.kind != "benjamin_feir_carrier_periods_floor_saved_grid":
                raise ValueError(
                    "Benjamin--Feir requires a per-simulation carrier-period horizon"
                )
            if self.jonswap_adjustment is not None:
                raise ValueError(
                    "only JONSWAP/TMA may set a nonlinear-adjustment policy"
                )
        else:
            if self.jonswap_quadrature_order is not None:
                raise ValueError("only JONSWAP/TMA may set a quadrature order")
            if self.horizon.kind != "fixed_terminal_time":
                raise ValueError("Tanaka requires a fixed horizon")
            if self.jonswap_adjustment is not None:
                raise ValueError(
                    "only JONSWAP/TMA may set a nonlinear-adjustment policy"
                )

        if self.role == "paper_dataset":
            expected_quadrature = 16 if self.family == "jonswap_tma" else None
            if (
                self.numerical != _paper_rollout_config(self.family)
                or self.horizon != _paper_horizon(self.family)
                or self.frame_selection != TrajectoryFrameSelectionConfig()
                or self.jonswap_quadrature_order != expected_quadrature
                or self.jonswap_adjustment
                != (
                    PAPER_JONSWAP_ADJUSTMENT_POLICY
                    if self.family == "jonswap_tma"
                    else None
                )
            ):
                raise ValueError(
                    "a reduced trajectory execution must be labeled "
                    "'reduced_wiring_evidence_only'"
                )

    def to_json_record(self) -> dict[str, object]:
        numerical = asdict(self.numerical)
        if numerical["internal_hamiltonian_drift_threshold"] is None:
            numerical.pop("internal_hamiltonian_drift_threshold")
        record: dict[str, object] = {
            "family": self.family,
            "role": self.role,
            "numerical": numerical,
            "horizon": self.horizon.to_json_record(),
            "frame_selection": asdict(self.frame_selection),
            "jonswap_quadrature_order": self.jonswap_quadrature_order,
        }
        if self.jonswap_adjustment is not None:
            record["jonswap_adjustment"] = self.jonswap_adjustment.to_json_record()
        return record


def paper_trajectory_execution(
    family: TrajectoryFamily,
) -> TrajectoryExecutionConfig:
    """Return the production settings for one trajectory family."""

    return TrajectoryExecutionConfig(
        family=family,
        role="paper_dataset",
        numerical=_paper_rollout_config(family),
        horizon=_paper_horizon(family),
        frame_selection=TrajectoryFrameSelectionConfig(),
        jonswap_quadrature_order=16 if family == "jonswap_tma" else None,
        jonswap_adjustment=(
            PAPER_JONSWAP_ADJUSTMENT_POLICY if family == "jonswap_tma" else None
        ),
    )


@dataclass(frozen=True)
class SimulationTimeGrid:
    """Intended and realized terminal times for one proposed simulation."""

    intended_terminal_time: float
    realized_terminal_time: float
    saved_times: np.ndarray

    def to_json_record(self) -> dict[str, object]:
        return {
            "intended_terminal_time": self.intended_terminal_time,
            "realized_terminal_time": self.realized_terminal_time,
            "saved_time_count": int(self.saved_times.size),
        }


def floor_saved_time_grid(
    terminal_time: float,
    *,
    saved_dt: float,
    horizon_name: str,
) -> np.ndarray:
    """Return the saved-time prefix ending immediately before a horizon."""

    step_count = math.floor(terminal_time / saved_dt)
    while step_count * saved_dt > terminal_time:
        step_count -= 1
    while (step_count + 1) * saved_dt <= terminal_time:
        step_count += 1
    if step_count < 1:
        raise ValueError(f"{horizon_name} horizon is shorter than one saved step")
    saved_times = saved_dt * np.arange(step_count + 1, dtype=np.float64)
    realized = float(saved_times[-1])
    if not (realized <= terminal_time and terminal_time - realized < saved_dt):
        raise RuntimeError(
            f"{horizon_name} saved-grid horizon was not strictly floored"
        )
    return saved_times


@dataclass(frozen=True)
class DeclaredConstructionFailure:
    """Declared invalid positions in the constructor's current sub-batch."""

    local_indices: tuple[int, ...]
    record: Mapping[str, object]


class ConstructionFailureClassifier(Protocol):
    def __call__(
        self,
        error: Exception,
    ) -> DeclaredConstructionFailure | None: ...


class TrajectoryConstructor(Protocol):
    def __call__(
        self,
        proposed: PreparedTrajectoryBatch[Any],
        *,
        selected_local_indices: tuple[int, ...] | None,
    ) -> TrajectoryInitialBatch: ...


def _finite_json_number(
    value: object,
    *,
    field_name: str,
    allow_none: bool,
) -> float | None:
    if value is None and allow_none:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        suffix = " or null" if allow_none else ""
        raise ValueError(f"Tanaka component {field_name} must be finite{suffix}")
    return float(value)


def _nonnegative_json_integer(
    value: object,
    *,
    field_name: str,
    allow_none: bool = False,
) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        suffix = " or null" if allow_none else ""
        raise ValueError(
            f"Tanaka component {field_name} must be a nonnegative integer{suffix}"
        )
    return value


def classify_tanaka_construction_failure(
    error: Exception,
) -> DeclaredConstructionFailure | None:
    """Recognize only a finite failure of the Tanaka square-root domain."""

    try:
        from solver.gen_data.tanaka_initial_conditions import (
            TanakaPotentialRadicandError,
        )
    except ImportError:
        return None
    if not isinstance(error, TanakaPotentialRadicandError):
        return None
    raw_record = error.failure_record
    if not isinstance(raw_record, dict):
        raise TypeError("Tanaka construction failure_record must be a dictionary")
    record = _strict_json_copy(raw_record)
    if record.get("schema") != "tanaka_potential_radicand_failure_v1":
        raise ValueError("unrecognized Tanaka construction-failure schema")
    if record.get("reason") not in _TANAKA_FAILURE_REASONS:
        raise ValueError("unrecognized Tanaka construction-failure reason")
    indices = record.get("invalid_simulation_indices")
    if not isinstance(indices, list) or not indices:
        raise TypeError(
            "Tanaka construction failure must list invalid_simulation_indices"
        )
    if (
        any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
        or tuple(sorted(set(indices))) != tuple(indices)
        or indices[0] < 0
    ):
        raise ValueError(
            "Tanaka invalid_simulation_indices must be unique, increasing, "
            "nonnegative integers"
        )
    components = record.get("components")
    if not isinstance(components, list) or not components:
        raise TypeError("Tanaka construction failure must list invalid components")
    component_simulations: set[int] = set()
    component_identities: set[tuple[int, int, int]] = set()
    global_component_indices: list[int] = []
    component_failures: list[tuple[bool, bool]] = []
    has_unclassified_component = False
    for component in components:
        if not isinstance(component, dict):
            raise TypeError("every Tanaka invalid component must be a JSON object")
        missing = _TANAKA_COMPONENT_FIELDS.difference(component)
        if missing:
            raise ValueError(
                f"Tanaka invalid component is missing fields: {sorted(missing)}"
            )
        local_simulation_index = component.get("local_simulation_index")
        component_within_simulation = component.get("component_within_simulation")
        global_component_index = component.get("global_component_index")
        local_simulation_index = _nonnegative_json_integer(
            local_simulation_index,
            field_name="local_simulation_index",
        )
        component_within_simulation = _nonnegative_json_integer(
            component_within_simulation,
            field_name="component_within_simulation",
        )
        global_component_index = _nonnegative_json_integer(
            global_component_index,
            field_name="global_component_index",
        )
        assert local_simulation_index is not None
        assert component_within_simulation is not None
        assert global_component_index is not None
        identity = (
            local_simulation_index,
            component_within_simulation,
            global_component_index,
        )
        if identity in component_identities:
            raise ValueError("Tanaka construction failure repeats a component")
        component_identities.add(identity)
        global_component_indices.append(global_component_index)
        component_simulations.add(local_simulation_index)

        alpha = _finite_json_number(
            component.get("alpha"),
            field_name="alpha",
            allow_none=False,
        )
        _finite_json_number(
            component.get("center"),
            field_name="center",
            allow_none=False,
        )
        depth = _finite_json_number(
            component.get("depth"),
            field_name="depth",
            allow_none=False,
        )
        if alpha is None or alpha <= 0.0:
            raise ValueError("Tanaka component alpha must be positive")
        if depth is None or depth <= 0.0:
            raise ValueError("Tanaka component depth must be positive")
        direction = component.get("direction")
        if (
            isinstance(direction, bool)
            or not isinstance(direction, int)
            or direction not in (-1, 1)
        ):
            raise ValueError("Tanaka component direction must equal -1 or +1")

        unsigned_speed = _finite_json_number(
            component.get("unsigned_speed"),
            field_name="unsigned_speed",
            allow_none=True,
        )
        speed_squared = _finite_json_number(
            component.get("speed_squared"),
            field_name="speed_squared",
            allow_none=True,
        )
        if unsigned_speed is not None and unsigned_speed < 0.0:
            raise ValueError("Tanaka component unsigned_speed must be nonnegative")
        if speed_squared is not None and speed_squared < 0.0:
            raise ValueError("Tanaka component speed_squared must be nonnegative")
        if (unsigned_speed is None) != (speed_squared is None):
            raise ValueError(
                "Tanaka component speed fields must both be finite or both be null"
            )
        if (
            unsigned_speed is not None
            and speed_squared is not None
            and not math.isclose(
                speed_squared,
                unsigned_speed * unsigned_speed,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        ):
            raise ValueError(
                "Tanaka component speed_squared disagrees with unsigned_speed"
            )

        negative_count = _nonnegative_json_integer(
            component.get("negative_count"),
            field_name="negative_count",
        )
        nonfinite_count = _nonnegative_json_integer(
            component.get("nonfinite_count"),
            field_name="nonfinite_count",
        )
        assert negative_count is not None
        assert nonfinite_count is not None
        minimum = _finite_json_number(
            component.get("minimum_radicand"),
            field_name="minimum_radicand",
            allow_none=True,
        )
        minimum_scaled = _finite_json_number(
            component.get("minimum_radicand_over_speed_squared"),
            field_name="minimum_radicand_over_speed_squared",
            allow_none=True,
        )
        minimum_index = _nonnegative_json_integer(
            component.get("minimum_radicand_grid_index"),
            field_name="minimum_radicand_grid_index",
            allow_none=True,
        )
        minimum_x = _finite_json_number(
            component.get("minimum_radicand_x"),
            field_name="minimum_radicand_x",
            allow_none=True,
        )
        first_nonfinite_index = _nonnegative_json_integer(
            component.get("first_nonfinite_grid_index"),
            field_name="first_nonfinite_grid_index",
            allow_none=True,
        )
        first_nonfinite_x = _finite_json_number(
            component.get("first_nonfinite_x"),
            field_name="first_nonfinite_x",
            allow_none=True,
        )
        if nonfinite_count:
            if (
                minimum is not None
                or minimum_scaled is not None
                or minimum_index is not None
                or minimum_x is not None
                or first_nonfinite_index is None
                or first_nonfinite_x is None
            ):
                raise ValueError(
                    "Tanaka nonfinite-radicand diagnostics are inconsistent"
                )
        else:
            if first_nonfinite_index is not None or first_nonfinite_x is not None:
                raise ValueError(
                    "Tanaka finite-radicand diagnostics name a nonfinite point"
                )
            if minimum is None or minimum_index is None or minimum_x is None:
                raise ValueError("Tanaka finite-radicand diagnostics omit the minimum")
            if speed_squared is not None and speed_squared > 0.0:
                if minimum_scaled is None:
                    raise ValueError(
                        "Tanaka radicand diagnostics omit the scaled minimum"
                    )
                if not math.isclose(
                    minimum_scaled,
                    minimum / speed_squared,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                ):
                    raise ValueError(
                        "Tanaka scaled minimum disagrees with minimum_radicand "
                        "and speed_squared"
                    )
            elif minimum_scaled is not None:
                raise ValueError(
                    "Tanaka invalid-speed diagnostics contain a scaled minimum"
                )
        if minimum is not None and (minimum < 0.0) != (negative_count > 0):
            raise ValueError(
                "Tanaka negative-radicand count disagrees with its minimum"
            )
        invalid_speed = speed_squared is None or speed_squared <= 0.0
        invalid_radicand = negative_count > 0 or nonfinite_count > 0
        if not invalid_speed and not invalid_radicand:
            raise ValueError("Tanaka component does not describe a domain failure")
        component_failures.append((invalid_speed, invalid_radicand))
        has_unclassified_component |= invalid_speed or nonfinite_count > 0

    if component_simulations != set(indices):
        raise ValueError(
            "Tanaka invalid components do not match invalid_simulation_indices"
        )
    if global_component_indices != sorted(set(global_component_indices)):
        raise ValueError(
            "Tanaka global component indices must be unique and increasing"
        )
    reason = record["reason"]
    if reason == "negative_or_nonfinite_surface_potential_radicand":
        if any(
            invalid_speed or not invalid_radicand
            for invalid_speed, invalid_radicand in component_failures
        ):
            raise ValueError(
                "Tanaka radicand-failure reason disagrees with its components"
            )
    elif not any(invalid_speed for invalid_speed, _ in component_failures):
        raise ValueError("Tanaka speed-failure reason has no invalid speed component")
    if has_unclassified_component:
        return None
    return DeclaredConstructionFailure(
        local_indices=tuple(indices),
        record=record,
    )


def classify_jonswap_construction_failure(
    error: Exception,
) -> DeclaredConstructionFailure | None:
    """Recognize a per-simulation JONSWAP initial graph-domain failure."""

    if not isinstance(error, JonswapInitialStateDomainError):
        return None
    raw_record = error.failure_record
    if not isinstance(raw_record, dict):
        raise TypeError("JONSWAP construction failure_record must be a dictionary")
    record = _strict_json_copy(raw_record)
    if record.get("schema") != "jonswap_initial_state_domain_failure_v1":
        raise ValueError("unrecognized JONSWAP construction-failure schema")
    if record.get("reason") != "invalid_initial_graph_state":
        raise ValueError("unrecognized JONSWAP construction-failure reason")
    if record.get("index_space") != "constructor_subbatch_local_index":
        raise ValueError("unrecognized JONSWAP construction-failure index space")

    indices = record.get("invalid_simulation_indices")
    if not isinstance(indices, list) or not indices:
        raise TypeError("JONSWAP construction failure must list invalid indices")
    if (
        any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
        or tuple(sorted(set(indices))) != tuple(indices)
        or indices[0] < 0
    ):
        raise ValueError(
            "JONSWAP invalid_simulation_indices must be unique, increasing, "
            "nonnegative integers"
        )

    simulations = record.get("simulations")
    if not isinstance(simulations, list) or not simulations:
        raise TypeError("JONSWAP construction failure must list invalid simulations")
    observed_indices: list[int] = []
    for simulation in simulations:
        if not isinstance(simulation, dict):
            raise TypeError("every invalid JONSWAP simulation must be a JSON object")
        local_index = simulation.get("local_simulation_index")
        if isinstance(local_index, bool) or not isinstance(local_index, int):
            raise TypeError("JONSWAP local_simulation_index must be an integer")
        observed_indices.append(local_index)
        state_finite = simulation.get("state_finite")
        if not isinstance(state_finite, bool):
            raise TypeError("JONSWAP state_finite must be Boolean")
        minimum = simulation.get("minimum_water_column")
        failure_reason = simulation.get("failure_reason")
        if state_finite:
            if (
                isinstance(minimum, bool)
                or not isinstance(minimum, (int, float))
                or not math.isfinite(float(minimum))
                or float(minimum) > 0.0
            ):
                raise ValueError(
                    "finite JONSWAP domain failures require a nonpositive "
                    "minimum water column"
                )
            if failure_reason != "nonpositive_initial_water_column":
                raise ValueError("JONSWAP bottom-failure reason is inconsistent")
        else:
            if minimum is not None:
                raise ValueError(
                    "nonfinite JONSWAP domain failures require a null minimum"
                )
            if failure_reason != "nonfinite_initial_state":
                raise ValueError("JONSWAP nonfinite-failure reason is inconsistent")
    if observed_indices != indices:
        raise ValueError(
            "JONSWAP invalid simulations do not match invalid_simulation_indices"
        )
    return DeclaredConstructionFailure(
        local_indices=tuple(indices),
        record=record,
    )


def classify_trajectory_construction_failure(
    error: Exception,
) -> DeclaredConstructionFailure | None:
    """Dispatch declared per-simulation constructor failures."""

    tanaka_failure = classify_tanaka_construction_failure(error)
    if tanaka_failure is not None:
        return tanaka_failure
    return classify_jonswap_construction_failure(error)


def _fixed_time_grid(
    horizon: TrajectoryHorizonPolicy,
    contract: RolloutConfig,
) -> SimulationTimeGrid:
    assert horizon.fixed_terminal_time is not None
    terminal = horizon.fixed_terminal_time
    step_count = int(round(terminal / contract.saved_dt))
    if step_count < 1 or not math.isclose(
        step_count * contract.saved_dt,
        terminal,
        rel_tol=0.0,
        abs_tol=2.0e-13,
    ):
        raise ValueError("fixed terminal time must lie exactly on the saved-time grid")
    saved_times = contract.saved_dt * np.arange(
        step_count + 1,
        dtype=np.float64,
    )
    return SimulationTimeGrid(
        intended_terminal_time=terminal,
        realized_terminal_time=float(saved_times[-1]),
        saved_times=saved_times,
    )


def _jonswap_time_grid(
    sample: JonswapTmaSample,
    execution: TrajectoryExecutionConfig,
) -> SimulationTimeGrid:
    count = execution.horizon.period_count
    assert count is not None
    parameters = sample.parameters
    angular_frequency = float(
        finite_depth_angular_frequency(
            np.asarray([parameters.peak_wavenumber], dtype=np.float64),
            depth=parameters.depth,
            gravity=execution.numerical.gravity,
        )[0]
    )
    intended = count * 2.0 * math.pi / angular_frequency
    saved_times = floor_saved_time_grid(
        intended,
        saved_dt=execution.numerical.saved_dt,
        horizon_name="JONSWAP/TMA",
    )
    return SimulationTimeGrid(
        intended_terminal_time=intended,
        realized_terminal_time=float(saved_times[-1]),
        saved_times=saved_times,
    )


def _benjamin_feir_time_grid(
    sample: BenjaminFeirSample,
    execution: TrajectoryExecutionConfig,
) -> SimulationTimeGrid:
    """Return a strictly floored horizon measured in carrier periods."""

    count = execution.horizon.period_count
    assert count is not None
    angular_frequency = math.sqrt(
        execution.numerical.gravity * sample.carrier_wavenumber
    )
    intended = count * 2.0 * math.pi / angular_frequency
    saved_times = floor_saved_time_grid(
        intended,
        saved_dt=execution.numerical.saved_dt,
        horizon_name="Benjamin--Feir",
    )
    return SimulationTimeGrid(
        intended_terminal_time=intended,
        realized_terminal_time=float(saved_times[-1]),
        saved_times=saved_times,
    )


def _construction_rejection(
    failure: DeclaredConstructionFailure,
    *,
    original_local_index: int,
) -> SimulationOutcome:
    evaluated = SimulationCheck.OUTSIDE_SUPPORT
    failed = SimulationCheck.OUTSIDE_SUPPORT
    failure_reason = failure.record.get("reason")
    if failure.record.get("schema") == "jonswap_initial_state_domain_failure_v1":
        simulations = failure.record.get("simulations")
        if not isinstance(simulations, list):
            raise TypeError(
                "JONSWAP construction failure must list invalid simulations"
            )
        matching = tuple(
            simulation
            for simulation in simulations
            if isinstance(simulation, dict)
            and simulation.get("local_simulation_index") == original_local_index
        )
        if len(matching) != 1:
            raise RuntimeError(
                "JONSWAP construction failure does not name the rejected simulation"
            )
        state_finite = matching[0].get("state_finite")
        if not isinstance(state_finite, bool):
            raise TypeError("JONSWAP state_finite must be Boolean")
        evaluated |= SimulationCheck.NONFINITE_STATE
        if state_finite:
            evaluated |= SimulationCheck.BOTTOM_CLEARANCE
            failed |= SimulationCheck.BOTTOM_CLEARANCE
        else:
            failed |= SimulationCheck.NONFINITE_STATE
    metrics: dict[str, JsonScalar] = {
        "construction_status": "declared_outside_support",
        "construction_failure_reason": (
            failure_reason if isinstance(failure_reason, str) else ""
        ),
        "original_local_index": original_local_index,
        "construction_failure_json": json_text(dict(failure.record)),
    }
    return SimulationOutcome(
        decision=SimulationCheckResult(
            required=TRAJECTORY_REQUIRED_CHECKS,
            evaluated=evaluated,
            failed=failed,
        ),
        rows=None,
        metrics=metrics,
    )


def _finalize_outcome(
    outcome: SimulationOutcome,
    grid: SimulationTimeGrid,
    *,
    construction_metrics: Mapping[str, JsonScalar] | None = None,
    support_evaluated: bool = False,
) -> SimulationOutcome:
    """Attach execution metadata and mark support checked after construction."""

    decision = outcome.decision
    if support_evaluated:
        decision = SimulationCheckResult(
            required=decision.required | SimulationCheck.OUTSIDE_SUPPORT,
            evaluated=decision.evaluated | SimulationCheck.OUTSIDE_SUPPORT,
            failed=decision.failed,
        )
    return SimulationOutcome(
        decision=decision,
        rows=outcome.rows,
        metrics={
            **outcome.metrics,
            **(construction_metrics or {}),
            "intended_terminal_time": grid.intended_terminal_time,
            "realized_terminal_time": grid.realized_terminal_time,
            "saved_time_count": int(grid.saved_times.size),
        },
    )


@dataclass(frozen=True)
class TrajectoryBatchExecutor:
    """Execute one rollout-data batch."""

    chunk_config: DatasetChunkConfig
    execution: TrajectoryExecutionConfig
    metadata: Mapping[str, object] | None = None
    rollout_executor: BatchIntegrator = integrate_batch
    constructor: TrajectoryConstructor | None = None
    construction_failure_classifier: ConstructionFailureClassifier = (
        classify_trajectory_construction_failure
    )

    def __post_init__(self) -> None:
        expected_family_id = _FAMILY_IDS[self.execution.family]
        if self.chunk_config.family_name != self.execution.family:
            raise ValueError("chunk family name differs from trajectory execution")
        if self.chunk_config.family_id is not expected_family_id:
            raise ValueError("chunk family ID differs from trajectory execution")
        unknown_parameter_groups = {
            target.parameter_group_id for target in self.chunk_config.simulation_targets
        }.difference(_FAMILY_PARAMETER_GROUPS[self.execution.family])
        if unknown_parameter_groups:
            raise ValueError(
                f"unknown {self.execution.family} parameter groups: "
                f"{sorted(unknown_parameter_groups)}"
            )
        if (
            self.execution.role == "paper_dataset"
            and self.execution.family == "jonswap_tma"
            and self._implemented_jonswap_adjustment()
            != self.execution.jonswap_adjustment
        ):
            raise ValueError(
                "paper-dataset JONSWAP/TMA requires the nonlinear-adjustment executor"
            )

        chunk_configuration = self.chunk_config.to_json_record()["configuration"]
        assert isinstance(chunk_configuration, dict)
        configured_execution = chunk_configuration.get("trajectory_execution")
        if configured_execution != self.execution.to_json_record():
            raise ValueError(
                "chunk configuration differs from the supplied trajectory execution"
            )
        if self.execution.role == "paper_dataset" and (
            self.rollout_executor is not integrate_batch
            or self.constructor is not None
            or (
                self.construction_failure_classifier
                is not classify_trajectory_construction_failure
            )
        ):
            raise ValueError(
                "paper-dataset execution requires the production constructor, "
                "rollout executor, and construction-failure classifier"
            )
        object.__setattr__(
            self,
            "metadata",
            _strict_json_copy(self.metadata or {}),
        )

    def _implemented_jonswap_adjustment(
        self,
    ) -> JonswapNonlinearAdjustmentPolicy | None:
        """Return the adjustment policy actually implemented by this executor."""

        return None

    def _sample(
        self,
        assignments: tuple[AttemptAssignment, ...],
    ) -> SampledSimulations[Any]:
        contract = self.execution.numerical
        if self.execution.family == "tanaka":
            sampled = sample_tanaka_simulations(
                assignments,
                contract=contract,
            )
        elif self.execution.family == "benjamin_feir":
            sampled = sample_benjamin_feir_simulations(
                assignments,
                contract=contract,
            )
        else:
            quadrature_order = self.execution.jonswap_quadrature_order
            assert quadrature_order is not None
            sampled = sample_jonswap_tma_simulations(
                assignments,
                contract=contract,
                quadrature_order=quadrature_order,
            )
        if sampled.assignments != assignments or sampled.contract != contract:
            raise RuntimeError("trajectory sampler changed assignments or contract")
        return sampled

    def _time_grids(
        self,
        sampled: SampledSimulations[Any],
    ) -> tuple[SimulationTimeGrid, ...]:
        if self.execution.family == "tanaka":
            grid = _fixed_time_grid(
                self.execution.horizon,
                self.execution.numerical,
            )
            return tuple(grid for _ in sampled.samples)
        if self.execution.family == "benjamin_feir":
            if not all(
                isinstance(sample, BenjaminFeirSample) for sample in sampled.samples
            ):
                raise TypeError("Benjamin--Feir sampler returned an unexpected sample")
            return tuple(
                _benjamin_feir_time_grid(sample, self.execution)
                for sample in sampled.samples
            )
        if not all(isinstance(sample, JonswapTmaSample) for sample in sampled.samples):
            raise TypeError("JONSWAP/TMA sampler returned an unexpected sample")
        return tuple(
            _jonswap_time_grid(sample, self.execution) for sample in sampled.samples
        )

    def _construct(
        self,
        proposed: PreparedTrajectoryBatch[Any],
        *,
        selected_local_indices: tuple[int, ...] | None,
    ) -> TrajectoryInitialBatch:
        if self.constructor is not None:
            return self.constructor(
                proposed,
                selected_local_indices=selected_local_indices,
            )
        if self.execution.family == "tanaka":
            return construct_tanaka_trajectory_batch(
                proposed,
                selected_local_indices=selected_local_indices,
            )
        if self.execution.family == "benjamin_feir":
            if selected_local_indices is not None:
                raise ValueError(
                    "selected construction is not supported for Benjamin--Feir"
                )
            return construct_benjamin_feir_trajectory_batch(proposed)
        return construct_jonswap_tma_trajectory_batch(
            proposed,
            selected_local_indices=selected_local_indices,
        )

    def _construct_tanaka(
        self,
        proposed: PreparedTrajectoryBatch[Any],
    ) -> tuple[
        tuple[int, ...],
        TrajectoryInitialBatch | None,
        Mapping[int, SimulationOutcome],
    ]:
        remaining = list(range(len(proposed.sampled.assignments)))
        rejected: dict[int, SimulationOutcome] = {}
        initial: TrajectoryInitialBatch | None = None
        while remaining:
            try:
                initial = self._construct(
                    proposed,
                    selected_local_indices=tuple(remaining),
                )
                break
            except Exception as error:
                failure = self.construction_failure_classifier(error)
                if failure is None:
                    raise
                positions = failure.local_indices
                if (
                    not positions
                    or any(
                        isinstance(index, bool) or not isinstance(index, int)
                        for index in positions
                    )
                    or tuple(sorted(set(positions))) != positions
                    or positions[0] < 0
                    or positions[-1] >= len(remaining)
                ):
                    raise RuntimeError(
                        "declared construction failure has invalid local indices"
                    ) from error
                original_indices = tuple(remaining[index] for index in positions)
                remapped_failure = self._remap_tanaka_failure(
                    proposed,
                    remaining=tuple(remaining),
                    failure=failure,
                )
                for original_index in original_indices:
                    rejected[original_index] = _construction_rejection(
                        remapped_failure,
                        original_local_index=original_index,
                    )
                failed_positions = set(positions)
                remaining = [
                    original_index
                    for position, original_index in enumerate(remaining)
                    if position not in failed_positions
                ]

        if initial is not None and initial.eta0.shape[0] != len(remaining):
            raise RuntimeError(
                "Tanaka constructor returned the wrong selected batch size"
            )
        return tuple(remaining), initial, MappingProxyType(rejected)

    def _construct_jonswap(
        self,
        proposed: PreparedTrajectoryBatch[Any],
    ) -> tuple[
        tuple[int, ...],
        TrajectoryInitialBatch | None,
        Mapping[int, SimulationOutcome],
    ]:
        """Retain valid siblings when one random-sea realization is invalid."""

        remaining = list(range(len(proposed.sampled.assignments)))
        rejected: dict[int, SimulationOutcome] = {}
        initial: TrajectoryInitialBatch | None = None
        while remaining:
            try:
                initial = self._construct(
                    proposed,
                    selected_local_indices=tuple(remaining),
                )
                break
            except Exception as error:
                failure = self.construction_failure_classifier(error)
                if failure is None:
                    raise
                positions = failure.local_indices
                if (
                    not positions
                    or any(
                        isinstance(index, bool) or not isinstance(index, int)
                        for index in positions
                    )
                    or tuple(sorted(set(positions))) != positions
                    or positions[0] < 0
                    or positions[-1] >= len(remaining)
                ):
                    raise RuntimeError(
                        "declared construction failure has invalid local indices"
                    ) from error
                remapped_failure = self._remap_jonswap_failure(
                    remaining=tuple(remaining),
                    failure=failure,
                )
                original_indices = remapped_failure.local_indices
                for original_index in original_indices:
                    rejected[original_index] = _construction_rejection(
                        remapped_failure,
                        original_local_index=original_index,
                    )
                failed_positions = set(positions)
                remaining = [
                    original_index
                    for position, original_index in enumerate(remaining)
                    if position not in failed_positions
                ]

        if initial is not None and initial.eta0.shape[0] != len(remaining):
            raise RuntimeError(
                "JONSWAP constructor returned the wrong selected batch size"
            )
        return tuple(remaining), initial, MappingProxyType(rejected)

    @staticmethod
    def _remap_jonswap_failure(
        *,
        remaining: tuple[int, ...],
        failure: DeclaredConstructionFailure,
    ) -> DeclaredConstructionFailure:
        """Rewrite constructor-subset indices into immutable batch-plan indices."""

        record = dict(_strict_json_copy(failure.record))
        record["constructor_subbatch_invalid_simulation_indices"] = list(
            failure.local_indices
        )
        original_indices = tuple(remaining[index] for index in failure.local_indices)
        record["invalid_simulation_indices"] = list(original_indices)
        record["index_space"] = "original_batch_plan_local_index"
        simulations = record.get("simulations")
        if not isinstance(simulations, list):
            raise TypeError(
                "JONSWAP construction failure must list invalid simulations"
            )
        for simulation in simulations:
            if not isinstance(simulation, dict):
                raise TypeError(
                    "every invalid JONSWAP simulation must be a JSON object"
                )
            subbatch_index = simulation.get("local_simulation_index")
            if (
                isinstance(subbatch_index, bool)
                or not isinstance(subbatch_index, int)
                or subbatch_index < 0
                or subbatch_index >= len(remaining)
            ):
                raise RuntimeError(
                    "JONSWAP failure refers to an absent constructor-subset simulation"
                )
            simulation["constructor_subbatch_local_simulation_index"] = subbatch_index
            simulation["local_simulation_index"] = remaining[subbatch_index]
        return DeclaredConstructionFailure(
            local_indices=original_indices,
            record=_strict_json_copy(record),
        )

    @staticmethod
    def _remap_tanaka_failure(
        proposed: PreparedTrajectoryBatch[Any],
        *,
        remaining: tuple[int, ...],
        failure: DeclaredConstructionFailure,
    ) -> DeclaredConstructionFailure:
        """Rewrite constructor-subset indices into immutable batch-plan indices."""

        samples = proposed.sampled.samples
        if not all(isinstance(sample, TanakaSample) for sample in samples):
            raise TypeError("Tanaka failure remapping requires Tanaka samples")
        tanaka_samples = tuple(samples)
        full_offsets: list[int] = []
        running = 0
        for sample in tanaka_samples:
            full_offsets.append(running)
            running += len(sample.crests)

        subbatch_offsets: list[int] = []
        running = 0
        for original_index in remaining:
            subbatch_offsets.append(running)
            running += len(tanaka_samples[original_index].crests)

        record = dict(_strict_json_copy(failure.record))
        record["constructor_subbatch_invalid_simulation_indices"] = list(
            failure.local_indices
        )
        original_indices = tuple(remaining[index] for index in failure.local_indices)
        record["invalid_simulation_indices"] = list(original_indices)
        record["index_space"] = "original_batch_plan_local_index"

        components = record["components"]
        assert isinstance(components, list)
        for component in components:
            assert isinstance(component, dict)
            subbatch_simulation = component["local_simulation_index"]
            component_within_simulation = component["component_within_simulation"]
            subbatch_global = component["global_component_index"]
            assert isinstance(subbatch_simulation, int)
            assert isinstance(component_within_simulation, int)
            assert isinstance(subbatch_global, int)
            if subbatch_simulation >= len(remaining):
                raise RuntimeError(
                    "Tanaka component refers to an absent constructor-subset simulation"
                )
            original_simulation = remaining[subbatch_simulation]
            crest_count = len(tanaka_samples[original_simulation].crests)
            if component_within_simulation >= crest_count:
                raise RuntimeError(
                    "Tanaka component refers to an absent crest within its simulation"
                )
            expected_subbatch_global = (
                subbatch_offsets[subbatch_simulation] + component_within_simulation
            )
            if subbatch_global != expected_subbatch_global:
                raise RuntimeError(
                    "Tanaka component global index is inconsistent with its simulation"
                )
            component["constructor_subbatch_local_simulation_index"] = (
                subbatch_simulation
            )
            component["constructor_subbatch_global_component_index"] = subbatch_global
            component["local_simulation_index"] = original_simulation
            component["global_component_index"] = (
                full_offsets[original_simulation] + component_within_simulation
            )

        return DeclaredConstructionFailure(
            local_indices=original_indices,
            record=_strict_json_copy(record),
        )

    def _produce(
        self,
        initial: TrajectoryInitialBatch,
        *,
        family: TrajectoryFamily,
        grid: SimulationTimeGrid,
    ) -> tuple[SimulationOutcome, ...]:
        minimum_stored = {
            "tanaka": self.execution.frame_selection.tanaka_count,
            "benjamin_feir": self.execution.frame_selection.benjamin_feir_count,
            "jonswap_tma": self.execution.frame_selection.jonswap_tma_count,
        }[family]
        if grid.saved_times.size < minimum_stored:
            raise ValueError(
                "trajectory horizon has fewer saved times than the storage policy"
            )
        simulations = execute_trajectory_batch(
            initial.eta0,
            initial.xi0,
            initial.depths,
            (grid.saved_times,) * initial.eta0.shape[0],
            config=self.execution.numerical,
            integrator=self.rollout_executor,
        )
        return tuple(
            _finalize_outcome(
                outcome,
                grid,
                construction_metrics=(
                    initial.construction_metrics[index]
                    if initial.construction_metrics
                    else None
                ),
                support_evaluated=True,
            )
            for index, outcome in enumerate(
                subsample_trajectories(
                    simulations,
                    initial.depths,
                    family=family,
                    length=self.execution.numerical.length,
                    frame_selection=self.execution.frame_selection,
                )
            )
        )

    def _produce_variable_horizons(
        self,
        initial: TrajectoryInitialBatch,
        grids: tuple[SimulationTimeGrid, ...],
        *,
        family: Literal["benjamin_feir", "jonswap_tma"],
    ) -> tuple[SimulationOutcome, ...]:
        if len(grids) != initial.eta0.shape[0]:
            raise ValueError(f"{family} requires one time grid per initial condition")
        minimum_stored = {
            "benjamin_feir": self.execution.frame_selection.benjamin_feir_count,
            "jonswap_tma": self.execution.frame_selection.jonswap_tma_count,
        }[family]
        if any(grid.saved_times.size < minimum_stored for grid in grids):
            raise ValueError(
                "trajectory horizon has fewer saved times than the storage policy"
            )
        simulations = execute_trajectory_batch(
            initial.eta0,
            initial.xi0,
            initial.depths,
            tuple(grid.saved_times for grid in grids),
            config=self.execution.numerical,
            integrator=self.rollout_executor,
        )
        outcomes = subsample_trajectories(
            simulations,
            initial.depths,
            family=family,
            length=self.execution.numerical.length,
            frame_selection=self.execution.frame_selection,
        )
        return tuple(
            _finalize_outcome(
                outcome,
                grid,
                construction_metrics=(
                    initial.construction_metrics[index]
                    if initial.construction_metrics
                    else None
                ),
                support_evaluated=True,
            )
            for index, (outcome, grid) in enumerate(zip(outcomes, grids))
        )

    def _produce_jonswap(
        self,
        initial: TrajectoryInitialBatch,
        grids: tuple[SimulationTimeGrid, ...],
    ) -> tuple[SimulationOutcome, ...]:
        """Produce JONSWAP simulations through the adjustment hook."""

        return self._produce_variable_horizons(
            initial,
            grids,
            family="jonswap_tma",
        )

    def _ordered_outcomes(
        self,
        proposed: PreparedTrajectoryBatch[Any],
        grids: tuple[SimulationTimeGrid, ...],
    ) -> tuple[SimulationOutcome, ...]:
        simulation_count = len(proposed.sampled.assignments)
        ordered: list[SimulationOutcome | None] = [None] * simulation_count
        if self.execution.family == "tanaka":
            valid_indices, initial, rejected = self._construct_tanaka(proposed)
            for index, outcome in rejected.items():
                ordered[index] = _finalize_outcome(outcome, grids[index])
            if initial is not None:
                valid_outcomes = self._produce(
                    initial,
                    family="tanaka",
                    grid=grids[valid_indices[0]],
                )
                for index, outcome in zip(valid_indices, valid_outcomes):
                    ordered[index] = outcome
        elif self.execution.family == "jonswap_tma":
            valid_indices, initial, rejected = self._construct_jonswap(proposed)
            for index, outcome in rejected.items():
                ordered[index] = _finalize_outcome(outcome, grids[index])
            if initial is not None:
                valid_grids = tuple(grids[index] for index in valid_indices)
                valid_outcomes = self._produce_jonswap(initial, valid_grids)
                for index, outcome in zip(valid_indices, valid_outcomes):
                    ordered[index] = outcome
        else:
            initial = self._construct(
                proposed,
                selected_local_indices=None,
            )
            if initial.eta0.shape[0] != simulation_count:
                raise RuntimeError("constructor returned the wrong batch size")
            outcomes = self._produce_variable_horizons(
                initial,
                family="benjamin_feir",
                grids=grids,
            )
            ordered[:] = outcomes

        if any(outcome is None for outcome in ordered):
            raise RuntimeError(
                "every proposed trajectory simulation must have an outcome"
            )
        return tuple(outcome for outcome in ordered if outcome is not None)

    def __call__(
        self,
        assignments: tuple[AttemptAssignment, ...],
        *,
        batch_id: int,
    ) -> Path:
        """Run and save one complete trajectory batch."""

        if not assignments:
            raise ValueError("trajectory attempt batches must not be empty")
        sampled = self._sample(assignments)
        grids = self._time_grids(sampled)
        execution_record = self.execution.to_json_record()
        grid_records = [grid.to_json_record() for grid in grids]
        additional_metadata = dict(self.metadata or {})
        proposed = prepare_trajectory_batch(
            sampled,
            root=self.chunk_config.root,
            family_name=self.chunk_config.family_name,
            batch_id=batch_id,
            metadata={
                "family": self.execution.family,
                "simulation_type": "trajectory",
                "trajectory_execution": execution_record,
                "simulation_time_grids": grid_records,
                "additional_metadata": additional_metadata,
            },
        )

        outcomes = self._ordered_outcomes(proposed, grids)
        commit_simulation_outcomes(
            proposed.path,
            proposed.batch_plan,
            outcomes,
            metadata={
                "family": self.execution.family,
                "simulation_type": "trajectory",
                "trajectory_execution": execution_record,
                "simulation_time_grids": grid_records,
                "attempted_simulations": len(outcomes),
                "accepted_simulations": sum(
                    outcome.decision.accepted for outcome in outcomes
                ),
                "additional_metadata": additional_metadata,
            },
        )
        return proposed.path
