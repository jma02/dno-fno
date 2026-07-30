"""Thin accepted-quota executor for the three rollout-data families."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import math
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np

from solver.gen_data.benjamin_feir_population import (
    BENJAMIN_FEIR_POPULATION_CELLS,
)
from solver.gen_data.jonswap_tma import finite_depth_angular_frequency
from solver.gen_data.jonswap_tma_population import (
    JONSWAP_TMA_POPULATION_CELLS,
    JonswapTmaPopulationSample,
)
from solver.gen_data.pipeline.archive import BatchPaths
from solver.gen_data.pipeline.archive import record_fatal_failure
from solver.gen_data.pipeline.production import (
    AttemptAssignment,
    PhysicalFamilyId,
)
from solver.gen_data.pipeline.quality import (
    QualityDecision,
    QualityReason,
    QualityScope,
)
from solver.gen_data.pipeline.quota_driver import AcceptedQuotaRunSpec
from solver.gen_data.pipeline.refinement import (
    PAPER_GL2_CONTRACT,
    ArmExecutor,
    ResidualControlledGL2Contract,
    execute_production_trajectory,
    execute_variable_horizon_production_trajectory,
    run_residual_controlled_arm,
)
from solver.gen_data.pipeline.trajectory_writer import (
    PAPER_STORED_TIME_POLICY,
    StoredTimePolicy,
    TrajectoryFamily,
    outcomes_from_production,
)
from solver.gen_data.pipeline.writer import (
    CaseOutcome,
    commit_case_outcomes,
)
from solver.gen_data.tanaka_population import TANAKA_POPULATION_CELLS
from solver.gen_data.tanaka_population import TanakaPopulationSample
from solver.gen_data.trajectory_family_adapters import (
    PersistedTrajectoryProposal,
    SampledTrajectoryCases,
    TrajectoryInitialBatch,
    construct_benjamin_feir_trajectory_batch,
    construct_jonswap_tma_trajectory_batch,
    construct_tanaka_trajectory_batch,
    persist_sampled_trajectory_proposal,
    sample_benjamin_feir_trajectory_cases,
    sample_jonswap_tma_trajectory_cases,
    sample_tanaka_trajectory_cases,
)


ContractRole: TypeAlias = Literal[
    "paper_corpus",
    "reduced_wiring_evidence_only",
]
HorizonKind: TypeAlias = Literal[
    "fixed_terminal_time",
    "jonswap_16_peak_periods_floor_saved_grid",
]
JsonScalar: TypeAlias = str | int | float | bool | None
TRAJECTORY_REQUIRED_CHECKS = (
    QualityReason.OUTSIDE_SUPPORT | QualityReason.INCOMPLETE_TRAJECTORY
)
_TANAKA_FAILURE_REASONS = frozenset(
    {
        "negative_or_nonfinite_surface_potential_radicand",
        "nonpositive_or_nonfinite_surface_potential_speed_squared",
    }
)
_TANAKA_COMPONENT_FIELDS = frozenset(
    {
        "local_case_index",
        "component_within_case",
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
_FAMILY_CELLS = {
    "tanaka": frozenset(cell.cell_id for cell in TANAKA_POPULATION_CELLS),
    "benjamin_feir": frozenset(cell.cell_id for cell in BENJAMIN_FEIR_POPULATION_CELLS),
    "jonswap_tma": frozenset(cell.cell_id for cell in JONSWAP_TMA_POPULATION_CELLS),
}


def _strict_json_copy(value: Mapping[str, object]) -> Mapping[str, object]:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    parsed = json.loads(encoded)
    if not isinstance(parsed, dict):
        raise TypeError("value must encode a JSON object")
    return MappingProxyType(parsed)


@dataclass(frozen=True)
class TrajectoryHorizonPolicy:
    """Family-level terminal-time construction on the saved-time grid."""

    kind: HorizonKind
    fixed_terminal_time: float | None
    peak_period_count: int | None

    def __post_init__(self) -> None:
        if self.kind == "fixed_terminal_time":
            if (
                self.fixed_terminal_time is None
                or not math.isfinite(self.fixed_terminal_time)
                or self.fixed_terminal_time <= 0.0
            ):
                raise ValueError("fixed terminal time must be finite and positive")
            if self.peak_period_count is not None:
                raise ValueError("a fixed horizon cannot set peak_period_count")
        elif self.kind == "jonswap_16_peak_periods_floor_saved_grid":
            if self.fixed_terminal_time is not None:
                raise ValueError("a peak-period horizon cannot set a fixed time")
            if (
                isinstance(self.peak_period_count, bool)
                or not isinstance(self.peak_period_count, int)
                or self.peak_period_count <= 0
            ):
                raise ValueError("peak_period_count must be a positive integer")
        else:
            raise ValueError(f"unknown trajectory horizon kind: {self.kind}")

    @classmethod
    def fixed(cls, terminal_time: float) -> TrajectoryHorizonPolicy:
        return cls(
            kind="fixed_terminal_time",
            fixed_terminal_time=terminal_time,
            peak_period_count=None,
        )

    @classmethod
    def jonswap_peak_periods(
        cls,
        count: int = 16,
    ) -> TrajectoryHorizonPolicy:
        return cls(
            kind="jonswap_16_peak_periods_floor_saved_grid",
            fixed_terminal_time=None,
            peak_period_count=count,
        )

    def to_json_record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "fixed_terminal_time": self.fixed_terminal_time,
            "peak_period_count": self.peak_period_count,
            "saved_grid_rounding": (
                "exact" if self.kind == "fixed_terminal_time" else "floor"
            ),
        }


@dataclass(frozen=True)
class TrajectoryExecutionConfig:
    """Full numerical and retained-time contract for one rollout family."""

    family: TrajectoryFamily
    role: ContractRole
    numerical: ResidualControlledGL2Contract
    horizon: TrajectoryHorizonPolicy
    stored_time_policy: StoredTimePolicy
    jonswap_quadrature_order: int | None

    def __post_init__(self) -> None:
        if self.family not in _FAMILY_IDS:
            raise ValueError(f"unknown trajectory family: {self.family}")
        if self.role not in (
            "paper_corpus",
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
            if self.horizon.kind != "jonswap_16_peak_periods_floor_saved_grid":
                raise ValueError("JONSWAP/TMA requires a per-case peak-period horizon")
        else:
            if self.jonswap_quadrature_order is not None:
                raise ValueError("only JONSWAP/TMA may set a quadrature order")
            if self.horizon.kind != "fixed_terminal_time":
                raise ValueError("Tanaka and Benjamin--Feir require a fixed horizon")

        if self.role == "paper_corpus":
            expected_horizon = (
                TrajectoryHorizonPolicy.jonswap_peak_periods(16)
                if self.family == "jonswap_tma"
                else TrajectoryHorizonPolicy.fixed(200.0)
            )
            expected_quadrature = 16 if self.family == "jonswap_tma" else None
            if (
                self.numerical != PAPER_GL2_CONTRACT
                or self.horizon != expected_horizon
                or self.stored_time_policy != PAPER_STORED_TIME_POLICY
                or self.jonswap_quadrature_order != expected_quadrature
            ):
                raise ValueError(
                    "a reduced trajectory execution must be labeled "
                    "'reduced_wiring_evidence_only'"
                )

    @classmethod
    def paper(cls, family: TrajectoryFamily) -> TrajectoryExecutionConfig:
        """Construct the exact paper-corpus contract for one family."""

        return cls(
            family=family,
            role="paper_corpus",
            numerical=PAPER_GL2_CONTRACT,
            horizon=(
                TrajectoryHorizonPolicy.jonswap_peak_periods(16)
                if family == "jonswap_tma"
                else TrajectoryHorizonPolicy.fixed(200.0)
            ),
            stored_time_policy=PAPER_STORED_TIME_POLICY,
            jonswap_quadrature_order=16 if family == "jonswap_tma" else None,
        )

    def to_json_record(self) -> dict[str, object]:
        numerical = asdict(self.numerical)
        numerical["dt"] = numerical.pop("production_dt")
        for audit_field in (
            "refinement_tolerance",
            "relative_floor",
        ):
            numerical.pop(audit_field)
        return {
            "family": self.family,
            "role": self.role,
            "numerical": numerical,
            "horizon": self.horizon.to_json_record(),
            "stored_time_policy": asdict(self.stored_time_policy),
            "jonswap_quadrature_order": self.jonswap_quadrature_order,
        }


@dataclass(frozen=True)
class CaseTimeGrid:
    """Intended and realized terminal times for one proposed case."""

    intended_terminal_time: float
    realized_terminal_time: float
    saved_times: np.ndarray

    def to_json_record(self) -> dict[str, object]:
        return {
            "intended_terminal_time": self.intended_terminal_time,
            "realized_terminal_time": self.realized_terminal_time,
            "saved_time_count": int(self.saved_times.size),
        }


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


@dataclass(frozen=True)
class DeclaredFatalFailure:
    """Explicit terminal failure information for a durable proposal."""

    phase: str
    telemetry: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.phase:
            raise ValueError("fatal-failure phase must not be empty")
        object.__setattr__(self, "telemetry", _strict_json_copy(self.telemetry))


class FatalFailureClassifier(Protocol):
    def __call__(
        self,
        error: Exception,
    ) -> DeclaredFatalFailure | None: ...


class DeclaredTrajectoryFatalError(RuntimeError):
    """An explicitly classified deterministic failure of a proposed batch."""

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        telemetry: Mapping[str, object] | None = None,
    ) -> None:
        self.failure = DeclaredFatalFailure(
            phase=phase,
            telemetry=telemetry or {},
        )
        super().__init__(message)


class TrajectoryConstructor(Protocol):
    def __call__(
        self,
        proposed: PersistedTrajectoryProposal[Any],
        *,
        selected_local_indices: tuple[int, ...] | None,
    ) -> TrajectoryInitialBatch: ...


def classify_declared_fatal_failure(
    error: Exception,
) -> DeclaredFatalFailure | None:
    """Recognize only the executor's explicit deterministic-failure type."""

    if not isinstance(error, DeclaredTrajectoryFatalError):
        return None
    return error.failure


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
        from solver.gen_data.generate_tanaka_dataset_v2 import (
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
    indices = record.get("invalid_case_indices")
    if not isinstance(indices, list) or not indices:
        raise TypeError("Tanaka construction failure must list invalid_case_indices")
    if (
        any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
        or tuple(sorted(set(indices))) != tuple(indices)
        or indices[0] < 0
    ):
        raise ValueError(
            "Tanaka invalid_case_indices must be unique, increasing, "
            "nonnegative integers"
        )
    components = record.get("components")
    if not isinstance(components, list) or not components:
        raise TypeError("Tanaka construction failure must list invalid components")
    component_cases: set[int] = set()
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
        local_case_index = component.get("local_case_index")
        component_within_case = component.get("component_within_case")
        global_component_index = component.get("global_component_index")
        local_case_index = _nonnegative_json_integer(
            local_case_index,
            field_name="local_case_index",
        )
        component_within_case = _nonnegative_json_integer(
            component_within_case,
            field_name="component_within_case",
        )
        global_component_index = _nonnegative_json_integer(
            global_component_index,
            field_name="global_component_index",
        )
        assert local_case_index is not None
        assert component_within_case is not None
        assert global_component_index is not None
        identity = (
            local_case_index,
            component_within_case,
            global_component_index,
        )
        if identity in component_identities:
            raise ValueError("Tanaka construction failure repeats a component")
        component_identities.add(identity)
        global_component_indices.append(global_component_index)
        component_cases.add(local_case_index)

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
        if (
            minimum is not None
            and (minimum < 0.0) != (negative_count > 0)
        ):
            raise ValueError(
                "Tanaka negative-radicand count disagrees with its minimum"
            )
        invalid_speed = speed_squared is None or speed_squared <= 0.0
        invalid_radicand = negative_count > 0 or nonfinite_count > 0
        if not invalid_speed and not invalid_radicand:
            raise ValueError("Tanaka component does not describe a domain failure")
        component_failures.append((invalid_speed, invalid_radicand))
        has_unclassified_component |= invalid_speed or nonfinite_count > 0

    if component_cases != set(indices):
        raise ValueError("Tanaka invalid components do not match invalid_case_indices")
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


def _fixed_time_grid(
    horizon: TrajectoryHorizonPolicy,
    contract: ResidualControlledGL2Contract,
) -> CaseTimeGrid:
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
    return CaseTimeGrid(
        intended_terminal_time=terminal,
        realized_terminal_time=float(saved_times[-1]),
        saved_times=saved_times,
    )


def _jonswap_time_grid(
    sample: JonswapTmaPopulationSample,
    execution: TrajectoryExecutionConfig,
) -> CaseTimeGrid:
    count = execution.horizon.peak_period_count
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
    saved_dt = execution.numerical.saved_dt
    step_count = math.floor(intended / saved_dt)
    while step_count * saved_dt > intended:
        step_count -= 1
    while (step_count + 1) * saved_dt <= intended:
        step_count += 1
    if step_count < 1:
        raise ValueError("JONSWAP/TMA horizon is shorter than one saved step")
    saved_times = saved_dt * np.arange(step_count + 1, dtype=np.float64)
    realized = float(saved_times[-1])
    if not (realized <= intended and intended - realized < saved_dt):
        raise RuntimeError("JONSWAP/TMA saved-grid horizon was not strictly floored")
    return CaseTimeGrid(
        intended_terminal_time=intended,
        realized_terminal_time=realized,
        saved_times=saved_times,
    )


def _construction_rejection(
    failure: DeclaredConstructionFailure,
    *,
    original_local_index: int,
) -> CaseOutcome:
    reason = QualityReason.OUTSIDE_SUPPORT
    failure_reason = failure.record.get("reason")
    metrics: dict[str, JsonScalar] = {
        "construction_status": "declared_outside_support",
        "construction_failure_reason": (
            failure_reason if isinstance(failure_reason, str) else ""
        ),
        "original_local_index": original_local_index,
        "construction_failure_json": json.dumps(
            dict(failure.record),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
    }
    return CaseOutcome(
        decision=QualityDecision(
            scope=QualityScope.TRAJECTORY,
            required=TRAJECTORY_REQUIRED_CHECKS,
            evaluated=reason,
            failed=reason,
        ),
        rows=None,
        metrics=metrics,
    )


def _with_support_evaluated(outcome: CaseOutcome) -> CaseOutcome:
    decision = outcome.decision
    return CaseOutcome(
        decision=QualityDecision(
            scope=QualityScope.TRAJECTORY,
            required=TRAJECTORY_REQUIRED_CHECKS,
            evaluated=decision.evaluated | QualityReason.OUTSIDE_SUPPORT,
            failed=decision.failed,
        ),
        rows=outcome.rows,
        metrics=outcome.metrics,
    )


def _with_time_grid(
    outcome: CaseOutcome,
    grid: CaseTimeGrid,
) -> CaseOutcome:
    return CaseOutcome(
        decision=outcome.decision,
        rows=outcome.rows,
        metrics={
            **outcome.metrics,
            "intended_terminal_time": grid.intended_terminal_time,
            "realized_terminal_time": grid.realized_terminal_time,
            "saved_time_count": int(grid.saved_times.size),
        },
    )


@dataclass(frozen=True)
class TrajectoryQuotaExecutor:
    """Execute one durable attempted batch for a rollout-data family."""

    run_spec: AcceptedQuotaRunSpec
    execution: TrajectoryExecutionConfig
    metadata: Mapping[str, object] | None = None
    arm_executor: ArmExecutor = run_residual_controlled_arm
    constructor: TrajectoryConstructor | None = None
    construction_failure_classifier: ConstructionFailureClassifier = (
        classify_tanaka_construction_failure
    )
    fatal_failure_classifier: FatalFailureClassifier = classify_declared_fatal_failure

    def __post_init__(self) -> None:
        expected_family_id = _FAMILY_IDS[self.execution.family]
        if self.run_spec.family_name != self.execution.family:
            raise ValueError("run family name differs from trajectory execution")
        if self.run_spec.family_id is not expected_family_id:
            raise ValueError("run family ID differs from trajectory execution")
        unknown_cells = set(self.run_spec.cell_codes).difference(
            _FAMILY_CELLS[self.execution.family]
        )
        if unknown_cells:
            raise ValueError(
                f"unknown {self.execution.family} cells: {sorted(unknown_cells)}"
            )

        run_configuration = self.run_spec.to_json_record()["configuration"]
        assert isinstance(run_configuration, dict)
        configured_execution = run_configuration.get("trajectory_execution")
        if configured_execution != self.execution.to_json_record():
            raise ValueError(
                "run configuration differs from the supplied trajectory execution"
            )
        if self.execution.role == "paper_corpus" and (
            self.arm_executor is not run_residual_controlled_arm
            or self.constructor is not None
            or (
                self.construction_failure_classifier
                is not classify_tanaka_construction_failure
            )
            or self.fatal_failure_classifier is not classify_declared_fatal_failure
        ):
            raise ValueError(
                "paper-corpus execution requires the production constructor, "
                "arm executor, and failure classifiers"
            )
        object.__setattr__(
            self,
            "metadata",
            _strict_json_copy(self.metadata or {}),
        )

    def _sample(
        self,
        assignments: tuple[AttemptAssignment, ...],
    ) -> SampledTrajectoryCases[Any]:
        contract = self.execution.numerical
        if self.execution.family == "tanaka":
            sampled = sample_tanaka_trajectory_cases(
                assignments,
                contract=contract,
            )
        elif self.execution.family == "benjamin_feir":
            sampled = sample_benjamin_feir_trajectory_cases(
                assignments,
                contract=contract,
            )
        else:
            quadrature_order = self.execution.jonswap_quadrature_order
            assert quadrature_order is not None
            sampled = sample_jonswap_tma_trajectory_cases(
                assignments,
                contract=contract,
                quadrature_order=quadrature_order,
            )
        if sampled.assignments != assignments or sampled.contract != contract:
            raise RuntimeError("trajectory sampler changed assignments or contract")
        return sampled

    def _time_grids(
        self,
        sampled: SampledTrajectoryCases[Any],
    ) -> tuple[CaseTimeGrid, ...]:
        if self.execution.family != "jonswap_tma":
            grid = _fixed_time_grid(
                self.execution.horizon,
                self.execution.numerical,
            )
            return tuple(grid for _ in sampled.samples)
        if not all(
            isinstance(sample, JonswapTmaPopulationSample) for sample in sampled.samples
        ):
            raise TypeError("JONSWAP/TMA sampler returned an unexpected sample")
        return tuple(
            _jonswap_time_grid(sample, self.execution) for sample in sampled.samples
        )

    def _construct(
        self,
        proposed: PersistedTrajectoryProposal[Any],
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
        if selected_local_indices is not None:
            raise ValueError("selected construction is supported only for Tanaka")
        if self.execution.family == "benjamin_feir":
            return construct_benjamin_feir_trajectory_batch(proposed)
        return construct_jonswap_tma_trajectory_batch(proposed)

    def _construct_tanaka(
        self,
        proposed: PersistedTrajectoryProposal[Any],
    ) -> tuple[
        tuple[int, ...],
        TrajectoryInitialBatch | None,
        Mapping[int, CaseOutcome],
    ]:
        remaining = list(range(len(proposed.sampled.assignments)))
        rejected: dict[int, CaseOutcome] = {}
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

    @staticmethod
    def _remap_tanaka_failure(
        proposed: PersistedTrajectoryProposal[Any],
        *,
        remaining: tuple[int, ...],
        failure: DeclaredConstructionFailure,
    ) -> DeclaredConstructionFailure:
        """Rewrite constructor-subset indices into immutable proposal indices."""

        samples = proposed.sampled.samples
        if not all(isinstance(sample, TanakaPopulationSample) for sample in samples):
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

        encoded = json.dumps(
            dict(failure.record),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        record = json.loads(encoded)
        record["constructor_subbatch_invalid_case_indices"] = list(
            failure.local_indices
        )
        original_indices = tuple(remaining[index] for index in failure.local_indices)
        record["invalid_case_indices"] = list(original_indices)
        record["index_space"] = "original_proposal_local_index"

        components = record["components"]
        assert isinstance(components, list)
        for component in components:
            assert isinstance(component, dict)
            subbatch_case = component["local_case_index"]
            component_within_case = component["component_within_case"]
            subbatch_global = component["global_component_index"]
            assert isinstance(subbatch_case, int)
            assert isinstance(component_within_case, int)
            assert isinstance(subbatch_global, int)
            if subbatch_case >= len(remaining):
                raise RuntimeError(
                    "Tanaka component refers to an absent constructor-subset case"
                )
            original_case = remaining[subbatch_case]
            crest_count = len(tanaka_samples[original_case].crests)
            if component_within_case >= crest_count:
                raise RuntimeError(
                    "Tanaka component refers to an absent crest within its case"
                )
            expected_subbatch_global = (
                subbatch_offsets[subbatch_case] + component_within_case
            )
            if subbatch_global != expected_subbatch_global:
                raise RuntimeError(
                    "Tanaka component global index is inconsistent with its case"
                )
            component["constructor_subbatch_local_case_index"] = subbatch_case
            component["constructor_subbatch_global_component_index"] = subbatch_global
            component["local_case_index"] = original_case
            component["global_component_index"] = (
                full_offsets[original_case] + component_within_case
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
        grid: CaseTimeGrid,
    ) -> tuple[CaseOutcome, ...]:
        minimum_stored = {
            "tanaka": self.execution.stored_time_policy.tanaka_count,
            "benjamin_feir": (self.execution.stored_time_policy.benjamin_feir_count),
            "jonswap_tma": self.execution.stored_time_policy.random_sea_count,
        }[family]
        if grid.saved_times.size < minimum_stored:
            raise ValueError(
                "trajectory horizon has fewer saved times than the storage policy"
            )
        execution = execute_production_trajectory(
            initial.eta0,
            initial.xi0,
            initial.depths,
            grid.saved_times,
            contract=self.execution.numerical,
            arm_executor=self.arm_executor,
        )
        return tuple(
            _with_time_grid(_with_support_evaluated(outcome), grid)
            for outcome in outcomes_from_production(
                execution,
                initial.depths,
                family=family,
                length=self.execution.numerical.length,
                policy=self.execution.stored_time_policy,
            )
        )

    def _produce_jonswap(
        self,
        initial: TrajectoryInitialBatch,
        grids: tuple[CaseTimeGrid, ...],
    ) -> tuple[CaseOutcome, ...]:
        if len(grids) != initial.eta0.shape[0]:
            raise ValueError(
                "JONSWAP/TMA requires one time grid per initial condition"
            )
        minimum_stored = self.execution.stored_time_policy.random_sea_count
        if any(grid.saved_times.size < minimum_stored for grid in grids):
            raise ValueError(
                "trajectory horizon has fewer saved times than the storage policy"
            )
        execution = execute_variable_horizon_production_trajectory(
            initial.eta0,
            initial.xi0,
            initial.depths,
            tuple(grid.saved_times for grid in grids),
            contract=self.execution.numerical,
            arm_executor=self.arm_executor,
        )
        outcomes = outcomes_from_production(
            execution,
            initial.depths,
            family="jonswap_tma",
            length=self.execution.numerical.length,
            policy=self.execution.stored_time_policy,
        )
        return tuple(
            _with_time_grid(_with_support_evaluated(outcome), grid)
            for outcome, grid in zip(outcomes, grids)
        )

    def _ordered_outcomes(
        self,
        proposed: PersistedTrajectoryProposal[Any],
        grids: tuple[CaseTimeGrid, ...],
    ) -> tuple[CaseOutcome, ...]:
        case_count = len(proposed.sampled.assignments)
        ordered: list[CaseOutcome | None] = [None] * case_count
        if self.execution.family == "tanaka":
            valid_indices, initial, rejected = self._construct_tanaka(proposed)
            for index, outcome in rejected.items():
                ordered[index] = _with_time_grid(outcome, grids[index])
            if initial is not None:
                valid_outcomes = self._produce(
                    initial,
                    family="tanaka",
                    grid=grids[valid_indices[0]],
                )
                for index, outcome in zip(valid_indices, valid_outcomes):
                    ordered[index] = outcome
        else:
            initial = self._construct(
                proposed,
                selected_local_indices=None,
            )
            if initial.eta0.shape[0] != case_count:
                raise RuntimeError("constructor returned the wrong batch size")
            if self.execution.family == "benjamin_feir":
                outcomes = self._produce(
                    initial,
                    family="benjamin_feir",
                    grid=grids[0],
                )
                ordered[:] = outcomes
            else:
                ordered[:] = self._produce_jonswap(initial, grids)

        if any(outcome is None for outcome in ordered):
            raise RuntimeError("every proposed trajectory case must have an outcome")
        return tuple(outcome for outcome in ordered if outcome is not None)

    def __call__(
        self,
        assignments: tuple[AttemptAssignment, ...],
        *,
        batch_id: int,
    ) -> BatchPaths:
        """Run the complete sample-to-commit transaction for one batch."""

        if not assignments:
            raise ValueError("trajectory attempt batches must not be empty")
        sampled = self._sample(assignments)
        grids = self._time_grids(sampled)
        execution_record = self.execution.to_json_record()
        grid_records = [grid.to_json_record() for grid in grids]
        additional_metadata = dict(self.metadata or {})
        proposed = persist_sampled_trajectory_proposal(
            sampled,
            root=self.run_spec.root,
            family_name=self.run_spec.family_name,
            batch_id=batch_id,
            cell_codes=self.run_spec.cell_codes,
            config_fingerprint=self.run_spec.config_fingerprint,
            metadata={
                "family": self.execution.family,
                "case_kind": "trajectory",
                "trajectory_execution": execution_record,
                "case_time_grids": grid_records,
                "additional_metadata": additional_metadata,
            },
        )

        # Construction, rollout, and time selection all follow this write.
        try:
            outcomes = self._ordered_outcomes(proposed, grids)
            commit_case_outcomes(
                proposed.paths,
                proposed.proposal_arrays,
                outcomes,
                metadata={
                    "family": self.execution.family,
                    "case_kind": "trajectory",
                    "trajectory_execution": execution_record,
                    "case_time_grids": grid_records,
                    "attempted_cases": len(outcomes),
                    "accepted_cases": sum(
                        outcome.decision.accepted for outcome in outcomes
                    ),
                    "additional_metadata": additional_metadata,
                },
            )
        except Exception as error:
            fatal = self.fatal_failure_classifier(error)
            if fatal is None:
                raise
            record_fatal_failure(
                proposed.paths,
                phase=fatal.phase,
                exception_type=type(error).__name__,
                message=str(error),
                telemetry=fatal.telemetry,
            )
        return proposed.paths
