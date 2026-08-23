"""Horizon-bucketed JONSWAP/TMA quota execution.

The durable quota driver may propose more cases than fit in one numerical
rollout.  This executor sorts those proposed cases by their declared saved-time
count, executes memory-safe groups of nearby horizons, and restores proposal
order before the existing atomic batch commit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Final, TypeVar, cast

import numpy as np

from solver.gen_data.jonswap_tma import finite_depth_angular_frequency
from solver.gen_data.pipeline.quality import (
    QualityDecision,
    QualityReason,
    QualityScope,
)
from solver.gen_data.pipeline.refinement import (
    NonlinearAdjustmentArmExecutor,
    NonlinearAdjustmentCaseResult,
    execute_variable_horizon_nonlinear_adjustment,
    run_nonlinear_adjustment_arm,
)
from solver.gen_data.pipeline.writer import CaseOutcome
from solver.gen_data.trajectory_family_adapters import TrajectoryInitialBatch
from solver.gen_data.trajectory_quota_executor import (
    CaseTimeGrid,
    JONSWAP_ADJUSTMENT_FORMULA,
    JONSWAP_ADJUSTMENT_SCHEMA,
    JonswapNonlinearAdjustmentPolicy,
    PAPER_JONSWAP_ADJUSTMENT_POLICY,
    TrajectoryQuotaExecutor,
)


POLICY_KEY: Final = "jonswap_horizon_bucketing"
POLICY_SCHEMA: Final = "jonswap_horizon_bucketing_v2"
SORT_RULE: Final = "stable_saved_time_count_then_proposal_index"
ADJUSTMENT_SCHEMA: Final = JONSWAP_ADJUSTMENT_SCHEMA
ADJUSTMENT_FORMULA: Final = JONSWAP_ADJUSTMENT_FORMULA
ADJUSTMENT_RAMP_ORDER: Final = PAPER_JONSWAP_ADJUSTMENT_POLICY.ramp_order
ADJUSTMENT_RAMP_PEAK_PERIODS: Final = (
    PAPER_JONSWAP_ADJUSTMENT_POLICY.ramp_time_peak_periods
)
ADJUSTMENT_BURN_PEAK_PERIODS: Final = (
    PAPER_JONSWAP_ADJUSTMENT_POLICY.burn_peak_periods
)

ItemT = TypeVar("ItemT")


def horizon_sorted_groups(
    grids: Sequence[CaseTimeGrid],
    *,
    solver_batch_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Partition proposal indices into stable horizon-near solver groups."""

    if (
        isinstance(solver_batch_size, bool)
        or not isinstance(solver_batch_size, int)
        or solver_batch_size <= 0
    ):
        raise ValueError("solver_batch_size must be a positive integer")
    if not grids:
        raise ValueError("grids must not be empty")

    ordered = tuple(
        sorted(
            range(len(grids)),
            key=lambda index: (int(grids[index].saved_times.size), index),
        )
    )
    return tuple(
        ordered[start : start + solver_batch_size]
        for start in range(0, len(ordered), solver_batch_size)
    )


def restore_proposal_order(
    grouped_indices: Sequence[Sequence[int]],
    grouped_values: Sequence[Sequence[ItemT]],
    *,
    case_count: int,
) -> tuple[ItemT, ...]:
    """Restore values from solver-group order to durable proposal order."""

    if (
        isinstance(case_count, bool)
        or not isinstance(case_count, int)
        or case_count <= 0
    ):
        raise ValueError("case_count must be a positive integer")
    if len(grouped_indices) != len(grouped_values):
        raise ValueError("indices and values must contain the same groups")

    missing = object()
    restored: list[ItemT | object] = [missing] * case_count
    for indices, values in zip(grouped_indices, grouped_values):
        if len(indices) != len(values):
            raise ValueError("each index group must match its value group")
        for index, value in zip(indices, values):
            if not 0 <= index < case_count:
                raise ValueError("group index lies outside proposal order")
            if restored[index] is not missing:
                raise ValueError("group indices must not repeat")
            restored[index] = value
    if any(value is missing for value in restored):
        raise ValueError("group indices must cover every proposed case")
    return cast(tuple[ItemT, ...], tuple(restored))


def policy_record(
    *,
    outer_proposal_size: int,
    solver_batch_size: int,
    adjustment_policy: JonswapNonlinearAdjustmentPolicy = (
        PAPER_JONSWAP_ADJUSTMENT_POLICY
    ),
) -> dict[str, object]:
    """Return the execution policy bound into the run fingerprint."""

    for value, name in (
        (outer_proposal_size, "outer_proposal_size"),
        (solver_batch_size, "solver_batch_size"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if solver_batch_size > outer_proposal_size:
        raise ValueError(
            "solver_batch_size cannot exceed outer_proposal_size"
        )
    return {
        "schema": POLICY_SCHEMA,
        "outer_proposal_size": outer_proposal_size,
        "solver_batch_size": solver_batch_size,
        "sort_rule": SORT_RULE,
        "commit_order": "durable_proposal_order",
        "transaction_rule": "all_solver_groups_then_one_atomic_commit",
        "nonlinear_adjustment": adjustment_policy.to_json_record(),
    }


def _select_initial_batch(
    initial: TrajectoryInitialBatch,
    indices: Sequence[int],
) -> TrajectoryInitialBatch:
    selected = np.asarray(indices, dtype=np.int64)
    return TrajectoryInitialBatch(
        eta0=np.take(initial.eta0, selected, axis=0),
        xi0=np.take(initial.xi0, selected, axis=0),
        depths=np.take(initial.depths, selected, axis=0),
        specification_records=tuple(
            initial.specification_records[index] for index in indices
        ),
        construction_metrics=(
            tuple(initial.construction_metrics[index] for index in indices)
            if initial.construction_metrics
            else ()
        ),
    )


def _positive_record_float(
    record: Mapping[str, object],
    key: str,
) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"JONSWAP specification {key} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"JONSWAP specification {key} must be finite and positive")
    return result


def _peak_periods(
    initial: TrajectoryInitialBatch,
    *,
    gravity: float,
) -> np.ndarray:
    """Compute each finite-depth peak period from its immutable proposal."""

    depths = np.asarray(initial.depths, dtype=np.float64)
    peak_wavenumbers: list[float] = []
    for depth, record in zip(depths, initial.specification_records):
        if record.get("family_id") != 3 or record.get("revision_id") != 4:
            raise ValueError("unrecognized JONSWAP sample identity")
        recorded_depth = _positive_record_float(record, "depth")
        if not math.isclose(
            recorded_depth,
            float(depth),
            rel_tol=1.0e-13,
            abs_tol=1.0e-15,
        ):
            raise ValueError("constructed depth differs from JONSWAP specification")
        peak_wavenumbers.append(_positive_record_float(record, "peak_wavenumber"))
    angular_frequencies = np.asarray(
        [
            finite_depth_angular_frequency(
                np.asarray([peak_wavenumber], dtype=np.float64),
                depth=float(depth),
                gravity=gravity,
            )[0]
            for peak_wavenumber, depth in zip(peak_wavenumbers, depths)
        ],
        dtype=np.float64,
    )
    periods = 2.0 * math.pi / np.asarray(angular_frequencies, dtype=np.float64)
    if periods.shape != depths.shape or not np.isfinite(periods).all():
        raise RuntimeError("failed to compute finite JONSWAP peak periods")
    return periods


def _floored_time_grid(
    terminal_time: float,
    *,
    saved_dt: float,
) -> np.ndarray:
    """Return the saved-time prefix ending immediately before a horizon."""

    step_count = math.floor(terminal_time / saved_dt)
    while step_count * saved_dt > terminal_time:
        step_count -= 1
    while (step_count + 1) * saved_dt <= terminal_time:
        step_count += 1
    if step_count < 1:
        raise ValueError("nonlinear-adjustment horizon is shorter than one saved step")
    saved_times = saved_dt * np.arange(step_count + 1, dtype=np.float64)
    realized = float(saved_times[-1])
    if not (realized <= terminal_time and terminal_time - realized < saved_dt):
        raise RuntimeError("nonlinear-adjustment horizon was not strictly floored")
    return saved_times


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _adjustment_metrics(
    case: NonlinearAdjustmentCaseResult,
    *,
    peak_period: float,
    saved_times: np.ndarray,
    policy: JonswapNonlinearAdjustmentPolicy,
) -> dict[str, float | int | bool | str | None]:
    intended_terminal = policy.burn_peak_periods * peak_period
    failed = case.decision.failed
    clearance_evaluated = bool(
        case.decision.evaluated & QualityReason.BOTTOM_CLEARANCE
    )
    return {
        "nonlinear_adjustment_schema": ADJUSTMENT_SCHEMA,
        "nonlinear_adjustment_accepted": case.accepted,
        "nonlinear_adjustment_complete_admissible_handoff": not bool(
            failed & QualityReason.INCOMPLETE_TRAJECTORY
        ),
        "nonlinear_adjustment_state_finite": not bool(
            failed & QualityReason.NONFINITE_STATE
        ),
        "nonlinear_adjustment_positive_water_column": (
            not bool(failed & QualityReason.BOTTOM_CLEARANCE)
            if clearance_evaluated
            else None
        ),
        "nonlinear_adjustment_all_stages_solved": (
            case.telemetry.all_stages_solved
        ),
        "nonlinear_adjustment_maximum_stage_residual": _finite_or_none(
            case.telemetry.maximum_stage_residual
        ),
        "nonlinear_adjustment_minimum_water_column": _finite_or_none(
            case.minimum_water_column
        ),
        "nonlinear_adjustment_ramp_order": policy.ramp_order,
        "nonlinear_adjustment_ramp_time": (
            policy.ramp_time_peak_periods * peak_period
        ),
        "nonlinear_adjustment_intended_terminal_time": intended_terminal,
        "nonlinear_adjustment_realized_terminal_time": float(saved_times[-1]),
        "nonlinear_adjustment_saved_time_count": int(saved_times.size),
        "nonlinear_adjustment_production_ramp": "disabled",
    }


def _failed_adjustment_outcome(
    case: NonlinearAdjustmentCaseResult,
    *,
    construction_metrics: Mapping[str, object],
    peak_period: float,
    saved_times: np.ndarray,
    production_grid: CaseTimeGrid,
    policy: JonswapNonlinearAdjustmentPolicy,
) -> CaseOutcome:
    decision = case.decision
    return CaseOutcome(
        decision=QualityDecision(
            scope=QualityScope.TRAJECTORY,
            required=decision.required | QualityReason.OUTSIDE_SUPPORT,
            evaluated=decision.evaluated | QualityReason.OUTSIDE_SUPPORT,
            failed=decision.failed,
        ),
        rows=None,
        metrics={
            **construction_metrics,
            **_adjustment_metrics(
                case,
                peak_period=peak_period,
                saved_times=saved_times,
                policy=policy,
            ),
            "production_status": "not_run_adjustment_failed",
            "intended_terminal_time": production_grid.intended_terminal_time,
            "realized_terminal_time": production_grid.realized_terminal_time,
            "saved_time_count": int(production_grid.saved_times.size),
        },
    )


def _with_adjustment_metrics(
    outcome: CaseOutcome,
    case: NonlinearAdjustmentCaseResult,
    *,
    peak_period: float,
    saved_times: np.ndarray,
    policy: JonswapNonlinearAdjustmentPolicy,
) -> CaseOutcome:
    return CaseOutcome(
        decision=outcome.decision,
        rows=outcome.rows,
        metrics={
            **outcome.metrics,
            **_adjustment_metrics(
                case,
                peak_period=peak_period,
                saved_times=saved_times,
                policy=policy,
            ),
            "production_status": "completed",
        },
    )


@dataclass(frozen=True)
class HorizonBucketedJonswapQuotaExecutor(TrajectoryQuotaExecutor):
    """Execute one durable JONSWAP proposal in horizon-near solver groups."""

    adjustment_arm_executor: NonlinearAdjustmentArmExecutor = (
        run_nonlinear_adjustment_arm
    )

    def __post_init__(self) -> None:
        if self.execution.family != "jonswap_tma":
            raise ValueError(
                "horizon bucketing is implemented only for JONSWAP/TMA"
            )
        adjustment_policy = self.execution.jonswap_adjustment
        if adjustment_policy is None:
            raise ValueError(
                "horizon-bucketed JONSWAP execution requires an adjustment policy"
            )
        raw_policy = self.run_spec.configuration.get(POLICY_KEY)
        if not isinstance(raw_policy, Mapping):
            raise ValueError("run configuration omits JONSWAP bucketing policy")
        outer_size = raw_policy.get("outer_proposal_size")
        solver_size = raw_policy.get("solver_batch_size")
        if (
            isinstance(solver_size, bool)
            or not isinstance(solver_size, int)
            or solver_size <= 0
        ):
            raise ValueError(
                "configured solver_batch_size must be a positive integer"
            )
        expected = policy_record(
            outer_proposal_size=self.run_spec.batch_size,
            solver_batch_size=solver_size,
            adjustment_policy=adjustment_policy,
        )
        if dict(raw_policy) != expected or outer_size != self.run_spec.batch_size:
            raise ValueError("configured JONSWAP bucketing policy is inconsistent")
        if (
            self.execution.role == "paper_dataset"
            and self.adjustment_arm_executor is not run_nonlinear_adjustment_arm
        ):
            raise ValueError(
                "paper-dataset execution requires the production adjustment executor"
            )

        metadata = dict(self.metadata or {})
        metadata["launcher"] = (
            "scripts/run_paper_dataset_jonswap_bucketed.py"
        )
        metadata[POLICY_KEY] = expected
        object.__setattr__(self, "metadata", metadata)
        super().__post_init__()

    def _implemented_jonswap_adjustment(
        self,
    ) -> JonswapNonlinearAdjustmentPolicy | None:
        """Return the policy implemented by the adjustment-first rollout."""

        return self.execution.jonswap_adjustment

    @property
    def solver_batch_size(self) -> int:
        """Return the numerical rollout width recorded in the run policy."""

        raw_policy = self.run_spec.configuration[POLICY_KEY]
        assert isinstance(raw_policy, Mapping)
        value = raw_policy["solver_batch_size"]
        assert isinstance(value, int) and not isinstance(value, bool)
        return value

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
        adjustment_policy = self.execution.jonswap_adjustment
        assert adjustment_policy is not None
        if any(grid.saved_times.size < minimum_stored for grid in grids):
            raise ValueError(
                "trajectory horizon has fewer saved times than the storage policy"
            )

        groups = horizon_sorted_groups(
            grids,
            solver_batch_size=self.solver_batch_size,
        )
        grouped_outcomes: list[tuple[CaseOutcome, ...]] = []
        for indices in groups:
            selected_initial = _select_initial_batch(initial, indices)
            selected_grids = tuple(grids[index] for index in indices)
            peak_periods = _peak_periods(
                selected_initial,
                gravity=self.execution.numerical.gravity,
            )
            adjustment_grids = tuple(
                _floored_time_grid(
                    adjustment_policy.burn_peak_periods * peak_period,
                    saved_dt=self.execution.numerical.saved_dt,
                )
                for peak_period in peak_periods
            )
            adjustment = execute_variable_horizon_nonlinear_adjustment(
                selected_initial.eta0,
                selected_initial.xi0,
                selected_initial.depths,
                adjustment_grids,
                nonlinear_ramp_times=(
                    adjustment_policy.ramp_time_peak_periods * peak_periods
                ),
                nonlinear_ramp_order=adjustment_policy.ramp_order,
                contract=self.execution.numerical,
                arm_executor=self.adjustment_arm_executor,
            )
            accepted_indices = tuple(
                index
                for index, case in enumerate(adjustment.cases)
                if case.accepted
            )
            production_by_index: dict[int, CaseOutcome] = {}
            if accepted_indices:
                accepted_cases = tuple(
                    adjustment.cases[index] for index in accepted_indices
                )
                if any(
                    case.terminal_eta is None or case.terminal_xi is None
                    for case in accepted_cases
                ):
                    raise RuntimeError(
                        "accepted nonlinear adjustment omitted a full-band endpoint"
                    )
                adjusted_initial = TrajectoryInitialBatch(
                    eta0=np.stack(
                        tuple(
                            cast(np.ndarray, case.terminal_eta)
                            for case in accepted_cases
                        )
                    ),
                    xi0=np.stack(
                        tuple(
                            cast(np.ndarray, case.terminal_xi)
                            for case in accepted_cases
                        )
                    ),
                    depths=np.take(
                        selected_initial.depths,
                        np.asarray(accepted_indices, dtype=np.int64),
                        axis=0,
                    ),
                    specification_records=tuple(
                        selected_initial.specification_records[index]
                        for index in accepted_indices
                    ),
                    construction_metrics=(
                        tuple(
                            selected_initial.construction_metrics[index]
                            for index in accepted_indices
                        )
                        if selected_initial.construction_metrics
                        else ()
                    ),
                )
                production = super()._produce_jonswap(
                    adjusted_initial,
                    tuple(selected_grids[index] for index in accepted_indices),
                )
                production_by_index = dict(zip(accepted_indices, production))

            group_outcomes = tuple(
                _with_adjustment_metrics(
                    production_by_index[index],
                    case,
                    peak_period=float(peak_periods[index]),
                    saved_times=adjustment_grids[index],
                    policy=adjustment_policy,
                )
                if case.accepted
                else _failed_adjustment_outcome(
                    case,
                    construction_metrics=(
                        selected_initial.construction_metrics[index]
                        if selected_initial.construction_metrics
                        else {}
                    ),
                    peak_period=float(peak_periods[index]),
                    saved_times=adjustment_grids[index],
                    production_grid=selected_grids[index],
                    policy=adjustment_policy,
                )
                for index, case in enumerate(adjustment.cases)
            )
            grouped_outcomes.append(group_outcomes)
        return restore_proposal_order(
            groups,
            grouped_outcomes,
            case_count=len(grids),
        )
