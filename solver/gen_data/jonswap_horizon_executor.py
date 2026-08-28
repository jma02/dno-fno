"""Run JONSWAP cases in small batches grouped by rollout length.

A proposal may contain more cases than fit in GPU memory. Sort its cases by
saved timestep count, solve them in smaller groups, then restore the original
proposal order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math

import numpy as np

from solver.gen_data.jonswap_tma import finite_depth_angular_frequency
from solver.gen_data.pipeline.case_checks import (
    CaseCheckResult,
    CaseCheck,
)
from solver.gen_data.pipeline.trajectory_integration import (
    AdjustmentBatchIntegrator,
    integrate_adjustment_batch,
)
from solver.gen_data.pipeline.trajectory_rollout import (
    AdjustmentCaseResult,
    execute_variable_horizon_adjustment,
)
from solver.gen_data.pipeline.writer import CaseOutcome, JsonScalar
from solver.gen_data.trajectory_family_adapters import TrajectoryInitialBatch
from solver.gen_data.trajectory_batch_executor import (
    CaseTimeGrid,
    JONSWAP_ADJUSTMENT_SCHEMA,
    JonswapNonlinearAdjustmentPolicy,
    PAPER_JONSWAP_ADJUSTMENT_POLICY,
    TrajectoryBatchExecutor,
)


BUCKETING_CONFIG_KEY = "jonswap_horizon_bucketing"


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


@dataclass(frozen=True)
class BucketingConfig:
    """Batch sizes used to solve one proposal without exhausting GPU memory."""

    outer_proposal_size: int
    solver_batch_size: int
    adjustment: JonswapNonlinearAdjustmentPolicy = PAPER_JONSWAP_ADJUSTMENT_POLICY

    def __post_init__(self) -> None:
        for value, name in (
            (self.outer_proposal_size, "outer_proposal_size"),
            (self.solver_batch_size, "solver_batch_size"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.solver_batch_size > self.outer_proposal_size:
            raise ValueError("solver_batch_size cannot exceed outer_proposal_size")

    def to_json_record(self) -> dict[str, object]:
        return {
            "outer_proposal_size": self.outer_proposal_size,
            "solver_batch_size": self.solver_batch_size,
            "nonlinear_adjustment": self.adjustment.to_json_record(),
        }

    def matches_record(self, record: Mapping[str, object]) -> bool:
        """Return whether a stored record has the configured execution fields."""

        return all(
            record.get(key) == value for key, value in self.to_json_record().items()
        )


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


def _accepted_adjustment_batch(
    initial: TrajectoryInitialBatch,
    cases: Sequence[AdjustmentCaseResult],
) -> tuple[tuple[int, ...], TrajectoryInitialBatch | None]:
    """Build the production batch from accepted full-band endpoints."""

    accepted_indices = tuple(
        index for index, case in enumerate(cases) if case.decision.accepted
    )
    if not accepted_indices:
        return (), None

    endpoints: list[tuple[np.ndarray, np.ndarray]] = []
    for index in accepted_indices:
        eta = cases[index].terminal_eta
        xi = cases[index].terminal_xi
        if eta is None or xi is None:
            raise RuntimeError(
                "accepted nonlinear adjustment omitted a full-band endpoint"
            )
        endpoints.append((eta, xi))

    selected = _select_initial_batch(initial, accepted_indices)
    return accepted_indices, replace(
        selected,
        eta0=np.stack(tuple(eta for eta, _ in endpoints)),
        xi0=np.stack(tuple(xi for _, xi in endpoints)),
    )


def _peak_periods(
    initial: TrajectoryInitialBatch,
    *,
    gravity: float,
) -> np.ndarray:
    """Compute each finite-depth peak period from its immutable proposal."""

    depths = np.asarray(initial.depths, dtype=np.float64)
    peak_wavenumbers = np.asarray(
        [record["peak_wavenumber"] for record in initial.specification_records],
        dtype=np.float64,
    )
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


def _adjustment_metrics(
    case: AdjustmentCaseResult,
    *,
    peak_period: float,
    saved_times: np.ndarray,
    policy: JonswapNonlinearAdjustmentPolicy,
) -> dict[str, float | int | bool | str | None]:
    intended_terminal = policy.burn_peak_periods * peak_period
    failed = case.decision.failed
    clearance_evaluated = bool(case.decision.evaluated & CaseCheck.BOTTOM_CLEARANCE)
    return {
        "nonlinear_adjustment_schema": JONSWAP_ADJUSTMENT_SCHEMA,
        "nonlinear_adjustment_accepted": case.decision.accepted,
        "nonlinear_adjustment_complete_admissible_handoff": not bool(
            failed & CaseCheck.INCOMPLETE_TRAJECTORY
        ),
        "nonlinear_adjustment_state_finite": not bool(
            failed & CaseCheck.NONFINITE_STATE
        ),
        "nonlinear_adjustment_positive_water_column": (
            not bool(failed & CaseCheck.BOTTOM_CLEARANCE)
            if clearance_evaluated
            else None
        ),
        "nonlinear_adjustment_all_stages_solved": not bool(
            failed & CaseCheck.GL2_STAGE_RESIDUAL
        ),
        "nonlinear_adjustment_maximum_stage_residual": (
            case.maximum_gl2_stage_residual
            if math.isfinite(case.maximum_gl2_stage_residual)
            else None
        ),
        "nonlinear_adjustment_minimum_water_column": case.minimum_water_column,
        "nonlinear_adjustment_ramp_order": policy.ramp_order,
        "nonlinear_adjustment_ramp_time": (policy.ramp_time_peak_periods * peak_period),
        "nonlinear_adjustment_intended_terminal_time": intended_terminal,
        "nonlinear_adjustment_realized_terminal_time": float(saved_times[-1]),
        "nonlinear_adjustment_saved_time_count": int(saved_times.size),
        "nonlinear_adjustment_production_ramp": "disabled",
    }


def _failed_adjustment_outcome(
    case: AdjustmentCaseResult,
    *,
    construction_metrics: Mapping[str, JsonScalar],
    peak_period: float,
    saved_times: np.ndarray,
    production_grid: CaseTimeGrid,
    policy: JonswapNonlinearAdjustmentPolicy,
) -> CaseOutcome:
    decision = case.decision
    return CaseOutcome(
        decision=CaseCheckResult(
            required=decision.required | CaseCheck.OUTSIDE_SUPPORT,
            evaluated=decision.evaluated | CaseCheck.OUTSIDE_SUPPORT,
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
    case: AdjustmentCaseResult,
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
class HorizonBucketedJonswapBatchExecutor(TrajectoryBatchExecutor):
    """Solve one JONSWAP proposal in groups with similar rollout lengths."""

    adjustment_rollout_executor: AdjustmentBatchIntegrator = integrate_adjustment_batch

    def __post_init__(self) -> None:
        if self.execution.family != "jonswap_tma":
            raise ValueError("horizon bucketing is implemented only for JONSWAP/TMA")
        adjustment_policy = self.execution.jonswap_adjustment
        if adjustment_policy is None:
            raise ValueError(
                "horizon-bucketed JONSWAP execution requires an adjustment policy"
            )
        raw_config = self.run_spec.configuration.get(BUCKETING_CONFIG_KEY)
        if not isinstance(raw_config, Mapping):
            raise ValueError("run configuration omits JONSWAP bucketing config")
        solver_size = raw_config.get("solver_batch_size")
        if (
            isinstance(solver_size, bool)
            or not isinstance(solver_size, int)
            or solver_size <= 0
        ):
            raise ValueError("configured solver_batch_size must be a positive integer")
        expected = BucketingConfig(
            outer_proposal_size=self.run_spec.batch_size,
            solver_batch_size=solver_size,
            adjustment=adjustment_policy,
        )
        if not expected.matches_record(raw_config):
            raise ValueError("configured JONSWAP bucketing config is inconsistent")
        if (
            self.execution.role == "paper_dataset"
            and self.adjustment_rollout_executor is not integrate_adjustment_batch
        ):
            raise ValueError(
                "paper-dataset execution requires the production adjustment executor"
            )

        metadata = dict(self.metadata or {})
        metadata["launcher"] = "scripts/generate_paper_dataset_jonswap.py"
        metadata[BUCKETING_CONFIG_KEY] = expected.to_json_record()
        object.__setattr__(self, "metadata", metadata)
        super().__post_init__()

    def _implemented_jonswap_adjustment(
        self,
    ) -> JonswapNonlinearAdjustmentPolicy | None:
        """Return the policy implemented by the adjustment-first rollout."""

        return self.execution.jonswap_adjustment

    @property
    def solver_batch_size(self) -> int:
        """Return the numerical rollout width recorded in the bucketing config."""

        raw_config = self.run_spec.configuration[BUCKETING_CONFIG_KEY]
        if not isinstance(raw_config, Mapping):
            raise RuntimeError("validated JONSWAP bucketing config is missing")
        solver_batch_size = raw_config.get("solver_batch_size")
        if (
            isinstance(solver_batch_size, bool)
            or not isinstance(solver_batch_size, int)
            or solver_batch_size <= 0
        ):
            raise RuntimeError("validated solver_batch_size is invalid")
        return solver_batch_size

    def _produce_adjusted_group(
        self,
        initial: TrajectoryInitialBatch,
        grids: tuple[CaseTimeGrid, ...],
        policy: JonswapNonlinearAdjustmentPolicy,
    ) -> tuple[CaseOutcome, ...]:
        """Adjust one horizon-near group, then run accepted cases."""

        peak_periods = _peak_periods(
            initial,
            gravity=self.execution.numerical.gravity,
        )
        adjustment_grids = tuple(
            _floored_time_grid(
                policy.burn_peak_periods * peak_period,
                saved_dt=self.execution.numerical.saved_dt,
            )
            for peak_period in peak_periods
        )
        adjustment_cases = execute_variable_horizon_adjustment(
            initial.eta0,
            initial.xi0,
            initial.depths,
            adjustment_grids,
            nonlinear_ramp_times=policy.ramp_time_peak_periods * peak_periods,
            nonlinear_ramp_order=policy.ramp_order,
            config=self.execution.numerical,
            rollout_executor=self.adjustment_rollout_executor,
        )
        accepted_indices, adjusted_initial = _accepted_adjustment_batch(
            initial,
            adjustment_cases,
        )
        production_by_index: dict[int, CaseOutcome] = {}
        if adjusted_initial is not None:
            production = super()._produce_jonswap(
                adjusted_initial,
                tuple(grids[index] for index in accepted_indices),
            )
            production_by_index = dict(zip(accepted_indices, production))

        return tuple(
            _with_adjustment_metrics(
                production_by_index[index],
                case,
                peak_period=float(peak_periods[index]),
                saved_times=adjustment_grids[index],
                policy=policy,
            )
            if case.decision.accepted
            else _failed_adjustment_outcome(
                case,
                construction_metrics=(
                    initial.construction_metrics[index]
                    if initial.construction_metrics
                    else {}
                ),
                peak_period=float(peak_periods[index]),
                saved_times=adjustment_grids[index],
                production_grid=grids[index],
                policy=policy,
            )
            for index, case in enumerate(adjustment_cases)
        )

    def _produce_jonswap(
        self,
        initial: TrajectoryInitialBatch,
        grids: tuple[CaseTimeGrid, ...],
    ) -> tuple[CaseOutcome, ...]:
        if len(grids) != initial.eta0.shape[0]:
            raise ValueError("JONSWAP/TMA requires one time grid per initial condition")
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
        ordered_outcomes: list[CaseOutcome | None] = [None] * len(grids)
        for indices in groups:
            outcomes = self._produce_adjusted_group(
                _select_initial_batch(initial, indices),
                tuple(grids[index] for index in indices),
                adjustment_policy,
            )
            for index, outcome in zip(indices, outcomes):
                ordered_outcomes[index] = outcome
        if any(outcome is None for outcome in ordered_outcomes):
            raise RuntimeError("every JONSWAP case must produce an outcome")
        return tuple(outcome for outcome in ordered_outcomes if outcome is not None)
