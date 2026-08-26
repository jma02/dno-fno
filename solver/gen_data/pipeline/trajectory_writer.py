"""Convert accepted trajectory executions into complete writer outcomes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.acceptance import TrajectorySamples
from solver.gen_data.pipeline.quality import QualityReason
from solver.gen_data.pipeline.refinement import (
    ProductionCaseResult,
    ProductionExecution,
)
from solver.gen_data.pipeline.time_selection import (
    select_tanaka_times,
    select_uniform_times,
)
from solver.gen_data.pipeline.writer import (
    AcceptedCaseRows,
    CaseOutcome,
)


FloatArray: TypeAlias = NDArray[np.float64]
TrajectoryFamily: TypeAlias = Literal[
    "tanaka",
    "benjamin_feir",
    "jonswap_tma",
]


@dataclass(frozen=True)
class StoredTimePolicy:
    """Number and weighting parameters for retained trajectory frames."""

    tanaka_count: int = 200
    tanaka_alpha: float = 0.5
    tanaka_sigma_steps: float = 50.0
    benjamin_feir_count: int = 200
    random_sea_count: int = 16

    def __post_init__(self) -> None:
        for count, name in (
            (self.tanaka_count, "tanaka_count"),
            (self.benjamin_feir_count, "benjamin_feir_count"),
            (self.random_sea_count, "random_sea_count"),
        ):
            if count < 2:
                raise ValueError(f"{name} must be at least two")
        for alpha, name in ((self.tanaka_alpha, "tanaka_alpha"),):
            if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        for sigma, name in ((self.tanaka_sigma_steps, "tanaka_sigma_steps"),):
            if not math.isfinite(sigma) or sigma < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


PAPER_STORED_TIME_POLICY = StoredTimePolicy()


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _selected_indices(
    trajectory: TrajectorySamples,
    *,
    family: TrajectoryFamily,
    length: float,
    policy: StoredTimePolicy,
) -> NDArray[np.int32]:
    if family == "tanaka":
        return select_tanaka_times(
            trajectory.eta,
            length=length,
            keep_samples=policy.tanaka_count,
            alpha=policy.tanaka_alpha,
            sigma_steps=policy.tanaka_sigma_steps,
        ).indices
    if family == "benjamin_feir":
        return select_uniform_times(
            trajectory.times.size,
            keep_samples=policy.benjamin_feir_count,
        )
    if family == "jonswap_tma":
        return select_uniform_times(
            trajectory.times.size,
            keep_samples=policy.random_sea_count,
        )
    raise ValueError(f"unknown trajectory family: {family}")


def _production_case_metrics(
    case: ProductionCaseResult,
) -> dict[str, float | int | bool | None]:
    failed = case.decision.failed
    internal = case.internal_metrics
    clearance_evaluated = bool(
        case.decision.evaluated & QualityReason.BOTTOM_CLEARANCE
    )
    return {
        "accepted": case.accepted,
        "complete_admissible_trajectory": not bool(
            failed & QualityReason.INCOMPLETE_TRAJECTORY
        ),
        "state_finite": not bool(failed & QualityReason.NONFINITE_STATE),
        "target_finite": not bool(failed & QualityReason.NONFINITE_TARGET),
        "positive_water_column": (
            not bool(failed & QualityReason.BOTTOM_CLEARANCE)
            if clearance_evaluated
            else None
        ),
        "all_stages_solved": case.telemetry.all_stages_solved,
        "maximum_stage_residual": _finite_or_none(
            case.telemetry.maximum_stage_residual
        ),
        "production_dt": case.dt,
        "internal_health_evaluated": internal is not None,
        "internal_state_finite": (
            internal.state_finite if internal is not None else None
        ),
        "internal_dno_finite": (
            internal.dno_output_finite if internal is not None else None
        ),
        "minimum_internal_water_column": (
            internal.minimum_water_column if internal is not None else None
        ),
        "initial_internal_hamiltonian": (
            internal.initial_hamiltonian if internal is not None else None
        ),
        "maximum_internal_hamiltonian_drift": (
            internal.maximum_relative_hamiltonian_drift
            if internal is not None
            else None
        ),
        "internal_hamiltonian_drift_threshold": (
            internal.hamiltonian_drift_threshold
            if internal is not None
            else None
        ),
    }


def outcomes_from_production(
    execution: ProductionExecution,
    depths: FloatArray,
    *,
    family: TrajectoryFamily,
    length: float,
    policy: StoredTimePolicy = PAPER_STORED_TIME_POLICY,
) -> tuple[CaseOutcome, ...]:
    """Select frames from complete single-arm production trajectories."""

    depth_values = np.asarray(depths, dtype=np.float64)
    if depth_values.shape != (len(execution.cases),):
        raise ValueError("depths must contain one value per production case")
    if not np.isfinite(depth_values).all() or np.any(depth_values <= 0.0):
        raise ValueError("depths must be finite and positive")
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("length must be finite and positive")
    if tuple(case.case_index for case in execution.cases) != tuple(
        range(len(execution.cases))
    ):
        raise ValueError("production cases must be in input order")

    outcomes: list[CaseOutcome] = []
    for case, depth in zip(execution.cases, depth_values):
        rows = None
        trajectory = case.retained_trajectory
        if case.accepted:
            if trajectory is None:
                raise RuntimeError("accepted production case has no trajectory")
            indices = _selected_indices(
                trajectory,
                family=family,
                length=length,
                policy=policy,
            )
            rows = AcceptedCaseRows(
                eta=np.take(trajectory.eta, indices, axis=0),
                xi=np.take(trajectory.xi, indices, axis=0),
                gxi=np.take(trajectory.gxi, indices, axis=0),
                depth=float(depth),
                time=np.take(trajectory.times, indices),
                selected_dense_index=indices,
            )
        elif trajectory is not None:
            raise RuntimeError("rejected production case retained a trajectory")
        outcomes.append(
            CaseOutcome(
                decision=case.decision,
                rows=rows,
                metrics=_production_case_metrics(case),
            )
        )
    return tuple(outcomes)
