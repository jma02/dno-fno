"""Single-arm GL2 production and separate time-refinement audit utilities.

Paper-corpus production uses the previously validated GL2 step and runs each
trajectory once.  The paired and single-halving executors remain available
for method-level numerical audits; they do not govern production acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Protocol, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.acceptance import (
    RefinementTrajectory,
    TemporalRefinementMetrics,
    evaluate_complete_numerical_trajectory,
    evaluate_temporal_refinement,
)
from solver.gen_data.pipeline.quality import QualityDecision
from solver.gen_data.pipeline.reference import (
    DiscreteDnoTarget,
    evaluate_discrete_dno_target,
)
from solver.solvers.dno_series_jax import (
    build_grid,
    make_linear_dno_symbol,
)
from solver.solvers.time_integrator import SolverParams, State, rollout

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int32]
BoolArray: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True)
class ResidualControlledGL2Contract:
    """Spatial, temporal, and nonlinear-solve choices for one execution."""

    nx: int = 1024
    length: float = 2.0 * math.pi
    gravity: float = 1.0
    dno_order: int = 6
    pad_factor: int = 8
    maximum_wavenumber: float = 128.0
    production_dt: float = 0.01
    saved_dt: float = 0.08
    gl2_residual_tolerance: float = 1.0e-8
    gl2_iteration_cap: int = 4
    refinement_tolerance: float = 1.0e-3
    relative_floor: float = 1.0e-12
    target_time_chunk_size: int = 8
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.nx % 2:
            raise ValueError("nx must be a positive even integer")
        if not math.isfinite(self.length) or self.length <= 0.0:
            raise ValueError("length must be finite and positive")
        if not math.isfinite(self.gravity) or self.gravity <= 0.0:
            raise ValueError("gravity must be finite and positive")
        if self.dno_order < 0:
            raise ValueError("dno_order must be nonnegative")
        if self.pad_factor < 1:
            raise ValueError("pad_factor must be positive")
        nyquist = math.pi * self.nx / self.length
        if not 0.0 < self.maximum_wavenumber < nyquist:
            raise ValueError(
                "maximum_wavenumber must lie strictly below Nyquist"
            )
        time_steps = (
            self.production_dt,
            self.audit_dt,
            self.audit_retry_dt,
        )
        if not all(math.isfinite(dt) and dt > 0.0 for dt in time_steps):
            raise ValueError("all time steps must be finite and positive")
        if not math.isfinite(self.saved_dt) or self.saved_dt <= 0.0:
            raise ValueError("saved_dt must be finite and positive")
        for dt in time_steps:
            _integer_ratio(self.saved_dt, dt, "saved_dt", "time step")
        if (
            not math.isfinite(self.gl2_residual_tolerance)
            or self.gl2_residual_tolerance <= 0.0
        ):
            raise ValueError(
                "gl2_residual_tolerance must be finite and positive"
            )
        if self.gl2_iteration_cap < 0:
            raise ValueError("gl2_iteration_cap must be nonnegative")
        if (
            not math.isfinite(self.refinement_tolerance)
            or self.refinement_tolerance < 0.0
        ):
            raise ValueError(
                "refinement_tolerance must be finite and nonnegative"
            )
        if (
            not math.isfinite(self.relative_floor)
            or self.relative_floor <= 0.0
        ):
            raise ValueError("relative_floor must be finite and positive")
        if self.target_time_chunk_size < 1:
            raise ValueError("target_time_chunk_size must be positive")
        if self.dtype != "float64":
            raise ValueError("the paper-corpus contract requires float64")

    @property
    def filter_fraction(self) -> float:
        """Hard-filter fraction that delivers ``maximum_wavenumber``."""

        nyquist = math.pi * self.nx / self.length
        return self.maximum_wavenumber / nyquist

    @property
    def audit_dt(self) -> float:
        """Half step used only by the time-refinement audit."""

        return 0.5 * self.production_dt

    @property
    def audit_retry_dt(self) -> float:
        """Quarter step retained by the bounded audit utility."""

        return 0.25 * self.production_dt

    @property
    def target_definition(self) -> DiscreteDnoTarget:
        """Frozen-style DNO target specialized to this numerical contract."""

        return DiscreteDnoTarget(
            nx=self.nx,
            length=self.length,
            dno_order=self.dno_order,
            pad_factor=self.pad_factor,
            maximum_wavenumber=self.maximum_wavenumber,
        )


@dataclass(frozen=True)
class ArmTiming:
    """Wall-clock timings for an actual numerical arm."""

    rollout_seconds: float
    target_seconds: float
    host_transfer_seconds: float

    @property
    def total_seconds(self) -> float:
        return (
            self.rollout_seconds
            + self.target_seconds
            + self.host_transfer_seconds
        )


@dataclass(frozen=True)
class ResidualControlledArm:
    """One fixed-step batch arm, including every GL2 stage diagnostic."""

    dt: float
    times: FloatArray
    eta: FloatArray
    xi: FloatArray
    q_ref: FloatArray
    complete: BoolArray
    gl2_stage_residual: FloatArray
    gl2_iterations: IntArray
    gl2_converged: BoolArray
    gl2_stage_finite: BoolArray
    gl2_state_finite: BoolArray
    gl2_hit_iteration_cap: BoolArray
    timing: ArmTiming | None = None


@dataclass(frozen=True)
class CaseGL2Telemetry:
    """All substep telemetry for one case in one arm."""

    residual: FloatArray
    iterations: IntArray
    converged: BoolArray
    stage_finite: BoolArray
    state_finite: BoolArray
    hit_iteration_cap: BoolArray
    all_stages_solved: bool
    maximum_stage_residual: float


@dataclass(frozen=True)
class RefinementAttempt:
    """One consecutive time-step comparison for one case."""

    coarse_dt: float
    fine_dt: float
    coarse_telemetry: CaseGL2Telemetry
    fine_telemetry: CaseGL2Telemetry
    metrics: TemporalRefinementMetrics
    decision: QualityDecision


@dataclass(frozen=True)
class CaseRefinementResult:
    """Final result for one input case after the bounded retry policy."""

    case_index: int
    primary: RefinementAttempt
    retry_eligible: bool
    retry_reason: str
    retry: RefinementAttempt | None
    accepted: bool
    retained_dt: float | None
    retained_trajectory: RefinementTrajectory | None

    @property
    def effective_decision(self) -> QualityDecision:
        """Return the decision whose masks govern final acceptance."""

        return self.retry.decision if self.retry is not None else self.primary.decision


@dataclass(frozen=True)
class RefinementExecution:
    """Batch timings, retry routing, and ordered per-case results.

    Rejected state arrays are deliberately absent. Their decision and GL2
    telemetry remain available, but only an accepted case owns a retained
    trajectory.
    """

    primary_coarse_timing: ArmTiming | None
    primary_fine_timing: ArmTiming | None
    retry_timing: ArmTiming | None
    retry_case_indices: NDArray[np.int64]
    cases: tuple[CaseRefinementResult, ...]

    @property
    def accepted_mask(self) -> BoolArray:
        return np.asarray([case.accepted for case in self.cases], dtype=np.bool_)

    @property
    def retry_was_run(self) -> bool:
        return bool(self.retry_case_indices.size)


@dataclass(frozen=True)
class ProductionCaseResult:
    """One single-arm production decision and its numerical diagnostics."""

    case_index: int
    dt: float
    telemetry: CaseGL2Telemetry
    decision: QualityDecision
    retained_trajectory: RefinementTrajectory | None

    @property
    def accepted(self) -> bool:
        return self.decision.accepted


@dataclass(frozen=True)
class ProductionExecution:
    """One arm timing and ordered results for a production batch."""

    timing: ArmTiming | None
    cases: tuple[ProductionCaseResult, ...]

    @property
    def accepted_mask(self) -> BoolArray:
        return np.asarray([case.accepted for case in self.cases], dtype=np.bool_)


class ArmExecutor(Protocol):
    """Callable used to run one fixed-step batch arm."""

    def __call__(
        self,
        *,
        eta0: FloatArray,
        xi0: FloatArray,
        depths: FloatArray,
        saved_times: FloatArray,
        contract: ResidualControlledGL2Contract,
        dt: float,
    ) -> ResidualControlledArm: ...


def _integer_ratio(
    numerator: float,
    denominator: float,
    numerator_name: str,
    denominator_name: str,
) -> int:
    ratio = int(round(numerator / denominator))
    if ratio < 1 or not math.isclose(
        ratio * denominator,
        numerator,
        rel_tol=0.0,
        abs_tol=1.0e-14,
    ):
        raise ValueError(
            f"{numerator_name} must be an integer multiple of {denominator_name}"
        )
    return ratio


PAPER_GL2_CONTRACT = ResidualControlledGL2Contract()


def _validated_inputs(
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    saved_times: FloatArray,
    contract: ResidualControlledGL2Contract,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    eta = np.asarray(eta0, dtype=np.float64)
    xi = np.asarray(xi0, dtype=np.float64)
    depth = np.asarray(depths, dtype=np.float64)
    times = np.asarray(saved_times, dtype=np.float64)
    if eta.ndim != 2 or eta.shape[-1] != contract.nx:
        raise ValueError(
            f"eta0 must have shape (batch, {contract.nx}), got {eta.shape}"
        )
    if xi.shape != eta.shape:
        raise ValueError("xi0 must have the same shape as eta0")
    if depth.shape != (eta.shape[0],):
        raise ValueError(f"depths must have shape ({eta.shape[0]},)")
    if not np.isfinite(depth).all() or np.any(depth <= 0.0):
        raise ValueError("depths must be finite and positive")
    if times.ndim != 1 or times.size < 2:
        raise ValueError("saved_times must contain at least two times")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
        raise ValueError("saved_times must be finite and strictly increasing")
    if not np.allclose(
        np.diff(times),
        contract.saved_dt,
        rtol=0.0,
        atol=1.0e-13,
    ):
        raise ValueError("saved_times must use the contract saved_dt")
    return eta, xi, depth, times


def _evaluate_saved_q_ref(
    eta: jax.Array,
    xi: jax.Array,
    depths: jax.Array,
    contract: ResidualControlledGL2Contract,
) -> FloatArray:
    """Evaluate ``P_K G_N^(M,p)(P_K eta) P_K xi`` in time chunks."""

    result = np.empty(eta.shape, dtype=np.float64)
    chunk_size = contract.target_time_chunk_size
    for start in range(0, eta.shape[0], chunk_size):
        stop = min(start + chunk_size, eta.shape[0])
        count = stop - start
        eta_chunk = eta[start:stop]
        xi_chunk = xi[start:stop]
        if count < chunk_size:
            padding = chunk_size - count
            eta_chunk = jnp.pad(eta_chunk, ((0, padding), (0, 0), (0, 0)))
            xi_chunk = jnp.pad(xi_chunk, ((0, padding), (0, 0), (0, 0)))
        _, _, q_ref = evaluate_discrete_dno_target(
            eta_chunk,
            xi_chunk,
            depths[:, None],
            definition=contract.target_definition,
        )
        result[start:stop] = np.asarray(
            jax.device_get(q_ref[:count]), dtype=np.float64
        )
    return result


def run_residual_controlled_arm(
    *,
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    saved_times: FloatArray,
    contract: ResidualControlledGL2Contract,
    dt: float,
) -> ResidualControlledArm:
    """Run one actual float64 GL2 arm and evaluate the common target."""

    if not jax.config.x64_enabled:
        raise RuntimeError("residual-controlled paper-corpus GL2 requires float64")
    eta, xi, depth, times = _validated_inputs(
        eta0, xi0, depths, saved_times, contract
    )
    substeps = _integer_ratio(
        contract.saved_dt,
        dt,
        "saved_dt",
        "arm dt",
    )
    _, wavenumbers = build_grid(contract.nx, contract.length)
    k = jnp.asarray(wavenumbers, dtype=jnp.float64)
    depth_device = jnp.asarray(depth, dtype=jnp.float64)
    depth_column = depth_device[:, None]
    parameters = SolverParams(
        nx=contract.nx,
        length=contract.length,
        depth=depth_column,
        gravity=contract.gravity,
        dno_order=contract.dno_order,
        pad_factor=contract.pad_factor,
        filter_fraction=contract.filter_fraction,
        k=k,
        g0=make_linear_dno_symbol(k, depth_column),
    )

    started = perf_counter()
    payload = rollout(
        State(
            eta=jnp.asarray(eta, dtype=jnp.float64),
            xi=jnp.asarray(xi, dtype=jnp.float64),
        ),
        jnp.asarray(times, dtype=jnp.float64),
        parameters,
        save_gxi=False,
        substeps_per_interval=substeps,
        method="gl2_if",
        implicit_iterations=contract.gl2_iteration_cap,
        implicit_residual_tolerance=contract.gl2_residual_tolerance,
        implicit_relaxation=1.0,
        zero_mean_xi=True,
    )
    jax.block_until_ready(payload["xi"])
    rollout_seconds = perf_counter() - started

    started = perf_counter()
    q_ref = _evaluate_saved_q_ref(
        payload["eta"],
        payload["xi"],
        depth_device,
        contract,
    )
    target_seconds = perf_counter() - started

    started = perf_counter()
    host = {
        name: np.asarray(jax.device_get(value))
        for name, value in payload.items()
    }
    host_transfer_seconds = perf_counter() - started
    arm = ResidualControlledArm(
        dt=float(dt),
        times=np.asarray(host["times"], dtype=np.float64),
        eta=np.asarray(host["eta"], dtype=np.float64),
        xi=np.asarray(host["xi"], dtype=np.float64),
        q_ref=q_ref,
        complete=np.ones(eta.shape[0], dtype=np.bool_),
        gl2_stage_residual=np.asarray(
            host["gl2_stage_residual"], dtype=np.float64
        ),
        gl2_iterations=np.asarray(host["gl2_iterations"], dtype=np.int32),
        gl2_converged=np.asarray(host["gl2_converged"], dtype=np.bool_),
        gl2_stage_finite=np.asarray(
            host["gl2_stage_finite"], dtype=np.bool_
        ),
        gl2_state_finite=np.asarray(
            host["gl2_state_finite"], dtype=np.bool_
        ),
        gl2_hit_iteration_cap=np.asarray(
            host["gl2_hit_iteration_cap"], dtype=np.bool_
        ),
        timing=ArmTiming(
            rollout_seconds=rollout_seconds,
            target_seconds=target_seconds,
            host_transfer_seconds=host_transfer_seconds,
        ),
    )
    _validate_arm(arm, eta.shape[0], contract.nx, times, dt)
    return arm


def _validate_arm(
    arm: ResidualControlledArm,
    batch_size: int,
    nx: int,
    saved_times: FloatArray,
    expected_dt: float,
) -> None:
    if arm.dt <= 0.0 or not math.isfinite(arm.dt):
        raise ValueError("arm dt must be finite and positive")
    if not math.isclose(
        arm.dt,
        expected_dt,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("arm dt does not equal the requested time step")
    if not np.array_equal(np.asarray(arm.times), saved_times):
        raise ValueError("arm times do not equal the requested saved times")
    field_shape = (saved_times.size, batch_size, nx)
    for name in ("eta", "xi", "q_ref"):
        if np.asarray(getattr(arm, name)).shape != field_shape:
            raise ValueError(f"arm {name} must have shape {field_shape}")
    if np.asarray(arm.complete).shape != (batch_size,):
        raise ValueError(f"arm complete must have shape ({batch_size},)")
    step_shape = np.asarray(arm.gl2_stage_residual).shape
    if len(step_shape) != 2 or step_shape[1] != batch_size:
        raise ValueError("arm GL2 telemetry must have shape (step, batch)")
    for name in (
        "gl2_iterations",
        "gl2_converged",
        "gl2_stage_finite",
        "gl2_state_finite",
        "gl2_hit_iteration_cap",
    ):
        if np.asarray(getattr(arm, name)).shape != step_shape:
            raise ValueError("all arm GL2 telemetry arrays must have one shape")


def _case_telemetry(
    arm: ResidualControlledArm,
    case_index: int,
    contract: ResidualControlledGL2Contract,
) -> CaseGL2Telemetry:
    residual = np.asarray(
        arm.gl2_stage_residual[:, case_index], dtype=np.float64
    )
    iterations = np.asarray(
        arm.gl2_iterations[:, case_index], dtype=np.int32
    )
    converged = np.asarray(arm.gl2_converged[:, case_index], dtype=np.bool_)
    stage_finite = np.asarray(
        arm.gl2_stage_finite[:, case_index], dtype=np.bool_
    )
    state_finite = np.asarray(
        arm.gl2_state_finite[:, case_index], dtype=np.bool_
    )
    hit_iteration_cap = np.asarray(
        arm.gl2_hit_iteration_cap[:, case_index], dtype=np.bool_
    )
    solved = (
        converged
        & stage_finite
        & state_finite
        & np.isfinite(residual)
        & (residual <= contract.gl2_residual_tolerance)
    )
    maximum = float(np.max(residual)) if residual.size else 0.0
    return CaseGL2Telemetry(
        residual=residual,
        iterations=iterations,
        converged=converged,
        stage_finite=stage_finite,
        state_finite=state_finite,
        hit_iteration_cap=hit_iteration_cap,
        all_stages_solved=bool(np.all(solved)),
        maximum_stage_residual=maximum,
    )


def _case_trajectory(
    arm: ResidualControlledArm,
    case_index: int,
    telemetry: CaseGL2Telemetry,
) -> RefinementTrajectory:
    return RefinementTrajectory(
        times=np.asarray(arm.times, dtype=np.float64),
        eta=np.asarray(arm.eta[:, case_index], dtype=np.float64),
        xi=np.asarray(arm.xi[:, case_index], dtype=np.float64),
        gxi=np.asarray(arm.q_ref[:, case_index], dtype=np.float64),
        complete=bool(arm.complete[case_index]),
        gl2_stages_solved=telemetry.all_stages_solved,
    )


def _evaluate_production_case(
    arm: ResidualControlledArm,
    case_index: int,
    depth: float,
    contract: ResidualControlledGL2Contract,
    *,
    result_case_index: int | None = None,
) -> ProductionCaseResult:
    telemetry = _case_telemetry(arm, case_index, contract)
    trajectory = _case_trajectory(arm, case_index, telemetry)
    decision = evaluate_complete_numerical_trajectory(
        trajectory,
        depth=depth,
    )
    return ProductionCaseResult(
        case_index=(
            case_index if result_case_index is None else result_case_index
        ),
        dt=arm.dt,
        telemetry=telemetry,
        decision=decision,
        retained_trajectory=trajectory if decision.accepted else None,
    )


def execute_production_trajectory(
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    saved_times: FloatArray,
    *,
    contract: ResidualControlledGL2Contract = PAPER_GL2_CONTRACT,
    arm_executor: ArmExecutor = run_residual_controlled_arm,
) -> ProductionExecution:
    """Run each fixed-horizon production trajectory once at ``0.01``."""

    eta, xi, depth, times = _validated_inputs(
        eta0, xi0, depths, saved_times, contract
    )
    arm = arm_executor(
        eta0=eta,
        xi0=xi,
        depths=depth,
        saved_times=times,
        contract=contract,
        dt=contract.production_dt,
    )
    _validate_arm(
        arm,
        eta.shape[0],
        contract.nx,
        times,
        contract.production_dt,
    )
    return ProductionExecution(
        timing=arm.timing,
        cases=tuple(
            _evaluate_production_case(
                arm,
                case_index,
                float(depth[case_index]),
                contract,
            )
            for case_index in range(eta.shape[0])
        ),
    )


def _evaluate_attempt(
    coarse: ResidualControlledArm,
    coarse_index: int,
    fine: ResidualControlledArm,
    fine_index: int,
    depth: float,
    contract: ResidualControlledGL2Contract,
) -> RefinementAttempt:
    coarse_telemetry = _case_telemetry(coarse, coarse_index, contract)
    fine_telemetry = _case_telemetry(fine, fine_index, contract)
    metrics, decision = evaluate_temporal_refinement(
        _case_trajectory(coarse, coarse_index, coarse_telemetry),
        _case_trajectory(fine, fine_index, fine_telemetry),
        depth=depth,
        gravity=contract.gravity,
        length=contract.length,
        maximum_wavenumber=contract.maximum_wavenumber,
        tolerance=contract.refinement_tolerance,
        relative_floor=contract.relative_floor,
    )
    return RefinementAttempt(
        coarse_dt=coarse.dt,
        fine_dt=fine.dt,
        coarse_telemetry=coarse_telemetry,
        fine_telemetry=fine_telemetry,
        metrics=metrics,
        decision=decision,
    )


def _retry_eligibility(
    primary: RefinementAttempt,
    fine: ResidualControlledArm,
    case_index: int,
    depth: float,
) -> tuple[bool, str]:
    if primary.decision.accepted:
        return False, "primary pair accepted"
    if not bool(fine.complete[case_index]):
        return False, "fine arm is incomplete"
    if not primary.fine_telemetry.all_stages_solved:
        return False, "fine arm has an unsolved GL2 stage"
    fields = (
        fine.eta[:, case_index],
        fine.xi[:, case_index],
        fine.q_ref[:, case_index],
    )
    if not all(np.isfinite(field).all() for field in fields):
        return False, "fine arm contains a nonfinite field"
    if float(np.min(depth + fine.eta[:, case_index])) <= 0.0:
        return False, "fine arm leaves the water-wave graph domain"
    return True, "failed primary pair has a valid fine arm"


def execute_residual_controlled_refinement(
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    saved_times: FloatArray,
    *,
    contract: ResidualControlledGL2Contract = PAPER_GL2_CONTRACT,
    arm_executor: ArmExecutor = run_residual_controlled_arm,
) -> RefinementExecution:
    """Run the primary pair and the one permitted selective final halving.

    A primary failure is retried only when its fine arm is complete, finite,
    remains in the graph domain, and solved every implicit GL2 stage. Rejected
    cases retain no trajectory prefix.
    """

    eta, xi, depth, times = _validated_inputs(
        eta0, xi0, depths, saved_times, contract
    )
    coarse = arm_executor(
        eta0=eta,
        xi0=xi,
        depths=depth,
        saved_times=times,
        contract=contract,
        dt=contract.production_dt,
    )
    fine = arm_executor(
        eta0=eta,
        xi0=xi,
        depths=depth,
        saved_times=times,
        contract=contract,
        dt=contract.audit_dt,
    )
    _validate_arm(
        coarse,
        eta.shape[0],
        contract.nx,
        times,
        contract.production_dt,
    )
    _validate_arm(
        fine,
        eta.shape[0],
        contract.nx,
        times,
        contract.audit_dt,
    )

    primary = tuple(
        _evaluate_attempt(coarse, index, fine, index, depth[index], contract)
        for index in range(eta.shape[0])
    )
    eligibility = tuple(
        _retry_eligibility(primary[index], fine, index, depth[index])
        for index in range(eta.shape[0])
    )
    retry_indices = np.asarray(
        [
            index
            for index, (eligible, _) in enumerate(eligibility)
            if eligible
        ],
        dtype=np.int64,
    )

    retry_arm: ResidualControlledArm | None = None
    retry_by_case: dict[int, RefinementAttempt] = {}
    if retry_indices.size:
        retry_arm = arm_executor(
            eta0=np.take(eta, retry_indices, axis=0),
            xi0=np.take(xi, retry_indices, axis=0),
            depths=np.take(depth, retry_indices, axis=0),
            saved_times=times,
            contract=contract,
            dt=contract.audit_retry_dt,
        )
        _validate_arm(
            retry_arm,
            retry_indices.size,
            contract.nx,
            times,
            contract.audit_retry_dt,
        )
        retry_by_case = {
            int(case_index): _evaluate_attempt(
                fine,
                int(case_index),
                retry_arm,
                retry_index,
                depth[case_index],
                contract,
            )
            for retry_index, case_index in enumerate(retry_indices)
        }

    cases: list[CaseRefinementResult] = []
    for case_index, primary_attempt in enumerate(primary):
        retry_attempt = retry_by_case.get(case_index)
        effective = retry_attempt or primary_attempt
        accepted = bool(effective.decision.accepted)
        retained_arm = retry_arm if retry_attempt is not None else fine
        retained_index = (
            int(np.flatnonzero(retry_indices == case_index)[0])
            if retry_attempt is not None
            else case_index
        )
        retained_trajectory = None
        retained_dt = None
        if accepted:
            retained_telemetry = (
                effective.fine_telemetry
                if retry_attempt is not None
                else primary_attempt.fine_telemetry
            )
            assert retained_arm is not None
            retained_trajectory = _case_trajectory(
                retained_arm,
                retained_index,
                retained_telemetry,
            )
            retained_dt = retained_arm.dt
        eligible, reason = eligibility[case_index]
        cases.append(
            CaseRefinementResult(
                case_index=case_index,
                primary=primary_attempt,
                retry_eligible=eligible,
                retry_reason=reason,
                retry=retry_attempt,
                accepted=accepted,
                retained_dt=retained_dt,
                retained_trajectory=retained_trajectory,
            )
        )

    return RefinementExecution(
        primary_coarse_timing=coarse.timing,
        primary_fine_timing=fine.timing,
        retry_timing=retry_arm.timing if retry_arm is not None else None,
        retry_case_indices=retry_indices,
        cases=tuple(cases),
    )


def _validated_variable_horizon_inputs(
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    case_saved_times: tuple[FloatArray, ...],
    contract: ResidualControlledGL2Contract,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    tuple[FloatArray, ...],
    FloatArray,
]:
    if not case_saved_times:
        raise ValueError("case_saved_times must contain one grid per case")
    grids = tuple(
        np.asarray(saved_times, dtype=np.float64)
        for saved_times in case_saved_times
    )
    longest_index = max(range(len(grids)), key=lambda index: grids[index].size)
    eta, xi, depth, longest = _validated_inputs(
        eta0,
        xi0,
        depths,
        grids[longest_index],
        contract,
    )
    if len(grids) != eta.shape[0]:
        raise ValueError("case_saved_times must contain one grid per input case")

    validated_grids: list[FloatArray] = []
    for case_index, saved_times in enumerate(grids):
        _, _, _, validated = _validated_inputs(
            eta[case_index : case_index + 1],
            xi[case_index : case_index + 1],
            depth[case_index : case_index + 1],
            saved_times,
            contract,
        )
        if not np.array_equal(validated, longest[: validated.size]):
            raise ValueError(
                "every variable saved-time grid must be a prefix of the "
                "longest grid"
            )
        validated_grids.append(validated)
    return eta, xi, depth, tuple(validated_grids), longest


def _single_case_prefix_arm(
    arm: ResidualControlledArm,
    case_index: int,
    saved_times: FloatArray,
    contract: ResidualControlledGL2Contract,
) -> ResidualControlledArm:
    """Restrict one batch arm to one case and its declared time prefix."""

    if not 0 <= case_index < np.asarray(arm.complete).size:
        raise IndexError("arm case index is out of range")
    steps_per_saved_interval = _integer_ratio(
        contract.saved_dt,
        arm.dt,
        "saved_dt",
        "arm dt",
    )
    full_step_count = (arm.times.size - 1) * steps_per_saved_interval
    observed_step_count = np.asarray(arm.gl2_stage_residual).shape[0]
    if observed_step_count != full_step_count:
        raise ValueError(
            "variable-horizon batching requires one GL2 telemetry row "
            "per numerical substep"
        )
    saved_count = saved_times.size
    if saved_count > arm.times.size or not np.array_equal(
        saved_times,
        np.asarray(arm.times, dtype=np.float64)[:saved_count],
    ):
        raise ValueError("case saved times must be a prefix of the arm times")
    step_count = (saved_count - 1) * steps_per_saved_interval
    prefix = ResidualControlledArm(
        dt=arm.dt,
        times=np.asarray(saved_times, dtype=np.float64),
        eta=np.asarray(
            arm.eta[:saved_count, case_index : case_index + 1],
            dtype=np.float64,
        ),
        xi=np.asarray(
            arm.xi[:saved_count, case_index : case_index + 1],
            dtype=np.float64,
        ),
        q_ref=np.asarray(
            arm.q_ref[:saved_count, case_index : case_index + 1],
            dtype=np.float64,
        ),
        complete=np.asarray(
            arm.complete[case_index : case_index + 1],
            dtype=np.bool_,
        ),
        gl2_stage_residual=np.asarray(
            arm.gl2_stage_residual[:step_count, case_index : case_index + 1],
            dtype=np.float64,
        ),
        gl2_iterations=np.asarray(
            arm.gl2_iterations[:step_count, case_index : case_index + 1],
            dtype=np.int32,
        ),
        gl2_converged=np.asarray(
            arm.gl2_converged[:step_count, case_index : case_index + 1],
            dtype=np.bool_,
        ),
        gl2_stage_finite=np.asarray(
            arm.gl2_stage_finite[:step_count, case_index : case_index + 1],
            dtype=np.bool_,
        ),
        gl2_state_finite=np.asarray(
            arm.gl2_state_finite[:step_count, case_index : case_index + 1],
            dtype=np.bool_,
        ),
        gl2_hit_iteration_cap=np.asarray(
            arm.gl2_hit_iteration_cap[
                :step_count, case_index : case_index + 1
            ],
            dtype=np.bool_,
        ),
        timing=arm.timing,
    )
    _validate_arm(
        prefix,
        1,
        contract.nx,
        saved_times,
        arm.dt,
    )
    return prefix


def execute_variable_horizon_production_trajectory(
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    case_saved_times: tuple[FloatArray, ...],
    *,
    contract: ResidualControlledGL2Contract = PAPER_GL2_CONTRACT,
    arm_executor: ArmExecutor = run_residual_controlled_arm,
) -> ProductionExecution:
    """Run one batch arm, then decide each case on its declared time prefix."""

    eta, xi, depth, grids, longest = _validated_variable_horizon_inputs(
        eta0,
        xi0,
        depths,
        case_saved_times,
        contract,
    )
    arm = arm_executor(
        eta0=eta,
        xi0=xi,
        depths=depth,
        saved_times=longest,
        contract=contract,
        dt=contract.production_dt,
    )
    _validate_arm(
        arm,
        eta.shape[0],
        contract.nx,
        longest,
        contract.production_dt,
    )
    case_arms = tuple(
        _single_case_prefix_arm(arm, index, grid, contract)
        for index, grid in enumerate(grids)
    )
    return ProductionExecution(
        timing=arm.timing,
        cases=tuple(
            _evaluate_production_case(
                case_arm,
                0,
                float(depth[case_index]),
                contract,
                result_case_index=case_index,
            )
            for case_index, case_arm in enumerate(case_arms)
        ),
    )


def execute_variable_horizon_residual_controlled_refinement(
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    case_saved_times: tuple[FloatArray, ...],
    *,
    contract: ResidualControlledGL2Contract = PAPER_GL2_CONTRACT,
    arm_executor: ArmExecutor = run_residual_controlled_arm,
) -> RefinementExecution:
    """Run independent prefix horizons in one primary and one retry batch.

    The numerical arms advance every case to the longest requested saved-time
    grid. Each case is then restricted to its own grid, including its exact
    GL2 substep telemetry, before refinement acceptance or retry routing.
    Consequently, fields or stage failures after a case's declared horizon
    cannot change that case's outcome.
    """

    eta, xi, depth, grids, longest = _validated_variable_horizon_inputs(
        eta0,
        xi0,
        depths,
        case_saved_times,
        contract,
    )
    coarse = arm_executor(
        eta0=eta,
        xi0=xi,
        depths=depth,
        saved_times=longest,
        contract=contract,
        dt=contract.production_dt,
    )
    fine = arm_executor(
        eta0=eta,
        xi0=xi,
        depths=depth,
        saved_times=longest,
        contract=contract,
        dt=contract.audit_dt,
    )
    _validate_arm(
        coarse,
        eta.shape[0],
        contract.nx,
        longest,
        contract.production_dt,
    )
    _validate_arm(
        fine,
        eta.shape[0],
        contract.nx,
        longest,
        contract.audit_dt,
    )
    coarse_cases = tuple(
        _single_case_prefix_arm(coarse, index, grid, contract)
        for index, grid in enumerate(grids)
    )
    fine_cases = tuple(
        _single_case_prefix_arm(fine, index, grid, contract)
        for index, grid in enumerate(grids)
    )
    primary = tuple(
        _evaluate_attempt(
            coarse_cases[index],
            0,
            fine_cases[index],
            0,
            depth[index],
            contract,
        )
        for index in range(eta.shape[0])
    )
    eligibility = tuple(
        _retry_eligibility(
            primary[index],
            fine_cases[index],
            0,
            depth[index],
        )
        for index in range(eta.shape[0])
    )
    retry_indices = np.asarray(
        [
            index
            for index, (eligible, _) in enumerate(eligibility)
            if eligible
        ],
        dtype=np.int64,
    )

    retry_arm: ResidualControlledArm | None = None
    retry_case_arms: dict[int, ResidualControlledArm] = {}
    retry_by_case: dict[int, RefinementAttempt] = {}
    if retry_indices.size:
        retry_grids = tuple(grids[int(index)] for index in retry_indices)
        retry_longest = max(retry_grids, key=lambda grid: grid.size)
        retry_arm = arm_executor(
            eta0=np.take(eta, retry_indices, axis=0),
            xi0=np.take(xi, retry_indices, axis=0),
            depths=np.take(depth, retry_indices, axis=0),
            saved_times=retry_longest,
            contract=contract,
            dt=contract.audit_retry_dt,
        )
        _validate_arm(
            retry_arm,
            retry_indices.size,
            contract.nx,
            retry_longest,
            contract.audit_retry_dt,
        )
        retry_case_arms = {
            int(case_index): _single_case_prefix_arm(
                retry_arm,
                retry_index,
                grids[int(case_index)],
                contract,
            )
            for retry_index, case_index in enumerate(retry_indices)
        }
        retry_by_case = {
            int(case_index): _evaluate_attempt(
                fine_cases[int(case_index)],
                0,
                retry_case_arms[int(case_index)],
                0,
                depth[int(case_index)],
                contract,
            )
            for case_index in retry_indices
        }

    cases: list[CaseRefinementResult] = []
    for case_index, primary_attempt in enumerate(primary):
        retry_attempt = retry_by_case.get(case_index)
        effective = retry_attempt or primary_attempt
        accepted = bool(effective.decision.accepted)
        retained_arm = (
            retry_case_arms[case_index]
            if retry_attempt is not None
            else fine_cases[case_index]
        )
        retained_trajectory = None
        retained_dt = None
        if accepted:
            retained_telemetry = (
                retry_attempt.fine_telemetry
                if retry_attempt is not None
                else primary_attempt.fine_telemetry
            )
            retained_trajectory = _case_trajectory(
                retained_arm,
                0,
                retained_telemetry,
            )
            retained_dt = retained_arm.dt
        eligible, reason = eligibility[case_index]
        cases.append(
            CaseRefinementResult(
                case_index=case_index,
                primary=primary_attempt,
                retry_eligible=eligible,
                retry_reason=reason,
                retry=retry_attempt,
                accepted=accepted,
                retained_dt=retained_dt,
                retained_trajectory=retained_trajectory,
            )
        )

    return RefinementExecution(
        primary_coarse_timing=coarse.timing,
        primary_fine_timing=fine.timing,
        retry_timing=retry_arm.timing if retry_arm is not None else None,
        retry_case_indices=retry_indices,
        cases=tuple(cases),
    )
