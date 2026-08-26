"""Single-arm GL2 production and nonlinear-adjustment utilities."""
from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Literal, Protocol, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.acceptance import (
    DenseTrajectoryHealthMetrics,
    TrajectorySamples,
    evaluate_dense_trajectory_health,
    evaluate_production_trajectory,
)
from solver.gen_data.pipeline.quality import (
    QualityDecision,
    QualityReason,
    QualityScope,
)
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
PostStepStateFilter: TypeAlias = Literal["sharp", "hou_li"]


@dataclass(frozen=True)
class ResidualControlledGL2Contract:
    """Spatial, temporal, and nonlinear-solve choices for one execution."""

    nx: int = 1024
    target_nx: int | None = None
    length: float = 2.0 * math.pi
    gravity: float = 1.0
    dno_order: int = 6
    target_dno_order: int | None = None
    pad_factor: int = 8
    maximum_wavenumber: float = 128.0
    target_maximum_wavenumber: float | None = None
    production_dt: float = 0.01
    saved_dt: float = 0.08
    gl2_residual_tolerance: float = 1.0e-8
    gl2_iteration_cap: int = 4
    target_time_chunk_size: int = 8
    dtype: str = "float64"
    post_step_state_filter: PostStepStateFilter = "sharp"
    post_step_maximum_wavenumber: float | None = None
    hou_li_coefficient: float = 36.0
    hou_li_power: int = 36
    internal_hamiltonian_drift_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.nx % 2:
            raise ValueError("nx must be a positive even integer")
        if self.target_nx is not None and (
            isinstance(self.target_nx, bool)
            or not isinstance(self.target_nx, int)
            or self.target_nx <= 0
            or self.target_nx % 2
        ):
            raise ValueError("target_nx must be a positive even integer")
        if not math.isfinite(self.length) or self.length <= 0.0:
            raise ValueError("length must be finite and positive")
        if not math.isfinite(self.gravity) or self.gravity <= 0.0:
            raise ValueError("gravity must be finite and positive")
        if self.dno_order < 0:
            raise ValueError("dno_order must be nonnegative")
        if self.target_dno_order is not None and self.target_dno_order < 0:
            raise ValueError("target_dno_order must be nonnegative")
        if self.pad_factor < 1:
            raise ValueError("pad_factor must be positive")
        nyquist = math.pi * self.nx / self.length
        if not 0.0 < self.maximum_wavenumber < nyquist:
            raise ValueError(
                "maximum_wavenumber must lie strictly below Nyquist"
            )
        if self.target_maximum_wavenumber is not None and (
            not math.isfinite(self.target_maximum_wavenumber)
            or self.target_maximum_wavenumber <= 0.0
            or self.target_maximum_wavenumber > self.maximum_wavenumber
        ):
            raise ValueError(
                "target_maximum_wavenumber must be positive and no larger "
                "than maximum_wavenumber"
            )
        target_nyquist = math.pi * self.delivered_nx / self.length
        if self.target_definition.maximum_wavenumber >= target_nyquist:
            raise ValueError(
                "target maximum_wavenumber must lie strictly below the "
                "target-grid Nyquist"
            )
        if not math.isfinite(self.production_dt) or self.production_dt <= 0.0:
            raise ValueError("production_dt must be finite and positive")
        if not math.isfinite(self.saved_dt) or self.saved_dt <= 0.0:
            raise ValueError("saved_dt must be finite and positive")
        _integer_ratio(
            self.saved_dt,
            self.production_dt,
            "saved_dt",
            "production_dt",
        )
        if (
            not math.isfinite(self.gl2_residual_tolerance)
            or self.gl2_residual_tolerance <= 0.0
        ):
            raise ValueError(
                "gl2_residual_tolerance must be finite and positive"
            )
        if self.gl2_iteration_cap < 0:
            raise ValueError("gl2_iteration_cap must be nonnegative")
        if self.target_time_chunk_size < 1:
            raise ValueError("target_time_chunk_size must be positive")
        if self.dtype != "float64":
            raise ValueError("the paper-dataset contract requires float64")
        if self.post_step_state_filter not in ("sharp", "hou_li"):
            raise ValueError(
                "post_step_state_filter must be 'sharp' or 'hou_li'"
            )
        if self.post_step_maximum_wavenumber is not None and (
            not math.isfinite(self.post_step_maximum_wavenumber)
            or self.post_step_maximum_wavenumber <= 0.0
            or self.post_step_maximum_wavenumber > self.maximum_wavenumber
        ):
            raise ValueError(
                "post_step_maximum_wavenumber must be positive and no "
                "larger than maximum_wavenumber"
            )
        if (
            not math.isfinite(self.hou_li_coefficient)
            or self.hou_li_coefficient <= 0.0
        ):
            raise ValueError("hou_li_coefficient must be finite and positive")
        if (
            isinstance(self.hou_li_power, bool)
            or not isinstance(self.hou_li_power, int)
            or self.hou_li_power <= 0
            or self.hou_li_power % 2
        ):
            raise ValueError("hou_li_power must be a positive even integer")
        if self.internal_hamiltonian_drift_threshold is not None and (
            not math.isfinite(self.internal_hamiltonian_drift_threshold)
            or self.internal_hamiltonian_drift_threshold < 0.0
        ):
            raise ValueError(
                "internal_hamiltonian_drift_threshold must be finite and "
                "nonnegative"
            )

    @property
    def filter_fraction(self) -> float:
        """Hard-filter fraction that delivers ``maximum_wavenumber``."""

        nyquist = math.pi * self.nx / self.length
        return self.maximum_wavenumber / nyquist

    @property
    def delivered_nx(self) -> int:
        """Spatial size of projected fields written to the paper dataset."""

        return self.nx if self.target_nx is None else self.target_nx

    @property
    def post_step_filter_fraction(self) -> float:
        """Filter fraction applied only after a completed GL2 step."""

        maximum_wavenumber = (
            self.maximum_wavenumber
            if self.post_step_maximum_wavenumber is None
            else self.post_step_maximum_wavenumber
        )
        nyquist = math.pi * self.nx / self.length
        return maximum_wavenumber / nyquist

    @property
    def target_definition(self) -> DiscreteDnoTarget:
        """Frozen-style DNO target specialized to this numerical contract."""

        dno_order = (
            self.dno_order
            if self.target_dno_order is None
            else self.target_dno_order
        )
        maximum_wavenumber = (
            self.maximum_wavenumber
            if self.target_maximum_wavenumber is None
            else self.target_maximum_wavenumber
        )
        return DiscreteDnoTarget(
            nx=self.delivered_nx,
            length=self.length,
            dno_order=dno_order,
            pad_factor=self.pad_factor,
            maximum_wavenumber=maximum_wavenumber,
        )

    @property
    def internal_definition(self) -> DiscreteDnoTarget:
        """DNO definition used by the internal evolution and Hamiltonian."""

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
    internal_telemetry: InternalTrajectoryTelemetry | None = None


@dataclass(frozen=True)
class InternalTrajectoryTelemetry:
    """Dense internal diagnostics retained without the full internal fields."""

    hamiltonian: FloatArray
    state_finite: BoolArray
    dno_output_finite: BoolArray
    minimum_water_column: FloatArray


@dataclass(frozen=True)
class NonlinearAdjustmentArm:
    """One ramped internal-state arm without projected targets."""

    dt: float
    times: FloatArray
    eta: FloatArray
    xi: FloatArray
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
class ProductionCaseResult:
    """One single-arm production decision and its numerical diagnostics."""

    case_index: int
    dt: float
    telemetry: CaseGL2Telemetry
    decision: QualityDecision
    retained_trajectory: TrajectorySamples | None
    internal_metrics: DenseTrajectoryHealthMetrics | None = None

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


@dataclass(frozen=True)
class NonlinearAdjustmentCaseResult:
    """One case at its own ramped burn-in endpoint."""

    case_index: int
    telemetry: CaseGL2Telemetry
    decision: QualityDecision
    terminal_eta: FloatArray | None
    terminal_xi: FloatArray | None
    minimum_water_column: float | None

    @property
    def accepted(self) -> bool:
        return self.decision.accepted


@dataclass(frozen=True)
class NonlinearAdjustmentExecution:
    """Ordered per-case handoffs from one variable-horizon ramped arm."""

    timing: ArmTiming | None
    cases: tuple[NonlinearAdjustmentCaseResult, ...]

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


class NonlinearAdjustmentArmExecutor(Protocol):
    """Callable used for one fixed-step nonlinear-adjustment arm."""

    def __call__(
        self,
        *,
        eta0: FloatArray,
        xi0: FloatArray,
        depths: FloatArray,
        saved_times: FloatArray,
        contract: ResidualControlledGL2Contract,
        dt: float,
        nonlinear_ramp_times: FloatArray,
        nonlinear_ramp_order: int,
    ) -> NonlinearAdjustmentArm: ...


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
PAPER_BENJAMIN_FEIR_GL2_CONTRACT = ResidualControlledGL2Contract(
    dno_order=4,
    target_dno_order=6,
    maximum_wavenumber=256.0,
    target_maximum_wavenumber=128.0,
    internal_hamiltonian_drift_threshold=1.0e-3,
)
PAPER_JONSWAP_GL2_CONTRACT = ResidualControlledGL2Contract(
    nx=2048,
    target_nx=1024,
    dno_order=4,
    target_dno_order=6,
    maximum_wavenumber=704.0,
    target_maximum_wavenumber=128.0,
    gl2_iteration_cap=5,
    internal_hamiltonian_drift_threshold=1.0e-3,
)
PAPER_TANAKA_GL2_CONTRACT = ResidualControlledGL2Contract(
    maximum_wavenumber=256.0,
    target_maximum_wavenumber=128.0,
    post_step_state_filter="hou_li",
    post_step_maximum_wavenumber=256.0,
    hou_li_coefficient=36.0,
    hou_li_power=36,
)
TANAKA_HOU_LI_GL2_CONTRACT = ResidualControlledGL2Contract(
    post_step_state_filter="hou_li",
)
TANAKA_BUFFERED_HOU_LI_GL2_CONTRACT = ResidualControlledGL2Contract(
    maximum_wavenumber=224.0,
    target_maximum_wavenumber=128.0,
    post_step_state_filter="hou_li",
    post_step_maximum_wavenumber=224.0,
)


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


def _project_fixed_band_to_target_grid(
    field: jax.Array,
    *,
    contract: ResidualControlledGL2Contract,
) -> jax.Array:
    """Project an internal real field and resample its normalized coefficients."""

    source = jnp.asarray(field, dtype=jnp.float64)
    if source.shape[-1] != contract.nx:
        raise ValueError(
            f"internal field must end in {contract.nx} grid points"
        )
    maximum_wavenumber = contract.target_definition.maximum_wavenumber
    maximum_mode = int(
        math.floor(
            maximum_wavenumber * contract.length / (2.0 * math.pi)
            + 1.0e-12
        )
    )
    if maximum_mode >= min(contract.nx, contract.delivered_nx) // 2:
        raise ValueError("delivered band must lie below both grid Nyquists")
    source_coefficients = jnp.fft.rfft(source, axis=-1) / contract.nx
    target_coefficients = jnp.zeros(
        (*source.shape[:-1], contract.delivered_nx // 2 + 1),
        dtype=source_coefficients.dtype,
    )
    target_coefficients = target_coefficients.at[
        ..., : maximum_mode + 1
    ].set(source_coefficients[..., : maximum_mode + 1])
    return jnp.fft.irfft(
        target_coefficients * contract.delivered_nx,
        n=contract.delivered_nx,
        axis=-1,
    )


def _evaluate_saved_target(
    eta: jax.Array,
    xi: jax.Array,
    depths: jax.Array,
    contract: ResidualControlledGL2Contract,
) -> tuple[
    FloatArray | None,
    FloatArray | None,
    FloatArray,
    InternalTrajectoryTelemetry | None,
]:
    """Evaluate the delivered target and compact internal health telemetry."""

    return_projected_state = (
        contract.target_maximum_wavenumber is not None
        or contract.delivered_nx != contract.nx
    )
    evaluate_internal_health = (
        contract.internal_hamiltonian_drift_threshold is not None
    )
    delivered_shape = (*eta.shape[:-1], contract.delivered_nx)
    eta_result = (
        np.empty(delivered_shape, dtype=np.float64)
        if return_projected_state
        else None
    )
    xi_result = (
        np.empty(delivered_shape, dtype=np.float64)
        if return_projected_state
        else None
    )
    q_result = np.empty(delivered_shape, dtype=np.float64)
    telemetry_shape = eta.shape[:2]
    internal_hamiltonian = (
        np.empty(telemetry_shape, dtype=np.float64)
        if evaluate_internal_health
        else None
    )
    internal_state_finite = (
        np.empty(telemetry_shape, dtype=np.bool_)
        if evaluate_internal_health
        else None
    )
    internal_dno_output_finite = (
        np.empty(telemetry_shape, dtype=np.bool_)
        if evaluate_internal_health
        else None
    )
    minimum_water_column = (
        np.empty(telemetry_shape, dtype=np.float64)
        if evaluate_internal_health
        else None
    )
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
        target_eta_chunk = eta_chunk
        target_xi_chunk = xi_chunk
        if contract.delivered_nx != contract.nx:
            target_eta_chunk = _project_fixed_band_to_target_grid(
                eta_chunk,
                contract=contract,
            )
            target_xi_chunk = _project_fixed_band_to_target_grid(
                xi_chunk,
                contract=contract,
            )
        eta_ref, xi_ref, q_ref = evaluate_discrete_dno_target(
            target_eta_chunk,
            target_xi_chunk,
            depths[:, None],
            definition=contract.target_definition,
        )
        if eta_result is not None and xi_result is not None:
            eta_result[start:stop] = np.asarray(
                jax.device_get(eta_ref[:count]), dtype=np.float64
            )
            xi_result[start:stop] = np.asarray(
                jax.device_get(xi_ref[:count]), dtype=np.float64
            )
        q_result[start:stop] = np.asarray(
            jax.device_get(q_ref[:count]), dtype=np.float64
        )
        if evaluate_internal_health:
            _, _, internal_q = evaluate_discrete_dno_target(
                eta_chunk,
                xi_chunk,
                depths[:, None],
                definition=contract.internal_definition,
            )
            state_finite = jnp.all(
                jnp.isfinite(eta_chunk) & jnp.isfinite(xi_chunk),
                axis=-1,
            )
            dno_output_finite = jnp.all(jnp.isfinite(internal_q), axis=-1)
            water_column = jnp.min(
                eta_chunk + depths[None, :, None],
                axis=-1,
            )
            hamiltonian = (
                0.5
                * (contract.length / contract.nx)
                * jnp.sum(
                    xi_chunk * internal_q
                    + contract.gravity * eta_chunk**2,
                    axis=-1,
                )
            )
            (
                hamiltonian_host,
                state_finite_host,
                dno_output_finite_host,
                water_column_host,
            ) = jax.device_get(
                (
                    hamiltonian[:count],
                    state_finite[:count],
                    dno_output_finite[:count],
                    water_column[:count],
                )
            )
            assert internal_hamiltonian is not None
            assert internal_state_finite is not None
            assert internal_dno_output_finite is not None
            assert minimum_water_column is not None
            internal_hamiltonian[start:stop] = np.asarray(
                hamiltonian_host,
                dtype=np.float64,
            )
            internal_state_finite[start:stop] = np.asarray(
                state_finite_host,
                dtype=np.bool_,
            )
            internal_dno_output_finite[start:stop] = np.asarray(
                dno_output_finite_host,
                dtype=np.bool_,
            )
            minimum_water_column[start:stop] = np.asarray(
                water_column_host,
                dtype=np.float64,
            )

    internal_telemetry = None
    if evaluate_internal_health:
        assert internal_hamiltonian is not None
        assert internal_state_finite is not None
        assert internal_dno_output_finite is not None
        assert minimum_water_column is not None
        internal_telemetry = InternalTrajectoryTelemetry(
            hamiltonian=internal_hamiltonian,
            state_finite=internal_state_finite,
            dno_output_finite=internal_dno_output_finite,
            minimum_water_column=minimum_water_column,
        )
    return eta_result, xi_result, q_result, internal_telemetry


def _solver_parameters(
    contract: ResidualControlledGL2Contract,
    depths: jax.Array,
    *,
    nonlinear_ramp_times: FloatArray | None,
    nonlinear_ramp_order: int,
) -> SolverParams:
    """Construct one batched solver parameter record."""

    _, wavenumbers = build_grid(contract.nx, contract.length)
    k = jnp.asarray(wavenumbers, dtype=jnp.float64)
    depth_column = depths[:, None]
    return SolverParams(
        nx=contract.nx,
        length=contract.length,
        depth=depth_column,
        gravity=contract.gravity,
        dno_order=contract.dno_order,
        pad_factor=contract.pad_factor,
        filter_fraction=contract.filter_fraction,
        k=k,
        g0=make_linear_dno_symbol(k, depth_column),
        gl2_post_step_houli=contract.post_step_state_filter == "hou_li",
        gl2_post_step_houli_a=contract.hou_li_coefficient,
        gl2_post_step_houli_m=0.5 * contract.hou_li_power,
        gl2_post_step_filter_fraction=contract.post_step_filter_fraction,
        nonlinear_ramp_time=(
            None
            if nonlinear_ramp_times is None
            else jnp.asarray(nonlinear_ramp_times, dtype=jnp.float64)
        ),
        nonlinear_ramp_order=nonlinear_ramp_order,
    )


def run_residual_controlled_arm(
    *,
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    saved_times: FloatArray,
    contract: ResidualControlledGL2Contract,
    dt: float,
    nonlinear_ramp_times: FloatArray | None = None,
    nonlinear_ramp_order: int = 4,
) -> ResidualControlledArm:
    """Run one actual float64 GL2 arm and evaluate the common target."""

    if not jax.config.x64_enabled:
        raise RuntimeError("residual-controlled paper-dataset GL2 requires float64")
    eta, xi, depth, times = _validated_inputs(
        eta0, xi0, depths, saved_times, contract
    )
    ramp_times = None
    if nonlinear_ramp_times is not None:
        ramp_times = np.asarray(nonlinear_ramp_times, dtype=np.float64)
        if ramp_times.shape != depth.shape:
            raise ValueError(
                f"nonlinear_ramp_times must have shape {depth.shape}"
            )
        if not np.isfinite(ramp_times).all() or np.any(ramp_times <= 0.0):
            raise ValueError("nonlinear_ramp_times must be finite and positive")
        if nonlinear_ramp_order < 1:
            raise ValueError("nonlinear_ramp_order must be positive")
    substeps = _integer_ratio(
        contract.saved_dt,
        dt,
        "saved_dt",
        "arm dt",
    )
    depth_device = jnp.asarray(depth, dtype=jnp.float64)
    parameters = _solver_parameters(
        contract,
        depth_device,
        nonlinear_ramp_times=ramp_times,
        nonlinear_ramp_order=nonlinear_ramp_order,
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
    delivered_eta, delivered_xi, q_ref, internal_telemetry = (
        _evaluate_saved_target(
        payload["eta"],
        payload["xi"],
        depth_device,
        contract,
        )
    )
    target_seconds = perf_counter() - started

    started = perf_counter()
    host = {
        name: np.asarray(jax.device_get(value))
        for name, value in payload.items()
        if delivered_eta is None or name not in ("eta", "xi")
    }
    host_transfer_seconds = perf_counter() - started
    arm = ResidualControlledArm(
        dt=float(dt),
        times=np.asarray(host["times"], dtype=np.float64),
        eta=(
            np.asarray(host["eta"], dtype=np.float64)
            if delivered_eta is None
            else delivered_eta
        ),
        xi=(
            np.asarray(host["xi"], dtype=np.float64)
            if delivered_xi is None
            else delivered_xi
        ),
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
        internal_telemetry=internal_telemetry,
        timing=ArmTiming(
            rollout_seconds=rollout_seconds,
            target_seconds=target_seconds,
            host_transfer_seconds=host_transfer_seconds,
        ),
    )
    _validate_arm(arm, eta.shape[0], contract.delivered_nx, times, dt)
    return arm


def run_nonlinear_adjustment_arm(
    *,
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    saved_times: FloatArray,
    contract: ResidualControlledGL2Contract,
    dt: float,
    nonlinear_ramp_times: FloatArray,
    nonlinear_ramp_order: int = 4,
) -> NonlinearAdjustmentArm:
    """Run a ramped internal arm and return no projected target arrays."""

    if not jax.config.x64_enabled:
        raise RuntimeError("nonlinear adjustment requires JAX float64")
    eta, xi, depth, times = _validated_inputs(
        eta0,
        xi0,
        depths,
        saved_times,
        contract,
    )
    ramp_times = np.asarray(nonlinear_ramp_times, dtype=np.float64)
    if ramp_times.shape != depth.shape:
        raise ValueError(
            f"nonlinear_ramp_times must have shape {depth.shape}"
        )
    if not np.isfinite(ramp_times).all() or np.any(ramp_times <= 0.0):
        raise ValueError("nonlinear_ramp_times must be finite and positive")
    if nonlinear_ramp_order < 1:
        raise ValueError("nonlinear_ramp_order must be positive")

    substeps = _integer_ratio(
        contract.saved_dt,
        dt,
        "saved_dt",
        "adjustment dt",
    )
    depth_device = jnp.asarray(depth, dtype=jnp.float64)
    parameters = _solver_parameters(
        contract,
        depth_device,
        nonlinear_ramp_times=ramp_times,
        nonlinear_ramp_order=nonlinear_ramp_order,
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
    host = {
        name: np.asarray(jax.device_get(value))
        for name, value in payload.items()
    }
    host_transfer_seconds = perf_counter() - started
    arm = NonlinearAdjustmentArm(
        dt=float(dt),
        times=np.asarray(host["times"], dtype=np.float64),
        eta=np.asarray(host["eta"], dtype=np.float64),
        xi=np.asarray(host["xi"], dtype=np.float64),
        gl2_stage_residual=np.asarray(
            host["gl2_stage_residual"],
            dtype=np.float64,
        ),
        gl2_iterations=np.asarray(host["gl2_iterations"], dtype=np.int32),
        gl2_converged=np.asarray(host["gl2_converged"], dtype=np.bool_),
        gl2_stage_finite=np.asarray(
            host["gl2_stage_finite"],
            dtype=np.bool_,
        ),
        gl2_state_finite=np.asarray(
            host["gl2_state_finite"],
            dtype=np.bool_,
        ),
        gl2_hit_iteration_cap=np.asarray(
            host["gl2_hit_iteration_cap"],
            dtype=np.bool_,
        ),
        timing=ArmTiming(
            rollout_seconds=rollout_seconds,
            target_seconds=0.0,
            host_transfer_seconds=host_transfer_seconds,
        ),
    )
    _validate_nonlinear_adjustment_arm(
        arm,
        eta.shape[0],
        contract.nx,
        times,
        dt,
    )
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
    internal = arm.internal_telemetry
    if internal is not None:
        internal_shape = (saved_times.size, batch_size)
        for name in (
            "hamiltonian",
            "state_finite",
            "dno_output_finite",
            "minimum_water_column",
        ):
            if np.asarray(getattr(internal, name)).shape != internal_shape:
                raise ValueError(
                    f"internal {name} must have shape {internal_shape}"
                )


def _validate_nonlinear_adjustment_arm(
    arm: NonlinearAdjustmentArm,
    batch_size: int,
    nx: int,
    saved_times: FloatArray,
    expected_dt: float,
) -> None:
    if not math.isclose(
        arm.dt,
        expected_dt,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("adjustment arm dt differs from the requested step")
    if not np.array_equal(np.asarray(arm.times), saved_times):
        raise ValueError(
            "adjustment arm times differ from the requested saved times"
        )
    field_shape = (saved_times.size, batch_size, nx)
    for name in ("eta", "xi"):
        if np.asarray(getattr(arm, name)).shape != field_shape:
            raise ValueError(
                f"adjustment arm {name} must have shape {field_shape}"
            )
    step_shape = np.asarray(arm.gl2_stage_residual).shape
    if len(step_shape) != 2 or step_shape[1] != batch_size:
        raise ValueError(
            "adjustment GL2 telemetry must have shape (step, batch)"
        )
    for name in (
        "gl2_iterations",
        "gl2_converged",
        "gl2_stage_finite",
        "gl2_state_finite",
        "gl2_hit_iteration_cap",
    ):
        if np.asarray(getattr(arm, name)).shape != step_shape:
            raise ValueError(
                "all adjustment GL2 telemetry arrays must have one shape"
            )


def _case_telemetry(
    arm: ResidualControlledArm | NonlinearAdjustmentArm,
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
) -> TrajectorySamples:
    return TrajectorySamples(
        times=np.asarray(arm.times, dtype=np.float64),
        eta=np.asarray(arm.eta[:, case_index], dtype=np.float64),
        xi=np.asarray(arm.xi[:, case_index], dtype=np.float64),
        gxi=np.asarray(arm.q_ref[:, case_index], dtype=np.float64),
        reached_final_time=bool(arm.complete[case_index]),
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
    decision = evaluate_production_trajectory(
        trajectory,
        depth=depth,
    )
    internal_metrics = None
    threshold = contract.internal_hamiltonian_drift_threshold
    if threshold is not None:
        internal = arm.internal_telemetry
        if internal is None:
            raise RuntimeError(
                "the production contract requires internal telemetry"
            )
        internal_metrics, internal_decision = (
            evaluate_dense_trajectory_health(
                internal.hamiltonian[:, case_index],
                internal.state_finite[:, case_index],
                internal.dno_output_finite[:, case_index],
                internal.minimum_water_column[:, case_index],
                hamiltonian_drift_threshold=threshold,
            )
        )
        required = decision.required | internal_decision.required
        evaluated = decision.evaluated | internal_decision.evaluated
        failed = decision.failed | internal_decision.failed
        if not internal_decision.accepted:
            failed |= QualityReason.INCOMPLETE_TRAJECTORY
        decision = QualityDecision(
            scope=decision.scope,
            required=required,
            evaluated=evaluated,
            failed=failed,
        )
    return ProductionCaseResult(
        case_index=(
            case_index if result_case_index is None else result_case_index
        ),
        dt=arm.dt,
        telemetry=telemetry,
        decision=decision,
        retained_trajectory=trajectory if decision.accepted else None,
        internal_metrics=internal_metrics,
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
        contract.delivered_nx,
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


def _single_case_prefix_adjustment_arm(
    arm: NonlinearAdjustmentArm,
    case_index: int,
    saved_times: FloatArray,
    contract: ResidualControlledGL2Contract,
) -> NonlinearAdjustmentArm:
    """Restrict a ramped arm to one case and its declared endpoint."""

    if not 0 <= case_index < arm.eta.shape[1]:
        raise IndexError("adjustment arm case index is out of range")
    saved_count = saved_times.size
    if saved_count > arm.times.size or not np.array_equal(
        saved_times,
        np.asarray(arm.times, dtype=np.float64)[:saved_count],
    ):
        raise ValueError(
            "adjustment saved times must be a prefix of the arm times"
        )
    steps_per_interval = _integer_ratio(
        contract.saved_dt,
        arm.dt,
        "saved_dt",
        "adjustment dt",
    )
    full_step_count = (arm.times.size - 1) * steps_per_interval
    if arm.gl2_stage_residual.shape[0] != full_step_count:
        raise ValueError(
            "adjustment batching requires one telemetry row per substep"
        )
    step_count = (saved_count - 1) * steps_per_interval
    prefix = NonlinearAdjustmentArm(
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
        gl2_stage_residual=np.asarray(
            arm.gl2_stage_residual[
                :step_count,
                case_index : case_index + 1,
            ],
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
            arm.gl2_stage_finite[
                :step_count,
                case_index : case_index + 1,
            ],
            dtype=np.bool_,
        ),
        gl2_state_finite=np.asarray(
            arm.gl2_state_finite[
                :step_count,
                case_index : case_index + 1,
            ],
            dtype=np.bool_,
        ),
        gl2_hit_iteration_cap=np.asarray(
            arm.gl2_hit_iteration_cap[
                :step_count,
                case_index : case_index + 1,
            ],
            dtype=np.bool_,
        ),
        timing=arm.timing,
    )
    _validate_nonlinear_adjustment_arm(
        prefix,
        1,
        contract.nx,
        saved_times,
        arm.dt,
    )
    return prefix


def _evaluate_nonlinear_adjustment_case(
    arm: NonlinearAdjustmentArm,
    *,
    case_index: int,
    depth: float,
    contract: ResidualControlledGL2Contract,
) -> NonlinearAdjustmentCaseResult:
    """Evaluate only the numerical existence of one ramped handoff."""

    telemetry = _case_telemetry(arm, 0, contract)
    eta = np.asarray(arm.eta[:, 0], dtype=np.float64)
    xi = np.asarray(arm.xi[:, 0], dtype=np.float64)
    evaluated = (
        QualityReason.INCOMPLETE_TRAJECTORY
        | QualityReason.GL2_STAGE_RESIDUAL
        | QualityReason.NONFINITE_STATE
    )
    failed = QualityReason.NONE
    if not telemetry.all_stages_solved:
        failed |= QualityReason.GL2_STAGE_RESIDUAL
    state_finite = bool(np.isfinite(eta).all() and np.isfinite(xi).all())
    minimum_water_column = None
    if not state_finite:
        failed |= QualityReason.NONFINITE_STATE
    else:
        evaluated |= QualityReason.BOTTOM_CLEARANCE
        minimum_water_column = float(np.min(depth + eta))
        if minimum_water_column <= 0.0:
            failed |= QualityReason.BOTTOM_CLEARANCE
    if failed:
        failed |= QualityReason.INCOMPLETE_TRAJECTORY
    decision = QualityDecision(
        scope=QualityScope.TRAJECTORY,
        required=QualityReason.INCOMPLETE_TRAJECTORY,
        evaluated=evaluated,
        failed=failed,
    )
    return NonlinearAdjustmentCaseResult(
        case_index=case_index,
        telemetry=telemetry,
        decision=decision,
        terminal_eta=(
            np.asarray(eta[-1], dtype=np.float64)
            if decision.accepted
            else None
        ),
        terminal_xi=(
            np.asarray(xi[-1], dtype=np.float64)
            if decision.accepted
            else None
        ),
        minimum_water_column=minimum_water_column,
    )


def execute_variable_horizon_nonlinear_adjustment(
    eta0: FloatArray,
    xi0: FloatArray,
    depths: FloatArray,
    case_saved_times: tuple[FloatArray, ...],
    *,
    nonlinear_ramp_times: FloatArray,
    nonlinear_ramp_order: int,
    contract: ResidualControlledGL2Contract,
    arm_executor: NonlinearAdjustmentArmExecutor = (
        run_nonlinear_adjustment_arm
    ),
) -> NonlinearAdjustmentExecution:
    """Run one ramped arm and return each case's full-band endpoint."""

    eta, xi, depth, grids, longest = _validated_variable_horizon_inputs(
        eta0,
        xi0,
        depths,
        case_saved_times,
        contract,
    )
    ramp_times = np.asarray(nonlinear_ramp_times, dtype=np.float64)
    if ramp_times.shape != depth.shape:
        raise ValueError(
            f"nonlinear_ramp_times must have shape {depth.shape}"
        )
    arm = arm_executor(
        eta0=eta,
        xi0=xi,
        depths=depth,
        saved_times=longest,
        contract=contract,
        dt=contract.production_dt,
        nonlinear_ramp_times=ramp_times,
        nonlinear_ramp_order=nonlinear_ramp_order,
    )
    _validate_nonlinear_adjustment_arm(
        arm,
        eta.shape[0],
        contract.nx,
        longest,
        contract.production_dt,
    )
    case_arms = tuple(
        _single_case_prefix_adjustment_arm(
            arm,
            case_index,
            grid,
            contract,
        )
        for case_index, grid in enumerate(grids)
    )
    return NonlinearAdjustmentExecution(
        timing=arm.timing,
        cases=tuple(
            _evaluate_nonlinear_adjustment_case(
                case_arm,
                case_index=case_index,
                depth=float(depth[case_index]),
                contract=contract,
            )
            for case_index, case_arm in enumerate(case_arms)
        ),
    )


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
    internal = arm.internal_telemetry
    internal_prefix = (
        None
        if internal is None
        else InternalTrajectoryTelemetry(
            hamiltonian=np.asarray(
                internal.hamiltonian[
                    :saved_count,
                    case_index : case_index + 1,
                ],
                dtype=np.float64,
            ),
            state_finite=np.asarray(
                internal.state_finite[
                    :saved_count,
                    case_index : case_index + 1,
                ],
                dtype=np.bool_,
            ),
            dno_output_finite=np.asarray(
                internal.dno_output_finite[
                    :saved_count,
                    case_index : case_index + 1,
                ],
                dtype=np.bool_,
            ),
            minimum_water_column=np.asarray(
                internal.minimum_water_column[
                    :saved_count,
                    case_index : case_index + 1,
                ],
                dtype=np.float64,
            ),
        )
    )
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
        internal_telemetry=internal_prefix,
    )
    _validate_arm(
        prefix,
        1,
        contract.delivered_nx,
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
        contract.delivered_nx,
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
