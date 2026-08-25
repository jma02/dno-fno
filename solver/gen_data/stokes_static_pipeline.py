"""Static Stokes adapter for the common paper-dataset transaction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from solver.reference_solutions.stokes_wave import stokes_eta_xi_at_phase
from solver.gen_data.pipeline.archive import BatchPaths, ensure_proposal
from solver.gen_data.pipeline.quality import (
    QualityDecision,
    QualityReason,
    QualityScope,
)
from solver.gen_data.pipeline.reference import (
    PAPER_DNO_TARGET,
    DiscreteDnoTarget,
    evaluate_discrete_dno_target,
    project_fixed_band,
)
from solver.gen_data.pipeline.writer import (
    AcceptedCaseRows,
    CaseOutcome,
    batch_paths_for_assignments,
    build_proposal_arrays,
    commit_case_outcomes,
)
from solver.gen_data.stokes_sampling import (
    PAPER_GRAVITY,
    STOKES_SAMPLE_CELLS,
    StokesSample,
    stokes_support_violations,
)
from solver.solvers.dno_series_jax import build_grid


Array: TypeAlias = jax.Array | NDArray[np.floating]
JsonScalar: TypeAlias = str | int | float | bool | None
ContractRole: TypeAlias = Literal[
    "paper_dataset",
    "reduced_wiring_evidence_only",
]

STATIC_STOKES_REQUIRED_CHECKS = (
    QualityReason.OUTSIDE_SUPPORT
    | QualityReason.NONFINITE_STATE
    | QualityReason.BOTTOM_CLEARANCE
    | QualityReason.NONFINITE_TARGET
)
STOKES_CELL_CODES = {
    cell_id: index for index, cell_id in enumerate(STOKES_SAMPLE_CELLS)
}


@dataclass(frozen=True)
class StaticStokesContract:
    """Spatial target and explicit role of a static Stokes execution."""

    target: DiscreteDnoTarget = PAPER_DNO_TARGET
    gravity: float = PAPER_GRAVITY
    role: ContractRole = "paper_dataset"

    def __post_init__(self) -> None:
        if not math.isfinite(self.gravity) or self.gravity <= 0.0:
            raise ValueError("gravity must be positive and finite")
        if self.role == "paper_dataset" and (
            self.target != PAPER_DNO_TARGET
            or not math.isclose(
                self.gravity,
                PAPER_GRAVITY,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        ):
            raise ValueError(
                "a reduced target must be labeled 'reduced_wiring_evidence_only'"
            )
        if self.role not in (
            "paper_dataset",
            "reduced_wiring_evidence_only",
        ):
            raise ValueError(f"unknown static Stokes contract role: {self.role}")

    @classmethod
    def reduced_wiring_evidence(
        cls,
        target: DiscreteDnoTarget,
        *,
        gravity: float = PAPER_GRAVITY,
    ) -> StaticStokesContract:
        """Construct an explicitly non-production contract for CPU wiring tests."""

        return cls(
            target=target,
            gravity=gravity,
            role="reduced_wiring_evidence_only",
        )

    def to_json_record(self) -> dict[str, object]:
        """Return the complete target contract for proposal metadata."""

        return {
            "role": self.role,
            "nx": self.target.nx,
            "length": self.target.length,
            "gravity": self.gravity,
            "dno_order": self.target.dno_order,
            "pad_factor": self.target.pad_factor,
            "maximum_wavenumber": self.target.maximum_wavenumber,
            "dtype": "float64",
        }


PAPER_STATIC_STOKES_CONTRACT = StaticStokesContract()


class StaticStokesStateConstructor(Protocol):
    """Callable that constructs one unprojected Stokes state."""

    def __call__(
        self,
        sample: StokesSample,
        contract: StaticStokesContract,
    ) -> tuple[Array, Array]: ...


class StaticDnoEvaluator(Protocol):
    """Callable with the frozen-target evaluator interface."""

    def __call__(
        self,
        eta: jax.Array,
        xi: jax.Array,
        depth: float | jax.Array,
        *,
        definition: DiscreteDnoTarget,
    ) -> tuple[jax.Array, jax.Array, jax.Array]: ...


@dataclass(frozen=True)
class StaticStokesBatchResult:
    """Paths and ordered decisions produced by one committed static batch."""

    paths: BatchPaths
    outcomes: tuple[CaseOutcome, ...]


def construct_stokes_state(
    sample: StokesSample,
    contract: StaticStokesContract,
) -> tuple[jax.Array, jax.Array]:
    """Construct one fifth-order state with the repository's public formula."""

    if not math.isclose(
        sample.domain_length,
        contract.target.length,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("sample and target domain lengths differ")
    if not math.isclose(
        sample.gravity,
        contract.gravity,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("sample and target gravities differ")

    x, _ = build_grid(contract.target.nx, contract.target.length)
    return stokes_eta_xi_at_phase(
        x=jnp.asarray(x, dtype=jnp.float64),
        phase=sample.phase,
        n0=sample.carrier_mode,
        a0=sample.amplitude,
        length=sample.domain_length,
        depth=sample.depth,
        gravity=sample.gravity,
        ichoi=1 if sample.branch == "finite" else 0,
    )


def _project_static_inputs(
    eta: Array,
    xi: Array,
    *,
    definition: DiscreteDnoTarget,
) -> tuple[jax.Array, jax.Array]:
    """Apply the common fixed band and remove the constant-potential gauge."""

    eta_array = jnp.asarray(eta, dtype=jnp.float64)
    xi_array = jnp.asarray(xi, dtype=jnp.float64)
    expected_shape = (definition.nx,)
    if eta_array.shape != expected_shape or xi_array.shape != expected_shape:
        raise ValueError(
            "static Stokes constructors must return shape "
            f"{expected_shape}, got {eta_array.shape} and {xi_array.shape}"
        )

    _, wavenumbers = build_grid(definition.nx, definition.length)
    wavenumber_array = jnp.asarray(wavenumbers, dtype=jnp.float64)
    eta_input = project_fixed_band(
        eta_array,
        wavenumber_array,
        maximum_wavenumber=definition.maximum_wavenumber,
    )
    xi_input = project_fixed_band(
        xi_array - jnp.mean(xi_array),
        wavenumber_array,
        maximum_wavenumber=definition.maximum_wavenumber,
        remove_mean=True,
    )
    return eta_input, xi_input


def _finite_or_none(value: float) -> float | None:
    """Return finite diagnostics and encode unavailable values as JSON null."""

    return value if math.isfinite(value) else None


def _maximum_absolute_or_none(field: NDArray[np.float64]) -> float | None:
    """Return the maximum absolute field value when the field is finite."""

    if not np.isfinite(field).all():
        return None
    return float(np.max(np.abs(field)))


def _decision(
    *,
    evaluated: QualityReason,
    failed: QualityReason,
) -> QualityDecision:
    """Build the common sample-level decision for one static state."""

    return QualityDecision(
        scope=QualityScope.SAMPLE,
        required=STATIC_STOKES_REQUIRED_CHECKS,
        evaluated=evaluated,
        failed=failed,
    )


def evaluate_static_stokes_sample(
    sample: StokesSample,
    *,
    contract: StaticStokesContract = PAPER_STATIC_STOKES_CONTRACT,
    state_constructor: StaticStokesStateConstructor = construct_stokes_state,
    target_evaluator: StaticDnoEvaluator = evaluate_discrete_dno_target,
) -> CaseOutcome:
    """Construct, validate, and label one static Stokes case."""

    support_violations = stokes_support_violations(sample)
    evaluated = QualityReason.OUTSIDE_SUPPORT
    failed = QualityReason.OUTSIDE_SUPPORT if support_violations else QualityReason.NONE
    metrics: dict[str, JsonScalar] = {
        "contract_role": contract.role,
        "support_violation_count": len(support_violations),
        "support_violations": "; ".join(support_violations),
        "support_resampling_count": sample.support_resampling_count,
        "ursell_upper_bound": sample.ursell_upper_bound,
        "state_finite": None,
        "minimum_water_column": None,
        "target_finite": None,
        "maximum_absolute_eta": None,
        "maximum_absolute_xi": None,
        "maximum_absolute_q_ref": None,
        "xi_input_mean": None,
        "q_ref_mean": None,
        "numerical_error": "",
    }
    if support_violations:
        raise RuntimeError(
            "Stokes sampler returned a case outside its declared support: "
            + "; ".join(support_violations)
        )

    with jax.enable_x64():
        eta_raw, xi_raw = state_constructor(sample, contract)
        eta_input, xi_input = _project_static_inputs(
            eta_raw,
            xi_raw,
            definition=contract.target,
        )
    eta_host = np.asarray(jax.device_get(eta_input), dtype=np.float64)
    xi_host = np.asarray(jax.device_get(xi_input), dtype=np.float64)

    evaluated |= QualityReason.NONFINITE_STATE
    state_finite = bool(np.isfinite(eta_host).all() and np.isfinite(xi_host).all())
    metrics["state_finite"] = state_finite
    metrics["maximum_absolute_eta"] = _maximum_absolute_or_none(eta_host)
    metrics["maximum_absolute_xi"] = _maximum_absolute_or_none(xi_host)
    metrics["xi_input_mean"] = _finite_or_none(float(np.mean(xi_host)))
    if not state_finite:
        failed |= QualityReason.NONFINITE_STATE
        return CaseOutcome(
            decision=_decision(evaluated=evaluated, failed=failed),
            rows=None,
            metrics=metrics,
        )

    evaluated |= QualityReason.BOTTOM_CLEARANCE
    minimum_water_column = float(np.min(sample.depth + eta_host))
    metrics["minimum_water_column"] = _finite_or_none(minimum_water_column)
    if not math.isfinite(minimum_water_column) or minimum_water_column <= 0.0:
        failed |= QualityReason.BOTTOM_CLEARANCE
        return CaseOutcome(
            decision=_decision(evaluated=evaluated, failed=failed),
            rows=None,
            metrics=metrics,
        )

    with jax.enable_x64():
        target_eta, target_xi, q_ref = target_evaluator(
            eta_input,
            xi_input,
            sample.depth,
            definition=contract.target,
        )
    target_eta_host = np.asarray(
        jax.device_get(target_eta),
        dtype=np.float64,
    )
    target_xi_host = np.asarray(
        jax.device_get(target_xi),
        dtype=np.float64,
    )
    q_ref_host = np.asarray(jax.device_get(q_ref), dtype=np.float64)
    expected_shape = (contract.target.nx,)
    if (
        target_eta_host.shape != expected_shape
        or target_xi_host.shape != expected_shape
        or q_ref_host.shape != expected_shape
    ):
        raise ValueError("target evaluator returned an unexpected shape")

    evaluated |= QualityReason.NONFINITE_TARGET
    delivered_state_finite = bool(
        np.isfinite(target_eta_host).all() and np.isfinite(target_xi_host).all()
    )
    target_finite = bool(np.isfinite(q_ref_host).all())
    metrics["state_finite"] = delivered_state_finite
    metrics["target_finite"] = target_finite
    metrics["maximum_absolute_eta"] = _maximum_absolute_or_none(target_eta_host)
    metrics["maximum_absolute_xi"] = _maximum_absolute_or_none(target_xi_host)
    metrics["maximum_absolute_q_ref"] = _maximum_absolute_or_none(q_ref_host)
    metrics["xi_input_mean"] = _finite_or_none(float(np.mean(target_xi_host)))
    metrics["q_ref_mean"] = _finite_or_none(float(np.mean(q_ref_host)))
    if not delivered_state_finite:
        failed |= QualityReason.NONFINITE_STATE
    if not target_finite:
        failed |= QualityReason.NONFINITE_TARGET

    decision = _decision(evaluated=evaluated, failed=failed)
    rows = (
        AcceptedCaseRows(
            eta=target_eta_host[None, :],
            xi=target_xi_host[None, :],
            gxi=q_ref_host[None, :],
            depth=sample.depth,
            time=np.asarray([0.0], dtype=np.float64),
            selected_dense_index=np.asarray([0], dtype=np.int32),
        )
        if decision.accepted
        else None
    )
    return CaseOutcome(decision=decision, rows=rows, metrics=metrics)


def write_static_stokes_batch(
    root: Path,
    samples: Sequence[StokesSample],
    *,
    batch_id: int,
    config_fingerprint: str,
    contract: StaticStokesContract = PAPER_STATIC_STOKES_CONTRACT,
    metadata: Mapping[str, object] | None = None,
    state_constructor: StaticStokesStateConstructor = construct_stokes_state,
    target_evaluator: StaticDnoEvaluator = evaluate_discrete_dno_target,
) -> StaticStokesBatchResult:
    """Propose, execute, and commit one complete static Stokes batch."""

    if not samples:
        raise ValueError("samples must not be empty")
    for sample in samples:
        if not math.isclose(
            sample.domain_length,
            contract.target.length,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError("sample and target domain lengths differ")
        if not math.isclose(
            sample.gravity,
            contract.gravity,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError("sample and target gravities differ")

    assignments = tuple(sample.assignment for sample in samples)
    contract_record = contract.to_json_record()
    proposal_metadata = {
        "family": "stokes",
        "case_kind": "static",
        "contract": contract_record,
        "retained_times": [0.0],
        "selected_dense_indices": [0],
        "additional_metadata": dict(metadata or {}),
    }
    proposal_arrays = build_proposal_arrays(
        assignments,
        tuple(sample.to_json_record() for sample in samples),
        cell_codes=STOKES_CELL_CODES,
        batch_id=batch_id,
        config_fingerprint=config_fingerprint,
        metadata=proposal_metadata,
    )
    paths = batch_paths_for_assignments(
        root,
        assignments,
        family_name="stokes",
        batch_id=batch_id,
    )

    # Write the proposal before constructing states or evaluating targets.
    ensure_proposal(paths, proposal_arrays)
    outcomes = tuple(
        evaluate_static_stokes_sample(
            sample,
            contract=contract,
            state_constructor=state_constructor,
            target_evaluator=target_evaluator,
        )
        for sample in samples
    )
    commit_case_outcomes(
        paths,
        proposal_arrays,
        outcomes,
        metadata={
            "family": "stokes",
            "case_kind": "static",
            "contract": contract_record,
            "attempted_cases": len(outcomes),
            "accepted_cases": sum(outcome.decision.accepted for outcome in outcomes),
            "additional_metadata": dict(metadata or {}),
        },
    )
    return StaticStokesBatchResult(paths=paths, outcomes=outcomes)
