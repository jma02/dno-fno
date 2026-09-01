"""Generate and validate one batch of static Stokes simulations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, TypeAlias

from solver.gen_data.pipeline.batch_storage import batch_path
from solver.gen_data.pipeline.simulation_allocation import (
    AttemptAssignment,
    PhysicalFamilyId,
)
from solver.gen_data.pipeline.simulation_checks import (
    SimulationCheckResult,
    SimulationCheck,
)
from solver.gen_data.pipeline.dataset_generation import (
    BatchExecutor,
    DatasetChunkConfig,
)
from solver.gen_data.pipeline.writer import (
    SimulationOutcome,
    build_batch_plan,
    commit_simulation_outcomes,
)
from solver.gen_data.stokes_sampling import (
    DEFAULT_MAXIMUM_URSELL_REDRAWS,
    STOKES_PARAMETER_GROUPS,
    StokesSample,
    UrsellRedrawLimitReached,
    sample_stokes_simulation,
)
from solver.gen_data.stokes_static_pipeline import (
    STATIC_STOKES_REQUIRED_CHECKS,
    StaticDnoEvaluator,
    StaticStokesContract,
    StaticStokesStateConstructor,
    construct_stokes_state,
    evaluate_static_stokes_sample,
)
from solver.gen_data.pipeline.dno_target import compute_dno_target


JsonRecord: TypeAlias = Mapping[str, object]
_STOKES_PARAMETER_GROUP_IDS = frozenset(STOKES_PARAMETER_GROUPS)


class StokesSampler(Protocol):
    """Callable with the complete deterministic Stokes sampler interface."""

    def __call__(
        self,
        assignment: AttemptAssignment,
        *,
        domain_length: float,
        gravity: float,
        maximum_ursell_redraws: int,
    ) -> StokesSample: ...


def _ursell_redraw_failure_outcome(
    contract: StaticStokesContract,
) -> SimulationOutcome:
    reason = SimulationCheck.OUTSIDE_SUPPORT
    return SimulationOutcome(
        decision=SimulationCheckResult(
            required=STATIC_STOKES_REQUIRED_CHECKS,
            evaluated=reason,
            failed=reason,
        ),
        rows=None,
        metrics={
            "contract_role": contract.role,
            "support_violation_count": 1,
            "support_violations": "finite-depth Ursell redraw limit reached",
            "state_finite": None,
            "minimum_water_column": None,
            "target_finite": None,
            "numerical_error": "",
        },
    )


def _validate_static_stokes_executor(
    chunk_config: DatasetChunkConfig,
    contract: StaticStokesContract,
    maximum_ursell_redraws: int,
    sampler: StokesSampler,
    state_constructor: StaticStokesStateConstructor,
    target_evaluator: StaticDnoEvaluator,
) -> None:
    if chunk_config.family_name != "stokes":
        raise ValueError("static Stokes generation must use family_name='stokes'")
    if chunk_config.family_id is not PhysicalFamilyId.STOKES:
        raise ValueError("static Stokes generation requires the Stokes family ID")
    unknown_parameter_groups = {
        target.parameter_group_id for target in chunk_config.simulation_targets
    }.difference(_STOKES_PARAMETER_GROUP_IDS)
    if unknown_parameter_groups:
        raise ValueError(
            f"unknown Stokes parameter groups: {sorted(unknown_parameter_groups)}"
        )
    if (
        isinstance(maximum_ursell_redraws, bool)
        or not isinstance(maximum_ursell_redraws, int)
        or maximum_ursell_redraws < 0
    ):
        raise ValueError("maximum_ursell_redraws must be a nonnegative integer")

    configured_contract = chunk_config.configuration.get("contract")
    if (
        not isinstance(configured_contract, Mapping)
        or dict(configured_contract) != contract.to_json_record()
    ):
        raise ValueError(
            "chunk configuration contract differs from the supplied static Stokes contract"
        )
    configured_sampler = chunk_config.configuration.get("sampler")
    if (
        not isinstance(configured_sampler, Mapping)
        or configured_sampler.get("maximum_ursell_redraws") != maximum_ursell_redraws
    ):
        raise ValueError(
            "chunk configuration sampler redraw limit differs from the supplied "
            "static Stokes executor"
        )
    if contract.role == "paper_dataset" and (
        maximum_ursell_redraws != DEFAULT_MAXIMUM_URSELL_REDRAWS
        or sampler is not sample_stokes_simulation
        or state_constructor is not construct_stokes_state
        or target_evaluator is not compute_dno_target
    ):
        raise ValueError(
            "paper-dataset execution requires the production redraw limit, sampler, "
            "state constructor, and target evaluator"
        )


def _resolve_stokes_attempt(
    assignment: AttemptAssignment,
    *,
    contract: StaticStokesContract,
    maximum_ursell_redraws: int,
    sampler: StokesSampler,
) -> tuple[JsonRecord, StokesSample | SimulationOutcome]:
    try:
        sample = sampler(
            assignment,
            domain_length=contract.length,
            gravity=contract.gravity,
            maximum_ursell_redraws=maximum_ursell_redraws,
        )
    except UrsellRedrawLimitReached as error:
        return (
            dict(error.failure_record),
            _ursell_redraw_failure_outcome(contract),
        )

    if sample.assignment != assignment:
        raise RuntimeError("Stokes sampler returned a different assignment")
    return sample.to_json_record(), sample


def _evaluate_stokes_attempt(
    result: StokesSample | SimulationOutcome,
    *,
    contract: StaticStokesContract,
    state_constructor: StaticStokesStateConstructor,
    target_evaluator: StaticDnoEvaluator,
) -> SimulationOutcome:
    if isinstance(result, SimulationOutcome):
        return result
    return evaluate_static_stokes_sample(
        result,
        contract=contract,
        state_constructor=state_constructor,
        target_evaluator=target_evaluator,
    )


def make_static_stokes_batch_executor(
    *,
    chunk_config: DatasetChunkConfig,
    contract: StaticStokesContract,
    maximum_ursell_redraws: int = DEFAULT_MAXIMUM_URSELL_REDRAWS,
    metadata: Mapping[str, object] | None = None,
    sampler: StokesSampler = sample_stokes_simulation,
    state_constructor: StaticStokesStateConstructor = construct_stokes_state,
    target_evaluator: StaticDnoEvaluator = compute_dno_target,
) -> BatchExecutor:
    """Build the callback that resolves and commits static Stokes batches."""

    _validate_static_stokes_executor(
        chunk_config,
        contract,
        maximum_ursell_redraws,
        sampler,
        state_constructor,
        target_evaluator,
    )
    contract_record = contract.to_json_record()
    additional_metadata = dict(metadata or {})
    common_metadata = {
        "family": "stokes",
        "simulation_type": "static",
        "contract": contract_record,
        "sampler": {"maximum_ursell_redraws": maximum_ursell_redraws},
        "additional_metadata": additional_metadata,
    }

    def execute(
        assignments: tuple[AttemptAssignment, ...],
        *,
        batch_id: int,
    ) -> Path:
        if not assignments:
            raise ValueError("static Stokes attempt batches must not be empty")
        resolved = tuple(
            _resolve_stokes_attempt(
                assignment,
                contract=contract,
                maximum_ursell_redraws=maximum_ursell_redraws,
                sampler=sampler,
            )
            for assignment in assignments
        )
        batch_plan = build_batch_plan(
            assignments,
            tuple(specification for specification, _ in resolved),
            metadata={
                **common_metadata,
                "retained_times": [0.0],
                "selected_dense_indices": [0],
            },
        )
        path = batch_path(
            chunk_config.root,
            family=chunk_config.family_name,
            split=chunk_config.dataset_split.value,
            batch_id=batch_id,
        )

        complete_outcomes = tuple(
            _evaluate_stokes_attempt(
                result,
                contract=contract,
                state_constructor=state_constructor,
                target_evaluator=target_evaluator,
            )
            for _, result in resolved
        )
        commit_simulation_outcomes(
            path,
            batch_plan,
            complete_outcomes,
            metadata={
                **common_metadata,
                "attempted_simulations": len(complete_outcomes),
                "accepted_simulations": sum(
                    outcome.decision.accepted for outcome in complete_outcomes
                ),
                "sampling_exhaustions": sum(
                    isinstance(result, SimulationOutcome) for _, result in resolved
                ),
            },
        )
        return path

    return execute
