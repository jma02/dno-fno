"""Generate and validate one batch of static Stokes cases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypeAlias

from solver.gen_data.pipeline.batch_storage import BatchPaths, ensure_proposal
from solver.gen_data.pipeline.case_allocation import (
    AttemptAssignment,
    PhysicalFamilyId,
)
from solver.gen_data.pipeline.case_checks import (
    CaseCheckResult,
    CaseCheck,
)
from solver.gen_data.pipeline.valid_case_generation import (
    BatchExecutor,
    DatasetGenerationSpec,
)
from solver.gen_data.pipeline.writer import (
    CaseOutcome,
    batch_paths_for_assignments,
    build_proposal_arrays,
    commit_case_outcomes,
)
from solver.gen_data.stokes_sampling import (
    DEFAULT_MAXIMUM_URSELL_REDRAWS,
    STOKES_SAMPLE_CELLS,
    StokesSample,
    UrsellRedrawLimitReached,
    sample_stokes_case,
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
_STOKES_CELL_IDS = frozenset(STOKES_SAMPLE_CELLS)


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
) -> CaseOutcome:
    reason = CaseCheck.OUTSIDE_SUPPORT
    return CaseOutcome(
        decision=CaseCheckResult(
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
    run_spec: DatasetGenerationSpec,
    contract: StaticStokesContract,
    maximum_ursell_redraws: int,
    sampler: StokesSampler,
    state_constructor: StaticStokesStateConstructor,
    target_evaluator: StaticDnoEvaluator,
) -> None:
    if run_spec.family_name != "stokes":
        raise ValueError("static Stokes generation must use family_name='stokes'")
    if run_spec.family_id is not PhysicalFamilyId.STOKES:
        raise ValueError("static Stokes generation requires the Stokes family ID")
    unknown_cells = set(run_spec.cell_codes).difference(_STOKES_CELL_IDS)
    if unknown_cells:
        raise ValueError(f"unknown Stokes sampling cells: {sorted(unknown_cells)}")
    if (
        isinstance(maximum_ursell_redraws, bool)
        or not isinstance(maximum_ursell_redraws, int)
        or maximum_ursell_redraws < 0
    ):
        raise ValueError("maximum_ursell_redraws must be a nonnegative integer")

    configured_contract = run_spec.configuration.get("contract")
    if (
        not isinstance(configured_contract, Mapping)
        or dict(configured_contract) != contract.to_json_record()
    ):
        raise ValueError(
            "run configuration contract differs from the supplied static Stokes contract"
        )
    configured_sampler = run_spec.configuration.get("sampler")
    if (
        not isinstance(configured_sampler, Mapping)
        or configured_sampler.get("maximum_ursell_redraws")
        != maximum_ursell_redraws
    ):
        raise ValueError(
            "run configuration sampler redraw limit differs from the supplied "
            "static Stokes executor"
        )
    if contract.role == "paper_dataset" and (
        maximum_ursell_redraws != DEFAULT_MAXIMUM_URSELL_REDRAWS
        or sampler is not sample_stokes_case
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
) -> tuple[JsonRecord, StokesSample | CaseOutcome]:
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
    result: StokesSample | CaseOutcome,
    *,
    contract: StaticStokesContract,
    state_constructor: StaticStokesStateConstructor,
    target_evaluator: StaticDnoEvaluator,
) -> CaseOutcome:
    if isinstance(result, CaseOutcome):
        return result
    return evaluate_static_stokes_sample(
        result,
        contract=contract,
        state_constructor=state_constructor,
        target_evaluator=target_evaluator,
    )


def make_static_stokes_batch_executor(
    *,
    run_spec: DatasetGenerationSpec,
    contract: StaticStokesContract,
    maximum_ursell_redraws: int = DEFAULT_MAXIMUM_URSELL_REDRAWS,
    metadata: Mapping[str, object] | None = None,
    sampler: StokesSampler = sample_stokes_case,
    state_constructor: StaticStokesStateConstructor = construct_stokes_state,
    target_evaluator: StaticDnoEvaluator = compute_dno_target,
) -> BatchExecutor:
    """Build the callback that resolves and commits static Stokes batches."""

    _validate_static_stokes_executor(
        run_spec,
        contract,
        maximum_ursell_redraws,
        sampler,
        state_constructor,
        target_evaluator,
    )
    contract_record = contract.to_json_record()
    additional_metadata = dict(metadata or {})

    def execute(
        assignments: tuple[AttemptAssignment, ...],
        *,
        batch_id: int,
    ) -> BatchPaths:
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
        proposal_arrays = build_proposal_arrays(
            assignments,
            tuple(specification for specification, _ in resolved),
            cell_codes=run_spec.cell_codes,
            batch_id=batch_id,
            metadata={
                "family": "stokes",
                "case_kind": "static",
                "contract": contract_record,
                "sampler": {
                    "maximum_ursell_redraws": maximum_ursell_redraws,
                },
                "retained_times": [0.0],
                "selected_dense_indices": [0],
                "additional_metadata": additional_metadata,
            },
        )
        paths = batch_paths_for_assignments(
            run_spec.root,
            assignments,
            family_name=run_spec.family_name,
            batch_id=batch_id,
        )

        # No state construction or target evaluation occurs before this write.
        ensure_proposal(paths, proposal_arrays)
        complete_outcomes = tuple(
            _evaluate_stokes_attempt(
                result,
                contract=contract,
                state_constructor=state_constructor,
                target_evaluator=target_evaluator,
            )
            for _, result in resolved
        )
        commit_case_outcomes(
            paths,
            proposal_arrays,
            complete_outcomes,
            metadata={
                "family": "stokes",
                "case_kind": "static",
                "contract": contract_record,
                "sampler": {
                    "maximum_ursell_redraws": maximum_ursell_redraws,
                },
                "attempted_cases": len(complete_outcomes),
                "accepted_cases": sum(
                    outcome.decision.accepted for outcome in complete_outcomes
                ),
                "sampling_exhaustions": sum(
                    isinstance(result, CaseOutcome) for _, result in resolved
                ),
                "additional_metadata": additional_metadata,
            },
        )
        return paths

    return execute
