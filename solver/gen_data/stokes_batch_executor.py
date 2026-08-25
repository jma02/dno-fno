"""Generate and validate one batch of static Stokes cases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from solver.gen_data.pipeline.archive import BatchPaths, ensure_proposal
from solver.gen_data.pipeline.production import (
    AttemptAssignment,
    PhysicalFamilyId,
)
from solver.gen_data.pipeline.quality import (
    QualityDecision,
    QualityReason,
    QualityScope,
)
from solver.gen_data.pipeline.valid_case_generation import DatasetGenerationSpec
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
from solver.gen_data.pipeline.reference import evaluate_discrete_dno_target


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
    reason = QualityReason.OUTSIDE_SUPPORT
    return CaseOutcome(
        decision=QualityDecision(
            scope=QualityScope.SAMPLE,
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


@dataclass(frozen=True)
class StaticStokesBatchExecutor:
    """Sample, propose, evaluate, and commit one static Stokes attempt batch."""

    run_spec: DatasetGenerationSpec
    contract: StaticStokesContract
    maximum_ursell_redraws: int = DEFAULT_MAXIMUM_URSELL_REDRAWS
    metadata: Mapping[str, object] | None = None
    sampler: StokesSampler = sample_stokes_case
    state_constructor: StaticStokesStateConstructor = construct_stokes_state
    target_evaluator: StaticDnoEvaluator = evaluate_discrete_dno_target

    def __post_init__(self) -> None:
        if self.run_spec.family_name != "stokes":
            raise ValueError("static Stokes generation must use family_name='stokes'")
        if self.run_spec.family_id is not PhysicalFamilyId.STOKES:
            raise ValueError("static Stokes generation requires the Stokes family ID")
        unknown_cells = set(self.run_spec.cell_codes).difference(_STOKES_CELL_IDS)
        if unknown_cells:
            raise ValueError(f"unknown Stokes sampling cells: {sorted(unknown_cells)}")
        if (
            isinstance(self.maximum_ursell_redraws, bool)
            or not isinstance(self.maximum_ursell_redraws, int)
            or self.maximum_ursell_redraws < 0
        ):
            raise ValueError("maximum_ursell_redraws must be a nonnegative integer")

        configured_contract = self.run_spec.configuration.get("contract")
        if (
            not isinstance(configured_contract, Mapping)
            or dict(configured_contract) != self.contract.to_json_record()
        ):
            raise ValueError(
                "run configuration contract differs from the supplied "
                "static Stokes contract"
            )
        configured_sampler = self.run_spec.configuration.get("sampler")
        if (
            not isinstance(configured_sampler, Mapping)
            or configured_sampler.get("maximum_ursell_redraws")
            != self.maximum_ursell_redraws
        ):
            raise ValueError(
                "run configuration sampler redraw limit differs from the "
                "supplied static Stokes executor"
            )
        if self.contract.role == "paper_dataset" and (
            self.maximum_ursell_redraws != DEFAULT_MAXIMUM_URSELL_REDRAWS
            or self.sampler is not sample_stokes_case
            or self.state_constructor is not construct_stokes_state
            or self.target_evaluator is not evaluate_discrete_dno_target
        ):
            raise ValueError(
                "paper-dataset execution requires the production redraw limit, "
                "sampler, state constructor, and target evaluator"
            )

    def _resolve(
        self,
        assignment: AttemptAssignment,
    ) -> tuple[JsonRecord, StokesSample | CaseOutcome]:
        try:
            sample = self.sampler(
                assignment,
                domain_length=self.contract.target.length,
                gravity=self.contract.gravity,
                maximum_ursell_redraws=self.maximum_ursell_redraws,
            )
        except UrsellRedrawLimitReached as error:
            return (
                dict(error.failure_record),
                _ursell_redraw_failure_outcome(self.contract),
            )

        if sample.assignment != assignment:
            raise RuntimeError("Stokes sampler returned a different assignment")
        return sample.to_json_record(), sample

    def _evaluate(self, result: StokesSample | CaseOutcome) -> CaseOutcome:
        if isinstance(result, CaseOutcome):
            return result
        return evaluate_static_stokes_sample(
            result,
            contract=self.contract,
            state_constructor=self.state_constructor,
            target_evaluator=self.target_evaluator,
        )

    def __call__(
        self,
        assignments: tuple[AttemptAssignment, ...],
        *,
        batch_id: int,
    ) -> BatchPaths:
        """Resolve and commit one batch in its immutable proposal order."""

        if not assignments:
            raise ValueError("static Stokes attempt batches must not be empty")
        resolved = tuple(map(self._resolve, assignments))
        contract_record = self.contract.to_json_record()
        additional_metadata = dict(self.metadata or {})
        proposal_arrays = build_proposal_arrays(
            assignments,
            tuple(specification for specification, _ in resolved),
            cell_codes=self.run_spec.cell_codes,
            batch_id=batch_id,
            config_fingerprint=self.run_spec.config_fingerprint,
            metadata={
                "family": "stokes",
                "case_kind": "static",
                "contract": contract_record,
                "sampler": {
                    "maximum_ursell_redraws": self.maximum_ursell_redraws,
                },
                "retained_times": [0.0],
                "selected_dense_indices": [0],
                "additional_metadata": additional_metadata,
            },
        )
        paths = batch_paths_for_assignments(
            self.run_spec.root,
            assignments,
            family_name=self.run_spec.family_name,
            batch_id=batch_id,
        )

        # No state construction or target evaluation occurs before this write.
        ensure_proposal(paths, proposal_arrays)
        complete_outcomes = tuple(self._evaluate(result) for _, result in resolved)
        commit_case_outcomes(
            paths,
            proposal_arrays,
            complete_outcomes,
            metadata={
                "family": "stokes",
                "case_kind": "static",
                "contract": contract_record,
                "sampler": {
                    "maximum_ursell_redraws": self.maximum_ursell_redraws,
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
