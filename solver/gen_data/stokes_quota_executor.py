"""Thin static-Stokes executor for the durable accepted-quota driver."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from types import MappingProxyType
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
from solver.gen_data.pipeline.quota_driver import AcceptedQuotaRunSpec
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
    StokesSamplingError,
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
JsonScalar: TypeAlias = str | int | float | bool | None
_STOKES_CELL_IDS = frozenset(cell.cell_id for cell in STOKES_SAMPLE_CELLS)


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


@dataclass(frozen=True)
class _ResolvedStokesAttempt:
    assignment: AttemptAssignment
    specification: JsonRecord
    sample: StokesSample | None
    preconstruction_outcome: CaseOutcome | None

    def __post_init__(self) -> None:
        if (self.sample is None) == (self.preconstruction_outcome is None):
            raise ValueError(
                "a resolved attempt must contain exactly one sample or rejection"
            )


def _strict_metadata_copy(
    metadata: Mapping[str, object],
) -> Mapping[str, object]:
    encoded = json.dumps(
        dict(metadata),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise TypeError("metadata must encode a JSON object")
    return MappingProxyType(value)


def _require_failure_identity(
    assignment: AttemptAssignment,
    record: Mapping[str, object],
) -> None:
    key = assignment.case_key
    expected: dict[str, object] = {
        "status": "failed_ursell_redraw_limit",
        "case_id": key.case_id,
        "family_id": key.family_id,
        "revision_id": key.revision_id,
        "split_id": key.split_id.value,
        "root_seed": key.root_seed,
        "stream_id": key.stream_id,
        "attempt_index": key.attempt_index,
        "cell_id": assignment.cell_id,
    }
    for name, expected_value in expected.items():
        if record.get(name) != expected_value:
            raise RuntimeError(f"Stokes exhaustion record has incorrect {name!r}")


def _exhaustion_outcome(
    record: Mapping[str, object],
    *,
    contract: StaticStokesContract,
) -> CaseOutcome:
    attempts = record.get("amplitude_attempts")
    if not isinstance(attempts, list) or not attempts:
        raise RuntimeError("Stokes exhaustion record must contain amplitude attempts")
    support_resampling_count = record.get("support_resampling_count")
    if (
        isinstance(support_resampling_count, bool)
        or not isinstance(support_resampling_count, int)
        or support_resampling_count != len(attempts) - 1
    ):
        raise RuntimeError("Stokes exhaustion record has inconsistent resampling count")
    last_attempt = attempts[-1]
    if not isinstance(last_attempt, dict):
        raise RuntimeError("Stokes amplitude attempt must be a JSON object")
    ursell = last_attempt.get("ursell_upper_bound")
    if ursell is not None and (
        isinstance(ursell, bool)
        or not isinstance(ursell, (int, float))
        or not math.isfinite(float(ursell))
    ):
        raise RuntimeError("persisted Ursell diagnostic must be finite or null")

    reason = QualityReason.OUTSIDE_SUPPORT
    decision = QualityDecision(
        scope=QualityScope.SAMPLE,
        required=STATIC_STOKES_REQUIRED_CHECKS,
        evaluated=reason,
        failed=reason,
    )
    metrics: dict[str, JsonScalar] = {
        "contract_role": contract.role,
        "sampling_status": "failed_ursell_redraw_limit",
        "support_violation_count": 1,
        "support_violations": (
            "finite-depth Stokes attempt exhausted same-cell amplitude redraws"
        ),
        "support_resampling_count": support_resampling_count,
        "amplitude_attempt_count": len(attempts),
        "last_ursell_upper_bound": (float(ursell) if ursell is not None else None),
        "state_finite": None,
        "minimum_water_column": None,
        "target_finite": None,
        "numerical_error": "",
    }
    return CaseOutcome(decision=decision, rows=None, metrics=metrics)


@dataclass(frozen=True)
class StaticStokesQuotaExecutor:
    """Sample, propose, evaluate, and commit one static Stokes attempt batch."""

    run_spec: AcceptedQuotaRunSpec
    contract: StaticStokesContract
    maximum_ursell_redraws: int = DEFAULT_MAXIMUM_URSELL_REDRAWS
    metadata: Mapping[str, object] | None = None
    sampler: StokesSampler = sample_stokes_case
    state_constructor: StaticStokesStateConstructor = construct_stokes_state
    target_evaluator: StaticDnoEvaluator = evaluate_discrete_dno_target

    def __post_init__(self) -> None:
        if self.run_spec.family_name != "stokes":
            raise ValueError("static Stokes quota runs must use family_name='stokes'")
        if self.run_spec.family_id is not PhysicalFamilyId.STOKES:
            raise ValueError("static Stokes quota runs require the Stokes family ID")
        unknown_cells = set(self.run_spec.cell_codes).difference(_STOKES_CELL_IDS)
        if unknown_cells:
            raise ValueError(f"unknown Stokes quota cells: {sorted(unknown_cells)}")
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
        object.__setattr__(
            self,
            "metadata",
            _strict_metadata_copy(self.metadata or {}),
        )

    def _resolve(
        self,
        assignment: AttemptAssignment,
    ) -> _ResolvedStokesAttempt:
        try:
            sample = self.sampler(
                assignment,
                domain_length=self.contract.target.length,
                gravity=self.contract.gravity,
                maximum_ursell_redraws=self.maximum_ursell_redraws,
            )
        except StokesSamplingError as error:
            record = dict(error.failure_record)
            _require_failure_identity(assignment, record)
            return _ResolvedStokesAttempt(
                assignment=assignment,
                specification=record,
                sample=None,
                preconstruction_outcome=_exhaustion_outcome(
                    record,
                    contract=self.contract,
                ),
            )

        if sample.assignment != assignment:
            raise RuntimeError("Stokes sampler returned a different assignment")
        return _ResolvedStokesAttempt(
            assignment=assignment,
            specification=sample.to_json_record(),
            sample=sample,
            preconstruction_outcome=None,
        )

    def _evaluate(self, attempt: _ResolvedStokesAttempt) -> CaseOutcome:
        if attempt.preconstruction_outcome is not None:
            return attempt.preconstruction_outcome
        if attempt.sample is None:
            raise RuntimeError("a successful Stokes attempt has no sample")
        return evaluate_static_stokes_sample(
            attempt.sample,
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
            tuple(attempt.specification for attempt in resolved),
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
        complete_outcomes = tuple(map(self._evaluate, resolved))
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
                    attempt.sample is None for attempt in resolved
                ),
                "additional_metadata": additional_metadata,
            },
        )
        return paths
