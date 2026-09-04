"""Generate and validate one batch of static Stokes simulations."""

from __future__ import annotations

from pathlib import Path

from solver.gen_data.pipeline.simulation_checks import SimulationCheckResult
from solver.gen_data.pipeline.types import DatasetSplit, PhysicalFamilyId
from solver.gen_data.pipeline.writer import (
    SimulationOutcome,
    build_batch_plan,
    commit_simulation_outcomes,
)
from solver.gen_data.stokes_sampling import sample_stokes_simulation
from solver.gen_data.stokes_static_pipeline import evaluate_static_stokes_sample


def generate_static_stokes_batch(
    parameter_group_ids: tuple[str, ...],
    first_attempt_number: int,
    output_path: Path,
    *,
    dataset_split: DatasetSplit,
) -> None:
    """Sample, evaluate, and save one static Stokes batch."""

    specifications: list[dict[str, object]] = []
    outcomes: list[SimulationOutcome] = []
    for offset, parameter_group_id in enumerate(parameter_group_ids):
        sample = sample_stokes_simulation(
            parameter_group_id,
            dataset_split=dataset_split,
            attempt_number=first_attempt_number + offset,
        )
        if sample is None:
            specifications.append({})
            outcomes.append(
                SimulationOutcome(
                    decision=SimulationCheckResult(
                        accepted=False,
                        outside_support=True,
                    ),
                    rows=None,
                    metrics={},
                )
            )
            continue

        specifications.append(sample._asdict())
        outcomes.append(evaluate_static_stokes_sample(sample))

    batch_plan = build_batch_plan(
        parameter_group_ids,
        specifications,
        family_id=PhysicalFamilyId.STOKES,
        dataset_split=dataset_split,
    )
    commit_simulation_outcomes(output_path, batch_plan, outcomes)
