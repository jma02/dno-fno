"""Generate and validate one batch of static Stokes simulations."""

from __future__ import annotations

from pathlib import Path

from solver.gen_data.pipeline.batch_storage import save_completed_batch
from solver.gen_data.pipeline.types import (
    PhysicalFamilyId,
    SimulationRows,
)
from solver.gen_data.stokes_sampling import sample_stokes_simulation
from solver.gen_data.stokes_static_pipeline import evaluate_static_stokes_sample


def generate_static_stokes_batch(
    parameter_group_ids: tuple[str, ...],
    first_attempt_number: int,
    output_path: Path,
    *,
    seed: int,
) -> int:
    """Save one static Stokes batch and return its accepted simulation count."""

    rows_by_simulation: list[SimulationRows | None] = []
    for offset, parameter_group_id in enumerate(parameter_group_ids):
        sample = sample_stokes_simulation(
            parameter_group_id,
            seed=seed,
            attempt_number=first_attempt_number + offset,
        )
        rows_by_simulation.append(
            evaluate_static_stokes_sample(sample) if sample is not None else None
        )

    return save_completed_batch(
        output_path,
        parameter_group_ids,
        rows_by_simulation,
        family_id=PhysicalFamilyId.STOKES,
        seed=seed,
    )
