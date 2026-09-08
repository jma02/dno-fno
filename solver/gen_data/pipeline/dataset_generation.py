"""Generate accepted simulations in restartable, self-contained batches."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

from solver.gen_data.pipeline.batch_storage import load_completed_batch
from solver.gen_data.pipeline.types import (
    PhysicalFamilyId,
    RequestedSimulationsPerGroup,
)


BatchGenerator: TypeAlias = Callable[[tuple[str, ...], int, Path], None]
GenerationResult: TypeAlias = tuple[dict[str, int], tuple[Path, ...]]


def generate_simulations(
    root: Path,
    *,
    family_id: PhysicalFamilyId,
    seed: int,
    requested_simulations_per_group: RequestedSimulationsPerGroup,
    batch_size: int,
    generate_batch: BatchGenerator,
) -> GenerationResult:
    """Generate the requested successful simulations for each parameter group."""

    family_name = family_id.name.lower()
    directory = root / "batches" / family_name
    completed_batches = sorted(
        directory.glob("batch_*.npz"),
        key=lambda path: int(path.stem.removeprefix("batch_")),
    )
    # counting number of successful simulations we get for our quotas
    successful_per_group = dict.fromkeys(requested_simulations_per_group, 0)
    # counting number of total attempts we get for our attempt quotas
    attempts_per_group = dict.fromkeys(requested_simulations_per_group, 0)

    # Recover progress from saved batches before generating anything new.
    for batch_id in range(len(completed_batches)):
        batch = load_completed_batch(directory / f"batch_{batch_id:06d}.npz")
        if batch.family_id != family_id:
            raise RuntimeError("completed batch belongs to a different family")
        if batch.seed != seed:
            raise RuntimeError("completed batch belongs to a different seed")

        for group, succeeded in zip(
            batch.parameter_group_ids,
            batch.accepted_simulations,
            strict=True,
        ):
            attempts_per_group[group] += 1
            if succeeded:
                successful_per_group[group] += 1
        del batch

    # Check all saved totals before writing any new batches.
    for group, requested in requested_simulations_per_group.items():
        if successful_per_group[group] > requested:
            raise RuntimeError(
                f"completed batches exceed the accepted target for {group}"
            )
        if attempts_per_group[group] > 2 * requested:
            raise RuntimeError(
                f"completed batches exceed the attempt limit for {group}"
            )

    # Finish each group in order; only the final dataset needs the requested mix.
    for group, requested in requested_simulations_per_group.items():
        attempt_limit = 2 * requested
        while successful_per_group[group] < requested:
            number_of_simulations = min(
                batch_size,
                requested - successful_per_group[group],
                attempt_limit - attempts_per_group[group],
            )
            if number_of_simulations == 0:
                raise RuntimeError(
                    f"attempt limit reached for {group}: "
                    f"successful={successful_per_group[group]}/{requested}, "
                    f"attempts={attempts_per_group[group]}/{attempt_limit}"
                )
            output_path = directory / f"batch_{len(completed_batches):06d}.npz"
            generate_batch(
                (group,) * number_of_simulations,
                sum(attempts_per_group.values()),
                output_path,
            )
            batch = load_completed_batch(output_path)
            attempts_per_group[group] += number_of_simulations
            successful_per_group[group] += int(batch.accepted_simulations.sum())
            del batch  # Release loaded rows before generating the next batch.
            completed_batches.append(output_path)

    return attempts_per_group, tuple(completed_batches)
