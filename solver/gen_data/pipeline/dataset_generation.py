"""Generate accepted simulations in restartable, self-contained batches."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

from solver.gen_data.pipeline.batch_storage import batch_path, load_completed_batch
from solver.gen_data.pipeline.types import DatasetSplit, PhysicalFamilyId


# Parameter-group name -> number of successful simulations requested.
RequestedSimulationsPerGroup: TypeAlias = dict[str, int]
BatchGenerator: TypeAlias = Callable[[tuple[str, ...], int, Path], None]
GenerationResult: TypeAlias = tuple[dict[str, int], tuple[Path, ...]]
MAX_RETRIES_PER_PARAMETER_GROUP = 64


def generate_simulations(
    root: Path,
    *,
    family_id: PhysicalFamilyId,
    dataset_split: DatasetSplit,
    requested_simulations_per_group: RequestedSimulationsPerGroup,
    batch_size: int,
    generate_batch: BatchGenerator,
) -> GenerationResult:
    """Generate the requested successful simulations for each parameter group."""

    family_name = family_id.name.lower()
    directory = batch_path(
        root,
        family=family_name,
        split=dataset_split.value,
        batch_id=0,
    ).parent
    completed_batches = sorted(
        directory.glob("batch_*.npz"),
        key=lambda path: int(path.stem.removeprefix("batch_")),
    )
    successful_per_group = dict.fromkeys(requested_simulations_per_group, 0)
    attempts_per_group = dict(successful_per_group)
    batch_id = 0
    while (
        batch_id < len(completed_batches)
        or successful_per_group != requested_simulations_per_group
    ):
        output_path = batch_path(
            root,
            family=family_name,
            split=dataset_split.value,
            batch_id=batch_id,
        )
        if batch_id == len(completed_batches):
            available_slots = sorted(
                [
                    (successful_per_group[group] + offset, group)
                    for group, requested in requested_simulations_per_group.items()
                    for offset in range(
                        min(
                            batch_size,
                            requested - successful_per_group[group],
                            requested
                            + MAX_RETRIES_PER_PARAMETER_GROUP
                            - attempts_per_group[group],
                        )
                    )
                ],
                key=lambda slot: slot[0],
            )
            parameter_group_ids = tuple(
                group for _, group in available_slots[:batch_size]
            )

            if not parameter_group_ids:
                unfinished = "; ".join(
                    f"{group}: accepted={successful_per_group[group]}/"
                    f"{requested}, attempts={attempts_per_group[group]}/"
                    f"{requested + MAX_RETRIES_PER_PARAMETER_GROUP}"
                    for group, requested in requested_simulations_per_group.items()
                    if successful_per_group[group] < requested
                )
                raise RuntimeError(
                    "attempt limits reached before all requested simulations were accepted; "
                    f"{unfinished}"
                )

            generate_batch(
                parameter_group_ids,
                sum(attempts_per_group.values()),
                output_path,
            )

        batch = load_completed_batch(output_path)
        if batch.family_id != family_id:
            raise RuntimeError("completed batch belongs to a different family")
        if batch.dataset_split != dataset_split:
            raise RuntimeError("completed batch belongs to a different dataset split")

        for group, succeeded in zip(
            batch.parameter_group_ids,
            batch.accepted_simulations,
            strict=True,
        ):
            requested = requested_simulations_per_group[group]
            attempts_per_group[group] += 1
            if succeeded:
                successful_per_group[group] += 1
            if successful_per_group[group] > requested:
                raise RuntimeError(
                    f"completed batches exceed the accepted target for {group}"
                )
            attempt_limit = (
                requested + MAX_RETRIES_PER_PARAMETER_GROUP if requested else 0
            )
            if attempts_per_group[group] > attempt_limit:
                raise RuntimeError(
                    f"completed batches exceed the attempt limit for {group}"
                )
        del batch  # Release loaded rows before generating the next batch.

        if batch_id == len(completed_batches):
            completed_batches.append(output_path)
        batch_id += 1

    return attempts_per_group, tuple(completed_batches)
