"""Generate accepted simulations in restartable, self-contained batches."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeAlias

from solver.gen_data.pipeline.batch_storage import batch_path, load_completed_batch
from solver.gen_data.pipeline.types import DatasetSplit, PhysicalFamilyId


BatchGenerator: TypeAlias = Callable[[tuple[str, ...], int, Path], None]
GenerationResult: TypeAlias = tuple[dict[str, int], tuple[Path, ...]]
MAX_RETRIES_PER_PARAMETER_GROUP = 64


def _update_counts_from_batch(
    path: Path,
    *,
    family_id: PhysicalFamilyId,
    dataset_split: DatasetSplit,
    accepted_targets: Mapping[str, int],
    accepted_counts: dict[str, int],
    attempt_counts: dict[str, int],
) -> None:
    batch = load_completed_batch(path)
    if batch.family_id != family_id:
        raise RuntimeError("completed batch belongs to a different family")
    if batch.dataset_split != dataset_split:
        raise RuntimeError("completed batch belongs to a different dataset split")

    for parameter_group_id, accepted in zip(
        batch.parameter_group_ids,
        batch.accepted_simulations,
        strict=True,
    ):
        target = accepted_targets[parameter_group_id]
        attempt_counts[parameter_group_id] += 1
        if accepted:
            accepted_counts[parameter_group_id] += 1
        if accepted_counts[parameter_group_id] > target:
            raise RuntimeError(
                f"completed batches exceed the accepted target for {parameter_group_id}"
            )
        maximum_attempts = target + MAX_RETRIES_PER_PARAMETER_GROUP if target else 0
        if attempt_counts[parameter_group_id] > maximum_attempts:
            raise RuntimeError(
                f"completed batches exceed the attempt limit for {parameter_group_id}"
            )


def generate_simulations(
    root: Path,
    *,
    family_id: PhysicalFamilyId,
    dataset_split: DatasetSplit,
    accepted_targets: Mapping[str, int],
    batch_size: int,
    generate_batch: BatchGenerator,
) -> GenerationResult:
    """Generate until every parameter group has its requested accepted count."""

    if (
        batch_size <= 0
        or not accepted_targets
        or any(
            not isinstance(parameter_group_id, str) or not parameter_group_id
            for parameter_group_id in accepted_targets
        )
        or any(
            not isinstance(target, int) or isinstance(target, bool) or target < 0
            for target in accepted_targets.values()
        )
    ):
        raise ValueError(
            "batch size must be positive and accepted targets must map nonempty "
            "strings to nonnegative integers"
        )

    family_name = family_id.name.lower()
    directory = batch_path(
        root,
        family=family_name,
        split=dataset_split.value,
        batch_id=0,
    ).parent
    if directory.exists() and not directory.is_dir():
        raise RuntimeError(f"completed batch path is not a directory: {directory}")
    completed_batches = sorted(
        directory.glob("batch_*.npz"),
        key=lambda path: int(path.stem.removeprefix("batch_")),
    )
    if any(
        path.name != f"batch_{batch_id:06d}.npz"
        for batch_id, path in enumerate(completed_batches)
    ):
        raise RuntimeError("completed batch IDs must be contiguous from zero")

    accepted_counts = dict.fromkeys(accepted_targets, 0)
    attempt_counts = dict(accepted_counts)
    for path in completed_batches:
        _update_counts_from_batch(
            path,
            family_id=family_id,
            dataset_split=dataset_split,
            accepted_targets=accepted_targets,
            accepted_counts=accepted_counts,
            attempt_counts=attempt_counts,
        )

    while accepted_counts != accepted_targets:
        available_slots = sorted(
            [
                (
                    accepted_counts[parameter_group_id] + offset,
                    parameter_group_id,
                )
                for parameter_group_id, target in accepted_targets.items()
                for offset in range(
                    min(
                        batch_size,
                        target - accepted_counts[parameter_group_id],
                        target
                        + MAX_RETRIES_PER_PARAMETER_GROUP
                        - attempt_counts[parameter_group_id],
                    )
                )
            ],
            key=lambda slot: slot[0],
        )
        parameter_group_ids = tuple(
            parameter_group_id for _, parameter_group_id in available_slots[:batch_size]
        )

        if not parameter_group_ids:
            unfinished = "; ".join(
                f"{parameter_group_id}: accepted={accepted_counts[parameter_group_id]}/"
                f"{target}, attempts={attempt_counts[parameter_group_id]}/"
                f"{target + MAX_RETRIES_PER_PARAMETER_GROUP}"
                for parameter_group_id, target in accepted_targets.items()
                if accepted_counts[parameter_group_id] < target
            )
            raise RuntimeError(
                "attempt limits reached before all requested simulations were accepted; "
                f"{unfinished}"
            )

        output_path = batch_path(
            root,
            family=family_name,
            split=dataset_split.value,
            batch_id=len(completed_batches),
        )
        generate_batch(
            parameter_group_ids,
            sum(attempt_counts.values()),
            output_path,
        )
        if not output_path.exists():
            raise RuntimeError("batch generator returned without saving the batch")

        _update_counts_from_batch(
            output_path,
            family_id=family_id,
            dataset_split=dataset_split,
            accepted_targets=accepted_targets,
            accepted_counts=accepted_counts,
            attempt_counts=attempt_counts,
        )
        completed_batches.append(output_path)

    return attempt_counts, tuple(completed_batches)
