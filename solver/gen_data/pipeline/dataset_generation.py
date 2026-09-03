"""Generate accepted simulations in restartable, self-contained batches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from solver.gen_data.pipeline.batch_storage import (
    batch_path,
    load_completed_batch,
)
from solver.gen_data.pipeline.simulation_allocation import (
    DatasetSplit,
    PhysicalFamilyId,
    select_next_parameter_groups,
)

MAX_RETRIES_PER_PARAMETER_GROUP = 32


@dataclass(frozen=True)
class DatasetChunkConfig:
    """Configuration for generating one dataset chunk across multiple batches."""

    root: Path
    family_name: str
    family_id: PhysicalFamilyId
    dataset_split: DatasetSplit
    simulation_targets: Mapping[str, int]
    batch_size: int
    solver_batch_size: int | None = None

    def __post_init__(self) -> None:
        if not self.simulation_targets:
            raise ValueError("simulation_targets must not be empty")
        if any(
            not isinstance(parameter_group_id, str) or not parameter_group_id
            for parameter_group_id in self.simulation_targets
        ):
            raise ValueError("simulation target keys must be nonempty strings")
        if any(
            not isinstance(simulation_count, int)
            or isinstance(simulation_count, bool)
            or simulation_count < 0
            for simulation_count in self.simulation_targets.values()
        ):
            raise ValueError("simulation target values must be nonnegative integers")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.solver_batch_size is not None and not (
            0 < self.solver_batch_size <= self.batch_size
        ):
            raise ValueError("solver_batch_size must be between 1 and batch_size")

    def to_json_record(self) -> dict[str, object]:
        """Return the complete chunk configuration."""

        accepted_count = sum(self.simulation_targets.values())
        record: dict[str, object] = {
            "family_name": self.family_name,
            "family_id": int(self.family_id),
            "dataset_split": self.dataset_split.value,
            "batch_size": self.batch_size,
            "accepted_simulation_count": accepted_count,
            "quotas": [
                {"parameter_group_id": parameter_group_id, "target_accepted": count}
                for parameter_group_id, count in self.simulation_targets.items()
            ],
        }
        if self.solver_batch_size is not None:
            record["solver_batch_size"] = self.solver_batch_size
        return record

    @property
    def maximum_attempts_by_parameter_group(self) -> Mapping[str, int]:
        """Return the simulation-attempt limit for each parameter group."""

        return {
            parameter_group_id: (
                simulation_count + MAX_RETRIES_PER_PARAMETER_GROUP
                if simulation_count > 0
                else 0
            )
            for parameter_group_id, simulation_count in self.simulation_targets.items()
        }


@dataclass(frozen=True)
class DatasetChunkState:
    """Counts reconstructed from completed batch artifacts."""

    accepted_simulation_counts: Mapping[str, int]
    simulation_attempt_counts: Mapping[str, int]
    completed_batches: tuple[Path, ...]
    complete: bool


class BatchGenerator(Protocol):
    """Generate and save one complete batch."""

    def __call__(
        self,
        parameter_group_ids: tuple[str, ...],
        *,
        first_attempt_number: int,
        batch_id: int,
    ) -> Path: ...


def _find_completed_batches(chunk_config: DatasetChunkConfig) -> tuple[Path, ...]:
    directory = batch_path(
        chunk_config.root,
        family=chunk_config.family_name,
        split=chunk_config.dataset_split.value,
        batch_id=0,
    ).parent
    if not directory.exists():
        return ()
    if not directory.is_dir():
        raise RuntimeError(f"completed batch path is not a directory: {directory}")

    completed_batches = tuple(
        sorted(
            directory.glob("batch_*.npz"),
            key=lambda path: int(path.stem.removeprefix("batch_")),
        )
    )
    for batch_id, path in enumerate(completed_batches):
        if path != batch_path(
            chunk_config.root,
            family=chunk_config.family_name,
            split=chunk_config.dataset_split.value,
            batch_id=batch_id,
        ):
            raise RuntimeError("completed batch IDs must be contiguous from zero")
    return completed_batches


def _add_completed_batch(
    chunk_config: DatasetChunkConfig,
    path: Path,
    accepted_counts: dict[str, int],
    attempt_counts: dict[str, int],
) -> None:
    """Validate one saved batch and add its counts to the generation state."""

    batch = load_completed_batch(path)
    plan = batch.plan
    if int(plan["family_id"].item()) != int(chunk_config.family_id):
        raise RuntimeError("completed batch belongs to a different family")
    if str(plan["dataset_split"].item()) != chunk_config.dataset_split.value:
        raise RuntimeError("completed batch belongs to a different dataset split")
    parameter_groups = tuple(str(value) for value in plan["parameter_group_id"])
    unknown_parameter_groups = set(parameter_groups) - set(accepted_counts)
    if unknown_parameter_groups:
        raise RuntimeError(
            "completed batch contains unknown parameter groups: "
            f"{sorted(unknown_parameter_groups)}"
        )
    for parameter_group_id, simulation in zip(
        parameter_groups, batch.simulations, strict=True
    ):
        attempt_counts[parameter_group_id] += 1
        if simulation.accepted:
            accepted_counts[parameter_group_id] += 1

    for (
        parameter_group_id,
        simulation_target,
    ) in chunk_config.simulation_targets.items():
        if accepted_counts[parameter_group_id] > simulation_target:
            raise RuntimeError(
                f"completed batches exceed the accepted target for {parameter_group_id}"
            )
        if (
            attempt_counts[parameter_group_id]
            > chunk_config.maximum_attempts_by_parameter_group[parameter_group_id]
        ):
            raise RuntimeError(
                f"completed batches exceed the attempt limit for {parameter_group_id}"
            )


def _targets_met(
    chunk_config: DatasetChunkConfig,
    accepted_counts: Mapping[str, int],
) -> bool:
    return all(
        accepted_counts[parameter_group_id] == simulation_target
        for parameter_group_id, simulation_target in (
            chunk_config.simulation_targets.items()
        )
    )


def scan_dataset_generation(chunk_config: DatasetChunkConfig) -> DatasetChunkState:
    """Count accepted and attempted simulations in completed batches."""

    accepted_counts = dict.fromkeys(chunk_config.simulation_targets, 0)
    attempt_counts = dict(accepted_counts)
    completed_batches = _find_completed_batches(chunk_config)
    for path in completed_batches:
        _add_completed_batch(chunk_config, path, accepted_counts, attempt_counts)

    return DatasetChunkState(
        accepted_simulation_counts=accepted_counts,
        simulation_attempt_counts=attempt_counts,
        completed_batches=completed_batches,
        complete=_targets_met(chunk_config, accepted_counts),
    )


def _attempt_limit_error(
    chunk_config: DatasetChunkConfig,
    state: DatasetChunkState,
) -> RuntimeError:
    limits = chunk_config.maximum_attempts_by_parameter_group
    unfinished = "; ".join(
        f"{parameter_group_id}: accepted="
        f"{state.accepted_simulation_counts[parameter_group_id]}/"
        f"{simulation_target}, attempts="
        f"{state.simulation_attempt_counts[parameter_group_id]}/"
        f"{limits[parameter_group_id]}"
        for parameter_group_id, simulation_target in (
            chunk_config.simulation_targets.items()
        )
        if state.accepted_simulation_counts[parameter_group_id] < simulation_target
    )
    return RuntimeError(
        "attempt limits reached before all requested simulations were accepted; "
        f"{unfinished}"
    )


def generate_simulations(
    chunk_config: DatasetChunkConfig,
    generator: BatchGenerator,
) -> DatasetChunkState:
    """Generate batches until every sampling-group target is met."""

    state = scan_dataset_generation(chunk_config)
    while not state.complete:
        parameter_group_ids = select_next_parameter_groups(
            chunk_config.simulation_targets,
            state.accepted_simulation_counts,
            state.simulation_attempt_counts,
            chunk_config.maximum_attempts_by_parameter_group,
            batch_size=chunk_config.batch_size,
        )
        if not parameter_group_ids:
            raise _attempt_limit_error(chunk_config, state)

        batch_id = len(state.completed_batches)
        first_attempt_number = sum(state.simulation_attempt_counts.values())
        expected_path = batch_path(
            chunk_config.root,
            family=chunk_config.family_name,
            split=chunk_config.dataset_split.value,
            batch_id=batch_id,
        )
        returned_path = generator(
            parameter_group_ids,
            first_attempt_number=first_attempt_number,
            batch_id=batch_id,
        )
        if returned_path != expected_path:
            raise RuntimeError("batch generator returned a nonstandard path")
        if not expected_path.exists():
            raise RuntimeError("batch generator returned without saving the batch")

        accepted_counts = dict(state.accepted_simulation_counts)
        attempt_counts = dict(state.simulation_attempt_counts)
        _add_completed_batch(
            chunk_config, expected_path, accepted_counts, attempt_counts
        )
        state = DatasetChunkState(
            accepted_simulation_counts=accepted_counts,
            simulation_attempt_counts=attempt_counts,
            completed_batches=(*state.completed_batches, expected_path),
            complete=_targets_met(chunk_config, accepted_counts),
        )
    return state
