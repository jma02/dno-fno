"""Generate accepted simulations in restartable, self-contained batches."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
from pathlib import Path
from typing import Protocol

import numpy as np

from solver.gen_data.pipeline.batch_storage import (
    CompletedBatch,
    batch_path,
    load_completed_batch,
)
from solver.gen_data.pipeline.simulation_allocation import (
    AttemptAssignment,
    DatasetSplit,
    ParameterGroupTarget,
    PhysicalFamilyId,
    SimulationKey,
    build_next_attempt_batch,
)

MAX_RETRIES_PER_PARAMETER_GROUP = 32


@dataclass(frozen=True)
class DatasetChunkConfig:
    """Configuration for generating one dataset chunk across multiple batches."""

    root: Path
    family_name: str
    family_id: PhysicalFamilyId
    dataset_split: DatasetSplit
    worker_stream_id: int
    simulation_targets: tuple[ParameterGroupTarget, ...]
    batch_size: int
    configuration: Mapping[str, object]
    first_attempt_index: int = 0

    def __post_init__(self) -> None:
        parameter_group_ids = tuple(
            target.parameter_group_id for target in self.simulation_targets
        )
        if not parameter_group_ids:
            raise ValueError("simulation_targets must not be empty")
        if any(not parameter_group_id for parameter_group_id in parameter_group_ids):
            raise ValueError("simulation target parameter_group_ids must not be empty")
        if len(set(parameter_group_ids)) != len(parameter_group_ids):
            raise ValueError("simulation target parameter_group_ids must be unique")
        if any(target.simulation_count < 0 for target in self.simulation_targets):
            raise ValueError("simulation target counts must be nonnegative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        object.__setattr__(self, "configuration", dict(self.configuration))

    def to_json_record(self) -> dict[str, object]:
        """Return the complete chunk configuration."""

        return {
            "family_name": self.family_name,
            "family_id": int(self.family_id),
            "dataset_split": self.dataset_split.value,
            "worker_stream_id": self.worker_stream_id,
            "first_attempt_index": self.first_attempt_index,
            "batch_size": self.batch_size,
            "quotas": [
                {
                    "parameter_group_id": target.parameter_group_id,
                    "target_accepted": target.simulation_count,
                }
                for target in self.simulation_targets
            ],
            "configuration": dict(self.configuration),
            "maximum_retries_per_parameter_group": MAX_RETRIES_PER_PARAMETER_GROUP,
        }

    @property
    def maximum_attempts_by_parameter_group(self) -> Mapping[str, int]:
        """Return the simulation-attempt limit for each parameter group."""

        return {
            target.parameter_group_id: (
                target.simulation_count + MAX_RETRIES_PER_PARAMETER_GROUP
                if target.simulation_count > 0
                else 0
            )
            for target in self.simulation_targets
        }


@dataclass(frozen=True)
class DatasetChunkState:
    """Counts reconstructed from completed batch artifacts."""

    accepted_simulation_counts: Mapping[str, int]
    simulation_attempt_counts: Mapping[str, int]
    completed_batches: tuple[Path, ...]
    complete: bool


class BatchExecutor(Protocol):
    """Execute and save one complete batch."""

    def __call__(
        self,
        assignments: tuple[AttemptAssignment, ...],
        *,
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

    batches_by_id: dict[int, Path] = {}
    for path in directory.glob("batch_*.npz"):
        batch_id_text = path.stem.removeprefix("batch_")
        if not batch_id_text.isdecimal():
            raise RuntimeError(f"invalid completed batch name: {path}")
        batch_id = int(batch_id_text)
        expected = batch_path(
            chunk_config.root,
            family=chunk_config.family_name,
            split=chunk_config.dataset_split.value,
            batch_id=batch_id,
        )
        if path != expected:
            raise RuntimeError(f"noncanonical completed batch path: {path}")
        batches_by_id[batch_id] = path

    batch_ids = tuple(sorted(batches_by_id))
    if batch_ids != tuple(range(len(batch_ids))):
        raise RuntimeError("completed batch IDs must be contiguous from zero")
    return tuple(batches_by_id[batch_id] for batch_id in batch_ids)


def _load_assignments(
    chunk_config: DatasetChunkConfig,
    batch: CompletedBatch,
) -> tuple[AttemptAssignment, ...]:
    plan = batch.plan
    if int(plan["family_id"].item()) != int(chunk_config.family_id):
        raise RuntimeError("completed batch belongs to a different family")
    if str(plan["dataset_split"].item()) != chunk_config.dataset_split.value:
        raise RuntimeError("completed batch belongs to a different dataset split")
    if not np.all(plan["worker_stream_id"] == chunk_config.worker_stream_id):
        raise RuntimeError("completed batch belongs to a different worker stream")

    known_parameter_groups = {
        target.parameter_group_id for target in chunk_config.simulation_targets
    }
    parameter_groups = tuple(str(value) for value in plan["parameter_group_id"])
    unknown_parameter_groups = set(parameter_groups) - known_parameter_groups
    if unknown_parameter_groups:
        raise RuntimeError(
            "completed batch contains unknown parameter groups: "
            f"{sorted(unknown_parameter_groups)}"
        )
    return tuple(
        AttemptAssignment(
            simulation_key=SimulationKey(
                family_id=int(chunk_config.family_id),
                dataset_split=chunk_config.dataset_split,
                worker_stream_id=chunk_config.worker_stream_id,
                attempt_index=int(attempt_index),
            ),
            parameter_group_id=parameter_group_id,
        )
        for attempt_index, parameter_group_id in zip(
            plan["attempt_index"],
            parameter_groups,
            strict=True,
        )
    )


def scan_dataset_generation(chunk_config: DatasetChunkConfig) -> DatasetChunkState:
    """Count accepted and attempted simulations in completed batches."""

    accepted_counts = {
        target.parameter_group_id: 0 for target in chunk_config.simulation_targets
    }
    attempt_counts = dict(accepted_counts)
    next_attempt_index = chunk_config.first_attempt_index
    completed_batches = _find_completed_batches(chunk_config)

    for path in completed_batches:
        batch = load_completed_batch(path)
        assignments = _load_assignments(chunk_config, batch)
        expected = build_next_attempt_batch(
            chunk_config.simulation_targets,
            accepted_counts,
            attempt_counts,
            chunk_config.maximum_attempts_by_parameter_group,
            family_id=int(chunk_config.family_id),
            dataset_split=chunk_config.dataset_split,
            worker_stream_id=chunk_config.worker_stream_id,
            first_attempt_index=next_attempt_index,
            batch_size=chunk_config.batch_size,
        )
        if assignments != expected:
            raise RuntimeError(
                "completed batch differs from the deterministic simulation schedule"
            )
        next_attempt_index += len(assignments)
        for assignment, simulation in zip(
            assignments,
            batch.simulations,
            strict=True,
        ):
            attempt_counts[assignment.parameter_group_id] += 1
            if simulation.accepted:
                accepted_counts[assignment.parameter_group_id] += 1

    complete = all(
        accepted_counts[target.parameter_group_id] == target.simulation_count
        for target in chunk_config.simulation_targets
    )
    return DatasetChunkState(
        accepted_simulation_counts=accepted_counts,
        simulation_attempt_counts=attempt_counts,
        completed_batches=completed_batches,
        complete=complete,
    )


@contextmanager
def dataset_generation_lock(chunk_config: DatasetChunkConfig) -> Iterator[None]:
    """Prevent two processes from writing the same dataset chunk."""

    path = (
        chunk_config.root
        / ".locks"
        / chunk_config.family_name
        / f"{chunk_config.dataset_split.value}.lock"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o600, exist_ok=True)
    with path.open("rb") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another dataset-generation process holds {path}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _attempt_limit_error(
    chunk_config: DatasetChunkConfig,
    state: DatasetChunkState,
) -> RuntimeError:
    limits = chunk_config.maximum_attempts_by_parameter_group
    unfinished = "; ".join(
        (
            f"{target.parameter_group_id}: accepted="
            f"{state.accepted_simulation_counts[target.parameter_group_id]}/"
            f"{target.simulation_count}, attempts="
            f"{state.simulation_attempt_counts[target.parameter_group_id]}/"
            f"{limits[target.parameter_group_id]}"
        )
        for target in chunk_config.simulation_targets
        if state.accepted_simulation_counts[target.parameter_group_id]
        < target.simulation_count
    )
    return RuntimeError(
        "attempt limits reached before all requested simulations were accepted; "
        f"{unfinished}"
    )


def generate_simulations(
    chunk_config: DatasetChunkConfig,
    executor: BatchExecutor,
) -> DatasetChunkState:
    """Generate batches until every sampling-group target is met."""

    with dataset_generation_lock(chunk_config):
        state = scan_dataset_generation(chunk_config)
        while not state.complete:
            assignments = build_next_attempt_batch(
                chunk_config.simulation_targets,
                state.accepted_simulation_counts,
                state.simulation_attempt_counts,
                chunk_config.maximum_attempts_by_parameter_group,
                family_id=int(chunk_config.family_id),
                dataset_split=chunk_config.dataset_split,
                worker_stream_id=chunk_config.worker_stream_id,
                first_attempt_index=(
                    chunk_config.first_attempt_index
                    + sum(state.simulation_attempt_counts.values())
                ),
                batch_size=chunk_config.batch_size,
            )
            if not assignments:
                raise _attempt_limit_error(chunk_config, state)

            batch_id = len(state.completed_batches)
            expected_path = batch_path(
                chunk_config.root,
                family=chunk_config.family_name,
                split=chunk_config.dataset_split.value,
                batch_id=batch_id,
            )
            returned_path = executor(assignments, batch_id=batch_id)
            if returned_path != expected_path:
                raise RuntimeError("batch executor returned a nonstandard path")
            if not expected_path.exists():
                raise RuntimeError("batch executor returned without saving the batch")
            state = scan_dataset_generation(chunk_config)
        return state
