"""Generate the requested number of accepted simulations per parameter group.

The numerical executor remains family-specific. This module recovers saved
counts, resumes an interrupted batch, schedules replacements for groups
that remain short, and stops when every target is met or the chunk fails.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
from pathlib import Path
from typing import Protocol

import numpy as np

from solver.gen_data.pipeline.batch_storage import (
    BatchPaths,
    BatchStatus,
    inspect_batch,
)
from solver.gen_data.pipeline.artifact_io import load_npz
from solver.gen_data.pipeline.simulation_allocation import (
    AttemptAssignment,
    PhysicalFamilyId,
    ParameterGroupTarget,
    SimulationKey,
    DatasetSplit,
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
    parameter_group_codes: Mapping[str, int]
    batch_size: int
    configuration: Mapping[str, object]
    first_attempt_index: int = 0

    def __post_init__(self) -> None:
        if not self.simulation_targets:
            raise ValueError("simulation_targets must not be empty")
        parameter_group_ids = tuple(
            target.parameter_group_id for target in self.simulation_targets
        )
        if any(not parameter_group_id for parameter_group_id in parameter_group_ids):
            raise ValueError("simulation target parameter_group_ids must not be empty")
        if len(set(parameter_group_ids)) != len(parameter_group_ids):
            raise ValueError("simulation target parameter_group_ids must be unique")
        if any(target.simulation_count < 0 for target in self.simulation_targets):
            raise ValueError("simulation target counts must be nonnegative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        codes = dict(self.parameter_group_codes)
        if set(codes) != set(parameter_group_ids):
            raise ValueError(
                "parameter_group_codes must contain exactly the target groups"
            )
        if any(type(code) is not int for code in codes.values()):
            raise TypeError("parameter-group codes must be integers")
        if any(code < 0 for code in codes.values()):
            raise ValueError("parameter-group codes must be nonnegative")
        if len(set(codes.values())) != len(codes):
            raise ValueError("parameter-group codes must be unique")

        object.__setattr__(self, "parameter_group_codes", codes)
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
            "parameter_group_codes": {
                target.parameter_group_id: self.parameter_group_codes[
                    target.parameter_group_id
                ]
                for target in self.simulation_targets
            },
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
class DatasetChunkPendingBatch:
    """An incomplete batch that can be resumed."""

    batch_id: int
    paths: BatchPaths
    assignments: tuple[AttemptAssignment, ...]


@dataclass(frozen=True)
class DatasetChunkState:
    """Chunk state reconstructed from validated batch files."""

    accepted_simulation_counts: Mapping[str, int]
    simulation_attempt_counts: Mapping[str, int]
    committed: tuple[BatchPaths, ...]
    pending_batch: DatasetChunkPendingBatch | None
    complete: bool


class BatchExecutor(Protocol):
    """Interface for a function or callable object that executes one batch."""

    def __call__(
        self,
        assignments: tuple[AttemptAssignment, ...],
        *,
        batch_id: int,
    ) -> BatchPaths: ...


def _find_saved_batch_ids(chunk_config: DatasetChunkConfig) -> tuple[int, ...]:
    """Find saved batch IDs and reject malformed names or gaps in the sequence."""

    template = BatchPaths.for_batch(
        chunk_config.root,
        family=chunk_config.family_name,
        split=chunk_config.dataset_split.value,
        batch_id=0,
    )
    batch_ids: set[int] = set()
    for artifact_name in ("batch_plan", "shard", "result", "failure"):
        directory = getattr(template, artifact_name).parent
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise RuntimeError(f"{artifact_name} artifact path is not a directory")
        for path in directory.glob("batch_*"):
            batch_id_text = path.stem.removeprefix("batch_")
            if not batch_id_text.isdecimal():
                raise RuntimeError(f"invalid batch artifact name: {path}")
            batch_id = int(batch_id_text)
            expected = getattr(
                BatchPaths.for_batch(
                    chunk_config.root,
                    family=chunk_config.family_name,
                    split=chunk_config.dataset_split.value,
                    batch_id=batch_id,
                ),
                artifact_name,
            )
            if path != expected:
                raise RuntimeError(f"noncanonical batch artifact path: {path}")
            batch_ids.add(batch_id)

    ordered_batch_ids = tuple(sorted(batch_ids))
    if ordered_batch_ids != tuple(range(len(ordered_batch_ids))):
        raise RuntimeError("batch artifact IDs must be contiguous from zero")
    return ordered_batch_ids


def _load_saved_batch_assignments(
    chunk_config: DatasetChunkConfig,
    paths: BatchPaths,
) -> tuple[AttemptAssignment, ...]:
    """Load a batch's saved assignments and verify they belong to this chunk."""

    batch_plan = load_npz(paths.batch_plan)
    mismatched_coordinates = []
    if int(batch_plan["family_id"].item()) != int(chunk_config.family_id):
        mismatched_coordinates.append("family_id")
    if str(batch_plan["dataset_split"].item()) != chunk_config.dataset_split.value:
        mismatched_coordinates.append("dataset_split")
    if mismatched_coordinates:
        raise RuntimeError(
            f"saved batch has mismatched coordinates: {tuple(mismatched_coordinates)}"
        )

    attempts = batch_plan["attempt_index"]
    if not np.all(batch_plan["worker_stream_id"] == chunk_config.worker_stream_id):
        raise RuntimeError("batch-plan worker_stream_id differs from the chunk")

    parameter_group_id_by_code = {
        code: parameter_group_id
        for parameter_group_id, code in chunk_config.parameter_group_codes.items()
    }
    parameter_group_codes = batch_plan["parameter_group_id"]
    unknown_codes = set(map(int, parameter_group_codes)).difference(
        parameter_group_id_by_code
    )
    if unknown_codes:
        raise RuntimeError(
            "batch plan contains unknown parameter-group codes: "
            f"{sorted(unknown_codes)}"
        )

    assignments: list[AttemptAssignment] = []
    for attempt_index, parameter_group_code in zip(
        attempts,
        parameter_group_codes,
        strict=True,
    ):
        simulation_key = SimulationKey(
            family_id=int(chunk_config.family_id),
            dataset_split=chunk_config.dataset_split,
            worker_stream_id=chunk_config.worker_stream_id,
            attempt_index=int(attempt_index),
        )
        assignments.append(
            AttemptAssignment(
                simulation_key=simulation_key,
                parameter_group_id=parameter_group_id_by_code[
                    int(parameter_group_code)
                ],
            )
        )
    return tuple(assignments)


def scan_dataset_generation(
    chunk_config: DatasetChunkConfig,
) -> DatasetChunkState:
    """Validate saved batches and determine what the chunk should do next."""

    accepted_simulation_counts = {
        target.parameter_group_id: 0 for target in chunk_config.simulation_targets
    }
    simulation_attempt_counts = {
        target.parameter_group_id: 0 for target in chunk_config.simulation_targets
    }
    committed: list[BatchPaths] = []
    pending_batch: DatasetChunkPendingBatch | None = None
    next_attempt_index = chunk_config.first_attempt_index
    batch_ids = _find_saved_batch_ids(chunk_config)

    for position, batch_id in enumerate(batch_ids):
        paths = BatchPaths.for_batch(
            chunk_config.root,
            family=chunk_config.family_name,
            split=chunk_config.dataset_split.value,
            batch_id=batch_id,
        )
        inspection = inspect_batch(paths)

        assignments = _load_saved_batch_assignments(
            chunk_config,
            paths,
        )
        expected = build_next_attempt_batch(
            chunk_config.simulation_targets,
            accepted_simulation_counts,
            simulation_attempt_counts,
            chunk_config.maximum_attempts_by_parameter_group,
            family_id=int(chunk_config.family_id),
            dataset_split=chunk_config.dataset_split,
            worker_stream_id=chunk_config.worker_stream_id,
            first_attempt_index=next_attempt_index,
            batch_size=chunk_config.batch_size,
        )
        if assignments != expected:
            raise RuntimeError(
                "batch-plan assignments differ from the deterministic simulation schedule"
            )
        next_attempt_index += len(assignments)
        for assignment in assignments:
            simulation_attempt_counts[assignment.parameter_group_id] += 1

        is_last = position == len(batch_ids) - 1
        if inspection.status is BatchStatus.COMMITTED:
            for assignment, simulation in zip(
                assignments,
                inspection.simulations,
                strict=True,
            ):
                if simulation.accepted:
                    accepted_simulation_counts[assignment.parameter_group_id] += 1
            committed.append(paths)
            continue
        if not is_last:
            raise RuntimeError("a nonterminal batch must be the final batch artifact")
        if inspection.status in (
            BatchStatus.PLAN_SAVED,
            BatchStatus.SHARD_WRITTEN,
        ):
            pending_batch = DatasetChunkPendingBatch(
                batch_id=batch_id,
                paths=paths,
                assignments=assignments,
            )
        elif inspection.status is BatchStatus.FAILED:
            raise RuntimeError(f"batch {batch_id} failed; see {paths.failure}")
        else:
            raise RuntimeError(f"unsupported batch status: {inspection.status.value}")

    complete = pending_batch is None and all(
        accepted_simulation_counts[target.parameter_group_id] == target.simulation_count
        for target in chunk_config.simulation_targets
    )
    return DatasetChunkState(
        accepted_simulation_counts=accepted_simulation_counts,
        simulation_attempt_counts=simulation_attempt_counts,
        committed=tuple(committed),
        pending_batch=pending_batch,
        complete=complete,
    )


@contextmanager
def dataset_generation_lock(chunk_config: DatasetChunkConfig) -> Iterator[None]:
    """Acquire a nonblocking single-writer lock for one family and split."""

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


def generate_simulations(
    chunk_config: DatasetChunkConfig,
    executor: BatchExecutor,
) -> DatasetChunkState:
    """Generate simulations, resuming an incomplete batch before scheduling more.

    Each executor call must save either a completed batch or a failure record.
    If it stops after saving only a plan or simulation rows, the next run resumes
    that batch with the same assignments.
    """

    with dataset_generation_lock(chunk_config):
        state = scan_dataset_generation(chunk_config)
        while True:
            if state.complete:
                return state

            # Finish the last incomplete batch before scheduling a new one.
            if state.pending_batch is not None:
                batch_id = state.pending_batch.batch_id
                assignments = state.pending_batch.assignments
                expected_paths = state.pending_batch.paths
            else:
                batch_id = len(state.committed)
                next_attempt_index = chunk_config.first_attempt_index + sum(
                    state.simulation_attempt_counts.values()
                )
                assignments = build_next_attempt_batch(
                    chunk_config.simulation_targets,
                    state.accepted_simulation_counts,
                    state.simulation_attempt_counts,
                    chunk_config.maximum_attempts_by_parameter_group,
                    family_id=int(chunk_config.family_id),
                    dataset_split=chunk_config.dataset_split,
                    worker_stream_id=chunk_config.worker_stream_id,
                    first_attempt_index=next_attempt_index,
                    batch_size=chunk_config.batch_size,
                )
                if not assignments:
                    attempt_limits = chunk_config.maximum_attempts_by_parameter_group
                    unfinished_groups = "; ".join(
                        (
                            f"{target.parameter_group_id}: accepted="
                            f"{state.accepted_simulation_counts[target.parameter_group_id]}/"
                            f"{target.simulation_count}, attempts="
                            f"{state.simulation_attempt_counts[target.parameter_group_id]}/"
                            f"{attempt_limits[target.parameter_group_id]}"
                        )
                        for target in chunk_config.simulation_targets
                        if state.accepted_simulation_counts[target.parameter_group_id]
                        < target.simulation_count
                    )
                    raise RuntimeError(
                        "attempt limits reached before all requested simulations were "
                        f"accepted; {unfinished_groups}"
                    )
                expected_paths = BatchPaths.for_batch(
                    chunk_config.root,
                    family=chunk_config.family_name,
                    split=chunk_config.dataset_split.value,
                    batch_id=batch_id,
                )

            returned_paths = executor(assignments, batch_id=batch_id)
            if returned_paths != expected_paths:
                raise RuntimeError("batch executor returned nonstandard paths")
            inspection = inspect_batch(expected_paths)
            if inspection.status not in (
                BatchStatus.COMMITTED,
                BatchStatus.FAILED,
            ):
                raise RuntimeError(
                    "batch executor returned without a terminal batch record"
                )
            persisted_assignments = _load_saved_batch_assignments(
                chunk_config,
                expected_paths,
            )
            if persisted_assignments != assignments:
                raise RuntimeError(
                    "persisted batch-plan assignments differ from the scheduled batch"
                )
            if inspection.status is BatchStatus.FAILED:
                raise RuntimeError(
                    f"batch {batch_id} failed; see {expected_paths.failure}"
                )

            simulation_attempt_counts = dict(state.simulation_attempt_counts)
            if state.pending_batch is None:
                for assignment in assignments:
                    simulation_attempt_counts[assignment.parameter_group_id] += 1
            accepted_simulation_counts = dict(state.accepted_simulation_counts)
            for assignment, simulation in zip(
                assignments,
                inspection.simulations,
                strict=True,
            ):
                if simulation.accepted:
                    accepted_simulation_counts[assignment.parameter_group_id] += 1
            state = DatasetChunkState(
                accepted_simulation_counts=accepted_simulation_counts,
                simulation_attempt_counts=simulation_attempt_counts,
                committed=(*state.committed, expected_paths),
                pending_batch=None,
                complete=all(
                    accepted_simulation_counts[target.parameter_group_id]
                    == target.simulation_count
                    for target in chunk_config.simulation_targets
                ),
            )
