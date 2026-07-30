"""Durable accepted-quota orchestration for paper-corpus families.

The numerical executor remains family-specific.  This module owns only the
outer transaction loop: recover committed counts, replay an interrupted batch,
schedule replacements in cells whose accepted quota is still short, and stop
only after every accepted-case quota is met or a terminal failure is present.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from solver.gen_data.pipeline.archive import (
    BatchPaths,
    BatchStatus,
    inspect_batch,
)
from solver.gen_data.pipeline.production import (
    AttemptAssignment,
    CaseKey,
    CellQuota,
    PhysicalFamilyId,
    SplitId,
    schedule_attempt_batch,
    split_code,
    split_root,
)


_BATCH_NAME = re.compile(r"batch_(\d+)\.(?:json|npz)")
_CASE_VECTOR_DTYPES = {
    "root_seed": np.dtype(np.uint64),
    "stream_id": np.dtype(np.uint32),
    "attempt_index": np.dtype(np.uint64),
}
DEFAULT_MAXIMUM_ATTEMPTS_PER_ACCEPTED_CASE = 4


def _strict_json_loads(text: str) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value!r}")

    return json.loads(text, parse_constant=reject_constant)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    """Return the SHA-256 digest of one strict canonical JSON value."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _strict_json_copy(
    value: Mapping[str, object],
) -> tuple[Mapping[str, object], str]:
    encoded = _canonical_json_bytes(dict(value)).decode("utf-8")
    parsed = _strict_json_loads(encoded)
    if not isinstance(parsed, dict):
        raise TypeError("configuration must encode a JSON object")
    frozen = _freeze_json(parsed)
    assert isinstance(frozen, Mapping)
    return frozen, encoded


@dataclass(frozen=True)
class AcceptedQuotaRunSpec:
    """Immutable identity, quota, and configuration of one production run."""

    root: Path
    family_name: str
    family_id: PhysicalFamilyId
    revision_id: int
    split_id: SplitId
    stream_id: int
    quotas: tuple[CellQuota, ...]
    cell_codes: Mapping[str, int]
    batch_size: int
    configuration: Mapping[str, object]
    first_attempt_index: int = 0
    maximum_attempts_per_accepted_case: int = (
        DEFAULT_MAXIMUM_ATTEMPTS_PER_ACCEPTED_CASE
    )
    _configuration_json: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        root = Path(self.root)
        quotas = tuple(self.quotas)
        if not quotas:
            raise ValueError("quotas must not be empty")
        cell_ids = tuple(quota.cell_id for quota in quotas)
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("quota cell_ids must be unique")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if (
            isinstance(self.maximum_attempts_per_accepted_case, bool)
            or not isinstance(self.maximum_attempts_per_accepted_case, int)
            or self.maximum_attempts_per_accepted_case <= 0
        ):
            raise ValueError(
                "maximum_attempts_per_accepted_case must be a positive integer"
            )
        if not isinstance(self.split_id, SplitId):
            raise TypeError("split_id must be a SplitId")
        if not isinstance(self.family_id, PhysicalFamilyId):
            raise TypeError("family_id must be a PhysicalFamilyId")
        if isinstance(self.revision_id, bool) or not isinstance(
            self.revision_id,
            int,
        ):
            raise TypeError("revision_id must be an integer")
        if isinstance(self.stream_id, bool) or not isinstance(self.stream_id, int):
            raise TypeError("stream_id must be an integer")
        if isinstance(self.first_attempt_index, bool) or not isinstance(
            self.first_attempt_index,
            int,
        ):
            raise TypeError("first_attempt_index must be an integer")

        # Validate path components and all packed case-key coordinates.
        BatchPaths.under(
            root,
            family=self.family_name,
            split=self.split_id.value,
            batch_id=0,
        )
        CaseKey(
            family_id=int(self.family_id),
            revision_id=self.revision_id,
            split_id=self.split_id,
            stream_id=self.stream_id,
            attempt_index=self.first_attempt_index,
        )
        maximum_attempt_count = self.maximum_attempts_per_accepted_case * sum(
            quota.target_accepted for quota in quotas
        )
        if maximum_attempt_count:
            CaseKey(
                family_id=int(self.family_id),
                revision_id=self.revision_id,
                split_id=self.split_id,
                stream_id=self.stream_id,
                attempt_index=(
                    self.first_attempt_index + maximum_attempt_count - 1
                ),
            )

        codes = dict(self.cell_codes)
        if set(codes) != set(cell_ids):
            raise ValueError("cell_codes must contain exactly the quota cells")
        if any(
            isinstance(code, bool) or not isinstance(code, int)
            for code in codes.values()
        ):
            raise TypeError("cell codes must be integers")
        if any(code < 0 for code in codes.values()):
            raise ValueError("cell codes must be nonnegative")
        if len(set(codes.values())) != len(codes):
            raise ValueError("cell codes must be unique")

        object.__setattr__(self, "root", root)
        object.__setattr__(self, "quotas", quotas)
        object.__setattr__(self, "cell_codes", MappingProxyType(codes))
        configuration, configuration_json = _strict_json_copy(self.configuration)
        object.__setattr__(self, "configuration", configuration)
        object.__setattr__(self, "_configuration_json", configuration_json)

    def to_json_record(self) -> dict[str, object]:
        """Return the complete run record bound by ``config_fingerprint``."""

        configuration = _strict_json_loads(self._configuration_json)
        assert isinstance(configuration, dict)
        record: dict[str, object] = {
            "schema": "paper_corpus_accepted_quota_run_v2",
            "family_name": self.family_name,
            "family_id": int(self.family_id),
            "revision_id": self.revision_id,
            "split_id": self.split_id.value,
            "root_seed": split_root(self.split_id),
            "stream_id": self.stream_id,
            "first_attempt_index": self.first_attempt_index,
            "batch_size": self.batch_size,
            "quotas": [
                {
                    "cell_id": quota.cell_id,
                    "target_accepted": quota.target_accepted,
                }
                for quota in self.quotas
            ],
            "cell_codes": {
                quota.cell_id: self.cell_codes[quota.cell_id] for quota in self.quotas
            },
            "configuration": configuration,
        }
        record["maximum_attempts_per_accepted_case"] = (
            self.maximum_attempts_per_accepted_case
        )
        return record

    @property
    def attempt_ceiling_by_cell(self) -> Mapping[str, int]:
        """Return each cell's immutable attempted-case ceiling."""

        multiplier = self.maximum_attempts_per_accepted_case
        return MappingProxyType(
            {
                quota.cell_id: multiplier * quota.target_accepted
                for quota in self.quotas
            }
        )

    @property
    def config_fingerprint(self) -> str:
        """Return the configuration digest required on every batch."""

        return canonical_json_sha256(self.to_json_record())


@dataclass(frozen=True)
class PendingBatch:
    """A durable nonterminal batch that must be replayed unchanged."""

    batch_id: int
    paths: BatchPaths
    status: BatchStatus
    assignments: tuple[AttemptAssignment, ...]

    def __post_init__(self) -> None:
        if self.status not in (BatchStatus.PROPOSED, BatchStatus.SHARD_WRITTEN):
            raise ValueError("pending status must be proposed or shard_written")


@dataclass(frozen=True)
class AttemptLimitFailure:
    """Terminal reason for a deficient cell that exhausted its attempt cap."""

    exhausted_cells: tuple[str, ...]
    message: str

    def __post_init__(self) -> None:
        if not self.exhausted_cells:
            raise ValueError("exhausted_cells must not be empty")
        if not self.message:
            raise ValueError("attempt-limit failure message must not be empty")


@dataclass(frozen=True)
class QuotaRunState:
    """State reconstructed exclusively from validated durable artifacts."""

    accepted_by_cell: Mapping[str, int]
    attempted_by_cell: Mapping[str, int]
    next_batch_id: int
    next_attempt_index: int
    committed: tuple[BatchPaths, ...]
    pending: PendingBatch | None
    terminal_failure: BatchPaths | None
    attempt_limit_failure: AttemptLimitFailure | None
    complete: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "accepted_by_cell",
            MappingProxyType(dict(self.accepted_by_cell)),
        )
        object.__setattr__(
            self,
            "attempted_by_cell",
            MappingProxyType(dict(self.attempted_by_cell)),
        )


class BatchExecutor(Protocol):
    """Family-specific callback that resolves one exact proposed batch."""

    def __call__(
        self,
        assignments: tuple[AttemptAssignment, ...],
        *,
        batch_id: int,
    ) -> BatchPaths: ...


def _read_npz(path: Path) -> dict[str, NDArray[object]]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _read_json_object(path: Path) -> dict[str, object]:
    value = _strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _artifact_directories(
    spec: AcceptedQuotaRunSpec,
) -> tuple[tuple[str, Path], ...]:
    template = BatchPaths.under(
        spec.root,
        family=spec.family_name,
        split=spec.split_id.value,
        batch_id=0,
    )
    return (
        ("proposal", template.proposal.parent),
        ("shard", template.shard.parent),
        ("result", template.result.parent),
        ("failure", template.failure.parent),
    )


def _discover_batch_ids(spec: AcceptedQuotaRunSpec) -> tuple[int, ...]:
    batch_ids: set[int] = set()
    for artifact_name, directory in _artifact_directories(spec):
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise RuntimeError(f"{artifact_name} artifact path is not a directory")
        for path in directory.glob("batch_*"):
            match = _BATCH_NAME.fullmatch(path.name)
            if match is None:
                raise RuntimeError(f"invalid batch artifact name: {path}")
            batch_id = int(match.group(1))
            expected = getattr(
                BatchPaths.under(
                    spec.root,
                    family=spec.family_name,
                    split=spec.split_id.value,
                    batch_id=batch_id,
                ),
                artifact_name,
            )
            if path != expected:
                raise RuntimeError(f"noncanonical batch artifact path: {path}")
            batch_ids.add(batch_id)

    ordered = tuple(sorted(batch_ids))
    if ordered and any(batch_id != index for index, batch_id in enumerate(ordered)):
        raise RuntimeError("batch artifact IDs must be contiguous from zero")
    return ordered


def _required_case_vector(
    proposal: Mapping[str, NDArray[object]],
    name: str,
    case_count: int,
) -> NDArray[object]:
    if name not in proposal:
        raise ValueError(f"proposal is missing required run coordinate {name!r}")
    values = proposal[name]
    expected_dtype = _CASE_VECTOR_DTYPES[name]
    if values.dtype != expected_dtype:
        raise TypeError(
            f"proposal field {name!r} has dtype {values.dtype}; "
            f"expected {expected_dtype}"
        )
    if values.shape != (case_count,):
        raise ValueError(f"proposal field {name!r} must have one value per case")
    return values


def _proposal_assignments(
    spec: AcceptedQuotaRunSpec,
    paths: BatchPaths,
    *,
    batch_id: int,
) -> tuple[AttemptAssignment, ...]:
    proposal = _read_npz(paths.proposal)
    if int(np.asarray(proposal["batch_id"]).item()) != batch_id:
        raise RuntimeError("proposal batch_id differs from its artifact name")
    if int(np.asarray(proposal["family_id"]).item()) != int(spec.family_id):
        raise RuntimeError("proposal family_id differs from the run")
    if int(np.asarray(proposal["revision_id"]).item()) != spec.revision_id:
        raise RuntimeError("proposal revision_id differs from the run")
    if int(np.asarray(proposal["split_id"]).item()) != split_code(spec.split_id):
        raise RuntimeError("proposal split_id differs from the run")

    case_ids = np.asarray(proposal["case_id"], dtype=np.int64)
    case_count = int(case_ids.size)
    roots = _required_case_vector(proposal, "root_seed", case_count)
    streams = _required_case_vector(proposal, "stream_id", case_count)
    attempts = _required_case_vector(proposal, "attempt_index", case_count)
    if not np.all(roots == split_root(spec.split_id)):
        raise RuntimeError("proposal root_seed differs from the run")
    if not np.all(streams == spec.stream_id):
        raise RuntimeError("proposal stream_id differs from the run")

    code_to_cell = {code: cell for cell, code in spec.cell_codes.items()}
    encoded_cells = np.asarray(proposal["cell_id"], dtype=np.int32)
    unknown_codes = set(map(int, encoded_cells)).difference(code_to_cell)
    if unknown_codes:
        raise RuntimeError(
            f"proposal contains unknown cell codes: {sorted(unknown_codes)}"
        )

    assignments = tuple(
        AttemptAssignment(
            case_key=CaseKey(
                family_id=int(spec.family_id),
                revision_id=spec.revision_id,
                split_id=spec.split_id,
                stream_id=spec.stream_id,
                attempt_index=int(attempt_index),
            ),
            cell_id=code_to_cell[int(cell_code)],
        )
        for attempt_index, cell_code in zip(attempts, encoded_cells)
    )
    expected_case_ids = np.asarray(
        [assignment.case_key.case_id for assignment in assignments],
        dtype=np.int64,
    )
    if not np.array_equal(case_ids, expected_case_ids):
        raise RuntimeError("proposal case_id differs from its run coordinates")
    return assignments


def _require_result_case(
    value: object,
    *,
    expected_case_id: int,
) -> tuple[bool, int, int]:
    if not isinstance(value, dict):
        raise TypeError("every result case must be a JSON object")
    case_id = value.get("case_id")
    if (
        isinstance(case_id, bool)
        or not isinstance(case_id, int)
        or case_id != expected_case_id
    ):
        raise RuntimeError("result case identity differs from its proposal")
    accepted = value.get("accepted")
    if not isinstance(accepted, bool):
        raise TypeError("result accepted field must be boolean")

    integer_fields: dict[str, int] = {}
    for name in (
        "required_bits",
        "evaluated_bits",
        "failed_bits",
        "first_row",
        "row_count",
    ):
        field = value.get(name)
        if isinstance(field, bool) or not isinstance(field, int):
            raise TypeError(f"result {name} field must be an integer")
        integer_fields[name] = field

    required = integer_fields["required_bits"]
    evaluated = integer_fields["evaluated_bits"]
    failed = integer_fields["failed_bits"]
    if any(not 0 <= bits < 1 << 32 for bits in (required, evaluated, failed)):
        raise ValueError("result quality masks must fit in uint32")
    if failed & ~evaluated:
        raise ValueError("result failed_bits must be a subset of evaluated_bits")
    mask_accepts = not (required & ~evaluated or required & failed)
    if accepted != mask_accepts:
        raise RuntimeError("result accepted field disagrees with quality masks")

    first_row = integer_fields["first_row"]
    row_count = integer_fields["row_count"]
    if accepted:
        if first_row < 0 or row_count <= 0:
            raise RuntimeError("accepted result must own at least one shard row")
    elif first_row != -1 or row_count != 0:
        raise RuntimeError("rejected result cannot own shard rows")
    return accepted, first_row, row_count


def _shard_row_blocks(paths: BatchPaths) -> dict[int, tuple[int, int]]:
    if not paths.shard.exists():
        return {}
    shard = _read_npz(paths.shard)
    local_indices = np.asarray(shard["case_local_index"], dtype=np.int32)
    return {
        int(local_index): (
            int(np.flatnonzero(local_indices == local_index)[0]),
            int(np.count_nonzero(local_indices == local_index)),
        )
        for local_index in np.unique(local_indices)
    }


def _accepted_cells(
    paths: BatchPaths,
    assignments: Sequence[AttemptAssignment],
) -> tuple[str, ...]:
    result = _read_json_object(paths.result)
    if result.get("schema") != "paper_corpus_batch_result_v1":
        raise RuntimeError("result has an unknown schema")
    values = result.get("cases")
    if not isinstance(values, list) or len(values) != len(assignments):
        raise RuntimeError("result must contain every proposed case")

    blocks = _shard_row_blocks(paths)
    accepted_cells: list[str] = []
    for local_index, (assignment, value) in enumerate(zip(assignments, values)):
        accepted, first_row, row_count = _require_result_case(
            value,
            expected_case_id=assignment.case_key.case_id,
        )
        expected_block = blocks.get(local_index)
        observed_block = (first_row, row_count) if accepted else None
        if observed_block != expected_block:
            raise RuntimeError("result row ownership differs from its shard")
        if accepted:
            accepted_cells.append(assignment.cell_id)
    return tuple(accepted_cells)


def _expected_assignments(
    spec: AcceptedQuotaRunSpec,
    accepted_by_cell: Mapping[str, int],
    attempted_by_cell: Mapping[str, int],
    *,
    first_attempt_index: int,
) -> tuple[AttemptAssignment, ...]:
    ceilings = spec.attempt_ceiling_by_cell
    over_limit = tuple(
        quota.cell_id
        for quota in spec.quotas
        if attempted_by_cell[quota.cell_id] > ceilings[quota.cell_id]
    )
    if over_limit:
        raise RuntimeError(
            "attempted-case count exceeds its immutable per-cell ceiling for "
            + ", ".join(over_limit)
        )
    effective_quotas = tuple(
        CellQuota(
            quota.cell_id,
            accepted_by_cell[quota.cell_id]
            + min(
                quota.target_accepted - accepted_by_cell[quota.cell_id],
                ceilings[quota.cell_id] - attempted_by_cell[quota.cell_id],
            ),
        )
        for quota in spec.quotas
    )
    return schedule_attempt_batch(
        effective_quotas,
        accepted_by_cell,
        family_id=int(spec.family_id),
        revision_id=spec.revision_id,
        split_id=spec.split_id,
        stream_id=spec.stream_id,
        first_attempt_index=first_attempt_index,
        batch_size=spec.batch_size,
    )


def _add_attempts(
    attempted_by_cell: Mapping[str, int],
    assignments: Sequence[AttemptAssignment],
) -> dict[str, int]:
    """Return attempted counts after assigning one durable batch."""

    updated = dict(attempted_by_cell)
    for assignment in assignments:
        updated[assignment.cell_id] += 1
    return updated


def _attempt_limit_failure(
    spec: AcceptedQuotaRunSpec,
    accepted_by_cell: Mapping[str, int],
    attempted_by_cell: Mapping[str, int],
    *,
    pending: PendingBatch | None,
    terminal_failure: BatchPaths | None,
    complete: bool,
) -> AttemptLimitFailure | None:
    """Return the deterministic cap-exhaustion terminal state, if any."""

    ceilings = spec.attempt_ceiling_by_cell
    if complete or pending is not None or terminal_failure is not None:
        return None

    over_limit = tuple(
        quota.cell_id
        for quota in spec.quotas
        if attempted_by_cell[quota.cell_id] > ceilings[quota.cell_id]
    )
    if over_limit:
        raise RuntimeError(
            "persisted attempted-case count exceeds its immutable per-cell "
            f"ceiling for {', '.join(over_limit)}"
        )

    exhausted = tuple(
        quota.cell_id
        for quota in spec.quotas
        if (
            accepted_by_cell[quota.cell_id] < quota.target_accepted
            and attempted_by_cell[quota.cell_id] == ceilings[quota.cell_id]
        )
    )
    if not exhausted:
        return None

    targets = {quota.cell_id: quota.target_accepted for quota in spec.quotas}
    details = "; ".join(
        (
            f"{cell_id}: attempted={attempted_by_cell[cell_id]}, "
            f"ceiling={ceilings[cell_id]}, "
            f"accepted={accepted_by_cell[cell_id]}, "
            f"target={targets[cell_id]}"
        )
        for cell_id in exhausted
    )
    return AttemptLimitFailure(
        exhausted_cells=exhausted,
        message=(
            "per-cell attempted-case ceiling reached before the accepted "
            f"quota; {details}"
        ),
    )


def scan_quota_run(spec: AcceptedQuotaRunSpec) -> QuotaRunState:
    """Validate all run artifacts and reconstruct the next durable action."""

    accepted_by_cell = {quota.cell_id: 0 for quota in spec.quotas}
    attempted_by_cell = {quota.cell_id: 0 for quota in spec.quotas}
    committed: list[BatchPaths] = []
    pending: PendingBatch | None = None
    terminal_failure: BatchPaths | None = None
    next_attempt_index = spec.first_attempt_index
    batch_ids = _discover_batch_ids(spec)

    for position, batch_id in enumerate(batch_ids):
        prefix_complete = all(
            accepted_by_cell[quota.cell_id] == quota.target_accepted
            for quota in spec.quotas
        )
        prefix_attempt_failure = _attempt_limit_failure(
            spec,
            accepted_by_cell,
            attempted_by_cell,
            pending=None,
            terminal_failure=None,
            complete=prefix_complete,
        )
        if prefix_complete:
            raise RuntimeError("batch artifact exists after quota completion")
        if prefix_attempt_failure is not None:
            raise RuntimeError(
                "batch artifact exists after attempted-case ceiling exhaustion"
            )

        paths = BatchPaths.under(
            spec.root,
            family=spec.family_name,
            split=spec.split_id.value,
            batch_id=batch_id,
        )
        inspection = inspect_batch(
            paths,
            expected_fingerprint=spec.config_fingerprint,
        )
        if inspection.status is BatchStatus.EMPTY:
            raise RuntimeError("discovered batch has no durable artifacts")

        assignments = _proposal_assignments(
            spec,
            paths,
            batch_id=batch_id,
        )
        expected = _expected_assignments(
            spec,
            accepted_by_cell,
            attempted_by_cell,
            first_attempt_index=next_attempt_index,
        )
        if assignments != expected:
            raise RuntimeError(
                "proposal assignments differ from the deterministic quota schedule"
            )
        next_attempt_index += len(assignments)
        attempted_by_cell = _add_attempts(
            attempted_by_cell,
            assignments,
        )
        if (
            sum(attempted_by_cell.values())
            != next_attempt_index - spec.first_attempt_index
        ):
            raise RuntimeError(
                "attempted-case counts disagree with the attempt-index prefix"
            )

        is_last = position == len(batch_ids) - 1
        if inspection.status is BatchStatus.COMMITTED:
            for cell_id in _accepted_cells(paths, assignments):
                accepted_by_cell[cell_id] += 1
            for quota in spec.quotas:
                if accepted_by_cell[quota.cell_id] > quota.target_accepted:
                    raise RuntimeError(
                        f"accepted count exceeds quota for cell {quota.cell_id!r}"
                    )
            committed.append(paths)
            continue
        if not is_last:
            raise RuntimeError("a nonterminal batch must be the final batch artifact")
        if inspection.status in (
            BatchStatus.PROPOSED,
            BatchStatus.SHARD_WRITTEN,
        ):
            pending = PendingBatch(
                batch_id=batch_id,
                paths=paths,
                status=inspection.status,
                assignments=assignments,
            )
        elif inspection.status is BatchStatus.FAILED:
            terminal_failure = paths
        else:
            raise RuntimeError(f"unsupported batch status: {inspection.status.value}")

    complete = (
        pending is None
        and terminal_failure is None
        and all(
            accepted_by_cell[quota.cell_id] == quota.target_accepted
            for quota in spec.quotas
        )
    )
    attempt_limit_failure = _attempt_limit_failure(
        spec,
        accepted_by_cell,
        attempted_by_cell,
        pending=pending,
        terminal_failure=terminal_failure,
        complete=complete,
    )
    return QuotaRunState(
        accepted_by_cell=accepted_by_cell,
        attempted_by_cell=attempted_by_cell,
        next_batch_id=(batch_ids[-1] + 1) if batch_ids else 0,
        next_attempt_index=next_attempt_index,
        committed=tuple(committed),
        pending=pending,
        terminal_failure=terminal_failure,
        attempt_limit_failure=attempt_limit_failure,
        complete=complete,
    )


def _lock_path(spec: AcceptedQuotaRunSpec) -> Path:
    return spec.root / ".locks" / spec.family_name / f"{spec.split_id.value}.lock"


@contextmanager
def quota_run_lock(spec: AcceptedQuotaRunSpec) -> Iterator[None]:
    """Acquire a nonblocking single-writer lock for one family and split."""

    path = _lock_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another accepted-quota writer holds {path}") from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _advance_terminal_batch(
    spec: AcceptedQuotaRunSpec,
    state: QuotaRunState,
    paths: BatchPaths,
    assignments: tuple[AttemptAssignment, ...],
    *,
    batch_id: int,
    status: BatchStatus,
) -> QuotaRunState:
    """Advance one validated in-memory state after a terminal batch commit.

    The initial ``scan_quota_run`` establishes the durable prefix under the
    single-writer lock.  Every later assignment is constructed from that state,
    and this function validates the newly persisted proposal before extending
    the prefix.  A process interruption still leaves the batch to be recovered
    by a complete scan on the next invocation.
    """

    persisted_assignments = _proposal_assignments(
        spec,
        paths,
        batch_id=batch_id,
    )
    if persisted_assignments != assignments:
        raise RuntimeError(
            "persisted proposal assignments differ from the scheduled batch"
        )

    next_attempt_index = state.next_attempt_index
    attempted_by_cell = dict(state.attempted_by_cell)
    if state.pending is None:
        next_attempt_index += len(assignments)
        attempted_by_cell = _add_attempts(
            attempted_by_cell,
            assignments,
        )
    next_batch_id = max(state.next_batch_id, batch_id + 1)

    if status is BatchStatus.FAILED:
        return QuotaRunState(
            accepted_by_cell=state.accepted_by_cell,
            attempted_by_cell=attempted_by_cell,
            next_batch_id=next_batch_id,
            next_attempt_index=next_attempt_index,
            committed=state.committed,
            pending=None,
            terminal_failure=paths,
            attempt_limit_failure=None,
            complete=False,
        )
    if status is not BatchStatus.COMMITTED:
        raise ValueError("in-memory advancement requires a terminal batch")

    accepted_by_cell = dict(state.accepted_by_cell)
    for cell_id in _accepted_cells(paths, assignments):
        accepted_by_cell[cell_id] += 1
    for quota in spec.quotas:
        if accepted_by_cell[quota.cell_id] > quota.target_accepted:
            raise RuntimeError(
                f"accepted count exceeds quota for cell {quota.cell_id!r}"
            )
    complete = all(
        accepted_by_cell[quota.cell_id] == quota.target_accepted
        for quota in spec.quotas
    )
    attempt_limit_failure = _attempt_limit_failure(
        spec,
        accepted_by_cell,
        attempted_by_cell,
        pending=None,
        terminal_failure=None,
        complete=complete,
    )
    return QuotaRunState(
        accepted_by_cell=accepted_by_cell,
        attempted_by_cell=attempted_by_cell,
        next_batch_id=next_batch_id,
        next_attempt_index=next_attempt_index,
        committed=(*state.committed, paths),
        pending=None,
        terminal_failure=None,
        attempt_limit_failure=attempt_limit_failure,
        complete=complete,
    )


def run_accepted_quotas(
    spec: AcceptedQuotaRunSpec,
    executor: BatchExecutor,
) -> QuotaRunState:
    """Replay or execute batches until quotas are met or a failure is terminal.

    The executor must durably resolve the supplied batch as either ``COMMITTED``
    or ``FAILED`` and return its standard paths.  If it raises after writing a
    proposal or shard, that nonterminal state is deliberately left for exact
    replay on the next invocation.
    """

    with quota_run_lock(spec):
        state = scan_quota_run(spec)
        while True:
            if (
                state.complete
                or state.terminal_failure is not None
                or state.attempt_limit_failure is not None
            ):
                return state

            if state.pending is not None:
                batch_id = state.pending.batch_id
                assignments = state.pending.assignments
                expected_paths = state.pending.paths
            else:
                batch_id = state.next_batch_id
                assignments = _expected_assignments(
                    spec,
                    state.accepted_by_cell,
                    state.attempted_by_cell,
                    first_attempt_index=state.next_attempt_index,
                )
                if not assignments:
                    raise RuntimeError(
                        "quota schedule is empty before the run is complete"
                    )
                expected_paths = BatchPaths.under(
                    spec.root,
                    family=spec.family_name,
                    split=spec.split_id.value,
                    batch_id=batch_id,
                )

            returned_paths = executor(assignments, batch_id=batch_id)
            if returned_paths != expected_paths:
                raise RuntimeError("batch executor returned nonstandard paths")
            inspection = inspect_batch(
                expected_paths,
                expected_fingerprint=spec.config_fingerprint,
            )
            if inspection.status not in (
                BatchStatus.COMMITTED,
                BatchStatus.FAILED,
            ):
                raise RuntimeError(
                    "batch executor returned without a terminal batch record"
                )
            state = _advance_terminal_batch(
                spec,
                state,
                expected_paths,
                assignments,
                batch_id=batch_id,
                status=inspection.status,
            )
