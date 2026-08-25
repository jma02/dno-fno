#!/usr/bin/env python3
"""Watch progress recorded by the long-running paper-dataset workers.

The numerical generators write proposal, shard, and result files once.
This process watches for those files without importing or changing
the source-fingerprinted generator.  Only exact commands from the current
revision-4 JONSWAP and revision-3 Tanaka launch plans, owned by the current
user and rooted in this checkout, are eligible for a signal.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import lzma
import math
import os
from pathlib import Path
import re
import signal
import stat
import struct
import sys
import time
from typing import Any, Callable, Final, Literal, Mapping, Protocol, Sequence
import zipfile
import zlib


ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON: Final = ROOT / ".venv/bin/python"
DEFAULT_JONSWAP_ROOT: Final = (
    ROOT / "outputs/paper_dataset_jonswap_revision4_relative_band_v1"
)
DEFAULT_TANAKA_ROOT: Final = (
    ROOT / "outputs/paper_dataset_revision3_literature_aligned_v1"
)
DEFAULT_FINAL_VIEW_SUMMARY: Final = (
    ROOT
    / "outputs/paper_dataset_literature_aligned_v1"
    / "combined/c16384_v01024_t01024"
    / "paper_dataset_all_splits_c16384.summary.json"
)
DEFAULT_LOG: Final = ROOT / "outputs/paper_dataset_worker_watchdog.log"

Family = Literal["jonswap_tma", "tanaka"]
Phase = Literal["running", "term_sent", "kill_sent"]

FINAL_VIEW_RELATIVE_PATH: Final = Path(
    "outputs/paper_dataset_literature_aligned_v1/combined/"
    "c16384_v01024_t01024/paper_dataset_all_splits_c16384.summary.json"
)
FINAL_VIEW_NAME: Final = "paper_dataset_all_splits_c16384"
FINAL_SOURCE_COUNT: Final = 26
FINAL_ACCEPTED_CASES: Final = 73_728
FINAL_RETAINED_ROWS: Final = 7_686_144
FINAL_DATASET_TARGET: Final = {
    "role": "paper_dataset",
    "nx": 1024,
    "length": 2.0 * math.pi,
    "gravity": 1.0,
    "dno_order": 6,
    "pad_factor": 8,
    "maximum_wavenumber": 128.0,
    "dtype": "float64",
}
FINAL_FAMILY_ORDER: Final = (
    "stokes",
    "tanaka",
    "benjamin_feir",
    "jonswap_tma",
)
FINAL_SPLIT_ORDER: Final = ("train", "validation", "test")
FINAL_FAMILY_IDS: Final = {
    "stokes": 1,
    "tanaka": 2,
    "benjamin_feir": 3,
    "jonswap_tma": 4,
}
FINAL_REVISIONS: Final = {
    "stokes": 2,
    "tanaka": 3,
    "benjamin_feir": 4,
    "jonswap_tma": 4,
}
FINAL_ROWS_PER_CASE: Final = {
    "stokes": 1,
    "tanaka": 200,
    "benjamin_feir": 200,
    "jonswap_tma": 16,
}
FINAL_BATCH_SIZES: Final = {
    "stokes": 32,
    "tanaka": 256,
    "benjamin_feir": 256,
    "jonswap_tma": 32,
}
FINAL_CASES_BY_SPLIT: Final = {
    "train": 16_384,
    "validation": 1_024,
    "test": 1_024,
}
FINAL_ROOT_SEEDS: Final = {
    "train": 2_026_072_210,
    "validation": 2_026_072_204,
    "test": 2_026_072_205,
}
FINAL_ORDERED_CELL_IDENTITIES: Final = {
    "stokes": (
        4,
        "8dfb5cb921f33365bee13c0ace07e6ff3d234a0f2c1b018cc247e84522dd8857",
    ),
    "tanaka": (
        11,
        "381256204e1c1ed7f3adee50f5a30ccef4e1d5b7081614f277c3271b011732eb",
    ),
    "benjamin_feir": (
        66,
        "26aa69528d19470a635588b415da15dcb78a4f23094ff1f1232e93369212fc92",
    ),
    "jonswap_tma": (
        27,
        "6e6b0733e1c72cfdd20f4c5f3381b5663d69ef635019d7bd43f69ce314eeccb5",
    ),
}
FINAL_SOURCE_ROOTS: Final = {
    "stokes": Path("outputs/paper_dataset_cap4_revision2_20260728"),
    "tanaka": Path("outputs/paper_dataset_revision3_literature_aligned_v1"),
    "benjamin_feir": Path(
        "outputs/paper_dataset_bf_revision4_jonswap_revision3_literature_aligned_v1"
    ),
    "jonswap_tma": Path("outputs/paper_dataset_jonswap_revision4_relative_band_v1"),
}
FINAL_STANDARD_TRAIN_LAYOUT: Final = (
    (0, 2_048, 0),
    (2_048, 2_048, 1),
    (4_096, 4_096, 2),
    (8_192, 8_192, 3),
)
FINAL_JONSWAP_TRAIN_LAYOUT: Final = (
    (0, 2_048, 0),
    (2_048, 2_048, 1),
    (4_096, 4_096, 2),
    (8_192, 4_096, 3),
    (12_288, 2_048, 4),
    (14_336, 2_048, 5),
)
SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
FINAL_CURRENT_EXECUTION_FINGERPRINTS: Final = {
    "stokes": ("4799045ea20b6fb91cc9efc093bac09346fed5e37461018f3b6d0b908bdd78d6"),
    "tanaka": ("73811acc430da1a1a3a30c18fa4b34b89e9e5756206adaf34975468f4931d963"),
    "benjamin_feir": (
        "8bf9403c21241e7c7701b04193420cca9bc3f8335f2dae4ea93f095f1a91b998"
    ),
    "jonswap_tma": ("ede0b41b4c2a1c9bc028b5ad6748f5ddc381ffdc4a866f1ce717e9e21e003ebd"),
}
FINAL_BENJAMIN_FEIR_SAMPLING_SUPPORT_FINGERPRINT: Final = (
    "2c8fc8cfa10e4c7ec071f0d18d974546490fd6a82b00d5f69cc58b8a1c3be416"
)
FINAL_DEPENDENCY_FINGERPRINT: Final = (
    "d966522b758f27d09a55ebd84c06dbd745ed7b14e5842c457d1cbaeb964f2f70"
)
FINAL_SOURCE_FINGERPRINTS: Final = {
    "stokes": "418e799c12d08d9f79046f4d5f1c116a3c814eed15a6652d33dae8000f14ed8f",
    "tanaka": "478eccd45edc683b137db17736f07c09741edc9f5f3f01effdbebead2bbb53f7",
    "benjamin_feir": (
        "d1ba68f08a5316d0b9ce0e399d48896d766044615504e461592eaee7ff3325ce"
    ),
    "jonswap_tma": ("84c3d88b0a06834839480a393218d376d9a52e84f4163caa48577c03a9a64bdf"),
}
QUALITY_REASON_ORDER: Final = (
    "NONFINITE_STATE",
    "NONFINITE_TARGET",
    "BOTTOM_CLEARANCE",
    "HAMILTONIAN_DRIFT",
    "GL2_STAGE_RESIDUAL",
    "TEMPORAL_DEFECT",
    "IC_GRID_DEFECT",
    "LABEL_GRID_DEFECT",
    "OUTSIDE_SUPPORT",
    "INCOMPLETE_TRAJECTORY",
)
FINAL_TRAJECTORY_MAP_DTYPES: Final = {
    "schema_version": ("<i2", 2, "schema"),
    "trajectory_index": ("<i4", 4, "row"),
    "frame_index": ("<i4", 4, "row"),
    "shard_index": ("<i4", 4, "row"),
    "shard_row": ("<i8", 8, "row"),
    "trajectory_family_id": ("<i2", 2, "trajectory"),
    "trajectory_revision_id": ("<i2", 2, "trajectory"),
    "trajectory_split_id": ("|u1", 1, "trajectory"),
    "trajectory_case_id": ("<i8", 8, "trajectory"),
    "trajectory_cell_id": ("<i4", 4, "trajectory"),
    "trajectory_accepted": ("|b1", 1, "trajectory"),
    "trajectory_required_bits": ("<u4", 4, "trajectory"),
    "trajectory_evaluated_bits": ("<u4", 4, "trajectory"),
    "trajectory_failed_bits": ("<u4", 4, "trajectory"),
    "trajectory_first_row": ("<i8", 8, "trajectory"),
    "trajectory_row_count": ("<i4", 4, "trajectory"),
}
TANAKA_CURRENT_EXECUTION_FINGERPRINT: Final = FINAL_CURRENT_EXECUTION_FINGERPRINTS[
    "tanaka"
]
TANAKA_LEGACY_EXECUTION_FINGERPRINT: Final = (
    "e8b7c0ff42240f2d0a9257fa55c71150fabc5fb66efc2704735dcf2e1e38ab4d"
)
TANAKA_LEGACY_SOURCE_FINGERPRINT: Final = (
    "a1cf8a3b2649e21161a15bf26d1bb6388f2b7f93996b1c4cf7e3c297d1b98d0a"
)


@dataclass(frozen=True)
class WorkerPlan:
    """One exact numerical command which the watchdog may supervise."""

    family: Family
    lane: int
    gpu_index: str
    split: str
    output_root: Path
    argv: tuple[str, ...]


@dataclass(frozen=True)
class ProcessSnapshot:
    """Stable identity and matching fields read from one procfs process."""

    pid: int
    start_ticks: int
    uid: int
    cwd: Path
    argv: tuple[str, ...]
    environment: Mapping[str, str]


@dataclass(frozen=True)
class MatchedWorker:
    """A process proven to equal one declared worker plan."""

    process: ProcessSnapshot
    plan: WorkerPlan


@dataclass(frozen=True)
class ProcessIdentity:
    """PID-reuse-safe state key."""

    pid: int
    start_ticks: int
    family: Family
    output_root: Path


@dataclass(frozen=True)
class FileToken:
    """Identity of the newest immutable file in one transaction directory."""

    relative_path: str
    mtime_ns: int
    size: int


@dataclass(frozen=True)
class FinalChunkPlan:
    """One exact source-summary slot in the canonical final combined view."""

    family: str
    family_id: int
    revision_id: int
    split: str
    split_id: int
    accepted_before: int
    accepted_count: int
    stream_id: int
    root: Path
    summary_path: Path


@dataclass(frozen=True)
class FinalChunkFacts:
    """Cheap authenticated facts recovered from one immutable source summary."""

    attempted_count: int
    configuration_fingerprint: str
    dependency_fingerprint: str
    execution: Mapping[str, Any]
    execution_fingerprint: str
    generation_compatibility_id: str | None
    source_fingerprint: str
    source_sha256: Mapping[str, str]
    execution_platform: str
    target: Mapping[str, object]
    trajectory_numerical: Mapping[str, object] | None
    batch_size: int
    committed_batches: int
    initial_accepted_count: int
    initial_attempted_count: int
    initial_committed_batches: int
    initial_next_batch_id: int
    initial_pending_attempt_count: int


@dataclass(frozen=True)
class AuthenticatedFile:
    """One stable-read physical artifact retained for terminal reauthentication."""

    path: Path
    size: int
    sha256: str
    identity: tuple[int, int, int, int, int]
    data: bytes | None


@dataclass(frozen=True)
class HeartbeatToken:
    """Newest saved proposal, result, and shard for one active chunk."""

    proposal: FileToken | None
    result: FileToken | None
    shard: FileToken | None


@dataclass
class WorkerState:
    """Monotonic liveness state for one process identity."""

    worker: MatchedWorker
    token: HeartbeatToken
    last_progress_at: float
    last_log_at: float
    phase: Phase = "running"
    kill_due_at: float | None = None


class ProcessSource(Protocol):
    """Read current processes without exposing procfs details to the engine."""

    def snapshots(self) -> tuple[ProcessSnapshot, ...]: ...

    def read_process(self, pid: int) -> ProcessSnapshot | None: ...


class SignalSender(Protocol):
    """Signal only after revalidating a previously matched process."""

    def send(self, worker: MatchedWorker, signal_number: int) -> bool: ...


class LogSink(Protocol):
    """Minimal logger interface used by the testable state machine."""

    def write(self, message: str) -> None: ...


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _absolute_without_resolving_symlinks(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _jonswap_argv(
    python: Path,
    *,
    split: str,
    count: int,
    accepted_before: int,
    stream: int,
    output_root: Path,
) -> tuple[str, ...]:
    return (
        str(python),
        "scripts/generate_paper_dataset_jonswap.py",
        "--solver-batch-size",
        "8",
        "--family",
        "jonswap_tma",
        "--split",
        split,
        "--accepted-cases",
        str(count),
        "--accepted-cases-before",
        str(accepted_before),
        "--stream-id",
        str(stream),
        "--first-attempt-index",
        "0",
        "--batch-size",
        "32",
        "--maximum-attempts-per-accepted-case",
        "4",
        "--platform",
        "gpu",
        "--output-root",
        str(output_root),
        "--execute",
    )


def _tanaka_argv(
    python: Path,
    *,
    split: str,
    count: int,
    accepted_before: int,
    stream: int,
    output_root: Path,
) -> tuple[str, ...]:
    return (
        str(python),
        "scripts/generate_paper_dataset.py",
        "--family",
        "tanaka",
        "--split",
        split,
        "--accepted-cases",
        str(count),
        "--accepted-cases-before",
        str(accepted_before),
        "--stream-id",
        str(stream),
        "--first-attempt-index",
        "0",
        "--batch-size",
        "256",
        "--maximum-attempts-per-accepted-case",
        "4",
        "--platform",
        "gpu",
        "--output-root",
        str(output_root),
        "--execute",
    )


def build_worker_plans(
    *,
    checkout_root: Path = ROOT,
    python: Path = DEFAULT_PYTHON,
    jonswap_root: Path = DEFAULT_JONSWAP_ROOT,
    tanaka_root: Path = DEFAULT_TANAKA_ROOT,
) -> tuple[WorkerPlan, ...]:
    """Return the exact plans declared by the two current bulk launchers."""

    checkout_root = _resolve(checkout_root)
    # Linux preserves the launcher's lexical argv[0] even when the virtual-
    # environment executable is a symlink.  Match that exact spelling.
    python = _absolute_without_resolving_symlinks(python)
    jonswap_root = _resolve(jonswap_root)
    tanaka_root = _resolve(tanaka_root)
    del checkout_root  # The cwd is checked by ``ExactWorkerMatcher``.

    jonswap_rows = (
        (0, "train", 2_048, 2_048, 1, "train/jonswap_tma/chunk_02048_02048"),
        (0, "train", 4_096, 4_096, 2, "train/jonswap_tma/chunk_04096_04096"),
        (1, "validation", 1_024, 0, 100, "validation/jonswap_tma/chunk_00000_01024"),
        (1, "test", 1_024, 0, 200, "test/jonswap_tma/chunk_00000_01024"),
        (1, "train", 4_096, 8_192, 3, "train/jonswap_tma/chunk_08192_04096"),
        (0, "train", 2_048, 12_288, 4, "train/jonswap_tma/chunk_12288_02048"),
        (1, "train", 2_048, 14_336, 5, "train/jonswap_tma/chunk_14336_02048"),
    )
    tanaka_rows = (
        (1, "train", 2_048, 0, 0, "train/tanaka/chunk_00000_02048"),
        (1, "train", 2_048, 2_048, 1, "train/tanaka/chunk_02048_02048"),
        (1, "train", 4_096, 4_096, 2, "train/tanaka/chunk_04096_04096"),
        (0, "train", 8_192, 8_192, 3, "train/tanaka/chunk_08192_08192"),
        (0, "validation", 1_024, 0, 100, "validation/tanaka/chunk_00000_01024"),
        (1, "test", 1_024, 0, 200, "test/tanaka/chunk_00000_01024"),
    )

    jonswap = tuple(
        WorkerPlan(
            family="jonswap_tma",
            lane=lane,
            gpu_index=str(lane),
            split=split,
            output_root=(jonswap_root / leaf),
            argv=_jonswap_argv(
                python,
                split=split,
                count=count,
                accepted_before=accepted_before,
                stream=stream,
                output_root=jonswap_root / leaf,
            ),
        )
        for lane, split, count, accepted_before, stream, leaf in jonswap_rows
    )
    tanaka = tuple(
        WorkerPlan(
            family="tanaka",
            lane=lane,
            gpu_index=str(lane),
            split=split,
            output_root=(tanaka_root / leaf),
            argv=_tanaka_argv(
                python,
                split=split,
                count=count,
                accepted_before=accepted_before,
                stream=stream,
                output_root=tanaka_root / leaf,
            ),
        )
        for lane, split, count, accepted_before, stream, leaf in tanaka_rows
    )
    return jonswap + tanaka


class ExactWorkerMatcher:
    """Fail closed unless every owner/cwd/environment/argv field is exact."""

    def __init__(
        self,
        plans: Sequence[WorkerPlan],
        *,
        checkout_root: Path,
        owner_uid: int,
    ) -> None:
        self._plans = {plan.argv: plan for plan in plans}
        if len(self._plans) != len(plans):
            raise ValueError("worker plans must have unique argument vectors")
        self._checkout_root = _resolve(checkout_root)
        self._owner_uid = owner_uid

    def match(self, process: ProcessSnapshot) -> WorkerPlan | None:
        """Return a plan only for this user's exact declared worker."""

        if process.uid != self._owner_uid or process.cwd != self._checkout_root:
            return None
        plan = self._plans.get(process.argv)
        if plan is None:
            return None
        if process.environment.get("CUDA_VISIBLE_DEVICES") != plan.gpu_index:
            return None
        return plan


def _parse_start_ticks(raw: str) -> int:
    """Parse Linux /proc/PID/stat field 22 despite spaces in comm."""

    suffix = raw.rsplit(")", maxsplit=1)
    if len(suffix) != 2:
        raise ValueError("malformed procfs stat record")
    fields = suffix[1].split()
    if len(fields) <= 19:
        raise ValueError("short procfs stat record")
    return int(fields[19])


def _parse_uid(raw: str) -> int:
    for line in raw.splitlines():
        if line.startswith("Uid:"):
            fields = line.split()
            if len(fields) >= 2:
                return int(fields[1])
    raise ValueError("procfs status has no real UID")


def _parse_null_fields(raw: bytes) -> tuple[str, ...]:
    fields = raw.rstrip(b"\0").split(b"\0") if raw else []
    return tuple(os.fsdecode(field) for field in fields if field)


def _parse_environment(raw: bytes) -> dict[str, str]:
    environment: dict[str, str] = {}
    for field in _parse_null_fields(raw):
        key, separator, value = field.partition("=")
        if separator and key == "CUDA_VISIBLE_DEVICES":
            environment[key] = value
    return environment


class ProcfsProcessSource:
    """Race-aware Linux procfs reader."""

    def __init__(self, proc_root: Path = Path("/proc")) -> None:
        self._proc_root = proc_root

    def read_process(self, pid: int) -> ProcessSnapshot | None:
        process_root = self._proc_root / str(pid)
        snapshot: ProcessSnapshot | None = None
        with suppress(
            FileNotFoundError,
            NotADirectoryError,
            PermissionError,
            ProcessLookupError,
            UnicodeDecodeError,
            ValueError,
        ):
            first_start = _parse_start_ticks(
                (process_root / "stat").read_text(encoding="utf-8")
            )
            uid = _parse_uid((process_root / "status").read_text(encoding="utf-8"))
            argv = _parse_null_fields((process_root / "cmdline").read_bytes())
            environment = _parse_environment((process_root / "environ").read_bytes())
            cwd = _resolve(Path(os.readlink(process_root / "cwd")))
            second_start = _parse_start_ticks(
                (process_root / "stat").read_text(encoding="utf-8")
            )
            if first_start != second_start or not argv:
                return None
            snapshot = ProcessSnapshot(
                pid=pid,
                start_ticks=first_start,
                uid=uid,
                cwd=cwd,
                argv=argv,
                environment=environment,
            )
        return snapshot

    def snapshots(self) -> tuple[ProcessSnapshot, ...]:
        pids: list[int] = []
        with suppress(FileNotFoundError, PermissionError):
            pids = sorted(
                int(path.name)
                for path in self._proc_root.iterdir()
                if path.name.isdigit()
            )
        snapshots = map(self.read_process, pids)
        return tuple(snapshot for snapshot in snapshots if snapshot is not None)


def _newest_file_token(
    directory: Path,
    *,
    pattern: str,
    output_root: Path,
) -> FileToken | None:
    tokens: list[FileToken] = []
    with suppress(FileNotFoundError, PermissionError):
        for path in directory.glob(pattern):
            with suppress(FileNotFoundError, PermissionError):
                metadata = path.stat()
                if path.is_file():
                    tokens.append(
                        FileToken(
                            relative_path=str(path.relative_to(output_root)),
                            mtime_ns=metadata.st_mtime_ns,
                            size=metadata.st_size,
                        )
                    )
    if not tokens:
        return None
    return max(tokens, key=lambda token: (token.mtime_ns, token.relative_path))


def read_heartbeat_token(plan: WorkerPlan) -> HeartbeatToken:
    """Read immutable transaction progress for exactly one worker root."""

    base = plan.output_root
    family_split = (plan.family, plan.split)
    proposal_dir = base / "proposals" / Path(*family_split)
    result_dir = base / "results" / Path(*family_split)
    shard_dir = base / "shards" / Path(*family_split)
    return HeartbeatToken(
        proposal=_newest_file_token(
            proposal_dir,
            pattern="batch_*.npz",
            output_root=base,
        ),
        result=_newest_file_token(
            result_dir,
            pattern="batch_*.json",
            output_root=base,
        ),
        shard=_newest_file_token(
            shard_dir,
            pattern="batch_*.npz",
            output_root=base,
        ),
    )


def _identity(worker: MatchedWorker) -> ProcessIdentity:
    return ProcessIdentity(
        pid=worker.process.pid,
        start_ticks=worker.process.start_ticks,
        family=worker.plan.family,
        output_root=worker.plan.output_root,
    )


class SafeSignalSender:
    """Re-read procfs before every signal to prevent PID-reuse accidents."""

    def __init__(
        self,
        source: ProcessSource,
        matcher: ExactWorkerMatcher,
        *,
        dry_run: bool = False,
    ) -> None:
        self._source = source
        self._matcher = matcher
        self._dry_run = dry_run

    def send(self, worker: MatchedWorker, signal_number: int) -> bool:
        current = self._source.read_process(worker.process.pid)
        if current is None or current.start_ticks != worker.process.start_ticks:
            return False
        current_plan = self._matcher.match(current)
        if current_plan != worker.plan:
            return False
        if self._dry_run:
            return True
        with suppress(ProcessLookupError, PermissionError):
            os.kill(current.pid, signal_number)
            return True
        return False


class TimestampedLogger:
    """Write flush-safe ISO timestamped lines to stderr and an optional file."""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}\n"
        sys.stderr.write(line)
        sys.stderr.flush()
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()


def _file_token_label(token: FileToken | None) -> str:
    return "none" if token is None else Path(token.relative_path).name


def heartbeat_label(token: HeartbeatToken) -> str:
    return (
        f"proposal={_file_token_label(token.proposal)} "
        f"result={_file_token_label(token.result)} "
        f"shard={_file_token_label(token.shard)}"
    )


class WorkerWatchdog:
    """Per-process transaction-heartbeat state machine."""

    def __init__(
        self,
        *,
        source: ProcessSource,
        matcher: ExactWorkerMatcher,
        signal_sender: SignalSender,
        logger: LogSink,
        stale_seconds: Mapping[Family, float],
        kill_grace_seconds: float,
        log_interval_seconds: float,
        token_reader: Callable[[WorkerPlan], HeartbeatToken] = read_heartbeat_token,
        dry_run: bool = False,
    ) -> None:
        self._source = source
        self._matcher = matcher
        self._signal_sender = signal_sender
        self._logger = logger
        self._stale_seconds = dict(stale_seconds)
        self._kill_grace_seconds = kill_grace_seconds
        self._log_interval_seconds = log_interval_seconds
        self._token_reader = token_reader
        self._dry_run = dry_run
        self._states: dict[ProcessIdentity, WorkerState] = {}
        self._last_idle_log_at: float | None = None

    @property
    def states(self) -> Mapping[ProcessIdentity, WorkerState]:
        return self._states

    def _log_worker(self, state: WorkerState, *, now: float, event: str) -> None:
        age = max(0.0, now - state.last_progress_at)
        worker = state.worker
        self._logger.write(
            f"{event} family={worker.plan.family} lane={worker.plan.lane} "
            f"gpu={worker.plan.gpu_index} pid={worker.process.pid} "
            f"phase={state.phase} heartbeat_age_seconds={age:.1f} "
            f"root={worker.plan.output_root} {heartbeat_label(state.token)}"
        )
        state.last_log_at = now

    def _active_workers(self) -> tuple[MatchedWorker, ...]:
        matched: list[MatchedWorker] = []
        for process in self._source.snapshots():
            plan = self._matcher.match(process)
            if plan is not None:
                matched.append(MatchedWorker(process=process, plan=plan))
        return tuple(
            sorted(
                matched,
                key=lambda worker: (
                    worker.plan.family,
                    worker.plan.lane,
                    worker.process.pid,
                ),
            )
        )

    def tick(self, now: float) -> int:
        """Observe once, advance signal states, and return the match count."""

        workers = self._active_workers()
        active_identities = {_identity(worker) for worker in workers}
        for identity in tuple(self._states):
            if identity not in active_identities:
                prior = self._states.pop(identity)
                self._log_worker(prior, now=now, event="worker_disappeared")

        if not workers:
            if (
                self._last_idle_log_at is None
                or now - self._last_idle_log_at >= self._log_interval_seconds
            ):
                self._logger.write("active_workers=0")
                self._last_idle_log_at = now
            return 0

        self._last_idle_log_at = None
        for worker in workers:
            identity = _identity(worker)
            token = self._token_reader(worker.plan)
            state = self._states.get(identity)
            if state is None:
                state = WorkerState(
                    worker=worker,
                    token=token,
                    last_progress_at=now,
                    last_log_at=now,
                )
                self._states[identity] = state
                self._log_worker(state, now=now, event="worker_matched")
            elif token != state.token:
                state.worker = worker
                state.token = token
                state.last_progress_at = now
                state.phase = "running"
                state.kill_due_at = None
                self._log_worker(state, now=now, event="transaction_progress")
            else:
                state.worker = worker

            heartbeat_age = max(0.0, now - state.last_progress_at)
            if (
                state.phase == "running"
                and heartbeat_age >= self._stale_seconds[worker.plan.family]
            ):
                sent = self._signal_sender.send(worker, signal.SIGTERM)
                action = "would_send_term" if self._dry_run else "sent_term"
                if sent:
                    state.phase = "term_sent"
                    state.kill_due_at = now + self._kill_grace_seconds
                    self._log_worker(state, now=now, event=action)
                else:
                    self._log_worker(
                        state,
                        now=now,
                        event="term_skipped_revalidation_failed",
                    )
            elif (
                state.phase == "term_sent"
                and state.kill_due_at is not None
                and now >= state.kill_due_at
            ):
                sent = self._signal_sender.send(worker, signal.SIGKILL)
                action = "would_send_kill" if self._dry_run else "sent_kill"
                if sent:
                    state.phase = "kill_sent"
                    state.kill_due_at = None
                    self._log_worker(state, now=now, event=action)
                else:
                    self._log_worker(
                        state,
                        now=now,
                        event="kill_skipped_revalidation_failed",
                    )

            if now - state.last_log_at >= self._log_interval_seconds:
                self._log_worker(state, now=now, event="heartbeat_status")
        return len(workers)


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _same_json(left: object, right: object) -> bool:
    return _canonical_json_sha256(left) == _canonical_json_sha256(right)


def _strict_json_object(raw: bytes, *, context: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{context} contains nonfinite JSON constant {value!r}")

    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        record: dict[str, object] = {}
        for key, value in pairs:
            if key in record:
                raise ValueError(f"{context} repeats JSON key {key!r}")
            record[key] = value
        return record

    value = json.loads(
        raw.decode("utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _sequence(value: object, *, context: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a JSON array")
    return value


def _integer(value: object, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if value < minimum:
        raise ValueError(f"{context} must be at least {minimum}")
    return value


def _digest(value: object, *, context: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _require_exact_keys(
    record: Mapping[str, object],
    expected: set[str],
    *,
    context: str,
) -> None:
    observed = set(record)
    if observed != expected:
        raise ValueError(
            f"{context} keys differ: "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _require_exact_json(
    value: object,
    expected: object,
    *,
    context: str,
) -> None:
    """Compare one JSON tree while rejecting bool-for-int coercion."""

    if isinstance(expected, bool):
        if value is not expected:
            raise ValueError(f"{context} differs")
        return
    if isinstance(expected, int):
        if _integer(value, context=context) != expected:
            raise ValueError(f"{context} differs")
        return
    if isinstance(expected, dict):
        observed = _mapping(value, context=context)
        _require_exact_keys(observed, set(expected), context=context)
        for key, expected_item in expected.items():
            _require_exact_json(
                observed[key],
                expected_item,
                context=f"{context}.{key}",
            )
        return
    if isinstance(expected, list):
        observed_items = _sequence(value, context=context)
        if len(observed_items) != len(expected):
            raise ValueError(f"{context} length differs")
        for index, (observed_item, expected_item) in enumerate(
            zip(observed_items, expected)
        ):
            _require_exact_json(
                observed_item,
                expected_item,
                context=f"{context}[{index}]",
            )
        return
    if not _same_json(value, expected):
        raise ValueError(f"{context} differs")


def _read_regular_file(
    path: Path,
    *,
    capture: bool,
) -> AuthenticatedFile:
    """Hash one canonical regular file while detecting path swaps."""

    requested = Path(path).expanduser()
    if not requested.is_absolute() or requested != _absolute_without_resolving_symlinks(
        requested
    ):
        raise ValueError(f"artifact path is not canonical and absolute: {requested}")
    if requested.resolve(strict=True) != requested:
        raise ValueError(f"artifact path traverses a symlink: {requested}")
    before = requested.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"artifact is not a regular file: {requested}")
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    digest = hashlib.sha256()
    captured: list[bytes] | None = [] if capture else None
    with requested.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        identity_opened = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if identity_opened != identity_before or not stat.S_ISREG(opened.st_mode):
            raise RuntimeError(f"artifact changed before hashing: {requested}")
        while block := stream.read(1024 * 1024):
            digest.update(block)
            if captured is not None:
                captured.append(block)
    after = requested.lstat()
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_after != identity_before or not stat.S_ISREG(after.st_mode):
        raise RuntimeError(f"artifact changed while hashing: {requested}")
    return AuthenticatedFile(
        path=requested,
        size=before.st_size,
        sha256=digest.hexdigest(),
        identity=identity_before,
        data=b"".join(captured) if captured is not None else None,
    )


def _read_exact_bytes(stream: Any, size: int, *, context: str) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError(f"{context} is truncated")
    return value


def _read_npy_header(
    stream: Any,
    *,
    context: str,
) -> tuple[Mapping[str, object], int]:
    if _read_exact_bytes(stream, 6, context=context) != b"\x93NUMPY":
        raise ValueError(f"{context} has the wrong NPY magic")
    version = _read_exact_bytes(stream, 2, context=context)
    if version == b"\x01\x00":
        length_size = 2
        encoding = "latin-1"
    elif version == b"\x02\x00":
        length_size = 4
        encoding = "latin-1"
    elif version == b"\x03\x00":
        length_size = 4
        encoding = "utf-8"
    else:
        raise ValueError(f"{context} uses an unsupported NPY version")
    raw_length = _read_exact_bytes(stream, length_size, context=context)
    header_length = int.from_bytes(raw_length, "little")
    if header_length <= 0 or header_length > 64 * 1024:
        raise ValueError(f"{context} has an invalid NPY header length")
    raw_header = _read_exact_bytes(stream, header_length, context=context)
    try:
        decoded = raw_header.decode(encoding)
        parsed = ast.literal_eval(decoded.strip())
    except (SyntaxError, UnicodeError, ValueError) as error:
        raise ValueError(f"{context} has an invalid NPY header") from error
    header = _mapping(parsed, context=f"{context} NPY header")
    _require_exact_keys(
        header,
        {"descr", "fortran_order", "shape"},
        context=f"{context} NPY header",
    )
    return header, 8 + length_size + header_length


def _validate_trajectory_map_npz(
    path: Path,
    *,
    attempted_total: int,
) -> None:
    """Stream-validate the bounded combined map without importing NumPy."""

    expected_names = {f"{name}.npy" for name in FINAL_TRAJECTORY_MAP_DTYPES}
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            observed_names = [info.filename for info in infos]
            if (
                len(observed_names) != len(expected_names)
                or len(set(observed_names)) != len(observed_names)
                or set(observed_names) != expected_names
            ):
                raise ValueError("final trajectory map has the wrong NPZ members")
            for info in infos:
                array_name = info.filename.removesuffix(".npy")
                descriptor, item_size, shape_kind = FINAL_TRAJECTORY_MAP_DTYPES[
                    array_name
                ]
                expected_shape = {
                    "schema": (),
                    "row": (FINAL_RETAINED_ROWS,),
                    "trajectory": (attempted_total,),
                }[shape_kind]
                context = f"final trajectory-map member {info.filename}"
                with archive.open(info, "r") as stream:
                    header, header_size = _read_npy_header(stream, context=context)
                    shape = header.get("shape")
                    if (
                        header.get("descr") != descriptor
                        or header.get("fortran_order") is not False
                        or not isinstance(shape, tuple)
                        or any(
                            isinstance(dimension, bool)
                            or not isinstance(dimension, int)
                            or dimension < 0
                            for dimension in shape
                        )
                        or shape != expected_shape
                    ):
                        raise ValueError(f"{context} has the wrong dtype or shape")
                    element_count = math.prod(expected_shape)
                    expected_payload_size = element_count * item_size
                    if info.file_size != header_size + expected_payload_size:
                        raise ValueError(f"{context} has the wrong byte length")
                    if shape_kind == "schema":
                        payload_prefix = _read_exact_bytes(
                            stream,
                            item_size,
                            context=context,
                        )
                        if struct.unpack("<h", payload_prefix)[0] != 2:
                            raise ValueError(
                                "final trajectory map has the wrong schema version"
                            )
            for info in infos:
                consumed = 0
                with archive.open(info, "r") as stream:
                    while block := stream.read(1024 * 1024):
                        consumed += len(block)
                if consumed != info.file_size:
                    raise ValueError(
                        f"final trajectory-map member {info.filename} is truncated"
                    )
    # ZipFile delegates stream failures to zlib, bz2, and lzma.  BZIP2 uses
    # OSError/EOFError; unsupported or encrypted methods use RuntimeError.
    except (
        KeyError,
        EOFError,
        lzma.LZMAError,
        OSError,
        RuntimeError,
        struct.error,
        zipfile.BadZipFile,
        zlib.error,
    ) as error:
        raise ValueError("final trajectory map is not a valid NPZ") from error


def _final_chunk_plans(repository_root: Path) -> tuple[FinalChunkPlan, ...]:
    plans: list[FinalChunkPlan] = []
    for split_id, split in enumerate(FINAL_SPLIT_ORDER):
        for family in FINAL_FAMILY_ORDER:
            train_layout = (
                FINAL_JONSWAP_TRAIN_LAYOUT
                if family == "jonswap_tma"
                else FINAL_STANDARD_TRAIN_LAYOUT
            )
            layout = (
                train_layout
                if split == "train"
                else ((0, 1_024, 100 if split == "validation" else 200),)
            )
            for accepted_before, accepted_count, stream_id in layout:
                family_root = repository_root / FINAL_SOURCE_ROOTS[family]
                if family == "stokes" and split != "train":
                    root = family_root / split / family / "c01024"
                else:
                    root = (
                        family_root
                        / split
                        / family
                        / (f"chunk_{accepted_before:05d}_{accepted_count:05d}")
                    )
                summary_path = root / f"paper_dataset_{family}_{split}.summary.json"
                plans.append(
                    FinalChunkPlan(
                        family=family,
                        family_id=FINAL_FAMILY_IDS[family],
                        revision_id=FINAL_REVISIONS[family],
                        split=split,
                        split_id=split_id,
                        accepted_before=accepted_before,
                        accepted_count=accepted_count,
                        stream_id=stream_id,
                        root=root,
                        summary_path=summary_path,
                    )
                )
    if len(plans) != FINAL_SOURCE_COUNT:
        raise AssertionError("canonical final chunk plan does not contain 26 sources")
    return tuple(plans)


def _validate_relative_artifact_record(
    value: object,
    *,
    expected_path: str,
    context: str,
) -> None:
    record = _mapping(value, context=context)
    _require_exact_keys(record, {"path", "bytes", "sha256"}, context=context)
    if record.get("path") != expected_path:
        raise ValueError(f"{context} path differs from the canonical artifact name")
    _integer(record.get("bytes"), context=f"{context} bytes", minimum=1)
    _digest(record.get("sha256"), context=f"{context} SHA-256")


def _generation_compatibility_id(
    *,
    family: str,
    execution_fingerprint: str,
    source_fingerprint: str,
) -> str | None:
    """Mirror the builder's current-first Tanaka compatibility resolution."""

    if family != "tanaka":
        return None
    if execution_fingerprint == TANAKA_CURRENT_EXECUTION_FINGERPRINT:
        return None
    if (
        execution_fingerprint,
        source_fingerprint,
    ) == (
        TANAKA_LEGACY_EXECUTION_FINGERPRINT,
        TANAKA_LEGACY_SOURCE_FINGERPRINT,
    ):
        return TANAKA_CURRENT_EXECUTION_FINGERPRINT
    raise ValueError("Tanaka generation identity is not current or explicitly bridged")


def _positive_finite_float(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite positive number")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(f"{context} must be a finite positive number") from error
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{context} must be a finite positive number")
    return converted


def _nonnegative_finite_float(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite nonnegative number")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(f"{context} must be a finite nonnegative number") from error
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{context} must be a finite nonnegative number")
    return converted


def _aware_datetime(value: object, *, context: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{context} must be a nonempty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{context} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{context} must include a UTC offset")
    return parsed


def _validate_dependency_environment(
    value: object,
) -> tuple[Mapping[str, object], str]:
    environment = _mapping(value, context="source dependency environment")
    _require_exact_keys(
        environment,
        {"python", "packages", "files_sha256"},
        context="source dependency environment",
    )
    python = _mapping(environment.get("python"), context="source dependency Python")
    _require_exact_keys(
        python,
        {"implementation", "version"},
        context="source dependency Python",
    )
    for key, field in (("implementation", "implementation"), ("version", "version")):
        if not isinstance(python.get(key), str) or not python[key]:
            raise ValueError(f"source dependency Python {field} is empty")
    packages = _mapping(
        environment.get("packages"), context="source dependency packages"
    )
    _require_exact_keys(
        packages,
        {"jax", "jaxlib", "numpy"},
        context="source dependency packages",
    )
    if any(
        not isinstance(packages.get(name), str) or not packages[name]
        for name in packages
    ):
        raise ValueError("source dependency package version is empty")
    files = _mapping(environment.get("files_sha256"), context="source dependency files")
    _require_exact_keys(
        files,
        {"pyproject.toml", "uv.lock"},
        context="source dependency files",
    )
    for name, digest_value in files.items():
        _digest(digest_value, context=f"source dependency file {name}")
    return environment, str(packages["jax"])


def _validate_runtime(
    value: object,
    *,
    execution_platform: str,
    expected_jax_version: str,
) -> Mapping[str, object]:
    runtime = _mapping(value, context="source runtime")
    _require_exact_keys(
        runtime,
        {
            "requested_platform",
            "jax_version",
            "default_backend",
            "x64_enabled",
            "devices",
        },
        context="source runtime",
    )
    if (
        runtime.get("requested_platform") != execution_platform
        or runtime.get("default_backend") != execution_platform
        or runtime.get("jax_version") != expected_jax_version
        or runtime.get("x64_enabled") is not True
    ):
        raise ValueError("source runtime differs from its execution environment")
    devices = _sequence(runtime.get("devices"), context="source runtime devices")
    if not devices:
        raise ValueError("source runtime device list is empty")
    device_ids: list[int] = []
    for index, raw_device in enumerate(devices):
        device = _mapping(raw_device, context=f"source runtime device {index}")
        _require_exact_keys(
            device,
            {"id", "platform", "device_kind"},
            context=f"source runtime device {index}",
        )
        device_ids.append(
            _integer(device.get("id"), context=f"source runtime device {index} ID")
        )
        if device.get("platform") != execution_platform:
            raise ValueError("source runtime device platform differs")
        if not isinstance(device.get("device_kind"), str) or not device["device_kind"]:
            raise ValueError("source runtime device kind is empty")
    if len(set(device_ids)) != len(device_ids):
        raise ValueError("source runtime device IDs are repeated")
    return runtime


def _validate_quota_state(
    value: object,
    *,
    quota_targets: Mapping[str, int],
    maximum_attempts: int,
    first_attempt_index: int,
    batch_size: int,
    context: str,
) -> tuple[Mapping[str, int], Mapping[str, int], int, int]:
    state = _mapping(value, context=context)
    _require_exact_keys(
        state,
        {
            "accepted_by_cell",
            "attempted_by_cell",
            "committed_batches",
            "next_attempt_index",
            "next_batch_id",
            "pending_batch_id",
            "pending_status",
            "complete",
            "terminal_failure",
            "attempt_limit_failure",
            "attempt_limit_exhausted_cells",
        },
        context=context,
    )
    accepted_record = _mapping(
        state.get("accepted_by_cell"), context=f"{context} accepted by cell"
    )
    attempted_record = _mapping(
        state.get("attempted_by_cell"), context=f"{context} attempted by cell"
    )
    if set(accepted_record) != set(quota_targets) or set(attempted_record) != set(
        quota_targets
    ):
        raise ValueError(f"{context} cell maps differ from the quota taxonomy")
    accepted_by_cell: dict[str, int] = {}
    attempted_by_cell: dict[str, int] = {}
    for cell_id, target in quota_targets.items():
        accepted = _integer(
            accepted_record[cell_id], context=f"{context} accepted {cell_id}"
        )
        attempted = _integer(
            attempted_record[cell_id],
            context=f"{context} attempted {cell_id}",
            minimum=accepted,
        )
        if accepted > target or attempted > maximum_attempts * target:
            raise ValueError(f"{context} cell count exceeds its immutable quota")
        accepted_by_cell[cell_id] = accepted
        attempted_by_cell[cell_id] = attempted

    committed_batches = _integer(
        state.get("committed_batches"), context=f"{context} committed batches"
    )
    next_attempt_index = _integer(
        state.get("next_attempt_index"), context=f"{context} next attempt index"
    )
    next_batch_id = _integer(
        state.get("next_batch_id"), context=f"{context} next batch ID"
    )
    raw_pending_batch_id = state.get("pending_batch_id")
    pending_batch_id = (
        None
        if raw_pending_batch_id is None
        else _integer(raw_pending_batch_id, context=f"{context} pending batch ID")
    )
    pending_status = state.get("pending_status")
    if pending_batch_id is None:
        if pending_status is not None or next_batch_id != committed_batches:
            raise ValueError(f"{context} has incoherent nonpending batch state")
    elif (
        pending_batch_id != committed_batches
        or pending_status not in {"proposed", "shard_written"}
        or next_batch_id != committed_batches + 1
    ):
        raise ValueError(f"{context} has incoherent pending batch state")
    if next_attempt_index != first_attempt_index + sum(attempted_by_cell.values()):
        raise ValueError(f"{context} attempt cursor differs from attempted counts")
    attempted_total = sum(attempted_by_cell.values())
    accepted_total = sum(accepted_by_cell.values())
    if (
        (next_batch_id == 0 and attempted_total != 0)
        or (next_batch_id > 0 and attempted_total < next_batch_id)
        or attempted_total > next_batch_id * batch_size
        or accepted_total > committed_batches * batch_size
    ):
        raise ValueError(f"{context} counts exceed its validated batch prefix")
    complete = state.get("complete")
    expected_complete = pending_batch_id is None and all(
        accepted_by_cell[cell_id] == target for cell_id, target in quota_targets.items()
    )
    if pending_batch_id is not None and all(
        accepted_by_cell[cell_id] == target for cell_id, target in quota_targets.items()
    ):
        raise ValueError(f"{context} has a pending batch after all targets were met")
    if complete is not expected_complete:
        raise ValueError(f"{context} completion flag differs from quota state")
    if (
        pending_batch_id is None
        and not expected_complete
        and any(
            accepted_by_cell[cell_id] < target
            and attempted_by_cell[cell_id] == maximum_attempts * target
            for cell_id, target in quota_targets.items()
        )
    ):
        raise ValueError(f"{context} omits an attempt-limit failure")
    if (
        state.get("terminal_failure") is not None
        or state.get("attempt_limit_failure") is not None
        or state.get("attempt_limit_exhausted_cells") != []
    ):
        raise ValueError(f"{context} records a terminal quota failure")
    return (
        accepted_by_cell,
        attempted_by_cell,
        committed_batches,
        next_batch_id,
    )


def _scheduled_cell_counts(
    quota_targets: Mapping[str, int],
    accepted_by_cell: Mapping[str, int],
    *,
    batch_size: int,
) -> Mapping[str, int]:
    """Reconstruct the deterministic assignments for one pending batch."""

    scheduled = {cell_id: 0 for cell_id in quota_targets}
    accepted_to_skip = dict(accepted_by_cell)
    target_size = min(
        batch_size,
        sum(
            quota_targets[cell_id] - accepted_by_cell[cell_id]
            for cell_id in quota_targets
        ),
    )
    for quota_level in range(max(quota_targets.values())):
        for cell_id, target in quota_targets.items():
            if target <= quota_level:
                continue
            if accepted_to_skip[cell_id] > 0:
                accepted_to_skip[cell_id] -= 1
                continue
            scheduled[cell_id] += 1
            if sum(scheduled.values()) == target_size:
                return scheduled
    if target_size:
        raise ValueError("source pending case schedule is incomplete")
    return scheduled


def _execution_contract_parts(
    execution: Mapping[str, Any],
    *,
    family: str,
) -> tuple[dict[str, object], Mapping[str, object] | None]:
    """Mirror the manifest builder's target/numerical extraction."""

    if family == "stokes":
        numerical = execution
        trajectory_numerical = None
    else:
        trajectory_numerical = _mapping(
            execution.get("numerical"),
            context=f"{family} trajectory numerical contract",
        )
        numerical = trajectory_numerical
    role = execution.get("role")
    dtype = numerical.get("dtype")
    if not isinstance(role, str) or not role:
        raise ValueError(f"{family} execution role is empty")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError(f"{family} execution dtype is empty")
    evolution_nx = _integer(
        numerical.get("nx"), context=f"{family} execution nx", minimum=1
    )
    raw_target_nx = numerical.get("target_nx")
    target_nx = (
        evolution_nx
        if raw_target_nx is None
        else _integer(raw_target_nx, context=f"{family} target nx", minimum=1)
    )
    evolution_dno_order = _integer(
        numerical.get("dno_order"), context=f"{family} execution DNO order"
    )
    raw_target_dno_order = numerical.get("target_dno_order")
    target_dno_order = (
        evolution_dno_order
        if raw_target_dno_order is None
        else _integer(raw_target_dno_order, context=f"{family} target DNO order")
    )
    evolution_maximum_wavenumber = _positive_finite_float(
        numerical.get("maximum_wavenumber"),
        context=f"{family} execution maximum wavenumber",
    )
    raw_target_maximum_wavenumber = numerical.get("target_maximum_wavenumber")
    target_maximum_wavenumber = (
        evolution_maximum_wavenumber
        if raw_target_maximum_wavenumber is None
        else _positive_finite_float(
            raw_target_maximum_wavenumber,
            context=f"{family} target maximum wavenumber",
        )
    )
    if target_maximum_wavenumber > evolution_maximum_wavenumber:
        raise ValueError(f"{family} target maximum wavenumber exceeds evolution band")
    return (
        {
            "role": role,
            "nx": target_nx,
            "length": _positive_finite_float(
                numerical.get("length"), context=f"{family} execution length"
            ),
            "gravity": _positive_finite_float(
                numerical.get("gravity"), context=f"{family} execution gravity"
            ),
            "dno_order": target_dno_order,
            "pad_factor": _integer(
                numerical.get("pad_factor"),
                context=f"{family} execution pad factor",
                minimum=1,
            ),
            "maximum_wavenumber": target_maximum_wavenumber,
            "dtype": dtype,
        },
        trajectory_numerical,
    )


def _validate_jonswap_bucketing_config(
    configuration: Mapping[str, Any],
    *,
    execution: Mapping[str, Any],
    batch_size: int,
) -> None:
    """Mirror the builder's exact proposal-bucketing config check."""

    config = _mapping(
        configuration.get("jonswap_horizon_bucketing"),
        context="JONSWAP horizon bucketing config",
    )
    solver_batch_size = _integer(
        config.get("solver_batch_size"),
        context="JONSWAP solver batch size",
        minimum=1,
    )
    if solver_batch_size != 8 or solver_batch_size > batch_size:
        raise ValueError("JONSWAP solver batch size differs from the release config")
    nonlinear_adjustment = _mapping(
        execution.get("jonswap_adjustment"),
        context="JONSWAP nonlinear adjustment policy",
    )
    expected = {
        "outer_proposal_size": batch_size,
        "solver_batch_size": solver_batch_size,
        "nonlinear_adjustment": dict(nonlinear_adjustment),
    }
    _require_exact_json(
        {key: config.get(key) for key in expected},
        expected,
        context="JONSWAP horizon bucketing config",
    )


def _validate_source_summary(
    record: Mapping[str, Any],
    *,
    plan: FinalChunkPlan,
) -> FinalChunkFacts:
    """Authenticate plan facts without scanning a chunk's batch artifacts."""

    _require_exact_keys(
        record,
        {
            "schema",
            "status",
            "configuration_fingerprint",
            "output_root",
            "run_spec",
            "execution",
            "runtime",
            "preflight",
            "resume",
            "counts",
            "dataset_view",
            "timing_seconds",
            "invocation_started_at",
            "invocation_finished_at",
        },
        context=f"{plan.family}/{plan.split} source summary",
    )
    if (
        record.get("schema") != "paper_dataset_quota_summary_v1"
        or record.get("status") != "complete"
        or record.get("output_root") != str(plan.root)
    ):
        raise ValueError("source summary identity or completion status differs")

    run_spec = _mapping(record.get("run_spec"), context="source run_spec")
    _require_exact_keys(
        run_spec,
        {
            "schema",
            "family_name",
            "family_id",
            "revision_id",
            "split_id",
            "root_seed",
            "stream_id",
            "first_attempt_index",
            "batch_size",
            "quotas",
            "cell_codes",
            "configuration",
            "maximum_attempts_per_accepted_case",
        },
        context="source run_spec",
    )
    family_id = _integer(run_spec.get("family_id"), context="source family ID")
    revision_id = _integer(run_spec.get("revision_id"), context="source revision ID")
    stream_id = _integer(run_spec.get("stream_id"), context="source stream ID")
    first_attempt_index = _integer(
        run_spec.get("first_attempt_index"),
        context="source first attempt index",
    )
    maximum_attempts = _integer(
        run_spec.get("maximum_attempts_per_accepted_case"),
        context="source maximum attempts per accepted case",
        minimum=1,
    )
    if (
        run_spec.get("schema") != "paper_dataset_accepted_quota_run_v2"
        or run_spec.get("family_name") != plan.family
        or family_id != plan.family_id
        or revision_id != plan.revision_id
        or run_spec.get("split_id") != plan.split
        or stream_id != plan.stream_id
        or first_attempt_index != 0
        or maximum_attempts != 4
    ):
        raise ValueError("source run_spec differs from its canonical chunk slot")
    root_seed = _integer(run_spec.get("root_seed"), context="source root seed")
    if root_seed != FINAL_ROOT_SEEDS[plan.split]:
        raise ValueError("source root seed differs from the canonical split seed")
    batch_size = _integer(
        run_spec.get("batch_size"), context="source batch size", minimum=1
    )
    if batch_size != FINAL_BATCH_SIZES[plan.family]:
        raise ValueError("source batch size differs from the canonical release plan")

    configuration = _mapping(
        run_spec.get("configuration"), context="source configuration"
    )
    configuration_keys = {
        "schema",
        "purpose",
        "case_kind",
        "accepted_case_count",
        "accepted_cases_before",
        "accepted_cases_after",
        "ordered_cell_ids",
        "execution_platform",
        "dependency_environment",
        "source_sha256",
        "contract" if plan.family == "stokes" else "trajectory_execution",
    }
    if plan.family == "stokes":
        configuration_keys.add("sampler")
    elif plan.family == "benjamin_feir":
        configuration_keys.add("sampling_support")
    elif plan.family == "jonswap_tma":
        configuration_keys.add("jonswap_horizon_bucketing")
    _require_exact_keys(
        configuration,
        configuration_keys,
        context="source configuration",
    )
    if (
        configuration.get("schema") != "paper_dataset_quota_configuration_v1"
        or configuration.get("purpose")
        != "exact paper-contract accepted-case generation"
        or configuration.get("case_kind")
        != ("static" if plan.family == "stokes" else "trajectory")
        or configuration.get("execution_platform")
        != ("cpu" if plan.family == "stokes" else "gpu")
    ):
        raise ValueError("source configuration has the wrong canonical identity")
    expected_interval = (
        plan.accepted_count,
        plan.accepted_before,
        plan.accepted_before + plan.accepted_count,
    )
    observed_interval = (
        _integer(
            configuration.get("accepted_case_count"),
            context="source accepted case count",
            minimum=1,
        ),
        _integer(
            configuration.get("accepted_cases_before"),
            context="source accepted cases before",
        ),
        _integer(
            configuration.get("accepted_cases_after"),
            context="source accepted cases after",
            minimum=1,
        ),
    )
    if observed_interval != expected_interval:
        raise ValueError("source configuration has the wrong accepted interval")

    fingerprint = _digest(
        record.get("configuration_fingerprint"),
        context="source configuration fingerprint",
    )
    if fingerprint != _canonical_json_sha256(run_spec):
        raise ValueError("source configuration fingerprint is not canonical")
    execution = _mapping(record.get("execution"), context="source execution")
    execution_key = "contract" if plan.family == "stokes" else "trajectory_execution"
    if not _same_json(configuration.get(execution_key), execution):
        raise ValueError("source execution differs from its run configuration")
    execution_fingerprint = _canonical_json_sha256(execution)
    expected_execution_fingerprint = FINAL_CURRENT_EXECUTION_FINGERPRINTS[plan.family]
    if execution_fingerprint != expected_execution_fingerprint:
        raise ValueError(
            f"{plan.family} source does not use the exact current execution contract"
        )
    if plan.family == "stokes":
        _require_exact_json(
            configuration.get("sampler"),
            {"maximum_ursell_redraws": 1_000},
            context="Stokes sampler policy",
        )
    elif plan.family == "benjamin_feir":
        if (
            _canonical_json_sha256(configuration.get("sampling_support"))
            != FINAL_BENJAMIN_FEIR_SAMPLING_SUPPORT_FINGERPRINT
        ):
            raise ValueError("Benjamin--Feir sampling support differs")
    elif plan.family == "jonswap_tma":
        _validate_jonswap_bucketing_config(
            configuration,
            execution=execution,
            batch_size=batch_size,
        )

    dependency_environment, expected_jax_version = _validate_dependency_environment(
        configuration.get("dependency_environment")
    )
    dependency_fingerprint = _canonical_json_sha256(dependency_environment)
    if dependency_fingerprint != FINAL_DEPENDENCY_FINGERPRINT:
        raise ValueError("source dependency environment differs from the release")
    raw_sources = _mapping(
        configuration.get("source_sha256"), context="source SHA-256 mapping"
    )
    if not raw_sources:
        raise ValueError("source SHA-256 mapping is empty")
    source_sha256: dict[str, str] = {}
    for source_path, digest_value in raw_sources.items():
        if not isinstance(source_path, str) or not source_path:
            raise ValueError("source SHA-256 mapping contains an empty path")
        source_sha256[source_path] = _digest(
            digest_value,
            context=f"source SHA-256 for {source_path}",
        )
    source_fingerprint = _canonical_json_sha256(source_sha256)
    if source_fingerprint != FINAL_SOURCE_FINGERPRINTS[plan.family]:
        raise ValueError("source file fingerprint differs from the release")
    execution_platform = configuration.get("execution_platform")
    if not isinstance(execution_platform, str) or not execution_platform:
        raise ValueError("source execution platform is empty")
    generation_compatibility_id = _generation_compatibility_id(
        family=plan.family,
        execution_fingerprint=execution_fingerprint,
        source_fingerprint=source_fingerprint,
    )
    if generation_compatibility_id is not None:
        raise ValueError("canonical final sources must use the fresh current identity")
    runtime = _validate_runtime(
        record.get("runtime"),
        execution_platform=execution_platform,
        expected_jax_version=expected_jax_version,
    )
    dataset_target, trajectory_numerical = _execution_contract_parts(
        execution, family=plan.family
    )

    quotas = _sequence(run_spec.get("quotas"), context="source quotas")
    cell_codes = _mapping(run_spec.get("cell_codes"), context="source cell codes")
    raw_ordered_cell_ids = _sequence(
        configuration.get("ordered_cell_ids"), context="source ordered cell IDs"
    )
    ordered_cell_ids = tuple(
        cell_id
        for cell_id in raw_ordered_cell_ids
        if isinstance(cell_id, str) and cell_id
    )
    if (
        len(ordered_cell_ids) != len(raw_ordered_cell_ids)
        or not ordered_cell_ids
        or len(set(ordered_cell_ids)) != len(ordered_cell_ids)
    ):
        raise ValueError("source ordered cell IDs are empty, invalid, or repeated")
    expected_cell_count, expected_cell_digest = FINAL_ORDERED_CELL_IDENTITIES[
        plan.family
    ]
    if (
        len(ordered_cell_ids) != expected_cell_count
        or _canonical_json_sha256(list(ordered_cell_ids)) != expected_cell_digest
    ):
        raise ValueError(
            "source ordered cell taxonomy differs from the canonical family"
        )
    if not quotas:
        raise ValueError("source valid-case targets are empty")
    quota_total = 0
    quota_cells: list[str] = []
    quota_targets: list[int] = []
    for raw_quota in quotas:
        quota = _mapping(raw_quota, context="source quota")
        _require_exact_keys(
            quota, {"cell_id", "target_accepted"}, context="source quota"
        )
        cell_id = quota.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError("source quota has an empty cell ID")
        quota_cells.append(cell_id)
        case_count = _integer(
            quota.get("target_accepted"),
            context="source valid-case target",
        )
        quota_targets.append(case_count)
        quota_total += case_count
    before_quotient, before_remainder = divmod(
        plan.accepted_before, len(ordered_cell_ids)
    )
    after_quotient, after_remainder = divmod(
        plan.accepted_before + plan.accepted_count, len(ordered_cell_ids)
    )
    expected_quota_targets = tuple(
        after_quotient
        + int(index < after_remainder)
        - before_quotient
        - int(index < before_remainder)
        for index in range(len(ordered_cell_ids))
    )
    if (
        len(set(quota_cells)) != len(quota_cells)
        or tuple(quota_cells) != ordered_cell_ids
        or tuple(quota_targets) != expected_quota_targets
        or set(cell_codes) != set(quota_cells)
        or quota_total != plan.accepted_count
    ):
        raise ValueError("source target cells or accepted total differ")
    cell_code_values = [
        _integer(cell_codes[cell], context=f"source code for {cell}")
        for cell in quota_cells
    ]
    if cell_code_values != list(range(len(quota_cells))):
        raise ValueError("source cell codes do not enumerate the ordered taxonomy")

    counts = _mapping(record.get("counts"), context="source counts")
    _require_exact_keys(
        counts,
        {"accepted", "attempted", "rejected", "by_cell", "rejection_reasons"},
        context="source counts",
    )
    accepted = _integer(counts.get("accepted"), context="source accepted count")
    attempted = _integer(
        counts.get("attempted"), context="source attempted count", minimum=accepted
    )
    rejected = _integer(counts.get("rejected"), context="source rejected count")
    if accepted != plan.accepted_count or rejected != attempted - accepted:
        raise ValueError("source accepted/attempted/rejected counts differ")
    by_cell = _mapping(counts.get("by_cell"), context="source counts by cell")
    if set(by_cell) != set(quota_cells):
        raise ValueError("source per-cell counts differ from its target cells")
    by_cell_accepted = 0
    by_cell_attempted = 0
    accepted_by_cell: dict[str, int] = {}
    attempted_by_cell: dict[str, int] = {}
    for raw_quota in quotas:
        assert isinstance(raw_quota, dict)
        cell_id = str(raw_quota["cell_id"])
        cell = _mapping(by_cell[cell_id], context=f"source count cell {cell_id}")
        _require_exact_keys(
            cell,
            {"accepted", "attempted", "rejected", "target_accepted"},
            context=f"source count cell {cell_id}",
        )
        target = _integer(raw_quota["target_accepted"], context="source cell target")
        cell_accepted = _integer(cell.get("accepted"), context="cell accepted")
        cell_attempted = _integer(
            cell.get("attempted"), context="cell attempted", minimum=cell_accepted
        )
        cell_rejected = _integer(cell.get("rejected"), context="cell rejected")
        cell_target = _integer(
            cell.get("target_accepted"), context="cell target accepted"
        )
        if (
            cell_target != target
            or cell_accepted != target
            or cell_attempted > maximum_attempts * target
            or cell_rejected != cell_attempted - cell_accepted
        ):
            raise ValueError("source per-cell count arithmetic differs")
        accepted_by_cell[cell_id] = cell_accepted
        attempted_by_cell[cell_id] = cell_attempted
        by_cell_accepted += cell_accepted
        by_cell_attempted += cell_attempted
    if (by_cell_accepted, by_cell_attempted) != (accepted, attempted):
        raise ValueError("source per-cell totals differ")
    reasons = _mapping(
        counts.get("rejection_reasons"), context="source rejection reasons"
    )
    reason_positions = {name: index for index, name in enumerate(QUALITY_REASON_ORDER)}
    for label in reasons:
        if not isinstance(label, str) or not label:
            raise ValueError("source rejection reason label is empty")
        if label == "missing_check":
            continue
        names = label.split("+")
        if any(name not in reason_positions for name in names):
            raise ValueError("source rejection reason label is not canonical")
        positions = [reason_positions[name] for name in names]
        if len(set(names)) != len(names) or positions != sorted(positions):
            raise ValueError("source rejection reason label is not canonical")
    if sum(
        _integer(value, context=f"source rejection reason {name}", minimum=1)
        for name, value in reasons.items()
        if isinstance(name, str) and name
    ) != rejected or any(not isinstance(name, str) or not name for name in reasons):
        raise ValueError("source rejection-reason totals differ")

    source_view = _mapping(record.get("dataset_view"), context="source dataset view")
    _require_exact_keys(
        source_view,
        {
            "schema_version",
            "configuration_fingerprint",
            "n_rows",
            "n_trajectories",
            "n_accepted_trajectories",
            "n_accepted_rows",
            "grid",
            "manifest",
            "trajectory_map",
        },
        context="source dataset view",
    )
    retained_rows = plan.accepted_count * FINAL_ROWS_PER_CASE[plan.family]
    view_schema_version = _integer(
        source_view.get("schema_version"), context="source view schema version"
    )
    view_rows = _integer(source_view.get("n_rows"), context="source view rows")
    view_trajectories = _integer(
        source_view.get("n_trajectories"), context="source view trajectories"
    )
    view_accepted_trajectories = _integer(
        source_view.get("n_accepted_trajectories"),
        context="source view accepted trajectories",
    )
    view_accepted_rows = _integer(
        source_view.get("n_accepted_rows"), context="source view accepted rows"
    )
    if (
        view_schema_version != 2
        or source_view.get("configuration_fingerprint") != fingerprint
        or view_rows != retained_rows
        or view_trajectories != attempted
        or view_accepted_trajectories != plan.accepted_count
        or view_accepted_rows != retained_rows
    ):
        raise ValueError("source dataset-view counts or grid differ")
    _require_exact_json(
        source_view.get("grid"),
        {"length": 2.0 * math.pi, "nx": 1024},
        context="source view grid",
    )
    source_name = f"paper_dataset_{plan.family}_{plan.split}"
    _validate_relative_artifact_record(
        source_view.get("manifest"),
        expected_path=f"{source_name}.dataset.json",
        context="source manifest record",
    )
    _validate_relative_artifact_record(
        source_view.get("trajectory_map"),
        expected_path=f"{source_name}.trajectory_map.npz",
        context="source trajectory-map record",
    )

    resume = _mapping(record.get("resume"), context="source resume")
    _require_exact_keys(resume, {"initial", "final"}, context="source resume")
    quota_target_by_cell = dict(zip(quota_cells, quota_targets))
    (
        initial_accepted_by_cell,
        initial_attempted_by_cell,
        initial_committed_batches,
        initial_next_batch_id,
    ) = _validate_quota_state(
        resume.get("initial"),
        quota_targets=quota_target_by_cell,
        maximum_attempts=maximum_attempts,
        first_attempt_index=first_attempt_index,
        batch_size=batch_size,
        context="source initial state",
    )
    (
        final_accepted_by_cell,
        final_attempted_by_cell,
        committed_batches,
        _,
    ) = _validate_quota_state(
        resume.get("final"),
        quota_targets=quota_target_by_cell,
        maximum_attempts=maximum_attempts,
        first_attempt_index=first_attempt_index,
        batch_size=batch_size,
        context="source final state",
    )
    _require_exact_json(
        final_accepted_by_cell,
        accepted_by_cell,
        context="source final accepted by cell",
    )
    _require_exact_json(
        final_attempted_by_cell,
        attempted_by_cell,
        context="source final attempted by cell",
    )
    initial_pending_assignments = (
        _scheduled_cell_counts(
            quota_target_by_cell,
            initial_accepted_by_cell,
            batch_size=batch_size,
        )
        if initial_next_batch_id > initial_committed_batches
        else {cell_id: 0 for cell_id in quota_cells}
    )
    committed_attempted_by_cell = {
        cell_id: (
            initial_attempted_by_cell[cell_id] - initial_pending_assignments[cell_id]
        )
        for cell_id in quota_cells
    }
    if any(
        committed_attempted_by_cell[cell_id] < initial_accepted_by_cell[cell_id]
        for cell_id in quota_cells
    ):
        raise ValueError("source pending assignments differ from the case scheduler")
    if initial_committed_batches and any(
        committed_attempted_by_cell[cell_id] < first_batch_count
        for cell_id, first_batch_count in _scheduled_cell_counts(
            quota_target_by_cell,
            {cell_id: 0 for cell_id in quota_cells},
            batch_size=batch_size,
        ).items()
    ):
        raise ValueError("source committed prefix omits the deterministic first batch")
    # The manifest exposes only aggregate accepted counts for older committed
    # batches, so their complete per-cell reachability is not summary-derived.
    # The canonical builder/postcompletion scans proposals and results for that
    # proof.  This bounded watchdog authenticates the exact derivable pending
    # delta, immutable batch-zero floor, and resume-to-final causal envelope.
    if any(
        accepted_by_cell[cell_id] - initial_accepted_by_cell[cell_id]
        > attempted_by_cell[cell_id]
        - initial_attempted_by_cell[cell_id]
        + initial_pending_assignments[cell_id]
        for cell_id in quota_cells
    ):
        raise ValueError("source resume-to-final per-cell counts are not causal")
    if (
        committed_batches < 1
        or initial_committed_batches > committed_batches
        or initial_next_batch_id > committed_batches
        or any(
            initial_accepted_by_cell[cell_id] > accepted_by_cell[cell_id]
            or initial_attempted_by_cell[cell_id] > attempted_by_cell[cell_id]
            for cell_id in quota_cells
        )
        or (
            all(
                initial_accepted_by_cell[cell_id] == quota_target_by_cell[cell_id]
                for cell_id in quota_cells
            )
            and (
                initial_accepted_by_cell != final_accepted_by_cell
                or initial_attempted_by_cell != final_attempted_by_cell
                or initial_committed_batches != committed_batches
            )
        )
    ):
        raise ValueError("source quota resume prefix differs from the final state")

    preflight = _mapping(record.get("preflight"), context="source preflight")
    preflight_keys = {
        "schema",
        "mode",
        "no_numerical_generation_performed",
        "output_root",
        "artifact_namespace",
        "configuration_fingerprint",
        "run_spec",
        "execution",
        "allocation",
        "expected_output",
        "resume_state",
        "runtime",
    }
    if plan.family != "stokes":
        preflight_keys.add("revision_id")
    _require_exact_keys(
        preflight,
        preflight_keys,
        context="source preflight",
    )
    preflight_revision_id = (
        plan.revision_id
        if plan.family == "stokes"
        else _integer(
            preflight.get("revision_id"), context="source preflight revision ID"
        )
    )
    if (
        preflight.get("schema") != "paper_dataset_quota_preflight_v1"
        or preflight.get("mode") != "dry_run"
        or preflight.get("no_numerical_generation_performed") is not True
        or (plan.family != "stokes" and preflight_revision_id != plan.revision_id)
        or preflight.get("output_root") != str(plan.root)
        or preflight.get("configuration_fingerprint") != fingerprint
        or not _same_json(preflight.get("run_spec"), run_spec)
        or not _same_json(preflight.get("execution"), execution)
    ):
        raise ValueError("source preflight differs from its completed summary")
    source_name = f"paper_dataset_{plan.family}_{plan.split}"
    _require_exact_json(
        preflight.get("artifact_namespace"),
        {
            "family": plan.family,
            "split": plan.split,
            "view_name": source_name,
            "summary_path": str(plan.summary_path),
        },
        context="source preflight artifact namespace",
    )
    allocation_quotas = [
        {
            "cell_id": cell_id,
            "accepted_before": before_quotient + int(index < before_remainder),
            "chunk_target_accepted": quota_targets[index],
            "accepted_after": after_quotient + int(index < after_remainder),
            "attempt_ceiling": maximum_attempts * quota_targets[index],
            "durable_attempted": initial_attempted_by_cell[cell_id],
            "remaining_attempt_capacity": (
                maximum_attempts * quota_targets[index]
                - initial_attempted_by_cell[cell_id]
            ),
        }
        for index, cell_id in enumerate(quota_cells)
    ]
    _require_exact_json(
        preflight.get("allocation"),
        {
            "chunk_accepted_cases": plan.accepted_count,
            "accepted_cases_before": plan.accepted_before,
            "accepted_cases_after": plan.accepted_before + plan.accepted_count,
            "cell_count": len(quota_cells),
            "nonzero_quota_cell_count": sum(target > 0 for target in quota_targets),
            "quota_minimum": min(quota_targets),
            "quota_maximum": max(quota_targets),
            "maximum_attempts_per_accepted_case": maximum_attempts,
            "quotas": allocation_quotas,
            "chunk_identity": {
                "stream_id": plan.stream_id,
                "first_attempt_index": first_attempt_index,
                "output_root": str(plan.root),
                "rule": (
                    "use a distinct stream_id and output_root for every additive chunk"
                ),
            },
        },
        context="source preflight allocation",
    )
    _require_exact_json(
        preflight.get("expected_output"),
        {
            "stored_rows_per_accepted_case": FINAL_ROWS_PER_CASE[plan.family],
            "retained_rows": retained_rows,
            "spatial_points_per_row": 1_024,
            "field_values_per_row": 3_072,
        },
        context="source preflight expected output",
    )
    _require_exact_json(
        preflight.get("resume_state"),
        resume.get("initial"),
        context="source preflight resume state",
    )
    _require_exact_json(
        preflight.get("runtime"),
        dict(runtime),
        context="source preflight runtime",
    )

    timing = _mapping(record.get("timing_seconds"), context="source timing")
    _require_exact_keys(
        timing,
        {"quota_driver", "dataset_view", "validation", "total"},
        context="source timing",
    )
    component_seconds = tuple(
        _nonnegative_finite_float(timing.get(name), context=f"source timing {name}")
        for name in ("quota_driver", "dataset_view", "validation")
    )
    total_seconds = _nonnegative_finite_float(
        timing.get("total"), context="source timing total"
    )
    if total_seconds < sum(component_seconds):
        raise ValueError("source total timing is shorter than its components")
    invocation_started_at = _aware_datetime(
        record.get("invocation_started_at"), context="source invocation start"
    )
    invocation_finished_at = _aware_datetime(
        record.get("invocation_finished_at"), context="source invocation finish"
    )
    if invocation_finished_at < invocation_started_at:
        raise ValueError("source invocation finishes before it starts")

    return FinalChunkFacts(
        attempted_count=attempted,
        configuration_fingerprint=fingerprint,
        dependency_fingerprint=dependency_fingerprint,
        execution=execution,
        execution_fingerprint=execution_fingerprint,
        generation_compatibility_id=generation_compatibility_id,
        source_fingerprint=source_fingerprint,
        source_sha256=source_sha256,
        execution_platform=execution_platform,
        target=dataset_target,
        trajectory_numerical=trajectory_numerical,
        batch_size=batch_size,
        committed_batches=committed_batches,
        initial_accepted_count=sum(initial_accepted_by_cell.values()),
        initial_attempted_count=sum(initial_attempted_by_cell.values()),
        initial_committed_batches=initial_committed_batches,
        initial_next_batch_id=initial_next_batch_id,
        initial_pending_attempt_count=sum(initial_pending_assignments.values()),
    )


def _artifact_record_matches(
    value: object,
    *,
    expected_path: Path,
    physical: AuthenticatedFile,
    context: str,
) -> None:
    record = _mapping(value, context=context)
    _require_exact_keys(record, {"path", "bytes", "sha256"}, context=context)
    size = _integer(record.get("bytes"), context=f"{context} bytes", minimum=1)
    if (
        record.get("path") != str(expected_path)
        or size != physical.size
        or record.get("sha256") != physical.sha256
    ):
        raise ValueError(f"{context} differs from its physical artifact")


def _relative_source_artifact(
    *,
    root: Path,
    kind: str,
    plan: FinalChunkPlan,
    batch_id: int,
    manifest_parent: Path,
) -> str:
    suffix = "json" if kind == "results" else "npz"
    absolute = root / kind / plan.family / plan.split / f"batch_{batch_id:06d}.{suffix}"
    return os.path.relpath(absolute, start=manifest_parent)


def _reconstruct_dataset_contract(
    plans: Sequence[FinalChunkPlan],
    facts: Sequence[FinalChunkFacts],
) -> dict[str, object]:
    """Recreate the builder-emitted contract from authenticated summaries."""

    if len(plans) != FINAL_SOURCE_COUNT or len(facts) != FINAL_SOURCE_COUNT:
        raise ValueError("dataset contract requires exactly 26 chunk facts")
    target = dict(facts[0].target)
    _require_exact_json(
        target,
        FINAL_DATASET_TARGET,
        context="final shared dataset target",
    )
    if any(not _same_json(fact.target, target) for fact in facts[1:]):
        raise ValueError("final chunk target contracts differ")
    dependency_fingerprints = {fact.dependency_fingerprint for fact in facts}
    if len(dependency_fingerprints) != 1:
        raise ValueError("final chunks use different dependency environments")

    family_numerical: dict[tuple[int, int], Mapping[str, object]] = {}
    family_facts: dict[tuple[int, int], FinalChunkFacts] = {}
    family_compatibility_ids: dict[tuple[int, int], str | None] = {}
    family_variants: dict[
        tuple[int, int],
        dict[tuple[str, str], FinalChunkFacts],
    ] = {}
    for plan, fact in zip(plans, facts):
        family_key = (plan.family_id, plan.revision_id)
        previous = family_facts.setdefault(family_key, fact)
        if previous.execution_platform != fact.execution_platform:
            raise ValueError("final family execution platforms differ")
        previous_compatibility_id = family_compatibility_ids.setdefault(
            family_key, fact.generation_compatibility_id
        )
        if previous_compatibility_id != fact.generation_compatibility_id:
            raise ValueError("final family generation compatibility IDs differ")
        family_variants.setdefault(family_key, {})[
            (fact.execution_fingerprint, fact.source_fingerprint)
        ] = fact
        if fact.trajectory_numerical is not None:
            previous_numerical = family_numerical.setdefault(
                family_key, fact.trajectory_numerical
            )
            if not _same_json(previous_numerical, fact.trajectory_numerical):
                raise ValueError("final family trajectory numerical contracts differ")

    numerical_records = [
        {
            "family_id": family_id,
            "revision_id": revision_id,
            "numerical": dict(numerical),
        }
        for (family_id, revision_id), numerical in sorted(family_numerical.items())
    ]
    numerical_values = tuple(family_numerical.values())
    common_numerical = (
        dict(numerical_values[0])
        if numerical_values
        and all(
            _same_json(numerical, numerical_values[0])
            for numerical in numerical_values[1:]
        )
        else None
    )

    family_revision_records: list[dict[str, object]] = []
    for (family_id, revision_id), fact in sorted(family_facts.items()):
        variants = [
            {
                "execution_record_fingerprint": execution_fingerprint,
                "source_sha256_fingerprint": source_fingerprint,
                "source_sha256": dict(sorted(variant.source_sha256.items())),
            }
            for (
                execution_fingerprint,
                source_fingerprint,
            ), variant in sorted(family_variants[(family_id, revision_id)].items())
        ]
        record: dict[str, object] = {
            "family_id": family_id,
            "revision_id": revision_id,
            "execution_platform": fact.execution_platform,
            "generation_variants": variants,
        }
        compatibility_id = family_compatibility_ids[(family_id, revision_id)]
        if compatibility_id is not None:
            record["generation_compatibility_id"] = compatibility_id
        if len(variants) == 1:
            record["source_sha256_fingerprint"] = variants[0][
                "source_sha256_fingerprint"
            ]
        family_revision_records.append(record)
    generation_identity: dict[str, object] = {
        "dependency_environment_fingerprint": next(iter(dependency_fingerprints)),
        "family_revisions": family_revision_records,
    }
    generation_identity["compatibility_fingerprint"] = _canonical_json_sha256(
        generation_identity
    )
    return {
        "target": target,
        "trajectory_numerical": common_numerical,
        "trajectory_numerical_by_family_revision": numerical_records,
        "stored_dtypes": {
            "eta": "float32",
            "xi": "float32",
            "gxi": "float32",
            "depth": "float64",
            "time": "float64",
        },
        "whole_case_rows": True,
        "generation_identity": generation_identity,
    }


def _validate_combined_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    trajectory_map_path: Path,
    trajectory_map_sha256: str,
    plans: Sequence[FinalChunkPlan],
    facts: Sequence[FinalChunkFacts],
    attempted_by_split: Mapping[str, int],
    attempted_total: int,
    fingerprints: Sequence[str],
    expected_dataset_contract: Mapping[str, object],
) -> tuple[int, int]:
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "configuration_fingerprint",
            "configuration_fingerprints",
            "dataset_contract",
            "dataset_contract_fingerprint",
            "dataset_batches",
            "dataset_shards",
            "trajectory_map_npz",
            "trajectory_map_sha256",
            "requires_trajectory_map",
            "n_rows",
            "n_trajectories",
            "n_accepted_trajectories",
            "n_accepted_rows",
            "grid",
            "split_counts",
        },
        context="combined manifest",
    )
    expected_split_counts = {
        split: {
            "attempted": attempted_by_split[split],
            "accepted": len(FINAL_FAMILY_ORDER) * FINAL_CASES_BY_SPLIT[split],
        }
        for split in FINAL_SPLIT_ORDER
    }
    schema_version = _integer(
        manifest.get("schema_version"), context="combined manifest schema version"
    )
    n_rows = _integer(manifest.get("n_rows"), context="combined manifest rows")
    n_trajectories = _integer(
        manifest.get("n_trajectories"), context="combined manifest trajectories"
    )
    n_accepted_trajectories = _integer(
        manifest.get("n_accepted_trajectories"),
        context="combined manifest accepted trajectories",
    )
    n_accepted_rows = _integer(
        manifest.get("n_accepted_rows"),
        context="combined manifest accepted rows",
    )
    if (
        schema_version != 2
        or manifest.get("configuration_fingerprint") is not None
        or manifest.get("configuration_fingerprints") != list(fingerprints)
        or manifest.get("trajectory_map_npz") != trajectory_map_path.name
        or manifest.get("trajectory_map_sha256") != trajectory_map_sha256
        or manifest.get("requires_trajectory_map") is not True
        or n_rows != FINAL_RETAINED_ROWS
        or n_trajectories != attempted_total
        or n_accepted_trajectories != FINAL_ACCEPTED_CASES
        or n_accepted_rows != FINAL_RETAINED_ROWS
    ):
        raise ValueError("combined manifest identity or counts differ")
    _require_exact_json(
        manifest.get("grid"),
        {"length": 2.0 * math.pi, "nx": 1024},
        context="combined manifest grid",
    )
    _require_exact_json(
        manifest.get("split_counts"),
        expected_split_counts,
        context="combined manifest split counts",
    )

    dataset_contract = _mapping(
        manifest.get("dataset_contract"), context="combined dataset contract"
    )
    _require_exact_json(
        dataset_contract,
        expected_dataset_contract,
        context="combined dataset contract",
    )
    contract_fingerprint = _digest(
        manifest.get("dataset_contract_fingerprint"),
        context="combined dataset contract fingerprint",
    )
    if contract_fingerprint != _canonical_json_sha256(dataset_contract):
        raise ValueError("combined dataset contract fingerprint differs")

    batches = _sequence(manifest.get("dataset_batches"), context="combined batches")
    shards = _sequence(manifest.get("dataset_shards"), context="combined shards")
    expected_batch_count = sum(fact.committed_batches for fact in facts)
    if len(batches) != expected_batch_count or not batches:
        raise ValueError("combined manifest has the wrong batch count")

    batch_cursor = 0
    shard_cursor = 0
    manifest_parent = manifest_path.parent
    for plan, fact in zip(plans, facts):
        chunk_attempted = 0
        chunk_accepted = 0
        chunk_rows = 0
        initial_prefix_attempted = 0
        initial_committed_prefix_attempted = 0
        initial_prefix_accepted = 0
        for batch_id in range(fact.committed_batches):
            batch = _mapping(
                batches[batch_cursor], context=f"combined batch {batch_cursor}"
            )
            _require_exact_keys(
                batch,
                {
                    "proposal_path",
                    "proposal_sha256",
                    "result_path",
                    "result_sha256",
                    "shard_index",
                    "configuration_fingerprint",
                    "family_id",
                    "revision_id",
                    "split_id",
                    "batch_id",
                    "n_attempted_trajectories",
                    "n_accepted_trajectories",
                    "n_rows",
                },
                context=f"combined batch {batch_cursor}",
            )
            attempted = _integer(
                batch.get("n_attempted_trajectories"),
                context="combined batch attempted count",
                minimum=1,
            )
            accepted = _integer(
                batch.get("n_accepted_trajectories"),
                context="combined batch accepted count",
            )
            rows = _integer(batch.get("n_rows"), context="combined batch rows")
            expected_attempted = min(
                fact.batch_size,
                plan.accepted_count - chunk_accepted,
            )
            if (
                attempted != expected_attempted
                or accepted > attempted
                or rows != accepted * FINAL_ROWS_PER_CASE[plan.family]
            ):
                raise ValueError("combined batch attempted/accepted/row counts differ")
            if batch_id < fact.initial_next_batch_id:
                initial_prefix_attempted += attempted
            if batch_id < fact.initial_committed_batches:
                initial_committed_prefix_attempted += attempted
                initial_prefix_accepted += accepted
            raw_shard_index = batch.get("shard_index")
            shard_index = (
                None
                if raw_shard_index is None
                else _integer(raw_shard_index, context="combined batch shard index")
            )
            family_id = _integer(
                batch.get("family_id"), context="combined batch family ID"
            )
            revision_id = _integer(
                batch.get("revision_id"), context="combined batch revision ID"
            )
            split_id = _integer(
                batch.get("split_id"), context="combined batch split ID"
            )
            observed_batch_id = _integer(
                batch.get("batch_id"), context="combined batch ID"
            )
            expected_proposal = _relative_source_artifact(
                root=plan.root,
                kind="proposals",
                plan=plan,
                batch_id=batch_id,
                manifest_parent=manifest_parent,
            )
            expected_result = _relative_source_artifact(
                root=plan.root,
                kind="results",
                plan=plan,
                batch_id=batch_id,
                manifest_parent=manifest_parent,
            )
            expected_shard_index = shard_cursor if rows else None
            if (
                batch.get("proposal_path") != expected_proposal
                or batch.get("result_path") != expected_result
                or shard_index != expected_shard_index
                or batch.get("configuration_fingerprint")
                != fact.configuration_fingerprint
                or family_id != plan.family_id
                or revision_id != plan.revision_id
                or split_id != plan.split_id
                or observed_batch_id != batch_id
            ):
                raise ValueError("combined batch identity or path differs")
            _digest(batch.get("proposal_sha256"), context="combined proposal SHA-256")
            _digest(batch.get("result_sha256"), context="combined result SHA-256")
            if rows:
                if shard_cursor >= len(shards):
                    raise ValueError("combined manifest omits a referenced shard")
                shard = _mapping(
                    shards[shard_cursor], context=f"combined shard {shard_cursor}"
                )
                _require_exact_keys(
                    shard,
                    {
                        "path",
                        "sha256",
                        "n_rows",
                        "batch_index",
                        "configuration_fingerprint",
                    },
                    context=f"combined shard {shard_cursor}",
                )
                expected_shard = _relative_source_artifact(
                    root=plan.root,
                    kind="shards",
                    plan=plan,
                    batch_id=batch_id,
                    manifest_parent=manifest_parent,
                )
                shard_rows = _integer(
                    shard.get("n_rows"), context="combined shard rows"
                )
                shard_batch_index = _integer(
                    shard.get("batch_index"),
                    context="combined shard batch index",
                )
                if (
                    shard.get("path") != expected_shard
                    or shard_rows != rows
                    or shard_batch_index != batch_cursor
                    or shard.get("configuration_fingerprint")
                    != fact.configuration_fingerprint
                ):
                    raise ValueError("combined shard identity or path differs")
                _digest(shard.get("sha256"), context="combined shard SHA-256")
                shard_cursor += 1
            chunk_attempted += attempted
            chunk_accepted += accepted
            chunk_rows += rows
            batch_cursor += 1
        if (
            chunk_attempted != fact.attempted_count
            or chunk_accepted != plan.accepted_count
            or chunk_rows != plan.accepted_count * FINAL_ROWS_PER_CASE[plan.family]
            or initial_prefix_attempted != fact.initial_attempted_count
            or initial_committed_prefix_attempted
            != fact.initial_attempted_count - fact.initial_pending_attempt_count
            or initial_prefix_accepted != fact.initial_accepted_count
        ):
            raise ValueError("combined manifest chunk totals differ")
    if batch_cursor != len(batches) or shard_cursor != len(shards):
        raise ValueError("combined manifest contains extra batch or shard records")
    return len(batches), len(shards)


def _terminally_reauthenticate_final_artifacts(
    artifacts: Sequence[AuthenticatedFile],
) -> None:
    """Recheck a cooperatively stable, stopped-generation 29-parent set.

    This is not an atomic hostile-concurrency snapshot: a mutation after one
    artifact's final hash remains outside the claim.
    """

    if len(artifacts) != 29 or len({artifact.path for artifact in artifacts}) != 29:
        raise RuntimeError("final release parent set is not exactly 29 unique files")
    for expected in artifacts:
        observed = _read_regular_file(expected.path, capture=False)
        if (
            observed.size,
            observed.sha256,
            observed.identity,
        ) != (
            expected.size,
            expected.sha256,
            expected.identity,
        ):
            raise RuntimeError(
                f"final release parent changed during authentication: {expected.path}"
            )


def _authenticate_final_view(path: Path, *, repository_root: Path) -> None:
    repository_root = _resolve(repository_root)
    expected_summary = repository_root / FINAL_VIEW_RELATIVE_PATH
    if path != expected_summary:
        raise ValueError("final summary path differs from the canonical c16384 path")
    summary_physical = _read_regular_file(path, capture=True)
    authenticated_artifacts = [summary_physical]
    summary_bytes = summary_physical.data
    assert summary_bytes is not None
    summary = _strict_json_object(summary_bytes, context="final combined summary")
    _require_exact_keys(
        summary,
        {
            "schema",
            "status",
            "preflight",
            "dataset_view",
            "timing_seconds",
            "invocation_started_at",
            "invocation_finished_at",
        },
        context="final combined summary",
    )
    if (
        summary.get("schema") != "paper_dataset_combined_view_summary_v1"
        or summary.get("status") != "complete"
    ):
        raise ValueError("final combined summary is not complete schema v1")
    timing = _mapping(summary.get("timing_seconds"), context="final timing")
    _require_exact_keys(timing, {"total"}, context="final timing")
    _nonnegative_finite_float(timing.get("total"), context="final timing total")
    invocation_started_at = _aware_datetime(
        summary.get("invocation_started_at"), context="final invocation start"
    )
    invocation_finished_at = _aware_datetime(
        summary.get("invocation_finished_at"), context="final invocation finish"
    )
    if invocation_finished_at < invocation_started_at:
        raise ValueError("final invocation finishes before it starts")

    preflight = _mapping(summary.get("preflight"), context="final preflight")
    _require_exact_keys(
        preflight,
        {
            "schema",
            "mode",
            "no_view_written",
            "output_root",
            "view_name",
            "splits",
            "accepted_cases_per_family_by_split",
            "accepted_cases_total",
            "attempted_cases_total",
            "attempted_cases_by_split",
            "expected_rows",
            "expected_rows_by_split_and_family",
            "configuration_fingerprints",
            "chunks",
        },
        context="final preflight",
    )
    expected_output_root = path.parent
    expected_rows_by_split_and_family = {
        split: {
            family: FINAL_CASES_BY_SPLIT[split] * FINAL_ROWS_PER_CASE[family]
            for family in FINAL_FAMILY_ORDER
        }
        for split in FINAL_SPLIT_ORDER
    }
    accepted_cases_total = _integer(
        preflight.get("accepted_cases_total"),
        context="final preflight accepted cases total",
    )
    attempted_cases_total = _integer(
        preflight.get("attempted_cases_total"),
        context="final preflight attempted cases total",
    )
    expected_rows = _integer(
        preflight.get("expected_rows"), context="final preflight expected rows"
    )
    if (
        preflight.get("schema") != "paper_dataset_combined_view_preflight_v1"
        or preflight.get("mode") != "dry_run"
        or preflight.get("no_view_written") is not True
        or preflight.get("output_root") != str(expected_output_root)
        or preflight.get("view_name") != FINAL_VIEW_NAME
        or preflight.get("splits") != list(FINAL_SPLIT_ORDER)
        or accepted_cases_total != FINAL_ACCEPTED_CASES
        or expected_rows != FINAL_RETAINED_ROWS
    ):
        raise ValueError("final preflight schema, plan, or counts differ")
    _require_exact_json(
        preflight.get("accepted_cases_per_family_by_split"),
        FINAL_CASES_BY_SPLIT,
        context="final preflight accepted cases by split",
    )
    _require_exact_json(
        preflight.get("expected_rows_by_split_and_family"),
        expected_rows_by_split_and_family,
        context="final preflight expected rows by split and family",
    )

    plans = _final_chunk_plans(repository_root)
    raw_chunks = _sequence(preflight.get("chunks"), context="final chunks")
    if len(raw_chunks) != FINAL_SOURCE_COUNT:
        raise ValueError("final preflight does not contain exactly 26 chunks")
    facts: list[FinalChunkFacts] = []
    fingerprints: list[str] = []
    for index, (raw_chunk, plan) in enumerate(zip(raw_chunks, plans)):
        chunk = _mapping(raw_chunk, context=f"final chunk {index}")
        _require_exact_keys(
            chunk,
            {
                "family",
                "revision_id",
                "split",
                "stream_id",
                "accepted_before",
                "accepted_count",
                "accepted_after",
                "attempted_count",
                "configuration_fingerprint",
                "dependency_fingerprint",
                "execution_fingerprint",
                "generation_compatibility_id",
                "source_fingerprint",
                "execution_platform",
                "summary_path",
                "summary_sha256",
                "committed_batches",
            },
            context=f"final chunk {index}",
        )
        chunk_revision_id = _integer(
            chunk.get("revision_id"), context=f"final chunk {index} revision ID"
        )
        chunk_stream_id = _integer(
            chunk.get("stream_id"), context=f"final chunk {index} stream ID"
        )
        chunk_accepted_before = _integer(
            chunk.get("accepted_before"),
            context=f"final chunk {index} accepted before",
        )
        chunk_accepted_count = _integer(
            chunk.get("accepted_count"),
            context=f"final chunk {index} accepted count",
            minimum=1,
        )
        chunk_accepted_after = _integer(
            chunk.get("accepted_after"),
            context=f"final chunk {index} accepted after",
            minimum=1,
        )
        chunk_attempted_count = _integer(
            chunk.get("attempted_count"),
            context=f"final chunk {index} attempted count",
            minimum=1,
        )
        chunk_committed_batches = _integer(
            chunk.get("committed_batches"),
            context=f"final chunk {index} committed batches",
            minimum=1,
        )
        expected_identity = (
            plan.family,
            plan.revision_id,
            plan.split,
            plan.stream_id,
            plan.accepted_before,
            plan.accepted_count,
            plan.accepted_before + plan.accepted_count,
            str(plan.summary_path),
        )
        observed_identity = (
            chunk.get("family"),
            chunk_revision_id,
            chunk.get("split"),
            chunk_stream_id,
            chunk_accepted_before,
            chunk_accepted_count,
            chunk_accepted_after,
            chunk.get("summary_path"),
        )
        if observed_identity != expected_identity:
            raise ValueError(f"final chunk {index} differs from the canonical plan")
        source_physical = _read_regular_file(plan.summary_path, capture=True)
        authenticated_artifacts.append(source_physical)
        if chunk.get("summary_sha256") != source_physical.sha256:
            raise ValueError(f"final chunk {index} source-summary SHA-256 differs")
        source_bytes = source_physical.data
        assert source_bytes is not None
        source = _strict_json_object(
            source_bytes, context=f"final chunk {index} source summary"
        )
        fact = _validate_source_summary(source, plan=plan)
        expected_facts = (
            fact.attempted_count,
            fact.configuration_fingerprint,
            fact.dependency_fingerprint,
            fact.execution_fingerprint,
            fact.generation_compatibility_id,
            fact.source_fingerprint,
            fact.execution_platform,
            fact.committed_batches,
        )
        observed_facts = (
            chunk_attempted_count,
            chunk.get("configuration_fingerprint"),
            chunk.get("dependency_fingerprint"),
            chunk.get("execution_fingerprint"),
            chunk.get("generation_compatibility_id"),
            chunk.get("source_fingerprint"),
            chunk.get("execution_platform"),
            chunk_committed_batches,
        )
        if observed_facts != expected_facts:
            raise ValueError(f"final chunk {index} differs from its source summary")
        facts.append(fact)
        fingerprints.append(fact.configuration_fingerprint)

    if len(set(fingerprints)) != FINAL_SOURCE_COUNT:
        raise ValueError("final chunk configuration fingerprints are not unique")
    sorted_fingerprints = sorted(fingerprints)
    if preflight.get("configuration_fingerprints") != sorted_fingerprints:
        raise ValueError("final preflight fingerprint list differs")
    if len({fact.dependency_fingerprint for fact in facts}) != 1:
        raise ValueError("final chunks use different dependency environments")
    for source_path in (
        "solver/gen_data/pipeline/reference.py",
        "solver/solvers/dno_series_jax.py",
    ):
        shared_digests = {fact.source_sha256.get(source_path) for fact in facts}
        if None in shared_digests or len(shared_digests) != 1:
            raise ValueError(f"final chunks differ on shared source {source_path}")
    for family in FINAL_FAMILY_ORDER:
        family_facts = tuple(
            fact for plan, fact in zip(plans, facts) if plan.family == family
        )
        if (
            len({fact.source_fingerprint for fact in family_facts}) != 1
            or len({fact.execution_fingerprint for fact in family_facts}) != 1
            or len({fact.execution_platform for fact in family_facts}) != 1
        ):
            raise ValueError(f"final {family} chunks use inconsistent identities")

    attempted_by_split = {
        split: sum(
            fact.attempted_count
            for plan, fact in zip(plans, facts)
            if plan.split == split
        )
        for split in FINAL_SPLIT_ORDER
    }
    attempted_total = sum(attempted_by_split.values())
    if attempted_cases_total != attempted_total:
        raise ValueError("final preflight attempted counts differ")
    _require_exact_json(
        preflight.get("attempted_cases_by_split"),
        attempted_by_split,
        context="final preflight attempted cases by split",
    )

    view = _mapping(summary.get("dataset_view"), context="final dataset view")
    _require_exact_keys(
        view,
        {
            "schema_version",
            "configuration_fingerprint",
            "configuration_fingerprints",
            "trajectory_map_npz",
            "requires_trajectory_map",
            "n_rows",
            "n_trajectories",
            "n_accepted_trajectories",
            "n_accepted_rows",
            "grid",
            "dataset_contract_fingerprint",
            "manifest_source_audit",
            "trajectory_map_audit",
            "manifest",
            "trajectory_map",
        },
        context="final dataset view",
    )
    manifest_path = expected_output_root / f"{FINAL_VIEW_NAME}.dataset.json"
    trajectory_map_path = expected_output_root / f"{FINAL_VIEW_NAME}.trajectory_map.npz"
    manifest_physical = _read_regular_file(manifest_path, capture=True)
    trajectory_map_physical = _read_regular_file(trajectory_map_path, capture=False)
    authenticated_artifacts.extend((manifest_physical, trajectory_map_physical))
    _artifact_record_matches(
        view.get("manifest"),
        expected_path=manifest_path,
        physical=manifest_physical,
        context="final manifest record",
    )
    _artifact_record_matches(
        view.get("trajectory_map"),
        expected_path=trajectory_map_path,
        physical=trajectory_map_physical,
        context="final trajectory-map record",
    )
    view_schema_version = _integer(
        view.get("schema_version"), context="final view schema version"
    )
    view_rows = _integer(view.get("n_rows"), context="final view rows")
    view_trajectories = _integer(
        view.get("n_trajectories"), context="final view trajectories"
    )
    view_accepted_trajectories = _integer(
        view.get("n_accepted_trajectories"),
        context="final view accepted trajectories",
    )
    view_accepted_rows = _integer(
        view.get("n_accepted_rows"), context="final view accepted rows"
    )
    if (
        view_schema_version != 2
        or view.get("configuration_fingerprint") is not None
        or view.get("configuration_fingerprints") != sorted_fingerprints
        or view.get("trajectory_map_npz") != trajectory_map_path.name
        or view.get("requires_trajectory_map") is not True
        or view_rows != FINAL_RETAINED_ROWS
        or view_trajectories != attempted_total
        or view_accepted_trajectories != FINAL_ACCEPTED_CASES
        or view_accepted_rows != FINAL_RETAINED_ROWS
    ):
        raise ValueError("final dataset-view schema, identity, or counts differ")
    _require_exact_json(
        view.get("grid"),
        {"length": 2.0 * math.pi, "nx": 1024},
        context="final view grid",
    )
    _validate_trajectory_map_npz(
        trajectory_map_path,
        attempted_total=attempted_total,
    )

    manifest_bytes = manifest_physical.data
    assert manifest_bytes is not None
    manifest = _strict_json_object(manifest_bytes, context="final combined manifest")
    expected_dataset_contract = _reconstruct_dataset_contract(plans, facts)
    batch_count, shard_count = _validate_combined_manifest(
        manifest,
        manifest_path=manifest_path,
        trajectory_map_path=trajectory_map_path,
        trajectory_map_sha256=trajectory_map_physical.sha256,
        plans=plans,
        facts=facts,
        attempted_by_split=attempted_by_split,
        attempted_total=attempted_total,
        fingerprints=sorted_fingerprints,
        expected_dataset_contract=expected_dataset_contract,
    )
    if view.get("dataset_contract_fingerprint") != manifest.get(
        "dataset_contract_fingerprint"
    ):
        raise ValueError("final dataset-contract fingerprint differs from manifest")
    expected_manifest_audit = {
        "committed_batches": batch_count,
        "committed_shards": shard_count,
    }
    _require_exact_json(
        view.get("manifest_source_audit"),
        expected_manifest_audit,
        context="final manifest-source audit",
    )
    accepted_by_split_and_family = {
        split: {family: FINAL_CASES_BY_SPLIT[split] for family in FINAL_FAMILY_ORDER}
        for split in FINAL_SPLIT_ORDER
    }
    accepted_rows_by_split_and_family = expected_rows_by_split_and_family
    attempted_by_split_and_family = {
        split: {
            family: sum(
                fact.attempted_count
                for plan, fact in zip(plans, facts)
                if plan.split == split and plan.family == family
            )
            for family in FINAL_FAMILY_ORDER
        }
        for split in FINAL_SPLIT_ORDER
    }
    expected_map_audit = {
        "schema_version": 2,
        "attempted_cases_by_split_and_family": attempted_by_split_and_family,
        "accepted_cases_by_split_and_family": accepted_by_split_and_family,
        "accepted_rows_by_split_and_family": accepted_rows_by_split_and_family,
        "total_row_ownership": {
            "attempted_trajectories": attempted_total,
            "declared_rows": FINAL_RETAINED_ROWS,
            "row_owner_entries": FINAL_RETAINED_ROWS,
            "expected_rows": FINAL_RETAINED_ROWS,
        },
        "source_binding": {
            "committed_batches": batch_count,
            "committed_shards": shard_count,
            "source_trajectories": attempted_total,
            "source_rows": FINAL_RETAINED_ROWS,
        },
    }
    _require_exact_json(
        view.get("trajectory_map_audit"),
        expected_map_audit,
        context="final trajectory-map audit",
    )
    _terminally_reauthenticate_final_artifacts(authenticated_artifacts)


def final_audited_view_exists(
    path: Path,
    *,
    repository_root: Path = ROOT,
) -> bool:
    """Authenticate the exact final view without reading numerical shards."""

    try:
        _authenticate_final_view(Path(path), repository_root=repository_root)
    except (OSError, TypeError, ValueError, RuntimeError, UnicodeError):
        return False
    return True


def should_exit_for_final_view(
    *,
    active_workers: int,
    path: Path,
    repository_root: Path = ROOT,
) -> bool:
    """Never perform final-artifact I/O while a declared worker is active."""

    return active_workers == 0 and final_audited_view_exists(
        path,
        repository_root=repository_root,
    )


def positive_float(value: str) -> float:
    parsed = float(value)
    if not parsed > 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=positive_float, default=60.0)
    parser.add_argument(
        "--jonswap-stale-seconds",
        type=positive_float,
        # Revision-4 has already committed a valid 4.218-hour transaction
        # under contention.  Keep enough margin that a slow but progressing
        # outer batch is not mistaken for a hung worker.
        default=6.0 * 60.0 * 60.0,
    )
    parser.add_argument(
        "--tanaka-stale-seconds",
        type=positive_float,
        # The longest comparable 256-case revision-3 transaction committed in
        # 2.370 hours.  Six hours leaves more than a 2.5x margin for the fresh
        # source-frozen dataset, for which no completed transaction exists yet.
        default=6.0 * 60.0 * 60.0,
    )
    parser.add_argument(
        "--kill-grace-seconds",
        type=nonnegative_float,
        default=120.0,
    )
    parser.add_argument(
        "--log-interval-seconds",
        type=positive_float,
        default=15.0 * 60.0,
    )
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--final-view-summary",
        type=Path,
        default=DEFAULT_FINAL_VIEW_SUMMARY,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log_path = None if str(args.log_file) == "-" else _resolve(args.log_file)
    logger = TimestampedLogger(log_path)
    plans = build_worker_plans()
    matcher = ExactWorkerMatcher(
        plans,
        checkout_root=ROOT,
        owner_uid=os.getuid(),
    )
    source = ProcfsProcessSource()
    signal_sender = SafeSignalSender(source, matcher, dry_run=args.dry_run)
    watchdog = WorkerWatchdog(
        source=source,
        matcher=matcher,
        signal_sender=signal_sender,
        logger=logger,
        stale_seconds={
            "jonswap_tma": args.jonswap_stale_seconds,
            "tanaka": args.tanaka_stale_seconds,
        },
        kill_grace_seconds=args.kill_grace_seconds,
        log_interval_seconds=args.log_interval_seconds,
        dry_run=args.dry_run,
    )
    logger.write(
        "watchdog_started "
        f"owner_uid={os.getuid()} poll_seconds={args.poll_seconds:g} "
        f"jonswap_stale_seconds={args.jonswap_stale_seconds:g} "
        f"tanaka_stale_seconds={args.tanaka_stale_seconds:g} "
        f"kill_grace_seconds={args.kill_grace_seconds:g} "
        f"dry_run={str(args.dry_run).lower()}"
    )

    stop_requested = False

    def request_stop(signal_number: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        logger.write(f"watchdog_stop_requested signal={signal_number}")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    while not stop_requested:
        active_workers = watchdog.tick(time.monotonic())
        final_view_summary = args.final_view_summary
        if should_exit_for_final_view(
            active_workers=active_workers,
            path=final_view_summary,
        ):
            logger.write(
                "final_audited_view_exists; watchdog_exiting "
                f"summary={final_view_summary}"
            )
            return 0
        if args.once:
            return 0
        time.sleep(args.poll_seconds)
    logger.write("watchdog_stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
