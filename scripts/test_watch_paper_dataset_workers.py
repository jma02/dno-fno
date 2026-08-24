"""Focused safety and state-transition tests for the dataset watchdog."""

from __future__ import annotations

import copy
from dataclasses import replace
from functools import cache
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import watch_paper_dataset_workers as watchdog_module
from scripts.watch_paper_dataset_workers import (
    ExactWorkerMatcher,
    FileToken,
    HeartbeatToken,
    MatchedWorker,
    ProcessSnapshot,
    ProcfsProcessSource,
    SafeSignalSender,
    WorkerPlan,
    WorkerWatchdog,
    build_worker_plans,
    final_audited_view_exists,
    parse_args,
    read_heartbeat_token,
    should_exit_for_final_view,
)


class MutableProcessSource:
    """In-memory process table for deterministic liveness tests."""

    def __init__(self, processes: tuple[ProcessSnapshot, ...] = ()) -> None:
        self.processes = {process.pid: process for process in processes}

    def snapshots(self) -> tuple[ProcessSnapshot, ...]:
        return tuple(self.processes.values())

    def read_process(self, pid: int) -> ProcessSnapshot | None:
        return self.processes.get(pid)


class RecordingSignalSender:
    """Record signals without touching an operating-system process."""

    def __init__(self) -> None:
        self.events: list[tuple[int, int]] = []

    def send(self, worker: MatchedWorker, signal_number: int) -> bool:
        self.events.append((worker.process.pid, signal_number))
        return True


class MemoryLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def write(self, message: str) -> None:
        self.messages.append(message)


class MutableTokenReader:
    def __init__(self, token: HeartbeatToken) -> None:
        self.token = token

    def __call__(self, _plan: WorkerPlan) -> HeartbeatToken:
        return self.token


def token(batch: int) -> HeartbeatToken:
    return HeartbeatToken(
        proposal=FileToken(
            relative_path=f"proposals/batch_{batch:06d}.npz",
            mtime_ns=batch,
            size=batch + 1,
        ),
        result=None,
        shard=None,
    )


@cache
def _canonical_trajectory_map_bytes(
    attempted_total: int,
    mutation: str | None = None,
) -> bytes:
    """Build one compact, structurally canonical all-zero map fixture."""

    output = io.BytesIO()
    zeros = b"\0" * (1024 * 1024)
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
    ) as archive:
        for name, (
            descriptor,
            item_size,
            shape_kind,
        ) in watchdog_module.FINAL_TRAJECTORY_MAP_DTYPES.items():
            if mutation == "missing_member" and name == "frame_index":
                continue
            shape = {
                "schema": (),
                "row": (watchdog_module.FINAL_RETAINED_ROWS,),
                "trajectory": (attempted_total,),
            }[shape_kind]
            if mutation == "wrong_shape" and name == "trajectory_family_id":
                shape = (attempted_total + 1,)
            if mutation == "wrong_dtype" and name == "trajectory_family_id":
                descriptor = "<u2"
            header_text = repr(
                {
                    "descr": descriptor,
                    "fortran_order": False,
                    "shape": shape,
                }
            )
            padding = (-(10 + len(header_text) + 1)) % 16
            header = (header_text + " " * padding + "\n").encode("latin-1")
            prefix = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header
            element_count = 1 if not shape else shape[0]
            payload_size = element_count * item_size
            member_name = f"{name}.npy"
            member_target: str | zipfile.ZipInfo = member_name
            if name == "trajectory_index" and mutation in {
                "bad_bzip2",
                "bad_lzma",
            }:
                member_target = zipfile.ZipInfo(member_name)
                member_target.compress_type = {
                    "bad_bzip2": zipfile.ZIP_BZIP2,
                    "bad_lzma": zipfile.ZIP_LZMA,
                }[mutation]
            with archive.open(member_target, "w") as member:
                member.write(prefix)
                if shape_kind == "schema":
                    member.write(
                        struct.pack("<h", 3 if mutation == "wrong_schema" else 2)
                    )
                    continue
                remaining = payload_size
                while remaining:
                    block = zeros[: min(remaining, len(zeros))]
                    member.write(block)
                    remaining -= len(block)
    return output.getvalue()


def _corrupt_zip_member_payload(raw: bytes, member_name: str) -> bytes:
    forged = bytearray(raw)
    with zipfile.ZipFile(io.BytesIO(forged), "r") as archive:
        info = archive.getinfo(member_name)
    name_length, extra_length = struct.unpack_from(
        "<HH",
        forged,
        info.header_offset + 26,
    )
    data_start = info.header_offset + 30 + name_length + extra_length
    forged[data_start + info.compress_size // 2] ^= 1
    return bytes(forged)


def _replace_zip_member_compression(
    raw: bytes,
    member_name: str,
    compression: int,
) -> bytes:
    forged = bytearray(raw)
    with zipfile.ZipFile(io.BytesIO(forged), "r") as archive:
        info = archive.getinfo(member_name)
    struct.pack_into("<H", forged, info.header_offset + 8, compression)

    end_record = forged.rfind(b"PK\x05\x06")
    if end_record < 0:
        raise AssertionError("fixture ZIP has no end record")
    central_offset = struct.unpack_from("<L", forged, end_record + 16)[0]
    cursor = central_offset
    while cursor < end_record:
        if forged[cursor : cursor + 4] != b"PK\x01\x02":
            raise AssertionError("fixture ZIP has an invalid central directory")
        name_length, extra_length, comment_length = struct.unpack_from(
            "<HHH",
            forged,
            cursor + 28,
        )
        name_start = cursor + 46
        name_end = name_start + name_length
        if bytes(forged[name_start:name_end]).decode("utf-8") == member_name:
            struct.pack_into("<H", forged, cursor + 10, compression)
            return bytes(forged)
        cursor = name_end + extra_length + comment_length
    raise AssertionError(f"fixture ZIP has no {member_name!r} member")


class DatasetWorkerWatchdogTest(unittest.TestCase):
    temporary: tempfile.TemporaryDirectory[str]
    checkout: Path
    plans: tuple[WorkerPlan, ...]
    plan: WorkerPlan
    matcher: ExactWorkerMatcher

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.checkout = Path(self.temporary.name).resolve()
        python = self.checkout / ".venv/bin/python"
        jonswap = self.checkout / "outputs/jonswap"
        tanaka = self.checkout / "outputs/tanaka"
        self.plans = build_worker_plans(
            checkout_root=self.checkout,
            python=python,
            jonswap_root=jonswap,
            tanaka_root=tanaka,
        )
        self.plan = self.plans[0]
        self.matcher = ExactWorkerMatcher(
            self.plans,
            checkout_root=self.checkout,
            owner_uid=os.getuid(),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _snapshot(
        self,
        plan: WorkerPlan | None = None,
        *,
        pid: int = 101,
        start_ticks: int = 7_001,
        uid: int | None = None,
        cwd: Path | None = None,
        argv: tuple[str, ...] | None = None,
        gpu: str | None = None,
    ) -> ProcessSnapshot:
        selected = self.plan if plan is None else plan
        return ProcessSnapshot(
            pid=pid,
            start_ticks=start_ticks,
            uid=os.getuid() if uid is None else uid,
            cwd=self.checkout if cwd is None else cwd,
            argv=selected.argv if argv is None else argv,
            environment={
                "CUDA_VISIBLE_DEVICES": (selected.gpu_index if gpu is None else gpu)
            },
        )

    def _watchdog(
        self,
        source: MutableProcessSource,
        sender: RecordingSignalSender,
        tokens: MutableTokenReader,
        logger: MemoryLogger | None = None,
        *,
        stale: float = 10.0,
        grace: float = 2.0,
    ) -> WorkerWatchdog:
        return WorkerWatchdog(
            source=source,
            matcher=self.matcher,
            signal_sender=sender,
            logger=MemoryLogger() if logger is None else logger,
            stale_seconds={"jonswap_tma": stale, "tanaka": stale},
            kill_grace_seconds=grace,
            log_interval_seconds=5.0,
            token_reader=tokens,
        )

    def test_exact_matcher_rejects_every_identity_mutation(self) -> None:
        exact = self._snapshot()
        self.assertEqual(self.matcher.match(exact), self.plan)

        output_index = self.plan.argv.index("--output-root") + 1
        wrong_root_argv = list(self.plan.argv)
        wrong_root_argv[output_index] = str(self.checkout / "outputs/unrelated")
        mutations = (
            replace(exact, uid=os.getuid() + 1),
            replace(exact, cwd=self.checkout / "other"),
            replace(exact, argv=tuple(wrong_root_argv)),
            replace(exact, argv=("sleep", "999")),
            replace(exact, environment={"CUDA_VISIBLE_DEVICES": "7"}),
        )
        self.assertTrue(all(self.matcher.match(case) is None for case in mutations))

    def test_unmatched_processes_are_never_signaled(self) -> None:
        unrelated = replace(self._snapshot(), argv=("python", "train.py"))
        source = MutableProcessSource((unrelated,))
        sender = RecordingSignalSender()
        watchdog = self._watchdog(source, sender, MutableTokenReader(token(0)))

        watchdog.tick(0.0)
        watchdog.tick(1_000.0)

        self.assertEqual(sender.events, [])
        self.assertEqual(dict(watchdog.states), {})

    def test_default_timeouts_exceed_observed_successful_batch_durations(self) -> None:
        args = parse_args([])

        # Revision-4 has a durable successful 4.218-hour transaction, while
        # the longest comparable Tanaka transaction observed is 2.370 hours.
        self.assertEqual(args.jonswap_stale_seconds, 6.0 * 60.0 * 60.0)
        self.assertGreater(args.jonswap_stale_seconds, 4.218 * 60.0 * 60.0)
        self.assertGreater(args.tanaka_stale_seconds, 2.370 * 60.0 * 60.0)

    def test_stale_worker_transitions_from_term_to_kill_after_grace(self) -> None:
        source = MutableProcessSource((self._snapshot(),))
        sender = RecordingSignalSender()
        logger = MemoryLogger()
        watchdog = self._watchdog(
            source,
            sender,
            MutableTokenReader(token(0)),
            logger,
        )

        watchdog.tick(0.0)
        watchdog.tick(10.0)
        watchdog.tick(11.99)
        watchdog.tick(12.0)

        self.assertEqual(
            sender.events,
            [(101, signal.SIGTERM), (101, signal.SIGKILL)],
        )
        self.assertTrue(any("sent_term" in message for message in logger.messages))
        self.assertTrue(any("sent_kill" in message for message in logger.messages))

    def test_transaction_token_mutation_resets_age_and_pending_kill(self) -> None:
        source = MutableProcessSource((self._snapshot(),))
        sender = RecordingSignalSender()
        tokens = MutableTokenReader(token(0))
        watchdog = self._watchdog(source, sender, tokens)

        watchdog.tick(0.0)
        watchdog.tick(10.0)
        self.assertEqual(sender.events, [(101, signal.SIGTERM)])

        tokens.token = token(1)
        watchdog.tick(11.0)
        watchdog.tick(20.99)
        self.assertEqual(sender.events, [(101, signal.SIGTERM)])

        watchdog.tick(21.0)
        self.assertEqual(
            sender.events,
            [(101, signal.SIGTERM), (101, signal.SIGTERM)],
        )

    def test_pid_or_start_time_change_gets_a_fresh_clock(self) -> None:
        source = MutableProcessSource((self._snapshot(),))
        sender = RecordingSignalSender()
        watchdog = self._watchdog(source, sender, MutableTokenReader(token(0)))
        watchdog.tick(0.0)

        replacement = self._snapshot(pid=202, start_ticks=9_000)
        source.processes = {replacement.pid: replacement}
        watchdog.tick(100.0)

        self.assertEqual(sender.events, [])
        self.assertEqual(tuple(watchdog.states)[0].pid, 202)

    def test_filesystem_heartbeat_tracks_each_transaction_kind(self) -> None:
        plan = self.plan
        proposal = (
            plan.output_root
            / "proposals"
            / plan.family
            / plan.split
            / "batch_000000.npz"
        )
        result = (
            plan.output_root
            / "results"
            / plan.family
            / plan.split
            / "batch_000000.json"
        )
        shard = (
            plan.output_root / "shards" / plan.family / plan.split / "batch_000000.npz"
        )
        for path in (proposal, result, shard):
            path.parent.mkdir(parents=True, exist_ok=True)

        empty = read_heartbeat_token(plan)
        proposal.write_bytes(b"proposal")
        proposed = read_heartbeat_token(plan)
        shard.write_bytes(b"shard")
        result.write_text("{}", encoding="utf-8")
        committed = read_heartbeat_token(plan)

        self.assertEqual(empty, HeartbeatToken(None, None, None))
        self.assertIsNotNone(proposed.proposal)
        self.assertIsNone(proposed.result)
        self.assertIsNotNone(committed.result)
        self.assertIsNotNone(committed.shard)

    def test_safe_sender_revalidates_start_time_command_and_root(self) -> None:
        expected_process = self._snapshot()
        worker = MatchedWorker(expected_process, self.plan)
        source = MutableProcessSource((expected_process,))
        sender = SafeSignalSender(source, self.matcher)

        with patch("os.kill") as kill:
            self.assertTrue(sender.send(worker, signal.SIGTERM))
            kill.assert_called_once_with(101, signal.SIGTERM)

            source.processes[101] = replace(expected_process, start_ticks=9_999)
            self.assertFalse(sender.send(worker, signal.SIGKILL))
            self.assertEqual(kill.call_count, 1)

            source.processes[101] = replace(
                expected_process,
                argv=("python", "unrelated.py"),
            )
            self.assertFalse(sender.send(worker, signal.SIGKILL))
            self.assertEqual(kill.call_count, 1)

    def test_procfs_reader_integrates_identity_command_cwd_and_environment(
        self,
    ) -> None:
        proc_root = self.checkout / "proc"
        process_root = proc_root / "101"
        process_root.mkdir(parents=True)
        stat_fields = ["R", *("0" for _ in range(18)), "7001"]
        (process_root / "stat").write_text(
            f"101 (python worker) {' '.join(stat_fields)}\n",
            encoding="utf-8",
        )
        (process_root / "status").write_text(
            f"Name:\tpython\nUid:\t{os.getuid()}\t{os.getuid()}\t"
            f"{os.getuid()}\t{os.getuid()}\n",
            encoding="utf-8",
        )
        (process_root / "cmdline").write_bytes(
            b"\0".join(os.fsencode(field) for field in self.plan.argv) + b"\0"
        )
        (process_root / "environ").write_bytes(
            f"CUDA_VISIBLE_DEVICES={self.plan.gpu_index}\0".encode()
        )
        os.symlink(self.checkout, process_root / "cwd", target_is_directory=True)

        source = ProcfsProcessSource(proc_root)
        snapshots = source.snapshots()

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].start_ticks, 7_001)
        self.assertEqual(self.matcher.match(snapshots[0]), self.plan)

    def test_final_view_exit_gate_requires_source_bound_audits_and_counts(self) -> None:
        summary, _ = self._canonical_final_fixture()

        self.assertTrue(
            final_audited_view_exists(summary, repository_root=self.checkout)
        )
        self.assertTrue(
            should_exit_for_final_view(
                active_workers=0,
                path=summary,
                repository_root=self.checkout,
            )
        )

    @staticmethod
    def _write_json(path: Path, record: object) -> tuple[int, str]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        raw = path.read_bytes()
        return len(raw), hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _set_nested(
        record: object,
        coordinates: tuple[str | int, ...],
        value: object,
    ) -> None:
        parent = record
        for coordinate in coordinates[:-1]:
            if isinstance(coordinate, int):
                assert isinstance(parent, list)
                parent = parent[coordinate]
            else:
                assert isinstance(parent, dict)
                parent = parent[coordinate]
        final_coordinate = coordinates[-1]
        if isinstance(final_coordinate, int):
            assert isinstance(parent, list)
            parent[final_coordinate] = value
        else:
            assert isinstance(parent, dict)
            parent[final_coordinate] = value

    @staticmethod
    def _rebind_source_summary(record: dict[str, object]) -> None:
        run_spec = record["run_spec"]
        execution = record["execution"]
        assert isinstance(run_spec, dict) and isinstance(execution, dict)
        fingerprint = watchdog_module._canonical_json_sha256(run_spec)
        record["configuration_fingerprint"] = fingerprint
        view = record["dataset_view"]
        preflight = record["preflight"]
        assert isinstance(view, dict) and isinstance(preflight, dict)
        view["configuration_fingerprint"] = fingerprint
        preflight["configuration_fingerprint"] = fingerprint
        preflight["run_spec"] = copy.deepcopy(run_spec)
        preflight["execution"] = copy.deepcopy(execution)

    def _write_manifest_and_rebind_summary(
        self,
        summary_path: Path,
        summary: dict[str, object],
        manifest: dict[str, object],
    ) -> None:
        manifest_path = summary_path.parent / (
            f"{watchdog_module.FINAL_VIEW_NAME}.dataset.json"
        )
        manifest_bytes, manifest_sha256 = self._write_json(manifest_path, manifest)
        view = summary["dataset_view"]
        assert isinstance(view, dict)
        manifest_record = view["manifest"]
        assert isinstance(manifest_record, dict)
        manifest_record["bytes"] = manifest_bytes
        manifest_record["sha256"] = manifest_sha256
        self._write_json(summary_path, summary)

    def _canonical_final_fixture(self) -> tuple[Path, dict[str, object]]:
        from scripts import run_paper_dataset_quota as quota_module
        from solver.gen_data.stokes_static_pipeline import (
            PAPER_STATIC_STOKES_CONTRACT,
        )
        from solver.gen_data.trajectory_quota_executor import (
            TrajectoryExecutionConfig,
        )

        outputs = self.checkout / "outputs"
        if outputs.exists():
            shutil.rmtree(outputs)
        summary_path = self.checkout / watchdog_module.FINAL_VIEW_RELATIVE_PATH
        output_root = summary_path.parent
        output_root.mkdir(parents=True, exist_ok=True)
        plans = watchdog_module._final_chunk_plans(self.checkout)
        dependency_environment = quota_module.dependency_environment()
        dependency_fingerprint = watchdog_module._canonical_json_sha256(
            dependency_environment
        )
        self.assertEqual(
            dependency_fingerprint,
            watchdog_module.FINAL_DEPENDENCY_FINGERPRINT,
        )
        family_sources = {
            family: quota_module.source_hashes(family)
            for family in watchdog_module.FINAL_FAMILY_ORDER
        }
        family_sources["stokes"].update(
            {
                "scripts/run_paper_dataset_quota.py": (
                    "ab99c067c2f22823fce861a69fd03bd8cc4524effbfd5ab1a5360bfa5e615d98"
                ),
                "solver/gen_data/pipeline/manifest.py": (
                    "e2cc1a20cc01fef9dd0bb84b99485b5cf90da71f7ee035a813893f2f6a37f421"
                ),
                "solver/gen_data/pipeline/production.py": (
                    "8ed36cf1bd57494f344096e4a508bcfb37b9103cf03ace8a63df2656c11dc735"
                ),
            }
        )
        family_sources["benjamin_feir"].update(
            {
                "solver/gen_data/pipeline/production.py": (
                    "2c2234caf1087c2982872eb44ba2abd457cf22b812784e1a3e9bd54e8c7129d4"
                ),
                "solver/gen_data/trajectory_family_adapters.py": (
                    "afb480a64b14a2b311bda067e6638208569cf3acfebc61d67a1f016a73fbef6a"
                ),
            }
        )
        family_sources["jonswap_tma"].update(
            {
                "scripts/run_paper_dataset_jonswap_bucketed.py": (
                    "a6d66274afb4c773d82fe9d7b40b34449089a371bc9defc20f549b9844f2755f"
                ),
                "solver/gen_data/jonswap_horizon_executor.py": (
                    "7cf6333c4df82f313d67cdc21b1fe884a100aee591ede6a0904afefaba013d6f"
                ),
            }
        )
        chunk_records: list[dict[str, object]] = []
        fingerprints: list[str] = []
        source_facts: list[watchdog_module.FinalChunkFacts] = []
        facts_by_plan: list[tuple[object, dict[str, object], list[int]]] = []
        attempted_by_split_and_family = {
            split: {
                family: watchdog_module.FINAL_CASES_BY_SPLIT[split]
                for family in watchdog_module.FINAL_FAMILY_ORDER
            }
            for split in watchdog_module.FINAL_SPLIT_ORDER
        }
        current_executions = {
            "stokes": PAPER_STATIC_STOKES_CONTRACT.to_json_record(),
            **{
                family: TrajectoryExecutionConfig.paper(family).to_json_record()
                for family in ("tanaka", "benjamin_feir", "jonswap_tma")
            },
        }
        self.assertEqual(
            watchdog_module._canonical_json_sha256(current_executions["tanaka"]),
            watchdog_module.TANAKA_CURRENT_EXECUTION_FINGERPRINT,
        )

        for plan in plans:
            execution = copy.deepcopy(current_executions[plan.family])
            cell_ids = quota_module.FAMILY_CELL_IDS[plan.family]
            before_quotient, before_remainder = divmod(
                plan.accepted_before, len(cell_ids)
            )
            after_quotient, after_remainder = divmod(
                plan.accepted_before + plan.accepted_count, len(cell_ids)
            )
            quota_targets = {
                cell_id: (
                    after_quotient
                    + int(index < after_remainder)
                    - before_quotient
                    - int(index < before_remainder)
                )
                for index, cell_id in enumerate(cell_ids)
            }
            source_sha256 = copy.deepcopy(family_sources[plan.family])
            platform = "cpu" if plan.family == "stokes" else "gpu"
            configuration = {
                "schema": "paper_dataset_quota_configuration_v1",
                "purpose": "exact paper-contract accepted-case generation",
                "case_kind": "static" if plan.family == "stokes" else "trajectory",
                "accepted_case_count": plan.accepted_count,
                "accepted_cases_before": plan.accepted_before,
                "accepted_cases_after": plan.accepted_before + plan.accepted_count,
                "ordered_cell_ids": list(cell_ids),
                "dependency_environment": dependency_environment,
                "execution_platform": platform,
                "source_sha256": source_sha256,
                (
                    "contract" if plan.family == "stokes" else "trajectory_execution"
                ): execution,
            }
            if plan.family == "stokes":
                configuration["sampler"] = {"maximum_ursell_redraws": 1_000}
            elif plan.family == "benjamin_feir":
                configuration["sampling_support"] = (
                    quota_module._benjamin_feir_sampling_support_record()
                )
            elif plan.family == "jonswap_tma":
                configuration["jonswap_horizon_bucketing"] = {
                    "outer_proposal_size": 32,
                    "solver_batch_size": 8,
                    "sort_rule": "stable_saved_time_count_then_proposal_index",
                    "commit_order": "durable_proposal_order",
                    "transaction_rule": ("all_solver_groups_then_one_atomic_commit"),
                    "nonlinear_adjustment": copy.deepcopy(
                        execution["jonswap_adjustment"]
                    ),
                }
            batch_size = watchdog_module.FINAL_BATCH_SIZES[plan.family]
            batch_counts: list[int] = []
            remaining = plan.accepted_count
            while remaining:
                batch_count = min(batch_size, remaining)
                batch_counts.append(batch_count)
                remaining -= batch_count
            committed_batches = len(batch_counts)
            run_spec = {
                "schema": "paper_dataset_accepted_quota_run_v2",
                "family_name": plan.family,
                "family_id": plan.family_id,
                "revision_id": plan.revision_id,
                "split_id": plan.split,
                "root_seed": watchdog_module.FINAL_ROOT_SEEDS[plan.split],
                "stream_id": plan.stream_id,
                "first_attempt_index": 0,
                "batch_size": batch_size,
                "quotas": [
                    {"cell_id": cell_id, "target_accepted": quota_targets[cell_id]}
                    for cell_id in cell_ids
                ],
                "cell_codes": {
                    cell_id: index for index, cell_id in enumerate(cell_ids)
                },
                "configuration": configuration,
                "maximum_attempts_per_accepted_case": 4,
            }
            fingerprint = watchdog_module._canonical_json_sha256(run_spec)
            source_fingerprint = watchdog_module._canonical_json_sha256(source_sha256)
            self.assertEqual(
                source_fingerprint,
                watchdog_module.FINAL_SOURCE_FINGERPRINTS[plan.family],
            )
            source_name = f"paper_dataset_{plan.family}_{plan.split}"
            zero_by_cell = {cell_id: 0 for cell_id in cell_ids}
            initial_state = {
                "accepted_by_cell": zero_by_cell,
                "attempted_by_cell": zero_by_cell,
                "committed_batches": 0,
                "next_attempt_index": 0,
                "next_batch_id": 0,
                "pending_batch_id": None,
                "pending_status": None,
                "complete": False,
                "terminal_failure": None,
                "attempt_limit_failure": None,
                "attempt_limit_exhausted_cells": [],
            }
            runtime = {
                "requested_platform": platform,
                "jax_version": dependency_environment["packages"]["jax"],
                "default_backend": platform,
                "x64_enabled": True,
                "devices": [
                    {
                        "id": 0,
                        "platform": platform,
                        "device_kind": f"fixture {platform}",
                    }
                ],
            }
            allocation_quotas = [
                {
                    "cell_id": cell_id,
                    "accepted_before": before_quotient + int(index < before_remainder),
                    "chunk_target_accepted": quota_targets[cell_id],
                    "accepted_after": after_quotient + int(index < after_remainder),
                    "attempt_ceiling": 4 * quota_targets[cell_id],
                    "durable_attempted": 0,
                    "remaining_attempt_capacity": 4 * quota_targets[cell_id],
                }
                for index, cell_id in enumerate(cell_ids)
            ]
            source_preflight = {
                "schema": "paper_dataset_quota_preflight_v1",
                "mode": "dry_run",
                "no_numerical_generation_performed": True,
                "output_root": str(plan.root),
                "artifact_namespace": {
                    "family": plan.family,
                    "split": plan.split,
                    "view_name": source_name,
                    "summary_path": str(plan.summary_path),
                },
                "configuration_fingerprint": fingerprint,
                "run_spec": run_spec,
                "execution": execution,
                "allocation": {
                    "chunk_accepted_cases": plan.accepted_count,
                    "accepted_cases_before": plan.accepted_before,
                    "accepted_cases_after": (
                        plan.accepted_before + plan.accepted_count
                    ),
                    "cell_count": len(cell_ids),
                    "nonzero_quota_cell_count": sum(
                        target > 0 for target in quota_targets.values()
                    ),
                    "quota_minimum": min(quota_targets.values()),
                    "quota_maximum": max(quota_targets.values()),
                    "maximum_attempts_per_accepted_case": 4,
                    "quotas": allocation_quotas,
                    "chunk_identity": {
                        "stream_id": plan.stream_id,
                        "first_attempt_index": 0,
                        "output_root": str(plan.root),
                        "rule": (
                            "use a distinct stream_id and output_root for every "
                            "additive chunk"
                        ),
                    },
                },
                "expected_output": {
                    "stored_rows_per_accepted_case": (
                        watchdog_module.FINAL_ROWS_PER_CASE[plan.family]
                    ),
                    "retained_rows": (
                        plan.accepted_count
                        * watchdog_module.FINAL_ROWS_PER_CASE[plan.family]
                    ),
                    "spatial_points_per_row": 1_024,
                    "field_values_per_row": 3_072,
                },
                "resume_state": initial_state,
                "runtime": runtime,
            }
            if plan.family != "stokes":
                source_preflight["revision_id"] = plan.revision_id
            source_summary = {
                "schema": "paper_dataset_quota_summary_v1",
                "status": "complete",
                "configuration_fingerprint": fingerprint,
                "output_root": str(plan.root),
                "run_spec": run_spec,
                "execution": execution,
                "runtime": runtime,
                "preflight": source_preflight,
                "resume": {
                    "initial": initial_state,
                    "final": {
                        "accepted_by_cell": quota_targets,
                        "attempted_by_cell": quota_targets,
                        "committed_batches": committed_batches,
                        "next_attempt_index": plan.accepted_count,
                        "next_batch_id": committed_batches,
                        "pending_batch_id": None,
                        "pending_status": None,
                        "complete": True,
                        "terminal_failure": None,
                        "attempt_limit_failure": None,
                        "attempt_limit_exhausted_cells": [],
                    },
                },
                "counts": {
                    "accepted": plan.accepted_count,
                    "attempted": plan.accepted_count,
                    "rejected": 0,
                    "by_cell": {
                        cell_id: {
                            "accepted": quota_targets[cell_id],
                            "attempted": quota_targets[cell_id],
                            "rejected": 0,
                            "target_accepted": quota_targets[cell_id],
                        }
                        for cell_id in cell_ids
                    },
                    "rejection_reasons": {},
                },
                "dataset_view": {
                    "schema_version": 2,
                    "configuration_fingerprint": fingerprint,
                    "n_rows": (
                        plan.accepted_count
                        * watchdog_module.FINAL_ROWS_PER_CASE[plan.family]
                    ),
                    "n_trajectories": plan.accepted_count,
                    "n_accepted_trajectories": plan.accepted_count,
                    "n_accepted_rows": (
                        plan.accepted_count
                        * watchdog_module.FINAL_ROWS_PER_CASE[plan.family]
                    ),
                    "grid": {"length": 2.0 * 3.141592653589793, "nx": 1024},
                    "manifest": {
                        "path": f"{source_name}.dataset.json",
                        "bytes": 1,
                        "sha256": "8" * 64,
                    },
                    "trajectory_map": {
                        "path": f"{source_name}.trajectory_map.npz",
                        "bytes": 1,
                        "sha256": "9" * 64,
                    },
                },
                "timing_seconds": {
                    "quota_driver": 0.4,
                    "dataset_view": 0.2,
                    "validation": 0.1,
                    "total": 1.0,
                },
                "invocation_started_at": "2026-08-09T00:00:00+00:00",
                "invocation_finished_at": "2026-08-09T00:00:01+00:00",
            }
            _, source_summary_sha256 = self._write_json(
                plan.summary_path, source_summary
            )
            source_facts.append(
                watchdog_module._validate_source_summary(source_summary, plan=plan)
            )
            execution_fingerprint = watchdog_module._canonical_json_sha256(execution)
            chunk_record = {
                "family": plan.family,
                "revision_id": plan.revision_id,
                "split": plan.split,
                "stream_id": plan.stream_id,
                "accepted_before": plan.accepted_before,
                "accepted_count": plan.accepted_count,
                "accepted_after": plan.accepted_before + plan.accepted_count,
                "attempted_count": plan.accepted_count,
                "configuration_fingerprint": fingerprint,
                "dependency_fingerprint": dependency_fingerprint,
                "execution_fingerprint": execution_fingerprint,
                "generation_compatibility_id": None,
                "source_fingerprint": source_fingerprint,
                "execution_platform": platform,
                "summary_path": str(plan.summary_path),
                "summary_sha256": source_summary_sha256,
                "committed_batches": committed_batches,
            }
            chunk_records.append(chunk_record)
            fingerprints.append(fingerprint)
            facts_by_plan.append((plan, chunk_record, batch_counts))

        sorted_fingerprints = sorted(fingerprints)
        attempted_by_split = {
            split: len(watchdog_module.FINAL_FAMILY_ORDER)
            * watchdog_module.FINAL_CASES_BY_SPLIT[split]
            for split in watchdog_module.FINAL_SPLIT_ORDER
        }
        attempted_total = sum(attempted_by_split.values())
        trajectory_map_path = (
            output_root / f"{watchdog_module.FINAL_VIEW_NAME}.trajectory_map.npz"
        )
        trajectory_map_path.write_bytes(
            _canonical_trajectory_map_bytes(attempted_total)
        )
        trajectory_map_bytes = trajectory_map_path.read_bytes()
        trajectory_map_sha256 = hashlib.sha256(trajectory_map_bytes).hexdigest()

        dataset_batches: list[dict[str, object]] = []
        dataset_shards: list[dict[str, object]] = []
        for plan, chunk, batch_counts in facts_by_plan:
            for batch_id, accepted_count in enumerate(batch_counts):
                batch_index = len(dataset_batches)
                shard_index = len(dataset_shards)
                rows = accepted_count * watchdog_module.FINAL_ROWS_PER_CASE[plan.family]
                dataset_batches.append(
                    {
                        "proposal_path": watchdog_module._relative_source_artifact(
                            root=plan.root,
                            kind="proposals",
                            plan=plan,
                            batch_id=batch_id,
                            manifest_parent=output_root,
                        ),
                        "proposal_sha256": "a" * 64,
                        "result_path": watchdog_module._relative_source_artifact(
                            root=plan.root,
                            kind="results",
                            plan=plan,
                            batch_id=batch_id,
                            manifest_parent=output_root,
                        ),
                        "result_sha256": "b" * 64,
                        "shard_index": shard_index,
                        "configuration_fingerprint": chunk["configuration_fingerprint"],
                        "family_id": plan.family_id,
                        "revision_id": plan.revision_id,
                        "split_id": plan.split_id,
                        "batch_id": batch_id,
                        "n_attempted_trajectories": accepted_count,
                        "n_accepted_trajectories": accepted_count,
                        "n_rows": rows,
                    }
                )
                dataset_shards.append(
                    {
                        "path": watchdog_module._relative_source_artifact(
                            root=plan.root,
                            kind="shards",
                            plan=plan,
                            batch_id=batch_id,
                            manifest_parent=output_root,
                        ),
                        "sha256": "c" * 64,
                        "n_rows": rows,
                        "batch_index": batch_index,
                        "configuration_fingerprint": chunk["configuration_fingerprint"],
                    }
                )
        dataset_contract = watchdog_module._reconstruct_dataset_contract(
            plans, source_facts
        )
        dataset_contract_fingerprint = watchdog_module._canonical_json_sha256(
            dataset_contract
        )
        manifest = {
            "schema_version": 2,
            "configuration_fingerprint": None,
            "configuration_fingerprints": sorted_fingerprints,
            "dataset_contract": dataset_contract,
            "dataset_contract_fingerprint": dataset_contract_fingerprint,
            "dataset_batches": dataset_batches,
            "dataset_shards": dataset_shards,
            "trajectory_map_npz": trajectory_map_path.name,
            "trajectory_map_sha256": trajectory_map_sha256,
            "requires_trajectory_map": True,
            "n_rows": watchdog_module.FINAL_RETAINED_ROWS,
            "n_trajectories": attempted_total,
            "n_accepted_trajectories": watchdog_module.FINAL_ACCEPTED_CASES,
            "n_accepted_rows": watchdog_module.FINAL_RETAINED_ROWS,
            "grid": {"length": 2.0 * 3.141592653589793, "nx": 1024},
            "split_counts": {
                split: {
                    "attempted": attempted_by_split[split],
                    "accepted": attempted_by_split[split],
                }
                for split in watchdog_module.FINAL_SPLIT_ORDER
            },
        }
        manifest_path = output_root / f"{watchdog_module.FINAL_VIEW_NAME}.dataset.json"
        manifest_bytes, manifest_sha256 = self._write_json(manifest_path, manifest)
        expected_rows = {
            split: {
                family: watchdog_module.FINAL_CASES_BY_SPLIT[split]
                * watchdog_module.FINAL_ROWS_PER_CASE[family]
                for family in watchdog_module.FINAL_FAMILY_ORDER
            }
            for split in watchdog_module.FINAL_SPLIT_ORDER
        }
        manifest_audit = {
            "committed_batches": len(dataset_batches),
            "committed_shards": len(dataset_shards),
        }
        map_audit = {
            "schema_version": 2,
            "attempted_cases_by_split_and_family": attempted_by_split_and_family,
            "accepted_cases_by_split_and_family": attempted_by_split_and_family,
            "accepted_rows_by_split_and_family": expected_rows,
            "total_row_ownership": {
                "attempted_trajectories": attempted_total,
                "declared_rows": watchdog_module.FINAL_RETAINED_ROWS,
                "row_owner_entries": watchdog_module.FINAL_RETAINED_ROWS,
                "expected_rows": watchdog_module.FINAL_RETAINED_ROWS,
            },
            "source_binding": {
                "committed_batches": len(dataset_batches),
                "committed_shards": len(dataset_shards),
                "source_trajectories": attempted_total,
                "source_rows": watchdog_module.FINAL_RETAINED_ROWS,
            },
        }
        final_summary: dict[str, object] = {
            "schema": "paper_dataset_combined_view_summary_v1",
            "status": "complete",
            "preflight": {
                "schema": "paper_dataset_combined_view_preflight_v1",
                "mode": "dry_run",
                "no_view_written": True,
                "output_root": str(output_root),
                "view_name": watchdog_module.FINAL_VIEW_NAME,
                "splits": list(watchdog_module.FINAL_SPLIT_ORDER),
                "accepted_cases_per_family_by_split": dict(
                    watchdog_module.FINAL_CASES_BY_SPLIT
                ),
                "accepted_cases_total": watchdog_module.FINAL_ACCEPTED_CASES,
                "attempted_cases_total": attempted_total,
                "attempted_cases_by_split": attempted_by_split,
                "expected_rows": watchdog_module.FINAL_RETAINED_ROWS,
                "expected_rows_by_split_and_family": expected_rows,
                "configuration_fingerprints": sorted_fingerprints,
                "chunks": chunk_records,
            },
            "dataset_view": {
                "schema_version": 2,
                "configuration_fingerprint": None,
                "configuration_fingerprints": sorted_fingerprints,
                "trajectory_map_npz": trajectory_map_path.name,
                "requires_trajectory_map": True,
                "n_rows": watchdog_module.FINAL_RETAINED_ROWS,
                "n_trajectories": attempted_total,
                "n_accepted_trajectories": watchdog_module.FINAL_ACCEPTED_CASES,
                "n_accepted_rows": watchdog_module.FINAL_RETAINED_ROWS,
                "grid": {"length": 2.0 * 3.141592653589793, "nx": 1024},
                "dataset_contract_fingerprint": dataset_contract_fingerprint,
                "manifest_source_audit": manifest_audit,
                "trajectory_map_audit": map_audit,
                "manifest": {
                    "path": str(manifest_path),
                    "bytes": manifest_bytes,
                    "sha256": manifest_sha256,
                },
                "trajectory_map": {
                    "path": str(trajectory_map_path),
                    "bytes": len(trajectory_map_bytes),
                    "sha256": trajectory_map_sha256,
                },
            },
            "timing_seconds": {"total": 1.0},
            "invocation_started_at": "2026-08-09T00:00:00+00:00",
            "invocation_finished_at": "2026-08-09T00:00:01+00:00",
        }
        self._write_json(summary_path, final_summary)
        return summary_path, final_summary

    def test_active_worker_skips_final_artifact_authentication(self) -> None:
        path = self.checkout / watchdog_module.FINAL_VIEW_RELATIVE_PATH
        with patch.object(
            watchdog_module,
            "final_audited_view_exists",
            side_effect=AssertionError("must not authenticate while active"),
        ) as authenticate:
            self.assertFalse(
                should_exit_for_final_view(
                    active_workers=1,
                    path=path,
                    repository_root=self.checkout,
                )
            )
        authenticate.assert_not_called()

    def test_valid_final_authentication_reads_exactly_29_small_parents_twice(
        self,
    ) -> None:
        summary_path, _ = self._canonical_final_fixture()
        original_reader = watchdog_module._read_regular_file
        read_paths: list[Path] = []

        def record_read(
            path: Path,
            *,
            capture: bool,
        ) -> watchdog_module.AuthenticatedFile:
            read_paths.append(path)
            return original_reader(path, capture=capture)

        with patch.object(
            watchdog_module,
            "_read_regular_file",
            side_effect=record_read,
        ):
            self.assertTrue(
                final_audited_view_exists(
                    summary_path,
                    repository_root=self.checkout,
                )
            )
        unique_paths = set(read_paths)
        self.assertEqual(len(read_paths), 58)
        self.assertEqual(len(unique_paths), 29)
        self.assertTrue(all(read_paths.count(path) == 2 for path in unique_paths))
        self.assertFalse(
            any(
                directory in path.parts
                for path in unique_paths
                for directory in ("proposals", "results", "shards")
            )
        )

    def test_idle_main_treats_symlink_loop_summary_as_unauthenticated(self) -> None:
        first = self.checkout / "summary-loop-a"
        second = self.checkout / "summary-loop-b"
        first.symlink_to(second)
        second.symlink_to(first)
        with (
            patch.object(watchdog_module.WorkerWatchdog, "tick", return_value=0),
            patch.object(watchdog_module.signal, "signal"),
            patch.object(watchdog_module.TimestampedLogger, "write"),
        ):
            self.assertEqual(
                watchdog_module.main(
                    [
                        "--once",
                        "--final-view-summary",
                        str(first),
                        "--log-file",
                        str(self.checkout / "watchdog.log"),
                    ]
                ),
                0,
            )

    def test_tanaka_compatibility_resolution_is_current_first_and_pair_exact(
        self,
    ) -> None:
        current = watchdog_module.TANAKA_CURRENT_EXECUTION_FINGERPRINT
        self.assertIsNone(
            watchdog_module._generation_compatibility_id(
                family="tanaka",
                execution_fingerprint=current,
                source_fingerprint="d" * 64,
            )
        )
        self.assertEqual(
            watchdog_module._generation_compatibility_id(
                family="tanaka",
                execution_fingerprint=(
                    watchdog_module.TANAKA_LEGACY_EXECUTION_FINGERPRINT
                ),
                source_fingerprint=watchdog_module.TANAKA_LEGACY_SOURCE_FINGERPRINT,
            ),
            current,
        )
        with self.assertRaisesRegex(ValueError, "not current or explicitly bridged"):
            watchdog_module._generation_compatibility_id(
                family="tanaka",
                execution_fingerprint="e" * 64,
                source_fingerprint="f" * 64,
            )

    def test_fresh_current_tanaka_chunks_require_null_compatibility(self) -> None:
        summary_path, record = self._canonical_final_fixture()
        preflight = record["preflight"]
        assert isinstance(preflight, dict)
        chunks = preflight["chunks"]
        assert isinstance(chunks, list)
        tanaka_chunks = [
            chunk
            for chunk in chunks
            if isinstance(chunk, dict) and chunk.get("family") == "tanaka"
        ]
        self.assertEqual(len(tanaka_chunks), 6)
        self.assertTrue(
            all(
                chunk.get("generation_compatibility_id") is None
                for chunk in tanaka_chunks
            )
        )
        self.assertTrue(
            final_audited_view_exists(summary_path, repository_root=self.checkout)
        )

        tanaka_chunks[0]["generation_compatibility_id"] = (
            watchdog_module.TANAKA_CURRENT_EXECUTION_FINGERPRINT
        )
        self._write_json(summary_path, record)
        self.assertFalse(
            final_audited_view_exists(summary_path, repository_root=self.checkout)
        )

    def test_source_summary_rejects_boolean_integer_fields(self) -> None:
        self._canonical_final_fixture()
        plans = watchdog_module._final_chunk_plans(self.checkout)
        source_by_index = {
            index: json.loads(plan.summary_path.read_text(encoding="utf-8"))
            for index, plan in enumerate(plans)
            if index in {0, 4}
        }
        stokes = source_by_index[0]
        tanaka = source_by_index[4]
        stokes_run_spec = stokes["run_spec"]
        tanaka_run_spec = tanaka["run_spec"]
        assert isinstance(stokes_run_spec, dict)
        assert isinstance(tanaka_run_spec, dict)
        stokes_cells = stokes_run_spec["configuration"]["ordered_cell_ids"]
        tanaka_cells = tanaka_run_spec["configuration"]["ordered_cell_ids"]
        assert isinstance(stokes_cells, list) and isinstance(tanaka_cells, list)
        stokes_cell = stokes_cells[0]
        tanaka_cell = tanaka_cells[0]
        assert isinstance(stokes_cell, str) and isinstance(tanaka_cell, str)
        mutations = (
            ("family_id", 0, ("run_spec", "family_id"), True),
            ("revision_id", 0, ("run_spec", "revision_id"), True),
            ("stream_id", 0, ("run_spec", "stream_id"), False),
            ("first_attempt", 0, ("run_spec", "first_attempt_index"), False),
            (
                "maximum_attempts",
                0,
                ("run_spec", "maximum_attempts_per_accepted_case"),
                True,
            ),
            ("root_seed", 0, ("run_spec", "root_seed"), True),
            ("batch_size", 0, ("run_spec", "batch_size"), True),
            (
                "accepted_count",
                0,
                ("run_spec", "configuration", "accepted_case_count"),
                True,
            ),
            (
                "accepted_before",
                0,
                ("run_spec", "configuration", "accepted_cases_before"),
                False,
            ),
            (
                "accepted_after",
                0,
                ("run_spec", "configuration", "accepted_cases_after"),
                True,
            ),
            ("quota_target", 0, ("run_spec", "quotas", 0, "target_accepted"), True),
            ("cell_code", 0, ("run_spec", "cell_codes", stokes_cell), False),
            ("count_accepted", 0, ("counts", "accepted"), True),
            ("count_attempted", 0, ("counts", "attempted"), True),
            ("count_rejected", 0, ("counts", "rejected"), False),
            (
                "cell_accepted",
                0,
                ("counts", "by_cell", stokes_cell, "accepted"),
                True,
            ),
            (
                "cell_attempted",
                0,
                ("counts", "by_cell", stokes_cell, "attempted"),
                True,
            ),
            (
                "cell_rejected",
                0,
                ("counts", "by_cell", stokes_cell, "rejected"),
                False,
            ),
            (
                "cell_target",
                0,
                ("counts", "by_cell", stokes_cell, "target_accepted"),
                True,
            ),
            ("view_schema", 0, ("dataset_view", "schema_version"), True),
            ("view_rows", 0, ("dataset_view", "n_rows"), True),
            (
                "view_trajectories",
                0,
                ("dataset_view", "n_trajectories"),
                True,
            ),
            (
                "view_accepted_trajectories",
                0,
                ("dataset_view", "n_accepted_trajectories"),
                True,
            ),
            (
                "view_accepted_rows",
                0,
                ("dataset_view", "n_accepted_rows"),
                True,
            ),
            ("manifest_bytes", 0, ("dataset_view", "manifest", "bytes"), True),
            (
                "map_bytes",
                0,
                ("dataset_view", "trajectory_map", "bytes"),
                True,
            ),
            (
                "resume_accepted",
                0,
                ("resume", "final", "accepted_by_cell", stokes_cell),
                True,
            ),
            (
                "resume_attempted",
                0,
                ("resume", "final", "attempted_by_cell", stokes_cell),
                True,
            ),
            (
                "resume_batches",
                0,
                ("resume", "final", "committed_batches"),
                True,
            ),
            (
                "resume_next_attempt",
                0,
                ("resume", "final", "next_attempt_index"),
                True,
            ),
            (
                "resume_next_batch",
                0,
                ("resume", "final", "next_batch_id"),
                True,
            ),
            (
                "preflight_revision",
                4,
                ("preflight", "revision_id"),
                True,
            ),
            (
                "tanaka_resume_accepted",
                4,
                ("resume", "final", "accepted_by_cell", tanaka_cell),
                True,
            ),
        )
        for name, plan_index, coordinates, value in mutations:
            with self.subTest(field=name):
                candidate = copy.deepcopy(source_by_index[plan_index])
                self._set_nested(candidate, coordinates, value)
                assert isinstance(candidate, dict)
                if coordinates[0] == "run_spec":
                    self._rebind_source_summary(candidate)
                with self.assertRaises(TypeError):
                    watchdog_module._validate_source_summary(
                        candidate,
                        plan=plans[plan_index],
                    )

    def test_source_summary_requires_builder_canonical_specs(self) -> None:
        self._canonical_final_fixture()
        plans = watchdog_module._final_chunk_plans(self.checkout)
        stokes_plan = plans[0]
        bf_index = next(
            index for index, plan in enumerate(plans) if plan.family == "benjamin_feir"
        )
        source_records = {
            0: json.loads(stokes_plan.summary_path.read_text(encoding="utf-8")),
            bf_index: json.loads(
                plans[bf_index].summary_path.read_text(encoding="utf-8")
            ),
        }
        for mutation in ("root_seed", "taxonomy", "quota", "cell_codes", "execution"):
            with self.subTest(mutation=mutation):
                plan_index = bf_index if mutation == "execution" else 0
                candidate = copy.deepcopy(source_records[plan_index])
                assert isinstance(candidate, dict)
                run_spec = candidate["run_spec"]
                assert isinstance(run_spec, dict)
                configuration = run_spec["configuration"]
                assert isinstance(configuration, dict)
                if mutation == "root_seed":
                    run_spec["root_seed"] = (
                        watchdog_module.FINAL_ROOT_SEEDS[plans[plan_index].split] + 1
                    )
                elif mutation == "taxonomy":
                    cells = configuration["ordered_cell_ids"]
                    assert isinstance(cells, list)
                    cells[0], cells[1] = cells[1], cells[0]
                elif mutation == "quota":
                    quotas = run_spec["quotas"]
                    assert isinstance(quotas, list)
                    assert isinstance(quotas[0], dict) and isinstance(quotas[1], dict)
                    quotas[0]["target_accepted"] += 1
                    quotas[1]["target_accepted"] -= 1
                elif mutation == "cell_codes":
                    codes = run_spec["cell_codes"]
                    cells = configuration["ordered_cell_ids"]
                    assert isinstance(codes, dict) and isinstance(cells, list)
                    codes[cells[0]], codes[cells[1]] = codes[cells[1]], codes[cells[0]]
                else:
                    execution = copy.deepcopy(candidate["execution"])
                    assert isinstance(execution, dict)
                    numerical = execution["numerical"]
                    assert isinstance(numerical, dict)
                    numerical["dno_order"] = 6
                    candidate["execution"] = execution
                    configuration["trajectory_execution"] = copy.deepcopy(execution)
                self._rebind_source_summary(candidate)
                with self.assertRaises(ValueError):
                    watchdog_module._validate_source_summary(
                        candidate,
                        plan=plans[plan_index],
                    )

    def test_source_summary_requires_exact_release_configuration(self) -> None:
        self._canonical_final_fixture()
        plans = watchdog_module._final_chunk_plans(self.checkout)
        indexes = {
            family: next(
                index for index, plan in enumerate(plans) if plan.family == family
            )
            for family in watchdog_module.FINAL_FAMILY_ORDER
        }
        sources = {
            family: json.loads(plans[index].summary_path.read_text(encoding="utf-8"))
            for family, index in indexes.items()
        }
        mutations = (
            ("batch_size", "stokes"),
            ("purpose", "stokes"),
            ("sampler", "stokes"),
            ("dependency", "tanaka"),
            ("source_map", "tanaka"),
            ("sampling_support", "benjamin_feir"),
            ("jonswap_policy", "jonswap_tma"),
        )
        for mutation, family in mutations:
            with self.subTest(mutation=mutation):
                candidate = copy.deepcopy(sources[family])
                assert isinstance(candidate, dict)
                run_spec = candidate["run_spec"]
                assert isinstance(run_spec, dict)
                configuration = run_spec["configuration"]
                assert isinstance(configuration, dict)
                if mutation == "batch_size":
                    run_spec["batch_size"] += 1
                elif mutation == "purpose":
                    configuration["purpose"] = "forged"
                elif mutation == "sampler":
                    configuration["sampler"]["maximum_ursell_redraws"] += 1
                elif mutation == "dependency":
                    packages = configuration["dependency_environment"]["packages"]
                    packages["jax"] = "forged"
                    runtime = candidate["runtime"]
                    preflight = candidate["preflight"]
                    runtime["jax_version"] = "forged"
                    preflight["runtime"]["jax_version"] = "forged"
                elif mutation == "source_map":
                    source_map = configuration["source_sha256"]
                    first_path = next(iter(source_map))
                    source_map[first_path] = "0" * 64
                elif mutation == "sampling_support":
                    configuration["sampling_support"]["schema"] = "forged"
                else:
                    configuration["jonswap_horizon_bucketing"]["solver_batch_size"] = 7
                self._rebind_source_summary(candidate)
                with self.assertRaises(ValueError):
                    watchdog_module._validate_source_summary(
                        candidate,
                        plan=plans[indexes[family]],
                    )

    def test_final_view_rejects_source_attestation_forgery_matrix(self) -> None:
        summary_path, summary = self._canonical_final_fixture()
        plan = watchdog_module._final_chunk_plans(self.checkout)[0]
        source = json.loads(plan.summary_path.read_text(encoding="utf-8"))
        mutations = (
            ("runtime", ("runtime", "x64_enabled"), False, True),
            (
                "preflight_runtime",
                ("preflight", "runtime", "jax_version"),
                "forged",
                False,
            ),
            ("timing", ("timing_seconds", "total"), 0.0, False),
            (
                "timestamp",
                ("invocation_finished_at",),
                "2026-08-08T23:59:59+00:00",
                False,
            ),
            (
                "resume_initial",
                ("resume", "initial", "forged"),
                1,
                True,
            ),
            (
                "resume_binding",
                ("preflight", "resume_state", "forged"),
                1,
                False,
            ),
            (
                "namespace",
                ("preflight", "artifact_namespace", "family"),
                "forged",
                False,
            ),
            (
                "allocation",
                ("preflight", "allocation", "chunk_accepted_cases"),
                1,
                False,
            ),
            (
                "expected_output",
                ("preflight", "expected_output", "retained_rows"),
                1,
                False,
            ),
        )
        for name, coordinates, value, mirror_preflight in mutations:
            with self.subTest(field=name):
                candidate_source = copy.deepcopy(source)
                self._set_nested(candidate_source, coordinates, value)
                if mirror_preflight and coordinates[0] == "runtime":
                    self._set_nested(
                        candidate_source,
                        ("preflight", *coordinates),
                        value,
                    )
                elif mirror_preflight and coordinates[:2] == ("resume", "initial"):
                    self._set_nested(
                        candidate_source,
                        ("preflight", "resume_state", *coordinates[2:]),
                        value,
                    )
                _, source_sha256 = self._write_json(
                    plan.summary_path,
                    candidate_source,
                )
                candidate_summary = copy.deepcopy(summary)
                preflight = candidate_summary["preflight"]
                assert isinstance(preflight, dict)
                chunks = preflight["chunks"]
                assert isinstance(chunks, list) and isinstance(chunks[0], dict)
                chunks[0]["summary_sha256"] = source_sha256
                self._write_json(summary_path, candidate_summary)
                self.assertFalse(
                    final_audited_view_exists(
                        summary_path,
                        repository_root=self.checkout,
                    )
                )

    def test_final_view_rejects_coherent_per_cell_attempt_ceiling_forgery(
        self,
    ) -> None:
        summary_path, summary = self._canonical_final_fixture()
        plans = watchdog_module._final_chunk_plans(self.checkout)
        plan = plans[0]
        source = json.loads(plan.summary_path.read_text(encoding="utf-8"))
        run_spec = source["run_spec"]
        counts = source["counts"]
        resume = source["resume"]
        source_view = source["dataset_view"]
        assert isinstance(run_spec, dict)
        assert isinstance(counts, dict)
        assert isinstance(resume, dict)
        assert isinstance(source_view, dict)
        configuration = run_spec["configuration"]
        by_cell = counts["by_cell"]
        final_state = resume["final"]
        assert isinstance(configuration, dict)
        assert isinstance(by_cell, dict)
        assert isinstance(final_state, dict)
        ordered_cells = configuration["ordered_cell_ids"]
        assert isinstance(ordered_cells, list)
        cell_id = ordered_cells[0]
        assert isinstance(cell_id, str)
        cell = by_cell[cell_id]
        accepted_by_cell = final_state["accepted_by_cell"]
        attempted_by_cell = final_state["attempted_by_cell"]
        assert isinstance(cell, dict)
        assert isinstance(accepted_by_cell, dict)
        assert isinstance(attempted_by_cell, dict)
        target = cell["target_accepted"]
        maximum_attempts = run_spec["maximum_attempts_per_accepted_case"]
        previous_cell_attempted = cell["attempted"]
        assert isinstance(target, int) and not isinstance(target, bool)
        assert isinstance(maximum_attempts, int) and not isinstance(
            maximum_attempts, bool
        )
        assert isinstance(previous_cell_attempted, int) and not isinstance(
            previous_cell_attempted, bool
        )
        forged_cell_attempted = maximum_attempts * target + 1
        attempted_delta = forged_cell_attempted - previous_cell_attempted
        forged_attempted = counts["attempted"] + attempted_delta
        forged_rejected = forged_attempted - counts["accepted"]
        cell["attempted"] = forged_cell_attempted
        cell["rejected"] = forged_cell_attempted - cell["accepted"]
        attempted_by_cell[cell_id] = forged_cell_attempted
        counts["attempted"] = forged_attempted
        counts["rejected"] = forged_rejected
        counts["rejection_reasons"] = {"forged_retry_overflow": forged_rejected}
        final_state["next_attempt_index"] = forged_attempted
        source_view["n_trajectories"] = forged_attempted

        with self.assertRaisesRegex(ValueError, "per-cell count arithmetic"):
            watchdog_module._validate_source_summary(source, plan=plan)
        _, source_sha256 = self._write_json(plan.summary_path, source)

        preflight = summary["preflight"]
        final_view = summary["dataset_view"]
        assert isinstance(preflight, dict) and isinstance(final_view, dict)
        chunks = preflight["chunks"]
        attempted_by_split = preflight["attempted_cases_by_split"]
        map_audit = final_view["trajectory_map_audit"]
        assert isinstance(chunks, list)
        assert isinstance(chunks[0], dict)
        assert isinstance(attempted_by_split, dict)
        assert isinstance(map_audit, dict)
        chunks[0]["attempted_count"] = forged_attempted
        chunks[0]["summary_sha256"] = source_sha256
        preflight["attempted_cases_total"] += attempted_delta
        attempted_by_split["train"] += attempted_delta
        final_view["n_trajectories"] += attempted_delta
        attempted_by_family = map_audit["attempted_cases_by_split_and_family"]
        total_ownership = map_audit["total_row_ownership"]
        source_binding = map_audit["source_binding"]
        assert isinstance(attempted_by_family, dict)
        assert isinstance(attempted_by_family["train"], dict)
        assert isinstance(total_ownership, dict)
        assert isinstance(source_binding, dict)
        attempted_by_family["train"]["stokes"] += attempted_delta
        total_ownership["attempted_trajectories"] += attempted_delta
        source_binding["source_trajectories"] += attempted_delta

        manifest_path = summary_path.parent / (
            f"{watchdog_module.FINAL_VIEW_NAME}.dataset.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["n_trajectories"] += attempted_delta
        manifest["split_counts"]["train"]["attempted"] += attempted_delta
        manifest["dataset_batches"][0]["n_attempted_trajectories"] = forged_attempted
        self._write_manifest_and_rebind_summary(summary_path, summary, manifest)

        self.assertFalse(
            final_audited_view_exists(
                summary_path,
                repository_root=self.checkout,
            )
        )

    def test_final_view_rejects_impossible_initial_quota_prefixes(self) -> None:
        summary_path, summary = self._canonical_final_fixture()
        plan = watchdog_module._final_chunk_plans(self.checkout)[0]
        source = json.loads(plan.summary_path.read_text(encoding="utf-8"))
        run_spec = source["run_spec"]
        assert isinstance(run_spec, dict)
        configuration = run_spec["configuration"]
        assert isinstance(configuration, dict)
        cell_ids = configuration["ordered_cell_ids"]
        assert isinstance(cell_ids, list) and isinstance(cell_ids[0], str)
        first_cell = cell_ids[0]
        for mutation in (
            "oversized_pending",
            "wrong_pending_distribution",
            "wrong_committed_distribution",
            "pending_after_complete",
        ):
            with self.subTest(mutation=mutation):
                candidate_source = copy.deepcopy(source)
                resume = candidate_source["resume"]
                preflight = candidate_source["preflight"]
                assert isinstance(resume, dict) and isinstance(preflight, dict)
                final_state = resume["final"]
                assert isinstance(final_state, dict)
                if mutation in {"oversized_pending", "wrong_pending_distribution"}:
                    initial = resume["initial"]
                    assert isinstance(initial, dict)
                    attempted_by_cell = initial["attempted_by_cell"]
                    assert isinstance(attempted_by_cell, dict)
                    if mutation == "oversized_pending":
                        attempted_by_cell[first_cell] = 1_000
                        attempted_total = 1_000
                    else:
                        for cell_id, attempted_count in zip(
                            cell_ids,
                            (9, 7, 8, 8),
                        ):
                            attempted_by_cell[cell_id] = attempted_count
                        attempted_total = 32
                    initial["next_attempt_index"] = attempted_total
                    initial["next_batch_id"] = 1
                    initial["pending_batch_id"] = 0
                    initial["pending_status"] = "proposed"
                    allocation = preflight["allocation"]
                    assert isinstance(allocation, dict)
                    quotas = allocation["quotas"]
                    assert isinstance(quotas, list) and isinstance(quotas[0], dict)
                    for quota in quotas:
                        assert isinstance(quota, dict)
                        cell_id = quota["cell_id"]
                        durable = attempted_by_cell[cell_id]
                        quota["durable_attempted"] = durable
                        quota["remaining_attempt_capacity"] -= durable
                elif mutation == "wrong_committed_distribution":
                    initial = resume["initial"]
                    assert isinstance(initial, dict)
                    accepted_by_cell = initial["accepted_by_cell"]
                    attempted_by_cell = initial["attempted_by_cell"]
                    assert isinstance(accepted_by_cell, dict)
                    assert isinstance(attempted_by_cell, dict)
                    for cell_id, count in zip(cell_ids, (9, 7, 8, 8)):
                        accepted_by_cell[cell_id] = count
                        attempted_by_cell[cell_id] = count
                    initial["committed_batches"] = 1
                    initial["next_attempt_index"] = 32
                    initial["next_batch_id"] = 1
                    allocation = preflight["allocation"]
                    assert isinstance(allocation, dict)
                    quotas = allocation["quotas"]
                    assert isinstance(quotas, list)
                    for quota in quotas:
                        assert isinstance(quota, dict)
                        cell_id = quota["cell_id"]
                        durable = attempted_by_cell[cell_id]
                        quota["durable_attempted"] = durable
                        quota["remaining_attempt_capacity"] -= durable
                else:
                    initial = copy.deepcopy(final_state)
                    committed_batches = initial["committed_batches"]
                    initial["complete"] = False
                    initial["pending_batch_id"] = committed_batches
                    initial["pending_status"] = "proposed"
                    initial["next_batch_id"] = committed_batches + 1
                    resume["initial"] = initial
                    allocation = preflight["allocation"]
                    assert isinstance(allocation, dict)
                    quotas = allocation["quotas"]
                    assert isinstance(quotas, list)
                    attempted_by_cell = initial["attempted_by_cell"]
                    assert isinstance(attempted_by_cell, dict)
                    for quota in quotas:
                        assert isinstance(quota, dict)
                        cell_id = quota["cell_id"]
                        durable = attempted_by_cell[cell_id]
                        quota["durable_attempted"] = durable
                        quota["remaining_attempt_capacity"] = (
                            quota["attempt_ceiling"] - durable
                        )
                preflight["resume_state"] = copy.deepcopy(resume["initial"])
                with self.assertRaises(ValueError):
                    watchdog_module._validate_source_summary(
                        candidate_source,
                        plan=plan,
                    )
                _, source_sha256 = self._write_json(
                    plan.summary_path,
                    candidate_source,
                )
                candidate_summary = copy.deepcopy(summary)
                combined_preflight = candidate_summary["preflight"]
                assert isinstance(combined_preflight, dict)
                chunks = combined_preflight["chunks"]
                assert isinstance(chunks, list) and isinstance(chunks[0], dict)
                chunks[0]["summary_sha256"] = source_sha256
                self._write_json(summary_path, candidate_summary)
                self.assertFalse(
                    final_audited_view_exists(
                        summary_path,
                        repository_root=self.checkout,
                    )
                )

    def test_combined_manifest_rejects_exact_scheduler_violations(self) -> None:
        for mutation in ("oversized", "tiny_extra"):
            with self.subTest(mutation=mutation):
                summary_path, summary = self._canonical_final_fixture()
                plans = watchdog_module._final_chunk_plans(self.checkout)
                plan = plans[0]
                manifest_path = summary_path.parent / (
                    f"{watchdog_module.FINAL_VIEW_NAME}.dataset.json"
                )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                batches = manifest["dataset_batches"]
                shards = manifest["dataset_shards"]
                assert isinstance(batches, list) and isinstance(shards, list)
                if mutation == "oversized":
                    for index, accepted_count in enumerate((33, 33, 30)):
                        batch = batches[index]
                        shard = shards[index]
                        assert isinstance(batch, dict) and isinstance(shard, dict)
                        batch["n_attempted_trajectories"] = accepted_count
                        batch["n_accepted_trajectories"] = accepted_count
                        batch["n_rows"] = accepted_count
                        shard["n_rows"] = accepted_count
                else:
                    first_batch = batches[0]
                    first_shard = shards[0]
                    assert isinstance(first_batch, dict) and isinstance(
                        first_shard, dict
                    )
                    extra_batch = copy.deepcopy(first_batch)
                    extra_shard = copy.deepcopy(first_shard)
                    first_batch["n_attempted_trajectories"] = 16
                    first_batch["n_accepted_trajectories"] = 16
                    first_batch["n_rows"] = 16
                    first_shard["n_rows"] = 16
                    extra_batch["n_attempted_trajectories"] = 16
                    extra_batch["n_accepted_trajectories"] = 16
                    extra_batch["n_rows"] = 16
                    extra_shard["n_rows"] = 16
                    batches.insert(1, extra_batch)
                    shards.insert(1, extra_shard)
                    source = json.loads(plan.summary_path.read_text(encoding="utf-8"))
                    resume = source["resume"]
                    assert isinstance(resume, dict)
                    final_state = resume["final"]
                    assert isinstance(final_state, dict)
                    final_state["committed_batches"] += 1
                    final_state["next_batch_id"] += 1
                    _, source_sha256 = self._write_json(plan.summary_path, source)
                    combined_preflight = summary["preflight"]
                    view = summary["dataset_view"]
                    assert isinstance(combined_preflight, dict) and isinstance(
                        view, dict
                    )
                    chunks = combined_preflight["chunks"]
                    assert isinstance(chunks, list) and isinstance(chunks[0], dict)
                    chunks[0]["committed_batches"] += 1
                    chunks[0]["summary_sha256"] = source_sha256
                    manifest_audit = view["manifest_source_audit"]
                    map_audit = view["trajectory_map_audit"]
                    assert isinstance(manifest_audit, dict) and isinstance(
                        map_audit, dict
                    )
                    manifest_audit["committed_batches"] += 1
                    manifest_audit["committed_shards"] += 1
                    source_binding = map_audit["source_binding"]
                    assert isinstance(source_binding, dict)
                    source_binding["committed_batches"] += 1
                    source_binding["committed_shards"] += 1
                    first_chunk_batch_count = chunks[0]["committed_batches"]
                    for local_batch_id in range(first_chunk_batch_count):
                        batch = batches[local_batch_id]
                        shard = shards[local_batch_id]
                        assert isinstance(batch, dict) and isinstance(shard, dict)
                        batch["batch_id"] = local_batch_id
                        batch["proposal_path"] = (
                            watchdog_module._relative_source_artifact(
                                root=plan.root,
                                kind="proposals",
                                plan=plan,
                                batch_id=local_batch_id,
                                manifest_parent=summary_path.parent,
                            )
                        )
                        batch["result_path"] = (
                            watchdog_module._relative_source_artifact(
                                root=plan.root,
                                kind="results",
                                plan=plan,
                                batch_id=local_batch_id,
                                manifest_parent=summary_path.parent,
                            )
                        )
                        shard["path"] = watchdog_module._relative_source_artifact(
                            root=plan.root,
                            kind="shards",
                            plan=plan,
                            batch_id=local_batch_id,
                            manifest_parent=summary_path.parent,
                        )
                    for global_index, (batch, shard) in enumerate(zip(batches, shards)):
                        assert isinstance(batch, dict) and isinstance(shard, dict)
                        batch["shard_index"] = global_index
                        shard["batch_index"] = global_index
                self._write_manifest_and_rebind_summary(
                    summary_path,
                    summary,
                    manifest,
                )
                self.assertFalse(
                    final_audited_view_exists(
                        summary_path,
                        repository_root=self.checkout,
                    )
                )

    def test_quota_state_rejects_unreported_attempt_limit_failure(self) -> None:
        state = {
            "accepted_by_cell": {"a": 1, "b": 1},
            "attempted_by_cell": {"a": 8, "b": 1},
            "committed_batches": 3,
            "next_attempt_index": 9,
            "next_batch_id": 3,
            "pending_batch_id": None,
            "pending_status": None,
            "complete": False,
            "terminal_failure": None,
            "attempt_limit_failure": None,
            "attempt_limit_exhausted_cells": [],
        }
        with self.assertRaisesRegex(ValueError, "attempt-limit failure"):
            watchdog_module._validate_quota_state(
                state,
                quota_targets={"a": 2, "b": 2},
                maximum_attempts=4,
                first_attempt_index=0,
                batch_size=4,
                context="fixture initial state",
            )

    def test_final_view_rejects_empty_or_extra_audit_records(self) -> None:
        summary_path, record = self._canonical_final_fixture()
        for mutation in ("empty_manifest", "empty_map", "extra_map"):
            with self.subTest(mutation=mutation):
                candidate = copy.deepcopy(record)
                view = candidate["dataset_view"]
                assert isinstance(view, dict)
                if mutation == "empty_manifest":
                    view["manifest_source_audit"] = {}
                elif mutation == "empty_map":
                    view["trajectory_map_audit"] = {}
                else:
                    audit = view["trajectory_map_audit"]
                    assert isinstance(audit, dict)
                    audit["forged"] = True
                self._write_json(summary_path, candidate)
                self.assertFalse(
                    final_audited_view_exists(
                        summary_path, repository_root=self.checkout
                    )
                )

    def test_final_view_rejects_count_hash_path_and_chunk_plan_mutations(self) -> None:
        summary_path, record = self._canonical_final_fixture()
        mutations = (
            ("count", ("dataset_view", "n_trajectories"), 73_729),
            ("hash", ("dataset_view", "manifest", "sha256"), "0" * 64),
            (
                "path",
                ("dataset_view", "manifest", "path"),
                str(
                    summary_path.parent
                    / ".."
                    / summary_path.parent.name
                    / (f"{watchdog_module.FINAL_VIEW_NAME}.dataset.json")
                ),
            ),
            ("chunk_plan", ("preflight", "chunks", 0, "stream_id"), 999),
        )
        for name, coordinates, value in mutations:
            with self.subTest(mutation=name):
                candidate = copy.deepcopy(record)
                parent: object = candidate
                for coordinate in coordinates[:-1]:
                    if isinstance(coordinate, int):
                        assert isinstance(parent, list)
                        parent = parent[coordinate]
                    else:
                        assert isinstance(parent, dict)
                        parent = parent[coordinate]
                final_coordinate = coordinates[-1]
                assert isinstance(parent, dict) and isinstance(final_coordinate, str)
                parent[final_coordinate] = value
                self._write_json(summary_path, candidate)
                self.assertFalse(
                    final_audited_view_exists(
                        summary_path, repository_root=self.checkout
                    )
                )

    def test_final_summary_rejects_boolean_integer_fields_and_huge_timing(self) -> None:
        summary_path, record = self._canonical_final_fixture()
        mutations = (
            (
                "accepted_by_split",
                ("preflight", "accepted_cases_per_family_by_split", "train"),
                True,
            ),
            ("accepted_total", ("preflight", "accepted_cases_total"), True),
            ("attempted_total", ("preflight", "attempted_cases_total"), True),
            (
                "attempted_split",
                ("preflight", "attempted_cases_by_split", "train"),
                True,
            ),
            ("expected_rows", ("preflight", "expected_rows"), True),
            (
                "expected_family_rows",
                ("preflight", "expected_rows_by_split_and_family", "train", "stokes"),
                True,
            ),
            ("chunk_revision", ("preflight", "chunks", 0, "revision_id"), True),
            ("chunk_stream", ("preflight", "chunks", 0, "stream_id"), False),
            ("chunk_before", ("preflight", "chunks", 0, "accepted_before"), False),
            ("chunk_count", ("preflight", "chunks", 0, "accepted_count"), True),
            ("chunk_after", ("preflight", "chunks", 0, "accepted_after"), True),
            ("chunk_attempted", ("preflight", "chunks", 0, "attempted_count"), True),
            ("chunk_batches", ("preflight", "chunks", 0, "committed_batches"), True),
            ("view_schema", ("dataset_view", "schema_version"), True),
            ("view_rows", ("dataset_view", "n_rows"), True),
            ("view_trajectories", ("dataset_view", "n_trajectories"), True),
            (
                "view_accepted_trajectories",
                ("dataset_view", "n_accepted_trajectories"),
                True,
            ),
            ("view_accepted_rows", ("dataset_view", "n_accepted_rows"), True),
            ("manifest_bytes", ("dataset_view", "manifest", "bytes"), True),
            ("map_bytes", ("dataset_view", "trajectory_map", "bytes"), True),
            (
                "manifest_audit_batches",
                ("dataset_view", "manifest_source_audit", "committed_batches"),
                True,
            ),
            (
                "map_audit_schema",
                ("dataset_view", "trajectory_map_audit", "schema_version"),
                True,
            ),
            (
                "map_audit_split_family",
                (
                    "dataset_view",
                    "trajectory_map_audit",
                    "accepted_cases_by_split_and_family",
                    "train",
                    "stokes",
                ),
                True,
            ),
            ("huge_timing", ("timing_seconds", "total"), 10**1000),
            (
                "reversed_timestamp",
                ("invocation_finished_at",),
                "2026-08-08T23:59:59+00:00",
            ),
        )
        for name, coordinates, value in mutations:
            with self.subTest(field=name):
                candidate = copy.deepcopy(record)
                self._set_nested(candidate, coordinates, value)
                self._write_json(summary_path, candidate)
                self.assertFalse(
                    final_audited_view_exists(
                        summary_path,
                        repository_root=self.checkout,
                    )
                )

    def test_combined_manifest_rejects_boolean_integer_fields(self) -> None:
        summary_path, summary = self._canonical_final_fixture()
        manifest_path = summary_path.parent / (
            f"{watchdog_module.FINAL_VIEW_NAME}.dataset.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mutations = (
            ("schema", ("schema_version",), True),
            ("rows", ("n_rows",), True),
            ("trajectories", ("n_trajectories",), True),
            ("accepted_trajectories", ("n_accepted_trajectories",), True),
            ("accepted_rows", ("n_accepted_rows",), True),
            ("split_attempted", ("split_counts", "train", "attempted"), True),
            ("split_accepted", ("split_counts", "train", "accepted"), True),
            ("batch_shard_index", ("dataset_batches", 0, "shard_index"), False),
            ("batch_family", ("dataset_batches", 0, "family_id"), True),
            ("batch_revision", ("dataset_batches", 0, "revision_id"), True),
            ("batch_split", ("dataset_batches", 0, "split_id"), False),
            ("batch_id", ("dataset_batches", 0, "batch_id"), False),
            (
                "batch_attempted",
                ("dataset_batches", 0, "n_attempted_trajectories"),
                True,
            ),
            (
                "batch_accepted",
                ("dataset_batches", 0, "n_accepted_trajectories"),
                True,
            ),
            ("batch_rows", ("dataset_batches", 0, "n_rows"), True),
            ("shard_rows", ("dataset_shards", 0, "n_rows"), True),
            ("shard_batch", ("dataset_shards", 0, "batch_index"), False),
        )
        for name, coordinates, value in mutations:
            with self.subTest(field=name):
                candidate_manifest = copy.deepcopy(manifest)
                candidate_summary = copy.deepcopy(summary)
                self._set_nested(candidate_manifest, coordinates, value)
                self._write_manifest_and_rebind_summary(
                    summary_path,
                    candidate_summary,
                    candidate_manifest,
                )
                self.assertFalse(
                    final_audited_view_exists(
                        summary_path,
                        repository_root=self.checkout,
                    )
                )

    def test_combined_manifest_requires_reconstructed_dataset_contract(self) -> None:
        summary_path, summary = self._canonical_final_fixture()
        manifest_path = summary_path.parent / (
            f"{watchdog_module.FINAL_VIEW_NAME}.dataset.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mutations = (
            ("target", ("dataset_contract", "target", "nx"), 2048),
            (
                "numerical",
                (
                    "dataset_contract",
                    "trajectory_numerical_by_family_revision",
                    0,
                    "numerical",
                    "dno_order",
                ),
                5,
            ),
            (
                "stored_dtype",
                ("dataset_contract", "stored_dtypes", "eta"),
                "float64",
            ),
            (
                "generation_identity",
                (
                    "dataset_contract",
                    "generation_identity",
                    "dependency_environment_fingerprint",
                ),
                "f" * 64,
            ),
        )
        for name, coordinates, value in mutations:
            with self.subTest(field=name):
                candidate_manifest = copy.deepcopy(manifest)
                candidate_summary = copy.deepcopy(summary)
                self._set_nested(candidate_manifest, coordinates, value)
                dataset_contract = candidate_manifest["dataset_contract"]
                assert isinstance(dataset_contract, dict)
                contract_fingerprint = watchdog_module._canonical_json_sha256(
                    dataset_contract
                )
                candidate_manifest["dataset_contract_fingerprint"] = (
                    contract_fingerprint
                )
                view = candidate_summary["dataset_view"]
                assert isinstance(view, dict)
                view["dataset_contract_fingerprint"] = contract_fingerprint
                self._write_manifest_and_rebind_summary(
                    summary_path,
                    candidate_summary,
                    candidate_manifest,
                )
                self.assertFalse(
                    final_audited_view_exists(
                        summary_path,
                        repository_root=self.checkout,
                    )
                )

    def test_final_view_rejects_missing_symlink_and_noncanonical_artifacts(
        self,
    ) -> None:
        for mutation in ("missing", "symlink", "noncanonical_summary"):
            with self.subTest(mutation=mutation):
                summary_path, _ = self._canonical_final_fixture()
                manifest_path = summary_path.parent / (
                    f"{watchdog_module.FINAL_VIEW_NAME}.dataset.json"
                )
                if mutation == "missing":
                    manifest_path.unlink()
                    candidate_path = summary_path
                elif mutation == "symlink":
                    manifest_bytes = manifest_path.read_bytes()
                    manifest_path.unlink()
                    target = self.checkout / "symlink-target.dataset.json"
                    target.write_bytes(manifest_bytes)
                    manifest_path.symlink_to(target)
                    candidate_path = summary_path
                else:
                    candidate_path = (
                        summary_path.parent
                        / ".."
                        / summary_path.parent.name
                        / summary_path.name
                    )
                self.assertFalse(
                    final_audited_view_exists(
                        candidate_path, repository_root=self.checkout
                    )
                )

    def test_final_view_rejects_physically_rehashed_manifest_or_map_mutation(
        self,
    ) -> None:
        for artifact_name in ("dataset.json", "trajectory_map.npz"):
            with self.subTest(artifact=artifact_name):
                summary_path, _ = self._canonical_final_fixture()
                artifact = summary_path.parent / (
                    f"{watchdog_module.FINAL_VIEW_NAME}.{artifact_name}"
                )
                artifact.write_bytes(artifact.read_bytes() + b"forged\n")
                self.assertFalse(
                    final_audited_view_exists(
                        summary_path, repository_root=self.checkout
                    )
                )

    def test_final_view_rejects_rehashed_noncanonical_trajectory_maps(self) -> None:
        for mutation in (
            "plain_text",
            "missing_member",
            "wrong_dtype",
            "wrong_shape",
            "wrong_schema",
            "bad_crc",
            "bad_bzip2",
            "bad_lzma",
            "unsupported_compression",
        ):
            with self.subTest(mutation=mutation):
                summary_path, summary = self._canonical_final_fixture()
                preflight = summary["preflight"]
                assert isinstance(preflight, dict)
                attempted_total = preflight["attempted_cases_total"]
                assert isinstance(attempted_total, int)
                map_path = summary_path.parent / (
                    f"{watchdog_module.FINAL_VIEW_NAME}.trajectory_map.npz"
                )
                if mutation == "plain_text":
                    forged = b"not an NPZ\n"
                elif mutation in {"bad_bzip2", "bad_crc", "bad_lzma"}:
                    compression_mutation = mutation if mutation != "bad_crc" else None
                    forged = _corrupt_zip_member_payload(
                        _canonical_trajectory_map_bytes(
                            attempted_total,
                            compression_mutation,
                        ),
                        "trajectory_index.npy",
                    )
                elif mutation == "unsupported_compression":
                    forged = _replace_zip_member_compression(
                        _canonical_trajectory_map_bytes(attempted_total),
                        "trajectory_index.npy",
                        99,
                    )
                else:
                    forged = _canonical_trajectory_map_bytes(
                        attempted_total,
                        mutation,
                    )
                map_path.write_bytes(forged)
                map_sha256 = hashlib.sha256(forged).hexdigest()
                view = summary["dataset_view"]
                assert isinstance(view, dict)
                map_record = view["trajectory_map"]
                assert isinstance(map_record, dict)
                map_record["bytes"] = len(forged)
                map_record["sha256"] = map_sha256
                manifest_path = summary_path.parent / (
                    f"{watchdog_module.FINAL_VIEW_NAME}.dataset.json"
                )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["trajectory_map_sha256"] = map_sha256
                self._write_manifest_and_rebind_summary(
                    summary_path,
                    summary,
                    manifest,
                )
                self.assertFalse(
                    final_audited_view_exists(
                        summary_path,
                        repository_root=self.checkout,
                    )
                )

    def test_final_view_rejects_rehashed_forged_manifest_records(self) -> None:
        summary_path, summary = self._canonical_final_fixture()
        manifest_path = summary_path.parent / (
            f"{watchdog_module.FINAL_VIEW_NAME}.dataset.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["dataset_batches"][0]["n_rows"] += 1
        manifest_bytes, manifest_sha256 = self._write_json(manifest_path, manifest)
        view = summary["dataset_view"]
        assert isinstance(view, dict)
        manifest_record = view["manifest"]
        assert isinstance(manifest_record, dict)
        manifest_record["bytes"] = manifest_bytes
        manifest_record["sha256"] = manifest_sha256
        self._write_json(summary_path, summary)

        self.assertFalse(
            final_audited_view_exists(
                summary_path,
                repository_root=self.checkout,
            )
        )

    def test_final_view_terminal_sweep_rejects_cross_file_mutations(self) -> None:
        for mutation in (
            "summary_after_read",
            "source_zero_after_source_one",
            "manifest_after_map",
            "map_after_read",
        ):
            with self.subTest(mutation=mutation):
                summary_path, _ = self._canonical_final_fixture()
                plans = watchdog_module._final_chunk_plans(self.checkout)
                manifest_path = summary_path.parent / (
                    f"{watchdog_module.FINAL_VIEW_NAME}.dataset.json"
                )
                map_path = summary_path.parent / (
                    f"{watchdog_module.FINAL_VIEW_NAME}.trajectory_map.npz"
                )
                trigger, target = {
                    "summary_after_read": (1, summary_path),
                    "source_zero_after_source_one": (3, plans[0].summary_path),
                    "manifest_after_map": (29, manifest_path),
                    "map_after_read": (29, map_path),
                }[mutation]
                original_reader = watchdog_module._read_regular_file
                calls = 0
                mutated = False

                def mutate_between_files(
                    path: Path,
                    *,
                    capture: bool,
                ) -> watchdog_module.AuthenticatedFile:
                    nonlocal calls, mutated
                    observed = original_reader(path, capture=capture)
                    calls += 1
                    if calls == trigger:
                        target.write_bytes(target.read_bytes() + b" \n")
                        mutated = True
                    return observed

                with patch.object(
                    watchdog_module,
                    "_read_regular_file",
                    side_effect=mutate_between_files,
                ):
                    self.assertFalse(
                        final_audited_view_exists(
                            summary_path,
                            repository_root=self.checkout,
                        )
                    )
                self.assertTrue(mutated)
                self.assertGreater(calls, 29)


if __name__ == "__main__":
    unittest.main()
