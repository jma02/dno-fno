"""Structural and source-binding tests for combined paper-dataset views."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import generate_paper_dataset_jonswap as bucketed
from scripts.build_paper_dataset_view import (
    CombinedDatasetPlan,
    CompletedChunk,
    FAMILY_IDS,
    FAMILY_ORDER,
    PAPER_GENERATION_COMPATIBILITY_POLICY,
    ROWS_PER_ACCEPTED_CASE,
    SHARED_TARGET_SOURCE_PATHS,
    TANAKA_REVISION3_AUDITED_REPLAY_EXECUTION_FINGERPRINT,
    TANAKA_REVISION3_AUDITED_REPLAY_SOURCE_FINGERPRINT,
    TANAKA_REVISION3_CANONICAL_EXECUTION_FINGERPRINT,
    TANAKA_REVISION3_LEGACY_EXECUTION_FINGERPRINT,
    TANAKA_REVISION3_LEGACY_SOURCE_FINGERPRINT,
    _configuration_record,
    _audit_trajectory_map_against_plan,
    _ordered_cell_ids,
    _required_sha256,
    _resolve_generation_compatibility,
    _validate_committed_proposal_contract,
    _validate_current_execution_contract,
    _validate_view,
    load_completed_chunk,
    validate_combined_plan,
)
from scripts.generate_paper_dataset import (
    GenerationRequest,
    build_run_spec,
    source_hashes,
)
from solver.gen_data.pipeline.production import (
    SplitId,
    paper_dataset_revision_id,
    split_code,
)
from solver.gen_data.pipeline.archive import BatchPaths, file_sha256
from solver.gen_data.pipeline.manifest import (
    DATASET_VIEW_SCHEMA_VERSION,
    TRAJECTORY_MAP_SCHEMA_VERSION,
    DatasetViewPaths,
)
from solver.gen_data.pipeline.valid_case_generation import canonical_json_sha256


def _chunk(
    family: str,
    *,
    before: int,
    count: int,
    stream_id: int,
    suffix: str,
    split: SplitId = SplitId.TRAIN,
    revision_id: int | None = None,
    source_digest: str | None = None,
    dependency_fingerprint: str = "d" * 64,
    execution_platform: str = "cpu",
    execution_fingerprint: str | None = None,
    generation_compatibility_id: str | None = None,
    shared_source_digest: str = "a" * 64,
) -> CompletedChunk:
    root = Path(f"/tmp/{split.value}_{family}_{suffix}")
    identity = f"{split.value}:{family}:{before}:{count}:{stream_id}:{suffix}"
    family_index = FAMILY_ORDER.index(family) + 1
    selected_revision = (
        paper_dataset_revision_id(FAMILY_IDS[family])
        if revision_id is None
        else revision_id
    )
    selected_source_digest = source_digest or f"{family_index}" * 64
    selected_execution_fingerprint = execution_fingerprint or "f" * 64
    return CompletedChunk(
        summary_path=root / f"{family}.summary.json",
        summary_sha256=hashlib.sha256(f"summary:{identity}".encode()).hexdigest(),
        root=root,
        family=family,
        revision_id=selected_revision,
        split=split,
        stream_id=stream_id,
        accepted_before=before,
        accepted_count=count,
        accepted_after=before + count,
        attempted_count=count + 1,
        fingerprint=hashlib.sha256(identity.encode()).hexdigest(),
        dependency_fingerprint=dependency_fingerprint,
        execution_fingerprint=selected_execution_fingerprint,
        generation_compatibility_id=generation_compatibility_id,
        source_fingerprint=selected_source_digest,
        source_sha256={
            "shared.py": selected_source_digest,
            **{path: shared_source_digest for path in SHARED_TARGET_SOURCE_PATHS},
        },
        execution_platform=execution_platform,
        batches=(),
    )


def _write_source_batch(
    chunk: CompletedChunk,
    *,
    case_id_start: int,
) -> BatchPaths:
    paths = BatchPaths.under(
        chunk.root,
        family=chunk.family,
        split=chunk.split.value,
        batch_id=0,
    )
    for parent in (paths.proposal.parent, paths.shard.parent, paths.result.parent):
        parent.mkdir(parents=True, exist_ok=True)
    case_count = chunk.attempted_count
    case_ids = np.arange(
        case_id_start,
        case_id_start + case_count,
        dtype=np.int64,
    )
    cell_ids = np.arange(case_count, dtype=np.int32)
    np.savez_compressed(
        paths.proposal,
        family_id=np.asarray(int(FAMILY_IDS[chunk.family]), dtype=np.int16),
        revision_id=np.asarray(chunk.revision_id, dtype=np.int16),
        split_id=np.asarray(split_code(chunk.split), dtype=np.uint8),
        batch_id=np.asarray(0, dtype=np.int64),
        case_id=case_ids,
        cell_id=cell_ids,
        config_fingerprint=np.asarray(chunk.fingerprint),
    )
    rows_per_case = ROWS_PER_ACCEPTED_CASE[chunk.family]
    accepted_local = np.arange(chunk.accepted_count, dtype=np.int32)
    case_local_index = np.repeat(accepted_local, rows_per_case)
    frame_index = np.tile(
        np.arange(rows_per_case, dtype=np.int32),
        chunk.accepted_count,
    )
    np.savez_compressed(
        paths.shard,
        case_local_index=case_local_index,
        frame_index=frame_index,
    )
    cases = [
        {
            "case_id": int(case_id),
            "accepted": local_index < chunk.accepted_count,
            "required_bits": 1,
            "evaluated_bits": 1,
            "failed_bits": int(local_index >= chunk.accepted_count),
            "first_row": (
                local_index * rows_per_case
                if local_index < chunk.accepted_count
                else -1
            ),
            "row_count": (rows_per_case if local_index < chunk.accepted_count else 0),
        }
        for local_index, case_id in enumerate(case_ids)
    ]
    paths.result.write_text(
        json.dumps(
            {
                "schema": "paper_dataset_batch_result_v1",
                "config_fingerprint": chunk.fingerprint,
                "proposal_sha256": file_sha256(paths.proposal),
                "shard_sha256": file_sha256(paths.shard),
                "cases": cases,
            },
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return paths


def _trajectory_map_arrays(plan: CombinedDatasetPlan) -> dict[str, np.ndarray]:
    family_ids: list[int] = []
    revision_ids: list[int] = []
    split_ids: list[int] = []
    case_ids: list[int] = []
    cell_ids: list[int] = []
    accepted_values: list[bool] = []
    required_bits: list[int] = []
    evaluated_bits: list[int] = []
    failed_bits: list[int] = []
    row_counts: list[int] = []
    row_trajectory: list[np.ndarray] = []
    row_frames: list[np.ndarray] = []
    row_shards: list[np.ndarray] = []
    row_in_shards: list[np.ndarray] = []
    trajectory_offset = 0
    shard_index = 0
    for chunk in plan.chunks:
        for paths in chunk.batches:
            with np.load(paths.proposal, allow_pickle=False) as proposal:
                proposed_case_ids = np.asarray(proposal["case_id"], dtype=np.int64)
                proposed_cell_ids = np.asarray(proposal["cell_id"], dtype=np.int32)
            result = json.loads(paths.result.read_text(encoding="utf-8"))
            cases = result["cases"]
            count = len(cases)
            family_ids.extend([int(FAMILY_IDS[chunk.family])] * count)
            revision_ids.extend([chunk.revision_id] * count)
            split_ids.extend([split_code(chunk.split)] * count)
            case_ids.extend(map(int, proposed_case_ids))
            cell_ids.extend(map(int, proposed_cell_ids))
            accepted_values.extend(case["accepted"] for case in cases)
            required_bits.extend(case["required_bits"] for case in cases)
            evaluated_bits.extend(case["evaluated_bits"] for case in cases)
            failed_bits.extend(case["failed_bits"] for case in cases)
            row_counts.extend(case["row_count"] for case in cases)
            with np.load(paths.shard, allow_pickle=False) as shard:
                local_index = np.asarray(shard["case_local_index"], dtype=np.int32)
                frame_index = np.asarray(shard["frame_index"], dtype=np.int32)
            row_trajectory.append(local_index + trajectory_offset)
            row_frames.append(frame_index)
            row_shards.append(np.full(local_index.size, shard_index, dtype=np.int32))
            row_in_shards.append(np.arange(local_index.size, dtype=np.int64))
            trajectory_offset += count
            shard_index += 1
    row_count_array = np.asarray(row_counts, dtype=np.int32)
    accepted = np.asarray(accepted_values, dtype=np.bool_)
    first_rows = np.cumsum(row_count_array, dtype=np.int64) - row_count_array
    first_rows[~accepted] = -1
    return {
        "schema_version": np.asarray(
            TRAJECTORY_MAP_SCHEMA_VERSION,
            dtype=np.int16,
        ),
        "trajectory_index": np.concatenate(row_trajectory),
        "frame_index": np.concatenate(row_frames),
        "shard_index": np.concatenate(row_shards),
        "shard_row": np.concatenate(row_in_shards),
        "trajectory_family_id": np.asarray(family_ids, dtype=np.int16),
        "trajectory_revision_id": np.asarray(revision_ids, dtype=np.int16),
        "trajectory_split_id": np.asarray(split_ids, dtype=np.uint8),
        "trajectory_case_id": np.asarray(case_ids, dtype=np.int64),
        "trajectory_cell_id": np.asarray(cell_ids, dtype=np.int32),
        "trajectory_accepted": accepted,
        "trajectory_required_bits": np.asarray(required_bits, dtype=np.uint32),
        "trajectory_evaluated_bits": np.asarray(evaluated_bits, dtype=np.uint32),
        "trajectory_failed_bits": np.asarray(failed_bits, dtype=np.uint32),
        "trajectory_first_row": first_rows,
        "trajectory_row_count": row_count_array,
    }


def _all_split_plan(root: Path) -> CombinedDatasetPlan:
    chunks: list[CompletedChunk] = []
    for case_index, (split, count, stream_id, family) in enumerate(
        (
            (split, count, stream_id, family)
            for split, count, stream_id in (
                (SplitId.TRAIN, 2, 0),
                (SplitId.VALIDATION, 1, 100),
                (SplitId.TEST, 1, 200),
            )
            for family in FAMILY_ORDER
        )
    ):
        chunk = _chunk(
            family,
            before=0,
            count=count,
            stream_id=stream_id,
            suffix=f"map_{split.value}_{family}",
            split=split,
        )
        chunk_root = root / split.value / family
        chunk = replace(
            chunk,
            root=chunk_root,
            summary_path=chunk_root / f"{family}.summary.json",
        )
        paths = _write_source_batch(
            chunk,
            case_id_start=10_000 * (case_index + 1),
        )
        chunks.append(replace(chunk, batches=(paths,)))
    return validate_combined_plan(chunks)


def _write_valid_combined_view(
    root: Path,
    *,
    plan: CombinedDatasetPlan,
) -> DatasetViewPaths:
    view = DatasetViewPaths(
        manifest=root / "combined.dataset.json",
        trajectory_map=root / "combined.trajectory_map.npz",
    )
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(view.trajectory_map, **_trajectory_map_arrays(plan))
    batch_records: list[dict[str, object]] = []
    shard_records: list[dict[str, object]] = []
    for batch_index, (chunk, paths) in enumerate(
        (chunk, paths) for chunk in plan.chunks for paths in chunk.batches
    ):
        with np.load(paths.proposal, allow_pickle=False) as proposal:
            family_id = int(np.asarray(proposal["family_id"]).item())
            revision_id = int(np.asarray(proposal["revision_id"]).item())
            split_id = int(np.asarray(proposal["split_id"]).item())
            batch_id = int(np.asarray(proposal["batch_id"]).item())
            attempted = int(np.asarray(proposal["case_id"]).size)
            fingerprint = str(np.asarray(proposal["config_fingerprint"]).item())
        result = json.loads(paths.result.read_text(encoding="utf-8"))
        with np.load(paths.shard, allow_pickle=False) as shard:
            row_count = int(np.asarray(shard["case_local_index"]).size)
        shard_index = len(shard_records)
        shard_records.append(
            {
                "path": os.path.relpath(paths.shard.resolve(), start=root.resolve()),
                "sha256": result["shard_sha256"],
                "n_rows": row_count,
                "batch_index": batch_index,
                "configuration_fingerprint": fingerprint,
            }
        )
        batch_records.append(
            {
                "proposal_path": os.path.relpath(
                    paths.proposal.resolve(),
                    start=root.resolve(),
                ),
                "proposal_sha256": file_sha256(paths.proposal),
                "result_path": os.path.relpath(
                    paths.result.resolve(),
                    start=root.resolve(),
                ),
                "result_sha256": file_sha256(paths.result),
                "shard_index": shard_index,
                "configuration_fingerprint": fingerprint,
                "family_id": family_id,
                "revision_id": revision_id,
                "split_id": split_id,
                "batch_id": batch_id,
                "n_attempted_trajectories": attempted,
                "n_accepted_trajectories": sum(
                    int(case["accepted"]) for case in result["cases"]
                ),
                "n_rows": row_count,
            }
        )
    dataset_contract = {"test_contract": "source_bound"}
    manifest = {
        "schema_version": DATASET_VIEW_SCHEMA_VERSION,
        "configuration_fingerprint": None,
        "configuration_fingerprints": list(plan.fingerprints),
        "dataset_contract": dataset_contract,
        "dataset_contract_fingerprint": canonical_json_sha256(dataset_contract),
        "dataset_batches": batch_records,
        "dataset_shards": shard_records,
        "trajectory_map_npz": view.trajectory_map.name,
        "trajectory_map_sha256": file_sha256(view.trajectory_map),
        "requires_trajectory_map": True,
        "n_rows": plan.expected_rows,
        "n_trajectories": plan.attempted_cases,
        "n_accepted_trajectories": plan.accepted_cases,
        "n_accepted_rows": plan.expected_rows,
        "grid": {"length": 2.0 * np.pi, "nx": 1024},
        "split_counts": {
            split.value: {
                "attempted": plan.attempted_cases_by_split.get(split.value, 0),
                "accepted": 4
                * plan.accepted_cases_per_family_by_split.get(split.value, 0),
            }
            for split in SplitId
        },
    }
    view.manifest.write_text(
        json.dumps(manifest, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return view


class CombinedPaperDatasetViewTests(unittest.TestCase):
    def test_tanaka_revision3_bridge_is_the_exact_audited_pair(self) -> None:
        observed = {
            (
                variant.execution_record_fingerprint,
                variant.source_sha256_fingerprint,
                variant.compatibility_id,
            )
            for variant in PAPER_GENERATION_COMPATIBILITY_POLICY.variants
        }
        self.assertEqual(
            observed,
            {
                (
                    TANAKA_REVISION3_LEGACY_EXECUTION_FINGERPRINT,
                    TANAKA_REVISION3_LEGACY_SOURCE_FINGERPRINT,
                    TANAKA_REVISION3_CANONICAL_EXECUTION_FINGERPRINT,
                ),
                (
                    TANAKA_REVISION3_AUDITED_REPLAY_EXECUTION_FINGERPRINT,
                    TANAKA_REVISION3_AUDITED_REPLAY_SOURCE_FINGERPRINT,
                    TANAKA_REVISION3_CANONICAL_EXECUTION_FINGERPRINT,
                ),
            },
        )

    def test_real_retained_tanaka_identity_resolves_only_via_bridge(self) -> None:
        summaries = tuple(
            Path("outputs/paper_dataset_revision3").glob(
                "**/paper_dataset_tanaka_*.summary.json"
            )
        )
        if not summaries:
            self.skipTest("retained Tanaka revision-3 summaries are unavailable")

        with self.assertRaisesRegex(
            ValueError,
            "exact current paper execution contract",
        ):
            load_completed_chunk(summaries[0])
        loaded = load_completed_chunk(
            summaries[0],
            generation_compatibility_policy=(PAPER_GENERATION_COMPATIBILITY_POLICY),
        )
        self.assertEqual(
            loaded.generation_compatibility_id,
            TANAKA_REVISION3_CANONICAL_EXECUTION_FINGERPRINT,
        )

        for path in summaries:
            with self.subTest(path=path):
                summary = json.loads(path.read_text(encoding="utf-8"))
                run_spec = summary["run_spec"]
                configuration = run_spec["configuration"]
                execution = configuration["trajectory_execution"]
                sources = configuration["source_sha256"]
                self.assertEqual(
                    canonical_json_sha256(execution),
                    TANAKA_REVISION3_LEGACY_EXECUTION_FINGERPRINT,
                )
                self.assertEqual(
                    canonical_json_sha256(sources),
                    TANAKA_REVISION3_LEGACY_SOURCE_FINGERPRINT,
                )
                resolution = PAPER_GENERATION_COMPATIBILITY_POLICY.resolve(
                    family_id=run_spec["family_id"],
                    revision_id=run_spec["revision_id"],
                    execution_record=execution,
                    source_sha256=sources,
                )
                assert resolution is not None
                self.assertEqual(
                    resolution.compatibility_id,
                    TANAKA_REVISION3_CANONICAL_EXECUTION_FINGERPRINT,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "exact current paper execution contract",
                ):
                    _validate_current_execution_contract(
                        configuration,
                        family="tanaka",
                    )

        changed_sources = dict(sources)
        changed_sources[next(iter(changed_sources))] = "0" * 64
        with self.assertRaisesRegex(
            ValueError,
            "not an explicitly audited compatibility variant",
        ):
            PAPER_GENERATION_COMPATIBILITY_POLICY.resolve(
                family_id=run_spec["family_id"],
                revision_id=run_spec["revision_id"],
                execution_record=execution,
                source_sha256=changed_sources,
            )

    def test_source_digest_parser_requires_lowercase_sha256(self) -> None:
        self.assertEqual(
            _required_sha256({"source": "a" * 64}, "source"),
            "a" * 64,
        )
        for invalid in ("a" * 63, "A" * 64, "z" * 64):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
                    _required_sha256({"source": invalid}, "source")

    def test_current_execution_contract_is_exact_and_fail_closed(self) -> None:
        for family in FAMILY_ORDER:
            with self.subTest(family=family):
                request = GenerationRequest(
                    output_root=Path(f"/tmp/current_contract_{family}"),
                    family=family,  # type: ignore[arg-type]
                    split=SplitId.TEST,
                    accepted_cases=1,
                    batch_size=1,
                    platform="cpu",
                )
                configuration = dict(build_run_spec(request).configuration)
                _validate_current_execution_contract(
                    configuration,
                    family=family,
                )
                key = "contract" if family == "stokes" else "trajectory_execution"
                changed = dict(configuration)
                execution = dict(configuration[key])  # type: ignore[arg-type]
                execution["role"] = "reduced_wiring_evidence_only"
                changed[key] = execution
                with self.assertRaisesRegex(
                    ValueError,
                    "exact current paper execution contract",
                ):
                    _validate_current_execution_contract(changed, family=family)

    def test_current_tanaka_source_does_not_require_legacy_bridge(self) -> None:
        request = GenerationRequest(
            output_root=Path("/tmp/current_tanaka_new_source"),
            family="tanaka",
            split=SplitId.TEST,
            accepted_cases=1,
            batch_size=1,
            platform="cpu",
        )
        spec = build_run_spec(request)
        configuration = dict(spec.configuration)
        execution = configuration["trajectory_execution"]
        sources = source_hashes("tanaka")

        self.assertEqual(dict(configuration["source_sha256"]), sources)
        self.assertNotEqual(
            canonical_json_sha256(sources),
            TANAKA_REVISION3_AUDITED_REPLAY_SOURCE_FINGERPRINT,
        )

        resolution = _resolve_generation_compatibility(
            configuration,
            family="tanaka",
            family_id=int(FAMILY_IDS["tanaka"]),
            revision_id=paper_dataset_revision_id(FAMILY_IDS["tanaka"]),
            execution_record=execution,  # type: ignore[arg-type]
            source_sha256=sources,
            policy=PAPER_GENERATION_COMPATIBILITY_POLICY,
        )

        self.assertIsNone(resolution)
        with self.assertRaisesRegex(
            ValueError,
            "not an explicitly audited compatibility variant",
        ):
            PAPER_GENERATION_COMPATIBILITY_POLICY.resolve(
                family_id=int(FAMILY_IDS["tanaka"]),
                revision_id=paper_dataset_revision_id(FAMILY_IDS["tanaka"]),
                execution_record=execution,  # type: ignore[arg-type]
                source_sha256=sources,
            )

    def test_current_jonswap_contract_requires_adjustment_policy(self) -> None:
        request = GenerationRequest(
            output_root=Path("/tmp/current_contract_jonswap_missing_adjustment"),
            family="jonswap_tma",
            split=SplitId.TEST,
            accepted_cases=1,
            batch_size=1,
            platform="cpu",
        )
        configuration = dict(build_run_spec(request).configuration)
        execution = dict(configuration["trajectory_execution"])  # type: ignore[arg-type]
        execution.pop("jonswap_adjustment")
        configuration["trajectory_execution"] = execution

        with self.assertRaisesRegex(
            ValueError,
            "exact current paper execution contract",
        ):
            _validate_current_execution_contract(
                configuration,
                family="jonswap_tma",
            )

    def test_current_execution_contract_is_json_type_strict(self) -> None:
        request = GenerationRequest(
            output_root=Path("/tmp/current_contract_bf_numeric_type"),
            family="benjamin_feir",
            split=SplitId.TEST,
            accepted_cases=1,
            batch_size=1,
            platform="cpu",
        )
        configuration = build_run_spec(request).to_json_record()["configuration"]
        assert isinstance(configuration, dict)
        numerical = configuration["trajectory_execution"]["numerical"]
        self.assertEqual(numerical["gravity"], 1.0)
        numerical["gravity"] = 1

        with self.assertRaisesRegex(
            ValueError,
            "exact current paper execution contract",
        ):
            _validate_current_execution_contract(
                configuration,
                family="benjamin_feir",
            )

    def test_committed_jonswap_proposal_is_bound_to_run_and_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = GenerationRequest(
                output_root=root,
                family="jonswap_tma",
                split=SplitId.TEST,
                accepted_cases=1,
                batch_size=1,
                platform="cpu",
            )
            spec = bucketed.build_bucketed_run_spec(
                request,
                solver_batch_size=1,
            )
            run_record = spec.to_json_record()
            configuration = run_record["configuration"]
            assert isinstance(configuration, dict)
            config = configuration[bucketed.BUCKETING_CONFIG_KEY]
            metadata = {
                "family": "jonswap_tma",
                "case_kind": "trajectory",
                "trajectory_execution": configuration["trajectory_execution"],
                "additional_metadata": {
                    "run_spec": run_record,
                    bucketed.BUCKETING_CONFIG_KEY: config,
                },
            }
            paths = BatchPaths.under(
                root,
                family="jonswap_tma",
                split="test",
                batch_id=0,
            )
            paths.proposal.parent.mkdir(parents=True)

            def write_proposal(record: dict[str, object]) -> None:
                np.savez_compressed(
                    paths.proposal,
                    metadata_json=np.asarray(
                        json.dumps(record, sort_keys=True, allow_nan=False)
                    ),
                )

            write_proposal(metadata)
            _validate_committed_proposal_contract(paths, spec=spec)

            changed_type = json.loads(json.dumps(metadata))
            changed_type["trajectory_execution"]["numerical"]["gravity"] = 1
            write_proposal(changed_type)
            with self.assertRaisesRegex(ValueError, "differs from its own run spec"):
                _validate_committed_proposal_contract(paths, spec=spec)

            changed_run = json.loads(json.dumps(run_record))
            changed_run["configuration"][  # type: ignore[index]
                "jonswap_horizon_bucketing"
            ]["outer_proposal_size"] = 2  # type: ignore[index]
            changed_configuration = changed_run["configuration"]
            assert isinstance(changed_configuration, dict)
            changed_spec = replace(
                spec,
                configuration=changed_configuration,
            )
            changed_metadata = json.loads(json.dumps(metadata))
            changed_metadata["additional_metadata"][  # type: ignore[index]
                "run_spec"
            ] = changed_spec.to_json_record()
            changed_metadata["additional_metadata"][  # type: ignore[index]
                "jonswap_horizon_bucketing"
            ] = changed_configuration["jonswap_horizon_bucketing"]
            write_proposal(changed_metadata)
            with self.assertRaisesRegex(
                ValueError,
                "exact current bucketing config",
            ):
                _validate_committed_proposal_contract(
                    paths,
                    spec=changed_spec,
                )

            changed = json.loads(json.dumps(metadata))
            changed["additional_metadata"][  # type: ignore[index]
                "jonswap_horizon_bucketing"
            ]["nonlinear_adjustment"]["ramp_order"] = 2  # type: ignore[index]
            write_proposal(changed)
            with self.assertRaisesRegex(ValueError, "differs from its run spec"):
                _validate_committed_proposal_contract(paths, spec=spec)

    def test_completed_chunk_accepts_frozen_ordered_cell_sequence(self) -> None:
        request = GenerationRequest(
            output_root=Path("/tmp/frozen_ordered_cells"),
            family="stokes",
            split=SplitId.VALIDATION,
            accepted_cases=1,
            batch_size=1,
            accepted_cases_before=0,
            platform="cpu",
            stream_id=0,
            first_attempt_index=0,
        )
        spec = build_run_spec(request)

        self.assertIsInstance(
            spec.configuration["ordered_cell_ids"],
            tuple,
        )
        self.assertEqual(
            _ordered_cell_ids(spec.configuration),
            (
                "finite_low",
                "finite_moderate",
                "deep_low",
                "deep_moderate",
            ),
        )
        configuration = _configuration_record(spec)
        self.assertIsInstance(configuration["ordered_cell_ids"], list)
        self.assertEqual(
            len(canonical_json_sha256(configuration)),
            64,
        )

    def test_equal_nested_chunks_have_expected_case_and_row_counts(self) -> None:
        chunks = tuple(
            chunk
            for family in FAMILY_ORDER
            for chunk in (
                _chunk(
                    family,
                    before=0,
                    count=2_048,
                    stream_id=0,
                    suffix="a",
                ),
                _chunk(
                    family,
                    before=2_048,
                    count=2_048,
                    stream_id=1,
                    suffix="b",
                ),
            )
        )
        plan = validate_combined_plan(chunks)

        self.assertEqual(
            plan.accepted_cases_per_family_by_split,
            {"train": 4_096},
        )
        self.assertEqual(plan.accepted_cases, 4 * 4_096)
        self.assertEqual(plan.expected_rows, 417 * 4_096)
        self.assertEqual(plan.attempted_cases, 4 * (2_049 + 2_049))
        self.assertEqual(
            tuple(chunk.family for chunk in plan.chunks[:2]),
            ("stokes", "stokes"),
        )
        self.assertEqual(
            {chunk.family: chunk.revision_id for chunk in plan.chunks},
            {
                family: paper_dataset_revision_id(FAMILY_IDS[family])
                for family in FAMILY_ORDER
            },
        )
        self.assertEqual(
            len({chunk.source_sha256["shared.py"] for chunk in plan.chunks}),
            len(FAMILY_ORDER),
        )

    def test_final_release_accepts_rebalanced_jonswap_shards(self) -> None:
        standard_intervals = (
            (0, 2_048, 0),
            (2_048, 2_048, 1),
            (4_096, 4_096, 2),
            (8_192, 8_192, 3),
        )
        jonswap_intervals = (
            (0, 2_048, 0),
            (2_048, 2_048, 1),
            (4_096, 4_096, 2),
            (8_192, 4_096, 3),
            (12_288, 2_048, 4),
            (14_336, 2_048, 5),
        )
        chunks = [
            _chunk(
                family,
                before=before,
                count=count,
                stream_id=stream_id,
                suffix=f"release_train_{family}_{stream_id}",
            )
            for family in FAMILY_ORDER
            for before, count, stream_id in (
                jonswap_intervals if family == "jonswap_tma" else standard_intervals
            )
        ]
        chunks.extend(
            _chunk(
                family,
                before=0,
                count=1_024,
                stream_id=stream_id,
                suffix=f"release_{split.value}_{family}",
                split=split,
            )
            for split, stream_id in (
                (SplitId.VALIDATION, 100),
                (SplitId.TEST, 200),
            )
            for family in FAMILY_ORDER
        )

        plan = validate_combined_plan(chunks)

        self.assertEqual(len(plan.chunks), 26)
        self.assertEqual(
            plan.accepted_cases_per_family_by_split,
            {"train": 16_384, "validation": 1_024, "test": 1_024},
        )
        self.assertEqual(plan.accepted_cases, 73_728)
        self.assertEqual(plan.expected_rows, 7_686_144)
        self.assertEqual(
            tuple(
                (
                    chunk.accepted_before,
                    chunk.accepted_count,
                    chunk.stream_id,
                )
                for chunk in plan.chunks
                if chunk.family == "jonswap_tma" and chunk.split is SplitId.TRAIN
            ),
            jonswap_intervals,
        )

    def test_postbuild_map_audit_recovers_every_split_family_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _all_split_plan(root / "sources")
            path = root / "combined.trajectory_map.npz"
            np.savez_compressed(path, **_trajectory_map_arrays(plan))

            audit = _audit_trajectory_map_against_plan(path, plan=plan)

        self.assertEqual(
            audit["attempted_cases_by_split_and_family"],
            {
                "train": {family: 3 for family in FAMILY_ORDER},
                "validation": {family: 2 for family in FAMILY_ORDER},
                "test": {family: 2 for family in FAMILY_ORDER},
            },
        )
        self.assertEqual(
            audit["accepted_cases_by_split_and_family"],
            {
                "train": {family: 2 for family in FAMILY_ORDER},
                "validation": {family: 1 for family in FAMILY_ORDER},
                "test": {family: 1 for family in FAMILY_ORDER},
            },
        )
        self.assertEqual(
            audit["accepted_rows_by_split_and_family"],
            {
                "train": {
                    family: 2 * ROWS_PER_ACCEPTED_CASE[family]
                    for family in FAMILY_ORDER
                },
                "validation": dict(ROWS_PER_ACCEPTED_CASE),
                "test": dict(ROWS_PER_ACCEPTED_CASE),
            },
        )
        self.assertEqual(
            audit["total_row_ownership"],
            {
                "attempted_trajectories": plan.attempted_cases,
                "declared_rows": plan.expected_rows,
                "row_owner_entries": plan.expected_rows,
                "expected_rows": plan.expected_rows,
            },
        )
        self.assertEqual(
            audit["source_binding"],
            {
                "committed_batches": 12,
                "committed_shards": 12,
                "source_trajectories": 28,
                "source_rows": 1_668,
            },
        )

    def test_postbuild_map_audit_rejects_wrong_cell_counts_and_rows(self) -> None:
        def matching_index(
            arrays: dict[str, np.ndarray],
            *,
            family_code: int,
            is_accepted: bool,
        ) -> int:
            matches = np.flatnonzero(
                (arrays["trajectory_split_id"] == train_code)
                & (arrays["trajectory_family_id"] == family_code)
                & (arrays["trajectory_accepted"] == is_accepted)
            )
            return int(matches[0])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _all_split_plan(root / "sources")
            train_code = split_code(SplitId.TRAIN)
            stokes_code = int(FAMILY_IDS["stokes"])
            tanaka_code = int(FAMILY_IDS["tanaka"])
            path = root / "combined.trajectory_map.npz"

            arrays = _trajectory_map_arrays(plan)
            rejected_stokes = matching_index(
                arrays,
                family_code=stokes_code,
                is_accepted=False,
            )
            arrays["trajectory_family_id"][rejected_stokes] = tanaka_code
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(RuntimeError, "wrong attempted count"):
                _audit_trajectory_map_against_plan(path, plan=plan)

            arrays = _trajectory_map_arrays(plan)
            accepted_stokes = matching_index(
                arrays,
                family_code=stokes_code,
                is_accepted=True,
            )
            rejected_tanaka = matching_index(
                arrays,
                family_code=tanaka_code,
                is_accepted=False,
            )
            arrays["trajectory_family_id"][
                [
                    accepted_stokes,
                    rejected_tanaka,
                ]
            ] = arrays["trajectory_family_id"][
                [
                    rejected_tanaka,
                    accepted_stokes,
                ]
            ]
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(RuntimeError, "wrong accepted count"):
                _audit_trajectory_map_against_plan(path, plan=plan)

            arrays = _trajectory_map_arrays(plan)
            accepted_stokes = matching_index(
                arrays,
                family_code=stokes_code,
                is_accepted=True,
            )
            accepted_tanaka = matching_index(
                arrays,
                family_code=tanaka_code,
                is_accepted=True,
            )
            arrays["trajectory_family_id"][
                [
                    accepted_stokes,
                    accepted_tanaka,
                ]
            ] = arrays["trajectory_family_id"][
                [
                    accepted_tanaka,
                    accepted_stokes,
                ]
            ]
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(RuntimeError, "wrong accepted-row sum"):
                _audit_trajectory_map_against_plan(path, plan=plan)

    def test_postbuild_map_audit_rejects_equal_row_family_swap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _all_split_plan(root / "sources")
            arrays = _trajectory_map_arrays(plan)
            train_code = split_code(SplitId.TRAIN)
            indices = tuple(
                int(
                    np.flatnonzero(
                        (arrays["trajectory_split_id"] == train_code)
                        & (arrays["trajectory_family_id"] == int(FAMILY_IDS[family]))
                        & arrays["trajectory_accepted"]
                    )[0]
                )
                for family in ("tanaka", "benjamin_feir")
            )
            arrays["trajectory_family_id"][list(indices)] = arrays[
                "trajectory_family_id"
            ][list(reversed(indices))]
            path = root / "combined.trajectory_map.npz"
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(
                RuntimeError,
                "identity sequence differs",
            ):
                _audit_trajectory_map_against_plan(path, plan=plan)

    def test_postbuild_map_audit_rejects_inconsistent_row_owners(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _all_split_plan(root / "sources")
            arrays = _trajectory_map_arrays(plan)
            arrays["trajectory_index"][0] = arrays["trajectory_index"][1]
            path = root / "combined.trajectory_map.npz"
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(
                RuntimeError,
                "row owners disagree with trajectory counts",
            ):
                _audit_trajectory_map_against_plan(path, plan=plan)

    def test_postbuild_map_audit_requires_every_schema_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _all_split_plan(root / "sources")
            arrays = _trajectory_map_arrays(plan)
            arrays.pop("trajectory_case_id")
            path = root / "combined.trajectory_map.npz"
            np.savez_compressed(path, **arrays)

            with self.assertRaisesRegex(RuntimeError, "exact schema-v2 arrays"):
                _audit_trajectory_map_against_plan(path, plan=plan)

    def test_postbuild_map_audit_binds_all_identity_and_row_fields(self) -> None:
        mutations = (
            ("trajectory_case_id", 1),
            ("trajectory_cell_id", 1),
            ("trajectory_required_bits", 1),
            ("trajectory_evaluated_bits", 1),
            ("trajectory_failed_bits", 1),
            ("frame_index", 1),
            ("shard_index", 1),
            ("shard_row", 1),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _all_split_plan(root / "sources")
            path = root / "combined.trajectory_map.npz"
            for name, delta in mutations:
                with self.subTest(name=name):
                    arrays = {
                        key: value.copy()
                        for key, value in _trajectory_map_arrays(plan).items()
                    }
                    arrays[name][0] += delta
                    np.savez_compressed(path, **arrays)
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "differs from its committed source",
                    ):
                        _audit_trajectory_map_against_plan(path, plan=plan)

    def test_postbuild_map_audit_rejects_source_acceptance_identity_swap(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _all_split_plan(root / "sources")
            arrays = _trajectory_map_arrays(plan)
            train_stokes = np.flatnonzero(
                (arrays["trajectory_split_id"] == split_code(SplitId.TRAIN))
                & (arrays["trajectory_family_id"] == int(FAMILY_IDS["stokes"]))
            )
            accepted_index = int(
                train_stokes[
                    np.flatnonzero(arrays["trajectory_accepted"][train_stokes])[0]
                ]
            )
            rejected_index = int(
                train_stokes[
                    np.flatnonzero(~arrays["trajectory_accepted"][train_stokes])[0]
                ]
            )
            for name in (
                "trajectory_accepted",
                "trajectory_required_bits",
                "trajectory_evaluated_bits",
                "trajectory_failed_bits",
                "trajectory_row_count",
            ):
                arrays[name][[accepted_index, rejected_index]] = arrays[name][
                    [rejected_index, accepted_index]
                ]
            row_counts = arrays["trajectory_row_count"]
            first_rows = np.cumsum(row_counts, dtype=np.int64) - row_counts
            first_rows[~arrays["trajectory_accepted"]] = -1
            arrays["trajectory_first_row"] = first_rows
            arrays["trajectory_index"] = np.repeat(
                np.arange(row_counts.size, dtype=np.int32),
                row_counts,
            )
            path = root / "combined.trajectory_map.npz"
            np.savez_compressed(path, **arrays)

            with self.assertRaisesRegex(
                RuntimeError,
                "differs from its committed source",
            ):
                _audit_trajectory_map_against_plan(path, plan=plan)

    def test_postbuild_view_binds_manifest_sources_and_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _all_split_plan(root / "sources")
            view = _write_valid_combined_view(root / "view", plan=plan)

            validated = _validate_view(view, plan=plan)

            self.assertEqual(
                validated["manifest_source_audit"],
                {"committed_batches": 12, "committed_shards": 12},
            )

    def test_postbuild_view_rejects_corrupt_manifest_source_records(self) -> None:
        mutations = (
            ("empty_batches", "dataset_batches", None, []),
            ("batch_path", "dataset_batches", "proposal_path", "wrong.npz"),
            ("batch_hash", "dataset_batches", "proposal_sha256", "0" * 64),
            ("batch_count", "dataset_batches", "n_rows", -1),
            ("empty_shards", "dataset_shards", None, []),
            ("shard_path", "dataset_shards", "path", "wrong.npz"),
            ("shard_hash", "dataset_shards", "sha256", "0" * 64),
            ("shard_count", "dataset_shards", "n_rows", -1),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _all_split_plan(root / "sources")
            view = _write_valid_combined_view(root / "view", plan=plan)
            original = json.loads(view.manifest.read_text(encoding="utf-8"))
            for name, collection, field, value in mutations:
                with self.subTest(name=name):
                    manifest = json.loads(json.dumps(original))
                    if field is None:
                        manifest[collection] = value
                    else:
                        manifest[collection][0][field] = value
                    view.manifest.write_text(
                        json.dumps(manifest, sort_keys=True, allow_nan=False),
                        encoding="utf-8",
                    )
                    expected_message = (
                        "batch records"
                        if collection == "dataset_batches"
                        else "shard records"
                    )
                    with self.assertRaisesRegex(RuntimeError, expected_message):
                        _validate_view(view, plan=plan)

    def test_postbuild_view_recomputes_dataset_contract_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _all_split_plan(root / "sources")
            view = _write_valid_combined_view(root / "view", plan=plan)
            manifest = json.loads(view.manifest.read_text(encoding="utf-8"))
            manifest["dataset_contract_fingerprint"] = "0" * 64
            view.manifest.write_text(
                json.dumps(manifest, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "fingerprint is incorrect"):
                _validate_view(view, plan=plan)

    def test_train_validation_and_test_combine_in_one_view(self) -> None:
        chunks = tuple(
            _chunk(
                family,
                before=0,
                count=count,
                stream_id=0,
                suffix=f"{split.value[0]}{index}",
                split=split,
            )
            for split, count in (
                (SplitId.TRAIN, 32),
                (SplitId.VALIDATION, 8),
                (SplitId.TEST, 4),
            )
            for index, family in enumerate(FAMILY_ORDER)
        )
        plan = validate_combined_plan(chunks)

        self.assertEqual(
            plan.splits,
            (SplitId.TRAIN, SplitId.VALIDATION, SplitId.TEST),
        )
        self.assertEqual(
            plan.accepted_cases_per_family_by_split,
            {"train": 32, "validation": 8, "test": 4},
        )
        self.assertEqual(plan.accepted_cases, 4 * 44)
        self.assertEqual(plan.expected_rows, 417 * 44)

    def test_gap_duplicate_stream_and_unequal_family_total_fail(self) -> None:
        baseline = [
            _chunk(
                family,
                before=0,
                count=16,
                stream_id=0,
                suffix=chr(ord("a") + index),
            )
            for index, family in enumerate(FAMILY_ORDER)
        ]

        gap = [
            *baseline,
            _chunk(
                "stokes",
                before=17,
                count=1,
                stream_id=1,
                suffix="z",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "gap or overlap"):
            validate_combined_plan(gap)

        repeated_stream = [
            *baseline,
            _chunk(
                "stokes",
                before=16,
                count=1,
                stream_id=0,
                suffix="y",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "repeat a stream"):
            validate_combined_plan(repeated_stream)

        unequal = [chunk for chunk in baseline if chunk.family != "jonswap_tma"]
        unequal.append(
            _chunk(
                "jonswap_tma",
                before=0,
                count=15,
                stream_id=0,
                suffix="x",
            )
        )
        with self.assertRaisesRegex(ValueError, "equal accepted-case counts"):
            validate_combined_plan(unequal)

        missing_test_family = [
            _chunk(
                family,
                before=0,
                count=4,
                stream_id=0,
                suffix=f"m{index}",
                split=SplitId.TEST,
            )
            for index, family in enumerate(FAMILY_ORDER[:-1])
        ]
        with self.assertRaisesRegex(
            ValueError,
            "test view requires all four",
        ):
            validate_combined_plan([*baseline, *missing_test_family])

    def test_source_and_platform_identity_are_scoped_by_family_revision(
        self,
    ) -> None:
        other_families = [
            _chunk(
                family,
                before=0,
                count=16,
                stream_id=0,
                suffix=f"base{index}",
            )
            for index, family in enumerate(FAMILY_ORDER)
            if family != "stokes"
        ]
        mismatched_sources = [
            _chunk(
                "stokes",
                before=0,
                count=8,
                stream_id=0,
                suffix="source_a",
            ),
            _chunk(
                "stokes",
                before=8,
                count=8,
                stream_id=1,
                suffix="source_b",
                source_digest="9" * 64,
            ),
            *other_families,
        ]
        with self.assertRaisesRegex(
            ValueError,
            "stokes revision 2 chunks use different source mappings",
        ):
            validate_combined_plan(mismatched_sources)

        mismatched_platforms = [
            _chunk(
                "stokes",
                before=0,
                count=8,
                stream_id=0,
                suffix="platform_a",
            ),
            _chunk(
                "stokes",
                before=8,
                count=8,
                stream_id=1,
                suffix="platform_b",
                execution_platform="gpu",
            ),
            *other_families,
        ]
        with self.assertRaisesRegex(
            ValueError,
            "stokes revision 2 chunks use different execution platforms",
        ):
            validate_combined_plan(mismatched_platforms)

    def test_tanaka_chunks_with_different_sources_cannot_mix(self) -> None:
        other_families = [
            _chunk(
                family,
                before=0,
                count=16,
                stream_id=0,
                suffix=f"base_{index}",
            )
            for index, family in enumerate(FAMILY_ORDER)
            if family != "tanaka"
        ]
        first = _chunk(
            "tanaka",
            before=0,
            count=8,
            stream_id=0,
            suffix="first",
            source_digest="6" * 64,
        )
        second = _chunk(
            "tanaka",
            before=8,
            count=8,
            stream_id=1,
            suffix="second",
            source_digest="7" * 64,
        )
        with self.assertRaisesRegex(ValueError, "different source mappings"):
            validate_combined_plan((*other_families, first, second))

    def test_paper_view_rejects_noncurrent_family_revision(self) -> None:
        for stale_family in FAMILY_ORDER:
            with self.subTest(family=stale_family):
                current_revision = paper_dataset_revision_id(FAMILY_IDS[stale_family])
                chunks = tuple(
                    _chunk(
                        family,
                        before=0,
                        count=8,
                        stream_id=0,
                        suffix=f"stale_{stale_family}_{index}",
                        revision_id=(
                            current_revision - 1 if family == stale_family else None
                        ),
                    )
                    for index, family in enumerate(FAMILY_ORDER)
                )

                with self.assertRaisesRegex(
                    ValueError,
                    rf"{stale_family} chunk revision .* is not current",
                ):
                    validate_combined_plan(chunks)

    def test_dependency_environment_remains_global(self) -> None:
        chunks = tuple(
            _chunk(
                family,
                before=0,
                count=8,
                stream_id=0,
                suffix=f"dependency_{index}",
                dependency_fingerprint=(
                    "e" * 64 if family == "jonswap_tma" else "d" * 64
                ),
            )
            for index, family in enumerate(FAMILY_ORDER)
        )

        with self.assertRaisesRegex(ValueError, "dependency environments"):
            validate_combined_plan(chunks)

    def test_shared_target_implementation_remains_global(self) -> None:
        chunks = [
            _chunk(
                family,
                before=0,
                count=8,
                stream_id=0,
                suffix=f"shared_target_{index}",
            )
            for index, family in enumerate(FAMILY_ORDER)
        ]
        changed = chunks[-1]
        chunks[-1] = replace(
            changed,
            source_sha256={
                **changed.source_sha256,
                SHARED_TARGET_SOURCE_PATHS[0]: "b" * 64,
            },
        )

        with self.assertRaisesRegex(
            ValueError,
            "different solver/gen_data/pipeline/reference.py implementations",
        ):
            validate_combined_plan(chunks)


if __name__ == "__main__":
    unittest.main()
