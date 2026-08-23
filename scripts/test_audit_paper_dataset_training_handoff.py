"""Focused tests for the streaming training-handoff audit."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault(
    "MPLCONFIGDIR",
    "/tmp/mpl-paper-dataset-training-handoff-tests",
)

import numpy as np

from scripts import audit_paper_dataset_training_handoff as handoff_audit

from solver.gen_data.pipeline.archive import (
    BatchPaths,
    CaseCommitRecord,
    commit_batch,
    ensure_proposal,
    ensure_shard,
    file_sha256,
)
from solver.gen_data.pipeline.manifest import build_dataset_view
from solver.gen_data.pipeline.production import SplitId, split_code
from solver.gen_data.pipeline.quota_driver import canonical_json_sha256

from scripts.audit_paper_dataset_training_handoff import (
    AUDIT_SCHEMA,
    COMBINED_PREFLIGHT_SCHEMA,
    COMBINED_SUMMARY_SCHEMA,
    Artifact,
    DatasetContract,
    FAMILY_IDS,
    FINAL_CONTRACT,
    FileAuthenticator,
    LoadedInput,
    Shard,
    TRAINING_IMPLEMENTATION_FILES,
    TRAINING_IMPLEMENTATION_SCHEMA,
    _default_output,
    _training_implementation_record,
    audit_training_handoff,
    compute_streaming_stats,
    parse_args,
    write_audit,
)


TRAIN_DIR = Path(__file__).resolve().parents[1] / "train-jax-10m"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from util import (  # noqa: E402
    _index_selection,
    build_dataset_split_indices,
    load_dataset_arrays,
    load_or_compute_stats,
)


FAMILIES = tuple(FAMILY_IDS)
SPLITS = tuple(split.value for split in SplitId)
REVISION_BY_FAMILY = {
    "stokes": 2,
    "tanaka": 3,
    "benjamin_feir": 4,
    "jonswap_tma": 4,
}
SYNTHETIC_CONTRACT = DatasetContract(
    accepted_cases_per_family_by_split={split: 1 for split in SPLITS},
    rows_per_accepted_case={family: 2 for family in FAMILIES},
    source_count_by_family={family: 3 for family in FAMILIES},
    chunk_layout_by_family_and_split={
        split: {
            family: ((0, 1, split_index * len(FAMILIES) + family_index),)
            for family_index, family in enumerate(FAMILIES)
        }
        for split_index, split in enumerate(SPLITS)
    },
    revision_by_family=REVISION_BY_FAMILY,
    nx=4,
    length=2.0 * np.pi,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _trajectory_metadata() -> dict[str, object]:
    return {
        "case_kind": "trajectory",
        "trajectory_execution": {
            "role": "paper_dataset_test",
            "numerical": {
                "nx": 4,
                "length": 2.0 * np.pi,
                "gravity": 9.81,
                "dno_order": 6,
                "pad_factor": 8,
                "maximum_wavenumber": 1.0,
                "dtype": "float64",
            },
        },
    }


def _write_batch(
    root: Path,
    *,
    family: str,
    split: str,
    ordinal: int,
) -> tuple[BatchPaths, str, dict[str, object]]:
    paths = BatchPaths.under(
        root,
        family=family,
        split=split,
        batch_id=0,
    )
    family_id = FAMILY_IDS[family]
    split_id = split_code(SplitId(split))
    case_id = (split_id << 60) | (family_id << 40) | ordinal
    run_spec: dict[str, object] = {
        "schema": "paper_dataset_accepted_quota_run_v2",
        "family_name": family,
        "family_id": family_id,
        "revision_id": REVISION_BY_FAMILY[family],
        "split_id": split,
        "stream_id": ordinal,
        "configuration": {
            "accepted_case_count": 1,
            "accepted_cases_before": 0,
            "accepted_cases_after": 1,
        },
    }
    fingerprint = canonical_json_sha256(run_spec)
    proposal = {
        "config_fingerprint": np.asarray(fingerprint),
        "family_id": np.asarray(family_id, dtype=np.int16),
        "revision_id": np.asarray(REVISION_BY_FAMILY[family], dtype=np.int16),
        "split_id": np.asarray(split_id, dtype=np.uint8),
        "batch_id": np.asarray(0, dtype=np.int64),
        "case_id": np.asarray((case_id,), dtype=np.int64),
        "cell_id": np.asarray((ordinal,), dtype=np.int32),
        "case_spec_json": np.asarray((json.dumps({"ordinal": ordinal}),)),
        "metadata_json": np.asarray(json.dumps(_trajectory_metadata())),
    }
    proposal_sha256 = ensure_proposal(paths, proposal)
    base = np.float32(ordinal * 10.0)
    eta = base + np.asarray(
        ((-1.0, 0.0, 1.0, 2.0), (2.0, 3.0, 4.0, 5.0)),
        dtype=np.float32,
    )
    xi = base + np.asarray(
        ((4.0, 1.0, -2.0, 9.0), (7.0, 3.0, 1.0, -5.0)),
        dtype=np.float32,
    )
    gxi = -base + np.asarray(
        ((-3.0, 2.0, 5.0, 7.0), (8.0, -4.0, 1.0, 0.0)),
        dtype=np.float32,
    )
    depth = 0.75 + ordinal / 100.0
    shard = {
        "eta": eta,
        "xi": xi,
        "gxi": gxi,
        "depth": np.asarray((depth, depth), dtype=np.float64),
        "time": np.asarray((0.0, 1.0), dtype=np.float64),
        "case_local_index": np.asarray((0, 0), dtype=np.int32),
        "frame_index": np.asarray((0, 1), dtype=np.int32),
        "selected_dense_index": np.asarray((0, 4), dtype=np.int32),
        "config_fingerprint": np.asarray(fingerprint),
        "proposal_sha256": np.asarray(proposal_sha256),
    }
    ensure_shard(paths, shard)
    commit_batch(
        paths,
        cases=(
            CaseCommitRecord(
                case_id=case_id,
                accepted=True,
                required_bits=1,
                evaluated_bits=1,
                failed_bits=0,
                first_row=0,
                row_count=2,
                metrics={},
            ),
        ),
        metadata={},
    )
    return paths, fingerprint, run_spec


def _fixture(root: Path) -> tuple[Path, Path, tuple[Path, ...]]:
    batches: list[BatchPaths] = []
    chunks: list[dict[str, object]] = []
    source_summaries: list[Path] = []
    fingerprints: list[str] = []
    attempted_by_split = {split: 0 for split in SPLITS}
    ordinal = 0
    for split in SPLITS:
        for family in FAMILIES:
            source_root = root / "sources" / split / family
            source_root.mkdir(parents=True)
            batch, fingerprint, run_spec = _write_batch(
                source_root,
                family=family,
                split=split,
                ordinal=ordinal,
            )
            batches.append(batch)
            fingerprints.append(fingerprint)
            source_view = build_dataset_view(
                source_root,
                (batch,),
                name=f"paper_dataset_{family}_{split}",
                expected_fingerprint=fingerprint,
            )
            source_summary = source_root / f"paper_dataset_{family}_{split}.summary.json"
            _write_json(
                source_summary,
                {
                    "schema": "paper_dataset_quota_summary_v1",
                    "status": "complete",
                    "configuration_fingerprint": fingerprint,
                    "output_root": str(source_root.resolve()),
                    "run_spec": run_spec,
                    "counts": {"accepted": 1, "attempted": 1},
                    "dataset_view": {
                        "manifest": _artifact(source_view.manifest),
                        "trajectory_map": _artifact(source_view.trajectory_map),
                    },
                },
            )
            source_summaries.append(source_summary)
            chunks.append(
                {
                    "family": family,
                    "revision_id": REVISION_BY_FAMILY[family],
                    "split": split,
                    "stream_id": ordinal,
                    "accepted_before": 0,
                    "accepted_count": 1,
                    "accepted_after": 1,
                    "attempted_count": 1,
                    "configuration_fingerprint": fingerprint,
                    "summary_path": str(source_summary.resolve()),
                    "summary_sha256": file_sha256(source_summary),
                }
            )
            attempted_by_split[split] += 1
            ordinal += 1

    combined_root = root / "combined"
    combined_root.mkdir()
    combined_view = build_dataset_view(
        combined_root,
        tuple(batches),
        name="paper_dataset_all_splits_c00001",
        expected_fingerprints=fingerprints,
    )
    manifest = json.loads(combined_view.manifest.read_text(encoding="utf-8"))
    expected_rows = {split: {family: 2 for family in FAMILIES} for split in SPLITS}
    combined_summary = combined_root / "paper_dataset_all_splits_c00001.summary.json"
    _write_json(
        combined_summary,
        {
            "schema": COMBINED_SUMMARY_SCHEMA,
            "status": "complete",
            "preflight": {
                "schema": COMBINED_PREFLIGHT_SCHEMA,
                "splits": list(SPLITS),
                "accepted_cases_per_family_by_split": {split: 1 for split in SPLITS},
                "accepted_cases_total": 12,
                "attempted_cases_total": 12,
                "attempted_cases_by_split": attempted_by_split,
                "expected_rows": 24,
                "expected_rows_by_split_and_family": expected_rows,
                "configuration_fingerprints": sorted(fingerprints),
                "chunks": chunks,
            },
            "dataset_view": {
                "manifest": _artifact(combined_view.manifest),
                "trajectory_map": _artifact(combined_view.trajectory_map),
                "dataset_contract_fingerprint": manifest[
                    "dataset_contract_fingerprint"
                ],
            },
        },
    )
    return combined_summary, combined_view.manifest, tuple(source_summaries)


def _refresh_combined_hashes(summary_path: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    map_path = manifest_path.parent / manifest["trajectory_map_npz"]
    manifest["trajectory_map_sha256"] = file_sha256(map_path)
    _write_json(manifest_path, manifest)
    summary["dataset_view"]["trajectory_map"] = _artifact(map_path)
    summary["dataset_view"]["manifest"] = _artifact(manifest_path)
    _write_json(summary_path, summary)


def _refresh_source_summary_hash(
    combined_summary_path: Path,
    source_summary_path: Path,
) -> None:
    summary = json.loads(combined_summary_path.read_text(encoding="utf-8"))
    matching = [
        chunk
        for chunk in summary["preflight"]["chunks"]
        if Path(chunk["summary_path"]).resolve() == source_summary_path.resolve()
    ]
    if len(matching) != 1:
        raise RuntimeError("fixture source summary is not unique")
    matching[0]["summary_sha256"] = file_sha256(source_summary_path)
    _write_json(combined_summary_path, summary)


class StreamingTrainingHandoffAuditTests(unittest.TestCase):
    def test_final_contract_has_the_exact_release_population(self) -> None:
        self.assertEqual(FINAL_CONTRACT.source_count, 26)
        self.assertEqual(FINAL_CONTRACT.accepted_cases, 73_728)
        self.assertEqual(FINAL_CONTRACT.accepted_rows, 7_686_144)
        self.assertEqual(
            sum(
                FINAL_CONTRACT.accepted_cases_per_family_by_split["train"]
                * FINAL_CONTRACT.rows_per_accepted_case[family]
                for family in FAMILIES
            ),
            6_832_128,
        )

    def test_streaming_stats_and_selection_match_trainer_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary_path, manifest_path, _ = _fixture(Path(directory))
            audit = audit_training_handoff(
                summary_path,
                seed=0,
                contract=SYNTHETIC_CONTRACT,
            )
            dataset = load_dataset_arrays(manifest_path)
            train_indices, _, _ = build_dataset_split_indices(dataset, seed=0)
            expected = load_or_compute_stats(
                manifest_path,
                dataset=dataset,
                indices=train_indices,
            )
            observed = audit["training_normalization"]["stats"]

            self.assertEqual(audit["schema"], AUDIT_SCHEMA)
            self.assertEqual(audit["status"], "complete")
            self.assertEqual(observed, expected)
            self.assertEqual(
                observed["index_selection"], _index_selection(train_indices)
            )
            self.assertEqual(observed["num_examples"], 8)
            self.assertEqual(observed["storage_num_examples"], 24)
            self.assertFalse(audit["execution"]["gpu_used"])
            implementation = audit["training_implementation"]
            self.assertEqual(implementation["schema"], TRAINING_IMPLEMENTATION_SCHEMA)
            self.assertEqual(
                [item["path"] for item in implementation["files"]],
                [path for path, _ in TRAINING_IMPLEMENTATION_FILES],
            )
            self.assertFalse(
                implementation["semantic_relationship"]["formal_equivalence_claim"]
            )
            self.assertEqual(
                implementation["fingerprint"],
                canonical_json_sha256(
                    {
                        name: implementation[name]
                        for name in ("schema", "files", "semantic_relationship")
                    }
                ),
            )
            handoff = audit["training_normalization"]["handoff"]
            self.assertFalse(handoff["stats_cache_written_by_audit"])
            self.assertEqual(
                Path(handoff["expected_trainer_stats_cache"]),
                manifest_path.with_suffix(".stats.json").resolve(),
            )
            self.assertIn("--dataset", handoff["canonical_command_template"])
            dataset_position = handoff["canonical_command_template"].index("--dataset")
            self.assertEqual(
                handoff["canonical_command_template"][dataset_position + 1],
                str(manifest_path.resolve()),
            )
            self.assertIn("--norm", handoff["canonical_command_template"])
            self.assertIn("scale", handoff["canonical_command_template"])

            expected_splits = build_dataset_split_indices(dataset, seed=0)
            observed_selection = audit["training_normalization"][
                "full_row_selection"
            ]
            self.assertEqual(
                observed_selection["schema"],
                "paper_dataset_full_row_selection_v1",
            )
            for split, expected_indices in zip(
                ("train", "validation", "test"), expected_splits
            ):
                observed_split = observed_selection["splits"][split]
                self.assertEqual(
                    observed_split["sha256"],
                    _index_selection(expected_indices)["sha256"],
                )
                self.assertEqual(observed_split["count"], expected_indices.size)
                self.assertTrue(observed_split["all_retained_rows"])
            self.assertEqual(
                observed_selection["test_rows_in_train_or_validation"], 0
            )

    def test_combined_summary_schema_status_and_contract_counts_fail_closed(
        self,
    ) -> None:
        mutations = (
            ("schema", "wrong", "wrong schema"),
            ("status", "incomplete", "not complete"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                summary_path, _, _ = _fixture(Path(directory))
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary[field] = value
                _write_json(summary_path, summary)
                with self.assertRaisesRegex(ValueError, message):
                    audit_training_handoff(summary_path, contract=SYNTHETIC_CONTRACT)

        with tempfile.TemporaryDirectory() as directory:
            summary_path, _, _ = _fixture(Path(directory))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["preflight"]["expected_rows"] -= 1
            _write_json(summary_path, summary)
            with self.assertRaisesRegex(RuntimeError, "expected_rows differs"):
                audit_training_handoff(summary_path, contract=SYNTHETIC_CONTRACT)

        with tempfile.TemporaryDirectory() as directory:
            summary_path, _, _ = _fixture(Path(directory))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["preflight"]["chunks"][0]["stream_id"] = 99
            _write_json(summary_path, summary)
            with self.assertRaisesRegex(RuntimeError, "frozen chunk layout differs"):
                audit_training_handoff(summary_path, contract=SYNTHETIC_CONTRACT)

    def test_manifest_hash_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary_path, manifest_path, _ = _fixture(Path(directory))
            manifest_path.write_text("forged\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "manifest SHA-256 differs"):
                audit_training_handoff(summary_path, contract=SYNTHETIC_CONTRACT)

    def test_map_hash_and_semantic_split_mutations_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary_path, manifest_path, _ = _fixture(Path(directory))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            map_path = manifest_path.parent / manifest["trajectory_map_npz"]
            with map_path.open("ab") as handle:
                handle.write(b"forged")
            with self.assertRaisesRegex(RuntimeError, "map SHA-256 differs"):
                audit_training_handoff(summary_path, contract=SYNTHETIC_CONTRACT)

        with tempfile.TemporaryDirectory() as directory:
            summary_path, manifest_path, _ = _fixture(Path(directory))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            map_path = manifest_path.parent / manifest["trajectory_map_npz"]
            with np.load(map_path, allow_pickle=False) as archive:
                arrays = {
                    name: np.asarray(archive[name]).copy() for name in archive.files
                }
            arrays["trajectory_split_id"][0] = np.uint8(1)
            np.savez_compressed(map_path, **arrays)
            _refresh_combined_hashes(summary_path, manifest_path)
            with self.assertRaisesRegex(
                RuntimeError, "population differs|differs from source"
            ):
                audit_training_handoff(summary_path, contract=SYNTHETIC_CONTRACT)

    def test_map_coordinate_mutation_fails_even_when_combined_hashes_are_refreshed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary_path, manifest_path, _ = _fixture(Path(directory))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            map_path = manifest_path.parent / manifest["trajectory_map_npz"]
            with np.load(map_path, allow_pickle=False) as archive:
                arrays = {
                    name: np.asarray(archive[name]).copy() for name in archive.files
                }
            arrays["frame_index"][0] += np.int32(1)
            np.savez_compressed(map_path, **arrays)
            _refresh_combined_hashes(summary_path, manifest_path)
            with self.assertRaisesRegex(RuntimeError, "differs from source"):
                audit_training_handoff(summary_path, contract=SYNTHETIC_CONTRACT)

    def test_source_summary_hash_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary_path, _, source_summaries = _fixture(Path(directory))
            source_summaries[0].write_text("forged\n", encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError, "source summary 0 SHA-256 differs"
            ):
                audit_training_handoff(summary_path, contract=SYNTHETIC_CONTRACT)

    def test_source_summary_semantics_are_bound_after_hash_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary_path, _, source_summaries = _fixture(Path(directory))
            source_summary = json.loads(source_summaries[0].read_text(encoding="utf-8"))
            source_summary["run_spec"]["split_id"] = "test"
            _write_json(source_summaries[0], source_summary)
            _refresh_source_summary_hash(summary_path, source_summaries[0])
            with self.assertRaisesRegex(RuntimeError, "run identity differs"):
                audit_training_handoff(summary_path, contract=SYNTHETIC_CONTRACT)

    def test_physical_proposal_and_result_hashes_are_authenticated(self) -> None:
        for path_field, expected_message in (
            ("proposal_path", "proposal_path SHA-256 differs"),
            ("result_path", "result_path SHA-256 differs"),
        ):
            with (
                self.subTest(path_field=path_field),
                tempfile.TemporaryDirectory() as directory,
            ):
                summary_path, _, source_summaries = _fixture(Path(directory))
                source_summary = json.loads(
                    source_summaries[0].read_text(encoding="utf-8")
                )
                source_manifest_path = Path(
                    source_summary["dataset_view"]["manifest"]["path"]
                )
                source_manifest = json.loads(
                    source_manifest_path.read_text(encoding="utf-8")
                )
                artifact_path = (
                    source_manifest_path.parent
                    / source_manifest["dataset_batches"][0][path_field]
                ).resolve()
                with artifact_path.open("ab") as handle:
                    handle.write(b"forged")
                with self.assertRaisesRegex(RuntimeError, expected_message):
                    audit_training_handoff(summary_path, contract=SYNTHETIC_CONTRACT)

    def test_authenticator_detects_change_before_semantic_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            _write_json(path, {"status": "complete"})
            authenticator = FileAuthenticator()
            artifact = authenticator.authenticate(path, context="record")
            _write_json(path, {"status": "changed"})
            with self.assertRaisesRegex(RuntimeError, "changed after authentication"):
                authenticator.assert_unchanged(artifact, context="record")

    def test_shard_hash_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary_path, manifest_path, _ = _fixture(Path(directory))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            shard_path = (
                manifest_path.parent / manifest["dataset_shards"][0]["path"]
            ).resolve()
            with shard_path.open("ab") as handle:
                handle.write(b"forged")
            with self.assertRaisesRegex(RuntimeError, "shard 0 SHA-256 differs"):
                audit_training_handoff(summary_path, contract=SYNTHETIC_CONTRACT)

    def test_selected_training_depth_must_be_strictly_positive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard_path = root / "shard.npz"
            np.savez(
                shard_path,
                eta=np.zeros((1, 4), dtype=np.float32),
                xi=np.zeros((1, 4), dtype=np.float32),
                gxi=np.zeros((1, 4), dtype=np.float32),
                depth=np.asarray((-0.5,), dtype=np.float64),
                time=np.asarray((0.0,), dtype=np.float64),
            )
            placeholder = root / "placeholder.json"
            placeholder.write_text("{}\n", encoding="utf-8")
            placeholder_artifact = Artifact(
                path=placeholder.resolve(),
                bytes=placeholder.stat().st_size,
                sha256=file_sha256(placeholder),
            )
            loaded = LoadedInput(
                summary=placeholder_artifact,
                manifest=placeholder_artifact,
                trajectory_map=placeholder_artifact,
                manifest_record={
                    "schema_version": 2,
                    "dataset_contract_fingerprint": "b" * 64,
                    "trajectory_map_sha256": "c" * 64,
                    "grid": {"nx": 4, "length": 2.0 * np.pi},
                },
                map_arrays={
                    "trajectory_index": np.asarray((0,), dtype=np.int32),
                    "trajectory_accepted": np.asarray((True,), dtype=np.bool_),
                    "trajectory_split_id": np.asarray((0,), dtype=np.uint8),
                },
                shards=(
                    Shard(
                        index=0,
                        path=shard_path.resolve(),
                        sha256=file_sha256(shard_path),
                        rows=1,
                        configuration_fingerprint="d" * 64,
                    ),
                ),
                chunks=(),
                source_artifacts=(),
                attempted_cases=1,
            )
            with self.assertRaisesRegex(RuntimeError, "strictly positive"):
                compute_streaming_stats(
                    loaded,
                    authenticator=FileAuthenticator(),
                    seed=0,
                )

    def test_atomic_writer_publishes_only_complete_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "audit.json"
            record = {
                "schema": AUDIT_SCHEMA,
                "status": "complete",
                "training_implementation": _training_implementation_record(),
            }
            write_audit(output, record)
            self.assertEqual(json.loads(output.read_text())["status"], "complete")
            write_audit(output, record)
            with self.assertRaisesRegex(FileExistsError, "different content"):
                write_audit(
                    output,
                    {
                        "schema": AUDIT_SCHEMA,
                        "status": "complete",
                        "training_implementation": _training_implementation_record(),
                        "seed": 1,
                    },
                )
            with self.assertRaisesRegex(ValueError, "only a complete"):
                write_audit(
                    root / "bad.json",
                    {"schema": AUDIT_SCHEMA, "status": "incomplete"},
                )

    def test_atomic_writer_refuses_unrelated_or_authenticated_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unrelated = root / "unrelated.json"
            _write_json(unrelated, {"schema": "other"})
            with self.assertRaisesRegex(FileExistsError, "not a training-handoff"):
                write_audit(
                    unrelated,
                    {
                        "schema": AUDIT_SCHEMA,
                        "status": "complete",
                        "training_implementation": (_training_implementation_record()),
                    },
                )

            authenticated = root / "manifest.dataset.json"
            _write_json(authenticated, {"schema": "input"})
            record = {
                "schema": AUDIT_SCHEMA,
                "status": "complete",
                "training_implementation": _training_implementation_record(),
                "dataset_view": {"manifest": {"path": str(authenticated.resolve())}},
            }
            with self.assertRaisesRegex(FileExistsError, "authenticated input"):
                write_audit(authenticated, record)

    def test_atomic_writer_rehashes_training_code_immediately_before_publish(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repository"
            for relative, _ in TRAINING_IMPLEMENTATION_FILES:
                path = repository / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"# {relative}\n", encoding="utf-8")
            record = {
                "schema": AUDIT_SCHEMA,
                "status": "complete",
                "training_implementation": _training_implementation_record(
                    repository_root=repository
                ),
            }
            output = Path(directory) / "audit.json"
            trainer = repository / "train-jax-10m/util.py"
            original = handoff_audit._authenticate_training_implementation_record
            calls = 0

            def mutate_before_second_authentication(value: object) -> object:
                nonlocal calls
                calls += 1
                if calls == 2:
                    trainer.write_text("# changed\n", encoding="utf-8")
                return original(value, repository_root=repository)

            with (
                patch.object(
                    handoff_audit,
                    "_authenticate_training_implementation_record",
                    side_effect=mutate_before_second_authentication,
                ),
                self.assertRaisesRegex(RuntimeError, "SHA-256 differs"),
            ):
                write_audit(output, record)
            self.assertEqual(calls, 2)
            self.assertFalse(output.exists())

    def test_training_code_set_sweep_detects_earlier_file_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repository"
            for relative, _ in TRAINING_IMPLEMENTATION_FILES:
                path = repository / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"# {relative}\n", encoding="utf-8")
            record = _training_implementation_record(repository_root=repository)
            first = repository / TRAINING_IMPLEMENTATION_FILES[0][0]
            original = handoff_audit.FileAuthenticator.authenticate
            mutated = False

            def mutate_first_while_hashing_later(
                authenticator: FileAuthenticator,
                path: Path,
                **kwargs: object,
            ) -> Artifact:
                nonlocal mutated
                artifact = original(authenticator, path, **kwargs)
                if path.name == "util.py" and not mutated:
                    first.write_text("# changed after first hash\n", encoding="utf-8")
                    mutated = True
                return artifact

            with (
                patch.object(
                    handoff_audit.FileAuthenticator,
                    "authenticate",
                    new=mutate_first_while_hashing_later,
                ),
                self.assertRaisesRegex(RuntimeError, "changed after authentication"),
            ):
                handoff_audit._authenticate_training_implementation_record(
                    record, repository_root=repository
                )

    def test_training_code_set_sweep_rejects_symlink_substitution(self) -> None:
        for substitution in ("direct", "ancestor"):
            with (
                self.subTest(substitution=substitution),
                tempfile.TemporaryDirectory() as directory,
            ):
                repository = Path(directory) / "repository"
                for relative, _ in TRAINING_IMPLEMENTATION_FILES:
                    path = repository / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"# {relative}\n", encoding="utf-8")
                record = _training_implementation_record(repository_root=repository)
                original_authenticate = handoff_audit.FileAuthenticator.authenticate
                substituted = False

                def substitute_after_authentication(
                    authenticator: FileAuthenticator,
                    path: Path,
                    **kwargs: object,
                ) -> Artifact:
                    nonlocal substituted
                    artifact = original_authenticate(authenticator, path, **kwargs)
                    trigger = (
                        path.name == "util.py"
                        if substitution == "direct"
                        else path.name == "1d_dno_fno_jax.py"
                    )
                    if trigger and not substituted:
                        if substitution == "direct":
                            first = repository / TRAINING_IMPLEMENTATION_FILES[0][0]
                            backup = first.with_suffix(".original.py")
                            first.rename(backup)
                            first.symlink_to(backup.name)
                        else:
                            trainer_dir = repository / "train-jax-10m"
                            backup = repository / "trainer-original"
                            trainer_dir.rename(backup)
                            trainer_dir.symlink_to(
                                backup.name, target_is_directory=True
                            )
                        substituted = True
                    return artifact

                with (
                    patch.object(
                        handoff_audit.FileAuthenticator,
                        "authenticate",
                        new=substitute_after_authentication,
                    ),
                    self.assertRaisesRegex(ValueError, "symbolic link"),
                ):
                    handoff_audit._authenticate_training_implementation_record(
                        record, repository_root=repository
                    )

    def test_cli_requires_input_and_has_deterministic_default_output(self) -> None:
        args = parse_args(("--combined-summary", "combined.summary.json"))
        self.assertEqual(args.seed, 0)
        self.assertEqual(
            _default_output(args.combined_summary),
            Path("combined.training_handoff_audit.json"),
        )


if __name__ == "__main__":
    unittest.main()
