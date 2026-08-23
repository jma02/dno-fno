from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.bind_completed_stokes_revision2_audit import (
    AUDIT_SCHEMA,
    EXPECTED_ACCEPTED,
    EXPECTED_CELL_IDS,
    EXPECTED_CHUNKS,
    EXPECTED_GENERATION_SOURCE_PATHS,
    EXPECTED_RETAINED_ROWS,
    EXPECTED_SOURCE_FINGERPRINT,
    HISTORICAL_SOURCE_SNAPSHOT_MANIFEST,
    HISTORICAL_SOURCE_SNAPSHOT_ROOT,
    HISTORICAL_SOURCE_SNAPSHOTS,
    LEGACY_AUDIT_SCHEMA,
    REQUIRED_LEGACY_TRUE_CHECKS,
    REQUIRED_LEGACY_ZERO_CHECKS,
    ROOT,
    _canonical_sha256,
    _current_source_record,
    _expected_chunk_cell_counts,
    _historical_source_snapshot_record,
    _load_legacy_audit,
    _source_mappings,
    _validate_chunk_plan,
    _validate_legacy_audit,
    _validate_output_path,
    _validate_summary_cell_counts,
    _validate_view_hashes,
)
from scripts.build_paper_dataset_view import CompletedChunk
from solver.gen_data.pipeline.archive import file_sha256


def _legacy_record() -> dict[str, object]:
    return {
        "schema": LEGACY_AUDIT_SCHEMA,
        "status": "pass",
        "attempted": EXPECTED_ACCEPTED,
        "accepted": EXPECTED_ACCEPTED,
        "rejected": 0,
        "retained_rows": EXPECTED_RETAINED_ROWS,
        "checks": {
            **{name: True for name in REQUIRED_LEGACY_TRUE_CHECKS},
            **{name: 0 for name in REQUIRED_LEGACY_ZERO_CHECKS},
        },
        "run_manifest_sha256": {
            expected.label: "1" * 64 for expected in EXPECTED_CHUNKS
        },
        "run_trajectory_map_sha256": {
            expected.label: "2" * 64 for expected in EXPECTED_CHUNKS
        },
    }


def _chunks(root: Path) -> tuple[CompletedChunk, ...]:
    source_mapping = {
        path: str(HISTORICAL_SOURCE_SNAPSHOTS[path]["sha256"])
        if path in HISTORICAL_SOURCE_SNAPSHOTS
        else file_sha256(ROOT / path)
        for path in EXPECTED_GENERATION_SOURCE_PATHS
    }
    return tuple(
        CompletedChunk(
            summary_path=(root / expected.relative_summary).resolve(),
            summary_sha256="3" * 64,
            root=(root / expected.relative_summary.parent).resolve(),
            family="stokes",
            revision_id=2,
            split=expected.split,
            stream_id=expected.stream_id,
            accepted_before=expected.accepted_before,
            accepted_count=expected.accepted_count,
            accepted_after=expected.accepted_after,
            attempted_count=expected.accepted_count,
            fingerprint=f"{index + 1:064x}",
            dependency_fingerprint="4" * 64,
            execution_fingerprint="5" * 64,
            generation_compatibility_id=None,
            source_fingerprint=_canonical_sha256(source_mapping),
            source_sha256=source_mapping,
            execution_platform="cpu",
            batches=(),
        )
        for index, expected in enumerate(EXPECTED_CHUNKS)
    )


class LegacyAuditTests(unittest.TestCase):
    def test_exact_legacy_record_passes(self) -> None:
        _validate_legacy_audit(_legacy_record())

    def test_missing_proof_and_digest_are_rejected(self) -> None:
        missing_proof = _legacy_record()
        checks = dict(missing_proof["checks"])
        checks[REQUIRED_LEGACY_TRUE_CHECKS[0]] = False
        missing_proof["checks"] = checks
        with self.assertRaisesRegex(RuntimeError, "did not prove"):
            _validate_legacy_audit(missing_proof)

        missing_digest = _legacy_record()
        manifests = dict(missing_digest["run_manifest_sha256"])
        manifests.pop(EXPECTED_CHUNKS[-1].label)
        missing_digest["run_manifest_sha256"] = manifests
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            _validate_legacy_audit(missing_digest)

    def test_legacy_audit_bytes_are_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(_legacy_record()), encoding="utf-8")
            digest = file_sha256(path)
            _load_legacy_audit(path, expected_sha256=digest)
            path.write_text(path.read_text(encoding="utf-8") + " ")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 differs"):
                _load_legacy_audit(path, expected_sha256=digest)


class ChunkPlanTests(unittest.TestCase):
    def test_exact_six_chunk_layout_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _validate_chunk_plan(_chunks(Path(directory)))

    def test_gap_or_mixed_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            chunks = list(_chunks(Path(directory)))
            chunks[1] = replace(chunks[1], accepted_before=2_049)
            with self.assertRaisesRegex(RuntimeError, "wrong identity"):
                _validate_chunk_plan(chunks)

            chunks = list(_chunks(Path(directory)))
            chunks[-1] = replace(chunks[-1], source_fingerprint="8" * 64)
            with self.assertRaisesRegex(RuntimeError, "source_fingerprint"):
                _validate_chunk_plan(chunks)


class SourceClosureTests(unittest.TestCase):
    def test_exact_archive_and_all_six_source_maps_are_physically_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            chunks = _chunks(Path(directory))
        mappings = _source_mappings(chunks)
        self.assertEqual(len(mappings), 6)
        self.assertEqual(len(mappings[0]), 13)
        self.assertEqual(_canonical_sha256(mappings[0]), EXPECTED_SOURCE_FINGERPRINT)

        historical = _historical_source_snapshot_record(mappings)
        current = _current_source_record(mappings)
        self.assertEqual(historical["historical_source_count"], 3)
        self.assertEqual(current["current_source_count"], 10)
        self.assertEqual(
            set(historical["sources"]),
            set(HISTORICAL_SOURCE_SNAPSHOTS),
        )
        self.assertEqual(
            set(current["sources"]),
            EXPECTED_GENERATION_SOURCE_PATHS - set(HISTORICAL_SOURCE_SNAPSHOTS),
        )
        self.assertFalse((HISTORICAL_SOURCE_SNAPSHOT_ROOT / "__init__.py").exists())
        checksum = HISTORICAL_SOURCE_SNAPSHOT_ROOT / "SHA256SUMS"
        self.assertEqual(
            checksum.stat().st_size,
            HISTORICAL_SOURCE_SNAPSHOT_MANIFEST["bytes"],
        )
        self.assertEqual(
            file_sha256(checksum),
            HISTORICAL_SOURCE_SNAPSHOT_MANIFEST["sha256"],
        )

    def test_missing_extra_unequal_and_forged_fingerprint_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            chunks = list(_chunks(Path(directory)))

        missing = dict(chunks[0].source_sha256)
        missing.pop("solver/gen_data/pipeline/archive.py")
        chunks[0] = replace(chunks[0], source_sha256=missing)
        with self.assertRaisesRegex(ValueError, "frozen 13-path contract"):
            _source_mappings(chunks)

        with tempfile.TemporaryDirectory() as directory:
            chunks = list(_chunks(Path(directory)))
        extra = dict(chunks[0].source_sha256)
        extra["solver/gen_data/pipeline/extra.py"] = "0" * 64
        chunks[0] = replace(chunks[0], source_sha256=extra)
        with self.assertRaisesRegex(ValueError, "frozen 13-path contract"):
            _source_mappings(chunks)

        with tempfile.TemporaryDirectory() as directory:
            chunks = list(_chunks(Path(directory)))
        unequal = dict(chunks[-1].source_sha256)
        unequal["solver/gen_data/pipeline/archive.py"] = "0" * 64
        chunks[-1] = replace(
            chunks[-1],
            source_sha256=unequal,
            source_fingerprint=_canonical_sha256(unequal),
        )
        with self.assertRaisesRegex(ValueError, "exactly equal"):
            _source_mappings(chunks)

        with tempfile.TemporaryDirectory() as directory:
            chunks = list(_chunks(Path(directory)))
        chunks[-1] = replace(chunks[-1], source_fingerprint="0" * 64)
        with self.assertRaisesRegex(ValueError, "source_fingerprint"):
            _source_mappings(chunks)

    def test_current_and_historical_byte_mutations_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            chunks = _chunks(temporary_root)
            mappings = _source_mappings(chunks)
            repository = temporary_root / "repository"
            for relative in EXPECTED_GENERATION_SOURCE_PATHS - set(
                HISTORICAL_SOURCE_SNAPSHOTS
            ):
                destination = repository / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, destination)
            _current_source_record(mappings, repository_root=repository)
            changed = repository / "solver/gen_data/pipeline/archive.py"
            changed.write_bytes(changed.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                _current_source_record(mappings, repository_root=repository)

        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "snapshots"
            shutil.copytree(HISTORICAL_SOURCE_SNAPSHOT_ROOT, copied)
            with tempfile.TemporaryDirectory() as chunk_directory:
                mappings = _source_mappings(_chunks(Path(chunk_directory)))
            _historical_source_snapshot_record(mappings, snapshot_root=copied)
            production = copied / "production.py"
            production.write_bytes(production.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "snapshot bytes differ"):
                _historical_source_snapshot_record(mappings, snapshot_root=copied)

        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "snapshots"
            shutil.copytree(HISTORICAL_SOURCE_SNAPSHOT_ROOT, copied)
            with tempfile.TemporaryDirectory() as chunk_directory:
                mappings = _source_mappings(_chunks(Path(chunk_directory)))
            checksum = copied / "SHA256SUMS"
            checksum.write_bytes(checksum.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA256SUMS"):
                _historical_source_snapshot_record(mappings, snapshot_root=copied)

    def test_historical_digest_mismatch_in_only_sixth_map_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mappings = list(_source_mappings(_chunks(Path(directory))))
        sixth = dict(mappings[-1])
        sixth["solver/gen_data/pipeline/production.py"] = "0" * 64
        mappings[-1] = sixth
        with self.assertRaisesRegex(ValueError, "differs from a chunk source map"):
            _historical_source_snapshot_record(tuple(mappings))

    def test_current_source_ancestor_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            repository = temporary_root / "repository"
            for relative in EXPECTED_GENERATION_SOURCE_PATHS - set(
                HISTORICAL_SOURCE_SNAPSHOTS
            ):
                destination = repository / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, destination)
            pipeline = repository / "solver/gen_data/pipeline"
            real_pipeline = temporary_root / "real_pipeline"
            pipeline.rename(real_pipeline)
            pipeline.symlink_to(real_pipeline, target_is_directory=True)
            with tempfile.TemporaryDirectory() as chunk_directory:
                mappings = _source_mappings(_chunks(Path(chunk_directory)))
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                _current_source_record(mappings, repository_root=repository)

    def test_source_roots_and_files_must_not_use_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copied = root / "snapshots"
            shutil.copytree(HISTORICAL_SOURCE_SNAPSHOT_ROOT, copied)
            alias = root / "snapshot_alias"
            alias.symlink_to(copied, target_is_directory=True)
            mappings = _source_mappings(_chunks(root))
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                _historical_source_snapshot_record(mappings, snapshot_root=alias)

            victim = copied / "production.py"
            victim_copy = copied / "production.real.py"
            victim.rename(victim_copy)
            victim.symlink_to(victim_copy.name)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                _historical_source_snapshot_record(mappings, snapshot_root=copied)


class ViewBindingTests(unittest.TestCase):
    def _write_views(
        self, root: Path, chunks: tuple[CompletedChunk, ...]
    ) -> dict[str, object]:
        legacy = _legacy_record()
        manifests: dict[str, str] = {}
        maps: dict[str, str] = {}
        for expected, chunk in zip(EXPECTED_CHUNKS, chunks):
            chunk.root.mkdir(parents=True, exist_ok=True)
            manifest_path = chunk.root / "view.dataset.json"
            map_path = chunk.root / "view.trajectory_map.npz"
            manifest_path.write_bytes(f"manifest-{expected.label}".encode())
            map_path.write_bytes(f"map-{expected.label}".encode())
            manifests[expected.label] = file_sha256(manifest_path)
            maps[expected.label] = file_sha256(map_path)
            chunk.summary_path.write_text(
                json.dumps(
                    {
                        "counts": {
                            "attempted": expected.accepted_count,
                            "accepted": expected.accepted_count,
                            "rejected": 0,
                            "by_cell": {
                                cell_id: {
                                    "target_accepted": count,
                                    "attempted": count,
                                    "accepted": count,
                                    "rejected": 0,
                                }
                                for cell_id, count in (
                                    _expected_chunk_cell_counts(expected).items()
                                )
                            },
                        },
                        "dataset_view": {
                            "manifest": {
                                "path": manifest_path.name,
                                "sha256": manifests[expected.label],
                            },
                            "trajectory_map": {
                                "path": map_path.name,
                                "sha256": maps[expected.label],
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
        legacy["run_manifest_sha256"] = manifests
        legacy["run_trajectory_map_sha256"] = maps
        return legacy

    def test_summary_legacy_and_present_bytes_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            chunks = _chunks(root)
            legacy = self._write_views(root, chunks)
            records = _validate_view_hashes(root, chunks, legacy)
            self.assertEqual(len(records), 6)
            self.assertEqual(
                sum(int(record["retained_rows"]) for record in records),
                EXPECTED_RETAINED_ROWS,
            )

    def test_changed_bytes_and_escaping_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            chunks = _chunks(root)
            legacy = self._write_views(root, chunks)
            (chunks[0].root / "view.dataset.json").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "manifest bytes changed"):
                _validate_view_hashes(root, chunks, legacy)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            chunks = _chunks(root)
            legacy = self._write_views(root, chunks)
            summary = json.loads(chunks[0].summary_path.read_text())
            summary["dataset_view"]["manifest"]["path"] = "../../escape.json"
            chunks[0].summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "escapes"):
                _validate_view_hashes(root, chunks, legacy)

    def test_cell_count_redistribution_missing_extra_and_bool_are_rejected(
        self,
    ) -> None:
        expected = EXPECTED_CHUNKS[0]
        counts = _expected_chunk_cell_counts(expected)

        def summary() -> dict[str, object]:
            return {
                "counts": {
                    "attempted": expected.accepted_count,
                    "accepted": expected.accepted_count,
                    "rejected": 0,
                    "by_cell": {
                        cell_id: {
                            "target_accepted": count,
                            "attempted": count,
                            "accepted": count,
                            "rejected": 0,
                        }
                        for cell_id, count in counts.items()
                    },
                }
            }

        _validate_summary_cell_counts(summary(), expected=expected)

        redistributed = summary()
        by_cell = redistributed["counts"]["by_cell"]
        by_cell[EXPECTED_CELL_IDS[0]]["attempted"] += 1
        by_cell[EXPECTED_CELL_IDS[1]]["attempted"] -= 1
        with self.assertRaisesRegex(ValueError, "wrong counts"):
            _validate_summary_cell_counts(redistributed, expected=expected)

        missing = summary()
        del missing["counts"]["by_cell"][EXPECTED_CELL_IDS[-1]]
        with self.assertRaisesRegex(ValueError, "wrong by_cell keys"):
            _validate_summary_cell_counts(missing, expected=expected)

        extra = summary()
        extra["counts"]["by_cell"]["extra"] = {
            "target_accepted": 0,
            "attempted": 0,
            "accepted": 0,
            "rejected": 0,
        }
        with self.assertRaisesRegex(ValueError, "wrong by_cell keys"):
            _validate_summary_cell_counts(extra, expected=expected)

        boolean = summary()
        boolean["counts"]["by_cell"][EXPECTED_CELL_IDS[0]]["accepted"] = True
        with self.assertRaisesRegex(ValueError, "integer"):
            _validate_summary_cell_counts(boolean, expected=expected)


class OutputSafetyTests(unittest.TestCase):
    def test_only_a_prior_binding_may_be_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            legacy = root / "stokes_completion_audit.json"
            output = root / "stokes_completion_binding.json"
            _validate_output_path(output, root=root, legacy=legacy)

            output.write_text(json.dumps({"schema": "unrelated"}))
            with self.assertRaisesRegex(FileExistsError, "non-binding"):
                _validate_output_path(output, root=root, legacy=legacy)

            output.write_text(json.dumps({"schema": AUDIT_SCHEMA}))
            _validate_output_path(output, root=root, legacy=legacy)

            with self.assertRaisesRegex(ValueError, "legacy"):
                _validate_output_path(legacy, root=root, legacy=legacy)

    def test_output_symlink_and_symlinked_parent_alias_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "dataset"
            root.mkdir()
            legacy = root / "stokes_completion_audit.json"
            victim = root / "victim.json"
            victim.write_text(json.dumps({"schema": AUDIT_SCHEMA}))
            output = root / "stokes_completion_binding.json"
            output.symlink_to(victim.name)
            with self.assertRaisesRegex(FileExistsError, "symbolic-link"):
                _validate_output_path(output, root=root, legacy=legacy)
            self.assertEqual(
                json.loads(victim.read_text(encoding="utf-8"))["schema"],
                AUDIT_SCHEMA,
            )

            output.unlink()
            alias = base / "dataset_alias"
            alias.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                _validate_output_path(
                    alias / "stokes_completion_binding.json",
                    root=root,
                    legacy=legacy,
                )

    def test_output_requires_exact_release_basename_and_regular_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            legacy = root / "stokes_completion_audit.json"
            with self.assertRaisesRegex(ValueError, "exact release basename"):
                _validate_output_path(
                    root / "alternate.json",
                    root=root,
                    legacy=legacy,
                )
            output = root / "stokes_completion_binding.json"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "regular file"):
                _validate_output_path(output, root=root, legacy=legacy)


if __name__ == "__main__":
    unittest.main()
