"""Focused tests for final-dataset binding and morphology diagnostics."""

from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import render_paper_dataset_worst_cases as renderer  # noqa: E402

from scripts.render_paper_dataset_worst_cases import (  # noqa: E402
    COMBINED_PREFLIGHT_SCHEMA,
    COMBINED_SUMMARY_SCHEMA,
    FINAL_PAPER_DATASET_ACCEPTED_CASES,
    FINAL_PAPER_DATASET_RETAINED_ROWS,
    FINAL_PAPER_DATASET_SOURCE_COUNT,
    FINAL_PAPER_DATASET_SPLIT_ACCEPTED_CASES,
    SIGN_DIFFERENCE_RELATIVE_THRESHOLD,
    LoadedTrajectory,
    TrajectoryIndex,
    _render_rank_one_gif,
    _artifact_record,
    _audit_shard,
    _descending_indices,
    animation_frame_indices,
    animation_record,
    padded_animation_limits,
    _write_diagnostic_summary,
    atomic_output_directory,
    load_combined_summary_binding,
    load_source_summary,
    parse_args,
    sha256,
    validate_bound_sources,
    validate_final_paper_dataset_counts,
    validate_scanned_population,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _combined_fixture(root: Path) -> tuple[Path, tuple[Path, ...]]:
    source_summaries: list[Path] = []
    chunks: list[dict[str, object]] = []
    for index, family in enumerate(("tanaka", "benjamin_feir")):
        source_root = root / f"chunk_{index}"
        source_root.mkdir()
        summary_path = source_root / f"paper_dataset_{family}_train.summary.json"
        _write_json(summary_path, {"status": "complete", "index": index})
        source_summaries.append(summary_path.resolve())
        chunks.append(
            {
                "summary_path": str(summary_path.resolve()),
                "summary_sha256": sha256(summary_path),
            }
        )
    combined_path = root / "paper_dataset_all_splits_c16384.summary.json"
    _write_json(
        combined_path,
        {
            "schema": COMBINED_SUMMARY_SCHEMA,
            "status": "complete",
            "preflight": {
                "schema": COMBINED_PREFLIGHT_SCHEMA,
                "chunks": chunks,
                "accepted_cases_total": 3,
                "expected_rows": 7,
            },
        },
    )
    return combined_path, tuple(source_summaries)


def _source_fixture(
    root: Path,
    family: str,
    *,
    shard_batch_index: int = 1,
) -> tuple[Path, Path, Path, Path]:
    """Write a minimal schema-shaped completed source with one dense shard."""

    root.mkdir(parents=True, exist_ok=True)
    stem = f"paper_dataset_{family}_train"
    map_path = root / f"{stem}.trajectory_map.npz"
    np.savez(
        map_path,
        trajectory_accepted=np.asarray((True,), dtype=np.bool_),
        trajectory_case_id=np.asarray((17,), dtype=np.int64),
        trajectory_cell_id=np.asarray((0,), dtype=np.int32),
        trajectory_first_row=np.asarray((0,), dtype=np.int64),
        trajectory_row_count=np.asarray((2,), dtype=np.int32),
        trajectory_index=np.asarray((0, 0), dtype=np.int32),
        shard_index=np.asarray((0, 0), dtype=np.int32),
        shard_row=np.asarray((0, 1), dtype=np.int64),
    )
    shard_path = root / "shards" / family / "train" / "batch_000001.npz"
    shard_path.parent.mkdir(parents=True)
    np.savez(shard_path, marker=np.asarray((1,), dtype=np.int8))
    manifest_path = root / f"{stem}.dataset.json"
    _write_json(
        manifest_path,
        {
            "trajectory_map_npz": map_path.name,
            "trajectory_map_sha256": sha256(map_path),
            "n_accepted_trajectories": 1,
            "n_accepted_rows": 2,
            "dataset_shards": [
                {
                    "batch_index": shard_batch_index,
                    "path": str(shard_path.relative_to(root)),
                    "sha256": sha256(shard_path),
                }
            ],
        },
    )
    summary_path = root / f"{stem}.summary.json"
    _write_json(
        summary_path,
        {
            "schema": "paper_dataset_quota_summary_v1",
            "status": "complete",
            "output_root": str(root.resolve()),
            "run_spec": {"cell_codes": {"known": 0}},
            "dataset_view": {
                "manifest": {
                    "path": manifest_path.name,
                    "bytes": manifest_path.stat().st_size,
                    "sha256": sha256(manifest_path),
                },
                "trajectory_map": {
                    "path": map_path.name,
                    "bytes": map_path.stat().st_size,
                    "sha256": sha256(map_path),
                },
            },
        },
    )
    return summary_path, manifest_path, map_path, shard_path


def _combined_from_summaries(
    path: Path,
    summaries: tuple[Path, ...],
    *,
    accepted_cases: int,
    retained_rows: int,
) -> Path:
    _write_json(
        path,
        {
            "schema": COMBINED_SUMMARY_SCHEMA,
            "status": "complete",
            "preflight": {
                "schema": COMBINED_PREFLIGHT_SCHEMA,
                "chunks": [
                    {
                        "summary_path": str(summary.resolve()),
                        "summary_sha256": sha256(summary),
                    }
                    for summary in summaries
                ],
                "accepted_cases_total": accepted_cases,
                "expected_rows": retained_rows,
            },
        },
    )
    return path


class CombinedSummaryBindingTests(unittest.TestCase):
    def test_implementation_mutation_during_run_fails_precommit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            for relative, _ in renderer.IMPLEMENTATION_FILES:
                path = repository / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"# {relative}\n", encoding="utf-8")
            original = renderer.renderer_implementation_record
            captured = original(repository)
            target = repository / renderer.IMPLEMENTATION_FILES[0][0]
            target.write_text("# changed during render\n", encoding="utf-8")
            with (
                patch.object(
                    renderer,
                    "renderer_implementation_record",
                    side_effect=lambda: original(repository),
                ),
                self.assertRaisesRegex(RuntimeError, "changed during execution"),
            ):
                renderer._require_renderer_implementation_current(captured)

    def test_valid_summary_binds_exact_chunk_paths_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined_path, source_summaries = _combined_fixture(root)

            binding = load_combined_summary_binding(combined_path)

            self.assertEqual(binding.path, combined_path.resolve())
            self.assertEqual(binding.sha256, sha256(combined_path))
            self.assertEqual(binding.source_summary_paths, source_summaries)
            self.assertEqual(
                binding.source_roots,
                tuple(path.parent for path in source_summaries),
            )
            self.assertEqual(binding.expected_source_count, 2)
            self.assertEqual(binding.expected_accepted_cases, 3)
            self.assertEqual(binding.expected_retained_rows, 7)
            validate_scanned_population(
                binding,
                source_count=2,
                accepted_cases=3,
                retained_rows=7,
            )

    def test_wrong_schema_or_status_fails_closed(self) -> None:
        mutations = (
            ("schema", "wrong", "schema"),
            ("status", "incomplete", "not complete"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                combined_path, _ = _combined_fixture(Path(directory))
                record = json.loads(combined_path.read_text(encoding="utf-8"))
                record[field] = value
                _write_json(combined_path, record)

                with self.assertRaisesRegex(ValueError, message):
                    load_combined_summary_binding(combined_path)

    def test_changed_chunk_summary_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            combined_path, source_summaries = _combined_fixture(Path(directory))
            source_summaries[0].write_text("forged\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "summary hash differs"):
                load_combined_summary_binding(combined_path)

    def test_invalid_preflight_count_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            combined_path, _ = _combined_fixture(Path(directory))
            record = json.loads(combined_path.read_text(encoding="utf-8"))
            record["preflight"]["accepted_cases_total"] = -1
            _write_json(combined_path, record)

            with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                load_combined_summary_binding(combined_path)

    def test_relative_chunk_summary_path_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            combined_path, _ = _combined_fixture(Path(directory))
            record = json.loads(combined_path.read_text(encoding="utf-8"))
            record["preflight"]["chunks"][0]["summary_path"] = "relative.json"
            _write_json(combined_path, record)

            with self.assertRaisesRegex(ValueError, "not absolute"):
                load_combined_summary_binding(combined_path)

    def test_scanned_population_count_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            combined_path, _ = _combined_fixture(Path(directory))
            binding = load_combined_summary_binding(combined_path)

            for counts in ((1, 3, 7), (2, 2, 7), (2, 3, 6)):
                with (
                    self.subTest(counts=counts),
                    self.assertRaisesRegex(
                        ValueError,
                        "counts differ",
                    ),
                ):
                    validate_scanned_population(
                        binding,
                        source_count=counts[0],
                        accepted_cases=counts[1],
                        retained_rows=counts[2],
                    )

    def test_source_and_combined_summary_options_are_mutually_exclusive(self) -> None:
        argv = [
            "render_paper_dataset_worst_cases.py",
            "--source",
            "chunk",
            "--combined-summary",
            "combined.json",
            "--output-dir",
            "output",
        ]
        with (
            patch.object(sys, "argv", argv),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            parse_args()

    def test_final_contract_flag_requires_combined_summary(self) -> None:
        argv = [
            "render_paper_dataset_worst_cases.py",
            "--source",
            "chunk",
            "--output-dir",
            "output",
            "--require-final-paper-dataset",
        ]
        with (
            patch.object(sys, "argv", argv),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            parse_args()

    def test_distinct_summaries_may_share_one_source_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared_root = root / "shared"
            first, *_ = _source_fixture(shared_root, "tanaka")
            second, *_ = _source_fixture(shared_root, "benjamin_feir")
            combined_path = _combined_from_summaries(
                root / "combined.summary.json",
                (first, second),
                accepted_cases=2,
                retained_rows=4,
            )

            binding = load_combined_summary_binding(combined_path)
            sources = tuple(
                load_source_summary(path) for path in binding.source_summary_paths
            )

            self.assertEqual(binding.source_roots, (shared_root, shared_root))
            validate_bound_sources(binding, sources)
            validate_scanned_population(
                binding,
                source_count=len(sources),
                accepted_cases=sum(len(source.trajectories) for source in sources),
                retained_rows=sum(
                    trajectory.row_count
                    for source in sources
                    for trajectory in source.trajectories
                ),
            )


class SourceAuthenticationTests(unittest.TestCase):
    def test_dense_shard_ordinal_is_not_sparse_batch_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary_path, _, _, shard_path = _source_fixture(
                Path(directory),
                "tanaka",
                shard_batch_index=1,
            )

            source = load_source_summary(summary_path)

            self.assertEqual(source.shard_paths, {0: shard_path.resolve()})
            self.assertEqual(source.trajectories[0].shard_index, 0)

    def test_source_summary_schema_and_status_fail_closed(self) -> None:
        mutations = (
            ("schema", "wrong", "unknown summary schema"),
            ("status", "incomplete", "not complete"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                summary_path, *_ = _source_fixture(Path(directory), "tanaka")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary[field] = value
                _write_json(summary_path, summary)

                with self.assertRaisesRegex(RuntimeError, message):
                    load_source_summary(summary_path)

    def test_changed_manifest_bytes_fail_summary_authentication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary_path, manifest_path, _, _ = _source_fixture(
                Path(directory),
                "tanaka",
            )
            manifest_path.write_text("forged\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "manifest SHA-256 differs"):
                load_source_summary(summary_path)

    def test_manifest_map_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary_path, manifest_path, _, _ = _source_fixture(
                Path(directory),
                "tanaka",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["trajectory_map_sha256"] = "0" * 64
            _write_json(manifest_path, manifest)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["dataset_view"]["manifest"].update(
                {
                    "bytes": manifest_path.stat().st_size,
                    "sha256": sha256(manifest_path),
                }
            )
            _write_json(summary_path, summary)

            with self.assertRaisesRegex(RuntimeError, "trajectory-map SHA-256"):
                load_source_summary(summary_path)

    def test_changed_trajectory_map_bytes_fail_summary_authentication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary_path, _, map_path, _ = _source_fixture(
                Path(directory),
                "tanaka",
            )
            with map_path.open("ab") as handle:
                handle.write(b"forged")

            with self.assertRaisesRegex(ValueError, "trajectory map SHA-256 differs"):
                load_source_summary(summary_path)

    def test_changed_shard_bytes_fail_manifest_authentication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary_path, _, _, shard_path = _source_fixture(
                Path(directory),
                "tanaka",
            )
            with shard_path.open("ab") as handle:
                handle.write(b"forged")

            with self.assertRaisesRegex(RuntimeError, "shard 0 SHA-256 differs"):
                load_source_summary(summary_path)


class FinalContractAndPublicationTests(unittest.TestCase):
    @staticmethod
    def _final_family_split_counts() -> dict[tuple[str, str], int]:
        return {
            (family, split): count
            for family in (
                "stokes",
                "tanaka",
                "benjamin_feir",
                "jonswap_tma",
            )
            for split, count in FINAL_PAPER_DATASET_SPLIT_ACCEPTED_CASES.items()
        }

    def test_final_contract_accepts_only_exact_counts(self) -> None:
        expected = self._final_family_split_counts()
        validate_final_paper_dataset_counts(
            source_count=FINAL_PAPER_DATASET_SOURCE_COUNT,
            family_split_accepted_cases=expected,
            accepted_cases=FINAL_PAPER_DATASET_ACCEPTED_CASES,
            retained_rows=FINAL_PAPER_DATASET_RETAINED_ROWS,
        )

        wrong_family_counts = dict(expected)
        wrong_family_counts[("tanaka", "test")] -= 1
        mismatches = (
            {
                "source_count": FINAL_PAPER_DATASET_SOURCE_COUNT - 1,
                "family_split_accepted_cases": expected,
                "accepted_cases": FINAL_PAPER_DATASET_ACCEPTED_CASES,
                "retained_rows": FINAL_PAPER_DATASET_RETAINED_ROWS,
            },
            {
                "source_count": FINAL_PAPER_DATASET_SOURCE_COUNT,
                "family_split_accepted_cases": wrong_family_counts,
                "accepted_cases": FINAL_PAPER_DATASET_ACCEPTED_CASES - 1,
                "retained_rows": FINAL_PAPER_DATASET_RETAINED_ROWS,
            },
            {
                "source_count": FINAL_PAPER_DATASET_SOURCE_COUNT,
                "family_split_accepted_cases": expected,
                "accepted_cases": FINAL_PAPER_DATASET_ACCEPTED_CASES,
                "retained_rows": FINAL_PAPER_DATASET_RETAINED_ROWS - 1,
            },
        )
        for values in mismatches:
            with (
                self.subTest(values=values),
                self.assertRaisesRegex(
                    ValueError, "final paper-dataset contract failed"
                ),
            ):
                validate_final_paper_dataset_counts(**values)

    def test_atomic_output_publishes_complete_summary_only_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requested = root / "diagnostics"
            with atomic_output_directory(requested) as (staging, final_path):
                self.assertEqual(final_path, requested.resolve())
                self.assertFalse(requested.exists())
                _write_diagnostic_summary(
                    staging / "summary.json",
                    {"schema": "test", "status": "complete"},
                )

            self.assertEqual(
                json.loads((requested / "summary.json").read_text(encoding="utf-8"))[
                    "status"
                ],
                "complete",
            )
            self.assertEqual(tuple(root.glob(".diagnostics.staging-*")), ())

    def test_artifact_record_hashes_staged_bytes_at_published_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / ".diagnostics.staging-test"
            staging.mkdir()
            published = root / "diagnostics"
            figure = staging / "all_families_worst_overview.png"
            figure.write_bytes(b"rendered figure")

            record = _artifact_record(
                figure,
                staging_output_dir=staging,
                published_output_dir=published,
            )

            self.assertEqual(
                record,
                {
                    "path": str(
                        (published / "all_families_worst_overview.png").resolve()
                    ),
                    "bytes": len(b"rendered figure"),
                    "sha256": sha256(figure),
                },
            )

    def test_atomic_output_failure_leaves_final_path_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requested = root / "diagnostics"
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                with atomic_output_directory(requested) as (staging, _):
                    (staging / "partial.png").write_bytes(b"partial")
                    raise RuntimeError("render failed")

            self.assertFalse(requested.exists())
            self.assertEqual(tuple(root.glob(".diagnostics.staging-*")), ())

    def test_atomic_output_refuses_preexisting_requested_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            requested = Path(directory) / "diagnostics"
            requested.mkdir()

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                with atomic_output_directory(requested):
                    self.fail("preexisting output should not yield a staging path")

    def test_summary_writer_rejects_noncomplete_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            with self.assertRaisesRegex(ValueError, "status must be complete"):
                _write_diagnostic_summary(
                    path,
                    {"schema": "test", "status": "incomplete"},
                )
            self.assertFalse(path.exists())


class MorphologyDiagnosticTests(unittest.TestCase):
    def test_descending_ranking_preserves_scan_order_for_exact_ties(self) -> None:
        values = np.asarray((3.0, 3.0, 2.0, 3.0), dtype=np.float64)

        self.assertEqual(_descending_indices(values, 4), (0, 1, 3, 2))
        self.assertEqual(_descending_indices(values, 2), (0, 1))

    def test_known_oscillatory_trajectory_has_nonzero_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            nx = 256
            x = 2.0 * np.pi * np.arange(nx, dtype=np.float64) / nx
            shard_path = Path(directory) / "shard.npz"
            np.savez(
                shard_path,
                eta=np.stack((np.sin(x), 2.0 * np.sin(x))).astype(np.float32),
                xi=np.zeros((2, nx), dtype=np.float32),
                gxi=np.stack((np.sin(100.0 * x), np.sin(100.0 * x))).astype(np.float32),
                depth=np.asarray((3.0, 3.0), dtype=np.float64),
                time=np.asarray((0.0, 1.0), dtype=np.float64),
            )
            trajectory = TrajectoryIndex(
                accepted_index=0,
                trajectory_index=0,
                case_id=17,
                category="known",
                shard_index=0,
                first_shard_row=0,
                row_count=2,
            )

            (metrics,) = _audit_shard(
                0,
                "tanaka",
                "train",
                0,
                shard_path,
                (trajectory,),
                2,
            )

            self.assertEqual(SIGN_DIFFERENCE_RELATIVE_THRESHOLD, 0.03)
            self.assertAlmostEqual(metrics.maximum_eta_slope, 2.0, places=5)
            self.assertAlmostEqual(
                metrics.maximum_gxi_high_band_fraction,
                1.0,
                places=12,
            )
            self.assertGreater(metrics.maximum_thresholded_gxi_sign_changes, 0)
            self.assertAlmostEqual(
                metrics.maximum_relative_stored_band_quadratic_energy_drift,
                3.0,
                places=5,
            )


class RankOneAnimationTests(unittest.TestCase):
    def _trajectory(self, frames: int) -> LoadedTrajectory:
        x = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        scale = np.arange(1, frames + 1, dtype=np.float64)[:, None]
        return LoadedTrajectory(
            eta=scale * np.sin(x)[None, :],
            xi=scale * np.cos(x)[None, :],
            gxi=scale * np.sin(2.0 * x)[None, :],
            depth=np.full(frames, 4.0),
            time=np.arange(frames, dtype=np.float64),
        )

    def _case(self, frames: int) -> renderer.CaseMetrics:
        return renderer.CaseMetrics(
            source_index=0,
            accepted_index=0,
            trajectory_index=0,
            family="stokes" if frames == 1 else "jonswap_tma",
            split="train",
            case_id=17,
            category="finite",
            shard_index=0,
            first_shard_row=0,
            row_count=frames,
            depth=4.0,
            all_frames_finite=True,
            constant_depth=True,
            ordered_time=True,
            minimum_water_column=3.0,
            minimum_water_fraction=0.75,
            maximum_eta_slope=1.0,
            maximum_eta_slope_frame=0,
            maximum_gxi_high_band_fraction=0.0,
            maximum_gxi_high_band_fraction_frame=0,
            maximum_thresholded_gxi_sign_changes=0,
            maximum_thresholded_gxi_sign_changes_frame=0,
            maximum_relative_stored_band_quadratic_energy_drift=0.0,
            maximum_relative_stored_band_quadratic_energy_drift_frame=0,
        )

    def test_frame_selection_is_inclusive_unique_and_capped(self) -> None:
        self.assertEqual(animation_frame_indices(1).tolist(), [0])
        indices = animation_frame_indices(200)
        self.assertEqual(indices.size, 100)
        self.assertEqual((int(indices[0]), int(indices[-1])), (0, 199))
        self.assertTrue(np.all(np.diff(indices) > 0))
        with self.assertRaisesRegex(ValueError, "must be positive"):
            animation_frame_indices(0)

    def test_limits_use_all_stored_frames_and_handle_constant_fields(self) -> None:
        values = np.asarray(((0.0, 1.0), (100.0, 101.0)))
        lower, upper = padded_animation_limits(values)
        self.assertLess(lower, 0.0)
        self.assertGreater(upper, 101.0)
        self.assertEqual(padded_animation_limits(np.zeros((1, 3))), (-0.06, 0.06))
        with self.assertRaisesRegex(ValueError, "finite"):
            padded_animation_limits(np.asarray([np.nan]))

    def test_static_and_multiframe_gifs_decode_to_contract(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for frames in (1, 4):
                trajectory = self._trajectory(frames)
                path = root / f"case_{frames}.gif"
                output, record = _render_rank_one_gif(
                    self._case(frames),
                    trajectory,
                    "Accepted trajectory",
                    path,
                )
                self.assertEqual(output, path)
                self.assertEqual(record, animation_record(trajectory))
                with Image.open(path) as image:
                    self.assertEqual(image.format, "GIF")
                    self.assertEqual(image.size, renderer.GIF_DIMENSIONS)
                    self.assertEqual(image.n_frames, frames)
                    self.assertEqual(image.info["loop"], 0)
                    for index in range(image.n_frames):
                        image.seek(index)
                        image.load()
                        self.assertEqual(image.info["duration"], 250)


if __name__ == "__main__":
    unittest.main()
