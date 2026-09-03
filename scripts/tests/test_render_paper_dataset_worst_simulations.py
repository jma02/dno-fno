"""Focused tests for dataset-tail diagnostics and rendering."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from scripts import render_paper_dataset_worst_simulations as renderer
from scripts.render_paper_dataset_worst_simulations import (
    FINAL_PAPER_DATASET_ACCEPTED_SIMULATIONS,
    FINAL_PAPER_DATASET_RETAINED_ROWS,
    FINAL_PAPER_DATASET_SOURCE_COUNT,
    FINAL_PAPER_DATASET_SPLIT_ACCEPTED_SIMULATIONS,
    SIGN_DIFFERENCE_RELATIVE_THRESHOLD,
    LoadedTrajectory,
    TrajectoryIndex,
    _artifact_record,
    _audit_shard,
    _descending_indices,
    _render_rank_one_gif,
    _write_diagnostic_summary,
    animation_frame_indices,
    animation_record,
    atomic_output_directory,
    load_combined_summary_binding,
    load_source_summary,
    padded_animation_limits,
    validate_final_paper_dataset_counts,
    validate_scanned_population,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class CombinedSummaryTests(unittest.TestCase):
    def test_loads_source_paths_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summaries = []
            for index in range(2):
                path = root / f"source_{index}.summary.json"
                _write_json(path, {"status": "complete"})
                summaries.append(path.resolve())
            combined = root / "combined.summary.json"
            _write_json(
                combined,
                {
                    "status": "complete",
                    "run_summaries": [str(path) for path in summaries],
                    "accepted_simulations": 3,
                    "rows": 7,
                },
            )

            binding = load_combined_summary_binding(combined)

        self.assertEqual(binding.source_summary_paths, tuple(summaries))
        validate_scanned_population(
            binding,
            source_count=2,
            accepted_simulations=3,
            retained_rows=7,
        )

    def test_rejects_relative_or_repeated_source_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.summary.json"
            _write_json(source, {})
            for run_summaries, message in (
                ([source.name], "not absolute"),
                (
                    [str(source.resolve()), str(source.resolve())],
                    "repeats",
                ),
            ):
                combined = root / "combined.summary.json"
                _write_json(
                    combined,
                    {
                        "status": "complete",
                        "run_summaries": run_summaries,
                        "accepted_simulations": 0,
                        "rows": 0,
                    },
                )
                with (
                    self.subTest(message=message),
                    self.assertRaisesRegex(
                        ValueError,
                        message,
                    ),
                ):
                    load_combined_summary_binding(combined)

    def test_loads_new_generation_summary_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest = root / "paper_dataset_tanaka_train.dataset.json"
            trajectory_map = root / "paper_dataset_tanaka_train.trajectory_map.npz"
            trajectory_map.touch()
            _write_json(
                manifest,
                {
                    "trajectory_map_npz": trajectory_map.name,
                    "dataset_shards": [],
                    "n_accepted_trajectories": 0,
                    "n_accepted_rows": 0,
                },
            )
            summary = root / "paper_dataset_tanaka_train.summary.json"
            _write_json(
                summary,
                {
                    "status": "complete",
                    "output_root": str(root),
                    "run_spec": {
                        "family_name": "tanaka",
                        "dataset_split": "train",
                    },
                    "dataset_view": {
                        "manifest": str(manifest),
                        "trajectory_map": str(trajectory_map),
                    },
                },
            )
            with mock.patch.object(
                renderer,
                "_trajectory_indices",
                return_value=(),
            ):
                source = load_source_summary(summary)

        self.assertEqual(source.family, "tanaka")
        self.assertEqual(source.split, "train")
        self.assertEqual(source.manifest_path, manifest)
        self.assertEqual(source.map_path, trajectory_map)


class DatasetContractAndPublicationTests(unittest.TestCase):
    @staticmethod
    def _family_split_counts() -> dict[tuple[str, str], int]:
        return {
            (family, split): count
            for family in ("stokes", "tanaka", "benjamin_feir", "jonswap_tma")
            for split, count in FINAL_PAPER_DATASET_SPLIT_ACCEPTED_SIMULATIONS.items()
        }

    def test_final_contract_accepts_only_exact_counts(self) -> None:
        expected = self._family_split_counts()
        validate_final_paper_dataset_counts(
            source_count=FINAL_PAPER_DATASET_SOURCE_COUNT,
            family_split_accepted_simulations=expected,
            accepted_simulations=FINAL_PAPER_DATASET_ACCEPTED_SIMULATIONS,
            retained_rows=FINAL_PAPER_DATASET_RETAINED_ROWS,
        )
        with self.assertRaisesRegex(ValueError, "contract failed"):
            validate_final_paper_dataset_counts(
                source_count=FINAL_PAPER_DATASET_SOURCE_COUNT,
                family_split_accepted_simulations=expected,
                accepted_simulations=FINAL_PAPER_DATASET_ACCEPTED_SIMULATIONS,
                retained_rows=FINAL_PAPER_DATASET_RETAINED_ROWS - 1,
            )

    def test_atomic_output_publishes_only_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requested = root / "diagnostics"
            with atomic_output_directory(requested) as (staging, final_path):
                self.assertEqual(final_path, requested.resolve())
                self.assertFalse(requested.exists())
                _write_diagnostic_summary(
                    staging / "summary.json",
                    {"status": "complete"},
                )
            self.assertTrue((requested / "summary.json").is_file())

            failed = root / "failed"
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                with atomic_output_directory(failed) as (staging, _):
                    (staging / "partial.png").write_bytes(b"partial")
                    raise RuntimeError("render failed")
            self.assertFalse(failed.exists())

    def test_artifact_record_has_path_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / ".staging"
            staging.mkdir()
            figure = staging / "figure.png"
            figure.write_bytes(b"rendered figure")
            record = _artifact_record(
                figure,
                staging_output_dir=staging,
                published_output_dir=root / "published",
            )
        self.assertEqual(record["bytes"], len(b"rendered figure"))
        self.assertTrue(str(record["path"]).endswith("published/figure.png"))


class MorphologyDiagnosticTests(unittest.TestCase):
    def test_descending_ranking_preserves_scan_order_for_ties(self) -> None:
        values = np.asarray((3.0, 3.0, 2.0, 3.0), dtype=np.float64)
        self.assertEqual(_descending_indices(values, 4), (0, 1, 3, 2))

    def test_known_oscillatory_trajectory_has_nonzero_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            nx = 256
            x = 2.0 * np.pi * np.arange(nx, dtype=np.float64) / nx
            shard_path = Path(temporary) / "shard.npz"
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
                simulation_id=17,
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
        self.assertAlmostEqual(metrics.maximum_gxi_high_band_fraction, 1.0)
        self.assertGreater(metrics.maximum_thresholded_gxi_sign_changes, 0)


class RankOneAnimationTests(unittest.TestCase):
    @staticmethod
    def _trajectory(frames: int) -> LoadedTrajectory:
        x = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        scale = np.arange(1, frames + 1, dtype=np.float64)[:, None]
        return LoadedTrajectory(
            eta=scale * np.sin(x)[None, :],
            xi=scale * np.cos(x)[None, :],
            gxi=scale * np.sin(2.0 * x)[None, :],
            depth=np.full(frames, 4.0),
            time=np.arange(frames, dtype=np.float64),
        )

    @staticmethod
    def _simulation(frames: int) -> renderer.SimulationMetrics:
        return renderer.SimulationMetrics(
            source_index=0,
            accepted_index=0,
            trajectory_index=0,
            family="stokes" if frames == 1 else "jonswap_tma",
            split="train",
            simulation_id=17,
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

    def test_frame_selection_and_limits(self) -> None:
        self.assertEqual(animation_frame_indices(1).tolist(), [0])
        indices = animation_frame_indices(200)
        self.assertEqual(indices.size, 100)
        self.assertEqual((int(indices[0]), int(indices[-1])), (0, 199))
        self.assertTrue(np.all(np.diff(indices) > 0))
        self.assertEqual(padded_animation_limits(np.zeros((1, 3))), (-0.06, 0.06))

    def test_static_and_multiframe_gifs_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for frames in (1, 4):
                trajectory = self._trajectory(frames)
                path, record = _render_rank_one_gif(
                    self._simulation(frames),
                    trajectory,
                    "Accepted trajectory",
                    root / f"simulation_{frames}.gif",
                )
                self.assertEqual(record, animation_record(trajectory))
                with Image.open(path) as image:
                    self.assertEqual(image.size, renderer.GIF_DIMENSIONS)
                    self.assertEqual(int(getattr(image, "n_frames", 1)), frames)


if __name__ == "__main__":
    unittest.main()
