"""Whole-pipeline tests for accepted dataset diagnostics and publication."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts import render_paper_dataset_worst_simulations as renderer


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_source(
    root: Path,
    family: str,
    frames: int,
    *,
    simulation_ids: tuple[int, ...] = (17,),
    category: str = "known",
) -> Path:
    root.mkdir()
    count = len(simulation_ids)
    rows = count * frames
    x = 2.0 * np.pi * np.arange(256) / 256
    scale = np.tile(np.arange(1, frames + 1), count)[:, None]
    np.savez(
        root / "shard.npz",
        eta=(scale * np.sin(x)).astype(np.float32),
        xi=(2.0 * scale * np.cos(x)).astype(np.float32),
        gxi=np.tile(np.sin(100.0 * x), (rows, 1)).astype(np.float32),
        depth=np.full(rows, 8.0),
        time=np.tile(np.arange(frames, dtype=np.float64), count),
    )
    family_id = tuple(renderer.FAMILY_LABELS).index(family) + 1
    np.savez(
        root / "map.npz",
        trajectory_accepted=np.ones(count, dtype=np.bool_),
        trajectory_simulation_id=np.asarray(simulation_ids),
        trajectory_parameter_group_id=np.full(count, category),
        trajectory_first_row=np.arange(count) * frames,
        trajectory_row_count=np.full(count, frames),
        trajectory_family_id=np.full(count, family_id),
        trajectory_dataset_split=np.full(count, "validation"),
        trajectory_index=np.repeat(np.arange(count), frames),
        shard_index=np.zeros(rows, dtype=np.int32),
        shard_row=np.arange(rows),
        frame_index=np.tile(np.arange(frames), count),
    )
    _write_json(
        root / "dataset.json",
        {
            "trajectory_map_npz": "map.npz",
            "dataset_shards": [{"path": "shard.npz"}],
            "n_accepted_trajectories": count,
            "n_accepted_rows": rows,
            "grid": {"length": 2.0 * np.pi, "nx": 256},
        },
    )
    summary = root / f"paper_dataset_{family}_validation.summary.json"
    _write_json(
        summary,
        {
            "status": "complete",
            "output_root": str(root),
            "run_spec": {"family_name": family, "dataset_split": "validation"},
            "dataset_view": {"manifest": "dataset.json", "trajectory_map": "map.npz"},
        },
    )
    return summary


class DatasetRenderingTests(unittest.TestCase):
    def test_combined_summary_binds_source_paths_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = _write_source(root / "stokes", "stokes", 1)
            source = renderer.load_source_summary(source_path)
            combined = root / "combined.summary.json"
            record = {
                "status": "complete",
                "run_summaries": [str(source_path)],
                "accepted_simulations": 1,
                "rows": 1,
            }
            _write_json(combined, record)
            binding = renderer.load_combined_summary_binding(combined)
            renderer.validate_bound_sources(binding, (source,))
            renderer.validate_scanned_population(
                binding, source_count=1, accepted_simulations=1, retained_rows=1
            )
            with self.assertRaisesRegex(ValueError, "counts differ"):
                renderer.validate_scanned_population(
                    binding, source_count=1, accepted_simulations=1, retained_rows=2
                )
            for paths, message in (
                ([source_path.name], "not absolute"),
                ([str(source_path)] * 2, "repeats"),
            ):
                _write_json(combined, {**record, "run_summaries": paths})
                with (
                    self.subTest(paths=paths),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    renderer.load_combined_summary_binding(combined)

    def test_final_population_is_recovered_from_source_maps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = renderer.load_source_summary(
                _write_source(Path(temporary) / "stokes", "stokes", 1)
            )
            sources = tuple(
                source._replace(
                    family=family, split=split, trajectories=source.trajectories * count
                )
                for family in renderer.FAMILY_LABELS
                for split, count in renderer.FINAL_PAPER_DATASET_SPLIT_ACCEPTED_SIMULATIONS.items()
            )
            renderer.validate_final_paper_dataset(
                sources, retained_rows=renderer.FINAL_PAPER_DATASET_RETAINED_ROWS
            )
            for population, rows in (
                (sources[:-1], renderer.FINAL_PAPER_DATASET_RETAINED_ROWS),
                (sources, renderer.FINAL_PAPER_DATASET_RETAINED_ROWS - 1),
            ):
                with self.assertRaisesRegex(ValueError, "contract failed"):
                    renderer.validate_final_paper_dataset(
                        population, retained_rows=rows
                    )

    def test_descending_ranking_preserves_scan_order_for_ties(self) -> None:
        self.assertEqual(
            renderer._descending_indices(np.asarray((3.0, 3.0, 2.0, 3.0)), 4),
            (0, 1, 3, 2),
        )

    def test_cli_ranks_known_waves_and_decodes_static_and_multiframe_gifs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for family, frames in (("stokes", 1), ("tanaka", 4)):
                _write_source(root / family, family, frames)
            output = root / "diagnostics"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(renderer.__file__)),
                    "--source",
                    str(root / "stokes"),
                    "--source",
                    str(root / "tanaka"),
                    "--output-dir",
                    str(output),
                    "--workers",
                    "1",
                    "--top-count",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "MPLCONFIGDIR": str(root / "matplotlib")},
                timeout=120,
            )
            result = json.loads(completed.stdout)
            record = json.loads((output / "summary.json").read_text())
            self.assertEqual(result["accepted_simulations"], 2)
            self.assertEqual(record["population"]["retained_rows"], 5)
            for family, frames in (("stokes", 1), ("tanaka", 4)):
                metrics = record["families"][family]["rankings"]["eta_slope"][0]
                self.assertAlmostEqual(metrics["maximum_eta_slope"], frames, places=5)
                self.assertAlmostEqual(metrics["maximum_gxi_high_band_fraction"], 1.0)
                self.assertGreater(metrics["maximum_thresholded_gxi_sign_changes"], 0)
                gif = output / f"{family}_worst_eta_slope.gif"
                animation = record["animations"][gif.name]
                self.assertEqual(animation["frame_indices"], list(range(frames)))
                self.assertAlmostEqual(
                    animation["field_y_limits"]["eta"][0], -1.12 * frames
                )
                self.assertAlmostEqual(
                    animation["field_y_limits"]["eta"][1], 1.12 * frames
                )
                with Image.open(gif) as image:
                    self.assertEqual(image.size, renderer.GIF_DIMENSIONS)
                    self.assertEqual(int(getattr(image, "n_frames", 1)), frames)
                    self.assertEqual(image.info["loop"], 0)
                    for index in range(frames):
                        image.seek(index)
                        image.load()
                        self.assertEqual(image.info["duration"], 250)
            for artifact in record["artifacts"].values():
                path = Path(artifact["path"])
                self.assertTrue(path.is_relative_to(output))
                self.assertEqual(path.stat().st_size, artifact["bytes"])
            self.assertFalse(tuple(root.glob(".diagnostics.staging-*")))

    def test_failed_audit_removes_staging_and_existing_outputs_are_preserved(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_source(root / "tanaka", "tanaka", 2)
            shard = root / "tanaka" / "shard.npz"
            with np.load(shard) as archive:
                arrays = {name: archive[name] for name in archive.files}
            np.savez(shard, **{**arrays, "time": np.zeros(2)})
            output = root / "diagnostics"
            command = [
                sys.executable,
                str(Path(renderer.__file__)),
                "--source",
                str(root / "tanaka"),
                "--output-dir",
                str(output),
                "--workers",
                "1",
            ]
            failed = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("failed a hard audit", failed.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(tuple(root.glob(".diagnostics.staging-*")))
            output.mkdir()
            sentinel = output / "existing.txt"
            sentinel.write_text("keep")
            failed = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertIn("output path already exists", failed.stderr)
            self.assertEqual(sentinel.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
