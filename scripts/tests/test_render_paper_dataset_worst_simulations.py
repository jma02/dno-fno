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
from solver.gen_data.pipeline.batch_storage import save_completed_batch
from solver.gen_data.pipeline.build_dataset import build_dataset
from solver.gen_data.pipeline.types import (
    PhysicalFamilyId,
    SimulationRows,
)


def _write_dataset(
    root: Path,
    families: tuple[tuple[str, int], ...],
    *,
    accepted_attempts: tuple[int, ...] = (17,),
    categories: dict[str, str] | None = None,
) -> Path:
    batches = []
    x = 2.0 * np.pi * np.arange(256) / 256
    for batch_index, (family, frames) in enumerate(families):
        scale = np.arange(1, frames + 1, dtype=np.float64)[:, None]
        rows = SimulationRows(
            eta=scale * np.sin(x),
            xi=2.0 * scale * np.cos(x),
            gxi=np.tile(np.sin(100.0 * x), (frames, 1)),
            depth=8.0,
            time=np.arange(frames, dtype=np.float64),
        )
        batch = root / f"batch_{batch_index:02d}.npz"
        attempts = tuple(
            rows if index in accepted_attempts else None
            for index in range(max(accepted_attempts) + 1)
        )
        save_completed_batch(
            batch,
            ((categories or {}).get(family, "known"),) * len(attempts),
            attempts,
            family_id=PhysicalFamilyId(tuple(renderer.FAMILY_LABELS).index(family) + 1),
            seed=42,
        )
        batches.append(batch)
    return build_dataset(
        root / "dataset", batches, validation_fraction=1.0, test_fraction=0.0
    )


class DatasetRenderingTests(unittest.TestCase):
    def test_groups_preserve_simulation_ids_and_row_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = _write_dataset(Path(temporary), (("stokes", 1), ("tanaka", 4)))
            groups = renderer.load_dataset_groups(dataset)
            self.assertEqual([group.family for group in groups], ["stokes", "tanaka"])
            self.assertEqual([group.split for group in groups], ["validation"] * 2)
            for simulation_id, (group, first, count) in enumerate(
                zip(groups, (0, 1), (1, 4), strict=True)
            ):
                self.assertEqual(
                    group.trajectories,
                    (
                        renderer.TrajectoryIndex(
                            0, simulation_id, "known", first, count
                        ),
                    ),
                )
            np.save(dataset / "frame_index.npy", np.zeros(5, dtype=np.int32))
            with self.assertRaisesRegex(ValueError, "frame indices"):
                renderer.load_dataset_groups(dataset)

    def test_final_population_is_recovered_from_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = renderer.load_dataset_groups(
                _write_dataset(Path(temporary), (("stokes", 1),))
            )[0]
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
            dataset = _write_dataset(root, (("stokes", 1), ("tanaka", 4)))
            output = root / "diagnostics"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(renderer.__file__)),
                    "--dataset",
                    str(dataset),
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
            dataset = _write_dataset(root, (("tanaka", 2),))
            np.save(dataset / "time.npy", np.zeros(2))
            output = root / "diagnostics"
            command = [
                sys.executable,
                str(Path(renderer.__file__)),
                "--dataset",
                str(dataset),
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
