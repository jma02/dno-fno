from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import build_parameterized_dataset_simulation_figure as illustration
from scripts.tests.test_render_paper_dataset_worst_simulations import (
    _write_dataset,
)


def _illustration_dataset(root: Path) -> Path:
    dataset = _write_dataset(
        root,
        tuple((family, 1) for family in illustration.FAMILY_ORDER),
        accepted_attempts=(10, 20, 30, 40),
        categories=illustration.CENTRAL_VALIDATION_CATEGORIES,
    )
    # Put complete simulations out of ID order to exercise the median-ID rule.
    order = np.concatenate([np.asarray((2, 0, 1, 3)) + 4 * i for i in range(4)])
    for path in dataset.glob("*.npy"):
        if path.stem != "x":
            np.save(path, np.load(path)[order])
    return dataset


class ParameterizedDatasetFigureTest(unittest.TestCase):
    def test_cli_selects_lower_medians_and_plots_dimensionless_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = _illustration_dataset(root)
            with (
                mock.patch(
                    "sys.argv",
                    [
                        illustration.__file__,
                        "--dataset",
                        str(dataset),
                        "--output-stem",
                        str(root / "figure"),
                    ],
                ),
                redirect_stdout(io.StringIO()),
            ):
                state = runpy.run_path(illustration.__file__, run_name="__main__")
            record = json.loads((root / "figure.json").read_text())
            self.assertTrue((root / "figure.pdf").is_file())
            self.assertTrue((root / "figure.png").is_file())
            self.assertEqual(
                [simulation["family"] for simulation in record["simulations"]],
                list(illustration.FAMILY_ORDER),
            )
            for row, simulation in enumerate(record["simulations"]):
                self.assertEqual(simulation["simulation_id"], row * 4 + 1)
                self.assertEqual(simulation["selection"]["candidate_count"], 4)
                self.assertEqual(
                    simulation["selection"]["lower_median_index_zero_based"], 1
                )
                self.assertEqual(simulation["gravity"], 1.0)
                self.assertEqual(simulation["domain_length"], 2.0 * np.pi)
                self.assertEqual(simulation["stored_nx"], 256)
                self.assertEqual(simulation["dataset_row"], row * 4 + 2)
                eta = np.asarray(
                    np.load(dataset / "eta.npy")[row * 4 + 2], dtype=np.float64
                )
                xi = np.asarray(
                    np.load(dataset / "xi.npy")[row * 4 + 2], dtype=np.float64
                )
                elevation = state["axes"][row, 0].lines[0]
                potential = state["axes"][row, 1].lines[0]
                np.testing.assert_array_equal(
                    elevation.get_xdata(), np.arange(256) / 256
                )
                np.testing.assert_array_equal(elevation.get_ydata(), eta / 8.0)
                np.testing.assert_array_equal(
                    potential.get_ydata(), xi / (8.0 * np.sqrt(8.0))
                )
            for artifact in record["artifacts"].values():
                self.assertEqual(
                    Path(artifact["path"]).stat().st_size, artifact["bytes"]
                )
            self.assertFalse(tuple(root.glob(".figure.staging-*")))

    def test_invalid_first_frame_does_not_publish_or_replace_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = _illustration_dataset(root)
            frames = np.load(dataset / "frame_index.npy")
            frames[4:8] = 1
            np.save(dataset / "frame_index.npy", frames)
            for suffix in (".pdf", ".png", ".json"):
                (root / f"figure{suffix}").write_bytes(b"existing")
            with (
                mock.patch(
                    "sys.argv",
                    [
                        illustration.__file__,
                        "--dataset",
                        str(dataset),
                        "--output-stem",
                        str(root / "figure"),
                    ],
                ),
                self.assertRaisesRegex(ValueError, "frame indices"),
            ):
                runpy.run_path(illustration.__file__, run_name="__main__")
            for suffix in (".pdf", ".png", ".json"):
                self.assertEqual((root / f"figure{suffix}").read_bytes(), b"existing")
            self.assertFalse(tuple(root.glob(".figure.staging-*")))
            self.assertEqual(illustration.plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
