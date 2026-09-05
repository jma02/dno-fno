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
from scripts import render_paper_dataset_worst_simulations as renderer
from scripts.tests.test_render_paper_dataset_worst_simulations import (
    _write_json,
    _write_source,
)


def _combined_sources(root: Path) -> Path:
    summaries = [
        _write_source(
            root / family,
            family,
            1,
            simulation_ids=(30, 10, 20, 40),
            category=illustration.CENTRAL_VALIDATION_CATEGORIES[family],
        )
        for family in illustration.FAMILY_ORDER
    ]
    combined = root / "combined.summary.json"
    _write_json(
        combined,
        {
            "status": "complete",
            "run_summaries": list(map(str, summaries)),
            "accepted_simulations": 16,
            "rows": 16,
        },
    )
    return combined


class ParameterizedDatasetFigureTest(unittest.TestCase):
    def test_cli_selects_lower_medians_and_plots_dimensionless_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            combined = _combined_sources(root)
            with (
                mock.patch(
                    "sys.argv",
                    [
                        illustration.__file__,
                        "--combined-summary",
                        str(combined),
                        "--output-stem",
                        str(root / "figure"),
                    ],
                ),
                # Population validation has separate tests; this fixture has four small sources.
                mock.patch.object(renderer, "validate_final_paper_dataset"),
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
                self.assertEqual(simulation["simulation_id"], 20)
                self.assertEqual(simulation["selection"]["candidate_count"], 4)
                self.assertEqual(
                    simulation["selection"]["lower_median_index_zero_based"], 1
                )
                self.assertEqual(simulation["gravity"], 1.0)
                self.assertEqual(simulation["domain_length"], 2.0 * np.pi)
                self.assertEqual(simulation["stored_nx"], 256)
                self.assertEqual(simulation["owned_row"]["shard_row"], 2)
                with np.load(root / simulation["family"] / "shard.npz") as archive:
                    eta = np.asarray(archive["eta"][2], dtype=np.float64)
                    xi = np.asarray(archive["xi"][2], dtype=np.float64)
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
            combined = _combined_sources(root)
            mapping = root / "tanaka" / "map.npz"
            with np.load(mapping) as archive:
                arrays = {name: archive[name] for name in archive.files}
            np.savez(mapping, **{**arrays, "frame_index": np.ones(4, dtype=np.int32)})
            for suffix in (".pdf", ".png", ".json"):
                (root / f"figure{suffix}").write_bytes(b"existing")
            with (
                mock.patch(
                    "sys.argv",
                    [
                        illustration.__file__,
                        "--combined-summary",
                        str(combined),
                        "--output-stem",
                        str(root / "figure"),
                    ],
                ),
                mock.patch.object(renderer, "validate_final_paper_dataset"),
                self.assertRaisesRegex(ValueError, "not frame zero"),
            ):
                runpy.run_path(illustration.__file__, run_name="__main__")
            for suffix in (".pdf", ".png", ".json"):
                self.assertEqual((root / f"figure{suffix}").read_bytes(), b"existing")
            self.assertFalse(tuple(root.glob(".figure.staging-*")))
            self.assertEqual(illustration.plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
