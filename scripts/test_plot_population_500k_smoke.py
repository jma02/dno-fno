"""Focused tests for the 500,000-case population-smoke plots."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.plot_population_500k_smoke import (
    PopulationData,
    _require_current_revision,
    _simplex_coordinates,
    balanced_expected_counts,
    dkw_half_width,
    empirical_cdf,
    load_population,
    plot_overview,
    rank_medoid_index,
    select_role_cases,
    shallow_chi_cdf,
    uniform_discrepancy,
)
from solver.gen_data.pipeline.production import PAPER_CORPUS_REVISION_ID


def _common_arrays(count: int, cell_code: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "attempt_index": np.arange(count, dtype=np.int64),
        "case_id": np.arange(1000, 1000 + count, dtype=np.int64),
        "cell_code": np.asarray(cell_code, dtype=np.int16),
    }


class MathematicalHelpersTest(unittest.TestCase):
    def test_field_replay_requires_the_archive_generator_revision(self) -> None:
        _require_current_revision(
            {"configuration": {"revision_id": PAPER_CORPUS_REVISION_ID}}
        )
        with self.assertRaisesRegex(RuntimeError, "requires the source"):
            _require_current_revision(
                {
                    "configuration": {
                        "revision_id": PAPER_CORPUS_REVISION_ID - 1
                    }
                }
            )

    def test_ordered_quotient_remainder_counts_are_exact(self) -> None:
        np.testing.assert_array_equal(
            balanced_expected_counts(125_000, 66),
            np.asarray([1894] * 62 + [1893] * 4, dtype=np.int64),
        )
        self.assertEqual(
            int(np.sum(balanced_expected_counts(125_000, 27))),
            125_000,
        )

    def test_empirical_cdf_and_uniform_discrepancy(self) -> None:
        values = np.asarray([0.625, 0.125, 0.875, 0.375])
        ordered, probability = empirical_cdf(values)
        np.testing.assert_allclose(
            ordered,
            np.asarray([0.125, 0.375, 0.625, 0.875]),
        )
        np.testing.assert_allclose(
            probability,
            np.asarray([0.25, 0.50, 0.75, 1.00]),
        )
        self.assertAlmostEqual(uniform_discrepancy(values), 0.125)
        self.assertGreater(dkw_half_width(4), 0.0)

    def test_rank_medoid_breaks_ties_by_attempt_index(self) -> None:
        features = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ]
        )
        attempts = np.asarray([40, 30, 20, 10], dtype=np.int64)
        # The middle two have equal rank distance; attempt 20 wins.
        self.assertEqual(rank_medoid_index(features, attempts), 2)

    def test_simplex_mapping_preserves_vertices(self) -> None:
        x, y = _simplex_coordinates(np.eye(3, dtype=np.float64))
        np.testing.assert_allclose(x, np.asarray([0.0, 1.0, 0.5]))
        np.testing.assert_allclose(
            y,
            np.asarray([0.0, 0.0, np.sqrt(3.0) / 2.0]),
        )

    def test_shallow_peak_depth_cdf_has_correct_endpoints(self) -> None:
        transition = 0.15 / 0.16
        chi = np.asarray([0.2, transition, 1.0, 1.5])
        cdf = shallow_chi_cdf(chi)
        self.assertAlmostEqual(float(cdf[0]), 0.0)
        self.assertAlmostEqual(float(cdf[-1]), 1.0)
        self.assertTrue(np.all(np.diff(cdf) > 0.0))


class ArchiveAndSelectionTest(unittest.TestCase):
    def test_loads_tiny_stokes_archive_with_declared_schema(self) -> None:
        count = 4
        arrays = {
            **_common_arrays(count, np.arange(count, dtype=np.int16)),
            "carrier_mode": np.asarray([14, 15, 3, 4]),
            "depth": np.asarray([0.1, 0.2, 5.0, 6.0]),
            "phase": np.linspace(0.1, 0.4, count),
            "amplitude": np.full(count, 0.002),
            "steepness": np.asarray([0.028, 0.03, 0.006, 0.008]),
            "depth_wavenumber": np.asarray([1.4, 3.0, 15.0, 24.0]),
            "ursell_upper_bound": np.asarray([10.0, 20.0, np.nan, np.nan]),
            "support_redraw_count": np.zeros(count, dtype=np.int16),
            "depth_draw_lower": np.full(count, 0.05),
            "depth_draw_upper": np.full(count, 10.0),
            "amplitude_draw_lower": np.full(count, 0.001),
            "amplitude_draw_upper": np.full(count, 0.003),
            "attempt_owner": np.arange(count, dtype=np.int64),
            "attempt_amplitude_unit": np.linspace(0.1, 0.9, count),
            "attempt_ursell_ratio": np.asarray([0.4, 0.8, np.nan, np.nan]),
            "attempt_accepted": np.ones(count, dtype=np.bool_),
        }
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            np.savez(directory / "stokes.npz", **arrays)
            summary = {
                "families": {
                    "stokes": {
                        "cell_order": [
                            "finite_low",
                            "finite_moderate",
                            "deep_low",
                            "deep_moderate",
                        ],
                        "stream_id": 7,
                        "npz_path": "stokes.npz",
                    }
                }
            }
            loaded = load_population(directory, summary, "stokes")
            self.assertEqual(loaded.count, count)
            self.assertEqual(loaded.stream_id, 7)
            np.testing.assert_array_equal(
                loaded.cell_counts,
                np.ones(4, dtype=np.int64),
            )

    def test_stokes_role_selection_is_unique_and_parameter_only(self) -> None:
        count = 8
        arrays = {
            **_common_arrays(
                count,
                np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int16),
            ),
            "depth": np.asarray([0.1, 0.2, 0.1, 0.2, 5.1, 6.0, 5.2, 8.0]),
            "steepness": np.asarray(
                [0.01, 0.02, 0.03, 0.14, 0.01, 0.02, 0.03, 0.04]
            ),
            "carrier_mode": np.asarray([14, 15, 16, 17, 3, 4, 5, 6]),
            "ursell_upper_bound": np.asarray(
                [10.0, 25.9, 20.0, 25.0, np.nan, np.nan, np.nan, np.nan]
            ),
            "depth_wavenumber": np.asarray(
                [1.4, 3.0, 1.6, 3.4, 5.1, 6.0, 5.2, 8.0]
            ),
            "support_redraw_count": np.asarray([0, 1, 0, 3, 0, 0, 0, 0]),
        }
        data = PopulationData(
            family="stokes",
            path=Path("unused.npz"),
            arrays=arrays,
            cell_order=(
                "finite_low",
                "finite_moderate",
                "deep_low",
                "deep_moderate",
            ),
            stream_id=0,
        )
        selections = select_role_cases(data)
        self.assertEqual(len(selections), 5)
        self.assertEqual(
            {selection.role for selection in selections},
            {
                "rank medoid",
                "maximum accepted finite Ursell ratio",
                "largest same-cell amplitude redraw count",
                "largest carrier steepness",
                "closest deep-water boundary",
            },
        )

    def test_overview_writes_png_and_pdf_from_small_populations(self) -> None:
        stokes = PopulationData(
            "stokes",
            Path("unused"),
            _common_arrays(4, np.asarray([0, 1, 2, 3])),
            ("a", "b", "c", "d"),
            0,
        )
        tanaka = PopulationData(
            "tanaka",
            Path("unused"),
            _common_arrays(4, np.asarray([0, 1, 2, 3])),
            ("a", "b", "c", "d"),
            0,
        )
        bf_arrays = {
            **_common_arrays(4, np.asarray([0, 0, 1, 1])),
            "carrier_mode": np.asarray([4, 4, 5, 5]),
            "sideband_offset": np.asarray([1, 1, 1, 1]),
        }
        bf = PopulationData(
            "benjamin_feir",
            Path("unused"),
            bf_arrays,
            ("pair_1", "pair_2"),
            0,
        )
        jonswap = PopulationData(
            "jonswap_tma",
            Path("unused"),
            _common_arrays(3, np.asarray([0, 1, 2])),
            ("shallow", "finite", "deep"),
            0,
        )
        with tempfile.TemporaryDirectory() as raw_directory:
            paths, record = plot_overview(
                {
                    "stokes": stokes,
                    "tanaka": tanaka,
                    "benjamin_feir": bf,
                    "jonswap_tma": jonswap,
                },
                Path(raw_directory),
            )
            self.assertTrue(all(path.exists() for path in paths))
            self.assertEqual(record["total_specifications"], 15)


if __name__ == "__main__":
    unittest.main()
