from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.build_parameterized_dataset_case_figure import (
    CENTRAL_VALIDATION_CATEGORIES,
    FAMILY_ORDER,
    DimensionlessProfile,
    Illustration,
    SelectedTrajectory,
    SourceDetails,
    dimensionless_profile,
    publish_outputs,
    select_validation_trajectories,
)
from scripts.render_paper_dataset_worst_cases import (
    CombinedSummaryBinding,
    DatasetSource,
    TrajectoryIndex,
)


def _source(family: str, root: Path) -> DatasetSource:
    return DatasetSource(
        root=root,
        family=family,
        split="validation",
        summary_path=root / f"paper_dataset_{family}_validation.summary.json",
        manifest_path=root / "dataset.json",
        map_path=root / "map.npz",
        shard_paths={0: root / "shard.npz"},
        trajectories=tuple(
            TrajectoryIndex(
                accepted_index=index,
                trajectory_index=index,
                case_id=case_id,
                category=CENTRAL_VALIDATION_CATEGORIES[family],
                shard_index=0,
                first_shard_row=index,
                row_count=1,
            )
            for index, case_id in enumerate((30, 10, 20, 40))
        ),
    )


def _details(family: str, root: Path) -> SourceDetails:
    revisions = {"stokes": 2, "tanaka": 3, "benjamin_feir": 4, "jonswap_tma": 4}
    return SourceDetails(
        source=_source(family, root),
        revision_id=revisions[family],
        length=2.0 * np.pi,
        gravity=9.81,
        stored_nx=4,
    )


class ParameterizedDatasetFigureTest(unittest.TestCase):
    def test_dimensionless_profile(self) -> None:
        profile = dimensionless_profile(
            np.asarray([0.0, 1.0, 2.0, 3.0]),
            np.asarray([2.0, 4.0, 6.0, 8.0]),
            depth=2.0,
            gravity=4.0,
        )
        np.testing.assert_allclose(profile.x_over_length, [0.0, 0.25, 0.5, 0.75])
        np.testing.assert_allclose(profile.eta_over_depth, [0.0, 0.5, 1.0, 1.5])
        np.testing.assert_allclose(
            profile.xi_over_depth_speed,
            np.asarray([2.0, 4.0, 6.0, 8.0]) / (2.0 * np.sqrt(8.0)),
        )

    def test_selects_lower_median_case_id_per_family(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = select_validation_trajectories(
                tuple(_details(family, root / family) for family in FAMILY_ORDER)
            )
        self.assertEqual(tuple(item.trajectory.case_id for item in selected), (20,) * 4)
        self.assertEqual(tuple(item.lower_median_index for item in selected), (1,) * 4)
        self.assertEqual(tuple(item.candidate_count for item in selected), (4,) * 4)

    def test_published_sidecar_records_cases_without_file_digests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            illustrations: list[Illustration] = []
            for family in FAMILY_ORDER:
                details = _details(family, root / family)
                selected = SelectedTrajectory(
                    details=details,
                    trajectory=details.source.trajectories[0],
                    candidate_count=4,
                    lower_median_index=1,
                )
                values = np.linspace(-1.0, 1.0, 4)
                illustrations.append(
                    Illustration(
                        selected=selected,
                        eta=values,
                        xi=-values,
                        depth=1.0,
                        time=0.0,
                        frame_index=0,
                        profile=DimensionlessProfile(
                            x_over_length=np.arange(4) / 4,
                            eta_over_depth=values,
                            xi_over_depth_speed=-values,
                        ),
                    )
                )
            summary_path = root / "combined.summary.json"
            summary_path.write_text("{}\n", encoding="utf-8")
            binding = CombinedSummaryBinding(
                path=summary_path,
                source_summary_paths=(),
                expected_source_count=0,
                expected_accepted_cases=0,
                expected_retained_rows=0,
            )
            pdf, png, sidecar = publish_outputs(
                illustrations,
                output_stem=root / "figure",
                binding=binding,
            )
            record = json.loads(sidecar.read_text(encoding="utf-8"))

            self.assertTrue(pdf.is_file())
            self.assertTrue(png.is_file())
            self.assertEqual(
                [case["family"] for case in record["cases"]], list(FAMILY_ORDER)
            )
            self.assertEqual(set(record["artifacts"]), {"pdf", "png"})


if __name__ == "__main__":
    unittest.main()
