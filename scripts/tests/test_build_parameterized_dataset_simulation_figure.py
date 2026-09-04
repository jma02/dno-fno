from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.build_parameterized_dataset_simulation_figure import (
    CENTRAL_VALIDATION_CATEGORIES,
    FAMILY_ORDER,
    DimensionlessProfile,
    Illustration,
    SelectedTrajectory,
    SourceDetails,
    dimensionless_profile,
    load_source_details,
    publish_outputs,
    select_validation_trajectories,
)
from scripts.render_paper_dataset_worst_simulations import (
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
                simulation_id=simulation_id,
                category=CENTRAL_VALIDATION_CATEGORIES[family],
                shard_index=0,
                first_shard_row=index,
                row_count=1,
            )
            for index, simulation_id in enumerate((30, 10, 20, 40))
        ),
    )


def _details(family: str, root: Path) -> SourceDetails:
    return SourceDetails(
        source=_source(family, root),
        length=2.0 * np.pi,
        gravity=9.81,
        stored_nx=4,
    )


class ParameterizedDatasetFigureTest(unittest.TestCase):
    def test_loads_scales_from_current_manifest_without_summary_configuration(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for family_id, family in enumerate(FAMILY_ORDER, start=1):
                with self.subTest(family=family):
                    source = _source(family, root)
                    source.summary_path.write_text(
                        json.dumps(
                            {
                                "run_spec": {
                                    "family_name": family,
                                    "dataset_split": "validation",
                                },
                                "dataset_view": {
                                    "manifest": source.manifest_path.name,
                                    "trajectory_map": source.map_path.name,
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    source.manifest_path.write_text(
                        json.dumps({"grid": {"length": 2.0 * np.pi, "nx": 4}}),
                        encoding="utf-8",
                    )
                    np.savez(
                        source.map_path,
                        trajectory_accepted=np.asarray([True]),
                        trajectory_family_id=np.asarray([family_id]),
                        trajectory_dataset_split=np.asarray(["validation"]),
                    )

                    details = load_source_details(source)

                    self.assertEqual(details.length, 2.0 * np.pi)
                    self.assertEqual(details.stored_nx, 4)
                    self.assertEqual(details.gravity, 1.0)

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

    def test_selects_lower_median_simulation_id_per_family(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = select_validation_trajectories(
                tuple(_details(family, root / family) for family in FAMILY_ORDER)
            )
        self.assertEqual(
            tuple(item.trajectory.simulation_id for item in selected), (20,) * 4
        )
        self.assertEqual(tuple(item.lower_median_index for item in selected), (1,) * 4)
        self.assertEqual(tuple(item.candidate_count for item in selected), (4,) * 4)

    def test_published_sidecar_records_simulations_without_file_digests(self) -> None:
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
                expected_accepted_simulations=0,
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
                [simulation["family"] for simulation in record["simulations"]],
                list(FAMILY_ORDER),
            )
            self.assertEqual(set(record["artifacts"]), {"pdf", "png"})


if __name__ == "__main__":
    unittest.main()
