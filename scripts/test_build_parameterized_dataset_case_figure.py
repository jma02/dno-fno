"""Focused tests for source-bound deterministic family illustrations."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import build_parameterized_dataset_case_figure as figure_builder
from scripts.build_parameterized_dataset_case_figure import (
    CENTRAL_VALIDATION_CATEGORIES,
    EXPECTED_CELL_IDS,
    EXPECTED_REVISIONS,
    FAMILY_IDS,
    FAMILY_ORDER,
    ArtifactIdentity,
    CombinedViewIdentity,
    SelectedTrajectory,
    SourceIdentity,
    _render_figure,
    _validate_chunk_taxonomy,
    _validate_final_plan_layout,
    dimensionless_profile,
    load_combined_view_identity,
    load_illustration,
    load_source_identity,
    publish_outputs,
    select_validation_trajectories,
    validate_release_population,
    validate_source_map_binding,
)
from scripts.render_paper_dataset_worst_cases import (
    COMBINED_PREFLIGHT_SCHEMA,
    COMBINED_SUMMARY_SCHEMA,
    CombinedSummaryBinding,
    load_combined_summary_binding,
    load_source_summary,
    sha256,
)
from solver.gen_data.pipeline.production import SplitId, split_code
from solver.gen_data.pipeline.manifest import TRAJECTORY_MAP_SCHEMA_VERSION


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _source_fixture(
    root: Path,
    family: str,
    *,
    revision_id: int | None = None,
    category: str | None = None,
    case_ids: tuple[int, ...] = (17,),
) -> tuple[Path, Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    revision = EXPECTED_REVISIONS[family] if revision_id is None else revision_id
    selected_category = category or CENTRAL_VALIDATION_CATEGORIES[family]
    trajectory_count = len(case_ids)
    map_path = root / f"paper_dataset_{family}_validation.trajectory_map.npz"
    np.savez(
        map_path,
        schema_version=np.asarray(TRAJECTORY_MAP_SCHEMA_VERSION, dtype=np.int16),
        frame_index=np.zeros(trajectory_count, dtype=np.int32),
        trajectory_accepted=np.ones(trajectory_count, dtype=np.bool_),
        trajectory_case_id=np.asarray(case_ids, dtype=np.int64),
        trajectory_cell_id=np.zeros(trajectory_count, dtype=np.int32),
        trajectory_family_id=np.full(
            trajectory_count,
            int(FAMILY_IDS[family]),
            dtype=np.int16,
        ),
        trajectory_revision_id=np.full(
            trajectory_count,
            revision,
            dtype=np.int16,
        ),
        trajectory_split_id=np.full(
            trajectory_count,
            split_code(SplitId.VALIDATION),
            dtype=np.uint8,
        ),
        trajectory_first_row=np.arange(trajectory_count, dtype=np.int64),
        trajectory_row_count=np.ones(trajectory_count, dtype=np.int32),
        trajectory_required_bits=np.zeros(trajectory_count, dtype=np.uint32),
        trajectory_evaluated_bits=np.zeros(trajectory_count, dtype=np.uint32),
        trajectory_failed_bits=np.zeros(trajectory_count, dtype=np.uint32),
        trajectory_index=np.arange(trajectory_count, dtype=np.int32),
        shard_index=np.zeros(trajectory_count, dtype=np.int32),
        shard_row=np.arange(trajectory_count, dtype=np.int64),
    )
    shard_path = root / "shards" / family / "validation" / "batch_000000.npz"
    shard_path.parent.mkdir(parents=True)
    x = np.arange(8, dtype=np.float64)[None, :]
    np.savez(
        shard_path,
        eta=np.repeat(x, trajectory_count, axis=0),
        xi=np.repeat(2.0 * x, trajectory_count, axis=0),
        depth=np.full(trajectory_count, 2.0, dtype=np.float64),
        time=np.zeros(trajectory_count, dtype=np.float64),
    )
    manifest_path = root / f"paper_dataset_{family}_validation.dataset.json"
    _write_json(
        manifest_path,
        {
            "trajectory_map_npz": map_path.name,
            "trajectory_map_sha256": sha256(map_path),
            "n_accepted_trajectories": trajectory_count,
            "n_accepted_rows": trajectory_count,
            "dataset_shards": [
                {
                    "batch_index": 0,
                    "path": str(shard_path.relative_to(root)),
                    "sha256": sha256(shard_path),
                }
            ],
        },
    )
    numerical = {
        "gravity": 1.0,
        "length": 2.0 * np.pi,
    }
    configuration = (
        {"contract": numerical}
        if family == "stokes"
        else {"trajectory_execution": {"numerical": numerical}}
    )
    summary_path = root / f"paper_dataset_{family}_validation.summary.json"
    _write_json(
        summary_path,
        {
            "schema": "paper_dataset_quota_summary_v1",
            "status": "complete",
            "output_root": str(root.resolve()),
            "run_spec": {
                "family_name": family,
                "split_id": "validation",
                "revision_id": revision,
                "cell_codes": {selected_category: 0},
                "configuration": configuration,
            },
            "dataset_view": {
                "grid": {"length": 2.0 * np.pi, "nx": 8},
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


def _identity_fixture(
    root: Path,
    family: str,
    *,
    case_ids: tuple[int, ...],
) -> SourceIdentity:
    summary_path, *_ = _source_fixture(
        root,
        family,
        case_ids=case_ids,
    )
    return load_source_identity(load_source_summary(summary_path))


def _combined_fixture(
    root: Path,
) -> tuple[CombinedSummaryBinding, CombinedViewIdentity]:
    manifest = root / "combined.dataset.json"
    trajectory_map = root / "combined.trajectory_map.npz"
    np.savez(trajectory_map, marker=np.asarray((1,), dtype=np.int8))
    _write_json(
        manifest,
        {
            "schema": "fixture",
            "trajectory_map_npz": trajectory_map.name,
            "trajectory_map_sha256": sha256(trajectory_map),
        },
    )
    source_summary = root / "source.summary.json"
    _write_json(source_summary, {"status": "complete"})
    combined = root / "combined.summary.json"
    _write_json(
        combined,
        {
            "schema": COMBINED_SUMMARY_SCHEMA,
            "status": "complete",
            "preflight": {
                "schema": COMBINED_PREFLIGHT_SCHEMA,
                "chunks": [
                    {
                        "summary_path": str(source_summary.resolve()),
                        "summary_sha256": sha256(source_summary),
                    }
                ],
                "accepted_cases_total": 1,
                "expected_rows": 1,
            },
            "dataset_view": {
                "n_accepted_trajectories": 1,
                "n_accepted_rows": 1,
                "manifest": {
                    "path": str(manifest.resolve()),
                    "bytes": manifest.stat().st_size,
                    "sha256": sha256(manifest),
                },
                "trajectory_map": {
                    "path": str(trajectory_map.resolve()),
                    "bytes": trajectory_map.stat().st_size,
                    "sha256": sha256(trajectory_map),
                },
            },
        },
    )
    binding = load_combined_summary_binding(combined)
    return binding, load_combined_view_identity(binding)


class DeterministicSelectionTests(unittest.TestCase):
    def test_implementation_mutation_during_run_fails_precommit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            for relative, _ in figure_builder.IMPLEMENTATION_FILES:
                path = repository / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"# {relative}\n", encoding="utf-8")
            original = figure_builder.figure_implementation_record
            captured = original(repository)
            target = repository / figure_builder.IMPLEMENTATION_FILES[0][0]
            target.write_text("# changed during figure\n", encoding="utf-8")
            with (
                patch.object(
                    figure_builder,
                    "figure_implementation_record",
                    side_effect=lambda: original(repository),
                ),
                self.assertRaisesRegex(RuntimeError, "changed during execution"),
            ):
                figure_builder._require_figure_implementation_current(captured)

    def test_case_ids_are_sorted_and_explicit_lower_median_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identities = tuple(
                _identity_fixture(
                    root / family,
                    family,
                    case_ids=(40, 10, 30, 20),
                )
                for family in FAMILY_ORDER
            )

            selected = select_validation_trajectories(identities)

            self.assertEqual(
                tuple(item.trajectory.case_id for item in selected),
                (20, 20, 20, 20),
            )
            self.assertTrue(all(item.candidate_count == 4 for item in selected))
            self.assertTrue(all(item.lower_median_index == 1 for item in selected))

    def test_missing_fixed_category_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identities = [
                _identity_fixture(root / family, family, case_ids=(1,))
                for family in FAMILY_ORDER
            ]
            wrong = identities[1]
            wrong_trajectory = replace(
                wrong.source.trajectories[0],
                category="not_the_fixed_cell",
            )
            identities[1] = replace(
                wrong,
                source=replace(wrong.source, trajectories=(wrong_trajectory,)),
            )

            with self.assertRaisesRegex(ValueError, "no accepted validation cases"):
                select_validation_trajectories(identities)


class ContractAndBindingTests(unittest.TestCase):
    def test_frozen_source_layout_rejects_an_interval_change(self) -> None:
        layout = figure_builder._expected_chunk_layout()
        chunks = tuple(
            SimpleNamespace(
                family=item.family,
                split=item.split,
                stream_id=item.stream_id,
                accepted_before=item.accepted_before,
                accepted_after=item.accepted_after,
            )
            for item in layout
        )
        plan = SimpleNamespace(
            chunks=chunks,
            splits=tuple(SplitId),
            accepted_cases_per_family_by_split={
                "train": 16_384,
                "validation": 1_024,
                "test": 1_024,
            },
            accepted_cases=73_728,
            expected_rows=7_686_144,
        )

        _validate_final_plan_layout(plan)
        chunks[0].accepted_after += 1
        with self.assertRaisesRegex(ValueError, "26-source interval layout"):
            _validate_final_plan_layout(plan)

    def test_current_cell_taxonomy_and_balanced_quotas_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary_path, *_ = _source_fixture(Path(directory), "stokes")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            configuration = summary["run_spec"]["configuration"]
            configuration.update(
                {
                    "accepted_case_count": 1,
                    "accepted_cases_before": 0,
                    "accepted_cases_after": 1,
                    "ordered_cell_ids": list(EXPECTED_CELL_IDS["stokes"]),
                }
            )
            summary["run_spec"]["cell_codes"] = {
                cell_id: code
                for code, cell_id in enumerate(EXPECTED_CELL_IDS["stokes"])
            }
            summary["run_spec"]["quotas"] = list(
                figure_builder._expected_incremental_valid_case_targets(
                    "stokes",
                    accepted_before=0,
                    accepted_after=1,
                )
            )
            _write_json(summary_path, summary)
            chunk = SimpleNamespace(
                family="stokes",
                summary_path=summary_path,
                accepted_count=1,
                accepted_before=0,
                accepted_after=1,
            )

            _validate_chunk_taxonomy(chunk)
            summary["run_spec"]["configuration"]["ordered_cell_ids"].reverse()
            _write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "ordered cell taxonomy"):
                _validate_chunk_taxonomy(chunk)
            summary["run_spec"]["configuration"]["ordered_cell_ids"] = list(
                EXPECTED_CELL_IDS["stokes"]
            )
            summary["run_spec"]["quotas"][0]["target_accepted"] = 2
            _write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "balanced valid-case targets"):
                _validate_chunk_taxonomy(chunk)

    def test_wrong_source_revision_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary_path, *_ = _source_fixture(
                Path(directory),
                "tanaka",
                revision_id=2,
            )
            source = load_source_summary(summary_path)

            with self.assertRaisesRegex(ValueError, "required revision 3"):
                load_source_identity(source)

    def test_empty_population_cannot_satisfy_final_contract(self) -> None:
        binding = CombinedSummaryBinding(
            path=Path("/tmp/nonexistent-combined-summary"),
            sha256="0" * 64,
            source_roots=(),
            source_summary_paths=(),
            source_summary_sha256=(),
            expected_source_count=0,
            expected_accepted_cases=0,
            expected_retained_rows=0,
        )

        with self.assertRaisesRegex(ValueError, "final paper-dataset contract failed"):
            validate_release_population(binding, ())

    def test_combined_view_hash_and_root_binding_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding, _ = _combined_fixture(root)
            combined = json.loads(binding.path.read_text(encoding="utf-8"))
            combined["dataset_view"]["manifest"]["sha256"] = "0" * 64
            _write_json(binding.path, combined)
            changed_binding = replace(binding, sha256=sha256(binding.path))
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                load_combined_view_identity(changed_binding)

            outside = root.parent / f"{root.name}-outside.json"
            _write_json(outside, {"schema": "outside"})
            self.addCleanup(outside.unlink, missing_ok=True)
            combined["dataset_view"]["manifest"] = {
                "path": str(outside),
                "bytes": outside.stat().st_size,
                "sha256": sha256(outside),
            }
            _write_json(binding.path, combined)
            escaped_binding = replace(binding, sha256=sha256(binding.path))
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_combined_view_identity(escaped_binding)

    def test_source_map_shard_and_path_binding_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path, manifest_path, map_path, shard_path = _source_fixture(
                root,
                "stokes",
            )
            source = load_source_summary(summary_path)
            identity = load_source_identity(source)
            selected = SelectedTrajectory(
                identity=identity,
                trajectory=source.trajectories[0],
                candidate_count=1,
                lower_median_index=0,
            )
            with shard_path.open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "selected shard SHA-256"):
                load_illustration(selected)

            # Rebuild, then corrupt the map bytes after its summary was written.
            clean_root = root / "map_corruption"
            summary_path, _, map_path, _ = _source_fixture(clean_root, "stokes")
            with map_path.open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "trajectory map SHA-256"):
                load_source_summary(summary_path)

            # Rebuild, bind a valid outside shard in the manifest, and require
            # the hardened source loader to reject the escaping path.
            escape_root = root / "path_escape"
            summary_path, manifest_path, _, _ = _source_fixture(
                escape_root,
                "stokes",
            )
            outside = root / "outside.npz"
            np.savez(outside, marker=np.asarray((1,), dtype=np.int8))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["dataset_shards"][0].update(
                {
                    "path": "../outside.npz",
                    "sha256": sha256(outside),
                }
            )
            _write_json(manifest_path, manifest)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["dataset_view"]["manifest"].update(
                {
                    "bytes": manifest_path.stat().st_size,
                    "sha256": sha256(manifest_path),
                }
            )
            _write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_source_summary(summary_path)

    def test_combined_map_must_be_exact_ordered_union_of_source_maps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path, manifest_path, _, _ = _source_fixture(root, "stokes")
            source = load_source_summary(summary_path)
            combined_map = root / "combined.trajectory_map.npz"
            with np.load(source.map_path, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
            np.savez(combined_map, **arrays)
            combined_view = CombinedViewIdentity(
                manifest=ArtifactIdentity(
                    path=manifest_path,
                    bytes=manifest_path.stat().st_size,
                    sha256=sha256(manifest_path),
                ),
                trajectory_map=ArtifactIdentity(
                    path=combined_map,
                    bytes=combined_map.stat().st_size,
                    sha256=sha256(combined_map),
                ),
            )

            validate_source_map_binding((source,), combined_view)
            arrays["trajectory_case_id"] = arrays["trajectory_case_id"] + 1
            np.savez(combined_map, **arrays)
            changed_view = replace(
                combined_view,
                trajectory_map=ArtifactIdentity(
                    path=combined_map,
                    bytes=combined_map.stat().st_size,
                    sha256=sha256(combined_map),
                ),
            )
            with self.assertRaisesRegex(ValueError, "trajectory_case_id differs"):
                validate_source_map_binding((source,), changed_view)


class PlotAndPublicationTests(unittest.TestCase):
    def test_frame_zero_requires_zero_stored_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path, manifest_path, _, shard_path = _source_fixture(root, "stokes")
            with np.load(shard_path, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
            arrays["time"] = np.ones_like(arrays["time"])
            np.savez(shard_path, **arrays)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["dataset_shards"][0]["sha256"] = sha256(shard_path)
            _write_json(manifest_path, manifest)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["dataset_view"]["manifest"].update(
                {
                    "bytes": manifest_path.stat().st_size,
                    "sha256": sha256(manifest_path),
                }
            )
            _write_json(summary_path, summary)
            source = load_source_summary(summary_path)
            selected = SelectedTrajectory(
                identity=load_source_identity(source),
                trajectory=source.trajectories[0],
                candidate_count=1,
                lower_median_index=0,
            )

            with self.assertRaisesRegex(ValueError, "time zero"):
                load_illustration(selected)

    def test_dimensionless_plot_inputs_use_declared_scales(self) -> None:
        profile = dimensionless_profile(
            np.asarray((2.0, 4.0)),
            np.asarray((8.0, 16.0)),
            depth=2.0,
            gravity=2.0,
        )

        np.testing.assert_allclose(profile.x_over_length, (0.0, 0.5))
        np.testing.assert_allclose(profile.eta_over_depth, (1.0, 2.0))
        np.testing.assert_allclose(profile.xi_over_depth_speed, (2.0, 4.0))

    def test_four_by_two_renderer_accepts_only_dimensionless_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identities = tuple(
                _identity_fixture(root / family, family, case_ids=(1,))
                for family in FAMILY_ORDER
            )
            illustrations = tuple(
                load_illustration(selected)
                for selected in select_validation_trajectories(identities)
            )
            pdf_path = root / "figure.pdf"
            png_path = root / "figure.png"

            _render_figure(
                illustrations,
                pdf_path=pdf_path,
                png_path=png_path,
            )

            self.assertGreater(pdf_path.stat().st_size, 0)
            self.assertGreater(png_path.stat().st_size, 0)

    def test_publication_stages_then_publishes_json_commit_last(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding, combined_view = _combined_fixture(root)
            output_stem = root / "figure"
            final_paths = tuple(
                output_stem.with_suffix(suffix) for suffix in (".pdf", ".png", ".json")
            )
            for path in final_paths:
                path.write_bytes(b"old")

            def fake_render(
                illustrations: object,
                *,
                pdf_path: Path,
                png_path: Path,
            ) -> None:
                del illustrations
                self.assertTrue(
                    all(path.read_bytes() == b"old" for path in final_paths)
                )
                pdf_path.write_bytes(b"new-pdf")
                png_path.write_bytes(b"new-png")

            with (
                patch.object(figure_builder, "_render_figure", side_effect=fake_render),
                patch.object(
                    figure_builder,
                    "_case_record",
                    return_value={"family": "fixture"},
                ),
            ):
                published = publish_outputs(
                    (),
                    output_stem=output_stem,
                    binding=binding,
                    combined_view=combined_view,
                )

            self.assertEqual(published, final_paths)
            self.assertEqual(final_paths[0].read_bytes(), b"new-pdf")
            self.assertEqual(final_paths[1].read_bytes(), b"new-png")
            sidecar = json.loads(final_paths[2].read_text(encoding="utf-8"))
            self.assertEqual(sidecar["status"], "complete")
            self.assertEqual(
                sidecar["artifacts"]["pdf"]["sha256"], sha256(final_paths[0])
            )
            self.assertEqual(
                sidecar["artifacts"]["png"]["sha256"], sha256(final_paths[1])
            )

    def test_replacement_failure_removes_completion_marker_and_can_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding, combined_view = _combined_fixture(root)
            output_stem = root / "figure"
            final_pdf, final_png, final_json = tuple(
                output_stem.with_suffix(suffix) for suffix in (".pdf", ".png", ".json")
            )
            final_pdf.write_bytes(b"old-pdf")
            final_png.write_bytes(b"old-png")
            final_json.write_bytes(b"old-json")

            def fake_render(
                illustrations: object,
                *,
                pdf_path: Path,
                png_path: Path,
            ) -> None:
                del illustrations
                pdf_path.write_bytes(b"new-pdf")
                png_path.write_bytes(b"new-png")

            real_replace = figure_builder.os.replace
            replacement_count = 0

            def fail_second_replacement(source: Path, destination: Path) -> None:
                nonlocal replacement_count
                replacement_count += 1
                if replacement_count == 2:
                    raise OSError("injected PNG replacement failure")
                real_replace(source, destination)

            with (
                patch.object(figure_builder, "_render_figure", side_effect=fake_render),
                patch.object(
                    figure_builder,
                    "_case_record",
                    return_value={"family": "fixture"},
                ),
                patch.object(
                    figure_builder.os,
                    "replace",
                    side_effect=fail_second_replacement,
                ),
                self.assertRaisesRegex(OSError, "PNG replacement failure"),
            ):
                publish_outputs(
                    (),
                    output_stem=output_stem,
                    binding=binding,
                    combined_view=combined_view,
                )

            self.assertEqual(final_pdf.read_bytes(), b"new-pdf")
            self.assertEqual(final_png.read_bytes(), b"old-png")
            self.assertFalse(final_json.exists())

            with (
                patch.object(figure_builder, "_render_figure", side_effect=fake_render),
                patch.object(
                    figure_builder,
                    "_case_record",
                    return_value={"family": "fixture"},
                ),
            ):
                publish_outputs(
                    (),
                    output_stem=output_stem,
                    binding=binding,
                    combined_view=combined_view,
                )
            self.assertEqual(final_pdf.read_bytes(), b"new-pdf")
            self.assertEqual(final_png.read_bytes(), b"new-png")
            self.assertEqual(
                json.loads(final_json.read_text(encoding="utf-8"))["status"],
                "complete",
            )

    def test_render_failure_leaves_existing_destinations_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding, combined_view = _combined_fixture(root)
            output_stem = root / "figure"
            final_paths = tuple(
                output_stem.with_suffix(suffix) for suffix in (".pdf", ".png", ".json")
            )
            for path in final_paths:
                path.write_bytes(b"old")

            with (
                patch.object(
                    figure_builder,
                    "_render_figure",
                    side_effect=RuntimeError("render failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "render failed"),
            ):
                publish_outputs(
                    (),
                    output_stem=output_stem,
                    binding=binding,
                    combined_view=combined_view,
                )

            self.assertTrue(all(path.read_bytes() == b"old" for path in final_paths))


if __name__ == "__main__":
    unittest.main()
