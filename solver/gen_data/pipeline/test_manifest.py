"""CPU tests for the multi-shard paper-dataset view."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from solver.gen_data.pipeline.archive import (
    BatchPaths,
    CaseCommitRecord,
    commit_batch,
    ensure_proposal,
    ensure_shard,
)
from solver.gen_data.pipeline.manifest import (
    GenerationCompatibilityPolicy,
    GenerationCompatibilityVariant,
    build_dataset_view,
)
from solver.gen_data.pipeline.quota_driver import canonical_json_sha256


FINGERPRINT = "c" * 64
SECOND_FINGERPRINT = "d" * 64
THIRD_FINGERPRINT = "e" * 64
COMMON_SOURCE_HASH = "1" * 64
STOKES_SOURCE_HASH = "2" * 64
TANAKA_SOURCE_HASH = "3" * 64


def _shared_target(*, maximum_wavenumber: float = 1.0) -> dict[str, object]:
    return {
        "nx": 4,
        "length": 2.0 * np.pi,
        "gravity": 1.0,
        "dno_order": 2,
        "pad_factor": 2,
        "maximum_wavenumber": maximum_wavenumber,
        "dtype": "float64",
    }


def _static_metadata(*, maximum_wavenumber: float = 1.0) -> dict[str, object]:
    return {
        "family": "stokes",
        "case_kind": "static",
        "contract": {
            "role": "reduced_wiring_evidence_only",
            **_shared_target(maximum_wavenumber=maximum_wavenumber),
        },
    }


def _trajectory_metadata(
    family: str,
    *,
    evolution_nx: int = 4,
    target_nx: int | None = None,
    fine_dt: float = 0.01,
    evolution_dno_order: int = 2,
    target_dno_order: int | None = None,
    evolution_maximum_wavenumber: float = 1.0,
    target_maximum_wavenumber: float | None = None,
) -> dict[str, object]:
    numerical = {
        **_shared_target(
            maximum_wavenumber=evolution_maximum_wavenumber,
        ),
        "coarse_dt": 2.0 * fine_dt,
        "fine_dt": fine_dt,
        "retry_dt": 0.5 * fine_dt,
        "saved_dt": 0.02,
        "gl2_residual_tolerance": 1.0e-8,
        "gl2_iteration_cap": 8,
        "refinement_tolerance": 1.0e-3,
        "relative_floor": 1.0e-12,
        "target_time_chunk_size": 2,
    }
    numerical["nx"] = evolution_nx
    if target_nx is not None:
        numerical["target_nx"] = target_nx
    numerical["dno_order"] = evolution_dno_order
    if target_dno_order is not None:
        numerical["target_dno_order"] = target_dno_order
    if target_maximum_wavenumber is not None:
        numerical["target_maximum_wavenumber"] = target_maximum_wavenumber
    return {
        "family": family,
        "case_kind": "trajectory",
        "trajectory_execution": {
            "family": family,
            "role": "reduced_wiring_evidence_only",
            "numerical": numerical,
            "horizon": {
                "kind": "fixed_terminal_time",
                "fixed_terminal_time": 0.04,
                "period_count": None,
                "saved_grid_rounding": "exact",
            },
            "stored_time_policy": {
                "tanaka_count": 2,
                "benjamin_feir_count": 2,
                "random_sea_count": 2,
            },
            "jonswap_quadrature_order": None,
        },
    }


def _with_generation_identity(
    metadata: dict[str, object],
    *,
    sources: dict[str, str],
    dependency_tag: str = "locked",
    execution_platform: str = "cpu",
) -> dict[str, object]:
    return {
        **metadata,
        "additional_metadata": {
            "run_spec": {
                "configuration": {
                    "schema": "paper_dataset_quota_configuration_v1",
                    "dependency_environment": {
                        "python": "3.11",
                        "lock": dependency_tag,
                    },
                    "source_sha256": sources,
                    "execution_platform": execution_platform,
                },
            },
        },
    }


def _proposal(
    *,
    family_id: int,
    split_id: int,
    batch_id: int,
    case_ids: tuple[int, ...],
    revision_id: int = 1,
    fingerprint: str = FINGERPRINT,
    metadata: dict[str, object] | None = None,
) -> dict[str, np.ndarray]:
    return {
        "config_fingerprint": np.asarray(fingerprint),
        "family_id": np.asarray(family_id, dtype=np.int16),
        "revision_id": np.asarray(revision_id, dtype=np.int16),
        "split_id": np.asarray(split_id, dtype=np.uint8),
        "batch_id": np.asarray(batch_id, dtype=np.int64),
        "case_id": np.asarray(case_ids, dtype=np.int64),
        "cell_id": np.arange(len(case_ids), dtype=np.int32),
        "case_spec_json": np.asarray(
            [json.dumps({"case_id": case_id}, sort_keys=True) for case_id in case_ids]
        ),
        "metadata_json": np.asarray(
            json.dumps(metadata if metadata is not None else {"batch": batch_id})
        ),
    }


def _shard(
    proposal_sha256: str,
    *,
    local_indices: tuple[int, ...],
    frames_per_case: int,
    fingerprint: str,
    spatial_size: int = 4,
) -> dict[str, np.ndarray]:
    case_local_index = np.repeat(
        np.asarray(local_indices, dtype=np.int32),
        frames_per_case,
    )
    frame_index = np.tile(
        np.arange(frames_per_case, dtype=np.int32),
        len(local_indices),
    )
    selected_dense_index = frame_index * np.int32(4)
    time = frame_index.astype(np.float64) * 0.25
    rows = case_local_index.size
    eta = (
        np.arange(rows * spatial_size, dtype=np.float32)
        .reshape(rows, spatial_size)
        / 100.0
    )
    return {
        "eta": eta,
        "xi": eta + np.float32(0.01),
        "gxi": eta - np.float32(0.01),
        "depth": np.repeat(
            np.arange(1, len(local_indices) + 1, dtype=np.float64),
            frames_per_case,
        ),
        "time": time,
        "case_local_index": case_local_index,
        "frame_index": frame_index,
        "selected_dense_index": selected_dense_index,
        "config_fingerprint": np.asarray(fingerprint),
        "proposal_sha256": np.asarray(proposal_sha256),
    }


def _write_batch(
    paths: BatchPaths,
    *,
    proposal: dict[str, np.ndarray],
    accepted_local_indices: tuple[int, ...],
    frames_per_case: int,
    spatial_size: int = 4,
) -> None:
    proposal_sha256 = ensure_proposal(paths, proposal)
    shard = _shard(
        proposal_sha256,
        local_indices=accepted_local_indices,
        frames_per_case=frames_per_case,
        fingerprint=str(proposal["config_fingerprint"].item()),
        spatial_size=spatial_size,
    )
    ensure_shard(paths, shard)
    blocks = {
        local_index: (position * frames_per_case, frames_per_case)
        for position, local_index in enumerate(accepted_local_indices)
    }
    cases = tuple(
        CaseCommitRecord(
            case_id=int(case_id),
            accepted=local_index in blocks,
            required_bits=63 if local_index in blocks else 32,
            evaluated_bits=63,
            failed_bits=0 if local_index in blocks else 32,
            first_row=blocks.get(local_index, (-1, 0))[0],
            row_count=blocks.get(local_index, (-1, 0))[1],
            metrics={"maximum_error": 1e-6 if local_index in blocks else None},
        )
        for local_index, case_id in enumerate(proposal["case_id"])
    )
    commit_batch(paths, cases=cases, metadata={})


class DatasetManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_view_preserves_rejected_attempts_and_shard_row_mapping(self) -> None:
        first = BatchPaths.under(
            self.root,
            family="stokes",
            split="train",
            batch_id=0,
        )
        second = BatchPaths.under(
            self.root,
            family="tanaka",
            split="validation",
            batch_id=0,
        )
        _write_batch(
            first,
            proposal=_proposal(
                family_id=0,
                split_id=0,
                batch_id=0,
                case_ids=(10, 11),
            ),
            accepted_local_indices=(0,),
            frames_per_case=2,
        )
        _write_batch(
            second,
            proposal=_proposal(
                family_id=1,
                split_id=1,
                batch_id=0,
                case_ids=(20,),
            ),
            accepted_local_indices=(0,),
            frames_per_case=3,
        )

        paths = build_dataset_view(
            self.root,
            (first, second),
            expected_fingerprint=FINGERPRINT,
        )
        manifest_bytes = paths.manifest.read_bytes()
        map_bytes = paths.trajectory_map.read_bytes()
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        with np.load(paths.trajectory_map, allow_pickle=False) as trajectory_map:
            self.assertEqual(int(trajectory_map["schema_version"]), 2)
            self.assertTrue(
                np.array_equal(
                    trajectory_map["trajectory_index"],
                    np.asarray([0, 0, 2, 2, 2], dtype=np.int32),
                )
            )
            self.assertTrue(
                np.array_equal(
                    trajectory_map["trajectory_split_id"],
                    np.asarray([0, 0, 1], dtype=np.uint8),
                )
            )
            self.assertTrue(
                np.array_equal(
                    trajectory_map["trajectory_accepted"],
                    np.asarray([True, False, True]),
                )
            )
            self.assertTrue(
                np.array_equal(
                    trajectory_map["trajectory_required_bits"],
                    np.asarray([63, 32, 63], dtype=np.uint32),
                )
            )
            self.assertTrue(
                np.array_equal(
                    trajectory_map["trajectory_first_row"],
                    np.asarray([0, -1, 2], dtype=np.int64),
                )
            )
            self.assertTrue(
                np.array_equal(
                    trajectory_map["trajectory_row_count"],
                    np.asarray([2, 0, 3], dtype=np.int32),
                )
            )
            self.assertTrue(
                np.array_equal(
                    trajectory_map["shard_index"],
                    np.asarray([0, 0, 1, 1, 1], dtype=np.int32),
                )
            )
            self.assertTrue(
                np.array_equal(
                    trajectory_map["shard_row"],
                    np.asarray([0, 1, 0, 1, 2], dtype=np.int64),
                )
            )

        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["n_rows"], 5)
        self.assertEqual(manifest["n_trajectories"], 3)
        self.assertEqual(manifest["n_accepted_trajectories"], 2)
        self.assertEqual(
            manifest["split_counts"]["train"],
            {
                "attempted": 2,
                "accepted": 1,
            },
        )
        self.assertEqual(
            manifest["split_counts"]["validation"],
            {
                "attempted": 1,
                "accepted": 1,
            },
        )
        self.assertEqual(len(manifest["dataset_shards"]), 2)

        build_dataset_view(
            self.root,
            (first, second),
            expected_fingerprint=FINGERPRINT,
        )
        self.assertEqual(paths.manifest.read_bytes(), manifest_bytes)
        self.assertEqual(paths.trajectory_map.read_bytes(), map_bytes)

    def test_independent_family_and_split_runs_share_one_view(self) -> None:
        stokes_train = BatchPaths.under(
            self.root,
            family="stokes",
            split="train",
            batch_id=0,
        )
        tanaka_validation = BatchPaths.under(
            self.root,
            family="tanaka",
            split="validation",
            batch_id=0,
        )
        tanaka_test = BatchPaths.under(
            self.root,
            family="tanaka",
            split="test",
            batch_id=0,
        )
        _write_batch(
            stokes_train,
            proposal=_proposal(
                family_id=0,
                revision_id=2,
                split_id=0,
                batch_id=0,
                case_ids=(100,),
                fingerprint=FINGERPRINT,
                metadata=_with_generation_identity(
                    _static_metadata(),
                    sources={
                        "common.py": "4" * 64,
                        "stokes.py": STOKES_SOURCE_HASH,
                    },
                ),
            ),
            accepted_local_indices=(0,),
            frames_per_case=1,
        )
        _write_batch(
            tanaka_validation,
            proposal=_proposal(
                family_id=1,
                revision_id=3,
                split_id=1,
                batch_id=0,
                case_ids=(200,),
                fingerprint=SECOND_FINGERPRINT,
                metadata=_with_generation_identity(
                    _trajectory_metadata("tanaka"),
                    sources={
                        "common.py": COMMON_SOURCE_HASH,
                        "tanaka.py": TANAKA_SOURCE_HASH,
                    },
                    execution_platform="gpu",
                ),
            ),
            accepted_local_indices=(0,),
            frames_per_case=2,
        )
        _write_batch(
            tanaka_test,
            proposal=_proposal(
                family_id=1,
                revision_id=3,
                split_id=2,
                batch_id=0,
                case_ids=(300,),
                fingerprint=THIRD_FINGERPRINT,
                metadata=_with_generation_identity(
                    _trajectory_metadata("tanaka"),
                    sources={
                        "common.py": COMMON_SOURCE_HASH,
                        "tanaka.py": TANAKA_SOURCE_HASH,
                    },
                    execution_platform="gpu",
                ),
            ),
            accepted_local_indices=(0,),
            frames_per_case=2,
        )

        paths = build_dataset_view(
            self.root,
            (stokes_train, tanaka_validation, tanaka_test),
            expected_fingerprints=(
                FINGERPRINT,
                SECOND_FINGERPRINT,
                THIRD_FINGERPRINT,
            ),
        )
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))

        self.assertIsNone(manifest["configuration_fingerprint"])
        self.assertEqual(
            manifest["configuration_fingerprints"],
            sorted((FINGERPRINT, SECOND_FINGERPRINT, THIRD_FINGERPRINT)),
        )
        self.assertEqual(
            [
                batch["configuration_fingerprint"]
                for batch in manifest["dataset_batches"]
            ],
            [FINGERPRINT, SECOND_FINGERPRINT, THIRD_FINGERPRINT],
        )
        self.assertEqual(
            [
                shard["configuration_fingerprint"]
                for shard in manifest["dataset_shards"]
            ],
            [FINGERPRINT, SECOND_FINGERPRINT, THIRD_FINGERPRINT],
        )
        self.assertEqual(manifest["dataset_contract"]["target"]["nx"], 4)
        self.assertEqual(
            [
                (
                    record["family_id"],
                    record["revision_id"],
                    record["numerical"]["fine_dt"],
                )
                for record in manifest["dataset_contract"][
                    "trajectory_numerical_by_family_revision"
                ]
            ],
            [(1, 3, 0.01)],
        )
        self.assertEqual(
            manifest["dataset_contract"]["trajectory_numerical"]["fine_dt"],
            0.01,
        )
        self.assertEqual(
            manifest["dataset_contract"]["stored_dtypes"]["eta"],
            "float32",
        )
        generation_identity = manifest["dataset_contract"][
            "generation_identity"
        ]
        self.assertEqual(
            len(generation_identity["dependency_environment_fingerprint"]),
            64,
        )
        self.assertNotIn("source_sha256_fingerprint", generation_identity)
        self.assertEqual(
            len(generation_identity["compatibility_fingerprint"]),
            64,
        )
        self.assertEqual(
            [
                (
                    record["family_id"],
                    record["revision_id"],
                    record["execution_platform"],
                )
                for record in generation_identity["family_revisions"]
            ],
            [(0, 2, "cpu"), (1, 3, "gpu")],
        )
        self.assertTrue(
            all(
                len(record["source_sha256_fingerprint"]) == 64
                for record in generation_identity["family_revisions"]
            )
        )
        self.assertEqual(len(manifest["dataset_contract_fingerprint"]), 64)

    def test_mixed_generation_identity_mismatches_fail_closed(self) -> None:
        base_sources = {
            "common.py": COMMON_SOURCE_HASH,
            "tanaka.py": TANAKA_SOURCE_HASH,
        }
        scenarios = (
            (
                "dependency",
                _with_generation_identity(
                    _trajectory_metadata("tanaka"),
                    sources=base_sources,
                    dependency_tag="first",
                ),
                _with_generation_identity(
                    _trajectory_metadata("tanaka"),
                    sources=base_sources,
                    dependency_tag="second",
                ),
                "dependency environment",
            ),
            (
                "changed_source_digest",
                _with_generation_identity(
                    _trajectory_metadata("tanaka"),
                    sources=base_sources,
                ),
                _with_generation_identity(
                    _trajectory_metadata("tanaka"),
                    sources={
                        **base_sources,
                        "common.py": "4" * 64,
                    },
                ),
                "source mappings",
            ),
            (
                "family_source_mapping",
                _with_generation_identity(
                    _trajectory_metadata("tanaka"),
                    sources=base_sources,
                ),
                _with_generation_identity(
                    _trajectory_metadata("tanaka"),
                    sources={
                        **base_sources,
                        "new_family_file.py": "5" * 64,
                    },
                ),
                "source mappings",
            ),
            (
                "family_platform",
                _with_generation_identity(
                    _trajectory_metadata("tanaka"),
                    sources=base_sources,
                    execution_platform="cpu",
                ),
                _with_generation_identity(
                    _trajectory_metadata("tanaka"),
                    sources=base_sources,
                    execution_platform="gpu",
                ),
                "execution platforms",
            ),
            (
                "missing_identity",
                _with_generation_identity(
                    _trajectory_metadata("tanaka"),
                    sources=base_sources,
                ),
                _trajectory_metadata("tanaka"),
                "must record dependency and source identity",
            ),
        )
        for name, first_metadata, second_metadata, message in scenarios:
            with self.subTest(name=name):
                root = self.root / name
                first = BatchPaths.under(
                    root,
                    family="tanaka",
                    split="train",
                    batch_id=0,
                )
                second = BatchPaths.under(
                    root,
                    family="tanaka",
                    split="validation",
                    batch_id=0,
                )
                _write_batch(
                    first,
                    proposal=_proposal(
                        family_id=1,
                        split_id=0,
                        batch_id=0,
                        case_ids=(1,),
                        fingerprint=FINGERPRINT,
                        metadata=first_metadata,
                    ),
                    accepted_local_indices=(0,),
                    frames_per_case=2,
                )
                _write_batch(
                    second,
                    proposal=_proposal(
                        family_id=1,
                        split_id=1,
                        batch_id=0,
                        case_ids=(2,),
                        fingerprint=SECOND_FINGERPRINT,
                        metadata=second_metadata,
                    ),
                    accepted_local_indices=(0,),
                    frames_per_case=2,
                )
                with self.assertRaisesRegex(ValueError, message):
                    build_dataset_view(root, (first, second))

    def test_single_legacy_run_spec_without_new_identity_still_loads(self) -> None:
        paths = BatchPaths.under(
            self.root,
            family="stokes",
            split="validation",
            batch_id=0,
        )
        metadata = {
            **_static_metadata(),
            "additional_metadata": {
                "run_spec": {
                    "configuration": {
                        "schema": "legacy_static_pilot_configuration_v1",
                        "source_sha256": {
                            "legacy.py": STOKES_SOURCE_HASH,
                        },
                    },
                },
            },
        }
        _write_batch(
            paths,
            proposal=_proposal(
                family_id=0,
                split_id=1,
                batch_id=0,
                case_ids=(10,),
                metadata=metadata,
            ),
            accepted_local_indices=(0,),
            frames_per_case=1,
        )

        view = build_dataset_view(
            self.root,
            (paths,),
            expected_fingerprint=FINGERPRINT,
        )
        manifest = json.loads(view.manifest.read_text(encoding="utf-8"))
        self.assertIsNone(
            manifest["dataset_contract"]["generation_identity"]
        )

    def test_target_cutoff_mismatch_fails_closed(self) -> None:
        stokes = BatchPaths.under(
            self.root,
            family="stokes",
            split="train",
            batch_id=0,
        )
        tanaka = BatchPaths.under(
            self.root,
            family="tanaka",
            split="validation",
            batch_id=0,
        )
        _write_batch(
            stokes,
            proposal=_proposal(
                family_id=0,
                split_id=0,
                batch_id=0,
                case_ids=(400,),
                fingerprint=FINGERPRINT,
                metadata=_static_metadata(maximum_wavenumber=128.0),
            ),
            accepted_local_indices=(0,),
            frames_per_case=1,
        )
        _write_batch(
            tanaka,
            proposal=_proposal(
                family_id=1,
                split_id=1,
                batch_id=0,
                case_ids=(500,),
                fingerprint=SECOND_FINGERPRINT,
                metadata=_trajectory_metadata(
                    "tanaka",
                    evolution_maximum_wavenumber=224.0,
                    target_maximum_wavenumber=64.0,
                ),
            ),
            accepted_local_indices=(0,),
            frames_per_case=2,
        )

        with self.assertRaisesRegex(ValueError, "DNO target contract"):
            build_dataset_view(self.root, (stokes, tanaka))

    def test_buffered_tanaka_and_sharp_family_share_delivered_target(self) -> None:
        tanaka = BatchPaths.under(
            self.root,
            family="tanaka",
            split="train",
            batch_id=0,
        )
        benjamin_feir = BatchPaths.under(
            self.root,
            family="benjamin_feir",
            split="validation",
            batch_id=0,
        )
        _write_batch(
            tanaka,
            proposal=_proposal(
                family_id=1,
                revision_id=3,
                split_id=0,
                batch_id=0,
                case_ids=(600,),
                fingerprint=FINGERPRINT,
                metadata=_trajectory_metadata(
                    "tanaka",
                    evolution_maximum_wavenumber=224.0,
                    target_maximum_wavenumber=128.0,
                ),
            ),
            accepted_local_indices=(0,),
            frames_per_case=2,
        )
        _write_batch(
            benjamin_feir,
            proposal=_proposal(
                family_id=2,
                revision_id=2,
                split_id=1,
                batch_id=0,
                case_ids=(700,),
                fingerprint=SECOND_FINGERPRINT,
                metadata=_trajectory_metadata(
                    "benjamin_feir",
                    fine_dt=0.005,
                    evolution_maximum_wavenumber=128.0,
                ),
            ),
            accepted_local_indices=(0,),
            frames_per_case=2,
        )

        view = build_dataset_view(self.root, (tanaka, benjamin_feir))
        manifest = json.loads(view.manifest.read_text(encoding="utf-8"))

        self.assertEqual(
            manifest["dataset_contract"]["target"]["maximum_wavenumber"],
            128.0,
        )
        numerical_records = manifest["dataset_contract"][
            "trajectory_numerical_by_family_revision"
        ]
        self.assertEqual(
            [
                (
                    record["family_id"],
                    record["revision_id"],
                    record["numerical"]["maximum_wavenumber"],
                    record["numerical"].get("target_maximum_wavenumber"),
                    record["numerical"]["fine_dt"],
                )
                for record in numerical_records
            ],
            [
                (1, 3, 224.0, 128.0, 0.01),
                (2, 2, 128.0, None, 0.005),
            ],
        )
        self.assertIsNone(
            manifest["dataset_contract"]["trajectory_numerical"]
        )

    def test_manifest_reports_target_order_not_evolution_order(self) -> None:
        benjamin_feir = BatchPaths.under(
            self.root,
            family="benjamin_feir",
            split="test",
            batch_id=0,
        )
        _write_batch(
            benjamin_feir,
            proposal=_proposal(
                family_id=2,
                revision_id=3,
                split_id=2,
                batch_id=0,
                case_ids=(701,),
                fingerprint=FINGERPRINT,
                metadata=_trajectory_metadata(
                    "benjamin_feir",
                    evolution_dno_order=4,
                    target_dno_order=6,
                    evolution_maximum_wavenumber=256.0,
                    target_maximum_wavenumber=128.0,
                ),
            ),
            accepted_local_indices=(0,),
            frames_per_case=2,
        )

        view = build_dataset_view(self.root, (benjamin_feir,))
        manifest = json.loads(view.manifest.read_text(encoding="utf-8"))
        contract = manifest["dataset_contract"]

        self.assertEqual(contract["target"]["dno_order"], 6)
        self.assertEqual(contract["target"]["maximum_wavenumber"], 128.0)
        numerical = contract["trajectory_numerical_by_family_revision"][0][
            "numerical"
        ]
        self.assertEqual(numerical["dno_order"], 4)
        self.assertEqual(numerical["target_dno_order"], 6)
        self.assertEqual(numerical["maximum_wavenumber"], 256.0)
        self.assertEqual(numerical["target_maximum_wavenumber"], 128.0)

    def test_dual_grid_jonswap_shard_and_manifest_use_target_nx(self) -> None:
        jonswap = BatchPaths.under(
            self.root,
            family="jonswap_tma",
            split="train",
            batch_id=0,
        )
        _write_batch(
            jonswap,
            proposal=_proposal(
                family_id=4,
                revision_id=3,
                split_id=0,
                batch_id=0,
                case_ids=(702,),
                fingerprint=FINGERPRINT,
                metadata=_trajectory_metadata(
                    "jonswap_tma",
                    evolution_nx=2048,
                    target_nx=1024,
                    evolution_dno_order=4,
                    target_dno_order=6,
                    evolution_maximum_wavenumber=704.0,
                    target_maximum_wavenumber=128.0,
                ),
            ),
            accepted_local_indices=(0,),
            frames_per_case=2,
            spatial_size=1024,
        )

        view = build_dataset_view(self.root, (jonswap,))
        manifest = json.loads(view.manifest.read_text(encoding="utf-8"))
        with np.load(jonswap.shard, allow_pickle=False) as shard:
            self.assertEqual(shard["eta"].shape, (2, 1024))
        self.assertEqual(manifest["grid"], {"length": 2.0 * math.pi, "nx": 1024})
        target = manifest["dataset_contract"]["target"]
        self.assertEqual(target["nx"], 1024)
        numerical = manifest["dataset_contract"][
            "trajectory_numerical_by_family_revision"
        ][0]["numerical"]
        self.assertEqual(numerical["nx"], 2048)
        self.assertEqual(numerical["target_nx"], 1024)

    def test_identical_family_numerics_preserve_common_legacy_contract(
        self,
    ) -> None:
        batches = tuple(
            BatchPaths.under(
                self.root,
                family=family,
                split=split,
                batch_id=0,
            )
            for family, split in (
                ("tanaka", "train"),
                ("benjamin_feir", "validation"),
            )
        )
        for paths, family, family_id, revision_id, split_id, fingerprint in (
            (batches[0], "tanaka", 1, 3, 0, FINGERPRINT),
            (
                batches[1],
                "benjamin_feir",
                2,
                2,
                1,
                SECOND_FINGERPRINT,
            ),
        ):
            _write_batch(
                paths,
                proposal=_proposal(
                    family_id=family_id,
                    revision_id=revision_id,
                    split_id=split_id,
                    batch_id=0,
                    case_ids=(800 + family_id,),
                    fingerprint=fingerprint,
                    metadata=_trajectory_metadata(family),
                ),
                accepted_local_indices=(0,),
                frames_per_case=2,
            )

        view = build_dataset_view(self.root, batches)
        manifest = json.loads(view.manifest.read_text(encoding="utf-8"))
        contract = manifest["dataset_contract"]
        numerical_records = contract[
            "trajectory_numerical_by_family_revision"
        ]

        self.assertEqual(len(numerical_records), 2)
        self.assertEqual(
            numerical_records[0]["numerical"],
            numerical_records[1]["numerical"],
        )
        self.assertEqual(
            contract["trajectory_numerical"],
            numerical_records[0]["numerical"],
        )

    def test_one_family_revision_cannot_mix_numerical_contracts(self) -> None:
        first = BatchPaths.under(
            self.root,
            family="tanaka",
            split="train",
            batch_id=0,
        )
        second = BatchPaths.under(
            self.root,
            family="tanaka",
            split="validation",
            batch_id=0,
        )
        for paths, split_id, case_id, fingerprint, fine_dt in (
            (first, 0, 710, FINGERPRINT, 0.01),
            (second, 1, 711, SECOND_FINGERPRINT, 0.005),
        ):
            _write_batch(
                paths,
                proposal=_proposal(
                    family_id=1,
                    revision_id=3,
                    split_id=split_id,
                    batch_id=0,
                    case_ids=(case_id,),
                    fingerprint=fingerprint,
                    metadata=_trajectory_metadata(
                        "tanaka",
                        fine_dt=fine_dt,
                        evolution_maximum_wavenumber=224.0,
                        target_maximum_wavenumber=128.0,
                    ),
                ),
                accepted_local_indices=(0,),
                frames_per_case=2,
            )

        with self.assertRaisesRegex(
            ValueError,
            "one family revision cannot mix execution contracts",
        ):
            build_dataset_view(self.root, (first, second))

    def test_family_execution_comparison_is_json_type_strict(self) -> None:
        batches = tuple(
            BatchPaths.under(
                self.root,
                family="tanaka",
                split=split,
                batch_id=0,
            )
            for split in ("train", "validation")
        )
        for paths, split_id, fine_dt, fingerprint in (
            (batches[0], 0, 1.0, FINGERPRINT),
            (batches[1], 1, 1, SECOND_FINGERPRINT),
        ):
            _write_batch(
                paths,
                proposal=_proposal(
                    family_id=1,
                    revision_id=3,
                    split_id=split_id,
                    batch_id=0,
                    case_ids=(730 + split_id,),
                    fingerprint=fingerprint,
                    metadata=_trajectory_metadata(
                        "tanaka",
                        fine_dt=fine_dt,
                    ),
                ),
                accepted_local_indices=(0,),
                frames_per_case=2,
            )

        with self.assertRaisesRegex(
            ValueError,
            "one family revision cannot mix execution contracts",
        ):
            build_dataset_view(self.root, batches)

    def test_target_cutoff_cannot_exceed_evolution_cutoff(self) -> None:
        paths = BatchPaths.under(
            self.root,
            family="tanaka",
            split="test",
            batch_id=0,
        )
        _write_batch(
            paths,
            proposal=_proposal(
                family_id=1,
                revision_id=3,
                split_id=2,
                batch_id=0,
                case_ids=(720,),
                metadata=_trajectory_metadata(
                    "tanaka",
                    evolution_maximum_wavenumber=128.0,
                    target_maximum_wavenumber=224.0,
                ),
            ),
            accepted_local_indices=(0,),
            frames_per_case=2,
        )

        with self.assertRaisesRegex(
            ValueError,
            "target_maximum_wavenumber cannot exceed maximum_wavenumber",
        ):
            build_dataset_view(self.root, (paths,))

    def test_numerical_and_source_identity_are_scoped_by_family_revision(
        self,
    ) -> None:
        revision_two = BatchPaths.under(
            self.root,
            family="tanaka_v2",
            split="train",
            batch_id=0,
        )
        revision_three = BatchPaths.under(
            self.root,
            family="tanaka_v3",
            split="validation",
            batch_id=0,
        )
        for paths, revision_id, split_id, fingerprint, metadata in (
            (
                revision_two,
                2,
                0,
                FINGERPRINT,
                _with_generation_identity(
                    _trajectory_metadata(
                        "tanaka",
                        evolution_maximum_wavenumber=128.0,
                    ),
                    sources={"shared.py": "4" * 64},
                ),
            ),
            (
                revision_three,
                3,
                1,
                SECOND_FINGERPRINT,
                _with_generation_identity(
                    _trajectory_metadata(
                        "tanaka",
                        evolution_maximum_wavenumber=224.0,
                        target_maximum_wavenumber=128.0,
                    ),
                    sources={"shared.py": "5" * 64},
                ),
            ),
        ):
            _write_batch(
                paths,
                proposal=_proposal(
                    family_id=1,
                    revision_id=revision_id,
                    split_id=split_id,
                    batch_id=0,
                    case_ids=(730 + revision_id,),
                    fingerprint=fingerprint,
                    metadata=metadata,
                ),
                accepted_local_indices=(0,),
                frames_per_case=2,
            )

        view = build_dataset_view(
            self.root,
            (revision_two, revision_three),
        )
        manifest = json.loads(view.manifest.read_text(encoding="utf-8"))
        contract = manifest["dataset_contract"]

        self.assertEqual(
            [
                (record["family_id"], record["revision_id"])
                for record in contract[
                    "trajectory_numerical_by_family_revision"
                ]
            ],
            [(1, 2), (1, 3)],
        )
        generation_identity = contract["generation_identity"]
        self.assertEqual(
            [
                (record["family_id"], record["revision_id"])
                for record in generation_identity["family_revisions"]
            ],
            [(1, 2), (1, 3)],
        )
        self.assertNotIn("source_sha256_fingerprint", generation_identity)

    def test_audited_generation_variants_require_policy_and_preserve_raw_maps(
        self,
    ) -> None:
        current = _trajectory_metadata("tanaka")
        legacy = json.loads(json.dumps(current))
        legacy_execution = legacy["trajectory_execution"]
        assert isinstance(legacy_execution, dict)
        horizon = legacy_execution["horizon"]
        stored_time_policy = legacy_execution["stored_time_policy"]
        assert isinstance(horizon, dict)
        assert isinstance(stored_time_policy, dict)
        self.assertIsNone(horizon.pop("period_count"))
        horizon["peak_period_count"] = None
        stored_time_policy["benjamin_feir_alpha"] = 0.85
        stored_time_policy["benjamin_feir_sigma_steps"] = 20.0
        current_execution = current["trajectory_execution"]
        assert isinstance(current_execution, dict)
        current_sources = {"common.py": "1" * 64, "tanaka.py": "2" * 64}
        legacy_sources = {"common.py": "1" * 64, "tanaka.py": "3" * 64}
        compatibility_id = canonical_json_sha256(current_execution)
        policy = GenerationCompatibilityPolicy(
            (
                GenerationCompatibilityVariant(
                    family_id=1,
                    revision_id=3,
                    execution_record_fingerprint=canonical_json_sha256(
                        current_execution
                    ),
                    source_sha256_fingerprint=canonical_json_sha256(
                        current_sources
                    ),
                    compatibility_id=compatibility_id,
                    canonical_execution_record=current_execution,
                ),
                GenerationCompatibilityVariant(
                    family_id=1,
                    revision_id=3,
                    execution_record_fingerprint=canonical_json_sha256(
                        legacy_execution
                    ),
                    source_sha256_fingerprint=canonical_json_sha256(
                        legacy_sources
                    ),
                    compatibility_id=compatibility_id,
                    canonical_execution_record=current_execution,
                ),
            )
        )
        batches = tuple(
            BatchPaths.under(
                self.root,
                family=f"tanaka_variant_{index}",
                split=split,
                batch_id=0,
            )
            for index, split in enumerate(("train", "validation"))
        )
        for (
            paths,
            split_id,
            fingerprint,
            metadata,
            sources,
        ) in (
            (
                batches[0],
                0,
                FINGERPRINT,
                current,
                current_sources,
            ),
            (
                batches[1],
                1,
                SECOND_FINGERPRINT,
                legacy,
                legacy_sources,
            ),
        ):
            _write_batch(
                paths,
                proposal=_proposal(
                    family_id=1,
                    revision_id=3,
                    split_id=split_id,
                    batch_id=0,
                    case_ids=(950 + split_id,),
                    fingerprint=fingerprint,
                    metadata=_with_generation_identity(
                        metadata,
                        sources=sources,
                    ),
                ),
                accepted_local_indices=(0,),
                frames_per_case=2,
            )

        with self.assertRaisesRegex(ValueError, "mix execution contracts"):
            build_dataset_view(self.root, batches)
        with self.assertRaisesRegex(
            ValueError,
            "not an explicitly audited compatibility variant",
        ):
            policy.resolve(
                family_id=1,
                revision_id=3,
                execution_record=legacy_execution,
                source_sha256=current_sources,
            )

        view = build_dataset_view(
            self.root,
            batches,
            generation_compatibility_policy=policy,
        )
        manifest = json.loads(view.manifest.read_text(encoding="utf-8"))
        family_revision = manifest["dataset_contract"][  # type: ignore[index]
            "generation_identity"
        ]["family_revisions"][0]  # type: ignore[index]
        self.assertEqual(
            family_revision["generation_compatibility_id"],
            compatibility_id,
        )
        self.assertNotIn("source_sha256_fingerprint", family_revision)
        self.assertEqual(
            {
                tuple(sorted(variant["source_sha256"].items()))
                for variant in family_revision["generation_variants"]
            },
            {
                tuple(sorted(current_sources.items())),
                tuple(sorted(legacy_sources.items())),
            },
        )

    def test_mixed_fingerprints_require_contract_metadata(self) -> None:
        first = BatchPaths.under(
            self.root,
            family="legacy_a",
            split="train",
            batch_id=0,
        )
        second = BatchPaths.under(
            self.root,
            family="legacy_b",
            split="validation",
            batch_id=0,
        )
        _write_batch(
            first,
            proposal=_proposal(
                family_id=0,
                split_id=0,
                batch_id=0,
                case_ids=(800,),
                fingerprint=FINGERPRINT,
            ),
            accepted_local_indices=(0,),
            frames_per_case=1,
        )
        _write_batch(
            second,
            proposal=_proposal(
                family_id=1,
                split_id=1,
                batch_id=0,
                case_ids=(900,),
                fingerprint=SECOND_FINGERPRINT,
            ),
            accepted_local_indices=(0,),
            frames_per_case=1,
        )

        with self.assertRaisesRegex(ValueError, "shared execution-contract"):
            build_dataset_view(self.root, (first, second))

    def test_uncommitted_batch_cannot_enter_view(self) -> None:
        paths = BatchPaths.under(
            self.root,
            family="stokes",
            split="test",
            batch_id=0,
        )
        ensure_proposal(
            paths,
            _proposal(
                family_id=0,
                split_id=2,
                batch_id=0,
                case_ids=(30,),
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "committed batches"):
            build_dataset_view(self.root, (paths,))


if __name__ == "__main__":
    unittest.main()
