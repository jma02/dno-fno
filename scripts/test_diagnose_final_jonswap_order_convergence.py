from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import diagnose_final_jonswap_order_convergence as diagnostic
from solver.gen_data.pipeline.archive import file_sha256


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fixture(root: Path) -> Path:
    chunk = root / "train/jonswap_tma/chunk_00000_00003"
    proposal_path = chunk / "proposals/jonswap_tma/train/batch_000000.npz"
    result_path = chunk / "results/jonswap_tma/train/batch_000000.json"
    shard_path = chunk / "shards/jonswap_tma/train/batch_000000.npz"
    proposal_path.parent.mkdir(parents=True)
    result_path.parent.mkdir(parents=True)
    shard_path.parent.mkdir(parents=True)

    case_ids = np.asarray((10, 11, 12), dtype=np.int64)
    specifications = tuple(
        json.dumps(
            {"case_id": int(case_id), "cell_id": "shallow", "depth": 0.1},
            sort_keys=True,
            separators=(",", ":"),
        )
        for case_id in case_ids
    )
    np.savez(
        proposal_path,
        batch_id=np.asarray(0, dtype=np.int64),
        case_id=case_ids,
        case_spec_json=np.asarray(specifications),
    )

    nx = diagnostic.NX
    frames = diagnostic.ROWS_PER_TRAJECTORY
    x = 2.0 * np.pi * np.arange(nx, dtype=np.float64) / nx
    eta = np.tile(0.01 * np.sin(2.0 * x), (3 * frames, 1)).astype(np.float32)
    xi = np.tile(0.02 * np.cos(3.0 * x), (3 * frames, 1)).astype(np.float32)
    gxi = np.empty_like(eta)
    amplitudes = (0.25, 0.5, 1.0)
    for case_index, amplitude in enumerate(amplitudes):
        for frame in range(frames):
            high_amplitude = 2.0 if case_index == 2 and frame == 7 else amplitude
            gxi[case_index * frames + frame] = np.cos(
                5.0 * x
            ) + high_amplitude * np.cos(100.0 * x)
    np.savez(
        shard_path,
        eta=eta,
        xi=xi,
        gxi=gxi,
        depth=np.full(3 * frames, 0.1, dtype=np.float64),
        case_local_index=np.repeat(np.arange(3, dtype=np.int32), frames),
        frame_index=np.tile(np.arange(frames, dtype=np.int32), 3),
    )
    proposal_sha256 = file_sha256(proposal_path)
    shard_sha256 = file_sha256(shard_path)
    _write_json(
        result_path,
        {
            "proposal_sha256": proposal_sha256,
            "shard_sha256": shard_sha256,
            "cases": [
                {
                    "accepted": True,
                    "case_id": int(case_id),
                    "first_row": index * frames,
                    "row_count": frames,
                }
                for index, case_id in enumerate(case_ids)
            ],
        },
    )
    result_sha256 = file_sha256(result_path)

    map_path = chunk / "paper_dataset_jonswap_tma_train.trajectory_map.npz"
    np.savez(
        map_path,
        schema_version=np.asarray(2, dtype=np.int16),
        frame_index=np.tile(np.arange(frames, dtype=np.int32), 3),
        shard_index=np.zeros(3 * frames, dtype=np.int32),
        shard_row=np.arange(3 * frames, dtype=np.int64),
        trajectory_accepted=np.ones(3, dtype=np.bool_),
        trajectory_case_id=case_ids,
        trajectory_cell_id=np.zeros(3, dtype=np.int32),
        trajectory_family_id=np.full(3, 4, dtype=np.int16),
        trajectory_first_row=np.arange(3, dtype=np.int64) * frames,
        trajectory_index=np.repeat(np.arange(3, dtype=np.int32), frames),
        trajectory_revision_id=np.full(3, 4, dtype=np.int16),
        trajectory_row_count=np.full(3, frames, dtype=np.int32),
        trajectory_split_id=np.zeros(3, dtype=np.uint8),
    )
    map_sha256 = file_sha256(map_path)

    manifest_path = chunk / "paper_dataset_jonswap_tma_train.dataset.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 2,
            "grid": {"nx": nx, "length": 2.0 * np.pi},
            "n_accepted_trajectories": 3,
            "n_accepted_rows": 3 * frames,
            "n_trajectories": 3,
            "dataset_batches": [
                {
                    "batch_id": 0,
                    "shard_index": 0,
                    "n_attempted_trajectories": 3,
                    "proposal_path": str(proposal_path.relative_to(chunk)),
                    "proposal_sha256": proposal_sha256,
                    "result_path": str(result_path.relative_to(chunk)),
                    "result_sha256": result_sha256,
                }
            ],
            "dataset_shards": [
                {
                    "batch_index": 0,
                    "n_rows": 3 * frames,
                    "path": str(shard_path.relative_to(chunk)),
                    "sha256": shard_sha256,
                }
            ],
        },
    )
    manifest_sha256 = file_sha256(manifest_path)

    summary_path = chunk / "paper_dataset_jonswap_tma_train.summary.json"
    _write_json(
        summary_path,
        {
            "status": "complete",
            "run_spec": {
                "cell_codes": {"shallow": 0},
                "configuration": {
                    "source_sha256": {
                        relative_path: file_sha256(diagnostic.ROOT / relative_path)
                        for relative_path in (
                            "solver/gen_data/pipeline/reference.py",
                            "solver/solvers/dno_series_jax.py",
                        )
                    }
                },
            },
            "dataset_view": {
                "manifest": {
                    "path": manifest_path.name,
                    "sha256": manifest_sha256,
                },
                "trajectory_map": {"path": map_path.name, "sha256": map_sha256},
            },
        },
    )
    summary_sha256 = file_sha256(summary_path)
    fractions = diagnostic.high_band_fraction(gxi[2 * frames : 3 * frames])
    audit_path = root / "jonswap_tma_completion_audit.json"
    _write_json(
        audit_path,
        {
            "schema": diagnostic.AUDIT_SCHEMA,
            "status": "pass",
            "dataset_root": str(root),
            "accepted": 3,
            "retained_rows": 3 * frames,
            "identity": {
                "source_sha256_fingerprint": "source",
                "dependency_environment_fingerprint": "dependency",
                "execution_record_fingerprint": "execution",
                "current_support_source_sha256": {"source.py": "hash"},
            },
            "trajectory_quality_diagnostics": {
                "metric_values_affect_acceptance_or_audit_status": False,
                "accepted_trajectories_checked": 3,
                "metrics": {
                    "gxi_high_band_energy_fraction_maximum": {
                        "maximum": {
                            "value": float(fractions[7]),
                            "identity": {
                                "case_id": 12,
                                "chunk_label": "train_00000_00003",
                                "split": "train",
                                "batch_id": 0,
                                "local_index": 2,
                                "frame_index": 7,
                                "cell_id": "shallow",
                                "shard_path": str(shard_path.relative_to(chunk)),
                            },
                        }
                    }
                },
            },
            "chunks": [
                {
                    "label": "train_00000_00003",
                    "split": "train",
                    "accepted_count": 3,
                    "attempted_count": 3,
                    "retained_rows": 3 * frames,
                    "summary_path": str(summary_path),
                    "summary_sha256": summary_sha256,
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": manifest_sha256,
                    "trajectory_map_path": str(map_path),
                    "trajectory_map_sha256": map_sha256,
                }
            ],
        },
    )
    return audit_path


def _fake_evaluator(
    eta: np.ndarray, xi: np.ndarray, depth: float
) -> dict[int, diagnostic.OrderFields]:
    del xi, depth
    base = 1.0 + eta
    return {
        order: diagnostic.OrderFields(
            raw=base * (1.0 + order * 1.0e-4),
            projected=base * (1.0 + order * 1.0e-4),
        )
        for order in diagnostic.ORDERS
    }


def _renderer_fixture(path: Path, scan: diagnostic.ScanResult) -> Path:
    combined_path = path / "combined.summary.json"
    _write_json(
        combined_path,
        {"schema": diagnostic.COMBINED_SUMMARY_SCHEMA, "status": "complete"},
    )
    selected = [
        {
            "family": "jonswap_tma",
            "split": candidate.split,
            "case_id": candidate.case_id,
            "category": candidate.cell_id,
            "trajectory_index": candidate.trajectory_index,
            "shard_index": candidate.batch.shard_index,
            "first_shard_row": candidate.shard_first_row,
            "row_count": diagnostic.ROWS_PER_TRAJECTORY,
            "maximum_gxi_high_band_fraction_frame": candidate.peak_frame_index,
            "maximum_gxi_high_band_fraction": candidate.peak_fraction,
            "source_root": str(candidate.summary_path.parent),
        }
        for candidate in scan.candidates
    ]
    renderer_path = path / "renderer.summary.json"
    _write_json(
        renderer_path,
        {
            "schema": diagnostic.RENDERER_SCHEMA,
            "status": "complete",
            "parameters": {"final_paper_dataset_contract_required": True},
            "source_binding": {
                "mode": "combined_summary",
                "combined_summary_path": str(combined_path),
                "combined_summary_sha256": file_sha256(combined_path),
                "expected_sources": 26,
                "expected_accepted_cases": 73_728,
                "expected_retained_rows": 7_686_144,
            },
            "population": {
                "sources": 26,
                "accepted_cases": 73_728,
                "retained_rows": 7_686_144,
            },
            "families": {
                "jonswap_tma": {
                    "accepted_cases": 3,
                    "retained_rows": 48,
                    "rankings": {"gxi_high_band": selected},
                }
            },
        },
    )
    return renderer_path


class FinalJonswapOrderDiagnosticTests(unittest.TestCase):
    def test_implementation_mutation_during_run_fails_precommit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            for relative, _ in diagnostic.IMPLEMENTATION_FILES:
                path = repository / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"# {relative}\n", encoding="utf-8")
            original = diagnostic.diagnostic_implementation_record
            captured = original(repository)
            target = repository / diagnostic.IMPLEMENTATION_FILES[0][0]
            target.write_text("# changed during diagnostic\n", encoding="utf-8")
            with (
                patch.object(
                    diagnostic,
                    "diagnostic_implementation_record",
                    side_effect=lambda: original(repository),
                ),
                self.assertRaisesRegex(RuntimeError, "changed during execution"),
            ):
                diagnostic._require_diagnostic_implementation_current(captured)

    def test_scan_reproduces_maximum_and_ranks_three_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit_path = _fixture(Path(directory))
            _, scan = diagnostic.scan_final_tail(
                audit_path, expected_accepted=3, expected_chunks=1, top_count=3
            )
            self.assertEqual([case.case_id for case in scan.candidates], [12, 11, 10])
            self.assertEqual(scan.candidates[0].peak_frame_index, 7)
            self.assertEqual(scan.accepted_scanned, 3)
            self.assertEqual(scan.rows_scanned, 48)

    def test_complete_record_is_explicitly_non_gating(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit_path = _fixture(Path(directory))
            record = diagnostic.run_diagnostic(
                audit_path,
                evaluator=_fake_evaluator,
                expected_accepted=3,
                expected_chunks=1,
                top_count=3,
            )
            self.assertEqual(record["status"], "pass")
            self.assertTrue(record["diagnostic_only"])
            self.assertFalse(record["affects_dataset_acceptance"])
            self.assertFalse(record["affects_dataset_release"])
            self.assertEqual(len(record["cases"]), 3)
            case = record["cases"][0]
            self.assertEqual(len(case["successive_order_changes"]), 4)
            self.assertEqual(len(case["orders"]["6"]["projected_l2_norm_by_frame"]), 16)

    def test_output_path_only_replaces_an_existing_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new_path = root / "new.json"
            self.assertEqual(
                diagnostic.validated_output_path(new_path), new_path.resolve()
            )

            existing_diagnostic = root / "diagnostic.json"
            _write_json(
                existing_diagnostic,
                {"schema": diagnostic.DIAGNOSTIC_SCHEMA, "status": "pass"},
            )
            self.assertEqual(
                diagnostic.validated_output_path(existing_diagnostic),
                existing_diagnostic.resolve(),
            )

            completion_audit = root / "completion_audit.json"
            _write_json(
                completion_audit,
                {"schema": diagnostic.AUDIT_SCHEMA, "status": "pass"},
            )
            with self.assertRaisesRegex(
                FileExistsError, "refusing to replace non-diagnostic artifact"
            ):
                diagnostic.validated_output_path(completion_audit)

    def test_wrong_audit_schema_or_status_fails_closed(self) -> None:
        for key, value in (("schema", "wrong"), ("status", "incomplete")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                audit_path = _fixture(Path(directory))
                record = json.loads(audit_path.read_text(encoding="utf-8"))
                record[key] = value
                _write_json(audit_path, record)
                with self.assertRaisesRegex(ValueError, "passed revision-4"):
                    diagnostic.scan_final_tail(
                        audit_path, expected_accepted=3, expected_chunks=1
                    )

    def test_chunk_summary_hash_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit_path = _fixture(Path(directory))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            summary_path = Path(audit["chunks"][0]["summary_path"])
            summary_path.write_text(summary_path.read_text() + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "chunk summary hash mismatch"):
                diagnostic.scan_final_tail(
                    audit_path, expected_accepted=3, expected_chunks=1
                )

    def test_map_hash_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit_path = _fixture(Path(directory))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            map_path = Path(audit["chunks"][0]["trajectory_map_path"])
            map_path.write_bytes(map_path.read_bytes() + b"mutation")
            with self.assertRaisesRegex(ValueError, "trajectory map hash mismatch"):
                diagnostic.scan_final_tail(
                    audit_path, expected_accepted=3, expected_chunks=1
                )

    def test_completion_audit_maximum_identity_is_independently_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit_path = _fixture(Path(directory))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            maximum = audit["trajectory_quality_diagnostics"]["metrics"][
                "gxi_high_band_energy_fraction_maximum"
            ]["maximum"]
            maximum["identity"]["case_id"] = 999
            _write_json(audit_path, audit)
            with self.assertRaisesRegex(ValueError, "maximum identity differs"):
                diagnostic.scan_final_tail(
                    audit_path, expected_accepted=3, expected_chunks=1
                )

    def test_selected_transaction_hash_is_rechecked_before_order_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit_path = _fixture(Path(directory))
            _, scan = diagnostic.scan_final_tail(
                audit_path, expected_accepted=3, expected_chunks=1
            )
            result_path = scan.candidates[0].batch.result_path
            result_path.write_text(result_path.read_text() + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selected result hash mismatch"):
                diagnostic.case_order_record(
                    scan.candidates[0], evaluator=_fake_evaluator
                )

    def test_high_band_fraction_matches_single_mode_energy_ratio(self) -> None:
        x = 2.0 * np.pi * np.arange(diagnostic.NX) / diagnostic.NX
        fields = (np.cos(5.0 * x) + 2.0 * np.cos(100.0 * x))[None, :]
        self.assertAlmostEqual(float(diagnostic.high_band_fraction(fields)[0]), 0.8)

    def test_renderer_selection_and_combined_summary_are_authenticated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_path = _fixture(root)
            _, scan = diagnostic.scan_final_tail(
                audit_path, expected_accepted=3, expected_chunks=1, top_count=3
            )
            renderer_path = _renderer_fixture(root, scan)
            binding = diagnostic.authenticate_renderer_selection(
                renderer_path, scan, expected_accepted=3
            )
            self.assertTrue(binding["exact_final_population_verified"])
            combined_path = Path(binding["combined_summary_path"])
            combined_path.write_text(combined_path.read_text() + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "combined summary hash mismatch"):
                diagnostic.authenticate_renderer_selection(
                    renderer_path, scan, expected_accepted=3
                )


if __name__ == "__main__":
    unittest.main()
