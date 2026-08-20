"""Focused fail-closed tests for the BF revision-4 completion audit."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_completed_benjamin_feir_revision4 import (  # noqa: E402
    AuditedCase,
    AuditTotals,
    Extrema,
    EXPECTED_CHUNKS,
    EXPECTED_EVALUATED_BITS,
    EXPECTED_GENERATION_SOURCE_PATHS,
    EXPECTED_MAP_ARRAYS,
    EXPECTED_REQUIRED_BITS,
    HISTORICAL_SHARED_SOURCE_SNAPSHOTS,
    ProposedCase,
    _accepted_cell_totals_by_split,
    _bf_horizon,
    _case_specifications,
    _expected_chunk_quotas,
    _historical_shared_source_snapshot_record,
    _nonhistorical_generation_source_record,
    _validate_summary_cell_counts,
    _validate_chunk_taxonomy_and_quotas,
    _validate_case_record,
    _validate_map_array_schema,
    _validate_shard,
)
from scripts.build_paper_corpus_view import TRAJECTORY_MAP_DTYPES  # noqa: E402
from solver.gen_data.benjamin_feir_population import (  # noqa: E402
    BENJAMIN_FEIR_POPULATION_CELLS,
)
from solver.gen_data.pipeline.production import (  # noqa: E402
    AttemptAssignment,
    CaseKey,
    SplitId,
)
from solver.gen_data.pipeline.archive import file_sha256  # noqa: E402
from solver.gen_data.trajectory_family_adapters import (  # noqa: E402
    sample_benjamin_feir_trajectory_cases,
)
from solver.gen_data.trajectory_quota_executor import (  # noqa: E402
    TrajectoryExecutionConfig,
)
from solver.gen_data.pipeline.time_selection import select_uniform_times  # noqa: E402


def _frozen_source_mapping() -> dict[str, str]:
    historical = {
        path: digest
        for path, (_name, digest) in HISTORICAL_SHARED_SOURCE_SNAPSHOTS.items()
    }
    return {
        path: historical.get(path, file_sha256(ROOT / path))
        for path in EXPECTED_GENERATION_SOURCE_PATHS
    }


def _support_extrema() -> dict[str, Extrema]:
    return {
        name: Extrema()
        for name in (
            "carrier_mode",
            "sideband_offset",
            "carrier_steepness",
            "sideband_ratio",
            "translation",
            "instability_band_fraction",
            "focused_steepness",
        )
    }


def _accepted_extrema() -> dict[str, Extrema]:
    return {
        name: Extrema()
        for name in (
            "initial_hamiltonian",
            "hamiltonian_drift",
            "stage_residual",
            "minimum_water_column",
        )
    }


def _attempted_extrema() -> dict[str, Extrema]:
    return {
        name: Extrema()
        for name in (
            "hamiltonian_drift",
            "stage_residual",
            "minimum_water_column",
        )
    }


def _proposed_case() -> ProposedCase:
    intended, realized, saved_count = _bf_horizon(4.0, 0.08)
    return ProposedCase(
        case_id=7,
        attempt_index=0,
        cell_code=0,
        cell_id=BENJAMIN_FEIR_POPULATION_CELLS[0].cell_id,
        depth=5.0,
        carrier_wavenumber=4.0,
        intended_terminal_time=intended,
        realized_terminal_time=realized,
        saved_time_count=saved_count,
    )


def _accepted_case(proposed: ProposedCase | None = None) -> dict[str, object]:
    proposal = proposed or _proposed_case()
    return {
        "accepted": True,
        "case_id": proposal.case_id,
        "required_bits": EXPECTED_REQUIRED_BITS,
        "evaluated_bits": EXPECTED_EVALUATED_BITS,
        "failed_bits": 0,
        "first_row": 0,
        "row_count": 200,
        "metrics": {
            "accepted": True,
            "all_stages_solved": True,
            "complete_admissible_trajectory": True,
            "initial_internal_hamiltonian": 0.01,
            "intended_terminal_time": proposal.intended_terminal_time,
            "internal_dno_finite": True,
            "internal_hamiltonian_drift_threshold": 1.0e-3,
            "internal_health_evaluated": True,
            "internal_state_finite": True,
            "maximum_internal_hamiltonian_drift": 1.0e-5,
            "maximum_stage_residual": 1.0e-9,
            "minimum_internal_water_column": 4.9,
            "positive_water_column": True,
            "production_dt": 0.01,
            "realized_terminal_time": proposal.realized_terminal_time,
            "saved_time_count": proposal.saved_time_count,
            "state_finite": True,
            "target_finite": True,
        },
    }


class BenjaminFeirCompletionAuditTests(unittest.TestCase):
    def test_exact_support_replay_rejects_parameter_mutation(self) -> None:
        cell = BENJAMIN_FEIR_POPULATION_CELLS[0]
        key = CaseKey(
            family_id=3,
            revision_id=4,
            split_id=SplitId.TRAIN,
            stream_id=0,
            attempt_index=0,
        )
        assignment = AttemptAssignment(key, cell.cell_id)
        sampled = sample_benjamin_feir_trajectory_cases(
            (assignment,),
            contract=TrajectoryExecutionConfig.paper("benjamin_feir").numerical,
        )
        record = dict(sampled.specification_records[0])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "proposal.npz"

            def write(specification: dict[str, object]) -> None:
                np.savez(
                    path,
                    batch_id=np.asarray(0, dtype=np.int64),
                    case_id=np.asarray([key.case_id], dtype=np.int64),
                    root_seed=np.asarray([key.root_seed], dtype=np.uint64),
                    stream_id=np.asarray([key.stream_id], dtype=np.uint32),
                    attempt_index=np.asarray(
                        [key.attempt_index],
                        dtype=np.uint64,
                    ),
                    cell_id=np.asarray([0], dtype=np.int32),
                    config_fingerprint=np.asarray("fingerprint"),
                    family_id=np.asarray(3, dtype=np.int16),
                    metadata_json=np.asarray("{}"),
                    revision_id=np.asarray(4, dtype=np.int16),
                    split_id=np.asarray(0, dtype=np.uint8),
                    case_spec_json=np.asarray(
                        [json.dumps(specification, allow_nan=False)]
                    ),
                )

            write(record)
            totals = AuditTotals()
            _case_specifications(
                path,
                split=SplitId.TRAIN,
                stream_id=0,
                batch_id=0,
                configuration_fingerprint="fingerprint",
                saved_dt=0.08,
                cell_ids_by_code={0: cell.cell_id},
                support_extrema=_support_extrema(),
                totals=totals,
            )
            self.assertEqual(totals.proposal_specs_checked, 1)

            record["first_harmonic_carrier_steepness"] = 0.2
            write(record)
            with self.assertRaisesRegex(ValueError, "deterministic sampling"):
                _case_specifications(
                    path,
                    split=SplitId.TRAIN,
                    stream_id=0,
                    batch_id=0,
                    configuration_fingerprint="fingerprint",
                    saved_dt=0.08,
                    cell_ids_by_code={0: cell.cell_id},
                    support_extrema=_support_extrema(),
                    totals=AuditTotals(),
                )

    def test_case_audit_rejects_contract_and_row_mutations(self) -> None:
        proposed = _proposed_case()
        case = _accepted_case(proposed)
        audited = _validate_case_record(
            case,
            proposed=proposed,
            batch_id=0,
            residual_tolerance=1.0e-8,
            hamiltonian_threshold=1.0e-3,
            production_dt=0.01,
            accepted_extrema=_accepted_extrema(),
            attempted_extrema=_attempted_extrema(),
            rejection_reasons=Counter(),
        )
        self.assertTrue(audited.accepted)

        metrics = case["metrics"]
        assert isinstance(metrics, dict)
        metrics["maximum_stage_residual"] = 1.1e-8
        with self.assertRaisesRegex(ValueError, "residual tolerance"):
            _validate_case_record(
                case,
                proposed=proposed,
                batch_id=0,
                residual_tolerance=1.0e-8,
                hamiltonian_threshold=1.0e-3,
                production_dt=0.01,
                accepted_extrema=_accepted_extrema(),
                attempted_extrema=_attempted_extrema(),
                rejection_reasons=Counter(),
            )

        rejected = _accepted_case(proposed)
        rejected["accepted"] = False
        rejected["failed_bits"] = 1
        rejected_metrics = rejected["metrics"]
        assert isinstance(rejected_metrics, dict)
        rejected_metrics["accepted"] = False
        with self.assertRaisesRegex(ValueError, "owns stored rows"):
            _validate_case_record(
                rejected,
                proposed=proposed,
                batch_id=0,
                residual_tolerance=1.0e-8,
                hamiltonian_threshold=1.0e-3,
                production_dt=0.01,
                accepted_extrema=_accepted_extrema(),
                attempted_extrema=_attempted_extrema(),
                rejection_reasons=Counter(),
            )

    def test_case_audit_derives_carrier_period_horizon(self) -> None:
        proposed = _proposed_case()
        case = _accepted_case(proposed)
        metrics = case["metrics"]
        assert isinstance(metrics, dict)
        metrics["intended_terminal_time"] = proposed.intended_terminal_time + 0.08
        with self.assertRaisesRegex(ValueError, "100 carrier periods"):
            _validate_case_record(
                case,
                proposed=proposed,
                batch_id=0,
                residual_tolerance=1.0e-8,
                hamiltonian_threshold=1.0e-3,
                production_dt=0.01,
                accepted_extrema=_accepted_extrema(),
                attempted_extrema=_attempted_extrema(),
                rejection_reasons=Counter(),
            )

    def test_shard_audit_rejects_physical_row_mutations(self) -> None:
        proposed = _proposed_case()
        case = AuditedCase(
            case_id=proposed.case_id,
            accepted=True,
            required_bits=EXPECTED_REQUIRED_BITS,
            evaluated_bits=EXPECTED_EVALUATED_BITS,
            failed_bits=0,
            first_row=0,
            row_count=200,
            batch_id=0,
        )
        dense = select_uniform_times(proposed.saved_time_count, keep_samples=200)
        arrays: dict[str, np.ndarray] = {
            "case_local_index": np.zeros(200, dtype=np.int32),
            "config_fingerprint": np.asarray("config"),
            "depth": np.full(200, proposed.depth, dtype=np.float64),
            "eta": np.zeros((200, 4), dtype=np.float32),
            "frame_index": np.arange(200, dtype=np.int32),
            "gxi": np.zeros((200, 4), dtype=np.float32),
            "proposal_sha256": np.asarray("proposal"),
            "selected_dense_index": dense,
            "time": 0.08 * dense.astype(np.float64),
            "xi": np.zeros((200, 4), dtype=np.float32),
        }

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "shard.npz"

            def validate(values: dict[str, np.ndarray]) -> int:
                np.savez(path, **values)
                return _validate_shard(
                    path,
                    proposal_sha256="proposal",
                    configuration_fingerprint="config",
                    proposed=(proposed,),
                    cases=(case,),
                    delivered_nx=4,
                    saved_dt=0.08,
                    totals=AuditTotals(),
                )

            self.assertEqual(validate(arrays), 200)
            mutations: tuple[tuple[str, str, object], ...] = (
                (
                    "dtype",
                    "wrong dtype",
                    lambda values: values.__setitem__(
                        "selected_dense_index",
                        values["selected_dense_index"].astype(np.int64),
                    ),
                ),
                (
                    "frame",
                    "frame indices",
                    lambda values: values["frame_index"].__setitem__(1, 7),
                ),
                (
                    "dense index",
                    "dense-time selection",
                    lambda values: values["selected_dense_index"].__setitem__(1, 1),
                ),
                (
                    "time",
                    "stored times",
                    lambda values: values["time"].__setitem__(1, 0.123),
                ),
                (
                    "depth",
                    "stored depth",
                    lambda values: values["depth"].__setitem__(1, 4.0),
                ),
                (
                    "bottom",
                    "crosses the bottom",
                    lambda values: values["eta"].__setitem__((1, 0), -5.0),
                ),
            )
            for name, message, mutation in mutations:
                with self.subTest(name=name):
                    changed = {key: value.copy() for key, value in arrays.items()}
                    mutation(changed)
                    with self.assertRaisesRegex((TypeError, ValueError), message):
                        validate(changed)

    def test_map_schema_rejects_noncanonical_dtype(self) -> None:
        arrays = {
            name: (
                np.asarray(2, dtype=dtype)
                if name == "schema_version"
                else np.asarray([], dtype=dtype)
            )
            for name, dtype in TRAJECTORY_MAP_DTYPES.items()
        }
        self.assertEqual(frozenset(arrays), EXPECTED_MAP_ARRAYS)
        _validate_map_array_schema(arrays)
        arrays["trajectory_cell_id"] = arrays["trajectory_cell_id"].astype(np.int64)
        with self.assertRaisesRegex(TypeError, "trajectory_cell_id"):
            _validate_map_array_schema(arrays)

    def test_exact_taxonomy_and_incremental_quotas_fail_closed(self) -> None:
        expected = EXPECTED_CHUNKS[0]
        cell_ids = [cell.cell_id for cell in BENJAMIN_FEIR_POPULATION_CELLS]
        quotas = _expected_chunk_quotas(expected)
        run_spec: dict[str, object] = {
            "cell_codes": {cell_id: index for index, cell_id in enumerate(cell_ids)},
            "configuration": {"ordered_cell_ids": cell_ids},
            "quotas": [
                {"cell_id": cell_id, "target_accepted": target}
                for cell_id, target in quotas.items()
            ],
        }
        mapping, observed = _validate_chunk_taxonomy_and_quotas(run_spec, expected)
        self.assertEqual(mapping[0], cell_ids[0])
        self.assertEqual(observed, quotas)

        wrong_order = deepcopy(run_spec)
        configuration = wrong_order["configuration"]
        assert isinstance(configuration, dict)
        configuration["ordered_cell_ids"] = list(reversed(cell_ids))
        with self.assertRaisesRegex(ValueError, "ordering"):
            _validate_chunk_taxonomy_and_quotas(wrong_order, expected)

        wrong_quota = deepcopy(run_spec)
        raw_quotas = wrong_quota["quotas"]
        assert isinstance(raw_quotas, list)
        assert isinstance(raw_quotas[0], dict)
        raw_quotas[0]["target_accepted"] = int(raw_quotas[0]["target_accepted"]) + 1
        with self.assertRaisesRegex(ValueError, "balanced 66-cell quotas"):
            _validate_chunk_taxonomy_and_quotas(wrong_quota, expected)

    def test_final_split_cell_totals_fail_closed(self) -> None:
        records = [
            {
                "split": expected.split.value,
                "accepted_by_cell": {
                    cell_id: count
                    for cell_id, count in _expected_chunk_quotas(expected).items()
                    if count
                },
            }
            for expected in EXPECTED_CHUNKS
        ]
        totals = _accepted_cell_totals_by_split(records)
        first_cell = BENJAMIN_FEIR_POPULATION_CELLS[0].cell_id
        self.assertEqual(totals["train"][first_cell], 249)
        self.assertEqual(totals["validation"][first_cell], 16)

        mutated = deepcopy(records)
        counts = mutated[0]["accepted_by_cell"]
        assert isinstance(counts, dict)
        counts[first_cell] = int(counts[first_cell]) - 1
        with self.assertRaisesRegex(ValueError, "not balanced"):
            _accepted_cell_totals_by_split(mutated)

    def test_summary_cell_counts_bind_reconstructed_attempts_and_schema(self) -> None:
        quotas = _expected_chunk_quotas(EXPECTED_CHUNKS[0])
        accepted = dict(quotas)
        attempted = {cell_id: target + 2 for cell_id, target in quotas.items()}
        by_cell = {
            cell_id: {
                "target_accepted": target,
                "attempted": attempted[cell_id],
                "accepted": accepted[cell_id],
                "rejected": attempted[cell_id] - accepted[cell_id],
            }
            for cell_id, target in quotas.items()
        }
        summary: dict[str, object] = {
            "counts": {
                "attempted": sum(attempted.values()),
                "accepted": sum(accepted.values()),
                "rejected": sum(attempted.values()) - sum(accepted.values()),
                "by_cell": by_cell,
            }
        }
        _validate_summary_cell_counts(
            summary,
            expected_quotas=quotas,
            attempted_by_cell=attempted,
            accepted_by_cell=accepted,
        )

        cell_ids = tuple(quotas)
        redistributed = deepcopy(summary)
        redistributed_by_cell = redistributed["counts"]["by_cell"]
        assert isinstance(redistributed_by_cell, dict)
        for field, delta in (("attempted", 1), ("rejected", 1)):
            redistributed_by_cell[cell_ids[0]][field] += delta
            redistributed_by_cell[cell_ids[1]][field] -= delta
        with self.assertRaisesRegex(ValueError, "reconstructed transactions"):
            _validate_summary_cell_counts(
                redistributed,
                expected_quotas=quotas,
                attempted_by_cell=attempted,
                accepted_by_cell=accepted,
            )

        for name, mutation in (
            ("missing", lambda cells: cells.pop(cell_ids[0])),
            (
                "extra",
                lambda cells: cells.__setitem__("unexpected", dict(cells[cell_ids[0]])),
            ),
        ):
            with self.subTest(name=name):
                malformed = deepcopy(summary)
                malformed_by_cell = malformed["counts"]["by_cell"]
                assert isinstance(malformed_by_cell, dict)
                mutation(malformed_by_cell)
                with self.assertRaisesRegex(ValueError, "exact 66-cell taxonomy"):
                    _validate_summary_cell_counts(
                        malformed,
                        expected_quotas=quotas,
                        attempted_by_cell=attempted,
                        accepted_by_cell=accepted,
                    )

        extra_field = deepcopy(summary)
        extra_field_by_cell = extra_field["counts"]["by_cell"]
        assert isinstance(extra_field_by_cell, dict)
        extra_field_by_cell[cell_ids[0]]["unbound"] = 0
        with self.assertRaisesRegex(ValueError, "exact count fields"):
            _validate_summary_cell_counts(
                extra_field,
                expected_quotas=quotas,
                attempted_by_cell=attempted,
                accepted_by_cell=accepted,
            )

    def test_historical_shared_source_snapshots_bind_all_six_chunks(self) -> None:
        expected_sources = {
            "solver/gen_data/pipeline/production.py": (
                "production.py",
                "2c2234caf1087c2982872eb44ba2abd457cf22b812784e1a3e9bd54e8c7129d4",
            ),
            "solver/gen_data/trajectory_family_adapters.py": (
                "trajectory_family_adapters.py",
                "afb480a64b14a2b311bda067e6638208569cf3acfebc61d67a1f016a73fbef6a",
            ),
        }
        source_mappings = tuple(
            {path: digest for path, (_name, digest) in expected_sources.items()}
            for _ in EXPECTED_CHUNKS
        )
        source_root = (
            ROOT / "reproducibility/source_snapshots/benjamin_feir_revision4_e16773f"
        )
        record = _historical_shared_source_snapshot_record(source_mappings)
        self.assertEqual(record["chunk_source_maps_checked"], 6)

        forged_mapping = [dict(mapping) for mapping in source_mappings]
        forged_mapping[-1]["solver/gen_data/pipeline/production.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "chunk source map 5"):
            _historical_shared_source_snapshot_record(tuple(forged_mapping))

        with tempfile.TemporaryDirectory() as temporary:
            copied_root = Path(temporary) / "snapshot"
            copied_root.mkdir()
            for name in (
                "SHA256SUMS",
                *(value[0] for value in expected_sources.values()),
            ):
                shutil.copy2(source_root / name, copied_root / name)
            checksums = copied_root / "SHA256SUMS"
            checksums.write_text(
                checksums.read_text(encoding="utf-8").replace(
                    "2c2234caf1087c2982872eb44ba2abd457cf22b812784e1a3e9bd54e8c7129d4",
                    "0" * 64,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "SHA256SUMS differs"):
                _historical_shared_source_snapshot_record(
                    source_mappings,
                    snapshot_root=copied_root,
                )
            shutil.copy2(source_root / "SHA256SUMS", checksums)
            production = copied_root / "production.py"
            production.write_bytes(production.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "snapshot bytes differ"):
                _historical_shared_source_snapshot_record(
                    source_mappings,
                    snapshot_root=copied_root,
                )

    def test_all_nonhistorical_generation_sources_are_physically_bound(self) -> None:
        source_mapping = _frozen_source_mapping()
        source_mappings = tuple(dict(source_mapping) for _ in EXPECTED_CHUNKS)
        record = _nonhistorical_generation_source_record(source_mappings)
        self.assertEqual(record["source_count"], 19)
        self.assertEqual(record["current_source_count"], 17)
        self.assertEqual(len(record["sources"]), 17)

        drifted = tuple(
            {
                **mapping,
                "solver/gen_data/pipeline/archive.py": "0" * 64,
            }
            for mapping in source_mappings
        )
        with self.assertRaisesRegex(ValueError, "current BF source bytes differ"):
            _nonhistorical_generation_source_record(drifted)

        mismatched = [dict(mapping) for mapping in source_mappings]
        mismatched[-1]["solver/gen_data/pipeline/archive.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "exactly equal source maps"):
            _nonhistorical_generation_source_record(tuple(mismatched))

        missing = tuple(
            {
                path: digest
                for path, digest in mapping.items()
                if path != "solver/gen_data/pipeline/archive.py"
            }
            for mapping in source_mappings
        )
        with self.assertRaisesRegex(ValueError, "frozen 19-path contract"):
            _nonhistorical_generation_source_record(missing)

        extra = tuple(
            {**mapping, "solver/gen_data/pipeline/extra.py": "0" * 64}
            for mapping in source_mappings
        )
        with self.assertRaisesRegex(ValueError, "frozen 19-path contract"):
            _nonhistorical_generation_source_record(extra)

        with tempfile.TemporaryDirectory() as temporary:
            copied_repository = Path(temporary) / "repository"
            for repository_path in EXPECTED_GENERATION_SOURCE_PATHS - set(
                HISTORICAL_SHARED_SOURCE_SNAPSHOTS
            ):
                destination = copied_repository / repository_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / repository_path, destination)
            _nonhistorical_generation_source_record(
                source_mappings, repository_root=copied_repository
            )

            target = copied_repository / "solver/gen_data/pipeline/archive.py"
            target.unlink()
            with self.assertRaisesRegex(ValueError, "not a regular repository file"):
                _nonhistorical_generation_source_record(
                    source_mappings, repository_root=copied_repository
                )

            shutil.copy2(ROOT / "solver/gen_data/pipeline/archive.py", target)
            target.unlink()
            target.symlink_to(ROOT / "solver/gen_data/pipeline/archive.py")
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                _nonhistorical_generation_source_record(
                    source_mappings, repository_root=copied_repository
                )


if __name__ == "__main__":
    unittest.main()
