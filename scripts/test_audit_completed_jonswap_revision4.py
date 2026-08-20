"""Focused mutation tests for the JONSWAP revision-4 completion audit."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_completed_jonswap_revision4 import (  # noqa: E402
    AuditTotals,
    AuditedCase,
    CELL_MARGINALS,
    CURRENT_JONSWAP_EXECUTION,
    EXPECTED_ADJUSTMENT_REQUIRED_BITS,
    EXPECTED_CHUNKS,
    EXPECTED_PRODUCTION_EVALUATED_BITS,
    EXPECTED_PRODUCTION_METRIC_KEYS,
    EXPECTED_PRODUCTION_REQUIRED_BITS,
    Extrema,
    PROPOSAL_LAW_APPLIES_TO,
    RELEASED_CASE_LAW,
    ReplayedSpecification,
    ROWS_PER_ACCEPTED_CASE,
    TrajectoryQualityDiagnostics,
    _case_specifications,
    _current_support_record,
    _floored_saved_count,
    _field_quality_metrics,
    _peak_period,
    _population_conditioning_record,
    _validate_case_record,
    _validate_dataset_view,
    _validate_shard_rows,
)
from scripts.build_paper_corpus_view import CompletedChunk  # noqa: E402
from solver.gen_data.jonswap_tma_population import (  # noqa: E402
    JONSWAP_TMA_POPULATION_CELLS,
)
from solver.gen_data.pipeline.production import (  # noqa: E402
    AttemptAssignment,
    CaseKey,
    SplitId,
)
from solver.gen_data.pipeline.archive import BatchPaths, file_sha256  # noqa: E402
from solver.gen_data.pipeline.quota_driver import (  # noqa: E402
    canonical_json_sha256,
)
from solver.gen_data.pipeline.quality import QualityReason  # noqa: E402
from solver.gen_data.pipeline.time_selection import select_uniform_times  # noqa: E402
from solver.gen_data.trajectory_family_adapters import (  # noqa: E402
    sample_jonswap_tma_trajectory_cases,
)


def _support_extrema() -> dict[str, Extrema]:
    return {
        name: Extrema()
        for name in (
            "depth",
            "significant_height",
            "peak_wavenumber",
            "peak_enhancement",
            "right_moving_fraction",
            "depth_wavenumber",
            "relative_height",
            "peak_steepness",
            "resolved_maximum_relative_frequency",
            "phase_count_per_direction",
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
            "adjustment_stage_residual",
            "adjustment_minimum_water_column",
        )
    }


def _attempted_extrema() -> dict[str, Extrema]:
    return {
        name: Extrema()
        for name in (
            "hamiltonian_drift",
            "stage_residual",
            "minimum_water_column",
            "adjustment_stage_residual",
            "adjustment_minimum_water_column",
        )
    }


def _conditioning_chunk(*, rejected_cell_index: int | None = None) -> dict[str, object]:
    cells: dict[str, object] = {}
    for index, cell in enumerate(JONSWAP_TMA_POPULATION_CELLS):
        rejected = int(index == rejected_cell_index)
        cells[cell.cell_id] = {
            "target_accepted": 1,
            "attempted": 1 + rejected,
            "accepted": 1,
            "rejected": rejected,
            "rejection_rate": rejected / (1 + rejected),
        }
    return {"cell_counts": cells}


def _sampled_specification() -> tuple[CaseKey, dict[str, object]]:
    key = CaseKey(
        family_id=4,
        revision_id=4,
        split_id=SplitId.TRAIN,
        stream_id=0,
        attempt_index=0,
    )
    assignment = AttemptAssignment(
        case_key=key,
        cell_id=JONSWAP_TMA_POPULATION_CELLS[0].cell_id,
    )
    sampled = sample_jonswap_tma_trajectory_cases(
        (assignment,),
        contract=CURRENT_JONSWAP_EXECUTION.numerical,
        quadrature_order=int(CURRENT_JONSWAP_EXECUTION.jonswap_quadrature_order),
    )
    return key, dict(sampled.specification_records[0])


def _horizon_metrics(specification: dict[str, object]) -> dict[str, object]:
    period = _peak_period(specification)
    saved_dt = CURRENT_JONSWAP_EXECUTION.numerical.saved_dt
    adjustment = CURRENT_JONSWAP_EXECUTION.jonswap_adjustment
    period_count = CURRENT_JONSWAP_EXECUTION.horizon.period_count
    assert adjustment is not None and period_count is not None
    production_intended = period_count * period
    production_count = _floored_saved_count(production_intended, saved_dt)
    adjustment_intended = adjustment.burn_peak_periods * period
    adjustment_count = _floored_saved_count(adjustment_intended, saved_dt)
    return {
        "intended_terminal_time": production_intended,
        "realized_terminal_time": (production_count - 1) * saved_dt,
        "saved_time_count": production_count,
        "nonlinear_adjustment_intended_terminal_time": adjustment_intended,
        "nonlinear_adjustment_realized_terminal_time": (adjustment_count - 1)
        * saved_dt,
        "nonlinear_adjustment_saved_time_count": adjustment_count,
        "nonlinear_adjustment_ramp_time": (adjustment.ramp_time_peak_periods * period),
    }


def _accepted_case(specification: dict[str, object]) -> dict[str, object]:
    metrics: dict[str, object] = {
        "accepted": True,
        "all_stages_solved": True,
        "complete_admissible_trajectory": True,
        "initial_discrete_peak_wavenumber": float(specification["peak_wavenumber"]),
        "initial_eta_rms": 0.001,
        "initial_expected_linear_hamiltonian": 2.0e-5,
        "initial_half_maximum_spectral_cell_count": 8,
        "initial_internal_hamiltonian": 2.01e-5,
        "initial_linear_hamiltonian": 2.0e-5,
        "initial_linear_hamiltonian_relative_error": 1.0e-15,
        "initial_minimum_water_column": 0.02,
        "initial_realized_height_ratio": 1.0,
        "initial_xi_rms": 0.0005,
        "internal_dno_finite": True,
        "internal_hamiltonian_drift_threshold": 1.0e-3,
        "internal_health_evaluated": True,
        "internal_state_finite": True,
        "maximum_internal_hamiltonian_drift": 1.0e-5,
        "maximum_stage_residual": 1.0e-9,
        "minimum_internal_water_column": 0.02,
        "nonlinear_adjustment_accepted": True,
        "nonlinear_adjustment_all_stages_solved": True,
        "nonlinear_adjustment_complete_admissible_handoff": True,
        "nonlinear_adjustment_maximum_stage_residual": 2.0e-9,
        "nonlinear_adjustment_minimum_water_column": 0.019,
        "nonlinear_adjustment_positive_water_column": True,
        "nonlinear_adjustment_production_ramp": "disabled",
        "nonlinear_adjustment_ramp_order": 4,
        "nonlinear_adjustment_schema": "dommermuth_nonlinear_adjustment_v1",
        "nonlinear_adjustment_state_finite": True,
        "positive_water_column": True,
        "production_dt": 0.01,
        "production_status": "completed",
        "state_finite": True,
        "target_finite": True,
        **_horizon_metrics(specification),
    }
    assert set(metrics) == EXPECTED_PRODUCTION_METRIC_KEYS
    key, _ = _sampled_specification()
    return {
        "accepted": True,
        "case_id": key.case_id,
        "required_bits": EXPECTED_PRODUCTION_REQUIRED_BITS,
        "evaluated_bits": EXPECTED_PRODUCTION_EVALUATED_BITS,
        "failed_bits": 0,
        "first_row": 0,
        "row_count": ROWS_PER_ACCEPTED_CASE,
        "metrics": metrics,
    }


def _replayed(specification: dict[str, object]) -> ReplayedSpecification:
    key, _ = _sampled_specification()
    return ReplayedSpecification(
        case_id=key.case_id,
        attempt_index=0,
        cell_code=0,
        cell_id=JONSWAP_TMA_POPULATION_CELLS[0].cell_id,
        record=specification,
    )


def _audit_case(
    case: dict[str, object],
    specification: dict[str, object],
) -> AuditedCase:
    return _validate_case_record(
        case,
        batch_id=0,
        local_index=0,
        specification=_replayed(specification),
        residual_tolerance=1.0e-8,
        hamiltonian_threshold=1.0e-3,
        production_dt=0.01,
        saved_dt=0.08,
        accepted_extrema=_accepted_extrema(),
        attempted_extrema=_attempted_extrema(),
        rejection_reasons=Counter(),
        totals=AuditTotals(),
    )


@dataclass
class _ViewFixture:
    chunk: CompletedChunk
    summary: dict[str, object]
    manifest: dict[str, object]
    arrays: dict[str, np.ndarray]
    manifest_path: Path
    map_path: Path
    specifications_by_batch: dict[int, tuple[ReplayedSpecification, ...]]
    cases_by_batch: dict[int, tuple[AuditedCase, ...]]
    result_by_batch: dict[int, dict[str, object]]

    def commit(self) -> None:
        np.savez(self.map_path, **self.arrays)
        self.manifest["trajectory_map_sha256"] = file_sha256(self.map_path)
        contract = self.manifest["dataset_contract"]
        self.manifest["dataset_contract_fingerprint"] = canonical_json_sha256(contract)
        self.manifest_path.write_text(
            json.dumps(self.manifest, allow_nan=False),
            encoding="utf-8",
        )
        view = self.summary["dataset_view"]
        assert isinstance(view, dict)
        view["manifest"] = {
            "path": str(self.manifest_path.relative_to(self.chunk.root)),
            "bytes": self.manifest_path.stat().st_size,
            "sha256": file_sha256(self.manifest_path),
        }
        view["trajectory_map"] = {
            "path": str(self.map_path.relative_to(self.chunk.root)),
            "bytes": self.map_path.stat().st_size,
            "sha256": file_sha256(self.map_path),
        }


def _view_fixture(root: Path) -> _ViewFixture:
    fingerprint = "f" * 64
    source_sha256 = {"solver/source.py": "a" * 64}
    execution = CURRENT_JONSWAP_EXECUTION.to_json_record()
    execution_fingerprint = canonical_json_sha256(execution)
    source_fingerprint = canonical_json_sha256(source_sha256)
    dependency_fingerprint = "d" * 64
    paths = BatchPaths.under(
        root,
        family="jonswap_tma",
        split="train",
        batch_id=0,
    )
    for path in (paths.proposal, paths.shard, paths.result):
        path.parent.mkdir(parents=True, exist_ok=True)
    paths.proposal.write_bytes(b"proposal")
    paths.shard.write_bytes(b"shard")
    paths.result.write_text("{}", encoding="utf-8")
    chunk = CompletedChunk(
        summary_path=root / "summary.json",
        summary_sha256="b" * 64,
        root=root,
        family="jonswap_tma",
        revision_id=4,
        split=SplitId.TRAIN,
        stream_id=0,
        accepted_before=0,
        accepted_count=1,
        accepted_after=1,
        attempted_count=1,
        fingerprint=fingerprint,
        dependency_fingerprint=dependency_fingerprint,
        execution_fingerprint=execution_fingerprint,
        generation_compatibility_id=None,
        source_fingerprint=source_fingerprint,
        source_sha256=source_sha256,
        execution_platform="gpu",
        batches=(paths,),
    )
    key, specification_record = _sampled_specification()
    specification = ReplayedSpecification(
        case_id=key.case_id,
        attempt_index=0,
        cell_code=0,
        cell_id=JONSWAP_TMA_POPULATION_CELLS[0].cell_id,
        record=specification_record,
    )
    case = AuditedCase(
        case_id=key.case_id,
        accepted=True,
        required_bits=EXPECTED_PRODUCTION_REQUIRED_BITS,
        evaluated_bits=EXPECTED_PRODUCTION_EVALUATED_BITS,
        failed_bits=0,
        first_row=0,
        row_count=ROWS_PER_ACCEPTED_CASE,
        batch_id=0,
    )
    numerical = execution["numerical"]
    assert isinstance(numerical, dict)
    target = {
        "role": execution["role"],
        "nx": numerical.get("target_nx", numerical["nx"]),
        "length": numerical["length"],
        "gravity": numerical["gravity"],
        "dno_order": numerical.get("target_dno_order", numerical["dno_order"]),
        "pad_factor": numerical["pad_factor"],
        "maximum_wavenumber": numerical.get(
            "target_maximum_wavenumber",
            numerical["maximum_wavenumber"],
        ),
        "dtype": numerical["dtype"],
    }
    generation_identity: dict[str, object] = {
        "dependency_environment_fingerprint": dependency_fingerprint,
        "family_revisions": [
            {
                "family_id": 4,
                "revision_id": 4,
                "execution_platform": "gpu",
                "generation_variants": [
                    {
                        "execution_record_fingerprint": execution_fingerprint,
                        "source_sha256_fingerprint": source_fingerprint,
                        "source_sha256": source_sha256,
                    }
                ],
                "source_sha256_fingerprint": source_fingerprint,
            }
        ],
    }
    generation_identity["compatibility_fingerprint"] = canonical_json_sha256(
        generation_identity
    )
    contract = {
        "target": target,
        "trajectory_numerical": numerical,
        "trajectory_numerical_by_family_revision": [
            {"family_id": 4, "revision_id": 4, "numerical": numerical}
        ],
        "stored_dtypes": {
            "eta": "float32",
            "xi": "float32",
            "gxi": "float32",
            "depth": "float64",
            "time": "float64",
        },
        "whole_case_rows": True,
        "generation_identity": generation_identity,
    }
    map_path = root / "view.trajectory_map.npz"
    manifest_path = root / "view.dataset.json"
    result = {
        "proposal_sha256": file_sha256(paths.proposal),
        "shard_sha256": file_sha256(paths.shard),
    }
    manifest: dict[str, object] = {
        "schema_version": 2,
        "configuration_fingerprint": fingerprint,
        "configuration_fingerprints": [fingerprint],
        "dataset_contract": contract,
        "dataset_contract_fingerprint": canonical_json_sha256(contract),
        "dataset_batches": [
            {
                "proposal_path": str(paths.proposal.relative_to(root)),
                "proposal_sha256": result["proposal_sha256"],
                "result_path": str(paths.result.relative_to(root)),
                "result_sha256": file_sha256(paths.result),
                "shard_index": 0,
                "configuration_fingerprint": fingerprint,
                "family_id": 4,
                "revision_id": 4,
                "split_id": 0,
                "batch_id": 0,
                "n_attempted_trajectories": 1,
                "n_accepted_trajectories": 1,
                "n_rows": ROWS_PER_ACCEPTED_CASE,
            }
        ],
        "dataset_shards": [
            {
                "path": str(paths.shard.relative_to(root)),
                "sha256": result["shard_sha256"],
                "n_rows": ROWS_PER_ACCEPTED_CASE,
                "batch_index": 0,
                "configuration_fingerprint": fingerprint,
            }
        ],
        "trajectory_map_npz": map_path.name,
        "trajectory_map_sha256": "",
        "requires_trajectory_map": True,
        "n_rows": ROWS_PER_ACCEPTED_CASE,
        "n_trajectories": 1,
        "n_accepted_trajectories": 1,
        "n_accepted_rows": ROWS_PER_ACCEPTED_CASE,
        "grid": {
            "length": CURRENT_JONSWAP_EXECUTION.numerical.length,
            "nx": CURRENT_JONSWAP_EXECUTION.numerical.delivered_nx,
        },
        "split_counts": {
            "train": {"attempted": 1, "accepted": 1},
            "validation": {"attempted": 0, "accepted": 0},
            "test": {"attempted": 0, "accepted": 0},
        },
    }
    arrays = {
        "schema_version": np.asarray(2, dtype=np.int16),
        "trajectory_index": np.zeros(ROWS_PER_ACCEPTED_CASE, dtype=np.int32),
        "frame_index": np.arange(ROWS_PER_ACCEPTED_CASE, dtype=np.int32),
        "shard_index": np.zeros(ROWS_PER_ACCEPTED_CASE, dtype=np.int32),
        "shard_row": np.arange(ROWS_PER_ACCEPTED_CASE, dtype=np.int64),
        "trajectory_family_id": np.asarray([4], dtype=np.int16),
        "trajectory_revision_id": np.asarray([4], dtype=np.int16),
        "trajectory_split_id": np.asarray([0], dtype=np.uint8),
        "trajectory_case_id": np.asarray([key.case_id], dtype=np.int64),
        "trajectory_cell_id": np.asarray([0], dtype=np.int32),
        "trajectory_accepted": np.asarray([True], dtype=np.bool_),
        "trajectory_required_bits": np.asarray(
            [EXPECTED_PRODUCTION_REQUIRED_BITS],
            dtype=np.uint32,
        ),
        "trajectory_evaluated_bits": np.asarray(
            [EXPECTED_PRODUCTION_EVALUATED_BITS],
            dtype=np.uint32,
        ),
        "trajectory_failed_bits": np.asarray([0], dtype=np.uint32),
        "trajectory_first_row": np.asarray([0], dtype=np.int64),
        "trajectory_row_count": np.asarray(
            [ROWS_PER_ACCEPTED_CASE],
            dtype=np.int32,
        ),
    }
    summary: dict[str, object] = {
        "execution": execution,
        "dataset_view": {
            "schema_version": 2,
            "configuration_fingerprint": fingerprint,
            "grid": manifest["grid"],
            "n_rows": ROWS_PER_ACCEPTED_CASE,
            "n_trajectories": 1,
            "n_accepted_trajectories": 1,
            "n_accepted_rows": ROWS_PER_ACCEPTED_CASE,
            "manifest": {},
            "trajectory_map": {},
        },
    }
    fixture = _ViewFixture(
        chunk=chunk,
        summary=summary,
        manifest=manifest,
        arrays=arrays,
        manifest_path=manifest_path,
        map_path=map_path,
        specifications_by_batch={0: (specification,)},
        cases_by_batch={0: (case,)},
        result_by_batch={0: result},
    )
    fixture.commit()
    return fixture


class JonswapCompletionAuditTests(unittest.TestCase):
    def test_population_conditioning_counts_and_semantics_fail_closed(self) -> None:
        chunks = (
            _conditioning_chunk(rejected_cell_index=0),
            _conditioning_chunk(),
        )
        record = _population_conditioning_record(
            chunks,
            attempted=55,
            accepted=54,
            rejected=1,
        )
        self.assertEqual(record["proposal_law_applies_to"], PROPOSAL_LAW_APPLIES_TO)
        self.assertEqual(record["released_case_law"], RELEASED_CASE_LAW)
        self.assertEqual(record["cell_marginals"], CELL_MARGINALS)
        self.assertIs(record["posthoc_parameter_gate"], False)
        cells = record["cells"]
        assert isinstance(cells, dict)
        self.assertEqual(
            set(cells), {cell.cell_id for cell in JONSWAP_TMA_POPULATION_CELLS}
        )
        first = cells[JONSWAP_TMA_POPULATION_CELLS[0].cell_id]
        self.assertEqual(
            first,
            {
                "target_accepted": 2,
                "attempted": 3,
                "accepted": 2,
                "rejected": 1,
                "rejection_rate": 1 / 3,
            },
        )

        malformed = _conditioning_chunk()
        malformed_cells = malformed["cell_counts"]
        assert isinstance(malformed_cells, dict)
        malformed_cells.pop(JONSWAP_TMA_POPULATION_CELLS[-1].cell_id)
        with self.assertRaisesRegex(ValueError, "exact 27 JONSWAP cells"):
            _population_conditioning_record(
                (malformed,),
                attempted=26,
                accepted=26,
                rejected=0,
            )

        malformed = _conditioning_chunk()
        malformed_cells = malformed["cell_counts"]
        assert isinstance(malformed_cells, dict)
        first = malformed_cells[JONSWAP_TMA_POPULATION_CELLS[0].cell_id]
        assert isinstance(first, dict)
        first["attempted"] = 2
        with self.assertRaisesRegex(ValueError, "counts do not close"):
            _population_conditioning_record(
                (malformed,),
                attempted=28,
                accepted=27,
                rejected=1,
            )

        malformed = _conditioning_chunk()
        malformed_cells = malformed["cell_counts"]
        assert isinstance(malformed_cells, dict)
        first = malformed_cells[JONSWAP_TMA_POPULATION_CELLS[0].cell_id]
        assert isinstance(first, dict)
        first["target_accepted"] = 2
        with self.assertRaisesRegex(ValueError, "does not meet its quota"):
            _population_conditioning_record(
                (malformed,),
                attempted=27,
                accepted=27,
                rejected=0,
            )

        with self.assertRaisesRegex(ValueError, "accepted counts differ"):
            _population_conditioning_record(
                (_conditioning_chunk(),),
                attempted=27,
                accepted=26,
                rejected=1,
            )

    def test_field_quality_metrics_have_known_fourier_values(self) -> None:
        x = np.linspace(0.0, 2.0 * np.pi, 1024, endpoint=False)
        fields = np.stack(
            (
                np.zeros_like(x),
                np.sin(4.0 * x),
                np.sin(100.0 * x),
            )
        )
        metrics = _field_quality_metrics(fields)
        np.testing.assert_allclose(
            metrics["effective_wavenumber"],
            np.asarray([0.0, 4.0, 100.0]),
            rtol=0.0,
            atol=1.0e-10,
        )
        np.testing.assert_array_equal(
            metrics["cyclic_difference_sign_flips"],
            np.asarray([0, 8, 200]),
        )
        np.testing.assert_allclose(
            metrics["high_band_energy_fraction"],
            np.asarray([0.0, 0.0, 1.0]),
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_exact_replay_rejects_phase_and_relative_band_mutations(self) -> None:
        key, record = _sampled_specification()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "proposal.npz"

            def write(specification: dict[str, object]) -> None:
                np.savez(
                    path,
                    case_id=np.asarray([key.case_id], dtype=np.int64),
                    family_id=np.asarray(4, dtype=np.int16),
                    revision_id=np.asarray(4, dtype=np.int16),
                    split_id=np.asarray(0, dtype=np.uint8),
                    root_seed=np.asarray([key.root_seed], dtype=np.uint64),
                    stream_id=np.asarray([0], dtype=np.uint32),
                    attempt_index=np.asarray([0], dtype=np.uint64),
                    cell_id=np.asarray([0], dtype=np.int32),
                    case_spec_json=np.asarray(
                        [json.dumps(specification, allow_nan=False)]
                    ),
                )

            write(record)
            totals = AuditTotals()
            replayed = _case_specifications(
                path,
                split=SplitId.TRAIN,
                stream_id=0,
                cell_ids_by_code={0: JONSWAP_TMA_POPULATION_CELLS[0].cell_id},
                support_extrema=_support_extrema(),
                totals=totals,
            )
            self.assertEqual(len(replayed), 1)
            self.assertEqual(totals.proposal_specs_checked, 1)

            phase_mutation = json.loads(json.dumps(record))
            phase_right = phase_mutation["phase_right"]
            assert isinstance(phase_right, list)
            phase_right[0] = float(phase_right[0]) + 0.125
            write(phase_mutation)
            with self.assertRaisesRegex(ValueError, "deterministic sampling"):
                _case_specifications(
                    path,
                    split=SplitId.TRAIN,
                    stream_id=0,
                    cell_ids_by_code={0: JONSWAP_TMA_POPULATION_CELLS[0].cell_id},
                    support_extrema=_support_extrema(),
                    totals=AuditTotals(),
                )

            band_mutation = json.loads(json.dumps(record))
            settings = band_mutation["constructor_settings"]
            assert isinstance(settings, dict)
            settings["relative_frequency_maximum"] = 2.4
            write(band_mutation)
            with self.assertRaisesRegex(ValueError, "deterministic sampling"):
                _case_specifications(
                    path,
                    split=SplitId.TRAIN,
                    stream_id=0,
                    cell_ids_by_code={0: JONSWAP_TMA_POPULATION_CELLS[0].cell_id},
                    support_extrema=_support_extrema(),
                    totals=AuditTotals(),
                )

    def test_accepted_case_rejects_adjustment_autonomous_and_horizon_mutations(
        self,
    ) -> None:
        _, specification = _sampled_specification()
        case = _accepted_case(specification)
        self.assertTrue(_audit_case(case, specification).accepted)

        mutations = (
            (
                "nonlinear_adjustment_maximum_stage_residual",
                1.1e-8,
                "accepted adjustment",
            ),
            (
                "maximum_internal_hamiltonian_drift",
                1.1e-3,
                "Hamiltonian threshold",
            ),
            (
                "intended_terminal_time",
                float(case["metrics"]["intended_terminal_time"]) + 0.08,  # type: ignore[index]
                "intended horizon",
            ),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                mutated = json.loads(json.dumps(case))
                metrics = mutated["metrics"]
                assert isinstance(metrics, dict)
                metrics[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    _audit_case(mutated, specification)

    def test_adjustment_failure_uses_exact_short_schema_and_no_rows(self) -> None:
        _, specification = _sampled_specification()
        accepted = _accepted_case(specification)
        metrics = accepted["metrics"]
        assert isinstance(metrics, dict)
        production_only = EXPECTED_PRODUCTION_METRIC_KEYS.difference(
            {
                "initial_discrete_peak_wavenumber",
                "initial_eta_rms",
                "initial_expected_linear_hamiltonian",
                "initial_half_maximum_spectral_cell_count",
                "initial_linear_hamiltonian",
                "initial_linear_hamiltonian_relative_error",
                "initial_minimum_water_column",
                "initial_realized_height_ratio",
                "initial_xi_rms",
                "intended_terminal_time",
                "nonlinear_adjustment_accepted",
                "nonlinear_adjustment_all_stages_solved",
                "nonlinear_adjustment_complete_admissible_handoff",
                "nonlinear_adjustment_intended_terminal_time",
                "nonlinear_adjustment_maximum_stage_residual",
                "nonlinear_adjustment_minimum_water_column",
                "nonlinear_adjustment_positive_water_column",
                "nonlinear_adjustment_production_ramp",
                "nonlinear_adjustment_ramp_order",
                "nonlinear_adjustment_ramp_time",
                "nonlinear_adjustment_realized_terminal_time",
                "nonlinear_adjustment_saved_time_count",
                "nonlinear_adjustment_schema",
                "nonlinear_adjustment_state_finite",
                "production_status",
                "realized_terminal_time",
                "saved_time_count",
            }
        )
        for field in production_only:
            metrics.pop(field)
        metrics.update(
            {
                "nonlinear_adjustment_accepted": False,
                "nonlinear_adjustment_all_stages_solved": False,
                "nonlinear_adjustment_complete_admissible_handoff": False,
                "nonlinear_adjustment_maximum_stage_residual": 2.0e-8,
                "production_status": "not_run_adjustment_failed",
            }
        )
        accepted.update(
            {
                "accepted": False,
                "required_bits": EXPECTED_ADJUSTMENT_REQUIRED_BITS,
                "evaluated_bits": int(
                    QualityReason.OUTSIDE_SUPPORT
                    | QualityReason.INCOMPLETE_TRAJECTORY
                    | QualityReason.GL2_STAGE_RESIDUAL
                    | QualityReason.NONFINITE_STATE
                    | QualityReason.BOTTOM_CLEARANCE
                ),
                "failed_bits": int(
                    QualityReason.GL2_STAGE_RESIDUAL
                    | QualityReason.INCOMPLETE_TRAJECTORY
                ),
                "first_row": -1,
                "row_count": 0,
            }
        )
        self.assertFalse(_audit_case(accepted, specification).accepted)
        accepted["row_count"] = 16
        with self.assertRaisesRegex(ValueError, "owns stored rows"):
            _audit_case(accepted, specification)

    def test_stored_rows_require_finite_complete_uniform_endpoints(self) -> None:
        _, specification = _sampled_specification()
        raw_case = _accepted_case(specification)
        audited = _audit_case(raw_case, specification)
        metrics = raw_case["metrics"]
        assert isinstance(metrics, dict)
        dense_count = int(metrics["saved_time_count"])
        indices = select_uniform_times(
            dense_count,
            keep_samples=ROWS_PER_ACCEPTED_CASE,
        )
        times = 0.08 * indices.astype(np.float64)
        depth = float(specification["depth"])
        arrays: dict[str, np.ndarray] = {
            "case_local_index": np.zeros(16, dtype=np.int32),
            "config_fingerprint": np.asarray("f" * 64),
            "depth": np.full(16, depth, dtype=np.float64),
            "eta": np.zeros((16, 1024), dtype=np.float32),
            "frame_index": np.arange(16, dtype=np.int32),
            "gxi": np.zeros((16, 1024), dtype=np.float32),
            "proposal_sha256": np.asarray("a" * 64),
            "selected_dense_index": indices,
            "time": times,
            "xi": np.zeros((16, 1024), dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "shard.npz"

            def write() -> None:
                np.savez(path, **arrays)

            write()
            totals = AuditTotals()
            diagnostics = TrajectoryQualityDiagnostics()
            _validate_shard_rows(
                path,
                shard_artifact_path="shards/train/shard.npz",
                chunk_label="test_chunk",
                split="train",
                proposal_sha256="a" * 64,
                configuration_fingerprint="f" * 64,
                specifications=(_replayed(specification),),
                raw_cases=(raw_case,),
                cases=(audited,),
                saved_dt=0.08,
                totals=totals,
                diagnostics=diagnostics,
            )
            self.assertEqual(totals.stored_case_blocks_checked, 1)

            x = np.linspace(0.0, 2.0 * np.pi, 1024, endpoint=False)
            arrays["gxi"] = np.repeat(
                np.sin(4.0 * x, dtype=np.float64)[None, :],
                ROWS_PER_ACCEPTED_CASE,
                axis=0,
            ).astype(np.float32)
            arrays["gxi"][7] = np.sin(100.0 * x, dtype=np.float64)
            write()
            high_tail_diagnostics = TrajectoryQualityDiagnostics()
            _validate_shard_rows(
                path,
                shard_artifact_path="shards/train/shard.npz",
                chunk_label="test_chunk",
                split="train",
                proposal_sha256="a" * 64,
                configuration_fingerprint="f" * 64,
                specifications=(_replayed(specification),),
                raw_cases=(raw_case,),
                cases=(audited,),
                saved_dt=0.08,
                totals=AuditTotals(),
                diagnostics=high_tail_diagnostics,
            )
            diagnostic_record = high_tail_diagnostics.record()
            self.assertFalse(
                diagnostic_record["metric_values_affect_acceptance_or_audit_status"]
            )
            self.assertTrue(
                diagnostic_record["diagnostic_coverage_required_for_audit_completion"]
            )
            metrics = diagnostic_record["metrics"]
            assert isinstance(metrics, dict)
            peak = metrics["gxi_high_band_energy_fraction_maximum"]
            assert isinstance(peak, dict)
            maximum = peak["maximum"]
            assert isinstance(maximum, dict)
            identity = maximum["identity"]
            assert isinstance(identity, dict)
            self.assertEqual(identity["shard_path"], "shards/train/shard.npz")
            self.assertEqual(identity["frame_index"], 7)
            tail_counts = diagnostic_record["tail_counts"]
            assert isinstance(tail_counts, dict)
            self.assertEqual(
                tail_counts["peak_gxi_high_band_energy_fraction_gt_0p10"],
                1,
            )
            self.assertEqual(
                tail_counts["peak_gxi_high_band_energy_fraction_gt_0p20"],
                1,
            )

            arrays["gxi"] = np.zeros((16, 1024), dtype=np.float32)
            arrays["gxi"][0, 0] = math.nan
            write()
            with self.assertRaisesRegex(ValueError, "must all be finite"):
                _validate_shard_rows(
                    path,
                    shard_artifact_path="shards/train/shard.npz",
                    chunk_label="test_chunk",
                    split="train",
                    proposal_sha256="a" * 64,
                    configuration_fingerprint="f" * 64,
                    specifications=(_replayed(specification),),
                    raw_cases=(raw_case,),
                    cases=(audited,),
                    saved_dt=0.08,
                    totals=AuditTotals(),
                    diagnostics=TrajectoryQualityDiagnostics(),
                )

            arrays["gxi"] = arrays["gxi"].astype(np.float64)
            write()
            with self.assertRaisesRegex(TypeError, "gxi has the wrong dtype"):
                _validate_shard_rows(
                    path,
                    shard_artifact_path="shards/train/shard.npz",
                    chunk_label="test_chunk",
                    split="train",
                    proposal_sha256="a" * 64,
                    configuration_fingerprint="f" * 64,
                    specifications=(_replayed(specification),),
                    raw_cases=(raw_case,),
                    cases=(audited,),
                    saved_dt=0.08,
                    totals=AuditTotals(),
                    diagnostics=TrajectoryQualityDiagnostics(),
                )

            arrays["gxi"] = np.zeros((16, 1024), dtype=np.float32)
            rejected = replace(audited, accepted=False, first_row=-1, row_count=0)
            write()
            with self.assertRaisesRegex(ValueError, "rejected.*owns stored rows"):
                _validate_shard_rows(
                    path,
                    shard_artifact_path="shards/train/shard.npz",
                    chunk_label="test_chunk",
                    split="train",
                    proposal_sha256="a" * 64,
                    configuration_fingerprint="f" * 64,
                    specifications=(_replayed(specification),),
                    raw_cases=(raw_case,),
                    cases=(rejected,),
                    saved_dt=0.08,
                    totals=AuditTotals(),
                    diagnostics=TrajectoryQualityDiagnostics(),
                )

    def test_dataset_view_rejects_every_previously_unbound_source_field(
        self,
    ) -> None:
        def validate(fixture: _ViewFixture) -> None:
            _validate_dataset_view(
                chunk=fixture.chunk,
                summary=fixture.summary,
                specifications_by_batch=fixture.specifications_by_batch,
                cases_by_batch=fixture.cases_by_batch,
                result_by_batch=fixture.result_by_batch,
                totals=AuditTotals(),
            )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = _view_fixture(Path(temporary) / "baseline")
            validate(fixture)

        mutations = (
            (
                "cell_id",
                lambda fixture: fixture.arrays.__setitem__(
                    "trajectory_cell_id",
                    np.asarray([1], dtype=np.int32),
                ),
                "trajectory_cell_id",
            ),
            (
                "extra_row",
                lambda fixture: fixture.arrays.__setitem__(
                    "trajectory_index",
                    np.zeros(ROWS_PER_ACCEPTED_CASE + 1, dtype=np.int32),
                ),
                "malformed row vector",
            ),
            (
                "map_dtype",
                lambda fixture: fixture.arrays.__setitem__(
                    "frame_index",
                    np.arange(ROWS_PER_ACCEPTED_CASE, dtype=np.int64),
                ),
                "frame_index has the wrong dtype",
            ),
            (
                "forged_contract",
                lambda fixture: fixture.manifest["dataset_contract"].update(  # type: ignore[union-attr]
                    {"whole_case_rows": False}
                ),
                "contract differs",
            ),
            (
                "wrong_source_path",
                lambda fixture: fixture.manifest["dataset_batches"][0].update(  # type: ignore[index,union-attr]
                    {"proposal_path": "shards/jonswap_tma/train/batch_000000.npz"}
                ),
                "proposal_path",
            ),
        )
        for label, mutate, message in mutations:
            with (
                self.subTest(mutation=label),
                tempfile.TemporaryDirectory() as temporary,
            ):
                fixture = _view_fixture(Path(temporary) / label)
                mutate(fixture)
                fixture.commit()
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    validate(fixture)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = _view_fixture(Path(temporary) / "unsafe_path")
            view = fixture.summary["dataset_view"]
            assert isinstance(view, dict)
            manifest = view["manifest"]
            assert isinstance(manifest, dict)
            manifest["path"] = "../outside.dataset.json"
            with self.assertRaisesRegex(ValueError, "escapes its chunk root"):
                validate(fixture)

    def test_release_plan_and_support_are_exactly_current(self) -> None:
        observed = tuple(
            (
                chunk.split.value,
                chunk.accepted_before,
                chunk.accepted_count,
                chunk.stream_id,
            )
            for chunk in EXPECTED_CHUNKS
        )
        self.assertEqual(
            observed,
            (
                ("train", 0, 2048, 0),
                ("train", 2048, 2048, 1),
                ("train", 4096, 4096, 2),
                ("train", 8192, 4096, 3),
                ("train", 12288, 2048, 4),
                ("train", 14336, 2048, 5),
                ("validation", 0, 1024, 100),
                ("test", 0, 1024, 200),
            ),
        )
        support = _current_support_record()
        self.assertEqual(len(support["allocation_cells"]), 27)  # type: ignore[arg-type]
        self.assertEqual(support["relative_frequency_interval"], [0.5, 2.5])


if __name__ == "__main__":
    unittest.main()
