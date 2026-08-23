"""Fail-closed CPU audit of the completed JONSWAP/TMA revision-4 dataset.

The quota scanner remains the authority for transaction replay.  This audit
adds dataset-wide interval and identity checks, exact replay of every proposed
parameter and phase specification, numerical-health extrema for the nonlinear
adjustment and autonomous production portions, stored-field finiteness, and an
independent check of each loader-facing trajectory map.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importing the paper contract initializes JAX.  An audit must never claim a
# GPU or perturb live generation jobs.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-jonswap-completion-audit")

import numpy as np  # noqa: E402

from scripts.build_paper_dataset_view import (  # noqa: E402
    CompletedChunk,
    TRAJECTORY_MAP_DTYPES,
    load_completed_chunk,
)
from scripts.run_paper_dataset_jonswap_bucketed import (  # noqa: E402
    EXTRA_SOURCE_PATHS,
)
from scripts.run_paper_dataset_quota import (  # noqa: E402
    dependency_environment,
    source_hashes,
)
from solver.gen_data.jonswap_horizon_executor import (  # noqa: E402
    POLICY_KEY,
    policy_record,
)
from solver.gen_data.jonswap_tma import (  # noqa: E402
    PAPER_PEAK_ENHANCEMENTS,
    PAPER_PEAK_STEEPNESS_MAXIMUM,
    PAPER_RELATIVE_FREQUENCY_MAXIMUM,
    PAPER_RELATIVE_FREQUENCY_MINIMUM,
    PAPER_RELATIVE_FREQUENCY_WINDOW,
    PAPER_RIGHT_MOVING_FRACTIONS,
    PAPER_SHALLOW_PEAK_MODES,
    finite_depth_angular_frequency,
    paper_support_violations,
    positive_mode_wavenumbers,
)
from solver.gen_data.jonswap_tma_sampling import (  # noqa: E402
    DEEP_DEPTH_BOUNDS,
    DEEP_PEAK_WAVENUMBER_BOUNDS,
    FINITE_DEPTH_BOUNDS,
    FINITE_PEAK_WAVENUMBER_BOUNDS,
    JONSWAP_TMA_SAMPLE_CELLS,
    SHALLOW_DEPTH_WAVENUMBER_BOUNDS,
    SHALLOW_RELATIVE_HEIGHT_BOUNDS,
    SIGNIFICANT_HEIGHT_BOUNDS,
)
from solver.gen_data.pipeline.archive import (  # noqa: E402
    file_sha256,
    write_json_atomic,
)
from solver.gen_data.pipeline.production import (  # noqa: E402
    AttemptAssignment,
    CaseKey,
    PhysicalFamilyId,
    SplitId,
)
from solver.gen_data.pipeline.quality import (  # noqa: E402
    QualityDecision,
    QualityReason,
    QualityScope,
    reasons_from_bits,
)
from solver.gen_data.pipeline.quota_driver import (  # noqa: E402
    canonical_json_sha256,
)
from solver.gen_data.trajectory_quota_executor import (  # noqa: E402
    TrajectoryExecutionConfig,
)
from solver.gen_data.trajectory_family_adapters import (  # noqa: E402
    resolved_band_for_contract,
    sample_jonswap_tma_trajectory_cases,
)
from solver.gen_data.pipeline.time_selection import (  # noqa: E402
    select_uniform_times,
)


DEFAULT_DATASET_ROOT = ROOT / "outputs/paper_dataset_jonswap_revision4_relative_band_v1"
AUDIT_SCHEMA = "paper_dataset_jonswap_tma_revision4_completion_audit_v1"
PROPOSAL_LAW_APPLIES_TO = "attempted_specifications"
RELEASED_CASE_LAW = (
    "proposal_conditioned_on_complete_case_acceptance_within_preassigned_cell"
)
CELL_MARGINALS = "accepted_quota"
ROWS_PER_ACCEPTED_CASE = 16
EXPECTED_FAMILY_ID = int(PhysicalFamilyId.JONSWAP_TMA)
EXPECTED_REVISION_ID = 4
EXPECTED_ACCEPTED = 18_432
EXPECTED_TRAIN_ACCEPTED = 16_384
EXPECTED_VALIDATION_ACCEPTED = 1_024
EXPECTED_TEST_ACCEPTED = 1_024
EXPECTED_BATCH_SIZE = 32
EXPECTED_SOLVER_BATCH_SIZE = 8
EXPECTED_MAXIMUM_ATTEMPTS_PER_ACCEPTED_CASE = 4
TRAJECTORY_DIAGNOSTIC_QUANTILES = (0.0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0)
TRAJECTORY_DIAGNOSTIC_SIGN_DEAD_ZONE = 0.03
TRAJECTORY_DIAGNOSTIC_HIGH_BAND = (96.0, 128.0)
CURRENT_JONSWAP_EXECUTION = TrajectoryExecutionConfig.paper("jonswap_tma")
EXPECTED_SHARD_DTYPES = {
    "eta": np.dtype(np.float32),
    "xi": np.dtype(np.float32),
    "gxi": np.dtype(np.float32),
    "depth": np.dtype(np.float64),
    "time": np.dtype(np.float64),
    "case_local_index": np.dtype(np.int32),
    "frame_index": np.dtype(np.int32),
    "selected_dense_index": np.dtype(np.int32),
}
EXPECTED_MAP_ARRAYS = frozenset(
    {
        "frame_index",
        "schema_version",
        "shard_index",
        "shard_row",
        "trajectory_accepted",
        "trajectory_case_id",
        "trajectory_cell_id",
        "trajectory_evaluated_bits",
        "trajectory_failed_bits",
        "trajectory_family_id",
        "trajectory_first_row",
        "trajectory_index",
        "trajectory_required_bits",
        "trajectory_revision_id",
        "trajectory_row_count",
        "trajectory_split_id",
    }
)
EXPECTED_PRODUCTION_METRIC_KEYS = frozenset(
    {
        "accepted",
        "all_stages_solved",
        "complete_admissible_trajectory",
        "initial_discrete_peak_wavenumber",
        "initial_eta_rms",
        "initial_expected_linear_hamiltonian",
        "initial_half_maximum_spectral_cell_count",
        "initial_internal_hamiltonian",
        "initial_linear_hamiltonian",
        "initial_linear_hamiltonian_relative_error",
        "initial_minimum_water_column",
        "initial_realized_height_ratio",
        "initial_xi_rms",
        "intended_terminal_time",
        "internal_dno_finite",
        "internal_hamiltonian_drift_threshold",
        "internal_health_evaluated",
        "internal_state_finite",
        "maximum_internal_hamiltonian_drift",
        "maximum_stage_residual",
        "minimum_internal_water_column",
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
        "positive_water_column",
        "production_dt",
        "production_status",
        "realized_terminal_time",
        "saved_time_count",
        "state_finite",
        "target_finite",
    }
)
EXPECTED_ADJUSTMENT_FAILURE_METRIC_KEYS = frozenset(
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
EXPECTED_CONSTRUCTION_FAILURE_METRIC_KEYS = frozenset(
    {
        "construction_status",
        "construction_failure_reason",
        "original_local_index",
        "construction_failure_json",
    }
)
EXPECTED_PRODUCTION_REQUIRED_BITS = int(
    QualityReason.NONFINITE_STATE
    | QualityReason.NONFINITE_TARGET
    | QualityReason.BOTTOM_CLEARANCE
    | QualityReason.HAMILTONIAN_DRIFT
    | QualityReason.OUTSIDE_SUPPORT
    | QualityReason.INCOMPLETE_TRAJECTORY
)
EXPECTED_PRODUCTION_EVALUATED_BITS = int(
    QualityReason(EXPECTED_PRODUCTION_REQUIRED_BITS) | QualityReason.GL2_STAGE_RESIDUAL
)
EXPECTED_ADJUSTMENT_REQUIRED_BITS = int(
    QualityReason.OUTSIDE_SUPPORT | QualityReason.INCOMPLETE_TRAJECTORY
)


@dataclass(frozen=True)
class ExpectedChunk:
    """One exact additive interval in the completed JONSWAP population."""

    label: str
    relative_root: Path
    split: SplitId
    stream_id: int
    accepted_before: int
    accepted_count: int

    @property
    def summary_path(self) -> Path:
        return self.relative_root / (
            f"paper_dataset_jonswap_tma_{self.split.value}.summary.json"
        )


EXPECTED_CHUNKS = (
    ExpectedChunk(
        "train_00000_02048",
        Path("train/jonswap_tma/chunk_00000_02048"),
        SplitId.TRAIN,
        0,
        0,
        2_048,
    ),
    ExpectedChunk(
        "train_02048_02048",
        Path("train/jonswap_tma/chunk_02048_02048"),
        SplitId.TRAIN,
        1,
        2_048,
        2_048,
    ),
    ExpectedChunk(
        "train_04096_04096",
        Path("train/jonswap_tma/chunk_04096_04096"),
        SplitId.TRAIN,
        2,
        4_096,
        4_096,
    ),
    ExpectedChunk(
        "train_08192_04096",
        Path("train/jonswap_tma/chunk_08192_04096"),
        SplitId.TRAIN,
        3,
        8_192,
        4_096,
    ),
    ExpectedChunk(
        "train_12288_02048",
        Path("train/jonswap_tma/chunk_12288_02048"),
        SplitId.TRAIN,
        4,
        12_288,
        2_048,
    ),
    ExpectedChunk(
        "train_14336_02048",
        Path("train/jonswap_tma/chunk_14336_02048"),
        SplitId.TRAIN,
        5,
        14_336,
        2_048,
    ),
    ExpectedChunk(
        "validation_00000_01024",
        Path("validation/jonswap_tma/chunk_00000_01024"),
        SplitId.VALIDATION,
        100,
        0,
        1_024,
    ),
    ExpectedChunk(
        "test_00000_01024",
        Path("test/jonswap_tma/chunk_00000_01024"),
        SplitId.TEST,
        200,
        0,
        1_024,
    ),
)


@dataclass(frozen=True)
class AuditedCase:
    """One result record in final view order."""

    case_id: int
    accepted: bool
    required_bits: int
    evaluated_bits: int
    failed_bits: int
    first_row: int
    row_count: int
    batch_id: int


@dataclass(frozen=True)
class ReplayedSpecification:
    """One proposal specification reconstructed from its deterministic key."""

    case_id: int
    attempt_index: int
    cell_code: int
    cell_id: str
    record: Mapping[str, object]


@dataclass
class Extrema:
    """Finite minimum and maximum accumulator."""

    minimum: float = math.inf
    maximum: float = -math.inf
    count: int = 0

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError("extrema values must be finite")
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        self.count += 1

    def record(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "minimum": self.minimum if self.count else None,
            "maximum": self.maximum if self.count else None,
        }


@dataclass
class AuditTotals:
    """Mutable dataset-wide counters accumulated during the read-only scan."""

    attempted: int = 0
    accepted: int = 0
    rejected: int = 0
    retained_rows: int = 0
    committed_batches: int = 0
    proposal_specs_checked: int = 0
    adjustment_horizons_checked: int = 0
    autonomous_horizons_checked: int = 0
    stored_case_blocks_checked: int = 0
    construction_failed: int = 0
    adjustment_failed: int = 0
    production_completed: int = 0
    float_arrays_checked: int = 0
    float_values_checked: int = 0
    transaction_artifact_bytes_checked: int = 0
    dataset_view_bytes_checked: int = 0


@dataclass(frozen=True)
class DiagnosticIdentity:
    """Durable location of one trajectory diagnostic observation."""

    case_id: int
    chunk_label: str
    split: str
    batch_id: int
    local_index: int
    frame_index: int
    cell_id: str
    shard_path: str

    def record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "chunk_label": self.chunk_label,
            "split": self.split,
            "batch_id": self.batch_id,
            "local_index": self.local_index,
            "frame_index": self.frame_index,
            "cell_id": self.cell_id,
            "shard_path": self.shard_path,
        }


@dataclass(frozen=True)
class DiagnosticObservation:
    """One finite scalar diagnostic and the row that attained it."""

    value: float
    identity: DiagnosticIdentity

    def record(self) -> dict[str, object]:
        return {"value": self.value, "identity": self.identity.record()}


@dataclass
class DiagnosticSeries:
    """O(number of trajectories) scalar storage for fixed dataset quantiles."""

    values: list[float] = dataclass_field(default_factory=list)
    minimum: DiagnosticObservation | None = None
    maximum: DiagnosticObservation | None = None

    def add(self, value: float, identity: DiagnosticIdentity) -> None:
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError("trajectory diagnostic values must be finite")
        observation = DiagnosticObservation(scalar, identity)
        self.values.append(scalar)
        if self.minimum is None or scalar < self.minimum.value:
            self.minimum = observation
        if self.maximum is None or scalar > self.maximum.value:
            self.maximum = observation

    def record(self) -> dict[str, object]:
        if self.minimum is None or self.maximum is None:
            return {
                "count": 0,
                "quantiles": {},
                "minimum": None,
                "maximum": None,
            }
        values = np.asarray(self.values, dtype=np.float64)
        return {
            "count": len(self.values),
            "quantiles": {
                f"q{probability:g}": float(np.quantile(values, probability))
                for probability in TRAJECTORY_DIAGNOSTIC_QUANTILES
            },
            "minimum": self.minimum.record(),
            "maximum": self.maximum.record(),
        }


TRAJECTORY_DIAGNOSTIC_NAMES = (
    "eta_effective_wavenumber_terminal",
    "eta_effective_wavenumber_maximum",
    "gxi_effective_wavenumber_terminal",
    "gxi_effective_wavenumber_maximum",
    "eta_cyclic_difference_sign_flips_terminal",
    "eta_cyclic_difference_sign_flips_maximum",
    "gxi_cyclic_difference_sign_flips_terminal",
    "gxi_cyclic_difference_sign_flips_maximum",
    "eta_high_band_energy_fraction_terminal",
    "eta_high_band_energy_fraction_maximum",
    "gxi_high_band_energy_fraction_terminal",
    "gxi_high_band_energy_fraction_maximum",
)


@dataclass
class TrajectoryQualityDiagnostics:
    """Non-gating morphology diagnostics over all accepted trajectories."""

    series: dict[str, DiagnosticSeries] = dataclass_field(
        default_factory=lambda: {
            name: DiagnosticSeries() for name in TRAJECTORY_DIAGNOSTIC_NAMES
        }
    )
    trajectory_count: int = 0
    peak_gxi_high_band_above_0p10: int = 0
    peak_gxi_high_band_above_0p20: int = 0

    def record(self) -> dict[str, object]:
        return {
            "diagnostic_only": True,
            "metric_values_affect_acceptance_or_audit_status": False,
            "diagnostic_coverage_required_for_audit_completion": True,
            "accepted_trajectories_checked": self.trajectory_count,
            "definitions": {
                "effective_wavenumber": (
                    "sqrt(sum_{k != 0} k^2 |f_hat_k|^2 / "
                    "sum_{k != 0} |f_hat_k|^2); zero for a constant field"
                ),
                "cyclic_difference_sign_flips": (
                    "cyclic sign transitions after forming forward differences "
                    "and discarding |difference| <= 0.03 max|difference|"
                ),
                "high_band_energy_fraction": (
                    "sum_{96 <= |k| <= 128} |f_hat_k|^2 / "
                    "sum_{1 <= |k| <= 128} |f_hat_k|^2"
                ),
                "terminal": "frame 15 of each delivered 16-frame trajectory",
                "maximum": "maximum over all 16 delivered frames",
            },
            "quantile_probabilities": list(TRAJECTORY_DIAGNOSTIC_QUANTILES),
            "metrics": {
                name: summary.record() for name, summary in self.series.items()
            },
            "tail_counts": {
                "peak_gxi_high_band_energy_fraction_gt_0p10": (
                    self.peak_gxi_high_band_above_0p10
                ),
                "peak_gxi_high_band_energy_fraction_gt_0p20": (
                    self.peak_gxi_high_band_above_0p20
                ),
            },
        }


def _field_quality_metrics(values: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-frame non-gating roughness diagnostics for one real field."""

    fields = np.asarray(values)
    if fields.ndim != 2:
        raise ValueError("trajectory diagnostic input must be a two-dimensional field")
    nx = fields.shape[1]
    length = CURRENT_JONSWAP_EXECUTION.numerical.target_definition.length
    wavenumbers = 2.0 * np.pi * np.fft.rfftfreq(nx, d=length / nx)
    spectrum = np.fft.rfft(fields, axis=1)
    multiplicity = np.full(wavenumbers.size, 2.0, dtype=np.float64)
    multiplicity[0] = 1.0
    if nx % 2 == 0:
        multiplicity[-1] = 1.0
    energy = multiplicity[None, :] * np.square(np.abs(spectrum))
    nonzero = wavenumbers > 0.0
    total_nonzero = np.sum(energy[:, nonzero], axis=1)
    effective_wavenumber = np.sqrt(
        np.divide(
            np.sum(
                energy[:, nonzero] * np.square(wavenumbers[nonzero])[None, :],
                axis=1,
            ),
            total_nonzero,
            out=np.zeros_like(total_nonzero),
            where=total_nonzero > 0.0,
        )
    )
    high_minimum, high_maximum = TRAJECTORY_DIAGNOSTIC_HIGH_BAND
    delivered = (wavenumbers >= 1.0) & (wavenumbers <= high_maximum)
    high = (wavenumbers >= high_minimum) & (wavenumbers <= high_maximum)
    delivered_energy = np.sum(energy[:, delivered], axis=1)
    high_band_fraction = np.divide(
        np.sum(energy[:, high], axis=1),
        delivered_energy,
        out=np.zeros_like(delivered_energy),
        where=delivered_energy > 0.0,
    )

    differences = np.roll(fields, -1, axis=1) - fields
    scales = np.max(np.abs(differences), axis=1)
    sign_flips = np.zeros(fields.shape[0], dtype=np.int64)
    for row_index, (difference, scale) in enumerate(zip(differences, scales)):
        if scale <= 0.0:
            continue
        retained = difference[
            np.abs(difference) > TRAJECTORY_DIAGNOSTIC_SIGN_DEAD_ZONE * scale
        ]
        if retained.size < 2:
            continue
        signs = np.signbit(retained)
        sign_flips[row_index] = np.count_nonzero(signs != np.roll(signs, -1))
    return {
        "effective_wavenumber": effective_wavenumber,
        "cyclic_difference_sign_flips": sign_flips,
        "high_band_energy_fraction": high_band_fraction,
    }


def _strict_json_object(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{path} contains nonfinite JSON constant {value!r}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _required_mapping(
    mapping: Mapping[str, object],
    name: str,
    *,
    context: str,
) -> Mapping[str, object]:
    value = mapping.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"{context}.{name} must be a JSON object")
    return value


def _required_list(
    mapping: Mapping[str, object],
    name: str,
    *,
    context: str,
) -> list[object]:
    value = mapping.get(name)
    if not isinstance(value, list):
        raise TypeError(f"{context}.{name} must be a JSON array")
    return value


def _required_integer(
    mapping: Mapping[str, object],
    name: str,
    *,
    context: str,
    minimum: int | None = None,
) -> int:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context}.{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{context}.{name} must be at least {minimum}")
    return value


def _required_string(
    mapping: Mapping[str, object],
    name: str,
    *,
    context: str,
) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{context}.{name} must be a nonempty string")
    return value


def _finite_number(value: object, *, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise TypeError(f"{context} must be a finite real number")
    return float(value)


def _rejection_rate(*, attempted: int, rejected: int, context: str) -> float:
    """Return one finite rejection rate after exact count validation."""

    if attempted < 0 or rejected < 0 or rejected > attempted:
        raise ValueError(f"{context} has inconsistent attempted/rejected counts")
    if attempted == 0:
        return 0.0
    return rejected / attempted


def _population_conditioning_record(
    chunk_records: Sequence[Mapping[str, object]],
    *,
    attempted: int,
    accepted: int,
    rejected: int,
) -> dict[str, object]:
    """Aggregate exact 27-cell counts and state released-population semantics."""

    if min(attempted, accepted, rejected) < 0 or attempted != accepted + rejected:
        raise ValueError("dataset attempted/accepted/rejected counts do not close")
    cell_ids = tuple(cell.cell_id for cell in JONSWAP_TMA_SAMPLE_CELLS)
    expected_cells = set(cell_ids)
    aggregate = {
        cell_id: {
            "target_accepted": 0,
            "attempted": 0,
            "accepted": 0,
            "rejected": 0,
        }
        for cell_id in cell_ids
    }
    for chunk_index, chunk in enumerate(chunk_records):
        chunk_context = f"chunk_records[{chunk_index}]"
        cell_counts = _required_mapping(
            chunk,
            "cell_counts",
            context=chunk_context,
        )
        context = f"{chunk_context}.cell_counts"
        if set(cell_counts) != expected_cells:
            raise ValueError(f"{context} must contain the exact 27 JONSWAP cells")
        for cell_id in cell_ids:
            record = _required_mapping(cell_counts, cell_id, context=context)
            values = {
                name: _required_integer(
                    record,
                    name,
                    context=f"{context}.{cell_id}",
                    minimum=0,
                )
                for name in (
                    "target_accepted",
                    "attempted",
                    "accepted",
                    "rejected",
                )
            }
            if values["accepted"] != values["target_accepted"]:
                raise ValueError(f"{context}.{cell_id} does not meet its quota")
            if values["attempted"] != values["accepted"] + values["rejected"]:
                raise ValueError(f"{context}.{cell_id} counts do not close")
            for name, value in values.items():
                aggregate[cell_id][name] += value

    cells: dict[str, object] = {}
    for cell_id in cell_ids:
        counts = aggregate[cell_id]
        if counts["accepted"] != counts["target_accepted"]:
            raise ValueError(f"aggregate {cell_id} does not meet its quota")
        if counts["attempted"] != counts["accepted"] + counts["rejected"]:
            raise ValueError(f"aggregate {cell_id} counts do not close")
        cells[cell_id] = {
            **counts,
            "rejection_rate": _rejection_rate(
                attempted=counts["attempted"],
                rejected=counts["rejected"],
                context=f"aggregate {cell_id}",
            ),
        }

    if sum(int(record["attempted"]) for record in aggregate.values()) != attempted:
        raise ValueError("aggregate cell attempted counts differ from dataset total")
    if sum(int(record["accepted"]) for record in aggregate.values()) != accepted:
        raise ValueError("aggregate cell accepted counts differ from dataset total")
    if sum(int(record["rejected"]) for record in aggregate.values()) != rejected:
        raise ValueError("aggregate cell rejected counts differ from dataset total")
    return {
        "proposal_law_applies_to": PROPOSAL_LAW_APPLIES_TO,
        "released_case_law": RELEASED_CASE_LAW,
        "cell_marginals": CELL_MARGINALS,
        "posthoc_parameter_gate": False,
        "cells": cells,
    }


def _required_boolean(
    mapping: Mapping[str, object],
    name: str,
    *,
    context: str,
) -> bool:
    value = mapping.get(name)
    if not isinstance(value, bool):
        raise TypeError(f"{context}.{name} must be Boolean")
    return value


def _same_json(left: object, right: object) -> bool:
    return canonical_json_sha256(left) == canonical_json_sha256(right)


def _scan_finite_archive(
    path: Path,
    *,
    totals: AuditTotals,
) -> None:
    """Require every stored floating or complex array to be finite."""

    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            array = np.asarray(archive[name])
            if array.dtype.kind not in {"f", "c"}:
                continue
            totals.float_arrays_checked += 1
            totals.float_values_checked += int(array.size)
            if not np.isfinite(array).all():
                raise ValueError(f"{path}:{name} contains a nonfinite value")


def _case_specifications(
    proposal_path: Path,
    *,
    split: SplitId,
    stream_id: int,
    cell_ids_by_code: Mapping[int, str],
    support_extrema: Mapping[str, Extrema],
    totals: AuditTotals,
) -> tuple[ReplayedSpecification, ...]:
    """Replay every JONSWAP parameter and phase record from its case key."""

    with np.load(proposal_path, allow_pickle=False) as archive:
        case_ids = np.asarray(archive["case_id"], dtype=np.int64)
        family_id = np.asarray(archive["family_id"])
        revision_id = np.asarray(archive["revision_id"])
        split_id = np.asarray(archive["split_id"])
        roots = np.asarray(archive["root_seed"], dtype=np.uint64)
        streams = np.asarray(archive["stream_id"], dtype=np.uint32)
        attempts = np.asarray(archive["attempt_index"], dtype=np.uint64)
        encoded_cells = np.asarray(archive["cell_id"], dtype=np.int32)
        specifications = np.asarray(archive["case_spec_json"])

    count = int(case_ids.size)
    vectors = (
        roots,
        streams,
        attempts,
        encoded_cells,
        specifications,
    )
    if any(vector.shape != (count,) for vector in vectors):
        raise ValueError(f"{proposal_path} has inconsistent proposal vectors")
    split_code = {
        SplitId.TRAIN: 0,
        SplitId.VALIDATION: 1,
        SplitId.TEST: 2,
    }[split]
    if (
        family_id.ndim != 0
        or revision_id.ndim != 0
        or split_id.ndim != 0
        or int(family_id.item()) != EXPECTED_FAMILY_ID
        or int(revision_id.item()) != EXPECTED_REVISION_ID
        or int(split_id.item()) != split_code
    ):
        raise ValueError(f"{proposal_path} has the wrong family/run coordinates")
    band = resolved_band_for_contract(
        CURRENT_JONSWAP_EXECUTION.numerical,
        quadrature_order=int(CURRENT_JONSWAP_EXECUTION.jonswap_quadrature_order),
    )
    expected_phase_count = int(positive_mode_wavenumbers(band=band).size)
    replayed: list[ReplayedSpecification] = []
    for index, encoded in enumerate(encoded_cells):
        try:
            cell_id = cell_ids_by_code[int(encoded)]
        except KeyError as error:
            raise ValueError(
                f"{proposal_path} contains unknown JONSWAP cell code {int(encoded)}"
            ) from error
        if int(streams[index]) != stream_id:
            raise ValueError(
                f"{proposal_path} case {index} has the wrong family/run coordinates"
            )
        key = CaseKey(
            family_id=EXPECTED_FAMILY_ID,
            revision_id=EXPECTED_REVISION_ID,
            split_id=split,
            stream_id=stream_id,
            attempt_index=int(attempts[index]),
        )
        if int(roots[index]) != key.root_seed or int(case_ids[index]) != key.case_id:
            raise ValueError(
                f"{proposal_path} case {index} has inconsistent run coordinates"
            )
        assignment = AttemptAssignment(case_key=key, cell_id=cell_id)
        sampled = sample_jonswap_tma_trajectory_cases(
            (assignment,),
            contract=CURRENT_JONSWAP_EXECUTION.numerical,
            quadrature_order=int(CURRENT_JONSWAP_EXECUTION.jonswap_quadrature_order),
        )
        expected_sample = sampled.samples[0]
        violations = paper_support_violations(
            expected_sample.parameters,
            stratum=expected_sample.cell.stratum,
            length=band.length,
            band=band,
            relative_frequency_maximum=PAPER_RELATIVE_FREQUENCY_MAXIMUM,
        )
        if violations:
            raise ValueError(
                f"current JONSWAP sampler produced unsupported case {key.case_id}: "
                + "; ".join(violations)
            )
        expected = sampled.specification_records[0]
        observed = json.loads(
            str(specifications[index]),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(
                    f"{proposal_path} specification contains nonfinite {value!r}"
                )
            ),
        )
        if not isinstance(observed, dict) or not _same_json(observed, expected):
            raise ValueError(
                f"{proposal_path} case {key.case_id} differs from exact "
                "current JONSWAP deterministic sampling"
            )
        phases = tuple(
            np.asarray(expected[name], dtype=np.float64)
            for name in ("phase_right", "phase_left")
        )
        if any(
            phase.shape != (expected_phase_count,)
            or not np.isfinite(phase).all()
            or np.any(phase < 0.0)
            or np.any(phase >= 2.0 * math.pi)
            for phase in phases
        ):
            raise ValueError("JONSWAP phase arrays violate their exact support")
        parameters = expected_sample.parameters
        omega = finite_depth_angular_frequency(
            np.asarray([parameters.peak_wavenumber, band.maximum_wavenumber]),
            depth=parameters.depth,
            gravity=CURRENT_JONSWAP_EXECUTION.numerical.gravity,
        )
        for name, value in (
            ("depth", parameters.depth),
            ("significant_height", parameters.significant_height),
            ("peak_wavenumber", parameters.peak_wavenumber),
            ("peak_enhancement", parameters.peak_enhancement),
            ("right_moving_fraction", parameters.right_moving_fraction),
            ("depth_wavenumber", parameters.depth * parameters.peak_wavenumber),
            (
                "relative_height",
                parameters.significant_height / (2.0 * parameters.depth),
            ),
            (
                "peak_steepness",
                parameters.peak_wavenumber * parameters.significant_height / 2.0,
            ),
            ("resolved_maximum_relative_frequency", omega[1] / omega[0]),
            ("phase_count_per_direction", float(expected_phase_count)),
        ):
            support_extrema[name].add(float(value))
        totals.proposal_specs_checked += 1
        replayed.append(
            ReplayedSpecification(
                case_id=key.case_id,
                attempt_index=key.attempt_index,
                cell_code=int(encoded),
                cell_id=cell_id,
                record=expected,
            )
        )
    return tuple(replayed)


def _validate_result_metadata(
    result: Mapping[str, object],
    *,
    run_spec: Mapping[str, object],
    execution: Mapping[str, object],
    context: str,
) -> None:
    metadata = _required_mapping(result, "metadata", context=context)
    if metadata.get("family") != "jonswap_tma":
        raise ValueError(f"{context} has the wrong result family")
    if metadata.get("case_kind") != "trajectory":
        raise ValueError(f"{context} has the wrong result case kind")
    actual_execution = _required_mapping(
        metadata,
        "trajectory_execution",
        context=f"{context}.metadata",
    )
    if not _same_json(actual_execution, execution):
        raise ValueError(f"{context} execution differs from its run spec")
    additional = _required_mapping(
        metadata,
        "additional_metadata",
        context=f"{context}.metadata",
    )
    result_run_spec = _required_mapping(
        additional,
        "run_spec",
        context=f"{context}.metadata.additional_metadata",
    )
    if not _same_json(result_run_spec, run_spec):
        raise ValueError(f"{context} run spec differs from its summary")
    if additional.get("launcher") != "scripts/run_paper_dataset_jonswap_bucketed.py":
        raise ValueError(f"{context} was not produced by the bucketed launcher")
    configuration = _required_mapping(
        run_spec,
        "configuration",
        context=f"{context}.run_spec",
    )
    configured_policy = _required_mapping(
        configuration,
        POLICY_KEY,
        context=f"{context}.run_spec.configuration",
    )
    result_policy = _required_mapping(
        additional,
        POLICY_KEY,
        context=f"{context}.metadata.additional_metadata",
    )
    if not _same_json(result_policy, configured_policy):
        raise ValueError(f"{context} bucketing policy differs from its run spec")


def _floored_saved_count(intended: float, saved_dt: float) -> int:
    step_count = math.floor(intended / saved_dt)
    while step_count * saved_dt > intended:
        step_count -= 1
    while (step_count + 1) * saved_dt <= intended:
        step_count += 1
    return step_count + 1


def _validate_floored_horizon(
    metrics: Mapping[str, object],
    *,
    prefix: str,
    expected_intended: float,
    saved_dt: float,
) -> int:
    intended = _finite_number(
        metrics.get(f"{prefix}intended_terminal_time"),
        context=f"metrics.{prefix}intended_terminal_time",
    )
    realized = _finite_number(
        metrics.get(f"{prefix}realized_terminal_time"),
        context=f"metrics.{prefix}realized_terminal_time",
    )
    raw_count = metrics.get(f"{prefix}saved_time_count")
    if isinstance(raw_count, bool) or not isinstance(raw_count, int):
        raise TypeError(f"metrics.{prefix}saved_time_count must be an integer")
    expected_count = _floored_saved_count(expected_intended, saved_dt)
    expected_realized = (expected_count - 1) * saved_dt
    if not math.isclose(intended, expected_intended, rel_tol=2.0e-14, abs_tol=1.0e-13):
        raise ValueError(f"metrics.{prefix}intended horizon is incorrect")
    if raw_count != expected_count:
        raise ValueError(f"metrics.{prefix}saved-time count is incorrect")
    if not math.isclose(realized, expected_realized, rel_tol=2.0e-14, abs_tol=1.0e-13):
        raise ValueError(f"metrics.{prefix}realized horizon is incorrect")
    if not (realized <= intended and intended - realized < saved_dt):
        raise ValueError(f"metrics.{prefix}horizon is not strictly floored")
    return raw_count


def _peak_period(specification: Mapping[str, object]) -> float:
    depth = _finite_number(specification.get("depth"), context="specification.depth")
    peak_wavenumber = _finite_number(
        specification.get("peak_wavenumber"),
        context="specification.peak_wavenumber",
    )
    angular_frequency = finite_depth_angular_frequency(
        np.asarray([peak_wavenumber], dtype=np.float64),
        depth=depth,
        gravity=CURRENT_JONSWAP_EXECUTION.numerical.gravity,
    )[0]
    return 2.0 * math.pi / float(angular_frequency)


def _validate_construction_failure_metrics(
    metrics: Mapping[str, object],
    *,
    local_index: int,
) -> tuple[int, int]:
    if set(metrics) != EXPECTED_CONSTRUCTION_FAILURE_METRIC_KEYS:
        raise ValueError("construction rejection has an unknown metric schema")
    if metrics.get("construction_status") != "declared_outside_support":
        raise ValueError("construction rejection has the wrong status")
    if metrics.get("construction_failure_reason") != "invalid_initial_graph_state":
        raise ValueError("construction rejection has the wrong failure reason")
    if metrics.get("original_local_index") != local_index:
        raise ValueError("construction rejection has the wrong local index")
    encoded = metrics.get("construction_failure_json")
    if not isinstance(encoded, str):
        raise TypeError("construction_failure_json must be a string")
    failure = json.loads(encoded)
    if not isinstance(failure, dict):
        raise TypeError("construction_failure_json must encode an object")
    if (
        failure.get("schema") != "jonswap_initial_state_domain_failure_v1"
        or failure.get("reason") != "invalid_initial_graph_state"
        or failure.get("index_space") != "original_proposal_local_index"
    ):
        raise ValueError("construction rejection has the wrong failure schema")
    invalid = failure.get("invalid_case_indices")
    if not isinstance(invalid, list) or local_index not in invalid:
        raise ValueError("construction rejection does not name its proposal case")
    cases = failure.get("cases")
    if not isinstance(cases, list):
        raise TypeError("construction failure must list invalid cases")
    matching = tuple(
        case
        for case in cases
        if isinstance(case, Mapping) and case.get("local_case_index") == local_index
    )
    if len(matching) != 1:
        raise ValueError("construction failure does not uniquely describe its case")
    case = matching[0]
    state_finite = _required_boolean(
        case,
        "state_finite",
        context="construction failure case",
    )
    minimum = case.get("minimum_water_column")
    evaluated = QualityReason.OUTSIDE_SUPPORT | QualityReason.NONFINITE_STATE
    failed = QualityReason.OUTSIDE_SUPPORT
    if state_finite:
        evaluated |= QualityReason.BOTTOM_CLEARANCE
        water = _finite_number(
            minimum,
            context="construction failure minimum_water_column",
        )
        if (
            water > 0.0
            or case.get("failure_reason") != "nonpositive_initial_water_column"
        ):
            raise ValueError("construction bottom failure is inconsistent")
        failed |= QualityReason.BOTTOM_CLEARANCE
    else:
        if (
            minimum is not None
            or case.get("failure_reason") != "nonfinite_initial_state"
        ):
            raise ValueError("construction nonfinite failure is inconsistent")
        failed |= QualityReason.NONFINITE_STATE
    return int(evaluated), int(failed)


def _validate_case_metrics(
    case: Mapping[str, object],
    *,
    accepted: bool,
    local_index: int,
    specification: Mapping[str, object],
    residual_tolerance: float,
    hamiltonian_threshold: float,
    production_dt: float,
    saved_dt: float,
    accepted_extrema: Mapping[str, Extrema],
    attempted_extrema: Mapping[str, Extrema],
    totals: AuditTotals,
) -> tuple[str, int, int]:
    metrics = _required_mapping(case, "metrics", context="result case")
    if set(metrics) == EXPECTED_CONSTRUCTION_FAILURE_METRIC_KEYS:
        if accepted:
            raise ValueError("accepted case cannot be a construction rejection")
        evaluated, failed = _validate_construction_failure_metrics(
            metrics,
            local_index=local_index,
        )
        return "construction_failed", evaluated, failed
    if set(metrics) not in {
        EXPECTED_PRODUCTION_METRIC_KEYS,
        EXPECTED_ADJUSTMENT_FAILURE_METRIC_KEYS,
    }:
        raise ValueError("result case metrics have an unknown JONSWAP schema")

    peak_period = _peak_period(specification)
    adjustment_policy = CURRENT_JONSWAP_EXECUTION.jonswap_adjustment
    assert adjustment_policy is not None
    production_periods = CURRENT_JONSWAP_EXECUTION.horizon.period_count
    assert production_periods is not None
    _validate_floored_horizon(
        metrics,
        prefix="nonlinear_adjustment_",
        expected_intended=adjustment_policy.burn_peak_periods * peak_period,
        saved_dt=saved_dt,
    )
    totals.adjustment_horizons_checked += 1
    if (
        metrics.get("nonlinear_adjustment_schema")
        != "dommermuth_nonlinear_adjustment_v1"
    ):
        raise ValueError("JONSWAP adjustment schema is incorrect")
    if metrics.get("nonlinear_adjustment_production_ramp") != "disabled":
        raise ValueError("JONSWAP autonomous production must disable the ramp")
    if metrics.get("nonlinear_adjustment_ramp_order") != adjustment_policy.ramp_order:
        raise ValueError("JONSWAP adjustment ramp order is incorrect")
    ramp_time = _finite_number(
        metrics.get("nonlinear_adjustment_ramp_time"),
        context="metrics.nonlinear_adjustment_ramp_time",
    )
    if not math.isclose(
        ramp_time,
        adjustment_policy.ramp_time_peak_periods * peak_period,
        rel_tol=2.0e-14,
        abs_tol=1.0e-13,
    ):
        raise ValueError("JONSWAP adjustment ramp time is incorrect")

    adjustment_accepted = _required_boolean(
        metrics,
        "nonlinear_adjustment_accepted",
        context="metrics",
    )
    adjustment_stages = _required_boolean(
        metrics,
        "nonlinear_adjustment_all_stages_solved",
        context="metrics",
    )
    adjustment_complete = _required_boolean(
        metrics,
        "nonlinear_adjustment_complete_admissible_handoff",
        context="metrics",
    )
    adjustment_state = _required_boolean(
        metrics,
        "nonlinear_adjustment_state_finite",
        context="metrics",
    )
    adjustment_clearance = metrics.get("nonlinear_adjustment_positive_water_column")
    if adjustment_clearance is not None and not isinstance(
        adjustment_clearance,
        bool,
    ):
        raise TypeError("nonlinear_adjustment_positive_water_column is invalid")
    adjustment_evaluated = QualityReason(
        EXPECTED_ADJUSTMENT_REQUIRED_BITS
        | int(QualityReason.NONFINITE_STATE)
        | int(QualityReason.GL2_STAGE_RESIDUAL)
    )
    adjustment_failed = QualityReason.NONE
    if not adjustment_stages:
        adjustment_failed |= QualityReason.GL2_STAGE_RESIDUAL
    if not adjustment_state:
        adjustment_failed |= QualityReason.NONFINITE_STATE
    if adjustment_clearance is not None:
        adjustment_evaluated |= QualityReason.BOTTOM_CLEARANCE
        if not adjustment_clearance:
            adjustment_failed |= QualityReason.BOTTOM_CLEARANCE
    if adjustment_failed:
        adjustment_failed |= QualityReason.INCOMPLETE_TRAJECTORY
    if adjustment_complete is bool(adjustment_failed):
        raise ValueError("adjustment completeness disagrees with its health metrics")
    if adjustment_accepted is bool(adjustment_failed):
        raise ValueError("adjustment acceptance disagrees with its health metrics")
    adjustment_residual = metrics.get("nonlinear_adjustment_maximum_stage_residual")
    adjustment_water = metrics.get("nonlinear_adjustment_minimum_water_column")
    if adjustment_residual is not None:
        attempted_extrema["adjustment_stage_residual"].add(
            _finite_number(
                adjustment_residual,
                context="metrics.nonlinear_adjustment_maximum_stage_residual",
            )
        )
    if adjustment_water is not None:
        attempted_extrema["adjustment_minimum_water_column"].add(
            _finite_number(
                adjustment_water,
                context="metrics.nonlinear_adjustment_minimum_water_column",
            )
        )

    if not adjustment_accepted:
        if set(metrics) != EXPECTED_ADJUSTMENT_FAILURE_METRIC_KEYS:
            raise ValueError(
                "failed adjustment unexpectedly contains production metrics"
            )
        if accepted or metrics.get("production_status") != "not_run_adjustment_failed":
            raise ValueError("failed adjustment cannot own an accepted production")
        return (
            "adjustment_failed",
            int(adjustment_evaluated),
            int(adjustment_failed),
        )

    if set(metrics) != EXPECTED_PRODUCTION_METRIC_KEYS:
        raise ValueError("successful adjustment omits autonomous metrics")
    if not all(
        (
            adjustment_stages,
            adjustment_complete,
            adjustment_clearance is True,
            adjustment_state,
        )
    ):
        raise ValueError("successful adjustment has a failed health metric")
    adjustment_residual_value = _finite_number(
        adjustment_residual,
        context="metrics.nonlinear_adjustment_maximum_stage_residual",
    )
    adjustment_water_value = _finite_number(
        adjustment_water,
        context="metrics.nonlinear_adjustment_minimum_water_column",
    )
    if not 0.0 <= adjustment_residual_value <= residual_tolerance:
        raise ValueError("accepted adjustment exceeds the GL2 residual tolerance")
    if adjustment_water_value <= 0.0:
        raise ValueError("accepted adjustment has nonpositive water depth")
    if metrics.get("production_status") != "completed":
        raise ValueError("successful adjustment did not run autonomous production")

    production_saved_count = _validate_floored_horizon(
        metrics,
        prefix="",
        expected_intended=production_periods * peak_period,
        saved_dt=saved_dt,
    )
    if production_saved_count < ROWS_PER_ACCEPTED_CASE:
        raise ValueError("JONSWAP trajectory has too few dense saved times")
    totals.autonomous_horizons_checked += 1
    if metrics.get("accepted") is not accepted:
        raise ValueError("result metric accepted flag disagrees with the case")
    if (
        _finite_number(
            metrics.get("production_dt"),
            context="metrics.production_dt",
        )
        != production_dt
    ):
        raise ValueError("result production_dt differs from the execution contract")
    if (
        _finite_number(
            metrics.get("internal_hamiltonian_drift_threshold"),
            context="metrics.internal_hamiltonian_drift_threshold",
        )
        != hamiltonian_threshold
    ):
        raise ValueError("result Hamiltonian threshold differs from the contract")

    for name in (
        "initial_discrete_peak_wavenumber",
        "initial_eta_rms",
        "initial_expected_linear_hamiltonian",
        "initial_internal_hamiltonian",
        "initial_linear_hamiltonian",
        "initial_linear_hamiltonian_relative_error",
        "initial_minimum_water_column",
        "initial_realized_height_ratio",
        "initial_xi_rms",
        "maximum_internal_hamiltonian_drift",
        "maximum_stage_residual",
        "minimum_internal_water_column",
    ):
        value = metrics.get(name)
        if value is not None:
            _finite_number(value, context=f"metrics.{name}")
    half_maximum_count = metrics.get("initial_half_maximum_spectral_cell_count")
    if (
        isinstance(half_maximum_count, bool)
        or not isinstance(half_maximum_count, int)
        or half_maximum_count <= 0
    ):
        raise ValueError("initial spectral cell count must be a positive integer")

    state_finite = _required_boolean(metrics, "state_finite", context="metrics")
    target_finite = _required_boolean(metrics, "target_finite", context="metrics")
    stages_solved = _required_boolean(
        metrics,
        "all_stages_solved",
        context="metrics",
    )
    complete = _required_boolean(
        metrics,
        "complete_admissible_trajectory",
        context="metrics",
    )
    positive_water = _required_boolean(
        metrics,
        "positive_water_column",
        context="metrics",
    )
    internal_evaluated = _required_boolean(
        metrics,
        "internal_health_evaluated",
        context="metrics",
    )
    internal_state = _required_boolean(
        metrics,
        "internal_state_finite",
        context="metrics",
    )
    internal_dno = _required_boolean(
        metrics,
        "internal_dno_finite",
        context="metrics",
    )
    if not internal_evaluated:
        raise ValueError("current JONSWAP production requires internal health")
    production_failed = QualityReason.NONE
    if not state_finite or not internal_state:
        production_failed |= QualityReason.NONFINITE_STATE
    if not target_finite or not internal_dno:
        production_failed |= QualityReason.NONFINITE_TARGET
    internal_water = metrics.get("minimum_internal_water_column")
    if (
        not positive_water
        or internal_water is None
        or _finite_number(
            internal_water,
            context="metrics.minimum_internal_water_column",
        )
        <= 0.0
    ):
        production_failed |= QualityReason.BOTTOM_CLEARANCE
    drift_metric = metrics.get("maximum_internal_hamiltonian_drift")
    if (
        drift_metric is None
        or _finite_number(
            drift_metric,
            context="metrics.maximum_internal_hamiltonian_drift",
        )
        > hamiltonian_threshold
    ):
        production_failed |= QualityReason.HAMILTONIAN_DRIFT
    if not stages_solved:
        production_failed |= QualityReason.GL2_STAGE_RESIDUAL
    if production_failed:
        production_failed |= QualityReason.INCOMPLETE_TRAJECTORY
    if not accepted and complete is bool(production_failed):
        raise ValueError("autonomous completeness disagrees with health metrics")

    if accepted:
        required_true = (
            "all_stages_solved",
            "complete_admissible_trajectory",
            "internal_dno_finite",
            "internal_health_evaluated",
            "internal_state_finite",
            "positive_water_column",
            "state_finite",
            "target_finite",
        )
        if any(metrics.get(name) is not True for name in required_true):
            raise ValueError("accepted JONSWAP case has a failed health metric")
        initial_h = _finite_number(
            metrics.get("initial_internal_hamiltonian"),
            context="metrics.initial_internal_hamiltonian",
        )
        drift = _finite_number(
            metrics.get("maximum_internal_hamiltonian_drift"),
            context="metrics.maximum_internal_hamiltonian_drift",
        )
        residual = _finite_number(
            metrics.get("maximum_stage_residual"),
            context="metrics.maximum_stage_residual",
        )
        water = _finite_number(
            metrics.get("minimum_internal_water_column"),
            context="metrics.minimum_internal_water_column",
        )
        if not 0.0 <= residual <= residual_tolerance:
            raise ValueError(
                "accepted autonomous case exceeds the GL2 residual tolerance"
            )
        if not 0.0 <= drift <= hamiltonian_threshold:
            raise ValueError(
                "accepted autonomous case exceeds the Hamiltonian threshold"
            )
        if water <= 0.0:
            raise ValueError("accepted autonomous case has nonpositive water depth")
        for name, value in (
            ("initial_hamiltonian", initial_h),
            ("hamiltonian_drift", drift),
            ("stage_residual", residual),
            ("minimum_water_column", water),
            ("adjustment_stage_residual", adjustment_residual_value),
            ("adjustment_minimum_water_column", adjustment_water_value),
        ):
            accepted_extrema[name].add(value)

    for name, field in (
        ("hamiltonian_drift", "maximum_internal_hamiltonian_drift"),
        ("stage_residual", "maximum_stage_residual"),
        ("minimum_water_column", "minimum_internal_water_column"),
    ):
        value = metrics.get(field)
        if value is not None:
            attempted_extrema[name].add(float(value))
    return (
        "production_completed",
        EXPECTED_PRODUCTION_EVALUATED_BITS,
        int(production_failed),
    )


def _validate_case_record(
    value: object,
    *,
    batch_id: int,
    local_index: int,
    specification: ReplayedSpecification,
    residual_tolerance: float,
    hamiltonian_threshold: float,
    production_dt: float,
    saved_dt: float,
    accepted_extrema: Mapping[str, Extrema],
    attempted_extrema: Mapping[str, Extrema],
    rejection_reasons: Counter[str],
    totals: AuditTotals,
) -> AuditedCase:
    if not isinstance(value, Mapping):
        raise TypeError("every result case must be a JSON object")
    accepted = value.get("accepted")
    if not isinstance(accepted, bool):
        raise TypeError("result case accepted must be Boolean")
    required = _required_integer(
        value,
        "required_bits",
        context="result case",
        minimum=0,
    )
    evaluated = _required_integer(
        value,
        "evaluated_bits",
        context="result case",
        minimum=0,
    )
    failed = _required_integer(
        value,
        "failed_bits",
        context="result case",
        minimum=0,
    )
    decision = QualityDecision.from_bits(
        QualityScope.TRAJECTORY,
        required,
        evaluated,
        failed,
    )
    if decision.accepted is not accepted:
        raise ValueError("result accepted flag disagrees with quality masks")
    case_id = _required_integer(value, "case_id", context="result case")
    if case_id != specification.case_id:
        raise ValueError("result case ID differs from its immutable proposal")
    first_row = _required_integer(
        value,
        "first_row",
        context="result case",
    )
    row_count = _required_integer(
        value,
        "row_count",
        context="result case",
        minimum=0,
    )
    if accepted and row_count != ROWS_PER_ACCEPTED_CASE:
        raise ValueError("accepted JONSWAP case does not own exactly 16 rows")
    if not accepted and (first_row != -1 or row_count != 0):
        raise ValueError("rejected JONSWAP case owns stored rows")
    status, expected_evaluated, expected_failed = _validate_case_metrics(
        value,
        accepted=accepted,
        local_index=local_index,
        specification=specification.record,
        residual_tolerance=residual_tolerance,
        hamiltonian_threshold=hamiltonian_threshold,
        production_dt=production_dt,
        saved_dt=saved_dt,
        accepted_extrema=accepted_extrema,
        attempted_extrema=attempted_extrema,
        totals=totals,
    )
    if status == "production_completed":
        totals.production_completed += 1
        if (
            required != EXPECTED_PRODUCTION_REQUIRED_BITS
            or evaluated != expected_evaluated
            or failed != expected_failed
        ):
            raise ValueError("production case does not use the exact revision-4 masks")
    elif status == "adjustment_failed":
        totals.adjustment_failed += 1
        if (
            required != EXPECTED_ADJUSTMENT_REQUIRED_BITS
            or evaluated != expected_evaluated
            or failed != expected_failed
        ):
            raise ValueError("adjustment failure does not use the exact quality masks")
    else:
        totals.construction_failed += 1
        if (
            required != EXPECTED_ADJUSTMENT_REQUIRED_BITS
            or evaluated != expected_evaluated
            or failed != expected_failed
        ):
            raise ValueError("construction rejection has the wrong required mask")
    if not accepted:
        names = "+".join(reason.name for reason in reasons_from_bits(failed))
        rejection_reasons[names] += 1
    return AuditedCase(
        case_id=case_id,
        accepted=accepted,
        required_bits=required,
        evaluated_bits=evaluated,
        failed_bits=failed,
        first_row=first_row,
        row_count=row_count,
        batch_id=batch_id,
    )


def _validate_shard_rows(
    path: Path,
    *,
    shard_artifact_path: str,
    chunk_label: str,
    split: str,
    proposal_sha256: str,
    configuration_fingerprint: str,
    specifications: Sequence[ReplayedSpecification],
    raw_cases: Sequence[Mapping[str, object]],
    cases: Sequence[AuditedCase],
    saved_dt: float,
    totals: AuditTotals,
    diagnostics: TrajectoryQualityDiagnostics,
) -> None:
    """Independently bind every stored row to one accepted proposal case."""

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "case_local_index",
        "config_fingerprint",
        "depth",
        "eta",
        "frame_index",
        "gxi",
        "proposal_sha256",
        "selected_dense_index",
        "time",
        "xi",
    }
    if set(arrays) != required:
        raise ValueError("JONSWAP shard has an unknown array schema")
    for name, dtype in EXPECTED_SHARD_DTYPES.items():
        if arrays[name].dtype != dtype:
            raise TypeError(f"JONSWAP shard {name} has the wrong dtype")
    for name in ("config_fingerprint", "proposal_sha256"):
        if arrays[name].ndim != 0 or arrays[name].dtype.kind not in {"U", "S"}:
            raise TypeError(f"JONSWAP shard {name} must be a scalar string")
    if str(arrays["config_fingerprint"].item()) != configuration_fingerprint:
        raise ValueError("JONSWAP shard has the wrong configuration fingerprint")
    if str(arrays["proposal_sha256"].item()) != proposal_sha256:
        raise ValueError("JONSWAP shard has the wrong proposal hash")
    eta = arrays["eta"]
    expected_nx = CURRENT_JONSWAP_EXECUTION.numerical.target_definition.nx
    if eta.ndim != 2 or eta.shape[1] != expected_nx:
        raise ValueError("JONSWAP shard has the wrong delivered spatial grid")
    if arrays["xi"].shape != eta.shape or arrays["gxi"].shape != eta.shape:
        raise ValueError("stored eta, xi, and G(eta)xi shapes differ")
    row_count = int(eta.shape[0])
    for name in (
        "case_local_index",
        "depth",
        "frame_index",
        "selected_dense_index",
        "time",
    ):
        if arrays[name].shape != (row_count,):
            raise ValueError(f"JONSWAP shard {name} has the wrong row shape")
    if not all(
        np.isfinite(arrays[name]).all()
        for name in ("eta", "xi", "gxi", "depth", "time")
    ):
        raise ValueError("stored eta/xi/G(eta)xi/depth/time must all be finite")
    quality_by_field = {
        "eta": _field_quality_metrics(arrays["eta"]),
        "gxi": _field_quality_metrics(arrays["gxi"]),
    }

    cursor = 0
    for local_index, (specification, raw_case, case) in enumerate(
        zip(specifications, raw_cases, cases)
    ):
        selected_rows = np.flatnonzero(arrays["case_local_index"] == local_index)
        if not case.accepted:
            if selected_rows.size:
                raise ValueError("rejected JONSWAP case owns stored rows")
            continue
        expected_rows = np.arange(cursor, cursor + ROWS_PER_ACCEPTED_CASE)
        if not np.array_equal(selected_rows, expected_rows):
            raise ValueError("accepted JONSWAP row blocks are not contiguous")
        if case.first_row != cursor:
            raise ValueError("result first_row differs from the stored shard")
        if not np.array_equal(
            arrays["frame_index"][selected_rows],
            np.arange(ROWS_PER_ACCEPTED_CASE, dtype=np.int32),
        ):
            raise ValueError("stored JONSWAP frame indices are incorrect")
        metrics = _required_mapping(raw_case, "metrics", context="result case")
        dense_count = _required_integer(
            metrics,
            "saved_time_count",
            context="result case metrics",
            minimum=ROWS_PER_ACCEPTED_CASE,
        )
        expected_dense_indices = select_uniform_times(
            dense_count,
            keep_samples=ROWS_PER_ACCEPTED_CASE,
        )
        if not np.array_equal(
            arrays["selected_dense_index"][selected_rows],
            expected_dense_indices,
        ):
            raise ValueError("stored JONSWAP dense-time selection is incorrect")
        expected_times = saved_dt * expected_dense_indices.astype(np.float64)
        if not np.array_equal(arrays["time"][selected_rows], expected_times):
            raise ValueError("stored JONSWAP times differ from their dense indices")
        realized = _finite_number(
            metrics.get("realized_terminal_time"),
            context="metrics.realized_terminal_time",
        )
        if expected_times[0] != 0.0 or expected_times[-1] != realized:
            raise ValueError("stored JONSWAP rows omit a trajectory endpoint")
        expected_depth = _finite_number(
            specification.record.get("depth"),
            context="specification.depth",
        )
        if not np.all(arrays["depth"][selected_rows] == expected_depth):
            raise ValueError("stored JONSWAP depth differs from its proposal")
        for field_name, quality in quality_by_field.items():
            for diagnostic_name, values in quality.items():
                trajectory_values = values[selected_rows]
                terminal_row = int(selected_rows[-1])
                maximum_row = int(selected_rows[np.argmax(trajectory_values)])
                for suffix, row in (
                    ("terminal", terminal_row),
                    ("maximum", maximum_row),
                ):
                    identity = DiagnosticIdentity(
                        case_id=case.case_id,
                        chunk_label=chunk_label,
                        split=split,
                        batch_id=case.batch_id,
                        local_index=local_index,
                        frame_index=int(arrays["frame_index"][row]),
                        cell_id=specification.cell_id,
                        shard_path=shard_artifact_path,
                    )
                    metric_name = f"{field_name}_{diagnostic_name}_{suffix}"
                    diagnostics.series[metric_name].add(float(values[row]), identity)
        peak_gxi_high_band = float(
            np.max(quality_by_field["gxi"]["high_band_energy_fraction"][selected_rows])
        )
        diagnostics.peak_gxi_high_band_above_0p10 += int(peak_gxi_high_band > 0.10)
        diagnostics.peak_gxi_high_band_above_0p20 += int(peak_gxi_high_band > 0.20)
        diagnostics.trajectory_count += 1
        cursor += ROWS_PER_ACCEPTED_CASE
        totals.stored_case_blocks_checked += 1
    if cursor != row_count:
        raise ValueError("JONSWAP shard contains unowned rows")


def _validate_dataset_view(
    *,
    chunk: CompletedChunk,
    summary: Mapping[str, object],
    specifications_by_batch: Mapping[int, Sequence[ReplayedSpecification]],
    cases_by_batch: Mapping[int, Sequence[AuditedCase]],
    result_by_batch: Mapping[int, Mapping[str, object]],
    totals: AuditTotals,
) -> dict[str, object]:
    """Bind the chunk manifest and row map to every immutable transaction."""

    view = _required_mapping(summary, "dataset_view", context="summary")
    manifest_record = _required_mapping(view, "manifest", context="dataset_view")
    map_record = _required_mapping(
        view,
        "trajectory_map",
        context="dataset_view",
    )
    manifest_path = (
        chunk.root
        / _required_string(
            manifest_record,
            "path",
            context="dataset_view.manifest",
        )
    ).resolve()
    map_path = (
        chunk.root
        / _required_string(
            map_record,
            "path",
            context="dataset_view.trajectory_map",
        )
    ).resolve()
    if not (
        manifest_path.is_relative_to(chunk.root.resolve())
        and map_path.is_relative_to(chunk.root.resolve())
    ):
        raise ValueError("JONSWAP dataset-view artifact escapes its chunk root")
    manifest_sha256 = file_sha256(manifest_path)
    map_sha256 = file_sha256(map_path)
    for record, path, digest, name in (
        (manifest_record, manifest_path, manifest_sha256, "manifest"),
        (map_record, map_path, map_sha256, "trajectory map"),
    ):
        if record.get("sha256") != digest:
            raise ValueError(f"summary {name} hash is stale or incorrect")
        if record.get("bytes") != path.stat().st_size:
            raise ValueError(f"summary {name} byte count is incorrect")
    totals.dataset_view_bytes_checked += (
        manifest_path.stat().st_size + map_path.stat().st_size
    )

    expected_rows = chunk.accepted_count * ROWS_PER_ACCEPTED_CASE
    expected_view_scalars = {
        "schema_version": 2,
        "configuration_fingerprint": chunk.fingerprint,
        "grid": {
            "length": CURRENT_JONSWAP_EXECUTION.numerical.length,
            "nx": CURRENT_JONSWAP_EXECUTION.numerical.delivered_nx,
        },
        "n_rows": expected_rows,
        "n_trajectories": chunk.attempted_count,
        "n_accepted_trajectories": chunk.accepted_count,
        "n_accepted_rows": expected_rows,
    }
    for name, expected in expected_view_scalars.items():
        if not _same_json(view.get(name), expected):
            raise ValueError(f"summary dataset view has incorrect {name}")

    manifest = _strict_json_object(manifest_path)
    expected_manifest_scalars = {
        **expected_view_scalars,
        "configuration_fingerprints": [chunk.fingerprint],
        "requires_trajectory_map": True,
        "trajectory_map_sha256": map_sha256,
    }
    for name, expected in expected_manifest_scalars.items():
        if not _same_json(manifest.get(name), expected):
            raise ValueError(f"dataset manifest has incorrect {name}")
    contract = _required_mapping(
        manifest,
        "dataset_contract",
        context="dataset manifest",
    )
    if manifest.get("dataset_contract_fingerprint") != canonical_json_sha256(contract):
        raise ValueError("dataset contract fingerprint is incorrect")
    execution = _required_mapping(summary, "execution", context="summary")
    numerical = _required_mapping(
        execution,
        "numerical",
        context="summary.execution",
    )
    target = {
        "role": execution.get("role"),
        "nx": numerical.get("target_nx", numerical.get("nx")),
        "length": numerical.get("length"),
        "gravity": numerical.get("gravity"),
        "dno_order": numerical.get(
            "target_dno_order",
            numerical.get("dno_order"),
        ),
        "pad_factor": numerical.get("pad_factor"),
        "maximum_wavenumber": numerical.get(
            "target_maximum_wavenumber",
            numerical.get("maximum_wavenumber"),
        ),
        "dtype": numerical.get("dtype"),
    }
    generation_identity: dict[str, object] = {
        "dependency_environment_fingerprint": chunk.dependency_fingerprint,
        "family_revisions": [
            {
                "family_id": EXPECTED_FAMILY_ID,
                "revision_id": EXPECTED_REVISION_ID,
                "execution_platform": chunk.execution_platform,
                "generation_variants": [
                    {
                        "execution_record_fingerprint": chunk.execution_fingerprint,
                        "source_sha256_fingerprint": chunk.source_fingerprint,
                        "source_sha256": dict(sorted(chunk.source_sha256.items())),
                    }
                ],
                "source_sha256_fingerprint": chunk.source_fingerprint,
            }
        ],
    }
    generation_identity["compatibility_fingerprint"] = canonical_json_sha256(
        generation_identity
    )
    expected_contract = {
        "target": target,
        "trajectory_numerical": dict(numerical),
        "trajectory_numerical_by_family_revision": [
            {
                "family_id": EXPECTED_FAMILY_ID,
                "revision_id": EXPECTED_REVISION_ID,
                "numerical": dict(numerical),
            }
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
    if not _same_json(contract, expected_contract):
        raise ValueError("JONSWAP dataset contract differs from current generation")
    if manifest.get("trajectory_map_npz") != map_path.name:
        raise ValueError("dataset manifest names the wrong trajectory map")
    expected_split_counts = {
        split.value: {
            "attempted": chunk.attempted_count if split is chunk.split else 0,
            "accepted": chunk.accepted_count if split is chunk.split else 0,
        }
        for split in SplitId
    }
    if not _same_json(manifest.get("split_counts"), expected_split_counts):
        raise ValueError("dataset manifest split counts are incorrect")

    raw_batches = _required_list(manifest, "dataset_batches", context="manifest")
    raw_shards = _required_list(manifest, "dataset_shards", context="manifest")
    if len(raw_batches) != len(chunk.batches):
        raise ValueError("dataset manifest omits a committed JONSWAP batch")
    expected_shard_count = sum(paths.shard.is_file() for paths in chunk.batches)
    if len(raw_shards) != expected_shard_count:
        raise ValueError("dataset manifest has the wrong JONSWAP shard count")
    shard_record_by_batch_position: dict[
        int,
        tuple[int, Mapping[str, object]],
    ] = {}
    for shard_index, raw_shard in enumerate(raw_shards):
        if not isinstance(raw_shard, Mapping):
            raise TypeError("every dataset shard record must be an object")
        batch_position = _required_integer(
            raw_shard,
            "batch_index",
            context="dataset shard",
            minimum=0,
        )
        if batch_position in shard_record_by_batch_position or batch_position >= len(
            chunk.batches
        ):
            raise ValueError("dataset shard has an invalid batch index")
        shard_record_by_batch_position[batch_position] = (
            shard_index,
            raw_shard,
        )

    shard_index_by_batch_id: dict[int, int] = {}
    for batch_position, (raw_batch, paths) in enumerate(
        zip(raw_batches, chunk.batches)
    ):
        if not isinstance(raw_batch, Mapping):
            raise TypeError("every dataset batch record must be an object")
        batch_id = int(paths.result.stem.removeprefix("batch_"))
        result = result_by_batch[batch_id]
        specifications = specifications_by_batch[batch_id]
        cases = cases_by_batch[batch_id]
        accepted = sum(case.accepted for case in cases)
        rows = accepted * ROWS_PER_ACCEPTED_CASE
        expected_batch_values = {
            "batch_id": batch_id,
            "configuration_fingerprint": chunk.fingerprint,
            "family_id": EXPECTED_FAMILY_ID,
            "revision_id": EXPECTED_REVISION_ID,
            "split_id": {
                SplitId.TRAIN: 0,
                SplitId.VALIDATION: 1,
                SplitId.TEST: 2,
            }[chunk.split],
            "n_attempted_trajectories": len(specifications),
            "n_accepted_trajectories": accepted,
            "n_rows": rows,
            "proposal_path": str(paths.proposal.relative_to(chunk.root)),
            "proposal_sha256": result.get("proposal_sha256"),
            "result_path": str(paths.result.relative_to(chunk.root)),
            "result_sha256": file_sha256(paths.result),
        }
        for name, expected in expected_batch_values.items():
            if raw_batch.get(name) != expected:
                raise ValueError(f"dataset batch record has incorrect {name}")
        shard_pair = shard_record_by_batch_position.get(batch_position)
        if paths.shard.is_file():
            if shard_pair is None:
                raise ValueError("dataset manifest omits an existing shard")
            shard_index, shard_record = shard_pair
            shard_index_by_batch_id[batch_id] = shard_index
            if raw_batch.get("shard_index") != shard_index:
                raise ValueError("dataset batch has the wrong shard index")
            expected_shard_values = {
                "batch_index": batch_position,
                "configuration_fingerprint": chunk.fingerprint,
                "n_rows": rows,
                "path": str(paths.shard.relative_to(chunk.root)),
                "sha256": file_sha256(paths.shard),
            }
            for name, expected in expected_shard_values.items():
                if shard_record.get(name) != expected:
                    raise ValueError(f"dataset shard record has incorrect {name}")
        elif shard_pair is not None or raw_batch.get("shard_index") is not None:
            raise ValueError("all-rejected batch incorrectly references a shard")

    with np.load(map_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if frozenset(arrays) != EXPECTED_MAP_ARRAYS:
        raise ValueError("trajectory map has an unknown array schema")
    if (
        arrays["schema_version"].ndim != 0
        or arrays["schema_version"].dtype != TRAJECTORY_MAP_DTYPES["schema_version"]
        or int(arrays["schema_version"].item()) != 2
    ):
        raise ValueError("trajectory map schema version is not 2")
    for name, dtype in TRAJECTORY_MAP_DTYPES.items():
        if name != "schema_version" and arrays[name].dtype != dtype:
            raise TypeError(f"trajectory map {name} has the wrong dtype")
    specifications = tuple(
        specification
        for paths in chunk.batches
        for specification in specifications_by_batch[
            int(paths.result.stem.removeprefix("batch_"))
        ]
    )
    cases = tuple(
        case
        for paths in chunk.batches
        for case in cases_by_batch[int(paths.result.stem.removeprefix("batch_"))]
    )
    trajectory_count = len(cases)
    if trajectory_count != chunk.attempted_count or len(specifications) != (
        trajectory_count
    ):
        raise RuntimeError("trajectory-map audit has inconsistent case counts")
    trajectory_fields = (
        "trajectory_accepted",
        "trajectory_case_id",
        "trajectory_cell_id",
        "trajectory_evaluated_bits",
        "trajectory_failed_bits",
        "trajectory_family_id",
        "trajectory_first_row",
        "trajectory_required_bits",
        "trajectory_revision_id",
        "trajectory_row_count",
        "trajectory_split_id",
    )
    if any(arrays[name].shape != (trajectory_count,) for name in trajectory_fields):
        raise ValueError("trajectory map has a malformed trajectory vector")
    expected_case_ids = np.asarray([case.case_id for case in cases], dtype=np.int64)
    expected_accepted = np.asarray([case.accepted for case in cases], dtype=np.bool_)
    expected_required = np.asarray(
        [case.required_bits for case in cases],
        dtype=np.uint32,
    )
    expected_evaluated = np.asarray(
        [case.evaluated_bits for case in cases],
        dtype=np.uint32,
    )
    expected_failed = np.asarray(
        [case.failed_bits for case in cases],
        dtype=np.uint32,
    )
    expected_row_count = np.where(
        expected_accepted,
        ROWS_PER_ACCEPTED_CASE,
        0,
    ).astype(np.int32)
    expected_first_row = np.full(trajectory_count, -1, dtype=np.int64)
    cursor = 0
    for index, case in enumerate(cases):
        if case.accepted:
            expected_first_row[index] = cursor
            cursor += ROWS_PER_ACCEPTED_CASE
    comparisons = (
        ("trajectory_case_id", expected_case_ids),
        (
            "trajectory_cell_id",
            np.asarray(
                [specification.cell_code for specification in specifications],
                dtype=np.int32,
            ),
        ),
        ("trajectory_accepted", expected_accepted),
        ("trajectory_required_bits", expected_required),
        ("trajectory_evaluated_bits", expected_evaluated),
        ("trajectory_failed_bits", expected_failed),
        ("trajectory_row_count", expected_row_count),
        ("trajectory_first_row", expected_first_row),
    )
    for name, expected in comparisons:
        if not np.array_equal(arrays[name], expected):
            raise ValueError(f"trajectory map {name} differs from result records")
    if not np.all(arrays["trajectory_family_id"] == EXPECTED_FAMILY_ID):
        raise ValueError("trajectory map has the wrong family ID")
    if not np.all(arrays["trajectory_revision_id"] == EXPECTED_REVISION_ID):
        raise ValueError("trajectory map has the wrong revision ID")
    split_code = {
        SplitId.TRAIN: 0,
        SplitId.VALIDATION: 1,
        SplitId.TEST: 2,
    }[chunk.split]
    if not np.all(arrays["trajectory_split_id"] == split_code):
        raise ValueError("trajectory map has the wrong split ID")

    row_fields = ("trajectory_index", "frame_index", "shard_index", "shard_row")
    if any(arrays[name].shape != (expected_rows,) for name in row_fields):
        raise ValueError("trajectory map has a malformed row vector")
    row_cursor = 0
    frame_indices = np.arange(ROWS_PER_ACCEPTED_CASE, dtype=np.int32)
    for trajectory_index, case in enumerate(cases):
        if not case.accepted:
            continue
        selected = slice(row_cursor, row_cursor + ROWS_PER_ACCEPTED_CASE)
        if not np.all(arrays["trajectory_index"][selected] == trajectory_index):
            raise ValueError("row-to-trajectory ownership is incorrect")
        if not np.array_equal(arrays["frame_index"][selected], frame_indices):
            raise ValueError("trajectory frame ownership is incorrect")
        if not np.all(
            arrays["shard_index"][selected] == shard_index_by_batch_id[case.batch_id]
        ):
            raise ValueError("row-to-shard ownership is incorrect")
        expected_shard_rows = case.first_row + frame_indices.astype(np.int64)
        if not np.array_equal(arrays["shard_row"][selected], expected_shard_rows):
            raise ValueError("stored shard-row ownership is incorrect")
        row_cursor += ROWS_PER_ACCEPTED_CASE
    if row_cursor != expected_rows:
        raise RuntimeError("accepted JONSWAP case rows do not sum to the manifest")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "trajectory_map_path": str(map_path),
        "trajectory_map_sha256": map_sha256,
        "dataset_contract_fingerprint": manifest["dataset_contract_fingerprint"],
    }


def _audit_chunk(
    expected: ExpectedChunk,
    chunk: CompletedChunk,
    *,
    totals: AuditTotals,
    support_extrema: Mapping[str, Extrema],
    accepted_extrema: Mapping[str, Extrema],
    attempted_extrema: Mapping[str, Extrema],
    rejection_reasons: Counter[str],
    diagnostics: TrajectoryQualityDiagnostics,
) -> dict[str, object]:
    summary = _strict_json_object(chunk.summary_path)
    run_spec = _required_mapping(summary, "run_spec", context="summary")
    execution = _required_mapping(summary, "execution", context="summary")
    numerical = _required_mapping(
        execution,
        "numerical",
        context="summary.execution",
    )
    residual_tolerance = _finite_number(
        numerical.get("gl2_residual_tolerance"),
        context="execution.numerical.gl2_residual_tolerance",
    )
    hamiltonian_threshold = _finite_number(
        numerical.get("internal_hamiltonian_drift_threshold"),
        context="execution.numerical.internal_hamiltonian_drift_threshold",
    )
    production_dt = _finite_number(
        numerical.get("dt"),
        context="execution.numerical.dt",
    )
    saved_dt = _finite_number(
        numerical.get("saved_dt"),
        context="execution.numerical.saved_dt",
    )
    raw_codes = _required_mapping(run_spec, "cell_codes", context="run_spec")
    cell_ids_by_code = {
        _required_integer(raw_codes, str(cell_id), context="run_spec.cell_codes"): str(
            cell_id
        )
        for cell_id in raw_codes
    }

    cases: list[AuditedCase] = []
    attempt_indices: list[int] = []
    attempted_by_cell: Counter[str] = Counter()
    accepted_by_cell: Counter[str] = Counter()
    rejected_by_cell: Counter[str] = Counter()
    specifications_by_batch: dict[int, tuple[ReplayedSpecification, ...]] = {}
    cases_by_batch: dict[int, tuple[AuditedCase, ...]] = {}
    result_by_batch: dict[int, Mapping[str, object]] = {}
    for paths in chunk.batches:
        batch_id = int(paths.result.stem.removeprefix("batch_"))
        totals.committed_batches += 1
        totals.transaction_artifact_bytes_checked += sum(
            path.stat().st_size for path in (paths.proposal, paths.shard, paths.result)
        )
        _scan_finite_archive(paths.proposal, totals=totals)
        _scan_finite_archive(paths.shard, totals=totals)
        specifications = _case_specifications(
            paths.proposal,
            split=chunk.split,
            stream_id=chunk.stream_id,
            cell_ids_by_code=cell_ids_by_code,
            support_extrema=support_extrema,
            totals=totals,
        )
        attempt_indices.extend(
            specification.attempt_index for specification in specifications
        )
        result = _strict_json_object(paths.result)
        _validate_result_metadata(
            result,
            run_spec=run_spec,
            execution=execution,
            context=str(paths.result),
        )
        values = _required_list(result, "cases", context=str(paths.result))
        if len(values) != len(specifications):
            raise ValueError("result omits an attempted JONSWAP proposal case")
        if result.get("config_fingerprint") != chunk.fingerprint:
            raise ValueError("result has the wrong configuration fingerprint")
        proposal_sha256 = file_sha256(paths.proposal)
        shard_sha256 = file_sha256(paths.shard)
        if result.get("proposal_sha256") != proposal_sha256:
            raise ValueError("result has the wrong proposal hash")
        if result.get("shard_sha256") != shard_sha256:
            raise ValueError("result has the wrong shard hash")
        batch_cases = tuple(
            _validate_case_record(
                value,
                batch_id=batch_id,
                local_index=local_index,
                specification=specification,
                residual_tolerance=residual_tolerance,
                hamiltonian_threshold=hamiltonian_threshold,
                production_dt=production_dt,
                saved_dt=saved_dt,
                accepted_extrema=accepted_extrema,
                attempted_extrema=attempted_extrema,
                rejection_reasons=rejection_reasons,
                totals=totals,
            )
            for local_index, (value, specification) in enumerate(
                zip(values, specifications)
            )
        )
        raw_cases = tuple(value for value in values if isinstance(value, Mapping))
        if len(raw_cases) != len(values):
            raise TypeError("every result case must be a JSON object")
        _validate_shard_rows(
            paths.shard,
            shard_artifact_path=str(paths.shard.relative_to(chunk.root)),
            chunk_label=expected.label,
            split=chunk.split.value,
            proposal_sha256=proposal_sha256,
            configuration_fingerprint=chunk.fingerprint,
            specifications=specifications,
            raw_cases=raw_cases,
            cases=batch_cases,
            saved_dt=saved_dt,
            totals=totals,
            diagnostics=diagnostics,
        )
        accepted_by_cell.update(
            specification.cell_id
            for specification, case in zip(specifications, batch_cases)
            if case.accepted
        )
        attempted_by_cell.update(
            specification.cell_id for specification in specifications
        )
        rejected_by_cell.update(
            specification.cell_id
            for specification, case in zip(specifications, batch_cases)
            if not case.accepted
        )
        cases.extend(batch_cases)
        specifications_by_batch[batch_id] = specifications
        cases_by_batch[batch_id] = batch_cases
        result_by_batch[batch_id] = result

    attempted = len(cases)
    accepted = sum(case.accepted for case in cases)
    rejected = attempted - accepted
    retained_rows = sum(case.row_count for case in cases)
    if attempt_indices != list(range(attempted)):
        raise ValueError("JONSWAP attempts are not retained exactly once in order")
    if len({case.case_id for case in cases}) != attempted:
        raise ValueError("JONSWAP case IDs repeat within a completed chunk")
    if (
        attempted != chunk.attempted_count
        or accepted != chunk.accepted_count
        or retained_rows != accepted * ROWS_PER_ACCEPTED_CASE
    ):
        raise ValueError("transaction counts disagree with the completed chunk")
    raw_quotas = _required_list(run_spec, "quotas", context="run_spec")
    expected_by_cell: dict[str, int] = {}
    for raw_quota in raw_quotas:
        if not isinstance(raw_quota, Mapping):
            raise TypeError("run_spec.quotas must contain objects")
        cell_id = _required_string(raw_quota, "cell_id", context="run_spec quota")
        expected_by_cell[cell_id] = _required_integer(
            raw_quota,
            "target_accepted",
            context="run_spec quota",
            minimum=0,
        )
    if set(expected_by_cell) != {cell.cell_id for cell in JONSWAP_TMA_SAMPLE_CELLS}:
        raise ValueError("run specification does not contain the exact 27 cells")
    if dict(accepted_by_cell) != {
        cell_id: count for cell_id, count in expected_by_cell.items() if count
    }:
        raise ValueError("actual accepted cases do not meet the exact 27-cell quotas")
    counts = _required_mapping(summary, "counts", context="summary")
    by_cell = _required_mapping(counts, "by_cell", context="summary.counts")
    if set(by_cell) != set(expected_by_cell):
        raise ValueError("summary cell counts do not contain the exact 27 cells")
    cell_counts: dict[str, object] = {}
    for cell_id, target in expected_by_cell.items():
        record = _required_mapping(by_cell, cell_id, context="summary.counts.by_cell")
        observed = {
            name: _required_integer(
                record,
                name,
                context=f"summary.counts.by_cell.{cell_id}",
                minimum=0,
            )
            for name in ("target_accepted", "attempted", "accepted", "rejected")
        }
        expected_counts = {
            "target_accepted": target,
            "attempted": attempted_by_cell[cell_id],
            "accepted": accepted_by_cell[cell_id],
            "rejected": rejected_by_cell[cell_id],
        }
        if observed != expected_counts:
            raise ValueError(
                "summary cell counts differ from exact attempted/accepted/rejected counts"
            )
        if observed["attempted"] != observed["accepted"] + observed["rejected"]:
            raise ValueError(
                "summary cell attempted/accepted/rejected counts do not close"
            )
        cell_counts[cell_id] = {
            **observed,
            "rejection_rate": _rejection_rate(
                attempted=observed["attempted"],
                rejected=observed["rejected"],
                context=f"summary.counts.by_cell.{cell_id}",
            ),
        }
    totals.attempted += attempted
    totals.accepted += accepted
    totals.rejected += rejected
    totals.retained_rows += retained_rows
    view_record = _validate_dataset_view(
        chunk=chunk,
        summary=summary,
        specifications_by_batch=specifications_by_batch,
        cases_by_batch=cases_by_batch,
        result_by_batch=result_by_batch,
        totals=totals,
    )
    return {
        "label": expected.label,
        "split": chunk.split.value,
        "stream_id": chunk.stream_id,
        "accepted_before": chunk.accepted_before,
        "accepted_count": accepted,
        "accepted_after": chunk.accepted_after,
        "attempted_count": attempted,
        "rejected_count": rejected,
        "retained_rows": retained_rows,
        "committed_batches": len(chunk.batches),
        "accepted_by_cell": dict(sorted(accepted_by_cell.items())),
        "cell_counts": dict(sorted(cell_counts.items())),
        "configuration_fingerprint": chunk.fingerprint,
        "summary_path": str(chunk.summary_path),
        "summary_sha256": file_sha256(chunk.summary_path),
        **view_record,
    }


def _load_exact_chunks(root: Path) -> tuple[CompletedChunk, ...]:
    expected_paths = {
        (root / expected.summary_path).resolve() for expected in EXPECTED_CHUNKS
    }
    discovered_paths = {
        path.resolve()
        for path in root.glob("*/jonswap_tma/*/paper_dataset_jonswap_tma_*.summary.json")
    }
    if discovered_paths != expected_paths:
        missing = sorted(map(str, expected_paths - discovered_paths))
        unexpected = sorted(map(str, discovered_paths - expected_paths))
        raise ValueError(
            "JONSWAP completion summaries differ from the exact eight-chunk plan; "
            f"missing={missing}, unexpected={unexpected}"
        )
    chunks = tuple(
        load_completed_chunk(root / expected.summary_path)
        for expected in EXPECTED_CHUNKS
    )
    for expected, chunk in zip(EXPECTED_CHUNKS, chunks):
        observed = (
            chunk.family,
            chunk.revision_id,
            chunk.split,
            chunk.stream_id,
            chunk.accepted_before,
            chunk.accepted_count,
            chunk.accepted_after,
        )
        wanted = (
            "jonswap_tma",
            EXPECTED_REVISION_ID,
            expected.split,
            expected.stream_id,
            expected.accepted_before,
            expected.accepted_count,
            expected.accepted_before + expected.accepted_count,
        )
        if observed != wanted:
            raise ValueError(f"{expected.label} differs from its immutable plan")
        if chunk.execution_platform != "gpu":
            raise ValueError("fresh JONSWAP release dataset was not generated on GPU")
    return chunks


def _current_support_record() -> dict[str, object]:
    """Return an elementary complete identity for the revision-4 support."""

    band = resolved_band_for_contract(
        CURRENT_JONSWAP_EXECUTION.numerical,
        quadrature_order=int(CURRENT_JONSWAP_EXECUTION.jonswap_quadrature_order),
    )
    return {
        "schema": "paper_jonswap_tma_sampling_support_v4",
        "allocation_cells": [
            {
                "cell_id": cell.cell_id,
                "stratum": cell.stratum,
                "peak_enhancement": cell.peak_enhancement,
                "right_moving_fraction": cell.right_moving_fraction,
            }
            for cell in JONSWAP_TMA_SAMPLE_CELLS
        ],
        "finite": {
            "peak_wavenumber_uniform_bounds": list(FINITE_PEAK_WAVENUMBER_BOUNDS),
            "depth_uniform_bounds": list(FINITE_DEPTH_BOUNDS),
            "significant_height_uniform_bounds": list(SIGNIFICANT_HEIGHT_BOUNDS),
        },
        "deep": {
            "peak_wavenumber_uniform_bounds": list(DEEP_PEAK_WAVENUMBER_BOUNDS),
            "depth_uniform_bounds": list(DEEP_DEPTH_BOUNDS),
            "significant_height_uniform_bounds": list(SIGNIFICANT_HEIGHT_BOUNDS),
        },
        "shallow": {
            "peak_modes": list(PAPER_SHALLOW_PEAK_MODES),
            "depth_wavenumber_uniform_bounds": list(SHALLOW_DEPTH_WAVENUMBER_BOUNDS),
            "relative_height_uniform_bounds": list(SHALLOW_RELATIVE_HEIGHT_BOUNDS),
        },
        "peak_enhancements": list(PAPER_PEAK_ENHANCEMENTS),
        "right_moving_fractions": list(PAPER_RIGHT_MOVING_FRACTIONS),
        "peak_steepness_upper_inclusive": PAPER_PEAK_STEEPNESS_MAXIMUM,
        "relative_frequency_interval": [
            PAPER_RELATIVE_FREQUENCY_MINIMUM,
            PAPER_RELATIVE_FREQUENCY_MAXIMUM,
        ],
        "density_window": PAPER_RELATIVE_FREQUENCY_WINDOW,
        "resolved_band": {
            "length": band.length,
            "maximum_wavenumber": band.maximum_wavenumber,
            "transition_wavenumber": band.transition_wavenumber,
            "quadrature_order": band.quadrature_order,
            "phase_count_per_direction": int(positive_mode_wavenumbers(band=band).size),
        },
        "phases": {
            "law": "independent_uniform",
            "interval": "[0, 2*pi)",
            "generator": "PCG64 seeded by complete CaseKey.seed_words",
            "directions": ["right", "left"],
        },
    }


def _identity_record(chunks: Sequence[CompletedChunk]) -> dict[str, object]:
    dependency_fingerprints = {chunk.dependency_fingerprint for chunk in chunks}
    execution_fingerprints = {chunk.execution_fingerprint for chunk in chunks}
    source_fingerprints = {chunk.source_fingerprint for chunk in chunks}
    if len(dependency_fingerprints) != 1:
        raise ValueError("JONSWAP chunks use different dependency environments")
    if len(execution_fingerprints) != 1:
        raise ValueError("JONSWAP chunks use different execution contracts")
    if len(source_fingerprints) != 1:
        raise ValueError("JONSWAP chunks use different generation source mappings")
    current_execution = CURRENT_JONSWAP_EXECUTION.to_json_record()
    current_execution_fingerprint = canonical_json_sha256(current_execution)
    execution_fingerprint = next(iter(execution_fingerprints))
    if execution_fingerprint != current_execution_fingerprint:
        raise ValueError("JONSWAP chunks do not use the current execution contract")
    current_dependency_fingerprint = canonical_json_sha256(dependency_environment())
    dependency_fingerprint = next(iter(dependency_fingerprints))
    if dependency_fingerprint != current_dependency_fingerprint:
        raise ValueError("JONSWAP chunks do not use the current dependency identity")

    first_sources = dict(chunks[0].source_sha256)
    current_sources = source_hashes("jonswap_tma")
    current_sources.update(
        {
            str(path.resolve().relative_to(ROOT)): file_sha256(path)
            for path in EXTRA_SOURCE_PATHS
        }
    )
    if first_sources != current_sources:
        raise ValueError(
            "current JONSWAP construction/support sources differ from generation"
        )
    current_support = _current_support_record()
    support_fingerprint = canonical_json_sha256(current_support)
    expected_policy = policy_record(
        outer_proposal_size=EXPECTED_BATCH_SIZE,
        solver_batch_size=EXPECTED_SOLVER_BATCH_SIZE,
        adjustment_policy=CURRENT_JONSWAP_EXECUTION.jonswap_adjustment,
    )
    for chunk in chunks:
        summary = _strict_json_object(chunk.summary_path)
        run_spec = _required_mapping(summary, "run_spec", context="summary")
        if (
            run_spec.get("batch_size") != EXPECTED_BATCH_SIZE
            or run_spec.get("first_attempt_index") != 0
            or run_spec.get("maximum_attempts_per_accepted_case")
            != EXPECTED_MAXIMUM_ATTEMPTS_PER_ACCEPTED_CASE
        ):
            raise ValueError("JONSWAP chunk differs from the exact quota-run policy")
        configuration = _required_mapping(
            run_spec,
            "configuration",
            context="summary.run_spec",
        )
        configured_policy = _required_mapping(
            configuration,
            POLICY_KEY,
            context="summary.run_spec.configuration",
        )
        if not _same_json(configured_policy, expected_policy):
            raise ValueError("JONSWAP chunk has a noncurrent bucketing policy")
        ordered_cells = _required_list(
            configuration,
            "ordered_cell_ids",
            context="summary.run_spec.configuration",
        )
        if ordered_cells != [cell.cell_id for cell in JONSWAP_TMA_SAMPLE_CELLS]:
            raise ValueError("JONSWAP chunk has a noncurrent 27-cell ordering")
    current_support_hashes = {
        path: current_sources[path]
        for path in (
            "solver/gen_data/jonswap_tma.py",
            "solver/gen_data/jonswap_tma_sampling.py",
            "solver/gen_data/trajectory_family_adapters.py",
            "solver/gen_data/jonswap_horizon_executor.py",
            "scripts/run_paper_dataset_jonswap_bucketed.py",
        )
    }
    return {
        "dependency_environment_fingerprint": dependency_fingerprint,
        "execution_record_fingerprint": execution_fingerprint,
        "source_sha256_fingerprint": next(iter(source_fingerprints)),
        "sampling_support_fingerprint": support_fingerprint,
        "sampling_support": current_support,
        "bucketing_policy_fingerprint": canonical_json_sha256(expected_policy),
        "current_support_source_sha256": current_support_hashes,
    }


def audit(root: Path) -> dict[str, object]:
    """Run the complete read-only JONSWAP dataset audit and return its record."""

    resolved_root = Path(root).expanduser().resolve()
    started_at = datetime.now().astimezone()
    started = perf_counter()
    chunks = _load_exact_chunks(resolved_root)
    identity = _identity_record(chunks)
    totals = AuditTotals()
    support_extrema = {
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
    accepted_extrema = {
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
    attempted_extrema = {
        name: Extrema()
        for name in (
            "hamiltonian_drift",
            "stage_residual",
            "minimum_water_column",
            "adjustment_stage_residual",
            "adjustment_minimum_water_column",
        )
    }
    rejection_reasons: Counter[str] = Counter()
    diagnostics = TrajectoryQualityDiagnostics()
    chunk_records = tuple(
        _audit_chunk(
            expected,
            chunk,
            totals=totals,
            support_extrema=support_extrema,
            accepted_extrema=accepted_extrema,
            attempted_extrema=attempted_extrema,
            rejection_reasons=rejection_reasons,
            diagnostics=diagnostics,
        )
        for expected, chunk in zip(EXPECTED_CHUNKS, chunks)
    )
    accepted_by_split = Counter(
        {
            split.value: sum(
                int(record["accepted_count"])
                for record in chunk_records
                if record["split"] == split.value
            )
            for split in SplitId
        }
    )
    if dict(accepted_by_split) != {
        "train": EXPECTED_TRAIN_ACCEPTED,
        "validation": EXPECTED_VALIDATION_ACCEPTED,
        "test": EXPECTED_TEST_ACCEPTED,
    }:
        raise ValueError("JONSWAP split totals differ from the paper dataset plan")
    if totals.accepted != EXPECTED_ACCEPTED:
        raise ValueError("JONSWAP accepted total differs from 18,432")
    if totals.proposal_specs_checked != totals.attempted:
        raise ValueError("not every JONSWAP proposal specification was replayed")
    if totals.retained_rows != EXPECTED_ACCEPTED * ROWS_PER_ACCEPTED_CASE:
        raise ValueError("JONSWAP retained-row total is incorrect")
    if len(JONSWAP_TMA_SAMPLE_CELLS) != 27:
        raise ValueError("current JONSWAP support no longer has 27 allocation cells")
    if totals.stored_case_blocks_checked != totals.accepted:
        raise ValueError("not every accepted JONSWAP trajectory owns one row block")
    if diagnostics.trajectory_count != totals.accepted:
        raise RuntimeError(
            "trajectory diagnostics do not cover every accepted JONSWAP case"
        )
    if (
        totals.construction_failed
        + totals.adjustment_failed
        + totals.production_completed
        != totals.attempted
    ):
        raise RuntimeError("JONSWAP attempt-status accounting is incomplete")
    if totals.adjustment_horizons_checked != (
        totals.adjustment_failed + totals.production_completed
    ):
        raise RuntimeError("not every constructed JONSWAP adjustment was audited")
    if totals.autonomous_horizons_checked != totals.production_completed:
        raise RuntimeError("not every autonomous JONSWAP horizon was audited")
    population_conditioning = _population_conditioning_record(
        chunk_records,
        attempted=totals.attempted,
        accepted=totals.accepted,
        rejected=totals.rejected,
    )

    elapsed = perf_counter() - started
    return {
        "schema": AUDIT_SCHEMA,
        "status": "pass",
        "dataset_root": str(resolved_root),
        "accepted": totals.accepted,
        "attempted": totals.attempted,
        "rejected": totals.rejected,
        "retained_rows": totals.retained_rows,
        "committed_batches": totals.committed_batches,
        "proposal_specs_checked": totals.proposal_specs_checked,
        "adjustment_horizons_checked": totals.adjustment_horizons_checked,
        "autonomous_horizons_checked": totals.autonomous_horizons_checked,
        "stored_case_blocks_checked": totals.stored_case_blocks_checked,
        "attempt_status_counts": {
            "construction_failed": totals.construction_failed,
            "adjustment_failed": totals.adjustment_failed,
            "production_completed": totals.production_completed,
        },
        "finite_float_arrays_checked": totals.float_arrays_checked,
        "finite_float_values_checked": totals.float_values_checked,
        "transaction_artifact_bytes_checked": (
            totals.transaction_artifact_bytes_checked
        ),
        "dataset_view_bytes_checked": totals.dataset_view_bytes_checked,
        "accepted_by_split": dict(accepted_by_split),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "population_conditioning": population_conditioning,
        "support": {
            "allocation_cell_count": len(JONSWAP_TMA_SAMPLE_CELLS),
            "extrema": {
                name: extrema.record() for name, extrema in support_extrema.items()
            },
        },
        "numerical_extrema": {
            "accepted": {
                name: extrema.record() for name, extrema in accepted_extrema.items()
            },
            "all_finite_attempted": {
                name: extrema.record() for name, extrema in attempted_extrema.items()
            },
        },
        "trajectory_quality_diagnostics": diagnostics.record(),
        "identity": identity,
        "chunks": list(chunk_records),
        "checks": {
            "exact_eight_chunk_plan": True,
            "nested_train_intervals": True,
            "balanced_27_cell_quotas": True,
            "immutable_parameter_and_phase_specs_rebuilt": True,
            "all_transactions_rescanned": True,
            "proposal_hashes_verified": True,
            "result_hashes_verified": True,
            "shard_hashes_verified": True,
            "manifest_hashes_verified": True,
            "trajectory_map_hashes_verified": True,
            "accepted_rejected_row_ownership_verified": True,
            "all_stored_float_fields_finite": True,
            "all_quality_masks_current_and_consistent": True,
            "accepted_adjustment_and_autonomous_residuals_within_contract": True,
            "accepted_autonomous_hamiltonian_drift_within_contract": True,
            "accepted_adjustment_and_autonomous_water_columns_positive": True,
            "complete_adjustment_and_autonomous_horizons_verified": True,
            "all_proposals_in_current_relative_frequency_support": True,
            "current_jonswap_specs_and_phases_replayed_exactly": True,
            "source_dependency_execution_bucketing_support_identity_verified": True,
            "all_attempts_retained_exactly_once": True,
            "population_conditioning_semantics_and_27_cell_counts_verified": True,
            "pending_batches": 0,
            "terminal_failures": 0,
            "attempt_limit_failures": 0,
        },
        "timing_seconds": elapsed,
        "invocation_started_at": started_at.isoformat(),
        "invocation_finished_at": datetime.now().astimezone().isoformat(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Completed revision-4 JONSWAP dataset root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Audit artifact path; defaults to <root>/jonswap_tma_completion_audit.json."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "jonswap_tma_completion_audit.json"
    )
    record = audit(root)
    write_json_atomic(output, record)
    print(
        json.dumps(
            {
                "status": record["status"],
                "accepted": record["accepted"],
                "attempted": record["attempted"],
                "rejected": record["rejected"],
                "retained_rows": record["retained_rows"],
                "timing_seconds": record["timing_seconds"],
                "audit_path": str(output),
                "audit_sha256": file_sha256(output),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
