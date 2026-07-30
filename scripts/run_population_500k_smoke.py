"""Sample and audit 500,000 paper-corpus parameter specifications on CPU.

This is a population-law smoke test.  It does not construct all fields,
evaluate DNO targets, or integrate trajectories.  Every sampled specification
contributes to the compact scalar archives and distribution diagnostics.  The
complete specification is exactly replayable from its persisted case key.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, TypeAlias

# This smoke is deliberately CPU-only.  These assignments precede imports
# that can initialize JAX through the Stokes sampler.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from solver.gen_data.benjamin_feir_population import (  # noqa: E402
    BENJAMIN_FEIR_POPULATION_CELLS,
    BenjaminFeirPopulationSample,
    benjamin_feir_support_violations,
    sample_benjamin_feir_population,
)
from solver.gen_data.jonswap_tma import ResolvedBand  # noqa: E402
from solver.gen_data.jonswap_tma_population import (  # noqa: E402
    DEEP_DEPTH_BOUNDS,
    DEEP_PEAK_WAVENUMBER_BOUNDS,
    FINITE_DEPTH_BOUNDS,
    FINITE_PEAK_WAVENUMBER_BOUNDS,
    JONSWAP_TMA_POPULATION_CELLS,
    SHALLOW_DEPTH_WAVENUMBER_BOUNDS,
    SHALLOW_RELATIVE_HEIGHT_BOUNDS,
    SIGNIFICANT_HEIGHT_BOUNDS,
    JonswapTmaPopulationSample,
    sample_jonswap_tma_population,
)
from solver.gen_data.pipeline.archive import (  # noqa: E402
    file_sha256,
    write_json_atomic,
)
from solver.gen_data.pipeline.production import (  # noqa: E402
    PAPER_CORPUS_REVISION_ID,
    AttemptAssignment,
    CaseKey,
    CellQuota,
    PhysicalFamilyId,
    SplitId,
    balanced_cell_quotas,
    schedule_attempt_batch,
)
from solver.gen_data.stokes_population import (  # noqa: E402
    DEFAULT_MAXIMUM_URSELL_REDRAWS,
    STOKES_POPULATION_CELLS,
    StokesPopulationSample,
    StokesPopulationSamplingError,
    sample_stokes_population,
    stokes_support_violations,
)
from solver.gen_data.tanaka_population import (  # noqa: E402
    MAIN_DEPTH_BOUNDS,
    MAIN_TOTAL_ALPHA_BOUNDS,
    STEEP_ALPHA_BOUNDS,
    STEEP_DEPTH_BOUNDS,
    TANAKA_POPULATION_CELLS,
    TanakaPopulationSample,
    sample_tanaka_population,
    tanaka_support_violations,
)

jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "outputs/paper_corpus_population_smoke_500k_20260725"
)
DEFAULT_TOTAL_SAMPLES = 500_000
DEFAULT_BATCH_SIZE = 4_096
SMOKE_STREAM_ID = 500_000
PHASE_HISTOGRAM_BINS = 64
SENTINEL_ROLES = ("first", "middle", "last")
PAPER_BAND = ResolvedBand(
    length=2.0 * math.pi,
    maximum_wavenumber=128.0,
    transition_wavenumber=96.0,
    quadrature_order=16,
)
FAMILY_ORDER = (
    "stokes",
    "tanaka",
    "benjamin_feir",
    "jonswap_tma",
)
SOURCE_PATHS = (
    Path(__file__).resolve(),
    ROOT / "solver/gen_data/pipeline/production.py",
    ROOT / "solver/gen_data/stokes_population.py",
    ROOT / "solver/gen_data/tanaka_population.py",
    ROOT / "solver/gen_data/benjamin_feir_population.py",
    ROOT / "solver/gen_data/benjamin_feir_jcp09.py",
    ROOT / "solver/gen_data/jonswap_tma_population.py",
    ROOT / "solver/gen_data/jonswap_tma.py",
    ROOT / "solver/data/stokes_truth_jax.py",
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.integer[Any]]
BoolArray: TypeAlias = NDArray[np.bool_]
ArrayMap: TypeAlias = dict[str, NDArray[Any]]
JsonRecord: TypeAlias = dict[str, object]


@dataclass(frozen=True)
class FamilyContract:
    """Stable family identity and ordered allocation cells."""

    name: str
    family_id: PhysicalFamilyId
    cell_ids: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedFamily:
    """One compact family archive and its strict summary."""

    contract: FamilyContract
    arrays: ArrayMap
    summary: JsonRecord
    failures: tuple[JsonRecord, ...] = ()


FAMILY_CONTRACTS = {
    "stokes": FamilyContract(
        "stokes",
        PhysicalFamilyId.STOKES,
        tuple(cell.cell_id for cell in STOKES_POPULATION_CELLS),
    ),
    "tanaka": FamilyContract(
        "tanaka",
        PhysicalFamilyId.TANAKA,
        tuple(cell.cell_id for cell in TANAKA_POPULATION_CELLS),
    ),
    "benjamin_feir": FamilyContract(
        "benjamin_feir",
        PhysicalFamilyId.BENJAMIN_FEIR,
        tuple(cell.cell_id for cell in BENJAMIN_FEIR_POPULATION_CELLS),
    ),
    "jonswap_tma": FamilyContract(
        "jonswap_tma",
        PhysicalFamilyId.JONSWAP_TMA,
        tuple(cell.cell_id for cell in JONSWAP_TMA_POPULATION_CELLS),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-samples", type=int, default=DEFAULT_TOTAL_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def source_hashes() -> dict[str, str]:
    """Return hashes of all sampling-law sources used by this smoke."""

    return {_relative(path): file_sha256(path) for path in SOURCE_PATHS}


def _strict_json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _configuration(
    *,
    total_samples: int,
    batch_size: int,
) -> JsonRecord:
    if total_samples <= 0 or total_samples % len(FAMILY_ORDER):
        raise ValueError("total_samples must be positive and divisible by four")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return {
        "schema": "paper_corpus_population_smoke_configuration_v1",
        "evidence_role": "population_law_smoke_only",
        "total_accepted_specifications": total_samples,
        "accepted_specifications_per_family": total_samples // len(FAMILY_ORDER),
        "family_order": list(FAMILY_ORDER),
        "split_id": SplitId.TEST.value,
        "root_seed": 2026072205,
        "revision_id": PAPER_CORPUS_REVISION_ID,
        "stream_id": SMOKE_STREAM_ID,
        "batch_size": batch_size,
        "stokes_maximum_ursell_redraws": DEFAULT_MAXIMUM_URSELL_REDRAWS,
        "jonswap_tma_resolved_band": {
            "length": PAPER_BAND.length,
            "maximum_wavenumber": PAPER_BAND.maximum_wavenumber,
            "transition_wavenumber": PAPER_BAND.transition_wavenumber,
            "quadrature_order": PAPER_BAND.quadrature_order,
            "phase_count_per_direction": 128,
        },
        "claim_scope": (
            "parameter sampling and support only; no all-case field "
            "construction, DNO targets, GL2 trajectories, or numerical "
            "acceptance rates"
        ),
        "source_sha256": source_hashes(),
    }


def _fingerprint(configuration: Mapping[str, object]) -> str:
    return hashlib.sha256(_strict_json_bytes(configuration)).hexdigest()


def _common_arrays(count: int) -> ArrayMap:
    return {
        "attempt_index": np.empty(count, dtype=np.int64),
        "case_id": np.empty(count, dtype=np.int64),
        "cell_code": np.empty(count, dtype=np.int16),
    }


def _initialize_float(count: int, *, columns: int | None = None) -> FloatArray:
    shape = (count,) if columns is None else (count, columns)
    return np.full(shape, np.nan, dtype=np.float64)


def _initialize_int(
    count: int,
    *,
    dtype: np.dtype[Any] = np.dtype(np.int16),
    columns: int | None = None,
    fill: int = 0,
) -> IntArray:
    shape = (count,) if columns is None else (count, columns)
    return np.full(shape, fill, dtype=dtype)


def _case_key(assignment: AttemptAssignment) -> CaseKey:
    return assignment.case_key


def _store_common(
    arrays: ArrayMap,
    row: int,
    assignment: AttemptAssignment,
    cell_codes: Mapping[str, int],
) -> None:
    arrays["attempt_index"][row] = assignment.case_key.attempt_index
    arrays["case_id"][row] = assignment.case_key.case_id
    arrays["cell_code"][row] = cell_codes[assignment.cell_id]


def _scheduled_batch(
    contract: FamilyContract,
    quotas: Sequence[CellQuota],
    accepted_by_cell: Mapping[str, int],
    *,
    first_attempt_index: int,
    batch_size: int,
) -> tuple[AttemptAssignment, ...]:
    return schedule_attempt_batch(
        quotas,
        accepted_by_cell,
        family_id=int(contract.family_id),
        revision_id=PAPER_CORPUS_REVISION_ID,
        split_id=SplitId.TEST,
        stream_id=SMOKE_STREAM_ID,
        first_attempt_index=first_attempt_index,
        batch_size=batch_size,
    )


def _quota_record(
    contract: FamilyContract,
    quotas: Sequence[CellQuota],
    observed: Mapping[str, int],
) -> JsonRecord:
    expected = {quota.cell_id: quota.target_accepted for quota in quotas}
    if dict(observed) != expected:
        raise RuntimeError(
            f"{contract.name} observed cell counts do not equal quotas"
        )
    values = tuple(expected.values())
    return {
        "cell_order": list(contract.cell_ids),
        "expected_by_cell": expected,
        "observed_by_cell": dict(observed),
        "minimum_cell_count": min(values),
        "maximum_cell_count": max(values),
        "maximum_minus_minimum": max(values) - min(values),
    }


def _finite_summary(values: NDArray[Any]) -> JsonRecord:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {"finite_count": 0, "nonfinite_count": int(array.size)}
    quantile_levels = np.asarray(
        (0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0),
        dtype=np.float64,
    )
    quantiles = np.quantile(finite, quantile_levels)
    return {
        "finite_count": int(finite.size),
        "nonfinite_count": int(array.size - finite.size),
        "mean": float(np.mean(finite)),
        "standard_deviation": float(np.std(finite)),
        "quantiles": {
            f"{level:g}": float(value)
            for level, value in zip(quantile_levels, quantiles)
        },
    }


def uniform_ecdf_distance(values: NDArray[Any]) -> float:
    """Return ``sup_u |F_n(u)-u|`` for values expected on ``[0,1]``."""

    array = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if not array.size:
        raise ValueError("uniform ECDF distance requires at least one value")
    if not np.isfinite(array).all() or array[0] < 0.0 or array[-1] > 1.0:
        raise ValueError("uniform ECDF values must be finite and lie in [0,1]")
    n = array.size
    upper = np.arange(1, n + 1, dtype=np.float64) / n - array
    lower = array - np.arange(0, n, dtype=np.float64) / n
    return float(max(np.max(upper), np.max(lower)))


def _uniform_check(values: NDArray[Any]) -> JsonRecord:
    n = int(np.asarray(values).size)
    return {
        "sample_count": n,
        "ecdf_supremum_distance": uniform_ecdf_distance(values),
        "dkw_95_half_width": math.sqrt(math.log(2.0 / 0.05) / (2.0 * n)),
    }


def _base_family_summary(
    contract: FamilyContract,
    quotas: Sequence[CellQuota],
    accepted_by_cell: Mapping[str, int],
    *,
    accepted_count: int,
    attempted_count: int,
    support_violation_count: int,
    timing_seconds: float,
) -> JsonRecord:
    return {
        "family": contract.name,
        "accepted_count": accepted_count,
        "attempted_count": attempted_count,
        "rejected_count": attempted_count - accepted_count,
        "support_violation_count": support_violation_count,
        "quota": _quota_record(contract, quotas, accepted_by_cell),
        "timing_seconds": timing_seconds,
    }


def generate_stokes(
    count: int,
    *,
    batch_size: int,
) -> GeneratedFamily:
    """Generate exactly ``count`` accepted Stokes specifications."""

    contract = FAMILY_CONTRACTS["stokes"]
    quotas = balanced_cell_quotas(contract.cell_ids, accepted_case_count=count)
    cell_codes = {cell_id: index for index, cell_id in enumerate(contract.cell_ids)}
    arrays = {
        **_common_arrays(count),
        "carrier_mode": _initialize_int(count),
        "depth": _initialize_float(count),
        "phase": _initialize_float(count),
        "amplitude": _initialize_float(count),
        "steepness": _initialize_float(count),
        "depth_wavenumber": _initialize_float(count),
        "ursell_upper_bound": _initialize_float(count),
        "support_redraw_count": _initialize_int(count),
        "depth_draw_lower": _initialize_float(count),
        "depth_draw_upper": _initialize_float(count),
        "amplitude_draw_lower": _initialize_float(count),
        "amplitude_draw_upper": _initialize_float(count),
    }
    attempt_owner: list[int] = []
    attempt_amplitude_unit: list[float] = []
    attempt_ursell_ratio: list[float] = []
    attempt_accepted: list[bool] = []
    failures: list[JsonRecord] = []
    accepted_by_cell = {cell_id: 0 for cell_id in contract.cell_ids}
    accepted_count = 0
    first_attempt_index = 0
    support_violation_count = 0
    started = perf_counter()

    while accepted_count < count:
        assignments = _scheduled_batch(
            contract,
            quotas,
            accepted_by_cell,
            first_attempt_index=first_attempt_index,
            batch_size=batch_size,
        )
        if not assignments:
            raise RuntimeError("Stokes quota scheduler stopped before completion")
        first_attempt_index += len(assignments)
        for assignment in assignments:
            try:
                sample = sample_stokes_population(
                    assignment,
                    maximum_ursell_redraws=DEFAULT_MAXIMUM_URSELL_REDRAWS,
                )
            except StokesPopulationSamplingError as error:
                failures.append(dict(error.failure_record))
                continue
            violations = stokes_support_violations(sample)
            support_violation_count += len(violations)
            if violations:
                raise RuntimeError("; ".join(violations))
            row = accepted_count
            _store_common(arrays, row, assignment, cell_codes)
            arrays["carrier_mode"][row] = sample.carrier_mode
            arrays["depth"][row] = sample.depth
            arrays["phase"][row] = sample.phase
            arrays["amplitude"][row] = sample.amplitude
            arrays["steepness"][row] = sample.steepness
            arrays["depth_wavenumber"][row] = sample.wavenumber * sample.depth
            arrays["ursell_upper_bound"][row] = (
                np.nan
                if sample.ursell_upper_bound is None
                else sample.ursell_upper_bound
            )
            arrays["support_redraw_count"][row] = sample.support_resampling_count
            arrays["depth_draw_lower"][row] = sample.depth_draw_bounds[0]
            arrays["depth_draw_upper"][row] = sample.depth_draw_bounds[1]
            arrays["amplitude_draw_lower"][row] = sample.amplitude_draw_bounds[0]
            arrays["amplitude_draw_upper"][row] = sample.amplitude_draw_bounds[1]
            for attempt in sample.amplitude_attempts:
                attempt_owner.append(row)
                attempt_amplitude_unit.append(
                    (attempt.amplitude - sample.amplitude_draw_bounds[0])
                    / (
                        sample.amplitude_draw_bounds[1]
                        - sample.amplitude_draw_bounds[0]
                    )
                )
                attempt_ursell_ratio.append(
                    np.nan
                    if attempt.ursell_upper_bound is None
                    else attempt.ursell_upper_bound / 26.0
                )
                attempt_accepted.append(attempt.accepted)
            accepted_by_cell[assignment.cell_id] += 1
            accepted_count += 1

    arrays.update(
        {
            "attempt_owner": np.asarray(attempt_owner, dtype=np.int32),
            "attempt_amplitude_unit": np.asarray(
                attempt_amplitude_unit, dtype=np.float64
            ),
            "attempt_ursell_ratio": np.asarray(
                attempt_ursell_ratio, dtype=np.float64
            ),
            "attempt_accepted": np.asarray(attempt_accepted, dtype=np.bool_),
        }
    )
    phase_unit = arrays["phase"] / (2.0 * math.pi)
    log_depth_unit = (
        np.log(arrays["depth"]) - np.log(arrays["depth_draw_lower"])
    ) / (
        np.log(arrays["depth_draw_upper"])
        - np.log(arrays["depth_draw_lower"])
    )
    summary = _base_family_summary(
        contract,
        quotas,
        accepted_by_cell,
        accepted_count=accepted_count,
        attempted_count=first_attempt_index,
        support_violation_count=support_violation_count,
        timing_seconds=perf_counter() - started,
    )
    summary.update(
        {
            "declared_sampling_exhaustions": len(failures),
            "total_amplitude_draws": len(attempt_owner),
            "total_rejected_amplitude_draws": int(
                np.count_nonzero(~arrays["attempt_accepted"])
            ),
            "maximum_support_redraw_count": int(
                np.max(arrays["support_redraw_count"])
            ),
            "finite_depth_ursell_ratio": _finite_summary(
                arrays["ursell_upper_bound"] / 26.0
            ),
            "depth_wavenumber": _finite_summary(arrays["depth_wavenumber"]),
            "steepness": _finite_summary(arrays["steepness"]),
            "uniform_checks": {
                "phase": _uniform_check(phase_unit),
                "conditional_log_depth": _uniform_check(log_depth_unit),
                "all_amplitude_draws_before_support_decision": _uniform_check(
                    arrays["attempt_amplitude_unit"]
                ),
            },
        }
    )
    return GeneratedFamily(
        contract=contract,
        arrays=arrays,
        summary=summary,
        failures=tuple(failures),
    )


def generate_tanaka(
    count: int,
    *,
    batch_size: int,
) -> GeneratedFamily:
    """Generate exactly ``count`` Tanaka specifications."""

    contract = FAMILY_CONTRACTS["tanaka"]
    quotas = balanced_cell_quotas(contract.cell_ids, accepted_case_count=count)
    cell_codes = {cell_id: index for index, cell_id in enumerate(contract.cell_ids)}
    arrays = {
        **_common_arrays(count),
        "depth": _initialize_float(count),
        "crest_count": _initialize_int(count),
        "total_alpha": _initialize_float(count),
        "required_separation": _initialize_float(count),
        "achieved_separation": _initialize_float(count),
        "crest_alpha": _initialize_float(count, columns=3),
        "crest_center": _initialize_float(count, columns=3),
        "crest_direction": _initialize_int(count, columns=3),
        "cyclic_gap": _initialize_float(count, columns=3),
        "depth_unit": _initialize_float(count),
        "total_alpha_unit": _initialize_float(count),
    }
    accepted_by_cell = {cell_id: 0 for cell_id in contract.cell_ids}
    accepted_count = 0
    first_attempt_index = 0
    support_violation_count = 0
    started = perf_counter()

    while accepted_count < count:
        assignments = _scheduled_batch(
            contract,
            quotas,
            accepted_by_cell,
            first_attempt_index=first_attempt_index,
            batch_size=batch_size,
        )
        if not assignments:
            raise RuntimeError("Tanaka quota scheduler stopped before completion")
        first_attempt_index += len(assignments)
        for assignment in assignments:
            sample = sample_tanaka_population(assignment)
            violations = tanaka_support_violations(sample)
            support_violation_count += len(violations)
            if violations:
                raise RuntimeError("; ".join(violations))
            row = accepted_count
            _store_common(arrays, row, assignment, cell_codes)
            arrays["depth"][row] = sample.depth
            arrays["crest_count"][row] = sample.cell.crest_count
            arrays["total_alpha"][row] = sample.total_dimensionless_amplitude
            arrays["required_separation"][row] = sample.required_minimum_separation
            arrays["achieved_separation"][row] = sample.achieved_minimum_separation
            for index, (crest, gap) in enumerate(
                zip(sample.crests, sample.cyclic_gaps)
            ):
                arrays["crest_alpha"][row, index] = crest.alpha
                arrays["crest_center"][row, index] = crest.center
                arrays["crest_direction"][row, index] = crest.direction
                arrays["cyclic_gap"][row, index] = gap
            if sample.cell.regime == "main":
                depth_bounds = MAIN_DEPTH_BOUNDS
                alpha_bounds = MAIN_TOTAL_ALPHA_BOUNDS
            else:
                depth_bounds = STEEP_DEPTH_BOUNDS
                alpha_bounds = STEEP_ALPHA_BOUNDS
            arrays["depth_unit"][row] = (
                math.log(sample.depth) - math.log(depth_bounds[0])
            ) / (math.log(depth_bounds[1]) - math.log(depth_bounds[0]))
            arrays["total_alpha_unit"][row] = (
                sample.total_dimensionless_amplitude - alpha_bounds[0]
            ) / (alpha_bounds[1] - alpha_bounds[0])
            accepted_by_cell[assignment.cell_id] += 1
            accepted_count += 1

    finite_alphas = arrays["crest_alpha"][np.isfinite(arrays["crest_alpha"])]
    separation_excess = (
        arrays["achieved_separation"] - arrays["required_separation"]
    )
    summary = _base_family_summary(
        contract,
        quotas,
        accepted_by_cell,
        accepted_count=accepted_count,
        attempted_count=first_attempt_index,
        support_violation_count=support_violation_count,
        timing_seconds=perf_counter() - started,
    )
    summary.update(
        {
            "depth": _finite_summary(arrays["depth"]),
            "total_dimensionless_amplitude": _finite_summary(
                arrays["total_alpha"]
            ),
            "individual_crest_alpha": _finite_summary(finite_alphas),
            "separation_excess": _finite_summary(separation_excess),
            "individual_crest_fraction_below": {
                "0.001": float(np.mean(finite_alphas < 0.001)),
                "0.005": float(np.mean(finite_alphas < 0.005)),
                "0.01": float(np.mean(finite_alphas < 0.01)),
            },
            "right_moving_crest_fraction": float(
                np.mean(
                    arrays["crest_direction"][
                        np.isfinite(arrays["crest_alpha"])
                    ]
                    > 0
                )
            ),
            "uniform_checks": {
                "conditional_log_depth": _uniform_check(arrays["depth_unit"]),
                "conditional_total_alpha": _uniform_check(
                    arrays["total_alpha_unit"]
                ),
            },
        }
    )
    return GeneratedFamily(contract=contract, arrays=arrays, summary=summary)


def generate_benjamin_feir(
    count: int,
    *,
    batch_size: int,
) -> GeneratedFamily:
    """Generate exactly ``count`` Benjamin--Feir specifications."""

    contract = FAMILY_CONTRACTS["benjamin_feir"]
    quotas = balanced_cell_quotas(contract.cell_ids, accepted_case_count=count)
    cell_codes = {cell_id: index for index, cell_id in enumerate(contract.cell_ids)}
    arrays = {
        **_common_arrays(count),
        "carrier_mode": _initialize_int(count),
        "sideband_offset": _initialize_int(count),
        "carrier_steepness": _initialize_float(count),
        "conditional_steepness_lower": _initialize_float(count),
        "carrier_steepness_unit": _initialize_float(count),
        "perturbation_ratio": _initialize_float(count),
        "perturbation_ratio_unit": _initialize_float(count),
        "sideband_phase": _initialize_float(count),
        "band_fraction": _initialize_float(count),
        "carrier_amplitude": _initialize_float(count),
    }
    accepted_by_cell = {cell_id: 0 for cell_id in contract.cell_ids}
    accepted_count = 0
    first_attempt_index = 0
    support_violation_count = 0
    started = perf_counter()

    while accepted_count < count:
        assignments = _scheduled_batch(
            contract,
            quotas,
            accepted_by_cell,
            first_attempt_index=first_attempt_index,
            batch_size=batch_size,
        )
        if not assignments:
            raise RuntimeError(
                "Benjamin--Feir quota scheduler stopped before completion"
            )
        first_attempt_index += len(assignments)
        for assignment in assignments:
            sample = sample_benjamin_feir_population(assignment)
            violations = benjamin_feir_support_violations(sample)
            support_violation_count += len(violations)
            if violations:
                raise RuntimeError("; ".join(violations))
            row = accepted_count
            _store_common(arrays, row, assignment, cell_codes)
            lower = sample.cell.conditional_steepness_lower_bound
            arrays["carrier_mode"][row] = sample.cell.carrier_mode
            arrays["sideband_offset"][row] = sample.cell.sideband_offset
            arrays["carrier_steepness"][row] = sample.carrier_steepness
            arrays["conditional_steepness_lower"][row] = lower
            arrays["carrier_steepness_unit"][row] = (
                sample.carrier_steepness - lower
            ) / (0.13 - lower)
            arrays["perturbation_ratio"][row] = sample.perturbation_ratio
            arrays["perturbation_ratio_unit"][row] = (
                sample.perturbation_ratio - 0.05
            ) / 0.15
            arrays["sideband_phase"][row] = sample.sideband_phase
            arrays["band_fraction"][row] = sample.band_fraction
            arrays["carrier_amplitude"][row] = sample.carrier_amplitude
            accepted_by_cell[assignment.cell_id] += 1
            accepted_count += 1

    band_margin = 1.0 - arrays["band_fraction"]
    summary = _base_family_summary(
        contract,
        quotas,
        accepted_by_cell,
        accepted_count=accepted_count,
        attempted_count=first_attempt_index,
        support_violation_count=support_violation_count,
        timing_seconds=perf_counter() - started,
    )
    summary.update(
        {
            "carrier_steepness": _finite_summary(arrays["carrier_steepness"]),
            "perturbation_ratio": _finite_summary(arrays["perturbation_ratio"]),
            "instability_band_fraction": _finite_summary(
                arrays["band_fraction"]
            ),
            "instability_band_margin": _finite_summary(band_margin),
            "fraction_with_band_margin_below": {
                "0.001": float(np.mean(band_margin < 0.001)),
                "0.01": float(np.mean(band_margin < 0.01)),
                "0.05": float(np.mean(band_margin < 0.05)),
            },
            "uniform_checks": {
                "conditional_carrier_steepness": _uniform_check(
                    arrays["carrier_steepness_unit"]
                ),
                "perturbation_ratio": _uniform_check(
                    arrays["perturbation_ratio_unit"]
                ),
                "sideband_phase": _uniform_check(
                    arrays["sideband_phase"] / (2.0 * math.pi)
                ),
            },
        }
    )
    return GeneratedFamily(contract=contract, arrays=arrays, summary=summary)


def _jonswap_units(
    sample: JonswapTmaPopulationSample,
) -> tuple[float, float, float]:
    parameters = sample.parameters
    if sample.cell.stratum == "finite":
        depth_bounds = FINITE_DEPTH_BOUNDS
        peak_bounds = FINITE_PEAK_WAVENUMBER_BOUNDS
    elif sample.cell.stratum == "deep":
        depth_bounds = DEEP_DEPTH_BOUNDS
        peak_bounds = DEEP_PEAK_WAVENUMBER_BOUNDS
    else:
        return math.nan, math.nan, math.nan
    return (
        (parameters.depth - depth_bounds[0])
        / (depth_bounds[1] - depth_bounds[0]),
        (parameters.significant_height - SIGNIFICANT_HEIGHT_BOUNDS[0])
        / (SIGNIFICANT_HEIGHT_BOUNDS[1] - SIGNIFICANT_HEIGHT_BOUNDS[0]),
        (parameters.peak_wavenumber - peak_bounds[0])
        / (peak_bounds[1] - peak_bounds[0]),
    )


def generate_jonswap_tma(
    count: int,
    *,
    batch_size: int,
) -> GeneratedFamily:
    """Generate exactly ``count`` JONSWAP/TMA specifications."""

    contract = FAMILY_CONTRACTS["jonswap_tma"]
    quotas = balanced_cell_quotas(contract.cell_ids, accepted_case_count=count)
    cell_codes = {cell_id: index for index, cell_id in enumerate(contract.cell_ids)}
    arrays = {
        **_common_arrays(count),
        "stratum_code": _initialize_int(count),
        "depth": _initialize_float(count),
        "significant_height": _initialize_float(count),
        "peak_wavenumber": _initialize_float(count),
        "peak_enhancement": _initialize_float(count),
        "right_moving_fraction": _initialize_float(count),
        "peak_depth": _initialize_float(count),
        "relative_height": _initialize_float(count),
        "peak_steepness": _initialize_float(count),
        "phase_right_first": _initialize_float(count),
        "phase_left_first": _initialize_float(count),
        "phase_right_resultant": _initialize_float(count),
        "phase_left_resultant": _initialize_float(count),
        "rectangular_depth_unit": _initialize_float(count),
        "rectangular_height_unit": _initialize_float(count),
        "rectangular_peak_unit": _initialize_float(count),
    }
    phase_edges = np.linspace(
        0.0,
        2.0 * math.pi,
        PHASE_HISTOGRAM_BINS + 1,
        dtype=np.float64,
    )
    phase_hist_right = np.zeros(PHASE_HISTOGRAM_BINS, dtype=np.int64)
    phase_hist_left = np.zeros(PHASE_HISTOGRAM_BINS, dtype=np.int64)
    right_cos_sum = 0.0
    right_sin_sum = 0.0
    left_cos_sum = 0.0
    left_sin_sum = 0.0
    phase_value_count = 0
    right_hasher = hashlib.sha256()
    left_hasher = hashlib.sha256()
    accepted_by_cell = {cell_id: 0 for cell_id in contract.cell_ids}
    accepted_count = 0
    first_attempt_index = 0
    support_violation_count = 0
    started = perf_counter()
    stratum_codes = {"shallow": 0, "finite": 1, "deep": 2}

    while accepted_count < count:
        assignments = _scheduled_batch(
            contract,
            quotas,
            accepted_by_cell,
            first_attempt_index=first_attempt_index,
            batch_size=batch_size,
        )
        if not assignments:
            raise RuntimeError(
                "JONSWAP/TMA quota scheduler stopped before completion"
            )
        first_attempt_index += len(assignments)
        for assignment in assignments:
            sample = sample_jonswap_tma_population(assignment, band=PAPER_BAND)
            row = accepted_count
            _store_common(arrays, row, assignment, cell_codes)
            parameters = sample.parameters
            peak_depth = parameters.peak_wavenumber * parameters.depth
            relative_height = parameters.significant_height / (
                2.0 * parameters.depth
            )
            arrays["stratum_code"][row] = stratum_codes[sample.cell.stratum]
            arrays["depth"][row] = parameters.depth
            arrays["significant_height"][row] = parameters.significant_height
            arrays["peak_wavenumber"][row] = parameters.peak_wavenumber
            arrays["peak_enhancement"][row] = parameters.peak_enhancement
            arrays["right_moving_fraction"][row] = (
                parameters.right_moving_fraction
            )
            arrays["peak_depth"][row] = peak_depth
            arrays["relative_height"][row] = relative_height
            arrays["peak_steepness"][row] = peak_depth * relative_height
            arrays["phase_right_first"][row] = sample.phase_right[0]
            arrays["phase_left_first"][row] = sample.phase_left[0]
            right_mean = np.mean(np.exp(1j * sample.phase_right))
            left_mean = np.mean(np.exp(1j * sample.phase_left))
            arrays["phase_right_resultant"][row] = abs(right_mean)
            arrays["phase_left_resultant"][row] = abs(left_mean)
            depth_unit, height_unit, peak_unit = _jonswap_units(sample)
            arrays["rectangular_depth_unit"][row] = depth_unit
            arrays["rectangular_height_unit"][row] = height_unit
            arrays["rectangular_peak_unit"][row] = peak_unit
            phase_hist_right += np.histogram(
                sample.phase_right, bins=phase_edges
            )[0]
            phase_hist_left += np.histogram(
                sample.phase_left, bins=phase_edges
            )[0]
            right_cos_sum += float(np.sum(np.cos(sample.phase_right)))
            right_sin_sum += float(np.sum(np.sin(sample.phase_right)))
            left_cos_sum += float(np.sum(np.cos(sample.phase_left)))
            left_sin_sum += float(np.sum(np.sin(sample.phase_left)))
            phase_value_count += int(sample.phase_right.size)
            right_hasher.update(
                np.ascontiguousarray(sample.phase_right).view(np.uint8)
            )
            left_hasher.update(
                np.ascontiguousarray(sample.phase_left).view(np.uint8)
            )
            accepted_by_cell[assignment.cell_id] += 1
            accepted_count += 1

    arrays.update(
        {
            "phase_hist_edges": phase_edges,
            "phase_hist_counts_right": phase_hist_right,
            "phase_hist_counts_left": phase_hist_left,
        }
    )
    rectangular = arrays["stratum_code"] != 0
    shallow = ~rectangular
    summary = _base_family_summary(
        contract,
        quotas,
        accepted_by_cell,
        accepted_count=accepted_count,
        attempted_count=first_attempt_index,
        support_violation_count=support_violation_count,
        timing_seconds=perf_counter() - started,
    )
    summary.update(
        {
            "depth": _finite_summary(arrays["depth"]),
            "significant_height": _finite_summary(arrays["significant_height"]),
            "peak_wavenumber": _finite_summary(arrays["peak_wavenumber"]),
            "peak_depth": _finite_summary(arrays["peak_depth"]),
            "relative_height": _finite_summary(arrays["relative_height"]),
            "peak_steepness": _finite_summary(arrays["peak_steepness"]),
            "phase_values_per_direction": phase_value_count,
            "phase_sha256": {
                "right": right_hasher.hexdigest(),
                "left": left_hasher.hexdigest(),
                "definition": (
                    "SHA-256 over consecutive C-order float64 phase arrays "
                    "in accepted-case order"
                ),
            },
            "global_phase_resultant": {
                "right": math.hypot(right_cos_sum, right_sin_sum)
                / phase_value_count,
                "left": math.hypot(left_cos_sum, left_sin_sum)
                / phase_value_count,
            },
            "shallow_support": {
                "sample_count": int(np.count_nonzero(shallow)),
                "minimum_peak_depth": float(np.min(arrays["peak_depth"][shallow])),
                "maximum_peak_depth": float(np.max(arrays["peak_depth"][shallow])),
                "minimum_relative_height": float(
                    np.min(arrays["relative_height"][shallow])
                ),
                "maximum_relative_height": float(
                    np.max(arrays["relative_height"][shallow])
                ),
                "maximum_peak_steepness": float(
                    np.max(arrays["peak_steepness"][shallow])
                ),
                "declared_peak_depth_bounds": list(
                    SHALLOW_DEPTH_WAVENUMBER_BOUNDS
                ),
                "declared_relative_height_bounds": list(
                    SHALLOW_RELATIVE_HEIGHT_BOUNDS
                ),
            },
            "uniform_checks": {
                "first_right_phase": _uniform_check(
                    arrays["phase_right_first"] / (2.0 * math.pi)
                ),
                "first_left_phase": _uniform_check(
                    arrays["phase_left_first"] / (2.0 * math.pi)
                ),
                "rectangular_depth": _uniform_check(
                    arrays["rectangular_depth_unit"][rectangular]
                ),
                "rectangular_significant_height": _uniform_check(
                    arrays["rectangular_height_unit"][rectangular]
                ),
                "rectangular_peak_wavenumber": _uniform_check(
                    arrays["rectangular_peak_unit"][rectangular]
                ),
            },
        }
    )
    return GeneratedFamily(contract=contract, arrays=arrays, summary=summary)


def _sample_from_assignment(
    family: str,
    assignment: AttemptAssignment,
) -> (
    StokesPopulationSample
    | TanakaPopulationSample
    | BenjaminFeirPopulationSample
    | JonswapTmaPopulationSample
):
    if family == "stokes":
        return sample_stokes_population(assignment)
    if family == "tanaka":
        return sample_tanaka_population(assignment)
    if family == "benjamin_feir":
        return sample_benjamin_feir_population(assignment)
    if family == "jonswap_tma":
        return sample_jonswap_tma_population(assignment, band=PAPER_BAND)
    raise ValueError(f"unknown family {family!r}")


def _validate_sentinel(
    family: str,
    sample: object,
    arrays: Mapping[str, NDArray[Any]],
    row: int,
) -> None:
    if family == "stokes":
        assert isinstance(sample, StokesPopulationSample)
        observed = (
            sample.carrier_mode,
            sample.depth,
            sample.phase,
            sample.amplitude,
        )
        expected = (
            int(arrays["carrier_mode"][row]),
            float(arrays["depth"][row]),
            float(arrays["phase"][row]),
            float(arrays["amplitude"][row]),
        )
    elif family == "tanaka":
        assert isinstance(sample, TanakaPopulationSample)
        observed = (
            sample.depth,
            sample.total_dimensionless_amplitude,
            tuple(crest.center for crest in sample.crests),
        )
        count = sample.cell.crest_count
        expected = (
            float(arrays["depth"][row]),
            float(arrays["total_alpha"][row]),
            tuple(float(value) for value in arrays["crest_center"][row, :count]),
        )
    elif family == "benjamin_feir":
        assert isinstance(sample, BenjaminFeirPopulationSample)
        observed = (
            sample.cell.carrier_mode,
            sample.cell.sideband_offset,
            sample.carrier_steepness,
            sample.perturbation_ratio,
            sample.sideband_phase,
        )
        expected = (
            int(arrays["carrier_mode"][row]),
            int(arrays["sideband_offset"][row]),
            float(arrays["carrier_steepness"][row]),
            float(arrays["perturbation_ratio"][row]),
            float(arrays["sideband_phase"][row]),
        )
    elif family == "jonswap_tma":
        assert isinstance(sample, JonswapTmaPopulationSample)
        observed = (
            sample.parameters.depth,
            sample.parameters.significant_height,
            sample.parameters.peak_wavenumber,
            sample.phase_right[0],
            sample.phase_left[0],
        )
        expected = (
            float(arrays["depth"][row]),
            float(arrays["significant_height"][row]),
            float(arrays["peak_wavenumber"][row]),
            float(arrays["phase_right_first"][row]),
            float(arrays["phase_left_first"][row]),
        )
    else:
        raise ValueError(f"unknown family {family!r}")
    if observed != expected:
        raise RuntimeError(f"{family} sentinel replay differs at archive row {row}")


def build_sentinel_records(
    generated: Sequence[GeneratedFamily],
) -> JsonRecord:
    """Replay first, middle, and last accepted samples in every cell."""

    families: dict[str, object] = {}
    total = 0
    for result in generated:
        family = result.contract.name
        records: list[JsonRecord] = []
        cell_codes = np.asarray(result.arrays["cell_code"], dtype=np.int64)
        for cell_code, cell_id in enumerate(result.contract.cell_ids):
            rows = np.flatnonzero(cell_codes == cell_code)
            if not rows.size:
                continue
            positions = (0, rows.size // 2, rows.size - 1)
            seen: set[int] = set()
            for role, position in zip(SENTINEL_ROLES, positions):
                row = int(rows[position])
                if row in seen:
                    continue
                seen.add(row)
                attempt_index = int(result.arrays["attempt_index"][row])
                assignment = AttemptAssignment(
                    case_key=CaseKey(
                        family_id=int(result.contract.family_id),
                        revision_id=PAPER_CORPUS_REVISION_ID,
                        split_id=SplitId.TEST,
                        stream_id=SMOKE_STREAM_ID,
                        attempt_index=attempt_index,
                    ),
                    cell_id=cell_id,
                )
                sample = _sample_from_assignment(family, assignment)
                _validate_sentinel(family, sample, result.arrays, row)
                record_method = getattr(sample, "to_json_record")
                record = record_method()
                records.append(
                    {
                        "selection_role": role,
                        "archive_row": row,
                        "cell_id": cell_id,
                        "attempt_index": attempt_index,
                        "record_sha256": hashlib.sha256(
                            _strict_json_bytes(record)
                        ).hexdigest(),
                        "record": record,
                    }
                )
                total += 1
        families[family] = records
    return {
        "schema": "paper_corpus_population_smoke_sentinels_v1",
        "selection": (
            "first, middle, and last accepted archive row in each nonempty "
            "ordered allocation cell"
        ),
        "replay_verified": True,
        "sentinel_count": total,
        "families": families,
    }


def _archive_record(path: Path, *, root: Path) -> JsonRecord:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _write_npz(path: Path, arrays: Mapping[str, NDArray[Any]]) -> None:
    np.savez(path, **arrays)


def run_smoke(
    *,
    total_samples: int,
    batch_size: int,
    output_dir: Path,
) -> JsonRecord:
    """Run the complete population smoke and atomically publish its directory."""

    configuration = _configuration(
        total_samples=total_samples,
        batch_size=batch_size,
    )
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing output: {output_dir}")
    staging = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    started_at = datetime.now().astimezone()
    started = perf_counter()
    count_per_family = total_samples // len(FAMILY_ORDER)

    generators: tuple[Callable[..., GeneratedFamily], ...] = (
        generate_stokes,
        generate_tanaka,
        generate_benjamin_feir,
        generate_jonswap_tma,
    )
    generated: list[GeneratedFamily] = []
    artifacts: dict[str, object] = {}
    failure_records: dict[str, object] = {}
    for family, generator in zip(FAMILY_ORDER, generators):
        result = generator(count_per_family, batch_size=batch_size)
        generated.append(result)
        archive_path = staging / f"{family}_samples.npz"
        _write_npz(archive_path, result.arrays)
        artifacts[family] = _archive_record(archive_path, root=staging)
        if result.failures:
            failure_records[family] = list(result.failures)

    sentinel_path = staging / "sentinel_records.json"
    sentinel_records = build_sentinel_records(generated)
    write_json_atomic(sentinel_path, sentinel_records)
    failure_path: Path | None = None
    if failure_records:
        failure_path = staging / "sampling_failures.json"
        write_json_atomic(
            failure_path,
            {
                "schema": "paper_corpus_population_smoke_failures_v1",
                "families": failure_records,
            },
        )

    devices = jax.devices()
    summary: JsonRecord = {
        "schema": "paper_corpus_population_smoke_summary_v1",
        "status": "complete",
        "configuration": configuration,
        "configuration_fingerprint": _fingerprint(configuration),
        "counts": {
            "accepted_specifications": sum(
                int(result.summary["accepted_count"]) for result in generated
            ),
            "attempted_specifications": sum(
                int(result.summary["attempted_count"]) for result in generated
            ),
            "families": len(generated),
            "accepted_per_family": count_per_family,
        },
        "families": {result.contract.name: result.summary for result in generated},
        "artifacts": {
            **artifacts,
            "sentinel_records": _archive_record(sentinel_path, root=staging),
            **(
                {"sampling_failures": _archive_record(failure_path, root=staging)}
                if failure_path is not None
                else {}
            ),
        },
        "sentinel_validation": {
            "count": sentinel_records["sentinel_count"],
            "replay_verified": sentinel_records["replay_verified"],
        },
        "runtime": {
            "cpu_only_verified": bool(
                devices
                and jax.default_backend() == "cpu"
                and all(device.platform == "cpu" for device in devices)
            ),
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.x64_enabled),
            "jax_devices": [
                {
                    "id": int(device.id),
                    "platform": device.platform,
                    "kind": device.device_kind,
                }
                for device in devices
            ],
        },
        "timing_seconds": {
            "family_sampling": {
                result.contract.name: result.summary["timing_seconds"]
                for result in generated
            },
            "total_including_archives_and_replay": perf_counter() - started,
        },
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "limitations": [
            "All 500,000 parameter specifications were sampled and checked.",
            "Only deterministic sentinels are field-constructed by the separate plotting step.",
            "This smoke does not evaluate DNO targets or integrate trajectories.",
            "It therefore does not estimate full numerical acceptance rates.",
        ],
    }
    if not summary["runtime"]["cpu_only_verified"]:  # type: ignore[index]
        raise RuntimeError("population smoke did not run on CPU only")
    if summary["counts"]["accepted_specifications"] != total_samples:  # type: ignore[index]
        raise RuntimeError("population smoke did not generate the requested count")
    summary_path = staging / "summary.json"
    write_json_atomic(summary_path, summary)

    # Recheck every artifact before publication.
    checked = json.loads(summary_path.read_text(encoding="utf-8"))
    for artifact in checked["artifacts"].values():
        artifact_path = staging / artifact["path"]
        if file_sha256(artifact_path) != artifact["sha256"]:
            raise RuntimeError(f"artifact hash changed before publication: {artifact_path}")

    staging.replace(output_dir)
    return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    summary = run_smoke(
        total_samples=args.total_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    summary_path = args.output_dir.expanduser().resolve() / "summary.json"
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary_path": str(summary_path),
                "summary_sha256": file_sha256(summary_path),
                "counts": summary["counts"],
                "timing_seconds": summary["timing_seconds"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
