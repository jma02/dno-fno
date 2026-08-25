"""Fail-closed CPU audit of the completed Benjamin--Feir revision-4 dataset.

The quota scanner remains the authority for transaction replay.  This audit
adds dataset-wide interval and identity checks, exact replay of every proposed
Benjamin--Feir parameter specification, numerical-health extrema, stored-field
finiteness, and an independent check of each loader-facing trajectory map.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-bf-completion-audit")

import numpy as np  # noqa: E402

from scripts.build_paper_dataset_view import (  # noqa: E402
    CompletedChunk,
    TRAJECTORY_MAP_DTYPES,
    load_completed_chunk,
)
from scripts.generate_paper_dataset import (  # noqa: E402
    _benjamin_feir_sampling_support_record,
    dependency_environment,
)
from solver.gen_data.benjamin_feir_sampling import (  # noqa: E402
    BENJAMIN_FEIR_SAMPLE_CELL_IDS,
    PAPER_FOCUSED_STEEPNESS_LIMIT,
    find_benjamin_feir_sample_violations,
    sample_benjamin_feir_case,
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
    balanced_valid_case_targets,
)
from solver.gen_data.pipeline.quality import (  # noqa: E402
    QualityDecision,
    QualityReason,
    QualityScope,
    reasons_from_bits,
)
from solver.gen_data.pipeline.valid_case_generation import (  # noqa: E402
    canonical_json_sha256,
)
from solver.gen_data.pipeline.time_selection import (  # noqa: E402
    select_uniform_times,
)
from solver.gen_data.trajectory_batch_executor import (  # noqa: E402
    TrajectoryExecutionConfig,
)
from solver.gen_data.trajectory_family_adapters import (  # noqa: E402
    sample_benjamin_feir_trajectory_cases,
)


DEFAULT_DATASET_ROOT = (
    ROOT / "outputs/paper_dataset_bf_revision4_jonswap_revision3_literature_aligned_v1"
)
AUDIT_SCHEMA = "paper_dataset_benjamin_feir_revision4_completion_audit_v1"
ROWS_PER_ACCEPTED_CASE = 200
EXPECTED_FAMILY_ID = int(PhysicalFamilyId.BENJAMIN_FEIR)
EXPECTED_REVISION_ID = 4
EXPECTED_ACCEPTED = 18_432
EXPECTED_TRAIN_ACCEPTED = 16_384
EXPECTED_VALIDATION_ACCEPTED = 1_024
EXPECTED_TEST_ACCEPTED = 1_024
EXPECTED_BATCH_SIZE = 256
EXPECTED_MAXIMUM_ATTEMPTS_PER_ACCEPTED_CASE = 4
CURRENT_BF_EXECUTION = TrajectoryExecutionConfig.paper("benjamin_feir")
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
EXPECTED_PROPOSAL_ARRAYS = frozenset(
    {
        "attempt_index",
        "batch_id",
        "case_id",
        "case_spec_json",
        "cell_id",
        "config_fingerprint",
        "family_id",
        "metadata_json",
        "revision_id",
        "root_seed",
        "split_id",
        "stream_id",
    }
)
EXPECTED_SHARD_ARRAYS = frozenset(
    {
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
)
EXPECTED_MAP_ARRAYS = frozenset(TRAJECTORY_MAP_DTYPES)
CURRENT_SUPPORT_SOURCE_PATHS = (
    "solver/reference_solutions/stokes_wave.py",
    "solver/gen_data/benjamin_feir_jcp09.py",
    "solver/gen_data/benjamin_feir_sampling.py",
)
HISTORICAL_SHARED_SOURCE_SNAPSHOT_ROOT = (
    ROOT / "reproducibility/source_snapshots/benjamin_feir_revision4_e16773f"
)
HISTORICAL_SHARED_SOURCE_SNAPSHOTS = {
    "solver/gen_data/pipeline/production.py": (
        "production.py",
        "2c2234caf1087c2982872eb44ba2abd457cf22b812784e1a3e9bd54e8c7129d4",
    ),
    "solver/gen_data/trajectory_family_adapters.py": (
        "trajectory_family_adapters.py",
        "afb480a64b14a2b311bda067e6638208569cf3acfebc61d67a1f016a73fbef6a",
    ),
}
EXPECTED_GENERATION_SOURCE_PATHS = frozenset(
    {
        "scripts/run_paper_dataset_quota.py",
        "solver/reference_solutions/stokes_wave.py",
        "solver/gen_data/benjamin_feir_jcp09.py",
        "solver/gen_data/benjamin_feir_sampling.py",
        "solver/gen_data/pipeline/acceptance.py",
        "solver/gen_data/pipeline/archive.py",
        "solver/gen_data/pipeline/manifest.py",
        "solver/gen_data/pipeline/production.py",
        "solver/gen_data/pipeline/quality.py",
        "solver/gen_data/pipeline/quota_driver.py",
        "solver/gen_data/pipeline/reference.py",
        "solver/gen_data/pipeline/refinement.py",
        "solver/gen_data/pipeline/time_selection.py",
        "solver/gen_data/pipeline/trajectory_writer.py",
        "solver/gen_data/pipeline/writer.py",
        "solver/gen_data/trajectory_family_adapters.py",
        "solver/gen_data/trajectory_quota_executor.py",
        "solver/solvers/dno_series_jax.py",
        "solver/solvers/time_integrator.py",
    }
)
CURRENT_SOURCE_PATH_BY_GENERATION_PATH = {
    "scripts/run_paper_dataset_quota.py": "scripts/generate_paper_dataset.py",
    "solver/gen_data/pipeline/quota_driver.py": (
        "solver/gen_data/pipeline/valid_case_generation.py"
    ),
    "solver/gen_data/trajectory_quota_executor.py": (
        "solver/gen_data/trajectory_batch_executor.py"
    ),
}
EXPECTED_METRIC_KEYS = frozenset(
    {
        "accepted",
        "all_stages_solved",
        "complete_admissible_trajectory",
        "initial_internal_hamiltonian",
        "intended_terminal_time",
        "internal_dno_finite",
        "internal_hamiltonian_drift_threshold",
        "internal_health_evaluated",
        "internal_state_finite",
        "maximum_internal_hamiltonian_drift",
        "maximum_stage_residual",
        "minimum_internal_water_column",
        "positive_water_column",
        "production_dt",
        "realized_terminal_time",
        "saved_time_count",
        "state_finite",
        "target_finite",
    }
)
EXPECTED_REQUIRED_BITS = int(
    QualityReason.NONFINITE_STATE
    | QualityReason.NONFINITE_TARGET
    | QualityReason.BOTTOM_CLEARANCE
    | QualityReason.HAMILTONIAN_DRIFT
    | QualityReason.OUTSIDE_SUPPORT
    | QualityReason.INCOMPLETE_TRAJECTORY
)
EXPECTED_EVALUATED_BITS = int(
    QualityReason(EXPECTED_REQUIRED_BITS) | QualityReason.GL2_STAGE_RESIDUAL
)


@dataclass(frozen=True)
class ExpectedChunk:
    """One exact additive interval in the completed BF population."""

    label: str
    relative_root: Path
    split: SplitId
    stream_id: int
    accepted_before: int
    accepted_count: int

    @property
    def summary_path(self) -> Path:
        return self.relative_root / (
            f"paper_dataset_benjamin_feir_{self.split.value}.summary.json"
        )


EXPECTED_CHUNKS = (
    ExpectedChunk(
        "train_00000_02048",
        Path("train/benjamin_feir/chunk_00000_02048"),
        SplitId.TRAIN,
        0,
        0,
        2_048,
    ),
    ExpectedChunk(
        "train_02048_02048",
        Path("train/benjamin_feir/chunk_02048_02048"),
        SplitId.TRAIN,
        1,
        2_048,
        2_048,
    ),
    ExpectedChunk(
        "train_04096_04096",
        Path("train/benjamin_feir/chunk_04096_04096"),
        SplitId.TRAIN,
        2,
        4_096,
        4_096,
    ),
    ExpectedChunk(
        "train_08192_08192",
        Path("train/benjamin_feir/chunk_08192_08192"),
        SplitId.TRAIN,
        3,
        8_192,
        8_192,
    ),
    ExpectedChunk(
        "validation_00000_01024",
        Path("validation/benjamin_feir/chunk_00000_01024"),
        SplitId.VALIDATION,
        100,
        0,
        1_024,
    ),
    ExpectedChunk(
        "test_00000_01024",
        Path("test/benjamin_feir/chunk_00000_01024"),
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
class ProposedCase:
    """One BF proposal replayed from its immutable run coordinates."""

    case_id: int
    attempt_index: int
    cell_code: int
    cell_id: str
    depth: float
    carrier_wavenumber: float
    intended_terminal_time: float
    realized_terminal_time: float
    saved_time_count: int


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
    float_arrays_checked: int = 0
    float_values_checked: int = 0
    transaction_artifact_bytes_checked: int = 0
    dataset_view_bytes_checked: int = 0


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


def _strict_json_text(text: str, *, context: str) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{context} contains nonfinite JSON constant {value!r}")

    value = json.loads(text, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"{context} must encode a JSON object")
    return value


def _split_code(split: SplitId) -> int:
    return {
        SplitId.TRAIN: 0,
        SplitId.VALIDATION: 1,
        SplitId.TEST: 2,
    }[split]


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


def _same_json(left: object, right: object) -> bool:
    return canonical_json_sha256(left) == canonical_json_sha256(right)


def _floored_saved_count(intended: float, saved_dt: float) -> int:
    """Mirror the strictly-floored saved-grid construction without metrics."""

    step_count = math.floor(intended / saved_dt)
    while step_count * saved_dt > intended:
        step_count -= 1
    while (step_count + 1) * saved_dt <= intended:
        step_count += 1
    return step_count + 1


def _bf_horizon(carrier_wavenumber: float, saved_dt: float) -> tuple[float, float, int]:
    """Derive the 100-carrier-period BF horizon from physical parameters."""

    period_count = CURRENT_BF_EXECUTION.horizon.period_count
    gravity = CURRENT_BF_EXECUTION.numerical.gravity
    if period_count != 100:
        raise RuntimeError("current BF paper horizon is not 100 carrier periods")
    if (
        not math.isfinite(carrier_wavenumber)
        or carrier_wavenumber <= 0.0
        or not math.isfinite(saved_dt)
        or saved_dt <= 0.0
    ):
        raise ValueError("BF horizon inputs must be finite and positive")
    angular_frequency = math.sqrt(gravity * carrier_wavenumber)
    intended = period_count * 2.0 * math.pi / angular_frequency
    saved_count = _floored_saved_count(intended, saved_dt)
    realized = (saved_count - 1) * saved_dt
    if not (realized <= intended and intended - realized < saved_dt):
        raise RuntimeError("BF saved-grid horizon is not strictly floored")
    return intended, realized, saved_count


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
    batch_id: int,
    configuration_fingerprint: str,
    saved_dt: float,
    cell_ids_by_code: Mapping[int, str],
    support_extrema: Mapping[str, Extrema],
    totals: AuditTotals,
) -> tuple[ProposedCase, ...]:
    """Replay every proposal record through the current BF sampler."""

    with np.load(proposal_path, allow_pickle=False) as archive:
        if frozenset(archive.files) != EXPECTED_PROPOSAL_ARRAYS:
            raise ValueError(f"{proposal_path} has an unknown proposal schema")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    vector_dtypes = {
        "case_id": np.dtype(np.int64),
        "root_seed": np.dtype(np.uint64),
        "stream_id": np.dtype(np.uint32),
        "attempt_index": np.dtype(np.uint64),
        "cell_id": np.dtype(np.int32),
    }
    for name, dtype in vector_dtypes.items():
        if arrays[name].dtype != dtype:
            raise TypeError(f"BF proposal {name} has the wrong dtype")
    scalar_dtypes = {
        "batch_id": np.dtype(np.int64),
        "family_id": np.dtype(np.int16),
        "revision_id": np.dtype(np.int16),
        "split_id": np.dtype(np.uint8),
    }
    for name, dtype in scalar_dtypes.items():
        if arrays[name].ndim != 0 or arrays[name].dtype != dtype:
            raise TypeError(f"BF proposal {name} has the wrong scalar dtype")
    for name in ("config_fingerprint", "metadata_json"):
        if arrays[name].ndim != 0 or arrays[name].dtype.kind not in {"U", "S"}:
            raise TypeError(f"BF proposal {name} must be a scalar string")
    case_ids = arrays["case_id"]
    roots = arrays["root_seed"]
    streams = arrays["stream_id"]
    attempts = arrays["attempt_index"]
    encoded_cells = arrays["cell_id"]
    specifications = arrays["case_spec_json"]
    count = int(case_ids.size)
    if any(
        array.shape != (count,)
        for array in (roots, streams, attempts, encoded_cells, specifications)
    ):
        raise ValueError(f"{proposal_path} has inconsistent proposal vectors")
    if specifications.dtype.kind not in {"U", "S"}:
        raise TypeError("BF proposal case_spec_json must be a string vector")
    if (
        int(arrays["batch_id"].item()) != batch_id
        or int(arrays["family_id"].item()) != EXPECTED_FAMILY_ID
        or int(arrays["revision_id"].item()) != EXPECTED_REVISION_ID
        or int(arrays["split_id"].item()) != _split_code(split)
        or str(arrays["config_fingerprint"].item()) != configuration_fingerprint
    ):
        raise ValueError(f"{proposal_path} has the wrong batch/run coordinates")
    _strict_json_text(
        str(arrays["metadata_json"].item()),
        context=f"{proposal_path} metadata_json",
    )

    replayed: list[ProposedCase] = []
    for index, encoded in enumerate(encoded_cells):
        try:
            cell_id = cell_ids_by_code[int(encoded)]
        except KeyError as error:
            raise ValueError(
                f"{proposal_path} contains unknown BF cell code {int(encoded)}"
            ) from error
        if int(streams[index]) != stream_id:
            raise ValueError(f"{proposal_path} has a case in the wrong stream")
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
        expected_sample = sample_benjamin_feir_case(assignment)
        violations = find_benjamin_feir_sample_violations(expected_sample)
        if violations:
            raise ValueError(
                f"current BF sampler produced unsupported case {key.case_id}: "
                + "; ".join(violations)
            )
        sampled = sample_benjamin_feir_trajectory_cases(
            (assignment,),
            contract=CURRENT_BF_EXECUTION.numerical,
        )
        if sampled.samples != (expected_sample,):
            raise RuntimeError("current BF trajectory adapter changed its sample")
        expected = sampled.specification_records[0]
        observed = _strict_json_text(
            str(specifications[index]),
            context=f"{proposal_path} case_spec_json[{index}]",
        )
        if not _same_json(observed, expected):
            raise ValueError(
                f"{proposal_path} case {key.case_id} differs from exact "
                "current BF deterministic sampling"
            )
        focused = float(expected["focused_steepness"])
        if focused > PAPER_FOCUSED_STEEPNESS_LIMIT:
            raise ValueError("BF focused-steepness support limit was exceeded")
        for name, field in (
            ("carrier_steepness", "first_harmonic_carrier_steepness"),
            ("sideband_ratio", "first_harmonic_sideband_ratio"),
            ("translation", "translation"),
            ("instability_band_fraction", "instability_band_fraction"),
            ("focused_steepness", "focused_steepness"),
        ):
            support_extrema[name].add(float(expected[field]))
        support_extrema["carrier_mode"].add(float(expected["carrier_mode"]))
        support_extrema["sideband_offset"].add(float(expected["sideband_offset"]))
        intended, realized, saved_count = _bf_horizon(
            expected_sample.carrier_wavenumber,
            saved_dt,
        )
        replayed.append(
            ProposedCase(
                case_id=key.case_id,
                attempt_index=int(attempts[index]),
                cell_code=int(encoded),
                cell_id=cell_id,
                depth=expected_sample.depth,
                carrier_wavenumber=expected_sample.carrier_wavenumber,
                intended_terminal_time=intended,
                realized_terminal_time=realized,
                saved_time_count=saved_count,
            )
        )
        totals.proposal_specs_checked += 1
    return tuple(replayed)


def _validate_result_metadata(
    result: Mapping[str, object],
    *,
    run_spec: Mapping[str, object],
    execution: Mapping[str, object],
    context: str,
) -> None:
    if result.get("schema") != "paper_dataset_batch_result_v1":
        raise ValueError(f"{context} has an unknown result schema")
    metadata = _required_mapping(result, "metadata", context=context)
    if metadata.get("family") != "benjamin_feir":
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


def _validate_case_metrics(
    case: Mapping[str, object],
    *,
    proposed: ProposedCase,
    accepted: bool,
    residual_tolerance: float,
    hamiltonian_threshold: float,
    production_dt: float,
    accepted_extrema: Mapping[str, Extrema],
    attempted_extrema: Mapping[str, Extrema],
) -> None:
    metrics = _required_mapping(case, "metrics", context="result case")
    if set(metrics) != EXPECTED_METRIC_KEYS:
        raise ValueError("result case metrics do not have the exact BF schema")
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
    saved_count = metrics.get("saved_time_count")
    if isinstance(saved_count, bool) or not isinstance(saved_count, int):
        raise TypeError("metrics.saved_time_count must be an integer")
    intended = _finite_number(
        metrics.get("intended_terminal_time"),
        context="metrics.intended_terminal_time",
    )
    realized = _finite_number(
        metrics.get("realized_terminal_time"),
        context="metrics.realized_terminal_time",
    )
    if not math.isclose(
        intended,
        proposed.intended_terminal_time,
        rel_tol=2.0e-14,
        abs_tol=1.0e-13,
    ):
        raise ValueError("BF intended horizon is not 100 carrier periods")
    if saved_count != proposed.saved_time_count:
        raise ValueError("BF dense saved-time count is not the floored horizon")
    if not math.isclose(
        realized,
        proposed.realized_terminal_time,
        rel_tol=2.0e-14,
        abs_tol=1.0e-13,
    ):
        raise ValueError("BF realized horizon differs from the saved-grid floor")
    saved_dt = CURRENT_BF_EXECUTION.numerical.saved_dt
    if not (realized <= intended and intended - realized < saved_dt):
        raise ValueError("BF horizon is not strictly floored on the saved grid")
    if saved_count < ROWS_PER_ACCEPTED_CASE:
        raise ValueError("BF trajectory has too few dense saved times")

    numeric_metrics = (
        "initial_internal_hamiltonian",
        "maximum_internal_hamiltonian_drift",
        "maximum_stage_residual",
        "minimum_internal_water_column",
    )
    for name in numeric_metrics:
        value = metrics.get(name)
        if value is not None:
            _finite_number(value, context=f"metrics.{name}")

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
            raise ValueError("accepted BF case has a failed health metric")
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
            raise ValueError("accepted BF case exceeds the GL2 residual tolerance")
        if not 0.0 <= drift <= hamiltonian_threshold:
            raise ValueError("accepted BF case exceeds the Hamiltonian threshold")
        if water <= 0.0:
            raise ValueError("accepted BF case has nonpositive water depth")
        for name, value in (
            ("initial_hamiltonian", initial_h),
            ("hamiltonian_drift", drift),
            ("stage_residual", residual),
            ("minimum_water_column", water),
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


def _validate_case_record(
    value: object,
    *,
    proposed: ProposedCase,
    batch_id: int,
    residual_tolerance: float,
    hamiltonian_threshold: float,
    production_dt: float,
    accepted_extrema: Mapping[str, Extrema],
    attempted_extrema: Mapping[str, Extrema],
    rejection_reasons: Counter[str],
) -> AuditedCase:
    if not isinstance(value, Mapping):
        raise TypeError("every result case must be a JSON object")
    accepted = value.get("accepted")
    if not isinstance(accepted, bool):
        raise TypeError("result case accepted must be Boolean")
    case_id = _required_integer(value, "case_id", context="result case")
    if case_id != proposed.case_id:
        raise ValueError("BF result case differs from its proposal identity")
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
    if required != EXPECTED_REQUIRED_BITS or evaluated != EXPECTED_EVALUATED_BITS:
        raise ValueError("result case does not use the exact revision-4 masks")
    if bool(failed & int(QualityReason.OUTSIDE_SUPPORT)):
        raise ValueError("a proposed BF case was outside current support")
    if decision.accepted is not accepted:
        raise ValueError("result accepted flag disagrees with quality masks")
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
        raise ValueError("accepted BF case does not own exactly 200 rows")
    if not accepted and (first_row != -1 or row_count != 0):
        raise ValueError("rejected BF case owns stored rows")
    _validate_case_metrics(
        value,
        proposed=proposed,
        accepted=accepted,
        residual_tolerance=residual_tolerance,
        hamiltonian_threshold=hamiltonian_threshold,
        production_dt=production_dt,
        accepted_extrema=accepted_extrema,
        attempted_extrema=attempted_extrema,
    )
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


def _validate_map_array_schema(arrays: Mapping[str, np.ndarray]) -> None:
    """Require the exact schema-v2 arrays and canonical loader dtypes."""

    if frozenset(arrays) != EXPECTED_MAP_ARRAYS:
        raise ValueError("trajectory map has an unknown array schema")
    schema = arrays["schema_version"]
    if (
        schema.ndim != 0
        or schema.dtype != TRAJECTORY_MAP_DTYPES["schema_version"]
        or int(schema.item()) != 2
    ):
        raise ValueError("trajectory map schema version is not canonical")
    for name, dtype in TRAJECTORY_MAP_DTYPES.items():
        if name != "schema_version" and arrays[name].dtype != dtype:
            raise TypeError(f"trajectory map {name} has the wrong dtype")


def _validate_shard(
    path: Path,
    *,
    proposal_sha256: str,
    configuration_fingerprint: str,
    proposed: Sequence[ProposedCase],
    cases: Sequence[AuditedCase],
    delivered_nx: int,
    saved_dt: float,
    totals: AuditTotals,
) -> int:
    """Bind every BF stored row to its proposal and physical time grid."""

    accepted_indices = tuple(index for index, case in enumerate(cases) if case.accepted)
    if not accepted_indices:
        if path.exists():
            raise ValueError("all-rejected BF batch unexpectedly has a shard")
        return 0
    if not path.is_file():
        raise FileNotFoundError(f"accepted BF batch has no shard: {path}")
    _scan_finite_archive(path, totals=totals)
    with np.load(path, allow_pickle=False) as archive:
        if frozenset(archive.files) != EXPECTED_SHARD_ARRAYS:
            raise ValueError(f"{path} has an unknown shard schema")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    for name, dtype in EXPECTED_SHARD_DTYPES.items():
        if arrays[name].dtype != dtype:
            raise TypeError(f"BF shard {name} has the wrong dtype")
    for name in ("config_fingerprint", "proposal_sha256"):
        if arrays[name].ndim != 0 or arrays[name].dtype.kind not in {"U", "S"}:
            raise TypeError(f"BF shard {name} must be a scalar string")
    if str(arrays["config_fingerprint"].item()) != configuration_fingerprint:
        raise ValueError("BF shard has the wrong configuration fingerprint")
    if str(arrays["proposal_sha256"].item()) != proposal_sha256:
        raise ValueError("BF shard has the wrong proposal hash")

    eta = arrays["eta"]
    expected_shape = (len(accepted_indices) * ROWS_PER_ACCEPTED_CASE, delivered_nx)
    if eta.shape != expected_shape:
        raise ValueError("BF shard has the wrong field shape")
    if arrays["xi"].shape != eta.shape or arrays["gxi"].shape != eta.shape:
        raise ValueError("BF shard eta, xi, and G(eta)xi shapes differ")
    row_count = int(eta.shape[0])
    for name in (
        "depth",
        "time",
        "case_local_index",
        "frame_index",
        "selected_dense_index",
    ):
        if arrays[name].shape != (row_count,):
            raise ValueError(f"BF shard {name} has the wrong row shape")
    if not all(
        np.isfinite(arrays[name]).all()
        for name in ("eta", "xi", "gxi", "depth", "time")
    ):
        raise ValueError("stored BF eta/xi/G(eta)xi/depth/time must be finite")

    cursor = 0
    frame_indices = np.arange(ROWS_PER_ACCEPTED_CASE, dtype=np.int32)
    for local_index, (proposal, case) in enumerate(zip(proposed, cases)):
        selected = np.flatnonzero(arrays["case_local_index"] == local_index)
        if not case.accepted:
            if selected.size:
                raise ValueError("rejected BF case appears in its shard")
            continue
        expected_positions = np.arange(cursor, cursor + ROWS_PER_ACCEPTED_CASE)
        if not np.array_equal(selected, expected_positions):
            raise ValueError("accepted BF case rows do not form a contiguous block")
        if case.first_row != cursor:
            raise ValueError("BF result first_row differs from its shard")
        if not np.array_equal(arrays["frame_index"][selected], frame_indices):
            raise ValueError("BF stored frame indices are not 0 through 199")
        expected_dense_indices = select_uniform_times(
            proposal.saved_time_count,
            keep_samples=ROWS_PER_ACCEPTED_CASE,
        )
        if not np.array_equal(
            arrays["selected_dense_index"][selected],
            expected_dense_indices,
        ):
            raise ValueError("BF stored dense-time selection is not uniform")
        expected_times = saved_dt * expected_dense_indices.astype(np.float64)
        if not np.array_equal(arrays["time"][selected], expected_times):
            raise ValueError("BF stored times differ from their dense indices")
        if (
            float(expected_times[0]) != 0.0
            or float(expected_times[-1]) != proposal.realized_terminal_time
        ):
            raise ValueError("BF stored rows omit t=0 or the realized endpoint")
        depths = arrays["depth"][selected]
        if not np.all(depths == proposal.depth):
            raise ValueError("BF stored depth differs from its proposal")
        if np.min(depths[:, None] + arrays["eta"][selected]) <= 0.0:
            raise ValueError("BF stored trajectory crosses the bottom")
        cursor += ROWS_PER_ACCEPTED_CASE
    if cursor != row_count:
        raise RuntimeError("BF shard row blocks do not cover the shard")
    return row_count


def _validate_dataset_view(
    *,
    chunk: CompletedChunk,
    summary: Mapping[str, object],
    proposed_by_batch: Mapping[int, Sequence[ProposedCase]],
    cases_by_batch: Mapping[int, Sequence[AuditedCase]],
    result_by_batch: Mapping[int, Mapping[str, object]],
    totals: AuditTotals,
) -> dict[str, object]:
    """Validate the chunk's derived manifest and row ownership map."""

    view = _required_mapping(summary, "dataset_view", context="summary")
    manifest_record = _required_mapping(view, "manifest", context="dataset_view")
    map_record = _required_mapping(
        view,
        "trajectory_map",
        context="dataset_view",
    )
    manifest_path = chunk.root / _required_string(
        manifest_record,
        "path",
        context="dataset_view.manifest",
    )
    map_path = chunk.root / _required_string(
        map_record,
        "path",
        context="dataset_view.trajectory_map",
    )
    manifest_sha256 = file_sha256(manifest_path)
    map_sha256 = file_sha256(map_path)
    if manifest_record.get("sha256") != manifest_sha256:
        raise ValueError("summary manifest hash is stale or incorrect")
    if map_record.get("sha256") != map_sha256:
        raise ValueError("summary trajectory-map hash is stale or incorrect")
    totals.dataset_view_bytes_checked += (
        manifest_path.stat().st_size + map_path.stat().st_size
    )

    manifest = _strict_json_object(manifest_path)
    expected_rows = chunk.accepted_count * ROWS_PER_ACCEPTED_CASE
    expected_scalars = {
        "schema_version": 2,
        "configuration_fingerprint": chunk.fingerprint,
        "configuration_fingerprints": [chunk.fingerprint],
        "n_rows": expected_rows,
        "n_trajectories": chunk.attempted_count,
        "n_accepted_trajectories": chunk.accepted_count,
        "n_accepted_rows": expected_rows,
        "requires_trajectory_map": True,
        "trajectory_map_sha256": map_sha256,
    }
    for name, expected in expected_scalars.items():
        if not _same_json(manifest.get(name), expected):
            raise ValueError(f"dataset manifest has incorrect {name}")
    contract = _required_mapping(
        manifest,
        "dataset_contract",
        context="dataset manifest",
    )
    if manifest.get("dataset_contract_fingerprint") != canonical_json_sha256(contract):
        raise ValueError("dataset contract fingerprint is incorrect")

    batches = _required_list(manifest, "dataset_batches", context="manifest")
    shards = _required_list(manifest, "dataset_shards", context="manifest")
    if len(batches) != len(chunk.batches) or len(shards) != len(chunk.batches):
        raise ValueError("dataset manifest omits a committed BF batch")
    shard_index_by_batch: dict[int, int] = {}
    for shard_index, (raw_batch, raw_shard, paths) in enumerate(
        zip(batches, shards, chunk.batches)
    ):
        if not isinstance(raw_batch, Mapping) or not isinstance(raw_shard, Mapping):
            raise TypeError("dataset batch and shard records must be objects")
        batch_id = int(paths.result.stem.removeprefix("batch_"))
        result = result_by_batch[batch_id]
        if raw_batch.get("batch_id") != batch_id:
            raise ValueError("dataset batch IDs are out of order")
        if raw_batch.get("proposal_sha256") != result.get("proposal_sha256"):
            raise ValueError("dataset proposal hash differs from its result")
        if raw_batch.get("result_sha256") != file_sha256(paths.result):
            raise ValueError("dataset result hash differs from the stored result")
        if raw_shard.get("batch_index") != batch_id:
            raise ValueError("dataset shard batch index is incorrect")
        if raw_shard.get("sha256") != result.get("shard_sha256"):
            raise ValueError("dataset shard hash differs from its result")
        shard_index_by_batch[batch_id] = shard_index

    with np.load(map_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    _validate_map_array_schema(arrays)
    proposed_cases = tuple(
        proposal
        for paths in chunk.batches
        for proposal in proposed_by_batch[int(paths.result.stem.removeprefix("batch_"))]
    )
    cases = tuple(
        case
        for paths in chunk.batches
        for case in cases_by_batch[int(paths.result.stem.removeprefix("batch_"))]
    )
    trajectory_count = len(cases)
    if trajectory_count != chunk.attempted_count or len(proposed_cases) != (
        trajectory_count
    ):
        raise RuntimeError("BF trajectory-map audit has inconsistent case counts")
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
    expected_cell_ids = np.asarray(
        [proposal.cell_code for proposal in proposed_cases],
        dtype=np.int32,
    )
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
        ("trajectory_cell_id", expected_cell_ids),
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
    split_code = _split_code(chunk.split)
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
            arrays["shard_index"][selected] == shard_index_by_batch[case.batch_id]
        ):
            raise ValueError("row-to-shard ownership is incorrect")
        expected_shard_rows = case.first_row + frame_indices.astype(np.int64)
        if not np.array_equal(arrays["shard_row"][selected], expected_shard_rows):
            raise ValueError("stored shard-row ownership is incorrect")
        row_cursor += ROWS_PER_ACCEPTED_CASE
    if row_cursor != expected_rows:
        raise RuntimeError("accepted BF case rows do not sum to the manifest")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "trajectory_map_path": str(map_path),
        "trajectory_map_sha256": map_sha256,
        "dataset_contract_fingerprint": manifest["dataset_contract_fingerprint"],
    }


def _cell_code_mapping(run_spec: Mapping[str, object]) -> dict[int, str]:
    """Require the exact current ordered 66-cell BF taxonomy."""

    raw_codes = _required_mapping(run_spec, "cell_codes", context="run_spec")
    expected_ids = BENJAMIN_FEIR_SAMPLE_CELL_IDS
    if tuple(raw_codes) != expected_ids:
        raise ValueError("BF run cell order is not the current 66-cell taxonomy")
    mapping = {
        _required_integer(raw_codes, cell_id, context="run_spec.cell_codes"): cell_id
        for cell_id in expected_ids
    }
    if mapping != {index: cell_id for index, cell_id in enumerate(expected_ids)}:
        raise ValueError("BF run uses noncanonical cell codes")
    configuration = _required_mapping(
        run_spec,
        "configuration",
        context="run_spec",
    )
    ordered = _required_list(
        configuration,
        "ordered_cell_ids",
        context="run_spec.configuration",
    )
    if ordered != list(expected_ids):
        raise ValueError("BF configuration has a noncurrent 66-cell ordering")
    return mapping


def _expected_chunk_quotas(expected: ExpectedChunk) -> dict[str, int]:
    cell_ids = BENJAMIN_FEIR_SAMPLE_CELL_IDS
    before = balanced_valid_case_targets(
        cell_ids,
        case_count=expected.accepted_before,
    )
    after = balanced_valid_case_targets(
        cell_ids,
        case_count=expected.accepted_before + expected.accepted_count,
    )
    return {
        after_quota.cell_id: (after_quota.case_count - before_quota.case_count)
        for before_quota, after_quota in zip(before, after)
    }


def _validate_chunk_taxonomy_and_quotas(
    run_spec: Mapping[str, object],
    expected: ExpectedChunk,
) -> tuple[dict[int, str], dict[str, int]]:
    """Require canonical codes and the exact incremental quota allocation."""

    mapping = _cell_code_mapping(run_spec)
    expected_quotas = _expected_chunk_quotas(expected)
    expected_records = [
        {"cell_id": cell_id, "target_accepted": target}
        for cell_id, target in expected_quotas.items()
    ]
    raw_quotas = _required_list(run_spec, "quotas", context="run_spec")
    if not _same_json(raw_quotas, expected_records):
        raise ValueError("BF chunk does not have exact balanced 66-cell quotas")
    return mapping, expected_quotas


def _validate_summary_cell_counts(
    summary: Mapping[str, object],
    *,
    expected_quotas: Mapping[str, int],
    attempted_by_cell: Mapping[str, int],
    accepted_by_cell: Mapping[str, int],
) -> None:
    """Bind every summary cell count to reconstructed proposals and results."""

    counts = _required_mapping(summary, "counts", context="summary")
    by_cell = _required_mapping(counts, "by_cell", context="summary.counts")
    expected_cell_ids = set(expected_quotas)
    if set(by_cell) != expected_cell_ids:
        raise ValueError("BF summary must contain the exact 66-cell taxonomy")

    expected_records: dict[str, dict[str, int]] = {}
    for cell_id, target in expected_quotas.items():
        attempted = int(attempted_by_cell.get(cell_id, 0))
        accepted = int(accepted_by_cell.get(cell_id, 0))
        rejected = attempted - accepted
        if accepted != target or rejected < 0:
            raise ValueError("BF reconstructed cell counts do not close")
        expected_records[cell_id] = {
            "target_accepted": target,
            "attempted": attempted,
            "accepted": accepted,
            "rejected": rejected,
        }

    fields = ("target_accepted", "attempted", "accepted", "rejected")
    for cell_id, expected_record in expected_records.items():
        cell_record = _required_mapping(
            by_cell,
            cell_id,
            context="summary.counts.by_cell",
        )
        if set(cell_record) != set(fields):
            raise ValueError(
                f"BF summary cell {cell_id} must contain the exact count fields"
            )
        observed = {
            field: _required_integer(
                cell_record,
                field,
                context=f"summary.counts.by_cell.{cell_id}",
                minimum=0,
            )
            for field in fields
        }
        if observed["attempted"] != observed["accepted"] + observed["rejected"]:
            raise ValueError(f"BF summary cell {cell_id} counts do not close")
        if observed != expected_record:
            raise ValueError(
                f"BF summary cell {cell_id} differs from reconstructed transactions"
            )

    expected_totals = {
        field: sum(record[field] for record in expected_records.values())
        for field in ("attempted", "accepted", "rejected")
    }
    for field, expected_total in expected_totals.items():
        observed_total = _required_integer(
            counts,
            field,
            context="summary.counts",
            minimum=0,
        )
        if observed_total != expected_total:
            raise ValueError(
                f"BF summary {field} total differs from reconstructed transactions"
            )


def _audit_chunk(
    expected: ExpectedChunk,
    chunk: CompletedChunk,
    *,
    totals: AuditTotals,
    support_extrema: Mapping[str, Extrema],
    accepted_extrema: Mapping[str, Extrema],
    attempted_extrema: Mapping[str, Extrema],
    rejection_reasons: Counter[str],
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
    if (
        run_spec.get("batch_size") != EXPECTED_BATCH_SIZE
        or run_spec.get("first_attempt_index") != 0
        or run_spec.get("maximum_attempts_per_accepted_case")
        != EXPECTED_MAXIMUM_ATTEMPTS_PER_ACCEPTED_CASE
    ):
        raise ValueError("BF chunk differs from the exact quota-run policy")
    cell_ids_by_code, expected_quotas = _validate_chunk_taxonomy_and_quotas(
        run_spec,
        expected,
    )

    proposed_by_batch: dict[int, tuple[ProposedCase, ...]] = {}
    cases_by_batch: dict[int, tuple[AuditedCase, ...]] = {}
    result_by_batch: dict[int, Mapping[str, object]] = {}
    accepted_by_cell: Counter[str] = Counter()
    attempted_by_cell: Counter[str] = Counter()
    for paths in chunk.batches:
        batch_id = int(paths.result.stem.removeprefix("batch_"))
        totals.committed_batches += 1
        artifact_paths = [paths.proposal, paths.result]
        if paths.shard.is_file():
            artifact_paths.append(paths.shard)
        totals.transaction_artifact_bytes_checked += sum(
            path.stat().st_size for path in artifact_paths
        )
        _scan_finite_archive(paths.proposal, totals=totals)
        proposals = _case_specifications(
            paths.proposal,
            split=chunk.split,
            stream_id=chunk.stream_id,
            batch_id=batch_id,
            configuration_fingerprint=chunk.fingerprint,
            saved_dt=saved_dt,
            cell_ids_by_code=cell_ids_by_code,
            support_extrema=support_extrema,
            totals=totals,
        )
        result = _strict_json_object(paths.result)
        _validate_result_metadata(
            result,
            run_spec=run_spec,
            execution=execution,
            context=str(paths.result),
        )
        if result.get("config_fingerprint") != chunk.fingerprint:
            raise ValueError("BF result has the wrong configuration fingerprint")
        proposal_sha256 = file_sha256(paths.proposal)
        if result.get("proposal_sha256") != proposal_sha256:
            raise ValueError("BF result has the wrong proposal hash")
        expected_shard_sha256 = (
            file_sha256(paths.shard) if paths.shard.is_file() else None
        )
        if result.get("shard_sha256") != expected_shard_sha256:
            raise ValueError("BF result has the wrong shard hash")
        values = _required_list(result, "cases", context=str(paths.result))
        if len(values) != len(proposals):
            raise ValueError("BF result omits proposed cases")
        batch_cases = tuple(
            _validate_case_record(
                value,
                proposed=proposal,
                batch_id=batch_id,
                residual_tolerance=residual_tolerance,
                hamiltonian_threshold=hamiltonian_threshold,
                production_dt=production_dt,
                accepted_extrema=accepted_extrema,
                attempted_extrema=attempted_extrema,
                rejection_reasons=rejection_reasons,
            )
            for value, proposal in zip(values, proposals)
        )
        shard_rows = _validate_shard(
            paths.shard,
            proposal_sha256=proposal_sha256,
            configuration_fingerprint=chunk.fingerprint,
            proposed=proposals,
            cases=batch_cases,
            delivered_nx=int(numerical.get("target_nx", numerical["nx"])),
            saved_dt=saved_dt,
            totals=totals,
        )
        if shard_rows != sum(case.row_count for case in batch_cases):
            raise ValueError("BF shard row count differs from its results")
        proposed_by_batch[batch_id] = proposals
        cases_by_batch[batch_id] = batch_cases
        result_by_batch[batch_id] = result
        attempted_by_cell.update(proposal.cell_id for proposal in proposals)
        accepted_by_cell.update(
            proposal.cell_id
            for proposal, case in zip(proposals, batch_cases)
            if case.accepted
        )

    proposals = tuple(
        proposal
        for paths in chunk.batches
        for proposal in proposed_by_batch[int(paths.result.stem.removeprefix("batch_"))]
    )
    cases = tuple(
        case
        for paths in chunk.batches
        for case in cases_by_batch[int(paths.result.stem.removeprefix("batch_"))]
    )
    attempted = len(cases)
    accepted = sum(case.accepted for case in cases)
    rejected = attempted - accepted
    retained_rows = sum(case.row_count for case in cases)
    if (
        attempted != chunk.attempted_count
        or accepted != chunk.accepted_count
        or retained_rows != accepted * ROWS_PER_ACCEPTED_CASE
    ):
        raise ValueError("transaction counts disagree with the completed chunk")
    if [proposal.attempt_index for proposal in proposals] != list(range(attempted)):
        raise ValueError("BF attempts are not retained exactly once in order")
    if len({proposal.case_id for proposal in proposals}) != attempted:
        raise ValueError("BF case IDs repeat within a completed chunk")
    expected_nonzero_quotas = {
        cell_id: count for cell_id, count in expected_quotas.items() if count
    }
    if dict(accepted_by_cell) != expected_nonzero_quotas:
        raise ValueError("BF accepted cases do not meet exact 66-cell quotas")
    if set(attempted_by_cell) - set(expected_quotas):
        raise ValueError("BF attempts contain an unknown cell")
    _validate_summary_cell_counts(
        summary,
        expected_quotas=expected_quotas,
        attempted_by_cell=attempted_by_cell,
        accepted_by_cell=accepted_by_cell,
    )
    totals.attempted += attempted
    totals.accepted += accepted
    totals.rejected += rejected
    totals.retained_rows += retained_rows
    view_record = _validate_dataset_view(
        chunk=chunk,
        summary=summary,
        proposed_by_batch=proposed_by_batch,
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
        "configuration_fingerprint": chunk.fingerprint,
        "accepted_by_cell": dict(accepted_by_cell),
        "attempted_by_cell": dict(attempted_by_cell),
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
        for path in root.glob(
            "*/benjamin_feir/*/paper_dataset_benjamin_feir_*.summary.json"
        )
    }
    if discovered_paths != expected_paths:
        missing = sorted(map(str, expected_paths - discovered_paths))
        unexpected = sorted(map(str, discovered_paths - expected_paths))
        raise ValueError(
            "BF completion summaries differ from the exact six-chunk plan; "
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
            "benjamin_feir",
            EXPECTED_REVISION_ID,
            expected.split,
            expected.stream_id,
            expected.accepted_before,
            expected.accepted_count,
            expected.accepted_before + expected.accepted_count,
        )
        if observed != wanted:
            raise ValueError(f"{expected.label} differs from its immutable plan")
    return chunks


def _historical_shared_source_snapshot_record(
    source_mappings: Sequence[Mapping[str, str]],
    *,
    snapshot_root: Path = HISTORICAL_SHARED_SOURCE_SNAPSHOT_ROOT,
) -> dict[str, object]:
    """Authenticate recovered shared bytes against all six BF source maps."""

    if len(source_mappings) != len(EXPECTED_CHUNKS):
        raise ValueError("historical BF source binding requires all six chunks")
    root = Path(snapshot_root).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("historical BF source snapshot root is not a real directory")
    root = root.resolve()
    checksum_path = root / "SHA256SUMS"
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise ValueError("historical BF source snapshot omits SHA256SUMS")
    expected_checksum_text = "".join(
        f"{digest}  {snapshot_name}\n"
        for snapshot_name, digest in HISTORICAL_SHARED_SOURCE_SNAPSHOTS.values()
    )
    if checksum_path.read_text(encoding="utf-8") != expected_checksum_text:
        raise ValueError("historical BF source snapshot SHA256SUMS differs")

    bindings: dict[str, object] = {}
    for repository_path, (
        snapshot_name,
        expected_digest,
    ) in HISTORICAL_SHARED_SOURCE_SNAPSHOTS.items():
        snapshot_path = root / snapshot_name
        if snapshot_path.is_symlink() or not snapshot_path.is_file():
            raise ValueError(f"historical BF snapshot is missing {snapshot_name}")
        if file_sha256(snapshot_path) != expected_digest:
            raise ValueError(f"historical BF snapshot bytes differ for {snapshot_name}")
        for chunk_index, source_mapping in enumerate(source_mappings):
            if source_mapping.get(repository_path) != expected_digest:
                raise ValueError(
                    "historical BF snapshot differs from chunk source map "
                    f"{chunk_index}: {repository_path}"
                )
        bindings[repository_path] = {
            "snapshot_path": str(snapshot_path),
            "bytes": snapshot_path.stat().st_size,
            "sha256": expected_digest,
        }

    return {
        "schema": "paper_dataset_bf_revision4_historical_source_binding_v1",
        "role": "inert_historical_byte_recovery_only",
        "snapshot_root": str(root),
        "sha256sums": {
            "path": str(checksum_path),
            "bytes": checksum_path.stat().st_size,
            "sha256": file_sha256(checksum_path),
        },
        "sources": bindings,
        "chunk_source_maps_checked": len(source_mappings),
    }


def _nonhistorical_generation_source_record(
    source_mappings: Sequence[Mapping[str, str]],
    *,
    repository_root: Path = ROOT,
) -> dict[str, object]:
    """Bind every nonhistorical generation source to current repository bytes."""

    if len(source_mappings) != len(EXPECTED_CHUNKS):
        raise ValueError("current BF source binding requires all six chunks")
    first_sources = dict(source_mappings[0])
    if any(dict(mapping) != first_sources for mapping in source_mappings[1:]):
        raise ValueError("BF chunks do not contain exactly equal source maps")
    if set(first_sources) != EXPECTED_GENERATION_SOURCE_PATHS:
        missing = sorted(EXPECTED_GENERATION_SOURCE_PATHS - set(first_sources))
        extra = sorted(set(first_sources) - EXPECTED_GENERATION_SOURCE_PATHS)
        raise ValueError(
            "BF source map differs from the exact frozen 19-path contract; "
            f"missing={missing}, extra={extra}"
        )
    if any(
        not isinstance(path, str)
        or not path
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for path, digest in first_sources.items()
    ):
        raise ValueError("BF source map paths/digests are malformed")
    historical_paths = set(HISTORICAL_SHARED_SOURCE_SNAPSHOTS)
    if len(first_sources) != 19 or len(first_sources) - len(historical_paths) != 17:
        raise ValueError("BF source map does not contain exactly 17 current sources")

    requested_root = Path(repository_root).expanduser()
    if requested_root.is_symlink() or not requested_root.is_dir():
        raise ValueError("BF repository root is not a real directory")
    resolved_root = requested_root.resolve()
    bindings: dict[str, object] = {}
    for repository_path, expected_digest in sorted(first_sources.items()):
        if repository_path in historical_paths:
            continue
        current_repository_path = CURRENT_SOURCE_PATH_BY_GENERATION_PATH.get(
            repository_path, repository_path
        )
        relative_path = Path(current_repository_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"BF source path escapes the repository: {repository_path}"
            )
        source_path = resolved_root / relative_path
        current = resolved_root
        for component in relative_path.parts:
            current /= component
            if current.is_symlink():
                raise ValueError(
                    f"BF source path contains a symbolic link: {repository_path}"
                )
        if not source_path.is_file():
            raise ValueError(
                f"BF source is not a regular repository file: {repository_path}"
            )
        resolved_source = source_path.resolve(strict=True)
        if not resolved_source.is_relative_to(resolved_root):
            raise ValueError(
                f"BF source path escapes the repository: {repository_path}"
            )
        observed_digest = file_sha256(resolved_source)
        if observed_digest != expected_digest:
            raise ValueError(
                f"current BF source bytes differ from generation: {repository_path}"
            )
        bindings[repository_path] = {
            "path": str(resolved_source),
            "current_repository_path": current_repository_path,
            "bytes": resolved_source.stat().st_size,
            "sha256": observed_digest,
        }

    return {
        "schema": "paper_dataset_bf_revision4_current_source_binding_v1",
        "role": "current_repository_bytes_for_all_nonhistorical_generation_sources",
        "repository_root": str(resolved_root),
        "source_map_fingerprint": canonical_json_sha256(first_sources),
        "source_count": len(first_sources),
        "current_source_count": len(bindings),
        "historical_snapshot_source_paths": sorted(historical_paths),
        "sources": bindings,
        "chunk_source_maps_checked": len(source_mappings),
    }


def _identity_record(chunks: Sequence[CompletedChunk]) -> dict[str, object]:
    dependency_fingerprints = {chunk.dependency_fingerprint for chunk in chunks}
    execution_fingerprints = {chunk.execution_fingerprint for chunk in chunks}
    source_fingerprints = {chunk.source_fingerprint for chunk in chunks}
    if len(dependency_fingerprints) != 1:
        raise ValueError("BF chunks use different dependency environments")
    if len(execution_fingerprints) != 1:
        raise ValueError("BF chunks use different execution contracts")
    if len(source_fingerprints) != 1:
        raise ValueError("BF chunks use different generation source mappings")
    source_mappings = tuple(chunk.source_sha256 for chunk in chunks)
    nonhistorical_sources = _nonhistorical_generation_source_record(source_mappings)
    historical_shared_sources = _historical_shared_source_snapshot_record(
        source_mappings
    )
    current_execution = CURRENT_BF_EXECUTION.to_json_record()
    current_execution_fingerprint = canonical_json_sha256(current_execution)
    execution_fingerprint = next(iter(execution_fingerprints))
    if execution_fingerprint != current_execution_fingerprint:
        raise ValueError("BF chunks do not use the current execution contract")
    current_dependency_fingerprint = canonical_json_sha256(dependency_environment())
    dependency_fingerprint = next(iter(dependency_fingerprints))
    if dependency_fingerprint != current_dependency_fingerprint:
        raise ValueError("BF chunks do not use the current dependency identity")

    first_sources = dict(chunks[0].source_sha256)
    current_support_hashes = {
        path: file_sha256(ROOT / path) for path in CURRENT_SUPPORT_SOURCE_PATHS
    }
    if any(
        first_sources.get(path) != digest
        for path, digest in current_support_hashes.items()
    ):
        raise ValueError(
            "current BF construction/support sources differ from generation"
        )
    current_support = _benjamin_feir_sampling_support_record()
    support_fingerprint = canonical_json_sha256(current_support)
    for chunk in chunks:
        summary = _strict_json_object(chunk.summary_path)
        run_spec = _required_mapping(summary, "run_spec", context="summary")
        configuration = _required_mapping(
            run_spec,
            "configuration",
            context="summary.run_spec",
        )
        declared_support = _required_mapping(
            configuration,
            "sampling_support",
            context="summary.run_spec.configuration",
        )
        if not _same_json(declared_support, current_support):
            raise ValueError("BF chunk declares a noncurrent sampling support")
    return {
        "dependency_environment_fingerprint": dependency_fingerprint,
        "execution_record_fingerprint": execution_fingerprint,
        "source_sha256_fingerprint": next(iter(source_fingerprints)),
        "sampling_support_fingerprint": support_fingerprint,
        "current_support_source_sha256": current_support_hashes,
        "historical_shared_source_snapshot_binding": historical_shared_sources,
        "nonhistorical_generation_source_binding": nonhistorical_sources,
    }


def _accepted_cell_totals_by_split(
    chunk_records: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, int]]:
    """Require the frozen split totals to balance all 66 current cells."""

    cell_ids = BENJAMIN_FEIR_SAMPLE_CELL_IDS
    expected_totals = {
        SplitId.TRAIN: EXPECTED_TRAIN_ACCEPTED,
        SplitId.VALIDATION: EXPECTED_VALIDATION_ACCEPTED,
        SplitId.TEST: EXPECTED_TEST_ACCEPTED,
    }
    output: dict[str, dict[str, int]] = {}
    for split, accepted_total in expected_totals.items():
        observed: Counter[str] = Counter()
        for record in chunk_records:
            if record.get("split") != split.value:
                continue
            raw_counts = record.get("accepted_by_cell")
            if not isinstance(raw_counts, Mapping):
                raise TypeError("BF chunk accepted_by_cell must be an object")
            observed.update(
                {str(cell_id): int(count) for cell_id, count in raw_counts.items()}
            )
        expected = {
            quota.cell_id: quota.case_count
            for quota in balanced_valid_case_targets(
                cell_ids,
                case_count=accepted_total,
            )
        }
        if dict(observed) != expected:
            raise ValueError(
                f"BF {split.value} population is not balanced over 66 cells"
            )
        output[split.value] = dict(observed)
    return output


def audit(root: Path) -> dict[str, object]:
    """Run the complete read-only BF dataset audit and return its record."""

    resolved_root = Path(root).expanduser().resolve()
    started_at = datetime.now().astimezone()
    started = perf_counter()
    chunks = _load_exact_chunks(resolved_root)
    identity = _identity_record(chunks)
    totals = AuditTotals()
    support_extrema = {
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
    accepted_extrema = {
        name: Extrema()
        for name in (
            "initial_hamiltonian",
            "hamiltonian_drift",
            "stage_residual",
            "minimum_water_column",
        )
    }
    attempted_extrema = {
        name: Extrema()
        for name in (
            "hamiltonian_drift",
            "stage_residual",
            "minimum_water_column",
        )
    }
    rejection_reasons: Counter[str] = Counter()
    chunk_records = tuple(
        _audit_chunk(
            expected,
            chunk,
            totals=totals,
            support_extrema=support_extrema,
            accepted_extrema=accepted_extrema,
            attempted_extrema=attempted_extrema,
            rejection_reasons=rejection_reasons,
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
        raise ValueError("BF split totals differ from the paper dataset plan")
    if totals.accepted != EXPECTED_ACCEPTED:
        raise ValueError("BF accepted total differs from 18,432")
    if totals.proposal_specs_checked != totals.attempted:
        raise ValueError("not every BF proposal specification was replayed")
    if totals.retained_rows != EXPECTED_ACCEPTED * ROWS_PER_ACCEPTED_CASE:
        raise ValueError("BF retained-row total is incorrect")
    if len(BENJAMIN_FEIR_SAMPLE_CELL_IDS) != 66:
        raise ValueError("current BF support no longer has 66 allocation cells")
    accepted_cells = _accepted_cell_totals_by_split(chunk_records)

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
        "finite_float_arrays_checked": totals.float_arrays_checked,
        "finite_float_values_checked": totals.float_values_checked,
        "transaction_artifact_bytes_checked": (
            totals.transaction_artifact_bytes_checked
        ),
        "dataset_view_bytes_checked": totals.dataset_view_bytes_checked,
        "accepted_by_split": dict(accepted_by_split),
        "accepted_by_split_and_cell": accepted_cells,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "support": {
            "allocation_cell_count": len(BENJAMIN_FEIR_SAMPLE_CELL_IDS),
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
        "identity": identity,
        "chunks": list(chunk_records),
        "checks": {
            "exact_six_chunk_plan": True,
            "nested_train_intervals": True,
            "exact_current_66_cell_taxonomy_verified": True,
            "balanced_66_cell_quotas": True,
            "summary_cell_counts_match_reconstructed_transactions": True,
            "historical_shared_source_snapshots_bound_to_all_chunks": True,
            "all_nonhistorical_generation_sources_match_current_bytes": True,
            "immutable_specs_rebuilt": True,
            "all_transactions_rescanned": True,
            "proposal_hashes_verified": True,
            "result_hashes_verified": True,
            "shard_hashes_verified": True,
            "manifest_hashes_verified": True,
            "trajectory_map_hashes_verified": True,
            "accepted_rejected_row_ownership_verified": True,
            "canonical_shard_schema_dtypes_shapes_verified": True,
            "accepted_cases_have_exactly_200_uniform_rows": True,
            "rejected_cases_have_zero_rows": True,
            "stored_eta_xi_gxi_depth_time_finite": True,
            "stored_depths_equal_proposals": True,
            "stored_water_columns_positive": True,
            "stored_times_include_zero_and_realized_endpoint": True,
            "floored_100_carrier_period_horizons_verified": True,
            "all_quality_masks_current_and_consistent": True,
            "accepted_residuals_within_contract": True,
            "accepted_hamiltonian_drift_within_contract": True,
            "accepted_water_columns_positive": True,
            "all_proposals_in_current_bf_support": True,
            "current_bf_specs_replayed_exactly": True,
            "source_dependency_execution_identity_verified": True,
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
        help="Completed revision-4 BF dataset root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Audit artifact path; defaults to "
            "<root>/benjamin_feir_completion_audit.json."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "benjamin_feir_completion_audit.json"
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
