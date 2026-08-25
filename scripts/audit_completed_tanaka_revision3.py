"""Fail-closed CPU audit of the fresh corrected Tanaka revision-3 dataset.

The quota scanner remains the authority for deterministic transaction replay.
This audit adds the release-level checks specific to the fresh Tanaka dataset:
the exact six-chunk interval plan, exact replay of every eleven-cell proposal,
the current corrected amplitude-inversion construction, numerical-health and
terminal-time checks, stored-field finiteness, and an independent audit of
each loader-facing manifest and trajectory map.

The saved artifacts contain every requested crest but not the constructor's
per-component achieved height.  Generation nevertheless cannot return from
the frozen constructor unless ``validate_solved_amplitudes`` passes.  This
audit proves that call path and its tolerances from the exact current source
identity; it deliberately does not claim that nonexistent achieved-height
measurements were persisted.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import inspect
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

# All imports below this point can initialize JAX.  A completion audit must
# never claim a GPU or perturb the live dataset generators.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["DNO_TANAKA_DTYPE"] = "float64"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-tanaka-completion-audit")

import numpy as np  # noqa: E402

from scripts.build_paper_dataset_view import (  # noqa: E402
    CompletedChunk,
    TRAJECTORY_MAP_DTYPES,
    load_completed_chunk,
)
from scripts.generate_paper_dataset import (  # noqa: E402
    dependency_environment,
    source_hashes,
)
from solver.gen_data.tanaka_initial_conditions import (  # noqa: E402
    TANAKA_FINE_FACTOR,
    TANAKA_POTENTIAL_RADICAND_FAILURE_SCHEMA,
    TANAKA_PROFILE_RECONSTRUCTION,
    TANAKA_PROFILE_TOLERANCE,
    TanakaPotentialRadicandError,
    build_per_case_initial_conditions,
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
from solver.gen_data.tanaka_sampling import (  # noqa: E402
    MAIN_DEPTH_BOUNDS,
    MAIN_TOTAL_ALPHA_BOUNDS,
    SEPARATION_TO_DEPTH_RATIO,
    STEEP_ALPHA_BOUNDS,
    STEEP_DEPTH_BOUNDS,
    TANAKA_DELIVERED_MAXIMUM_WAVENUMBER,
    TANAKA_MINIMUM_RESOLUTION_RATIO,
    TANAKA_SAMPLE_CELL_IDS,
    TANAKA_SAMPLE_CELLS,
    TANAKA_SAMPLING_REVISION_V3,
    TanakaSample,
    sample_tanaka_case,
    tanaka_support_violations,
)
from solver.gen_data.trajectory_family_adapters import (  # noqa: E402
    construct_tanaka_trajectory_batch,
    sample_tanaka_trajectory_cases,
)
from solver.gen_data.trajectory_batch_executor import (  # noqa: E402
    TrajectoryExecutionConfig,
    classify_tanaka_construction_failure,
)
from solver.tanaka_ICs import modified_tanaka as tanaka_solver  # noqa: E402
from solver.tanaka_ICs.modified_tanaka import (  # noqa: E402
    AMPLITUDE_ABSOLUTE_TOLERANCE,
    AMPLITUDE_RELATIVE_TOLERANCE,
    DEFAULT_OUTER_ITERATIONS,
    DEFAULT_QC_UPPER,
    make_default_tanaka_template,
    solve_modified_tanaka_batched,
    validate_solved_amplitudes,
)


DEFAULT_DATASET_ROOT = ROOT / "outputs/paper_dataset_revision3_literature_aligned_v1"
AUDIT_SCHEMA = "paper_dataset_tanaka_revision3_completion_audit_v1"
ROWS_PER_ACCEPTED_CASE = 200
EXPECTED_FAMILY_ID = int(PhysicalFamilyId.TANAKA)
EXPECTED_REVISION_ID = TANAKA_SAMPLING_REVISION_V3
EXPECTED_ACCEPTED = 18_432
EXPECTED_TRAIN_ACCEPTED = 16_384
EXPECTED_VALIDATION_ACCEPTED = 1_024
EXPECTED_TEST_ACCEPTED = 1_024
EXPECTED_TERMINAL_TIME = 200.0
EXPECTED_BATCH_SIZE = 256
EXPECTED_MAXIMUM_ATTEMPTS_PER_ACCEPTED_CASE = 4
BELOW_OLD_FLOOR_REQUESTED_AMPLITUDE = 2.4102831187118947e-5
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
EXPECTED_ROLLOUT_METRIC_KEYS = frozenset(
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
EXPECTED_CONSTRUCTION_REJECTION_METRIC_KEYS = frozenset(
    {
        "construction_status",
        "construction_failure_json",
        "construction_failure_reason",
        "original_local_index",
    }
)
EXPECTED_REQUIRED_BITS = int(
    QualityReason.OUTSIDE_SUPPORT | QualityReason.INCOMPLETE_TRAJECTORY
)
EXPECTED_ACCEPTED_EVALUATED_BITS = int(
    QualityReason.OUTSIDE_SUPPORT
    | QualityReason.INCOMPLETE_TRAJECTORY
    | QualityReason.GL2_STAGE_RESIDUAL
    | QualityReason.NONFINITE_STATE
    | QualityReason.NONFINITE_TARGET
    | QualityReason.BOTTOM_CLEARANCE
)


@dataclass(frozen=True)
class ExpectedChunk:
    """One exact additive interval in the final Tanaka population."""

    label: str
    relative_root: Path
    split: SplitId
    stream_id: int
    accepted_before: int
    accepted_count: int

    @property
    def summary_path(self) -> Path:
        return (
            self.relative_root / f"paper_dataset_tanaka_{self.split.value}.summary.json"
        )


EXPECTED_CHUNKS = (
    ExpectedChunk(
        "train_00000_02048",
        Path("train/tanaka/chunk_00000_02048"),
        SplitId.TRAIN,
        0,
        0,
        2_048,
    ),
    ExpectedChunk(
        "train_02048_02048",
        Path("train/tanaka/chunk_02048_02048"),
        SplitId.TRAIN,
        1,
        2_048,
        2_048,
    ),
    ExpectedChunk(
        "train_04096_04096",
        Path("train/tanaka/chunk_04096_04096"),
        SplitId.TRAIN,
        2,
        4_096,
        4_096,
    ),
    ExpectedChunk(
        "train_08192_08192",
        Path("train/tanaka/chunk_08192_08192"),
        SplitId.TRAIN,
        3,
        8_192,
        8_192,
    ),
    ExpectedChunk(
        "validation_00000_01024",
        Path("validation/tanaka/chunk_00000_01024"),
        SplitId.VALIDATION,
        100,
        0,
        1_024,
    ),
    ExpectedChunk(
        "test_00000_01024",
        Path("test/tanaka/chunk_00000_01024"),
        SplitId.TEST,
        200,
        0,
        1_024,
    ),
)


@dataclass(frozen=True)
class ProposedCase:
    """One exactly replayed proposal in immutable batch order."""

    case_id: int
    cell_code: int
    cell_id: str
    depth: float
    sample: TanakaSample


@dataclass(frozen=True)
class AuditedCase:
    """One committed result record in immutable batch order."""

    case_id: int
    accepted: bool
    required_bits: int
    evaluated_bits: int
    failed_bits: int
    first_row: int
    row_count: int
    batch_id: int


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
    """Dataset-wide counters accumulated by the read-only audit."""

    attempted: int = 0
    accepted: int = 0
    rejected: int = 0
    retained_rows: int = 0
    committed_batches: int = 0
    proposal_specs_checked: int = 0
    requested_crests_checked: int = 0
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


def _scan_finite_archive(path: Path, *, totals: AuditTotals) -> None:
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


def _direct_call_names(function: object) -> frozenset[str]:
    """Return direct call-site names from one current Python function."""

    tree = ast.parse(inspect.getsource(function))
    return frozenset(
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    )


def _amplitude_behavioral_sentinel() -> dict[str, object]:
    """Execute the corrected below-old-floor solve and its validator."""

    requested = np.asarray(
        (BELOW_OLD_FLOOR_REQUESTED_AMPLITUDE,),
        dtype=np.float64,
    )
    template = make_default_tanaka_template(
        nx=32,
        dno_order=0,
        pad_factor=1,
    )
    validation_call_count = 0
    validator = tanaka_solver.validate_solved_amplitudes

    def counting_validator(
        eta_profile: object,
        requested_amplitudes: object,
    ) -> None:
        nonlocal validation_call_count
        validation_call_count += 1
        validator(eta_profile, requested_amplitudes)

    tanaka_solver.validate_solved_amplitudes = counting_validator
    try:
        solution = tanaka_solver.solve_modified_tanaka_batched(template, requested)
    finally:
        tanaka_solver.validate_solved_amplitudes = validator

    if validation_call_count != 1:
        raise RuntimeError(
            "the Tanaka below-old-floor solve must execute amplitude validation "
            f"exactly once, observed {validation_call_count} calls"
        )
    achieved_values = np.max(
        np.asarray(solution.eta_profile, dtype=np.float64),
        axis=-1,
    )
    if achieved_values.shape != requested.shape:
        raise RuntimeError("the Tanaka behavioral sentinel returned the wrong shape")
    achieved = float(achieved_values[0])
    absolute_error = abs(achieved - BELOW_OLD_FLOOR_REQUESTED_AMPLITUDE)
    allowed_error = max(
        AMPLITUDE_ABSOLUTE_TOLERANCE,
        AMPLITUDE_RELATIVE_TOLERANCE * BELOW_OLD_FLOOR_REQUESTED_AMPLITUDE,
    )
    if not math.isfinite(achieved) or absolute_error > allowed_error:
        raise RuntimeError(
            "the Tanaka below-old-floor solve did not achieve its requested crest"
        )

    mismatch_requested = requested * 2.0
    mismatch_rejected = False
    try:
        validator(solution.eta_profile, mismatch_requested)
    except ValueError:
        mismatch_rejected = True
    if not mismatch_rejected:
        raise RuntimeError(
            "the Tanaka amplitude validator accepted a deliberate crest mismatch"
        )

    return {
        "schema": "tanaka_below_old_floor_amplitude_behavior_v1",
        "requested_amplitude": BELOW_OLD_FLOOR_REQUESTED_AMPLITUDE,
        "achieved_amplitude": achieved,
        "absolute_error": absolute_error,
        "allowed_error": allowed_error,
        "validation_call_count": validation_call_count,
        "deliberate_mismatch_requested_amplitude": float(mismatch_requested[0]),
        "deliberate_mismatch_rejected": mismatch_rejected,
    }


def _amplitude_inversion_record(
    execution: TrajectoryExecutionConfig,
) -> dict[str, object]:
    """Verify and describe the corrected constructor's current success path."""

    numerical = execution.numerical
    template = make_default_tanaka_template(
        depth=1.0,
        gravity=numerical.gravity,
        direction=1,
        nx=numerical.nx,
        length=numerical.length,
        center=0.0,
        dno_order=numerical.dno_order,
        pad_factor=numerical.pad_factor,
    )
    required_edges = (
        (
            construct_tanaka_trajectory_batch,
            "build_per_case_initial_conditions",
        ),
        (
            build_per_case_initial_conditions,
            "solve_modified_tanaka_batched",
        ),
        (solve_modified_tanaka_batched, validate_solved_amplitudes.__name__),
    )
    missing = tuple(
        f"{function.__name__}->{callee}"
        for function, callee in required_edges
        if callee not in _direct_call_names(function)
    )
    if missing:
        raise RuntimeError(
            "current Tanaka constructor no longer has the required amplitude "
            f"validation call path: {missing}"
        )
    if (
        template.qc_upper != DEFAULT_QC_UPPER
        or template.outer_iterations != DEFAULT_OUTER_ITERATIONS
        or DEFAULT_QC_UPPER != 1.0 - 1.0e-12
        or DEFAULT_OUTER_ITERATIONS != 48
        or AMPLITUDE_RELATIVE_TOLERANCE != 1.0e-6
        or AMPLITUDE_ABSOLUTE_TOLERANCE != 1.0e-14
    ):
        raise RuntimeError("current Tanaka amplitude-inversion constants changed")
    behavioral_sentinel = _amplitude_behavioral_sentinel()
    return {
        "schema": "tanaka_corrected_amplitude_inversion_identity_v2",
        "profile_reconstruction": TANAKA_PROFILE_RECONSTRUCTION,
        "profile_fine_factor": TANAKA_FINE_FACTOR,
        "profile_join_tolerance": TANAKA_PROFILE_TOLERANCE,
        "constructor_settings": {
            "dimensionless_template_depth": 1.0,
            "crest_spec_amplitude_is_alpha": True,
        },
        "template_parameters": asdict(template),
        "amplitude_bisection": {
            "qc_upper": DEFAULT_QC_UPPER,
            "outer_iterations": DEFAULT_OUTER_ITERATIONS,
        },
        "success_postcondition": {
            "function": validate_solved_amplitudes.__name__,
            "relative_tolerance": AMPLITUDE_RELATIVE_TOLERANCE,
            "absolute_tolerance": AMPLITUDE_ABSOLUTE_TOLERANCE,
            "call_path": [
                "construct_tanaka_trajectory_batch",
                "build_per_case_initial_conditions",
                "solve_modified_tanaka_batched",
                "validate_solved_amplitudes",
            ],
            "must_pass_before_constructor_returns": True,
        },
        "behavioral_sentinel": behavioral_sentinel,
        "durable_artifact_limitation": {
            "requested_per_crest_amplitudes_persisted": True,
            "achieved_per_crest_amplitudes_persisted": False,
            "posthoc_claim": (
                "exact current success-path validation is source-bound; "
                "per-crest achieved heights cannot be independently "
                "recomputed from aggregated stored fields"
            ),
        },
    }


def _support_record() -> dict[str, object]:
    """Return the elementary revision-3 support bound by current source."""

    return {
        "schema": "paper_tanaka_sampling_support_revision3_v1",
        "revision_id": EXPECTED_REVISION_ID,
        "cells": [
            {
                "cell_id": cell_id,
                "regime": regime,
                "crest_count": crest_count,
                "right_moving_count": right_moving_count,
            }
            for cell_id, (regime, crest_count, right_moving_count) in (
                TANAKA_SAMPLE_CELLS.items()
            )
        ],
        "main_depth_bounds": list(MAIN_DEPTH_BOUNDS),
        "main_total_alpha_bounds": list(MAIN_TOTAL_ALPHA_BOUNDS),
        "steep_depth_bounds": list(STEEP_DEPTH_BOUNDS),
        "steep_alpha_bounds": list(STEEP_ALPHA_BOUNDS),
        "separation_to_depth_ratio": SEPARATION_TO_DEPTH_RATIO,
        "delivered_maximum_wavenumber": TANAKA_DELIVERED_MAXIMUM_WAVENUMBER,
        "minimum_wavenumbers_per_inverse_width": (TANAKA_MINIMUM_RESOLUTION_RATIO),
        "depth_law": (
            "log-uniform on the cell interval after raising its lower bound "
            "to satisfy the profile-resolution condition"
        ),
    }


def _proposal_cases(
    proposal_path: Path,
    *,
    split: SplitId,
    cell_ids_by_code: Mapping[int, str],
    support_extrema: Mapping[str, Extrema],
    totals: AuditTotals,
) -> tuple[ProposedCase, ...]:
    """Replay every proposed Tanaka specification through current code."""

    with np.load(proposal_path, allow_pickle=False) as archive:
        if frozenset(archive.files) != EXPECTED_PROPOSAL_ARRAYS:
            raise ValueError(f"{proposal_path} has an unknown proposal schema")
        case_ids = np.asarray(archive["case_id"], dtype=np.int64)
        roots = np.asarray(archive["root_seed"], dtype=np.uint64)
        streams = np.asarray(archive["stream_id"], dtype=np.uint32)
        attempts = np.asarray(archive["attempt_index"], dtype=np.uint64)
        encoded_cells = np.asarray(archive["cell_id"], dtype=np.int32)
        specifications = np.asarray(archive["case_spec_json"])
    if not (
        roots.shape
        == streams.shape
        == attempts.shape
        == encoded_cells.shape
        == specifications.shape
        == case_ids.shape
    ):
        raise ValueError(f"{proposal_path} proposal vectors have different shapes")

    replayed: list[ProposedCase] = []
    for index, encoded in enumerate(encoded_cells):
        try:
            cell_id = cell_ids_by_code[int(encoded)]
        except KeyError as error:
            raise ValueError(
                f"{proposal_path} contains unknown Tanaka cell code {int(encoded)}"
            ) from error
        key = CaseKey(
            family_id=EXPECTED_FAMILY_ID,
            revision_id=EXPECTED_REVISION_ID,
            split_id=split,
            stream_id=int(streams[index]),
            attempt_index=int(attempts[index]),
        )
        if int(roots[index]) != key.root_seed or int(case_ids[index]) != key.case_id:
            raise ValueError(
                f"{proposal_path} case {index} has inconsistent run coordinates"
            )
        assignment = AttemptAssignment(case_key=key, cell_id=cell_id)
        expected_sample = sample_tanaka_case(
            assignment,
            domain_length=TrajectoryExecutionConfig.paper("tanaka").numerical.length,
        )
        violations = tanaka_support_violations(expected_sample)
        if violations:
            raise ValueError(
                f"current Tanaka sampler produced unsupported case {key.case_id}: "
                + "; ".join(violations)
            )
        sampled = sample_tanaka_trajectory_cases(
            (assignment,),
            contract=TrajectoryExecutionConfig.paper("tanaka").numerical,
        )
        if sampled.samples != (expected_sample,):
            raise RuntimeError("current Tanaka trajectory adapter changed its sample")
        expected = sampled.specification_records[0]
        observed = _strict_json_text(
            str(specifications[index]),
            context=f"{proposal_path} case_spec_json[{index}]",
        )
        if not _same_json(observed, expected):
            raise ValueError(
                f"{proposal_path} case {key.case_id} differs from exact "
                "current Tanaka deterministic sampling"
            )
        if observed.get("initial_condition_constructor") != (
            TANAKA_PROFILE_RECONSTRUCTION
        ):
            raise ValueError("Tanaka proposal uses the wrong profile reconstruction")
        settings = _required_mapping(
            observed,
            "constructor_settings",
            context="Tanaka case specification",
        )
        if not _same_json(
            settings,
            {
                "dimensionless_template_depth": 1.0,
                "crest_spec_amplitude_is_alpha": True,
            },
        ):
            raise ValueError("Tanaka proposal uses wrong constructor settings")
        crests = _required_list(
            observed,
            "crests",
            context="Tanaka case specification",
        )
        if len(crests) != expected_sample.crest_count:
            raise ValueError("Tanaka proposal has the wrong requested crest count")
        for raw_crest, crest in zip(crests, expected_sample.crests):
            if not isinstance(raw_crest, Mapping):
                raise TypeError("each persisted Tanaka crest must be an object")
            alpha = _finite_number(
                raw_crest.get("alpha"),
                context="Tanaka crest alpha",
            )
            physical_amplitude = _finite_number(
                raw_crest.get("amplitude"),
                context="Tanaka crest amplitude",
            )
            if (
                alpha != crest.alpha
                or physical_amplitude != expected_sample.depth * alpha
                or raw_crest.get("direction") != crest.direction
                or raw_crest.get("center") != crest.center
            ):
                raise ValueError("persisted Tanaka requested crest changed")
            support_extrema["crest_alpha"].add(alpha)
            support_extrema["physical_crest_amplitude"].add(physical_amplitude)
            totals.requested_crests_checked += 1
        support_extrema["depth"].add(expected_sample.depth)
        support_extrema["total_alpha"].add(
            expected_sample.total_dimensionless_amplitude
        )
        support_extrema["resolution_ratio"].add(expected_sample.resolution_ratio)
        support_extrema["minimum_separation"].add(
            expected_sample.achieved_minimum_separation
        )
        totals.proposal_specs_checked += 1
        replayed.append(
            ProposedCase(
                case_id=key.case_id,
                cell_code=int(encoded),
                cell_id=cell_id,
                depth=expected_sample.depth,
                sample=expected_sample,
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
    if result.get("schema") != "paper_dataset_batch_result_v1":
        raise ValueError(f"{context} has an unknown result schema")
    metadata = _required_mapping(result, "metadata", context=context)
    if metadata.get("family") != "tanaka":
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


def _validate_construction_rejection(
    metrics: Mapping[str, object],
    *,
    proposed: ProposedCase,
    local_index: int,
) -> None:
    """Validate a declared finite square-root-domain rejection exactly."""

    if set(metrics) != EXPECTED_CONSTRUCTION_REJECTION_METRIC_KEYS:
        raise ValueError("Tanaka construction rejection has an unknown schema")
    if metrics.get("construction_status") != "declared_outside_support":
        raise ValueError("Tanaka construction rejection has the wrong status")
    if metrics.get("original_local_index") != local_index:
        raise ValueError("Tanaka construction rejection has the wrong local index")
    reason = metrics.get("construction_failure_reason")
    if reason != "negative_or_nonfinite_surface_potential_radicand":
        raise ValueError("Tanaka construction rejection has an unsupported reason")
    encoded = metrics.get("construction_failure_json")
    if not isinstance(encoded, str):
        raise TypeError("Tanaka construction failure JSON must be a string")
    failure = _strict_json_text(
        encoded,
        context="Tanaka construction_failure_json",
    )
    if (
        failure.get("schema") != TANAKA_POTENTIAL_RADICAND_FAILURE_SCHEMA
        or failure.get("reason") != reason
        or failure.get("index_space") != "original_proposal_local_index"
    ):
        raise ValueError("Tanaka construction failure identity is inconsistent")
    classified = classify_tanaka_construction_failure(
        TanakaPotentialRadicandError(failure)
    )
    if classified is None:
        raise ValueError(
            "Tanaka construction rejection is not recognized by the current "
            "finite-domain failure classifier"
        )
    invalid = failure.get("invalid_case_indices")
    if (
        not isinstance(invalid, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in invalid
        )
        or tuple(sorted(set(invalid))) != tuple(invalid)
        or local_index not in invalid
    ):
        raise ValueError("Tanaka construction failure has invalid case indices")
    components = failure.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("Tanaka construction failure has no component records")
    matching = tuple(
        component
        for component in components
        if isinstance(component, Mapping)
        and component.get("local_case_index") == local_index
    )
    if not matching:
        raise ValueError("Tanaka construction failure omits the rejected case")
    seen: set[int] = set()
    for component in matching:
        crest_index = component.get("component_within_case")
        if (
            isinstance(crest_index, bool)
            or not isinstance(crest_index, int)
            or not 0 <= crest_index < len(proposed.sample.crests)
            or crest_index in seen
        ):
            raise ValueError("Tanaka construction failure names an invalid crest")
        seen.add(crest_index)
        crest = proposed.sample.crests[crest_index]
        for name, expected in (
            ("alpha", crest.alpha),
            ("center", crest.center),
            ("depth", proposed.depth),
        ):
            if (
                _finite_number(
                    component.get(name),
                    context=f"Tanaka failure component {name}",
                )
                != expected
            ):
                raise ValueError(
                    "Tanaka construction failure differs from its proposal"
                )
        if component.get("direction") != crest.direction:
            raise ValueError("Tanaka construction failure changed crest direction")
        negative_count = component.get("negative_count")
        nonfinite_count = component.get("nonfinite_count")
        minimum = component.get("minimum_radicand")
        if (
            isinstance(negative_count, bool)
            or not isinstance(negative_count, int)
            or negative_count <= 0
            or nonfinite_count != 0
            or _finite_number(
                minimum,
                context="Tanaka failure minimum_radicand",
            )
            >= 0.0
        ):
            raise ValueError(
                "Tanaka committed construction rejection is not a finite "
                "negative-radicand failure"
            )


def _validate_rollout_metrics(
    metrics: Mapping[str, object],
    *,
    accepted: bool,
    required_bits: int,
    evaluated_bits: int,
    failed_bits: int,
    residual_tolerance: float,
    production_dt: float,
    saved_time_count: int,
    residual_extrema: Extrema,
) -> None:
    """Require one result to follow the current single-arm Tanaka gates."""

    if set(metrics) != EXPECTED_ROLLOUT_METRIC_KEYS:
        raise ValueError("Tanaka rollout result has an unknown metric schema")
    if metrics.get("accepted") is not accepted:
        raise ValueError("result metric accepted flag disagrees with the case")
    if required_bits != EXPECTED_REQUIRED_BITS:
        raise ValueError("Tanaka result does not use the current required mask")
    state_finite = metrics.get("state_finite")
    target_finite = metrics.get("target_finite")
    all_stages = metrics.get("all_stages_solved")
    complete = metrics.get("complete_admissible_trajectory")
    if any(
        not isinstance(value, bool)
        for value in (state_finite, target_finite, all_stages, complete)
    ):
        raise TypeError("Tanaka rollout health flags must be Boolean")
    expected_evaluated = QualityReason(
        QualityReason.OUTSIDE_SUPPORT
        | QualityReason.INCOMPLETE_TRAJECTORY
        | QualityReason.GL2_STAGE_RESIDUAL
        | QualityReason.NONFINITE_STATE
        | QualityReason.NONFINITE_TARGET
    )
    if state_finite:
        expected_evaluated |= QualityReason.BOTTOM_CLEARANCE
    if evaluated_bits != int(expected_evaluated):
        raise ValueError("Tanaka result does not use the current evaluated mask")
    expected_failed = QualityReason.NONE
    if not all_stages:
        expected_failed |= QualityReason.GL2_STAGE_RESIDUAL
    if not state_finite:
        expected_failed |= QualityReason.NONFINITE_STATE
    if not target_finite:
        expected_failed |= QualityReason.NONFINITE_TARGET
    positive_water = metrics.get("positive_water_column")
    if state_finite:
        if not isinstance(positive_water, bool):
            raise TypeError("evaluated Tanaka water-column flag must be Boolean")
        if not positive_water:
            expected_failed |= QualityReason.BOTTOM_CLEARANCE
    elif positive_water is not None:
        raise ValueError("unevaluated Tanaka water-column flag must be null")
    if not complete or expected_failed != QualityReason.NONE:
        expected_failed |= QualityReason.INCOMPLETE_TRAJECTORY
    if failed_bits != int(expected_failed):
        raise ValueError("Tanaka failed mask disagrees with its health metrics")
    if accepted and evaluated_bits != EXPECTED_ACCEPTED_EVALUATED_BITS:
        raise ValueError("accepted Tanaka case has the wrong evaluated mask")
    if accepted and not (
        state_finite and target_finite and all_stages and complete and positive_water
    ):
        raise ValueError("accepted Tanaka case has a failed health metric")
    if (
        _finite_number(
            metrics.get("production_dt"),
            context="metrics.production_dt",
        )
        != production_dt
    ):
        raise ValueError("Tanaka production_dt differs from its contract")
    if (
        _finite_number(
            metrics.get("intended_terminal_time"),
            context="metrics.intended_terminal_time",
        )
        != EXPECTED_TERMINAL_TIME
    ):
        raise ValueError("Tanaka intended terminal time is not 200")
    if (
        _finite_number(
            metrics.get("realized_terminal_time"),
            context="metrics.realized_terminal_time",
        )
        != EXPECTED_TERMINAL_TIME
    ):
        raise ValueError("Tanaka realized terminal time is not 200")
    if metrics.get("saved_time_count") != saved_time_count:
        raise ValueError("Tanaka dense saved-time count differs from its contract")
    if metrics.get("internal_health_evaluated") is not False or any(
        metrics.get(name) is not None
        for name in (
            "internal_state_finite",
            "internal_dno_finite",
            "minimum_internal_water_column",
            "initial_internal_hamiltonian",
            "maximum_internal_hamiltonian_drift",
            "internal_hamiltonian_drift_threshold",
        )
    ):
        raise ValueError("Tanaka result invents undeclared internal-health gates")
    residual_value = metrics.get("maximum_stage_residual")
    residual = (
        None
        if residual_value is None
        else _finite_number(residual_value, context="maximum_stage_residual")
    )
    if accepted:
        if residual is None or not 0.0 <= residual <= residual_tolerance:
            raise ValueError("accepted Tanaka case exceeds the GL2 residual tolerance")
        residual_extrema.add(residual)


def _validate_case_record(
    value: object,
    *,
    proposed: ProposedCase,
    local_index: int,
    batch_id: int,
    residual_tolerance: float,
    production_dt: float,
    saved_time_count: int,
    residual_extrema: Extrema,
    rejection_reasons: Counter[str],
) -> AuditedCase:
    if not isinstance(value, Mapping):
        raise TypeError("every Tanaka result case must be a JSON object")
    accepted = value.get("accepted")
    if not isinstance(accepted, bool):
        raise TypeError("Tanaka result case accepted must be Boolean")
    case_id = _required_integer(value, "case_id", context="result case")
    if case_id != proposed.case_id:
        raise ValueError("Tanaka result case differs from its proposal identity")
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
        raise ValueError("Tanaka accepted flag disagrees with quality masks")
    first_row = _required_integer(value, "first_row", context="result case")
    row_count = _required_integer(
        value,
        "row_count",
        context="result case",
        minimum=0,
    )
    if accepted and row_count != ROWS_PER_ACCEPTED_CASE:
        raise ValueError("accepted Tanaka case does not own exactly 200 rows")
    if not accepted and (first_row != -1 or row_count != 0):
        raise ValueError("rejected Tanaka case owns stored rows")
    metrics = _required_mapping(value, "metrics", context="result case")
    construction_rejected = bool(failed & int(QualityReason.OUTSIDE_SUPPORT))
    if construction_rejected:
        if (
            accepted
            or required != EXPECTED_REQUIRED_BITS
            or (
                evaluated != int(QualityReason.OUTSIDE_SUPPORT)
                or failed != int(QualityReason.OUTSIDE_SUPPORT)
            )
        ):
            raise ValueError("Tanaka construction rejection has wrong quality masks")
        _validate_construction_rejection(
            metrics,
            proposed=proposed,
            local_index=local_index,
        )
    else:
        _validate_rollout_metrics(
            metrics,
            accepted=accepted,
            required_bits=required,
            evaluated_bits=evaluated,
            failed_bits=failed,
            residual_tolerance=residual_tolerance,
            production_dt=production_dt,
            saved_time_count=saved_time_count,
            residual_extrema=residual_extrema,
        )
    if not accepted:
        rejection_reasons[
            "+".join(reason.name for reason in reasons_from_bits(failed))
        ] += 1
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


def _validate_shard(
    path: Path,
    *,
    proposal_sha256: str,
    configuration_fingerprint: str,
    proposed: Sequence[ProposedCase],
    cases: Sequence[AuditedCase],
    delivered_nx: int,
    dense_last_index: int,
    saved_dt: float,
    totals: AuditTotals,
) -> int:
    """Validate all accepted row blocks and their physical ownership."""

    accepted_indices = tuple(index for index, case in enumerate(cases) if case.accepted)
    if not accepted_indices:
        if path.exists():
            raise ValueError("all-rejected Tanaka batch unexpectedly has a shard")
        return 0
    if not path.is_file():
        raise FileNotFoundError(f"accepted Tanaka batch has no shard: {path}")
    _scan_finite_archive(path, totals=totals)
    with np.load(path, allow_pickle=False) as archive:
        if frozenset(archive.files) != EXPECTED_SHARD_ARRAYS:
            raise ValueError(f"{path} has an unknown shard schema")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    for name, dtype in EXPECTED_SHARD_DTYPES.items():
        if arrays[name].dtype != dtype:
            raise TypeError(f"Tanaka shard {name} has the wrong dtype")
    for name in ("config_fingerprint", "proposal_sha256"):
        if arrays[name].ndim != 0 or arrays[name].dtype.kind not in {"U", "S"}:
            raise TypeError(f"Tanaka shard {name} must be a scalar string")
    if arrays["eta"].shape != (
        len(accepted_indices) * ROWS_PER_ACCEPTED_CASE,
        delivered_nx,
    ):
        raise ValueError("Tanaka shard has the wrong field shape")
    if arrays["xi"].shape != arrays["eta"].shape or arrays["gxi"].shape != (
        arrays["eta"].shape
    ):
        raise ValueError("Tanaka shard fields have different shapes")
    row_count = int(arrays["eta"].shape[0])
    for name in (
        "depth",
        "time",
        "case_local_index",
        "frame_index",
        "selected_dense_index",
    ):
        if arrays[name].shape != (row_count,):
            raise ValueError(f"Tanaka shard {name} has the wrong shape")
    if str(arrays["config_fingerprint"].item()) != configuration_fingerprint:
        raise ValueError("Tanaka shard has the wrong configuration fingerprint")
    if str(arrays["proposal_sha256"].item()) != proposal_sha256:
        raise ValueError("Tanaka shard has the wrong proposal hash")

    expected_first = 0
    for local_index, (proposal, case) in enumerate(zip(proposed, cases)):
        selected = np.flatnonzero(arrays["case_local_index"] == local_index)
        if not case.accepted:
            if selected.size:
                raise ValueError("rejected Tanaka case appears in its shard")
            continue
        if selected.size != ROWS_PER_ACCEPTED_CASE:
            raise ValueError("accepted Tanaka case has the wrong shard row count")
        expected_positions = np.arange(
            expected_first,
            expected_first + ROWS_PER_ACCEPTED_CASE,
        )
        if not np.array_equal(selected, expected_positions):
            raise ValueError("Tanaka case rows do not form the expected block")
        if case.first_row != expected_first:
            raise ValueError("Tanaka result first_row differs from its shard")
        if not np.array_equal(
            arrays["frame_index"][selected],
            np.arange(ROWS_PER_ACCEPTED_CASE, dtype=np.int32),
        ):
            raise ValueError("Tanaka stored frame indices are not 0 through 199")
        dense_indices = arrays["selected_dense_index"][selected]
        if (
            int(dense_indices[0]) != 0
            or int(dense_indices[-1]) != dense_last_index
            or np.any(np.diff(dense_indices) <= 0)
        ):
            raise ValueError("Tanaka adaptive-time indices omit an endpoint")
        times = arrays["time"][selected]
        if (
            float(times[0]) != 0.0
            or float(times[-1]) != EXPECTED_TERMINAL_TIME
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("Tanaka stored trajectory is not complete through t=200")
        if not np.array_equal(
            times,
            saved_dt * dense_indices.astype(np.float64),
        ):
            raise ValueError("Tanaka stored times differ from their dense indices")
        depths = arrays["depth"][selected]
        if not np.all(depths == proposal.depth):
            raise ValueError("Tanaka stored depth differs from its proposal")
        if np.min(depths[:, None] + arrays["eta"][selected]) <= 0.0:
            raise ValueError("Tanaka stored trajectory crosses the bottom")
        expected_first += ROWS_PER_ACCEPTED_CASE
    if expected_first != row_count:
        raise RuntimeError("Tanaka shard row blocks do not cover the shard")
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
    """Validate the chunk's manifest and complete attempted-case row map."""

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
        raise ValueError("Tanaka dataset-view artifact escapes its chunk root")
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
            "length": TrajectoryExecutionConfig.paper("tanaka").numerical.length,
            "nx": TrajectoryExecutionConfig.paper("tanaka").numerical.delivered_nx,
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
                        "execution_record_fingerprint": (chunk.execution_fingerprint),
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
        raise ValueError("Tanaka dataset contract differs from current generation")
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
        raise ValueError("dataset manifest omits a committed Tanaka batch")
    expected_shard_count = sum(paths.shard.is_file() for paths in chunk.batches)
    if len(raw_shards) != expected_shard_count:
        raise ValueError("dataset manifest has the wrong Tanaka shard count")
    shard_record_by_batch_position: dict[int, tuple[int, Mapping[str, object]]] = {}
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
        proposed = proposed_by_batch[batch_id]
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
            "n_attempted_trajectories": len(proposed),
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
    _validate_map_array_schema(arrays)
    proposed_cases = tuple(
        proposed
        for paths in chunk.batches
        for proposed in proposed_by_batch[int(paths.result.stem.removeprefix("batch_"))]
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
    accepted_values = np.asarray(
        [case.accepted for case in cases],
        dtype=np.bool_,
    )
    row_counts = np.asarray(
        [ROWS_PER_ACCEPTED_CASE if case.accepted else 0 for case in cases],
        dtype=np.int32,
    )
    first_rows = np.cumsum(row_counts, dtype=np.int64) - row_counts
    first_rows[~accepted_values] = -1
    comparisons = (
        (
            "trajectory_case_id",
            np.asarray([case.case_id for case in cases], dtype=np.int64),
        ),
        (
            "trajectory_cell_id",
            np.asarray(
                [proposed.cell_code for proposed in proposed_cases],
                dtype=np.int32,
            ),
        ),
        ("trajectory_accepted", accepted_values),
        (
            "trajectory_required_bits",
            np.asarray([case.required_bits for case in cases], dtype=np.uint32),
        ),
        (
            "trajectory_evaluated_bits",
            np.asarray([case.evaluated_bits for case in cases], dtype=np.uint32),
        ),
        (
            "trajectory_failed_bits",
            np.asarray([case.failed_bits for case in cases], dtype=np.uint32),
        ),
        ("trajectory_row_count", row_counts),
        ("trajectory_first_row", first_rows),
    )
    for name, expected in comparisons:
        if not np.array_equal(arrays[name], expected):
            raise ValueError(f"trajectory map {name} differs from transactions")
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

    expected_rows = int(np.sum(row_counts, dtype=np.int64))
    row_fields = ("trajectory_index", "frame_index", "shard_index", "shard_row")
    if any(arrays[name].shape != (expected_rows,) for name in row_fields):
        raise ValueError("trajectory map has a malformed row vector")
    row_cursor = 0
    for trajectory_index, case in enumerate(cases):
        if not case.accepted:
            continue
        selected = slice(row_cursor, row_cursor + ROWS_PER_ACCEPTED_CASE)
        if not np.all(arrays["trajectory_index"][selected] == trajectory_index):
            raise ValueError("row-to-trajectory ownership is incorrect")
        if not np.array_equal(
            arrays["frame_index"][selected],
            np.arange(ROWS_PER_ACCEPTED_CASE, dtype=np.int32),
        ):
            raise ValueError("trajectory frame ownership is incorrect")
        shard_index = shard_index_by_batch_id.get(case.batch_id)
        if shard_index is None or not np.all(
            arrays["shard_index"][selected] == shard_index
        ):
            raise ValueError("row-to-shard ownership is incorrect")
        expected_shard_rows = case.first_row + np.arange(
            ROWS_PER_ACCEPTED_CASE,
            dtype=np.int64,
        )
        if not np.array_equal(arrays["shard_row"][selected], expected_shard_rows):
            raise ValueError("stored shard-row ownership is incorrect")
        row_cursor += ROWS_PER_ACCEPTED_CASE
    if row_cursor != expected_rows:
        raise RuntimeError("accepted Tanaka rows do not sum to the manifest")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "trajectory_map_path": str(map_path),
        "trajectory_map_sha256": map_sha256,
        "dataset_contract_fingerprint": manifest["dataset_contract_fingerprint"],
    }


def _cell_code_mapping(run_spec: Mapping[str, object]) -> dict[int, str]:
    raw_codes = _required_mapping(run_spec, "cell_codes", context="run_spec")
    expected_ids = TANAKA_SAMPLE_CELL_IDS
    if tuple(raw_codes) != expected_ids:
        raise ValueError("Tanaka run cell order is not the current eleven cells")
    mapping = {
        _required_integer(
            raw_codes,
            cell_id,
            context="run_spec.cell_codes",
        ): cell_id
        for cell_id in expected_ids
    }
    if mapping != {index: cell_id for index, cell_id in enumerate(expected_ids)}:
        raise ValueError("Tanaka run uses noncanonical cell codes")
    return mapping


def _expected_chunk_quotas(expected: ExpectedChunk) -> dict[str, int]:
    cell_ids = TANAKA_SAMPLE_CELL_IDS
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
        raise ValueError("Tanaka summary must contain the exact eleven-cell taxonomy")

    expected_records: dict[str, dict[str, int]] = {}
    for cell_id, target in expected_quotas.items():
        attempted = int(attempted_by_cell.get(cell_id, 0))
        accepted = int(accepted_by_cell.get(cell_id, 0))
        rejected = attempted - accepted
        if accepted != target or rejected < 0:
            raise ValueError("Tanaka reconstructed cell counts do not close")
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
                f"Tanaka summary cell {cell_id} must contain the exact count fields"
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
            raise ValueError(f"Tanaka summary cell {cell_id} counts do not close")
        if observed != expected_record:
            raise ValueError(
                f"Tanaka summary cell {cell_id} differs from reconstructed transactions"
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
                f"Tanaka summary {field} total differs from reconstructed transactions"
            )


def _audit_chunk(
    expected: ExpectedChunk,
    chunk: CompletedChunk,
    *,
    totals: AuditTotals,
    support_extrema: Mapping[str, Extrema],
    residual_extrema: Extrema,
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
    production_dt = _finite_number(
        numerical.get("dt"),
        context="execution.numerical.dt",
    )
    saved_dt = _finite_number(
        numerical.get("saved_dt"),
        context="execution.numerical.saved_dt",
    )
    saved_time_count = int(round(EXPECTED_TERMINAL_TIME / saved_dt)) + 1
    if (
        residual_tolerance != 1.0e-8
        or production_dt != 0.01
        or not math.isclose(
            (saved_time_count - 1) * saved_dt,
            EXPECTED_TERMINAL_TIME,
            rel_tol=0.0,
            abs_tol=1.0e-13,
        )
    ):
        raise ValueError("Tanaka numerical constants differ from release contract")
    cell_ids_by_code = _cell_code_mapping(run_spec)
    expected_quotas = _expected_chunk_quotas(expected)
    raw_quotas = _required_list(run_spec, "quotas", context="run_spec")
    observed_quotas = {
        _required_string(quota, "cell_id", context="run_spec quota"): _required_integer(
            quota,
            "target_accepted",
            context="run_spec quota",
            minimum=0,
        )
        for quota in raw_quotas
        if isinstance(quota, Mapping)
    }
    if len(observed_quotas) != len(raw_quotas) or observed_quotas != expected_quotas:
        raise ValueError("Tanaka chunk does not have exact balanced cell quotas")

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
        proposals = _proposal_cases(
            paths.proposal,
            split=chunk.split,
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
            raise ValueError("Tanaka result has the wrong configuration fingerprint")
        proposal_sha256 = file_sha256(paths.proposal)
        if result.get("proposal_sha256") != proposal_sha256:
            raise ValueError("Tanaka result has the wrong proposal hash")
        expected_shard_sha256 = (
            file_sha256(paths.shard) if paths.shard.is_file() else None
        )
        if result.get("shard_sha256") != expected_shard_sha256:
            raise ValueError("Tanaka result has the wrong shard hash")
        values = _required_list(result, "cases", context=str(paths.result))
        if len(values) != len(proposals):
            raise ValueError("Tanaka result omits proposed cases")
        batch_cases = tuple(
            _validate_case_record(
                value,
                proposed=proposal,
                local_index=local_index,
                batch_id=batch_id,
                residual_tolerance=residual_tolerance,
                production_dt=production_dt,
                saved_time_count=saved_time_count,
                residual_extrema=residual_extrema,
                rejection_reasons=rejection_reasons,
            )
            for local_index, (value, proposal) in enumerate(zip(values, proposals))
        )
        shard_rows = _validate_shard(
            paths.shard,
            proposal_sha256=proposal_sha256,
            configuration_fingerprint=chunk.fingerprint,
            proposed=proposals,
            cases=batch_cases,
            delivered_nx=int(numerical["nx"]),
            dense_last_index=saved_time_count - 1,
            saved_dt=saved_dt,
            totals=totals,
        )
        if shard_rows != sum(case.row_count for case in batch_cases):
            raise ValueError("Tanaka shard row count differs from its results")
        proposed_by_batch[batch_id] = proposals
        cases_by_batch[batch_id] = batch_cases
        result_by_batch[batch_id] = result
        attempted_by_cell.update(proposal.cell_id for proposal in proposals)
        accepted_by_cell.update(
            proposal.cell_id
            for proposal, case in zip(proposals, batch_cases)
            if case.accepted
        )

    cases = tuple(
        case
        for paths in chunk.batches
        for case in cases_by_batch[int(paths.result.stem.removeprefix("batch_"))]
    )
    attempted = len(cases)
    accepted = sum(case.accepted for case in cases)
    retained_rows = sum(case.row_count for case in cases)
    if (
        attempted != chunk.attempted_count
        or accepted != chunk.accepted_count
        or retained_rows != accepted * ROWS_PER_ACCEPTED_CASE
    ):
        raise ValueError("Tanaka transaction counts disagree with completed chunk")
    if dict(accepted_by_cell) != {
        cell_id: count for cell_id, count in expected_quotas.items() if count
    }:
        raise ValueError("Tanaka accepted cases do not meet exact cell quotas")
    if set(attempted_by_cell) - set(expected_quotas):
        raise ValueError("Tanaka attempts contain an unknown cell")
    _validate_summary_cell_counts(
        summary,
        expected_quotas=expected_quotas,
        attempted_by_cell=attempted_by_cell,
        accepted_by_cell=accepted_by_cell,
    )
    totals.attempted += attempted
    totals.accepted += accepted
    totals.rejected += attempted - accepted
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
        "rejected_count": attempted - accepted,
        "retained_rows": retained_rows,
        "committed_batches": len(chunk.batches),
        "accepted_by_cell": dict(accepted_by_cell),
        "attempted_by_cell": dict(attempted_by_cell),
        "configuration_fingerprint": chunk.fingerprint,
        "summary_path": str(chunk.summary_path),
        "summary_sha256": file_sha256(chunk.summary_path),
        **view_record,
    }


def _load_exact_chunks(root: Path) -> tuple[CompletedChunk, ...]:
    """Load only the immutable four-train-plus-validation/test plan."""

    expected_paths = {
        (root / expected.summary_path).resolve() for expected in EXPECTED_CHUNKS
    }
    discovered_paths = {
        path.resolve()
        for path in root.glob("*/tanaka/*/paper_dataset_tanaka_*.summary.json")
    }
    if discovered_paths != expected_paths:
        missing = sorted(map(str, expected_paths - discovered_paths))
        unexpected = sorted(map(str, discovered_paths - expected_paths))
        raise ValueError(
            "Tanaka completion summaries differ from the exact six-chunk "
            f"plan; missing={missing}, unexpected={unexpected}"
        )
    chunks = tuple(
        # Deliberately omit the historical compatibility policy.  This fresh
        # dataset must validate directly against the current execution contract.
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
            "tanaka",
            EXPECTED_REVISION_ID,
            expected.split,
            expected.stream_id,
            expected.accepted_before,
            expected.accepted_count,
            expected.accepted_before + expected.accepted_count,
        )
        if observed != wanted:
            raise ValueError(f"{expected.label} differs from its immutable plan")
        if chunk.generation_compatibility_id is not None:
            raise ValueError(
                "fresh Tanaka dataset resolved through a historical compatibility "
                "variant"
            )
        if chunk.execution_platform != "gpu":
            raise ValueError("fresh Tanaka release dataset was not generated on GPU")
    return chunks


def _identity_record(chunks: Sequence[CompletedChunk]) -> dict[str, object]:
    """Require exact current dependency, source, execution, and support identity."""

    dependency_fingerprints = {chunk.dependency_fingerprint for chunk in chunks}
    execution_fingerprints = {chunk.execution_fingerprint for chunk in chunks}
    source_fingerprints = {chunk.source_fingerprint for chunk in chunks}
    if len(dependency_fingerprints) != 1:
        raise ValueError("Tanaka chunks use different dependency environments")
    if len(execution_fingerprints) != 1:
        raise ValueError("Tanaka chunks use different execution contracts")
    if len(source_fingerprints) != 1:
        raise ValueError("Tanaka chunks use different generation source mappings")

    current_execution = TrajectoryExecutionConfig.paper("tanaka")
    current_execution_record = current_execution.to_json_record()
    current_execution_fingerprint = canonical_json_sha256(current_execution_record)
    execution_fingerprint = next(iter(execution_fingerprints))
    if execution_fingerprint != current_execution_fingerprint:
        raise ValueError("Tanaka chunks do not use the current execution contract")
    current_dependency = dependency_environment()
    current_dependency_fingerprint = canonical_json_sha256(current_dependency)
    dependency_fingerprint = next(iter(dependency_fingerprints))
    if dependency_fingerprint != current_dependency_fingerprint:
        raise ValueError("Tanaka chunks do not use the current dependency identity")
    current_sources = source_hashes("tanaka")
    current_source_fingerprint = canonical_json_sha256(current_sources)
    source_fingerprint = next(iter(source_fingerprints))
    if source_fingerprint != current_source_fingerprint:
        raise ValueError("Tanaka chunks do not use the complete current source map")
    for chunk in chunks:
        if dict(chunk.source_sha256) != current_sources:
            raise ValueError("Tanaka chunk source mapping differs from current code")
        summary = _strict_json_object(chunk.summary_path)
        run_spec = _required_mapping(summary, "run_spec", context="summary")
        if (
            run_spec.get("batch_size") != EXPECTED_BATCH_SIZE
            or run_spec.get("first_attempt_index") != 0
            or run_spec.get("maximum_attempts_per_accepted_case")
            != EXPECTED_MAXIMUM_ATTEMPTS_PER_ACCEPTED_CASE
        ):
            raise ValueError("Tanaka chunk differs from the exact quota-run policy")
        configuration = _required_mapping(
            run_spec,
            "configuration",
            context="summary.run_spec",
        )
        declared_dependency = _required_mapping(
            configuration,
            "dependency_environment",
            context="summary.run_spec.configuration",
        )
        declared_sources = _required_mapping(
            configuration,
            "source_sha256",
            context="summary.run_spec.configuration",
        )
        if not _same_json(declared_dependency, current_dependency):
            raise ValueError("Tanaka chunk declares a noncurrent dependency record")
        if not _same_json(declared_sources, current_sources):
            raise ValueError("Tanaka chunk declares a noncurrent source record")

    support = _support_record()
    amplitude_inversion = _amplitude_inversion_record(current_execution)
    return {
        "dependency_environment_fingerprint": dependency_fingerprint,
        "execution_record_fingerprint": execution_fingerprint,
        "source_sha256_fingerprint": source_fingerprint,
        "sampling_support_fingerprint": canonical_json_sha256(support),
        "sampling_support": support,
        "corrected_amplitude_inversion_fingerprint": canonical_json_sha256(
            amplitude_inversion
        ),
        "corrected_amplitude_inversion": amplitude_inversion,
        "historical_generation_compatibility_used": False,
    }


def _accepted_cell_totals_by_split(
    chunk_records: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, int]]:
    """Require the final accepted population to balance all eleven cells."""

    cell_ids = TANAKA_SAMPLE_CELL_IDS
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
                raise TypeError("chunk accepted_by_cell must be an object")
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
                f"Tanaka {split.value} population is not balanced over 11 cells"
            )
        output[split.value] = dict(observed)
    return output


def audit(root: Path) -> dict[str, object]:
    """Run the complete read-only fresh Tanaka dataset audit."""

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
            "total_alpha",
            "crest_alpha",
            "physical_crest_amplitude",
            "resolution_ratio",
            "minimum_separation",
        )
    }
    residual_extrema = Extrema()
    rejection_reasons: Counter[str] = Counter()
    chunk_records = tuple(
        _audit_chunk(
            expected,
            chunk,
            totals=totals,
            support_extrema=support_extrema,
            residual_extrema=residual_extrema,
            rejection_reasons=rejection_reasons,
        )
        for expected, chunk in zip(EXPECTED_CHUNKS, chunks)
    )
    accepted_by_split = {
        split.value: sum(
            int(record["accepted_count"])
            for record in chunk_records
            if record["split"] == split.value
        )
        for split in SplitId
    }
    if accepted_by_split != {
        "train": EXPECTED_TRAIN_ACCEPTED,
        "validation": EXPECTED_VALIDATION_ACCEPTED,
        "test": EXPECTED_TEST_ACCEPTED,
    }:
        raise ValueError("Tanaka split totals differ from the paper dataset plan")
    if totals.accepted != EXPECTED_ACCEPTED:
        raise ValueError("Tanaka accepted total differs from 18,432")
    if totals.proposal_specs_checked != totals.attempted:
        raise ValueError("not every Tanaka proposal specification was replayed")
    if totals.requested_crests_checked < totals.attempted:
        raise ValueError("not every Tanaka proposal had a requested crest checked")
    if totals.retained_rows != EXPECTED_ACCEPTED * ROWS_PER_ACCEPTED_CASE:
        raise ValueError("Tanaka retained-row total is incorrect")
    if len(TANAKA_SAMPLE_CELLS) != 11:
        raise ValueError("current Tanaka support no longer has eleven cells")
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
        "requested_crests_checked": totals.requested_crests_checked,
        "finite_float_arrays_checked": totals.float_arrays_checked,
        "finite_float_values_checked": totals.float_values_checked,
        "transaction_artifact_bytes_checked": (
            totals.transaction_artifact_bytes_checked
        ),
        "dataset_view_bytes_checked": totals.dataset_view_bytes_checked,
        "accepted_by_split": accepted_by_split,
        "accepted_by_split_and_cell": accepted_cells,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "support": {
            "cell_count": len(TANAKA_SAMPLE_CELLS),
            "direction_aware_cells": [
                {
                    "cell_id": cell_id,
                    "regime": regime,
                    "crest_count": crest_count,
                    "right_moving_count": right_moving_count,
                }
                for cell_id, (regime, crest_count, right_moving_count) in (
                    TANAKA_SAMPLE_CELLS.items()
                )
            ],
            "extrema": {
                name: extrema.record() for name, extrema in support_extrema.items()
            },
        },
        "numerical_extrema": {
            "accepted_maximum_stage_residual": residual_extrema.record(),
        },
        "identity": identity,
        "chunks": list(chunk_records),
        "checks": {
            "exact_six_chunk_plan": True,
            "exact_four_train_intervals_plus_validation_and_test": True,
            "balanced_eleven_direction_aware_cell_quotas": True,
            "summary_cell_counts_match_reconstructed_transactions": True,
            "immutable_specs_rebuilt": True,
            "all_requested_crests_replayed_exactly": True,
            "corrected_amplitude_inversion_identity_verified": True,
            "constructor_amplitude_success_postcondition_source_bound": True,
            "below_old_floor_amplitude_behavior_verified": True,
            "solver_amplitude_validator_called_exactly_once": True,
            "amplitude_validator_deliberate_mismatch_rejected": True,
            "achieved_per_crest_amplitudes_persisted": False,
            "all_transactions_rescanned": True,
            "proposal_hashes_verified": True,
            "result_hashes_verified": True,
            "shard_hashes_verified": True,
            "manifest_hashes_verified": True,
            "trajectory_map_hashes_verified": True,
            "accepted_rejected_row_ownership_verified": True,
            "accepted_cases_have_exactly_200_rows": True,
            "rejected_cases_have_zero_rows": True,
            "stored_eta_xi_gxi_depth_time_finite": True,
            "stored_water_columns_positive": True,
            "all_quality_masks_current_and_consistent": True,
            "accepted_residuals_at_most_1e-8": True,
            "accepted_trajectories_complete_through_t200": True,
            "all_proposals_in_current_tanaka_support": True,
            "source_dependency_execution_support_identity_verified": True,
            "historical_generation_compatibility_used": False,
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
        help="Completed fresh corrected revision-3 Tanaka dataset root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Audit artifact; defaults to <root>/tanaka_completion_audit.json.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "tanaka_completion_audit.json"
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
