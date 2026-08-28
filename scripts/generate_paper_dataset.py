"""Preflight, generate, or resume one paper-dataset family.

The default mode is a read-only preflight.  Numerical generation begins only
when ``--execute`` is supplied. Existing batches are resumed from their saved
case assignments and validated numerical outputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Literal, Sequence, TypeAlias


Platform: TypeAlias = Literal["cpu", "gpu"]


def _bootstrap_platform(argv: Sequence[str]) -> Platform:
    """Read only ``--platform`` before importing modules that initialize JAX."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parsed, _ = parser.parse_known_args(argv)
    platform = parsed.platform
    if platform not in ("cpu", "gpu"):
        raise ValueError(f"unsupported platform: {platform}")
    return platform


BOOTSTRAP_PLATFORM = _bootstrap_platform(sys.argv[1:])
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["DNO_TANAKA_DTYPE"] = "float64"
os.environ["JAX_PLATFORMS"] = "cpu" if BOOTSTRAP_PLATFORM == "cpu" else "cuda"
if BOOTSTRAP_PLATFORM == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


import jax  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.benjamin_feir_sampling import (  # noqa: E402
    BENJAMIN_FEIR_SAMPLE_CELL_IDS,
    BENJAMIN_FEIR_SAMPLE_CELLS,
    PAPER_FOCUSED_STEEPNESS_LIMIT,
    PAPER_PERTURBATION_RATIO_MAX,
)
from solver.gen_data.benjamin_feir_jcp09 import (  # noqa: E402
    CARRIER_STEEPNESS_MAX,
    CARRIER_STEEPNESS_MIN,
    PERTURBATION_RATIO_MIN,
)
from solver.gen_data.jonswap_tma_sampling import (  # noqa: E402
    JONSWAP_TMA_SAMPLE_CELL_IDS,
)
from solver.gen_data.pipeline.artifact_io import write_json_atomic  # noqa: E402
from solver.gen_data.pipeline.build_dataset_view import (  # noqa: E402
    DATASET_VIEW_SCHEMA_VERSION,
    DatasetViewPaths,
    build_dataset_view,
)
from solver.gen_data.pipeline.case_allocation import (  # noqa: E402
    SampleCellTarget,
    PhysicalFamilyId,
    SplitId,
    balanced_valid_case_targets,
    DATASET_REVISION_BY_FAMILY,
)
from solver.gen_data.pipeline.case_checks import (  # noqa: E402
    checks_from_bits,
)
from solver.gen_data.pipeline.valid_case_generation import (  # noqa: E402
    DEFAULT_MAXIMUM_ATTEMPTS_PER_VALID_CASE,
    DatasetGenerationSpec,
    DatasetGenerationState,
    generate_valid_cases,
    scan_dataset_generation,
)
from solver.gen_data.stokes_sampling import (  # noqa: E402
    DEFAULT_MAXIMUM_URSELL_REDRAWS,
    STOKES_SAMPLE_CELL_IDS,
)
from solver.gen_data.stokes_batch_executor import (  # noqa: E402
    make_static_stokes_batch_executor,
)
from solver.gen_data.stokes_static_pipeline import (  # noqa: E402
    PAPER_STATIC_STOKES_CONTRACT,
    StaticStokesContract,
)
from solver.gen_data.tanaka_sampling import (  # noqa: E402
    TANAKA_SAMPLE_CELL_IDS,
)
from solver.gen_data.trajectory_batch_executor import (  # noqa: E402
    TrajectoryExecutionConfig,
    TrajectoryBatchExecutor,
    paper_trajectory_execution,
)


jax.config.update("jax_enable_x64", True)


PaperFamily: TypeAlias = Literal[
    "stokes",
    "tanaka",
    "benjamin_feir",
    "jonswap_tma",
]
PaperExecution: TypeAlias = StaticStokesContract | TrajectoryExecutionConfig
FAMILY_IDS: dict[PaperFamily, PhysicalFamilyId] = {
    "stokes": PhysicalFamilyId.STOKES,
    "tanaka": PhysicalFamilyId.TANAKA,
    "benjamin_feir": PhysicalFamilyId.BENJAMIN_FEIR,
    "jonswap_tma": PhysicalFamilyId.JONSWAP_TMA,
}
FAMILY_CELL_IDS: dict[PaperFamily, tuple[str, ...]] = {
    "stokes": STOKES_SAMPLE_CELL_IDS,
    "tanaka": TANAKA_SAMPLE_CELL_IDS,
    "benjamin_feir": BENJAMIN_FEIR_SAMPLE_CELL_IDS,
    "jonswap_tma": JONSWAP_TMA_SAMPLE_CELL_IDS,
}


@dataclass(frozen=True)
class GenerationRequest:
    """User-visible identity and allocation choices for one generation run."""

    output_root: Path
    family: PaperFamily
    split: SplitId
    accepted_cases: int
    batch_size: int
    accepted_cases_before: int = 0
    platform: Platform = "cpu"
    stream_id: int = 0
    first_attempt_index: int = 0
    maximum_attempts_per_accepted_case: int = DEFAULT_MAXIMUM_ATTEMPTS_PER_VALID_CASE

    def __post_init__(self) -> None:
        if self.family not in FAMILY_IDS:
            raise ValueError(f"unknown paper-dataset family: {self.family}")
        if not isinstance(self.split, SplitId):
            raise TypeError("split must be a SplitId")
        for value, name, positive in (
            (self.accepted_cases, "accepted_cases", True),
            (self.batch_size, "batch_size", True),
            (self.accepted_cases_before, "accepted_cases_before", False),
            (self.stream_id, "stream_id", False),
            (self.first_attempt_index, "first_attempt_index", False),
            (
                self.maximum_attempts_per_accepted_case,
                "maximum_attempts_per_accepted_case",
                True,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if (positive and value <= 0) or (not positive and value < 0):
                qualifier = "positive" if positive else "nonnegative"
                raise ValueError(f"{name} must be {qualifier}")
        if self.platform not in ("cpu", "gpu"):
            raise ValueError(f"unsupported platform: {self.platform}")
        object.__setattr__(
            self,
            "output_root",
            Path(self.output_root).expanduser().resolve(),
        )


@dataclass(frozen=True)
class GenerationRunResult:
    """Products of one completed exact-contract invocation."""

    summary_path: Path
    view: DatasetViewPaths
    state: DatasetGenerationState
    summary: dict[str, object]


def _positive_integer(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _nonnegative_integer(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=tuple(FAMILY_IDS),
        required=True,
        help="Physical paper-dataset family to generate.",
    )
    parser.add_argument(
        "--split",
        choices=tuple(split.value for split in SplitId),
        required=True,
        help="Case-level dataset split.",
    )
    parser.add_argument(
        "--accepted-cases",
        type=_positive_integer,
        required=True,
        help="Number of valid cases to generate, balanced across sampling cells.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root containing saved proposals, shards, results, and views.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_integer,
        required=True,
        help="Maximum attempted cases in one transaction.",
    )
    parser.add_argument(
        "--accepted-cases-before",
        type=_nonnegative_integer,
        default=0,
        help=(
            "Cumulative accepted count already assigned to earlier chunks; "
            "defaults to zero for a standalone pilot."
        ),
    )
    parser.add_argument(
        "--platform",
        choices=("cpu", "gpu"),
        default="cpu",
        help="JAX execution platform; defaults to the fail-safe CPU path.",
    )
    parser.add_argument(
        "--stream-id",
        type=_nonnegative_integer,
        default=0,
        help="Independent deterministic stream within this family and split.",
    )
    parser.add_argument(
        "--first-attempt-index",
        type=_nonnegative_integer,
        default=0,
        help="First attempt coordinate in the selected deterministic stream.",
    )
    parser.add_argument(
        "--maximum-attempts-per-accepted-case",
        type=_positive_integer,
        default=DEFAULT_MAXIMUM_ATTEMPTS_PER_VALID_CASE,
        help=(
            "Per-cell attempted-case ceiling multiplier; a cell with target "
            "Q stops the run after this value times Q recorded attempts "
            f"(default: {DEFAULT_MAXIMUM_ATTEMPTS_PER_VALID_CASE})."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Run or resume numerical generation after the preflight scan.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the read-only preflight explicitly (this is the default).",
    )
    return parser.parse_args(argv)


def request_from_args(args: argparse.Namespace) -> GenerationRequest:
    return GenerationRequest(
        output_root=args.output_root,
        family=args.family,
        split=SplitId(args.split),
        accepted_cases=args.accepted_cases,
        batch_size=args.batch_size,
        accepted_cases_before=args.accepted_cases_before,
        platform=args.platform,
        stream_id=args.stream_id,
        first_attempt_index=args.first_attempt_index,
        maximum_attempts_per_accepted_case=(args.maximum_attempts_per_accepted_case),
    )


def _paper_execution(family: PaperFamily) -> PaperExecution:
    if family == "stokes":
        return PAPER_STATIC_STOKES_CONTRACT
    return paper_trajectory_execution(family)


def _execution_record(execution: PaperExecution) -> dict[str, object]:
    return execution.to_json_record()


def _benjamin_feir_sampling_support_record() -> dict[str, object]:
    """Return an elementary, complete description of the BF sampling law."""

    return {
        "schema": "paper_benjamin_feir_sampling_support_v1",
        "mode_pair_cells": [
            {
                "cell_id": cell_id,
                "carrier_mode": carrier_mode,
                "sideband_offset": sideband_offset,
            }
            for cell_id, (
                carrier_mode,
                sideband_offset,
            ) in BENJAMIN_FEIR_SAMPLE_CELLS.items()
        ],
        "conditional_carrier_steepness": {
            "law": "uniform",
            "interval": "(lower, upper]",
            "lower_formula": "max(0.05, Delta_n/(2*sqrt(2)*n_c))",
            "upper_formula": "min(0.13, (-F+2*sqrt(F^2+3*ell^2))/3)",
            "ell_definition": "Delta_n/(2*sqrt(2)*n_c)",
            "base_lower": CARRIER_STEEPNESS_MIN,
            "base_upper": CARRIER_STEEPNESS_MAX,
        },
        "focused_steepness": {
            "formula": "epsilon_c*(1+2*sqrt(1-beta^2))",
            "beta_definition": "Delta_n/(2*sqrt(2)*epsilon_c*n_c)",
            "upper_inclusive": PAPER_FOCUSED_STEEPNESS_LIMIT,
        },
        "sideband_to_carrier_ratio": {
            "law": "uniform",
            "lower_inclusive": PERTURBATION_RATIO_MIN,
            "upper_inclusive": PAPER_PERTURBATION_RATIO_MAX,
        },
        "translation": {
            "law": "uniform",
            "interval": "[0, L)",
        },
    }


def incremental_valid_case_targets(
    cell_ids: Sequence[str],
    *,
    accepted_cases_before: int,
    case_count: int,
) -> tuple[SampleCellTarget, ...]:
    """Return this chunk's balanced valid-case targets."""

    before = balanced_valid_case_targets(
        cell_ids,
        case_count=accepted_cases_before,
    )
    after = balanced_valid_case_targets(
        cell_ids,
        case_count=accepted_cases_before + case_count,
    )
    targets = tuple(
        SampleCellTarget(
            cell_id=after_target.cell_id,
            case_count=(after_target.case_count - before_target.case_count),
        )
        for before_target, after_target in zip(before, after)
    )
    if any(target.case_count < 0 for target in targets):
        raise RuntimeError("balanced cumulative targets must be monotone")
    if sum(target.case_count for target in targets) != case_count:
        raise RuntimeError("valid-case targets do not sum to the chunk size")
    return targets


def build_run_spec(
    request: GenerationRequest,
    *,
    execution: PaperExecution | None = None,
) -> DatasetGenerationSpec:
    """Build the immutable exact-contract allocation for ``request``."""

    selected_execution = execution or _paper_execution(request.family)
    if selected_execution != _paper_execution(request.family):
        raise ValueError("the paper-dataset launcher requires the exact contract")
    cells = FAMILY_CELL_IDS[request.family]
    case_targets = incremental_valid_case_targets(
        cells,
        accepted_cases_before=request.accepted_cases_before,
        case_count=request.accepted_cases,
    )
    common_configuration: dict[str, object] = {
        "schema": "paper_dataset_quota_configuration_v1",
        "purpose": "exact paper-contract accepted-case generation",
        "case_kind": "static" if request.family == "stokes" else "trajectory",
        "accepted_case_count": request.accepted_cases,
        "accepted_cases_before": request.accepted_cases_before,
        "accepted_cases_after": (
            request.accepted_cases_before + request.accepted_cases
        ),
        "ordered_cell_ids": list(cells),
        "execution_platform": request.platform,
    }
    if request.family == "stokes":
        assert isinstance(selected_execution, StaticStokesContract)
        family_configuration = {
            "contract": selected_execution.to_json_record(),
            "sampler": {
                "maximum_ursell_redraws": DEFAULT_MAXIMUM_URSELL_REDRAWS,
            },
        }
    else:
        assert isinstance(selected_execution, TrajectoryExecutionConfig)
        family_configuration = {
            "trajectory_execution": selected_execution.to_json_record(),
        }
        if request.family == "benjamin_feir":
            family_configuration["sampling_support"] = (
                _benjamin_feir_sampling_support_record()
            )
    return DatasetGenerationSpec(
        root=request.output_root,
        family_name=request.family,
        family_id=FAMILY_IDS[request.family],
        revision_id=DATASET_REVISION_BY_FAMILY[FAMILY_IDS[request.family]],
        split_id=request.split,
        stream_id=request.stream_id,
        case_targets=case_targets,
        cell_codes={cell_id: cell_code for cell_code, cell_id in enumerate(cells)},
        batch_size=request.batch_size,
        first_attempt_index=request.first_attempt_index,
        maximum_attempts_per_accepted_case=(request.maximum_attempts_per_accepted_case),
        configuration={
            **common_configuration,
            **family_configuration,
        },
    )


def _stored_rows_per_case(execution: PaperExecution) -> int:
    if isinstance(execution, StaticStokesContract):
        return 1
    return {
        "tanaka": execution.stored_time_policy.tanaka_count,
        "benjamin_feir": execution.stored_time_policy.benjamin_feir_count,
        "jonswap_tma": execution.stored_time_policy.random_sea_count,
    }[execution.family]


def _state_record(state: DatasetGenerationState) -> dict[str, object]:
    return {
        "complete": state.complete,
        "accepted_by_cell": dict(state.accepted_by_cell),
        "attempted_by_cell": dict(state.attempted_by_cell),
        "committed_batches": len(state.committed),
        "pending_batch_id": (
            state.pending.batch_id if state.pending is not None else None
        ),
        "pending_status": (
            state.pending.status.value if state.pending is not None else None
        ),
        "terminal_failure": (
            str(state.terminal_failure.failure)
            if state.terminal_failure is not None
            else None
        ),
        "attempt_limit_exhausted_cells": (
            list(state.attempt_limit_failure.exhausted_cells)
            if state.attempt_limit_failure is not None
            else []
        ),
        "attempt_limit_failure": (
            state.attempt_limit_failure.message
            if state.attempt_limit_failure is not None
            else None
        ),
        "next_batch_id": state.next_batch_id,
        "next_attempt_index": state.next_attempt_index,
    }


def _runtime_record() -> dict[str, object]:
    devices = jax.devices()
    return {
        "requested_platform": BOOTSTRAP_PLATFORM,
        "jax_version": jax.__version__,
        "default_backend": jax.default_backend(),
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "devices": [
            {
                "id": int(device.id),
                "platform": device.platform,
                "device_kind": device.device_kind,
            }
            for device in devices
        ],
    }


def _require_runtime(request: GenerationRequest) -> dict[str, object]:
    if request.platform != BOOTSTRAP_PLATFORM:
        raise RuntimeError(
            "the requested platform differs from the platform selected before "
            "JAX initialization; invoke this script in a fresh process"
        )
    runtime = _runtime_record()
    expected_backend = request.platform
    if not runtime["x64_enabled"]:
        raise RuntimeError("paper generation requires JAX float64 mode")
    if runtime["default_backend"] != expected_backend:
        raise RuntimeError(
            f"requested {request.platform}, but JAX initialized "
            f"{runtime['default_backend']}"
        )
    devices = runtime["devices"]
    if not isinstance(devices, list) or not devices:
        raise RuntimeError("JAX reported no execution devices")
    return runtime


def preflight(
    request: GenerationRequest,
) -> tuple[
    DatasetGenerationSpec,
    PaperExecution,
    DatasetGenerationState,
    dict[str, object],
]:
    """Perform a read-only compatibility scan and return its complete plan."""

    execution = _paper_execution(request.family)
    spec = build_run_spec(request, execution=execution)
    if request.family == "jonswap_tma":
        assert isinstance(execution, TrajectoryExecutionConfig)
        try:
            TrajectoryBatchExecutor(
                run_spec=spec,
                execution=execution,
            )
        except ValueError as error:
            raise RuntimeError(
                "paper-dataset JONSWAP/TMA must use "
                "scripts/generate_paper_dataset_jonswap.py so the "
                "nonlinear adjustment cannot be bypassed"
            ) from error
    state = scan_dataset_generation(spec)
    cumulative_before = balanced_valid_case_targets(
        FAMILY_CELL_IDS[request.family],
        case_count=request.accepted_cases_before,
    )
    cumulative_after = balanced_valid_case_targets(
        FAMILY_CELL_IDS[request.family],
        case_count=(request.accepted_cases_before + request.accepted_cases),
    )
    attempt_ceilings = spec.attempt_ceiling_by_cell
    case_targets = [
        {
            "cell_id": target.cell_id,
            "accepted_before": before.case_count,
            "chunk_target_accepted": target.case_count,
            "accepted_after": after.case_count,
            "attempt_ceiling": attempt_ceilings[target.cell_id],
            "durable_attempted": state.attempted_by_cell[target.cell_id],
            "remaining_attempt_capacity": (
                attempt_ceilings[target.cell_id]
                - state.attempted_by_cell[target.cell_id]
            ),
        }
        for before, target, after in zip(
            cumulative_before,
            spec.case_targets,
            cumulative_after,
        )
    ]
    nonzero_cells = sum(target.case_count > 0 for target in spec.case_targets)
    stored_rows_per_case = _stored_rows_per_case(execution)
    plan: dict[str, object] = {
        "schema": "paper_dataset_quota_preflight_v1",
        "mode": "dry_run",
        "no_numerical_generation_performed": True,
        "revision_id": spec.revision_id,
        "output_root": str(request.output_root),
        "artifact_namespace": {
            "family": request.family,
            "split": request.split.value,
            "view_name": _view_name(request),
            "summary_path": str(_summary_path(request)),
        },
        "run_spec": spec.to_json_record(),
        "execution": _execution_record(execution),
        "allocation": {
            "chunk_accepted_cases": request.accepted_cases,
            "accepted_cases_before": request.accepted_cases_before,
            "accepted_cases_after": (
                request.accepted_cases_before + request.accepted_cases
            ),
            "cell_count": len(spec.case_targets),
            "nonzero_quota_cell_count": nonzero_cells,
            "quota_minimum": min(target.case_count for target in spec.case_targets),
            "quota_maximum": max(target.case_count for target in spec.case_targets),
            "maximum_attempts_per_accepted_case": (
                spec.maximum_attempts_per_accepted_case
            ),
            "quotas": case_targets,
            "chunk_identity": {
                "stream_id": request.stream_id,
                "first_attempt_index": request.first_attempt_index,
                "output_root": str(request.output_root),
                "rule": (
                    "use a distinct stream_id and output_root for every additive chunk"
                ),
            },
        },
        "expected_output": {
            "stored_rows_per_accepted_case": stored_rows_per_case,
            "retained_rows": request.accepted_cases * stored_rows_per_case,
            "spatial_points_per_row": (
                execution.nx
                if isinstance(execution, StaticStokesContract)
                else execution.numerical.target_nx
            ),
            "field_values_per_row": 3
            * (
                execution.nx
                if isinstance(execution, StaticStokesContract)
                else execution.numerical.target_nx
            ),
        },
        "resume_state": _state_record(state),
        "runtime": _runtime_record(),
    }
    return spec, execution, state, plan


def _view_name(request: GenerationRequest) -> str:
    return f"paper_dataset_{request.family}_{request.split.value}"


def _summary_path(request: GenerationRequest) -> Path:
    return request.output_root / f"{_view_name(request)}.summary.json"


def _read_json_object(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{path} contains nonfinite JSON constant {value!r}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _artifact_record(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "bytes": path.stat().st_size,
    }


def _summarize_committed_cases(
    state: DatasetGenerationState,
    spec: DatasetGenerationSpec,
) -> dict[str, object]:
    code_to_cell = {code: cell for cell, code in spec.cell_codes.items()}
    attempted = Counter({target.cell_id: 0 for target in spec.case_targets})
    accepted = Counter({target.cell_id: 0 for target in spec.case_targets})
    rejection_reasons: Counter[str] = Counter()

    for paths in state.committed:
        with np.load(paths.proposal, allow_pickle=False) as proposal:
            encoded_cells = np.asarray(proposal["cell_id"], dtype=np.int32)
        result = _read_json_object(paths.result)
        cases = result.get("cases")
        if not isinstance(cases, list) or len(cases) != encoded_cells.size:
            raise RuntimeError("result does not contain every proposed case")
        for local_index, case in enumerate(cases):
            if not isinstance(case, dict):
                raise TypeError("result case must be a JSON object")
            cell_id = code_to_cell[int(encoded_cells[local_index])]
            attempted[cell_id] += 1
            if bool(case.get("accepted")):
                accepted[cell_id] += 1
                continue
            failed_bits = case.get("failed_bits")
            if isinstance(failed_bits, bool) or not isinstance(failed_bits, int):
                raise TypeError("result failed_bits must be an integer")
            names = tuple(
                reason.name or str(reason.value)
                for reason in checks_from_bits(failed_bits)
            )
            rejection_reasons["+".join(names) if names else "missing_check"] += 1

    if dict(accepted) != dict(state.accepted_by_cell):
        raise RuntimeError("summary counts disagree with generation state")
    if dict(attempted) != dict(state.attempted_by_cell):
        raise RuntimeError("summary attempts disagree with generation state")
    by_cell = {
        target.cell_id: {
            "target_accepted": target.case_count,
            "attempted": attempted[target.cell_id],
            "accepted": accepted[target.cell_id],
            "rejected": attempted[target.cell_id] - accepted[target.cell_id],
        }
        for target in spec.case_targets
    }
    attempted_total = sum(attempted.values())
    accepted_total = sum(accepted.values())
    return {
        "attempted": attempted_total,
        "accepted": accepted_total,
        "rejected": attempted_total - accepted_total,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "by_cell": by_cell,
    }


def _validate_view(
    view: DatasetViewPaths,
    *,
    request: GenerationRequest,
    execution: PaperExecution,
    attempted_cases: int,
) -> dict[str, object]:
    manifest = _read_json_object(view.manifest)
    expected_rows = request.accepted_cases * _stored_rows_per_case(execution)
    expected_values = {
        "schema_version": DATASET_VIEW_SCHEMA_VERSION,
        "n_rows": expected_rows,
        "n_trajectories": attempted_cases,
        "n_accepted_trajectories": request.accepted_cases,
        "n_accepted_rows": expected_rows,
    }
    for name, expected in expected_values.items():
        if manifest.get(name) != expected:
            raise RuntimeError(
                f"dataset view {name} is {manifest.get(name)!r}, expected {expected!r}"
            )
    grid = manifest.get("grid")
    if not isinstance(grid, dict):
        raise TypeError("dataset view grid must be a JSON object")
    expected_grid = (
        {
            "length": execution.length,
            "nx": execution.nx,
        }
        if isinstance(execution, StaticStokesContract)
        else {
            "length": execution.numerical.length,
            "nx": execution.numerical.target_nx,
        }
    )
    if grid != expected_grid:
        raise RuntimeError(f"dataset view grid is {grid!r}, expected {expected_grid!r}")
    return {
        **expected_values,
        "grid": expected_grid,
        "manifest": _artifact_record(view.manifest, root=request.output_root),
        "trajectory_map": _artifact_record(
            view.trajectory_map,
            root=request.output_root,
        ),
    }


def run_generation(request: GenerationRequest) -> GenerationRunResult:
    """Generate the requested valid cases, then validate the dataset view."""

    spec, execution, initial_state, plan = preflight(request)
    runtime = _require_runtime(request)
    if initial_state.terminal_failure is not None:
        raise RuntimeError(
            f"generation already terminated at {initial_state.terminal_failure.failure}"
        )
    if initial_state.attempt_limit_failure is not None:
        raise RuntimeError(initial_state.attempt_limit_failure.message)

    invocation_started_at = datetime.now().astimezone()
    total_started = perf_counter()
    metadata = {
        "launcher": "scripts/generate_paper_dataset.py",
        "run_spec": spec.to_json_record(),
    }
    if isinstance(execution, StaticStokesContract):
        executor = make_static_stokes_batch_executor(
            run_spec=spec,
            contract=execution,
            maximum_ursell_redraws=DEFAULT_MAXIMUM_URSELL_REDRAWS,
            metadata=metadata,
        )
        length = execution.length
    else:
        executor = TrajectoryBatchExecutor(
            run_spec=spec,
            execution=execution,
            metadata=metadata,
        )
        length = execution.numerical.length
    generation_started = perf_counter()
    state = generate_valid_cases(spec, executor)
    generation_seconds = perf_counter() - generation_started
    if state.terminal_failure is not None:
        raise RuntimeError(f"generation terminated at {state.terminal_failure.failure}")
    if state.attempt_limit_failure is not None:
        raise RuntimeError(state.attempt_limit_failure.message)
    if not state.complete:
        raise RuntimeError("generation stopped before every valid-case target was met")

    view_started = perf_counter()
    view = build_dataset_view(
        request.output_root,
        state.committed,
        name=_view_name(request),
        length=length,
    )
    view_seconds = perf_counter() - view_started
    validation_started = perf_counter()
    counts = _summarize_committed_cases(state, spec)
    attempted_cases = counts["attempted"]
    if isinstance(attempted_cases, bool) or not isinstance(attempted_cases, int):
        raise TypeError("attempted case count must be an integer")
    view_record = _validate_view(
        view,
        request=request,
        execution=execution,
        attempted_cases=attempted_cases,
    )
    validation_seconds = perf_counter() - validation_started

    summary: dict[str, object] = {
        "schema": "paper_dataset_quota_summary_v1",
        "status": "complete",
        "output_root": str(request.output_root),
        "run_spec": spec.to_json_record(),
        "execution": _execution_record(execution),
        "runtime": runtime,
        "preflight": plan,
        "resume": {
            "initial": _state_record(initial_state),
            "final": _state_record(state),
        },
        "counts": counts,
        "dataset_view": view_record,
        "timing_seconds": {
            "quota_driver": generation_seconds,
            "dataset_view": view_seconds,
            "validation": validation_seconds,
            "total": perf_counter() - total_started,
        },
        "invocation_started_at": invocation_started_at.isoformat(),
        "invocation_finished_at": datetime.now().astimezone().isoformat(),
    }
    summary_path = _summary_path(request)
    write_json_atomic(summary_path, summary)
    return GenerationRunResult(
        summary_path=summary_path,
        view=view,
        state=state,
        summary=summary,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    request = request_from_args(args)
    if args.execute:
        result = run_generation(request)
        output: dict[str, object] = {
            "mode": "execute",
            "status": result.summary["status"],
            "summary_path": str(result.summary_path),
            "counts": result.summary["counts"],
            "dataset_view": result.summary["dataset_view"],
            "timing_seconds": result.summary["timing_seconds"],
        }
    else:
        _, _, _, output = preflight(request)
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
