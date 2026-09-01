"""Preflight or generate one paper-dataset family.

The default mode is a read-only preflight.  Numerical generation begins only
when ``--execute`` is supplied. Existing completed batches are reused; an
interrupted batch is run again from the beginning.
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

from solver.gen_data.benjamin_feir_sampling import (  # noqa: E402
    BENJAMIN_FEIR_PARAMETER_GROUP_IDS,
    BENJAMIN_FEIR_PARAMETER_GROUPS,
    PAPER_FOCUSED_STEEPNESS_LIMIT,
    PAPER_PERTURBATION_RATIO_MAX,
)
from solver.gen_data.benjamin_feir_jcp09 import (  # noqa: E402
    CARRIER_STEEPNESS_MAX,
    CARRIER_STEEPNESS_MIN,
    PERTURBATION_RATIO_MIN,
)
from solver.gen_data.jonswap_tma_sampling import (  # noqa: E402
    JONSWAP_TMA_PARAMETER_GROUP_IDS,
)
from solver.gen_data.pipeline.artifact_io import write_json_atomic  # noqa: E402
from solver.gen_data.pipeline.batch_storage import load_completed_batch  # noqa: E402
from solver.gen_data.pipeline.build_dataset_view import (  # noqa: E402
    DATASET_VIEW_SCHEMA_VERSION,
    DatasetViewPaths,
    build_dataset_view,
)
from solver.gen_data.pipeline.simulation_allocation import (  # noqa: E402
    ParameterGroupTarget,
    PhysicalFamilyId,
    DatasetSplit,
    balanced_simulation_targets,
)
from solver.gen_data.pipeline.simulation_checks import (  # noqa: E402
    checks_from_bits,
)
from solver.gen_data.pipeline.dataset_generation import (  # noqa: E402
    MAX_RETRIES_PER_PARAMETER_GROUP,
    DatasetChunkConfig,
    DatasetChunkState,
    generate_simulations,
    scan_dataset_generation,
)
from solver.gen_data.stokes_sampling import (  # noqa: E402
    DEFAULT_MAXIMUM_URSELL_REDRAWS,
    STOKES_PARAMETER_GROUP_IDS,
)
from solver.gen_data.stokes_batch_executor import (  # noqa: E402
    make_static_stokes_batch_executor,
)
from solver.gen_data.stokes_static_pipeline import (  # noqa: E402
    PAPER_STATIC_STOKES_CONTRACT,
    StaticStokesContract,
)
from solver.gen_data.tanaka_sampling import (  # noqa: E402
    TANAKA_PARAMETER_GROUP_IDS,
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
FAMILY_PARAMETER_GROUP_IDS: dict[PaperFamily, tuple[str, ...]] = {
    "stokes": STOKES_PARAMETER_GROUP_IDS,
    "tanaka": TANAKA_PARAMETER_GROUP_IDS,
    "benjamin_feir": BENJAMIN_FEIR_PARAMETER_GROUP_IDS,
    "jonswap_tma": JONSWAP_TMA_PARAMETER_GROUP_IDS,
}


@dataclass(frozen=True)
class GenerationRequest:
    """User-visible identity and allocation choices for one dataset chunk."""

    output_root: Path
    family: PaperFamily
    split: DatasetSplit
    accepted_simulations: int
    batch_size: int
    accepted_simulations_before: int = 0
    platform: Platform = "cpu"
    worker_stream_id: int = 0
    first_attempt_index: int = 0

    def __post_init__(self) -> None:
        if self.family not in FAMILY_IDS:
            raise ValueError(f"unknown paper-dataset family: {self.family}")
        if not isinstance(self.split, DatasetSplit):
            raise TypeError("split must be a DatasetSplit")
        for value, name, positive in (
            (self.accepted_simulations, "accepted_simulations", True),
            (self.batch_size, "batch_size", True),
            (
                self.accepted_simulations_before,
                "accepted_simulations_before",
                False,
            ),
            (self.worker_stream_id, "worker_stream_id", False),
            (self.first_attempt_index, "first_attempt_index", False),
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
    state: DatasetChunkState
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
        choices=tuple(split.value for split in DatasetSplit),
        required=True,
        help="Simulation-level dataset split.",
    )
    parser.add_argument(
        "--accepted-simulations",
        type=_positive_integer,
        required=True,
        help="Number of valid simulations to generate across parameter groups.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root containing completed batches and dataset views.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_integer,
        required=True,
        help="Maximum attempted simulations in one transaction.",
    )
    parser.add_argument(
        "--accepted-simulations-before",
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
        "--worker-stream-id",
        type=_nonnegative_integer,
        default=0,
        help="Deterministic ID and RNG stream assigned to this worker.",
    )
    parser.add_argument(
        "--first-attempt-index",
        type=_nonnegative_integer,
        default=0,
        help="First attempt index assigned to this worker stream.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Run numerical generation after the preflight scan.",
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
        split=DatasetSplit(args.split),
        accepted_simulations=args.accepted_simulations,
        batch_size=args.batch_size,
        accepted_simulations_before=args.accepted_simulations_before,
        platform=args.platform,
        worker_stream_id=args.worker_stream_id,
        first_attempt_index=args.first_attempt_index,
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
                "parameter_group_id": parameter_group_id,
                "carrier_mode": carrier_mode,
                "sideband_offset": sideband_offset,
            }
            for parameter_group_id, (
                carrier_mode,
                sideband_offset,
            ) in BENJAMIN_FEIR_PARAMETER_GROUPS.items()
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


def incremental_simulation_targets(
    parameter_group_ids: Sequence[str],
    *,
    accepted_simulations_before: int,
    simulation_count: int,
) -> tuple[ParameterGroupTarget, ...]:
    """Return this chunk's balanced valid-simulation targets."""

    before = balanced_simulation_targets(
        parameter_group_ids,
        simulation_count=accepted_simulations_before,
    )
    after = balanced_simulation_targets(
        parameter_group_ids,
        simulation_count=accepted_simulations_before + simulation_count,
    )
    targets = tuple(
        ParameterGroupTarget(
            parameter_group_id=after_target.parameter_group_id,
            simulation_count=(
                after_target.simulation_count - before_target.simulation_count
            ),
        )
        for before_target, after_target in zip(before, after)
    )
    if any(target.simulation_count < 0 for target in targets):
        raise RuntimeError("balanced cumulative targets must be monotone")
    if sum(target.simulation_count for target in targets) != simulation_count:
        raise RuntimeError("valid-simulation targets do not sum to the chunk size")
    return targets


def build_chunk_config(
    request: GenerationRequest,
    *,
    execution: PaperExecution | None = None,
) -> DatasetChunkConfig:
    """Build the immutable configuration for the requested dataset chunk."""

    selected_execution = execution or _paper_execution(request.family)
    if selected_execution != _paper_execution(request.family):
        raise ValueError("the paper-dataset launcher requires the exact contract")
    parameter_group_ids = FAMILY_PARAMETER_GROUP_IDS[request.family]
    simulation_targets = incremental_simulation_targets(
        parameter_group_ids,
        accepted_simulations_before=request.accepted_simulations_before,
        simulation_count=request.accepted_simulations,
    )
    common_configuration: dict[str, object] = {
        "schema": "paper_dataset_quota_configuration_v1",
        "purpose": "exact paper-contract accepted-simulation generation",
        "simulation_type": ("static" if request.family == "stokes" else "trajectory"),
        "accepted_simulation_count": request.accepted_simulations,
        "accepted_simulations_before": request.accepted_simulations_before,
        "accepted_simulations_after": (
            request.accepted_simulations_before + request.accepted_simulations
        ),
        "ordered_parameter_group_ids": list(parameter_group_ids),
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
    return DatasetChunkConfig(
        root=request.output_root,
        family_name=request.family,
        family_id=FAMILY_IDS[request.family],
        dataset_split=request.split,
        worker_stream_id=request.worker_stream_id,
        simulation_targets=simulation_targets,
        batch_size=request.batch_size,
        first_attempt_index=request.first_attempt_index,
        configuration={
            **common_configuration,
            **family_configuration,
        },
    )


def _stored_rows_per_simulation(execution: PaperExecution) -> int:
    if isinstance(execution, StaticStokesContract):
        return 1
    return {
        "tanaka": execution.frame_selection.tanaka_count,
        "benjamin_feir": execution.frame_selection.benjamin_feir_count,
        "jonswap_tma": execution.frame_selection.jonswap_tma_count,
    }[execution.family]


def _state_record(state: DatasetChunkState) -> dict[str, object]:
    return {
        "complete": state.complete,
        "accepted_simulation_counts": dict(state.accepted_simulation_counts),
        "simulation_attempt_counts": dict(state.simulation_attempt_counts),
        "completed_batches": len(state.completed_batches),
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
    DatasetChunkConfig,
    PaperExecution,
    DatasetChunkState,
    dict[str, object],
]:
    """Perform a read-only compatibility scan and return its complete plan."""

    execution = _paper_execution(request.family)
    chunk_config = build_chunk_config(request, execution=execution)
    if request.family == "jonswap_tma":
        assert isinstance(execution, TrajectoryExecutionConfig)
        try:
            TrajectoryBatchExecutor(
                chunk_config=chunk_config,
                execution=execution,
            )
        except ValueError as error:
            raise RuntimeError(
                "paper-dataset JONSWAP/TMA must use "
                "scripts/generate_paper_dataset_jonswap.py so the "
                "nonlinear adjustment cannot be bypassed"
            ) from error
    state = scan_dataset_generation(chunk_config)
    cumulative_before = balanced_simulation_targets(
        FAMILY_PARAMETER_GROUP_IDS[request.family],
        simulation_count=request.accepted_simulations_before,
    )
    cumulative_after = balanced_simulation_targets(
        FAMILY_PARAMETER_GROUP_IDS[request.family],
        simulation_count=(
            request.accepted_simulations_before + request.accepted_simulations
        ),
    )
    maximum_attempts = chunk_config.maximum_attempts_by_parameter_group
    simulation_targets = [
        {
            "parameter_group_id": target.parameter_group_id,
            "accepted_before": before.simulation_count,
            "chunk_target_accepted": target.simulation_count,
            "accepted_after": after.simulation_count,
            "maximum_attempts": maximum_attempts[target.parameter_group_id],
            "attempted": state.simulation_attempt_counts[target.parameter_group_id],
            "remaining_attempts": (
                maximum_attempts[target.parameter_group_id]
                - state.simulation_attempt_counts[target.parameter_group_id]
            ),
        }
        for before, target, after in zip(
            cumulative_before,
            chunk_config.simulation_targets,
            cumulative_after,
        )
    ]
    nonzero_cells = sum(
        target.simulation_count > 0 for target in chunk_config.simulation_targets
    )
    stored_rows_per_simulation = _stored_rows_per_simulation(execution)
    plan: dict[str, object] = {
        "schema": "paper_dataset_quota_preflight_v1",
        "mode": "dry_run",
        "no_numerical_generation_performed": True,
        "output_root": str(request.output_root),
        "artifact_namespace": {
            "family": request.family,
            "split": request.split.value,
            "view_name": _view_name(request),
            "summary_path": str(_summary_path(request)),
        },
        "run_spec": chunk_config.to_json_record(),
        "execution": _execution_record(execution),
        "allocation": {
            "chunk_accepted_simulations": request.accepted_simulations,
            "accepted_simulations_before": request.accepted_simulations_before,
            "accepted_simulations_after": (
                request.accepted_simulations_before + request.accepted_simulations
            ),
            "cell_count": len(chunk_config.simulation_targets),
            "nonzero_quota_cell_count": nonzero_cells,
            "quota_minimum": min(
                target.simulation_count for target in chunk_config.simulation_targets
            ),
            "quota_maximum": max(
                target.simulation_count for target in chunk_config.simulation_targets
            ),
            "maximum_retries_per_parameter_group": (MAX_RETRIES_PER_PARAMETER_GROUP),
            "quotas": simulation_targets,
            "chunk_identity": {
                "worker_stream_id": request.worker_stream_id,
                "first_attempt_index": request.first_attempt_index,
                "output_root": str(request.output_root),
                "rule": (
                    "use a distinct worker_stream_id and output_root for every additive chunk"
                ),
            },
        },
        "expected_output": {
            "stored_rows_per_accepted_simulation": stored_rows_per_simulation,
            "retained_rows": request.accepted_simulations * stored_rows_per_simulation,
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
        "generation_state": _state_record(state),
        "runtime": _runtime_record(),
    }
    return chunk_config, execution, state, plan


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


def _summarize_completed_simulations(
    state: DatasetChunkState,
    chunk_config: DatasetChunkConfig,
) -> dict[str, object]:
    attempted = Counter(
        {target.parameter_group_id: 0 for target in chunk_config.simulation_targets}
    )
    accepted = Counter(
        {target.parameter_group_id: 0 for target in chunk_config.simulation_targets}
    )
    rejection_reasons: Counter[str] = Counter()

    for path in state.completed_batches:
        batch = load_completed_batch(path)
        parameter_groups = tuple(
            str(value) for value in batch.plan["parameter_group_id"]
        )
        for parameter_group_id, simulation in zip(
            parameter_groups,
            batch.simulations,
            strict=True,
        ):
            attempted[parameter_group_id] += 1
            if simulation.accepted:
                accepted[parameter_group_id] += 1
                continue
            names = tuple(
                reason.name or str(reason.value)
                for reason in checks_from_bits(simulation.failed_bits)
            )
            rejection_reasons["+".join(names) if names else "missing_check"] += 1

    if dict(accepted) != dict(state.accepted_simulation_counts):
        raise RuntimeError("summary counts disagree with generation state")
    if dict(attempted) != dict(state.simulation_attempt_counts):
        raise RuntimeError("summary attempts disagree with generation state")
    by_cell = {
        target.parameter_group_id: {
            "target_accepted": target.simulation_count,
            "attempted": attempted[target.parameter_group_id],
            "accepted": accepted[target.parameter_group_id],
            "rejected": attempted[target.parameter_group_id]
            - accepted[target.parameter_group_id],
        }
        for target in chunk_config.simulation_targets
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
    attempted_simulations: int,
) -> dict[str, object]:
    manifest = _read_json_object(view.manifest)
    expected_rows = request.accepted_simulations * _stored_rows_per_simulation(
        execution
    )
    expected_values = {
        "schema_version": DATASET_VIEW_SCHEMA_VERSION,
        "n_rows": expected_rows,
        "n_trajectories": attempted_simulations,
        "n_accepted_trajectories": request.accepted_simulations,
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
    """Generate the requested valid simulations, then validate the dataset view."""

    chunk_config, execution, initial_state, plan = preflight(request)
    runtime = _require_runtime(request)
    invocation_started_at = datetime.now().astimezone()
    total_started = perf_counter()
    metadata = {
        "launcher": "scripts/generate_paper_dataset.py",
        "run_spec": chunk_config.to_json_record(),
    }
    if isinstance(execution, StaticStokesContract):
        executor = make_static_stokes_batch_executor(
            chunk_config=chunk_config,
            contract=execution,
            maximum_ursell_redraws=DEFAULT_MAXIMUM_URSELL_REDRAWS,
            metadata=metadata,
        )
        length = execution.length
    else:
        executor = TrajectoryBatchExecutor(
            chunk_config=chunk_config,
            execution=execution,
            metadata=metadata,
        )
        length = execution.numerical.length
    generation_started = perf_counter()
    state = generate_simulations(chunk_config, executor)
    generation_seconds = perf_counter() - generation_started
    if not state.complete:
        raise RuntimeError(
            "generation stopped before every valid-simulation target was met"
        )

    view_started = perf_counter()
    view = build_dataset_view(
        request.output_root,
        state.completed_batches,
        name=_view_name(request),
        length=length,
    )
    view_seconds = perf_counter() - view_started
    validation_started = perf_counter()
    counts = _summarize_completed_simulations(state, chunk_config)
    attempted_simulations = counts["attempted"]
    if isinstance(attempted_simulations, bool) or not isinstance(
        attempted_simulations, int
    ):
        raise TypeError("attempted simulation count must be an integer")
    view_record = _validate_view(
        view,
        request=request,
        execution=execution,
        attempted_simulations=attempted_simulations,
    )
    validation_seconds = perf_counter() - validation_started

    summary: dict[str, object] = {
        "schema": "paper_dataset_quota_summary_v1",
        "status": "complete",
        "output_root": str(request.output_root),
        "run_spec": chunk_config.to_json_record(),
        "execution": _execution_record(execution),
        "runtime": runtime,
        "preflight": plan,
        "generation_state": {
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
