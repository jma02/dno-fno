"""Generate one family and split of the paper dataset.

Without ``--execute``, the script only prints the requested run and any
completed progress. Existing completed batches are reused; an interrupted
batch is run again from the beginning.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Literal, Sequence, TypeAlias


ComputePlatform: TypeAlias = Literal["cpu", "gpu"]

_bootstrap_parser = argparse.ArgumentParser(add_help=False)
_bootstrap_parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
_bootstrap_args, _ = _bootstrap_parser.parse_known_args(sys.argv[1:])
BOOTSTRAP_PLATFORM: ComputePlatform = _bootstrap_args.platform
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["DNO_TANAKA_DTYPE"] = "float64"
os.environ["JAX_PLATFORMS"] = "cpu" if BOOTSTRAP_PLATFORM == "cpu" else "cuda"
if BOOTSTRAP_PLATFORM == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


import jax  # noqa: E402

from solver.gen_data.benjamin_feir_sampling import (  # noqa: E402
    BENJAMIN_FEIR_PARAMETER_GROUP_IDS,
)
from solver.gen_data.jonswap_tma_sampling import (  # noqa: E402
    JONSWAP_TMA_PARAMETER_GROUP_IDS,
)
from solver.gen_data.jonswap_horizon_generator import (  # noqa: E402
    HorizonBucketedJonswapBatchGenerator,
)
from solver.gen_data.pipeline.artifact_io import write_json_atomic  # noqa: E402
from solver.gen_data.pipeline.build_dataset_view import (  # noqa: E402
    build_dataset_view,
)
from solver.gen_data.pipeline.simulation_allocation import (  # noqa: E402
    PhysicalFamilyId,
    DatasetSplit,
    balanced_simulation_targets,
)
from solver.gen_data.pipeline.dataset_generation import (  # noqa: E402
    DatasetChunkConfig,
    generate_simulations,
    scan_dataset_generation,
)
from solver.gen_data.stokes_sampling import (  # noqa: E402
    DEFAULT_MAXIMUM_URSELL_REDRAWS,
    STOKES_PARAMETER_GROUP_IDS,
)
from solver.gen_data.stokes_batch_generator import (  # noqa: E402
    make_static_stokes_batch_generator,
)
from solver.gen_data.stokes_static_pipeline import (  # noqa: E402
    PAPER_STATIC_STOKES_CONTRACT,
    StaticStokesContract,
)
from solver.gen_data.tanaka_sampling import (  # noqa: E402
    TANAKA_PARAMETER_GROUP_IDS,
)
from solver.gen_data.pipeline.trajectory_config import (  # noqa: E402
    TrajectoryExecutionConfig,
    paper_trajectory_execution,
)
from solver.gen_data.trajectory_batch_generator import TrajectoryBatchGenerator  # noqa: E402


jax.config.update("jax_enable_x64", True)


PaperFamily: TypeAlias = Literal[
    "stokes",
    "tanaka",
    "benjamin_feir",
    "jonswap_tma",
]
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
    """Inputs for one family/split generation run."""

    output_root: Path
    family: PaperFamily
    split: DatasetSplit
    accepted_simulations: int
    batch_size: int
    platform: ComputePlatform = "cpu"
    solver_batch_size: int | None = None

    def __post_init__(self) -> None:
        if self.family not in FAMILY_IDS:
            raise ValueError(f"unknown paper-dataset family: {self.family}")
        if not isinstance(self.split, DatasetSplit):
            raise TypeError("split must be a DatasetSplit")
        for value, name in (
            (self.accepted_simulations, "accepted_simulations"),
            (self.batch_size, "batch_size"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.platform not in ("cpu", "gpu"):
            raise ValueError(f"unsupported platform: {self.platform}")
        if self.family == "jonswap_tma":
            if self.solver_batch_size is None:
                raise ValueError("JONSWAP/TMA requires solver_batch_size")
            if not 0 < self.solver_batch_size <= self.batch_size:
                raise ValueError("solver_batch_size must be between 1 and batch_size")
        elif self.solver_batch_size is not None:
            raise ValueError("solver_batch_size is only used for JONSWAP/TMA")
        object.__setattr__(
            self,
            "output_root",
            Path(self.output_root).expanduser().resolve(),
        )


def _positive_integer(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def build_chunk_config(request: GenerationRequest) -> DatasetChunkConfig:
    """Build the immutable configuration for the requested dataset chunk."""

    parameter_group_ids = FAMILY_PARAMETER_GROUP_IDS[request.family]
    simulation_targets = balanced_simulation_targets(
        parameter_group_ids,
        simulation_count=request.accepted_simulations,
    )
    return DatasetChunkConfig(
        root=request.output_root,
        family_name=request.family,
        family_id=FAMILY_IDS[request.family],
        dataset_split=request.split,
        simulation_targets=simulation_targets,
        batch_size=request.batch_size,
        solver_batch_size=request.solver_batch_size,
    )


def main(argv: Sequence[str] | None = None) -> None:
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
        help="Maximum attempted simulations in one batch.",
    )
    parser.add_argument(
        "--platform",
        choices=("cpu", "gpu"),
        default="cpu",
        help="JAX execution platform; defaults to the fail-safe CPU path.",
    )
    parser.add_argument(
        "--solver-batch-size",
        type=_positive_integer,
        help="JONSWAP simulations solved together after sorting by rollout length.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run numerical generation; without this flag, print the plan.",
    )
    args = parser.parse_args(argv)
    request = GenerationRequest(
        output_root=args.output_root,
        family=args.family,
        split=DatasetSplit(args.split),
        accepted_simulations=args.accepted_simulations,
        batch_size=args.batch_size,
        platform=args.platform,
        solver_batch_size=args.solver_batch_size,
    )
    chunk_config = build_chunk_config(request)
    if args.execute:
        execution = (
            PAPER_STATIC_STOKES_CONTRACT
            if request.family == "stokes"
            else paper_trajectory_execution(request.family)
        )
        if request.platform != BOOTSTRAP_PLATFORM:
            raise RuntimeError(
                "the requested platform differs from the platform selected before "
                "JAX initialization; invoke this script in a fresh process"
            )
        if not jax.config.read("jax_enable_x64"):
            raise RuntimeError("paper generation requires JAX float64 mode")
        backend = jax.default_backend()
        if backend != request.platform:
            raise RuntimeError(
                f"requested {request.platform}, but JAX initialized {backend}"
            )
        if not jax.devices():
            raise RuntimeError("JAX reported no execution devices")

        total_started = perf_counter()
        if isinstance(execution, StaticStokesContract):
            generator = make_static_stokes_batch_generator(
                chunk_config=chunk_config,
                contract=execution,
                maximum_ursell_redraws=DEFAULT_MAXIMUM_URSELL_REDRAWS,
            )
            length = execution.length
        elif request.family == "jonswap_tma":
            assert isinstance(execution, TrajectoryExecutionConfig)
            generator = HorizonBucketedJonswapBatchGenerator(
                chunk_config=chunk_config,
                execution=execution,
            )
            length = execution.numerical.length
        else:
            generator = TrajectoryBatchGenerator(
                chunk_config=chunk_config,
                execution=execution,
            )
            length = execution.numerical.length

        generation_started = perf_counter()
        state = generate_simulations(chunk_config, generator)
        generation_seconds = perf_counter() - generation_started
        view_name = f"paper_dataset_{request.family}_{request.split.value}"
        view_started = perf_counter()
        view = build_dataset_view(
            request.output_root,
            state.completed_batches,
            name=view_name,
            length=length,
        )
        view_seconds = perf_counter() - view_started
        attempted = state.simulation_attempt_counts
        accepted = state.accepted_simulation_counts
        attempted_total = sum(attempted.values())
        accepted_total = sum(accepted.values())
        counts = {
            "attempted": attempted_total,
            "accepted": accepted_total,
            "rejected": attempted_total - accepted_total,
            "by_parameter_group": {
                parameter_group_id: {
                    "target_accepted": target_count,
                    "attempted": attempted[parameter_group_id],
                    "accepted": accepted[parameter_group_id],
                    "rejected": (
                        attempted[parameter_group_id] - accepted[parameter_group_id]
                    ),
                }
                for parameter_group_id, target_count in (
                    chunk_config.simulation_targets.items()
                )
            },
        }
        view_record = {
            "manifest": str(view.manifest),
            "trajectory_map": str(view.trajectory_map),
        }
        summary: dict[str, object] = {
            "status": "complete",
            "output_root": str(request.output_root),
            "run_spec": chunk_config.to_json_record(),
            "batch_paths": [
                str(path.resolve().relative_to(request.output_root))
                for path in state.completed_batches
            ],
            "counts": counts,
            "dataset_view": view_record,
            "timing_seconds": {
                "generation": generation_seconds,
                "dataset_view": view_seconds,
                "total": perf_counter() - total_started,
            },
        }
        summary_path = request.output_root / f"{view_name}.summary.json"
        write_json_atomic(summary_path, summary)
        output: dict[str, object] = {
            "mode": "execute",
            "status": summary["status"],
            "summary_path": str(summary_path),
            "counts": summary["counts"],
            "dataset_view": summary["dataset_view"],
            "timing_seconds": summary["timing_seconds"],
        }
    else:
        state = scan_dataset_generation(chunk_config)
        output = {
            "mode": "dry_run",
            "run_spec": chunk_config.to_json_record(),
            "completed_batches": len(state.completed_batches),
            "accepted_simulation_counts": dict(state.accepted_simulation_counts),
        }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
