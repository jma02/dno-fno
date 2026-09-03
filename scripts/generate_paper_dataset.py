"""Generate one family and split of the paper dataset.

Without ``--execute``, the script only prints the requested run and any
completed progress. Existing completed batches are reused; an interrupted
batch is run again from the beginning.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Sequence


FAMILIES = (
    "stokes",
    "tanaka",
    "benjamin_feir",
    "jonswap_tma",
)
SPLITS = ("train", "validation", "test")


def _positive_integer(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=FAMILIES,
        required=True,
        help="Physical paper-dataset family to generate.",
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
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
    if args.family == "jonswap_tma":
        if args.solver_batch_size is None:
            raise ValueError("JONSWAP/TMA requires solver_batch_size")
        if args.solver_batch_size > args.batch_size:
            raise ValueError("solver_batch_size must not exceed batch_size")
    elif args.solver_batch_size is not None:
        raise ValueError("solver_batch_size is only used for JONSWAP/TMA")

    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["DNO_TANAKA_DTYPE"] = "float64"
    os.environ["JAX_PLATFORMS"] = "cpu" if args.platform == "cpu" else "cuda"
    if args.platform == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import jax

    from solver.gen_data.benjamin_feir_sampling import (
        BENJAMIN_FEIR_PARAMETER_GROUP_IDS,
    )
    from solver.gen_data.jonswap_tma_sampling import JONSWAP_TMA_PARAMETER_GROUP_IDS
    from solver.gen_data.pipeline.artifact_io import write_json_atomic
    from solver.gen_data.pipeline.build_dataset_view import build_dataset_view
    from solver.gen_data.pipeline.dataset_generation import (
        DatasetChunkConfig,
        generate_simulations,
        scan_dataset_generation,
    )
    from solver.gen_data.pipeline.simulation_allocation import (
        DatasetSplit,
        PhysicalFamilyId,
        balanced_simulation_targets,
    )
    from solver.gen_data.pipeline.trajectory_config import (
        TrajectoryExecutionConfig,
        paper_trajectory_execution,
    )
    from solver.gen_data.stokes_sampling import STOKES_PARAMETER_GROUP_IDS
    from solver.gen_data.tanaka_sampling import TANAKA_PARAMETER_GROUP_IDS

    family_ids = {
        "stokes": PhysicalFamilyId.STOKES,
        "tanaka": PhysicalFamilyId.TANAKA,
        "benjamin_feir": PhysicalFamilyId.BENJAMIN_FEIR,
        "jonswap_tma": PhysicalFamilyId.JONSWAP_TMA,
    }
    parameter_group_ids = {
        "stokes": STOKES_PARAMETER_GROUP_IDS,
        "tanaka": TANAKA_PARAMETER_GROUP_IDS,
        "benjamin_feir": BENJAMIN_FEIR_PARAMETER_GROUP_IDS,
        "jonswap_tma": JONSWAP_TMA_PARAMETER_GROUP_IDS,
    }[args.family]
    output_root = args.output_root.expanduser().resolve()
    dataset_split = DatasetSplit(args.split)
    chunk_config = DatasetChunkConfig(
        root=output_root,
        family_name=args.family,
        family_id=family_ids[args.family],
        dataset_split=dataset_split,
        simulation_targets=balanced_simulation_targets(
            parameter_group_ids,
            simulation_count=args.accepted_simulations,
        ),
        batch_size=args.batch_size,
        solver_batch_size=args.solver_batch_size,
    )

    if args.execute:
        from solver.gen_data.jonswap_horizon_generator import (
            HorizonBucketedJonswapBatchGenerator,
        )
        from solver.gen_data.stokes_batch_generator import (
            make_static_stokes_batch_generator,
        )
        from solver.gen_data.stokes_sampling import DEFAULT_MAXIMUM_URSELL_REDRAWS
        from solver.gen_data.stokes_static_pipeline import (
            PAPER_STATIC_STOKES_CONTRACT,
            StaticStokesContract,
        )
        from solver.gen_data.trajectory_batch_generator import (
            TrajectoryBatchGenerator,
        )

        execution = (
            PAPER_STATIC_STOKES_CONTRACT
            if args.family == "stokes"
            else paper_trajectory_execution(args.family)
        )
        backend = jax.default_backend()
        if backend != args.platform:
            raise RuntimeError(
                f"requested {args.platform}, but JAX initialized {backend}"
            )
        total_started = perf_counter()
        if isinstance(execution, StaticStokesContract):
            generator = make_static_stokes_batch_generator(
                chunk_config=chunk_config,
                contract=execution,
                maximum_ursell_redraws=DEFAULT_MAXIMUM_URSELL_REDRAWS,
            )
            length = execution.length
        elif args.family == "jonswap_tma":
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
        view_name = f"paper_dataset_{args.family}_{dataset_split.value}"
        view_started = perf_counter()
        view = build_dataset_view(
            output_root,
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
            "output_root": str(output_root),
            "run_spec": chunk_config.to_json_record(),
            "batch_paths": [
                str(path.resolve().relative_to(output_root))
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
        summary_path = output_root / f"{view_name}.summary.json"
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
