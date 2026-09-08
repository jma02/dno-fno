"""Generate accepted simulations from one paper-dataset family.

Existing completed batches are reused; an interrupted batch is run again from
the beginning.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Sequence

# Tanaka chooses its precision at import time.
os.environ["DNO_TANAKA_DTYPE"] = "float64"

import jax

from solver.gen_data.benjamin_feir_sampling import BENJAMIN_FEIR_PARAMETER_GROUPS
from solver.gen_data.jonswap_tma_sampling import JONSWAP_TMA_PARAMETER_GROUPS
from solver.gen_data.pipeline.artifact_io import write_json_atomic
from solver.gen_data.pipeline.dataset_generation import generate_simulations
from solver.gen_data.pipeline.trajectory_config import PAPER_ROLLOUT_NUMERICS
from solver.gen_data.pipeline.types import (
    PhysicalFamilyId,
    RequestedSimulationsPerGroup,
)
from solver.gen_data.stokes_batch_generator import generate_static_stokes_batch
from solver.gen_data.stokes_sampling import STOKES_PARAMETER_GROUPS
from solver.gen_data.tanaka_sampling import TANAKA_PARAMETER_GROUPS
from solver.gen_data.trajectory_batch_generator import generate_trajectory_batch


FAMILIES = {
    "stokes": STOKES_PARAMETER_GROUPS,
    "tanaka": TANAKA_PARAMETER_GROUPS,
    "benjamin_feir": BENJAMIN_FEIR_PARAMETER_GROUPS,
    "jonswap_tma": JONSWAP_TMA_PARAMETER_GROUPS,
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=FAMILIES,
        required=True,
        help="Physical paper-dataset family to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026072210,
        help="Random seed for sampling simulations.",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        required=True,
        help=(
            "Number of valid simulations to generate for this family. "
            "Rejected attempts do not count."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root containing completed batches and the generation summary.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Maximum attempted simulations in one batch.",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU instead of CPU.")
    parser.add_argument(
        "--solver-batch-size",
        type=int,
        help="JONSWAP simulations solved together after sorting by rollout length.",
    )
    args = parser.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cuda" if args.gpu else "cpu")
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    output_root = args.output_root.expanduser().resolve()
    family_parameter_groups = FAMILIES[args.family]
    per_group, remainder = divmod(args.num_simulations, len(family_parameter_groups))
    requested_simulations_per_group: RequestedSimulationsPerGroup = {
        group: per_group + (index < remainder)
        for index, group in enumerate(family_parameter_groups)
    }
    family_id = PhysicalFamilyId[args.family.upper()]

    jax.default_backend()
    total_started = perf_counter()
    if args.family == "stokes":
        generate_batch = partial(
            generate_static_stokes_batch,
            seed=args.seed,
        )
    else:
        numerical = PAPER_ROLLOUT_NUMERICS[args.family]
        generate_batch = partial(
            generate_trajectory_batch,
            seed=args.seed,
            family=args.family,
            numerical=numerical,
            solver_batch_size=args.solver_batch_size,
        )

    generation_started = perf_counter()
    attempts_per_group, completed_batches = generate_simulations(
        output_root,
        family_id=family_id,
        seed=args.seed,
        requested_simulations_per_group=requested_simulations_per_group,
        batch_size=args.batch_size,
        generate_batch=generate_batch,
    )
    generation_seconds = perf_counter() - generation_started
    total_attempts = sum(attempts_per_group.values())
    total_successful = sum(requested_simulations_per_group.values())
    simulation_summary = {
        "attempted": total_attempts,
        "accepted": total_successful,
        "rejected": total_attempts - total_successful,
        "by_parameter_group": {
            parameter_group_id: {
                "target_accepted": target_count,
                "attempted": attempts_per_group[parameter_group_id],
                "accepted": target_count,
                "rejected": attempts_per_group[parameter_group_id] - target_count,
            }
            for parameter_group_id, target_count in requested_simulations_per_group.items()
        },
    }
    run_spec: dict[str, object] = {
        "family_name": args.family,
        "family_id": int(family_id),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "accepted_simulation_count": sum(requested_simulations_per_group.values()),
        "quotas": [
            {
                "parameter_group_id": parameter_group_id,
                "target_accepted": target,
            }
            for parameter_group_id, target in requested_simulations_per_group.items()
        ],
    }
    if args.solver_batch_size is not None:
        run_spec["solver_batch_size"] = args.solver_batch_size
    summary: dict[str, object] = {
        "status": "complete",
        "output_root": str(output_root),
        "run_spec": run_spec,
        "batch_paths": [
            str(path.resolve().relative_to(output_root)) for path in completed_batches
        ],
        "counts": simulation_summary,
        "timing_seconds": {
            "generation": generation_seconds,
            "total": perf_counter() - total_started,
        },
    }
    summary_path = output_root / f"paper_dataset_{args.family}.summary.json"
    write_json_atomic(summary_path, summary)
    output: dict[str, object] = {
        "status": summary["status"],
        "summary_path": str(summary_path),
        "counts": summary["counts"],
        "timing_seconds": summary["timing_seconds"],
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
