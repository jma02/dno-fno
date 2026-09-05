"""Generate one family and split of the paper dataset.

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
from solver.gen_data.pipeline.build_dataset_view import build_dataset_view
from solver.gen_data.pipeline.dataset_generation import generate_simulations
from solver.gen_data.pipeline.trajectory_config import PAPER_ROLLOUT_NUMERICS
from solver.gen_data.pipeline.types import DatasetSplit, PhysicalFamilyId
from solver.gen_data.stokes_batch_generator import generate_static_stokes_batch
from solver.gen_data.stokes_sampling import (
    PAPER_DOMAIN_LENGTH,
    STOKES_PARAMETER_GROUPS,
)
from solver.gen_data.tanaka_sampling import TANAKA_PARAMETER_GROUPS
from solver.gen_data.trajectory_batch_generator import generate_trajectory_batch


FAMILIES = {
    "stokes": STOKES_PARAMETER_GROUPS,
    "tanaka": TANAKA_PARAMETER_GROUPS,
    "benjamin_feir": BENJAMIN_FEIR_PARAMETER_GROUPS,
    "jonswap_tma": JONSWAP_TMA_PARAMETER_GROUPS,
}
SPLITS = ("train", "validation", "test")


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
        "--num-simulations",
        type=int,
        required=True,
        help=(
            "Number of valid simulations to generate for this family and split. "
            "Rejected attempts do not count."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root containing completed batches and dataset views.",
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
    dataset_split = DatasetSplit(args.split)
    family_parameter_groups = FAMILIES[args.family]
    count, remainder = divmod(args.num_simulations, len(family_parameter_groups))
    accepted_targets = {
        group: count + (index < remainder)
        for index, group in enumerate(family_parameter_groups)
    }
    family_id = PhysicalFamilyId[args.family.upper()]

    jax.default_backend()
    total_started = perf_counter()
    if args.family == "stokes":
        generate_batch = partial(
            generate_static_stokes_batch,
            dataset_split=dataset_split,
        )
        length = PAPER_DOMAIN_LENGTH
    else:
        numerical = PAPER_ROLLOUT_NUMERICS[args.family]
        generate_batch = partial(
            generate_trajectory_batch,
            dataset_split=dataset_split,
            family=args.family,
            numerical=numerical,
            solver_batch_size=args.solver_batch_size,
        )
        length = numerical.length

    generation_started = perf_counter()
    attempted, completed_batches = generate_simulations(
        output_root,
        family_id=family_id,
        dataset_split=dataset_split,
        accepted_targets=accepted_targets,
        batch_size=args.batch_size,
        generate_batch=generate_batch,
    )
    generation_seconds = perf_counter() - generation_started
    view_name = f"paper_dataset_{args.family}_{dataset_split.value}"
    view_started = perf_counter()
    view = build_dataset_view(
        output_root,
        completed_batches,
        name=view_name,
        length=length,
    )
    view_seconds = perf_counter() - view_started
    attempted_total = sum(attempted.values())
    accepted_total = sum(accepted_targets.values())
    counts = {
        "attempted": attempted_total,
        "accepted": accepted_total,
        "rejected": attempted_total - accepted_total,
        "by_parameter_group": {
            parameter_group_id: {
                "target_accepted": target_count,
                "attempted": attempted[parameter_group_id],
                "accepted": target_count,
                "rejected": attempted[parameter_group_id] - target_count,
            }
            for parameter_group_id, target_count in accepted_targets.items()
        },
    }
    view_record = {
        "manifest": str(view.manifest),
        "trajectory_map": str(view.trajectory_map),
    }
    run_spec: dict[str, object] = {
        "family_name": args.family,
        "family_id": int(family_id),
        "dataset_split": dataset_split.value,
        "batch_size": args.batch_size,
        "accepted_simulation_count": sum(accepted_targets.values()),
        "quotas": [
            {
                "parameter_group_id": parameter_group_id,
                "target_accepted": target,
            }
            for parameter_group_id, target in accepted_targets.items()
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
        "status": summary["status"],
        "summary_path": str(summary_path),
        "counts": summary["counts"],
        "dataset_view": summary["dataset_view"],
        "timing_seconds": summary["timing_seconds"],
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
