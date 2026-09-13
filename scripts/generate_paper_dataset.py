"""Generate accepted simulations from one paper-dataset family.

Existing completed batches are reused; an interrupted batch is run again from
the beginning.
"""

from __future__ import annotations

import argparse
from functools import partial
import os
from pathlib import Path
from typing import Sequence

# Tanaka chooses its precision at import time.
os.environ["DNO_TANAKA_DTYPE"] = "float64"
os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")

import jax

from solver.gen_data.benjamin_feir_sampling import BENJAMIN_FEIR_PARAMETER_GROUPS
from solver.gen_data.jonswap_tma_sampling import JONSWAP_TMA_PARAMETER_GROUPS
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
        help="Root containing completed simulation batches.",
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

    attempts_per_group, completed_batches = generate_simulations(
        output_root,
        family_id=family_id,
        seed=args.seed,
        requested_simulations_per_group=requested_simulations_per_group,
        batch_size=args.batch_size,
        generate_batch=generate_batch,
    )
    print(
        f"{args.family}: accepted={args.num_simulations}, "
        f"attempted={sum(attempts_per_group.values())}, batches={len(completed_batches)}"
    )


if __name__ == "__main__":
    main()
