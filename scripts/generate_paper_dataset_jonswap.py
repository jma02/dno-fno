"""Generate JONSWAP/TMA simulations in rollout-length-sorted microbatches.

This launcher extends the current shared JONSWAP chunk configuration with the
horizon-bucketing configuration and uses the adjustment-first executor. Each
simulation receives a nonlinear burn-in before the autonomous trajectory begins
from the full internal-band endpoint.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Iterator, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.generate_paper_dataset as base  # noqa: E402
from solver.gen_data.jonswap_horizon_executor import (  # noqa: E402
    BUCKETING_CONFIG_KEY,
    BucketingConfig,
    HorizonBucketedJonswapBatchExecutor,
)
from solver.gen_data.pipeline.dataset_generation import (  # noqa: E402
    DatasetChunkConfig,
)
from solver.gen_data.trajectory_batch_executor import (  # noqa: E402
    TrajectoryExecutionConfig,
    paper_trajectory_execution,
)


def _positive_integer(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _split_policy_args(
    argv: Sequence[str],
) -> tuple[int, tuple[str, ...]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--solver-batch-size",
        type=_positive_integer,
        required=True,
        help="Maximum simulations in one numerical rollout after sorting.",
    )
    parsed, remaining = parser.parse_known_args(argv)
    return parsed.solver_batch_size, tuple(remaining)


def build_bucketed_chunk_config(
    request: base.GenerationRequest,
    *,
    solver_batch_size: int,
    execution: base.PaperExecution | None = None,
) -> DatasetChunkConfig:
    """Add rollout bucketing to the JONSWAP chunk configuration."""

    if request.family != "jonswap_tma":
        raise ValueError("bucketed launcher supports only JONSWAP/TMA")
    if (
        isinstance(solver_batch_size, bool)
        or not isinstance(solver_batch_size, int)
        or solver_batch_size <= 0
    ):
        raise ValueError("solver batch size must be a positive integer")
    if solver_batch_size > request.batch_size:
        raise ValueError("solver batch size cannot exceed dataset batch size")

    selected_execution = (
        paper_trajectory_execution("jonswap_tma") if execution is None else execution
    )
    if not isinstance(selected_execution, TrajectoryExecutionConfig):
        raise TypeError("JONSWAP/TMA requires a trajectory execution contract")
    adjustment_policy = selected_execution.jonswap_adjustment
    if adjustment_policy is None:
        raise ValueError("JONSWAP/TMA bucketed execution requires adjustment")
    baseline = _BASE_BUILD_CHUNK_CONFIG(request, execution=selected_execution)
    record = baseline.to_json_record()
    configuration = record["configuration"]
    assert isinstance(configuration, dict)
    configuration[BUCKETING_CONFIG_KEY] = BucketingConfig(
        batch_size=request.batch_size,
        solver_batch_size=solver_batch_size,
        adjustment=adjustment_policy,
    ).to_json_record()
    return DatasetChunkConfig(
        root=baseline.root,
        family_name=baseline.family_name,
        family_id=baseline.family_id,
        dataset_split=baseline.dataset_split,
        worker_stream_id=baseline.worker_stream_id,
        simulation_targets=baseline.simulation_targets,
        parameter_group_codes=baseline.parameter_group_codes,
        batch_size=baseline.batch_size,
        first_attempt_index=baseline.first_attempt_index,
        configuration=configuration,
    )


@contextmanager
def _bucketed_runtime(solver_batch_size: int) -> Iterator[None]:
    """Install the JONSWAP chunk builder and executor in the shared runner."""

    original_builder = base.build_chunk_config
    original_executor = base.TrajectoryBatchExecutor

    def builder(
        request: base.GenerationRequest,
        *,
        execution: base.PaperExecution | None = None,
    ) -> DatasetChunkConfig:
        return build_bucketed_chunk_config(
            request,
            solver_batch_size=solver_batch_size,
            execution=execution,
        )

    base.build_chunk_config = builder
    base.TrajectoryBatchExecutor = HorizonBucketedJonswapBatchExecutor
    try:
        yield
    finally:
        base.build_chunk_config = original_builder
        base.TrajectoryBatchExecutor = original_executor


_BASE_BUILD_CHUNK_CONFIG = base.build_chunk_config


def main(argv: Sequence[str] | None = None) -> None:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    solver_batch_size, remaining = _split_policy_args(raw_argv)
    args = base.parse_args(remaining)
    request = base.request_from_args(args)
    if request.family != "jonswap_tma":
        raise ValueError("bucketed launcher supports only JONSWAP/TMA")

    with _bucketed_runtime(solver_batch_size):
        if args.execute:
            result = base.run_generation(request)
            output: dict[str, object] = {
                "mode": "execute",
                "status": result.summary["status"],
                "summary_path": str(result.summary_path),
                "counts": result.summary["counts"],
                "dataset_view": result.summary["dataset_view"],
                "timing_seconds": result.summary["timing_seconds"],
            }
        else:
            _, _, _, output = base.preflight(request)
    print(base.json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
