"""Run adjusted JONSWAP/TMA quotas in horizon-sorted numerical microbatches.

This launcher extends the current shared JONSWAP run identity with the
horizon-bucketing configuration and uses the adjustment-first executor. Each case
receives its fingerprinted nonlinear burn-in before the autonomous production
trajectory begins from the full internal-band endpoint.
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

import scripts.run_paper_dataset_quota as base  # noqa: E402
from solver.gen_data.jonswap_horizon_executor import (  # noqa: E402
    BUCKETING_CONFIG_KEY,
    BucketingConfig,
    HorizonBucketedJonswapQuotaExecutor,
)
from solver.gen_data.pipeline.archive import file_sha256  # noqa: E402
from solver.gen_data.pipeline.quota_driver import (  # noqa: E402
    AcceptedQuotaRunSpec,
)
from solver.gen_data.trajectory_quota_executor import (  # noqa: E402
    TrajectoryExecutionConfig,
)


EXTRA_SOURCE_PATHS = (
    Path(__file__).resolve(),
    ROOT / "solver/gen_data/jonswap_horizon_executor.py",
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
        help="Maximum cases in one numerical rollout after horizon sorting.",
    )
    parsed, remaining = parser.parse_known_args(argv)
    return parsed.solver_batch_size, tuple(remaining)


def build_bucketed_run_spec(
    request: base.GenerationRequest,
    *,
    solver_batch_size: int,
    execution: base.PaperExecution | None = None,
) -> AcceptedQuotaRunSpec:
    """Extend the exact JONSWAP run identity with its bucketing configuration."""

    if request.family != "jonswap_tma":
        raise ValueError("bucketed launcher supports only JONSWAP/TMA")
    if (
        isinstance(solver_batch_size, bool)
        or not isinstance(solver_batch_size, int)
        or solver_batch_size <= 0
    ):
        raise ValueError("solver batch size must be a positive integer")
    if solver_batch_size > request.batch_size:
        raise ValueError("solver batch size cannot exceed proposal batch size")

    selected_execution = (
        TrajectoryExecutionConfig.paper("jonswap_tma")
        if execution is None
        else execution
    )
    if not isinstance(selected_execution, TrajectoryExecutionConfig):
        raise TypeError("JONSWAP/TMA requires a trajectory execution contract")
    adjustment_policy = selected_execution.jonswap_adjustment
    if adjustment_policy is None:
        raise ValueError("JONSWAP/TMA bucketed execution requires adjustment")
    baseline = _BASE_BUILD_RUN_SPEC(request, execution=selected_execution)
    record = baseline.to_json_record()
    configuration = record["configuration"]
    assert isinstance(configuration, dict)
    source_sha256 = configuration["source_sha256"]
    assert isinstance(source_sha256, dict)
    source_sha256.update(
        {
            str(path.resolve().relative_to(ROOT)): file_sha256(path)
            for path in EXTRA_SOURCE_PATHS
        }
    )
    configuration[BUCKETING_CONFIG_KEY] = BucketingConfig(
        outer_proposal_size=request.batch_size,
        solver_batch_size=solver_batch_size,
        adjustment=adjustment_policy,
    ).to_json_record()
    return AcceptedQuotaRunSpec(
        root=baseline.root,
        family_name=baseline.family_name,
        family_id=baseline.family_id,
        revision_id=baseline.revision_id,
        split_id=baseline.split_id,
        stream_id=baseline.stream_id,
        quotas=baseline.quotas,
        cell_codes=baseline.cell_codes,
        batch_size=baseline.batch_size,
        first_attempt_index=baseline.first_attempt_index,
        maximum_attempts_per_accepted_case=(
            baseline.maximum_attempts_per_accepted_case
        ),
        configuration=configuration,
    )


@contextmanager
def _bucketed_runtime(solver_batch_size: int) -> Iterator[None]:
    """Install the JONSWAP-only spec and executor in the shared runner."""

    original_builder = base.build_run_spec
    original_executor = base.TrajectoryQuotaExecutor

    def builder(
        request: base.GenerationRequest,
        *,
        execution: base.PaperExecution | None = None,
    ) -> AcceptedQuotaRunSpec:
        return build_bucketed_run_spec(
            request,
            solver_batch_size=solver_batch_size,
            execution=execution,
        )

    base.build_run_spec = builder
    base.TrajectoryQuotaExecutor = HorizonBucketedJonswapQuotaExecutor
    try:
        yield
    finally:
        base.build_run_spec = original_builder
        base.TrajectoryQuotaExecutor = original_executor


_BASE_BUILD_RUN_SPEC = base.build_run_spec


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
                "configuration_fingerprint": result.summary[
                    "configuration_fingerprint"
                ],
                "counts": result.summary["counts"],
                "dataset_view": result.summary["dataset_view"],
                "timing_seconds": result.summary["timing_seconds"],
            }
        else:
            _, _, _, output = base.preflight(request)
    print(base.json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
