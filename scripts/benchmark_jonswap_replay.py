"""Measure FP64 JONSWAP replay batch throughput without writing dataset arrays.

Default: two identical 30-interval adjustment/production workloads, exposing cold
startup versus warm timing. The shortened burn makes this a throughput proxy,
not a production-coverage or full-horizon memory test. --full-horizon restores
the saved adjustment and production lengths; --longest selects the longest cases.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Sequence

os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["DNO_TANAKA_DTYPE"] = "float64"

import jax
import numpy as np

from solver.gen_data.jonswap_tma import JonswapTmaParameters, ResolvedBand
from solver.gen_data.jonswap_tma_sampling import JonswapTmaSample
from solver.gen_data.pipeline.trajectory_config import PAPER_ROLLOUT_NUMERICS
from solver.gen_data.pipeline.trajectory_rollout import execute_adjustment_batch, execute_trajectory_batch
from solver.gen_data.trajectory_family_adapters import construct_jonswap_tma_trajectory_batch


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-horizon", action="store_true")
    parser.add_argument("--longest", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    numerical = PAPER_ROLLOUT_NUMERICS["jonswap_tma"]
    with np.load(args.inputs, allow_pickle=False) as archive:
        inputs = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(inputs["metadata_json"].item()))
    if metadata["current_numerical"] != numerical._asdict() or metadata["numerical_differences"]:
        raise ValueError("Benchmark inputs do not match the current rollout numerics")
    counts = inputs["production_time_count"]
    distance = -counts if args.longest else np.abs(counts - np.median(counts))
    indices = np.argsort(distance, kind="stable")[:args.batch_size]
    grids = {
        phase: tuple(
            numerical.saved_dt * np.arange(
                int(inputs[f"{phase}_time_count"][index]) if args.full_horizon else 31, dtype=np.float64,
            )
            for index in indices
        )
        for phase in ("adjustment", "production")
    }
    comparisons = {int(index): row for row, index in enumerate(inputs["comparison_indices"])}
    device = jax.local_devices()[0]
    report = {
        "device": device.device_kind, "batch_size": int(indices.size),
        "full_horizon": args.full_horizon, "longest": args.longest,
        "simulation_ids": inputs["simulation_id"][indices].tolist(),
        "numerical": numerical._asdict(),
        "saved_time_counts": {phase: [len(grid) for grid in times] for phase, times in grids.items()},
        "maximum_saved_time_counts": {phase: max(map(len, times)) for phase, times in grids.items()},
        "timing_note": "First run includes compilation; later repeats reuse identical inputs and shapes. Short mode is not an archived production replay.",
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeats):
        result = {"repeat": repeat + 1, "status": "running"}
        report["runs"].append(result)
        print(f"BEGIN repeat={repeat + 1}", flush=True)
        started = perf_counter()
        try:
            samples = tuple(
                JonswapTmaSample(
                    JonswapTmaParameters(*(float(inputs[name][index]) for name in JonswapTmaParameters._fields)),
                    inputs["phase_right"][index], inputs["phase_left"][index],
                )
                for index in indices
            )
            initial, valid = construct_jonswap_tma_trajectory_batch(
                samples, numerical, band=ResolvedBand(numerical.length, numerical.target_maximum_wavenumber),
            )
            result["constructor_seconds"] = perf_counter() - started
            result["valid_initial"] = len(valid)
            if initial is None or valid != tuple(range(indices.size)):
                raise RuntimeError("Benchmark initial conditions were rejected")
            phase_start = perf_counter()
            endpoints = execute_adjustment_batch(
                initial.eta0, initial.xi0, initial.depths, grids["adjustment"],
                nonlinear_ramp_times=inputs["nonlinear_ramp_time"][indices], nonlinear_ramp_order=4, config=numerical,
            )
            result["adjustment_seconds"] = perf_counter() - phase_start
            result["valid_adjustment"] = sum(endpoint is not None for endpoint in endpoints)
            if result["valid_adjustment"] != indices.size:
                raise RuntimeError("Benchmark adjustment failed")
            phase_start = perf_counter()
            trajectories = execute_trajectory_batch(
                np.stack([endpoint[0] for endpoint in endpoints if endpoint is not None]),
                np.stack([endpoint[1] for endpoint in endpoints if endpoint is not None]),
                initial.depths, grids["production"], config=numerical,
            )
            result["production_and_labels_seconds"] = perf_counter() - phase_start
            result["total_seconds"] = perf_counter() - started
            result["valid_production"] = sum(case is not None for case in trajectories)
            if result["valid_production"] != indices.size:
                raise RuntimeError("Benchmark production failed its numerical health checks")
            # Only full-horizon cases with embedded references support this comparison.
            maximum_error = 0.0
            compared = 0
            for index, case in zip(indices, trajectories, strict=True):
                if not args.full_horizon or int(index) not in comparisons:
                    continue
                assert case is not None
                for name in ("eta", "xi", "gxi"):
                    reference = inputs[f"comparison_{name}"][comparisons[int(index)]].astype(np.float64)
                    difference = getattr(case, name)[inputs["old_dense_indices"][index]] - reference
                    for numerator, denominator in (
                        (np.linalg.norm(difference, axis=1), np.linalg.norm(reference, axis=1)),
                        (np.max(np.abs(difference), axis=1), np.max(np.abs(reference), axis=1)),
                    ):
                        error = float(np.max(numerator / np.maximum(denominator, np.finfo(float).tiny)))
                        if not np.isfinite(error) or error > 1e-6:
                            raise RuntimeError(f"Archived-frame comparison failed: simulation {inputs['simulation_id'][index]}, {name}: {error}")
                        maximum_error = max(maximum_error, error)
                compared += 1
            result.update(compared_simulations=compared, maximum_relative_error=maximum_error, status="passed")
            intervals = sum(max(map(len, times)) - 1 for times in grids.values())
            result["case_saved_intervals_per_second"] = float(indices.size * intervals / result["total_seconds"])
            if repeat:
                result["first_minus_repeat_seconds"] = report["runs"][0]["total_seconds"] - result["total_seconds"]
        except Exception as error:
            result.update(status="failed", error=str(error))
            raise
        finally:
            memory = device.memory_stats() or {}
            result["allocator_memory_bytes"] = {
                name: int(memory[name]) for name in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit") if name in memory
            }
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
