"""Replay saved JONSWAP cases, retaining their 16 old rows plus 184 new frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Sequence

os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")
os.environ["DNO_TANAKA_DTYPE"] = "float64"

import jax
import numpy as np

from solver.gen_data.jonswap_tma import JonswapTmaParameters, ResolvedBand
from solver.gen_data.jonswap_tma_sampling import JonswapTmaSample
from solver.gen_data.pipeline.time_selection import select_uniform_times
from solver.gen_data.pipeline.trajectory_config import PAPER_ROLLOUT_NUMERICS
from solver.gen_data.pipeline.trajectory_rollout import (
    execute_adjustment_batch,
    execute_trajectory_batch,
)
from solver.gen_data.trajectory_family_adapters import construct_jonswap_tma_trajectory_batch


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    numerical = PAPER_ROLLOUT_NUMERICS["jonswap_tma"]
    with args.inputs.open("rb") as handle:
        fingerprint = hashlib.file_digest(handle, "sha256").hexdigest()
    with np.load(args.inputs, allow_pickle=False) as archive:
        inputs = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(inputs["metadata_json"].item()))
    if metadata["current_numerical"] != numerical._asdict() or metadata["numerical_differences"]:
        raise ValueError("Saved replay inputs do not match the current rollout numerics")

    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    progress_path = root / ("pilot.json" if args.pilot_only else "progress.json")
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else {"input_sha256": fingerprint, "completed_simulations": 0, "elapsed_seconds": 0.0}
    )
    if progress["input_sha256"] != fingerprint:
        raise ValueError("Replay inputs changed since this output directory was started")
    count = inputs["simulation_id"].size
    selected_cases = inputs["comparison_indices"] if args.pilot_only else np.arange(count)
    order = selected_cases[np.argsort(inputs["production_time_count"][selected_cases], kind="stable")]
    comparison_lookup = {int(index): row for row, index in enumerate(inputs["comparison_indices"])}
    source = {}
    arrays = {}
    if not args.pilot_only:
        pilot = json.loads((root / "pilot.json").read_text())
        if not pilot["complete"] or pilot["input_sha256"] != fingerprint:
            raise ValueError("Complete a successful pilot on these inputs before bulk replay")
        if args.source_dataset is None:
            parser.error("bulk replay requires --source-dataset for the original 16 rows")
        source = {
            name: np.load(args.source_dataset / f"{name}.npy", mmap_mode="r")
            for name in ("eta", "xi", "gxi", "depth", "time")
        }
        dtypes = {
            "eta": "float32", "xi": "float32", "gxi": "float32",
            "depth": "float64", "time": "float64", "frame_index": "int32",
            "simulation_id": "int64", "family_id": "int16",
            "parameter_group_id": inputs["parameter_group_id"].dtype,
            "dataset_split": inputs["dataset_split"].dtype,
        }
        for name, dtype in dtypes.items():
            path = root / f"{name}.npy"
            shape = (count * 200, numerical.target_nx) if name in ("eta", "xi", "gxi") else (count * 200,)
            arrays[name] = np.lib.format.open_memmap(
                path, mode="r+" if path.exists() else "w+", dtype=dtype, shape=shape,
            )
            if arrays[name].shape != shape or arrays[name].dtype != np.dtype(dtype):
                raise ValueError(f"Unexpected replay output shape or dtype: {path}")
        np.save(root / "x.npy", numerical.length * np.arange(numerical.target_nx) / numerical.target_nx)

    total_frames = inputs["production_time_count"] + inputs["adjustment_time_count"]
    completed = progress["completed_simulations"]
    for batch_number, start in enumerate(range(completed, order.size, args.batch_size)):
        if args.max_batches is not None and batch_number >= args.max_batches:
            break
        began = perf_counter()
        indices = order[start : start + args.batch_size]
        samples = tuple(
            JonswapTmaSample(
                JonswapTmaParameters(*(float(inputs[name][index]) for name in JonswapTmaParameters._fields)),
                inputs["phase_right"][index], inputs["phase_left"][index],
            )
            for index in indices
        )
        initial, valid = construct_jonswap_tma_trajectory_batch(
            samples, numerical,
            band=ResolvedBand(numerical.length, numerical.target_maximum_wavenumber),
        )
        if initial is None or valid != tuple(range(indices.size)):
            raise RuntimeError(f"Archived accepted initial conditions were rejected: {indices.tolist()}")
        endpoints = execute_adjustment_batch(
            initial.eta0, initial.xi0, initial.depths,
            tuple(numerical.saved_dt * np.arange(inputs["adjustment_time_count"][index]) for index in indices),
            nonlinear_ramp_times=inputs["nonlinear_ramp_time"][indices],
            nonlinear_ramp_order=4, config=numerical,
        )
        if any(endpoint is None for endpoint in endpoints):
            raise RuntimeError(f"Archived accepted adjustment failed: {indices.tolist()}")
        trajectories = execute_trajectory_batch(
            np.stack([endpoint[0] for endpoint in endpoints if endpoint is not None]),
            np.stack([endpoint[1] for endpoint in endpoints if endpoint is not None]),
            initial.depths,
            tuple(numerical.saved_dt * np.arange(inputs["production_time_count"][index]) for index in indices),
            config=numerical,
        )
        errors = {}
        for index, trajectory in zip(indices, trajectories, strict=True):
            if trajectory is None:
                raise RuntimeError(f"Archived accepted production failed: simulation {inputs['simulation_id'][index]}")
            old_indices = inputs["old_dense_indices"][index]
            np.testing.assert_allclose(trajectory.times[old_indices], inputs["old_times"][index], rtol=0.0, atol=1e-12)
            old_start = int(inputs["old_first_row"][index])
            old_rows = slice(old_start, old_start + 16)
            remaining = np.setdiff1d(np.arange(trajectory.times.size), old_indices, assume_unique=True)
            selected = np.sort(np.concatenate((old_indices, remaining[select_uniform_times(remaining.size, keep_samples=184)])))
            if selected.size != 200 or np.any(np.diff(selected) <= 0):
                raise ValueError("Replay must retain 200 distinct dense frames")
            old_positions = np.searchsorted(selected, old_indices)
            rows = slice(int(index) * 200, (int(index) + 1) * 200)
            for name in ("eta", "xi", "gxi"):
                reference = (
                    inputs[f"comparison_{name}"][comparison_lookup[int(index)]]
                    if args.pilot_only else source[name][old_rows]
                )
                replayed = getattr(trajectory, name)[old_indices]
                reference64 = reference.astype(np.float64)
                difference = replayed - reference64
                relative_l2 = float(np.max(
                    np.linalg.norm(difference, axis=1)
                    / np.maximum(np.linalg.norm(reference64, axis=1), np.finfo(float).tiny)
                ))
                relative_peak = float(np.max(
                    np.max(np.abs(difference), axis=1)
                    / np.maximum(np.max(np.abs(reference64), axis=1), np.finfo(float).tiny)
                ))
                errors[name] = max(errors.get(name, 0.0), relative_l2, relative_peak)
                if not np.isfinite(errors[name]) or errors[name] > 1e-6:
                    raise RuntimeError(
                        f"Replay mismatch: simulation {inputs['simulation_id'][index]}, {name}, "
                        f"relative L2={relative_l2:.9g}, peak={relative_peak:.9g}; limit=1e-6"
                    )
                if not args.pilot_only:
                    values = getattr(trajectory, name)[selected].astype(np.float32)
                    values[old_positions] = reference
                    arrays[name][rows] = values
            if not args.pilot_only:
                times = trajectory.times[selected].copy()
                times[old_positions] = source["time"][old_rows]
                arrays["time"][rows] = times
                arrays["depth"][rows] = source["depth"][old_start]
                arrays["frame_index"][rows] = np.arange(200)
                arrays["family_id"][rows] = 4
                for name in ("simulation_id", "dataset_split", "parameter_group_id"):
                    arrays[name][rows] = inputs[name][index]
        for array in arrays.values():
            array.flush()
        completed = start + indices.size
        elapsed = perf_counter() - began
        progress.update(
            completed_simulations=int(completed), complete=completed == order.size,
            elapsed_seconds=progress["elapsed_seconds"] + elapsed,
            maximum_relative_error=max(progress.get("maximum_relative_error", 0.0), *errors.values()),
            numerical=numerical._asdict(),
        )
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2))
        temporary.replace(progress_path)
        estimate = progress["elapsed_seconds"] * float(total_frames.sum() / total_frames[order[:completed]].sum())
        print(json.dumps({
            "completed_simulations": int(completed), "total_simulations": int(order.size),
            "batch_seconds": elapsed, "errors": errors, "estimated_full_seconds": estimate,
        }), flush=True)
    if not args.pilot_only and completed == order.size:
        (root / "complete.json").write_text(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()
