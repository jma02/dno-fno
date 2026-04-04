from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from numpy.lib import format as npy_format

from ..gen_data.multi_crest import (
    DEFAULT_RANDOM_MAX_CRESTS,
    DEFAULT_RANDOM_MIN_CRESTS,
    DEFAULT_RANDOM_MIN_SEPARATION,
    flatten_case_specs,
    sample_random_cases,
    serialize_case_specs,
    specs_to_jax_arrays,
)
from ..solvers.dno_series_jax import build_grid, dno_series_eval
from ..solvers.time_integrator import (
    State,
    batched_rollout,
    cast_solver_params_dtype,
    cast_state_dtype,
    make_normalized_rollout_settings,
    make_solver_params,
)
from ..tanaka_ICs.modified_tanaka import make_default_tanaka_template, solve_modified_tanaka_batched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a large Tanaka multicrest dataset by streaming batched rollout chunks into a .npz archive."
    )
    parser.add_argument("--output", default="data/tanaka_1.npz")
    parser.add_argument("--target_samples", type=int, default=10_000_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsample_stride", type=int, default=20)
    parser.add_argument("--keep_samples", type=int)
    parser.add_argument("--gxi_chunk_size", type=int, default=16)
    parser.add_argument("--case_id_offset", type=int, default=0)
    parser.add_argument("--rng_stream_id", type=int)
    parser.add_argument("--rollout_dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--min_crests", type=int, default=DEFAULT_RANDOM_MIN_CRESTS)
    parser.add_argument("--max_crests", type=int, default=DEFAULT_RANDOM_MAX_CRESTS)
    parser.add_argument("--min_separation", type=float, default=DEFAULT_RANDOM_MIN_SEPARATION)
    parser.add_argument("--amplitude_min", type=float, default=0.05)
    parser.add_argument("--amplitude_max", type=float, default=0.35)
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=164.0)
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--tmax", type=float, default=200.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_time_indices(n_times: int, subsample_stride: int, keep_samples: int | None) -> np.ndarray:
    if keep_samples is None:
        return np.arange(0, n_times, subsample_stride, dtype=np.int32)
    if keep_samples <= 0:
        raise ValueError("--keep_samples must be positive.")
    if keep_samples >= n_times:
        return np.arange(n_times, dtype=np.int32)
    return np.linspace(0, n_times - 1, keep_samples, dtype=np.int32)


def chunked_gxi_over_time(
    eta: jnp.ndarray,
    xi: jnp.ndarray,
    params,
    chunk_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    n_steps = int(eta.shape[0])
    for start_idx in range(0, n_steps, chunk_size):
        end_idx = min(start_idx + chunk_size, n_steps)
        gxi_chunk = dno_series_eval(
            eta[start_idx:end_idx],
            xi[start_idx:end_idx],
            params.k,
            params.depth,
            params.dno_order,
            pad_factor=params.pad_factor,
        )
        chunks.append(np.asarray(jax.device_get(gxi_chunk), dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def flatten_samples(
    eta: np.ndarray,
    xi: np.ndarray,
    gxi: np.ndarray,
    times: np.ndarray,
    global_case_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eta_case_major = np.swapaxes(eta, 0, 1).reshape(-1, eta.shape[-1])
    xi_case_major = np.swapaxes(xi, 0, 1).reshape(-1, xi.shape[-1])
    gxi_case_major = np.swapaxes(gxi, 0, 1).reshape(-1, gxi.shape[-1])
    sample_times = np.tile(times, global_case_ids.shape[0])
    sample_case_ids = np.repeat(global_case_ids, times.shape[0])
    return eta_case_major, xi_case_major, gxi_case_major, sample_times, sample_case_ids


def write_npy_entry(zf: zipfile.ZipFile, name: str, array: np.ndarray) -> None:
    with zf.open(name, mode="w", force_zip64=True) as handle:
        npy_format.write_array(handle, np.asarray(array), allow_pickle=False)


def load_state(state_path: Path) -> dict[str, object] | None:
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(state_path: Path, state: dict[str, object]) -> None:
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def make_batch_rng(seed: int, batch_idx: int, rng_stream_id: int) -> np.random.Generator:
    seed_sequence = np.random.SeedSequence([seed, rng_stream_id, batch_idx])
    return np.random.default_rng(seed_sequence)


def main() -> None:
    args = parse_args()
    rollout_defaults = make_normalized_rollout_settings()
    rng_stream_id = args.rng_stream_id if args.rng_stream_id is not None else args.case_id_offset

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = output_path.with_suffix(".state.json")

    if args.overwrite:
        if output_path.exists():
            output_path.unlink()
        if state_path.exists():
            state_path.unlink()

    times = np.arange(0.0, args.tmax + 0.5 * args.dt, args.dt, dtype=np.float32)
    subsample_indices = select_time_indices(
        times.shape[0],
        args.subsample_stride,
        args.keep_samples,
    )
    subsample_times = times[subsample_indices]
    samples_per_batch = int(args.batch_size * subsample_times.shape[0])
    n_batches = int(math.ceil(args.target_samples / samples_per_batch))

    existing_state = load_state(state_path)
    if existing_state is None:
        samples_written = 0
        next_batch = 0
        with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            x, _ = build_grid(args.nx, args.length)
            meta = {
                "target_samples": args.target_samples,
                "batch_size": args.batch_size,
                "subsample_stride": args.subsample_stride,
                "keep_samples": args.keep_samples,
                "samples_per_full_batch": samples_per_batch,
                "n_batches_planned": n_batches,
                "tanaka_dtype": "float64",
                "rollout_dtype": args.rollout_dtype,
                "dt": args.dt,
                "tmax": args.tmax,
                "n_time_samples_full": int(times.shape[0]),
                "n_time_samples_subsampled": int(subsample_times.shape[0]),
                "method": rollout_defaults.method,
                "substeps": rollout_defaults.substeps_per_interval,
                "implicit_iterations": rollout_defaults.implicit_iterations,
                "filter_fraction": rollout_defaults.filter_fraction,
                "zero_mean_xi": rollout_defaults.zero_mean_xi,
                "min_crests": args.min_crests,
                "max_crests": args.max_crests,
                "min_separation": args.min_separation,
                "amplitude_min": args.amplitude_min,
                "amplitude_max": args.amplitude_max,
                "seed": args.seed,
                "case_id_offset": args.case_id_offset,
                "rng_stream_id": rng_stream_id,
                "nx": args.nx,
                "length": args.length,
                "depth": args.depth,
                "gravity": args.gravity,
            }
            write_npy_entry(zf, "x.npy", np.asarray(x, dtype=np.float32))
            write_npy_entry(zf, "subsample_indices.npy", subsample_indices)
            write_npy_entry(zf, "subsample_times.npy", subsample_times)
            zf.writestr("meta.json", json.dumps(meta, indent=2))
        save_state(
            state_path,
            {
                "output_path": str(output_path),
                "samples_written": 0,
                "next_batch": 0,
                "n_batches_planned": n_batches,
                "complete": False,
            },
        )
    else:
        samples_written = int(existing_state["samples_written"])
        next_batch = int(existing_state["next_batch"])
        if bool(existing_state.get("complete", False)) or samples_written >= args.target_samples:
            print(json.dumps(existing_state, indent=2))
            return

    template_params = make_default_tanaka_template(
        depth=args.depth,
        gravity=args.gravity,
        direction=1,
        nx=args.nx,
        length=args.length,
        center=0.0,
        dno_order=rollout_defaults.dno_order,
        pad_factor=rollout_defaults.pad_factor,
    )
    solver_params = make_solver_params(
        nx=args.nx,
        length=args.length,
        depth=args.depth,
        gravity=args.gravity,
        dno_order=rollout_defaults.dno_order,
        pad_factor=rollout_defaults.pad_factor,
        filter_fraction=rollout_defaults.filter_fraction,
    )
    rollout_dtype = jnp.float32 if args.rollout_dtype == "float32" else jnp.float64
    solver_params = cast_solver_params_dtype(solver_params, rollout_dtype)
    rollout_times = jnp.asarray(times, dtype=rollout_dtype)

    total_start = perf_counter()
    for batch_idx in range(next_batch, n_batches):
        batch_start = perf_counter()
        batch_rng = make_batch_rng(args.seed, batch_idx, rng_stream_id)
        case_specs = sample_random_cases(
            batch_rng,
            args.batch_size,
            length=args.length,
            amplitude_min=args.amplitude_min,
            amplitude_max=args.amplitude_max,
            min_crests=args.min_crests,
            max_crests=args.max_crests,
            min_separation=args.min_separation,
        )
        flat_specs, crest_case_ids = flatten_case_specs(case_specs)
        flat_amplitudes, flat_centers, flat_directions = specs_to_jax_arrays(flat_specs)
        crest_case_ids_arr = jnp.asarray(crest_case_ids)

        tanaka_batch = solve_modified_tanaka_batched(
            template_params,
            flat_amplitudes,
            centers=flat_centers,
            directions=flat_directions,
        )
        initial_eta = jnp.zeros((args.batch_size, args.nx), dtype=tanaka_batch.eta_periodic.dtype)
        initial_xi = jnp.zeros((args.batch_size, args.nx), dtype=tanaka_batch.xi_periodic.dtype)
        initial_eta = initial_eta.at[crest_case_ids_arr].add(tanaka_batch.eta_periodic)
        initial_xi = initial_xi.at[crest_case_ids_arr].add(tanaka_batch.xi_periodic)
        if rollout_defaults.zero_mean_xi:
            initial_xi = initial_xi - jnp.mean(initial_xi, axis=-1, keepdims=True)

        rollout_payload = batched_rollout(
            cast_state_dtype(State(eta=initial_eta, xi=initial_xi), rollout_dtype),
            rollout_times,
            solver_params,
            save_gxi=False,
            substeps_per_interval=rollout_defaults.substeps_per_interval,
            method=rollout_defaults.method,
            implicit_iterations=rollout_defaults.implicit_iterations,
            implicit_relaxation=rollout_defaults.implicit_relaxation,
            zero_mean_xi=rollout_defaults.zero_mean_xi,
        )
        jax.block_until_ready(rollout_payload["xi"])

        pred_eta = np.asarray(jax.device_get(rollout_payload["eta"][subsample_indices]), dtype=np.float32)
        pred_xi = np.asarray(jax.device_get(rollout_payload["xi"][subsample_indices]), dtype=np.float32)
        pred_gxi = chunked_gxi_over_time(
            jnp.asarray(pred_eta),
            jnp.asarray(pred_xi),
            solver_params,
            args.gxi_chunk_size,
        )

        global_case_ids = np.arange(
            args.case_id_offset + batch_idx * args.batch_size,
            args.case_id_offset + (batch_idx + 1) * args.batch_size,
            dtype=np.int64,
        )
        eta_samples, xi_samples, gxi_samples, time_samples, case_id_samples = flatten_samples(
            pred_eta,
            pred_xi,
            pred_gxi,
            subsample_times,
            global_case_ids,
        )

        remaining = args.target_samples - samples_written
        keep = min(remaining, eta_samples.shape[0])
        eta_samples = eta_samples[:keep]
        xi_samples = xi_samples[:keep]
        gxi_samples = gxi_samples[:keep]
        time_samples = time_samples[:keep]
        case_id_samples = case_id_samples[:keep]

        with zipfile.ZipFile(output_path, mode="a", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            batch_tag = f"batch_{batch_idx:04d}"
            write_npy_entry(zf, f"eta_{batch_tag}.npy", eta_samples)
            write_npy_entry(zf, f"xi_{batch_tag}.npy", xi_samples)
            write_npy_entry(zf, f"gxi_{batch_tag}.npy", gxi_samples)
            write_npy_entry(zf, f"time_{batch_tag}.npy", time_samples)
            write_npy_entry(zf, f"case_id_{batch_tag}.npy", case_id_samples)
            zf.writestr(f"specs_{batch_tag}.json", json.dumps(serialize_case_specs(case_specs), indent=2))

        samples_written += keep
        batch_seconds = perf_counter() - batch_start
        save_state(
            state_path,
            {
                "output_path": str(output_path),
                "samples_written": samples_written,
                "next_batch": batch_idx + 1,
                "n_batches_planned": n_batches,
                "complete": samples_written >= args.target_samples,
                "last_batch_seconds": batch_seconds,
                "elapsed_seconds": perf_counter() - total_start,
            },
        )
        print(
            json.dumps(
                {
                    "batch_idx": batch_idx,
                    "samples_written": samples_written,
                    "target_samples": args.target_samples,
                    "batch_seconds": batch_seconds,
                    "device": str(jax.devices()[0]),
                }
            ),
            flush=True,
        )
        if samples_written >= args.target_samples:
            break


if __name__ == "__main__":
    main()
