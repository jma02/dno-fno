from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

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
from .render_rollout_movie import render_rollout_gif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a batched random Tanaka multicrest rollout, save the full trajectory, and render selected GIFs.")
    parser.add_argument("--output_dir", default="outputs/tanaka_batch_e2e")
    parser.add_argument("--n_cases", type=int, default=128)
    parser.add_argument("--render_stride", type=int, default=16)
    parser.add_argument("--min_crests", type=int, default=DEFAULT_RANDOM_MIN_CRESTS)
    parser.add_argument("--max_crests", type=int, default=DEFAULT_RANDOM_MAX_CRESTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amplitude_min", type=float, default=0.05)
    parser.add_argument("--amplitude_max", type=float, default=0.35)
    parser.add_argument("--min_separation", type=float, default=DEFAULT_RANDOM_MIN_SEPARATION)
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=164.0)
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--tmax", type=float, default=200.0)
    parser.add_argument("--rollout_dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--gxi_chunk_size", type=int, default=32)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--n_frames", type=int, default=200)
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


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
        chunks.append(np.asarray(jax.device_get(gxi_chunk)))
    return np.concatenate(chunks, axis=0)


def main() -> None:
    args = parse_args()
    rollout_defaults = make_normalized_rollout_settings()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gifs_dir = output_dir / "gifs"
    gifs_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    case_specs = sample_random_cases(
        rng,
        args.n_cases,
        length=args.length,
        amplitude_min=args.amplitude_min,
        amplitude_max=args.amplitude_max,
        min_crests=args.min_crests,
        max_crests=args.max_crests,
        min_separation=args.min_separation,
    )
    flat_specs, crest_case_ids = flatten_case_specs(case_specs)

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
    flat_amplitudes, flat_centers, flat_directions = specs_to_jax_arrays(flat_specs)
    crest_case_ids_arr = jnp.asarray(crest_case_ids)

    ic_start = perf_counter()
    tanaka_batch = solve_modified_tanaka_batched(
        template_params,
        flat_amplitudes,
        centers=flat_centers,
        directions=flat_directions,
    )
    initial_eta = jnp.zeros((args.n_cases, args.nx), dtype=tanaka_batch.eta_periodic.dtype)
    initial_xi = jnp.zeros((args.n_cases, args.nx), dtype=tanaka_batch.xi_periodic.dtype)
    initial_eta = initial_eta.at[crest_case_ids_arr].add(tanaka_batch.eta_periodic)
    initial_xi = initial_xi.at[crest_case_ids_arr].add(tanaka_batch.xi_periodic)
    if rollout_defaults.zero_mean_xi:
        initial_xi = initial_xi - jnp.mean(initial_xi, axis=-1, keepdims=True)

    solver_params = make_solver_params(
        nx=args.nx,
        length=args.length,
        depth=args.depth,
        gravity=args.gravity,
        dno_order=rollout_defaults.dno_order,
        pad_factor=rollout_defaults.pad_factor,
        filter_fraction=rollout_defaults.filter_fraction,
    )
    initial_gxi = dno_series_eval(
        initial_eta,
        initial_xi,
        solver_params.k.astype(initial_eta.dtype),
        args.depth,
        rollout_defaults.dno_order,
        pad_factor=rollout_defaults.pad_factor,
    )
    jax.block_until_ready(initial_gxi)
    ic_seconds = perf_counter() - ic_start

    rollout_dtype = jnp.float32 if args.rollout_dtype == "float32" else jnp.float64
    rollout_params = cast_solver_params_dtype(solver_params, rollout_dtype)
    times = jnp.arange(0.0, args.tmax + 0.5 * args.dt, args.dt, dtype=rollout_dtype)
    rollout_state = cast_state_dtype(State(eta=initial_eta, xi=initial_xi), rollout_dtype)

    rollout_start = perf_counter()
    rollout_payload = batched_rollout(
        rollout_state,
        times,
        rollout_params,
        save_gxi=False,
        substeps_per_interval=rollout_defaults.substeps_per_interval,
        method=rollout_defaults.method,
        implicit_iterations=rollout_defaults.implicit_iterations,
        implicit_relaxation=rollout_defaults.implicit_relaxation,
        zero_mean_xi=rollout_defaults.zero_mean_xi,
    )
    jax.block_until_ready(rollout_payload["xi"])
    rollout_seconds = perf_counter() - rollout_start

    gxi_start = perf_counter()
    pred_gxi = chunked_gxi_over_time(
        rollout_payload["eta"],
        rollout_payload["xi"],
        rollout_params,
        args.gxi_chunk_size,
    )
    gxi_seconds = perf_counter() - gxi_start

    save_start = perf_counter()
    x, _ = build_grid(args.nx, args.length)
    full_payload_path = output_dir / "full_trajectory.npz"
    pred_eta = np.asarray(jax.device_get(rollout_payload["eta"]))
    pred_xi = np.asarray(jax.device_get(rollout_payload["xi"]))
    np.savez(
        full_payload_path,
        x=np.asarray(x),
        times=np.asarray(times),
        initial_eta=np.asarray(jax.device_get(initial_eta)),
        initial_xi=np.asarray(jax.device_get(initial_xi)),
        initial_gxi=np.asarray(jax.device_get(initial_gxi)),
        pred_eta=pred_eta,
        pred_xi=pred_xi,
        pred_gxi=pred_gxi,
        component_qc=np.asarray(jax.device_get(tanaka_batch.qc)),
        component_speed=np.asarray(jax.device_get(tanaka_batch.speed)),
        crest_case_ids=np.asarray(crest_case_ids_arr),
    )
    save_seconds = perf_counter() - save_start

    render_indices = list(range(0, args.n_cases, args.render_stride))
    render_start = perf_counter()
    times_np = np.asarray(times)
    x_np = np.asarray(x)
    initial_eta_np = np.asarray(jax.device_get(initial_eta))
    initial_xi_np = np.asarray(jax.device_get(initial_xi))
    initial_gxi_np = np.asarray(jax.device_get(initial_gxi))
    gif_paths: list[str] = []
    for case_idx in render_indices:
        stem = f"random_multicrest_{case_idx:03d}_{rollout_defaults.method}_sub{rollout_defaults.substeps_per_interval}_t{int(args.tmax)}_{args.rollout_dtype}"
        per_case_npz = gifs_dir / f"{stem}.npz"
        gif_path = gifs_dir / f"{stem}.gif"
        payload = {
            "x": x_np,
            "t": times_np,
            "pred_eta": pred_eta[:, case_idx],
            "pred_xi": pred_xi[:, case_idx],
            "pred_gxi": pred_gxi[:, case_idx],
            "eta0": initial_eta_np[case_idx],
            "xi0": initial_xi_np[case_idx],
            "gxi0": initial_gxi_np[case_idx],
        }
        np.savez_compressed(per_case_npz, **payload)
        render_rollout_gif(
            payload,
            output_path=gif_path,
            title=f"random_multicrest_{case_idx:03d} | {rollout_defaults.method} | substeps={rollout_defaults.substeps_per_interval}"
            + (" | xi-zero-mean" if rollout_defaults.zero_mean_xi else ""),
            fps=args.fps,
            n_frames=args.n_frames,
            dpi=args.dpi,
        )
        gif_paths.append(str(gif_path))
    render_seconds = perf_counter() - render_start

    (output_dir / "case_specs.json").write_text(
        json.dumps(serialize_case_specs(case_specs), indent=2),
        encoding="utf-8",
    )

    summary = {
        "n_cases": args.n_cases,
        "n_total_crests": int(len(flat_specs)),
        "tanaka_dtype": "float64",
        "rollout_dtype": args.rollout_dtype,
        "device": str(jax.devices()[0]),
        "dt": args.dt,
        "tmax": args.tmax,
        "n_time_samples": int(times.shape[0]),
        "method": rollout_defaults.method,
        "substeps": rollout_defaults.substeps_per_interval,
        "implicit_iterations": rollout_defaults.implicit_iterations,
        "filter_fraction": rollout_defaults.filter_fraction,
        "zero_mean_xi": rollout_defaults.zero_mean_xi,
        "gxi_chunk_size": args.gxi_chunk_size,
        "render_stride": args.render_stride,
        "render_indices": render_indices,
        "ic_seconds": ic_seconds,
        "rollout_seconds": rollout_seconds,
        "gxi_seconds": gxi_seconds,
        "save_seconds": save_seconds,
        "render_seconds": render_seconds,
        "total_seconds": ic_seconds + rollout_seconds + gxi_seconds + save_seconds + render_seconds,
        "full_payload_path": str(full_payload_path),
        "case_specs_path": str(output_dir / "case_specs.json"),
        "gif_paths": gif_paths,
        "args_json": json.dumps(vars(args)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
