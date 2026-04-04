from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import replace

from ..solvers.time_integrator import (
    State,
    cast_solver_params_dtype,
    cast_state_dtype,
    make_normalized_rollout_settings,
    make_solver_params,
    rollout,
)
from ..tanaka_ICs.modified_tanaka import make_default_tanaka_template, solve_modified_tanaka
from .render_rollout_movie import render_rollout_gif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Tanaka initial condition, roll it out, and render it with the rollout movie style.")
    parser.add_argument("--output_dir", default="outputs/tanaka_e2e")
    parser.add_argument("--amplitude", type=float, default=0.2)
    parser.add_argument("--direction", type=int, default=1, choices=(-1, 1))
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=164.0)
    parser.add_argument("--center", type=float, default=43.5625)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--tmax", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--n_frames", type=int, default=200)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--rollout_dtype", choices=("float32", "float64"), default="float32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollout_defaults = make_normalized_rollout_settings()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tanaka_params = replace(
        make_default_tanaka_template(
            depth=args.depth,
            gravity=args.gravity,
            nx=args.nx,
            length=args.length,
            dno_order=rollout_defaults.dno_order,
            pad_factor=rollout_defaults.pad_factor,
        ),
        depth=args.depth,
        direction=args.direction,
        amplitude=args.amplitude,
        center=args.center,
    )
    tanaka_solution = solve_modified_tanaka(tanaka_params)

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

    times = jnp.arange(0.0, args.tmax + 0.5 * args.dt, args.dt, dtype=rollout_dtype)
    initial_state = cast_state_dtype(
        State(eta=tanaka_solution.eta_periodic, xi=tanaka_solution.xi_periodic),
        rollout_dtype,
    )

    start = perf_counter()
    rollout_payload = rollout(
        initial_state,
        times,
        solver_params,
        save_gxi=True,
        substeps_per_interval=rollout_defaults.substeps_per_interval,
        method=rollout_defaults.method,
        implicit_iterations=rollout_defaults.implicit_iterations,
        implicit_relaxation=rollout_defaults.implicit_relaxation,
        zero_mean_xi=rollout_defaults.zero_mean_xi,
    )
    runtime_seconds = perf_counter() - start

    stem = (
        f"tanaka_a{args.amplitude:.3f}_{rollout_defaults.method}"
        f"_sub{rollout_defaults.substeps_per_interval}_t{args.tmax:.0f}_{args.rollout_dtype}"
        + ("_zero_mean_xi" if rollout_defaults.zero_mean_xi else "")
    )
    npz_path = output_dir / f"{stem}.npz"
    gif_path = output_dir / f"{stem}.gif"
    json_path = output_dir / f"{stem}.json"

    payload = {
        "x": np.asarray(tanaka_solution.x_periodic),
        "t": np.asarray(times),
        "pred_eta": np.asarray(rollout_payload["eta"]),
        "pred_xi": np.asarray(rollout_payload["xi"]),
        "pred_gxi": np.asarray(rollout_payload["gxi"]),
        "eta0": np.asarray(tanaka_solution.eta_periodic),
        "xi0": np.asarray(tanaka_solution.xi_periodic),
        "gxi0": np.asarray(tanaka_solution.gxi_periodic),
    }
    np.savez_compressed(npz_path, **payload)

    render_rollout_gif(
        payload,
        output_path=gif_path,
        title=(
            f"tanaka a={args.amplitude:.3f} | {rollout_defaults.method} | "
            f"substeps={rollout_defaults.substeps_per_interval}" + (" | xi-zero-mean" if rollout_defaults.zero_mean_xi else "")
        ),
        fps=args.fps,
        n_frames=args.n_frames,
        dpi=args.dpi,
    )

    summary = {
        "amplitude": args.amplitude,
        "direction": args.direction,
        "method": rollout_defaults.method,
        "substeps": rollout_defaults.substeps_per_interval,
        "zero_mean_xi": rollout_defaults.zero_mean_xi,
        "rollout_dtype": args.rollout_dtype,
        "runtime_seconds": runtime_seconds,
        "qc": float(tanaka_solution.qc),
        "froude": float(tanaka_solution.froude),
        "speed": float(tanaka_solution.speed),
        "rollout_device": str(jax.devices()[0]),
        "npz_path": str(npz_path),
        "gif_path": str(gif_path),
        "tanaka_params_json": json.dumps(vars(args)),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
