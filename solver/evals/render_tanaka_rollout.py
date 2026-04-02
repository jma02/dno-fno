from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from ..solvers.time_integrator import (
    State,
    cast_solver_params_dtype,
    cast_state_dtype,
    make_solver_params,
    rollout,
)
from ..tanaka_ICs.modified_tanaka import ModifiedTanakaParams, solve_modified_tanaka
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
    parser.add_argument("--dno_order", type=int, default=6)
    parser.add_argument("--pad_factor", type=int, default=8)
    parser.add_argument("--filter_fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--method", default="gl2_if")
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--implicit_iterations", type=int, default=4)
    parser.add_argument("--implicit_relaxation", type=float, default=1.0)
    parser.add_argument("--zero_mean_xi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--n_frames", type=int, default=200)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--grid_mode", choices=("auto", "manual"), default="manual")
    parser.add_argument("--collocation_points", type=int, default=257)
    parser.add_argument("--quadrature_substeps", type=int, default=4)
    parser.add_argument("--interpolation_degree", type=int, default=3)
    parser.add_argument("--s_max", type=float, default=2.5)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--transform_power", type=int, default=5)
    parser.add_argument("--qc_lower", type=float, default=0.2)
    parser.add_argument("--qc_upper", type=float, default=0.999)
    parser.add_argument("--outer_iterations", type=int, default=24)
    parser.add_argument("--fixed_point_iterations", type=int, default=80)
    parser.add_argument("--f2_tolerance", type=float, default=1e-10)
    parser.add_argument("--rollout_dtype", choices=("float32", "float64"), default="float32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tanaka_params = ModifiedTanakaParams(
        amplitude=args.amplitude,
        depth=args.depth,
        gravity=args.gravity,
        direction=args.direction,
        nx=args.nx,
        length=args.length,
        center=args.center,
        dno_order=args.dno_order,
        pad_factor=args.pad_factor,
        grid_mode=args.grid_mode,
        collocation_points=args.collocation_points,
        quadrature_substeps=args.quadrature_substeps,
        interpolation_degree=args.interpolation_degree,
        s_max=args.s_max,
        alpha=args.alpha,
        transform_power=args.transform_power,
        qc_lower=args.qc_lower,
        qc_upper=args.qc_upper,
        outer_iterations=args.outer_iterations,
        fixed_point_iterations=args.fixed_point_iterations,
        f2_tolerance=args.f2_tolerance,
    )
    tanaka_solution = solve_modified_tanaka(tanaka_params)

    solver_params = make_solver_params(
        nx=args.nx,
        length=args.length,
        depth=args.depth,
        gravity=args.gravity,
        dno_order=args.dno_order,
        pad_factor=args.pad_factor,
        filter_fraction=args.filter_fraction,
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
        substeps_per_interval=args.substeps,
        method=args.method,
        implicit_iterations=args.implicit_iterations,
        implicit_relaxation=args.implicit_relaxation,
        zero_mean_xi=args.zero_mean_xi,
    )
    runtime_seconds = perf_counter() - start

    stem = (
        f"tanaka_a{args.amplitude:.3f}_{args.method}"
        f"_sub{args.substeps}_t{args.tmax:.0f}_{args.rollout_dtype}"
        + ("_zero_mean_xi" if args.zero_mean_xi else "")
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
            f"tanaka a={args.amplitude:.3f} | {args.method} | "
            f"substeps={args.substeps}" + (" | xi-zero-mean" if args.zero_mean_xi else "")
        ),
        fps=args.fps,
        n_frames=args.n_frames,
        dpi=args.dpi,
    )

    summary = {
        "amplitude": args.amplitude,
        "direction": args.direction,
        "method": args.method,
        "substeps": args.substeps,
        "zero_mean_xi": args.zero_mean_xi,
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
