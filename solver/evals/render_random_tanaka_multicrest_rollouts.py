from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..gen_data.multi_crest import (
    DEFAULT_RANDOM_MAX_CRESTS,
    DEFAULT_RANDOM_MIN_CRESTS,
    DEFAULT_RANDOM_MIN_SEPARATION,
    build_multi_crest_initial_condition,
    sample_random_cases,
    serialize_specs,
    stack_case_field,
)
from ..solvers.time_integrator import (
    State,
    batched_rollout,
    cast_solver_params_dtype,
    cast_state_dtype,
    make_normalized_rollout_settings,
    make_solver_params,
)
from .render_rollout_movie import render_rollout_gif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample random Tanaka multicrest ICs, roll them out, and render GIFs with the rollout movie style.")
    parser.add_argument("--output_dir", default="outputs/tanaka_e2e")
    parser.add_argument("--n_cases", type=int, default=3)
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
    cases = [
        build_multi_crest_initial_condition(
            specs,
            nx=args.nx,
            length=args.length,
            depth=args.depth,
            gravity=args.gravity,
            dno_order=rollout_defaults.dno_order,
            pad_factor=rollout_defaults.pad_factor,
            zero_mean_xi=rollout_defaults.zero_mean_xi,
            name=f"random_multicrest_{case_idx + 1:02d}",
        )
        for case_idx, specs in enumerate(case_specs)
    ]

    params = make_solver_params(
        nx=args.nx,
        length=args.length,
        depth=args.depth,
        gravity=args.gravity,
        dno_order=rollout_defaults.dno_order,
        pad_factor=rollout_defaults.pad_factor,
        filter_fraction=rollout_defaults.filter_fraction,
    )
    rollout_dtype = jnp.float32 if args.rollout_dtype == "float32" else jnp.float64
    params = cast_solver_params_dtype(params, rollout_dtype)
    times = jnp.arange(0.0, args.tmax + 0.5 * args.dt, args.dt, dtype=rollout_dtype)

    initial_eta = stack_case_field(cases, "eta")
    initial_xi = stack_case_field(cases, "xi")
    rollout_payload = batched_rollout(
        cast_state_dtype(State(eta=initial_eta, xi=initial_xi), rollout_dtype),
        times,
        params,
        save_gxi=True,
        substeps_per_interval=rollout_defaults.substeps_per_interval,
        method=rollout_defaults.method,
        implicit_iterations=rollout_defaults.implicit_iterations,
        implicit_relaxation=rollout_defaults.implicit_relaxation,
        zero_mean_xi=rollout_defaults.zero_mean_xi,
    )

    pred_eta = np.asarray(rollout_payload["eta"])
    pred_xi = np.asarray(rollout_payload["xi"])
    pred_gxi = np.asarray(rollout_payload["gxi"])
    times_np = np.asarray(times)

    for case_idx, case in enumerate(cases):
        name = str(case["name"])
        stem = f"{name}_{rollout_defaults.method}_sub{rollout_defaults.substeps_per_interval}_t{int(args.tmax)}_{args.rollout_dtype}"
        npz_path = output_dir / f"{stem}.npz"
        gif_path = output_dir / f"{stem}.gif"
        json_path = output_dir / f"{stem}.json"

        single_payload = {
            "x": np.asarray(case["x"]),
            "t": times_np,
            "pred_eta": pred_eta[:, case_idx],
            "pred_xi": pred_xi[:, case_idx],
            "pred_gxi": pred_gxi[:, case_idx],
            "eta0": np.asarray(case["eta"]),
            "xi0": np.asarray(case["xi"]),
            "gxi0": np.asarray(case["gxi"]),
        }
        np.savez_compressed(npz_path, **single_payload)
        render_rollout_gif(
            single_payload,
            output_path=gif_path,
            title=f"{name} | {rollout_defaults.method} | substeps={rollout_defaults.substeps_per_interval}" + (" | xi-zero-mean" if rollout_defaults.zero_mean_xi else ""),
            fps=args.fps,
            n_frames=args.n_frames,
            dpi=args.dpi,
        )

        summary = {
            "name": name,
            "method": rollout_defaults.method,
            "substeps": rollout_defaults.substeps_per_interval,
            "zero_mean_xi": rollout_defaults.zero_mean_xi,
            "rollout_dtype": args.rollout_dtype,
            "specs": serialize_specs(case["specs"]),
            "component_qc": np.asarray(case["component_qc"]).tolist(),
            "component_speed": np.asarray(case["component_speed"]).tolist(),
            "npz_path": str(npz_path),
            "gif_path": str(gif_path),
        }
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
