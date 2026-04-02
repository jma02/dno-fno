from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..gen_data.multi_crest import CrestSpec, build_multi_crest_initial_condition
from ..solvers.time_integrator import (
    State,
    batched_rollout,
    cast_solver_params_dtype,
    cast_state_dtype,
    make_solver_params,
)
from .render_rollout_movie import render_rollout_gif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample random Tanaka multicrest ICs, roll them out, and render GIFs with the rollout movie style.")
    parser.add_argument("--output_dir", default="outputs/tanaka_e2e")
    parser.add_argument("--n_cases", type=int, default=3)
    parser.add_argument("--min_crests", type=int, default=3)
    parser.add_argument("--max_crests", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amplitude_min", type=float, default=0.05)
    parser.add_argument("--amplitude_max", type=float, default=0.35)
    parser.add_argument("--min_separation", type=float, default=18.0)
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=164.0)
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=1.0)
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


def _periodic_distance(a: float, b: float, length: float) -> float:
    delta = abs(a - b)
    return min(delta, length - delta)


def _sample_centers(
    rng: np.random.Generator,
    n_centers: int,
    length: float,
    min_separation: float,
) -> list[float]:
    centers: list[float] = []
    for _ in range(10_000):
        if len(centers) == n_centers:
            break
        candidate = float(rng.uniform(0.0, length))
        if all(_periodic_distance(candidate, center, length) >= min_separation for center in centers):
            centers.append(candidate)
    if len(centers) != n_centers:
        raise RuntimeError("Failed to sample well-separated periodic centers.")
    centers.sort()
    return centers


def _sample_random_specs(
    rng: np.random.Generator,
    length: float,
    amplitude_min: float,
    amplitude_max: float,
    min_crests: int,
    max_crests: int,
    min_separation: float,
) -> list[CrestSpec]:
    n_crests = int(rng.integers(min_crests, max_crests + 1))
    centers = _sample_centers(rng, n_crests, length, min_separation)
    amplitudes = rng.uniform(amplitude_min, amplitude_max, size=n_crests)
    directions = rng.choice(np.array([-1, 1], dtype=int), size=n_crests)
    return [
        CrestSpec(
            amplitude=float(amplitude),
            center=float(center),
            direction=int(direction),
        )
        for amplitude, center, direction in zip(amplitudes, centers, directions)
    ]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    cases: list[dict[str, object]] = []
    for case_idx in range(args.n_cases):
        specs = _sample_random_specs(
            rng,
            length=args.length,
            amplitude_min=args.amplitude_min,
            amplitude_max=args.amplitude_max,
            min_crests=args.min_crests,
            max_crests=args.max_crests,
            min_separation=args.min_separation,
        )
        name = f"random_multicrest_{case_idx + 1:02d}"
        payload = build_multi_crest_initial_condition(
            specs,
            nx=args.nx,
            length=args.length,
            depth=args.depth,
            gravity=args.gravity,
            dno_order=args.dno_order,
            pad_factor=args.pad_factor,
            zero_mean_xi=args.zero_mean_xi,
            name=name,
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
        cases.append(payload)

    params = make_solver_params(
        nx=args.nx,
        length=args.length,
        depth=args.depth,
        gravity=args.gravity,
        dno_order=args.dno_order,
        pad_factor=args.pad_factor,
        filter_fraction=args.filter_fraction,
    )
    rollout_dtype = jnp.float32 if args.rollout_dtype == "float32" else jnp.float64
    params = cast_solver_params_dtype(params, rollout_dtype)
    times = jnp.arange(0.0, args.tmax + 0.5 * args.dt, args.dt, dtype=rollout_dtype)

    initial_eta = np.stack([np.asarray(case["eta"]) for case in cases], axis=0)
    initial_xi = np.stack([np.asarray(case["xi"]) for case in cases], axis=0)
    rollout_payload = batched_rollout(
        cast_state_dtype(State(eta=initial_eta, xi=initial_xi), rollout_dtype),
        times,
        params,
        save_gxi=True,
        substeps_per_interval=args.substeps,
        method=args.method,
        implicit_iterations=args.implicit_iterations,
        implicit_relaxation=args.implicit_relaxation,
        zero_mean_xi=args.zero_mean_xi,
    )

    pred_eta = np.asarray(rollout_payload["eta"])
    pred_xi = np.asarray(rollout_payload["xi"])
    pred_gxi = np.asarray(rollout_payload["gxi"])
    times_np = np.asarray(times)

    for case_idx, case in enumerate(cases):
        name = str(case["name"])
        stem = f"{name}_{args.method}_sub{args.substeps}_t{int(args.tmax)}_{args.rollout_dtype}"
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
            title=f"{name} | {args.method} | substeps={args.substeps}" + (" | xi-zero-mean" if args.zero_mean_xi else ""),
            fps=args.fps,
            n_frames=args.n_frames,
            dpi=args.dpi,
        )

        summary = {
            "name": name,
            "method": args.method,
            "substeps": args.substeps,
            "zero_mean_xi": args.zero_mean_xi,
            "rollout_dtype": args.rollout_dtype,
            "specs": [spec.__dict__ for spec in case["specs"]],
            "component_qc": np.asarray(case["component_qc"]).tolist(),
            "component_speed": np.asarray(case["component_speed"]).tolist(),
            "npz_path": str(npz_path),
            "gif_path": str(gif_path),
        }
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
