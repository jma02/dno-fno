from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from ..solvers.dno_series_jax import dno_series_eval
from ..solvers.time_integrator import (
    State,
    batched_rollout,
    cast_solver_params_dtype,
    cast_state_dtype,
    make_solver_params,
)
from ..tanaka_ICs.modified_tanaka import ModifiedTanakaParams, solve_modified_tanaka_batched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark batched fp64 Tanaka IC generation plus fp32 batched rollout for random multicrest states.")
    parser.add_argument("--output_dir", default="outputs/tanaka_batch_rollouts")
    parser.add_argument("--n_cases", type=int, default=128)
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
    parser.add_argument("--tmax", type=float, default=200.0)
    parser.add_argument("--dno_order", type=int, default=6)
    parser.add_argument("--pad_factor", type=int, default=8)
    parser.add_argument("--filter_fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--method", default="gl2_if")
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--implicit_iterations", type=int, default=4)
    parser.add_argument("--implicit_relaxation", type=float, default=1.0)
    parser.add_argument("--zero_mean_xi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rollout_dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--save_gxi", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gxi_chunk_size", type=int, default=32)
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
    parser.add_argument("--sample_cases_to_save", type=int, default=8)
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


def _sample_case(
    rng: np.random.Generator,
    length: float,
    amplitude_min: float,
    amplitude_max: float,
    min_crests: int,
    max_crests: int,
    min_separation: float,
) -> list[dict[str, float | int]]:
    n_crests = int(rng.integers(min_crests, max_crests + 1))
    centers = _sample_centers(rng, n_crests, length, min_separation)
    amplitudes = rng.uniform(amplitude_min, amplitude_max, size=n_crests)
    directions = rng.choice(np.array([-1, 1], dtype=int), size=n_crests)
    return [
        {
            "amplitude": float(amplitude),
            "center": float(center),
            "direction": int(direction),
        }
        for amplitude, center, direction in zip(amplitudes, centers, directions)
    ]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    case_specs = [
        _sample_case(
            rng,
            length=args.length,
            amplitude_min=args.amplitude_min,
            amplitude_max=args.amplitude_max,
            min_crests=args.min_crests,
            max_crests=args.max_crests,
            min_separation=args.min_separation,
        )
        for _ in range(args.n_cases)
    ]

    flat_specs: list[dict[str, float | int]] = []
    crest_case_ids: list[int] = []
    for case_idx, specs in enumerate(case_specs):
        for spec in specs:
            flat_specs.append(spec)
            crest_case_ids.append(case_idx)

    template_params = ModifiedTanakaParams(
        amplitude=0.0,
        depth=args.depth,
        gravity=args.gravity,
        direction=1,
        nx=args.nx,
        length=args.length,
        center=0.0,
        dno_order=args.dno_order,
        pad_factor=args.pad_factor,
        grid_mode="manual",
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

    flat_amplitudes = jnp.asarray([spec["amplitude"] for spec in flat_specs], dtype=jnp.float64)
    flat_centers = jnp.asarray([spec["center"] for spec in flat_specs], dtype=jnp.float64)
    flat_directions = jnp.asarray([spec["direction"] for spec in flat_specs], dtype=jnp.float64)
    crest_case_ids_arr = jnp.asarray(crest_case_ids, dtype=jnp.int32)

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
    if args.zero_mean_xi:
        initial_xi = initial_xi - jnp.mean(initial_xi, axis=-1, keepdims=True)

    dno_params = make_solver_params(
        nx=args.nx,
        length=args.length,
        depth=args.depth,
        gravity=args.gravity,
        dno_order=args.dno_order,
        pad_factor=args.pad_factor,
        filter_fraction=args.filter_fraction,
    )
    initial_gxi = dno_series_eval(
        initial_eta,
        initial_xi,
        dno_params.k.astype(initial_eta.dtype),
        args.depth,
        args.dno_order,
        pad_factor=args.pad_factor,
    )
    jax.block_until_ready(initial_gxi)
    ic_seconds = perf_counter() - ic_start

    rollout_dtype = jnp.float32 if args.rollout_dtype == "float32" else jnp.float64
    rollout_params = dno_params
    rollout_params = cast_solver_params_dtype(rollout_params, rollout_dtype)
    times = jnp.arange(0.0, args.tmax + 0.5 * args.dt, args.dt, dtype=rollout_dtype)
    rollout_state = cast_state_dtype(State(eta=initial_eta, xi=initial_xi), rollout_dtype)

    rollout_start = perf_counter()
    rollout_payload = batched_rollout(
        rollout_state,
        times,
        rollout_params,
        save_gxi=False,
        substeps_per_interval=args.substeps,
        method=args.method,
        implicit_iterations=args.implicit_iterations,
        implicit_relaxation=args.implicit_relaxation,
        zero_mean_xi=args.zero_mean_xi,
    )
    jax.block_until_ready(rollout_payload["xi"])
    rollout_seconds = perf_counter() - rollout_start

    total_seconds = ic_seconds + rollout_seconds
    sample_count = min(args.sample_cases_to_save, args.n_cases)
    sample_indices = np.arange(sample_count, dtype=np.int32)

    gxi_seconds = 0.0
    pred_gxi_samples = None
    if args.save_gxi:
        gxi_start = perf_counter()
        gxi_chunks: list[np.ndarray] = []
        for start_idx in range(0, int(times.shape[0]), args.gxi_chunk_size):
            end_idx = min(start_idx + args.gxi_chunk_size, int(times.shape[0]))
            gxi_chunk = dno_series_eval(
                rollout_payload["eta"][start_idx:end_idx],
                rollout_payload["xi"][start_idx:end_idx],
                rollout_params.k,
                rollout_params.depth,
                rollout_params.dno_order,
                pad_factor=rollout_params.pad_factor,
            )
            gxi_chunk = jax.device_get(gxi_chunk[:, sample_indices])
            gxi_chunks.append(np.asarray(gxi_chunk))
        pred_gxi_samples = np.concatenate(gxi_chunks, axis=0)
        gxi_seconds = perf_counter() - gxi_start

    sample_payload = {
        "times": np.asarray(times),
        "sample_indices": sample_indices,
        "initial_eta": np.asarray(initial_eta[sample_indices]),
        "initial_xi": np.asarray(initial_xi[sample_indices]),
        "initial_gxi": np.asarray(initial_gxi[sample_indices]),
        "pred_eta": np.asarray(rollout_payload["eta"][:, sample_indices]),
        "pred_xi": np.asarray(rollout_payload["xi"][:, sample_indices]),
        "component_qc": np.asarray(tanaka_batch.qc),
        "component_speed": np.asarray(tanaka_batch.speed),
        "crest_case_ids": np.asarray(crest_case_ids_arr),
    }
    if args.save_gxi:
        sample_payload["pred_gxi"] = pred_gxi_samples

    np.savez_compressed(
        output_dir / "sample_payload.npz",
        **sample_payload,
    )

    (output_dir / "case_specs.json").write_text(json.dumps(case_specs, indent=2), encoding="utf-8")

    crest_counts = np.asarray([len(specs) for specs in case_specs], dtype=np.int32)
    summary = {
        "n_cases": args.n_cases,
        "n_total_crests": int(len(flat_specs)),
        "mean_crests_per_case": float(np.mean(crest_counts)),
        "min_crests_per_case": int(np.min(crest_counts)),
        "max_crests_per_case": int(np.max(crest_counts)),
        "tanaka_dtype": "float64",
        "rollout_dtype": args.rollout_dtype,
        "device": str(jax.devices()[0]),
        "dt": args.dt,
        "tmax": args.tmax,
        "n_time_samples": int(times.shape[0]),
        "method": args.method,
        "substeps": args.substeps,
        "implicit_iterations": args.implicit_iterations,
        "filter_fraction": args.filter_fraction,
        "zero_mean_xi": args.zero_mean_xi,
        "save_gxi": args.save_gxi,
        "gxi_chunk_size": args.gxi_chunk_size,
        "ic_seconds": ic_seconds,
        "ic_seconds_per_case": ic_seconds / args.n_cases,
        "rollout_seconds": rollout_seconds,
        "rollout_seconds_per_case": rollout_seconds / args.n_cases,
        "gxi_seconds": gxi_seconds,
        "total_seconds": total_seconds + gxi_seconds,
        "total_seconds_per_case": (total_seconds + gxi_seconds) / args.n_cases,
        "sample_payload_path": str(output_dir / "sample_payload.npz"),
        "case_specs_path": str(output_dir / "case_specs.json"),
        "benchmark_args_json": json.dumps(vars(args)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
