from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .modified_tanaka import (
    DEFAULT_OUTER_ITERATIONS,
    DEFAULT_QC_UPPER,
    make_default_tanaka_template,
    solve_modified_tanaka_batched,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and plot a batched Tanaka solve.")
    parser.add_argument("--output_dir", default="outputs/tanaka_batch")
    parser.add_argument(
        "--mode", choices=("centers", "amplitudes"), default="amplitudes"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--amplitude", type=float, default=0.1)
    parser.add_argument("--amplitude_min", type=float, default=0.05)
    parser.add_argument("--amplitude_max", type=float, default=0.4)
    parser.add_argument("--direction", type=int, default=1, choices=(-1, 1))
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=164.0)
    parser.add_argument("--center", type=float, default=82.0)
    parser.add_argument("--dno_order", type=int, default=6)
    parser.add_argument("--pad_factor", type=int, default=8)
    parser.add_argument("--grid_mode", choices=("auto", "manual"), default="manual")
    parser.add_argument("--collocation_points", type=int, default=257)
    parser.add_argument("--quadrature_substeps", type=int, default=4)
    parser.add_argument("--interpolation_degree", type=int, default=3)
    parser.add_argument("--s_max", type=float, default=2.5)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--transform_power", type=int, default=5)
    parser.add_argument("--qc_lower", type=float, default=0.2)
    parser.add_argument("--qc_upper", type=float, default=DEFAULT_QC_UPPER)
    parser.add_argument(
        "--outer_iterations",
        type=int,
        default=DEFAULT_OUTER_ITERATIONS,
    )
    parser.add_argument("--fixed_point_iterations", type=int, default=80)
    parser.add_argument("--f2_tolerance", type=float, default=1e-10)
    parser.add_argument("--sample_count", type=int, default=8)
    return parser.parse_args()


def _block(solution: object) -> None:
    for leaf in jax.tree_util.tree_leaves(solution):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _save_field_figure(
    x: np.ndarray,
    field: np.ndarray,
    varying_values: np.ndarray,
    sample_indices: np.ndarray,
    ylabel: str,
    value_label: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5))
    colors = matplotlib.colormaps["viridis"](
        np.linspace(0.1, 0.95, sample_indices.size)
    )
    for color, idx in zip(colors, sample_indices):
        label = f"#{idx}  {value_label}={varying_values[idx]:.3f}"
        ax.plot(x, field[idx], color=color, linewidth=1.8, label=label)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("x")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    params = make_default_tanaka_template(
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

    if args.mode == "centers":
        centers = jnp.linspace(
            0.0,
            params.length - params.length / params.nx,
            args.batch_size,
            dtype=jnp.float64,
        )
        amplitudes = jnp.full((args.batch_size,), args.amplitude, dtype=jnp.float64)
        varying_values = np.asarray(centers)
        value_label = "x0"
    else:
        amplitudes = jnp.linspace(
            args.amplitude_min, args.amplitude_max, args.batch_size, dtype=jnp.float64
        )
        centers = jnp.full((args.batch_size,), params.center, dtype=jnp.float64)
        varying_values = np.asarray(amplitudes)
        value_label = "a"
    directions = jnp.full((args.batch_size,), args.direction, dtype=jnp.float64)

    start = time.perf_counter()
    batch = solve_modified_tanaka_batched(
        params, amplitudes, centers=centers, directions=directions
    )
    _block(batch)
    cold_seconds = time.perf_counter() - start

    start = time.perf_counter()
    batch = solve_modified_tanaka_batched(
        params, amplitudes, centers=centers, directions=directions
    )
    _block(batch)
    warm_seconds = time.perf_counter() - start

    x = np.asarray(batch.x_periodic)
    centers_np = np.asarray(centers)
    eta = np.asarray(batch.eta_periodic)
    xi = np.asarray(batch.xi_periodic)
    gxi = np.asarray(batch.gxi_periodic)
    qc = np.asarray(batch.qc)
    speed = np.asarray(batch.speed)

    sample_count = min(args.sample_count, args.batch_size)
    sample_indices = np.linspace(0, args.batch_size - 1, sample_count, dtype=int)

    np.savez_compressed(
        output_dir / "tanaka_batch_128.npz",
        x=x,
        centers=centers_np,
        amplitudes=np.asarray(amplitudes),
        eta=eta,
        xi=xi,
        gxi=gxi,
        qc=qc,
        speed=speed,
        cold_seconds=float(cold_seconds),
        warm_seconds=float(warm_seconds),
        warm_per_wave_seconds=float(warm_seconds / args.batch_size),
        params_json=json.dumps(vars(args)),
        devices_json=json.dumps([str(device) for device in jax.devices()]),
    )

    summary = {
        "devices": [str(device) for device in jax.devices()],
        "mode": args.mode,
        "batch_size": int(args.batch_size),
        "amplitude": float(args.amplitude),
        "amplitude_min": float(args.amplitude_min),
        "amplitude_max": float(args.amplitude_max),
        "cold_seconds": float(cold_seconds),
        "warm_seconds": float(warm_seconds),
        "warm_per_wave_seconds": float(warm_seconds / args.batch_size),
        "qc_first": float(qc[0]),
        "qc_last": float(qc[-1]),
        "eta_shape": list(eta.shape),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    _save_field_figure(
        x,
        eta,
        varying_values,
        sample_indices,
        r"$\eta$",
        value_label,
        output_dir / "eta_lines.png",
    )
    _save_field_figure(
        x,
        xi,
        varying_values,
        sample_indices,
        r"$\xi$",
        value_label,
        output_dir / "xi_lines.png",
    )
    _save_field_figure(
        x,
        gxi,
        varying_values,
        sample_indices,
        r"$G(\eta)\xi$",
        value_label,
        output_dir / "gxi_lines.png",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
