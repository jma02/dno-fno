from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..data.solitary_loader_jax import DEFAULT_SOLITON_ROOT, load_soliton_dataset
from .modified_tanaka import REAL_DTYPE, ModifiedTanakaParams, solve_modified_tanaka_batched
import jax.numpy as jnp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a canonical Tanaka solitary-wave branch sampled uniformly in amplitude.")
    parser.add_argument("--soliton_root", default=str(DEFAULT_SOLITON_ROOT))
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", default="outputs/tanaka_branch/tanaka_branch_uniform.npz")
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--direction", type=int, default=1, choices=(-1, 1))
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
    parser.add_argument("--qc_upper", type=float, default=0.999)
    parser.add_argument("--outer_iterations", type=int, default=24)
    parser.add_argument("--fixed_point_iterations", type=int, default=80)
    parser.add_argument("--f2_tolerance", type=float, default=1e-10)
    return parser.parse_args()


def _dataset_amplitude_range(soliton_root: str | Path) -> tuple[float, float]:
    trajectories = load_soliton_dataset(soliton_root)
    amplitudes = [float(np.max(np.asarray(traj["eta"][0], dtype=np.float64))) for traj in trajectories]
    return min(amplitudes), max(amplitudes)


def main() -> None:
    args = parse_args()
    amp_min, amp_max = _dataset_amplitude_range(args.soliton_root)
    amplitudes = np.linspace(amp_min, amp_max, args.num_samples, dtype=np.float64)

    base_params = ModifiedTanakaParams(
        amplitude=float(amplitudes[0]),
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

    batches = []
    for start in range(0, amplitudes.shape[0], args.batch_size):
        amplitude_batch = jnp.asarray(amplitudes[start : start + args.batch_size], dtype=REAL_DTYPE)
        batches.append(solve_modified_tanaka_batched(base_params, amplitude_batch))

    def cat(name: str) -> np.ndarray:
        return np.concatenate([np.asarray(getattr(batch, name), dtype=np.float64) for batch in batches], axis=0)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        amplitudes=np.asarray(amplitudes, dtype=np.float64),
        qc=cat("qc"),
        froude=cat("froude"),
        speed=cat("speed"),
        x=np.asarray(batches[0].x_periodic, dtype=np.float64),
        eta=cat("eta_periodic"),
        xi=cat("xi_periodic"),
        gxi=cat("gxi_periodic"),
        phi=np.asarray(batches[0].phi_profile, dtype=np.float64),
        tau=cat("tau_profile"),
        q=cat("q_profile"),
        theta=cat("theta_profile"),
        amplitude_min=float(amp_min),
        amplitude_max=float(amp_max),
        params_json=json.dumps(asdict(base_params)),
    )
    print(output_path)


if __name__ == "__main__":
    main()
