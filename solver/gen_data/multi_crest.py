from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..solvers.dno_series_jax import build_grid, dno_series_eval
from ..tanaka_ICs.modified_tanaka import ModifiedTanakaParams, solve_modified_tanaka_batched


@dataclass(frozen=True)
class CrestSpec:
    amplitude: float
    center: float
    direction: int


def available_reference_amplitudes() -> list[float]:
    amplitudes = {
        spec.amplitude
        for preset_name in ("head_on_2v2", "head_on_3v2", "head_on_3v3")
        for spec in make_preset_specs(preset_name)
    }
    return sorted(amplitudes)


def _build_tanaka_template_params(
    nx: int,
    length: float,
    depth: float,
    gravity: float,
    dno_order: int,
    pad_factor: int,
    collocation_points: int,
    quadrature_substeps: int,
    interpolation_degree: int,
    s_max: float,
    alpha: float,
    transform_power: int,
    qc_lower: float,
    qc_upper: float,
    outer_iterations: int,
    fixed_point_iterations: int,
    f2_tolerance: float,
) -> ModifiedTanakaParams:
    return ModifiedTanakaParams(
        amplitude=0.0,
        depth=depth,
        gravity=gravity,
        direction=1,
        nx=nx,
        length=length,
        center=0.0,
        dno_order=dno_order,
        pad_factor=pad_factor,
        grid_mode="manual",
        collocation_points=collocation_points,
        quadrature_substeps=quadrature_substeps,
        interpolation_degree=interpolation_degree,
        s_max=s_max,
        alpha=alpha,
        transform_power=transform_power,
        qc_lower=qc_lower,
        qc_upper=qc_upper,
        outer_iterations=outer_iterations,
        fixed_point_iterations=fixed_point_iterations,
        f2_tolerance=f2_tolerance,
    )


def build_multi_crest_initial_condition(
    specs: list[CrestSpec],
    nx: int = 1024,
    length: float = 164.0,
    depth: float = 1.0,
    gravity: float = 1.0,
    dno_order: int = 6,
    pad_factor: int = 8,
    zero_mean_xi: bool = True,
    name: str = "multi_crest",
    collocation_points: int = 257,
    quadrature_substeps: int = 4,
    interpolation_degree: int = 3,
    s_max: float = 2.5,
    alpha: float = 0.01,
    transform_power: int = 5,
    qc_lower: float = 0.2,
    qc_upper: float = 0.999,
    outer_iterations: int = 24,
    fixed_point_iterations: int = 80,
    f2_tolerance: float = 1e-10,
) -> dict[str, object]:
    x, k = build_grid(nx, length)
    x = np.asarray(x, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)

    template_params = _build_tanaka_template_params(
        nx=nx,
        length=length,
        depth=depth,
        gravity=gravity,
        dno_order=dno_order,
        pad_factor=pad_factor,
        collocation_points=collocation_points,
        quadrature_substeps=quadrature_substeps,
        interpolation_degree=interpolation_degree,
        s_max=s_max,
        alpha=alpha,
        transform_power=transform_power,
        qc_lower=qc_lower,
        qc_upper=qc_upper,
        outer_iterations=outer_iterations,
        fixed_point_iterations=fixed_point_iterations,
        f2_tolerance=f2_tolerance,
    )

    amplitudes = jnp.asarray([spec.amplitude for spec in specs], dtype=jnp.float64)
    centers = jnp.asarray([spec.center for spec in specs], dtype=jnp.float64)
    directions = jnp.asarray([spec.direction for spec in specs], dtype=jnp.float64)
    batch = solve_modified_tanaka_batched(
        template_params,
        amplitudes,
        centers=centers,
        directions=directions,
    )

    component_eta = np.asarray(batch.eta_periodic, dtype=np.float64)
    component_xi = np.asarray(batch.xi_periodic, dtype=np.float64)
    component_gxi = np.asarray(batch.gxi_periodic, dtype=np.float64)

    eta = np.sum(component_eta, axis=0)
    xi = np.sum(component_xi, axis=0)
    if zero_mean_xi:
        xi = xi - np.mean(xi)

    gxi = np.asarray(
        dno_series_eval(
            jnp.asarray(eta),
            jnp.asarray(xi),
            jnp.asarray(k),
            depth,
            dno_order,
            pad_factor=pad_factor,
        ),
        dtype=np.float64,
    )

    return {
        "name": name,
        "x": x,
        "k": k,
        "eta": eta,
        "xi": xi,
        "gxi": gxi,
        "nx": nx,
        "length": length,
        "depth": depth,
        "gravity": gravity,
        "dno_order": dno_order,
        "pad_factor": pad_factor,
        "component_eta": component_eta,
        "component_xi": component_xi,
        "component_gxi": component_gxi,
        "component_qc": np.asarray(batch.qc, dtype=np.float64),
        "component_froude": np.asarray(batch.froude, dtype=np.float64),
        "component_speed": np.asarray(batch.speed, dtype=np.float64),
        "specs": specs,
        "tanaka_params": template_params,
    }


def make_preset_specs(name: str) -> list[CrestSpec]:
    presets = {
        "head_on_2v2": [
            CrestSpec(amplitude=0.35, center=28.0, direction=1),
            CrestSpec(amplitude=0.15, center=52.0, direction=1),
            CrestSpec(amplitude=0.30, center=112.0, direction=-1),
            CrestSpec(amplitude=0.20, center=136.0, direction=-1),
        ],
        "head_on_3v2": [
            CrestSpec(amplitude=0.30, center=22.0, direction=1),
            CrestSpec(amplitude=0.20, center=42.0, direction=1),
            CrestSpec(amplitude=0.10, center=62.0, direction=1),
            CrestSpec(amplitude=0.25, center=112.0, direction=-1),
            CrestSpec(amplitude=0.15, center=136.0, direction=-1),
        ],
        "head_on_3v3": [
            CrestSpec(amplitude=0.30, center=20.0, direction=1),
            CrestSpec(amplitude=0.20, center=38.0, direction=1),
            CrestSpec(amplitude=0.10, center=56.0, direction=1),
            CrestSpec(amplitude=0.35, center=108.0, direction=-1),
            CrestSpec(amplitude=0.25, center=126.0, direction=-1),
            CrestSpec(amplitude=0.15, center=144.0, direction=-1),
        ],
    }
    return presets[name]


def build_multi_crest_preset(
    preset_name: str,
    nx: int = 1024,
    length: float = 164.0,
    depth: float = 1.0,
    gravity: float = 1.0,
    dno_order: int = 6,
    pad_factor: int = 8,
    zero_mean_xi: bool = True,
    collocation_points: int = 257,
    quadrature_substeps: int = 4,
    interpolation_degree: int = 3,
    s_max: float = 2.5,
    alpha: float = 0.01,
    transform_power: int = 5,
    qc_lower: float = 0.2,
    qc_upper: float = 0.999,
    outer_iterations: int = 24,
    fixed_point_iterations: int = 80,
    f2_tolerance: float = 1e-10,
) -> dict[str, object]:
    return build_multi_crest_initial_condition(
        make_preset_specs(preset_name),
        nx=nx,
        length=length,
        depth=depth,
        gravity=gravity,
        dno_order=dno_order,
        pad_factor=pad_factor,
        zero_mean_xi=zero_mean_xi,
        name=preset_name,
        collocation_points=collocation_points,
        quadrature_substeps=quadrature_substeps,
        interpolation_degree=interpolation_degree,
        s_max=s_max,
        alpha=alpha,
        transform_power=transform_power,
        qc_lower=qc_lower,
        qc_upper=qc_upper,
        outer_iterations=outer_iterations,
        fixed_point_iterations=fixed_point_iterations,
        f2_tolerance=f2_tolerance,
    )


def save_multi_crest_initial_condition(
    payload: dict[str, object],
    output_path: str | Path,
) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output,
        name=np.asarray(payload["name"]),
        x=np.asarray(payload["x"]),
        k=np.asarray(payload["k"]),
        eta=np.asarray(payload["eta"]),
        xi=np.asarray(payload["xi"]),
        gxi=np.asarray(payload["gxi"]),
        component_eta=np.asarray(payload["component_eta"]),
        component_xi=np.asarray(payload["component_xi"]),
        component_gxi=np.asarray(payload["component_gxi"]),
        component_qc=np.asarray(payload["component_qc"]),
        component_froude=np.asarray(payload["component_froude"]),
        component_speed=np.asarray(payload["component_speed"]),
        spec_json=np.asarray(json.dumps([asdict(spec) for spec in payload["specs"]])),
        tanaka_params_json=np.asarray(json.dumps(asdict(payload["tanaka_params"]))),
        nx=np.asarray(payload["nx"]),
        length=np.asarray(payload["length"]),
        depth=np.asarray(payload["depth"]),
        gravity=np.asarray(payload["gravity"]),
        dno_order=np.asarray(payload["dno_order"]),
        pad_factor=np.asarray(payload["pad_factor"]),
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multi-crest solitary-wave initial conditions from batched Tanaka solves.")
    parser.add_argument("--preset", default="head_on_2v2")
    parser.add_argument("--output", default="outputs/gen_data/head_on_2v2.npz")
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=164.0)
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--dno_order", type=int, default=6)
    parser.add_argument("--pad_factor", type=int, default=8)
    parser.add_argument("--keep_mean_xi", action="store_true")
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


def main() -> None:
    args = parse_args()
    payload = build_multi_crest_preset(
        args.preset,
        nx=args.nx,
        length=args.length,
        depth=args.depth,
        gravity=args.gravity,
        dno_order=args.dno_order,
        pad_factor=args.pad_factor,
        zero_mean_xi=not args.keep_mean_xi,
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
    output = save_multi_crest_initial_condition(payload, args.output)
    print(output)


if __name__ == "__main__":
    main()
