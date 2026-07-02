from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..solvers.dno_series_jax import build_grid, dno_series_eval
from ..tanaka_ICs.modified_tanaka import make_default_tanaka_template, solve_modified_tanaka_batched

DEFAULT_RANDOM_MIN_SEPARATION = 50.0
DEFAULT_RANDOM_MIN_CRESTS = 1
DEFAULT_RANDOM_MAX_CRESTS = 3


@dataclass(frozen=True)
class CrestSpec:
    amplitude: float
    center: float
    direction: int


def serialize_specs(specs: list[CrestSpec]) -> list[dict[str, float | int]]:
    return [asdict(spec) for spec in specs]


def serialize_case_specs(case_specs: list[list[CrestSpec]]) -> list[list[dict[str, float | int]]]:
    return [serialize_specs(specs) for specs in case_specs]


def periodic_distance(a: float, b: float, length: float) -> float:
    delta = abs(a - b)
    return min(delta, length - delta)


def sample_centers(
    rng: np.random.Generator,
    n_centers: int,
    length: float,
    min_separation: float,
) -> list[float]:
    required_length = n_centers * min_separation
    if required_length > length:
        raise ValueError("Requested crest count and minimum separation do not fit on the periodic domain.")

    slack = length - required_length
    if slack > 0.0:
        extra_gaps = rng.dirichlet(np.ones(n_centers)) * slack
    else:
        extra_gaps = np.zeros(n_centers, dtype=np.float64)

    gaps = min_separation + extra_gaps
    start = float(rng.uniform(0.0, length))
    offsets = np.concatenate(([0.0], np.cumsum(gaps[:-1])))
    centers = (start + offsets) % length
    return sorted(float(center) for center in centers)


def sample_random_crest_specs(
    rng: np.random.Generator,
    *,
    length: float,
    amplitude_min: float,
    amplitude_max: float,
    min_crests: int,
    max_crests: int,
    min_separation: float,
) -> list[CrestSpec]:
    n_crests = int(rng.integers(min_crests, max_crests + 1))
    centers = sample_centers(rng, n_crests, length, min_separation)
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


def sample_random_cases(
    rng: np.random.Generator,
    n_cases: int,
    *,
    length: float,
    amplitude_min: float,
    amplitude_max: float,
    min_crests: int,
    max_crests: int,
    min_separation: float,
) -> list[list[CrestSpec]]:
    return [
        sample_random_crest_specs(
            rng,
            length=length,
            amplitude_min=amplitude_min,
            amplitude_max=amplitude_max,
            min_crests=min_crests,
            max_crests=max_crests,
            min_separation=min_separation,
        )
        for _ in range(n_cases)
    ]


def sample_sum_budgeted_crest_specs(
    rng: np.random.Generator,
    *,
    length: float,
    case_amplitude_min: float,
    case_amplitude_max: float,
    min_crests: int,
    max_crests: int,
    min_separation: float,
    per_crest_floor: float,
) -> list[CrestSpec]:
    """Draw a case where the *sum* of per-crest amplitudes is budgeted.

    Picks a case-level budget S ~ U(case_amplitude_min, case_amplitude_max),
    then splits via Dirichlet(1,...,1) into N crests; clamps each crest to at
    least `per_crest_floor` and renormalizes so the sum still equals S. The
    sum-budget is the relevant quantity for DNO-series convergence on the
    [0, 2π] domain because soliton tails superpose additively.
    """
    n_crests = int(rng.integers(min_crests, max_crests + 1))
    if per_crest_floor * n_crests > case_amplitude_max:
        raise ValueError(
            f"per_crest_floor*max_crests={per_crest_floor * n_crests} exceeds "
            f"case_amplitude_max={case_amplitude_max}; loosen the floor or the budget."
        )
    case_budget = float(rng.uniform(case_amplitude_min, case_amplitude_max))
    case_budget = max(case_budget, per_crest_floor * n_crests)
    weights = rng.dirichlet(np.ones(n_crests))
    raw = case_budget * weights
    clamped = np.maximum(raw, per_crest_floor)
    amps = clamped * (case_budget / clamped.sum())
    centers = sample_centers(rng, n_crests, length, min_separation)
    directions = rng.choice(np.array([-1, 1], dtype=int), size=n_crests)
    return [
        CrestSpec(amplitude=float(a), center=float(c), direction=int(d))
        for a, c, d in zip(amps, centers, directions)
    ]


def sample_sum_budgeted_cases(
    rng: np.random.Generator,
    n_cases: int,
    *,
    length: float,
    case_amplitude_min: float,
    case_amplitude_max: float,
    min_crests: int,
    max_crests: int,
    min_separation: float,
    per_crest_floor: float,
) -> list[list[CrestSpec]]:
    return [
        sample_sum_budgeted_crest_specs(
            rng,
            length=length,
            case_amplitude_min=case_amplitude_min,
            case_amplitude_max=case_amplitude_max,
            min_crests=min_crests,
            max_crests=max_crests,
            min_separation=min_separation,
            per_crest_floor=per_crest_floor,
        )
        for _ in range(n_cases)
    ]


def flatten_case_specs(case_specs: list[list[CrestSpec]]) -> tuple[list[CrestSpec], np.ndarray]:
    flat_specs: list[CrestSpec] = []
    crest_case_ids: list[int] = []
    for case_idx, specs in enumerate(case_specs):
        flat_specs.extend(specs)
        crest_case_ids.extend([case_idx] * len(specs))
    return flat_specs, np.asarray(crest_case_ids, dtype=np.int32)


def specs_to_jax_arrays(specs: list[CrestSpec]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    amplitudes = jnp.asarray([spec.amplitude for spec in specs], dtype=jnp.float64)
    centers = jnp.asarray([spec.center for spec in specs], dtype=jnp.float64)
    directions = jnp.asarray([spec.direction for spec in specs], dtype=jnp.float64)
    return amplitudes, centers, directions


def stack_case_field(cases: list[dict[str, object]], key: str) -> np.ndarray:
    return np.stack([np.asarray(case[key]) for case in cases], axis=0)


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

    template_params = make_default_tanaka_template(
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

    amplitudes, centers, directions = specs_to_jax_arrays(specs)
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
        spec_json=np.asarray(json.dumps(serialize_specs(payload["specs"]))),
        tanaka_params_json=np.asarray(json.dumps(asdict(payload["tanaka_params"]))),
        nx=np.asarray(payload["nx"]),
        length=np.asarray(payload["length"]),
        depth=np.asarray(payload["depth"]),
        gravity=np.asarray(payload["gravity"]),
        dno_order=np.asarray(payload["dno_order"]),
        pad_factor=np.asarray(payload["pad_factor"]),
    )
    return output


def _load_specs(spec_json: str | None, spec_file: str | None) -> list[CrestSpec]:
    if spec_json is None and spec_file is None:
        raise ValueError("Provide --spec_json or --spec_file.")

    if spec_json is not None:
        raw_specs = json.loads(spec_json)
    else:
        raw_specs = json.loads(Path(spec_file).expanduser().read_text(encoding="utf-8"))

    return [CrestSpec(**raw_spec) for raw_spec in raw_specs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multi-crest solitary-wave initial conditions from batched Tanaka solves.")
    parser.add_argument("--spec_json")
    parser.add_argument("--spec_file")
    parser.add_argument("--name", default="multi_crest")
    parser.add_argument("--output", default="outputs/gen_data/multi_crest.npz")
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
    payload = build_multi_crest_initial_condition(
        _load_specs(args.spec_json, args.spec_file),
        nx=args.nx,
        length=args.length,
        depth=args.depth,
        gravity=args.gravity,
        dno_order=args.dno_order,
        pad_factor=args.pad_factor,
        zero_mean_xi=not args.keep_mean_xi,
        name=args.name,
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
    print(json.dumps({"output_path": str(output)}, indent=2))


if __name__ == "__main__":
    main()
