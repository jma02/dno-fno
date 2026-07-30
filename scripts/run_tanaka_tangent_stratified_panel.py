"""Deterministic CPU validation panel for tangent-aware Tanaka placement.

The static panel spans the main and steep Tanaka supports actually used by
the project.  It independently reconstructs every specification on
N=1024, 2048, and 4096 grids, holds the delivered Fourier band |k|<=128
fixed, and compares the state and order-six DNO label.  It also compares the
default 257-point Tanaka solve with a 513-point solve.

The tangent-Hermite evaluator is imported from the production generator.
The only placement implemented locally is the former piecewise-linear rule,
which is retained solely as the paired control.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from solver.gen_data.generate_tanaka_dataset_v2 import (
    TANAKA_FINE_FACTOR,
    TANAKA_PROFILE_RECONSTRUCTION,
    build_per_case_initial_conditions,
    chunked_gxi_over_time,
    cubic_hermite_zero_exterior,
    tanaka_periodic_image_radius,
)
from solver.gen_data.multi_crest import (
    CrestSpec,
    flatten_case_specs,
    specs_to_jax_arrays,
)
from solver.solvers.dno_series_jax import (
    build_grid,
    dno_series_eval,
    make_linear_dno_symbol,
    myfft,
    myifft,
)
from solver.solvers.time_integrator import (
    SolverParams,
    State,
    _spectral_truncate_real,
    batched_rollout,
    cast_solver_params_dtype,
    cast_state_dtype,
    make_normalized_rollout_settings,
    spectral_dx,
)
from solver.tanaka_ICs.modified_tanaka import (
    ModifiedTanakaBatchSolution,
    ModifiedTanakaParams,
    make_default_tanaka_template,
    solve_modified_tanaka_batched,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "tanaka_tangent_stratified_panel_20260723"
    / "summary.json"
)

LENGTH = 2.0 * math.pi
GRAVITY = 1.0
BASE_NX = 1024
GRID_SIZES = (1024, 2048, 4096)
FIXED_MODE_CUTOFF = 128
DEFAULT_COLLOCATION_POINTS = 257
REFINED_COLLOCATION_POINTS = 513
GRID_STATE_TOLERANCE = 3.0e-4
GRID_LABEL_TOLERANCE = 1.0e-3
COLLOCATION_LOW_TOLERANCE = 1.0e-3
COLLOCATION_FULL_TOLERANCE = 2.0e-3
SYMMETRY_TOLERANCE = 1.0e-8
TRAVELING_RELATIVE_FACTOR = 1.10
TRAVELING_ABSOLUTE_ALLOWANCE = 2.0e-3
CASE31_MINIMUM_HIGH_BAND_REDUCTION = 20.0
CASE31_LOW_BAND_TOLERANCE = 1.0e-3
HAMILTONIAN_DRIFT_TOLERANCE = 1.0e-3
ROLLOUT_LOW_BAND_TOLERANCE = 5.0e-3
MINIMUM_WATER_COLUMN_FRACTION = 0.1
BERNOULLI_TOLERANCE = 1.0e-11
BANDS = {
    "low_0_32": (0, 32),
    "middle_33_79": (33, 79),
    "high_80_128": (80, 128),
    "retained_0_128": (0, 128),
}

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int32]
FieldName: TypeAlias = Literal["eta", "xi", "q"]


@dataclass(frozen=True)
class PanelCase:
    name: str
    stratum: str
    purpose: str
    depth: float
    crests: tuple[CrestSpec, ...]


@dataclass(frozen=True)
class BuiltFields:
    eta: FloatArray
    xi: FloatArray
    q: FloatArray
    component_eta: FloatArray
    component_xi: FloatArray
    component_q: FloatArray
    raw_component_eta: FloatArray


@dataclass(frozen=True)
class ProfileDiagnostics:
    maximum_knot_value_error: float
    maximum_knot_slope_error: float
    maximum_endpoint_eta: float
    maximum_endpoint_slope: float
    maximum_crest_slope: float
    minimum_x_spacing: float
    minimum_cos_theta: float


@dataclass(frozen=True)
class LegacyBuild:
    fields: BuiltFields
    component_froude: FloatArray
    profile_diagnostics: ProfileDiagnostics


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    value: float | bool
    comparison: str
    threshold: float | bool
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run-rollout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the four-case paired GL2 rollout after the static panel.",
    )
    parser.add_argument("--rollout-tmax", type=float, default=20.0)
    parser.add_argument("--output-dt", type=float, default=0.08)
    parser.add_argument("--gxi-chunk-size", type=int, default=8)
    return parser.parse_args()


def make_panel() -> tuple[PanelCase, ...]:
    return (
        PanelCase(
            name="main_lower_left",
            stratum="main_single",
            purpose="minimum depth and minimum single-crest steepness",
            depth=0.01,
            crests=(CrestSpec(0.10, LENGTH / 2.0, 1),),
        ),
        PanelCase(
            name="main_narrow_steep",
            stratum="main_single",
            purpose="minimum depth and maximum main steepness",
            depth=0.01,
            crests=(CrestSpec(0.35, LENGTH / 4.0, -1),),
        ),
        PanelCase(
            name="main_log_midpoint",
            stratum="main_single",
            purpose="log-depth and steepness midpoint",
            depth=math.sqrt(0.01 * 0.30),
            crests=(CrestSpec(0.225, LENGTH / 2.0, 1),),
        ),
        PanelCase(
            name="main_broad_weak",
            stratum="main_single",
            purpose="maximum archived main depth and minimum steepness",
            depth=0.30,
            crests=(CrestSpec(0.10, 3.0 * LENGTH / 4.0, -1),),
        ),
        PanelCase(
            name="main_upper_corner",
            stratum="main_single",
            purpose="maximum archived main depth and steepness",
            depth=0.30,
            crests=(CrestSpec(0.35, LENGTH / 2.0, 1),),
        ),
        PanelCase(
            name="steep_lower_corner",
            stratum="steep_single",
            purpose="lower corner of the steep-Tanaka stratum",
            depth=0.20,
            crests=(CrestSpec(0.25, LENGTH / 4.0, 1),),
        ),
        PanelCase(
            name="steep_upper_seam",
            stratum="steep_single",
            purpose="upper steep-Tanaka corner placed on the periodic seam",
            depth=0.35,
            crests=(CrestSpec(0.45, 0.0, 1),),
        ),
        PanelCase(
            name="steep_upper_shift",
            stratum="symmetry",
            purpose="half-period translation twin of steep_upper_seam",
            depth=0.35,
            crests=(CrestSpec(0.45, LENGTH / 2.0, 1),),
        ),
        PanelCase(
            name="steep_upper_reverse",
            stratum="symmetry",
            purpose="direction-reversal twin of steep_upper_seam",
            depth=0.35,
            crests=(CrestSpec(0.45, 0.0, -1),),
        ),
        PanelCase(
            name="may_case31",
            stratum="counterpropagating",
            purpose="exact May-v2 cancellation and interpolation-tail sentinel",
            depth=0.26861433760407505,
            crests=(
                CrestSpec(
                    0.05548354495289499,
                    3.5548496920089176,
                    -1,
                ),
                CrestSpec(
                    0.053531413616445936,
                    5.032703928715591,
                    1,
                ),
            ),
        ),
        PanelCase(
            name="main_subfloor_multicrest",
            stratum="main_multicrest",
            purpose="observed post-renormalization component tail below 0.05",
            depth=0.03,
            crests=(
                CrestSpec(0.085, 0.0, 1),
                CrestSpec(0.0325, LENGTH / 3.0, -1),
                CrestSpec(0.0325, 2.0 * LENGTH / 3.0, 1),
            ),
        ),
        PanelCase(
            name="main_max_budget_multicrest",
            stratum="main_multicrest",
            purpose="maximum main sum budget with unequal components",
            depth=0.20,
            crests=(
                CrestSpec(0.20, 0.0, 1),
                CrestSpec(0.10, LENGTH / 3.0, -1),
                CrestSpec(0.05, 2.0 * LENGTH / 3.0, 1),
            ),
        ),
    )


def make_gate(
    *,
    name: str,
    value: float,
    comparison: Literal["<=", ">="],
    threshold: float,
    detail: str,
) -> GateResult:
    passed = value <= threshold if comparison == "<=" else value >= threshold
    return GateResult(
        name=name,
        passed=bool(passed),
        value=float(value),
        comparison=comparison,
        threshold=float(threshold),
        detail=detail,
    )


def make_boolean_gate(name: str, value: bool, detail: str) -> GateResult:
    return GateResult(
        name=name,
        passed=bool(value),
        value=bool(value),
        comparison="==",
        threshold=True,
        detail=detail,
    )


def component_layout(
    cases: tuple[PanelCase, ...],
) -> tuple[list[CrestSpec], IntArray, FloatArray]:
    case_specs = [list(case.crests) for case in cases]
    flat_specs, owners = flatten_case_specs(case_specs)
    depths = np.asarray(
        [cases[int(owner)].depth for owner in owners],
        dtype=np.float64,
    )
    return flat_specs, np.asarray(owners, dtype=np.int32), depths


def make_template(
    nx: int,
    collocation_points: int,
) -> ModifiedTanakaParams:
    return make_default_tanaka_template(
        depth=1.0,
        gravity=GRAVITY,
        direction=1,
        nx=nx,
        length=LENGTH,
        center=0.0,
        dno_order=6,
        pad_factor=8,
        collocation_points=collocation_points,
    )


def fixed_mode_project(
    field: jax.Array,
    k: jax.Array,
) -> jax.Array:
    mask = (jnp.abs(k) <= FIXED_MODE_CUTOFF).astype(jnp.complex128)
    return jnp.fft.ifft(
        jnp.fft.fft(field, axis=-1) * mask,
        axis=-1,
    ).real


def aggregate_components(
    components: jax.Array,
    owners: IntArray,
    n_cases: int,
) -> jax.Array:
    result = jnp.zeros(
        (n_cases, components.shape[-1]),
        dtype=components.dtype,
    )
    return result.at[jnp.asarray(owners)].add(components)


def project_and_label(
    *,
    raw_component_eta: jax.Array,
    raw_component_xi: jax.Array,
    owners: IntArray,
    case_depths: FloatArray,
    component_depths: FloatArray,
    nx: int,
) -> BuiltFields:
    _, k = build_grid(nx, LENGTH)
    k = jnp.asarray(k, dtype=jnp.float64)
    raw_eta = aggregate_components(raw_component_eta, owners, len(case_depths))
    raw_xi = aggregate_components(raw_component_xi, owners, len(case_depths))

    eta = fixed_mode_project(raw_eta, k)
    xi = fixed_mode_project(raw_xi, k)
    xi = xi - jnp.mean(xi, axis=-1, keepdims=True)
    component_eta = fixed_mode_project(raw_component_eta, k)
    component_xi = fixed_mode_project(raw_component_xi, k)
    component_xi = component_xi - jnp.mean(
        component_xi,
        axis=-1,
        keepdims=True,
    )

    all_eta = jnp.concatenate((eta, component_eta), axis=0)
    all_xi = jnp.concatenate((xi, component_xi), axis=0)
    all_depths = jnp.asarray(
        np.concatenate((case_depths, component_depths))[:, None],
        dtype=jnp.float64,
    )
    all_q = dno_series_eval(
        all_eta,
        all_xi,
        k,
        all_depths,
        6,
        pad_factor=8,
    )
    all_q = fixed_mode_project(all_q, k)
    jax.block_until_ready(all_q)

    n_cases = len(case_depths)
    return BuiltFields(
        eta=np.asarray(jax.device_get(eta), dtype=np.float64),
        xi=np.asarray(jax.device_get(xi), dtype=np.float64),
        q=np.asarray(jax.device_get(all_q[:n_cases]), dtype=np.float64),
        component_eta=np.asarray(
            jax.device_get(component_eta),
            dtype=np.float64,
        ),
        component_xi=np.asarray(
            jax.device_get(component_xi),
            dtype=np.float64,
        ),
        component_q=np.asarray(
            jax.device_get(all_q[n_cases:]),
            dtype=np.float64,
        ),
        raw_component_eta=np.asarray(
            jax.device_get(raw_component_eta),
            dtype=np.float64,
        ),
    )


def build_tangent(
    cases: tuple[PanelCase, ...],
    nx: int,
    collocation_points: int,
) -> BuiltFields:
    flat_specs, owners, component_depths = component_layout(cases)
    component_cases = [[spec] for spec in flat_specs]
    template = make_template(nx, collocation_points)
    raw_component_eta, raw_component_xi = build_per_case_initial_conditions(
        template_params=template,
        case_h_ref=component_depths,
        case_specs=component_cases,
        length=LENGTH,
        nx=nx,
        gravity=GRAVITY,
    )
    return project_and_label(
        raw_component_eta=raw_component_eta,
        raw_component_xi=raw_component_xi,
        owners=owners,
        case_depths=np.asarray([case.depth for case in cases], dtype=np.float64),
        component_depths=component_depths,
        nx=nx,
    )


def legacy_linear_place(
    *,
    x_profile: jax.Array,
    eta_profile: jax.Array,
    depth: jax.Array,
    center: jax.Array,
    x_fine: jax.Array,
    nx: int,
    image_radius: int,
) -> jax.Array:
    x_nodes = x_profile * depth + center
    eta_nodes = eta_profile * depth
    periodic_center = jnp.mod(center, LENGTH)
    x_nodes = x_profile * depth + periodic_center
    shifts = LENGTH * jnp.arange(
        -image_radius,
        image_radius + 1,
        dtype=x_fine.dtype,
    )
    queries = x_fine[None, :] + shifts[:, None]
    eta_fine = jnp.sum(
        jnp.interp(
            queries,
            x_nodes,
            eta_nodes,
            left=0.0,
            right=0.0,
        ),
        axis=0,
    )
    spectrum = jnp.fft.rfft(eta_fine)
    native = spectrum[: nx // 2 + 1] / TANAKA_FINE_FACTOR
    return jnp.fft.irfft(native, n=nx).astype(eta_profile.dtype)


def profile_diagnostics(
    solution: ModifiedTanakaBatchSolution,
) -> ProfileDiagnostics:
    x_profile = solution.x_profile
    eta_profile = solution.eta_profile
    theta_profile = solution.theta_profile
    midpoint = x_profile.shape[-1] // 2
    eta_nodes = eta_profile.at[:, 0].set(0.0).at[:, -1].set(0.0)
    slopes = (
        jnp.tan(theta_profile)
        .at[:, 0]
        .set(0.0)
        .at[:, midpoint]
        .set(0.0)
        .at[:, -1]
        .set(0.0)
    )

    reconstructed = jax.vmap(cubic_hermite_zero_exterior)(
        x_profile,
        eta_nodes,
        slopes,
        x_profile,
    )

    def reconstructed_slopes(
        x_nodes: jax.Array,
        y_nodes: jax.Array,
        node_slopes: jax.Array,
    ) -> jax.Array:
        def evaluate_scalar(value: jax.Array) -> jax.Array:
            return cubic_hermite_zero_exterior(
                x_nodes,
                y_nodes,
                node_slopes,
                value[None],
            )[0]

        return jax.vmap(jax.grad(evaluate_scalar))(x_nodes[1:-1])

    differentiated = jax.vmap(reconstructed_slopes)(
        x_profile,
        eta_nodes,
        slopes,
    )
    jax.block_until_ready(differentiated)
    return ProfileDiagnostics(
        maximum_knot_value_error=float(
            jnp.max(jnp.abs(reconstructed - eta_nodes))
        ),
        maximum_knot_slope_error=float(
            jnp.max(jnp.abs(differentiated - slopes[:, 1:-1]))
        ),
        maximum_endpoint_eta=float(
            jnp.max(jnp.abs(eta_profile[:, (0, -1)]))
        ),
        maximum_endpoint_slope=float(
            jnp.max(
                jnp.abs(jnp.tan(theta_profile[:, (0, -1)]))
            )
        ),
        maximum_crest_slope=float(
            jnp.max(jnp.abs(jnp.tan(theta_profile[:, midpoint])))
        ),
        minimum_x_spacing=float(jnp.min(jnp.diff(x_profile, axis=-1))),
        minimum_cos_theta=float(jnp.min(jnp.cos(theta_profile))),
    )


def build_legacy(
    cases: tuple[PanelCase, ...],
    nx: int,
    collocation_points: int,
) -> LegacyBuild:
    flat_specs, owners, component_depths = component_layout(cases)
    steepness, centers, directions = specs_to_jax_arrays(flat_specs)
    template = make_template(nx, collocation_points)
    solution = solve_modified_tanaka_batched(
        template,
        steepness,
        centers=jnp.zeros_like(steepness),
        directions=directions,
    )
    diagnostics = profile_diagnostics(solution)
    nx_fine = nx * TANAKA_FINE_FACTOR
    x_fine = (LENGTH / nx_fine) * jnp.arange(
        nx_fine,
        dtype=jnp.float64,
    )
    depth = jnp.asarray(component_depths, dtype=jnp.float64)
    image_radius = tanaka_periodic_image_radius(
        solution.x_profile,
        depth,
        LENGTH,
    )

    raw_component_eta = jax.vmap(
        lambda x_row, eta_row, h, center: legacy_linear_place(
            x_profile=x_row,
            eta_profile=eta_row,
            depth=h,
            center=center,
            x_fine=x_fine,
            nx=nx,
            image_radius=image_radius,
        )
    )(
        solution.x_profile,
        solution.eta_profile,
        depth,
        centers,
    )

    _, k = build_grid(nx, LENGTH)
    k = jnp.asarray(k, dtype=jnp.float64)
    signed_speed = (
        directions
        * solution.froude
        * jnp.sqrt(GRAVITY * depth)
    )
    eta_x = spectral_dx(raw_component_eta, k)
    radical = (1.0 + eta_x**2) * (
        signed_speed[:, None] ** 2
        - 2.0 * GRAVITY * raw_component_eta
    )
    raw_xi_x = (
        signed_speed[:, None]
        - directions[:, None] * jnp.sqrt(jnp.maximum(radical, 0.0))
    )
    inv_ik = jnp.where(k != 0.0, 1.0 / (1j * k), 0.0)
    raw_component_xi = myifft(inv_ik * myfft(raw_xi_x, nx))
    raw_component_xi = raw_component_xi - jnp.mean(
        raw_component_xi,
        axis=-1,
        keepdims=True,
    )
    fields = project_and_label(
        raw_component_eta=raw_component_eta,
        raw_component_xi=raw_component_xi,
        owners=owners,
        case_depths=np.asarray([case.depth for case in cases], dtype=np.float64),
        component_depths=component_depths,
        nx=nx,
    )
    return LegacyBuild(
        fields=fields,
        component_froude=np.asarray(
            jax.device_get(solution.froude),
            dtype=np.float64,
        ),
        profile_diagnostics=diagnostics,
    )


def restrict_to_base(field: FloatArray) -> FloatArray:
    if field.shape[-1] == BASE_NX:
        return np.asarray(field, dtype=np.float64)
    return np.asarray(
        _spectral_truncate_real(
            jnp.asarray(field, dtype=jnp.float64),
            BASE_NX,
        ),
        dtype=np.float64,
    )


def field_from(fields: BuiltFields, name: FieldName) -> FloatArray:
    return np.asarray(getattr(fields, name), dtype=np.float64)


def component_field_from(
    fields: BuiltFields,
    name: FieldName,
) -> FloatArray:
    return np.asarray(getattr(fields, f"component_{name}"), dtype=np.float64)


def coefficient_slice(
    field: FloatArray,
    lower: int,
    upper: int,
) -> NDArray[np.complex128]:
    spectrum = np.fft.rfft(field, axis=-1) / field.shape[-1]
    return np.asarray(
        spectrum[..., lower : upper + 1],
        dtype=np.complex128,
    )


def band_norm(
    field: FloatArray,
    lower: int,
    upper: int,
) -> FloatArray:
    coefficients = coefficient_slice(field, lower, upper)
    return np.asarray(
        np.linalg.norm(coefficients, axis=-1),
        dtype=np.float64,
    )


def relative_band_error(
    candidate: FloatArray,
    reference: FloatArray,
    lower: int,
    upper: int,
) -> float:
    difference = coefficient_slice(candidate - reference, lower, upper)
    denominator = float(
        np.linalg.norm(coefficient_slice(reference, lower, upper))
    )
    return float(
        np.linalg.norm(difference)
        / max(denominator, np.finfo(np.float64).tiny)
    )


def component_scale(
    components: FloatArray,
    owners: IntArray,
    case_index: int,
    lower: int,
    upper: int,
) -> float:
    norms = band_norm(components[owners == case_index], lower, upper)
    return float(np.sum(norms))


def component_normalized_error(
    candidate: FloatArray,
    reference: FloatArray,
    reference_components: FloatArray,
    owners: IntArray,
    case_index: int,
    lower: int,
    upper: int,
) -> float:
    numerator = float(
        np.linalg.norm(
            coefficient_slice(
                candidate - reference,
                lower,
                upper,
            )
        )
    )
    denominator = component_scale(
        reference_components,
        owners,
        case_index,
        lower,
        upper,
    )
    return numerator / max(denominator, np.finfo(np.float64).tiny)


def grid_consistency(
    cases: tuple[PanelCase, ...],
    tangent_by_n: dict[int, BuiltFields],
    owners: IntArray,
) -> tuple[list[dict[str, Any]], list[GateResult]]:
    restricted = {
        nx: BuiltFields(
            eta=restrict_to_base(fields.eta),
            xi=restrict_to_base(fields.xi),
            q=restrict_to_base(fields.q),
            component_eta=restrict_to_base(fields.component_eta),
            component_xi=restrict_to_base(fields.component_xi),
            component_q=restrict_to_base(fields.component_q),
            raw_component_eta=restrict_to_base(fields.raw_component_eta),
        )
        for nx, fields in tangent_by_n.items()
    }
    pairs = ((1024, 2048), (2048, 4096), (1024, 4096))
    rows: list[dict[str, Any]] = []
    gates: list[GateResult] = []
    for case_index, case in enumerate(cases):
        for field_name in ("eta", "xi", "q"):
            field = field_name
            pair_metrics: dict[str, dict[str, float]] = {}
            for coarse_nx, fine_nx in pairs:
                coarse = field_from(restricted[coarse_nx], field)[case_index]
                fine = field_from(restricted[fine_nx], field)[case_index]
                components = component_field_from(
                    restricted[4096],
                    field,
                )
                pair_metrics[f"{coarse_nx}_{fine_nx}"] = {
                    "net_relative": relative_band_error(
                        coarse,
                        fine,
                        0,
                        FIXED_MODE_CUTOFF,
                    ),
                    "component_normalized": component_normalized_error(
                        coarse,
                        fine,
                        components,
                        owners,
                        case_index,
                        0,
                        FIXED_MODE_CUTOFF,
                    ),
                }
            maximum_net = max(
                value["net_relative"] for value in pair_metrics.values()
            )
            maximum_component = max(
                value["component_normalized"]
                for value in pair_metrics.values()
            )
            primary = (
                maximum_component
                if len(case.crests) > 1
                else maximum_net
            )
            tolerance = (
                GRID_LABEL_TOLERANCE
                if field == "q"
                else GRID_STATE_TOLERANCE
            )
            row = {
                "case": case.name,
                "field": field,
                "pair_defects": pair_metrics,
                "maximum_net_relative": maximum_net,
                "maximum_component_normalized": maximum_component,
                "primary_defect": primary,
                "tolerance": tolerance,
                "normalization": (
                    "sum_of_component_norms"
                    if len(case.crests) > 1
                    else "net_field_norm"
                ),
            }
            rows.append(row)
            gates.append(
                make_gate(
                    name=f"grid_consistency/{case.name}/{field}",
                    value=primary,
                    comparison="<=",
                    threshold=tolerance,
                    detail=(
                        "maximum fixed-band defect over independently "
                        "constructed N/2N/4N states"
                    ),
                )
            )
    return rows, gates


def collocation_consistency(
    cases: tuple[PanelCase, ...],
    tangent_257: BuiltFields,
    tangent_513: BuiltFields,
    owners: IntArray,
) -> tuple[list[dict[str, Any]], list[GateResult]]:
    rows: list[dict[str, Any]] = []
    gates: list[GateResult] = []
    for case_index, case in enumerate(cases):
        for field_name in ("eta", "xi", "q"):
            field = field_name
            coarse = field_from(tangent_257, field)[case_index]
            reference = field_from(tangent_513, field)[case_index]
            reference_components = component_field_from(
                tangent_513,
                field,
            )
            low_net = relative_band_error(coarse, reference, 0, 32)
            full_net = relative_band_error(
                coarse,
                reference,
                0,
                FIXED_MODE_CUTOFF,
            )
            low_component = component_normalized_error(
                coarse,
                reference,
                reference_components,
                owners,
                case_index,
                0,
                32,
            )
            full_component = component_normalized_error(
                coarse,
                reference,
                reference_components,
                owners,
                case_index,
                0,
                FIXED_MODE_CUTOFF,
            )
            low_primary = (
                low_component if len(case.crests) > 1 else low_net
            )
            full_primary = (
                full_component if len(case.crests) > 1 else full_net
            )
            rows.append(
                {
                    "case": case.name,
                    "field": field,
                    "low_0_32_net_relative": low_net,
                    "low_0_32_component_normalized": low_component,
                    "retained_0_128_net_relative": full_net,
                    "retained_0_128_component_normalized": full_component,
                    "low_primary": low_primary,
                    "full_primary": full_primary,
                }
            )
            gates.extend(
                (
                    make_gate(
                        name=f"collocation_low/{case.name}/{field}",
                        value=low_primary,
                        comparison="<=",
                        threshold=COLLOCATION_LOW_TOLERANCE,
                        detail="257 versus 513 Tanaka knots in modes 0:32",
                    ),
                    make_gate(
                        name=f"collocation_full/{case.name}/{field}",
                        value=full_primary,
                        comparison="<=",
                        threshold=COLLOCATION_FULL_TOLERANCE,
                        detail="257 versus 513 Tanaka knots in modes 0:128",
                    ),
                )
            )
    return rows, gates


def spectral_derivative(field: FloatArray) -> FloatArray:
    nx = field.shape[-1]
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=LENGTH / nx)
    return np.asarray(
        np.fft.ifft(
            1j * k * np.fft.fft(field, axis=-1),
            axis=-1,
        ).real,
        dtype=np.float64,
    )


def rms(field: FloatArray) -> float:
    return float(np.sqrt(np.mean(np.asarray(field, dtype=np.float64) ** 2)))


def physical_component_checks(
    cases: tuple[PanelCase, ...],
    tangent: BuiltFields,
    legacy: LegacyBuild,
    owners: IntArray,
    component_depths: FloatArray,
    flat_specs: list[CrestSpec],
) -> tuple[list[dict[str, Any]], list[GateResult]]:
    rows: list[dict[str, Any]] = []
    gates: list[GateResult] = []
    directions = np.asarray(
        [spec.direction for spec in flat_specs],
        dtype=np.float64,
    )
    signed_speeds = (
        directions
        * legacy.component_froude
        * np.sqrt(GRAVITY * component_depths)
    )

    for component_index, spec in enumerate(flat_specs):
        owner = int(owners[component_index])
        case = cases[owner]
        speed = float(signed_speeds[component_index])
        eta_x_tangent = spectral_derivative(
            tangent.component_eta[component_index]
        )
        eta_x_legacy = spectral_derivative(
            legacy.fields.component_eta[component_index]
        )
        expected_tangent = -speed * eta_x_tangent
        expected_legacy = -speed * eta_x_legacy
        tangent_residual = rms(
            tangent.component_q[component_index] - expected_tangent
        ) / max(rms(expected_tangent), np.finfo(np.float64).tiny)
        legacy_residual = rms(
            legacy.fields.component_q[component_index] - expected_legacy
        ) / max(rms(expected_legacy), np.finfo(np.float64).tiny)
        allowed = max(
            TRAVELING_RELATIVE_FACTOR * legacy_residual,
            legacy_residual + TRAVELING_ABSOLUTE_ALLOWANCE,
        )

        raw_eta = tangent.raw_component_eta[component_index]
        raw_eta_x = spectral_derivative(raw_eta)
        unsigned_speed = abs(speed)
        radical = (1.0 + raw_eta_x**2) * (
            speed**2 - 2.0 * GRAVITY * raw_eta
        )
        direction = float(spec.direction)
        candidate_xi_x = (
            speed
            - direction * np.sqrt(np.maximum(radical, 0.0))
        )
        expected_q = -speed * raw_eta_x
        term_1 = -speed * candidate_xi_x
        term_2 = GRAVITY * raw_eta
        term_3 = 0.5 * candidate_xi_x**2
        term_4 = -0.5 * (
            expected_q + raw_eta_x * candidate_xi_x
        ) ** 2 / (1.0 + raw_eta_x**2)
        bernoulli = term_1 + term_2 + term_3 + term_4
        bernoulli_scale = sum(
            rms(term) for term in (term_1, term_2, term_3, term_4)
        )
        bernoulli_relative = rms(bernoulli) / max(
            bernoulli_scale,
            np.finfo(np.float64).tiny,
        )
        radicand_scale = max(
            unsigned_speed**2,
            np.finfo(np.float64).tiny,
        )
        minimum_scaled_radicand = float(np.min(radical) / radicand_scale)
        mean_derivative_ratio = abs(float(np.mean(candidate_xi_x))) / max(
            rms(candidate_xi_x),
            np.finfo(np.float64).tiny,
        )

        row = {
            "case": case.name,
            "component_index": component_index,
            "component_within_case": int(
                np.sum(owners[:component_index] == owner)
            ),
            "amplitude_ratio": spec.amplitude,
            "center": spec.center,
            "direction": spec.direction,
            "signed_speed": speed,
            "traveling_residual_tangent": tangent_residual,
            "traveling_residual_linear": legacy_residual,
            "traveling_residual_allowed": allowed,
            "minimum_scaled_radicand": minimum_scaled_radicand,
            "bernoulli_relative_residual_pre_mean_removal": (
                bernoulli_relative
            ),
            "removed_xi_derivative_mean_relative": mean_derivative_ratio,
        }
        rows.append(row)
        gates.extend(
            (
                make_gate(
                    name=(
                        f"traveling_wave/{case.name}/"
                        f"component_{row['component_within_case']}"
                    ),
                    value=tangent_residual,
                    comparison="<=",
                    threshold=allowed,
                    detail=(
                        "P128 G6 residual may not worsen by more than 10% "
                        "or 2e-3 relative to the linear control"
                    ),
                ),
                make_gate(
                    name=(
                        f"radicand/{case.name}/"
                        f"component_{row['component_within_case']}"
                    ),
                    value=minimum_scaled_radicand,
                    comparison=">=",
                    threshold=-100.0 * np.finfo(np.float64).eps,
                    detail="radicand before clipping, normalized by c^2",
                ),
                make_gate(
                    name=(
                        f"bernoulli/{case.name}/"
                        f"component_{row['component_within_case']}"
                    ),
                    value=bernoulli_relative,
                    comparison="<=",
                    threshold=BERNOULLI_TOLERANCE,
                    detail=(
                        "steady Bernoulli identity before removing the "
                        "candidate xi_x zero mode"
                    ),
                ),
            )
        )
    return rows, gates


def symmetry_checks(
    cases: tuple[PanelCase, ...],
    tangent: BuiltFields,
) -> tuple[dict[str, Any], list[GateResult]]:
    index = {case.name: position for position, case in enumerate(cases)}
    seam = index["steep_upper_seam"]
    shifted = index["steep_upper_shift"]
    reversed_direction = index["steep_upper_reverse"]
    rows: dict[str, Any] = {"translation": {}, "direction_reversal": {}}
    gates: list[GateResult] = []

    for field_name in ("eta", "xi", "q"):
        values = field_from(tangent, field_name)
        translated = np.roll(values[seam], BASE_NX // 2)
        translation_error = relative_band_error(
            values[shifted],
            translated,
            0,
            FIXED_MODE_CUTOFF,
        )
        sign = 1.0 if field_name == "eta" else -1.0
        direction_error = relative_band_error(
            values[reversed_direction],
            sign * values[seam],
            0,
            FIXED_MODE_CUTOFF,
        )
        rows["translation"][field_name] = translation_error
        rows["direction_reversal"][field_name] = direction_error
        gates.extend(
            (
                make_gate(
                    name=f"translation_symmetry/{field_name}",
                    value=translation_error,
                    comparison="<=",
                    threshold=SYMMETRY_TOLERANCE,
                    detail="seam case versus exact half-period grid shift",
                ),
                make_gate(
                    name=f"direction_symmetry/{field_name}",
                    value=direction_error,
                    comparison="<=",
                    threshold=SYMMETRY_TOLERANCE,
                    detail="eta is even in direction; xi and q are odd",
                ),
            )
        )
    return rows, gates


def case31_checks(
    cases: tuple[PanelCase, ...],
    tangent: BuiltFields,
    legacy: BuiltFields,
) -> tuple[dict[str, Any], list[GateResult]]:
    case_index = next(
        index for index, case in enumerate(cases) if case.name == "may_case31"
    )
    rows: dict[str, Any] = {}
    gates: list[GateResult] = []
    for field_name in ("eta", "xi", "q"):
        tangent_field = field_from(tangent, field_name)[case_index]
        legacy_field = field_from(legacy, field_name)[case_index]
        tangent_high = float(band_norm(tangent_field, 80, 128))
        legacy_high = float(band_norm(legacy_field, 80, 128))
        reduction = legacy_high / max(
            tangent_high,
            np.finfo(np.float64).tiny,
        )
        low_difference = relative_band_error(
            tangent_field,
            legacy_field,
            0,
            32,
        )
        rows[field_name] = {
            "linear_high_band_norm": legacy_high,
            "tangent_high_band_norm": tangent_high,
            "reduction_factor": reduction,
            "low_band_relative_difference": low_difference,
        }
        gates.extend(
            (
                make_gate(
                    name=f"case31_high_band_reduction/{field_name}",
                    value=reduction,
                    comparison=">=",
                    threshold=CASE31_MINIMUM_HIGH_BAND_REDUCTION,
                    detail="linear divided by tangent modes 80:128",
                ),
                make_gate(
                    name=f"case31_low_band_preservation/{field_name}",
                    value=low_difference,
                    comparison="<=",
                    threshold=CASE31_LOW_BAND_TOLERANCE,
                    detail="tangent versus linear modes 0:32",
                ),
            )
        )
    return rows, gates


def band_summary(
    cases: tuple[PanelCase, ...],
    tangent_257: BuiltFields,
    tangent_4096: BuiltFields,
    legacy: BuiltFields,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference = BuiltFields(
        eta=restrict_to_base(tangent_4096.eta),
        xi=restrict_to_base(tangent_4096.xi),
        q=restrict_to_base(tangent_4096.q),
        component_eta=restrict_to_base(tangent_4096.component_eta),
        component_xi=restrict_to_base(tangent_4096.component_xi),
        component_q=restrict_to_base(tangent_4096.component_q),
        raw_component_eta=restrict_to_base(
            tangent_4096.raw_component_eta
        ),
    )
    for case_index, case in enumerate(cases):
        for field_name in ("eta", "xi", "q"):
            row: dict[str, Any] = {
                "case": case.name,
                "field": field_name,
                "bands": {},
            }
            for band_name, (lower, upper) in BANDS.items():
                row["bands"][band_name] = {
                    "linear_257_n1024": float(
                        band_norm(
                            field_from(legacy, field_name)[case_index],
                            lower,
                            upper,
                        )
                    ),
                    "tangent_257_n1024": float(
                        band_norm(
                            field_from(
                                tangent_257,
                                field_name,
                            )[case_index],
                            lower,
                            upper,
                        )
                    ),
                    "tangent_257_n4096_restricted": float(
                        band_norm(
                            field_from(
                                reference,
                                field_name,
                            )[case_index],
                            lower,
                            upper,
                        )
                    ),
                }
            rows.append(row)
    return rows


def structural_checks(
    cases: tuple[PanelCase, ...],
    tangent: BuiltFields,
    profile: ProfileDiagnostics,
) -> tuple[dict[str, Any], list[GateResult]]:
    case_depths = np.asarray([case.depth for case in cases], dtype=np.float64)
    minimum_water_fraction = float(
        np.min(
            (case_depths[:, None] + tangent.eta)
            / case_depths[:, None]
        )
    )
    all_finite = all(
        bool(np.all(np.isfinite(field_from(tangent, field_name))))
        for field_name in ("eta", "xi", "q")
    )
    rows = {
        "profile": asdict(profile),
        "all_delivered_fields_finite": all_finite,
        "minimum_water_column_fraction": minimum_water_fraction,
    }
    gates = [
        make_boolean_gate(
            "all_static_fields_finite",
            all_finite,
            "eta, xi, and q are finite for all cases",
        ),
        make_gate(
            name="minimum_water_column_fraction",
            value=minimum_water_fraction,
            comparison=">=",
            threshold=MINIMUM_WATER_COLUMN_FRACTION,
            detail="minimum (h+eta)/h over the static panel",
        ),
        make_gate(
            name="profile_knot_values",
            value=profile.maximum_knot_value_error,
            comparison="<=",
            threshold=1.0e-12,
            detail="production Hermite evaluator at every Tanaka knot",
        ),
        make_gate(
            name="profile_knot_slopes",
            value=profile.maximum_knot_slope_error,
            comparison="<=",
            threshold=1.0e-10,
            detail="automatic derivative of production evaluator at knots",
        ),
        make_gate(
            name="profile_endpoint_eta",
            value=profile.maximum_endpoint_eta,
            comparison="<=",
            threshold=1.0e-12,
            detail="profile joins zero exterior in value",
        ),
        make_gate(
            name="profile_endpoint_slope",
            value=profile.maximum_endpoint_slope,
            comparison="<=",
            threshold=1.0e-12,
            detail="profile joins zero exterior in slope",
        ),
        make_gate(
            name="profile_crest_slope",
            value=profile.maximum_crest_slope,
            comparison="<=",
            threshold=1.0e-12,
            detail="symmetric solitary-wave crest has zero slope",
        ),
        make_gate(
            name="profile_x_spacing",
            value=profile.minimum_x_spacing,
            comparison=">=",
            threshold=np.finfo(np.float64).eps,
            detail="profile x knots are strictly increasing",
        ),
        make_gate(
            name="profile_graph_condition",
            value=profile.minimum_cos_theta,
            comparison=">=",
            threshold=np.finfo(np.float64).eps,
            detail="cos(theta)>0, so the profile remains a graph",
        ),
    ]
    return rows, gates


def rollout_panel(
    *,
    cases: tuple[PanelCase, ...],
    tangent: BuiltFields,
    legacy: BuiltFields,
    tmax: float,
    output_dt: float,
    gxi_chunk_size: int,
) -> tuple[dict[str, Any], list[GateResult]]:
    selected_names = (
        "main_narrow_steep",
        "main_upper_corner",
        "steep_upper_seam",
        "may_case31",
    )
    index = {case.name: position for position, case in enumerate(cases)}
    eta_rows: list[FloatArray] = []
    xi_rows: list[FloatArray] = []
    depths: list[float] = []
    labels: list[tuple[str, str]] = []
    for case_name in selected_names:
        case_index = index[case_name]
        for arm_name, fields in (
            ("piecewise_linear", legacy),
            ("tangent_hermite", tangent),
        ):
            eta_rows.append(fields.eta[case_index])
            xi_rows.append(fields.xi[case_index])
            depths.append(cases[case_index].depth)
            labels.append((case_name, arm_name))

    initial_eta = jnp.asarray(np.stack(eta_rows), dtype=jnp.float64)
    initial_xi = jnp.asarray(np.stack(xi_rows), dtype=jnp.float64)
    depth_2d = jnp.asarray(depths, dtype=jnp.float64)[:, None]
    times_np = np.arange(
        0.0,
        tmax + 0.5 * output_dt,
        output_dt,
        dtype=np.float64,
    )
    times = jnp.asarray(times_np, dtype=jnp.float64)
    defaults = make_normalized_rollout_settings()._replace(
        filter_fraction=0.25
    )
    _, k = build_grid(BASE_NX, LENGTH)
    k = jnp.asarray(k, dtype=jnp.float64)
    solver_params = SolverParams(
        nx=BASE_NX,
        length=LENGTH,
        depth=depth_2d,
        gravity=GRAVITY,
        dno_order=defaults.dno_order,
        pad_factor=defaults.pad_factor,
        filter_fraction=defaults.filter_fraction,
        k=k,
        g0=make_linear_dno_symbol(k, depth_2d),
    )
    solver_params = cast_solver_params_dtype(solver_params, jnp.float64)

    rollout_start = time.perf_counter()
    payload = batched_rollout(
        cast_state_dtype(
            State(eta=initial_eta, xi=initial_xi),
            jnp.float64,
        ),
        times,
        solver_params,
        save_gxi=False,
        substeps_per_interval=defaults.substeps_per_interval,
        method=defaults.method,
        implicit_iterations=defaults.implicit_iterations,
        implicit_relaxation=defaults.implicit_relaxation,
        zero_mean_xi=defaults.zero_mean_xi,
    )
    jax.block_until_ready(payload["xi"])
    rollout_seconds = time.perf_counter() - rollout_start
    eta = np.asarray(jax.device_get(payload["eta"]), dtype=np.float64)
    xi = np.asarray(jax.device_get(payload["xi"]), dtype=np.float64)

    gxi_start = time.perf_counter()
    q = chunked_gxi_over_time(
        jnp.asarray(eta),
        jnp.asarray(xi),
        k,
        depth_2d,
        defaults.dno_order,
        defaults.pad_factor,
        defaults.filter_fraction,
        gxi_chunk_size,
    )
    q = np.asarray(q, dtype=np.float64)
    gxi_seconds = time.perf_counter() - gxi_start

    dx = LENGTH / BASE_NX
    hamiltonian = 0.5 * dx * np.sum(
        xi * q + GRAVITY * eta**2,
        axis=-1,
    )
    drift = np.abs(hamiltonian - hamiltonian[:1]) / (
        np.abs(hamiltonian[:1]) + np.finfo(np.float64).tiny
    )
    rows: dict[str, Any] = {
        "protocol": {
            "selected_cases": list(selected_names),
            "arms": ["piecewise_linear", "tangent_hermite"],
            "nx": BASE_NX,
            "fixed_mode_cutoff": FIXED_MODE_CUTOFF,
            "tmax": float(times_np[-1]),
            "output_dt": output_dt,
            "substeps_per_interval": defaults.substeps_per_interval,
            "method": defaults.method,
            "implicit_iterations": defaults.implicit_iterations,
        },
        "timing_seconds": {
            "rollout": rollout_seconds,
            "gxi_postprocessing": gxi_seconds,
        },
        "per_arm": {},
        "paired_final_low_band_differences": {},
    }
    gates: list[GateResult] = []
    all_finite = bool(
        np.all(np.isfinite(eta))
        and np.all(np.isfinite(xi))
        and np.all(np.isfinite(q))
    )
    gates.append(
        make_boolean_gate(
            "rollout/all_fields_finite",
            all_finite,
            "eta, xi, and q are finite on the paired T=20 panel",
        )
    )

    for row_index, (case_name, arm_name) in enumerate(labels):
        depth = depths[row_index]
        minimum_water_fraction = float(
            np.min((depth + eta[:, row_index]) / depth)
        )
        maximum_drift = float(np.max(drift[:, row_index]))
        high_history = {
            field_name: {
                "initial": float(
                    band_norm(field[0, row_index], 80, 128)
                ),
                "maximum": float(
                    np.max(
                        band_norm(field[:, row_index], 80, 128)
                    )
                ),
                "final": float(
                    band_norm(field[-1, row_index], 80, 128)
                ),
            }
            for field_name, field in (
                ("eta", eta),
                ("xi", xi),
                ("q", q),
            )
        }
        rows["per_arm"][f"{case_name}/{arm_name}"] = {
            "maximum_relative_hamiltonian_drift": maximum_drift,
            "minimum_water_column_fraction": minimum_water_fraction,
            "high_band_history": high_history,
        }
        gates.extend(
            (
                make_gate(
                    name=f"rollout/hamiltonian/{case_name}/{arm_name}",
                    value=maximum_drift,
                    comparison="<=",
                    threshold=HAMILTONIAN_DRIFT_TOLERANCE,
                    detail="maximum relative Hamiltonian drift",
                ),
                make_gate(
                    name=f"rollout/water_column/{case_name}/{arm_name}",
                    value=minimum_water_fraction,
                    comparison=">=",
                    threshold=MINIMUM_WATER_COLUMN_FRACTION,
                    detail="minimum (h+eta)/h over the rollout",
                ),
            )
        )

    for case_offset, case_name in enumerate(selected_names):
        linear_index = 2 * case_offset
        tangent_index = linear_index + 1
        pair: dict[str, float] = {}
        for field_name, field in (("eta", eta), ("xi", xi)):
            difference = relative_band_error(
                field[-1, tangent_index],
                field[-1, linear_index],
                0,
                32,
            )
            pair[field_name] = difference
            gates.append(
                make_gate(
                    name=f"rollout/low_band_pair/{case_name}/{field_name}",
                    value=difference,
                    comparison="<=",
                    threshold=ROLLOUT_LOW_BAND_TOLERANCE,
                    detail=(
                        "tangent versus linear final state in modes 0:32"
                    ),
                )
            )
        rows["paired_final_low_band_differences"][case_name] = pair
    return rows, gates


def serialize_cases(cases: tuple[PanelCase, ...]) -> list[dict[str, Any]]:
    return [
        {
            "name": case.name,
            "stratum": case.stratum,
            "purpose": case.purpose,
            "depth": case.depth,
            "sum_steepness": float(
                sum(crest.amplitude for crest in case.crests)
            ),
            "crests": [asdict(crest) for crest in case.crests],
        }
        for case in cases
    ]


def summarize_gates(gates: list[GateResult]) -> dict[str, Any]:
    failed = [gate for gate in gates if not gate.passed]
    return {
        "passed": not failed,
        "total": len(gates),
        "failed": len(failed),
        "failed_names": [gate.name for gate in failed],
        "results": [asdict(gate) for gate in gates],
    }


def main() -> None:
    args = parse_args()
    if jax.default_backend() != "cpu":
        raise RuntimeError("This validation must run on CPU.")
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("This validation requires JAX float64.")
    if args.rollout_tmax <= 0.0 or args.output_dt <= 0.0:
        raise ValueError("--rollout-tmax and --output-dt must be positive.")
    if args.gxi_chunk_size <= 0:
        raise ValueError("--gxi-chunk-size must be positive.")

    cases = make_panel()
    flat_specs, owners, component_depths = component_layout(cases)
    start = time.perf_counter()
    timing: dict[str, float] = {}

    tangent_by_n: dict[int, BuiltFields] = {}
    for nx in GRID_SIZES:
        print(
            f"building tangent panel: collocation=257, N={nx}",
            flush=True,
        )
        build_start = time.perf_counter()
        tangent_by_n[nx] = build_tangent(
            cases,
            nx,
            DEFAULT_COLLOCATION_POINTS,
        )
        timing[f"tangent_257_n{nx}"] = time.perf_counter() - build_start

    print("building tangent panel: collocation=513, N=1024", flush=True)
    build_start = time.perf_counter()
    tangent_513 = build_tangent(
        cases,
        BASE_NX,
        REFINED_COLLOCATION_POINTS,
    )
    timing["tangent_513_n1024"] = time.perf_counter() - build_start

    print("building piecewise-linear control: collocation=257, N=1024", flush=True)
    build_start = time.perf_counter()
    legacy = build_legacy(
        cases,
        BASE_NX,
        DEFAULT_COLLOCATION_POINTS,
    )
    timing["linear_257_n1024"] = time.perf_counter() - build_start
    tangent_257 = tangent_by_n[BASE_NX]

    all_gates: list[GateResult] = []
    structural, gates = structural_checks(
        cases,
        tangent_257,
        legacy.profile_diagnostics,
    )
    all_gates.extend(gates)
    grid_rows, gates = grid_consistency(cases, tangent_by_n, owners)
    all_gates.extend(gates)
    collocation_rows, gates = collocation_consistency(
        cases,
        tangent_257,
        tangent_513,
        owners,
    )
    all_gates.extend(gates)
    physical_rows, gates = physical_component_checks(
        cases,
        tangent_257,
        legacy,
        owners,
        component_depths,
        flat_specs,
    )
    all_gates.extend(gates)
    symmetry, gates = symmetry_checks(cases, tangent_257)
    all_gates.extend(gates)
    case31, gates = case31_checks(
        cases,
        tangent_257,
        legacy.fields,
    )
    all_gates.extend(gates)
    bands = band_summary(
        cases,
        tangent_257,
        tangent_by_n[4096],
        legacy.fields,
    )

    rollout: dict[str, Any] | None = None
    if args.run_rollout:
        print(
            "running four-case paired GL2 rollout "
            f"through T={args.rollout_tmax:g}",
            flush=True,
        )
        rollout, gates = rollout_panel(
            cases=cases,
            tangent=tangent_257,
            legacy=legacy.fields,
            tmax=args.rollout_tmax,
            output_dt=args.output_dt,
            gxi_chunk_size=args.gxi_chunk_size,
        )
        all_gates.extend(gates)

    timing["total"] = time.perf_counter() - start
    summary = {
        "protocol": {
            "device": str(jax.devices()[0]),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "length": LENGTH,
            "gravity": GRAVITY,
            "grid_sizes": list(GRID_SIZES),
            "fixed_mode_cutoff": FIXED_MODE_CUTOFF,
            "dno_order": 6,
            "pad_factor": 8,
            "collocation_points": [
                DEFAULT_COLLOCATION_POINTS,
                REFINED_COLLOCATION_POINTS,
            ],
            "fine_factor": TANAKA_FINE_FACTOR,
            "profile_reconstruction": TANAKA_PROFILE_RECONSTRUCTION,
            "tangent_builder": (
                "solver.gen_data.generate_tanaka_dataset_v2."
                "build_per_case_initial_conditions"
            ),
            "tangent_evaluator": (
                "solver.gen_data.generate_tanaka_dataset_v2."
                "cubic_hermite_zero_exterior"
            ),
            "linear_control": (
                "local piecewise-linear control using the production "
                "compact-support periodic image radius"
            ),
            "run_rollout": bool(args.run_rollout),
        },
        "thresholds": {
            "grid_state": GRID_STATE_TOLERANCE,
            "grid_label": GRID_LABEL_TOLERANCE,
            "collocation_low": COLLOCATION_LOW_TOLERANCE,
            "collocation_full": COLLOCATION_FULL_TOLERANCE,
            "symmetry": SYMMETRY_TOLERANCE,
            "traveling_relative_factor": TRAVELING_RELATIVE_FACTOR,
            "traveling_absolute_allowance": (
                TRAVELING_ABSOLUTE_ALLOWANCE
            ),
            "case31_high_band_reduction": (
                CASE31_MINIMUM_HIGH_BAND_REDUCTION
            ),
            "case31_low_band": CASE31_LOW_BAND_TOLERANCE,
            "hamiltonian_drift": HAMILTONIAN_DRIFT_TOLERANCE,
            "rollout_low_band": ROLLOUT_LOW_BAND_TOLERANCE,
            "minimum_water_column_fraction": (
                MINIMUM_WATER_COLUMN_FRACTION
            ),
            "bernoulli": BERNOULLI_TOLERANCE,
        },
        "cases": serialize_cases(cases),
        "static": {
            "structural": structural,
            "grid_consistency": grid_rows,
            "collocation_consistency": collocation_rows,
            "component_physics": physical_rows,
            "symmetry": symmetry,
            "case31_artifact_sentinel": case31,
            "band_norms": bands,
        },
        "rollout": rollout,
        "timing_seconds": timing,
        "gates": summarize_gates(all_gates),
        "support_note": {
            "main_archived": {
                "depth": [0.01, 0.30],
                "sum_steepness": [0.10, 0.35],
                "crest_count": [1, 3],
            },
            "main_current_cli_default_depth": [0.01, 0.08],
            "steep": {
                "depth": [0.20, 0.35],
                "single_crest_steepness": [0.25, 0.45],
            },
            "observed_component_minimum": (
                "about 0.030 because clamp-then-renormalize does not "
                "strictly preserve the nominal 0.05 floor"
            ),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": summary["gates"]["passed"],
                "failed": summary["gates"]["failed"],
                "failed_names": summary["gates"]["failed_names"],
                "timing_seconds": timing,
            },
            indent=2,
        ),
        flush=True,
    )
    if not summary["gates"]["passed"]:
        raise AssertionError(
            "Tanaka tangent validation failed: "
            + ", ".join(summary["gates"]["failed_names"])
        )


if __name__ == "__main__":
    main()
