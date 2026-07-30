"""Run the frozen paper-corpus full-horizon temporal-refinement panel.

The production command uses the first visible JAX device and refuses to run
on CPU unless explicitly allowed:

    uv run python scripts/run_full_horizon_refinement_panel.py

A deliberately small, non-production CPU exercise of the orchestration is:

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
      uv run python scripts/run_full_horizon_refinement_panel.py \
      --smoke --output-dir outputs/full_horizon_refinement_smoke

The production calculation is expensive.  It is split into restartable NPZ
artifacts by family group and time-step arm.  ``summary.json`` records exact
case definitions, realized horizons, telemetry summaries, timings, hashes,
and the final delivered-band refinement decisions.  ``case_manifest.npz``
stores the exact initial arrays and random-sea phases.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, TypeAlias

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("DNO_TANAKA_DTYPE", "float64")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from solver.data.stokes_truth_jax import (  # noqa: E402
    FINITE_DEPTH_STOKES_URSELL_LIMIT,
    finite_depth_stokes_ursell_upper_bound,
    stokes_eta_xi,
)
from solver.gen_data.benjamin_feir_jcp09 import (  # noqa: E402
    build_initial_conditions as build_benjamin_feir_initial_conditions,
)
from solver.gen_data.benjamin_feir_jcp09 import (  # noqa: E402
    deep_water_proxy_depth,
    instability_band_fraction,
    is_supported as benjamin_feir_is_supported,
)
from solver.gen_data.generate_tanaka_dataset_v2 import (  # noqa: E402
    TANAKA_PROFILE_RECONSTRUCTION,
    build_per_case_initial_conditions,
)
from solver.gen_data.jonswap_tma import (  # noqa: E402
    JonswapTmaParameters,
    ResolvedBand,
    build_jonswap_tma_initial_condition,
    finite_depth_angular_frequency,
    is_in_paper_support as jonswap_tma_is_supported,
    sample_jonswap_tma_phases,
)
from solver.gen_data.multi_crest import CrestSpec  # noqa: E402
from solver.gen_data.pipeline.acceptance import (  # noqa: E402
    RefinementTrajectory,
    evaluate_temporal_refinement,
)
from solver.gen_data.pipeline.quality import reasons_from_bits  # noqa: E402
from solver.gen_data.tanaka_population import (  # noqa: E402
    MAIN_DEPTH_BOUNDS,
    MAIN_TOTAL_ALPHA_BOUNDS,
    SEPARATION_TO_DEPTH_RATIO,
    STEEP_ALPHA_BOUNDS,
    STEEP_DEPTH_BOUNDS,
)
from solver.solvers.dno_series_jax import (  # noqa: E402
    build_grid,
    dno_series_eval,
    make_linear_dno_symbol,
)
from solver.solvers.time_integrator import (  # noqa: E402
    SolverParams,
    State,
    apply_lowpass,
    rollout,
)
from solver.tanaka_ICs.modified_tanaka import (  # noqa: E402
    make_default_tanaka_template,
)

jax.config.update("jax_enable_x64", True)

FloatArray: TypeAlias = NDArray[np.float64]
ArrayDict: TypeAlias = dict[str, NDArray[Any]]
CaseFamily: TypeAlias = Literal[
    "stokes_finite",
    "stokes_deep",
    "tanaka",
    "benjamin_feir",
    "jonswap_tma_shallow",
    "jonswap_tma_finite",
    "jonswap_tma_deep",
]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs/full_horizon_refinement_panel_20260725"
PILOT_RANDOM_SEA_STATES = (
    ROOT / "outputs/jonswap_tma_pilot_20260725/states_and_targets.npz"
)


@dataclass(frozen=True)
class NumericalContract:
    """Every numerical choice entering one panel run."""

    nx: int
    length: float
    gravity: float
    dno_order: int
    pad_factor: int
    delivered_wavenumber: float
    filter_fraction: float
    coarse_dt: float
    fine_dt: float
    final_retry_dt: float
    saved_dt: float
    gl2_residual_tolerance: float
    gl2_iteration_cap: int
    refinement_tolerance: float
    relative_floor: float
    gxi_chunk_size: int
    dtype: str
    smoke: bool


@dataclass(frozen=True)
class CaseDefinition:
    """One predeclared physical case, before array construction."""

    case_id: str
    family: CaseFamily
    depth: float
    parameters: dict[str, Any]
    source: str


@dataclass(frozen=True)
class GroupDefinition:
    """Cases sharing one saved-time grid and one batched rollout."""

    name: str
    case_ids: tuple[str, ...]
    intended_terminal_time: float
    realized_terminal_time: float


@dataclass(frozen=True)
class ConstructedPanel:
    """Exact initial arrays and phase data in the case-definition order."""

    x: FloatArray
    eta: FloatArray
    xi: FloatArray
    phase_right: FloatArray
    phase_left: FloatArray
    phase_count: NDArray[np.int32]
    pilot_reuse: dict[str, Any]


def production_contract() -> NumericalContract:
    """Return the frozen production discretization."""

    return NumericalContract(
        nx=1024,
        length=2.0 * math.pi,
        gravity=1.0,
        dno_order=6,
        pad_factor=8,
        delivered_wavenumber=128.0,
        filter_fraction=0.25,
        coarse_dt=0.01,
        fine_dt=0.005,
        final_retry_dt=0.0025,
        saved_dt=0.08,
        gl2_residual_tolerance=1.0e-8,
        gl2_iteration_cap=8,
        refinement_tolerance=1.0e-3,
        relative_floor=1.0e-12,
        gxi_chunk_size=8,
        dtype="float64",
        smoke=False,
    )


def smoke_contract() -> NumericalContract:
    """Return a cheap orchestration check that is not scientific evidence."""

    return NumericalContract(
        nx=256,
        length=2.0 * math.pi,
        gravity=1.0,
        dno_order=2,
        pad_factor=2,
        delivered_wavenumber=32.0,
        filter_fraction=0.25,
        coarse_dt=0.01,
        fine_dt=0.005,
        final_retry_dt=0.0025,
        saved_dt=0.02,
        gl2_residual_tolerance=1.0e-8,
        gl2_iteration_cap=8,
        refinement_tolerance=1.0e-3,
        relative_floor=1.0e-12,
        gxi_chunk_size=2,
        dtype="float64",
        smoke=True,
    )


def build_case_definitions() -> tuple[CaseDefinition, ...]:
    """Return the immutable, hand-auditable boundary/stress panel."""

    tanaka_source = (
        "outputs/tanaka_tangent_stratified_panel_20260723/static_summary.json"
    )
    random_source = "outputs/jonswap_tma_pilot_20260725/summary.json"
    return (
        CaseDefinition(
            case_id="stokes_finite_99000002",
            family="stokes_finite",
            depth=0.03260507316767692,
            parameters={
                "panel_case_id": 99000002,
                "n0": 20,
                "a0": 0.004085235009886773,
                "phase_time": 17.39178747486052,
                "ichoi": 1,
            },
            source=(
                "data/manuscript_ic_panels_v1_20260715/"
                "stokes_finite_ics.npz; nearest Ur-supported boundary case"
            ),
        ),
        CaseDefinition(
            case_id="stokes_deep_98000193",
            family="stokes_deep",
            depth=4.000498966848842,
            parameters={
                "panel_case_id": 98000193,
                "n0": 20,
                "a0": 0.007119249238689001,
                "phase_time": 2.8106701133953482,
                "ichoi": 0,
            },
            source=(
                "data/manuscript_ic_panels_v1_20260715/"
                "stokes_deep_ics.npz; largest-steepness archived panel case"
            ),
        ),
        CaseDefinition(
            case_id="tanaka_main_narrow_steep",
            family="tanaka",
            depth=0.01,
            parameters={
                "profile_reconstruction": TANAKA_PROFILE_RECONSTRUCTION,
                "crests": [
                    {
                        "amplitude": 0.35,
                        "center": math.pi / 2.0,
                        "direction": -1,
                    }
                ],
            },
            source=tanaka_source,
        ),
        CaseDefinition(
            case_id="tanaka_main_upper_corner",
            family="tanaka",
            depth=0.3,
            parameters={
                "profile_reconstruction": TANAKA_PROFILE_RECONSTRUCTION,
                "crests": [
                    {
                        "amplitude": 0.35,
                        "center": math.pi,
                        "direction": 1,
                    }
                ],
            },
            source=tanaka_source,
        ),
        CaseDefinition(
            case_id="tanaka_steep_upper_seam",
            family="tanaka",
            depth=0.35,
            parameters={
                "profile_reconstruction": TANAKA_PROFILE_RECONSTRUCTION,
                "crests": [
                    {
                        "amplitude": 0.45,
                        "center": 0.0,
                        "direction": 1,
                    }
                ],
            },
            source=tanaka_source,
        ),
        CaseDefinition(
            case_id="tanaka_may_case31",
            family="tanaka",
            depth=0.26861433760407505,
            parameters={
                "profile_reconstruction": TANAKA_PROFILE_RECONSTRUCTION,
                "historical_case_id": 31,
                "crests": [
                    {
                        "amplitude": 0.05548354495289499,
                        "center": 3.5548496920089176,
                        "direction": -1,
                    },
                    {
                        "amplitude": 0.053531413616445936,
                        "center": 5.032703928715591,
                        "direction": 1,
                    },
                ],
            },
            source=tanaka_source,
        ),
        CaseDefinition(
            case_id="bf_jcp09_canonical",
            family="benjamin_feir",
            depth=deep_water_proxy_depth(2.0 * math.pi),
            parameters={
                "n_carr": 9,
                "side_offset": 2,
                "eps_carrier": 0.13,
                "eps_pert": 0.10,
                "phase": -math.pi / 4.0,
            },
            source="Xu--Guyenne JCP 2009 equation (33) diagnostic point",
        ),
        CaseDefinition(
            case_id="bf_instability_upper_boundary",
            family="benjamin_feir",
            depth=deep_water_proxy_depth(2.0 * math.pi),
            parameters={
                "n_carr": 20,
                "side_offset": 7,
                "eps_carrier": 0.1245678346,
                "eps_pert": 0.1868935260,
                "phase": 3.9011600778,
            },
            source=(
                "generator seed 20260954; high-mode point at 0.99338 of the "
                "leading deep-water instability boundary"
            ),
        ),
        CaseDefinition(
            case_id="bf_low_carrier_instability_boundary",
            family="benjamin_feir",
            depth=deep_water_proxy_depth(2.0 * math.pi),
            parameters={
                "n_carr": 4,
                "side_offset": 1,
                "eps_carrier": 0.0885725287,
                "eps_pert": 0.1942104051,
                "phase": 1.1493223142,
            },
            source=(
                "generator seed 20269268; lowest-carrier point at 0.99792 "
                "of the leading deep-water instability boundary"
            ),
        ),
        CaseDefinition(
            case_id="jonswap_tma_shallow_pilot_5",
            family="jonswap_tma_shallow",
            depth=0.06328125,
            parameters={
                "pilot_case_id": 5,
                "seed": 2026072505,
                "significant_height": 0.0172546875,
                "peak_wavenumber": 16.0,
                "peak_enhancement": 3.3,
                "right_moving_fraction": 1.0,
            },
            source=random_source,
        ),
        CaseDefinition(
            case_id="jonswap_tma_finite_pilot_14",
            family="jonswap_tma_finite",
            depth=0.14028505520066742,
            parameters={
                "pilot_case_id": 14,
                "seed": 2026072514,
                "significant_height": 0.026875,
                "peak_wavenumber": 8.25,
                "peak_enhancement": 3.3,
                "right_moving_fraction": 1.0,
            },
            source=random_source,
        ),
        CaseDefinition(
            case_id="jonswap_tma_deep_pilot_23",
            family="jonswap_tma_deep",
            depth=6.11422272496926,
            parameters={
                "pilot_case_id": 23,
                "seed": 2026072523,
                "significant_height": 0.026875,
                "peak_wavenumber": 8.25,
                "peak_enhancement": 3.3,
                "right_moving_fraction": 1.0,
            },
            source=random_source,
        ),
    )


def _floor_to_saved_grid(value: float, saved_dt: float) -> float:
    steps = math.floor((value + 1.0e-12 * saved_dt) / saved_dt)
    return steps * saved_dt


def _random_sea_intended_horizon(case: CaseDefinition, gravity: float) -> float:
    parameters = case.parameters
    peak_wavenumber = float(parameters["peak_wavenumber"])
    angular_frequency = float(
        finite_depth_angular_frequency(
            np.asarray([peak_wavenumber]),
            depth=case.depth,
            gravity=gravity,
        )[0]
    )
    return 16.0 * 2.0 * math.pi / angular_frequency


def build_group_definitions(
    cases: tuple[CaseDefinition, ...],
    contract: NumericalContract,
) -> tuple[GroupDefinition, ...]:
    """Assign every case to its declared full-horizon saved-time grid."""

    by_family = {case.case_id: case for case in cases}
    if contract.smoke:
        return (
            GroupDefinition(
                name="smoke_all_families",
                case_ids=tuple(case.case_id for case in cases),
                intended_terminal_time=contract.saved_dt,
                realized_terminal_time=contract.saved_dt,
            ),
        )

    groups = [
        GroupDefinition(
            name="stokes",
            case_ids=(
                "stokes_finite_99000002",
                "stokes_deep_98000193",
            ),
            intended_terminal_time=20.0,
            realized_terminal_time=20.0,
        ),
        GroupDefinition(
            name="tanaka_benjamin_feir",
            case_ids=(
                "tanaka_main_narrow_steep",
                "tanaka_main_upper_corner",
                "tanaka_steep_upper_seam",
                "tanaka_may_case31",
                "bf_jcp09_canonical",
                "bf_instability_upper_boundary",
                "bf_low_carrier_instability_boundary",
            ),
            intended_terminal_time=200.0,
            realized_terminal_time=200.0,
        ),
    ]
    for suffix, case_id in (
        ("shallow", "jonswap_tma_shallow_pilot_5"),
        ("finite", "jonswap_tma_finite_pilot_14"),
        ("deep", "jonswap_tma_deep_pilot_23"),
    ):
        case = by_family[case_id]
        intended = _random_sea_intended_horizon(case, contract.gravity)
        groups.append(
            GroupDefinition(
                name=f"jonswap_tma_{suffix}",
                case_ids=(case_id,),
                intended_terminal_time=intended,
                realized_terminal_time=_floor_to_saved_grid(
                    intended, contract.saved_dt
                ),
            )
        )
    assigned = tuple(case_id for group in groups for case_id in group.case_ids)
    expected = tuple(case.case_id for case in cases)
    if sorted(assigned) != sorted(expected) or len(assigned) != len(set(assigned)):
        raise RuntimeError("group definitions must contain every case exactly once")
    return tuple(groups)


def _random_sea_parameters(case: CaseDefinition) -> JonswapTmaParameters:
    parameters = case.parameters
    return JonswapTmaParameters(
        depth=case.depth,
        significant_height=float(parameters["significant_height"]),
        peak_wavenumber=float(parameters["peak_wavenumber"]),
        peak_enhancement=float(parameters["peak_enhancement"]),
        right_moving_fraction=float(parameters["right_moving_fraction"]),
    )


def case_support_record(
    case: CaseDefinition,
    contract: NumericalContract,
) -> dict[str, Any]:
    """Evaluate only the predeclared family-domain condition."""

    if case.family == "stokes_finite":
        mode = int(case.parameters["n0"])
        wavenumber = 2.0 * math.pi * mode / contract.length
        ursell_upper_bound = float(
            finite_depth_stokes_ursell_upper_bound(
                wavenumber,
                case.depth,
                contract.gravity,
                float(case.parameters["a0"]),
            )
        )
        return {
            "inside": bool(
                math.isfinite(ursell_upper_bound)
                and ursell_upper_bound <= FINITE_DEPTH_STOKES_URSELL_LIMIT
            ),
            "criterion": ("conservative continuous-profile Ursell upper bound <= 26"),
            "ursell_upper_bound": ursell_upper_bound,
            "ursell_limit": FINITE_DEPTH_STOKES_URSELL_LIMIT,
        }
    if case.family == "stokes_deep":
        mode = int(case.parameters["n0"])
        carrier_depth = 2.0 * math.pi * mode * case.depth / contract.length
        return {
            "inside": bool(carrier_depth >= 5.0),
            "criterion": "deep branch with carrier k h >= 5",
            "carrier_depth": carrier_depth,
        }
    if case.family == "benjamin_feir":
        parameters = case.parameters
        phase = float(parameters["phase"])
        phase_modulo_period = phase % (2.0 * math.pi)
        expected_depth = deep_water_proxy_depth(contract.length)
        inside = bool(
            benjamin_feir_is_supported(
                int(parameters["n_carr"]),
                int(parameters["side_offset"]),
                float(parameters["eps_carrier"]),
                float(parameters["eps_pert"]),
            )
            and math.isfinite(phase)
            and math.isclose(
                case.depth,
                expected_depth,
                rel_tol=0.0,
                abs_tol=1.0e-14,
            )
        )
        return {
            "inside": inside,
            "criterion": (
                "declared deep JCP09-style support, finite sideband phase "
                "interpreted modulo 2pi, leading instability band, and "
                "fixed deep-water proxy depth"
            ),
            "phase": phase,
            "phase_modulo_2pi": phase_modulo_period,
            "expected_depth": expected_depth,
            "instability_band_fraction": float(
                instability_band_fraction(
                    int(parameters["n_carr"]),
                    int(parameters["side_offset"]),
                    float(parameters["eps_carrier"]),
                )
            ),
        }
    if case.family.startswith("jonswap_tma_"):
        stratum = case.family.removeprefix("jonswap_tma_")
        parameters = _random_sea_parameters(case)
        return {
            "inside": jonswap_tma_is_supported(
                parameters,
                stratum=stratum,  # type: ignore[arg-type]
                length=contract.length,
            ),
            "criterion": f"declared JONSWAP/TMA {stratum} support",
        }
    if case.family == "tanaka":
        crests = case.parameters["crests"]
        amplitudes = [float(crest["amplitude"]) for crest in crests]
        centers = [float(crest["center"]) for crest in crests]
        directions = [int(crest["direction"]) for crest in crests]
        crest_count = len(crests)
        total_amplitude = sum(amplitudes)
        sorted_centers = sorted(centers)
        cyclic_gaps = (
            [contract.length]
            if crest_count == 1
            else [
                (sorted_centers[(index + 1) % crest_count] - center) % contract.length
                for index, center in enumerate(sorted_centers)
            ]
            if crest_count > 0
            else []
        )
        achieved_separation = min(cyclic_gaps) if cyclic_gaps else None
        main_supported = bool(
            crest_count in (1, 2, 3)
            and MAIN_DEPTH_BOUNDS[0] <= case.depth <= MAIN_DEPTH_BOUNDS[1]
            and MAIN_TOTAL_ALPHA_BOUNDS[0]
            <= total_amplitude
            <= MAIN_TOTAL_ALPHA_BOUNDS[1]
        )
        steep_supported = bool(
            crest_count == 1
            and STEEP_DEPTH_BOUNDS[0] <= case.depth <= STEEP_DEPTH_BOUNDS[1]
            and STEEP_ALPHA_BOUNDS[0] <= total_amplitude <= STEEP_ALPHA_BOUNDS[1]
        )
        finite_parameters = all(
            math.isfinite(value) for value in (case.depth, *amplitudes, *centers)
        )
        separated = bool(
            crest_count == 1
            or (
                achieved_separation is not None
                and achieved_separation >= SEPARATION_TO_DEPTH_RATIO * case.depth
            )
        )
        return {
            "inside": bool(
                finite_parameters
                and all(amplitude > 0.0 for amplitude in amplitudes)
                and all(0.0 <= center < contract.length for center in centers)
                and all(direction in (-1, 1) for direction in directions)
                and separated
                and case.parameters.get("profile_reconstruction")
                == TANAKA_PROFILE_RECONSTRUCTION
                and (main_supported or steep_supported)
            ),
            "criterion": (
                "declared main or one-crest steep Tanaka support, periodic "
                "center gap at least 3h, and tangent-Hermite reconstruction"
            ),
            "crest_count": crest_count,
            "sum_steepness": total_amplitude,
            "minimum_cyclic_center_gap": achieved_separation,
            "main_supported": main_supported,
            "steep_supported": steep_supported,
        }
    raise AssertionError(f"unhandled family {case.family!r}")


def _construct_stokes(
    case: CaseDefinition,
    x: jax.Array,
    contract: NumericalContract,
) -> tuple[jax.Array, jax.Array]:
    parameters = case.parameters
    return stokes_eta_xi(
        x=x,
        time=jnp.asarray(parameters["phase_time"], dtype=jnp.float64),
        n0=int(parameters["n0"]),
        a0=float(parameters["a0"]),
        length=contract.length,
        depth=case.depth,
        gravity=contract.gravity,
        ichoi=int(parameters["ichoi"]),
    )


def _construct_tanaka_batch(
    cases: tuple[CaseDefinition, ...],
    contract: NumericalContract,
) -> tuple[jax.Array, jax.Array]:
    template = make_default_tanaka_template(
        depth=1.0,
        gravity=contract.gravity,
        direction=1,
        nx=contract.nx,
        length=contract.length,
        center=0.0,
        dno_order=contract.dno_order,
        pad_factor=contract.pad_factor,
    )
    specifications = [
        [
            CrestSpec(
                amplitude=float(crest["amplitude"]),
                center=float(crest["center"]),
                direction=int(crest["direction"]),
            )
            for crest in case.parameters["crests"]
        ]
        for case in cases
    ]
    return build_per_case_initial_conditions(
        template_params=template,
        case_h_ref=np.asarray([case.depth for case in cases], dtype=np.float64),
        case_specs=specifications,
        length=contract.length,
        nx=contract.nx,
        gravity=contract.gravity,
    )


def _construct_benjamin_feir_batch(
    cases: tuple[CaseDefinition, ...],
    x: jax.Array,
    contract: NumericalContract,
) -> tuple[jax.Array, jax.Array]:
    parameters = {
        name: np.asarray([case.parameters[name] for case in cases], dtype=dtype)
        for name, dtype in (
            ("n_carr", np.int32),
            ("side_offset", np.int32),
            ("eps_carrier", np.float64),
            ("eps_pert", np.float64),
            ("phase", np.float64),
        )
    }
    return build_benjamin_feir_initial_conditions(
        x=x,
        parameters=parameters,
        length=contract.length,
        gravity=contract.gravity,
        dtype=jnp.float64,
    )


def _random_sea_band(contract: NumericalContract) -> ResolvedBand:
    transition = 0.75 * contract.delivered_wavenumber
    return ResolvedBand(
        length=contract.length,
        maximum_wavenumber=contract.delivered_wavenumber,
        transition_wavenumber=transition,
    )


def _construct_random_sea(
    case: CaseDefinition,
    x: FloatArray,
    contract: NumericalContract,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    band = _random_sea_band(contract)
    phases = sample_jonswap_tma_phases(
        np.random.default_rng(int(case.parameters["seed"])),
        band=band,
    )
    state = build_jonswap_tma_initial_condition(
        x,
        parameters=_random_sea_parameters(case),
        phase_right=phases[0],
        phase_left=phases[1],
        band=band,
        gravity=contract.gravity,
    )
    return state.eta, state.xi, phases[0], phases[1]


def _pilot_reuse_diagnostics(
    cases: tuple[CaseDefinition, ...],
    eta: FloatArray,
    xi: FloatArray,
    contract: NumericalContract,
) -> dict[str, Any]:
    """Verify the production random states against the existing pilot arrays."""

    if contract.smoke or not PILOT_RANDOM_SEA_STATES.exists():
        return {
            "checked": False,
            "reason": (
                "smoke uses a reduced band"
                if contract.smoke
                else f"{PILOT_RANDOM_SEA_STATES} is unavailable"
            ),
        }
    with np.load(PILOT_RANDOM_SEA_STATES, allow_pickle=False) as archive:
        pilot_ids = np.asarray(archive["case_id"], dtype=np.int64)
        pilot_eta = np.asarray(archive["eta"], dtype=np.float64)
        pilot_xi = np.asarray(archive["xi"], dtype=np.float64)

    records: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not case.family.startswith("jonswap_tma_"):
            continue
        pilot_id = int(case.parameters["pilot_case_id"])
        match = np.flatnonzero(pilot_ids == pilot_id)
        if match.size != 1:
            raise RuntimeError(f"pilot case {pilot_id} is not uniquely available")
        pilot_index = int(match[0])
        eta_error = float(np.max(np.abs(eta[index] - pilot_eta[pilot_index])))
        xi_error = float(np.max(np.abs(xi[index] - pilot_xi[pilot_index])))
        records.append(
            {
                "case_id": case.case_id,
                "pilot_case_id": pilot_id,
                "maximum_absolute_eta_difference": eta_error,
                "maximum_absolute_xi_difference": xi_error,
            }
        )
        if max(eta_error, xi_error) > 5.0e-14:
            raise RuntimeError(
                f"{case.case_id} does not reproduce its saved pilot state"
            )
    return {
        "checked": True,
        "source": str(PILOT_RANDOM_SEA_STATES.relative_to(ROOT)),
        "cases": records,
    }


def construct_panel(
    cases: tuple[CaseDefinition, ...],
    contract: NumericalContract,
) -> ConstructedPanel:
    """Construct every exact initial state, then apply the common projection."""

    x, wavenumbers = build_grid(contract.nx, contract.length)
    x_jax = jnp.asarray(x, dtype=jnp.float64)
    eta_by_id: dict[str, jax.Array | FloatArray] = {}
    xi_by_id: dict[str, jax.Array | FloatArray] = {}
    phase_by_id: dict[str, tuple[FloatArray, FloatArray]] = {}

    for case in cases:
        if case.family.startswith("stokes_"):
            eta_by_id[case.case_id], xi_by_id[case.case_id] = _construct_stokes(
                case, x_jax, contract
            )

    tanaka_cases = tuple(case for case in cases if case.family == "tanaka")
    tanaka_eta, tanaka_xi = _construct_tanaka_batch(tanaka_cases, contract)
    for index, case in enumerate(tanaka_cases):
        eta_by_id[case.case_id] = tanaka_eta[index]
        xi_by_id[case.case_id] = tanaka_xi[index]

    bf_cases = tuple(case for case in cases if case.family == "benjamin_feir")
    bf_eta, bf_xi = _construct_benjamin_feir_batch(bf_cases, x_jax, contract)
    for index, case in enumerate(bf_cases):
        eta_by_id[case.case_id] = bf_eta[index]
        xi_by_id[case.case_id] = bf_xi[index]

    x_numpy = np.asarray(x, dtype=np.float64)
    for case in cases:
        if case.family.startswith("jonswap_tma_"):
            eta, xi, phase_right, phase_left = _construct_random_sea(
                case, x_numpy, contract
            )
            eta_by_id[case.case_id] = eta
            xi_by_id[case.case_id] = xi
            phase_by_id[case.case_id] = (phase_right, phase_left)

    eta = jnp.stack(
        [jnp.asarray(eta_by_id[case.case_id], dtype=jnp.float64) for case in cases]
    )
    xi = jnp.stack(
        [jnp.asarray(xi_by_id[case.case_id], dtype=jnp.float64) for case in cases]
    )
    k = jnp.asarray(wavenumbers, dtype=jnp.float64)
    eta = apply_lowpass(eta, k, contract.filter_fraction)
    xi = apply_lowpass(xi, k, contract.filter_fraction)
    xi -= jnp.mean(xi, axis=-1, keepdims=True)
    eta_numpy = np.asarray(jax.device_get(eta), dtype=np.float64)
    xi_numpy = np.asarray(jax.device_get(xi), dtype=np.float64)

    maximum_phase_count = max(
        (right.size for right, _ in phase_by_id.values()),
        default=0,
    )
    phase_right = np.full((len(cases), maximum_phase_count), np.nan, dtype=np.float64)
    phase_left = np.full_like(phase_right, np.nan)
    phase_count = np.zeros(len(cases), dtype=np.int32)
    for index, case in enumerate(cases):
        if case.case_id not in phase_by_id:
            continue
        right, left = phase_by_id[case.case_id]
        phase_count[index] = right.size
        phase_right[index, : right.size] = right
        phase_left[index, : left.size] = left

    if not np.isfinite(eta_numpy).all() or not np.isfinite(xi_numpy).all():
        raise RuntimeError("a predeclared initial state is nonfinite")
    depths = np.asarray([case.depth for case in cases], dtype=np.float64)
    if not np.all(depths[:, None] + eta_numpy > 0.0):
        raise RuntimeError("a predeclared initial state touches the bottom")
    pilot_reuse = _pilot_reuse_diagnostics(cases, eta_numpy, xi_numpy, contract)
    return ConstructedPanel(
        x=x_numpy,
        eta=eta_numpy,
        xi=xi_numpy,
        phase_right=phase_right,
        phase_left=phase_left,
        phase_count=phase_count,
        pilot_reuse=pilot_reuse,
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _config_fingerprint(
    cases: tuple[CaseDefinition, ...],
    groups: tuple[GroupDefinition, ...],
    contract: NumericalContract,
    source_files: dict[str, str | None],
) -> str:
    payload = {
        "contract": asdict(contract),
        "cases": [asdict(case) for case in cases],
        "groups": [asdict(group) for group in groups],
        "source_files": source_files,
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _write_npz_atomic(path: Path, arrays: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_hash(path: Path) -> str | None:
    return _sha256(path) if path.exists() else None


def _source_record() -> dict[str, Any]:
    revision = "unavailable"
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        revision = result.stdout.strip()
    sources = (
        Path(__file__).resolve(),
        ROOT / "solver/solvers/time_integrator.py",
        ROOT / "solver/solvers/dno_series_jax.py",
        ROOT / "solver/data/stokes_truth_jax.py",
        ROOT / "solver/gen_data/generate_tanaka_dataset_v2.py",
        ROOT / "solver/gen_data/multi_crest.py",
        ROOT / "solver/gen_data/tanaka_population.py",
        ROOT / "solver/gen_data/benjamin_feir_jcp09.py",
        ROOT / "solver/gen_data/jonswap_tma.py",
        ROOT / "solver/gen_data/pipeline/acceptance.py",
        ROOT / "solver/tanaka_ICs/modified_tanaka.py",
    )
    return {
        "git_revision": revision,
        "files": {str(path.relative_to(ROOT)): _file_hash(path) for path in sources},
    }


def _case_json_array(cases: tuple[CaseDefinition, ...]) -> NDArray[np.str_]:
    return np.asarray([_canonical_json(asdict(case)) for case in cases])


def write_case_manifest(
    path: Path,
    panel: ConstructedPanel,
    cases: tuple[CaseDefinition, ...],
    contract: NumericalContract,
    fingerprint: str,
    *,
    overwrite: bool,
) -> None:
    """Persist the exact ICs and random phases, or validate a prior copy."""

    if path.exists() and not overwrite:
        with np.load(path, allow_pickle=False) as archive:
            stored = str(np.asarray(archive["config_fingerprint"]).item())
            identifiers = tuple(np.asarray(archive["case_id"]).tolist())
        if stored != fingerprint or identifiers != tuple(
            case.case_id for case in cases
        ):
            raise RuntimeError(
                f"{path} belongs to another configuration; use --overwrite"
            )
        return
    metadata = {
        "schema": "paper_full_horizon_case_manifest_v1",
        "config_fingerprint": fingerprint,
        "contract": asdict(contract),
        "pilot_reuse": panel.pilot_reuse,
    }
    _write_npz_atomic(
        path,
        {
            "schema": np.asarray(metadata["schema"]),
            "config_fingerprint": np.asarray(fingerprint),
            "metadata_json": np.asarray(_canonical_json(metadata)),
            "case_id": np.asarray([case.case_id for case in cases]),
            "family": np.asarray([case.family for case in cases]),
            "depth": np.asarray([case.depth for case in cases], dtype=np.float64),
            "case_spec_json": _case_json_array(cases),
            "x": panel.x,
            "eta0": panel.eta,
            "xi0": panel.xi,
            "phase_right": panel.phase_right,
            "phase_left": panel.phase_left,
            "phase_count": panel.phase_count,
        },
    )


def _group_indices(
    group: GroupDefinition,
    cases: tuple[CaseDefinition, ...],
) -> NDArray[np.int64]:
    index_by_id = {case.case_id: index for index, case in enumerate(cases)}
    return np.asarray(
        [index_by_id[case_id] for case_id in group.case_ids],
        dtype=np.int64,
    )


def _saved_times(group: GroupDefinition, contract: NumericalContract) -> FloatArray:
    count = int(round(group.realized_terminal_time / contract.saved_dt))
    times = contract.saved_dt * np.arange(count + 1, dtype=np.float64)
    if not math.isclose(
        float(times[-1]),
        group.realized_terminal_time,
        rel_tol=0.0,
        abs_tol=2.0e-13,
    ):
        raise RuntimeError(f"{group.name} terminal time is off the saved grid")
    return times


def _evaluate_saved_gxi(
    eta: jax.Array,
    xi: jax.Array,
    *,
    depths: jax.Array,
    wavenumbers: jax.Array,
    contract: NumericalContract,
) -> FloatArray:
    """Evaluate saved DNO targets in bounded-memory time chunks."""

    result = np.empty(eta.shape, dtype=np.float64)
    chunk_size = contract.gxi_chunk_size
    for start in range(0, eta.shape[0], chunk_size):
        stop = min(start + chunk_size, eta.shape[0])
        count = stop - start
        eta_chunk = eta[start:stop]
        xi_chunk = xi[start:stop]
        if count < chunk_size:
            padding = chunk_size - count
            eta_chunk = jnp.pad(eta_chunk, ((0, padding), (0, 0), (0, 0)))
            xi_chunk = jnp.pad(xi_chunk, ((0, padding), (0, 0), (0, 0)))
        values = dno_series_eval(
            eta_chunk,
            xi_chunk,
            wavenumbers,
            depths,
            contract.dno_order,
            pad_factor=contract.pad_factor,
        )
        values = apply_lowpass(values, wavenumbers, contract.filter_fraction)
        values -= jnp.mean(values, axis=-1, keepdims=True)
        result[start:stop] = np.asarray(
            jax.device_get(values[:count]), dtype=np.float64
        )
    return result


def _substeps(dt: float, saved_dt: float) -> int:
    count = int(round(saved_dt / dt))
    if not math.isclose(count * dt, saved_dt, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("saved_dt must be an integer multiple of each inner dt")
    return count


def run_arm(
    *,
    group: GroupDefinition,
    group_cases: tuple[CaseDefinition, ...],
    eta0: FloatArray,
    xi0: FloatArray,
    contract: NumericalContract,
    dt: float,
) -> tuple[ArrayDict, dict[str, float]]:
    """Run one fixed-step arm and materialize fields plus GL2 telemetry."""

    times = _saved_times(group, contract)
    _, wavenumbers = build_grid(contract.nx, contract.length)
    k = jnp.asarray(wavenumbers, dtype=jnp.float64)
    depths = jnp.asarray([case.depth for case in group_cases], dtype=jnp.float64)[
        :, None
    ]
    parameters = SolverParams(
        nx=contract.nx,
        length=contract.length,
        depth=depths,
        gravity=contract.gravity,
        dno_order=contract.dno_order,
        pad_factor=contract.pad_factor,
        filter_fraction=contract.filter_fraction,
        k=k,
        g0=make_linear_dno_symbol(k, depths),
    )
    start = perf_counter()
    payload = rollout(
        State(
            eta=jnp.asarray(eta0, dtype=jnp.float64),
            xi=jnp.asarray(xi0, dtype=jnp.float64),
        ),
        jnp.asarray(times, dtype=jnp.float64),
        parameters,
        save_gxi=False,
        substeps_per_interval=_substeps(dt, contract.saved_dt),
        method="gl2_if",
        implicit_iterations=contract.gl2_iteration_cap,
        implicit_residual_tolerance=contract.gl2_residual_tolerance,
        implicit_relaxation=1.0,
        zero_mean_xi=True,
    )
    jax.block_until_ready(payload["xi"])
    rollout_seconds = perf_counter() - start

    start = perf_counter()
    gxi = _evaluate_saved_gxi(
        payload["eta"],
        payload["xi"],
        depths=depths,
        wavenumbers=k,
        contract=contract,
    )
    gxi_seconds = perf_counter() - start

    start = perf_counter()
    arrays: ArrayDict = {
        name: np.asarray(jax.device_get(value)) for name, value in payload.items()
    }
    arrays["gxi"] = gxi
    arrays["dt"] = np.asarray(dt, dtype=np.float64)
    host_transfer_seconds = perf_counter() - start
    del payload
    gc.collect()
    return arrays, {
        "rollout_including_compilation": rollout_seconds,
        "saved_gxi_including_compilation": gxi_seconds,
        "remaining_host_transfer": host_transfer_seconds,
        "compute_total": (rollout_seconds + gxi_seconds + host_transfer_seconds),
    }


def _arm_path(output_dir: Path, group_name: str, dt: float) -> Path:
    known_labels = {
        0.01: "0p010",
        0.005: "0p005",
        0.0025: "0p0025",
    }
    dt_label = known_labels.get(dt, f"{dt:.12g}".replace(".", "p"))
    return output_dir / f"{group_name}_dt_{dt_label}.npz"


def save_arm(
    path: Path,
    arrays: ArrayDict,
    *,
    group: GroupDefinition,
    group_cases: tuple[CaseDefinition, ...],
    contract: NumericalContract,
    fingerprint: str,
    timings: dict[str, float],
) -> float:
    start = perf_counter()
    metadata = {
        "schema": "paper_full_horizon_refinement_arm_v1",
        "config_fingerprint": fingerprint,
        "group": asdict(group),
        "contract": asdict(contract),
        "timing_seconds": timings,
    }
    stored: dict[str, Any] = dict(arrays)
    stored.update(
        {
            "schema": np.asarray(metadata["schema"]),
            "config_fingerprint": np.asarray(fingerprint),
            "metadata_json": np.asarray(_canonical_json(metadata)),
            "case_id": np.asarray([case.case_id for case in group_cases]),
            "family": np.asarray([case.family for case in group_cases]),
            "depth": np.asarray([case.depth for case in group_cases], dtype=np.float64),
            "case_spec_json": _case_json_array(group_cases),
        }
    )
    _write_npz_atomic(path, stored)
    return perf_counter() - start


def load_arm(
    path: Path,
    *,
    expected_fingerprint: str,
    expected_case_ids: tuple[str, ...],
) -> tuple[ArrayDict, dict[str, float]]:
    with np.load(path, allow_pickle=False) as archive:
        fingerprint = str(np.asarray(archive["config_fingerprint"]).item())
        identifiers = tuple(np.asarray(archive["case_id"]).tolist())
        if fingerprint != expected_fingerprint or identifiers != expected_case_ids:
            raise RuntimeError(
                f"{path} belongs to another configuration; use --overwrite"
            )
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        excluded = {
            "schema",
            "config_fingerprint",
            "metadata_json",
            "case_id",
            "family",
            "depth",
            "case_spec_json",
        }
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name not in excluded
        }
    return arrays, {
        name: float(value) for name, value in metadata["timing_seconds"].items()
    }


def run_or_load_arm(
    *,
    output_dir: Path,
    group: GroupDefinition,
    group_cases: tuple[CaseDefinition, ...],
    eta0: FloatArray,
    xi0: FloatArray,
    contract: NumericalContract,
    fingerprint: str,
    dt: float,
    overwrite: bool,
) -> tuple[ArrayDict, dict[str, Any]]:
    path = _arm_path(output_dir, group.name, dt)
    if path.exists() and not overwrite:
        start = perf_counter()
        arrays, original_timings = load_arm(
            path,
            expected_fingerprint=fingerprint,
            expected_case_ids=group.case_ids,
        )
        return arrays, {
            "artifact": path.name,
            "artifact_sha256": _sha256(path),
            "reused": True,
            "load_seconds": perf_counter() - start,
            "original_compute_seconds": original_timings,
        }

    arrays, timings = run_arm(
        group=group,
        group_cases=group_cases,
        eta0=eta0,
        xi0=xi0,
        contract=contract,
        dt=dt,
    )
    write_seconds = save_arm(
        path,
        arrays,
        group=group,
        group_cases=group_cases,
        contract=contract,
        fingerprint=fingerprint,
        timings=timings,
    )
    return arrays, {
        "artifact": path.name,
        "artifact_sha256": _sha256(path),
        "reused": False,
        "timing_seconds": {**timings, "artifact_write": write_seconds},
    }


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def summarize_stage_telemetry(
    arm: ArrayDict,
    *,
    case_index: int,
    iteration_cap: int,
) -> dict[str, Any]:
    """Summarize one case while retaining every step value in the NPZ."""

    residual = np.asarray(arm["gl2_stage_residual"][:, case_index], dtype=np.float64)
    iterations = np.asarray(arm["gl2_iterations"][:, case_index], dtype=np.int32)
    converged = np.asarray(arm["gl2_converged"][:, case_index], dtype=np.bool_)
    stage_finite = np.asarray(arm["gl2_stage_finite"][:, case_index], dtype=np.bool_)
    state_finite = np.asarray(arm["gl2_state_finite"][:, case_index], dtype=np.bool_)
    hit_cap = np.asarray(arm["gl2_hit_iteration_cap"][:, case_index], dtype=np.bool_)
    failures = ~converged
    first_failure = int(np.flatnonzero(failures)[0]) if np.any(failures) else -1
    finite_residual = residual[np.isfinite(residual)]
    maximum = float(np.max(finite_residual)) if finite_residual.size else float("inf")
    histogram = np.bincount(iterations, minlength=iteration_cap + 1)[
        : iteration_cap + 1
    ]
    step_times = np.asarray(arm["gl2_step_times"], dtype=np.float64)
    step_dts = np.asarray(arm["gl2_step_dts"], dtype=np.float64)
    return {
        "number_of_steps": int(residual.size),
        "all_stages_converged": bool(np.all(converged)),
        "maximum_stage_residual": _finite_or_none(maximum),
        "maximum_stage_residual_is_finite": math.isfinite(maximum),
        "failed_stage_count": int(np.count_nonzero(failures)),
        "stage_nonfinite_count": int(np.count_nonzero(~stage_finite)),
        "state_nonfinite_count": int(np.count_nonzero(~state_finite)),
        "iteration_cap_hit_count": int(np.count_nonzero(hit_cap)),
        "iteration_histogram": {
            str(index): int(count) for index, count in enumerate(histogram)
        },
        "first_failed_step": first_failure,
        "first_failed_step_time": (
            float(step_times[first_failure]) if first_failure >= 0 else None
        ),
        "first_failed_step_dt": (
            float(step_dts[first_failure]) if first_failure >= 0 else None
        ),
    }


def _trajectory(
    arm: ArrayDict,
    *,
    case_index: int,
    stages_solved: bool,
) -> RefinementTrajectory:
    return RefinementTrajectory(
        times=np.asarray(arm["times"], dtype=np.float64),
        eta=np.asarray(arm["eta"][:, case_index], dtype=np.float64),
        xi=np.asarray(arm["xi"][:, case_index], dtype=np.float64),
        gxi=np.asarray(arm["gxi"][:, case_index], dtype=np.float64),
        complete=True,
        gl2_stages_solved=stages_solved,
    )


def _reason_names(bits: int) -> list[str]:
    return [reason.name for reason in reasons_from_bits(bits)]


def evaluate_group(
    *,
    group: GroupDefinition,
    group_cases: tuple[CaseDefinition, ...],
    coarse: ArrayDict,
    fine: ArrayDict,
    contract: NumericalContract,
) -> tuple[list[dict[str, Any]], bool]:
    """Evaluate support and one paired-refinement decision for every case."""

    if not np.array_equal(coarse["times"], fine["times"]):
        raise RuntimeError(f"{group.name} arms do not share saved times")
    records: list[dict[str, Any]] = []
    for case_index, case in enumerate(group_cases):
        support = case_support_record(case, contract)
        coarse_stages = summarize_stage_telemetry(
            coarse,
            case_index=case_index,
            iteration_cap=contract.gl2_iteration_cap,
        )
        fine_stages = summarize_stage_telemetry(
            fine,
            case_index=case_index,
            iteration_cap=contract.gl2_iteration_cap,
        )
        metrics, decision = evaluate_temporal_refinement(
            _trajectory(
                coarse,
                case_index=case_index,
                stages_solved=bool(coarse_stages["all_stages_converged"]),
            ),
            _trajectory(
                fine,
                case_index=case_index,
                stages_solved=bool(fine_stages["all_stages_converged"]),
            ),
            depth=case.depth,
            gravity=contract.gravity,
            length=contract.length,
            maximum_wavenumber=contract.delivered_wavenumber,
            tolerance=contract.refinement_tolerance,
            relative_floor=contract.relative_floor,
        )
        maximum_error = float(metrics.maximum_error)
        accepted = bool(support["inside"] and decision.accepted)
        records.append(
            {
                "case_id": case.case_id,
                "family": case.family,
                "depth": case.depth,
                "source": case.source,
                "parameters": case.parameters,
                "support": support,
                "intended_terminal_time": group.intended_terminal_time,
                "realized_terminal_time": group.realized_terminal_time,
                "saved_time_count": int(coarse["times"].size),
                "coarse_stage_telemetry": coarse_stages,
                "fine_stage_telemetry": fine_stages,
                "dimensionless_delivered_band_defect": {
                    "eta": _finite_or_none(metrics.eta_error),
                    "xi": _finite_or_none(metrics.xi_error),
                    "gxi": _finite_or_none(metrics.gxi_error),
                    "maximum": _finite_or_none(maximum_error),
                    "maximum_is_finite": math.isfinite(maximum_error),
                    "tolerance": contract.refinement_tolerance,
                },
                "quality_decision": {
                    "scope": decision.scope.value,
                    "required_bits": decision.required_bits,
                    "evaluated_bits": decision.evaluated_bits,
                    "failed_bits": decision.failed_bits,
                    "failed_reasons": _reason_names(decision.failed_bits),
                    "refinement_accepted": decision.accepted,
                },
                "accepted": accepted,
            }
        )
    return records, all(bool(record["accepted"]) for record in records)


def _subset_arm_cases(
    arm: ArrayDict,
    case_indices: NDArray[np.int64],
) -> ArrayDict:
    """Select batch members without changing saved-time or step coordinates."""

    time_batch_fields = {
        "eta",
        "xi",
        "gxi",
        "gl2_stage_residual",
        "gl2_iterations",
        "gl2_converged",
        "gl2_stage_finite",
        "gl2_state_finite",
        "gl2_hit_iteration_cap",
    }
    batch_fields = {
        "gl2_all_stages_converged",
        "gl2_max_stage_residual",
        "gl2_first_failed_step",
    }
    return {
        name: (
            np.take(value, case_indices, axis=1)
            if name in time_batch_fields
            else np.take(value, case_indices, axis=0)
            if name in batch_fields
            else value
        )
        for name, value in arm.items()
    }


def final_halving_warrant(
    primary_record: dict[str, Any],
    fine_arm: ArrayDict,
    *,
    case_index: int,
    depth: float,
) -> tuple[bool, str]:
    """Return whether a failed primary pair can meaningfully be halved once.

    The second primary arm becomes the coarse member of the retry pair.  A
    retry therefore cannot produce an acceptable comparison when that arm is
    itself nonfinite, touches the bottom, or has an unsolved GL2 stage.
    """

    if bool(primary_record["quality_decision"]["refinement_accepted"]):
        return False, "primary dt=0.01/0.005 pair accepted"
    if not bool(primary_record["support"]["inside"]):
        return False, "case lies outside its declared family support"
    if not bool(np.all(np.asarray(fine_arm["gl2_converged"][:, case_index]))):
        return False, "dt=0.005 arm has an unsolved GL2 stage"

    eta = np.asarray(fine_arm["eta"][:, case_index], dtype=np.float64)
    xi = np.asarray(fine_arm["xi"][:, case_index], dtype=np.float64)
    gxi = np.asarray(fine_arm["gxi"][:, case_index], dtype=np.float64)
    if not (
        np.isfinite(eta).all() and np.isfinite(xi).all() and np.isfinite(gxi).all()
    ):
        return False, "dt=0.005 arm contains a nonfinite field"
    if float(np.min(depth + eta)) <= 0.0:
        return False, "dt=0.005 arm leaves the water-wave graph domain"
    return True, (
        "primary pair failed and its dt=0.005 arm is a valid coarse "
        "trajectory for one final halving"
    )


def _merge_final_halving_records(
    primary_records: list[dict[str, Any]],
    retry_records: list[dict[str, Any]],
    warrant_by_id: dict[str, tuple[bool, str]],
    contract: NumericalContract,
) -> tuple[list[dict[str, Any]], bool]:
    """Attach retry evidence while preserving the primary-pair decision."""

    retry_by_id = {record["case_id"]: record for record in retry_records}
    merged: list[dict[str, Any]] = []
    for primary in primary_records:
        case_id = str(primary["case_id"])
        primary_accepted = bool(primary["accepted"])
        warranted, reason = warrant_by_id[case_id]
        record = dict(primary)
        record["primary_pair_accepted"] = primary_accepted
        if warranted:
            retry = retry_by_id[case_id]
            retry_accepted = bool(retry["accepted"])
            record["final_halving_retry"] = {
                "triggered": True,
                "reason": reason,
                "coarse_dt": contract.fine_dt,
                "fine_dt": contract.final_retry_dt,
                "coarse_stage_telemetry": retry["coarse_stage_telemetry"],
                "fine_stage_telemetry": retry["fine_stage_telemetry"],
                "dimensionless_delivered_band_defect": retry[
                    "dimensionless_delivered_band_defect"
                ],
                "quality_decision": retry["quality_decision"],
                "accepted": retry_accepted,
            }
            effective_accepted = retry_accepted
        else:
            record["final_halving_retry"] = {
                "triggered": False,
                "reason": reason,
            }
            effective_accepted = primary_accepted
        record["accepted_after_final_halving_policy"] = effective_accepted
        record["accepted"] = effective_accepted
        merged.append(record)
    return merged, all(bool(record["accepted"]) for record in merged)


def _device_record() -> list[dict[str, Any]]:
    return [
        {
            "id": int(device.id),
            "platform": device.platform,
            "device_kind": device.device_kind,
        }
        for device in jax.devices()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a reduced N=256, M=2, pad=2, T=.02 CPU-sized "
            "orchestration check. Its results are not production evidence."
        ),
    )
    parser.add_argument(
        "--allow-cpu-production",
        action="store_true",
        help="Permit the expensive production contract when no GPU is visible.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace matching manifest and arm artifacts instead of resuming.",
    )
    return parser.parse_args()


def _remove_expected_artifacts(
    output_dir: Path,
    groups: tuple[GroupDefinition, ...],
) -> None:
    paths = [
        output_dir / "case_manifest.npz",
        output_dir / "summary.json",
    ]
    paths.extend(
        _arm_path(output_dir, group.name, dt)
        for group in groups
        for dt in (0.01, 0.005)
    )
    paths.extend(
        _arm_path(
            output_dir,
            f"{group.name}_final_halving",
            0.0025,
        )
        for group in groups
    )
    for path in paths:
        path.unlink(missing_ok=True)


def run_panel(args: argparse.Namespace) -> dict[str, Any]:
    contract = smoke_contract() if args.smoke else production_contract()
    devices = _device_record()
    if (
        not contract.smoke
        and not args.allow_cpu_production
        and not any(device["platform"] in {"gpu", "cuda", "rocm"} for device in devices)
    ):
        raise RuntimeError(
            "production panel requires a visible GPU; use "
            "--allow-cpu-production only if the long CPU run is intentional"
        )

    cases = build_case_definitions()
    groups = build_group_definitions(cases, contract)
    support = [case_support_record(case, contract) for case in cases]
    if not all(bool(record["inside"]) for record in support):
        raise RuntimeError("a predeclared case lies outside its family support")
    source = _source_record()
    fingerprint = _config_fingerprint(
        cases,
        groups,
        contract,
        source["files"],
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        _remove_expected_artifacts(output_dir, groups)

    panel_start = perf_counter()
    panel = construct_panel(cases, contract)
    construction_seconds = perf_counter() - panel_start
    case_manifest = output_dir / "case_manifest.npz"
    write_case_manifest(
        case_manifest,
        panel,
        cases,
        contract,
        fingerprint,
        overwrite=args.overwrite,
    )

    summary_path = output_dir / "summary.json"
    summary: dict[str, Any] = {
        "schema": "paper_full_horizon_refinement_panel_v1",
        "status": "running",
        "production_evidence": not contract.smoke,
        "started_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, *sys.argv],
        "config_fingerprint": fingerprint,
        "source": source,
        "jax": {
            "version": jax.__version__,
            "x64_enabled": bool(jax.config.x64_enabled),
            "devices": devices,
        },
        "contract": asdict(contract),
        "method_statement": {
            "target": (
                "float64 order-6 pad-8 Craig--Sulem DNO, projected with the "
                "rollout to |k| <= 128"
                if not contract.smoke
                else "reduced non-production orchestration target"
            ),
            "refinement_pair": (
                "GL2 integrating-factor dt=0.01 versus dt=0.005 at identical "
                "saved times"
            ),
            "final_halving": (
                "only a failed primary case whose dt=0.005 arm is finite, "
                "graph-valued, and stage-converged is recomputed at "
                "dt=0.0025 and judged from the dt=0.005/0.0025 pair"
            ),
            "stage_solve": (
                "dimensionless fixed-point residual <= 1e-8, at most 8 "
                "Picard updates per stage pair"
            ),
            "acceptance": (
                "maximum dimensionless delivered-band eta/xi/G(eta)xi "
                "trajectory defect <= 1e-3; incomplete, nonfinite, "
                "bottom-contact, or unsolved-stage trajectories have "
                "infinite defect"
            ),
        },
        "case_manifest": {
            "artifact": case_manifest.name,
            "artifact_sha256": _sha256(case_manifest),
            "construction_seconds": construction_seconds,
            "pilot_reuse": panel.pilot_reuse,
        },
        "cases": [
            {**asdict(case), "support": support_record}
            for case, support_record in zip(cases, support, strict=True)
        ],
        "groups": [],
    }
    _write_json_atomic(summary_path, summary)

    case_index = {case.case_id: index for index, case in enumerate(cases)}
    all_cases_accepted = True
    total_start = perf_counter()
    try:
        for group in groups:
            indices = np.asarray(
                [case_index[case_id] for case_id in group.case_ids],
                dtype=np.int64,
            )
            group_cases = tuple(cases[int(index)] for index in indices)
            coarse, coarse_record = run_or_load_arm(
                output_dir=output_dir,
                group=group,
                group_cases=group_cases,
                eta0=panel.eta[indices],
                xi0=panel.xi[indices],
                contract=contract,
                fingerprint=fingerprint,
                dt=contract.coarse_dt,
                overwrite=args.overwrite,
            )
            fine, fine_record = run_or_load_arm(
                output_dir=output_dir,
                group=group,
                group_cases=group_cases,
                eta0=panel.eta[indices],
                xi0=panel.xi[indices],
                contract=contract,
                fingerprint=fingerprint,
                dt=contract.fine_dt,
                overwrite=args.overwrite,
            )
            primary_records, primary_group_accepted = evaluate_group(
                group=group,
                group_cases=group_cases,
                coarse=coarse,
                fine=fine,
                contract=contract,
            )

            warrant_by_id = {
                case.case_id: final_halving_warrant(
                    primary_records[local_index],
                    fine,
                    case_index=local_index,
                    depth=case.depth,
                )
                for local_index, case in enumerate(group_cases)
            }
            retry_local_indices = np.asarray(
                [
                    local_index
                    for local_index, case in enumerate(group_cases)
                    if warrant_by_id[case.case_id][0]
                ],
                dtype=np.int64,
            )
            retry_records: list[dict[str, Any]] = []
            retry_summary: dict[str, Any] = {
                "triggered": bool(retry_local_indices.size),
                "coarse_dt": contract.fine_dt,
                "fine_dt": contract.final_retry_dt,
                "case_ids": [
                    group_cases[int(index)].case_id for index in retry_local_indices
                ],
                "eligibility": {
                    case_id: {
                        "warranted": warranted,
                        "reason": reason,
                    }
                    for case_id, (warranted, reason) in warrant_by_id.items()
                },
            }
            if retry_local_indices.size:
                retry_cases = tuple(
                    group_cases[int(index)] for index in retry_local_indices
                )
                retry_group = GroupDefinition(
                    name=f"{group.name}_final_halving",
                    case_ids=tuple(case.case_id for case in retry_cases),
                    intended_terminal_time=group.intended_terminal_time,
                    realized_terminal_time=group.realized_terminal_time,
                )
                retry_global_indices = indices[retry_local_indices]
                retry_finer, retry_finer_record = run_or_load_arm(
                    output_dir=output_dir,
                    group=retry_group,
                    group_cases=retry_cases,
                    eta0=panel.eta[retry_global_indices],
                    xi0=panel.xi[retry_global_indices],
                    contract=contract,
                    fingerprint=fingerprint,
                    dt=contract.final_retry_dt,
                    overwrite=args.overwrite,
                )
                retry_coarse = _subset_arm_cases(fine, retry_local_indices)
                retry_records, retry_all_accepted = evaluate_group(
                    group=retry_group,
                    group_cases=retry_cases,
                    coarse=retry_coarse,
                    fine=retry_finer,
                    contract=contract,
                )
                retry_summary.update(
                    {
                        "finer": retry_finer_record,
                        "all_retry_cases_accepted": retry_all_accepted,
                    }
                )
                del retry_coarse, retry_finer

            case_records, group_accepted = _merge_final_halving_records(
                primary_records,
                retry_records,
                warrant_by_id,
                contract,
            )
            all_cases_accepted &= group_accepted
            summary["groups"].append(
                {
                    **asdict(group),
                    "coarse": coarse_record,
                    "fine": fine_record,
                    "primary_all_cases_accepted": primary_group_accepted,
                    "final_halving_retry": retry_summary,
                    "all_cases_accepted": group_accepted,
                    "cases": case_records,
                }
            )
            _write_json_atomic(summary_path, summary)
            del coarse, fine
            gc.collect()
    except BaseException as error:
        summary["status"] = "failed"
        summary["failed_at"] = datetime.now().astimezone().isoformat()
        summary["exception"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        summary["elapsed_seconds"] = perf_counter() - total_start
        _write_json_atomic(summary_path, summary)
        raise

    summary["status"] = "complete"
    summary["completed_at"] = datetime.now().astimezone().isoformat()
    summary["all_cases_accepted"] = all_cases_accepted
    summary["elapsed_seconds"] = perf_counter() - total_start
    summary["result"] = (
        "PASS" if all_cases_accepted else "FAIL: inspect per-case decisions"
    )
    _write_json_atomic(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_panel(args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "production_evidence": summary["production_evidence"],
                "all_cases_accepted": summary["all_cases_accepted"],
                "output": str(args.output_dir.resolve() / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
