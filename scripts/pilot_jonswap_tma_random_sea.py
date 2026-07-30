"""Run a deterministic CPU pilot of the public JONSWAP/TMA constructor."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, TypeAlias

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solver.gen_data.jonswap_tma import (
    PAPER_PEAK_ENHANCEMENTS,
    PAPER_RIGHT_MOVING_FRACTIONS,
    JonswapTmaParameters,
    JonswapTmaState,
    RandomSeaStratum,
    ResolvedBand,
    build_jonswap_tma_initial_condition,
    is_in_paper_support,
    sample_jonswap_tma_phases,
    tma_depth_factor,
)
from solver.gen_data.pipeline.acceptance import (
    RefinementTrajectory,
    evaluate_temporal_refinement,
)
from solver.solvers.dno_series_jax import (
    build_grid,
    dno_series_eval,
    make_linear_dno_symbol,
)
from solver.solvers.time_integrator import (
    SolverParams,
    State,
    apply_lowpass,
    rollout,
)


FloatArray: TypeAlias = NDArray[np.float64]
LENGTH = 2.0 * np.pi
GRAVITY = 1.0
PAPER_BAND = ResolvedBand(
    length=LENGTH,
    maximum_wavenumber=128.0,
    transition_wavenumber=96.0,
)


@dataclass(frozen=True)
class PilotSpecification:
    """One predeclared parameter point in the balanced pilot."""

    case_id: int
    stratum: RandomSeaStratum
    parameters: JonswapTmaParameters
    seed: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/jonswap_tma_pilot_20260725"),
    )
    parser.add_argument("--nx-fine", type=int, default=1024)
    parser.add_argument("--nx-coarse", type=int, default=512)
    return parser.parse_args()


def balanced_discrete_parameters() -> tuple[tuple[float, float], ...]:
    """Return the nine ``(gamma, r_d)`` combinations exactly once."""

    return tuple(
        (peak_enhancement, right_fraction)
        for peak_enhancement in PAPER_PEAK_ENHANCEMENTS
        for right_fraction in PAPER_RIGHT_MOVING_FRACTIONS
    )


def shallow_parameters(
    index: int,
    *,
    peak_enhancement: float,
    right_fraction: float,
) -> JonswapTmaParameters:
    """Construct one parameter-only admissible shallow point."""

    peak_modes = (16.0, 18.0, 20.0, 22.0, 24.0, 16.0, 20.0, 22.0, 24.0)
    depth_wavenumbers = np.linspace(0.2, 1.5, 9)
    height_fractions = (0.0, 1.0, 0.5, 0.75, 0.25, 0.9, 0.1, 0.6, 0.4)
    peak_wavenumber = peak_modes[index]
    depth_wavenumber = float(depth_wavenumbers[index])
    depth = depth_wavenumber / peak_wavenumber
    upper_relative_height = min(0.16, 0.15 / depth_wavenumber)
    relative_height = 0.03 + height_fractions[index] * (upper_relative_height - 0.03)
    return JonswapTmaParameters(
        depth=depth,
        significant_height=2.0 * depth * relative_height,
        peak_wavenumber=peak_wavenumber,
        peak_enhancement=peak_enhancement,
        right_moving_fraction=right_fraction,
    )


def rectangular_parameters(
    index: int,
    *,
    stratum: RandomSeaStratum,
    peak_enhancement: float,
    right_fraction: float,
) -> JonswapTmaParameters:
    """Construct one point spanning a finite- or deep-depth rectangle."""

    peak_wavenumbers = np.linspace(2.0, 12.0, 9)
    height_indices = np.asarray((0, 8, 4, 6, 2, 7, 1, 5, 3))
    depth_indices = np.asarray((8, 0, 4, 2, 6, 1, 7, 3, 5))
    significant_heights = np.linspace(0.005, 0.03, 9)[height_indices]
    if stratum == "finite":
        depths = np.geomspace(0.1, 1.5, 9)[depth_indices]
    else:
        depths = np.geomspace(5.0, 25.0, 9)[depth_indices]
    return JonswapTmaParameters(
        depth=float(depths[index]),
        significant_height=float(significant_heights[index]),
        peak_wavenumber=float(peak_wavenumbers[index]),
        peak_enhancement=peak_enhancement,
        right_moving_fraction=right_fraction,
    )


def build_specifications() -> tuple[PilotSpecification, ...]:
    """Build 27 cases: nine in each of the three strata."""

    specifications: list[PilotSpecification] = []
    case_id = 0
    for stratum in ("shallow", "finite", "deep"):
        for index, (peak_enhancement, right_fraction) in enumerate(
            balanced_discrete_parameters()
        ):
            if stratum == "shallow":
                parameters = shallow_parameters(
                    index,
                    peak_enhancement=peak_enhancement,
                    right_fraction=right_fraction,
                )
            else:
                parameters = rectangular_parameters(
                    index,
                    stratum=stratum,
                    peak_enhancement=peak_enhancement,
                    right_fraction=right_fraction,
                )
            specifications.append(
                PilotSpecification(
                    case_id=case_id,
                    stratum=stratum,
                    parameters=parameters,
                    seed=2026072500 + case_id,
                )
            )
            case_id += 1
    return tuple(specifications)


def periodic_grid(nx: int) -> FloatArray:
    """Return the endpoint-excluded periodic grid."""

    return np.arange(nx, dtype=np.float64) * LENGTH / nx


def build_states(
    specifications: tuple[PilotSpecification, ...],
    *,
    nx: int,
) -> tuple[JonswapTmaState, ...]:
    """Build all pilot states on one spatial grid."""

    x = periodic_grid(nx)
    states: list[JonswapTmaState] = []
    for specification in specifications:
        phases = sample_jonswap_tma_phases(
            np.random.default_rng(specification.seed), band=PAPER_BAND
        )
        states.append(
            build_jonswap_tma_initial_condition(
                x,
                parameters=specification.parameters,
                phase_right=phases[0],
                phase_left=phases[1],
                band=PAPER_BAND,
                gravity=GRAVITY,
            )
        )
    return tuple(states)


def spectral_derivative(field: FloatArray) -> FloatArray:
    """Differentiate a real periodic field spectrally."""

    nx = field.size
    wavenumbers = 2.0 * np.pi * np.fft.rfftfreq(nx, d=LENGTH / nx)
    return np.fft.irfft(1j * wavenumbers * np.fft.rfft(field), n=nx)


def relative_band_error(
    coarse: FloatArray,
    fine: FloatArray,
    *,
    maximum_wavenumber: int,
) -> float:
    """Compare normalized Fourier coefficients on a common fixed band."""

    coarse_coefficients = np.fft.rfft(coarse)[: maximum_wavenumber + 1] / coarse.size
    fine_coefficients = np.fft.rfft(fine)[: maximum_wavenumber + 1] / fine.size
    weights = np.ones(maximum_wavenumber + 1, dtype=np.float64)
    weights[1:] = 2.0
    difference = np.sqrt(
        np.sum(weights * np.abs(coarse_coefficients - fine_coefficients) ** 2)
    )
    reference = np.sqrt(np.sum(weights * np.abs(fine_coefficients) ** 2))
    return float(difference / max(reference, 1.0e-30))


def evaluate_dno(
    states: tuple[JonswapTmaState, ...],
    specifications: tuple[PilotSpecification, ...],
    *,
    nx: int,
) -> tuple[FloatArray, float]:
    """Evaluate the order-six DNO for every state and return wall time."""

    _, wavenumbers = build_grid(nx, LENGTH)
    eta = jnp.asarray(np.stack([state.eta for state in states]))
    xi = jnp.asarray(np.stack([state.xi for state in states]))
    depths = jnp.asarray(
        [specification.parameters.depth for specification in specifications]
    )[:, None]
    start = perf_counter()
    values = dno_series_eval(
        eta,
        xi,
        jnp.asarray(wavenumbers),
        depths,
        6,
        pad_factor=8,
    )
    values.block_until_ready()
    wall_seconds = perf_counter() - start
    return np.asarray(values, dtype=np.float64), wall_seconds


def state_record(
    specification: PilotSpecification,
    state: JonswapTmaState,
    *,
    gxi_coarse: FloatArray,
    gxi_fine: FloatArray,
) -> dict[str, Any]:
    """Compute scalar constructor and fixed-band diagnostics for one case."""

    parameters = specification.parameters
    eta = state.eta
    xi = state.xi
    eta_spectrum = np.abs(np.fft.rfft(eta) / eta.size) ** 2
    positive_energy = eta_spectrum[1:129]
    high_quarter_fraction = float(
        np.sum(positive_energy[96:]) / max(float(np.sum(positive_energy)), 1.0e-30)
    )
    expected_linear_energy = (
        GRAVITY * LENGTH * (parameters.significant_height / 4.0) ** 2
    )
    flat_wavenumbers = 2.0 * np.pi * np.fft.rfftfreq(eta.size, d=LENGTH / eta.size)
    flat_symbol = flat_wavenumbers * np.tanh(flat_wavenumbers * parameters.depth)
    flat_dno_xi = np.fft.irfft(flat_symbol * np.fft.rfft(xi), n=eta.size)
    realized_linear_energy = float(
        0.5 * LENGTH * np.mean(xi * flat_dno_xi + GRAVITY * eta**2)
    )
    maximum_cell_energy = float(np.max(state.spectrum.energy_fractions))
    half_maximum_cells = int(
        np.count_nonzero(state.spectrum.energy_fractions >= 0.5 * maximum_cell_energy)
    )
    return {
        "case_id": specification.case_id,
        "stratum": specification.stratum,
        "seed": specification.seed,
        "parameters": {
            "depth": parameters.depth,
            "significant_height": parameters.significant_height,
            "peak_wavenumber": parameters.peak_wavenumber,
            "peak_enhancement": parameters.peak_enhancement,
            "right_moving_fraction": parameters.right_moving_fraction,
            "peak_depth": parameters.peak_wavenumber * parameters.depth,
            "relative_height": (
                parameters.significant_height / (2.0 * parameters.depth)
            ),
            "peak_steepness": (
                parameters.peak_wavenumber * parameters.significant_height / 2.0
            ),
        },
        "in_declared_support": is_in_paper_support(
            parameters, stratum=specification.stratum, length=LENGTH
        ),
        "finite_eta_xi_gxi": bool(
            np.isfinite(eta).all()
            and np.isfinite(xi).all()
            and np.isfinite(gxi_fine).all()
        ),
        "mean_eta": float(np.mean(eta)),
        "mean_xi": float(np.mean(xi)),
        "realized_height_ratio": float(
            4.0 * np.std(eta) / parameters.significant_height
        ),
        "minimum_water_column_over_depth": float(
            np.min(parameters.depth + eta) / parameters.depth
        ),
        "maximum_abs_eta_over_depth": float(np.max(np.abs(eta)) / parameters.depth),
        "maximum_abs_slope": float(np.max(np.abs(spectral_derivative(eta)))),
        "high_quarter_eta_energy_fraction": high_quarter_fraction,
        "half_maximum_spectral_cells": half_maximum_cells,
        "realized_peak_wavenumber": float(
            state.spectrum.wavenumbers[int(np.argmax(state.spectrum.energy_fractions))]
        ),
        "tma_factor_at_peak": float(
            tma_depth_factor(
                np.asarray([parameters.peak_wavenumber * parameters.depth])
            )[0]
        ),
        "linear_energy_relative_error": float(
            abs(realized_linear_energy - expected_linear_energy)
            / expected_linear_energy
        ),
        "gxi_fixed_band_relative_error_n512_n1024": relative_band_error(
            gxi_coarse, gxi_fine, maximum_wavenumber=128
        ),
    }


def _refinement_trajectory(
    payload: dict[str, jax.Array],
    *,
    case_index: int,
) -> RefinementTrajectory:
    """Convert one batched rollout member to the acceptance representation."""

    return RefinementTrajectory(
        times=np.asarray(payload["times"], dtype=np.float64),
        eta=np.asarray(payload["eta"][:, case_index], dtype=np.float64),
        xi=np.asarray(payload["xi"][:, case_index], dtype=np.float64),
        gxi=np.asarray(payload["gxi"][:, case_index], dtype=np.float64),
    )


def run_short_refinement_smoke(
    specifications: tuple[PilotSpecification, ...],
    states: tuple[JonswapTmaState, ...],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run one saved interval for the steepest case in each stratum."""

    selected_indices = [
        max(
            (
                index
                for index, record in enumerate(records)
                if record["stratum"] == stratum
            ),
            key=lambda index: records[index]["maximum_abs_slope"],
        )
        for stratum in ("shallow", "finite", "deep")
    ]
    nx = states[0].eta.size
    _, wavenumbers = build_grid(nx, LENGTH)
    k = jnp.asarray(wavenumbers, dtype=jnp.float64)
    eta = apply_lowpass(
        jnp.asarray(np.stack([states[index].eta for index in selected_indices])),
        k,
        0.25,
    )
    xi = apply_lowpass(
        jnp.asarray(np.stack([states[index].xi for index in selected_indices])),
        k,
        0.25,
    )
    xi = xi - jnp.mean(xi, axis=-1, keepdims=True)
    depths = jnp.asarray(
        [specifications[index].parameters.depth for index in selected_indices],
        dtype=jnp.float64,
    )[:, None]
    solver_parameters = SolverParams(
        nx=nx,
        length=LENGTH,
        depth=depths,
        gravity=GRAVITY,
        dno_order=6,
        pad_factor=8,
        filter_fraction=0.25,
        k=k,
        g0=make_linear_dno_symbol(k, depths),
    )
    common = {
        "initial_state": State(eta=eta, xi=xi),
        "times": jnp.asarray([0.0, 0.08], dtype=jnp.float64),
        "params": solver_parameters,
        "save_gxi": True,
        "method": "gl2_if",
        "implicit_iterations": 8,
        "zero_mean_xi": True,
    }
    start = perf_counter()
    coarse = rollout(**common, substeps_per_interval=8)
    coarse["gxi"].block_until_ready()
    coarse_seconds = perf_counter() - start
    start = perf_counter()
    fine = rollout(**common, substeps_per_interval=16)
    fine["gxi"].block_until_ready()
    fine_seconds = perf_counter() - start

    cases: list[dict[str, Any]] = []
    for batch_index, specification_index in enumerate(selected_indices):
        specification = specifications[specification_index]
        metrics, decision = evaluate_temporal_refinement(
            _refinement_trajectory(coarse, case_index=batch_index),
            _refinement_trajectory(fine, case_index=batch_index),
            depth=specification.parameters.depth,
            gravity=GRAVITY,
            length=LENGTH,
            maximum_wavenumber=PAPER_BAND.maximum_wavenumber,
            tolerance=1.0e-3,
        )
        cases.append(
            {
                "case_id": specification.case_id,
                "stratum": specification.stratum,
                "accepted": decision.accepted,
                "eta_error": metrics.eta_error,
                "xi_error": metrics.xi_error,
                "gxi_error": metrics.gxi_error,
                "maximum_error": metrics.maximum_error,
            }
        )
    return {
        "saved_times": [0.0, 0.08],
        "coarse_substeps": 8,
        "fine_substeps": 16,
        "implicit_iterations": 8,
        "tolerance": 1.0e-3,
        "coarse_wall_seconds": coarse_seconds,
        "fine_wall_seconds": fine_seconds,
        "all_accepted": all(bool(case["accepted"]) for case in cases),
        "maximum_error": max(float(case["maximum_error"]) for case in cases),
        "cases": cases,
    }


def scalar_summary(values: FloatArray) -> dict[str, float]:
    """Summarize one finite vector."""

    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
    }


def summarize_records(
    records: list[dict[str, Any]],
    *,
    constructor_seconds: float,
    dno_coarse_seconds: float,
    dno_fine_seconds: float,
    short_refinement: dict[str, Any],
) -> dict[str, Any]:
    """Build the machine-readable pilot summary."""

    metric_names = (
        "realized_height_ratio",
        "minimum_water_column_over_depth",
        "maximum_abs_eta_over_depth",
        "maximum_abs_slope",
        "high_quarter_eta_energy_fraction",
        "half_maximum_spectral_cells",
        "linear_energy_relative_error",
        "gxi_fixed_band_relative_error_n512_n1024",
    )
    by_stratum: dict[str, Any] = {}
    for stratum in ("shallow", "finite", "deep"):
        selected = [record for record in records if record["stratum"] == stratum]
        by_stratum[stratum] = {
            "n_cases": len(selected),
            "all_in_declared_support": all(
                bool(record["in_declared_support"]) for record in selected
            ),
            "all_finite_eta_xi_gxi": all(
                bool(record["finite_eta_xi_gxi"]) for record in selected
            ),
            "metrics": {
                name: scalar_summary(np.asarray([record[name] for record in selected]))
                for name in metric_names
            },
        }
    return {
        "schema": "jonswap_tma_constructor_pilot_v1",
        "n_cases": len(records),
        "domain": {
            "length": LENGTH,
            "nx_coarse": 512,
            "nx_fine": 1024,
            "maximum_wavenumber": PAPER_BAND.maximum_wavenumber,
            "transition_wavenumber": PAPER_BAND.transition_wavenumber,
            "dno_order": 6,
            "dno_pad_factor": 8,
        },
        "decision_rule": (
            "No phase-dependent geometry or appearance gate is used. "
            "This pilot checks construction, support, and fixed-band spatial "
            "agreement; full trajectories still require paired time refinement."
        ),
        "timing_seconds": {
            "construct_both_grids": constructor_seconds,
            "dno_n512": dno_coarse_seconds,
            "dno_n1024": dno_fine_seconds,
        },
        "short_refinement": short_refinement,
        "by_stratum": by_stratum,
        "cases": records,
    }


def plot_representatives(
    specifications: tuple[PilotSpecification, ...],
    states: tuple[JonswapTmaState, ...],
    records: list[dict[str, Any]],
    *,
    output_path: Path,
) -> None:
    """Plot the steepest realized case from each stratum."""

    figure, axes = plt.subplots(3, 2, figsize=(9.0, 7.4), constrained_layout=True)
    x_over_length = periodic_grid(states[0].eta.size) / LENGTH
    colors = {"shallow": "#b54a3a", "finite": "#6b4c9a", "deep": "#2a8c6a"}
    for row, stratum in enumerate(("shallow", "finite", "deep")):
        indices = [
            index
            for index, record in enumerate(records)
            if record["stratum"] == stratum
        ]
        selected = max(indices, key=lambda index: records[index]["maximum_abs_slope"])
        specification = specifications[selected]
        state = states[selected]
        record = records[selected]
        parameters = specification.parameters
        color = colors[stratum]
        axes[row, 0].plot(
            x_over_length, state.eta / parameters.depth, color=color, lw=1.0
        )
        axes[row, 0].axhline(-1.0, color="black", lw=0.7, ls=":")
        axes[row, 0].set_ylabel(r"$\eta/h$")
        axes[row, 0].set_title(
            f"{stratum}: case {specification.case_id}, "
            rf"$k_ph={parameters.peak_wavenumber * parameters.depth:.2f}$, "
            rf"$H_s/(2h)={parameters.significant_height / (2 * parameters.depth):.3f}$"
        )

        weights = state.spectrum.energy_fractions
        positive = weights > 0.0
        axes[row, 1].semilogy(
            state.spectrum.wavenumbers[positive],
            weights[positive] / np.max(weights),
            color=color,
            lw=1.0,
        )
        axes[row, 1].axvline(parameters.peak_wavenumber, color="black", lw=0.7, ls="--")
        axes[row, 1].axvline(
            PAPER_BAND.transition_wavenumber,
            color="0.5",
            lw=0.7,
            ls=":",
        )
        axes[row, 1].set_title(
            rf"weights; $\|\eta_x\|_\infty={record['maximum_abs_slope']:.3f}$"
        )
        axes[row, 1].set_ylim(1.0e-8, 2.0)
        axes[row, 1].set_ylabel("energy relative to largest mode")
    axes[-1, 0].set_xlabel(r"$x/L$")
    axes[-1, 1].set_xlabel("wavenumber")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run the pilot and write JSON, NPZ, and PNG outputs."""

    args = parse_args()
    if args.nx_coarse != 512 or args.nx_fine != 1024:
        raise ValueError(
            "this fixed-band pilot currently requires nx-coarse=512 and nx-fine=1024"
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specifications = build_specifications()
    if not all(
        is_in_paper_support(
            specification.parameters,
            stratum=specification.stratum,
            length=LENGTH,
        )
        for specification in specifications
    ):
        raise ValueError("pilot specification escaped its declared stratum")

    start = perf_counter()
    coarse_states = build_states(specifications, nx=args.nx_coarse)
    fine_states = build_states(specifications, nx=args.nx_fine)
    constructor_seconds = perf_counter() - start
    coarse_gxi, coarse_seconds = evaluate_dno(
        coarse_states, specifications, nx=args.nx_coarse
    )
    fine_gxi, fine_seconds = evaluate_dno(fine_states, specifications, nx=args.nx_fine)
    records = [
        state_record(
            specification,
            state,
            gxi_coarse=coarse_gxi[index],
            gxi_fine=fine_gxi[index],
        )
        for index, (specification, state) in enumerate(zip(specifications, fine_states))
    ]
    short_refinement = run_short_refinement_smoke(specifications, fine_states, records)
    summary = summarize_records(
        records,
        constructor_seconds=constructor_seconds,
        dno_coarse_seconds=coarse_seconds,
        dno_fine_seconds=fine_seconds,
        short_refinement=short_refinement,
    )
    summary["domain"]["nx_coarse"] = args.nx_coarse
    summary["domain"]["nx_fine"] = args.nx_fine
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "states_and_targets.npz",
        eta=np.stack([state.eta for state in fine_states]),
        xi=np.stack([state.xi for state in fine_states]),
        gxi=fine_gxi,
        depth=np.asarray(
            [specification.parameters.depth for specification in specifications]
        ),
        case_id=np.asarray([specification.case_id for specification in specifications]),
    )
    plot_representatives(
        specifications,
        fine_states,
        records,
        output_path=output_dir / "representative_stress_cases.png",
    )
    print(json.dumps(summary["by_stratum"], indent=2))
    print(json.dumps(summary["timing_seconds"], indent=2))
    print(output_dir)


if __name__ == "__main__":
    main()
