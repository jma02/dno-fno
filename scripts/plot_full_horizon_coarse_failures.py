"""Plot and record the two failed coarse full-horizon trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
IntArray: TypeAlias = NDArray[np.int32]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "outputs/full_horizon_refinement_panel_20260725/"
    "tanaka_benjamin_feir_dt_0p010.npz"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs/full_horizon_refinement_panel_20260725"
FAILED_CASE_IDS = (
    "tanaka_steep_upper_seam",
    "bf_jcp09_canonical",
)
HIGH_BAND = (80, 128)
DELIVERED_BAND = (1, 128)


@dataclass(frozen=True)
class CaseHistory:
    """Finite saved-frame diagnostics for one trajectory."""

    case_id: str
    family: str
    depth: float
    times: FloatArray
    eta: FloatArray
    xi: FloatArray
    gxi: FloatArray
    hamiltonian: FloatArray
    signed_hamiltonian_drift: FloatArray
    slope: FloatArray
    water_column_fraction: FloatArray
    eta_high_fraction: FloatArray
    xi_high_fraction: FloatArray
    gxi_high_fraction: FloatArray
    eta_generated_tail: FloatArray
    xi_generated_tail: FloatArray
    gxi_generated_tail: FloatArray
    first_failed_step: int
    first_failed_step_time: float | None
    step_times: FloatArray
    stage_residual: FloatArray
    iterations: IntArray
    converged: BoolArray
    stage_finite: BoolArray
    state_finite: BoolArray
    hit_iteration_cap: BoolArray


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spectral_derivative(field: FloatArray) -> FloatArray:
    """Differentiate a field on the length-2π grid."""

    modes = np.arange(field.shape[-1] // 2 + 1, dtype=np.float64)
    return np.asarray(
        np.fft.irfft(
            1j * modes * np.fft.rfft(field, axis=-1),
            n=field.shape[-1],
            axis=-1,
        ),
        dtype=np.float64,
    )


def spectral_diagnostics(
    field: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return high-band fraction and generated-tail energy.

    The generated-tail energy is the positive increase in high-band energy,
    normalized by the initial energy over the delivered modes.
    """

    nx = field.shape[-1]
    coefficients = np.fft.rfft(field, axis=-1) / nx
    high_energy = np.sum(
        np.abs(
            coefficients[..., HIGH_BAND[0] : HIGH_BAND[1] + 1]
        )
        ** 2,
        axis=-1,
    )
    delivered_energy = np.sum(
        np.abs(
            coefficients[..., DELIVERED_BAND[0] : DELIVERED_BAND[1] + 1]
        )
        ** 2,
        axis=-1,
    )
    tiny = np.finfo(np.float64).tiny
    high_fraction = high_energy / np.maximum(delivered_energy, tiny)
    generated_tail = np.maximum(high_energy - high_energy[0], 0.0) / max(
        float(delivered_energy[0]),
        tiny,
    )
    return (
        np.asarray(high_fraction, dtype=np.float64),
        np.asarray(generated_tail, dtype=np.float64),
    )


def build_history(
    *,
    case_index: int,
    case_id: str,
    family: str,
    depth: float,
    times: FloatArray,
    eta: FloatArray,
    xi: FloatArray,
    gxi: FloatArray,
    step_times: FloatArray,
    stage_residual: FloatArray,
    iterations: IntArray,
    converged: BoolArray,
    stage_finite: BoolArray,
    state_finite: BoolArray,
    hit_iteration_cap: BoolArray,
    gravity: float,
) -> CaseHistory:
    """Compute all finite saved-frame diagnostics for one case."""

    saved_finite = (
        np.all(np.isfinite(eta[:, case_index]), axis=-1)
        & np.all(np.isfinite(xi[:, case_index]), axis=-1)
        & np.all(np.isfinite(gxi[:, case_index]), axis=-1)
    )
    finite_indices = np.flatnonzero(saved_finite)
    if finite_indices.size == 0:
        raise RuntimeError(f"{case_id} has no finite saved frame")
    stop = int(finite_indices[-1]) + 1
    case_times = np.asarray(times[:stop], dtype=np.float64)
    case_eta = np.asarray(eta[:stop, case_index], dtype=np.float64)
    case_xi = np.asarray(xi[:stop, case_index], dtype=np.float64)
    case_gxi = np.asarray(gxi[:stop, case_index], dtype=np.float64)
    dx = 2.0 * math.pi / case_eta.shape[-1]
    hamiltonian = 0.5 * dx * np.sum(
        case_xi * case_gxi + gravity * case_eta**2,
        axis=-1,
    )
    signed_drift = (hamiltonian - hamiltonian[0]) / abs(hamiltonian[0])
    slope = np.max(np.abs(spectral_derivative(case_eta)), axis=-1)
    water_fraction = np.min((depth + case_eta) / depth, axis=-1)
    eta_high_fraction, eta_generated_tail = spectral_diagnostics(case_eta)
    xi_high_fraction, xi_generated_tail = spectral_diagnostics(case_xi)
    gxi_high_fraction, gxi_generated_tail = spectral_diagnostics(case_gxi)

    case_converged = np.asarray(converged[:, case_index], dtype=np.bool_)
    failures = np.flatnonzero(~case_converged)
    first_failed_step = int(failures[0]) if failures.size else -1
    first_failed_step_time = (
        float(step_times[first_failed_step])
        if first_failed_step >= 0
        else None
    )
    return CaseHistory(
        case_id=case_id,
        family=family,
        depth=depth,
        times=case_times,
        eta=case_eta,
        xi=case_xi,
        gxi=case_gxi,
        hamiltonian=np.asarray(hamiltonian, dtype=np.float64),
        signed_hamiltonian_drift=np.asarray(signed_drift, dtype=np.float64),
        slope=np.asarray(slope, dtype=np.float64),
        water_column_fraction=np.asarray(water_fraction, dtype=np.float64),
        eta_high_fraction=eta_high_fraction,
        xi_high_fraction=xi_high_fraction,
        gxi_high_fraction=gxi_high_fraction,
        eta_generated_tail=eta_generated_tail,
        xi_generated_tail=xi_generated_tail,
        gxi_generated_tail=gxi_generated_tail,
        first_failed_step=first_failed_step,
        first_failed_step_time=first_failed_step_time,
        step_times=np.asarray(step_times, dtype=np.float64),
        stage_residual=np.asarray(
            stage_residual[:, case_index], dtype=np.float64
        ),
        iterations=np.asarray(iterations[:, case_index], dtype=np.int32),
        converged=case_converged,
        stage_finite=np.asarray(
            stage_finite[:, case_index], dtype=np.bool_
        ),
        state_finite=np.asarray(
            state_finite[:, case_index], dtype=np.bool_
        ),
        hit_iteration_cap=np.asarray(
            hit_iteration_cap[:, case_index], dtype=np.bool_
        ),
    )


def finite_or_none(value: float) -> float | None:
    """Return a finite JSON number or null."""

    return float(value) if math.isfinite(float(value)) else None


def first_threshold_time(
    times: FloatArray,
    values: FloatArray,
    threshold: float,
) -> float | None:
    """Return the first saved time at or above a threshold."""

    indices = np.flatnonzero(values >= threshold)
    return float(times[indices[0]]) if indices.size else None


def first_iteration_record(
    history: CaseHistory,
    threshold: int,
) -> dict[str, Any] | None:
    """Record the first step requiring at least a given iteration count."""

    indices = np.flatnonzero(history.iterations >= threshold)
    if indices.size == 0:
        return None
    index = int(indices[0])
    return {
        "step_index": index,
        "step_start_time": float(history.step_times[index]),
        "iterations": int(history.iterations[index]),
        "residual": finite_or_none(history.stage_residual[index]),
        "converged": bool(history.converged[index]),
    }


def case_record(history: CaseHistory) -> dict[str, Any]:
    """Serialize the evidence for one case without non-standard JSON values."""

    last = history.times.size - 1
    failure = history.first_failed_step
    hamiltonian_drift = np.abs(history.signed_hamiltonian_drift)
    failure_record: dict[str, Any] | None = None
    if failure >= 0:
        failure_record = {
            "step_index": failure,
            "step_start_time": float(history.step_times[failure]),
            "step_end_time": float(
                history.step_times[failure]
                + (
                    history.step_times[1] - history.step_times[0]
                    if history.step_times.size > 1
                    else 0.0
                )
            ),
            "residual": finite_or_none(history.stage_residual[failure]),
            "iterations": int(history.iterations[failure]),
            "converged": bool(history.converged[failure]),
            "stage_finite": bool(history.stage_finite[failure]),
            "state_finite": bool(history.state_finite[failure]),
            "hit_iteration_cap": bool(
                history.hit_iteration_cap[failure]
            ),
        }
    return {
        "case_id": history.case_id,
        "family": history.family,
        "depth": history.depth,
        "last_finite_saved_frame": {
            "index": last,
            "time": float(history.times[last]),
        },
        "first_failed_gl2_step": failure_record,
        "hamiltonian": {
            "initial": float(history.hamiltonian[0]),
            "signed_relative_drift_at_last_finite": float(
                history.signed_hamiltonian_drift[-1]
            ),
            "maximum_absolute_relative_drift": float(
                np.max(hamiltonian_drift)
            ),
        },
        "surface_slope": {
            "initial_maximum_absolute": float(history.slope[0]),
            "last_finite_maximum_absolute": float(history.slope[-1]),
            "maximum_absolute": float(np.max(history.slope)),
            "last_finite_amplification": float(
                history.slope[-1] / history.slope[0]
            ),
            "maximum_amplification": float(
                np.max(history.slope) / history.slope[0]
            ),
        },
        "minimum_water_column_fraction": float(
            np.min(history.water_column_fraction)
        ),
        "last_finite_high_band_fraction": {
            "eta": float(history.eta_high_fraction[-1]),
            "xi": float(history.xi_high_fraction[-1]),
            "G6_eta_xi": float(history.gxi_high_fraction[-1]),
        },
        "maximum_generated_tail": {
            "eta": float(np.max(history.eta_generated_tail)),
            "xi": float(np.max(history.xi_generated_tail)),
            "G6_eta_xi": float(np.max(history.gxi_generated_tail)),
        },
        "G6_eta_xi_high_fraction_threshold_times": {
            f"{threshold:g}": first_threshold_time(
                history.times,
                history.gxi_high_fraction,
                threshold,
            )
            for threshold in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1)
        },
        "slope_amplification_threshold_times": {
            f"{threshold:g}": first_threshold_time(
                history.times,
                history.slope / history.slope[0],
                threshold,
            )
            for threshold in (1.5, 2.0, 3.0, 5.0, 8.0, 10.0)
        },
        "first_gl2_iteration_threshold_steps": {
            str(threshold): first_iteration_record(history, threshold)
            for threshold in (4, 5, 6, 7, 8)
        },
    }


def plot_panel(
    histories: dict[str, CaseHistory],
    *,
    successful_envelope: dict[str, float],
    residual_tolerance: float,
    output_png: Path,
    output_pdf: Path,
) -> None:
    """Write the compact two-case diagnostic panel."""

    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(14.0, 6.2),
        constrained_layout=True,
    )
    colors = {
        "tail": "#6F4E7C",
        "slope": "#0072B2",
        "hamiltonian": "#D55E00",
        "residual": "#009E73",
        "iterations": "#222222",
        "initial": "#666666",
        "last": "#6F4E7C",
        "failure": "#CC3311",
        "envelope": "#777777",
    }
    row_labels = {
        "tanaka_steep_upper_seam": "Tanaka upper-seam case",
        "bf_jcp09_canonical": "equation-(33) Benjamin–Feir stress case",
    }
    spectral_floor = 1.0e-16
    diagnostic_floor = 1.0e-14

    for row, case_id in enumerate(FAILED_CASE_IDS):
        history = histories[case_id]
        last_time = float(history.times[-1])
        failure_time = history.first_failed_step_time
        if failure_time is None:
            raise RuntimeError(f"{case_id} was expected to fail")

        tail_axis = axes[row, 0]
        tail_axis.semilogy(
            history.times,
            np.maximum(history.gxi_generated_tail, diagnostic_floor),
            color=colors["tail"],
            label=r"$\mathcal{G}_B^{G_6(\eta)\xi}(t)$",
        )
        tail_axis.axhline(
            successful_envelope["maximum_generated_tail_G6_eta_xi"],
            color=colors["envelope"],
            linestyle="--",
            linewidth=1.0,
            label="maximum over 5 successes",
        )
        tail_axis.axvline(
            last_time, color=colors["failure"], linestyle=":", linewidth=0.9
        )
        tail_axis.set_ylim(diagnostic_floor, 1.0)
        tail_axis.set_xlim(0.0, failure_time)
        tail_axis.set_xlabel("time")
        tail_axis.set_ylabel("generated high-band energy")
        tail_axis.text(
            0.03,
            0.96,
            (
                f"{row_labels[case_id]}\n"
                f"last finite {last_time:.2f}; failed step {failure_time:.2f}"
            ),
            transform=tail_axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "0.8",
                "alpha": 0.9,
            },
        )
        if row == 0:
            tail_axis.set_title(
                r"generated tail of $G_6(\eta)\xi$"
            )
            tail_axis.legend(loc="lower right", frameon=False)

        geometry_axis = axes[row, 1]
        slope_amplification = history.slope / history.slope[0]
        slope_line = geometry_axis.plot(
            history.times,
            slope_amplification,
            color=colors["slope"],
            label=r"$\|\eta_x(t)\|_\infty/\|\eta_x(0)\|_\infty$",
        )[0]
        geometry_axis.axhline(
            successful_envelope["maximum_slope_amplification"],
            color=colors["slope"],
            linestyle="--",
            linewidth=0.9,
            alpha=0.65,
        )
        geometry_axis.set_xlim(0.0, failure_time)
        geometry_axis.set_ylim(
            0.8, 1.08 * float(np.max(slope_amplification))
        )
        geometry_axis.set_xlabel("time")
        geometry_axis.set_ylabel("slope amplification", color=colors["slope"])
        geometry_axis.tick_params(axis="y", colors=colors["slope"])
        drift_axis = geometry_axis.twinx()
        drift = np.maximum(
            np.abs(history.signed_hamiltonian_drift), diagnostic_floor
        )
        drift_line = drift_axis.semilogy(
            history.times,
            drift,
            color=colors["hamiltonian"],
            label=r"$|H(t)-H(0)|/|H(0)|$",
        )[0]
        drift_axis.axhline(
            successful_envelope["maximum_absolute_hamiltonian_drift"],
            color=colors["hamiltonian"],
            linestyle="--",
            linewidth=0.9,
            alpha=0.65,
        )
        drift_axis.set_ylim(diagnostic_floor, 1.0)
        drift_axis.set_ylabel(
            "relative Hamiltonian drift", color=colors["hamiltonian"]
        )
        drift_axis.tick_params(axis="y", colors=colors["hamiltonian"])
        if row == 0:
            geometry_axis.set_title("surface steepening and energy drift")
            geometry_axis.legend(
                [slope_line, drift_line],
                [slope_line.get_label(), drift_line.get_label()],
                loc="upper left",
                frameon=False,
            )

        gl2_axis = axes[row, 2]
        final_window = (
            (history.step_times >= failure_time - 1.0)
            & (history.step_times <= failure_time)
        )
        relative_step_time = history.step_times[final_window] - failure_time
        relative_residual = (
            history.stage_residual[final_window] / residual_tolerance
        )
        finite_residual = np.isfinite(relative_residual)
        residual_line = gl2_axis.semilogy(
            relative_step_time[finite_residual],
            relative_residual[finite_residual],
            color=colors["residual"],
            label="stage residual / tolerance",
        )[0]
        gl2_axis.axhline(1.0, color="0.55", linestyle="--", linewidth=0.9)
        gl2_axis.axvline(0.0, color=colors["failure"], linewidth=0.9)
        failure_residual = history.stage_residual[history.first_failed_step]
        if math.isfinite(float(failure_residual)):
            gl2_axis.scatter(
                [0.0],
                [failure_residual / residual_tolerance],
                marker="x",
                color=colors["failure"],
                zorder=5,
            )
        else:
            gl2_axis.scatter(
                [0.0],
                [30.0],
                marker="x",
                color=colors["failure"],
                zorder=5,
            )
            gl2_axis.annotate(
                "nonfinite stage",
                xy=(0.0, 30.0),
                xytext=(-0.55, 40.0),
                fontsize=7.0,
                color=colors["failure"],
                arrowprops={
                    "arrowstyle": "->",
                    "color": colors["failure"],
                    "linewidth": 0.7,
                },
            )
        gl2_axis.set_xlim(-1.0, 0.02)
        gl2_axis.set_ylim(1.0e-3, 1.0e2)
        gl2_axis.set_xlabel("time relative to failed step")
        gl2_axis.set_ylabel(
            "relative GL2 residual", color=colors["residual"]
        )
        gl2_axis.tick_params(axis="y", colors=colors["residual"])
        iteration_axis = gl2_axis.twinx()
        iteration_line = iteration_axis.step(
            relative_step_time,
            history.iterations[final_window],
            where="post",
            color=colors["iterations"],
            alpha=0.7,
            linewidth=0.9,
            label="fixed-point iterations",
        )[0]
        iteration_axis.set_ylim(0.0, 8.8)
        iteration_axis.set_ylabel("iterations")
        if row == 0:
            gl2_axis.set_title("GL2 convergence in final unit")
            gl2_axis.legend(
                [residual_line, iteration_line],
                [residual_line.get_label(), iteration_line.get_label()],
                loc="lower left",
                frameon=False,
            )

        spectrum_axis = axes[row, 3]
        nx = history.gxi.shape[-1]
        initial_coefficients = np.fft.rfft(history.gxi[0]) / nx
        last_coefficients = np.fft.rfft(history.gxi[-1]) / nx
        initial_delivered_energy = np.sum(
            np.abs(
                initial_coefficients[
                    DELIVERED_BAND[0] : DELIVERED_BAND[1] + 1
                ]
            )
            ** 2
        )
        modes = np.arange(1, DELIVERED_BAND[1] + 1)
        initial_modal_energy = (
            np.abs(initial_coefficients[modes]) ** 2
            / initial_delivered_energy
        )
        last_modal_energy = (
            np.abs(last_coefficients[modes]) ** 2
            / initial_delivered_energy
        )
        spectrum_axis.semilogy(
            modes,
            np.maximum(initial_modal_energy, spectral_floor),
            color=colors["initial"],
            linestyle="--",
            label="initial",
        )
        spectrum_axis.semilogy(
            modes,
            np.maximum(last_modal_energy, spectral_floor),
            color=colors["last"],
            label="last finite",
        )
        spectrum_axis.axvspan(
            HIGH_BAND[0],
            HIGH_BAND[1],
            color="#D9D9D9",
            alpha=0.45,
            zorder=0,
            label=r"$B=\{80,\ldots,128\}$",
        )
        spectrum_axis.set_xlim(1, DELIVERED_BAND[1])
        spectrum_axis.set_ylim(spectral_floor, 1.0)
        spectrum_axis.set_xlabel("Fourier mode $k$")
        spectrum_axis.set_ylabel(
            r"$|\widehat{G_6(\eta)\xi}_k|^2/E_{1:128}(0)$"
        )
        if row == 0:
            spectrum_axis.set_title(
                r"initial and last-finite $G_6(\eta)\xi$ spectra"
            )
            spectrum_axis.legend(loc="lower left", frameon=False)

        for axis in axes[row]:
            axis.grid(alpha=0.18, linewidth=0.5)

    figure.savefig(output_png, dpi=300)
    figure.savefig(output_pdf)
    plt.close(figure)


def main() -> None:
    """Load the coarse arm, write the figure, and record its exact metrics."""

    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / "coarse_failure_spectral_audit.png"
    output_pdf = output_dir / "coarse_failure_spectral_audit.pdf"
    output_json = output_dir / "coarse_failure_spectral_audit_metrics.json"

    with np.load(input_path, allow_pickle=False) as archive:
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        case_ids = np.asarray(archive["case_id"]).tolist()
        families = np.asarray(archive["family"]).tolist()
        depths = np.asarray(archive["depth"], dtype=np.float64)
        times = np.asarray(archive["times"], dtype=np.float64)
        eta = np.asarray(archive["eta"], dtype=np.float64)
        xi = np.asarray(archive["xi"], dtype=np.float64)
        gxi = np.asarray(archive["gxi"], dtype=np.float64)
        step_times = np.asarray(
            archive["gl2_step_times"], dtype=np.float64
        )
        stage_residual = np.asarray(
            archive["gl2_stage_residual"], dtype=np.float64
        )
        iterations = np.asarray(
            archive["gl2_iterations"], dtype=np.int32
        )
        converged = np.asarray(archive["gl2_converged"], dtype=np.bool_)
        stage_finite = np.asarray(
            archive["gl2_stage_finite"], dtype=np.bool_
        )
        state_finite = np.asarray(
            archive["gl2_state_finite"], dtype=np.bool_
        )
        hit_iteration_cap = np.asarray(
            archive["gl2_hit_iteration_cap"], dtype=np.bool_
        )
        residual_tolerance = float(
            np.asarray(archive["gl2_residual_tolerance"]).item()
        )

    histories = {
        case_id: build_history(
            case_index=index,
            case_id=case_id,
            family=str(families[index]),
            depth=float(depths[index]),
            times=times,
            eta=eta,
            xi=xi,
            gxi=gxi,
            step_times=step_times,
            stage_residual=stage_residual,
            iterations=iterations,
            converged=converged,
            stage_finite=stage_finite,
            state_finite=state_finite,
            hit_iteration_cap=hit_iteration_cap,
            gravity=float(metadata["contract"]["gravity"]),
        )
        for index, case_id in enumerate(case_ids)
    }
    successful_ids = [
        case_id
        for case_id in case_ids
        if histories[case_id].first_failed_step < 0
    ]
    if len(successful_ids) != 5:
        raise RuntimeError(
            f"expected five successful controls, found {successful_ids}"
        )
    successful_envelope = {
        "maximum_absolute_hamiltonian_drift": max(
            float(
                np.max(
                    np.abs(histories[case_id].signed_hamiltonian_drift)
                )
            )
            for case_id in successful_ids
        ),
        "maximum_slope_amplification": max(
            float(
                np.max(histories[case_id].slope)
                / histories[case_id].slope[0]
            )
            for case_id in successful_ids
        ),
        "minimum_water_column_fraction": min(
            float(np.min(histories[case_id].water_column_fraction))
            for case_id in successful_ids
        ),
        "maximum_generated_tail_eta": max(
            float(np.max(histories[case_id].eta_generated_tail))
            for case_id in successful_ids
        ),
        "maximum_generated_tail_xi": max(
            float(np.max(histories[case_id].xi_generated_tail))
            for case_id in successful_ids
        ),
        "maximum_generated_tail_G6_eta_xi": max(
            float(np.max(histories[case_id].gxi_generated_tail))
            for case_id in successful_ids
        ),
    }
    plot_panel(
        histories,
        successful_envelope=successful_envelope,
        residual_tolerance=residual_tolerance,
        output_png=output_png,
        output_pdf=output_pdf,
    )

    payload = {
        "schema": "paper_coarse_failure_spectral_audit_v1",
        "created_local": datetime.now().astimezone().isoformat(),
        "source": {
            "artifact": str(input_path.relative_to(ROOT)),
            "sha256": sha256(input_path),
            "saved_frame_spacing": float(times[1] - times[0]),
            "gl2_step_spacing": float(step_times[1] - step_times[0]),
            "number_of_saved_frames": int(times.size),
            "number_of_gl2_steps": int(step_times.size),
        },
        "definitions": {
            "fourier_coefficients": (
                "numpy rfft coefficients divided by N=1024"
            ),
            "delivered_modes": {
                "symbol": "D",
                "inclusive_integer_modes": list(DELIVERED_BAND),
            },
            "high_band": {
                "symbol": "B",
                "inclusive_integer_modes": list(HIGH_BAND),
            },
            "band_energy": (
                "E_I(f,t) = sum_{k in I} |f_hat_k(t)|^2"
            ),
            "high_band_fraction": (
                "F_B(f,t) = E_B(f,t) / E_D(f,t)"
            ),
            "generated_tail": (
                "G_B(f,t) = max(E_B(f,t)-E_B(f,0),0) / E_D(f,0)"
            ),
            "surface_slope_amplification": (
                "max_x |eta_x(x,t)| / max_x |eta_x(x,0)|"
            ),
            "hamiltonian": (
                "H(t) = (dx/2) sum_j [xi_j(t) G_6(eta(t))xi_j(t)"
                " + eta_j(t)^2], with gravity g=1"
            ),
            "relative_hamiltonian_drift": (
                "|H(t)-H(0)| / |H(0)|"
            ),
            "minimum_water_column_fraction": (
                "min_{t,x} (h+eta(x,t))/h over finite saved frames"
            ),
            "gl2_relative_residual": (
                "stored implicit-stage residual divided by the declared"
                " tolerance 1e-8"
            ),
            "G6_eta_xi": (
                "the stored reference DNO-series target G_6(eta)xi,"
                " called gxi in the source NPZ"
            ),
        },
        "successful_control_envelope": {
            "case_ids": successful_ids,
            "scope": (
                "extrema over all finite frames through T=200 for all five"
                " complete coarse trajectories"
            ),
            **successful_envelope,
        },
        "failed_cases": {
            case_id: case_record(histories[case_id])
            for case_id in FAILED_CASE_IDS
        },
        "interpretation": (
            "In both cases the generated high-frequency tail and surface"
            " slope grow for many saved frames before GL2 fails. The water"
            " column stays comparable to successful controls. GL2"
            " nonconvergence is therefore the terminal symptom of a prior"
            " resolved spectral cascade, not an isolated first event."
        ),
        "limitations": (
            "The dt=0.01 arm alone cannot distinguish temporal truncation"
            " error from breakdown of the M=6, |k|<=128 spatial model."
            " The paired dt=0.005 arm tests the temporal explanation;"
            " spatial cutoff/order refinement is needed if the onset persists."
        ),
        "figures": {
            "png": {
                "artifact": str(output_png.relative_to(ROOT)),
                "sha256": sha256(output_png),
            },
            "pdf": {
                "artifact": str(output_pdf.relative_to(ROOT)),
                "sha256": sha256(output_pdf),
            },
        },
    }
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(output_png)
    print(output_pdf)
    print(output_json)


if __name__ == "__main__":
    main()
