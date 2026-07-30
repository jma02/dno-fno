"""Render constructor and discarded-filter stress cases for the paper corpus."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray
from scipy.optimize import brentq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from solver.data.stokes_truth_jax import (  # noqa: E402
    _finite_depth_coeffs,
    finite_depth_eta_harmonics,
    stokes_eta_xi,
)
from solver.solvers.dno_series_jax import (  # noqa: E402
    build_grid,
    dno_series_eval,
)


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
STOKES_PANEL = (
    ROOT / "data/manuscript_ic_panels_v1_20260715/stokes_finite_ics.npz"
)
LINEAR_PANEL = ROOT / "data/manuscript_ic_panels_v1_20260715/linear_ics.npz"
TANAKA_TRAJECTORIES = (
    ROOT
    / "outputs/tanaka_case31_tangent_resampling_trial_20260723"
    / "paired_trajectories.npz"
)
TANAKA_SUMMARY = (
    ROOT
    / "outputs/tanaka_case31_tangent_resampling_trial_20260723"
    / "summary.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs/paper_corpus_shape_stress_20260724"

LENGTH = 2.0 * math.pi
GRAVITY = 1.0
DELIVERED_MODE = 128
TANAKA_TARGET_TIME = 76.32
BOUNDARY_DEPTH_WAVENUMBERS = (0.5, 0.55, 0.6, 0.625)
BOUNDARY_RATIOS = (1.0, 0.75)
KINEMATIC_NX = 1024
KINEMATIC_DNO_ORDER = 6
KINEMATIC_PAD_FACTOR = 8

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


@dataclass(frozen=True)
class PanelData:
    """Initial-condition panel arrays and decoded metadata."""

    eta: FloatArray
    xi: FloatArray
    depth: FloatArray
    case_ids: IntArray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class StokesExtreme:
    """One selected finite-depth Stokes state."""

    index: int
    case_id: int
    eta: FloatArray
    depth: float
    mode: int
    amplitude: float
    depth_wavenumber: float
    steepness: float
    harmonics: FloatArray
    ordering_ratio: float


@dataclass(frozen=True)
class BoundaryExample:
    """One dimensionless finite-Stokes ordering-boundary example."""

    depth_wavenumber: float
    steepness: float
    amplitude_over_depth: float
    target_ratio: float
    ordering_ratio: float
    harmonics_over_depth: FloatArray
    crest_over_depth: float
    trough_over_depth: float
    maximum_absolute_slope: float
    kinematic_defect: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def _decode_metadata(archive: NpzFile) -> dict[str, Any]:
    """Decode the scalar JSON metadata stored in an IC panel."""

    encoded = archive["meta.json"].item()
    text = encoded.decode("utf-8") if isinstance(encoded, bytes) else str(encoded)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("panel metadata must decode to an object")
    return value


def load_panel(path: Path) -> PanelData:
    """Load one immutable manuscript IC panel."""

    with np.load(path, allow_pickle=False) as archive:
        return PanelData(
            eta=np.asarray(archive["eta"], dtype=np.float64),
            xi=np.asarray(archive["xi"], dtype=np.float64),
            depth=np.asarray(archive["depth"], dtype=np.float64),
            case_ids=np.asarray(archive["case_ids"], dtype=np.int64),
            metadata=_decode_metadata(archive),
        )


def ordering_ratio(harmonics: FloatArray) -> float:
    """Return the aggregate correction-to-fundamental elevation ratio."""

    return float(np.sum(np.abs(harmonics[1:])) / abs(harmonics[0]))


def stokes_extremes(panel: PanelData) -> tuple[StokesExtreme, StokesExtreme]:
    """Find the most unordered and closest retained finite-Stokes cases."""

    specifications = panel.metadata["case_specs"]
    length = float(panel.metadata["length"])
    harmonics: list[FloatArray] = []
    ratios: list[float] = []
    for specification in specifications:
        mode = int(specification["n0"])
        wavenumber = 2.0 * math.pi * mode / length
        values = np.asarray(
            finite_depth_eta_harmonics(
                k0=wavenumber,
                depth=float(specification["depth"]),
                gravity=GRAVITY,
                a0=float(specification["a0"]),
            ),
            dtype=np.float64,
        )
        harmonics.append(values)
        ratios.append(ordering_ratio(values))

    ratio_array = np.asarray(ratios, dtype=np.float64)
    rejected = np.flatnonzero(ratio_array > 1.0)
    retained = np.flatnonzero(ratio_array <= 1.0)
    if rejected.size == 0 or retained.size == 0:
        raise ValueError("finite-Stokes panel must contain both support classes")

    rejected_index = int(rejected[np.argmax(ratio_array[rejected])])
    retained_index = int(retained[np.argmax(ratio_array[retained])])

    def build(index: int) -> StokesExtreme:
        specification = specifications[index]
        mode = int(specification["n0"])
        amplitude = float(specification["a0"])
        depth = float(specification["depth"])
        wavenumber = 2.0 * math.pi * mode / length
        return StokesExtreme(
            index=index,
            case_id=int(panel.case_ids[index]),
            eta=panel.eta[index],
            depth=depth,
            mode=mode,
            amplitude=amplitude,
            depth_wavenumber=wavenumber * depth,
            steepness=wavenumber * amplitude,
            harmonics=harmonics[index],
            ordering_ratio=float(ratio_array[index]),
        )

    return build(rejected_index), build(retained_index)


def one_sided_amplitude(field: FloatArray, scale: float) -> FloatArray:
    """Return cosine-equivalent one-sided Fourier amplitudes."""

    coefficients = np.fft.rfft(field) / field.size
    amplitudes = np.abs(coefficients)
    if amplitudes.size > 2:
        amplitudes[1:-1] *= 2.0
    return np.asarray(amplitudes / scale, dtype=np.float64)


def coefficient_band_norm(
    field: FloatArray,
    lower_mode: int,
    upper_mode: int,
) -> float:
    """Return the unscaled complex-coefficient norm over a Fourier band."""

    coefficients = np.fft.rfft(field) / field.size
    return float(np.linalg.norm(coefficients[lower_mode : upper_mode + 1]))


def stokes_phase_profile(case: StokesExtreme) -> tuple[FloatArray, FloatArray]:
    """Evaluate one phase-aligned carrier period from stored harmonics."""

    phase = np.linspace(-math.pi, math.pi, 2049, dtype=np.float64)
    modes = np.arange(1, 6, dtype=np.float64)
    eta_over_depth = np.sum(
        (case.harmonics / case.depth)[:, None]
        * np.cos(modes[:, None] * phase[None, :]),
        axis=0,
    )
    return phase / (2.0 * math.pi), eta_over_depth


def load_tanaka_pair() -> dict[str, Any]:
    """Load old and corrected case-31 fields at the archived tail event."""

    with np.load(TANAKA_TRAJECTORIES, allow_pickle=False) as archive:
        times = np.asarray(archive["times"], dtype=np.float64)
        frame = int(np.argmin(np.abs(times - TANAKA_TARGET_TIME)))
        eta = np.asarray(archive["eta"][frame], dtype=np.float64)
        sign_counts = np.asarray(archive["sign_counts"][frame], dtype=np.int64)
        time = float(times[frame])

    summary = json.loads(TANAKA_SUMMARY.read_text(encoding="utf-8"))
    depth = float(summary["depth_f64"])
    old_high = coefficient_band_norm(eta[0], 80, 128)
    corrected_high = coefficient_band_norm(eta[1], 80, 128)
    return {
        "case_id": int(summary["case_id"]),
        "time": time,
        "depth": depth,
        "eta": eta,
        "sign_counts": sign_counts,
        "old_high_band_norm": old_high,
        "corrected_high_band_norm": corrected_high,
        "high_band_reduction": old_high / corrected_high,
    }


def select_linear_margin_case(panel: PanelData) -> dict[str, Any]:
    """Select the smooth Airy state furthest below the discarded half-depth margin."""

    remaining_fraction = np.min(
        panel.depth[:, None] + panel.eta,
        axis=1,
    ) / panel.depth
    index = int(np.argmin(remaining_fraction))
    specification = panel.metadata["case_specs"][index]
    return {
        "index": index,
        "case_id": int(panel.case_ids[index]),
        "eta": panel.eta[index],
        "depth": float(panel.depth[index]),
        "mode": int(specification["n0"]),
        "amplitude": float(specification["a0"]),
        "remaining_fraction": float(remaining_fraction[index]),
    }


def _spectrum_axis(
    axis: plt.Axes,
    field: FloatArray,
    depth: float,
    *,
    color: str,
    label: str | None = None,
    linestyle: str = "-",
) -> None:
    """Plot the delivered-band elevation spectrum."""

    amplitudes = one_sided_amplitude(field, depth)
    modes = np.arange(DELIVERED_MODE + 1)
    floor = 1.0e-13
    axis.semilogy(
        modes,
        np.maximum(amplitudes[: DELIVERED_MODE + 1], floor),
        color=color,
        linewidth=1.2,
        linestyle=linestyle,
        label=label,
    )


def _annotate(axis: plt.Axes, text: str) -> None:
    """Add a compact white annotation box."""

    axis.text(
        0.02,
        0.96,
        text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={
            "facecolor": "white",
            "edgecolor": "0.8",
            "alpha": 0.92,
            "pad": 3.0,
        },
    )


def plot_shape_stress(
    output_dir: Path,
    rejected: StokesExtreme,
    retained: StokesExtreme,
    tanaka: dict[str, Any],
    linear: dict[str, Any],
) -> tuple[Path, Path]:
    """Render the requested four-row profile and spectrum comparison."""

    red = "#b33b2e"
    blue = "#2f6da3"
    orange = "#d87818"
    purple = "#7651a8"

    figure, axes = plt.subplots(
        4,
        2,
        figsize=(11.0, 11.5),
        constrained_layout=True,
    )
    figure.suptitle(
        "Paper-corpus shape stress cases",
        fontsize=16,
    )
    axes[0, 0].set_title("Dimensionless surface profile", fontsize=12)
    axes[0, 1].set_title(
        rf"Elevation spectrum in delivered band $0\leq m\leq{DELIVERED_MODE}$",
        fontsize=12,
    )

    for row, case, color, status in (
        (0, rejected, red, "outside finite-Stokes support"),
        (1, retained, blue, "inside finite-Stokes support"),
    ):
        phase, profile = stokes_phase_profile(case)
        axes[row, 0].plot(phase, profile, color=color, linewidth=1.8)
        _spectrum_axis(
            axes[row, 1],
            case.eta,
            case.depth,
            color=color,
        )
        axes[row, 0].set_xlabel(r"carrier phase $\theta/(2\pi)$")
        _annotate(
            axes[row, 0],
            f"Finite Stokes, case {case.case_id}\n"
            rf"$kh={case.depth_wavenumber:.3f}$, "
            rf"$ka={case.steepness:.3f}$, $R={case.ordering_ratio:.3f}$"
            f"\n{status}",
        )

    horizontal_coordinate = np.arange(tanaka["eta"].shape[-1]) / tanaka[
        "eta"
    ].shape[-1]
    axes[2, 0].plot(
        horizontal_coordinate,
        tanaka["eta"][0] / tanaka["depth"],
        color=orange,
        linewidth=1.4,
        label="piecewise-linear placement",
    )
    axes[2, 0].plot(
        horizontal_coordinate,
        tanaka["eta"][1] / tanaka["depth"],
        color=blue,
        linewidth=1.2,
        linestyle="--",
        label="tangent-Hermite placement",
    )
    _spectrum_axis(
        axes[2, 1],
        tanaka["eta"][0],
        tanaka["depth"],
        color=orange,
        label="piecewise linear",
    )
    _spectrum_axis(
        axes[2, 1],
        tanaka["eta"][1],
        tanaka["depth"],
        color=blue,
        label="tangent Hermite",
        linestyle="--",
    )
    axes[2, 0].legend(loc="lower left", fontsize=8)
    axes[2, 1].legend(loc="lower left", fontsize=8)
    axes[2, 0].set_xlabel(r"$x/L$")
    _annotate(
        axes[2, 0],
        f"Tanaka case {tanaka['case_id']} at t={tanaka['time']:.2f}\n"
        f"sign transitions: {int(tanaka['sign_counts'][0])}"
        f" → {int(tanaka['sign_counts'][1])}\n"
        rf"$\|\widehat{{\eta}}_{{80:128}}\|_2$ reduction "
        f"{tanaka['high_band_reduction']:.1f}×",
    )

    linear_phase = np.linspace(-0.5, 0.5, 2049, dtype=np.float64)
    linear_profile = (
        linear["amplitude"]
        / linear["depth"]
        * np.cos(2.0 * math.pi * linear_phase)
    )
    axes[3, 0].plot(
        linear_phase,
        linear_profile,
        color=purple,
        linewidth=1.8,
    )
    _spectrum_axis(
        axes[3, 1],
        linear["eta"],
        linear["depth"],
        color=purple,
    )
    axes[3, 0].set_xlabel(r"carrier phase $\theta/(2\pi)$")
    _annotate(
        axes[3, 0],
        f"Airy case {linear['case_id']}, mode {linear['mode']}\n"
        rf"$\min_x(h+\eta)/h={linear['remaining_fraction']:.3f}<1/2$"
        "\nprofile remains a single smooth sinusoid",
    )

    row_labels = (
        "unordered Stokes",
        "near-boundary retained Stokes",
        "Tanaka reconstruction",
        "discarded margin example",
    )
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(r"$\eta/h$" + f"\n{label}")
        axes[row, 1].set_ylabel(r"one-sided $|\widehat{\eta}_m|/h$")
        axes[row, 1].set_xlabel("Fourier mode m")
        axes[row, 1].set_xlim(0, DELIVERED_MODE)
        axes[row, 1].set_ylim(1.0e-12, 1.0)
        for axis in axes[row]:
            axis.grid(alpha=0.22, linewidth=0.5)

    png_path = output_dir / "paper_corpus_shape_stress.png"
    pdf_path = output_dir / "paper_corpus_shape_stress.pdf"
    figure.savefig(png_path, dpi=200)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def dimensionless_ordering_ratio(
    depth_wavenumber: float,
    steepness: float,
) -> float:
    """Evaluate the ordering ratio with k=1 in dimensionless variables."""

    harmonics = np.asarray(
        finite_depth_eta_harmonics(
            k0=1.0,
            depth=depth_wavenumber,
            gravity=GRAVITY,
            a0=steepness,
        ),
        dtype=np.float64,
    )
    return ordering_ratio(harmonics)


def solve_boundary_steepness(
    depth_wavenumber: float,
    target_ratio: float,
) -> float:
    """Solve R(kh, ka)=target_ratio on the shallow support boundary."""

    return float(
        brentq(
            lambda steepness: (
                dimensionless_ordering_ratio(
                    depth_wavenumber,
                    steepness,
                )
                - target_ratio
            ),
            1.0e-4,
            0.15,
            xtol=5.0e-15,
        )
    )


def boundary_example(
    depth_wavenumber: float,
    target_ratio: float,
    x_grid: jax.Array,
    wavenumbers: jax.Array,
) -> BoundaryExample:
    """Construct one boundary state and evaluate its kinematic defect."""

    steepness = solve_boundary_steepness(depth_wavenumber, target_ratio)
    harmonics = np.asarray(
        finite_depth_eta_harmonics(
            k0=1.0,
            depth=depth_wavenumber,
            gravity=GRAVITY,
            a0=steepness,
        ),
        dtype=np.float64,
    )
    eta, xi = stokes_eta_xi(
        x=x_grid,
        time=jnp.asarray(0.0, dtype=jnp.float64),
        n0=1,
        a0=steepness,
        length=LENGTH,
        depth=depth_wavenumber,
        gravity=GRAVITY,
        ichoi=1,
    )
    coefficients = _finite_depth_coeffs(
        1.0,
        depth_wavenumber,
        GRAVITY,
        steepness,
    )
    modes = jnp.arange(1, 6, dtype=jnp.float64)
    eta_time_derivative = jnp.sum(
        (
            modes
            * coefficients["om"]
            * jnp.asarray(harmonics, dtype=jnp.float64)
        )[:, None]
        * jnp.sin(modes[:, None] * x_grid[None, :]),
        axis=0,
    )
    gxi = dno_series_eval(
        eta[None, :],
        xi[None, :],
        wavenumbers,
        jnp.asarray([[depth_wavenumber]], dtype=jnp.float64),
        KINEMATIC_DNO_ORDER,
        pad_factor=KINEMATIC_PAD_FACTOR,
    )[0]
    defect = float(
        jnp.linalg.norm(eta_time_derivative - gxi)
        / jnp.linalg.norm(eta_time_derivative)
    )

    phase = np.linspace(-math.pi, math.pi, 32769, dtype=np.float64)
    integer_modes = np.arange(1, 6, dtype=np.float64)
    harmonics_over_depth = harmonics / depth_wavenumber
    profile = np.sum(
        harmonics_over_depth[:, None]
        * np.cos(integer_modes[:, None] * phase[None, :]),
        axis=0,
    )
    slope = -depth_wavenumber * np.sum(
        (integer_modes * harmonics_over_depth)[:, None]
        * np.sin(integer_modes[:, None] * phase[None, :]),
        axis=0,
    )
    return BoundaryExample(
        depth_wavenumber=depth_wavenumber,
        steepness=steepness,
        amplitude_over_depth=steepness / depth_wavenumber,
        target_ratio=target_ratio,
        ordering_ratio=ordering_ratio(harmonics),
        harmonics_over_depth=harmonics_over_depth,
        crest_over_depth=float(np.max(profile)),
        trough_over_depth=float(np.min(profile)),
        maximum_absolute_slope=float(np.max(np.abs(slope))),
        kinematic_defect=defect,
    )


def build_boundary_examples() -> tuple[BoundaryExample, ...]:
    """Build the predeclared eight-state ordering-boundary panel."""

    x_grid, wavenumbers = build_grid(KINEMATIC_NX, LENGTH)
    return tuple(
        boundary_example(
            depth_wavenumber,
            target_ratio,
            x_grid,
            wavenumbers,
        )
        for target_ratio in BOUNDARY_RATIOS
        for depth_wavenumber in BOUNDARY_DEPTH_WAVENUMBERS
    )


def plot_boundary_examples(
    output_dir: Path,
    examples: tuple[BoundaryExample, ...],
) -> tuple[Path, Path]:
    """Plot the two finite-Stokes coefficient-ratio boundaries."""

    phase = np.linspace(-0.5, 0.5, 2049, dtype=np.float64)
    integer_modes = np.arange(1, 6, dtype=np.float64)
    figure, axes = plt.subplots(
        len(BOUNDARY_RATIOS),
        len(BOUNDARY_DEPTH_WAVENUMBERS),
        figsize=(12.0, 5.6),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    colors = {1.0: "#b33b2e", 0.75: "#2f6da3"}
    for example, axis in zip(examples, axes.flat):
        profile = np.sum(
            example.harmonics_over_depth[:, None]
            * np.cos(
                2.0
                * math.pi
                * integer_modes[:, None]
                * phase[None, :]
            ),
            axis=0,
        )
        axis.plot(
            phase,
            profile,
            color=colors[example.target_ratio],
            linewidth=1.7,
        )
        axis.set_title(rf"$kh={example.depth_wavenumber:g}$", fontsize=11)
        axis.text(
            0.03,
            0.95,
            rf"$ka={example.steepness:.5f}$"
            "\n"
            rf"$\max\eta/h={example.crest_over_depth:.3f}$"
            "\n"
            rf"$D_{{\rm kin}}={example.kinematic_defect:.3f}$",
            transform=axis.transAxes,
            va="top",
            fontsize=8.5,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
        )
        axis.grid(alpha=0.22, linewidth=0.5)
        axis.set_xlabel(r"carrier phase $\theta/(2\pi)$")

    axes[0, 0].set_ylabel(
        r"$\eta/h$"
        "\n"
        r"$\sum_{m=2}^5|E_m|/|E_1|=1$"
    )
    axes[1, 0].set_ylabel(
        r"$\eta/h$"
        "\n"
        r"$\sum_{m=2}^5|E_m|/|E_1|=0.75$"
    )
    figure.suptitle(
        "Finite-depth fifth-order Stokes ordering boundaries",
        fontsize=15,
    )
    png_path = output_dir / "finite_stokes_boundary_panel.png"
    pdf_path = output_dir / "finite_stokes_boundary_panel.pdf"
    figure.savefig(png_path, dpi=200)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def _stokes_record(case: StokesExtreme) -> dict[str, Any]:
    """Serialize a selected Stokes case."""

    return {
        "panel_index": case.index,
        "case_id": case.case_id,
        "mode": case.mode,
        "depth": case.depth,
        "amplitude": case.amplitude,
        "kh": case.depth_wavenumber,
        "ka": case.steepness,
        "elevation_harmonics": case.harmonics.tolist(),
        "elevation_harmonics_over_depth": (
            case.harmonics / case.depth
        ).tolist(),
        "higher_to_fundamental_ratio": case.ordering_ratio,
    }


def _boundary_record(example: BoundaryExample) -> dict[str, Any]:
    """Serialize one ordering-boundary example."""

    return {
        "kh": example.depth_wavenumber,
        "ka": example.steepness,
        "a_over_h": example.amplitude_over_depth,
        "target_higher_to_fundamental_ratio": example.target_ratio,
        "higher_to_fundamental_ratio": example.ordering_ratio,
        "elevation_harmonics_over_depth": (
            example.harmonics_over_depth.tolist()
        ),
        "crest_over_depth": example.crest_over_depth,
        "trough_over_depth": example.trough_over_depth,
        "maximum_absolute_surface_slope": example.maximum_absolute_slope,
        "kinematic_defect": example.kinematic_defect,
    }


def write_summary(
    output_dir: Path,
    rejected: StokesExtreme,
    retained: StokesExtreme,
    tanaka: dict[str, Any],
    linear: dict[str, Any],
    boundary_examples: tuple[BoundaryExample, ...],
) -> Path:
    """Write exact figure inputs, metrics, and numerical definitions."""

    summary = {
        "definitions": {
            "stokes_ordering_ratio": (
                "sum(abs(E_2),...,abs(E_5)) / abs(E_1)"
            ),
            "delivered_spectrum": (
                "one-sided FFT elevation amplitudes divided by depth; "
                "modes 0 through 128"
            ),
            "kinematic_defect": (
                "||eta_t - G_6(eta;h)xi||_2 / ||eta_t||_2"
            ),
            "kinematic_eta_t": (
                "exact time derivative of the implemented fifth-order "
                "traveling-wave elevation"
            ),
            "kinematic_discretization": {
                "nx": KINEMATIC_NX,
                "length": LENGTH,
                "dtype": "float64",
                "dno_order": KINEMATIC_DNO_ORDER,
                "pad_factor": KINEMATIC_PAD_FACTOR,
                "phase": 0.0,
            },
        },
        "sources": {
            "stokes_panel": str(STOKES_PANEL.relative_to(ROOT)),
            "linear_panel": str(LINEAR_PANEL.relative_to(ROOT)),
            "tanaka_trajectories": str(TANAKA_TRAJECTORIES.relative_to(ROOT)),
            "tanaka_summary": str(TANAKA_SUMMARY.relative_to(ROOT)),
        },
        "cases": {
            "worst_excluded_stokes": _stokes_record(rejected),
            "nearest_retained_stokes": _stokes_record(retained),
            "tanaka_case31": {
                "case_id": tanaka["case_id"],
                "time": tanaka["time"],
                "depth": tanaka["depth"],
                "piecewise_linear_sign_transitions": int(
                    tanaka["sign_counts"][0]
                ),
                "tangent_hermite_sign_transitions": int(
                    tanaka["sign_counts"][1]
                ),
                "piecewise_linear_eta_modes_80_128_norm": tanaka[
                    "old_high_band_norm"
                ],
                "tangent_hermite_eta_modes_80_128_norm": tanaka[
                    "corrected_high_band_norm"
                ],
                "eta_modes_80_128_reduction": tanaka[
                    "high_band_reduction"
                ],
            },
            "linear_below_discarded_margin": {
                "panel_index": linear["index"],
                "case_id": linear["case_id"],
                "mode": linear["mode"],
                "depth": linear["depth"],
                "amplitude": linear["amplitude"],
                "minimum_water_column_over_depth": linear[
                    "remaining_fraction"
                ],
            },
        },
        "finite_stokes_boundary_examples": [
            _boundary_record(example) for example in boundary_examples
        ],
    }
    path = output_dir / "shape_stress_metrics.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    """Build both figures and their numerical record."""

    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stokes_panel = load_panel(STOKES_PANEL)
    rejected, retained = stokes_extremes(stokes_panel)
    tanaka = load_tanaka_pair()
    linear = select_linear_margin_case(load_panel(LINEAR_PANEL))
    boundary_examples = build_boundary_examples()

    shape_paths = plot_shape_stress(
        output_dir,
        rejected,
        retained,
        tanaka,
        linear,
    )
    boundary_paths = plot_boundary_examples(output_dir, boundary_examples)
    summary_path = write_summary(
        output_dir,
        rejected,
        retained,
        tanaka,
        linear,
        boundary_examples,
    )
    for path in (*shape_paths, *boundary_paths, summary_path):
        print(path)


if __name__ == "__main__":
    main()
