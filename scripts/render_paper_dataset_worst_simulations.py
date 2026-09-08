"""Rank and render diagnostic tails in the accepted paper-dataset rows.

The rankings are descriptive and do not alter dataset acceptance. Each simulation
is scanned at every retained time, then ranked within its initial-condition
family by surface slope, high-band content of ``G(eta)xi``, and an
amplitude-thresholded count of spatial oscillations in ``G(eta)xi``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Final, NamedTuple

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402


FIELD_NAMES = ("eta", "xi", "gxi")
FIELD_TITLES = (r"$\eta(x)$", r"$\xi(x)$", r"$G(\eta)\xi(x)$")
FAMILY_LABELS = {
    "stokes": "Finite-depth Stokes",
    "tanaka": "Tanaka",
    "benjamin_feir": "Benjamin--Feir",
    "jonswap_tma": "JONSWAP/TMA",
}
DELIVERED_MODE = 128
HIGH_BAND_START = 96
SIGN_DIFFERENCE_RELATIVE_THRESHOLD = 0.03
GIF_MAXIMUM_FRAMES = 100
GIF_SHORT_FRAME_THRESHOLD = 20
GIF_SHORT_FPS = 4
GIF_LONG_FPS = 12
GIF_DIMENSIONS = (1_250, 360)
GIF_Y_LIMIT_PADDING_FRACTION = 0.06
QUANTILES = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
FINAL_PAPER_DATASET_SOURCE_COUNT = 12
FINAL_PAPER_DATASET_SPLIT_ACCEPTED_SIMULATIONS = {
    "train": 16_384,
    "validation": 1_024,
    "test": 1_024,
}
FINAL_PAPER_DATASET_ACCEPTED_SIMULATIONS = 73_728
FINAL_PAPER_DATASET_RETAINED_ROWS = 7_686_144
DIAGNOSTIC_SCHEMA = "paper_dataset_all_family_diagnostic_tail_v3"
INTERPRETATION: Final = (
    "Descriptive full-trajectory ranking only. Every plotted simulation was accepted "
    "by the frozen dataset construction and trajectory checks. Morphology and "
    "stored-band quadratic-energy diagnostics do not alter dataset acceptance "
    "or constitute release thresholds."
)
PARAMETERS: Final = {
    "final_paper_dataset_contract_required": True,
    "delivered_maximum_wavenumber": DELIVERED_MODE,
    "high_band_minimum_wavenumber": HIGH_BAND_START,
    "cyclic_difference_relative_dead_zone": SIGN_DIFFERENCE_RELATIVE_THRESHOLD,
    "morphology_diagnostics_are_release_thresholds": False,
    "rank_one_fixed_axis_gifs": True,
    "gif_maximum_frames": GIF_MAXIMUM_FRAMES,
    "gif_short_frame_threshold": GIF_SHORT_FRAME_THRESHOLD,
    "gif_short_fps": GIF_SHORT_FPS,
    "gif_long_fps": GIF_LONG_FPS,
    "gif_dimensions_pixels": list(GIF_DIMENSIONS),
    "gif_y_limit_scope": "per_simulation_all_stored_frames",
    "gif_y_limit_padding_fraction": GIF_Y_LIMIT_PADDING_FRACTION,
    "gif_loop_forever": True,
    "gif_descriptive_only": True,
}
DEFINITIONS: Final = {
    "eta_slope": "maximum over retained times of max_x |partial_x eta|",
    "gxi_high_band": (
        f"maximum over retained times of the G(eta)xi Fourier-energy "
        f"fraction in {HIGH_BAND_START} <= |k| <= {DELIVERED_MODE}, "
        f"relative to 1 <= |k| <= {DELIVERED_MODE}"
    ),
    "gxi_sign_changes": (
        "maximum over retained times of cyclic sign changes in the "
        "forward difference of G(eta)xi after discarding differences "
        f"below {SIGN_DIFFERENCE_RELATIVE_THRESHOLD:.0%} of that frame's maximum"
    ),
    "combined": (
        "mean of the within-family empirical ranks of eta_slope, "
        "gxi_high_band, and gxi_sign_changes"
    ),
    "stored_band_quadratic_energy_drift": (
        "maximum over retained times of the relative drift in "
        "0.5*dx*(sum(xi*G(eta)xi)+sum(eta^2)), evaluated only from "
        "the stored delivered-band fields; this is not the internal-band "
        "Hamiltonian used by Benjamin--Feir or JONSWAP/TMA acceptance"
    ),
}


TrajectoryIndex = NamedTuple(
    "TrajectoryIndex",
    [
        ("accepted_index", int),
        ("simulation_id", int),
        ("category", str),
        ("first_row", int),
        ("row_count", int),
    ],
)


DatasetGroup = NamedTuple(
    "DatasetGroup",
    [
        ("root", Path),
        ("family", str),
        ("split", str),
        ("trajectories", tuple[TrajectoryIndex, ...]),
    ],
)


SimulationMetrics = NamedTuple(
    "SimulationMetrics",
    [
        ("source_index", int),
        ("accepted_index", int),
        ("family", str),
        ("split", str),
        ("simulation_id", int),
        ("category", str),
        ("first_row", int),
        ("row_count", int),
        ("depth", float),
        ("all_frames_finite", bool),
        ("constant_depth", bool),
        ("ordered_time", bool),
        ("minimum_water_column", float),
        ("minimum_water_fraction", float),
        ("maximum_eta_slope", float),
        ("maximum_eta_slope_frame", int),
        ("maximum_gxi_high_band_fraction", float),
        ("maximum_gxi_high_band_fraction_frame", int),
        ("maximum_thresholded_gxi_sign_changes", int),
        ("maximum_thresholded_gxi_sign_changes_frame", int),
        ("maximum_relative_stored_band_quadratic_energy_drift", float),
        ("maximum_relative_stored_band_quadratic_energy_drift_frame", int),
    ],
)


LoadedTrajectory = NamedTuple(
    "LoadedTrajectory",
    [
        ("eta", np.ndarray),
        ("xi", np.ndarray),
        ("gxi", np.ndarray),
        ("depth", np.ndarray),
        ("time", np.ndarray),
    ],
)


def validate_final_paper_dataset(
    sources: Sequence[DatasetGroup],
    *,
    retained_rows: int,
) -> None:
    """Require the paper release population in the stored rows."""

    source_count = len(sources)
    observed_family_split_counts: dict[tuple[str, str], int] = {}
    for source in sources:
        key = (source.family, source.split)
        observed_family_split_counts[key] = observed_family_split_counts.get(
            key, 0
        ) + len(source.trajectories)
    accepted_simulations = sum(observed_family_split_counts.values())
    expected_family_split_counts = {
        (family, split): split_count
        for family in FAMILY_LABELS
        for split, split_count in FINAL_PAPER_DATASET_SPLIT_ACCEPTED_SIMULATIONS.items()
    }
    mismatches: list[str] = []
    if source_count != FINAL_PAPER_DATASET_SOURCE_COUNT:
        mismatches.append(
            f"sources={source_count}, expected={FINAL_PAPER_DATASET_SOURCE_COUNT}"
        )
    if observed_family_split_counts != expected_family_split_counts:
        mismatches.append(
            "family/split accepted counts differ: "
            f"observed={observed_family_split_counts}, "
            f"expected={expected_family_split_counts}"
        )
    if accepted_simulations != FINAL_PAPER_DATASET_ACCEPTED_SIMULATIONS:
        mismatches.append(
            f"accepted_simulations={accepted_simulations}, "
            f"expected={FINAL_PAPER_DATASET_ACCEPTED_SIMULATIONS}"
        )
    if retained_rows != FINAL_PAPER_DATASET_RETAINED_ROWS:
        mismatches.append(
            f"retained_rows={retained_rows}, "
            f"expected={FINAL_PAPER_DATASET_RETAINED_ROWS}"
        )
    if mismatches:
        raise ValueError(
            "final paper-dataset contract failed: " + "; ".join(mismatches)
        )


def load_dataset_groups(dataset_path: Path) -> tuple[DatasetGroup, ...]:
    """Group contiguous simulations by family and split without loading wave fields."""
    root = dataset_path.expanduser().resolve(strict=True)
    arrays = {
        name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in (
            "family_id",
            "dataset_split",
            "simulation_id",
            "parameter_group_id",
            "frame_index",
        )
    }
    family_ids = arrays["family_id"]
    splits = arrays["dataset_split"]
    simulation_ids = arrays["simulation_id"]
    if family_ids.size == 0 or any(
        values.shape != (family_ids.size,) for values in arrays.values()
    ):
        raise ValueError("dataset row metadata must have matching nonempty shapes")
    starts = np.r_[
        0,
        np.flatnonzero(
            (family_ids[1:] != family_ids[:-1])
            | (splits[1:] != splits[:-1])
            | (simulation_ids[1:] != simulation_ids[:-1])
        )
        + 1,
    ]
    ends = np.r_[starts[1:], family_ids.size]
    family_names = dict(enumerate(FAMILY_LABELS, start=1))
    groups: dict[tuple[str, str], list[TrajectoryIndex]] = {}
    seen: set[tuple[str, str, int]] = set()
    for first, end in zip(starts, ends, strict=True):
        family = family_names[int(family_ids[first])]
        split = str(splits[first])
        simulation_id = int(simulation_ids[first])
        identity = (family, split, simulation_id)
        if identity in seen:
            raise ValueError("simulation rows must form one contiguous block")
        seen.add(identity)
        if not np.array_equal(arrays["frame_index"][first:end], np.arange(end - first)):
            raise ValueError("simulation frame indices must start at zero and increase")
        category = str(arrays["parameter_group_id"][first])
        if np.any(arrays["parameter_group_id"][first:end] != category):
            raise ValueError("simulation parameter group must not change between rows")
        trajectories = groups.setdefault((family, split), [])
        trajectories.append(
            TrajectoryIndex(
                accepted_index=len(trajectories),
                simulation_id=simulation_id,
                category=category,
                first_row=int(first),
                row_count=int(end - first),
            )
        )
    return tuple(
        DatasetGroup(root, family, split, tuple(trajectories))
        for (family, split), trajectories in groups.items()
    )


def _argmax(values: np.ndarray) -> tuple[float, int]:
    index = int(np.argmax(values))
    return float(values[index]), index


def _audit_simulations(
    task: tuple[int, DatasetGroup, int],
) -> tuple[SimulationMetrics, ...]:
    source_index, source, block_rows = task
    trajectories = source.trajectories
    offset = trajectories[0].first_row
    end = trajectories[-1].first_row + trajectories[-1].row_count
    eta, xi, gxi, depth, time = (
        np.load(source.root / f"{name}.npy", mmap_mode="r", allow_pickle=False)[
            offset:end
        ]
        for name in (*FIELD_NAMES, "depth", "time")
    )
    if eta.shape != xi.shape or eta.shape != gxi.shape or eta.ndim != 2:
        raise RuntimeError(f"field shape mismatch in {source.root}")
    row_count, nx = eta.shape
    if depth.shape != (row_count,) or time.shape != (row_count,):
        raise RuntimeError(f"scalar row shape mismatch in {source.root}")

    finite = np.empty(row_count, dtype=np.bool_)
    minimum_water = np.empty(row_count, dtype=np.float64)
    eta_slope = np.empty(row_count, dtype=np.float64)
    gxi_high_fraction = np.empty(row_count, dtype=np.float64)
    thresholded_sign_changes = np.empty(row_count, dtype=np.int32)
    stored_band_quadratic_energy = np.empty(row_count, dtype=np.float64)

    modes = np.fft.rfftfreq(nx, d=2.0 * np.pi / nx) * (2.0 * np.pi)
    delivered = (modes >= 1.0) & (modes <= DELIVERED_MODE)
    high = (modes >= HIGH_BAND_START) & (modes <= DELIVERED_MODE)
    tiny = np.finfo(np.float64).tiny
    dx = 2.0 * np.pi / nx

    for start in range(0, row_count, block_rows):
        stop = min(start + block_rows, row_count)
        rows = slice(start, stop)
        eta_block = np.asarray(eta[rows], dtype=np.float64)
        xi_block = np.asarray(xi[rows], dtype=np.float64)
        gxi_block = np.asarray(gxi[rows], dtype=np.float64)
        finite[rows] = (
            np.all(np.isfinite(eta_block), axis=1)
            & np.all(np.isfinite(xi_block), axis=1)
            & np.all(np.isfinite(gxi_block), axis=1)
            & np.isfinite(depth[rows])
            & np.isfinite(time[rows])
        )
        minimum_water[rows] = depth[rows] + np.min(eta_block, axis=1)
        eta_coefficients = np.fft.rfft(eta_block, axis=1)
        gxi_coefficients = np.fft.rfft(gxi_block, axis=1)
        eta_slope[rows] = np.max(
            np.abs(
                np.fft.irfft(
                    1j * modes[None, :] * eta_coefficients,
                    n=nx,
                    axis=1,
                )
            ),
            axis=1,
        )
        gxi_energy = np.abs(gxi_coefficients) ** 2
        gxi_high_fraction[rows] = np.sum(gxi_energy[:, high], axis=1) / np.maximum(
            np.sum(gxi_energy[:, delivered], axis=1),
            tiny,
        )
        difference = np.roll(gxi_block, -1, axis=1) - gxi_block
        row_scales = np.max(np.abs(difference), axis=1, keepdims=True)
        thresholds = SIGN_DIFFERENCE_RELATIVE_THRESHOLD * row_scales
        signs = np.where(
            difference > thresholds,
            1,
            np.where(difference < -thresholds, -1, 0),
        ).astype(np.int8)
        nonzero = signs != 0
        nonzero_counts = np.count_nonzero(nonzero, axis=1)
        max_nonzero = int(np.max(nonzero_counts))
        if max_nonzero == 0:
            sign_change_counts = np.zeros(difference.shape[0], dtype=np.int32)
        else:
            nonzero_signs = np.zeros(
                (difference.shape[0], max_nonzero),
                dtype=np.int8,
            )
            order = np.cumsum(nonzero, axis=1) - 1
            row_ids, column_ids = np.nonzero(nonzero)
            nonzero_signs[row_ids, order[row_ids, column_ids]] = signs[
                row_ids,
                column_ids,
            ]
            valid_next = np.arange(max_nonzero - 1)[None, :] < (
                nonzero_counts[:, None] - 1
            )
            adjacent_changes = (
                nonzero_signs[:, 1:] != nonzero_signs[:, :-1]
            ) & valid_next
            final_indices = np.maximum(nonzero_counts - 1, 0)
            wrap_changes = (nonzero_counts > 1) & (
                nonzero_signs[:, 0]
                != nonzero_signs[np.arange(difference.shape[0]), final_indices]
            )
            sign_change_counts = adjacent_changes.sum(axis=1).astype(
                np.int32
            ) + wrap_changes.astype(np.int32)
        thresholded_sign_changes[rows] = sign_change_counts
        stored_band_quadratic_energy[rows] = (
            0.5
            * dx
            * (np.sum(xi_block * gxi_block, axis=1) + np.sum(eta_block**2, axis=1))
        )

    records: list[SimulationMetrics] = []
    for trajectory in trajectories:
        first = trajectory.first_row - offset
        rows = slice(first, first + trajectory.row_count)
        simulation_time = time[rows]
        simulation_depth = depth[rows]
        slope_value, slope_frame = _argmax(eta_slope[rows])
        high_value, high_frame = _argmax(gxi_high_fraction[rows])
        sign_value, sign_frame = _argmax(thresholded_sign_changes[rows])
        initial_stored_energy = float(stored_band_quadratic_energy[first])
        relative_drift = np.abs(
            stored_band_quadratic_energy[rows] - initial_stored_energy
        ) / max(
            abs(initial_stored_energy),
            tiny,
        )
        drift_value, drift_frame = _argmax(relative_drift)
        minimum_water_value = float(np.min(minimum_water[rows]))
        records.append(
            SimulationMetrics(
                source_index=source_index,
                accepted_index=trajectory.accepted_index,
                family=source.family,
                split=source.split,
                simulation_id=trajectory.simulation_id,
                category=trajectory.category,
                first_row=trajectory.first_row,
                row_count=trajectory.row_count,
                depth=float(simulation_depth[0]),
                all_frames_finite=bool(np.all(finite[rows])),
                constant_depth=bool(np.all(simulation_depth == simulation_depth[0])),
                ordered_time=bool(
                    simulation_time.size == 1 or np.all(np.diff(simulation_time) > 0.0)
                ),
                minimum_water_column=minimum_water_value,
                minimum_water_fraction=minimum_water_value / float(simulation_depth[0]),
                maximum_eta_slope=slope_value,
                maximum_eta_slope_frame=slope_frame,
                maximum_gxi_high_band_fraction=high_value,
                maximum_gxi_high_band_fraction_frame=high_frame,
                maximum_thresholded_gxi_sign_changes=int(sign_value),
                maximum_thresholded_gxi_sign_changes_frame=sign_frame,
                maximum_relative_stored_band_quadratic_energy_drift=drift_value,
                maximum_relative_stored_band_quadratic_energy_drift_frame=drift_frame,
            )
        )
    return tuple(records)


def _descending_indices(values: np.ndarray, count: int) -> tuple[int, ...]:
    return tuple(
        map(
            int,
            np.argsort(-values, kind="stable")[: min(count, values.size)],
        )
    )


def _simulation_key(simulation: SimulationMetrics) -> tuple[int, int]:
    return simulation.source_index, simulation.accepted_index


def _line_style(role: str) -> tuple[str, str, float]:
    if "worst" in role:
        return "#d97706", "-", 1.45
    if "terminal" in role:
        return "#2563eb", "--", 1.15
    return "#64748b", ":", 1.05


def _plot_simulations(
    simulations: Sequence[SimulationMetrics],
    trajectories: Mapping[tuple[int, int], LoadedTrajectory],
    frame_fields: Sequence[int],
    row_labels: Sequence[str],
    title: str,
    output_stem: Path,
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(
        len(simulations),
        4,
        figsize=(17.5, 2.8 * len(simulations)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, (simulation, worst_frame, row_label) in enumerate(
        zip(simulations, frame_fields, row_labels)
    ):
        trajectory = trajectories[_simulation_key(simulation)]
        x = np.linspace(0.0, 2.0 * np.pi, trajectory.eta.shape[1], endpoint=False)
        fields = (trajectory.eta, trajectory.xi, trajectory.gxi)
        roles: dict[int, list[str]] = {}
        for frame, role in (
            (0, "initial"),
            (worst_frame, "worst"),
            (trajectory.time.size - 1, "terminal"),
        ):
            roles.setdefault(frame, []).append(role)
        frame_roles = tuple(
            (frame, "/".join(value)) for frame, value in sorted(roles.items())
        )
        for column, (field, field_title) in enumerate(zip(fields, FIELD_TITLES)):
            axis = axes[row, column]
            for frame, role in frame_roles:
                color, style, width = _line_style(role)
                axis.plot(
                    x,
                    field[frame],
                    color=color,
                    linestyle=style,
                    linewidth=width,
                    label=rf"{role}, $t={trajectory.time[frame]:.3g}$",
                )
            axis.grid(alpha=0.2, linewidth=0.5)
            axis.set_xlim(0.0, 2.0 * np.pi)
            axis.set_xticks((0.0, np.pi, 2.0 * np.pi))
            axis.set_xticklabels(("0", r"$\pi$", r"$2\pi$"))
            if row == 0:
                axis.set_title(field_title)
            if row == len(simulations) - 1:
                axis.set_xlabel(r"$x$")
            if column == 0:
                axis.set_ylabel(
                    row_label,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=16,
                    fontsize=8,
                )

        spectrum_axis = axes[row, 3]
        for frame, role in frame_roles:
            color, style, width = _line_style(role)
            field = trajectory.gxi[frame]
            amplitude = np.abs(np.fft.rfft(field - float(np.mean(field))))
            upper = min(DELIVERED_MODE + 1, amplitude.size)
            scale = float(np.max(amplitude[1:upper])) if upper > 1 else 0.0
            spectrum = amplitude / scale if scale > 0.0 else amplitude
            modes = np.arange(min(DELIVERED_MODE + 1, spectrum.size))
            spectrum_axis.semilogy(
                modes,
                np.maximum(spectrum[: modes.size], 1.0e-13),
                color=color,
                linestyle=style,
                linewidth=width,
                label=rf"{role}, $t={trajectory.time[frame]:.3g}$",
            )
        spectrum_axis.axvspan(
            HIGH_BAND_START,
            DELIVERED_MODE,
            color="#f59e0b",
            alpha=0.10,
        )
        spectrum_axis.set_xlim(0, DELIVERED_MODE)
        spectrum_axis.set_ylim(1.0e-13, 2.0)
        spectrum_axis.grid(alpha=0.2, linewidth=0.5)
        if row == 0:
            spectrum_axis.set_title(r"normalized $|\widehat{G(\eta)\xi}_k|$")
        if row == len(simulations) - 1:
            spectrum_axis.set_xlabel(r"mode $|k|$")
        spectrum_axis.legend(frameon=False, fontsize=7, loc="best")

    figure.suptitle(title, fontsize=14)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=180)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        f"q{int(round(100.0 * quantile)):03d}": float(np.quantile(values, quantile))
        for quantile in QUANTILES
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--block-rows", type=int, default=256)
    parser.add_argument("--top-count", type=int, default=6)
    parser.add_argument(
        "--require-final-paper-dataset",
        action="store_true",
        help=("Require the exact four-family paper release population."),
    )
    args = parser.parse_args()
    sources = load_dataset_groups(args.dataset)
    retained_rows_from_metadata = sum(
        trajectory.row_count for source in sources for trajectory in source.trajectories
    )
    if args.require_final_paper_dataset:
        validate_final_paper_dataset(
            sources,
            retained_rows=retained_rows_from_metadata,
        )

    final_output_dir = args.output_dir.expanduser().resolve()
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_output_dir.exists() or final_output_dir.is_symlink():
        raise FileExistsError(f"output path already exists: {final_output_dir}")
    output_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output_dir.name}.staging-", dir=final_output_dir.parent
        )
    )
    try:
        tasks = (
            (
                source_index,
                source._replace(trajectories=source.trajectories[start : start + 32]),
                args.block_rows,
            )
            for source_index, source in enumerate(sources)
            for start in range(0, len(source.trajectories), 32)
        )
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            simulation_groups = tuple(pool.map(_audit_simulations, tasks))
        simulations = tuple(
            simulation for group in simulation_groups for simulation in group
        )
        retained_rows = sum(simulation.row_count for simulation in simulations)
        if args.require_final_paper_dataset:
            validate_final_paper_dataset(sources, retained_rows=retained_rows)
        if any(
            not (
                simulation.all_frames_finite
                and simulation.constant_depth
                and simulation.ordered_time
            )
            or simulation.minimum_water_column <= 0.0
            for simulation in simulations
        ):
            raise RuntimeError("a completed accepted trajectory failed a hard audit")

        by_family = {
            family: tuple(
                simulation for simulation in simulations if simulation.family == family
            )
            for family in sorted({simulation.family for simulation in simulations})
        }
        family_results: dict[str, Any] = {}
        all_selected: list[SimulationMetrics] = []
        ranking_data: dict[
            str, tuple[dict[str, tuple[int, ...]], np.ndarray, tuple[int, ...]]
        ] = {}
        for family, family_simulations in by_family.items():
            combined_values = {
                "eta_slope": np.asarray(
                    [simulation.maximum_eta_slope for simulation in family_simulations],
                    dtype=np.float64,
                ),
                "gxi_high_band": np.asarray(
                    [
                        simulation.maximum_gxi_high_band_fraction
                        for simulation in family_simulations
                    ],
                    dtype=np.float64,
                ),
                "gxi_sign_changes": np.asarray(
                    [
                        simulation.maximum_thresholded_gxi_sign_changes
                        for simulation in family_simulations
                    ],
                    dtype=np.float64,
                ),
            }
            rank_fractions: dict[str, np.ndarray] = {}
            for name, value in combined_values.items():
                order = np.argsort(value, kind="stable")
                ranks = np.empty(value.size, dtype=np.float64)
                ranks[order] = np.arange(value.size, dtype=np.float64)
                rank_fractions[name] = ranks / max(value.size - 1, 1)
            combined = np.mean(np.stack(tuple(rank_fractions.values())), axis=0)
            values = {
                **combined_values,
                "stored_band_quadratic_energy_drift": np.asarray(
                    [
                        simulation.maximum_relative_stored_band_quadratic_energy_drift
                        for simulation in family_simulations
                    ],
                    dtype=np.float64,
                ),
            }
            rankings = {
                "combined": _descending_indices(combined, args.top_count),
                **{
                    name: _descending_indices(value, args.top_count)
                    for name, value in values.items()
                },
            }
            combined_frames: list[int] = []
            frame_fields = (
                "maximum_eta_slope_frame",
                "maximum_gxi_high_band_fraction_frame",
                "maximum_thresholded_gxi_sign_changes_frame",
            )
            rank_names = tuple(combined_values)
            for index in rankings["combined"]:
                dominant = int(
                    np.argmax([rank_fractions[name][index] for name in rank_names])
                )
                combined_frames.append(
                    int(getattr(family_simulations[index], frame_fields[dominant]))
                )
            combined_frame_indices = tuple(combined_frames)
            ranking_data[family] = (rankings, combined, combined_frame_indices)
            selected_indices = tuple(
                dict.fromkeys(
                    index for ranking in rankings.values() for index in ranking
                )
            )
            all_selected.extend(family_simulations[index] for index in selected_indices)
            family_results[family] = {
                "accepted_simulations": len(family_simulations),
                "retained_rows": sum(
                    simulation.row_count for simulation in family_simulations
                ),
                "splits": {
                    split: sum(
                        simulation.split == split for simulation in family_simulations
                    )
                    for split in ("train", "validation", "test")
                },
                "quantiles": {
                    **{name: _quantiles(value) for name, value in values.items()},
                    "minimum_water_fraction": _quantiles(
                        np.asarray(
                            [
                                simulation.minimum_water_fraction
                                for simulation in family_simulations
                            ]
                        )
                    ),
                },
                "rankings": {
                    name: [
                        {
                            **family_simulations[index]._asdict(),
                            "source_root": str(
                                sources[family_simulations[index].source_index].root
                            ),
                            "combined_empirical_rank": float(combined[index]),
                        }
                        for index in indices
                    ]
                    for name, indices in rankings.items()
                },
            }

        arrays = {
            name: np.load(
                sources[0].root / f"{name}.npy", mmap_mode="r", allow_pickle=False
            )
            for name in (*FIELD_NAMES, "depth", "time")
        }
        loaded: dict[tuple[int, int], LoadedTrajectory] = {}
        for simulation in all_selected:
            rows = slice(
                simulation.first_row, simulation.first_row + simulation.row_count
            )
            loaded[_simulation_key(simulation)] = LoadedTrajectory(
                **{
                    name: np.asarray(values[rows], dtype=np.float64)
                    for name, values in arrays.items()
                }
            )
        figures: list[Path] = []
        animations: dict[str, object] = {}
        overview_simulations: list[SimulationMetrics] = []
        overview_frames: list[int] = []
        overview_labels: list[str] = []
        plot_definitions = {
            "combined": (
                "combined diagnostic rank",
                None,
            ),
            "eta_slope": (
                r"maximum $|\partial_x\eta|$",
                "maximum_eta_slope_frame",
            ),
            "gxi_high_band": (
                rf"maximum $G(\eta)\xi$ energy fraction in {HIGH_BAND_START}--{DELIVERED_MODE}",
                "maximum_gxi_high_band_fraction_frame",
            ),
            "gxi_sign_changes": (
                r"most amplitude-thresholded sign changes of $\Delta_xG(\eta)\xi$",
                "maximum_thresholded_gxi_sign_changes_frame",
            ),
            "stored_band_quadratic_energy_drift": (
                "maximum relative stored-band quadratic-energy drift",
                "maximum_relative_stored_band_quadratic_energy_drift_frame",
            ),
        }

        for family, family_simulations in by_family.items():
            rankings, combined, combined_frame_indices = ranking_data[family]
            family_label = FAMILY_LABELS.get(family, family)
            for ranking_name, (metric_label, frame_field) in plot_definitions.items():
                indices = rankings[ranking_name]
                selected = tuple(family_simulations[index] for index in indices)
                frames = (
                    combined_frame_indices
                    if frame_field is None
                    else tuple(
                        int(getattr(simulation, frame_field)) for simulation in selected
                    )
                )
                labels = tuple(
                    f"#{rank} {simulation.split}; simulation {simulation.simulation_id}\n"
                    f"{simulation.category}; h={simulation.depth:.4g}\n"
                    f"slope={simulation.maximum_eta_slope:.3g}; "
                    f"high={100.0 * simulation.maximum_gxi_high_band_fraction:.3g}%; "
                    f"signs={simulation.maximum_thresholded_gxi_sign_changes}; "
                    "stored-band dE="
                    f"{simulation.maximum_relative_stored_band_quadratic_energy_drift:.3g}"
                    for rank, simulation in enumerate(selected, start=1)
                )
                figures.extend(
                    _plot_simulations(
                        selected,
                        loaded,
                        frames,
                        labels,
                        f"{family_label}: accepted simulations with the {metric_label}",
                        output_dir / f"{family}_worst_{ranking_name}",
                    )
                )
                top_simulation = selected[0]
                loaded_trajectory = loaded[_simulation_key(top_simulation)]
                title = f"{family_label}: rank-one accepted simulation by the {metric_label}"
                gif_path = output_dir / f"{family}_worst_{ranking_name}.gif"
                stored_frames = int(loaded_trajectory.time.size)
                frame_indices = np.rint(
                    np.linspace(
                        0, stored_frames - 1, min(stored_frames, GIF_MAXIMUM_FRAMES)
                    )
                ).astype(np.int32)
                fps = (
                    GIF_SHORT_FPS
                    if frame_indices.size <= GIF_SHORT_FRAME_THRESHOLD
                    else GIF_LONG_FPS
                )
                fields = (
                    loaded_trajectory.eta,
                    loaded_trajectory.xi,
                    loaded_trajectory.gxi,
                )
                limits: dict[str, tuple[float, float]] = {}
                for name, field in zip(FIELD_NAMES, fields):
                    lower, upper = float(np.min(field)), float(np.max(field))
                    span = upper - lower
                    padding = GIF_Y_LIMIT_PADDING_FRACTION * (
                        span if span > 0.0 else max(abs(lower), 1.0)
                    )
                    limits[name] = (lower - padding, upper + padding)
                animation_record = {
                    "stored_frames": stored_frames,
                    "frame_indices": frame_indices.tolist(),
                    "first_time": float(loaded_trajectory.time[0]),
                    "last_time": float(loaded_trajectory.time[-1]),
                    "fps": fps,
                    "dimensions_pixels": list(GIF_DIMENSIONS),
                    "field_y_limits": {
                        name: list(bounds) for name, bounds in limits.items()
                    },
                }
                x = np.linspace(
                    0.0, 2.0 * np.pi, loaded_trajectory.eta.shape[1], endpoint=False
                )
                figure, axes = plt.subplots(
                    1,
                    len(fields),
                    figsize=(12.5, 3.6),
                    constrained_layout=True,
                )
                lines = []
                try:
                    for axis, name, field, field_title in zip(
                        axes,
                        FIELD_NAMES,
                        fields,
                        FIELD_TITLES,
                    ):
                        (line,) = axis.plot(
                            x,
                            field[frame_indices[0]],
                            color="#2563eb",
                            linewidth=1.4,
                        )
                        axis.set_xlim(0.0, 2.0 * np.pi)
                        axis.set_ylim(*limits[name])
                        axis.set_xticks((0.0, np.pi, 2.0 * np.pi))
                        axis.set_xticklabels(("0", r"$\pi$", r"$2\pi$"))
                        axis.set_xlabel(r"$x$")
                        axis.set_title(field_title)
                        axis.grid(alpha=0.2, linewidth=0.5)
                        lines.append(line)
                    figure.suptitle(
                        f"{title}   "
                        rf"($h={top_simulation.depth:.3g}$, simulation {top_simulation.simulation_id})",
                        fontsize=12,
                    )
                    time_text = figure.text(
                        0.5,
                        0.92,
                        "",
                        ha="center",
                        va="center",
                        fontsize=11,
                    )

                    def update(animation_index: int) -> tuple[Any, ...]:
                        frame_index = int(frame_indices[animation_index])
                        for line, field in zip(lines, fields):
                            line.set_ydata(field[frame_index])
                        time_text.set_text(
                            rf"$t={loaded_trajectory.time[frame_index]:.2f}$"
                        )
                        return (*lines, time_text)

                    if frame_indices.size == 1:
                        update(0)
                        writer = PillowWriter(fps=fps)
                        writer.setup(figure, gif_path, dpi=100)
                        writer.grab_frame()
                        writer.finish()
                    else:
                        animation = FuncAnimation(
                            figure,
                            update,
                            frames=frame_indices.size,
                            interval=1_000.0 / fps,
                            blit=False,
                            repeat=True,
                        )
                        animation.save(
                            gif_path,
                            writer=PillowWriter(fps=fps),
                            dpi=100,
                        )
                finally:
                    plt.close(figure)

                with Image.open(gif_path) as image:
                    expected_duration = 250 if fps == GIF_SHORT_FPS else 80
                    if (
                        image.format != "GIF"
                        or tuple(map(int, image.size)) != GIF_DIMENSIONS
                        or int(getattr(image, "n_frames", 1)) != frame_indices.size
                        or image.info.get("loop") != 0
                    ):
                        raise RuntimeError(f"rendered GIF contract differs: {gif_path}")
                    for index in range(frame_indices.size):
                        image.seek(index)
                        image.load()
                        if image.info.get("duration") != expected_duration:
                            raise RuntimeError(
                                f"rendered GIF frame duration differs: {gif_path}"
                            )
                figures.append(gif_path)
                animations[gif_path.name] = {
                    "family": family,
                    "ranking": ranking_name,
                    "rank": 1,
                    "source_index": top_simulation.source_index,
                    "accepted_index": top_simulation.accepted_index,
                    "simulation_id": top_simulation.simulation_id,
                    "category": top_simulation.category,
                    "split": top_simulation.split,
                    **animation_record,
                }

            top_index = rankings["combined"][0]
            top_simulation = family_simulations[top_index]
            overview_simulations.append(top_simulation)
            overview_frames.append(combined_frame_indices[0])
            overview_labels.append(
                f"{family_label}; {top_simulation.split}; "
                f"simulation {top_simulation.simulation_id}\n"
                f"{top_simulation.category}; h={top_simulation.depth:.4g}\n"
                f"slope={top_simulation.maximum_eta_slope:.3g}; "
                f"high={100.0 * top_simulation.maximum_gxi_high_band_fraction:.3g}%; "
                f"signs={top_simulation.maximum_thresholded_gxi_sign_changes}; "
                "stored-band dE="
                f"{top_simulation.maximum_relative_stored_band_quadratic_energy_drift:.3g}"
            )

        figures.extend(
            _plot_simulations(
                overview_simulations,
                loaded,
                overview_frames,
                overview_labels,
                "Largest combined diagnostic rank in each completed dataset family",
                output_dir / "all_families_worst_overview",
            )
        )

        record = {
            "schema": DIAGNOSTIC_SCHEMA,
            "status": "complete",
            "interpretation": INTERPRETATION,
            "dataset": str(sources[0].root),
            "parameters": {
                **PARAMETERS,
                "final_paper_dataset_contract_required": bool(
                    args.require_final_paper_dataset
                ),
            },
            "definitions": dict(DEFINITIONS),
            "population": {
                "sources": len(sources),
                "accepted_simulations": len(simulations),
                "retained_rows": retained_rows,
            },
            "sources": [
                {
                    "root": str(source.root),
                    "family": source.family,
                    "split": source.split,
                    "accepted_simulations": len(source.trajectories),
                    "retained_rows": sum(
                        trajectory.row_count for trajectory in source.trajectories
                    ),
                }
                for source in sources
            ],
            "families": family_results,
            "animations": animations,
            "artifacts": {
                path.name: {
                    "path": str(
                        final_output_dir / path.resolve().relative_to(output_dir)
                    ),
                    "bytes": path.stat().st_size,
                }
                for path in figures
            },
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        accepted_simulations = len(simulations)
        figure_relative_paths = tuple(path.relative_to(output_dir) for path in figures)
        summary_relative_path = summary_path.relative_to(output_dir)
        if final_output_dir.exists() or final_output_dir.is_symlink():
            raise FileExistsError(
                f"output path appeared during rendering: {final_output_dir}"
            )
        output_dir.rename(final_output_dir)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)

    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(final_output_dir),
                "sources": len(sources),
                "accepted_simulations": accepted_simulations,
                "retained_rows": retained_rows,
                "figures": [
                    str(final_output_dir / path) for path in figure_relative_paths
                ],
                "summary": str(final_output_dir / summary_relative_path),
            },
            indent=2,
        )
    )
