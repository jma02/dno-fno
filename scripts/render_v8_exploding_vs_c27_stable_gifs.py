from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V8_DIR = (
    ROOT / "outputs/cs_dno_w512b8_l256_v8_2gpu_20260619_024621" / "eval_suite_f64h"
)
DEFAULT_C27_DIR = (
    ROOT
    / "outputs/c27_h1_to_l2_full_20260717_212550"
    / "eval_final_soliton_spectral_guard_20260719_191228"
)
DEFAULT_OUTPUT = ROOT / "outputs/v8_exploding_vs_c27_stable_gifs_20260820"
DEFAULT_SIMULATIONS = ("tanaka_g0:5", "tanaka_g0:11", "tanaka_g1:1000006")


@dataclass(frozen=True)
class RolloutArchive:
    path: Path
    simulation_ids: np.ndarray
    times: np.ndarray
    depths: np.ndarray
    truth_eta: np.ndarray
    truth_xi: np.ndarray
    pred_eta: np.ndarray
    pred_xi: np.ndarray
    pred_gxi: np.ndarray


def _load_archive(path: Path) -> RolloutArchive:
    with np.load(path, allow_pickle=False) as archive:
        return RolloutArchive(
            path=path.resolve(),
            simulation_ids=np.array(archive["simulation_ids"], copy=True),
            times=np.array(archive["times"], copy=True),
            depths=np.array(archive["depths"], copy=True),
            truth_eta=np.array(archive["truth_eta"], copy=True),
            truth_xi=np.array(archive["truth_xi"], copy=True),
            pred_eta=np.array(archive["pred_eta"], copy=True),
            pred_xi=np.array(archive["pred_xi"], copy=True),
            pred_gxi=np.array(archive["pred_gxi"], copy=True),
        )


def _simulation_index(archive: RolloutArchive, simulation_id: int) -> int:
    matches = np.flatnonzero(archive.simulation_ids == simulation_id)
    if matches.size != 1:
        raise ValueError(
            f"expected exactly one simulation ID {simulation_id} in {archive.path}, "
            f"found {matches.size}"
        )
    return int(matches[0])


def _validate_match(
    old: RolloutArchive,
    old_index: int,
    new: RolloutArchive,
    new_index: int,
) -> None:
    if not np.array_equal(old.times, new.times):
        raise ValueError("saved time grids differ between old and C27 archives")
    if old.depths[old_index] != new.depths[new_index]:
        raise ValueError("depths differ between the matched simulations")
    for name in ("truth_eta", "truth_xi"):
        old_initial = getattr(old, name)[0, old_index]
        new_initial = getattr(new, name)[0, new_index]
        if not np.array_equal(old_initial, new_initial):
            raise ValueError(
                f"frame-zero {name} differs between the matched simulations"
            )


def _finite_frames(archive: RolloutArchive, simulation_index: int) -> np.ndarray:
    fields = (archive.pred_eta, archive.pred_xi, archive.pred_gxi)
    return np.logical_and.reduce(
        tuple(
            np.all(np.isfinite(field[:, simulation_index]), axis=1) for field in fields
        )
    )


def _last_finite_index(archive: RolloutArchive, simulation_index: int) -> int:
    indices = np.flatnonzero(_finite_frames(archive, simulation_index))
    if indices.size == 0:
        raise ValueError(
            f"simulation index {simulation_index} has no finite saved frame"
        )
    return int(indices[-1])


def _frame_indices(
    frame_count: int, saved_count: int, required: list[int]
) -> np.ndarray:
    if frame_count < 2:
        raise ValueError("frame_count must be at least two")
    sampled = np.linspace(0, saved_count - 1, frame_count, dtype=np.int64)
    return np.unique(np.concatenate((sampled, np.asarray(required, dtype=np.int64))))


def _finite_max_per_frame(values: np.ndarray) -> np.ndarray:
    finite_abs = np.where(np.isfinite(values), np.abs(values), 0.0)
    return np.max(finite_abs, axis=1)


def _positive_limit(value: float, floor: float) -> float:
    return max(float(value) * 1.08, float(floor), np.finfo(float).tiny)


def _render_simulation(
    family: str,
    simulation_id: int,
    old: RolloutArchive,
    new: RolloutArchive,
    output_dir: Path,
    frame_count: int,
    fps: int,
    dpi: int,
) -> dict[str, Any]:
    old_index = _simulation_index(old, simulation_id)
    new_index = _simulation_index(new, simulation_id)
    _validate_match(old, old_index, new, new_index)

    old_finite = _finite_frames(old, old_index)
    new_finite = _finite_frames(new, new_index)
    last_old = _last_finite_index(old, old_index)
    if np.all(old_finite):
        raise ValueError(f"old simulation {family}:{simulation_id} does not explode")
    if not np.all(new_finite):
        raise ValueError(
            f"C27 simulation {family}:{simulation_id} is not finite to T=200"
        )

    indices = _frame_indices(
        frame_count,
        new.times.size,
        [last_old, min(last_old + 1, new.times.size - 1)],
    )
    x = np.linspace(0.0, 2.0 * np.pi, old.pred_eta.shape[-1], endpoint=False)
    old_fields = (old.pred_eta[:, old_index], old.pred_gxi[:, old_index])
    new_fields = (new.pred_eta[:, new_index], new.pred_gxi[:, new_index])
    new_limits = tuple(
        _positive_limit(np.max(_finite_max_per_frame(values)), 1e-12)
        for values in new_fields
    )
    old_cumulative_limits = tuple(
        np.maximum.accumulate(_finite_max_per_frame(values)) for values in old_fields
    )
    old_floors = tuple(
        _positive_limit(np.max(np.abs(values[0])), new_limits[column] * 0.05)
        for column, values in enumerate(old_fields)
    )

    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.8), sharex=True)
    figure.subplots_adjust(top=0.84, hspace=0.34, wspace=0.25)
    colors = ("#b42318", "#1167b1")
    lines = []
    amplitude_labels = []
    failure_labels = []
    column_labels = (r"surface elevation $\eta$", r"predicted $G(\eta)\xi$")
    row_labels = (
        "OLD v8 model — ",
        "C27 (+ 3 auxiliary losses) — ",
    )

    for row in range(2):
        for column in range(2):
            axis = axes[row, column]
            (line,) = axis.plot(x, np.zeros_like(x), color=colors[row], linewidth=1.4)
            lines.append(line)
            axis.axhline(0.0, color="#777777", linewidth=0.6, alpha=0.65)
            axis.set_xlim(0.0, 2.0 * np.pi)
            axis.grid(alpha=0.18)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            axis.set_xlabel(r"$x$")
            axis.set_ylabel("amplitude")
            axis.set_title(row_labels[row] + column_labels[column], fontsize=10)
            amplitude_labels.append(
                axis.text(
                    0.02,
                    0.96,
                    "",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
                )
            )
            failure_labels.append(
                axis.text(
                    0.5,
                    0.08,
                    "",
                    transform=axis.transAxes,
                    ha="center",
                    va="bottom",
                    color="#b42318",
                    fontweight="bold",
                    fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "#b42318", "alpha": 0.9},
                )
            )

    terminal_time = float(new.times[-1])

    def update(frame_index: int) -> list[Any]:
        comparison_time = float(new.times[frame_index])
        old_frame = min(frame_index, last_old)
        displayed = (
            old_fields[0][old_frame],
            old_fields[1][old_frame],
            new_fields[0][frame_index],
            new_fields[1][frame_index],
        )
        for line, label, values in zip(lines, amplitude_labels, displayed, strict=True):
            line.set_ydata(values)
            label.set_text(rf"$\max|\cdot|={np.max(np.abs(values)):.3e}$")

        for column in range(2):
            old_limit = _positive_limit(
                old_cumulative_limits[column][old_frame], old_floors[column]
            )
            axes[0, column].set_ylim(-old_limit, old_limit)
            axes[1, column].set_ylim(-new_limits[column], new_limits[column])

        failure_message = (
            f"MODEL ROLLOUT NONFINITE — frozen at last finite t="
            f"{old.times[last_old]:.1f}"
            if frame_index > last_old
            else ""
        )
        for index, label in enumerate(failure_labels):
            label.set_text(failure_message if index < 2 else "")

        figure.suptitle(
            "Same Tanaka initial condition: old unstable model vs stable C27\n"
            f"{family}, simulation {simulation_id}, h={old.depths[old_index]:.4f}"
            f" • t={comparison_time:.1f}/{terminal_time:.0f}",
            fontsize=14,
            fontweight="bold",
        )
        return [*lines, *amplitude_labels, *failure_labels]

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{family}_simulation_{simulation_id}_v8_exploding_top_c27_stable_bottom"
    gif_path = output_dir / f"{stem}.gif"
    animation = FuncAnimation(
        figure,
        update,
        frames=indices,
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )
    animation.save(gif_path, writer=PillowWriter(fps=fps), dpi=dpi)

    keyframe_indices = [0, max(last_old - 5, 0), last_old, new.times.size - 1]
    keyframe_paths: list[Path] = []
    for label, index in zip(
        ("initial", "pre_failure", "last_finite", "terminal"),
        keyframe_indices,
        strict=True,
    ):
        update(index)
        keyframe_path = output_dir / f"{stem}_{label}.png"
        figure.savefig(keyframe_path, dpi=dpi, bbox_inches="tight")
        keyframe_paths.append(keyframe_path)
    plt.close(figure)

    return {
        "simulation_id": simulation_id,
        "c27_finite_to_terminal": True,
        "depth": float(old.depths[old_index]),
        "family": family,
        "fps": fps,
        "frame_count": int(indices.size),
        "frame_zero_eta_exact": True,
        "frame_zero_xi_exact": True,
        "gif": {
            "bytes": gif_path.stat().st_size,
            "path": str(gif_path.resolve()),
        },
        "keyframes": [
            {
                "bytes": path.stat().st_size,
                "path": str(path.resolve()),
            }
            for path in keyframe_paths
        ],
        "stable_terminal_time": terminal_time,
        "v8_first_nonfinite_time": float(old.times[last_old + 1]),
        "v8_last_finite_time": float(old.times[last_old]),
    }


def _parse_simulation(value: str) -> tuple[str, int]:
    family, separator, simulation_id = value.partition(":")
    if not separator or family not in {"tanaka_g0", "tanaka_g1"}:
        raise argparse.ArgumentTypeError(
            "simulation must have form tanaka_g0:ID or tanaka_g1:ID"
        )
    try:
        return family, int(simulation_id)
    except ValueError as error:
        raise argparse.ArgumentTypeError("simulation ID must be an integer") from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render matched v8-exploding/C27-stable learned-model GIFs."
    )
    parser.add_argument("--v8-dir", type=Path, default=DEFAULT_V8_DIR)
    parser.add_argument("--c27-dir", type=Path, default=DEFAULT_C27_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--simulation",
        action="append",
        type=_parse_simulation,
        dest="simulations",
        help="Matched simulation as FAMILY:ID; repeat for multiple simulations.",
    )
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    simulations = args.simulations or tuple(map(_parse_simulation, DEFAULT_SIMULATIONS))
    archives: dict[tuple[str, str], RolloutArchive] = {}

    def archive(version: str, family: str) -> RolloutArchive:
        key = (version, family)
        if key not in archives:
            base = args.v8_dir if version == "v8" else args.c27_dir / family
            archives[key] = _load_archive(base / f"{family}_trajs.npz")
        return archives[key]

    records = [
        _render_simulation(
            family,
            simulation_id,
            archive("v8", family),
            archive("c27", family),
            args.output_dir,
            args.frames,
            args.fps,
            args.dpi,
        )
        for family, simulation_id in simulations
    ]
    sources = {
        str(archive.path): {
            "bytes": archive.path.stat().st_size,
        }
        for archive in archives.values()
    }
    summary = {
        "simulations": records,
        "created_at": datetime.now().astimezone().isoformat(),
        "layout": {
            "bottom": "C27 learned model: relative L2 plus mode, tangent, and Hadamard terms",
            "columns": ["predicted surface elevation eta", "predicted G(eta)xi"],
            "top": "old v8 learned model rollout",
        },
        "notes": {
            "c27_guard": (
                "C27 archive enabled the soliton-selective spectral guard; the saved-frame "
                "audit found no Tanaka frame at its nominal trigger and maximum observed "
                "weight 3.15333e-4."
            ),
            "hou_li": (
                "Hou-Li filtering is not the method comparison in these GIFs and is not "
                "shown in the panel labels."
            ),
        },
        "schema": "v8_exploding_vs_c27_stable_gif_summary_v1",
        "sources": sources,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
