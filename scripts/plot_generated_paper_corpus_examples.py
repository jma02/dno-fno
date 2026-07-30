"""Plot representative examples directly from completed paper-corpus shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs/paper_corpus_examples_20260728"
FIELD_NAMES = ("eta", "xi", "gxi")
FIELD_TITLES = (r"$\eta(x)$", r"$\xi(x)$", r"$G(\eta)\xi(x)$")


@dataclass(frozen=True)
class DatasetSource:
    root: Path
    manifest_name: str
    summary_name: str


@dataclass(frozen=True)
class StoredTrajectory:
    category: str
    case_id: int
    depth: float
    eta: np.ndarray
    xi: np.ndarray
    gxi: np.ndarray
    time: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot examples directly from completed paper-corpus shards."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG, PDF, and JSON outputs.",
    )
    parser.add_argument(
        "--render-gifs",
        action="store_true",
        help="Also render CPU-only GIFs for the three stored trajectories.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trajectory(source: DatasetSource, category: str) -> StoredTrajectory:
    manifest_path = source.root / source.manifest_name
    summary_path = source.root / source.summary_name
    manifest = read_json(manifest_path)
    summary = read_json(summary_path)
    cell_codes = summary["run_spec"]["cell_codes"]
    if category not in cell_codes:
        raise KeyError(f"{category!r} is absent from {summary_path}")
    cell_code = int(cell_codes[category])

    map_path = source.root / str(manifest["trajectory_map_npz"])
    with np.load(map_path, allow_pickle=False) as archive:
        trajectory_accepted = np.asarray(archive["trajectory_accepted"], dtype=np.bool_)
        trajectory_cell_id = np.asarray(archive["trajectory_cell_id"], dtype=np.int32)
        trajectory_case_id = np.asarray(archive["trajectory_case_id"], dtype=np.int64)
        row_trajectory_index = np.asarray(archive["trajectory_index"], dtype=np.int32)
        row_shard_index = np.asarray(archive["shard_index"], dtype=np.int32)
        row_shard_row = np.asarray(archive["shard_row"], dtype=np.int64)

    candidates = np.flatnonzero(
        trajectory_accepted & (trajectory_cell_id == cell_code)
    )
    if candidates.size == 0:
        raise RuntimeError(f"no accepted trajectory found for {category!r}")
    trajectory_index = int(candidates[0])
    row_positions = np.flatnonzero(row_trajectory_index == trajectory_index)
    if row_positions.size == 0:
        raise RuntimeError(f"accepted trajectory {trajectory_index} has no stored rows")

    shard_paths = {
        int(record["batch_index"]): source.root / str(record["path"])
        for record in manifest["dataset_shards"]
    }
    shard_cache: dict[int, dict[str, np.ndarray]] = {}

    def shard_arrays(shard_index: int) -> dict[str, np.ndarray]:
        if shard_index not in shard_cache:
            with np.load(shard_paths[shard_index], allow_pickle=False) as archive:
                shard_cache[shard_index] = {
                    name: np.asarray(archive[name]) for name in (*FIELD_NAMES, "depth", "time")
                }
        return shard_cache[shard_index]

    rows: dict[str, list[np.ndarray | float]] = {
        "eta": [],
        "xi": [],
        "gxi": [],
        "depth": [],
        "time": [],
    }
    for position in row_positions:
        shard_index = int(row_shard_index[position])
        shard_row = int(row_shard_row[position])
        arrays = shard_arrays(shard_index)
        for field in FIELD_NAMES:
            rows[field].append(np.asarray(arrays[field][shard_row], dtype=np.float64))
        rows["depth"].append(float(arrays["depth"][shard_row]))
        rows["time"].append(float(arrays["time"][shard_row]))

    depth = np.asarray(rows["depth"], dtype=np.float64)
    if not np.allclose(depth, depth[0], rtol=0.0, atol=0.0):
        raise RuntimeError(f"depth changes within stored trajectory {trajectory_index}")
    time = np.asarray(rows["time"], dtype=np.float64)
    if not np.all(np.diff(time) >= 0.0):
        raise RuntimeError(f"stored times are not ordered for trajectory {trajectory_index}")

    fields = {
        name: np.stack(rows[name]).astype(np.float64, copy=False)
        for name in FIELD_NAMES
    }
    if not all(np.isfinite(value).all() for value in (*fields.values(), depth, time)):
        raise RuntimeError(f"nonfinite stored value in trajectory {trajectory_index}")
    return StoredTrajectory(
        category=category,
        case_id=int(trajectory_case_id[trajectory_index]),
        depth=float(depth[0]),
        eta=fields["eta"],
        xi=fields["xi"],
        gxi=fields["gxi"],
        time=time,
    )


def selected_frame_indices(count: int) -> tuple[int, ...]:
    return tuple(dict.fromkeys((0, count // 2, count - 1)))


def plot_rows(
    trajectories: tuple[StoredTrajectory, ...],
    row_labels: tuple[str, ...],
    title: str,
    output_stem: Path,
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(
        len(trajectories),
        len(FIELD_NAMES),
        figsize=(14.0, 2.75 * len(trajectories)),
        squeeze=False,
        constrained_layout=True,
    )
    colors = ("#6b7280", "#2563eb", "#dc6b19")
    styles = (":", "--", "-")

    for row, (trajectory, row_label) in enumerate(zip(trajectories, row_labels)):
        field_values = (trajectory.eta, trajectory.xi, trajectory.gxi)
        x = np.linspace(0.0, 2.0 * np.pi, trajectory.eta.shape[1], endpoint=False)
        frame_indices = selected_frame_indices(trajectory.time.size)
        static_case = len(frame_indices) == 1
        for column, (field, field_title) in enumerate(
            zip(field_values, FIELD_TITLES)
        ):
            axis = axes[row, column]
            for line_index, frame_index in enumerate(frame_indices):
                axis.plot(
                    x,
                    field[frame_index],
                    color="#334155" if static_case else colors[line_index],
                    linestyle="-" if static_case else styles[line_index],
                    linewidth=1.25,
                    label=None
                    if static_case
                    else rf"$t={trajectory.time[frame_index]:.2f}$",
                )
            axis.grid(alpha=0.2, linewidth=0.5)
            axis.set_xlim(0.0, 2.0 * np.pi)
            axis.set_xticks((0.0, np.pi, 2.0 * np.pi))
            axis.set_xticklabels(("0", r"$\pi$", r"$2\pi$"))
            if row == 0:
                axis.set_title(field_title)
            if row == len(trajectories) - 1:
                axis.set_xlabel(r"$x$")
            if column == 0:
                axis.set_ylabel(
                    f"{row_label}\n"
                    rf"$h={trajectory.depth:.3g}$",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=15,
                )
            if column == len(FIELD_NAMES) - 1 and not static_case:
                axis.legend(loc="best", fontsize=8, frameon=False)

    figure.suptitle(title, fontsize=14)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=180)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def trajectory_record(
    trajectory: StoredTrajectory, label: str, source: DatasetSource
) -> dict[str, Any]:
    return {
        "label": label,
        "category": trajectory.category,
        "case_id": trajectory.case_id,
        "depth": trajectory.depth,
        "stored_row_count": int(trajectory.time.size),
        "displayed_times": [
            float(trajectory.time[index])
            for index in selected_frame_indices(trajectory.time.size)
        ],
        "source_manifest": str(source.root / source.manifest_name),
        "source_summary": str(source.root / source.summary_name),
    }


def padded_limits(values: np.ndarray) -> tuple[float, float]:
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    padding = 0.06 * span if span > 0.0 else max(abs(lower), 1.0) * 0.06
    return lower - padding, upper + padding


def animate_trajectory(
    trajectory: StoredTrajectory,
    label: str,
    output_path: Path,
    *,
    maximum_frames: int = 100,
) -> Path:
    frame_indices = np.unique(
        np.linspace(
            0,
            trajectory.time.size - 1,
            min(trajectory.time.size, maximum_frames),
            dtype=np.int32,
        )
    )
    fps = 4 if frame_indices.size <= 20 else 12
    fields = (trajectory.eta, trajectory.xi, trajectory.gxi)
    x = np.linspace(0.0, 2.0 * np.pi, trajectory.eta.shape[1], endpoint=False)
    figure, axes = plt.subplots(
        1,
        len(fields),
        figsize=(12.5, 3.6),
        constrained_layout=True,
    )
    lines = []
    for axis, field, field_title in zip(axes, fields, FIELD_TITLES):
        (line,) = axis.plot(x, field[frame_indices[0]], color="#2563eb", linewidth=1.4)
        axis.set_xlim(0.0, 2.0 * np.pi)
        axis.set_ylim(*padded_limits(field[frame_indices]))
        axis.set_xticks((0.0, np.pi, 2.0 * np.pi))
        axis.set_xticklabels(("0", r"$\pi$", r"$2\pi$"))
        axis.set_xlabel(r"$x$")
        axis.set_title(field_title)
        axis.grid(alpha=0.2, linewidth=0.5)
        lines.append(line)

    clean_label = " ".join(label.replace("\n", " ").split())
    figure.suptitle(
        f"{clean_label}   "
        rf"($h={trajectory.depth:.3g}$, case {trajectory.case_id})",
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
        time_text.set_text(rf"$t={trajectory.time[frame_index]:.2f}$")
        return (*lines, time_text)

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_indices.size,
        interval=1000.0 / fps,
        blit=False,
        repeat=True,
    )
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stokes_source = DatasetSource(
        root=ROOT / "outputs/paper_corpus/train/stokes/chunk_00000_02048",
        manifest_name="paper_corpus_stokes_train.dataset.json",
        summary_name="paper_corpus_stokes_train.summary.json",
    )
    gate_root = ROOT / "outputs/paper_corpus_all_category_gpu_pilot_20260727"
    trajectory_sources = {
        "tanaka": DatasetSource(
            root=gate_root / "tanaka",
            manifest_name="paper_corpus_tanaka_validation.dataset.json",
            summary_name="paper_corpus_tanaka_validation.summary.json",
        ),
        "benjamin_feir": DatasetSource(
            root=gate_root / "benjamin_feir",
            manifest_name="paper_corpus_benjamin_feir_validation.dataset.json",
            summary_name="paper_corpus_benjamin_feir_validation.summary.json",
        ),
        "jonswap_tma": DatasetSource(
            root=gate_root / "jonswap_tma",
            manifest_name="paper_corpus_jonswap_tma_validation.dataset.json",
            summary_name="paper_corpus_jonswap_tma_validation.summary.json",
        ),
    }

    stokes_categories = (
        "finite_low",
        "finite_moderate",
        "deep_low",
        "deep_moderate",
    )
    stokes_labels = (
        "finite depth,\nlow steepness",
        "finite depth,\nmoderate steepness",
        "deep water,\nlow steepness",
        "deep water,\nmoderate steepness",
    )
    stokes_trajectories = tuple(
        load_trajectory(stokes_source, category) for category in stokes_categories
    )
    stokes_png, stokes_pdf = plot_rows(
        stokes_trajectories,
        stokes_labels,
        "Examples from the final Stokes training split",
        output_dir / "final_stokes_examples",
    )

    selected_categories = {
        "tanaka": "main_m3_q1",
        "benjamin_feir": "n_c_20__delta_n_07",
        "jonswap_tma": "shallow__gamma_5__right_0p5",
    }
    trajectory_labels = (
        "Tanaka:\n3 crests, 1 right-moving",
        "Benjamin–Feir:\n"
        r"$n_c=20,\ \Delta n=7$",
        "JONSWAP/TMA:\n"
        r"shallow, $\gamma=5$, 50% right",
    )
    trajectory_order = ("tanaka", "benjamin_feir", "jonswap_tma")
    trajectories = tuple(
        load_trajectory(trajectory_sources[family], selected_categories[family])
        for family in trajectory_order
    )
    trajectories_png, trajectories_pdf = plot_rows(
        trajectories,
        trajectory_labels,
        "Accepted trajectories from the audited 104-category gate",
        output_dir / "accepted_trajectory_examples",
    )

    gif_paths: tuple[Path, ...] = ()
    if args.render_gifs:
        gif_paths = tuple(
            animate_trajectory(
                trajectory,
                label,
                output_dir / f"{family}_accepted_trajectory.gif",
            )
            for family, trajectory, label in zip(
                trajectory_order, trajectories, trajectory_labels
            )
        )

    figure_paths = (
        stokes_png,
        stokes_pdf,
        trajectories_png,
        trajectories_pdf,
        *gif_paths,
    )
    record = {
        "schema": "paper_corpus_stored_example_gallery_v1",
        "note": (
            "Stokes examples are from the final training split. Time-dependent "
            "examples are completed accepted trajectories from the non-corpus "
            "104-category gate and illustrate the same frozen numerical contract."
        ),
        "stokes": [
            trajectory_record(trajectory, label, stokes_source)
            for trajectory, label in zip(stokes_trajectories, stokes_labels)
        ],
        "trajectories": [
            trajectory_record(
                trajectory, label, trajectory_sources[family]
            )
            for trajectory, label, family in zip(
                trajectories, trajectory_labels, trajectory_order
            )
        ],
        "figures": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in figure_paths
        },
    }
    record_path = output_dir / "examples.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "figures": [str(path) for path in figure_paths],
                "record": str(record_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
