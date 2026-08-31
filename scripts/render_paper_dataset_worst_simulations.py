"""Rank and render diagnostic tails across completed paper-dataset chunks.

The rankings are descriptive and do not alter dataset acceptance. Each simulation
is scanned at every retained time, then ranked within its initial-condition
family by surface slope, high-band content of ``G(eta)xi``, and an
amplitude-thresholded count of spatial oscillations in ``G(eta)xi``.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Final

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


SUMMARY_PATTERN = re.compile(
    r"^paper_dataset_(?P<family>.+)_(?P<split>train|validation|test)"
    r"\.summary\.json$"
)
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
FINAL_PAPER_DATASET_SOURCE_COUNT = 26
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


def _count_thresholded_sign_changes(
    differences: np.ndarray,
    relative_threshold: float,
) -> np.ndarray:
    """Count cyclic sign changes after ignoring small differences in each row."""

    row_scales = np.max(np.abs(differences), axis=1, keepdims=True)
    thresholds = relative_threshold * row_scales
    signs = np.where(
        differences > thresholds,
        1,
        np.where(differences < -thresholds, -1, 0),
    ).astype(np.int8)
    nonzero = signs != 0
    nonzero_counts = np.count_nonzero(nonzero, axis=1)

    max_nonzero = int(np.max(nonzero_counts))
    if max_nonzero == 0:
        return np.zeros(differences.shape[0], dtype=np.int32)

    nonzero_signs = np.zeros(
        (differences.shape[0], max_nonzero),
        dtype=np.int8,
    )
    order = np.cumsum(nonzero, axis=1) - 1
    row_ids, column_ids = np.nonzero(nonzero)
    nonzero_signs[row_ids, order[row_ids, column_ids]] = signs[
        row_ids,
        column_ids,
    ]

    valid_next = np.arange(max_nonzero - 1)[None, :] < (nonzero_counts[:, None] - 1)
    adjacent_changes = (nonzero_signs[:, 1:] != nonzero_signs[:, :-1]) & valid_next
    final_indices = np.maximum(nonzero_counts - 1, 0)
    wrap_changes = (nonzero_counts > 1) & (
        nonzero_signs[:, 0]
        != nonzero_signs[np.arange(differences.shape[0]), final_indices]
    )
    return adjacent_changes.sum(axis=1).astype(np.int32) + wrap_changes.astype(np.int32)


@dataclass(frozen=True)
class TrajectoryIndex:
    """Location of one accepted trajectory in a dataset shard."""

    accepted_index: int
    trajectory_index: int
    simulation_id: int
    category: str
    shard_index: int
    first_shard_row: int
    row_count: int


@dataclass(frozen=True)
class DatasetSource:
    """One complete immutable dataset chunk."""

    root: Path
    family: str
    split: str
    summary_path: Path
    manifest_path: Path
    map_path: Path
    shard_paths: dict[int, Path]
    trajectories: tuple[TrajectoryIndex, ...]


@dataclass(frozen=True)
class CombinedSummaryBinding:
    """Source population recorded by one completed combined view."""

    path: Path
    source_summary_paths: tuple[Path, ...]
    expected_source_count: int
    expected_accepted_simulations: int
    expected_retained_rows: int


@dataclass(frozen=True)
class SimulationMetrics:
    """Whole-trajectory diagnostic maxima for one accepted simulation."""

    source_index: int
    accepted_index: int
    trajectory_index: int
    family: str
    split: str
    simulation_id: int
    category: str
    shard_index: int
    first_shard_row: int
    row_count: int
    depth: float
    all_frames_finite: bool
    constant_depth: bool
    ordered_time: bool
    minimum_water_column: float
    minimum_water_fraction: float
    maximum_eta_slope: float
    maximum_eta_slope_frame: int
    maximum_gxi_high_band_fraction: float
    maximum_gxi_high_band_fraction_frame: int
    maximum_thresholded_gxi_sign_changes: int
    maximum_thresholded_gxi_sign_changes_frame: int
    maximum_relative_stored_band_quadratic_energy_drift: float
    maximum_relative_stored_band_quadratic_energy_drift_frame: int


@dataclass(frozen=True)
class LoadedTrajectory:
    """Stored fields for a selected accepted trajectory."""

    eta: np.ndarray
    xi: np.ndarray
    gxi: np.ndarray
    depth: np.ndarray
    time: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument(
        "--source",
        action="append",
        type=Path,
        help=(
            "Completed dataset chunk root. Repeat for development scans; final "
            "dataset review should use --combined-summary."
        ),
    )
    sources.add_argument(
        "--combined-summary",
        type=Path,
        help=(
            "Completed combined-view summary whose preflight record binds the "
            "exact canonical chunk population."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--block-rows", type=int, default=256)
    parser.add_argument("--top-count", type=int, default=6)
    parser.add_argument(
        "--require-final-paper-dataset",
        action="store_true",
        help=(
            "Require the exact four-family paper release population. Valid only "
            "with --combined-summary."
        ),
    )
    args = parser.parse_args()
    if min(args.workers, args.block_rows, args.top_count) < 1:
        parser.error("--workers, --block-rows, and --top-count must be positive")
    if args.require_final_paper_dataset and args.combined_summary is None:
        parser.error("--require-final-paper-dataset requires --combined-summary")
    return args


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _sequence(value: object, *, context: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be a JSON array")
    return value


def _nonnegative_integer(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a nonnegative integer")
    return value


def load_combined_summary_binding(path: Path) -> CombinedSummaryBinding:
    """Load the chunk population recorded by a completed combined view."""

    resolved = path.expanduser().resolve(strict=True)
    summary = read_json(resolved)
    if summary.get("status") != "complete":
        raise ValueError("combined summary is not complete")
    preflight = _mapping(summary.get("preflight"), context="combined preflight")
    chunks = _sequence(preflight.get("chunks"), context="combined preflight chunks")
    if not chunks:
        raise ValueError("combined preflight contains no chunks")

    summary_paths: list[Path] = []
    for chunk_index, raw_chunk in enumerate(chunks):
        chunk = _mapping(raw_chunk, context=f"combined preflight chunk {chunk_index}")
        raw_summary_path = chunk.get("summary_path")
        if not isinstance(raw_summary_path, str) or not raw_summary_path:
            raise ValueError(
                f"combined preflight chunk {chunk_index} has no summary path"
            )
        summary_path = Path(raw_summary_path).expanduser()
        if not summary_path.is_absolute():
            raise ValueError(
                f"combined preflight chunk {chunk_index} summary path is not absolute"
            )
        summary_path = summary_path.resolve(strict=True)
        summary_paths.append(summary_path)

    if len(set(summary_paths)) != len(summary_paths):
        raise ValueError("combined preflight repeats a chunk summary")
    expected_accepted_simulations = _nonnegative_integer(
        preflight.get("accepted_simulations_total"),
        context="combined preflight accepted_simulations_total",
    )
    expected_retained_rows = _nonnegative_integer(
        preflight.get("expected_rows"),
        context="combined preflight expected_rows",
    )
    return CombinedSummaryBinding(
        path=resolved,
        source_summary_paths=tuple(summary_paths),
        expected_source_count=len(summary_paths),
        expected_accepted_simulations=expected_accepted_simulations,
        expected_retained_rows=expected_retained_rows,
    )


def validate_bound_sources(
    binding: CombinedSummaryBinding,
    sources: Sequence[DatasetSource],
) -> None:
    """Require loaded sources to be exactly those named by the combined view."""

    if len(sources) != binding.expected_source_count:
        raise ValueError("scanned source count differs from combined preflight")
    observed_paths = tuple(source.summary_path.resolve() for source in sources)
    if observed_paths != binding.source_summary_paths:
        raise ValueError(
            "scanned source order or identity differs from combined preflight"
        )


def validate_scanned_population(
    binding: CombinedSummaryBinding,
    *,
    source_count: int,
    accepted_simulations: int,
    retained_rows: int,
) -> None:
    """Compare final scan totals with the combined-view plan."""

    observed = (source_count, accepted_simulations, retained_rows)
    expected = (
        binding.expected_source_count,
        binding.expected_accepted_simulations,
        binding.expected_retained_rows,
    )
    if observed != expected:
        raise ValueError(
            "scanned source/simulation/row counts differ from combined preflight: "
            f"observed={observed}, expected={expected}"
        )


def validate_final_paper_dataset_counts(
    *,
    source_count: int,
    family_split_accepted_simulations: Mapping[tuple[str, str], int],
    accepted_simulations: int,
    retained_rows: int,
) -> None:
    """Require the exact, independently recovered paper release population."""

    expected_family_split_counts = {
        (family, split): split_count
        for family in FAMILY_LABELS
        for split, split_count in FINAL_PAPER_DATASET_SPLIT_ACCEPTED_SIMULATIONS.items()
    }
    observed_family_split_counts = dict(family_split_accepted_simulations)
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


def validate_final_paper_dataset(
    sources: Sequence[DatasetSource],
    *,
    retained_rows: int,
) -> None:
    """Recover final contract counts from source trajectory maps."""

    family_split_counts: dict[tuple[str, str], int] = {}
    for source in sources:
        key = (source.family, source.split)
        family_split_counts[key] = family_split_counts.get(key, 0) + len(
            source.trajectories
        )
    validate_final_paper_dataset_counts(
        source_count=len(sources),
        family_split_accepted_simulations=family_split_counts,
        accepted_simulations=sum(family_split_counts.values()),
        retained_rows=retained_rows,
    )


@contextmanager
def atomic_output_directory(requested: Path) -> Iterator[tuple[Path, Path]]:
    """Yield an owned sibling staging directory, then atomically publish it."""

    final_path = requested.expanduser().resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists() or final_path.is_symlink():
        raise FileExistsError(f"output path already exists: {final_path}")
    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f".{final_path.name}.staging-",
            dir=final_path.parent,
        )
    )
    committed = False
    try:
        yield staging_path, final_path
        if final_path.exists() or final_path.is_symlink():
            raise FileExistsError(
                f"output path appeared during rendering: {final_path}"
            )
        staging_path.rename(final_path)
        committed = True
    finally:
        if not committed:
            shutil.rmtree(staging_path, ignore_errors=True)


def _write_diagnostic_summary(path: Path, record: Mapping[str, Any]) -> None:
    """Write only a complete top-level diagnostic summary."""

    if record.get("status") != "complete":
        raise ValueError("diagnostic summary status must be complete")
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _artifact_record(
    path: Path,
    *,
    staging_output_dir: Path,
    published_output_dir: Path,
) -> dict[str, object]:
    """Describe staged bytes at the path they will have after publication."""

    relative_path = path.resolve().relative_to(staging_output_dir.resolve())
    return {
        "path": str((published_output_dir / relative_path).resolve()),
        "bytes": path.stat().st_size,
    }


def _summary_identity(path: Path) -> tuple[str, str]:
    match = SUMMARY_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"unrecognized dataset summary name: {path}")
    return match.group("family"), match.group("split")


def _artifact_path(root: Path, value: object, *, context: str) -> Path:
    """Resolve one named source artifact without allowing root escape."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} path must be a nonempty string")
    path = (root / value).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"{context} path escapes its source root")
    return path


def _dataset_artifact(
    root: Path,
    record: Mapping[str, Any],
    *,
    context: str,
) -> Path:
    """Resolve one dataset-view artifact and check its recorded size."""

    path = _artifact_path(root, record.get("path"), context=context)
    expected_bytes = _nonnegative_integer(
        record.get("bytes"),
        context=f"{context} bytes",
    )
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{context} byte count differs from its source summary")
    return path


def _trajectory_indices(
    summary: Mapping[str, Any],
    map_path: Path,
) -> tuple[TrajectoryIndex, ...]:
    cell_codes = summary["run_spec"]["cell_codes"]
    if not isinstance(cell_codes, dict):
        raise TypeError("summary cell_codes must be a dictionary")
    categories = {int(code): str(name) for name, code in cell_codes.items()}

    with np.load(map_path, allow_pickle=False) as archive:
        accepted = np.asarray(archive["trajectory_accepted"], dtype=np.bool_)
        simulation_ids = np.asarray(
            archive["trajectory_simulation_id"],
            dtype=np.int64,
        )
        cell_ids = np.asarray(archive["trajectory_cell_id"], dtype=np.int32)
        first_rows = np.asarray(archive["trajectory_first_row"], dtype=np.int64)
        row_counts = np.asarray(archive["trajectory_row_count"], dtype=np.int32)
        row_trajectories = np.asarray(archive["trajectory_index"], dtype=np.int32)
        row_shards = np.asarray(archive["shard_index"], dtype=np.int32)
        shard_rows = np.asarray(archive["shard_row"], dtype=np.int64)

    records: list[TrajectoryIndex] = []
    for accepted_index, trajectory_index_value in enumerate(np.flatnonzero(accepted)):
        trajectory_index = int(trajectory_index_value)
        first = int(first_rows[trajectory_index])
        count = int(row_counts[trajectory_index])
        if count < 1:
            raise RuntimeError(f"accepted trajectory {trajectory_index} has no rows")
        positions = slice(first, first + count)
        if not np.all(row_trajectories[positions] == trajectory_index):
            raise RuntimeError("trajectory-map rows are not contiguous")
        shards = np.unique(row_shards[positions])
        if shards.size != 1:
            raise RuntimeError("one accepted trajectory crosses shard boundaries")
        local_rows = shard_rows[positions]
        if not np.array_equal(
            local_rows,
            np.arange(local_rows[0], local_rows[0] + count),
        ):
            raise RuntimeError("trajectory rows are not contiguous in its shard")
        cell_code = int(cell_ids[trajectory_index])
        records.append(
            TrajectoryIndex(
                accepted_index=accepted_index,
                trajectory_index=trajectory_index,
                simulation_id=int(simulation_ids[trajectory_index]),
                category=categories[cell_code],
                shard_index=int(shards[0]),
                first_shard_row=int(local_rows[0]),
                row_count=count,
            )
        )
    return tuple(records)


def load_source_summary(summary_path: Path) -> DatasetSource:
    """Load one completed chunk summary and its dataset view."""

    summary_path = summary_path.expanduser().resolve(strict=True)
    resolved = summary_path.parent
    family, split = _summary_identity(summary_path)
    summary = read_json(summary_path)
    if summary.get("schema") != "paper_dataset_quota_summary_v1":
        raise RuntimeError(f"source has an unknown summary schema: {summary_path}")
    if summary.get("status") != "complete":
        raise RuntimeError(f"source is not complete: {resolved}")
    raw_output_root = summary.get("output_root")
    if not isinstance(raw_output_root, str) or (
        Path(raw_output_root).expanduser().resolve() != resolved
    ):
        raise RuntimeError(f"source summary has the wrong output root: {summary_path}")

    view = _mapping(summary.get("dataset_view"), context="source dataset_view")
    manifest_record = _mapping(
        view.get("manifest"),
        context="source dataset_view manifest",
    )
    map_record = _mapping(
        view.get("trajectory_map"),
        context="source dataset_view trajectory map",
    )
    manifest_path = _dataset_artifact(
        resolved,
        manifest_record,
        context="source dataset manifest",
    )
    map_path = _dataset_artifact(
        resolved,
        map_record,
        context="source trajectory map",
    )
    manifest = read_json(manifest_path)
    manifest_map_path = _artifact_path(
        resolved,
        manifest.get("trajectory_map_npz"),
        context="dataset manifest trajectory map",
    )
    if manifest_map_path != map_path:
        raise RuntimeError("dataset manifest names a different trajectory map")

    raw_shards = _sequence(
        manifest.get("dataset_shards"),
        context="dataset manifest shards",
    )
    shard_paths: dict[int, Path] = {}
    batch_indices: set[int] = set()
    for shard_index, raw_record in enumerate(raw_shards):
        record = _mapping(
            raw_record,
            context=f"dataset manifest shard {shard_index}",
        )
        batch_index = _nonnegative_integer(
            record.get("batch_index"),
            context=f"dataset manifest shard {shard_index} batch_index",
        )
        if batch_index in batch_indices:
            raise RuntimeError("dataset manifest repeats a shard batch index")
        batch_indices.add(batch_index)
        shard_path = _artifact_path(
            resolved,
            record.get("path"),
            context=f"dataset manifest shard {shard_index}",
        )
        # ``trajectory_map.shard_index`` is the dense ordinal in this list;
        # ``batch_index`` is source-batch metadata and may contain gaps.
        shard_paths[shard_index] = shard_path

    trajectories = _trajectory_indices(summary, map_path)
    if len(trajectories) != int(manifest["n_accepted_trajectories"]):
        raise RuntimeError(f"accepted trajectory count mismatch in {resolved}")
    if sum(value.row_count for value in trajectories) != int(
        manifest["n_accepted_rows"]
    ):
        raise RuntimeError(f"accepted row count mismatch in {resolved}")
    return DatasetSource(
        root=resolved,
        family=family,
        split=split,
        summary_path=summary_path,
        manifest_path=manifest_path,
        map_path=map_path,
        shard_paths=shard_paths,
        trajectories=trajectories,
    )


def load_source(root: Path) -> DatasetSource:
    """Load the only completed dataset summary in a development source root."""

    resolved = root.expanduser().resolve(strict=True)
    summaries = tuple(resolved.glob("paper_dataset_*_*.summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(
            f"expected one dataset summary in {resolved}, found {len(summaries)}"
        )
    return load_source_summary(summaries[0])


def _argmax(values: np.ndarray) -> tuple[float, int]:
    index = int(np.argmax(values))
    return float(values[index]), index


def _audit_shard(
    source_index: int,
    family: str,
    split: str,
    shard_index: int,
    shard_path: Path,
    trajectories: tuple[TrajectoryIndex, ...],
    block_rows: int,
) -> tuple[SimulationMetrics, ...]:
    with np.load(shard_path, allow_pickle=False) as archive:
        eta = np.asarray(archive["eta"])
        xi = np.asarray(archive["xi"])
        gxi = np.asarray(archive["gxi"])
        depth = np.asarray(archive["depth"], dtype=np.float64)
        time = np.asarray(archive["time"], dtype=np.float64)

    if eta.shape != xi.shape or eta.shape != gxi.shape or eta.ndim != 2:
        raise RuntimeError(f"field shape mismatch in {shard_path}")
    row_count, nx = eta.shape
    if depth.shape != (row_count,) or time.shape != (row_count,):
        raise RuntimeError(f"scalar row shape mismatch in {shard_path}")

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
        thresholded_sign_changes[rows] = _count_thresholded_sign_changes(
            difference,
            SIGN_DIFFERENCE_RELATIVE_THRESHOLD,
        )
        stored_band_quadratic_energy[rows] = (
            0.5
            * dx
            * (np.sum(xi_block * gxi_block, axis=1) + np.sum(eta_block**2, axis=1))
        )

    records: list[SimulationMetrics] = []
    for trajectory in trajectories:
        first = trajectory.first_shard_row
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
                trajectory_index=trajectory.trajectory_index,
                family=family,
                split=split,
                simulation_id=trajectory.simulation_id,
                category=trajectory.category,
                shard_index=shard_index,
                first_shard_row=first,
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


def _audit_task(
    values: tuple[
        int,
        str,
        str,
        int,
        Path,
        tuple[TrajectoryIndex, ...],
        int,
    ],
) -> tuple[SimulationMetrics, ...]:
    return _audit_shard(*values)


def _rank_fraction(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks / max(values.size - 1, 1)


def _descending_indices(values: np.ndarray, count: int) -> tuple[int, ...]:
    return tuple(
        map(
            int,
            np.argsort(-values, kind="stable")[: min(count, values.size)],
        )
    )


def _simulation_key(simulation: SimulationMetrics) -> tuple[int, int]:
    return simulation.source_index, simulation.accepted_index


def _load_selected(
    sources: tuple[DatasetSource, ...],
    simulations: Sequence[SimulationMetrics],
) -> dict[tuple[int, int], LoadedTrajectory]:
    by_shard: dict[tuple[int, int], list[SimulationMetrics]] = {}
    for simulation in simulations:
        by_shard.setdefault(
            (simulation.source_index, simulation.shard_index), []
        ).append(simulation)

    loaded: dict[tuple[int, int], LoadedTrajectory] = {}
    for (source_index, shard_index), selected in sorted(by_shard.items()):
        source = sources[source_index]
        with np.load(source.shard_paths[shard_index], allow_pickle=False) as archive:
            arrays = {
                name: np.asarray(archive[name])
                for name in (*FIELD_NAMES, "depth", "time")
            }
        for simulation in selected:
            rows = slice(
                simulation.first_shard_row,
                simulation.first_shard_row + simulation.row_count,
            )
            loaded[_simulation_key(simulation)] = LoadedTrajectory(
                eta=np.asarray(arrays["eta"][rows], dtype=np.float64),
                xi=np.asarray(arrays["xi"][rows], dtype=np.float64),
                gxi=np.asarray(arrays["gxi"][rows], dtype=np.float64),
                depth=np.asarray(arrays["depth"][rows], dtype=np.float64),
                time=np.asarray(arrays["time"][rows], dtype=np.float64),
            )
    return loaded


def _frame_roles(
    trajectory: LoadedTrajectory,
    worst_frame: int,
) -> tuple[tuple[int, str], ...]:
    roles: dict[int, list[str]] = {}
    for frame, role in (
        (0, "initial"),
        (worst_frame, "worst"),
        (trajectory.time.size - 1, "terminal"),
    ):
        roles.setdefault(frame, []).append(role)
    return tuple((frame, "/".join(value)) for frame, value in sorted(roles.items()))


def _line_style(role: str) -> tuple[str, str, float]:
    if "worst" in role:
        return "#d97706", "-", 1.45
    if "terminal" in role:
        return "#2563eb", "--", 1.15
    return "#64748b", ":", 1.05


def _normalized_spectrum(field: np.ndarray) -> np.ndarray:
    amplitude = np.abs(np.fft.rfft(field - float(np.mean(field))))
    upper = min(DELIVERED_MODE + 1, amplitude.size)
    scale = float(np.max(amplitude[1:upper])) if upper > 1 else 0.0
    return amplitude / scale if scale > 0.0 else amplitude


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
        frame_roles = _frame_roles(trajectory, worst_frame)
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
            spectrum = _normalized_spectrum(trajectory.gxi[frame])
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


def animation_frame_indices(
    stored_frames: int,
    maximum_frames: int = GIF_MAXIMUM_FRAMES,
) -> np.ndarray:
    """Return inclusive, monotonically increasing stored-frame indices."""

    if stored_frames < 1 or maximum_frames < 1:
        raise ValueError("stored_frames and maximum_frames must be positive")
    count = min(stored_frames, maximum_frames)
    if count == 1:
        return np.asarray([0], dtype=np.int32)
    indices = np.rint(np.linspace(0, stored_frames - 1, count)).astype(np.int32)
    if np.unique(indices).size != count:
        raise RuntimeError("animation frame selection contains duplicates")
    return indices


def padded_animation_limits(values: np.ndarray) -> tuple[float, float]:
    """Return finite nondegenerate limits over every stored frame."""

    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("animation limits require finite nonempty values")
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    padding = (
        GIF_Y_LIMIT_PADDING_FRACTION * span
        if span > 0.0
        else GIF_Y_LIMIT_PADDING_FRACTION * max(abs(lower), 1.0)
    )
    return lower - padding, upper + padding


def animation_record(trajectory: LoadedTrajectory) -> dict[str, object]:
    """Describe the deterministic rank-one animation for one trajectory."""

    frame_indices = animation_frame_indices(int(trajectory.time.size))
    fps = (
        GIF_SHORT_FPS
        if frame_indices.size <= GIF_SHORT_FRAME_THRESHOLD
        else GIF_LONG_FPS
    )
    return {
        "stored_frames": int(trajectory.time.size),
        "frame_indices": frame_indices.tolist(),
        "first_time": float(trajectory.time[0]),
        "last_time": float(trajectory.time[-1]),
        "fps": fps,
        "dimensions_pixels": list(GIF_DIMENSIONS),
        "field_y_limits": {
            name: list(padded_animation_limits(field))
            for name, field in zip(
                FIELD_NAMES,
                (trajectory.eta, trajectory.xi, trajectory.gxi),
            )
        },
    }


def _draw_animation_frame(
    *,
    lines: Sequence[Any],
    fields: Sequence[np.ndarray],
    time_text: Any,
    time: np.ndarray,
    frame_index: int,
) -> tuple[Any, ...]:
    for line, field in zip(lines, fields):
        line.set_ydata(field[frame_index])
    time_text.set_text(rf"$t={time[frame_index]:.2f}$")
    return (*lines, time_text)


def _render_rank_one_gif(
    simulation: SimulationMetrics,
    trajectory: LoadedTrajectory,
    title: str,
    output_path: Path,
) -> tuple[Path, dict[str, object]]:
    """Render and decode-check one fixed-axis accepted-trajectory GIF."""

    record = animation_record(trajectory)
    frame_indices = np.asarray(record["frame_indices"], dtype=np.int32)
    fps = (
        GIF_SHORT_FPS
        if frame_indices.size <= GIF_SHORT_FRAME_THRESHOLD
        else GIF_LONG_FPS
    )
    fields = (trajectory.eta, trajectory.xi, trajectory.gxi)
    x = np.linspace(0.0, 2.0 * np.pi, trajectory.eta.shape[1], endpoint=False)
    figure, axes = plt.subplots(
        1,
        len(fields),
        figsize=(12.5, 3.6),
        constrained_layout=True,
    )
    lines = []
    try:
        limits = record["field_y_limits"]
        if not isinstance(limits, Mapping):
            raise TypeError("animation field limits are not a mapping")
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
            rf"($h={simulation.depth:.3g}$, simulation {simulation.simulation_id})",
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
            return _draw_animation_frame(
                lines=lines,
                fields=fields,
                time_text=time_text,
                time=trajectory.time,
                frame_index=int(frame_indices[animation_index]),
            )

        if frame_indices.size == 1:
            update(0)
            writer = PillowWriter(fps=fps)
            writer.setup(figure, output_path, dpi=100)
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
                output_path,
                writer=PillowWriter(fps=fps),
                dpi=100,
            )
    finally:
        plt.close(figure)

    with Image.open(output_path) as image:
        expected_duration = 250 if record["fps"] == GIF_SHORT_FPS else 80
        if (
            image.format != "GIF"
            or tuple(map(int, image.size)) != GIF_DIMENSIONS
            or int(getattr(image, "n_frames", 1)) != frame_indices.size
            or image.info.get("loop") != 0
        ):
            raise RuntimeError(f"rendered GIF contract differs: {output_path}")
        for index in range(frame_indices.size):
            image.seek(index)
            image.load()
            if image.info.get("duration") != expected_duration:
                raise RuntimeError(
                    f"rendered GIF frame duration differs: {output_path}"
                )
    return output_path, record


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        f"q{int(round(100.0 * quantile)):03d}": float(np.quantile(values, quantile))
        for quantile in QUANTILES
    }


def _simulation_record(
    simulation: SimulationMetrics,
    source: DatasetSource,
    combined_rank: float,
) -> dict[str, Any]:
    return {
        **asdict(simulation),
        "source_root": str(source.root),
        "combined_empirical_rank": combined_rank,
    }


def _family_rankings(
    simulations: tuple[SimulationMetrics, ...],
    top_count: int,
) -> tuple[
    dict[str, tuple[int, ...]],
    np.ndarray,
    dict[str, np.ndarray],
    tuple[int, ...],
]:
    combined_values = {
        "eta_slope": np.asarray(
            [simulation.maximum_eta_slope for simulation in simulations],
            dtype=np.float64,
        ),
        "gxi_high_band": np.asarray(
            [simulation.maximum_gxi_high_band_fraction for simulation in simulations],
            dtype=np.float64,
        ),
        "gxi_sign_changes": np.asarray(
            [
                simulation.maximum_thresholded_gxi_sign_changes
                for simulation in simulations
            ],
            dtype=np.float64,
        ),
    }
    rank_fractions = {
        name: _rank_fraction(value) for name, value in combined_values.items()
    }
    combined = np.mean(np.stack(tuple(rank_fractions.values())), axis=0)
    values = {
        **combined_values,
        "stored_band_quadratic_energy_drift": np.asarray(
            [
                simulation.maximum_relative_stored_band_quadratic_energy_drift
                for simulation in simulations
            ],
            dtype=np.float64,
        ),
    }
    rankings = {
        "combined": _descending_indices(combined, top_count),
        **{
            name: _descending_indices(value, top_count)
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
        dominant = int(np.argmax([rank_fractions[name][index] for name in rank_names]))
        combined_frames.append(int(getattr(simulations[index], frame_fields[dominant])))
    return rankings, combined, values, tuple(combined_frames)


def _render_to_directory(
    *,
    args: argparse.Namespace,
    binding: CombinedSummaryBinding | None,
    sources: tuple[DatasetSource, ...],
    output_dir: Path,
    published_output_dir: Path,
) -> tuple[int, int, tuple[Path, ...], Path]:
    """Audit dataset sources and render into one owned directory."""

    tasks = []
    for source_index, source in enumerate(sources):
        by_shard: dict[int, list[TrajectoryIndex]] = {}
        for trajectory in source.trajectories:
            by_shard.setdefault(trajectory.shard_index, []).append(trajectory)
        tasks.extend(
            (
                source_index,
                source.family,
                source.split,
                shard_index,
                source.shard_paths[shard_index],
                tuple(trajectories),
                args.block_rows,
            )
            for shard_index, trajectories in sorted(by_shard.items())
        )

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        simulation_groups = tuple(executor.map(_audit_task, tasks))
    simulations = tuple(
        simulation for group in simulation_groups for simulation in group
    )
    if len(simulations) != sum(len(source.trajectories) for source in sources):
        raise RuntimeError("whole-dataset audit lost trajectories")
    retained_rows = sum(simulation.row_count for simulation in simulations)
    if binding is not None:
        validate_scanned_population(
            binding,
            source_count=len(sources),
            accepted_simulations=len(simulations),
            retained_rows=retained_rows,
        )
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
        rankings, combined, values, combined_frames = _family_rankings(
            family_simulations,
            args.top_count,
        )
        ranking_data[family] = (rankings, combined, combined_frames)
        selected_indices = tuple(
            dict.fromkeys(index for ranking in rankings.values() for index in ranking)
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
                    _simulation_record(
                        family_simulations[index],
                        sources[family_simulations[index].source_index],
                        float(combined[index]),
                    )
                    for index in indices
                ]
                for name, indices in rankings.items()
            },
        }

    loaded = _load_selected(sources, all_selected)
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
        rankings, combined, combined_frames = ranking_data[family]
        family_label = FAMILY_LABELS.get(family, family)
        for ranking_name, (metric_label, frame_field) in plot_definitions.items():
            indices = rankings[ranking_name]
            selected = tuple(family_simulations[index] for index in indices)
            frames = (
                combined_frames
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
            gif_path, animation = _render_rank_one_gif(
                top_simulation,
                loaded[_simulation_key(top_simulation)],
                (f"{family_label}: rank-one accepted simulation by the {metric_label}"),
                output_dir / f"{family}_worst_{ranking_name}.gif",
            )
            figures.append(gif_path)
            animations[gif_path.name] = {
                "family": family,
                "ranking": ranking_name,
                "rank": 1,
                "source_index": top_simulation.source_index,
                "accepted_index": top_simulation.accepted_index,
                "trajectory_index": top_simulation.trajectory_index,
                "simulation_id": top_simulation.simulation_id,
                "category": top_simulation.category,
                "split": top_simulation.split,
                **animation,
            }

        top_index = rankings["combined"][0]
        top_simulation = family_simulations[top_index]
        overview_simulations.append(top_simulation)
        overview_frames.append(combined_frames[0])
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
        "dataset": (
            {
                "mode": "combined_summary",
                "combined_summary_path": str(binding.path),
                "expected_sources": binding.expected_source_count,
                "expected_accepted_simulations": binding.expected_accepted_simulations,
                "expected_retained_rows": binding.expected_retained_rows,
            }
            if binding is not None
            else {
                "mode": "explicit_sources_development_fallback",
                "combined_summary_path": None,
            }
        ),
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
                "summary_path": str(source.summary_path),
                "manifest_path": str(source.manifest_path),
                "trajectory_map_path": str(source.map_path),
            }
            for source in sources
        ],
        "families": family_results,
        "animations": animations,
        "artifacts": {
            path.name: _artifact_record(
                path,
                staging_output_dir=output_dir,
                published_output_dir=published_output_dir,
            )
            for path in figures
        },
    }
    summary_path = output_dir / "summary.json"
    _write_diagnostic_summary(summary_path, record)
    return len(simulations), retained_rows, tuple(figures), summary_path


def main() -> None:
    """Audit all sources and atomically publish diagnostic-tail figures."""

    args = parse_args()
    binding = (
        load_combined_summary_binding(args.combined_summary)
        if args.combined_summary is not None
        else None
    )
    sources = (
        tuple(
            load_source_summary(summary_path)
            for summary_path in binding.source_summary_paths
        )
        if binding is not None
        else tuple(
            load_source(source)
            for source in tuple(args.source if args.source is not None else ())
        )
    )
    retained_rows_from_maps = sum(
        trajectory.row_count for source in sources for trajectory in source.trajectories
    )
    if binding is not None:
        validate_bound_sources(binding, sources)
        validate_scanned_population(
            binding,
            source_count=len(sources),
            accepted_simulations=sum(len(source.trajectories) for source in sources),
            retained_rows=retained_rows_from_maps,
        )
    if args.require_final_paper_dataset:
        validate_final_paper_dataset(
            sources,
            retained_rows=retained_rows_from_maps,
        )

    with atomic_output_directory(args.output_dir) as (
        staging_output_dir,
        final_output_dir,
    ):
        accepted_simulations, retained_rows, figures, summary_path = (
            _render_to_directory(
                args=args,
                binding=binding,
                sources=sources,
                output_dir=staging_output_dir,
                published_output_dir=final_output_dir,
            )
        )
        figure_relative_paths = tuple(
            path.relative_to(staging_output_dir) for path in figures
        )
        summary_relative_path = summary_path.relative_to(staging_output_dir)

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


if __name__ == "__main__":
    main()
