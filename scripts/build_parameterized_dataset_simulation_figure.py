"""Build deterministic family illustrations from the completed paper dataset.

The four profiles are accepted validation simulations, not manufactured states
or empirical medoids. For each family, this selects the lower-median simulation
ID in a fixed central parameter category and plots its first stored row.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Final

import numpy as np

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / f"dno-fno-matplotlib-{os.getuid()}"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_paper_dataset_worst_simulations import (  # noqa: E402
    CombinedSummaryBinding,
    DatasetSource,
    TrajectoryIndex,
    load_combined_summary_binding,
    load_source_summary,
    read_json,
    validate_bound_sources,
    validate_final_paper_dataset,
    validate_scanned_population,
)
from solver.gen_data.pipeline.simulation_allocation import (  # noqa: E402
    DatasetSplit,
    PhysicalFamilyId,
)


DESCRIPTION = (
    "Deterministic accepted validation illustrations; these are lower-median "
    "simulation IDs in fixed central parameter groups, not medoids."
)
SELECTION_RULE = "sort accepted validation simulation IDs; choose lower median"
DIMENSIONLESS_VARIABLES: Final = {
    "horizontal": "x/L",
    "surface_elevation": "eta/h",
    "surface_potential": "xi/(h*sqrt(g*h))",
}
FAMILY_ORDER: Final = (
    "stokes",
    "tanaka",
    "benjamin_feir",
    "jonswap_tma",
)
FAMILY_LABELS: Final = {
    "stokes": "Finite-depth Stokes",
    "tanaka": "Tanaka",
    "benjamin_feir": "Benjamin--Feir",
    "jonswap_tma": "JONSWAP/TMA",
}
FAMILY_IDS: Final = {
    "stokes": PhysicalFamilyId.STOKES,
    "tanaka": PhysicalFamilyId.TANAKA,
    "benjamin_feir": PhysicalFamilyId.BENJAMIN_FEIR,
    "jonswap_tma": PhysicalFamilyId.JONSWAP_TMA,
}
CENTRAL_VALIDATION_CATEGORIES: Final = {
    "stokes": "finite_moderate",
    "tanaka": "main_m2_q1",
    "benjamin_feir": "n_c_09__delta_n_02",
    "jonswap_tma": "finite__gamma_3p3__right_0p5",
}


@dataclass(frozen=True)
class SourceDetails:
    """Numerical settings needed to interpret one dataset source."""

    source: DatasetSource
    length: float
    gravity: float
    stored_nx: int


@dataclass(frozen=True)
class SelectedTrajectory:
    """Deterministically selected accepted validation trajectory."""

    details: SourceDetails
    trajectory: TrajectoryIndex
    candidate_count: int
    lower_median_index: int


@dataclass(frozen=True)
class DimensionlessProfile:
    """Dimensionless variables shown in one figure row."""

    x_over_length: np.ndarray
    eta_over_depth: np.ndarray
    xi_over_depth_speed: np.ndarray


@dataclass(frozen=True)
class Illustration:
    """One selected first-row state and its dimensionless representation."""

    selected: SelectedTrajectory
    eta: np.ndarray
    xi: np.ndarray
    depth: float
    time: float
    frame_index: int
    profile: DimensionlessProfile


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _nonnegative_integer(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a nonnegative integer")
    return value


def _positive_float(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{context} must be finite and positive")
    return result


def _source_numerical_scales(
    summary: Mapping[str, Any],
    *,
    family: str,
) -> tuple[float, float, int]:
    run_spec = _mapping(summary.get("run_spec"), context="source run_spec")
    configuration = _mapping(
        run_spec.get("configuration"),
        context="source configuration",
    )
    if family == "stokes":
        numerical = _mapping(
            configuration.get("contract"),
            context="Stokes numerical contract",
        )
    else:
        execution = _mapping(
            configuration.get("trajectory_execution"),
            context=f"{family} trajectory execution",
        )
        numerical = _mapping(
            execution.get("numerical"),
            context=f"{family} numerical contract",
        )
    view = _mapping(summary.get("dataset_view"), context="source dataset_view")
    grid = _mapping(view.get("grid"), context="source stored grid")
    length = _positive_float(numerical.get("length"), context="domain length")
    gravity = _positive_float(numerical.get("gravity"), context="gravity")
    stored_nx = _nonnegative_integer(grid.get("nx"), context="stored grid nx")
    if stored_nx < 2:
        raise ValueError("stored grid nx must be at least two")
    if not math.isclose(
        length,
        _positive_float(grid.get("length"), context="stored grid length"),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("stored-grid length differs from numerical contract")
    return length, gravity, stored_nx


def _validate_source_map(
    source: DatasetSource,
) -> None:
    with np.load(source.map_path, allow_pickle=False) as archive:
        accepted = np.asarray(archive["trajectory_accepted"], dtype=np.bool_)
        family_ids = np.asarray(archive["trajectory_family_id"], dtype=np.int64)
        dataset_splits = np.asarray(archive["trajectory_dataset_split"])
    if accepted.ndim != 1 or any(
        values.shape != accepted.shape for values in (family_ids, dataset_splits)
    ):
        raise ValueError("source trajectory arrays have inconsistent shapes")
    if not np.all(family_ids == int(FAMILY_IDS[source.family])):
        raise ValueError(f"{source.family} trajectory map has a wrong family ID")
    if not np.all(dataset_splits == source.split):
        raise ValueError(f"{source.family} trajectory map has a wrong dataset split")


def load_source_details(source: DatasetSource) -> SourceDetails:
    summary = read_json(source.summary_path)
    run_spec = _mapping(summary.get("run_spec"), context="source run_spec")
    if run_spec.get("family_name") != source.family:
        raise ValueError("source run-spec family differs from its summary name")
    if run_spec.get("dataset_split") != source.split:
        raise ValueError("source run-spec split differs from its summary name")
    _validate_source_map(source)
    length, gravity, stored_nx = _source_numerical_scales(
        summary,
        family=source.family,
    )
    return SourceDetails(
        source=source,
        length=length,
        gravity=gravity,
        stored_nx=stored_nx,
    )


def select_validation_trajectories(
    details: Sequence[SourceDetails],
) -> tuple[SelectedTrajectory, ...]:
    """Choose the lower-median simulation ID in each validation category."""

    selected: list[SelectedTrajectory] = []
    for family in FAMILY_ORDER:
        category = CENTRAL_VALIDATION_CATEGORIES[family]
        candidates = sorted(
            (
                (item, trajectory)
                for item in details
                if item.source.family == family
                and item.source.split == DatasetSplit.VALIDATION.value
                for trajectory in item.source.trajectories
                if trajectory.category == category
            ),
            key=lambda pair: pair[1].simulation_id,
        )
        if not candidates:
            raise ValueError(f"no validation simulations found for {family}/{category}")
        simulation_ids = [trajectory.simulation_id for _, trajectory in candidates]
        if len(simulation_ids) != len(set(simulation_ids)):
            raise ValueError(f"duplicate simulation IDs in {family}/{category}")
        lower_median_index = (len(candidates) - 1) // 2
        source_details, trajectory = candidates[lower_median_index]
        selected.append(
            SelectedTrajectory(
                details=source_details,
                trajectory=trajectory,
                candidate_count=len(candidates),
                lower_median_index=lower_median_index,
            )
        )
    return tuple(selected)


def dimensionless_profile(
    eta: np.ndarray,
    xi: np.ndarray,
    *,
    depth: float,
    gravity: float,
) -> DimensionlessProfile:
    eta_values = np.asarray(eta, dtype=np.float64)
    xi_values = np.asarray(xi, dtype=np.float64)
    if eta_values.ndim != 1 or xi_values.shape != eta_values.shape:
        raise ValueError("eta and xi must be one-dimensional arrays of equal shape")
    if eta_values.size < 2:
        raise ValueError("a profile must contain at least two grid points")
    if not np.all(np.isfinite(eta_values)) or not np.all(np.isfinite(xi_values)):
        raise ValueError("profile fields must be finite")
    positive_depth = _positive_float(depth, context="profile depth")
    positive_gravity = _positive_float(gravity, context="profile gravity")
    return DimensionlessProfile(
        x_over_length=np.arange(eta_values.size, dtype=np.float64) / eta_values.size,
        eta_over_depth=eta_values / positive_depth,
        xi_over_depth_speed=xi_values
        / (positive_depth * math.sqrt(positive_gravity * positive_depth)),
    )


def _selected_first_row(selected: SelectedTrajectory) -> tuple[int, int]:
    trajectory = selected.trajectory
    with np.load(selected.details.source.map_path, allow_pickle=False) as archive:
        first_rows = np.asarray(archive["trajectory_first_row"], dtype=np.int64)
        row_counts = np.asarray(archive["trajectory_row_count"], dtype=np.int64)
        row_trajectories = np.asarray(archive["trajectory_index"], dtype=np.int64)
        row_shards = np.asarray(archive["shard_index"], dtype=np.int64)
        shard_rows = np.asarray(archive["shard_row"], dtype=np.int64)
        frame_indices = np.asarray(archive["frame_index"], dtype=np.int64)
    index = trajectory.trajectory_index
    if not 0 <= index < first_rows.size:
        raise ValueError("selected trajectory index is outside its source map")
    first = int(first_rows[index])
    if first < 0 or int(row_counts[index]) != trajectory.row_count:
        raise ValueError("selected trajectory ownership differs from its source map")
    if not 0 <= first < row_trajectories.size:
        raise ValueError("selected first row is outside its source map")
    observed = (
        int(row_trajectories[first]),
        int(row_shards[first]),
        int(shard_rows[first]),
    )
    expected = (
        index,
        trajectory.shard_index,
        trajectory.first_shard_row,
    )
    if observed != expected:
        raise ValueError("selected first-row ownership differs from its source map")
    frame_index = int(frame_indices[first])
    if frame_index != 0:
        raise ValueError("selected trajectory's first row is not frame zero")
    return trajectory.first_shard_row, frame_index


def load_illustration(selected: SelectedTrajectory) -> Illustration:
    details = selected.details
    row, frame_index = _selected_first_row(selected)
    shard_path = details.source.shard_paths[selected.trajectory.shard_index]
    with np.load(shard_path, allow_pickle=False) as archive:
        missing = {"eta", "xi", "depth", "time"}.difference(archive.files)
        if missing:
            raise ValueError(f"selected shard lacks fields: {sorted(missing)}")
        eta_all = np.asarray(archive["eta"])
        xi_all = np.asarray(archive["xi"])
        depths = np.asarray(archive["depth"], dtype=np.float64)
        times = np.asarray(archive["time"], dtype=np.float64)
    if eta_all.ndim != 2 or xi_all.shape != eta_all.shape:
        raise ValueError("selected shard eta/xi shapes differ")
    if eta_all.shape[1] != details.stored_nx:
        raise ValueError("selected shard width differs from the stored grid")
    if depths.shape != (eta_all.shape[0],) or times.shape != (eta_all.shape[0],):
        raise ValueError("selected shard scalar-row shapes differ")
    if not 0 <= row < eta_all.shape[0]:
        raise ValueError("selected shard row is outside the shard")
    eta = np.asarray(eta_all[row], dtype=np.float64)
    xi = np.asarray(xi_all[row], dtype=np.float64)
    depth = float(depths[row])
    time = float(times[row])
    if time != 0.0:
        raise ValueError("selected frame-zero row must have time zero")
    return Illustration(
        selected=selected,
        eta=eta,
        xi=xi,
        depth=depth,
        time=time,
        frame_index=frame_index,
        profile=dimensionless_profile(
            eta,
            xi,
            depth=depth,
            gravity=details.gravity,
        ),
    )


def _simulation_record(illustration: Illustration) -> dict[str, object]:
    selected = illustration.selected
    details = selected.details
    source = details.source
    trajectory = selected.trajectory
    return {
        "family": source.family,
        "split": source.split,
        "category": trajectory.category,
        "simulation_id": trajectory.simulation_id,
        "time": illustration.time,
        "depth": illustration.depth,
        "gravity": details.gravity,
        "domain_length": details.length,
        "stored_nx": details.stored_nx,
        "selection": {
            "rule": SELECTION_RULE,
            "candidate_count": selected.candidate_count,
            "lower_median_index_zero_based": selected.lower_median_index,
        },
        "owned_row": {
            "trajectory_index": trajectory.trajectory_index,
            "frame_index": illustration.frame_index,
            "shard_index": trajectory.shard_index,
            "shard_row": trajectory.first_shard_row,
        },
        "dimensionless_variables": dict(DIMENSIONLESS_VARIABLES),
        "source_summary": str(source.summary_path),
        "selected_shard": str(source.shard_paths[trajectory.shard_index]),
    }


def publish_outputs(
    illustrations: Sequence[Illustration],
    *,
    output_stem: Path,
    binding: CombinedSummaryBinding,
) -> tuple[Path, Path, Path]:
    """Render to temporary files, then replace the three outputs together."""

    resolved_stem = output_stem.expanduser().resolve()
    resolved_stem.parent.mkdir(parents=True, exist_ok=True)
    final_pdf = resolved_stem.with_suffix(".pdf")
    final_png = resolved_stem.with_suffix(".png")
    final_json = resolved_stem.with_suffix(".json")
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{resolved_stem.name}.staging-",
            dir=resolved_stem.parent,
        )
    )
    try:
        staged_pdf = staging_root / final_pdf.name
        staged_png = staging_root / final_png.name
        staged_json = staging_root / final_json.name
        if tuple(item.selected.details.source.family for item in illustrations) != (
            FAMILY_ORDER
        ):
            raise ValueError("illustrations must follow the four-family order")
        figure, axes = plt.subplots(
            4,
            2,
            figsize=(8.3, 8.8),
            sharex="col",
            constrained_layout=True,
        )
        colors = ("#3366a6", "#b54a3a", "#6b4c9a", "#2a8c6a")
        for row, (illustration, color) in enumerate(zip(illustrations, colors)):
            source = illustration.selected.details.source
            trajectory = illustration.selected.trajectory
            profile = illustration.profile
            axes[row, 0].plot(
                profile.x_over_length,
                profile.eta_over_depth,
                color=color,
                linewidth=1.05,
            )
            axes[row, 1].plot(
                profile.x_over_length,
                profile.xi_over_depth_speed,
                color=color,
                linewidth=1.05,
            )
            axes[row, 0].set_ylabel(r"$\eta/h$")
            axes[row, 1].set_ylabel(r"$\xi/(h\sqrt{gh})$")
            axes[row, 0].text(
                0.02,
                0.92,
                (
                    f"{FAMILY_LABELS[source.family]}\n"
                    f"{trajectory.category}; simulation {trajectory.simulation_id}"
                ),
                transform=axes[row, 0].transAxes,
                va="top",
                fontsize=8.1,
            )
            for axis in axes[row]:
                axis.grid(alpha=0.22, linewidth=0.5)
        axes[0, 0].set_title("surface elevation")
        axes[0, 1].set_title("surface potential")
        axes[-1, 0].set_xlabel(r"$x/L$")
        axes[-1, 1].set_xlabel(r"$x/L$")
        figure.suptitle("Deterministic validation illustrations", fontsize=10)
        figure.savefig(
            staged_pdf,
            bbox_inches="tight",
            metadata={
                "Creator": "build_parameterized_dataset_simulation_figure.py",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        figure.savefig(
            staged_png,
            dpi=220,
            bbox_inches="tight",
            metadata={"Software": "build_parameterized_dataset_simulation_figure.py"},
        )
        plt.close(figure)
        staged_json.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "description": DESCRIPTION,
                    "combined_summary": str(binding.path),
                    "simulations": [_simulation_record(item) for item in illustrations],
                    "artifacts": {
                        "pdf": {
                            "path": str(final_pdf),
                            "bytes": staged_pdf.stat().st_size,
                        },
                        "png": {
                            "path": str(final_png),
                            "bytes": staged_png.stat().st_size,
                        },
                    },
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staged_pdf, final_pdf)
        os.replace(staged_png, final_png)
        os.replace(staged_json, final_json)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return final_pdf, final_png, final_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-summary", type=Path, required=True)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=ROOT / "notes/figures/parameterized_dataset_simulation_examples",
    )
    args = parser.parse_args()

    binding = load_combined_summary_binding(args.combined_summary)
    sources = tuple(load_source_summary(path) for path in binding.source_summary_paths)
    validate_bound_sources(binding, sources)
    accepted_simulations = sum(len(source.trajectories) for source in sources)
    retained_rows = sum(
        trajectory.row_count for source in sources for trajectory in source.trajectories
    )
    validate_scanned_population(
        binding,
        source_count=len(sources),
        accepted_simulations=accepted_simulations,
        retained_rows=retained_rows,
    )
    validate_final_paper_dataset(sources, retained_rows=retained_rows)
    details = tuple(load_source_details(source) for source in sources)
    selected = select_validation_trajectories(details)
    illustrations = tuple(load_illustration(item) for item in selected)
    pdf_path, png_path, json_path = publish_outputs(
        illustrations,
        output_stem=args.output_stem,
        binding=binding,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "pdf": str(pdf_path),
                "png": str(png_path),
                "sidecar": str(json_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
