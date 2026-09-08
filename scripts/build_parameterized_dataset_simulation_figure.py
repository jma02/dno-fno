"""Build deterministic family illustrations from the completed paper dataset.

The four profiles are accepted validation simulations, not manufactured states
or empirical medoids. For each family, this selects the lower-median simulation
ID in a fixed central parameter category and plots its first stored row.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Final

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
    load_dataset_groups,
    validate_final_paper_dataset,
)
from solver.gen_data.pipeline.types import DatasetSplit  # noqa: E402


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
CENTRAL_VALIDATION_CATEGORIES: Final = {
    "stokes": "finite_moderate",
    "tanaka": "main_m2_q1",
    "benjamin_feir": "n_c_09__delta_n_02",
    "jonswap_tma": "finite__gamma_3p3__right_0p5",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--require-final-paper-dataset",
        action="store_true",
        help="Require the original fixed family/split release counts.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=ROOT / "notes/figures/parameterized_dataset_simulation_examples",
    )
    args = parser.parse_args()

    dataset_path = args.dataset.expanduser().resolve(strict=True)
    sources = load_dataset_groups(dataset_path)
    retained_rows = sum(
        trajectory.row_count for source in sources for trajectory in source.trajectories
    )
    if args.require_final_paper_dataset:
        validate_final_paper_dataset(sources, retained_rows=retained_rows)
    arrays = {
        name: np.load(dataset_path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ("eta", "xi", "depth", "time", "x")
    }
    x = arrays["x"]
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("stored grid must contain finite coordinates")
    stored_nx = x.size
    length = float((x[1] - x[0]) * stored_nx)
    if length <= 0.0:
        raise ValueError("stored grid length must be positive")
    resolved_stem = args.output_stem.expanduser().resolve()
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
    figure = None
    try:
        staged_pdf = staging_root / final_pdf.name
        staged_png = staging_root / final_png.name
        staged_json = staging_root / final_json.name
        figure, axes = plt.subplots(
            4,
            2,
            figsize=(8.3, 8.8),
            sharex="col",
            constrained_layout=True,
        )
        colors = ("#3366a6", "#b54a3a", "#6b4c9a", "#2a8c6a")
        simulations: list[dict[str, object]] = []
        for row, (family, color) in enumerate(zip(FAMILY_ORDER, colors)):
            category = CENTRAL_VALIDATION_CATEGORIES[family]
            candidates = sorted(
                (
                    (source, trajectory)
                    for source in sources
                    if source.family == family
                    and source.split == DatasetSplit.VALIDATION.value
                    for trajectory in source.trajectories
                    if trajectory.category == category
                ),
                key=lambda pair: pair[1].simulation_id,
            )
            if not candidates:
                raise ValueError(
                    f"no validation simulations found for {family}/{category}"
                )
            simulation_ids = [trajectory.simulation_id for _, trajectory in candidates]
            if len(simulation_ids) != len(set(simulation_ids)):
                raise ValueError(f"duplicate simulation IDs in {family}/{category}")
            lower_median_index = (len(candidates) - 1) // 2
            source, trajectory = candidates[lower_median_index]
            row_index = trajectory.first_row
            eta = np.asarray(arrays["eta"][row_index], dtype=np.float64)
            xi = np.asarray(arrays["xi"][row_index], dtype=np.float64)
            depth = float(arrays["depth"][row_index])
            time = float(arrays["time"][row_index])
            if eta.shape != (stored_nx,) or xi.shape != eta.shape:
                raise ValueError("selected eta/xi shapes differ from the stored grid")
            if time != 0.0:
                raise ValueError("selected frame-zero row must have time zero")
            if not np.all(np.isfinite(eta)) or not np.all(np.isfinite(xi)):
                raise ValueError("profile fields must be finite")
            if not math.isfinite(depth) or depth <= 0.0:
                raise ValueError("profile depth must be finite and positive")
            x_over_length = np.arange(eta.size, dtype=np.float64) / eta.size
            eta_over_depth = eta / depth
            # All paper families use nondimensional gravity g = 1.
            xi_over_depth_speed = xi / (depth * math.sqrt(depth))
            axes[row, 0].plot(
                x_over_length,
                eta_over_depth,
                color=color,
                linewidth=1.05,
            )
            axes[row, 1].plot(
                x_over_length,
                xi_over_depth_speed,
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
            simulations.append(
                {
                    "family": source.family,
                    "split": source.split,
                    "category": trajectory.category,
                    "simulation_id": trajectory.simulation_id,
                    "time": time,
                    "depth": depth,
                    "gravity": 1.0,
                    "domain_length": length,
                    "stored_nx": stored_nx,
                    "selection": {
                        "rule": SELECTION_RULE,
                        "candidate_count": len(candidates),
                        "lower_median_index_zero_based": lower_median_index,
                    },
                    "dataset_row": row_index,
                    "dimensionless_variables": dict(DIMENSIONLESS_VARIABLES),
                }
            )
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
        staged_json.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "description": DESCRIPTION,
                    "dataset": str(dataset_path),
                    "simulations": simulations,
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
        if figure is not None:
            plt.close(figure)
        shutil.rmtree(staging_root, ignore_errors=True)

    print(
        json.dumps(
            {
                "status": "complete",
                "pdf": str(final_pdf),
                "png": str(final_png),
                "sidecar": str(final_json),
            },
            indent=2,
            sort_keys=True,
        )
    )
