"""Plot the last common finite states of rejected refinement-panel cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FloatArray: TypeAlias = NDArray[np.float64]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL_DIR = ROOT / "outputs/full_horizon_refinement_panel_20260725"
CASE_IDS = ("tanaka_steep_upper_seam", "bf_jcp09_canonical")
CASE_LABELS = {
    "tanaka_steep_upper_seam": "Steep Tanaka validation case",
    "bf_jcp09_canonical": "Equation-(33) Benjamin--Feir validation case",
}


@dataclass(frozen=True)
class CaseArm:
    """The saved history and first failed step of one time-step arm."""

    dt: float
    times: FloatArray
    depth: float
    eta: FloatArray
    xi: FloatArray
    gxi: FloatArray
    first_failed_step_time: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=DEFAULT_PANEL_DIR / "rejected_last_finite_profiles",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_case_arms(path: Path) -> dict[str, CaseArm]:
    """Load the selected case histories from one refinement artifact."""

    with np.load(path, allow_pickle=False) as archive:
        case_ids = tuple(map(str, archive["case_id"]))
        positions = [case_ids.index(case_id) for case_id in CASE_IDS]
        times = np.asarray(archive["times"], dtype=np.float64)
        depths = np.asarray(archive["depth"], dtype=np.float64)[positions]
        eta = np.asarray(archive["eta"][:, positions, :], dtype=np.float64)
        xi = np.asarray(archive["xi"][:, positions, :], dtype=np.float64)
        gxi = np.asarray(archive["gxi"][:, positions, :], dtype=np.float64)
        first_failed_steps = np.asarray(
            archive["gl2_first_failed_step"], dtype=np.int64
        )[positions]
        step_times = np.asarray(archive["gl2_step_times"], dtype=np.float64)
        dt = float(archive["dt"])

    return {
        case_id: CaseArm(
            dt=dt,
            times=times,
            depth=float(depths[local_index]),
            eta=eta[:, local_index],
            xi=xi[:, local_index],
            gxi=gxi[:, local_index],
            first_failed_step_time=float(step_times[first_failed_steps[local_index]]),
        )
        for local_index, case_id in enumerate(CASE_IDS)
    }


def finite_saved_mask(arm: CaseArm) -> NDArray[np.bool_]:
    """Return the saved times at which all three delivered fields are finite."""

    return np.asarray(
        np.all(np.isfinite(arm.eta), axis=1)
        & np.all(np.isfinite(arm.xi), axis=1)
        & np.all(np.isfinite(arm.gxi), axis=1),
        dtype=np.bool_,
    )


def last_common_finite_indices(
    coarse: CaseArm,
    fine: CaseArm,
) -> tuple[int, int]:
    """Return indices of the last common saved time finite in both arms."""

    common_times, coarse_indices, fine_indices = np.intersect1d(
        coarse.times,
        fine.times,
        assume_unique=True,
        return_indices=True,
    )
    common_finite = (
        finite_saved_mask(coarse)[coarse_indices]
        & finite_saved_mask(fine)[fine_indices]
    )
    finite_positions = np.flatnonzero(common_finite)
    if finite_positions.size == 0:
        raise ValueError("the refinement arms have no common finite saved state")
    last_position = int(finite_positions[-1])
    if not np.isfinite(common_times[last_position]):
        raise ValueError("the last common saved time is nonfinite")
    return int(coarse_indices[last_position]), int(fine_indices[last_position])


def normalized_l2_difference(coarse: FloatArray, fine: FloatArray) -> float:
    """Return the relative discrete L2 difference of two one-dimensional fields."""

    return float(
        np.linalg.norm(fine - coarse)
        / max(float(np.linalg.norm(fine)), np.finfo(np.float64).tiny)
    )


def dimensionless_fields(
    arm: CaseArm,
    index: int,
    gravity: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return the three dimensionless delivered fields at one saved time."""

    velocity_scale = np.sqrt(gravity * arm.depth)
    return (
        np.asarray(arm.eta[index] / arm.depth, dtype=np.float64),
        np.asarray(
            arm.xi[index] / (arm.depth * velocity_scale),
            dtype=np.float64,
        ),
        np.asarray(arm.gxi[index] / velocity_scale, dtype=np.float64),
    )


def make_figure(
    *,
    x_over_length: FloatArray,
    coarse_arms: dict[str, CaseArm],
    fine_arms: dict[str, CaseArm],
    gravity: float,
    output_stem: Path,
) -> dict[str, object]:
    """Create the physical-field panel and return its numerical metadata."""

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    figure, axes = plt.subplots(
        len(CASE_IDS),
        3,
        figsize=(11.5, 5.3),
        sharex=True,
        constrained_layout=True,
    )
    column_titles = (
        r"$\eta/h$",
        r"$\xi/(h\sqrt{gh})$",
        r"$G_{\mathrm{ref}}(\eta;h)\xi/\sqrt{gh}$",
    )
    records: list[dict[str, object]] = []

    for row, case_id in enumerate(CASE_IDS):
        coarse = coarse_arms[case_id]
        fine = fine_arms[case_id]
        coarse_index, fine_index = last_common_finite_indices(coarse, fine)
        last_time = float(coarse.times[coarse_index])
        if not np.isclose(last_time, fine.times[fine_index], rtol=0.0, atol=1e-12):
            raise ValueError(f"last common times disagree for {case_id}")

        initial_fields = dimensionless_fields(coarse, 0, gravity)
        coarse_fields = dimensionless_fields(coarse, coarse_index, gravity)
        fine_fields = dimensionless_fields(fine, fine_index, gravity)
        discrepancies = tuple(
            normalized_l2_difference(coarse_field, fine_field)
            for coarse_field, fine_field in zip(
                coarse_fields,
                fine_fields,
                strict=True,
            )
        )

        for column, (initial, coarse_field, fine_field) in enumerate(
            zip(initial_fields, coarse_fields, fine_fields, strict=True)
        ):
            axis = axes[row, column]
            axis.plot(
                x_over_length,
                initial,
                color="0.55",
                linewidth=1.0,
                linestyle=(0, (3, 2)),
                label=r"$t=0$",
            )
            axis.plot(
                x_over_length,
                coarse_field,
                color="#3568b8",
                linewidth=1.0,
                linestyle=(0, (4, 2)),
                label=r"$\Delta t=0.01$",
            )
            axis.plot(
                x_over_length,
                fine_field,
                color="#d95f02",
                linewidth=1.1,
                label=r"$\Delta t=0.005$",
            )
            axis.grid(alpha=0.2, linewidth=0.5)
            if row == 0:
                axis.set_title(column_titles[column])
            if row == len(CASE_IDS) - 1:
                axis.set_xlabel(r"$x/L$")
            if column == 0:
                axis.set_ylabel(
                    f"{CASE_LABELS[case_id]}\n"
                    rf"last finite $t={last_time:.2f}$"
                )

        records.append(
            {
                "case_id": case_id,
                "last_common_finite_saved_time": last_time,
                "coarse_first_failed_step_time": coarse.first_failed_step_time,
                "fine_first_failed_step_time": fine.first_failed_step_time,
                "relative_l2_difference_at_displayed_time": {
                    "eta": discrepancies[0],
                    "xi": discrepancies[1],
                    "gxi": discrepancies[2],
                },
            }
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncols=3,
        frameon=False,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.close(figure)

    return {
        "cases": records,
        "figure_pdf": str(pdf_path.relative_to(ROOT)),
        "figure_pdf_sha256": sha256(pdf_path),
        "figure_png": str(png_path.relative_to(ROOT)),
        "figure_png_sha256": sha256(png_path),
    }


def main() -> None:
    """Create the plot and its machine-readable record."""

    args = parse_args()
    panel_dir = args.panel_dir.resolve()
    coarse_path = panel_dir / "tanaka_benjamin_feir_dt_0p010.npz"
    fine_path = panel_dir / "tanaka_benjamin_feir_dt_0p005.npz"
    manifest_path = panel_dir / "case_manifest.npz"
    summary_path = panel_dir / "summary.json"

    with np.load(manifest_path, allow_pickle=False) as manifest:
        x = np.asarray(manifest["x"], dtype=np.float64)
    summary = json.loads(summary_path.read_text())
    length = float(summary["contract"]["length"])
    gravity = float(summary["contract"]["gravity"])

    output_stem = args.output_stem.resolve()
    record = make_figure(
        x_over_length=np.asarray(x / length, dtype=np.float64),
        coarse_arms=load_case_arms(coarse_path),
        fine_arms=load_case_arms(fine_path),
        gravity=gravity,
        output_stem=output_stem,
    )
    record.update(
        {
            "schema": "rejected_last_finite_profiles_v1",
            "source_coarse": str(coarse_path.relative_to(ROOT)),
            "source_fine": str(fine_path.relative_to(ROOT)),
            "source_manifest": str(manifest_path.relative_to(ROOT)),
            "requested_terminal_time": 200.0,
            "interpretation": (
                "These are fixed validation-panel rejections, not production-corpus "
                "samples. No finite terminal state exists; the figure shows the "
                "last common finite saved state before the first failed GL2 step."
            ),
        }
    )
    metrics_path = output_stem.with_name(f"{output_stem.name}_metrics.json")
    metrics_path.write_text(f"{json.dumps(record, indent=2, sort_keys=True)}\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
