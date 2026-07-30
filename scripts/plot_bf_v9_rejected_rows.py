"""Plot the worst Benjamin--Feir rows removed during v9 assembly.

The historical v9 assembler kept rows only when every stored field was finite
and ``max(abs(G(eta; h) xi)) <= 1``.  Of the 14,751 removed BF rows, 14,750
were nonfinite and one was finite but exceeded the magnitude cap.  Nonfinite
arrays cannot be drawn, so this script shows the finite row immediately before
three representative nonfinite tails and the sole finite magnitude outlier.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RowSelection:
    label: str
    archive: str
    batch: int
    case_id: int
    row: int
    role: str


@dataclass(frozen=True)
class RowData:
    selection: RowSelection
    x: np.ndarray
    depth: float
    time: float
    eta: np.ndarray
    xi: np.ndarray
    gxi: np.ndarray
    case_times: np.ndarray
    case_eta_max: np.ndarray
    case_xi_max: np.ndarray
    case_gxi_max: np.ndarray
    next_time: float
    next_is_nonfinite: bool


SELECTIONS = (
    RowSelection(
        label="BF g0",
        archive="data/bf_2_adaptive_g0.npz",
        batch=1,
        case_id=445,
        row=37868,
        role="last finite frame before a nonfinite tail",
    ),
    RowSelection(
        label="BF g1",
        archive="data/bf_2_adaptive_g1.npz",
        batch=8,
        case_id=1002084,
        row=7266,
        role="last finite frame before a nonfinite tail",
    ),
    RowSelection(
        label="BF modal shard",
        archive="data/bf_2_adaptive_modal.npz",
        batch=13,
        case_id=2003578,
        row=50072,
        role="last finite frame before a nonfinite tail",
    ),
    RowSelection(
        label="BF modal shard",
        archive="data/bf_2_adaptive_modal.npz",
        batch=17,
        case_id=2004594,
        row=48587,
        role="sole finite row rejected by the magnitude cap",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/bf_v9_rejected_rows_20260723",
    )
    return parser.parse_args()


def load_npy(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    with archive.open(name) as handle:
        return np.load(handle, allow_pickle=False)


def load_group(
    archive_path: Path,
    batch: int,
    selections: list[RowSelection],
) -> list[RowData]:
    tag = f"{batch:04d}"
    with zipfile.ZipFile(archive_path) as archive:
        x = load_npy(archive, "x.npy")
        case_ids = load_npy(archive, f"case_id_batch_{tag}.npy")
        times = load_npy(archive, f"time_batch_{tag}.npy")
        depths = load_npy(archive, f"depth_batch_{tag}.npy")
        eta = load_npy(archive, f"eta_batch_{tag}.npy")
        xi = load_npy(archive, f"xi_batch_{tag}.npy")
        gxi = load_npy(archive, f"gxi_batch_{tag}.npy")

    results = []
    for selection in selections:
        row = selection.row
        if int(case_ids[row]) != selection.case_id:
            raise RuntimeError(
                f"row lookup failed for {selection.label} case {selection.case_id}"
            )
        next_row = row + 1
        if int(case_ids[next_row]) != selection.case_id:
            raise RuntimeError(
                f"selected row is the final frame for case {selection.case_id}"
            )
        case_mask = case_ids == selection.case_id
        row_finite = bool(
            np.isfinite(eta[row]).all()
            and np.isfinite(xi[row]).all()
            and np.isfinite(gxi[row]).all()
        )
        if not row_finite:
            raise RuntimeError(
                f"selected predecessor is nonfinite for case {selection.case_id}"
            )
        next_is_nonfinite = bool(
            not np.isfinite(eta[next_row]).all()
            or not np.isfinite(xi[next_row]).all()
            or not np.isfinite(gxi[next_row]).all()
        )
        if not next_is_nonfinite:
            raise RuntimeError(
                f"next row is unexpectedly finite for case {selection.case_id}"
            )

        case_eta = eta[case_mask]
        case_xi = xi[case_mask]
        case_gxi = gxi[case_mask]
        results.append(
            RowData(
                selection=selection,
                x=np.asarray(x),
                depth=float(depths[row]),
                time=float(times[row]),
                eta=np.asarray(eta[row]),
                xi=np.asarray(xi[row]),
                gxi=np.asarray(gxi[row]),
                case_times=np.asarray(times[case_mask]),
                case_eta_max=np.max(np.abs(case_eta), axis=-1),
                case_xi_max=np.max(np.abs(case_xi), axis=-1),
                case_gxi_max=np.max(np.abs(case_gxi), axis=-1),
                next_time=float(times[next_row]),
                next_is_nonfinite=next_is_nonfinite,
            )
        )
    return results


def load_rows() -> list[RowData]:
    groups: dict[tuple[str, int], list[RowSelection]] = {}
    for selection in SELECTIONS:
        groups.setdefault((selection.archive, selection.batch), []).append(
            selection
        )
    loaded: dict[int, RowData] = {}
    for (archive_name, batch), selections in groups.items():
        rows = load_group(REPO_ROOT / archive_name, batch, selections)
        loaded.update({row.selection.case_id: row for row in rows})
    return [loaded[selection.case_id] for selection in SELECTIONS]


def finite_curve(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values), values, np.nan)


def plot_rows(rows: list[RowData], output_path: Path) -> None:
    figure, axes = plt.subplots(
        len(rows),
        4,
        figsize=(16.0, 3.1 * len(rows)),
        constrained_layout=True,
    )
    for row_index, row in enumerate(rows):
        axes[row_index, 0].plot(row.x, row.eta, lw=1.0)
        axes[row_index, 1].plot(row.x, row.xi, lw=1.0)
        axes[row_index, 2].plot(row.x, row.gxi, lw=1.0)
        axes[row_index, 2].axhline(1.0, color="tab:red", ls="--", lw=0.9)
        axes[row_index, 2].axhline(-1.0, color="tab:red", ls="--", lw=0.9)

        axes[row_index, 3].semilogy(
            row.case_times,
            finite_curve(row.case_eta_max),
            label=r"$\max|\eta|$",
            lw=1.0,
        )
        axes[row_index, 3].semilogy(
            row.case_times,
            finite_curve(row.case_xi_max),
            label=r"$\max|\xi|$",
            lw=1.0,
        )
        axes[row_index, 3].semilogy(
            row.case_times,
            finite_curve(row.case_gxi_max),
            label=r"$\max|G(\eta;h)\xi|$",
            lw=1.0,
        )
        axes[row_index, 3].axvline(
            row.time,
            color="tab:red",
            ls="--",
            lw=0.9,
            label="shown frame",
        )
        axes[row_index, 3].axhline(
            1.0,
            color="0.3",
            ls=":",
            lw=0.8,
            label="old magnitude cap",
        )
        axes[row_index, 0].set_ylabel(
            f"{row.selection.label}\n"
            f"case {row.selection.case_id}\n"
            f"t={row.time:.2f}, h={row.depth:.4f}\n"
            f"next t={row.next_time:.2f}: nonfinite"
        )
        for column in range(4):
            axes[row_index, column].grid(alpha=0.2)
        axes[row_index, 2].text(
            0.98,
            0.94,
            rf"$\max|G(\eta;h)\xi|={np.max(np.abs(row.gxi)):.3g}$",
            transform=axes[row_index, 2].transAxes,
            ha="right",
            va="top",
        )
        axes[row_index, 3].legend(loc="best", fontsize=7)

    axes[0, 0].set_title(r"surface elevation $\eta$")
    axes[0, 1].set_title(r"surface potential $\xi$")
    axes[0, 2].set_title(r"$G(\eta;h)\xi$ (red: old $\pm1$ cap)")
    axes[0, 3].set_title("case history of maximum magnitudes")
    for axis in axes[-1, :3]:
        axis.set_xlabel("x")
    axes[-1, 3].set_xlabel("time")
    figure.suptitle(
        "Benjamin–Feir rows removed during v9 assembly\n"
        "The first three are the last finite saved frames before nonfinite tails; "
        "the fourth is the sole finite row that exceeded the old magnitude cap.",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_summary(rows: list[RowData], output_path: Path) -> None:
    payload: dict[str, Any] = {
        "historical_v9_assembly_rule": {
            "finite_required": True,
            "maximum_absolute_gxi": 1.0,
            "total_bf_rows_removed": 14751,
            "nonfinite_bf_rows_removed": 14750,
            "finite_magnitude_rows_removed": 1,
        },
        "important_limitation": (
            "This was a rowwise historical assembly filter, not the proposed "
            "whole-trajectory numerical-acceptance protocol."
        ),
        "rows": [
            {
                "label": row.selection.label,
                "archive": row.selection.archive,
                "batch": row.selection.batch,
                "case_id": row.selection.case_id,
                "row": row.selection.row,
                "time": row.time,
                "depth": row.depth,
                "role": row.selection.role,
                "maximum_absolute_eta": float(np.max(np.abs(row.eta))),
                "maximum_absolute_xi": float(np.max(np.abs(row.xi))),
                "maximum_absolute_gxi": float(np.max(np.abs(row.gxi))),
                "next_time": row.next_time,
                "next_is_nonfinite": row.next_is_nonfinite,
            }
            for row in rows
        ],
    }
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    plot_rows(rows, output_dir / "bf_v9_rejected_rows.png")
    write_summary(rows, output_dir / "bf_v9_rejected_rows.json")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "rows": [
                    {
                        "case_id": row.selection.case_id,
                        "time": row.time,
                        "maximum_absolute_gxi": float(
                            np.max(np.abs(row.gxi))
                        ),
                    }
                    for row in rows
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
