"""Plot the completed paper-corpus full-horizon refinement summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


FAMILY_COLORS = {
    "stokes_finite": "#4477AA",
    "stokes_deep": "#66CCEE",
    "tanaka": "#228833",
    "benjamin_feir": "#CCBB44",
    "jonswap_tma_shallow": "#EE6677",
    "jonswap_tma_finite": "#AA3377",
    "jonswap_tma_deep": "#BBBBBB",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "outputs/full_horizon_refinement_panel_20260725/summary.json"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            "outputs/full_horizon_refinement_panel_20260725/"
            "full_horizon_refinement_panel"
        ),
    )
    return parser.parse_args()


def _case_records(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten completed group records while preserving their declared order."""

    return [
        case
        for group in summary["groups"]
        for case in group["cases"]
    ]


def _finite_plot_value(value: float | None, upper: float) -> float:
    """Map a missing or infinite diagnostic to the right edge of a log axis."""

    if value is None or not math.isfinite(float(value)):
        return upper
    return max(float(value), np.finfo(np.float64).tiny)


def _effective_pair(case: dict[str, Any]) -> dict[str, Any]:
    """Return the consecutive-step pair used for the final case decision."""

    retry = case.get("final_halving_retry", {})
    if retry.get("triggered", False):
        return retry
    return case


def plot_summary(
    summary: dict[str, Any],
    *,
    output_prefix: Path,
) -> tuple[Path, Path]:
    """Create PNG and PDF summaries of refinement and stage-solve defects."""

    if summary.get("status") != "complete":
        raise ValueError("the full-horizon summary is not complete")
    cases = _case_records(summary)
    if not cases:
        raise ValueError("the full-horizon summary contains no case results")

    labels = [case["case_id"].replace("_", " ") for case in cases]
    colors = [FAMILY_COLORS[case["family"]] for case in cases]
    refinement_limit = float(summary["contract"]["refinement_tolerance"])
    stage_limit = float(summary["contract"]["gl2_residual_tolerance"])
    refinement_upper = max(1.0, 10.0 * refinement_limit)
    stage_upper = max(1.0e-5, 10.0 * stage_limit)
    refinement = np.asarray(
        [
            _finite_plot_value(
                _effective_pair(case)[
                    "dimensionless_delivered_band_defect"
                ]["maximum"],
                refinement_upper,
            )
            for case in cases
        ]
    )
    coarse_residual = np.asarray(
        [
            _finite_plot_value(
                _effective_pair(case)["coarse_stage_telemetry"][
                    "maximum_stage_residual"
                ],
                stage_upper,
            )
            for case in cases
        ]
    )
    fine_residual = np.asarray(
        [
            _finite_plot_value(
                _effective_pair(case)["fine_stage_telemetry"][
                    "maximum_stage_residual"
                ],
                stage_upper,
            )
            for case in cases
        ]
    )

    y = np.arange(len(cases))
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14.0, 7.2),
        sharey=True,
        constrained_layout=True,
    )
    axes[0].barh(y, refinement, color=colors, edgecolor="black", linewidth=0.4)
    axes[0].axvline(
        refinement_limit,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=rf"acceptance $10^{{{math.log10(refinement_limit):.0f}}}$",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("maximum defect for the accepted consecutive-step pair")
    axes[0].set_ylabel("predeclared case")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", which="both", alpha=0.2)
    axes[0].legend(loc="lower right")

    height = 0.36
    axes[1].barh(
        y - height / 2.0,
        coarse_residual,
        height=height,
        color="#4477AA",
        label="coarser member of final pair",
    )
    axes[1].barh(
        y + height / 2.0,
        fine_residual,
        height=height,
        color="#EE6677",
        label="finer member of final pair",
    )
    axes[1].axvline(
        stage_limit,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=rf"stage tolerance $10^{{{math.log10(stage_limit):.0f}}}$",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("largest final GL2 stage defect")
    axes[1].grid(axis="x", which="both", alpha=0.2)
    axes[1].legend(loc="lower right")

    result = summary.get("result", "UNKNOWN")
    figure.suptitle(
        "Full-horizon reference validation\n"
        rf"$N=1024$, $M=6$, pad 8, delivered band $|k|\leq128$ — {result}",
        fontsize=13,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    figure.savefig(png_path, dpi=200)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    """Load the completed result and write both figure formats."""

    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    png_path, pdf_path = plot_summary(
        summary,
        output_prefix=args.output_prefix,
    )
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
