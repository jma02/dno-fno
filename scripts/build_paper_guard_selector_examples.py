"""Plot initial conditions for which the soliton guard is active or inactive."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TANAKA_ARCHIVE = ROOT / (
    "outputs/c27_h1_to_l2_full_20260717_212550/"
    "eval_best_soliton_spectral_guard_20260719_185423/"
    "tanaka_g0/tanaka_g0_trajs.npz"
)
NON_TANAKA_ROOT = ROOT / (
    "outputs/c27_h1_to_l2_full_20260717_212550/"
    "eval_final_guarded_non_tanaka_suite_n32_20260719_221813"
)
OUTPUT_STEM = ROOT / "paper/figures/soliton_selector_examples"


@dataclass(frozen=True)
class Example:
    name: str
    archive: Path
    case_id: int


EXAMPLES = (
    Example("Tanaka", TANAKA_ARCHIVE, 23),
    Example(
        "finite-depth Stokes",
        NON_TANAKA_ROOT / "stokes_finite/stokes_finite_trajs.npz",
        49,
    ),
    Example(
        "Benjamin--Feir",
        NON_TANAKA_ROOT / "bf_g0/bf_g0_trajs.npz",
        11,
    ),
    Example(
        "deep-water random sea",
        NON_TANAKA_ROOT / "random_sea_deep/random_sea_deep_trajs.npz",
        12,
    ),
)


def load_initial_surface(example: Example) -> tuple[np.ndarray, float, float, bool]:
    """Load eta(0) and compute the two selector statistics."""
    with np.load(example.archive) as archive:
        case_ids = np.asarray(archive["case_ids"])
        matches = np.flatnonzero(case_ids == example.case_id)
        if matches.size != 1:
            raise ValueError(
                f"expected one case {example.case_id} in {example.archive}, "
                f"found {matches.size}"
            )
        eta = np.asarray(archive["truth_eta"][0, matches[0]], dtype=np.float64)

    eta_energy = float(np.sum(eta**2))
    negative_energy_fraction = float(np.sum(np.minimum(eta, 0.0) ** 2) / eta_energy)
    mean_eta = float(np.mean(eta))
    guard_active = negative_energy_fraction < 1.0e-3 and mean_eta > 0.0
    return eta, negative_energy_fraction, mean_eta, guard_active


def main() -> None:
    """Build PDF and PNG versions of the selector-example figure."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), constrained_layout=True)

    for panel_index, (axis, example) in enumerate(zip(axes.flat, EXAMPLES, strict=True)):
        eta, negative_fraction, mean_eta, guard_active = load_initial_surface(example)
        x_grid = np.linspace(0.0, 2.0 * np.pi, eta.size, endpoint=False)
        color = "#0072B2" if guard_active else "#D55E00"
        decision = "guard active" if guard_active else "guard inactive"

        axis.plot(x_grid, eta, color=color, linewidth=1.4)
        axis.fill_between(
            x_grid,
            eta,
            0.0,
            where=eta < 0.0,
            color="#999999",
            alpha=0.28,
            linewidth=0.0,
        )
        axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.55)
        axis.set_xlim(0.0, 2.0 * np.pi)
        axis.set_xticks((0.0, np.pi, 2.0 * np.pi), ("0", r"$\pi$", r"$2\pi$"))
        axis.set_title(
            f"({chr(ord('a') + panel_index)}) {example.name}, case {example.case_id}: "
            f"{decision}"
        )
        axis.text(
            0.02,
            0.95,
            rf"$r_-={negative_fraction:.3g}$" + "\n" + rf"$\overline{{\eta}}={mean_eta:.2e}$",
            transform=axis.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
        )
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$\eta_0(x)$")

    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
