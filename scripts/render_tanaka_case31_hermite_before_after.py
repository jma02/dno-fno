"""Render the exact case-31 rollout before and after tangent-Hermite placement."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "outputs/tanaka_case31_tangent_resampling_trial_20260723"
    / "paired_trajectories.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tanaka_case31_tangent_resampling_trial_20260723"
    / "before_after_t76p32.png"
)
LENGTH = 2.0 * math.pi
TARGET_TIME = 76.32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--time", type=float, default=TARGET_TIME)
    return parser.parse_args()


def retained_band(
    field: NDArray[np.float64],
    lower_mode: int,
    upper_mode: int,
) -> NDArray[np.float64]:
    spectrum = np.fft.rfft(field)
    retained = np.zeros_like(spectrum)
    retained[lower_mode : upper_mode + 1] = spectrum[
        lower_mode : upper_mode + 1
    ]
    return np.fft.irfft(retained, n=field.shape[-1])


def coefficient_band_norm(
    field: NDArray[np.float64],
    lower_mode: int,
    upper_mode: int,
) -> float:
    coefficients = np.fft.rfft(field) / field.shape[-1]
    return float(
        np.linalg.norm(coefficients[lower_mode : upper_mode + 1])
    )


def render(
    input_path: Path,
    output_path: Path,
    target_time: float,
) -> None:
    with np.load(input_path) as archive:
        times = np.asarray(archive["times"], dtype=np.float64)
        eta = np.asarray(archive["eta"], dtype=np.float64)
        gxi = np.asarray(archive["gxi"], dtype=np.float64)
        sign_counts = np.asarray(archive["sign_counts"], dtype=np.int32)

    frame = int(np.argmin(np.abs(times - target_time)))
    time = float(times[frame])
    x = LENGTH * np.arange(eta.shape[-1]) / eta.shape[-1]
    colors = ("#2c6ba1", "#d95f02")
    column_titles = (
        "Before: piecewise-linear placement",
        r"After: Hermite placement with $\eta_x=\tan\theta$",
    )
    high_band = np.stack(
        tuple(retained_band(gxi[frame, arm], 80, 128) for arm in range(2))
    )
    band_norms = tuple(
        coefficient_band_norm(gxi[frame, arm], 80, 128)
        for arm in range(2)
    )

    figure, axes = plt.subplots(
        3,
        2,
        figsize=(12.0, 9.0),
        sharex=True,
        constrained_layout=True,
    )
    figure.suptitle(
        rf"Exact Tanaka case 31 at $t={time:.2f}$",
        fontsize=17,
    )
    for arm in range(2):
        axes[0, arm].set_title(column_titles[arm], fontsize=13)
        axes[0, arm].plot(x, eta[frame, arm], color=colors[arm], lw=2.0)
        axes[1, arm].plot(x, gxi[frame, arm], color=colors[arm], lw=1.6)
        axes[2, arm].plot(x, high_band[arm], color=colors[arm], lw=1.3)
        axes[1, arm].text(
            0.03,
            0.94,
            f"sign count = {int(sign_counts[frame, arm])}",
            transform=axes[1, arm].transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        axes[2, arm].text(
            0.03,
            0.94,
            rf"$\|\widehat{{G\xi}}_{{80:128}}\|_2={band_norms[arm]:.2e}$",
            transform=axes[2, arm].transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        axes[2, arm].set_xlabel(r"$x$")
        for axis in axes[:, arm]:
            axis.grid(alpha=0.25)
            axis.set_xlim(0.0, LENGTH)

    axes[0, 0].set_ylabel(r"surface $\eta$")
    axes[1, 0].set_ylabel(r"$G(\eta;h)\xi$")
    axes[2, 0].set_ylabel(r"$P_{80:128}G(\eta;h)\xi$")
    axes[0, 0].set_ylim(axes[0, 1].get_ylim())
    axes[1, 0].set_ylim(axes[1, 1].get_ylim())
    high_limit = 1.05 * float(np.max(np.abs(high_band)))
    axes[2, 0].set_ylim(-high_limit, high_limit)
    axes[2, 1].set_ylim(-high_limit, high_limit)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    render(args.input.resolve(), args.output.resolve(), args.time)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
