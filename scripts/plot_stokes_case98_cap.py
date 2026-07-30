"""Plot the finite-depth Stokes case 98 before and after a shallow-water cap."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import TypeAlias

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

from solver.data.stokes_truth_jax import _finite_depth_coeffs

jax.config.update("jax_enable_x64", True)

FloatArray: TypeAlias = np.ndarray

LENGTH = 164.0
DEPTH = 1.0
GRAVITY = 1.0
N0 = 14
ORIGINAL_A0 = 0.22543466384990768
SOURCE_TRAJECTORY = 316
SOURCE_PHASE_INDEX = 34
SOURCE_TIME = 13.6


def eta_harmonics(k0: float, depth: float, a0: float) -> FloatArray:
    """Return the five cosine amplitudes in the implemented Stokes expansion."""
    coeffs = {
        name: float(value)
        for name, value in _finite_depth_coeffs(k0, depth, GRAVITY, a0).items()
    }
    return np.asarray(
        [
            a0
            * (
                1.0
                + coeffs["eps2"] * coeffs["B31"]
                + coeffs["eps4"] * coeffs["B51"]
            ),
            a0
            * coeffs["eps"]
            * (coeffs["B22"] + coeffs["eps2"] * coeffs["B42"]),
            a0
            * coeffs["eps2"]
            * (coeffs["B33"] + coeffs["eps2"] * coeffs["B53"]),
            a0 * coeffs["eps3"] * coeffs["B44"],
            a0 * coeffs["eps4"] * coeffs["B55"],
        ],
        dtype=np.float64,
    )


def eta_from_phase(phase: FloatArray, amplitudes: FloatArray) -> FloatArray:
    modes = np.arange(1, amplitudes.size + 1, dtype=np.float64)
    return np.sum(
        amplitudes[:, None] * np.cos(modes[:, None] * phase[None, :]),
        axis=0,
    )


def diagnostics(
    phase: FloatArray,
    amplitudes: FloatArray,
    k0: float,
    depth: float,
    a0: float,
) -> dict[str, float]:
    eta = eta_from_phase(phase, amplitudes)
    modes = np.arange(1, amplitudes.size + 1, dtype=np.float64)
    eta_x = -k0 * np.sum(
        modes[:, None]
        * amplitudes[:, None]
        * np.sin(modes[:, None] * phase[None, :]),
        axis=0,
    )
    crest = float(np.max(eta) / depth)
    trough = float(np.min(eta) / depth)
    return {
        "a0": a0,
        "ka": k0 * a0,
        "cap_ratio_3ka_over_kh_cubed": 3.0 * k0 * a0 / (k0 * depth) ** 3,
        "second_over_fundamental": float(abs(amplitudes[1] / amplitudes[0])),
        "higher_over_fundamental": float(
            np.sum(np.abs(amplitudes[1:])) / abs(amplitudes[0])
        ),
        "crest_over_depth": crest,
        "trough_over_depth": trough,
        "crest_to_trough_magnitude": crest / abs(trough),
        "max_abs_eta_x": float(np.max(np.abs(eta_x))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stokes_case98_audit_20260723"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    k0 = 2.0 * math.pi * N0 / LENGTH
    kh = k0 * DEPTH
    capped_ka = kh**3 / 3.0
    capped_a0 = capped_ka / k0
    wavelength = 2.0 * math.pi / k0

    phase = np.linspace(-math.pi, math.pi, 4097, dtype=np.float64)
    x_over_wavelength = phase / (2.0 * math.pi)
    original_amplitudes = eta_harmonics(k0, DEPTH, ORIGINAL_A0)
    capped_amplitudes = eta_harmonics(k0, DEPTH, capped_a0)
    original_eta = eta_from_phase(phase, original_amplitudes) / DEPTH
    capped_eta = eta_from_phase(phase, capped_amplitudes) / DEPTH

    original_metrics = diagnostics(
        phase, original_amplitudes, k0, DEPTH, ORIGINAL_A0
    )
    capped_metrics = diagnostics(phase, capped_amplitudes, k0, DEPTH, capped_a0)

    colors = {"original": "#d1495b", "capped": "#277da1"}
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))

    axes[0].plot(
        x_over_wavelength,
        original_eta,
        color=colors["original"],
        linewidth=2.4,
        label="generated case 98",
    )
    axes[0].plot(
        x_over_wavelength,
        capped_eta,
        color=colors["capped"],
        linewidth=2.4,
        label=r"same $(k,h,\theta)$; $ka=(kh)^3/3$",
    )
    axes[0].axhline(0.0, color="0.35", linewidth=0.8)
    axes[0].set_xlabel(r"position over one wavelength, $x/\lambda$")
    axes[0].set_ylabel(r"surface elevation, $\eta/h$")
    axes[0].set_title("Surface profile (crests phase-aligned)")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].grid(alpha=0.2)

    modes = np.arange(1, 6)
    width = 0.36
    axes[1].bar(
        modes - width / 2.0,
        np.abs(original_amplitudes) / DEPTH,
        width,
        color=colors["original"],
        label="generated case 98",
    )
    axes[1].bar(
        modes + width / 2.0,
        np.abs(capped_amplitudes) / DEPTH,
        width,
        color=colors["capped"],
        label="amplitude-capped",
    )
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_locator(LogLocator(base=10.0))
    axes[1].set_xticks(modes)
    axes[1].set_xlabel(r"harmonic $m$ in $\cos(m\theta)$")
    axes[1].set_ylabel(r"harmonic amplitude, $|A_m|/h$")
    axes[1].set_title("Fifth-order elevation coefficients")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", which="both", alpha=0.2)

    parameter_text = (
        rf"$L={LENGTH:.0f}$, $h={DEPTH:.0f}$, $n_0={N0}$, "
        rf"$kh={kh:.6f}$, $\lambda/h={wavelength / DEPTH:.3f}$"
        "\n"
        rf"generated: $a={ORIGINAL_A0:.6f}$, $ka={k0 * ORIGINAL_A0:.6f}$; "
        rf"capped: $a={capped_a0:.6f}$, $ka={capped_ka:.6f}$"
        "\n"
        rf"$\sum_{{m=2}}^5|A_m|/|A_1|$: "
        rf"{original_metrics['higher_over_fundamental']:.3f} "
        rf"$\rightarrow$ {capped_metrics['higher_over_fundamental']:.3f}; "
        rf"$\max(\eta)/h$: {original_metrics['crest_over_depth']:.3f} "
        rf"$\rightarrow$ {capped_metrics['crest_over_depth']:.3f}"
    )
    figure.suptitle("Finite-depth Stokes case 98: sampled outside the ordered regime")
    figure.text(
        0.5,
        0.005,
        parameter_text,
        ha="center",
        va="bottom",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0.0, 0.13, 1.0, 0.95))

    figure_path = args.output_dir / "stokes_case98_original_vs_shallow_cap.png"
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    npz_path = args.output_dir / "stokes_case98_original_vs_shallow_cap.npz"
    np.savez(
        npz_path,
        phase=phase,
        x_over_wavelength=x_over_wavelength,
        original_eta_over_h=original_eta,
        capped_eta_over_h=capped_eta,
        original_harmonics_over_h=original_amplitudes / DEPTH,
        capped_harmonics_over_h=capped_amplitudes / DEPTH,
    )

    summary = {
        "case_id": 98,
        "source_trajectory": SOURCE_TRAJECTORY,
        "source_phase_index": SOURCE_PHASE_INDEX,
        "source_time": SOURCE_TIME,
        "length": LENGTH,
        "depth": DEPTH,
        "gravity": GRAVITY,
        "n0": N0,
        "k0": k0,
        "kh": kh,
        "wavelength_over_depth": wavelength / DEPTH,
        "cap": "ka <= (kh)^3 / 3",
        "original": {
            **original_metrics,
            "eta_harmonics_over_depth": (
                original_amplitudes / DEPTH
            ).tolist(),
        },
        "capped": {
            **capped_metrics,
            "eta_harmonics_over_depth": (capped_amplitudes / DEPTH).tolist(),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(figure_path)
    print(npz_path)
    print(summary_path)


if __name__ == "__main__":
    main()
