"""Plot one representative initial condition from each proposed training family."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeAlias

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax.numpy as jnp
import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solver.data.stokes_truth_jax import stokes_eta_xi


ROOT = Path(__file__).resolve().parents[1]
LENGTH = 2.0 * np.pi
GRAVITY = 1.0
N_POINTS = 1024

FloatArray: TypeAlias = NDArray[np.float64]
Case: TypeAlias = tuple[str, str, FloatArray, FloatArray, float]


def _stokes_case(x: FloatArray) -> Case:
    """Construct one finite-depth fifth-order Stokes state."""
    depth = 0.10
    mode = 14
    amplitude = 0.005
    eta, xi = stokes_eta_xi(
        x=jnp.asarray(x),
        time=jnp.asarray(0.0),
        n0=mode,
        a0=amplitude,
        length=LENGTH,
        depth=depth,
        gravity=GRAVITY,
        ichoi=1,
    )
    return (
        "Stokes",
        rf"$h={depth:g},\ n={mode},\ ka={mode * amplitude:.2f}$",
        np.asarray(eta, dtype=np.float64),
        np.asarray(xi, dtype=np.float64),
        depth,
    )


def _tanaka_case() -> Case:
    """Select a fixed three-profile Tanaka state from the held-out case archive."""
    path = ROOT / "data/manuscript_ic_panels_v1_20260715/tanaka_g0_ics.npz"
    with np.load(path, allow_pickle=False) as archive:
        eta = np.asarray(archive["eta"], dtype=np.float64)
        xi = np.asarray(archive["xi"], dtype=np.float64)
        depths = np.asarray(archive["depth"], dtype=np.float64)
        case_ids = np.asarray(archive["case_ids"], dtype=np.int64)
        metadata = json.loads(archive["meta.json"].item().decode("utf-8"))

    specifications = metadata["case_specs"]
    matches = np.flatnonzero(case_ids == 90000101)
    if matches.size != 1:
        raise ValueError("expected exactly one Tanaka case with ID 90000101")
    index = int(matches[0])
    depth = float(depths[index])
    amplitude_ratio = sum(
        float(component["amplitude"]) for component in specifications[index]
    )
    return (
        "Tanaka sum",
        rf"$h={depth:.3f},\ m=3,\ \sum_j a_j/h={amplitude_ratio:.2f}$",
        eta[index],
        xi[index],
        depth,
    )


def _benjamin_feir_case(x: FloatArray) -> Case:
    """Construct a Stokes carrier with two explicit Airy sidebands."""
    depth = 1.5
    carrier_mode = 10
    carrier_steepness = 0.10
    sideband_offset = 2
    sideband_ratio = 0.15
    carrier_amplitude = carrier_steepness / carrier_mode
    eta_carrier, xi_carrier = stokes_eta_xi(
        x=jnp.asarray(x),
        time=jnp.asarray(0.0),
        n0=carrier_mode,
        a0=carrier_amplitude,
        length=LENGTH,
        depth=depth,
        gravity=GRAVITY,
        ichoi=0,
    )
    eta = np.array(eta_carrier, dtype=np.float64, copy=True)
    xi = np.array(xi_carrier, dtype=np.float64, copy=True)
    phases = (7.0 * np.pi / 4.0, np.pi / 3.0)
    for mode, phase in zip(
        (carrier_mode - sideband_offset, carrier_mode + sideband_offset),
        phases,
    ):
        amplitude = sideband_ratio * carrier_amplitude
        flat_dno = mode * np.tanh(mode * depth)
        frequency = np.sqrt(GRAVITY * flat_dno)
        eta += amplitude * np.cos(mode * x + phase)
        xi += amplitude * frequency / flat_dno * np.sin(mode * x + phase)
    return (
        "Benjamin--Feir",
        rf"$h={depth:g},\ n_c={carrier_mode},\ \Delta n={sideband_offset},\ "
        rf"k_ca={carrier_steepness:.2f}$",
        eta,
        xi,
        depth,
    )


def _tma_factor(depth_wavenumber: FloatArray) -> FloatArray:
    """Evaluate the TMA finite-depth multiplier."""
    return np.tanh(depth_wavenumber) ** 2 / (
        1.0 + 2.0 * depth_wavenumber / np.sinh(2.0 * depth_wavenumber)
    )


def _jonswap_tma_case(x: FloatArray) -> Case:
    """Construct one resolved-band JONSWAP/TMA random-phase state."""
    depth = 0.20
    significant_height = 0.020
    peak_wavenumber = 9.0
    peak_enhancement = 3.3
    right_moving_fraction = 0.5
    cutoff = 128
    transition = 96.0

    modes = np.arange(1, cutoff + 1, dtype=np.float64)
    nodes, weights = np.polynomial.legendre.leggauss(16)
    cell_wavenumbers = modes[:, None] + 0.5 * nodes[None, :]
    angular_frequency = np.sqrt(
        GRAVITY
        * cell_wavenumbers
        * np.tanh(cell_wavenumbers * depth)
    )
    peak_frequency = np.sqrt(
        GRAVITY * peak_wavenumber * np.tanh(peak_wavenumber * depth)
    )
    width = np.where(angular_frequency <= peak_frequency, 0.07, 0.09)
    peak_shape = np.exp(
        -(angular_frequency - peak_frequency) ** 2
        / (2.0 * width**2 * peak_frequency**2)
    )
    jonswap = (
        GRAVITY**2
        * angular_frequency ** (-5)
        * np.exp(-1.25 * (peak_frequency / angular_frequency) ** 4)
        * peak_enhancement**peak_shape
    )
    depth_wavenumber = cell_wavenumbers * depth
    group_velocity = GRAVITY * (
        np.tanh(depth_wavenumber)
        + depth_wavenumber / np.cosh(depth_wavenumber) ** 2
    ) / (2.0 * angular_frequency)
    window = np.where(
        cell_wavenumbers <= transition,
        1.0,
        np.where(
            cell_wavenumbers < cutoff,
            np.cos(
                0.5
                * np.pi
                * (cell_wavenumbers - transition)
                / (cutoff - transition)
            )
            ** 2,
            0.0,
        ),
    )
    density = jonswap * _tma_factor(depth_wavenumber) * group_velocity * window
    cell_energy = 0.5 * np.sum(weights[None, :] * density, axis=1)
    energy_fraction = cell_energy / np.sum(cell_energy)

    variance = (significant_height / 4.0) ** 2
    amplitude_right = np.sqrt(
        2.0 * right_moving_fraction * variance * energy_fraction
    )
    amplitude_left = np.sqrt(
        2.0 * (1.0 - right_moving_fraction) * variance * energy_fraction
    )
    rng = np.random.default_rng(20260722)
    phase_right = rng.uniform(0.0, 2.0 * np.pi, size=modes.size)
    phase_left = rng.uniform(0.0, 2.0 * np.pi, size=modes.size)
    arguments_right = modes[:, None] * x[None, :] + phase_right[:, None]
    arguments_left = modes[:, None] * x[None, :] + phase_left[:, None]
    eta = np.sum(
        amplitude_right[:, None] * np.cos(arguments_right)
        + amplitude_left[:, None] * np.cos(arguments_left),
        axis=0,
    )
    flat_dno = modes * np.tanh(modes * depth)
    frequency = np.sqrt(GRAVITY * flat_dno)
    xi = np.sum(
        (frequency / flat_dno)[:, None]
        * (
            amplitude_right[:, None] * np.sin(arguments_right)
            - amplitude_left[:, None] * np.sin(arguments_left)
        ),
        axis=0,
    )
    return (
        "JONSWAP/TMA",
        rf"$h={depth:g},\ H_s={significant_height:.3f},\ k_p={peak_wavenumber:g}$"
        "\n"
        rf"$\gamma={peak_enhancement:g},\ r_d={right_moving_fraction:g}$",
        eta,
        xi,
        depth,
    )


def _plot_cases(cases: tuple[Case, ...], output_stem: Path) -> None:
    """Render the four cases with common dimensionless quantities."""
    figure, axes = plt.subplots(
        len(cases),
        2,
        figsize=(8.0, 8.0),
        sharex="col",
        constrained_layout=True,
    )
    horizontal_coordinate = np.arange(N_POINTS, dtype=np.float64) / N_POINTS
    colors = ("#3366a6", "#b54a3a", "#6b4c9a", "#2a8c6a")

    for row, ((name, parameters, eta, xi, depth), color) in enumerate(
        zip(cases, colors)
    ):
        axes[row, 0].plot(horizontal_coordinate, eta / depth, color=color, lw=1.1)
        axes[row, 1].plot(
            horizontal_coordinate,
            xi / (depth * np.sqrt(GRAVITY * depth)),
            color=color,
            lw=1.1,
        )
        axes[row, 0].set_ylabel(r"$\eta/h$")
        axes[row, 0].text(
            0.02,
            0.92,
            f"{name}\n{parameters}",
            transform=axes[row, 0].transAxes,
            va="top",
            fontsize=8.5,
        )
        for axis in axes[row]:
            axis.grid(alpha=0.22, linewidth=0.5)

    axes[0, 0].set_title("surface elevation")
    axes[0, 1].set_title("surface potential")
    axes[-1, 0].set_xlabel(r"$x/L$")
    axes[-1, 1].set_xlabel(r"$x/L$")
    for axis in axes[:, 1]:
        axis.set_ylabel(r"$\xi/(h\sqrt{gh})$")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Build the representative-family figure."""
    x = np.linspace(0.0, LENGTH, N_POINTS, endpoint=False)
    cases = (
        _stokes_case(x),
        _tanaka_case(),
        _benjamin_feir_case(x),
        _jonswap_tma_case(x),
    )
    _plot_cases(
        cases,
        ROOT / "notes/figures/parameterized_corpus_case_examples",
    )


if __name__ == "__main__":
    main()
