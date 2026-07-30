"""Paired CPU rollout for the exact May-v2 Tanaka case-31 construction.

The control uses the production piecewise-linear profile placement. The trial
uses cubic Hermite reconstruction with the Tanaka slope eta_x = tan(theta).
Every subsequent operation is shared between the two arms.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from solver.gen_data.generate_tanaka_dataset_v2 import chunked_gxi_over_time
from solver.solvers.dno_series_jax import (
    build_grid,
    dno_series_eval,
    make_linear_dno_symbol,
    myfft,
    myifft,
)
from solver.solvers.time_integrator import (
    SolverParams,
    State,
    apply_lowpass,
    batched_rollout,
    cast_solver_params_dtype,
    cast_state_dtype,
    make_normalized_rollout_settings,
    spectral_dx,
)
from solver.tanaka_ICs.modified_tanaka import (
    make_default_tanaka_template,
    solve_modified_tanaka_batched,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = ROOT / "data/tanaka_2_adaptive_g0.npz"
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs/tanaka_case31_tangent_resampling_trial_20260723"
)

NX = 1024
LENGTH = 2.0 * math.pi
GRAVITY = 1.0
DEPTH = 0.26861433760407505
FINE_FACTOR = 8
FILTER_FRACTION = 0.25
CRITICAL_TIME = 76.32
CASE_ID = 31
SAMPLES_PER_CASE = 200
ARCHIVE_INITIAL_ROW = CASE_ID * SAMPLES_PER_CASE
ARCHIVE_CRITICAL_ROW = ARCHIVE_INITIAL_ROW + 76
RELATIVE_DEADZONE = 0.03

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]


@dataclass(frozen=True)
class Crest:
    amplitude: float
    center: float
    direction: int


CRESTS = (
    Crest(
        amplitude=0.05548354495289499,
        center=3.5548496920089176,
        direction=-1,
    ),
    Crest(
        amplitude=0.053531413616445936,
        center=5.032703928715591,
        direction=1,
    ),
)


@dataclass(frozen=True)
class InitialCondition:
    eta: jax.Array
    xi: jax.Array
    eta_per_crest: jax.Array
    xi_per_crest: jax.Array
    signed_speeds: jax.Array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the exact case-31 linear-versus-Hermite CPU trial."
    )
    parser.add_argument("--tmax", type=float, default=200.0)
    parser.add_argument("--output_dt", type=float, default=0.08)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gxi_chunk_size", type=int, default=16)
    parser.add_argument(
        "--save_full",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the complete paired eta/xi/G(eta)xi trajectory as float32.",
    )
    return parser.parse_args()


def cubic_hermite_nonuniform(
    x_nodes: jax.Array,
    y_nodes: jax.Array,
    slopes: jax.Array,
    x_eval: jax.Array,
) -> jax.Array:
    """Evaluate a nonuniform cubic Hermite profile with zero exterior."""
    inside = (x_eval >= x_nodes[0]) & (x_eval <= x_nodes[-1])
    safe_x = jnp.where(inside, x_eval, x_nodes[0])
    interval = (
        jnp.searchsorted(x_nodes, safe_x, side="right", method="scan") - 1
    )
    interval = jnp.clip(interval, 0, x_nodes.shape[0] - 2)

    x_left = x_nodes[interval]
    width = x_nodes[interval + 1] - x_left
    fraction = (safe_x - x_left) / width
    fraction_squared = fraction * fraction
    fraction_cubed = fraction_squared * fraction

    h00 = 2.0 * fraction_cubed - 3.0 * fraction_squared + 1.0
    h10 = fraction_cubed - 2.0 * fraction_squared + fraction
    h01 = -2.0 * fraction_cubed + 3.0 * fraction_squared
    h11 = fraction_cubed - fraction_squared

    value = (
        h00 * y_nodes[interval]
        + h10 * width * slopes[interval]
        + h01 * y_nodes[interval + 1]
        + h11 * width * slopes[interval + 1]
    )
    return jnp.where(inside, value, jnp.zeros_like(value))


def downsample_from_fine_grid(field: jax.Array) -> jax.Array:
    spectrum = jnp.fft.rfft(field)
    native_spectrum = spectrum[: NX // 2 + 1] / FINE_FACTOR
    return jnp.fft.irfft(native_spectrum, n=NX)


def place_linear(
    x_profile: jax.Array,
    eta_profile: jax.Array,
    _theta_profile: jax.Array,
    depth: jax.Array,
    center: jax.Array,
    x_fine: jax.Array,
) -> jax.Array:
    x_nodes = x_profile * depth + center
    eta_nodes = eta_profile * depth
    copies = (
        jnp.interp(x_fine, x_nodes, eta_nodes, left=0.0, right=0.0),
        jnp.interp(
            x_fine - LENGTH,
            x_nodes,
            eta_nodes,
            left=0.0,
            right=0.0,
        ),
        jnp.interp(
            x_fine + LENGTH,
            x_nodes,
            eta_nodes,
            left=0.0,
            right=0.0,
        ),
    )
    return downsample_from_fine_grid(sum(copies))


def place_tangent_hermite(
    x_profile: jax.Array,
    eta_profile: jax.Array,
    theta_profile: jax.Array,
    depth: jax.Array,
    center: jax.Array,
    x_fine: jax.Array,
) -> jax.Array:
    midpoint = x_profile.shape[0] // 2
    eta_nodes = eta_profile.at[0].set(0.0).at[-1].set(0.0)
    slopes = (
        jnp.tan(theta_profile)
        .at[0]
        .set(0.0)
        .at[-1]
        .set(0.0)
        .at[midpoint]
        .set(0.0)
    )
    shifts = jnp.asarray((-LENGTH, 0.0, LENGTH), dtype=x_fine.dtype)
    dimensionless_queries = (
        x_fine[None, :] + shifts[:, None] - center
    ) / depth
    copies = cubic_hermite_nonuniform(
        x_profile,
        eta_nodes,
        slopes,
        dimensionless_queries,
    )
    return depth * downsample_from_fine_grid(jnp.sum(copies, axis=0))


def solve_profiles() -> tuple[Any, dict[str, float]]:
    amplitudes = jnp.asarray(
        [crest.amplitude for crest in CRESTS],
        dtype=jnp.float64,
    )
    directions = jnp.asarray(
        [crest.direction for crest in CRESTS],
        dtype=jnp.float64,
    )
    template = make_default_tanaka_template(
        depth=1.0,
        gravity=GRAVITY,
        direction=1,
        nx=NX,
        length=LENGTH,
        center=0.0,
        dno_order=6,
        pad_factor=8,
    )
    solution = solve_modified_tanaka_batched(
        template,
        amplitudes,
        centers=jnp.zeros_like(amplitudes),
        directions=directions,
    )
    endpoint_diagnostics = {
        "maximum_endpoint_eta": float(
            jnp.max(jnp.abs(solution.eta_profile[:, (0, -1)]))
        ),
        "maximum_endpoint_slope": float(
            jnp.max(jnp.abs(jnp.tan(solution.theta_profile[:, (0, -1)])))
        ),
        "maximum_crest_slope": float(
            jnp.max(
                jnp.abs(
                    jnp.tan(
                        solution.theta_profile[
                            :, solution.theta_profile.shape[-1] // 2
                        ]
                    )
                )
            )
        ),
        "minimum_x_spacing": float(
            jnp.min(jnp.diff(solution.x_profile, axis=-1))
        ),
        "minimum_cos_theta": float(jnp.min(jnp.cos(solution.theta_profile))),
    }
    return solution, endpoint_diagnostics


def build_initial_condition(
    solution: Any,
    placement: Callable[
        [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        jax.Array,
    ],
) -> InitialCondition:
    nx_fine = NX * FINE_FACTOR
    x_fine = (LENGTH / nx_fine) * jnp.arange(
        nx_fine,
        dtype=jnp.float64,
    )
    depth = jnp.asarray(DEPTH, dtype=jnp.float64)
    centers = jnp.asarray(
        [crest.center for crest in CRESTS],
        dtype=jnp.float64,
    )
    eta_per_crest = jax.vmap(
        lambda x_profile, eta_profile, theta_profile, center: placement(
            x_profile,
            eta_profile,
            theta_profile,
            depth,
            center,
            x_fine,
        )
    )(
        solution.x_profile,
        solution.eta_profile,
        solution.theta_profile,
        centers,
    )

    _, k = build_grid(NX, LENGTH)
    k = jnp.asarray(k, dtype=jnp.float64)
    directions = jnp.asarray(
        [crest.direction for crest in CRESTS],
        dtype=jnp.float64,
    )
    signed_speeds = directions * solution.froude * jnp.sqrt(GRAVITY * depth)
    eta_x = spectral_dx(eta_per_crest, k)
    radical = (1.0 + eta_x**2) * (
        signed_speeds[:, None] ** 2
        - 2.0 * GRAVITY * eta_per_crest
    )
    xi_x = signed_speeds[:, None] - directions[:, None] * jnp.sqrt(
        jnp.maximum(radical, 0.0)
    )
    inv_ik = jnp.where(k != 0.0, 1.0 / (1j * k), 0.0)
    xi_per_crest = myifft(inv_ik * myfft(xi_x, NX))
    xi_per_crest = xi_per_crest - jnp.mean(
        xi_per_crest,
        axis=-1,
        keepdims=True,
    )

    eta = jnp.sum(eta_per_crest, axis=0)
    xi = jnp.sum(xi_per_crest, axis=0)
    eta = apply_lowpass(eta, k, FILTER_FRACTION)
    xi = apply_lowpass(xi, k, FILTER_FRACTION)
    xi = xi - jnp.mean(xi)
    return InitialCondition(
        eta=eta,
        xi=xi,
        eta_per_crest=apply_lowpass(
            eta_per_crest,
            k,
            FILTER_FRACTION,
        ),
        xi_per_crest=apply_lowpass(
            xi_per_crest,
            k,
            FILTER_FRACTION,
        ),
        signed_speeds=signed_speeds,
    )


def read_npy_member(
    archive: zipfile.ZipFile,
    member: str,
) -> NDArray[Any]:
    with archive.open(member) as handle:
        return np.load(handle, allow_pickle=False)


def load_archived_frames() -> dict[str, FloatArray | float | int]:
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        suffix = "batch_0000.npy"
        times = read_npy_member(archive, f"time_{suffix}")
        case_ids = read_npy_member(archive, f"case_id_{suffix}")
        indices = read_npy_member(
            archive,
            "subsample_indices_batch_0000.npy",
        )
        eta = read_npy_member(archive, f"eta_{suffix}")
        xi = read_npy_member(archive, f"xi_{suffix}")
        gxi = read_npy_member(archive, f"gxi_{suffix}")
        depth = read_npy_member(archive, f"depth_{suffix}")

    if int(case_ids[ARCHIVE_INITIAL_ROW]) != CASE_ID:
        raise ValueError("Archive row mapping does not identify case 31.")
    if int(indices[CASE_ID, 76]) != 954:
        raise ValueError("Archive critical sample does not map to dense index 954.")
    if not np.isclose(times[ARCHIVE_CRITICAL_ROW], CRITICAL_TIME):
        raise ValueError("Archive critical sample time does not equal 76.32.")
    return {
        "initial_eta": np.asarray(eta[ARCHIVE_INITIAL_ROW], dtype=np.float64),
        "initial_xi": np.asarray(xi[ARCHIVE_INITIAL_ROW], dtype=np.float64),
        "initial_gxi": np.asarray(gxi[ARCHIVE_INITIAL_ROW], dtype=np.float64),
        "critical_eta": np.asarray(
            eta[ARCHIVE_CRITICAL_ROW],
            dtype=np.float64,
        ),
        "critical_xi": np.asarray(
            xi[ARCHIVE_CRITICAL_ROW],
            dtype=np.float64,
        ),
        "critical_gxi": np.asarray(
            gxi[ARCHIVE_CRITICAL_ROW],
            dtype=np.float64,
        ),
        "stored_depth": float(depth[ARCHIVE_INITIAL_ROW]),
        "critical_dense_index": int(indices[CASE_ID, 76]),
    }


def relative_l2(candidate: FloatArray, reference: FloatArray) -> float:
    denominator = float(np.linalg.norm(reference.ravel()))
    return float(
        np.linalg.norm((candidate - reference).ravel())
        / (denominator + 1e-300)
    )


def coefficient_band_norm(
    field: FloatArray,
    lower: int,
    upper: int,
) -> FloatArray:
    coefficients = np.fft.rfft(field, axis=-1) / field.shape[-1]
    return np.linalg.norm(coefficients[..., lower : upper + 1], axis=-1)


def relative_band_error(
    candidate: FloatArray,
    reference: FloatArray,
    lower: int,
    upper: int,
) -> float:
    candidate_hat = (
        np.fft.rfft(candidate, axis=-1)[..., lower : upper + 1]
        / candidate.shape[-1]
    )
    reference_hat = (
        np.fft.rfft(reference, axis=-1)[..., lower : upper + 1]
        / reference.shape[-1]
    )
    return relative_l2(candidate_hat, reference_hat)


def historical_sign_count(field: FloatArray) -> int:
    difference = np.roll(field, -1) - field
    threshold = RELATIVE_DEADZONE * float(np.max(np.abs(difference)))
    signs = np.where(
        difference > threshold,
        1,
        np.where(difference < -threshold, -1, 0),
    )
    active = signs[signs != 0]
    if active.size <= 1:
        return 0
    return int(np.sum(active != np.roll(active, -1)))


def sign_count_history(gxi: FloatArray) -> IntArray:
    flat = gxi.reshape((-1, gxi.shape[-1]))
    counts = np.fromiter(
        (historical_sign_count(row) for row in flat),
        dtype=np.int32,
        count=flat.shape[0],
    )
    return counts.reshape(gxi.shape[:-1])


def gxi_for_states(
    eta: jax.Array,
    xi: jax.Array,
    k: jax.Array,
    depth: jax.Array,
) -> jax.Array:
    result = dno_series_eval(
        eta,
        xi,
        k,
        depth,
        6,
        pad_factor=8,
    )
    return apply_lowpass(result, k, FILTER_FRACTION)


def traveling_wave_residuals(
    initial_condition: InitialCondition,
    k: jax.Array,
) -> FloatArray:
    depth = jnp.full((len(CRESTS), 1), DEPTH, dtype=jnp.float64)
    gxi = gxi_for_states(
        initial_condition.eta_per_crest,
        initial_condition.xi_per_crest,
        k,
        depth,
    )
    expected = -initial_condition.signed_speeds[:, None] * spectral_dx(
        initial_condition.eta_per_crest,
        k,
    )
    expected = apply_lowpass(expected, k, FILTER_FRACTION)
    numerator = jnp.linalg.norm(gxi - expected, axis=-1)
    denominator = jnp.linalg.norm(expected, axis=-1)
    return np.asarray(numerator / denominator, dtype=np.float64)


def compute_metrics(
    times: FloatArray,
    eta: FloatArray,
    xi: FloatArray,
    gxi: FloatArray,
    sign_counts: IntArray,
    archive: dict[str, FloatArray | float | int],
    control_initial_gxi: FloatArray,
    endpoint_diagnostics: dict[str, float],
    traveling_residuals: dict[str, FloatArray],
    timing: dict[str, float],
) -> dict[str, Any]:
    arm_names = ("piecewise_linear", "tangent_hermite")
    critical_index = int(round(CRITICAL_TIME / float(times[1] - times[0])))
    critical_available = critical_index < times.shape[0]
    observed_index = (
        critical_index if critical_available else int(times.shape[0] - 1)
    )
    dx = LENGTH / NX
    hamiltonian = 0.5 * dx * np.sum(
        xi * gxi + GRAVITY * eta**2,
        axis=-1,
    )
    hamiltonian_drift = (
        hamiltonian - hamiltonian[0:1]
    ) / (np.abs(hamiltonian[0:1]) + 1e-300)

    bands = {
        "0_32": (0, 32),
        "33_79": (33, 79),
        "80_128": (80, 128),
    }
    band_norms = {
        field_name: {
            band_name: coefficient_band_norm(field, *bounds)
            for band_name, bounds in bands.items()
        }
        for field_name, field in (
            ("eta", eta),
            ("xi", xi),
            ("gxi", gxi),
        )
    }

    archive_initial_eta = np.asarray(archive["initial_eta"])
    archive_initial_xi = np.asarray(archive["initial_xi"])
    archive_initial_gxi = np.asarray(archive["initial_gxi"])
    archive_reproduction = {
        "eta_relative_l2": relative_l2(
            eta[0, 0],
            archive_initial_eta,
        ),
        "xi_relative_l2": relative_l2(
            xi[0, 0],
            archive_initial_xi,
        ),
        "gxi_relative_l2": relative_l2(
            control_initial_gxi,
            archive_initial_gxi,
        ),
        "eta_float32_bitwise_equal": bool(
            np.array_equal(eta[0, 0].astype(np.float32), archive_initial_eta)
        ),
        "xi_float32_bitwise_equal": bool(
            np.array_equal(xi[0, 0].astype(np.float32), archive_initial_xi)
        ),
        "gxi_float32_bitwise_equal": bool(
            np.array_equal(
                control_initial_gxi.astype(np.float32),
                archive_initial_gxi,
            )
        ),
    }

    critical_archive_comparison: dict[str, float] | None = None
    if critical_available:
        critical_archive_comparison = {
            "eta_relative_l2": relative_l2(
                eta[critical_index, 0],
                np.asarray(archive["critical_eta"]),
            ),
            "xi_relative_l2": relative_l2(
                xi[critical_index, 0],
                np.asarray(archive["critical_xi"]),
            ),
            "gxi_relative_l2": relative_l2(
                gxi[critical_index, 0],
                np.asarray(archive["critical_gxi"]),
            ),
        }

    per_arm: dict[str, Any] = {}
    for arm_index, arm_name in enumerate(arm_names):
        max_count_index = int(np.argmax(sign_counts[:, arm_index]))
        per_arm[arm_name] = {
            "maximum_sign_count": int(sign_counts[max_count_index, arm_index]),
            "maximum_sign_count_time": float(times[max_count_index]),
            "sign_count_at_observed_time": int(
                sign_counts[observed_index, arm_index]
            ),
            "observed_time": float(times[observed_index]),
            "rows_above_historical_threshold": int(
                np.sum(sign_counts[:, arm_index] > 10)
            ),
            "maximum_relative_hamiltonian_drift": float(
                np.max(np.abs(hamiltonian_drift[:, arm_index]))
            ),
            "minimum_water_column": float(
                np.min(DEPTH + eta[:, arm_index])
            ),
            "initial_eta_maximum": float(np.max(eta[0, arm_index])),
            "initial_eta_mass": float(dx * np.sum(eta[0, arm_index])),
            "traveling_wave_relative_residuals": [
                float(value)
                for value in traveling_residuals[arm_name]
            ],
            "initial_band_norms": {
                field_name: {
                    band_name: float(
                        field_bands[band_name][0, arm_index]
                    )
                    for band_name in bands
                }
                for field_name, field_bands in band_norms.items()
            },
            "observed_band_norms": {
                field_name: {
                    band_name: float(
                        field_bands[band_name][observed_index, arm_index]
                    )
                    for band_name in bands
                }
                for field_name, field_bands in band_norms.items()
            },
        }

    return {
        "case_id": CASE_ID,
        "depth_f64": DEPTH,
        "depth_f32_archive": float(archive["stored_depth"]),
        "crests": [
            {
                "amplitude": crest.amplitude,
                "center": crest.center,
                "direction": crest.direction,
            }
            for crest in CRESTS
        ],
        "protocol": {
            "nx": NX,
            "length": LENGTH,
            "gravity": GRAVITY,
            "dno_order": 6,
            "pad_factor": 8,
            "filter_fraction": FILTER_FRACTION,
            "method": "gl2_if",
            "implicit_iterations": 4,
            "substeps_per_output": 8,
            "output_dt": float(times[1] - times[0]),
            "inner_dt": float((times[1] - times[0]) / 8.0),
            "tmax": float(times[-1]),
        },
        "endpoint_diagnostics": endpoint_diagnostics,
        "archive_reproduction": archive_reproduction,
        "critical_time_available": critical_available,
        "critical_archive_comparison": critical_archive_comparison,
        "initial_arm_differences": {
            "eta_relative_l2": relative_l2(eta[0, 1], eta[0, 0]),
            "xi_relative_l2": relative_l2(xi[0, 1], xi[0, 0]),
            "gxi_relative_l2": relative_l2(gxi[0, 1], gxi[0, 0]),
            "eta_modes_0_32_relative_l2": relative_band_error(
                eta[0, 1],
                eta[0, 0],
                0,
                32,
            ),
            "xi_modes_0_32_relative_l2": relative_band_error(
                xi[0, 1],
                xi[0, 0],
                0,
                32,
            ),
            "gxi_modes_0_32_relative_l2": relative_band_error(
                gxi[0, 1],
                gxi[0, 0],
                0,
                32,
            ),
        },
        "per_arm": per_arm,
        "timing_seconds": timing,
    }


def plot_diagnostics(
    output: Path,
    times: FloatArray,
    eta: FloatArray,
    xi: FloatArray,
    gxi: FloatArray,
    sign_counts: IntArray,
) -> None:
    x = np.linspace(0.0, LENGTH, NX, endpoint=False)
    observed_index = min(
        int(round(CRITICAL_TIME / float(times[1] - times[0]))),
        times.shape[0] - 1,
    )
    labels = ("piecewise linear", "tangent Hermite")
    colors = ("tab:blue", "tab:orange")

    figure, axes = plt.subplots(3, 3, figsize=(13.0, 9.0))
    for arm, (label, color) in enumerate(zip(labels, colors, strict=True)):
        axes[0, 0].plot(x, eta[0, arm], label=label, color=color)
        axes[0, 1].plot(x, xi[0, arm], label=label, color=color)
        axes[0, 2].plot(x, gxi[0, arm], label=label, color=color)
        axes[1, 0].plot(
            x,
            gxi[observed_index, arm],
            label=label,
            color=color,
        )
        axes[1, 1].semilogy(
            np.arange(NX // 2 + 1),
            np.abs(np.fft.rfft(eta[0, arm]) / NX) + 1e-30,
            label=label,
            color=color,
        )
        axes[1, 2].semilogy(
            np.arange(NX // 2 + 1),
            np.abs(np.fft.rfft(gxi[0, arm]) / NX) + 1e-30,
            label=label,
            color=color,
        )
        axes[2, 0].plot(
            times,
            sign_counts[:, arm],
            label=label,
            color=color,
        )
        axes[2, 1].semilogy(
            times,
            coefficient_band_norm(eta[:, arm], 80, 128) + 1e-30,
            label=label,
            color=color,
        )
        axes[2, 2].semilogy(
            times,
            coefficient_band_norm(gxi[:, arm], 80, 128) + 1e-30,
            label=label,
            color=color,
        )

    axes[0, 0].set_title(r"initial $\eta$")
    axes[0, 1].set_title(r"initial $\xi$")
    axes[0, 2].set_title(r"initial $G(\eta;h)\xi$")
    axes[1, 0].set_title(
        rf"$G(\eta;h)\xi$ at $t={times[observed_index]:.2f}$"
    )
    axes[1, 1].set_title(r"initial $|\widehat{\eta}_k|$")
    axes[1, 2].set_title(r"initial $|\widehat{G(\eta;h)\xi}_k|$")
    axes[2, 0].set_title("historical sign-transition count")
    axes[2, 1].set_title(r"$\|\widehat{\eta}_{80:128}\|_2$")
    axes[2, 2].set_title(
        r"$\|\widehat{G(\eta;h)\xi}_{80:128}\|_2$"
    )
    axes[2, 0].axhline(10, color="black", linestyle="--", linewidth=0.8)
    axes[1, 1].set_xlim(0, 160)
    axes[1, 2].set_xlim(0, 160)
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    for axis in axes[0]:
        axis.set_xlabel("x")
    axes[1, 0].set_xlabel("x")
    for axis in axes[2]:
        axis.set_xlabel("time")
    figure.suptitle(
        "Exact Tanaka case 31: piecewise-linear versus tangent-Hermite placement"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if jax.default_backend() != "cpu":
        raise RuntimeError(
            "This trial must run on CPU. Set JAX_PLATFORMS=cpu before launch."
        )
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("This trial requires JAX_ENABLE_X64=true.")
    if args.tmax <= 0.0 or args.output_dt <= 0.0:
        raise ValueError("--tmax and --output_dt must be positive.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = load_archived_frames()

    setup_start = time.perf_counter()
    solution, endpoint_diagnostics = solve_profiles()
    if endpoint_diagnostics["minimum_x_spacing"] <= 0.0:
        raise ValueError("Tanaka x-profile knots are not strictly increasing.")
    if endpoint_diagnostics["maximum_endpoint_eta"] > 1e-12:
        raise ValueError("Tanaka profile does not join zero at its endpoints.")
    if endpoint_diagnostics["maximum_endpoint_slope"] > 1e-12:
        raise ValueError("Tanaka profile slope does not join zero.")
    if endpoint_diagnostics["minimum_cos_theta"] <= 0.0:
        raise ValueError("Tanaka profile is not a graph over x.")

    control = build_initial_condition(solution, place_linear)
    tangent = build_initial_condition(solution, place_tangent_hermite)
    initial_eta = jnp.stack((control.eta, tangent.eta))
    initial_xi = jnp.stack((control.xi, tangent.xi))
    _, k = build_grid(NX, LENGTH)
    k = jnp.asarray(k, dtype=jnp.float64)
    depth_2d = jnp.full((2, 1), DEPTH, dtype=jnp.float64)
    initial_gxi = gxi_for_states(initial_eta, initial_xi, k, depth_2d)
    initial_gxi_np = np.asarray(jax.device_get(initial_gxi), dtype=np.float64)

    archive_initial_eta = np.asarray(archive["initial_eta"])
    archive_initial_xi = np.asarray(archive["initial_xi"])
    archive_initial_gxi = np.asarray(archive["initial_gxi"])
    if not np.array_equal(
        np.asarray(control.eta).astype(np.float32),
        archive_initial_eta.astype(np.float32),
    ):
        raise ValueError("Control eta does not reproduce archived float32 t=0.")
    if not np.array_equal(
        np.asarray(control.xi).astype(np.float32),
        archive_initial_xi.astype(np.float32),
    ):
        raise ValueError("Control xi does not reproduce archived float32 t=0.")
    if not np.array_equal(
        initial_gxi_np[0].astype(np.float32),
        archive_initial_gxi.astype(np.float32),
    ):
        raise ValueError(
            "Control G(eta;h)xi does not reproduce archived float32 t=0."
        )

    traveling_residuals = {
        "piecewise_linear": traveling_wave_residuals(control, k),
        "tangent_hermite": traveling_wave_residuals(tangent, k),
    }
    setup_seconds = time.perf_counter() - setup_start

    times_np = np.arange(
        0.0,
        args.tmax + 0.5 * args.output_dt,
        args.output_dt,
        dtype=np.float32,
    )
    times = jnp.asarray(times_np, dtype=jnp.float64)
    defaults = make_normalized_rollout_settings()._replace(
        filter_fraction=FILTER_FRACTION
    )
    solver_params = SolverParams(
        nx=NX,
        length=LENGTH,
        depth=depth_2d,
        gravity=GRAVITY,
        dno_order=defaults.dno_order,
        pad_factor=defaults.pad_factor,
        filter_fraction=defaults.filter_fraction,
        k=k,
        g0=make_linear_dno_symbol(k, depth_2d),
    )
    solver_params = cast_solver_params_dtype(solver_params, jnp.float64)

    rollout_start = time.perf_counter()
    payload = batched_rollout(
        cast_state_dtype(
            State(eta=initial_eta, xi=initial_xi),
            jnp.float64,
        ),
        times,
        solver_params,
        save_gxi=False,
        substeps_per_interval=defaults.substeps_per_interval,
        method=defaults.method,
        implicit_iterations=defaults.implicit_iterations,
        implicit_relaxation=defaults.implicit_relaxation,
        zero_mean_xi=defaults.zero_mean_xi,
    )
    jax.block_until_ready(payload["xi"])
    rollout_seconds = time.perf_counter() - rollout_start

    eta = np.asarray(jax.device_get(payload["eta"]), dtype=np.float64)
    xi = np.asarray(jax.device_get(payload["xi"]), dtype=np.float64)
    gxi_start = time.perf_counter()
    gxi = chunked_gxi_over_time(
        jnp.asarray(eta),
        jnp.asarray(xi),
        k,
        depth_2d,
        defaults.dno_order,
        defaults.pad_factor,
        defaults.filter_fraction,
        args.gxi_chunk_size,
    )
    gxi = np.asarray(gxi, dtype=np.float64)
    gxi_seconds = time.perf_counter() - gxi_start

    sign_counts = sign_count_history(gxi)
    timing = {
        "setup": setup_seconds,
        "rollout": rollout_seconds,
        "gxi_postprocessing": gxi_seconds,
        "total": setup_seconds + rollout_seconds + gxi_seconds,
    }
    metrics = compute_metrics(
        np.asarray(times),
        eta,
        xi,
        gxi,
        sign_counts,
        archive,
        initial_gxi_np[0],
        endpoint_diagnostics,
        traveling_residuals,
        timing,
    )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    diagnostics_path = output_dir / "diagnostics.png"
    plot_diagnostics(
        diagnostics_path,
        np.asarray(times),
        eta,
        xi,
        gxi,
        sign_counts,
    )

    reduced_path = output_dir / "diagnostics.npz"
    np.savez_compressed(
        reduced_path,
        times=np.asarray(times, dtype=np.float64),
        sign_counts=sign_counts,
        eta_80_128=coefficient_band_norm(eta, 80, 128),
        xi_80_128=coefficient_band_norm(xi, 80, 128),
        gxi_80_128=coefficient_band_norm(gxi, 80, 128),
        eta_initial=eta[0],
        xi_initial=xi[0],
        gxi_initial=gxi[0],
        eta_observed=eta[
            min(
                int(round(CRITICAL_TIME / args.output_dt)),
                eta.shape[0] - 1,
            )
        ],
        xi_observed=xi[
            min(
                int(round(CRITICAL_TIME / args.output_dt)),
                xi.shape[0] - 1,
            )
        ],
        gxi_observed=gxi[
            min(
                int(round(CRITICAL_TIME / args.output_dt)),
                gxi.shape[0] - 1,
            )
        ],
    )
    if args.save_full:
        np.savez(
            output_dir / "paired_trajectories.npz",
            times=np.asarray(times, dtype=np.float32),
            eta=eta.astype(np.float32),
            xi=xi.astype(np.float32),
            gxi=gxi.astype(np.float32),
            sign_counts=sign_counts,
        )

    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(summary_path)
    print(diagnostics_path)


if __name__ == "__main__":
    main()
