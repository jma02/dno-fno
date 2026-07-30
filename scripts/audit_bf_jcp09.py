"""CPU audit of the Benjamin--Feir initial condition against JCP09 equation (33).

The script keeps the historical constructor unchanged and compares it with:

1. the same constructor without its empirical bound-mode additions;
2. a literal deep-water implementation of Xu--Guyenne equation (33); and
3. a finite-depth-consistent Stokes/Airy construction.

It also reproduces the canonical JCP09 recurrence test and checks that the
conclusions are insensitive to the fixed-point count and a halved time step.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, NamedTuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solver.data.stokes_truth_jax import _finite_depth_coeffs, stokes_eta_xi
from solver.gen_data.generate_bf_dataset import _bf_ic_single
from solver.solvers.dno_series_jax import build_grid, dno_series_eval
from solver.solvers.time_integrator import (
    SolverParams,
    State,
    apply_lowpass,
    batched_rollout,
    cast_solver_params_dtype,
    make_solver_params,
)

jax.config.update("jax_enable_x64", True)


LENGTH = 2.0 * math.pi
GRAVITY = 1.0


class BFSpec(NamedTuple):
    name: str
    n_carr: int
    n_l: int
    n_r: int
    eps_carrier: float
    eps_pert_l: float
    eps_pert_r: float
    phase_l: float
    phase_r: float
    depth: float


class Arm(NamedTuple):
    name: str
    builder: Callable[[jnp.ndarray, BFSpec], tuple[jnp.ndarray, jnp.ndarray]]


CANONICAL = BFSpec(
    name="jcp09_canonical",
    n_carr=9,
    n_l=7,
    n_r=11,
    eps_carrier=0.13,
    eps_pert_l=0.10,
    eps_pert_r=0.10,
    phase_l=-math.pi / 4.0,
    phase_r=-math.pi / 4.0,
    depth=20.0,
)

# Adaptive-g1 case 1,002,084: the last finite stored frame is t=66.48 and the
# next retained frame is nonfinite.
ARCHIVED_FAILURE = BFSpec(
    name="archived_failure_1002084",
    n_carr=13,
    n_l=11,
    n_r=15,
    eps_carrier=0.1297646297876914,
    eps_pert_l=0.1731717182394893,
    eps_pert_r=0.19090164001875853,
    phase_l=0.33810782020341706,
    phase_r=4.904384231859565,
    depth=2.2858021319838593,
)

FINITE_DEPTH_CORNER = BFSpec(
    name="finite_depth_corner",
    n_carr=4,
    n_l=2,
    n_r=6,
    eps_carrier=0.13,
    eps_pert_l=0.20,
    eps_pert_r=0.20,
    phase_l=-math.pi / 4.0,
    phase_r=-math.pi / 4.0,
    depth=0.5,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bf_jcp09_audit_20260723"),
    )
    parser.add_argument("--skip-canonical-rollout", action="store_true")
    parser.add_argument("--canonical-nx", type=int, default=64)
    parser.add_argument("--run-failure-rollout", action="store_true")
    parser.add_argument("--failure-nx", type=int, default=512)
    parser.add_argument("--failure-tmax", type=float, default=80.0)
    return parser.parse_args()


def _deep_carrier(
    x: jnp.ndarray,
    spec: BFSpec,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Fifth-order deep carrier with the historical fundamental normalization."""
    k_carr = jnp.asarray(spec.n_carr, dtype=x.dtype)
    a = jnp.asarray(spec.eps_carrier, dtype=x.dtype) / k_carr
    eps_s = jnp.asarray(spec.eps_carrier, dtype=x.dtype)
    for _ in range(8):
        factor = 1.0 + eps_s**2 / 8.0 + 121.0 * eps_s**4 / 192.0
        a0 = a / factor
        eps_s = k_carr * a0
    return stokes_eta_xi(
        x=x,
        time=jnp.asarray(0.0, dtype=x.dtype),
        n0=spec.n_carr,
        a0=a0,
        length=LENGTH,
        depth=spec.depth,
        gravity=GRAVITY,
        ichoi=0,
    )


def _finite_carrier(
    x: jnp.ndarray,
    spec: BFSpec,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Fifth-order finite-depth carrier with matched fundamental eta amplitude."""
    k_carr = jnp.asarray(spec.n_carr, dtype=x.dtype)
    target_a = jnp.asarray(spec.eps_carrier, dtype=x.dtype) / k_carr
    a0 = target_a
    for _ in range(12):
        coeffs = _finite_depth_coeffs(
            k0=k_carr,
            depth=spec.depth,
            gravity=GRAVITY,
            a0=a0,
        )
        fundamental_factor = (
            1.0
            + coeffs["eps2"] * coeffs["B31"]
            + coeffs["eps4"] * coeffs["B51"]
        )
        a0 = target_a / fundamental_factor
    return stokes_eta_xi(
        x=x,
        time=jnp.asarray(0.0, dtype=x.dtype),
        n0=spec.n_carr,
        a0=a0,
        length=LENGTH,
        depth=spec.depth,
        gravity=GRAVITY,
        ichoi=1,
    )


def _production(
    x: jnp.ndarray,
    spec: BFSpec,
    *,
    empirical_terms: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return _bf_ic_single(
        x,
        n_carr=jnp.asarray(spec.n_carr, dtype=x.dtype),
        n_l=jnp.asarray(spec.n_l, dtype=x.dtype),
        n_r=jnp.asarray(spec.n_r, dtype=x.dtype),
        eps_carrier=jnp.asarray(spec.eps_carrier, dtype=x.dtype),
        eps_pert_l=jnp.asarray(spec.eps_pert_l, dtype=x.dtype),
        eps_pert_r=jnp.asarray(spec.eps_pert_r, dtype=x.dtype),
        phase_l_extra=jnp.asarray(spec.phase_l, dtype=x.dtype),
        phase_r_extra=jnp.asarray(spec.phase_r, dtype=x.dtype),
        length=LENGTH,
        gravity=GRAVITY,
        bf_2nd_order=empirical_terms,
    )


def _jcp09_deep(
    x: jnp.ndarray,
    spec: BFSpec,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generalized literal form of JCP09 (33), with each sideband's own k."""
    eta_carr, xi_carr = _deep_carrier(x, spec)
    dtype = x.dtype
    k_l = jnp.asarray(spec.n_l, dtype=dtype)
    k_r = jnp.asarray(spec.n_r, dtype=dtype)
    a = jnp.asarray(spec.eps_carrier / spec.n_carr, dtype=dtype)
    a_l = jnp.asarray(spec.eps_pert_l, dtype=dtype) * a
    a_r = jnp.asarray(spec.eps_pert_r, dtype=dtype) * a
    phase_l = k_l * x + jnp.asarray(spec.phase_l, dtype=dtype)
    phase_r = k_r * x + jnp.asarray(spec.phase_r, dtype=dtype)
    eta = eta_carr + a_l * jnp.cos(phase_l) + a_r * jnp.cos(phase_r)
    xi_sidebands = (
        a_l * jnp.sqrt(GRAVITY / k_l) * jnp.exp(k_l * eta) * jnp.sin(phase_l)
        + a_r * jnp.sqrt(GRAVITY / k_r) * jnp.exp(k_r * eta) * jnp.sin(phase_r)
    )
    return eta, xi_carr + xi_sidebands


def _finite_depth_consistent(
    x: jnp.ndarray,
    spec: BFSpec,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Finite-depth Stokes carrier plus finite-depth linear Airy sidebands."""
    # The closed finite-depth coefficient formulas contain ratios of very large
    # hyperbolic functions and overflow numerically in their deep-water limit.
    # In that limit the deep coefficients are the same construction.
    eta_carr, xi_carr = (
        _deep_carrier(x, spec)
        if spec.n_carr * spec.depth > 20.0
        else _finite_carrier(x, spec)
    )
    dtype = x.dtype
    k_l = jnp.asarray(spec.n_l, dtype=dtype)
    k_r = jnp.asarray(spec.n_r, dtype=dtype)
    depth = jnp.asarray(spec.depth, dtype=dtype)
    a = jnp.asarray(spec.eps_carrier / spec.n_carr, dtype=dtype)
    a_l = jnp.asarray(spec.eps_pert_l, dtype=dtype) * a
    a_r = jnp.asarray(spec.eps_pert_r, dtype=dtype) * a
    phase_l = k_l * x + jnp.asarray(spec.phase_l, dtype=dtype)
    phase_r = k_r * x + jnp.asarray(spec.phase_r, dtype=dtype)
    eta = eta_carr + a_l * jnp.cos(phase_l) + a_r * jnp.cos(phase_r)

    def sideband(
        k_side: jnp.ndarray,
        amplitude: jnp.ndarray,
        phase: jnp.ndarray,
    ) -> jnp.ndarray:
        omega = jnp.sqrt(GRAVITY * k_side * jnp.tanh(k_side * depth))
        vertical_factor = (
            jnp.cosh(k_side * (eta + depth)) / jnp.cosh(k_side * depth)
        )
        return amplitude * GRAVITY / omega * vertical_factor * jnp.sin(phase)

    return (
        eta,
        xi_carr
        + sideband(k_l, a_l, phase_l)
        + sideband(k_r, a_r, phase_r),
    )


ARMS = (
    Arm(
        "production_empirical",
        lambda x, spec: _production(x, spec, empirical_terms=True),
    ),
    Arm(
        "production_no_empirical",
        lambda x, spec: _production(x, spec, empirical_terms=False),
    ),
    Arm("jcp09_deep", _jcp09_deep),
    Arm("finite_depth_consistent", _finite_depth_consistent),
)


def _as_float(value: Any) -> float:
    return float(np.asarray(jax.device_get(value)))


def _relative_l2(lhs: np.ndarray, rhs: np.ndarray) -> float:
    denominator = np.linalg.norm(rhs)
    return float(np.linalg.norm(lhs - rhs) / max(denominator, np.finfo(float).tiny))


def _fixed_band_coefficients(field: np.ndarray, k_max: int) -> np.ndarray:
    return np.fft.rfft(field)[: k_max + 1] / field.shape[-1]


def _fixed_cutoff(
    field: jnp.ndarray,
    k: jnp.ndarray,
    cutoff: float,
) -> jnp.ndarray:
    k_nyquist = _as_float(jnp.max(jnp.abs(k)))
    return apply_lowpass(field, k, min(cutoff / k_nyquist, 1.0))


def _evaluate_initial_state(
    arm: Arm,
    spec: BFSpec,
    nx: int,
    *,
    dno_order: int = 6,
    pad_factor: int = 8,
    cutoff: float = 128.0,
) -> dict[str, np.ndarray]:
    x, k = build_grid(nx, LENGTH)
    x = jnp.asarray(x, dtype=jnp.float64)
    k = jnp.asarray(k, dtype=jnp.float64)
    eta, xi = arm.builder(x, spec)
    eta = _fixed_cutoff(eta, k, cutoff)
    xi = _fixed_cutoff(xi, k, cutoff)
    gxi = dno_series_eval(
        eta,
        xi,
        k,
        spec.depth,
        dno_order,
        pad_factor=pad_factor,
    )
    gxi = _fixed_cutoff(gxi, k, cutoff)
    return {
        "eta": np.asarray(jax.device_get(eta)),
        "xi": np.asarray(jax.device_get(xi)),
        "gxi": np.asarray(jax.device_get(gxi)),
    }


def static_audit(output_dir: Path) -> dict[str, Any]:
    start = perf_counter()
    specs = (CANONICAL, ARCHIVED_FAILURE, FINITE_DEPTH_CORNER)
    resolutions = (512, 1024, 2048)
    k_max = 128
    states: dict[str, dict[str, dict[int, dict[str, np.ndarray]]]] = {}
    summary: dict[str, Any] = {}

    for spec in specs:
        states[spec.name] = {}
        summary[spec.name] = {"arms": {}, "pairwise_at_n1024": {}}
        for arm in ARMS:
            arm_states = {
                nx: _evaluate_initial_state(arm, spec, nx)
                for nx in resolutions
            }
            states[spec.name][arm.name] = arm_states
            convergence: dict[str, Any] = {}
            for field in ("eta", "xi", "gxi"):
                coefficients = {
                    nx: _fixed_band_coefficients(arm_states[nx][field], k_max)
                    for nx in resolutions
                }
                convergence[field] = {
                    "n512_vs_n1024": _relative_l2(
                        coefficients[512], coefficients[1024]
                    ),
                    "n1024_vs_n2048": _relative_l2(
                        coefficients[1024], coefficients[2048]
                    ),
                    "n512_vs_n2048": _relative_l2(
                        coefficients[512], coefficients[2048]
                    ),
                }
            summary[spec.name]["arms"][arm.name] = {
                "fixed_band_convergence": convergence,
                "maximum_abs_at_n1024": {
                    field: float(np.max(np.abs(arm_states[1024][field])))
                    for field in ("eta", "xi", "gxi")
                },
            }

        reference = states[spec.name]["jcp09_deep"][1024]
        for arm in ARMS:
            trial = states[spec.name][arm.name][1024]
            summary[spec.name]["pairwise_at_n1024"][arm.name] = {
                field: _relative_l2(trial[field], reference[field])
                for field in ("eta", "xi", "gxi")
            }

    figure, axes = plt.subplots(3, 3, figsize=(15.0, 11.0), constrained_layout=True)
    for row, spec in enumerate(specs):
        x = np.linspace(0.0, LENGTH, 1024, endpoint=False)
        for col, field in enumerate(("eta", "xi", "gxi")):
            axis = axes[row, col]
            for arm in ARMS:
                axis.plot(
                    x,
                    states[spec.name][arm.name][1024][field],
                    label=arm.name,
                    linewidth=1.0,
                )
            axis.set_title(f"{spec.name}: {field}")
            axis.set_xlabel("x")
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    figure.savefig(output_dir / "initial_condition_comparison.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(3, 3, figsize=(15.0, 11.0), constrained_layout=True)
    modes = np.arange(k_max + 1)
    for row, spec in enumerate(specs):
        for col, field in enumerate(("eta", "xi", "gxi")):
            axis = axes[row, col]
            for arm in ARMS:
                coefficients = _fixed_band_coefficients(
                    states[spec.name][arm.name][1024][field],
                    k_max,
                )
                axis.semilogy(
                    modes,
                    np.maximum(np.abs(coefficients), 1e-18),
                    label=arm.name,
                    linewidth=1.0,
                )
            axis.set_title(f"{spec.name}: |{field}_k|")
            axis.set_xlabel("Fourier mode")
            axis.set_ylim(1e-18, None)
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    figure.savefig(output_dir / "initial_condition_spectra.png", dpi=180)
    plt.close(figure)

    summary["wall_seconds"] = perf_counter() - start
    return summary


def _solver_params(
    nx: int,
    depth: float,
    *,
    dno_order: int,
    pad_factor: int,
    filter_fraction: float,
) -> SolverParams:
    return cast_solver_params_dtype(
        make_solver_params(
            nx=nx,
            length=LENGTH,
            depth=depth,
            gravity=GRAVITY,
            dno_order=dno_order,
            pad_factor=pad_factor,
            filter_fraction=filter_fraction,
        ),
        jnp.float64,
    )


def _rollout_arms(
    arms: tuple[Arm, ...],
    spec: BFSpec,
    *,
    nx: int,
    times: np.ndarray,
    substeps: int,
    dno_order: int,
    pad_factor: int,
    filter_fraction: float,
    implicit_iterations: int,
) -> dict[str, np.ndarray]:
    x, k = build_grid(nx, LENGTH)
    states = [arm.builder(jnp.asarray(x, dtype=jnp.float64), spec) for arm in arms]
    eta = jnp.stack([state[0] for state in states])
    xi = jnp.stack([state[1] for state in states])
    if filter_fraction < 1.0:
        eta = apply_lowpass(eta, jnp.asarray(k, dtype=jnp.float64), filter_fraction)
        xi = apply_lowpass(xi, jnp.asarray(k, dtype=jnp.float64), filter_fraction)
    params = _solver_params(
        nx,
        spec.depth,
        dno_order=dno_order,
        pad_factor=pad_factor,
        filter_fraction=filter_fraction,
    )
    result = batched_rollout(
        State(eta=eta, xi=xi),
        jnp.asarray(times, dtype=jnp.float64),
        params,
        save_gxi=True,
        substeps_per_interval=substeps,
        method="gl2_if",
        implicit_iterations=implicit_iterations,
        implicit_relaxation=1.0,
        zero_mean_xi=True,
    )
    return {
        key: np.asarray(jax.device_get(value))
        for key, value in result.items()
    }


def _mode_amplitude(field: np.ndarray, mode: int) -> np.ndarray:
    return 2.0 * np.abs(np.fft.rfft(field, axis=-1)[..., mode]) / field.shape[-1]


def _rollout_metrics(
    result: dict[str, np.ndarray],
    spec: BFSpec,
    arm_names: tuple[str, ...],
) -> dict[str, Any]:
    times = result["times"]
    eta = result["eta"]
    xi = result["xi"]
    gxi = result["gxi"]
    dx = LENGTH / eta.shape[-1]
    hamiltonian = 0.5 * dx * np.sum(xi * gxi + eta**2, axis=-1)
    drift = (hamiltonian - hamiltonian[:1]) / (
        np.abs(hamiltonian[:1]) + np.finfo(float).tiny
    )
    omega = math.sqrt(spec.n_carr) * (
        1.0 + 0.5 * spec.eps_carrier**2 + 5.0 * spec.eps_carrier**4 / 8.0
    )
    period = 2.0 * math.pi / omega
    metrics: dict[str, Any] = {
        "carrier_period": period,
        "arms": {},
    }
    for arm_index, arm_name in enumerate(arm_names):
        carrier = _mode_amplitude(eta[:, arm_index], spec.n_carr)
        left = _mode_amplitude(eta[:, arm_index], spec.n_l)
        right = _mode_amplitude(eta[:, arm_index], spec.n_r)
        search = np.logical_and(times / period >= 30.0, times / period <= 90.0)
        search = np.logical_and(search, np.isfinite(carrier))
        candidate_indices = np.flatnonzero(search)
        minimum_index = (
            int(candidate_indices[np.argmin(carrier[search])])
            if candidate_indices.size
            else 0
        )
        all_finite = bool(
            np.isfinite(eta[:, arm_index]).all()
            and np.isfinite(xi[:, arm_index]).all()
            and np.isfinite(gxi[:, arm_index]).all()
        )
        finite_frames = np.isfinite(eta[:, arm_index]).all(axis=-1)
        first_nonfinite = np.flatnonzero(~finite_frames)
        metrics["arms"][arm_name] = {
            "all_finite": all_finite,
            "first_nonfinite_time": (
                float(times[first_nonfinite[0]]) if first_nonfinite.size else None
            ),
            "maximum_abs_hamiltonian_drift": (
                float(np.nanmax(np.abs(drift[:, arm_index])))
                if np.isfinite(drift[:, arm_index]).any()
                else None
            ),
            "carrier_first_minimum_time": float(times[minimum_index]),
            "carrier_first_minimum_t_over_period": float(
                times[minimum_index] / period
            ),
            "carrier_first_minimum_relative_amplitude": float(
                carrier[minimum_index] / carrier[0]
            ),
            "left_maximum_relative_to_initial_carrier": float(
                np.nanmax(left) / carrier[0]
            ),
            "right_maximum_relative_to_initial_carrier": float(
                np.nanmax(right) / carrier[0]
            ),
        }
    return metrics


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def canonical_rollout_audit(
    output_dir: Path,
    *,
    nx: int,
) -> dict[str, Any]:
    start = perf_counter()
    canonical_arms = ARMS[:3]
    times = np.arange(0.0, 140.0 + 0.25, 0.5, dtype=np.float64)
    result = _rollout_arms(
        canonical_arms,
        CANONICAL,
        nx=nx,
        times=times,
        substeps=50,
        dno_order=4,
        pad_factor=3,
        filter_fraction=1.0,
        implicit_iterations=4,
    )
    arm_names = tuple(arm.name for arm in canonical_arms)
    metrics = _rollout_metrics(result, CANONICAL, arm_names)

    # JCP09 reports 2--4 fixed-point iterations to a residual tolerance. Compare
    # the historical fixed count with a stricter count and a halved inner step.
    control_times = np.arange(0.0, 20.0 + 0.25, 0.5, dtype=np.float64)
    reference_arm = (ARMS[2],)
    iter8 = _rollout_arms(
        reference_arm,
        CANONICAL,
        nx=nx,
        times=control_times,
        substeps=50,
        dno_order=4,
        pad_factor=3,
        filter_fraction=1.0,
        implicit_iterations=8,
    )
    half_dt = _rollout_arms(
        reference_arm,
        CANONICAL,
        nx=nx,
        times=control_times,
        substeps=100,
        dno_order=4,
        pad_factor=3,
        filter_fraction=1.0,
        implicit_iterations=8,
    )
    main_jcp_index = arm_names.index("jcp09_deep")
    metrics["solver_controls_at_t20"] = {
        "four_vs_eight_iterations": {
            field: _relative_l2(
                result[field][40, main_jcp_index],
                iter8[field][-1, 0],
            )
            for field in ("eta", "xi", "gxi")
        },
        "dt_0p01_vs_dt_0p005_at_eight_iterations": {
            field: _relative_l2(
                iter8[field][-1, 0],
                half_dt[field][-1, 0],
            )
            for field in ("eta", "xi", "gxi")
        },
    }

    period = metrics["carrier_period"]
    figure, axes = plt.subplots(3, 1, figsize=(10.0, 10.0), sharex=True)
    for arm_index, arm_name in enumerate(arm_names):
        carrier0 = _mode_amplitude(result["eta"][:, arm_index], CANONICAL.n_carr)[0]
        axes[0].plot(
            times / period,
            _mode_amplitude(result["eta"][:, arm_index], CANONICAL.n_carr)
            / carrier0,
            label=arm_name,
        )
        axes[1].plot(
            times / period,
            _mode_amplitude(result["eta"][:, arm_index], CANONICAL.n_l)
            / carrier0,
            label=arm_name,
        )
        axes[2].plot(
            times / period,
            _mode_amplitude(result["eta"][:, arm_index], CANONICAL.n_r)
            / carrier0,
            label=arm_name,
        )
    axes[0].set_ylabel("carrier 9 / initial carrier")
    axes[1].set_ylabel("sideband 7 / initial carrier")
    axes[2].set_ylabel("sideband 11 / initial carrier")
    axes[2].set_xlabel("t / carrier period")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.savefig(output_dir / f"canonical_recurrence_n{nx}.png", dpi=180)
    plt.close(figure)

    np.savez_compressed(
        output_dir / f"canonical_recurrence_n{nx}.npz",
        times=result["times"],
        eta=result["eta"],
        xi=result["xi"],
        gxi=result["gxi"],
        arm_names=np.asarray(arm_names),
    )
    metrics["nx"] = nx
    metrics["wall_seconds"] = perf_counter() - start
    return metrics


def failure_rollout_audit(
    output_dir: Path,
    *,
    nx: int,
    tmax: float,
) -> dict[str, Any]:
    start = perf_counter()
    cutoff = 128.0
    filter_fraction = min(cutoff / (nx / 2.0), 1.0)
    times = np.arange(0.0, tmax + 0.04, 0.08, dtype=np.float64)
    result = _rollout_arms(
        ARMS,
        ARCHIVED_FAILURE,
        nx=nx,
        times=times,
        substeps=8,
        dno_order=6,
        pad_factor=8,
        filter_fraction=filter_fraction,
        implicit_iterations=4,
    )
    arm_names = tuple(arm.name for arm in ARMS)
    metrics = _rollout_metrics(result, ARCHIVED_FAILURE, arm_names)
    metrics["nx"] = nx
    metrics["fixed_cutoff"] = cutoff

    figure, axes = plt.subplots(2, 1, figsize=(10.0, 7.0), sharex=True)
    for arm_index, arm_name in enumerate(arm_names):
        axes[0].plot(
            times,
            np.nanmax(np.abs(result["eta"][:, arm_index]), axis=-1),
            label=arm_name,
        )
        axes[1].plot(
            times,
            np.nanmax(np.abs(result["gxi"][:, arm_index]), axis=-1),
            label=arm_name,
        )
    axes[0].set_ylabel("max |eta|")
    axes[1].set_ylabel("max |G(eta)xi|")
    axes[1].set_xlabel("t")
    for axis in axes:
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.savefig(output_dir / "archived_failure_counterfactual.png", dpi=180)
    plt.close(figure)

    np.savez_compressed(
        output_dir / "archived_failure_counterfactual.npz",
        times=result["times"],
        eta=result["eta"],
        xi=result["xi"],
        gxi=result["gxi"],
        arm_names=np.asarray(arm_names),
    )
    metrics["wall_seconds"] = perf_counter() - start
    return metrics


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    total_start = perf_counter()
    payload: dict[str, Any] = {
        "device": str(jax.devices()[0]),
        "static": static_audit(output_dir),
    }
    if not args.skip_canonical_rollout:
        payload["canonical_rollout"] = canonical_rollout_audit(
            output_dir,
            nx=args.canonical_nx,
        )
    if args.run_failure_rollout:
        payload["archived_failure_rollout"] = failure_rollout_audit(
            output_dir,
            nx=args.failure_nx,
            tmax=args.failure_tmax,
        )
    payload["total_wall_seconds"] = perf_counter() - total_start
    payload = _json_ready(payload)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
