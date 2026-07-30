"""Generate a deep-water Benjamin--Feir dataset on the ``L=2 pi`` domain.

Mirrors generate_tanaka_dataset_v2.py: same rollout settings, same NPZ/state.json
sidecar machinery, same flatten layout (eta, xi, gxi, time, case_id, depth per
sample).  The initial condition has the form of Xu--Guyenne JCP09 equation
(33): the project's fifth-order deep-water Stokes carrier plus two symmetric
Airy sidebands.  Integer carrier/sideband pairs are sampled only inside the
leading deep-water modulational-instability band.

Defaults match the v2 Tanaka production config: dt=0.08, tmax=200, B=256,
keep_samples=200, filter_fraction=1/4, float64. Throughput ~0.07 s/sample on an
RTX 6000 Ada (memory-bandwidth saturated at B=256, same as Tanaka).
"""
from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from numpy.lib import format as npy_format

from .benjamin_feir_jcp09 import (
    CARRIER_MODE_MAX,
    CARRIER_MODE_MIN,
    CARRIER_STEEPNESS_MAX,
    CARRIER_STEEPNESS_MIN,
    DEEP_WATER_MINIMUM_KH,
    PERTURBATION_RATIO_MAX,
    PERTURBATION_RATIO_MIN,
    build_initial_conditions,
    sample_parameters,
    serialize_parameters,
)
from ..solvers.dno_series_jax import build_grid, dno_series_eval, make_linear_dno_symbol
from .adaptive_sampling import adaptive_indices_from_energy
from ..solvers.time_integrator import (
    SolverParams,
    State,
    apply_lowpass,
    batched_rollout,
    cast_solver_params_dtype,
    cast_state_dtype,
    make_normalized_rollout_settings,
)

jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Benjamin-Feir rollout dataset on the [0, 2pi] reference domain."
    )
    parser.add_argument("--output", default="data/bf_jcp09_deep.npz")
    parser.add_argument("--target_samples", type=int, default=10_000_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep_samples", type=int, default=200)
    parser.add_argument("--gxi_chunk_size", type=int, default=16)
    parser.add_argument("--case_id_offset", type=int, default=0)
    parser.add_argument("--rng_stream_id", type=int)
    parser.add_argument(
        "--rollout_dtype", choices=("float32", "float64"), default="float64",
        help="float64 needed at L=2pi: k_max=512 with M=6 amplifies float32 eps to amplitude order, NaN'ing the rollout.",
    )
    parser.add_argument("--n_carr_min", type=int, default=CARRIER_MODE_MIN)
    parser.add_argument("--n_carr_max", type=int, default=CARRIER_MODE_MAX)
    parser.add_argument(
        "--eps_carrier_min",
        type=float,
        default=CARRIER_STEEPNESS_MIN,
    )
    parser.add_argument(
        "--eps_carrier_max",
        type=float,
        default=CARRIER_STEEPNESS_MAX,
    )
    parser.add_argument(
        "--eps_pert_min",
        type=float,
        default=PERTURBATION_RATIO_MIN,
    )
    parser.add_argument(
        "--eps_pert_max",
        type=float,
        default=PERTURBATION_RATIO_MAX,
    )
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=2.0 * math.pi)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--tmax", type=float, default=200.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--adaptive", action="store_true",
        help="Use focusing-aware per-case temporal subsampling driven by |d||η_x||²/dt|/||η_x||² "
             "(captures BF envelope recurrence events).",
    )
    parser.add_argument("--adaptive_alpha", type=float, default=0.5)
    parser.add_argument("--adaptive_smooth_sigma", type=float, default=10.0,
                        help="Smoothing in dense-time steps. BF dt=0.08 so 10 steps ~ 0.8 s.")
    parser.add_argument(
        "--adaptive_signal",
        choices=("envelope", "grad_energy"),
        default="envelope",
        help="Activity proxy. 'envelope' = ||eta||_inf (BF-friendly); 'grad_energy' = ||eta_x||^2.",
    )
    parser.add_argument(
        "--adaptive_density_mode",
        choices=("rate", "magnitude"),
        default="magnitude",
        help="'magnitude' (default for BF) uses S directly so samples cluster at envelope peaks. "
             "'rate' uses |dS/dt|/S — better for localized events.",
    )
    parser.add_argument(
        "--adaptive_power",
        type=float,
        default=4.0,
        help="Sharpens contrast by raising (signal - per_case_min) to this power. "
             "BF envelope only modulates ~20%%, so power=4 is needed to bias sampling toward peaks.",
    )
    return parser.parse_args()


def select_time_indices(n_times: int, keep_samples: int) -> np.ndarray:
    if keep_samples <= 0:
        raise ValueError("--keep_samples must be positive.")
    if keep_samples >= n_times:
        return np.arange(n_times, dtype=np.int32)
    return np.linspace(0, n_times - 1, keep_samples, dtype=np.int32)


def chunked_gxi_over_time(
    eta: jnp.ndarray,
    xi: jnp.ndarray,
    k: jnp.ndarray,
    depth_per_case: jnp.ndarray,  # shape (B, 1)
    dno_order: int,
    pad_factor: int,
    filter_fraction: float,
    chunk_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    n_steps = int(eta.shape[0])
    for start_idx in range(0, n_steps, chunk_size):
        end_idx = min(start_idx + chunk_size, n_steps)
        gxi_chunk = dno_series_eval(
            eta[start_idx:end_idx],
            xi[start_idx:end_idx],
            k,
            depth_per_case,
            dno_order,
            pad_factor=pad_factor,
        )
        gxi_chunk = apply_lowpass(gxi_chunk, k, filter_fraction)
        chunks.append(np.asarray(jax.device_get(gxi_chunk)))
    return np.concatenate(chunks, axis=0)


def flatten_samples(
    eta: np.ndarray,
    xi: np.ndarray,
    gxi: np.ndarray,
    times: np.ndarray,
    global_case_ids: np.ndarray,
    depth_per_case: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eta_case_major = np.swapaxes(eta, 0, 1).reshape(-1, eta.shape[-1])
    xi_case_major = np.swapaxes(xi, 0, 1).reshape(-1, xi.shape[-1])
    gxi_case_major = np.swapaxes(gxi, 0, 1).reshape(-1, gxi.shape[-1])
    sample_times = np.tile(times, global_case_ids.shape[0])
    sample_case_ids = np.repeat(global_case_ids, times.shape[0])
    sample_depths = np.repeat(depth_per_case, times.shape[0])
    return eta_case_major, xi_case_major, gxi_case_major, sample_times, sample_case_ids, sample_depths


def flatten_samples_per_case(
    eta_BKN: np.ndarray, xi_BKN: np.ndarray, gxi_BKN: np.ndarray,
    times_BK: np.ndarray, global_case_ids: np.ndarray, depth_per_case: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nx = eta_BKN.shape[-1]
    k_per_case = eta_BKN.shape[1]
    return (
        eta_BKN.reshape(-1, nx),
        xi_BKN.reshape(-1, nx),
        gxi_BKN.reshape(-1, nx),
        times_BK.reshape(-1),
        np.repeat(global_case_ids, k_per_case),
        np.repeat(depth_per_case, k_per_case),
    )


@jax.jit
def _grad_energy_traj(eta_traj: jnp.ndarray, length: float) -> jnp.ndarray:
    n = eta_traj.shape[-1]
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=length / n)

    def step(_, eta_t: jnp.ndarray) -> tuple[None, jnp.ndarray]:
        eta_hat = jnp.fft.fft(eta_t, axis=-1)
        eta_x = jnp.real(jnp.fft.ifft(1j * k * eta_hat, axis=-1))
        return None, jnp.sum(eta_x ** 2, axis=-1)

    _, s_TB = jax.lax.scan(step, None, eta_traj)
    return jnp.swapaxes(s_TB, 0, 1)


@jax.jit
def _envelope_peak_traj(eta_traj: jnp.ndarray) -> jnp.ndarray:
    """Per-case ``||η(t)||_∞`` for ``eta_traj`` of shape (T, B, N). Returns (B, T)."""
    return jnp.swapaxes(jnp.max(jnp.abs(eta_traj), axis=-1), 0, 1)


def _gather_per_case_jax(traj: jnp.ndarray, idx_BK: jnp.ndarray) -> jnp.ndarray:
    traj_BT = jnp.swapaxes(traj, 0, 1)
    return jnp.take_along_axis(traj_BT, idx_BK[:, :, None], axis=1)


def write_npy_entry(zf: zipfile.ZipFile, name: str, array: np.ndarray) -> None:
    with zf.open(name, mode="w", force_zip64=True) as handle:
        npy_format.write_array(handle, np.asarray(array), allow_pickle=False)


def load_state(state_path: Path) -> dict[str, object] | None:
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(state_path: Path, state: dict[str, object]) -> None:
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def make_batch_rng(seed: int, batch_idx: int, rng_stream_id: int) -> np.random.Generator:
    seed_sequence = np.random.SeedSequence([seed, rng_stream_id, batch_idx])
    return np.random.default_rng(seed_sequence)


def sample_log_uniform(rng: np.random.Generator, lo: float, hi: float, shape: tuple[int, ...]) -> np.ndarray:
    return np.exp(rng.uniform(math.log(lo), math.log(hi), size=shape))


def sample_bf_case_params(
    rng: np.random.Generator,
    *,
    batch_size: int,
    n_carr_min: int,
    n_carr_max: int,
    n_carr_set: tuple[int, ...] | None,
    eps_carrier_min: float,
    eps_carrier_max: float,
    side_offset_min: int,
    side_offset_max: int,
    eps_pert_min: float,
    eps_pert_max: float,
    depth_min: float,
    depth_max: float,
) -> dict[str, np.ndarray]:
    """Sample the historical finite-depth/empirical BF generator.

    Kept only so archived-data audits can reconstruct the old corpus.  New
    generation uses :func:`sample_parameters`.
    """

    if n_carr_set:
        n_carr = np.asarray(rng.choice(np.asarray(n_carr_set, dtype=np.int32), size=batch_size), dtype=np.int32)
    else:
        n_carr = rng.integers(n_carr_min, n_carr_max + 1, size=batch_size).astype(np.int32)
    side_offset = rng.integers(side_offset_min, side_offset_max + 1, size=batch_size).astype(np.int32)
    # Clamp side_offset so that n_l = n_carr - side_offset >= 1; n_l = 0 produces a
    # DC "sideband" (constant offset to eta), which is not a Benjamin-Feir setup.
    side_offset = np.minimum(side_offset, n_carr - 1).astype(np.int32)
    # n_carr=10 with side_offset=2 (sidebands {8,12}) hits a 2nd-harmonic-resonance regime
    # at eps_carrier >= 0.11 that NaN'd ~5% of cases in the bf_2 production run. Bump to
    # side_offset >= 3 specifically for n_carr=10 to kill that failure mode at root.
    nc10_offset_floor = 3
    side_offset = np.where((n_carr == 10) & (side_offset < nc10_offset_floor),
                           nc10_offset_floor, side_offset).astype(np.int32)
    n_l = (n_carr - side_offset).astype(np.int32)
    n_r = (n_carr + side_offset).astype(np.int32)
    eps_carrier = rng.uniform(eps_carrier_min, eps_carrier_max, size=batch_size).astype(np.float64)
    eps_pert_l = rng.uniform(eps_pert_min, eps_pert_max, size=batch_size).astype(np.float64)
    eps_pert_r = rng.uniform(eps_pert_min, eps_pert_max, size=batch_size).astype(np.float64)
    phase_l = rng.uniform(0.0, 2.0 * math.pi, size=batch_size).astype(np.float64)
    phase_r = rng.uniform(0.0, 2.0 * math.pi, size=batch_size).astype(np.float64)
    depth = sample_log_uniform(rng, depth_min, depth_max, (batch_size,)).astype(np.float64)
    return {
        "n_carr": n_carr, "n_l": n_l, "n_r": n_r, "side_offset": side_offset,
        "eps_carrier": eps_carrier, "eps_pert_l": eps_pert_l, "eps_pert_r": eps_pert_r,
        "phase_l": phase_l, "phase_r": phase_r, "depth": depth,
    }


def serialize_bf_specs(params: dict[str, np.ndarray]) -> list[dict[str, float | int]]:
    """Serialize historical empirical-constructor parameters."""

    n = int(params["n_carr"].shape[0])
    return [
        {
            "n_carr": int(params["n_carr"][i]),
            "n_l": int(params["n_l"][i]),
            "n_r": int(params["n_r"][i]),
            "side_offset": int(params["side_offset"][i]),
            "eps_carrier": float(params["eps_carrier"][i]),
            "eps_pert_l": float(params["eps_pert_l"][i]),
            "eps_pert_r": float(params["eps_pert_r"][i]),
            "phase_l": float(params["phase_l"][i]),
            "phase_r": float(params["phase_r"][i]),
            "depth": float(params["depth"][i]),
        }
        for i in range(n)
    ]


def _bf_ic_single(
    x: jnp.ndarray,
    *,
    n_carr: jnp.ndarray, n_l: jnp.ndarray, n_r: jnp.ndarray,
    eps_carrier: jnp.ndarray, eps_pert_l: jnp.ndarray, eps_pert_r: jnp.ndarray,
    phase_l_extra: jnp.ndarray, phase_r_extra: jnp.ndarray,
    length: float, gravity: float, bf_2nd_order: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build one historical empirical BF state for archive reconstruction."""
    k_carr = n_carr.astype(x.dtype) * (2.0 * jnp.pi / length)
    k_l = n_l.astype(x.dtype) * (2.0 * jnp.pi / length)
    k_r = n_r.astype(x.dtype) * (2.0 * jnp.pi / length)
    a = eps_carrier / k_carr
    a_l = eps_pert_l * a
    a_r = eps_pert_r * a

    # Renormalize Stokes parameter so eta_fund == a. 4 fixed-point iterations,
    # unrolled (jax.vmap can't carry a Python loop with .item()-style branches).
    eps_s = eps_carrier
    for _ in range(4):
        F = 1.0 + eps_s**2 / 8.0 + 121.0 * eps_s**4 / 192.0
        a0 = a / F
        eps_s = k_carr * a0
    eps2, eps3, eps4 = eps_s**2, eps_s**3, eps_s**4

    theta_c = k_carr * x
    eta_c = a0 * (
        (1.0 + eps2 / 8.0 + 121.0 * eps4 / 192.0) * jnp.cos(theta_c)
        + (0.5 * eps_s + 5.0 * eps3 / 6.0) * jnp.cos(2.0 * theta_c)
        + (3.0 * eps2 / 8.0 + 171.0 * eps4 / 128.0) * jnp.cos(3.0 * theta_c)
        + (eps3 / 3.0) * jnp.cos(4.0 * theta_c)
        + (125.0 * eps4 / 384.0) * jnp.cos(5.0 * theta_c)
    )

    phase_l = k_l * x + phase_l_extra
    phase_r = k_r * x + phase_r_extra
    eta_sb = a_l * jnp.cos(phase_l) + a_r * jnp.cos(phase_r)

    if bf_2nd_order:
        eta_2nd = (k_carr / 2.0) * a * (
            a_l * jnp.cos(theta_c + phase_l) + a_r * jnp.cos(theta_c + phase_r)
        )
        eta_3rd = (3.0 * k_carr**2 / 8.0) * a * a * (
            a_l * jnp.cos(2.0 * theta_c + phase_l) + a_r * jnp.cos(2.0 * theta_c + phase_r)
        )
    else:
        eta_2nd = jnp.zeros_like(x)
        eta_3rd = jnp.zeros_like(x)

    eta_total = eta_c + eta_sb + eta_2nd + eta_3rd

    om_c = jnp.sqrt(gravity * k_carr)
    xi_c = (a0 * om_c / k_carr) * (
        jnp.exp(k_carr * eta_total) * jnp.sin(theta_c)
        + 0.5 * eps3 * jnp.exp(2.0 * k_carr * eta_total) * jnp.sin(2.0 * theta_c)
        + (eps4 / 12.0) * jnp.exp(3.0 * k_carr * eta_total) * jnp.sin(3.0 * theta_c)
    )
    xi_sb = (om_c / k_carr) * (
        a_l * jnp.exp(k_l * eta_total) * jnp.sin(phase_l)
        + a_r * jnp.exp(k_r * eta_total) * jnp.sin(phase_r)
    )

    if bf_2nd_order:
        K_l = (k_carr - k_l) / jnp.sqrt(gravity * k_carr)
        K_r = (k_r - k_carr) / jnp.sqrt(gravity * k_carr)
        xi_2nd = (
            K_l * a * a_l * jnp.cos(theta_c) * jnp.sin(phase_l)
            - K_r * a * a_r * jnp.cos(theta_c) * jnp.sin(phase_r)
        )
    else:
        xi_2nd = jnp.zeros_like(x)

    return eta_total, xi_c + xi_sb + xi_2nd


def build_bf_initial_conditions_batched(
    *,
    x: jnp.ndarray,
    params: dict[str, np.ndarray],
    length: float,
    gravity: float,
    bf_2nd_order: bool,
    dtype: jnp.dtype,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorize the historical empirical constructor for archive audits."""
    n_carr = jnp.asarray(params["n_carr"], dtype=dtype)
    n_l = jnp.asarray(params["n_l"], dtype=dtype)
    n_r = jnp.asarray(params["n_r"], dtype=dtype)
    eps_carrier = jnp.asarray(params["eps_carrier"], dtype=dtype)
    eps_pert_l = jnp.asarray(params["eps_pert_l"], dtype=dtype)
    eps_pert_r = jnp.asarray(params["eps_pert_r"], dtype=dtype)
    phase_l = jnp.asarray(params["phase_l"], dtype=dtype)
    phase_r = jnp.asarray(params["phase_r"], dtype=dtype)
    x_d = x.astype(dtype)

    def per_case(nc, nl, nr, ec, epl, epr, phl, phr):
        return _bf_ic_single(
            x_d, n_carr=nc, n_l=nl, n_r=nr,
            eps_carrier=ec, eps_pert_l=epl, eps_pert_r=epr,
            phase_l_extra=phl, phase_r_extra=phr,
            length=length, gravity=gravity, bf_2nd_order=bf_2nd_order,
        )

    eta, xi = jax.vmap(per_case)(n_carr, n_l, n_r, eps_carrier, eps_pert_l, eps_pert_r, phase_l, phase_r)
    return eta, xi


def main() -> None:
    args = parse_args()
    rollout_defaults = make_normalized_rollout_settings()._replace(filter_fraction=0.25)
    rng_stream_id = args.rng_stream_id if args.rng_stream_id is not None else args.case_id_offset

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = output_path.with_suffix(".state.json")

    if args.overwrite:
        if output_path.exists():
            output_path.unlink()
        if state_path.exists():
            state_path.unlink()

    times = np.arange(0.0, args.tmax + 0.5 * args.dt, args.dt, dtype=np.float32)
    if args.adaptive:
        if args.keep_samples <= 1:
            raise SystemExit("--adaptive requires --keep_samples >= 2")
        subsample_indices = None
        subsample_times = None
        samples_per_batch = int(args.batch_size * args.keep_samples)
    else:
        subsample_indices = select_time_indices(times.shape[0], args.keep_samples)
        subsample_times = times[subsample_indices]
        samples_per_batch = int(args.batch_size * subsample_times.shape[0])
    n_batches = int(math.ceil(args.target_samples / samples_per_batch))

    existing_state = load_state(state_path)
    if existing_state is None:
        samples_written = 0
        next_batch = 0
        with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            x_grid, _ = build_grid(args.nx, args.length)
            meta = {
                "dataset_kind": "bf_jcp09_deep_rollout_v1",
                "initial_condition": (
                    "Xu--Guyenne JCP09 equation (33) with the project "
                    "fifth-order deep-water Stokes carrier"
                ),
                "target_samples": args.target_samples,
                "batch_size": args.batch_size,
                "keep_samples": args.keep_samples,
                "samples_per_full_batch": samples_per_batch,
                "n_batches_planned": n_batches,
                "rollout_dtype": args.rollout_dtype,
                "dt": args.dt,
                "tmax": args.tmax,
                "n_time_samples_full": int(times.shape[0]),
                "n_time_samples_subsampled": int(args.keep_samples) if args.adaptive else int(subsample_times.shape[0]),
                "adaptive_sampling": bool(args.adaptive),
                "adaptive_alpha": float(args.adaptive_alpha) if args.adaptive else None,
                "adaptive_smooth_sigma": float(args.adaptive_smooth_sigma) if args.adaptive else None,
                "adaptive_signal": args.adaptive_signal if args.adaptive else None,
                "adaptive_density_mode": args.adaptive_density_mode if args.adaptive else None,
                "adaptive_power": float(args.adaptive_power) if args.adaptive else None,
                "method": rollout_defaults.method,
                "substeps": rollout_defaults.substeps_per_interval,
                "implicit_iterations": rollout_defaults.implicit_iterations,
                "filter_fraction": rollout_defaults.filter_fraction,
                "zero_mean_xi": rollout_defaults.zero_mean_xi,
                "n_carr_min": args.n_carr_min, "n_carr_max": args.n_carr_max,
                "eps_carrier_min": args.eps_carrier_min, "eps_carrier_max": args.eps_carrier_max,
                "eps_pert_min": args.eps_pert_min, "eps_pert_max": args.eps_pert_max,
                "mode_pair_distribution": "uniform_over_feasible_pairs",
                "instability_support": "0 < (Delta n/n_c)/(2 sqrt(2) epsilon_c) < 1",
                "sideband_symmetry": "equal amplitude and common phase",
                "minimum_resolved_kh": DEEP_WATER_MINIMUM_KH,
                "depth_rule": "h = minimum_resolved_kh * L / (2 pi)",
                "empirical_cross_terms": False,
                "seed": args.seed,
                "case_id_offset": args.case_id_offset,
                "rng_stream_id": rng_stream_id,
                "nx": args.nx,
                "length": args.length,
                "gravity": args.gravity,
            }
            write_npy_entry(zf, "x.npy", np.asarray(x_grid, dtype=np.float32))
            if not args.adaptive:
                write_npy_entry(zf, "subsample_indices.npy", subsample_indices)
                write_npy_entry(zf, "subsample_times.npy", subsample_times)
            zf.writestr("meta.json", json.dumps(meta, indent=2))
        save_state(
            state_path,
            {
                "output_path": str(output_path),
                "samples_written": 0,
                "next_batch": 0,
                "n_batches_planned": n_batches,
                "complete": False,
            },
        )
    else:
        samples_written = int(existing_state["samples_written"])
        next_batch = int(existing_state["next_batch"])
        if bool(existing_state.get("complete", False)) or samples_written >= args.target_samples:
            print(json.dumps(existing_state, indent=2))
            return

    rollout_dtype = jnp.float32 if args.rollout_dtype == "float32" else jnp.float64
    rollout_times = jnp.asarray(times, dtype=rollout_dtype)
    x_grid, k_grid = build_grid(args.nx, args.length)
    x_grid = jnp.asarray(x_grid, dtype=rollout_dtype)
    k_grid = jnp.asarray(k_grid, dtype=rollout_dtype)

    total_start = perf_counter()
    for batch_idx in range(next_batch, n_batches):
        batch_start = perf_counter()
        batch_rng = make_batch_rng(args.seed, batch_idx, rng_stream_id)

        params = sample_parameters(
            batch_rng,
            batch_size=args.batch_size,
            length=args.length,
            carrier_mode_min=args.n_carr_min,
            carrier_mode_max=args.n_carr_max,
            carrier_steepness_min=args.eps_carrier_min,
            carrier_steepness_max=args.eps_carrier_max,
            perturbation_ratio_min=args.eps_pert_min,
            perturbation_ratio_max=args.eps_pert_max,
        )

        initial_eta, initial_xi = build_initial_conditions(
            x=x_grid,
            parameters=params,
            length=args.length,
            gravity=args.gravity,
            dtype=rollout_dtype,
        )
        if rollout_defaults.zero_mean_xi:
            initial_xi = initial_xi - jnp.mean(initial_xi, axis=-1, keepdims=True)
        initial_eta = apply_lowpass(initial_eta, k_grid, rollout_defaults.filter_fraction)
        initial_xi = apply_lowpass(initial_xi, k_grid, rollout_defaults.filter_fraction)
        if rollout_defaults.zero_mean_xi:
            initial_xi = initial_xi - jnp.mean(initial_xi, axis=-1, keepdims=True)

        depth_2d = jnp.asarray(params["depth"], dtype=rollout_dtype)[:, None]
        g0_per_case = make_linear_dno_symbol(k_grid, depth_2d)
        solver_params = SolverParams(
            nx=args.nx, length=args.length, depth=depth_2d, gravity=args.gravity,
            dno_order=rollout_defaults.dno_order, pad_factor=rollout_defaults.pad_factor,
            filter_fraction=rollout_defaults.filter_fraction, k=k_grid, g0=g0_per_case,
        )
        solver_params = cast_solver_params_dtype(solver_params, rollout_dtype)

        rollout_payload = batched_rollout(
            cast_state_dtype(State(eta=initial_eta, xi=initial_xi), rollout_dtype),
            rollout_times, solver_params, save_gxi=False,
            substeps_per_interval=rollout_defaults.substeps_per_interval,
            method=rollout_defaults.method,
            implicit_iterations=rollout_defaults.implicit_iterations,
            implicit_relaxation=rollout_defaults.implicit_relaxation,
            zero_mean_xi=rollout_defaults.zero_mean_xi,
        )
        jax.block_until_ready(rollout_payload["xi"])

        if args.adaptive:
            if args.adaptive_signal == "envelope":
                signal_BT = np.asarray(jax.device_get(_envelope_peak_traj(rollout_payload["eta"])))
            else:
                signal_BT = np.asarray(jax.device_get(_grad_energy_traj(rollout_payload["eta"], float(args.length))))
            idx_BK = adaptive_indices_from_energy(
                signal_BT,
                keep_samples=int(args.keep_samples),
                alpha=float(args.adaptive_alpha),
                smooth_sigma_steps=float(args.adaptive_smooth_sigma),
                density_mode=args.adaptive_density_mode,
                power=float(args.adaptive_power),
            )
            idx_BK_j = jnp.asarray(idx_BK)
            pred_eta_BKN = jax.device_get(_gather_per_case_jax(rollout_payload["eta"], idx_BK_j))
            pred_xi_BKN = jax.device_get(_gather_per_case_jax(rollout_payload["xi"], idx_BK_j))
            pred_eta_KBN = np.swapaxes(pred_eta_BKN, 0, 1)
            pred_xi_KBN = np.swapaxes(pred_xi_BKN, 0, 1)
            pred_gxi_KBN = chunked_gxi_over_time(
                jnp.asarray(pred_eta_KBN), jnp.asarray(pred_xi_KBN),
                k_grid, depth_2d,
                rollout_defaults.dno_order, rollout_defaults.pad_factor,
                rollout_defaults.filter_fraction, args.gxi_chunk_size,
            )
            pred_gxi_BKN = np.swapaxes(np.asarray(pred_gxi_KBN), 0, 1)
            times_BK = times[idx_BK].astype(np.float32)
            pred_eta_arr = pred_eta_BKN.astype(np.float32)
            pred_xi_arr = pred_xi_BKN.astype(np.float32)
            pred_gxi_arr = pred_gxi_BKN.astype(np.float32)
        else:
            pred_eta_native = jax.device_get(rollout_payload["eta"][subsample_indices])
            pred_xi_native = jax.device_get(rollout_payload["xi"][subsample_indices])
            pred_gxi_native = chunked_gxi_over_time(
                jnp.asarray(pred_eta_native), jnp.asarray(pred_xi_native),
                k_grid, depth_2d,
                rollout_defaults.dno_order, rollout_defaults.pad_factor,
                rollout_defaults.filter_fraction, args.gxi_chunk_size,
            )
            pred_eta_arr = np.asarray(pred_eta_native, dtype=np.float32)
            pred_xi_arr = np.asarray(pred_xi_native, dtype=np.float32)
            pred_gxi_arr = np.asarray(pred_gxi_native, dtype=np.float32)

        global_case_ids = np.arange(
            args.case_id_offset + batch_idx * args.batch_size,
            args.case_id_offset + (batch_idx + 1) * args.batch_size,
            dtype=np.int64,
        )
        if args.adaptive:
            eta_samples, xi_samples, gxi_samples, time_samples, case_id_samples, depth_samples = flatten_samples_per_case(
                pred_eta_arr, pred_xi_arr, pred_gxi_arr, times_BK, global_case_ids,
                np.asarray(params["depth"], dtype=np.float32),
            )
        else:
            eta_samples, xi_samples, gxi_samples, time_samples, case_id_samples, depth_samples = flatten_samples(
                pred_eta_arr, pred_xi_arr, pred_gxi_arr, subsample_times, global_case_ids,
                np.asarray(params["depth"], dtype=np.float32),
            )

        remaining = args.target_samples - samples_written
        keep = min(remaining, eta_samples.shape[0])
        eta_samples = eta_samples[:keep]
        xi_samples = xi_samples[:keep]
        gxi_samples = gxi_samples[:keep]
        time_samples = time_samples[:keep]
        case_id_samples = case_id_samples[:keep]
        depth_samples = depth_samples[:keep]

        with zipfile.ZipFile(output_path, mode="a", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            batch_tag = f"batch_{batch_idx:04d}"
            write_npy_entry(zf, f"eta_{batch_tag}.npy", eta_samples)
            write_npy_entry(zf, f"xi_{batch_tag}.npy", xi_samples)
            write_npy_entry(zf, f"gxi_{batch_tag}.npy", gxi_samples)
            write_npy_entry(zf, f"time_{batch_tag}.npy", time_samples)
            write_npy_entry(zf, f"case_id_{batch_tag}.npy", case_id_samples)
            write_npy_entry(zf, f"depth_{batch_tag}.npy", depth_samples)
            if args.adaptive:
                write_npy_entry(zf, f"subsample_indices_{batch_tag}.npy",
                                 idx_BK[:global_case_ids.shape[0]].astype(np.int32))
            zf.writestr(
                f"specs_{batch_tag}.json",
                json.dumps(serialize_parameters(params), indent=2),
            )

        samples_written += keep
        batch_seconds = perf_counter() - batch_start
        save_state(
            state_path,
            {
                "output_path": str(output_path),
                "samples_written": samples_written,
                "next_batch": batch_idx + 1,
                "n_batches_planned": n_batches,
                "complete": samples_written >= args.target_samples,
                "last_batch_seconds": batch_seconds,
                "elapsed_seconds": perf_counter() - total_start,
            },
        )
        print(
            json.dumps(
                {
                    "batch_idx": batch_idx,
                    "samples_written": samples_written,
                    "target_samples": args.target_samples,
                    "batch_seconds": batch_seconds,
                    "device": str(jax.devices()[0]),
                }
            ),
            flush=True,
        )
        if samples_written >= args.target_samples:
            break


if __name__ == "__main__":
    main()
