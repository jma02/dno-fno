"""Generate a shallow-steep wavetrain rollout dataset on the L=2pi reference domain.

Coverage target: the model's one systematic gap — shallow water (h ~ 0.02-0.12)
at moderate steepness (a/h up to ~0.25), NON-soliton states. v3's only data below
h=0.1 is tanaka solitons; the eval "linear" regime (h=0.038 wavetrains) showed the
model over-drives the harmonic cascade and diverges there.

Mirrors generate_bf_dataset.py: same rollout settings (filter_fraction=1/4, f64),
same NPZ/state.json sidecar machinery, same flatten layout. Differs in:
* IC builder: multi-mode wavetrains (1..n_modes_max cosines near a dominant mode
  chosen by target kh), per-mode finite-depth linear xi transfer with random
  direction signs, total amplitude set by a/h, hard cap on total steepness ka.
* Per-case truth-health rejection: cases whose rollout drifts in Hamiltonian
  energy, goes non-finite, or approaches the bottom are dropped entirely —
  the truth integrator itself fails above a/h ~ 0.2 at these depths, and such
  trajectories must not become training labels.
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

jax.config.update("jax_enable_x64", True)

from ..solvers.dno_series_jax import build_grid, make_linear_dno_symbol  # noqa: E402  (x64 flag must precede solver imports)
from ..solvers.time_integrator import (  # noqa: E402
    SolverParams,
    State,
    apply_lowpass,
    batched_rollout,
    cast_solver_params_dtype,
    cast_state_dtype,
    make_normalized_rollout_settings,
)
from .generate_bf_dataset import (  # noqa: E402
    chunked_gxi_over_time,
    flatten_samples,
    load_state,
    make_batch_rng,
    sample_log_uniform,
    save_state,
    select_time_indices,
    write_npy_entry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a shallow-steep wavetrain rollout dataset on the [0, 2pi] reference domain."
    )
    parser.add_argument("--output", default="data/shallow_steep.npz")
    parser.add_argument("--target_samples", type=int, default=480_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--keep_samples", type=int, default=120)
    parser.add_argument("--gxi_chunk_size", type=int, default=16)
    parser.add_argument("--case_id_offset", type=int, default=0)
    parser.add_argument("--rng_stream_id", type=int)
    parser.add_argument(
        "--rollout_dtype", choices=("float32", "float64"), default="float64",
        help="float64 needed at L=2pi: k_max=512 with M=6 amplifies float32 eps to amplitude order.",
    )
    parser.add_argument("--depth_min", type=float, default=0.02)
    parser.add_argument("--depth_max", type=float, default=0.12)
    parser.add_argument(
        "--kh_min", type=float, default=0.2,
        help="Dominant-mode kh target lower bound (shallow-to-intermediate regime).",
    )
    parser.add_argument("--kh_max", type=float, default=1.5)
    parser.add_argument("--a_over_h_min", type=float, default=0.03)
    parser.add_argument(
        "--a_over_h_max", type=float, default=0.25,
        help="Total nominal amplitude / depth. Truth fails above ~0.2 at the shallow end; "
             "the energy-drift gate drops those cases.",
    )
    parser.add_argument(
        "--ka_max", type=float, default=0.15,
        help="Hard cap on total steepness sum(n_m * a_m); amplitudes are rescaled down to meet it.",
    )
    parser.add_argument("--n_modes_max", type=int, default=5)
    parser.add_argument("--n_mode_cap", type=int, default=64,
                        help="Largest allowed mode number (filter cutoff is 128).")
    parser.add_argument(
        "--drift_tol", type=float, default=1e-5,
        help="Per-case max |H(t)-H(0)|/|H(0)| over kept snapshots; healthy rollouts sit at ~1e-7.",
    )
    parser.add_argument("--bottom_frac", type=float, default=0.9,
                        help="Reject case if min eta <= -bottom_frac * h at any kept snapshot.")
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=2.0 * math.pi)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--tmax", type=float, default=120.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sample_wavetrain_case_params(
    rng: np.random.Generator,
    *,
    batch_size: int,
    depth_min: float,
    depth_max: float,
    kh_min: float,
    kh_max: float,
    a_over_h_min: float,
    a_over_h_max: float,
    ka_max: float,
    n_modes_max: int,
    n_mode_cap: int,
) -> dict[str, np.ndarray]:
    depth = sample_log_uniform(rng, depth_min, depth_max, (batch_size,)).astype(np.float64)
    kh_target = rng.uniform(kh_min, kh_max, size=batch_size).astype(np.float64)
    n0 = np.clip(np.round(kh_target / depth), 2, n_mode_cap).astype(np.int32)

    n_modes = rng.integers(1, n_modes_max + 1, size=batch_size).astype(np.int32)
    lo = np.maximum(2, n0 // 2).astype(np.float64)
    hi = np.minimum(n_mode_cap, np.ceil(1.8 * n0)).astype(np.float64)
    u = rng.uniform(0.0, 1.0, size=(batch_size, n_modes_max))
    modes = np.round(lo[:, None] + u * (hi - lo)[:, None]).astype(np.int32)
    modes[:, 0] = n0

    mode_mask = np.arange(n_modes_max)[None, :] < n_modes[:, None]
    weights = rng.uniform(0.3, 1.0, size=(batch_size, n_modes_max)) * mode_mask
    weights /= weights.sum(axis=1, keepdims=True)

    a_over_h = rng.uniform(a_over_h_min, a_over_h_max, size=batch_size).astype(np.float64)
    amps = (a_over_h * depth)[:, None] * weights
    steepness = (modes * amps).sum(axis=1)
    ka_scale = np.minimum(1.0, ka_max / np.maximum(steepness, 1e-12))
    amps *= ka_scale[:, None]

    phases = rng.uniform(0.0, 2.0 * math.pi, size=(batch_size, n_modes_max)).astype(np.float64)
    dirs = rng.choice(np.asarray([-1.0, 1.0]), size=(batch_size, n_modes_max)).astype(np.float64)
    return {
        "depth": depth, "kh_target": kh_target, "n_modes": n_modes, "modes": modes,
        "amps": amps, "phases": phases, "dirs": dirs,
        "a_over_h_nominal": a_over_h, "ka_scale": ka_scale,
    }


def serialize_specs(
    params: dict[str, np.ndarray],
    drift: np.ndarray,
    kept: np.ndarray,
) -> list[dict[str, object]]:
    n = int(params["depth"].shape[0])
    return [
        {
            "depth": float(params["depth"][i]),
            "kh_target": float(params["kh_target"][i]),
            "n_modes": int(params["n_modes"][i]),
            "modes": [int(m) for m in params["modes"][i, : params["n_modes"][i]]],
            "amps": [float(a) for a in params["amps"][i, : params["n_modes"][i]]],
            "dirs": [float(d) for d in params["dirs"][i, : params["n_modes"][i]]],
            "a_over_h_nominal": float(params["a_over_h_nominal"][i]),
            "ka_scale": float(params["ka_scale"][i]),
            "energy_drift_max": float(drift[i]) if np.isfinite(drift[i]) else None,
            "kept": bool(kept[i]),
        }
        for i in range(n)
    ]


def _wavetrain_ic_single(
    x: jnp.ndarray,
    modes: jnp.ndarray,
    amps: jnp.ndarray,
    phases: jnp.ndarray,
    dirs: jnp.ndarray,
    depth: jnp.ndarray,
    gravity: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """eta = sum_m a_m cos(n_m x + phi_m); xi from the finite-depth linear
    transfer per mode, dir_m = +-1 picking the propagation branch. Padded modes
    carry a_m = 0 and contribute nothing."""
    theta = modes[:, None] * x[None, :] + phases[:, None]
    eta = jnp.sum(amps[:, None] * jnp.cos(theta), axis=0)
    t = jnp.tanh(modes * depth)
    g0_sym = modes * t
    omega = jnp.sqrt(gravity * g0_sym)
    xi = jnp.sum((dirs * amps * omega / g0_sym)[:, None] * jnp.sin(theta), axis=0)
    return eta, xi


def build_wavetrain_initial_conditions_batched(
    x: jnp.ndarray,
    params: dict[str, np.ndarray],
    gravity: float,
    dtype: jnp.dtype,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    modes = jnp.asarray(params["modes"], dtype=dtype)
    amps = jnp.asarray(params["amps"], dtype=dtype)
    phases = jnp.asarray(params["phases"], dtype=dtype)
    dirs = jnp.asarray(params["dirs"], dtype=dtype)
    depth = jnp.asarray(params["depth"], dtype=dtype)
    x_d = x.astype(dtype)

    def per_case(n, a, ph, d, h):
        return _wavetrain_ic_single(x_d, n, a, ph, d, h, gravity)

    return jax.vmap(per_case)(modes, amps, phases, dirs, depth)


def case_energy_drift(
    eta_KBN: np.ndarray,
    xi_KBN: np.ndarray,
    gxi_KBN: np.ndarray,
    gravity: float,
    dx: float,
) -> np.ndarray:
    """Max relative Hamiltonian drift per case over kept snapshots; NaN-poisoned
    cases return inf so the gate drops them."""
    H = 0.5 * np.sum(xi_KBN * gxi_KBN + gravity * eta_KBN**2, axis=-1) * dx  # (K, B)
    drift = np.abs(H - H[:1]) / (np.abs(H[:1]) + 1e-30)
    drift[~np.isfinite(drift)] = np.inf
    return drift.max(axis=0)


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
    subsample_indices = select_time_indices(times.shape[0], args.keep_samples)
    subsample_times = times[subsample_indices]
    samples_per_batch = int(args.batch_size * subsample_times.shape[0])
    # Rejection shrinks batches below samples_per_batch; 1.5x headroom on planned batches.
    n_batches = int(math.ceil(1.5 * args.target_samples / samples_per_batch))

    existing_state = load_state(state_path)
    if existing_state is None:
        samples_written = 0
        cases_rejected = 0
        next_batch = 0
        with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            x_grid, _ = build_grid(args.nx, args.length)
            meta = {
                "dataset_kind": "shallow_steep_rollout_v1",
                "target_samples": args.target_samples,
                "batch_size": args.batch_size,
                "keep_samples": args.keep_samples,
                "samples_per_full_batch": samples_per_batch,
                "n_batches_planned": n_batches,
                "rollout_dtype": args.rollout_dtype,
                "dt": args.dt,
                "tmax": args.tmax,
                "n_time_samples_full": int(times.shape[0]),
                "n_time_samples_subsampled": int(subsample_times.shape[0]),
                "method": rollout_defaults.method,
                "substeps": rollout_defaults.substeps_per_interval,
                "implicit_iterations": rollout_defaults.implicit_iterations,
                "filter_fraction": rollout_defaults.filter_fraction,
                "zero_mean_xi": rollout_defaults.zero_mean_xi,
                "depth_min": args.depth_min, "depth_max": args.depth_max,
                "depth_distribution": "log_uniform",
                "kh_min": args.kh_min, "kh_max": args.kh_max,
                "a_over_h_min": args.a_over_h_min, "a_over_h_max": args.a_over_h_max,
                "ka_max": args.ka_max,
                "n_modes_max": args.n_modes_max, "n_mode_cap": args.n_mode_cap,
                "drift_tol": args.drift_tol, "bottom_frac": args.bottom_frac,
                "seed": args.seed,
                "case_id_offset": args.case_id_offset,
                "rng_stream_id": rng_stream_id,
                "nx": args.nx,
                "length": args.length,
                "gravity": args.gravity,
            }
            write_npy_entry(zf, "x.npy", np.asarray(x_grid, dtype=np.float32))
            write_npy_entry(zf, "subsample_indices.npy", subsample_indices)
            write_npy_entry(zf, "subsample_times.npy", subsample_times)
            zf.writestr("meta.json", json.dumps(meta, indent=2))
        save_state(
            state_path,
            {
                "output_path": str(output_path),
                "samples_written": 0,
                "cases_rejected": 0,
                "next_batch": 0,
                "n_batches_planned": n_batches,
                "complete": False,
            },
        )
    else:
        samples_written = int(existing_state["samples_written"])
        cases_rejected = int(existing_state.get("cases_rejected", 0))
        next_batch = int(existing_state["next_batch"])
        if bool(existing_state.get("complete", False)) or samples_written >= args.target_samples:
            print(json.dumps(existing_state, indent=2))
            return

    rollout_dtype = jnp.float32 if args.rollout_dtype == "float32" else jnp.float64
    rollout_times = jnp.asarray(times, dtype=rollout_dtype)
    x_grid, k_grid = build_grid(args.nx, args.length)
    x_grid = jnp.asarray(x_grid, dtype=rollout_dtype)
    k_grid = jnp.asarray(k_grid, dtype=rollout_dtype)
    dx = args.length / args.nx

    total_start = perf_counter()
    for batch_idx in range(next_batch, n_batches):
        batch_start = perf_counter()
        batch_rng = make_batch_rng(args.seed, batch_idx, rng_stream_id)

        params = sample_wavetrain_case_params(
            batch_rng, batch_size=args.batch_size,
            depth_min=args.depth_min, depth_max=args.depth_max,
            kh_min=args.kh_min, kh_max=args.kh_max,
            a_over_h_min=args.a_over_h_min, a_over_h_max=args.a_over_h_max,
            ka_max=args.ka_max,
            n_modes_max=args.n_modes_max, n_mode_cap=args.n_mode_cap,
        )

        initial_eta, initial_xi = build_wavetrain_initial_conditions_batched(
            x=x_grid, params=params, gravity=args.gravity, dtype=rollout_dtype,
        )
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

        pred_eta_native = jax.device_get(rollout_payload["eta"][subsample_indices])
        pred_xi_native = jax.device_get(rollout_payload["xi"][subsample_indices])
        pred_gxi_native = chunked_gxi_over_time(
            jnp.asarray(pred_eta_native), jnp.asarray(pred_xi_native),
            k_grid, depth_2d,
            rollout_defaults.dno_order, rollout_defaults.pad_factor,
            rollout_defaults.filter_fraction, args.gxi_chunk_size,
        )

        drift = case_energy_drift(
            pred_eta_native, pred_xi_native, np.asarray(pred_gxi_native),
            args.gravity, dx,
        )
        finite_ok = (
            np.isfinite(pred_eta_native).all(axis=(0, 2))
            & np.isfinite(pred_xi_native).all(axis=(0, 2))
            & np.isfinite(np.asarray(pred_gxi_native)).all(axis=(0, 2))
        )
        min_eta = np.min(pred_eta_native, axis=(0, 2))
        bottom_ok = min_eta > -args.bottom_frac * params["depth"]
        kept = finite_ok & bottom_ok & (drift <= args.drift_tol)

        global_case_ids = np.arange(
            args.case_id_offset + batch_idx * args.batch_size,
            args.case_id_offset + (batch_idx + 1) * args.batch_size,
            dtype=np.int64,
        )

        eta_samples, xi_samples, gxi_samples, time_samples, case_id_samples, depth_samples = flatten_samples(
            pred_eta_native[:, kept].astype(np.float32),
            pred_xi_native[:, kept].astype(np.float32),
            np.asarray(pred_gxi_native)[:, kept].astype(np.float32),
            subsample_times, global_case_ids[kept],
            np.asarray(params["depth"], dtype=np.float32)[kept],
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
            zf.writestr(f"specs_{batch_tag}.json", json.dumps(serialize_specs(params, drift, kept), indent=2))

        samples_written += keep
        cases_rejected += int((~kept).sum())
        batch_seconds = perf_counter() - batch_start
        save_state(
            state_path,
            {
                "output_path": str(output_path),
                "samples_written": samples_written,
                "cases_rejected": cases_rejected,
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
                    "kept_cases": int(kept.sum()),
                    "rejected_cases": int((~kept).sum()),
                    "drift_median": float(np.median(drift[np.isfinite(drift)])) if np.isfinite(drift).any() else None,
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
