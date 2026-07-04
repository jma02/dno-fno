"""Generate a steep-Tanaka rollout dataset targeting the a/h ∈ [0.25, 0.45] failure regime.

Coverage target: the a/h ≈ 0.28 Tanaka manifold where every trained CS-DNO surrogate
NaNs on rollout (v5, v7, v8, v8-fftfp64, v9-jacreg all fail at case_id=5, h=0.276;
several also at case_id=11, h=0.234). The v7 tanaka failure audit measured only
1.0% of training rows at h ≥ 0.23 ∧ a/h ≥ 0.27, so a dedicated pack at
h ∈ [0.20, 0.35], a/h ∈ [0.25, 0.45] fills the hole.

Structure mirrors ``generate_tanaka_dataset_v2`` (fresh Tanaka ICs via
``solve_modified_tanaka_batched``, ``filter_fraction=0.25`` label convention
matching v3/v5/v8) but adds the per-case truth-health rejection gate from
``generate_shallow_steep_dataset``: cases whose Hamiltonian drifts, goes
non-finite, or hits the bottom are dropped. Truth is known to fail above
a/h ≈ 0.4 at these depths (v2 stress-grid backfill row 2026-05-02), and those
trajectories must not become training labels.

Single-crest cases only; steepness knob directly sets a/h for the crest.
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

from ..gen_data.multi_crest import (  # noqa: E402
    sample_sum_budgeted_cases,
    serialize_case_specs,
)
from ..solvers.dno_series_jax import build_grid, make_linear_dno_symbol  # noqa: E402
from ..solvers.time_integrator import (  # noqa: E402
    SolverParams,
    State,
    apply_lowpass,
    batched_rollout,
    cast_solver_params_dtype,
    cast_state_dtype,
    make_normalized_rollout_settings,
)
from ..tanaka_ICs.modified_tanaka import make_default_tanaka_template  # noqa: E402
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
from .generate_tanaka_dataset_v2 import build_per_case_initial_conditions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/steep_tanaka_v2.npz")
    parser.add_argument("--target_samples", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--keep_samples", type=int, default=26,
                        help="Uniform temporal subsampling; 26 matches tanaka_2 convention.")
    parser.add_argument("--gxi_chunk_size", type=int, default=16)
    parser.add_argument("--case_id_offset", type=int, default=0)
    parser.add_argument("--rng_stream_id", type=int)
    parser.add_argument("--rollout_dtype", choices=("float32", "float64"), default="float64",
                        help="float64 needed at L=2π: k_max=512 with M=6 amplifies float32 eps to amplitude order.")
    parser.add_argument("--depth_min", type=float, default=0.20)
    parser.add_argument("--depth_max", type=float, default=0.35)
    parser.add_argument("--steepness_min", type=float, default=0.25,
                        help="Case-level a/h lower bound. With max_crests=1, this IS the crest steepness.")
    parser.add_argument("--steepness_max", type=float, default=0.45)
    parser.add_argument("--min_crests", type=int, default=1)
    parser.add_argument("--max_crests", type=int, default=1)
    parser.add_argument("--per_crest_steepness_floor", type=float, default=0.20)
    parser.add_argument("--separation_widths", type=float, default=3.0)
    parser.add_argument("--drift_tol", type=float, default=1e-3,
                        help="Per-case max |H(t)-H(0)|/|H(0)| over kept snapshots; healthy ~1e-7. "
                             "Truth becomes unstable above a/h~0.4 at these depths (v2 stress-grid).")
    parser.add_argument("--bottom_frac", type=float, default=0.9,
                        help="Reject case if min eta <= -bottom_frac * h at any kept snapshot.")
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=2.0 * math.pi)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tmax", type=float, default=200.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def case_energy_drift(
    eta_KBN: np.ndarray,
    xi_KBN: np.ndarray,
    gxi_KBN: np.ndarray,
    gravity: float,
    dx: float,
) -> np.ndarray:
    """Max relative Hamiltonian drift per case over kept snapshots; NaN cases → inf."""
    H = 0.5 * np.sum(xi_KBN * gxi_KBN + gravity * eta_KBN**2, axis=-1) * dx  # (K, B)
    drift = np.abs(H - H[:1]) / (np.abs(H[:1]) + 1e-30)
    drift[~np.isfinite(drift)] = np.inf
    return drift.max(axis=0)


def _serialize_specs_with_diagnostics(case_specs, drift: np.ndarray, kept: np.ndarray) -> list[dict]:
    out = []
    for i, crests in enumerate(serialize_case_specs(case_specs)):
        out.append({
            "crests": crests,
            "energy_drift_max": float(drift[i]) if np.isfinite(drift[i]) else None,
            "kept": bool(kept[i]),
        })
    return out


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
    n_batches = int(math.ceil(1.5 * args.target_samples / samples_per_batch))

    existing_state = load_state(state_path)
    if existing_state is None:
        samples_written = 0
        cases_rejected = 0
        next_batch = 0
        with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            x_grid, _ = build_grid(args.nx, args.length)
            meta = {
                "dataset_kind": "steep_tanaka_v2",
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
                "steepness_min": args.steepness_min, "steepness_max": args.steepness_max,
                "min_crests": args.min_crests, "max_crests": args.max_crests,
                "per_crest_steepness_floor": args.per_crest_steepness_floor,
                "separation_widths": args.separation_widths,
                "drift_tol": args.drift_tol,
                "bottom_frac": args.bottom_frac,
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
        save_state(state_path, {
            "output_path": str(output_path),
            "samples_written": 0,
            "cases_rejected": 0,
            "next_batch": 0,
            "n_batches_planned": n_batches,
            "complete": False,
        })
    else:
        samples_written = int(existing_state["samples_written"])
        cases_rejected = int(existing_state.get("cases_rejected", 0))
        next_batch = int(existing_state["next_batch"])
        if bool(existing_state.get("complete", False)) or samples_written >= args.target_samples:
            print(json.dumps(existing_state, indent=2))
            return

    template_params = make_default_tanaka_template(
        depth=1.0, gravity=args.gravity, direction=1,
        nx=args.nx, length=args.length, center=0.0,
        dno_order=rollout_defaults.dno_order, pad_factor=rollout_defaults.pad_factor,
    )

    rollout_dtype = jnp.float32 if args.rollout_dtype == "float32" else jnp.float64
    rollout_times = jnp.asarray(times, dtype=rollout_dtype)
    _, k_grid = build_grid(args.nx, args.length)
    k_grid = jnp.asarray(k_grid, dtype=rollout_dtype)
    dx = args.length / args.nx

    total_start = perf_counter()
    for batch_idx in range(next_batch, n_batches):
        batch_start = perf_counter()
        batch_rng = make_batch_rng(args.seed, batch_idx, rng_stream_id)

        case_h_ref = sample_log_uniform(batch_rng, args.depth_min, args.depth_max, (args.batch_size,))

        case_specs: list[list] = []
        for hi in case_h_ref:
            min_sep = args.separation_widths * float(hi)
            n_max = int(max(1, math.floor(args.length / min_sep)))
            case_max_crests = min(args.max_crests, n_max)
            case_min_crests = min(args.min_crests, case_max_crests)
            specs_i = sample_sum_budgeted_cases(
                batch_rng, 1,
                length=args.length,
                case_amplitude_min=args.steepness_min,
                case_amplitude_max=args.steepness_max,
                min_crests=case_min_crests, max_crests=case_max_crests,
                min_separation=min_sep,
                per_crest_floor=args.per_crest_steepness_floor,
            )
            case_specs.extend(specs_i)

        initial_eta, initial_xi = build_per_case_initial_conditions(
            template_params=template_params,
            case_h_ref=case_h_ref,
            case_specs=case_specs,
            length=args.length,
            nx=args.nx,
            gravity=args.gravity,
        )
        if rollout_defaults.zero_mean_xi:
            initial_xi = initial_xi - jnp.mean(initial_xi, axis=-1, keepdims=True)

        initial_eta = apply_lowpass(initial_eta, k_grid, rollout_defaults.filter_fraction)
        initial_xi = apply_lowpass(initial_xi, k_grid, rollout_defaults.filter_fraction)
        if rollout_defaults.zero_mean_xi:
            initial_xi = initial_xi - jnp.mean(initial_xi, axis=-1, keepdims=True)

        depth_2d = jnp.asarray(case_h_ref, dtype=rollout_dtype)[:, None]
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
        pred_gxi_np = np.asarray(pred_gxi_native)

        drift = case_energy_drift(pred_eta_native, pred_xi_native, pred_gxi_np, args.gravity, dx)
        finite_ok = (
            np.isfinite(pred_eta_native).all(axis=(0, 2))
            & np.isfinite(pred_xi_native).all(axis=(0, 2))
            & np.isfinite(pred_gxi_np).all(axis=(0, 2))
        )
        min_eta = np.min(pred_eta_native, axis=(0, 2))
        bottom_ok = min_eta > -args.bottom_frac * case_h_ref
        kept = finite_ok & bottom_ok & (drift <= args.drift_tol)

        global_case_ids = np.arange(
            args.case_id_offset + batch_idx * args.batch_size,
            args.case_id_offset + (batch_idx + 1) * args.batch_size,
            dtype=np.int64,
        )

        eta_samples, xi_samples, gxi_samples, time_samples, case_id_samples, depth_samples = flatten_samples(
            pred_eta_native[:, kept].astype(np.float32),
            pred_xi_native[:, kept].astype(np.float32),
            pred_gxi_np[:, kept].astype(np.float32),
            subsample_times, global_case_ids[kept],
            np.asarray(case_h_ref, dtype=np.float32)[kept],
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
            zf.writestr(f"specs_{batch_tag}.json",
                        json.dumps(_serialize_specs_with_diagnostics(case_specs, drift, kept), indent=2))

        samples_written += keep
        cases_rejected += int((~kept).sum())
        batch_seconds = perf_counter() - batch_start
        save_state(state_path, {
            "output_path": str(output_path),
            "samples_written": samples_written,
            "cases_rejected": cases_rejected,
            "next_batch": batch_idx + 1,
            "n_batches_planned": n_batches,
            "complete": samples_written >= args.target_samples,
            "last_batch_seconds": batch_seconds,
            "elapsed_seconds": perf_counter() - total_start,
        })
        print(json.dumps({
            "batch_idx": batch_idx,
            "samples_written": samples_written,
            "kept_cases": int(kept.sum()),
            "rejected_cases": int((~kept).sum()),
            "drift_median": float(np.median(drift[np.isfinite(drift)])) if np.isfinite(drift).any() else None,
            "drift_p95": float(np.quantile(drift[np.isfinite(drift)], 0.95)) if np.isfinite(drift).any() else None,
            "batch_seconds": batch_seconds,
            "device": str(jax.devices()[0]),
        }), flush=True)
        if samples_written >= args.target_samples:
            break


if __name__ == "__main__":
    main()
