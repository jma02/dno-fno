"""Harvest off-manifold states by rolling a trained surrogate from real ICs.

For each sampled IC (time=0 row of the source dataset), this:
  1. Rolls the v5 surrogate forward in the same GL2 integrator and filter
     convention the eval_suite uses (matches the rollout the model fails on).
  2. Snapshots the state at a handful of times (default 1, 2, 5, 10, 20).
  3. Recomputes the analytical Craig-Sulem operator on each snapshot at
     order=6, pad=8, then applies the standard apply_lowpass(0.25) filter so
     the labels match the training-convention used everywhere else.
  4. Drops any snapshot whose model rollout produced a non-finite state
     (i.e. divergent ICs contribute their pre-NaN portion, but the NaN
     state itself is filtered out).

Output is a single .npz shard in the same per-batch format as
solver/gen_data/generate_shallow_steep_dataset (eta_batch_0000.npy etc.) so
build_combined_v4 can stream it into combined_dataset_v6 unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time as _time
import zipfile
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver.evals.model_rollout import (  # noqa: E402
    build_predict_gxi_with_depth,
    load_run,
    rollout_surrogate,
)
from solver.solvers import time_integrator as ti  # noqa: E402
from solver.solvers.dno_series_jax import build_grid, dno_series_eval  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True,
                   help="Run dir of the surrogate (e.g. cs_dno_w512b8_l256_v5_...).")
    p.add_argument("--source_dataset", default="data/combined_dataset_v5.npz",
                   help="Dataset to sample initial conditions from.")
    p.add_argument("--output", default="data/model_drift_v1.npz")
    p.add_argument("--n_ics", type=int, default=24_000,
                   help="Total ICs to roll.")
    p.add_argument(
        "--family_weights", type=str,
        default="tanaka_g0:3,tanaka_g1:3,shallow_steep:2,shallow_steep_mid:2,"
                "bf_modal:1,bf_g1:1,linear:2,stokes_finite:1,random_sea_finite:1",
        help="comma-list of family:weight pairs; ICs are sampled proportionally "
             "(only from rows where time==0).",
    )
    p.add_argument(
        "--snapshot_times", type=str, default="1,2,5,10,20",
        help="Comma list of physical times at which to snapshot drifted states.",
    )
    p.add_argument("--substeps", type=int, default=4,
                   help="GL2 substeps per dt (eval uses 8; reduced for harvest speed).")
    p.add_argument("--gl2_iterations", type=int, default=2)
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=16,
                   help="ICs vmapped per jit call.")
    p.add_argument("--target_order", type=int, default=6)
    p.add_argument("--target_pad_factor", type=int, default=8)
    p.add_argument("--filter_fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=20260616)
    p.add_argument("--max_abs_eta", type=float, default=10.0,
                   help="drop samples whose |eta|_max exceeds this (near-overflow filter).")
    p.add_argument("--max_abs_xi", type=float, default=10.0)
    p.add_argument("--max_abs_gxi", type=float, default=100.0)
    p.add_argument("--gravity", type=float, default=1.0)
    p.add_argument("--length", type=float, default=2.0 * math.pi)
    p.add_argument("--nx", type=int, default=1024)
    return p.parse_args()


def sample_ic_rows(args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    """Return sorted row indices drawn from time-zero rows of each requested family."""
    meta = json.loads(Path(args.source_dataset).with_suffix(".meta.json").read_text())
    name_to_id = {v: int(k) for k, v in meta["source_legend"].items()}

    weights = {}
    for tok in args.family_weights.split(","):
        name, w = tok.split(":")
        if name not in name_to_id:
            raise ValueError(f"family {name!r} not in dataset legend")
        weights[name] = float(w)
    total_w = sum(weights.values())

    with np.load(args.source_dataset) as d:
        src = np.asarray(d["source"])
        t = np.asarray(d["time"])
    zero_mask = (t == 0.0)

    chosen: list[np.ndarray] = []
    for name, w in weights.items():
        sid = name_to_id[name]
        rows = np.flatnonzero(zero_mask & (src == sid))
        if rows.size == 0:
            print(f"  warning: family {name} has no time=0 rows; skipping", flush=True)
            continue
        take = int(round(args.n_ics * w / total_w))
        take = min(take, rows.size)
        chosen.append(rng.choice(rows, size=take, replace=False))
        print(f"  {name:20s} pool={rows.size:6d}  taking {take}", flush=True)
    rows = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
    return np.sort(rows)


def build_rollout_fn(args: argparse.Namespace, predict_gxi_with_depth):
    nx, length = args.nx, args.length
    _, k_grid_np = build_grid(nx, length)
    k_grid = jnp.asarray(k_grid_np, dtype=jnp.float64)
    snapshot_times = np.asarray(
        [float(x) for x in args.snapshot_times.split(",")], dtype=np.float64,
    )
    if not (snapshot_times > 0).all():
        raise ValueError("snapshot_times must be positive")
    times = jnp.asarray(np.concatenate([[0.0], snapshot_times]), dtype=jnp.float64)
    filt = float(args.filter_fraction)

    def rollout_one(eta0, xi0, depth, log_h):
        g0 = ti.make_linear_dno_symbol(k_grid, depth)
        sp = ti.SolverParams(
            nx=nx, length=length, depth=depth, gravity=args.gravity,
            dno_order=args.target_order, pad_factor=args.target_pad_factor,
            filter_fraction=filt, k=k_grid, g0=g0,
        )
        state0 = ti.State(eta=eta0, xi=xi0)

        def predict(eta, xi):
            gxi = predict_gxi_with_depth(
                eta.astype(jnp.float32), xi.astype(jnp.float32), log_h,
            ).astype(jnp.float64)
            return ti.apply_lowpass(gxi, k_grid, filt) if filt < 1.0 else gxi

        out = rollout_surrogate(
            state0, times, sp, predict,
            substeps=args.substeps, zero_mean_xi=True,
            gl2_iterations=args.gl2_iterations,
        )
        return out["eta"], out["xi"]

    batched = jax.jit(jax.vmap(rollout_one, in_axes=(0, 0, 0, 0)))

    @jax.jit
    def analytical_targets(eta_batch, xi_batch, depth_batch):
        def one(eta, xi, h):
            gxi = dno_series_eval(
                eta, xi, k_grid, h,
                args.target_order, pad_factor=args.target_pad_factor,
            )
            return ti.apply_lowpass(gxi, k_grid, filt) if filt < 1.0 else gxi
        return jax.vmap(one, in_axes=(0, 0, 0))(eta_batch, xi_batch, depth_batch)

    return batched, analytical_targets, snapshot_times, k_grid


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists; remove it first.")

    print(f"loading run: {args.run_dir}", flush=True)
    loaded = load_run(args.run_dir)
    predict_gxi_with_depth = build_predict_gxi_with_depth(loaded)

    print("sampling ICs:", flush=True)
    ic_rows = sample_ic_rows(args, rng)
    n_ics = ic_rows.size
    print(f"total ICs: {n_ics}", flush=True)

    print("loading IC states from disk...", flush=True)
    with np.load(args.source_dataset) as d:
        eta_ics = d["eta"][ic_rows].astype(np.float64, copy=False)
        xi_ics = d["xi"][ic_rows].astype(np.float64, copy=False)
        depth_ics = d["depth"][ic_rows].astype(np.float64, copy=False)
        source_ics = d["source"][ic_rows].astype(np.int8, copy=False)
    log_h_ics = np.log(np.clip(depth_ics, 1e-12, None)).astype(np.float32)

    rollout_batched, analytical_targets, snapshot_times, _k_grid = build_rollout_fn(
        args, predict_gxi_with_depth,
    )
    n_snap = snapshot_times.size

    out_eta = np.empty((n_ics * n_snap, args.nx), dtype=np.float32)
    out_xi = np.empty((n_ics * n_snap, args.nx), dtype=np.float32)
    out_gxi = np.empty((n_ics * n_snap, args.nx), dtype=np.float32)
    out_depth = np.empty((n_ics * n_snap,), dtype=np.float32)
    out_time = np.empty((n_ics * n_snap,), dtype=np.float32)
    out_case = np.empty((n_ics * n_snap,), dtype=np.int32)
    keep_mask = np.zeros((n_ics * n_snap,), dtype=bool)

    t_start = _time.perf_counter()
    B = args.batch_size
    for batch_start in range(0, n_ics, B):
        bs = slice(batch_start, min(batch_start + B, n_ics))
        eta0_b = jnp.asarray(eta_ics[bs], dtype=jnp.float64)
        xi0_b = jnp.asarray(xi_ics[bs], dtype=jnp.float64)
        depth_b = jnp.asarray(depth_ics[bs], dtype=jnp.float64)
        log_h_b = jnp.asarray(log_h_ics[bs], dtype=jnp.float32)

        eta_traj, xi_traj = rollout_batched(eta0_b, xi0_b, depth_b, log_h_b)
        # eta_traj: (B, T=n_snap+1, nx). Drop t=0 entry.
        eta_traj = eta_traj[:, 1:, :]
        xi_traj = xi_traj[:, 1:, :]

        # Reshape to (B*n_snap, nx)
        eta_flat = eta_traj.reshape(-1, args.nx)
        xi_flat = xi_traj.reshape(-1, args.nx)
        depth_flat = jnp.repeat(depth_b, n_snap)
        gxi_flat = analytical_targets(eta_flat, xi_flat, depth_flat)

        eta_np = np.asarray(eta_flat, dtype=np.float32)
        xi_np = np.asarray(xi_flat, dtype=np.float32)
        gxi_np = np.asarray(gxi_flat, dtype=np.float32)

        # finite + magnitude filter: drop near-overflow samples whose
        # analytical Gxi blew up despite the rolled state still being finite.
        finite = (
            np.isfinite(eta_np).all(axis=1)
            & np.isfinite(xi_np).all(axis=1)
            & np.isfinite(gxi_np).all(axis=1)
            & (np.abs(eta_np).max(axis=1) < args.max_abs_eta)
            & (np.abs(xi_np).max(axis=1) < args.max_abs_xi)
            & (np.abs(gxi_np).max(axis=1) < args.max_abs_gxi)
        )

        out_idx_start = batch_start * n_snap
        out_idx_end = out_idx_start + eta_np.shape[0]
        out_eta[out_idx_start:out_idx_end] = eta_np
        out_xi[out_idx_start:out_idx_end] = xi_np
        out_gxi[out_idx_start:out_idx_end] = gxi_np
        out_depth[out_idx_start:out_idx_end] = np.repeat(
            depth_ics[bs].astype(np.float32), n_snap,
        )
        out_time[out_idx_start:out_idx_end] = np.tile(
            snapshot_times.astype(np.float32), eta_traj.shape[0],
        )
        out_case[out_idx_start:out_idx_end] = np.repeat(
            np.arange(batch_start, batch_start + eta_traj.shape[0], dtype=np.int32),
            n_snap,
        )
        keep_mask[out_idx_start:out_idx_end] = finite

        elapsed = _time.perf_counter() - t_start
        kept_so_far = int(keep_mask[:out_idx_end].sum())
        total_so_far = out_idx_end
        rate = (batch_start + B) / max(elapsed, 1e-6)
        print(
            f"  batch {batch_start // B:4d}/{(n_ics + B - 1) // B:4d}  "
            f"ICs={batch_start + B:>6d}  kept={kept_so_far:>6d}/{total_so_far:>6d} "
            f"({100 * kept_so_far / max(total_so_far, 1):.1f}%)  rate={rate:.2f} IC/s",
            flush=True,
        )

    print("filtering NaN snapshots...", flush=True)
    out_eta = out_eta[keep_mask]
    out_xi = out_xi[keep_mask]
    out_gxi = out_gxi[keep_mask]
    out_depth = out_depth[keep_mask]
    out_time = out_time[keep_mask]
    out_case = out_case[keep_mask]
    n_kept = out_eta.shape[0]
    print(f"  kept {n_kept} of {n_ics * n_snap} samples "
          f"({100 * n_kept / max(n_ics * n_snap, 1):.1f}%)", flush=True)

    print(f"writing {out_path}...", flush=True)
    x_grid, _ = build_grid(args.nx, args.length)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        def write_entry(name: str, arr: np.ndarray) -> None:
            with zf.open(name, mode="w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, arr, allow_pickle=False)
        write_entry("eta_batch_0000.npy", out_eta)
        write_entry("xi_batch_0000.npy", out_xi)
        write_entry("gxi_batch_0000.npy", out_gxi)
        write_entry("depth_batch_0000.npy", out_depth)
        write_entry("time_batch_0000.npy", out_time)
        write_entry("case_id_batch_0000.npy", out_case)
        write_entry("x.npy", np.asarray(x_grid, dtype=np.float32))
        zf.writestr("meta.json", json.dumps({
            "kind": "model_drift_shard",
            "version": "v1",
            "source_run_dir": str(args.run_dir),
            "source_dataset": str(args.source_dataset),
            "n_ics_planned": int(n_ics),
            "snapshot_times": [float(x) for x in snapshot_times],
            "n_samples_kept": int(n_kept),
            "n_samples_dropped_nan": int((~keep_mask).sum()),
            "substeps": int(args.substeps),
            "gl2_iterations": int(args.gl2_iterations),
            "dt": float(args.dt),
            "target_order": int(args.target_order),
            "target_pad_factor": int(args.target_pad_factor),
            "filter_fraction": float(args.filter_fraction),
            "seed": int(args.seed),
            "source_id_per_ic_sampling": {
                int(s): int((source_ics == s).sum()) for s in np.unique(source_ics)
            },
        }, indent=2))
    print(f"done in {_time.perf_counter() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
