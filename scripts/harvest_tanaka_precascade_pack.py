"""Build a small targeted off-manifold Tanaka pre-cascade training shard.

This targets the corrected C2 residual failure population directly:

- tanaka_g0 ICs 5, 22, and 11;
- finite snapshots before the known NaN intervals;
- controlled k=32..128 perturbations around those snapshots;
- order-6 Craig-Sulem labels with the same 0.25 low-pass convention as training.

The output is a shard-format `.npz` containing `eta_batch_0000.npy`,
`xi_batch_0000.npy`, `gxi_batch_0000.npy`, `depth_batch_0000.npy`,
`time_batch_0000.npy`, `case_id_batch_0000.npy`, `x.npy`, and `meta.json`.
It can be appended with the existing combined-dataset builders.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver.evals.eval_suite import REGISTRY, RegimeConfig, build_ics_and_truth_targets  # noqa: E402
from solver.evals.model_rollout import build_predict_gxi_with_depth, load_run, rollout_surrogate  # noqa: E402
from solver.solvers import time_integrator as ti  # noqa: E402
from solver.solvers.dno_series_jax import build_grid, dno_series_eval, make_linear_dno_symbol  # noqa: E402


DEFAULT_SNAPSHOTS: dict[int, tuple[float, ...]] = {
    # Last finite saved frames from the C2 forensic audit were:
    # cid5 t=33.6, cid22 t=45.6, cid11 t=189.6.
    5: (8.0, 16.0, 24.0, 28.8, 32.0, 33.6),
    22: (8.0, 16.0, 24.0, 32.0, 40.0, 44.0, 45.6),
    11: (80.0, 120.0, 160.0, 180.0, 188.0, 189.6),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", default="outputs/c2_stage_match_gain_from_v85b_20260707_053707")
    p.add_argument("--checkpoint", choices=("best", "final"), default="final")
    p.add_argument("--output", default="data/tanaka_precascade_c2_v1.npz")
    p.add_argument("--case_ids", type=int, nargs="+", default=sorted(DEFAULT_SNAPSHOTS))
    p.add_argument("--n_perturb_per_state", type=int, default=256)
    p.add_argument("--perturb_rel_min", type=float, default=1e-6)
    p.add_argument("--perturb_rel_max", type=float, default=1e-3)
    p.add_argument("--perturb_k_lo", type=float, default=32.0)
    p.add_argument("--perturb_k_hi", type=float, default=128.0)
    p.add_argument("--target_order", type=int, default=6)
    p.add_argument("--target_pad_factor", type=int, default=8)
    p.add_argument("--filter_fraction", type=float, default=0.25)
    p.add_argument("--target_batch_size", type=int, default=128)
    p.add_argument("--max_abs_eta", type=float, default=5.0)
    p.add_argument("--max_abs_xi", type=float, default=20.0)
    p.add_argument("--max_abs_gxi", type=float, default=200.0)
    p.add_argument("--seed", type=int, default=20260708)
    p.add_argument("--length", type=float, default=2.0 * math.pi)
    p.add_argument("--nx", type=int, default=1024)
    return p.parse_args()


def _write_shard(
    out_path: Path,
    *,
    eta: np.ndarray,
    xi: np.ndarray,
    gxi: np.ndarray,
    depth: np.ndarray,
    sample_time: np.ndarray,
    case_id: np.ndarray,
    x_grid: np.ndarray,
    meta: dict[str, object],
) -> None:
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        def write_entry(name: str, arr: np.ndarray) -> None:
            with zf.open(name, mode="w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, arr, allow_pickle=False)

        write_entry("eta_batch_0000.npy", eta.astype(np.float32, copy=False))
        write_entry("xi_batch_0000.npy", xi.astype(np.float32, copy=False))
        write_entry("gxi_batch_0000.npy", gxi.astype(np.float32, copy=False))
        write_entry("depth_batch_0000.npy", depth.astype(np.float32, copy=False))
        write_entry("time_batch_0000.npy", sample_time.astype(np.float32, copy=False))
        write_entry("case_id_batch_0000.npy", case_id.astype(np.int32, copy=False))
        write_entry("x.npy", x_grid.astype(np.float32, copy=False))
        zf.writestr("meta.json", json.dumps(meta, indent=2))


def _bandlimited_noise(
    rng: np.random.Generator,
    *,
    nx: int,
    k_abs: np.ndarray,
    k_lo: float,
    k_hi: float,
) -> np.ndarray:
    coeff = np.zeros((nx // 2 + 1,), dtype=np.complex128)
    mask = (k_abs >= k_lo) & (k_abs < k_hi)
    real = rng.normal(size=int(mask.sum()))
    imag = rng.normal(size=int(mask.sum()))
    coeff[mask] = real + 1j * imag
    coeff[0] = 0.0
    if nx % 2 == 0:
        coeff[-1] = coeff[-1].real + 0.0j
    noise = np.fft.irfft(coeff, n=nx).astype(np.float64)
    rms = float(np.sqrt(np.mean(noise**2)))
    return noise / max(rms, 1e-30)


def _augment_states(
    base_eta: np.ndarray,
    base_xi: np.ndarray,
    base_depth: np.ndarray,
    base_time: np.ndarray,
    base_case: np.ndarray,
    args: argparse.Namespace,
    k_abs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    rows_eta: list[np.ndarray] = []
    rows_xi: list[np.ndarray] = []
    rows_depth: list[float] = []
    rows_time: list[float] = []
    rows_case: list[int] = []

    for eta0, xi0, depth0, time0, case0 in zip(base_eta, base_xi, base_depth, base_time, base_case, strict=True):
        eta_rms = float(np.sqrt(np.mean(eta0**2)))
        xi_rms = float(np.sqrt(np.mean(xi0**2)))
        rows_eta.append(eta0)
        rows_xi.append(xi0 - xi0.mean())
        rows_depth.append(float(depth0))
        rows_time.append(float(time0))
        rows_case.append(int(case0))

        for perturb_idx in range(args.n_perturb_per_state):
            amp = float(np.exp(rng.uniform(np.log(args.perturb_rel_min), np.log(args.perturb_rel_max))))
            eta_noise = _bandlimited_noise(
                rng, nx=args.nx, k_abs=k_abs, k_lo=args.perturb_k_lo, k_hi=args.perturb_k_hi,
            )
            xi_noise = _bandlimited_noise(
                rng, nx=args.nx, k_abs=k_abs, k_lo=args.perturb_k_lo, k_hi=args.perturb_k_hi,
            )
            rows_eta.append(eta0 + amp * max(eta_rms, 1e-6) * eta_noise)
            rows_xi.append(xi0 + amp * max(xi_rms, 1e-6) * xi_noise)
            rows_xi[-1] = rows_xi[-1] - rows_xi[-1].mean()
            rows_depth.append(float(depth0))
            rows_time.append(float(time0))
            rows_case.append(int(case0) * 1_000_000 + perturb_idx + 1)

    return (
        np.asarray(rows_eta, dtype=np.float64),
        np.asarray(rows_xi, dtype=np.float64),
        np.asarray(rows_depth, dtype=np.float64),
        np.asarray(rows_time, dtype=np.float64),
        np.asarray(rows_case, dtype=np.int32),
    )


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists; remove it first.")

    t0 = time.perf_counter()
    loaded = load_run(args.run_dir, checkpoint=args.checkpoint)
    predict_gxi = build_predict_gxi_with_depth(loaded)

    base_cfg = REGISTRY["tanaka_g0"]
    n_ics = max(args.case_ids) + 1
    cfg = RegimeConfig(
        name=base_cfg.name,
        source=base_cfg.source,
        dt=base_cfg.dt,
        tmax=base_cfg.tmax,
        n_ics=n_ics,
        substeps=base_cfg.substeps,
        implicit_iters=base_cfg.implicit_iters,
        filter_fraction=base_cfg.filter_fraction,
        truth_kind=base_cfg.truth_kind,
    )
    ics, _, _, _, _ = build_ics_and_truth_targets(cfg)
    ic_by_case = {int(ic.case_id): ic for ic in ics}

    x_grid_np, k_grid_np = build_grid(args.nx, args.length)
    k_grid = jnp.asarray(k_grid_np, dtype=jnp.float64)
    k_abs = np.abs(k_grid_np[: args.nx // 2 + 1])

    base_eta_rows: list[np.ndarray] = []
    base_xi_rows: list[np.ndarray] = []
    base_depth_rows: list[float] = []
    base_time_rows: list[float] = []
    base_case_rows: list[int] = []

    for case_id in args.case_ids:
        ic = ic_by_case[case_id]
        snapshots = np.asarray(DEFAULT_SNAPSHOTS[case_id], dtype=np.float64)
        times = np.arange(0.0, snapshots.max() + 0.5 * base_cfg.dt, base_cfg.dt, dtype=np.float64)
        depth = float(ic.depth)
        depth_j = jnp.asarray(depth, dtype=jnp.float64)
        log_depth = jnp.asarray(np.log(depth), dtype=jnp.float32)
        g0 = make_linear_dno_symbol(k_grid, depth_j)
        params = ti.SolverParams(
            nx=args.nx,
            length=args.length,
            depth=depth_j,
            gravity=1.0,
            dno_order=args.target_order,
            pad_factor=args.target_pad_factor,
            filter_fraction=args.filter_fraction,
            k=k_grid,
            g0=g0,
        )

        def predict(eta: jnp.ndarray, xi: jnp.ndarray) -> jnp.ndarray:
            gxi = predict_gxi(
                eta.astype(jnp.float32), xi.astype(jnp.float32), log_depth,
            ).astype(jnp.float64)
            return ti.apply_lowpass(gxi, k_grid, args.filter_fraction)

        print(f"rolling case {case_id} to t={times[-1]:.1f} ({len(times)} saved frames)", flush=True)
        out = rollout_surrogate(
            ti.State(
                eta=jnp.asarray(ic.eta, dtype=jnp.float64),
                xi=jnp.asarray(ic.xi, dtype=jnp.float64),
            ),
            jnp.asarray(times, dtype=jnp.float64),
            params,
            predict,
            substeps=base_cfg.substeps,
            gl2_iterations=base_cfg.implicit_iters,
            zero_mean_xi=True,
        )
        eta_traj = np.asarray(out["eta"], dtype=np.float64)
        xi_traj = np.asarray(out["xi"], dtype=np.float64)
        for snap in snapshots:
            idx = int(np.argmin(np.abs(times - snap)))
            eta = eta_traj[idx]
            xi = xi_traj[idx]
            if not (np.isfinite(eta).all() and np.isfinite(xi).all()):
                print(f"  skip case {case_id} t={times[idx]:.1f}: non-finite", flush=True)
                continue
            base_eta_rows.append(eta)
            base_xi_rows.append(xi - xi.mean())
            base_depth_rows.append(depth)
            base_time_rows.append(float(times[idx]))
            base_case_rows.append(case_id)

    base_eta = np.asarray(base_eta_rows, dtype=np.float64)
    base_xi = np.asarray(base_xi_rows, dtype=np.float64)
    base_depth = np.asarray(base_depth_rows, dtype=np.float64)
    base_time = np.asarray(base_time_rows, dtype=np.float64)
    base_case = np.asarray(base_case_rows, dtype=np.int32)
    print(f"base finite snapshots: {base_eta.shape[0]}", flush=True)

    eta_aug, xi_aug, depth_aug, time_aug, case_aug = _augment_states(
        base_eta, base_xi, base_depth, base_time, base_case, args, k_abs,
    )
    print(f"augmented samples before label/filter: {eta_aug.shape[0]}", flush=True)

    @jax.jit
    def targets(eta_b: jnp.ndarray, xi_b: jnp.ndarray, depth_b: jnp.ndarray) -> jnp.ndarray:
        def one(eta: jnp.ndarray, xi: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
            gxi = dno_series_eval(
                eta, xi, k_grid, h,
                args.target_order, pad_factor=args.target_pad_factor,
            )
            return ti.apply_lowpass(gxi, k_grid, args.filter_fraction)

        return jax.vmap(one)(eta_b, xi_b, depth_b)

    gxi_chunks: list[np.ndarray] = []
    for start in range(0, eta_aug.shape[0], args.target_batch_size):
        stop = min(start + args.target_batch_size, eta_aug.shape[0])
        gxi_b = targets(
            jnp.asarray(eta_aug[start:stop], dtype=jnp.float64),
            jnp.asarray(xi_aug[start:stop], dtype=jnp.float64),
            jnp.asarray(depth_aug[start:stop], dtype=jnp.float64),
        )
        gxi_chunks.append(np.asarray(gxi_b, dtype=np.float32))
        print(f"  labeled {stop}/{eta_aug.shape[0]}", flush=True)
    gxi_aug = np.concatenate(gxi_chunks, axis=0)

    finite = (
        np.isfinite(eta_aug).all(axis=1)
        & np.isfinite(xi_aug).all(axis=1)
        & np.isfinite(gxi_aug).all(axis=1)
        & (np.abs(eta_aug).max(axis=1) < args.max_abs_eta)
        & (np.abs(xi_aug).max(axis=1) < args.max_abs_xi)
        & (np.abs(gxi_aug).max(axis=1) < args.max_abs_gxi)
    )
    print(f"kept {int(finite.sum())}/{finite.size} samples after finite/magnitude filter", flush=True)

    meta = {
        "kind": "tanaka_precascade_shard",
        "version": "v1",
        "n_batches_planned": 1,
        "source_run_dir": str(Path(args.run_dir).resolve()),
        "checkpoint": args.checkpoint,
        "case_ids": [int(c) for c in args.case_ids],
        "snapshots": {str(k): list(v) for k, v in DEFAULT_SNAPSHOTS.items() if k in args.case_ids},
        "n_base_snapshots": int(base_eta.shape[0]),
        "n_perturb_per_state": int(args.n_perturb_per_state),
        "perturb_rel_min": float(args.perturb_rel_min),
        "perturb_rel_max": float(args.perturb_rel_max),
        "perturb_k_lo": float(args.perturb_k_lo),
        "perturb_k_hi": float(args.perturb_k_hi),
        "target_order": int(args.target_order),
        "target_pad_factor": int(args.target_pad_factor),
        "filter_fraction": float(args.filter_fraction),
        "n_samples_kept": int(finite.sum()),
        "n_samples_dropped": int((~finite).sum()),
        "elapsed_s": float(time.perf_counter() - t0),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_shard(
        out_path,
        eta=eta_aug[finite],
        xi=xi_aug[finite],
        gxi=gxi_aug[finite],
        depth=depth_aug[finite],
        sample_time=time_aug[finite],
        case_id=case_aug[finite],
        x_grid=np.asarray(x_grid_np, dtype=np.float32),
        meta=meta,
    )
    print(f"wrote {out_path} in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
