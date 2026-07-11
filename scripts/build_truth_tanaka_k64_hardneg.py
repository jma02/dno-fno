"""Build clean truth-state Tanaka hard negatives in the diagnosed k=64..128 band.

Unlike ``harvest_tanaka_precascade_pack.py``, this script does not roll out a
surrogate and does not use model-contaminated pre-NaN states. It starts from a
cached f64 truth trajectory, adds small band-limited perturbations to clean
truth states, and recomputes Craig-Sulem labels.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from solver.solvers import time_integrator as ti  # noqa: E402
from solver.solvers.dno_series_jax import build_grid, dno_series_eval  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--truth_npz",
        type=Path,
        default=Path(
            "outputs/cs_dno_w512b8_l256_v9_h100x2_20260703_035745/"
            "eval_2x2v3_off_20260706_070633/tanaka_g0_trajs.npz",
        ),
    )
    p.add_argument("--output", type=Path, default=Path("data/tanaka_truth_k64_hardneg_v1.npz"))
    p.add_argument("--case_ids", type=int, nargs="+", default=[5, 6, 11, 22])
    p.add_argument(
        "--snapshot_times",
        type=float,
        nargs="+",
        default=[0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 64.0, 80.0, 120.0, 160.0, 188.0],
    )
    p.add_argument("--n_perturb_per_state", type=int, default=128)
    p.add_argument("--perturb_rel_min", type=float, default=1e-6)
    p.add_argument("--perturb_rel_max", type=float, default=1e-3)
    p.add_argument("--perturb_k_lo", type=float, default=64.0)
    p.add_argument("--perturb_k_hi", type=float, default=128.0)
    p.add_argument("--target_order", type=int, default=6)
    p.add_argument("--target_pad_factor", type=int, default=8)
    p.add_argument("--filter_fraction", type=float, default=0.25)
    p.add_argument("--target_batch_size", type=int, default=128)
    p.add_argument("--max_abs_eta", type=float, default=5.0)
    p.add_argument("--max_abs_xi", type=float, default=20.0)
    p.add_argument("--max_abs_gxi", type=float, default=200.0)
    p.add_argument("--seed", type=int, default=20260709)
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


def _bandlimited_unit_noise(
    rng: np.random.Generator,
    *,
    nx: int,
    k_abs: np.ndarray,
    k_lo: float,
    k_hi: float,
) -> np.ndarray:
    coeff = np.zeros((nx // 2 + 1,), dtype=np.complex128)
    mask = (k_abs >= k_lo) & (k_abs <= k_hi)
    coeff[mask] = rng.normal(size=int(mask.sum())) + 1j * rng.normal(size=int(mask.sum()))
    coeff[0] = 0.0
    if nx % 2 == 0:
        coeff[-1] = coeff[-1].real + 0.0j
    noise = np.fft.irfft(coeff, n=nx)
    rms = float(np.sqrt(np.mean(noise**2)))
    return (noise / max(rms, 1e-30)).astype(np.float64)


def _append_perturbed_rows(
    *,
    rng: np.random.Generator,
    eta0: np.ndarray,
    xi0: np.ndarray,
    depth0: float,
    time0: float,
    case0: int,
    args: argparse.Namespace,
    k_abs: np.ndarray,
    rows: dict[str, list[np.ndarray | float | int]],
) -> None:
    eta_rms = float(np.sqrt(np.mean(eta0**2)))
    xi_rms = float(np.sqrt(np.mean(xi0**2)))
    rows["eta"].append(eta0.astype(np.float64, copy=False))
    rows["xi"].append((xi0 - xi0.mean()).astype(np.float64, copy=False))
    rows["depth"].append(depth0)
    rows["time"].append(time0)
    rows["case"].append(case0)

    for perturb_idx in range(args.n_perturb_per_state):
        amp = float(np.exp(rng.uniform(np.log(args.perturb_rel_min), np.log(args.perturb_rel_max))))
        eta_noise = _bandlimited_unit_noise(
            rng, nx=args.nx, k_abs=k_abs, k_lo=args.perturb_k_lo, k_hi=args.perturb_k_hi,
        )
        xi_noise = _bandlimited_unit_noise(
            rng, nx=args.nx, k_abs=k_abs, k_lo=args.perturb_k_lo, k_hi=args.perturb_k_hi,
        )
        eta = eta0 + amp * max(eta_rms, 1e-6) * eta_noise
        xi = xi0 + amp * max(xi_rms, 1e-6) * xi_noise
        rows["eta"].append(eta.astype(np.float64, copy=False))
        rows["xi"].append((xi - xi.mean()).astype(np.float64, copy=False))
        rows["depth"].append(depth0)
        rows["time"].append(time0)
        rows["case"].append(case0 * 1_000_000 + perturb_idx + 1)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists; remove it first.")

    t0 = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    x_grid_np, k_grid_np = build_grid(args.nx, args.length)
    k_grid = jnp.asarray(k_grid_np, dtype=jnp.float64)
    k_abs = np.abs(k_grid_np[: args.nx // 2 + 1])

    with np.load(args.truth_npz) as truth:
        times = np.asarray(truth["times"], dtype=np.float64)
        case_ids = np.asarray(truth["case_ids"], dtype=np.int64)
        depths = np.asarray(truth["depths"], dtype=np.float64)
        truth_eta = truth["truth_eta"]
        truth_xi = truth["truth_xi"]
        case_to_col = {int(case_id): idx for idx, case_id in enumerate(case_ids)}

        rows: dict[str, list[np.ndarray | float | int]] = {
            "eta": [],
            "xi": [],
            "depth": [],
            "time": [],
            "case": [],
        }
        for case_id in args.case_ids:
            if case_id not in case_to_col:
                raise ValueError(f"case_id {case_id} not present in {args.truth_npz}")
            col = case_to_col[case_id]
            for snapshot_time in args.snapshot_times:
                time_idx = int(np.argmin(np.abs(times - snapshot_time)))
                eta0 = np.asarray(truth_eta[time_idx, col], dtype=np.float64)
                xi0 = np.asarray(truth_xi[time_idx, col], dtype=np.float64)
                if not (np.isfinite(eta0).all() and np.isfinite(xi0).all()):
                    print(f"skip case={case_id} t={times[time_idx]:.1f}: non-finite truth", flush=True)
                    continue
                _append_perturbed_rows(
                    rng=rng,
                    eta0=eta0,
                    xi0=xi0,
                    depth0=float(depths[col]),
                    time0=float(times[time_idx]),
                    case0=int(case_id),
                    args=args,
                    k_abs=k_abs,
                    rows=rows,
                )

    eta_aug = np.asarray(rows["eta"], dtype=np.float64)
    xi_aug = np.asarray(rows["xi"], dtype=np.float64)
    depth_aug = np.asarray(rows["depth"], dtype=np.float64)
    time_aug = np.asarray(rows["time"], dtype=np.float64)
    case_aug = np.asarray(rows["case"], dtype=np.int32)
    print(f"augmented clean truth samples before label/filter: {eta_aug.shape[0]}", flush=True)

    @jax.jit
    def targets(eta_b: jnp.ndarray, xi_b: jnp.ndarray, depth_b: jnp.ndarray) -> jnp.ndarray:
        def one(eta: jnp.ndarray, xi: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
            gxi = dno_series_eval(
                eta,
                xi,
                k_grid,
                h,
                args.target_order,
                pad_factor=args.target_pad_factor,
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
        "kind": "tanaka_truth_k64_hardneg_shard",
        "version": "v1",
        "n_batches_planned": 1,
        "truth_npz": str(args.truth_npz.resolve()),
        "case_ids": [int(c) for c in args.case_ids],
        "snapshot_times": [float(t) for t in args.snapshot_times],
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_shard(
        args.output,
        eta=eta_aug[finite],
        xi=xi_aug[finite],
        gxi=gxi_aug[finite],
        depth=depth_aug[finite],
        sample_time=time_aug[finite],
        case_id=case_aug[finite],
        x_grid=np.asarray(x_grid_np, dtype=np.float32),
        meta=meta,
    )
    print(f"wrote {args.output} in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
