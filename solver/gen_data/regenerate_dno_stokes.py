"""Regenerate the stokes portion of data/dno_dataset.npz with deep/finite labels.

The legacy MATLAB stokes_dno_gen_data.m writes both ichoi=0 (deep, h_use=1000) and
ichoi=1 (finite, h=1) samples into a single concatenated 'stokes_*' subset with no
regime tag. Downstream, make_dno_test_set.py blindly stamps every stokes sample with
h_new = 1/α ≈ 0.0383 — correct only for the finite half. The deep half gets fed to
the model with a depth that disagrees with the operator that produced its gxi target.

This script regenerates both regimes from scratch using stokes_eta_xi (Python port of
the MATLAB Stokes 5th-order formula) and dno_series_eval (Python port of the MATLAB
DNO Taylor series), then rewrites dno_dataset.npz so that the stokes data is split
into stokes_deep_* and stokes_finite_*. Everything else (soliton, linear, x) passes
through unchanged.

Sampling mirrors MATLAB:
    Nsamples = 392 per regime; dt = 0.1, Tmax = 20 → 51 saves per (a0, n0) draw
    ichoi=0: n0 ∈ [1, 20],  a0 ∈ [0.02, 0.30] s.t. k0*a0 ≤ 0.15, h_use = 1000
    ichoi=1: n0 ∈ [14, 26], a0 ∈ [0.02, 0.30] s.t. k0*a0 ≤ 0.15, h     = 1
    L = 164, NX = 1024, g = 1, DNO order = 6, dealias pad = 8

Run:
    uv run python -m solver.gen_data.regenerate_dno_stokes
"""
from __future__ import annotations

import argparse
import math
import shutil
import zipfile
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from numpy.lib import format as npy_format
from tqdm.auto import tqdm

from solver.data.stokes_truth_jax import stokes_eta_xi
from solver.solvers.dno_series_jax import build_grid, dno_series_eval


L = 164.0
NX = 1024
DT = 0.1
TMAX = 20.0
G = 1.0
H_FINITE = 1.0
H_DEEP = 1000.0
NSAMPLES_PER_REGIME = 392
A0_MIN, A0_MAX = 0.02, 0.30
STEEPNESS_MAX = 0.15
NPRINT = 50
DNO_ORDER = 6
PAD_FACTOR = 8


def build_save_idx() -> np.ndarray:
    nt = int(round(TMAX / DT)) + 1
    iprint = max(1, nt // NPRINT)
    return np.unique(np.concatenate([np.arange(0, nt, iprint), [nt - 1]]))


def sample_a0(rng: np.random.Generator, k0: float) -> float:
    for _ in range(1000):
        a0 = float(rng.uniform(A0_MIN, A0_MAX))
        if k0 * a0 <= STEEPNESS_MAX:
            return a0
    return float(min(A0_MAX, STEEPNESS_MAX / max(k0, 1e-12)))


def regenerate_regime(
    *,
    ichoi: int,
    n0_min: int,
    n0_max: int,
    h_use: float,
    seed: int,
    save_times: np.ndarray,
    x_jnp: jnp.ndarray,
    k_jnp: jnp.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_saves = save_times.shape[0]
    total = NSAMPLES_PER_REGIME * n_saves
    eta = np.empty((total, NX), dtype=np.float64)
    xi = np.empty((total, NX), dtype=np.float64)
    gxi = np.empty((total, NX), dtype=np.float64)
    save_times_jnp = jnp.asarray(save_times, dtype=jnp.float64)
    h_arr = jnp.asarray(np.full((n_saves, 1), h_use, dtype=np.float64))

    @jax.jit
    def per_sample(n0_i: jnp.ndarray, a0_i: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        def per_t(t_: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            return stokes_eta_xi(x_jnp, t_, n0_i, a0_i, L, h_use, G, ichoi)

        eta_st, xi_st = jax.vmap(per_t)(save_times_jnp)
        gxi_st = dno_series_eval(eta_st, xi_st, k_jnp, h_arr, DNO_ORDER, pad_factor=PAD_FACTOR)
        return eta_st, xi_st, gxi_st

    cursor = 0
    for _ in tqdm(range(NSAMPLES_PER_REGIME), desc=f"ichoi={ichoi}", dynamic_ncols=True):
        n0 = int(rng.integers(n0_min, n0_max + 1))
        k0 = n0 * (2.0 * math.pi / L)
        a0 = sample_a0(rng, k0)
        e, x_, g = per_sample(jnp.asarray(n0, dtype=jnp.float64), jnp.asarray(a0, dtype=jnp.float64))
        sl = slice(cursor, cursor + n_saves)
        eta[sl] = np.asarray(jax.device_get(e), dtype=np.float64)
        xi[sl] = np.asarray(jax.device_get(x_), dtype=np.float64)
        gxi[sl] = np.asarray(jax.device_get(g), dtype=np.float64)
        cursor += n_saves
    return eta, xi, gxi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/dno_dataset.npz")
    p.add_argument("--output", default=None, help="default: overwrite input in place")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve() if args.output else in_path

    save_idx = build_save_idx()
    times_full = np.arange(int(round(TMAX / DT)) + 1, dtype=np.float64) * DT
    save_times = times_full[save_idx]
    print(
        f"nt={len(times_full)}, n_saves={len(save_idx)}, "
        f"total per regime = {NSAMPLES_PER_REGIME * len(save_idx)}",
    )

    x_grid, k_grid = build_grid(NX, L)
    x_jnp = jnp.asarray(x_grid, dtype=jnp.float64)
    k_jnp = jnp.asarray(k_grid, dtype=jnp.float64)

    t0 = perf_counter()
    eta_d, xi_d, gxi_d = regenerate_regime(
        ichoi=0, n0_min=1, n0_max=20, h_use=H_DEEP,
        seed=args.seed, save_times=save_times, x_jnp=x_jnp, k_jnp=k_jnp,
    )
    eta_f, xi_f, gxi_f = regenerate_regime(
        ichoi=1, n0_min=14, n0_max=26, h_use=H_FINITE,
        seed=args.seed + 1, save_times=save_times, x_jnp=x_jnp, k_jnp=k_jnp,
    )
    print(f"  regimes generated in {perf_counter() - t0:.0f}s")
    print(f"  deep   shape={eta_d.shape} max|eta|={np.abs(eta_d).max():.4f} std xi={xi_d.std():.4f}")
    print(f"  finite shape={eta_f.shape} max|eta|={np.abs(eta_f).max():.4f} std xi={xi_f.std():.4f}")

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    DROP_PREFIXES = ("stokes_eta", "stokes_xi", "stokes_Gxi")
    new_entries = [
        ("stokes_deep_eta.npy", eta_d),
        ("stokes_deep_xi.npy", xi_d),
        ("stokes_deep_Gxi.npy", gxi_d),
        ("stokes_finite_eta.npy", eta_f),
        ("stokes_finite_xi.npy", xi_f),
        ("stokes_finite_Gxi.npy", gxi_f),
    ]
    print(f"  rewriting {out_path} ...")
    with zipfile.ZipFile(in_path, "r") as src, zipfile.ZipFile(
        tmp_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True,
    ) as dst:
        for name in src.namelist():
            if any(name.startswith(p) for p in DROP_PREFIXES):
                continue
            with src.open(name) as fr, dst.open(name, "w", force_zip64=True) as fw:
                shutil.copyfileobj(fr, fw)
        for arr_name, arr in new_entries:
            with dst.open(arr_name, "w", force_zip64=True) as fw:
                npy_format.write_array(fw, np.ascontiguousarray(arr), allow_pickle=False)
    tmp_path.replace(out_path)
    sz = out_path.stat().st_size / 1e9
    print(f"  done. wrote {out_path} ({sz:.2f} GB)")


if __name__ == "__main__":
    main()
