"""Compare dno_dataset (Zakharov-rescaled to L=2pi) against our new linear /
stokes_deep / stokes_finite generators. Saves plots to outputs/dist_check/.

Run on GPU:
    JAX_ENABLE_X64=1 uv run python scripts/plot_dist_check.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solver.solvers.dno_series_jax import build_grid
from solver.gen_data.generate_linear_dataset import (
    sample_linear_case_params, build_linear_batch,
)
from solver.gen_data.generate_stokes_dataset import (
    sample_stokes_case_params, build_stokes_batch,
)

OUT_DIR = Path("/home/johnma/dno-fno/outputs/dist_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LENGTH = 2.0 * math.pi
NX = 1024
GRAVITY = 1.0
DNO_ORDER = 6
PAD_FACTOR = 8

ALPHA = 164.0 / LENGTH  # Zakharov rescaling from MATLAB L=164 to ours L=2pi
A0_MIN_LINEAR = 1e-4      # below steepness_min/n0_max=2.5e-4 binding floor
A0_MIN_STOKES = 7.66e-4   # MATLAB 0.02 / alpha
A0_MAX = 1.1494e-2        # MATLAB 0.30 / alpha
STEEP_MIN_LINEAR = 5e-3
STEEP_MAX = 0.15
N_GEN = 5000

LIN_DEPTH = (0.02, 1.5);  LIN_KH = (0.02, 20.0); LIN_N0 = (1, 20)
SD_DEPTH  = (4.0, 50.0);                        SD_N0  = (1, 20)
SF_DEPTH  = (0.02, 1.5);  SF_KH = (0.5, 5.0);   SF_N0  = (14, 26)


def _rescale_dno(eta: np.ndarray, xi: np.ndarray, gxi: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "eta": eta / ALPHA,
        "xi":  xi  / (ALPHA ** 1.5),
        "gxi": gxi / (ALPHA ** 0.5),
    }


def load_dno_dataset(path: Path) -> dict[str, dict[str, np.ndarray]]:
    raw = np.load(path)
    half = raw["stokes_eta"].shape[0] // 2
    return {
        "linear":        _rescale_dno(raw["linear_eta"], raw["linear_xi"], raw["linear_Gxi"]),
        "stokes_deep":   _rescale_dno(raw["stokes_eta"][:half], raw["stokes_xi"][:half], raw["stokes_Gxi"][:half]),
        "stokes_finite": _rescale_dno(raw["stokes_eta"][half:], raw["stokes_xi"][half:], raw["stokes_Gxi"][half:]),
    }


def _to_f64(d: dict[str, np.ndarray]) -> tuple[jnp.ndarray, ...]:
    return tuple(jnp.asarray(d[k], dtype=jnp.float64) for k in d)


def generate_linear(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    params = sample_linear_case_params(
        rng, batch_size=n, length=LENGTH,
        n0_min=LIN_N0[0], n0_max=LIN_N0[1],
        a0_min=A0_MIN_LINEAR, a0_max=A0_MAX,
        steepness_min=STEEP_MIN_LINEAR, steepness_max=STEEP_MAX,
        depth_min=LIN_DEPTH[0], depth_max=LIN_DEPTH[1],
        kh_min=LIN_KH[0], kh_max=LIN_KH[1],
        rejection_attempts=1000,
    )
    x_np, k_np = build_grid(NX, LENGTH)
    x = jnp.asarray(x_np, dtype=jnp.float64)
    k = jnp.asarray(k_np, dtype=jnp.float64)
    eta, xi, gxi = build_linear_batch(
        x=x, k=k, length=LENGTH, gravity=GRAVITY,
        n0=jnp.asarray(params["n0"], dtype=jnp.float64),
        a0=jnp.asarray(params["a0"], dtype=jnp.float64),
        depth=jnp.asarray(params["depth"], dtype=jnp.float64),
        phase=jnp.asarray(params["phase"], dtype=jnp.float64),
        direction=jnp.asarray(params["direction"], dtype=jnp.float64),
        dno_order=DNO_ORDER, pad_factor=PAD_FACTOR,
    )
    return {
        "eta": np.asarray(eta), "xi": np.asarray(xi), "gxi": np.asarray(gxi),
        "depth": params["depth"].astype(np.float64),
    }


def generate_stokes(regime: str, n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if regime == "deep":
        n0 = SD_N0; depth = SD_DEPTH; kh = (1e-3, 1e6)
        ichoi = 0
    else:
        n0 = SF_N0; depth = SF_DEPTH; kh = SF_KH
        ichoi = 1
    params = sample_stokes_case_params(
        rng, batch_size=n, length=LENGTH, gravity=GRAVITY,
        n0_min=n0[0], n0_max=n0[1],
        a0_min=A0_MIN_STOKES, a0_max=A0_MAX,
        steepness_max=STEEP_MAX,
        depth_min=depth[0], depth_max=depth[1],
        kh_min=kh[0], kh_max=kh[1],
        rejection_attempts=1000,
        finite_depth=ichoi == 1,
    )
    x_np, k_np = build_grid(NX, LENGTH)
    x = jnp.asarray(x_np, dtype=jnp.float64)
    k = jnp.asarray(k_np, dtype=jnp.float64)
    eta, xi, gxi = build_stokes_batch(
        x=x, k=k, length=LENGTH, gravity=GRAVITY,
        n0=jnp.asarray(params["n0"], dtype=jnp.float64),
        a0=jnp.asarray(params["a0"], dtype=jnp.float64),
        depth=jnp.asarray(params["depth"], dtype=jnp.float64),
        phase=jnp.asarray(params["phase"], dtype=jnp.float64),
        ichoi=ichoi, dno_order=DNO_ORDER, pad_factor=PAD_FACTOR,
    )
    return {
        "eta": np.asarray(eta), "xi": np.asarray(xi), "gxi": np.asarray(gxi),
        "depth": params["depth"].astype(np.float64),
    }


def peak(a: np.ndarray) -> np.ndarray:
    return np.max(np.abs(a), axis=1)


def plot_hist_pair(dno: dict[str, np.ndarray], gen: dict[str, np.ndarray], title: str, fname: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, lbl in zip(axes, ("eta", "gxi"), (r"peak $|\eta|$", r"peak $|G\xi|$")):
        dv = peak(dno[key]); gv = peak(gen[key])
        lo = max(min(dv.min(), gv.min()), 1e-7)
        hi = max(dv.max(), gv.max()) * 1.01
        bins = np.geomspace(lo, hi, 60)
        ax.hist(dv, bins=bins, alpha=0.5, density=True, label=f"dno_dataset (N={len(dv)})", color="C0")
        ax.hist(gv, bins=bins, alpha=0.5, density=True, label=f"new generator (N={len(gv)})", color="C3")
        ax.set_xscale("log"); ax.set_xlabel(lbl); ax.set_ylabel("density")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=110); plt.close(fig)


def plot_joint(dno: dict[str, np.ndarray], gen: dict[str, np.ndarray], title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(peak(dno["eta"]), peak(dno["gxi"]), s=4, alpha=0.25, color="C0", label="dno_dataset")
    ax.scatter(peak(gen["eta"]), peak(gen["gxi"]), s=4, alpha=0.25, color="C3", label="new generator")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"peak $|\eta|$"); ax.set_ylabel(r"peak $|G\xi|$")
    ax.legend(); ax.grid(True, alpha=0.3, which="both"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(OUT_DIR / fname, dpi=110); plt.close(fig)


def plot_samples(dno: dict[str, np.ndarray], gen: dict[str, np.ndarray], title: str, fname: str) -> None:
    rng = np.random.default_rng(0)
    n_show = 6
    di = rng.choice(dno["eta"].shape[0], size=n_show, replace=False)
    gi = rng.choice(gen["eta"].shape[0], size=n_show, replace=False)
    x = np.linspace(0, LENGTH, dno["eta"].shape[1], endpoint=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    axes = axes.flatten()
    for k in range(n_show):
        ax = axes[k]
        ax.plot(x, dno["gxi"][di[k]], color="C0", alpha=0.85, lw=0.9, label="dno")
        ax.plot(x, gen["gxi"][gi[k]], color="C3", alpha=0.85, lw=0.9, label="new")
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"{title} — random $G\\xi$ samples")
    fig.tight_layout(); fig.savefig(OUT_DIR / fname, dpi=110); plt.close(fig)


def main() -> None:
    print(f"jax devices: {jax.devices()}")
    print(f"output dir: {OUT_DIR}")
    print(f"alpha (Zakharov) = {ALPHA:.6f}")

    print("loading dno_dataset.npz ...")
    dno = load_dno_dataset(Path("data/dno_dataset.npz"))
    for kind, v in dno.items():
        pe = peak(v["eta"]); pg = peak(v["gxi"])
        print(f"  dno/{kind:<14s}: N={v['eta'].shape[0]:>6d}  peak|eta| [{pe.min():.3e}, {pe.max():.3e}]  peak|gxi| [{pg.min():.3e}, {pg.max():.3e}]")

    gens: dict[str, dict[str, np.ndarray]] = {}
    print("generating linear ...")
    gens["linear"] = generate_linear(N_GEN, seed=20260514)
    print("generating stokes_deep ...")
    gens["stokes_deep"] = generate_stokes("deep", N_GEN, seed=20260515)
    print("generating stokes_finite ...")
    gens["stokes_finite"] = generate_stokes("shallow", N_GEN, seed=20260516)
    for kind, v in gens.items():
        pe = peak(v["eta"]); pg = peak(v["gxi"])
        print(f"  gen/{kind:<14s}: N={v['eta'].shape[0]:>6d}  peak|eta| [{pe.min():.3e}, {pe.max():.3e}]  peak|gxi| [{pg.min():.3e}, {pg.max():.3e}]  depth [{v['depth'].min():.3f}, {v['depth'].max():.3f}]")

    for kind in ("linear", "stokes_deep", "stokes_finite"):
        plot_hist_pair(dno[kind], gens[kind], f"{kind}: peak amplitudes", f"hist_{kind}.png")
        plot_joint(dno[kind], gens[kind], f"{kind}: peak |eta| vs peak |gxi|", f"joint_{kind}.png")
        plot_samples(dno[kind], gens[kind], kind, f"samples_{kind}.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for kind, color in zip(("linear", "stokes_deep", "stokes_finite"), ("C0", "C1", "C2")):
        ax.hist(gens[kind]["depth"], bins=40, alpha=0.55, density=True, label=kind, color=color)
    ax.set_xlabel("depth h"); ax.set_ylabel("density")
    ax.set_title("Generator depth coverage"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / "depth_coverage.png", dpi=110); plt.close(fig)

    print("done. files:")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
