"""Render rollout GIFs for the FNO Tanaka sensitivity sweep configs.

For each (alpha, substeps, Nx) sweep point in fno_tanaka_sensitivity.py, run
the full FNO + truth rollout and save a GIF showing truth-vs-prediction
side by side. Designed to run on CPU so it doesn't fight GPU training.

Skips combos that NaN within a tiny time window (substeps=1,2) since the
movie would be empty; for Nx=2048 (truth diverges around t=32) the GIF
shows the divergence up to that point.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Force CPU before any JAX import below.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from solver.solvers import time_integrator as ti  # noqa: E402
from solver.evals.model_rollout import (  # noqa: E402
    build_predict_gxi,
    load_run,
    rollout_surrogate,
    save_rollout_gif,
    truth_gxi_from_state,
)
from solver.solvers.dno_series_jax import build_grid  # noqa: E402

DEFAULT_RUN = REPO_ROOT / "outputs" / "fno_w128b6_v3_hclip5_20260514_063811"
DEFAULT_NPZ = REPO_ROOT / "data" / "tanaka_2_adaptive_g0.npz"


def load_case_ic(npz_path: Path, case_id: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    with np.load(npz_path) as d:
        tag = "batch_0000"
        case = d[f"case_id_{tag}"]
        mask = case == case_id
        order = np.argsort(d[f"time_{tag}"][mask])
        eta0 = np.asarray(d[f"eta_{tag}"][mask][order][0], dtype=np.float64)
        xi0 = np.asarray(d[f"xi_{tag}"][mask][order][0], dtype=np.float64)
        depth = float(d[f"depth_{tag}"][mask][0])
        x = np.asarray(d["x"])
        length = float(x[-1] + (x[1] - x[0]))
    return eta0, xi0, depth, length


def resample_spectral(u: np.ndarray, nx_new: int) -> np.ndarray:
    n_old = u.shape[0]
    if nx_new == n_old:
        return u.astype(np.float64)
    u_hat = np.fft.rfft(u)
    k_old = u_hat.shape[0]
    k_new = nx_new // 2 + 1
    if k_new >= k_old:
        pad = np.zeros(k_new - k_old, dtype=u_hat.dtype)
        u_hat_new = np.concatenate([u_hat, pad])
    else:
        u_hat_new = u_hat[:k_new]
    return (np.fft.irfft(u_hat_new, n=nx_new) * (nx_new / n_old)).astype(np.float64)


def trim_to_finite(arr: np.ndarray) -> int:
    """Return the first index where any spatial value is non-finite (or len(arr) if all good)."""
    bad = ~np.isfinite(arr).all(axis=-1)
    if bad.any():
        return int(np.argmax(bad))
    return int(arr.shape[0])


def render_one(
    label: str,
    eta0_base: np.ndarray,
    xi0_base: np.ndarray,
    depth: float,
    length: float,
    *,
    alpha: float,
    nx: int,
    substeps: int,
    tmax: float,
    dt: float,
    predict_gxi_fn,
    out_dir: Path,
    fps: int,
    n_frames: int,
    dpi: int,
) -> None:
    eta0 = resample_spectral(eta0_base, nx) * alpha
    xi0 = resample_spectral(xi0_base, nx) * alpha
    times = jnp.asarray(np.arange(0.0, tmax + 0.5 * dt, dt), dtype=jnp.float32)

    solver_params = ti.make_solver_params(
        nx=nx, length=length, depth=depth,
        gravity=1.0, dno_order=6, pad_factor=8, filter_fraction=2.0 / 3.0,
    )
    ic = ti.State(eta=jnp.asarray(eta0, dtype=jnp.float32),
                  xi=jnp.asarray(xi0, dtype=jnp.float32))

    truth = ti.rollout(ic, times, solver_params, substeps_per_interval=substeps, zero_mean_xi=True)
    truth_eta = np.asarray(truth["eta"]); truth_xi = np.asarray(truth["xi"])
    pred = rollout_surrogate(ic, times, solver_params, predict_gxi_fn,
                             substeps=substeps, zero_mean_xi=True)
    pred_eta = np.asarray(pred["eta"]); pred_xi = np.asarray(pred["xi"])

    _, k_grid = build_grid(nx, length)
    truth_gxi = np.asarray(truth_gxi_from_state(jnp.asarray(truth_eta), jnp.asarray(truth_xi), k_grid, depth))
    pred_gxi = np.asarray(truth_gxi_from_state(jnp.asarray(pred_eta), jnp.asarray(pred_xi), k_grid, depth))

    cut_truth = trim_to_finite(truth_eta)
    cut_pred = trim_to_finite(pred_eta)
    cut = min(cut_truth, cut_pred)
    if cut < 4:
        print(f"  [skip] {label}: diverges immediately (cut={cut})")
        return

    x = np.linspace(0.0, length, nx, endpoint=False)
    out_path = out_dir / f"{label}.gif"
    npz_path = out_dir / f"{label}.npz"
    np.savez(
        npz_path,
        x=x, t=np.asarray(times)[:cut],
        truth_eta=truth_eta[:cut], pred_eta=pred_eta[:cut],
        truth_xi=truth_xi[:cut], pred_xi=pred_xi[:cut],
        truth_gxi=truth_gxi[:cut], pred_gxi=pred_gxi[:cut],
        alpha=float(alpha), nx=int(nx), substeps=int(substeps), depth=float(depth),
    )
    save_rollout_gif(
        out_path,
        title=label.replace("_", " "),
        x=x, t=np.asarray(times)[:cut],
        truth_eta=truth_eta[:cut], pred_eta=pred_eta[:cut],
        truth_xi=truth_xi[:cut], pred_xi=pred_xi[:cut],
        truth_gxi=truth_gxi[:cut], pred_gxi=pred_gxi[:cut],
        fps=fps, n_frames=n_frames, dpi=dpi,
    )
    print(f"  saved -> {out_path.name} (frames cut at idx={cut}/{truth_eta.shape[0]})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", default=str(DEFAULT_RUN))
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--case_id", type=int, default=0)
    p.add_argument("--tmax", type=float, default=100.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--out_dir", default=str(REPO_ROOT / "playground" / "tanaka_sensitivity" / "gifs"))
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--n_frames", type=int, default=100)
    p.add_argument("--dpi", type=int, default=110)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eta0, xi0, depth, length = load_case_ic(Path(args.npz), args.case_id)
    loaded = load_run(Path(args.run_dir), checkpoint="best")
    predict_gxi_fn = build_predict_gxi(loaded, depth)
    print(f"Running on backend = {jax.default_backend()}")
    print(f"IC: case {args.case_id}, depth={depth:.4f}, length={length:.4f}, Nx_native={eta0.shape[0]}")

    # Base config (also one of the sweep points).
    base = dict(alpha=1.0, nx=1024, substeps=8)
    configs: list[tuple[str, dict]] = []
    for a in (0.5, 0.75, 1.0, 1.5, 2.0):
        configs.append((f"amp_alpha{a:.2f}", {**base, "alpha": a}))
    for s in (2, 4, 8, 16):  # substeps=1 NaN's within a few steps -> not worth a gif
        configs.append((f"sub_substeps{s:02d}", {**base, "substeps": s}))
    for n in (512, 1024, 2048):
        configs.append((f"nx_Nx{n}", {**base, "nx": n}))

    seen: dict[tuple[float, int, int], str] = {}
    deduped: list[tuple[str, dict]] = []
    for label, cfg in configs:
        key = (float(cfg["alpha"]), int(cfg["nx"]), int(cfg["substeps"]))
        if key in seen:
            print(f"  [dedupe] {label} same as {seen[key]}")
            continue
        seen[key] = label
        deduped.append((label, cfg))

    print(f"Rendering {len(deduped)} configs")
    for i, (label, cfg) in enumerate(deduped, 1):
        print(f"[{i}/{len(deduped)}] {label}: {cfg}")
        render_one(
            label, eta0, xi0, depth, length,
            tmax=args.tmax, dt=args.dt, predict_gxi_fn=predict_gxi_fn,
            out_dir=out_dir, fps=args.fps, n_frames=args.n_frames, dpi=args.dpi,
            **cfg,
        )


if __name__ == "__main__":
    main()
