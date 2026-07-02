"""Sensitivity sweep for FNO discrepancy on a single Tanaka soliton.

Tests whether the rollout discrepancy observed against the analytic-DNO
truth solver on Tanaka case 0 persists when we vary

* wave amplitude alpha applied to (eta0, xi0)            -- physical
* integrator substeps per output interval                -- numerical (dt)
* spatial resolution Nx                                  -- numerical (Nx)

while holding all other knobs fixed. Plots rel-L2 eta-error vs time for
each axis as a 3-panel figure.

Uses the FNO baseline run `fno_w128b6_v3_hclip5_20260514_063811`. Truth is
generated for each (alpha, Nx) by `ti.rollout` from the (possibly
rescaled, possibly resampled) IC. The substeps sweep varies the integrator's
fine-grained step while keeping the output time grid fixed; truth and
prediction share substeps so we are comparing FNO+integrator at the same
numerical resolution as the truth at that substeps value.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from solver.solvers import time_integrator as ti
from solver.evals.model_rollout import (
    build_predict_gxi,
    load_run,
    rollout_surrogate,
    truth_gxi_from_state,
)


DEFAULT_RUN = REPO_ROOT / "outputs" / "fno_w128b6_v3_hclip5_20260514_063811"
DEFAULT_NPZ = REPO_ROOT / "data" / "tanaka_2_adaptive_g0.npz"
DEFAULT_CASE = 0
DEFAULT_NX = 1024
DEFAULT_SUBSTEPS = 8
DEFAULT_TMAX = 100.0
DEFAULT_DT = 1.0
DEFAULT_ALPHAS = (0.5, 0.75, 1.0, 1.5, 2.0)
DEFAULT_SUBSTEP_SWEEP = (1, 2, 4, 8, 16)
DEFAULT_NX_SWEEP = (512, 1024, 2048)


def load_case_ic(npz_path: Path, case_id: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (eta0, xi0, depth, length) for the t=0 frame of the given case."""
    with np.load(npz_path) as d:
        tag = "batch_0000"
        case = d[f"case_id_{tag}"]
        mask = case == case_id
        if not mask.any():
            raise SystemExit(f"case_id={case_id} not present in {npz_path.name}")
        order = np.argsort(d[f"time_{tag}"][mask])
        eta0 = np.asarray(d[f"eta_{tag}"][mask][order][0], dtype=np.float64)
        xi0 = np.asarray(d[f"xi_{tag}"][mask][order][0], dtype=np.float64)
        depth = float(d[f"depth_{tag}"][mask][0])
        x = np.asarray(d["x"])
        length = float(x[-1] + (x[1] - x[0]))
    return eta0, xi0, depth, length


def resample_spectral(u: np.ndarray, nx_new: int) -> np.ndarray:
    """Zero-pad or truncate rfft to resample u from len(u) to nx_new (preserves amplitude)."""
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
    u_new = np.fft.irfft(u_hat_new, n=nx_new) * (nx_new / n_old)
    return u_new.astype(np.float64)


def rel_l2(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-time rel-L2 error: ||pred - truth|| / ||truth||."""
    num = np.sqrt(np.sum((pred - truth) ** 2, axis=-1))
    den = np.sqrt(np.sum(truth ** 2, axis=-1))
    return num / np.maximum(den, 1e-30)


def run_one(
    eta0_base: np.ndarray,
    xi0_base: np.ndarray,
    depth: float,
    length: float,
    *,
    alpha: float,
    nx: int,
    substeps: int,
    times: jnp.ndarray,
    predict_gxi_fn,  # callable -- closure over loaded FNO
) -> dict[str, np.ndarray]:
    eta0 = resample_spectral(eta0_base, nx) * alpha
    xi0 = resample_spectral(xi0_base, nx) * alpha

    solver_params = ti.make_solver_params(
        nx=nx, length=length, depth=depth,
        gravity=1.0, dno_order=6, pad_factor=8, filter_fraction=2.0 / 3.0,
    )
    ic = ti.State(eta=jnp.asarray(eta0, dtype=jnp.float32),
                  xi=jnp.asarray(xi0, dtype=jnp.float32))

    truth = ti.rollout(ic, times, solver_params,
                       substeps_per_interval=substeps, zero_mean_xi=True)
    truth_eta = np.asarray(truth["eta"])
    truth_xi = np.asarray(truth["xi"])

    pred = rollout_surrogate(
        ic, times, solver_params, predict_gxi_fn,
        substeps=substeps, zero_mean_xi=True,
    )
    pred_eta = np.asarray(pred["eta"])
    pred_xi = np.asarray(pred["xi"])

    return {
        "t": np.asarray(times),
        "eta_rel_l2": rel_l2(pred_eta, truth_eta),
        "xi_rel_l2": rel_l2(pred_xi, truth_xi),
        "truth_eta_max_abs": np.max(np.abs(truth_eta), axis=-1),
    }


def sweep_axis(
    label_fmt: str,
    values: tuple,
    *,
    base_kwargs: dict,
    overridden_key: str,
    eta0_base: np.ndarray,
    xi0_base: np.ndarray,
    depth: float,
    length: float,
    times: jnp.ndarray,
    predict_gxi_fn,
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for v in values:
        kwargs = dict(base_kwargs)
        kwargs[overridden_key] = v
        label = label_fmt.format(v=v)
        print(f"  -> {label}  (alpha={kwargs['alpha']}, nx={kwargs['nx']}, substeps={kwargs['substeps']})")
        out[label] = run_one(
            eta0_base, xi0_base, depth, length,
            times=times, predict_gxi_fn=predict_gxi_fn, **kwargs,
        )
    return out


def plot_sweeps(results: dict[str, dict[str, dict[str, np.ndarray]]], out_path: Path,
                title_suffix: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    cmap_names = {"amplitude": "plasma", "substeps": "viridis", "nx": "cividis"}
    titles = {
        "amplitude": "Physical: IC amplitude scale alpha",
        "substeps": "Numerical: integrator substeps / output dt",
        "nx": "Numerical: spatial resolution Nx",
    }
    for ax, (axis_name, sweep) in zip(axes, results.items()):
        labels = list(sweep.keys())
        cmap = plt.get_cmap(cmap_names[axis_name])
        colors = cmap(np.linspace(0.15, 0.85, len(labels)))
        for label, color in zip(labels, colors):
            r = sweep[label]
            ax.plot(r["t"], r["eta_rel_l2"], color=color, lw=1.6, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("time")
        ax.set_ylabel(r"rel-$L^2$ $\eta$-error")
        ax.set_title(titles[axis_name])
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"FNO vs analytic-DNO truth: Tanaka case 0 sensitivity ({title_suffix})", fontsize=11)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", default=str(DEFAULT_RUN))
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--case_id", type=int, default=DEFAULT_CASE)
    p.add_argument("--tmax", type=float, default=DEFAULT_TMAX)
    p.add_argument("--dt", type=float, default=DEFAULT_DT)
    p.add_argument("--out_dir", default=str(REPO_ROOT / "playground" / "tanaka_sensitivity"))
    p.add_argument("--quick", action="store_true", help="Shortened sweeps + tmax=30 for smoke-test.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eta0, xi0, depth, length = load_case_ic(Path(args.npz), args.case_id)
    print(f"IC: case {args.case_id}, depth={depth:.4f}, length={length:.4f}, Nx_native={eta0.shape[0]}")
    print(f"     |eta|_max={float(np.max(np.abs(eta0))):.5f}, |xi|_max={float(np.max(np.abs(xi0))):.5f}")

    loaded = load_run(Path(args.run_dir), checkpoint="best")
    predict_gxi_fn = build_predict_gxi(loaded, depth)
    print(f"Loaded FNO from {Path(args.run_dir).name} (epoch {loaded.epoch})")

    tmax = 30.0 if args.quick else args.tmax
    alphas = (0.5, 1.0, 2.0) if args.quick else DEFAULT_ALPHAS
    substep_vals = (2, 8, 16) if args.quick else DEFAULT_SUBSTEP_SWEEP
    nx_vals = (512, 1024) if args.quick else DEFAULT_NX_SWEEP

    times = jnp.asarray(np.arange(0.0, tmax + 0.5 * args.dt, args.dt), dtype=jnp.float32)

    base = dict(alpha=1.0, nx=DEFAULT_NX, substeps=DEFAULT_SUBSTEPS)

    print("\n[1/3] Amplitude sweep")
    amp_results = sweep_axis(
        "alpha={v:.2f}", alphas,
        base_kwargs=base, overridden_key="alpha",
        eta0_base=eta0, xi0_base=xi0, depth=depth, length=length,
        times=times, predict_gxi_fn=predict_gxi_fn,
    )
    print("\n[2/3] Substeps sweep")
    sub_results = sweep_axis(
        "substeps={v}", substep_vals,
        base_kwargs=base, overridden_key="substeps",
        eta0_base=eta0, xi0_base=xi0, depth=depth, length=length,
        times=times, predict_gxi_fn=predict_gxi_fn,
    )
    print("\n[3/3] Nx sweep")
    nx_results = sweep_axis(
        "Nx={v}", nx_vals,
        base_kwargs=base, overridden_key="nx",
        eta0_base=eta0, xi0_base=xi0, depth=depth, length=length,
        times=times, predict_gxi_fn=predict_gxi_fn,
    )

    all_results = {"amplitude": amp_results, "substeps": sub_results, "nx": nx_results}

    title_suffix = f"depth={depth:.3f}, T={tmax:g}, dt_out={args.dt:g}"
    plot_path = out_dir / "fno_tanaka_sensitivity.png"
    plot_sweeps(all_results, plot_path, title_suffix=title_suffix)
    print(f"\nsaved -> {plot_path}")

    npz_payload: dict[str, np.ndarray] = {}
    for axis, sweep in all_results.items():
        for label, r in sweep.items():
            tag = f"{axis}__{label.replace('=', '').replace('.', 'p').replace(' ', '')}"
            npz_payload[f"{tag}__t"] = r["t"]
            npz_payload[f"{tag}__eta_rel_l2"] = r["eta_rel_l2"]
            npz_payload[f"{tag}__xi_rel_l2"] = r["xi_rel_l2"]
            npz_payload[f"{tag}__truth_eta_max_abs"] = r["truth_eta_max_abs"]
    np.savez(out_dir / "fno_tanaka_sensitivity.npz", **npz_payload)
    print(f"saved -> {out_dir / 'fno_tanaka_sensitivity.npz'}")

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for axis, sweep in all_results.items():
        summary[axis] = {}
        for label, r in sweep.items():
            summary[axis][label] = {
                "eta_rel_l2_final": float(r["eta_rel_l2"][-1]),
                "eta_rel_l2_mid": float(r["eta_rel_l2"][len(r["t"]) // 2]),
                "xi_rel_l2_final": float(r["xi_rel_l2"][-1]),
            }
    (out_dir / "fno_tanaka_sensitivity.json").write_text(json.dumps(summary, indent=2))
    print(f"saved -> {out_dir / 'fno_tanaka_sensitivity.json'}")
    print("\nFinal-time errors:")
    for axis, axis_sum in summary.items():
        print(f"  [{axis}]")
        for label, m in axis_sum.items():
            print(f"    {label:18s}  eta_rel_l2_T={m['eta_rel_l2_final']:.4f}  xi={m['xi_rel_l2_final']:.4f}")


if __name__ == "__main__":
    main()
