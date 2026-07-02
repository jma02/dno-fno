"""Diagnose where DNO/FNO rollouts lose accuracy.

Three concrete hypotheses:
  (A) Spectral truncation: G(eta)xi has content above modes=64 that the model
      cannot represent, so its single-step prediction has a permanent error
      floor visible in the residual spectrum.
  (B) Distribution shift: the model is accurate on training-distribution
      snapshots but degrades on out-of-distribution states produced by its
      own rollout (this is what pushforward training fixes).
  (C) Loss-floor mismatch: the val_loss=0.4% is Sobolev H^1, dominated by
      high modes; plain L2 might be much smaller, which would suggest the
      issue is really high-mode content, not bulk accuracy.

Approach for each saved eval_suite trajs npz:
  1. Reload the trained checkpoint and rebuild predict_gxi.
  2. At each saved time slice, pull (eta_truth, xi_truth) and pull (eta_pred,
     xi_pred). Query the surrogate at BOTH states and compare against the
     analytical G(eta)xi computed via dno_series_eval.
  3. Compute per-time:
       - rel-L2 of G(truth) - G_truth (the single-step accuracy on truth-dist
         states — should match val_loss-ish)
       - rel-L2 of G(pred) - G_truth_recomputed (single-step accuracy on
         out-of-distribution rollout states)
       - Sobolev H^1 versions of the above
  4. Compute the spectrum of |G_truth| averaged over time vs energy of
     residual G_pred - G_truth, both as |fft_k|^2 vs k. Identifies whether
     the error spectrum tracks the signal spectrum (uniform relative error)
     or piles up at high k (spectral-truncation issue).

Writes ``playground/runs/rollout_drift/<run_tag>/<regime>_drift.json`` and
companion png.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver.solvers.dno_series_jax import build_grid, dno_series_eval, make_linear_dno_symbol  # noqa: E402
from solver.evals.model_rollout import build_predict_gxi_with_depth, load_run  # noqa: E402


def rel_l2(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.linalg.norm(a - b, axis=axis) / (np.linalg.norm(b, axis=axis) + 1e-12)


def sobolev_h1_rel(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    """Relative H^1 spectral error: ||(1+k^2)^.5 (rfft(a-b))|| / ||(1+k^2)^.5 rfft(b)||."""
    n = a.shape[axis]
    fa = np.fft.rfft(a, axis=axis)
    fb = np.fft.rfft(b, axis=axis)
    nf = fa.shape[axis]
    w = np.sqrt(1.0 + np.arange(nf, dtype=np.float64) ** 2)
    shape = [1] * a.ndim
    shape[axis] = nf
    w = w.reshape(shape)
    num = np.linalg.norm((fa - fb) * w, axis=axis)
    den = np.linalg.norm(fb * w, axis=axis) + 1e-12
    return num / den


def analyse_regime(
    run_dir: Path,
    regime: str,
    out_dir: Path,
    *,
    length: float = 2.0 * np.pi,
    nx: int = 1024,
    n_time_samples: int = 32,
) -> dict:
    """For a given (run, regime), compute drift diagnostics."""
    trajs_path = run_dir / "eval_suite" / f"{regime}_trajs.npz"
    if not trajs_path.exists():
        print(f"[{regime}] skip — no trajs at {trajs_path}", flush=True)
        return {}

    print(f"[{regime}] loading {trajs_path}", flush=True)
    with np.load(trajs_path) as d:
        times = np.asarray(d["times"], dtype=np.float64)
        depths = np.asarray(d["depths"], dtype=np.float64)
        truth_eta = np.asarray(d["truth_eta"], dtype=np.float64)
        truth_xi = np.asarray(d["truth_xi"], dtype=np.float64)
        truth_gxi = np.asarray(d["truth_gxi"], dtype=np.float64)
        pred_eta = np.asarray(d["pred_eta"], dtype=np.float64)
        pred_xi = np.asarray(d["pred_xi"], dtype=np.float64)
        pred_gxi_rollout = np.asarray(d["pred_gxi"], dtype=np.float64)

    n_t, NB, nx_actual = truth_eta.shape
    assert nx_actual == nx, f"nx mismatch {nx_actual} vs {nx}"

    # Subsample time slices for efficiency.
    if n_time_samples >= n_t:
        time_idx = np.arange(n_t)
    else:
        time_idx = np.linspace(0, n_t - 1, n_time_samples, dtype=int)

    loaded = load_run(run_dir, checkpoint="best")
    predict = build_predict_gxi_with_depth(loaded)

    _, k_np = build_grid(nx, length)
    k_jax_f32 = jnp.asarray(k_np, dtype=jnp.float32)

    @jax.jit
    def truth_gxi_for(eta: jnp.ndarray, xi: jnp.ndarray, depth: jnp.ndarray) -> jnp.ndarray:
        k_local = jnp.asarray(k_np, dtype=eta.dtype)
        return dno_series_eval(eta, xi, k_local, depth.astype(eta.dtype), 6, pad_factor=8)

    # We will compute single-step error TWO ways:
    #  (1) on (eta_truth, xi_truth) — in-distribution
    #  (2) on (eta_pred, xi_pred) — out-of-distribution (rollout state)
    # For each, compare against truth recomputed via dno_series_eval.
    rl2_truth_state = np.zeros((len(time_idx), NB))
    rl2_pred_state = np.zeros((len(time_idx), NB))
    h1_truth_state = np.zeros((len(time_idx), NB))
    h1_pred_state = np.zeros((len(time_idx), NB))

    # Power-spectrum accumulators.
    n_freq = nx // 2 + 1
    pow_truth_sig = np.zeros(n_freq)
    pow_truth_err = np.zeros(n_freq)
    pow_pred_err = np.zeros(n_freq)
    pow_count = 0

    for ti_out, ti_in in enumerate(time_idx):
        for j in range(NB):
            depth = depths[j]
            log_h = float(np.log(max(depth, 1e-12)))

            # (1) Probe on truth state
            eta_t = jnp.asarray(truth_eta[ti_in, j], dtype=jnp.float32)
            xi_t = jnp.asarray(truth_xi[ti_in, j], dtype=jnp.float32)
            log_h_jax = jnp.asarray(log_h, dtype=jnp.float32)
            pred_on_truth = np.asarray(predict(eta_t, xi_t, log_h_jax), dtype=np.float64)
            truth_re = np.asarray(
                truth_gxi_for(
                    jnp.asarray(truth_eta[ti_in, j], dtype=jnp.float64),
                    jnp.asarray(truth_xi[ti_in, j], dtype=jnp.float64),
                    jnp.asarray(depth, dtype=jnp.float64),
                ),
                dtype=np.float64,
            )
            rl2_truth_state[ti_out, j] = rel_l2(pred_on_truth, truth_re, axis=-1)
            h1_truth_state[ti_out, j] = sobolev_h1_rel(pred_on_truth, truth_re, axis=-1)

            # (2) Probe on rollout state
            eta_p = jnp.asarray(pred_eta[ti_in, j], dtype=jnp.float32)
            xi_p = jnp.asarray(pred_xi[ti_in, j], dtype=jnp.float32)
            pred_on_pred = np.asarray(predict(eta_p, xi_p, log_h_jax), dtype=np.float64)
            truth_on_pred = np.asarray(
                truth_gxi_for(
                    jnp.asarray(pred_eta[ti_in, j], dtype=jnp.float64),
                    jnp.asarray(pred_xi[ti_in, j], dtype=jnp.float64),
                    jnp.asarray(depth, dtype=jnp.float64),
                ),
                dtype=np.float64,
            )
            rl2_pred_state[ti_out, j] = rel_l2(pred_on_pred, truth_on_pred, axis=-1)
            h1_pred_state[ti_out, j] = sobolev_h1_rel(pred_on_pred, truth_on_pred, axis=-1)

            # Spectra (use middle of trajectory only to avoid IC bias).
            if 0.3 * n_t <= ti_in <= 0.7 * n_t:
                fft_truth = np.fft.rfft(truth_re)
                fft_err_t = np.fft.rfft(pred_on_truth - truth_re)
                fft_err_p = np.fft.rfft(pred_on_pred - truth_on_pred)
                pow_truth_sig += np.abs(fft_truth) ** 2
                pow_truth_err += np.abs(fft_err_t) ** 2
                pow_pred_err += np.abs(fft_err_p) ** 2
                pow_count += 1

        print(
            f"[{regime}] t={times[ti_in]:6.2f}  rl2(truth)={np.nanmedian(rl2_truth_state[ti_out]):.4e}  "
            f"rl2(pred)={np.nanmedian(rl2_pred_state[ti_out]):.4e}  "
            f"h1(truth)={np.nanmedian(h1_truth_state[ti_out]):.4e}  "
            f"h1(pred)={np.nanmedian(h1_pred_state[ti_out]):.4e}",
            flush=True,
        )

    # Aggregate.
    summary = {
        "regime": regime,
        "n_ics": int(NB),
        "tmax": float(times[-1]),
        "time_idx": time_idx.tolist(),
        "times": times[time_idx].tolist(),
        "rl2_truth_state_median": np.nanmedian(rl2_truth_state, axis=-1).tolist(),
        "rl2_pred_state_median": np.nanmedian(rl2_pred_state, axis=-1).tolist(),
        "h1_truth_state_median": np.nanmedian(h1_truth_state, axis=-1).tolist(),
        "h1_pred_state_median": np.nanmedian(h1_pred_state, axis=-1).tolist(),
        "spectra": {
            "truth_signal_mean_power": (pow_truth_sig / max(pow_count, 1)).tolist(),
            "single_step_err_on_truth": (pow_truth_err / max(pow_count, 1)).tolist(),
            "single_step_err_on_pred": (pow_pred_err / max(pow_count, 1)).tolist(),
            "k": (2.0 * np.pi / length * np.arange(n_freq)).tolist(),
        },
        "pow_count": int(pow_count),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{regime}_drift.json").write_text(json.dumps(summary, indent=2))

    # Plots: drift over time + spectra
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    t_plot = np.asarray(summary["times"])
    axes[0].semilogy(t_plot, np.nanmedian(rl2_truth_state, axis=-1), label="rl2 on truth state (in-dist)")
    axes[0].semilogy(t_plot, np.nanmedian(rl2_pred_state, axis=-1), label="rl2 on pred state (rollout-dist)")
    axes[0].semilogy(t_plot, np.nanmedian(h1_truth_state, axis=-1), "--", label="H^1 on truth state")
    axes[0].semilogy(t_plot, np.nanmedian(h1_pred_state, axis=-1), "--", label="H^1 on pred state")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("median rel error of G(eta)xi over ICs")
    axes[0].set_title(f"{regime}: single-step error along rollout")
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=8)

    k_arr = np.asarray(summary["spectra"]["k"])
    axes[1].loglog(k_arr[1:], np.asarray(summary["spectra"]["truth_signal_mean_power"])[1:],
                   "k", lw=1.2, label="|G(eta)xi|^2 truth")
    axes[1].loglog(k_arr[1:], np.asarray(summary["spectra"]["single_step_err_on_truth"])[1:],
                   "tab:blue", lw=1.0, label="|err on truth state|^2")
    axes[1].loglog(k_arr[1:], np.asarray(summary["spectra"]["single_step_err_on_pred"])[1:],
                   "tab:red", lw=1.0, label="|err on pred state|^2")
    axes[1].axvline(64 * 2 * np.pi / length, color="green", ls=":", label="modes=64 cutoff")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("power")
    axes[1].set_title(f"{regime}: spectra of signal vs single-step error")
    axes[1].grid(True, alpha=0.3, which="both"); axes[1].legend(fontsize=8)
    fig.savefig(out_dir / f"{regime}_drift.png", dpi=150)
    plt.close(fig)
    print(f"[{regime}] wrote {out_dir / f'{regime}_drift.json'} and .png", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--regimes", default="tanaka_g0,bf_g0,stokes_finite,random_sea_deep,linear")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n_time_samples", type=int, default=24)
    parser.add_argument("--length", type=float, default=2.0 * np.pi)
    parser.add_argument("--nx", type=int, default=1024)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    tag = run_dir.name
    out_dir = Path(args.output_dir).resolve() if args.output_dir \
        else REPO_ROOT / "playground" / "runs" / "rollout_drift" / tag
    print(f"writing to {out_dir}", flush=True)

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    for r in regimes:
        analyse_regime(run_dir, r, out_dir,
                       length=args.length, nx=args.nx, n_time_samples=args.n_time_samples)


if __name__ == "__main__":
    main()
