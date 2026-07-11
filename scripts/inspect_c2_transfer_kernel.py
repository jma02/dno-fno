"""Compute the fixed-eta transfer kernel for the C2 final CS-DNO (CPU).

For a fixed surface profile eta taken from a failing Tanaka rollout, this
replaces the actual xi with a bank of single-Fourier-mode perturbations
cos(k_in x) and records the resulting gxi spectrum. Because the model is
linear in xi, this is the discretized action of the eta-dependent linear
operator L_eta : xi -> gxi(eta, xi). The same is done with the order-6
Craig-Sulem truth operator for comparison.

This is diagnostic A2 from the NaN-mechanism discussion. It tells us whether
the moderate learned multipliers from A1 are turned into a high-k cascade by
the phi(eta) projections and the physical-space products inside the CS blocks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Force CPU before any JAX import.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from solver.evals import model_rollout as mr
from solver.solvers import dno_series_jax as ds


def load_failing_state(run_dir: Path, regime: str, case_id: int, frame: int):
    """Load a single (eta, xi, depth, time) from a saved surrogate rollout."""
    npz_dir = run_dir / f"eval_final_2reg_f64h_batched_cached_20260707_133545" / regime
    npz = np.load(npz_dir / f"{regime}_trajs.npz")
    case_idx = int(np.where(npz["case_ids"] == case_id)[0][0])
    depth = float(npz["depths"][case_idx])
    time = float(npz["times"][frame])
    eta = np.asarray(npz["pred_eta"][frame, case_idx, :], dtype=np.float64)
    xi = np.asarray(npz["pred_xi"][frame, case_idx, :], dtype=np.float64)
    truth_eta = np.asarray(npz["truth_eta"][frame, case_idx, :], dtype=np.float64)
    return eta, xi, truth_eta, depth, time


def build_transfer_kernel(
    predict_fn,
    eta: jnp.ndarray,
    log_depth: jnp.ndarray,
    h: float,
    k_full: jnp.ndarray,
    x: jnp.ndarray,
    k_in_modes: list[int] | None = None,
):
    """Return (model_resp, truth_resp, k_out, k_in) arrays.

    model_resp and truth_resp have shape (n_k_out, n_k_in) and contain
    |rfft(gxi)| magnitudes for unit-amplitude cos(k_in x) inputs.
    """
    nx = x.shape[-1]
    if k_in_modes is None:
        k_in_modes = list(range(1, nx // 8))
    k_in = np.array(k_in_modes)
    n_k_in = len(k_in)

    # Build batched test xi: each row is cos(2π k_in x / L).
    xi_test = np.cos(2.0 * np.pi * k_in[:, None] * x[None, :] / (x[-1] + x[1] - x[0]))
    eta_batch = np.tile(eta, (n_k_in, 1))
    log_depth_batch = np.full((n_k_in,), float(log_depth))

    # Model prediction.
    gxi_model = predict_fn(
        jnp.asarray(eta_batch, dtype=jnp.float32),
        jnp.asarray(xi_test, dtype=jnp.float32),
        jnp.asarray(log_depth_batch, dtype=jnp.float32),
    )  # (n_k_in, nx) f64 from cast
    gxi_model = np.asarray(gxi_model).astype(np.float64)
    spec_model = np.abs(np.fft.rfft(gxi_model, axis=-1))  # (n_k_in, n_k_out)

    # Truth operator.
    truth_fn = jax.jit(lambda xi: ds.dno_series_eval(eta, xi, k_full, h, order=6, pad_factor=8))
    # Run one-by-one to avoid huge padded batch memory on CPU.
    spec_truth = []
    for row in xi_test:
        gxi_truth = np.asarray(truth_fn(jnp.asarray(row, dtype=jnp.float64)))
        spec_truth.append(np.abs(np.fft.rfft(gxi_truth)))
    spec_truth = np.stack(spec_truth, axis=0)

    model_resp = spec_model.T  # (n_k_out, n_k_in)
    truth_resp = spec_truth.T
    k_out = np.arange(model_resp.shape[0])
    return model_resp, truth_resp, k_out, k_in


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path,
                        default=Path("/home/johnma/dno-fno/outputs/c2_stage_match_gain_from_v85b_20260707_053707"))
    parser.add_argument("--regime", type=str, default="tanaka_g0")
    parser.add_argument("--case_id", type=int, default=5)
    parser.add_argument("--frame", type=int, default=40,
                        help="Frame index in the saved rollout (0 = t=0, dt=0.8).")
    parser.add_argument("--k_in_max", type=int, default=128)
    parser.add_argument("--extra_k_in", type=int, nargs="*", default=[150, 200, 256])
    parser.add_argument("--use_truth_eta", action="store_true",
                        help="Use truth eta instead of pred eta to build the operator.")
    parser.add_argument("--block_k_cut", type=int, default=0,
                        help="Override cs_block_k_cut for inference (0 = use checkpoint value).")
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if args.out_dir is None:
        suffix = "truth_eta" if args.use_truth_eta else "pred_eta"
        bc = f"_bc{args.block_k_cut}" if args.block_k_cut > 0 else ""
        out_dir = run_dir / f"transfer_kernel_{suffix}_case{args.case_id}_frame{args.frame}{bc}"
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config_overrides = {}
    if args.block_k_cut > 0:
        config_overrides["cs_block_k_cut"] = args.block_k_cut
    loaded = mr.load_run(run_dir, checkpoint="final", config_overrides=config_overrides)
    predict_fn = mr.build_predict_gxi_batched(loaded)

    eta_state, xi_state, truth_eta_state, h, time = load_failing_state(
        run_dir, args.regime, args.case_id, args.frame
    )
    eta = truth_eta_state if args.use_truth_eta else eta_state

    x, k_full = ds.build_grid(eta.shape[-1], float(loaded.config["domain_length"]))
    x = np.asarray(x)
    k_full = np.asarray(k_full)
    log_depth = float(np.log(h))

    k_in_modes = list(range(1, args.k_in_max + 1))
    if args.extra_k_in:
        k_in_modes += [k for k in args.extra_k_in if k > args.k_in_max]

    print(f"Run: {run_dir}")
    print(f"Case {args.case_id}, h={h:.4f}, t={time:.2f}, frame={args.frame}, eta source={'truth' if args.use_truth_eta else 'pred'}")
    print(f"Computing transfer kernel for {len(k_in_modes)} input modes ...")

    model_resp, truth_resp, k_out, k_in = build_transfer_kernel(
        predict_fn, jnp.asarray(eta, dtype=jnp.float64), log_depth, h, k_full, x, k_in_modes
    )

    ratio = model_resp / (truth_resp + 1e-30)
    log_ratio = np.log10(np.maximum(ratio, 1e-12))

    # Summary: worst output modes and hot (k_in, k_out) pairs.
    high_band = (k_out >= 96) & (k_out < 128)
    low_band = (k_out >= 0) & (k_out < 32)
    mid_band = (k_out >= 32) & (k_out < 96)

    summary = {
        "case_id": args.case_id,
        "frame": args.frame,
        "time": time,
        "depth": h,
        "eta_source": "truth" if args.use_truth_eta else "pred",
        "n_k_in": len(k_in_modes),
        "max_ratio_overall": float(np.max(ratio)),
        "k_out_max_ratio": int(k_out[np.unravel_index(np.argmax(ratio), ratio.shape)[0]]),
        "k_in_max_ratio": int(k_in[np.unravel_index(np.argmax(ratio), ratio.shape)[1]]),
    }

    # Per-output-band max ratios.
    for band_name, mask in [("low", low_band), ("mid", mid_band), ("high", high_band)]:
        sub = np.where(mask[:, None], ratio, -np.inf)
        idx = np.unravel_index(np.argmax(sub), sub.shape)
        summary[f"max_ratio_{band_name}"] = float(ratio[idx])
        summary[f"max_ratio_{band_name}_k_out"] = int(k_out[idx[0]])
        summary[f"max_ratio_{band_name}_k_in"] = int(k_in[idx[1]])

    # Top 20 hot pairs in the high band.
    sub_high = np.where(high_band[:, None], ratio, -np.inf)
    flat_idx = np.argsort(sub_high.ravel())[-20:][::-1]
    hot_pairs = []
    for fidx in flat_idx:
        i, j = np.unravel_index(fidx, sub_high.shape)
        hot_pairs.append({
            "k_out": int(k_out[i]),
            "k_in": int(k_in[j]),
            "model": float(model_resp[i, j]),
            "truth": float(truth_resp[i, j]),
            "ratio": float(ratio[i, j]),
        })
    summary["hot_pairs_high_band"] = hot_pairs

    # Per-k_in high-band energy and ratio.
    model_high_energy = np.sum(model_resp[high_band, :] ** 2, axis=0)
    truth_high_energy = np.sum(truth_resp[high_band, :] ** 2, axis=0)
    summary["per_k_in"] = [
        {
            "k_in": int(k_in[j]),
            "model_high_energy": float(model_high_energy[j]),
            "truth_high_energy": float(truth_high_energy[j]),
            "high_energy_ratio": float(model_high_energy[j] / (truth_high_energy[j] + 1e-30)),
            "max_ratio_in_high": float(np.max(ratio[high_band, j])),
            "k_out_of_max": int(k_out[high_band][np.argmax(ratio[high_band, j])]),
        }
        for j in range(len(k_in))
    ]

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plots.
    k_out_plot = k_out[: np.argmax(k_out >= 128) + 1]
    model_plot = model_resp[: len(k_out_plot), :]
    truth_plot = truth_resp[: len(k_out_plot), :]
    ratio_plot = ratio[: len(k_out_plot), :]
    log_ratio_plot = log_ratio[: len(k_out_plot), :]

    vmax = max(np.log10(np.maximum(model_plot, 1e-12).max()),
               np.log10(np.maximum(truth_plot, 1e-12).max()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, data, title, cmap, vmin in [
        (axes[0], np.log10(np.maximum(model_plot, 1e-12)), "model |gxi(k_out)|", "viridis", -8),
        (axes[1], np.log10(np.maximum(truth_plot, 1e-12)), "truth |gxi(k_out)|", "viridis", -8),
        (axes[2], np.clip(log_ratio_plot, -2, 3), "log10(model/truth)", "RdBu_r", -2),
    ]:
        im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax if cmap != "RdBu_r" else 3)
        ax.set_xlabel("k_in index")
        ax.set_ylabel("k_out")
        ax.set_title(title)
        ax.set_xticks(np.arange(0, len(k_in), max(1, len(k_in) // 8)))
        ax.set_xticklabels([str(k_in[i]) for i in ax.get_xticks()], fontsize=6)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "transfer_kernel_heatmaps.png", dpi=200)
    plt.close(fig)

    # Line plots.
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # Max ratio per k_out.
    ax = axes[0, 0]
    ax.plot(k_out_plot, np.max(ratio_plot, axis=1), lw=0.8)
    ax.axvspan(96, 128, color="red", alpha=0.1)
    ax.set_title("max over k_in of model/truth")
    ax.set_xlabel("k_out")
    ax.set_ylabel("max ratio")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    # Max ratio per k_in.
    ax = axes[0, 1]
    ax.plot(k_in, np.max(ratio, axis=0), lw=0.8)
    ax.axvspan(96, 128, color="red", alpha=0.1)
    ax.set_title("max over k_out of model/truth")
    ax.set_xlabel("k_in")
    ax.set_ylabel("max ratio")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    # High-band energy ratio per k_in.
    ax = axes[1, 0]
    ax.plot(k_in, model_high_energy / (truth_high_energy + 1e-30), lw=0.8)
    ax.axvspan(96, 128, color="red", alpha=0.1)
    ax.set_title("high-band [96,128) energy ratio per k_in")
    ax.set_xlabel("k_in")
    ax.set_ylabel("model/truth energy")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    # Response at k_out=101 vs k_in.
    ax = axes[1, 1]
    k101 = np.argmin(np.abs(k_out - 101))
    ax.plot(k_in, model_resp[k101, :], label="model", lw=0.8)
    ax.plot(k_in, truth_resp[k101, :], label="truth", lw=0.8)
    ax.set_title("|gxi(k_out=101)| vs k_in")
    ax.set_xlabel("k_in")
    ax.set_ylabel("|gxi(k=101)|")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "transfer_kernel_lines.png", dpi=200)
    plt.close(fig)

    print(f"Saved outputs to {out_dir}")
    print(f"Max overall ratio = {summary['max_ratio_overall']:.2f} at (k_out={summary['k_out_max_ratio']}, k_in={summary['k_in_max_ratio']})")
    print(f"Max high-band ratio = {summary['max_ratio_high']:.2f} at (k_out={summary['max_ratio_high_k_out']}, k_in={summary['max_ratio_high_k_in']})")


if __name__ == "__main__":
    main()
