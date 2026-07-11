"""Inspect learned DepthAwareMultiplier spectra for a CS-DNO checkpoint (CPU).

Loads a trained cs_dno checkpoint, evaluates each block's learned Fourier
multipliers M_i(k, h) / M_out(k, h) over a grid of physical depths h, and
compares them to the flat-surface linear symbol G0(k, h) = k tanh(h k).

This is diagnostic A1 from the NaN-mechanism discussion: if the learned
multipliers have sharp high-k peaks at the depths of failing Tanaka cases,
the NaN cascade is likely a learned resonant Fourier multiplier.
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

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
from flax.training import checkpoints

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "models" / "dno-net"))

from dno_net_v2 import DepthAwareMultiplier


def load_params(run_dir: Path) -> dict:
    ckpt_dir = run_dir / "final_ckpt"
    if not ckpt_dir.exists():
        ckpt_dir = run_dir / "best_val_ckpt"
    restored = checkpoints.restore_checkpoint(
        ckpt_dir,
        target=None,
        prefix="ckpt_",
        orbax_checkpointer=ocp.PyTreeCheckpointer(),
    )
    params = restored["params"]
    return jax.tree_util.tree_map(jnp.asarray, params)


def compute_multiplier_spectrum(
    params: dict,
    block_idx: int,
    h: float,
    n_freq: int,
    domain_length: float,
    h_clip_max: float,
    mult_hidden: int,
    latent: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (k_arr, M_abs) for one block and depth.

    M_abs has shape (n_freq, latent) — absolute value of the learned
    per-branch, per-mode multiplier.
    """
    log_h = np.log(min(h, h_clip_max))
    depth = jnp.array([[log_h]], dtype=jnp.float32)

    block_params = params[f"cs_block_{block_idx}"]
    mult_params = block_params["m_shared"]

    multiplier = DepthAwareMultiplier(
        out_channels=latent,
        domain_length=domain_length,
        h_clip_max=h_clip_max,
        hidden=mult_hidden,
        zero_init_output=False,
    )
    out = multiplier.apply({"params": mult_params}, depth, n_freq)
    out = jnp.squeeze(out, axis=0)  # (n_freq, latent)

    k_arr = (2.0 * jnp.pi / domain_length) * jnp.arange(n_freq)
    return k_arr, jnp.abs(out)


def compute_g0_symbol(k_arr: jnp.ndarray, h: float) -> jnp.ndarray:
    return jnp.abs(k_arr * jnp.tanh(h * k_arr))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path,
                        default=Path("/home/johnma/dno-fno/outputs/c2_stage_match_gain_from_v85b_20260707_053707"))
    parser.add_argument("--n_freq", type=int, default=513,
                        help="Number of positive Fourier modes; nx=1024 -> 513.")
    parser.add_argument("--h_values", type=float, nargs="+",
                        default=[0.05, 0.1, 0.2, 0.234, 0.276, 0.293, 1.0])
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text())
    if args.out_dir is None:
        out_dir = run_dir / "multiplier_spectra"
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    params = load_params(run_dir)
    n_blocks = int(config["n_blocks"])
    latent = int(config["latent"])
    mult_hidden = int(config["cs_mult_hidden"])
    domain_length = float(config["domain_length"])
    h_clip_max = float(config.get("h_clip_max", 5.0))
    nx = 2 * (args.n_freq - 1)

    print(f"Run: {run_dir}")
    print(f"n_blocks={n_blocks}, latent={latent}, mult_hidden={mult_hidden}")
    print(f"domain_length={domain_length:.6f}, h_clip_max={h_clip_max}")
    print(f"Evaluating at h = {args.h_values}")

    # Compute spectra.
    spectra: dict[float, list[jnp.ndarray]] = {}
    g0_by_h: dict[float, jnp.ndarray] = {}
    k_arr = None
    for h in args.h_values:
        block_spectra = []
        g0 = None
        for b in range(n_blocks):
            k, M_abs = compute_multiplier_spectrum(
                params, b, h, args.n_freq, domain_length, h_clip_max,
                mult_hidden, latent,
            )
            if k_arr is None:
                k_arr = np.asarray(k)
            if g0 is None:
                g0 = np.asarray(compute_g0_symbol(k, h))
            block_spectra.append(np.asarray(M_abs))
        spectra[h] = block_spectra
        g0_by_h[h] = g0

    # Summary statistics.
    summary: dict[str, dict] = {}
    for h in args.h_values:
        g0 = g0_by_h[h]
        g0_max = float(np.max(g0))
        g0_kmax = int(np.argmax(g0))
        summary[str(h)] = {
            "g0_max": g0_max,
            "g0_max_k": g0_kmax,
            "blocks": [],
        }
        for b, M_abs in enumerate(spectra[h]):
            # Per-branch statistics over k for each branch.
            max_over_k_per_branch = np.max(M_abs, axis=0)
            mean_over_k_per_branch = np.mean(M_abs, axis=0)
            p95_over_k_per_branch = np.percentile(M_abs, 95, axis=0)

            block_summary = {
                "max_over_k_per_branch_p95": float(np.percentile(max_over_k_per_branch, 95)),
                "max_over_k_per_branch_max": float(np.max(max_over_k_per_branch)),
                "mean_over_k_per_branch_p95": float(np.percentile(mean_over_k_per_branch, 95)),
                "max_to_g0_ratio": float(np.max(max_over_k_per_branch) / (g0_max + 1e-30)),
                "k_of_global_max": int(np.unravel_index(np.argmax(M_abs), M_abs.shape)[0]),
                "branch_of_global_max": int(np.unravel_index(np.argmax(M_abs), M_abs.shape)[1]),
            }
            summary[str(h)]["blocks"].append(block_summary)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot 1: per-depth figure showing per-block mean/max spectra vs G0.
    n_h = len(args.h_values)
    n_cols = min(3, n_h)
    n_rows = (n_h + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    for idx, h in enumerate(args.h_values):
        ax = axes[idx // n_cols, idx % n_cols]
        g0 = g0_by_h[h]
        for b, M_abs in enumerate(spectra[h]):
            mean_across_branches = np.mean(M_abs, axis=1)
            max_across_branches = np.max(M_abs, axis=1)
            p95_across_branches = np.percentile(M_abs, 95, axis=1)
            ax.plot(k_arr, mean_across_branches, alpha=0.5, lw=0.8, label=f"b{b} mean" if b == 0 else None)
            ax.plot(k_arr, p95_across_branches, alpha=0.5, lw=0.8, ls="--", label=f"b{b} p95" if b == 0 else None)
        ax.plot(k_arr, g0, "k-", lw=2.0, label="G0(k)=k tanh(hk)")
        ax.set_title(f"h={h:.3f}")
        ax.set_xlabel("|k|")
        ax.set_ylabel("multiplier magnitude")
        ax.set_yscale("log")
        ax.set_xlim(0, args.n_freq - 1)
        ax.grid(True, which="both", ls=":", alpha=0.5)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=6)
    for j in range(idx + 1, n_rows * n_cols):
        fig.delaxes(axes[j // n_cols, j % n_cols])
    fig.tight_layout()
    fig.savefig(out_dir / "multiplier_spectra_per_depth.png", dpi=200)
    plt.close(fig)

    # Plot 2: for the failing depths, show each block's max-per-k and the
    # branch index of the max, plus a heatmap of M_abs for block 0.
    failing_hs = [h for h in args.h_values if h >= 0.2]
    if failing_hs:
        fig, axes = plt.subplots(len(failing_hs), 2, figsize=(10, 3 * len(failing_hs)), squeeze=False)
        for idx, h in enumerate(failing_hs):
            M0 = spectra[h][0]
            im = axes[idx, 0].imshow(
                M0.T,
                aspect="auto",
                origin="lower",
                extent=[0, args.n_freq - 1, 0, latent - 1],
                norm=matplotlib.colors.LogNorm(vmin=max(1e-6, np.min(M0[M0 > 0])), vmax=np.max(M0)),
            )
            axes[idx, 0].set_title(f"h={h:.3f}, block 0 |M(k,branch)|")
            axes[idx, 0].set_xlabel("|k|")
            axes[idx, 0].set_ylabel("branch")
            fig.colorbar(im, ax=axes[idx, 0])

            for b, M_abs in enumerate(spectra[h]):
                axes[idx, 1].plot(k_arr, np.max(M_abs, axis=1), alpha=0.7, lw=0.8, label=f"b{b}")
            axes[idx, 1].plot(k_arr, g0_by_h[h], "k-", lw=2.0, label="G0")
            axes[idx, 1].set_title(f"h={h:.3f}, max over branches")
            axes[idx, 1].set_xlabel("|k|")
            axes[idx, 1].set_ylabel("max |M|")
            axes[idx, 1].set_yscale("log")
            axes[idx, 1].set_xlim(0, args.n_freq - 1)
            axes[idx, 1].grid(True, which="both", ls=":", alpha=0.5)
            if idx == 0:
                axes[idx, 1].legend(loc="upper right", fontsize=5, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "multiplier_spectra_failing_depths.png", dpi=200)
        plt.close(fig)

    # Plot 3: ratio of learned multiplier magnitude to G0 symbol, per depth.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    for idx, h in enumerate(args.h_values):
        ax = axes[idx // n_cols, idx % n_cols]
        g0 = g0_by_h[h]
        g0_safe = np.where(g0 > 1e-12, g0, 1e-12)
        for b, M_abs in enumerate(spectra[h]):
            ratio = np.max(M_abs, axis=1) / g0_safe
            ax.plot(k_arr, ratio, alpha=0.7, lw=0.8, label=f"b{b}")
        ax.axhline(1.0, color="k", ls="--", lw=1.0, label="G0")
        ax.set_title(f"h={h:.3f}, max|M|/G0")
        ax.set_xlabel("|k|")
        ax.set_ylabel("ratio")
        ax.set_yscale("log")
        ax.set_xlim(0, args.n_freq - 1)
        ax.grid(True, which="both", ls=":", alpha=0.5)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=5, ncol=2)
    for j in range(idx + 1, n_rows * n_cols):
        fig.delaxes(axes[j // n_cols, j % n_cols])
    fig.tight_layout()
    fig.savefig(out_dir / "multiplier_g0_ratio.png", dpi=200)
    plt.close(fig)

    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
