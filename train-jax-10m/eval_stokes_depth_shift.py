"""Evaluate a trained FNO on Stokes waves: finite depth (ichoi=1) vs deep water (ichoi=0).

Generates fresh Stokes data for both regimes across a sweep of (n0, a0) parameters,
runs the checkpoint on each, and reports per-regime error statistics.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax.training import checkpoints

REPO_ROOT = Path(__file__).resolve().parent.parent
FNO_DIR = REPO_ROOT / "models" / "fno-jax"
DNO_DIR = REPO_ROOT / "models" / "dno-net"
for _d in (REPO_ROOT, FNO_DIR, DNO_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from fno1d import FNO1d
from dno_net import SpectralDNO
from solver.data.stokes_truth_jax import stokes_truth_trajectory
from util import NormStats, compute_log_depth, denormalize_targets, normalize_features, require_jax_devices


def load_checkpoint(run_dir: Path, checkpoint_name: str) -> tuple[Path, dict, dict]:
    checkpoint_dir = run_dir / ("best_val_ckpt" if checkpoint_name == "best" else "final_ckpt")
    metadata_path = checkpoint_dir / "metadata.json"
    restored = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=None,
        prefix="ckpt_",
        orbax_checkpointer=ocp.PyTreeCheckpointer(),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    params = jax.tree_util.tree_map(jnp.asarray, restored["params"])
    return checkpoint_dir, params, metadata


def generate_stokes_samples(
    ichoi: int,
    n0_values: list[int],
    a0_values: list[float],
    nx: int = 1024,
    length: float = 164.0,
    depth: float = 1.0,
    tmax: float = 20.0,
    dt: float = 0.1,
    n0_a0_map: dict[int, list[float]] | None = None,
) -> dict[str, np.ndarray]:
    """Generate Stokes wave snapshots across a grid of (n0, a0) parameters."""
    eta_all: list[np.ndarray] = []
    xi_all: list[np.ndarray] = []
    gxi_all: list[np.ndarray] = []
    x_grid: np.ndarray | None = None
    param_labels: list[dict[str, float]] = []

    for n0 in n0_values:
        allowed_a0 = n0_a0_map[n0] if n0_a0_map is not None else a0_values
        for a0 in allowed_a0:
            traj = stokes_truth_trajectory(
                nx=nx, length=length, depth=depth, gravity=1.0,
                dt=dt, tmax=tmax, n0=n0, a0=a0, ichoi=ichoi,
            )
            eta = np.asarray(traj["eta"], dtype=np.float32)
            xi = np.asarray(traj["xi"], dtype=np.float32)
            gxi = np.asarray(traj["gxi"], dtype=np.float32)
            if x_grid is None:
                x_grid = np.asarray(traj["x"], dtype=np.float32)

            eta_all.append(eta)
            xi_all.append(xi)
            gxi_all.append(gxi)
            for t_idx in range(eta.shape[0]):
                param_labels.append({"n0": n0, "a0": a0, "t_idx": t_idx, "ichoi": ichoi})

    return {
        "eta": np.concatenate(eta_all, axis=0),
        "xi": np.concatenate(xi_all, axis=0),
        "gxi": np.concatenate(gxi_all, axis=0),
        "x": x_grid,
        "param_labels": param_labels,
    }


def evaluate_on_stokes(
    params: dict,
    model: FNO1d | SpectralDNO,
    ns: NormStats,
    stokes_data: dict[str, np.ndarray],
    device: jax.Device,
    model_depth: float,
    batch_size: int = 256,
) -> dict[str, object]:
    eta = stokes_data["eta"]
    xi = stokes_data["xi"]
    gxi = stokes_data["gxi"]

    # Zero-mean xi to match eval_on_dno_dataset convention
    xi = xi - np.mean(xi, axis=1, keepdims=True)

    @jax.jit
    def predict_batch(current_params, batch_inputs, batch_depth):
        return model.apply({"params": current_params}, batch_inputs, batch_depth)

    prediction_chunks: list[np.ndarray] = []
    depth_value = compute_log_depth(np.full((eta.shape[0],), model_depth, dtype=np.float32))
    for start in range(0, eta.shape[0], batch_size):
        end = min(start + batch_size, eta.shape[0])
        batch_inputs = normalize_features(eta[start:end], xi[start:end], ns)
        batch_inputs_device = jax.device_put(batch_inputs, device)
        batch_depth_device = jax.device_put(depth_value[start:end], device)
        predicted_norm = np.asarray(jax.device_get(predict_batch(params, batch_inputs_device, batch_depth_device)))
        predicted_raw = denormalize_targets(predicted_norm, ns)
        prediction_chunks.append(predicted_raw.astype(np.float32))

    predictions = np.concatenate(prediction_chunks, axis=0)
    flat_pred = predictions.reshape((predictions.shape[0], -1))
    flat_target = gxi[..., None].reshape((gxi.shape[0], -1))

    rel_l2 = np.linalg.norm(flat_pred - flat_target, axis=1) / (
        np.linalg.norm(flat_target, axis=1) + 1e-12
    )
    rel_l1 = np.sum(np.abs(flat_pred - flat_target), axis=1) / (
        np.sum(np.abs(flat_target), axis=1) + 1e-12
    )

    return {
        "num_examples": int(eta.shape[0]),
        "mean_rel_l2": float(np.mean(rel_l2)),
        "median_rel_l2": float(np.median(rel_l2)),
        "std_rel_l2": float(np.std(rel_l2)),
        "p90_rel_l2": float(np.percentile(rel_l2, 90)),
        "p99_rel_l2": float(np.percentile(rel_l2, 99)),
        "mean_rel_l1": float(np.mean(rel_l1)),
        "median_rel_l1": float(np.median(rel_l1)),
        "rel_l2_per_sample": rel_l2,
        "rel_l1_per_sample": rel_l1,
        "eta_absmax": float(np.max(np.abs(eta))),
        "xi_absmax": float(np.max(np.abs(xi))),
        "gxi_absmax": float(np.max(np.abs(gxi))),
        "predictions": predictions,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate FNO on Stokes finite-depth vs deep-water.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--allow_cpu", action="store_true")
    args = parser.parse_args()

    backend, devices = require_jax_devices(allow_cpu=args.allow_cpu)
    device = devices[0]

    run_dir = (REPO_ROOT / args.run_dir).resolve()
    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    checkpoint_dir, params, metadata = load_checkpoint(run_dir, args.checkpoint)

    train_stats = metadata.get("stats")
    if train_stats is None:
        raise ValueError("Checkpoint metadata has no 'stats' entry.")
    norm_mode = config.get("norm", "minmax")
    ns = NormStats.from_dict(train_stats, mode=norm_mode)

    if config.get("model", "fno") == "spectral_dno":
        model = SpectralDNO(
            modes=int(config["modes"]),
            width=int(config["width"]),
            n_blocks=int(config.get("n_blocks", 4)),
            latent=int(config.get("latent", 64)),
            domain_length=float(config.get("domain_length", train_stats.get("domain_length", 2.0 * np.pi))),
            xi_scale=float(config.get("xi_scale", np.asarray(ns.feature_absmax).reshape(-1)[1])),
            target_scale=float(config.get("target_scale", ns.target_absmax)),
        )
    else:
        model = FNO1d(
            modes=int(config["modes"]),
            width=int(config["width"]),
            n_blocks=int(config.get("n_blocks", 4)),
            deriv_features=bool(config.get("deriv_features", False)),
        )

    # Parameter sweep: wavenumbers and amplitudes
    # Filter out combos where the 5th-order Stokes expansion diverges.
    # At n0=4, a0>=0.05 with depth=1, length=164, cosh(k0*(eta+h)) blows up.
    length = 164.0
    n0_values = [4, 8, 14, 20]
    a0_values = [0.01, 0.05, 0.1]
    valid_combos = [
        (n0, a0)
        for n0 in n0_values
        for a0 in a0_values
        if n0 * (2.0 * np.pi / length) * a0 < 0.01 or n0 >= 8
    ]
    valid_n0 = sorted({n0 for n0, _ in valid_combos})
    # Build per-n0 a0 lists so we skip divergent combos
    n0_a0_map: dict[int, list[float]] = {}
    for n0, a0 in valid_combos:
        n0_a0_map.setdefault(n0, []).append(a0)
    print(f"Valid (n0, a0) combos: {valid_combos}")

    print("Generating Stokes waves...")
    finite_data = generate_stokes_samples(ichoi=1, n0_values=valid_n0, a0_values=a0_values, n0_a0_map=n0_a0_map)
    deep_data = generate_stokes_samples(ichoi=0, n0_values=valid_n0, a0_values=a0_values, n0_a0_map=n0_a0_map)

    print(f"Finite depth samples: {finite_data['eta'].shape[0]}")
    print(f"Deep water samples: {deep_data['eta'].shape[0]}")

    print("\nEvaluating on finite depth (ichoi=1)...")
    finite_results = evaluate_on_stokes(params, model, ns, finite_data, device, model_depth=1.0, batch_size=args.batch_size)

    print("Evaluating on deep water (ichoi=0)...")
    deep_results = evaluate_on_stokes(params, model, ns, deep_data, device, model_depth=5.0, batch_size=args.batch_size)

    # Per-(n0, a0) breakdown
    def per_param_breakdown(
        data: dict[str, np.ndarray],
        results: dict[str, object],
    ) -> list[dict[str, object]]:
        rel_l2 = results["rel_l2_per_sample"]
        labels = data["param_labels"]
        seen: dict[tuple[int, float], list[int]] = {}
        for i, label in enumerate(labels):
            key = (label["n0"], label["a0"])
            seen.setdefault(key, []).append(i)

        rows = []
        for (n0, a0), indices in sorted(seen.items()):
            idx = np.array(indices)
            rows.append({
                "n0": n0,
                "a0": a0,
                "num_samples": len(indices),
                "mean_rel_l2": float(np.mean(rel_l2[idx])),
                "median_rel_l2": float(np.median(rel_l2[idx])),
                "p90_rel_l2": float(np.percentile(rel_l2[idx], 90)),
            })
        return rows

    finite_breakdown = per_param_breakdown(finite_data, finite_results)
    deep_breakdown = per_param_breakdown(deep_data, deep_results)

    # Print summary table
    def print_table(title: str, breakdown: list[dict[str, object]]) -> None:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")
        print(f"  {'n0':>4}  {'a0':>6}  {'n':>5}  {'mean_L2':>10}  {'med_L2':>10}  {'p90_L2':>10}")
        print(f"  {'-'*4}  {'-'*6}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}")
        for row in breakdown:
            print(
                f"  {row['n0']:4d}  {row['a0']:6.3f}  {row['num_samples']:5d}"
                f"  {row['mean_rel_l2']:10.4f}  {row['median_rel_l2']:10.4f}  {row['p90_rel_l2']:10.4f}"
            )

    print_table("FINITE DEPTH (ichoi=1, depth=1.0)", finite_breakdown)
    print_table("DEEP WATER (ichoi=0, depth→∞)", deep_breakdown)

    print(f"\n{'=' * 70}")
    print("  AGGREGATE COMPARISON")
    print(f"{'=' * 70}")
    for label, results in [("Finite depth", finite_results), ("Deep water", deep_results)]:
        print(
            f"  {label:15s}:  mean_L2={results['mean_rel_l2']:.4f}"
            f"  median_L2={results['median_rel_l2']:.4f}"
            f"  p90_L2={results['p90_rel_l2']:.4f}"
            f"  n={results['num_examples']}"
        )

    # Save results
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "eval_stokes_depth_shift"
    output_dir.mkdir(parents=True, exist_ok=True)

    def strip_arrays(d: dict[str, object]) -> dict[str, object]:
        return {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}

    summary = {
        "checkpoint": str(checkpoint_dir),
        "norm": norm_mode,
        "model": config.get("model", "fno"),
        "n0_values": n0_values,
        "a0_values": a0_values,
        "finite_depth": {
            "aggregate": strip_arrays(finite_results),
            "per_param": finite_breakdown,
        },
        "deep_water": {
            "aggregate": strip_arrays(deep_results),
            "per_param": deep_breakdown,
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        output_dir / "per_sample_metrics.npz",
        finite_rel_l2=finite_results["rel_l2_per_sample"],
        finite_rel_l1=finite_results["rel_l1_per_sample"],
        deep_rel_l2=deep_results["rel_l2_per_sample"],
        deep_rel_l1=deep_results["rel_l1_per_sample"],
    )

    # --- Plots ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "text.usetex": False,
        "font.family": "DejaVu Serif",
        "axes.unicode_minus": False,
    })

    # 1) Grouped bar chart: mean rel L2 by (n0, a0), finite vs deep
    fig, ax = plt.subplots(figsize=(12, 5))
    x_labels = [f"n0={r['n0']}\na0={r['a0']}" for r in finite_breakdown]
    x_pos = np.arange(len(x_labels))
    bar_w = 0.35
    finite_vals = [r["mean_rel_l2"] for r in finite_breakdown]
    deep_vals = [r["mean_rel_l2"] for r in deep_breakdown]
    ax.bar(x_pos - bar_w / 2, finite_vals, bar_w, label="Finite depth (ichoi=1)", color="#4C72B0")
    ax.bar(x_pos + bar_w / 2, deep_vals, bar_w, label="Deep water (ichoi=0)", color="#C44E52")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("Mean Relative L2 Error")
    ax.set_title("FNO Error: Finite Depth vs Deep Water Stokes Waves")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "bar_finite_vs_deep.png", dpi=150)
    plt.close(fig)
    print("  Saved bar_finite_vs_deep.png")

    # 2) Heatmaps: mean L2 as function of n0 x a0, side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax_idx, (title, breakdown) in enumerate([
        ("Finite Depth (ichoi=1)", finite_breakdown),
        ("Deep Water (ichoi=0)", deep_breakdown),
    ]):
        grid = np.full((len(a0_values), len(valid_n0)), np.nan)
        for row in breakdown:
            i = a0_values.index(row["a0"])
            j = valid_n0.index(row["n0"])
            grid[i, j] = row["mean_rel_l2"]
        ax = axes[ax_idx]
        im = ax.imshow(grid, vmin=0, vmax=1.0, cmap="RdYlGn_r", aspect="auto", origin="lower")
        ax.set_xticks(range(len(valid_n0)))
        ax.set_xticklabels(valid_n0)
        ax.set_yticks(range(len(a0_values)))
        ax.set_yticklabels(a0_values)
        ax.set_xlabel("n0 (wavenumber)")
        ax.set_ylabel("a0 (amplitude)")
        ax.set_title(title, fontweight="bold")
        for i in range(len(a0_values)):
            for j in range(len(valid_n0)):
                val = grid[i, j]
                if np.isnan(val):
                    ax.text(j, i, "N/A", ha="center", va="center", fontsize=9, color="gray")
                else:
                    color = "white" if val > 0.5 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(im, ax=axes, label="Mean Relative L2", shrink=0.8)
    fig.suptitle("FNO Prediction Error by Stokes Wave Parameters", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "heatmap_finite_vs_deep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved heatmap_finite_vs_deep.png")

    # 3) Representative sample plots: best/worst for each regime
    def plot_representative_cases(
        regime_label: str,
        data: dict[str, np.ndarray],
        results: dict[str, object],
        filename: str,
    ) -> None:
        rel_l2 = results["rel_l2_per_sample"]
        predictions = results["predictions"]
        eta = data["eta"]
        xi = data["xi"]
        gxi = data["gxi"]
        x = data["x"]
        labels_list = data["param_labels"]

        order = np.argsort(rel_l2)
        chosen = [order[0], order[len(order) // 4], order[len(order) // 2], order[-1]]
        titles = ["Best", "25th pctl", "Median", "Worst"]

        fig, axes = plt.subplots(3, 4, figsize=(18, 10))
        for col, (idx, title) in enumerate(zip(chosen, titles)):
            lbl = labels_list[idx]
            tag = f"n0={lbl['n0']}, a0={lbl['a0']}"

            axes[0, col].plot(x, eta[idx], color="#1f77b4")
            axes[0, col].set_title(f"{title} eta(x)\n{tag}", fontsize=10)
            axes[0, col].set_ylabel("eta" if col == 0 else "")

            axes[1, col].plot(x, xi[idx], color="#2ca02c")
            axes[1, col].set_title("xi(x)", fontsize=10)
            axes[1, col].set_ylabel("xi" if col == 0 else "")

            axes[2, col].plot(x, gxi[idx], color="black", label="Ground Truth", linewidth=1.5)
            axes[2, col].plot(x, predictions[idx, :, 0], color="#d62728", label="Prediction", linewidth=1.0, linestyle="--")
            axes[2, col].set_title(f"G(eta)xi  L2={rel_l2[idx]:.3e}", fontsize=10)
            axes[2, col].set_ylabel("G(eta)xi" if col == 0 else "")
            axes[2, col].set_xlabel("x")
            if col == 0:
                axes[2, col].legend(fontsize=8)

        fig.suptitle(f"{regime_label}: Representative Samples", fontweight="bold", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {filename}")

    plot_representative_cases("Finite Depth (ichoi=1)", finite_data, finite_results, "representative_finite.png")
    plot_representative_cases("Deep Water (ichoi=0)", deep_data, deep_results, "representative_deep.png")

    # 4) Histogram: rel L2 distributions overlaid
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1.2, 50)
    ax.hist(finite_results["rel_l2_per_sample"], bins=bins, alpha=0.6, label="Finite depth", color="#4C72B0", density=True)
    ax.hist(deep_results["rel_l2_per_sample"], bins=bins, alpha=0.6, label="Deep water", color="#C44E52", density=True)
    ax.set_xlabel("Relative L2 Error")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Prediction Error: Finite Depth vs Deep Water")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "histogram_rel_l2.png", dpi=150)
    plt.close(fig)
    print("  Saved histogram_rel_l2.png")

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
