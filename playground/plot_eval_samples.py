"""Generate dark-theme evaluation plots for beamer slides.

For each dataset subset, produces:
  - 4-column sample comparison (best / median / worst / random)
    showing truth vs prediction for G(eta)xi, with eta shown above
  - Rel-L2 error histogram

Usage:
    uv run python playground/plot_eval_samples.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fno_jax.fno1d import FNO1d

# ---- Style constants (matching render_rollout_movie.py) ----
SKY = "#0b1d35"
BLUE = "#8ed6ff"
RED = "#ff5a5f"
GREEN = "#5aff8a"
ORANGE = "#ffb05a"
PURPLE = "#c88aff"
TEXT = "#c8e0f0"
AXIS = "#aac8e0"
SPINE = "#304a60"

matplotlib.rcParams["mathtext.fontset"] = "cm"


def style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(SKY)
    ax.tick_params(colors=AXIS, labelsize=7)
    ax.grid(True, alpha=0.15, color="#6c8aa3")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(SPINE)


def plot_eval_samples(
    x: np.ndarray,
    eta: np.ndarray,
    gxi_true: np.ndarray,
    gxi_pred: np.ndarray,
    rel_l2: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Plot best/median/worst/random samples with truth vs prediction."""
    sorted_idx = np.argsort(rel_l2)
    rng = np.random.default_rng(42)
    picks = {
        "Best": sorted_idx[0],
        "Median": sorted_idx[len(sorted_idx) // 2],
        "Worst": sorted_idx[-1],
        "Random": rng.integers(0, len(rel_l2)),
    }

    fig, axes = plt.subplots(
        2, 4, figsize=(16, 5.5), facecolor=SKY,
        gridspec_kw={"height_ratios": [0.6, 1.0]},
    )

    for col, (label, idx) in enumerate(picks.items()):
        err = rel_l2[idx]

        # Top row: eta
        ax_eta = axes[0, col]
        style_ax(ax_eta)
        ax_eta.plot(x, eta[idx], color=BLUE, lw=0.9, alpha=0.95)
        ax_eta.fill_between(x, eta[idx], 0, color=BLUE, alpha=0.06)
        ax_eta.set_title(
            f"{label}  (rel-$L^2$ = {err:.2e})",
            color=TEXT, fontsize=9, pad=4,
        )
        if col == 0:
            ax_eta.set_ylabel(r"$\eta(x)$", color=AXIS, fontsize=10, rotation=0, labelpad=14)

        # Bottom row: G(eta)xi — truth vs prediction
        ax_g = axes[1, col]
        style_ax(ax_g)
        ax_g.plot(x, gxi_true[idx], color=BLUE, lw=1.0, alpha=0.9, label="Truth")
        ax_g.plot(x, gxi_pred[idx], color=RED, lw=1.0, alpha=0.85, label="FNO", linestyle="--")
        ax_g.set_xlabel("$x$", color=AXIS, fontsize=8)
        if col == 0:
            ax_g.set_ylabel(r"$\mathcal{G}\xi$", color=AXIS, fontsize=10, rotation=0, labelpad=14)
        if col == 3:
            leg = ax_g.legend(loc="upper right", frameon=False, fontsize=8)
            for txt in leg.get_texts():
                txt.set_color(TEXT)

    fig.suptitle(title, color=TEXT, fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.4, w_pad=0.6)
    fig.savefig(output_path, dpi=180, facecolor=SKY, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


def plot_error_histogram(
    rel_l2: np.ndarray,
    output_path: Path,
    title: str,
    summary: dict,
) -> None:
    """Plot a dark-themed rel-L2 histogram with summary stats."""
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SKY)
    style_ax(ax)

    # Clip outliers for cleaner histogram
    clip = np.percentile(rel_l2, 99.5)
    clipped = rel_l2[rel_l2 <= clip]

    ax.hist(clipped, bins=80, color=BLUE, alpha=0.7, edgecolor=SKY, linewidth=0.3)

    # Mark key percentiles
    mean_v = summary["mean_rel_l2"]
    med_v = summary["median_rel_l2"]
    p95_v = summary.get("p95_rel_l2")
    p99_v = summary.get("p99_rel_l2")

    ymax = ax.get_ylim()[1]
    ax.axvline(mean_v, color=RED, lw=1.3, alpha=0.9, label=f"Mean = {mean_v:.4f}")
    ax.axvline(med_v, color=GREEN, lw=1.3, alpha=0.9, label=f"Median = {med_v:.4f}")
    if p95_v is not None:
        ax.axvline(p95_v, color=ORANGE, lw=1.1, alpha=0.8, linestyle="--", label=f"P95 = {p95_v:.4f}")
    if p99_v is not None:
        ax.axvline(p99_v, color=PURPLE, lw=1.1, alpha=0.8, linestyle=":", label=f"P99 = {p99_v:.4f}")

    ax.set_xlabel("Relative $L^2$ Error", color=AXIS, fontsize=10)
    ax.set_ylabel("Count", color=AXIS, fontsize=10)

    leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(TEXT)

    fig.suptitle(title, color=TEXT, fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=180, facecolor=SKY, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


# ---- Model loading (reused from eval_on_dno_dataset.py) ----
def load_model(run_dir: Path) -> tuple:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ns = json.loads((run_dir / "norm_stats.json").read_text(encoding="utf-8"))
    fa = np.asarray(ns["feature_absmax"], dtype=np.float32)
    ta = float(ns["target_absmax"]) if ns["target_absmax"] > 0 else 1.0
    model = FNO1d(
        modes=config["modes"], width=config["width"],
        n_blocks=config["n_blocks"], xi_scale=float(fa[1]), target_scale=ta,
    )
    with np.load(run_dir / "best_params.npz") as f:
        flat = [jnp.asarray(f[k]) for k in sorted(f.files, key=lambda s: int(s.split("_")[1]))]
    with open(run_dir / "tree_def.pkl", "rb") as fh:
        tree_def = pickle.load(fh)
    params = tree_def.unflatten(flat)
    return model, params, ns


def normalize_features(eta: np.ndarray, xi: np.ndarray, ns: dict) -> np.ndarray:
    stacked = np.stack((eta, xi), axis=-1).astype(np.float32)
    absmax = np.asarray(ns["feature_absmax"], dtype=np.float32).reshape(1, 1, 2)
    return stacked / np.where(absmax > 0, absmax, 1.0)


def denormalize_targets(arr: np.ndarray, ns: dict) -> np.ndarray:
    ta = float(ns["target_absmax"]) if ns["target_absmax"] > 0 else 1.0
    return arr * ta


def run_inference(
    model: FNO1d, params: dict, ns: dict,
    eta: np.ndarray, xi: np.ndarray, depth: float,
    batch_size: int = 64,
) -> np.ndarray:
    """Run batched FNO inference, return predictions (N, nx)."""
    log_depth = np.log(depth).astype(np.float32)

    @jax.jit
    def predict(p: dict, x: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
        return model.apply({"params": p}, x, d)

    parts = []
    for start in range(0, len(eta), batch_size):
        end = min(start + batch_size, len(eta))
        inp = normalize_features(eta[start:end], xi[start:end], ns)
        depth_arr = jnp.full((end - start, 1), log_depth)
        pred_norm = predict(params, jnp.asarray(inp), depth_arr)
        pred_raw = denormalize_targets(np.asarray(jax.device_get(pred_norm)), ns)
        parts.append(pred_raw[..., 0])
    return np.concatenate(parts, axis=0)


def main() -> None:
    run_dir = Path("playground/runs/combined_fno")
    out_dir = Path("notes/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, params, ns = load_model(run_dir)

    # ================================================================
    # 1. DNO Dataset subsets: soliton, linear
    # ================================================================
    dno_data = "data/dno_dataset.npz"
    rng = np.random.default_rng(42)

    for subset, depth, max_n in [("soliton", 1.0, 30030), ("linear", 1.0, 5000)]:
        print(f"\n--- {subset} (h={depth}) ---")
        with np.load(dno_data) as f:
            eta = np.asarray(f[f"{subset}_eta"], dtype=np.float32)
            xi = np.asarray(f[f"{subset}_xi"], dtype=np.float32)
            gxi = np.asarray(f[f"{subset}_Gxi"], dtype=np.float32)
        xi -= xi.mean(axis=1, keepdims=True)

        if len(eta) > max_n:
            idx = rng.choice(len(eta), size=max_n, replace=False)
            eta, xi, gxi = eta[idx], xi[idx], gxi[idx]

        pred = run_inference(model, params, ns, eta, xi, depth)
        rel_l2 = np.linalg.norm(pred - gxi, axis=1) / (np.linalg.norm(gxi, axis=1) + 1e-12)
        x = np.asarray(np.load(dno_data)["x"], dtype=np.float32)

        summary = json.loads((run_dir / f"eval_{subset}.json").read_text())
        label = subset.title()

        print(f"  mean={np.mean(rel_l2):.4f}, median={np.median(rel_l2):.4f}")
        plot_eval_samples(
            x, eta, gxi, pred, rel_l2,
            out_dir / f"eval_{subset}_samples.png",
            f"{label} Eval ($h = {depth:.0f}$, $n = {len(eta)}$): "
            f"mean rel-$L^2$ = {np.mean(rel_l2):.4f}",
        )
        plot_error_histogram(
            rel_l2, out_dir / f"eval_{subset}_hist.png",
            f"{label} Rel-$L^2$ Distribution ($h = {depth:.0f}$, $n = {len(eta)}$)",
            summary,
        )

    # ================================================================
    # 2. Tanaka validation (from training split)
    # ================================================================
    print("\n--- Tanaka validation ---")
    import zipfile
    tanaka_path = "data/tanaka_1_clean_sub500k.npz"
    with np.load(tanaka_path) as f:
        t_x = np.asarray(f["x"], dtype=np.float32)
        # Gather samples from first shard
        t_eta = np.asarray(f["eta_batch_0000"], dtype=np.float32)
        t_xi = np.asarray(f["xi_batch_0000"], dtype=np.float32)
        t_gxi = np.asarray(f["gxi_batch_0000"], dtype=np.float32)

    # Subsample to 5000
    if len(t_eta) > 5000:
        idx = rng.choice(len(t_eta), size=5000, replace=False)
        t_eta, t_xi, t_gxi = t_eta[idx], t_xi[idx], t_gxi[idx]

    pred = run_inference(model, params, ns, t_eta, t_xi, 1.0)
    rel_l2 = np.linalg.norm(pred - t_gxi, axis=1) / (np.linalg.norm(t_gxi, axis=1) + 1e-12)

    summary = json.loads((run_dir / "eval_tanaka.json").read_text())
    print(f"  mean={np.mean(rel_l2):.4f}, median={np.median(rel_l2):.4f}")

    plot_eval_samples(
        t_x, t_eta, t_gxi, pred, rel_l2,
        out_dir / "eval_tanaka_samples.png",
        f"Tanaka Soliton Eval ($h = 1$, $n = {len(t_eta)}$): "
        f"mean rel-$L^2$ = {np.mean(rel_l2):.4f}",
    )
    plot_error_histogram(
        rel_l2, out_dir / "eval_tanaka_hist.png",
        f"Tanaka Rel-$L^2$ Distribution ($h = 1$, $n = {len(t_eta)}$)",
        summary,
    )

    # ================================================================
    # 3. Stokes waves — generate, run inference, plot samples
    # ================================================================
    from eval_on_stokes import generate_stokes_samples

    n0_values = [4, 6, 8, 10, 14, 20]
    a0_values = [0.01, 0.02, 0.05, 0.08, 0.10]

    for ichoi, depth, label in [(0, 1000.0, "deep"), (1, 1.0, "finite")]:
        print(f"\n--- Stokes {label} (h={depth}) ---")
        stokes = generate_stokes_samples(
            n0_values, a0_values, ichoi=ichoi, depth=depth,
        )
        s_eta = stokes["eta"]
        s_xi = stokes["xi"]
        s_gxi = stokes["gxi"]
        s_x = stokes["x"]
        n_s = len(s_eta)

        pred = run_inference(model, params, ns, s_eta, s_xi, depth)
        rel_l2 = np.linalg.norm(pred - s_gxi, axis=1) / (np.linalg.norm(s_gxi, axis=1) + 1e-12)
        print(f"  {n_s} samples, mean={np.mean(rel_l2):.4f}, median={np.median(rel_l2):.4f}")

        plot_eval_samples(
            s_x, s_eta, s_gxi, pred, rel_l2,
            out_dir / f"eval_stokes_{label}_samples.png",
            f"Stokes Eval ({label} water, $h = {int(depth)}$, $n = {n_s}$): "
            f"mean rel-$L^2$ = {np.mean(rel_l2):.4f}",
        )

    # ================================================================
    # 4. Benjamin-Feir: existing dataset (Philippe) + our generated ICs
    # ================================================================
    from solver.solvers.dno_series_jax import build_grid, dno_series_eval

    # ---- 4a. Existing BF dataset on L=2π, Nx=512 → resample to L=164, Nx=1024
    print("\n--- BF (Philippe dataset) ---")
    L_OLD, L_NEW = 2.0 * np.pi, 164.0
    ALPHA = L_NEW / L_OLD  # ≈ 26.10

    with np.load("data/stokes_bf_dataset.npz") as f:
        bf_eta = np.asarray(f["stokes_eta"], dtype=np.float32)
        bf_xi = np.asarray(f["stokes_xi"], dtype=np.float32)
        bf_gxi = np.asarray(f["stokes_Gxi"], dtype=np.float32)

    # Subsample 4000 cases to keep eval cheap
    if len(bf_eta) > 4000:
        idx = rng.choice(len(bf_eta), size=4000, replace=False)
        bf_eta, bf_xi, bf_gxi = bf_eta[idx], bf_xi[idx], bf_gxi[idx]

    def _upsample_512_to_1024(y: np.ndarray) -> np.ndarray:
        n_old = y.shape[-1]
        fy = np.fft.rfft(y, axis=-1)
        fpad = np.zeros(y.shape[:-1] + (1024 // 2 + 1,), dtype=fy.dtype)
        fpad[..., : fy.shape[-1]] = fy * (1024 / n_old)
        return np.fft.irfft(fpad, n=1024, axis=-1)

    bf_eta_1k = _upsample_512_to_1024(bf_eta).astype(np.float32)
    bf_xi_1k = _upsample_512_to_1024(bf_xi).astype(np.float32)
    bf_gxi_1k = (_upsample_512_to_1024(bf_gxi) / ALPHA).astype(np.float32)
    x_new = np.linspace(0, L_NEW, 1024, endpoint=False, dtype=np.float32)

    pred = run_inference(model, params, ns, bf_eta_1k, bf_xi_1k, 1000.0)
    rel_l2 = np.linalg.norm(pred - bf_gxi_1k, axis=1) / (np.linalg.norm(bf_gxi_1k, axis=1) + 1e-12)
    print(f"  {len(bf_eta_1k)} samples, mean={np.mean(rel_l2):.4f}, median={np.median(rel_l2):.4f}")

    summary_bf = {
        "mean_rel_l2": float(np.mean(rel_l2)),
        "median_rel_l2": float(np.median(rel_l2)),
        "p95_rel_l2": float(np.percentile(rel_l2, 95)),
        "p99_rel_l2": float(np.percentile(rel_l2, 99)),
    }
    (run_dir / "eval_bf_dataset.json").write_text(json.dumps(
        {**summary_bf, "n_samples": int(len(bf_eta_1k))}, indent=2,
    ))

    plot_eval_samples(
        x_new, bf_eta_1k, bf_gxi_1k, pred, rel_l2,
        out_dir / "eval_bf_dataset_samples.png",
        f"Benjamin-Feir Eval (deep water, $n = {len(bf_eta_1k)}$, rescaled to $L = 164$): "
        f"mean rel-$L^2$ = {np.mean(rel_l2):.4f}",
    )
    plot_error_histogram(
        rel_l2, out_dir / "eval_bf_dataset_hist.png",
        f"BF Rel-$L^2$ Distribution ($n = {len(bf_eta_1k)}$)",
        summary_bf,
    )

    # ---- 4b. Our own BF ICs (per JCP09 eq. 33, native L=164)
    bf_path = Path("data/bf_ics.npz")
    if bf_path.exists():
        print("\n--- BF (our generated ICs, JCP09 eq. 33) ---")
        with np.load(bf_path) as f:
            our_eta = np.asarray(f["eta"], dtype=np.float32)
            our_xi = np.asarray(f["xi"], dtype=np.float32)
            our_gxi = np.asarray(f["gxi"], dtype=np.float32)
            our_x = np.asarray(f["x"], dtype=np.float32)

        pred = run_inference(model, params, ns, our_eta, our_xi, 1000.0)
        rel_l2 = np.linalg.norm(pred - our_gxi, axis=1) / (np.linalg.norm(our_gxi, axis=1) + 1e-12)
        print(f"  {len(our_eta)} ICs, mean={np.mean(rel_l2):.4f}, median={np.median(rel_l2):.4f}")

        plot_eval_samples(
            our_x, our_eta, our_gxi, pred, rel_l2,
            out_dir / "eval_bf_ours_samples.png",
            f"Benjamin-Feir IC Eval (JCP09 eq. 33, $n = {len(our_eta)}$): "
            f"mean rel-$L^2$ = {np.mean(rel_l2):.4f}",
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
