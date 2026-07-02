"""Generate dark-theme sample plots for beamer slides.

Produces:
  1. Random sea state samples (eta, xi, Gxi) — grid of cases
  2. Tanaka soliton ICs (eta, xi, Gxi) — grid of 1/2/3 crest cases
  3. Tanaka trajectory snapshots — time evolution panels

Usage:
    uv run python playground/plot_data_samples.py
"""
from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

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
COLORS = [BLUE, RED, GREEN, ORANGE, PURPLE, "#ff8ad6", "#ffff5a", "#5affff"]

matplotlib.rcParams["mathtext.fontset"] = "cm"


def style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(SKY)
    ax.tick_params(colors=AXIS, labelsize=7)
    ax.grid(True, alpha=0.15, color="#6c8aa3")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(SPINE)


def plot_sample_grid(
    x: np.ndarray,
    samples: list[dict[str, np.ndarray]],
    output_path: Path,
    title: str,
    n_cols: int = 4,
) -> None:
    """Plot a grid of (eta, xi, Gxi) panels, one column per sample."""
    n = len(samples)
    n_cols = min(n_cols, n)
    n_rows = 3  # eta, xi, Gxi

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.5 * n_cols, 2.2 * n_rows),
        sharex=True, facecolor=SKY,
    )
    if n_cols == 1:
        axes = axes[:, None]

    row_labels = [r"$\eta(x)$", r"$\xi(x)$", r"$\mathcal{G}(\eta)\xi$"]
    row_keys = ["eta", "xi", "gxi"]

    for col, sample in enumerate(samples[:n_cols]):
        for row, (key, label) in enumerate(zip(row_keys, row_labels)):
            ax = axes[row, col]
            style_ax(ax)
            y = sample[key]
            color = BLUE if key != "gxi" else GREEN
            ax.plot(x, y, color=color, lw=0.9, alpha=0.95)
            ax.fill_between(x, y, 0, color=color, alpha=0.08)
            if col == 0:
                ax.set_ylabel(label, color=AXIS, fontsize=11, rotation=0, labelpad=18)
            if row == 0 and "label" in sample:
                ax.set_title(sample["label"], color=TEXT, fontsize=9, pad=4)
            if row == n_rows - 1:
                ax.set_xlabel("x", color=AXIS, fontsize=8)

    fig.suptitle(title, color=TEXT, fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.5, w_pad=0.8)
    fig.savefig(output_path, dpi=180, facecolor=SKY, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


def plot_trajectory_snapshots(
    x: np.ndarray,
    eta_traj: np.ndarray,
    xi_traj: np.ndarray,
    gxi_traj: np.ndarray,
    times: np.ndarray,
    output_path: Path,
    title: str,
    n_snapshots: int = 6,
) -> None:
    """Plot time-evolution snapshots of a single trajectory."""
    n_steps = eta_traj.shape[0]
    snap_idx = np.linspace(0, n_steps - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 7.5), sharex=True, facecolor=SKY,
        gridspec_kw={"height_ratios": [1.4, 1.0, 1.0]},
    )

    for ax in axes:
        style_ax(ax)

    for i, si in enumerate(snap_idx):
        alpha = 0.3 + 0.7 * (i / (len(snap_idx) - 1))
        color = COLORS[i % len(COLORS)]
        t_val = times[si]
        lbl = f"t={t_val:.1f}"
        axes[0].plot(x, eta_traj[si], color=color, lw=0.9, alpha=alpha, label=lbl)
        axes[1].plot(x, xi_traj[si], color=color, lw=0.9, alpha=alpha)
        axes[2].plot(x, gxi_traj[si], color=color, lw=0.9, alpha=alpha)

    axes[0].set_ylabel(r"$\eta$", color=AXIS, fontsize=12, rotation=0, labelpad=14)
    axes[1].set_ylabel(r"$\xi$", color=AXIS, fontsize=12, rotation=0, labelpad=14)
    axes[2].set_ylabel(r"$\mathcal{G}(\eta)\xi$", color=AXIS, fontsize=12, rotation=0, labelpad=18)
    axes[2].set_xlabel("x", color=AXIS, fontsize=10)

    leg = axes[0].legend(loc="upper right", frameon=False, fontsize=8, ncol=3)
    for txt in leg.get_texts():
        txt.set_color(TEXT)

    fig.suptitle(title, color=TEXT, fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.5)
    fig.savefig(output_path, dpi=180, facecolor=SKY, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


def main() -> None:
    out_dir = Path("notes/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # 1. Random sea state samples
    # ================================================================
    print("Loading random sea data...")
    with np.load("playground/data/random_sea_deep_100k.npz") as f:
        rs_eta = np.asarray(f["eta"], dtype=np.float32)
        rs_xi = np.asarray(f["xi"], dtype=np.float32)
        rs_gxi = np.asarray(f["gxi"], dtype=np.float32)
        x = np.asarray(f["x"], dtype=np.float32)

    rng = np.random.default_rng(7)
    rs_idx = rng.choice(len(rs_eta), size=8, replace=False)
    rs_samples = [
        {"eta": rs_eta[i], "xi": rs_xi[i], "gxi": rs_gxi[i],
         "label": f"$H_s$={4*np.std(rs_eta[i]):.2f}"}
        for i in rs_idx
    ]

    print("Plotting random sea states...")
    plot_sample_grid(x, rs_samples[:4], out_dir / "random_sea_samples_1.png",
                     title="Deep-Water Random Sea States ($h = 1000$)", n_cols=4)
    plot_sample_grid(x, rs_samples[4:], out_dir / "random_sea_samples_2.png",
                     title="Deep-Water Random Sea States ($h = 1000$)", n_cols=4)

    # ================================================================
    # 2. Tanaka soliton ICs (1, 2, 3 crests)
    # ================================================================
    print("Loading Tanaka data...")
    tanaka_path = "data/tanaka_1_clean_sub500k.npz"
    with zipfile.ZipFile(tanaka_path, "r") as zf:
        meta = json.loads(zf.read("meta.json"))

    with np.load(tanaka_path) as f:
        # Load first shard and pick ICs (time=0 for each case)
        t_eta = np.asarray(f["eta_batch_0000"], dtype=np.float32)
        t_xi = np.asarray(f["xi_batch_0000"], dtype=np.float32)
        t_gxi = np.asarray(f["gxi_batch_0000"], dtype=np.float32)
        t_time = np.asarray(f["time_batch_0000"], dtype=np.float32)
        t_case = np.asarray(f["case_id_batch_0000"])
        t_x = np.asarray(f["x"], dtype=np.float32)

    # Find ICs (first snapshot of each case)
    unique_cases = np.unique(t_case)
    ic_samples_1, ic_samples_2, ic_samples_3 = [], [], []

    for cid in unique_cases[:200]:
        mask = t_case == cid
        idx_first = np.where(mask)[0][0]
        eta_ic = t_eta[idx_first]
        xi_ic = t_xi[idx_first]
        gxi_ic = t_gxi[idx_first]

        peaks, props = find_peaks(eta_ic, height=0.02, distance=30, prominence=0.015)
        n_peaks = len(peaks)
        amp = float(np.max(np.abs(eta_ic)))

        sample = {"eta": eta_ic, "xi": xi_ic, "gxi": gxi_ic,
                  "label": f"{n_peaks}-crest, $a$={amp:.2f}"}

        if n_peaks == 1 and len(ic_samples_1) < 3:
            ic_samples_1.append(sample)
        elif n_peaks == 2 and len(ic_samples_2) < 3:
            ic_samples_2.append(sample)
        elif n_peaks >= 3 and len(ic_samples_3) < 3:
            ic_samples_3.append(sample)

    tanaka_ics = ic_samples_1[:2] + ic_samples_2[:2] + ic_samples_3[:2]
    if not tanaka_ics:
        tanaka_ics = ic_samples_1[:3] + ic_samples_2[:3]

    print(f"  Found {len(ic_samples_1)} single, {len(ic_samples_2)} double, {len(ic_samples_3)} triple crest ICs")
    print("Plotting Tanaka ICs...")
    n_ic = min(len(tanaka_ics), 6)
    plot_sample_grid(t_x, tanaka_ics[:n_ic], out_dir / "tanaka_ics.png",
                     title="Tanaka Soliton Initial Conditions ($h = 1$)",
                     n_cols=n_ic)

    # ================================================================
    # 3. Trajectory snapshots
    # ================================================================
    print("Plotting trajectory snapshots...")

    # Helper: find a trajectory case and plot it
    def _find_and_plot(
        target_peaks: int, min_snapshots: int, search_limit: int,
        out_name: str, title_fmt: str,
    ) -> None:
        ge = target_peaks >= 3
        for cid in unique_cases[:search_limit]:
            mask = t_case == cid
            idx_all = np.where(mask)[0]
            if len(idx_all) < min_snapshots:
                continue
            eta_ic = t_eta[idx_all[0]]
            peaks, _ = find_peaks(eta_ic, height=0.03, distance=30, prominence=0.02)
            match = (len(peaks) >= target_peaks) if ge else (len(peaks) == target_peaks)
            if match:
                traj_eta = t_eta[idx_all]
                traj_xi = t_xi[idx_all]
                traj_gxi = t_gxi[idx_all]
                traj_times = t_time[idx_all]
                amp = float(np.max(np.abs(eta_ic)))
                n_snap = len(idx_all)
                print(f"  Using case {cid}: {len(peaks)} crest(s), amp={amp:.3f}, {n_snap} snapshots")
                plot_trajectory_snapshots(
                    t_x, traj_eta, traj_xi, traj_gxi, traj_times,
                    out_dir / out_name,
                    title=title_fmt.format(amp=amp),
                    n_snapshots=min(n_snap, 8),
                )
                return
        print(f"  WARNING: no {target_peaks}-crest case with >= {min_snapshots} snapshots found")

    _find_and_plot(2, 4, 2000, "tanaka_trajectory_2crest.png",
                   "2-Crest Soliton Trajectory ($a = {amp:.2f}$, $h = 1$)")
    _find_and_plot(3, 4, 5000, "tanaka_trajectory_3crest.png",
                   "3-Crest Soliton Trajectory ($a = {amp:.2f}$, $h = 1$)")
    _find_and_plot(1, 4, 2000, "tanaka_trajectory_1crest.png",
                   "Single Soliton Trajectory ($a = {amp:.2f}$, $h = 1$)")

    print("\nDone!")


if __name__ == "__main__":
    main()
