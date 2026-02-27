from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_stokes_dataset(mat_path: Path):
    with h5py.File(mat_path, "r") as f:
        def read(name: str):
            return np.array(f[name]).T

        eta = read("eta_data")
        xi = read("xi_data")
        Gxi = read("Gxi_data")

        eta_traj = np.transpose(eta, (2, 1, 0))
        xi_traj = np.transpose(xi, (2, 1, 0))
        Gxi_traj = np.transpose(Gxi, (2, 1, 0))

        save_idx_mat = read("save_idx").astype(int).ravel()
        meta = {
            "x": read("x").ravel(),
            "t": read("t").ravel(),
            "save_idx": save_idx_mat - 1,
            "params": {
                key: read(f"params/{key}").ravel()
                for key in ("a0", "n0", "k0", "ichoi")
            },
        }
    return eta_traj, xi_traj, Gxi_traj, meta


def oscillation_score_eta(eta_traj: np.ndarray) -> np.ndarray:
    # Mean spatial total variation over saved times; higher means more oscillatory.
    tv_time = np.sum(np.abs(np.diff(eta_traj, axis=-1)), axis=-1)  # (n_traj, n_time)
    return tv_time.mean(axis=1)  # (n_traj,)


def build_plot_for_file(mat_path: Path, out_dir: Path, num_samples: int, time_idx: int) -> Path:
    eta, xi, gxi, meta = load_stokes_dataset(mat_path)
    x = meta["x"]
    t_saved = meta["t"][meta["save_idx"]]
    a0_vals = meta["params"]["a0"]
    n0_vals = meta["params"]["n0"]
    k0_vals = meta["params"]["k0"]
    ichoi_vals = meta["params"]["ichoi"]

    a0_max = np.max(a0_vals)
    mask = np.isclose(a0_vals, a0_max)
    candidate_idx = np.where(mask)[0]
    if len(candidate_idx) == 0:
        raise RuntimeError(f"No trajectories found with a0=max in {mat_path.name}")

    scores = oscillation_score_eta(eta)
    sorted_candidates = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
    selected = sorted_candidates[: min(num_samples, len(sorted_candidates))]
    if len(selected) == 0:
        raise RuntimeError(f"No selected trajectories for {mat_path.name}")

    rows = len(selected)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 3.2 * rows), sharex=True)
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    actual_time_idx = min(time_idx, eta.shape[1] - 1)

    for row, idx in enumerate(selected):
        a0_val = a0_vals[idx]
        n0_val = n0_vals[idx]
        k0_val = k0_vals[idx]
        score = scores[idx]

        axes[row, 0].plot(x, eta[idx, actual_time_idx, :], color="tab:blue")
        axes[row, 0].set_title("η(x)", fontsize=10)
        axes[row, 0].set_ylabel(
            f"traj {idx}\nscore={score:.2f}\na0={a0_val:.3f}\nn0={n0_val:.3f}\nk0={k0_val:.3f}"
        )

        axes[row, 1].plot(x, xi[idx, actual_time_idx, :], color="tab:green")
        axes[row, 1].set_title("ξ(x)", fontsize=10)

        axes[row, 2].plot(x, gxi[idx, actual_time_idx, :], color="tab:red")
        axes[row, 2].set_title("G(η,ξ)", fontsize=10)

    for col in range(3):
        axes[-1, col].set_xlabel("x")

    ichoi_label = int(np.rint(np.median(ichoi_vals[selected])))
    fig.suptitle(
        f"{mat_path.name} | a0=max={a0_max:.3f} | ichoi={ichoi_label} | "
        f"Top {len(selected)} oscillatory (score = mean TV of η over saved times) | t={t_saved[actual_time_idx]:.3f}",
        y=0.995,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{mat_path.stem}_a0max_most_oscillatory.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot most oscillatory extreme solutions after filtering a0=max")
    parser.add_argument("--extreme-dir", default="grid_data/extreme")
    parser.add_argument("--out-dir", default="plots-outputs")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--time-idx", type=int, default=10)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    extreme_dir = (repo_root / args.extreme_dir).resolve()
    out_dir = (repo_root / args.out_dir).resolve()

    mats = sorted(extreme_dir.glob("stokes_dno_training_Nx*_M*_ichoi*.mat"))
    if not mats:
        raise FileNotFoundError(f"No .mat files found in {extreme_dir}")

    for mat_path in mats:
        out_path = build_plot_for_file(mat_path, out_dir, args.num_samples, args.time_idx)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
