import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_stokes_dataset(mat_path):
    """
    Returns tensors with shape (Nsamples, Nsaves, Nx)
    plus metadata (grid/time/params).
    """
    with h5py.File(mat_path, "r") as f:
        def read(name):
            return np.array(f[name]).T  # MATLAB -> NumPy order

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
            "save_idx": save_idx_mat - 1,   # convert to 0-based
            "params": {
                key: read(f"params/{key}").ravel()
                for key in ("a0", "n0", "k0", "ichoi")
            }
        }

    return eta_traj, xi_traj, Gxi_traj, meta

# --- Load deep-water dataset ---
data_dir = Path(__file__).resolve().parent.parent / "data"
data_path = data_dir / "stokes_dno_training_ichoi0.mat"
if not data_path.exists():
    raise FileNotFoundError(f"Could not find dataset at {data_path}")

eta, xi, Gxi, meta = load_stokes_dataset(str(data_path))
x = meta["x"]
t_saved = meta["t"][meta["save_idx"]]  # actual saved times (0-based)
a0_vals = meta["params"]["a0"]
n0_vals = meta["params"]["n0"]
k0_vals = meta["params"]["k0"]

output_dir = Path(__file__).resolve().parent.parent / "plots-outputs"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"eta shape: {eta.shape}")
print(f"xi shape: {xi.shape}")
print(f"Gxi shape: {Gxi.shape}")
print(f"x shape: {x.shape}")
print(f"t shape: {meta['t'].shape}")
print(f"save_idx shape: {meta['save_idx'].shape}")
for key, arr in meta["params"].items():
    print(f"params/{key} shape: {arr.shape}")

# --- Plot trajectory i at multiple times ---
traj_idx = 0
time_indices = [0, 10, 20, 30, 40]
a0_traj = a0_vals[traj_idx]
n0_traj = n0_vals[traj_idx]
k0_traj = k0_vals[traj_idx]

plt.figure(figsize=(10, 5))
for j in time_indices:
    plt.plot(x, eta[traj_idx, j, :], label=f"t = {t_saved[j]:.2f}")
plt.title(
    f"eta(x,t) trajectory {traj_idx} (a0={a0_traj:.3f}, n0={n0_traj:.3f}, k0={k0_traj:.3f})"
)
plt.xlabel("x")
plt.ylabel("eta")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "eta_trajectory.png")

# --- Plot xi trajectory at multiple times ---
plt.figure(figsize=(10, 5))
for j in time_indices:
    plt.plot(x, xi[traj_idx, j, :], label=f"t = {t_saved[j]:.2f}")
plt.title(
    f"xi(x,t) trajectory {traj_idx} (a0={a0_traj:.3f}, n0={n0_traj:.3f}, k0={k0_traj:.3f})"
)
plt.xlabel("x")
plt.ylabel("xi")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "xi_trajectory.png")

# --- Plot G(eta)xi vs xi at a single time snapshot ---
time_idx = 10
plt.figure(figsize=(10, 5))
plt.plot(x, xi[traj_idx, time_idx, :], label="xi")
plt.plot(x, Gxi[traj_idx, time_idx, :], label="G(eta)xi")
plt.title(
    f"Trajectory {traj_idx}, time {t_saved[time_idx]:.2f} "
    f"(a0={a0_traj:.3f}, n0={n0_traj:.3f}, k0={k0_traj:.3f})"
)
plt.xlabel("x")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "Getaxi_trajectory.png")

# --- Stacked view of G(eta)xi evolution over selected times ---
fig, axes = plt.subplots(len(time_indices), 1, figsize=(10, 2.5 * len(time_indices)), sharex=True)
for ax, j in zip(axes, time_indices):
    ax.plot(x, Gxi[traj_idx, j, :], color="tab:orange")
    ax.set_ylabel("G(eta)xi")
    ax.set_title(f"t = {t_saved[j]:.2f}")
axes[-1].set_xlabel("x")
fig.suptitle(
    f"G(eta)xi evolution for trajectory {traj_idx} "
    f"(a0={a0_traj:.3f}, n0={n0_traj:.3f}, k0={k0_traj:.3f})",
    y=0.995,
)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(output_dir / "Getaxi_stacked.png")

# --- Zoomed-in view of the same snapshot to highlight G(eta)xi ---
zoom_fraction = 0.15  # fraction of domain width to display
domain_span = x.max() - x.min()
half_window = 0.5 * domain_span * zoom_fraction
x_center = 0.5 * (x.min() + x.max())
zoom_mask = (x >= x_center - half_window) & (x <= x_center + half_window)

plt.figure(figsize=(10, 5))
plt.plot(x[zoom_mask], xi[traj_idx, time_idx, zoom_mask], label="xi")
plt.plot(x[zoom_mask], Gxi[traj_idx, time_idx, zoom_mask], label="G(eta)xi")
plt.title(
    f"Trajectory {traj_idx}, time {t_saved[time_idx]:.2f} (zoomed, center={x_center:.2f}, "
    f"a0={a0_traj:.3f}, n0={n0_traj:.3f}, k0={k0_traj:.3f})"
)
plt.xlabel("x")
plt.xlim(x_center - half_window, x_center + half_window)
gxi_window = Gxi[traj_idx, time_idx, zoom_mask]
y_margin = 0.1 * (gxi_window.max() - gxi_window.min() + 1e-8)
plt.ylim(gxi_window.min() - y_margin, gxi_window.max() + y_margin)
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "Getaxi_trajectory_zoom.png")

# --- Histogram of ||G(eta,xi)|| using absolute value over (time, space) per trajectory ---
gxi_abs_norm = np.linalg.norm(np.abs(Gxi).reshape(Gxi.shape[0], -1), axis=1)
plt.figure(figsize=(10, 5))
plt.hist(gxi_abs_norm, bins=40, color="tab:orange", edgecolor="black", alpha=0.8)
plt.title("Histogram of ||G(eta,xi)|| (abs, per trajectory)")
plt.xlabel("L2 norm of |G(eta,xi)|")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(output_dir / "Getaxi_abs_norm_hist.png")

# --- Plot a few lowest-norm trajectories for G(eta)xi ---
num_low = 4
low_idx = np.argsort(gxi_abs_norm)[:num_low]
fig, axes = plt.subplots(num_low, 1, figsize=(10, 2.5 * num_low), sharex=True)
for ax, idx in zip(axes, low_idx):
    a0_val = a0_vals[idx]
    n0_val = n0_vals[idx]
    k0_val = k0_vals[idx]
    ax.plot(x, Gxi[idx, time_idx, :], color="tab:orange")
    ax.set_ylabel("G(eta)xi")
    ax.set_title(
        f"Low-norm traj {idx} (||G||={gxi_abs_norm[idx]:.2f}, "
        f"a0={a0_val:.3f}, n0={n0_val:.3f}, k0={k0_val:.3f})"
    )
axes[-1].set_xlabel("x")
fig.suptitle(f"Lowest-norm G(eta)xi trajectories at t={t_saved[time_idx]:.2f}", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(output_dir / "Getaxi_low_norm_trajectories.png")

# --- Representative eta/xi/G(eta)xi for each k0 (sorted increasing) ---
k0_unique = np.unique(k0_vals)
max_rows = 6
if len(k0_unique) > max_rows:
    idxs = np.linspace(0, len(k0_unique) - 1, max_rows, dtype=int)
    k0_selected = k0_unique[idxs]
else:
    k0_selected = k0_unique
rows = len(k0_selected)
fig, axes = plt.subplots(rows, 3, figsize=(15, 3 * rows), sharex=True)
if rows == 1:
    axes = np.expand_dims(axes, axis=0)

for row, k0_value in enumerate(k0_selected):
    traj_indices = np.where(k0_vals == k0_value)[0]
    idx = traj_indices[0]
    a0_val = a0_vals[idx]
    n0_val = n0_vals[idx]

    axes[row, 0].plot(x, eta[idx, time_idx, :], color="tab:blue")
    axes[row, 0].set_title(f"η(x) k0={k0_value:.3f}", fontsize=10)
    axes[row, 0].set_ylabel(f"Traj {idx}\na0={a0_val:.3f}\nn0={n0_val:.3f}")

    axes[row, 1].plot(x, xi[idx, time_idx, :], color="tab:green")
    axes[row, 1].set_title("ξ(x)", fontsize=10)

    axes[row, 2].plot(x, Gxi[idx, time_idx, :], color="tab:red")
    axes[row, 2].set_title("G(η,ξ)", fontsize=10)

axes[-1, 0].set_xlabel("x")
axes[-1, 1].set_xlabel("x")
axes[-1, 2].set_xlabel("x")
fig.suptitle(
    f"Representative trajectories per k0 at t={t_saved[time_idx]:.2f}",
    y=0.995,
)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(output_dir / "Getaxi_k0_sweep.png")

# --- Detailed view: for each a0, show multiple n0 samples with stacked η/ξ/G(η,ξ) ---
samples_per_a0 = 3
a0_sorted = np.unique(a0_vals)
for a0_value in a0_sorted:
    a0_mask = np.isclose(a0_vals, a0_value)
    traj_indices = np.where(a0_mask)[0]
    if len(traj_indices) == 0:
        continue
    n0_subset = n0_vals[traj_indices]
    unique_n0 = np.unique(n0_subset)
    if len(unique_n0) == 0:
        continue
    if len(unique_n0) > samples_per_a0:
        idxs = np.linspace(0, len(unique_n0) - 1, samples_per_a0, dtype=int)
        selected_n0 = unique_n0[idxs]
    else:
        selected_n0 = unique_n0

    rows = len(selected_n0)
    fig = plt.figure(figsize=(7, 2.2 * rows * 3))
    outer = fig.add_gridspec(rows, 1, hspace=0.25)

    for row, n0_value in enumerate(selected_n0):
        sample_candidates = traj_indices[np.isclose(n0_subset, n0_value)]
        sample_idx = sample_candidates[0]
        k0_value = k0_vals[sample_idx]

        inner = outer[row].subgridspec(3, 1, hspace=0.05)
        ax_eta = fig.add_subplot(inner[0])
        ax_xi = fig.add_subplot(inner[1], sharex=ax_eta)
        ax_g = fig.add_subplot(inner[2], sharex=ax_eta)

        ax_eta.plot(x, eta[sample_idx, time_idx, :], color="tab:blue")
        ax_eta.set_ylabel("η")
        ax_eta.set_title(
            f"a0={a0_value:.3f}, n0={n0_value:.3f}, k0={k0_value:.3f}", fontsize=10
        )

        ax_xi.plot(x, xi[sample_idx, time_idx, :], color="tab:green")
        ax_xi.set_ylabel("ξ")

        ax_g.plot(x, Gxi[sample_idx, time_idx, :], color="tab:red")
        ax_g.set_ylabel("G(η,ξ)")
        ax_g.set_xlabel("x")

    fig.suptitle(
        f"Stacked samples for a0={a0_value:.3f} at t={t_saved[time_idx]:.2f}", y=0.995
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    safe_a0 = f"{a0_value:.3f}".replace(".", "p")
    fig.savefig(output_dir / f"Getaxi_a0_{safe_a0}_stacked.png")
    plt.close(fig)

# --- Representative eta/xi/G(eta)xi for each a0 (sorted increasing) ---
a0_unique = np.unique(a0_vals)
if len(a0_unique) > max_rows:
    idxs = np.linspace(0, len(a0_unique) - 1, max_rows, dtype=int)
    a0_selected = a0_unique[idxs]
else:
    a0_selected = a0_unique
rows = len(a0_selected)
fig, axes = plt.subplots(rows, 3, figsize=(15, 3 * rows), sharex=True)
if rows == 1:
    axes = np.expand_dims(axes, axis=0)

for row, a0_value in enumerate(a0_selected):
    traj_indices = np.where(a0_vals == a0_value)[0]
    idx = traj_indices[0]
    n0_val = n0_vals[idx]
    k0_val = k0_vals[idx]

    axes[row, 0].plot(x, eta[idx, time_idx, :], color="tab:blue")
    axes[row, 0].set_title(f"η(x) a0={a0_value:.3f}", fontsize=10)
    axes[row, 0].set_ylabel(f"Traj {idx}\nn0={n0_val:.3f}\nk0={k0_val:.3f}")

    axes[row, 1].plot(x, xi[idx, time_idx, :], color="tab:green")
    axes[row, 1].set_title("ξ(x)", fontsize=10)

    axes[row, 2].plot(x, Gxi[idx, time_idx, :], color="tab:red")
    axes[row, 2].set_title("G(η,ξ)", fontsize=10)

axes[-1, 0].set_xlabel("x")
axes[-1, 1].set_xlabel("x")
axes[-1, 2].set_xlabel("x")
fig.suptitle(
    f"Representative trajectories per a0 at t={t_saved[time_idx]:.2f}",
    y=0.995,
)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(output_dir / "Getaxi_a0_sweep.png")