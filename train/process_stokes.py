import argparse
from pathlib import Path
import numpy as np
import h5py


def load_stokes_dataset(mat_path):
    with h5py.File(mat_path, "r") as f:
        def read(name):
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


def process_file(mat_path: Path, output_dir: Path):
    print(f"Processing {mat_path.name}...")
    eta, xi, Gxi, meta = load_stokes_dataset(mat_path)

    # Flatten the time dimension into samples: (Nsamples * Nsaves, Nx)
    n_traj, n_time, nx = eta.shape
    eta_flat = eta.reshape(n_traj * n_time, nx)
    xi_flat = xi.reshape(n_traj * n_time, nx)
    Gxi_flat = Gxi.reshape(n_traj * n_time, nx)

    out_name = mat_path.stem + ".npz"
    out_path = output_dir / out_name

    np.savez_compressed(
        out_path,
        eta=eta_flat,
        xi=xi_flat,
        Gxi=Gxi_flat,
        x=meta["x"],
        t=meta["t"][meta["save_idx"]],
    )
    print(f"Saved to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat-file", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    mat_path = Path(args.mat_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    process_file(mat_path, out_dir)
