"""CPU-only learned preconditioner for Laplace solves on nine Tanaka surfaces."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Callable, cast

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from pyamg.krylov import fgmres  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.linalg import LinearOperator, spilu, splu  # noqa: E402
import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from laplace import assemble_stiffness  # noqa: E402

Array = NDArray[np.float64]
Inverse = Callable[[Array], Array]
ROOT = Path(__file__).resolve().parents[1]
ROWS = [16584, 19784, 16984, 18784, 18184, 24984, 34984, 38384, 59984]


def geometry_features(eta: Array, depth: float, nz: int) -> Tensor:
    nx = len(eta)
    slope = np.fft.irfft(1j * np.arange(nx // 2 + 1) * np.fft.rfft(eta), n=nx)
    s = np.arange(nz)[:, None] / nz
    fields = np.stack(np.broadcast_arrays((depth + eta)[None, :] / depth,
                                          s * slope[None, :], s + 0 * eta,
                                          np.full((nz, nx), depth)))
    return torch.tensor(fields, dtype=torch.float32)[None]


def correction(layers: nn.ModuleList, residual: Tensor, geometry: Tensor, diagonal: Tensor) -> Tensor:
    """Jacobi plus a residual-scaled U-Net correction; periodic only in x."""
    scale = residual.square().mean((1, 2, 3), keepdim=True).sqrt()
    scaled = residual / scale.clamp_min(1e-20)
    values = torch.cat((scaled / diagonal, geometry.expand(len(residual), -1, -1, -1)), dim=1)
    skips = []
    for layer in layers[:4]:
        values = F.pad(F.pad(values, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1), mode="replicate")
        values = F.gelu(layer(values))
        skips.append(values)
        if len(skips) < 4:
            values = F.avg_pool2d(values, 2)
    for layer, skip in zip(layers[4:7], reversed(skips[:-1]), strict=True):
        values = F.interpolate(values, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        values = torch.cat((values, skip), dim=1)
        values = F.pad(F.pad(values, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1), mode="replicate")
        values = F.gelu(layer(values))
    return scale * (scaled / diagonal + layers[-1](values))


@torch.inference_mode()
def neural_inverse(residual: Array, layers: nn.ModuleList, geometry: Tensor, diagonal: Tensor) -> Array:
    rhs = torch.tensor(residual.reshape(diagonal.shape), dtype=torch.float32)
    return correction(layers, rhs, geometry, diagonal).numpy().ravel().astype(np.float64)


def solve(matrix: csr_matrix, rhs: Array, inverse: Inverse | None) -> tuple[Array, dict]:
    history = []
    # SciPy's factory accepts matvec via __new__, which its base-class stub omits.
    preconditioner = None if inverse is None else LinearOperator(shape=matrix.shape, matvec=inverse, dtype=np.float64)  # pyright: ignore[reportCallIssue]
    started = perf_counter()
    value, info = fgmres(matrix, rhs, M=preconditioner, tol=1e-9, restart=40,
                         maxiter=20, residuals=history)
    seconds = perf_counter() - started
    residual = float(np.linalg.norm(rhs - matrix @ value) / np.linalg.norm(rhs))
    return value, {"seconds": seconds, "iterations": len(history) - 1,
                   "relative_residual": residual, "converged": bool(info == 0 and residual <= 1e-9)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-surface", type=int, default=1024)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    nx, nz, batch_size = 64, 16, 16
    data_dir = ROOT / "outputs/paper_dataset/arrays"
    metadata = {key: np.load(data_dir / f"{key}.npy", mmap_mode="r")[ROWS]
                for key in ("depth", "simulation_id", "parameter_group_id", "dataset_split", "family_id", "frame_index")}
    values = np.stack([np.load(data_dir / f"{field}.npy", mmap_mode="r")[ROWS]
                       for field in ("eta", "xi")]).astype(np.float64)
    spectrum = np.fft.rfft(values, axis=-1)[..., :nx // 2 + 1].copy()
    spectrum[..., -1] = 0
    eta, xi = np.fft.irfft(spectrum, n=nx, axis=-1) * nx / values.shape[-1]
    if not np.isfinite(values).all() or np.any(metadata["depth"][:, None] + eta <= 0):
        raise ValueError("Invalid external Tanaka initial conditions")
    if not (np.all(metadata["family_id"] == 2) and np.all(metadata["frame_index"] == 0)
            and len(np.unique(metadata["simulation_id"])) == 9
            and np.all(metadata["dataset_split"][:6] == "train") and np.all(metadata["dataset_split"][6:] == "test")):
        raise ValueError("Expected six distinct training ICs and three held-out test ICs")
    stiffness = [assemble_stiffness(surface, float(depth), nz) for surface, depth in zip(eta, metadata["depth"], strict=True)]
    matrices = [matrix[:-nx, :-nx].tocsr() for matrix in stiffness]
    geometry = [geometry_features(surface, float(depth), nz) for surface, depth in zip(eta, metadata["depth"], strict=True)]
    diagonals = [torch.tensor(matrix.diagonal().reshape(1, 1, nz, nx), dtype=torch.float32) for matrix in matrices]
    head = nn.Conv2d(8, 1, 1)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(cast(Tensor, head.bias))
    layers = nn.ModuleList([nn.Conv2d(5, 8, 3), *(nn.Conv2d(8, 8, 3) for _ in range(3)),
                            *(nn.Conv2d(16, 8, 3) for _ in range(3)), head])
    optimizer = torch.optim.Adam(layers.parameters(), lr=2e-5)

    # Manufactured interior corrections give exact (A e, e) pairs without solving A.
    # Cosine vertical modes vanish at the top and have zero bottom normal derivative.
    vertical = np.cos(np.pi * (np.arange(8)[:, None] + .5) * np.arange(nz)[None, :] / nz)
    horizontal_k = np.arange(nx // 2 + 1)
    started = perf_counter()
    training = []
    batches = [(surface, first) for surface in range(6)
               for first in range(0, args.samples_per_surface, batch_size)]
    rng.shuffle(batches)
    layers.train()
    for step, (surface, first) in enumerate(batches):
        count = min(batch_size, args.samples_per_surface - first)
        coefficients = rng.normal(size=(count, 8, nx // 2 + 1)) + 1j * rng.normal(size=(count, 8, nx // 2 + 1))
        # Half smooth, half broad-band in x; both use the first eight vertical modes.
        decay = (1 + horizontal_k[None, :] ** 2 + np.arange(8)[:, None] ** 2)
        coefficients[:count // 2] /= decay
        coefficients[..., -1] = 0
        errors = np.einsum("vm,bvx->bmx", vertical, np.fft.irfft(coefficients, n=nx))
        errors /= np.sqrt(np.mean(errors ** 2, axis=(1, 2), keepdims=True))
        rhs = (matrices[surface] @ errors.reshape(count, -1).T).T
        target = torch.tensor(errors[:, None], dtype=torch.float32)
        rhs_tensor = torch.tensor(rhs.reshape(count, 1, nz, nx), dtype=torch.float32)
        prediction = correction(layers, rhs_tensor, geometry[surface], diagonals[surface])
        loss = (prediction - target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training.append(float(loss.detach()))
        if step % 64 == 0 or step == len(batches) - 1:
            print(f"epoch 1, batch {step + 1}/{len(batches)}: correction MSE {training[-1]:.6g}", flush=True)
    training_seconds = perf_counter() - started
    layers.eval()
    torch.save(layers.state_dict(), args.output / "weights.pt")
    report = {"nx": nx, "nz": nz, "epochs": 1, "learning_rate": 2e-5,
              "training_samples": 6 * args.samples_per_surface, "seed": 0,
              "parameters": sum(value.numel() for value in layers.parameters()),
              "training_seconds": training_seconds, "training_loss": training,
              "torch_version": torch.__version__, "cases": []}

    for index in range(6, 9):
        matrix, full, phi = matrices[index], stiffness[index], xi[index]
        rhs = -full[:-nx, -nx:] @ phi
        started = perf_counter()
        factor = splu(matrix.tocsc())
        direct_setup = perf_counter() - started
        reference = factor.solve(rhs)
        reference_flux = (full[-nx:, :-nx] @ reference + full[-nx:, -nx:] @ phi) / (2 * np.pi / nx)
        started = perf_counter()
        ilu = spilu(matrix.tocsc(), drop_tol=1e-3, fill_factor=10)
        ilu_setup = perf_counter() - started
        jacobi_diagonal = matrix.diagonal()
        methods = {"none": None, "jacobi": lambda r: r / jacobi_diagonal,
                   "neural": lambda r: neural_inverse(r, layers, geometry[index], diagonals[index]),
                   "ilu": ilu.solve}
        case = {"simulation_id": int(metadata["simulation_id"][index]),
                "depth": float(metadata["depth"][index]), "methods": {},
                "direct_setup_seconds": direct_setup, "ilu_setup_seconds": ilu_setup}
        for name, inverse in methods.items():
            solve(matrix, rhs, inverse)  # Warm the same operation outside timings.
            measured = [solve(matrix, rhs, inverse) for _ in range(3)]
            value, metrics = measured[-1]
            metrics["repeat_seconds"] = [record["seconds"] for _, record in measured]
            metrics["seconds"] = float(np.median(metrics["repeat_seconds"]))
            flux = (full[-nx:, :-nx] @ value + full[-nx:, -nx:] @ phi) / (2 * np.pi / nx)
            metrics["potential_error"] = float(np.linalg.norm(value - reference) / np.linalg.norm(reference))
            metrics["dno_error"] = float(np.linalg.norm(flux - reference_flux) / np.linalg.norm(reference_flux))
            case["methods"][name] = metrics
            print(f"test {case['simulation_id']} {name}: {metrics}", flush=True)
        direct_times = []
        for _ in range(3):
            started = perf_counter()
            factor.solve(rhs)
            direct_times.append(perf_counter() - started)
        case["direct_solve_seconds"] = float(np.median(direct_times))
        report["cases"].append(case)
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    # A short coupled water-wave smoke test, not a production GL2 rollout benchmark.
    # Both paths use this same coarse FEM DNO and RK4; only the interior solver differs.
    trajectories = []
    for method in ("direct", "neural"):
        state = np.stack((eta[6], xi[6]))
        initial = state.copy()
        started = perf_counter()
        for _ in range(4):
            stages = []
            for offset in (0., .5, .5, 1.):
                stage = state if not stages else state + .001 * offset * stages[-1]
                surface, phi = stage
                full = assemble_stiffness(surface, float(metadata["depth"][6]), nz)
                matrix = full[:-nx, :-nx].tocsr()
                rhs = -full[:-nx, -nx:] @ phi
                if method == "direct":
                    potential = splu(matrix.tocsc()).solve(rhs)
                else:
                    features = geometry_features(surface, float(metadata["depth"][6]), nz)
                    diagonal = torch.tensor(matrix.diagonal().reshape(1, 1, nz, nx), dtype=torch.float32)
                    potential, metrics = solve(matrix, rhs, lambda r: neural_inverse(r, layers, features, diagonal))
                    if not metrics["converged"]:
                        raise RuntimeError(f"Rollout interior solve failed: {metrics}")
                flux = (full[-nx:, :-nx] @ potential + full[-nx:, -nx:] @ phi) / (2 * np.pi / nx)
                dx = np.fft.irfft(1j * np.arange(nx // 2 + 1) * np.fft.rfft(stage), n=nx)
                stages.append(np.stack((flux, -surface - .5 * dx[1] ** 2
                                        + .5 * (flux + dx[0] * dx[1]) ** 2 / (1 + dx[0] ** 2))))
            state += .001 * (stages[0] + 2 * stages[1] + 2 * stages[2] + stages[3]) / 6
            state[1] -= state[1].mean()
        trajectories.append(state)
        report[f"smoke_{method}_seconds"] = perf_counter() - started
    report["smoke_horizon"] = .004
    report["smoke_relative_difference"] = (np.linalg.norm(trajectories[1] - trajectories[0], axis=1)
                                             / np.linalg.norm(trajectories[0], axis=1)).tolist()
    np.savez_compressed(args.output / "smoke.npz", initial=initial, direct=trajectories[0], neural=trajectories[1])
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Smoke rollout difference (eta, xi): {report['smoke_relative_difference']}", flush=True)
