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


def flat_inverse_blocks(height: float, nx: int, nz: int) -> Array:
    """Exact flat-strip FEM inverse: one small vertical matrix per Fourier mode."""
    dx, dz = 2 * np.pi / nx, height / nz
    vertical = np.diag(np.r_[1., np.full(nz - 1, 2.)])
    vertical -= np.diag(np.ones(nz - 1), 1) + np.diag(np.ones(nz - 1), -1)
    horizontal = 4 * np.sin(np.pi * np.arange(nx // 2 + 1) / nx) ** 2
    matrices = dx / dz * vertical + dz / dx * horizontal[:, None, None] * np.diag(np.r_[.5, np.ones(nz - 1)])
    return cast(Array, np.linalg.inv(matrices))


def flat_inverse(residual: Array, inverse_blocks: Array) -> Array:
    nz, nx = inverse_blocks.shape[-1], 2 * (len(inverse_blocks) - 1)
    spectrum = np.fft.rfft(residual.reshape(*residual.shape[:-1], nz, nx), axis=-1)
    filtered = (inverse_blocks @ spectrum.swapaxes(-1, -2)[..., None])[..., 0].swapaxes(-1, -2)
    return np.fft.irfft(filtered, n=nx, axis=-1).reshape(residual.shape)


def diagonal_scaling(layers: nn.Sequential, geometry: Tensor) -> Tensor:
    """Positive geometry-only scaling; exactly one for a flat surface."""
    height = geometry[:, :1]
    deviation = height / height.mean(dim=-1, keepdim=True) - 1
    coefficients = layers(geometry)
    return torch.exp(deviation * coefficients[:, :1] + geometry[:, 1:2].square() * coefficients[:, 1:2])


def cached_inverse(residual: Array, scaling: Array, inverse_blocks: Array) -> Array:
    return scaling * flat_inverse(scaling * residual, inverse_blocks)


def tensor_inverse(rhs: Tensor, scaling: Tensor, inverse_blocks: Tensor) -> Tensor:
    """Differentiable form of the same diagonal-flat-diagonal inverse."""
    spectrum = torch.fft.rfft(scaling * rhs, dim=-1)
    filtered = (inverse_blocks @ spectrum.transpose(-1, -2).unsqueeze(-1)).squeeze(-1).transpose(-1, -2)
    return scaling * torch.fft.irfft(filtered, n=rhs.shape[-1], dim=-1)


def training_pairs(full: csr_matrix, nx: int, nz: int, count: int,
                   rng: np.random.Generator, inverse_blocks: Array) -> tuple[Array, Array]:
    """Half mixed forcing, half directions requested by flat-preconditioned GMRES."""
    matrix = full[:-nx, :-nx].tocsr()
    ordinary_count = count // 2
    rhs = rng.normal(size=(ordinary_count, nz * nx))
    quarter = ordinary_count // 4
    vertical = np.cos(np.pi * (np.arange(nz)[:, None] + .5) * np.arange(nz)[None, :] / nz)
    k = np.arange(nx // 2 + 1)
    coefficients = np.fft.rfft(rng.normal(size=(quarter, nz, nx)), axis=-1)
    coefficients /= 1 + k[None, :] ** 2 + np.arange(nz)[:, None] ** 2
    errors = np.einsum("vm,bvx->bmx", vertical, np.fft.irfft(coefficients, n=nx))
    rhs[quarter:2 * quarter] = (matrix @ errors.reshape(quarter, -1).T).T
    boundary = np.fft.rfft(rng.normal(size=(ordinary_count - 2 * quarter, nx)), axis=-1)
    boundary[:quarter] /= 1 + k ** 2
    boundary[:, 0] = 0
    rhs[2 * quarter:] = -(full[:-nx, -nx:] @ np.fft.irfft(boundary, n=nx).T).T
    directions = []

    def record_direction(residual: Array) -> Array:
        directions.append(residual.copy())
        return flat_inverse(residual, inverse_blocks)

    for source in rhs[rng.permutation(ordinary_count)]:
        _, metrics = solve(matrix, source, record_direction)
        if not metrics["converged"]:
            raise RuntimeError(f"Training-direction solve failed: {metrics}")
        if len(directions) >= count - ordinary_count:
            break
    rhs = np.concatenate((rhs, np.asarray(directions[:count - ordinary_count])))
    target = splu(matrix.tocsc()).solve(rhs.T).T
    scale = np.sqrt(np.mean(target ** 2, axis=1, keepdims=True))
    return rhs / scale, target / scale


def inverse_loss(layers: nn.Sequential, rhs: Tensor, target: Tensor, geometry: Tensor,
                 matrix: Tensor, inverse_blocks: Tensor) -> Tensor:
    prediction = tensor_inverse(rhs, diagonal_scaling(layers, geometry), inverse_blocks)
    residual = torch.sparse.mm(matrix, prediction.flatten(1).T).T.reshape_as(rhs) - rhs
    relative_residual = residual.square().mean((1, 2, 3)) / rhs.square().mean((1, 2, 3))
    # Train A P(r) close to r, rather than prioritizing interior potential accuracy.
    return relative_residual.mean() + .01 * (prediction - target).square().mean()


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


def validation_iteration_counts(matrices: list[csr_matrix], probes: list[Array],
                                layers: nn.Sequential, geometry: list[Tensor],
                                flat_blocks: list[Array]) -> list[float]:
    iterations = []
    for index, sources in enumerate(probes):
        with torch.inference_mode():
            scaling = diagonal_scaling(layers, geometry[index]).numpy().ravel().astype(np.float64)
        for source in sources:
            _, metrics = solve(matrices[index], source,
                               lambda r: cached_inverse(r, scaling, flat_blocks[index]))
            iterations.append(metrics["iterations"] if metrics["converged"] else float("inf"))
    return iterations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-surface", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
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
    started = perf_counter()
    flat_blocks = [flat_inverse_blocks(float(depth + surface.mean()), nx, nz)
                   for surface, depth in zip(eta, metadata["depth"], strict=True)]
    flat_setup_seconds = (perf_counter() - started) / len(flat_blocks)
    tensor_blocks = [torch.tensor(blocks, dtype=torch.complex64) for blocks in flat_blocks]
    head = nn.Conv2d(8, 2, 1)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(cast(Tensor, head.bias))
    layers = nn.Sequential(nn.Conv2d(4, 8, 1), nn.Tanh(), nn.Conv2d(8, 8, 1), nn.Tanh(), head)
    optimizer = torch.optim.Adam(layers.parameters(), lr=args.learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.learning_rate / 10)

    # Both sets use only the six training geometries. Validation has independent
    # RHS draws; the three held-out geometries are evaluated only after selection.
    started = perf_counter()
    datasets = []
    for count in (args.samples_per_surface, 128):
        samples = []
        for index in range(6):
            rhs, target = training_pairs(stiffness[index], nx, nz, count, rng, flat_blocks[index])
            samples.append(tuple(torch.tensor(value.reshape(count, 1, nz, nx), dtype=torch.float32)
                                 for value in (rhs, target)))
        datasets.append(samples)
    train_data, validation_data = datasets
    sparse_matrices = [torch.sparse_coo_tensor(torch.tensor(np.stack((coo.row, coo.col))), torch.tensor(coo.data),
                                              size=coo.shape, dtype=torch.float32).coalesce()
                       for coo in (matrix.tocoo() for matrix in matrices[:6])]
    data_seconds = perf_counter() - started
    started = perf_counter()
    training, validation = [], []
    batches = [(surface, first) for surface in range(6)
               for first in range(0, args.samples_per_surface, batch_size)]
    with torch.no_grad():
        baseline_loss = float(torch.stack([inverse_loss(layers, rhs, target, geometry[index], sparse_matrices[index], tensor_blocks[index])
                                           for index, (rhs, target) in enumerate(validation_data)]).mean())
    # These 48 fixed validation RHSs never come from the three test geometries.
    validation_rhs = [values[0][np.linspace(0, 127, 8, dtype=int)].numpy().reshape(8, -1).astype(np.float64)
                      for values in validation_data]
    baseline_iterations = validation_iteration_counts(matrices, validation_rhs, layers, geometry, flat_blocks)
    best_score = (sum(baseline_iterations), baseline_loss)
    best_epoch = 0
    validation_iterations = [{"epoch": 0, "iterations": baseline_iterations}]
    torch.save(layers.state_dict(), args.output / "weights.pt")
    print(f"Flat-only validation loss: {baseline_loss:.6g}", flush=True)
    print(f"Flat-only validation GMRES mean/max: {np.mean(baseline_iterations):.3f}/{max(baseline_iterations)}", flush=True)
    for epoch in range(args.epochs):
        rng.shuffle(batches)
        permutations = [torch.randperm(args.samples_per_surface) for _ in range(6)]
        layers.train()
        losses = []
        for surface, first in batches:
            selected = permutations[surface][first:first + batch_size]
            rhs, target = (value[selected] for value in train_data[surface])
            loss = inverse_loss(layers, rhs, target, geometry[surface], sparse_matrices[surface], tensor_blocks[surface])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at epoch {epoch + 1}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        training.append(float(np.mean(losses)))
        layers.eval()
        with torch.no_grad():
            validation.append(float(torch.stack([inverse_loss(layers, rhs, target, geometry[index], sparse_matrices[index], tensor_blocks[index])
                                                  for index, (rhs, target) in enumerate(validation_data)]).mean()))
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            iterations = validation_iteration_counts(matrices, validation_rhs, layers, geometry, flat_blocks)
            validation_iterations.append({"epoch": epoch + 1, "iterations": iterations})
            score = (sum(iterations), validation[-1])
            if score < best_score:
                best_score, best_epoch = score, epoch + 1
                torch.save(layers.state_dict(), args.output / "weights.pt")
            print(f"validation GMRES mean/max: {np.mean(iterations):.3f}/{max(iterations)}", flush=True)
        schedule.step()
        print(f"epoch {epoch + 1}/{args.epochs}: train {training[-1]:.6g}, validation {validation[-1]:.6g}, best {best_epoch}", flush=True)
    training_seconds = perf_counter() - started
    torch.set_num_threads(1)  # Tiny coefficient inference is faster without thread overhead.
    layers.load_state_dict(torch.load(args.output / "weights.pt", map_location="cpu", weights_only=True))
    layers.eval()
    report = {"nx": nx, "nz": nz, "epochs": args.epochs, "learning_rate": args.learning_rate,
              "training_samples": 6 * args.samples_per_surface, "seed": 0,
              "parameters": sum(value.numel() for value in layers.parameters()),
              "training_seconds": training_seconds, "training_loss": training,
              "validation_loss": validation, "flat_validation_loss": baseline_loss,
              "validation_iterations": validation_iterations,
              "best_epoch": best_epoch, "data_seconds": data_seconds,
              "flat_setup_seconds_per_geometry": flat_setup_seconds,
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
        with torch.inference_mode():
            diagonal_scaling(layers, geometry[index])  # Warm setup outside its timing.
            started = perf_counter()
            features = geometry_features(eta[index], float(metadata["depth"][index]), nz)
            scaling = diagonal_scaling(layers, features).numpy().ravel().astype(np.float64)
            neural_setup = perf_counter() - started
        methods = {"none": None, "jacobi": lambda r: r / jacobi_diagonal,
                   "flat": lambda r: flat_inverse(r, flat_blocks[index]),
                   "neural": lambda r: cached_inverse(r, scaling, flat_blocks[index]),
                   "ilu": ilu.solve}
        case = {"simulation_id": int(metadata["simulation_id"][index]),
                "depth": float(metadata["depth"][index]), "methods": {}, "standalone": {},
                "direct_setup_seconds": direct_setup, "ilu_setup_seconds": ilu_setup,
                "neural_setup_seconds": neural_setup}
        for name, inverse in methods.items():
            if inverse is not None:
                prediction = inverse(rhs)
                times = []
                for _ in range(3):
                    started = perf_counter()
                    prediction = inverse(rhs)
                    times.append(perf_counter() - started)
                flux = (full[-nx:, :-nx] @ prediction + full[-nx:, -nx:] @ phi) / (2 * np.pi / nx)
                case["standalone"][name] = {
                    "seconds": float(np.median(times)),
                    "potential_error": float(np.linalg.norm(prediction - reference) / np.linalg.norm(reference)),
                    "relative_residual": float(np.linalg.norm(rhs - matrix @ prediction) / np.linalg.norm(rhs)),
                    "dno_error": float(np.linalg.norm(flux - reference_flux) / np.linalg.norm(reference_flux))}
                print(f"test {case['simulation_id']} {name} WITHOUT GMRES: {case['standalone'][name]}", flush=True)
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
        # Fresh mixed RHS probes on the unseen geometries, not just physical xi.
        probe_rhs, probe_target = training_pairs(full, nx, nz, 128, rng, flat_blocks[index])
        probe_flat = flat_inverse(probe_rhs, flat_blocks[index])
        probe_neural = cached_inverse(probe_rhs, scaling, flat_blocks[index])
        case["unseen_rhs"] = {}
        for name, prediction in (("flat", probe_flat), ("neural", probe_neural)):
            errors = np.linalg.norm(prediction - probe_target, axis=1) / np.linalg.norm(probe_target, axis=1)
            residuals = np.linalg.norm((matrix @ prediction.T).T - probe_rhs, axis=1) / np.linalg.norm(probe_rhs, axis=1)
            case["unseen_rhs"][name] = {"potential_median_p95": np.percentile(errors, [50, 95]).tolist(),
                                       "residual_median_p95": np.percentile(residuals, [50, 95]).tolist()}
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
                    blocks = flat_inverse_blocks(float(metadata["depth"][6] + surface.mean()), nx, nz)
                    with torch.inference_mode():
                        scaling = diagonal_scaling(layers, features).numpy().ravel().astype(np.float64)
                    potential, metrics = solve(matrix, rhs, lambda r: cached_inverse(r, scaling, blocks))
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
