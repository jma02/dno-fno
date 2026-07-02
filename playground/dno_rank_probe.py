from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver.solvers.dno_series_jax import build_grid, dno_series_eval


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    kind: str
    prefix: str | None = None
    default_depth: float = 1.0


@dataclass(frozen=True)
class DatasetSample:
    name: str
    eta: np.ndarray
    xi: np.ndarray
    x: np.ndarray
    depth: np.ndarray
    path: Path


DEFAULT_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec("linear", REPO_ROOT / "data" / "linear.npz", "sharded", default_depth=1.0),
    DatasetSpec("stokes", REPO_ROOT / "data" / "stokes_finite.npz", "sharded", default_depth=1.0),
    DatasetSpec("stokes_deep", REPO_ROOT / "data" / "stokes_deep.npz", "sharded", default_depth=1000.0),
    DatasetSpec("tanaka", REPO_ROOT / "playground" / "data" / "tanaka_100k.npz", "flat", default_depth=1.0),
    DatasetSpec("bf", REPO_ROOT / "data" / "stokes_bf_dataset.npz", "prefixed", prefix="stokes", default_depth=1000.0),
    DatasetSpec("random", REPO_ROOT / "playground" / "data" / "random_sea_deep_100k.npz", "flat", default_depth=1000.0),
)


def _zero_mean_rows(array: np.ndarray) -> np.ndarray:
    return (array - array.mean(axis=1, keepdims=True)).astype(np.float32, copy=False)


def _take_rows(array: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.asarray(array[rows], dtype=np.float32)


def _sample_rows(num_rows: int, max_samples: int, seed: int) -> np.ndarray:
    if num_rows <= max_samples:
        return np.arange(num_rows)
    return np.sort(np.random.default_rng(seed).choice(num_rows, size=max_samples, replace=False))


def _load_flat(spec: DatasetSpec, max_samples: int, seed: int) -> DatasetSample:
    with np.load(spec.path) as archive:
        rows = _sample_rows(int(archive["eta"].shape[0]), max_samples, seed)
        eta = _take_rows(archive["eta"], rows)
        xi = _take_rows(archive["xi"], rows)
        x = np.asarray(archive["x"], dtype=np.float32)
        depth = (
            _take_rows(archive["depth"], rows).reshape(-1)
            if "depth" in archive.files
            else np.full(rows.shape[0], spec.default_depth, dtype=np.float32)
        )
    return DatasetSample(spec.name, eta, _zero_mean_rows(xi), x, depth, spec.path)


def _load_prefixed(spec: DatasetSpec, max_samples: int, seed: int) -> DatasetSample:
    if spec.prefix is None:
        raise ValueError(f"prefixed dataset {spec.name!r} needs prefix")
    with np.load(spec.path) as archive:
        eta_key = f"{spec.prefix}_eta"
        xi_key = f"{spec.prefix}_xi"
        rows = _sample_rows(int(archive[eta_key].shape[0]), max_samples, seed)
        eta = _take_rows(archive[eta_key], rows)
        xi = _take_rows(archive[xi_key], rows)
        x = np.asarray(archive["x"], dtype=np.float32)
        depth = np.full(rows.shape[0], spec.default_depth, dtype=np.float32)
    return DatasetSample(spec.name, eta, _zero_mean_rows(xi), x, depth, spec.path)


def _shard_tags(files: Iterable[str]) -> list[str]:
    return sorted(name.removeprefix("eta_batch_") for name in files if name.startswith("eta_batch_"))


def _load_sharded(spec: DatasetSpec, max_samples: int, seed: int) -> DatasetSample:
    eta_parts: list[np.ndarray] = []
    xi_parts: list[np.ndarray] = []
    depth_parts: list[np.ndarray] = []
    remaining = max_samples
    with np.load(spec.path) as archive:
        x = np.asarray(archive["x"], dtype=np.float32)
        tags = _shard_tags(archive.files)
        for tag in tags:
            if remaining <= 0:
                break
            eta_key = f"eta_batch_{tag}"
            xi_key = f"xi_batch_{tag}"
            depth_key = f"depth_batch_{tag}"
            n_rows = int(archive[eta_key].shape[0])
            take = min(remaining, n_rows)
            rows = _sample_rows(n_rows, take, seed + len(eta_parts))
            eta_parts.append(_take_rows(archive[eta_key], rows))
            xi_parts.append(_take_rows(archive[xi_key], rows))
            if depth_key in archive.files:
                depth_parts.append(_take_rows(archive[depth_key], rows).reshape(-1))
            else:
                depth_parts.append(np.full(rows.shape[0], spec.default_depth, dtype=np.float32))
            remaining -= rows.shape[0]
    eta = np.concatenate(eta_parts, axis=0)
    xi = np.concatenate(xi_parts, axis=0)
    depth = np.concatenate(depth_parts, axis=0)
    return DatasetSample(spec.name, eta, _zero_mean_rows(xi), x, depth, spec.path)


def load_dataset_sample(spec: DatasetSpec, max_samples: int, seed: int) -> DatasetSample:
    if spec.kind == "flat":
        return _load_flat(spec, max_samples, seed)
    if spec.kind == "prefixed":
        return _load_prefixed(spec, max_samples, seed)
    if spec.kind == "sharded":
        return _load_sharded(spec, max_samples, seed)
    raise ValueError(f"unknown dataset kind {spec.kind!r}")


def compute_pca_modes(xi: np.ndarray, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    centered = xi - xi.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    eig = singular_values**2
    explained = eig / max(float(eig.sum()), 1e-30)
    return vh[:n_modes].astype(np.float32), explained[:n_modes].astype(np.float64)


def choose_eta_rows(eta: np.ndarray, quantiles: tuple[float, ...]) -> list[int]:
    eta_std = eta.std(axis=1)
    return [
        int(np.argmin(np.abs(eta_std - np.quantile(eta_std, q))))
        for q in quantiles
    ]


def numerical_ranks(singular_values: np.ndarray) -> dict[str, int]:
    if singular_values.size == 0 or singular_values[0] <= 0:
        return {"rel_1e-2": 0, "rel_1e-3": 0, "rel_1e-4": 0, "energy_99": 0, "energy_999": 0}
    rel = singular_values / singular_values[0]
    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    return {
        "rel_1e-2": int(np.count_nonzero(rel >= 1e-2)),
        "rel_1e-3": int(np.count_nonzero(rel >= 1e-3)),
        "rel_1e-4": int(np.count_nonzero(rel >= 1e-4)),
        "energy_99": int(np.searchsorted(energy, 0.99) + 1),
        "energy_999": int(np.searchsorted(energy, 0.999) + 1),
    }


def apply_dno_to_modes(
    eta: np.ndarray,
    xi_modes: np.ndarray,
    x: np.ndarray,
    depth: float,
    *,
    order: int,
    pad_factor: int,
    mode_batch_size: int,
) -> np.ndarray:
    length = float(x[1] - x[0]) * float(x.shape[0])
    _, k = build_grid(int(x.shape[0]), length)

    @jax.jit
    def eval_batch(eta_batch: jax.Array, xi_batch: jax.Array) -> jax.Array:
        return dno_series_eval(eta_batch, xi_batch, k, depth, order, pad_factor=pad_factor)

    chunks: list[np.ndarray] = []
    for start in range(0, xi_modes.shape[0], mode_batch_size):
        xi_chunk = xi_modes[start : start + mode_batch_size]
        eta_chunk = np.broadcast_to(eta[None, :], xi_chunk.shape)
        out = eval_batch(jnp.asarray(eta_chunk), jnp.asarray(xi_chunk))
        chunks.append(np.asarray(jax.device_get(out), dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def plot_spectra(output_path: Path, results: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, result in zip(axes.ravel(), results):
        for probe in result["eta_probes"]:
            s = np.asarray(probe["singular_values"], dtype=np.float64)
            ax.semilogy(np.arange(1, s.shape[0] + 1), s / max(s[0], 1e-30), label=f"q={probe['eta_quantile']}")
        ax.set_title(str(result["dataset"]))
        ax.set_xlabel("restricted singular index")
        ax.set_ylabel("sigma / sigma_1")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(results):]:
        ax.axis("off")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_dataset_names(raw: str) -> tuple[str, ...]:
    if raw == "all":
        return tuple(spec.name for spec in DEFAULT_DATASETS)
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe numerical rank of G(eta) on xi PCA modes.")
    parser.add_argument("--datasets", default="all", help="Comma-separated names or 'all'.")
    parser.add_argument("--max_samples", type=int, default=2048)
    parser.add_argument("--pca_modes", type=int, default=128)
    parser.add_argument("--eta_quantiles", default="0.1,0.5,0.9")
    parser.add_argument("--dno_order", type=int, default=6)
    parser.add_argument("--pad_factor", type=int, default=8)
    parser.add_argument("--mode_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="playground/runs/dno_rank_probe")
    args = parser.parse_args()

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = {spec.name: spec for spec in DEFAULT_DATASETS}
    dataset_names = parse_dataset_names(args.datasets)
    quantiles = tuple(float(part) for part in args.eta_quantiles.split(",") if part.strip())
    results: list[dict[str, object]] = []

    for dataset_index, name in enumerate(dataset_names):
        if name not in specs:
            raise ValueError(f"unknown dataset {name!r}; available={sorted(specs)}")
        spec = specs[name]
        print(f"[{name}] loading capped sample from {spec.path}", flush=True)
        sample = load_dataset_sample(spec, args.max_samples, args.seed + dataset_index * 1000)
        print(f"[{name}] PCA on xi shape={sample.xi.shape}", flush=True)
        pca_modes, explained = compute_pca_modes(sample.xi, args.pca_modes)

        probes: list[dict[str, object]] = []
        for q, eta_row in zip(quantiles, choose_eta_rows(sample.eta, quantiles)):
            depth = float(sample.depth[eta_row])
            print(f"[{name}] applying G(eta) for q={q:g}, row={eta_row}, depth={depth:g}", flush=True)
            t0 = time.time()
            outputs = apply_dno_to_modes(
                sample.eta[eta_row],
                pca_modes,
                sample.x,
                depth,
                order=args.dno_order,
                pad_factor=args.pad_factor,
                mode_batch_size=args.mode_batch_size,
            )
            singular_values = np.linalg.svd(outputs.T, compute_uv=False)
            probes.append(
                {
                    "eta_quantile": float(q),
                    "eta_row": int(eta_row),
                    "eta_std": float(sample.eta[eta_row].std()),
                    "depth": depth,
                    "elapsed_seconds": float(time.time() - t0),
                    "singular_values": singular_values.astype(float).tolist(),
                    "ranks": numerical_ranks(singular_values),
                }
            )

        results.append(
            {
                "dataset": name,
                "path": str(sample.path),
                "num_samples": int(sample.xi.shape[0]),
                "nx": int(sample.xi.shape[1]),
                "pca_modes": int(pca_modes.shape[0]),
                "xi_pca_explained": explained.tolist(),
                "xi_pca_cumulative_16": float(np.sum(explained[:16])),
                "xi_pca_cumulative_32": float(np.sum(explained[:32])),
                "xi_pca_cumulative_64": float(np.sum(explained[:64])),
                "xi_pca_cumulative_all": float(np.sum(explained)),
                "eta_probes": probes,
            }
        )

    summary = {
        "backend": jax.default_backend(),
        "max_samples": args.max_samples,
        "pca_modes": args.pca_modes,
        "dno_order": args.dno_order,
        "pad_factor": args.pad_factor,
        "mode_batch_size": args.mode_batch_size,
        "eta_quantiles": list(quantiles),
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_spectra(output_dir / "restricted_singular_spectra.png", results)
    print(f"wrote {output_dir / 'summary.json'}", flush=True)
    print(f"wrote {output_dir / 'restricted_singular_spectra.png'}", flush=True)


if __name__ == "__main__":
    main()
