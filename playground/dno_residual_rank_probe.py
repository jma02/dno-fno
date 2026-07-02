"""Probe numerical rank of the DNO residual G_M(eta) - G_0 on xi PCA modes.

Mirrors ``dno_rank_probe.py``: same dataset specs, same xi-PCA basis per dataset,
same eta-quantile probe rows. The target operator differs:

    R_M(eta) xi = G_M(eta) xi - G_0(h) xi,

where G_M is the M-th order DNO series (``dno_series_eval``) and G_0(h) xi =
IFFT(k tanh(hk) FFT(xi)) is the flat-surface (linear) DNO. The residual isolates
the eta-induced nonlinear part of the operator; its restricted singular spectrum
on the xi-PCA basis tells us how "low-rank" that nonlinearity is.

M is swept over ``--orders`` (default 4,5,6,7,8). The xi-PCA basis is computed
once per dataset and reused across M, so the only thing that varies is the
restricted singular spectrum of R_M.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver.solvers.dno_series_jax import (  # noqa: E402
    build_grid,
    dno_series_eval,
    make_linear_dno_symbol,
    myfft,
    myifft,
)
from playground.dno_rank_probe import (  # noqa: E402
    DEFAULT_DATASETS,
    choose_eta_rows,
    compute_pca_modes,
    load_dataset_sample,
    numerical_ranks,
    parse_dataset_names,
)


def apply_dno_residual_to_modes(
    eta: np.ndarray,
    xi_modes: np.ndarray,
    x: np.ndarray,
    depth: float,
    *,
    order: int,
    pad_factor: int,
    mode_batch_size: int,
) -> np.ndarray:
    """Return R_M(eta) on each xi mode, shape (n_modes, nx)."""
    nx = int(x.shape[0])
    length = float(x[1] - x[0]) * float(nx)
    _, k = build_grid(nx, length)
    g0 = make_linear_dno_symbol(k, depth)

    @jax.jit
    def eval_batch(eta_batch: jax.Array, xi_batch: jax.Array) -> jax.Array:
        full = dno_series_eval(eta_batch, xi_batch, k, depth, order, pad_factor=pad_factor)
        # Linear DNO action on xi: IFFT(g0 * FFT(xi)). myfft zeros the Nyquist.
        xi_hat = myfft(xi_batch, nx)
        linear = myifft(xi_hat * g0)
        return full - linear

    chunks: list[np.ndarray] = []
    for start in range(0, xi_modes.shape[0], mode_batch_size):
        xi_chunk = xi_modes[start : start + mode_batch_size]
        eta_chunk = np.broadcast_to(eta[None, :], xi_chunk.shape)
        out = eval_batch(jnp.asarray(eta_chunk), jnp.asarray(xi_chunk))
        chunks.append(np.asarray(jax.device_get(out), dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def plot_spectra_by_order(
    output_path: Path,
    results: list[dict[str, object]],
    quantile_to_plot: float,
) -> None:
    n = len(results)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, result in zip(axes_flat, results):
        chosen = None
        for probe in result["eta_probes"]:
            if abs(float(probe["eta_quantile"]) - quantile_to_plot) < 1e-6:
                chosen = probe
                break
        if chosen is None:
            ax.set_title(f"{result['dataset']} (q={quantile_to_plot} missing)")
            ax.axis("off")
            continue
        for entry in chosen["by_order"]:
            M = int(entry["order"])
            s = np.asarray(entry["singular_values"], dtype=np.float64)
            s_norm = s / max(s[0], 1e-30)
            ax.semilogy(np.arange(1, s.shape[0] + 1), s_norm, label=f"M={M}")
        ax.set_title(f"{result['dataset']} (q={quantile_to_plot})")
        ax.set_xlabel("restricted singular index")
        ax.set_ylabel(r"$\sigma_i / \sigma_1$")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)
    for ax in axes_flat[len(results):]:
        ax.axis("off")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_absolute_magnitude_by_order(
    output_path: Path,
    results: list[dict[str, object]],
    quantile_to_plot: float,
) -> None:
    """Absolute (un-normalized) singular spectrum — shows how residual MAGNITUDE grows with M."""
    n = len(results)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, result in zip(axes_flat, results):
        chosen = None
        for probe in result["eta_probes"]:
            if abs(float(probe["eta_quantile"]) - quantile_to_plot) < 1e-6:
                chosen = probe
                break
        if chosen is None:
            ax.set_title(f"{result['dataset']} (q={quantile_to_plot} missing)")
            ax.axis("off")
            continue
        for entry in chosen["by_order"]:
            M = int(entry["order"])
            s = np.asarray(entry["singular_values"], dtype=np.float64)
            ax.semilogy(np.arange(1, s.shape[0] + 1), np.maximum(s, 1e-30), label=f"M={M}")
        ax.set_title(f"{result['dataset']} (q={quantile_to_plot}) — absolute σ")
        ax.set_xlabel("restricted singular index")
        ax.set_ylabel(r"$\sigma_i$")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)
    for ax in axes_flat[len(results):]:
        ax.axis("off")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="all", help="Comma-separated names or 'all'.")
    parser.add_argument("--max_samples", type=int, default=2048)
    parser.add_argument("--pca_modes", type=int, default=128)
    parser.add_argument("--eta_quantiles", default="0.1,0.5,0.9")
    parser.add_argument("--orders", default="4,5,6,7,8", help="Comma-separated DNO series orders M.")
    parser.add_argument("--pad_factor", type=int, default=8)
    parser.add_argument("--mode_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="playground/runs/dno_residual_rank_probe")
    args = parser.parse_args()

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = {spec.name: spec for spec in DEFAULT_DATASETS}
    dataset_names = parse_dataset_names(args.datasets)
    quantiles = tuple(float(part) for part in args.eta_quantiles.split(",") if part.strip())
    orders = tuple(int(part) for part in args.orders.split(",") if part.strip())
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
            print(f"[{name}] residual probe q={q:g}, row={eta_row}, depth={depth:g}", flush=True)
            by_order: list[dict[str, object]] = []
            for M in orders:
                t0 = time.time()
                outputs = apply_dno_residual_to_modes(
                    sample.eta[eta_row],
                    pca_modes,
                    sample.x,
                    depth,
                    order=M,
                    pad_factor=args.pad_factor,
                    mode_batch_size=args.mode_batch_size,
                )
                singular_values = np.linalg.svd(outputs.T, compute_uv=False)
                elapsed = float(time.time() - t0)
                by_order.append(
                    {
                        "order": int(M),
                        "elapsed_seconds": elapsed,
                        "frobenius": float(np.linalg.norm(outputs)),
                        "singular_values": singular_values.astype(float).tolist(),
                        "ranks": numerical_ranks(singular_values),
                    }
                )
                print(
                    f"  M={M:>2}: σ1={singular_values[0]:.3e} "
                    f"σ1/σ_max_M={singular_values[0]/by_order[0]['singular_values'][0]:.2f} "
                    f"ranks={by_order[-1]['ranks']} t={elapsed:.1f}s",
                    flush=True,
                )
            probes.append(
                {
                    "eta_quantile": float(q),
                    "eta_row": int(eta_row),
                    "eta_std": float(sample.eta[eta_row].std()),
                    "depth": depth,
                    "by_order": by_order,
                }
            )

        results.append(
            {
                "dataset": name,
                "path": str(sample.path),
                "num_samples": int(sample.xi.shape[0]),
                "nx": int(sample.xi.shape[1]),
                "pca_modes": int(pca_modes.shape[0]),
                "xi_pca_explained_top64": explained[:64].tolist(),
                "eta_probes": probes,
            }
        )

    summary = {
        "backend": jax.default_backend(),
        "max_samples": args.max_samples,
        "pca_modes": args.pca_modes,
        "pad_factor": args.pad_factor,
        "mode_batch_size": args.mode_batch_size,
        "eta_quantiles": list(quantiles),
        "orders": list(orders),
        "target_operator": "R_M(eta) = G_M(eta) - G_0(h)",
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for q in quantiles:
        rel_path = output_dir / f"residual_singular_spectra_q{q:g}.png"
        plot_spectra_by_order(rel_path, results, q)
        print(f"wrote {rel_path}", flush=True)
        abs_path = output_dir / f"residual_singular_spectra_abs_q{q:g}.png"
        plot_absolute_magnitude_by_order(abs_path, results, q)
        print(f"wrote {abs_path}", flush=True)
    print(f"wrote {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
