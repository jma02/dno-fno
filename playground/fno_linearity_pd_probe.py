"""Linearity and positive-definiteness probe for a trained FNO surrogate.

The water-wave Dirichlet-to-Neumann operator G(eta) is linear in xi at fixed
eta and positive: <xi, G(eta) xi> >= 0. A faithful surrogate should
approximately preserve both properties even though the FNO architecture has
no constraint enforcing them.

Tests at one or more fixed eta snapshots:

1. Linearity in xi. Build K real-valued unit-L^2 Fourier basis vectors
   {xi_k}. Evaluate g_k = model(eta, xi_k). For arbitrary convex
   combinations xi_alpha = sum a_i xi_i (a_i >= 0, sum a_i = 1) check
       || model(eta, xi_alpha) - sum a_i g_i || / || model(eta, xi_alpha) ||
   on (a) all pairwise edges of the K-simplex with an alpha grid, and
   (b) Dirichlet-sampled interior points.

2. Positive definiteness. Form the K x K matrix
       A_ij = < xi_i, model(eta, xi_j) >_{L^2}
   Report the antisymmetric defect || A - A^T || / || A || (the true G is
   self-adjoint), and the eigenvalues of (A + A^T) / 2 (must be >= 0 for a
   positive operator).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver.evals.model_rollout import build_predict_gxi, load_run  # noqa: E402


def fourier_basis(nx: int, length: float, k_max: int) -> tuple[np.ndarray, list[str]]:
    """Real-valued, unit-L^2 cosine/sine Fourier basis up to wavenumber k_max."""
    x = np.linspace(0.0, length, nx, endpoint=False)
    basis: list[np.ndarray] = []
    labels: list[str] = []
    dx = length / nx
    for k in range(1, k_max + 1):
        for kind in ("cos", "sin"):
            v = (np.cos if kind == "cos" else np.sin)(k * 2.0 * np.pi * x / length)
            v = v / np.sqrt(float(np.sum(v * v) * dx))
            basis.append(v.astype(np.float32))
            labels.append(f"{kind}({k})")
    return np.stack(basis, axis=0), labels


def inner_product(a: np.ndarray, b: np.ndarray, dx: float) -> float:
    return float(np.sum(a * b) * dx)


def select_eta(trajs_dir: Path, regime: str) -> tuple[np.ndarray, float, dict[str, object]]:
    """Median-final-error IC, mid-rollout snapshot. Returns (eta, depth, meta)."""
    if regime == "flat":
        # Caller must supply nx/depth; we return a sentinel and let main fill it.
        raise RuntimeError("flat regime is handled in main(); should not call select_eta('flat').")
    npz_path = trajs_dir / f"{regime}_trajs.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found")
    with np.load(npz_path) as d:
        truth_eta = np.asarray(d["truth_eta"])
        depths = np.asarray(d["depths"])
        rel_l2_eta = np.asarray(d["rel_l2_eta"])
    n_t, _, nx = truth_eta.shape
    final = rel_l2_eta[-1]
    order = np.argsort(np.where(np.isfinite(final), final, np.inf))
    j = int(order[len(order) // 2])
    t_idx = n_t // 2
    eta = truth_eta[t_idx, j, :].astype(np.float32)
    depth = float(depths[j])
    return eta, depth, {"ic_idx": j, "t_idx": int(t_idx), "n_t": int(n_t), "nx": int(nx)}


def evaluate_basis(predict, eta: np.ndarray, basis: np.ndarray) -> np.ndarray:
    K = basis.shape[0]
    g = np.empty_like(basis)
    eta_j = jnp.asarray(eta)
    for k in range(K):
        g[k] = np.asarray(jax.device_get(predict(eta_j, jnp.asarray(basis[k]))), dtype=np.float32)
    return g


def linearity_defects(
    predict,
    eta: np.ndarray,
    basis: np.ndarray,
    g_basis: np.ndarray,
    alpha_grid: tuple[float, ...],
    n_random: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    K, _ = basis.shape
    eta_j = jnp.asarray(eta)
    edge: list[float] = []
    for i in range(K):
        for j in range(i + 1, K):
            for a in alpha_grid:
                xi_a = (a * basis[i] + (1.0 - a) * basis[j]).astype(np.float32)
                g_pred = np.asarray(jax.device_get(predict(eta_j, jnp.asarray(xi_a))))
                g_lin = a * g_basis[i] + (1.0 - a) * g_basis[j]
                edge.append(float(np.linalg.norm(g_pred - g_lin) /
                                  (np.linalg.norm(g_pred) + 1e-30)))
    random: list[float] = []
    for _ in range(n_random):
        alpha = rng.dirichlet(np.ones(K))
        xi_a = (alpha[:, None] * basis).sum(axis=0).astype(np.float32)
        g_pred = np.asarray(jax.device_get(predict(eta_j, jnp.asarray(xi_a))))
        g_lin = (alpha[:, None] * g_basis).sum(axis=0)
        random.append(float(np.linalg.norm(g_pred - g_lin) /
                            (np.linalg.norm(g_pred) + 1e-30)))
    edge_arr = np.asarray(edge, dtype=np.float64)
    rand_arr = np.asarray(random, dtype=np.float64)
    return {
        "edge_defects": edge_arr.tolist(),
        "random_defects": rand_arr.tolist(),
        "edge_summary": {
            "max": float(edge_arr.max()) if edge_arr.size else 0.0,
            "p90": float(np.quantile(edge_arr, 0.9)) if edge_arr.size else 0.0,
            "median": float(np.median(edge_arr)) if edge_arr.size else 0.0,
        },
        "random_summary": {
            "max": float(rand_arr.max()) if rand_arr.size else 0.0,
            "p90": float(np.quantile(rand_arr, 0.9)) if rand_arr.size else 0.0,
            "median": float(np.median(rand_arr)) if rand_arr.size else 0.0,
        },
    }


def positive_def_check(
    basis: np.ndarray,
    g_basis: np.ndarray,
    length: float,
) -> dict[str, object]:
    K, nx = basis.shape
    dx = length / nx
    A = (basis @ g_basis.T) * dx  # (K, K): A_ij = <basis_i, g_basis_j>
    sym = 0.5 * (A + A.T)
    asym = A - A.T
    sym_def = float(np.linalg.norm(asym, "fro") / (np.linalg.norm(A, "fro") + 1e-30))
    eigs = np.linalg.eigvalsh(sym)
    eig_min = float(np.min(eigs))
    eig_max = float(np.max(eigs))
    neg_frac = float(np.sum(eigs < -1e-10 * max(abs(eig_max), 1e-30)) / K)
    return {
        "A_norm": float(np.linalg.norm(A, "fro")),
        "antisym_defect": sym_def,
        "eig_min": eig_min,
        "eig_max": eig_max,
        "neg_eig_frac": neg_frac,
        "condition_number": float(eig_max / max(eig_min, 1e-30)) if eig_min > 0 else float("inf"),
        "eigenvalues": eigs.astype(float).tolist(),
    }


def plot_results(all_results: list[dict], output_dir: Path) -> None:
    n = len(all_results)
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 5.6), constrained_layout=True, squeeze=False)
    for col, r in enumerate(all_results):
        # Top row: eigenvalue spectrum
        eigs = np.asarray(r["pd"]["eigenvalues"], dtype=np.float64)
        idx = np.arange(eigs.size) + 1
        pos = eigs > 0
        ax = axes[0, col]
        ax.bar(idx[pos], eigs[pos], color="#4c72b0", label="positive")
        if (~pos).any():
            ax.bar(idx[~pos], eigs[~pos], color="#c44e52", label="negative")
        ax.axhline(0, color="0.4", lw=0.7)
        ax.set_title(f"{r['regime']}  $\\eta_\\infty={r['eta_absmax']:.2g}$")
        ax.set_xlabel("eigenvalue index")
        ax.set_ylabel(r"eig of $(A+A^T)/2$")
        if (~pos).any():
            ax.legend(fontsize=7)
        # Bottom row: linearity defects
        edge = np.asarray(r["linearity"]["edge_defects"], dtype=np.float64)
        rnd = np.asarray(r["linearity"]["random_defects"], dtype=np.float64)
        ax = axes[1, col]
        if edge.size > 0:
            ax.hist(np.log10(np.maximum(edge, 1e-12)), bins=40, alpha=0.7, color="#4c72b0", label="pairwise edges")
        if rnd.size > 0:
            ax.hist(np.log10(np.maximum(rnd, 1e-12)), bins=20, alpha=0.7, color="#dd8452", label="random simplex")
        ax.set_xlabel(r"$\log_{10}$ relative defect")
        ax.set_ylabel("count")
        ax.legend(fontsize=7)
        ax.set_title("linearity defect")
    fig.savefig(output_dir / "linearity_pd_summary.png", dpi=300)
    fig.savefig(output_dir / "linearity_pd_summary.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default="outputs/fno_w128b6_v3_hclip5_20260514_063811")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--trajs_dir",
                        default="outputs/fno_w128b6_v3_hclip5_20260514_063811/eval_suite")
    parser.add_argument("--regimes", default="flat,linear,bf_g0,tanaka_g0",
                        help='Comma-separated regimes. Use "flat" for eta=0 sanity.')
    parser.add_argument("--k_max", type=int, default=10,
                        help="Fourier basis up to this wavenumber (K = 2 * k_max).")
    parser.add_argument("--n_random_simplex", type=int, default=64)
    parser.add_argument("--alpha_grid", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--length", type=float, default=2.0 * np.pi)
    parser.add_argument("--flat_nx", type=int, default=1024,
                        help='nx for the "flat" sanity case.')
    parser.add_argument("--flat_depth", type=float, default=1.0)
    parser.add_argument("--output_dir", default="playground/runs/fno_linearity_pd")
    parser.add_argument("--scale_to_training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = (REPO_ROOT / args.run_dir).resolve()
    print(f"loading {run_dir}", flush=True)
    loaded = load_run(run_dir, checkpoint=args.checkpoint)
    xi_scale = float(np.asarray(loaded.stats["feature_absmax"]).reshape(-1)[1])
    print(f"  model={loaded.config['model']} epoch={loaded.epoch} xi_scale={xi_scale:g}", flush=True)

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trajs_dir = (REPO_ROOT / args.trajs_dir).resolve()
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    alpha_grid = tuple(float(a) for a in args.alpha_grid.split(",") if a.strip())
    rng = np.random.default_rng(args.seed)
    scale = xi_scale if args.scale_to_training else 1.0

    all_results: list[dict] = []
    for regime in regimes:
        if regime == "flat":
            nx = args.flat_nx
            eta = np.zeros((nx,), dtype=np.float32)
            depth = float(args.flat_depth)
            meta = {"ic_idx": -1, "t_idx": -1, "n_t": 0, "nx": int(nx)}
        else:
            try:
                eta, depth, meta = select_eta(trajs_dir, regime)
            except FileNotFoundError as exc:
                print(f"[{regime}] skipping: {exc}", flush=True)
                continue
            nx = meta["nx"]
        eta_absmax = float(np.max(np.abs(eta)))
        print(f"[{regime}] depth={depth:g} nx={nx} eta_absmax={eta_absmax:.3g}", flush=True)

        basis_unit, labels = fourier_basis(nx, args.length, args.k_max)
        basis = (basis_unit * scale).astype(np.float32)

        predict = build_predict_gxi(loaded, depth)
        # warm
        _ = predict(jnp.asarray(eta), jnp.asarray(basis[0]))

        g_basis = evaluate_basis(predict, eta, basis)
        lin = linearity_defects(predict, eta, basis, g_basis,
                                alpha_grid=alpha_grid,
                                n_random=args.n_random_simplex,
                                rng=rng)
        pd = positive_def_check(basis, g_basis, args.length)

        print(f"  linearity edge   max={lin['edge_summary']['max']:.3e}  "
              f"p90={lin['edge_summary']['p90']:.3e}  med={lin['edge_summary']['median']:.3e}",
              flush=True)
        print(f"  linearity random max={lin['random_summary']['max']:.3e}  "
              f"p90={lin['random_summary']['p90']:.3e}  med={lin['random_summary']['median']:.3e}",
              flush=True)
        print(f"  PD: antisym={pd['antisym_defect']:.3e}  "
              f"eig_min={pd['eig_min']:+.3e}  eig_max={pd['eig_max']:.3e}  "
              f"neg_frac={pd['neg_eig_frac']:.1%}",
              flush=True)

        all_results.append({
            "regime": regime,
            "depth": depth,
            "eta_absmax": eta_absmax,
            "ic_idx": int(meta["ic_idx"]),
            "t_idx": int(meta["t_idx"]),
            "nx": int(nx),
            "K": int(basis.shape[0]),
            "labels": labels,
            "linearity": lin,
            "pd": pd,
        })

    if not all_results:
        print("no regimes ran; exiting.", flush=True)
        return

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "model": loaded.config.get("model", "fno"),
        "epoch": int(loaded.epoch),
        "xi_scale": xi_scale,
        "scale_to_training": bool(args.scale_to_training),
        "k_max": int(args.k_max),
        "alpha_grid": list(alpha_grid),
        "n_random_simplex": int(args.n_random_simplex),
        "results": all_results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_results(all_results, output_dir)
    print(f"wrote {output_dir / 'summary.json'}", flush=True)
    print(f"wrote {output_dir / 'linearity_pd_summary.png'}", flush=True)


if __name__ == "__main__":
    main()
