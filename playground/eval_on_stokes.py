"""Evaluate a trained FNO on Stokes waves.

Usage:
    uv run python playground/eval_on_stokes.py --run_dir playground/runs/combined_fno --depth 1000
    uv run python playground/eval_on_stokes.py --run_dir playground/runs/combined_fno --depth 1 --ichoi 1
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from fno_jax.fno1d import FNO1d
from solver.data.stokes_truth_jax import stokes_eta_xi
from solver.solvers.dno_series_jax import build_grid, dno_series_eval


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _load_norm(run_dir: Path) -> dict:
    return json.loads((run_dir / "norm_stats.json").read_text(encoding="utf-8"))


def normalize_features(eta: np.ndarray, xi: np.ndarray, ns: dict) -> np.ndarray:
    stacked = np.stack((eta, xi), axis=-1).astype(np.float32)
    if ns["mode"] == "scale":
        absmax = np.asarray(ns["feature_absmax"], dtype=np.float32).reshape(1, 1, 2)
        return stacked / np.where(absmax > 0, absmax, 1.0)
    fmin = np.asarray(ns["feature_min"], dtype=np.float32).reshape(1, 1, 2)
    fmax = np.asarray(ns["feature_max"], dtype=np.float32).reshape(1, 1, 2)
    return (stacked - fmin) / (fmax - fmin + 1e-8) * 2 - 1


def denormalize_targets(arr: np.ndarray, ns: dict) -> np.ndarray:
    if ns["mode"] == "scale":
        ta = float(ns["target_absmax"]) if ns["target_absmax"] > 0 else 1.0
        return arr * ta
    tmin, tmax = float(ns["target_min"]), float(ns["target_max"])
    return ((arr + 1.0) * 0.5) * (tmax - tmin + 1e-8) + tmin


# ---------------------------------------------------------------------------
# Stokes generation
# ---------------------------------------------------------------------------

def generate_stokes_samples(
    n0_values: list[int],
    a0_values: list[float],
    *,
    nx: int = 1024,
    length: float = 164.0,
    depth: float = 1000.0,
    gravity: float = 1.0,
    dno_order: int = 6,
    pad_factor: int = 8,
    ichoi: int = 0,
) -> dict[str, np.ndarray | list[dict]]:
    x, k = build_grid(nx, length)
    eta_list, xi_list, gxi_list, labels = [], [], [], []

    for n0 in n0_values:
        for a0 in a0_values:
            k0 = n0 * 2.0 * jnp.pi / length
            eps = float(k0) * a0
            if eps > 0.30:
                continue
            try:
                eta_arr, xi_arr = stokes_eta_xi(
                    x, 0.0, n0, a0, length,
                    depth if ichoi == 1 else 1000.0,
                    gravity, ichoi=ichoi,
                )
                if float(jnp.max(jnp.abs(eta_arr))) > 50 or float(jnp.max(jnp.abs(xi_arr))) > 500:
                    print(f"  SKIP n0={n0}, a0={a0}: blowup")
                    continue
                gxi_arr = dno_series_eval(eta_arr, xi_arr, k, depth, dno_order, pad_factor=pad_factor)
                if float(jnp.max(jnp.abs(gxi_arr))) > 500:
                    print(f"  SKIP n0={n0}, a0={a0}: DNO blowup")
                    continue
                eta_list.append(np.asarray(eta_arr, dtype=np.float32))
                xi_list.append(np.asarray(xi_arr, dtype=np.float32))
                gxi_list.append(np.asarray(gxi_arr, dtype=np.float32))
                labels.append({"n0": n0, "a0": a0, "eps": eps, "ichoi": ichoi})
            except Exception as e:
                print(f"  SKIP n0={n0}, a0={a0}: {e}")

    return {
        "eta": np.stack(eta_list),
        "xi": np.stack(xi_list),
        "gxi": np.stack(gxi_list),
        "x": np.asarray(x, dtype=np.float32),
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FNO on Stokes waves.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--n0", type=int, nargs="+", default=[4, 6, 8, 10, 14, 20])
    parser.add_argument("--a0", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.08, 0.10])
    parser.add_argument("--ichoi", type=int, default=0, help="0=deep, 1=finite depth")
    parser.add_argument("--depth", type=float, default=1000.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ns = _load_norm(run_dir)
    log_depth = np.log(args.depth).astype(np.float32)

    # ---- Load model ----
    fa = np.asarray(ns["feature_absmax"], dtype=np.float32)
    ta = float(ns["target_absmax"]) if ns["target_absmax"] > 0 else 1.0
    model = FNO1d(
        modes=config["modes"],
        width=config["width"],
        n_blocks=config["n_blocks"],
        xi_scale=float(fa[1]),
        target_scale=ta,
    )

    with np.load(run_dir / "best_params.npz") as f:
        flat_arrays = [jnp.asarray(f[k]) for k in sorted(f.files, key=lambda s: int(s.split("_")[1]))]
    with open(run_dir / "tree_def.pkl", "rb") as f:
        tree_def = pickle.load(f)
    params = tree_def.unflatten(flat_arrays)

    @jax.jit
    def predict(p: dict, x: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
        return model.apply({"params": p}, x, d)

    # ---- Generate Stokes ----
    print(f"Generating Stokes waves (ichoi={args.ichoi}, depth={args.depth})...", flush=True)
    stokes = generate_stokes_samples(
        args.n0, args.a0, ichoi=args.ichoi, depth=args.depth,
    )
    n_samples = stokes["eta"].shape[0]
    print(f"  {n_samples} valid Stokes samples", flush=True)

    # ---- Evaluate ----
    print("Running inference...", flush=True)
    results = []
    pred_all = []
    for i in range(n_samples):
        eta_i = stokes["eta"][i:i+1]
        xi_i = stokes["xi"][i:i+1]
        gxi_true = stokes["gxi"][i]

        inp = normalize_features(eta_i, xi_i, ns)
        depth_arr = jnp.array([[log_depth]])
        pred_norm = predict(params, jnp.asarray(inp), depth_arr)
        pred_raw = denormalize_targets(np.asarray(jax.device_get(pred_norm)), ns).reshape(-1)
        pred_all.append(pred_raw)

        rel_l2 = float(np.linalg.norm(pred_raw - gxi_true) / (np.linalg.norm(gxi_true) + 1e-12))
        rel_l1 = float(np.sum(np.abs(pred_raw - gxi_true)) / (np.sum(np.abs(gxi_true)) + 1e-12))

        label = stokes["labels"][i]
        results.append({**label, "rel_l2": rel_l2, "rel_l1": rel_l1})

    predictions = np.stack(pred_all)[:, :, None]  # (N, nx, 1)

    # ---- Report ----
    rel_l2_all = np.array([r["rel_l2"] for r in results])
    rel_l1_all = np.array([r["rel_l1"] for r in results])

    print(f"\n{'n0':>4s} {'a0':>6s} {'eps':>6s} {'rel_l2':>8s} {'rel_l1':>8s}", flush=True)
    print("-" * 40, flush=True)
    for r in sorted(results, key=lambda d: (d["n0"], d["a0"])):
        print(f"{r['n0']:4d} {r['a0']:6.3f} {r['eps']:6.4f} {r['rel_l2']:8.4f} {r['rel_l1']:8.4f}", flush=True)

    tag = f"ichoi{args.ichoi}_h{int(args.depth)}"
    summary = {
        "ichoi": args.ichoi,
        "depth": args.depth,
        "n_samples": n_samples,
        "mean_rel_l2": float(np.mean(rel_l2_all)),
        "median_rel_l2": float(np.median(rel_l2_all)),
        "std_rel_l2": float(np.std(rel_l2_all)),
        "mean_rel_l1": float(np.mean(rel_l1_all)),
        "median_rel_l1": float(np.median(rel_l1_all)),
        "per_sample": results,
    }
    out_path = run_dir / f"eval_stokes_{tag}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSummary: mean_rel_l2={summary['mean_rel_l2']:.4f}, "
          f"median_rel_l2={summary['median_rel_l2']:.4f}", flush=True)
    print(f"Saved → {out_path}", flush=True)

    # ---- Plot ----
    from jax_training_util import plot_representative_samples

    plot_path = run_dir / f"eval_stokes_{tag}.png"
    plot_representative_samples(
        plot_path,
        f"Stokes ({tag}): mean L2={summary['mean_rel_l2']:.4f}",
        stokes["x"],
        stokes["eta"],
        stokes["xi"],
        stokes["gxi"][:, :, None],
        predictions,
        rel_l2_all,
        rel_l1_all,
        random_seed=42,
    )
    print(f"Plot → {plot_path}", flush=True)


if __name__ == "__main__":
    main()
