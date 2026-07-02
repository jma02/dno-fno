"""Wall-clock comparison: DNO series evaluation vs FNO forward pass.

Usage:
    uv run python playground/bench_dno_vs_fno.py --run_dir playground/runs/combined_fno --gpu
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from fno_jax.fno1d import FNO1d
from solver.solvers.dno_series_jax import build_grid, dno_series_eval


def bench(fn: callable, n_warmup: int = 5, n_iter: int = 100) -> dict[str, float]:
    """Time a JAX function after warmup. Returns dict with mean/std/min in ms."""
    for _ in range(n_warmup):
        out = fn()
        jax.block_until_ready(out)

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    return {"mean_ms": float(np.mean(arr)), "std_ms": float(np.std(arr)), "min_ms": float(np.min(arr))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DNO series vs FNO.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=164.0)
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--dno_order", type=int, default=6)
    parser.add_argument("--pad_factor", type=int, default=8)
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_iter", type=int, default=200)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    if not args.gpu:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    print(f"Platform: {jax.default_backend()}")
    print(f"Devices:  {jax.devices()}")

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ns = json.loads((run_dir / "norm_stats.json").read_text(encoding="utf-8"))

    # ---- Build FNO ----
    fa = np.asarray(ns["feature_absmax"], dtype=np.float32)
    ta = float(ns["target_absmax"]) if ns["target_absmax"] > 0 else 1.0
    model = FNO1d(
        modes=config["modes"], width=config["width"], n_blocks=config["n_blocks"],
        xi_scale=float(fa[1]), target_scale=ta,
    )
    with np.load(run_dir / "best_params.npz") as f:
        flat_arrays = [jnp.asarray(f[k]) for k in sorted(f.files, key=lambda s: int(s.split("_")[1]))]
    with open(run_dir / "tree_def.pkl", "rb") as f:
        tree_def = pickle.load(f)
    fno_params = tree_def.unflatten(flat_arrays)

    # ---- Build grid and test data ----
    _, k = build_grid(args.nx, args.length)
    log_depth = float(np.log(args.depth))

    absmax = np.asarray(ns["feature_absmax"], dtype=np.float32).reshape(1, 1, 2)
    absmax_j = jnp.where(jnp.asarray(absmax) > 0, jnp.asarray(absmax), 1.0)

    results = {}

    for bs_label, bs in [("single", 1), ("batch8", 8), ("batch64", 64)]:
        print(f"\n{'='*60}")
        print(f"Batch size = {bs}")
        print(f"{'='*60}")

        # Random test data
        rng = np.random.default_rng(42)
        eta_np = rng.normal(0, 0.1, (bs, args.nx)).astype(np.float32)
        xi_np = rng.normal(0, 0.3, (bs, args.nx)).astype(np.float32)

        eta_j = jnp.asarray(eta_np)
        xi_j = jnp.asarray(xi_np)

        # ---- DNO series (unbatched for bs=1, batched for bs>1) ----
        @jax.jit
        def dno_eval(eta: jnp.ndarray, xi: jnp.ndarray) -> jnp.ndarray:
            return dno_series_eval(eta, xi, k, args.depth, args.dno_order, pad_factor=args.pad_factor)

        print(f"\nDNO series (order={args.dno_order}, pad={args.pad_factor}x):")
        dno_times = bench(lambda: dno_eval(eta_j, xi_j), n_warmup=args.n_warmup, n_iter=args.n_iter)
        print(f"  mean = {dno_times['mean_ms']:.3f} ms, std = {dno_times['std_ms']:.3f} ms, min = {dno_times['min_ms']:.3f} ms")

        # ---- FNO forward pass ----
        inp = jnp.stack((eta_j, xi_j), axis=-1) / absmax_j  # (bs, nx, 2)
        depth_arr = jnp.full((bs, 1), log_depth)

        @jax.jit
        def fno_eval(x: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
            return model.apply({"params": fno_params}, x, d)

        print(f"\nFNO ({config['n_blocks']} blocks, {config['modes']} modes, {config['width']}W):")
        fno_times = bench(lambda: fno_eval(inp, depth_arr), n_warmup=args.n_warmup, n_iter=args.n_iter)
        print(f"  mean = {fno_times['mean_ms']:.3f} ms, std = {fno_times['std_ms']:.3f} ms, min = {fno_times['min_ms']:.3f} ms")

        speedup = dno_times["mean_ms"] / fno_times["mean_ms"]
        print(f"\n  Speedup: {speedup:.1f}x")

        results[bs_label] = {
            "batch_size": bs,
            "dno": dno_times,
            "fno": fno_times,
            "speedup": speedup,
        }

    # ---- Also time a single GL2-IF step with DNO vs FNO ----
    from solver.solvers import time_integrator as ti

    print(f"\n{'='*60}")
    print("Full GL2-IF step (single sample)")
    print(f"{'='*60}")

    solver_params = ti.make_solver_params(
        nx=args.nx, length=args.length, depth=args.depth, gravity=1.0,
        dno_order=args.dno_order, pad_factor=args.pad_factor,
        filter_fraction=2.0 / 3.0,
    )

    state = ti.State(eta=jnp.asarray(eta_np[0]), xi=jnp.asarray(xi_np[0]))

    @jax.jit
    def gl2_step_dno(s: ti.State) -> ti.State:
        return ti.gauss_legendre_2_if_step(s, 0.0, 0.2, solver_params, iterations=4)

    print("\nGL2-IF step with DNO series:")
    gl2_dno = bench(lambda: gl2_step_dno(state), n_warmup=args.n_warmup, n_iter=args.n_iter)
    print(f"  mean = {gl2_dno['mean_ms']:.3f} ms, std = {gl2_dno['std_ms']:.3f} ms")

    # GL2-IF step with FNO surrogate (import from rollout script)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rollout_tanaka import build_predict_gxi, rhs_nonlinear_if_surrogate, gl2_if_step_surrogate

    predict_gxi = build_predict_gxi(model, fno_params, ns, log_depth)

    @jax.jit
    def gl2_step_fno(s: ti.State) -> ti.State:
        return gl2_if_step_surrogate(s, 0.0, 0.2, solver_params, predict_gxi, iterations=4)

    print("\nGL2-IF step with FNO surrogate:")
    gl2_fno = bench(lambda: gl2_step_fno(state), n_warmup=args.n_warmup, n_iter=args.n_iter)
    print(f"  mean = {gl2_fno['mean_ms']:.3f} ms, std = {gl2_fno['std_ms']:.3f} ms")

    gl2_speedup = gl2_dno["mean_ms"] / gl2_fno["mean_ms"]
    print(f"\n  GL2 step speedup: {gl2_speedup:.1f}x")

    results["gl2_step"] = {
        "dno": gl2_dno,
        "fno": gl2_fno,
        "speedup": gl2_speedup,
    }

    # Save results
    results["platform"] = jax.default_backend()
    results["device"] = str(jax.devices()[0])
    results["nx"] = args.nx
    results["dno_order"] = args.dno_order
    results["pad_factor"] = args.pad_factor
    results["n_iter"] = args.n_iter

    out_path = run_dir / "bench_dno_vs_fno.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
