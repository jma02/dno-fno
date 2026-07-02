"""Diagnostic: is the v5 tanaka rollout failure integrator-amplification or model error?

For the 4 ICs that fail in eval_suite_f64h on v5
(g0 IC5 h=0.276, g0 IC11 h=0.234, g1 IC6 h=0.293, g1 IC11 h=0.011), roll the
model forward at three inner-dt settings: 0.01 (baseline), 0.005, 0.0025
(i.e. substeps = 80, 160, 320 against outer dt = 0.8). Same f64, same filter,
same GL2 implicit iterations as eval_suite_f64h.

Question: does the rollout survive longer (or to lower final error) when
inner dt shrinks?
  - Yes => GL2 is amplifying tiny per-step errors. Failure is integrator-
    limited, not a model-knowledge gap.
  - No  => model's Gxi on those states is genuinely wrong. No amount of
    integrator refinement will fix it.

Output: prints a table (IC, substeps, nan_time, max_finite_t, rel_l2_eta@end);
also dumps the same to JSON for downstream notes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver.evals.model_rollout import (  # noqa: E402
    build_predict_gxi_with_depth,
    load_run,
    rollout_surrogate,
)
from solver.solvers import time_integrator as ti  # noqa: E402
from solver.solvers.dno_series_jax import build_grid, dno_series_eval  # noqa: E402


# (regime, case_id, expected_nan_time) — case_ids match those stored in
# outputs/cs_dno_w512b8_l256_v5_*/eval_suite_f64h/tanaka_*_summary.json.
FAILING_ICS = [
    ("tanaka_g0", 5,       42.0),    # h=0.276
    ("tanaka_g0", 11,      91.0),    # h=0.234
    ("tanaka_g1", 1000006, 97.0),    # h=0.293
    ("tanaka_g1", 1000011, None),    # h=0.011 — diverged (rel~1.43), not NaN
]

DATA_DIR = REPO_ROOT / "data"


def load_ic(regime: str, case_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Match the eval_suite loader: batch_0000 is a flat (rows, nx) table covering many cases.

    Rows are grouped by case_id; within a case_id, rows are in time order.
    """
    path = DATA_DIR / f"tanaka_2_adaptive_{regime.replace('tanaka_', '')}.npz"
    with np.load(path, mmap_mode="r") as d:
        case_id_arr = np.asarray(d["case_id_batch_0000"])
        rows = np.nonzero(case_id_arr == int(case_id))[0]
        if rows.size == 0:
            raise KeyError(f"case_id={case_id} not found in {path.name}")
        eta = np.asarray(d["eta_batch_0000"][rows], dtype=np.float64)
        xi  = np.asarray(d["xi_batch_0000"][rows],  dtype=np.float64)
        t   = np.asarray(d["time_batch_0000"][rows], dtype=np.float64)
        depth = float(np.asarray(d["depth_batch_0000"][rows[0]]))
    order = np.argsort(t)
    return eta[order[0]], xi[order[0]], eta[order], depth, t[order]


def build_rollout(loaded, nx: int, length: float):
    _, k_np = build_grid(nx, length)
    k_grid = jnp.asarray(k_np, dtype=jnp.float64)
    predict_gxi_with_depth = build_predict_gxi_with_depth(loaded)

    def rollout(eta0_np, xi0_np, depth, log_h, outer_times, substeps, gl2_iter):
        sp = ti.SolverParams(
            nx=nx, length=length, depth=depth, gravity=1.0,
            dno_order=6, pad_factor=8, filter_fraction=0.25,
            k=k_grid, g0=ti.make_linear_dno_symbol(k_grid, depth),
        )
        state0 = ti.State(
            eta=jnp.asarray(eta0_np, dtype=jnp.float64),
            xi=jnp.asarray(xi0_np, dtype=jnp.float64),
        )

        def predict(eta, xi):
            gxi = predict_gxi_with_depth(
                eta.astype(jnp.float32), xi.astype(jnp.float32), log_h
            ).astype(jnp.float64)
            return ti.apply_lowpass(gxi, k_grid, 0.25)

        out = rollout_surrogate(
            state0, jnp.asarray(outer_times, dtype=jnp.float64),
            sp, predict, substeps=substeps, gl2_iterations=gl2_iter,
        )
        return np.asarray(out["eta"]), np.asarray(out["xi"])

    return rollout


def first_nan_time(eta_traj: np.ndarray, outer_times: np.ndarray) -> tuple[float | None, int]:
    """Index of first frame with any non-finite eta, and that time. Returns (None, -1) if all finite."""
    bad = np.any(~np.isfinite(eta_traj), axis=1)
    if not bad.any():
        return None, -1
    idx = int(np.argmax(bad))
    return float(outer_times[idx]), idx


def rel_l2_eta(pred: np.ndarray, truth: np.ndarray) -> float:
    """Relative L2 of eta at a single time slice (1D arrays)."""
    num = float(np.linalg.norm(pred - truth))
    den = float(np.linalg.norm(truth))
    return num / max(den, 1e-30)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", default="outputs/cs_dno_w512b8_l256_v5_2gpu_20260615_141717")
    p.add_argument("--substeps", type=int, nargs="+", default=[80, 160, 320])
    p.add_argument("--gl2_iters", type=int, default=4)
    p.add_argument("--dt", type=float, default=0.8, help="outer save dt")
    p.add_argument("--tmax", type=float, default=200.0)
    p.add_argument("--nx", type=int, default=1024)
    p.add_argument("--length", type=float, default=2.0 * float(np.pi))
    p.add_argument("--output_json", default="notes/diag_tanaka_substeps.json")
    args = p.parse_args()

    print(f"loading run: {args.run_dir}", flush=True)
    loaded = load_run(args.run_dir)
    rollout = build_rollout(loaded, args.nx, args.length)
    outer_times = np.arange(0.0, args.tmax + 0.5 * args.dt, args.dt, dtype=np.float64)
    print(f"outer dt = {args.dt}, n_t = {outer_times.size}, substeps grid = {args.substeps}", flush=True)

    results: list[dict] = []
    for regime, cid, prior_nan in FAILING_ICS:
        try:
            eta0, xi0, truth_eta, h, truth_t = load_ic(regime, cid)
        except KeyError as e:
            print(f"  ! skipping {regime} IC{cid}: missing key {e}", flush=True)
            continue
        log_h = float(np.log(max(h, 1e-12)))
        print(f"\n=== {regime} IC{cid}  h={h:.4f}  (prior NaN @t={prior_nan})", flush=True)
        for substeps in args.substeps:
            inner_dt = args.dt / substeps
            t0 = _time.perf_counter()
            pred_eta, _pred_xi = rollout(
                eta0, xi0, h, log_h, outer_times, substeps, args.gl2_iters,
            )
            wall = _time.perf_counter() - t0
            t_nan, idx_nan = first_nan_time(pred_eta, outer_times)
            # Final-time relative L2 vs stored truth (if available); align indices by closest t.
            if t_nan is None:
                # rollout went all the way; compare to truth at the latest common time
                t_end = float(outer_times[-1])
                tj = int(np.argmin(np.abs(truth_t - t_end)))
                if abs(truth_t[tj] - t_end) < 0.5 * args.dt:
                    rel = rel_l2_eta(pred_eta[-1], truth_eta[tj])
                else:
                    rel = float("nan")
            else:
                # compare just before NaN
                idx_last = max(idx_nan - 1, 0)
                t_last = float(outer_times[idx_last])
                tj = int(np.argmin(np.abs(truth_t - t_last)))
                rel = rel_l2_eta(pred_eta[idx_last], truth_eta[tj]) if abs(truth_t[tj] - t_last) < 0.5 * args.dt else float("nan")
            print(
                f"  substeps={substeps:4d}  inner_dt={inner_dt:7.4f}  "
                f"nan_t={('—' if t_nan is None else f'{t_nan:6.1f}'):>7}  "
                f"max_finite_t={(args.tmax if t_nan is None else (outer_times[max(idx_nan-1,0)])):>6.1f}  "
                f"rel_l2_eta@last={rel:7.3e}  wall={wall:.1f}s",
                flush=True,
            )
            results.append({
                "regime": regime, "case_id": int(cid), "depth": float(h),
                "substeps": int(substeps), "inner_dt": float(inner_dt),
                "nan_time": (None if t_nan is None else float(t_nan)),
                "max_finite_t": float(args.tmax if t_nan is None else outer_times[max(idx_nan-1,0)]),
                "rel_l2_eta_at_last_finite": float(rel),
                "wall_s": float(wall),
            })

    out_path = REPO_ROOT / args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "run_dir": str(args.run_dir),
        "outer_dt": float(args.dt),
        "tmax": float(args.tmax),
        "gl2_iters": int(args.gl2_iters),
        "results": results,
    }, indent=2))
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
