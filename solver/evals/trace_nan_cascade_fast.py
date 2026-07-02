"""NaN-cascade trace as a thin shell over rollout_surrogate(substeps=1).

Calls the production integrator with substeps=1 over a fine outer-dt grid (one
outer step = one substep), then recomputes gxi/RHS/band-energy diagnostics from
the saved (eta, xi) trajectory in a post-processing pass. Numerical equivalence
with eval_suite's rollout is guaranteed by construction — there is no duplicated
integration loop to drift from production.

Output: a NPZ with per-substep band energies for each tracked field, plus the
substep index where NaN first appears. Same schema as previous versions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from solver.evals.model_rollout import build_predict_gxi, load_run, rollout_surrogate
from solver.solvers import time_integrator as ti


KBAND_EDGES: tuple[int, ...] = (0, 32, 64, 128, 256, 513)
FIELD_NAMES: tuple[str, ...] = ("eta", "xi", "gxi", "eta_t", "xi_t", "etax_xix", "num", "1+etax^2")


def band_energies(field: jnp.ndarray) -> jnp.ndarray:
    """field shape (..., nx) -> (..., 5) sqrt(sum |X̂_k|²) per k-band."""
    fh = jnp.fft.rfft(field, axis=-1)
    n_half = fh.shape[-1]
    edges = [min(e, n_half) for e in KBAND_EDGES]
    bands = [jnp.sqrt(jnp.sum(jnp.abs(fh[..., edges[i]:edges[i + 1]]) ** 2, axis=-1))
             for i in range(len(edges) - 1)]
    return jnp.stack(bands, axis=-1)


def find_case(npz_path: Path, case_id: int) -> tuple[np.ndarray, np.ndarray, float, dict]:
    with np.load(npz_path, mmap_mode="r") as d:
        meta = json.loads(bytes(d["meta.json"]).decode())
        ids = np.asarray(d["case_id_batch_0000"])
        depths = np.asarray(d["depth_batch_0000"])
        eta = np.asarray(d["eta_batch_0000"])
        xi = np.asarray(d["xi_batch_0000"])
    rows = np.nonzero(ids == case_id)[0]
    if rows.size == 0:
        raise SystemExit(f"case_id {case_id} not found in {npz_path}")
    row0 = int(rows[0])
    return eta[row0].astype(np.float64), xi[row0].astype(np.float64), float(depths[row0]), meta


def compute_diagnostics(
    eta_traj: jnp.ndarray,
    xi_traj: jnp.ndarray,
    gxi_traj: jnp.ndarray,
    params: ti.SolverParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (T, 8, 5) band-energy stats and (T, 8) max-abs from saved frames.

    All eight tracked fields are derived without re-calling the model — `gxi` is
    read straight from `rollout_surrogate`'s saved output, everything else is a
    cheap derivative of (eta, xi, gxi). No batching needed.
    """
    @jax.jit
    def _diag(eta_t: jnp.ndarray, xi_t: jnp.ndarray, gxi: jnp.ndarray):
        eta_x = jax.vmap(lambda e: ti.spectral_dx(e, params.k))(eta_t)
        xi_x = jax.vmap(lambda x: ti.spectral_dx(x, params.k))(xi_t)
        etax_xix = eta_x * xi_x
        numerator = gxi + etax_xix
        denom = 1.0 + eta_x ** 2
        linear_gxi = jax.vmap(lambda x: ti.linear_dno_action(x, params.g0))(xi_t)
        eta_dot = gxi - linear_gxi
        xi_dot = -0.5 * xi_x ** 2 + 0.5 * numerator ** 2 / denom

        fields = jnp.stack([eta_t, xi_t, gxi, eta_dot, xi_dot, etax_xix, numerator, denom], axis=1)  # (T, 8, nx)
        bands = jax.vmap(jax.vmap(band_energies))(fields)  # (T, 8, 5)
        maxabs = jnp.max(jnp.abs(fields), axis=-1)         # (T, 8)
        return bands, maxabs

    bands, maxabs = _diag(eta_traj, xi_traj, gxi_traj)
    return np.asarray(bands), np.asarray(maxabs)


def print_pre_nan_window(stats: np.ndarray, maxabs: np.ndarray, nan_substep: int, window: int = 20) -> None:
    s_pre = max(0, nan_substep - window)
    n = min(nan_substep + 1, stats.shape[0])
    print(f"\n=== last {n - s_pre} substeps before/at NaN ===", flush=True)
    for s in range(s_pre, n):
        row = stats[s]
        mx = maxabs[s]
        bands_g = " ".join(f"k[{KBAND_EDGES[b]}-{KBAND_EDGES[b+1]}]={row[2,b]:.2e}" for b in range(5))
        print(f" s={s:>5} gxi:  {bands_g}  max|gxi|={mx[2]:.3e}", flush=True)
        bands_d = " ".join(f"k[{KBAND_EDGES[b]}-{KBAND_EDGES[b+1]}]={row[7,b]:.2e}" for b in range(5))
        print(f"        1+etax^2:  {bands_d}  max={mx[7]:.3e}", flush=True)
        bands_n = " ".join(f"k[{KBAND_EDGES[b]}-{KBAND_EDGES[b+1]}]={row[6,b]:.2e}" for b in range(5))
        print(f"        num:       {bands_n}  max={mx[6]:.3e}", flush=True)
        bands_xt = " ".join(f"k[{KBAND_EDGES[b]}-{KBAND_EDGES[b+1]}]={row[4,b]:.2e}" for b in range(5))
        print(f"        xi_t:      {bands_xt}  max={mx[4]:.3e}", flush=True)
        print("        max-abs: " + " ".join(f"{name}={mx[i]:.2e}" for i, name in enumerate(FIELD_NAMES)), flush=True)
        print(flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--case_id", type=int, default=5)
    parser.add_argument("--substeps_per_outer", type=int, default=80,
                        help="Production substeps per outer step. Combined with --outer_dt this "
                             "fixes the inner_dt the trace integrates with.")
    parser.add_argument("--outer_dt", type=float, default=0.8,
                        help="eval_suite RegimeConfig.dt for tanaka_g0/g1 is 0.8. "
                             "The npz's meta['dt'] is the dataset SAVE cadence (0.08), not the "
                             "rollout outer step — do not pass that here.")
    parser.add_argument("--impl", type=int, default=4)
    parser.add_argument("--filter_fraction", type=float, default=0.25)
    parser.add_argument("--max_substeps", type=int, default=20000)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--out_npz", default=None)
    args = parser.parse_args()

    loaded = load_run(args.run_dir, checkpoint=args.checkpoint)
    eta0_np, xi0_np, depth, _ = find_case(Path(args.npz), args.case_id)
    nx = int(eta0_np.shape[-1])
    inner_dt = float(args.outer_dt) / float(args.substeps_per_outer)
    print(f"# run_dir   : {args.run_dir}", flush=True)
    print(f"# epoch     : {loaded.epoch}  cs_tie_xi_out_mult={loaded.config.get('cs_tie_xi_out_mult')} cs_mult_hidden={loaded.config.get('cs_mult_hidden')}", flush=True)
    print(f"# npz       : {args.npz}  case_id={args.case_id}  depth={depth:.4f}", flush=True)
    print(f"# grid      : nx={nx}  outer_dt={args.outer_dt:.4f}  inner_dt={inner_dt:.6f}  substeps/outer={args.substeps_per_outer}  impl={args.impl}", flush=True)
    print(f"# trace     : rollout_surrogate(substeps=1) × {args.max_substeps} outer steps, post-process diagnostics", flush=True)

    params = ti.make_solver_params(
        nx=nx, length=2.0 * np.pi, depth=depth, gravity=1.0,
        dno_order=6, pad_factor=8, filter_fraction=args.filter_fraction,
    )
    params = ti.cast_solver_params_dtype(params, jnp.float64)
    predict = build_predict_gxi(loaded, depth)

    initial = ti.State(
        eta=jnp.asarray(eta0_np, dtype=jnp.float64),
        xi=jnp.asarray(xi0_np, dtype=jnp.float64),
    )
    times = jnp.arange(args.max_substeps + 1, dtype=jnp.float64) * inner_dt

    print(f"# rolling out {args.max_substeps} substeps...", flush=True)
    out = rollout_surrogate(
        initial, times, params, predict,
        substeps=1, zero_mean_xi=True, gl2_iterations=args.impl,
    )
    eta_traj = out["eta"]   # (T, nx) jnp
    xi_traj = out["xi"]
    gxi_traj = out["gxi"]

    print("# computing diagnostics...", flush=True)
    stats_arr, maxabs_arr = compute_diagnostics(eta_traj, xi_traj, gxi_traj, params)
    # stats_arr is (T, 8, 5), maxabs_arr is (T, 8). T = max_substeps + 1, including t=0 frame.

    bad_any = (~np.isfinite(maxabs_arr)).any(axis=1)
    if bad_any.any():
        nan_substep = int(np.argmax(bad_any))
        first_field = int(np.argmax(~np.isfinite(maxabs_arr[nan_substep])))
        t_nan = nan_substep * inner_dt
        print(f"!! NaN/Inf at substep {nan_substep} (t={t_nan:.3f}), first field={FIELD_NAMES[first_field]}", flush=True)
        print_pre_nan_window(stats_arr, maxabs_arr, nan_substep)
    else:
        nan_substep = -1
        print(f"# completed {args.max_substeps} substeps without NaN (t={args.max_substeps * inner_dt:.4f})", flush=True)

    out_npz = args.out_npz or (Path(args.run_dir) / f"trace_cid{args.case_id}.npz")
    np.savez_compressed(
        out_npz,
        stats=stats_arr, maxabs=maxabs_arr,
        nan_substep=np.asarray(nan_substep, dtype=np.int64),
        kband_edges=np.asarray(KBAND_EDGES, dtype=np.int64),
        field_names=np.asarray(FIELD_NAMES),
        inner_dt=np.asarray(inner_dt),
    )
    print(f"# saved arrays to {out_npz}", flush=True)


if __name__ == "__main__":
    main()
