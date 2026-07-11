"""Probe the actual GL2 Picard stage-update residual on healthy tanaka_g0.

Runs one _gl2_if_step_with_residual against the C1 checkpoint on tanaka_g0 ICs
0, 1, 5 (mix of healthy and known-Mode-1). Prints per-sample r_final and r_prev
so we can compare to my assumed 1e-2 divergence threshold.

If healthy r_final is above 1e-2, the threshold is wrong. If it's below, some
other bug is triggering the containment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solver.evals.eval_suite import (  # noqa: E402
    REGISTRY,
    RegimeConfig,
    build_ics_and_truth_targets,
)
from solver.evals.model_rollout import (  # noqa: E402
    _gl2_if_step_with_residual,
    build_predict_gxi_batched,
    load_run,
)
from solver.solvers import time_integrator as ti  # noqa: E402
from solver.solvers.dno_series_jax import build_grid, make_linear_dno_symbol  # noqa: E402


def main() -> None:
    jax.config.update("jax_enable_x64", True)

    run_dir = Path("/home/johnma/dno-fno/outputs/c1_stage_match_from_v85b_20260704_210952")
    print(f"loading run: {run_dir}", flush=True)
    loaded = load_run(run_dir, checkpoint="best")
    predict_batched = build_predict_gxi_batched(loaded)

    regime = "tanaka_g0"
    base = REGISTRY[regime]
    cfg = RegimeConfig(
        name=base.name, source=base.source,
        dt=base.dt, tmax=base.dt,   # just one save-interval worth
        n_ics=8, substeps=base.substeps, implicit_iters=base.implicit_iters,
        filter_fraction=base.filter_fraction, truth_kind=base.truth_kind,
    )
    ics, dt, tmax, _, _ = build_ics_and_truth_targets(cfg)
    print(f"loaded {len(ics)} tanaka_g0 ICs at t=0; base dt_outer={dt} substeps={cfg.substeps}", flush=True)

    nx = 1024
    length = 2.0 * float(np.pi)
    dtype = jnp.float64

    eta0_np = np.stack([ic.eta for ic in ics], axis=0).astype(np.float64)
    xi0_np = np.stack([ic.xi for ic in ics], axis=0).astype(np.float64)
    depths_np = np.asarray([ic.depth for ic in ics], dtype=np.float64)
    log_depth = jnp.asarray(np.log(np.maximum(depths_np, 1e-12)), dtype=jnp.float32)

    _, k_grid = build_grid(nx, length)
    k_grid = jnp.asarray(k_grid, dtype=dtype)
    depth_2d = jnp.asarray(depths_np, dtype=dtype)[:, None]
    g0 = make_linear_dno_symbol(k_grid, depth_2d)
    sp = ti.SolverParams(
        nx=nx, length=length, depth=depth_2d, gravity=1.0,
        dno_order=6, pad_factor=8, filter_fraction=0.25, k=k_grid, g0=g0,
    )

    def predict(eta_in: jnp.ndarray, xi_in: jnp.ndarray) -> jnp.ndarray:
        return predict_batched(
            eta_in.astype(jnp.float32), xi_in.astype(jnp.float32), log_depth,
        ).astype(dtype)

    state0 = ti.State(
        eta=jnp.asarray(eta0_np, dtype=dtype),
        xi=jnp.asarray(xi0_np, dtype=dtype),
    )
    state0 = ti.project_zero_mean_xi(state0)

    dt_sub = dt / cfg.substeps
    print(f"dt_substep = {dt_sub}")

    @jax.jit
    def one_step(state):
        return _gl2_if_step_with_residual(
            state, jnp.asarray(0.0, dtype=dtype), jnp.asarray(dt_sub, dtype=dtype),
            sp, predict,
            iterations=4,
            filter_shape="hard", houli_a=36.0, houli_m=36.0,
            cascade_gate_enabled=False,
            cascade_k_cut=32.0, cascade_r_threshold=1e-3, cascade_sharpness=10.0,
            cascade_houli_a=0.69, cascade_houli_m=4.0, cascade_k_eff=128.0,
            cascade_filter_xi=True,
        )

    print("Running one substep of _gl2_if_step_with_residual on 8 tanaka_g0 ICs at t=0...", flush=True)
    new_state, r_final, r_prev = one_step(state0)
    r_final_np = np.asarray(r_final)
    r_prev_np = np.asarray(r_prev)
    print("per-sample results:")
    print("  IC  depth        r_final          r_prev           r_final<1e-2   r_final<1e-6")
    for i in range(len(ics)):
        rf = float(r_final_np[i])
        rp = float(r_prev_np[i])
        print(f"  {i:2d}  {depths_np[i]:.4f}  {rf:.6e}  {rp:.6e}   {rf < 1e-2!s:5s}          {rf < 1e-6!s}")

    print(f"\naggregates: r_final min={r_final_np.min():.3e} max={r_final_np.max():.3e} median={np.median(r_final_np):.3e}")
    print(f"            r_prev  min={r_prev_np.min():.3e} max={r_prev_np.max():.3e} median={np.median(r_prev_np):.3e}")


if __name__ == "__main__":
    main()
