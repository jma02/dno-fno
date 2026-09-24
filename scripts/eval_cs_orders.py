"""Compare CS orders 1--6 on the current 32-per-family held-out rollout panel.

Example (one process per GPU)::

    CUDA_VISIBLE_DEVICES=0 uv run scripts/eval_cs_orders.py --families tanaka stokes
    CUDA_VISIBLE_DEVICES=1 uv run scripts/eval_cs_orders.py --families benjamin_feir jonswap_tma

Each family produces one compact NPZ with per-case errors and synchronized
full-rollout timings. Compilation, short warm-up, and host transfers are excluded.
Order M includes G_0 through G_M. All arithmetic is FP64. The default is one
timed run per order; use --repeats to measure timing variability.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from solver.evals.eval_suite import FAMILY_CONFIGS  # noqa: E402
from solver.solvers import time_integrator as ti  # noqa: E402

jax.config.update("jax_enable_x64", True)
RUN = ROOT / "outputs/c27_tanaka_hard128_full_equal_local_20260918"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=FAMILY_CONFIGS, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cs_order_sweep_20260923")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--frames", type=int, default=251, help="Use 5 for a short numerical smoke test.")
    args = parser.parse_args()
    if args.repeats < 1 or not 2 <= args.frames <= 251:
        parser.error("repeats must be positive and frames must be between 2 and 251")
    args.output.mkdir(parents=True, exist_ok=True)
    device = jax.devices()[0]
    fields = ("eta", "xi", "gxi")
    for family in args.families:
        cfg = FAMILY_CONFIGS[family]
        gpu = 0 if family in ("stokes", "tanaka") else 1
        source = RUN / f"eval_best_current_test_stratified_n32_fp32net_fp64solver_gpu{gpu}"
        with np.load(source / f"{family}_trajs.npz") as archive:
            times = jnp.asarray(archive["times"][:args.frames])
            depths = np.asarray(archive["depths"])
            simulation_ids = np.asarray(archive["simulation_ids"])
            initial = ti.State(
                eta=jnp.asarray(archive["truth_eta"][0]),
                xi=jnp.asarray(archive["truth_xi"][0]),
            )
        nx = initial.eta.shape[-1]
        data: dict[str, list[np.ndarray]] = {
            key: [] for key in (
                "orders", "timings_s", "finite", "minimum_depth", "successful",
                *(f"{field}_{metric}" for field in fields for metric in ("rel_l2", "trajectory_error")),
            )
        }
        metadata = {
            "family": family, "reference_order": 6, "nx": nx,
            "length": 2 * np.pi, "gravity": 1.0, "padding": 8,
            "cutoff_mode": 128, "internal_dt": cfg.dt / cfg.substeps,
            "substeps": cfg.substeps, "gl2_iterations": 4, "relaxation": 1.0,
            "precision": "float64", "batch_size": len(depths),
            "device": device.device_kind, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "timing": "full rollout including saved Gxi; excludes compilation, short warm-up, host transfer and metrics",
            "warmup_intervals": min(4, args.frames - 1),
            "repeats": args.repeats,
        }
        reference: dict[str, np.ndarray] = {}
        for order in (6, 1, 2, 3, 4, 5):
            params = ti.make_solver_params(
                nx, 2 * np.pi, jnp.asarray(depths)[:, None],
                dno_order=order, pad_factor=8, filter_fraction=cfg.filter_fraction,
            )

            def integrate(state: ti.State, saved_times: jax.Array) -> dict[str, jax.Array]:
                return ti.rollout(
                    state, saved_times, params, save_gxi=True,
                    substeps_per_interval=cfg.substeps, method="gl2_if",
                    implicit_iterations=4, zero_mean_xi=True,
                )

            runner = jax.jit(integrate)
            compiled = runner.lower(initial, times).compile()
            result = jax.block_until_ready(runner(initial, times[:5]))
            print(f"{family} M={order}: compiled and warmed; T={float(times[-1]):g}, batch={len(depths)}", flush=True)
            durations = []
            for repeat in range(args.repeats):
                started = time.perf_counter()
                result = jax.block_until_ready(compiled(initial, times))
                durations.append(time.perf_counter() - started)
                print(f"{family} M={order}: repeat {repeat + 1}, {durations[-1]:.3f}s", flush=True)
            values = {field: np.asarray(result[field]) for field in fields}
            values["xi"] = values["xi"] - values["xi"].mean(axis=-1, keepdims=True)
            if order == 6:
                reference = values
            finite = np.logical_and.reduce([np.isfinite(values[field]).all(axis=(0, 2)) for field in fields])
            minimum_depth = (values["eta"] + depths[None, :, None]).min(axis=(0, 2))
            successful = finite & (minimum_depth > 0)
            for field in fields:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    difference = values[field] - reference[field]
                    errors = np.linalg.norm(difference, axis=-1) / np.maximum(
                        np.linalg.norm(reference[field], axis=-1), 1e-30,
                    )
                    trajectory_error = np.sqrt(
                        np.sum(difference**2, axis=(0, 2))
                        / np.maximum(np.sum(reference[field]**2, axis=(0, 2)), 1e-30)
                    )
                data[f"{field}_rel_l2"].append(errors)
                data[f"{field}_trajectory_error"].append(trajectory_error)
                successful &= np.isfinite(errors).all(axis=0) & np.isfinite(trajectory_error)
            data["orders"].append(np.asarray(order))
            data["timings_s"].append(np.asarray(durations))
            data["finite"].append(finite)
            data["minimum_depth"].append(minimum_depth)
            data["successful"].append(successful)
            np.savez_compressed(
                args.output / f"{family}.npz", allow_pickle=False,
                **{key: np.stack(value) for key, value in data.items()},
                times=np.asarray(times), depths=depths, simulation_ids=simulation_ids,
                metadata=np.asarray(json.dumps(metadata)),
            )
            terminal = data["eta_rel_l2"][-1][-1, successful]
            quantiles = np.quantile(terminal, [0.5, 0.95]) if terminal.size else np.full(2, np.nan)
            print(f"{family} M={order}: successful={successful.sum()}/{len(depths)}, eta median/p95={quantiles}", flush=True)
        jax.clear_caches()
