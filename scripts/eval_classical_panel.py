"""32-case classical CS-order terminal eta errors against a cached M6 panel reference (FP64, GL2, pad 8).

Same rollout as scripts/eval_cs_orders.py, but initial states, depths and the M6 truth come from a saved
reference archive (REFERENCE/<family>_trajs.npz) instead of recomputing M6. One JSON per family; orders
already present in it are skipped. The median is over cases that stay finite with positive depth.

    CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/eval_classical_panel.py --families stokes tanaka \
        --output outputs/hero37_classical_panel_20261027
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

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
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", type=Path, default=Path("/home/johnma/dno-fno/outputs/shared_samzheng_20261026/reference"))
parser.add_argument("--families", nargs="+", required=True)
parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4, 5])
parser.add_argument("--frames", type=int, default=251)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
for family in args.families:
    cfg = FAMILY_CONFIGS[family]
    with np.load(args.reference / f"{family}_trajs.npz") as saved:
        times = jnp.asarray(saved["times"][:args.frames])
        depths = np.asarray(saved["depths"])
        truth = np.asarray(saved["truth_eta"][args.frames - 1])
        initial = ti.State(eta=jnp.asarray(saved["truth_eta"][0]), xi=jnp.asarray(saved["truth_xi"][0]))
    out = args.output / f"{family}.json"
    record = json.loads(out.read_text()) if out.exists() else {"family": family, "frames": args.frames, "orders": {}}
    for order in args.orders:
        if str(order) in record["orders"]:
            continue
        params = ti.make_solver_params(1024, 2 * np.pi, jnp.asarray(depths)[:, None], dno_order=order,
                                       pad_factor=8, filter_fraction=cfg.filter_fraction)
        runner = jax.jit(lambda state, saved_times, p=params: ti.rollout(
            state, saved_times, p, save_gxi=False, substeps_per_interval=cfg.substeps, method="gl2_if",
            implicit_iterations=4, zero_mean_xi=True)["eta"][-1])
        started = time.perf_counter()
        eta = np.asarray(jax.block_until_ready(runner(initial, times)))
        finite = np.isfinite(eta).all(axis=-1) & ((eta + depths[:, None]).min(axis=-1) > 0)
        with np.errstate(invalid="ignore", over="ignore"):
            errors = np.linalg.norm(eta - truth, axis=-1) / np.linalg.norm(truth, axis=-1)
        record["orders"][str(order)] = {
            "terminal_eta_errors": [float(e) if ok else None for e, ok in zip(errors, finite)],
            "finite": int(finite.sum()),
            "median_finite": float(np.median(errors[finite])) if finite.any() else None,
            "wall_s": time.perf_counter() - started,
        }
        out.write_text(json.dumps(record, indent=1))
        print(f"{family} M{order}: finite {finite.sum()}/32 median {record['orders'][str(order)]['median_finite']:.4e} "
              f"wall {record['orders'][str(order)]['wall_s']:.1f}s", flush=True)
