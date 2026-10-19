"""CPU cost of producing G0..G3 fields for an encoder-decoder, not four solver calls."""
from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from solver.solvers.dno_series_jax import dno_series_eval  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "outputs/paper_dataset_full_equal_tanaka_hard128_20260918/arrays")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 16, 256])
    parser.add_argument("--pads", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    fields = {name: np.load(args.dataset / f"{name}.npy", mmap_mode="r")
              for name in ("eta", "xi", "depth", "family_id", "dataset_split", "x")}
    rng = np.random.default_rng(42)
    count = max(args.batches)
    training = fields["dataset_split"] == "train"
    groups = [rng.choice(np.flatnonzero((fields["family_id"] == family) & training), count, replace=False)
              for family in np.unique(fields["family_id"])]
    rows = np.stack(groups, axis=1).reshape(-1)[:count]
    eta_all = np.asarray(fields["eta"][rows], dtype=np.float64)
    xi_all = np.asarray(fields["xi"][rows], dtype=np.float64)
    xi_all -= xi_all.mean(axis=-1, keepdims=True)
    depth_all = np.asarray(fields["depth"][rows, None], dtype=np.float64)
    nx = eta_all.shape[-1]
    length = float((fields["x"][1] - fields["x"][0]) * nx)
    k = jnp.arange(nx // 2 + 1, dtype=jnp.float64) * (2 * np.pi / length)
    report = {
        "device": str(jax.devices()[0]), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "nx": nx, "order": 3, "precision": "float64", "repeats": args.repeats,
        "dataset": str(args.dataset), "rows": rows.tolist(),
        "scope": "CPU-resident physical inputs to four physical output fields; no GPU transfer or decoder",
        "results": [],
    }
    reference_fn = jax.jit(partial(dno_series_eval, order=3, pad_factor=8, return_terms=True))
    for batch in args.batches:
        inputs = tuple(map(jnp.asarray, (eta_all[:batch], xi_all[:batch], depth_all[:batch])))
        eta, xi, depth = inputs
        reference = np.asarray(reference_fn(eta, xi, k, depth).block_until_ready())
        reference_hat = np.fft.rfft(reference, axis=-1)[..., :129]
        for pad in args.pads:
            started = perf_counter()
            compiled = jax.jit(partial(dno_series_eval, order=3, pad_factor=pad, return_terms=True)).lower(eta, xi, k, depth).compile()
            compile_seconds = perf_counter() - started
            actual = np.asarray(compiled(eta, xi, k, depth).block_until_ready())
            expected_sum = np.asarray(jax.jit(partial(dno_series_eval, order=3, pad_factor=pad))(eta, xi, k, depth))
            np.testing.assert_allclose(actual.sum(axis=-2), expected_sum, atol=2e-12, rtol=2e-12)
            if not np.isfinite(actual).all():
                raise FloatingPointError("Non-finite CS feature fields")
            actual_hat = np.fft.rfft(actual, axis=-1)[..., :129]
            relative = np.linalg.norm(actual_hat - reference_hat, axis=-1) / np.maximum(np.linalg.norm(reference_hat, axis=-1), 1e-30)
            full_relative = np.linalg.norm(actual - reference, axis=-1) / np.maximum(np.linalg.norm(reference, axis=-1), 1e-30)
            for _ in range(5):
                compiled(eta, xi, k, depth).block_until_ready()
            seconds = []
            for _ in range(args.repeats):
                started = perf_counter()
                compiled(eta, xi, k, depth).block_until_ready()
                seconds.append(perf_counter() - started)
            row = {
                "batch": batch, "pad": pad, "compile_seconds": compile_seconds,
                "median_ms": float(np.median(seconds) * 1000),
                "p10_p90_ms": (np.percentile(seconds, [10, 90]) * 1000).tolist(),
                "microseconds_per_row": float(np.median(seconds) * 1e6 / batch),
                "max_retained_relative_by_term": relative.max(axis=0).tolist(),
                "max_full_relative_by_term": full_relative.max(axis=0).tolist(), "seconds": seconds,
            }
            report["results"].append(row)
            print(json.dumps({key: value for key, value in row.items() if key != "seconds"}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
