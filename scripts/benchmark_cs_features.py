"""Measure CS feature generation, optionally overlapping CPU terms and GPU encoding."""
from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import cast

os.environ["JAX_PLATFORMS"] = "cuda,cpu" if "--overlap" in sys.argv else "cpu"
if "--overlap" not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dno-matplotlib")
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
    parser.add_argument("--overlap", action="store_true", help="Compare all-GPU, serial CPU/GPU, and concurrent CPU/GPU forward passes.")
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
    if args.overlap:
        from concurrent.futures import ThreadPoolExecutor

        sys.path.insert(0, str(ROOT / "models/dno-net"))
        from dno_net_v2 import CraigSulemTermsResidual

        if args.pads != [2]:
            parser.error("the overlap trial uses --pads 2")
        cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
        stats = json.loads((args.dataset / "stats.json").read_text())
        eta_scale, xi_scale = stats["feature_absmax"]
        model = CraigSulemTermsResidual(
            width=320, surface_context_width=8, eta_scale=eta_scale,
            xi_scale=xi_scale, target_scale=stats["target_absmax"], domain_length=length,
            surface_feature_scales=(0.06807905261284868, 0.032644640850224144,
                                   0.02186158732707002, 0.35135935945234514,
                                   7.985415697040461, 0.11091670707039558,
                                   0.05708673914278277),
        )
        variables = model.init(jax.random.PRNGKey(0), jnp.zeros((1, nx, 2), jnp.float32),
                               jnp.zeros((1, 1), jnp.float32), method=model.encode_geometry)
        # Exercise a live decoder: a zero head would eliminate the entire encoder.
        head = cast(jax.Array, variables["params"]["terms_head"]["kernel"])
        variables["params"]["terms_head"]["kernel"] = 0.001 * jax.random.normal(jax.random.PRNGKey(1), head.shape, dtype=head.dtype)
        report.update(device=gpu.device_kind, scope="Forward-only random nonzero-head model; transfers and join included; no training",
                      parameters=sum(value.size for value in jax.tree.leaves(variables)))
        k_cpu = jax.device_put(np.asarray(k), cpu)
        with ThreadPoolExecutor(max_workers=1) as pool:
            for batch in args.batches:
                host_eta, host_xi = eta_all[:batch], xi_all[:batch]
                device_eta, device_xi = jax.device_put((host_eta, host_xi), gpu)
                depth = jax.device_put(depth_all[:batch], gpu)
                depth_cpu = jax.device_put(depth_all[:batch], cpu)
                log_depth = jnp.log(depth).astype(jnp.float32)
                encode = jax.jit(lambda eta: model.apply(
                    variables, jnp.stack((eta / eta_scale, jnp.zeros_like(eta)), axis=-1).astype(jnp.float32),
                    log_depth, method=model.encode_geometry,
                )).lower(device_eta).compile()
                coefficients = encode(device_eta).block_until_ready()
                cpu_fn = jax.jit(partial(dno_series_eval, k=k_cpu, depth=depth_cpu,
                                         order=3, pad_factor=2, return_terms=True),
                                 in_shardings=jax.sharding.SingleDeviceSharding(cpu),
                                 out_shardings=jax.sharding.SingleDeviceSharding(cpu)).lower(
                    jax.device_put(host_eta, cpu), jax.device_put(host_xi, cpu)).compile()
                terms = jax.device_put(cpu_fn(jax.device_put(host_eta, cpu), jax.device_put(host_xi, cpu)), gpu)
                decode = jax.jit(partial(model.apply, {}, method=model.decode_terms)).lower(coefficients, terms).compile()

                def all_gpu(eta: jax.Array, xi: jax.Array) -> jax.Array:
                    features = model.apply(variables, jnp.stack((eta / eta_scale, jnp.zeros_like(eta)), axis=-1).astype(jnp.float32),
                                           log_depth, method=model.encode_geometry)
                    values = dno_series_eval(eta, xi, k, depth, order=3, pad_factor=2, return_terms=True)
                    return cast(jax.Array, model.apply({}, features, values, method=model.decode_terms))

                native = jax.jit(all_gpu).lower(device_eta, device_xi).compile()

                def cpu_terms(eta: np.ndarray | jax.Array, xi: np.ndarray | jax.Array) -> jax.Array:
                    return cpu_fn(jax.device_put(eta, cpu), jax.device_put(xi, cpu)).block_until_ready()

                def evaluate(mode: str, resident: bool) -> jax.Array:
                    eta, xi = (device_eta, device_xi) if resident else (host_eta, host_xi)
                    if mode == "all_gpu":
                        return native(jax.device_put(eta, gpu), jax.device_put(xi, gpu))
                    if mode == "serial":
                        values = cpu_terms(eta, xi)
                        encoded = encode(jax.device_put(eta, gpu)).block_until_ready()
                    else:
                        pending = pool.submit(cpu_terms, eta, xi)
                        encoded = encode(jax.device_put(eta, gpu))
                        values = pending.result()
                    return decode(encoded, jax.device_put(values, gpu))

                expected = np.asarray(native(device_eta, device_xi).block_until_ready())
                for resident in (False, True):
                    for mode in ("all_gpu", "serial", "overlap"):
                        actual = np.asarray(evaluate(mode, resident).block_until_ready())
                        relative = float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))
                        if not np.isfinite(actual).all() or relative > 1e-9:
                            raise AssertionError(f"{mode} B{batch} does not match all-GPU reference: {relative}")
                        for _ in range(5):
                            evaluate(mode, resident).block_until_ready()
                        seconds = []
                        for _ in range(args.repeats):
                            started = perf_counter()
                            evaluate(mode, resident).block_until_ready()
                            seconds.append(perf_counter() - started)
                        row = {"batch": batch, "mode": mode, "input_residency": "gpu" if resident else "host",
                               "median_ms": float(np.median(seconds) * 1000), "relative_l2": relative,
                               "p10_p90_ms": (np.percentile(seconds, [10, 90]) * 1000).tolist(), "seconds": seconds}
                        report["results"].append(row)
                        print(json.dumps({key: value for key, value in row.items() if key != "seconds"}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        raise SystemExit(0)
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
