"""Fixed-order classical DNO trials; --rollouts times complete single trajectories."""
from __future__ import annotations

import argparse
from functools import partial
from itertools import product
import json
import os
from pathlib import Path
import sys
import time
from types import FunctionType, SimpleNamespace
from typing import Any, Literal, cast

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from scripts.benchmark_cufftdx import classical_fused_hat, fft  # noqa: E402
from scripts.benchmark_surrogate_rhs import rollout_if_trial  # noqa: E402
from solver.evals.eval_suite import FAMILY_CONFIGS  # noqa: E402
from solver.evals.model_rollout import LoadedRun, load_run  # noqa: E402
from solver.solvers import time_integrator as ti  # noqa: E402
from solver.solvers.dno_series_jax import _dno_series_hat  # noqa: E402

Backend = Literal["cufft", "cufftdx_all", "cufftdx_products", "cufftdx_all_products",
                  "specialized", "specialized_cufftdx", "specialized_cufftdx_products",
                  "interleaved", "interleaved_cufftdx"]
Precision = Literal["fp64", "fp32", "mixed"]
VARIANTS = {f"{backend}_{precision}_p{pad}": dict(backend=backend, precision=precision, pad_factor=pad)
            for backend, precision, pad in product(
                ("cufft", "cufftdx_all", "cufftdx_products", "cufftdx_all_products",
                 "specialized", "specialized_cufftdx", "specialized_cufftdx_products",
                 "interleaved", "interleaved_cufftdx"),
                ("fp64", "fp32", "mixed"), (1, 2))}
CONTROL = "cufft_fp64_p2"


def classical_hat(
    eta_hat: jax.Array, xi_hat: jax.Array, k: jax.Array, g0: jax.Array, *,
    nx: int, order: int, pad_factor: int = 2, backend: Backend = "cufft",
    precision: Precision = "fp64", ept: int = 8,
) -> jax.Array:
    """Mixed uses FP32 physical FFT/products, FP64 spectral arithmetic and exact G0.

    All modes through N/2 are retained; only the Nyquist mode is zeroed.
    Return complex128 so the surrounding time integrator remains FP64.
    """
    spectral = jnp.complex64 if precision == "fp32" else jnp.complex128
    eta_hat, xi_hat = eta_hat.astype(spectral), xi_hat.astype(spectral)
    k, g0 = (value.astype(jnp.float32 if precision == "fp32" else jnp.float64) for value in (k, g0))
    native_all = backend in ("cufftdx_all", "cufftdx_all_products", "specialized_cufftdx",
                            "specialized_cufftdx_products", "interleaved_cufftdx")
    fused_products = backend.endswith("products")

    def transform(
        values: jax.Array, n: int | None = None, axis: int = -1, *,
        inverse: bool = False, b: jax.Array | None = None,
    ) -> jax.Array:
        dtype = ((jnp.complex128 if inverse else jnp.float64) if precision == "fp64"
                 else jnp.complex64 if inverse else jnp.float32)
        values = values.astype(dtype)
        if native_all or b is not None:
            result = fft(values, values.shape[-1] if n is None else n, inverse=inverse,
                         b=None if b is None else b.astype(dtype), ept=ept)
        else:
            result = (jnp.fft.irfft if inverse else jnp.fft.rfft)(values, n=n, axis=axis)
        return result if inverse else result.astype(spectral)

    if backend.startswith("specialized"):
        padded_nx = pad_factor * nx

        def projected_product(a: jax.Array, b: jax.Array) -> jax.Array:
            if fused_products:
                a, b = jnp.broadcast_arrays(a, b)
                coefficients = transform(a, b=b)
            else:
                coefficients = transform(a * b)
            return pad_factor * coefficients[..., :nx // 2 + 1].at[..., nx // 2].set(0)

        q0 = g0 * xi_hat
        eta, xi_x, q0_physical = jnp.moveaxis(transform(jnp.stack(
            (eta_hat, 1j * k * xi_hat, q0), axis=-2), padded_nx, inverse=True), -2, 0)
        fields = (eta, xi_x, q0_physical) if order == 2 else (xi_x, q0_physical)
        products = projected_product(eta[..., None, :], jnp.stack(fields, axis=-2))
        symbols = jnp.stack(jnp.broadcast_arrays(1j * k, g0), axis=-2)
        q1 = -jnp.sum(symbols * products[..., -2:, :], axis=-2)
        result = q0 + q1
        if order == 2:
            e2, q1_physical = jnp.moveaxis(transform(jnp.stack(
                (products[..., 0, :] / 2, q1), axis=-2), padded_nx, inverse=True), -2, 0)
            products = projected_product(jnp.stack((e2, e2, eta), axis=-2),
                                         jnp.stack((xi_x, q0_physical, q1_physical), axis=-2))
            symbols = jnp.stack(jnp.broadcast_arrays(g0 * (1j * k), k ** 2, g0), axis=-2)
            result -= jnp.sum(symbols * products, axis=-2)
    elif backend.startswith("interleaved"):
        padded_nx = pad_factor * nx
        gm_hats = [g0 * xi_hat]
        eta, xi_x, q0_physical = jnp.moveaxis(transform(jnp.stack(
            (eta_hat, 1j * k * xi_hat, gm_hats[0]), axis=-2), padded_nx, inverse=True), -2, 0)
        eta_powers, gm_padded = [eta, eta], [q0_physical]
        for m in range(1, order + 1):
            products = [(eta_powers[m], xi_x)]
            if m % 2 == 0:
                r = m // 2
                symbols = [g0 * k ** (2 * (r - 1)) * (1j * k)]
                for s in range(r):
                    products.extend(((eta_powers[2 * (r - s)], gm_padded[2 * s]),
                                     (eta_powers[2 * (r - s) - 1], gm_padded[2 * s + 1])))
                    symbols.extend((k ** (2 * (r - s)), g0 * k ** (2 * (r - s - 1))))
            else:
                r = (m + 1) // 2
                symbols = [k ** (2 * (r - 1)) * (1j * k)]
                for s in range(r - 1):
                    products.extend(((eta_powers[2 * (r - s) - 1], gm_padded[2 * s]),
                                     (eta_powers[2 * (r - s - 1)], gm_padded[2 * s + 1])))
                    symbols.extend((g0 * k ** (2 * (r - s - 1)), k ** (2 * (r - s - 1))))
                products.append((eta_powers[1], gm_padded[2 * (r - 1)]))
                symbols.append(g0)
            # The next surface power shares this FFT batch, retaining its projection.
            if m < order:
                products.append((eta, eta_powers[m]))
            left, right = zip(*products)
            coefficients = transform(jnp.stack(left, axis=-2) * jnp.stack(right, axis=-2))[..., :nx // 2 + 1]
            product_hats = pad_factor * coefficients.at[..., nx // 2].set(0)
            gm_hat = -jnp.sum(jnp.stack(jnp.broadcast_arrays(*symbols), axis=-2)
                             * product_hats[..., :len(symbols), :], axis=-2)
            gm_hats.append(gm_hat)
            if m < order:
                gm, power = jnp.moveaxis(transform(jnp.stack(
                    (gm_hat, product_hats[..., -1, :] / (m + 1)), axis=-2), padded_nx, inverse=True), -2, 0)
                gm_padded.append(gm)
                eta_powers.append(power)
        result = sum(gm_hats, start=jnp.zeros_like(gm_hats[0]))
    elif backend == "cufft" and precision == "fp64":
        result = _dno_series_hat(eta_hat, xi_hat, k, g0, nx=nx, order=order, pad_factor=pad_factor)
    else:
        source = classical_fused_hat if fused_products else _dno_series_hat
        # Private FFT bindings preserve the recurrence without process-global patches.
        namespace = SimpleNamespace(**vars(jnp))
        namespace.fft = SimpleNamespace(rfft=transform, irfft=partial(transform, inverse=True))
        recurrence = FunctionType(source.__code__, {**source.__globals__, "jnp": namespace, "fft": transform})
        result = recurrence(eta_hat, xi_hat, k, g0, nx=nx, order=order, pad_factor=pad_factor)
    return result.astype(jnp.complex128)


def errors(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = (actual - reference).reshape(actual.shape[0], -1)
    target = reference.reshape(reference.shape[0], -1)
    relative = np.linalg.norm(delta, axis=-1) / np.maximum(np.linalg.norm(target, axis=-1), np.finfo(float).tiny)
    return {"relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(target), np.finfo(float).tiny)),
            "max_relative_l2": float(relative.max()), "terminal_relative_l2": float(relative[-1]),
            "max_absolute": float(np.abs(delta).max())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, choices=range(1, 7), default=2)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=[CONTROL, "specialized_fp64_p2"])
    parser.add_argument("--control", choices=[f"{name}_e{ept}" for name in VARIANTS
                        for ept in ((4, 8, 16) if "cufftdx" in name else (8,))], default=f"{CONTROL}_e8")
    parser.add_argument("--families", nargs="+", choices=FAMILY_CONFIGS, default=list(FAMILY_CONFIGS))
    parser.add_argument("--batches", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    parser.add_argument("--repeats", type=int, help="Default: 100 forward calls or 3 full rollouts.")
    parser.add_argument("--epts", nargs="+", type=int, choices=(4, 8, 16), default=[8])
    parser.add_argument("--rollouts", action="store_true", help="Full horizons with same-executable warm-up.")
    parser.add_argument("--sweep", action="store_true", help="Time all six orders with cuFFT controls and optimized native FFTs.")
    parser.add_argument("--candidate-run", type=Path, help="Also time previous/current Fourier-input candidate kernels.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/classical_tuning_20260927/micro.json")
    args = parser.parse_args()
    if args.repeats is None:
        args.repeats = 3 if args.rollouts else 100
    if args.repeats < 1 or (not args.sweep and args.order > 2 and any(name.startswith("specialized") for name in (*args.variants, args.control))):
        parser.error("positive repeats required; specialized algebra supports M1/M2 only")
    if (args.sweep or args.candidate_run) and not args.rollouts:
        parser.error("--sweep and --candidate-run require --rollouts")
    jax.config.update("jax_enable_x64", True)
    control = args.control
    control_name, _, control_ept = control.rpartition("_e")
    variants = {control: {**VARIANTS[control_name], "ept": int(control_ept)}, **{
        f"{name}_e{ept}": {**VARIANTS[name], "ept": ept}
        for name in args.variants for ept in (args.epts if "cufftdx" in name else (8,))}}
    variants = {name: {"order": args.order, **settings} for name, settings in variants.items()}
    if args.sweep:
        variants = {f"M{order}_{kind}": {
            "order": order, "precision": "fp64",
            "backend": ("cufft" if kind == "cufft" else "specialized_cufftdx" if order == 2 else "cufftdx_all"),
            "pad_factor": 1 if kind == "native" and order <= 2 else 2,
            "ept": 16 if kind == "native" and order == 2 else 8,
        } for order in range(1, 7) for kind in ("cufft", "native")}
    controls = {name: f"M{settings['order']}_cufft" if args.sweep else control for name, settings in variants.items()}
    candidates = ({f"{variant}_spectral_all_boundaries_stages_carry_local": variant for variant in (
        "fourier_fused_features_overlap", "fourier_fused_features_product_overlap")}
        if args.candidate_run else {})
    loaded = load_run(args.candidate_run) if args.candidate_run else None
    controls.update({name: next(iter(candidates)) for name in candidates})
    controls["M6"] = "M6_cufft" if args.sweep else control
    rollout = rollout_if_trial
    if args.rollouts:
        from scripts.time_single_rollouts import packed_boundary

        namespace = SimpleNamespace(**vars(ti))
        namespace._state_to_hat = packed_boundary
        namespace._hat_to_state = partial(packed_boundary, inverse=True)
        rollout = FunctionType(rollout_if_trial.__code__, {**rollout_if_trial.__globals__, "ti": namespace})
    report: dict[str, Any] = {"device": jax.devices()[0].device_kind, "order": list(range(1, 7)) if args.sweep else args.order,
        "reference": "M6 cufft FP64 pad2", "control": controls if args.sweep or candidates else control,
        "variants": variants, "candidates": candidates, "candidate_run": str(args.candidate_run) if candidates else None,
        "fft_epts": sorted({settings["ept"] for settings in variants.values()}),
        "timing_repeats": {name: 1 if args.sweep and name.endswith("_cufft") else args.repeats
                           for name in (*variants, *candidates)},
        "mode": "full_single_rollouts" if args.rollouts else "spectral_forward",
        "precision": "FP64 integrator; mixed means FP32 FFT/physical products, FP64 spectral arithmetic",
        "families": {}}
    for family in args.families:
        cfg = FAMILY_CONFIGS[family]
        source = next((ROOT / "outputs/c27_tanaka_hard128_full_equal_local_20260918").glob(
            f"eval_best_current_test_stratified_n32*/{family}_trajs.npz"))
        with np.load(source) as saved:
            times = jnp.asarray(saved["times"], dtype=jnp.float64)
            eta, xi = (jnp.asarray(saved[f"truth_{key}"][:, :1], dtype=jnp.float64) for key in ("eta", "xi"))
            depth = jnp.asarray(saved["depths"][:1, None], dtype=jnp.float64)
            cached = {key: saved[f"truth_{key}"][:, :1] for key in ("eta", "xi", "gxi")}
            simulation_id = int(saved["simulation_ids"][0])
        nx = eta.shape[-1]
        params = ti.make_solver_params(nx, 2 * np.pi, depth, dno_order=args.order,
                                       pad_factor=2, filter_fraction=cfg.filter_fraction)
        k, g0 = params.k[:nx // 2 + 1], params.g0[..., :nx // 2 + 1]
        rows: dict[str, Any] = {}
        report["families"][family] = {"simulation_id": simulation_id, "nx": nx, "horizon": float(times[-1]),
                                       "substeps": cfg.substeps, "filter_fraction": cfg.filter_fraction, "results": rows}
        for batch in ([1] if args.rollouts else args.batches):
            inputs = ((ti.State(eta[0], xi[0]), times) if args.rollouts else tuple(
                jnp.fft.rfft(field[:batch, 0]).at[..., nx // 2].set(0) for field in (eta, xi)))
            compiled, outputs, records = {}, {}, {}
            for name in ("M6", *variants, *candidates):
                physical_predict = None
                if name in candidates:
                    from scripts.benchmark_dno_fusion import build_variant

                    predict = build_variant(cast(LoadedRun, loaded), jnp.log(depth[:, 0]), nx, candidates[name], fft_ept=8)
                    physical_predict = partial(lambda e, x, fn: jnp.fft.irfft(
                        fn(jnp.fft.rfft(e), jnp.fft.rfft(x)), n=e.shape[-1]), fn=predict)
                else:
                    settings = {**VARIANTS[CONTROL], "ept": 8, "order": 6} if name == "M6" else variants[name]
                    predict = partial(classical_hat, k=k, g0=g0, nx=nx, **settings)
                function = (partial(rollout, params=params, substeps=cfg.substeps,
                                    carry_spectral=True, local_if=True, half_spectrum=False,
                                    predict_hat=predict, predict_gxi=physical_predict)
                            if args.rollouts else predict)
                started = time.perf_counter()
                compiled[name] = jax.jit(function).lower(*inputs).compile()
                records[name] = {"compile_s": time.perf_counter() - started, "seconds": []}
                started = time.perf_counter()
                for _ in range(1 if args.rollouts else 10):
                    outputs[name] = jax.block_until_ready(compiled[name](*inputs))
                records[name]["warmup_s"] = time.perf_counter() - started
                if not args.rollouts:
                    flat = jnp.zeros_like(cast(jax.Array, inputs[0]))
                    zero = np.asarray(compiled[name](flat, inputs[1]))
                    records[name]["flat_g0_max_absolute"] = float(np.abs(zero - np.asarray(g0 * inputs[1])).max())
                print(f"{family} B{batch} {name}: compiled and warmed", flush=True)
            for repeat in range(args.repeats):
                for name in [*variants, *candidates][::1 if repeat % 2 == 0 else -1]:
                    if args.sweep and name.endswith("_cufft") and repeat > 0:
                        continue
                    started = time.perf_counter()
                    outputs[name] = jax.block_until_ready(compiled[name](*inputs))
                    records[name]["seconds"].append(time.perf_counter() - started)
                    print(f"{family} B{batch} {name}: repeat {repeat + 1}: {records[name]['seconds'][-1]:.6f}s", flush=True)
            for name, result in outputs.items():
                record = records[name]
                if record["seconds"]:
                    record["median_seconds"] = float(np.median(record["seconds"]))
                if args.rollouts:
                    record["finite"] = all(bool(np.isfinite(np.asarray(result[key])).all()) for key in cached)
                    record["minimum_depth"] = (float((np.asarray(result["eta"]) + np.asarray(depth)).min())
                                               if record["finite"] else None)
                    for key in cached:
                        actual = np.asarray(result[key])
                        if record["finite"]:
                            record[key] = {"control": errors(actual, np.asarray(outputs[controls[name]][key])),
                                           "M6": errors(actual, np.asarray(outputs["M6"][key])),
                                           "cached_M6": errors(actual, cached[key])}
                else:
                    record["finite"] = bool(np.isfinite(np.asarray(result)).all())
                    if record["finite"]:
                        record.update(control=errors(np.asarray(result), np.asarray(outputs[controls[name]])),
                                      M6=errors(np.asarray(result), np.asarray(outputs["M6"])))
                print(f"{family} B{batch} {name}: {record.get('median_seconds')} s; finite={record['finite']}", flush=True)
            rows[f"B{batch}"] = records
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        jax.clear_caches()
