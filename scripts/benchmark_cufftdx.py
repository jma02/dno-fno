"""Screen native-precision cuFFTDx kernels before changing any production code.

Download NVIDIA MathDx into outputs/cufftdx_20260925, then run with
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/benchmark_cufftdx.py --build.
"""
from __future__ import annotations

import argparse
import ctypes
from functools import cache, partial
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, cast
from unittest.mock import patch

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.stages import Compiled  # noqa: E402

OUTPUT = ROOT / "outputs/cufftdx_20260925"


@cache
def register() -> ctypes.CDLL:
    library = ctypes.CDLL(str(OUTPUT / "cufftdx.so"))
    for name in ("dno_cufftdx", "dno_cufftdx_features"):
        jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(getattr(library, name)), platform="CUDA")
    return library


def fft(
    a: jax.Array, n: int, *, inverse: bool = False, b: jax.Array | None = None,
    outer: jax.Array | None = None, ept: int = 8, normalization: jax.Array | None = None,
    promote: bool = False,
) -> jax.Array:
    """Fuse inverse normalization or promote FP32 inputs on load into an FP64 FFT."""
    register()
    fp64 = promote or a.dtype in (jnp.float64, jnp.complex128)
    dtype = (jnp.float64 if fp64 else jnp.float32) if inverse else (jnp.complex128 if fp64 else jnp.complex64)
    if normalization is not None:
        dtype = jnp.float32
    result = jax.ShapeDtypeStruct((*a.shape[:-1], n if inverse else n // 2 + 1), dtype)
    return jax.ffi.ffi_call("dno_cufftdx", result, vmap_method="broadcast_all")(
        a, normalization if normalization is not None else a if b is None else b, a if outer is None else outer,
        size=np.int64(n), operation=np.int64(9 if promote else 8 if normalization is not None else 4 if outer is not None else inverse),
        multiply=b is not None, ept=np.int64(ept),
    )


def field_irfft(
    inputs_hat: jax.Array, surface_ops: jax.Array, filters: jax.Array, *, n: int, ept: int = 8,
    normalized_inputs: jax.Array | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Form fields on load; optionally emit surface features [B,7,N] and branch fields."""
    register()
    if normalized_inputs is not None:
        results = (jax.ShapeDtypeStruct((*normalized_inputs.shape[:-2], surface_ops.shape[-2] + 3, n), jnp.float32),
                   jax.ShapeDtypeStruct((*normalized_inputs.shape[:-2], filters.shape[-2], n), jnp.float32))
        features, branches = jax.ffi.ffi_call("dno_cufftdx_features", results, vmap_method="broadcast_all")(
            inputs_hat, surface_ops, filters, normalized_inputs, ept=np.int64(ept),
        )
        return features, branches
    result = jax.ShapeDtypeStruct(
        (*inputs_hat.shape[:-2], surface_ops.shape[-2] + filters.shape[-2], n), jnp.float32,
    )
    return jax.ffi.ffi_call("dno_cufftdx", result, vmap_method="broadcast_all")(
        inputs_hat, surface_ops, filters, size=np.int64(n), operation=np.int64(7),
        multiply=False, ept=np.int64(ept),
    )


def sandwich(
    eta: jax.Array, xi: jax.Array, symbols: jax.Array, *, joint: bool = False,
    spectral: bool = False, ept: int = 8, eta_scale: float | None = None,
) -> jax.Array:
    """Evaluate L[eta L xi]; optional eta_scale scales FP32 eta before FP64 promotion."""
    register()
    shape = symbols.shape if spectral else xi.shape if joint else (*symbols.shape[:-1], xi.shape[-1])
    result = jax.ShapeDtypeStruct(shape, xi.dtype)
    if eta_scale is not None:
        symbols = jnp.concatenate((symbols, jnp.full(
            (*symbols.shape[:-1], 1), jnp.float32(eta_scale), dtype=symbols.dtype,
        )), axis=-1)
    return jax.ffi.ffi_call("dno_cufftdx", result, vmap_method="broadcast_all")(
        eta, xi, symbols, size=np.int64(eta.shape[-1]), operation=np.int64(10 if eta_scale is not None else 6 if spectral else 3 if joint else 2),
        multiply=False, ept=np.int64(ept),
    )


def classical_fused_hat(
    eta_hat: jax.Array, xi_hat: jax.Array, k: jax.Array, g0: jax.Array, *,
    nx: int, order: int, pad_factor: int = 8,
) -> jax.Array:
    """Keep the padded classical recurrence, fusing only product-to-forward FFTs."""
    padded_nx = pad_factor * nx

    def pad(coefficients: jax.Array) -> jax.Array:
        return jnp.fft.irfft(coefficients, n=padded_nx, axis=-1)

    def product_hat(a: jax.Array, b: jax.Array) -> jax.Array:
        coefficients = fft(a, padded_nx, b=b)[..., : nx // 2 + 1]
        return pad_factor * coefficients.at[..., nx // 2].set(0)

    gm_hats = [g0 * xi_hat]
    eta_padded, xi_x, g0_padded = jnp.moveaxis(
        pad(jnp.stack((eta_hat, 1j * k * xi_hat, gm_hats[0]), axis=-2)), -2, 0,
    )
    eta_powers = [eta_padded, eta_padded]
    for m in range(2, order + 1):
        eta_powers.append(pad(product_hat(eta_padded, eta_powers[m - 1]) / m))

    gm_padded = [g0_padded]
    for m in range(1, order + 1):
        if m % 2 == 0:
            r = m // 2
            products = [(eta_powers[m], xi_x)]
            symbols = [g0 * k ** (2 * (r - 1)) * (1j * k)]
            for s in range(r):
                products.extend(
                    ((eta_powers[2 * (r - s)], gm_padded[2 * s]),
                     (eta_powers[2 * (r - s) - 1], gm_padded[2 * s + 1]))
                )
                symbols.extend((k ** (2 * (r - s)), g0 * k ** (2 * (r - s - 1))))
        else:
            r = (m + 1) // 2
            products = [(eta_powers[m], xi_x)]
            symbols = [k ** (2 * (r - 1)) * (1j * k)]
            for s in range(r - 1):
                products.extend(
                    ((eta_powers[2 * (r - s) - 1], gm_padded[2 * s]),
                     (eta_powers[2 * (r - s - 1)], gm_padded[2 * s + 1]))
                )
                symbols.extend((g0 * k ** (2 * (r - s - 1)), k ** (2 * (r - s - 1))))
            products.append((eta_powers[1], gm_padded[2 * (r - 1)]))
            symbols.append(g0)

        left, right = zip(*products)
        product_hats = product_hat(jnp.stack(left, axis=-2), jnp.stack(right, axis=-2))
        gm_hat = -jnp.sum(
            jnp.stack(jnp.broadcast_arrays(*symbols), axis=-2) * product_hats, axis=-2,
        )
        gm_hats.append(gm_hat)
        if m < order:
            gm_padded.append(pad(gm_hat))
    return sum(gm_hats, start=jnp.zeros_like(gm_hats[0]))


def measure(functions: list[Compiled], x: jax.Array, y: jax.Array, repeats: int) -> dict[str, Any]:
    expected, actual = (np.asarray(f(x, y)) for f in functions)
    tolerance = 1e-12 if expected.dtype in (np.float64, np.complex128) else 2e-6
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance * np.max(np.abs(expected)))
    for f in functions:
        for _ in range(50):
            f(x, y).block_until_ready()
    seconds: list[list[float]] = [[], []]
    for repeat in range(repeats):
        for index in (0, 1) if repeat % 2 == 0 else (1, 0):
            started = time.perf_counter()
            functions[index](x, y).block_until_ready()
            seconds[index].append(time.perf_counter() - started)
    medians = np.median(seconds, axis=1) * 1e6
    return {"jax_us": float(medians[0]), "cufftdx_us": float(medians[1]),
            "relative_l2": float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)), "seconds": seconds}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--sm", type=int, default=89)
    parser.add_argument("--ept", nargs="+", type=int, choices=(4, 8, 16), default=[8])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--repeats", type=int, default=300)
    classical = parser.add_mutually_exclusive_group()
    classical.add_argument("--classical", action="store_true", help="Screen the same FFT backend in the complete padded M1–M6 recurrence.")
    classical.add_argument("--classical-fused", action="store_true", help="Fuse padded classical products with forward FFTs, retaining JAX inverse FFTs.")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.build or args.build_only:
        mathdx = next(OUTPUT.glob("nvidia-mathdx-*/nvidia/mathdx/*/include"))
        subprocess.run([
            shutil.which("nvcc") or "/usr/local/cuda-12.9/bin/nvcc", "-std=c++17", "-O3",
            f"-arch=sm_{args.sm}", f"-DDNO_SM={args.sm * 10}", "--shared",
            "-Xcompiler", "-fPIC", "-DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY",
            f"-I{mathdx}", f"-I{jax.ffi.include_dir()}",
            str(ROOT / "scripts/cufftdx_fft.cu"), "-o", str(OUTPUT / "cufftdx.so"),
        ], check=True)
    if args.build_only:
        raise SystemExit(0)
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(42)
    records = {}
    for n, batch, dtype in (() if args.classical or args.classical_fused else (
        (1024, 1, jnp.float64), (1024, 2, jnp.float64), (8192, 3, jnp.float64), (1024, 32, jnp.float32),
    )):
        a, b = (jnp.asarray(rng.normal(size=(batch, n)), dtype=dtype) for _ in range(2))
        spectrum = jnp.fft.rfft(a)
        cases = {
            "rfft": (lambda x, y: jnp.fft.rfft(x), lambda x, y: fft(x, n), a, b),
            "irfft": (lambda x, y: jnp.fft.irfft(x, n=n), lambda x, y: fft(x, n, inverse=True), spectrum, spectrum),
            "product_rfft": (lambda x, y: jnp.fft.rfft(x * y), lambda x, y: fft(x, n, b=y), a, b),
        }
        if n == 1024 and batch == 32:
            outer = jnp.asarray(rng.normal(size=(batch, n // 2 + 1)), dtype=jnp.complex64)
            for ept in args.ept:
                cases[f"outer_product_rfft_e{ept}"] = (
                    lambda x, y: outer * jnp.fft.rfft(x * y, norm="forward"),
                    lambda x, y, ept=ept: fft(x, n, b=y, outer=outer, ept=ept), a, b,
                )
        if n == 1024 and batch in (1, 2):
            input_pair = jnp.stack((jnp.fft.rfft(a), jnp.fft.rfft(b)), axis=1)
            scales = jnp.asarray((1.3, 0.7), dtype=jnp.float32)
            normalization = jnp.concatenate((1 / scales, jnp.asarray((0.9,), dtype=jnp.float32)))
            surface_ops = jnp.asarray(
                rng.normal(size=(4, n // 2 + 1)) + 1j * rng.normal(size=(4, n // 2 + 1)),
                dtype=jnp.complex64,
            )
            field_filters = jnp.asarray(rng.normal(size=(batch, 32, n // 2 + 1)), dtype=jnp.float32)
            for ept in args.ept:
                cases[f"promoted_rfft_e{ept}"] = (
                    lambda x, y, ept=ept: fft(x.astype(jnp.float64), n, ept=ept),
                    lambda x, y, ept=ept: fft(x, n, promote=True, ept=ept),
                    a.astype(jnp.float32), b,
                )
                cases[f"normalized_irfft_e{ept}"] = (
                    lambda x, y, ept=ept: (fft(x, n, inverse=True, ept=ept).astype(jnp.float32)
                        / scales[None, :, None]) * jnp.asarray((1.0, normalization[2]))[None, :, None],
                    lambda x, y, ept=ept: fft(x, n, inverse=True, ept=ept, normalization=normalization),
                    input_pair, input_pair,
                )
                cases[f"field_irfft_e{ept}"] = (
                    lambda x, y, ept=ept: fft(jnp.concatenate((
                        surface_ops[None] * x[:, :1], y * x[:, 1:2],
                    ), axis=1), n, inverse=True, ept=ept),
                    lambda x, y, ept=ept: cast(jax.Array, field_irfft(x, surface_ops, y, n=n, ept=ept)),
                    input_pair.astype(jnp.complex64), field_filters,
                )
                for stage_pair in (False, True):
                    for output in (0, 1):
                        def feature_reference(x: jax.Array, y: jax.Array, ept: int = ept, output: int = output) -> jax.Array:
                            fields = cast(jax.Array, field_irfft(x, surface_ops, field_filters, n=n, ept=ept))
                            eta = y[:, 0]
                            return (jnp.concatenate((jnp.stack((eta, eta ** 2, eta ** 3), axis=1), fields[:, :4]), axis=1)
                                    if output == 0 else fields[:, 4:])

                        def feature_candidate(x: jax.Array, y: jax.Array, ept: int = ept, output: int = output) -> jax.Array:
                            return field_irfft(x, surface_ops, field_filters, n=n, ept=ept, normalized_inputs=y)[output]

                        inputs, normalized = input_pair.astype(jnp.complex64), jnp.stack((a, b), axis=1).astype(jnp.float32)
                        if stage_pair:
                            inputs, normalized = jnp.stack((inputs, -inputs)), jnp.stack((normalized, -normalized))
                        cases[f"feature_irfft_{output}_{'vmap' if stage_pair else 'direct'}_e{ept}"] = (
                            jax.vmap(feature_reference) if stage_pair else feature_reference,
                            jax.vmap(feature_candidate) if stage_pair else feature_candidate, inputs, normalized,
                        )
            k = jnp.arange(n // 2 + 1, dtype=dtype)
            symbols = jnp.broadcast_to(jnp.stack((k * jnp.tanh(0.1 * k), 1j * k)), (batch, 2, n // 2 + 1))
            outer_symbols = symbols.at[..., [0, -1]].set(symbols[..., [0, -1]].real)
            nyquist = (-1.0) ** jnp.arange(n)
            spectral_eta, spectral_xi = a + 0.125 * nyquist, jnp.fft.rfft(b + 0.25 * nyquist)
            cases["g1"] = (
                lambda x, y: jnp.fft.irfft((symbols * jnp.fft.rfft(
                    x[:, None, :] * jnp.fft.irfft(symbols * jnp.fft.rfft(y)[:, None, :], n=n)
                )).sum(axis=1), n=n),
                lambda x, y: sandwich(x, y, symbols).sum(axis=1), a, b,
            )
            for ept in args.ept:
                eta_scale = 0.133552521
                cases[f"g1_prepared_eta_e{ept}"] = (
                    lambda x, y, ept=ept: sandwich((x * jnp.float32(eta_scale)).astype(jnp.float64),
                                                   y, symbols, spectral=True, ept=ept),
                    lambda x, y, ept=ept: sandwich(x, y, symbols, spectral=True, ept=ept, eta_scale=eta_scale),
                    spectral_eta.astype(jnp.float32), spectral_xi,
                )
                cases[f"g1_prepared_vmap_e{ept}"] = (
                    lambda x, y, ept=ept: jax.vmap(lambda e, z: sandwich(
                        (e * jnp.float32(eta_scale)).astype(jnp.float64),
                        fft(z.astype(jnp.float64), n, ept=ept), symbols, spectral=True, ept=ept,
                    ))(x, y),
                    lambda x, y, ept=ept: jax.vmap(lambda e, z: sandwich(
                        e, fft(z, n, promote=True, ept=ept), symbols,
                        spectral=True, ept=ept, eta_scale=eta_scale,
                    ))(x, y),
                    jnp.stack((a, b)).astype(jnp.float32), jnp.stack((b, -a)).astype(jnp.float32),
                )
                cases[f"g1_joint_e{ept}"] = (
                    cases["g1"][0], lambda x, y, ept=ept: sandwich(x, y, symbols, joint=True, ept=ept), a, b,
                )
                cases[f"g1_parallel_e{ept}"] = (
                    cases["g1"][0], lambda x, y, ept=ept: sandwich(x, y, symbols, ept=ept).sum(axis=1), a, b,
                )
                cases[f"g1_spectral_e{ept}"] = (
                    lambda x, y: outer_symbols * jnp.fft.rfft(
                        x[:, None, :] * jnp.fft.irfft(symbols * y[:, None, :], n=n)),
                    lambda x, y, ept=ept: sandwich(x, y, symbols, spectral=True, ept=ept), spectral_eta, spectral_xi,
                )
                cases[f"g1_spectral_roundtrip_e{ept}"] = (
                    lambda x, y, ept=ept: jnp.fft.rfft(sandwich(x, jnp.fft.irfft(y, n=n), symbols, ept=ept)),
                    lambda x, y, ept=ept: sandwich(x, y, symbols, spectral=True, ept=ept), spectral_eta, spectral_xi,
                )
        for operation, (reference, candidate, x, y) in cases.items():
            functions = [jax.jit(f).lower(x, y).compile() for f in (reference, candidate)]
            name = f"{operation}_{n}_batch{batch}_{jnp.dtype(dtype).name}"
            record = records[name] = measure(functions, x, y, args.repeats)
            print(f"{name}: JAX {record['jax_us']:.2f} us; cuFFTDx {record['cufftdx_us']:.2f} us; "
                  f"difference {record['relative_l2']:.3g}", flush=True)
            (args.output / "microbench.json").write_text(json.dumps(records, indent=2) + "\n")
    if args.classical or args.classical_fused:
        from solver.solvers.dno_series_jax import _dno_series_hat

        source = next((ROOT / "outputs/c27_tanaka_hard128_full_equal_local_20260918").glob(
            "eval_best_current_test_stratified_n32*/stokes_trajs.npz"
        ))
        with np.load(source) as saved:
            eta_hat, xi_hat = (jnp.fft.rfft(jnp.asarray(saved[f"truth_{key}"][0, :1])).at[:, 512].set(0)
                               for key in ("eta", "xi"))
            k = jnp.arange(513, dtype=jnp.float64)
            symbol = k * jnp.tanh(float(saved["depths"][0]) * k)
        for order in range(1, 7):
            reference_fn = partial(_dno_series_hat, k=k, g0=symbol, nx=1024, order=order, pad_factor=8)
            reference_exe = jax.jit(reference_fn).lower(eta_hat, xi_hat).compile()
            if args.classical_fused:
                candidate_fn = partial(classical_fused_hat, k=k, g0=symbol, nx=1024, order=order, pad_factor=8)
                candidate_exe = jax.jit(candidate_fn).lower(eta_hat, xi_hat).compile()
            else:
                # Patch only during this fresh trace; the production module is never edited.
                with patch.multiple(jnp.fft,
                                    rfft=lambda a, n=None, axis=-1: fft(a, a.shape[-1] if n is None else n),
                                    irfft=lambda a, n, axis=-1: fft(a, n, inverse=True)):
                    candidate_exe = jax.jit(lambda x, y: reference_fn(x, y)).lower(eta_hat, xi_hat).compile()
            record = records[f"M{order}"] = measure([reference_exe, candidate_exe], eta_hat, xi_hat, args.repeats)
            print(f"M{order}: JAX {record['jax_us']:.2f} us; cuFFTDx {record['cufftdx_us']:.2f} us; "
                  f"difference {record['relative_l2']:.3g}", flush=True)
            filename = "classical_fused_screen.json" if args.classical_fused else "classical_screen.json"
            (args.output / filename).write_text(json.dumps(records, indent=2) + "\n")
