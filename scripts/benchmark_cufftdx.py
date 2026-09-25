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
import subprocess
import sys
import time
from typing import Any
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
    jax.ffi.register_ffi_target("dno_cufftdx", jax.ffi.pycapsule(library.dno_cufftdx), platform="CUDA")
    return library


def fft(a: jax.Array, n: int, *, inverse: bool = False, b: jax.Array | None = None) -> jax.Array:
    register()
    fp64 = a.dtype in (jnp.float64, jnp.complex128)
    dtype = (jnp.float64 if fp64 else jnp.float32) if inverse else (jnp.complex128 if fp64 else jnp.complex64)
    result = jax.ShapeDtypeStruct((*a.shape[:-1], n if inverse else n // 2 + 1), dtype)
    return jax.ffi.ffi_call("dno_cufftdx", result, vmap_method="broadcast_all")(
        a, a if b is None else b, a, size=np.int64(n), operation=np.int64(inverse), multiply=b is not None,
    )


def sandwich(eta: jax.Array, xi: jax.Array, symbols: jax.Array) -> jax.Array:
    """Evaluate L[eta L xi] on the fixed 1024-point model grid."""
    register()
    result = jax.ShapeDtypeStruct((*symbols.shape[:-1], xi.shape[-1]), xi.dtype)
    return jax.ffi.ffi_call("dno_cufftdx", result, vmap_method="broadcast_all")(
        eta, xi, symbols, size=np.int64(xi.shape[-1]), operation=np.int64(2), multiply=False,
    )


def measure(functions: list[Compiled], x: jax.Array, y: jax.Array, repeats: int) -> dict[str, Any]:
    expected, actual = (np.asarray(f(x, y)) for f in functions)
    tolerance = 1e-12 if x.dtype in (jnp.float64, jnp.complex128) else 2e-6
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
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--classical", action="store_true", help="Screen the same FFT backend in the complete padded M1–M6 recurrence.")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.build:
        mathdx = next(OUTPUT.glob("nvidia-mathdx-*/nvidia/mathdx/*/include"))
        subprocess.run([
            "/usr/local/cuda-12.9/bin/nvcc", "-std=c++17", "-O3", "-arch=sm_89", "--shared",
            "-Xcompiler", "-fPIC", "-DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY",
            f"-I{mathdx}", f"-I{jax.ffi.include_dir()}",
            str(ROOT / "scripts/cufftdx_fft.cu"), "-o", str(OUTPUT / "cufftdx.so"),
        ], check=True)
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(42)
    records = {}
    for n, batch, dtype in (() if args.classical else (
        (1024, 1, jnp.float64), (1024, 2, jnp.float64), (8192, 3, jnp.float64), (1024, 32, jnp.float32),
    )):
        a, b = (jnp.asarray(rng.normal(size=(batch, n)), dtype=dtype) for _ in range(2))
        spectrum = jnp.fft.rfft(a)
        cases = {
            "rfft": (lambda x, y: jnp.fft.rfft(x), lambda x, y: fft(x, n), a, b),
            "irfft": (lambda x, y: jnp.fft.irfft(x, n=n), lambda x, y: fft(x, n, inverse=True), spectrum, spectrum),
            "product_rfft": (lambda x, y: jnp.fft.rfft(x * y), lambda x, y: fft(x, n, b=y), a, b),
        }
        if n == 1024 and batch == 1:
            k = jnp.arange(n // 2 + 1, dtype=dtype)
            symbols = jnp.stack((k * jnp.tanh(0.1 * k), 1j * k))[None]
            cases["g1"] = (
                lambda x, y: jnp.fft.irfft((symbols * jnp.fft.rfft(
                    x[:, None, :] * jnp.fft.irfft(symbols * jnp.fft.rfft(y)[:, None, :], n=n)
                )).sum(axis=1), n=n),
                lambda x, y: sandwich(x, y, symbols).sum(axis=1), a, b,
            )
        for operation, (reference, candidate, x, y) in cases.items():
            functions = [jax.jit(f).lower(x, y).compile() for f in (reference, candidate)]
            name = f"{operation}_{n}_batch{batch}_{jnp.dtype(dtype).name}"
            record = records[name] = measure(functions, x, y, args.repeats)
            print(f"{name}: JAX {record['jax_us']:.2f} us; cuFFTDx {record['cufftdx_us']:.2f} us; "
                  f"difference {record['relative_l2']:.3g}", flush=True)
            (OUTPUT / "microbench.json").write_text(json.dumps(records, indent=2) + "\n")
    if args.classical:
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
            # Patch only during this fresh trace; the production module is never edited.
            with patch.multiple(jnp.fft,
                                rfft=lambda a, n=None, axis=-1: fft(a, a.shape[-1] if n is None else n),
                                irfft=lambda a, n, axis=-1: fft(a, n, inverse=True)):
                candidate_exe = jax.jit(lambda x, y: reference_fn(x, y)).lower(eta_hat, xi_hat).compile()
            record = records[f"M{order}"] = measure([reference_exe, candidate_exe], eta_hat, xi_hat, args.repeats)
            print(f"M{order}: JAX {record['jax_us']:.2f} us; cuFFTDx {record['cufftdx_us']:.2f} us; "
                  f"difference {record['relative_l2']:.3g}", flush=True)
            (OUTPUT / "classical_screen.json").write_text(json.dumps(records, indent=2) + "\n")
