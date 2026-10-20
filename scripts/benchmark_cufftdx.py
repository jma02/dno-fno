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
from jax.custom_batching import custom_vmap  # noqa: E402
from jax.stages import Compiled  # noqa: E402

OUTPUT = ROOT / "outputs/cufftdx_20260925"


@cache
def register(field_stride: bool = False, branch_major: bool = False, bernoulli: bool = False,
             pair_coupling: bool = False) -> ctypes.CDLL:
    stem = "cufftdx_paircoupling" if pair_coupling else "cufftdx_bernoulli" if bernoulli else "cufftdx_branchmajor" if branch_major else "cufftdx_fieldstride" if field_stride else "cufftdx"
    library = ctypes.CDLL(str(OUTPUT / f"{stem}.so"))
    targets = ((f"dno_{stem}", "dno_cufftdx"),) if field_stride or branch_major or bernoulli or pair_coupling else tuple(
        (name, name) for name in ("dno_cufftdx", "dno_cufftdx_features"))
    for name, symbol in targets:
        jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(getattr(library, symbol)), platform="CUDA")
    return library


def _branch_major_fft(a: jax.Array, b: jax.Array, outer: jax.Array, n: int, ept: int,
                      pair_coupling: bool = False) -> jax.Array:
    """Flatten stage/batch axes before imposing the GEMM's branch-major storage."""
    a_flat = a.reshape((-1, *a.shape[-2:]))
    branches = outer.shape[-2]
    result = jax.ShapeDtypeStruct((a_flat.shape[0], branches, n // 2 + 1), jnp.complex64)
    spectrum = jax.ffi.ffi_call(
        "dno_cufftdx_paircoupling" if pair_coupling else "dno_cufftdx_branchmajor",
        result, vmap_method="broadcast_all",
        input_layouts=((2, 0, 1), None, None),
    )(
        a_flat, b.reshape((-1, *b.shape[-2:])), outer.reshape((-1, *outer.shape[-2:])),
        size=np.int64(n), operation=np.int64(15 if pair_coupling else 13), multiply=True, ept=np.int64(ept),
    )
    return spectrum.reshape((*a.shape[:-2], branches, n // 2 + 1))


def bernoulli_rfft(fields: jax.Array) -> jax.Array:
    """Double unscaled [...,3,2048] FP64 inverse fields and transform the Bernoulli term."""
    if fields.dtype != jnp.float64 or fields.shape[-2:] != (3, 2048):
        raise ValueError("Bernoulli FFT requires unscaled FP64 [...,3,2048] fields.")
    register(bernoulli=True)
    return jax.ffi.ffi_call(
        "dno_cufftdx_bernoulli", jax.ShapeDtypeStruct((*fields.shape[:-2], 1025), jnp.complex128),
        vmap_method="broadcast_all",
    )(fields, fields, fields, size=np.int64(2048), operation=np.int64(14), multiply=False, ept=np.int64(8))


def fft(
    a: jax.Array, n: int, *, inverse: bool = False, b: jax.Array | None = None,
    outer: jax.Array | None = None, ept: int = 8, normalization: jax.Array | None = None,
    promote: bool = False, channels_last: bool = False, field_stride: bool = False,
    branch_major: bool = False, pair_coupling: bool = False,
) -> jax.Array:
    """Fuse FFT loads/stores; channels_last consumes a column-major [B,N,R] head with real outer filters."""
    if branch_major or pair_coupling:
        if not channels_last or not field_stride or b is None or outer is None or a.dtype != jnp.float32:
            raise ValueError("Branch-major FFT requires an FP32 column-major head, strided fields, and real filters.")
        if pair_coupling and (n != 1024 or ept != 8 or outer.shape[-2] % 2
                              or a.shape[-1] != 3 * outer.shape[-2] // 2):
            raise ValueError("Pair coupling requires a 1024-point EPT8 head with R diagonal and R/2 cross columns.")
        register(branch_major=not pair_coupling, pair_coupling=pair_coupling)

        @custom_vmap
        def product(x: jax.Array, y: jax.Array, filters: jax.Array) -> jax.Array:
            return _branch_major_fft(x, y, filters, n, ept, pair_coupling)

        @product.def_vmap
        def product_vmap(axis_size: int, batched: list[bool], x: jax.Array,
                         y: jax.Array, filters: jax.Array) -> tuple[jax.Array, bool]:
            x, y, filters = tuple(value if mapped else jnp.broadcast_to(value, (axis_size, *value.shape))
                                  for value, mapped in zip((x, y, filters), batched))
            return _branch_major_fft(x, y, filters, n, ept, pair_coupling), True

        return product(a, b, outer)
    if field_stride:
        register(True)
    else:
        register()
    fp64 = promote or a.dtype in (jnp.float64, jnp.complex128)
    dtype = (jnp.float64 if fp64 else jnp.float32) if inverse else (jnp.complex128 if fp64 else jnp.complex64)
    if normalization is not None:
        dtype = jnp.float32
    shape = (*a.shape[:-2], a.shape[-1], n // 2 + 1) if channels_last else (*a.shape[:-1], n if inverse else n // 2 + 1)
    result = jax.ShapeDtypeStruct(shape, dtype)
    layouts = (tuple(range(a.ndim - 2)) + (a.ndim - 1, a.ndim - 2), None, None) if channels_last else None
    return jax.ffi.ffi_call("dno_cufftdx_fieldstride" if field_stride else "dno_cufftdx", result,
                          vmap_method="broadcast_all", input_layouts=layouts)(
        a, normalization if normalization is not None else a if b is None else b, a if outer is None else outer,
        size=np.int64(n), operation=np.int64(12 if field_stride else 11 if channels_last else 9 if promote else 8 if normalization is not None else 4 if outer is not None else inverse),
        multiply=b is not None, ept=np.int64(ept),
    )


def field_irfft(
    inputs_hat: jax.Array, surface_ops: jax.Array, filters: jax.Array, *, n: int, ept: int = 8,
    normalized_inputs: jax.Array | None = None, balanced: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Form fields on load; optionally emit surface features [B,7,N] and branch fields."""
    register()
    if normalized_inputs is not None:
        results = (jax.ShapeDtypeStruct((*normalized_inputs.shape[:-2], surface_ops.shape[-2] + 3, n), jnp.float32),
                   jax.ShapeDtypeStruct((*normalized_inputs.shape[:-2], filters.shape[-2], n), jnp.float32))
        features, branches = jax.ffi.ffi_call("dno_cufftdx_features", results, vmap_method="broadcast_all")(
            inputs_hat, surface_ops, filters, normalized_inputs, ept=np.int64(ept), balanced=balanced,
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


def branch_sandwich(
    xi_hat: jax.Array, weights: jax.Array, filters: jax.Array, *, ept: int = 8,
) -> jax.Array:
    """Use unnormalized xi_hat[B,F] and real weights[B,N,R]/filters[B,F,R].

    Return each branch's forward-normalized output spectrum with shape [B,R,F].
    """
    register()
    result = jax.ShapeDtypeStruct((*xi_hat.shape[:-1], weights.shape[-1], xi_hat.shape[-1]), xi_hat.dtype)
    return jax.ffi.ffi_call("dno_cufftdx", result, vmap_method="broadcast_all")(
        xi_hat, weights, filters, size=np.int64(weights.shape[-2]), operation=np.int64(5),
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
    parser.add_argument("--field-stride", action="store_true", help="Build/check the isolated strided-field product FFT.")
    parser.add_argument("--branch-major", action="store_true", help="Build/check the isolated branch-major product FFT against strided fields.")
    parser.add_argument("--pair-coupling", action="store_true", help="Build/check the isolated FP32 EPT8 symmetric-pair product FFT.")
    parser.add_argument("--bernoulli", action="store_true", help="Build the isolated FP64 Bernoulli-to-forward-FFT trial.")
    parser.add_argument("--sm", type=int, default=89)
    parser.add_argument("--ept", nargs="+", type=int, choices=(4, 8, 16), default=[8])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--repeats", type=int, default=300)
    classical = parser.add_mutually_exclusive_group()
    classical.add_argument("--classical", action="store_true", help="Screen the same FFT backend in the complete padded M1–M6 recurrence.")
    classical.add_argument("--classical-fused", action="store_true", help="Fuse padded classical products with forward FFTs, retaining JAX inverse FFTs.")
    args = parser.parse_args()
    if args.pair_coupling and args.ept != [8]:
        parser.error("The pair-coupling trial uses EPT8.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.build or args.build_only:
        mathdx = next(OUTPUT.glob("nvidia-mathdx-*/nvidia/mathdx/*/include"))
        subprocess.run([
            shutil.which("nvcc") or "/usr/local/cuda-12.9/bin/nvcc", "-std=c++17", "-O3",
            f"-arch=sm_{args.sm}", f"-DDNO_SM={args.sm * 10}", "--shared",
            "-Xcompiler", "-fPIC", "-DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY",
            *(["-DDNO_PAIR_COUPLING_ONLY"] if args.pair_coupling else []),
            f"-I{mathdx}", f"-I{jax.ffi.include_dir()}",
            str(ROOT / "scripts/cufftdx_fft.cu"), "-o", str(OUTPUT / ("cufftdx_paircoupling.so" if args.pair_coupling else "cufftdx_bernoulli.so" if args.bernoulli else "cufftdx_branchmajor.so" if args.branch_major else "cufftdx_fieldstride.so" if args.field_stride else "cufftdx.so")),
        ], check=True)
    if args.build_only:
        raise SystemExit(0)
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(42)
    records = {}
    if args.pair_coupling:
        for batch, paired, active, source_branches in (
            (batch, paired, active, source_branches)
            for batch in (1, 2, 32) for paired in (False, True)
            for active in (False, True) for source_branches in (28, 32)
        ):
            prefix = (2, batch) if paired else (batch,)
            weights = jnp.asarray(rng.normal(size=(*prefix, 1024, 42)), dtype=jnp.float32)
            if not active:
                weights = weights.at[..., 28:].set(0)
            fields = jnp.asarray(rng.normal(size=(*prefix, source_branches, 1024)), dtype=jnp.float32)
            filters = jnp.asarray(rng.normal(size=(batch, 28, 513)), dtype=jnp.float32)

            def pair_reference(x: jax.Array, y: jax.Array) -> jax.Array:
                coefficients = jnp.moveaxis(x, -1, -2)
                weighted = (coefficients[..., :28, :] * y[..., :28, :]
                            + jnp.repeat(coefficients[..., 28:, :], 2, axis=-2)
                            * y[..., jnp.arange(28) ^ 1, :])
                return filters * jnp.fft.rfft(weighted)

            def pair_candidate(x: jax.Array, y: jax.Array) -> jax.Array:
                return fft(x, 1024, b=y, outer=filters, channels_last=True,
                           field_stride=True, pair_coupling=True)

            candidate = jax.vmap(pair_candidate) if paired else pair_candidate
            reference = jax.jit(pair_reference)(weights, fields)
            actual = jax.jit(candidate)(weights, fields)
            relative = float(jnp.linalg.norm(actual - reference) / jnp.linalg.norm(reference))
            np.testing.assert_allclose(actual, reference, rtol=2e-6,
                                       atol=2e-6 * float(jnp.max(jnp.abs(reference))))
            if not active:
                diagonal = lambda x, y: fft(  # noqa: E731
                    x[..., :28], 1024, b=y, outer=filters, channels_last=True,
                    field_stride=True, branch_major=True)
                expected = jax.jit(jax.vmap(diagonal) if paired else diagonal)(weights, fields)
                np.testing.assert_array_equal(actual, expected)
            key = f"batch{batch}_{'paired' if paired else 'direct'}_{'active' if active else 'zero'}_fields{source_branches}"
            records[key] = {"relative_l2": relative}
            print(key, json.dumps(records[key]), flush=True)
        (args.output / "paircoupling_native.json").write_text(json.dumps(records, indent=2) + "\n")
        raise SystemExit(0)
    if args.field_stride or args.branch_major:
        for batch in (1, 2, 32):
            filters = jnp.asarray(rng.normal(size=(batch, 28, 513)), dtype=jnp.float32)
            for stage_pair in (False, True):
                weights = jnp.asarray(rng.normal(size=(batch, 1024, 28)), dtype=jnp.float32)
                fields = jnp.asarray(rng.normal(size=(batch, 32, 1024)), dtype=jnp.float32)
                if stage_pair:
                    weights = jnp.asarray(rng.normal(size=(2, batch, 1024, 28)), dtype=jnp.float32)
                    fields = jnp.asarray(rng.normal(size=(2, batch, 32, 1024)), dtype=jnp.float32)
                for ept in args.ept:
                    functions = []
                    for trial in (False, True):
                        strided = args.branch_major or trial
                        candidate = lambda x, y, strided=strided, trial=trial: fft(  # noqa: E731
                            x, 1024, b=y if strided else y[..., :28, :], outer=filters,
                            ept=ept, channels_last=True, field_stride=strided,
                            branch_major=args.branch_major and trial,
                        )
                        if stage_pair:
                            candidate = jax.vmap(candidate)
                        functions.append(jax.jit(candidate).lower(weights, fields).compile())
                    np.testing.assert_array_equal(np.asarray(functions[0](weights, fields)),
                                                  np.asarray(functions[1](weights, fields)))
                    name = f"{'branchmajor' if args.branch_major else 'fieldstride'}_batch{batch}_{'paired' if stage_pair else 'direct'}_e{ept}"
                    records[name] = measure(functions, weights, fields, args.repeats)
                    print(name, json.dumps(records[name]), flush=True)
        (args.output / ("branchmajor_native.json" if args.branch_major else "fieldstride_native.json")).write_text(json.dumps(records, indent=2) + "\n")
        raise SystemExit(0)
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
            branch_filters = outer.real.T[None]
            for ept in args.ept:
                cases[f"outer_product_rfft_e{ept}"] = (
                    lambda x, y: outer * jnp.fft.rfft(x * y, norm="forward"),
                    lambda x, y, ept=ept: fft(x, n, b=y, outer=outer, ept=ept), a, b,
                )
                cases[f"branch_sandwich_e{ept}"] = (
                    lambda x, y: jnp.moveaxis(branch_filters * jnp.fft.rfft(
                        y * jnp.fft.irfft(branch_filters * x[..., None], n=n, axis=1),
                        axis=1, norm="forward"), 1, -1),
                    lambda x, y, ept=ept: branch_sandwich(x, y, branch_filters, ept=ept), spectrum[:1], b.T[None],
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
                    weights = jnp.asarray(rng.normal(size=(batch, n, 32)), dtype=jnp.float32)
                    branch_fields = jnp.asarray(rng.normal(size=(batch, 32, n)), dtype=jnp.float32)

                    def product_reference(x: jax.Array, y: jax.Array) -> jax.Array:
                        return field_filters * jnp.fft.rfft(jnp.moveaxis(x, -1, -2) * y)

                    def product_candidate(x: jax.Array, y: jax.Array, ept: int = ept) -> jax.Array:
                        return fft(x, n, b=y, outer=field_filters, ept=ept, channels_last=True)

                    if stage_pair:
                        weights, branch_fields = jnp.stack((weights, -weights)), jnp.stack((branch_fields, -branch_fields))
                    cases[f"head_product_{'vmap' if stage_pair else 'direct'}_e{ept}"] = (
                        jax.vmap(product_reference) if stage_pair else product_reference,
                        jax.vmap(product_candidate) if stage_pair else product_candidate, weights, branch_fields,
                    )
                    for output in (0, 1):
                        def feature_reference(x: jax.Array, y: jax.Array, ept: int = ept, output: int = output) -> jax.Array:
                            fields = cast(jax.Array, field_irfft(x, surface_ops, field_filters, n=n, ept=ept))
                            eta = y[:, 0]
                            return (jnp.concatenate((jnp.stack((eta, eta ** 2, eta ** 3), axis=1), fields[:, :4]), axis=1)
                                    if output == 0 else fields[:, 4:])

                        def feature_candidate(x: jax.Array, y: jax.Array, ept: int = ept,
                                              output: int = output, balanced: bool = False) -> jax.Array:
                            return field_irfft(x, surface_ops, field_filters, n=n, ept=ept,
                                               normalized_inputs=y, balanced=balanced)[output]

                        inputs, normalized = input_pair.astype(jnp.complex64), jnp.stack((a, b), axis=1).astype(jnp.float32)
                        if stage_pair:
                            inputs, normalized = jnp.stack((inputs, -inputs)), jnp.stack((normalized, -normalized))
                        for balanced in (False, True):
                            candidate = partial(feature_candidate, balanced=balanced)
                            cases[f"feature_irfft_{output}_{'vmap' if stage_pair else 'direct'}_{'balanced' if balanced else 'original'}_e{ept}"] = (
                                jax.vmap(feature_reference) if stage_pair else feature_reference,
                                jax.vmap(candidate) if stage_pair else candidate, inputs, normalized,
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
