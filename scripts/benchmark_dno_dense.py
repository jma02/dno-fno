"""Benchmark-only surface-network GEMMs with explicit precision and fused epilogues."""
from __future__ import annotations

import argparse
import ctypes
from functools import cache, partial
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Literal, cast

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.custom_batching import custom_vmap  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import triton as plt  # noqa: E402
from jax.ref import Ref  # noqa: E402
import numpy as np  # noqa: E402

Activation = Literal["none", "gelu", "gelu_lift"]
TRUNK_SOURCE = Path(__file__).resolve().parent / "dno_trunk_head.cu"
TRUNK_LIBRARY = Path(__file__).resolve().parents[1] / "outputs/dno_trunk_20261026/dno_trunk_head.so"
CONTEXT_SOURCE = Path(__file__).resolve().parent / "dno_context_filter.cu"
CONTEXT_LIBRARY = TRUNK_LIBRARY.with_name("dno_context_filter.so")


def dense(
    x: jax.Array, weight: jax.Array, activation: Activation = "none", *,
    tile: tuple[int, int, int] = (32, 32, 32), num_warps: int = 4,
    interpret: bool = False, residual: jax.Array | None = None, input_scale: float = 1.0,
    input_transposed: bool = False, precision: jax.lax.Precision = jax.lax.Precision.HIGHEST,
) -> jax.Array:
    """One non-split-K GEMM and optional epilogue, preserving leading dimensions."""
    rows, inner, columns = x.size // x.shape[-1], x.shape[-1], weight.shape[-1]
    bm, bn, bk = tile

    def kernel(x_ref: Ref, w_ref: Ref, residual_ref: Ref | None, out_ref: Ref) -> None:
        m = pl.program_id(0) * bm + jnp.arange(bm)
        n = pl.program_id(1) * bn + jnp.arange(bn)
        accumulator = jnp.zeros((bm, bn), dtype=jnp.float32)
        for offset in range(0, inner, bk):
            k = offset + jnp.arange(bk)
            a = plt.load(x_ref.at[k[None, :], m[:, None]] if input_transposed
                         else x_ref.at[m[:, None], k[None, :]],
                         mask=(m[:, None] < rows) & (k[None, :] < inner), other=0)
            if input_scale != 1.0:
                a *= jnp.float32(input_scale)
            b = plt.load(w_ref.at[k[:, None], n[None, :]],
                         mask=(k[:, None] < inner) & (n[None, :] < columns), other=0)
            accumulator += pl.dot(a, b, precision=precision)
        if residual is not None:
            skip = plt.load(cast(Ref, residual_ref).at[m[:, None], n[None, :]],
                            mask=(m[:, None] < rows) & (n[None, :] < columns), other=0)
            if interpret:
                accumulator += skip
            else:
                # Keep the rounded dot result: Triton otherwise folds the skip into its accumulator.
                accumulator, = plt.elementwise_inline_asm(
                    "add.rn.f32 $0, $1, $2;", args=(accumulator, skip),
                    constraints="=f,f,f", pack=1,
                    result_shape_dtypes=(jax.ShapeDtypeStruct(accumulator.shape, jnp.float32),),
                )
        if activation != "none":
            accumulator = jax.nn.gelu(accumulator)
        if activation == "gelu_lift":
            accumulator *= jnp.tanh(accumulator)
        plt.store(out_ref.at[m[:, None], n[None, :]], accumulator,
                  mask=(m[:, None] < rows) & (n[None, :] < columns))

    output = pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((rows, columns), jnp.float32),
        grid=(pl.cdiv(rows, bm), pl.cdiv(columns, bn)), interpret=interpret,
        compiler_params=plt.CompilerParams(num_warps=num_warps, num_stages=1),
        name=f"dno_dense_{inner}_{columns}_{activation}",
    )(x.reshape(rows, inner).T if input_transposed else x.reshape(rows, inner),
      weight, None if residual is None else residual.reshape(rows, columns))
    return output.reshape(*x.shape[:-1], columns)


def _xla_tanh_ptx(x: str, out: str) -> str:
    """XLA:GPU's F32 tanh as emitted for the hero: clamp, small-input passthrough, rational form, div.full."""
    return f"""
    setp.ge.f32 p0, {x}, 0f40FFF644;
    selp.f32 c, 0f40FFF644, {x}, p0;
    setp.le.f32 p0, c, 0fC0FFF644;
    selp.f32 c, 0fC0FFF644, c, p0;
    abs.f32 s, {x};
    setp.lt.f32 p1, s, 0f39D1B717;
    mul.rn.f32 s, c, c;
    fma.rn.f32 n, s, 0fA59F25C0, 0f2A61337E;
    fma.rn.f32 n, s, n, 0fAEBD37FF;
    fma.rn.f32 n, s, n, 0f335C0041;
    fma.rn.f32 n, s, n, 0f3779434A;
    fma.rn.f32 n, s, n, 0f3A270DED;
    fma.rn.f32 n, s, n, 0f3BA059DC;
    mul.rn.f32 n, c, n;
    fma.rn.f32 d, s, 0f35A0D3D8, 0f38F895D6;
    fma.rn.f32 d, s, d, 0f3B14AA05;
    fma.rn.f32 d, s, d, 0f3BA059DD;
    div.full.f32 n, n, d;
    selp.f32 {out}, c, n, p1;
    """


# jax.nn.gelu followed by g * tanh(g), instruction for instruction as XLA emits the accepted fusion.
_GELU_LIFT_PTX = "{ .reg .f32 z, c, s, n, d, t, g; .reg .pred p0, p1;" + f"""
    mul.rn.f32 z, $1, $1;
    mul.rn.f32 z, $1, z;
    mul.rn.f32 z, z, 0f3D372713;
    add.rn.f32 z, $1, z;
    mul.rn.f32 z, z, 0f3F4C422A;
    {_xla_tanh_ptx("z", "t")}
    add.rn.f32 t, t, 0f3F800000;
    mul.rn.f32 t, t, 0f3F000000;
    mul.rn.f32 g, $1, t;
    {_xla_tanh_ptx("g", "t")}
    mul.rn.f32 $0, g, t;
""" + "}"


def _ptx(asm: str, *args: jax.Array) -> jax.Array:
    result, = plt.elementwise_inline_asm(
        asm, args=args, constraints="=f" + ",f" * len(args), pack=1,
        result_shape_dtypes=(jax.ShapeDtypeStruct(args[0].shape, jnp.float32),),
    )
    return result


def fused_tail(
    context: jax.Array, h1: jax.Array, w_expand: jax.Array, w_mix: jax.Array, w_head: jax.Array, *,
    mix_rows: int = 128, mix_warps: int = 4, head_rows: int = 64, head_warps: int = 4, stages: int = 3,
) -> jax.Array:
    """Context expansion, residual, mix GEMM and GELU lift in one kernel; the split-K=2 head in another.

    context [S*C, N] is the unnormalized inverse FFT, h1 [S*N, W]; returns head weights [B, S*N].
    Dots are TF32 and accumulate in increasing k like the accepted XLA kernels.
    """
    channels = w_expand.shape[0]
    points = context.shape[1]
    rows, width = h1.shape
    branches = w_head.shape[-1]
    chunk, k_pad, b_pad = 32, 16, 32
    split = width // 2
    dot = partial(pl.dot, precision=jax.lax.Precision.DEFAULT)

    def mix_kernel(ctx_ref: Ref, h1_ref: Ref, we_ref: Ref, wm_ref: Ref, g_ref: Ref) -> None:
        m = pl.program_id(0) * mix_rows + jnp.arange(mix_rows)
        n = pl.program_id(1) * chunk + jnp.arange(chunk)
        c = jnp.arange(k_pad)
        a = plt.load(ctx_ref.at[(m // points)[:, None] * channels + c[None, :], (m % points)[:, None]],
                     mask=c[None, :] < channels, other=0) * jnp.float32(1 / points)

        def step(i: jax.Array, mix: jax.Array) -> jax.Array:
            k = i * chunk + jnp.arange(chunk)
            we = plt.load(we_ref.at[c[:, None], k[None, :]], mask=c[:, None] < channels, other=0)
            # A separate add keeps the rounded expansion, as in the accepted residual fusion.
            h2 = _ptx("add.rn.f32 $0, $1, $2;", dot(a, we), plt.load(h1_ref.at[m[:, None], k[None, :]]))
            return mix + dot(h2, plt.load(wm_ref.at[k[:, None], n[None, :]]))

        mix = jax.lax.fori_loop(0, width // chunk, step, jnp.zeros((mix_rows, chunk), jnp.float32))
        plt.store(g_ref.at[m[:, None], n[None, :]], _ptx(_GELU_LIFT_PTX, mix))

    def head_kernel(g_ref: Ref, wh_ref: Ref, out_ref: Ref) -> None:
        m = pl.program_id(0) * head_rows + jnp.arange(head_rows)
        b = jnp.arange(b_pad)

        def step(i: jax.Array, sums: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            k = i * chunk + jnp.arange(chunk)
            g = plt.load(g_ref.at[m[:, None], k[None, :]])
            wh = plt.load(wh_ref.at[k[:, None], b[None, :]], mask=b[None, :] < branches, other=0)
            low = k[None, :] < split
            return sums[0] + dot(jnp.where(low, g, 0), wh), sums[1] + dot(jnp.where(low, 0, g), wh)

        zeros = jnp.zeros((head_rows, b_pad), jnp.float32)
        first, second = jax.lax.fori_loop(0, width // chunk, step, (zeros, zeros))
        plt.store(out_ref.at[b[None, :], m[:, None]], _ptx("add.rn.f32 $0, $1, $2;", first, second),
                  mask=b[None, :] < branches)

    g = pl.pallas_call(
        mix_kernel, out_shape=jax.ShapeDtypeStruct((rows, width), jnp.float32),
        grid=(pl.cdiv(rows, mix_rows), width // chunk),
        compiler_params=plt.CompilerParams(num_warps=mix_warps, num_stages=stages),
        name=f"dno_fused_mix_{width}",
    )(context, h1, w_expand, w_mix)
    return pl.pallas_call(
        head_kernel, out_shape=jax.ShapeDtypeStruct((branches, rows), jnp.float32),
        grid=(pl.cdiv(rows, head_rows),),
        compiler_params=plt.CompilerParams(num_warps=head_warps, num_stages=stages),
        name=f"dno_fused_head_{width}_{branches}",
    )(g, w_head)


def _stacked_tail(context: jax.Array, features: jax.Array, w_expand: jax.Array, w_mix: jax.Array,
                  w_head: jax.Array) -> jax.Array:
    """Flatten all leading axes so paired stages share one launch and branch-major output."""
    leading, points = context.shape[:-2], context.shape[-2]
    weights = fused_tail(jnp.moveaxis(context, -1, -2).reshape(-1, points),
                         features.reshape(-1, features.shape[-1]), w_expand, w_mix, w_head)
    return jnp.moveaxis(weights.reshape(-1, *leading, points), 0, -1)


@custom_vmap
def tail_weights(context: jax.Array, features: jax.Array, w_expand: jax.Array, w_mix: jax.Array,
                 w_head: jax.Array) -> jax.Array:
    """Head weights [..., N, B] from the inverse context FFT [..., N, C] and first activation [..., N, W]."""
    return _stacked_tail(context, features, w_expand, w_mix, w_head)


@tail_weights.def_vmap
def _tail_weights_vmap(axis_size: int, batched: list[bool], context: jax.Array, features: jax.Array,
                       w_expand: jax.Array, w_mix: jax.Array, w_head: jax.Array) -> tuple[jax.Array, bool]:
    if any(batched[2:]):
        raise ValueError("Fused tail weights must be shared across the mapped axis.")
    context, features = (value if mapped else jnp.broadcast_to(value, (axis_size, *value.shape))
                         for value, mapped in zip((context, features), batched))
    return _stacked_tail(context, features, w_expand, w_mix, w_head), True


def build_trunk(sm: int = 89) -> None:
    """Compile the CUDA trunk-head and context-filter FFI libraries; the latter needs MathDx cuFFTDx."""
    TRUNK_LIBRARY.parent.mkdir(parents=True, exist_ok=True)
    mathdx = next((TRUNK_LIBRARY.parents[1] / "cufftdx_20260925").glob("nvidia-mathdx-*/nvidia/mathdx/*/include"))
    nvcc = [shutil.which("nvcc") or "/usr/local/cuda-12.9/bin/nvcc", "-std=c++17", "-O3", f"-arch=sm_{sm}",
            "--shared", "-Xcompiler", "-fPIC", f"-I{jax.ffi.include_dir()}"]
    subprocess.run([*nvcc, str(TRUNK_SOURCE), "-o", str(TRUNK_LIBRARY)], check=True)
    subprocess.run([*nvcc, f"-DDNO_SM={sm * 10}", "-DCUFFTDX_DISABLE_CUTLASS_DEPENDENCY", f"-I{mathdx}",
                    str(CONTEXT_SOURCE), "-o", str(CONTEXT_LIBRARY)], check=True)


@cache
def _register_trunk() -> tuple[ctypes.CDLL, ctypes.CDLL]:
    head, context = ctypes.CDLL(str(TRUNK_LIBRARY)), ctypes.CDLL(str(CONTEXT_LIBRARY))
    jax.ffi.register_ffi_target("dno_trunk_head", jax.ffi.pycapsule(head.dno_trunk_head), platform="CUDA")
    jax.ffi.register_ffi_target("dno_context_filter", jax.ffi.pycapsule(context.dno_context_filter), platform="CUDA")
    return head, context


def trunk_head(
    features: jax.Array, normalized: jax.Array, fields: jax.Array, feature_scales: jax.Array, w_first: jax.Array,
    w_context: jax.Array, nested_scale: float, nested_from: int, tiles_per_warp: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """CUDA trunk head: first activation [..., N, W] and batched context-FFT input [..., C', N].

    features [..., F, N] are unscaled; normalized [..., 2, N] supplies eta_norm; fields [..., R, N].
    Divisions by constants are XLA reciprocal multiplies, so the reciprocals are rounded the same way.
    """
    _register_trunk()
    *leading, _, points = features.shape
    out_channels = w_context.shape[-1] + fields.shape[-2] - nested_from
    h1, spectrum_input = jax.ffi.ffi_call("dno_trunk_head", (
        jax.ShapeDtypeStruct((*leading, points, w_first.shape[-1]), jnp.float32),
        jax.ShapeDtypeStruct((*leading, out_channels, points), jnp.float32),
    ), vmap_method="expand_dims")(
        features, 1 / feature_scales.astype(jnp.float32), w_first, w_context, normalized, fields,
        nested_reciprocal=np.float32(1) / np.float32(nested_scale), nested_from=np.int64(nested_from),
        tiles_per_warp=np.int64(tiles_per_warp),
    )
    return h1, spectrum_input


def context_filter(
    spectrum_input: jax.Array, context_symbol: jax.Array, inner_filters: jax.Array, gamma: jax.Array,
    normalized: jax.Array, nested_scale: float, ept: int = 8,
) -> tuple[jax.Array, jax.Array]:
    """cuFFTDx context filter on the trunk-head FFT input [..., C+R, N].

    Returns the unnormalized inverse-filtered context [..., C, N] and the nested sandwich spectra
    [..., R, F] before the outer filters. context_symbol [F, C] and inner_filters [R, F] are real.
    """
    _register_trunk()
    *leading, _, points = spectrum_input.shape
    channels, n_nested = context_symbol.shape[-1], gamma.shape[-1]
    return tuple(jax.ffi.ffi_call("dno_context_filter", (
        jax.ShapeDtypeStruct((*leading, channels, points), jnp.float32),
        jax.ShapeDtypeStruct((*leading, n_nested, points // 2 + 1), jnp.complex64),
    ), vmap_method="expand_dims")(
        spectrum_input, context_symbol.astype(jnp.float32), inner_filters.astype(jnp.float32),
        gamma.astype(jnp.float32), normalized,
        nested_reciprocal=np.float32(1) / np.float32(nested_scale), ept=np.int64(ept),
    ))  # pyright: ignore[reportReturnType]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpret", action="store_true", help="CPU correctness only; no meaningful timings.")
    parser.add_argument("--build-trunk", action="store_true", help="Compile the CUDA trunk-head library and exit.")
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("outputs/cufftdx_20260925/dense_screen.json"))
    args = parser.parse_args()
    if args.build_trunk:
        build_trunk()
        raise SystemExit(0)
    rng = np.random.default_rng(42)
    records = {}
    cases: tuple[tuple[int, int, Activation], ...] = ((7, 160, "gelu"), (160, 160, "gelu_lift"), (160, 32, "none"))
    for inner, columns, activation in cases:
        x = jnp.asarray(rng.normal(size=(1, 1024, inner)), dtype=jnp.float32)
        weight = jnp.asarray(rng.normal(size=(inner, columns)) / np.sqrt(inner), dtype=jnp.float32)

        def reference(x: jax.Array, weight: jax.Array) -> jax.Array:
            y = jnp.matmul(x, weight, precision=jax.lax.Precision.HIGHEST)
            if activation != "none":
                y = jax.nn.gelu(y)
            return y * jnp.tanh(y) if activation == "gelu_lift" else y

        expected_fn = jax.jit(reference).lower(x, weight).compile()
        expected = np.asarray(expected_fn(x, weight))
        for tile, warps in (((32, 32, 32), 4), ((32, 32, 16), 4), ((32, 16, 32), 4),
                            ((16, 32, 32), 4), ((64, 16, 32), 4), ((32, 32, 32), 2)):
            candidate = jax.jit(partial(dense, activation=activation, tile=tile,
                                        num_warps=warps, interpret=args.interpret)).lower(x, weight).compile()
            actual = np.asarray(candidate(x, weight))
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
            record = {"tile": tile, "warps": warps,
                      "relative_l2": float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))}
            if not args.interpret:
                functions = (expected_fn, candidate)
                for f in functions:
                    for _ in range(50):
                        f(x, weight).block_until_ready()
                seconds: list[list[float]] = [[], []]
                for repeat in range(args.repeats):
                    for index in (0, 1) if repeat % 2 == 0 else (1, 0):
                        started = time.perf_counter()
                        functions[index](x, weight).block_until_ready()
                        seconds[index].append(time.perf_counter() - started)
                record.update(jax_us=float(np.median(seconds[0]) * 1e6),
                              fused_us=float(np.median(seconds[1]) * 1e6), seconds=seconds)
            name = f"{inner}x{columns}_{activation}_{tile}_w{warps}"
            records[name] = record
            print(name, json.dumps({key: value for key, value in record.items() if key != "seconds"}), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(records, indent=2) + "\n")
