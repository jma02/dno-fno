"""Benchmark-only IEEE FP32 surface-network GEMMs with fused activation epilogues."""
from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import time
from typing import Literal

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import triton as plt  # noqa: E402
from jax.ref import Ref  # noqa: E402
import numpy as np  # noqa: E402

Activation = Literal["none", "gelu", "gelu_lift"]


def dense(
    x: jax.Array, weight: jax.Array, activation: Activation = "none", *,
    tile: tuple[int, int, int] = (32, 32, 32), num_warps: int = 4,
    interpret: bool = False,
) -> jax.Array:
    """One non-split-K GEMM and optional epilogue, preserving leading dimensions."""
    rows, inner, columns = x.size // x.shape[-1], x.shape[-1], weight.shape[-1]
    bm, bn, bk = tile

    def kernel(x_ref: Ref, w_ref: Ref, out_ref: Ref) -> None:
        m = pl.program_id(0) * bm + jnp.arange(bm)
        n = pl.program_id(1) * bn + jnp.arange(bn)
        accumulator = jnp.zeros((bm, bn), dtype=jnp.float32)
        for offset in range(0, inner, bk):
            k = offset + jnp.arange(bk)
            a = plt.load(x_ref.at[m[:, None], k[None, :]],
                         mask=(m[:, None] < rows) & (k[None, :] < inner), other=0)
            b = plt.load(w_ref.at[k[:, None], n[None, :]],
                         mask=(k[:, None] < inner) & (n[None, :] < columns), other=0)
            accumulator += pl.dot(a, b, precision=jax.lax.Precision.HIGHEST)
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
    )(x.reshape(rows, inner), weight)
    return output.reshape(*x.shape[:-1], columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpret", action="store_true", help="CPU correctness only; no meaningful timings.")
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("outputs/cufftdx_20260925/dense_screen.json"))
    args = parser.parse_args()
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
