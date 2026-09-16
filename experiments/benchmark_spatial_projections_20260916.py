"""Isolated C27 spatial projections on one H100; synthetic device-resident inputs."""

from __future__ import annotations

import modal

app = modal.App("c27-spatial-projection-benchmark")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy==2.4.2", "jax[cuda12]==0.9.2"
)


@app.function(
    image=image, gpu="H100:1", cpu=4, memory=32768,
    timeout=300, retries=0, scaledown_window=2,
)
def benchmark() -> str:
    import json
    import os
    import statistics
    import subprocess
    from datetime import datetime, timezone
    from time import perf_counter

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    started = perf_counter()
    weights_type = tuple[jax.Array, ...]

    def separate(x: jax.Array, weights: weights_type) -> weights_type:
        return tuple(x @ weight for weight in weights)

    def packed(x: jax.Array, weights: weights_type) -> weights_type:
        return tuple(jnp.split(x @ jnp.concatenate(weights, axis=-1), 8, axis=-1))

    def separate_training(
        x: jax.Array, weights: weights_type, cotangents: weights_type,
    ) -> tuple[weights_type, tuple[jax.Array, weights_type]]:
        outputs, backward = jax.vjp(separate, x, weights)
        return outputs, backward(cotangents)

    def packed_training(
        x: jax.Array, weights: weights_type, cotangents: weights_type,
    ) -> tuple[weights_type, tuple[jax.Array, weights_type]]:
        outputs, backward = jax.vjp(packed, x, weights)
        return outputs, backward(cotangents)

    @jax.jit
    def error(left: jax.Array, right: jax.Array) -> tuple[jax.Array, ...]:
        difference = left - right
        relative = jnp.sqrt(jnp.sum(difference**2) / jnp.maximum(jnp.sum(left**2), 1e-30))
        return relative, jnp.max(jnp.abs(difference)), jnp.all(jnp.isfinite(right))

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    print(f"GPU: {gpu}; JAX {jax.__version__}; local batch=512, grid=1024, width=320, blocks=8", flush=True)
    keys = jax.random.split(jax.random.key(42), 17)
    x = jax.random.normal(keys[0], (512, 1024, 320), dtype=jnp.float32)
    weights = tuple(jax.random.normal(key, (320, 320), dtype=jnp.float32) / 320**0.5 for key in keys[1:9])
    cotangents = tuple(jax.random.normal(key, x.shape, dtype=jnp.float32) for key in keys[9:17])
    jax.block_until_ready((x, weights, cotangents))
    result: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu, "jax_version": jax.__version__, "dtype": "float32",
        "jax_enable_x64": True, "matmul_precision": jax.config.jax_default_matmul_precision,
        "batch": 512, "grid": 1024, "input_width": 320, "output_width_per_block": 320,
        "blocks": 8, "seed": 42, "warmups_per_variant": 3, "paired_trials": 31,
        "scope": "Projections only. Outputs and gradients w.r.t. shared features and all eight weights; no FFT, optimizer, data loading, or checkpointing.",
        "packed_layout": "Concatenate original weight leaves inside JIT; return eight output arrays as in the original model.",
    }
    for stage, functions, arguments in (
        ("forward", (separate, packed), (x, weights)),
        ("forward_backward", (separate_training, packed_training), (x, weights, cotangents)),
    ):
        executables = []
        compile_times = []
        compiler_stats = []
        for variant, function in zip(("separate", "packed"), functions):
            before = perf_counter()
            executable = jax.jit(function).lower(*arguments).compile()
            compile_times.append(perf_counter() - before)
            executables.append(executable)
            hlo = executable.as_text()
            compiler_stats.append({
                "cost_analysis": executable.cost_analysis(),
                "memory_analysis": str(executable.memory_analysis()),
                "gemm_hlo": [line.strip() for line in hlo.splitlines() if "custom_call_target=" in line and ("gemm" in line.lower() or "cublas" in line.lower())],
            })
            print(f"Compiled {stage}/{variant} in {compile_times[-1]:.2f}s", flush=True)

        reference, candidate = (jax.block_until_ready(executable(*arguments)) for executable in executables)
        comparisons = [
            tuple(map(float, jax.device_get(error(left, right))))
            for left, right in zip(jax.tree.leaves(reference), jax.tree.leaves(candidate))
        ]
        worst_relative = max(value[0] for value in comparisons)
        worst_absolute = max(value[1] for value in comparisons)
        assert all(value[2] and value[0] < 1e-3 for value in comparisons), comparisons
        del reference, candidate
        for _ in range(3):
            for executable in executables:
                jax.block_until_ready(executable(*arguments))

        timings: list[list[float]] = [[], []]
        for trial in range(31):
            for index in ((0, 1) if trial % 2 == 0 else (1, 0)):
                before = perf_counter()
                jax.block_until_ready(executables[index](*arguments))
                timings[index].append((perf_counter() - before) * 1000)
        medians = list(map(statistics.median, timings))
        result[stage] = {
            "separate_median_ms": medians[0], "packed_median_ms": medians[1],
            "speedup": medians[0] / medians[1],
            "paired_speedup_median": statistics.median(left / right for left, right in zip(*timings)),
            "separate_ms": timings[0], "packed_ms": timings[1],
            "worst_relative_l2": worst_relative, "worst_absolute_error": worst_absolute,
            "comparisons": comparisons, "compile_seconds": compile_times,
            "compiler_stats": compiler_stats,
        }
        print(f"{stage}: separate={medians[0]:.3f}ms packed={medians[1]:.3f}ms speedup={medians[0]/medians[1]:.3f}x max_rel_l2={worst_relative:.3g}", flush=True)
    result["function_wall_seconds"] = perf_counter() - started
    return json.dumps(result, indent=2)
