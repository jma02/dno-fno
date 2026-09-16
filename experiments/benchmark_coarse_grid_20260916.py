"""Compare full-grid CS-DNO with a coarse learned correction on one H100."""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("c27-coarse-grid-benchmark")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.4.2", "jax[cuda12]==0.9.2")
    .pip_install("flax==0.12.6", "optax==0.2.5")
)
for source in (
    "models/dno-net/dno_net_v2.py", "models/fno-jax/losses.py",
    "train-jax-10m/mode_balanced_regularizer.py",
    "train-jax-10m/translation_tangent_regularizer.py",
):
    image = image.add_local_file(ROOT / source, f"/bench/{Path(source).name}")


@app.function(
    image=image, gpu="H100:1", cpu=4, memory=32768,
    timeout=600, retries=0, scaledown_window=2,
)
def benchmark() -> str:
    import json
    import os
    import statistics
    import subprocess
    import sys
    from datetime import datetime, timezone
    from time import perf_counter

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, "/bench")
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training.train_state import TrainState

    from dno_net_v2 import CraigSulemDNO
    from losses import relative_l2_loss
    from mode_balanced_regularizer import compute_mode_balanced_loss
    from translation_tangent_regularizer import compute_translation_tangent_loss

    jax.config.update("jax_enable_x64", True)
    started = perf_counter()

    def resample(field: jax.Array, size: int) -> jax.Array:
        source_size = field.shape[1]
        if source_size == size:
            return field
        spectrum = jnp.fft.rfft(field, axis=1, norm="forward")
        if size < source_size:
            spectrum = spectrum[:, :size // 2 + 1]
            spectrum = spectrum.at[:, -1].set(2 * spectrum[:, -1].real)
        else:
            spectrum = spectrum.at[:, -1].multiply(0.5)
        return jnp.fft.irfft(spectrum, n=size, axis=1, norm="forward")

    class LearnedCorrection(CraigSulemDNO):
        def _linear_baseline(
            self, xi_norm: jax.Array, depth: jax.Array,
        ) -> jax.Array:
            return jnp.zeros_like(xi_norm)

        def _g1_baseline(
            self, eta_norm: jax.Array, xi_norm: jax.Array, depth: jax.Array,
        ) -> jax.Array:
            return jnp.zeros_like(eta_norm)

    # Check Fourier normalization and the even-grid Nyquist merge/split.
    x = jnp.arange(1024, dtype=jnp.float32) * (2 * jnp.pi / 1024)
    probe = (jnp.sin(3 * x) + jnp.cos(17 * x) + jnp.cos(128 * x))[None, :, None]
    roundtrip_error = float(jnp.max(jnp.abs(resample(resample(probe, 256), 1024) - probe)))
    assert roundtrip_error < 1e-4, roundtrip_error

    keys = jax.random.split(jax.random.key(42), 3)
    raw = jax.random.normal(keys[0], (512, 1024, 2), dtype=jnp.float32)
    spectrum = jnp.fft.rfft(raw, axis=1)
    k_rfft = jnp.arange(513, dtype=jnp.float32)
    spectrum *= jnp.where(k_rfft <= 128, 1 / (1 + k_rfft)**2, 0)[None, :, None]
    inputs = jnp.fft.irfft(spectrum, n=1024, axis=1)
    inputs -= jnp.mean(inputs, axis=1, keepdims=True)
    inputs *= jnp.array([0.01, 0.1], dtype=jnp.float32) / jnp.sqrt(
        jnp.mean(inputs**2, axis=1, keepdims=True)
    )
    depth = jax.random.uniform(keys[1], (512, 1), dtype=jnp.float32, minval=-1, maxval=1)
    targets = (inputs[:, :, 1] + 0.1 * inputs[:, :, 0])[..., None]
    k = jnp.fft.fftfreq(1024, d=1 / 1024).astype(jnp.float32)
    tanaka_mask = jnp.arange(512) % 4 == 0
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    result: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "gpu": gpu,
        "jax_version": jax.__version__, "batch_per_gpu": 512, "input_grid": 1024,
        "width": 640, "blocks": 8, "multiplier_hidden": 160,
        "matmul_precision": jax.config.jax_default_matmul_precision,
        "dtype": "float32 learned path, float64 analytic G1", "seed": 42,
        "scope": "Synthetic resident data; full forward and parameter-gradient + AdamW step. Original relative-L2, mode-balanced (weight 6), Tanaka-masked tangent (weight 10) losses at 1024 points. Unit normalizers. Excludes periodic Hadamard, multi-GPU communication, data loading, checkpoints, compilation.",
        "resampling": "Fourier resampling with even-grid Nyquist merge/split; included in compiled timings. Analytic G0+G1 stays at 1024. Learned correction interpolated back to 1024.",
        "roundtrip_max_abs_error": roundtrip_error,
        "warmups": 3, "trials": 21,
        "parameter_initialization": "Seeded random weights; replace zero phi kernels with small random values so all branches have nonzero gradients. No trained checkpoint or accuracy claim.",
    }
    print(f"GPU: {gpu}", flush=True)
    compiled = []
    for grid, latent in ((1024, 320), (256, 320), (256, 256)):
        name = f"grid{grid}_latent{latent}"
        model = CraigSulemDNO(width=640, n_blocks=8, latent=latent, mult_hidden=160)
        correction = LearnedCorrection(width=640, n_blocks=8, latent=latent, mult_hidden=160)
        params = model.init(keys[2], inputs[:1, :256], depth[:1])["params"]
        for block in range(8):
            kernel = params[f"cs_block_{block}"]["phi_proj"]["kernel"]
            params[f"cs_block_{block}"]["phi_proj"]["kernel"] = (
                jax.random.normal(jax.random.fold_in(keys[2], block), kernel.shape, kernel.dtype)
                * (0.01 / 320**0.5)
            )

        def forward(parameters: dict, fields: jax.Array, depths: jax.Array) -> jax.Array:
            variables = {"params": parameters}
            if grid == 1024:
                return model.apply(variables, fields, depths)
            baseline = model.apply(variables, fields[:, :, 1], depths, method=model._linear_baseline)
            baseline += model.apply(
                variables, fields[:, :, 0], fields[:, :, 1], depths, method=model._g1_baseline,
            )
            learned = correction.apply(variables, resample(fields, grid), depths)
            return baseline[..., None] + resample(learned, 1024)

        # At full resolution the residual-only wrapper must reproduce the original model.
        if grid == 1024:
            variables = {"params": params}
            reference = model.apply(variables, inputs[:2], depth[:2])
            learned = correction.apply(variables, inputs[:2], depth[:2])
            baseline = model.apply(variables, inputs[:2, :, 1], depth[:2], method=model._linear_baseline)
            baseline += model.apply(
                variables, inputs[:2, :, 0], inputs[:2, :, 1], depth[:2], method=model._g1_baseline,
            )
            wrapper_error = float(jnp.max(jnp.abs(reference - baseline[..., None] - learned)))
            assert wrapper_error < 1e-5, wrapper_error
            result["full_grid_wrapper_max_abs_error"] = wrapper_error

        def train_step(
            state: TrainState, fields: jax.Array, depths: jax.Array, target: jax.Array,
        ) -> tuple[TrainState, jax.Array]:
            def loss(parameters: dict) -> jax.Array:
                prediction = forward(parameters, fields, depths)
                physical_depth = jnp.exp(jnp.minimum(depths[:, 0], jnp.log(5.0)))
                mode_loss = compute_mode_balanced_loss(
                    fields[:, :, 0], prediction[..., 0], target[..., 0], physical_depth, k_rfft,
                )
                tangent_loss, _ = compute_translation_tangent_loss(
                    fields[:, :, 0], prediction[..., 0], target[..., 0], physical_depth, k,
                    sample_mask=tanaka_mask,
                )
                return relative_l2_loss(prediction, target) + 6 * mode_loss + 10 * tangent_loss

            value, gradients = jax.value_and_grad(loss)(state.params)
            return state.apply_gradients(grads=gradients), value

        state = TrainState.create(
            apply_fn=model.apply, params=params,
            tx=optax.adamw(learning_rate=2e-5, weight_decay=1e-4),
        )
        record: dict[str, object] = {
            "learned_grid": grid, "latent": latent,
            "parameter_count": sum(leaf.size for leaf in jax.tree.leaves(params)),
            "one_branch_group_activation_bytes": 512 * grid * latent * 4,
        }
        for stage, function, arguments in (
            ("forward", forward, (params, inputs, depth)),
            ("train_step", train_step, (state, inputs, depth, targets)),
        ):
            before = perf_counter()
            executable = jax.jit(function).lower(*arguments).compile()
            memory = executable.memory_analysis()
            stats = {
                "compile_seconds": perf_counter() - before,
                "compiler_memory_analysis": str(memory),
                "compiler_total_buffer_bytes": memory.argument_size_in_bytes
                + memory.output_size_in_bytes + memory.temp_size_in_bytes - memory.alias_size_in_bytes,
                "compiler_cost_analysis": executable.cost_analysis(),
            }
            output = jax.block_until_ready(executable(*arguments))
            assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(output)), name
            if stage == "train_step":
                assert float(optax.global_norm(jax.tree.map(
                    lambda new, old: new - old, output[0].params, params,
                ))) > 0, "No parameter update"
                stats["initial_loss"] = float(output[1])
            del output
            for _ in range(3):
                jax.block_until_ready(executable(*arguments))
            record[stage] = stats
            compiled.append((name, stage, executable, arguments, []))
            print(f"Compiled {name}/{stage}: {stats['compile_seconds']:.2f}s, buffers={stats['compiler_total_buffer_bytes'] / 2**30:.3f}GiB", flush=True)
        result[name] = record

    # Alternate traversal direction to reduce systematic clock/thermal ordering bias.
    for trial in range(21):
        for name, stage, executable, arguments, timings in (compiled if trial % 2 == 0 else compiled[::-1]):
            before = perf_counter()
            jax.block_until_ready(executable(*arguments))
            timings.append((perf_counter() - before) * 1000)
    for name, stage, _, _, timings in compiled:
        result[name][stage].update(median_ms=statistics.median(timings), trials_ms=timings)
        print(f"{name}/{stage}: {statistics.median(timings):.3f}ms", flush=True)
    result["function_wall_seconds"] = perf_counter() - started
    return json.dumps(result, indent=2)
