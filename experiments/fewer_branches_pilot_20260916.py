"""Bounded real-data pilots for branch reduction, FFT fusion and compact corrections."""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("c27-fewer-branches-pilot")
volume = modal.Volume.from_name("dno-fno-train-data")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.4.2", "jax[cuda12]==0.9.2")
    .pip_install("flax==0.12.6", "optax==0.2.5")
    .add_local_dir(ROOT / "models" / "dno-net", "/bench/models")
)
for source in (
    "models/fno-jax/losses.py", "train-jax-10m/mode_balanced_regularizer.py",
    "train-jax-10m/translation_tangent_regularizer.py",
):
    image = image.add_local_file(ROOT / source, f"/bench/{Path(source).name}")


@app.function(image=image, volumes={"/data": volume}, cpu=4, memory=8192, timeout=600, retries=0)
def prepare(run_name: str) -> dict:
    import json
    from time import perf_counter

    import numpy as np

    started = perf_counter()
    source = Path("/data/outputs/paper_dataset/arrays")
    destination = Path("/data/experiments") / run_name
    destination.mkdir(parents=True, exist_ok=False)
    splits = np.load(source / "dataset_split.npy", mmap_mode="r")
    families = np.load(source / "family_id.npy", mmap_mode="r")
    simulations = np.load(source / "simulation_id.npy", mmap_mode="r")
    rng = np.random.default_rng(42)
    indices = {}
    for split, per_family, chunk in (("train", 4096, 512), ("validation", 512, 128)):
        selected = []
        for family in (1, 2, 3, 4):
            candidates = np.flatnonzero((splits == split) & (families == family))
            if candidates.size < per_family:
                raise ValueError(f"Insufficient {split} family {family} rows: {candidates.size}")
            # Read a few contiguous regions rather than faulting thousands of remote pages.
            anchors = rng.integers(0, candidates.size - chunk + 1, size=32)
            pool = np.unique(np.concatenate([candidates[start:start + chunk] for start in anchors]))
            selected.append(rng.choice(pool, per_family, replace=False))
        indices[split] = np.sort(np.concatenate(selected))
    train_simulations = np.unique(simulations[indices["train"]])
    validation_simulations = np.unique(simulations[indices["validation"]])
    assert np.intersect1d(train_simulations, validation_simulations).size == 0
    payload = {}
    for name in ("eta", "xi", "gxi", "depth", "family_id"):
        array = np.load(source / f"{name}.npy", mmap_mode="r")
        for split, rows in indices.items():
            selected = np.asarray(array[rows])
            if name == "xi":
                selected = selected - selected.mean(axis=1, keepdims=True)
            assert np.isfinite(selected).all(), (split, name)
            payload[f"{split}_{name}"] = selected
        print(f"Prepared {name}", flush=True)
    stats = {
        "feature_absmax": [float(np.max(np.abs(payload[f"train_{name}"]))) for name in ("eta", "xi")],
        "target_absmax": float(np.max(np.abs(payload["train_gxi"]))),
    }
    x = np.load(source / "x.npy")
    metadata = {
        "source": str(source), "seed": 42, "stats": stats,
        "normalization": "Extrema fitted only on the pilot training subset, shared across variants.",
        "train_rows": len(indices["train"]), "validation_rows": len(indices["validation"]),
        "train_simulations": len(train_simulations), "validation_simulations": len(validation_simulations),
        "domain_length": float((x[1] - x[0]) * len(x)),
    }
    np.savez(destination / "pilot_data.npz", **payload, **{f"{key}_indices": value for key, value in indices.items()})
    metadata["prepare_seconds"] = perf_counter() - started
    (destination / "data_metadata.json").write_text(json.dumps(metadata, indent=2))
    volume.commit()
    return metadata


def run_pilot(run_name: str, *, fusion_only: bool, batch_sweep: bool = False,
              profile_step: bool = False, compact_pilot: bool = False,
              spectral_benchmark: bool = False, fno_benchmark: bool = False,
              fno_fold_benchmark: bool = False, fno_gemm_benchmark: bool = False,
              fno_transform_benchmark: bool = False) -> str:
    import json
    import os
    import statistics
    import subprocess
    import sys
    from time import perf_counter

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path[:0] = ["/bench", "/bench/models"]
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax import serialization
    from flax.training.train_state import TrainState

    from dno_net_v2 import CraigSulemDNO
    from fused_fft import check_fused_roundtrip
    from losses import relative_l2_loss
    from mode_balanced_regularizer import compute_mode_balanced_loss
    from translation_tangent_regularizer import compute_translation_tangent_loss

    started = perf_counter()
    deadline = started + (140 if fusion_only or batch_sweep or compact_pilot or spectral_benchmark else 540)
    jax.config.update("jax_enable_x64", True)
    volume.reload()
    destination = Path("/data/experiments") / run_name
    metadata = json.loads((destination / "data_metadata.json").read_text())
    data = np.load(destination / "pilot_data.npz")
    scales = np.asarray(metadata["stats"]["feature_absmax"], dtype=np.float32)
    scales = np.where(scales > 0, scales, 1)
    target_scale = max(metadata["stats"]["target_absmax"], 1e-12)
    datasets = {}
    for split in ("train", "validation"):
        fields = np.stack((data[f"{split}_eta"], data[f"{split}_xi"]), axis=-1) / scales
        batch_depth = np.log(np.maximum(data[f"{split}_depth"], 1e-12)).astype(np.float32)[:, None]
        targets = data[f"{split}_gxi"][..., None] / target_scale
        mask = data[f"{split}_family_id"] == 2
        datasets[split] = tuple(jax.device_put(array) for array in (fields, batch_depth, targets, mask))
    del data
    batch = tuple(array[:512] for array in datasets["train"])
    k = jnp.fft.fftfreq(1024, d=metadata["domain_length"] / (2 * np.pi * 1024)).astype(jnp.float32)
    k_rfft = jnp.abs(k[:513])
    key = jax.random.key(42)
    result = {
        "run_name": run_name, "data": metadata, "jax_version": jax.__version__,
        "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout.strip(),
        "scope": "Real-data paired learning pilot, local batch 512, learned grid 256, width 640, multiplier hidden 160. Relative-L2 + mode-balanced + Tanaka-tangent losses; excludes periodic Hadamard and long-rollout evaluation. Fresh zero-residual initialization; train-only subset normalization. GPU function hard cap 600s.",
        "variants": {}, "validation_history": [],
        "fusion_only": fusion_only,
    }
    if fusion_only:
        result["scope"] = "Paired fused/unfused verification and 32 training updates from the same 128-branch step-1024 pilot checkpoint. Restored states and batches are GPU resident. Same losses as the pilot; no Hadamard or long rollouts. GPU hard cap 180s."
    if batch_sweep:
        result["scope"] = "Batch-size throughput sweep of the trained 128-branch fused model on resident real data. Same losses and optimizer as the pilot; no Hadamard, data loading or convergence comparison. GPU hard cap 180s."
    if profile_step:
        result["scope"] = "Batch-4096 memory, compiler-cost and GPU-kernel trace of the trained 128-branch fused model. Same losses and optimizer as the pilot; no Hadamard or convergence comparison. GPU hard cap 180s. Compiler costs are estimates, not hardware counters."
    if compact_pilot:
        result["scope"] = "Fresh paired learning pilot: fused 128-branch reference versus compact Fourier ranks 32 and 64, compact hidden width 128. Same 16384/2048 real train/validation subset, normalization, batch-512 updates, sample order, optimizer and losses as the branch pilot. Also benchmark batch 4096 and validate shifted inputs/targets. No Hadamard or long rollouts. GPU hard cap 180s."
    if spectral_benchmark:
        result["scope"] = "Throughput only: fused 128-branch reference (shared320) versus FFT/global MLP/IFFT/decoder, hidden256 x4, IFFT channels16, decoder hidden64. Batch512 and4096, resident real pilot data, relative-L2/mode-balanced/Tanaka losses and AdamW. No learning, Hadamard, data loading or convergence claim. GPU hard cap180s."
    print(result["gpu"], flush=True)
    fused_ok = False
    try:
        result["fused_kernel_relative_errors"] = check_fused_roundtrip(interpret=False)
        fused_ok = True
        print(f"Fused kernel correctness: {result['fused_kernel_relative_errors']}", flush=True)
    except Exception as error:
        result["fusion_error"] = repr(error)
        print(f"Fusion unavailable; retaining cuFFT for training: {error}", flush=True)
        if fusion_only or batch_sweep or compact_pilot or spectral_benchmark:
            raise

    compiled = {}
    variants = (
        ("2048_branches", 8, 256, False, "branches", 64),
        ("128_branches", 2, 64, False, "branches", 64),
        ("128_branches_fused", 2, 64, True, "branches", 64),
    )
    if compact_pilot:
        variants = (
            ("compact_reference", 2, 64, True, "branches", 64),
            ("compact_rank32", 2, 64, False, "compact", 32),
            ("compact_rank64", 2, 64, False, "compact", 64),
        )
    if spectral_benchmark:
        variants = (
            ("spectral_reference", 2, 64, True, "branches", 64),
            ("spectral_mlp", 2, 64, False, "spectral_mlp", 64),
        )
    if fno_benchmark:
        result["scope"] = "Paired batch4096 throughput only: spectral MLP versus four canonical FNO blocks, width32, all129 Fourier bins, decoder64. Learned grid256 and full1024 G0+G1 baseline. Resident real data, relative-L2/mode-balanced/Tanaka losses and AdamW; excludes Hadamard, loading and compilation. Three warmups and21 synchronized trials. GPU hard cap180s."
        batch = tuple(array[:4096] for array in datasets["train"])
        jax.block_until_ready(batch)
        variants = (
            ("spectral_mlp", 2, 64, False, "spectral_mlp", 64),
            ("canonical_fno", 4, 32, False, "canonical_fno", 64),
        )
        if fno_fold_benchmark:
            result["scope"] = result["scope"].replace(
                "spectral MLP versus four canonical FNO blocks",
                "unfolded versus folded spatial linear branch in four canonical FNO blocks",
            )
            variants = (
                ("canonical_fno", 4, 32, False, "canonical_fno", 64),
                ("canonical_fno_folded", 4, 32, False, "canonical_fno", 64),
            )
        if fno_gemm_benchmark:
            result["scope"] = result["scope"].replace(
                "spectral MLP versus four canonical FNO blocks",
                "folded canonical FNO with complex, packed-real and split-real spectral GEMMs",
            )
            variants = tuple((f"fno_{mode}", 4, 32, False, "canonical_fno", 64)
                             for mode in ("complex", "packed", "split"))
        if fno_transform_benchmark:
            result["scope"] = result["scope"].replace(
                "complex, packed-real and split-real spectral GEMMs",
                "packed-real GEMMs and ortho FFT, backward-normalized FFT or dense DFT",
            )
            variants = tuple((f"fno_{transform}", 4, 32, False, "canonical_fno", 64)
                             for transform in ("fft", "fft_backward", "dft"))
    for name, blocks, latent, fused, correction_kind, rank in variants:
        if batch_sweep and not fused:
            continue
        if fusion_only and name == "2048_branches":
            continue
        if fused and not fused_ok:
            continue
        model = CraigSulemDNO(
            width=640, n_blocks=blocks, latent=latent, mult_hidden=160,
            learned_grid=256, fuse_fft=fused, domain_length=metadata["domain_length"],
            correction_kind=correction_kind, compact_rank=rank, compact_hidden=128,
            fno_fold_spatial=fno_gemm_benchmark or name == "canonical_fno_folded",
            fno_spectral_gemm="packed" if fno_transform_benchmark else name.removeprefix("fno_") if fno_gemm_benchmark else "complex",
            fno_transform=name.removeprefix("fno_") if fno_transform_benchmark else "fft",
            eta_scale=float(scales[0]), xi_scale=float(scales[1]), target_scale=target_scale,
        )
        params = model.init(key, batch[0][:1], batch[1][:1])["params"]
        if fno_gemm_benchmark:
            assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(params))
            assert model.apply({"params": params}, batch[0][:1], batch[1][:1]).dtype == jnp.float32
        # Use the same optimizer schedule and sample sequence for both capacities.
        schedule = optax.join_schedules(
            (optax.linear_schedule(0, 2e-5, 500), optax.constant_schedule(2e-5)), (500,),
        )
        state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adamw(schedule, weight_decay=1e-4))
        if fusion_only or batch_sweep:
            state = jax.device_put(serialization.from_bytes(
                state, (destination / "128_branches.msgpack").read_bytes(),
            ))
            params = state.params

        def objective(parameters: dict, fields: jax.Array, depths: jax.Array,
                      targets: jax.Array, mask: jax.Array, step: jax.Array) -> tuple[jax.Array, jax.Array]:
            prediction = model.apply({"params": parameters}, fields, depths)
            physical_eta = fields[..., 0] * scales[0]
            physical_depth = jnp.exp(jnp.minimum(depths[:, 0], jnp.log(5.0)))
            mode = compute_mode_balanced_loss(
                physical_eta, prediction[..., 0] * target_scale, targets[..., 0] * target_scale,
                physical_depth, k_rfft,
            )
            tangent, _ = compute_translation_tangent_loss(
                physical_eta, prediction[..., 0] * target_scale, targets[..., 0] * target_scale,
                physical_depth, k, sample_mask=mask,
            )
            relative = relative_l2_loss(prediction, targets)
            return relative + 6 * jnp.minimum(step / 500, 1) * mode + 10 * tangent, relative

        def step(current: TrainState, fields: jax.Array, depths: jax.Array,
                 targets: jax.Array, mask: jax.Array) -> tuple[TrainState, jax.Array, jax.Array]:
            (loss, relative), gradients = jax.value_and_grad(objective, has_aux=True)(
                current.params, fields, depths, targets, mask, current.step,
            )
            return current.apply_gradients(grads=gradients), loss, relative

        if (fno_fold_benchmark or fno_gemm_benchmark) and name == variants[0][0]:
            check_params = jax.tree.map(lambda value: value, params)
            decoder = check_params["canonical_fno"]["decoder_out"]["kernel"]
            check_params["canonical_fno"]["decoder_out"]["kernel"] = 0.1 * jax.random.normal(
                jax.random.key(52), decoder.shape, dtype=decoder.dtype,
            )
            checks = []
            configurations = ((False, "complex", "fft"), (True, "complex", "fft"))
            if fno_gemm_benchmark:
                configurations = tuple((True, mode, "fft") for mode in ("complex", "packed", "split"))
            if fno_transform_benchmark:
                configurations = tuple((True, "packed", transform) for transform in ("fft", "fft_backward", "dft"))
            for folded, gemm, transform in configurations:
                check_model = model.clone(fno_fold_spatial=folded, fno_spectral_gemm=gemm, fno_transform=transform)

                def check_loss(parameters: dict, fields: jax.Array) -> tuple[jax.Array, jax.Array]:
                    prediction = check_model.apply({"params": parameters}, fields, batch[1][:4])
                    return jnp.mean((prediction - batch[2][:4])**2), prediction

                checks.append(jax.block_until_ready(jax.jit(jax.value_and_grad(
                    check_loss, argnums=(0, 1), has_aux=True,
                ))(check_params, batch[0][:4])))
            errors = {}
            for configuration, check in zip(configurations[1:], checks[1:], strict=True):
                for label, original, candidate in (
                    ("prediction", checks[0][0][1], check[0][1]),
                    ("parameter_gradient", checks[0][1][0], check[1][0]),
                    ("input_gradient", checks[0][1][1], check[1][1]),
                ):
                    left = np.concatenate([np.asarray(leaf).ravel() for leaf in jax.tree.leaves(original)])
                    right = np.concatenate([np.asarray(leaf).ravel() for leaf in jax.tree.leaves(candidate)])
                    error = float(np.linalg.norm(left - right) / max(np.linalg.norm(left), 1e-12))
                    assert np.isfinite(error) and error < 1e-3, (configuration, label, error)
                    errors[f"{configuration[1]}_{configuration[2]}_{label}"] = error
            result["optimization_relative_errors"] = errors
            print(f"Optimization output/gradient relative errors: {errors}", flush=True)

        if batch_sweep:
            rows = np.random.default_rng(123).permutation(metadata["train_rows"])
            for size in ((4096,) if profile_step else (512, 1024, 2048, 4096, 8192)):
                if perf_counter() >= deadline:
                    break
                batch = tuple(array[rows[:size]] for array in datasets["train"])
                jax.block_until_ready(batch)
                before = perf_counter()
                executable = jax.jit(step).lower(state, *batch).compile()
                compile_seconds = perf_counter() - before
                for _ in range(3):
                    jax.block_until_ready(executable(state, *batch))
                timings = []
                for _ in range(21):
                    tick = perf_counter()
                    output = jax.block_until_ready(executable(state, *batch))
                    timings.append((perf_counter() - tick) * 1000)
                assert np.isfinite(float(output[1]))
                assert all(bool(jnp.isfinite(leaf).all()) for leaf in jax.tree.leaves(output[0]))
                memory = executable.memory_analysis()
                median = statistics.median(timings)
                record = {
                    "batch_size": size, "step_median_ms": median,
                    "samples_per_second": size * 1000 / median,
                    "step_trials_ms": timings, "compile_seconds": compile_seconds,
                    "compiler_buffer_bytes": memory.argument_size_in_bytes + memory.output_size_in_bytes + memory.temp_size_in_bytes - memory.alias_size_in_bytes,
                    "finite_loss_and_updated_state": True,
                }
                result["variants"][str(size)] = record
                print(f"Batch {size}: {median:.3f}ms, {record['samples_per_second']:.0f} samples/s", flush=True)
                if profile_step:
                    import gzip
                    import math
                    from collections import Counter

                    record["matmul_precision"] = jax.config.jax_default_matmul_precision
                    record["compiler_cost_estimates"] = executable.cost_analysis()
                    record["allocator_memory_stats"] = jax.devices()[0].memory_stats()
                    record["nvidia_smi_memory_used_mib"] = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        check=True, capture_output=True, text=True,
                    ).stdout.strip()
                    (destination / "profile_4096_optimized_hlo.txt").write_text(executable.as_text())
                    pending = [jax.make_jaxpr(step)(state, *batch).jaxpr]
                    dots = []
                    while pending:
                        for equation in pending.pop().eqns:
                            if equation.primitive.name == "dot_general":
                                axes = equation.params["dimension_numbers"][0][0]
                                contraction = math.prod(equation.invars[0].aval.shape[axis] for axis in axes)
                                dots.append({
                                    "name": str(equation.source_info.name_stack),
                                    "flops": 2 * math.prod(equation.outvars[0].aval.shape) * contraction,
                                    "lhs": equation.invars[0].aval.shape,
                                    "rhs": equation.invars[1].aval.shape,
                                    "precision": str(equation.params["precision"]),
                                })
                            for value in equation.params.values():
                                if hasattr(value, "jaxpr"):
                                    pending.append(value.jaxpr)
                                elif hasattr(value, "eqns"):
                                    pending.append(value)
                    record["dense_operations"] = dots
                    record["dense_flops"] = sum(dot["flops"] for dot in dots)
                    trace_dir = destination / "profile_4096"
                    with jax.profiler.trace(str(trace_dir)):
                        for _ in range(3):
                            jax.block_until_ready(executable(state, *batch))
                    trace_path = max(trace_dir.rglob("*.trace.json.gz"), key=lambda path: path.stat().st_mtime)
                    with gzip.open(trace_path, "rt") as stream:
                        events = json.load(stream)["traceEvents"]
                    processes = [event for event in events if event.get("name") == "process_name"]
                    record["trace_processes"] = processes
                    gpu_pids = {event["pid"] for event in processes if "GPU" in event.get("args", {}).get("name", "")}
                    durations = Counter()
                    counts = Counter()
                    for event in events:
                        if event.get("pid") in gpu_pids and event.get("ph") == "X":
                            durations[event["name"]] += event.get("dur", 0)
                            counts[event["name"]] += 1
                    record["gpu_trace_kernels"] = [
                        {"name": name, "total_us": duration, "count": counts[name]}
                        for name, duration in durations.most_common()
                    ]
                    print(f"Dense FLOPs: {record['dense_flops']}; traced GPU kernels: {len(durations)}", flush=True)
            result["initial_checkpoint_step"] = int(state.step)
            result["function_wall_seconds"] = perf_counter() - started
            filename = "profile_results_h100.json" if profile_step else "batch_sweep_results_h100.json"
            (destination / filename).write_text(json.dumps(result, indent=2))
            volume.commit()
            return json.dumps(result, indent=2)

        before = perf_counter()
        executable = jax.jit(step).lower(state, *batch).compile()
        evaluate = jax.jit(objective).lower(params, *batch, jnp.asarray(1000, dtype=jnp.int64)).compile()
        memory = executable.memory_analysis()
        for _ in range(3):
            jax.block_until_ready(executable(state, *batch))
        timings = []
        for _ in range(21):
            tick = perf_counter()
            output = jax.block_until_ready(executable(state, *batch))
            timings.append((perf_counter() - tick) * 1000)
        assert np.isfinite(float(output[1]))
        if fno_benchmark:
            assert all(bool(jnp.isfinite(leaf).all()) for leaf in jax.tree.leaves(output[0]))
        record = {
            "blocks": blocks, "latent": latent, "fused": fused,
            "correction_kind": correction_kind, "compact_rank": rank, "compact_hidden": 128,
            "fno_fold_spatial": model.fno_fold_spatial,
            "fno_spectral_gemm": model.fno_spectral_gemm,
            "fno_transform": model.fno_transform,
            "parameter_dtypes": sorted({str(leaf.dtype) for leaf in jax.tree.leaves(params)}),
            "parameters": sum(leaf.size for leaf in jax.tree.leaves(params)),
            "step_median_ms": statistics.median(timings), "step_trials_ms": timings,
            "compile_and_benchmark_seconds": perf_counter() - before,
            "compiler_buffer_bytes": memory.argument_size_in_bytes + memory.output_size_in_bytes + memory.temp_size_in_bytes - memory.alias_size_in_bytes,
        }
        result["variants"][name] = record
        compiled[name] = [state, executable, evaluate]
        print(f"{name}: {record['step_median_ms']:.3f}ms, params={record['parameters']}", flush=True)
        if fno_benchmark:
            record["batch_size"] = 4096
        if fno_gemm_benchmark:
            import gzip
            from collections import Counter

            trace_dir = destination / f"profile_{name}"
            with jax.profiler.trace(str(trace_dir)):
                for _ in range(3):
                    jax.block_until_ready(executable(state, *batch))
            trace_path = max(trace_dir.rglob("*.trace.json.gz"), key=lambda path: path.stat().st_mtime)
            with gzip.open(trace_path, "rt") as stream:
                events = json.load(stream)["traceEvents"]
            gpu_pids = {event["pid"] for event in events if event.get("name") == "process_name"
                        and "GPU" in event.get("args", {}).get("name", "")}
            durations = Counter()
            for event in events:
                if event.get("pid") in gpu_pids and event.get("ph") == "X":
                    durations[event["name"]] += event.get("dur", 0)
            record["gpu_kernel_ms_per_step"] = {
                name: duration / 3000 for name, duration in durations.most_common()
            }
        if compact_pilot or (spectral_benchmark and not fno_benchmark):
            large_batch = tuple(array[:4096] for array in datasets["train"])
            large_step = jax.jit(step).lower(state, *large_batch).compile()
            for _ in range(3):
                jax.block_until_ready(large_step(state, *large_batch))
            timings = []
            for _ in range(21):
                tick = perf_counter()
                output = jax.block_until_ready(large_step(state, *large_batch))
                timings.append((perf_counter() - tick) * 1000)
            assert np.isfinite(float(output[1]))
            memory = large_step.memory_analysis()
            record["batch4096"] = {
                "step_median_ms": statistics.median(timings), "step_trials_ms": timings,
                "compiler_buffer_bytes": memory.argument_size_in_bytes + memory.output_size_in_bytes + memory.temp_size_in_bytes - memory.alias_size_in_bytes,
            }
            print(f"{name}, batch 4096: {statistics.median(timings):.3f}ms", flush=True)

    if spectral_benchmark:
        result["function_wall_seconds"] = perf_counter() - started
        filename = "fno_benchmark_h100.json" if fno_benchmark else "spectral_benchmark_h100.json"
        if fno_fold_benchmark:
            filename = "fno_fold_benchmark_h100.json"
        if fno_gemm_benchmark:
            filename = "fno_gemm_fp32_benchmark_h100.json"
        if fno_transform_benchmark:
            filename = "fno_transform_benchmark_h100.json"
        (destination / filename).write_text(json.dumps(result, indent=2))
        volume.commit()
        return json.dumps(result, indent=2)

    if compact_pilot:
        names = tuple(compiled)
    else:
        selected_small = min(
            (name for name in compiled if name.startswith("128_")),
            key=lambda name: result["variants"][name]["step_median_ms"],
        )
        result["selected_small_implementation"] = selected_small
        names = tuple(compiled) if fusion_only else ("2048_branches", selected_small)
    if fusion_only:
        result["initial_checkpoint_step"] = int(compiled[names[0]][0].step)
        outputs = [jax.block_until_ready(compiled[name][1](compiled[name][0], *batch)) for name in names]
        comparisons = [float(jnp.linalg.norm(left - right) / jnp.maximum(jnp.linalg.norm(left), 1e-20))
                       for left, right in zip(jax.tree.leaves(outputs[0]), jax.tree.leaves(outputs[1]))]
        result["whole_step_max_leaf_relative_error"] = max(comparisons)
        assert max(comparisons) < 2e-3, max(comparisons)
        paired = {name: [] for name in names}
        for trial in range(31):
            for name in names if trial % 2 == 0 else names[::-1]:
                tick = perf_counter()
                jax.block_until_ready(compiled[name][1](compiled[name][0], *batch))
                paired[name].append((perf_counter() - tick) * 1000)
        for name in names:
            result["variants"][name].update(step_median_ms=statistics.median(paired[name]), step_trials_ms=paired[name])
            print(f"Paired {name}: {statistics.median(paired[name]):.3f}ms", flush=True)
    rng = np.random.default_rng(123)
    completed = 0
    orders = []
    for epoch in range(32):
        orders.extend(np.split(rng.permutation(metadata["train_rows"]), metadata["train_rows"] // 512))
    steps_to_run = 32 if fusion_only else 1024
    for iteration in range(steps_to_run + 1):
        if iteration % 128 == 0 or iteration == steps_to_run:
            record = {"step": completed}
            for name in names:
                current, _, evaluate = compiled[name]
                values = []
                for start in range(0, metadata["validation_rows"], 512):
                    validation = tuple(array[start:start + 512] for array in datasets["validation"])
                    values.append(float(evaluate(current.params, *validation, jnp.asarray(1000, dtype=jnp.int64))[1]))
                record[name] = float(np.mean(values))
            result["validation_history"].append(record)
            print(f"Validation relative-L2: {record}", flush=True)
        if iteration == steps_to_run or perf_counter() >= deadline:
            break
        fields = tuple(array[orders[iteration]] for array in datasets["train"])
        for name in names:
            current, executable, _ = compiled[name]
            next_state, loss, _ = jax.block_until_ready(executable(current, *fields))
            if not np.isfinite(float(loss)):
                raise FloatingPointError(f"Nonfinite {name} loss at step {iteration}")
            compiled[name][0] = next_state
        completed += 1

    result["steps_per_model"] = completed
    for name in names:
        current, _, evaluate = compiled[name]
        values = [float(evaluate(current.params, *(array[start:start + 512] for array in datasets["validation"]), jnp.asarray(1000, dtype=jnp.int64))[1])
                  for start in range(0, metadata["validation_rows"], 512)]
        result["variants"][name]["final_validation_relative_l2"] = float(np.mean(values))
        if compact_pilot:
            shifted_values = []
            for start in range(0, metadata["validation_rows"], 512):
                fields, depths, targets, mask = (array[start:start + 512] for array in datasets["validation"])
                shifted_values.append(float(evaluate(
                    current.params, jnp.roll(fields, 128, axis=1), depths,
                    jnp.roll(targets, 128, axis=1), mask, jnp.asarray(1000, dtype=jnp.int64),
                )[1]))
            result["variants"][name]["shifted_validation_relative_l2"] = float(np.mean(shifted_values))
        prefix = "fusion_check_" if fusion_only else ""
        (destination / f"{prefix}{name}.msgpack").write_bytes(serialization.to_bytes(jax.device_get(current)))
    result["function_wall_seconds"] = perf_counter() - started
    result["checkpoint_format"] = "Flax TrainState msgpack; restore with matching model, optimizer schedule and scales in this script. Pilot checkpoints, not production Orbax runs."
    filename = "compact_results_h100.json" if compact_pilot else ("fusion_results_h100.json" if fusion_only else "results.json")
    (destination / filename).write_text(json.dumps(result, indent=2))
    volume.commit()
    return json.dumps(result, indent=2)


@app.function(
    image=image, volumes={"/data": volume}, gpu="H100!:1", cpu=4, memory=16384,
    timeout=600, retries=0, scaledown_window=2,
)
def pilot(run_name: str) -> str:
    return run_pilot(run_name, fusion_only=False)


@app.function(
    image=image, volumes={"/data": volume}, gpu="H100!:1", cpu=4, memory=16384,
    timeout=180, retries=0, scaledown_window=2,
)
def fusion_check(run_name: str) -> str:
    return run_pilot(run_name, fusion_only=True)


@app.function(
    image=image, volumes={"/data": volume}, gpu="H100!:1", cpu=4, memory=16384,
    timeout=180, retries=0, scaledown_window=2,
)
def batch_size_check(run_name: str, profile_step: bool = False) -> str:
    return run_pilot(run_name, fusion_only=False, batch_sweep=True, profile_step=profile_step)


@app.function(
    image=image, volumes={"/data": volume}, gpu="H100!:1", cpu=4, memory=16384,
    timeout=180, retries=0, scaledown_window=2,
)
def compact_check(run_name: str, spectral_benchmark: bool = False, fno_benchmark: bool = False,
                  fno_fold_benchmark: bool = False, fno_gemm_benchmark: bool = False,
                  fno_transform_benchmark: bool = False) -> str:
    return run_pilot(run_name, fusion_only=False, compact_pilot=not spectral_benchmark,
                     spectral_benchmark=spectral_benchmark, fno_benchmark=fno_benchmark,
                     fno_fold_benchmark=fno_fold_benchmark, fno_gemm_benchmark=fno_gemm_benchmark,
                     fno_transform_benchmark=fno_transform_benchmark)


@app.local_entrypoint()
def main(run_name: str = "fewer_branches_20260916", prepared: bool = False,
         fusion_only: bool = False, batch_sweep: bool = False, profile_step: bool = False,
         compact_pilot: bool = False, spectral_benchmark: bool = False,
         fno_benchmark: bool = False, fno_fold_benchmark: bool = False,
         fno_gemm_benchmark: bool = False, fno_transform_benchmark: bool = False) -> None:
    fno_gemm_benchmark = fno_gemm_benchmark or fno_transform_benchmark
    fno_benchmark = fno_benchmark or fno_fold_benchmark or fno_gemm_benchmark
    if not prepared:
        print(prepare.remote(run_name))
    if compact_pilot or spectral_benchmark or fno_benchmark:
        result = compact_check.remote(
            run_name, spectral_benchmark or fno_benchmark, fno_benchmark, fno_fold_benchmark,
            fno_gemm_benchmark, fno_transform_benchmark,
        )
        suffix = "_fno_benchmark_h100" if fno_benchmark else "_spectral_benchmark_h100" if spectral_benchmark else "_compact_h100"
        if fno_fold_benchmark:
            suffix = "_fno_fold_benchmark_h100"
        if fno_gemm_benchmark:
            suffix = "_fno_gemm_fp32_benchmark_h100"
        if fno_transform_benchmark:
            suffix = "_fno_transform_benchmark_h100"
    elif batch_sweep or profile_step:
        result = batch_size_check.remote(run_name, profile_step)
        suffix = "_profile_h100" if profile_step else "_batch_sweep_h100"
    else:
        result = fusion_check.remote(run_name) if fusion_only else pilot.remote(run_name)
        suffix = "_fusion_h100" if fusion_only else ""
    (ROOT / "experiments" / f"{run_name}{suffix}.json").write_text(result)
    print(result)
