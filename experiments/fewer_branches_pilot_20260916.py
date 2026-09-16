"""Bounded real-data pilot: 2048 versus 128 branches, with optional fused FFTs."""

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


def run_pilot(run_name: str, *, fusion_only: bool, batch_sweep: bool = False) -> str:
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
    deadline = started + (140 if fusion_only or batch_sweep else 540)
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
    print(result["gpu"], flush=True)
    fused_ok = False
    try:
        result["fused_kernel_relative_errors"] = check_fused_roundtrip(interpret=False)
        fused_ok = True
        print(f"Fused kernel correctness: {result['fused_kernel_relative_errors']}", flush=True)
    except Exception as error:
        result["fusion_error"] = repr(error)
        print(f"Fusion unavailable; retaining cuFFT for training: {error}", flush=True)
        if fusion_only or batch_sweep:
            raise

    compiled = {}
    for name, blocks, latent, fused in (
        ("2048_branches", 8, 256, False), ("128_branches", 2, 64, False),
        ("128_branches_fused", 2, 64, True),
    ):
        if batch_sweep and not fused:
            continue
        if fusion_only and name == "2048_branches":
            continue
        if fused and not fused_ok:
            continue
        model = CraigSulemDNO(
            width=640, n_blocks=blocks, latent=latent, mult_hidden=160,
            learned_grid=256, fuse_fft=fused, domain_length=metadata["domain_length"],
            eta_scale=float(scales[0]), xi_scale=float(scales[1]), target_scale=target_scale,
        )
        params = model.init(key, batch[0][:1], batch[1][:1])["params"]
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

        if batch_sweep:
            rows = np.random.default_rng(123).permutation(metadata["train_rows"])
            for size in (512, 1024, 2048, 4096, 8192):
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
            result["initial_checkpoint_step"] = int(state.step)
            result["function_wall_seconds"] = perf_counter() - started
            (destination / "batch_sweep_results_h100.json").write_text(json.dumps(result, indent=2))
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
        record = {
            "blocks": blocks, "latent": latent, "fused": fused,
            "parameters": sum(leaf.size for leaf in jax.tree.leaves(params)),
            "step_median_ms": statistics.median(timings), "step_trials_ms": timings,
            "compile_and_benchmark_seconds": perf_counter() - before,
            "compiler_buffer_bytes": memory.argument_size_in_bytes + memory.output_size_in_bytes + memory.temp_size_in_bytes - memory.alias_size_in_bytes,
        }
        result["variants"][name] = record
        compiled[name] = [state, executable, evaluate]
        print(f"{name}: {record['step_median_ms']:.3f}ms, params={record['parameters']}", flush=True)

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
        prefix = "fusion_check_" if fusion_only else ""
        (destination / f"{prefix}{name}.msgpack").write_bytes(serialization.to_bytes(jax.device_get(current)))
    result["function_wall_seconds"] = perf_counter() - started
    result["checkpoint_format"] = "Flax TrainState msgpack; restore with matching model, optimizer schedule and scales in this script. Pilot checkpoints, not production Orbax runs."
    (destination / ("fusion_results_h100.json" if fusion_only else "results.json")).write_text(json.dumps(result, indent=2))
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
def batch_size_check(run_name: str) -> str:
    return run_pilot(run_name, fusion_only=False, batch_sweep=True)


@app.local_entrypoint()
def main(run_name: str = "fewer_branches_20260916", prepared: bool = False,
         fusion_only: bool = False, batch_sweep: bool = False) -> None:
    if not prepared:
        print(prepare.remote(run_name))
    if batch_sweep:
        result = batch_size_check.remote(run_name)
        suffix = "_batch_sweep_h100"
    else:
        result = fusion_check.remote(run_name) if fusion_only else pilot.remote(run_name)
        suffix = "_fusion_h100" if fusion_only else ""
    (ROOT / "experiments" / f"{run_name}{suffix}.json").write_text(result)
    print(result)
