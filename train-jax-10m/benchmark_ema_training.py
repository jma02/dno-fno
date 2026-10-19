"""Bounded exact-update timing of native versus CPU-prefetched CS decoder terms.

Use one GPU with JAX_PLATFORMS=cuda,cpu. The balanced TRAIN fixture, fresh
parameters and discarded updates are for throughput only, not model selection.
Compilation and warmup are separate from the one-pass end-to-end timings.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import jax
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Timing JSON; no weights are saved.")
    parser.add_argument("--dataset", type=Path, default=root / "outputs/paper_dataset_full_equal_tanaka_hard128_20260918/arrays")
    parser.add_argument("--updates", type=int, choices=(64, 128), default=64)
    parser.add_argument("--check", action="store_true", help="Validate TRAIN inputs and source extraction on CPU; no updates.")
    options = parser.parse_args()
    if not options.check and options.output is None:
        parser.error("--output is required for timing")
    if options.check:
        jax.config.update("jax_platforms", "cpu")
    source = root / "train-jax-10m/1d_dno_fno_jax_ema_grad.py"
    sys.path.insert(0, str(source.parent))
    sys.path.insert(0, str(root))
    from solver.gen_data.pipeline.types import DatasetSplit, PhysicalFamilyId
    from util import load_dataset_arrays

    dataset = load_dataset_arrays(options.dataset)
    family_ids = np.load(options.dataset / "family_id.npy", mmap_mode="r")
    training_rows = dataset["dataset_split"] == DatasetSplit.TRAIN.value
    batch_size = 16
    full_train_updates = (int(training_rows.sum()) + batch_size - 1) // batch_size
    rng = np.random.default_rng(0)
    rows = np.stack([
        rng.choice(np.flatnonzero(training_rows & (family_ids == family)),
                   options.updates * batch_size // len(PhysicalFamilyId), replace=False)
        for family in PhysicalFamilyId
    ], axis=1).reshape(-1)
    fixture = {name: np.asarray(dataset[name][rows]) for name in ("eta", "xi", "gxi", "depth")}
    fixture.update(x=dataset["x"], domain_length=dataset["domain_length"], family=np.asarray(family_ids[rows]))
    assert all(np.isfinite(value).all() for value in fixture.values())
    assert np.all(fixture["eta"] + fixture["depth"][:, None] > 0)
    assert np.all(training_rows[rows])
    assert np.all(fixture["family"].reshape(-1, len(PhysicalFamilyId)) == list(PhysicalFamilyId))
    fixture_indices = np.arange(rows.size)
    stats = json.loads((options.dataset / "stats.json").read_text())
    scale_source = root / "outputs/c27_phase1_lr8_momentum_half_ema995_grad05_adam05_b2099_batch16_hd04_band128_nested1_gamma32_full_20261012/config.json"
    feature_scales = json.loads(scale_source.read_text())["cs_surface_feature_scales"]
    assert len(feature_scales) == 7 and np.all(np.asarray(feature_scales) > 0)

    tree = ast.parse(source.read_text())
    entry = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    stop = next(i for i, node in enumerate(entry.body) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "eval_steps" for target in node.targets))
    cpu_start = next(i for i, node in enumerate(entry.body) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "cpu_terms" for target in node.targets))
    cpu_stop = next(i for i, node in enumerate(entry.body)
                    if isinstance(node, ast.FunctionDef) and node.name == "prefetch_batches")
    # Keep the actual initialization, full losses, update, and background producer;
    # omit checkpoint restore, validation/epoch loops and all checkpoint writes.
    entry.body = entry.body[:stop + 1] + entry.body[cpu_start:cpu_stop + 1] + ast.parse("return locals()").body
    replacements = {
        "dataset": "dataset = fixture",
        "family_ids": "family_ids = fixture['family']",
        "stats": "stats = fixture_stats",
        "train_steps_per_epoch": "train_steps_per_epoch = full_train_updates",
    }
    for i, node in enumerate(entry.body):
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in replacements:
                entry.body[i] = ast.parse(replacements[target.id]).body[0]
            elif isinstance(target, ast.Tuple) and ast.unparse(target) == "(train_indices, val_indices, _)":
                entry.body[i] = ast.parse("train_indices = val_indices = fixture_indices").body[0]
        elif isinstance(node, ast.If) and ast.unparse(node.test) == "args.cs_normalize_surface_features":
            entry.body[i] = ast.parse("args.cs_surface_feature_scales = fixture_feature_scales").body[0]
    code = compile(ast.fix_missing_locations(tree), str(source), "exec")
    recipe = (
        "--model cs_dno --cs_terms_residual --norm scale --width 320 --n_blocks 8 --latent 4 "
        "--cs_mult_hidden 80 --cs_n_polys 3 --cs_normalize_surface_features --cs_surface_context_width 8 "
        "--cs_baseline_order 3 --cs_baseline_true_depth --batch_size 16 --epochs 1 --schedule_epochs 1 --seed 0 --lr 8e-5 "
        "--lr_warmup_steps 500 --gradient_ema_decay .5 --adam_b1 .5 --adam_b2 .99 "
        "--weight_ema_decay .995 --weight_ema_start_fraction .8 --data_normalization target --data_k_max 128 "
        "--mode_balanced_weight 6 --mode_balanced_warmup_steps 500 --translation_tangent_weight 10 "
        "--translation_tangent_scope tanaka --phase_mechanism all_momentum --phase_mechanism_weight .002574349830754761 "
        "--hadamard_weight .04 --hadamard_warmup_steps 500 --hadamard_interval 16 --val_hadamard_interval 16 "
        "--hadamard_microbatch 8 --metric_interval 16 --rollout_training_weight 1 --rollout_training_loss full "
        "--rollout_training_start_epoch 1 --rollout_training_ramp_epochs 1 --rollout_training_interval 16 "
        "--rollout_training_microbatch 8 --rollout_training_steps 10"
    ).split()
    if options.check:
        assert all(device.platform == "cpu" for device in jax.devices())
        print(f"PASS: {rows.size} balanced TRAIN rows, B16, saved TRAIN-fitted RMS, full-update and CPU-producer extraction; no updates")
        return
    if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
        parser.error("Expose exactly one GPU and use JAX_PLATFORMS=cuda,cpu")
    jax.devices("cpu")  # Fail before GPU initialization work if the CPU backend is disabled.
    phase_step = int(np.ceil(0.9 * full_train_updates / 16)) * 16
    results: dict[str, Any] = {
        "device": jax.devices()[0].device_kind, "dataset": str(options.dataset),
        "fixture_train_rows": int(rows.size), "fixture_seed": 0, "batch_size": batch_size,
        "feature_scales_source": str(scale_source), "feature_scales": feature_scales,
        "train_updates_per_epoch": full_train_updates, "schedule_start_step": phase_step,
        "measured_updates": options.updates, "warmup_updates": 16,
        "scheduled_hd_and_rollout_updates": options.updates // 16,
        "ordinary_updates": options.updates - options.updates // 16,
        "initialization": "Fresh seed-0 parameters and zero optimizer buffers; all counters positioned at 90% of epoch1 for timing; warmup discarded",
        "recipe": recipe, "arms": {},
    }
    initial_values = None
    with TemporaryDirectory(prefix="dno-cpu-prefetch-bench-") as setup:
        for arm in ("native", "cpu_prefetch"):
            sys.argv = [str(source), *recipe, "--dataset", str(options.dataset),
                        "--output_root", setup, "--run_name", arm,
                        *(["--cs_terms_cpu_prefetch"] if arm == "cpu_prefetch" else [])]
            namespace = {
                "__file__": str(source), "__name__": "ema_benchmark",
                "fixture": fixture, "fixture_stats": stats, "fixture_indices": fixture_indices,
                "fixture_feature_scales": feature_scales, "full_train_updates": full_train_updates,
            }
            exec(code, namespace)
            context = namespace["main"]()
            initial = context["training_state"]
            current_values = jax.device_get(jax.tree.leaves(initial))
            if initial_values is None:
                initial_values = current_values
            else:
                assert all(np.array_equal(a, b) for a, b in zip(initial_values, current_values, strict=True))
            step = jax.device_put(np.int32(phase_step), context["replicated"])
            gradient_ema, adam_chain = initial.opt_state
            initial = initial.replace(
                step=step,
                opt_state=(gradient_ema._replace(count=step), tuple(
                    transform._replace(count=step) if "count" in transform._fields else transform
                    for transform in adam_chain
                )),
            )
            rng_key = jax.device_put(jax.random.PRNGKey(1), context["replicated"])
            started = perf_counter()
            # Exhaust the producer before compiling; no queued batches are reused.
            compile_batch, = list(context["prefetch_batches"](fixture_indices[:batch_size], None))
            jax.block_until_ready(compile_batch)
            preparation_s = perf_counter() - started
            started = perf_counter()
            update = context["train_steps"][namespace["P"]("batch")].lower(initial, rng_key, *compile_batch).compile()
            compile_s = perf_counter() - started
            warmup = (initial, None, None)
            started = perf_counter()
            for batch in context["prefetch_batches"](fixture_indices[:16 * batch_size], None):
                warmup = update(warmup[0], rng_key, *batch)
            jax.block_until_ready(warmup)
            warmup_s = perf_counter() - started

            # One measured pass. Includes CPU terms, gathering/transfers, bounded
            # queue startup/drain, updates and the trainer's 16-update host reads.
            state, pending, measured = initial, [], []
            started = perf_counter()
            for i, batch in enumerate(context["prefetch_batches"](fixture_indices, None), 1):
                state, loss, metrics = update(state, rng_key, *batch)
                pending.append((loss, metrics))
                if i % 16 == 0:
                    measured.extend(jax.device_get(pending))
                    pending.clear()
            jax.block_until_ready(state)
            elapsed = perf_counter() - started
            assert len(measured) == options.updates and not pending
            assert all(np.isfinite(value).all() for value in jax.tree.leaves(measured))
            assert all(np.isfinite(value).all() for value in jax.device_get(jax.tree.leaves(state)))
            for i, (_, metric) in enumerate(measured):
                assert metric["hadamard_active"] == metric["rollout_active"] == (i % 16 == 0)
            results["arms"][arm] = {
                "first_batch_preparation_s": preparation_s, "update_compile_s": compile_s,
                "warmup_s": warmup_s, "elapsed_s": elapsed, "seconds_per_update": elapsed / options.updates,
                "rollout_valid_fraction_min": min(float(metric["rollout_valid_fraction"]) for _, metric in measured[::16]),
            }
            print(arm, results["arms"][arm], flush=True)
            options.output.parent.mkdir(parents=True, exist_ok=True)
            options.output.write_text(json.dumps(results, indent=2))
    results["cpu_prefetch_speedup"] = results["arms"]["native"]["elapsed_s"] / results["arms"]["cpu_prefetch"]["elapsed_s"]
    assert results["arms"]["native"]["rollout_valid_fraction_min"] == results["arms"]["cpu_prefetch"]["rollout_valid_fraction_min"]
    options.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
