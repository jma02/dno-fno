"""Bounded timing of the live EMA trainer on the approved small reference bundle.

Execute its initialization and exact update functions in memory, but not its
epoch/checkpoint loop. No training dataset or trained weights are used.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Check inputs and source extraction on CPU; no updates.")
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "train-jax-10m/1d_dno_fno_jax_ema_grad.py"
    sys.path.insert(0, str(source.parent))
    sys.path.insert(0, str(root))
    from solver.gen_data.pipeline.types import PhysicalFamilyId

    fields: dict[str, list[np.ndarray]] = {name: [] for name in ("eta", "xi", "gxi", "depth", "family")}
    reference = root / "outputs/c27_tanaka_hard128_full_equal_local_20260918"
    for name in ("stokes", "tanaka", "benjamin_feir", "jonswap_tma"):
        with np.load(next(reference.glob(f"eval_best_current_test_stratified_n32*/{name}_trajs.npz"))) as saved:
            frames = np.linspace(0, len(saved["times"]) - 1, 64, dtype=int)
            for field in ("eta", "xi", "gxi"):
                fields[field].append(saved[f"truth_{field}"][frames, :4].reshape(256, 1024).astype(np.float32))
            fields["depth"].append(np.tile(saved["depths"][:4], 64).astype(np.float32))
            fields["family"].append(np.full(256, PhysicalFamilyId[name.upper()].value, np.int32))
    fixture = {name: np.stack(values, axis=1).reshape((-1, 1024) if name in ("eta", "xi", "gxi") else (-1,))
               for name, values in fields.items()}
    fixture["xi"] -= fixture["xi"].mean(axis=-1, keepdims=True)
    assert all(np.isfinite(value).all() for value in fixture.values())
    assert np.all(fixture["eta"] + fixture["depth"][:, None] > 0)
    fixture.update(x=np.arange(1024) * (2 * np.pi / 1024), domain_length=np.asarray(2 * np.pi))
    stats = json.loads((root / "outputs/c27_w320_b4_h80_tanaka_hard128_20260924/best_val_ckpt/metadata.json").read_text())["stats"]

    tree = ast.parse(source.read_text())
    entry = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    stop = next(i for i, node in enumerate(entry.body) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "eval_steps" for target in node.targets))
    entry.body = entry.body[:stop + 1] + ast.parse("return locals()").body
    replacements = {
        "dataset": "dataset = fixture",
        "family_ids": "family_ids = fixture['family']",
        "stats": "stats = fixture_stats",
    }
    for i, node in enumerate(entry.body):
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in replacements:
                entry.body[i] = ast.parse(replacements[target.id]).body[0]
            elif isinstance(target, ast.Tuple) and ast.unparse(target) == "(train_indices, val_indices, _)":
                entry.body[i:i + 1] = ast.parse(
                    "train_indices = np.arange(11796720, dtype=np.int32)\n"
                    "val_indices = np.arange(1474440, dtype=np.int32)"
                ).body
    code = compile(ast.fix_missing_locations(tree), str(source), "exec")
    if options.check:
        print("PASS: B1024 reference fixture, four interleaved families, positive depth; trainer prefix compiles")
        return

    results: dict[str, Any] = {
        "device": str(jax.devices()[0]), "device_kind": jax.devices()[0].device_kind,
        "fixture": "64 reference frames x 4 cases x 4 families; timing only, discarded updates",
        "train_updates_per_epoch": 11521, "validation_batches_per_epoch": 1440,
        "phases": {}, "compile_s": {},
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    recipe = (
        "--model cs_dno --norm scale --dataset . --width 320 --n_blocks 8 --latent 4 --cs_mult_hidden 80 "
        "--cs_n_polys 3 --batch_size 1024 --epochs 120 --stop_after_epoch 120 --seed 0 --lr 2e-5 "
        "--lr_warmup_steps 500 --gradient_ema_decay .9 --cs_baseline_true_depth --data_normalization target "
        "--mode_balanced_weight 6 --mode_balanced_warmup_steps 500 --translation_tangent_weight 10 "
        "--translation_tangent_scope tanaka --hadamard_weight .01 --hadamard_warmup_steps 500 "
        "--hadamard_interval 16 --val_hadamard_interval 16 --hadamard_microbatch 8 --metric_interval 16 "
        "--rollout_training_weight 1 --rollout_training_start_epoch 60 --rollout_training_ramp_epochs 5 "
        "--rollout_training_interval 16 --rollout_training_microbatch 8"
    ).split()
    for horizon in (10, 50):
        sys.argv = [str(source), *recipe, "--rollout_training_steps", str(horizon),
                    "--rollout_training_weight", "2" if horizon == 10 else "1",
                    "--output_root", str(options.output.parent), "--run_name", f"setup_{horizon}"]
        namespace = {"__file__": str(source), "__name__": "ema_benchmark", "fixture": fixture, "fixture_stats": stats}
        exec(code, namespace)
        context = namespace["main"]()
        assert context["train_steps_per_epoch"] == 11521
        batch = tuple(jax.device_put(value, context["data_sharding"]) for value in (
            fixture["eta"], fixture["xi"], fixture["gxi"],
            np.log(fixture["depth"])[:, None], fixture["family"],
        ))
        rng = jax.device_put(jax.random.PRNGKey(1), context["replicated"])
        initial = context["training_state"].replace(step=jnp.int32(0))
        started = perf_counter()
        update = context["train_steps"][namespace["P"]("batch")].lower(initial, rng, *batch).compile()
        results["compile_s"][f"train_{horizon}"] = perf_counter() - started
        for phase, first_step in (("early", 32 * 11521), (f"rollout_{horizon}", 64 * 11521)):
            if horizon == 50 and phase == "early":
                continue
            first_step -= first_step % 16
            state = jax.tree.map(
                lambda value: jnp.full_like(value, first_step) if np.issubdtype(value.dtype, np.integer) else value,
                initial,
            )
            state, loss, metrics = update(state, rng, *batch)
            for _ in range(15):
                state, loss, metrics = update(state, rng, *batch)
            jax.block_until_ready((state, loss, metrics))
            repeats = []
            for _ in range(3):
                started = perf_counter()
                pending = []
                for i in range(128):
                    state, loss, metrics = update(state, rng, *batch)
                    pending.append((loss, metrics))
                    if (i + 1) % 16 == 0:
                        measured = jax.device_get(pending)
                        assert all(np.isfinite(value).all() for value in jax.tree.leaves(measured))
                        scheduled = measured[0][1]
                        assert scheduled["hadamard_active"] == 1
                        assert scheduled["rollout_active"] == (phase != "early")
                        if phase != "early":
                            assert scheduled["rollout_valid_fraction"] == 1
                        pending.clear()
                jax.block_until_ready(state)
                repeats.append((perf_counter() - started) / 128)
            assert all(np.isfinite(value).all() for value in jax.device_get(jax.tree.leaves(state)))
            results["phases"][phase] = {"seconds_per_update": repeats, "median_s": float(np.median(repeats))}
            print(phase, results["phases"][phase], flush=True)
            options.output.write_text(json.dumps(results, indent=2))
        if horizon == 10:
            for hadamard in (False, True):
                started = perf_counter()
                evaluate = context["eval_steps"][(namespace["P"]("batch"), hadamard)].lower(
                    state.params, rng, jnp.int32(0), *batch,
                ).compile()
                results["compile_s"][f"validation_hd{int(hadamard)}"] = perf_counter() - started
                values = evaluate(state.params, rng, jnp.int32(0), *batch)
                jax.block_until_ready(values)
                repeats = []
                for _ in range(3):
                    started = perf_counter()
                    for i in range(16):
                        values = evaluate(state.params, rng, jnp.int32(i), *batch)
                    jax.block_until_ready(values)
                    repeats.append((perf_counter() - started) / 16)
                assert all(np.isfinite(value).all() for value in jax.device_get(values))
                results["phases"][f"validation_hd{int(hadamard)}"] = {"seconds_per_batch": repeats, "median_s": float(np.median(repeats))}
            options.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
