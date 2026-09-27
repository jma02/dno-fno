"""Local full-size attention/gradient-EMA smoke test on synthetic DNO data."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = list(map(str, (ROOT, ROOT / "train-jax-10m", ROOT / "models/dno-net", ROOT / "models/fno-jax")))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from flax import serialization  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402

from checkpoint_util import training_counter_values  # noqa: E402
from dno_net_v2 import CraigSulemDNO  # noqa: E402
from hadamard_shape_regularizer import compute_hadamard_loss  # noqa: E402
from losses import relative_l2_loss  # noqa: E402
from mode_balanced_regularizer import compute_mode_balanced_loss  # noqa: E402
from solver.solvers.dno_series_jax import build_grid, dno_series_eval  # noqa: E402
from translation_tangent_regularizer import compute_translation_tangent_loss  # noqa: E402


def main() -> None:
    started = perf_counter()
    build_optimizer = importlib.import_module("1d_dno_fno_jax").build_optimizer
    x, k = build_grid(1024, 2 * jnp.pi)
    rng = np.random.default_rng(24)
    modes = np.arange(1, 17)
    basis = np.exp(1j * np.asarray(x)[:, None] * modes)
    coefficients = rng.normal(size=(2, 64, 16)) + 1j * rng.normal(size=(2, 64, 16))
    eta, xi = np.real(np.einsum("cbm,nm->cbn", coefficients / modes**2, basis))
    eta = jnp.asarray(.03 * eta)
    xi = jnp.asarray(.05 * xi)
    depth = jnp.asarray(rng.uniform(.6, 1.4, (64, 1)))
    # Smooth small-amplitude waves: order four supplies a nonzero correction target.
    targets = jax.jit(lambda surface, potential: dno_series_eval(
        surface, potential, k, depth, order=4,
    ))(eta, xi)[..., None].astype(jnp.float32)
    fields = jnp.stack((eta, xi), -1).astype(jnp.float32)
    log_depth = jnp.log(depth).astype(jnp.float32)
    model = CraigSulemDNO(width=64, latent=32, correction_kind="spatial_spectral_attention")
    params = model.init(jax.random.key(0), fields[:1], log_depth[:1])["params"]
    state = TrainState.create(
        apply_fn=model.apply, params=params,
        tx=build_optimizer(optax.constant_schedule(1e-4), 1e-4, .9),
    )
    print(f"Device={jax.devices()}, parameters={sum(p.size for p in jax.tree.leaves(params))}, batch=64, grid=1024", flush=True)

    def objective(parameters: dict[str, Any], regularized: bool) -> tuple[jax.Array, jax.Array]:
        prediction = model.apply({"params": parameters}, fields, log_depth)
        relative = relative_l2_loss(prediction, targets)
        mode = compute_mode_balanced_loss(fields[..., 0], prediction[..., 0], targets[..., 0], depth[:, 0], k[:513])
        # Synthetic mask exercises the tangent-loss code, not a Tanaka-family claim.
        tangent, _ = compute_translation_tangent_loss(
            fields[..., 0], prediction[..., 0], targets[..., 0], depth[:, 0], k,
            sample_mask=jnp.ones(64, dtype=jnp.bool_),
        )
        total = relative + 6 * mode + 10 * tangent
        if regularized:
            hadamard = compute_hadamard_loss(
                rng=jax.random.key(90), apply_fn=model.apply, model_params=parameters,
                eta_phys=fields[:8, :, 0], xi_phys=fields[:8, :, 1],
                batch_depth_local=log_depth[:8].astype(jnp.float64),
                norm_inputs_fn=lambda surface, potential: jnp.stack((surface, potential), -1),
                denorm_targets_fn=lambda output: output,
                k=k, dtype=jnp.float64,
            )
            total += .01 * hadamard
        return total, relative

    def step(current: TrainState, regularized: bool = False) -> tuple[TrainState, jax.Array, jax.Array, jax.Array]:
        (loss, relative), gradients = jax.value_and_grad(objective, has_aux=True)(current.params, regularized)
        finite = jnp.all(jnp.stack([jnp.isfinite(leaf).all() for leaf in jax.tree.leaves(gradients)]))
        return current.apply_gradients(grads=gradients), loss, relative, finite

    ordinary_step = jax.jit(step, static_argnames=("regularized",))
    history = []
    for index in range(64):
        tick = perf_counter()
        state, loss, relative, finite = jax.block_until_ready(ordinary_step(state))
        elapsed = perf_counter() - tick
        assert bool(finite) and np.isfinite(float(loss)), "Nonfinite ordinary loss/gradient"
        assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(state))
        history.append({"step": index + 1, "loss": float(loss), "relative_l2_before_update": float(relative), "seconds": elapsed})
        if index < 8 or (index + 1) % 8 == 0:
            print(json.dumps(history[-1]), flush=True)
    # Exercise the full float64 Hadamard path at a nonzero weight and trained head.
    print("Compiling full Hadamard update (microbatch8)...", flush=True)
    tick = perf_counter()
    state, loss, relative, finite = jax.block_until_ready(ordinary_step(state, regularized=True))
    hadamard_seconds = perf_counter() - tick
    assert bool(finite) and np.isfinite(float(loss)), "Nonfinite Hadamard loss/gradient"
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(state))
    assert training_counter_values(state, context="local attention smoke") == (65, 65)
    restored = jax.device_put(serialization.from_bytes(state, serialization.to_bytes(state)))
    assert training_counter_values(restored, context="local attention restore") == (65, 65)
    for actual, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(state), strict=True):
        np.testing.assert_array_equal(actual, expected)
    final_relative = float(jax.jit(lambda p: relative_l2_loss(model.apply({"params": p}, fields, log_depth), targets))(state.params))
    result = {
        "scope": "CPU correctness smoke only; repeated synthetic batch, not production validation or GPU timing. Full physics weights active immediately to exercise gradients.",
        "device": str(jax.devices()), "jax": jax.__version__, "batch_size": 64, "grid_size": 1024,
        "parameters": sum(p.size for p in jax.tree.leaves(params)),
        "gradient_ema_decay": .9, "lr": 1e-4, "history": history,
        "ordinary_median_seconds": statistics.median(item["seconds"] for item in history[1:]),
        "hadamard_compile_and_update_seconds": hadamard_seconds,
        "hadamard_total_loss": float(loss), "final_synthetic_relative_l2": final_relative,
        "steps_and_adam_counter": [65, 65], "ema_counter": int(state.opt_state[0].count),
        "all_gradients_states_finite": True, "serialization_roundtrip_exact": True,
        "total_seconds": perf_counter() - started,
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
