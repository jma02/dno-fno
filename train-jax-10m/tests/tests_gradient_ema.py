"""Check gradient EMA semantics, disabled behavior, and optimizer-state resume."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training import train_state

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint_util import training_counter_values  # noqa: E402

build_optimizer = importlib.import_module("1d_dno_fno_jax").build_optimizer


def test_gradient_ema_and_resume() -> None:
    for decay in (0.0, 0.9):
        schedule = optax.constant_schedule(1e-4)
        optimizer = build_optimizer(schedule, 1e-4, decay)
        adamw = optax.adamw(schedule, weight_decay=1e-4)
        params = {"w": jnp.array([1.0, -0.5], dtype=jnp.float32)}
        state = train_state.TrainState.create(apply_fn=lambda: None, params=params, tx=optimizer)
        state = state.replace(step=jnp.int32(0))
        reference_state = adamw.init(params)
        reference_params = params
        average = np.zeros(2, dtype=np.float64)
        resumed = None
        for step, values in enumerate(([2., -1.], [-3., 4.], [0.5, -2.], [1., 3.]), 1):
            gradient = {"w": jnp.array(values, dtype=jnp.float32)}
            average = decay * average + (1 - decay) * np.asarray(values)
            smoothed = {"w": jnp.asarray(average / (1 - decay**step), dtype=jnp.float32)}
            updates, reference_state = adamw.update(smoothed, reference_state, reference_params)
            reference_params = optax.apply_updates(reference_params, updates)
            state = state.apply_gradients(grads=gradient)
            np.testing.assert_allclose(state.params["w"], reference_params["w"], rtol=1e-7)
            actual_adam = state.opt_state[1][0] if decay else state.opt_state[0]
            np.testing.assert_allclose(actual_adam.mu["w"], reference_state[0].mu["w"], rtol=2e-6)
            np.testing.assert_allclose(actual_adam.nu["w"], reference_state[0].nu["w"], rtol=2e-6)
            assert training_counter_values(state, context="EMA test") == (step, step)
            if resumed is not None:
                resumed = resumed.apply_gradients(grads=gradient)
                for actual, expected in zip(jax.tree.leaves(resumed), jax.tree.leaves(state)):
                    np.testing.assert_array_equal(actual, expected)
            if step == 2:
                resumed = serialization.from_bytes(state, serialization.to_bytes(state))
        if decay:
            broken = state.replace(opt_state=(state.opt_state[0]._replace(count=jnp.int32(0)), state.opt_state[1]))
            try:
                training_counter_values(broken, context="broken EMA")
            except RuntimeError as error:
                assert "gradient EMA count" in str(error)
            else:
                raise AssertionError("Mismatched EMA count was accepted")


if __name__ == "__main__":
    test_gradient_ema_and_resume()
    print("PASS: EMA before AdamW, plain AdamW unchanged, resume and counter checks")
