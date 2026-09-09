"""CPU regression for finite gradients at an exact supervised fit."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models" / "fno-jax"))

from losses import relative_l2_loss  # noqa: E402

jax.config.update("jax_enable_x64", True)


def test_exact_fit_and_mixed_batches_have_finite_gradients() -> None:
    differentiate = jax.value_and_grad(relative_l2_loss, argnums=(0, 1))
    for dtype in (jnp.float32, jnp.float64):
        target = jnp.arange(48, dtype=dtype).reshape(3, 16, 1).at[2].set(0)
        for evaluate in (differentiate, jax.jit(differentiate)):
            loss, gradients = evaluate(target, target)
            assert loss == 0.0
            assert loss.dtype == jnp.float64
            for gradient in gradients:
                np.testing.assert_array_equal(gradient, jnp.zeros_like(target))

            prediction = target.at[1].multiply(1.001)
            loss, gradients = evaluate(prediction, target)
            assert loss > 0.0
            for gradient in gradients:
                assert bool(jnp.all(jnp.isfinite(gradient)))
                np.testing.assert_array_equal(gradient[jnp.array([0, 2])], 0.0)
                assert bool(jnp.any(gradient[1] != 0.0))

            assert jnp.isnan(evaluate(target.at[0, 0, 0].set(jnp.nan), target)[0])
            assert not jnp.isfinite(
                evaluate(target.at[0, 0, 0].set(jnp.inf), target)[0]
            )


if __name__ == "__main__":
    test_exact_fit_and_mixed_batches_have_finite_gradients()
    print("[PASS] exact-fit and mixed-batch supervised loss gradients")
