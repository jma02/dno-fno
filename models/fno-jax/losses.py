from __future__ import annotations

import jax
import jax.numpy as jnp


def relative_l2_loss(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
) -> jnp.ndarray:
    """Relative spectral L2 used by the locked C27 training objective."""
    nx = prediction.shape[1]
    n_freq = nx // 2 + 1
    prediction = prediction.reshape((prediction.shape[0], nx, -1))
    target = target.reshape((target.shape[0], nx, -1))

    pred_fft = jnp.fft.rfft(prediction, axis=1)
    tgt_fft = jnp.fft.rfft(target, axis=1)

    # With x64 enabled, these weights promote float32 FFT results to complex128
    # before the norm.
    weight = jnp.ones((1, n_freq, 1))

    diff_w = (pred_fft - tgt_fft) * weight
    tgt_w = tgt_fft * weight
    squared_error = jnp.sum(jnp.abs(diff_w) ** 2, axis=(1, 2))
    zero_error = squared_error == 0.0
    # Mask before sqrt as well, so exact fits have a finite zero gradient.
    diff_norm = jnp.where(
        zero_error, 0.0, jnp.sqrt(jnp.where(zero_error, 1.0, squared_error))
    )
    tgt_norm = jnp.sqrt(jnp.clip(jnp.sum(jnp.abs(tgt_w) ** 2, axis=(1, 2)), 1e-12))
    return jnp.mean(diff_norm / tgt_norm)


def count_params(params: object) -> int:
    return sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))
