from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def mse_loss(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((prediction - target) ** 2)


def relative_l2_loss(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    spatial_axes = tuple(range(1, prediction.ndim))
    diff_norm = jnp.sqrt(jnp.sum((prediction - target) ** 2, axis=spatial_axes))
    tgt_norm = jnp.sqrt(jnp.clip(jnp.sum(target ** 2, axis=spatial_axes), 1e-12))
    return jnp.mean(diff_norm / tgt_norm)


def _spectral_relative_l2(
    prediction: jnp.ndarray, target: jnp.ndarray, weight: jnp.ndarray,
) -> jnp.ndarray:
    diff = (prediction - target) * weight
    base = target * weight
    diff_norm = jnp.sqrt(jnp.sum(jnp.abs(diff) ** 2, axis=(1, 2)))
    base_norm = jnp.sqrt(jnp.clip(jnp.sum(jnp.abs(base) ** 2, axis=(1, 2)), 1e-12))
    return jnp.mean(diff_norm / base_norm)


def sobolev_loss(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    k: int = 1,
) -> jnp.ndarray:
    nx = prediction.shape[1]
    n_freq = nx // 2 + 1
    prediction = prediction.reshape((prediction.shape[0], nx, -1))
    target = target.reshape((target.shape[0], nx, -1))

    pred_fft = jnp.fft.rfft(prediction, axis=1)
    tgt_fft = jnp.fft.rfft(target, axis=1)

    wave = jnp.arange(n_freq).reshape((1, n_freq, 1))
    weight = jnp.ones((1, n_freq, 1))
    for s in range(1, k + 1):
        weight = weight + wave ** (2 * s)
    weight = jnp.sqrt(weight)

    return _spectral_relative_l2(pred_fft, tgt_fft, weight)


LossFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def build_loss(loss_name: str) -> LossFn:
    if loss_name == "mse":
        return mse_loss
    if loss_name in ("relative_l2", "lp"):
        return relative_l2_loss
    if loss_name == "sobolev":
        return sobolev_loss
    raise ValueError(f"Unknown loss: {loss_name}")


def count_params(params) -> int:
    return sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))
