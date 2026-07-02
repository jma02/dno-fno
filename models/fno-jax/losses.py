from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def _relative_l2(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
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
    """Relative H^k Sobolev L2 in the spectral domain.

    For each sample:
        L = ||weight . (pred_fft - tgt_fft)||_2 / ||weight . tgt_fft||_2
    """
    nx = prediction.shape[1]
    n_freq = nx // 2 + 1
    prediction = prediction.reshape((prediction.shape[0], nx, -1))
    target = target.reshape((target.shape[0], nx, -1))

    pred_fft = jnp.fft.rfft(prediction, axis=1)
    tgt_fft = jnp.fft.rfft(target, axis=1)

    wave = jnp.arange(n_freq).reshape((1, n_freq, 1))
    weight_sq = jnp.ones((1, n_freq, 1))
    for s in range(1, k + 1):
        weight_sq = weight_sq + wave ** (2 * s)
    weight = jnp.sqrt(weight_sq)

    diff_w = (pred_fft - tgt_fft) * weight
    tgt_w = tgt_fft * weight
    diff_norm = jnp.sqrt(jnp.sum(jnp.abs(diff_w) ** 2, axis=(1, 2)))
    tgt_norm = jnp.sqrt(jnp.clip(jnp.sum(jnp.abs(tgt_w) ** 2, axis=(1, 2)), 1e-12))
    return jnp.mean(diff_norm / tgt_norm)


LossFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def build_loss(*, sobolev_k: int = 1) -> LossFn:
    return lambda pred, tgt: sobolev_loss(pred, tgt, k=sobolev_k)


def count_params(params) -> int:
    return sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))
