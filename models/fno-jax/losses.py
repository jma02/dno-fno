from __future__ import annotations

import jax
import jax.numpy as jnp


def _complex_lp_norm(values: jnp.ndarray, p: int = 2) -> jnp.ndarray:
    flat = values.reshape((values.shape[0], -1))
    return jnp.sum(jnp.abs(flat) ** p, axis=1) ** (1.0 / p)


def _relative_lp(prediction: jnp.ndarray, target: jnp.ndarray, p: int = 2) -> jnp.ndarray:
    error = prediction.reshape((prediction.shape[0], -1)) - target.reshape((target.shape[0], -1))
    baseline = target.reshape((target.shape[0], -1))
    error_norm = _complex_lp_norm(error, p=p)
    baseline_norm = jnp.clip(_complex_lp_norm(baseline, p=p), a_min=1e-12)
    return jnp.mean(error_norm / baseline_norm)


def lp_loss(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return _relative_lp(prediction, target, p=2)


def hs_loss(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    k: int = 1,
    weights: tuple[float, ...] | None = None,
    group: bool = False,
) -> jnp.ndarray:
    if k < 1:
        raise ValueError("k must be at least 1")

    a = weights or tuple(1.0 for _ in range(k))
    nx = prediction.shape[1]
    prediction = prediction.reshape((prediction.shape[0], nx, -1))
    target = target.reshape((target.shape[0], nx, -1))

    wave_numbers = jnp.concatenate(
        (jnp.arange(0, nx // 2), jnp.arange(-nx // 2, 0))
    )
    wave_numbers = jnp.abs(wave_numbers).reshape((1, nx, 1))

    prediction_fft = jnp.fft.fftn(prediction, axes=(1,))
    target_fft = jnp.fft.fftn(target, axes=(1,))

    if not group:
        weight = jnp.ones((1, nx, 1), dtype=prediction.dtype)
        if k >= 1:
            weight = weight + a[0] ** 2 * wave_numbers**2
        if k >= 2:
            weight = weight + a[1] ** 2 * wave_numbers**4
        return _relative_lp(
            prediction_fft * jnp.sqrt(weight),
            target_fft * jnp.sqrt(weight),
            p=2,
        )

    loss = _relative_lp(prediction_fft, target_fft, p=2)
    if k >= 1:
        first_order = a[0] * jnp.sqrt(wave_numbers**2)
        loss = loss + _relative_lp(
            prediction_fft * first_order,
            target_fft * first_order,
            p=2,
        )
    if k >= 2:
        second_order = a[1] * jnp.sqrt(wave_numbers**4)
        loss = loss + _relative_lp(
            prediction_fft * second_order,
            target_fft * second_order,
            p=2,
        )
    return loss / (k + 1)


def build_loss(loss_name: str):
    if loss_name == "sobolev":
        return hs_loss
    if loss_name == "lp":
        return lp_loss
    raise ValueError(f"Unknown loss_name: {loss_name}")


def count_params(params) -> int:
    return sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))
