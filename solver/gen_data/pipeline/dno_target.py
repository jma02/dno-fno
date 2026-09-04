"""Compute the Dirichlet--Neumann targets stored in the paper dataset."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from solver.solvers.dno_series_jax import build_grid, dno_series_eval


def project_fixed_band(
    field: jax.Array,
    wavenumbers: jax.Array,
    *,
    maximum_wavenumber: float,
    remove_mean: bool = False,
) -> jax.Array:
    """Apply the sharp physical-wavenumber projection used by the target."""

    coefficients = jnp.fft.fft(field, axis=-1)
    mask = jnp.abs(wavenumbers) <= maximum_wavenumber
    projected = jnp.where(mask, coefficients, 0.0)
    if remove_mean:
        projected = projected.at[..., 0].set(0.0)
    return jnp.fft.ifft(projected, axis=-1).real


def compute_dno_target(
    eta: jax.Array,
    xi: jax.Array,
    depth: float | jax.Array,
    *,
    nx: int,
    length: float,
    dno_order: int,
    pad_factor: int,
    maximum_wavenumber: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return projected inputs and their discrete DNO target."""

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("the paper DNO target requires JAX float64 mode")
    eta_array = jnp.asarray(eta, dtype=jnp.float64)
    xi_array = jnp.asarray(xi, dtype=jnp.float64)
    depth_array = jnp.asarray(depth, dtype=jnp.float64)
    if eta_array.shape != xi_array.shape:
        raise ValueError("eta and xi must have identical shapes")
    if eta_array.shape[-1] != nx:
        raise ValueError(f"expected {nx} grid points, got {eta_array.shape[-1]}")

    _, wavenumbers = build_grid(nx, length)
    wavenumbers = jnp.asarray(wavenumbers, dtype=eta_array.dtype)
    eta_input = project_fixed_band(
        eta_array,
        wavenumbers,
        maximum_wavenumber=maximum_wavenumber,
    )
    xi_input = project_fixed_band(
        xi_array,
        wavenumbers,
        maximum_wavenumber=maximum_wavenumber,
    )
    target = dno_series_eval(
        eta_input,
        xi_input,
        wavenumbers,
        depth_array,
        dno_order,
        pad_factor=pad_factor,
    )
    target = project_fixed_band(
        target,
        wavenumbers,
        maximum_wavenumber=maximum_wavenumber,
        remove_mean=True,
    )
    return eta_input, xi_input, target
