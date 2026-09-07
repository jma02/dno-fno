"""Compute the Dirichlet--Neumann targets stored in the paper dataset."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from solver.solvers.dno_series_jax import build_grid, dno_series_eval


def bandlimit_field(
    field: jax.Array,
    wavenumbers: jax.Array,
    *,
    maximum_wavenumber: float,
    remove_mean: bool = False,
) -> jax.Array:
    """Remove Fourier modes above the wavenumber cutoff, optionally zeroing the mean."""

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

    eta_array = jnp.asarray(eta, dtype=jnp.float64)
    xi_array = jnp.asarray(xi, dtype=jnp.float64)
    depth_array = jnp.asarray(depth, dtype=jnp.float64)

    _, wavenumbers = build_grid(nx, length)
    wavenumbers = jnp.asarray(wavenumbers, dtype=eta_array.dtype)
    eta_input = bandlimit_field(
        eta_array,
        wavenumbers,
        maximum_wavenumber=maximum_wavenumber,
    )
    xi_input = bandlimit_field(
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
    target = bandlimit_field(
        target,
        wavenumbers,
        maximum_wavenumber=maximum_wavenumber,
        remove_mean=True,
    )
    return eta_input, xi_input, target
