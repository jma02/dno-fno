"""Compute the Dirichlet--Neumann targets stored in the paper dataset."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from solver.solvers.dno_series_jax import build_grid, dno_series_eval


def project_fixed_band(
    field: jax.Array,
    wavenumbers: jax.Array,
    *,
    maximum_wavenumber: float,
) -> jax.Array:
    """Remove Fourier modes above the physical-wavenumber cutoff."""

    coefficients = jnp.fft.fft(field, axis=-1)
    mask = jnp.abs(wavenumbers) <= maximum_wavenumber
    projected = jnp.where(mask, coefficients, 0.0)
    return jnp.fft.ifft(projected, axis=-1).real


@partial(
    jax.jit,
    static_argnames=("nx", "length", "dno_order", "pad_factor", "maximum_wavenumber"),
)
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
    )
    target = target - target.mean(axis=-1, keepdims=True)
    return eta_input, xi_input, target
