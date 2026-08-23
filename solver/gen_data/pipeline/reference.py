"""Frozen discrete Dirichlet--Neumann target for the paper dataset."""
from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp

from solver.solvers.dno_series_jax import build_grid, dno_series_eval


@dataclass(frozen=True)
class DiscreteDnoTarget:
    """Numerical target for the proposed regenerated-dataset model."""

    nx: int = 1024
    length: float = 2.0 * math.pi
    dno_order: int = 6
    pad_factor: int = 8
    maximum_wavenumber: float = 128.0

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.nx % 2:
            raise ValueError("nx must be a positive even integer")
        if self.length <= 0.0:
            raise ValueError("length must be positive")
        if self.dno_order < 0:
            raise ValueError("dno_order must be nonnegative")
        if self.pad_factor < 1:
            raise ValueError("pad_factor must be positive")
        nyquist = math.pi * self.nx / self.length
        if not 0.0 < self.maximum_wavenumber < nyquist:
            raise ValueError(
                "maximum_wavenumber must lie strictly below the Nyquist wavenumber"
            )


PAPER_DNO_TARGET = DiscreteDnoTarget()


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


def evaluate_discrete_dno_target(
    eta: jax.Array,
    xi: jax.Array,
    depth: float | jax.Array,
    *,
    definition: DiscreteDnoTarget = PAPER_DNO_TARGET,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the projected inputs and their frozen order-six DNO target."""

    if not jax.config.x64_enabled:
        raise RuntimeError("the paper DNO target requires JAX float64 mode")
    eta_array = jnp.asarray(eta, dtype=jnp.float64)
    xi_array = jnp.asarray(xi, dtype=jnp.float64)
    depth_array = jnp.asarray(depth, dtype=jnp.float64)
    if eta_array.shape != xi_array.shape:
        raise ValueError("eta and xi must have identical shapes")
    if eta_array.shape[-1] != definition.nx:
        raise ValueError(
            f"expected {definition.nx} grid points, got {eta_array.shape[-1]}"
        )

    _, wavenumbers = build_grid(definition.nx, definition.length)
    wavenumbers = jnp.asarray(wavenumbers, dtype=eta_array.dtype)
    eta_input = project_fixed_band(
        eta_array,
        wavenumbers,
        maximum_wavenumber=definition.maximum_wavenumber,
    )
    xi_input = project_fixed_band(
        xi_array,
        wavenumbers,
        maximum_wavenumber=definition.maximum_wavenumber,
    )
    target = dno_series_eval(
        eta_input,
        xi_input,
        wavenumbers,
        depth_array,
        definition.dno_order,
        pad_factor=definition.pad_factor,
    )
    target = project_fixed_band(
        target,
        wavenumbers,
        maximum_wavenumber=definition.maximum_wavenumber,
        remove_mean=True,
    )
    return eta_input, xi_input, target
