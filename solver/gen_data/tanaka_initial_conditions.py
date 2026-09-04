"""Construct tangent-Hermite Tanaka initial conditions."""

from __future__ import annotations

import math
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from solver.gen_data.tanaka_sampling import TanakaCrest
from solver.solvers.dno_series_jax import build_grid, myfft, myifft
from solver.solvers.time_integrator import spectral_dx
from solver.tanaka_ICs.modified_tanaka import (
    ModifiedTanakaParams,
    solve_modified_tanaka_batched,
)

TANAKA_FINE_FACTOR = 8
FloatArray: TypeAlias = NDArray[np.float64]


class TanakaPotentialRadicandError(ValueError):
    """Tanaka simulations whose real surface potential cannot be constructed."""

    def __init__(self, invalid_simulation_indices: tuple[int, ...]) -> None:
        self.invalid_simulation_indices = tuple(sorted(set(invalid_simulation_indices)))
        super().__init__(
            "negative Tanaka surface-potential radicand in simulations "
            f"{self.invalid_simulation_indices}"
        )


def cubic_hermite_zero_exterior(
    x_nodes: jax.Array,
    y_nodes: jax.Array,
    slopes: jax.Array,
    x_eval: jax.Array,
) -> jax.Array:
    """Evaluate a nonuniform cubic Hermite profile, returning zero off support."""

    inside = (x_eval >= x_nodes[0]) & (x_eval <= x_nodes[-1])
    safe_x = jnp.where(inside, x_eval, x_nodes[0])
    interval = jnp.searchsorted(x_nodes, safe_x, side="right", method="scan") - 1
    interval = jnp.clip(interval, 0, x_nodes.shape[0] - 2)

    x_left = x_nodes[interval]
    width = x_nodes[interval + 1] - x_left
    fraction = (safe_x - x_left) / width
    fraction_squared = fraction * fraction
    fraction_cubed = fraction_squared * fraction
    h00 = 2.0 * fraction_cubed - 3.0 * fraction_squared + 1.0
    h10 = fraction_cubed - 2.0 * fraction_squared + fraction
    h01 = -2.0 * fraction_cubed + 3.0 * fraction_squared
    h11 = fraction_cubed - fraction_squared
    value = (
        h00 * y_nodes[interval]
        + h10 * width * slopes[interval]
        + h01 * y_nodes[interval + 1]
        + h11 * width * slopes[interval + 1]
    )
    return jnp.where(inside, value, jnp.zeros_like(value))


def place_tanaka_profile_periodic(
    x_profile: jax.Array,
    eta_profile: jax.Array,
    theta_profile: jax.Array,
    depth: jax.Array,
    center: jax.Array,
    x_fine: jax.Array,
    *,
    length: float,
    nx: int,
    image_radius: int,
) -> jax.Array:
    """Place one dimensionless Tanaka profile using its exact surface tangent."""

    midpoint = x_profile.shape[0] // 2
    eta_nodes = eta_profile.at[0].set(0.0).at[-1].set(0.0)
    slopes = jnp.tan(theta_profile).at[0].set(0.0).at[midpoint].set(0.0).at[-1].set(0.0)
    shifts = length * jnp.arange(
        -image_radius,
        image_radius + 1,
        dtype=x_fine.dtype,
    )
    queries = (x_fine[None, :] + shifts[:, None] - jnp.mod(center, length)) / depth
    eta_fine = depth * jnp.sum(
        cubic_hermite_zero_exterior(
            x_profile,
            eta_nodes,
            slopes,
            queries,
        ),
        axis=0,
    )
    eta_hat_fine = jnp.fft.rfft(eta_fine)
    eta_hat_native = eta_hat_fine[: nx // 2 + 1] / TANAKA_FINE_FACTOR
    return jnp.fft.irfft(eta_hat_native, n=nx).astype(eta_profile.dtype)


def tanaka_periodic_image_radius(
    x_profile: jax.Array,
    depth: jax.Array,
    length: float,
) -> int:
    """Return the images needed to periodize every supplied compact profile."""

    half_support = jnp.maximum(
        jnp.abs(x_profile[..., 0]),
        jnp.abs(x_profile[..., -1]),
    )
    maximum_half_support = float(jnp.max(half_support * jnp.asarray(depth)))
    return math.floor(maximum_half_support / length) + 1


def build_tanaka_initial_conditions(
    template: ModifiedTanakaParams,
    depths: FloatArray,
    crests_by_simulation: tuple[tuple[TanakaCrest, ...], ...],
    *,
    length: float,
    nx: int,
    gravity: float,
) -> tuple[jax.Array, jax.Array]:
    """Build one periodic multicrest initial state per simulation."""

    flat_crests = tuple(
        crest
        for simulation_crests in crests_by_simulation
        for crest in simulation_crests
    )
    crest_simulation_ids = np.asarray(
        [
            simulation_index
            for simulation_index, simulation_crests in enumerate(crests_by_simulation)
            for _ in simulation_crests
        ],
        dtype=np.int32,
    )
    crest_simulation_ids_device = jnp.asarray(crest_simulation_ids)
    steepness = jnp.asarray([crest.alpha for crest in flat_crests], dtype=jnp.float64)
    centers = jnp.asarray([crest.center for crest in flat_crests], dtype=jnp.float64)
    directions = jnp.asarray(
        [crest.direction for crest in flat_crests], dtype=jnp.float64
    )

    tanaka = solve_modified_tanaka_batched(
        template,
        steepness,
        centers=jnp.zeros_like(steepness),
        directions=directions,
    )
    x_profile = tanaka.x_profile / template.depth
    eta_profile = tanaka.eta_profile / template.depth
    theta_profile = tanaka.theta_profile
    crest_depths = jnp.asarray(depths, dtype=x_profile.dtype)[
        crest_simulation_ids_device
    ]

    nx_fine = nx * TANAKA_FINE_FACTOR
    x_fine = (length / nx_fine) * jnp.arange(nx_fine, dtype=x_profile.dtype)
    image_radius = tanaka_periodic_image_radius(
        x_profile,
        crest_depths,
        length,
    )
    eta_per_crest = jax.vmap(
        lambda x_row, eta_row, theta_row, depth, center: place_tanaka_profile_periodic(
            x_row,
            eta_row,
            theta_row,
            depth,
            center,
            x_fine,
            length=length,
            nx=nx,
            image_radius=image_radius,
        )
    )(
        x_profile,
        eta_profile,
        theta_profile,
        crest_depths,
        jnp.asarray(centers, dtype=x_profile.dtype),
    )

    speed_per_crest = tanaka.froude * jnp.sqrt(gravity * crest_depths)
    direction_array = jnp.asarray(directions, dtype=eta_per_crest.dtype)
    signed_speed = (direction_array * speed_per_crest)[..., None]
    direction_column = direction_array[..., None]
    _, wavenumbers = build_grid(nx, length)
    eta_x = spectral_dx(eta_per_crest, wavenumbers)
    radicand = (1.0 + eta_x**2) * (signed_speed**2 - 2.0 * gravity * eta_per_crest)
    radicand_host, speed_squared_host = (
        np.asarray(value)
        for value in jax.device_get((radicand, jnp.square(speed_per_crest)))
    )
    if not np.all(np.isfinite(speed_squared_host) & (speed_squared_host > 0.0)):
        raise ValueError("Tanaka wave speeds must be finite and positive")
    if not np.isfinite(radicand_host).all():
        raise ValueError("Tanaka surface-potential radicand must be finite")
    invalid_crests = np.flatnonzero(np.any(radicand_host < 0.0, axis=1))
    if invalid_crests.size:
        raise TanakaPotentialRadicandError(
            tuple(np.unique(crest_simulation_ids[invalid_crests]).astype(int).tolist())
        )

    xi_x = signed_speed - direction_column * jnp.sqrt(radicand)
    inverse_ik = jnp.where(wavenumbers != 0.0, 1.0 / (1j * wavenumbers), 0.0)
    xi_per_crest = myifft(inverse_ik * myfft(xi_x, nx))
    xi_per_crest -= jnp.mean(xi_per_crest, axis=-1, keepdims=True)

    batch_size = len(crests_by_simulation)
    eta = jnp.zeros((batch_size, nx), dtype=eta_per_crest.dtype)
    xi = jnp.zeros((batch_size, nx), dtype=xi_per_crest.dtype)
    eta = eta.at[crest_simulation_ids_device].add(eta_per_crest)
    xi = xi.at[crest_simulation_ids_device].add(xi_per_crest)
    return eta, xi
