"""Construct tangent-Hermite Tanaka initial conditions for trajectory datasets."""

from __future__ import annotations

import json
import math

import jax
import jax.numpy as jnp
import numpy as np

from solver.gen_data.multi_crest import (
    CrestSpec,
    flatten_case_specs,
    specs_to_jax_arrays,
)
from solver.solvers.dno_series_jax import build_grid, myfft, myifft
from solver.solvers.time_integrator import spectral_dx
from solver.tanaka_ICs.modified_tanaka import (
    ModifiedTanakaParams,
    solve_modified_tanaka_batched,
)


TANAKA_FINE_FACTOR = 8
TANAKA_PROFILE_RECONSTRUCTION = "periodic_tangent_hermite_tan_theta_v1"
TANAKA_PROFILE_TOLERANCE = 1e-12
TANAKA_POTENTIAL_RADICAND_FAILURE_SCHEMA = (
    "tanaka_potential_radicand_failure_v1"
)


class TanakaPotentialRadicandError(ValueError):
    """A rejected real-valued Tanaka surface-potential construction."""

    def __init__(self, failure_record: dict[str, object]) -> None:
        serialized = json.dumps(
            failure_record,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self.failure_record: dict[str, object] = json.loads(serialized)
        super().__init__(serialized)


def cubic_hermite_zero_exterior(
    x_nodes: jax.Array,
    y_nodes: jax.Array,
    slopes: jax.Array,
    x_eval: jax.Array,
) -> jax.Array:
    """Evaluate a nonuniform cubic Hermite profile, returning zero off support."""
    inside = (x_eval >= x_nodes[0]) & (x_eval <= x_nodes[-1])
    safe_x = jnp.where(inside, x_eval, x_nodes[0])
    interval = jnp.searchsorted(
        x_nodes,
        safe_x,
        side="right",
        method="scan",
    ) - 1
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
    fine_factor: int = TANAKA_FINE_FACTOR,
) -> jax.Array:
    """Place one dimensionless Tanaka profile using its exact surface tangent."""
    midpoint = x_profile.shape[0] // 2
    eta_nodes = eta_profile.at[0].set(0.0).at[-1].set(0.0)
    slopes = (
        jnp.tan(theta_profile)
        .at[0]
        .set(0.0)
        .at[midpoint]
        .set(0.0)
        .at[-1]
        .set(0.0)
    )

    periodic_center = jnp.mod(center, length)
    shifts = length * jnp.arange(
        -image_radius,
        image_radius + 1,
        dtype=x_fine.dtype,
    )
    dimensionless_queries = (
        x_fine[None, :] + shifts[:, None] - periodic_center
    ) / depth
    eta_fine = depth * jnp.sum(
        cubic_hermite_zero_exterior(
            x_profile,
            eta_nodes,
            slopes,
            dimensionless_queries,
        ),
        axis=0,
    )

    eta_hat_fine = jnp.fft.rfft(eta_fine)
    eta_hat_native = eta_hat_fine[: nx // 2 + 1] / fine_factor
    return jnp.fft.irfft(eta_hat_native, n=nx).astype(eta_profile.dtype)


def tanaka_periodic_image_radius(
    x_profile: jax.Array,
    depth: jax.Array,
    length: float,
) -> int:
    """Return the images needed to periodize every supplied compact profile."""
    if length <= 0.0:
        raise ValueError("Periodic length must be positive.")
    if not bool(jnp.all(jnp.asarray(depth) > 0.0)):
        raise ValueError("Tanaka depths must be positive.")

    half_support = jnp.maximum(
        jnp.abs(x_profile[..., 0]),
        jnp.abs(x_profile[..., -1]),
    )
    maximum_physical_half_support = float(
        jnp.max(half_support * jnp.asarray(depth))
    )
    return math.floor(maximum_physical_half_support / length) + 1


def _validate_tanaka_profile_batch(
    x_profile: jax.Array,
    eta_profile: jax.Array,
    theta_profile: jax.Array,
) -> None:
    """Check assumptions needed for a compactly supported Hermite placement."""
    if (
        x_profile.ndim != 2
        or eta_profile.shape != x_profile.shape
        or theta_profile.shape != x_profile.shape
        or x_profile.shape[-1] < 3
        or x_profile.shape[-1] % 2 != 1
    ):
        raise ValueError(
            "Tanaka x, eta, and theta profiles must have the same two-dimensional "
            "shape with an odd number of at least three knots."
        )
    if not bool(
        jnp.all(jnp.isfinite(x_profile))
        & jnp.all(jnp.isfinite(eta_profile))
        & jnp.all(jnp.isfinite(theta_profile))
    ):
        raise ValueError("Tanaka profile contains nonfinite values.")
    if not bool(jnp.all(jnp.diff(x_profile, axis=-1) > 0.0)):
        raise ValueError("Tanaka x-profile knots must be strictly increasing.")
    if not bool(jnp.all(jnp.cos(theta_profile) > 0.0)):
        raise ValueError("Tanaka profile is not a graph over x.")

    slopes = jnp.tan(theta_profile)
    midpoint = x_profile.shape[-1] // 2
    maximum_join_defect = jnp.max(
        jnp.abs(
            jnp.concatenate(
                (
                    eta_profile[:, (0, -1)],
                    slopes[:, (0, midpoint, -1)],
                ),
                axis=-1,
            )
        )
    )
    if float(maximum_join_defect) > TANAKA_PROFILE_TOLERANCE:
        raise ValueError(
            "Tanaka profile does not join the zero exterior smoothly: "
            f"maximum endpoint/crest defect is {float(maximum_join_defect):.3e}."
        )


def _finite_or_none(value: float) -> float | None:
    """Return a finite JSON number, or ``None`` for a nonfinite value."""
    return value if math.isfinite(value) else None


def _tanaka_radicand_component_record(
    *,
    global_component_index: int,
    component_within_case: int,
    case_index: int,
    spec: CrestSpec,
    depth: float,
    unsigned_speed: float,
    speed_squared: float,
    radicand: np.ndarray,
    x_grid: np.ndarray,
) -> dict[str, object]:
    """Describe one invalid crest without emitting nonstandard JSON numbers."""
    finite = np.isfinite(radicand)
    negative = finite & (radicand < 0.0)
    nonfinite_indices = np.flatnonzero(~finite)

    minimum_index: int | None = None
    minimum_radicand: float | None = None
    minimum_radicand_x: float | None = None
    minimum_scaled_radicand: float | None = None
    if bool(np.all(finite)):
        minimum_index = int(np.argmin(radicand))
        minimum_radicand = float(radicand[minimum_index])
        minimum_radicand_x = _finite_or_none(float(x_grid[minimum_index]))
        if math.isfinite(speed_squared) and speed_squared > 0.0:
            minimum_scaled_radicand = _finite_or_none(
                minimum_radicand / speed_squared
            )

    first_nonfinite_index = (
        int(nonfinite_indices[0]) if nonfinite_indices.size else None
    )
    first_nonfinite_x = (
        _finite_or_none(float(x_grid[first_nonfinite_index]))
        if first_nonfinite_index is not None
        else None
    )
    return {
        "local_case_index": case_index,
        "component_within_case": component_within_case,
        "global_component_index": global_component_index,
        "alpha": _finite_or_none(float(spec.amplitude)),
        "center": _finite_or_none(float(spec.center)),
        "direction": int(spec.direction),
        "depth": _finite_or_none(depth),
        "unsigned_speed": _finite_or_none(unsigned_speed),
        "speed_squared": _finite_or_none(speed_squared),
        "minimum_radicand": minimum_radicand,
        "minimum_radicand_over_speed_squared": minimum_scaled_radicand,
        "minimum_radicand_grid_index": minimum_index,
        "minimum_radicand_x": minimum_radicand_x,
        "negative_count": int(np.count_nonzero(negative)),
        "nonfinite_count": int(nonfinite_indices.size),
        "first_nonfinite_grid_index": first_nonfinite_index,
        "first_nonfinite_x": first_nonfinite_x,
    }


def _validate_tanaka_surface_potential_radicand(
    radical: jax.Array | np.ndarray,
    *,
    x_grid: jax.Array | np.ndarray,
    speed_per_crest: jax.Array | np.ndarray,
    case_h_ref: np.ndarray,
    flat_specs: list[CrestSpec],
    crest_case_ids: np.ndarray,
    components_within_case: tuple[int, ...],
) -> None:
    """Reject every crest for which the surface-potential root is not real."""
    radical_host = np.asarray(jax.device_get(radical))
    x_host = np.asarray(jax.device_get(x_grid))
    speed_host, speed_squared_host = (
        np.asarray(value)
        for value in jax.device_get(
            (speed_per_crest, jnp.square(speed_per_crest))
        )
    )
    depths_host = np.asarray(case_h_ref)
    case_ids_host = np.asarray(crest_case_ids)

    component_count = len(flat_specs)
    if radical_host.ndim != 2 or radical_host.shape[0] != component_count:
        raise ValueError(
            "Tanaka radicand must have shape (number of crest components, nx)."
        )
    if x_host.shape != (radical_host.shape[1],):
        raise ValueError("Tanaka x grid must contain one point per radicand column.")
    if speed_host.shape != (component_count,):
        raise ValueError("Tanaka speed must contain one value per crest component.")
    if speed_squared_host.shape != speed_host.shape:
        raise ValueError("Tanaka squared speed shape does not match crest speeds.")
    if case_ids_host.shape != (component_count,):
        raise ValueError("Tanaka case ids must contain one value per crest component.")
    if len(components_within_case) != component_count:
        raise ValueError(
            "Tanaka within-case component ids must match the flattened crests."
        )
    if np.any(case_ids_host < 0) or np.any(case_ids_host >= depths_host.size):
        raise ValueError("Tanaka crest component refers to an absent local case.")

    speed_invalid = (~np.isfinite(speed_squared_host)) | (
        speed_squared_host <= 0.0
    )
    radicand_invalid = np.any(
        (~np.isfinite(radical_host))
        | (np.isfinite(radical_host) & (radical_host < 0.0)),
        axis=1,
    )
    if not bool(np.any(speed_invalid) or np.any(radicand_invalid)):
        return

    if bool(np.any(speed_invalid)):
        reason = "nonpositive_or_nonfinite_surface_potential_speed_squared"
        invalid_components = np.flatnonzero(speed_invalid | radicand_invalid)
    else:
        reason = "negative_or_nonfinite_surface_potential_radicand"
        invalid_components = np.flatnonzero(radicand_invalid)

    components = [
        _tanaka_radicand_component_record(
            global_component_index=int(component_index),
            component_within_case=components_within_case[component_index],
            case_index=int(case_ids_host[component_index]),
            spec=flat_specs[component_index],
            depth=float(depths_host[case_ids_host[component_index]]),
            unsigned_speed=float(speed_host[component_index]),
            speed_squared=float(speed_squared_host[component_index]),
            radicand=radical_host[component_index],
            x_grid=x_host,
        )
        for component_index in invalid_components
    ]
    raise TanakaPotentialRadicandError(
        {
            "schema": TANAKA_POTENTIAL_RADICAND_FAILURE_SCHEMA,
            "reason": reason,
            "invalid_case_indices": sorted(
                {
                    int(case_ids_host[component_index])
                    for component_index in invalid_components
                }
            ),
            "components": components,
        }
    )


def build_per_case_initial_conditions(
    *,
    template_params: ModifiedTanakaParams,
    case_h_ref: np.ndarray,
    case_specs: list[list[CrestSpec]],
    length: float,
    nx: int,
    gravity: float,
) -> tuple[jax.Array, jax.Array]:
    """Generate per-case tangent-Hermite Tanaka multicrest initial conditions."""
    flat_specs, crest_case_ids = flatten_case_specs(case_specs)
    flat_steepness, flat_centers, flat_directions = specs_to_jax_arrays(
        flat_specs
    )
    crest_case_ids_array = jnp.asarray(crest_case_ids)

    tanaka_batch = solve_modified_tanaka_batched(
        template_params,
        flat_steepness,
        centers=jnp.zeros_like(flat_steepness),
        directions=flat_directions,
    )
    x_profile = tanaka_batch.x_profile / template_params.depth
    eta_profile = tanaka_batch.eta_profile / template_params.depth
    theta_profile = tanaka_batch.theta_profile
    froude = tanaka_batch.froude
    _validate_tanaka_profile_batch(x_profile, eta_profile, theta_profile)

    crest_depths = jnp.asarray(
        case_h_ref,
        dtype=x_profile.dtype,
    )[crest_case_ids_array]
    crest_centers = jnp.asarray(flat_centers, dtype=x_profile.dtype)

    nx_fine = nx * TANAKA_FINE_FACTOR
    x_fine = (length / nx_fine) * jnp.arange(
        nx_fine,
        dtype=x_profile.dtype,
    )
    image_radius = tanaka_periodic_image_radius(
        x_profile,
        crest_depths,
        length,
    )
    eta_per_crest = jax.vmap(
        lambda x_row, eta_row, theta_row, depth, center: (
            place_tanaka_profile_periodic(
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
        )
    )(
        x_profile,
        eta_profile,
        theta_profile,
        crest_depths,
        crest_centers,
    )

    speed_per_crest = froude * jnp.sqrt(gravity * crest_depths)
    direction_array = jnp.asarray(
        flat_directions,
        dtype=eta_per_crest.dtype,
    )
    signed_speed = (direction_array * speed_per_crest)[..., None]
    direction_column = direction_array[..., None]

    x_grid, k_grid = build_grid(nx, length)
    eta_x = spectral_dx(eta_per_crest, k_grid)
    radical = (1.0 + eta_x**2) * (
        signed_speed**2 - 2.0 * gravity * eta_per_crest
    )
    components_within_case = tuple(
        component_index
        for specs in case_specs
        for component_index in range(len(specs))
    )
    _validate_tanaka_surface_potential_radicand(
        radical,
        x_grid=x_grid,
        speed_per_crest=speed_per_crest,
        case_h_ref=case_h_ref,
        flat_specs=flat_specs,
        crest_case_ids=crest_case_ids,
        components_within_case=components_within_case,
    )
    xi_x = signed_speed - direction_column * jnp.sqrt(radical)
    inverse_ik = jnp.where(k_grid != 0.0, 1.0 / (1j * k_grid), 0.0)
    xi_per_crest = myifft(inverse_ik * myfft(xi_x, nx))
    xi_per_crest -= jnp.mean(xi_per_crest, axis=-1, keepdims=True)

    batch_size = case_h_ref.shape[0]
    eta_case = jnp.zeros((batch_size, nx), dtype=eta_per_crest.dtype)
    xi_case = jnp.zeros((batch_size, nx), dtype=xi_per_crest.dtype)
    eta_case = eta_case.at[crest_case_ids_array].add(eta_per_crest)
    xi_case = xi_case.at[crest_case_ids_array].add(xi_per_crest)
    return eta_case, xi_case
