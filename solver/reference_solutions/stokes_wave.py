from __future__ import annotations

from typing import TypeAlias, cast

import jax
import jax.numpy as jnp

from ..solvers.dno_series_jax import build_grid, dno_series_eval


FINITE_DEPTH_STOKES_URSELL_LIMIT = 26.0
FINITE_DEPTH_STOKES_HEIGHT_PHASE_POINTS = 4096
StokesScalar: TypeAlias = float | jax.Array


def _finite_depth_coeffs(
    k0: StokesScalar,
    depth: StokesScalar,
    gravity: StokesScalar,
    a0: StokesScalar,
) -> dict[str, StokesScalar]:
    eps = k0 * a0
    eps2 = eps**2
    eps3 = eps**3
    eps4 = eps**4

    sigma = jnp.tanh(k0 * depth)
    alfa1 = jnp.cosh(2.0 * k0 * depth)

    om0 = jnp.sqrt(gravity * k0 * sigma)
    om2 = 0.25 * (2.0 * alfa1**2 + 7.0) / (alfa1 - 1.0) ** 2
    om4 = (
        20.0 * alfa1**5
        + 112.0 * alfa1**4
        - 100.0 * alfa1**3
        - 68.0 * alfa1**2
        - 211.0 * alfa1
        + 328.0
    ) / (32.0 * (alfa1 - 1.0) ** 5)
    om = om0 * (1.0 + eps2 * om2 + eps4 * om4)

    b31 = (3.0 + 8.0 * sigma**2 - 9.0 * sigma**4) / (16.0 * sigma**4)
    b51 = (
        121.0 * alfa1**5
        + 263.0 * alfa1**4
        + 376.0 * alfa1**3
        - 1999.0 * alfa1**2
        + 2509.0 * alfa1
        - 1108.0
    ) / (192.0 * (alfa1 - 1.0) ** 5)
    b22 = 0.25 * (3.0 - sigma**2) / sigma**3
    denom = 24.0 * jnp.sinh(2.0 * k0 * depth) * (3.0 * alfa1 + 2.0) * (alfa1 - 1.0) ** 4
    b42 = (
        60.0 * alfa1**6
        + 232.0 * alfa1**5
        - 118.0 * alfa1**4
        - 989.0 * alfa1**3
        - 607.0 * alfa1**2
        + 352.0 * alfa1
        + 260.0
    ) / denom
    b33 = (27.0 - 9.0 * sigma**2 + 9.0 * sigma**4 - 3.0 * sigma**6) / (64.0 * sigma**6)
    denom = 128.0 * (3.0 * alfa1 + 2.0) * (alfa1 - 1.0) ** 6
    b53 = (
        9.0
        * (
            57.0 * alfa1**7
            + 204.0 * alfa1**6
            - 53.0 * alfa1**5
            - 782.0 * alfa1**4
            - 741.0 * alfa1**3
            - 52.0 * alfa1**2
            + 371.0 * alfa1
            + 186.0
        )
        / denom
    )
    denom = 24.0 * jnp.sinh(2.0 * k0 * depth) * (3.0 * alfa1 + 2.0) * (alfa1 - 1.0) ** 4
    b44 = (
        24.0 * alfa1**6
        + 116.0 * alfa1**5
        + 214.0 * alfa1**4
        + 188.0 * alfa1**3
        + 133.0 * alfa1**2
        + 101.0 * alfa1
        + 34.0
    ) / denom
    denom = 384.0 * (12.0 * alfa1**2 + 11.0 * alfa1 + 2.0) * (alfa1 - 1.0) ** 6
    b55 = (
        5.0
        * (
            300.0 * alfa1**8
            + 1579.0 * alfa1**7
            + 3176.0 * alfa1**6
            + 2949.0 * alfa1**5
            + 1188.0 * alfa1**4
            + 675.0 * alfa1**3
            + 1326.0 * alfa1**2
            + 827.0 * alfa1
            + 130.0
        )
        / denom
    )

    a11 = 1.0 / jnp.sinh(k0 * depth)
    a22 = 3.0 / (8.0 * jnp.sinh(k0 * depth) ** 4)
    a42 = (12.0 * alfa1**4 + 22.0 * alfa1**3 - 84.0 * alfa1**2 - 135.0 * alfa1 + 104.0) / (
        24.0 * (alfa1 - 1.0) ** 5
    )
    a33 = (9.0 - 4.0 * jnp.sinh(k0 * depth) ** 2) / (64.0 * jnp.sinh(k0 * depth) ** 7)
    denom = 64.0 * jnp.sinh(k0 * depth) * (3.0 * alfa1 + 2.0) * (alfa1 - 1.0) ** 6
    a53 = (
        8.0 * alfa1**6
        + 138.0 * alfa1**5
        + 384.0 * alfa1**4
        - 568.0 * alfa1**3
        - 2388.0 * alfa1**2
        + 237.0 * alfa1
        + 974.0
    ) / denom
    denom = 48.0 * (3.0 * alfa1 + 2.0) * (alfa1 - 1.0) ** 5
    a44 = (10.0 * alfa1**3 - 174.0 * alfa1**2 + 291.0 * alfa1 + 278.0) / denom
    denom = 64.0 * jnp.sinh(k0 * depth) * (3.0 * alfa1 + 2.0) * (4.0 * alfa1 + 1.0) * (alfa1 - 1.0) ** 6
    a55 = (-6.0 * alfa1**5 + 272.0 * alfa1**4 - 1552.0 * alfa1**3 + 852.0 * alfa1**2 + 2029.0 * alfa1 + 430.0) / denom
    c2 = 0.25 * (sigma**2 - 1.0) / sigma
    c4 = -9.0 / (4.0 * jnp.sinh(2.0 * k0 * depth) * (alfa1 - 1.0) ** 3)

    return {
        "eps": eps,
        "eps2": eps2,
        "eps3": eps3,
        "eps4": eps4,
        "sigma": sigma,
        "om0": om0,
        "om": om,
        "B31": b31,
        "B51": b51,
        "B22": b22,
        "B42": b42,
        "B33": b33,
        "B53": b53,
        "B44": b44,
        "B55": b55,
        "A11": a11,
        "A22": a22,
        "A42": a42,
        "A33": a33,
        "A53": a53,
        "A44": a44,
        "A55": a55,
        "C2": c2,
        "C4": c4,
    }


def finite_depth_eta_harmonics(
    k0: float | jax.Array,
    depth: float | jax.Array,
    gravity: float | jax.Array,
    a0: float | jax.Array,
) -> jax.Array:
    """Return the five elevation harmonics of the finite-depth expansion."""

    coeffs = _finite_depth_coeffs(k0, depth, gravity, a0)
    harmonic_factors = jnp.stack(
        (
            1.0 + coeffs["eps2"] * coeffs["B31"] + coeffs["eps4"] * coeffs["B51"],
            coeffs["eps"]
            * (coeffs["B22"] + coeffs["eps2"] * coeffs["B42"]),
            coeffs["eps2"]
            * (coeffs["B33"] + coeffs["eps2"] * coeffs["B53"]),
            coeffs["eps3"] * coeffs["B44"],
            coeffs["eps4"] * coeffs["B55"],
        ),
        axis=-1,
    )
    return jnp.asarray(a0)[..., None] * harmonic_factors


def finite_depth_stokes_wave_height(
    k0: float | jax.Array,
    depth: float | jax.Array,
    gravity: float | jax.Array,
    a0: float | jax.Array,
    phase_points: int = FINITE_DEPTH_STOKES_HEIGHT_PHASE_POINTS,
) -> jax.Array:
    """Return a rigorous upper bound for the profile's continuous wave height.

    The trigonometric polynomial is sampled at equally spaced phases.  If
    ``B = sum(m^2 |E_m|)``, then ``B`` bounds the absolute second derivative.
    Taylor's theorem bounds the missed height between grid points by
    ``B * (2 pi / phase_points)^2 / 4``.
    """

    harmonics = finite_depth_eta_harmonics(k0, depth, gravity, a0)
    phases = jnp.linspace(
        0.0,
        2.0 * jnp.pi,
        phase_points,
        endpoint=False,
    )
    modes = jnp.arange(1, 6, dtype=phases.dtype)
    elevation = jnp.sum(
        harmonics[..., :, None] * jnp.cos(modes[:, None] * phases),
        axis=-2,
    )
    sampled_height = jnp.max(elevation, axis=-1) - jnp.min(
        elevation, axis=-1
    )
    curvature_bound = jnp.sum(
        modes**2 * jnp.abs(harmonics),
        axis=-1,
    )
    phase_spacing = 2.0 * jnp.pi / phase_points
    return sampled_height + 0.25 * curvature_bound * phase_spacing**2


def finite_depth_stokes_ursell_upper_bound(
    k0: float | jax.Array,
    depth: float | jax.Array,
    gravity: float | jax.Array,
    a0: float | jax.Array,
) -> jax.Array:
    """Return a conservative upper bound for ``H lambda^2 / h^3``."""

    wave_height = finite_depth_stokes_wave_height(
        k0,
        depth,
        gravity,
        a0,
    )
    wavelength = 2.0 * jnp.pi / jnp.asarray(k0)
    return wave_height * wavelength**2 / jnp.asarray(depth) ** 3


def finite_depth_stokes_in_ursell_support(
    k0: float | jax.Array,
    depth: float | jax.Array,
    gravity: float | jax.Array,
    a0: float | jax.Array,
) -> jax.Array:
    """Return whether fifth-order Stokes theory is used within ``Ur <= 26``."""

    ursell_upper_bound = finite_depth_stokes_ursell_upper_bound(
        k0,
        depth,
        gravity,
        a0,
    )
    return jnp.isfinite(ursell_upper_bound) & (
        ursell_upper_bound <= FINITE_DEPTH_STOKES_URSELL_LIMIT
    )


def _deep_stokes_eta_xi_from_theta(
    theta: jax.Array,
    *,
    k0: float | jax.Array,
    a0: float | jax.Array,
    gravity: float | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    eps = k0 * a0
    eps2 = eps**2
    eps3 = eps**3
    eps4 = eps**4
    om0 = jnp.sqrt(gravity * k0)
    eta = (
        (1.0 + eps2 / 8.0 + 121.0 * eps4 / 192.0) * jnp.cos(theta)
        + (0.5 * eps + 5.0 * eps3 / 6.0) * jnp.cos(2.0 * theta)
        + (3.0 * eps2 / 8.0 + 171.0 * eps4 / 128.0)
        * jnp.cos(3.0 * theta)
        + eps3 * jnp.cos(4.0 * theta) / 3.0
        + 125.0 * eps4 * jnp.cos(5.0 * theta) / 384.0
    )
    eta = a0 * eta
    xi = jnp.exp(k0 * eta) * jnp.sin(theta)
    xi = xi + 0.5 * eps3 * jnp.exp(2.0 * k0 * eta) * jnp.sin(
        2.0 * theta
    )
    xi = (
        xi
        + eps4
        * jnp.exp(3.0 * k0 * eta)
        * jnp.sin(3.0 * theta)
        / 12.0
    )
    xi = a0 * om0 * xi / k0
    return eta, xi


def _finite_stokes_eta_xi_from_theta(
    theta: jax.Array,
    *,
    k0: float | jax.Array,
    a0: float | jax.Array,
    depth: float | jax.Array,
    gravity: float | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    coeffs = _finite_depth_coeffs(k0, depth, gravity, a0)
    eta_harmonics = finite_depth_eta_harmonics(k0, depth, gravity, a0)
    eta = sum(
        eta_harmonics[mode - 1] * jnp.cos(mode * theta)
        for mode in range(1, 6)
    )
    xi = (
        jnp.cosh(k0 * (eta + depth))
        * (coeffs["A11"] * jnp.sin(theta))
        + coeffs["eps"]
        * (coeffs["A22"] + coeffs["eps2"] * coeffs["A42"])
        * jnp.cosh(2.0 * k0 * (eta + depth))
        * jnp.sin(2.0 * theta)
        + coeffs["eps2"]
        * (coeffs["A33"] + coeffs["eps2"] * coeffs["A53"])
        * jnp.cosh(3.0 * k0 * (eta + depth))
        * jnp.sin(3.0 * theta)
        + coeffs["eps3"]
        * coeffs["A44"]
        * jnp.cosh(4.0 * k0 * (eta + depth))
        * jnp.sin(4.0 * theta)
        + coeffs["eps4"]
        * coeffs["A55"]
        * jnp.cosh(5.0 * k0 * (eta + depth))
        * jnp.sin(5.0 * theta)
    )
    xi = a0 * coeffs["om0"] * xi / k0
    return cast(jax.Array, eta), xi


def stokes_eta_xi_at_phase(
    x: jax.Array,
    phase: float | jax.Array,
    n0: int | jax.Array,
    a0: float | jax.Array,
    length: float,
    depth: float | jax.Array,
    gravity: float,
    ichoi: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate one fifth-order Stokes snapshot at a prescribed phase.

    The phase is sampled directly, so changing the amplitude during a support
    redraw does not change the spatial translation.  The spatially constant
    potential gauge is removed from ``xi``.
    """

    k0 = n0 * (2.0 * jnp.pi / length)
    theta = k0 * x + phase
    if ichoi == 0:
        eta, xi = _deep_stokes_eta_xi_from_theta(
            theta,
            k0=k0,
            a0=a0,
            gravity=gravity,
        )
    else:
        eta, xi = _finite_stokes_eta_xi_from_theta(
            theta,
            k0=k0,
            a0=a0,
            depth=depth,
            gravity=gravity,
        )
    return eta, xi - jnp.mean(xi)


def stokes_eta_xi(
    x: jnp.ndarray,
    time: jnp.ndarray,
    n0: int | jax.Array,
    a0: float | jax.Array,
    length: float,
    depth: float,
    gravity: float,
    ichoi: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    k0 = n0 * (2.0 * jnp.pi / length)
    eps = k0 * a0
    eps4 = eps**4

    if ichoi == 0:
        om0 = jnp.sqrt(gravity * k0)
        om = om0 * (1.0 + 0.5 * eps**2 + 5.0 * eps4 / 8.0)
        theta = k0 * x - om * time
        return _deep_stokes_eta_xi_from_theta(
            theta,
            k0=k0,
            a0=a0,
            gravity=gravity,
        )

    coeffs = _finite_depth_coeffs(k0, depth, gravity, a0)
    theta = k0 * x - coeffs["om"] * time
    eta, xi = _finite_stokes_eta_xi_from_theta(
        theta,
        k0=k0,
        a0=a0,
        depth=depth,
        gravity=gravity,
    )
    gauge = (
        (coeffs["eps"] * coeffs["C2"] + coeffs["eps3"] * coeffs["C4"])
        * coeffs["om0"]
        * time
        / coeffs["sigma"]
    )
    return eta, xi + a0 * coeffs["om0"] * gauge / k0


def stokes_truth_trajectory(
    nx: int = 1024,
    length: float = 164.0,
    depth: float = 1.0,
    gravity: float = 1.0,
    dt: float = 0.1,
    tmax: float = 20.0,
    n0: int = 14,
    a0: float = 0.1,
    ichoi: int = 1,
    gxi_order: int = 6,
    pad_factor: int = 8,
) -> dict[str, jnp.ndarray | float | int | str]:
    x, k = build_grid(nx, length)
    times = jnp.arange(0.0, tmax + 0.5 * dt, dt)

    eta, xi = jax.vmap(
        lambda tau: stokes_eta_xi(
            x,
            tau,
            n0=n0,
            a0=a0,
            length=length,
            depth=depth,
            gravity=gravity,
            ichoi=ichoi,
        )
    )(times)

    gxi = jax.vmap(
        lambda eta_row, xi_row: dno_series_eval(
            eta_row,
            xi_row,
            k,
            depth if ichoi == 1 else 1000.0,
            gxi_order,
            pad_factor=pad_factor,
        )
    )(eta, xi)

    regime = "finite" if ichoi == 1 else "deep"
    return {
        "name": f"stokes_{regime}_n{n0}_a{a0:.3f}",
        "x": x,
        "t": times,
        "eta": eta,
        "xi": xi,
        "gxi": gxi,
        "nx": nx,
        "length": length,
        "depth": depth,
        "gravity": gravity,
        "dt": dt,
        "tmax": tmax,
        "n0": n0,
        "a0": a0,
        "ichoi": ichoi,
    }
