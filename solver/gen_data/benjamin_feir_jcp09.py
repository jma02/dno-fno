"""Deep-water Benjamin--Feir initial conditions in the form of JCP09 (33).

A simulation is described by a carrier mode, first-harmonic carrier steepness,
symmetric sideband offset, one relative sideband amplitude, and one global
translation.  In the translated frame, both sidebands have the JCP09 phase
shift ``-pi/4``.  The discrete modes are restricted to the leading deep-water
modulational-instability band.  JCP09 used a numerically computed steady
Stokes carrier; this module uses the project's analytic fifth-order deep-water
carrier and does not claim that stronger equivalence.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from ..reference_solutions.stokes_wave import stokes_eta_xi

DEEP_WATER_MINIMUM_KH = 5.0
JCP09_RELATIVE_SIDEBAND_PHASE = -math.pi / 4.0


def deep_water_proxy_depth(length: float) -> float:
    """Choose finite depth so every nonzero Fourier mode has ``kh >= 5``."""

    return DEEP_WATER_MINIMUM_KH * length / (2.0 * math.pi)


def _initial_condition(
    x: jax.Array,
    *,
    carrier_mode: jax.Array,
    sideband_offset: jax.Array,
    carrier_steepness: jax.Array,
    perturbation_ratio: jax.Array,
    translation: jax.Array,
    length: float,
    gravity: float,
) -> tuple[jax.Array, jax.Array]:
    """Construct one JCP09 equation-(33) state with the project carrier."""

    translated_x = x - translation
    dtype = x.dtype
    carrier_wavenumber = carrier_mode.astype(dtype) * (2.0 * jnp.pi / length)
    carrier_amplitude = carrier_steepness / carrier_wavenumber
    bare_amplitude = carrier_amplitude
    # Invert the fifth-order carrier's first-harmonic amplitude correction.
    for _ in range(8):
        bare_steepness = carrier_wavenumber * bare_amplitude
        correction = 1.0 + bare_steepness**2 / 8.0 + 121.0 * bare_steepness**4 / 192.0
        bare_amplitude = carrier_amplitude / correction
    eta_carrier, xi_carrier = stokes_eta_xi(
        x=translated_x,
        time=jnp.asarray(0.0, dtype=dtype),
        n0=carrier_mode,
        a0=bare_amplitude,
        length=length,
        depth=deep_water_proxy_depth(length),
        gravity=gravity,
        ichoi=0,
    )
    fundamental = 2.0 * jnp.pi / length
    left_wavenumber = (carrier_mode - sideband_offset).astype(dtype) * fundamental
    right_wavenumber = (carrier_mode + sideband_offset).astype(dtype) * fundamental
    sideband_amplitude = perturbation_ratio * carrier_amplitude
    left_phase = left_wavenumber * translated_x + JCP09_RELATIVE_SIDEBAND_PHASE
    right_phase = right_wavenumber * translated_x + JCP09_RELATIVE_SIDEBAND_PHASE
    eta = (
        eta_carrier
        + sideband_amplitude * jnp.cos(left_phase)
        + sideband_amplitude * jnp.cos(right_phase)
    )
    xi = (
        xi_carrier
        + sideband_amplitude
        * jnp.sqrt(gravity / left_wavenumber)
        * jnp.exp(left_wavenumber * eta)
        * jnp.sin(left_phase)
        + sideband_amplitude
        * jnp.sqrt(gravity / right_wavenumber)
        * jnp.exp(right_wavenumber * eta)
        * jnp.sin(right_phase)
    )
    return eta, xi - jnp.mean(xi)


def build_initial_conditions(
    *,
    x: jax.Array | NDArray[np.float64],
    carrier_modes: jax.Array | NDArray[np.int32],
    sideband_offsets: jax.Array | NDArray[np.int32],
    carrier_steepnesses: jax.Array | NDArray[np.float64],
    perturbation_ratios: jax.Array | NDArray[np.float64],
    translations: jax.Array | NDArray[np.float64],
    length: float,
    gravity: float,
) -> tuple[jax.Array, jax.Array]:
    """Vectorize the deep-water JCP09 construction over a simulation batch."""

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("Benjamin--Feir construction requires JAX float64")
    x_array = jnp.asarray(x, dtype=jnp.float64)
    return jax.vmap(
        lambda mode, offset, steepness, ratio, translation: _initial_condition(
            x_array,
            carrier_mode=mode,
            sideband_offset=offset,
            carrier_steepness=steepness,
            perturbation_ratio=ratio,
            translation=translation,
            length=length,
            gravity=gravity,
        )
    )(
        jnp.asarray(carrier_modes, dtype=jnp.int32),
        jnp.asarray(sideband_offsets, dtype=jnp.int32),
        jnp.asarray(carrier_steepnesses, dtype=jnp.float64),
        jnp.asarray(perturbation_ratios, dtype=jnp.float64),
        jnp.asarray(translations, dtype=jnp.float64),
    )
