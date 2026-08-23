"""Deep-water Benjamin--Feir initial conditions in the form of JCP09 (33).

A case is described by a carrier mode, first-harmonic carrier steepness,
symmetric sideband offset, one relative sideband amplitude, and one global
translation.  In the translated frame, both sidebands have the JCP09 phase
shift ``-pi/4``.  The discrete modes are restricted to the leading deep-water
modulational-instability band.  JCP09 used a numerically computed steady
Stokes carrier; this module uses the project's analytic fifth-order deep-water
carrier and does not claim that stronger equivalence.
"""
from __future__ import annotations

import math
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from ..reference_solutions.stokes_wave import stokes_eta_xi

ParameterArrays: TypeAlias = dict[str, np.ndarray]

CARRIER_MODE_MIN = 4
CARRIER_MODE_MAX = 20
CARRIER_STEEPNESS_MIN = 0.05
CARRIER_STEEPNESS_MAX = 0.13
PERTURBATION_RATIO_MIN = 0.05
PERTURBATION_RATIO_MAX = 0.20
DEEP_WATER_MINIMUM_KH = 5.0
JCP09_RELATIVE_SIDEBAND_PHASE = -math.pi / 4.0
BENJAMIN_FEIR_CONSTRUCTOR = (
    "jcp09_equation_33_with_project_fifth_order_carrier_v2"
)


def deep_water_proxy_depth(
    length: float,
    *,
    minimum_kh: float = DEEP_WATER_MINIMUM_KH,
) -> float:
    """Return a depth for which the first Fourier mode has ``kh=minimum_kh``."""

    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("length must be finite and positive")
    if not math.isfinite(minimum_kh) or minimum_kh <= 0.0:
        raise ValueError("minimum_kh must be finite and positive")
    return minimum_kh * length / (2.0 * math.pi)


def instability_band_fraction(
    carrier_mode: int | np.ndarray,
    sideband_offset: int | np.ndarray,
    carrier_steepness: float | np.ndarray,
) -> np.ndarray:
    """Return ``(Delta n/n_c)/(2 sqrt(2) epsilon_c)``.

    Leading deep-water NLS theory predicts modulational instability when this
    number is strictly between zero and one.
    """

    carrier = np.asarray(carrier_mode)
    offset = np.asarray(sideband_offset)
    steepness = np.asarray(carrier_steepness)
    denominator = 2.0 * np.sqrt(2.0) * steepness * carrier
    return offset / denominator


def focused_steepness_proxy(
    carrier_mode: int | np.ndarray,
    sideband_offset: int | np.ndarray,
    carrier_steepness: float | np.ndarray,
) -> np.ndarray:
    """Return the leading-NLS focused-envelope steepness proxy.

    If ``beta`` is the value returned by :func:`instability_band_fraction`,
    the proxy is ``epsilon_c * (1 + 2 sqrt(1 - beta**2))``.  It is undefined
    outside the leading instability band and is returned as ``nan`` there.
    """

    steepness = np.asarray(carrier_steepness)
    band_fraction = instability_band_fraction(
        carrier_mode,
        sideband_offset,
        steepness,
    )
    radicand = 1.0 - band_fraction**2
    in_band = (band_fraction > 0.0) & (radicand >= 0.0)
    value = steepness * (
        1.0 + 2.0 * np.sqrt(np.maximum(radicand, 0.0))
    )
    return np.where(in_band, value, np.nan)


def focused_steepness_carrier_upper_bound(
    carrier_mode: int | np.ndarray,
    sideband_offset: int | np.ndarray,
    *,
    focused_steepness_limit: float,
) -> np.ndarray:
    """Return the carrier-steepness endpoint implied by the focus bound."""

    if (
        not math.isfinite(focused_steepness_limit)
        or focused_steepness_limit <= 0.0
    ):
        raise ValueError("focused_steepness_limit must be finite and positive")
    carrier = np.asarray(carrier_mode)
    offset = np.asarray(sideband_offset)
    instability_threshold = offset / (2.0 * np.sqrt(2.0) * carrier)
    return (
        -focused_steepness_limit
        + 2.0
        * np.sqrt(
            focused_steepness_limit**2 + 3.0 * instability_threshold**2
        )
    ) / 3.0


def is_supported(
    carrier_mode: int | np.ndarray,
    sideband_offset: int | np.ndarray,
    carrier_steepness: float | np.ndarray,
    perturbation_ratio: float | np.ndarray,
    *,
    carrier_mode_min: int = CARRIER_MODE_MIN,
    carrier_mode_max: int = CARRIER_MODE_MAX,
    carrier_steepness_min: float = CARRIER_STEEPNESS_MIN,
    carrier_steepness_max: float = CARRIER_STEEPNESS_MAX,
    perturbation_ratio_min: float = PERTURBATION_RATIO_MIN,
    perturbation_ratio_max: float = PERTURBATION_RATIO_MAX,
) -> np.ndarray:
    """Return whether parameters belong to the declared JCP09-style support."""

    carrier = np.asarray(carrier_mode)
    offset = np.asarray(sideband_offset)
    steepness = np.asarray(carrier_steepness)
    ratio = np.asarray(perturbation_ratio)
    band_fraction = instability_band_fraction(carrier, offset, steepness)
    return (
        (carrier >= carrier_mode_min)
        & (carrier <= carrier_mode_max)
        & (offset >= 1)
        & (offset < carrier)
        & (steepness >= carrier_steepness_min)
        & (steepness <= carrier_steepness_max)
        & (ratio >= perturbation_ratio_min)
        & (ratio <= perturbation_ratio_max)
        & (band_fraction > 0.0)
        & (band_fraction < 1.0)
    )


def feasible_mode_pairs(
    *,
    carrier_mode_min: int = CARRIER_MODE_MIN,
    carrier_mode_max: int = CARRIER_MODE_MAX,
    carrier_steepness_max: float = CARRIER_STEEPNESS_MAX,
) -> np.ndarray:
    """Enumerate ``(n_c, Delta n)`` pairs intersecting the instability band."""

    if carrier_mode_min < 2 or carrier_mode_max < carrier_mode_min:
        raise ValueError("carrier-mode bounds must satisfy 2 <= min <= max")
    if not math.isfinite(carrier_steepness_max) or carrier_steepness_max <= 0.0:
        raise ValueError("carrier_steepness_max must be finite and positive")
    pairs = tuple(
        (carrier, offset)
        for carrier in range(carrier_mode_min, carrier_mode_max + 1)
        for offset in range(1, carrier)
        if offset / carrier < 2.0 * math.sqrt(2.0) * carrier_steepness_max
    )
    if not pairs:
        raise ValueError("the requested bounds contain no unstable mode pair")
    return np.asarray(pairs, dtype=np.int32)


def sample_parameters(
    rng: np.random.Generator,
    *,
    batch_size: int,
    length: float,
    carrier_mode_min: int = CARRIER_MODE_MIN,
    carrier_mode_max: int = CARRIER_MODE_MAX,
    carrier_steepness_min: float = CARRIER_STEEPNESS_MIN,
    carrier_steepness_max: float = CARRIER_STEEPNESS_MAX,
    perturbation_ratio_min: float = PERTURBATION_RATIO_MIN,
    perturbation_ratio_max: float = PERTURBATION_RATIO_MAX,
) -> ParameterArrays:
    """Sample the declared support without outcome-dependent mode rewrites.

    Feasible integer mode pairs are balanced uniformly.  Conditional on a
    pair, steepness is uniform over the part of the declared interval inside
    the leading instability band.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if (
        not math.isfinite(carrier_steepness_min)
        or not math.isfinite(carrier_steepness_max)
        or not 0.0 < carrier_steepness_min <= carrier_steepness_max
    ):
        raise ValueError("invalid carrier-steepness bounds")
    if (
        not math.isfinite(perturbation_ratio_min)
        or not math.isfinite(perturbation_ratio_max)
        or not 0.0 < perturbation_ratio_min <= perturbation_ratio_max
    ):
        raise ValueError("invalid perturbation-ratio bounds")

    pairs = feasible_mode_pairs(
        carrier_mode_min=carrier_mode_min,
        carrier_mode_max=carrier_mode_max,
        carrier_steepness_max=carrier_steepness_max,
    )
    pair_indices = rng.integers(0, pairs.shape[0], size=batch_size)
    carrier_mode = pairs[pair_indices, 0]
    sideband_offset = pairs[pair_indices, 1]
    instability_lower_bound = sideband_offset / (
        2.0 * np.sqrt(2.0) * carrier_mode
    )
    steepness_lower_bound = np.maximum(
        carrier_steepness_min,
        np.nextafter(instability_lower_bound, np.inf),
    )
    carrier_steepness = rng.uniform(
        steepness_lower_bound,
        carrier_steepness_max,
    ).astype(np.float64)
    perturbation_ratio = rng.uniform(
        perturbation_ratio_min,
        perturbation_ratio_max,
        size=batch_size,
    ).astype(np.float64)
    translation = rng.uniform(
        0.0,
        length,
        size=batch_size,
    ).astype(np.float64)
    depth = np.full(
        batch_size,
        deep_water_proxy_depth(length),
        dtype=np.float64,
    )

    parameters: ParameterArrays = {
        "n_carr": carrier_mode.astype(np.int32),
        "side_offset": sideband_offset.astype(np.int32),
        "n_l": (carrier_mode - sideband_offset).astype(np.int32),
        "n_r": (carrier_mode + sideband_offset).astype(np.int32),
        "eps_carrier": carrier_steepness,
        "eps_pert": perturbation_ratio,
        "translation": translation,
        "depth": depth,
    }
    if not np.all(
        is_supported(
            parameters["n_carr"],
            parameters["side_offset"],
            parameters["eps_carrier"],
            parameters["eps_pert"],
            carrier_mode_min=carrier_mode_min,
            carrier_mode_max=carrier_mode_max,
            carrier_steepness_min=carrier_steepness_min,
            carrier_steepness_max=carrier_steepness_max,
            perturbation_ratio_min=perturbation_ratio_min,
            perturbation_ratio_max=perturbation_ratio_max,
        )
    ):
        raise RuntimeError("internal error: sampler produced an unsupported case")
    return parameters


def serialize_parameters(
    parameters: ParameterArrays,
) -> list[dict[str, float | int | str]]:
    """Convert a batch to complete JSON-compatible case specifications."""

    count = int(parameters["n_carr"].shape[0])
    return [
        {
            "n_carr": int(parameters["n_carr"][index]),
            "side_offset": int(parameters["side_offset"][index]),
            "n_l": int(parameters["n_l"][index]),
            "n_r": int(parameters["n_r"][index]),
            "first_harmonic_carrier_steepness": float(
                parameters["eps_carrier"][index]
            ),
            "first_harmonic_sideband_ratio": float(
                parameters["eps_pert"][index]
            ),
            "translation": float(parameters["translation"][index]),
            "relative_sideband_phase": JCP09_RELATIVE_SIDEBAND_PHASE,
            "carrier_phase_in_translated_frame": 0.0,
            "depth": float(parameters["depth"][index]),
            "instability_band_fraction": float(
                instability_band_fraction(
                    parameters["n_carr"][index],
                    parameters["side_offset"][index],
                    parameters["eps_carrier"][index],
                )
            ),
        }
        for index in range(count)
    ]


def _deep_stokes_carrier(
    x: jax.Array,
    *,
    carrier_mode: jax.Array,
    carrier_steepness: jax.Array,
    length: float,
    gravity: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return a fifth-order deep-water carrier with prescribed first harmonic."""

    dtype = x.dtype
    wavenumber = carrier_mode.astype(dtype) * (2.0 * jnp.pi / length)
    amplitude = carrier_steepness / wavenumber
    bare_amplitude = amplitude
    for _ in range(8):
        bare_steepness = wavenumber * bare_amplitude
        factor = (
            1.0
            + bare_steepness**2 / 8.0
            + 121.0 * bare_steepness**4 / 192.0
        )
        bare_amplitude = amplitude / factor
    eta, xi = stokes_eta_xi(
        x=x,
        time=jnp.asarray(0.0, dtype=dtype),
        n0=carrier_mode,
        a0=bare_amplitude,
        length=length,
        depth=deep_water_proxy_depth(length),
        gravity=gravity,
        ichoi=0,
    )
    return eta, xi, amplitude


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
    eta_carrier, xi_carrier, carrier_amplitude = _deep_stokes_carrier(
        translated_x,
        carrier_mode=carrier_mode,
        carrier_steepness=carrier_steepness,
        length=length,
        gravity=gravity,
    )
    dtype = x.dtype
    fundamental = 2.0 * jnp.pi / length
    left_wavenumber = (carrier_mode - sideband_offset).astype(dtype) * fundamental
    right_wavenumber = (
        carrier_mode + sideband_offset
    ).astype(dtype) * fundamental
    sideband_amplitude = perturbation_ratio * carrier_amplitude
    left_phase = (
        left_wavenumber * translated_x + JCP09_RELATIVE_SIDEBAND_PHASE
    )
    right_phase = (
        right_wavenumber * translated_x + JCP09_RELATIVE_SIDEBAND_PHASE
    )
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
    x: jax.Array,
    parameters: ParameterArrays,
    length: float,
    gravity: float,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:
    """Vectorize the deep-water JCP09-form construction over a case batch."""

    x_array = jnp.asarray(x, dtype=dtype)
    carrier_mode = jnp.asarray(parameters["n_carr"], dtype=jnp.int32)
    sideband_offset = jnp.asarray(parameters["side_offset"], dtype=jnp.int32)
    carrier_steepness = jnp.asarray(parameters["eps_carrier"], dtype=dtype)
    perturbation_ratio = jnp.asarray(parameters["eps_pert"], dtype=dtype)
    translation = jnp.asarray(parameters["translation"], dtype=dtype)

    def construct(
        mode: jax.Array,
        offset: jax.Array,
        steepness: jax.Array,
        ratio: jax.Array,
        global_translation: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return _initial_condition(
            x_array,
            carrier_mode=mode,
            sideband_offset=offset,
            carrier_steepness=steepness,
            perturbation_ratio=ratio,
            translation=global_translation,
            length=length,
            gravity=gravity,
        )

    return jax.vmap(construct)(
        carrier_mode,
        sideband_offset,
        carrier_steepness,
        perturbation_ratio,
        translation,
    )
