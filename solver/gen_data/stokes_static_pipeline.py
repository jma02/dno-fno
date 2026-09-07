"""Construct and label the paper dataset's static Stokes states."""

from __future__ import annotations

import math

import jax
import numpy as np

from solver.reference_solutions.stokes_wave import stokes_eta_xi_at_phase
from solver.gen_data.pipeline.dno_target import bandlimit_field, compute_dno_target
from solver.gen_data.pipeline.types import SimulationRows
from solver.gen_data.stokes_sampling import (
    PAPER_DOMAIN_LENGTH,
    PAPER_GRAVITY,
    StokesSample,
)
from solver.solvers.dno_series_jax import build_grid


PAPER_STATIC_STOKES_NX = 1024
PAPER_STATIC_STOKES_DNO_ORDER = 6
PAPER_STATIC_STOKES_PAD_FACTOR = 8
PAPER_STATIC_STOKES_MAXIMUM_WAVENUMBER = 128.0


def evaluate_static_stokes_sample(sample: StokesSample) -> SimulationRows | None:
    """Construct, validate, and label one static Stokes simulation."""

    with jax.enable_x64():
        x, wavenumbers = build_grid(PAPER_STATIC_STOKES_NX, PAPER_DOMAIN_LENGTH)
        eta, xi = stokes_eta_xi_at_phase(
            x=x,
            phase=sample.phase,
            n0=sample.carrier_mode,
            a0=sample.amplitude,
            length=PAPER_DOMAIN_LENGTH,
            depth=sample.depth,
            gravity=PAPER_GRAVITY,
            ichoi=1 if sample.branch == "finite" else 0,
        )
        eta = bandlimit_field(
            eta,
            wavenumbers,
            maximum_wavenumber=PAPER_STATIC_STOKES_MAXIMUM_WAVENUMBER,
        )
        xi = bandlimit_field(
            xi,
            wavenumbers,
            maximum_wavenumber=PAPER_STATIC_STOKES_MAXIMUM_WAVENUMBER,
            remove_mean=True,
        )
        eta_host, xi_host = map(np.asarray, jax.device_get((eta, xi)))

        if not all(np.isfinite(field).all() for field in (eta_host, xi_host)):
            return None

        minimum_water_column = float(np.min(sample.depth + eta_host))
        if not math.isfinite(minimum_water_column) or minimum_water_column <= 0.0:
            return None

        eta, xi, gxi = compute_dno_target(
            eta,
            xi,
            sample.depth,
            nx=PAPER_STATIC_STOKES_NX,
            length=PAPER_DOMAIN_LENGTH,
            dno_order=PAPER_STATIC_STOKES_DNO_ORDER,
            pad_factor=PAPER_STATIC_STOKES_PAD_FACTOR,
            maximum_wavenumber=PAPER_STATIC_STOKES_MAXIMUM_WAVENUMBER,
        )
    eta_host, xi_host, gxi_host = map(np.asarray, jax.device_get((eta, xi, gxi)))
    if not all(np.isfinite(field).all() for field in (eta_host, xi_host, gxi_host)):
        return None
    return SimulationRows(
        eta=eta_host[None, :],
        xi=xi_host[None, :],
        gxi=gxi_host[None, :],
        depth=sample.depth,
        time=np.asarray([0.0], dtype=np.float64),
    )
