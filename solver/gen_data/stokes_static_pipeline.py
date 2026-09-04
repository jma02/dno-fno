"""Construct and label the paper dataset's static Stokes states."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from solver.reference_solutions.stokes_wave import stokes_eta_xi_at_phase
from solver.gen_data.pipeline.simulation_checks import SimulationCheckResult
from solver.gen_data.pipeline.dno_target import compute_dno_target, project_fixed_band
from solver.gen_data.pipeline.writer import (
    AcceptedSimulationRows,
    SimulationOutcome,
)
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


def evaluate_static_stokes_sample(sample: StokesSample) -> SimulationOutcome:
    """Construct, validate, and label one static Stokes simulation."""

    with jax.enable_x64():
        x, wavenumbers = build_grid(PAPER_STATIC_STOKES_NX, PAPER_DOMAIN_LENGTH)
        eta_raw, xi_raw = stokes_eta_xi_at_phase(
            x=jnp.asarray(x, dtype=jnp.float64),
            phase=sample.phase,
            n0=sample.carrier_mode,
            a0=sample.amplitude,
            length=PAPER_DOMAIN_LENGTH,
            depth=sample.depth,
            gravity=PAPER_GRAVITY,
            ichoi=1 if sample.branch == "finite" else 0,
        )
        eta_array = jnp.asarray(eta_raw, dtype=jnp.float64)
        xi_array = jnp.asarray(xi_raw, dtype=jnp.float64)
        wavenumber_array = jnp.asarray(wavenumbers, dtype=jnp.float64)
        eta_input = project_fixed_band(
            eta_array,
            wavenumber_array,
            maximum_wavenumber=PAPER_STATIC_STOKES_MAXIMUM_WAVENUMBER,
        )
        xi_input = project_fixed_band(
            xi_array - jnp.mean(xi_array),
            wavenumber_array,
            maximum_wavenumber=PAPER_STATIC_STOKES_MAXIMUM_WAVENUMBER,
            remove_mean=True,
        )
    eta_host = np.asarray(jax.device_get(eta_input), dtype=np.float64)
    xi_host = np.asarray(jax.device_get(xi_input), dtype=np.float64)

    if not np.isfinite(eta_host).all() or not np.isfinite(xi_host).all():
        return SimulationOutcome(
            decision=SimulationCheckResult(accepted=False, nonfinite_state=True),
            rows=None,
            metrics={},
        )

    minimum_water_column = float(np.min(sample.depth + eta_host))
    if not math.isfinite(minimum_water_column) or minimum_water_column <= 0.0:
        return SimulationOutcome(
            decision=SimulationCheckResult(
                accepted=False, nonpositive_water_height=True
            ),
            rows=None,
            metrics={},
        )

    with jax.enable_x64():
        target_eta, target_xi, q_ref = compute_dno_target(
            eta_input,
            xi_input,
            sample.depth,
            nx=PAPER_STATIC_STOKES_NX,
            length=PAPER_DOMAIN_LENGTH,
            dno_order=PAPER_STATIC_STOKES_DNO_ORDER,
            pad_factor=PAPER_STATIC_STOKES_PAD_FACTOR,
            maximum_wavenumber=PAPER_STATIC_STOKES_MAXIMUM_WAVENUMBER,
        )
    target_eta_host = np.asarray(jax.device_get(target_eta), dtype=np.float64)
    target_xi_host = np.asarray(jax.device_get(target_xi), dtype=np.float64)
    q_ref_host = np.asarray(jax.device_get(q_ref), dtype=np.float64)
    delivered_state_finite = bool(
        np.isfinite(target_eta_host).all() and np.isfinite(target_xi_host).all()
    )
    target_finite = bool(np.isfinite(q_ref_host).all())
    decision = SimulationCheckResult(
        accepted=delivered_state_finite and target_finite,
        nonfinite_state=not delivered_state_finite,
        nonfinite_target=not target_finite,
    )
    rows = (
        AcceptedSimulationRows(
            eta=target_eta_host[None, :],
            xi=target_xi_host[None, :],
            gxi=q_ref_host[None, :],
            depth=sample.depth,
            time=np.asarray([0.0], dtype=np.float64),
        )
        if decision.accepted
        else None
    )
    return SimulationOutcome(decision=decision, rows=rows, metrics={})
