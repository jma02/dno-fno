"""Generate and label one batch of static Stokes simulations."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from solver.reference_solutions.stokes_wave import stokes_eta_xi_at_phase
from solver.gen_data.pipeline.batch_storage import save_completed_batch
from solver.gen_data.pipeline.dno_target import compute_dno_target
from solver.gen_data.pipeline.types import (
    PhysicalFamilyId,
    SimulationRows,
)
from solver.gen_data.stokes_sampling import (
    PAPER_DOMAIN_LENGTH,
    PAPER_GRAVITY,
    sample_stokes_simulation,
)
from solver.solvers.dno_series_jax import build_grid


PAPER_STATIC_STOKES_NX = 1024
PAPER_STATIC_STOKES_DNO_ORDER = 6
PAPER_STATIC_STOKES_PAD_FACTOR = 8
PAPER_STATIC_STOKES_MAXIMUM_WAVENUMBER = 128.0

_construct_stokes_batch = jax.jit(
    jax.vmap(
        stokes_eta_xi_at_phase,
        in_axes=(None, 0, 0, 0, None, 0, None, None),
    ),
    static_argnums=(7,),
)


def generate_static_stokes_batch(
    parameter_group_ids: tuple[str, ...],
    first_attempt_number: int,
    output_path: Path,
    *,
    seed: int,
) -> int:
    """Save one static Stokes batch and return its accepted simulation count."""

    samples = tuple(
        sample_stokes_simulation(
            parameter_group_id,
            seed=seed,
            attempt_number=first_attempt_number + offset,
        )
        for offset, parameter_group_id in enumerate(parameter_group_ids)
    )
    accepted_indices = tuple(
        index for index, sample in enumerate(samples) if sample is not None
    )
    rows_by_index: dict[int, SimulationRows] = {}
    if accepted_indices:
        accepted = tuple(sample for sample in samples if sample is not None)
        depths = np.fromiter(
            (sample.depth for sample in accepted), dtype=np.float64
        )
        with jax.enable_x64():
            x, _ = build_grid(PAPER_STATIC_STOKES_NX, PAPER_DOMAIN_LENGTH)
            eta, xi = _construct_stokes_batch(
                jnp.asarray(x, dtype=jnp.float64),
                jnp.asarray([sample.phase for sample in accepted], dtype=jnp.float64),
                jnp.asarray(
                    [sample.carrier_mode for sample in accepted], dtype=jnp.int32
                ),
                jnp.asarray(
                    [sample.amplitude for sample in accepted], dtype=jnp.float64
                ),
                PAPER_DOMAIN_LENGTH,
                jnp.asarray(depths, dtype=jnp.float64),
                PAPER_GRAVITY,
                1 if accepted[0].branch == "finite" else 0,
            )
            eta, xi, gxi = compute_dno_target(
                eta,
                xi,
                jnp.asarray(depths, dtype=jnp.float64)[:, None],
                nx=PAPER_STATIC_STOKES_NX,
                length=PAPER_DOMAIN_LENGTH,
                dno_order=PAPER_STATIC_STOKES_DNO_ORDER,
                pad_factor=PAPER_STATIC_STOKES_PAD_FACTOR,
                maximum_wavenumber=PAPER_STATIC_STOKES_MAXIMUM_WAVENUMBER,
            )
        eta, xi, gxi = map(np.asarray, jax.device_get((eta, xi, gxi)))
        valid = (
            np.isfinite(eta).all(axis=1)
            & np.isfinite(xi).all(axis=1)
            & np.isfinite(gxi).all(axis=1)
            & (np.min(depths[:, None] + eta, axis=1) > 0.0)
        )
        rows_by_index = {
            source_index: SimulationRows(
                eta=eta[row : row + 1],
                xi=xi[row : row + 1],
                gxi=gxi[row : row + 1],
                depth=float(depths[row]),
                time=np.zeros(1, dtype=np.float64),
            )
            for row, source_index in enumerate(accepted_indices)
            if valid[row]
        }

    return save_completed_batch(
        output_path,
        parameter_group_ids,
        tuple(rows_by_index.get(index) for index in range(len(samples))),
        family_id=PhysicalFamilyId.STOKES,
        seed=seed,
    )
