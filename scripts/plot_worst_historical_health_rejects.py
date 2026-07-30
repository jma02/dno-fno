"""Reconstruct and plot the largest finite historical health-gate failures.

The v9 shallow-wavetrain and steep-Tanaka generators retained rejected case
specifications and scalar Hamiltonian drift, but not the rejected field
arrays.  This script deterministically replays the relevant generator RNG
streams and reruns the five largest finite-drift cases.

The steep-Tanaka archives predate the tangent-Hermite reconstruction, so their
initial states are rebuilt with the archived piecewise-linear three-image
constructor.
"""
from __future__ import annotations

import argparse
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

jax.config.update("jax_enable_x64", True)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from solver.gen_data.generate_bf_dataset import (  # noqa: E402
    chunked_gxi_over_time,
    make_batch_rng,
    sample_log_uniform,
    select_time_indices,
)
from solver.gen_data.generate_shallow_steep_dataset import (  # noqa: E402
    build_wavetrain_initial_conditions_batched,
    sample_wavetrain_case_params,
)
from solver.gen_data.multi_crest import (  # noqa: E402
    CrestSpec,
    flatten_case_specs,
    sample_sum_budgeted_cases,
    specs_to_jax_arrays,
)
from solver.solvers.dno_series_jax import (  # noqa: E402
    build_grid,
    make_linear_dno_symbol,
    myfft,
    myifft,
)
from solver.solvers.time_integrator import (  # noqa: E402
    SolverParams,
    State,
    apply_lowpass,
    batched_rollout,
    cast_solver_params_dtype,
    cast_state_dtype,
    make_normalized_rollout_settings,
    spectral_dx,
)
from solver.tanaka_ICs.modified_tanaka import (  # noqa: E402
    ModifiedTanakaParams,
    make_default_tanaka_template,
    solve_modified_tanaka_batched,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Selection:
    family: str
    archive: str
    batch: int
    local_index: int
    case_id: int
    archived_drift: float


@dataclass(frozen=True)
class ReconstructedCase:
    selection: Selection
    meta: dict[str, Any]
    archived_spec: dict[str, Any]
    depth: float
    initial_eta: np.ndarray
    initial_xi: np.ndarray


@dataclass(frozen=True)
class CaseResult:
    case: ReconstructedCase
    x: np.ndarray
    retained_times: np.ndarray
    eta: np.ndarray
    xi: np.ndarray
    gxi: np.ndarray
    relative_drift: np.ndarray
    peak_index: int


SELECTIONS = (
    Selection(
        family="shallow wavetrain",
        archive="data/shallow_steep_v1_shard00.npz",
        batch=7,
        local_index=124,
        case_id=1916,
        archived_drift=0.3569140399774981,
    ),
    Selection(
        family="shallow wavetrain",
        archive="data/shallow_steep_v1_shard01.npz",
        batch=1,
        local_index=17,
        case_id=100273,
        archived_drift=0.22977960092857674,
    ),
    Selection(
        family="shallow wavetrain",
        archive="data/shallow_steep_v1_shard00.npz",
        batch=3,
        local_index=12,
        case_id=780,
        archived_drift=0.11872097191446354,
    ),
    Selection(
        family="steep Tanaka (pre-Hermite)",
        archive="data/steep_tanaka_v2_shard01.npz",
        batch=3,
        local_index=21,
        case_id=100405,
        archived_drift=0.056340687310005536,
    ),
    Selection(
        family="steep Tanaka (pre-Hermite)",
        archive="data/steep_tanaka_v2_shard00.npz",
        batch=5,
        local_index=39,
        case_id=679,
        archived_drift=0.05095373574779253,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/worst_historical_health_rejects_20260723",
    )
    return parser.parse_args()


def read_archive_record(
    selection: Selection,
) -> tuple[dict[str, Any], dict[str, Any]]:
    archive_path = REPO_ROOT / selection.archive
    spec_name = f"specs_batch_{selection.batch:04d}.json"
    with zipfile.ZipFile(archive_path) as archive:
        meta = json.loads(archive.read("meta.json"))
        specs = json.loads(archive.read(spec_name))
    return meta, specs[selection.local_index]


def slice_batch(params: dict[str, np.ndarray], index: int) -> dict[str, np.ndarray]:
    return {name: values[index : index + 1] for name, values in params.items()}


def reconstruct_shallow(selection: Selection) -> ReconstructedCase:
    meta, archived_spec = read_archive_record(selection)
    rng = make_batch_rng(
        int(meta["seed"]),
        selection.batch,
        int(meta["rng_stream_id"]),
    )
    params = sample_wavetrain_case_params(
        rng,
        batch_size=int(meta["batch_size"]),
        depth_min=float(meta["depth_min"]),
        depth_max=float(meta["depth_max"]),
        kh_min=float(meta["kh_min"]),
        kh_max=float(meta["kh_max"]),
        a_over_h_min=float(meta["a_over_h_min"]),
        a_over_h_max=float(meta["a_over_h_max"]),
        ka_max=float(meta["ka_max"]),
        n_modes_max=int(meta["n_modes_max"]),
        n_mode_cap=int(meta["n_mode_cap"]),
    )
    selected = slice_batch(params, selection.local_index)
    archived_modes = np.asarray(archived_spec["modes"], dtype=np.int32)
    rebuilt_modes = selected["modes"][0, : int(selected["n_modes"][0])]
    if not np.array_equal(archived_modes, rebuilt_modes):
        raise RuntimeError(f"RNG replay failed for shallow case {selection.case_id}")

    x, _ = build_grid(int(meta["nx"]), float(meta["length"]))
    eta, xi = build_wavetrain_initial_conditions_batched(
        jnp.asarray(x, dtype=jnp.float64),
        selected,
        float(meta["gravity"]),
        jnp.float64,
    )
    return ReconstructedCase(
        selection=selection,
        meta=meta,
        archived_spec=archived_spec,
        depth=float(selected["depth"][0]),
        initial_eta=np.asarray(eta[0]),
        initial_xi=np.asarray(xi[0]),
    )


def build_piecewise_linear_tanaka_initial_conditions(
    *,
    template_params: ModifiedTanakaParams,
    case_h_ref: np.ndarray,
    case_specs: list[list[CrestSpec]],
    length: float,
    nx: int,
    gravity: float,
) -> tuple[jax.Array, jax.Array]:
    """Archived fine-grid linear interpolation with three periodic images."""
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
    crest_depths = jnp.asarray(
        case_h_ref, dtype=x_profile.dtype
    )[crest_case_ids_array]
    crest_centers = jnp.asarray(flat_centers, dtype=x_profile.dtype)

    fine_factor = 8
    fine_nx = nx * fine_factor
    x_fine = (length / fine_nx) * jnp.arange(
        fine_nx, dtype=x_profile.dtype
    )

    def place_one(
        x_row: jax.Array,
        eta_row: jax.Array,
        depth: jax.Array,
        center: jax.Array,
    ) -> jax.Array:
        shifted = x_row * depth + center
        scaled = eta_row * depth
        central = jnp.interp(
            x_fine, shifted, scaled, left=0.0, right=0.0
        )
        left = jnp.interp(
            x_fine - length, shifted, scaled, left=0.0, right=0.0
        )
        right = jnp.interp(
            x_fine + length, shifted, scaled, left=0.0, right=0.0
        )
        fine_values = central + left + right
        native_spectrum = jnp.fft.rfft(fine_values)[: nx // 2 + 1]
        return jnp.fft.irfft(
            native_spectrum / fine_factor, n=nx
        ).astype(eta_row.dtype)

    eta_per_crest = jax.vmap(place_one)(
        x_profile,
        eta_profile,
        crest_depths,
        crest_centers,
    )
    speeds = tanaka_batch.froude * jnp.sqrt(gravity * crest_depths)
    directions = jnp.asarray(flat_directions, dtype=eta_per_crest.dtype)
    signed_speeds = (directions * speeds)[..., None]

    _, k = build_grid(nx, length)
    eta_x = spectral_dx(eta_per_crest, k)
    radical = (
        (1.0 + eta_x**2)
        * (signed_speeds**2 - 2.0 * gravity * eta_per_crest)
    )
    xi_x = signed_speeds - directions[..., None] * jnp.sqrt(
        jnp.maximum(radical, 0.0)
    )
    inverse_ik = jnp.where(k != 0.0, 1.0 / (1j * k), 0.0)
    xi_per_crest = myifft(inverse_ik * myfft(xi_x, nx))
    xi_per_crest -= jnp.mean(xi_per_crest, axis=-1, keepdims=True)

    batch_size = case_h_ref.shape[0]
    eta = jnp.zeros((batch_size, nx), dtype=eta_per_crest.dtype)
    xi = jnp.zeros((batch_size, nx), dtype=xi_per_crest.dtype)
    eta = eta.at[crest_case_ids_array].add(eta_per_crest)
    xi = xi.at[crest_case_ids_array].add(xi_per_crest)
    return eta, xi


def replay_steep_batch(
    selection: Selection,
    meta: dict[str, Any],
) -> tuple[np.ndarray, list[list[CrestSpec]]]:
    rng = make_batch_rng(
        int(meta["seed"]),
        selection.batch,
        int(meta["rng_stream_id"]),
    )
    depths = sample_log_uniform(
        rng,
        float(meta["depth_min"]),
        float(meta["depth_max"]),
        (int(meta["batch_size"]),),
    )
    case_specs: list[list[CrestSpec]] = []
    for depth in depths:
        minimum_separation = float(meta["separation_widths"]) * float(depth)
        maximum_count = int(
            max(1, math.floor(float(meta["length"]) / minimum_separation))
        )
        max_crests = min(int(meta["max_crests"]), maximum_count)
        min_crests = min(int(meta["min_crests"]), max_crests)
        case_specs.extend(
            sample_sum_budgeted_cases(
                rng,
                1,
                length=float(meta["length"]),
                case_amplitude_min=float(meta["steepness_min"]),
                case_amplitude_max=float(meta["steepness_max"]),
                min_crests=min_crests,
                max_crests=max_crests,
                min_separation=minimum_separation,
                per_crest_floor=float(meta["per_crest_steepness_floor"]),
            )
        )
    return depths, case_specs


def reconstruct_steep(selection: Selection) -> ReconstructedCase:
    meta, archived_spec = read_archive_record(selection)
    depths, case_specs = replay_steep_batch(selection, meta)
    selected_specs = [case_specs[selection.local_index]]
    archived_amplitude = float(archived_spec["crests"][0]["amplitude"])
    rebuilt_amplitude = selected_specs[0][0].amplitude
    if not np.isclose(archived_amplitude, rebuilt_amplitude, rtol=0.0, atol=1e-14):
        raise RuntimeError(f"RNG replay failed for steep case {selection.case_id}")

    template = make_default_tanaka_template(
        depth=1.0,
        gravity=float(meta["gravity"]),
        direction=1,
        nx=int(meta["nx"]),
        length=float(meta["length"]),
        center=0.0,
        dno_order=6,
        pad_factor=8,
    )
    selected_depth = np.asarray(
        [depths[selection.local_index]], dtype=np.float64
    )
    eta, xi = build_piecewise_linear_tanaka_initial_conditions(
        template_params=template,
        case_h_ref=selected_depth,
        case_specs=selected_specs,
        length=float(meta["length"]),
        nx=int(meta["nx"]),
        gravity=float(meta["gravity"]),
    )
    return ReconstructedCase(
        selection=selection,
        meta=meta,
        archived_spec=archived_spec,
        depth=float(selected_depth[0]),
        initial_eta=np.asarray(eta[0]),
        initial_xi=np.asarray(xi[0]),
    )


def reconstruct(selection: Selection) -> ReconstructedCase:
    if selection.family == "shallow wavetrain":
        return reconstruct_shallow(selection)
    return reconstruct_steep(selection)


def run_group(cases: list[ReconstructedCase]) -> list[CaseResult]:
    if not cases:
        return []
    meta = cases[0].meta
    settings = make_normalized_rollout_settings()._replace(
        filter_fraction=float(meta["filter_fraction"])
    )
    nx = int(meta["nx"])
    length = float(meta["length"])
    gravity = float(meta["gravity"])
    times = np.arange(
        0.0,
        float(meta["tmax"]) + 0.5 * float(meta["dt"]),
        float(meta["dt"]),
        dtype=np.float64,
    )
    retained_indices = select_time_indices(
        times.shape[0], int(meta["keep_samples"])
    )
    retained_times = times[retained_indices]
    x, k = build_grid(nx, length)
    k = jnp.asarray(k, dtype=jnp.float64)

    eta0 = apply_lowpass(
        jnp.asarray(
            np.stack([case.initial_eta for case in cases]), dtype=jnp.float64
        ),
        k,
        settings.filter_fraction,
    )
    xi0 = apply_lowpass(
        jnp.asarray(
            np.stack([case.initial_xi for case in cases]), dtype=jnp.float64
        ),
        k,
        settings.filter_fraction,
    )
    xi0 -= jnp.mean(xi0, axis=-1, keepdims=True)
    depths = jnp.asarray(
        [case.depth for case in cases], dtype=jnp.float64
    )[:, None]
    params = SolverParams(
        nx=nx,
        length=length,
        depth=depths,
        gravity=gravity,
        dno_order=settings.dno_order,
        pad_factor=settings.pad_factor,
        filter_fraction=settings.filter_fraction,
        k=k,
        g0=make_linear_dno_symbol(k, depths),
    )
    params = cast_solver_params_dtype(params, jnp.float64)
    payload = batched_rollout(
        cast_state_dtype(State(eta=eta0, xi=xi0), jnp.float64),
        jnp.asarray(times, dtype=jnp.float64),
        params,
        save_gxi=False,
        substeps_per_interval=int(meta["substeps"]),
        method=str(meta["method"]),
        implicit_iterations=int(meta["implicit_iterations"]),
        implicit_relaxation=settings.implicit_relaxation,
        zero_mean_xi=bool(meta["zero_mean_xi"]),
    )
    jax.block_until_ready(payload["xi"])
    eta = np.asarray(jax.device_get(payload["eta"][retained_indices]))
    xi = np.asarray(jax.device_get(payload["xi"][retained_indices]))
    gxi = np.asarray(
        chunked_gxi_over_time(
            jnp.asarray(eta),
            jnp.asarray(xi),
            k,
            depths,
            settings.dno_order,
            settings.pad_factor,
            settings.filter_fraction,
            min(16, len(retained_indices)),
        )
    )

    dx = length / nx
    energy = (
        0.5
        * np.sum(xi * gxi + gravity * eta**2, axis=-1)
        * dx
    )
    drift = np.abs(energy - energy[:1]) / (
        np.abs(energy[:1]) + 1e-30
    )
    results = []
    for index, case in enumerate(cases):
        case_drift = drift[:, index]
        peak_index = int(np.nanargmax(case_drift))
        results.append(
            CaseResult(
                case=case,
                x=np.asarray(x),
                retained_times=retained_times,
                eta=eta[:, index],
                xi=xi[:, index],
                gxi=gxi[:, index],
                relative_drift=case_drift,
                peak_index=peak_index,
            )
        )
    return results


def plot_results(results: list[CaseResult], output_path: Path) -> None:
    figure, axes = plt.subplots(
        len(results),
        4,
        figsize=(16.0, 3.0 * len(results)),
        constrained_layout=True,
    )
    for row, result in enumerate(results):
        depth = result.case.depth
        gravity = float(result.case.meta["gravity"])
        velocity_scale = math.sqrt(gravity * depth)
        peak = result.peak_index
        initial_style = {"color": "0.6", "lw": 1.0, "ls": "--", "label": "t=0"}
        peak_style = {"color": "tab:red", "lw": 1.2, "label": f"t={result.retained_times[peak]:.2f}"}

        axes[row, 0].plot(
            result.x, result.eta[0] / depth, **initial_style
        )
        axes[row, 0].plot(
            result.x, result.eta[peak] / depth, **peak_style
        )
        axes[row, 1].plot(
            result.x,
            result.xi[0] / (depth * velocity_scale),
            **initial_style,
        )
        axes[row, 1].plot(
            result.x,
            result.xi[peak] / (depth * velocity_scale),
            **peak_style,
        )
        axes[row, 2].plot(
            result.x, result.gxi[0] / velocity_scale, **initial_style
        )
        axes[row, 2].plot(
            result.x, result.gxi[peak] / velocity_scale, **peak_style
        )
        axes[row, 3].semilogy(
            result.retained_times,
            np.maximum(result.relative_drift, 1e-16),
            color="tab:blue",
            lw=1.2,
        )
        tolerance = float(result.case.meta["drift_tol"])
        axes[row, 3].axhline(
            tolerance,
            color="tab:red",
            lw=1.0,
            ls="--",
            label=f"gate={tolerance:.0e}",
        )
        axes[row, 3].scatter(
            [result.retained_times[peak]],
            [result.relative_drift[peak]],
            color="tab:red",
            s=18,
            zorder=3,
        )

        specification = result.case.archived_spec
        if result.case.selection.family == "shallow wavetrain":
            shape_text = (
                f"a/h={float(specification['a_over_h_nominal']):.3f}, "
                f"modes={specification['modes']}"
            )
        else:
            crest = specification["crests"][0]
            shape_text = f"a/h={float(crest['amplitude']):.3f}"
        axes[row, 0].set_ylabel(
            f"{result.case.selection.family}\n"
            f"case {result.case.selection.case_id}, h={depth:.4f}\n"
            f"{shape_text}\n"
            f"max drift={result.relative_drift[peak]:.3g}"
        )
        for column in range(4):
            axes[row, column].grid(alpha=0.2)
        axes[row, 0].legend(loc="best", fontsize=8)
        axes[row, 3].legend(loc="best", fontsize=8)

    axes[0, 0].set_title(r"surface elevation $\eta/h$")
    axes[0, 1].set_title(r"surface potential $\xi/(h\sqrt{gh})$")
    axes[0, 2].set_title(r"$G(\eta;h)\xi/\sqrt{gh}$")
    axes[0, 3].set_title(r"relative Hamiltonian drift")
    for axis in axes[-1, :]:
        axis.set_xlabel("x" if axis is not axes[-1, 3] else "time")
    figure.suptitle(
        "Largest finite Hamiltonian-drift failures in historical v9 health-gated sources\n"
        "Gray: initial state. Red: retained frame with maximum drift. "
        "Steep-Tanaka rows use the archived pre-Hermite constructor.",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_summary(results: list[CaseResult], output_path: Path, wall_seconds: float) -> None:
    payload = {
        "definition": {
            "population": "historical v9 whole-case health-gate rejects",
            "ranking": "largest finite archived Hamiltonian drift",
            "important_limitations": [
                "The proposed final shared whole-trajectory acceptance protocol has not yet been run.",
                "The historical gate inspected retained frames only and did not test GL stage residuals.",
                "The historical steep-Tanaka cases predate tangent-Hermite reconstruction.",
            ],
        },
        "wall_seconds": wall_seconds,
        "cases": [
            {
                "family": result.case.selection.family,
                "archive": result.case.selection.archive,
                "case_id": result.case.selection.case_id,
                "batch": result.case.selection.batch,
                "local_index": result.case.selection.local_index,
                "depth": result.case.depth,
                "archived_maximum_drift": result.case.selection.archived_drift,
                "recomputed_maximum_drift": float(
                    result.relative_drift[result.peak_index]
                ),
                "peak_retained_time": float(
                    result.retained_times[result.peak_index]
                ),
                "drift_tolerance": float(result.case.meta["drift_tol"]),
                "specification": result.case.archived_spec,
            }
            for result in results
        ],
    }
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    cases = [reconstruct(selection) for selection in SELECTIONS]
    shallow = [case for case in cases if case.selection.family == "shallow wavetrain"]
    steep = [case for case in cases if case.selection.family != "shallow wavetrain"]
    results = run_group(shallow) + run_group(steep)
    results_by_id = {result.case.selection.case_id: result for result in results}
    ordered = [results_by_id[selection.case_id] for selection in SELECTIONS]
    wall_seconds = perf_counter() - started
    plot_results(ordered, output_dir / "worst_finite_health_rejects.png")
    write_summary(
        ordered,
        output_dir / "worst_finite_health_rejects.json",
        wall_seconds,
    )
    np.savez(
        output_dir / "worst_finite_health_rejects.npz",
        case_id=np.asarray([result.case.selection.case_id for result in ordered]),
        depth=np.asarray([result.case.depth for result in ordered]),
        archived_drift=np.asarray(
            [result.case.selection.archived_drift for result in ordered]
        ),
        recomputed_drift=np.asarray(
            [result.relative_drift[result.peak_index] for result in ordered]
        ),
        peak_time=np.asarray(
            [result.retained_times[result.peak_index] for result in ordered]
        ),
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "wall_seconds": wall_seconds,
                "cases": [
                    {
                        "case_id": result.case.selection.case_id,
                        "archived_drift": result.case.selection.archived_drift,
                        "recomputed_drift": float(
                            result.relative_drift[result.peak_index]
                        ),
                    }
                    for result in ordered
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
