"""Test the legacy Tanaka sign rule against fixed-band grid refinement.

The historical pre-cleaning archive is unavailable, but its case specifications
and the retained-row archive remain.  This script reconstructs a deterministic
frame-zero panel at N and 2N using the corresponding historical or corrected IC
builder, holds the physical Fourier band fixed, and compares the DNO output.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeAlias

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from solver.gen_data.generate_tanaka_dataset_v2 import (
    build_per_case_initial_conditions,
)
from solver.gen_data.multi_crest import (
    CrestSpec,
    flatten_case_specs,
    sample_random_cases,
    specs_to_jax_arrays,
)
from solver.solvers.dno_series_jax import dno_series_eval
from solver.solvers.time_integrator import (
    _spectral_truncate_real,
    make_solver_params,
)
from solver.tanaka_ICs.modified_tanaka import (
    make_default_tanaka_template,
    solve_modified_tanaka_batched,
)


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_ARCHIVE = ROOT / "data/old_tanaka_1_clean.npz"
CURRENT_ARCHIVE = ROOT / "data/tanaka_2_adaptive_g0.npz"
CURRENT_SIGN_REJECTED_ICS = (27, 51, 226)
FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class CaseMetric:
    population: str
    case_id: int
    legacy_retained_at_t0: bool | None
    sign_changes: int
    fixed_band_q_l2_defect: float
    fixed_band_q_h1_defect: float
    fixed_band_eta_l2_defect: float
    fixed_band_xi_l2_defect: float
    raw_q_out_of_band_l2: float
    min_center_boundary_distance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--random_historical_controls", type=int, default=32)
    parser.add_argument("--boundary_historical_controls", type=int, default=8)
    parser.add_argument("--provisional_tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=ROOT / "notes/signflip_refinement_transition_20260722.json",
    )
    parser.add_argument(
        "--output_png",
        type=Path,
        default=ROOT / "notes/signflip_refinement_transition_20260722.png",
    )
    return parser.parse_args()


def read_array(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    with archive.open(name) as handle:
        return np.load(handle, allow_pickle=False)


def deserialize_case(raw: list[dict[str, float | int]]) -> list[CrestSpec]:
    return [
        CrestSpec(
            amplitude=float(item["amplitude"]),
            center=float(item["center"]),
            direction=int(item["direction"]),
        )
        for item in raw
    ]


def historical_t0_retained(archive: zipfile.ZipFile, batch_id: int) -> set[int]:
    case_ids = read_array(archive, f"case_id_batch_{batch_id:04d}.npy")
    times = read_array(archive, f"time_batch_{batch_id:04d}.npy")
    return set(int(value) for value in case_ids[np.isclose(times, 0.0)])


def regenerate_historical_specs(
    metadata: dict[str, object], batch_id: int
) -> list[list[CrestSpec]]:
    seed = int(metadata["seed"])
    stream_id = int(metadata.get("rng_stream_id", metadata.get("case_id_offset", 0)))
    rng = np.random.default_rng(np.random.SeedSequence([seed, stream_id, batch_id]))
    return sample_random_cases(
        rng,
        int(metadata["batch_size"]),
        length=float(metadata["length"]),
        amplitude_min=float(metadata["amplitude_min"]),
        amplitude_max=float(metadata["amplitude_max"]),
        min_crests=int(metadata["min_crests"]),
        max_crests=int(metadata["max_crests"]),
        min_separation=float(metadata["min_separation"]),
    )


def historical_pre_sign_case_ids(
    archive: zipfile.ZipFile,
    batch_id: int,
    regenerated_specs: list[list[CrestSpec]],
) -> set[int]:
    raw_specs = json.loads(archive.read(f"specs_batch_{batch_id:04d}.json"))
    saved_specs = [deserialize_case(raw) for raw in raw_specs]
    case_id_by_specs = {
        tuple(case_specs): case_id
        for case_id, case_specs in enumerate(regenerated_specs)
    }
    mapped_ids = [case_id_by_specs.get(tuple(case_specs)) for case_specs in saved_specs]
    if any(case_id is None for case_id in mapped_ids):
        raise ValueError(
            "saved historical specifications do not match the seeded cases"
        )
    present_ids = {int(case_id) for case_id in mapped_ids if case_id is not None}
    if len(present_ids) != len(saved_specs):
        raise ValueError("historical specification mapping is not one-to-one")
    return present_ids


def select_historical_cases(
    archive: zipfile.ZipFile,
    seed: int,
    random_controls: int,
    boundary_controls: int,
) -> tuple[
    list[int],
    dict[int, list[CrestSpec]],
    dict[int, bool],
    dict[str, object],
]:
    metadata = json.loads(archive.read("meta.json"))
    batch_zero_specs = regenerate_historical_specs(metadata, 0)
    pre_sign_ids = historical_pre_sign_case_ids(archive, 0, batch_zero_specs)
    batch_zero_retained = historical_t0_retained(archive, 0)
    retained_ids = np.asarray(
        sorted(pre_sign_ids & batch_zero_retained),
        dtype=np.int64,
    )
    rejected_ids = np.asarray(
        sorted(pre_sign_ids - batch_zero_retained), dtype=np.int64
    )

    rng = np.random.default_rng(seed)
    random_ids = rng.choice(
        retained_ids,
        size=min(random_controls, retained_ids.size),
        replace=False,
    )
    boundary_distance = np.asarray(
        [
            min(min(spec.center, 164.0 - spec.center) for spec in specs)
            for specs in batch_zero_specs
        ]
    )
    retained_by_boundary = retained_ids[
        np.argsort(boundary_distance[retained_ids])[:boundary_controls]
    ]
    selected_ids = sorted(
        set(rejected_ids.tolist())
        | set(random_ids.tolist())
        | set(retained_by_boundary.tolist())
    )

    specs_by_id: dict[int, list[CrestSpec]] = {}
    retained_by_id: dict[int, bool] = {}
    for case_id in selected_ids:
        specs_by_id[case_id] = batch_zero_specs[case_id]
        retained_by_id[case_id] = case_id in batch_zero_retained
    provenance = {
        "historical_batch_id": 0,
        "seeded_cases": len(batch_zero_specs),
        "pre_sign_present_cases": len(pre_sign_ids),
        "pre_sign_missing_case_ids": sorted(
            set(range(len(batch_zero_specs))) - pre_sign_ids
        ),
        "sign_rejected_at_t0_case_ids": rejected_ids.tolist(),
        "sign_rejected_at_t0_cases": int(rejected_ids.size),
        "sign_retained_at_t0_cases": int(retained_ids.size),
    }
    return selected_ids, specs_by_id, retained_by_id, provenance


def projector(k: jax.Array, cutoff: float) -> Callable[[jax.Array], jax.Array]:
    mask = (jnp.abs(k) <= cutoff).astype(jnp.complex128)

    def project(field: jax.Array) -> jax.Array:
        return jnp.fft.ifft(jnp.fft.fft(field, axis=-1) * mask, axis=-1).real

    return project


def count_sign_changes(row: FloatArray, relative_threshold: float = 0.03) -> int:
    difference = np.roll(row, -1) - row
    threshold = relative_threshold * float(np.max(np.abs(difference)))
    signs = np.where(
        difference > threshold,
        1,
        np.where(difference < -threshold, -1, 0),
    )
    nonzero = signs[signs != 0]
    if nonzero.size <= 1:
        return 0
    return int(np.sum(nonzero != np.roll(nonzero, -1)))


def relative_l2(value: FloatArray, reference: FloatArray) -> float:
    return float(
        np.linalg.norm(value - reference)
        / max(float(np.linalg.norm(reference)), np.finfo(np.float64).tiny)
    )


def relative_h1(
    value: FloatArray,
    reference: FloatArray,
    physical_k: FloatArray,
) -> float:
    difference_hat = np.fft.fft(value - reference)
    reference_hat = np.fft.fft(reference)
    weights = 1.0 + physical_k**2
    return float(
        np.sqrt(
            np.sum(weights * np.abs(difference_hat) ** 2)
            / max(
                float(np.sum(weights * np.abs(reference_hat) ** 2)),
                np.finfo(np.float64).tiny,
            )
        )
    )


def build_historical(
    case_specs: list[list[CrestSpec]], nx: int
) -> tuple[jax.Array, jax.Array]:
    flat_specs, owner = flatten_case_specs(case_specs)
    amplitudes, centers, directions = specs_to_jax_arrays(flat_specs)
    template = make_default_tanaka_template(
        depth=1.0,
        gravity=1.0,
        direction=1,
        nx=nx,
        length=164.0,
        center=0.0,
        dno_order=6,
        pad_factor=8,
    )
    solution = solve_modified_tanaka_batched(
        template,
        amplitudes,
        centers=centers,
        directions=directions,
    )
    owner_array = jnp.asarray(owner)
    eta = jnp.zeros((len(case_specs), nx), dtype=solution.eta_periodic.dtype)
    xi = jnp.zeros_like(eta)
    eta = eta.at[owner_array].add(solution.eta_periodic)
    xi = xi.at[owner_array].add(solution.xi_periodic)
    return eta, xi - jnp.mean(xi, axis=-1, keepdims=True)


def evaluate_refinement(
    *,
    population: str,
    case_ids: list[int],
    case_specs: list[list[CrestSpec]],
    eta_n: jax.Array,
    xi_n: jax.Array,
    eta_2n: jax.Array,
    xi_2n: jax.Array,
    depth: jax.Array | float,
    length: float,
    cutoff_mode: int,
    retained_labels: dict[int, bool] | None,
) -> list[CaseMetric]:
    nx = int(eta_n.shape[-1])
    params_n = make_solver_params(
        nx,
        length,
        depth,
        dno_order=6,
        pad_factor=8,
        filter_fraction=1.0,
    )
    params_2n = make_solver_params(
        2 * nx,
        length,
        depth,
        dno_order=6,
        pad_factor=8,
        filter_fraction=1.0,
    )
    cutoff = 2.0 * np.pi * cutoff_mode / length
    project_n = projector(params_n.k, cutoff)
    project_2n = projector(params_2n.k, cutoff)

    raw_q_n = dno_series_eval(
        eta_n,
        xi_n,
        params_n.k,
        depth,
        6,
        pad_factor=8,
    )
    eta_n_projected = project_n(eta_n)
    xi_n_projected = project_n(xi_n)
    eta_2n_projected = project_2n(eta_2n)
    xi_2n_projected = project_2n(xi_2n)
    q_n = project_n(
        dno_series_eval(
            eta_n_projected,
            xi_n_projected,
            params_n.k,
            depth,
            6,
            pad_factor=8,
        )
    )
    q_2n = project_2n(
        dno_series_eval(
            eta_2n_projected,
            xi_2n_projected,
            params_2n.k,
            depth,
            6,
            pad_factor=8,
        )
    )
    eta_2n_low = _spectral_truncate_real(eta_2n_projected, nx)
    xi_2n_low = _spectral_truncate_real(xi_2n_projected, nx)
    q_2n_low = _spectral_truncate_real(q_2n, nx)
    jax.block_until_ready(q_2n_low)

    arrays = map(
        np.asarray,
        (
            raw_q_n,
            project_n(raw_q_n),
            eta_n_projected,
            xi_n_projected,
            q_n,
            eta_2n_low,
            xi_2n_low,
            q_2n_low,
        ),
    )
    (
        raw_q,
        raw_q_projected,
        eta_low,
        xi_low,
        q_low,
        eta_ref,
        xi_ref,
        q_ref,
    ) = arrays
    mode_indices = np.fft.fftfreq(nx) * nx
    physical_k = 2.0 * np.pi * mode_indices / length

    return [
        CaseMetric(
            population=population,
            case_id=case_id,
            legacy_retained_at_t0=(
                retained_labels[case_id] if retained_labels is not None else None
            ),
            sign_changes=count_sign_changes(
                q_low[index]
                if population == "current_periodic_builder"
                else raw_q[index]
            ),
            fixed_band_q_l2_defect=relative_l2(q_low[index], q_ref[index]),
            fixed_band_q_h1_defect=relative_h1(q_low[index], q_ref[index], physical_k),
            fixed_band_eta_l2_defect=relative_l2(eta_low[index], eta_ref[index]),
            fixed_band_xi_l2_defect=relative_l2(xi_low[index], xi_ref[index]),
            raw_q_out_of_band_l2=relative_l2(raw_q[index], raw_q_projected[index]),
            min_center_boundary_distance=min(
                min(spec.center, length - spec.center) for spec in case_specs[index]
            ),
        )
        for index, case_id in enumerate(case_ids)
    ]


def historical_metrics(
    args: argparse.Namespace,
) -> tuple[list[CaseMetric], dict[str, object]]:
    with zipfile.ZipFile(HISTORICAL_ARCHIVE) as archive:
        case_ids, specs_by_id, retained_by_id, provenance = select_historical_cases(
            archive,
            seed=args.seed,
            random_controls=args.random_historical_controls,
            boundary_controls=args.boundary_historical_controls,
        )
    case_specs = [specs_by_id[case_id] for case_id in case_ids]
    eta_n, xi_n = build_historical(case_specs, 1024)
    eta_2n, xi_2n = build_historical(case_specs, 2048)
    return (
        evaluate_refinement(
            population="historical_zero_extension",
            case_ids=case_ids,
            case_specs=case_specs,
            eta_n=eta_n,
            xi_n=xi_n,
            eta_2n=eta_2n,
            xi_2n=xi_2n,
            depth=1.0,
            length=164.0,
            cutoff_mode=341,
            retained_labels=retained_by_id,
        ),
        provenance,
    )


def current_metrics() -> list[CaseMetric]:
    with zipfile.ZipFile(CURRENT_ARCHIVE) as archive:
        raw_specs = json.loads(archive.read("specs_batch_0000.json"))
        case_ids = read_array(archive, "case_id_batch_0000.npy")
        times = read_array(archive, "time_batch_0000.npy")
        depths = read_array(archive, "depth_batch_0000.npy")
    selected_specs = [
        deserialize_case(raw_specs[case_id]) for case_id in CURRENT_SIGN_REJECTED_ICS
    ]
    selected_depths = np.asarray(
        [
            depths[np.flatnonzero((case_ids == case_id) & np.isclose(times, 0.0))[0]]
            for case_id in CURRENT_SIGN_REJECTED_ICS
        ],
        dtype=np.float64,
    )
    template = make_default_tanaka_template(
        depth=1.0,
        gravity=1.0,
        direction=1,
        nx=1024,
        length=2.0 * np.pi,
        center=0.0,
        dno_order=6,
        pad_factor=8,
    )
    eta_n, xi_n = build_per_case_initial_conditions(
        template_params=template,
        case_h_ref=selected_depths,
        case_specs=selected_specs,
        length=2.0 * np.pi,
        nx=1024,
        gravity=1.0,
    )
    eta_2n, xi_2n = build_per_case_initial_conditions(
        template_params=template,
        case_h_ref=selected_depths,
        case_specs=selected_specs,
        length=2.0 * np.pi,
        nx=2048,
        gravity=1.0,
    )
    return evaluate_refinement(
        population="current_periodic_builder",
        case_ids=list(CURRENT_SIGN_REJECTED_ICS),
        case_specs=selected_specs,
        eta_n=eta_n,
        xi_n=xi_n,
        eta_2n=eta_2n,
        xi_2n=xi_2n,
        depth=jnp.asarray(selected_depths)[:, None],
        length=2.0 * np.pi,
        cutoff_mode=128,
        retained_labels=None,
    )


def summarize(
    metrics: list[CaseMetric],
    tolerance: float,
    historical_provenance: dict[str, object],
) -> dict[str, object]:
    historical = [
        metric for metric in metrics if metric.population == "historical_zero_extension"
    ]
    current = [
        metric for metric in metrics if metric.population == "current_periodic_builder"
    ]
    historical_rejected = [
        metric for metric in historical if metric.legacy_retained_at_t0 is False
    ]
    historical_retained = [
        metric for metric in historical if metric.legacy_retained_at_t0 is True
    ]

    def flag_rate(rows: list[CaseMetric]) -> float:
        return float(
            np.mean([metric.fixed_band_q_l2_defect > tolerance for metric in rows])
        )

    return {
        "definition": {
            "defect": (
                "relative L2 difference of P_K G6(P_K eta, P_K xi) between "
                "independently constructed N=1024 and N=2048 grids"
            ),
            "historical_fixed_mode_cutoff": 341,
            "current_fixed_mode_cutoff": 128,
            "provisional_tolerance": tolerance,
            "legacy_rule": "reject C_0.03(Gxi) > 10",
        },
        "selection": {
            **historical_provenance,
            "historical_rows": len(historical),
            "historical_legacy_rejected_at_t0": len(historical_rejected),
            "historical_legacy_retained_at_t0": len(historical_retained),
            "current_sign_rejected_frame_zero_rows": len(current),
        },
        "provisional_tolerance_results": {
            "historical_legacy_rejected_flag_rate": flag_rate(historical_rejected),
            "historical_legacy_retained_flag_rate": flag_rate(historical_retained),
            "current_sign_rejected_flag_rate": flag_rate(current),
        },
        "metrics": [metric.__dict__ for metric in metrics],
        "limitations": [
            "The complete 548360-row historical rejected population is unavailable.",
            "The historical panel is deterministic but not a random sample of every rejected trajectory frame.",
            "The tolerance is provisional until a larger N,2N,4N calibration panel is run.",
        ],
    }


def plot_metrics(path: Path, metrics: list[CaseMetric], tolerance: float) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    styles = {
        ("historical_zero_extension", False): ("tab:red", "historical: sign-rejected"),
        ("historical_zero_extension", True): ("tab:blue", "historical: sign-retained"),
        ("current_periodic_builder", None): ("tab:green", "current: sign-rejected"),
    }
    for key, (color, label) in styles.items():
        rows = [
            metric
            for metric in metrics
            if (metric.population, metric.legacy_retained_at_t0) == key
        ]
        if rows:
            axis.scatter(
                [metric.sign_changes for metric in rows],
                [metric.fixed_band_q_l2_defect for metric in rows],
                color=color,
                s=28,
                alpha=0.8,
                label=label,
            )
    axis.axvline(10, color="black", linestyle="--", linewidth=1.0)
    axis.axhline(tolerance, color="black", linestyle=":", linewidth=1.0)
    axis.set_yscale("log")
    axis.set_xlabel("legacy 3%-deadzone sign changes")
    axis.set_ylabel(r"fixed-band $N$--$2N$ relative $q$ defect")
    axis.grid(alpha=0.25)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    jax.config.update("jax_enable_x64", True)
    historical, historical_provenance = historical_metrics(args)
    metrics = historical + current_metrics()
    summary = summarize(metrics, args.provisional_tolerance, historical_provenance)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_metrics(args.output_png, metrics, args.provisional_tolerance)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
