"""Generate (eta, xi) -> G(eta) xi data from fifth-order Stokes wavetrains.

No time integration. For each random (a0, n0, h, phase) draw, evaluate the
analytical Stokes wave (deep or finite-depth) and compute G(eta) xi via the
Craig-Sulem Taylor series at order M.

One snapshot per case, with phase uniform modulo 2 pi, gives independent
samples. Two regimes are selected by --regime {deep,shallow}.

Output layout matches the rollout-based generators (per-batch zip entries,
meta.json, state sidecar).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from numpy.lib import format as npy_format
from tqdm.auto import tqdm

from ..data.stokes_truth_jax import (
    FINITE_DEPTH_STOKES_HEIGHT_PHASE_POINTS,
    FINITE_DEPTH_STOKES_URSELL_LIMIT,
    finite_depth_stokes_ursell_upper_bound,
    stokes_eta_xi_at_phase,
)
from .pipeline.reference import (
    PAPER_DNO_TARGET,
    evaluate_discrete_dno_target,
)
from ..solvers.dno_series_jax import build_grid, dno_series_eval


_finite_depth_ursell_batch = jax.jit(
    finite_depth_stokes_ursell_upper_bound
)

PAPER_STOKES_STEEPNESS_CELLS = np.asarray(
    ((0.005, 0.03), (0.03, 0.15)),
    dtype=np.float64,
)
PAPER_GRAVITY = 1.0
ROOT = Path(__file__).resolve().parents[2]


class StokesSupportSamplingError(ValueError):
    """Finite-depth Stokes sampling exhausted its same-cell redraws."""

    def __init__(
        self,
        message: str,
        *,
        case_records: list[dict[str, object]],
    ) -> None:
        super().__init__(message)
        self.case_records = case_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stokes-wavetrain (eta, xi, G eta xi) dataset.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--regime", choices=("deep", "shallow"), required=True)
    parser.add_argument("--target_samples", type=int, default=250_000)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rng_stream_id", type=int, default=0)
    parser.add_argument("--case_id_offset", type=int, default=0)
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--length", type=float, default=2.0 * math.pi)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--n0_min", type=int, default=None)
    parser.add_argument("--n0_max", type=int, default=None)
    parser.add_argument("--a0_min", type=float, default=0.000766)
    parser.add_argument("--a0_max", type=float, default=0.011494)
    parser.add_argument("--steepness_max", type=float, default=0.15)
    parser.add_argument("--rejection_attempts", type=int, default=1000,
                        help="maximum same-cell amplitude redraws used to satisfy "
                             "the finite-depth Ursell support condition")
    parser.add_argument("--depth_min", type=float, default=None,
                        help="default: 0.02 (shallow), 4.0 (deep)")
    parser.add_argument("--depth_max", type=float, default=None,
                        help="default: 1.5 (shallow), 50.0 (deep)")
    parser.add_argument("--kh_min", type=float, default=None,
                        help="default: 0.5 (shallow), 5.0 (deep)")
    parser.add_argument("--kh_max", type=float, default=None,
                        help="default: 5.0 (shallow), unbounded (deep)")
    parser.add_argument("--dno_order", type=int, default=6)
    parser.add_argument("--pad_factor", type=int, default=8)
    parser.add_argument("--rollout_dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.depth_min is None:
        args.depth_min = 0.02 if args.regime == "shallow" else 4.0
    if args.depth_max is None:
        args.depth_max = 1.5 if args.regime == "shallow" else 50.0
    if args.n0_min is None:
        args.n0_min = 14 if args.regime == "shallow" else 1
    if args.n0_max is None:
        args.n0_max = 26 if args.regime == "shallow" else 20
    if args.kh_min is None:
        args.kh_min = 0.5 if args.regime == "shallow" else 5.0
    if args.kh_max is None:
        args.kh_max = 5.0 if args.regime == "shallow" else math.inf
    return args


def validate_paper_target_configuration(
    *,
    nx: int,
    length: float,
    gravity: float,
    dno_order: int,
    pad_factor: int,
    rollout_dtype: str,
) -> None:
    """Reject a CLI configuration that would change the frozen paper target."""

    observed = (
        nx,
        length,
        gravity,
        dno_order,
        pad_factor,
        rollout_dtype,
    )
    expected = (
        PAPER_DNO_TARGET.nx,
        PAPER_DNO_TARGET.length,
        PAPER_GRAVITY,
        PAPER_DNO_TARGET.dno_order,
        PAPER_DNO_TARGET.pad_factor,
        "float64",
    )
    if (
        observed[0] != expected[0]
        or not math.isclose(
            observed[1],
            expected[1],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or not math.isclose(
            observed[2],
            expected[2],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or observed[3:] != expected[3:]
    ):
        raise ValueError(
            "paper Stokes generation requires "
            "(nx, length, dno_order, pad_factor, dtype)="
            f"{expected}, got {observed}"
        )


def write_npy_entry(zf: zipfile.ZipFile, name: str, array: np.ndarray) -> None:
    with zf.open(name, mode="w", force_zip64=True) as handle:
        npy_format.write_array(handle, np.asarray(array), allow_pickle=False)


def load_state(state_path: Path) -> dict[str, object] | None:
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(state_path: Path, state: dict[str, object]) -> None:
    temporary = state_path.with_name(f".{state_path.name}.tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(state_path)


def save_json(path: Path, payload: dict[str, object]) -> None:
    """Atomically write a strict-JSON record."""

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_bound(value: float) -> float | str:
    if math.isfinite(value):
        return value
    return "Infinity" if value > 0.0 else "-Infinity"


def generation_configuration(
    args: argparse.Namespace,
    *,
    ichoi: int,
) -> dict[str, object]:
    """Return the strict-JSON configuration that defines archive contents."""

    source_paths = (
        Path(__file__).resolve(),
        ROOT / "solver/data/stokes_truth_jax.py",
        ROOT / "solver/gen_data/pipeline/reference.py",
        ROOT / "solver/solvers/dno_series_jax.py",
    )
    return {
        "schema": "paper_stokes_static_v1",
        "family": {
            "formula": "project_fifth_order_stokes_fixed_phase_v1",
            "regime": args.regime,
            "ichoi": ichoi,
            "n0_min": args.n0_min,
            "n0_max": args.n0_max,
            "a0_min": args.a0_min,
            "a0_max": args.a0_max,
            "steepness_max": args.steepness_max,
            "steepness_cells": PAPER_STOKES_STEEPNESS_CELLS.tolist(),
            "depth_min": args.depth_min,
            "depth_max": args.depth_max,
            "kh_min": args.kh_min,
            "kh_max": _finite_bound(args.kh_max),
            "finite_depth_ursell_limit": (
                FINITE_DEPTH_STOKES_URSELL_LIMIT
                if ichoi == 1
                else None
            ),
            "finite_depth_height_phase_points": (
                FINITE_DEPTH_STOKES_HEIGHT_PHASE_POINTS
                if ichoi == 1
                else None
            ),
            "phase_distribution": "uniform_on_[0,2pi)",
            "potential_gauge": "spatial_mean_removed",
        },
        "sampling": {
            "algorithm": "numpy_pcg64",
            "seed_sequence_entropy": [
                "seed",
                "rng_stream_id",
                "batch_index",
            ],
            "target_samples": args.target_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "rng_stream_id": args.rng_stream_id,
            "case_id_offset": args.case_id_offset,
            "rejection_attempts": args.rejection_attempts,
        },
        "target": {
            "formula": "P_K_mean_zero_G_M_P_K_eta_P_K_xi",
            "nx": PAPER_DNO_TARGET.nx,
            "length": PAPER_DNO_TARGET.length,
            "gravity": PAPER_GRAVITY,
            "dno_order": PAPER_DNO_TARGET.dno_order,
            "pad_factor": PAPER_DNO_TARGET.pad_factor,
            "maximum_wavenumber": PAPER_DNO_TARGET.maximum_wavenumber,
            "evaluation_dtype": "float64",
            "field_storage_dtype": "float32",
            "parameter_storage_dtype": "float64",
        },
        "software": {
            "jax_version": jax.__version__,
            "numpy_version": np.__version__,
            "source_sha256": {
                str(path.relative_to(ROOT)): _file_sha256(path)
                for path in source_paths
            },
        },
    }


def configuration_fingerprint(configuration: dict[str, object]) -> str:
    canonical = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_resume_artifacts(
    *,
    output_path: Path,
    state: dict[str, object],
    expected_fingerprint: str,
    target_samples: int,
    batch_size: int,
) -> None:
    """Refuse an incomplete or differently configured append-only archive."""

    if not output_path.exists():
        raise RuntimeError(
            f"state exists but archive is missing: {output_path}; use --overwrite"
        )
    state_fingerprint = str(state.get("config_fingerprint", ""))
    if state_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "state belongs to another configuration; use --overwrite"
        )
    next_batch = int(state["next_batch"])
    samples_written = int(state["samples_written"])
    expected_samples = min(next_batch * batch_size, target_samples)
    if samples_written != expected_samples:
        raise RuntimeError(
            "state batch/sample counts are inconsistent; use --overwrite"
        )

    with zipfile.ZipFile(output_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError(
                "archive contains duplicate entries; use --overwrite"
            )
        try:
            meta = json.loads(archive.read("meta.json"))
        except (KeyError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "archive metadata is missing or invalid; use --overwrite"
            ) from error
        stored_configuration = meta.get("configuration")
        stored_fingerprint = meta.get("config_fingerprint")
        if (
            not isinstance(stored_configuration, dict)
            or configuration_fingerprint(stored_configuration)
            != stored_fingerprint
        ):
            raise RuntimeError(
                "archive metadata configuration is corrupted; use --overwrite"
            )
        if stored_fingerprint != expected_fingerprint:
            raise RuntimeError(
                "archive belongs to another configuration; use --overwrite"
            )

        required = {"x.npy", "meta.json"}
        for batch_index in range(next_batch):
            tag = f"batch_{batch_index:04d}"
            required.update(
                {
                    f"eta_{tag}.npy",
                    f"xi_{tag}.npy",
                    f"gxi_{tag}.npy",
                    f"phase_{tag}.npy",
                    f"case_id_{tag}.npy",
                    f"depth_{tag}.npy",
                    f"specs_{tag}.json",
                }
            )
        missing = required.difference(names)
        if missing:
            raise RuntimeError(
                f"archive is missing expected entries {sorted(missing)}; "
                "use --overwrite"
            )
        unexpected = set(names).difference(required)
        if unexpected:
            raise RuntimeError(
                f"archive has orphaned entries {sorted(unexpected)}; "
                "use --overwrite"
            )


def make_batch_rng(seed: int, batch_idx: int, rng_stream_id: int) -> np.random.Generator:
    seed_sequence = np.random.SeedSequence(
        [seed, rng_stream_id, batch_idx]
    )
    return np.random.Generator(np.random.PCG64(seed_sequence))


def _sample_amplitudes_in_bounds(
    rng: np.random.Generator,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Draw amplitudes uniformly from case-specific half-open intervals."""

    return np.asarray(rng.uniform(lower, upper), dtype=np.float64)


def _assign_steepness_cells(
    rng: np.random.Generator,
    k0: np.ndarray,
    *,
    a0_min: float,
    a0_max: float,
    steepness_max: float,
) -> dict[str, np.ndarray]:
    """Assign a feasible paper-corpus steepness cell to every carrier mode."""

    cell_lower = PAPER_STOKES_STEEPNESS_CELLS[:, 0][None, :]
    cell_upper = np.minimum(
        PAPER_STOKES_STEEPNESS_CELLS[:, 1],
        steepness_max,
    )[None, :]
    amplitude_lower = np.maximum(a0_min, cell_lower / k0[:, None])
    amplitude_upper = np.minimum(a0_max, cell_upper / k0[:, None])
    feasible = amplitude_lower < amplitude_upper
    if np.any(~np.any(feasible, axis=1)):
        bad_modes = k0[~np.any(feasible, axis=1)]
        raise ValueError(
            "no paper Stokes steepness cell intersects the requested "
            f"amplitude interval for wavenumbers {bad_modes.tolist()}"
        )

    cell_index = np.where(feasible[:, 1], 1, 0).astype(np.int32)
    both_feasible = np.flatnonzero(np.all(feasible, axis=1))
    first_cell = int(rng.integers(0, 2))
    balanced_assignment = (
        np.arange(both_feasible.size, dtype=np.int32) + first_cell
    ) % 2
    rng.shuffle(balanced_assignment)
    cell_index[both_feasible] = balanced_assignment
    row_index = np.arange(k0.size)
    assigned_lower = amplitude_lower[row_index, cell_index]
    assigned_upper = amplitude_upper[row_index, cell_index]
    return {
        "steepness_cell_index": cell_index,
        "steepness_cell_lower": PAPER_STOKES_STEEPNESS_CELLS[
            cell_index, 0
        ],
        "steepness_cell_upper": np.minimum(
            PAPER_STOKES_STEEPNESS_CELLS[cell_index, 1],
            steepness_max,
        ),
        "amplitude_cell_lower": assigned_lower,
        "amplitude_cell_upper": assigned_upper,
    }


def _pack_rejection_history(
    histories: list[list[float]],
) -> np.ndarray:
    """Pack variable-length proposal histories into a NaN-padded array."""

    maximum_count = max(map(len, histories), default=0)
    packed = np.full(
        (len(histories), maximum_count),
        np.nan,
        dtype=np.float64,
    )
    for case_index, history in enumerate(histories):
        packed[case_index, : len(history)] = history
    return packed


def _finite_json_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _failed_support_case_records(
    *,
    params: dict[str, np.ndarray],
    ursell_number: np.ndarray,
    unsupported: np.ndarray,
    support_resampling_count: np.ndarray,
    rejected_amplitudes: list[list[float]],
    rejected_ursell_numbers: list[list[float]],
) -> list[dict[str, object]]:
    """Describe every uncommitted case in an exhausted Stokes batch."""

    records: list[dict[str, object]] = []
    for case_index in range(int(params["a0"].shape[0])):
        rejected_a0 = list(rejected_amplitudes[case_index])
        rejected_ursell = list(rejected_ursell_numbers[case_index])
        if unsupported[case_index]:
            rejected_a0.append(float(params["a0"][case_index]))
            rejected_ursell.append(float(ursell_number[case_index]))
        records.append(
            {
                "batch_case_index": case_index,
                "status": (
                    "failed_ursell_redraw_limit"
                    if unsupported[case_index]
                    else "admissible_but_batch_not_written"
                ),
                "n0": int(params["n0"][case_index]),
                "a0": float(params["a0"][case_index]),
                "depth": float(params["depth"][case_index]),
                "phase": float(params["phase"][case_index]),
                "steepness_cell_index": int(
                    params["steepness_cell_index"][case_index]
                ),
                "steepness_cell_lower": float(
                    params["steepness_cell_lower"][case_index]
                ),
                "steepness_cell_upper": float(
                    params["steepness_cell_upper"][case_index]
                ),
                "amplitude_cell_lower": float(
                    params["amplitude_cell_lower"][case_index]
                ),
                "amplitude_cell_upper": float(
                    params["amplitude_cell_upper"][case_index]
                ),
                "support_resampling_count": int(
                    support_resampling_count[case_index]
                ),
                "ursell_upper_bound": _finite_json_number(
                    float(ursell_number[case_index])
                ),
                "rejected_a0": rejected_a0,
                "rejected_ursell_upper_bound": list(
                    map(_finite_json_number, rejected_ursell)
                ),
            }
        )
    return records


def sample_stokes_case_params(
    rng: np.random.Generator, *, batch_size: int, length: float, gravity: float,
    n0_min: int, n0_max: int, a0_min: float, a0_max: float,
    steepness_max: float,
    depth_min: float, depth_max: float,
    kh_min: float, kh_max: float,
    rejection_attempts: int,
    finite_depth: bool,
) -> dict[str, np.ndarray]:
    n0 = rng.integers(n0_min, n0_max + 1, size=batch_size).astype(np.int32)
    k0 = n0.astype(np.float64) * (2.0 * np.pi / length)
    cell_assignment = _assign_steepness_cells(
        rng,
        k0,
        a0_min=a0_min,
        a0_max=a0_max,
        steepness_max=steepness_max,
    )
    a0 = _sample_amplitudes_in_bounds(
        rng,
        cell_assignment["amplitude_cell_lower"],
        cell_assignment["amplitude_cell_upper"],
    )
    lo = np.maximum(depth_min, kh_min / k0)
    hi = np.minimum(depth_max, kh_max / k0)
    if np.any(lo > hi):
        raise ValueError(
            f"kh constraint produces empty depth range for some n0; "
            f"got lo={lo.min():.3f} hi={hi.max():.3f}. Adjust kh_min/kh_max or n0 range."
        )
    if np.allclose(lo, hi):
        depth = lo
    else:
        u = rng.uniform(0.0, 1.0, size=batch_size)
        depth = (np.exp(np.log(lo) + u * (np.log(np.maximum(hi, lo)) - np.log(lo)))).astype(np.float64)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=batch_size).astype(np.float64)
    params = {
        "n0": n0,
        "a0": a0,
        "depth": depth,
        "phase": phase,
        **cell_assignment,
    }
    if not finite_depth:
        return params

    support_resampling_count = np.zeros(batch_size, dtype=np.int32)
    rejected_amplitudes: list[list[float]] = [
        [] for _ in range(batch_size)
    ]
    rejected_ursell_numbers: list[list[float]] = [
        [] for _ in range(batch_size)
    ]
    ursell_number = np.asarray(
        jax.device_get(
            _finite_depth_ursell_batch(
                jnp.asarray(k0),
                jnp.asarray(depth),
                gravity=gravity,
                a0=jnp.asarray(a0),
            )
        ),
        dtype=np.float64,
    ).copy()
    unsupported = ~np.isfinite(ursell_number) | (
        ursell_number > FINITE_DEPTH_STOKES_URSELL_LIMIT
    )
    for _ in range(rejection_attempts):
        if not unsupported.any():
            params["ursell_upper_bound"] = ursell_number
            params["support_resampling_count"] = support_resampling_count
            params["rejected_a0"] = _pack_rejection_history(
                rejected_amplitudes
            )
            params["rejected_ursell_upper_bound"] = _pack_rejection_history(
                rejected_ursell_numbers
            )
            return params
        for case_index in np.flatnonzero(unsupported):
            rejected_amplitudes[int(case_index)].append(
                float(a0[case_index])
            )
            rejected_ursell_numbers[int(case_index)].append(
                float(ursell_number[case_index])
            )
        support_resampling_count[unsupported] += 1
        a0[unsupported] = _sample_amplitudes_in_bounds(
            rng,
            cell_assignment["amplitude_cell_lower"][unsupported],
            cell_assignment["amplitude_cell_upper"][unsupported],
        )
        ursell_number = np.asarray(
            jax.device_get(
                _finite_depth_ursell_batch(
                    jnp.asarray(k0),
                    jnp.asarray(depth),
                    gravity=gravity,
                    a0=jnp.asarray(a0),
                )
            ),
            dtype=np.float64,
        ).copy()
        unsupported = ~np.isfinite(ursell_number) | (
            ursell_number > FINITE_DEPTH_STOKES_URSELL_LIMIT
        )
    if not unsupported.any():
        params["ursell_upper_bound"] = ursell_number
        params["support_resampling_count"] = support_resampling_count
        params["rejected_a0"] = _pack_rejection_history(
            rejected_amplitudes
        )
        params["rejected_ursell_upper_bound"] = _pack_rejection_history(
            rejected_ursell_numbers
        )
        return params
    raise StokesSupportSamplingError(
        "failed to sample finite-depth Stokes amplitudes with Ur <= "
        f"{FINITE_DEPTH_STOKES_URSELL_LIMIT:g}; check the requested parameter cell",
        case_records=_failed_support_case_records(
            params=params,
            ursell_number=ursell_number,
            unsupported=unsupported,
            support_resampling_count=support_resampling_count,
            rejected_amplitudes=rejected_amplitudes,
            rejected_ursell_numbers=rejected_ursell_numbers,
        ),
    )


def serialize_specs(
    params: dict[str, np.ndarray],
    ichoi: int,
) -> list[dict[str, object]]:
    n = int(params["a0"].shape[0])
    return [
        {
            "n0": int(params["n0"][i]),
            "a0": float(params["a0"][i]),
            "depth": float(params["depth"][i]),
            "phase": float(params["phase"][i]),
            "steepness_cell_index": int(
                params["steepness_cell_index"][i]
            ),
            "steepness_cell_lower": float(
                params["steepness_cell_lower"][i]
            ),
            "steepness_cell_upper": float(
                params["steepness_cell_upper"][i]
            ),
            "amplitude_cell_lower": float(
                params["amplitude_cell_lower"][i]
            ),
            "amplitude_cell_upper": float(
                params["amplitude_cell_upper"][i]
            ),
            "ichoi": ichoi,
            **(
                {
                    "ursell_upper_bound": float(
                        params["ursell_upper_bound"][i]
                    )
                }
                if "ursell_upper_bound" in params
                else {}
            ),
            **(
                {
                    "support_resampling_count": int(
                        params["support_resampling_count"][i]
                    )
                }
                if "support_resampling_count" in params
                else {}
            ),
            **(
                {
                    "rejected_a0": [
                        float(value)
                        for value in params["rejected_a0"][i][
                            : int(params["support_resampling_count"][i])
                        ]
                    ],
                    "rejected_ursell_upper_bound": [
                        (
                            float(value)
                            if np.isfinite(value)
                            else None
                        )
                        for value in params[
                            "rejected_ursell_upper_bound"
                        ][i][
                            : int(params["support_resampling_count"][i])
                        ]
                    ],
                }
                if "rejected_a0" in params
                else {}
            ),
        }
        for i in range(n)
    ]


def build_stokes_batch(
    *, x: jnp.ndarray, k: jnp.ndarray, length: float, gravity: float,
    n0: jnp.ndarray, a0: jnp.ndarray, depth: jnp.ndarray, phase: jnp.ndarray,
    ichoi: int, dno_order: int, pad_factor: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build a generic Stokes batch and an unprojected configurable DNO."""

    eta, xi = build_stokes_states(
        x=x,
        length=length,
        gravity=gravity,
        n0=n0,
        a0=a0,
        depth=depth,
        phase=phase,
        ichoi=ichoi,
    )
    depth_2d = depth[:, None]
    gxi = dno_series_eval(
        eta,
        xi,
        k,
        depth_2d,
        dno_order,
        pad_factor=pad_factor,
    )
    return eta, xi, gxi


def build_stokes_states(
    *,
    x: jnp.ndarray,
    length: float,
    gravity: float,
    n0: jnp.ndarray,
    a0: jnp.ndarray,
    depth: jnp.ndarray,
    phase: jnp.ndarray,
    ichoi: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Construct a vectorized batch of fixed-phase Stokes states."""

    def per_case(n0_i, a0_i, depth_i, phase_i):
        return stokes_eta_xi_at_phase(
            x=x, phase=phase_i, n0=n0_i, a0=a0_i, length=length, depth=depth_i,
            gravity=gravity, ichoi=ichoi,
        )
    eta_xi = jax.vmap(per_case)(n0, a0, depth, phase)
    return eta_xi


def build_paper_stokes_batch(
    *,
    gravity: float,
    n0: jnp.ndarray,
    a0: jnp.ndarray,
    depth: jnp.ndarray,
    phase: jnp.ndarray,
    ichoi: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build Stokes states and the frozen paper-corpus DNO target."""

    if not math.isclose(
        gravity,
        PAPER_GRAVITY,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(f"paper Stokes generation requires gravity={PAPER_GRAVITY}")
    x, _ = build_grid(PAPER_DNO_TARGET.nx, PAPER_DNO_TARGET.length)
    eta, xi = build_stokes_states(
        x=jnp.asarray(x, dtype=jnp.float64),
        length=PAPER_DNO_TARGET.length,
        gravity=gravity,
        n0=n0,
        a0=a0,
        depth=depth,
        phase=phase,
        ichoi=ichoi,
    )
    return evaluate_discrete_dno_target(
        eta,
        xi,
        depth[:, None],
    )


def main() -> None:
    args = parse_args()
    validate_paper_target_configuration(
        nx=args.nx,
        length=args.length,
        gravity=args.gravity,
        dno_order=args.dno_order,
        pad_factor=args.pad_factor,
        rollout_dtype=args.rollout_dtype,
    )
    jax.config.update("jax_enable_x64", args.rollout_dtype == "float64")
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = output_path.with_suffix(".state.json")
    failure_path = output_path.with_suffix(".failure.json")
    ichoi = 0 if args.regime == "deep" else 1
    configuration = generation_configuration(args, ichoi=ichoi)
    config_fingerprint = configuration_fingerprint(configuration)

    if args.overwrite:
        if output_path.exists():
            output_path.unlink()
        if state_path.exists():
            state_path.unlink()
        if failure_path.exists():
            failure_path.unlink()

    samples_per_batch = args.batch_size
    n_batches = int(math.ceil(args.target_samples / samples_per_batch))

    existing_state = load_state(state_path)
    if existing_state is None:
        if output_path.exists():
            raise RuntimeError(
                f"archive exists without a matching state: {output_path}; "
                "use --overwrite"
            )
        samples_written = 0
        next_batch = 0
        support_rejections_total = 0
        prior_elapsed_seconds = 0.0
        with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            x_grid, _ = build_grid(args.nx, args.length)
            meta = {
                "dataset_kind": "stokes_analytic",
                "schema": "paper_stokes_static_v1",
                "config_fingerprint": config_fingerprint,
                "configuration": configuration,
                "regime": args.regime,
                "ichoi": ichoi,
                "target_samples": args.target_samples,
                "batch_size": args.batch_size,
                "samples_per_full_batch": samples_per_batch,
                "n_batches_planned": n_batches,
                "rollout_dtype": args.rollout_dtype,
                "field_storage_dtype": "float32",
                "parameter_storage_dtype": "float64",
                "n0_min": args.n0_min, "n0_max": args.n0_max,
                "a0_min": args.a0_min, "a0_max": args.a0_max,
                "steepness_max": args.steepness_max,
                "rejection_attempts": args.rejection_attempts,
                "steepness_cells": (
                    PAPER_STOKES_STEEPNESS_CELLS.tolist()
                ),
                "steepness_cell_distribution": (
                    "balanced_to_within_one_over_cases_where_both_cells_"
                    "are_feasible"
                ),
                "a0_distribution": (
                    "uniform_within_preassigned_steepness_cell"
                ),
                "depth_min": args.depth_min, "depth_max": args.depth_max,
                "kh_min": args.kh_min,
                "kh_max": _finite_bound(args.kh_max),
                "depth_distribution": "log_uniform_clipped_to_kh_range",
                "finite_depth_ursell_limit": (
                    FINITE_DEPTH_STOKES_URSELL_LIMIT
                    if ichoi == 1
                    else None
                ),
                "finite_depth_support": (
                    "conservative_H_plus_times_wavelength_squared_over_depth_cubed"
                    if ichoi == 1
                    else None
                ),
                "finite_depth_height_phase_points": (
                    FINITE_DEPTH_STOKES_HEIGHT_PHASE_POINTS
                    if ichoi == 1
                    else None
                ),
                "finite_depth_height_bound": (
                    "H_grid_plus_one_quarter_phase_spacing_squared_times_"
                    "sum_m_squared_abs_E_m"
                    if ichoi == 1
                    else None
                ),
                "phase_distribution": "uniform_on_[0,2pi)",
                "potential_gauge": "spatial_mean_removed",
                "dno_order": args.dno_order, "pad_factor": args.pad_factor,
                "seed": args.seed, "rng_stream_id": args.rng_stream_id,
                "rng_algorithm": "numpy_pcg64",
                "rng_seed_sequence_entropy": [
                    "seed",
                    "rng_stream_id",
                    "batch_index",
                ],
                "case_id_offset": args.case_id_offset,
                "nx": args.nx, "length": args.length, "gravity": args.gravity,
            }
            write_npy_entry(zf, "x.npy", np.asarray(x_grid, dtype=np.float64))
            zf.writestr(
                "meta.json",
                json.dumps(
                    meta,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ),
            )
        save_state(state_path, {
            "output_path": str(output_path), "samples_written": 0, "next_batch": 0,
            "n_batches_planned": n_batches, "complete": False,
            "support_rejections_total": 0,
            "config_fingerprint": config_fingerprint,
        })
    else:
        validate_resume_artifacts(
            output_path=output_path,
            state=existing_state,
            expected_fingerprint=config_fingerprint,
            target_samples=args.target_samples,
            batch_size=args.batch_size,
        )
        samples_written = int(existing_state["samples_written"])
        next_batch = int(existing_state["next_batch"])
        support_rejections_total = int(
            existing_state.get("support_rejections_total", 0)
        )
        prior_elapsed_seconds = float(
            existing_state.get("elapsed_seconds", 0.0)
        )
        if bool(existing_state.get("complete", False)) or samples_written >= args.target_samples:
            print(json.dumps(existing_state, indent=2))
            return

    rollout_dtype = jnp.float32 if args.rollout_dtype == "float32" else jnp.float64
    field_save_dtype = np.float32
    parameter_save_dtype = np.float64

    @jax.jit
    def jit_build(n0, a0, depth, phase):
        return build_paper_stokes_batch(
            gravity=args.gravity,
            n0=n0, a0=a0, depth=depth, phase=phase, ichoi=ichoi,
        )

    total_start = perf_counter()
    pbar = tqdm(range(next_batch, n_batches), desc=f"stokes-{args.regime}",
                initial=next_batch, total=n_batches, dynamic_ncols=True, mininterval=1.0)
    for batch_idx in pbar:
        batch_start = perf_counter()
        remaining = args.target_samples - samples_written
        draw_count = min(args.batch_size, remaining)
        rng = make_batch_rng(args.seed, batch_idx, args.rng_stream_id)
        try:
            params = sample_stokes_case_params(
                rng, batch_size=draw_count, length=args.length,
                gravity=args.gravity,
                n0_min=args.n0_min, n0_max=args.n0_max,
                a0_min=args.a0_min, a0_max=args.a0_max,
                steepness_max=args.steepness_max,
                depth_min=args.depth_min, depth_max=args.depth_max,
                kh_min=args.kh_min, kh_max=args.kh_max,
                rejection_attempts=args.rejection_attempts,
                finite_depth=ichoi == 1,
            )
        except StokesSupportSamplingError as error:
            first_case_id = (
                args.case_id_offset + batch_idx * args.batch_size
            )
            save_json(
                failure_path,
                {
                    "schema": "paper_stokes_support_failure_v1",
                    "output_path": str(output_path),
                    "config_fingerprint": config_fingerprint,
                    "batch_index": batch_idx,
                    "first_case_id": first_case_id,
                    "case_ids": list(
                        range(first_case_id, first_case_id + draw_count)
                    ),
                    "finite_depth_ursell_limit": (
                        FINITE_DEPTH_STOKES_URSELL_LIMIT
                    ),
                    "maximum_same_cell_redraws": args.rejection_attempts,
                    "cases": error.case_records,
                },
            )
            raise
        support_resampling_count = params.get(
            "support_resampling_count",
            np.zeros(draw_count, dtype=np.int32),
        )
        eta, xi, gxi = jit_build(
            jnp.asarray(params["n0"], dtype=rollout_dtype),
            jnp.asarray(params["a0"], dtype=rollout_dtype),
            jnp.asarray(params["depth"], dtype=rollout_dtype),
            jnp.asarray(params["phase"], dtype=rollout_dtype),
        )
        jax.block_until_ready(gxi)

        eta_np = np.asarray(jax.device_get(eta), dtype=field_save_dtype)
        xi_np = np.asarray(jax.device_get(xi), dtype=field_save_dtype)
        gxi_np = np.asarray(jax.device_get(gxi), dtype=field_save_dtype)

        global_case_ids = np.arange(
            args.case_id_offset + batch_idx * args.batch_size,
            args.case_id_offset + batch_idx * args.batch_size + draw_count,
            dtype=np.int64,
        )
        phase_arr = params["phase"].astype(parameter_save_dtype)
        depth_arr = params["depth"].astype(parameter_save_dtype)

        keep = draw_count
        support_rejections_total += int(
            np.sum(support_resampling_count[:keep])
        )
        eta_np = eta_np[:keep]
        xi_np = xi_np[:keep]
        gxi_np = gxi_np[:keep]
        global_case_ids = global_case_ids[:keep]
        phase_arr = phase_arr[:keep]
        depth_arr = depth_arr[:keep]

        with zipfile.ZipFile(output_path, mode="a", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            tag = f"batch_{batch_idx:04d}"
            write_npy_entry(zf, f"eta_{tag}.npy", eta_np)
            write_npy_entry(zf, f"xi_{tag}.npy", xi_np)
            write_npy_entry(zf, f"gxi_{tag}.npy", gxi_np)
            write_npy_entry(zf, f"phase_{tag}.npy", phase_arr)
            write_npy_entry(zf, f"case_id_{tag}.npy", global_case_ids)
            write_npy_entry(zf, f"depth_{tag}.npy", depth_arr)
            zf.writestr(
                f"specs_{tag}.json",
                json.dumps(
                    serialize_specs(params, ichoi)[:keep],
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ),
            )

        samples_written += keep
        batch_seconds = perf_counter() - batch_start
        pbar.set_postfix(samples=f"{samples_written}/{args.target_samples}", batch_s=f"{batch_seconds:.2f}")
        save_state(state_path, {
            "output_path": str(output_path), "samples_written": samples_written,
            "next_batch": batch_idx + 1, "n_batches_planned": n_batches,
            "complete": samples_written >= args.target_samples,
            "support_rejections_total": support_rejections_total,
            "last_batch_seconds": batch_seconds,
            "session_elapsed_seconds": perf_counter() - total_start,
            "elapsed_seconds": (
                prior_elapsed_seconds + perf_counter() - total_start
            ),
            "config_fingerprint": config_fingerprint,
        })


if __name__ == "__main__":
    main()
