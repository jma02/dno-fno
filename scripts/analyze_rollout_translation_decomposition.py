"""Decompose saved periodic rollout errors into translation and shape parts.

For every saved frame and simulation, this script finds the continuous periodic
displacement that minimizes the elevation error.  The same elevation-derived
displacement is then applied to every requested field, so the optional
``xi`` and ``q = G(eta) xi`` diagnostics describe one common state-space
registration rather than unrelated field-wise alignments.

The input is an ``eval_suite.py`` trajectory archive with arrays shaped
``(time, simulation, grid)``. The output JSON is accompanied by a compressed NPZ
of framewise arrays and CSV tables for per-simulation onsets and population
correlations.  This is a CPU-only, saved-data analysis; it does not load a
model or run a solver.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr, spearmanr


Array = np.ndarray
FIELD_KEYS: dict[str, tuple[str, str]] = {
    "eta": ("truth_eta", "pred_eta"),
    "xi": ("truth_xi", "pred_xi"),
    "q": ("truth_gxi", "pred_gxi"),
}


def periodic_shift(
    values: Array,
    displacement: float,
    length: float,
) -> Array:
    """Return ``values(x - displacement)`` by Fourier interpolation."""
    nx = values.shape[-1]
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    return np.fft.ifft(
        np.fft.fft(values) * np.exp(-1j * wave_numbers * displacement)
    ).real


def optimal_displacement(
    prediction: Array,
    truth: Array,
    length: float,
) -> float:
    """Find the continuous periodic displacement aligning prediction to truth."""
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        return float("nan")
    nx = truth.size
    dx = length / nx
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    prediction_hat = np.fft.fft(prediction)
    correlation = np.fft.ifft(prediction_hat * np.conj(np.fft.fft(truth))).real
    candidate_count = min(8, nx)
    peak_indices = np.argpartition(correlation, -candidate_count)[-candidate_count:]

    def objective(displacement: float) -> float:
        aligned = np.fft.ifft(
            prediction_hat * np.exp(1j * wave_numbers * displacement)
        ).real
        difference = aligned - truth
        return float(np.vdot(difference, difference).real)

    candidates: list[tuple[float, float]] = []
    for peak_index in peak_indices:
        signed_index = int(peak_index)
        if signed_index > nx // 2:
            signed_index -= nx
        coarse = signed_index * dx
        result: Any = minimize_scalar(
            objective,
            bounds=(coarse - dx, coarse + dx),
            method="bounded",
            options={"xatol": 1e-12},
        )
        candidates.append((float(result.fun), float(result.x)))
    return min(candidates)[1]


def unwrap_displacements(displacements: Array, length: float) -> Array:
    """Unwrap each contiguous finite portion of a periodic displacement series."""
    result = np.asarray(displacements, dtype=np.float64).copy()
    finite_indices = np.flatnonzero(np.isfinite(result))
    if not finite_indices.size:
        return result
    split_points = np.flatnonzero(np.diff(finite_indices) > 1) + 1
    for segment in np.split(finite_indices, split_points):
        angles = result[segment] * (2.0 * np.pi / length)
        result[segment] = np.unwrap(angles) * (length / (2.0 * np.pi))
    return result


def spectral_derivative(values: Array, length: float) -> Array:
    """Return the periodic Fourier derivative along the last axis."""
    nx = values.shape[-1]
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    return np.fft.ifft(
        1j * wave_numbers * np.fft.fft(values, axis=-1),
        axis=-1,
    ).real


def aligned_fields(
    prediction: Array,
    displacement: Array,
    length: float,
) -> Array:
    """Shift each predicted row by the negative fitted displacement."""
    nx = prediction.shape[-1]
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    phase = np.exp(1j * displacement[:, None] * wave_numbers[None, :])
    return np.fft.ifft(
        np.fft.fft(prediction, axis=-1) * phase,
        axis=-1,
    ).real


def parse_float_list(value: str) -> tuple[float, ...]:
    """Parse a comma-separated finite float list."""
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values or not all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("expected a nonempty list of finite floats")
    return values


def parse_field_list(value: str) -> tuple[str, ...]:
    """Parse and validate a comma-separated field list."""
    fields = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = sorted(set(fields).difference(FIELD_KEYS))
    if not fields or "eta" not in fields or invalid:
        raise argparse.ArgumentTypeError(
            f"fields must include eta and be drawn from eta,xi,q; got {fields}"
        )
    return tuple(dict.fromkeys(fields))


def parse_int_list(value: str) -> tuple[int, ...]:
    """Parse a comma-separated integer list."""
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated integer list"
        ) from error
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("expected nonnegative simulation indices")
    return tuple(dict.fromkeys(values))


def validate_field_pair(
    truth: Array,
    prediction: Array,
    *,
    name: str,
    expected_time_simulations: tuple[int, int],
) -> None:
    """Validate one saved truth/prediction field pair."""
    if truth.ndim != 3 or prediction.shape != truth.shape:
        raise ValueError(
            f"{name} truth/prediction must share shape (time, simulation, grid), got "
            f"{truth.shape} and {prediction.shape}"
        )
    if truth.shape[:2] != expected_time_simulations:
        raise ValueError(
            f"{name} has time/simulation shape {truth.shape[:2]}, expected "
            f"{expected_time_simulations}"
        )


def compute_eta_alignment(
    truth_eta: Array,
    pred_eta: Array,
    length: float,
    *,
    progress: bool = False,
) -> tuple[Array, Array, Array]:
    """Return wrapped displacement, unwrapped displacement, and finite mask.

    The wrapped displacement ``d[t, c]`` is the global candidate returned by
    the existing continuous Fourier registration, namely

    ``argmin_d ||pred_eta(t,c,x+d) - truth_eta(t,c,x)||_2``.

    Unwrapping changes only the representative modulo ``length`` and is done
    separately on every contiguous finite segment of each simulation.
    """
    n_times, n_simulations, _ = truth_eta.shape
    wrapped = np.full((n_times, n_simulations), np.nan, dtype=np.float64)
    finite = np.all(
        np.isfinite(truth_eta) & np.isfinite(pred_eta),
        axis=-1,
    )
    for frame in range(n_times):
        frame_mask = finite[frame]
        if np.any(frame_mask):
            prediction_rows = np.asarray(pred_eta[frame, frame_mask], dtype=np.float64)
            truth_rows = np.asarray(truth_eta[frame, frame_mask], dtype=np.float64)
            wrapped[frame, frame_mask] = np.asarray(
                [
                    optimal_displacement(prediction, truth, length)
                    for prediction, truth in zip(prediction_rows, truth_rows)
                ],
                dtype=np.float64,
            )
        if progress and (frame == 0 or (frame + 1) % 10 == 0 or frame + 1 == n_times):
            print(
                f"alignment frame {frame + 1}/{n_times}: "
                f"{int(np.count_nonzero(frame_mask))}/{n_simulations} finite",
                flush=True,
            )
    unwrapped = np.stack(
        [
            unwrap_displacements(wrapped[:, simulation], length)
            for simulation in range(n_simulations)
        ],
        axis=1,
    )
    return wrapped, unwrapped, finite


def integrate_finite_segments(values: Array, times: Array) -> Array:
    """Cumulatively trapezoid-integrate contiguous finite time segments."""
    integral = np.full(values.shape, np.nan, dtype=np.float64)
    for simulation in range(values.shape[1]):
        finite_indices = np.flatnonzero(np.isfinite(values[:, simulation]))
        if not finite_indices.size:
            continue
        split_points = np.flatnonzero(np.diff(finite_indices) > 1) + 1
        for segment in np.split(finite_indices, split_points):
            integral[segment[0], simulation] = 0.0
            for previous, current in zip(segment[:-1], segment[1:]):
                dt = times[current] - times[previous]
                integral[current, simulation] = integral[
                    previous, simulation
                ] + 0.5 * dt * (
                    values[previous, simulation] + values[current, simulation]
                )
    return integral


def compute_alignment_velocity_identity(
    truth_eta: Array,
    pred_eta: Array,
    truth_q: Array,
    pred_q: Array,
    displacement: Array,
    times: Array,
    length: float,
    *,
    denominator_tolerance: float = 1e-12,
) -> dict[str, Array]:
    r"""Evaluate the exact velocity of the least-squares alignment.

    Write ``S_d f(x)=f(x+d)``, ``a=S_d eta_prediction`` and
    ``e=a-eta_truth``.  At a differentiable, isolated minimizer,
    ``<e,a_x>=0``.  Differentiating this stationarity equation and using
    ``eta_t=q`` gives

    .. math::

       d'=-\frac{\langle S_dq_p-q_y,a_x\rangle
                    +\langle e,S_d(q_p)_x\rangle}
                   {\langle a_x,a_x\rangle+\langle e,a_{xx}\rangle}.

    The returned direct and geometry terms are the two numerator
    contributions after division by the common denominator.  Frames with a
    denominator smaller than ``denominator_tolerance * <a_x,a_x>`` are marked
    undefined because the fitted group-orbit coordinate is ill-conditioned.
    """
    n_times, n_simulations, _ = truth_eta.shape
    shape = (n_times, n_simulations)
    result: dict[str, Array] = {
        "identity_velocity": np.full(shape, np.nan, dtype=np.float64),
        "direct_velocity": np.full(shape, np.nan, dtype=np.float64),
        "geometry_velocity": np.full(shape, np.nan, dtype=np.float64),
        "denominator": np.full(shape, np.nan, dtype=np.float64),
        "denominator_over_tangent_energy": np.full(shape, np.nan, dtype=np.float64),
        "stationarity_cosine": np.full(shape, np.nan, dtype=np.float64),
        "stationarity_correction": np.full(shape, np.nan, dtype=np.float64),
    }
    for frame in range(n_times):
        finite = (
            np.isfinite(displacement[frame])
            & np.all(np.isfinite(truth_eta[frame]), axis=-1)
            & np.all(np.isfinite(pred_eta[frame]), axis=-1)
            & np.all(np.isfinite(truth_q[frame]), axis=-1)
            & np.all(np.isfinite(pred_q[frame]), axis=-1)
        )
        if not np.any(finite):
            continue
        shifts = displacement[frame, finite]
        aligned_eta = aligned_fields(
            np.asarray(pred_eta[frame, finite], dtype=np.float64),
            shifts,
            length,
        )
        aligned_q = aligned_fields(
            np.asarray(pred_q[frame, finite], dtype=np.float64),
            shifts,
            length,
        )
        truth_eta_rows = np.asarray(truth_eta[frame, finite], dtype=np.float64)
        truth_q_rows = np.asarray(truth_q[frame, finite], dtype=np.float64)
        error = aligned_eta - truth_eta_rows
        aligned_eta_x = spectral_derivative(aligned_eta, length)
        aligned_eta_xx = spectral_derivative(aligned_eta_x, length)
        aligned_q_x = spectral_derivative(aligned_q, length)
        q_error = aligned_q - truth_q_rows
        tangent_energy = np.sum(aligned_eta_x**2, axis=-1)
        denominator = tangent_energy + np.sum(error * aligned_eta_xx, axis=-1)
        direct_numerator = np.sum(q_error * aligned_eta_x, axis=-1)
        geometry_numerator = np.sum(error * aligned_q_x, axis=-1)
        well_conditioned = np.abs(denominator) > (
            denominator_tolerance
            * np.maximum(tangent_energy, np.finfo(np.float64).tiny)
        )
        direct_velocity = np.full(denominator.shape, np.nan, dtype=np.float64)
        geometry_velocity = np.full(denominator.shape, np.nan, dtype=np.float64)
        direct_velocity[well_conditioned] = (
            -direct_numerator[well_conditioned] / denominator[well_conditioned]
        )
        geometry_velocity[well_conditioned] = (
            -geometry_numerator[well_conditioned] / denominator[well_conditioned]
        )
        denominator_ratio = np.full(denominator.shape, np.nan, dtype=np.float64)
        positive_tangent = tangent_energy > 0.0
        denominator_ratio[positive_tangent] = (
            denominator[positive_tangent] / tangent_energy[positive_tangent]
        )
        stationarity = np.full(denominator.shape, np.nan, dtype=np.float64)
        stationarity_inner_product = np.sum(error * aligned_eta_x, axis=-1)
        error_norm = np.linalg.norm(error, axis=-1)
        tangent_norm = np.sqrt(tangent_energy)
        nonzero_norm = (error_norm > 0.0) & (tangent_norm > 0.0)
        stationarity[nonzero_norm] = stationarity_inner_product[nonzero_norm] / (
            error_norm[nonzero_norm] * tangent_norm[nonzero_norm]
        )
        stationarity_correction = np.full(denominator.shape, np.nan, dtype=np.float64)
        stationarity_correction[well_conditioned] = (
            stationarity_inner_product[well_conditioned] / denominator[well_conditioned]
        )

        result["identity_velocity"][frame, finite] = direct_velocity + geometry_velocity
        result["direct_velocity"][frame, finite] = direct_velocity
        result["geometry_velocity"][frame, finite] = geometry_velocity
        result["denominator"][frame, finite] = denominator
        result["denominator_over_tangent_energy"][frame, finite] = denominator_ratio
        result["stationarity_cosine"][frame, finite] = stationarity
        result["stationarity_correction"][frame, finite] = stationarity_correction

    unwrapped_displacement = np.stack(
        [
            unwrap_displacements(displacement[:, simulation], length)
            for simulation in range(n_simulations)
        ],
        axis=1,
    )
    numerical_velocity: Array = np.full(
        unwrapped_displacement.shape,
        np.nan,
        dtype=np.float64,
    )
    for simulation in range(unwrapped_displacement.shape[1]):
        finite_indices = np.flatnonzero(
            np.isfinite(unwrapped_displacement[:, simulation])
        )
        if not finite_indices.size:
            continue
        split_points = np.flatnonzero(np.diff(finite_indices) > 1) + 1
        for segment in np.split(finite_indices, split_points):
            if segment.size < 2:
                continue
            edge_order = 2 if segment.size >= 3 else 1
            numerical_velocity[segment, simulation] = np.gradient(
                unwrapped_displacement[segment, simulation],
                times[segment],
                edge_order=edge_order,
            )
    integrated_velocity = integrate_finite_segments(result["identity_velocity"], times)
    result["numerical_velocity"] = numerical_velocity
    result["integrated_identity_velocity"] = integrated_velocity
    result["integrated_direct_velocity"] = integrate_finite_segments(
        result["direct_velocity"], times
    )
    result["integrated_geometry_velocity"] = integrate_finite_segments(
        result["geometry_velocity"], times
    )
    return result


def _center_rows(values: Array) -> Array:
    """Remove the spatial mean independently from every row."""
    return values - np.mean(values, axis=-1, keepdims=True)


def compute_field_metrics(
    truth: Array,
    prediction: Array,
    displacement: Array,
    length: float,
    *,
    center: bool,
) -> dict[str, Array]:
    """Compute raw and elevation-aligned errors for one field.

    If ``e_r`` and ``e_a`` are the raw and aligned relative errors, the
    translation-removable component is defined by

    ``e_T = sqrt(max(e_r**2 - e_a**2, 0))``.

    For elevation, global minimization guarantees ``e_a <= e_r`` up to
    numerical tolerance.  For another field aligned by elevation, the signed
    squared-error removal can be negative; this correctly records that a
    common elevation shift worsened that field rather than silently fitting a
    second displacement.
    """
    n_times, n_simulations, nx = truth.shape
    shape = (n_times, n_simulations)
    metrics = {
        "truth_l2_norm": np.full(shape, np.nan, dtype=np.float64),
        "raw_relative_error": np.full(shape, np.nan, dtype=np.float64),
        "aligned_relative_error": np.full(shape, np.nan, dtype=np.float64),
        "translation_relative_error": np.full(shape, np.nan, dtype=np.float64),
        "signed_squared_error_removed": np.full(shape, np.nan, dtype=np.float64),
        "raw_rms_error": np.full(shape, np.nan, dtype=np.float64),
        "aligned_rms_error": np.full(shape, np.nan, dtype=np.float64),
        "translation_rms_error": np.full(shape, np.nan, dtype=np.float64),
    }
    root_nx = np.sqrt(float(nx))
    for frame in range(n_times):
        truth_frame = np.asarray(truth[frame], dtype=np.float64)
        prediction_frame = np.asarray(prediction[frame], dtype=np.float64)
        if center:
            truth_frame = _center_rows(truth_frame)
            prediction_frame = _center_rows(prediction_frame)
        finite = (
            np.isfinite(displacement[frame])
            & np.all(np.isfinite(truth_frame), axis=-1)
            & np.all(np.isfinite(prediction_frame), axis=-1)
        )
        if not np.any(finite):
            continue
        truth_rows = truth_frame[finite]
        prediction_rows = prediction_frame[finite]
        aligned_rows = aligned_fields(
            prediction_rows,
            displacement[frame, finite],
            length,
        )
        raw_norm = np.linalg.norm(prediction_rows - truth_rows, axis=-1)
        aligned_norm = np.linalg.norm(aligned_rows - truth_rows, axis=-1)
        truth_norm = np.linalg.norm(truth_rows, axis=-1)
        gain_squared = raw_norm**2 - aligned_norm**2
        translation_norm = np.sqrt(np.maximum(gain_squared, 0.0))
        has_reference = truth_norm > 0.0
        has_raw_error = raw_norm > 0.0

        metrics["truth_l2_norm"][frame, finite] = truth_norm
        for key, numerator in (
            ("raw_relative_error", raw_norm),
            ("aligned_relative_error", aligned_norm),
            ("translation_relative_error", translation_norm),
        ):
            values = np.full(truth_norm.shape, np.nan, dtype=np.float64)
            values[has_reference] = numerator[has_reference] / truth_norm[has_reference]
            metrics[key][frame, finite] = values
        removed = np.zeros(raw_norm.shape, dtype=np.float64)
        removed[has_raw_error] = (
            gain_squared[has_raw_error] / raw_norm[has_raw_error] ** 2
        )
        metrics["signed_squared_error_removed"][frame, finite] = removed
        metrics["raw_rms_error"][frame, finite] = raw_norm / root_nx
        metrics["aligned_rms_error"][frame, finite] = aligned_norm / root_nx
        metrics["translation_rms_error"][frame, finite] = translation_norm / root_nx
    return metrics


def first_crossing_index(values: Array, threshold: float) -> int | None:
    """Return the first index at which a finite series strictly exceeds a threshold."""
    indices = np.flatnonzero(np.isfinite(values) & (values > threshold))
    return int(indices[0]) if indices.size else None


def _finite_float(value: float) -> float | None:
    """Convert a NumPy scalar to a JSON-safe float."""
    result = float(value)
    return result if np.isfinite(result) else None


def _finite_extreme(values: Array, operation: str) -> float | None:
    """Return a finite min/max over finite values, or ``None``."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None
    return float(np.min(finite) if operation == "min" else np.max(finite))


def _time_at(times: Array, index: int | None) -> float | None:
    return None if index is None else float(times[index])


def threshold_label(threshold: float) -> str:
    """Return a stable column-name representation of a threshold."""
    return f"{threshold:g}".replace("-", "m").replace(".", "p")


def _median_or_none(values: list[float]) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if not finite.size else float(np.median(finite))


def _correlation(x: Array, y: Array, mask: Array) -> dict[str, float | int | None]:
    keep = mask & np.isfinite(x) & np.isfinite(y)
    count = int(np.count_nonzero(keep))
    x_kept = x[keep]
    y_kept = y[keep]
    x_constant_tolerance = (
        100.0
        * np.finfo(np.float64).eps
        * max(float(np.max(np.abs(x_kept), initial=0.0)), 1e-30)
    )
    y_constant_tolerance = (
        100.0
        * np.finfo(np.float64).eps
        * max(float(np.max(np.abs(y_kept), initial=0.0)), 1e-30)
    )
    if (
        count < 3
        or np.ptp(x_kept) <= x_constant_tolerance
        or np.ptp(y_kept) <= y_constant_tolerance
    ):
        return {"n": count, "pearson": None, "spearman": None}
    pearson_result: Any = pearsonr(x_kept, y_kept)
    spearman_result: Any = spearmanr(x_kept, y_kept)
    return {
        "n": count,
        "pearson": float(pearson_result.statistic),
        "spearman": float(spearman_result.statistic),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write heterogeneous dictionaries as a rectangular CSV table."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Saved rollout trajectory NPZ.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    parser.add_argument(
        "--length",
        type=float,
        default=None,
        help="Periodic length; defaults to the archive's truth protocol.",
    )
    parser.add_argument(
        "--fields",
        type=parse_field_list,
        default=("eta", "xi", "q"),
        help="Comma-separated fields; eta is required.",
    )
    parser.add_argument(
        "--thresholds",
        type=parse_float_list,
        default=(0.25, 0.5, 0.75, 1.0),
        help="Comma-separated relative-error onset thresholds.",
    )
    parser.add_argument(
        "--translation-dominance-fraction",
        type=float,
        default=0.9,
        help="Squared raw-error fraction that must be removed at raw onset.",
    )
    parser.add_argument(
        "--correlation-times",
        type=parse_float_list,
        default=None,
        help="Optional saved-time targets; default uses 0,10,20,40,60,80,100%% of the horizon.",
    )
    parser.add_argument(
        "--raw-xi-gauge",
        action="store_true",
        help="Do not remove the spatial mean independently from truth/predicted xi.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress framewise alignment progress.",
    )
    parser.add_argument(
        "--focus-simulation-indices",
        type=parse_int_list,
        default=(),
        help="Optional comma-separated archive-local indices copied to a focused table.",
    )
    parser.add_argument(
        "--focus-label",
        default=None,
        help="Description of the preselected focused simulations.",
    )
    args = parser.parse_args()
    archive_path, output_path = args.archive, args.output
    length, fields, thresholds = args.length, args.fields, args.thresholds
    dominance_fraction = args.translation_dominance_fraction
    center_xi = not args.raw_xi_gauge
    focus_simulation_indices = args.focus_simulation_indices
    if not 0.0 <= dominance_fraction <= 1.0:
        raise ValueError(
            f"dominance_fraction must lie in [0, 1], got {dominance_fraction}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(archive_path, allow_pickle=False) as archive:
        times = np.asarray(archive["times"], dtype=np.float64)
        if times.ndim != 1 or not times.size or not np.all(np.isfinite(times)):
            raise ValueError(
                f"times must be a nonempty finite vector, got {times.shape}"
            )
        if times.size > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        periodic_length = (
            float(length)
            if length is not None
            else float(json.loads(str(archive["truth_protocol_json"].item()))["length"])
        )
        if not np.isfinite(periodic_length) or periodic_length <= 0.0:
            raise ValueError(
                f"periodic length must be positive and finite, got {periodic_length}"
            )
        truth_eta = np.asarray(archive["truth_eta"])
        pred_eta = np.asarray(archive["pred_eta"])
        if truth_eta.ndim != 3:
            raise ValueError(
                "truth_eta must have shape (time, simulation, grid), got "
                f"{truth_eta.shape}"
            )
        validate_field_pair(
            truth_eta,
            pred_eta,
            name="eta",
            expected_time_simulations=(times.size, truth_eta.shape[1]),
        )
        n_times, n_simulations, nx = truth_eta.shape
        simulation_ids = np.asarray(archive["simulation_ids"], dtype=np.int64)
        depths = np.asarray(archive["depths"], dtype=np.float64)
        truth_valid = np.asarray(archive["truth_valid"], dtype=bool)
        if simulation_ids.shape != (n_simulations,) or depths.shape != (n_simulations,):
            raise ValueError(
                "simulation_ids and depths must each have shape (simulation,)"
            )
        if truth_valid.shape != (n_simulations,):
            raise ValueError("truth_valid must have shape (simulation,)")

        wrapped, unwrapped, alignment_finite = compute_eta_alignment(
            truth_eta,
            pred_eta,
            periodic_length,
            progress=not args.quiet,
        )
        drift = np.asarray(unwrapped, dtype=np.float64).copy()
        for simulation in range(drift.shape[1]):
            finite_indices = np.flatnonzero(np.isfinite(drift[:, simulation]))
            if finite_indices.size:
                drift[:, simulation] -= drift[finite_indices[0], simulation]
        field_metrics: dict[str, dict[str, Array]] = {}
        velocity_metrics: dict[str, Array] | None = None
        truth_field: Array | None = None
        prediction_field: Array | None = None
        ordered_fields = tuple(field for field in ("eta", "q", "xi") if field in fields)
        for field in ordered_fields:
            truth_key, prediction_key = FIELD_KEYS[field]
            truth_field = (
                truth_eta if field == "eta" else np.asarray(archive[truth_key])
            )
            prediction_field = (
                pred_eta if field == "eta" else np.asarray(archive[prediction_key])
            )
            validate_field_pair(
                truth_field,
                prediction_field,
                name=field,
                expected_time_simulations=(n_times, n_simulations),
            )
            if truth_field.shape[-1] != nx:
                raise ValueError(
                    f"{field} grid has {truth_field.shape[-1]} points, expected {nx}"
                )
            field_metrics[field] = compute_field_metrics(
                truth_field,
                prediction_field,
                wrapped,
                periodic_length,
                center=field == "xi" and center_xi,
            )
            if field == "q":
                velocity_metrics = compute_alignment_velocity_identity(
                    truth_eta,
                    pred_eta,
                    truth_field,
                    prediction_field,
                    wrapped,
                    times,
                    periodic_length,
                )
        del truth_eta, pred_eta, truth_field, prediction_field

    dx = periodic_length / nx
    eta_metrics = field_metrics["eta"]
    simulation_records: list[dict[str, Any]] = []
    for simulation, simulation_id in enumerate(simulation_ids):
        one_grid_index = first_crossing_index(np.abs(drift[:, simulation]) / dx, 1.0)
        nonfinite_indices = np.flatnonzero(~alignment_finite[:, simulation])
        nonfinite_index = int(nonfinite_indices[0]) if nonfinite_indices.size else None
        record: dict[str, Any] = {
            "simulation_index": simulation,
            "simulation_id": int(simulation_id),
            "depth": _finite_float(depths[simulation]),
            "truth_valid": bool(truth_valid[simulation]),
            "first_nonfinite_time": _time_at(times, nonfinite_index),
            "one_grid_displacement_onset_time": _time_at(times, one_grid_index),
            "final_displacement_grid_points": _finite_float(drift[-1, simulation] / dx),
            "max_abs_displacement_grid_points": _finite_extreme(
                np.abs(drift[:, simulation]) / dx, "max"
            ),
        }
        for field, metrics in field_metrics.items():
            for metric in (
                "raw_relative_error",
                "aligned_relative_error",
                "translation_relative_error",
                "signed_squared_error_removed",
            ):
                record[f"final_{field}_{metric}"] = _finite_float(
                    metrics[metric][-1, simulation]
                )
        if velocity_metrics is not None:
            identity_velocity = velocity_metrics["identity_velocity"][:, simulation]
            numerical_velocity = velocity_metrics["numerical_velocity"][:, simulation]
            velocity_correlation = _correlation(
                identity_velocity,
                numerical_velocity,
                np.ones(times.shape, dtype=bool),
            )
            integrated = velocity_metrics["integrated_identity_velocity"][
                -1, simulation
            ]
            record.update(
                {
                    "alignment_velocity_correlation_n": velocity_correlation["n"],
                    "alignment_velocity_pearson": velocity_correlation["pearson"],
                    "alignment_velocity_spearman": velocity_correlation["spearman"],
                    "final_observed_displacement_drift": _finite_float(
                        drift[-1, simulation]
                    ),
                    "final_integrated_identity_velocity": _finite_float(integrated),
                    "final_integrated_direct_velocity": _finite_float(
                        velocity_metrics["integrated_direct_velocity"][-1, simulation]
                    ),
                    "final_integrated_geometry_velocity": _finite_float(
                        velocity_metrics["integrated_geometry_velocity"][-1, simulation]
                    ),
                    "final_velocity_integration_residual": _finite_float(
                        integrated - drift[-1, simulation]
                    ),
                    "final_velocity_integration_residual_grid_points": _finite_float(
                        (integrated - drift[-1, simulation]) / dx
                    ),
                    "max_abs_alignment_stationarity_correction_grid_points": (
                        _finite_extreme(
                            np.abs(
                                velocity_metrics["stationarity_correction"][
                                    :, simulation
                                ]
                            )
                            / dx,
                            "max",
                        )
                    ),
                    "min_abs_velocity_denominator_over_tangent_energy": (
                        _finite_extreme(
                            np.abs(
                                velocity_metrics["denominator_over_tangent_energy"][
                                    :, simulation
                                ]
                            ),
                            "min",
                        )
                    ),
                }
            )
        for threshold in thresholds:
            label = threshold_label(threshold)
            raw_index = first_crossing_index(
                eta_metrics["raw_relative_error"][:, simulation], threshold
            )
            shape_index = first_crossing_index(
                eta_metrics["aligned_relative_error"][:, simulation], threshold
            )
            translation_index = first_crossing_index(
                eta_metrics["translation_relative_error"][:, simulation], threshold
            )
            record[f"eta_raw_gt_{label}_onset_time"] = _time_at(times, raw_index)
            record[f"eta_shape_gt_{label}_onset_time"] = _time_at(times, shape_index)
            record[f"eta_translation_gt_{label}_onset_time"] = _time_at(
                times, translation_index
            )
            record[f"eta_raw_gt_{label}_translation_dominated_at_onset"] = (
                None
                if raw_index is None
                else bool(
                    eta_metrics["signed_squared_error_removed"][raw_index, simulation]
                    >= dominance_fraction
                )
            )
            record[f"one_grid_lead_to_eta_raw_gt_{label}"] = (
                None
                if raw_index is None or one_grid_index is None
                else float(times[raw_index] - times[one_grid_index])
            )
        simulation_records.append(record)

    valid_records = [record for record in simulation_records if record["truth_valid"]]
    onset_table: list[dict[str, Any]] = []
    for threshold in thresholds:
        label = threshold_label(threshold)
        raw_key = f"eta_raw_gt_{label}_onset_time"
        shape_key = f"eta_shape_gt_{label}_onset_time"
        translation_key = f"eta_translation_gt_{label}_onset_time"
        dominated_key = f"eta_raw_gt_{label}_translation_dominated_at_onset"
        lead_key = f"one_grid_lead_to_eta_raw_gt_{label}"
        raw_crossers = [
            record for record in valid_records if record[raw_key] is not None
        ]
        onset_table.append(
            {
                "threshold": threshold,
                "n_truth_valid": len(valid_records),
                "n_raw_crossings": len(raw_crossers),
                "n_shape_crossings": sum(
                    record[shape_key] is not None for record in valid_records
                ),
                "n_translation_crossings": sum(
                    record[translation_key] is not None for record in valid_records
                ),
                "n_raw_crossings_without_shape_crossing": sum(
                    record[shape_key] is None for record in raw_crossers
                ),
                "n_translation_dominated_at_raw_onset": sum(
                    record[dominated_key] is True for record in raw_crossers
                ),
                "median_raw_onset_time": _median_or_none(
                    [record[raw_key] for record in raw_crossers]
                ),
                "median_shape_onset_time": _median_or_none(
                    [
                        record[shape_key]
                        for record in valid_records
                        if record[shape_key] is not None
                    ]
                ),
                "median_translation_onset_time": _median_or_none(
                    [
                        record[translation_key]
                        for record in valid_records
                        if record[translation_key] is not None
                    ]
                ),
                "median_one_grid_lead_to_raw_onset": _median_or_none(
                    [
                        record[lead_key]
                        for record in raw_crossers
                        if record[lead_key] is not None
                    ]
                ),
            }
        )
    correlation_times = args.correlation_times
    if correlation_times is None:
        start, stop = float(times[0]), float(times[-1])
        correlation_times = tuple(
            start + fraction * (stop - start)
            for fraction in (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
        )
    correlation_frames = list(
        dict.fromkeys(
            int(np.argmin(np.abs(times - requested))) for requested in correlation_times
        )
    )
    correlation_targets = {
        "terminal_eta_raw": field_metrics["eta"]["raw_relative_error"][-1],
        "terminal_eta_shape": field_metrics["eta"]["aligned_relative_error"][-1],
        "terminal_eta_translation": field_metrics["eta"]["translation_relative_error"][
            -1
        ],
    }
    correlation_table: list[dict[str, Any]] = []
    for frame in correlation_frames:
        predictors: dict[str, Array] = {
            "abs_displacement_drift_grid_points": np.abs(drift[frame]) / dx,
        }
        for field, metrics in field_metrics.items():
            predictors[f"{field}_raw"] = metrics["raw_relative_error"][frame]
            predictors[f"{field}_aligned"] = metrics["aligned_relative_error"][frame]
            predictors[f"{field}_translation"] = metrics["translation_relative_error"][
                frame
            ]
        if velocity_metrics is not None:
            predictors["abs_alignment_velocity_identity"] = np.abs(
                velocity_metrics["identity_velocity"][frame]
            )
            predictors["abs_alignment_velocity_direct"] = np.abs(
                velocity_metrics["direct_velocity"][frame]
            )
            predictors["abs_alignment_velocity_geometry"] = np.abs(
                velocity_metrics["geometry_velocity"][frame]
            )
        for predictor_name, predictor in predictors.items():
            for target_name, target in correlation_targets.items():
                correlation_table.append(
                    {
                        "frame": frame,
                        "time": float(times[frame]),
                        "predictor": predictor_name,
                        "target": target_name,
                        **_correlation(predictor, target, truth_valid),
                    }
                )

    framewise_path = output_path.with_suffix(".framewise.npz")
    simulations_path = output_path.with_suffix(".simulations.csv")
    onsets_path = output_path.with_suffix(".onsets.csv")
    correlations_path = output_path.with_suffix(".correlations.csv")
    focus_path = output_path.with_suffix(".focus.csv")
    framewise_arrays: dict[str, Array] = {
        "times": times,
        "simulation_ids": simulation_ids,
        "depths": depths,
        "truth_valid": truth_valid,
        "alignment_finite": alignment_finite,
        "displacement_wrapped": wrapped,
        "displacement_unwrapped": unwrapped,
        "displacement_drift": drift,
    }
    for field, metrics in field_metrics.items():
        for metric, values in metrics.items():
            framewise_arrays[f"{field}_{metric}"] = values
    if velocity_metrics is not None:
        for metric, values in velocity_metrics.items():
            framewise_arrays[f"alignment_velocity_{metric}"] = values
    np.savez_compressed(framewise_path, allow_pickle=False, **framewise_arrays)
    write_csv(simulations_path, simulation_records)
    write_csv(onsets_path, onset_table)
    write_csv(correlations_path, correlation_table)

    invalid_focus = [
        index for index in focus_simulation_indices if index >= n_simulations
    ]
    if invalid_focus:
        raise ValueError(
            "focus simulation indices exceed the "
            f"{n_simulations}-simulation archive: {invalid_focus}"
        )
    focus_records = [simulation_records[index] for index in focus_simulation_indices]
    write_csv(focus_path, focus_records)

    alignment_velocity_validation: dict[str, Any] | None = None
    if velocity_metrics is not None:
        frame_mask = np.broadcast_to(truth_valid[None, :], drift.shape)
        pooled = _correlation(
            velocity_metrics["identity_velocity"].reshape(-1),
            velocity_metrics["numerical_velocity"].reshape(-1),
            frame_mask.reshape(-1),
        )
        integrated = velocity_metrics["integrated_identity_velocity"][-1]
        observed = drift[-1]
        closure = integrated - observed
        closure_keep = truth_valid & np.isfinite(closure)
        alignment_velocity_validation = {
            "pooled_identity_vs_numerical_velocity": pooled,
            "terminal_closure_n": int(np.count_nonzero(closure_keep)),
            "terminal_closure_median_abs_grid_points": (
                None
                if not np.any(closure_keep)
                else float(np.median(np.abs(closure[closure_keep])) / dx)
            ),
            "terminal_closure_max_abs_grid_points": (
                None
                if not np.any(closure_keep)
                else float(np.max(np.abs(closure[closure_keep])) / dx)
            ),
        }

    terminal_summary: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for field, metrics in field_metrics.items():
        terminal_summary[field] = {}
        for metric in (
            "raw_relative_error",
            "aligned_relative_error",
            "translation_relative_error",
            "signed_squared_error_removed",
        ):
            selected = metrics[metric][-1, truth_valid]
            selected = selected[np.isfinite(selected)]
            terminal_summary[field][metric] = (
                {
                    "n": int(selected.size),
                    "median": float(np.median(selected)),
                    "p90": float(np.percentile(selected, 90)),
                    "p95": float(np.percentile(selected, 95)),
                    "maximum": float(np.max(selected)),
                }
                if selected.size
                else {"n": 0, "median": None, "p90": None, "p95": None, "maximum": None}
            )

    result: dict[str, Any] = {
        "definition": {
            "translation_operator": "T_d f(x) = f(x-d)",
            "fitted_displacement": (
                "argmin_d ||T_{-d} eta_prediction - eta_truth||_2 over the "
                "continuous periodic orbit"
            ),
            "shape_error": (
                "||T_{-d} prediction - truth||_2 / ||truth||_2, using the "
                "elevation-fitted d for every field"
            ),
            "translation_error": ("sqrt(max(raw_relative_error^2 - shape_error^2, 0))"),
            "xi_gauge": "zero spatial mean" if center_xi else "raw",
            "translation_dominance_fraction": dominance_fraction,
            "threshold_crossing": "first saved frame with error strictly greater than threshold",
            "alignment_velocity_identity": (
                "for a=S_d p, e=a-y, S_d f(x)=f(x+d): d' = "
                "-(<S_d q_p-q_y,a_x>+<e,S_d(q_p)_x>)/"
                "(<a_x,a_x>+<e,a_xx>)"
            ),
        },
        "archive": str(archive_path.resolve()),
        "periodic_length": periodic_length,
        "grid_points": nx,
        "grid_spacing": dx,
        "n_times": n_times,
        "n_simulations": n_simulations,
        "fields": ordered_fields,
        "n_truth_valid": int(np.count_nonzero(truth_valid)),
        "terminal_summary": terminal_summary,
        "alignment_velocity_validation": alignment_velocity_validation,
        "onset_table": onset_table,
        "correlation_table": correlation_table,
        "simulation_records": simulation_records,
        "focus": {
            "label": args.focus_label,
            "simulation_indices": list(focus_simulation_indices),
            "simulation_records": focus_records,
        },
        "outputs": {
            "framewise_npz": str(framewise_path.resolve()),
            "simulations_csv": str(simulations_path.resolve()),
            "onsets_csv": str(onsets_path.resolve()),
            "correlations_csv": str(correlations_path.resolve()),
            "focus_csv": str(focus_path.resolve()),
        },
    }
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "n_simulations": result["n_simulations"],
                "n_truth_valid": result["n_truth_valid"],
                "fields": result["fields"],
                "onset_table": result["onset_table"],
            },
            indent=2,
            allow_nan=False,
        )
    )
