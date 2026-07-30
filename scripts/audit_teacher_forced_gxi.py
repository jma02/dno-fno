"""Audit a checkpoint's DNO action on archived truth trajectories.

For every selected saved state ``z_* = (eta_*, xi_*)``, this script evaluates
``q_theta(z_*)`` from the checkpoint and compares it with the archived
order-six reference ``q_* = truth_gxi``.  It does not advance either state.
Predictions are evaluated in fixed, padded batches so the final partial batch
does not trigger another JAX compilation or a larger transient allocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver.evals.model_rollout import (  # noqa: E402
    LoadedRun,
    build_predict_gxi_batched,
    load_run,
)


Predictor = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


@dataclass(frozen=True)
class CaseTrajectory:
    """One archived truth trajectory and its fixed physical metadata."""

    regime: str
    case_index: int
    case_id: int
    depth: float
    archive_path: Path
    truth_protocol_sha256: str | None
    times: np.ndarray
    truth_eta: np.ndarray
    truth_xi: np.ndarray
    truth_gxi: np.ndarray
    archived_initial_pred_gxi: np.ndarray | None


@dataclass(frozen=True)
class MetricConfig:
    """Spectral bands and numerical activity floors for the audit."""

    length: float
    low_band_k_max: float
    modal_k_max: float
    modal_denominator_floor_relative: float
    modal_activity_floor_relative: float
    eta_activity_floor_relative: float
    sign_activity_floor_relative: float


@dataclass(frozen=True)
class CaseMetrics:
    """Framewise arrays and strict-JSON summary for one case."""

    summary: dict[str, Any]
    relative_l2: np.ndarray
    low_band_relative_l2: np.ndarray
    tangent_speed: np.ndarray
    tangent_energy_fraction: np.ndarray
    low_band_tangent_speed: np.ndarray
    low_band_tangent_energy_fraction: np.ndarray
    modal_relative_error: np.ndarray
    modal_phase_rate_error: np.ndarray
    modal_active: np.ndarray
    modal_reference_scale: np.ndarray
    modal_error_amplitude: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--eval-dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--case",
        action="append",
        default=None,
        metavar="REGIME:INDEX[,INDEX...]",
        help="Select positional case indices; repeat for multiple regimes.",
    )
    selection.add_argument(
        "--all-cases",
        action="store_true",
        help="Audit every case in every <regime>_trajs.npz below --eval-dir.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--length", type=float, default=2.0 * float(np.pi))
    parser.add_argument("--low-band-k-max", type=float, default=8.0)
    parser.add_argument("--modal-k-max", type=float, default=8.0)
    parser.add_argument(
        "--modal-denominator-floor-relative", type=float, default=1e-6
    )
    parser.add_argument(
        "--modal-activity-floor-relative", type=float, default=1e-4
    )
    parser.add_argument(
        "--eta-activity-floor-relative", type=float, default=1e-4
    )
    parser.add_argument(
        "--sign-activity-floor-relative", type=float, default=1e-3
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--series-output",
        type=Path,
        default=None,
        help="Defaults to --output with a .npz suffix.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Include q_theta(z_truth) in the series NPZ.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch size": args.batch_size,
        "length": args.length,
        "low-band cutoff": args.low_band_k_max,
        "modal cutoff": args.modal_k_max,
        "modal denominator floor": args.modal_denominator_floor_relative,
        "modal activity floor": args.modal_activity_floor_relative,
        "eta activity floor": args.eta_activity_floor_relative,
        "sign activity floor": args.sign_activity_floor_relative,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"all batch, domain, cutoff, and floor values must be positive: {invalid}")


def directory_sha256(path: Path) -> str:
    """Hash relative paths and contents of every regular checkpoint file."""
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"cannot hash empty checkpoint directory: {path}")
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        with candidate.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def parse_case_requests(tokens: Sequence[str]) -> dict[str, list[int]]:
    """Parse and deduplicate ``REGIME:INDEX[,INDEX...]`` selections."""
    requests: dict[str, list[int]] = {}
    for token in tokens:
        regime, separator, index_text = token.partition(":")
        if not separator or not regime or not index_text:
            raise ValueError(f"invalid --case {token!r}; expected REGIME:INDEX[,INDEX...]")
        try:
            indices = [int(value) for value in index_text.split(",")]
        except ValueError as error:
            raise ValueError(f"invalid integer in --case {token!r}") from error
        if not indices or any(index < 0 for index in indices):
            raise ValueError(f"case indices must be nonnegative in --case {token!r}")
        selected = requests.setdefault(regime, [])
        selected.extend(index for index in indices if index not in selected)
    return requests


def discover_all_requests(eval_dir: Path) -> dict[str, list[int] | None]:
    """Discover every eval-suite archive, deferring its case count to loading."""
    archives = sorted(eval_dir.glob("*/*_trajs.npz"))
    if not archives:
        raise FileNotFoundError(f"no */*_trajs.npz archives below {eval_dir}")
    requests: dict[str, list[int] | None] = {}
    for archive_path in archives:
        regime = archive_path.name.removesuffix("_trajs.npz")
        if regime in requests:
            raise ValueError(f"multiple archives found for regime {regime!r}")
        requests[regime] = None
    return requests


def _required_archive_fields() -> tuple[str, ...]:
    return (
        "times",
        "depths",
        "case_ids",
        "truth_eta",
        "truth_xi",
        "truth_gxi",
    )


def load_cases(
    eval_dir: Path,
    requests: dict[str, list[int] | None],
) -> list[CaseTrajectory]:
    """Load selected truth paths, grouping reads by compressed archive."""
    cases: list[CaseTrajectory] = []
    for regime, requested_indices in requests.items():
        archive_path = eval_dir / regime / f"{regime}_trajs.npz"
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        with np.load(archive_path, allow_pickle=False) as archive:
            missing = set(_required_archive_fields()) - set(archive.files)
            if missing:
                raise KeyError(f"{archive_path}: missing fields {sorted(missing)}")
            times = np.asarray(archive["times"], dtype=np.float64)
            depths = np.asarray(archive["depths"], dtype=np.float64)
            case_ids = np.asarray(archive["case_ids"], dtype=np.int64)
            truth_eta = np.asarray(archive["truth_eta"], dtype=np.float32)
            truth_xi = np.asarray(archive["truth_xi"], dtype=np.float32)
            truth_gxi = np.asarray(archive["truth_gxi"], dtype=np.float32)
            archived_pred_gxi = (
                np.asarray(archive["pred_gxi"][0], dtype=np.float32)
                if "pred_gxi" in archive.files
                else None
            )
            protocol_sha = (
                str(np.asarray(archive["truth_protocol_sha256"]).item())
                if "truth_protocol_sha256" in archive.files
                else None
            )
        expected_shape = truth_eta.shape
        if truth_xi.shape != expected_shape or truth_gxi.shape != expected_shape:
            raise ValueError(
                f"{archive_path}: truth fields have inconsistent shapes "
                f"{truth_eta.shape}, {truth_xi.shape}, {truth_gxi.shape}"
            )
        if expected_shape[:2] != (times.size, depths.size):
            raise ValueError(
                f"{archive_path}: trajectory shape {expected_shape} disagrees with "
                f"times/depths {(times.size, depths.size)}"
            )
        indices = (
            list(range(depths.size))
            if requested_indices is None
            else requested_indices
        )
        invalid = [index for index in indices if index >= depths.size]
        if invalid:
            raise IndexError(
                f"{archive_path}: indices {invalid} outside [0, {depths.size})"
            )
        cases.extend(
            CaseTrajectory(
                regime=regime,
                case_index=index,
                case_id=int(case_ids[index]),
                depth=float(depths[index]),
                archive_path=archive_path,
                truth_protocol_sha256=protocol_sha,
                times=times.copy(),
                truth_eta=truth_eta[:, index].copy(),
                truth_xi=truth_xi[:, index].copy(),
                truth_gxi=truth_gxi[:, index].copy(),
                archived_initial_pred_gxi=(
                    None
                    if archived_pred_gxi is None
                    else archived_pred_gxi[index].copy()
                ),
            )
            for index in indices
        )
    if not cases:
        raise ValueError("no cases selected")
    nx_values = {case.truth_eta.shape[-1] for case in cases}
    if len(nx_values) != 1:
        raise ValueError(f"selected cases use different spatial grids: {nx_values}")
    return cases


def predict_truth_path(
    predictor: Predictor,
    case: CaseTrajectory,
    batch_size: int,
) -> np.ndarray:
    """Evaluate one truth path in identically shaped, padded JAX batches."""
    n_frames, nx = case.truth_eta.shape
    result = np.empty((n_frames, nx), dtype=np.float32)
    log_depth = np.float32(np.log(max(case.depth, 1e-12)))
    for start in range(0, n_frames, batch_size):
        stop = min(start + batch_size, n_frames)
        count = stop - start
        eta_batch = np.empty((batch_size, nx), dtype=np.float32)
        xi_batch = np.empty((batch_size, nx), dtype=np.float32)
        eta_batch[:count] = case.truth_eta[start:stop]
        xi_batch[:count] = case.truth_xi[start:stop]
        if count < batch_size:
            eta_batch[count:] = eta_batch[count - 1]
            xi_batch[count:] = xi_batch[count - 1]
        predicted = predictor(
            jnp.asarray(eta_batch),
            jnp.asarray(xi_batch),
            jnp.full((batch_size,), log_depth, dtype=jnp.float32),
        )
        result[start:stop] = np.asarray(jax.device_get(predicted))[:count]
    return result


def spectral_derivative(values: np.ndarray, length: float) -> np.ndarray:
    """Return the periodic derivative along the last axis."""
    nx = values.shape[-1]
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    return np.fft.ifft(
        1j * wave_numbers[None, :] * np.fft.fft(values, axis=-1),
        axis=-1,
    ).real


def low_pass(values: np.ndarray, length: float, k_max: float) -> np.ndarray:
    """Project onto nonzero Fourier modes with ``|k| <= k_max``."""
    nx = values.shape[-1]
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    band = (np.abs(wave_numbers) > 0.0) & (np.abs(wave_numbers) <= k_max)
    return np.fft.ifft(
        np.fft.fft(values, axis=-1) * band[None, :], axis=-1
    ).real


def relative_l2_series(error: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return framewise relative Euclidean error."""
    numerator = np.linalg.norm(error, axis=-1)
    denominator = np.linalg.norm(reference, axis=-1)
    return numerator / np.maximum(denominator, np.finfo(np.float64).tiny)


def tangent_projection(
    error: np.ndarray,
    eta_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return translation-speed defect and fraction of defect energy projected."""
    tangent_energy = np.sum(eta_x**2, axis=-1)
    cross = np.sum(error * eta_x, axis=-1)
    speed = -cross / np.maximum(tangent_energy, np.finfo(np.float64).tiny)
    component_energy = speed**2 * tangent_energy
    error_energy = np.sum(error**2, axis=-1)
    fraction = component_energy / np.maximum(
        error_energy, np.finfo(np.float64).tiny
    )
    return speed, np.minimum(fraction, 1.0)


def finite_stats(values: np.ndarray, times: np.ndarray) -> dict[str, Any]:
    """Summarize a framewise scalar without admitting nonfinite JSON values."""
    finite = np.isfinite(values)
    if not np.any(finite):
        return {key: None for key in ("initial", "final", "median", "p95", "maximum", "time_of_maximum")}
    finite_values = values[finite]
    finite_times = times[finite]
    maximum_index = int(np.argmax(finite_values))
    return {
        "initial": float(values[0]) if np.isfinite(values[0]) else None,
        "final": float(values[-1]) if np.isfinite(values[-1]) else None,
        "median": float(np.median(finite_values)),
        "p95": float(np.percentile(finite_values, 95)),
        "maximum": float(finite_values[maximum_index]),
        "time_of_maximum": float(finite_times[maximum_index]),
    }


def sign_persistence(
    values: np.ndarray,
    times: np.ndarray,
    activity_floor_relative: float,
) -> dict[str, Any]:
    """Measure sign cancellation after excluding numerically negligible frames."""
    finite = np.isfinite(values)
    maximum = float(np.max(np.abs(values[finite]))) if np.any(finite) else 0.0
    active = finite & (np.abs(values) > activity_floor_relative * maximum)
    signed_integral = float(np.trapezoid(np.where(finite, values, 0.0), times))
    absolute_integral = float(
        np.trapezoid(np.where(finite, np.abs(values), 0.0), times)
    )
    dominant_sign = int(np.sign(signed_integral))
    active_signs = np.sign(values[active]).astype(np.int8)
    sign_changes = (
        int(np.count_nonzero(active_signs[1:] != active_signs[:-1]))
        if active_signs.size > 1
        else 0
    )
    agreeing = (
        float(np.mean(active_signs == dominant_sign))
        if active_signs.size and dominant_sign
        else None
    )
    return {
        "activity_floor_relative_to_max_abs": activity_floor_relative,
        "active_frames": int(active_signs.size),
        "active_fraction": float(np.mean(active)),
        "dominant_sign": dominant_sign,
        "fraction_active_frames_with_dominant_sign": agreeing,
        "sign_changes_across_active_frames": sign_changes,
        "signed_time_integral": signed_integral,
        "absolute_time_integral": absolute_integral,
        "sign_coherence": (
            abs(signed_integral) / absolute_integral
            if absolute_integral > 0.0
            else None
        ),
    }


def compute_case_metrics(
    case: CaseTrajectory,
    teacher_prediction: np.ndarray,
    config: MetricConfig,
) -> CaseMetrics:
    """Compute field, modal, and translation-tangent diagnostics."""
    eta = np.asarray(case.truth_eta, dtype=np.float64)
    truth = np.asarray(case.truth_gxi, dtype=np.float64)
    prediction = np.asarray(teacher_prediction, dtype=np.float64)
    error = prediction - truth
    relative_l2 = relative_l2_series(error, truth)

    eta_x = spectral_derivative(eta, config.length)
    tangent_speed, tangent_fraction = tangent_projection(error, eta_x)
    low_error = low_pass(error, config.length, config.low_band_k_max)
    low_truth = low_pass(truth, config.length, config.low_band_k_max)
    low_eta_x = low_pass(eta_x, config.length, config.low_band_k_max)
    low_relative_l2 = relative_l2_series(low_error, low_truth)
    low_speed, low_fraction = tangent_projection(low_error, low_eta_x)

    nx = eta.shape[-1]
    k_rfft = 2.0 * np.pi * np.fft.rfftfreq(nx, d=config.length / nx)
    modal_indices = np.flatnonzero(
        (k_rfft > 0.0) & (k_rfft <= config.modal_k_max)
    )
    if not modal_indices.size:
        raise ValueError(
            f"no positive Fourier modes below modal cutoff {config.modal_k_max}"
        )
    eta_hat = np.fft.rfft(eta, axis=-1, norm="forward")[:, modal_indices]
    truth_hat = np.fft.rfft(truth, axis=-1, norm="forward")[:, modal_indices]
    error_hat = np.fft.rfft(error, axis=-1, norm="forward")[:, modal_indices]
    modal_k = k_rfft[modal_indices]
    omega_squared = modal_k[None, :] * np.tanh(case.depth * modal_k[None, :])
    physical_scale_squared = (
        np.abs(truth_hat) ** 2 + omega_squared * np.abs(eta_hat) ** 2
    )
    sample_scale_squared = np.max(physical_scale_squared, axis=-1, keepdims=True)
    denominator_floor = (
        config.modal_denominator_floor_relative * sample_scale_squared
        + np.finfo(np.float64).tiny
    )
    modal_reference_scale = np.sqrt(physical_scale_squared + denominator_floor)
    modal_error_amplitude = np.abs(error_hat)
    modal_relative_error = modal_error_amplitude / modal_reference_scale
    activity_floor = (
        config.modal_activity_floor_relative * sample_scale_squared
        + np.finfo(np.float64).tiny
    )
    modal_active = physical_scale_squared > activity_floor

    eta_amplitude = np.abs(eta_hat)
    eta_floor = (
        config.eta_activity_floor_relative
        * np.max(eta_amplitude, axis=0, keepdims=True)
        + np.finfo(np.float64).tiny
    )
    eta_active = eta_amplitude > eta_floor
    modal_phase_rate = np.full(error_hat.shape, np.nan, dtype=np.float64)
    modal_phase_rate[eta_active] = np.imag(
        error_hat[eta_active] / eta_hat[eta_active]
    )

    modal_summaries: list[dict[str, Any]] = []
    for column, (mode_index, wave_number) in enumerate(
        zip(modal_indices, modal_k, strict=True)
    ):
        active_relative = np.where(
            modal_active[:, column], modal_relative_error[:, column], np.nan
        )
        phase_values = modal_phase_rate[:, column]
        phase_sign = sign_persistence(
            phase_values, case.times, config.sign_activity_floor_relative
        )
        modal_summaries.append(
            {
                "rfft_index": int(mode_index),
                "wave_number": float(wave_number),
                "physical_scale_active_fraction": float(
                    np.mean(modal_active[:, column])
                ),
                "eta_phase_rate_active_fraction": float(
                    np.mean(eta_active[:, column])
                ),
                "q_error_over_physical_scale": finite_stats(
                    active_relative, case.times
                ),
                "phase_rate_error": finite_stats(phase_values, case.times),
                "phase_rate_sign_persistence": phase_sign,
                "integrated_phase_rate_error": phase_sign[
                    "signed_time_integral"
                ],
                "implied_translation_displacement": (
                    -phase_sign["signed_time_integral"] / wave_number
                ),
            }
        )

    parity: dict[str, float] | None = None
    if case.archived_initial_pred_gxi is not None:
        parity_error = prediction[0] - case.archived_initial_pred_gxi
        parity = {
            "max_abs": float(np.max(np.abs(parity_error))),
            "relative_l2": float(
                np.linalg.norm(parity_error)
                / max(
                    np.linalg.norm(case.archived_initial_pred_gxi),
                    np.finfo(np.float64).tiny,
                )
            ),
        }

    summary = {
        "regime": case.regime,
        "case_index": case.case_index,
        "case_id": case.case_id,
        "depth": case.depth,
        "archive": str(case.archive_path),
        "truth_protocol_sha256": case.truth_protocol_sha256,
        "n_frames": int(case.times.size),
        "time_start": float(case.times[0]),
        "time_final": float(case.times[-1]),
        "initial_archived_prediction_parity": parity,
        "q_relative_l2": finite_stats(relative_l2, case.times),
        "q_low_band_relative_l2": finite_stats(low_relative_l2, case.times),
        "translation_tangent": {
            "speed_defect": finite_stats(tangent_speed, case.times),
            "energy_fraction": finite_stats(tangent_fraction, case.times),
            "sign_persistence": sign_persistence(
                tangent_speed,
                case.times,
                config.sign_activity_floor_relative,
            ),
        },
        "low_band_translation_tangent": {
            "speed_defect": finite_stats(low_speed, case.times),
            "energy_fraction": finite_stats(low_fraction, case.times),
            "sign_persistence": sign_persistence(
                low_speed,
                case.times,
                config.sign_activity_floor_relative,
            ),
        },
        "modal": modal_summaries,
    }
    return CaseMetrics(
        summary=summary,
        relative_l2=relative_l2,
        low_band_relative_l2=low_relative_l2,
        tangent_speed=tangent_speed,
        tangent_energy_fraction=tangent_fraction,
        low_band_tangent_speed=low_speed,
        low_band_tangent_energy_fraction=low_fraction,
        modal_relative_error=modal_relative_error,
        modal_phase_rate_error=modal_phase_rate,
        modal_active=modal_active,
        modal_reference_scale=modal_reference_scale,
        modal_error_amplitude=modal_error_amplitude,
    )


def series_payload(
    cases: Sequence[CaseTrajectory],
    metrics: Sequence[CaseMetrics],
    predictions: Sequence[np.ndarray],
    config: MetricConfig,
    save_predictions: bool,
) -> dict[str, np.ndarray]:
    """Pack variable-length case series using a flat frame axis and offsets."""
    offsets = np.concatenate(
        ([0], np.cumsum([case.times.size for case in cases], dtype=np.int64))
    )
    nx = cases[0].truth_eta.shape[-1]
    k_rfft = 2.0 * np.pi * np.fft.rfftfreq(nx, d=config.length / nx)
    modal_indices = np.flatnonzero(
        (k_rfft > 0.0) & (k_rfft <= config.modal_k_max)
    )
    payload = {
        "case_regimes": np.asarray([case.regime for case in cases]),
        "case_indices": np.asarray([case.case_index for case in cases], np.int64),
        "case_ids": np.asarray([case.case_id for case in cases], np.int64),
        "depths": np.asarray([case.depth for case in cases], np.float64),
        "case_offsets": offsets,
        "times": np.concatenate([case.times for case in cases]),
        "modal_rfft_indices": modal_indices.astype(np.int64),
        "modal_wave_numbers": k_rfft[modal_indices].astype(np.float64),
        "q_relative_l2": np.concatenate(
            [item.relative_l2 for item in metrics]
        ),
        "q_low_band_relative_l2": np.concatenate(
            [item.low_band_relative_l2 for item in metrics]
        ),
        "tangent_speed_defect": np.concatenate(
            [item.tangent_speed for item in metrics]
        ),
        "tangent_energy_fraction": np.concatenate(
            [item.tangent_energy_fraction for item in metrics]
        ),
        "low_band_tangent_speed_defect": np.concatenate(
            [item.low_band_tangent_speed for item in metrics]
        ),
        "low_band_tangent_energy_fraction": np.concatenate(
            [item.low_band_tangent_energy_fraction for item in metrics]
        ),
        "modal_q_relative_error": np.concatenate(
            [item.modal_relative_error for item in metrics], axis=0
        ),
        "modal_phase_rate_error": np.concatenate(
            [item.modal_phase_rate_error for item in metrics], axis=0
        ),
        "modal_active": np.concatenate(
            [item.modal_active for item in metrics], axis=0
        ),
        "modal_reference_scale": np.concatenate(
            [item.modal_reference_scale for item in metrics], axis=0
        ),
        "modal_error_amplitude": np.concatenate(
            [item.modal_error_amplitude for item in metrics], axis=0
        ),
    }
    if save_predictions:
        payload["teacher_q_prediction"] = np.concatenate(predictions, axis=0)
    return payload


def provenance(
    args: argparse.Namespace,
    loaded: LoadedRun,
    cases: Sequence[CaseTrajectory],
    series_output: Path,
    config: MetricConfig,
) -> dict[str, Any]:
    """Return enough execution metadata to reproduce the audit."""
    checkpoint_dir = args.run_dir.resolve() / (
        "best_val_ckpt" if args.checkpoint == "best" else "final_ckpt"
    )
    return {
        "schema_version": 1,
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint_selection": args.checkpoint,
        "checkpoint_epoch": loaded.epoch,
        "checkpoint_path": str(checkpoint_dir),
        "checkpoint_sha256": directory_sha256(checkpoint_dir),
        "eval_dir": str(args.eval_dir.resolve()),
        "series_output": str(series_output.resolve()),
        "argv": sys.argv,
        "execution_environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "JAX_PLATFORMS",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
            )
        },
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "batch_size": args.batch_size,
        "n_cases": len(cases),
        "n_saved_states": int(sum(case.times.size for case in cases)),
        "definitions": {
            "q_defect": "q_theta(eta_truth, xi_truth; h) - truth_gxi",
            "relative_l2": "||q_defect||_2 / ||truth_gxi||_2",
            "low_band": (
                f"nonzero Fourier modes with |k| <= {config.low_band_k_max}"
            ),
            "translation_speed_defect": (
                "-<q_defect, d_x eta_truth> / ||d_x eta_truth||_2^2"
            ),
            "modal_physical_scale_squared": (
                "|truth_gxi_hat_k|^2 + |k| tanh(|k|h) |eta_truth_hat_k|^2"
            ),
            "modal_phase_rate_error": (
                "Im(q_defect_hat_k / eta_truth_hat_k) on eta-active frames"
            ),
            "sign_coherence": (
                "|integral a(t)dt| / integral |a(t)|dt; 1 means no cancellation"
            ),
        },
        "metric_config": {
            "length": config.length,
            "low_band_k_max": config.low_band_k_max,
            "modal_k_max": config.modal_k_max,
            "modal_denominator_floor_relative": (
                config.modal_denominator_floor_relative
            ),
            "modal_activity_floor_relative": config.modal_activity_floor_relative,
            "eta_activity_floor_relative": config.eta_activity_floor_relative,
            "sign_activity_floor_relative": config.sign_activity_floor_relative,
        },
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    eval_dir = args.eval_dir.resolve()
    requests: dict[str, list[int] | None] = (
        discover_all_requests(eval_dir)
        if args.all_cases
        else parse_case_requests(args.case or [])
    )
    cases = load_cases(eval_dir, requests)
    config = MetricConfig(
        length=args.length,
        low_band_k_max=args.low_band_k_max,
        modal_k_max=args.modal_k_max,
        modal_denominator_floor_relative=args.modal_denominator_floor_relative,
        modal_activity_floor_relative=args.modal_activity_floor_relative,
        eta_activity_floor_relative=args.eta_activity_floor_relative,
        sign_activity_floor_relative=args.sign_activity_floor_relative,
    )

    print(
        f"loading {args.checkpoint} checkpoint from {args.run_dir.resolve()}",
        flush=True,
    )
    loaded = load_run(args.run_dir, checkpoint=args.checkpoint)
    predictor = build_predict_gxi_batched(loaded)
    print(
        f"checkpoint epoch={loaded.epoch}; backend={jax.default_backend()}; "
        f"devices={[str(device) for device in jax.devices()]}",
        flush=True,
    )

    predictions: list[np.ndarray] = []
    metrics: list[CaseMetrics] = []
    for position, case in enumerate(cases, start=1):
        print(
            f"[{position}/{len(cases)}] {case.regime} index={case.case_index} "
            f"case_id={case.case_id} h={case.depth:.8g}",
            flush=True,
        )
        prediction = predict_truth_path(predictor, case, args.batch_size)
        predictions.append(prediction)
        metrics.append(compute_case_metrics(case, prediction, config))

    output = args.output.resolve()
    series_output = (
        args.series_output.resolve()
        if args.series_output is not None
        else output.with_suffix(".npz")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    series_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        series_output,
        **series_payload(
            cases, metrics, predictions, config, args.save_predictions
        ),
    )
    result = {
        "provenance": provenance(
            args, loaded, cases, series_output, config
        ),
        "cases": [item.summary for item in metrics],
    }
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"summary -> {output}", flush=True)
    print(f"series  -> {series_output}", flush=True)


if __name__ == "__main__":
    main()
