"""Focused tests for the archived-state DNO audit."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_teacher_forced_gxi import (  # noqa: E402
    CaseTrajectory,
    MetricConfig,
    compute_case_metrics,
    parse_case_requests,
    sign_persistence,
    spectral_derivative,
)


def test_parse_case_requests_preserves_order_and_deduplicates() -> None:
    assert parse_case_requests(
        ("tanaka_g0:4,21,4", "tanaka_g1:11", "tanaka_g1:24,11")
    ) == {"tanaka_g0": [4, 21], "tanaka_g1": [11, 24]}


def test_teacher_forced_rigid_speed_and_phase_rate_are_recovered() -> None:
    nx = 128
    length = 2.0 * np.pi
    x = length * np.arange(nx) / nx
    times = np.linspace(0.0, 2.0, 5)
    eta_row = np.cos(3.0 * x)
    eta = np.repeat(eta_row[None, :], times.size, axis=0)
    eta_x = spectral_derivative(eta, length)
    speed = np.linspace(0.01, 0.03, times.size)
    truth_gxi = np.repeat(np.cos(2.0 * x)[None, :], times.size, axis=0)
    prediction = truth_gxi - speed[:, None] * eta_x
    case = CaseTrajectory(
        regime="synthetic",
        case_index=0,
        case_id=7,
        depth=0.05,
        archive_path=Path("synthetic.npz"),
        truth_protocol_sha256=None,
        times=times,
        truth_eta=eta.astype(np.float32),
        truth_xi=np.zeros_like(eta, dtype=np.float32),
        truth_gxi=truth_gxi.astype(np.float32),
        archived_initial_pred_gxi=prediction[0].astype(np.float32),
    )
    config = MetricConfig(
        length=length,
        low_band_k_max=8.0,
        modal_k_max=8.0,
        modal_denominator_floor_relative=1e-6,
        modal_activity_floor_relative=1e-4,
        eta_activity_floor_relative=1e-4,
        sign_activity_floor_relative=1e-3,
    )

    result = compute_case_metrics(case, prediction.astype(np.float32), config)

    np.testing.assert_allclose(result.tangent_speed, speed, rtol=2e-6)
    np.testing.assert_allclose(result.low_band_tangent_speed, speed, rtol=2e-6)
    np.testing.assert_allclose(result.tangent_energy_fraction, 1.0, atol=1e-12)
    mode_three = result.summary["modal"][2]
    np.testing.assert_allclose(
        result.modal_phase_rate_error[:, 2], -3.0 * speed, rtol=2e-6
    )
    np.testing.assert_allclose(
        mode_three["implied_translation_displacement"],
        np.trapezoid(speed, times),
        rtol=2e-6,
    )
    assert (
        result.summary["translation_tangent"]["sign_persistence"][
            "sign_coherence"
        ]
        == 1.0
    )
    assert result.summary["initial_archived_prediction_parity"]["max_abs"] == 0.0


def test_sign_coherence_detects_complete_cancellation() -> None:
    times = np.arange(5, dtype=np.float64)
    values = np.asarray((1.0, 1.0, 0.0, -1.0, -1.0))
    result = sign_persistence(values, times, activity_floor_relative=1e-3)
    np.testing.assert_allclose(result["sign_coherence"], 0.0, atol=1e-15)
