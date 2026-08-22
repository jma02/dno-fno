"""CPU invariants for saved-rollout translation decomposition."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from analyze_neutral_multiarm import periodic_shift
from analyze_rollout_translation_decomposition import (
    analyze_archive,
    compute_alignment_velocity_identity,
    compute_eta_alignment,
    compute_field_metrics,
    spectral_derivative,
)


def _grid(nx: int = 128) -> tuple[float, np.ndarray, np.ndarray]:
    length = 2.0 * np.pi
    x = np.arange(nx, dtype=np.float64) * length / nx
    base = np.cos(3.0 * x) + 0.3 * np.sin(7.0 * x) + 0.1 * np.cos(11.0 * x)
    return length, x, base


def test_rigid_translation_is_removed_exactly() -> None:
    length, _, base = _grid()
    shifts = np.asarray(
        [[0.0, 0.0], [0.05, -0.03], [0.14, -0.11], [0.28, -0.19]],
        dtype=np.float64,
    )
    truth = np.broadcast_to(base, shifts.shape + (base.size,)).copy()
    prediction = np.asarray(
        [
            [periodic_shift(base, shift, length) for shift in frame]
            for frame in shifts
        ]
    )

    wrapped, unwrapped, finite = compute_eta_alignment(truth, prediction, length)
    metrics = compute_field_metrics(
        truth,
        prediction,
        wrapped,
        length,
        center=False,
    )

    np.testing.assert_array_equal(finite, True)
    np.testing.assert_allclose(unwrapped, shifts, rtol=0.0, atol=2e-9)
    np.testing.assert_allclose(
        metrics["aligned_relative_error"], 0.0, rtol=0.0, atol=2e-9
    )
    np.testing.assert_allclose(
        metrics["translation_relative_error"],
        metrics["raw_relative_error"],
        rtol=1e-9,
        atol=1e-12,
    )


def test_pure_shape_error_is_not_mislabeled_translation() -> None:
    length, _, base = _grid()
    truth = base[None, None, :]
    prediction = 1.1 * truth
    wrapped, _, _ = compute_eta_alignment(truth, prediction, length)
    metrics = compute_field_metrics(
        truth,
        prediction,
        wrapped,
        length,
        center=False,
    )

    np.testing.assert_allclose(wrapped, 0.0, rtol=0.0, atol=5e-9)
    np.testing.assert_allclose(
        metrics["aligned_relative_error"],
        metrics["raw_relative_error"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        metrics["translation_relative_error"], 0.0, rtol=0.0, atol=5e-9
    )


def test_exact_alignment_velocity_identity_for_two_travel_speeds() -> None:
    length, _, base = _grid(256)
    times = np.linspace(0.0, 2.0, 21)
    truth_speed = -0.07
    prediction_speed = 0.31
    speed_difference = prediction_speed - truth_speed
    base_x = spectral_derivative(base, length)
    truth_eta = np.asarray(
        [periodic_shift(base, truth_speed * time, length) for time in times]
    )[:, None, :]
    pred_eta = np.asarray(
        [periodic_shift(base, prediction_speed * time, length) for time in times]
    )[:, None, :]
    truth_q = np.asarray(
        [
            -truth_speed * periodic_shift(base_x, truth_speed * time, length)
            for time in times
        ]
    )[:, None, :]
    pred_q = np.asarray(
        [
            -prediction_speed
            * periodic_shift(base_x, prediction_speed * time, length)
            for time in times
        ]
    )[:, None, :]
    wrapped, unwrapped, _ = compute_eta_alignment(truth_eta, pred_eta, length)
    identity = compute_alignment_velocity_identity(
        truth_eta,
        pred_eta,
        truth_q,
        pred_q,
        wrapped,
        times,
        length,
    )

    np.testing.assert_allclose(
        unwrapped[:, 0], speed_difference * times, rtol=0.0, atol=3e-9
    )
    np.testing.assert_allclose(
        identity["identity_velocity"], speed_difference, rtol=0.0, atol=2e-12
    )
    np.testing.assert_allclose(
        identity["direct_velocity"], speed_difference, rtol=0.0, atol=2e-12
    )
    np.testing.assert_allclose(
        identity["geometry_velocity"], 0.0, rtol=0.0, atol=2e-12
    )
    np.testing.assert_allclose(
        identity["integrated_identity_velocity"][:, 0],
        speed_difference * times,
        rtol=0.0,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        identity["integrated_direct_velocity"][:, 0],
        speed_difference * times,
        rtol=0.0,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        identity["integrated_geometry_velocity"], 0.0, rtol=0.0, atol=2e-12
    )


def test_eta_alignment_is_shared_with_xi_and_q() -> None:
    length, x, base = _grid()
    shift = 0.17
    truth_eta = base[None, None, :]
    pred_eta = periodic_shift(base, shift, length)[None, None, :]
    truth_xi = (np.sin(2.0 * x) + 0.2 * np.cos(5.0 * x))[None, None, :]
    pred_xi = (periodic_shift(truth_xi[0, 0], shift, length) + 3.0)[
        None, None, :
    ]
    wrapped, _, _ = compute_eta_alignment(truth_eta, pred_eta, length)
    xi_metrics = compute_field_metrics(
        truth_xi,
        pred_xi,
        wrapped,
        length,
        center=True,
    )

    assert float(xi_metrics["raw_relative_error"][0, 0]) > 0.0
    np.testing.assert_allclose(
        xi_metrics["aligned_relative_error"], 0.0, rtol=0.0, atol=2e-9
    )


def test_archive_outputs_are_json_safe_and_include_onsets() -> None:
    length, x, base = _grid(64)
    times = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    shifts = np.asarray(
        [[0.0, 0.0, 0.0], [0.03, 0.08, 0.0], [0.07, 0.16, 0.0]],
        dtype=np.float64,
    )
    truth_eta = np.broadcast_to(base, shifts.shape + (base.size,)).copy()
    pred_eta = np.asarray(
        [
            [periodic_shift(base, shift, length) for shift in frame]
            for frame in shifts
        ]
    )
    truth_q = np.broadcast_to(np.sin(3.0 * x), truth_eta.shape).copy()
    pred_q = np.asarray(
        [
            [periodic_shift(truth_q[frame, case], shifts[frame, case], length)
             for case in range(shifts.shape[1])]
            for frame in range(shifts.shape[0])
        ]
    )
    truth_xi = np.broadcast_to(np.cos(2.0 * x), truth_eta.shape).copy()
    pred_xi = np.asarray(
        [
            [periodic_shift(truth_xi[frame, case], shifts[frame, case], length)
             for case in range(shifts.shape[1])]
            for frame in range(shifts.shape[0])
        ]
    )
    pred_eta[-1, -1] = np.nan

    with TemporaryDirectory() as directory:
        root = Path(directory)
        archive_path = root / "rollout.npz"
        output_path = root / "analysis.json"
        np.savez_compressed(
            archive_path,
            times=times,
            case_ids=np.asarray([4, 21, 27]),
            depths=np.asarray([0.1, 0.2, 0.3]),
            truth_valid=np.ones(3, dtype=bool),
            truth_eta=truth_eta,
            pred_eta=pred_eta,
            truth_xi=truth_xi,
            pred_xi=pred_xi,
            truth_gxi=truth_q,
            pred_gxi=pred_q,
            length=length,
        )
        result = analyze_archive(
            archive_path,
            output_path,
            length=None,
            fields=("eta", "xi", "q"),
            thresholds=(0.05, 0.25),
            dominance_fraction=0.9,
            correlation_times_requested=None,
            center_xi=True,
            progress=False,
            focus_case_indices=(0, 1),
            focus_label="synthetic focus",
        )
        loaded = json.loads(output_path.read_text(encoding="utf-8"))

        assert result["n_cases"] == 3
        assert loaded["focus"]["label"] == "synthetic focus"
        assert len(loaded["onset_table"]) == 2
        assert loaded["case_records"][2]["first_nonfinite_time"] == 2.0
        for suffix in (
            ".framewise.npz",
            ".cases.csv",
            ".onsets.csv",
            ".correlations.csv",
            ".focus.csv",
        ):
            assert output_path.with_suffix(suffix).is_file()


if __name__ == "__main__":
    tests = (
        test_rigid_translation_is_removed_exactly,
        test_pure_shape_error_is_not_mislabeled_translation,
        test_exact_alignment_velocity_identity_for_two_travel_speeds,
        test_eta_alignment_is_shared_with_xi_and_q,
        test_archive_outputs_are_json_safe_and_include_onsets,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
