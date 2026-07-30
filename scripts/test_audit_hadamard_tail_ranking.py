"""Focused tests for held-out Hadamard tail ranking."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_hadamard_tail_ranking import (  # noqa: E402
    partial_rank_summary,
    relative_h1_series,
    select_frame_indices,
    split_half_stability,
    stratified_partial_rank_summary,
    tail_rank_test,
)


def test_select_frame_indices_includes_unique_endpoints() -> None:
    np.testing.assert_array_equal(
        select_frame_indices(251, 5), np.asarray([0, 62, 125, 188, 250])
    )
    np.testing.assert_array_equal(select_frame_indices(3, 7), np.arange(3))
    np.testing.assert_array_equal(select_frame_indices(8, 1), np.asarray([7]))


def test_tail_rank_test_detects_perfect_enrichment() -> None:
    score = np.asarray([9.0, 8.0, 7.0, 2.0, 1.0, 0.0])
    tail = np.asarray([True, True, True, False, False, False])
    result = tail_rank_test(score, tail, tuple(f"case-{i}" for i in range(6)))

    assert result["roc_auc"] == 1.0
    assert result["median_tail_rank"] == 2.0
    assert result["top_k_recall"]["3"]["hits"] == 3
    assert [item["case"] for item in result["tail_cases"]] == [
        "case-0",
        "case-1",
        "case-2",
    ]


def test_split_half_stability_uses_casewise_path_means() -> None:
    base = np.arange(1.0, 6.0)[:, None, None]
    values = np.broadcast_to(base, (5, 3, 4)).copy()
    values[..., 2:] *= 2.0

    result = split_half_stability(values)

    np.testing.assert_allclose(result["rho"], 1.0)


def test_relative_h1_series_matches_single_mode_scaling() -> None:
    nx = 64
    x = 2.0 * np.pi * np.arange(nx) / nx
    target = np.stack((np.cos(x), np.cos(3.0 * x)))
    prediction = 1.05 * target

    np.testing.assert_allclose(relative_h1_series(prediction, target), 0.05)


def test_partial_rank_removes_shared_monotone_covariate() -> None:
    covariate = np.repeat(np.arange(10.0), 2)
    perturbation = np.tile((-0.1, 0.1), 10)
    score = covariate + perturbation
    target = covariate - perturbation

    result = partial_rank_summary(score, target, (covariate,))

    assert result["partial_rho"] < 0.0


def test_stratified_partial_rank_removes_family_specific_control() -> None:
    groups = np.repeat((0, 1), 10)
    control = np.tile(np.arange(10.0), 2)
    perturbation = np.tile((-0.1, 0.1), 10)
    score = control + perturbation
    target = control - perturbation

    result = stratified_partial_rank_summary(score, target, control, groups)

    assert result["partial_rho"] < 0.0
