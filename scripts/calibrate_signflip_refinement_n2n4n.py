"""Calibrate a fixed-band N/2N/4N replacement for the legacy sign screen.

This is a deliberately small, deterministic CPU panel.  It spans every
frame-zero sign rejection in historical batch 0, clean interior controls, and
a matched periodic reconstruction of the sign-rejected cases.
"""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from audit_signflip_refinement_transition import (
    HISTORICAL_ARCHIVE,
    build_historical,
    projector,
    regenerate_historical_specs,
    relative_h1,
    relative_l2,
)
from solver.gen_data.generate_tanaka_dataset_v2 import (
    build_per_case_initial_conditions,
)
from solver.gen_data.multi_crest import CrestSpec
from solver.solvers.dno_series_jax import dno_series_eval
from solver.solvers.time_integrator import (
    _spectral_truncate_real,
    make_solver_params,
)
from solver.tanaka_ICs.modified_tanaka import make_default_tanaka_template


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notes/signflip_refinement_n2n4n_calibration_20260722.json"
NX_VALUES = (1024, 2048, 4096)
LENGTH = 164.0
DEPTH = 1.0
FIXED_MODE_CUTOFF = 341
TOLERANCES = (3e-4, 1e-3, 3e-3)
PRIMARY_TOLERANCE = 1e-3

SIGN_REJECTED_IDS = (11, 28, 100, 106, 124, 148, 150, 186, 187, 211, 214, 254)
CLEAN_CONTROL_IDS = (26, 43, 76, 225)

FloatArray: TypeAlias = NDArray[np.float64]
Builder: TypeAlias = Callable[[list[list[CrestSpec]], int], tuple[jax.Array, jax.Array]]


@dataclass(frozen=True)
class TripletMetric:
    population: str
    stratum: str
    case_id: int
    l2_defect_n_2n: float
    l2_defect_2n_4n: float
    l2_defect_n_4n: float
    max_l2_defect: float
    l2_contraction_ratio: float
    l2_observed_order: float | None
    h1_defect_n_2n: float
    h1_defect_2n_4n: float
    h1_defect_n_4n: float
    h1_contraction_ratio: float
    h1_observed_order: float | None
    primary_reject: bool


def build_periodic(
    case_specs: list[list[CrestSpec]], nx: int
) -> tuple[jax.Array, jax.Array]:
    template = make_default_tanaka_template(
        depth=DEPTH,
        gravity=1.0,
        direction=1,
        nx=NX_VALUES[0],
        length=LENGTH,
        center=0.0,
        dno_order=6,
        pad_factor=8,
    )
    return build_per_case_initial_conditions(
        template_params=template,
        case_h_ref=jnp.ones(len(case_specs), dtype=jnp.float64),
        case_specs=case_specs,
        length=LENGTH,
        nx=nx,
        gravity=1.0,
    )


def evaluate_q(
    eta: jax.Array,
    xi: jax.Array,
    nx: int,
) -> FloatArray:
    params = make_solver_params(
        nx,
        LENGTH,
        DEPTH,
        dno_order=6,
        pad_factor=8,
        filter_fraction=1.0,
    )
    cutoff = 2.0 * np.pi * FIXED_MODE_CUTOFF / LENGTH
    project = projector(params.k, cutoff)
    q = project(
        dno_series_eval(
            project(eta),
            project(xi),
            params.k,
            DEPTH,
            6,
            pad_factor=8,
        )
    )
    jax.block_until_ready(q)
    return np.asarray(q, dtype=np.float64)


def restrict(field: FloatArray, nx: int) -> FloatArray:
    return np.asarray(_spectral_truncate_real(jnp.asarray(field), nx))


def evaluate_population(
    *,
    population: str,
    case_ids: tuple[int, ...],
    strata: dict[int, str],
    all_specs: list[list[CrestSpec]],
    builder: Builder,
) -> list[TripletMetric]:
    selected_specs = [all_specs[case_id] for case_id in case_ids]
    q_by_n: dict[int, FloatArray] = {}
    for nx in NX_VALUES:
        print(f"building {population} at N={nx}", flush=True)
        eta, xi = builder(selected_specs, nx)
        q_by_n[nx] = evaluate_q(eta, xi, nx)

    q_n = q_by_n[1024]
    q_2n = q_by_n[2048]
    q_4n = q_by_n[4096]
    q_2n_to_n = restrict(q_2n, 1024)
    q_4n_to_2n = restrict(q_4n, 2048)
    q_4n_to_n = restrict(q_4n, 1024)
    mode_n = np.fft.fftfreq(1024) * 1024
    mode_2n = np.fft.fftfreq(2048) * 2048
    physical_k_n = 2.0 * np.pi * mode_n / LENGTH
    physical_k_2n = 2.0 * np.pi * mode_2n / LENGTH

    metrics: list[TripletMetric] = []
    for index, case_id in enumerate(case_ids):
        l2_n_2n = relative_l2(q_n[index], q_2n_to_n[index])
        l2_2n_4n = relative_l2(q_2n[index], q_4n_to_2n[index])
        l2_n_4n = relative_l2(q_n[index], q_4n_to_n[index])
        max_l2_defect = max(l2_n_2n, l2_2n_4n, l2_n_4n)
        l2_ratio = l2_2n_4n / max(l2_n_2n, np.finfo(np.float64).tiny)
        l2_order = -math.log2(l2_ratio) if l2_ratio > 0.0 else None
        h1_n_2n = relative_h1(q_n[index], q_2n_to_n[index], physical_k_n)
        h1_2n_4n = relative_h1(
            q_2n[index], q_4n_to_2n[index], physical_k_2n
        )
        h1_n_4n = relative_h1(q_n[index], q_4n_to_n[index], physical_k_n)
        h1_ratio = h1_2n_4n / max(h1_n_2n, np.finfo(np.float64).tiny)
        h1_order = -math.log2(h1_ratio) if h1_ratio > 0.0 else None
        metrics.append(
            TripletMetric(
                population=population,
                stratum=strata[case_id],
                case_id=case_id,
                l2_defect_n_2n=l2_n_2n,
                l2_defect_2n_4n=l2_2n_4n,
                l2_defect_n_4n=l2_n_4n,
                max_l2_defect=max_l2_defect,
                l2_contraction_ratio=l2_ratio,
                l2_observed_order=l2_order,
                h1_defect_n_2n=h1_n_2n,
                h1_defect_2n_4n=h1_2n_4n,
                h1_defect_n_4n=h1_n_4n,
                h1_contraction_ratio=h1_ratio,
                h1_observed_order=h1_order,
                primary_reject=max_l2_defect > PRIMARY_TOLERANCE,
            )
        )
    return metrics


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        f"q{int(100 * level):02d}": float(np.quantile(array, level))
        for level in (0.0, 0.25, 0.5, 0.75, 1.0)
    }


def summarize(metrics: list[TripletMetric]) -> dict[str, object]:
    strata = sorted({metric.stratum for metric in metrics})
    by_stratum = {}
    for stratum in strata:
        rows = [metric for metric in metrics if metric.stratum == stratum]
        by_stratum[stratum] = {
            "count": len(rows),
            "max_l2_defect": quantiles([row.max_l2_defect for row in rows]),
            "l2_defect_n_2n": quantiles([row.l2_defect_n_2n for row in rows]),
            "l2_defect_2n_4n": quantiles(
                [row.l2_defect_2n_4n for row in rows]
            ),
            "l2_defect_n_4n": quantiles([row.l2_defect_n_4n for row in rows]),
            "l2_contraction_ratio": quantiles(
                [row.l2_contraction_ratio for row in rows]
            ),
            "h1_defect_n_2n": quantiles([row.h1_defect_n_2n for row in rows]),
            "h1_defect_2n_4n": quantiles(
                [row.h1_defect_2n_4n for row in rows]
            ),
            "h1_defect_n_4n": quantiles([row.h1_defect_n_4n for row in rows]),
            "h1_contraction_ratio": quantiles(
                [row.h1_contraction_ratio for row in rows]
            ),
            "decision_sensitivity": {
                f"tau_{tolerance:.0e}": {
                    "tolerance": tolerance,
                    "rejected": sum(
                        row.max_l2_defect > tolerance for row in rows
                    ),
                    "retained": sum(
                        row.max_l2_defect <= tolerance for row in rows
                    ),
                    "rejection_fraction": sum(
                        row.max_l2_defect > tolerance for row in rows
                    )
                    / len(rows),
                }
                for tolerance in TOLERANCES
            },
        }

    return {
        "definition": {
            "grid_sizes": list(NX_VALUES),
            "fixed_mode_cutoff": FIXED_MODE_CUTOFF,
            "relative_defect": (
                "||P_K G6(P_K eta_N,P_K xi_N) - "
                "R P_K G6(P_K eta_M,P_K xi_M)||_2 / ||R q_M||_2"
            ),
            "primary_audit_statistic": (
                "max(defect_N_2N, defect_2N_4N, defect_N_4N)"
            ),
            "tested_tolerances": list(TOLERANCES),
            "primary_tolerance": PRIMARY_TOLERANCE,
            "decision_rule": "reject when max_l2_defect > tolerance",
            "contraction_diagnostic": (
                "rho = defect_2N_4N / defect_N_2N is reported diagnostically; "
                "it is not an acceptance gate"
            ),
        },
        "selection": {
            "sign_rejected_ids": list(SIGN_REJECTED_IDS),
            "clean_control_ids": list(CLEAN_CONTROL_IDS),
            "periodic_counterfactual_ids": list(SIGN_REJECTED_IDS),
            "selection_note": (
                "This is a regime-spanning calibration panel, not a random sample "
                "for estimating population rejection rates."
            ),
        },
        "by_stratum": by_stratum,
        "metrics": [asdict(metric) for metric in metrics],
    }


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    with zipfile.ZipFile(HISTORICAL_ARCHIVE) as archive:
        metadata = json.loads(archive.read("meta.json"))
    all_specs = regenerate_historical_specs(metadata, 0)

    historical_ids = (
        SIGN_REJECTED_IDS
        + CLEAN_CONTROL_IDS
    )
    historical_strata = {
        **dict.fromkeys(SIGN_REJECTED_IDS, "historical_sign_rejected"),
        **dict.fromkeys(CLEAN_CONTROL_IDS, "historical_clean_control"),
    }
    historical = evaluate_population(
        population="historical_zero_extension",
        case_ids=historical_ids,
        strata=historical_strata,
        all_specs=all_specs,
        builder=build_historical,
    )

    periodic = evaluate_population(
        population="corrected_periodic_counterfactual",
        case_ids=SIGN_REJECTED_IDS,
        strata=dict.fromkeys(SIGN_REJECTED_IDS, "corrected_matched_counterfactual"),
        all_specs=all_specs,
        builder=build_periodic,
    )
    summary = summarize(historical + periodic)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
