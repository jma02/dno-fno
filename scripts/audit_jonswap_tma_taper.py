#!/usr/bin/env python3
"""Compare fixed JONSWAP/TMA taper endpoints without time integration."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.gen_data.jonswap_tma import (  # noqa: E402
    PAPER_RESOLVED_BAND_MAXIMUM_WAVENUMBER,
    PAPER_RESOLVED_BAND_QUADRATURE_ORDER,
    PAPER_RESOLVED_BAND_TRANSITION_WAVENUMBER,
    ResolvedBand,
    finite_depth_angular_frequency,
    jonswap_tma_spectrum,
)
from solver.gen_data.jonswap_tma_population import (  # noqa: E402
    JONSWAP_TMA_POPULATION_CELLS,
    JonswapTmaPopulationSample,
    sample_jonswap_tma_population,
)
from solver.gen_data.pipeline.production import (  # noqa: E402
    AttemptAssignment,
    CaseKey,
    SplitId,
)


FloatArray: TypeAlias = NDArray[np.float64]
MetricTable: TypeAlias = dict[str, list[float]]

LENGTH = 2.0 * np.pi
NX = 512
REFERENCE_TRANSITION = 128.0 - 1.0e-6
CURRENT_NAME = "96_128"


@dataclass(frozen=True)
class TaperChoice:
    """One transition and terminal wavenumber pair."""

    name: str
    transition_wavenumber: float
    maximum_wavenumber: float

    @property
    def band(self) -> ResolvedBand:
        """Return the constructor band for this choice."""

        return ResolvedBand(
            length=LENGTH,
            transition_wavenumber=self.transition_wavenumber,
            maximum_wavenumber=self.maximum_wavenumber,
            quadrature_order=16,
        )


CHOICES = (
    TaperChoice("48_64", 48.0, 64.0),
    TaperChoice("64_96", 64.0, 96.0),
    TaperChoice("80_112", 80.0, 112.0),
    TaperChoice(
        CURRENT_NAME,
        PAPER_RESOLVED_BAND_TRANSITION_WAVENUMBER,
        PAPER_RESOLVED_BAND_MAXIMUM_WAVENUMBER,
    ),
)
REFERENCE_BAND = ResolvedBand(
    length=LENGTH,
    transition_wavenumber=REFERENCE_TRANSITION,
    maximum_wavenumber=PAPER_RESOLVED_BAND_MAXIMUM_WAVENUMBER,
    quadrature_order=PAPER_RESOLVED_BAND_QUADRATURE_ORDER,
)
SAMPLING_BAND = CHOICES[-1].band


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases-per-cell",
        type=int,
        default=32,
        help="deterministic cases sampled from each of the 27 population cells",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "notes/jonswap_tma_taper_freeze_20260726.json",
    )
    return parser.parse_args()


def assignment(cell_index: int, replicate: int) -> AttemptAssignment:
    """Return one deterministic population assignment."""

    return AttemptAssignment(
        case_key=CaseKey(
            family_id=4,
            revision_id=1,
            split_id=SplitId.VALIDATION,
            stream_id=260726,
            attempt_index=replicate * len(JONSWAP_TMA_POPULATION_CELLS) + cell_index,
        ),
        cell_id=JONSWAP_TMA_POPULATION_CELLS[cell_index].cell_id,
    )


def padded_energy(
    sample: JonswapTmaPopulationSample,
    choice: TaperChoice,
) -> FloatArray:
    """Return normalized modal energies padded through mode 128."""

    spectrum = jonswap_tma_spectrum(sample.parameters, band=choice.band)
    result = np.zeros(128, dtype=np.float64)
    mode_indices = np.rint(spectrum.wavenumbers).astype(np.int64) - 1
    result[mode_indices] = spectrum.energy_fractions
    return result


def reference_energy(sample: JonswapTmaPopulationSample) -> FloatArray:
    """Return the practically untapered modal law through mode 128."""

    spectrum = jonswap_tma_spectrum(sample.parameters, band=REFERENCE_BAND)
    return np.asarray(spectrum.energy_fractions, dtype=np.float64)


def raw_mass_ratio(
    candidate: FloatArray,
    reference: FloatArray,
    *,
    transition: int,
) -> float:
    """Recover candidate/reference unnormalized mass from an untapered mode."""

    interior = np.arange(1, 129) <= transition
    usable = interior & (reference > np.finfo(np.float64).tiny)
    index = int(np.argmax(np.where(usable, reference, -1.0)))
    return float(reference[index] / candidate[index])


def realized_fields(
    sample: JonswapTmaPopulationSample,
    energy: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return elevation and its derivative on an exactly resolving grid."""

    count = int(np.count_nonzero(energy))
    k = np.arange(1, count + 1, dtype=np.float64)
    fractions = energy[:count]
    variance = (sample.parameters.significant_height / 4.0) ** 2
    right_fraction = sample.parameters.right_moving_fraction
    amplitude_right = np.sqrt(2.0 * right_fraction * variance * fractions)
    amplitude_left = np.sqrt(2.0 * (1.0 - right_fraction) * variance * fractions)
    x = LENGTH * np.arange(NX, dtype=np.float64) / NX
    right_argument = (
        k[:, None] * x[None, :] + sample.phase_right[:count, None]
    )
    left_argument = k[:, None] * x[None, :] + sample.phase_left[:count, None]
    eta = np.sum(
        amplitude_right[:, None] * np.cos(right_argument)
        + amplitude_left[:, None] * np.cos(left_argument),
        axis=0,
    )
    eta_x = -np.sum(
        k[:, None]
        * (
            amplitude_right[:, None] * np.sin(right_argument)
            + amplitude_left[:, None] * np.sin(left_argument)
        ),
        axis=0,
    )
    return eta, eta_x


def append_metrics(
    table: MetricTable,
    *,
    sample: JonswapTmaPopulationSample,
    choice: TaperChoice,
    candidate: FloatArray,
    reference: FloatArray,
    current: FloatArray,
    current_fields: tuple[FloatArray, FloatArray],
) -> None:
    """Append phase-independent and realized-field diagnostics."""

    modes = np.arange(1, 129, dtype=np.float64)
    frequencies = finite_depth_angular_frequency(
        modes,
        depth=sample.parameters.depth,
        gravity=1.0,
    )
    eta, eta_x = realized_fields(sample, candidate)
    current_eta, current_eta_x = current_fields
    table["raw_mass_retained_vs_hard_128"].append(
        raw_mass_ratio(
            candidate,
            reference,
            transition=int(choice.transition_wavenumber),
        )
    )
    table["spectral_total_variation_vs_hard_128"].append(
        float(0.5 * np.sum(np.abs(candidate - reference)))
    )
    table["ensemble_rms_slope_vs_current"].append(
        float(
            np.sqrt(np.sum(modes**2 * candidate))
            / np.sqrt(np.sum(modes**2 * current))
        )
    )
    table["ensemble_rms_linear_rate_vs_current"].append(
        float(
            np.sqrt(np.sum(frequencies**2 * candidate))
            / np.sqrt(np.sum(frequencies**2 * current))
        )
    )
    table["realized_eta_relative_l2_vs_current"].append(
        float(np.linalg.norm(eta - current_eta) / np.linalg.norm(current_eta))
    )
    table["realized_max_slope_vs_current"].append(
        float(np.max(np.abs(eta_x)) / np.max(np.abs(current_eta_x)))
    )
    table["hard_128_mass_above_transition"].append(
        float(np.sum(reference[modes > choice.transition_wavenumber]))
    )
    table["hard_128_mass_above_terminal"].append(
        float(np.sum(reference[modes > choice.maximum_wavenumber]))
    )


def padded_current(sample: JonswapTmaPopulationSample) -> FloatArray:
    """Return the current 96--128 normalized energy law."""

    return padded_energy(sample, CHOICES[-1])


def summarize(values: list[float]) -> dict[str, float]:
    """Return fixed empirical quantiles for one scalar diagnostic."""

    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def run(cases_per_cell: int) -> dict[str, object]:
    """Run the deterministic endpoint comparison."""

    if cases_per_cell <= 0:
        raise ValueError("cases_per_cell must be positive")
    metrics: dict[tuple[str, str], MetricTable] = defaultdict(
        lambda: defaultdict(list)
    )
    minimum_frequency_ratio = np.inf

    for cell_index, cell in enumerate(JONSWAP_TMA_POPULATION_CELLS):
        for replicate in range(cases_per_cell):
            sample = sample_jonswap_tma_population(
                assignment(cell_index, replicate),
                band=SAMPLING_BAND,
            )
            current = padded_current(sample)
            current_fields = realized_fields(sample, current)
            reference = reference_energy(sample)
            peak_frequency = finite_depth_angular_frequency(
                np.asarray([sample.parameters.peak_wavenumber]),
                depth=sample.parameters.depth,
                gravity=1.0,
            )[0]
            transition_frequency = finite_depth_angular_frequency(
                np.asarray([96.0]),
                depth=sample.parameters.depth,
                gravity=1.0,
            )[0]
            minimum_frequency_ratio = min(
                minimum_frequency_ratio,
                float(transition_frequency / peak_frequency),
            )

            for choice in CHOICES:
                candidate = padded_energy(sample, choice)
                append_metrics(
                    metrics[(cell.stratum, choice.name)],
                    sample=sample,
                    choice=choice,
                    candidate=candidate,
                    reference=reference,
                    current=current,
                    current_fields=current_fields,
                )

    by_stratum = {
        stratum: {
            choice.name: {
                metric: summarize(values)
                for metric, values in metrics[(stratum, choice.name)].items()
            }
            for choice in CHOICES
        }
        for stratum in ("shallow", "finite", "deep")
    }
    return {
        "schema": "jonswap_tma_taper_endpoint_audit_v1",
        "cpu_only": True,
        "time_integration": False,
        "cases_per_cell": cases_per_cell,
        "number_of_cells": len(JONSWAP_TMA_POPULATION_CELLS),
        "number_of_cases": cases_per_cell * len(JONSWAP_TMA_POPULATION_CELLS),
        "domain_length": LENGTH,
        "grid_points_for_realized_fields": NX,
        "practically_untapered_reference": {
            "transition_wavenumber": REFERENCE_TRANSITION,
            "maximum_wavenumber": 128.0,
        },
        "choices": [
            {
                "name": choice.name,
                "transition_wavenumber": choice.transition_wavenumber,
                "maximum_wavenumber": choice.maximum_wavenumber,
            }
            for choice in CHOICES
        ],
        "support_argument": {
            "maximum_peak_wavenumber": 24.0,
            "current_transition_over_maximum_peak": 4.0,
            "analytic_minimum_frequency_ratio": 2.0,
            "sampled_minimum_frequency_ratio_at_current_transition": (
                minimum_frequency_ratio
            ),
            "explanation": (
                "For r>=1, omega(r*k_p,h)/omega(k_p,h) "
                ">=sqrt(r). Hence K0=4*max(k_p)=96 leaves every frequency "
                "through 2*omega_p untapered over the declared support."
            ),
        },
        "by_stratum": by_stratum,
    }


def main() -> None:
    """Run the audit and write strict JSON."""

    args = parse_args()
    result = run(args.cases_per_cell)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
