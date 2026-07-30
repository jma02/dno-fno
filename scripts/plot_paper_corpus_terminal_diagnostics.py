"""Plot rough and sign-changing completed endpoints from the rollout panel.

These quantities are descriptive diagnostics.  They do not participate in
the corpus acceptance decision, and incomplete trajectories are excluded
before ranking.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import TypeAlias

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float64]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs/full_horizon_refinement_panel_20260725"
DEFAULT_OUTPUT = ROOT / "outputs/paper_corpus_terminal_diagnostics_20260726"
RELATIVE_DIFFERENCE_DEAD_ZONE = 0.03
TOP_CASE_COUNT = 6


@dataclass(frozen=True)
class TerminalCase:
    """Initial and terminal diagnostic data for one rollout case."""

    case_id: str
    family: str
    depth: float
    endpoint_time: float
    completed: bool
    eta_initial: FloatArray
    eta_endpoint: FloatArray
    gxi_initial: FloatArray
    gxi_endpoint: FloatArray
    eta_effective_wavenumber: float
    gxi_effective_wavenumber: float
    gxi_sign_changes: int
    eta_high_band_energy_fraction: float
    gxi_high_band_energy_fraction: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-count", type=int, default=TOP_CASE_COUNT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rfft_energy(field: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return nonnegative wavenumbers and Parseval-weighted modal energies."""

    centered = np.asarray(field, dtype=np.float64) - float(np.mean(field))
    coefficients = np.fft.rfft(centered)
    weights = np.full(coefficients.size, 2.0, dtype=np.float64)
    weights[0] = 1.0
    if centered.size % 2 == 0:
        weights[-1] = 1.0
    wavenumbers = np.arange(coefficients.size, dtype=np.float64)
    energies = weights * np.abs(coefficients) ** 2
    return wavenumbers, np.asarray(energies, dtype=np.float64)


def effective_wavenumber(field: FloatArray) -> float:
    """Return the root-mean-square wavenumber of a nonconstant field."""

    wavenumbers, energies = rfft_energy(field)
    denominator = float(np.sum(energies[1:]))
    if denominator <= 0.0:
        return 0.0
    numerator = float(np.sum(wavenumbers[1:] ** 2 * energies[1:]))
    return float(np.sqrt(numerator / denominator))


def high_band_energy_fraction(
    field: FloatArray,
    *,
    lower_wavenumber: int = 64,
) -> float:
    """Return the energy fraction at wavenumbers at least ``lower_wavenumber``."""

    wavenumbers, energies = rfft_energy(field)
    denominator = float(np.sum(energies[1:]))
    if denominator <= 0.0:
        return 0.0
    return float(
        np.sum(energies[wavenumbers >= lower_wavenumber]) / denominator
    )


def cyclic_sign_changes(
    field: FloatArray,
    *,
    relative_dead_zone: float = RELATIVE_DIFFERENCE_DEAD_ZONE,
) -> int:
    """Count sign changes in the dead-zoned cyclic forward difference."""

    difference = np.roll(field, -1) - field
    scale = float(np.max(np.abs(difference)))
    if scale == 0.0:
        return 0
    threshold = relative_dead_zone * scale
    signs = np.where(
        difference > threshold,
        1,
        np.where(difference < -threshold, -1, 0),
    )
    nonzero = signs[signs != 0]
    if nonzero.size <= 1:
        return 0
    return int(np.count_nonzero(nonzero != np.roll(nonzero, -1)))


def rollout_paths(source_dir: Path) -> tuple[Path, ...]:
    """Return the production-step trajectory artifacts used by the audit."""

    return (
        source_dir / "tanaka_benjamin_feir_dt_0p010.npz",
        source_dir / "jonswap_tma_shallow_dt_0p010.npz",
        source_dir / "jonswap_tma_finite_dt_0p010.npz",
        source_dir / "jonswap_tma_deep_dt_0p010.npz",
    )


def load_terminal_cases(source_dir: Path) -> tuple[TerminalCase, ...]:
    """Load initial and last finite states from every time-dependent case."""

    cases: list[TerminalCase] = []
    for path in rollout_paths(source_dir):
        with np.load(path, allow_pickle=False) as archive:
            times = np.asarray(archive["times"], dtype=np.float64)
            eta = archive["eta"]
            xi = archive["xi"]
            gxi = archive["gxi"]
            case_ids = archive["case_id"]
            families = archive["family"]
            depths = np.asarray(archive["depth"], dtype=np.float64)
            converged = np.asarray(
                archive["gl2_all_stages_converged"],
                dtype=np.bool_,
            )

            for case_index, case_id in enumerate(case_ids):
                finite = (
                    np.isfinite(eta[:, case_index]).all(axis=1)
                    & np.isfinite(xi[:, case_index]).all(axis=1)
                    & np.isfinite(gxi[:, case_index]).all(axis=1)
                )
                finite_indices = np.flatnonzero(finite)
                if finite_indices.size == 0:
                    continue
                endpoint_index = int(finite_indices[-1])
                eta_initial = np.asarray(
                    eta[0, case_index],
                    dtype=np.float64,
                ).copy()
                eta_endpoint = np.asarray(
                    eta[endpoint_index, case_index],
                    dtype=np.float64,
                ).copy()
                gxi_initial = np.asarray(
                    gxi[0, case_index],
                    dtype=np.float64,
                ).copy()
                gxi_endpoint = np.asarray(
                    gxi[endpoint_index, case_index],
                    dtype=np.float64,
                ).copy()
                completed = bool(
                    converged[case_index]
                    and endpoint_index == times.size - 1
                )
                cases.append(
                    TerminalCase(
                        case_id=str(case_id),
                        family=str(families[case_index]),
                        depth=float(depths[case_index]),
                        endpoint_time=float(times[endpoint_index]),
                        completed=completed,
                        eta_initial=eta_initial,
                        eta_endpoint=eta_endpoint,
                        gxi_initial=gxi_initial,
                        gxi_endpoint=gxi_endpoint,
                        eta_effective_wavenumber=effective_wavenumber(
                            eta_endpoint
                        ),
                        gxi_effective_wavenumber=effective_wavenumber(
                            gxi_endpoint
                        ),
                        gxi_sign_changes=cyclic_sign_changes(gxi_endpoint),
                        eta_high_band_energy_fraction=(
                            high_band_energy_fraction(eta_endpoint)
                        ),
                        gxi_high_band_energy_fraction=(
                            high_band_energy_fraction(gxi_endpoint)
                        ),
                    )
                )
    return tuple(cases)


def normalized_spectrum(field: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return a unit-maximum one-sided amplitude spectrum."""

    centered = field - float(np.mean(field))
    amplitude = np.abs(np.fft.rfft(centered))
    scale = float(np.max(amplitude[1:]))
    if scale > 0.0:
        amplitude = amplitude / scale
    return np.arange(amplitude.size, dtype=np.float64), amplitude


def family_label(family: str) -> str:
    """Return a compact plotting label for one family."""

    return {
        "tanaka": "Tanaka",
        "benjamin_feir": "Benjamin--Feir",
        "jonswap_tma_shallow": "JONSWAP/TMA shallow",
        "jonswap_tma_finite": "JONSWAP/TMA finite",
        "jonswap_tma_deep": "JONSWAP/TMA deep",
    }.get(family, family)


def plot_ranked_cases(
    cases: tuple[TerminalCase, ...],
    *,
    key: str,
    metric_label: str,
    output_stem: Path,
    scope_label: str,
    top_count: int,
) -> None:
    """Plot initial and endpoint fields for the largest values of one metric."""

    ranked = sorted(cases, key=lambda case: getattr(case, key), reverse=True)
    selected = ranked[: min(top_count, len(ranked))]
    x = np.linspace(0.0, 2.0 * np.pi, selected[0].eta_initial.size, endpoint=False)
    figure, axes = plt.subplots(
        len(selected),
        3,
        figsize=(11.5, 2.05 * len(selected)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, case in enumerate(selected):
        endpoint_label = "terminal" if case.completed else "last finite"
        eta_scale = case.depth
        gxi_scale = np.sqrt(case.depth)
        axes[row_index, 0].plot(
            x,
            case.eta_initial / eta_scale,
            color="0.55",
            linewidth=0.9,
            linestyle="--",
            label="initial",
        )
        axes[row_index, 0].plot(
            x,
            case.eta_endpoint / eta_scale,
            color="tab:blue",
            linewidth=1.0,
            label=endpoint_label,
        )
        axes[row_index, 0].set_ylabel(r"$\eta/h$")

        axes[row_index, 1].plot(
            x,
            case.gxi_initial / gxi_scale,
            color="0.55",
            linewidth=0.9,
            linestyle="--",
        )
        axes[row_index, 1].plot(
            x,
            case.gxi_endpoint / gxi_scale,
            color="tab:orange",
            linewidth=1.0,
        )
        axes[row_index, 1].set_ylabel(r"$G(\eta)\xi/\sqrt{h}$")

        eta_k, eta_spectrum = normalized_spectrum(case.eta_endpoint)
        gxi_k, gxi_spectrum = normalized_spectrum(case.gxi_endpoint)
        axes[row_index, 2].semilogy(
            eta_k[1:129],
            np.maximum(eta_spectrum[1:129], 1.0e-14),
            color="tab:blue",
            linewidth=0.9,
            label=r"$\eta$",
        )
        axes[row_index, 2].semilogy(
            gxi_k[1:129],
            np.maximum(gxi_spectrum[1:129], 1.0e-14),
            color="tab:orange",
            linewidth=0.9,
            label=r"$G(\eta)\xi$",
        )
        axes[row_index, 2].set_ylim(1.0e-12, 2.0)
        axes[row_index, 2].set_ylabel("normalized amplitude")

        status = "" if case.completed else "; incomplete"
        metric_value = getattr(case, key)
        row_title = (
            f"{family_label(case.family)}; {case.case_id}; "
            f"{endpoint_label} t={case.endpoint_time:g}; "
            f"{metric_label}={metric_value:g}{status}"
        )
        axes[row_index, 0].set_title(
            row_title,
            loc="left",
            fontsize=8.5,
            color="firebrick" if not case.completed else "black",
        )
        for axis in axes[row_index]:
            axis.grid(alpha=0.18, linewidth=0.4)
            axis.tick_params(labelsize=7)

    for axis in axes[-1, :2]:
        axis.set_xlabel(r"$x$")
    axes[-1, 2].set_xlabel(r"wavenumber $k$")
    axes[0, 0].legend(loc="best", fontsize=7)
    axes[0, 2].legend(loc="best", fontsize=7)
    figure.suptitle(
        f"{scope_label}: diagnostic rankings, not rejection rules",
        fontsize=11,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=180)
    figure.savefig(output_stem.with_suffix(".pdf"))
    plt.close(figure)


def case_record(case: TerminalCase) -> dict[str, object]:
    """Return the JSON-ready scalar record for one terminal case."""

    return {
        "case_id": case.case_id,
        "family": case.family,
        "depth": case.depth,
        "endpoint_time": case.endpoint_time,
        "endpoint_kind": "terminal" if case.completed else "last_finite",
        "completed": case.completed,
        "eta_effective_wavenumber": case.eta_effective_wavenumber,
        "gxi_effective_wavenumber": case.gxi_effective_wavenumber,
        "gxi_cyclic_forward_difference_sign_changes": case.gxi_sign_changes,
        "eta_k_ge_64_energy_fraction": case.eta_high_band_energy_fraction,
        "gxi_k_ge_64_energy_fraction": case.gxi_high_band_energy_fraction,
    }


def main() -> None:
    """Rank completed endpoints and write a machine-readable summary."""

    args = parse_args()
    if args.top_count <= 0:
        raise ValueError("--top-count must be positive")
    paths = rollout_paths(args.source_dir)
    missing = tuple(path for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"missing rollout artifacts: {missing}")

    cases = load_terminal_cases(args.source_dir)
    if not cases:
        raise RuntimeError("the source panel contains no finite trajectory frame")
    completed_cases = tuple(case for case in cases if case.completed)
    if not completed_cases:
        raise RuntimeError("the source panel contains no completed trajectory")
    non_jonswap_cases = tuple(
        case
        for case in completed_cases
        if not case.family.startswith("jonswap_tma")
    )
    if not non_jonswap_cases:
        raise RuntimeError(
            "the source panel contains no completed non-JONSWAP trajectory"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_ranked_cases(
        completed_cases,
        key="eta_effective_wavenumber",
        metric_label=r"$k_{\rm eff}(\eta)$",
        output_stem=args.output_dir / "top_terminal_spatial_roughness",
        scope_label="Completed-trajectory endpoints",
        top_count=args.top_count,
    )
    plot_ranked_cases(
        completed_cases,
        key="gxi_sign_changes",
        metric_label=r"$C_{0.03}(G(\eta)\xi)$",
        output_stem=args.output_dir / "top_terminal_sign_changes",
        scope_label="Completed-trajectory endpoints",
        top_count=args.top_count,
    )
    plot_ranked_cases(
        non_jonswap_cases,
        key="eta_effective_wavenumber",
        metric_label=r"$k_{\rm eff}(\eta)$",
        output_stem=(
            args.output_dir / "top_non_jonswap_terminal_spatial_roughness"
        ),
        scope_label="Completed non-JONSWAP endpoints",
        top_count=args.top_count,
    )
    plot_ranked_cases(
        non_jonswap_cases,
        key="gxi_sign_changes",
        metric_label=r"$C_{0.03}(G(\eta)\xi)$",
        output_stem=args.output_dir / "top_non_jonswap_terminal_sign_changes",
        scope_label="Completed non-JONSWAP endpoints",
        top_count=args.top_count,
    )

    records = [case_record(case) for case in cases]
    summary = {
        "schema": "paper_corpus_terminal_diagnostics_v3",
        "scope": (
            "completed trajectories only; incomplete trajectories are excluded "
            "before descriptive endpoint ranking; separate display rankings "
            "also omit JONSWAP/TMA; no metric participates in corpus acceptance"
        ),
        "source": {
            "directory": str(args.source_dir.resolve()),
            "artifacts": {
                path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                for path in paths
            },
        },
        "definitions": {
            "eta_effective_wavenumber": (
                "sqrt(sum_{k!=0} k^2 |eta_hat_k|^2 / "
                "sum_{k!=0} |eta_hat_k|^2)"
            ),
            "gxi_sign_changes": (
                "cyclic sign changes among nonzero signs of the forward "
                "difference of G(eta)xi after a 3%-of-row-maximum dead zone"
            ),
            "high_band_energy_fraction": (
                "sum_{|k|>=64} |field_hat_k|^2 / "
                "sum_{|k|>=1} |field_hat_k|^2"
            ),
        },
        "source_case_count": len(cases),
        "ranked_complete_case_count": len(completed_cases),
        "excluded_incomplete_case_count": len(cases) - len(completed_cases),
        "excluded_incomplete_case_ids": [
            case.case_id for case in cases if not case.completed
        ],
        "ranked_non_jonswap_case_count": len(non_jonswap_cases),
        "display_excluded_jonswap_case_ids": [
            case.case_id
            for case in completed_cases
            if case.family.startswith("jonswap_tma")
        ],
        "source_cases": records,
        "ranking_by_eta_effective_wavenumber": [
            case.case_id
            for case in sorted(
                completed_cases,
                key=lambda item: item.eta_effective_wavenumber,
                reverse=True,
            )
        ],
        "ranking_by_gxi_sign_changes": [
            case.case_id
            for case in sorted(
                completed_cases,
                key=lambda item: item.gxi_sign_changes,
                reverse=True,
            )
        ],
        "non_jonswap_ranking_by_eta_effective_wavenumber": [
            case.case_id
            for case in sorted(
                non_jonswap_cases,
                key=lambda item: item.eta_effective_wavenumber,
                reverse=True,
            )
        ],
        "non_jonswap_ranking_by_gxi_sign_changes": [
            case.case_id
            for case in sorted(
                non_jonswap_cases,
                key=lambda item: item.gxi_sign_changes,
                reverse=True,
            )
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
