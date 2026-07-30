"""Plot the conservative Ursell support diagnosis for the fresh Stokes panel."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from solver.data.stokes_truth_jax import (  # noqa: E402
    FINITE_DEPTH_STOKES_URSELL_LIMIT,
    finite_depth_eta_harmonics,
    finite_depth_stokes_ursell_upper_bound,
)

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
IC_PANEL = (
    ROOT / "data/manuscript_ic_panels_v1_20260715/stokes_finite_ics.npz"
)
TRUTH_PANEL = (
    ROOT
    / "outputs/c21_tangent_w100_from_c20_20260714_171442"
    / "eval_final_heldout_unguarded_n256_20260715_035324"
    / "stokes_finite"
    / "stokes_finite_trajs.npz"
)
OUTPUT_DIR = ROOT / "outputs/paper_corpus_shape_stress_20260724"


def _decode_metadata(encoded: np.ndarray) -> dict[str, Any]:
    raw = encoded.item()
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("panel metadata must decode to an object")
    return value


def _maximum_finite_drift(drift: np.ndarray) -> np.ndarray:
    def maximum(row: np.ndarray) -> float:
        finite = np.abs(row[np.isfinite(row)])
        return float(np.max(finite)) if finite.size else float("nan")

    return np.asarray(list(map(maximum, drift.T)), dtype=np.float64)


def _phase_profile(
    specification: dict[str, Any],
    *,
    length: float,
) -> tuple[np.ndarray, np.ndarray]:
    mode = int(specification["n0"])
    depth = float(specification["depth"])
    wavenumber = 2.0 * np.pi * mode / length
    harmonics = np.asarray(
        finite_depth_eta_harmonics(
            wavenumber,
            depth,
            1.0,
            float(specification["a0"]),
        ),
        dtype=np.float64,
    )
    phase = np.linspace(-np.pi, np.pi, 2049)
    harmonic_modes = np.arange(1, 6, dtype=np.float64)
    elevation = np.sum(
        harmonics[:, None]
        * np.cos(harmonic_modes[:, None] * phase[None, :]),
        axis=0,
    )
    return phase / (2.0 * np.pi), elevation / depth


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(IC_PANEL, allow_pickle=False) as archive:
        case_ids = np.asarray(archive["case_ids"], dtype=np.int64)
        metadata = _decode_metadata(archive["meta.json"])
    with np.load(TRUTH_PANEL, allow_pickle=False) as archive:
        truth_case_ids = np.asarray(archive["case_ids"], dtype=np.int64)
        truth_valid = np.asarray(archive["truth_valid"], dtype=np.bool_)
        energy_drift = np.asarray(
            archive["energy_drift_truth"],
            dtype=np.float64,
        )
    np.testing.assert_array_equal(case_ids, truth_case_ids)

    length = float(metadata["length"])
    specifications = metadata["case_specs"]
    ursell = np.asarray(
        [
            float(
                finite_depth_stokes_ursell_upper_bound(
                    2.0 * np.pi * int(specification["n0"]) / length,
                    float(specification["depth"]),
                    1.0,
                    float(specification["a0"]),
                )
            )
            for specification in specifications
        ],
        dtype=np.float64,
    )
    maximum_drift = _maximum_finite_drift(energy_drift)
    nonfinite = ~truth_valid & np.any(~np.isfinite(energy_drift), axis=0)
    finite_invalid = ~truth_valid & ~nonfinite

    inside_index = int(
        np.flatnonzero(ursell <= FINITE_DEPTH_STOKES_URSELL_LIMIT)[
            np.argmax(ursell[ursell <= FINITE_DEPTH_STOKES_URSELL_LIMIT])
        ]
    )
    outside_index = int(
        np.flatnonzero(ursell > FINITE_DEPTH_STOKES_URSELL_LIMIT)[
            np.argmin(ursell[ursell > FINITE_DEPTH_STOKES_URSELL_LIMIT])
        ]
    )
    invalid_indices = np.flatnonzero(~truth_valid)
    worst_index = int(invalid_indices[np.argmax(ursell[invalid_indices])])

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.5),
        constrained_layout=True,
    )
    axes[0].scatter(
        ursell[truth_valid],
        np.maximum(maximum_drift[truth_valid], 1e-12),
        s=24,
        color="#2f6da3",
        alpha=0.72,
        label="numerically healthy",
    )
    axes[0].scatter(
        ursell[finite_invalid],
        maximum_drift[finite_invalid],
        s=48,
        color="#b33b2e",
        marker="D",
        label=r"finite, drift $>10^{-3}$",
    )
    axes[0].scatter(
        ursell[nonfinite],
        np.full(np.count_nonzero(nonfinite), 1.0),
        s=68,
        color="#7f1d1d",
        marker="x",
        linewidths=2.0,
        label="nonfinite (shown at ordinate 1)",
    )
    axes[0].axvline(
        FINITE_DEPTH_STOKES_URSELL_LIMIT,
        color="black",
        linestyle="--",
        linewidth=1.3,
        label=r"Stokes support: $\mathrm{Ur}_{+}=26$",
    )
    axes[0].axhline(1e-3, color="0.45", linestyle=":", linewidth=1.0)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(
        r"conservative Ursell value $\mathrm{Ur}_{+}=H_{+}\lambda^2/h^3$"
    )
    axes[0].set_ylabel("maximum stored relative energy drift")
    axes[0].set_title("Fresh 256-case finite-Stokes panel")
    axes[0].legend(fontsize=8, loc="lower right")

    selected = (
        (inside_index, "#2f6da3", "nearest inside"),
        (outside_index, "#d87818", "nearest outside; healthy"),
        (worst_index, "#b33b2e", "largest unhealthy"),
    )
    for index, color, label in selected:
        phase, profile = _phase_profile(
            specifications[index],
            length=length,
        )
        axes[1].plot(
            phase,
            profile,
            color=color,
            linewidth=1.8,
            label=(
                f"{label}: case {case_ids[index]}, "
                rf"$\mathrm{{Ur}}_+={ursell[index]:.2f}$"
            ),
        )
    axes[1].set_xlabel(r"carrier phase $\theta/(2\pi)$")
    axes[1].set_ylabel(r"$\eta/h$")
    axes[1].set_title("Phase-aligned constructed profiles")
    axes[1].legend(fontsize=8)

    png_path = OUTPUT_DIR / "finite_stokes_ursell_panel.png"
    pdf_path = OUTPUT_DIR / "finite_stokes_ursell_panel.pdf"
    figure.savefig(png_path, dpi=220)
    figure.savefig(pdf_path)
    plt.close(figure)

    summary = {
        "ursell_limit": FINITE_DEPTH_STOKES_URSELL_LIMIT,
        "cases": int(case_ids.size),
        "truth_healthy": int(np.count_nonzero(truth_valid)),
        "truth_unhealthy": int(np.count_nonzero(~truth_valid)),
        "outside_support": int(
            np.count_nonzero(ursell > FINITE_DEPTH_STOKES_URSELL_LIMIT)
        ),
        "unhealthy_outside_support": int(
            np.count_nonzero(
                (~truth_valid)
                & (ursell > FINITE_DEPTH_STOKES_URSELL_LIMIT)
            )
        ),
        "selected_profiles": [
            {
                "role": label,
                "index": index,
                "case_id": int(case_ids[index]),
                "ursell_upper_bound": float(ursell[index]),
                "truth_valid": bool(truth_valid[index]),
                "maximum_finite_energy_drift": (
                    float(maximum_drift[index])
                    if np.isfinite(maximum_drift[index])
                    else None
                ),
            }
            for index, _, label in selected
        ],
    }
    (OUTPUT_DIR / "finite_stokes_ursell_panel.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(png_path)


if __name__ == "__main__":
    main()
