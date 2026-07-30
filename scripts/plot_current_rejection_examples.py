"""Plot examples excluded by the current paper-corpus rules.

The first row replays one amplitude proposal rejected while sampling the
declared finite-depth Stokes support.  The other rows use only the production
``dt=0.01`` arms of two full-horizon validation cases that do not return a
complete trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import TypeAlias

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from solver.data.stokes_truth_jax import (  # noqa: E402
    FINITE_DEPTH_STOKES_URSELL_LIMIT,
    stokes_eta_xi_at_phase,
)
from solver.gen_data.pipeline.production import (  # noqa: E402
    AttemptAssignment,
    CaseKey,
    PhysicalFamilyId,
    SplitId,
)
from solver.gen_data.stokes_population import (  # noqa: E402
    StokesAmplitudeAttempt,
    StokesPopulationSample,
    sample_stokes_population,
)


FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
JsonRecord: TypeAlias = dict[str, object]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "outputs/full_horizon_refinement_panel_20260725"
    / "tanaka_benjamin_feir_dt_0p010.npz"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs/paper_corpus_current_rejection_examples_20260726"
)
TRAJECTORY_CASES = (
    ("tanaka_steep_upper_seam", "Tanaka"),
    ("bf_jcp09_canonical", "Benjamin--Feir"),
)
STOKES_ATTEMPT_INDEX = 2
STOKES_CELL_ID = "finite_moderate"
STOKES_STREAM_ID = 0
GRAVITY = 1.0


def parse_args() -> argparse.Namespace:
    """Parse plotting paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stokes_assignment() -> AttemptAssignment:
    """Return the fixed current-sampler assignment used in the first row."""

    return AttemptAssignment(
        case_key=CaseKey(
            family_id=PhysicalFamilyId.STOKES,
            revision_id=1,
            split_id=SplitId.VALIDATION,
            stream_id=STOKES_STREAM_ID,
            attempt_index=STOKES_ATTEMPT_INDEX,
        ),
        cell_id=STOKES_CELL_ID,
    )


def require_stokes_support_pair(
) -> tuple[
    StokesPopulationSample,
    StokesAmplitudeAttempt,
    StokesAmplitudeAttempt,
]:
    """Replay one rejected amplitude followed by its accepted redraw."""

    sample = sample_stokes_population(stokes_assignment())
    if sample.support_resampling_count != 1:
        raise RuntimeError("the fixed Stokes example no longer has one redraw")
    rejected, accepted = sample.amplitude_attempts
    if (
        rejected.accepted
        or not accepted.accepted
        or rejected.ursell_upper_bound is None
        or accepted.ursell_upper_bound is None
        or rejected.ursell_upper_bound <= FINITE_DEPTH_STOKES_URSELL_LIMIT
        or accepted.ursell_upper_bound > FINITE_DEPTH_STOKES_URSELL_LIMIT
    ):
        raise RuntimeError("the fixed Stokes support decisions changed")
    return sample, rejected, accepted


def stokes_fields(
    sample: StokesPopulationSample,
    attempt: StokesAmplitudeAttempt,
    *,
    nx: int,
) -> tuple[FloatArray, FloatArray]:
    """Evaluate a Stokes proposal without applying the support decision."""

    x = (
        sample.domain_length
        * np.arange(nx, dtype=np.float64)
        / nx
    )
    with jax.enable_x64():
        eta, xi = stokes_eta_xi_at_phase(
            x=jnp.asarray(x, dtype=jnp.float64),
            phase=sample.phase,
            n0=sample.carrier_mode,
            a0=attempt.amplitude,
            length=sample.domain_length,
            depth=sample.depth,
            gravity=sample.gravity,
            ichoi=1,
        )
    eta_host, xi_host = (
        np.asarray(value, dtype=np.float64)
        for value in jax.device_get((eta, xi))
    )
    if not np.isfinite(eta_host).all() or not np.isfinite(xi_host).all():
        raise RuntimeError("the displayed Stokes support proposals must be finite")
    return eta_host, xi_host


def saved_state_is_finite(
    eta: FloatArray,
    xi: FloatArray,
    gxi: FloatArray,
) -> BoolArray:
    """Return the saved rows having three finite delivered fields."""

    return np.asarray(
        np.all(np.isfinite(eta), axis=1)
        & np.all(np.isfinite(xi), axis=1)
        & np.all(np.isfinite(gxi), axis=1),
        dtype=np.bool_,
    )


def normalized_fields(
    eta: FloatArray,
    xi: FloatArray,
    gxi: FloatArray,
    *,
    depth: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return the three dimensionless delivered fields."""

    velocity = math.sqrt(GRAVITY * depth)
    return (
        np.asarray(eta / depth, dtype=np.float64),
        np.asarray(xi / (depth * velocity), dtype=np.float64),
        np.asarray(gxi / velocity, dtype=np.float64),
    )


def load_trajectory_examples(
    source: Path,
) -> tuple[FloatArray, list[JsonRecord]]:
    """Load and verify the two incomplete ``dt=0.01`` trajectories."""

    records: list[JsonRecord] = []
    with np.load(source, allow_pickle=False) as archive:
        if not math.isclose(
            float(archive["dt"]),
            0.01,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError("trajectory examples must use the production step")
        case_ids = tuple(map(str, archive["case_id"]))
        times = np.asarray(archive["times"], dtype=np.float64)
        step_times = np.asarray(archive["gl2_step_times"], dtype=np.float64)
        tolerance = float(archive["gl2_residual_tolerance"])
        x_over_length = (
            np.arange(archive["eta"].shape[-1], dtype=np.float64)
            / archive["eta"].shape[-1]
        )

        for case_id, label in TRAJECTORY_CASES:
            case_index = case_ids.index(case_id)
            eta = np.asarray(archive["eta"][:, case_index], dtype=np.float64)
            xi = np.asarray(archive["xi"][:, case_index], dtype=np.float64)
            gxi = np.asarray(archive["gxi"][:, case_index], dtype=np.float64)
            finite_indices = np.flatnonzero(saved_state_is_finite(eta, xi, gxi))
            if finite_indices.size == 0:
                raise RuntimeError(f"{case_id} has no finite saved state")
            last_finite_index = int(finite_indices[-1])
            first_failed_step = int(
                archive["gl2_first_failed_step"][case_index]
            )
            if first_failed_step < 0:
                raise RuntimeError(f"{case_id} is not an incomplete trajectory")
            first_residual = float(
                archive["gl2_stage_residual"][first_failed_step, case_index]
            )
            first_stage_finite = bool(
                archive["gl2_stage_finite"][first_failed_step, case_index]
            )
            first_state_finite = bool(
                archive["gl2_state_finite"][first_failed_step, case_index]
            )
            first_iterations = int(
                archive["gl2_iterations"][first_failed_step, case_index]
            )
            first_hit_iteration_cap = bool(
                archive["gl2_hit_iteration_cap"][
                    first_failed_step,
                    case_index,
                ]
            )
            first_converged = bool(
                archive["gl2_converged"][first_failed_step, case_index]
            )
            if first_converged:
                raise RuntimeError(f"{case_id} has no failed GL2 stage")
            depth = float(archive["depth"][case_index])
            fields_initial = normalized_fields(
                eta[0],
                xi[0],
                gxi[0],
                depth=depth,
            )
            fields_last = normalized_fields(
                eta[last_finite_index],
                xi[last_finite_index],
                gxi[last_finite_index],
                depth=depth,
            )
            records.append(
                {
                    "case_id": case_id,
                    "label": label,
                    "depth": depth,
                    "requested_terminal_time": float(times[-1]),
                    "last_finite_saved_index": last_finite_index,
                    "last_finite_saved_time": float(times[last_finite_index]),
                    "first_failed_step": first_failed_step,
                    "first_failed_step_time": float(step_times[first_failed_step]),
                    "first_failed_stage_residual": (
                        first_residual if math.isfinite(first_residual) else None
                    ),
                    "gl2_residual_tolerance": tolerance,
                    "first_failed_stage_finite": first_stage_finite,
                    "first_failed_state_finite": first_state_finite,
                    "first_failed_stage_iterations": first_iterations,
                    "first_failed_stage_hit_iteration_cap": (
                        first_hit_iteration_cap
                    ),
                    "fields_initial": fields_initial,
                    "fields_last": fields_last,
                    "minimum_water_column_at_last_finite_state": float(
                        np.min(depth + eta[last_finite_index])
                    ),
                }
            )
    return x_over_length, records


def plot_support_row(
    axes: NDArray[np.object_],
    *,
    x_over_length: FloatArray,
    sample: StokesPopulationSample,
    rejected: StokesAmplitudeAttempt,
    accepted: StokesAmplitudeAttempt,
) -> JsonRecord:
    """Plot the current Stokes conditional-support example."""

    nx = x_over_length.size
    rejected_eta, rejected_xi = stokes_fields(sample, rejected, nx=nx)
    accepted_eta, accepted_xi = stokes_fields(sample, accepted, nx=nx)
    depth = sample.depth
    velocity = math.sqrt(GRAVITY * depth)
    pairs = (
        (rejected_eta / depth, accepted_eta / depth),
        (
            rejected_xi / (depth * velocity),
            accepted_xi / (depth * velocity),
        ),
    )
    for axis, (rejected_field, accepted_field) in zip(
        axes[:2],
        pairs,
        strict=True,
    ):
        axis.plot(
            x_over_length,
            accepted_field,
            color="#3568b8",
            linewidth=1.2,
            label="accepted redraw",
        )
        axis.plot(
            x_over_length,
            rejected_field,
            color="#d95f02",
            linewidth=1.2,
            label="excluded proposal",
        )
        axis.grid(alpha=0.2, linewidth=0.5)

    support_axis = axes[2]
    rejected_ursell = float(rejected.ursell_upper_bound)
    accepted_ursell = float(accepted.ursell_upper_bound)
    support_axis.axvline(
        FINITE_DEPTH_STOKES_URSELL_LIMIT,
        color="black",
        linestyle=(0, (4, 2)),
        linewidth=1.0,
        label=r"$\mathrm{Ur}_+=26$",
    )
    support_axis.scatter(
        [accepted_ursell],
        [0.35],
        color="#3568b8",
        s=35,
        zorder=3,
    )
    support_axis.scatter(
        [rejected_ursell],
        [0.65],
        color="#d95f02",
        marker="x",
        s=45,
        linewidths=1.6,
        zorder=3,
    )
    support_axis.annotate(
        f"accepted: {accepted_ursell:.2f}",
        (accepted_ursell, 0.35),
        xytext=(4, -12),
        textcoords="offset points",
        color="#3568b8",
    )
    support_axis.annotate(
        f"excluded: {rejected_ursell:.2f}",
        (rejected_ursell, 0.65),
        xytext=(4, 5),
        textcoords="offset points",
        color="#d95f02",
    )
    support_axis.set_xlim(
        min(accepted_ursell, FINITE_DEPTH_STOKES_URSELL_LIMIT) - 2.0,
        max(rejected_ursell, FINITE_DEPTH_STOKES_URSELL_LIMIT) + 2.0,
    )
    support_axis.set_ylim(0.0, 1.0)
    support_axis.set_yticks(())
    support_axis.set_xlabel(r"conservative $\mathrm{Ur}_+$")
    support_axis.grid(axis="x", alpha=0.2, linewidth=0.5)

    case_key = sample.assignment.case_key
    return {
        "kind": "declared_support_proposal",
        "family": "stokes",
        "case_id": int(case_key.case_id),
        "split": case_key.split_id.value,
        "stream_id": case_key.stream_id,
        "attempt_index": case_key.attempt_index,
        "cell_id": sample.cell.cell_id,
        "carrier_mode": sample.carrier_mode,
        "depth": depth,
        "phase": sample.phase,
        "ursell_limit": FINITE_DEPTH_STOKES_URSELL_LIMIT,
        "excluded_amplitude": rejected.amplitude,
        "excluded_ursell_upper_bound": rejected_ursell,
        "accepted_amplitude": accepted.amplitude,
        "accepted_ursell_upper_bound": accepted_ursell,
        "interpretation": (
            "The first amplitude proposal lies outside the declared "
            "finite-depth Stokes support. The same fixed carrier, depth, "
            "phase, and cell are retained while amplitude is redrawn."
        ),
    }


def make_figure(
    *,
    source: Path,
    output_dir: Path,
) -> JsonRecord:
    """Create the current-rule rejection panel and its numerical record."""

    x_over_length, trajectory_records = load_trajectory_examples(source)
    sample, rejected, accepted = require_stokes_support_pair()

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(11.5, 7.2),
        sharex=False,
        constrained_layout=True,
    )
    axes[0, 0].set_title(r"$\eta/h$")
    axes[0, 1].set_title(r"$\xi/(h\sqrt{gh})$")
    axes[0, 2].set_title("declared support")
    support_record = plot_support_row(
        axes[0],
        x_over_length=x_over_length,
        sample=sample,
        rejected=rejected,
        accepted=accepted,
    )
    axes[0, 0].set_ylabel(
        "1. Stokes construction\n"
        r"excluded: $\mathrm{Ur}_+>26$"
    )
    axes[0, 0].legend(loc="best", frameon=False)

    field_titles = (
        r"$\eta/h$",
        r"$\xi/(h\sqrt{gh})$",
        r"$G_{\mathrm{ref}}(\eta;h)\xi/\sqrt{gh}$",
    )
    for row, record in enumerate(trajectory_records, start=1):
        initial_fields = record.pop("fields_initial")
        last_fields = record.pop("fields_last")
        assert isinstance(initial_fields, tuple)
        assert isinstance(last_fields, tuple)
        for column, (initial, last) in enumerate(
            zip(initial_fields, last_fields, strict=True)
        ):
            axis = axes[row, column]
            axis.plot(
                x_over_length,
                initial,
                color="0.55",
                linestyle=(0, (3, 2)),
                linewidth=1.0,
                label=r"$t=0$",
            )
            axis.plot(
                x_over_length,
                last,
                color="#d95f02",
                linewidth=1.1,
                label="last finite saved state",
            )
            axis.grid(alpha=0.2, linewidth=0.5)
            axis.set_title(field_titles[column])
            axis.set_xlabel(r"$x/L$")
        label = str(record["label"])
        last_time = float(record["last_finite_saved_time"])
        failure_time = float(record["first_failed_step_time"])
        requested = float(record["requested_terminal_time"])
        axes[row, 0].set_ylabel(
            f"{row + 1}. {label} rollout\n"
            rf"$t_{{\rm last}}={last_time:.2f}$, "
            rf"fails at ${failure_time:.2f}<T={requested:.0f}$"
        )
        axes[row, 0].legend(loc="best", frameon=False)
        record["kind"] = "incomplete_trajectory"
        record["production_decision"] = (
            "zero rows: the dt=0.01 method did not reach the requested horizon"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "current_rejection_examples.png"
    pdf_path = output_dir / "current_rejection_examples.pdf"
    figure.savefig(png_path, dpi=240, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    result: JsonRecord = {
        "schema": "paper_corpus_current_rejection_examples_v1",
        "policy": {
            "stage_1": (
                "sample and construct inside the declared mathematical family"
            ),
            "stage_2": (
                "for rollout families, the single dt=0.01 calculation must "
                "return the complete requested trajectory"
            ),
            "not_used": [
                "sign count",
                "slope",
                "spectral shape",
                "Hamiltonian drift",
                "visual appearance",
                "per-case time refinement",
            ],
        },
        "construction_example": support_record,
        "trajectory_examples": trajectory_records,
        "source_dt_0p01": str(source.relative_to(ROOT)),
        "figure_png": str(png_path.relative_to(ROOT)),
        "figure_pdf": str(pdf_path.relative_to(ROOT)),
    }
    png_sha256 = sha256(png_path)
    pdf_sha256 = sha256(pdf_path)
    result["figure_png_sha256"] = png_sha256
    result["figure_pdf_sha256"] = pdf_sha256
    json_path = output_dir / "current_rejection_examples.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    """Generate the figure and print its strict numerical record."""

    args = parse_args()
    result = make_figure(
        source=args.source.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
