"""Visualize every specification in the 500,000-case population smoke.

The input is a parameter-population smoke, not a trajectory corpus.  Every
stored case contributes to the count, histogram, hexbin, or empirical-CDF
panels below.  A small deterministic subset is reconstructed separately with
the production initial-condition formulas so that the parameter plots can be
connected to physical fields without pretending to plot 500,000 curves.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, TypeAlias

# These assignments precede imports that can initialize JAX.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["DNO_TANAKA_DTYPE"] = "float64"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from solver.gen_data.benjamin_feir_jcp09 import (  # noqa: E402
    ParameterArrays,
    build_initial_conditions as build_benjamin_feir_initial_conditions,
)
from solver.gen_data.benjamin_feir_population import (  # noqa: E402
    BenjaminFeirPopulationSample,
    sample_benjamin_feir_population,
)
from solver.gen_data.generate_tanaka_dataset_v2 import (  # noqa: E402
    build_per_case_initial_conditions,
)
from solver.gen_data.jonswap_tma import (  # noqa: E402
    ResolvedBand,
    build_jonswap_tma_initial_condition,
)
from solver.gen_data.jonswap_tma_population import (  # noqa: E402
    JonswapTmaPopulationSample,
    sample_jonswap_tma_population,
)
from solver.gen_data.multi_crest import CrestSpec  # noqa: E402
from solver.gen_data.pipeline.production import (  # noqa: E402
    PAPER_CORPUS_REVISION_ID,
    AttemptAssignment,
    CaseKey,
    PhysicalFamilyId,
    SplitId,
)
from solver.gen_data.pipeline.reference import (  # noqa: E402
    PAPER_DNO_TARGET,
    project_fixed_band,
)
from solver.gen_data.stokes_population import (  # noqa: E402
    StokesPopulationSample,
    sample_stokes_population,
)
from solver.gen_data.stokes_static_pipeline import (  # noqa: E402
    PAPER_STATIC_STOKES_CONTRACT,
    construct_stokes_state,
)
from solver.gen_data.tanaka_population import (  # noqa: E402
    TanakaPopulationSample,
    sample_tanaka_population,
)
from solver.solvers.dno_series_jax import build_grid  # noqa: E402
from solver.tanaka_ICs.modified_tanaka import (  # noqa: E402
    make_default_tanaka_template,
)


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    ROOT / "outputs/paper_corpus_population_smoke_500k_20260725"
)
FAMILIES = ("stokes", "tanaka", "benjamin_feir", "jonswap_tma")
FAMILY_LABELS = {
    "stokes": "Fifth-order Stokes",
    "tanaka": "Tanaka-profile sums",
    "benjamin_feir": "Benjamin--Feir",
    "jonswap_tma": "JONSWAP/TMA",
}
FAMILY_COLORS = {
    "stokes": "#4477AA",
    "tanaka": "#228833",
    "benjamin_feir": "#CC6677",
    "jonswap_tma": "#AA3377",
}
STOKES_URSELL_LIMIT = 26.0
DOMAIN_LENGTH = 2.0 * math.pi
GRAVITY = 1.0
PLOT_DPI = 210
DKW_ALPHA = 0.05

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.integer[Any]]
ArrayMap: TypeAlias = dict[str, NDArray[Any]]
JsonMap: TypeAlias = dict[str, object]

COMMON_KEYS = (
    "attempt_index",
    "case_id",
    "cell_code",
)
FAMILY_KEYS = {
    "stokes": (
        "carrier_mode",
        "depth",
        "phase",
        "amplitude",
        "steepness",
        "depth_wavenumber",
        "ursell_upper_bound",
        "support_redraw_count",
        "depth_draw_lower",
        "depth_draw_upper",
        "amplitude_draw_lower",
        "amplitude_draw_upper",
        "attempt_owner",
        "attempt_amplitude_unit",
        "attempt_ursell_ratio",
        "attempt_accepted",
    ),
    "tanaka": (
        "depth",
        "crest_count",
        "total_alpha",
        "required_separation",
        "achieved_separation",
        "crest_alpha",
        "crest_center",
        "crest_direction",
        "crest_gap",
    ),
    "benjamin_feir": (
        "carrier_mode",
        "sideband_offset",
        "carrier_steepness",
        "conditional_steepness_lower",
        "perturbation_ratio",
        "sideband_phase",
        "band_fraction",
        "carrier_amplitude",
    ),
    "jonswap_tma": (
        "stratum_code",
        "depth",
        "significant_height",
        "peak_wavenumber",
        "peak_enhancement",
        "right_moving_fraction",
        "peak_depth",
        "relative_height",
        "peak_steepness",
        "phase_right_first",
        "phase_left_first",
        "phase_right_resultant",
        "phase_left_resultant",
        "phase_hist_edges",
        "phase_hist_counts_right",
        "phase_hist_counts_left",
    ),
}
FAMILY_KEY_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "tanaka": {"crest_gap": ("crest_gap", "cyclic_gap")},
}


@dataclass(frozen=True)
class PopulationData:
    """One validated family archive and its declared cell order."""

    family: str
    path: Path
    arrays: ArrayMap
    cell_order: tuple[str, ...]
    stream_id: int

    @property
    def count(self) -> int:
        """Return the number of complete case specifications."""

        return int(self.arrays["attempt_index"].shape[0])

    @property
    def cell_counts(self) -> NDArray[np.int64]:
        """Return exact observed counts in declared cell order."""

        return np.bincount(
            np.asarray(self.arrays["cell_code"], dtype=np.int64),
            minlength=len(self.cell_order),
        ).astype(np.int64)


@dataclass(frozen=True)
class RoleSelection:
    """One deterministic gallery selection."""

    role: str
    index: int
    score_name: str
    score: float


@dataclass(frozen=True)
class ConstructedField:
    """One selected, resampled, fixed-band initial condition."""

    index: int
    case_id: int
    attempt_index: int
    cell_id: str
    depth: float
    eta: FloatArray
    xi: FloatArray
    parameter_text: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to INPUT_DIR/figures.",
    )
    return parser.parse_args()


def _strict_json_object(path: Path) -> JsonMap:
    """Load a strict JSON object."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"{path} contains nonfinite JSON constant {value!r}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _family_record(summary: Mapping[str, object], family: str) -> Mapping[str, object]:
    families = summary.get("families")
    if not isinstance(families, Mapping):
        raise TypeError("summary.families must be an object")
    record = families.get(family)
    if not isinstance(record, Mapping):
        raise TypeError(f"summary.families.{family} must be an object")
    return record


def _require_current_revision(summary: Mapping[str, object]) -> None:
    """Fail before replay when an archive uses a different generator revision."""

    configuration = summary.get("configuration")
    if not isinstance(configuration, Mapping):
        raise TypeError("summary.configuration must be an object")
    revision_id = configuration.get("revision_id")
    if not isinstance(revision_id, int) or isinstance(revision_id, bool):
        raise TypeError("summary.configuration.revision_id must be an integer")
    if revision_id != PAPER_CORPUS_REVISION_ID:
        raise RuntimeError(
            f"population archive uses generator revision {revision_id}, "
            f"but this plotter imports revision {PAPER_CORPUS_REVISION_ID}; "
            "exact field replay requires the source from the archived revision"
        )


def _family_path(
    input_dir: Path,
    family: str,
    record: Mapping[str, object],
    summary: Mapping[str, object],
) -> Path:
    """Resolve a family NPZ path, accepting a few explicit summary spellings."""

    for key in ("npz_path", "archive_path", "path"):
        candidate = record.get(key)
        if isinstance(candidate, str):
            path = Path(candidate)
            return path if path.is_absolute() else input_dir / path
        if isinstance(candidate, Mapping):
            nested_path = candidate.get("path")
            if isinstance(nested_path, str):
                path = Path(nested_path)
                return path if path.is_absolute() else input_dir / path
    artifacts = summary.get("artifacts")
    if isinstance(artifacts, Mapping):
        artifact = artifacts.get(family)
        if isinstance(artifact, Mapping):
            artifact_path = artifact.get("path")
            if isinstance(artifact_path, str):
                path = Path(artifact_path)
                return path if path.is_absolute() else input_dir / path
    candidates = (
        input_dir / f"{family}.npz",
        input_dir / f"{family}_population.npz",
        input_dir / f"{family}_samples.npz",
    )
    existing = tuple(path for path in candidates if path.exists())
    if len(existing) != 1:
        raise FileNotFoundError(
            f"could not identify exactly one NPZ archive for {family}: "
            + ", ".join(map(str, candidates))
        )
    return existing[0]


def _family_stream_id(
    summary: Mapping[str, object],
    record: Mapping[str, object],
) -> int:
    """Read the deterministic stream identifier used to construct CaseKey."""

    candidates = (
        record.get("stream_id"),
        summary.get("stream_id"),
    )
    configuration = summary.get("configuration")
    if isinstance(configuration, Mapping):
        candidates += (configuration.get("stream_id"),)
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    raise TypeError("family summary must record an integer stream_id")


def load_population(
    input_dir: Path,
    summary: Mapping[str, object],
    family: str,
) -> PopulationData:
    """Load and validate one family archive."""

    record = _family_record(summary, family)
    cell_order_raw = record.get("cell_order")
    quota_record = record.get("quota")
    if cell_order_raw is None and isinstance(quota_record, Mapping):
        cell_order_raw = quota_record.get("cell_order")
    if not isinstance(cell_order_raw, list) or not all(
        isinstance(value, str) and value for value in cell_order_raw
    ):
        raise TypeError(f"{family} summary cell_order must be a list of names")
    cell_order = tuple(cell_order_raw)
    if len(set(cell_order)) != len(cell_order):
        raise ValueError(f"{family} cell_order contains duplicates")

    path = _family_path(input_dir, family, record, summary)
    required = COMMON_KEYS + FAMILY_KEYS[family]
    with np.load(path, allow_pickle=False) as archive:
        aliases = FAMILY_KEY_ALIASES.get(family, {})
        missing = {
            key
            for key in required
            if not any(
                candidate in archive.files
                for candidate in aliases.get(key, (key,))
            )
        }
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        arrays = {
            key: np.asarray(
                archive[
                    next(
                        candidate
                        for candidate in aliases.get(key, (key,))
                        if candidate in archive.files
                    )
                ]
            )
            for key in required
        }

    count = int(arrays["attempt_index"].shape[0])
    if count <= 0:
        raise ValueError(f"{family} archive contains no cases")
    for key in COMMON_KEYS:
        if arrays[key].shape != (count,):
            raise ValueError(f"{family}.{key} must have shape ({count},)")
    case_level_exceptions = {
        "attempt_owner",
        "attempt_amplitude_unit",
        "attempt_ursell_ratio",
        "attempt_accepted",
        "phase_hist_edges",
        "phase_hist_counts_right",
        "phase_hist_counts_left",
    }
    for key, values in arrays.items():
        if key in COMMON_KEYS or key in case_level_exceptions:
            continue
        if values.shape[0] != count:
            raise ValueError(
                f"{family}.{key} must have first dimension {count}, "
                f"got {values.shape}"
            )

    codes = np.asarray(arrays["cell_code"], dtype=np.int64)
    if np.any(codes < 0) or np.any(codes >= len(cell_order)):
        raise ValueError(f"{family} cell_code lies outside declared cell_order")
    if np.unique(np.asarray(arrays["case_id"], dtype=np.int64)).size != count:
        raise ValueError(f"{family} case_id values are not unique")
    if np.unique(np.asarray(arrays["attempt_index"], dtype=np.int64)).size != count:
        raise ValueError(f"{family} attempt_index values are not unique")

    return PopulationData(
        family=family,
        path=path,
        arrays=arrays,
        cell_order=cell_order,
        stream_id=_family_stream_id(summary, record),
    )


def balanced_expected_counts(total: int, cell_count: int) -> NDArray[np.int64]:
    """Return the ordered quotient--remainder allocation."""

    if total < 0 or cell_count <= 0:
        raise ValueError("total must be nonnegative and cell_count positive")
    quotient, remainder = divmod(total, cell_count)
    return np.asarray(
        [quotient + int(index < remainder) for index in range(cell_count)],
        dtype=np.int64,
    )


def empirical_cdf(values: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return sorted finite values and right-continuous empirical probabilities."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("an empirical CDF requires at least one finite value")
    ordered = np.sort(finite, kind="stable")
    probability = np.arange(1, ordered.size + 1, dtype=np.float64) / ordered.size
    return ordered, probability


def uniform_discrepancy(values: FloatArray) -> float:
    """Return ``sup_u |F_n(u)-u|`` for observations expected on ``[0,1]``."""

    ordered, upper = empirical_cdf(values)
    if np.any(ordered < 0.0) or np.any(ordered > 1.0):
        raise ValueError("uniform-transform values must lie in [0, 1]")
    lower = upper - 1.0 / ordered.size
    return float(
        max(
            np.max(np.abs(upper - ordered)),
            np.max(np.abs(lower - ordered)),
        )
    )


def dkw_half_width(sample_count: int, alpha: float = DKW_ALPHA) -> float:
    """Return the two-sided DKW half-width at confidence ``1-alpha``."""

    if sample_count <= 0 or not 0.0 < alpha < 1.0:
        raise ValueError("sample_count and alpha must define a valid DKW bound")
    return math.sqrt(math.log(2.0 / alpha) / (2.0 * sample_count))


def rank_unit(values: FloatArray) -> FloatArray:
    """Map finite values to deterministic midpoint ranks in ``(0,1)``."""

    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        raise ValueError("rank_unit requires a nonempty finite vector")
    order = np.argsort(data, kind="stable")
    ranks = np.empty(data.size, dtype=np.float64)
    ranks[order] = (np.arange(data.size, dtype=np.float64) + 0.5) / data.size
    return ranks


def rank_medoid_index(features: FloatArray, attempt_index: IntArray) -> int:
    """Select the case nearest the componentwise midpoint in empirical ranks."""

    matrix = np.asarray(features, dtype=np.float64)
    attempts = np.asarray(attempt_index, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != attempts.size or matrix.shape[0] == 0:
        raise ValueError("features and attempt_index must have matching rows")
    transformed = np.column_stack(
        tuple(rank_unit(matrix[:, column]) for column in range(matrix.shape[1]))
    )
    distance = np.sum((transformed - 0.5) ** 2, axis=1)
    order = np.lexsort((attempts, distance))
    return int(order[0])


def _cell_rank_distance(data: PopulationData, index: int) -> float:
    """Return a selected row's rank-midpoint distance within its cell."""

    features = _selection_features(data)
    codes = np.asarray(data.arrays["cell_code"], dtype=np.int64)
    members = np.flatnonzero(codes == codes[index])
    local_positions = np.flatnonzero(members == index)
    if local_positions.size != 1:
        raise RuntimeError("selected cell representative is not in its cell")
    local = int(local_positions[0])
    transformed = np.column_stack(
        tuple(
            rank_unit(features[members, column])
            for column in range(features.shape[1])
        )
    )
    return float(np.sum((transformed[local] - 0.5) ** 2))


def unique_score_selection(
    *,
    role: str,
    score_name: str,
    scores: FloatArray,
    attempt_index: IntArray,
    excluded: set[int],
    maximize: bool,
    eligible: NDArray[np.bool_] | None = None,
) -> RoleSelection:
    """Select one finite score with deterministic attempt-index tie breaking."""

    values = np.asarray(scores, dtype=np.float64)
    attempts = np.asarray(attempt_index, dtype=np.int64)
    if values.shape != attempts.shape:
        raise ValueError("scores and attempt_index must have equal shape")
    allowed = np.isfinite(values)
    if eligible is not None:
        eligibility = np.asarray(eligible, dtype=np.bool_)
        if eligibility.shape != values.shape:
            raise ValueError("eligible must have the score shape")
        allowed &= eligibility
    if excluded:
        allowed[np.asarray(sorted(excluded), dtype=np.int64)] = False
    indices = np.flatnonzero(allowed)
    if indices.size == 0:
        raise ValueError(f"no eligible case remains for role {role!r}")
    primary = -values[indices] if maximize else values[indices]
    order = np.lexsort((attempts[indices], primary))
    index = int(indices[order[0]])
    return RoleSelection(
        role=role,
        index=index,
        score_name=score_name,
        score=float(values[index]),
    )


def _cell_names(data: PopulationData) -> NDArray[np.str_]:
    codes = np.asarray(data.arrays["cell_code"], dtype=np.int64)
    names = np.asarray(data.cell_order, dtype=np.str_)
    return names[codes]


def _uniform_transform(
    values: FloatArray,
    lower: FloatArray | float,
    upper: FloatArray | float,
) -> FloatArray:
    data = np.asarray(values, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    width = high - low
    if np.any(width <= 0.0):
        raise ValueError("uniform-transform intervals must have positive width")
    transformed = (data - low) / width
    tolerance = 32.0 * np.finfo(np.float64).eps
    if np.any(transformed < -tolerance) or np.any(transformed > 1.0 + tolerance):
        raise ValueError("value lies outside its uniform-transform interval")
    return np.clip(transformed, 0.0, 1.0)


def _log_uniform_transform(
    values: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
) -> FloatArray:
    data = np.asarray(values, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if np.any(data <= 0.0) or np.any(low <= 0.0) or np.any(high <= low):
        raise ValueError("log-uniform transform requires 0 < lower < upper")
    return _uniform_transform(np.log(data), np.log(low), np.log(high))


def _plot_uniform_ecdfs(
    axis: plt.Axes,
    transformed: Mapping[str, FloatArray],
) -> dict[str, dict[str, float | int]]:
    """Plot transformed ECDFs and return their exact discrepancies."""

    result: dict[str, dict[str, float | int]] = {}
    axis.plot([0.0, 1.0], [0.0, 1.0], color="black", lw=0.9, label="uniform law")
    for label, values in transformed.items():
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        ordered, probability = empirical_cdf(finite)
        discrepancy = uniform_discrepancy(finite)
        axis.plot(
            ordered,
            probability,
            lw=1.15,
            label=rf"{label}: $D_n={discrepancy:.4f}$",
        )
        result[label] = {
            "count": int(finite.size),
            "uniform_discrepancy": discrepancy,
            "dkw_95_half_width": dkw_half_width(int(finite.size)),
        }
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("transformed coordinate")
    axis.set_ylabel("empirical CDF")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7)
    return result


def _hexbin(
    axis: plt.Axes,
    x: FloatArray,
    y: FloatArray,
    *,
    xlabel: str,
    ylabel: str,
    gridsize: int = 60,
    xscale: str = "linear",
) -> None:
    """Plot an all-point logarithmic-count hexbin."""

    artist = axis.hexbin(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        gridsize=gridsize,
        mincnt=1,
        bins="log",
        cmap="viridis",
        rasterized=True,
        xscale=xscale,
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.12)
    colorbar = axis.figure.colorbar(artist, ax=axis, pad=0.02)
    colorbar.set_label("bin count (logarithmic color scale)")


def _save_figure(figure: plt.Figure, output_stem: Path) -> tuple[Path, Path]:
    """Save one figure in PNG and PDF form."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    figure.savefig(png, dpi=PLOT_DPI, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _annotated_cell_bars(axis: plt.Axes, data: PopulationData) -> None:
    """Plot exact observed and expected cell counts."""

    observed = data.cell_counts
    expected = balanced_expected_counts(data.count, len(data.cell_order))
    locations = np.arange(len(data.cell_order))
    axis.bar(locations, observed, color=FAMILY_COLORS[data.family], alpha=0.85)
    axis.plot(
        locations,
        expected,
        color="black",
        marker="_",
        linestyle="none",
        markersize=7,
        label="ordered quotient--remainder target",
    )
    axis.set_xticks(
        locations,
        [name.replace("__", "\n") for name in data.cell_order],
        rotation=45 if len(data.cell_order) > 8 else 20,
        ha="right",
        fontsize=6 if len(data.cell_order) > 12 else 8,
    )
    axis.set_ylabel("number of specifications")
    axis.set_title(FAMILY_LABELS[data.family])
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=7)


def plot_overview(
    populations: Mapping[str, PopulationData],
    output_dir: Path,
) -> tuple[tuple[Path, Path], JsonMap]:
    """Plot exact cell balance for all four populations."""

    figure, axes = plt.subplots(2, 2, figsize=(15.5, 10.0), constrained_layout=True)
    _annotated_cell_bars(axes[0, 0], populations["stokes"])
    _annotated_cell_bars(axes[0, 1], populations["tanaka"])

    bf = populations["benjamin_feir"]
    bf_modes = np.asarray(bf.arrays["carrier_mode"], dtype=np.int64)
    bf_offsets = np.asarray(bf.arrays["sideband_offset"], dtype=np.int64)
    carrier_values = np.arange(np.min(bf_modes), np.max(bf_modes) + 1)
    offset_values = np.arange(1, np.max(bf_offsets) + 1)
    bf_grid = np.full((carrier_values.size, offset_values.size), np.nan)
    for row, carrier in enumerate(carrier_values):
        for column, offset in enumerate(offset_values):
            selected = (bf_modes == carrier) & (bf_offsets == offset)
            if np.any(selected):
                bf_grid[row, column] = np.count_nonzero(selected)
    image = axes[1, 0].imshow(
        bf_grid,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
    )
    axes[1, 0].set_xticks(np.arange(offset_values.size), offset_values)
    axes[1, 0].set_yticks(np.arange(carrier_values.size), carrier_values)
    axes[1, 0].set_xlabel(r"sideband offset $\Delta n$")
    axes[1, 0].set_ylabel(r"carrier mode $n_c$")
    axes[1, 0].set_title("Benjamin--Feir: all 66 allocation cells")
    figure.colorbar(image, ax=axes[1, 0], pad=0.02, label="specifications")

    jonswap = populations["jonswap_tma"]
    jonswap_counts = jonswap.cell_counts
    if jonswap_counts.size == 27:
        grid = jonswap_counts.reshape(3, 9)
        image = axes[1, 1].imshow(
            grid,
            origin="upper",
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
        )
        axes[1, 1].set_yticks(np.arange(3), ("shallow", "finite", "deep"))
        axes[1, 1].set_xticks(
            np.arange(9),
            tuple(
                rf"$\gamma={gamma}$" + "\n" + rf"$r_d={direction}$"
                for gamma in ("1", "3.3", "5")
                for direction in ("0", "0.5", "1")
            ),
            fontsize=7,
        )
        axes[1, 1].set_title("JONSWAP/TMA: 27 allocation cells")
        figure.colorbar(image, ax=axes[1, 1], pad=0.02, label="specifications")
    else:
        _annotated_cell_bars(axes[1, 1], jonswap)

    total = sum(population.count for population in populations.values())
    figure.suptitle(
        f"Parameterized corpus population smoke: {total:,} specifications\n"
        "Every stored case contributes to these exact counts",
        fontsize=15,
    )
    paths = _save_figure(figure, output_dir / "overview_counts_support")
    record: JsonMap = {
        "description": (
            "Exact observed cell counts compared with the ordered "
            "quotient--remainder allocation."
        ),
        "total_specifications": total,
        "families": {
            family: {
                "count": population.count,
                "cell_order": list(population.cell_order),
                "observed_cell_counts": population.cell_counts.tolist(),
                "expected_cell_counts": balanced_expected_counts(
                    population.count,
                    len(population.cell_order),
                ).tolist(),
            }
            for family, population in populations.items()
        },
    }
    return paths, record


def plot_stokes(
    data: PopulationData,
    output_dir: Path,
) -> tuple[tuple[Path, Path], JsonMap]:
    """Plot all Stokes specifications and all amplitude attempts."""

    arrays = data.arrays
    names = _cell_names(data)
    finite = np.char.startswith(names, "finite")
    deep = np.char.startswith(names, "deep")
    kh = np.asarray(arrays["depth_wavenumber"], dtype=np.float64)
    steepness = np.asarray(arrays["steepness"], dtype=np.float64)
    ursell = np.asarray(arrays["ursell_upper_bound"], dtype=np.float64)
    redraw = np.asarray(arrays["support_redraw_count"], dtype=np.int64)

    figure, axes = plt.subplots(2, 3, figsize=(15.0, 8.4), constrained_layout=True)
    _hexbin(
        axes[0, 0],
        kh[finite],
        steepness[finite],
        xlabel=r"carrier depth $kh$",
        ylabel=r"carrier steepness $ka$",
    )
    kh_line = np.linspace(max(0.45, float(np.min(kh[finite]))), 5.0, 300)
    leading_boundary = STOKES_URSELL_LIMIT * kh_line**3 / (8.0 * math.pi**2)
    visible_boundary = leading_boundary <= 0.15
    axes[0, 0].plot(
        kh_line[visible_boundary],
        leading_boundary[visible_boundary],
        color="white",
        lw=1.4,
        ls="--",
        label=r"leading estimate: $\mathrm{Ur}=26$",
    )
    axes[0, 0].set_ylim(0.0, 0.155)
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].set_title("Finite-depth branch")

    _hexbin(
        axes[0, 1],
        kh[deep],
        steepness[deep],
        xlabel=r"carrier depth $kh$",
        ylabel=r"carrier steepness $ka$",
        xscale="log",
    )
    axes[0, 1].axvline(5.0, color="white", lw=1.4, ls="--")
    axes[0, 1].set_title("Deep-water branch")

    modes = np.asarray(arrays["carrier_mode"], dtype=np.int64)
    mode_values = np.arange(np.min(modes), np.max(modes) + 1)
    width = 0.8 / len(data.cell_order)
    for cell_code, cell_name in enumerate(data.cell_order):
        selected = np.asarray(arrays["cell_code"]) == cell_code
        counts = np.asarray(
            [np.count_nonzero(selected & (modes == mode)) for mode in mode_values]
        )
        axes[0, 2].bar(
            mode_values
            + (cell_code - (len(data.cell_order) - 1) / 2.0) * width,
            counts,
            width=width,
            label=cell_name,
        )
    axes[0, 2].set_xlabel("carrier mode")
    axes[0, 2].set_ylabel("specifications")
    axes[0, 2].set_title("Cell-conditioned mode laws")
    axes[0, 2].legend(fontsize=7)
    axes[0, 2].grid(axis="y", alpha=0.2)

    finite_ratio = ursell[finite] / STOKES_URSELL_LIMIT
    ordered, probability = empirical_cdf(finite_ratio)
    axes[1, 0].plot(ordered, probability, color="#4477AA")
    axes[1, 0].axvline(1.0, color="black", lw=1.0, ls="--")
    axes[1, 0].set_xlabel(r"accepted $\mathrm{Ur}_+/26$")
    axes[1, 0].set_ylabel("empirical CDF")
    axes[1, 0].set_title(
        "Finite-depth support margin\n"
        rf"$\min(1-\mathrm{{Ur}}_+/26)={np.min(1.0-finite_ratio):.3e}$"
    )
    axes[1, 0].grid(alpha=0.2)

    redraw_values, redraw_counts = np.unique(redraw, return_counts=True)
    axes[1, 1].bar(redraw_values, redraw_counts, color="#CC6677")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("rejected amplitudes before accepted amplitude")
    axes[1, 1].set_ylabel("specifications")
    axes[1, 1].set_title("Same-cell finite-depth redraws")
    axes[1, 1].grid(axis="y", which="both", alpha=0.2)

    depth_unit = _log_uniform_transform(
        np.asarray(arrays["depth"], dtype=np.float64),
        np.asarray(arrays["depth_draw_lower"], dtype=np.float64),
        np.asarray(arrays["depth_draw_upper"], dtype=np.float64),
    )
    attempt_unit = np.asarray(arrays["attempt_amplitude_unit"], dtype=np.float64)
    uniform_record = _plot_uniform_ecdfs(
        axes[1, 2],
        {
            "conditional log-depth": depth_unit,
            "every amplitude attempt": attempt_unit,
        },
    )
    axes[1, 2].set_title("Coordinates with declared uniform laws")

    figure.suptitle(
        f"Fifth-order Stokes population: all {data.count:,} specifications",
        fontsize=14,
    )
    paths = _save_figure(figure, output_dir / "stokes_population")
    record: JsonMap = {
        "description": (
            "All accepted Stokes specifications and every recorded amplitude "
            "attempt. The dashed finite-depth curve is the leading Ursell "
            "estimate, while acceptance used the exact conservative Ur_+."
        ),
        "count": data.count,
        "finite_count": int(np.count_nonzero(finite)),
        "deep_count": int(np.count_nonzero(deep)),
        "finite_ursell_ratio": _scalar_summary(finite_ratio),
        "support_redraw_count": _integer_summary(redraw),
        "uniform_coordinates": uniform_record,
    }
    return paths, record


def _simplex_coordinates(weights: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Map three-component simplex weights to an equilateral triangle."""

    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("simplex coordinates require shape (n, 3)")
    if np.any(values < 0.0) or not np.allclose(
        np.sum(values, axis=1),
        1.0,
        rtol=0.0,
        atol=2.0e-12,
    ):
        raise ValueError("simplex rows must be nonnegative and sum to one")
    return (
        values[:, 1] + 0.5 * values[:, 2],
        (math.sqrt(3.0) / 2.0) * values[:, 2],
    )


def _plot_simplex(
    axis: plt.Axes,
    weights: FloatArray,
    *,
    labels: tuple[str, str, str],
) -> None:
    x, y = _simplex_coordinates(weights)
    artist = axis.hexbin(
        x,
        y,
        gridsize=45,
        mincnt=1,
        bins="log",
        cmap="viridis",
        rasterized=True,
    )
    triangle_x = (0.0, 1.0, 0.5, 0.0)
    triangle_y = (0.0, 0.0, math.sqrt(3.0) / 2.0, 0.0)
    axis.plot(triangle_x, triangle_y, color="black", lw=0.8)
    axis.text(-0.02, -0.04, labels[0], ha="right", va="top")
    axis.text(1.02, -0.04, labels[1], ha="left", va="top")
    axis.text(0.5, math.sqrt(3.0) / 2.0 + 0.03, labels[2], ha="center")
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.figure.colorbar(artist, ax=axis, pad=0.02, label="bin count")


def _tanaka_parameter_units(
    data: PopulationData,
) -> tuple[FloatArray, FloatArray]:
    names = _cell_names(data)
    depth = np.asarray(data.arrays["depth"], dtype=np.float64)
    alpha = np.asarray(data.arrays["total_alpha"], dtype=np.float64)
    main = np.char.startswith(names, "main")
    depth_lower = np.where(main, 0.01, 0.20)
    depth_upper = np.where(main, 0.30, 0.35)
    alpha_lower = np.where(main, 0.10, 0.25)
    alpha_upper = np.where(main, 0.35, 0.45)
    return (
        _log_uniform_transform(depth, depth_lower, depth_upper),
        _uniform_transform(alpha, alpha_lower, alpha_upper),
    )


def plot_tanaka(
    data: PopulationData,
    output_dir: Path,
) -> tuple[tuple[Path, Path], JsonMap]:
    """Plot every Tanaka parameter specification."""

    arrays = data.arrays
    names = _cell_names(data)
    main = np.char.startswith(names, "main")
    steep = np.char.startswith(names, "steep")
    crest_count = np.asarray(arrays["crest_count"], dtype=np.int64)
    depth = np.asarray(arrays["depth"], dtype=np.float64)
    total_alpha = np.asarray(arrays["total_alpha"], dtype=np.float64)
    crest_alpha = np.asarray(arrays["crest_alpha"], dtype=np.float64)
    crest_gap = np.asarray(arrays["crest_gap"], dtype=np.float64)
    required = np.asarray(arrays["required_separation"], dtype=np.float64)
    achieved = np.asarray(arrays["achieved_separation"], dtype=np.float64)

    figure, axes = plt.subplots(2, 3, figsize=(15.0, 8.5), constrained_layout=True)
    _hexbin(
        axes[0, 0],
        depth[main],
        total_alpha[main],
        xlabel=r"depth $h$",
        ylabel=r"total dimensionless amplitude $\sum_i\alpha_i$",
        xscale="log",
    )
    axes[0, 0].set_title("Main cells")
    _hexbin(
        axes[0, 1],
        depth[steep],
        total_alpha[steep],
        xlabel=r"depth $h$",
        ylabel=r"dimensionless amplitude $\alpha$",
    )
    axes[0, 1].set_title("Steep one-crest cell")

    three = crest_count == 3
    alpha_weights = crest_alpha[three, :3] / total_alpha[three, None]
    _plot_simplex(
        axes[0, 2],
        alpha_weights,
        labels=(r"$\alpha_1$", r"$\alpha_2$", r"$\alpha_3$"),
    )
    valid_alpha = (
        np.arange(3, dtype=np.int64)[None, :] < crest_count[:, None]
    )
    individual_alpha = crest_alpha[valid_alpha]
    small_alpha_fraction = float(np.mean(individual_alpha < 0.01))
    axes[0, 2].set_title(
        "Three-crest amplitude split\n"
        rf"all crests: fraction with $\alpha_i<0.01$ "
        f"= {small_alpha_fraction:.3%}"
    )

    gap_slack = np.empty((np.count_nonzero(three), 3), dtype=np.float64)
    three_indices = np.flatnonzero(three)
    for row, index in enumerate(three_indices):
        denominator = DOMAIN_LENGTH - 3.0 * required[index]
        gap_slack[row] = (crest_gap[index, :3] - required[index]) / denominator
    _plot_simplex(
        axes[1, 0],
        gap_slack,
        labels=(r"$g_1-3h$", r"$g_2-3h$", r"$g_3-3h$"),
    )
    axes[1, 0].set_title("Three-crest allocation of separation slack")

    multiple = crest_count > 1
    separation_margin = (achieved[multiple] - required[multiple]) / DOMAIN_LENGTH
    ordered, probability = empirical_cdf(separation_margin)
    axes[1, 1].plot(ordered, probability, color="#228833")
    axes[1, 1].axvline(0.0, color="black", ls="--", lw=0.9)
    axes[1, 1].set_xlabel(r"$(\min_i g_i-3h)/L$")
    axes[1, 1].set_ylabel("empirical CDF")
    axes[1, 1].set_title(
        "Periodic separation margin\n"
        rf"minimum $={np.min(separation_margin):.3e}$"
    )
    axes[1, 1].grid(alpha=0.2)

    depth_unit, alpha_unit = _tanaka_parameter_units(data)
    uniform_record = _plot_uniform_ecdfs(
        axes[1, 2],
        {
            "cell-conditioned log-depth": depth_unit,
            "cell-conditioned total amplitude": alpha_unit,
        },
    )
    right = np.asarray(arrays["crest_direction"], dtype=np.int64) == 1
    valid_direction = np.arange(3)[None, :] < crest_count[:, None]
    right_fraction = float(np.count_nonzero(right & valid_direction)) / float(
        np.count_nonzero(valid_direction)
    )
    axes[1, 2].set_title(
        "Coordinates with declared uniform laws\n"
        rf"right-moving crest fraction $={right_fraction:.4f}$"
    )

    figure.suptitle(
        f"Tanaka-profile population: all {data.count:,} specifications",
        fontsize=14,
    )
    paths = _save_figure(figure, output_dir / "tanaka_population")
    record: JsonMap = {
        "description": (
            "All Tanaka specifications. Simplex panels include every "
            "three-crest case; the separation margin includes every "
            "multi-crest case."
        ),
        "count": data.count,
        "crest_count_counts": {
            str(value): int(np.count_nonzero(crest_count == value))
            for value in np.unique(crest_count)
        },
        "separation_margin_over_length": _scalar_summary(separation_margin),
        "right_moving_crest_fraction": right_fraction,
        "individual_crest_alpha": {
            **_scalar_summary(individual_alpha),
            "fraction_below_0p01": small_alpha_fraction,
        },
        "uniform_coordinates": uniform_record,
    }
    return paths, record


def plot_benjamin_feir(
    data: PopulationData,
    output_dir: Path,
) -> tuple[tuple[Path, Path], JsonMap]:
    """Plot every Benjamin--Feir specification."""

    arrays = data.arrays
    carrier = np.asarray(arrays["carrier_mode"], dtype=np.int64)
    offset = np.asarray(arrays["sideband_offset"], dtype=np.int64)
    steepness = np.asarray(arrays["carrier_steepness"], dtype=np.float64)
    lower = np.asarray(
        arrays["conditional_steepness_lower"],
        dtype=np.float64,
    )
    ratio = np.asarray(arrays["perturbation_ratio"], dtype=np.float64)
    phase = np.asarray(arrays["sideband_phase"], dtype=np.float64)
    band_fraction = np.asarray(arrays["band_fraction"], dtype=np.float64)
    carrier_amplitude = np.asarray(arrays["carrier_amplitude"], dtype=np.float64)

    figure, axes = plt.subplots(2, 3, figsize=(15.0, 8.5), constrained_layout=True)
    carrier_values = np.arange(np.min(carrier), np.max(carrier) + 1)
    offset_values = np.arange(1, np.max(offset) + 1)
    grid = np.full((carrier_values.size, offset_values.size), np.nan)
    for row, carrier_value in enumerate(carrier_values):
        for column, offset_value in enumerate(offset_values):
            selected = (carrier == carrier_value) & (offset == offset_value)
            if np.any(selected):
                grid[row, column] = np.count_nonzero(selected)
    image = axes[0, 0].imshow(
        grid,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
    )
    axes[0, 0].set_xticks(np.arange(offset_values.size), offset_values)
    axes[0, 0].set_yticks(np.arange(carrier_values.size), carrier_values)
    axes[0, 0].set_xlabel(r"$\Delta n$")
    axes[0, 0].set_ylabel(r"$n_c$")
    axes[0, 0].set_title("Exact allocation over 66 mode pairs")
    figure.colorbar(image, ax=axes[0, 0], pad=0.02, label="specifications")

    modulation_ratio = offset / carrier
    _hexbin(
        axes[0, 1],
        steepness,
        modulation_ratio,
        xlabel=r"carrier steepness $\varepsilon_c$",
        ylabel=r"relative sideband offset $\Delta n/n_c$",
    )
    epsilon_line = np.linspace(0.05, 0.13, 200)
    axes[0, 1].plot(
        epsilon_line,
        2.0 * math.sqrt(2.0) * epsilon_line,
        color="white",
        ls="--",
        lw=1.4,
        label=r"$\Delta n/n_c=2\sqrt{2}\varepsilon_c$",
    )
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].set_title("Leading instability band")

    band_margin = 1.0 - band_fraction
    ordered, probability = empirical_cdf(band_margin)
    axes[0, 2].plot(ordered, probability, color="#CC6677")
    axes[0, 2].axvline(0.0, color="black", ls="--", lw=0.9)
    axes[0, 2].set_xlabel(r"instability-band margin $1-f$")
    axes[0, 2].set_ylabel("empirical CDF")
    axes[0, 2].set_title(
        rf"$f=(\Delta n/n_c)/(2\sqrt{{2}}\varepsilon_c)$"
        "\n"
        rf"minimum $={np.min(band_margin):.3e}$; fraction below $10^{{-3}}$ "
        f"= {np.mean(band_margin < 1.0e-3):.3%}"
    )
    axes[0, 2].grid(alpha=0.2)

    epsilon_unit = _uniform_transform(steepness, lower, 0.13)
    ratio_unit = _uniform_transform(ratio, 0.05, 0.20)
    phase_unit = _uniform_transform(phase, 0.0, 2.0 * math.pi)
    uniform_record = _plot_uniform_ecdfs(
        axes[1, 0],
        {
            "conditional steepness": epsilon_unit,
            "sideband ratio": ratio_unit,
            "common phase": phase_unit,
        },
    )
    axes[1, 0].set_title("Coordinates with declared uniform laws")

    _hexbin(
        axes[1, 1],
        carrier_amplitude,
        ratio,
        xlabel=r"carrier first-harmonic amplitude $a$",
        ylabel=r"sideband-to-carrier ratio $\rho$",
    )
    axes[1, 1].set_title("Dimensional scale and modulation amplitude")

    mode_counts = np.asarray(
        [np.count_nonzero(carrier == value) for value in carrier_values]
    )
    axes[1, 2].bar(carrier_values, mode_counts, color="#CC6677")
    axes[1, 2].set_xlabel(r"carrier mode $n_c$")
    axes[1, 2].set_ylabel("specifications")
    axes[1, 2].set_title("Marginal induced by equal pair-cell quotas")
    axes[1, 2].grid(axis="y", alpha=0.2)

    figure.suptitle(
        f"Benjamin--Feir population: all {data.count:,} specifications",
        fontsize=14,
    )
    paths = _save_figure(figure, output_dir / "benjamin_feir_population")
    record: JsonMap = {
        "description": (
            "All Benjamin--Feir specifications. Pair cells are balanced, "
            "so the carrier-mode marginal is intentionally not uniform."
        ),
        "count": data.count,
        "instability_band_margin": _scalar_summary(band_margin),
        "uniform_coordinates": uniform_record,
    }
    return paths, record


def shallow_chi_cdf(peak_depth: FloatArray) -> FloatArray:
    """Return the exact shallow-law marginal CDF of ``chi=k_p h``.

    The accepted set is uniform in area on
    ``0.2 <= chi <= 1.5``, ``0.03 <= delta <= 0.16``,
    ``chi * delta <= 0.15``.
    """

    chi = np.asarray(peak_depth, dtype=np.float64)
    lower = 0.2
    upper = 1.5
    transition = 0.15 / 0.16
    if np.any(chi < lower) or np.any(chi > upper):
        raise ValueError("shallow peak-depth values must lie in [0.2, 1.5]")

    def area_to(values: FloatArray) -> FloatArray:
        clipped = np.clip(values, lower, upper)
        rectangular = 0.13 * (np.minimum(clipped, transition) - lower)
        tail_upper = np.maximum(clipped, transition)
        tail = np.where(
            clipped > transition,
            0.15 * np.log(tail_upper / transition)
            - 0.03 * (tail_upper - transition),
            0.0,
        )
        return rectangular + tail

    total = float(area_to(np.asarray([upper]))[0])
    return np.asarray(area_to(chi) / total, dtype=np.float64)


def _jonswap_rectangular_margin(
    data: PopulationData,
) -> tuple[FloatArray, FloatArray]:
    arrays = data.arrays
    stratum = np.asarray(arrays["stratum_code"], dtype=np.int64)
    depth = np.asarray(arrays["depth"], dtype=np.float64)
    height = np.asarray(arrays["significant_height"], dtype=np.float64)
    peak = np.asarray(arrays["peak_wavenumber"], dtype=np.float64)
    rectangular = stratum != 0
    low_h = np.where(stratum[rectangular] == 1, 0.1, 5.0)
    high_h = np.where(stratum[rectangular] == 1, 1.5, 25.0)
    u_h = _uniform_transform(depth[rectangular], low_h, high_h)
    u_height = _uniform_transform(height[rectangular], 0.005, 0.03)
    u_peak = _uniform_transform(peak[rectangular], 2.0, 12.0)
    margin = np.min(
        np.column_stack(
            (u_h, 1.0 - u_h, u_height, 1.0 - u_height, u_peak, 1.0 - u_peak)
        ),
        axis=1,
    )
    return rectangular, margin


def plot_jonswap_tma(
    data: PopulationData,
    output_dir: Path,
) -> tuple[tuple[Path, Path], JsonMap]:
    """Plot every JONSWAP/TMA specification and all accumulated phases."""

    arrays = data.arrays
    stratum = np.asarray(arrays["stratum_code"], dtype=np.int64)
    shallow = stratum == 0
    finite = stratum == 1
    deep = stratum == 2
    depth = np.asarray(arrays["depth"], dtype=np.float64)
    height = np.asarray(arrays["significant_height"], dtype=np.float64)
    peak = np.asarray(arrays["peak_wavenumber"], dtype=np.float64)
    chi = np.asarray(arrays["peak_depth"], dtype=np.float64)
    delta = np.asarray(arrays["relative_height"], dtype=np.float64)
    peak_steepness = np.asarray(arrays["peak_steepness"], dtype=np.float64)

    figure, axes = plt.subplots(2, 3, figsize=(15.0, 8.5), constrained_layout=True)
    _hexbin(
        axes[0, 0],
        chi[shallow],
        delta[shallow],
        xlabel=r"peak depth $\chi=k_ph$",
        ylabel=r"relative height $\delta=H_s/(2h)$",
    )
    chi_line = np.linspace(0.2, 1.5, 300)
    axes[0, 0].plot(
        chi_line,
        np.minimum(0.16, 0.15 / chi_line),
        color="white",
        lw=1.4,
        ls="--",
        label=r"$\chi\delta=0.15$",
    )
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].set_title(
        "Shallow TMA: uniform area on truncated support\n"
        rf"$\max(k_pH_s/2)={np.max(peak_steepness[shallow]):.6f}$"
    )

    _hexbin(
        axes[0, 1],
        peak[finite],
        depth[finite],
        xlabel=r"peak wavenumber $k_p$",
        ylabel=r"depth $h$",
    )
    axes[0, 1].set_title("Finite-depth TMA")
    _hexbin(
        axes[0, 2],
        peak[deep],
        depth[deep],
        xlabel=r"peak wavenumber $k_p$",
        ylabel=r"depth $h$",
    )
    axes[0, 2].set_title("Deep JONSWAP")

    shallow_margin = (0.15 - peak_steepness[shallow]) / 0.15
    _, rectangular_margin = _jonswap_rectangular_margin(data)
    for values, label, color in (
        (shallow_margin, "shallow steepness boundary", "#EE6677"),
        (rectangular_margin, "finite/deep rectangular edges", "#4477AA"),
    ):
        ordered, probability = empirical_cdf(values)
        axes[1, 0].plot(ordered, probability, label=label, color=color)
    axes[1, 0].set_xlabel("dimensionless support margin")
    axes[1, 0].set_ylabel("empirical CDF")
    axes[1, 0].set_title("Distance to declared parameter boundaries")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(alpha=0.2)

    shallow_delta_upper = np.minimum(0.16, 0.15 / chi[shallow])
    shallow_delta_unit = _uniform_transform(
        delta[shallow],
        0.03,
        shallow_delta_upper,
    )
    rectangular, _ = _jonswap_rectangular_margin(data)
    rectangular_depth_lower = np.where(stratum[rectangular] == 1, 0.1, 5.0)
    rectangular_depth_upper = np.where(stratum[rectangular] == 1, 1.5, 25.0)
    uniform_record = _plot_uniform_ecdfs(
        axes[1, 1],
        {
            "shallow peak depth": shallow_chi_cdf(chi[shallow]),
            "shallow conditional height": shallow_delta_unit,
            "finite/deep depth": _uniform_transform(
                depth[rectangular],
                rectangular_depth_lower,
                rectangular_depth_upper,
            ),
            "finite/deep significant height": _uniform_transform(
                height[rectangular],
                0.005,
                0.03,
            ),
            "finite/deep peak wavenumber": _uniform_transform(
                peak[rectangular],
                2.0,
                12.0,
            ),
        },
    )
    axes[1, 1].set_title("Coordinates with declared uniform laws")

    phase_edges = np.asarray(arrays["phase_hist_edges"], dtype=np.float64)
    phase_right = np.asarray(arrays["phase_hist_counts_right"], dtype=np.int64)
    phase_left = np.asarray(arrays["phase_hist_counts_left"], dtype=np.int64)
    if phase_edges.size != phase_right.size + 1 or phase_right.shape != phase_left.shape:
        raise ValueError("phase histogram arrays have inconsistent shapes")
    axes[1, 2].stairs(
        phase_right / np.sum(phase_right),
        phase_edges,
        label="right-moving phases",
        color="#4477AA",
    )
    axes[1, 2].stairs(
        phase_left / np.sum(phase_left),
        phase_edges,
        label="left-moving phases",
        color="#CC6677",
    )
    axes[1, 2].axhline(
        1.0 / phase_right.size,
        color="black",
        lw=0.9,
        ls="--",
        label="uniform bin mass",
    )
    axes[1, 2].set_xlabel("phase")
    axes[1, 2].set_ylabel("fraction of all stored phases")
    axes[1, 2].set_title("All resolved-mode phases")
    axes[1, 2].legend(fontsize=7)
    axes[1, 2].grid(alpha=0.2)

    figure.suptitle(
        f"JONSWAP/TMA population: all {data.count:,} specifications",
        fontsize=14,
    )
    paths = _save_figure(figure, output_dir / "jonswap_tma_population")
    record: JsonMap = {
        "description": (
            "All JONSWAP/TMA specifications and the accumulated histogram "
            "of every resolved-mode phase."
        ),
        "count": data.count,
        "stratum_counts": {
            "shallow": int(np.count_nonzero(shallow)),
            "finite": int(np.count_nonzero(finite)),
            "deep": int(np.count_nonzero(deep)),
        },
        "shallow_steepness_margin": _scalar_summary(shallow_margin),
        "rectangular_support_margin": _scalar_summary(rectangular_margin),
        "uniform_coordinates": uniform_record,
        "phase_histogram": {
            "right_count": int(np.sum(phase_right)),
            "left_count": int(np.sum(phase_left)),
            "bin_edges": phase_edges.tolist(),
        },
    }
    return paths, record


def _scalar_summary(values: FloatArray) -> JsonMap:
    """Return finite extrema and selected quantiles."""

    data = np.asarray(values, dtype=np.float64)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("scalar summary requires finite values")
    quantiles = np.quantile(finite, (0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0))
    return {
        "count": int(finite.size),
        "minimum": float(quantiles[0]),
        "q01": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q99": float(quantiles[5]),
        "maximum": float(quantiles[6]),
    }


def _integer_summary(values: IntArray) -> JsonMap:
    """Return exact counts of integer values."""

    data = np.asarray(values, dtype=np.int64)
    unique, counts = np.unique(data, return_counts=True)
    return {
        "minimum": int(np.min(data)),
        "maximum": int(np.max(data)),
        "mean": float(np.mean(data)),
        "counts": {
            str(int(value)): int(count)
            for value, count in zip(unique, counts, strict=True)
        },
    }


def _selection_features(data: PopulationData) -> FloatArray:
    """Return elementary physical coordinates used only for rank medoids."""

    arrays = data.arrays
    if data.family == "stokes":
        return np.column_stack(
            (
                np.log(np.asarray(arrays["depth"], dtype=np.float64)),
                np.asarray(arrays["steepness"], dtype=np.float64),
                np.asarray(arrays["carrier_mode"], dtype=np.float64),
            )
        )
    if data.family == "tanaka":
        alpha = np.asarray(arrays["crest_alpha"], dtype=np.float64)
        total = np.asarray(arrays["total_alpha"], dtype=np.float64)
        imbalance = np.nanmax(alpha, axis=1) / total
        return np.column_stack(
            (
                np.log(np.asarray(arrays["depth"], dtype=np.float64)),
                total,
                np.asarray(arrays["crest_count"], dtype=np.float64),
                imbalance,
            )
        )
    if data.family == "benjamin_feir":
        return np.column_stack(
            (
                np.asarray(arrays["carrier_steepness"], dtype=np.float64),
                np.asarray(arrays["perturbation_ratio"], dtype=np.float64),
                np.asarray(arrays["carrier_amplitude"], dtype=np.float64),
            )
        )
    if data.family == "jonswap_tma":
        return np.column_stack(
            (
                np.log(np.asarray(arrays["depth"], dtype=np.float64)),
                np.asarray(arrays["significant_height"], dtype=np.float64),
                np.asarray(arrays["peak_wavenumber"], dtype=np.float64),
                np.asarray(arrays["peak_enhancement"], dtype=np.float64),
            )
        )
    raise ValueError(f"unknown family: {data.family}")


def select_role_cases(data: PopulationData) -> tuple[RoleSelection, ...]:
    """Select four unique cases using declared, deterministic parameter rules."""

    arrays = data.arrays
    attempts = np.asarray(arrays["attempt_index"], dtype=np.int64)
    features = _selection_features(data)
    medoid = rank_medoid_index(features, attempts)
    selections = [
        RoleSelection(
            role="rank medoid",
            index=medoid,
            score_name="sum of squared distances from rank midpoint",
            score=float(
                np.sum(
                    (
                        np.column_stack(
                            tuple(
                                rank_unit(features[:, column])
                                for column in range(features.shape[1])
                            )
                        )[medoid]
                        - 0.5
                    )
                    ** 2
                )
            ),
        )
    ]

    if data.family == "stokes":
        names = _cell_names(data)
        finite = np.char.startswith(names, "finite")
        deep = np.char.startswith(names, "deep")
        ursell = np.asarray(arrays["ursell_upper_bound"], dtype=np.float64)
        selections.append(
            unique_score_selection(
                role="maximum accepted finite Ursell ratio",
                score_name="1 - Ur_+/26",
                scores=1.0 - ursell / STOKES_URSELL_LIMIT,
                attempt_index=attempts,
                excluded=set(),
                maximize=False,
                eligible=finite,
            )
        )
        selections.append(
            unique_score_selection(
                role="largest same-cell amplitude redraw count",
                score_name="number of rejected amplitudes before acceptance",
                scores=np.asarray(
                    arrays["support_redraw_count"],
                    dtype=np.float64,
                ),
                attempt_index=attempts,
                excluded=set(),
                maximize=True,
            )
        )
        selections.append(
            unique_score_selection(
                role="largest carrier steepness",
                score_name="ka",
                scores=np.asarray(arrays["steepness"], dtype=np.float64),
                attempt_index=attempts,
                excluded=set(),
                maximize=True,
            )
        )
        selections.append(
            unique_score_selection(
                role="closest deep-water boundary",
                score_name="kh - 5",
                scores=np.asarray(
                    arrays["depth_wavenumber"],
                    dtype=np.float64,
                )
                - 5.0,
                attempt_index=attempts,
                excluded=set(),
                maximize=False,
                eligible=deep,
            )
        )
    elif data.family == "tanaka":
        depth_unit, alpha_unit = _tanaka_parameter_units(data)
        edge_margin = np.min(
            np.column_stack(
                (
                    depth_unit,
                    1.0 - depth_unit,
                    alpha_unit,
                    1.0 - alpha_unit,
                )
            ),
            axis=1,
        )
        selections.append(
            unique_score_selection(
                role="closest scalar support edge",
                score_name="minimum normalized depth/amplitude edge distance",
                scores=edge_margin,
                attempt_index=attempts,
                excluded=set(),
                maximize=False,
            )
        )
        alpha = np.asarray(arrays["crest_alpha"], dtype=np.float64)
        total = np.asarray(arrays["total_alpha"], dtype=np.float64)
        valid = (
            np.arange(alpha.shape[1], dtype=np.int64)[None, :]
            < np.asarray(arrays["crest_count"], dtype=np.int64)[:, None]
        )
        fractions = np.where(valid, alpha / total[:, None], np.nan)
        selections.append(
            unique_score_selection(
                role="smallest individual crest fraction",
                score_name="min_i alpha_i / sum_j alpha_j",
                scores=np.nanmin(fractions, axis=1),
                attempt_index=attempts,
                excluded=set(),
                maximize=False,
            )
        )
        selections.append(
            unique_score_selection(
                role="largest total dimensionless amplitude",
                score_name="sum_i alpha_i",
                scores=np.asarray(arrays["total_alpha"], dtype=np.float64),
                attempt_index=attempts,
                excluded=set(),
                maximize=True,
            )
        )
        multiple = np.asarray(arrays["crest_count"], dtype=np.int64) > 1
        separation = (
            np.asarray(arrays["achieved_separation"], dtype=np.float64)
            - np.asarray(arrays["required_separation"], dtype=np.float64)
        ) / DOMAIN_LENGTH
        selections.append(
            unique_score_selection(
                role="closest periodic-separation boundary",
                score_name="(min_i g_i - 3h)/L",
                scores=separation,
                attempt_index=attempts,
                excluded=set(),
                maximize=False,
                eligible=multiple,
            )
        )
    elif data.family == "benjamin_feir":
        selections.append(
            unique_score_selection(
                role="closest instability boundary",
                score_name="1 - band fraction",
                scores=1.0
                - np.asarray(arrays["band_fraction"], dtype=np.float64),
                attempt_index=attempts,
                excluded=set(),
                maximize=False,
            )
        )
        sideband_scale = np.asarray(
            arrays["carrier_amplitude"],
            dtype=np.float64,
        ) * np.asarray(arrays["perturbation_ratio"], dtype=np.float64)
        selections.append(
            unique_score_selection(
                role="largest sideband amplitude",
                score_name="rho a",
                scores=sideband_scale,
                attempt_index=attempts,
                excluded=set(),
                maximize=True,
            )
        )
        selections.append(
            unique_score_selection(
                role="largest relative sideband offset",
                score_name="Delta n / n_c",
                scores=np.asarray(
                    arrays["sideband_offset"],
                    dtype=np.float64,
                )
                / np.asarray(arrays["carrier_mode"], dtype=np.float64),
                attempt_index=attempts,
                excluded=set(),
                maximize=True,
            )
        )
    elif data.family == "jonswap_tma":
        stratum = np.asarray(arrays["stratum_code"], dtype=np.int64)
        shallow = stratum == 0
        selections.append(
            unique_score_selection(
                role="closest shallow steepness boundary",
                score_name="(0.15 - k_p H_s/2)/0.15",
                scores=(
                    0.15
                    - np.asarray(arrays["peak_steepness"], dtype=np.float64)
                )
                / 0.15,
                attempt_index=attempts,
                excluded=set(),
                maximize=False,
                eligible=shallow,
            )
        )
        selections.append(
            unique_score_selection(
                role="largest relative height",
                score_name="H_s/(2h)",
                scores=np.asarray(arrays["relative_height"], dtype=np.float64),
                attempt_index=attempts,
                excluded=set(),
                maximize=True,
            )
        )
        selections.append(
            unique_score_selection(
                role="largest peak enhancement",
                score_name="gamma",
                scores=np.asarray(
                    arrays["peak_enhancement"],
                    dtype=np.float64,
                ),
                attempt_index=attempts,
                excluded=set(),
                maximize=True,
            )
        )
    else:
        raise ValueError(f"unknown family: {data.family}")
    return tuple(selections)


def select_cell_representatives(data: PopulationData) -> tuple[int, ...]:
    """Select the empirical-rank medoid inside every declared allocation cell."""

    features = _selection_features(data)
    attempts = np.asarray(data.arrays["attempt_index"], dtype=np.int64)
    codes = np.asarray(data.arrays["cell_code"], dtype=np.int64)
    selected: list[int] = []
    for cell_code in range(len(data.cell_order)):
        members = np.flatnonzero(codes == cell_code)
        if members.size == 0:
            raise ValueError(f"cell {data.cell_order[cell_code]!r} is empty")
        local = rank_medoid_index(features[members], attempts[members])
        selected.append(int(members[local]))
    return tuple(selected)


def _assignment_for_index(
    data: PopulationData,
    index: int,
) -> AttemptAssignment:
    """Recreate the exact assignment for a stored population row."""

    family_id = {
        "stokes": PhysicalFamilyId.STOKES,
        "tanaka": PhysicalFamilyId.TANAKA,
        "benjamin_feir": PhysicalFamilyId.BENJAMIN_FEIR,
        "jonswap_tma": PhysicalFamilyId.JONSWAP_TMA,
    }[data.family]
    attempt_index = int(data.arrays["attempt_index"][index])
    cell_code = int(data.arrays["cell_code"][index])
    assignment = AttemptAssignment(
        case_key=CaseKey(
            family_id=int(family_id),
            revision_id=PAPER_CORPUS_REVISION_ID,
            split_id=SplitId.TEST,
            stream_id=data.stream_id,
            attempt_index=attempt_index,
        ),
        cell_id=data.cell_order[cell_code],
    )
    stored_case_id = int(data.arrays["case_id"][index])
    if assignment.case_key.case_id != stored_case_id:
        raise RuntimeError(
            f"{data.family} row {index} CaseKey gives "
            f"{assignment.case_key.case_id}, archive stores {stored_case_id}"
        )
    return assignment


def _assert_close(
    label: str,
    actual: float,
    expected: float,
    *,
    tolerance: float = 2.0e-13,
) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise RuntimeError(
            f"resampled {label} differs from archive: {actual} versus {expected}"
        )


def _resample_selected(
    data: PopulationData,
    indices: Sequence[int],
) -> tuple[object, ...]:
    """Replay selected cases from their stored CaseKey coordinates."""

    samples: list[object] = []
    band = ResolvedBand()
    for index in indices:
        assignment = _assignment_for_index(data, index)
        if data.family == "stokes":
            sample = sample_stokes_population(assignment)
            _assert_close(
                "Stokes depth",
                sample.depth,
                float(data.arrays["depth"][index]),
            )
            _assert_close(
                "Stokes amplitude",
                sample.amplitude,
                float(data.arrays["amplitude"][index]),
            )
        elif data.family == "tanaka":
            sample = sample_tanaka_population(assignment)
            _assert_close(
                "Tanaka depth",
                sample.depth,
                float(data.arrays["depth"][index]),
            )
            _assert_close(
                "Tanaka total alpha",
                sample.total_dimensionless_amplitude,
                float(data.arrays["total_alpha"][index]),
            )
        elif data.family == "benjamin_feir":
            sample = sample_benjamin_feir_population(assignment)
            _assert_close(
                "Benjamin--Feir steepness",
                sample.carrier_steepness,
                float(data.arrays["carrier_steepness"][index]),
            )
            _assert_close(
                "Benjamin--Feir perturbation ratio",
                sample.perturbation_ratio,
                float(data.arrays["perturbation_ratio"][index]),
            )
        elif data.family == "jonswap_tma":
            sample = sample_jonswap_tma_population(assignment, band=band)
            _assert_close(
                "JONSWAP/TMA depth",
                sample.parameters.depth,
                float(data.arrays["depth"][index]),
            )
            _assert_close(
                "JONSWAP/TMA significant height",
                sample.parameters.significant_height,
                float(data.arrays["significant_height"][index]),
            )
        else:
            raise ValueError(f"unknown family: {data.family}")
        samples.append(sample)
    return tuple(samples)


def _project_fields(
    eta: FloatArray | jax.Array,
    xi: FloatArray | jax.Array,
) -> tuple[FloatArray, FloatArray]:
    """Apply the common paper fixed-band input map."""

    eta_array = jnp.asarray(eta, dtype=jnp.float64)
    xi_array = jnp.asarray(xi, dtype=jnp.float64)
    if eta_array.ndim == 1:
        eta_array = eta_array[None, :]
        xi_array = xi_array[None, :]
    _, wavenumbers = build_grid(
        PAPER_DNO_TARGET.nx,
        PAPER_DNO_TARGET.length,
    )
    k = jnp.asarray(wavenumbers, dtype=jnp.float64)
    projected_eta = project_fixed_band(
        eta_array,
        k,
        maximum_wavenumber=PAPER_DNO_TARGET.maximum_wavenumber,
    )
    projected_xi = project_fixed_band(
        xi_array,
        k,
        maximum_wavenumber=PAPER_DNO_TARGET.maximum_wavenumber,
        remove_mean=True,
    )
    projected_xi.block_until_ready()
    return (
        np.asarray(jax.device_get(projected_eta), dtype=np.float64),
        np.asarray(jax.device_get(projected_xi), dtype=np.float64),
    )


def _parameter_text(sample: object) -> str:
    if isinstance(sample, StokesPopulationSample):
        return (
            rf"$n={sample.carrier_mode}$, $h={sample.depth:.4g}$, "
            rf"$ka={sample.steepness:.4g}$"
        )
    if isinstance(sample, TanakaPopulationSample):
        return (
            rf"$m={sample.cell.crest_count}$, $h={sample.depth:.4g}$, "
            rf"$\sum\alpha_i={sample.total_dimensionless_amplitude:.4g}$"
        )
    if isinstance(sample, BenjaminFeirPopulationSample):
        return (
            rf"$n_c={sample.cell.carrier_mode}$, "
            rf"$\Delta n={sample.cell.sideband_offset}$, "
            rf"$\varepsilon_c={sample.carrier_steepness:.4g}$"
        )
    if isinstance(sample, JonswapTmaPopulationSample):
        parameters = sample.parameters
        return (
            rf"$h={parameters.depth:.4g}$, $H_s={parameters.significant_height:.4g}$, "
            rf"$k_p={parameters.peak_wavenumber:.4g}$"
        )
    raise TypeError(f"unknown sampled type: {type(sample)}")


def _construct_selected_fields(
    data: PopulationData,
    indices: Sequence[int],
) -> dict[int, ConstructedField]:
    """Replay and construct a unique selected subset with production formulas."""

    unique_indices = tuple(dict.fromkeys(indices))
    samples = _resample_selected(data, unique_indices)
    x, _ = build_grid(PAPER_DNO_TARGET.nx, PAPER_DNO_TARGET.length)

    if data.family == "stokes":
        typed = tuple(
            sample
            for sample in samples
            if isinstance(sample, StokesPopulationSample)
        )
        if len(typed) != len(samples):
            raise TypeError("Stokes selection replay returned an incorrect type")
        raw = tuple(
            construct_stokes_state(sample, PAPER_STATIC_STOKES_CONTRACT)
            for sample in typed
        )
        eta_raw = jnp.stack(tuple(value[0] for value in raw))
        xi_raw = jnp.stack(tuple(value[1] for value in raw))
        depths = np.asarray([sample.depth for sample in typed], dtype=np.float64)
    elif data.family == "tanaka":
        typed = tuple(
            sample
            for sample in samples
            if isinstance(sample, TanakaPopulationSample)
        )
        if len(typed) != len(samples):
            raise TypeError("Tanaka selection replay returned an incorrect type")
        template = make_default_tanaka_template(
            depth=1.0,
            gravity=GRAVITY,
            direction=1,
            nx=PAPER_DNO_TARGET.nx,
            length=DOMAIN_LENGTH,
            center=0.0,
            dno_order=PAPER_DNO_TARGET.dno_order,
            pad_factor=PAPER_DNO_TARGET.pad_factor,
        )
        depths = np.asarray([sample.depth for sample in typed], dtype=np.float64)
        case_specs = [
            [
                CrestSpec(
                    amplitude=crest.alpha,
                    center=crest.center,
                    direction=crest.direction,
                )
                for crest in sample.crests
            ]
            for sample in typed
        ]
        eta_raw, xi_raw = build_per_case_initial_conditions(
            template_params=template,
            case_h_ref=depths,
            case_specs=case_specs,
            length=DOMAIN_LENGTH,
            nx=PAPER_DNO_TARGET.nx,
            gravity=GRAVITY,
        )
    elif data.family == "benjamin_feir":
        typed = tuple(
            sample
            for sample in samples
            if isinstance(sample, BenjaminFeirPopulationSample)
        )
        if len(typed) != len(samples):
            raise TypeError(
                "Benjamin--Feir selection replay returned an incorrect type"
            )
        per_case = tuple(sample.to_parameter_arrays() for sample in typed)
        parameter_names = tuple(per_case[0])
        parameters: ParameterArrays = {
            name: np.concatenate(
                tuple(case_parameters[name] for case_parameters in per_case)
            )
            for name in parameter_names
        }
        eta_raw, xi_raw = build_benjamin_feir_initial_conditions(
            x=jnp.asarray(x, dtype=jnp.float64),
            parameters=parameters,
            length=DOMAIN_LENGTH,
            gravity=GRAVITY,
            dtype=jnp.float64,
        )
        depths = np.asarray([sample.depth for sample in typed], dtype=np.float64)
    elif data.family == "jonswap_tma":
        typed = tuple(
            sample
            for sample in samples
            if isinstance(sample, JonswapTmaPopulationSample)
        )
        if len(typed) != len(samples):
            raise TypeError("JONSWAP/TMA selection replay returned an incorrect type")
        band = ResolvedBand()
        states = tuple(
            build_jonswap_tma_initial_condition(
                np.asarray(x, dtype=np.float64),
                parameters=sample.parameters,
                phase_right=sample.phase_right,
                phase_left=sample.phase_left,
                band=band,
                gravity=GRAVITY,
            )
            for sample in typed
        )
        eta_raw = np.stack(tuple(state.eta for state in states))
        xi_raw = np.stack(tuple(state.xi for state in states))
        depths = np.asarray(
            [sample.parameters.depth for sample in typed],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"unknown family: {data.family}")

    eta, xi = _project_fields(eta_raw, xi_raw)
    if eta.shape != (len(unique_indices), PAPER_DNO_TARGET.nx):
        raise RuntimeError(f"constructed {data.family} fields have shape {eta.shape}")
    if not np.isfinite(eta).all() or not np.isfinite(xi).all():
        raise RuntimeError(f"constructed {data.family} gallery contains nonfinite data")

    result: dict[int, ConstructedField] = {}
    for local, (index, sample) in enumerate(
        zip(unique_indices, samples, strict=True)
    ):
        result[index] = ConstructedField(
            index=index,
            case_id=int(data.arrays["case_id"][index]),
            attempt_index=int(data.arrays["attempt_index"][index]),
            cell_id=data.cell_order[int(data.arrays["cell_code"][index])],
            depth=float(depths[local]),
            eta=eta[local],
            xi=xi[local],
            parameter_text=_parameter_text(sample),
        )
    return result


def _one_sided_amplitude(field: FloatArray, scale: float) -> FloatArray:
    coefficients = np.fft.rfft(np.asarray(field, dtype=np.float64)) / field.size
    amplitude = np.abs(coefficients)
    if amplitude.size > 2:
        amplitude[1:-1] *= 2.0
    return np.asarray(amplitude / scale, dtype=np.float64)


def _field_record(
    field: ConstructedField,
    *,
    role: str,
    score_name: str,
    score: float,
) -> JsonMap:
    dx = DOMAIN_LENGTH / field.eta.size
    slope = np.fft.ifft(
        1j
        * (2.0 * math.pi * np.fft.fftfreq(field.eta.size, d=dx))
        * np.fft.fft(field.eta)
    ).real
    return {
        "role": role,
        "row_index": field.index,
        "case_id": field.case_id,
        "attempt_index": field.attempt_index,
        "cell_id": field.cell_id,
        "score_name": score_name,
        "score": score,
        "selection_uses_realized_field": False,
        "parameters": field.parameter_text,
        "field_diagnostics": {
            "minimum_water_column_over_depth": float(
                np.min(field.depth + field.eta) / field.depth
            ),
            "maximum_absolute_eta_over_depth": float(
                np.max(np.abs(field.eta)) / field.depth
            ),
            "maximum_absolute_dimensionless_xi": float(
                np.max(np.abs(field.xi))
                / (field.depth * math.sqrt(GRAVITY * field.depth))
            ),
            "maximum_absolute_slope": float(np.max(np.abs(slope))),
        },
    }


def _plot_role_gallery(
    data: PopulationData,
    selections: Sequence[RoleSelection],
    fields: Mapping[int, ConstructedField],
    output_dir: Path,
) -> tuple[tuple[Path, Path], list[JsonMap]]:
    """Plot four preselected fields with common physical normalizations."""

    figure, axes = plt.subplots(
        len(selections),
        3,
        figsize=(12.0, 2.55 * len(selections)),
        constrained_layout=True,
    )
    x_over_length = (
        np.arange(PAPER_DNO_TARGET.nx, dtype=np.float64)
        / PAPER_DNO_TARGET.nx
    )
    records: list[JsonMap] = []
    for row, selection in enumerate(selections):
        field = fields[selection.index]
        color = FAMILY_COLORS[data.family]
        eta_scale = field.depth
        xi_scale = field.depth * math.sqrt(GRAVITY * field.depth)
        axes[row, 0].plot(x_over_length, field.eta / eta_scale, color=color, lw=0.9)
        axes[row, 1].plot(x_over_length, field.xi / xi_scale, color=color, lw=0.9)
        amplitude = _one_sided_amplitude(field.eta, eta_scale)
        modes = np.arange(min(129, amplitude.size))
        axes[row, 2].semilogy(
            modes,
            np.maximum(amplitude[: modes.size], 1.0e-14),
            color=color,
            lw=0.9,
        )
        axes[row, 0].set_ylabel(r"$\eta/h$")
        axes[row, 1].set_ylabel(r"$\xi/(h\sqrt{gh})$")
        axes[row, 2].set_ylabel(r"one-sided $|\widehat\eta_k|/h$")
        axes[row, 0].set_title(
            f"{selection.role}; {field.cell_id}\n"
            f"case {field.case_id}; {field.parameter_text}",
            fontsize=8,
        )
        axes[row, 1].set_title(
            f"{selection.score_name} = {selection.score:.5g}",
            fontsize=8,
        )
        axes[row, 2].set_title(r"fixed delivered band $0\leq k\leq128$", fontsize=8)
        for axis in axes[row]:
            axis.grid(alpha=0.2)
        records.append(
            _field_record(
                field,
                role=selection.role,
                score_name=selection.score_name,
                score=selection.score,
            )
        )
    axes[-1, 0].set_xlabel(r"$x/L$")
    axes[-1, 1].set_xlabel(r"$x/L$")
    axes[-1, 2].set_xlabel("Fourier mode")
    figure.suptitle(
        f"{FAMILY_LABELS[data.family]}: deterministic parameter selections",
        fontsize=14,
    )
    return (
        _save_figure(figure, output_dir / f"{data.family}_field_gallery"),
        records,
    )


def _plot_cell_atlas(
    data: PopulationData,
    cell_indices: Sequence[int],
    fields: Mapping[int, ConstructedField],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Plot one rank-medoid elevation profile from every allocation cell."""

    columns = 2 if len(cell_indices) <= 4 else (6 if len(cell_indices) > 30 else 3)
    rows = math.ceil(len(cell_indices) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.2 * columns, 1.75 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    x_over_length = (
        np.arange(PAPER_DNO_TARGET.nx, dtype=np.float64)
        / PAPER_DNO_TARGET.nx
    )
    for axis, index in zip(axes.flat, cell_indices, strict=False):
        field = fields[index]
        axis.plot(
            x_over_length,
            field.eta / field.depth,
            color=FAMILY_COLORS[data.family],
            lw=0.75,
        )
        axis.set_title(field.cell_id.replace("__", "\n"), fontsize=7)
        axis.grid(alpha=0.15)
        axis.tick_params(labelsize=6)
    for axis in axes.flat[len(cell_indices) :]:
        axis.set_axis_off()
    figure.supxlabel(r"$x/L$")
    figure.supylabel(r"$\eta/h$")
    figure.suptitle(
        f"{FAMILY_LABELS[data.family]}: one parameter-rank medoid per cell",
        fontsize=13,
    )
    return _save_figure(figure, output_dir / f"{data.family}_cell_atlas")


def build_field_galleries(
    populations: Mapping[str, PopulationData],
    output_dir: Path,
) -> tuple[JsonMap, tuple[Path, ...]]:
    """Construct deterministic role cases and one representative per cell."""

    output_paths: list[Path] = []
    family_records: dict[str, object] = {}
    for family in FAMILIES:
        data = populations[family]
        roles = select_role_cases(data)
        cell_indices = select_cell_representatives(data)
        all_indices = tuple(
            dict.fromkeys(
                tuple(selection.index for selection in roles) + cell_indices
            )
        )
        fields = _construct_selected_fields(data, all_indices)
        role_paths, role_records = _plot_role_gallery(
            data,
            roles,
            fields,
            output_dir,
        )
        cell_paths = _plot_cell_atlas(
            data,
            cell_indices,
            fields,
            output_dir,
        )
        output_paths.extend(role_paths)
        output_paths.extend(cell_paths)
        family_records[family] = {
            "role_cases": role_records,
            "cell_representatives": [
                _field_record(
                    fields[index],
                    role="cell parameter-rank medoid",
                    score_name=(
                        "minimum squared distance from componentwise empirical-"
                        "rank midpoint inside this cell"
                    ),
                    score=_cell_rank_distance(data, index),
                )
                for index in cell_indices
            ],
        }
    record: JsonMap = {
        "construction": {
            "grid_size": PAPER_DNO_TARGET.nx,
            "domain_length": PAPER_DNO_TARGET.length,
            "gravity": GRAVITY,
            "dno_order": PAPER_DNO_TARGET.dno_order,
            "pad_factor": PAPER_DNO_TARGET.pad_factor,
            "fixed_band_maximum_wavenumber": (
                PAPER_DNO_TARGET.maximum_wavenumber
            ),
            "case_replay": (
                "Each selected row is resampled from its TEST CaseKey using "
                f"revision {PAPER_CORPUS_REVISION_ID}, the family stream_id, "
                "stored attempt_index, and stored parameter category. Selected "
                "scalar parameters are checked against the archive before "
                "construction."
            ),
            "role_selection": (
                "The medoid minimizes squared distance from 1/2 after replacing "
                "each named parameter by its empirical midpoint rank. Every tail "
                "role independently orders one explicit stored parameter or "
                "support-margin score and breaks ties by attempt_index. A row may "
                "therefore occupy two roles when it is the exact extremizer of "
                "both. No realized field is inspected during selection."
            ),
            "cell_selection": (
                "Within each declared cell, choose the case minimizing squared "
                "distance from the componentwise empirical-rank midpoint; break "
                "ties by attempt_index."
            ),
            "normalization": {
                "surface_elevation": "eta/h",
                "surface_potential": "xi/(h sqrt(g h))",
                "elevation_spectrum": "one-sided |eta_hat_k|/h",
            },
        },
        "families": family_records,
    }
    return record, tuple(output_paths)


def _artifact_record(path: Path, root: Path) -> JsonMap:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_plot_summary(
    output_dir: Path,
    *,
    input_dir: Path,
    input_summary: Path,
    populations: Mapping[str, PopulationData],
    figure_records: Mapping[str, JsonMap],
    output_paths: Sequence[Path],
    gallery_record: JsonMap,
) -> Path:
    """Write a strict machine-readable definition of every plot."""

    record: JsonMap = {
        "schema": "paper_population_smoke_plots_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "plot_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "input_role": (
            "parameter-population smoke; no trajectory or DNO-label claim"
        ),
        "all_point_statement": (
            "Every stored specification contributes to its family count, "
            "histogram, hexbin, empirical CDF, or exact phase histogram. "
            "Only the explicitly recorded deterministic subset is reconstructed "
            "as a physical field."
        ),
        "input": {
            "directory": str(input_dir.resolve()),
            "summary": {
                "path": str(input_summary.resolve()),
                "sha256": _sha256(input_summary),
            },
            "families": {
                family: {
                    "count": population.count,
                    "stream_id": population.stream_id,
                    "cell_order": list(population.cell_order),
                    "archive": {
                        "path": str(population.path.resolve()),
                        "bytes": population.path.stat().st_size,
                        "sha256": _sha256(population.path),
                    },
                }
                for family, population in populations.items()
            },
        },
        "distribution_figures": dict(figure_records),
        "field_gallery": gallery_record,
        "output_artifacts": [
            _artifact_record(path, output_dir) for path in output_paths
        ],
        "definitions": {
            "uniform_discrepancy": "D_n = sup_{0<=u<=1} |F_n(u)-u|",
            "dkw_95_half_width": (
                "sqrt(log(2/0.05)/(2n)); plotted discrepancies are descriptive, "
                "not p-values"
            ),
            "stokes_finite_margin": "1 - Ur_+/26",
            "tanaka_separation_margin": "(min_i g_i - 3h)/L",
            "benjamin_feir_margin": (
                "1 - (Delta n/n_c)/(2 sqrt(2) epsilon_c)"
            ),
            "jonswap_shallow_margin": "(0.15-k_p H_s/2)/0.15",
        },
    }
    path = output_dir / "plot_summary.json"
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    """Load the smoke, create all distribution figures, and write an audit."""

    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_dir / "figures"
    )
    summary_path = input_dir / "summary.json"
    sentinel_path = input_dir / "sentinel_records.json"
    summary = _strict_json_object(summary_path)
    _require_current_revision(summary)
    if not sentinel_path.exists():
        raise FileNotFoundError(f"missing sentinel records: {sentinel_path}")
    _strict_json_object(sentinel_path)
    populations = {
        family: load_population(input_dir, summary, family)
        for family in FAMILIES
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    figure_records: dict[str, JsonMap] = {}
    overview_paths, figure_records["overview"] = plot_overview(
        populations,
        output_dir,
    )
    output_paths.extend(overview_paths)
    for family, plotter in (
        ("stokes", plot_stokes),
        ("tanaka", plot_tanaka),
        ("benjamin_feir", plot_benjamin_feir),
        ("jonswap_tma", plot_jonswap_tma),
    ):
        paths, record = plotter(populations[family], output_dir)
        output_paths.extend(paths)
        figure_records[family] = record

    # Field reconstruction is filled by the deterministic gallery pass below.
    gallery_record, gallery_paths = build_field_galleries(
        populations,
        output_dir,
    )
    output_paths.extend(gallery_paths)
    plot_summary = _write_plot_summary(
        output_dir,
        input_dir=input_dir,
        input_summary=summary_path,
        populations=populations,
        figure_records=figure_records,
        output_paths=output_paths,
        gallery_record=gallery_record,
    )
    print(plot_summary)
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
