"""Historical July-15 trajectory-disjoint manuscript evaluation panels.

The historical :mod:`solver.evals.eval_suite` panels take the first trajectories
from the same rollout archives that supplied training snapshots.  This script
instead replayed each production IC *distribution* with a new deterministic
``SeedSequence`` namespace and stores ICs only.  No truth or surrogate rollout
is performed here.  The stored panels remain immutable historical evaluation
artifacts.  The Stokes generator has since changed to the paper-corpus
phase/steepness/Ursell contract, so this script deliberately refuses to
rebuild the old Stokes panels under new semantics.

The output schema is intentionally small and uniform.  Each
``<regime>_ics.npz`` contains ``eta``, ``xi``, ``depth``, ``case_ids``, and a
UTF-8 ``meta.json`` scalar.  A directory-level ``manifest.json`` records file
and content hashes.

Typical use (CPU is the default so panel construction cannot contend with an
evaluation job):

    uv run python -m solver.evals.build_manuscript_ic_panels \
        --output_dir data/manuscript_ic_panels_v1_20260715 --n_ics 256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
BuilderKind: TypeAlias = Literal["tanaka", "bf", "random_sea", "linear", "stokes"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = DATA_DIR / "manuscript_ic_panels_v1_20260715"

# This namespace is absent from all production-source metadata used by v9.
# Individual families receive distinct streams below, so g0/g1/modal remain
# independent replicates without reusing any production RNG state.
DEFAULT_MASTER_SEED = 20_260_715
STREAM_BASE = 2_026_071_500
CASE_ID_BASE = 90_000_000
CASE_ID_STRIDE = 1_000_000


@dataclass(frozen=True)
class FamilyConfig:
    name: str
    builder: BuilderKind
    production_source: Path
    distribution_source: Path
    replicate_group: str
    stream_index: int
    dt: float
    tmax: float
    substeps: int
    truth_note: str
    stokes_regime: Literal["deep", "shallow"] | None = None


@dataclass(frozen=True)
class BuiltPanel:
    eta: np.ndarray
    xi: np.ndarray
    depth: np.ndarray
    specs: JsonValue
    construction: dict[str, JsonValue]


FAMILIES: tuple[FamilyConfig, ...] = (
    FamilyConfig(
        "tanaka_g0", "tanaka",
        DATA_DIR / "tanaka_2_adaptive_g0.npz",
        DATA_DIR / "tanaka_2_adaptive_g0.npz",
        "tanaka_v2_sum_budgeted", 0, 0.8, 200.0, 80,
        "Nonlinear order-6 Craig--Sulem truth under the GL2 integrating-factor solver.",
    ),
    FamilyConfig(
        "tanaka_g1", "tanaka",
        DATA_DIR / "tanaka_2_adaptive_g1.npz",
        DATA_DIR / "tanaka_2_adaptive_g1.npz",
        "tanaka_v2_sum_budgeted", 1, 0.8, 200.0, 80,
        "Nonlinear order-6 Craig--Sulem truth under the GL2 integrating-factor solver.",
    ),
    FamilyConfig(
        "bf_g0", "bf",
        DATA_DIR / "bf_2_adaptive_g0.npz",
        DATA_DIR / "bf_2_adaptive_g0.npz",
        "bf_v2", 2, 0.8, 200.0, 80,
        "Nonlinear order-6 Craig--Sulem truth under the GL2 integrating-factor solver.",
    ),
    FamilyConfig(
        "bf_g1", "bf",
        DATA_DIR / "bf_2_adaptive_g1.npz",
        DATA_DIR / "bf_2_adaptive_g1.npz",
        "bf_v2", 3, 0.8, 200.0, 80,
        "Nonlinear order-6 Craig--Sulem truth under the GL2 integrating-factor solver.",
    ),
    FamilyConfig(
        "bf_modal", "bf",
        DATA_DIR / "bf_2_adaptive_modal_shard_00.npz",
        # The Modal shard's abbreviated meta omits the IC ranges. modal_bf.py
        # invokes generate_bf_dataset with its defaults, which are exactly the
        # g0/g1 distribution recorded in this full production meta.
        DATA_DIR / "bf_2_adaptive_g0.npz",
        "bf_v2", 4, 0.8, 200.0, 80,
        "Nonlinear order-6 Craig--Sulem truth under the GL2 integrating-factor solver.",
    ),
    FamilyConfig(
        "linear", "linear",
        DATA_DIR / "linear.npz", DATA_DIR / "linear.npz",
        "linear_single_mode", 5, 0.08, 20.0, 8,
        "Current protocol: nonlinear order-6 GL2 truth from a nominal linear-wave IC; not analytic linear propagation.",
    ),
    FamilyConfig(
        "random_sea_deep", "random_sea",
        DATA_DIR / "random_sea_deep.npz", DATA_DIR / "random_sea_deep.npz",
        "random_sea_deep_v2", 6, 0.08, 20.0, 8,
        "Nonlinear order-6 Craig--Sulem truth under the GL2 integrating-factor solver.",
    ),
    FamilyConfig(
        "random_sea_finite", "random_sea",
        DATA_DIR / "random_sea_finite.npz", DATA_DIR / "random_sea_finite.npz",
        "random_sea_finite_v2", 7, 0.08, 20.0, 8,
        "Nonlinear order-6 Craig--Sulem truth under the GL2 integrating-factor solver.",
    ),
    FamilyConfig(
        "stokes_deep", "stokes",
        DATA_DIR / "stokes_deep.npz", DATA_DIR / "stokes_deep.npz",
        "stokes_analytic_deep", 8, 0.08, 20.0, 8,
        "Current protocol: nonlinear order-6 GL2 truth initialized from a random-phase fifth-order Stokes snapshot.",
        stokes_regime="deep",
    ),
    FamilyConfig(
        "stokes_finite", "stokes",
        DATA_DIR / "stokes_finite.npz", DATA_DIR / "stokes_finite.npz",
        "stokes_analytic_finite", 9, 0.08, 20.0, 8,
        "Current protocol: nonlinear order-6 GL2 truth initialized from a random-phase fifth-order Stokes snapshot.",
        stokes_regime="shallow",
    ),
)


DISTRIBUTION_KEYS: dict[BuilderKind, tuple[str, ...]] = {
    "tanaka": (
        "nx", "length", "gravity", "min_crests", "max_crests",
        "separation_widths", "steepness_min", "steepness_max",
        "per_crest_steepness_floor", "depth_min", "depth_max",
        "depth_distribution", "filter_fraction", "zero_mean_xi",
    ),
    "bf": (
        "nx", "length", "gravity", "n_carr_min", "n_carr_max", "n_carr_set",
        "eps_carrier_min", "eps_carrier_max", "side_offset_min", "side_offset_max",
        "eps_pert_min", "eps_pert_max", "depth_min", "depth_max",
        "depth_distribution", "bf_2nd_order", "filter_fraction", "zero_mean_xi",
    ),
    "random_sea": (
        "nx", "length", "gravity", "Hs_lo", "Hs_hi", "kp_lo", "kp_hi",
        "bw_lo", "bw_hi", "depth_min", "depth_max", "depth_distribution",
        "spectrum_kind", "xi_dispersion", "filter_fraction", "zero_mean_xi",
    ),
    "linear": (
        "nx", "length", "gravity", "n0_min", "n0_max", "a0_min", "a0_max",
        "steepness_min", "steepness_max", "depth_min", "depth_max", "kh_min",
        "kh_max", "depth_distribution", "rejection_attempts", "dno_order",
        "pad_factor", "bidirectional",
    ),
    "stokes": (
        "nx", "length", "gravity", "regime", "ichoi", "n0_min", "n0_max",
        "a0_min", "a0_max", "steepness_max", "depth_min", "depth_max", "kh_min",
        "kh_max", "depth_distribution", "rejection_attempts",
        "dno_order", "pad_factor",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n_ics", type=int, default=256)
    parser.add_argument("--seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument(
        "--families", nargs="+", choices=[cfg.name for cfg in FAMILIES],
        default=[cfg.name for cfg in FAMILIES],
    )
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _json_safe(value: object) -> JsonValue:
    """Convert NumPy values and non-finite metadata to strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
        return "Infinity" if numeric > 0 else "-Infinity"
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    raise TypeError(f"cannot JSON-serialize value of type {type(value).__name__}")


def _strict_json_bytes(value: JsonValue, *, pretty: bool = False) -> bytes:
    indent = 2 if pretty else None
    separators = None if pretty else (",", ":")
    return json.dumps(
        value, sort_keys=True, indent=indent, separators=separators,
        allow_nan=False,
    ).encode("utf-8")


def _load_zip_meta(path: Path) -> tuple[dict[str, object], str]:
    if not path.exists():
        raise FileNotFoundError(f"missing production source: {path}")
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("meta.json")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}: meta.json is not an object")
    return parsed, hashlib.sha256(raw).hexdigest()


def _distribution_from_meta(meta: dict[str, object], kind: BuilderKind, path: Path) -> dict[str, object]:
    keys = DISTRIBUTION_KEYS[kind]
    missing = [key for key in keys if key not in meta]
    if missing:
        raise ValueError(f"{path}: production meta lacks required {kind} keys: {missing}")
    distribution = {key: meta[key] for key in keys}
    nx = int(distribution["nx"])
    length = float(distribution["length"])
    gravity = float(distribution["gravity"])
    if nx != 1024 or not math.isclose(length, 2.0 * math.pi) or gravity != 1.0:
        raise ValueError(
            f"{path}: unexpected reference grid/physics nx={nx}, length={length}, gravity={gravity}"
        )
    return distribution


def _seeded_rng(seed: int, stream_id: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, stream_id, 0]))


def _filter_initial_conditions(
    eta: object,
    xi: object,
    *,
    k: object,
    filter_fraction: float,
    zero_mean_xi: bool,
) -> tuple[np.ndarray, np.ndarray]:
    import jax
    import jax.numpy as jnp

    from solver.solvers.time_integrator import apply_lowpass

    eta_j = jnp.asarray(eta, dtype=jnp.float64)
    xi_j = jnp.asarray(xi, dtype=jnp.float64)
    if zero_mean_xi:
        xi_j = xi_j - jnp.mean(xi_j, axis=-1, keepdims=True)
    eta_j = apply_lowpass(eta_j, jnp.asarray(k, dtype=jnp.float64), filter_fraction)
    xi_j = apply_lowpass(xi_j, jnp.asarray(k, dtype=jnp.float64), filter_fraction)
    if zero_mean_xi:
        xi_j = xi_j - jnp.mean(xi_j, axis=-1, keepdims=True)
    eta_np, xi_np = jax.device_get((eta_j, xi_j))
    return np.asarray(eta_np, dtype=np.float64), np.asarray(xi_np, dtype=np.float64)


def _build_tanaka(
    distribution: dict[str, object], *, n_ics: int, seed: int, stream_id: int,
) -> BuiltPanel:
    import jax

    from solver.gen_data.generate_tanaka_dataset_v2 import (
        build_per_case_initial_conditions,
        sample_log_uniform,
    )
    from solver.gen_data.multi_crest import sample_sum_budgeted_cases, serialize_case_specs
    from solver.solvers.dno_series_jax import build_grid
    from solver.tanaka_ICs.modified_tanaka import make_default_tanaka_template

    nx = int(distribution["nx"])
    length = float(distribution["length"])
    gravity = float(distribution["gravity"])
    rng = _seeded_rng(seed, stream_id)
    depth = sample_log_uniform(
        rng, float(distribution["depth_min"]), float(distribution["depth_max"]), (n_ics,),
    )

    case_specs = []
    for depth_i in depth:
        min_separation = float(distribution["separation_widths"]) * float(depth_i)
        fit_max = max(1, math.floor(length / min_separation))
        max_crests = min(int(distribution["max_crests"]), fit_max)
        min_crests = min(int(distribution["min_crests"]), max_crests)
        case_specs.extend(
            sample_sum_budgeted_cases(
                rng, 1, length=length,
                case_amplitude_min=float(distribution["steepness_min"]),
                case_amplitude_max=float(distribution["steepness_max"]),
                min_crests=min_crests, max_crests=max_crests,
                min_separation=min_separation,
                per_crest_floor=float(distribution["per_crest_steepness_floor"]),
            )
        )

    template = make_default_tanaka_template(
        depth=1.0, gravity=gravity, direction=1, nx=nx, length=length,
        center=0.0, dno_order=6, pad_factor=8,
    )
    eta, xi = build_per_case_initial_conditions(
        template_params=template, case_h_ref=depth, case_specs=case_specs,
        length=length, nx=nx, gravity=gravity,
    )
    _, k = build_grid(nx, length)
    eta_np, xi_np = _filter_initial_conditions(
        eta, xi, k=k,
        filter_fraction=float(distribution["filter_fraction"]),
        zero_mean_xi=bool(distribution["zero_mean_xi"]),
    )
    jax.clear_caches()
    return BuiltPanel(
        eta_np, xi_np, np.asarray(depth, dtype=np.float64),
        _json_safe(serialize_case_specs(case_specs)),
        {
            "generator": "solver.gen_data.generate_tanaka_dataset_v2",
            "sampler": "sample_sum_budgeted_cases",
            "ic_builder": "build_per_case_initial_conditions",
            "dno_order_for_template": 6,
            "pad_factor_for_template": 8,
            "initial_lowpass_fraction": float(distribution["filter_fraction"]),
            "zero_mean_xi": bool(distribution["zero_mean_xi"]),
        },
    )


def _build_bf(
    distribution: dict[str, object], *, n_ics: int, seed: int, stream_id: int,
) -> BuiltPanel:
    import jax
    import jax.numpy as jnp

    from solver.gen_data.generate_bf_dataset import (
        build_bf_initial_conditions_batched,
        sample_bf_case_params,
        serialize_bf_specs,
    )
    from solver.solvers.dno_series_jax import build_grid

    rng = _seeded_rng(seed, stream_id)
    n_carr_raw = distribution["n_carr_set"]
    n_carr_set = tuple(int(value) for value in n_carr_raw) if isinstance(n_carr_raw, list) else None
    params = sample_bf_case_params(
        rng, batch_size=n_ics,
        n_carr_min=int(distribution["n_carr_min"]),
        n_carr_max=int(distribution["n_carr_max"]),
        n_carr_set=n_carr_set,
        eps_carrier_min=float(distribution["eps_carrier_min"]),
        eps_carrier_max=float(distribution["eps_carrier_max"]),
        side_offset_min=int(distribution["side_offset_min"]),
        side_offset_max=int(distribution["side_offset_max"]),
        eps_pert_min=float(distribution["eps_pert_min"]),
        eps_pert_max=float(distribution["eps_pert_max"]),
        depth_min=float(distribution["depth_min"]),
        depth_max=float(distribution["depth_max"]),
    )
    nx = int(distribution["nx"])
    length = float(distribution["length"])
    x, k = build_grid(nx, length)
    eta, xi = build_bf_initial_conditions_batched(
        x=jnp.asarray(x, dtype=jnp.float64), params=params,
        length=length, gravity=float(distribution["gravity"]),
        bf_2nd_order=bool(distribution["bf_2nd_order"]), dtype=jnp.float64,
    )
    eta_np, xi_np = _filter_initial_conditions(
        eta, xi, k=k,
        filter_fraction=float(distribution["filter_fraction"]),
        zero_mean_xi=bool(distribution["zero_mean_xi"]),
    )
    jax.clear_caches()
    return BuiltPanel(
        eta_np, xi_np, np.asarray(params["depth"], dtype=np.float64),
        _json_safe(serialize_bf_specs(params)),
        {
            "generator": "solver.gen_data.generate_bf_dataset",
            "sampler": "sample_bf_case_params",
            "ic_builder": "build_bf_initial_conditions_batched",
            "initial_lowpass_fraction": float(distribution["filter_fraction"]),
            "zero_mean_xi": bool(distribution["zero_mean_xi"]),
        },
    )


def _build_random_sea(
    distribution: dict[str, object], *, n_ics: int, seed: int, stream_id: int,
) -> BuiltPanel:
    import jax

    from solver.gen_data.generate_random_sea_dataset import (
        build_random_sea_initial_conditions,
        sample_random_sea_case_params,
        serialize_random_sea_specs,
    )
    from solver.solvers.dno_series_jax import build_grid

    rng = _seeded_rng(seed, stream_id)
    params = sample_random_sea_case_params(
        rng, batch_size=n_ics,
        Hs_lo=float(distribution["Hs_lo"]), Hs_hi=float(distribution["Hs_hi"]),
        kp_lo=float(distribution["kp_lo"]), kp_hi=float(distribution["kp_hi"]),
        bw_lo=float(distribution["bw_lo"]), bw_hi=float(distribution["bw_hi"]),
        depth_min=float(distribution["depth_min"]), depth_max=float(distribution["depth_max"]),
    )
    nx = int(distribution["nx"])
    _, k = build_grid(nx, float(distribution["length"]))
    eta, xi = build_random_sea_initial_conditions(
        rng, k_np=np.asarray(k, dtype=np.float64), params=params,
        gravity=float(distribution["gravity"]),
    )
    eta_np, xi_np = _filter_initial_conditions(
        eta, xi, k=k,
        filter_fraction=float(distribution["filter_fraction"]),
        zero_mean_xi=bool(distribution["zero_mean_xi"]),
    )
    jax.clear_caches()
    return BuiltPanel(
        eta_np, xi_np, np.asarray(params["depth"], dtype=np.float64),
        _json_safe(serialize_random_sea_specs(params)),
        {
            "generator": "solver.gen_data.generate_random_sea_dataset",
            "sampler": "sample_random_sea_case_params",
            "ic_builder": "build_random_sea_initial_conditions",
            "initial_lowpass_fraction": float(distribution["filter_fraction"]),
            "zero_mean_xi": bool(distribution["zero_mean_xi"]),
        },
    )


def _build_linear(
    distribution: dict[str, object], *, n_ics: int, seed: int, stream_id: int,
) -> BuiltPanel:
    import jax
    import jax.numpy as jnp

    from solver.gen_data.generate_linear_dataset import (
        build_linear_batch,
        sample_linear_case_params,
        serialize_specs,
    )
    from solver.solvers.dno_series_jax import build_grid

    rng = _seeded_rng(seed, stream_id)
    length = float(distribution["length"])
    params = sample_linear_case_params(
        rng, batch_size=n_ics, length=length,
        n0_min=int(distribution["n0_min"]), n0_max=int(distribution["n0_max"]),
        a0_min=float(distribution["a0_min"]), a0_max=float(distribution["a0_max"]),
        steepness_min=float(distribution["steepness_min"]),
        steepness_max=float(distribution["steepness_max"]),
        depth_min=float(distribution["depth_min"]), depth_max=float(distribution["depth_max"]),
        kh_min=float(distribution["kh_min"]), kh_max=float(distribution["kh_max"]),
        rejection_attempts=int(distribution["rejection_attempts"]),
    )
    x, k = build_grid(int(distribution["nx"]), length)
    eta, xi, gxi = build_linear_batch(
        x=jnp.asarray(x, dtype=jnp.float64), k=jnp.asarray(k, dtype=jnp.float64),
        length=length, gravity=float(distribution["gravity"]),
        n0=jnp.asarray(params["n0"], dtype=jnp.float64),
        a0=jnp.asarray(params["a0"], dtype=jnp.float64),
        depth=jnp.asarray(params["depth"], dtype=jnp.float64),
        phase=jnp.asarray(params["phase"], dtype=jnp.float64),
        direction=jnp.asarray(params["direction"], dtype=jnp.float64),
        dno_order=int(distribution["dno_order"]), pad_factor=int(distribution["pad_factor"]),
    )
    eta_np, xi_np, _ = jax.device_get((eta, xi, gxi))
    jax.clear_caches()
    return BuiltPanel(
        np.asarray(eta_np, dtype=np.float64), np.asarray(xi_np, dtype=np.float64),
        np.asarray(params["depth"], dtype=np.float64),
        _json_safe(serialize_specs(params)),
        {
            "generator": "solver.gen_data.generate_linear_dataset",
            "sampler": "sample_linear_case_params",
            "ic_builder": "build_linear_batch",
            "important_semantics": (
                "The production builder's xi includes its free-surface cosh correction. "
                "The manuscript eval deliberately retains nonlinear GL2 truth rather than "
                "the unused linear_analytic evaluator branch."
            ),
        },
    )


def _build_stokes(
    distribution: dict[str, object], *, n_ics: int, seed: int, stream_id: int,
) -> BuiltPanel:
    raise RuntimeError(
        "historical Stokes panel replay is frozen because the public "
        "generator now implements the paper-corpus sampler; use the existing "
        "July-15 panel or solver.gen_data.generate_stokes_dataset"
    )


def _build_panel(
    cfg: FamilyConfig,
    distribution: dict[str, object],
    *,
    n_ics: int,
    seed: int,
    stream_id: int,
) -> BuiltPanel:
    builders = {
        "tanaka": _build_tanaka,
        "bf": _build_bf,
        "random_sea": _build_random_sea,
        "linear": _build_linear,
        "stokes": _build_stokes,
    }
    return builders[cfg.builder](distribution, n_ics=n_ics, seed=seed, stream_id=stream_id)


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _case_digests(eta: np.ndarray, xi: np.ndarray, depth: np.ndarray) -> list[str]:
    return [
        _array_digest(eta[index], xi[index], depth[index : index + 1])
        for index in range(eta.shape[0])
    ]


def _validate_panel(panel: BuiltPanel, *, n_ics: int, nx: int, family: str) -> None:
    expected_2d = (n_ics, nx)
    if panel.eta.shape != expected_2d or panel.xi.shape != expected_2d:
        raise ValueError(
            f"{family}: expected eta/xi {expected_2d}, got {panel.eta.shape}/{panel.xi.shape}"
        )
    if panel.depth.shape != (n_ics,):
        raise ValueError(f"{family}: expected depth {(n_ics,)}, got {panel.depth.shape}")
    if not (
        np.isfinite(panel.eta).all()
        and np.isfinite(panel.xi).all()
        and np.isfinite(panel.depth).all()
        and np.all(panel.depth > 0.0)
    ):
        raise ValueError(f"{family}: generated panel contains invalid values")
    hashes = _case_digests(panel.eta, panel.xi, panel.depth)
    if len(set(hashes)) != n_ics:
        raise ValueError(f"{family}: generated panel contains duplicate IC states")


def _write_panel(
    path: Path,
    *,
    panel: BuiltPanel,
    case_ids: np.ndarray,
    meta: dict[str, JsonValue],
) -> None:
    payload = {
        "eta": np.ascontiguousarray(panel.eta, dtype=np.float64),
        "xi": np.ascontiguousarray(panel.xi, dtype=np.float64),
        "depth": np.ascontiguousarray(panel.depth, dtype=np.float64),
        "case_ids": np.ascontiguousarray(case_ids, dtype=np.int64),
        "meta.json": np.asarray(_strict_json_bytes(meta, pretty=True)),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path, manifest: dict[str, JsonValue]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_strict_json_bytes(manifest, pretty=True) + b"\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.n_ics <= 0:
        raise SystemExit("--n_ics must be positive")
    if args.n_ics >= CASE_ID_STRIDE:
        raise SystemExit(f"--n_ics must be less than {CASE_ID_STRIDE}")

    # Configure before importing JAX or any generator module.
    os.environ["JAX_PLATFORMS"] = args.platform
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    selected_names = set(args.families)
    selected = [cfg for cfg in FAMILIES if cfg.name in selected_names]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_paths = {cfg.name: output_dir / f"{cfg.name}_ics.npz" for cfg in selected}
    existing = [path for path in target_paths.values() if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise SystemExit(f"refusing to replace existing panels without --overwrite: {names}")

    script_sha = _file_sha256(Path(__file__))
    entries: list[JsonValue] = []
    all_case_hashes: set[str] = set()
    for cfg in selected:
        print(f"[{cfg.name}] loading production metadata", flush=True)
        production_meta, production_meta_sha = _load_zip_meta(cfg.production_source)
        distribution_meta, distribution_meta_sha = _load_zip_meta(cfg.distribution_source)
        distribution = _distribution_from_meta(distribution_meta, cfg.builder, cfg.distribution_source)
        stream_id = STREAM_BASE + cfg.stream_index
        print(
            f"[{cfg.name}] building {args.n_ics} fresh ICs on {args.platform} "
            f"with SeedSequence([{args.seed}, {stream_id}, 0])",
            flush=True,
        )
        panel = _build_panel(
            cfg, distribution, n_ics=args.n_ics, seed=args.seed, stream_id=stream_id,
        )
        nx = int(distribution["nx"])
        _validate_panel(panel, n_ics=args.n_ics, nx=nx, family=cfg.name)
        case_ids = np.arange(
            CASE_ID_BASE + cfg.stream_index * CASE_ID_STRIDE,
            CASE_ID_BASE + cfg.stream_index * CASE_ID_STRIDE + args.n_ics,
            dtype=np.int64,
        )
        case_hashes = _case_digests(panel.eta, panel.xi, panel.depth)
        duplicate_across_families = all_case_hashes.intersection(case_hashes)
        if duplicate_across_families:
            raise ValueError(f"{cfg.name}: IC state duplicates an earlier family")
        all_case_hashes.update(case_hashes)
        content_sha = _array_digest(panel.eta, panel.xi, panel.depth, case_ids)

        production_rel = cfg.production_source.relative_to(REPO_ROOT).as_posix()
        distribution_rel = cfg.distribution_source.relative_to(REPO_ROOT).as_posix()
        meta: dict[str, JsonValue] = {
            "schema": "manuscript_ic_panel_v1",
            "regime": cfg.name,
            "family": cfg.name,
            "replicate_group": cfg.replicate_group,
            "n_ics": args.n_ics,
            "nx": nx,
            "length": float(distribution["length"]),
            "gravity": float(distribution["gravity"]),
            "dt": cfg.dt,
            "tmax": cfg.tmax,
            "substeps": cfg.substeps,
            "inner_dt": cfg.dt / cfg.substeps,
            "implicit_iterations": 4,
            "filter_fraction": 0.25,
            "storage_dtype": "float64",
            "case_id_range": [int(case_ids[0]), int(case_ids[-1])],
            "content_sha256": content_sha,
            "case_state_sha256": case_hashes,
            "distribution": _json_safe(distribution),
            "case_specs": panel.specs,
            "truth_semantics": {
                "kind": "nonlinear",
                "dno_order": 6,
                "pad_factor": 8,
                "integrator": "gl2_if",
                "note": cfg.truth_note,
            },
            "provenance": {
                "trajectory_disjoint": True,
                "disjointness_basis": (
                    "Fresh IC parameters and phases are drawn from a dedicated SeedSequence "
                    "namespace absent from the v9 production-source metadata; no archived "
                    "training or prior-evaluation trajectory is selected."
                ),
                "master_seed": int(args.seed),
                "rng_stream_id": stream_id,
                "batch_index": 0,
                "seed_sequence_entropy": [int(args.seed), stream_id, 0],
                "production_source": production_rel,
                "production_source_meta_sha256": production_meta_sha,
                "production_source_meta": _json_safe(production_meta),
                "distribution_source": distribution_rel,
                "distribution_source_meta_sha256": distribution_meta_sha,
                "distribution_source_meta": _json_safe(distribution_meta),
                "script": "solver/evals/build_manuscript_ic_panels.py",
                "script_sha256": script_sha,
                "construction": _json_safe(panel.construction),
                "replicate_semantics": (
                    "g0/g1 (and bf_modal) are independent RNG replicates of one physical "
                    "distribution, not distinct physical regimes."
                    if cfg.replicate_group in {"tanaka_v2_sum_budgeted", "bf_v2"}
                    else "This label denotes a distinct production IC distribution."
                ),
                "bf_modal_note": (
                    "modal_bf.py invokes generate_bf_dataset with the same default IC ranges "
                    "as bf_g0/g1; its shard meta is abbreviated, so bf_g0's full meta is the "
                    "audited distribution source."
                    if cfg.name == "bf_modal" else None
                ),
            },
        }
        path = target_paths[cfg.name]
        _write_panel(path, panel=panel, case_ids=case_ids, meta=meta)
        file_sha = _file_sha256(path)
        entries.append(
            {
                "regime": cfg.name,
                "file": path.name,
                "bytes": path.stat().st_size,
                "file_sha256": file_sha,
                "content_sha256": content_sha,
                "n_ics": args.n_ics,
                "case_id_range": [int(case_ids[0]), int(case_ids[-1])],
                "seed_sequence_entropy": [int(args.seed), stream_id, 0],
            }
        )
        print(f"[{cfg.name}] wrote {path} ({path.stat().st_size / 2**20:.2f} MiB)", flush=True)

    manifest: dict[str, JsonValue] = {
        "schema": "manuscript_ic_panel_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "n_ics_per_family": args.n_ics,
        "families": entries,
        "master_seed": int(args.seed),
        "rng_stream_base": STREAM_BASE,
        "platform_used_for_ic_construction": args.platform,
        "script": "solver/evals/build_manuscript_ic_panels.py",
        "script_sha256": script_sha,
        "disjointness_statement": (
            "All parameter draws, phases, depths, crest configurations, and case IDs are new. "
            "The panels reproduce production distributions but contain no states selected from "
            "the snapshot-level training corpus or the legacy first-N evaluation archives."
        ),
        "replicate_groups": {
            "tanaka_v2_sum_budgeted": ["tanaka_g0", "tanaka_g1"],
            "bf_v2": ["bf_g0", "bf_g1", "bf_modal"],
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_manifest(manifest_path, manifest)
    print(f"wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
