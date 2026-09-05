"""Evaluate a trained surrogate on the released four-family paper dataset.

The dataset stores adaptive, simulation-specific save times, so this evaluator uses
test-split initial conditions and recomputes both f64 truth and
surrogate trajectories on one fixed evaluation grid per physical family.

Typical use::

    uv run python -m solver.evals.eval_suite \
        --run_dir outputs/c27_rerun \
        --dataset outputs/paper_dataset_literature_aligned_v1/combined/\
c16384_v01024_t01024/paper_dataset_all_splits_c16384.dataset.json \
        --n_ics 16 --gpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from solver.evals.model_rollout import (  # noqa: E402
    GL2_ITERATIONS,
    LoadedRun,
    build_predict_gxi_batched,
    load_run,
    rollout_surrogate,
)
from solver.gen_data.pipeline.types import DatasetSplit  # noqa: E402
from solver.solvers import time_integrator as ti  # noqa: E402
from solver.solvers.dno_series_jax import build_grid, make_linear_dno_symbol  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET = (
    REPO_ROOT
    / "outputs/paper_dataset_literature_aligned_v1/combined/c16384_v01024_t01024"
    / "paper_dataset_all_splits_c16384.dataset.json"
)
RolloutPayload = dict[str, np.ndarray | float]
TRUTH_DRIFT_TOL = 1e-3
TERMINAL_FAILURE_THRESHOLDS = (0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class IC:
    eta: np.ndarray
    xi: np.ndarray
    depth: float
    simulation_id: int
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FamilyConfig:
    family_id: int
    dt: float
    tmax: float
    substeps: int
    filter_fraction: float = 0.25


FAMILY_CONFIGS: dict[str, FamilyConfig] = {
    "stokes": FamilyConfig(family_id=1, dt=0.08, tmax=20.0, substeps=8),
    "tanaka": FamilyConfig(family_id=2, dt=0.8, tmax=200.0, substeps=80),
    "benjamin_feir": FamilyConfig(family_id=3, dt=0.8, tmax=200.0, substeps=80),
    "jonswap_tma": FamilyConfig(family_id=4, dt=0.08, tmax=20.0, substeps=8),
}


def _resolve_manifest_path(manifest_path: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"dataset manifest must define nonempty {label}")
    return (manifest_path.parent / value).resolve()


def _load_paper_dataset_ics(
    dataset_path: Path,
    family: str,
    n_ics: int,
) -> tuple[list[IC], dict[str, object], int, float]:
    """Load the first accepted test initial conditions for one family."""
    dataset_path = dataset_path.resolve()
    manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"dataset manifest must contain a JSON object: {dataset_path}")
    if manifest.get("schema_version") != 2:
        raise ValueError("eval_suite requires paper-dataset schema version 2")
    if manifest.get("requires_trajectory_map") is not True:
        raise ValueError("paper-dataset manifest must require its trajectory map")

    grid = manifest.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("dataset manifest is missing grid metadata")
    nx = int(grid["nx"])
    length = float(grid["length"])
    if nx <= 0 or not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"invalid dataset grid: nx={nx}, length={length}")

    trajectory_map_path = _resolve_manifest_path(
        dataset_path, manifest.get("trajectory_map_npz"), "trajectory_map_npz"
    )
    family_id = FAMILY_CONFIGS[family].family_id

    with np.load(trajectory_map_path, mmap_mode="r", allow_pickle=False) as mapping:
        required = {
            "frame_index",
            "shard_index",
            "shard_row",
            "trajectory_accepted",
            "trajectory_simulation_id",
            "trajectory_family_id",
            "trajectory_first_row",
            "trajectory_index",
            "trajectory_row_count",
            "trajectory_dataset_split",
        }
        missing = sorted(required.difference(mapping.files))
        if missing:
            raise ValueError(f"trajectory map is missing arrays: {missing}")

        accepted = np.asarray(mapping["trajectory_accepted"], dtype=bool)
        family_ids = np.asarray(mapping["trajectory_family_id"])
        dataset_splits = np.asarray(mapping["trajectory_dataset_split"])
        candidates = np.flatnonzero(
            accepted
            & (family_ids == family_id)
            & (dataset_splits == DatasetSplit.TEST.value)
        )
        if candidates.size < n_ics:
            raise ValueError(
                f"family {family!r} has only {candidates.size} accepted test trajectories; "
                f"requested {n_ics}"
            )
        selected = candidates[:n_ics]
        first_rows = np.asarray(mapping["trajectory_first_row"])[selected].astype(
            np.int64
        )
        row_counts = np.asarray(mapping["trajectory_row_count"])[selected]
        simulation_ids = np.asarray(mapping["trajectory_simulation_id"])[
            selected
        ].astype(np.int64)
        trajectory_index = np.asarray(mapping["trajectory_index"])[first_rows]
        frame_indices = np.asarray(mapping["frame_index"])[first_rows]
        shard_indices = np.asarray(mapping["shard_index"])[first_rows].astype(np.int64)
        shard_rows = np.asarray(mapping["shard_row"])[first_rows].astype(np.int64)

    if np.any(row_counts <= 0):
        raise ValueError(f"family {family!r} contains an empty selected trajectory")
    if not np.array_equal(trajectory_index, selected):
        raise ValueError(
            f"family {family!r} has inconsistent trajectory first-row pointers"
        )
    if np.any(frame_indices != 0):
        raise ValueError(
            f"family {family!r} selected trajectory does not begin at frame 0"
        )
    if np.unique(simulation_ids).size != n_ics:
        raise ValueError(
            f"family {family!r} selected test simulation IDs are not unique"
        )

    shard_specs = manifest.get("dataset_shards")
    if not isinstance(shard_specs, list):
        raise ValueError("dataset manifest is missing dataset_shards")
    loaded_rows: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    loaded_shards: list[dict[str, object]] = []
    for shard_index in np.unique(shard_indices):
        try:
            shard_spec = shard_specs[int(shard_index)]
        except (IndexError, TypeError) as exc:
            raise ValueError(
                f"trajectory map references invalid shard {shard_index}"
            ) from exc
        if not isinstance(shard_spec, dict):
            raise ValueError(f"dataset shard {shard_index} metadata is not an object")
        shard_path = _resolve_manifest_path(
            dataset_path, shard_spec.get("path"), f"shard {shard_index}"
        )
        positions = np.flatnonzero(shard_indices == shard_index)
        rows = shard_rows[positions]
        with np.load(shard_path, mmap_mode="r", allow_pickle=False) as shard:
            required_arrays = {"eta", "xi", "depth", "frame_index"}
            missing = sorted(required_arrays.difference(shard.files))
            if missing:
                raise ValueError(
                    f"dataset shard {shard_path} is missing arrays: {missing}"
                )
            if np.any(rows < 0) or np.any(rows >= shard["eta"].shape[0]):
                raise ValueError(f"trajectory map references rows outside {shard_path}")
            if np.any(np.asarray(shard["frame_index"])[rows] != 0):
                raise ValueError(
                    f"trajectory map frame-0 rows disagree with {shard_path}"
                )
            for position, row in zip(positions, rows, strict=True):
                loaded_rows[int(position)] = (
                    np.asarray(shard["eta"][row], dtype=np.float64),
                    np.asarray(shard["xi"][row], dtype=np.float64),
                    np.asarray(shard["depth"][row], dtype=np.float64),
                )
        loaded_shards.append(
            {
                "index": int(shard_index),
                "path": str(shard_path),
            }
        )

    ics = []
    for position, simulation_id in enumerate(simulation_ids):
        eta, xi, depth_array = loaded_rows[position]
        depth = float(depth_array)
        if eta.shape != (nx,) or xi.shape != (nx,):
            raise ValueError(
                f"family {family!r} IC shape mismatch: eta={eta.shape}, xi={xi.shape}, "
                f"expected ({nx},)"
            )
        if not (
            np.isfinite(eta).all() and np.isfinite(xi).all() and np.isfinite(depth)
        ):
            raise ValueError(f"family {family!r} contains non-finite initial data")
        if depth <= 0.0:
            raise ValueError(f"family {family!r} contains non-positive depth {depth}")
        ics.append(
            IC(
                eta=eta,
                xi=xi,
                depth=depth,
                simulation_id=int(simulation_id),
                meta={
                    "trajectory_index": int(selected[position]),
                    "shard_index": int(shard_indices[position]),
                    "shard_row": int(shard_rows[position]),
                },
            )
        )

    source = {
        "kind": "paper_dataset_test_split",
        "family": family,
        "family_id": family_id,
        "dataset_split": 2,
        "dataset_manifest": str(dataset_path),
        "trajectory_map": str(trajectory_map_path),
        "loaded_shards": loaded_shards,
    }
    return ics, source, nx, length


def _truth_protocol(
    family: str,
    cfg: FamilyConfig,
    *,
    nx: int,
    length: float,
    ics: list[IC],
) -> str:
    protocol = {
        "schema_version": 2,
        "family": family,
        "dt": cfg.dt,
        "tmax": cfg.tmax,
        "substeps": cfg.substeps,
        "implicit_iterations": GL2_ITERATIONS,
        "filter_fraction": cfg.filter_fraction,
        "method": "gl2_if",
        "zero_mean_xi": True,
        "dtype": "float64",
        "nx": nx,
        "length": length,
        "simulation_ids": [ic.simulation_id for ic in ics],
        "depths": [ic.depth for ic in ics],
    }
    return json.dumps(protocol, sort_keys=True, separators=(",", ":"))


def _try_load_cached_truth(
    family: str,
    cache_dir: Path | None,
    *,
    expected_shape: tuple[int, int, int],
    expected_simulation_ids: list[int],
    expected_protocol_json: str,
) -> RolloutPayload | None:
    if cache_dir is None:
        return None
    candidates = (
        cache_dir / f"{family}_truth_cache.npz",
        cache_dir / f"{family}_trajs.npz",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                fields = {
                    name: np.asarray(archive[f"truth_{name}"])
                    for name in ("eta", "xi", "gxi")
                }
                simulation_ids = np.asarray(archive["simulation_ids"]).tolist()
                protocol_json = str(archive["truth_protocol_json"].item())
        except (KeyError, OSError, ValueError) as exc:
            print(f"[{family}] truth cache rejected ({path}): {exc}", flush=True)
            continue
        if any(values.shape != expected_shape for values in fields.values()):
            print(
                f"[{family}] truth cache rejected ({path}): shape mismatch", flush=True
            )
            continue
        if simulation_ids != expected_simulation_ids:
            print(
                f"[{family}] truth cache rejected ({path}): simulation IDs differ",
                flush=True,
            )
            continue
        if protocol_json != expected_protocol_json:
            print(
                f"[{family}] truth cache rejected ({path}): protocol differs",
                flush=True,
            )
            continue
        print(f"[{family}] using truth cache {path}", flush=True)
        return {**fields, "wall_s": 0.0}
    return None


def _write_truth_cache(
    out_dir: Path,
    family: str,
    truth: RolloutPayload,
    *,
    times: np.ndarray,
    depths: np.ndarray,
    simulation_ids: np.ndarray,
    truth_protocol_json: str,
) -> Path:
    path = out_dir / f"{family}_truth_cache.npz"
    temporary = out_dir / f".{family}_truth_cache.tmp.npz"
    np.savez(
        temporary,
        times=np.asarray(times, dtype=np.float64),
        depths=np.asarray(depths, dtype=np.float64),
        simulation_ids=np.asarray(simulation_ids, dtype=np.int64),
        truth_eta=truth["eta"],
        truth_xi=truth["xi"],
        truth_gxi=truth["gxi"],
        truth_protocol_json=np.asarray(truth_protocol_json),
    )
    temporary.replace(path)
    return path


def truth_rollout_batched(
    ics: list[IC],
    times: jnp.ndarray,
    nx: int,
    length: float,
    cfg: FamilyConfig,
) -> RolloutPayload:
    dtype = jnp.float64
    _, k_grid_values = build_grid(nx, length)
    k_grid = jnp.asarray(k_grid_values, dtype=dtype)
    depths = jnp.asarray([ic.depth for ic in ics], dtype=dtype)[:, None]
    params = ti.SolverParams(
        nx=nx,
        length=length,
        depth=depths,
        gravity=1.0,
        dno_order=6,
        pad_factor=8,
        filter_fraction=cfg.filter_fraction,
        k=k_grid,
        g0=make_linear_dno_symbol(k_grid, depths),
    )
    state = ti.State(
        eta=jnp.asarray(np.stack([ic.eta for ic in ics]), dtype=dtype),
        xi=jnp.asarray(np.stack([ic.xi for ic in ics]), dtype=dtype),
    )
    started = time.perf_counter()
    result = ti.batched_rollout(
        state,
        times,
        params,
        save_gxi=True,
        substeps_per_interval=cfg.substeps,
        method="gl2_if",
        implicit_iterations=GL2_ITERATIONS,
        zero_mean_xi=True,
    )
    jax.block_until_ready(result["eta"])
    return {
        "eta": np.asarray(result["eta"], dtype=np.float32),
        "xi": np.asarray(result["xi"], dtype=np.float32),
        "gxi": np.asarray(result["gxi"], dtype=np.float32),
        "wall_s": float(time.perf_counter() - started),
    }


def surrogate_rollout_batched(
    ics: list[IC],
    times: jnp.ndarray,
    nx: int,
    length: float,
    cfg: FamilyConfig,
    predict_gxi_batched: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> RolloutPayload:
    """Run the locked-in unguarded surrogate in an f64 integration harness."""
    dtype = jnp.float64
    _, k_grid_values = build_grid(nx, length)
    k_grid = jnp.asarray(k_grid_values, dtype=dtype)
    depths_values = np.asarray([ic.depth for ic in ics], dtype=np.float64)
    depths = jnp.asarray(depths_values, dtype=dtype)[:, None]
    log_depths = jnp.asarray(np.log(depths_values), dtype=jnp.float32)
    params = ti.SolverParams(
        nx=nx,
        length=length,
        depth=depths,
        gravity=1.0,
        dno_order=6,
        pad_factor=8,
        filter_fraction=cfg.filter_fraction,
        k=k_grid,
        g0=make_linear_dno_symbol(k_grid, depths),
    )
    state = ti.State(
        eta=jnp.asarray(np.stack([ic.eta for ic in ics]), dtype=dtype),
        xi=jnp.asarray(np.stack([ic.xi for ic in ics]), dtype=dtype),
    )

    def predict(eta: jnp.ndarray, xi: jnp.ndarray) -> jnp.ndarray:
        return predict_gxi_batched(
            eta.astype(jnp.float32), xi.astype(jnp.float32), log_depths
        ).astype(dtype)

    @jax.jit
    def rollout() -> dict[str, jnp.ndarray]:
        return rollout_surrogate(
            state,
            times,
            params,
            predict,
            substeps=cfg.substeps,
        )

    started = time.perf_counter()
    result = rollout()
    jax.block_until_ready(result["eta"])
    return {
        "eta": np.asarray(result["eta"], dtype=np.float32),
        "xi": np.asarray(result["xi"], dtype=np.float32),
        "gxi": np.asarray(result["gxi"], dtype=np.float32),
        "wall_s": float(time.perf_counter() - started),
    }


def _rollout_ic_chunks(
    ics: list[IC],
    batch_size: int | None,
    rollout: Callable[[list[IC]], RolloutPayload],
    *,
    label: str,
) -> RolloutPayload:
    effective_batch_size = min(batch_size or len(ics), len(ics))
    if effective_batch_size == len(ics):
        return rollout(ics)

    chunks: dict[str, list[np.ndarray]] = {"eta": [], "xi": [], "gxi": []}
    total_wall = 0.0
    for start in range(0, len(ics), effective_batch_size):
        stop = min(start + effective_batch_size, len(ics))
        print(f"    {label} IC chunk {start}:{stop} of {len(ics)}", flush=True)
        result = rollout(ics[start:stop])
        for field_name in chunks:
            chunks[field_name].append(np.asarray(result[field_name]))
        total_wall += float(result["wall_s"])
    return {
        **{name: np.concatenate(values, axis=1) for name, values in chunks.items()},
        "wall_s": total_wall,
    }


def _rel_l2(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - truth, axis=-1) / (
        np.linalg.norm(truth, axis=-1) + 1e-12
    )


def _rate(mask: np.ndarray, cohort: np.ndarray) -> float:
    denominator = int(np.count_nonzero(cohort))
    return (
        float(np.count_nonzero(mask & cohort) / denominator)
        if denominator
        else float("nan")
    )


def _conditional_finite_stats(values: np.ndarray) -> tuple[int, float, float, float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0, float("nan"), float("nan"), float("nan")
    return (
        int(finite_values.size),
        float(np.mean(finite_values)),
        float(np.median(finite_values)),
        float(np.percentile(finite_values, 95)),
    )


def _tau_label(tau: float) -> str:
    return f"{tau:g}".replace(".", "p")


def _json_ready(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _summary_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def compute_metrics(
    truth: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    times: np.ndarray,
    length: float,
) -> dict[str, Any]:
    """Compute failure-aware rollout metrics without compatibility aliases."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        errors = {
            field: _rel_l2(pred[field], truth[field]) for field in ("eta", "xi", "gxi")
        }
        n_t, n_ics = errors["eta"].shape
        dx = length / truth["eta"].shape[-1]
        h_truth = (
            0.5 * np.sum(truth["xi"] * truth["gxi"] + truth["eta"] ** 2, axis=-1) * dx
        )
        h_pred = 0.5 * np.sum(pred["xi"] * pred["gxi"] + pred["eta"] ** 2, axis=-1) * dx
        drift_truth = (h_truth - h_truth[:1]) / (np.abs(h_truth[:1]) + 1e-12)
        drift_pred = (h_pred - h_pred[:1]) / (np.abs(h_pred[:1]) + 1e-12)
        energy_error = (h_pred - h_truth) / (np.abs(h_truth) + 1e-12)
    truth_nonfinite_by_field: dict[str, np.ndarray] = {
        field: np.asarray(
            ~np.isfinite(truth[field]).all(axis=(0, 2)),
            dtype=np.bool_,
        )
        for field in ("eta", "xi", "gxi")
    }
    truth_nonfinite = np.logical_or.reduce(tuple(truth_nonfinite_by_field.values()))
    drift_abs = np.abs(drift_truth)
    drift_abs[~np.isfinite(drift_abs)] = np.inf
    truth_drift_invalid = ~truth_nonfinite & (drift_abs.max(axis=0) > TRUTH_DRIFT_TOL)
    truth_invalid = truth_nonfinite | truth_drift_invalid
    valid = ~truth_invalid

    truth_invalid_reasons = []
    for ic_index in np.flatnonzero(truth_invalid):
        reasons = [
            f"nonfinite_{field}"
            for field, mask in truth_nonfinite_by_field.items()
            if mask[ic_index]
        ]
        if truth_drift_invalid[ic_index]:
            reasons.append("energy_drift_gt_tol")
        truth_invalid_reasons.append({"ic_index": int(ic_index), "reasons": reasons})

    pred_nonfinite_by_field = {
        field: ~np.isfinite(pred[field]).all(axis=(0, 2))
        for field in ("eta", "xi", "gxi")
    }
    pred_nonfinite_any = np.logical_or.reduce(tuple(pred_nonfinite_by_field.values()))
    model_finite_truth_valid = valid & ~pred_nonfinite_any
    out: dict[str, Any] = {
        "n_t": n_t,
        "n_ics_attempted": n_ics,
        "n_truth_valid": int(valid.sum()),
        "n_truth_invalid": int(truth_invalid.sum()),
        "truth_valid_rate_attempted": _rate(valid, np.ones(n_ics, dtype=bool)),
        "truth_invalid_rate_attempted": _rate(
            truth_invalid, np.ones(n_ics, dtype=bool)
        ),
        "n_model_finite_truth_valid": int(model_finite_truth_valid.sum()),
        "truth_valid_ics": [int(index) for index in np.flatnonzero(valid)],
        "truth_invalid_ics": [int(index) for index in np.flatnonzero(truth_invalid)],
        "truth_invalid_reasons": truth_invalid_reasons,
        "truth_invalid_reason_counts": {
            **{
                f"nonfinite_{field}": int(np.count_nonzero(mask))
                for field, mask in truth_nonfinite_by_field.items()
            },
            "energy_drift_gt_tol": int(np.count_nonzero(truth_drift_invalid)),
        },
        "truth_drift_tol": TRUTH_DRIFT_TOL,
        "model_nonfinite_any_ics": [
            int(index) for index in np.flatnonzero(pred_nonfinite_any)
        ],
        "model_nonfinite_any_ics_truth_valid": [
            int(index) for index in np.flatnonzero(pred_nonfinite_any & valid)
        ],
        "model_nonfinite_any_count_attempted": int(
            np.count_nonzero(pred_nonfinite_any)
        ),
        "model_nonfinite_any_rate_attempted": _rate(
            pred_nonfinite_any, np.ones(n_ics, dtype=bool)
        ),
        "model_nonfinite_any_count_truth_valid": int(
            np.count_nonzero(pred_nonfinite_any & valid)
        ),
        "model_nonfinite_any_rate_truth_valid": _rate(pred_nonfinite_any, valid),
        "model_nonfinite_reason_counts_truth_valid": {
            f"nonfinite_{field}": int(np.count_nonzero(mask & valid))
            for field, mask in pred_nonfinite_by_field.items()
        },
    }

    terminal_metric_nonfinite = ~np.isfinite(errors["eta"][-1])
    for tau in TERMINAL_FAILURE_THRESHOLDS:
        label = _tau_label(tau)
        failed = (
            pred_nonfinite_any | terminal_metric_nonfinite | (errors["eta"][-1] > tau)
        )
        out[f"terminal_eta_failure_count_tau_{label}"] = int(
            np.count_nonzero(failed & valid)
        )
        out[f"terminal_eta_failure_rate_tau_{label}"] = _rate(failed, valid)
    out["terminal_eta_failure_definition"] = (
        "truth-valid IC; final eta relative L2 > tau, or any nonfinite value in "
        "predicted eta/xi/gxi at any saved frame"
    )

    for fraction, label in ((0.1, "t10p"), (0.5, "t50p"), (1.0, "tfinal")):
        frame = int(round((n_t - 1) * fraction))
        out[f"metric_frame_index_{label}"] = frame
        out[f"metric_time_{label}"] = float(times[frame])
        for field_name, field_errors in errors.items():
            count, mean, median, p95 = _conditional_finite_stats(
                field_errors[frame, model_finite_truth_valid]
            )
            prefix = f"rel_l2_{field_name}"
            out[f"{prefix}_n_conditional_finite_{label}"] = count
            out[f"{prefix}_mean_conditional_finite_{label}"] = mean
            out[f"{prefix}_median_conditional_finite_{label}"] = median
            out[f"{prefix}_p95_conditional_finite_{label}"] = p95

    h_count, _, h_median, h_p95 = _conditional_finite_stats(
        np.abs(drift_pred[-1, model_finite_truth_valid])
    )
    out.update(
        {
            "hamiltonian_drift_pred_n_conditional_finite_tfinal": h_count,
            "hamiltonian_drift_pred_median_abs_conditional_finite_tfinal": h_median,
            "hamiltonian_drift_pred_p95_abs_conditional_finite_tfinal": h_p95,
            "hamiltonian_drift_pred_reference": "predicted Hamiltonian at t=0",
        }
    )
    e_count, _, e_median, e_p95 = _conditional_finite_stats(
        np.abs(energy_error[-1, model_finite_truth_valid])
    )
    out.update(
        {
            "energy_error_pred_vs_truth_n_conditional_finite_tfinal": e_count,
            "energy_error_pred_vs_truth_median_abs_conditional_finite_tfinal": e_median,
            "energy_error_pred_vs_truth_p95_abs_conditional_finite_tfinal": e_p95,
            "_arrays": {
                "rel_l2_eta": errors["eta"],
                "rel_l2_xi": errors["xi"],
                "rel_l2_gxi": errors["gxi"],
                "hamiltonian_drift_pred": drift_pred,
                "energy_drift_truth": drift_truth,
                "energy_error_pred_vs_truth": energy_error,
                "truth_valid": valid,
                "model_nonfinite_any": pred_nonfinite_any,
            },
        }
    )
    return out


def compute_macro_summary(summaries: dict[str, dict[str, Any]]) -> dict[str, object]:
    """Compute equal-family macro rates and pooled-simulation micro rates."""
    items = [
        (name, summary)
        for name, summary in summaries.items()
        if "n_ics_attempted" in summary
    ]
    attempted_total = sum(int(summary["n_ics_attempted"]) for _, summary in items)
    truth_valid_total = sum(int(summary["n_truth_valid"]) for _, summary in items)
    truth_invalid_total = sum(int(summary["n_truth_invalid"]) for _, summary in items)
    macro: dict[str, object] = {
        "n_families": len(items),
        "families": [name for name, _ in items],
        "family_macro_definition": "equal weight for every physical dataset family",
        "n_ics_attempted_total": attempted_total,
        "n_truth_valid_total": truth_valid_total,
        "n_truth_invalid_total": truth_invalid_total,
    }
    attempted = attempted_total
    valid = truth_valid_total
    invalid_rates = np.asarray(
        [
            _summary_float(summary["truth_invalid_rate_attempted"])
            for _, summary in items
            if int(summary["n_ics_attempted"]) > 0
        ],
        dtype=np.float64,
    )
    macro["truth_invalid_rate_attempted_macro"] = (
        float(np.mean(invalid_rates)) if invalid_rates.size else float("nan")
    )
    macro["truth_invalid_rate_attempted_micro"] = (
        float(truth_invalid_total / attempted) if attempted else float("nan")
    )

    rate_and_count_keys = [
        (
            "model_nonfinite_any_rate_truth_valid",
            "model_nonfinite_any_count_truth_valid",
        ),
        *[
            (
                f"terminal_eta_failure_rate_tau_{_tau_label(tau)}",
                f"terminal_eta_failure_count_tau_{_tau_label(tau)}",
            )
            for tau in TERMINAL_FAILURE_THRESHOLDS
        ],
    ]
    for rate_key, count_key in rate_and_count_keys:
        rates = np.asarray(
            [
                _summary_float(summary[rate_key])
                for _, summary in items
                if int(summary["n_truth_valid"]) > 0
                and np.isfinite(_summary_float(summary[rate_key]))
            ],
            dtype=np.float64,
        )
        macro[f"{rate_key}_macro"] = (
            float(np.mean(rates)) if rates.size else float("nan")
        )
        failures = sum(int(summary[count_key]) for _, summary in items)
        macro[f"{rate_key}_micro"] = float(failures / valid) if valid else float("nan")

    for metric_key in (
        "rel_l2_eta_median_conditional_finite_tfinal",
        "rel_l2_eta_p95_conditional_finite_tfinal",
    ):
        values = np.asarray(
            [
                _summary_float(summary[metric_key])
                for _, summary in items
                if np.isfinite(_summary_float(summary[metric_key]))
            ],
            dtype=np.float64,
        )
        macro[f"{metric_key}_macro_mean"] = (
            float(np.mean(values)) if values.size else float("nan")
        )
    return macro


def run_family(
    family: str,
    cfg: FamilyConfig,
    loaded: LoadedRun,
    predict_gxi_batched: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    out_dir: Path,
    dataset_path: Path,
    n_ics: int,
    truth_cache_dir: Path | None,
    checkpoint_source: dict[str, object],
    rollout_batch_size: int | None,
) -> dict[str, Any]:
    ics, source, nx, length = _load_paper_dataset_ics(dataset_path, family, n_ics)
    times_np = np.arange(0.0, cfg.tmax + 0.5 * cfg.dt, cfg.dt, dtype=np.float64)
    protocol_json = _truth_protocol(
        family,
        cfg,
        nx=nx,
        length=length,
        ics=ics,
    )
    shape = (len(times_np), len(ics), nx)
    print(
        f"[{family}] n_ics={len(ics)} dt={cfg.dt} tmax={cfg.tmax} "
        f"n_t={len(times_np)} batch={rollout_batch_size or len(ics)}",
        flush=True,
    )
    truth = _try_load_cached_truth(
        family,
        truth_cache_dir,
        expected_shape=shape,
        expected_simulation_ids=[ic.simulation_id for ic in ics],
        expected_protocol_json=protocol_json,
    )
    truth_cache_path: Path | None = None
    if truth is None:
        print(f"[{family}] f64 truth rollout", flush=True)
        truth = _rollout_ic_chunks(
            ics,
            rollout_batch_size,
            lambda chunk: truth_rollout_batched(
                chunk,
                jnp.asarray(times_np, dtype=jnp.float64),
                nx,
                length,
                cfg,
            ),
            label=f"{family} truth",
        )
        truth_cache_path = _write_truth_cache(
            out_dir,
            family,
            truth,
            times=times_np,
            depths=np.asarray([ic.depth for ic in ics]),
            simulation_ids=np.asarray([ic.simulation_id for ic in ics]),
            truth_protocol_json=protocol_json,
        )
    print(f"[{family}] truth wall={float(truth['wall_s']):.1f}s", flush=True)

    print(f"[{family}] batched surrogate rollout (f64 harness, f32 model)", flush=True)
    pred = _rollout_ic_chunks(
        ics,
        rollout_batch_size,
        lambda chunk: surrogate_rollout_batched(
            chunk,
            jnp.asarray(times_np, dtype=jnp.float64),
            nx,
            length,
            cfg,
            predict_gxi_batched,
        ),
        label=f"{family} surrogate",
    )
    print(f"[{family}] surrogate wall={float(pred['wall_s']):.1f}s", flush=True)
    metrics = compute_metrics(
        {name: np.asarray(truth[name]) for name in ("eta", "xi", "gxi")},
        {name: np.asarray(pred[name]) for name in ("eta", "xi", "gxi")},
        times_np,
        length,
    )
    arrays = metrics["_arrays"]
    np.savez_compressed(
        out_dir / f"{family}_trajs.npz",
        times=times_np.astype(np.float32),
        depths=np.asarray([ic.depth for ic in ics], dtype=np.float32),
        simulation_ids=np.asarray([ic.simulation_id for ic in ics], dtype=np.int64),
        truth_eta=truth["eta"],
        truth_xi=truth["xi"],
        truth_gxi=truth["gxi"],
        pred_eta=pred["eta"],
        pred_xi=pred["xi"],
        pred_gxi=pred["gxi"],
        rel_l2_eta=arrays["rel_l2_eta"],
        rel_l2_xi=arrays["rel_l2_xi"],
        rel_l2_gxi=arrays["rel_l2_gxi"],
        hamiltonian_drift_pred=arrays["hamiltonian_drift_pred"],
        energy_drift_truth=arrays["energy_drift_truth"],
        energy_error_pred_vs_truth=arrays["energy_error_pred_vs_truth"],
        truth_valid=arrays["truth_valid"],
        model_nonfinite_any=arrays["model_nonfinite_any"],
        truth_protocol_json=np.asarray(protocol_json),
    )

    summary = {key: value for key, value in metrics.items() if key != "_arrays"}
    summary.update(
        {
            "family": family,
            "dt": cfg.dt,
            "tmax": cfg.tmax,
            "length": length,
            "nx": nx,
            "n_ics": len(ics),
            "depths": [ic.depth for ic in ics],
            "simulation_ids": [ic.simulation_id for ic in ics],
            "epoch": loaded.epoch,
            "checkpoint_source": checkpoint_source,
            "evaluation_source": source,
            "truth_wall_s": truth["wall_s"],
            "surrogate_wall_s": pred["wall_s"],
            "truth_protocol": json.loads(protocol_json),
            "precision": "f64 integration harness / f32 model",
            "rollout_batch_size": min(rollout_batch_size or len(ics), len(ics)),
        }
    )
    _write_json(out_dir / f"{family}_summary.json", summary)
    if truth_cache_path is not None:
        truth_cache_path.unlink(missing_ok=True)
    print(
        f"[{family}] done: nonfinite="
        f"{_summary_float(summary['model_nonfinite_any_rate_truth_valid']):.3f}, "
        f"eta median="
        f"{_summary_float(summary['rel_l2_eta_median_conditional_finite_tfinal']):.4g}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on paper-dataset test initial conditions."
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=tuple(FAMILY_CONFIGS),
        default=list(FAMILY_CONFIGS),
    )
    parser.add_argument("--n_ics", type=int, default=16)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--truth_cache", default=None)
    parser.add_argument("--rollout_batch_size", type=int, default=None)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if not args.gpu:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    jax.config.update("jax_enable_x64", True)

    run_dir = Path(args.run_dir).resolve()
    dataset_path = Path(args.dataset).resolve()
    out_dir = (
        Path(args.output_dir).resolve() if args.output_dir else run_dir / "eval_suite"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_run(run_dir, checkpoint=args.checkpoint)
    checkpoint_dir = run_dir / (
        "best_val_ckpt" if args.checkpoint == "best" else "final_ckpt"
    )
    checkpoint_source: dict[str, object] = {
        "run_dir": str(run_dir),
        "selection": args.checkpoint,
        "path": str(checkpoint_dir),
        "epoch": loaded.epoch,
    }
    predict_gxi_batched = build_predict_gxi_batched(loaded)
    truth_cache_dir = Path(args.truth_cache).resolve() if args.truth_cache else None
    summaries = {
        family: run_family(
            family,
            FAMILY_CONFIGS[family],
            loaded,
            predict_gxi_batched,
            out_dir,
            dataset_path,
            args.n_ics,
            truth_cache_dir,
            checkpoint_source,
            args.rollout_batch_size,
        )
        for family in args.families
    }
    _write_json(out_dir / "all_summaries.json", summaries)
    _write_json(out_dir / "macro_summary.json", compute_macro_summary(summaries))
    print(f"Done. Results written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
