"""CPU-only source-protocol audit for the stored v9 ``G(eta) xi`` labels.

The 88 GiB dataset is a ZIP_STORED NPZ.  This script memory-maps its NPY
members in place, chooses a small deterministic rank-stratified sample from
each source, and recomputes the labels with the production convention:
float64 Craig--Sulem order 6 and padding factor 8.  It then applies the
source-specific label protocol: rollout sources use the hard 0.25 low-pass,
whereas the static linear and Stokes generators (source IDs 2, 3, and 4)
store the unfiltered series result.

The sample is rank-stratified by depth (four groups), ``max|eta| / depth``
(two groups within each depth group), and the fraction of eta Fourier energy
in modes ``|k| >= 32`` (two groups within each preceding group).  Thus the
default 64 rows per source contribute four rows from each of 16 strata even
when a source has constant depth or tied diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# These must be set before importing JAX.  They make accidental GPU use an
# error rather than allowing this audit to contend with training.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import numpy as np
from numpy.lib import format as npy_format

from solver.solvers.dno_series_jax import build_grid, dno_series_eval, myfft, myifft


QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
METRIC_NAMES = (
    "relative_l2",
    "relative_h1",
    "max_abs_error",
    "correlation",
    "sign_agreement_active",
    "stored_rejected_band_relative_l2",
)


@dataclass(frozen=True)
class Sample:
    global_indices: np.ndarray
    source_ids: np.ndarray
    depth_bins: np.ndarray
    amplitude_bins: np.ndarray
    highband_bins: np.ndarray
    amplitude_over_depth: np.ndarray
    eta_highband_fraction: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/combined_dataset_v9.npz"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("notes/v9_label_provenance_audit_20260720"),
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--rows-per-source", type=int, default=64)
    parser.add_argument("--candidate-rows-per-source", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dno-order", type=int, default=6)
    parser.add_argument("--pad-factor", type=int, default=8)
    parser.add_argument("--filter-fraction", type=float, default=0.25)
    parser.add_argument(
        "--filtered-source-ids",
        default="0,1,5,6,7,8,9,11,12,13,14",
        help=(
            "Comma-separated rollout source IDs whose stored-label protocol "
            "includes --filter-fraction. Other source IDs are recomputed unfiltered."
        ),
    )
    parser.add_argument("--highband-k", type=float, default=32.0)
    return parser.parse_args()


def _entry_payload_offset(npz_path: Path, member: str) -> int:
    with zipfile.ZipFile(npz_path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{npz_path}:{member} is compressed; safe mmap is unavailable")

    with npz_path.open("rb") as handle:
        handle.seek(info.header_offset)
        local_header = handle.read(30)
    if local_header[:4] != b"PK\x03\x04":
        raise ValueError(f"bad local ZIP header for {npz_path}:{member}")
    filename_length, extra_length = struct.unpack("<HH", local_header[26:30])
    return info.header_offset + 30 + filename_length + extra_length


def mmap_npy_member(npz_path: Path, member: str) -> np.memmap:
    """Memory-map an uncompressed NPY member without reading the NPZ payload."""
    payload_offset = _entry_payload_offset(npz_path, member)
    with npz_path.open("rb") as handle:
        handle.seek(payload_offset)
        version = npy_format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported NPY version {version} in {npz_path}:{member}")
        data_offset = handle.tell()
    if fortran_order:
        raise ValueError(f"{npz_path}:{member} is Fortran-order; unsupported")
    return np.memmap(npz_path, dtype=dtype, mode="r", offset=data_offset, shape=shape)


def _rank_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Return exactly populated rank bins, with stable handling of tied values."""
    if values.ndim != 1:
        raise ValueError(f"rank-bin input must be one-dimensional, got {values.shape}")
    order = np.argsort(values, kind="stable")
    labels = np.empty(values.shape[0], dtype=np.int8)
    labels[order] = np.minimum(
        (np.arange(values.shape[0], dtype=np.int64) * n_bins) // values.shape[0],
        n_bins - 1,
    ).astype(np.int8)
    return labels


def _eta_diagnostics(
    eta: np.ndarray,
    depth: np.ndarray,
    length: float,
    highband_k: float,
) -> tuple[np.ndarray, np.ndarray]:
    amplitude = np.max(np.abs(eta), axis=1) / np.maximum(np.abs(depth), 1.0e-12)
    modes = 2.0 * np.pi * np.fft.rfftfreq(eta.shape[1], d=length / eta.shape[1])
    spectrum = np.fft.rfft(eta.astype(np.float64, copy=False), axis=1)
    energy = np.abs(spectrum) ** 2
    total = np.sum(energy[:, modes > 0.0], axis=1)
    high = np.sum(energy[:, modes >= highband_k], axis=1)
    fraction = np.sqrt(high / np.maximum(total, 1.0e-300))
    return amplitude.astype(np.float64), fraction.astype(np.float64)


def _sample_one_source(
    *,
    source_id: int,
    source_rows: np.ndarray,
    eta_map: np.memmap,
    depth_map: np.memmap,
    length: float,
    highband_k: float,
    candidate_count: int,
    rows_per_source: int,
    rng: np.random.Generator,
) -> Sample:
    n_cells = 4 * 2 * 2
    if rows_per_source % n_cells != 0:
        raise ValueError(f"rows-per-source must be divisible by {n_cells}")
    if source_rows.size < rows_per_source:
        raise ValueError(
            f"source {source_id} has {source_rows.size} rows, fewer than {rows_per_source}"
        )
    candidate_count = min(candidate_count, int(source_rows.size))
    if candidate_count < rows_per_source:
        raise ValueError("candidate rows must be at least rows-per-source")

    candidate_indices = np.sort(
        rng.choice(source_rows, size=candidate_count, replace=False).astype(np.int64)
    )
    candidate_depth = np.asarray(depth_map[candidate_indices], dtype=np.float64)
    candidate_eta = np.asarray(eta_map[candidate_indices], dtype=np.float32)
    amplitude, highband = _eta_diagnostics(
        candidate_eta,
        candidate_depth,
        length,
        highband_k,
    )

    depth_bins = _rank_bins(candidate_depth, 4)
    amplitude_bins = np.zeros(candidate_count, dtype=np.int8)
    highband_bins = np.zeros(candidate_count, dtype=np.int8)
    for depth_bin in range(4):
        depth_positions = np.flatnonzero(depth_bins == depth_bin)
        amplitude_bins[depth_positions] = _rank_bins(amplitude[depth_positions], 2)
        for amplitude_bin in range(2):
            cell_positions = depth_positions[
                amplitude_bins[depth_positions] == amplitude_bin
            ]
            highband_bins[cell_positions] = _rank_bins(highband[cell_positions], 2)

    rows_per_cell = rows_per_source // n_cells
    selected_positions: list[np.ndarray] = []
    for depth_bin in range(4):
        for amplitude_bin in range(2):
            for highband_bin in range(2):
                cell = np.flatnonzero(
                    (depth_bins == depth_bin)
                    & (amplitude_bins == amplitude_bin)
                    & (highband_bins == highband_bin)
                )
                if cell.size < rows_per_cell:
                    raise RuntimeError(
                        f"source {source_id} stratum {(depth_bin, amplitude_bin, highband_bin)} "
                        f"has {cell.size} candidates, needs {rows_per_cell}"
                    )
                selected_positions.append(
                    np.sort(rng.choice(cell, size=rows_per_cell, replace=False))
                )
    positions = np.concatenate(selected_positions)
    return Sample(
        global_indices=candidate_indices[positions],
        source_ids=np.full(rows_per_source, source_id, dtype=np.int16),
        depth_bins=depth_bins[positions],
        amplitude_bins=amplitude_bins[positions],
        highband_bins=highband_bins[positions],
        amplitude_over_depth=amplitude[positions],
        eta_highband_fraction=highband[positions],
    )


def build_sample(
    *,
    source_map: np.memmap,
    eta_map: np.memmap,
    depth_map: np.memmap,
    length: float,
    highband_k: float,
    candidate_count: int,
    rows_per_source: int,
    seed: int,
) -> tuple[Sample, dict[int, int]]:
    source_ids, counts = np.unique(np.asarray(source_map), return_counts=True)
    source_counts = {
        int(source_id): int(count)
        for source_id, count in zip(source_ids.tolist(), counts.tolist())
    }
    rng = np.random.default_rng(seed)
    samples = [
        _sample_one_source(
            source_id=int(source_id),
            source_rows=np.flatnonzero(source_map == source_id),
            eta_map=eta_map,
            depth_map=depth_map,
            length=length,
            highband_k=highband_k,
            candidate_count=candidate_count,
            rows_per_source=rows_per_source,
            rng=rng,
        )
        for source_id in source_ids
    ]
    return (
        Sample(
            global_indices=np.concatenate([sample.global_indices for sample in samples]),
            source_ids=np.concatenate([sample.source_ids for sample in samples]),
            depth_bins=np.concatenate([sample.depth_bins for sample in samples]),
            amplitude_bins=np.concatenate([sample.amplitude_bins for sample in samples]),
            highband_bins=np.concatenate([sample.highband_bins for sample in samples]),
            amplitude_over_depth=np.concatenate(
                [sample.amplitude_over_depth for sample in samples]
            ),
            eta_highband_fraction=np.concatenate(
                [sample.eta_highband_fraction for sample in samples]
            ),
        ),
        source_counts,
    )


def recompute_labels(
    eta: np.ndarray,
    xi: np.ndarray,
    depth: np.ndarray,
    label_filter_fractions: np.ndarray,
    *,
    length: float,
    dno_order: int,
    pad_factor: int,
    batch_size: int,
) -> np.ndarray:
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"CPU backend required, found {jax.default_backend()!r}")
    jax.config.update("jax_enable_x64", True)
    _, k = build_grid(eta.shape[1], length)
    k = jnp.asarray(k, dtype=jnp.float64)

    @jax.jit
    def evaluate_batch(
        eta_batch: jax.Array,
        xi_batch: jax.Array,
        depth_batch: jax.Array,
        filter_fraction_batch: jax.Array,
    ) -> jax.Array:
        raw = dno_series_eval(
            eta_batch,
            xi_batch,
            k,
            depth_batch[:, None],
            dno_order,
            pad_factor=pad_factor,
        )
        cutoff = filter_fraction_batch[:, None] * jnp.max(jnp.abs(k))
        mask = (jnp.abs(k)[None, :] <= cutoff).astype(myfft(raw, raw.shape[-1]).dtype)
        return myifft(mask * myfft(raw, raw.shape[-1]))

    output = np.empty(eta.shape, dtype=np.float64)
    for start in range(0, eta.shape[0], batch_size):
        stop = min(start + batch_size, eta.shape[0])
        # Keep the compiled shape fixed.  The default sample size is divisible
        # by 16, but padding makes arbitrary CLI choices deterministic too.
        count = stop - start
        eta_batch = np.zeros((batch_size, eta.shape[1]), dtype=np.float64)
        xi_batch = np.zeros_like(eta_batch)
        depth_batch = np.ones(batch_size, dtype=np.float64)
        filter_batch = np.ones(batch_size, dtype=np.float64)
        eta_batch[:count] = eta[start:stop]
        xi_batch[:count] = xi[start:stop]
        depth_batch[:count] = depth[start:stop]
        filter_batch[:count] = label_filter_fractions[start:stop]
        computed = np.asarray(
            jax.device_get(
                evaluate_batch(eta_batch, xi_batch, depth_batch, filter_batch)
            ),
            dtype=np.float64,
        )
        output[start:stop] = computed[:count]
        if start == 0 or stop == eta.shape[0] or (start // batch_size + 1) % 8 == 0:
            print(f"recomputed {stop}/{eta.shape[0]} rows", flush=True)
    return output


def compute_metrics(
    stored: np.ndarray,
    recomputed: np.ndarray,
    *,
    length: float,
    label_filter_fractions: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    n_rows, nx = stored.shape
    valid = np.all(np.isfinite(stored), axis=1) & np.all(np.isfinite(recomputed), axis=1)
    metrics = {name: np.zeros(n_rows, dtype=np.float64) for name in METRIC_NAMES}
    if not np.any(valid):
        return metrics, valid

    stored_valid = stored[valid].astype(np.float64, copy=False)
    reference = recomputed[valid]
    error = stored_valid - reference
    reference_l2 = np.linalg.norm(reference, axis=1)
    error_l2 = np.linalg.norm(error, axis=1)
    metrics["relative_l2"][valid] = error_l2 / np.maximum(reference_l2, 1.0e-300)
    metrics["max_abs_error"][valid] = np.max(np.abs(error), axis=1)

    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    h1_weight = np.sqrt(1.0 + k**2)[None, :]
    reference_hat = np.fft.fft(reference, axis=1)
    error_hat = np.fft.fft(error, axis=1)
    reference_h1 = np.linalg.norm(h1_weight * reference_hat, axis=1)
    error_h1 = np.linalg.norm(h1_weight * error_hat, axis=1)
    metrics["relative_h1"][valid] = error_h1 / np.maximum(reference_h1, 1.0e-300)

    stored_l2 = np.linalg.norm(stored_valid, axis=1)
    correlation_denominator = stored_l2 * reference_l2
    correlation = np.ones(reference.shape[0], dtype=np.float64)
    nonzero = correlation_denominator > 1.0e-300
    correlation[nonzero] = (
        np.sum(stored_valid[nonzero] * reference[nonzero], axis=1)
        / correlation_denominator[nonzero]
    )
    metrics["correlation"][valid] = np.clip(correlation, -1.0, 1.0)

    active_threshold = np.maximum(
        1.0e-12,
        1.0e-6 * np.max(np.abs(reference), axis=1),
    )
    active = np.abs(reference) > active_threshold[:, None]
    active_count = np.sum(active, axis=1)
    sign_match = np.sum(
        active & (np.signbit(stored_valid) == np.signbit(reference)), axis=1
    )
    metrics["sign_agreement_active"][valid] = np.divide(
        sign_match,
        active_count,
        out=np.ones_like(sign_match, dtype=np.float64),
        where=active_count > 0,
    )

    stored_hat = np.fft.fft(stored_valid, axis=1)
    valid_fractions = label_filter_fractions[valid]
    rejected = np.abs(k)[None, :] > (
        valid_fractions[:, None] * np.max(np.abs(k))
    )
    rejected_band = np.linalg.norm(np.where(rejected, stored_hat, 0.0), axis=1)
    metrics["stored_rejected_band_relative_l2"][valid] = rejected_band / np.maximum(
        np.linalg.norm(stored_hat, axis=1),
        1.0e-300,
    )
    return metrics, valid


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    result = np.quantile(finite, QUANTILES)
    return {
        f"q{int(round(100.0 * quantile)):02d}": float(value)
        for quantile, value in zip(QUANTILES, result.tolist())
    }


def _metric_summary(
    metrics: dict[str, np.ndarray],
    selector: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    selected_valid = selector & valid
    summary: dict[str, Any] = {
        "rows": int(np.sum(selector)),
        "metric_valid_rows": int(np.sum(selected_valid)),
        "metrics": {
            name: _quantiles(values[selected_valid])
            for name, values in metrics.items()
        },
    }
    relative_l2 = metrics["relative_l2"][selected_valid]
    summary["relative_l2_exceedance_counts"] = {
        "1e-4": int(np.sum(relative_l2 > 1.0e-4)),
        "1e-3": int(np.sum(relative_l2 > 1.0e-3)),
        "1e-2": int(np.sum(relative_l2 > 1.0e-2)),
        "1e-1": int(np.sum(relative_l2 > 1.0e-1)),
    }
    summary["negative_correlation_rows"] = int(
        np.sum(metrics["correlation"][selected_valid] < 0.0)
    )
    return summary


def _zip_source_record(path: Path) -> dict[str, Any]:
    members = ("eta.npy", "xi.npy", "gxi.npy", "depth.npy", "time.npy", "source.npy")
    with zipfile.ZipFile(path) as archive:
        member_info = {
            member: {
                "crc32": f"{archive.getinfo(member).CRC:08x}",
                "file_size": int(archive.getinfo(member).file_size),
                "compress_type": int(archive.getinfo(member).compress_type),
            }
            for member in members
        }
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "members": member_info,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(serialized + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    dataset = args.dataset.resolve()
    meta_path = dataset.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    length = float(meta.get("length", 2.0 * np.pi))
    legend = {int(key): str(value) for key, value in meta["source_legend"].items()}

    arrays = {
        member: mmap_npy_member(dataset, f"{member}.npy")
        for member in ("eta", "xi", "gxi", "depth", "time", "source")
    }
    n_rows = int(arrays["source"].shape[0])
    nx = int(arrays["eta"].shape[1])
    expected_shapes = {
        "eta": (n_rows, nx),
        "xi": (n_rows, nx),
        "gxi": (n_rows, nx),
        "depth": (n_rows,),
        "time": (n_rows,),
        "source": (n_rows,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name} shape {arrays[name].shape} != {shape}")

    selection_started = time.perf_counter()
    sample, source_counts = build_sample(
        source_map=arrays["source"],
        eta_map=arrays["eta"],
        depth_map=arrays["depth"],
        length=length,
        highband_k=args.highband_k,
        candidate_count=args.candidate_rows_per_source,
        rows_per_source=args.rows_per_source,
        seed=args.seed,
    )
    selection_seconds = time.perf_counter() - selection_started
    print(
        f"selected {sample.global_indices.size} rows across {len(source_counts)} sources "
        f"in {selection_seconds:.1f}s",
        flush=True,
    )

    indices = sample.global_indices
    eta = np.asarray(arrays["eta"][indices], dtype=np.float64)
    xi = np.asarray(arrays["xi"][indices], dtype=np.float64)
    stored_gxi = np.asarray(arrays["gxi"][indices], dtype=np.float32)
    depth = np.asarray(arrays["depth"][indices], dtype=np.float64)
    times = np.asarray(arrays["time"][indices], dtype=np.float64)
    filtered_source_ids = tuple(
        int(value.strip())
        for value in str(args.filtered_source_ids).split(",")
        if value.strip()
    )
    available_source_ids = set(source_counts)
    unknown_filtered_ids = set(filtered_source_ids) - available_source_ids
    if unknown_filtered_ids:
        raise ValueError(f"filtered source IDs not present in dataset: {unknown_filtered_ids}")
    label_filter_fractions = np.where(
        np.isin(sample.source_ids, filtered_source_ids),
        float(args.filter_fraction),
        1.0,
    ).astype(np.float64)

    input_finite = (
        np.all(np.isfinite(eta), axis=1)
        & np.all(np.isfinite(xi), axis=1)
        & np.isfinite(depth)
    )
    stored_finite = np.all(np.isfinite(stored_gxi), axis=1)
    recompute_started = time.perf_counter()
    recomputed_gxi = recompute_labels(
        eta,
        xi,
        depth,
        label_filter_fractions,
        length=length,
        dno_order=args.dno_order,
        pad_factor=args.pad_factor,
        batch_size=args.batch_size,
    )
    recompute_seconds = time.perf_counter() - recompute_started
    recomputed_finite = np.all(np.isfinite(recomputed_gxi), axis=1)
    metrics, metric_valid = compute_metrics(
        stored_gxi,
        recomputed_gxi,
        length=length,
        label_filter_fractions=label_filter_fractions,
    )
    metric_valid &= input_finite

    source_summaries: dict[str, Any] = {}
    for source_id in sorted(source_counts):
        selector = sample.source_ids == source_id
        summary = _metric_summary(metrics, selector, metric_valid)
        summary.update(
            {
                "source_id": source_id,
                "source_name": legend.get(source_id, f"unknown_{source_id}"),
                "dataset_rows": source_counts[source_id],
                "label_filter_fraction": float(label_filter_fractions[selector][0]),
                "nonfinite_input_rows": int(np.sum(selector & ~input_finite)),
                "nonfinite_stored_label_rows": int(np.sum(selector & ~stored_finite)),
                "nonfinite_recomputed_label_rows": int(
                    np.sum(selector & ~recomputed_finite)
                ),
                "sample_depth": _quantiles(depth[selector]),
                "sample_amplitude_over_depth": _quantiles(
                    sample.amplitude_over_depth[selector]
                ),
                "sample_eta_highband_fraction": _quantiles(
                    sample.eta_highband_fraction[selector]
                ),
            }
        )
        source_summaries[str(source_id)] = summary

    all_rows = np.ones(indices.size, dtype=bool)
    pooled = _metric_summary(metrics, all_rows, metric_valid)
    nonfinite_any = ~(input_finite & stored_finite & recomputed_finite)
    negative_corr = metric_valid & (metrics["correlation"] < 0.0)
    total_seconds = time.perf_counter() - started
    script_path = Path(__file__).resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "audit_status": (
            "completed_no_nonfinite_or_sign_reversal"
            if not np.any(nonfinite_any) and not np.any(negative_corr)
            else "completed_with_anomalies"
        ),
        "interpretation_scope": (
            "Recomputation uses the stored float32 eta/xi inputs promoted to float64; "
            "differences therefore include the irreversible input-quantization error, "
            "not only label-storage error."
        ),
        "dataset": _zip_source_record(dataset),
        "dataset_meta": {
            "version": meta.get("version"),
            "n_samples": int(meta["n_samples"]),
            "nx": nx,
            "length": length,
            "source_legend": {str(key): value for key, value in sorted(legend.items())},
        },
        "script": {
            "path": str(script_path),
            "sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        },
        "execution": {
            "backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "process_niceness": int(os.nice(0)),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "seed": int(args.seed),
            "candidate_rows_per_source": int(args.candidate_rows_per_source),
            "rows_per_source": int(args.rows_per_source),
            "batch_size": int(args.batch_size),
            "selection_seconds": float(selection_seconds),
            "recompute_seconds": float(recompute_seconds),
            "total_seconds": float(total_seconds),
        },
        "operator": {
            "dtype": "float64",
            "dno_order": int(args.dno_order),
            "pad_factor": int(args.pad_factor),
            "label_protocol": "source_specific",
            "rollout_source_filter": "hard_abs_k_cutoff",
            "rollout_source_filter_fraction": float(args.filter_fraction),
            "rollout_source_filter_cutoff_abs_k": float(
                args.filter_fraction * (nx // 2)
            ),
            "filtered_rollout_source_ids": list(filtered_source_ids),
            "unfiltered_static_source_ids": sorted(
                available_source_ids - set(filtered_source_ids)
            ),
            "protocol_basis": {
                "0,1": "generate_random_sea_dataset.chunked_gxi_over_time applies low-pass",
                "2": "generate_linear_dataset.build_linear_batch stores raw series output",
                "3,4": "generate_stokes_dataset.build_stokes_batch stores raw series output",
                "5,6": "generate_tanaka_dataset_v2.chunked_gxi_over_time applies low-pass",
                "7,8,9": "generate_bf_dataset.chunked_gxi_over_time applies low-pass",
                "11,12,13": "generate_shallow_steep_dataset uses filtered rollout-label helper",
                "14": "generate_steep_tanaka_dataset uses filtered rollout-label helper",
            },
        },
        "sampling": {
            "method": "rank_stratified_4_depth_x_2_amplitude_x_2_eta_highband",
            "amplitude_diagnostic": "max_abs_eta / max(abs(depth), 1e-12)",
            "highband_diagnostic": (
                f"sqrt(eta Fourier energy at |k| >= {args.highband_k:g} / "
                "eta Fourier energy at |k| > 0)"
            ),
            "source_counts": {str(key): value for key, value in source_counts.items()},
            "available_source_ids": sorted(source_counts),
            "sample_rows": int(indices.size),
        },
        "nonfinite_counts": {
            "input_rows": int(np.sum(~input_finite)),
            "stored_label_rows": int(np.sum(~stored_finite)),
            "recomputed_label_rows": int(np.sum(~recomputed_finite)),
            "any_rows": int(np.sum(nonfinite_any)),
        },
        "pooled_source_balanced": pooled,
        "sources": source_summaries,
    }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    npz_path = args.output_prefix.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        global_index=indices,
        source_id=sample.source_ids,
        depth_bin=sample.depth_bins,
        amplitude_bin=sample.amplitude_bins,
        highband_bin=sample.highband_bins,
        label_filter_fraction=label_filter_fractions,
        depth=depth,
        time=times,
        amplitude_over_depth=sample.amplitude_over_depth,
        eta_highband_fraction=sample.eta_highband_fraction,
        input_finite=input_finite,
        stored_label_finite=stored_finite,
        recomputed_label_finite=recomputed_finite,
        metric_valid=metric_valid,
        stored_gxi=stored_gxi,
        recomputed_gxi=recomputed_gxi,
        **metrics,
    )
    _write_json(json_path, report)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {npz_path}", flush=True)
    print(json.dumps(report["nonfinite_counts"], sort_keys=True), flush=True)
    print(json.dumps(report["pooled_source_balanced"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
