"""Compare the historical Gxi sign-flip rule with spectral-tail diagnostics.

The default audit scans batch 0 of the two corrected Tanaka-v2 sources.  It is
CPU-only and reads one source array at a time.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from solver.gen_data.clean_tanaka_dataset import (
    count_thresholded_sign_changes_vectorized,
    periodic_forward_diff,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = (
    f"tanaka_g0={ROOT / 'data/tanaka_2_adaptive_g0.npz'}",
    f"tanaka_g1={ROOT / 'data/tanaka_2_adaptive_g1.npz'}",
)
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.integer]


@dataclass(frozen=True)
class Source:
    label: str
    path: Path


@dataclass(frozen=True)
class ScanResult:
    label: str
    case_id: IntArray
    time: FloatArray
    sign_changes: IntArray
    dominant_derivative_mode: IntArray
    q_tail_energy: dict[int, FloatArray]
    derivative_tail_energy: dict[int, FloatArray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Source in LABEL=PATH form; repeat for multiple sources.",
    )
    parser.add_argument("--batch_tag", default="0000")
    parser.add_argument("--relative_threshold", type=float, default=0.03)
    parser.add_argument("--max_sign_changes", type=int, default=10)
    parser.add_argument("--cutoffs", default="5,16,32,64,80,96")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=ROOT / "notes/signflip_spectral_overlap_20260722.json",
    )
    parser.add_argument(
        "--output_npz",
        type=Path,
        default=ROOT / "notes/signflip_spectral_overlap_20260722.npz",
    )
    parser.add_argument(
        "--output_png",
        type=Path,
        default=ROOT / "notes/signflip_spectral_overlap_20260722.png",
    )
    return parser.parse_args()


def parse_source(value: str) -> Source:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise ValueError(f"source must have LABEL=PATH form; got {value!r}")
    return Source(label=label, path=Path(raw_path).expanduser().resolve())


def read_array(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    with archive.open(name) as handle:
        return np.load(handle, allow_pickle=False)


def scan_source(
    source: Source,
    batch_tag: str,
    relative_threshold: float,
    cutoffs: tuple[int, ...],
    chunk_size: int,
) -> ScanResult:
    with zipfile.ZipFile(source.path) as archive:
        q = np.asarray(
            read_array(archive, f"gxi_batch_{batch_tag}.npy"), dtype=np.float64
        )
        case_id = np.asarray(read_array(archive, f"case_id_batch_{batch_tag}.npy"))
        time = np.asarray(
            read_array(archive, f"time_batch_{batch_tag}.npy"), dtype=np.float64
        )
        x = np.asarray(read_array(archive, "x.npy"), dtype=np.float64)

    if q.ndim != 2 or case_id.shape != q.shape[:1] or time.shape != q.shape[:1]:
        raise ValueError(
            f"incompatible shapes for {source.label}: q={q.shape}, "
            f"case_id={case_id.shape}, time={time.shape}"
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    n_modes = q.shape[-1] // 2 + 1
    modes = np.arange(n_modes, dtype=np.float64)
    parseval_weights = np.full(n_modes, 2.0, dtype=np.float64)
    parseval_weights[0] = 1.0
    if q.shape[-1] % 2 == 0:
        parseval_weights[-1] = 1.0

    sign_parts: list[IntArray] = []
    dominant_parts: list[IntArray] = []
    q_tail_parts: dict[int, list[FloatArray]] = {cutoff: [] for cutoff in cutoffs}
    derivative_tail_parts: dict[int, list[FloatArray]] = {
        cutoff: [] for cutoff in cutoffs
    }
    dx = float(x[1] - x[0])

    for start in range(0, q.shape[0], chunk_size):
        rows = q[start : start + chunk_size]
        differences = periodic_forward_diff(rows, dx)
        sign_parts.append(
            count_thresholded_sign_changes_vectorized(differences, relative_threshold)
        )

        spectrum = np.fft.rfft(rows, axis=-1)
        q_energy = parseval_weights[None, :] * np.abs(spectrum) ** 2
        derivative_energy = q_energy * modes[None, :] ** 2
        q_total = np.maximum(np.sum(q_energy, axis=-1), np.finfo(np.float64).tiny)
        derivative_total = np.maximum(
            np.sum(derivative_energy, axis=-1), np.finfo(np.float64).tiny
        )
        dominant_parts.append(
            np.argmax(derivative_energy[:, 1:], axis=-1).astype(np.int32) + 1
        )

        for cutoff in cutoffs:
            tail = modes > cutoff
            q_tail_parts[cutoff].append(np.sum(q_energy[:, tail], axis=-1) / q_total)
            derivative_tail_parts[cutoff].append(
                np.sum(derivative_energy[:, tail], axis=-1) / derivative_total
            )

    concatenate = lambda parts: np.concatenate(parts, axis=0)  # noqa: E731
    return ScanResult(
        label=source.label,
        case_id=case_id,
        time=time,
        sign_changes=concatenate(sign_parts),
        dominant_derivative_mode=concatenate(dominant_parts),
        q_tail_energy={
            cutoff: concatenate(parts) for cutoff, parts in q_tail_parts.items()
        },
        derivative_tail_energy={
            cutoff: concatenate(parts)
            for cutoff, parts in derivative_tail_parts.items()
        },
    )


def quantiles(values: FloatArray | IntArray) -> dict[str, float]:
    probabilities = (0.0, 0.01, 0.5, 0.95, 0.99, 1.0)
    return {
        label: float(value)
        for label, value in zip(
            ("min", "p01", "median", "p95", "p99", "max"),
            np.quantile(values, probabilities),
            strict=True,
        )
    }


def containment_summary(
    values: FloatArray, rejected: NDArray[np.bool_]
) -> dict[str, object]:
    rejected_values = values[rejected]
    retained_values = values[~rejected]
    if rejected_values.size == 0 or retained_values.size == 0:
        raise ValueError("containment summary requires both rejected and retained rows")

    summaries: dict[str, object] = {
        "legacy_rejected": quantiles(rejected_values),
        "legacy_retained": quantiles(retained_values),
    }
    for label, missed_fraction in (("recall_1", 0.0), ("recall_0.99", 0.01)):
        sorted_rejected = np.sort(rejected_values)
        allowed_misses = int(np.floor(missed_fraction * sorted_rejected.size))
        threshold = float(sorted_rejected[allowed_misses])
        summaries[label] = {
            "threshold": threshold,
            "realized_recall": float(np.mean(rejected_values >= threshold)),
            "legacy_retained_flag_rate": float(np.mean(retained_values >= threshold)),
        }
    return summaries


def summarize_result(
    result: ScanResult,
    max_sign_changes: int,
    example_cutoff: int,
) -> dict[str, object]:
    rejected = result.sign_changes > max_sign_changes
    example_metric = result.derivative_tail_energy[example_cutoff]
    rejected_indices = np.flatnonzero(rejected)
    ranked_examples = rejected_indices[np.argsort(example_metric[rejected_indices])[:8]]

    return {
        "rows": int(result.sign_changes.size),
        "legacy_rejected_rows": int(np.sum(rejected)),
        "legacy_rejected_fraction": float(np.mean(rejected)),
        "sign_changes": quantiles(result.sign_changes),
        "q_tail_energy": {
            str(cutoff): containment_summary(values, rejected)
            for cutoff, values in result.q_tail_energy.items()
        },
        "derivative_tail_energy": {
            str(cutoff): containment_summary(values, rejected)
            for cutoff, values in result.derivative_tail_energy.items()
        },
        f"smallest_derivative_tail_{example_cutoff}_legacy_rejects": [
            {
                "row": int(index),
                "case_id": int(result.case_id[index]),
                "time": float(result.time[index]),
                "sign_changes": int(result.sign_changes[index]),
                "dominant_derivative_mode": int(result.dominant_derivative_mode[index]),
                "derivative_tail_energy": float(example_metric[index]),
            }
            for index in ranked_examples
        ],
    }


def save_arrays(
    path: Path, results: list[ScanResult], cutoffs: tuple[int, ...]
) -> None:
    labels = np.concatenate(
        [np.full(result.sign_changes.size, result.label) for result in results]
    )
    payload: dict[str, np.ndarray] = {
        "source": labels,
        "case_id": np.concatenate([result.case_id for result in results]),
        "time": np.concatenate([result.time for result in results]),
        "sign_changes": np.concatenate([result.sign_changes for result in results]),
        "dominant_derivative_mode": np.concatenate(
            [result.dominant_derivative_mode for result in results]
        ),
    }
    for cutoff in cutoffs:
        payload[f"q_tail_energy_k{cutoff}"] = np.concatenate(
            [result.q_tail_energy[cutoff] for result in results]
        )
        payload[f"derivative_tail_energy_k{cutoff}"] = np.concatenate(
            [result.derivative_tail_energy[cutoff] for result in results]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def plot_overlap(
    path: Path,
    results: list[ScanResult],
    max_sign_changes: int,
    cutoff: int,
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for result in results:
        axis.scatter(
            result.sign_changes,
            np.maximum(result.derivative_tail_energy[cutoff], 1e-20),
            s=5,
            alpha=0.22,
            label=result.label,
            rasterized=True,
        )
    axis.axvline(max_sign_changes, color="black", linestyle="--", linewidth=1.0)
    axis.set_yscale("log")
    axis.set_xlabel(r"legacy dead-zoned sign changes in $\partial_x q$")
    axis.set_ylabel(rf"fraction of $\|q_x\|_2^2$ in modes $|k|>{cutoff}$")
    axis.legend()
    axis.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    raw_sources = args.source if args.source is not None else DEFAULT_SOURCES
    sources = tuple(parse_source(value) for value in raw_sources)
    cutoffs = tuple(int(value) for value in args.cutoffs.split(","))
    if not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
        raise ValueError("cutoffs must be positive integers")

    results = [
        scan_source(
            source,
            batch_tag=args.batch_tag,
            relative_threshold=args.relative_threshold,
            cutoffs=cutoffs,
            chunk_size=args.chunk_size,
        )
        for source in sources
    ]
    example_cutoff = 80 if 80 in cutoffs else cutoffs[-1]
    summary = {
        "definition": {
            "q": "stored G(eta)xi",
            "legacy_relative_derivative_deadzone": args.relative_threshold,
            "legacy_reject_if_sign_changes_greater_than": args.max_sign_changes,
            "spectral_metric": (
                "Parseval energy fraction of q or q_x strictly above the "
                "listed Fourier-mode cutoff"
            ),
            "batch_tag": args.batch_tag,
            "sources": [
                {"label": source.label, "path": str(source.path)} for source in sources
            ],
        },
        "sources": {
            result.label: summarize_result(
                result, args.max_sign_changes, example_cutoff
            )
            for result in results
        },
        "limitation": (
            "The unavailable pre-cleaning tanaka_1 archive prevents a full "
            "rowwise audit of all 548360 historical rejections. This scan "
            "measures overlap and false positives on corrected periodic data."
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    save_arrays(args.output_npz, results, cutoffs)
    plot_overlap(args.output_png, results, args.max_sign_changes, example_cutoff)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
