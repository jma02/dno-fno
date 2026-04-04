from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SHARD_PATTERN = re.compile(r"^gxi_batch_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot raw and thresholded derivative sign-change count histograms for tanaka_1 gxi rows."
    )
    parser.add_argument("--dataset", default="data/tanaka_1.npz")
    parser.add_argument("--output_dir", default="outputs/tanaka_1_sign_change_histograms")
    parser.add_argument("--samples_per_shard", type=int, default=1000)
    parser.add_argument("--all_rows", action="store_true")
    parser.add_argument("--relative_derivative_threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def list_shard_tags(archive: np.lib.npyio.NpzFile) -> list[str]:
    tags: list[str] = []
    for key in archive.files:
        match = SHARD_PATTERN.match(key)
        if match is not None:
            tags.append(match.group(1))
    return sorted(tags)


def periodic_forward_diff(values: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(values, -1, axis=1) - values) / dx


def count_raw_sign_changes(diff: np.ndarray) -> np.ndarray:
    sign = diff > 0.0
    changes = np.count_nonzero(sign[:, 1:] != sign[:, :-1], axis=1)
    wrap = (sign[:, 0] != sign[:, -1]).astype(np.int32)
    return np.asarray(changes + wrap, dtype=np.int32)


def count_thresholded_sign_changes(diff: np.ndarray, relative_threshold: float) -> np.ndarray:
    counts = np.zeros(diff.shape[0], dtype=np.int32)
    for row_idx, row in enumerate(diff):
        row_abs = np.abs(row)
        row_scale = float(np.max(row_abs))
        if row_scale == 0.0:
            continue
        threshold = relative_threshold * row_scale
        signs = np.where(row > threshold, 1, np.where(row < -threshold, -1, 0))
        nz = signs[signs != 0]
        if nz.size <= 1:
            continue
        counts[row_idx] = int(np.count_nonzero(nz[1:] != nz[:-1]) + (nz[0] != nz[-1]))
    return counts


def sample_sign_change_counts(
    archive: np.lib.npyio.NpzFile,
    tags: list[str],
    samples_per_shard: int,
    seed: int,
    relative_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.asarray(archive["x"], dtype=np.float64)
    dx = float(x[1] - x[0])

    raw_chunks: list[np.ndarray] = []
    thresholded_chunks: list[np.ndarray] = []
    for tag in tags:
        gxi = np.asarray(archive[f"gxi_batch_{tag}"], dtype=np.float32)
        finite_idx = np.flatnonzero(np.isfinite(gxi).all(axis=1))
        if finite_idx.size == 0:
            continue
        take = min(samples_per_shard, int(finite_idx.size))
        chosen = rng.choice(finite_idx, size=take, replace=False)
        rows = np.asarray(gxi[chosen], dtype=np.float64)
        diff = periodic_forward_diff(rows, dx)
        raw_chunks.append(count_raw_sign_changes(diff))
        thresholded_chunks.append(count_thresholded_sign_changes(diff, relative_threshold))

    return np.concatenate(raw_chunks, axis=0), np.concatenate(thresholded_chunks, axis=0)


def count_thresholded_sign_changes_vectorized(diff: np.ndarray, relative_threshold: float) -> np.ndarray:
    row_scales = np.max(np.abs(diff), axis=1, keepdims=True)
    thresholds = relative_threshold * row_scales
    signs = np.where(diff > thresholds, 1, np.where(diff < -thresholds, -1, 0)).astype(np.int8)
    nonzero = signs != 0
    nonzero_counts = np.count_nonzero(nonzero, axis=1)

    max_nonzero = int(np.max(nonzero_counts))
    if max_nonzero == 0:
        return np.zeros(diff.shape[0], dtype=np.int32)

    nz_signs = np.zeros((diff.shape[0], max_nonzero), dtype=np.int8)
    order = np.cumsum(nonzero, axis=1) - 1
    row_ids, col_ids = np.nonzero(nonzero)
    nz_signs[row_ids, order[row_ids, col_ids]] = signs[row_ids, col_ids]

    valid_next = (np.arange(max_nonzero - 1)[None, :] < (nonzero_counts[:, None] - 1))
    adjacent_changes = (nz_signs[:, 1:] != nz_signs[:, :-1]) & valid_next
    wrap_changes = (nonzero_counts > 1) & (nz_signs[:, 0] != nz_signs[np.arange(diff.shape[0]), np.maximum(nonzero_counts - 1, 0)])
    return adjacent_changes.sum(axis=1).astype(np.int32) + wrap_changes.astype(np.int32)


def stream_thresholded_histogram(
    archive: np.lib.npyio.NpzFile,
    tags: list[str],
    relative_threshold: float,
) -> tuple[np.ndarray, int]:
    histogram: np.ndarray | None = None
    total_count = 0
    x = np.asarray(archive["x"], dtype=np.float64)
    dx = float(x[1] - x[0])

    for tag in tags:
        gxi = np.asarray(archive[f"gxi_batch_{tag}"], dtype=np.float32)
        finite_rows = gxi[np.isfinite(gxi).all(axis=1)]
        if finite_rows.size == 0:
            continue
        diff = periodic_forward_diff(np.asarray(finite_rows, dtype=np.float32), dx)
        thresholded = count_thresholded_sign_changes_vectorized(diff, relative_threshold)
        bincount = np.bincount(thresholded)
        if histogram is None:
            histogram = bincount.astype(np.int64)
        else:
            if bincount.size > histogram.size:
                histogram = np.pad(histogram, (0, bincount.size - histogram.size))
            histogram[:bincount.size] += bincount.astype(np.int64)
        total_count += int(thresholded.size)

    if histogram is None:
        histogram = np.zeros(1, dtype=np.int64)
    return histogram, total_count


def quantile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": int(np.min(values)),
        "max": int(np.max(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "p999": float(np.quantile(values, 0.999)),
    }


def plot_histograms(output_path: Path, raw_counts: np.ndarray, thresholded_counts: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    panels = [
        (axes[0], raw_counts, "Raw derivative sign changes"),
        (axes[1], thresholded_counts, "Thresholded derivative sign changes"),
    ]
    for ax, values, title in panels:
        bins = np.arange(int(np.min(values)), int(np.max(values)) + 2)
        ax.hist(values, bins=bins, color="tab:blue", alpha=0.85, log=True)
        ax.set_xlabel("sign changes")
        ax.set_ylabel("count")
        ax.set_title(title)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_thresholded_histogram(output_path: Path, histogram: np.ndarray) -> None:
    support = np.arange(histogram.size, dtype=np.int32)
    keep = histogram > 0
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
    ax.bar(support[keep], histogram[keep], width=1.0, color="tab:blue", alpha=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("thresholded derivative sign changes")
    ax.set_ylabel("count")
    ax.set_title("Thresholded derivative sign changes over all finite rows")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.dataset, allow_pickle=False) as archive:
        tags = list_shard_tags(archive)
        if args.all_rows:
            thresholded_histogram, total_count = stream_thresholded_histogram(
                archive,
                tags,
                relative_threshold=args.relative_derivative_threshold,
            )
            values = np.repeat(np.arange(thresholded_histogram.size, dtype=np.int32), thresholded_histogram)
            plot_thresholded_histogram(output_dir / "thresholded_sign_change_histogram_all_rows.png", thresholded_histogram)
            summary = {
                "dataset": str(Path(args.dataset).resolve()),
                "all_rows": True,
                "finite_row_count": int(total_count),
                "relative_derivative_threshold": args.relative_derivative_threshold,
                "thresholded": quantile_summary(values),
            }
        else:
            raw_counts, thresholded_counts = sample_sign_change_counts(
                archive,
                tags,
                samples_per_shard=args.samples_per_shard,
                seed=args.seed,
                relative_threshold=args.relative_derivative_threshold,
            )
            plot_histograms(output_dir / "sign_change_histograms.png", raw_counts, thresholded_counts)
            summary = {
                "dataset": str(Path(args.dataset).resolve()),
                "all_rows": False,
                "samples_per_shard": args.samples_per_shard,
                "sampled_count": int(raw_counts.size),
                "relative_derivative_threshold": args.relative_derivative_threshold,
                "raw": quantile_summary(raw_counts),
                "thresholded": quantile_summary(thresholded_counts),
            }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
