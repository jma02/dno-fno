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
        description="Plot thresholded derivative sign-change count versus gxi row max/min on a tanaka_1 subsample."
    )
    parser.add_argument("--dataset", default="data/tanaka_1.npz")
    parser.add_argument("--output_dir", default="outputs/tanaka_1_roughness_binned_gxi")
    parser.add_argument("--samples_per_shard", type=int, default=1000)
    parser.add_argument("--relative_derivative_threshold", type=float, default=0.05)
    parser.add_argument("--min_count_for_plot", type=int, default=50)
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


def count_sign_changes_with_deadzone(diff: np.ndarray, relative_threshold: float) -> np.ndarray:
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
            counts[row_idx] = 0
            continue
        counts[row_idx] = int(np.count_nonzero(nz[1:] != nz[:-1]) + (nz[0] != nz[-1]))
    return counts


def sample_rows(
    archive: np.lib.npyio.NpzFile,
    tags: list[str],
    samples_per_shard: int,
    seed: int,
    relative_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.asarray(archive["x"], dtype=np.float64)
    dx = float(x[1] - x[0])

    sign_change_chunks: list[np.ndarray] = []
    row_max_chunks: list[np.ndarray] = []
    row_min_chunks: list[np.ndarray] = []

    for tag in tags:
        gxi = np.asarray(archive[f"gxi_batch_{tag}"], dtype=np.float32)
        finite_idx = np.flatnonzero(np.isfinite(gxi).all(axis=1))
        if finite_idx.size == 0:
            continue
        take = min(samples_per_shard, int(finite_idx.size))
        chosen = rng.choice(finite_idx, size=take, replace=False)
        rows = np.asarray(gxi[chosen], dtype=np.float64)
        diff = periodic_forward_diff(rows, dx)
        sign_changes = count_sign_changes_with_deadzone(diff, relative_threshold)
        sign_change_chunks.append(sign_changes)
        row_max_chunks.append(np.max(rows, axis=1))
        row_min_chunks.append(np.min(rows, axis=1))

    return (
        np.concatenate(sign_change_chunks, axis=0),
        np.concatenate(row_max_chunks, axis=0),
        np.concatenate(row_min_chunks, axis=0),
    )


def summarize_by_sign_changes(sign_changes: np.ndarray, values: np.ndarray) -> dict[str, list[float | int | None]]:
    unique_counts = np.arange(int(np.min(sign_changes)), int(np.max(sign_changes)) + 1, dtype=np.int32)
    counts: list[int] = []
    means: list[float | None] = []
    medians: list[float | None] = []
    q10: list[float | None] = []
    q90: list[float | None] = []
    for k in unique_counts:
        members = values[sign_changes == k]
        counts.append(int(members.size))
        if members.size == 0:
            means.append(None)
            medians.append(None)
            q10.append(None)
            q90.append(None)
            continue
        means.append(float(np.mean(members)))
        medians.append(float(np.median(members)))
        q10.append(float(np.quantile(members, 0.10)))
        q90.append(float(np.quantile(members, 0.90)))
    return {
        "sign_changes": unique_counts.tolist(),
        "counts": counts,
        "means": means,
        "medians": medians,
        "q10": q10,
        "q90": q90,
    }


def plot_sign_change_summary(
    output_path: Path,
    max_summary: dict[str, list[float | int | None]],
    min_summary: dict[str, list[float | int | None]],
    min_count_for_plot: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    panels = [
        (axes[0], max_summary, r"$\max_x G(\eta)\xi$"),
        (axes[1], min_summary, r"$\min_x G(\eta)\xi$"),
    ]
    for ax, summary, ylabel in panels:
        sign_changes = np.asarray(summary["sign_changes"], dtype=np.int32)
        counts = np.asarray(summary["counts"], dtype=np.int32)
        means = np.asarray([np.nan if v is None else v for v in summary["means"]], dtype=np.float64)
        medians = np.asarray([np.nan if v is None else v for v in summary["medians"]], dtype=np.float64)
        q10 = np.asarray([np.nan if v is None else v for v in summary["q10"]], dtype=np.float64)
        q90 = np.asarray([np.nan if v is None else v for v in summary["q90"]], dtype=np.float64)
        keep = counts >= min_count_for_plot
        ax.fill_between(sign_changes[keep], q10[keep], q90[keep], color="tab:blue", alpha=0.18, label="q10-q90")
        ax.plot(sign_changes[keep], means[keep], color="tab:blue", linewidth=2.0, label="mean")
        ax.plot(sign_changes[keep], medians[keep], color="tab:orange", linewidth=1.8, linestyle="--", label="median")
        ax.set_xlabel("thresholded derivative sign changes")
        ax.set_ylabel(ylabel)
        twin = ax.twinx()
        twin.bar(sign_changes[keep], counts[keep], width=1.0, color="0.85", alpha=0.5)
        twin.set_ylabel("count", color="0.35")
        twin.tick_params(axis="y", colors="0.35")
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("gxi row extrema versus thresholded derivative sign changes")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for stale_name in ("roughness_to_gxi_heatmaps.png", "roughness_to_gxi_bin_stats.png", "sign_changes_to_gxi.png"):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    with np.load(args.dataset, allow_pickle=False) as archive:
        tags = list_shard_tags(archive)
        sign_changes, row_max, row_min = sample_rows(
            archive,
            tags,
            samples_per_shard=args.samples_per_shard,
            seed=args.seed,
            relative_threshold=args.relative_derivative_threshold,
        )

    max_summary = summarize_by_sign_changes(sign_changes, row_max)
    min_summary = summarize_by_sign_changes(sign_changes, row_min)

    plot_sign_change_summary(
        output_dir / "sign_changes_to_gxi.png",
        max_summary,
        min_summary,
        min_count_for_plot=args.min_count_for_plot,
    )

    summary = {
        "dataset": str(Path(args.dataset).resolve()),
        "method": "uniform per-shard finite-row sample, thresholded derivative sign changes",
        "samples_per_shard": args.samples_per_shard,
        "sampled_count": int(sign_changes.size),
        "relative_derivative_threshold": args.relative_derivative_threshold,
        "min_count_for_plot": args.min_count_for_plot,
        "sign_change_stats": {
            "min": int(np.min(sign_changes)),
            "max": int(np.max(sign_changes)),
            "mean": float(np.mean(sign_changes)),
        },
        "gxi_extrema_stats": {
            "row_max": {
                "min": float(np.min(row_max)),
                "max": float(np.max(row_max)),
                "mean": float(np.mean(row_max)),
            },
            "row_min": {
                "min": float(np.min(row_min)),
                "max": float(np.max(row_min)),
                "mean": float(np.mean(row_min)),
            },
        },
        "binned": {
            "gxi_max_given_sign_changes": max_summary,
            "gxi_min_given_sign_changes": min_summary,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
