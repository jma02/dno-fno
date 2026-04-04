from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SHARD_PATTERN = re.compile(r"^(eta|xi|gxi|time|case_id)_batch_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot fixed-count example buckets for thresholded derivative sign changes."
    )
    parser.add_argument("--dataset", default="data/tanaka_1.npz")
    parser.add_argument("--output_dir", default="outputs/tanaka_1_sign_change_buckets")
    parser.add_argument("--min_sign_changes", type=int, default=1)
    parser.add_argument("--max_sign_changes", type=int, default=40)
    parser.add_argument("--examples_per_bucket", type=int, default=5)
    parser.add_argument("--relative_derivative_threshold", type=float, default=0.05)
    return parser.parse_args()


def list_shard_tags(archive: np.lib.npyio.NpzFile) -> list[str]:
    tags: set[str] = set()
    for key in archive.files:
        match = SHARD_PATTERN.match(key)
        if match is not None:
            tags.add(match.group(2))
    return sorted(tags)


def load_batch_field(archive: np.lib.npyio.NpzFile, field: str, tag: str) -> np.ndarray:
    return np.asarray(archive[f"{field}_batch_{tag}"])


def periodic_forward_diff(values: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(values, -1, axis=1) - values) / dx


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

    valid_next = np.arange(max_nonzero - 1)[None, :] < (nonzero_counts[:, None] - 1)
    adjacent_changes = (nz_signs[:, 1:] != nz_signs[:, :-1]) & valid_next
    wrap_changes = (nonzero_counts > 1) & (
        nz_signs[:, 0] != nz_signs[np.arange(diff.shape[0]), np.maximum(nonzero_counts - 1, 0)]
    )
    return adjacent_changes.sum(axis=1).astype(np.int32) + wrap_changes.astype(np.int32)


def dedup_key(case_id: int, time_value: float) -> tuple[int, float]:
    return int(case_id) % 1_000_000, round(float(time_value), 6)


def collect_examples(
    archive: np.lib.npyio.NpzFile,
    tags: list[str],
    min_sign_changes: int,
    max_sign_changes: int,
    examples_per_bucket: int,
    relative_threshold: float,
) -> dict[int, list[dict[str, int | float]]]:
    x = np.asarray(archive["x"], dtype=np.float64)
    dx = float(x[1] - x[0])

    targets = range(min_sign_changes, max_sign_changes + 1)
    buckets: dict[int, list[dict[str, int | float]]] = {k: [] for k in targets}
    seen: dict[int, set[tuple[int, float]]] = {k: set() for k in targets}

    for tag in tags:
        gxi = load_batch_field(archive, "gxi", tag).astype(np.float32)
        finite_mask = np.isfinite(gxi).all(axis=1)
        finite_idx = np.flatnonzero(finite_mask)
        if finite_idx.size == 0:
            continue

        gxi_rows = np.asarray(gxi[finite_idx], dtype=np.float32)
        diff = periodic_forward_diff(gxi_rows, dx)
        sign_changes = count_thresholded_sign_changes_vectorized(diff, relative_threshold)
        relevant = (sign_changes >= min_sign_changes) & (sign_changes <= max_sign_changes)
        if not np.any(relevant):
            continue

        times = load_batch_field(archive, "time", tag)[finite_idx]
        case_ids = load_batch_field(archive, "case_id", tag)[finite_idx]
        relevant_indices = np.flatnonzero(relevant)
        for ridx in relevant_indices:
            sc = int(sign_changes[ridx])
            if len(buckets[sc]) >= examples_per_bucket:
                continue
            key = dedup_key(int(case_ids[ridx]), float(times[ridx]))
            if key in seen[sc]:
                continue
            seen[sc].add(key)
            buckets[sc].append(
                {
                    "shard_id": int(tag),
                    "local_idx": int(finite_idx[ridx]),
                    "sign_changes": sc,
                    "time": float(times[ridx]),
                    "case_id": int(case_ids[ridx]),
                }
            )
        if all(len(buckets[k]) >= examples_per_bucket for k in targets if len(seen[k]) >= 0):
            done = True
            for k in targets:
                if len(buckets[k]) < examples_per_bucket:
                    done = False
                    break
            if done:
                break

    return buckets


def fetch_rows(
    archive: np.lib.npyio.NpzFile,
    examples: list[dict[str, int | float]],
) -> list[dict[str, np.ndarray | int | float]]:
    rows: list[dict[str, np.ndarray | int | float]] = []
    for item in examples:
        tag = f"{int(item['shard_id']):04d}"
        idx = int(item["local_idx"])
        rows.append(
            {
                "eta": load_batch_field(archive, "eta", tag)[idx].astype(np.float64),
                "xi": load_batch_field(archive, "xi", tag)[idx].astype(np.float64),
                "gxi": load_batch_field(archive, "gxi", tag)[idx].astype(np.float64),
                "sign_changes": int(item["sign_changes"]),
                "time": float(item["time"]),
                "case_id": int(item["case_id"]),
                "shard_id": int(item["shard_id"]),
                "local_idx": idx,
            }
        )
    return rows


def plot_bucket(output_path: Path, x: np.ndarray, rows: list[dict[str, np.ndarray | int | float]], sign_changes: int) -> None:
    fig, axes = plt.subplots(len(rows), 3, figsize=(14, 2.7 * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        axes = np.asarray([axes])
    labels = [("eta", r"$\eta$"), ("xi", r"$\xi$"), ("gxi", r"$G(\eta)\xi$")]
    for row_idx, row in enumerate(rows):
        meta = f"sc={sign_changes}  t={float(row['time']):.3f}  case={int(row['case_id']) % 1_000_000}"
        for col_idx, (field, ylabel) in enumerate(labels):
            ax = axes[row_idx, col_idx]
            ax.plot(x, np.asarray(row[field], dtype=np.float64), color="tab:blue", linewidth=1.6)
            ax.set_ylabel(ylabel)
            if col_idx == 1:
                ax.set_title(meta)
            if row_idx == len(rows) - 1:
                ax.set_xlabel("x")
    fig.suptitle(f"Thresholded sign changes = {sign_changes}")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.dataset, allow_pickle=False) as archive:
        x = np.asarray(archive["x"], dtype=np.float64)
        tags = list_shard_tags(archive)
        buckets = collect_examples(
            archive,
            tags,
            min_sign_changes=args.min_sign_changes,
            max_sign_changes=args.max_sign_changes,
            examples_per_bucket=args.examples_per_bucket,
            relative_threshold=args.relative_derivative_threshold,
        )

        summary: dict[str, object] = {
            "dataset": str(Path(args.dataset).resolve()),
            "relative_derivative_threshold": args.relative_derivative_threshold,
            "examples_per_bucket": args.examples_per_bucket,
            "min_sign_changes": args.min_sign_changes,
            "max_sign_changes": args.max_sign_changes,
            "buckets": {},
        }

        for sc in range(args.min_sign_changes, args.max_sign_changes + 1):
            examples = buckets[sc]
            summary["buckets"][str(sc)] = examples
            if not examples:
                continue
            rows = fetch_rows(archive, examples)
            plot_bucket(output_dir / f"sign_changes_{sc:02d}.png", x, rows, sc)

    (output_dir / "selected_examples.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
