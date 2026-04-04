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
        description="Plot low- and high-sign-change gxi examples from a tanaka dataset subsample."
    )
    parser.add_argument("--dataset", default="data/tanaka_1.npz")
    parser.add_argument("--output_dir", default="outputs/tanaka_1_sign_change_examples")
    parser.add_argument("--samples_per_shard", type=int, default=1000)
    parser.add_argument("--relative_derivative_threshold", type=float, default=0.05)
    parser.add_argument("--n_examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def list_shard_tags(archive: np.lib.npyio.NpzFile) -> list[str]:
    tags: set[str] = set()
    for key in archive.files:
        match = SHARD_PATTERN.match(key)
        if match is not None:
            tags.add(match.group(2))
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
            continue
        counts[row_idx] = int(np.count_nonzero(nz[1:] != nz[:-1]) + (nz[0] != nz[-1]))
    return counts


def load_batch_field(archive: np.lib.npyio.NpzFile, field: str, tag: str) -> np.ndarray:
    return np.asarray(archive[f"{field}_batch_{tag}"])


def dedup_key(case_id: int, time_value: float) -> tuple[int, float]:
    return int(case_id) % 1_000_000, round(float(time_value), 6)


def collect_candidates(
    archive: np.lib.npyio.NpzFile,
    tags: list[str],
    samples_per_shard: int,
    relative_threshold: float,
    seed: int,
) -> list[dict[str, int | float]]:
    rng = np.random.default_rng(seed)
    x = np.asarray(archive["x"], dtype=np.float64)
    dx = float(x[1] - x[0])
    candidates: list[dict[str, int | float]] = []
    for tag in tags:
        gxi = load_batch_field(archive, "gxi", tag).astype(np.float64)
        finite_idx = np.flatnonzero(np.isfinite(gxi).all(axis=1))
        if finite_idx.size == 0:
            continue
        take = min(samples_per_shard, int(finite_idx.size))
        chosen = np.sort(rng.choice(finite_idx, size=take, replace=False))
        rows = gxi[chosen]
        diff = periodic_forward_diff(rows, dx)
        sign_changes = count_sign_changes_with_deadzone(diff, relative_threshold)
        times = load_batch_field(archive, "time", tag)[chosen]
        case_ids = load_batch_field(archive, "case_id", tag)[chosen]
        for local_idx, sc, time_value, case_id in zip(chosen, sign_changes, times, case_ids):
            candidates.append(
                {
                    "shard_id": int(tag),
                    "local_idx": int(local_idx),
                    "sign_changes": int(sc),
                    "time": float(time_value),
                    "case_id": int(case_id),
                }
            )
    return candidates


def select_examples(candidates: list[dict[str, int | float]], n_examples: int, reverse: bool) -> list[dict[str, int | float]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            int(item["sign_changes"]),
            abs(float(item["time"])),
            int(item["case_id"]),
        ),
        reverse=reverse,
    )
    selected: list[dict[str, int | float]] = []
    seen: set[tuple[int, float]] = set()
    for item in ordered:
        key = dedup_key(int(item["case_id"]), float(item["time"]))
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
        if len(selected) == n_examples:
            break
    return selected


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


def plot_examples(output_path: Path, x: np.ndarray, rows: list[dict[str, np.ndarray | int | float]], title: str) -> None:
    fig, axes = plt.subplots(len(rows), 3, figsize=(14, 2.7 * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        axes = np.asarray([axes])
    labels = [("eta", r"$\eta$"), ("xi", r"$\xi$"), ("gxi", r"$G(\eta)\xi$")]
    for row_idx, row in enumerate(rows):
        meta = f"sc={row['sign_changes']}  t={float(row['time']):.3f}  case={int(row['case_id']) % 1_000_000}"
        for col_idx, (field, ylabel) in enumerate(labels):
            ax = axes[row_idx, col_idx]
            values = np.asarray(row[field], dtype=np.float64)
            ax.plot(x, values, color="tab:blue", linewidth=1.8)
            ax.set_ylabel(ylabel)
            if col_idx == 1:
                ax.set_title(meta)
            if row_idx == len(rows) - 1:
                ax.set_xlabel("x")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.dataset, allow_pickle=False) as archive:
        x = np.asarray(archive["x"], dtype=np.float64)
        tags = list_shard_tags(archive)
        candidates = collect_candidates(
            archive,
            tags,
            samples_per_shard=args.samples_per_shard,
            relative_threshold=args.relative_derivative_threshold,
            seed=args.seed,
        )
        low = select_examples(candidates, args.n_examples, reverse=False)
        high = select_examples(candidates, args.n_examples, reverse=True)
        low_rows = fetch_rows(archive, low)
        high_rows = fetch_rows(archive, high)

    plot_examples(output_dir / "low_sign_change_examples.png", x, low_rows, "Low thresholded sign-change examples")
    plot_examples(output_dir / "high_sign_change_examples.png", x, high_rows, "High thresholded sign-change examples")

    summary = {
        "dataset": str(Path(args.dataset).resolve()),
        "samples_per_shard": args.samples_per_shard,
        "relative_derivative_threshold": args.relative_derivative_threshold,
        "n_examples": args.n_examples,
        "low_examples": low,
        "high_examples": high,
    }
    (output_dir / "selected_examples.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
