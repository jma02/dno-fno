from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot shuffled snapshot samples from a streamed Tanaka dataset archive."
    )
    parser.add_argument("--dataset", default="data/tanaka_1.npz")
    parser.add_argument("--output_dir", default="outputs/tanaka_1_snapshots")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_stride", type=int, default=500_000)
    parser.add_argument("--per_page", type=int, default=5)
    return parser.parse_args()


def load_meta(dataset_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(dataset_path, mode="r") as zf:
        return json.loads(zf.read("meta.json"))


def select_indices(n_samples: int, seed: int, sample_stride: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_samples)
    positions = np.arange(0, n_samples, sample_stride, dtype=np.int64)
    return positions, permutation[positions]


def fetch_records(
    dataset_path: Path,
    selected_indices: np.ndarray,
    samples_per_full_batch: int,
) -> list[dict[str, object]]:
    grouped: dict[int, list[tuple[int, int]]] = {}
    for record_idx, global_idx in enumerate(selected_indices.tolist()):
        batch_idx = int(global_idx // samples_per_full_batch)
        local_idx = int(global_idx % samples_per_full_batch)
        grouped.setdefault(batch_idx, []).append((record_idx, local_idx))

    records: list[dict[str, object] | None] = [None] * len(selected_indices)
    archive = np.load(dataset_path)
    x = np.asarray(archive["x"])
    for batch_idx, offsets in grouped.items():
        batch_tag = f"{batch_idx:04d}"
        eta = np.asarray(archive[f"eta_batch_{batch_tag}"])
        xi = np.asarray(archive[f"xi_batch_{batch_tag}"])
        gxi = np.asarray(archive[f"gxi_batch_{batch_tag}"])
        times = np.asarray(archive[f"time_batch_{batch_tag}"])
        case_ids = np.asarray(archive[f"case_id_batch_{batch_tag}"])
        for record_idx, local_idx in offsets:
            records[record_idx] = {
                "global_index": int(selected_indices[record_idx]),
                "batch_idx": batch_idx,
                "local_idx": local_idx,
                "time": float(times[local_idx]),
                "case_id": int(case_ids[local_idx]),
                "x": x,
                "eta": eta[local_idx],
                "xi": xi[local_idx],
                "gxi": gxi[local_idx],
            }
    return [record for record in records if record is not None]


def field_limits(records: list[dict[str, object]]) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for key in ("eta", "xi", "gxi"):
        values = np.concatenate([np.asarray(record[key]).ravel() for record in records])
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if vmin == vmax:
            pad = 1.0 if vmin == 0.0 else 0.05 * abs(vmin)
            limits[key] = (vmin - pad, vmax + pad)
        else:
            pad = 0.05 * (vmax - vmin)
            limits[key] = (vmin - pad, vmax + pad)
    return limits


def plot_pages(
    records: list[dict[str, object]],
    output_dir: Path,
    per_page: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    limits = field_limits(records)
    labels = [("eta", "eta"), ("xi", "xi"), ("gxi", "Geta_xi")]
    page_paths: list[str] = []
    for page_start in range(0, len(records), per_page):
        page_records = records[page_start : page_start + per_page]
        fig, axes = plt.subplots(
            nrows=len(page_records),
            ncols=3,
            figsize=(14, 3.2 * len(page_records)),
            squeeze=False,
        )
        for row_idx, record in enumerate(page_records):
            x = np.asarray(record["x"])
            for col_idx, (key, ylabel) in enumerate(labels):
                ax = axes[row_idx, col_idx]
                ax.plot(x, np.asarray(record[key]), color="#d62728", linewidth=1.5)
                ax.set_xlim(float(x[0]), float(x[-1]))
                ax.set_ylim(*limits[key])
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.25)
                if row_idx == len(page_records) - 1:
                    ax.set_xlabel("x")
                title = (
                    f"idx={record['global_index']} | case={record['case_id']} | "
                    f"t={record['time']:.3f}"
                )
                ax.set_title(title, fontsize=9)
        fig.tight_layout()
        page_idx = page_start // per_page
        page_path = output_dir / f"snapshots_page_{page_idx:02d}.png"
        fig.savefig(page_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        page_paths.append(str(page_path))
    return page_paths


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    meta = load_meta(dataset_path)
    n_samples = int(meta["target_samples"])
    samples_per_full_batch = int(meta["samples_per_full_batch"])

    positions, selected_indices = select_indices(n_samples, args.seed, args.sample_stride)
    records = fetch_records(dataset_path, selected_indices, samples_per_full_batch)
    page_paths = plot_pages(records, output_dir, args.per_page)

    summary = {
        "dataset": str(dataset_path),
        "seed": args.seed,
        "sample_stride": args.sample_stride,
        "positions": positions.tolist(),
        "selected_indices": selected_indices.tolist(),
        "page_paths": page_paths,
        "records": [
            {
                "global_index": int(record["global_index"]),
                "batch_idx": int(record["batch_idx"]),
                "local_idx": int(record["local_idx"]),
                "time": float(record["time"]),
                "case_id": int(record["case_id"]),
            }
            for record in records
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
