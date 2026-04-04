from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format


GXI_PATTERN = re.compile(r"^gxi_batch_(\d+)\.npy$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter tanaka_1 by finiteness and thresholded gxi derivative sign changes."
    )
    parser.add_argument("--input", default="data/tanaka_1.npz")
    parser.add_argument("--output", default="data/tanaka_1_clean.npz")
    parser.add_argument("--max_sign_changes", type=int, default=40)
    parser.add_argument("--relative_derivative_threshold", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_source_tags(zf: zipfile.ZipFile) -> list[str]:
    tags: list[str] = []
    for name in zf.namelist():
        match = GXI_PATTERN.match(name)
        if match is not None:
            tags.append(match.group(1))
    return sorted(tags)


def read_npy(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name, "r") as handle:
        return np.load(handle, allow_pickle=False)


def write_npy(zf: zipfile.ZipFile, name: str, array: np.ndarray) -> None:
    with zf.open(name, mode="w", force_zip64=True) as handle:
        npy_format.write_array(handle, np.asarray(array), allow_pickle=False)


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


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
        output_path.unlink()

    total_rows = 0
    kept_rows = 0
    dropped_nonfinite = 0
    dropped_sign_changes = 0
    written_shards = 0

    with zipfile.ZipFile(input_path, mode="r") as src:
        tags = list_source_tags(src)
        source_meta = json.loads(src.read("meta.json"))
        x = read_npy(src, "x.npy").astype(np.float32)
        subsample_indices = read_npy(src, "subsample_indices.npy")
        subsample_times = read_npy(src, "subsample_times.npy")
        dx = float(x[1] - x[0])

        with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as dst:
            write_npy(dst, "x.npy", x)
            write_npy(dst, "subsample_indices.npy", subsample_indices)
            write_npy(dst, "subsample_times.npy", subsample_times)

            for src_idx, tag in enumerate(tags):
                eta = read_npy(src, f"eta_batch_{tag}.npy").astype(np.float32, copy=False)
                xi = read_npy(src, f"xi_batch_{tag}.npy").astype(np.float32, copy=False)
                gxi = read_npy(src, f"gxi_batch_{tag}.npy").astype(np.float32, copy=False)
                time = read_npy(src, f"time_batch_{tag}.npy")
                case_id = read_npy(src, f"case_id_batch_{tag}.npy")

                total_rows += int(eta.shape[0])
                finite_mask = (
                    np.isfinite(eta).all(axis=1)
                    & np.isfinite(xi).all(axis=1)
                    & np.isfinite(gxi).all(axis=1)
                )
                dropped_nonfinite += int((~finite_mask).sum())

                if not np.any(finite_mask):
                    continue

                eta_f = eta[finite_mask]
                xi_f = xi[finite_mask]
                gxi_f = gxi[finite_mask]
                time_f = time[finite_mask]
                case_id_f = case_id[finite_mask]

                diff = periodic_forward_diff(gxi_f, dx)
                sign_changes = count_thresholded_sign_changes_vectorized(diff, args.relative_derivative_threshold)
                keep_mask = sign_changes <= args.max_sign_changes
                dropped_sign_changes += int((~keep_mask).sum())

                if not np.any(keep_mask):
                    continue

                dest_tag = f"{written_shards:04d}"
                write_npy(dst, f"eta_batch_{dest_tag}.npy", eta_f[keep_mask])
                write_npy(dst, f"xi_batch_{dest_tag}.npy", xi_f[keep_mask])
                write_npy(dst, f"gxi_batch_{dest_tag}.npy", gxi_f[keep_mask])
                write_npy(dst, f"time_batch_{dest_tag}.npy", time_f[keep_mask])
                write_npy(dst, f"case_id_batch_{dest_tag}.npy", case_id_f[keep_mask])

                specs_name = f"specs_batch_{tag}.json"
                if specs_name in src.namelist():
                    dst.writestr(f"specs_batch_{dest_tag}.json", src.read(specs_name))

                kept_rows += int(keep_mask.sum())
                written_shards += 1

                if (src_idx + 1) % 10 == 0 or src_idx + 1 == len(tags):
                    print(
                        json.dumps(
                            {
                                "processed_source_shards": src_idx + 1,
                                "source_shards_total": len(tags),
                                "written_clean_shards": written_shards,
                                "rows_seen": total_rows,
                                "rows_kept": kept_rows,
                            }
                        ),
                        flush=True,
                    )

            clean_meta = dict(source_meta)
            clean_meta.update(
                {
                    "source_dataset": str(input_path),
                    "n_batches_planned": written_shards,
                    "samples_before_cleaning": total_rows,
                    "target_samples": kept_rows,
                    "samples_after_cleaning": kept_rows,
                    "rows_dropped_nonfinite": dropped_nonfinite,
                    "rows_dropped_sign_changes": dropped_sign_changes,
                    "clean_filter": {
                        "finite_only": True,
                        "max_thresholded_sign_changes": args.max_sign_changes,
                        "relative_derivative_threshold": args.relative_derivative_threshold,
                    },
                }
            )
            dst.writestr("meta.json", json.dumps(clean_meta, indent=2))

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "source_shards": len(tags),
        "clean_shards": written_shards,
        "rows_before": total_rows,
        "rows_after": kept_rows,
        "rows_dropped_nonfinite": dropped_nonfinite,
        "rows_dropped_sign_changes": dropped_sign_changes,
        "relative_derivative_threshold": args.relative_derivative_threshold,
        "max_sign_changes": args.max_sign_changes,
        "keep_fraction": kept_rows / total_rows if total_rows else 0.0,
    }
    summary_path = output_path.with_suffix(".clean.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
