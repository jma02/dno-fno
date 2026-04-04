from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format


CASE_ID_PATTERN = re.compile(r"^case_id_batch_(\d+)\.npy$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drop mirrored duplicate Tanaka cases by keeping the canonical case_id modulo a fixed offset."
    )
    parser.add_argument("--input", default="data/tanaka_1_clean.npz")
    parser.add_argument("--output", default="data/tanaka_1_clean_dedup.npz")
    parser.add_argument("--case_id_modulus", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_source_tags(zf: zipfile.ZipFile) -> list[str]:
    tags: list[str] = []
    for name in zf.namelist():
        match = CASE_ID_PATTERN.match(name)
        if match is not None:
            tags.append(match.group(1))
    return sorted(tags)


def read_npy(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name, "r") as handle:
        return np.load(handle, allow_pickle=False)


def write_npy(zf: zipfile.ZipFile, name: str, array: np.ndarray) -> None:
    with zf.open(name, mode="w", force_zip64=True) as handle:
        npy_format.write_array(handle, np.asarray(array), allow_pickle=False)


def canonical_case_ids(case_id: np.ndarray, case_id_modulus: int) -> np.ndarray:
    return np.mod(case_id, case_id_modulus).astype(np.int64, copy=False)


def filter_specs_for_present_cases(specs: list[list[dict[str, object]]], case_id: np.ndarray, batch_size: int) -> list[list[dict[str, object]]]:
    if case_id.size == 0:
        return []
    batch_start = int((int(case_id.min()) // batch_size) * batch_size)
    present_case_ids = np.unique(case_id)
    filtered_specs: list[list[dict[str, object]]] = []
    for case_id_value in present_case_ids:
        local_case_idx = int(case_id_value) - batch_start
        if 0 <= local_case_idx < len(specs):
            filtered_specs.append(specs[local_case_idx])
    return filtered_specs


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
    dropped_duplicate_rows = 0
    written_shards = 0

    with zipfile.ZipFile(input_path, mode="r") as src:
        tags = list_source_tags(src)
        source_meta = json.loads(src.read("meta.json"))
        batch_size = int(source_meta["batch_size"])
        x = read_npy(src, "x.npy").astype(np.float32)
        subsample_indices = read_npy(src, "subsample_indices.npy")
        subsample_times = read_npy(src, "subsample_times.npy")

        with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as dst:
            write_npy(dst, "x.npy", x)
            write_npy(dst, "subsample_indices.npy", subsample_indices)
            write_npy(dst, "subsample_times.npy", subsample_times)

            for src_idx, tag in enumerate(tags):
                case_id = read_npy(src, f"case_id_batch_{tag}.npy").astype(np.int64, copy=False)
                canonical_ids = canonical_case_ids(case_id, args.case_id_modulus)
                keep_mask = case_id == canonical_ids

                total_rows += int(case_id.shape[0])
                dropped_duplicate_rows += int((~keep_mask).sum())

                if not np.any(keep_mask):
                    continue

                eta = read_npy(src, f"eta_batch_{tag}.npy").astype(np.float32, copy=False)
                xi = read_npy(src, f"xi_batch_{tag}.npy").astype(np.float32, copy=False)
                gxi = read_npy(src, f"gxi_batch_{tag}.npy").astype(np.float32, copy=False)
                time = read_npy(src, f"time_batch_{tag}.npy")

                eta_keep = eta[keep_mask]
                xi_keep = xi[keep_mask]
                gxi_keep = gxi[keep_mask]
                time_keep = time[keep_mask]
                case_id_keep = canonical_ids[keep_mask]

                dest_tag = f"{written_shards:04d}"
                write_npy(dst, f"eta_batch_{dest_tag}.npy", eta_keep)
                write_npy(dst, f"xi_batch_{dest_tag}.npy", xi_keep)
                write_npy(dst, f"gxi_batch_{dest_tag}.npy", gxi_keep)
                write_npy(dst, f"time_batch_{dest_tag}.npy", time_keep)
                write_npy(dst, f"case_id_batch_{dest_tag}.npy", case_id_keep)

                specs_name = f"specs_batch_{tag}.json"
                if specs_name in src.namelist():
                    specs = json.loads(src.read(specs_name))
                    filtered_specs = filter_specs_for_present_cases(specs, case_id_keep, batch_size)
                    dst.writestr(f"specs_batch_{dest_tag}.json", json.dumps(filtered_specs))

                kept_rows += int(keep_mask.sum())
                written_shards += 1

                if (src_idx + 1) % 10 == 0 or src_idx + 1 == len(tags):
                    print(
                        json.dumps(
                            {
                                "processed_source_shards": src_idx + 1,
                                "source_shards_total": len(tags),
                                "written_dedup_shards": written_shards,
                                "rows_seen": total_rows,
                                "rows_kept": kept_rows,
                            }
                        ),
                        flush=True,
                    )

            dedup_meta = dict(source_meta)
            dedup_meta.update(
                {
                    "source_dataset": str(input_path),
                    "n_batches_planned": written_shards,
                    "samples_before_deduplication": total_rows,
                    "target_samples": kept_rows,
                    "samples_after_deduplication": kept_rows,
                    "rows_dropped_duplicate": dropped_duplicate_rows,
                    "deduplication": {
                        "case_id_modulus": args.case_id_modulus,
                        "canonical_case_rule": "keep rows where case_id == case_id % case_id_modulus",
                    },
                }
            )
            dst.writestr("meta.json", json.dumps(dedup_meta, indent=2))

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "source_shards": len(tags),
        "dedup_shards": written_shards,
        "rows_before": total_rows,
        "rows_after": kept_rows,
        "rows_dropped_duplicate": dropped_duplicate_rows,
        "case_id_modulus": args.case_id_modulus,
        "keep_fraction": kept_rows / total_rows if total_rows else 0.0,
    }
    summary_path = output_path.with_suffix(".dedup.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
