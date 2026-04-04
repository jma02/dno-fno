from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream an NPZ dataset shard-by-shard and report structural integrity checks."
    )
    parser.add_argument("--dataset", default="data/tanaka_1_clean.npz")
    parser.add_argument("--modulo_base", type=int, default=1_000_000)
    parser.add_argument("--time_tol", type=float, default=1e-6)
    return parser.parse_args()


def update_case_counts(
    case_row_counts: dict[int, int],
    case_ids: np.ndarray,
) -> None:
    unique_case_ids, counts = np.unique(case_ids, return_counts=True)
    for case_id, count in zip(unique_case_ids.tolist(), counts.tolist()):
        case_row_counts[case_id] = case_row_counts.get(case_id, 0) + count


def build_local_case_bits(
    case_ids: np.ndarray,
    time_indices: np.ndarray,
) -> tuple[dict[int, int], int]:
    pairs = np.rec.fromarrays([case_ids, time_indices], names="case_id,time_index")
    unique_pairs, pair_counts = np.unique(pairs, return_counts=True)

    local_bits: dict[int, int] = {}
    duplicate_rows = int(np.sum(pair_counts - 1))
    pair_case_ids = unique_pairs["case_id"]
    pair_time_indices = unique_pairs["time_index"]
    unique_case_ids, starts = np.unique(pair_case_ids, return_index=True)
    ends = np.concatenate((starts[1:], np.array([pair_case_ids.shape[0]])))

    for case_id, start, end in zip(unique_case_ids.tolist(), starts.tolist(), ends.tolist()):
        bitset = 0
        for time_index in pair_time_indices[start:end].tolist():
            bitset |= 1 << int(time_index)
        local_bits[int(case_id)] = bitset

    return local_bits, duplicate_rows


def summarize_counts(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.int64)
    return {
        "min": int(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": int(array.max()),
    }


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()

    with zipfile.ZipFile(dataset_path, mode="r") as zf:
        meta = json.loads(zf.read("meta.json"))

    with np.load(dataset_path) as archive:
        subsample_times = np.asarray(archive["subsample_times"], dtype=np.float64)
        n_shards = int(meta["n_batches_planned"])

        field_mins = {"eta": np.inf, "xi": np.inf, "gxi": np.inf}
        field_maxs = {"eta": -np.inf, "xi": -np.inf, "gxi": -np.inf}
        nonfinite_rows = {"eta": 0, "xi": 0, "gxi": 0}

        total_rows = 0
        invalid_time_rows = 0
        duplicate_case_time_rows = 0
        cross_shard_duplicate_case_time_pairs = 0
        case_row_counts: dict[int, int] = {}
        case_time_bits: dict[int, int] = {}
        base_to_case_ids: dict[int, set[int]] = {}

        case_id_min = np.inf
        case_id_max = -np.inf

        for shard_id in range(n_shards):
            tag = f"{shard_id:04d}"

            for field_name in ("eta", "xi", "gxi"):
                field = np.asarray(archive[f"{field_name}_batch_{tag}"], dtype=np.float32)
                field_mins[field_name] = min(field_mins[field_name], float(np.min(field)))
                field_maxs[field_name] = max(field_maxs[field_name], float(np.max(field)))
                nonfinite_rows[field_name] += int(np.any(~np.isfinite(field), axis=1).sum())

            case_ids = np.asarray(archive[f"case_id_batch_{tag}"], dtype=np.int64)
            times = np.asarray(archive[f"time_batch_{tag}"], dtype=np.float64)

            total_rows += int(case_ids.shape[0])
            case_id_min = min(case_id_min, int(case_ids.min()))
            case_id_max = max(case_id_max, int(case_ids.max()))

            update_case_counts(case_row_counts, case_ids)

            time_indices = np.searchsorted(subsample_times, times)
            valid = time_indices < subsample_times.shape[0]
            valid[valid] &= np.abs(subsample_times[time_indices[valid]] - times[valid]) <= args.time_tol
            invalid_time_rows += int((~valid).sum())

            local_bits, local_duplicate_rows = build_local_case_bits(
                case_ids[valid],
                time_indices[valid],
            )
            duplicate_case_time_rows += local_duplicate_rows

            for case_id, bitset in local_bits.items():
                previous = case_time_bits.get(case_id, 0)
                overlap = previous & bitset
                if overlap:
                    cross_shard_duplicate_case_time_pairs += int(overlap.bit_count())
                case_time_bits[case_id] = previous | bitset

                base_case_id = int(case_id % args.modulo_base)
                case_set = base_to_case_ids.setdefault(base_case_id, set())
                case_set.add(int(case_id))

        rows_per_case = list(case_row_counts.values())
        unique_times_per_case = [bitset.bit_count() for bitset in case_time_bits.values()]
        cases_with_missing_times = int(
            sum(unique_time_count < subsample_times.shape[0] for unique_time_count in unique_times_per_case)
        )
        cases_with_duplicate_rows = int(
            sum(
                row_count > case_time_bits.get(case_id, 0).bit_count()
                for case_id, row_count in case_row_counts.items()
            )
        )
        modulo_duplicates = {
            base_case_id: sorted(case_ids)
            for base_case_id, case_ids in base_to_case_ids.items()
            if len(case_ids) > 1
        }
        unique_case_ids = sorted(case_row_counts)
        modulo_buckets: dict[int, int] = {}
        for case_id in unique_case_ids:
            bucket = int(case_id // args.modulo_base)
            modulo_buckets[bucket] = modulo_buckets.get(bucket, 0) + 1

        summary = {
            "dataset": str(dataset_path),
            "n_shards": n_shards,
            "total_rows": total_rows,
            "meta_target_samples": int(meta.get("target_samples", -1)),
            "case_id_min": int(case_id_min),
            "case_id_max": int(case_id_max),
            "num_unique_case_ids": len(case_row_counts),
            "case_id_modulo_base": args.modulo_base,
            "unique_case_ids_per_modulo_bucket": modulo_buckets,
            "num_base_case_ids_with_multiple_case_ids": int(len(modulo_duplicates)),
            "sample_base_case_id_collisions": {
                str(base_case_id): case_ids
                for base_case_id, case_ids in list(sorted(modulo_duplicates.items()))[:10]
            },
            "rows_per_case": summarize_counts(rows_per_case),
            "unique_times_per_case": summarize_counts(unique_times_per_case),
            "cases_with_missing_times": cases_with_missing_times,
            "cases_with_duplicate_case_time_rows": cases_with_duplicate_rows,
            "duplicate_case_time_rows_within_shards": duplicate_case_time_rows,
            "cross_shard_duplicate_case_time_pairs": cross_shard_duplicate_case_time_pairs,
            "invalid_time_rows": invalid_time_rows,
            "field_min": field_mins,
            "field_max": field_maxs,
            "nonfinite_rows": nonfinite_rows,
        }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
