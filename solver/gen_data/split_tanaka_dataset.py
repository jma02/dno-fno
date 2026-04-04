from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from .clean_tanaka_dataset import list_source_tags, read_npy, write_npy


SPLIT_NAMES = ("train", "val", "test")
SPLIT_LABELS = {"train": 0, "val": 1, "test": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize fixed train/val/test .npz archives from a Tanaka dataset using a saved split manifest."
    )
    parser.add_argument("--input", default="data/tanaka_1_clean.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_split_indices(num_examples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    permutation = np.random.default_rng(seed).permutation(num_examples)
    val_count = int(num_examples * 0.1)
    test_count = int(num_examples * 0.1)
    train_count = num_examples - val_count - test_count
    train_indices = permutation[:train_count]
    val_indices = permutation[train_count : train_count + val_count]
    test_indices = permutation[train_count + val_count :]
    return train_indices, val_indices, test_indices


def split_path_for_dataset(dataset_path: Path, seed: int) -> Path:
    return dataset_path.parent / f"{dataset_path.stem}.split_seed{seed}.npz"


def load_or_create_split_manifest(
    dataset_path: Path,
    num_examples: int,
    seed: int,
) -> tuple[Path, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    split_path = split_path_for_dataset(dataset_path, seed)
    if split_path.exists():
        with np.load(split_path) as split_file:
            train_indices = np.asarray(split_file["train_indices"], dtype=np.int64)
            val_indices = np.asarray(split_file["val_indices"], dtype=np.int64)
            test_indices = np.asarray(split_file["test_indices"], dtype=np.int64)
            saved_num_examples = int(split_file["num_examples"])
            saved_seed = int(split_file["seed"])
        if (
            saved_num_examples == num_examples
            and saved_seed == seed
            and train_indices.shape[0] + val_indices.shape[0] + test_indices.shape[0] == num_examples
        ):
            return split_path, (train_indices, val_indices, test_indices)

    train_indices, val_indices, test_indices = build_split_indices(num_examples, seed)
    np.savez_compressed(
        split_path,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        num_examples=np.asarray(num_examples, dtype=np.int64),
        seed=np.asarray(seed, dtype=np.int64),
    )
    return split_path, (train_indices, val_indices, test_indices)


def make_output_paths(dataset_path: Path) -> dict[str, Path]:
    return {
        split_name: dataset_path.parent / f"{dataset_path.stem}_{split_name}.npz"
        for split_name in SPLIT_NAMES
    }


def split_labels_from_indices(
    num_examples: int,
    split_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    labels = np.full(num_examples, -1, dtype=np.int8)
    labels[split_indices[0]] = SPLIT_LABELS["train"]
    labels[split_indices[1]] = SPLIT_LABELS["val"]
    labels[split_indices[2]] = SPLIT_LABELS["test"]
    if np.any(labels < 0):
        raise ValueError("Split manifest did not assign every example to exactly one split.")
    return labels


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find input dataset at {input_path}")

    output_paths = make_output_paths(input_path)
    if args.overwrite:
        for path in output_paths.values():
            if path.exists():
                path.unlink()
    else:
        existing = [path for path in output_paths.values() if path.exists()]
        if existing:
            names = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"Split outputs already exist: {names}. Pass --overwrite to replace them.")

    with zipfile.ZipFile(input_path, mode="r") as src:
        source_meta = json.loads(src.read("meta.json"))
        num_examples = int(source_meta["samples_after_cleaning"])
        split_path, split_indices = load_or_create_split_manifest(input_path, num_examples, args.seed)
        labels = split_labels_from_indices(num_examples, split_indices)
        tags = list_source_tags(src)

        x = read_npy(src, "x.npy").astype(np.float32)
        subsample_indices = read_npy(src, "subsample_indices.npy")
        subsample_times = read_npy(src, "subsample_times.npy")

        split_counts = {
            "train": int(split_indices[0].shape[0]),
            "val": int(split_indices[1].shape[0]),
            "test": int(split_indices[2].shape[0]),
        }
        written_counts = {name: 0 for name in SPLIT_NAMES}
        written_shards = {name: 0 for name in SPLIT_NAMES}

        with (
            zipfile.ZipFile(output_paths["train"], mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as train_zf,
            zipfile.ZipFile(output_paths["val"], mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as val_zf,
            zipfile.ZipFile(output_paths["test"], mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as test_zf,
        ):
            archives = {"train": train_zf, "val": val_zf, "test": test_zf}

            for archive in archives.values():
                write_npy(archive, "x.npy", x)
                write_npy(archive, "subsample_indices.npy", subsample_indices)
                write_npy(archive, "subsample_times.npy", subsample_times)

            global_start = 0
            for src_idx, tag in enumerate(tags):
                eta = read_npy(src, f"eta_batch_{tag}.npy").astype(np.float32, copy=False)
                xi = read_npy(src, f"xi_batch_{tag}.npy").astype(np.float32, copy=False)
                gxi = read_npy(src, f"gxi_batch_{tag}.npy").astype(np.float32, copy=False)
                time = read_npy(src, f"time_batch_{tag}.npy")
                case_id = read_npy(src, f"case_id_batch_{tag}.npy")

                shard_size = int(eta.shape[0])
                shard_labels = labels[global_start : global_start + shard_size]
                global_start += shard_size

                for split_name in SPLIT_NAMES:
                    split_mask = shard_labels == SPLIT_LABELS[split_name]
                    if not np.any(split_mask):
                        continue
                    dest_tag = f"{written_shards[split_name]:04d}"
                    archive = archives[split_name]
                    write_npy(archive, f"eta_batch_{dest_tag}.npy", eta[split_mask])
                    write_npy(archive, f"xi_batch_{dest_tag}.npy", xi[split_mask])
                    write_npy(archive, f"gxi_batch_{dest_tag}.npy", gxi[split_mask])
                    write_npy(archive, f"time_batch_{dest_tag}.npy", time[split_mask])
                    write_npy(archive, f"case_id_batch_{dest_tag}.npy", case_id[split_mask])
                    written_shards[split_name] += 1
                    written_counts[split_name] += int(split_mask.sum())

                if (src_idx + 1) % 10 == 0 or src_idx + 1 == len(tags):
                    print(
                        json.dumps(
                            {
                                "processed_source_shards": src_idx + 1,
                                "source_shards_total": len(tags),
                                "rows_seen": global_start,
                                "written_counts": written_counts,
                                "written_shards": written_shards,
                            }
                        ),
                        flush=True,
                    )

            for split_name, archive in archives.items():
                split_meta = dict(source_meta)
                split_meta.update(
                    {
                        "source_dataset": str(input_path),
                        "source_split_file": str(split_path),
                        "split_name": split_name,
                        "split_seed": args.seed,
                        "target_samples": written_counts[split_name],
                        "samples_after_split": written_counts[split_name],
                        "n_batches_planned": written_shards[split_name],
                        "split_counts": split_counts,
                    }
                )
                archive.writestr("meta.json", json.dumps(split_meta, indent=2))

    summary = {
        "input": str(input_path),
        "split_file": str(split_path),
        "seed": args.seed,
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "counts": written_counts,
        "shards": written_shards,
    }
    summary_path = input_path.parent / f"{input_path.stem}.split_seed{args.seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
