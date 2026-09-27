"""Verify local staging, cache reuse/recovery, and exact shuffled batches."""

import hashlib
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "torch-attention"))
from train import Waves, build_model, cache_data, make_loader  # noqa: E402
from spectral import surface_features  # noqa: E402


def main() -> None:
    torch.set_num_threads(4)
    source = ROOT.parent / "local-data/paper_equal_subset_20260924"
    with TemporaryDirectory(prefix="dno-cache-check-") as temporary:
        destination = Path(temporary) / "cache"
        started = perf_counter()
        cache_data(source, destination)
        copy_seconds = perf_counter() - started
        manifest = json.loads((destination / "cache_manifest.json").read_text())
        for name in manifest["files"]:
            assert hashlib.sha256((source / name).read_bytes()).digest() == hashlib.sha256((destination / name).read_bytes()).digest()
        modified = {name: (destination / name).stat().st_mtime_ns for name in manifest["files"]}
        started = perf_counter()
        cache_data(source, destination)
        reuse_seconds = perf_counter() - started
        assert modified == {name: (destination / name).stat().st_mtime_ns for name in manifest["files"]}
        datasets = [Waves(root, "train") for root in (source, destination)]
        for dataset in datasets:
            dataset.rows = dataset.rows[:513]
        loaders = [make_loader(dataset, 256, bulk=True, shuffle=True,
                               generator=torch.Generator().manual_seed(0)) for dataset in datasets]
        for _ in range(2):
            sizes = []
            for original, copied in zip(*loaders, strict=True):
                for a, b in zip(original, copied, strict=True):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                sizes.append(len(original[0]))
            assert sizes == [256, 256, 1]
        # A truncated cache must not pass the manifest's completeness check.
        with (destination / "depth.npy").open("r+b") as handle:
            handle.truncate(8)
        cache_data(source, destination)
        assert (source / "depth.npy").read_bytes() == (destination / "depth.npy").read_bytes()
        for name, info in manifest["files"].items():
            assert (source / name).stat().st_mtime_ns == info["mtime_ns"]
        sums = torch.zeros(7, dtype=torch.float64)
        for eta, *_ in DataLoader(datasets[0], batch_size=256):
            sums += surface_features(eta, 2 * math.pi).double().square().sum((0, 1))
        expected = (sums / (len(datasets[0]) * 1024)).sqrt().float().clamp_min(1e-12)
        model = build_model("spectral", datasets[0], 1024, 2 * math.pi, 2, False)
        torch.testing.assert_close(model.feature_scales, expected, rtol=0, atol=0)
        report = {"copy_seconds": copy_seconds, "reuse_seconds": reuse_seconds,
                  "bytes_staged": sum(info["bytes"] for info in manifest["files"].values()),
                  "file_hashes_exact": True, "shuffle_and_tails_exact": True,
                  "cache_reused_without_rewrite": True, "truncated_file_recovered": True,
                  "source_unmodified": True, "bulk_feature_scales_exact": True}
        Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
