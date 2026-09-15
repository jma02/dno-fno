"""CPU integration tests for saved dataset arrays."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np

from solver.gen_data.pipeline.batch_storage import save_completed_batch
from solver.gen_data.pipeline.types import (
    PhysicalFamilyId,
    SimulationRows,
)

TRAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(TRAIN_DIR.parent))

from scripts.build_paper_dataset import build_dataset  # noqa: E402
from util import (  # noqa: E402
    build_dataset_split_indices,
    device_prefetch,
    get_batches,
    load_dataset_arrays,
    load_or_compute_stats,
    make_normalizers,
)


def _write_batch(
    root: Path,
    *,
    family_id: PhysicalFamilyId,
    simulation_count: int,
    accepted_local_indices: tuple[int, ...],
) -> Path:
    path = root / "batches" / family_id.name.lower() / "batch_000000.npz"
    frames_per_simulation = 2
    accepted = set(accepted_local_indices)
    save_completed_batch(
        path,
        tuple(f"group_{index}" for index in range(simulation_count)),
        tuple(
            SimulationRows(
                eta=np.full((frames_per_simulation, 4), local_index + 1.0),
                xi=np.full((frames_per_simulation, 4), local_index + 1.1)
                + np.linspace(-0.5, 0.5, 4),
                gxi=np.full((frames_per_simulation, 4), local_index + 0.9),
                depth=float(local_index + 1),
                time=np.arange(frames_per_simulation, dtype=np.float64),
            )
            if local_index in accepted
            else None
            for local_index in range(simulation_count)
        ),
        family_id=family_id,
        seed=42,
    )
    return path


def test_loads_readonly_arrays_and_keeps_simulation_splits() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        root = Path(raw_directory)
        paths = tuple(
            _write_batch(
                root,
                family_id=family,
                simulation_count=6,
                accepted_local_indices=(0, 1, 2, 3, 4),
            )
            for family in PhysicalFamilyId
        )
        dataset_path = build_dataset(root / "dataset", paths)

        dataset = load_dataset_arrays(dataset_path)
        assert dataset["eta"].shape == (40, 4)
        assert all(
            isinstance(dataset[name], np.memmap) and not dataset[name].flags.writeable
            for name in ("eta", "xi", "gxi", "depth", "dataset_split", "x")
        )
        splits = build_dataset_split_indices(dataset)
        train_rows, validation_rows, test_rows = splits
        assert tuple(map(len, splits)) == (32, 4, 4)
        np.testing.assert_array_equal(np.sort(np.concatenate(splits)), np.arange(40))
        simulation_ids = np.load(dataset_path / "simulation_id.npy")
        np.testing.assert_array_equal(np.unique(simulation_ids), np.arange(20))
        for split_rows in splits:
            assert np.isin(simulation_ids, simulation_ids[split_rows]).sum() == len(
                split_rows
            )
        stored_xi = np.array(dataset["xi"])
        epoch_indices = []
        for seed in np.random.SeedSequence(17).spawn(2):
            batches = list(
                get_batches(
                    dataset["eta"],
                    dataset["xi"],
                    dataset["gxi"],
                    dataset["depth"],
                    train_rows,
                    batch_size=7,
                    rng=np.random.default_rng(seed),
                )
            )
            batch_indices = np.concatenate([batch[-1] for batch in batches])
            np.testing.assert_array_equal(np.sort(batch_indices), train_rows)
            assert len(batches[-1][-1]) == 4
            np.testing.assert_array_equal(
                np.concatenate([batch[1] for batch in batches]),
                (stored_xi - stored_xi.mean(axis=1, keepdims=True))[batch_indices],
            )
            epoch_indices.append(batch_indices)
        assert not np.array_equal(*epoch_indices)
        for name in ("eta", "gxi"):
            array = np.load(dataset_path / f"{name}.npy", mmap_mode="r+")
            array[np.concatenate((validation_rows, test_rows))] = 999.0
            array.flush()
        stats = load_or_compute_stats(dataset_path, dataset, indices=train_rows)
        centered_xi = stored_xi[train_rows] - stored_xi[train_rows].mean(
            axis=1, keepdims=True
        )
        assert np.asarray(stats["feature_absmax"])[1] == np.abs(centered_xi).max()
        assert (
            np.asarray(stats["feature_absmax"])[0]
            == np.abs(dataset["eta"][train_rows]).max()
            < 999
        )
        assert stats["target_absmax"] == np.abs(dataset["gxi"][train_rows]).max() < 999
        np.testing.assert_array_equal(dataset["xi"], stored_xi)


def test_normalizers_preserve_both_modes_and_constant_fields() -> None:
    for values in (
        np.arange(12).reshape(2, 6) - 3,
        np.zeros((2, 6)),
        np.full((2, 6), 2),
    ):
        eta, xi, gxi = (
            jnp.asarray(values * factor, dtype=jnp.float32) for factor in (1, -2, 3)
        )
        features = np.stack((eta, xi), axis=-1)
        minimum, maximum = features.min(axis=(0, 1)), features.max(axis=(0, 1))
        scales = np.abs(features).max(axis=(0, 1))
        target_min, target_max, target_scale = map(
            float, (gxi.min(), gxi.max(), jnp.abs(gxi).max())
        )
        stats = {
            "feature_min": minimum.tolist(),
            "feature_max": maximum.tolist(),
            "feature_absmax": scales.tolist(),
            "target_min": target_min,
            "target_max": target_max,
            "target_absmax": target_scale,
        }
        for mode in ("scale", "minmax"):
            norm_inputs, norm_targets, denorm_targets = make_normalizers(stats, mode)
            inputs = jax.jit(norm_inputs)(eta, xi)
            targets = jax.jit(norm_targets)(gxi)
            expected_inputs = (
                features / np.where(scales > 0, scales, 1)
                if mode == "scale"
                else (features - minimum) / (maximum - minimum + 1e-8) * 2 - 1
            )
            expected_targets = (
                np.asarray(gxi) / (target_scale or 1)
                if mode == "scale"
                else (np.asarray(gxi) - np.float32(target_min))
                / np.float32(target_max - target_min + 1e-8)
                * 2
                - 1
            )
            assert inputs.dtype == targets.dtype == jnp.float32
            np.testing.assert_allclose(inputs, expected_inputs, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(
                targets[..., 0], expected_targets, rtol=1e-6, atol=1e-6
            )
            np.testing.assert_allclose(
                jax.jit(denorm_targets)(targets)[..., 0], gxi, rtol=1e-6, atol=1e-6
            )


def test_prefetch_preserves_batches_and_producer_exceptions() -> None:
    mesh = jax.sharding.Mesh(np.asarray(jax.local_devices()), axis_names=("batch",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))
    batch = (np.arange(mesh.size * 4).reshape(mesh.size, 4), np.ones((mesh.size, 1)))
    assert list(device_prefetch((), sharding=sharding, depth=1)) == []
    prefetched = list(device_prefetch((batch, batch), sharding=sharding, depth=1))
    assert len(prefetched) == 2
    for result in prefetched:
        for actual, expected in zip(result, batch, strict=True):
            np.testing.assert_array_equal(actual, expected)

    for expected_error in (RuntimeError("producer failure"), KeyboardInterrupt()):

        def failing_batches() -> Iterator[tuple[np.ndarray, ...]]:
            yield batch
            raise expected_error

        stream = device_prefetch(failing_batches(), sharding=sharding, depth=1)
        for actual, expected in zip(next(stream), batch, strict=True):
            np.testing.assert_array_equal(actual, expected)
        try:
            next(stream)
        except BaseException as error:
            assert error is expected_error
        else:
            raise AssertionError("Producer failure did not reach the consumer")


def main() -> int:
    test_loads_readonly_arrays_and_keeps_simulation_splits()
    test_normalizers_preserve_both_modes_and_constant_fields()
    test_prefetch_preserves_batches_and_producer_exceptions()
    print(
        "[PASS] dataset splits, epoch shuffles, train-only/cache stats, both normalizers, prefetch"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
