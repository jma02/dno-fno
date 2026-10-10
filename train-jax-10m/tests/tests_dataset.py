"""CPU integration tests for saved dataset arrays."""

from __future__ import annotations

import ast
from itertools import product
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
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


def test_surface_feature_rms_uses_normalized_training_rows() -> None:
    path = TRAIN_DIR / "1d_dno_fno_jax_ema_grad.py"
    fitting = next(
        node for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.If) and isinstance(node.test, ast.Attribute)
        and node.test.attr == "cs_normalize_surface_features"
    )
    code = compile(ast.Module(body=fitting.body, type_ignores=[]), str(path), "exec")
    nx, wave_number, eta_scale = 64, 3, 2.0
    train_indices = np.arange(0, 1200, 2)
    x = np.arange(nx) * (2 * np.pi / nx)
    for amplitude, whiten in product((0.0, 0.25), (False, True)):
        args = SimpleNamespace(
            cs_n_polys=3, cs_use_first_deriv=True, cs_use_second_deriv=True,
            cs_use_half_deriv=True, cs_use_hilbert=True,
            cs_use_relative_elevation=True,
            cs_whiten_surface_features=whiten, cs_surface_feature_transform=None,
        )
        # Interleaved held-out rows must not affect fitting, including the zero floor.
        eta = np.full((1200, nx), 1e9, dtype=np.float32)
        eta[train_indices] = eta_scale * amplitude * np.cos(wave_number * x)
        namespace = {
            "np": np, "args": args, "dataset": {"eta": eta, "depth": np.full(1200, 0.5)},
            "train_indices": train_indices, "eta_scale": eta_scale,
            "domain_length": 2 * np.pi, "nx": nx,
        }
        exec(code, namespace)
        rms = amplitude / np.sqrt(2)
        expected = np.maximum([
            rms, amplitude**2 * np.sqrt(3 / 8), amplitude**3 * np.sqrt(5 / 16),
            wave_number * rms, wave_number**2 * rms, np.sqrt(wave_number) * rms,
            rms, eta_scale / 0.5 * rms,
        ], 1e-12)
        np.testing.assert_allclose(args.cs_surface_feature_scales, expected, rtol=1e-6)
        np.testing.assert_array_equal(np.sort(namespace["feature_rows"]), train_indices)
        if whiten:
            transform = np.asarray(args.cs_surface_feature_transform)
            assert np.isfinite(transform).all()
            np.testing.assert_allclose(transform, transform.T, atol=1e-12)
            assert np.linalg.eigvalsh(transform).min() > 0
            field = amplitude * np.cos(wave_number * x)
            sine = amplitude * np.sin(wave_number * x)
            features = np.stack((field, field**2, field**3, -wave_number * sine,
                                 -wave_number**2 * field, np.sqrt(wave_number) * field,
                                 sine, eta_scale / 0.5 * field), axis=-1) / expected
            moment = features.T @ features / nx
            # This is the uncentered second moment, including eta^2's nonzero mean.
            np.testing.assert_allclose(namespace["second_moment"], moment, rtol=1e-6, atol=1e-7)
            spectrum = np.linalg.eigvalsh(transform.T @ moment @ transform)
            assert spectrum.min() > -1e-8 and spectrum.max() < 1 + 1e-6
            if amplitude:
                np.testing.assert_allclose(spectrum[-3:], 1, rtol=1e-6)
        else:
            assert args.cs_surface_feature_transform is None


def main() -> int:
    test_loads_readonly_arrays_and_keeps_simulation_splits()
    test_normalizers_preserve_both_modes_and_constant_fields()
    test_prefetch_preserves_batches_and_producer_exceptions()
    test_surface_feature_rms_uses_normalized_training_rows()
    print(
        "[PASS] dataset splits, epoch shuffles, train-only/cache stats, both normalizers, prefetch"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
