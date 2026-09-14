"""Export the original accepted JONSWAP cases for a denser numerical replay."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
from typing import Any

import numpy as np

from solver.gen_data.jonswap_tma import (
    JonswapTmaParameters,
    PAPER_RELATIVE_FREQUENCY_MAXIMUM,
    PAPER_RELATIVE_FREQUENCY_MINIMUM,
    PAPER_RESOLVED_BAND_QUADRATURE_ORDER,
)
from solver.gen_data.pipeline.trajectory_config import PAPER_ROLLOUT_NUMERICS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("outputs/paper_dataset/arrays"))
    parser.add_argument(
        "--source-manifest", type=Path,
        default=Path("outputs/paper_dataset_literature_aligned_v1/combined/"
                     "c16384_v01024_t01024/paper_dataset_all_splits_c16384.dataset.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/paper_dataset_regenerated_20260914/replay_inputs.npz"),
    )
    args = parser.parse_args()
    started = time.monotonic()
    started_local = datetime.now().astimezone().isoformat()
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = {
        name: np.load(args.dataset / f"{name}.npy", mmap_mode="r")
        for name in ("family_id", "frame_index", "simulation_id", "dataset_split",
                     "parameter_group_id", "depth", "time", "eta", "xi", "gxi")
    }
    first_rows = np.flatnonzero((dataset["family_id"] == 4) & (dataset["frame_index"] == 0))
    first_rows = first_rows[np.argsort(dataset["simulation_id"][first_rows])]
    simulation_ids = np.asarray(dataset["simulation_id"][first_rows])
    assert first_rows.size == np.unique(simulation_ids).size == 18_432
    old_rows = first_rows[:, None] + np.arange(16)
    assert np.all(dataset["simulation_id"][old_rows] == simulation_ids[:, None])
    assert np.all(dataset["frame_index"][old_rows] == np.arange(16))
    assert np.all(dataset["dataset_split"][old_rows] == dataset["dataset_split"][first_rows, None])

    manifest = json.loads(args.source_manifest.read_text())
    map_path = args.source_manifest.parent / manifest["trajectory_map_npz"]
    with np.load(map_path) as trajectory_map:
        accepted = np.flatnonzero(trajectory_map["trajectory_accepted"])
        trajectory_indices = accepted[simulation_ids]
        assert np.all(trajectory_map["trajectory_family_id"][trajectory_indices] == 4)
        assert np.array_equal(trajectory_map["trajectory_first_row"][trajectory_indices], first_rows)
        assert np.all(trajectory_map["trajectory_row_count"][trajectory_indices] == 16)
        shard_indices = trajectory_map["shard_index"][first_rows]
        shard_rows = trajectory_map["shard_row"][first_rows]

    count = first_rows.size
    arrays: dict[str, np.ndarray] = {
        "simulation_id": simulation_ids,
        "dataset_split": np.asarray(dataset["dataset_split"][first_rows]),
        "parameter_group_id": np.asarray(dataset["parameter_group_id"][first_rows]),
        "old_first_row": first_rows,
        "old_times": np.asarray(dataset["time"][old_rows]),
        **{name: np.empty(count, dtype=np.float64) for name in JonswapTmaParameters._fields},
        **{name: np.empty((count, 128), dtype=np.float64) for name in ("phase_right", "phase_left")},
        "old_dense_indices": np.empty((count, 16), dtype=np.int32),
        **{name: np.empty(count, dtype=np.int32) for name in (
            "production_time_count", "adjustment_time_count", "source_chunk_index",
            "source_batch_id", "source_proposal_row",
        )},
        "source_case_id": np.empty(count, dtype=np.int64),
        "nonlinear_ramp_time": np.empty(count, dtype=np.float64),
    }
    source_chunks: list[str] = []
    original_execution: dict[str, Any] | None = None
    for shard_index in np.unique(shard_indices):
        positions = np.flatnonzero(shard_indices == shard_index)
        shard_path = (args.source_manifest.parent / manifest["dataset_shards"][int(shard_index)]["path"]).resolve()
        chunk_path = shard_path.parents[3]
        if str(chunk_path) not in source_chunks:
            source_chunks.append(str(chunk_path))
        proposal_path = chunk_path / "proposals" / shard_path.relative_to(chunk_path / "shards")
        result_path = chunk_path / "results" / shard_path.relative_to(chunk_path / "shards").with_suffix(".json")
        results = json.loads(result_path.read_text())
        with np.load(shard_path) as shard, np.load(proposal_path) as proposal:
            local_rows = shard_rows[positions, None] + np.arange(16)
            local_cases = shard["case_local_index"][shard_rows[positions]]
            assert np.all(shard["case_local_index"][local_rows] == local_cases[:, None])
            assert np.array_equal(shard["time"][local_rows], arrays["old_times"][positions])
            assert np.array_equal(shard["depth"][shard_rows[positions]], dataset["depth"][first_rows[positions]])
            arrays["old_dense_indices"][positions] = shard["selected_dense_index"][local_rows]
            metadata = json.loads(str(proposal["metadata_json"]))
            if original_execution is None:
                original_execution = metadata["trajectory_execution"]
            assert original_execution == metadata["trajectory_execution"]
            for position, local_case in zip(positions, local_cases, strict=True):
                spec = json.loads(str(proposal["case_spec_json"][local_case]))
                result = results["cases"][int(local_case)]
                assert result["accepted"] and result["row_count"] == 16
                assert result["case_id"] == spec["case_id"] == int(proposal["case_id"][local_case])
                assert spec["cell_id"] == arrays["parameter_group_id"][position]
                assert spec["depth"] == float(dataset["depth"][first_rows[position]])
                assert spec["constructor_settings"] == {
                    "density_window": "sharp_relative_frequency_interval_v1",
                    "quadrature_order": PAPER_RESOLVED_BAND_QUADRATURE_ORDER,
                    "relative_frequency_minimum": PAPER_RELATIVE_FREQUENCY_MINIMUM,
                    "relative_frequency_maximum": PAPER_RELATIVE_FREQUENCY_MAXIMUM,
                }
                for name in (*JonswapTmaParameters._fields, "phase_right", "phase_left"):
                    arrays[name][position] = spec[name]
                metrics = result["metrics"]
                arrays["production_time_count"][position] = metrics["saved_time_count"]
                arrays["adjustment_time_count"][position] = metrics["nonlinear_adjustment_saved_time_count"]
                arrays["nonlinear_ramp_time"][position] = metrics["nonlinear_adjustment_ramp_time"]
                arrays["source_chunk_index"][position] = source_chunks.index(str(chunk_path))
                arrays["source_batch_id"][position] = int(proposal["batch_id"])
                arrays["source_proposal_row"][position] = local_case
                arrays["source_case_id"][position] = spec["case_id"]

    assert original_execution is not None
    original_numerical = original_execution["numerical"]
    config = PAPER_ROLLOUT_NUMERICS["jonswap_tma"]
    current_numerical = {
        name: getattr(config, name) for name in (
            "nx", "target_nx", "length", "gravity", "pad_factor", "maximum_wavenumber",
            "target_maximum_wavenumber", "saved_dt", "gl2_residual_tolerance",
            "gl2_iteration_cap", "internal_hamiltonian_drift_threshold",
        )
    }
    current_numerical.update(
        dno_order=config.integration_dno_order, target_dno_order=config.label_dno_order,
        dt=config.saved_dt / config.substeps_per_saved_frame, dtype="float64",
        post_step_state_filter="sharp", post_step_maximum_wavenumber=None,
        target_time_chunk_size=8,
    )
    differences = {
        name: {"original": original_numerical[name], "current": value}
        for name, value in current_numerical.items() if original_numerical[name] != value
    }
    assert not differences, differences
    assert np.all(arrays["production_time_count"] >= 200)
    assert np.all(np.diff(arrays["old_dense_indices"], axis=1) > 0)
    assert np.array_equal(arrays["old_dense_indices"][:, 0], np.zeros(count, dtype=np.int32))
    assert np.array_equal(arrays["old_dense_indices"][:, -1], arrays["production_time_count"] - 1)
    assert np.array_equal(arrays["old_times"], config.saved_dt * arrays["old_dense_indices"])
    identities = np.stack(tuple(arrays[name] for name in ("source_chunk_index", "source_case_id")), axis=1)
    assert np.unique(identities, axis=0).shape[0] == count
    _, representatives = np.unique(arrays["parameter_group_id"], return_index=True)
    comparison_indices = np.unique(np.r_[
        representatives, np.argmin(arrays["production_time_count"]),
        np.argmax(arrays["production_time_count"]),
    ]).astype(np.int32)
    arrays["comparison_indices"] = comparison_indices
    for name in ("eta", "xi", "gxi"):
        arrays[f"comparison_{name}"] = np.asarray(dataset[name][old_rows[comparison_indices]])
    assert all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind in "fi")
    split_names, split_counts = np.unique(arrays["dataset_split"], return_counts=True)
    metadata = {
        "started_local": started_local, "completed_local": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": time.monotonic() - started, "simulations": count,
        "dataset": str(args.dataset.resolve()), "source_manifest": str(args.source_manifest.resolve()),
        "source_chunks": source_chunks, "original_execution": original_execution,
        "original_numerical": original_numerical, "current_numerical": config._asdict(),
        "numerical_differences": differences,
        "splits": dict(zip(split_names.tolist(), split_counts.tolist(), strict=True)),
        "comparison_simulation_ids": simulation_ids[comparison_indices].tolist(),
        "comparison_selection": "One case per parameter group plus minimum/maximum production frame count.",
        "mapping_verified": "Current dense simulation ID -> accepted original trajectory -> shard -> accepted proposal row.",
        "replay_policy": "Preserve current splits and all 16 old dense indices; add 184 distinct frames. Do not resample parameters or phases.",
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.output.parent, prefix=".replay_inputs-") as temporary:
        np.savez_compressed(temporary, allow_pickle=False, **arrays)
        temporary.flush()
        os.link(temporary.name, args.output)
    print(json.dumps(metadata, indent=2))
