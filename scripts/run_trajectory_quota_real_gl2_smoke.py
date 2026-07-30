"""Run a resumable real-GL2 quota smoke for all trajectory families.

This is deliberately reduced software-wiring evidence, not validation of the
paper numerical contract.  It nevertheless uses the production
``run_residual_controlled_arm`` through ``TrajectoryQuotaExecutor``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType
from typing import TypeAlias

# These assignments precede every import that can initialize JAX.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["DNO_TANAKA_DTYPE"] = "float64"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.benjamin_feir_population import (  # noqa: E402
    BENJAMIN_FEIR_POPULATION_CELLS,
)
from solver.gen_data.jonswap_tma_population import (  # noqa: E402
    JONSWAP_TMA_POPULATION_CELLS,
)
from solver.gen_data.pipeline.archive import (  # noqa: E402
    file_sha256,
    write_json_atomic,
)
from solver.gen_data.pipeline.manifest import (  # noqa: E402
    DATASET_VIEW_SCHEMA_VERSION,
    TRAJECTORY_MAP_SCHEMA_VERSION,
    DatasetViewPaths,
    build_dataset_view,
)
from solver.gen_data.pipeline.production import (  # noqa: E402
    PAPER_CORPUS_REVISION_ID,
    CellQuota,
    PhysicalFamilyId,
    SplitId,
    split_code,
)
from solver.gen_data.pipeline.quota_driver import (  # noqa: E402
    AcceptedQuotaRunSpec,
    QuotaRunState,
    run_accepted_quotas,
    scan_quota_run,
)
from solver.gen_data.pipeline.refinement import (  # noqa: E402
    ResidualControlledGL2Contract,
)
from solver.gen_data.pipeline.trajectory_writer import (  # noqa: E402
    StoredTimePolicy,
    TrajectoryFamily,
)
from solver.gen_data.tanaka_population import (  # noqa: E402
    TANAKA_POPULATION_CELLS,
)
from solver.gen_data.trajectory_quota_executor import (  # noqa: E402
    TrajectoryExecutionConfig,
    TrajectoryHorizonPolicy,
    TrajectoryQuotaExecutor,
)

jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/trajectory_quota_real_gl2_smoke_20260725"
SUMMARY_NAME = "summary.json"
FAMILIES: tuple[TrajectoryFamily, ...] = (
    "tanaka",
    "benjamin_feir",
    "jonswap_tma",
)
FamilyResultMap: TypeAlias = dict[str, object]

SMOKE_CONTRACT = ResidualControlledGL2Contract(
    nx=64,
    length=2.0 * math.pi,
    gravity=1.0,
    dno_order=0,
    pad_factor=1,
    maximum_wavenumber=16.0,
    production_dt=0.04,
    saved_dt=0.08,
    gl2_residual_tolerance=1.0e-8,
    gl2_iteration_cap=8,
    refinement_tolerance=1.0e-3,
    relative_floor=1.0e-12,
    target_time_chunk_size=2,
)
SMOKE_STORED_TIME_POLICY = StoredTimePolicy(
    tanaka_count=3,
    tanaka_alpha=0.5,
    tanaka_sigma_steps=1.0,
    benjamin_feir_count=3,
    benjamin_feir_alpha=0.5,
    benjamin_feir_sigma_steps=1.0,
    random_sea_count=3,
)
SELECTED_CELLS: dict[TrajectoryFamily, str] = {
    "tanaka": TANAKA_POPULATION_CELLS[0].cell_id,
    "benjamin_feir": BENJAMIN_FEIR_POPULATION_CELLS[0].cell_id,
    # The first finite-depth cell: gamma=1 and right-moving fraction zero.
    "jonswap_tma": JONSWAP_TMA_POPULATION_CELLS[9].cell_id,
}
FAMILY_IDS: dict[TrajectoryFamily, PhysicalFamilyId] = {
    "tanaka": PhysicalFamilyId.TANAKA,
    "benjamin_feir": PhysicalFamilyId.BENJAMIN_FEIR,
    "jonswap_tma": PhysicalFamilyId.JONSWAP_TMA,
}
SOURCE_PATHS = (
    Path(__file__).resolve(),
    ROOT / "solver/gen_data/trajectory_quota_executor.py",
    ROOT / "solver/gen_data/trajectory_family_adapters.py",
    ROOT / "solver/gen_data/tanaka_population.py",
    ROOT / "solver/gen_data/benjamin_feir_population.py",
    ROOT / "solver/gen_data/benjamin_feir_jcp09.py",
    ROOT / "solver/gen_data/jonswap_tma_population.py",
    ROOT / "solver/gen_data/jonswap_tma.py",
    ROOT / "solver/gen_data/generate_tanaka_dataset_v2.py",
    ROOT / "solver/gen_data/pipeline/acceptance.py",
    ROOT / "solver/gen_data/pipeline/archive.py",
    ROOT / "solver/gen_data/pipeline/manifest.py",
    ROOT / "solver/gen_data/pipeline/production.py",
    ROOT / "solver/gen_data/pipeline/quota_driver.py",
    ROOT / "solver/gen_data/pipeline/reference.py",
    ROOT / "solver/gen_data/pipeline/refinement.py",
    ROOT / "solver/gen_data/pipeline/time_selection.py",
    ROOT / "solver/gen_data/pipeline/trajectory_writer.py",
    ROOT / "solver/gen_data/pipeline/writer.py",
    ROOT / "solver/solvers/dno_series_jax.py",
    ROOT / "solver/solvers/time_integrator.py",
    ROOT / "train-jax-10m/util.py",
)
DIAGNOSTIC_KEYS = (
    "complete_admissible_trajectory",
    "state_finite",
    "target_finite",
    "positive_water_column",
    "all_stages_solved",
    "maximum_stage_residual",
    "production_dt",
)


@dataclass(frozen=True)
class FamilySmokeResult:
    """Durable products and audit record for one family."""

    family: TrajectoryFamily
    execution: TrajectoryExecutionConfig
    spec: AcceptedQuotaRunSpec
    state: QuotaRunState
    view: DatasetViewPaths
    summary: FamilyResultMap


@dataclass(frozen=True)
class SmokeRunResult:
    """Products returned by one complete three-family invocation."""

    summary_path: Path
    families: tuple[FamilySmokeResult, ...]
    summary: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _strict_json_object(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{path} contains nonfinite JSON constant {value!r}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _strict_json_text(text: str, *, label: str) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains nonfinite JSON constant {value!r}")

    value = json.loads(text, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def source_hashes() -> dict[str, str]:
    """Return hashes of every implementation file used by this smoke."""

    return {
        _relative(path, ROOT): file_sha256(path)
        for path in SOURCE_PATHS
    }


def build_execution(
    family: TrajectoryFamily,
    *,
    contract: ResidualControlledGL2Contract = SMOKE_CONTRACT,
) -> TrajectoryExecutionConfig:
    """Return the explicitly reduced real-GL2 execution contract."""

    return TrajectoryExecutionConfig(
        family=family,
        role="reduced_wiring_evidence_only",
        numerical=contract,
        horizon=(
            TrajectoryHorizonPolicy.jonswap_peak_periods(16)
            if family == "jonswap_tma"
            else TrajectoryHorizonPolicy.fixed(0.16)
        ),
        stored_time_policy=SMOKE_STORED_TIME_POLICY,
        # Quadrature four is part of this reduced wiring contract.  The
        # horizon remains the declared sixteen finite-depth peak periods.
        jonswap_quadrature_order=4 if family == "jonswap_tma" else None,
    )


def build_run_spec(
    output_dir: Path,
    execution: TrajectoryExecutionConfig,
) -> AcceptedQuotaRunSpec:
    """Return the immutable one-cell, one-accepted-case family quota."""

    family = execution.family
    cell_id = SELECTED_CELLS[family]
    return AcceptedQuotaRunSpec(
        root=output_dir,
        family_name=family,
        family_id=FAMILY_IDS[family],
        revision_id=PAPER_CORPUS_REVISION_ID,
        split_id=SplitId.TEST,
        stream_id=0,
        quotas=(CellQuota(cell_id, 1),),
        cell_codes={cell_id: 0},
        batch_size=1,
        first_attempt_index=0,
        configuration={
            "schema": "trajectory_quota_real_gl2_smoke_configuration_v1",
            "purpose": (
                "reduced real-GL2 trajectory quota software-wiring evidence"
            ),
            "selected_cell": cell_id,
            "accepted_cases_per_cell": 1,
            "trajectory_execution": execution.to_json_record(),
            "source_sha256": source_hashes(),
        },
    )


def _load_training_util() -> ModuleType:
    path = ROOT / "train-jax-10m/util.py"
    module_name = "_trajectory_quota_real_gl2_smoke_training_util"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"could not load training utility from {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[module_name]
        raise
    return module


def _artifact_record(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": _relative(path, root),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _stored_row_count(execution: TrajectoryExecutionConfig) -> int:
    return {
        "tanaka": execution.stored_time_policy.tanaka_count,
        "benjamin_feir": execution.stored_time_policy.benjamin_feir_count,
        "jonswap_tma": execution.stored_time_policy.random_sea_count,
    }[execution.family]


def _validate_loader_and_map(
    view: DatasetViewPaths,
    *,
    execution: TrajectoryExecutionConfig,
    expected_fingerprint: str,
    attempted_cases: int,
    accepted_cases: int,
) -> dict[str, object]:
    manifest = _strict_json_object(view.manifest)
    if manifest.get("schema_version") != DATASET_VIEW_SCHEMA_VERSION:
        raise RuntimeError("dataset view does not use schema version two")
    if manifest.get("configuration_fingerprint") != expected_fingerprint:
        raise RuntimeError("dataset view fingerprint differs from its run")
    expected_rows = accepted_cases * _stored_row_count(execution)
    for name, expected in (
        ("n_rows", expected_rows),
        ("n_trajectories", attempted_cases),
        ("n_accepted_trajectories", accepted_cases),
        ("n_accepted_rows", expected_rows),
    ):
        if manifest.get(name) != expected:
            raise RuntimeError(
                f"dataset view {name} is {manifest.get(name)!r}, expected {expected}"
            )
    if manifest.get("trajectory_map_sha256") != file_sha256(view.trajectory_map):
        raise RuntimeError("trajectory-map hash differs from its manifest")

    with np.load(view.trajectory_map, allow_pickle=False) as archive:
        trajectory_map = {
            name: np.asarray(archive[name]) for name in archive.files
        }
    if int(trajectory_map["schema_version"]) != TRAJECTORY_MAP_SCHEMA_VERSION:
        raise RuntimeError("trajectory map does not use schema version two")
    accepted = np.asarray(
        trajectory_map["trajectory_accepted"], dtype=np.bool_
    )
    row_counts = np.asarray(
        trajectory_map["trajectory_row_count"], dtype=np.int32
    )
    if accepted.shape != (attempted_cases,) or row_counts.shape != accepted.shape:
        raise RuntimeError("trajectory-map attempted-case cardinality is wrong")
    if int(np.count_nonzero(accepted)) != accepted_cases:
        raise RuntimeError("trajectory-map accepted count is wrong")
    if not np.all(row_counts[accepted] == _stored_row_count(execution)):
        raise RuntimeError("accepted trajectory-map row counts are wrong")
    if not np.all(row_counts[~accepted] == 0):
        raise RuntimeError("rejected trajectories own dataset rows")

    module = _load_training_util()
    loader = getattr(module, "load_dataset_arrays", None)
    if not callable(loader):
        raise RuntimeError("training utility has no callable load_dataset_arrays")
    dataset = loader(view.manifest)
    if not isinstance(dataset, dict):
        raise TypeError("training loader did not return a dictionary")
    required = (
        "eta",
        "xi",
        "gxi",
        "depth",
        "time",
        "case_id",
        "family_id",
        "split_id",
        "trajectory_index",
        "accepted_mask",
    )
    missing = set(required).difference(dataset)
    if missing:
        raise RuntimeError(f"training loader omitted fields: {sorted(missing)}")

    eta = np.asarray(dataset["eta"])
    xi = np.asarray(dataset["xi"])
    gxi = np.asarray(dataset["gxi"])
    depth = np.asarray(dataset["depth"])
    times = np.asarray(dataset["time"])
    if eta.shape != (expected_rows, execution.numerical.nx):
        raise RuntimeError(
            f"training loader returned eta shape {eta.shape}, "
            f"expected {(expected_rows, execution.numerical.nx)}"
        )
    if xi.shape != eta.shape or gxi.shape != eta.shape:
        raise RuntimeError("training loader returned inconsistent field shapes")
    if depth.shape != (expected_rows,) or times.shape != (expected_rows,):
        raise RuntimeError("training loader returned inconsistent scalar shapes")
    if not all(
        np.isfinite(value).all() for value in (eta, xi, gxi, depth, times)
    ):
        raise RuntimeError("training loader returned nonfinite data")
    if not np.all(np.asarray(dataset["accepted_mask"], dtype=np.bool_)):
        raise RuntimeError("loader exposed a rejected trajectory row")
    expected_family_id = int(FAMILY_IDS[execution.family])
    if not np.all(np.asarray(dataset["family_id"]) == expected_family_id):
        raise RuntimeError("training loader returned the wrong family ID")
    expected_split = split_code(SplitId.TEST)
    if not np.all(np.asarray(dataset["split_id"]) == expected_split):
        raise RuntimeError("training loader returned the wrong split ID")
    unique_cases = int(np.unique(np.asarray(dataset["case_id"])).size)
    if unique_cases != accepted_cases:
        raise RuntimeError("training loader returned the wrong case count")

    return {
        "loader": "train-jax-10m/util.py::load_dataset_arrays",
        "dataset_schema_version": DATASET_VIEW_SCHEMA_VERSION,
        "trajectory_map_schema_version": TRAJECTORY_MAP_SCHEMA_VERSION,
        "attempted_trajectories": attempted_cases,
        "accepted_trajectories": accepted_cases,
        "loaded_rows": int(eta.shape[0]),
        "spatial_size": int(eta.shape[1]),
        "unique_case_count": unique_cases,
        "eta_dtype": str(eta.dtype),
        "xi_dtype": str(xi.dtype),
        "gxi_dtype": str(gxi.dtype),
        "all_fields_and_scalars_finite": True,
        "minimum_depth": float(np.min(depth)),
        "minimum_time": float(np.min(times)),
        "maximum_time": float(np.max(times)),
        "maximum_absolute_loaded_xi_mean": float(
            np.max(np.abs(np.mean(xi, axis=-1)))
        ),
    }


def _summarize_batches(
    state: QuotaRunState,
    spec: AcceptedQuotaRunSpec,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    code_to_cell = {code: cell for cell, code in spec.cell_codes.items()}
    batch_records: list[dict[str, object]] = []
    case_records: list[dict[str, object]] = []
    for paths in state.committed:
        retained_by_local_index: dict[int, dict[str, object]] = {}
        if paths.shard.exists():
            with np.load(paths.shard, allow_pickle=False) as shard:
                local_indices = np.asarray(
                    shard["case_local_index"], dtype=np.int32
                )
                shard_times = np.asarray(shard["time"], dtype=np.float64)
                selected = np.asarray(
                    shard["selected_dense_index"], dtype=np.int32
                )
                depths = np.asarray(shard["depth"], dtype=np.float64)
            for local_index in np.unique(local_indices):
                mask = local_indices == local_index
                retained_by_local_index[int(local_index)] = {
                    "retained_times": shard_times[mask].tolist(),
                    "selected_dense_indices": selected[mask].tolist(),
                    "depth": float(depths[mask][0]),
                }

        with np.load(paths.proposal, allow_pickle=False) as proposal:
            case_ids = np.asarray(proposal["case_id"], dtype=np.int64)
            attempt_indices = np.asarray(
                proposal["attempt_index"], dtype=np.uint64
            )
            encoded_cells = np.asarray(proposal["cell_id"], dtype=np.int32)
            specifications = tuple(
                _strict_json_text(
                    str(value),
                    label=f"{paths.proposal}:case_spec_json",
                )
                for value in proposal["case_spec_json"]
            )
            proposal_metadata = _strict_json_text(
                str(np.asarray(proposal["metadata_json"]).item()),
                label=f"{paths.proposal}:metadata_json",
            )
            batch_id = int(np.asarray(proposal["batch_id"]).item())
        grid_records = proposal_metadata.get("case_time_grids")
        if not isinstance(grid_records, list) or len(grid_records) != len(case_ids):
            raise RuntimeError("proposal metadata does not contain every time grid")

        result = _strict_json_object(paths.result)
        result_cases = result.get("cases")
        if not isinstance(result_cases, list) or len(result_cases) != len(case_ids):
            raise RuntimeError("result does not contain every proposed case")

        for local_index, value in enumerate(result_cases):
            if not isinstance(value, dict):
                raise TypeError("result case must be a JSON object")
            metrics = value.get("metrics")
            if not isinstance(metrics, dict):
                raise TypeError("result case metrics must be a JSON object")
            grid = grid_records[local_index]
            if not isinstance(grid, dict):
                raise TypeError("case time grid must be a JSON object")
            cell_id = code_to_cell[int(encoded_cells[local_index])]
            row_count = int(value["row_count"])
            retained = retained_by_local_index.get(local_index)
            if (retained is not None) != (row_count > 0):
                raise RuntimeError("shard rows disagree with result row ownership")
            case_records.append(
                {
                    "batch_id": batch_id,
                    "case_local_index": local_index,
                    "cell_id": cell_id,
                    "case_id": int(case_ids[local_index]),
                    "attempt_index": int(attempt_indices[local_index]),
                    "accepted": bool(value["accepted"]),
                    "required_bits": int(value["required_bits"]),
                    "evaluated_bits": int(value["evaluated_bits"]),
                    "failed_bits": int(value["failed_bits"]),
                    "first_row": int(value["first_row"]),
                    "row_count": row_count,
                    "time_grid": grid,
                    "case_specification": specifications[local_index],
                    "retained_rows": retained,
                    "diagnostics": {
                        name: metrics.get(name) for name in DIAGNOSTIC_KEYS
                    },
                    "metrics": metrics,
                }
            )

        artifacts: dict[str, object] = {
            "proposal": _artifact_record(paths.proposal, root=spec.root),
            "result": _artifact_record(paths.result, root=spec.root),
        }
        if paths.shard.exists():
            artifacts["shard"] = _artifact_record(paths.shard, root=spec.root)
        batch_records.append(
            {
                "batch_id": batch_id,
                "attempted_cases": len(result_cases),
                "accepted_cases": sum(
                    bool(value["accepted"])
                    for value in result_cases
                    if isinstance(value, dict)
                ),
                "artifacts": artifacts,
            }
        )
    return batch_records, case_records


def run_family(
    output_dir: Path,
    family: TrajectoryFamily,
    *,
    contract: ResidualControlledGL2Contract = SMOKE_CONTRACT,
) -> FamilySmokeResult:
    """Run or resume one family through quota, view, and loader validation."""

    output_dir = output_dir.expanduser().resolve()
    execution = build_execution(family, contract=contract)
    spec = build_run_spec(output_dir, execution)
    invocation_started_at = datetime.now().astimezone()
    total_started = perf_counter()

    scan_started = perf_counter()
    initial_state = scan_quota_run(spec)
    initial_scan_seconds = perf_counter() - scan_started

    executor = TrajectoryQuotaExecutor(
        run_spec=spec,
        execution=execution,
        metadata={
            "smoke": "trajectory_quota_real_gl2_smoke_20260725",
            "evidence_role": "reduced_wiring_evidence_only",
            "real_arm_executor": (
                "solver.gen_data.pipeline.refinement."
                "run_residual_controlled_arm"
            ),
            "run_spec": spec.to_json_record(),
        },
    )
    execution_started = perf_counter()
    state = run_accepted_quotas(spec, executor)
    quota_driver_seconds = perf_counter() - execution_started
    if state.terminal_failure is not None:
        raise RuntimeError(
            f"{family} quota run terminated at "
            f"{state.terminal_failure.failure}"
        )
    if not state.complete:
        raise RuntimeError(f"{family} quota run did not meet its accepted quota")

    view_started = perf_counter()
    view = build_dataset_view(
        output_dir,
        state.committed,
        name=f"trajectory_quota_real_gl2_{family}",
        length=contract.length,
        expected_fingerprint=spec.config_fingerprint,
    )
    dataset_view_seconds = perf_counter() - view_started

    validation_started = perf_counter()
    batches, cases = _summarize_batches(state, spec)
    attempted = len(cases)
    accepted = sum(bool(case["accepted"]) for case in cases)
    rejected = attempted - accepted
    loader_validation = _validate_loader_and_map(
        view,
        execution=execution,
        expected_fingerprint=spec.config_fingerprint,
        attempted_cases=attempted,
        accepted_cases=accepted,
    )
    validation_seconds = perf_counter() - validation_started
    invocation_finished_at = datetime.now().astimezone()

    family_summary: FamilyResultMap = {
        "status": "complete",
        "family": family,
        "selected_cell": SELECTED_CELLS[family],
        "configuration_fingerprint": spec.config_fingerprint,
        "run_spec": spec.to_json_record(),
        "execution": execution.to_json_record(),
        "horizon": execution.horizon.to_json_record(),
        "resume": {
            "had_existing_batches": bool(
                initial_state.committed or initial_state.pending is not None
            ),
            "initial_committed_batches": len(initial_state.committed),
            "initial_accepted_by_cell": dict(initial_state.accepted_by_cell),
            "final_committed_batches": len(state.committed),
            "next_batch_id": state.next_batch_id,
            "next_attempt_index": state.next_attempt_index,
        },
        "counts": {
            "target_accepted": 1,
            "attempted": attempted,
            "accepted": accepted,
            "rejected": rejected,
            "by_cell": {
                SELECTED_CELLS[family]: {
                    "target_accepted": 1,
                    "attempted": attempted,
                    "accepted": accepted,
                    "rejected": rejected,
                }
            },
        },
        "batches": batches,
        "cases": cases,
        "dataset_view": {
            "manifest": _artifact_record(view.manifest, root=output_dir),
            "trajectory_map": _artifact_record(
                view.trajectory_map,
                root=output_dir,
            ),
            "loader_validation": loader_validation,
        },
        "timing_seconds": {
            "initial_scan": initial_scan_seconds,
            "quota_driver": quota_driver_seconds,
            "dataset_view": dataset_view_seconds,
            "validation": validation_seconds,
            "total": perf_counter() - total_started,
        },
        "invocation_started_at": invocation_started_at.isoformat(),
        "invocation_finished_at": invocation_finished_at.isoformat(),
    }
    return FamilySmokeResult(
        family=family,
        execution=execution,
        spec=spec,
        state=state,
        view=view,
        summary=family_summary,
    )


def _runtime_record() -> dict[str, object]:
    devices = jax.devices()
    return {
        "jax_version": jax.__version__,
        "default_backend": jax.default_backend(),
        "x64_enabled": bool(jax.config.x64_enabled),
        "jax_platforms_environment": os.environ.get("JAX_PLATFORMS"),
        "cuda_visible_devices_environment": os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ),
        "devices": [
            {
                "id": int(device.id),
                "platform": device.platform,
                "device_kind": device.device_kind,
            }
            for device in devices
        ],
        "cpu_only_verified": bool(
            devices
            and jax.default_backend() == "cpu"
            and all(device.platform == "cpu" for device in devices)
        ),
    }


def run_smoke(output_dir: Path) -> SmokeRunResult:
    """Run all three real-GL2 quota smokes and write one strict summary."""

    output_dir = output_dir.expanduser().resolve()
    started_at = datetime.now().astimezone()
    started = perf_counter()
    runtime = _runtime_record()
    if not runtime["cpu_only_verified"]:
        raise RuntimeError("trajectory quota smoke must run on CPU only")

    families = tuple(
        run_family(output_dir, family, contract=SMOKE_CONTRACT)
        for family in FAMILIES
    )
    family_summaries = {
        result.family: result.summary for result in families
    }
    attempted = sum(
        int(result.summary["counts"]["attempted"])  # type: ignore[index]
        for result in families
    )
    accepted = sum(
        int(result.summary["counts"]["accepted"])  # type: ignore[index]
        for result in families
    )
    rejected = attempted - accepted
    summary: dict[str, object] = {
        "schema": "trajectory_quota_real_gl2_smoke_summary_v1",
        "status": "complete",
        "evidence_role": "reduced_wiring_evidence_only",
        "claim_scope": (
            "software wiring only; not spatial-, DNO-order-, or "
            "paper-contract numerical validation"
        ),
        "real_arm_executor": (
            "solver.gen_data.pipeline.refinement.run_residual_controlled_arm"
        ),
        "runtime": runtime,
        "contract": asdict(SMOKE_CONTRACT),
        "stored_time_policy": asdict(SMOKE_STORED_TIME_POLICY),
        "source_sha256": source_hashes(),
        "counts": {
            "families": len(families),
            "attempted": attempted,
            "accepted": accepted,
            "rejected": rejected,
        },
        "families": family_summaries,
        "timing_seconds": {
            "family_total_sum": sum(
                float(result.summary["timing_seconds"]["total"])  # type: ignore[index]
                for result in families
            ),
            "total": perf_counter() - started,
        },
        "invocation_started_at": started_at.isoformat(),
        "invocation_finished_at": datetime.now().astimezone().isoformat(),
    }
    summary_path = output_dir / SUMMARY_NAME
    write_json_atomic(summary_path, summary)
    return SmokeRunResult(
        summary_path=summary_path,
        families=families,
        summary=summary,
    )


def main() -> None:
    args = parse_args()
    result = run_smoke(args.output_dir)
    print(
        json.dumps(
            {
                "status": result.summary["status"],
                "summary_path": str(result.summary_path),
                "summary_sha256": file_sha256(result.summary_path),
                "counts": result.summary["counts"],
                "timing_seconds": result.summary["timing_seconds"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
