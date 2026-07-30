"""Run a resumable exact-contract static Stokes accepted-quota pilot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402

from solver.gen_data.pipeline.archive import (  # noqa: E402
    file_sha256,
    write_json_atomic,
)
from solver.gen_data.pipeline.manifest import (  # noqa: E402
    DatasetViewPaths,
    build_dataset_view,
)
from solver.gen_data.pipeline.production import (  # noqa: E402
    PAPER_CORPUS_REVISION_ID,
    CellQuota,
    PhysicalFamilyId,
    SplitId,
)
from solver.gen_data.pipeline.quota_driver import (  # noqa: E402
    AcceptedQuotaRunSpec,
    QuotaRunState,
    run_accepted_quotas,
    scan_quota_run,
)
from solver.gen_data.stokes_population import (  # noqa: E402
    DEFAULT_MAXIMUM_URSELL_REDRAWS,
    STOKES_POPULATION_CELLS,
)
from solver.gen_data.stokes_quota_executor import (  # noqa: E402
    StaticStokesQuotaExecutor,
)
from solver.gen_data.stokes_static_pipeline import (  # noqa: E402
    PAPER_STATIC_STOKES_CONTRACT,
    STOKES_CELL_CODES,
    StaticStokesContract,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/static_stokes_quota_pilot_20260725"
VIEW_NAME = "static_stokes_quota_pilot"
SOURCE_PATHS = (
    Path(__file__).resolve(),
    ROOT / "solver/data/stokes_truth_jax.py",
    ROOT / "solver/gen_data/stokes_population.py",
    ROOT / "solver/gen_data/stokes_quota_executor.py",
    ROOT / "solver/gen_data/stokes_static_pipeline.py",
    ROOT / "solver/gen_data/pipeline/archive.py",
    ROOT / "solver/gen_data/pipeline/manifest.py",
    ROOT / "solver/gen_data/pipeline/production.py",
    ROOT / "solver/gen_data/pipeline/quota_driver.py",
    ROOT / "solver/gen_data/pipeline/reference.py",
    ROOT / "solver/gen_data/pipeline/writer.py",
    ROOT / "solver/solvers/dno_series_jax.py",
    ROOT / "train-jax-10m/util.py",
)


@dataclass(frozen=True)
class PilotRunResult:
    """Paths and audit record returned by one pilot invocation."""

    summary_path: Path
    view: DatasetViewPaths
    state: QuotaRunState
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


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def source_hashes() -> dict[str, str]:
    """Return hashes of every implementation file used by this pilot."""

    return {
        _relative(path, ROOT): file_sha256(path)
        for path in SOURCE_PATHS
    }


def build_run_spec(
    output_dir: Path,
    *,
    contract: StaticStokesContract = PAPER_STATIC_STOKES_CONTRACT,
    maximum_ursell_redraws: int = DEFAULT_MAXIMUM_URSELL_REDRAWS,
) -> AcceptedQuotaRunSpec:
    """Return the immutable four-cell pilot specification."""

    cells = tuple(cell.cell_id for cell in STOKES_POPULATION_CELLS)
    configuration: dict[str, object] = {
        "schema": "static_stokes_quota_pilot_configuration_v1",
        "purpose": "exact-contract accepted-quota pilot",
        "contract": contract.to_json_record(),
        "sampler": {
            "maximum_ursell_redraws": maximum_ursell_redraws,
        },
        "case_kind": "static",
        "retained_times": [0.0],
        "accepted_cases_per_cell": 1,
        "cell_order": list(cells),
        "source_sha256": source_hashes(),
    }
    return AcceptedQuotaRunSpec(
        root=output_dir,
        family_name="stokes",
        family_id=PhysicalFamilyId.STOKES,
        revision_id=PAPER_CORPUS_REVISION_ID,
        split_id=SplitId.VALIDATION,
        stream_id=0,
        quotas=tuple(CellQuota(cell_id, 1) for cell_id in cells),
        cell_codes={
            cell_id: STOKES_CELL_CODES[cell_id]
            for cell_id in cells
        },
        batch_size=len(cells),
        first_attempt_index=0,
        configuration=configuration,
    )


def _load_training_util() -> ModuleType:
    path = ROOT / "train-jax-10m/util.py"
    module_name = "_static_stokes_quota_training_util"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
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


def _validate_loader(
    view: DatasetViewPaths,
    *,
    contract: StaticStokesContract,
    expected_cases: int,
) -> dict[str, object]:
    module = _load_training_util()
    loader = getattr(module, "load_dataset_arrays", None)
    if not callable(loader):
        raise RuntimeError("training utility has no callable load_dataset_arrays")
    dataset = loader(view.manifest)
    if not isinstance(dataset, dict):
        raise TypeError("training loader did not return a dictionary")

    required = ("eta", "xi", "gxi", "case_id", "family_id", "split_id")
    missing = set(required).difference(dataset)
    if missing:
        raise RuntimeError(f"training loader omitted fields: {sorted(missing)}")
    eta = np.asarray(dataset["eta"])
    xi = np.asarray(dataset["xi"])
    gxi = np.asarray(dataset["gxi"])
    case_ids = np.asarray(dataset["case_id"])
    if eta.shape != (expected_cases, contract.target.nx):
        raise RuntimeError(
            "training loader returned unexpected eta shape "
            f"{eta.shape}"
        )
    if xi.shape != eta.shape or gxi.shape != eta.shape:
        raise RuntimeError("training loader returned inconsistent field shapes")
    if np.unique(case_ids).size != expected_cases:
        raise RuntimeError("training loader did not preserve one row per case")
    if not all(np.isfinite(field).all() for field in (eta, xi, gxi)):
        raise RuntimeError("training loader returned a nonfinite field")
    family_ids = np.asarray(dataset["family_id"])
    split_ids = np.asarray(dataset["split_id"])
    if not np.all(family_ids == int(PhysicalFamilyId.STOKES)):
        raise RuntimeError("training loader returned a non-Stokes family ID")
    if not np.all(split_ids == 1):
        raise RuntimeError("training loader returned a non-validation split")

    return {
        "loader": "train-jax-10m/util.py::load_dataset_arrays",
        "loaded_rows": int(eta.shape[0]),
        "spatial_size": int(eta.shape[1]),
        "unique_case_count": int(np.unique(case_ids).size),
        "eta_dtype": str(eta.dtype),
        "xi_dtype": str(xi.dtype),
        "gxi_dtype": str(gxi.dtype),
        "all_fields_finite": True,
        "maximum_absolute_loaded_xi_mean": float(
            np.max(np.abs(np.mean(xi, axis=-1)))
        ),
    }


def _artifact_record(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": _relative(path, root),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _summarize_batches(
    state: QuotaRunState,
    spec: AcceptedQuotaRunSpec,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    code_to_cell = {code: cell for cell, code in spec.cell_codes.items()}
    batch_records: list[dict[str, object]] = []
    case_records: list[dict[str, object]] = []
    for paths in state.committed:
        with np.load(paths.proposal, allow_pickle=False) as proposal:
            case_ids = np.asarray(proposal["case_id"], dtype=np.int64)
            attempt_indices = np.asarray(
                proposal["attempt_index"],
                dtype=np.uint64,
            )
            encoded_cells = np.asarray(proposal["cell_id"], dtype=np.int32)
            specifications = tuple(
                json.loads(str(value))
                for value in proposal["case_spec_json"]
            )
            batch_id = int(np.asarray(proposal["batch_id"]).item())
        result = _strict_json_object(paths.result)
        result_cases = result.get("cases")
        if not isinstance(result_cases, list) or len(result_cases) != len(case_ids):
            raise RuntimeError("result does not contain every proposed case")

        for local_index, value in enumerate(result_cases):
            if not isinstance(value, dict):
                raise TypeError("result case must be a JSON object")
            cell_id = code_to_cell[int(encoded_cells[local_index])]
            specification = specifications[local_index]
            if not isinstance(specification, dict):
                raise TypeError("case specification must be a JSON object")
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
                    "row_count": int(value["row_count"]),
                    "support_resampling_count": specification.get(
                        "support_resampling_count"
                    ),
                    "branch": specification.get("branch"),
                    "carrier_mode": specification.get("carrier_mode"),
                    "depth": specification.get("depth"),
                    "amplitude": specification.get("amplitude"),
                    "steepness": specification.get("steepness"),
                    "ursell_upper_bound": specification.get(
                        "finite_depth_ursell_limit"
                    )
                    and specification.get("amplitude_attempts", [{}])[-1].get(
                        "ursell_upper_bound"
                    ),
                    "sampling_status": specification.get("status"),
                    "metrics": value.get("metrics", {}),
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


def run_pilot(
    output_dir: Path,
    *,
    contract: StaticStokesContract = PAPER_STATIC_STOKES_CONTRACT,
    maximum_ursell_redraws: int = DEFAULT_MAXIMUM_URSELL_REDRAWS,
) -> PilotRunResult:
    """Run or resume the pilot and write its derived audit products."""

    output_dir = output_dir.expanduser().resolve()
    spec = build_run_spec(
        output_dir,
        contract=contract,
        maximum_ursell_redraws=maximum_ursell_redraws,
    )
    invocation_started_at = datetime.now().astimezone()
    total_started = perf_counter()

    # This scan is deliberately first: existing incompatible proposal
    # fingerprints fail closed before any new numerical work is scheduled.
    scan_started = perf_counter()
    initial_state = scan_quota_run(spec)
    initial_scan_seconds = perf_counter() - scan_started

    executor = StaticStokesQuotaExecutor(
        run_spec=spec,
        contract=contract,
        maximum_ursell_redraws=maximum_ursell_redraws,
        metadata={
            "pilot": VIEW_NAME,
            "run_spec": spec.to_json_record(),
        },
    )
    execution_started = perf_counter()
    state = run_accepted_quotas(spec, executor)
    quota_driver_seconds = perf_counter() - execution_started
    if state.terminal_failure is not None:
        raise RuntimeError(
            f"quota run terminated at {state.terminal_failure.failure}"
        )
    if not state.complete:
        raise RuntimeError("quota run returned without meeting every quota")

    view_started = perf_counter()
    view = build_dataset_view(
        output_dir,
        state.committed,
        name=VIEW_NAME,
        length=contract.target.length,
        expected_fingerprint=spec.config_fingerprint,
    )
    dataset_view_seconds = perf_counter() - view_started

    validation_started = perf_counter()
    batch_records, case_records = _summarize_batches(state, spec)
    attempted_by_cell = {
        quota.cell_id: sum(
            record["cell_id"] == quota.cell_id for record in case_records
        )
        for quota in spec.quotas
    }
    accepted_by_cell = dict(state.accepted_by_cell)
    counts_by_cell = {
        quota.cell_id: {
            "target_accepted": quota.target_accepted,
            "attempted": attempted_by_cell[quota.cell_id],
            "accepted": accepted_by_cell[quota.cell_id],
            "rejected": (
                attempted_by_cell[quota.cell_id]
                - accepted_by_cell[quota.cell_id]
            ),
        }
        for quota in spec.quotas
    }
    accepted_cases = sum(accepted_by_cell.values())
    loader_validation = _validate_loader(
        view,
        contract=contract,
        expected_cases=accepted_cases,
    )
    validation_seconds = perf_counter() - validation_started
    invocation_finished_at = datetime.now().astimezone()

    summary: dict[str, object] = {
        "schema": "static_stokes_quota_pilot_summary_v1",
        "status": "complete",
        "configuration_fingerprint": spec.config_fingerprint,
        "run_spec": spec.to_json_record(),
        "contract": contract.to_json_record(),
        "source_sha256": source_hashes(),
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
            "attempted": len(case_records),
            "accepted": accepted_cases,
            "rejected": len(case_records) - accepted_cases,
            "by_cell": counts_by_cell,
        },
        "batches": batch_records,
        "cases": case_records,
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
    summary_path = output_dir / "summary.json"
    write_json_atomic(summary_path, summary)
    return PilotRunResult(
        summary_path=summary_path,
        view=view,
        state=state,
        summary=summary,
    )


def main() -> None:
    args = parse_args()
    result = run_pilot(args.output_dir)
    print(json.dumps(result.summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
