"""Preflight, run, or resume one exact-contract paper-corpus family quota.

The default mode is a read-only preflight.  Numerical generation begins only
when ``--execute`` is supplied.  A run is identified by its output root,
family, split, stream, accepted-case quota, batch size, exact paper contract,
execution platform, and source hashes.  Changing any of those values makes an
existing run fail closed instead of silently mixing artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
import os
import platform as python_platform
from pathlib import Path
import sys
from time import perf_counter
from typing import Literal, Sequence, TypeAlias


Platform: TypeAlias = Literal["cpu", "gpu"]


def _bootstrap_platform(argv: Sequence[str]) -> Platform:
    """Read only ``--platform`` before importing modules that initialize JAX."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parsed, _ = parser.parse_known_args(argv)
    platform = parsed.platform
    if platform not in ("cpu", "gpu"):
        raise ValueError(f"unsupported platform: {platform}")
    return platform


BOOTSTRAP_PLATFORM = _bootstrap_platform(sys.argv[1:])
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["DNO_TANAKA_DTYPE"] = "float64"
os.environ["JAX_PLATFORMS"] = "cpu" if BOOTSTRAP_PLATFORM == "cpu" else "cuda"
if BOOTSTRAP_PLATFORM == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


import jax  # noqa: E402
import jaxlib  # noqa: E402
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
    DatasetViewPaths,
    build_dataset_view,
)
from solver.gen_data.pipeline.production import (  # noqa: E402
    PAPER_CORPUS_REVISION_ID,
    CellQuota,
    PhysicalFamilyId,
    SplitId,
    balanced_cell_quotas,
)
from solver.gen_data.pipeline.quality import (  # noqa: E402
    reasons_from_bits,
)
from solver.gen_data.pipeline.quota_driver import (  # noqa: E402
    DEFAULT_MAXIMUM_ATTEMPTS_PER_ACCEPTED_CASE,
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
    StaticStokesContract,
)
from solver.gen_data.tanaka_population import (  # noqa: E402
    TANAKA_POPULATION_CELLS,
)
from solver.gen_data.trajectory_quota_executor import (  # noqa: E402
    TrajectoryExecutionConfig,
    TrajectoryQuotaExecutor,
)


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[1]
PaperFamily: TypeAlias = Literal[
    "stokes",
    "tanaka",
    "benjamin_feir",
    "jonswap_tma",
]
PaperExecution: TypeAlias = StaticStokesContract | TrajectoryExecutionConfig
COMMON_SOURCE_PATHS = (
    Path(__file__).resolve(),
    ROOT / "solver/gen_data/pipeline/archive.py",
    ROOT / "solver/gen_data/pipeline/manifest.py",
    ROOT / "solver/gen_data/pipeline/production.py",
    ROOT / "solver/gen_data/pipeline/quality.py",
    ROOT / "solver/gen_data/pipeline/quota_driver.py",
    ROOT / "solver/gen_data/pipeline/reference.py",
    ROOT / "solver/gen_data/pipeline/writer.py",
    ROOT / "solver/solvers/dno_series_jax.py",
)
TRAJECTORY_SOURCE_PATHS = (
    ROOT / "solver/gen_data/trajectory_family_adapters.py",
    ROOT / "solver/gen_data/trajectory_quota_executor.py",
    ROOT / "solver/gen_data/pipeline/acceptance.py",
    ROOT / "solver/gen_data/pipeline/refinement.py",
    ROOT / "solver/gen_data/pipeline/time_selection.py",
    ROOT / "solver/gen_data/pipeline/trajectory_writer.py",
    ROOT / "solver/solvers/time_integrator.py",
)
FAMILY_SOURCE_PATHS: dict[PaperFamily, tuple[Path, ...]] = {
    "stokes": (
        ROOT / "solver/data/stokes_truth_jax.py",
        ROOT / "solver/gen_data/generate_stokes_dataset.py",
        ROOT / "solver/gen_data/stokes_population.py",
        ROOT / "solver/gen_data/stokes_quota_executor.py",
        ROOT / "solver/gen_data/stokes_static_pipeline.py",
    ),
    "tanaka": (
        *TRAJECTORY_SOURCE_PATHS,
        ROOT / "solver/gen_data/adaptive_sampling.py",
        ROOT / "solver/gen_data/generate_tanaka_dataset_v2.py",
        ROOT / "solver/gen_data/multi_crest.py",
        ROOT / "solver/gen_data/tanaka_population.py",
        ROOT / "solver/tanaka_ICs/modified_tanaka.py",
    ),
    "benjamin_feir": (
        *TRAJECTORY_SOURCE_PATHS,
        ROOT / "solver/data/stokes_truth_jax.py",
        ROOT / "solver/gen_data/benjamin_feir_jcp09.py",
        ROOT / "solver/gen_data/benjamin_feir_population.py",
    ),
    "jonswap_tma": (
        *TRAJECTORY_SOURCE_PATHS,
        ROOT / "solver/gen_data/jonswap_tma.py",
        ROOT / "solver/gen_data/jonswap_tma_population.py",
    ),
}
DEPENDENCY_FILES = (
    ROOT / "pyproject.toml",
    ROOT / "uv.lock",
)
FAMILY_IDS: dict[PaperFamily, PhysicalFamilyId] = {
    "stokes": PhysicalFamilyId.STOKES,
    "tanaka": PhysicalFamilyId.TANAKA,
    "benjamin_feir": PhysicalFamilyId.BENJAMIN_FEIR,
    "jonswap_tma": PhysicalFamilyId.JONSWAP_TMA,
}
FAMILY_CELL_IDS: dict[PaperFamily, tuple[str, ...]] = {
    "stokes": tuple(cell.cell_id for cell in STOKES_POPULATION_CELLS),
    "tanaka": tuple(cell.cell_id for cell in TANAKA_POPULATION_CELLS),
    "benjamin_feir": tuple(
        cell.cell_id for cell in BENJAMIN_FEIR_POPULATION_CELLS
    ),
    "jonswap_tma": tuple(
        cell.cell_id for cell in JONSWAP_TMA_POPULATION_CELLS
    ),
}


@dataclass(frozen=True)
class GenerationRequest:
    """User-visible identity and allocation choices for one quota run."""

    output_root: Path
    family: PaperFamily
    split: SplitId
    accepted_cases: int
    batch_size: int
    accepted_cases_before: int = 0
    platform: Platform = "cpu"
    stream_id: int = 0
    first_attempt_index: int = 0
    maximum_attempts_per_accepted_case: int = (
        DEFAULT_MAXIMUM_ATTEMPTS_PER_ACCEPTED_CASE
    )

    def __post_init__(self) -> None:
        if self.family not in FAMILY_IDS:
            raise ValueError(f"unknown paper-corpus family: {self.family}")
        if not isinstance(self.split, SplitId):
            raise TypeError("split must be a SplitId")
        for value, name, positive in (
            (self.accepted_cases, "accepted_cases", True),
            (self.batch_size, "batch_size", True),
            (self.accepted_cases_before, "accepted_cases_before", False),
            (self.stream_id, "stream_id", False),
            (self.first_attempt_index, "first_attempt_index", False),
            (
                self.maximum_attempts_per_accepted_case,
                "maximum_attempts_per_accepted_case",
                True,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if (positive and value <= 0) or (not positive and value < 0):
                qualifier = "positive" if positive else "nonnegative"
                raise ValueError(f"{name} must be {qualifier}")
        if self.platform not in ("cpu", "gpu"):
            raise ValueError(f"unsupported platform: {self.platform}")
        object.__setattr__(
            self,
            "output_root",
            Path(self.output_root).expanduser().resolve(),
        )


@dataclass(frozen=True)
class GenerationRunResult:
    """Products of one completed exact-contract invocation."""

    summary_path: Path
    view: DatasetViewPaths
    state: QuotaRunState
    summary: dict[str, object]


def _positive_integer(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _nonnegative_integer(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=tuple(FAMILY_IDS),
        required=True,
        help="Physical paper-corpus family to generate.",
    )
    parser.add_argument(
        "--split",
        choices=tuple(split.value for split in SplitId),
        required=True,
        help="Case-level dataset split.",
    )
    parser.add_argument(
        "--accepted-cases",
        type=_positive_integer,
        required=True,
        help="Total accepted-case quota, balanced over the ordered family cells.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root containing durable proposals, shards, results, and views.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_integer,
        required=True,
        help="Maximum attempted cases in one transaction.",
    )
    parser.add_argument(
        "--accepted-cases-before",
        type=_nonnegative_integer,
        default=0,
        help=(
            "Cumulative accepted count already assigned to earlier chunks; "
            "defaults to zero for a standalone pilot."
        ),
    )
    parser.add_argument(
        "--platform",
        choices=("cpu", "gpu"),
        default="cpu",
        help="JAX execution platform; defaults to the fail-safe CPU path.",
    )
    parser.add_argument(
        "--stream-id",
        type=_nonnegative_integer,
        default=0,
        help="Independent deterministic stream within this family and split.",
    )
    parser.add_argument(
        "--first-attempt-index",
        type=_nonnegative_integer,
        default=0,
        help="First attempt coordinate in the selected deterministic stream.",
    )
    parser.add_argument(
        "--maximum-attempts-per-accepted-case",
        type=_positive_integer,
        default=DEFAULT_MAXIMUM_ATTEMPTS_PER_ACCEPTED_CASE,
        help=(
            "Per-cell attempted-case ceiling multiplier; a cell with target "
            "Q stops the run after this value times Q durable attempts "
            f"(default: {DEFAULT_MAXIMUM_ATTEMPTS_PER_ACCEPTED_CASE})."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Run or resume numerical generation after the preflight scan.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the read-only preflight explicitly (this is the default).",
    )
    return parser.parse_args(argv)


def request_from_args(args: argparse.Namespace) -> GenerationRequest:
    return GenerationRequest(
        output_root=args.output_root,
        family=args.family,
        split=SplitId(args.split),
        accepted_cases=args.accepted_cases,
        batch_size=args.batch_size,
        accepted_cases_before=args.accepted_cases_before,
        platform=args.platform,
        stream_id=args.stream_id,
        first_attempt_index=args.first_attempt_index,
        maximum_attempts_per_accepted_case=(
            args.maximum_attempts_per_accepted_case
        ),
    )


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def source_hashes(family: PaperFamily) -> dict[str, str]:
    """Hash the actual implementation files used by ``family``."""

    paths = tuple(dict.fromkeys((*COMMON_SOURCE_PATHS, *FAMILY_SOURCE_PATHS[family])))
    missing = tuple(path for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "generation source path does not exist: "
            + ", ".join(map(str, missing))
        )
    return {
        _relative(path, ROOT): file_sha256(path)
        for path in paths
    }


def dependency_environment() -> dict[str, object]:
    """Return the dependency identity bound into every generation run."""

    missing = tuple(path for path in DEPENDENCY_FILES if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "dependency lock/configuration path does not exist: "
            + ", ".join(map(str, missing))
        )
    return {
        "python": {
            "implementation": python_platform.python_implementation(),
            "version": python_platform.python_version(),
        },
        "packages": {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": np.__version__,
        },
        "files_sha256": {
            _relative(path, ROOT): file_sha256(path)
            for path in DEPENDENCY_FILES
        },
    }


def _paper_execution(family: PaperFamily) -> PaperExecution:
    if family == "stokes":
        return PAPER_STATIC_STOKES_CONTRACT
    return TrajectoryExecutionConfig.paper(family)


def _execution_record(execution: PaperExecution) -> dict[str, object]:
    return execution.to_json_record()


def incremental_cell_quotas(
    cell_ids: Sequence[str],
    *,
    accepted_cases_before: int,
    accepted_case_count: int,
) -> tuple[CellQuota, ...]:
    """Return the balanced cumulative-quota increment for one chunk."""

    before = balanced_cell_quotas(
        cell_ids,
        accepted_case_count=accepted_cases_before,
    )
    after = balanced_cell_quotas(
        cell_ids,
        accepted_case_count=accepted_cases_before + accepted_case_count,
    )
    quotas = tuple(
        CellQuota(
            cell_id=after_quota.cell_id,
            target_accepted=(
                after_quota.target_accepted - before_quota.target_accepted
            ),
        )
        for before_quota, after_quota in zip(before, after)
    )
    if any(quota.target_accepted < 0 for quota in quotas):
        raise RuntimeError("balanced cumulative quotas must be monotone")
    if sum(quota.target_accepted for quota in quotas) != accepted_case_count:
        raise RuntimeError("incremental cell quotas do not sum to the chunk size")
    return quotas


def build_run_spec(
    request: GenerationRequest,
    *,
    execution: PaperExecution | None = None,
) -> AcceptedQuotaRunSpec:
    """Build the immutable exact-contract allocation for ``request``."""

    selected_execution = execution or _paper_execution(request.family)
    if selected_execution != _paper_execution(request.family):
        raise ValueError("the paper-corpus launcher requires the exact contract")
    cells = FAMILY_CELL_IDS[request.family]
    quotas = incremental_cell_quotas(
        cells,
        accepted_cases_before=request.accepted_cases_before,
        accepted_case_count=request.accepted_cases,
    )
    common_configuration: dict[str, object] = {
        "schema": "paper_corpus_quota_configuration_v1",
        "purpose": "exact paper-contract accepted-case generation",
        "case_kind": "static" if request.family == "stokes" else "trajectory",
        "accepted_case_count": request.accepted_cases,
        "accepted_cases_before": request.accepted_cases_before,
        "accepted_cases_after": (
            request.accepted_cases_before + request.accepted_cases
        ),
        "ordered_cell_ids": list(cells),
        "execution_platform": request.platform,
        "dependency_environment": dependency_environment(),
        "source_sha256": source_hashes(request.family),
    }
    if request.family == "stokes":
        assert isinstance(selected_execution, StaticStokesContract)
        family_configuration = {
            "contract": selected_execution.to_json_record(),
            "sampler": {
                "maximum_ursell_redraws": DEFAULT_MAXIMUM_URSELL_REDRAWS,
            },
        }
    else:
        assert isinstance(selected_execution, TrajectoryExecutionConfig)
        family_configuration = {
            "trajectory_execution": selected_execution.to_json_record(),
        }
    return AcceptedQuotaRunSpec(
        root=request.output_root,
        family_name=request.family,
        family_id=FAMILY_IDS[request.family],
        revision_id=PAPER_CORPUS_REVISION_ID,
        split_id=request.split,
        stream_id=request.stream_id,
        quotas=quotas,
        cell_codes={
            cell_id: cell_code for cell_code, cell_id in enumerate(cells)
        },
        batch_size=request.batch_size,
        first_attempt_index=request.first_attempt_index,
        maximum_attempts_per_accepted_case=(
            request.maximum_attempts_per_accepted_case
        ),
        configuration={
            **common_configuration,
            **family_configuration,
        },
    )


def _stored_rows_per_case(execution: PaperExecution) -> int:
    if isinstance(execution, StaticStokesContract):
        return 1
    return {
        "tanaka": execution.stored_time_policy.tanaka_count,
        "benjamin_feir": execution.stored_time_policy.benjamin_feir_count,
        "jonswap_tma": execution.stored_time_policy.random_sea_count,
    }[execution.family]


def _state_record(state: QuotaRunState) -> dict[str, object]:
    return {
        "complete": state.complete,
        "accepted_by_cell": dict(state.accepted_by_cell),
        "attempted_by_cell": dict(state.attempted_by_cell),
        "committed_batches": len(state.committed),
        "pending_batch_id": (
            state.pending.batch_id if state.pending is not None else None
        ),
        "pending_status": (
            state.pending.status.value if state.pending is not None else None
        ),
        "terminal_failure": (
            str(state.terminal_failure.failure)
            if state.terminal_failure is not None
            else None
        ),
        "attempt_limit_exhausted_cells": (
            list(state.attempt_limit_failure.exhausted_cells)
            if state.attempt_limit_failure is not None
            else []
        ),
        "attempt_limit_failure": (
            state.attempt_limit_failure.message
            if state.attempt_limit_failure is not None
            else None
        ),
        "next_batch_id": state.next_batch_id,
        "next_attempt_index": state.next_attempt_index,
    }


def _runtime_record() -> dict[str, object]:
    devices = jax.devices()
    return {
        "requested_platform": BOOTSTRAP_PLATFORM,
        "jax_version": jax.__version__,
        "default_backend": jax.default_backend(),
        "x64_enabled": bool(jax.config.x64_enabled),
        "devices": [
            {
                "id": int(device.id),
                "platform": device.platform,
                "device_kind": device.device_kind,
            }
            for device in devices
        ],
    }


def _require_runtime(request: GenerationRequest) -> dict[str, object]:
    if request.platform != BOOTSTRAP_PLATFORM:
        raise RuntimeError(
            "the requested platform differs from the platform selected before "
            "JAX initialization; invoke this script in a fresh process"
        )
    runtime = _runtime_record()
    expected_backend = request.platform
    if not runtime["x64_enabled"]:
        raise RuntimeError("paper generation requires JAX float64 mode")
    if runtime["default_backend"] != expected_backend:
        raise RuntimeError(
            f"requested {request.platform}, but JAX initialized "
            f"{runtime['default_backend']}"
        )
    devices = runtime["devices"]
    if not isinstance(devices, list) or not devices:
        raise RuntimeError("JAX reported no execution devices")
    return runtime


def preflight(request: GenerationRequest) -> tuple[
    AcceptedQuotaRunSpec,
    PaperExecution,
    QuotaRunState,
    dict[str, object],
]:
    """Perform a read-only compatibility scan and return its complete plan."""

    execution = _paper_execution(request.family)
    spec = build_run_spec(request, execution=execution)
    state = scan_quota_run(spec)
    cumulative_before = balanced_cell_quotas(
        FAMILY_CELL_IDS[request.family],
        accepted_case_count=request.accepted_cases_before,
    )
    cumulative_after = balanced_cell_quotas(
        FAMILY_CELL_IDS[request.family],
        accepted_case_count=(
            request.accepted_cases_before + request.accepted_cases
        ),
    )
    attempt_ceilings = spec.attempt_ceiling_by_cell
    quotas = [
        {
            "cell_id": quota.cell_id,
            "accepted_before": before.target_accepted,
            "chunk_target_accepted": quota.target_accepted,
            "accepted_after": after.target_accepted,
            "attempt_ceiling": attempt_ceilings[quota.cell_id],
            "durable_attempted": state.attempted_by_cell[quota.cell_id],
            "remaining_attempt_capacity": (
                attempt_ceilings[quota.cell_id]
                - state.attempted_by_cell[quota.cell_id]
            ),
        }
        for before, quota, after in zip(
            cumulative_before,
            spec.quotas,
            cumulative_after,
        )
    ]
    nonzero_cells = sum(quota.target_accepted > 0 for quota in spec.quotas)
    stored_rows_per_case = _stored_rows_per_case(execution)
    plan: dict[str, object] = {
        "schema": "paper_corpus_quota_preflight_v1",
        "mode": "dry_run",
        "no_numerical_generation_performed": True,
        "output_root": str(request.output_root),
        "artifact_namespace": {
            "family": request.family,
            "split": request.split.value,
            "view_name": _view_name(request),
            "summary_path": str(_summary_path(request)),
        },
        "configuration_fingerprint": spec.config_fingerprint,
        "run_spec": spec.to_json_record(),
        "execution": _execution_record(execution),
        "allocation": {
            "chunk_accepted_cases": request.accepted_cases,
            "accepted_cases_before": request.accepted_cases_before,
            "accepted_cases_after": (
                request.accepted_cases_before + request.accepted_cases
            ),
            "cell_count": len(spec.quotas),
            "nonzero_quota_cell_count": nonzero_cells,
            "quota_minimum": min(
                quota.target_accepted for quota in spec.quotas
            ),
            "quota_maximum": max(
                quota.target_accepted for quota in spec.quotas
            ),
            "maximum_attempts_per_accepted_case": (
                spec.maximum_attempts_per_accepted_case
            ),
            "quotas": quotas,
            "chunk_identity": {
                "stream_id": request.stream_id,
                "first_attempt_index": request.first_attempt_index,
                "output_root": str(request.output_root),
                "rule": (
                    "use a distinct stream_id and output_root for every "
                    "additive chunk"
                ),
            },
        },
        "expected_output": {
            "stored_rows_per_accepted_case": stored_rows_per_case,
            "retained_rows": request.accepted_cases * stored_rows_per_case,
            "spatial_points_per_row": (
                execution.target.nx
                if isinstance(execution, StaticStokesContract)
                else execution.numerical.nx
            ),
            "field_values_per_row": 3
            * (
                execution.target.nx
                if isinstance(execution, StaticStokesContract)
                else execution.numerical.nx
            ),
        },
        "resume_state": _state_record(state),
        "runtime": _runtime_record(),
    }
    return spec, execution, state, plan


def _view_name(request: GenerationRequest) -> str:
    return f"paper_corpus_{request.family}_{request.split.value}"


def _summary_path(request: GenerationRequest) -> Path:
    return request.output_root / f"{_view_name(request)}.summary.json"


def _read_json_object(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{path} contains nonfinite JSON constant {value!r}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _artifact_record(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _summarize_committed_cases(
    state: QuotaRunState,
    spec: AcceptedQuotaRunSpec,
) -> dict[str, object]:
    code_to_cell = {code: cell for cell, code in spec.cell_codes.items()}
    attempted = Counter({quota.cell_id: 0 for quota in spec.quotas})
    accepted = Counter({quota.cell_id: 0 for quota in spec.quotas})
    rejection_reasons: Counter[str] = Counter()

    for paths in state.committed:
        with np.load(paths.proposal, allow_pickle=False) as proposal:
            encoded_cells = np.asarray(proposal["cell_id"], dtype=np.int32)
        result = _read_json_object(paths.result)
        cases = result.get("cases")
        if not isinstance(cases, list) or len(cases) != encoded_cells.size:
            raise RuntimeError("result does not contain every proposed case")
        for local_index, case in enumerate(cases):
            if not isinstance(case, dict):
                raise TypeError("result case must be a JSON object")
            cell_id = code_to_cell[int(encoded_cells[local_index])]
            attempted[cell_id] += 1
            if bool(case.get("accepted")):
                accepted[cell_id] += 1
                continue
            failed_bits = case.get("failed_bits")
            if isinstance(failed_bits, bool) or not isinstance(failed_bits, int):
                raise TypeError("result failed_bits must be an integer")
            names = tuple(reason.name for reason in reasons_from_bits(failed_bits))
            rejection_reasons["+".join(names) if names else "missing_check"] += 1

    if dict(accepted) != dict(state.accepted_by_cell):
        raise RuntimeError("summary counts disagree with quota-driver state")
    if dict(attempted) != dict(state.attempted_by_cell):
        raise RuntimeError("summary attempts disagree with quota-driver state")
    by_cell = {
        quota.cell_id: {
            "target_accepted": quota.target_accepted,
            "attempted": attempted[quota.cell_id],
            "accepted": accepted[quota.cell_id],
            "rejected": attempted[quota.cell_id] - accepted[quota.cell_id],
        }
        for quota in spec.quotas
    }
    attempted_total = sum(attempted.values())
    accepted_total = sum(accepted.values())
    return {
        "attempted": attempted_total,
        "accepted": accepted_total,
        "rejected": attempted_total - accepted_total,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "by_cell": by_cell,
    }


def _validate_view(
    view: DatasetViewPaths,
    *,
    request: GenerationRequest,
    spec: AcceptedQuotaRunSpec,
    execution: PaperExecution,
    attempted_cases: int,
) -> dict[str, object]:
    manifest = _read_json_object(view.manifest)
    expected_rows = request.accepted_cases * _stored_rows_per_case(execution)
    expected_values = {
        "schema_version": DATASET_VIEW_SCHEMA_VERSION,
        "configuration_fingerprint": spec.config_fingerprint,
        "n_rows": expected_rows,
        "n_trajectories": attempted_cases,
        "n_accepted_trajectories": request.accepted_cases,
        "n_accepted_rows": expected_rows,
    }
    for name, expected in expected_values.items():
        if manifest.get(name) != expected:
            raise RuntimeError(
                f"dataset view {name} is {manifest.get(name)!r}, expected {expected!r}"
            )
    if manifest.get("trajectory_map_sha256") != file_sha256(view.trajectory_map):
        raise RuntimeError("trajectory-map hash differs from its manifest")
    grid = manifest.get("grid")
    if not isinstance(grid, dict):
        raise TypeError("dataset view grid must be a JSON object")
    expected_grid = (
        {
            "length": execution.target.length,
            "nx": execution.target.nx,
        }
        if isinstance(execution, StaticStokesContract)
        else {
            "length": execution.numerical.length,
            "nx": execution.numerical.nx,
        }
    )
    if grid != expected_grid:
        raise RuntimeError(
            f"dataset view grid is {grid!r}, expected {expected_grid!r}"
        )
    return {
        **expected_values,
        "grid": expected_grid,
        "manifest": _artifact_record(view.manifest, root=request.output_root),
        "trajectory_map": _artifact_record(
            view.trajectory_map,
            root=request.output_root,
        ),
    }


def run_generation(request: GenerationRequest) -> GenerationRunResult:
    """Run or resume the exact quota, then build and validate its dataset view."""

    spec, execution, initial_state, plan = preflight(request)
    runtime = _require_runtime(request)
    if initial_state.terminal_failure is not None:
        raise RuntimeError(
            f"quota run already terminated at {initial_state.terminal_failure.failure}"
        )
    if initial_state.attempt_limit_failure is not None:
        raise RuntimeError(initial_state.attempt_limit_failure.message)

    invocation_started_at = datetime.now().astimezone()
    total_started = perf_counter()
    metadata = {
        "launcher": "scripts/run_paper_corpus_quota.py",
        "run_spec": spec.to_json_record(),
    }
    if isinstance(execution, StaticStokesContract):
        executor = StaticStokesQuotaExecutor(
            run_spec=spec,
            contract=execution,
            maximum_ursell_redraws=DEFAULT_MAXIMUM_URSELL_REDRAWS,
            metadata=metadata,
        )
        length = execution.target.length
    else:
        executor = TrajectoryQuotaExecutor(
            run_spec=spec,
            execution=execution,
            metadata=metadata,
        )
        length = execution.numerical.length
    quota_started = perf_counter()
    state = run_accepted_quotas(spec, executor)
    quota_seconds = perf_counter() - quota_started
    if state.terminal_failure is not None:
        raise RuntimeError(
            f"quota run terminated at {state.terminal_failure.failure}"
        )
    if state.attempt_limit_failure is not None:
        raise RuntimeError(state.attempt_limit_failure.message)
    if not state.complete:
        raise RuntimeError("quota run returned before meeting every accepted quota")

    view_started = perf_counter()
    view = build_dataset_view(
        request.output_root,
        state.committed,
        name=_view_name(request),
        length=length,
        expected_fingerprint=spec.config_fingerprint,
    )
    view_seconds = perf_counter() - view_started
    validation_started = perf_counter()
    counts = _summarize_committed_cases(state, spec)
    attempted_cases = counts["attempted"]
    if isinstance(attempted_cases, bool) or not isinstance(attempted_cases, int):
        raise TypeError("attempted case count must be an integer")
    view_record = _validate_view(
        view,
        request=request,
        spec=spec,
        execution=execution,
        attempted_cases=attempted_cases,
    )
    validation_seconds = perf_counter() - validation_started

    summary: dict[str, object] = {
        "schema": "paper_corpus_quota_summary_v1",
        "status": "complete",
        "configuration_fingerprint": spec.config_fingerprint,
        "output_root": str(request.output_root),
        "run_spec": spec.to_json_record(),
        "execution": _execution_record(execution),
        "runtime": runtime,
        "preflight": plan,
        "resume": {
            "initial": _state_record(initial_state),
            "final": _state_record(state),
        },
        "counts": counts,
        "dataset_view": view_record,
        "timing_seconds": {
            "quota_driver": quota_seconds,
            "dataset_view": view_seconds,
            "validation": validation_seconds,
            "total": perf_counter() - total_started,
        },
        "invocation_started_at": invocation_started_at.isoformat(),
        "invocation_finished_at": datetime.now().astimezone().isoformat(),
    }
    summary_path = _summary_path(request)
    write_json_atomic(summary_path, summary)
    return GenerationRunResult(
        summary_path=summary_path,
        view=view,
        state=state,
        summary=summary,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    request = request_from_args(args)
    if args.execute:
        result = run_generation(request)
        output: dict[str, object] = {
            "mode": "execute",
            "status": result.summary["status"],
            "summary_path": str(result.summary_path),
            "configuration_fingerprint": result.summary[
                "configuration_fingerprint"
            ],
            "counts": result.summary["counts"],
            "dataset_view": result.summary["dataset_view"],
            "timing_seconds": result.summary["timing_seconds"],
        }
    else:
        _, _, _, output = preflight(request)
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
