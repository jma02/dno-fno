"""Run one exact-paper-target static Stokes case in each allocation cell."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from solver.gen_data.pipeline.archive import (  # noqa: E402
    file_sha256,
    write_json_atomic,
)
from solver.gen_data.pipeline.manifest import build_dataset_view  # noqa: E402
from solver.gen_data.pipeline.production import (  # noqa: E402
    PAPER_CORPUS_REVISION_ID,
    AttemptAssignment,
    CaseKey,
    PhysicalFamilyId,
    SplitId,
)
from solver.gen_data.pipeline.reference import PAPER_DNO_TARGET  # noqa: E402
from solver.gen_data.stokes_population import (  # noqa: E402
    STOKES_POPULATION_CELLS,
    sample_stokes_population,
)
from solver.gen_data.stokes_static_pipeline import (  # noqa: E402
    PAPER_STATIC_STOKES_CONTRACT,
    write_static_stokes_batch,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "outputs/static_stokes_exact_target_pilot_20260725"
)
HASHED_SOURCES = (
    ROOT / "solver/data/stokes_truth_jax.py",
    ROOT / "solver/gen_data/stokes_population.py",
    ROOT / "solver/gen_data/stokes_static_pipeline.py",
    ROOT / "solver/gen_data/pipeline/reference.py",
    ROOT / "solver/gen_data/pipeline/writer.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configuration_record() -> dict[str, object]:
    """Return the exact numerical and source configuration for this pilot."""

    return {
        "schema": "static_stokes_exact_target_pilot_configuration_v1",
        "purpose": "exact paper-target software and numerical pilot",
        "contract": PAPER_STATIC_STOKES_CONTRACT.to_json_record(),
        "target": asdict(PAPER_DNO_TARGET),
        "family_id": int(PhysicalFamilyId.STOKES),
        "revision_id": PAPER_CORPUS_REVISION_ID,
        "split": SplitId.VALIDATION.value,
        "stream_id": 0,
        "cells": [cell.cell_id for cell in STOKES_POPULATION_CELLS],
        "source_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in HASHED_SOURCES
        },
    }


def configuration_fingerprint(configuration: dict[str, object]) -> str:
    """Return the SHA-256 hash of one strict canonical JSON configuration."""

    encoded = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(
            f"{output_dir} already exists; choose a new output directory"
        )

    assignments = tuple(
        AttemptAssignment(
            case_key=CaseKey(
                family_id=PhysicalFamilyId.STOKES,
                revision_id=PAPER_CORPUS_REVISION_ID,
                split_id=SplitId.VALIDATION,
                stream_id=0,
                attempt_index=index,
            ),
            cell_id=cell.cell_id,
        )
        for index, cell in enumerate(STOKES_POPULATION_CELLS)
    )
    configuration = configuration_record()
    fingerprint = configuration_fingerprint(configuration)

    started = perf_counter()
    samples = tuple(sample_stokes_population(item) for item in assignments)
    sampling_seconds = perf_counter() - started

    started = perf_counter()
    result = write_static_stokes_batch(
        output_dir,
        samples,
        batch_id=0,
        config_fingerprint=fingerprint,
        metadata={
            "configuration": configuration,
            "configuration_fingerprint": fingerprint,
        },
    )
    execution_seconds = perf_counter() - started

    started = perf_counter()
    view = build_dataset_view(
        output_dir,
        (result.paths,),
        name="static_stokes_exact_target_pilot",
        expected_fingerprint=fingerprint,
    )
    view_seconds = perf_counter() - started

    accepted = sum(outcome.decision.accepted for outcome in result.outcomes)
    summary = {
        "schema": "static_stokes_exact_target_pilot_summary_v1",
        "configuration": configuration,
        "configuration_fingerprint": fingerprint,
        "attempted_cases": len(result.outcomes),
        "accepted_cases": accepted,
        "rejected_cases": len(result.outcomes) - accepted,
        "case_records": [
            {
                "cell_id": sample.cell.cell_id,
                "case_id": sample.assignment.case_key.case_id,
                "accepted": outcome.decision.accepted,
                "required_bits": outcome.decision.required_bits,
                "evaluated_bits": outcome.decision.evaluated_bits,
                "failed_bits": outcome.decision.failed_bits,
                "support_resampling_count": sample.support_resampling_count,
                "metrics": dict(outcome.metrics),
            }
            for sample, outcome in zip(samples, result.outcomes)
        ],
        "timing_seconds": {
            "sampling": sampling_seconds,
            "construction_target_and_commit": execution_seconds,
            "dataset_view": view_seconds,
            "total": sampling_seconds + execution_seconds + view_seconds,
        },
        "artifacts": {
            "proposal": str(result.paths.proposal.relative_to(output_dir)),
            "proposal_sha256": file_sha256(result.paths.proposal),
            "shard": str(result.paths.shard.relative_to(output_dir)),
            "shard_sha256": file_sha256(result.paths.shard),
            "result": str(result.paths.result.relative_to(output_dir)),
            "result_sha256": file_sha256(result.paths.result),
            "manifest": str(view.manifest.relative_to(output_dir)),
            "manifest_sha256": file_sha256(view.manifest),
            "trajectory_map": str(view.trajectory_map.relative_to(output_dir)),
            "trajectory_map_sha256": file_sha256(view.trajectory_map),
        },
    }
    summary_path = output_dir / "summary.json"
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
