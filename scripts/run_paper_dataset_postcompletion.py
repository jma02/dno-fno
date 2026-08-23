"""Run final, read-only paper-dataset diagnostics after generation completes.

This process is deliberately separate from the live JONSWAP/Tanaka supervisor.
It waits for the immutable final combined view and all family audit records,
authenticates their common source population, and then runs four CPU-only
post-processing stages.  Outputs live below a directory named by the complete
SHA-256 digest of the combined summary, so a rebuilt view cannot silently reuse
diagnostics from a different release.

The family audits and combined dataset are release evidence.  This runner never
writes to them.  Its stages are diagnostics: a stage failure is recorded and
causes a nonzero exit, but does not alter dataset acceptance or release status.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform as python_platform
import re
import signal
import subprocess
import tempfile
import time
from typing import Any, BinaryIO, Final
import uuid

import jax
import jaxlib
import numpy as np
from PIL import Image

if __package__:
    from scripts import build_paper_dataset_view as combined_view_builder
else:  # Direct ``python scripts/<name>.py`` entrypoint.
    import build_paper_dataset_view as combined_view_builder
from solver.gen_data.pipeline import manifest as dataset_manifest_builder


ROOT = Path(__file__).resolve().parents[1]
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

RUNNER_SCHEMA = "paper_dataset_postcompletion_runner_v1"
DOCUMENT_HANDOFF_NAME: Final = "paper_dataset_document_handoff.json"
RELEASE_IDENTITY_SCHEMA = "paper_dataset_postcompletion_release_identity_v2"
COMBINED_SUMMARY_SCHEMA = "paper_dataset_combined_view_summary_v1"
COMBINED_PREFLIGHT_SCHEMA = "paper_dataset_combined_view_preflight_v1"
DEPENDENCY_BINDING_SCHEMA = "paper_dataset_postcompletion_dependency_binding_v1"
GENERATION_SOURCE_BINDING_SCHEMA = (
    "paper_dataset_postcompletion_generation_source_binding_v1"
)
RENDERER_SCHEMA = "paper_dataset_all_family_diagnostic_tail_v3"
RENDERER_IMPLEMENTATION_SCHEMA = "paper_dataset_renderer_implementation_v1"
ORDER_SCHEMA = "paper_dataset_jonswap_revision4_order_convergence_v1"
ORDER_IMPLEMENTATION_SCHEMA = "paper_dataset_order_diagnostic_implementation_v1"
TRAINING_SCHEMA = "paper_dataset_training_handoff_audit_v1"
TRAINING_IMPLEMENTATION_SCHEMA = "paper_dataset_training_implementation_v1"
FIGURE_SCHEMA = "paper_dataset_deterministic_family_illustrations_v1"
FIGURE_IMPLEMENTATION_SCHEMA = "paper_dataset_family_figure_implementation_v1"

TRAINING_IMPLEMENTATION_FILES: Final = (
    (
        "scripts/audit_paper_dataset_training_handoff.py",
        "independent_cpu_streaming_handoff_audit",
    ),
    (
        "train-jax-10m/util.py",
        "trainer_dataset_loading_split_selection_and_normalization",
    ),
    (
        "train-jax-10m/1d_dno_fno_jax.py",
        "canonical_trainer_entrypoint_for_handoff_command_template",
    ),
    (
        "scripts/test_audit_paper_dataset_training_handoff.py",
        "semantic_regression_evidence",
    ),
)
TRAINING_IMPLEMENTATION_RELATIONSHIP: Final = {
    "audit_strategy": "independent_numpy_cpu_streaming_reimplementation",
    "trainer_helpers_compared_by_regression_test": [
        "load_dataset_arrays",
        "build_dataset_split_indices",
        "load_or_compute_stats",
    ],
    "regression_test_path": "scripts/test_audit_paper_dataset_training_handoff.py",
    "claim_scope": (
        "The audit independently reconstructs the selected training rows and "
        "normalization statistics, and the named regression test compares those "
        "results with the trainer helpers. This is evidence for the tested "
        "dataset-handoff and statistics semantics plus the canonical command "
        "entrypoint bytes, not a proof that the full trainer is equivalent; it "
        "does not authenticate the model or transitive training stack."
    ),
    "formal_equivalence_claim": False,
}

FAMILY_ORDER: Final = (
    "stokes",
    "tanaka",
    "benjamin_feir",
    "jonswap_tma",
)
SPLIT_ORDER: Final = ("train", "validation", "test")
REVISION_BY_FAMILY: Final = {
    "stokes": 2,
    "tanaka": 3,
    "benjamin_feir": 4,
    "jonswap_tma": 4,
}
ROWS_PER_CASE: Final = {
    "stokes": 1,
    "tanaka": 200,
    "benjamin_feir": 200,
    "jonswap_tma": 16,
}
CASES_PER_FAMILY_BY_SPLIT: Final = {
    "train": 16_384,
    "validation": 1_024,
    "test": 1_024,
}
SOURCE_COUNT_BY_FAMILY: Final = {
    "stokes": 6,
    "tanaka": 6,
    "benjamin_feir": 6,
    "jonswap_tma": 8,
}
FINAL_SOURCE_COUNT = 26
FINAL_ACCEPTED_CASES = 73_728
FINAL_RETAINED_ROWS = 7_686_144
RENDERER_TOP_COUNT = 6
RENDERER_RANKINGS: Final = (
    "combined",
    "eta_slope",
    "gxi_high_band",
    "gxi_sign_changes",
    "stored_band_quadratic_energy_drift",
)
CENTRAL_VALIDATION_CATEGORY: Final = {
    "stokes": "finite_moderate",
    "tanaka": "main_m2_q1",
    "benjamin_feir": "n_c_09__delta_n_02",
    "jonswap_tma": "finite__gamma_3p3__right_0p5",
}
EXPECTED_SPLIT_COUNTS = {
    "train": 16_384,
    "validation": 1_024,
    "test": 1_024,
}
EXPECTED_ROWS_BY_FAMILY = {
    family: sum(CASES_PER_FAMILY_BY_SPLIT.values()) * rows
    for family, rows in ROWS_PER_CASE.items()
}
EXPECTED_STOKES_LEGACY_SHA256 = (
    "d59d40f819a7b05f01eac90a31e74121f2ee570641d42e3cf6850b756af136aa"
)
EXPECTED_STOKES_BINDING_SHA256 = (
    "d2223fddeedebd541ecdb19752ca547bbd0325ef886c6d97de821b24935dd4f5"
)
STOKES_EXPECTED_SOURCE_FINGERPRINT: Final = (
    "cac7a23529ca12c934770ac0a5c4f6c7ff094a22d2cdd5e59db5ba0d00098ae2"
)
STOKES_HISTORICAL_SOURCE_SNAPSHOT_ROOT: Final = (
    ROOT / "reproducibility/source_snapshots/stokes_revision2_60a28ff"
)
STOKES_HISTORICAL_SOURCE_COMMIT: Final = "60a28ffae394465c6ea295eb4ed6c075fbc756a4"
STOKES_HISTORICAL_SOURCE_SNAPSHOT_MANIFEST: Final = {
    "name": "SHA256SUMS",
    "bytes": 251,
    "sha256": "a251ed4369184a7b8fed7fb6ce7315ba5da1c5bb5fc0e283b19b2c520827ca78",
}
STOKES_HISTORICAL_SOURCE_SNAPSHOTS: Final = {
    "scripts/run_paper_dataset_quota.py": {
        "name": "run_paper_dataset_quota.py",
        "bytes": 33_974,
        "sha256": "ab99c067c2f22823fce861a69fd03bd8cc4524effbfd5ab1a5360bfa5e615d98",
    },
    "solver/gen_data/pipeline/manifest.py": {
        "name": "manifest.py",
        "bytes": 28_988,
        "sha256": "e2cc1a20cc01fef9dd0bb84b99485b5cf90da71f7ee035a813893f2f6a37f421",
    },
    "solver/gen_data/pipeline/production.py": {
        "name": "production.py",
        "bytes": 7_854,
        "sha256": "8ed36cf1bd57494f344096e4a508bcfb37b9103cf03ace8a63df2656c11dc735",
    },
}
STOKES_EXPECTED_GENERATION_SOURCE_PATHS: Final = frozenset(
    {
        "scripts/run_paper_dataset_quota.py",
        "solver/reference_solutions/stokes_wave.py",
        "solver/gen_data/pipeline/archive.py",
        "solver/gen_data/pipeline/manifest.py",
        "solver/gen_data/pipeline/production.py",
        "solver/gen_data/pipeline/quality.py",
        "solver/gen_data/pipeline/quota_driver.py",
        "solver/gen_data/pipeline/reference.py",
        "solver/gen_data/pipeline/writer.py",
        "solver/gen_data/stokes_sampling.py",
        "solver/gen_data/stokes_quota_executor.py",
        "solver/gen_data/stokes_static_pipeline.py",
        "solver/solvers/dno_series_jax.py",
    }
)
BF_HISTORICAL_SOURCE_SNAPSHOT_ROOT: Final = (
    ROOT / "reproducibility/source_snapshots/benjamin_feir_revision4_e16773f"
)
BF_HISTORICAL_SOURCE_SNAPSHOT_MANIFEST: Final = {
    "name": "SHA256SUMS",
    "bytes": 176,
    "sha256": "29c92a3d0ca8e534b24ad7cfacf6f5e7671a0a5a4b4a236e4523719ae14abf1e",
}
BF_HISTORICAL_SOURCE_SNAPSHOTS: Final = {
    "solver/gen_data/pipeline/production.py": {
        "name": "production.py",
        "bytes": 8_494,
        "sha256": "2c2234caf1087c2982872eb44ba2abd457cf22b812784e1a3e9bd54e8c7129d4",
    },
    "solver/gen_data/trajectory_family_adapters.py": {
        "name": "trajectory_family_adapters.py",
        "bytes": 28_852,
        "sha256": "afb480a64b14a2b311bda067e6638208569cf3acfebc61d67a1f016a73fbef6a",
    },
}
BF_EXPECTED_GENERATION_SOURCE_PATHS: Final = frozenset(
    {
        "scripts/run_paper_dataset_quota.py",
        "solver/reference_solutions/stokes_wave.py",
        "solver/gen_data/benjamin_feir_jcp09.py",
        "solver/gen_data/benjamin_feir_sampling.py",
        "solver/gen_data/pipeline/acceptance.py",
        "solver/gen_data/pipeline/archive.py",
        "solver/gen_data/pipeline/manifest.py",
        "solver/gen_data/pipeline/production.py",
        "solver/gen_data/pipeline/quality.py",
        "solver/gen_data/pipeline/quota_driver.py",
        "solver/gen_data/pipeline/reference.py",
        "solver/gen_data/pipeline/refinement.py",
        "solver/gen_data/pipeline/time_selection.py",
        "solver/gen_data/pipeline/trajectory_writer.py",
        "solver/gen_data/pipeline/writer.py",
        "solver/gen_data/trajectory_family_adapters.py",
        "solver/gen_data/trajectory_quota_executor.py",
        "solver/solvers/dno_series_jax.py",
        "solver/solvers/time_integrator.py",
    }
)
TANAKA_EXPECTED_GENERATION_SOURCE_PATHS: Final = frozenset(
    {
        "scripts/run_paper_dataset_quota.py",
        "solver/gen_data/multi_crest.py",
        "solver/gen_data/pipeline/acceptance.py",
        "solver/gen_data/pipeline/archive.py",
        "solver/gen_data/pipeline/manifest.py",
        "solver/gen_data/pipeline/production.py",
        "solver/gen_data/pipeline/quality.py",
        "solver/gen_data/pipeline/quota_driver.py",
        "solver/gen_data/pipeline/reference.py",
        "solver/gen_data/pipeline/refinement.py",
        "solver/gen_data/pipeline/time_selection.py",
        "solver/gen_data/pipeline/trajectory_writer.py",
        "solver/gen_data/pipeline/writer.py",
        "solver/gen_data/tanaka_sampling.py",
        "solver/gen_data/tanaka_initial_conditions.py",
        "solver/gen_data/trajectory_family_adapters.py",
        "solver/gen_data/trajectory_quota_executor.py",
        "solver/solvers/dno_series_jax.py",
        "solver/solvers/time_integrator.py",
        "solver/tanaka_ICs/modified_tanaka.py",
    }
)
JONSWAP_EXPECTED_GENERATION_SOURCE_PATHS: Final = frozenset(
    {
        "scripts/run_paper_dataset_jonswap_bucketed.py",
        "scripts/run_paper_dataset_quota.py",
        "solver/gen_data/jonswap_horizon_executor.py",
        "solver/gen_data/jonswap_tma.py",
        "solver/gen_data/jonswap_tma_sampling.py",
        "solver/gen_data/pipeline/acceptance.py",
        "solver/gen_data/pipeline/archive.py",
        "solver/gen_data/pipeline/manifest.py",
        "solver/gen_data/pipeline/production.py",
        "solver/gen_data/pipeline/quality.py",
        "solver/gen_data/pipeline/quota_driver.py",
        "solver/gen_data/pipeline/reference.py",
        "solver/gen_data/pipeline/refinement.py",
        "solver/gen_data/pipeline/time_selection.py",
        "solver/gen_data/pipeline/trajectory_writer.py",
        "solver/gen_data/pipeline/writer.py",
        "solver/gen_data/trajectory_family_adapters.py",
        "solver/gen_data/trajectory_quota_executor.py",
        "solver/solvers/dno_series_jax.py",
        "solver/solvers/time_integrator.py",
    }
)
EXPECTED_GENERATION_SOURCE_PATHS_BY_FAMILY: Final = {
    "stokes": STOKES_EXPECTED_GENERATION_SOURCE_PATHS,
    "tanaka": TANAKA_EXPECTED_GENERATION_SOURCE_PATHS,
    "benjamin_feir": BF_EXPECTED_GENERATION_SOURCE_PATHS,
    "jonswap_tma": JONSWAP_EXPECTED_GENERATION_SOURCE_PATHS,
}
HISTORICAL_GENERATION_SOURCE_PATHS_BY_FAMILY: Final = {
    "stokes": frozenset(STOKES_HISTORICAL_SOURCE_SNAPSHOTS),
    "tanaka": frozenset(),
    "benjamin_feir": frozenset(BF_HISTORICAL_SOURCE_SNAPSHOTS),
    "jonswap_tma": frozenset(),
}
CURRENT_GENERATION_SOURCE_PATHS: Final = frozenset().union(
    *(
        EXPECTED_GENERATION_SOURCE_PATHS_BY_FAMILY[family]
        - HISTORICAL_GENERATION_SOURCE_PATHS_BY_FAMILY[family]
        for family in FAMILY_ORDER
    )
)
DEPENDENCY_FILE_PATHS: Final = ("pyproject.toml", "uv.lock")

EXPECTED_AUDIT_CHECKS: Final[dict[str, dict[str, object]]] = {
    "stokes": {
        "legacy_numerical_audit_authenticated": True,
        "current_chunk_loader_passed": True,
        "exact_six_chunk_layout": True,
        "exact_split_counts": True,
        "summary_cell_counts_match_completed_quota_state": True,
        "manifest_and_trajectory_map_hashes_bound": True,
        "common_source_execution_and_dependency_identity": True,
        "exact_equal_13_path_source_maps": True,
        "source_map_fingerprint_verified": True,
        "historical_generation_source_snapshots_bound_to_all_chunks": True,
        "all_nonhistorical_generation_sources_match_current_bytes": True,
        "no_rejected_or_unowned_rows": True,
    },
    "benjamin_feir": {
        "exact_six_chunk_plan": True,
        "nested_train_intervals": True,
        "balanced_66_cell_quotas": True,
        "summary_cell_counts_match_reconstructed_transactions": True,
        "historical_shared_source_snapshots_bound_to_all_chunks": True,
        "all_nonhistorical_generation_sources_match_current_bytes": True,
        "exact_current_66_cell_taxonomy_verified": True,
        "immutable_specs_rebuilt": True,
        "all_transactions_rescanned": True,
        "proposal_hashes_verified": True,
        "result_hashes_verified": True,
        "shard_hashes_verified": True,
        "manifest_hashes_verified": True,
        "trajectory_map_hashes_verified": True,
        "accepted_rejected_row_ownership_verified": True,
        "canonical_shard_schema_dtypes_shapes_verified": True,
        "accepted_cases_have_exactly_200_uniform_rows": True,
        "rejected_cases_have_zero_rows": True,
        "stored_eta_xi_gxi_depth_time_finite": True,
        "stored_depths_equal_proposals": True,
        "stored_water_columns_positive": True,
        "stored_times_include_zero_and_realized_endpoint": True,
        "floored_100_carrier_period_horizons_verified": True,
        "all_quality_masks_current_and_consistent": True,
        "accepted_residuals_within_contract": True,
        "accepted_hamiltonian_drift_within_contract": True,
        "accepted_water_columns_positive": True,
        "all_proposals_in_current_bf_support": True,
        "current_bf_specs_replayed_exactly": True,
        "source_dependency_execution_identity_verified": True,
        "pending_batches": 0,
        "terminal_failures": 0,
        "attempt_limit_failures": 0,
    },
    "jonswap_tma": {
        "exact_eight_chunk_plan": True,
        "nested_train_intervals": True,
        "balanced_27_cell_quotas": True,
        "immutable_parameter_and_phase_specs_rebuilt": True,
        "all_transactions_rescanned": True,
        "proposal_hashes_verified": True,
        "result_hashes_verified": True,
        "shard_hashes_verified": True,
        "manifest_hashes_verified": True,
        "trajectory_map_hashes_verified": True,
        "accepted_rejected_row_ownership_verified": True,
        "all_stored_float_fields_finite": True,
        "all_quality_masks_current_and_consistent": True,
        "accepted_adjustment_and_autonomous_residuals_within_contract": True,
        "accepted_autonomous_hamiltonian_drift_within_contract": True,
        "accepted_adjustment_and_autonomous_water_columns_positive": True,
        "complete_adjustment_and_autonomous_horizons_verified": True,
        "all_proposals_in_current_relative_frequency_support": True,
        "current_jonswap_specs_and_phases_replayed_exactly": True,
        "source_dependency_execution_bucketing_support_identity_verified": True,
        "all_attempts_retained_exactly_once": True,
        "population_conditioning_semantics_and_27_cell_counts_verified": True,
        "pending_batches": 0,
        "terminal_failures": 0,
        "attempt_limit_failures": 0,
    },
    "tanaka": {
        "exact_six_chunk_plan": True,
        "exact_four_train_intervals_plus_validation_and_test": True,
        "balanced_eleven_direction_aware_cell_quotas": True,
        "summary_cell_counts_match_reconstructed_transactions": True,
        "immutable_specs_rebuilt": True,
        "all_requested_crests_replayed_exactly": True,
        "corrected_amplitude_inversion_identity_verified": True,
        "constructor_amplitude_success_postcondition_source_bound": True,
        "below_old_floor_amplitude_behavior_verified": True,
        "solver_amplitude_validator_called_exactly_once": True,
        "amplitude_validator_deliberate_mismatch_rejected": True,
        "achieved_per_crest_amplitudes_persisted": False,
        "all_transactions_rescanned": True,
        "proposal_hashes_verified": True,
        "result_hashes_verified": True,
        "shard_hashes_verified": True,
        "manifest_hashes_verified": True,
        "trajectory_map_hashes_verified": True,
        "accepted_rejected_row_ownership_verified": True,
        "accepted_cases_have_exactly_200_rows": True,
        "rejected_cases_have_zero_rows": True,
        "stored_eta_xi_gxi_depth_time_finite": True,
        "stored_water_columns_positive": True,
        "all_quality_masks_current_and_consistent": True,
        "accepted_residuals_at_most_1e-8": True,
        "accepted_trajectories_complete_through_t200": True,
        "all_proposals_in_current_tanaka_support": True,
        "source_dependency_execution_support_identity_verified": True,
        "historical_generation_compatibility_used": False,
        "pending_batches": 0,
        "terminal_failures": 0,
        "attempt_limit_failures": 0,
    },
}

STANDARD_TRAIN_LAYOUT: Final = (
    (0, 2_048, 0),
    (2_048, 2_048, 1),
    (4_096, 4_096, 2),
    (8_192, 8_192, 3),
)
JONSWAP_TRAIN_LAYOUT: Final = (
    (0, 2_048, 0),
    (2_048, 2_048, 1),
    (4_096, 4_096, 2),
    (8_192, 4_096, 3),
    (12_288, 2_048, 4),
    (14_336, 2_048, 5),
)


def expected_layout(family: str) -> tuple[tuple[str, int, int, int], ...]:
    """Return the exact additive source intervals for one final family."""

    train = JONSWAP_TRAIN_LAYOUT if family == "jonswap_tma" else STANDARD_TRAIN_LAYOUT
    return tuple(("train", *values) for values in train) + (
        ("validation", 0, 1_024, 100),
        ("test", 0, 1_024, 200),
    )


@dataclass(frozen=True)
class Artifact:
    """Authenticated immutable file identity."""

    path: Path
    bytes: int
    sha256: str

    def record(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class AuditPaths:
    """The five records constituting the four family audits."""

    stokes_binding: Path
    stokes_legacy: Path
    benjamin_feir: Path
    jonswap_tma: Path
    tanaka: Path

    def all(self) -> tuple[Path, ...]:
        return (
            self.stokes_binding,
            self.stokes_legacy,
            self.benjamin_feir,
            self.jonswap_tma,
            self.tanaka,
        )


@dataclass(frozen=True)
class CombinedBinding:
    """Authenticated lightweight identity of the exact final combined view."""

    summary: Artifact
    manifest: Artifact
    trajectory_map: Artifact
    chunks: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class DependencyBinding:
    """Exact generation dependency record and its physical repository files."""

    environment: Mapping[str, object]
    fingerprint: str
    files: Mapping[str, Artifact]
    chunk_count: int

    def record(self) -> dict[str, object]:
        return {
            "schema": DEPENDENCY_BINDING_SCHEMA,
            "chunk_count": self.chunk_count,
            "environment": dict(self.environment),
            "fingerprint": self.fingerprint,
            "files": {
                name: artifact.record() for name, artifact in sorted(self.files.items())
            },
        }


@dataclass(frozen=True)
class GenerationSourceBinding:
    """One family's exact generation source map and current physical files."""

    source_sha256: Mapping[str, str]
    fingerprint: str
    current_sources: Mapping[str, Artifact]
    historical_source_paths: tuple[str, ...]
    chunk_count: int

    def record(self, *, family: str) -> dict[str, object]:
        return {
            "revision_id": REVISION_BY_FAMILY[family],
            "chunk_source_maps_checked": self.chunk_count,
            "source_count": len(self.source_sha256),
            "source_map_fingerprint": self.fingerprint,
            "source_sha256": dict(sorted(self.source_sha256.items())),
            "historical_snapshot_source_paths": list(self.historical_source_paths),
            "current_repository_source_count": len(self.current_sources),
            "current_repository_sources": {
                name: artifact.record()
                for name, artifact in sorted(self.current_sources.items())
            },
        }


@dataclass(frozen=True)
class ReleaseBinding:
    """Combined-view and family-audit identities for one post-processing run."""

    combined: CombinedBinding
    audits: Mapping[str, Artifact]
    dependency: DependencyBinding
    generation_sources: Mapping[str, GenerationSourceBinding]
    historical_source_snapshots: Mapping[str, Mapping[str, Artifact]] = field(
        default_factory=dict
    )

    def identity_record(self) -> dict[str, object]:
        combined_record = _strict_json(self.combined.summary.path)
        audit_records = {
            name: _strict_json(artifact.path) for name, artifact in self.audits.items()
        }
        source_cross_binding = [
            {
                "family": chunk["family"],
                "split": chunk["split"],
                "accepted_before": chunk["accepted_before"],
                "accepted_count": chunk["accepted_count"],
                "accepted_after": chunk["accepted_after"],
                "attempted_count": chunk["attempted_count"],
                "stream_id": chunk["stream_id"],
                "configuration_fingerprint": chunk["configuration_fingerprint"],
                "summary_path": chunk["summary_path"],
                "summary_sha256": chunk["summary_sha256"],
            }
            for chunk in self.combined.chunks
        ]
        return {
            "schema": RELEASE_IDENTITY_SCHEMA,
            "combined_summary": {
                **self.combined.summary.record(),
                "schema": combined_record.get("schema"),
                "status": combined_record.get("status"),
            },
            "combined_manifest": self.combined.manifest.record(),
            "combined_trajectory_map": self.combined.trajectory_map.record(),
            "generation_dependency": self.dependency.record(),
            "generation_sources": {
                "schema": GENERATION_SOURCE_BINDING_SCHEMA,
                "family_count": len(self.generation_sources),
                "families": {
                    family: binding.record(family=family)
                    for family, binding in sorted(self.generation_sources.items())
                },
            },
            "family_audits": {
                name: {
                    **artifact.record(),
                    "schema": audit_records[name].get("schema"),
                    "status": audit_records[name].get("status"),
                }
                for name, artifact in sorted(self.audits.items())
            },
            "historical_source_snapshots": {
                "schema": (
                    "paper_dataset_postcompletion_historical_source_snapshots_v2"
                ),
                "role": "inert_historical_byte_recovery_only",
                "family_count": len(self.historical_source_snapshots),
                "artifact_count": sum(
                    len(artifacts)
                    for artifacts in self.historical_source_snapshots.values()
                ),
                "families": {
                    family: {
                        "revision_id": REVISION_BY_FAMILY[family],
                        "artifact_count": len(artifacts),
                        "artifacts": {
                            name: artifact.record()
                            for name, artifact in sorted(artifacts.items())
                        },
                    }
                    for family, artifacts in sorted(
                        self.historical_source_snapshots.items()
                    )
                },
            },
            "source_cross_binding": {
                "verified": True,
                "source_count": len(source_cross_binding),
                "definition": (
                    "Every family-audit chunk summary path/hash equals the exact "
                    "combined-preflight source; each authenticated source summary "
                    "in turn names the same manifest/map path/hash as that audit."
                ),
                "chunks": source_cross_binding,
                "fingerprint": _canonical_sha256(source_cross_binding),
            },
            "contract": {
                "families": list(FAMILY_ORDER),
                "revisions": dict(REVISION_BY_FAMILY),
                "sources": FINAL_SOURCE_COUNT,
                "accepted_cases": FINAL_ACCEPTED_CASES,
                "retained_rows": FINAL_RETAINED_ROWS,
                "accepted_cases_per_family_by_split": dict(CASES_PER_FAMILY_BY_SPLIT),
                "rows_per_case": dict(ROWS_PER_CASE),
            },
        }


@dataclass(frozen=True)
class StageSpec:
    """One bounded post-completion subprocess and its committed outputs."""

    name: str
    command: tuple[str, ...]
    timeout_seconds: float
    outputs: tuple[Path, ...]
    validate: Callable[[], Mapping[str, object]]
    requires: tuple[str, ...] = ()
    completion_marker: Path | None = None
    producer_owns_uncommitted_outputs: bool = False

    def __post_init__(self) -> None:
        """Require explicit ownership for marker-based partial-set recovery."""

        if self.producer_owns_uncommitted_outputs:
            if self.completion_marker is None:
                raise ValueError(
                    "producer-owned uncommitted outputs require a completion marker"
                )
            if self.completion_marker not in self.outputs:
                raise ValueError("completion marker must be one of the stage outputs")
        elif self.completion_marker is not None:
            raise ValueError(
                "completion marker requires producer-owned uncommitted outputs"
            )


@dataclass(frozen=True)
class RunnerConfig:
    """Runtime configuration for the restartable post-completion process."""

    combined_summary: Path
    audits: AuditPaths
    state_root: Path
    python: Path
    poll_seconds: float = 60.0
    wait_timeout_seconds: float | None = None
    stage_attempts: int = 2
    retry_delay_seconds: float = 10.0
    kill_after_seconds: float = 120.0
    renderer_timeout_seconds: float = 7_200.0
    order_timeout_seconds: float = 3_600.0
    training_timeout_seconds: float = 7_200.0
    figure_timeout_seconds: float = 3_600.0
    renderer_workers: int = 4


class ExistingOutputError(RuntimeError):
    """A stage output exists but cannot be authenticated for this release."""


def _strict_json(path: Path) -> dict[str, Any]:
    """Read a finite JSON object."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"{path} contains nonfinite JSON constant {value!r}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, *, expected_sha256: object | None = None) -> Artifact:
    requested = path.expanduser()
    if requested.is_symlink():
        raise ValueError(f"artifact must not be a symbolic link: {requested}")
    resolved = requested.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"artifact must be a regular non-symlink file: {resolved}")
    observed_sha256 = _sha256(resolved)
    if expected_sha256 is not None:
        expected = _digest(expected_sha256, context=f"{resolved} SHA-256")
        if observed_sha256 != expected:
            raise RuntimeError(f"artifact SHA-256 differs: {resolved}")
    return Artifact(resolved, resolved.stat().st_size, observed_sha256)


def _digest(value: object, *, context: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _integer(value: object, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _sequence(value: object, *, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be a JSON array")
    return value


def _same_json(left: object, right: object) -> bool:
    def canonical(value: object) -> str:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    return canonical(left) == canonical(right)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_record_path(base: Path, value: object, *, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} path must be a nonempty string")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError(f"{context} escapes its declared root")
    return resolved


def _require_real_directory(path: Path, *, context: str) -> Path:
    """Reject every symlink component before resolving a source root."""

    requested = path.expanduser()
    if ".." in requested.parts:
        raise ValueError(f"{context} must not contain parent traversal")
    absolute = requested if requested.is_absolute() else Path.cwd() / requested
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{context} contains a symbolic link: {current}")
    if not absolute.is_dir():
        raise ValueError(f"{context} is not a real directory")
    return absolute.resolve(strict=True)


def _source_file_inside(root: Path, relative: str, *, context: str) -> Path:
    """Resolve one strict relative source without following path symlinks."""

    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{context} escapes its declared root")
    current = root
    for component in relative_path.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{context} contains a symbolic link")
    if not current.is_file():
        raise ValueError(f"{context} is not a regular file")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{context} escapes its declared root")
    return resolved


def _artifact_from_record(
    base: Path,
    value: object,
    *,
    context: str,
) -> Artifact:
    record = _mapping(value, context=context)
    path = _resolve_record_path(base, record.get("path"), context=context)
    artifact = _artifact(path, expected_sha256=record.get("sha256"))
    if artifact.bytes != _integer(record.get("bytes"), context=f"{context} bytes"):
        raise RuntimeError(f"{context} byte count differs")
    return artifact


def _expected_rows_by_split_and_family() -> dict[str, dict[str, int]]:
    return {
        split: {family: count * ROWS_PER_CASE[family] for family in FAMILY_ORDER}
        for split, count in CASES_PER_FAMILY_BY_SPLIT.items()
    }


def _reconstruct_dataset_contract(
    plan: combined_view_builder.CombinedDatasetPlan,
) -> dict[str, object]:
    """Recover the canonical contract directly from immutable proposals."""

    policy = (
        combined_view_builder.PAPER_GENERATION_COMPATIBILITY_POLICY
        if any(chunk.generation_compatibility_id is not None for chunk in plan.chunks)
        else None
    )
    shared_target: Mapping[str, object] | None = None
    family_execution: dict[tuple[int, int], Mapping[str, object]] = {}
    family_numerical: dict[tuple[int, int], Mapping[str, object]] = {}
    dependency_environment: Mapping[str, object] | None = None
    family_generation: dict[tuple[int, int], Any] = {}
    family_compatibility_ids: dict[tuple[int, int], str | None] = {}
    family_variants: dict[tuple[int, int], dict[tuple[str, str], Any]] = {}

    for chunk in plan.chunks:
        for paths in chunk.batches:
            with np.load(paths.proposal, allow_pickle=False) as stored:
                proposal = {name: np.asarray(stored[name]) for name in stored.files}
            contract = dataset_manifest_builder._batch_contract(proposal)
            generation = dataset_manifest_builder._generation_compatibility(proposal)
            if contract is None or generation is None:
                raise RuntimeError(
                    "final combined proposal omits its dataset/generation contract"
                )
            family_id = int(np.asarray(proposal["family_id"]).item())
            revision_id = int(np.asarray(proposal["revision_id"]).item())
            family_key = (family_id, revision_id)
            resolution = (
                policy.resolve(
                    family_id=family_id,
                    revision_id=revision_id,
                    execution_record=contract.family_execution,
                    source_sha256=generation.source_sha256,
                )
                if policy is not None
                and policy.applies_to(
                    family_id=family_id,
                    revision_id=revision_id,
                )
                else None
            )
            compatibility_id = (
                resolution.compatibility_id if resolution is not None else None
            )
            if compatibility_id != chunk.generation_compatibility_id:
                raise RuntimeError(
                    "combined plan/proposal generation compatibility differs"
                )
            effective_execution = (
                resolution.canonical_execution_record
                if resolution is not None
                else contract.family_execution
            )
            if shared_target is None:
                shared_target = contract.target
            elif not _same_json(shared_target, contract.target):
                raise RuntimeError("combined proposal targets differ")
            previous_execution = family_execution.setdefault(
                family_key,
                effective_execution,
            )
            if not _same_json(previous_execution, effective_execution):
                raise RuntimeError("combined family execution contracts differ")
            if contract.trajectory_numerical is not None:
                previous_numerical = family_numerical.setdefault(
                    family_key,
                    contract.trajectory_numerical,
                )
                if not _same_json(
                    previous_numerical,
                    contract.trajectory_numerical,
                ):
                    raise RuntimeError("combined family numerical contracts differ")
            if dependency_environment is None:
                dependency_environment = generation.dependency_environment
            elif not _same_json(
                dependency_environment,
                generation.dependency_environment,
            ):
                raise RuntimeError("combined proposal dependencies differ")
            previous_generation = family_generation.setdefault(
                family_key,
                generation,
            )
            previous_compatibility_id = family_compatibility_ids.setdefault(
                family_key,
                compatibility_id,
            )
            if previous_compatibility_id != compatibility_id:
                raise RuntimeError("combined family compatibility IDs differ")
            if (
                previous_generation.source_sha256 != generation.source_sha256
                and compatibility_id is None
            ):
                raise RuntimeError("combined family source maps differ")
            if previous_generation.execution_platform != generation.execution_platform:
                raise RuntimeError("combined family execution platforms differ")
            execution_fingerprint = _canonical_sha256(contract.family_execution)
            source_fingerprint = _canonical_sha256(generation.source_sha256)
            family_variants.setdefault(family_key, {})[
                (execution_fingerprint, source_fingerprint)
            ] = generation

    if shared_target is None or dependency_environment is None:
        raise RuntimeError("final combined plan has no reconstructable contract")
    numerical_records = [
        {
            "family_id": family_id,
            "revision_id": revision_id,
            "numerical": dict(numerical),
        }
        for (family_id, revision_id), numerical in sorted(family_numerical.items())
    ]
    numerical_values = tuple(family_numerical.values())
    common_numerical = (
        dict(numerical_values[0])
        if numerical_values
        and all(
            _same_json(numerical, numerical_values[0])
            for numerical in numerical_values[1:]
        )
        else None
    )
    family_revision_records: list[dict[str, object]] = []
    for (family_id, revision_id), generation in sorted(family_generation.items()):
        variants = [
            {
                "execution_record_fingerprint": execution_fingerprint,
                "source_sha256_fingerprint": source_fingerprint,
                "source_sha256": dict(sorted(raw_generation.source_sha256.items())),
            }
            for (
                execution_fingerprint,
                source_fingerprint,
            ), raw_generation in sorted(
                family_variants[(family_id, revision_id)].items()
            )
        ]
        record: dict[str, object] = {
            "family_id": family_id,
            "revision_id": revision_id,
            "execution_platform": generation.execution_platform,
            "generation_variants": variants,
        }
        compatibility_id = family_compatibility_ids[(family_id, revision_id)]
        if compatibility_id is not None:
            record["generation_compatibility_id"] = compatibility_id
        if len(variants) == 1:
            record["source_sha256_fingerprint"] = variants[0][
                "source_sha256_fingerprint"
            ]
        family_revision_records.append(record)
    generation_identity: dict[str, object] = {
        "dependency_environment_fingerprint": _canonical_sha256(dependency_environment),
        "family_revisions": family_revision_records,
    }
    generation_identity["compatibility_fingerprint"] = _canonical_sha256(
        generation_identity
    )
    return {
        "target": dict(shared_target),
        "trajectory_numerical": common_numerical,
        "trajectory_numerical_by_family_revision": numerical_records,
        "stored_dtypes": dict(dataset_manifest_builder._STORED_DTYPES),
        "whole_case_rows": True,
        "generation_identity": generation_identity,
    }


def authenticate_combined_summary(path: Path) -> CombinedBinding:
    """Canonically reconstruct and authenticate the exact final c16384 view."""

    summary_artifact = _artifact(path)
    if summary_artifact.path.name != "paper_dataset_all_splits_c16384.summary.json":
        raise ValueError("post-completion runner requires the exact c16384 summary")
    summary = _strict_json(summary_artifact.path)
    if (
        summary.get("schema") != COMBINED_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
    ):
        raise ValueError("combined summary is not a completed v1 view")
    preflight = _mapping(summary.get("preflight"), context="combined preflight")
    if preflight.get("schema") != COMBINED_PREFLIGHT_SCHEMA:
        raise ValueError("combined preflight has the wrong schema")
    chunks_raw = _sequence(preflight.get("chunks"), context="combined chunks")
    if len(chunks_raw) != FINAL_SOURCE_COUNT:
        raise RuntimeError("combined preflight must contain exactly 26 sources")
    summary_paths: list[Path] = []
    for index, raw_chunk in enumerate(chunks_raw):
        chunk = _mapping(raw_chunk, context=f"combined chunk {index}")
        summary_value = chunk.get("summary_path")
        if not isinstance(summary_value, str) or not Path(summary_value).is_absolute():
            raise ValueError("combined source-summary paths must be absolute")
        summary_paths.append(Path(summary_value))
    if len(set(summary_paths)) != FINAL_SOURCE_COUNT:
        raise RuntimeError("combined preflight repeats a source summary")

    view_name = "paper_dataset_all_splits_c16384"
    plan, output_root, canonical_name, expected_preflight = (
        combined_view_builder.preflight(
            tuple(summary_paths),
            output_root=summary_artifact.path.parent,
            name=view_name,
        )
    )
    if output_root != summary_artifact.path.parent or canonical_name != view_name:
        raise RuntimeError("combined canonical view layout differs")
    if not _same_json(preflight, expected_preflight):
        raise RuntimeError("combined preflight differs from canonical reconstruction")
    if (
        tuple(split.value for split in plan.splits) != SPLIT_ORDER
        or not _same_json(
            plan.accepted_cases_per_family_by_split,
            CASES_PER_FAMILY_BY_SPLIT,
        )
        or plan.accepted_cases != FINAL_ACCEPTED_CASES
        or plan.expected_rows != FINAL_RETAINED_ROWS
        or not _same_json(
            expected_preflight.get("expected_rows_by_split_and_family"),
            _expected_rows_by_split_and_family(),
        )
    ):
        raise RuntimeError("combined canonical plan differs from the final contract")

    view = _mapping(summary.get("dataset_view"), context="combined dataset_view")
    manifest = _artifact_from_record(
        summary_artifact.path.parent,
        view.get("manifest"),
        context="combined manifest",
    )
    trajectory_map = _artifact_from_record(
        summary_artifact.path.parent,
        view.get("trajectory_map"),
        context="combined trajectory map",
    )
    expected_manifest = summary_artifact.path.parent / f"{view_name}.dataset.json"
    expected_map = summary_artifact.path.parent / f"{view_name}.trajectory_map.npz"
    if manifest.path != expected_manifest or trajectory_map.path != expected_map:
        raise RuntimeError("combined dataset-view artifact names differ from contract")
    validated_view = combined_view_builder._validate_view(
        combined_view_builder.DatasetViewPaths(
            manifest=manifest.path,
            trajectory_map=trajectory_map.path,
        ),
        plan=plan,
    )
    manifest_record = _strict_json(manifest.path)
    expected_contract = _reconstruct_dataset_contract(plan)
    expected_contract_fingerprint = _canonical_sha256(expected_contract)
    if (
        not _same_json(
            manifest_record.get("dataset_contract"),
            expected_contract,
        )
        or manifest_record.get("dataset_contract_fingerprint")
        != expected_contract_fingerprint
        or validated_view.get("dataset_contract_fingerprint")
        != expected_contract_fingerprint
    ):
        raise RuntimeError(
            "combined dataset contract differs from proposal reconstruction"
        )
    if not _same_json(view, validated_view):
        raise RuntimeError("combined dataset_view differs from canonical validation")

    for label, artifact in (
        ("summary", summary_artifact),
        ("manifest", manifest),
        ("trajectory map", trajectory_map),
    ):
        current = _artifact(artifact.path, expected_sha256=artifact.sha256)
        if current != artifact:
            raise RuntimeError(f"combined {label} changed during authentication")
    return CombinedBinding(
        summary=summary_artifact,
        manifest=manifest,
        trajectory_map=trajectory_map,
        chunks=tuple(
            dict(_mapping(chunk, context=f"canonical combined chunk {index}"))
            for index, chunk in enumerate(
                _sequence(
                    expected_preflight.get("chunks"),
                    context="canonical combined chunks",
                )
            )
        ),
    )


@dataclass(frozen=True)
class FamilyAuditSpec:
    name: str
    schema: str
    path: Path
    rows: int
    nested_counts: bool = False


def _validate_audit_checks(record: Mapping[str, Any], *, family: str) -> None:
    checks = _mapping(record.get("checks"), context=f"{family} audit checks")
    expected = EXPECTED_AUDIT_CHECKS.get(family)
    if expected is None:
        raise ValueError(f"unknown family audit check contract: {family}")
    if not _same_json(checks, expected):
        missing = sorted(set(expected) - set(checks))
        extra = sorted(set(checks) - set(expected))
        differing = sorted(
            name
            for name in set(expected) & set(checks)
            if not _same_json(checks[name], expected[name])
        )
        raise RuntimeError(
            f"{family} audit checks differ from schema contract: "
            f"missing={missing}, extra={extra}, differing={differing}"
        )


def _require_exact_keys(
    record: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    observed = set(record)
    if observed != expected:
        raise RuntimeError(
            f"{context} fields differ: "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _strict_source_map(value: object, *, context: str) -> dict[str, str]:
    raw = _mapping(value, context=context)
    result: dict[str, str] = {}
    for path, digest in raw.items():
        if not isinstance(path, str) or not path:
            raise ValueError(f"{context} paths must be nonempty strings")
        relative = Path(path)
        if relative.is_absolute() or path == "." or ".." in relative.parts:
            raise ValueError(f"{context} paths must be strict relative paths")
        result[path] = _digest(digest, context=f"{context} digest for {path}")
    return result


def _strict_dependency_environment(
    value: object,
    *,
    context: str,
) -> dict[str, object]:
    environment = _mapping(value, context=context)
    _require_exact_keys(
        environment,
        {"python", "packages", "files_sha256"},
        context=context,
    )
    python = _mapping(environment.get("python"), context=f"{context} python")
    packages = _mapping(environment.get("packages"), context=f"{context} packages")
    files = _mapping(environment.get("files_sha256"), context=f"{context} files_sha256")
    _require_exact_keys(
        python,
        {"implementation", "version"},
        context=f"{context} python",
    )
    _require_exact_keys(
        packages,
        {"jax", "jaxlib", "numpy"},
        context=f"{context} packages",
    )
    _require_exact_keys(
        files,
        set(DEPENDENCY_FILE_PATHS),
        context=f"{context} files_sha256",
    )
    normalized_python: dict[str, str] = {}
    for name in ("implementation", "version"):
        item = python.get(name)
        if not isinstance(item, str) or not item:
            raise ValueError(f"{context} python {name} must be a nonempty string")
        normalized_python[name] = item
    normalized_packages: dict[str, str] = {}
    for name in ("jax", "jaxlib", "numpy"):
        item = packages.get(name)
        if not isinstance(item, str) or not item:
            raise ValueError(f"{context} package {name} must be a nonempty string")
        normalized_packages[name] = item
    return {
        "python": normalized_python,
        "packages": normalized_packages,
        "files_sha256": {
            name: _digest(files.get(name), context=f"{context} dependency file {name}")
            for name in DEPENDENCY_FILE_PATHS
        },
    }


def _terminal_reauthenticate_artifact_groups(
    groups: Mapping[str, tuple[Path, Mapping[str, Artifact]]],
) -> None:
    """Canonically re-resolve and freshly rehash a small bound artifact set."""

    for group, (requested_root, artifacts) in sorted(groups.items()):
        root = _require_real_directory(
            requested_root,
            context=f"final {group} root",
        )
        for relative, expected in sorted(artifacts.items()):
            current_path = _source_file_inside(
                root,
                relative,
                context=f"final {group} artifact {relative}",
            )
            if current_path != expected.path:
                raise RuntimeError(f"final {group} artifact path changed: {relative}")
            current = _artifact(current_path, expected_sha256=expected.sha256)
            if current != expected:
                raise RuntimeError(
                    f"final {group} artifact byte identity changed: {relative}"
                )


def _release_terminal_artifact_groups(
    *,
    dependency: DependencyBinding,
    generation_sources: Mapping[str, GenerationSourceBinding],
    historical_source_snapshots: Mapping[str, Mapping[str, Artifact]],
    repository_root: Path | None = ROOT,
) -> dict[str, tuple[Path, dict[str, Artifact]]]:
    """Build collision-free canonical groups for the final provenance sweep."""

    current: dict[str, Artifact] = {}
    for family in FAMILY_ORDER:
        binding = generation_sources[family]
        for relative, artifact in binding.current_sources.items():
            previous = current.get(relative)
            if previous is not None and previous != artifact:
                raise RuntimeError(
                    f"release families conflict on current source: {relative}"
                )
            current[relative] = artifact
    for relative, artifact in dependency.files.items():
        if relative in current:
            raise RuntimeError(f"dependency path collides with a source: {relative}")
        current[relative] = artifact
    if repository_root is None:
        inferred_roots = {
            artifact.path.parents[len(Path(relative).parts) - 1]
            for relative, artifact in current.items()
        }
        if len(inferred_roots) != 1:
            raise RuntimeError("release current artifacts do not share one exact root")
        repository_root = inferred_roots.pop()

    groups: dict[str, tuple[Path, dict[str, Artifact]]] = {
        "generation repository": (repository_root, current),
    }
    historical_specs = {
        "stokes": (
            STOKES_HISTORICAL_SOURCE_SNAPSHOT_ROOT,
            STOKES_HISTORICAL_SOURCE_SNAPSHOTS,
        ),
        "benjamin_feir": (
            BF_HISTORICAL_SOURCE_SNAPSHOT_ROOT,
            BF_HISTORICAL_SOURCE_SNAPSHOTS,
        ),
    }
    for family, artifacts in historical_source_snapshots.items():
        snapshot_root, source_specs = historical_specs[family]
        physical_names = {
            "SHA256SUMS": "SHA256SUMS",
            **{
                repository_path: str(spec["name"])
                for repository_path, spec in source_specs.items()
            },
        }
        if set(artifacts) != set(physical_names):
            raise RuntimeError(f"{family} historical sweep set is incomplete")
        groups[f"{family} historical snapshot"] = (
            snapshot_root,
            {physical_names[name]: artifact for name, artifact in artifacts.items()},
        )
    return groups


def _validate_family_generation_identity(
    *,
    family: str,
    audit_record: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
    combined_chunks: Sequence[Mapping[str, object]],
    repository_root: Path,
    artifact_cache: dict[Path, Artifact],
) -> tuple[GenerationSourceBinding, dict[str, object]]:
    """Authenticate one exact family source map and dependency declaration."""

    expected_paths = EXPECTED_GENERATION_SOURCE_PATHS_BY_FAMILY[family]
    expected_chunk_count = SOURCE_COUNT_BY_FAMILY[family]
    if len(source_records) != expected_chunk_count:
        raise RuntimeError(f"{family} generation identity has the wrong chunk count")
    source_maps: list[dict[str, str]] = []
    dependency_records: list[dict[str, object]] = []
    for index, source_record in enumerate(source_records):
        run_spec = _mapping(
            source_record.get("run_spec"), context=f"{family} source {index} run_spec"
        )
        configuration = _mapping(
            run_spec.get("configuration"),
            context=f"{family} source {index} configuration",
        )
        source_map = _strict_source_map(
            configuration.get("source_sha256"),
            context=f"{family} source {index} source_sha256",
        )
        if set(source_map) != expected_paths:
            raise RuntimeError(
                f"{family} generation source paths differ: "
                f"missing={sorted(expected_paths - set(source_map))}, "
                f"extra={sorted(set(source_map) - expected_paths)}"
            )
        source_maps.append(source_map)
        dependency_records.append(
            _strict_dependency_environment(
                configuration.get("dependency_environment"),
                context=f"{family} source {index} dependency_environment",
            )
        )
    first_sources = source_maps[0]
    if any(source_map != first_sources for source_map in source_maps[1:]):
        raise RuntimeError(f"{family} source summaries use unequal exact source maps")
    first_dependency = dependency_records[0]
    if any(
        not _same_json(record, first_dependency) for record in dependency_records[1:]
    ):
        raise RuntimeError(f"{family} source summaries use unequal dependencies")

    source_fingerprint = _canonical_sha256(first_sources)
    dependency_fingerprint = _canonical_sha256(first_dependency)
    identity = _mapping(audit_record.get("identity"), context=f"{family} identity")
    audit_source_key = (
        "source_fingerprint" if family == "stokes" else "source_sha256_fingerprint"
    )
    audit_dependency_key = (
        "dependency_fingerprint"
        if family == "stokes"
        else "dependency_environment_fingerprint"
    )
    if (
        _digest(
            identity.get(audit_source_key),
            context=f"{family} audit source fingerprint",
        )
        != source_fingerprint
        or _digest(
            identity.get(audit_dependency_key),
            context=f"{family} audit dependency fingerprint",
        )
        != dependency_fingerprint
    ):
        raise RuntimeError(f"{family} audit generation identity differs from summaries")

    family_chunks = tuple(
        chunk for chunk in combined_chunks if chunk.get("family") == family
    )
    if len(family_chunks) != expected_chunk_count or any(
        _digest(
            chunk.get("source_fingerprint"),
            context=f"{family} combined source fingerprint",
        )
        != source_fingerprint
        or _digest(
            chunk.get("dependency_fingerprint"),
            context=f"{family} combined dependency fingerprint",
        )
        != dependency_fingerprint
        for chunk in family_chunks
    ):
        raise RuntimeError(
            f"{family} combined generation identity differs from source summaries"
        )

    root = _require_real_directory(
        repository_root, context="generation repository root"
    )
    historical_paths = HISTORICAL_GENERATION_SOURCE_PATHS_BY_FAMILY[family]
    current_paths = expected_paths - historical_paths
    current_sources: dict[str, Artifact] = {}
    for relative in sorted(current_paths):
        source_path = _source_file_inside(
            root,
            relative,
            context=f"{family} current generation source {relative}",
        )
        expected_digest = first_sources[relative]
        artifact = artifact_cache.get(source_path)
        if artifact is None:
            artifact = _artifact(source_path, expected_sha256=expected_digest)
            artifact_cache[source_path] = artifact
        elif artifact.sha256 != expected_digest:
            raise RuntimeError(
                f"{family} source conflicts with another family: {relative}"
            )
        current_sources[relative] = artifact
    if len(current_sources) != len(current_paths):
        raise RuntimeError(f"{family} current generation source set is incomplete")
    return (
        GenerationSourceBinding(
            source_sha256=dict(sorted(first_sources.items())),
            fingerprint=source_fingerprint,
            current_sources=current_sources,
            historical_source_paths=tuple(sorted(historical_paths)),
            chunk_count=expected_chunk_count,
        ),
        first_dependency,
    )


def _authenticate_release_generation_identity(
    *,
    combined: CombinedBinding,
    audits: Mapping[str, Artifact],
    audit_sources_by_family: Mapping[str, Sequence[Mapping[str, object]]],
    repository_root: Path = ROOT,
) -> tuple[DependencyBinding, dict[str, GenerationSourceBinding]]:
    """Bind every generation source byte and the common current dependency state."""

    artifact_cache: dict[Path, Artifact] = {}
    generation_sources: dict[str, GenerationSourceBinding] = {}
    dependency_records: list[dict[str, object]] = []
    for family in FAMILY_ORDER:
        audit_record = _strict_json(audits[family].path)
        source_records = tuple(
            _strict_json(Path(str(binding["summary_path"])))
            for binding in audit_sources_by_family[family]
        )
        generation, dependency = _validate_family_generation_identity(
            family=family,
            audit_record=audit_record,
            source_records=source_records,
            combined_chunks=combined.chunks,
            repository_root=repository_root,
            artifact_cache=artifact_cache,
        )
        generation_sources[family] = generation
        dependency_records.extend(
            dict(dependency) for _ in range(SOURCE_COUNT_BY_FAMILY[family])
        )
    if set(generation_sources) != set(FAMILY_ORDER) or len(
        artifact_cache
    ) != len(CURRENT_GENERATION_SOURCE_PATHS):
        raise RuntimeError("release generation source binding is incomplete")
    if len(dependency_records) != FINAL_SOURCE_COUNT or any(
        not _same_json(record, dependency_records[0])
        for record in dependency_records[1:]
    ):
        raise RuntimeError("release chunks do not use one exact dependency environment")

    root = _require_real_directory(
        repository_root, context="dependency repository root"
    )
    dependency_files = {
        relative: _artifact(
            _source_file_inside(
                root,
                relative,
                context=f"dependency file {relative}",
            ),
            expected_sha256=_mapping(
                dependency_records[0].get("files_sha256"),
                context="release dependency files_sha256",
            ).get(relative),
        )
        for relative in DEPENDENCY_FILE_PATHS
    }
    current_environment = {
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
            name: artifact.sha256 for name, artifact in dependency_files.items()
        },
    }
    declared_environment = dependency_records[0]
    if not _same_json(declared_environment, current_environment):
        raise RuntimeError("release dependency environment is not current")
    fingerprint = _canonical_sha256(declared_environment)
    dependency_binding = DependencyBinding(
        environment=declared_environment,
        fingerprint=fingerprint,
        files=dependency_files,
        chunk_count=FINAL_SOURCE_COUNT,
    )
    _terminal_reauthenticate_artifact_groups(
        _release_terminal_artifact_groups(
            dependency=dependency_binding,
            generation_sources=generation_sources,
            historical_source_snapshots={},
            repository_root=root,
        )
    )
    return (
        dependency_binding,
        generation_sources,
    )


def _stokes_source_maps(
    source_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], ...]:
    """Parse one exact 13-path map independently from all six summaries."""

    mappings: list[dict[str, str]] = []
    for index, source_record in enumerate(source_records):
        run_spec = _mapping(
            source_record.get("run_spec"), context=f"Stokes source {index} run_spec"
        )
        configuration = _mapping(
            run_spec.get("configuration"),
            context=f"Stokes source {index} configuration",
        )
        raw_sources = _mapping(
            configuration.get("source_sha256"),
            context=f"Stokes source {index} source_sha256",
        )
        source_mapping: dict[str, str] = {}
        for path, digest in raw_sources.items():
            if (
                not isinstance(path, str)
                or not path
                or Path(path).is_absolute()
                or ".." in Path(path).parts
            ):
                raise ValueError(
                    "Stokes source-map paths must be strict relative paths"
                )
            source_mapping[path] = _digest(
                digest,
                context=f"Stokes source {index} digest for {path}",
            )
        mappings.append(source_mapping)
    if len(mappings) != SOURCE_COUNT_BY_FAMILY["stokes"]:
        raise RuntimeError("Stokes source binding does not contain all six chunks")
    first = mappings[0]
    if any(mapping != first for mapping in mappings[1:]):
        raise RuntimeError("Stokes source summaries do not contain exactly equal maps")
    if set(first) != STOKES_EXPECTED_GENERATION_SOURCE_PATHS:
        missing = sorted(STOKES_EXPECTED_GENERATION_SOURCE_PATHS - set(first))
        extra = sorted(set(first) - STOKES_EXPECTED_GENERATION_SOURCE_PATHS)
        raise RuntimeError(
            "Stokes source summaries differ from the frozen 13-path contract: "
            f"missing={missing}, extra={extra}"
        )
    if _canonical_sha256(first) != STOKES_EXPECTED_SOURCE_FINGERPRINT:
        raise RuntimeError("Stokes source map differs from the frozen fingerprint")
    return tuple(mappings)


def _validate_stokes_source_identity(
    audit_record: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
    *,
    snapshot_root: Path = STOKES_HISTORICAL_SOURCE_SNAPSHOT_ROOT,
    repository_root: Path = ROOT,
) -> dict[str, Artifact]:
    """Independently authenticate Stokes historical and current source bytes."""

    source_mappings = _stokes_source_maps(source_records)
    first_sources = source_mappings[0]
    identity = _mapping(audit_record.get("identity"), context="Stokes identity")
    _require_exact_keys(
        identity,
        {
            "dependency_fingerprint",
            "execution_fingerprint",
            "source_fingerprint",
            "execution_platform",
            "configuration_fingerprints",
            "source_sha256",
            "historical_generation_source_snapshot_binding",
            "nonhistorical_generation_source_binding",
        },
        context="Stokes identity",
    )
    raw_identity_sources = _mapping(
        identity.get("source_sha256"), context="Stokes identity source_sha256"
    )
    if not _same_json(raw_identity_sources, first_sources):
        raise RuntimeError("Stokes identity source_sha256 differs from chunk maps")
    source_fingerprint = _digest(
        identity.get("source_fingerprint"), context="Stokes source_fingerprint"
    )
    if source_fingerprint != _canonical_sha256(first_sources):
        raise RuntimeError("Stokes source_fingerprint differs from source_sha256")
    configuration_fingerprints = [
        _digest(
            source_record.get("configuration_fingerprint"),
            context=f"Stokes source {index} configuration fingerprint",
        )
        for index, source_record in enumerate(source_records)
    ]
    if not _same_json(
        identity.get("configuration_fingerprints"),
        configuration_fingerprints,
    ):
        raise RuntimeError("Stokes configuration fingerprints differ from summaries")

    resolved_snapshot_root = _require_real_directory(
        snapshot_root,
        context="Stokes historical snapshot root",
    )
    historical = _mapping(
        identity.get("historical_generation_source_snapshot_binding"),
        context="Stokes historical source snapshot binding",
    )
    _require_exact_keys(
        historical,
        {
            "schema",
            "role",
            "source_commit",
            "snapshot_root",
            "source_map_fingerprint",
            "source_count",
            "historical_source_count",
            "sha256sums",
            "sources",
            "chunk_source_maps_checked",
        },
        context="Stokes historical source snapshot binding",
    )
    if (
        historical.get("schema")
        != "paper_dataset_stokes_revision2_historical_source_binding_v1"
        or historical.get("role") != "inert_historical_byte_recovery_only"
        or historical.get("source_commit") != STOKES_HISTORICAL_SOURCE_COMMIT
        or historical.get("snapshot_root") != str(resolved_snapshot_root)
        or historical.get("source_map_fingerprint") != source_fingerprint
        or historical.get("source_count") != len(first_sources)
        or historical.get("historical_source_count")
        != len(STOKES_HISTORICAL_SOURCE_SNAPSHOTS)
        or historical.get("chunk_source_maps_checked")
        != SOURCE_COUNT_BY_FAMILY["stokes"]
    ):
        raise RuntimeError("Stokes historical source snapshot identity differs")

    manifest_spec = STOKES_HISTORICAL_SOURCE_SNAPSHOT_MANIFEST
    manifest_path = _source_file_inside(
        resolved_snapshot_root,
        str(manifest_spec["name"]),
        context="Stokes historical SHA256SUMS",
    )
    manifest_record = _mapping(
        historical.get("sha256sums"), context="Stokes historical SHA256SUMS record"
    )
    _require_exact_keys(
        manifest_record,
        {"path", "bytes", "sha256"},
        context="Stokes historical SHA256SUMS record",
    )
    if not _same_json(
        manifest_record,
        {
            "path": str(manifest_path),
            "bytes": manifest_spec["bytes"],
            "sha256": manifest_spec["sha256"],
        },
    ):
        raise RuntimeError("Stokes historical SHA256SUMS identity differs")
    expected_manifest_text = "".join(
        f"{spec['sha256']}  {spec['name']}\n"
        for spec in STOKES_HISTORICAL_SOURCE_SNAPSHOTS.values()
    )
    if manifest_path.read_text(encoding="utf-8") != expected_manifest_text:
        raise RuntimeError("Stokes historical SHA256SUMS content differs")
    manifest_artifact = _artifact(
        manifest_path,
        expected_sha256=manifest_spec["sha256"],
    )
    if manifest_artifact.bytes != manifest_spec["bytes"]:
        raise RuntimeError("Stokes historical SHA256SUMS byte count differs")

    raw_historical_sources = _mapping(
        historical.get("sources"), context="Stokes historical source records"
    )
    if set(raw_historical_sources) != set(STOKES_HISTORICAL_SOURCE_SNAPSHOTS):
        raise RuntimeError("Stokes historical snapshot source set differs")
    historical_artifacts: dict[str, Artifact] = {"SHA256SUMS": manifest_artifact}
    for repository_path, snapshot_spec in STOKES_HISTORICAL_SOURCE_SNAPSHOTS.items():
        expected_digest = str(snapshot_spec["sha256"])
        if any(
            mapping.get(repository_path) != expected_digest
            for mapping in source_mappings
        ):
            raise RuntimeError(
                f"Stokes historical digest differs from a chunk map: {repository_path}"
            )
        snapshot_path = _source_file_inside(
            resolved_snapshot_root,
            str(snapshot_spec["name"]),
            context=f"Stokes historical source {repository_path}",
        )
        source_record = _mapping(
            raw_historical_sources.get(repository_path),
            context=f"Stokes historical source record {repository_path}",
        )
        _require_exact_keys(
            source_record,
            {"snapshot_path", "bytes", "sha256"},
            context=f"Stokes historical source record {repository_path}",
        )
        if not _same_json(
            source_record,
            {
                "snapshot_path": str(snapshot_path),
                "bytes": snapshot_spec["bytes"],
                "sha256": expected_digest,
            },
        ):
            raise RuntimeError(
                f"Stokes historical source identity differs: {repository_path}"
            )
        artifact = _artifact(snapshot_path, expected_sha256=expected_digest)
        if artifact.bytes != snapshot_spec["bytes"]:
            raise RuntimeError(
                f"Stokes historical source byte count differs: {repository_path}"
            )
        historical_artifacts[repository_path] = artifact

    resolved_repository_root = _require_real_directory(
        repository_root,
        context="Stokes repository root",
    )
    current = _mapping(
        identity.get("nonhistorical_generation_source_binding"),
        context="Stokes nonhistorical generation source binding",
    )
    _require_exact_keys(
        current,
        {
            "schema",
            "role",
            "repository_root",
            "source_map_fingerprint",
            "source_count",
            "current_source_count",
            "historical_snapshot_source_paths",
            "sources",
            "chunk_source_maps_checked",
        },
        context="Stokes nonhistorical generation source binding",
    )
    historical_paths = set(STOKES_HISTORICAL_SOURCE_SNAPSHOTS)
    current_sources = {
        path: digest
        for path, digest in first_sources.items()
        if path not in historical_paths
    }
    if (
        current.get("schema")
        != "paper_dataset_stokes_revision2_current_source_binding_v1"
        or current.get("role")
        != "current_repository_bytes_for_all_nonhistorical_generation_sources"
        or current.get("repository_root") != str(resolved_repository_root)
        or current.get("source_map_fingerprint") != source_fingerprint
        or current.get("source_count") != len(first_sources)
        or current.get("current_source_count") != len(current_sources)
        or current.get("historical_snapshot_source_paths") != sorted(historical_paths)
        or current.get("chunk_source_maps_checked") != SOURCE_COUNT_BY_FAMILY["stokes"]
    ):
        raise RuntimeError("Stokes nonhistorical generation source identity differs")
    raw_current_sources = _mapping(
        current.get("sources"), context="Stokes current source records"
    )
    if set(raw_current_sources) != set(current_sources):
        raise RuntimeError("Stokes current source record set differs from chunk maps")
    for repository_path, expected_digest in current_sources.items():
        source_path = _source_file_inside(
            resolved_repository_root,
            repository_path,
            context=f"Stokes current source {repository_path}",
        )
        source_record = _mapping(
            raw_current_sources.get(repository_path),
            context=f"Stokes current source record {repository_path}",
        )
        _require_exact_keys(
            source_record,
            {"path", "bytes", "sha256"},
            context=f"Stokes current source record {repository_path}",
        )
        if source_record.get("path") != str(source_path):
            raise RuntimeError(f"Stokes current source path differs: {repository_path}")
        if source_record.get("sha256") != expected_digest:
            raise RuntimeError(
                f"Stokes current source digest differs: {repository_path}"
            )
        artifact = _artifact(source_path, expected_sha256=expected_digest)
        if artifact.bytes != _integer(
            source_record.get("bytes"),
            context=f"Stokes current source bytes {repository_path}",
        ):
            raise RuntimeError(
                f"Stokes current source byte count differs: {repository_path}"
            )
    return historical_artifacts


def _bf_source_maps(
    source_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], ...]:
    mappings: list[dict[str, str]] = []
    for index, source_record in enumerate(source_records):
        run_spec = _mapping(
            source_record.get("run_spec"), context=f"BF source {index} run_spec"
        )
        configuration = _mapping(
            run_spec.get("configuration"),
            context=f"BF source {index} configuration",
        )
        raw_sources = _mapping(
            configuration.get("source_sha256"),
            context=f"BF source {index} source_sha256",
        )
        source_mapping: dict[str, str] = {}
        for path, digest in raw_sources.items():
            if not isinstance(path, str) or not path:
                raise ValueError("BF source-map paths must be nonempty strings")
            source_mapping[path] = _digest(
                digest, context=f"BF source {index} digest for {path}"
            )
        mappings.append(source_mapping)
    if len(mappings) != SOURCE_COUNT_BY_FAMILY["benjamin_feir"]:
        raise RuntimeError("BF source binding does not contain all six chunks")
    first = mappings[0]
    if any(mapping != first for mapping in mappings[1:]):
        raise RuntimeError("BF source summaries do not contain exactly equal maps")
    if set(first) != BF_EXPECTED_GENERATION_SOURCE_PATHS:
        missing = sorted(BF_EXPECTED_GENERATION_SOURCE_PATHS - set(first))
        extra = sorted(set(first) - BF_EXPECTED_GENERATION_SOURCE_PATHS)
        raise RuntimeError(
            "BF source summaries differ from the frozen 19-path contract: "
            f"missing={missing}, extra={extra}"
        )
    return tuple(mappings)


def _validate_bf_source_identity(
    audit_record: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
    *,
    snapshot_root: Path = BF_HISTORICAL_SOURCE_SNAPSHOT_ROOT,
    repository_root: Path = ROOT,
) -> dict[str, Artifact]:
    """Physically authenticate BF historical and current source bindings."""

    source_mappings = _bf_source_maps(source_records)
    first_sources = source_mappings[0]
    identity = _mapping(audit_record.get("identity"), context="BF audit identity")
    if identity.get("source_sha256_fingerprint") != _canonical_sha256(first_sources):
        raise RuntimeError("BF audit source fingerprint differs from chunk maps")

    requested_snapshot_root = Path(snapshot_root).expanduser()
    if requested_snapshot_root.is_symlink() or not requested_snapshot_root.is_dir():
        raise ValueError("BF historical snapshot root is not a real directory")
    resolved_snapshot_root = requested_snapshot_root.resolve()
    historical = _mapping(
        identity.get("historical_shared_source_snapshot_binding"),
        context="BF historical source snapshot binding",
    )
    _require_exact_keys(
        historical,
        {
            "schema",
            "role",
            "snapshot_root",
            "sha256sums",
            "sources",
            "chunk_source_maps_checked",
        },
        context="BF historical source snapshot binding",
    )
    if (
        historical.get("schema")
        != "paper_dataset_bf_revision4_historical_source_binding_v1"
        or historical.get("role") != "inert_historical_byte_recovery_only"
        or historical.get("snapshot_root") != str(resolved_snapshot_root)
        or historical.get("chunk_source_maps_checked")
        != SOURCE_COUNT_BY_FAMILY["benjamin_feir"]
    ):
        raise RuntimeError("BF historical source snapshot identity differs")

    manifest_spec = BF_HISTORICAL_SOURCE_SNAPSHOT_MANIFEST
    manifest_path = resolved_snapshot_root / str(manifest_spec["name"])
    manifest_record = _mapping(
        historical.get("sha256sums"), context="BF historical SHA256SUMS record"
    )
    _require_exact_keys(
        manifest_record,
        {"path", "bytes", "sha256"},
        context="BF historical SHA256SUMS record",
    )
    if not _same_json(
        manifest_record,
        {
            "path": str(manifest_path),
            "bytes": manifest_spec["bytes"],
            "sha256": manifest_spec["sha256"],
        },
    ):
        raise RuntimeError("BF historical SHA256SUMS identity differs")
    manifest_artifact = _artifact(
        manifest_path, expected_sha256=manifest_spec["sha256"]
    )
    if manifest_artifact.bytes != manifest_spec["bytes"]:
        raise RuntimeError("BF historical SHA256SUMS byte count differs")

    raw_historical_sources = _mapping(
        historical.get("sources"), context="BF historical source records"
    )
    if set(raw_historical_sources) != set(BF_HISTORICAL_SOURCE_SNAPSHOTS):
        raise RuntimeError("BF historical snapshot source set differs")
    historical_artifacts: dict[str, Artifact] = {"SHA256SUMS": manifest_artifact}
    for repository_path, snapshot_spec in BF_HISTORICAL_SOURCE_SNAPSHOTS.items():
        expected_digest = str(snapshot_spec["sha256"])
        if any(
            mapping.get(repository_path) != expected_digest
            for mapping in source_mappings
        ):
            raise RuntimeError(
                f"BF historical digest differs from a chunk map: {repository_path}"
            )
        snapshot_path = resolved_snapshot_root / str(snapshot_spec["name"])
        source_record = _mapping(
            raw_historical_sources.get(repository_path),
            context=f"BF historical source record {repository_path}",
        )
        _require_exact_keys(
            source_record,
            {"snapshot_path", "bytes", "sha256"},
            context=f"BF historical source record {repository_path}",
        )
        if not _same_json(
            source_record,
            {
                "snapshot_path": str(snapshot_path),
                "bytes": snapshot_spec["bytes"],
                "sha256": expected_digest,
            },
        ):
            raise RuntimeError(
                f"BF historical source identity differs: {repository_path}"
            )
        artifact = _artifact(snapshot_path, expected_sha256=expected_digest)
        if artifact.bytes != snapshot_spec["bytes"]:
            raise RuntimeError(
                f"BF historical source byte count differs: {repository_path}"
            )
        historical_artifacts[repository_path] = artifact

    requested_repository_root = Path(repository_root).expanduser()
    if requested_repository_root.is_symlink() or not requested_repository_root.is_dir():
        raise ValueError("BF repository root is not a real directory")
    resolved_repository_root = requested_repository_root.resolve()
    current = _mapping(
        identity.get("nonhistorical_generation_source_binding"),
        context="BF nonhistorical generation source binding",
    )
    _require_exact_keys(
        current,
        {
            "schema",
            "role",
            "repository_root",
            "source_map_fingerprint",
            "source_count",
            "current_source_count",
            "historical_snapshot_source_paths",
            "sources",
            "chunk_source_maps_checked",
        },
        context="BF nonhistorical generation source binding",
    )
    historical_paths = set(BF_HISTORICAL_SOURCE_SNAPSHOTS)
    current_sources = {
        path: digest
        for path, digest in first_sources.items()
        if path not in historical_paths
    }
    if (
        current.get("schema") != "paper_dataset_bf_revision4_current_source_binding_v1"
        or current.get("role")
        != "current_repository_bytes_for_all_nonhistorical_generation_sources"
        or current.get("repository_root") != str(resolved_repository_root)
        or current.get("source_map_fingerprint") != _canonical_sha256(first_sources)
        or current.get("source_count") != len(first_sources)
        or current.get("current_source_count") != len(current_sources)
        or current.get("historical_snapshot_source_paths") != sorted(historical_paths)
        or current.get("chunk_source_maps_checked")
        != SOURCE_COUNT_BY_FAMILY["benjamin_feir"]
    ):
        raise RuntimeError("BF nonhistorical generation source identity differs")
    raw_current_sources = _mapping(
        current.get("sources"), context="BF current source records"
    )
    if set(raw_current_sources) != set(current_sources):
        raise RuntimeError("BF current source record set differs from chunk maps")
    for repository_path, expected_digest in current_sources.items():
        source_path = resolved_repository_root / repository_path
        source_record = _mapping(
            raw_current_sources.get(repository_path),
            context=f"BF current source record {repository_path}",
        )
        _require_exact_keys(
            source_record,
            {"path", "bytes", "sha256"},
            context=f"BF current source record {repository_path}",
        )
        if source_record.get("path") != str(source_path.resolve(strict=True)):
            raise RuntimeError(f"BF current source path differs: {repository_path}")
        if source_record.get("sha256") != expected_digest:
            raise RuntimeError(f"BF current source digest differs: {repository_path}")
        artifact = _artifact(source_path, expected_sha256=expected_digest)
        if artifact.bytes != _integer(
            source_record.get("bytes"),
            context=f"BF current source bytes {repository_path}",
        ):
            raise RuntimeError(
                f"BF current source byte count differs: {repository_path}"
            )
    return historical_artifacts


def _validate_family_audit(
    spec: FamilyAuditSpec,
) -> tuple[
    Artifact,
    tuple[dict[str, object], ...],
    dict[str, Artifact],
]:
    artifact = _artifact(
        spec.path,
        expected_sha256=(
            EXPECTED_STOKES_BINDING_SHA256 if spec.name == "stokes" else None
        ),
    )
    record = _strict_json(artifact.path)
    if record.get("schema") != spec.schema or record.get("status") != "pass":
        raise RuntimeError(f"{spec.name} audit schema/status differs")
    root_field = "root" if spec.nested_counts else "dataset_root"
    root_value = record.get(root_field)
    if (
        not isinstance(root_value, str)
        or Path(root_value).resolve() != artifact.path.parent
    ):
        raise RuntimeError(f"{spec.name} audit has the wrong dataset root")
    counts = (
        _mapping(record.get("counts"), context="Stokes binding counts")
        if spec.nested_counts
        else record
    )
    accepted = _integer(counts.get("accepted"), context="audit accepted")
    attempted = _integer(counts.get("attempted"), context="audit attempted")
    rejected = _integer(counts.get("rejected"), context="audit rejected")
    retained_rows = _integer(counts.get("retained_rows"), context="audit rows")
    if (
        accepted != 18_432
        or retained_rows != spec.rows
        or attempted < accepted
        or rejected != attempted - accepted
        or not _same_json(counts.get("accepted_by_split"), EXPECTED_SPLIT_COUNTS)
    ):
        raise RuntimeError(f"{spec.name} audit counts differ from final contract")
    if spec.nested_counts:
        if (
            record.get("family") != "stokes"
            or record.get("family_id") != 1
            or record.get("revision_id") != 2
        ):
            raise RuntimeError("Stokes binding family identity differs")
    if spec.name == "tanaka":
        identity = _mapping(record.get("identity"), context="Tanaka audit identity")
        if identity.get("historical_generation_compatibility_used") is not False:
            raise RuntimeError(
                "Tanaka audit identity does not prove a fresh revision-3 dataset"
            )
    _validate_audit_checks(record, family=spec.name)

    raw_chunks = _sequence(record.get("chunks"), context=f"{spec.name} audit chunks")
    layout = expected_layout(spec.name)
    if len(raw_chunks) != len(layout):
        raise RuntimeError(f"{spec.name} audit has the wrong chunk count")
    source_bindings: list[dict[str, object]] = []
    source_records: list[Mapping[str, Any]] = []
    source_paths: set[Path] = set()
    for index, (raw_chunk, expected) in enumerate(zip(raw_chunks, layout)):
        chunk = _mapping(raw_chunk, context=f"{spec.name} audit chunk {index}")
        split, before, count, stream = expected
        observed = (
            chunk.get("split"),
            chunk.get("accepted_before"),
            chunk.get("accepted_count"),
            chunk.get("stream_id"),
        )
        if observed != expected:
            raise RuntimeError(f"{spec.name} audit chunk {index} layout differs")
        if chunk.get("accepted_after") != before + count:
            raise RuntimeError(f"{spec.name} audit chunk interval is inconsistent")
        attempted_count = _integer(
            chunk.get("attempted_count"), context="audit chunk attempted_count"
        )
        if attempted_count < count:
            raise RuntimeError("audit chunk attempted count is too small")
        if chunk.get("retained_rows") != count * ROWS_PER_CASE[spec.name]:
            raise RuntimeError(f"{spec.name} audit chunk row count differs")
        summary_value = chunk.get("summary_path")
        if not isinstance(summary_value, str) or not Path(summary_value).is_absolute():
            raise ValueError("audit source-summary paths must be absolute")
        summary = _artifact(
            Path(summary_value), expected_sha256=chunk.get("summary_sha256")
        )
        if not summary.path.is_relative_to(artifact.path.parent):
            raise ValueError("audit source summary escapes its dataset root")
        manifest_value = chunk.get("manifest_path")
        map_value = chunk.get("trajectory_map_path")
        if not isinstance(manifest_value, str) or not isinstance(map_value, str):
            raise ValueError("audit chunk omits manifest or trajectory-map path")
        manifest = _artifact(
            Path(manifest_value), expected_sha256=chunk.get("manifest_sha256")
        )
        trajectory_map = _artifact(
            Path(map_value), expected_sha256=chunk.get("trajectory_map_sha256")
        )
        if not manifest.path.is_relative_to(artifact.path.parent) or not (
            trajectory_map.path.is_relative_to(artifact.path.parent)
        ):
            raise ValueError("audit chunk artifact escapes its dataset root")
        source_record = _strict_json(summary.path)
        source_records.append(source_record)
        if (
            source_record.get("schema") != "paper_dataset_quota_summary_v1"
            or source_record.get("status") != "complete"
            or Path(str(source_record.get("output_root"))).expanduser().resolve()
            != summary.path.parent
        ):
            raise RuntimeError("audited source summary is not a completed quota view")
        source_view = _mapping(
            source_record.get("dataset_view"), context="source dataset_view"
        )
        source_manifest = _artifact_from_record(
            summary.path.parent,
            source_view.get("manifest"),
            context="source manifest",
        )
        source_map = _artifact_from_record(
            summary.path.parent,
            source_view.get("trajectory_map"),
            context="source trajectory map",
        )
        if source_manifest != manifest or source_map != trajectory_map:
            raise RuntimeError(
                "family audit manifest/map identity differs from source summary"
            )
        if summary.path in source_paths:
            raise RuntimeError("family audit repeats a source summary")
        source_paths.add(summary.path)
        source_bindings.append(
            {
                "summary_path": str(summary.path),
                "summary_bytes": summary.bytes,
                "summary_sha256": summary.sha256,
                "manifest_path": str(manifest.path),
                "manifest_bytes": manifest.bytes,
                "manifest_sha256": manifest.sha256,
                "trajectory_map_path": str(trajectory_map.path),
                "trajectory_map_bytes": trajectory_map.bytes,
                "trajectory_map_sha256": trajectory_map.sha256,
                "family": spec.name,
                "split": split,
                "accepted_before": before,
                "accepted_count": count,
                "accepted_after": before + count,
                "attempted_count": attempted_count,
                "stream_id": stream,
                "configuration_fingerprint": _digest(
                    chunk.get("configuration_fingerprint"),
                    context=(f"{spec.name} audit chunk configuration fingerprint"),
                ),
            }
        )
    if spec.name == "stokes":
        historical_source_snapshots = _validate_stokes_source_identity(
            record,
            source_records,
        )
    elif spec.name == "benjamin_feir":
        historical_source_snapshots = _validate_bf_source_identity(
            record,
            source_records,
        )
    else:
        historical_source_snapshots = {}
    return artifact, tuple(source_bindings), historical_source_snapshots


def _validate_stokes_legacy(binding_path: Path, legacy_path: Path) -> Artifact:
    binding = _strict_json(binding_path)
    legacy = _artifact(legacy_path, expected_sha256=EXPECTED_STOKES_LEGACY_SHA256)
    record = _strict_json(legacy.path)
    if record.get("schema") != "paper_dataset_cap4_stokes_audit_v1":
        raise RuntimeError("legacy Stokes audit has the wrong schema")
    if record.get("status") != "pass":
        raise RuntimeError("legacy Stokes numerical audit did not pass")
    expected_counts = {
        "attempted": 18_432,
        "accepted": 18_432,
        "rejected": 0,
        "retained_rows": 18_432,
    }
    if any(record.get(name) != value for name, value in expected_counts.items()):
        raise RuntimeError("legacy Stokes audit counts differ")
    checks = _mapping(record.get("checks"), context="legacy Stokes checks")
    required_true = (
        "immutable_specs_rebuilt",
        "all_transactions_rescanned",
        "proposal_hashes_verified",
        "result_hashes_verified",
        "shard_hashes_verified",
        "manifest_hashes_verified",
        "trajectory_map_hashes_verified",
        "all_fields_finite",
        "all_depths_positive",
        "all_quality_masks_accept",
    )
    if any(checks.get(name) is not True for name in required_true):
        raise RuntimeError("legacy Stokes audit has a failed required check")
    if any(
        checks.get(name) != 0
        for name in ("pending_batches", "terminal_failures", "attempt_limit_failures")
    ):
        raise RuntimeError("legacy Stokes audit has incomplete transactions")
    binding_record = _mapping(
        binding.get("legacy_numerical_audit"), context="Stokes legacy binding"
    )
    if (
        Path(str(binding_record.get("path"))).expanduser().resolve() != legacy.path
        or binding_record.get("sha256") != legacy.sha256
        or binding_record.get("schema") != record.get("schema")
        or binding_record.get("status") != "pass"
    ):
        raise RuntimeError("Stokes binding does not authenticate the legacy audit")
    return legacy


def _cross_bind_audit_sources(
    combined_chunks: Sequence[Mapping[str, object]],
    audit_sources_by_family: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    """Require each ordered family audit subsequence to equal the combined view."""

    binding_fields = (
        "summary_path",
        "summary_sha256",
        "family",
        "split",
        "accepted_before",
        "accepted_count",
        "accepted_after",
        "attempted_count",
        "stream_id",
        "configuration_fingerprint",
    )
    for family in FAMILY_ORDER:
        combined_family = tuple(
            {name: chunk[name] for name in binding_fields}
            for chunk in combined_chunks
            if chunk["family"] == family
        )
        audited_family = tuple(
            {name: record[name] for name in binding_fields}
            for record in audit_sources_by_family[family]
        )
        if combined_family != audited_family:
            raise RuntimeError(
                f"combined {family} source order/identity differs from its audit"
            )


def _terminal_reauthenticate_release_parents(
    *,
    combined: CombinedBinding,
    audits: Mapping[str, Artifact],
    audit_sources_by_family: Mapping[
        str,
        Sequence[Mapping[str, object]],
    ],
) -> None:
    """Freshly rehash the cheap release parents after all deep authentication."""

    expected_by_path: dict[Path, tuple[Artifact, str]] = {}

    def remember(expected: Artifact, *, context: str) -> None:
        previous = expected_by_path.get(expected.path)
        if previous is not None and previous[0] != expected:
            raise RuntimeError(f"{context} conflicts with another release parent")
        expected_by_path[expected.path] = (expected, context)

    source_count = 0
    for family in FAMILY_ORDER:
        for index, source in enumerate(audit_sources_by_family[family]):
            summary = Artifact(
                path=Path(str(source["summary_path"])),
                bytes=_integer(
                    source["summary_bytes"],
                    context=f"final {family} source {index} summary bytes",
                ),
                sha256=_digest(
                    source["summary_sha256"],
                    context=f"final {family} source {index} summary SHA-256",
                ),
            )
            remember(
                summary,
                context=f"final {family} source {index} summary",
            )
            for label in ("manifest", "trajectory_map"):
                expected_path = Path(str(source[f"{label}_path"]))
                expected = Artifact(
                    path=expected_path,
                    bytes=_integer(
                        source[f"{label}_bytes"],
                        context=f"final {family} source {index} {label} bytes",
                    ),
                    sha256=_digest(
                        source[f"{label}_sha256"],
                        context=f"final {family} source {index} {label} SHA-256",
                    ),
                )
                remember(
                    expected,
                    context=f"final {family} source {index} {label}",
                )
            source_count += 1
    if source_count != FINAL_SOURCE_COUNT:
        raise RuntimeError("final source-summary parent sweep is incomplete")
    for name, expected in sorted(audits.items()):
        remember(expected, context=f"final audit {name}")
    for name, expected in (
        ("manifest", combined.manifest),
        ("trajectory map", combined.trajectory_map),
        ("summary", combined.summary),
    ):
        remember(expected, context=f"final combined {name}")
    for expected, context in expected_by_path.values():
        observed = _artifact(expected.path, expected_sha256=expected.sha256)
        if observed != expected:
            raise RuntimeError(f"{context} changed during release authentication")


def authenticate_release(combined_path: Path, audits: AuditPaths) -> ReleaseBinding:
    """Authenticate all release inputs and bind each audit to combined sources."""

    combined = authenticate_combined_summary(combined_path)
    specs = (
        FamilyAuditSpec(
            "stokes",
            "paper_dataset_stokes_revision2_completion_binding_v1",
            audits.stokes_binding,
            EXPECTED_ROWS_BY_FAMILY["stokes"],
            nested_counts=True,
        ),
        FamilyAuditSpec(
            "tanaka",
            "paper_dataset_tanaka_revision3_completion_audit_v1",
            audits.tanaka,
            EXPECTED_ROWS_BY_FAMILY["tanaka"],
        ),
        FamilyAuditSpec(
            "benjamin_feir",
            "paper_dataset_benjamin_feir_revision4_completion_audit_v1",
            audits.benjamin_feir,
            EXPECTED_ROWS_BY_FAMILY["benjamin_feir"],
        ),
        FamilyAuditSpec(
            "jonswap_tma",
            "paper_dataset_jonswap_tma_revision4_completion_audit_v1",
            audits.jonswap_tma,
            EXPECTED_ROWS_BY_FAMILY["jonswap_tma"],
        ),
    )
    artifacts: dict[str, Artifact] = {}
    historical_source_snapshots: dict[str, dict[str, Artifact]] = {}
    audit_sources_by_family: dict[str, tuple[dict[str, object], ...]] = {}
    all_audit_source_paths: set[Path] = set()
    for spec in specs:
        artifact, bindings, historical_artifacts = _validate_family_audit(spec)
        artifacts[spec.name] = artifact
        if spec.name in {"stokes", "benjamin_feir"}:
            historical_source_snapshots[spec.name] = historical_artifacts
        elif historical_artifacts:
            raise RuntimeError(
                "non-Stokes/BF audit returned historical source snapshots"
            )
        audit_sources_by_family[spec.name] = bindings
        for source_binding in bindings:
            source_path = Path(str(source_binding["summary_path"]))
            if source_path in all_audit_source_paths:
                raise RuntimeError("family audits repeat a source summary")
            all_audit_source_paths.add(source_path)
    legacy = _validate_stokes_legacy(
        artifacts["stokes"].path,
        audits.stokes_legacy,
    )
    artifacts["stokes_legacy"] = legacy

    _cross_bind_audit_sources(combined.chunks, audit_sources_by_family)
    expected_snapshot_artifacts = {
        "stokes": {"SHA256SUMS", *STOKES_HISTORICAL_SOURCE_SNAPSHOTS},
        "benjamin_feir": {"SHA256SUMS", *BF_HISTORICAL_SOURCE_SNAPSHOTS},
    }
    if set(historical_source_snapshots) != set(expected_snapshot_artifacts) or any(
        set(historical_source_snapshots[family]) != expected
        for family, expected in expected_snapshot_artifacts.items()
    ):
        raise RuntimeError("historical source release artifacts are incomplete")
    if sum(map(len, historical_source_snapshots.values())) != 7:
        raise RuntimeError("historical source release requires exactly seven artifacts")
    dependency, generation_sources = _authenticate_release_generation_identity(
        combined=combined,
        audits=artifacts,
        audit_sources_by_family=audit_sources_by_family,
    )
    _terminal_reauthenticate_artifact_groups(
        _release_terminal_artifact_groups(
            dependency=dependency,
            generation_sources=generation_sources,
            historical_source_snapshots=historical_source_snapshots,
        )
    )
    _terminal_reauthenticate_release_parents(
        combined=combined,
        audits=artifacts,
        audit_sources_by_family=audit_sources_by_family,
    )
    return ReleaseBinding(
        combined=combined,
        audits=artifacts,
        dependency=dependency,
        generation_sources=generation_sources,
        historical_source_snapshots=historical_source_snapshots,
    )


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Durably replace a mutable runner record with finite JSON."""

    destination = path.expanduser().absolute()
    if destination.is_symlink():
        raise FileExistsError(f"refusing symbolic-link output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    """Create an immutable identity or require byte-independent JSON equality."""

    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ExistingOutputError(f"identity path is not a regular file: {path}")
        if not _same_json(_strict_json(path), payload):
            raise ExistingOutputError(
                "SHA-keyed output root is bound to different release evidence"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise ExistingOutputError(f"identity appeared concurrently: {path}")
        os.link(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


class RunJournal:
    """Atomic current status plus append-only, fsynced JSON event history."""

    def __init__(self, root: Path, invocation_id: str, base: Mapping[str, object]):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.status_path = root / "status.json"
        self.events_path = root / "events.jsonl"
        self.record: dict[str, Any] = {
            "schema": RUNNER_SCHEMA,
            "status": "running",
            "invocation_id": invocation_id,
            "started_at": datetime.now().astimezone().isoformat(),
            "updated_at": datetime.now().astimezone().isoformat(),
            "release_artifacts_mutated": False,
            "stages": {},
            **base,
        }
        self._publish()

    def _publish(self) -> None:
        self.record["updated_at"] = datetime.now().astimezone().isoformat()
        _write_json_atomic(self.status_path, self.record)

    def event(self, event: str, **fields: object) -> None:
        entry = {
            "at": datetime.now().astimezone().isoformat(),
            "invocation_id": self.record["invocation_id"],
            "event": event,
            **fields,
        }
        encoded = json.dumps(entry, sort_keys=True, allow_nan=False) + "\n"
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self._publish()

    def stage(self, name: str, value: Mapping[str, object]) -> None:
        stages = dict(_mapping(self.record.get("stages"), context="runner stages"))
        stages[name] = dict(value)
        self.record["stages"] = stages
        self._publish()

    def finish(self, status: str) -> None:
        self.record["status"] = status
        self.record["finished_at"] = datetime.now().astimezone().isoformat()
        self.event("runner_finished", status=status)


def wait_for_inputs(
    paths: Sequence[Path],
    *,
    poll_seconds: float,
    timeout_seconds: float | None,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    on_poll: Callable[[tuple[Path, ...]], None] | None = None,
) -> None:
    """Wait until every input exists, polling no less often than every minute."""

    if not 0.0 <= poll_seconds <= 60.0:
        raise ValueError("poll interval must be between 0 and 60 seconds")
    started = monotonic()
    while True:
        missing = tuple(path for path in paths if not path.is_file())
        if not missing:
            return
        if on_poll is not None:
            on_poll(missing)
        elapsed = monotonic() - started
        if timeout_seconds is not None and elapsed >= timeout_seconds:
            names = ", ".join(str(path) for path in missing)
            raise TimeoutError(f"timed out waiting for: {names}")
        delay = poll_seconds
        if timeout_seconds is not None:
            delay = min(delay, max(0.0, timeout_seconds - elapsed))
        sleeper(delay)


def _canonical_recorded_path(
    value: object,
    *,
    expected_path: Path,
    context: str,
) -> Path:
    """Require the producer's literal canonical absolute path spelling."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} path must be a nonempty string")
    requested = Path(value).expanduser()
    canonical_expected = expected_path.expanduser().resolve(strict=True)
    if (
        not requested.is_absolute()
        or requested.is_symlink()
        or value != str(canonical_expected)
        or requested.resolve(strict=True) != canonical_expected
    ):
        raise RuntimeError(f"{context} path differs")
    return canonical_expected


def _canonical_existing_recorded_path(value: object, *, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} path must be a nonempty string")
    return _canonical_recorded_path(
        value,
        expected_path=Path(value),
        context=context,
    )


def _verify_record_artifact(
    record: object,
    *,
    expected_path: Path,
    context: str,
) -> Artifact:
    value = _mapping(record, context=context)
    _require_exact_keys(
        value,
        {"path", "bytes", "sha256"},
        context=context,
    )
    path_value = value.get("path")
    canonical_expected = _canonical_recorded_path(
        path_value,
        expected_path=expected_path,
        context=context,
    )
    artifact = _artifact(canonical_expected, expected_sha256=value.get("sha256"))
    if artifact.bytes != _integer(value.get("bytes"), context=f"{context} bytes"):
        raise RuntimeError(f"{context} byte count differs")
    return artifact


def _reauthenticate_expected_artifact(expected: Artifact, *, context: str) -> None:
    observed = _artifact(expected.path, expected_sha256=expected.sha256)
    if observed != expected:
        raise RuntimeError(f"{context} byte identity differs")


def _release_source_records(release: ReleaseBinding) -> list[dict[str, object]]:
    """Return the renderer's exact source records from authenticated release data."""

    if len(release.combined.chunks) != FINAL_SOURCE_COUNT:
        raise RuntimeError("release does not contain the exact source population")
    parents: dict[Path, tuple[str, str, str, str]] = {}
    for family in FAMILY_ORDER:
        audit_artifact = release.audits[family]
        _reauthenticate_expected_artifact(
            audit_artifact, context=f"renderer {family} completion audit"
        )
        audit = _strict_json(audit_artifact.path)
        chunks = _sequence(audit.get("chunks"), context=f"{family} audit chunks")
        for index, raw_chunk in enumerate(chunks):
            chunk = _mapping(raw_chunk, context=f"{family} audit chunk {index}")
            summary_path = _canonical_existing_recorded_path(
                chunk.get("summary_path"),
                context=f"{family} audit chunk {index} summary",
            )
            if summary_path in parents:
                raise RuntimeError("release audits repeat a source summary")
            manifest_path = _canonical_existing_recorded_path(
                chunk.get("manifest_path"),
                context=f"{family} audit chunk {index} manifest",
            )
            map_path = _canonical_existing_recorded_path(
                chunk.get("trajectory_map_path"),
                context=f"{family} audit chunk {index} trajectory map",
            )
            summary_sha256 = _digest(
                chunk.get("summary_sha256"),
                context=f"{family} audit chunk {index} summary SHA-256",
            )
            manifest_sha256 = _digest(
                chunk.get("manifest_sha256"),
                context=f"{family} audit chunk {index} manifest SHA-256",
            )
            map_sha256 = _digest(
                chunk.get("trajectory_map_sha256"),
                context=f"{family} audit chunk {index} map SHA-256",
            )
            _artifact(summary_path, expected_sha256=summary_sha256)
            _artifact(manifest_path, expected_sha256=manifest_sha256)
            _artifact(map_path, expected_sha256=map_sha256)
            parents[summary_path] = (
                summary_sha256,
                manifest_sha256,
                map_sha256,
                str(chunk.get("split")),
            )
    if len(parents) != FINAL_SOURCE_COUNT:
        raise RuntimeError("release audits do not bind exactly 26 source parents")

    records: list[dict[str, object]] = []
    for index, chunk in enumerate(release.combined.chunks):
        context = f"release source {index}"
        family = chunk.get("family")
        split = chunk.get("split")
        if family not in FAMILY_ORDER or split not in SPLIT_ORDER:
            raise RuntimeError(f"{context} family or split differs")
        summary_path = _canonical_existing_recorded_path(
            chunk.get("summary_path"), context=f"{context} summary"
        )
        parent = parents.get(summary_path)
        if parent is None:
            raise RuntimeError(f"{context} is absent from its family audit")
        summary_sha256, manifest_sha256, map_sha256, audit_split = parent
        accepted = _integer(
            chunk.get("accepted_count"), context=f"{context} accepted cases"
        )
        if audit_split != split or chunk.get("summary_sha256") != summary_sha256:
            raise RuntimeError(f"{context} differs between release and audit")
        records.append(
            {
                "root": str(summary_path.parent),
                "family": family,
                "split": split,
                "accepted_cases": accepted,
                "retained_rows": accepted * ROWS_PER_CASE[str(family)],
                "summary_sha256": summary_sha256,
                "manifest_sha256": manifest_sha256,
                "trajectory_map_sha256": map_sha256,
            }
        )
    if len({record["root"] for record in records}) != FINAL_SOURCE_COUNT:
        raise RuntimeError("release source roots are not unique")
    return records


def _renderer_source_proofs(
    release: ReleaseBinding,
) -> list[dict[str, object]]:
    """Recover exact accepted ownership metadata without rescanning state fields."""

    if __package__:
        from scripts import render_paper_dataset_worst_cases as producer
    else:
        import render_paper_dataset_worst_cases as producer

    parent_paths: dict[Path, tuple[Path, Path]] = {}
    for family in FAMILY_ORDER:
        audit_artifact = release.audits[family]
        _reauthenticate_expected_artifact(
            audit_artifact, context=f"renderer {family} source-proof audit"
        )
        audit = _strict_json(audit_artifact.path)
        for index, raw_chunk in enumerate(
            _sequence(audit.get("chunks"), context=f"{family} audit chunks")
        ):
            chunk = _mapping(raw_chunk, context=f"{family} audit chunk {index}")
            summary_path = _canonical_existing_recorded_path(
                chunk.get("summary_path"),
                context=f"renderer {family} source {index} summary",
            )
            parent_paths[summary_path] = (
                _canonical_existing_recorded_path(
                    chunk.get("manifest_path"),
                    context=f"renderer {family} source {index} manifest",
                ),
                _canonical_existing_recorded_path(
                    chunk.get("trajectory_map_path"),
                    context=f"renderer {family} source {index} trajectory map",
                ),
            )
    proofs: list[dict[str, object]] = []
    for index, chunk in enumerate(release.combined.chunks):
        summary_path = _canonical_existing_recorded_path(
            chunk.get("summary_path"), context=f"renderer source {index} summary"
        )
        paths = parent_paths.get(summary_path)
        if paths is None:
            raise RuntimeError(f"renderer source {index} parent paths are missing")
        manifest_path, map_path = paths
        summary = _strict_json(summary_path)
        manifest = _strict_json(manifest_path)
        trajectories = producer._trajectory_indices(summary, map_path)
        raw_shards = _sequence(
            manifest.get("dataset_shards"),
            context=f"renderer source {index} shards",
        )
        shards: list[tuple[Path, str]] = []
        for shard_index, raw_shard in enumerate(raw_shards):
            shard = _mapping(
                raw_shard, context=f"renderer source {index} shard {shard_index}"
            )
            raw_path = shard.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(
                    f"renderer source {index} shard {shard_index} path differs"
                )
            requested = summary_path.parent / raw_path
            if requested.is_symlink():
                raise ValueError("renderer source shard must not be a symbolic link")
            path = requested.resolve(strict=True)
            if not path.is_relative_to(summary_path.parent):
                raise ValueError("renderer source shard escapes its source root")
            shards.append(
                (
                    path,
                    _digest(
                        shard.get("sha256"),
                        context=(
                            f"renderer source {index} shard {shard_index} SHA-256"
                        ),
                    ),
                )
            )
        proofs.append(
            {
                "summary_path": summary_path,
                "manifest_path": manifest_path,
                "map_path": map_path,
                "summary": summary,
                "trajectories": trajectories,
                "shards": tuple(shards),
                "depths": {},
            }
        )
    return proofs


def _renderer_case_depth(
    proof: Mapping[str, object],
    *,
    shard_index: int,
    first_shard_row: int,
    row_count: int,
    context: str,
) -> float:
    shards = _sequence(proof.get("shards"), context=f"{context} shards")
    if shard_index >= len(shards):
        raise RuntimeError(f"{context} shard index is outside its manifest")
    depths = proof.get("depths")
    if not isinstance(depths, dict):
        raise TypeError(f"{context} depth cache differs")
    values = depths.get(shard_index)
    if values is None:
        raw_shard = shards[shard_index]
        if not isinstance(raw_shard, tuple) or len(raw_shard) != 2:
            raise TypeError(f"{context} shard proof differs")
        path = raw_shard[0]
        digest = raw_shard[1]
        if not isinstance(path, Path) or not isinstance(digest, str):
            raise TypeError(f"{context} shard identity differs")
        _artifact(path, expected_sha256=digest)
        with np.load(path, allow_pickle=False) as archive:
            if "depth" not in archive.files:
                raise RuntimeError(f"{context} shard omits depth")
            values = np.asarray(archive["depth"], dtype=np.float64)
        depths[shard_index] = values
    if not isinstance(values, np.ndarray) or values.ndim != 1:
        raise RuntimeError(f"{context} shard depth shape differs")
    stop = first_shard_row + row_count
    if first_shard_row < 0 or stop > values.size:
        raise RuntimeError(f"{context} rows are outside the selected shard")
    selected = values[first_shard_row:stop]
    if not np.all(np.isfinite(selected)) or not np.all(selected == selected[0]):
        raise RuntimeError(f"{context} depth is not finite and constant")
    return float(selected[0])


def _validate_renderer_case(
    value: object,
    *,
    family: str,
    source_records: Sequence[Mapping[str, object]],
    source_proofs: Sequence[Mapping[str, object]],
    context: str,
) -> tuple[tuple[object, ...], dict[str, object]]:
    """Validate one complete renderer case and return its canonical identity."""

    record = _mapping(value, context=context)
    _require_exact_keys(
        record,
        {
            "source_index",
            "accepted_index",
            "trajectory_index",
            "family",
            "split",
            "case_id",
            "category",
            "shard_index",
            "first_shard_row",
            "row_count",
            "depth",
            "all_frames_finite",
            "constant_depth",
            "ordered_time",
            "minimum_water_column",
            "minimum_water_fraction",
            "maximum_eta_slope",
            "maximum_eta_slope_frame",
            "maximum_gxi_high_band_fraction",
            "maximum_gxi_high_band_fraction_frame",
            "maximum_thresholded_gxi_sign_changes",
            "maximum_thresholded_gxi_sign_changes_frame",
            "maximum_relative_stored_band_quadratic_energy_drift",
            "maximum_relative_stored_band_quadratic_energy_drift_frame",
            "source_root",
            "combined_empirical_rank",
        },
        context=context,
    )
    source_index = _integer(record.get("source_index"), context=f"{context} source")
    if source_index >= len(source_records):
        raise RuntimeError(f"{context} source index is outside the final release")
    source = source_records[source_index]
    proof = source_proofs[source_index]
    split = record.get("split")
    category = record.get("category")
    if (
        record.get("family") != family
        or source.get("family") != family
        or split != source.get("split")
        or split not in SPLIT_ORDER
        or not isinstance(category, str)
        or not category
        or record.get("source_root") != source.get("root")
    ):
        raise RuntimeError(f"{context} source order/identity differs")
    accepted_index = _integer(
        record.get("accepted_index"), context=f"{context} accepted index"
    )
    if accepted_index >= _integer(
        source.get("accepted_cases"), context=f"{context} source cases"
    ):
        raise RuntimeError(f"{context} accepted index is outside its source")
    trajectories = _sequence(
        proof.get("trajectories"), context=f"{context} source trajectories"
    )
    if accepted_index >= len(trajectories):
        raise RuntimeError(f"{context} accepted index is absent from its source map")
    trajectory = trajectories[accepted_index]
    integer_fields = (
        "trajectory_index",
        "case_id",
        "shard_index",
        "first_shard_row",
        "maximum_eta_slope_frame",
        "maximum_gxi_high_band_fraction_frame",
        "maximum_thresholded_gxi_sign_changes",
        "maximum_thresholded_gxi_sign_changes_frame",
        "maximum_relative_stored_band_quadratic_energy_drift_frame",
    )
    integers = {
        name: _integer(record.get(name), context=f"{context} {name}")
        for name in integer_fields
    }
    row_count = _integer(record.get("row_count"), context=f"{context} row count")
    if row_count != ROWS_PER_CASE[family]:
        raise RuntimeError(f"{context} row count differs")
    expected_ownership = {
        "accepted_index": trajectory.accepted_index,
        "trajectory_index": trajectory.trajectory_index,
        "case_id": trajectory.case_id,
        "category": trajectory.category,
        "shard_index": trajectory.shard_index,
        "first_shard_row": trajectory.first_shard_row,
        "row_count": trajectory.row_count,
    }
    if any(
        record.get(name) != expected for name, expected in expected_ownership.items()
    ):
        raise RuntimeError(f"{context} ownership differs from its trajectory map")
    for name in (
        "maximum_eta_slope_frame",
        "maximum_gxi_high_band_fraction_frame",
        "maximum_thresholded_gxi_sign_changes_frame",
        "maximum_relative_stored_band_quadratic_energy_drift_frame",
    ):
        if integers[name] >= row_count:
            raise RuntimeError(f"{context} {name} is outside the trajectory")
    if any(
        record.get(name) is not True
        for name in ("all_frames_finite", "constant_depth", "ordered_time")
    ):
        raise RuntimeError(f"{context} hard trajectory checks are not true")
    numeric = {
        name: _finite_number(record.get(name), context=f"{context} {name}")
        for name in (
            "depth",
            "minimum_water_column",
            "minimum_water_fraction",
            "maximum_eta_slope",
            "maximum_gxi_high_band_fraction",
            "maximum_relative_stored_band_quadratic_energy_drift",
            "combined_empirical_rank",
        )
    }
    if (
        numeric["depth"] <= 0.0
        or numeric["minimum_water_column"] <= 0.0
        or numeric["minimum_water_fraction"] <= 0.0
        or numeric["minimum_water_fraction"]
        != numeric["minimum_water_column"] / numeric["depth"]
        or numeric["maximum_eta_slope"] < 0.0
        or not 0.0 <= numeric["maximum_gxi_high_band_fraction"] <= 1.0
        or numeric["maximum_relative_stored_band_quadratic_energy_drift"] < 0.0
        or not 0.0 <= numeric["combined_empirical_rank"] <= 1.0
        or integers["maximum_thresholded_gxi_sign_changes"] > 1_024
    ):
        raise RuntimeError(f"{context} metric range differs")
    expected_depth = _renderer_case_depth(
        proof,
        shard_index=integers["shard_index"],
        first_shard_row=integers["first_shard_row"],
        row_count=row_count,
        context=context,
    )
    if numeric["depth"] != expected_depth:
        raise RuntimeError(f"{context} depth differs from its selected shard")
    identity = (
        source_index,
        accepted_index,
        integers["trajectory_index"],
        family,
        split,
        integers["case_id"],
        category,
    )
    return identity, dict(record)


def validate_renderer_output(
    output_dir: Path, release: ReleaseBinding
) -> Mapping[str, object]:
    combined = release.combined.summary
    _reauthenticate_expected_artifact(combined, context="renderer combined summary")
    summary_path = output_dir / "summary.json"
    summary = _strict_json(summary_path)
    _require_exact_keys(
        summary,
        {
            "schema",
            "status",
            "interpretation",
            "source_binding",
            "parameters",
            "definitions",
            "population",
            "sources",
            "families",
            "renderer_implementation",
            "animations",
            "artifacts",
        },
        context="renderer summary",
    )
    if summary.get("schema") != RENDERER_SCHEMA or summary.get("status") != "complete":
        raise RuntimeError("renderer output is not a complete v3 diagnostic")
    binding = _mapping(summary.get("source_binding"), context="renderer binding")
    _require_exact_keys(
        binding,
        {
            "mode",
            "combined_summary_path",
            "combined_summary_sha256",
            "expected_sources",
            "expected_accepted_cases",
            "expected_retained_rows",
        },
        context="renderer binding",
    )
    if (
        binding.get("mode") != "combined_summary"
        or binding.get("combined_summary_sha256") != combined.sha256
        or binding.get("expected_sources") != FINAL_SOURCE_COUNT
        or binding.get("expected_accepted_cases") != FINAL_ACCEPTED_CASES
        or binding.get("expected_retained_rows") != FINAL_RETAINED_ROWS
    ):
        raise RuntimeError("renderer output is bound to a different release")
    _canonical_recorded_path(
        binding.get("combined_summary_path"),
        expected_path=combined.path,
        context="renderer combined summary",
    )
    if __package__:
        from scripts import render_paper_dataset_worst_cases as producer
    else:
        import render_paper_dataset_worst_cases as producer

    if summary.get("interpretation") != producer.INTERPRETATION:
        raise RuntimeError("renderer interpretation differs from the producer schema")
    parameters = _mapping(summary.get("parameters"), context="renderer parameters")
    if not _same_json(parameters, producer.PARAMETERS):
        raise RuntimeError("renderer parameters differ from the producer schema")
    definitions = _mapping(summary.get("definitions"), context="renderer definitions")
    if not _same_json(definitions, producer.DEFINITIONS):
        raise RuntimeError("renderer definitions differ from the producer schema")
    population = _mapping(summary.get("population"), context="renderer population")
    _require_exact_keys(
        population,
        {"sources", "accepted_cases", "retained_rows"},
        context="renderer population",
    )
    if (
        population.get("sources"),
        population.get("accepted_cases"),
        population.get("retained_rows"),
    ) != (FINAL_SOURCE_COUNT, FINAL_ACCEPTED_CASES, FINAL_RETAINED_ROWS):
        raise RuntimeError("renderer population differs from final dataset")
    source_records = _release_source_records(release)
    if not _same_json(summary.get("sources"), source_records):
        raise RuntimeError("renderer source records differ from the exact release")
    source_proofs = _renderer_source_proofs(release)
    families = _mapping(summary.get("families"), context="renderer families")
    if set(families) != set(FAMILY_ORDER):
        raise RuntimeError("renderer family set differs")
    rank_one_cases: dict[tuple[str, str], dict[str, object]] = {}
    for family in FAMILY_ORDER:
        family_record = _mapping(families[family], context=f"renderer {family}")
        _require_exact_keys(
            family_record,
            {"accepted_cases", "retained_rows", "splits", "quantiles", "rankings"},
            context=f"renderer {family}",
        )
        if (
            family_record.get("accepted_cases") != 18_432
            or family_record.get("retained_rows") != EXPECTED_ROWS_BY_FAMILY[family]
            or not _same_json(
                family_record.get("splits"),
                EXPECTED_SPLIT_COUNTS,
            )
        ):
            raise RuntimeError(f"renderer {family} counts differ")
        rankings = _mapping(
            family_record.get("rankings"), context=f"renderer {family} rankings"
        )
        if set(rankings) != set(RENDERER_RANKINGS) or any(
            len(_sequence(ranking, context=f"renderer {family} ranking {name}"))
            != RENDERER_TOP_COUNT
            for name, ranking in rankings.items()
        ):
            raise RuntimeError(f"renderer {family} rankings differ")
        quantiles = _mapping(
            family_record.get("quantiles"), context=f"renderer {family} quantiles"
        )
        if set(quantiles) != {
            "eta_slope",
            "gxi_high_band",
            "gxi_sign_changes",
            "stored_band_quadratic_energy_drift",
            "minimum_water_fraction",
        }:
            raise RuntimeError(f"renderer {family} quantile metrics differ")
        quantile_keys = ("q000", "q050", "q090", "q095", "q099", "q100")
        quantile_bounds: dict[str, tuple[float, float]] = {}
        for metric, raw_quantiles in quantiles.items():
            metric_quantiles = _mapping(
                raw_quantiles, context=f"renderer {family} {metric} quantiles"
            )
            if tuple(metric_quantiles) != quantile_keys:
                raise RuntimeError(f"renderer {family} {metric} quantiles differ")
            values = [
                _finite_number(
                    metric_quantiles[key],
                    context=f"renderer {family} {metric} {key}",
                )
                for key in quantile_keys
            ]
            if any(left > right for left, right in zip(values, values[1:])):
                raise RuntimeError(
                    f"renderer {family} {metric} quantiles are not monotone"
                )
            if (
                (
                    metric
                    in {
                        "eta_slope",
                        "stored_band_quadratic_energy_drift",
                    }
                    and values[0] < 0.0
                )
                or (
                    metric == "gxi_sign_changes"
                    and (values[0] < 0.0 or values[-1] > 1_024.0)
                )
                or (
                    metric == "gxi_high_band"
                    and (values[0] < 0.0 or values[-1] > 1.0)
                )
                or (metric == "minimum_water_fraction" and values[0] <= 0.0)
            ):
                raise RuntimeError(f"renderer {family} {metric} quantile range differs")
            quantile_bounds[metric] = (values[0], values[-1])
        repeated: dict[tuple[object, ...], dict[str, object]] = {}
        for ranking_name, raw_ranking in rankings.items():
            ranking = _sequence(
                raw_ranking, context=f"renderer {family} ranking {ranking_name}"
            )
            seen: set[tuple[object, ...]] = set()
            simple_case_ids: set[tuple[object, ...]] = set()
            canonical_cases = []
            for index, raw_case in enumerate(ranking):
                identity, canonical = _validate_renderer_case(
                    raw_case,
                    family=family,
                    source_records=source_records,
                    source_proofs=source_proofs,
                    context=f"renderer {family} {ranking_name} case {index}",
                )
                simple_identity = (
                    canonical["family"],
                    canonical["split"],
                    canonical["case_id"],
                )
                if identity in seen or simple_identity in simple_case_ids:
                    raise RuntimeError(
                        f"renderer {family} {ranking_name} repeats a case identity"
                    )
                seen.add(identity)
                simple_case_ids.add(simple_identity)
                prior = repeated.get(identity)
                if prior is not None and not _same_json(prior, canonical):
                    raise RuntimeError(
                        f"renderer {family} case identity has conflicting records"
                    )
                repeated[identity] = canonical
                canonical_cases.append(canonical)
            bounded_metrics = {
                "eta_slope": "maximum_eta_slope",
                "gxi_high_band": "maximum_gxi_high_band_fraction",
                "gxi_sign_changes": "maximum_thresholded_gxi_sign_changes",
                "stored_band_quadratic_energy_drift": (
                    "maximum_relative_stored_band_quadratic_energy_drift"
                ),
                "minimum_water_fraction": "minimum_water_fraction",
            }
            if any(
                not quantile_bounds[metric][0]
                <= float(case[field])
                <= quantile_bounds[metric][1]
                for case in canonical_cases
                for metric, field in bounded_metrics.items()
            ):
                raise RuntimeError(
                    f"renderer {family} ranked case lies outside its quantiles"
                )
            metric_field = {
                "combined": "combined_empirical_rank",
                "eta_slope": "maximum_eta_slope",
                "gxi_high_band": "maximum_gxi_high_band_fraction",
                "gxi_sign_changes": "maximum_thresholded_gxi_sign_changes",
                "stored_band_quadratic_energy_drift": (
                    "maximum_relative_stored_band_quadratic_energy_drift"
                ),
            }[ranking_name]
            metric_values = [float(case[metric_field]) for case in canonical_cases]
            if any(
                left < right for left, right in zip(metric_values, metric_values[1:])
            ):
                raise RuntimeError(
                    f"renderer {family} {ranking_name} ranking order differs"
                )
            if (
                ranking_name != "combined"
                and metric_values[0] != quantile_bounds[ranking_name][1]
            ):
                raise RuntimeError(
                    f"renderer {family} {ranking_name} maximum differs from q100"
                )
            rank_one_cases[(family, ranking_name)] = canonical_cases[0]
    renderer_implementation = validate_renderer_implementation(
        summary.get("renderer_implementation")
    )
    animations = _mapping(summary.get("animations"), context="renderer animations")
    expected_animations = {
        f"{family}_worst_{ranking}.gif"
        for family in FAMILY_ORDER
        for ranking in RENDERER_RANKINGS
    }
    if set(animations) != expected_animations:
        raise RuntimeError("renderer animation set differs from the final contract")
    artifacts = _mapping(summary.get("artifacts"), context="renderer artifacts")
    expected_artifacts = {
        f"{family}_worst_{ranking}.{suffix}"
        for family in FAMILY_ORDER
        for ranking in RENDERER_RANKINGS
        for suffix in ("gif", "pdf", "png")
    } | {f"all_families_worst_overview.{suffix}" for suffix in ("pdf", "png")}
    if set(artifacts) != expected_artifacts:
        raise RuntimeError("renderer artifact set differs from the final contract")
    for name, raw_record in artifacts.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("renderer artifact names must be plain filenames")
        _verify_record_artifact(
            raw_record,
            expected_path=output_dir / name,
            context=f"renderer artifact {name}",
        )
    for name, raw_animation in sorted(animations.items()):
        animation = _mapping(raw_animation, context=f"renderer animation {name}")
        _require_exact_keys(
            animation,
            {
                "family",
                "ranking",
                "rank",
                "source_index",
                "accepted_index",
                "trajectory_index",
                "case_id",
                "category",
                "split",
                "stored_frames",
                "frame_indices",
                "first_time",
                "last_time",
                "fps",
                "dimensions_pixels",
                "field_y_limits",
            },
            context=f"renderer animation {name}",
        )
        family = animation.get("family")
        ranking = animation.get("ranking")
        if not isinstance(family, str) or not isinstance(ranking, str):
            raise TypeError(f"renderer animation {name} identity differs")
        case = rank_one_cases.get((family, ranking))
        if (
            case is None
            or name != f"{family}_worst_{ranking}.gif"
            or animation.get("rank") != 1
            or any(
                animation.get(field) != case[field]
                for field in (
                    "source_index",
                    "accepted_index",
                    "trajectory_index",
                    "case_id",
                    "category",
                    "split",
                )
            )
        ):
            raise RuntimeError(f"renderer animation {name} rank-one identity differs")
        source_index = _integer(
            animation.get("source_index"), context=f"renderer animation {name} source"
        )
        proof = source_proofs[source_index]
        shards = _sequence(
            proof.get("shards"), context=f"renderer animation {name} shards"
        )
        shard_index = _integer(
            case["shard_index"], context=f"renderer animation {name} shard"
        )
        raw_shard = shards[shard_index]
        if not isinstance(raw_shard, tuple) or len(raw_shard) != 2:
            raise TypeError(f"renderer animation {name} shard proof differs")
        shard_path, shard_sha256 = raw_shard
        if not isinstance(shard_path, Path) or not isinstance(shard_sha256, str):
            raise TypeError(f"renderer animation {name} shard identity differs")
        _artifact(shard_path, expected_sha256=shard_sha256)
        first = _integer(
            case["first_shard_row"], context=f"renderer animation {name} first row"
        )
        stored_frames = _integer(
            animation.get("stored_frames"),
            context=f"renderer animation {name} stored frames",
            minimum=1,
        )
        if stored_frames != _integer(
            case["row_count"], context=f"renderer animation {name} row count"
        ):
            raise RuntimeError(f"renderer animation {name} frame count differs")
        with np.load(shard_path, allow_pickle=False) as archive:
            arrays = {
                field: np.asarray(archive[field][first : first + stored_frames])
                for field in (*producer.FIELD_NAMES, "time")
            }
        if any(not np.all(np.isfinite(values)) for values in arrays.values()):
            raise RuntimeError(f"renderer animation {name} source is nonfinite")
        expected_animation = producer.animation_record(
            producer.LoadedTrajectory(
                eta=np.asarray(arrays["eta"], dtype=np.float64),
                xi=np.asarray(arrays["xi"], dtype=np.float64),
                gxi=np.asarray(arrays["gxi"], dtype=np.float64),
                depth=np.full(stored_frames, float(case["depth"])),
                time=np.asarray(arrays["time"], dtype=np.float64),
            )
        )
        for record_field, expected in expected_animation.items():
            if not _same_json(animation.get(record_field), expected):
                raise RuntimeError(
                    f"renderer animation {name} {record_field} differs from its source"
                )
        gif_path = output_dir / name
        with Image.open(gif_path) as image:
            expected_frames = len(expected_animation["frame_indices"])
            expected_duration = (
                250 if expected_animation["fps"] == producer.GIF_SHORT_FPS else 80
            )
            if (
                image.format != "GIF"
                or image.info.get("version") != b"GIF89a"
                or tuple(map(int, image.size)) != producer.GIF_DIMENSIONS
                or int(getattr(image, "n_frames", 1)) != expected_frames
                or image.info.get("loop") != 0
            ):
                raise RuntimeError(f"renderer animation {name} GIF contract differs")
            for frame in range(expected_frames):
                image.seek(frame)
                image.load()
                if image.info.get("duration") != expected_duration:
                    raise RuntimeError(
                        f"renderer animation {name} GIF contract differs"
                    )
    return {
        "summary": _artifact(summary_path).record(),
        "artifacts": len(artifacts),
        "renderer_implementation": renderer_implementation,
    }


def _validated_order_series(
    record: Mapping[str, object],
    *,
    series_name: str,
    maximum_name: str,
    context: str,
    upper_bound: float | None = None,
) -> list[float]:
    values = _sequence(record.get(series_name), context=f"{context} {series_name}")
    if len(values) != 16:
        raise RuntimeError(f"{context} {series_name} is incomplete")
    numbers = [
        _finite_number(value, context=f"{context} {series_name}[{index}]")
        for index, value in enumerate(values)
    ]
    if any(value < 0.0 for value in numbers) or (
        upper_bound is not None and any(value > upper_bound for value in numbers)
    ):
        raise RuntimeError(f"{context} {series_name} range differs")
    maximum = _finite_number(record.get(maximum_name), context=f"{context} maximum")
    if maximum != max(numbers):
        raise RuntimeError(f"{context} maximum differs from its series")
    return numbers


def _resolved_child_artifact(
    root: Path,
    path_value: object,
    digest_value: object,
    *,
    context: str,
) -> Artifact:
    if not isinstance(path_value, (str, Path)) or not str(path_value):
        raise ValueError(f"{context} path must be a nonempty string")
    requested = Path(path_value).expanduser()
    if not requested.is_absolute():
        requested = root / requested
    if requested.is_symlink():
        raise ValueError(f"{context} must not be a symbolic link")
    path = requested.resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"{context} escapes its source root")
    return _artifact(path, expected_sha256=digest_value)


def validate_order_output(
    output: Path,
    *,
    release: ReleaseBinding,
    renderer_summary: Path,
) -> Mapping[str, object]:
    if __package__:
        from scripts import diagnose_final_jonswap_order_convergence as producer
    else:
        import diagnose_final_jonswap_order_convergence as producer

    combined = release.combined.summary
    _reauthenticate_expected_artifact(combined, context="order combined summary")
    validate_renderer_output(renderer_summary.parent, release)
    renderer = _strict_json(renderer_summary)
    renderer_families = _mapping(
        renderer.get("families"), context="order renderer families"
    )
    renderer_jonswap = _mapping(
        renderer_families.get("jonswap_tma"), context="order renderer JONSWAP"
    )
    renderer_rankings = _mapping(
        renderer_jonswap.get("rankings"), context="order renderer rankings"
    )
    renderer_cases = tuple(
        _mapping(value, context=f"order renderer case {index}")
        for index, value in enumerate(
            _sequence(
                renderer_rankings.get("gxi_high_band"),
                context="order renderer high-band ranking",
            )[:3]
        )
    )
    if len(renderer_cases) != 3:
        raise RuntimeError("order renderer cross-stage selection is incomplete")
    release_source_indices = {
        _canonical_existing_recorded_path(
            chunk.get("summary_path"), context=f"order release source {index} summary"
        ): index
        for index, chunk in enumerate(release.combined.chunks)
    }

    record = _strict_json(output)
    _require_exact_keys(
        record,
        {
            "schema",
            "status",
            "diagnostic_only",
            "affects_dataset_acceptance",
            "affects_dataset_release",
            "completion_audit",
            "selection",
            "numerical_definition",
            "identity",
            "diagnostic_implementation",
            "chunk_bindings",
            "cases",
            "interpretation_scope",
            "timing_seconds",
            "invocation_started_at",
            "invocation_finished_at",
        },
        context="order diagnostic",
    )
    if record.get("schema") != ORDER_SCHEMA or record.get("status") != "pass":
        raise RuntimeError("JONSWAP order diagnostic did not pass")
    if (
        record.get("diagnostic_only") is not True
        or record.get("affects_dataset_acceptance") is not False
        or record.get("affects_dataset_release") is not False
        or record.get("interpretation_scope") != producer.INTERPRETATION_SCOPE
        or _finite_number(record.get("timing_seconds"), context="order timing") < 0.0
    ):
        raise RuntimeError("JONSWAP order diagnostic scope differs")
    timestamps: dict[str, datetime] = {}
    for name in ("invocation_started_at", "invocation_finished_at"):
        value = record.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"order {name} must be a nonempty timestamp")
        parsed = datetime.fromisoformat(value)
        if parsed.utcoffset() is None:
            raise ValueError(f"order {name} must be timezone-aware")
        timestamps[name] = parsed
    if timestamps["invocation_finished_at"] < timestamps["invocation_started_at"]:
        raise RuntimeError("order invocation timestamps are reversed")

    jonswap_audit = release.audits["jonswap_tma"]
    _reauthenticate_expected_artifact(
        jonswap_audit, context="order JONSWAP completion audit"
    )
    audit_record = _strict_json(jonswap_audit.path)
    audit = _mapping(record.get("completion_audit"), context="order completion audit")
    _require_exact_keys(
        audit,
        {"path", "sha256", "schema", "accepted", "retained_rows"},
        context="order completion audit",
    )
    if (
        audit.get("sha256") != jonswap_audit.sha256
        or audit.get("schema") != producer.AUDIT_SCHEMA
        or audit.get("accepted") != 18_432
        or audit.get("retained_rows") != EXPECTED_ROWS_BY_FAMILY["jonswap_tma"]
    ):
        raise RuntimeError("order diagnostic is bound to a different JONSWAP audit")
    _canonical_recorded_path(
        audit.get("path"),
        expected_path=jonswap_audit.path,
        context="order completion audit",
    )
    selection = _mapping(record.get("selection"), context="order selection")
    _require_exact_keys(
        selection,
        {
            "definition",
            "independently_rescanned",
            "completion_audit_global_maximum_reproduced",
            "completion_audit_global_maximum_value",
            "independent_global_maximum_value",
            "maximum_value_absolute_difference",
            "maximum_value_comparison_tolerance",
            "accepted_trajectories_scanned",
            "stored_rows_scanned",
            "shards_scanned",
            "selected_count",
            "final_renderer_cross_check",
        },
        context="order selection",
    )
    audit_maximum = _finite_number(
        selection.get("completion_audit_global_maximum_value"),
        context="order audit maximum",
    )
    independent_maximum = _finite_number(
        selection.get("independent_global_maximum_value"),
        context="order independent maximum",
    )
    difference = _finite_number(
        selection.get("maximum_value_absolute_difference"),
        context="order maximum difference",
    )
    tolerance = _finite_number(
        selection.get("maximum_value_comparison_tolerance"),
        context="order maximum tolerance",
    )
    source_audit_maximum, source_audit_maximum_identity = producer._audit_maximum(
        audit_record
    )
    renderer_maximum = _finite_number(
        renderer_cases[0].get("maximum_gxi_high_band_fraction"),
        context="order renderer rank-one maximum",
    )
    if (
        selection.get("definition") != producer.SELECTION_DEFINITION
        or selection.get("independently_rescanned") is not True
        or selection.get("completion_audit_global_maximum_reproduced") is not True
        or selection.get("accepted_trajectories_scanned") != 18_432
        or selection.get("stored_rows_scanned")
        != EXPECTED_ROWS_BY_FAMILY["jonswap_tma"]
        or selection.get("selected_count") != 3
        or tolerance != 1.0e-7
        or audit_maximum != source_audit_maximum
        or not math.isclose(
            independent_maximum,
            renderer_maximum,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or difference != abs(independent_maximum - audit_maximum)
        or difference > tolerance
    ):
        raise RuntimeError("order diagnostic selection proof differs")
    cross = _mapping(
        selection.get("final_renderer_cross_check"),
        context="order renderer cross-check",
    )
    expected_cross = {
        "path": str(renderer_summary.resolve()),
        "sha256": _sha256(renderer_summary),
        "schema": RENDERER_SCHEMA,
        "status": "complete",
        "combined_summary_path": str(combined.path),
        "combined_summary_sha256": combined.sha256,
        "exact_final_population_verified": True,
        "independent_top_selection_reproduced": True,
    }
    if not _same_json(cross, expected_cross):
        raise RuntimeError("order diagnostic renderer cross-check differs")

    numerical = _mapping(
        record.get("numerical_definition"), context="order numerical definition"
    )
    expected_numerical = {
        "execution_platform": "cpu",
        "dtype": "float64",
        "nx": 1_024,
        "length": 2.0 * math.pi,
        "input_and_output_projection": "sharp |k| <= 128",
        "zero_output_mean": True,
        "pad_factor": 8,
        "cumulative_orders": [4, 5, 6, 7, 8],
        "relative_l2_definition": producer.RELATIVE_L2_DEFINITION,
    }
    if not _same_json(numerical, expected_numerical):
        raise RuntimeError("order diagnostic numerical definition differs")

    implementation = validate_order_implementation(
        record.get("diagnostic_implementation")
    )
    implementation_hashes = {
        item["path"]: item["sha256"]
        for item in _sequence(
            implementation.get("files"), context="order implementation files"
        )
    }
    identity = _mapping(record.get("identity"), context="order identity")
    _require_exact_keys(
        identity,
        {
            "generation_source_sha256_fingerprint",
            "generation_dependency_environment_fingerprint",
            "generation_execution_record_fingerprint",
            "generation_support_source_sha256",
            "diagnostic_source_sha256",
        },
        context="order identity",
    )
    audit_identity = _mapping(
        audit_record.get("identity"), context="JONSWAP audit identity"
    )
    expected_identity = {
        "generation_source_sha256_fingerprint": audit_identity.get(
            "source_sha256_fingerprint"
        ),
        "generation_dependency_environment_fingerprint": audit_identity.get(
            "dependency_environment_fingerprint"
        ),
        "generation_execution_record_fingerprint": audit_identity.get(
            "execution_record_fingerprint"
        ),
        "generation_support_source_sha256": audit_identity.get(
            "current_support_source_sha256"
        ),
        "diagnostic_source_sha256": implementation_hashes,
    }
    if not _same_json(identity, expected_identity):
        raise RuntimeError(
            "order diagnostic implementation/generation identity differs"
        )

    audit_chunks = _sequence(audit_record.get("chunks"), context="JONSWAP audit chunks")
    raw_bindings = _sequence(record.get("chunk_bindings"), context="order chunks")
    if len(audit_chunks) != 8 or len(raw_bindings) != len(audit_chunks):
        raise RuntimeError("order diagnostic chunk binding set differs")
    target_sources = {
        path: release.generation_sources["jonswap_tma"].source_sha256[path]
        for path in (
            "solver/gen_data/pipeline/reference.py",
            "solver/solvers/dno_series_jax.py",
        )
    }
    bindings_by_label: dict[str, dict[str, object]] = {}
    total_shards = 0
    for index, (raw_binding, raw_audit_chunk) in enumerate(
        zip(raw_bindings, audit_chunks, strict=True)
    ):
        context = f"order chunk {index}"
        binding = _mapping(raw_binding, context=context)
        _require_exact_keys(
            binding,
            {
                "label",
                "split",
                "accepted_scanned",
                "rows_scanned",
                "shards_scanned",
                "summary_path",
                "summary_sha256",
                "manifest_path",
                "manifest_sha256",
                "trajectory_map_path",
                "trajectory_map_sha256",
                "generation_target_source_sha256",
            },
            context=context,
        )
        audit_chunk = _mapping(raw_audit_chunk, context=f"{context} audit")
        expected_fields = {
            "label": audit_chunk.get("label"),
            "split": audit_chunk.get("split"),
            "accepted_scanned": audit_chunk.get("accepted_count"),
            "rows_scanned": audit_chunk.get("retained_rows"),
            "summary_path": audit_chunk.get("summary_path"),
            "summary_sha256": audit_chunk.get("summary_sha256"),
            "manifest_path": audit_chunk.get("manifest_path"),
            "manifest_sha256": audit_chunk.get("manifest_sha256"),
            "trajectory_map_path": audit_chunk.get("trajectory_map_path"),
            "trajectory_map_sha256": audit_chunk.get("trajectory_map_sha256"),
            "generation_target_source_sha256": target_sources,
        }
        if any(binding.get(name) != value for name, value in expected_fields.items()):
            raise RuntimeError(f"{context} differs from the completion audit")
        summary_path = _canonical_existing_recorded_path(
            binding.get("summary_path"), context=f"{context} summary"
        )
        source_root = summary_path.parent
        _resolved_child_artifact(
            source_root,
            summary_path,
            binding.get("summary_sha256"),
            context=f"{context} summary",
        )
        manifest_path = _canonical_existing_recorded_path(
            binding.get("manifest_path"), context=f"{context} manifest"
        )
        map_path = _canonical_existing_recorded_path(
            binding.get("trajectory_map_path"),
            context=f"{context} trajectory map",
        )
        manifest_artifact = _resolved_child_artifact(
            source_root,
            manifest_path,
            binding.get("manifest_sha256"),
            context=f"{context} manifest",
        )
        _resolved_child_artifact(
            source_root,
            map_path,
            binding.get("trajectory_map_sha256"),
            context=f"{context} trajectory map",
        )
        manifest = _strict_json(manifest_artifact.path)
        shard_count = len(
            _sequence(manifest.get("dataset_shards"), context=f"{context} shards")
        )
        if binding.get("shards_scanned") != shard_count:
            raise RuntimeError(f"{context} shard count differs")
        total_shards += shard_count
        label = binding.get("label")
        if not isinstance(label, str) or not label or label in bindings_by_label:
            raise RuntimeError("order diagnostic chunk labels are not unique")
        bindings_by_label[label] = dict(binding)
    if selection.get("shards_scanned") != total_shards:
        raise RuntimeError("order selection shard total differs")

    cases = _sequence(record.get("cases"), context="order cases")
    if len(cases) != 3:
        raise RuntimeError("order diagnostic does not contain three case records")
    seen_identities: set[tuple[object, ...]] = set()
    previous_rank = math.inf
    for index, (raw_case, renderer_case) in enumerate(
        zip(cases, renderer_cases, strict=True)
    ):
        context = f"order case {index}"
        case = _mapping(raw_case, context=context)
        _require_exact_keys(
            case,
            {
                "rank_metric",
                "identity",
                "trajectory_index_within_chunk",
                "depth",
                "sources",
                "orders",
                "successive_order_changes",
                "stored_float32_vs_recomputed_order6",
                "order6_to_order8",
            },
            context=context,
        )
        rank_metric = _finite_number(case.get("rank_metric"), context=f"{context} rank")
        if (
            not 0.0 <= rank_metric <= 1.0
            or rank_metric > previous_rank
            or not math.isclose(
                rank_metric,
                _finite_number(
                    renderer_case.get("maximum_gxi_high_band_fraction"),
                    context=f"{context} renderer rank",
                ),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise RuntimeError(f"{context} rank differs from renderer top three")
        previous_rank = rank_metric
        case_identity = _mapping(case.get("identity"), context=f"{context} identity")
        _require_exact_keys(
            case_identity,
            {
                "case_id",
                "chunk_label",
                "split",
                "batch_id",
                "local_index",
                "frame_index",
                "cell_id",
                "shard_path",
            },
            context=f"{context} identity",
        )
        identity_tuple = (
            _integer(case_identity.get("case_id"), context=f"{context} case ID"),
            case_identity.get("chunk_label"),
            case_identity.get("split"),
            _integer(case_identity.get("batch_id"), context=f"{context} batch"),
            _integer(case_identity.get("local_index"), context=f"{context} local"),
            _integer(case_identity.get("frame_index"), context=f"{context} frame"),
            case_identity.get("cell_id"),
            case_identity.get("shard_path"),
        )
        if identity_tuple in seen_identities:
            raise RuntimeError("order diagnostic repeats a selected identity")
        seen_identities.add(identity_tuple)
        overlap = {
            "case_id": renderer_case.get("case_id"),
            "split": renderer_case.get("split"),
            "frame_index": renderer_case.get("maximum_gxi_high_band_fraction_frame"),
            "cell_id": renderer_case.get("category"),
        }
        if any(case_identity.get(name) != value for name, value in overlap.items()):
            raise RuntimeError(f"{context} identity differs from renderer top three")
        if index == 0 and not _same_json(case_identity, source_audit_maximum_identity):
            raise RuntimeError(
                "order rank-one identity differs from completion-audit maximum"
            )
        trajectory_index = _integer(
            case.get("trajectory_index_within_chunk"),
            context=f"{context} trajectory index",
        )
        if trajectory_index != renderer_case.get("trajectory_index"):
            raise RuntimeError(f"{context} trajectory index differs from renderer")
        depth = _finite_number(case.get("depth"), context=f"{context} depth")
        if depth <= 0.0 or depth != renderer_case.get("depth"):
            raise RuntimeError(f"{context} depth differs from renderer")

        sources = _mapping(case.get("sources"), context=f"{context} sources")
        source_keys = {
            f"{name}_{field}"
            for name in (
                "summary",
                "manifest",
                "trajectory_map",
                "proposal",
                "result",
                "shard",
            )
            for field in ("path", "sha256")
        }
        _require_exact_keys(sources, source_keys, context=f"{context} sources")
        canonical_source_paths = {}
        for name in (
            "summary",
            "manifest",
            "trajectory_map",
            "proposal",
            "result",
            "shard",
        ):
            canonical_source_paths[name] = _canonical_existing_recorded_path(
                sources.get(f"{name}_path"), context=f"{context} {name}"
            )
        summary_path = canonical_source_paths["summary"]
        binding = next(
            (
                value
                for value in bindings_by_label.values()
                if Path(str(value["summary_path"])).resolve() == summary_path
            ),
            None,
        )
        if binding is None or binding.get("label") != case_identity.get("chunk_label"):
            raise RuntimeError(f"{context} source summary differs from its chunk")
        for name in ("summary", "manifest", "trajectory_map"):
            if (
                Path(str(sources.get(f"{name}_path"))).resolve()
                != Path(str(binding[f"{name}_path"])).resolve()
                or sources.get(f"{name}_sha256") != binding[f"{name}_sha256"]
            ):
                raise RuntimeError(f"{context} {name} parent differs")
        source_root = summary_path.parent
        if renderer_case.get("source_root") != str(source_root) or renderer_case.get(
            "source_index"
        ) != release_source_indices.get(summary_path):
            raise RuntimeError(
                f"{context} source ownership differs from renderer top three"
            )
        manifest_path = canonical_source_paths["manifest"]
        manifest = _strict_json(manifest_path)
        child_artifacts = {
            name: _resolved_child_artifact(
                source_root,
                sources.get(f"{name}_path"),
                sources.get(f"{name}_sha256"),
                context=f"{context} {name}",
            )
            for name in ("proposal", "result", "shard")
        }
        batches = _sequence(
            manifest.get("dataset_batches"), context=f"{context} manifest batches"
        )
        matching_batches = []
        for raw_batch in batches:
            batch = _mapping(raw_batch, context=f"{context} manifest batch")
            proposal = _resolved_child_artifact(
                source_root,
                batch.get("proposal_path"),
                batch.get("proposal_sha256"),
                context=f"{context} manifest proposal",
            )
            result = _resolved_child_artifact(
                source_root,
                batch.get("result_path"),
                batch.get("result_sha256"),
                context=f"{context} manifest result",
            )
            if (
                proposal == child_artifacts["proposal"]
                and result == child_artifacts["result"]
            ):
                matching_batches.append(batch)
        if len(matching_batches) != 1:
            raise RuntimeError(f"{context} proposal/result parent differs")
        batch = matching_batches[0]
        if batch.get("batch_id") != case_identity.get("batch_id"):
            raise RuntimeError(f"{context} batch identity differs")
        shards = _sequence(
            manifest.get("dataset_shards"), context=f"{context} manifest shards"
        )
        matching_shards = []
        for shard_index, raw_shard in enumerate(shards):
            shard = _mapping(raw_shard, context=f"{context} manifest shard")
            artifact = _resolved_child_artifact(
                source_root,
                shard.get("path"),
                shard.get("sha256"),
                context=f"{context} manifest shard",
            )
            if artifact == child_artifacts["shard"]:
                matching_shards.append((shard_index, shard))
        if len(matching_shards) != 1:
            raise RuntimeError(f"{context} shard parent differs")
        matched_shard_index, _ = matching_shards[0]
        if matched_shard_index != renderer_case.get("shard_index"):
            raise RuntimeError(
                f"{context} shard ownership differs from renderer top three"
            )
        relative_shard = child_artifacts["shard"].path.relative_to(source_root)
        if str(relative_shard) != case_identity.get("shard_path"):
            raise RuntimeError(f"{context} shard identity differs")

        local_index = int(case_identity["local_index"])
        with np.load(child_artifacts["proposal"].path, allow_pickle=False) as proposal:
            if int(proposal["batch_id"]) != case_identity.get("batch_id"):
                raise RuntimeError(f"{context} proposal batch differs")
            if int(proposal["case_id"][local_index]) != case_identity.get("case_id"):
                raise RuntimeError(f"{context} proposal case differs")
            specification = json.loads(str(proposal["case_spec_json"][local_index]))
        if (
            not isinstance(specification, dict)
            or specification.get("cell_id") != case_identity.get("cell_id")
            or specification.get("depth") != depth
        ):
            raise RuntimeError(f"{context} proposal specification differs")
        result = _strict_json(child_artifacts["result"].path)
        result_cases = _sequence(result.get("cases"), context=f"{context} result cases")
        if local_index >= len(result_cases):
            raise RuntimeError(f"{context} result local index differs")
        result_case = _mapping(
            result_cases[local_index], context=f"{context} result case"
        )
        first_row = renderer_case.get("first_shard_row")
        row_count = renderer_case.get("row_count")
        expected_result_case = {
            "accepted": True,
            "case_id": case_identity.get("case_id"),
            "first_row": first_row,
            "row_count": row_count,
        }
        if (
            any(
                result_case.get(name) != value
                for name, value in expected_result_case.items()
            )
            or result.get("proposal_sha256") != child_artifacts["proposal"].sha256
            or result.get("shard_sha256") != child_artifacts["shard"].sha256
        ):
            raise RuntimeError(f"{context} result ownership differs")
        first = _integer(first_row, context=f"{context} first shard row")
        count = _integer(row_count, context=f"{context} row count")
        if count != 16:
            raise RuntimeError(f"{context} row count differs from renderer top three")
        with np.load(child_artifacts["shard"].path, allow_pickle=False) as shard:
            depths = np.asarray(shard["depth"][first : first + count], dtype=np.float64)
            stored = np.asarray(shard["gxi"][first : first + count], dtype=np.float64)
        if depths.shape != (count,) or not np.all(depths == depth):
            raise RuntimeError(f"{context} shard depth differs")
        fractions = producer.high_band_fraction(stored)
        peak_frame = int(np.argmax(fractions))
        if peak_frame != case_identity.get("frame_index") or not math.isclose(
            float(fractions[peak_frame]), rank_metric, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise RuntimeError(f"{context} shard rank metric differs")

        orders = _mapping(case.get("orders"), context=f"{context} orders")
        if set(orders) != {"4", "5", "6", "7", "8"}:
            raise RuntimeError(f"{context} has incomplete order comparisons")
        for order in (4, 5, 6, 7, 8):
            order_record = _mapping(
                orders[str(order)], context=f"{context} order {order}"
            )
            _require_exact_keys(
                order_record,
                {
                    "projected_l2_norm_by_frame",
                    "projected_high_band_energy_fraction_by_frame",
                    "maximum_projected_high_band_energy_fraction",
                    "maximum_projected_l2_norm",
                },
                context=f"{context} order {order}",
            )
            _validated_order_series(
                order_record,
                series_name="projected_l2_norm_by_frame",
                maximum_name="maximum_projected_l2_norm",
                context=f"{context} order {order}",
            )
            _validated_order_series(
                order_record,
                series_name="projected_high_band_energy_fraction_by_frame",
                maximum_name="maximum_projected_high_band_energy_fraction",
                context=f"{context} order {order}",
                upper_bound=1.0,
            )
        transitions = _sequence(
            case.get("successive_order_changes"), context=f"{context} transitions"
        )
        if len(transitions) != 4:
            raise RuntimeError(f"{context} transition set differs")
        for transition, (previous, current) in zip(
            transitions, zip((4, 5, 6, 7), (5, 6, 7, 8)), strict=True
        ):
            transition_record = _mapping(
                transition, context=f"{context} transition {previous}-{current}"
            )
            _require_exact_keys(
                transition_record,
                {
                    "from_order",
                    "to_order",
                    "projected_relative_l2_difference_by_frame",
                    "maximum_projected_relative_l2_difference",
                    "raw_preprojection_relative_l2_difference_by_frame",
                    "maximum_raw_preprojection_relative_l2_difference",
                    "absolute_high_band_fraction_change_by_frame",
                    "maximum_absolute_high_band_fraction_change",
                },
                context=f"{context} transition {previous}-{current}",
            )
            if (
                transition_record.get("from_order") != previous
                or transition_record.get("to_order") != current
            ):
                raise RuntimeError(f"{context} transition order differs")
            for series_name, maximum_name, upper in (
                (
                    "projected_relative_l2_difference_by_frame",
                    "maximum_projected_relative_l2_difference",
                    None,
                ),
                (
                    "raw_preprojection_relative_l2_difference_by_frame",
                    "maximum_raw_preprojection_relative_l2_difference",
                    None,
                ),
                (
                    "absolute_high_band_fraction_change_by_frame",
                    "maximum_absolute_high_band_fraction_change",
                    1.0,
                ),
            ):
                _validated_order_series(
                    transition_record,
                    series_name=series_name,
                    maximum_name=maximum_name,
                    context=f"{context} transition {previous}-{current}",
                    upper_bound=upper,
                )
        for name, series_name, maximum_name in (
            (
                "stored_float32_vs_recomputed_order6",
                "relative_l2_difference_by_frame",
                "maximum_relative_l2_difference",
            ),
            (
                "order6_to_order8",
                "projected_relative_l2_difference_by_frame",
                "maximum_projected_relative_l2_difference",
            ),
        ):
            comparison = _mapping(case.get(name), context=f"{context} {name}")
            _require_exact_keys(
                comparison, {series_name, maximum_name}, context=f"{context} {name}"
            )
            _validated_order_series(
                comparison,
                series_name=series_name,
                maximum_name=maximum_name,
                context=f"{context} {name}",
            )
    return {
        "output": _artifact(output).record(),
        "selected_count": 3,
        "diagnostic_implementation": implementation,
    }


def _validate_repository_implementation(
    value: object,
    *,
    schema: str,
    files: Sequence[tuple[str, str]],
    semantic_relationship: Mapping[str, object],
    context: str,
) -> dict[str, object]:
    """Validate one exact, canonical map of repository implementation bytes."""

    record = _mapping(value, context=context)
    _require_exact_keys(
        record,
        {"schema", "files", "semantic_relationship", "fingerprint"},
        context=context,
    )
    if record.get("schema") != schema:
        raise RuntimeError(f"{context} schema differs")
    relationship = _mapping(
        record.get("semantic_relationship"),
        context=f"{context} semantic relationship",
    )
    if not _same_json(relationship, semantic_relationship):
        raise RuntimeError(f"{context} semantic relationship differs")
    raw_files = _sequence(record.get("files"), context=f"{context} files")
    if len(raw_files) != len(files):
        raise RuntimeError(f"{context} file set differs")
    repository_root = _require_real_directory(
        ROOT, context=f"{context} repository root"
    )
    canonical_files = []
    physical_identities = []
    for index, ((expected_path, expected_role), raw_file) in enumerate(
        zip(files, raw_files, strict=True)
    ):
        file_context = f"{context} file {index}"
        file_record = _mapping(raw_file, context=file_context)
        _require_exact_keys(
            file_record,
            {"path", "role", "bytes", "sha256"},
            context=file_context,
        )
        if (
            file_record.get("path") != expected_path
            or file_record.get("role") != expected_role
        ):
            raise RuntimeError(f"{file_context} path or role differs")
        expected_bytes = _integer(
            file_record.get("bytes"), context=f"{file_context} bytes"
        )
        expected_sha256 = _digest(
            file_record.get("sha256"), context=f"{file_context} SHA-256"
        )
        path = _source_file_inside(repository_root, expected_path, context=file_context)
        before = path.stat()
        observed = _artifact(path, expected_sha256=expected_sha256)
        after = path.stat()
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise RuntimeError(f"{file_context} changed while being hashed")
        if observed.bytes != expected_bytes:
            raise RuntimeError(f"{file_context} byte count differs")
        physical_identities.append((expected_path, path, identity_after))
        canonical_files.append(
            {
                "path": expected_path,
                "role": expected_role,
                "bytes": expected_bytes,
                "sha256": expected_sha256,
            }
        )
    for index, (relative, path, expected_identity) in enumerate(physical_identities):
        current_path = _source_file_inside(
            repository_root,
            relative,
            context=f"{context} file {index}",
        )
        if current_path != path:
            raise RuntimeError(f"{context} file {index} path changed")
        current = path.stat()
        current_identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if current_identity != expected_identity:
            raise RuntimeError(f"{context} file {index} changed after authentication")
    payload: dict[str, object] = {
        "schema": schema,
        "files": canonical_files,
        "semantic_relationship": dict(semantic_relationship),
    }
    if record.get("fingerprint") != _canonical_sha256(payload):
        raise RuntimeError(f"{context} fingerprint differs")
    return {**payload, "fingerprint": record["fingerprint"]}


def validate_training_implementation(value: object) -> dict[str, object]:
    """Validate and rehash the handoff implementation and semantic evidence."""

    return _validate_repository_implementation(
        value,
        schema=TRAINING_IMPLEMENTATION_SCHEMA,
        files=TRAINING_IMPLEMENTATION_FILES,
        semantic_relationship=TRAINING_IMPLEMENTATION_RELATIONSHIP,
        context="training implementation",
    )


def validate_renderer_implementation(value: object) -> dict[str, object]:
    """Validate and rehash the renderer's canonical implementation record."""

    if __package__:
        from scripts import render_paper_dataset_worst_cases as producer
    else:
        import render_paper_dataset_worst_cases as producer

    return _validate_repository_implementation(
        value,
        schema=producer.IMPLEMENTATION_SCHEMA,
        files=producer.IMPLEMENTATION_FILES,
        semantic_relationship=producer.IMPLEMENTATION_RELATIONSHIP,
        context="renderer implementation",
    )


def validate_order_implementation(value: object) -> dict[str, object]:
    """Validate and rehash the order diagnostic's implementation record."""

    if __package__:
        from scripts import diagnose_final_jonswap_order_convergence as producer
    else:
        import diagnose_final_jonswap_order_convergence as producer

    return _validate_repository_implementation(
        value,
        schema=producer.IMPLEMENTATION_SCHEMA,
        files=producer.IMPLEMENTATION_FILES,
        semantic_relationship=producer.IMPLEMENTATION_RELATIONSHIP,
        context="order diagnostic implementation",
    )


def validate_figure_implementation(value: object) -> dict[str, object]:
    """Validate and rehash the family figure's canonical implementation record."""

    if __package__:
        from scripts import build_parameterized_dataset_case_figure as producer
    else:
        import build_parameterized_dataset_case_figure as producer

    return _validate_repository_implementation(
        value,
        schema=producer.IMPLEMENTATION_SCHEMA,
        files=producer.IMPLEMENTATION_FILES,
        semantic_relationship=producer.IMPLEMENTATION_RELATIONSHIP,
        context="figure implementation",
    )


def _expected_training_rows() -> int:
    return sum(
        CASES_PER_FAMILY_BY_SPLIT["train"] * ROWS_PER_CASE[family]
        for family in FAMILY_ORDER
    )


def _validate_training_dataset_view(
    record: Mapping[str, object],
    *,
    combined: Artifact,
) -> dict[str, object]:
    """Authenticate the training audit's exact combined-view storage identity."""

    combined_record = _strict_json(combined.path)
    combined_view = _mapping(
        combined_record.get("dataset_view"), context="combined dataset view"
    )

    def combined_parent(name: str) -> Artifact:
        raw = _mapping(combined_view.get(name), context=f"combined dataset view {name}")
        _require_exact_keys(
            raw, {"path", "bytes", "sha256"}, context=f"combined {name}"
        )
        path_value = raw.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"combined {name} path differs")
        return _verify_record_artifact(
            raw,
            expected_path=Path(path_value),
            context=f"combined {name}",
        )

    manifest_artifact = combined_parent("manifest")
    trajectory_map_artifact = combined_parent("trajectory_map")
    view = _mapping(record.get("dataset_view"), context="training dataset view")
    _require_exact_keys(
        view,
        {
            "manifest",
            "trajectory_map",
            "dataset_contract_fingerprint",
            "shard_count",
            "shard_rows",
            "shard_bytes",
            "shard_identity_fingerprint",
            "shards",
        },
        context="training dataset view",
    )
    for name, expected in (
        ("manifest", manifest_artifact),
        ("trajectory_map", trajectory_map_artifact),
    ):
        observed = _mapping(view.get(name), context=f"training {name}")
        _require_exact_keys(
            observed, {"path", "bytes", "sha256"}, context=f"training {name}"
        )
        if not _same_json(observed, expected.record()):
            raise RuntimeError(f"training {name} differs from combined view")
    contract_fingerprint = _digest(
        view.get("dataset_contract_fingerprint"),
        context="training dataset-contract fingerprint",
    )
    if contract_fingerprint != combined_view.get("dataset_contract_fingerprint"):
        raise RuntimeError("training dataset contract differs from combined view")

    manifest = _strict_json(manifest_artifact.path)
    if manifest.get("dataset_contract_fingerprint") != contract_fingerprint:
        raise RuntimeError("training dataset contract differs from combined manifest")
    manifest_shards = _sequence(
        manifest.get("dataset_shards"), context="combined manifest shards"
    )
    manifest_batches = _sequence(
        manifest.get("dataset_batches"), context="combined manifest batches"
    )
    shards = _sequence(view.get("shards"), context="training dataset shards")
    shard_count = _integer(view.get("shard_count"), context="training shard count")
    if shard_count != len(shards) or shard_count != len(manifest_shards):
        raise RuntimeError("training shard count differs from combined manifest")
    identity_payload: list[dict[str, object]] = []
    shard_rows = 0
    shard_bytes = 0
    selected_training_rows = 0
    canonical_shards: list[dict[str, object]] = []
    for index, (raw_shard, raw_manifest_shard) in enumerate(
        zip(shards, manifest_shards, strict=True)
    ):
        context = f"training shard {index}"
        shard = _mapping(raw_shard, context=context)
        _require_exact_keys(
            shard,
            {
                "index",
                "rows",
                "selected_training_rows",
                "path",
                "bytes",
                "sha256",
            },
            context=context,
        )
        if shard.get("index") != index:
            raise RuntimeError("training shard indices are not canonical")
        manifest_shard = _mapping(
            raw_manifest_shard, context=f"combined manifest shard {index}"
        )
        manifest_path_value = manifest_shard.get("path")
        if not isinstance(manifest_path_value, str) or not manifest_path_value:
            raise ValueError(f"combined manifest shard {index} path differs")
        manifest_path = Path(manifest_path_value).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = manifest_artifact.path.parent / manifest_path
        manifest_path = manifest_path.resolve(strict=True)
        physical = _verify_record_artifact(
            {
                "path": shard.get("path"),
                "bytes": shard.get("bytes"),
                "sha256": shard.get("sha256"),
            },
            expected_path=manifest_path,
            context=context,
        )
        rows = _integer(shard.get("rows"), context=f"{context} rows")
        selected_rows = _integer(
            shard.get("selected_training_rows"),
            context=f"{context} selected training rows",
        )
        batch_index = _integer(
            manifest_shard.get("batch_index"),
            context=f"combined manifest shard {index} batch index",
        )
        if batch_index >= len(manifest_batches):
            raise RuntimeError(f"{context} batch index differs")
        batch = _mapping(
            manifest_batches[batch_index],
            context=f"combined manifest batch {batch_index}",
        )
        expected_selected_rows = rows if batch.get("split_id") == 0 else 0
        if (
            rows != manifest_shard.get("n_rows")
            or physical.sha256 != manifest_shard.get("sha256")
            or selected_rows != expected_selected_rows
        ):
            raise RuntimeError(f"{context} differs from combined manifest")
        canonical = {
            "index": index,
            "rows": rows,
            "selected_training_rows": selected_rows,
            **physical.record(),
        }
        canonical_shards.append(canonical)
        identity_payload.append(
            {
                key: canonical[key]
                for key in ("index", "path", "bytes", "sha256", "rows")
            }
        )
        shard_rows += rows
        shard_bytes += physical.bytes
        selected_training_rows += selected_rows
    normalization = _mapping(
        record.get("training_normalization"), context="training normalization"
    )
    selection = _mapping(
        normalization.get("accepted_training_row_selection"),
        context="training row selection",
    )
    if (
        selected_training_rows != selection.get("count")
        or selected_training_rows != _expected_training_rows()
    ):
        raise RuntimeError("training selected-row totals differ from final contract")
    if (
        view.get("shard_rows") != shard_rows
        or view.get("shard_bytes") != shard_bytes
        or view.get("shard_identity_fingerprint") != _canonical_sha256(identity_payload)
    ):
        raise RuntimeError("training shard totals or fingerprint differ")
    return {
        "manifest": manifest_artifact.record(),
        "trajectory_map": trajectory_map_artifact.record(),
        "dataset_contract_fingerprint": contract_fingerprint,
        "shard_count": shard_count,
        "shard_rows": shard_rows,
        "shard_bytes": shard_bytes,
        "selected_training_rows": selected_training_rows,
        "shard_identity_fingerprint": view["shard_identity_fingerprint"],
        "shards": canonical_shards,
    }


def validate_training_output(output: Path, combined: Artifact) -> Mapping[str, object]:
    _reauthenticate_expected_artifact(combined, context="training combined summary")
    record = _strict_json(output)
    if record.get("schema") != TRAINING_SCHEMA or record.get("status") != "complete":
        raise RuntimeError("training-handoff audit is incomplete")
    if (
        record.get("purpose") != "training_handoff_normalization_audit"
        or record.get("audit_only") is not True
        or record.get("acceptance_or_generation_effect") is not False
    ):
        raise RuntimeError("training-handoff audit scope differs")
    training_implementation = validate_training_implementation(
        record.get("training_implementation")
    )
    dataset_view = _validate_training_dataset_view(record, combined=combined)
    summary = _mapping(record.get("combined_summary"), context="training summary")
    if (
        summary.get("sha256") != combined.sha256
        or summary.get("bytes") != combined.bytes
    ):
        raise RuntimeError("training-handoff audit is bound to another summary")
    _canonical_recorded_path(
        summary.get("path"),
        expected_path=combined.path,
        context="training combined summary",
    )
    execution = _mapping(record.get("execution"), context="training execution")
    if (
        execution.get("gpu_used") is not False
        or execution.get("training_split_seed") != 0
    ):
        raise RuntimeError("training-handoff audit did not use CPU/seed zero")
    final_contract = _mapping(record.get("final_contract"), context="training contract")
    expected = _mapping(final_contract.get("expected"), context="training expected")
    observed = _mapping(final_contract.get("observed"), context="training observed")
    if (
        expected.get("families") != list(FAMILY_ORDER)
        or not _same_json(expected.get("revision_by_family"), REVISION_BY_FAMILY)
        or not _same_json(
            expected.get("source_count_by_family"), SOURCE_COUNT_BY_FAMILY
        )
        or expected.get("source_count") != FINAL_SOURCE_COUNT
        or not _same_json(
            expected.get("accepted_cases_per_family_by_split"),
            CASES_PER_FAMILY_BY_SPLIT,
        )
        or not _same_json(expected.get("rows_per_accepted_case"), ROWS_PER_CASE)
        or expected.get("accepted_cases") != FINAL_ACCEPTED_CASES
        or expected.get("accepted_rows") != FINAL_RETAINED_ROWS
        or observed.get("accepted_cases") != FINAL_ACCEPTED_CASES
        or observed.get("accepted_rows") != FINAL_RETAINED_ROWS
    ):
        raise RuntimeError("training-handoff final contract differs")
    normalization = _mapping(
        record.get("training_normalization"), context="training normalization"
    )
    selection = _mapping(
        normalization.get("accepted_training_row_selection"),
        context="training row selection",
    )
    expected_training_rows = _expected_training_rows()
    if selection.get("seed") != 0 or selection.get("count") != expected_training_rows:
        raise RuntimeError("training row selection differs from seed-zero contract")
    stats = _mapping(normalization.get("stats"), context="training stats")
    expected_index_selection = {
        "count": selection.get("count"),
        "sha256": selection.get("sha256"),
    }
    if (
        stats.get("num_examples") != expected_training_rows
        or stats.get("storage_num_examples") != FINAL_RETAINED_ROWS
        or not _same_json(stats.get("index_selection"), expected_index_selection)
        or normalization.get("stats_fingerprint") != _canonical_sha256(stats)
    ):
        raise RuntimeError("training normalization statistics differ")
    for name in ("feature_min", "feature_max", "feature_absmax"):
        values = _sequence(stats.get(name), context=f"training stats {name}")
        if len(values) != 2:
            raise RuntimeError(f"training stats {name} has the wrong length")
        for index, value in enumerate(values):
            _finite_number(value, context=f"training stats {name}[{index}]")
    scalar_stats = {
        name: _finite_number(stats.get(name), context=f"training stats {name}")
        for name in (
            "target_min",
            "target_max",
            "target_absmax",
            "depth_min",
            "depth_max",
            "log_depth_min",
            "log_depth_max",
            "domain_length",
        )
    }
    if scalar_stats["depth_min"] <= 0.0 or not math.isclose(
        scalar_stats["domain_length"], 2.0 * math.pi, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise RuntimeError("training normalization physical scales differ")
    full_selection = _mapping(
        normalization.get("full_row_selection"),
        context="training full-row selection",
    )
    split_records = _mapping(
        full_selection.get("splits"), context="training split rows"
    )
    if (
        full_selection.get("schema") != "paper_dataset_full_row_selection_v1"
        or full_selection.get("seed") != 0
        or full_selection.get("test_rows_in_train_or_validation") != 0
        or set(split_records) != set(SPLIT_ORDER)
    ):
        raise RuntimeError("training full-row selection contract differs")
    for split in SPLIT_ORDER:
        split_record = _mapping(
            split_records.get(split), context=f"training {split} rows"
        )
        family_rows = _mapping(
            split_record.get("per_family_row_count"),
            context=f"training {split} family rows",
        )
        expected_family_rows = {
            family: CASES_PER_FAMILY_BY_SPLIT[split] * ROWS_PER_CASE[family]
            for family in FAMILY_ORDER
        }
        if (
            split_record.get("policy") != "all_retained_rows"
            or split_record.get("all_retained_rows") is not True
            or split_record.get("count") != sum(expected_family_rows.values())
            or not _same_json(family_rows, expected_family_rows)
        ):
            raise RuntimeError(f"training {split} full-row selection differs")
    train_record = _mapping(split_records.get("train"), context="training rows")
    if train_record.get("sha256") != selection.get("sha256"):
        raise RuntimeError("training row selection SHA differs from normalization")
    return {
        "output": _artifact(output).record(),
        "training_rows": expected_training_rows,
        "training_implementation": training_implementation,
        "dataset_view": dataset_view,
    }


def _expected_figure_cases(
    release: ReleaseBinding,
) -> list[dict[str, object]]:
    """Reconstruct the four lower-median selections from authenticated maps."""

    if __package__:
        from scripts import build_parameterized_dataset_case_figure as producer
    else:
        import build_parameterized_dataset_case_figure as producer

    source_records = _release_source_records(release)
    proofs = _renderer_source_proofs(release)
    expected: list[dict[str, object]] = []
    for family in FAMILY_ORDER:
        category = producer.CENTRAL_VALIDATION_CATEGORIES[family]
        candidates: list[tuple[int, int, object]] = []
        for source_index, (source, proof) in enumerate(
            zip(source_records, proofs, strict=True)
        ):
            if source["family"] != family or source["split"] != "validation":
                continue
            for trajectory in _sequence(
                proof.get("trajectories"),
                context=f"figure {family} source trajectories",
            ):
                if trajectory.category == category:
                    candidates.append((trajectory.case_id, source_index, trajectory))
        candidates.sort(key=lambda item: item[0])
        if not candidates or len({item[0] for item in candidates}) != len(candidates):
            raise RuntimeError(f"figure {family} lower-median candidates differ")
        lower_median = (len(candidates) - 1) // 2
        case_id, source_index, trajectory = candidates[lower_median]
        source = source_records[source_index]
        proof = proofs[source_index]
        summary_path = proof.get("summary_path")
        manifest_path = proof.get("manifest_path")
        map_path = proof.get("map_path")
        if not all(
            isinstance(path, Path) for path in (summary_path, manifest_path, map_path)
        ):
            raise TypeError(f"figure {family} source parent paths differ")
        summary_artifact = _artifact(
            summary_path, expected_sha256=source.get("summary_sha256")
        )
        manifest_artifact = _artifact(
            manifest_path, expected_sha256=source.get("manifest_sha256")
        )
        map_artifact = _artifact(
            map_path, expected_sha256=source.get("trajectory_map_sha256")
        )
        shard_records = _sequence(
            proof.get("shards"), context=f"figure {family} source shards"
        )
        raw_shard = shard_records[trajectory.shard_index]
        if not isinstance(raw_shard, tuple) or len(raw_shard) != 2:
            raise TypeError(f"figure {family} shard proof differs")
        shard_path, shard_sha256 = raw_shard
        if not isinstance(shard_path, Path):
            raise TypeError(f"figure {family} shard path differs")
        shard_artifact = _artifact(shard_path, expected_sha256=shard_sha256)
        depth = _renderer_case_depth(
            proof,
            shard_index=trajectory.shard_index,
            first_shard_row=trajectory.first_shard_row,
            row_count=trajectory.row_count,
            context=f"figure {family}",
        )
        with np.load(shard_path, allow_pickle=False) as shard:
            if "time" not in shard.files:
                raise RuntimeError(f"figure {family} shard omits time")
            times = np.asarray(shard["time"], dtype=np.float64)
        if trajectory.first_shard_row >= times.size:
            raise RuntimeError(f"figure {family} selected time row differs")
        time_value = float(times[trajectory.first_shard_row])
        summary_record = _mapping(
            proof.get("summary"), context=f"figure {family} source summary"
        )
        run_spec = _mapping(
            summary_record.get("run_spec"), context=f"figure {family} run spec"
        )
        revision = _integer(
            run_spec.get("revision_id"), context=f"figure {family} revision"
        )
        length, gravity, stored_nx = producer._source_numerical_scales(
            summary_record, family=family
        )
        if (
            revision != REVISION_BY_FAMILY[family]
            or time_value != 0.0
            or depth <= 0.0
            or stored_nx != 1_024
        ):
            raise RuntimeError(f"figure {family} selected source contract differs")
        expected.append(
            {
                "family": family,
                "revision_id": revision,
                "split": "validation",
                "category": category,
                "case_id": case_id,
                "time": time_value,
                "depth": depth,
                "gravity": gravity,
                "domain_length": length,
                "stored_nx": stored_nx,
                "selection": {
                    "rule": producer.SELECTION_RULE,
                    "candidate_count": len(candidates),
                    "lower_median_index_zero_based": lower_median,
                    "is_empirical_medoid": False,
                },
                "owned_row": {
                    "accepted_index": trajectory.accepted_index,
                    "trajectory_index": trajectory.trajectory_index,
                    "frame_index": 0,
                    "shard_index": trajectory.shard_index,
                    "shard_row": trajectory.first_shard_row,
                },
                "dimensionless_variables": dict(producer.DIMENSIONLESS_VARIABLES),
                "source_summary": summary_artifact.record(),
                "source_manifest": manifest_artifact.record(),
                "source_trajectory_map": map_artifact.record(),
                "selected_shard": shard_artifact.record(),
            }
        )
    return expected


def validate_figure_output(
    output_stem: Path,
    release: ReleaseBinding,
) -> Mapping[str, object]:
    combined = release.combined.summary
    _reauthenticate_expected_artifact(combined, context="figure combined summary")
    pdf = output_stem.with_suffix(".pdf")
    png = output_stem.with_suffix(".png")
    sidecar_path = output_stem.with_suffix(".json")
    sidecar = _strict_json(sidecar_path)
    _require_exact_keys(
        sidecar,
        {
            "schema",
            "status",
            "description",
            "combined_summary",
            "combined_view",
            "release_contract",
            "publication",
            "cases",
            "figure_implementation",
            "artifacts",
        },
        context="figure sidecar",
    )
    if sidecar.get("schema") != FIGURE_SCHEMA or sidecar.get("status") != "complete":
        raise RuntimeError("family illustration sidecar is incomplete")
    if __package__:
        from scripts import build_parameterized_dataset_case_figure as producer
    else:
        import build_parameterized_dataset_case_figure as producer

    if sidecar.get("description") != producer.SIDECAR_DESCRIPTION:
        raise RuntimeError("family illustration description differs")
    summary = _mapping(sidecar.get("combined_summary"), context="figure summary")
    _require_exact_keys(summary, {"path", "bytes", "sha256"}, context="figure summary")
    if (
        summary.get("sha256") != combined.sha256
        or summary.get("bytes") != combined.bytes
    ):
        raise RuntimeError("family illustration is bound to another summary")
    _canonical_recorded_path(
        summary.get("path"),
        expected_path=combined.path,
        context="figure combined summary",
    )
    combined_view = _mapping(
        sidecar.get("combined_view"), context="figure combined view"
    )
    _require_exact_keys(
        combined_view,
        {"manifest", "trajectory_map"},
        context="figure combined view",
    )
    manifest = _verify_record_artifact(
        combined_view.get("manifest"),
        expected_path=release.combined.manifest.path,
        context="figure combined manifest",
    )
    trajectory_map = _verify_record_artifact(
        combined_view.get("trajectory_map"),
        expected_path=release.combined.trajectory_map.path,
        context="figure combined trajectory map",
    )
    if (
        manifest != release.combined.manifest
        or trajectory_map != release.combined.trajectory_map
    ):
        raise RuntimeError("family illustration combined view differs")
    release_contract = _mapping(
        sidecar.get("release_contract"), context="figure release"
    )
    expected_release_contract = {
        "source_count": FINAL_SOURCE_COUNT,
        "source_count_by_family": dict(producer.EXPECTED_SOURCE_COUNT_BY_FAMILY),
        "accepted_cases": FINAL_ACCEPTED_CASES,
        "retained_rows": FINAL_RETAINED_ROWS,
        "family_revisions": dict(producer.EXPECTED_REVISIONS),
        "ordered_cell_ids": dict(producer.EXPECTED_CELL_IDS),
        "central_validation_categories": dict(producer.CENTRAL_VALIDATION_CATEGORIES),
        "source_intervals": [
            {
                "family": item.family,
                "split": item.split.value,
                "stream_id": item.stream_id,
                "accepted_before": item.accepted_before,
                "accepted_after": item.accepted_after,
            }
            for item in producer._expected_chunk_layout()
        ],
    }
    if not _same_json(release_contract, expected_release_contract):
        raise RuntimeError("family illustration release contract differs")
    publication = _mapping(sidecar.get("publication"), context="figure publication")
    expected_publication = {
        "commit_marker": str(sidecar_path.resolve()),
        "rule": producer.PUBLICATION_RULE,
    }
    if not _same_json(publication, expected_publication):
        raise RuntimeError("family illustration publication contract differs")
    expected_cases = _expected_figure_cases(release)
    if not _same_json(sidecar.get("cases"), expected_cases):
        raise RuntimeError("family illustration lower-median case proofs differ")
    figure_implementation = validate_figure_implementation(
        sidecar.get("figure_implementation")
    )
    artifacts = _mapping(sidecar.get("artifacts"), context="figure artifacts")
    _require_exact_keys(artifacts, {"pdf", "png"}, context="figure artifacts")
    pdf_artifact = _verify_record_artifact(
        artifacts.get("pdf"), expected_path=pdf, context="figure PDF"
    )
    png_artifact = _verify_record_artifact(
        artifacts.get("png"), expected_path=png, context="figure PNG"
    )
    return {
        "sidecar": _artifact(sidecar_path).record(),
        "pdf": pdf_artifact.record(),
        "png": png_artifact.record(),
        "figure_implementation": figure_implementation,
    }


def default_stages(
    config: RunnerConfig,
    release: ReleaseBinding,
    output_root: Path,
) -> tuple[StageSpec, ...]:
    """Construct the exact ordered CPU diagnostic pipeline."""

    python = str(config.python)
    combined = release.combined.summary
    renderer_dir = output_root / "worst_cases"
    renderer_summary = renderer_dir / "summary.json"
    order_output = output_root / "jonswap_order_convergence.json"
    training_output = output_root / "training_handoff_audit.json"
    figure_stem = output_root / "family_case_examples"
    jonswap_audit = release.audits["jonswap_tma"]
    return (
        StageSpec(
            name="final_renderer",
            command=(
                python,
                str(ROOT / "scripts/render_paper_dataset_worst_cases.py"),
                "--combined-summary",
                str(combined.path),
                "--require-final-paper-dataset",
                "--output-dir",
                str(renderer_dir),
                "--workers",
                str(config.renderer_workers),
                "--block-rows",
                "256",
                "--top-count",
                "6",
            ),
            timeout_seconds=config.renderer_timeout_seconds,
            outputs=(renderer_dir,),
            validate=lambda: validate_renderer_output(renderer_dir, release),
        ),
        StageSpec(
            name="jonswap_order_diagnostic",
            command=(
                python,
                str(ROOT / "scripts/diagnose_final_jonswap_order_convergence.py"),
                "--audit",
                str(jonswap_audit.path),
                "--renderer-summary",
                str(renderer_summary),
                "--output",
                str(order_output),
            ),
            timeout_seconds=config.order_timeout_seconds,
            outputs=(order_output,),
            validate=lambda: validate_order_output(
                order_output,
                release=release,
                renderer_summary=renderer_summary,
            ),
            requires=("final_renderer",),
        ),
        StageSpec(
            name="training_handoff_audit",
            command=(
                python,
                str(ROOT / "scripts/audit_paper_dataset_training_handoff.py"),
                "--combined-summary",
                str(combined.path),
                "--seed",
                "0",
                "--output",
                str(training_output),
            ),
            timeout_seconds=config.training_timeout_seconds,
            outputs=(training_output,),
            validate=lambda: validate_training_output(training_output, combined),
        ),
        StageSpec(
            name="family_case_figure",
            command=(
                python,
                str(ROOT / "scripts/build_parameterized_dataset_case_figure.py"),
                "--combined-summary",
                str(combined.path),
                "--output-stem",
                str(figure_stem),
            ),
            timeout_seconds=config.figure_timeout_seconds,
            outputs=(
                figure_stem.with_suffix(".pdf"),
                figure_stem.with_suffix(".png"),
                figure_stem.with_suffix(".json"),
            ),
            validate=lambda: validate_figure_output(figure_stem, release),
            completion_marker=figure_stem.with_suffix(".json"),
            producer_owns_uncommitted_outputs=True,
        ),
    )


def _cpu_environment(output_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "true",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "MPLCONFIGDIR": str(output_root / "mplconfig"),
        }
    )
    return environment


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _run_bounded(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    kill_after_seconds: float,
    cwd: Path,
    environment: Mapping[str, str],
) -> tuple[int, bool, bytes, bytes, float]:
    started = time.monotonic()
    process = subprocess.Popen(
        tuple(command),
        cwd=cwd,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=kill_after_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
    return (
        int(process.returncode),
        timed_out,
        stdout,
        stderr,
        time.monotonic() - started,
    )


def _outputs_present(outputs: Sequence[Path]) -> tuple[bool, ...]:
    return tuple(path.exists() or path.is_symlink() for path in outputs)


def _validate_existing(spec: StageSpec) -> Mapping[str, object] | None:
    present = _outputs_present(spec.outputs)
    if not any(present):
        return None
    if any(path.is_symlink() for path in spec.outputs):
        raise ExistingOutputError(
            f"{spec.name} has a symbolic-link output; refusing overwrite"
        )
    marker_present = spec.completion_marker is not None and (
        spec.completion_marker.exists() or spec.completion_marker.is_symlink()
    )
    if spec.producer_owns_uncommitted_outputs and not marker_present:
        non_files = tuple(
            path
            for path, exists in zip(spec.outputs, present)
            if exists and not path.is_file()
        )
        if non_files:
            raise ExistingOutputError(
                f"{spec.name} has a non-file uncommitted output; refusing overwrite"
            )
        return None
    if not all(present):
        raise ExistingOutputError(
            f"{spec.name} has a partial pre-existing output set; refusing overwrite"
        )
    try:
        return spec.validate()
    except Exception as error:
        raise ExistingOutputError(
            f"{spec.name} has unauthenticated pre-existing output; refusing overwrite: "
            f"{type(error).__name__}: {error}"
        ) from error


def execute_stage(
    spec: StageSpec,
    *,
    config: RunnerConfig,
    output_root: Path,
    invocation_log_root: Path,
    journal: RunJournal,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Validate/skip or execute one stage without replacing existing outputs."""

    existing = _validate_existing(spec)
    if existing is not None:
        result = {"status": "skipped_valid", "attempts": 0, "artifacts": dict(existing)}
        journal.stage(spec.name, result)
        journal.event("stage_skipped_valid", stage=spec.name)
        return result

    environment = _cpu_environment(output_root)
    attempts: list[dict[str, object]] = []
    for attempt in range(1, config.stage_attempts + 1):
        attempt_root = invocation_log_root / spec.name / f"attempt_{attempt:02d}"
        attempt_root.mkdir(parents=True, exist_ok=False)
        command_record = {
            "command": list(spec.command),
            "cwd": str(ROOT),
            "timeout_seconds": spec.timeout_seconds,
            "cpu_environment": {
                name: environment[name]
                for name in (
                    "CUDA_VISIBLE_DEVICES",
                    "JAX_PLATFORMS",
                    "JAX_ENABLE_X64",
                    "XLA_PYTHON_CLIENT_PREALLOCATE",
                    "MPLCONFIGDIR",
                )
            },
        }
        _write_immutable_json(attempt_root / "command.json", command_record)
        journal.event("stage_attempt_started", stage=spec.name, attempt=attempt)
        returncode, timed_out, stdout, stderr, elapsed = _run_bounded(
            spec.command,
            timeout_seconds=spec.timeout_seconds,
            kill_after_seconds=config.kill_after_seconds,
            cwd=ROOT,
            environment=environment,
        )
        _write_bytes_exclusive(attempt_root / "stdout.bin", stdout)
        _write_bytes_exclusive(attempt_root / "stderr.bin", stderr)
        attempt_record: dict[str, object] = {
            "attempt": attempt,
            "returncode": returncode,
            "timed_out": timed_out,
            "elapsed_seconds": elapsed,
            "stdout": _artifact(attempt_root / "stdout.bin").record(),
            "stderr": _artifact(attempt_root / "stderr.bin").record(),
        }
        attempts.append(attempt_record)
        try:
            validated = _validate_existing(spec)
        except ExistingOutputError as error:
            result = {
                "status": "failed_existing_output",
                "attempts": attempts,
                "error": f"{type(error).__name__}: {error}",
            }
            journal.stage(spec.name, result)
            journal.event(
                "stage_failed",
                stage=spec.name,
                attempt=attempt,
                reason=result["error"],
            )
            return result
        if validated is not None:
            result = {
                "status": "complete",
                "attempts": attempts,
                "artifacts": dict(validated),
                "committed_output_authoritative_despite_process_status": bool(
                    returncode != 0 or timed_out
                ),
            }
            journal.stage(spec.name, result)
            journal.event("stage_complete", stage=spec.name, attempt=attempt)
            return result
        reason = (
            f"timeout after {spec.timeout_seconds}s"
            if timed_out
            else f"exit {returncode} without a valid committed output"
        )
        journal.event(
            "stage_attempt_failed", stage=spec.name, attempt=attempt, reason=reason
        )
        if attempt < config.stage_attempts:
            sleeper(config.retry_delay_seconds)
    result = {
        "status": "failed",
        "attempts": attempts,
        "error": reason,
    }
    journal.stage(spec.name, result)
    return result


def _invocation_id() -> str:
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    return f"{timestamp}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _release_output_root(config: RunnerConfig, release: ReleaseBinding) -> Path:
    """Return the SHA-keyed root shared by the runner and document handoff."""

    return config.state_root / release.combined.summary.sha256


def _terminal_document_handoff_path(output_root: Path) -> Path:
    """Return the canonical terminal marker for one authenticated release."""

    return output_root / DOCUMENT_HANDOFF_NAME


def _terminal_document_handoff_exists(output_root: Path) -> bool:
    """Fail closed on any filesystem entry at the terminal handoff path."""

    handoff = _terminal_document_handoff_path(output_root)
    return handoff.exists() or handoff.is_symlink()


def _lock_release_root(output_root: Path) -> BinaryIO | None:
    """Acquire the shared release lock without waiting for another lifecycle."""

    output_root.mkdir(parents=True, exist_ok=True)
    lock = (output_root / "runner.lock").open("a+b")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    return lock


def run_postcompletion(
    config: RunnerConfig,
    *,
    authenticate: Callable[[Path, AuditPaths], ReleaseBinding] = authenticate_release,
    stage_factory: Callable[
        [RunnerConfig, ReleaseBinding, Path], Sequence[StageSpec]
    ] = default_stages,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    """Wait, authenticate, and run all independent post-completion diagnostics."""

    required_inputs = (config.combined_summary, *config.audits.all())
    release: ReleaseBinding | None = None
    release_lock: BinaryIO | None = None
    if all(path.is_file() for path in required_inputs):
        try:
            release = authenticate(config.combined_summary, config.audits)
        except Exception:
            # Preserve the existing journaled authentication-failure path below.
            release = None
        else:
            output_root = _release_output_root(config, release)
            release_lock = _lock_release_root(output_root)
            if release_lock is None:
                return 1
            if _terminal_document_handoff_exists(output_root):
                release_lock.close()
                return 1

    invocation_id = _invocation_id()
    waiting = RunJournal(
        config.state_root,
        invocation_id,
        {
            "phase": "waiting",
            "combined_summary_path": str(config.combined_summary.resolve()),
            "required_audit_paths": [
                str(path.resolve()) for path in config.audits.all()
            ],
        },
    )

    def report_wait(missing: tuple[Path, ...]) -> None:
        waiting.record["missing_inputs"] = [str(path.resolve()) for path in missing]
        waiting.event("waiting_for_inputs", missing=len(missing))

    if release is None:
        try:
            wait_for_inputs(
                required_inputs,
                poll_seconds=config.poll_seconds,
                timeout_seconds=config.wait_timeout_seconds,
                sleeper=sleeper,
                monotonic=monotonic,
                on_poll=report_wait,
            )
        except Exception as error:
            waiting.record["error"] = f"{type(error).__name__}: {error}"
            waiting.finish("wait_failed")
            return 1

    if release is None:
        try:
            release = authenticate(config.combined_summary, config.audits)
        except Exception as error:
            waiting.record["error"] = f"{type(error).__name__}: {error}"
            waiting.finish("authentication_failed")
            return 1

    output_root = _release_output_root(config, release)
    if release_lock is None:
        release_lock = _lock_release_root(output_root)
        if release_lock is None:
            waiting.record.pop("missing_inputs", None)
            waiting.event("release_inputs_present")
            waiting.record["error"] = "another runner owns this release SHA"
            waiting.finish("lock_failed")
            return 1
    with release_lock:
        if _terminal_document_handoff_exists(output_root):
            return 1
        waiting.record.pop("missing_inputs", None)
        waiting.event("release_inputs_present")
        try:
            _write_immutable_json(
                output_root / "release_identity.json", release.identity_record()
            )
        except Exception as error:
            waiting.record["error"] = f"{type(error).__name__}: {error}"
            waiting.finish("release_identity_failed")
            return 1

        waiting.record["phase"] = "authenticated"
        waiting.record["output_root"] = str(output_root.resolve())
        waiting.record["combined_summary_sha256"] = release.combined.summary.sha256
        waiting.event("release_authenticated")
        journal = RunJournal(
            output_root,
            invocation_id,
            {
                "phase": "diagnostics",
                "combined_summary": release.combined.summary.record(),
                "release_evidence": release.identity_record(),
                "release_identity": _artifact(
                    output_root / "release_identity.json"
                ).record(),
                "diagnostics_affect_dataset_release": False,
            },
        )
        log_root = output_root / "attempt_logs" / invocation_id
        log_root.mkdir(parents=True, exist_ok=False)
        results: dict[str, dict[str, object]] = {}
        stage_specs = tuple(stage_factory(config, release, output_root))
        stage_names = tuple(spec.name for spec in stage_specs)
        if len(set(stage_names)) != len(stage_names):
            duplicate_names = sorted(
                name for name in set(stage_names) if stage_names.count(name) > 1
            )
            error = "postcompletion stage names must be unique: " + ", ".join(
                duplicate_names
            )
            failure = {
                "status": "failed",
                "attempts": 0,
                "error": f"RuntimeError: {error}",
            }
            results["stage_configuration"] = failure
            journal.stage("stage_configuration", failure)
            journal.event(
                "stage_configuration_failed",
                reason=failure["error"],
            )
            journal.record["release_reauthentication"] = {
                "status": "not_run",
                "verified_unchanged": False,
                "reason": "invalid duplicate stage names",
            }
            journal.record["release_artifacts_mutated_by_runner"] = False
            journal.record["release_evidence_unchanged_after_stages"] = False
            journal.record["diagnostic_failures"] = ["stage_configuration"]
            journal.finish("diagnostics_failed")
            waiting.record["phase"] = "finished"
            waiting.record["child_status_path"] = str(journal.status_path.resolve())
            waiting.record["child_status_sha256"] = _sha256(journal.status_path)
            waiting.record["diagnostic_failures"] = ["stage_configuration"]
            waiting.finish("diagnostics_failed")
            return 1
        for spec in stage_specs:
            unmet = tuple(
                dependency
                for dependency in spec.requires
                if results.get(dependency, {}).get("status")
                not in {"complete", "skipped_valid"}
            )
            if unmet:
                result = {
                    "status": "not_run_dependency_failed",
                    "attempts": 0,
                    "dependencies": list(unmet),
                }
                journal.stage(spec.name, result)
                journal.event(
                    "stage_not_run", stage=spec.name, dependencies=list(unmet)
                )
            else:
                try:
                    result = execute_stage(
                        spec,
                        config=config,
                        output_root=output_root,
                        invocation_log_root=log_root,
                        journal=journal,
                        sleeper=sleeper,
                    )
                except Exception as error:
                    result = {
                        "status": "failed",
                        "attempts": 0,
                        "error": f"{type(error).__name__}: {error}",
                    }
                    journal.stage(spec.name, result)
                    journal.event(
                        "stage_failed", stage=spec.name, reason=result["error"]
                    )
            results[spec.name] = result

        initial_identity = release.identity_record()
        reauthentication: dict[str, object]
        try:
            final_release = authenticate(config.combined_summary, config.audits)
            final_identity = final_release.identity_record()
            if not _same_json(final_identity, initial_identity):
                raise RuntimeError(
                    "release evidence changed after initial authentication"
                )
            reauthentication = {
                "status": "pass",
                "verified_unchanged": True,
                "combined_summary_sha256": release.combined.summary.sha256,
                "release_identity_fingerprint": _canonical_sha256(final_identity),
            }
            journal.event("release_reauthenticated", status="pass")
        except Exception as error:
            reauthentication = {
                "status": "failed",
                "verified_unchanged": False,
                "error": f"{type(error).__name__}: {error}",
            }
            results["release_reauthentication"] = reauthentication
            journal.event(
                "release_reauthenticated",
                status="failed",
                reason=reauthentication["error"],
            )
        journal.record["release_reauthentication"] = reauthentication

        # Every successful diagnostic is revalidated after the potentially long
        # release authentication. This binds the final status to one exact,
        # current artifact map for every stage, including implementation proofs.
        for spec in stage_specs:
            previous = results.get(spec.name)
            if previous is None or previous.get("status") not in {
                "complete",
                "skipped_valid",
            }:
                continue
            try:
                final_artifacts = dict(spec.validate())
                if not _same_json(previous.get("artifacts"), final_artifacts):
                    raise RuntimeError(
                        f"{spec.name} artifact map changed after initial validation"
                    )
                refreshed = {**previous, "artifacts": final_artifacts}
                results[spec.name] = refreshed
                journal.stage(spec.name, refreshed)
                journal.event("stage_final_revalidated", stage=spec.name, status="pass")
            except Exception as error:
                failed_stage = {
                    **previous,
                    "status": "failed_final_validation",
                    "error": f"{type(error).__name__}: {error}",
                }
                results[spec.name] = failed_stage
                journal.stage(spec.name, failed_stage)
                journal.event(
                    "stage_final_revalidated",
                    stage=spec.name,
                    status="failed",
                    reason=failed_stage["error"],
                )

        success_statuses = {"complete", "skipped_valid"}
        failed = {
            name: record
            for name, record in results.items()
            if record.get("status") not in success_statuses
        }
        journal.record["release_artifacts_mutated_by_runner"] = False
        journal.record["release_evidence_unchanged_after_stages"] = (
            reauthentication["status"] == "pass"
        )
        journal.record["diagnostic_failures"] = sorted(failed)
        final_status = "complete" if not failed else "diagnostics_failed"
        journal.finish(final_status)
        waiting.record["phase"] = "finished"
        waiting.record["child_status_path"] = str(journal.status_path.resolve())
        waiting.record["child_status_sha256"] = _sha256(journal.status_path)
        waiting.record["diagnostic_failures"] = sorted(failed)
        waiting.finish(final_status)
        return 0 if not failed else 1


def _positive_number(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _poll_number(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 1.0 <= parsed <= 60.0:
        raise argparse.ArgumentTypeError("must be between 1 and 60 seconds")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    final_root = (
        ROOT / "outputs/paper_dataset_literature_aligned_v1/combined/"
        "c16384_v01024_t01024"
    )
    parser.add_argument(
        "--combined-summary",
        type=Path,
        default=final_root / "paper_dataset_all_splits_c16384.summary.json",
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=ROOT / "outputs/paper_dataset_final_postcompletion",
    )
    parser.add_argument("--python", type=Path, default=ROOT / ".venv/bin/python")
    stokes = ROOT / "outputs/paper_dataset_cap4_revision2_20260728"
    bf = (
        ROOT
        / "outputs/paper_dataset_bf_revision4_jonswap_revision3_literature_aligned_v1"
    )
    jonswap = ROOT / "outputs/paper_dataset_jonswap_revision4_relative_band_v1"
    tanaka = ROOT / "outputs/paper_dataset_revision3_literature_aligned_v1"
    parser.add_argument(
        "--stokes-binding",
        type=Path,
        default=stokes / "stokes_completion_binding.json",
    )
    parser.add_argument(
        "--stokes-legacy-audit",
        type=Path,
        default=stokes / "stokes_completion_audit.json",
    )
    parser.add_argument(
        "--bf-audit",
        type=Path,
        default=bf / "benjamin_feir_completion_audit.json",
    )
    parser.add_argument(
        "--jonswap-audit",
        type=Path,
        default=jonswap / "jonswap_tma_completion_audit.json",
    )
    parser.add_argument(
        "--tanaka-audit",
        type=Path,
        default=tanaka / "tanaka_completion_audit.json",
    )
    parser.add_argument("--poll-seconds", type=_poll_number, default=60.0)
    parser.add_argument(
        "--wait-timeout-seconds",
        type=_positive_number,
        help="Optional wait deadline; default waits indefinitely.",
    )
    parser.add_argument("--stage-attempts", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--retry-delay-seconds", type=_positive_number, default=10.0)
    parser.add_argument("--kill-after-seconds", type=_positive_number, default=120.0)
    parser.add_argument(
        "--renderer-timeout-seconds", type=_positive_number, default=7_200.0
    )
    parser.add_argument(
        "--order-timeout-seconds", type=_positive_number, default=3_600.0
    )
    parser.add_argument(
        "--training-timeout-seconds", type=_positive_number, default=7_200.0
    )
    parser.add_argument(
        "--figure-timeout-seconds", type=_positive_number, default=3_600.0
    )
    parser.add_argument("--renderer-workers", type=int, choices=range(1, 65), default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # Preserve a virtual-environment launcher path.  Resolving its symlink to
    # the base interpreter strips the venv context from stage subprocesses.
    python = args.python.expanduser().absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(f"Python executable is unavailable: {python}")
    config = RunnerConfig(
        combined_summary=args.combined_summary.expanduser().resolve(),
        audits=AuditPaths(
            stokes_binding=args.stokes_binding.expanduser().resolve(),
            stokes_legacy=args.stokes_legacy_audit.expanduser().resolve(),
            benjamin_feir=args.bf_audit.expanduser().resolve(),
            jonswap_tma=args.jonswap_audit.expanduser().resolve(),
            tanaka=args.tanaka_audit.expanduser().resolve(),
        ),
        state_root=args.state_root.expanduser().resolve(),
        python=python,
        poll_seconds=args.poll_seconds,
        wait_timeout_seconds=args.wait_timeout_seconds,
        stage_attempts=args.stage_attempts,
        retry_delay_seconds=args.retry_delay_seconds,
        kill_after_seconds=args.kill_after_seconds,
        renderer_timeout_seconds=args.renderer_timeout_seconds,
        order_timeout_seconds=args.order_timeout_seconds,
        training_timeout_seconds=args.training_timeout_seconds,
        figure_timeout_seconds=args.figure_timeout_seconds,
        renderer_workers=args.renderer_workers,
    )
    return run_postcompletion(config)


if __name__ == "__main__":
    raise SystemExit(main())
