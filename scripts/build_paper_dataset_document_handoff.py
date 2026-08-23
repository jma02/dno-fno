"""Build one immutable, document-only handoff from completed dataset evidence.

The caller-selected release input is a SHA-keyed postcompletion output
directory, or its ``status.json``/``release_identity.json`` file.  This script
does not discover or accept independent dataset paths.  It reauthenticates the
release, all four cumulative combined views, every source category count, all
four validated postcompletion stages, and the fixed path and SHA-256 of the
completed revision-4 JONSWAP constructor-adoption gate.  It then copies only
scoped facts needed to update the paper into one strict JSON record.  The
adoption criterion is historical pre-bulk evidence, not a final bulk release
predicate.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from scripts import build_paper_dataset_view as combined_view_builder
from scripts import run_paper_dataset_postcompletion as postcompletion


SCHEMA = "paper_dataset_document_handoff_v1"
QUOTA_SUMMARY_SCHEMA = "paper_dataset_quota_summary_v1"
DEFAULT_NAME = postcompletion.DOCUMENT_HANDOFF_NAME
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
JONSWAP_ADOPTION_GATE_PATH = (
    REPOSITORY_ROOT / "outputs/jonswap_relative_band_fresh_gate_20260804/"
    "jonswap_relative_band_fresh_gate.summary.json"
)
JONSWAP_ADOPTION_GATE_SHA256 = (
    "4228660e0db7fbf83caa05596d5409a93e183d59454453ded4b3aeee2670a9d8"
)
JONSWAP_ADOPTION_GATE_SCHEMA = "jonswap_relative_band_fresh_gate_summary_v1"
JONSWAP_ADOPTION_GATE_COUNTS = {
    "accepted": 540,
    "attempted": 548,
    "rejected": 8,
}
JONSWAP_ADOPTION_GATE_CRITERION = {
    "accepted_per_cell": 20,
    "accepted_total": 540,
    "maximum_attempts_per_accepted_case": 4,
    "maximum_rejection_rate": 0.02,
}
STAGE_NAMES = (
    "final_renderer",
    "jonswap_order_diagnostic",
    "training_handoff_audit",
    "family_case_figure",
)
PRIMARY_AUDITS = (
    "stokes",
    "tanaka",
    "benjamin_feir",
    "jonswap_tma",
)
CUMULATIVE_TRAINING_CASES = (2_048, 4_096, 8_192, 16_384)
CATEGORY_COUNT_BY_FAMILY = {
    "stokes": 4,
    "tanaka": 11,
    "benjamin_feir": 66,
    "jonswap_tma": 27,
}
STOKES_CATEGORY_IDS = frozenset(
    {"finite_low", "finite_moderate", "deep_low", "deep_moderate"}
)
JONSWAP_POPULATION_CONDITIONING = {
    "proposal_law_applies_to": "attempted_specifications",
    "released_case_law": (
        "proposal_conditioned_on_complete_case_acceptance_within_preassigned_cell"
    ),
    "cell_marginals": "accepted_quota",
    "posthoc_parameter_gate": False,
}
FAMILY_ACCEPTANCE_EVENTS = {
    "stokes": (
        "declared_static_support_finite_state_positive_water_column_and_finite_target"
    ),
    "tanaka": (
        "declared_construction_support_then_complete_autonomous_trajectory_acceptance"
    ),
    "benjamin_feir": (
        "declared_initial_state_validity_then_complete_autonomous_trajectory_acceptance"
    ),
    "jonswap_tma": (
        "declared_initial_construction_and_graph_validity_then_nonlinear_adjustment_"
        "and_complete_autonomous_trajectory_acceptance"
    ),
}
RELEASE_POPULATION_CONDITIONING = {
    **JONSWAP_POPULATION_CONDITIONING,
    "family_acceptance_events": FAMILY_ACCEPTANCE_EVENTS,
}
FAMILY_AUDIT_SCHEMAS = {
    "stokes": "paper_dataset_stokes_revision2_completion_binding_v1",
    "tanaka": "paper_dataset_tanaka_revision3_completion_audit_v1",
    "benjamin_feir": "paper_dataset_benjamin_feir_revision4_completion_audit_v1",
    "jonswap_tma": "paper_dataset_jonswap_tma_revision4_completion_audit_v1",
}
STOKES_LEGACY_AUDIT_SCHEMA = "paper_dataset_cap4_stokes_audit_v1"
SUPPORT_EXTREMA_METRICS = {
    "benjamin_feir": (
        "carrier_mode",
        "sideband_offset",
        "carrier_steepness",
        "sideband_ratio",
        "translation",
        "instability_band_fraction",
        "focused_steepness",
    ),
    "jonswap_tma": (
        "depth",
        "significant_height",
        "peak_wavenumber",
        "peak_enhancement",
        "right_moving_fraction",
        "depth_wavenumber",
        "relative_height",
        "peak_steepness",
        "resolved_maximum_relative_frequency",
        "phase_count_per_direction",
    ),
    "tanaka": (
        "depth",
        "total_alpha",
        "crest_alpha",
        "physical_crest_amplitude",
        "resolution_ratio",
        "minimum_separation",
    ),
}
HEALTH_EXTREMA_METRICS = {
    "benjamin_feir": {
        "accepted": (
            "initial_hamiltonian",
            "hamiltonian_drift",
            "stage_residual",
            "minimum_water_column",
        ),
        "all_finite_attempted": (
            "hamiltonian_drift",
            "stage_residual",
            "minimum_water_column",
        ),
    },
    "jonswap_tma": {
        "accepted": (
            "initial_hamiltonian",
            "hamiltonian_drift",
            "stage_residual",
            "minimum_water_column",
            "adjustment_stage_residual",
            "adjustment_minimum_water_column",
        ),
        "all_finite_attempted": (
            "hamiltonian_drift",
            "stage_residual",
            "minimum_water_column",
            "adjustment_stage_residual",
            "adjustment_minimum_water_column",
        ),
    },
}
RENDERER_PARAMETER_FIELDS = (
    "final_paper_dataset_contract_required",
    "delivered_maximum_wavenumber",
    "high_band_minimum_wavenumber",
    "cyclic_difference_relative_dead_zone",
    "morphology_diagnostics_are_release_thresholds",
    "rank_one_fixed_axis_gifs",
    "gif_maximum_frames",
    "gif_short_frame_threshold",
    "gif_short_fps",
    "gif_long_fps",
    "gif_dimensions_pixels",
    "gif_y_limit_scope",
    "gif_y_limit_padding_fraction",
    "gif_loop_forever",
    "gif_descriptive_only",
)
RENDERER_QUANTILE_METRICS = (
    "eta_slope",
    "gxi_high_band",
    "gxi_sign_changes",
    "stored_band_quadratic_energy_drift",
    "minimum_water_fraction",
)
RENDERER_QUANTILES = ("q000", "q050", "q090", "q095", "q099", "q100")
ORDER_IDENTITY_FIELDS = (
    "case_id",
    "chunk_label",
    "split",
    "batch_id",
    "local_index",
    "frame_index",
    "cell_id",
    "shard_path",
)
ORDER_NUMERICAL_FIELDS = (
    "execution_platform",
    "dtype",
    "nx",
    "length",
    "input_and_output_projection",
    "zero_output_mean",
    "pad_factor",
    "cumulative_orders",
    "relative_l2_definition",
)

ReleaseAuthenticator = Callable[
    [Path, postcompletion.AuditPaths], postcompletion.ReleaseBinding
]


@dataclass(frozen=True)
class AuthenticatedPostcompletion:
    """Authenticated completed release and its four stage records."""

    root: Path
    status: postcompletion.Artifact
    release_identity: postcompletion.Artifact
    jonswap_adoption_gate: postcompletion.Artifact
    release: postcompletion.ReleaseBinding
    status_record: Mapping[str, Any]
    identity_record: Mapping[str, Any]
    jonswap_adoption_gate_record: Mapping[str, Any]
    cumulative_combined_views: Mapping[int, postcompletion.CombinedBinding]
    renderer_record: Mapping[str, Any]
    order_record: Mapping[str, Any]
    training_record: Mapping[str, Any]
    figure_record: Mapping[str, Any]


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _sequence(value: object, *, context: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a JSON array")
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


def _rejection_rate(*, attempted: object, rejected: object, context: str) -> float:
    attempted_count = _integer(
        attempted,
        context=f"{context} attempted count",
        minimum=1,
    )
    rejected_count = _integer(
        rejected,
        context=f"{context} rejected count",
    )
    if rejected_count > attempted_count:
        raise ValueError(f"{context} rejected count exceeds attempted count")
    return _finite_number(
        rejected_count / attempted_count,
        context=f"{context} rejection rate",
    )


def _string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a nonempty string")
    return value


def _digest(value: object, *, context: str) -> str:
    return postcompletion._digest(value, context=context)


def _same(left: object, right: object) -> bool:
    return postcompletion._same_json(left, right)


def _read_bound_json(
    path: Path,
    artifact_record: object,
    *,
    context: str,
) -> dict[str, Any]:
    """Read exactly the JSON bytes authenticated by one artifact record."""

    expected = _mapping(artifact_record, context=f"{context} artifact")
    requested = path.expanduser()
    if requested.is_symlink():
        raise ValueError(f"{context} must not be a symbolic link: {requested}")
    resolved = requested.resolve(strict=True)
    expected_path_value = expected.get("path")
    if (
        not isinstance(expected_path_value, str)
        or not Path(expected_path_value).is_absolute()
        or Path(expected_path_value).resolve(strict=True) != resolved
    ):
        raise RuntimeError(f"{context} path differs from its validated artifact")
    encoded = resolved.read_bytes()
    expected_bytes = _integer(expected.get("bytes"), context=f"{context} bytes")
    expected_sha256 = _digest(expected.get("sha256"), context=f"{context} SHA-256")
    if len(encoded) != expected_bytes:
        raise RuntimeError(f"{context} byte count changed after validation")
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise RuntimeError(f"{context} SHA-256 changed after validation")

    def reject_constant(value: str) -> None:
        raise ValueError(f"{context} contains nonfinite JSON constant {value!r}")

    value = json.loads(encoded.decode("utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"{context} must contain a JSON object")
    return value


def _json_artifact(
    artifact: postcompletion.Artifact,
    record: Mapping[str, Any],
    *,
    require_status: bool = True,
) -> dict[str, object]:
    schema = record.get("schema")
    status = record.get("status")
    if not isinstance(schema, str) or not schema:
        raise ValueError(f"JSON artifact omits schema: {artifact.path}")
    if require_status and (not isinstance(status, str) or not status):
        raise ValueError(f"JSON artifact omits status: {artifact.path}")
    result = {**artifact.record(), "schema": schema}
    if isinstance(status, str) and status:
        result["status"] = status
    return result


def _authenticated_proof(
    record: Mapping[str, object], *, context: str
) -> dict[str, object]:
    """Return a detached canonical copy of one strictly validated stage record."""

    encoded = json.dumps(
        record,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    proof = json.loads(encoded)
    if not isinstance(proof, dict):
        raise TypeError(f"{context} authenticated proof must be an object")
    return proof


def _resolve_release_root(source: Path) -> Path:
    requested = source.expanduser()
    if requested.is_symlink():
        raise ValueError(
            f"postcompletion input must not be a symbolic link: {requested}"
        )
    resolved = requested.resolve(strict=True)
    if resolved.is_file():
        if resolved.name not in {"status.json", "release_identity.json"}:
            raise ValueError(
                "postcompletion file input must be status.json or release_identity.json"
            )
        root = resolved.parent
    elif resolved.is_dir():
        root = resolved
    else:
        raise ValueError("postcompletion input must be a directory or regular file")
    if postcompletion.SHA256_PATTERN.fullmatch(root.name) is None:
        raise ValueError("postcompletion output directory must be SHA-256 keyed")
    return root


def _authenticate_jonswap_adoption_gate() -> tuple[
    postcompletion.Artifact, dict[str, Any]
]:
    """Authenticate the fixed revision-4 constructor-adoption evidence."""

    expected_path = JONSWAP_ADOPTION_GATE_PATH.expanduser()
    if not expected_path.is_absolute():
        raise ValueError("JONSWAP adoption-gate path must be absolute")
    artifact = postcompletion._artifact(
        expected_path,
        expected_sha256=JONSWAP_ADOPTION_GATE_SHA256,
    )
    if artifact.path != expected_path.resolve(strict=True):
        raise RuntimeError("JONSWAP adoption-gate path differs from its fixed path")
    record = _read_bound_json(
        artifact.path,
        artifact.record(),
        context="JONSWAP revision-4 adoption gate",
    )
    expected_top_level = {
        "observed",
        "passed",
        "predeclared_gate",
        "schema",
        "source",
        "status",
        "streams",
    }
    if set(record) != expected_top_level:
        raise RuntimeError("JONSWAP adoption gate has the wrong fields")
    if (
        record.get("schema") != JONSWAP_ADOPTION_GATE_SCHEMA
        or record.get("status") != "pass"
        or record.get("passed") is not True
    ):
        raise RuntimeError("JONSWAP adoption gate is not the passed v1 record")

    observed = _mapping(
        record.get("observed"), context="JONSWAP adoption-gate observations"
    )
    expected_observed_fields = {
        "accepted",
        "attempted",
        "rejected",
        "rejection_rate",
        "rejection_reasons",
    }
    if set(observed) != expected_observed_fields:
        raise RuntimeError("JONSWAP adoption-gate observations have the wrong fields")
    counts = {
        name: _integer(observed.get(name), context=f"JONSWAP adoption-gate {name}")
        for name in JONSWAP_ADOPTION_GATE_COUNTS
    }
    if counts != JONSWAP_ADOPTION_GATE_COUNTS:
        raise RuntimeError("JONSWAP adoption-gate counts differ")
    rejection_rate = _finite_number(
        observed.get("rejection_rate"),
        context="JONSWAP adoption-gate rejection rate",
    )
    expected_rate = counts["rejected"] / counts["attempted"]
    if rejection_rate != expected_rate:
        raise RuntimeError("JONSWAP adoption-gate rejection rate differs")

    criterion = _mapping(
        record.get("predeclared_gate"),
        context="JONSWAP adoption-gate criterion",
    )
    if not _same(criterion, JONSWAP_ADOPTION_GATE_CRITERION):
        raise RuntimeError("JONSWAP adoption-gate criterion differs")
    if rejection_rate > JONSWAP_ADOPTION_GATE_CRITERION["maximum_rejection_rate"]:
        raise RuntimeError("JONSWAP adoption gate does not satisfy its criterion")
    _mapping(record.get("source"), context="JONSWAP adoption-gate source")
    _sequence(record.get("streams"), context="JONSWAP adoption-gate streams")
    return artifact, record


def _audit_paths(identity: Mapping[str, Any]) -> postcompletion.AuditPaths:
    audits = _mapping(identity.get("family_audits"), context="release family audits")
    expected = {*PRIMARY_AUDITS, "stokes_legacy"}
    if set(audits) != expected:
        raise RuntimeError("release identity has the wrong family-audit set")

    def path(name: str) -> Path:
        record = _mapping(audits[name], context=f"release audit {name}")
        value = record.get("path")
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"release audit {name} path must be absolute")
        return Path(value)

    return postcompletion.AuditPaths(
        stokes_binding=path("stokes"),
        stokes_legacy=path("stokes_legacy"),
        benjamin_feir=path("benjamin_feir"),
        jonswap_tma=path("jonswap_tma"),
        tanaka=path("tanaka"),
    )


def _cumulative_view_path(final_summary: Path, training_cases: int) -> Path:
    tag = f"{training_cases:05d}"
    combined_root = final_summary.parent.parent
    return (
        combined_root
        / f"c{tag}_v01024_t01024"
        / f"paper_dataset_all_splits_c{tag}.summary.json"
    )


def _cumulative_view_chunks(
    final_chunks: Sequence[Mapping[str, object]],
    *,
    training_cases: int,
) -> tuple[Mapping[str, object], ...]:
    chunks = tuple(
        chunk
        for chunk in final_chunks
        if chunk.get("split") != "train"
        or _integer(
            chunk.get("accepted_after"),
            context="combined-view chunk accepted-after count",
        )
        <= training_cases
    )
    train_counts = {
        family: sum(
            _integer(
                chunk.get("accepted_count"),
                context=f"{family} cumulative-view accepted count",
            )
            for chunk in chunks
            if chunk.get("family") == family and chunk.get("split") == "train"
        )
        for family in postcompletion.FAMILY_ORDER
    }
    if set(train_counts.values()) != {training_cases}:
        raise RuntimeError("cumulative-view source prefixes do not close")
    return chunks


def _cumulative_rows_by_split_and_family(
    training_cases: int,
) -> dict[str, dict[str, int]]:
    cases_by_split = {
        "train": training_cases,
        "validation": 1_024,
        "test": 1_024,
    }
    return {
        split: {
            family: cases * postcompletion.ROWS_PER_CASE[family]
            for family in postcompletion.FAMILY_ORDER
        }
        for split, cases in cases_by_split.items()
    }


def _authenticate_cumulative_view(
    path: Path,
    *,
    training_cases: int,
    final_chunks: Sequence[Mapping[str, object]],
    final_dataset_contract: Mapping[str, Any],
    final_dataset_contract_fingerprint: str,
) -> postcompletion.CombinedBinding:
    """Authenticate one exact cumulative view against final release sources."""

    tag = f"{training_cases:05d}"
    expected_name = f"paper_dataset_all_splits_c{tag}"
    expected_parent = f"c{tag}_v01024_t01024"
    if (
        path.name != f"{expected_name}.summary.json"
        or path.parent.name != expected_parent
    ):
        raise ValueError(f"cumulative c{tag} combined-summary path differs")
    summary_artifact = postcompletion._artifact(path)
    summary = _read_bound_json(
        summary_artifact.path,
        summary_artifact.record(),
        context=f"cumulative c{tag} combined summary",
    )
    if (
        summary.get("schema") != postcompletion.COMBINED_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
    ):
        raise RuntimeError(f"cumulative c{tag} combined summary is incomplete")

    chunks = _cumulative_view_chunks(
        final_chunks,
        training_cases=training_cases,
    )
    summary_paths = tuple(
        Path(
            _string(
                chunk.get("summary_path"),
                context=f"cumulative c{tag} source-summary path",
            )
        )
        for chunk in chunks
    )
    plan, output_root, view_name, expected_preflight = combined_view_builder.preflight(
        summary_paths,
        output_root=summary_artifact.path.parent,
        name=expected_name,
    )
    if output_root != summary_artifact.path.parent or view_name != expected_name:
        raise RuntimeError(f"cumulative c{tag} canonical view layout differs")
    preflight = _mapping(
        summary.get("preflight"), context=f"cumulative c{tag} preflight"
    )
    if not _same(preflight, expected_preflight):
        raise RuntimeError(f"cumulative c{tag} preflight differs from canonical plan")

    view = _mapping(
        summary.get("dataset_view"), context=f"cumulative c{tag} dataset view"
    )
    manifest = postcompletion._artifact_from_record(
        summary_artifact.path.parent,
        view.get("manifest"),
        context=f"cumulative c{tag} manifest",
    )
    trajectory_map = postcompletion._artifact_from_record(
        summary_artifact.path.parent,
        view.get("trajectory_map"),
        context=f"cumulative c{tag} trajectory map",
    )
    manifest_record = _read_bound_json(
        manifest.path,
        manifest.record(),
        context=f"cumulative c{tag} manifest",
    )
    dataset_contract = _mapping(
        manifest_record.get("dataset_contract"),
        context=f"cumulative c{tag} dataset contract",
    )
    dataset_contract_fingerprint = _digest(
        manifest_record.get("dataset_contract_fingerprint"),
        context=f"cumulative c{tag} dataset-contract fingerprint",
    )
    if (
        not _same(dataset_contract, final_dataset_contract)
        or dataset_contract_fingerprint != final_dataset_contract_fingerprint
    ):
        raise RuntimeError(
            f"cumulative c{tag} dataset contract differs from final release"
        )
    validated_view = combined_view_builder._validate_view(
        combined_view_builder.DatasetViewPaths(
            manifest=manifest.path,
            trajectory_map=trajectory_map.path,
        ),
        plan=plan,
    )
    if not _same(view, validated_view):
        raise RuntimeError(
            f"cumulative c{tag} dataset-view record differs from canonical validation"
        )
    return postcompletion.CombinedBinding(
        summary=summary_artifact,
        manifest=manifest,
        trajectory_map=trajectory_map,
        chunks=tuple(dict(chunk) for chunk in chunks),
    )


def _authenticate_cumulative_views(
    final: postcompletion.CombinedBinding,
) -> dict[int, postcompletion.CombinedBinding]:
    expected_final = _cumulative_view_path(
        final.summary.path,
        CUMULATIVE_TRAINING_CASES[-1],
    )
    if final.summary.path != expected_final:
        raise RuntimeError(
            "final combined summary is outside the cumulative-view layout"
        )
    final_manifest = _read_bound_json(
        final.manifest.path,
        final.manifest.record(),
        context="final combined manifest",
    )
    final_dataset_contract = _mapping(
        final_manifest.get("dataset_contract"),
        context="final combined dataset contract",
    )
    final_dataset_contract_fingerprint = _digest(
        final_manifest.get("dataset_contract_fingerprint"),
        context="final combined dataset-contract fingerprint",
    )
    if final_dataset_contract_fingerprint != postcompletion._canonical_sha256(
        final_dataset_contract
    ):
        raise RuntimeError("final combined dataset-contract fingerprint differs")
    views: dict[int, postcompletion.CombinedBinding] = {}
    for training_cases in CUMULATIVE_TRAINING_CASES:
        path = _cumulative_view_path(final.summary.path, training_cases)
        authenticated = _authenticate_cumulative_view(
            path,
            training_cases=training_cases,
            final_chunks=final.chunks,
            final_dataset_contract=final_dataset_contract,
            final_dataset_contract_fingerprint=final_dataset_contract_fingerprint,
        )
        if training_cases == CUMULATIVE_TRAINING_CASES[-1] and authenticated != final:
            raise RuntimeError(
                "canonical c16384 binding differs from the authenticated release"
            )
        views[training_cases] = authenticated
    return views


def _validate_completed_status(
    status: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    status_artifact: postcompletion.Artifact,
    identity_artifact: postcompletion.Artifact,
    release: postcompletion.ReleaseBinding,
    stage_artifacts: Mapping[str, Mapping[str, object]],
) -> None:
    if (
        status.get("schema") != postcompletion.RUNNER_SCHEMA
        or status.get("status") != "complete"
    ):
        raise RuntimeError("postcompletion status is not complete")
    if status_artifact.path.parent != identity_artifact.path.parent:
        raise RuntimeError("status and release identity are not adjacent")
    if not _same(status.get("combined_summary"), release.combined.summary.record()):
        raise RuntimeError("postcompletion status names a different combined summary")
    if not _same(status.get("release_evidence"), identity):
        raise RuntimeError("postcompletion status release evidence differs")
    if not _same(status.get("release_identity"), identity_artifact.record()):
        raise RuntimeError("postcompletion status release-identity artifact differs")
    if (
        status.get("release_artifacts_mutated") is not False
        or status.get("release_artifacts_mutated_by_runner") is not False
        or status.get("release_evidence_unchanged_after_stages") is not True
        or status.get("diagnostics_affect_dataset_release") is not False
        or status.get("diagnostic_failures") != []
    ):
        raise RuntimeError(
            "postcompletion status does not certify an unchanged release"
        )
    reauthentication = _mapping(
        status.get("release_reauthentication"),
        context="postcompletion release reauthentication",
    )
    if (
        reauthentication.get("status") != "pass"
        or reauthentication.get("verified_unchanged") is not True
        or reauthentication.get("combined_summary_sha256")
        != release.combined.summary.sha256
        or reauthentication.get("release_identity_fingerprint")
        != postcompletion._canonical_sha256(identity)
    ):
        raise RuntimeError("postcompletion release reauthentication differs")
    stages = _mapping(status.get("stages"), context="postcompletion stages")
    if set(stages) != set(STAGE_NAMES):
        raise RuntimeError("postcompletion status has the wrong stage set")
    for name in STAGE_NAMES:
        stage = _mapping(stages[name], context=f"postcompletion stage {name}")
        if stage.get("status") not in {"complete", "skipped_valid"}:
            raise RuntimeError(f"postcompletion stage {name} is not complete")
        if not _same(stage.get("artifacts"), stage_artifacts[name]):
            raise RuntimeError(f"postcompletion stage {name} artifact binding differs")


def authenticate_postcompletion(
    source: Path,
    *,
    release_authenticator: ReleaseAuthenticator = postcompletion.authenticate_release,
) -> AuthenticatedPostcompletion:
    """Authenticate one completed SHA-keyed postcompletion output."""

    root = _resolve_release_root(source)
    adoption_gate_artifact, adoption_gate_record = _authenticate_jonswap_adoption_gate()
    status_artifact = postcompletion._artifact(root / "status.json")
    identity_artifact = postcompletion._artifact(root / "release_identity.json")
    status = _read_bound_json(
        status_artifact.path,
        status_artifact.record(),
        context="postcompletion status",
    )
    identity = _read_bound_json(
        identity_artifact.path,
        identity_artifact.record(),
        context="release identity",
    )
    if identity.get("schema") != postcompletion.RELEASE_IDENTITY_SCHEMA:
        raise RuntimeError("release identity has the wrong schema")
    combined_record = _mapping(
        identity.get("combined_summary"), context="release combined summary"
    )
    combined_path_value = combined_record.get("path")
    if (
        not isinstance(combined_path_value, str)
        or not Path(combined_path_value).is_absolute()
    ):
        raise ValueError("release combined-summary path must be absolute")
    combined_path = Path(combined_path_value)
    combined_sha256 = _digest(
        combined_record.get("sha256"), context="release combined-summary SHA-256"
    )
    if root.name != combined_sha256:
        raise RuntimeError("postcompletion directory key differs from combined SHA-256")
    audits = _audit_paths(identity)
    release = release_authenticator(combined_path, audits)
    if len(release.combined.chunks) != postcompletion.FINAL_SOURCE_COUNT:
        raise RuntimeError("authenticated release has the wrong source count")
    cumulative_combined_views = _authenticate_cumulative_views(release.combined)
    expected_identity = release.identity_record()
    if not _same(identity, expected_identity):
        raise RuntimeError(
            "release identity differs from authenticated release evidence"
        )

    renderer_dir = root / "worst_cases"
    renderer_summary = renderer_dir / "summary.json"
    order_path = root / "jonswap_order_convergence.json"
    training_path = root / "training_handoff_audit.json"
    figure_stem = root / "family_case_examples"
    if renderer_dir.is_symlink():
        raise ValueError("renderer output directory must not be a symbolic link")
    for stage_path in (
        renderer_summary,
        order_path,
        training_path,
        figure_stem.with_suffix(".json"),
        figure_stem.with_suffix(".pdf"),
        figure_stem.with_suffix(".png"),
    ):
        if stage_path.is_symlink():
            raise ValueError(f"stage output must not be a symbolic link: {stage_path}")
    stage_artifacts = {
        "final_renderer": dict(
            postcompletion.validate_renderer_output(renderer_dir, release)
        ),
        "jonswap_order_diagnostic": dict(
            postcompletion.validate_order_output(
                order_path,
                release=release,
                renderer_summary=renderer_summary,
            )
        ),
        "training_handoff_audit": dict(
            postcompletion.validate_training_output(
                training_path, release.combined.summary
            )
        ),
        "family_case_figure": dict(
            postcompletion.validate_figure_output(figure_stem, release)
        ),
    }
    _validate_completed_status(
        status,
        identity=identity,
        status_artifact=status_artifact,
        identity_artifact=identity_artifact,
        release=release,
        stage_artifacts=stage_artifacts,
    )
    final_status = postcompletion._artifact(status_artifact.path)
    final_identity = postcompletion._artifact(identity_artifact.path)
    final_adoption_gate = postcompletion._artifact(
        adoption_gate_artifact.path,
        expected_sha256=JONSWAP_ADOPTION_GATE_SHA256,
    )
    if (
        final_status != status_artifact
        or final_identity != identity_artifact
        or final_adoption_gate != adoption_gate_artifact
    ):
        raise RuntimeError(
            "postcompletion or adoption-gate evidence changed during audit"
        )
    renderer_record = _read_bound_json(
        renderer_summary,
        _mapping(
            stage_artifacts["final_renderer"],
            context="validated renderer artifacts",
        ).get("summary"),
        context="renderer summary",
    )
    order_record = _read_bound_json(
        order_path,
        _mapping(
            stage_artifacts["jonswap_order_diagnostic"],
            context="validated order artifacts",
        ).get("output"),
        context="order diagnostic",
    )
    training_record = _read_bound_json(
        training_path,
        _mapping(
            stage_artifacts["training_handoff_audit"],
            context="validated training artifacts",
        ).get("output"),
        context="training handoff audit",
    )
    figure_record = _read_bound_json(
        figure_stem.with_suffix(".json"),
        _mapping(
            stage_artifacts["family_case_figure"],
            context="validated figure artifacts",
        ).get("sidecar"),
        context="family-case figure sidecar",
    )
    return AuthenticatedPostcompletion(
        root=root,
        status=status_artifact,
        release_identity=identity_artifact,
        jonswap_adoption_gate=adoption_gate_artifact,
        release=release,
        status_record=status,
        identity_record=identity,
        jonswap_adoption_gate_record=adoption_gate_record,
        cumulative_combined_views=cumulative_combined_views,
        renderer_record=renderer_record,
        order_record=order_record,
        training_record=training_record,
        figure_record=figure_record,
    )


def _chunk_category_counts(
    chunk: Mapping[str, object],
    *,
    family: str,
    split: str,
) -> dict[str, dict[str, int | float]]:
    summary_path = _string(
        chunk.get("summary_path"),
        context=f"{family} {split} source-summary path",
    )
    path = Path(summary_path)
    if not path.is_absolute():
        raise ValueError(f"{family} {split} source-summary path must be absolute")
    artifact = postcompletion._artifact(
        path,
        expected_sha256=chunk.get("summary_sha256"),
    )
    summary = _read_bound_json(
        artifact.path,
        artifact.record(),
        context=f"{family} {split} source summary",
    )
    if (
        summary.get("schema") != QUOTA_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
    ):
        raise RuntimeError(f"{family} {split} source summary is incomplete")
    counts = _mapping(summary.get("counts"), context=f"{family} {split} counts")
    by_cell = _mapping(
        counts.get("by_cell"), context=f"{family} {split} category counts"
    )
    if len(by_cell) != CATEGORY_COUNT_BY_FAMILY[family]:
        raise RuntimeError(f"{family} source summary has the wrong category count")
    result: dict[str, dict[str, int | float]] = {}
    for raw_name, raw_record in sorted(by_cell.items()):
        name = _string(raw_name, context=f"{family} category name")
        cell = _mapping(raw_record, context=f"{family} category {name}")
        _require_exact_keys(
            cell,
            ("target_accepted", "attempted", "accepted", "rejected"),
            context=f"{family} category {name}",
        )
        target = _integer(
            cell.get("target_accepted"),
            context=f"{family} category {name} target",
            minimum=1,
        )
        accepted = _integer(
            cell.get("accepted"),
            context=f"{family} category {name} accepted",
            minimum=1,
        )
        attempted = _integer(
            cell.get("attempted"),
            context=f"{family} category {name} attempted",
            minimum=accepted,
        )
        rejected = _integer(
            cell.get("rejected"),
            context=f"{family} category {name} rejected",
        )
        if accepted != target or attempted != accepted + rejected:
            raise RuntimeError(f"{family} category {name} counts do not close")
        result[name] = {
            "target_accepted": target,
            "attempted": attempted,
            "accepted": accepted,
            "rejected": rejected,
            "rejection_rate": _rejection_rate(
                attempted=attempted,
                rejected=rejected,
                context=f"{family} category {name}",
            ),
        }
    expected = {
        "accepted": _integer(
            chunk.get("accepted_count"), context=f"{family} chunk accepted"
        ),
        "attempted": _integer(
            chunk.get("attempted_count"), context=f"{family} chunk attempted"
        ),
    }
    expected["rejected"] = expected["attempted"] - expected["accepted"]
    for name, expected_count in expected.items():
        observed = _integer(counts.get(name), context=f"{family} source-summary {name}")
        cell_total = sum(int(record[name]) for record in result.values())
        if observed != expected_count or cell_total != expected_count:
            raise RuntimeError(f"{family} source-summary {name} count differs")
    return result


def _add_category_counts(
    totals: dict[str, dict[str, int]],
    categories: Mapping[str, Mapping[str, int | float]],
) -> None:
    for name, record in categories.items():
        target = totals.setdefault(
            name,
            {"target_accepted": 0, "attempted": 0, "accepted": 0, "rejected": 0},
        )
        for field in ("target_accepted", "attempted", "accepted", "rejected"):
            target[field] += int(record[field])


def _category_records(
    totals: Mapping[str, Mapping[str, int]],
    *,
    context: str,
) -> dict[str, dict[str, int | float]]:
    return {
        name: {
            **record,
            "rejection_rate": _rejection_rate(
                attempted=record["attempted"],
                rejected=record["rejected"],
                context=f"{context} category {name}",
            ),
        }
        for name, record in sorted(totals.items())
    }


def _jonswap_population_conditioning(
    audit: Mapping[str, Any],
    categories: Mapping[str, Mapping[str, int | float]],
) -> dict[str, object]:
    conditioning = _mapping(
        audit.get("population_conditioning"),
        context="JONSWAP population conditioning",
    )
    _require_exact_keys(
        conditioning,
        (*JONSWAP_POPULATION_CONDITIONING, "cells"),
        context="JONSWAP population conditioning",
    )
    for name, expected in JONSWAP_POPULATION_CONDITIONING.items():
        if conditioning.get(name) != expected:
            raise RuntimeError(f"JONSWAP population-conditioning {name} differs")
    cells = _mapping(
        conditioning.get("cells"), context="JONSWAP population-conditioning cells"
    )
    normalized: dict[str, dict[str, int | float]] = {}
    for raw_name, raw_record in sorted(cells.items()):
        name = _string(raw_name, context="JONSWAP conditioning cell name")
        record = _mapping(raw_record, context=f"JONSWAP conditioning cell {name}")
        _require_exact_keys(
            record,
            ("target_accepted", "attempted", "accepted", "rejected", "rejection_rate"),
            context=f"JONSWAP conditioning cell {name}",
        )
        target = _integer(
            record.get("target_accepted"),
            context=f"JONSWAP conditioning cell {name} target",
            minimum=1,
        )
        accepted = _integer(
            record.get("accepted"),
            context=f"JONSWAP conditioning cell {name} accepted",
            minimum=1,
        )
        attempted = _integer(
            record.get("attempted"),
            context=f"JONSWAP conditioning cell {name} attempted",
            minimum=accepted,
        )
        rejected = _integer(
            record.get("rejected"),
            context=f"JONSWAP conditioning cell {name} rejected",
        )
        rejection_rate = _finite_number(
            record.get("rejection_rate"),
            context=f"JONSWAP conditioning cell {name} rejection rate",
        )
        if (
            target != accepted
            or attempted != accepted + rejected
            or rejection_rate != rejected / attempted
        ):
            raise RuntimeError(f"JONSWAP conditioning cell {name} counts differ")
        normalized[name] = {
            "target_accepted": target,
            "attempted": attempted,
            "accepted": accepted,
            "rejected": rejected,
            "rejection_rate": rejection_rate,
        }
    if not _same(normalized, categories):
        raise RuntimeError(
            "JONSWAP conditioning cells differ from authenticated source summaries"
        )
    return {**JONSWAP_POPULATION_CONDITIONING, "cells": normalized}


def _validate_declared_category_identities(
    *,
    family: str,
    audit: Mapping[str, Any],
    split_records: Mapping[str, Mapping[str, object]],
    categories: Mapping[str, Mapping[str, int | float]],
) -> None:
    """Bind source-summary category IDs to the current completion evidence."""

    if family == "stokes":
        if set(categories) != STOKES_CATEGORY_IDS:
            raise RuntimeError("Stokes categories differ from the declared taxonomy")
        if any(
            set(
                _mapping(
                    split_records[split].get("categories"),
                    context=f"Stokes {split} categories",
                )
            )
            != STOKES_CATEGORY_IDS
            for split in postcompletion.SPLIT_ORDER
        ):
            raise RuntimeError("Stokes split categories differ from the taxonomy")
        return
    if family == "jonswap_tma":
        return

    accepted_by_split_and_cell = _mapping(
        audit.get("accepted_by_split_and_cell"),
        context=f"{family} audit accepted counts by split and category",
    )
    _require_exact_keys(
        accepted_by_split_and_cell,
        postcompletion.SPLIT_ORDER,
        context=f"{family} audit category splits",
    )
    declared_ids: set[str] | None = None
    for split in postcompletion.SPLIT_ORDER:
        audit_cells = _mapping(
            accepted_by_split_and_cell.get(split),
            context=f"{family} audit {split} categories",
        )
        source_cells = _mapping(
            split_records[split].get("categories"),
            context=f"{family} source {split} categories",
        )
        observed_ids = set(audit_cells)
        if declared_ids is None:
            declared_ids = observed_ids
        elif observed_ids != declared_ids:
            raise RuntimeError(f"{family} audit category identities differ by split")
        if observed_ids != set(source_cells):
            raise RuntimeError(
                f"{family} source category identities differ from its audit"
            )
        for name, expected in audit_cells.items():
            accepted = _integer(
                expected,
                context=f"{family} audit {split} category {name}",
                minimum=1,
            )
            source = _mapping(
                source_cells[name],
                context=f"{family} source {split} category {name}",
            )
            if source.get("accepted") != accepted:
                raise RuntimeError(
                    f"{family} {split} category {name} differs from its audit"
                )
    if declared_ids is None or set(categories) != declared_ids:
        raise RuntimeError(f"{family} aggregate categories differ from its audit")


def _family_counts(
    authenticated: AuthenticatedPostcompletion,
) -> tuple[dict[str, object], dict[str, object]]:
    release = authenticated.release
    result: dict[str, object] = {}
    total_accepted = 0
    total_attempted = 0
    total_rejected = 0
    total_rows = 0
    for family in postcompletion.FAMILY_ORDER:
        split_records: dict[str, dict[str, object]] = {}
        family_category_totals: dict[str, dict[str, int]] = {}
        category_names: set[str] | None = None
        for split in postcompletion.SPLIT_ORDER:
            chunks = tuple(
                chunk
                for chunk in release.combined.chunks
                if chunk["family"] == family and chunk["split"] == split
            )
            split_category_totals: dict[str, dict[str, int]] = {}
            for chunk in chunks:
                chunk_categories = _chunk_category_counts(
                    chunk,
                    family=family,
                    split=split,
                )
                observed_names = set(chunk_categories)
                if category_names is None:
                    category_names = observed_names
                elif observed_names != category_names:
                    raise RuntimeError(
                        f"{family} category identities differ across sources"
                    )
                _add_category_counts(split_category_totals, chunk_categories)
            split_categories = _category_records(
                split_category_totals,
                context=f"{family} {split}",
            )
            _add_category_counts(family_category_totals, split_categories)
            accepted = sum(
                _integer(
                    chunk.get("accepted_count"),
                    context=f"{family} {split} accepted count",
                )
                for chunk in chunks
            )
            attempted = sum(
                _integer(
                    chunk.get("attempted_count"),
                    context=f"{family} {split} attempted count",
                )
                for chunk in chunks
            )
            rejected = attempted - accepted
            category_counts = {
                name: sum(int(record[name]) for record in split_categories.values())
                for name in ("accepted", "attempted", "rejected")
            }
            if category_counts != {
                "accepted": accepted,
                "attempted": attempted,
                "rejected": rejected,
            }:
                raise RuntimeError(f"{family} {split} category totals do not close")
            split_records[split] = {
                "accepted": accepted,
                "attempted": attempted,
                "rejected": rejected,
                "rejection_rate": _rejection_rate(
                    attempted=attempted,
                    rejected=rejected,
                    context=f"{family} {split}",
                ),
                "retained_rows": accepted * postcompletion.ROWS_PER_CASE[family],
                "categories": split_categories,
            }
        accepted = sum(int(item["accepted"]) for item in split_records.values())
        attempted = sum(int(item["attempted"]) for item in split_records.values())
        rejected = attempted - accepted
        retained_rows = sum(
            int(item["retained_rows"]) for item in split_records.values()
        )
        categories = _category_records(
            family_category_totals,
            context=family,
        )
        if set(categories) != (category_names or set()):
            raise RuntimeError(f"{family} aggregate category identities differ")
        audit_artifact = release.audits[family]
        audit = _read_bound_json(
            audit_artifact.path,
            audit_artifact.record(),
            context=f"{family} completion audit",
        )
        raw_counts = audit.get("counts") if family == "stokes" else audit
        counts = _mapping(raw_counts, context=f"{family} audit counts")
        expected = {
            "accepted": accepted,
            "attempted": attempted,
            "rejected": rejected,
            "retained_rows": retained_rows,
        }
        if any(counts.get(name) != value for name, value in expected.items()):
            raise RuntimeError(f"{family} audit totals differ from combined sources")
        accepted_by_split = {
            split: split_records[split]["accepted"]
            for split in postcompletion.SPLIT_ORDER
        }
        if not _same(counts.get("accepted_by_split"), accepted_by_split):
            raise RuntimeError(f"{family} audit split totals differ")
        _validate_declared_category_identities(
            family=family,
            audit=audit,
            split_records=split_records,
            categories=categories,
        )
        family_record: dict[str, object] = {
            "revision": postcompletion.REVISION_BY_FAMILY[family],
            "rows_per_accepted_case": postcompletion.ROWS_PER_CASE[family],
            "splits": split_records,
            "categories": categories,
            **expected,
            "rejection_rate": _rejection_rate(
                attempted=attempted,
                rejected=rejected,
                context=family,
            ),
        }
        if family == "jonswap_tma":
            family_record["population_conditioning"] = _jonswap_population_conditioning(
                audit, categories
            )
        result[family] = family_record
        total_accepted += accepted
        total_attempted += attempted
        total_rejected += rejected
        total_rows += retained_rows
    totals = {
        "accepted_cases": postcompletion.FINAL_ACCEPTED_CASES,
        "attempted_cases": total_attempted,
        "rejected_cases": total_rejected,
        "rejection_rate": _rejection_rate(
            attempted=total_attempted,
            rejected=total_rejected,
            context="aggregate dataset",
        ),
        "retained_rows": postcompletion.FINAL_RETAINED_ROWS,
    }
    if (
        total_accepted != postcompletion.FINAL_ACCEPTED_CASES
        or total_attempted - total_rejected != total_accepted
        or total_rows != postcompletion.FINAL_RETAINED_ROWS
    ):
        raise RuntimeError("aggregate family counts do not close")
    return result, totals


def _audit_records(
    authenticated: AuthenticatedPostcompletion,
) -> tuple[dict[str, object], dict[str, object]]:
    primary: dict[str, object] = {}
    for family in PRIMARY_AUDITS:
        artifact = authenticated.release.audits[family]
        primary[family] = _json_artifact(
            artifact,
            _read_bound_json(
                artifact.path,
                artifact.record(),
                context=f"{family} completion audit",
            ),
        )
    legacy = authenticated.release.audits["stokes_legacy"]
    supporting = {
        "stokes_legacy_numerical_audit": _json_artifact(
            legacy,
            _read_bound_json(
                legacy.path,
                legacy.record(),
                context="Stokes legacy numerical audit",
            ),
        )
    }
    return primary, supporting


def _require_exact_keys(
    record: Mapping[str, Any],
    expected: Sequence[str],
    *,
    context: str,
) -> None:
    if set(record) != set(expected):
        raise RuntimeError(
            f"{context} has the wrong fields: expected {sorted(expected)}, "
            f"observed {sorted(record)}"
        )


def _extrema_record(
    value: object,
    *,
    context: str,
    expected_count: int | None = None,
    minimum_count: int = 1,
    maximum_count: int | None = None,
) -> dict[str, float | int]:
    record = _mapping(value, context=context)
    _require_exact_keys(record, ("count", "minimum", "maximum"), context=context)
    count = _integer(record.get("count"), context=f"{context} count", minimum=1)
    if expected_count is not None and count != expected_count:
        raise RuntimeError(f"{context} count differs from its audited population")
    if count < minimum_count or (maximum_count is not None and count > maximum_count):
        raise RuntimeError(f"{context} count is outside its audited population")
    minimum = _finite_number(record.get("minimum"), context=f"{context} minimum")
    maximum = _finite_number(record.get("maximum"), context=f"{context} maximum")
    if minimum > maximum:
        raise RuntimeError(f"{context} minimum exceeds its maximum")
    return {"count": count, "minimum": minimum, "maximum": maximum}


def _extrema_group(
    value: object,
    metrics: Sequence[str],
    *,
    context: str,
    expected_counts: Mapping[str, int] | None = None,
    minimum_count: int = 1,
    maximum_count: int | None = None,
) -> dict[str, object]:
    group = _mapping(value, context=context)
    _require_exact_keys(group, metrics, context=context)
    return {
        metric: _extrema_record(
            group[metric],
            context=f"{context} {metric}",
            expected_count=(
                None if expected_counts is None else expected_counts[metric]
            ),
            minimum_count=minimum_count,
            maximum_count=maximum_count,
        )
        for metric in metrics
    }


def _completion_audit(
    authenticated: AuthenticatedPostcompletion,
    family: str,
) -> tuple[postcompletion.Artifact, dict[str, Any]]:
    artifact = authenticated.release.audits[family]
    record = _read_bound_json(
        artifact.path,
        artifact.record(),
        context=f"{family} completion audit",
    )
    if (
        record.get("schema") != FAMILY_AUDIT_SCHEMAS[family]
        or record.get("status") != "pass"
    ):
        raise RuntimeError(f"{family} completion audit has the wrong schema or status")
    return artifact, record


def _stokes_extrema_facts(
    authenticated: AuthenticatedPostcompletion,
) -> dict[str, object]:
    binding_artifact, binding = _completion_audit(authenticated, "stokes")
    legacy_artifact = authenticated.release.audits["stokes_legacy"]
    legacy = _read_bound_json(
        legacy_artifact.path,
        legacy_artifact.record(),
        context="Stokes legacy numerical audit",
    )
    if (
        legacy.get("schema") != STOKES_LEGACY_AUDIT_SCHEMA
        or legacy.get("status") != "pass"
    ):
        raise RuntimeError("Stokes legacy audit has the wrong schema or status")
    legacy_reference = _mapping(
        binding.get("legacy_numerical_audit"),
        context="Stokes binding legacy numerical audit",
    )
    reference_path = Path(
        _string(
            legacy_reference.get("path"),
            context="Stokes binding legacy audit path",
        )
    )
    if (
        not reference_path.is_absolute()
        or reference_path.resolve(strict=True) != legacy_artifact.path
        or legacy_reference.get("sha256") != legacy_artifact.sha256
        or legacy_reference.get("schema") != STOKES_LEGACY_AUDIT_SCHEMA
        or legacy_reference.get("status") != "pass"
    ):
        raise RuntimeError("Stokes binding names a different legacy numerical audit")
    if any(
        field in record
        for record in (binding, legacy)
        for field in ("support", "numerical_extrema")
    ):
        raise RuntimeError("Stokes audit schema unexpectedly reports extrema")
    return {
        "reported": False,
        "reason": (
            "The authenticated Stokes completion binding and legacy numerical "
            "audit report no support.extrema or numerical_extrema fields."
        ),
        "source_audits": {
            "completion_binding": _json_artifact(binding_artifact, binding),
            "legacy_numerical_audit": _json_artifact(legacy_artifact, legacy),
        },
    }


def _trajectory_extrema_facts(
    authenticated: AuthenticatedPostcompletion,
    family: str,
) -> dict[str, object]:
    artifact, audit = _completion_audit(authenticated, family)
    accepted = _integer(audit.get("accepted"), context=f"{family} accepted", minimum=1)
    attempted = _integer(
        audit.get("attempted"), context=f"{family} attempted", minimum=accepted
    )
    support = _mapping(audit.get("support"), context=f"{family} support")
    support_counts = {metric: attempted for metric in SUPPORT_EXTREMA_METRICS[family]}
    if family == "tanaka":
        crest_count = _integer(
            audit.get("requested_crests_checked"),
            context="Tanaka requested crests checked",
            minimum=attempted,
        )
        support_counts.update(
            {
                metric: crest_count
                for metric in ("crest_alpha", "physical_crest_amplitude")
            }
        )
    support_extrema = _extrema_group(
        support.get("extrema"),
        SUPPORT_EXTREMA_METRICS[family],
        context=f"{family} support extrema",
        expected_counts=support_counts,
    )
    numerical = _mapping(
        audit.get("numerical_extrema"), context=f"{family} numerical extrema"
    )
    if family == "tanaka":
        _require_exact_keys(
            numerical,
            ("accepted_maximum_stage_residual",),
            context="Tanaka numerical extrema",
        )
        checks = _mapping(audit.get("checks"), context="Tanaka audit checks")
        if checks.get("achieved_per_crest_amplitudes_persisted") is not False:
            raise RuntimeError(
                "Tanaka audit does not certify that achieved per-crest "
                "amplitudes are unavailable"
            )
        return {
            "reported": True,
            "source_audit": _json_artifact(artifact, audit),
            "requested_support_parameter_extrema": support_extrema,
            "numerical_health_extrema": {
                "accepted_maximum_stage_residual": _extrema_record(
                    numerical.get("accepted_maximum_stage_residual"),
                    context="Tanaka accepted maximum stage residual",
                    expected_count=accepted,
                )
            },
            "achieved_per_crest_amplitudes_persisted": False,
        }

    health_metrics = HEALTH_EXTREMA_METRICS[family]
    _require_exact_keys(
        numerical, tuple(health_metrics), context=f"{family} numerical extrema"
    )
    accepted_extrema = _extrema_group(
        numerical.get("accepted"),
        health_metrics["accepted"],
        context=f"{family} accepted numerical extrema",
        expected_counts={metric: accepted for metric in health_metrics["accepted"]},
    )
    attempted_extrema = _extrema_group(
        numerical.get("all_finite_attempted"),
        health_metrics["all_finite_attempted"],
        context=f"{family} all-finite-attempted numerical extrema",
        minimum_count=accepted,
        maximum_count=attempted,
    )
    return {
        "reported": True,
        "source_audit": _json_artifact(artifact, audit),
        "support_parameter_extrema": support_extrema,
        "numerical_health_extrema": {
            "accepted": accepted_extrema,
            "all_finite_attempted": attempted_extrema,
        },
    }


def _family_numerical_extrema(
    authenticated: AuthenticatedPostcompletion,
) -> dict[str, object]:
    return {
        "stokes": _stokes_extrema_facts(authenticated),
        "tanaka": _trajectory_extrema_facts(authenticated, "tanaka"),
        "benjamin_feir": _trajectory_extrema_facts(authenticated, "benjamin_feir"),
        "jonswap_tma": _trajectory_extrema_facts(authenticated, "jonswap_tma"),
    }


def _renderer_facts(authenticated: AuthenticatedPostcompletion) -> dict[str, object]:
    record = authenticated.renderer_record
    renderer_implementation = postcompletion.validate_renderer_implementation(
        record.get("renderer_implementation")
    )
    summary_path = authenticated.root / "worst_cases" / "summary.json"
    summary = postcompletion._artifact(summary_path)
    families = _mapping(record.get("families"), context="renderer families")
    family_records: dict[str, object] = {}
    for family in postcompletion.FAMILY_ORDER:
        family_record = _mapping(families[family], context=f"renderer {family}")
        rankings = _mapping(
            family_record.get("rankings"), context=f"renderer {family} rankings"
        )
        top_cases: dict[str, object] = {}
        for ranking in postcompletion.RENDERER_RANKINGS:
            cases = _sequence(
                rankings.get(ranking), context=f"renderer {family} {ranking}"
            )
            compact_cases: list[dict[str, object]] = []
            for index, raw_case in enumerate(cases):
                case = _mapping(
                    raw_case, context=f"renderer {family} {ranking} case {index}"
                )
                split = _string(
                    case.get("split"), context=f"renderer {family} case split"
                )
                if split not in postcompletion.SPLIT_ORDER:
                    raise ValueError(f"renderer {family} case has unknown split")
                compact_cases.append(
                    {
                        "case_id": _integer(
                            case.get("case_id"),
                            context=f"renderer {family} case ID",
                        ),
                        "split": split,
                        "category": _string(
                            case.get("category"),
                            context=f"renderer {family} case category",
                        ),
                        "depth": _finite_number(
                            case.get("depth"),
                            context=f"renderer {family} case depth",
                        ),
                        "maximum_eta_slope": _finite_number(
                            case.get("maximum_eta_slope"),
                            context=f"renderer {family} eta slope",
                        ),
                        "maximum_gxi_high_band_fraction": _finite_number(
                            case.get("maximum_gxi_high_band_fraction"),
                            context=f"renderer {family} high-band fraction",
                        ),
                        "maximum_thresholded_gxi_sign_changes": _integer(
                            case.get("maximum_thresholded_gxi_sign_changes"),
                            context=f"renderer {family} sign changes",
                        ),
                        "maximum_relative_stored_band_quadratic_energy_drift": (
                            _finite_number(
                                case.get(
                                    "maximum_relative_stored_band_quadratic_energy_drift"
                                ),
                                context=f"renderer {family} energy drift",
                            )
                        ),
                        "combined_empirical_rank": _finite_number(
                            case.get("combined_empirical_rank"),
                            context=f"renderer {family} combined rank",
                        ),
                    }
                )
            top_cases[ranking] = compact_cases
        quantiles = _mapping(
            family_record.get("quantiles"), context=f"renderer {family} quantiles"
        )
        quantile_records: dict[str, dict[str, float]] = {}
        for metric in RENDERER_QUANTILE_METRICS:
            raw_quantiles = quantiles.get(metric)
            metric_quantiles = _mapping(
                raw_quantiles, context=f"renderer {family} {metric} quantiles"
            )
            quantile_records[metric] = {
                quantile: _finite_number(
                    metric_quantiles.get(quantile),
                    context=f"renderer {family} {metric} quantile {quantile}",
                )
                for quantile in RENDERER_QUANTILES
            }
        family_records[family] = {
            "accepted_cases": family_record.get("accepted_cases"),
            "retained_rows": family_record.get("retained_rows"),
            "splits": family_record.get("splits"),
            "quantiles": quantile_records,
            "top_cases": top_cases,
        }
    artifacts = _mapping(record.get("artifacts"), context="renderer artifacts")
    artifact_records: dict[str, object] = {}
    for name, raw_artifact in sorted(artifacts.items()):
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("renderer artifact name must be a plain filename")
        artifact_path = summary_path.parent / name
        if artifact_path.is_symlink():
            raise ValueError(f"renderer artifact must not be a symbolic link: {name}")
        artifact = postcompletion._verify_record_artifact(
            raw_artifact,
            expected_path=artifact_path,
            context=f"renderer artifact {name}",
        )
        artifact_records[name] = artifact.record()
    population = _mapping(record.get("population"), context="renderer population")
    population_record = {
        "sources": _integer(population.get("sources"), context="renderer sources"),
        "accepted_cases": _integer(
            population.get("accepted_cases"), context="renderer accepted cases"
        ),
        "retained_rows": _integer(
            population.get("retained_rows"), context="renderer retained rows"
        ),
    }
    parameters = _mapping(record.get("parameters"), context="renderer parameters")
    if (
        parameters.get("final_paper_dataset_contract_required") is not True
        or parameters.get("morphology_diagnostics_are_release_thresholds") is not False
    ):
        raise RuntimeError("renderer parameter scope differs")
    parameter_values = {
        "final_paper_dataset_contract_required": True,
        "delivered_maximum_wavenumber": _integer(
            parameters.get("delivered_maximum_wavenumber"),
            context="renderer delivered maximum wavenumber",
        ),
        "high_band_minimum_wavenumber": _integer(
            parameters.get("high_band_minimum_wavenumber"),
            context="renderer high-band minimum wavenumber",
        ),
        "cyclic_difference_relative_dead_zone": _finite_number(
            parameters.get("cyclic_difference_relative_dead_zone"),
            context="renderer cyclic-difference dead zone",
        ),
        "morphology_diagnostics_are_release_thresholds": False,
        "rank_one_fixed_axis_gifs": parameters.get("rank_one_fixed_axis_gifs"),
        "gif_maximum_frames": _integer(
            parameters.get("gif_maximum_frames"),
            context="renderer GIF maximum frames",
        ),
        "gif_short_frame_threshold": _integer(
            parameters.get("gif_short_frame_threshold"),
            context="renderer GIF short-frame threshold",
        ),
        "gif_short_fps": _integer(
            parameters.get("gif_short_fps"), context="renderer GIF short fps"
        ),
        "gif_long_fps": _integer(
            parameters.get("gif_long_fps"), context="renderer GIF long fps"
        ),
        "gif_dimensions_pixels": parameters.get("gif_dimensions_pixels"),
        "gif_y_limit_scope": parameters.get("gif_y_limit_scope"),
        "gif_y_limit_padding_fraction": _finite_number(
            parameters.get("gif_y_limit_padding_fraction"),
            context="renderer GIF limit padding",
        ),
        "gif_loop_forever": parameters.get("gif_loop_forever"),
        "gif_descriptive_only": parameters.get("gif_descriptive_only"),
    }
    if any(
        parameter_values[name] is not True
        for name in (
            "rank_one_fixed_axis_gifs",
            "gif_loop_forever",
            "gif_descriptive_only",
        )
    ):
        raise RuntimeError("renderer GIF parameter scope differs")
    parameter_record = {
        name: parameter_values[name] for name in RENDERER_PARAMETER_FIELDS
    }
    definitions = _mapping(record.get("definitions"), context="renderer definitions")
    definition_record = {
        name: _string(definitions.get(name), context=f"renderer {name} definition")
        for name in postcompletion.RENDERER_RANKINGS
    }
    return {
        "summary": _json_artifact(summary, record),
        "authenticated_proof": _authenticated_proof(record, context="renderer"),
        "population": population_record,
        "parameters": parameter_record,
        "definitions": definition_record,
        "renderer_implementation": renderer_implementation,
        "families": family_records,
        "artifacts": artifact_records,
    }


def _order_facts(authenticated: AuthenticatedPostcompletion) -> dict[str, object]:
    record = authenticated.order_record
    diagnostic_implementation = postcompletion.validate_order_implementation(
        record.get("diagnostic_implementation")
    )
    output = postcompletion._artifact(
        authenticated.root / "jonswap_order_convergence.json"
    )
    selection = _mapping(record.get("selection"), context="order selection")
    cases = _sequence(record.get("cases"), context="order cases")
    top_cases: list[dict[str, object]] = []
    for index, raw_case in enumerate(cases):
        case = _mapping(raw_case, context=f"order case {index}")
        stored = _mapping(
            case.get("stored_float32_vs_recomputed_order6"),
            context=f"order case {index} stored comparison",
        )
        six_to_eight = _mapping(
            case.get("order6_to_order8"),
            context=f"order case {index} order6-to-order8",
        )
        transitions = _sequence(
            case.get("successive_order_changes"),
            context=f"order case {index} transitions",
        )
        identity = _mapping(
            case.get("identity"), context=f"order case {index} identity"
        )
        identity_record = {
            name: _integer(identity.get(name), context=f"order case {index} {name}")
            for name in ("case_id", "batch_id", "local_index", "frame_index")
        }
        identity_record.update(
            {
                name: _string(identity.get(name), context=f"order case {index} {name}")
                for name in ("chunk_label", "split", "cell_id", "shard_path")
            }
        )
        transition_maxima = []
        for raw_transition in transitions:
            transition = _mapping(raw_transition, context="order transition")
            transition_maxima.append(
                {
                    "from_order": _integer(
                        transition.get("from_order"), context="order from-order"
                    ),
                    "to_order": _integer(
                        transition.get("to_order"), context="order to-order"
                    ),
                    "maximum_projected_relative_l2_difference": _finite_number(
                        transition.get("maximum_projected_relative_l2_difference"),
                        context="order transition maximum",
                    ),
                }
            )
        top_cases.append(
            {
                "rank_metric": _finite_number(
                    case.get("rank_metric"), context=f"order case {index} rank metric"
                ),
                "identity": {
                    name: identity_record[name] for name in ORDER_IDENTITY_FIELDS
                },
                "depth": _finite_number(
                    case.get("depth"), context=f"order case {index} depth"
                ),
                "stored_float32_vs_recomputed_order6_maximum": _finite_number(
                    stored.get("maximum_relative_l2_difference"),
                    context=f"order case {index} stored maximum",
                ),
                "order6_to_order8_maximum": _finite_number(
                    six_to_eight.get("maximum_projected_relative_l2_difference"),
                    context=f"order case {index} order6-to-order8 maximum",
                ),
                "successive_order_change_maxima": transition_maxima,
            }
        )
    selection_facts = {
        "definition": _string(
            selection.get("definition"), context="order selection definition"
        ),
        "completion_audit_global_maximum_value": _finite_number(
            selection.get("completion_audit_global_maximum_value"),
            context="order audit global maximum",
        ),
        "independent_global_maximum_value": _finite_number(
            selection.get("independent_global_maximum_value"),
            context="order independent global maximum",
        ),
        "maximum_value_absolute_difference": _finite_number(
            selection.get("maximum_value_absolute_difference"),
            context="order maximum absolute difference",
        ),
        "maximum_value_comparison_tolerance": _finite_number(
            selection.get("maximum_value_comparison_tolerance"),
            context="order maximum comparison tolerance",
        ),
        "accepted_trajectories_scanned": _integer(
            selection.get("accepted_trajectories_scanned"),
            context="order accepted trajectories",
        ),
        "stored_rows_scanned": _integer(
            selection.get("stored_rows_scanned"), context="order stored rows"
        ),
        "shards_scanned": _integer(
            selection.get("shards_scanned"), context="order scanned shards"
        ),
        "selected_count": _integer(
            selection.get("selected_count"), context="order selected count"
        ),
    }
    numerical = _mapping(
        record.get("numerical_definition"), context="order numerical definition"
    )
    if numerical.get("zero_output_mean") is not True:
        raise RuntimeError("order numerical zero-mean convention differs")
    numerical_record = {
        "execution_platform": _string(
            numerical.get("execution_platform"), context="order execution platform"
        ),
        "dtype": _string(numerical.get("dtype"), context="order dtype"),
        "nx": _integer(numerical.get("nx"), context="order nx"),
        "length": _finite_number(numerical.get("length"), context="order length"),
        "input_and_output_projection": _string(
            numerical.get("input_and_output_projection"),
            context="order input/output projection",
        ),
        "zero_output_mean": True,
        "pad_factor": _integer(numerical.get("pad_factor"), context="order pad factor"),
        "cumulative_orders": [
            _integer(value, context="order cumulative order")
            for value in _sequence(
                numerical.get("cumulative_orders"),
                context="order cumulative orders",
            )
        ],
        "relative_l2_definition": _string(
            numerical.get("relative_l2_definition"),
            context="order relative-L2 definition",
        ),
    }
    return {
        "output": _json_artifact(output, record),
        "authenticated_proof": _authenticated_proof(record, context="order diagnostic"),
        "diagnostic_implementation": diagnostic_implementation,
        "selection": selection_facts,
        "numerical_definition": {
            name: numerical_record[name] for name in ORDER_NUMERICAL_FIELDS
        },
        "top_cases": top_cases,
        "interpretation_scope": _string(
            record.get("interpretation_scope"), context="order interpretation scope"
        ),
    }


def _training_facts(authenticated: AuthenticatedPostcompletion) -> dict[str, object]:
    record = authenticated.training_record
    training_implementation = postcompletion.validate_training_implementation(
        record.get("training_implementation")
    )
    output = postcompletion._artifact(
        authenticated.root / "training_handoff_audit.json"
    )
    view = _mapping(record.get("dataset_view"), context="training dataset view")
    shards = _sequence(view.get("shards"), context="training dataset shards")
    manifest_artifact = authenticated.release.combined.manifest
    manifest = _read_bound_json(
        manifest_artifact.path,
        manifest_artifact.record(),
        context="combined manifest",
    )
    manifest_shards = _sequence(
        manifest.get("dataset_shards"), context="combined manifest shards"
    )
    shard_count = _integer(view.get("shard_count"), context="training shard count")
    shard_rows = _integer(view.get("shard_rows"), context="training shard rows")
    shard_bytes = _integer(view.get("shard_bytes"), context="training shard bytes")
    if shard_count != len(shards) or shard_count != len(manifest_shards):
        raise RuntimeError("training shard count differs from shard records")
    if shard_rows != sum(
        _integer(
            _mapping(item, context="training shard").get("rows"), context="shard rows"
        )
        for item in shards
    ):
        raise RuntimeError("training shard-row storage total differs")
    if shard_bytes != sum(
        _integer(
            _mapping(item, context="training shard").get("bytes"),
            context="shard bytes",
        )
        for item in shards
    ):
        raise RuntimeError("training shard-byte storage total differs")
    identity_payload: list[dict[str, object]] = []
    for expected_index, raw_item in enumerate(shards):
        item = _mapping(raw_item, context=f"training shard {expected_index}")
        manifest_item = _mapping(
            manifest_shards[expected_index],
            context=f"combined manifest shard {expected_index}",
        )
        index = _integer(item.get("index"), context="training shard index")
        if index != expected_index:
            raise RuntimeError("training shard indices are not canonical")
        path = _string(item.get("path"), context="training shard path")
        if not Path(path).is_absolute():
            raise ValueError("training shard path must be absolute")
        manifest_path_value = _string(
            manifest_item.get("path"), context="combined manifest shard path"
        )
        manifest_path = Path(manifest_path_value).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = manifest_artifact.path.parent / manifest_path
        manifest_path = manifest_path.resolve(strict=True)
        training_path = Path(path).expanduser()
        training_sha256 = _digest(item.get("sha256"), context="training shard SHA-256")
        training_rows = _integer(item.get("rows"), context="training shard rows")
        if (
            training_path.resolve(strict=True) != manifest_path
            or manifest_item.get("sha256") != training_sha256
            or manifest_item.get("n_rows") != training_rows
        ):
            raise RuntimeError(
                f"training shard {expected_index} differs from the combined manifest"
            )
        physical = postcompletion._artifact(
            training_path, expected_sha256=training_sha256
        )
        training_bytes = _integer(item.get("bytes"), context="training shard bytes")
        if physical.bytes != training_bytes:
            raise RuntimeError(
                f"training shard {expected_index} physical byte count differs"
            )
        identity_payload.append(
            {
                "index": index,
                "path": str(physical.path),
                "bytes": training_bytes,
                "sha256": training_sha256,
                "rows": training_rows,
            }
        )
    if view.get("shard_identity_fingerprint") != postcompletion._canonical_sha256(
        identity_payload
    ):
        raise RuntimeError("training shard identity fingerprint differs")
    for name, expected in (
        ("manifest", authenticated.release.combined.manifest),
        ("trajectory_map", authenticated.release.combined.trajectory_map),
    ):
        raw = _mapping(view.get(name), context=f"training {name}")
        if not _same(raw, expected.record()):
            raise RuntimeError(f"training {name} identity differs from combined view")

    normalization = _mapping(
        record.get("training_normalization"), context="training normalization"
    )
    stats = _mapping(normalization.get("stats"), context="training stats")
    selection = _mapping(
        normalization.get("accepted_training_row_selection"),
        context="training row selection",
    )
    full_selection = _mapping(
        normalization.get("full_row_selection"),
        context="training full-row selection",
    )
    split_rows = _mapping(
        full_selection.get("splits"), context="training split rows"
    )
    selection_sha = _digest(selection.get("sha256"), context="training selection SHA")
    split_shas = {
        split: _digest(
            _mapping(split_rows.get(split), context=f"training {split} rows").get(
                "sha256"
            ),
            context=f"training {split} row SHA",
        )
        for split in postcompletion.SPLIT_ORDER
    }
    stats_fingerprint = _digest(
        normalization.get("stats_fingerprint"),
        context="normalization statistics fingerprint",
    )
    if full_selection.get("schema") != "paper_dataset_full_row_selection_v1":
        raise RuntimeError("training full-row selection has the wrong schema")
    extrema_names = (
        "feature_min",
        "feature_max",
        "target_min",
        "target_max",
        "depth_min",
        "depth_max",
        "log_depth_min",
        "log_depth_max",
    )
    scale_names = ("feature_absmax", "target_absmax", "domain_length")
    return {
        "output": _json_artifact(output, record),
        "training_implementation": training_implementation,
        "measured_storage": {
            "combined_shard_count": shard_count,
            "combined_shard_rows": shard_rows,
            "combined_shard_bytes": shard_bytes,
            "combined_shard_identity_fingerprint": view["shard_identity_fingerprint"],
        },
        "normalization": {
            "statistics_fingerprint": stats_fingerprint,
            "training_rows": stats.get("num_examples"),
            "storage_rows": stats.get("storage_num_examples"),
            "extrema": {name: stats.get(name) for name in extrema_names},
            "absolute_maximum_scales": {name: stats.get(name) for name in scale_names},
            "accepted_training_row_selection": {
                "count": selection.get("count"),
                "sha256": selection_sha,
                "seed": selection.get("seed"),
            },
        },
        "row_selection": {
            "schema": full_selection.get("schema"),
            "seed": full_selection.get("seed"),
            "splits": {
                split: {
                    "count": _mapping(
                        split_rows.get(split), context=f"training {split} rows"
                    ).get("count"),
                    "sha256": split_shas[split],
                    "per_family_row_count": _mapping(
                        split_rows.get(split), context=f"training {split} rows"
                    ).get("per_family_row_count"),
                }
                for split in postcompletion.SPLIT_ORDER
            },
            "test_rows_in_train_or_validation": full_selection.get(
                "test_rows_in_train_or_validation"
            ),
        },
    }


def _figure_facts(authenticated: AuthenticatedPostcompletion) -> dict[str, object]:
    record = authenticated.figure_record
    figure_implementation = postcompletion.validate_figure_implementation(
        record.get("figure_implementation")
    )
    stem = authenticated.root / "family_case_examples"
    sidecar = postcompletion._artifact(stem.with_suffix(".json"))
    artifacts = _mapping(record.get("artifacts"), context="figure artifacts")
    pdf = postcompletion._verify_record_artifact(
        artifacts.get("pdf"),
        expected_path=stem.with_suffix(".pdf"),
        context="figure PDF",
    )
    png = postcompletion._verify_record_artifact(
        artifacts.get("png"),
        expected_path=stem.with_suffix(".png"),
        context="figure PNG",
    )
    cases = _sequence(record.get("cases"), context="figure cases")
    selected = []
    for index, raw_case in enumerate(cases):
        case = _mapping(raw_case, context=f"figure case {index}")
        family = _string(case.get("family"), context=f"figure case {index} family")
        if family not in postcompletion.FAMILY_ORDER:
            raise ValueError(f"figure case {index} has an unknown family")
        selected.append(
            {
                "family": family,
                "revision_id": _integer(
                    case.get("revision_id"), context=f"figure {family} revision"
                ),
                "split": _string(case.get("split"), context=f"figure {family} split"),
                "category": _string(
                    case.get("category"), context=f"figure {family} category"
                ),
                "case_id": _integer(
                    case.get("case_id"), context=f"figure {family} case ID"
                ),
                "time": _finite_number(
                    case.get("time"), context=f"figure {family} time"
                ),
                "depth": _finite_number(
                    case.get("depth"), context=f"figure {family} depth"
                ),
            }
        )
    return {
        "sidecar": _json_artifact(sidecar, record),
        "authenticated_proof": _authenticated_proof(
            record, context="family-case figure"
        ),
        "figure_implementation": figure_implementation,
        "pdf": pdf.record(),
        "png": png.record(),
        "selected_cases": selected,
    }


def _combined_view_facts(
    binding: postcompletion.CombinedBinding,
    *,
    training_cases: int,
) -> dict[str, object]:
    cases_by_split = {
        "train": training_cases,
        "validation": 1_024,
        "test": 1_024,
    }
    attempted_by_split = {
        split: sum(
            _integer(
                chunk.get("attempted_count"),
                context=f"c{training_cases:05d} {split} attempted count",
            )
            for chunk in binding.chunks
            if chunk.get("split") == split
        )
        for split in postcompletion.SPLIT_ORDER
    }
    rows_by_split_and_family = _cumulative_rows_by_split_and_family(training_cases)
    accepted_cases = len(postcompletion.FAMILY_ORDER) * sum(cases_by_split.values())
    attempted_cases = sum(attempted_by_split.values())
    rejected_cases = attempted_cases - accepted_cases
    return {
        "training_cases_per_family": training_cases,
        "accepted_cases_per_family_by_split": cases_by_split,
        "accepted_cases": accepted_cases,
        "attempted_cases_by_split": attempted_by_split,
        "attempted_cases": attempted_cases,
        "rejected_cases": rejected_cases,
        "rejection_rate": _rejection_rate(
            attempted=attempted_cases,
            rejected=rejected_cases,
            context=f"c{training_cases:05d} combined view",
        ),
        "retained_rows": sum(
            sum(family_rows.values())
            for family_rows in rows_by_split_and_family.values()
        ),
        "summary": _json_artifact(
            binding.summary,
            _read_bound_json(
                binding.summary.path,
                binding.summary.record(),
                context=f"c{training_cases:05d} combined summary",
            ),
        ),
        "manifest": binding.manifest.record(),
        "trajectory_map": binding.trajectory_map.record(),
        "source_count": len(binding.chunks),
    }


def build_record(authenticated: AuthenticatedPostcompletion) -> dict[str, object]:
    """Build a deterministic strict-JSON document handoff."""

    family_counts, totals = _family_counts(authenticated)
    primary_audits, supporting_audits = _audit_records(authenticated)
    release = authenticated.release
    record = {
        "schema": SCHEMA,
        "status": "complete",
        "purpose": "paper_dataset_document_update_handoff",
        "read_only_source_audit": True,
        "postcompletion": {
            "root": str(authenticated.root),
            "status": _json_artifact(authenticated.status, authenticated.status_record),
            "release_identity": _json_artifact(
                authenticated.release_identity,
                authenticated.identity_record,
                require_status=False,
            ),
            "release_identity_fingerprint": postcompletion._canonical_sha256(
                authenticated.identity_record
            ),
        },
        "validated_manuscript_evidence": {
            "jonswap_tma_revision4_prebulk_adoption_gate": {
                "artifact": _json_artifact(
                    authenticated.jonswap_adoption_gate,
                    authenticated.jonswap_adoption_gate_record,
                ),
                "decision_scope": "revision_4_constructor_prebulk_adoption_only",
                "applies_to_final_bulk_release": False,
                "passed": True,
                "observed": {
                    **JONSWAP_ADOPTION_GATE_COUNTS,
                    "rejection_rate": (
                        JONSWAP_ADOPTION_GATE_COUNTS["rejected"]
                        / JONSWAP_ADOPTION_GATE_COUNTS["attempted"]
                    ),
                },
                "predeclared_gate": dict(JONSWAP_ADOPTION_GATE_CRITERION),
            }
        },
        "release": {
            "family_revisions": dict(postcompletion.REVISION_BY_FAMILY),
            "population_conditioning": {
                **RELEASE_POPULATION_CONDITIONING,
                "family_acceptance_events": dict(FAMILY_ACCEPTANCE_EVENTS),
            },
            "family_counts": family_counts,
            "totals": totals,
            "family_audits": primary_audits,
            "supporting_audits": supporting_audits,
            "family_numerical_extrema": _family_numerical_extrema(authenticated),
            "combined_view": _combined_view_facts(
                release.combined,
                training_cases=CUMULATIVE_TRAINING_CASES[-1],
            ),
            "cumulative_combined_views": {
                f"c{training_cases:05d}": _combined_view_facts(
                    authenticated.cumulative_combined_views[training_cases],
                    training_cases=training_cases,
                )
                for training_cases in CUMULATIVE_TRAINING_CASES
            },
        },
        "validated_stage_outputs": {
            "worst_case_renderer": _renderer_facts(authenticated),
            "jonswap_order_diagnostic": _order_facts(authenticated),
            "training_handoff": _training_facts(authenticated),
            "family_case_figure": _figure_facts(authenticated),
        },
    }
    json.dumps(record, allow_nan=False)
    return record


def _authentication_fingerprint(
    authenticated: AuthenticatedPostcompletion,
) -> str:
    """Fingerprint every release and stage record used by the handoff."""

    return postcompletion._canonical_sha256(
        {
            "root": str(authenticated.root),
            "status": authenticated.status.record(),
            "release_identity": authenticated.release_identity.record(),
            "jonswap_adoption_gate": authenticated.jonswap_adoption_gate.record(),
            "release": authenticated.release.identity_record(),
            "status_record": authenticated.status_record,
            "identity_record": authenticated.identity_record,
            "jonswap_adoption_gate_record": (
                authenticated.jonswap_adoption_gate_record
            ),
            "cumulative_combined_views": {
                f"c{training_cases:05d}": {
                    "summary": view.summary.record(),
                    "manifest": view.manifest.record(),
                    "trajectory_map": view.trajectory_map.record(),
                    "chunks": list(view.chunks),
                }
                for training_cases, view in sorted(
                    authenticated.cumulative_combined_views.items()
                )
            },
            "renderer_record": authenticated.renderer_record,
            "order_record": authenticated.order_record,
            "training_record": authenticated.training_record,
            "figure_record": authenticated.figure_record,
        }
    )


_CUMULATIVE_BATCH_RECORD_KEYS = (
    "proposal_path",
    "proposal_sha256",
    "result_path",
    "result_sha256",
    "shard_index",
    "configuration_fingerprint",
    "family_id",
    "revision_id",
    "split_id",
    "batch_id",
    "n_attempted_trajectories",
    "n_accepted_trajectories",
    "n_rows",
)
_CUMULATIVE_SHARD_RECORD_KEYS = (
    "path",
    "sha256",
    "n_rows",
    "batch_index",
    "configuration_fingerprint",
)


def _resolve_cumulative_manifest_artifact(
    manifest_parent: Path,
    value: object,
    *,
    allowed_roots: Sequence[Path],
    context: str,
) -> Path:
    """Resolve one canonical source reference without permitting substitution."""

    path_text = _string(value, context=f"{context} path")
    candidate = Path(path_text).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_parent / candidate
    normalized = Path(os.path.abspath(candidate))
    if normalized.is_symlink():
        raise ValueError(f"{context} must not be a symbolic link: {normalized}")
    resolved = normalized.resolve(strict=True)
    if resolved != normalized:
        raise ValueError(
            f"{context} uses symbolic-link or path substitution: {normalized}"
        )
    roots = tuple(root.resolve(strict=True) for root in allowed_roots)
    if not any(resolved.is_relative_to(root) for root in roots):
        raise ValueError(f"{context} escapes authenticated source roots: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{context} must be a regular file: {resolved}")
    return resolved


def _verify_authenticated_artifacts_current(
    authenticated: AuthenticatedPostcompletion,
) -> None:
    """Rehash every bound release and document artifact before publication."""

    observed_by_path: dict[Path, postcompletion.Artifact] = {}
    terminal_parents: dict[Path, tuple[postcompletion.Artifact, str]] = {}
    release_provenance_groups = postcompletion._release_terminal_artifact_groups(
        dependency=authenticated.release.dependency,
        generation_sources=authenticated.release.generation_sources,
        historical_source_snapshots=(authenticated.release.historical_source_snapshots),
        repository_root=None,
    )
    training_implementation = postcompletion.validate_training_implementation(
        authenticated.training_record.get("training_implementation")
    )
    renderer_implementation = postcompletion.validate_renderer_implementation(
        authenticated.renderer_record.get("renderer_implementation")
    )
    order_implementation = postcompletion.validate_order_implementation(
        authenticated.order_record.get("diagnostic_implementation")
    )
    figure_implementation = postcompletion.validate_figure_implementation(
        authenticated.figure_record.get("figure_implementation")
    )

    def retain_terminal(
        artifact: postcompletion.Artifact,
        *,
        context: str,
    ) -> None:
        previous = terminal_parents.get(artifact.path)
        if previous is not None and previous[0] != artifact:
            raise RuntimeError(f"{context} has conflicting terminal identities")
        terminal_parents[artifact.path] = (artifact, context)

    def remember(artifact: postcompletion.Artifact, *, context: str) -> None:
        previous = observed_by_path.get(artifact.path)
        if previous is not None and previous != artifact:
            raise RuntimeError(f"{context} has conflicting authenticated identities")
        observed_by_path[artifact.path] = artifact

    def verify_current_path(
        path: Path,
        digest_value: object,
        *,
        context: str,
        expected_bytes: int | None = None,
        force_rehash: bool = False,
    ) -> postcompletion.Artifact:
        expected_digest = _digest(digest_value, context=f"{context} SHA-256")
        requested = path.expanduser()
        if requested.is_symlink():
            raise ValueError(f"{context} must not be a symbolic link: {requested}")
        resolved = requested.resolve(strict=True)
        previous = observed_by_path.get(resolved)
        if previous is not None and not force_rehash:
            if previous.sha256 != expected_digest:
                raise RuntimeError(
                    f"{context} has conflicting authenticated SHA-256 identities"
                )
            if expected_bytes is not None and previous.bytes != expected_bytes:
                raise RuntimeError(f"{context} byte count differs")
            return previous
        observed = postcompletion._artifact(
            requested,
            expected_sha256=expected_digest,
        )
        if expected_bytes is not None and observed.bytes != expected_bytes:
            raise RuntimeError(f"{context} byte count differs")
        remember(observed, context=context)
        return observed

    def verify_artifact(
        expected: postcompletion.Artifact,
        *,
        context: str,
        force_rehash: bool = False,
    ) -> None:
        observed = verify_current_path(
            expected.path,
            expected.sha256,
            context=context,
            expected_bytes=expected.bytes,
            force_rehash=force_rehash,
        )
        if observed != expected:
            raise RuntimeError(f"{context} byte identity differs")

    def verify_record(
        value: object,
        *,
        context: str,
    ) -> postcompletion.Artifact:
        record = _mapping(value, context=context)
        path_value = _string(record.get("path"), context=f"{context} path")
        path = Path(path_value)
        if not path.is_absolute():
            raise ValueError(f"{context} path must be absolute")
        return verify_current_path(
            path,
            record.get("sha256"),
            context=context,
            expected_bytes=_integer(record.get("bytes"), context=f"{context} bytes"),
        )

    def verify_path_digest(
        path_value: object,
        digest_value: object,
        *,
        context: str,
    ) -> postcompletion.Artifact:
        path_text = _string(path_value, context=f"{context} path")
        path = Path(path_text)
        if not path.is_absolute():
            raise ValueError(f"{context} path must be absolute")
        return verify_current_path(path, digest_value, context=context)

    def verify_embedded_records(value: object, *, context: str) -> None:
        if isinstance(value, Mapping):
            implementation_validators = {
                postcompletion.TRAINING_IMPLEMENTATION_SCHEMA: (
                    postcompletion.validate_training_implementation
                ),
                postcompletion.RENDERER_IMPLEMENTATION_SCHEMA: (
                    postcompletion.validate_renderer_implementation
                ),
                postcompletion.ORDER_IMPLEMENTATION_SCHEMA: (
                    postcompletion.validate_order_implementation
                ),
                postcompletion.FIGURE_IMPLEMENTATION_SCHEMA: (
                    postcompletion.validate_figure_implementation
                ),
            }
            validator = implementation_validators.get(value.get("schema"))
            if validator is not None:
                validated = validator(value)
                if not _same(value, validated):
                    raise RuntimeError(f"{context} implementation record differs")
                return
            if {"path", "bytes", "sha256"}.issubset(value):
                verify_record(value, context=context)
                return
            for name, item in value.items():
                verify_embedded_records(item, context=f"{context}.{name}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                verify_embedded_records(item, context=f"{context}[{index}]")

    for name, artifact in (
        ("postcompletion status", authenticated.status),
        ("release identity", authenticated.release_identity),
        ("JONSWAP adoption gate", authenticated.jonswap_adoption_gate),
        ("final combined summary", authenticated.release.combined.summary),
        ("final combined manifest", authenticated.release.combined.manifest),
        (
            "final combined trajectory map",
            authenticated.release.combined.trajectory_map,
        ),
    ):
        verify_artifact(artifact, context=name)
        retain_terminal(artifact, context=name)
    for name, artifact in sorted(authenticated.release.audits.items()):
        verify_artifact(artifact, context=f"release audit {name}")
        retain_terminal(artifact, context=f"release audit {name}")
    for training_cases, view in sorted(authenticated.cumulative_combined_views.items()):
        tag = f"c{training_cases:05d}"
        for label, artifact in (
            ("summary", view.summary),
            ("manifest", view.manifest),
            ("trajectory map", view.trajectory_map),
        ):
            context = f"{tag} combined {label}"
            verify_artifact(artifact, context=context)
            retain_terminal(artifact, context=context)

        manifest = _read_bound_json(
            view.manifest.path,
            view.manifest.record(),
            context=f"{tag} combined manifest",
        )
        allowed_roots = (
            view.manifest.path.parent,
            *(
                Path(
                    _string(
                        chunk.get("summary_path"),
                        context=f"{tag} source-summary path",
                    )
                ).parent
                for chunk in view.chunks
            ),
        )
        batches = _sequence(
            manifest.get("dataset_batches"),
            context=f"{tag} combined manifest dataset batches",
        )
        for index, raw_batch in enumerate(batches):
            context = f"{tag} combined manifest dataset batch {index}"
            batch = _mapping(raw_batch, context=context)
            _require_exact_keys(
                batch,
                _CUMULATIVE_BATCH_RECORD_KEYS,
                context=context,
            )
            for label in ("proposal", "result"):
                artifact_context = f"{context} {label}"
                path = _resolve_cumulative_manifest_artifact(
                    view.manifest.path.parent,
                    batch.get(f"{label}_path"),
                    allowed_roots=allowed_roots,
                    context=artifact_context,
                )
                verify_current_path(
                    path,
                    batch.get(f"{label}_sha256"),
                    context=artifact_context,
                )
        shards = _sequence(
            manifest.get("dataset_shards"),
            context=f"{tag} combined manifest dataset shards",
        )
        for index, raw_shard in enumerate(shards):
            context = f"{tag} combined manifest dataset shard {index}"
            shard = _mapping(raw_shard, context=context)
            _require_exact_keys(
                shard,
                _CUMULATIVE_SHARD_RECORD_KEYS,
                context=context,
            )
            path = _resolve_cumulative_manifest_artifact(
                view.manifest.path.parent,
                shard.get("path"),
                allowed_roots=allowed_roots,
                context=context,
            )
            verify_current_path(
                path,
                shard.get("sha256"),
                context=context,
            )
    for index, chunk in enumerate(authenticated.release.combined.chunks):
        source_summary = verify_path_digest(
            chunk.get("summary_path"),
            chunk.get("summary_sha256"),
            context=f"release source summary {index}",
        )
        retain_terminal(
            source_summary,
            context=f"release source summary {index}",
        )

    source_parent_paths: set[Path] = set()
    for family in PRIMARY_AUDITS:
        artifact = authenticated.release.audits[family]
        audit = _read_bound_json(
            artifact.path,
            artifact.record(),
            context=f"{family} completion audit",
        )
        for index, raw_chunk in enumerate(
            _sequence(audit.get("chunks"), context=f"{family} audit chunks")
        ):
            chunk = _mapping(raw_chunk, context=f"{family} audit chunk {index}")
            for label in ("summary", "manifest", "trajectory_map"):
                context = f"{family} audit chunk {index} {label}"
                source_parent = verify_path_digest(
                    chunk.get(f"{label}_path"),
                    chunk.get(f"{label}_sha256"),
                    context=context,
                )
                retain_terminal(source_parent, context=context)
                source_parent_paths.add(source_parent.path)
    if len(source_parent_paths) != postcompletion.FINAL_SOURCE_COUNT * 3:
        raise RuntimeError("release source-parent terminal set is incomplete")

    verify_embedded_records(
        authenticated.status_record,
        context="postcompletion status record",
    )
    verify_embedded_records(
        authenticated.renderer_record,
        context="renderer record",
    )
    verify_embedded_records(
        authenticated.order_record,
        context="order record",
    )
    for case_index, raw_case in enumerate(
        _sequence(authenticated.order_record.get("cases"), context="order cases")
    ):
        sources = _mapping(
            _mapping(raw_case, context=f"order case {case_index}").get("sources"),
            context=f"order case {case_index} sources",
        )
        for name in (
            "summary",
            "manifest",
            "trajectory_map",
            "proposal",
            "result",
            "shard",
        ):
            context = f"order case {case_index} {name} proof"
            artifact = verify_path_digest(
                sources.get(f"{name}_path"),
                sources.get(f"{name}_sha256"),
                context=context,
            )
            retain_terminal(artifact, context=context)
    verify_embedded_records(
        {
            name: value
            for name, value in authenticated.training_record.items()
            if name != "training_implementation"
        },
        context="training record",
    )
    repository_root = postcompletion._require_real_directory(
        REPOSITORY_ROOT,
        context="diagnostic implementation repository root",
    )
    implementation_code_paths: dict[str, tuple[Path, object, int, str]] = {}
    for label, implementation in (
        ("training", training_implementation),
        ("renderer", renderer_implementation),
        ("order diagnostic", order_implementation),
        ("figure", figure_implementation),
    ):
        for index, raw_file in enumerate(
            _sequence(
                implementation.get("files"),
                context=f"{label} implementation files",
            )
        ):
            context = f"{label} implementation file {index}"
            file_record = _mapping(raw_file, context=context)
            relative = _string(file_record.get("path"), context=f"{context} path")
            path = postcompletion._source_file_inside(
                repository_root,
                relative,
                context=context,
            )
            byte_count = _integer(file_record.get("bytes"), context=f"{context} bytes")
            candidate = (path, file_record.get("sha256"), byte_count, context)
            previous = implementation_code_paths.get(relative)
            if previous is not None and previous[:3] != candidate[:3]:
                raise RuntimeError(
                    f"{context} conflicts with another implementation identity"
                )
            implementation_code_paths[relative] = candidate
            verify_current_path(
                path,
                file_record.get("sha256"),
                context=context,
                expected_bytes=byte_count,
            )
            retain_terminal(
                postcompletion.Artifact(
                    path=path,
                    bytes=byte_count,
                    sha256=_digest(
                        file_record.get("sha256"),
                        context=f"{context} SHA-256",
                    ),
                ),
                context=context,
            )
    verify_embedded_records(
        authenticated.figure_record,
        context="figure record",
    )
    for case_index, raw_case in enumerate(
        _sequence(authenticated.figure_record.get("cases"), context="figure cases")
    ):
        case = _mapping(raw_case, context=f"figure case {case_index}")
        for name in (
            "source_summary",
            "source_manifest",
            "source_trajectory_map",
            "selected_shard",
        ):
            context = f"figure case {case_index} {name} proof"
            artifact = verify_record(case.get(name), context=context)
            retain_terminal(artifact, context=context)

    stages = _mapping(
        authenticated.status_record.get("stages"),
        context="postcompletion stage records",
    )

    def retain_stage_record(stage: str, name: str) -> None:
        stage_record = _mapping(stages.get(stage), context=f"{stage} stage")
        artifacts = _mapping(
            stage_record.get("artifacts"),
            context=f"{stage} stage artifacts",
        )
        context = f"{stage} {name}"
        retain_terminal(
            verify_record(artifacts.get(name), context=context),
            context=context,
        )

    retain_stage_record("final_renderer", "summary")
    renderer_artifacts = _mapping(
        authenticated.renderer_record.get("artifacts"),
        context="renderer publishable artifacts",
    )
    for name, record in sorted(renderer_artifacts.items()):
        context = f"renderer publishable artifact {name}"
        retain_terminal(
            verify_record(record, context=context),
            context=context,
        )
    retain_stage_record("jonswap_order_diagnostic", "output")
    retain_stage_record("training_handoff_audit", "output")
    for name in ("sidecar", "pdf", "png"):
        retain_stage_record("family_case_figure", name)

    # Close the manifest-traversal window: a parent view changed while its
    # transitive children were being hashed must not survive to publication.
    for training_cases, view in sorted(authenticated.cumulative_combined_views.items()):
        tag = f"c{training_cases:05d}"
        verify_artifact(
            view.summary,
            context=f"final {tag} combined summary currentness",
            force_rehash=True,
        )
        verify_artifact(
            view.manifest,
            context=f"final {tag} combined manifest currentness",
            force_rehash=True,
        )
        verify_artifact(
            view.trajectory_map,
            context=f"final {tag} combined trajectory-map currentness",
            force_rehash=True,
        )
    for index, (relative, values) in enumerate(
        sorted(implementation_code_paths.items())
    ):
        path, digest, byte_count, source_context = values
        current_path = postcompletion._source_file_inside(
            repository_root,
            relative,
            context=f"final diagnostic implementation file {index}",
        )
        if current_path != path:
            raise RuntimeError(
                f"final diagnostic implementation file {index} path changed"
            )
        verify_current_path(
            current_path,
            digest,
            context=(
                f"final diagnostic implementation file {index} currentness "
                f"({source_context})"
            ),
            expected_bytes=byte_count,
            force_rehash=True,
        )
    postcompletion._terminal_reauthenticate_artifact_groups(release_provenance_groups)

    # This is the absolute-final cheap publication-parent sweep. Cooperative
    # stability is still required after the last hash; finite sequential
    # reauthentication cannot eliminate a hostile post-sweep mutation window.
    for expected, context in terminal_parents.values():
        verify_artifact(
            expected,
            context=f"absolute-final {context}",
            force_rehash=True,
        )


def _write_immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    destination = path.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise FileExistsError(
                f"handoff output is not a regular file: {destination}"
            )
        existing = postcompletion._strict_json(destination)
        if not _same(existing, payload):
            raise FileExistsError("refusing to overwrite unrelated handoff output")
        return
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise FileNotFoundError(f"handoff output parent must already exist: {parent}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            existing = postcompletion._strict_json(destination)
            if not _same(existing, payload):
                raise FileExistsError(
                    "handoff output appeared concurrently with different content"
                ) from None
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def build_document_handoff(
    source: Path,
    *,
    output: Path | None = None,
    release_authenticator: ReleaseAuthenticator = postcompletion.authenticate_release,
) -> Path:
    """Authenticate ``source`` and atomically create its document handoff."""

    root = _resolve_release_root(source)
    with (root / "runner.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        authenticated = authenticate_postcompletion(
            source, release_authenticator=release_authenticator
        )
        if authenticated.root != root:
            raise RuntimeError("authenticated postcompletion root changed")
        destination = output or authenticated.root / DEFAULT_NAME
        authentication_fingerprint = _authentication_fingerprint(authenticated)
        record = build_record(authenticated)
        # This second full authentication rereads and semantically validates the
        # fixed adoption-gate JSON, in addition to rehashing every bound artifact.
        reauthenticated = authenticate_postcompletion(
            source, release_authenticator=release_authenticator
        )
        if _authentication_fingerprint(reauthenticated) != authentication_fingerprint:
            raise RuntimeError("postcompletion evidence changed before handoff commit")
        _verify_authenticated_artifacts_current(reauthenticated)
        _write_immutable_json(destination, record)
    return destination.expanduser().absolute()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "postcompletion",
        type=Path,
        help=(
            "Completed SHA-keyed postcompletion directory, or its adjacent "
            "status.json/release_identity.json."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"New immutable JSON path; defaults to <postcompletion>/{DEFAULT_NAME}.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = build_document_handoff(args.postcompletion, output=args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
